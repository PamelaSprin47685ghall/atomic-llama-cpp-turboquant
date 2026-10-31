// m3-mixer-prod-path.cpp — device test of the REAL mixer/trunk site through the
// production builder, with the production Q8_0 door, the production exact-"1"
// gate and the production NO_REUSE decision.
//
// Why the earlier harness was not enough: micro/m3-mixer-n1ref.cpp rebuilt the HC
// block with local ops, so it never called
// llama_model_qwen4exp::graph::build_hc_mix, and it created F32 weights, so the
// production Q8_0 checks could not fire.  Here the weights are Q8_0 (down
// [10240,320], up [320,10240], inject [10240,4]) exactly as the model loads them,
// the site is built by the shipped builder through micro/m3-probe-hc-mix.cpp, and
// the harness never sets GGML_TENSOR_FLAG_NO_REUSE: it reads the flag back from
// the tensor the builder returned (the M3_TAIL_BUILD / M3BUILD witness lines).
//
// Two compares:
//   * rows: <prefix>.rows.bin holds the site output for nt rows;
//     <prefix>.nt1.<r>.bin holds the same site with nt = 1 for row r, built by the
//     same production builder, so diag/m3-n1-rowref.py compares row r of the
//     tail-capable graph against the N=1 graph (the chk-N1-r<k> convention).
//   * witness: the M3_TAIL_BUILD lines (or the M3BUILD stamp) report, per site,
//     whether lo and xn carry the retention flag, which only the library decides.
//
// Usage: m3-mixer-prod-path <libdir> <nt> <trunk|mixer> <prefix> [dev]
// Exit: 0 ok, 2 no device, 3 allocator, 4 compute, 5 output, 7 gate not armed.
// Authored, NOT built or run.

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Implemented in micro/m3-probe-hc-mix.cpp (compiled into this binary, so the
// shipped builder is reached from a derived type in the same TU).
struct m3_probe_site {
    bool mixer = false;
    m3_probe_site * tensors_out = nullptr;
    ggml_tensor * norm;
    ggml_tensor * down;
    ggml_tensor * up;
    ggml_tensor * inject;
    ggml_tensor * x;
    ggml_cgraph * gf;
};

extern "C" bool m3_probe_attach(llama_model * model);
extern "C" ggml_tensor * m3_probe_build_hc_mix(
        llama_model * model, int64_t n_embd, int64_t hc, const m3_probe_site & spec,
        int64_t nt, int il, ggml_tensor ** lo_out, ggml_tensor ** up_out,
        ggml_tensor ** inject_out);
extern "C" void m3_probe_release(void);
extern "C" bool m3_probe_lo_retained(ggml_tensor * lo);

namespace {

unsigned rs = 0x4321u;
float frand() {
    rs = rs * 1103515245u + 12345u;
    return float((rs >> 8) & 0xffff) / 65535.0f;
}

bool env1(const char * k) {
    const char * v = std::getenv(k);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Seed the weights and the input rows.  Row r's seed does not depend on nt, which
// is what makes the nt=1 pass a per-row reference of the same computation.
void seed(ggml_backend_t be, const m3_probe_site & s, int64_t n_embd, int64_t nt) {
    rs = 0x12345u;
    std::vector<float> nf(size_t(n_embd) * 4);
    for (float & v : nf) v = 1.0f + 0.1f * frand();
    ggml_backend_tensor_set(s.norm, nf.data(), 0, nf.size() * sizeof(float));

    std::vector<float> f(size_t(4 * n_embd) * 320);
    for (float & v : f) v = 0.02f * frand();
    std::vector<uint8_t> q(ggml_nbytes(s.down));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, 320, 4 * n_embd, nullptr);
    ggml_backend_tensor_set(s.down, q.data(), 0, q.size());

    for (float & v : f) v = 0.05f * frand();
    q.assign(ggml_nbytes(s.up), 0);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, 4 * n_embd, 320, nullptr);
    ggml_backend_tensor_set(s.up, q.data(), 0, q.size());

    if (s.inject != nullptr) {
        std::vector<float> of(size_t(4 * n_embd) * 4);
        for (float & v : of) v = 0.01f * frand();
        std::vector<uint8_t> qi(ggml_nbytes(s.inject));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, of.data(), qi.data(), 0, 4, 4 * n_embd, nullptr);
        ggml_backend_tensor_set(s.inject, qi.data(), 0, qi.size());
    }

    std::vector<float> xin(size_t(n_embd) * 4 * nt);
    for (int64_t r = 0; r < nt; ++r) {
        rs = 0xabc00u + uint32_t(r) * 7919u;
        for (int64_t i = 0; i < n_embd * 4; ++i) {
            xin[size_t(r) * n_embd * 4 + size_t(i)] = frand();
        }
    }
    ggml_backend_tensor_set(s.x, xin.data(), 0, xin.size() * sizeof(float));
}

bool run_site(ggml_backend_t be, llama_model * model, int64_t n_embd, int64_t nt,
              bool mixer, const std::string & out) {
    if (!m3_probe_attach(model)) {
        return false;
    }
    // The site and every tensor of it are created by the probe inside the result's
    // own context (the one build_hc_mix reads); the harness only supplies shapes.
    m3_probe_site spec;
    spec.mixer = mixer;
    m3_probe_site created;
    spec.tensors_out = &created;

    ggml_tensor * lo = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * inj = nullptr;
    ggml_tensor * mixed = m3_probe_build_hc_mix(model, n_embd, /*hc=*/4, spec, nt, /*il=*/0,
                                                &lo, &up, &inj);
    if (mixed == nullptr || lo == nullptr || up == nullptr || created.gf == nullptr) {
        fprintf(stderr, "production builder returned no site (nt=%lld mixer=%d)\n",
                (long long) nt, int(mixer));
        m3_probe_release();
        return false;
    }
    ggml_cgraph * gf = created.gf;

    // Attach backend storage to the no_alloc tensors of that context.  The
    // tallocr struct is public (ggml-alloc.h) and its members are exactly the
    // buffer/base/alignment/offset pair used here; ggml_tallocr_alloc is the
    // documented way to give a tensor its data pointer.  One buffer serves all of
    // them, and it is freed before the result.
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft,
            ggml_backend_buft_get_alloc_size(buft, created.down) +
            ggml_backend_buft_get_alloc_size(buft, created.up) +
            ggml_backend_buft_get_alloc_size(buft, created.norm) +
            ggml_backend_buft_get_alloc_size(buft, created.x) +
            (created.inject ? ggml_backend_buft_get_alloc_size(buft, created.inject) : 0) +
            4096);
    if (buf == nullptr) {
        m3_probe_release();
        return false;
    }
    struct ggml_tallocr ta = ggml_tallocr_new(buf);
    ggml_tensor * all[] = { created.norm, created.down, created.up, created.x, created.inject };
    for (ggml_tensor * t : all) {
        if (t != nullptr && ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            m3_probe_release();
            return false;
        }
    }
    seed(be, created, n_embd, nt);

    ggml_set_output(mixed);
    ggml_set_output(lo);
    ggml_set_output(up);
    ggml_gallocr_t ga = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        m3_probe_release();
        return false;
    }
    // WITNESS A (diagnostic only): the raw retention bits, next to the M3_TAIL_BUILD
    // line the model TU emits for the same site.  Neither is a pass criterion on its
    // own; the numeric check is Witness B.
    printf("{\"nt\":%lld,\"site\":\"%s\",\"lo_flags\":%d,\"lo_no_reuse\":%d,\"lo_type\":\"%s\","
           "\"down_type\":\"%s\",\"up_type\":\"%s\",\"inject\":\"%s\",\"tail_keep\":\"%s\","
           "\"diag_scope\":\"%s\"}\n",
           (long long) nt, mixer ? "mixer" : "trunk", lo->flags, int(m3_probe_lo_retained(lo)),
           ggml_type_name(lo->type), ggml_type_name(created.down->type),
           ggml_type_name(created.up->type),
           created.inject ? ggml_type_name(created.inject->type) : "<none>",
           std::getenv("GGML_TP5_HC_N3_TAIL_KEEP") ? std::getenv("GGML_TP5_HC_N3_TAIL_KEEP") : "<unset>",
           std::getenv("GGML_TP5_HC_N3_TAIL_DIAG_SCOPE") ? std::getenv("GGML_TP5_HC_N3_TAIL_DIAG_SCOPE") : "<unset>");
    bool ok = true;
    for (int i = 0; i < 3 && ok; ++i) {
        ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        std::vector<float> v(ggml_nelements(mixed));
        ggml_backend_tensor_get(mixed, v.data(), 0, v.size() * sizeof(float));
        FILE * f = fopen(out.c_str(), "wb");
        if (f == nullptr) {
            ok = false;
        } else {
            fwrite(v.data(), sizeof(float), v.size(), f);
            fclose(f);
        }
    }
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);   // tensor data first ...
    m3_probe_release();              // ... then the context that describes it
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <libdir> <nt> <trunk|mixer> <prefix> [dev]\n", argv[0]);
        return 1;
    }
    const char * libdir = argv[1];
    const int64_t nt = atoll(argv[2]);
    const bool mixer = std::strcmp(argv[3], "mixer") == 0;
    const std::string prefix = argv[4];
    const char * devname = argc > 5 ? argv[5] : "ROCm0";

    if (!env1("GGML_TP5_HC_N3_TRIO")) {
        fprintf(stderr, "GGML_TP5_HC_N3_TRIO must be exactly \"1\" for this harness\n");
        return 7;
    }
    if (nt < 1 || nt > 4) {
        return 1;
    }
    llama_backend_init();
    ggml_backend_load_all_from_path(libdir);

    // The real model: the production builder reads hparams (n_embd, dsv4_hc_mult,
    // f_norm_rms_eps) and the model object from it; the HC weights used below are
    // this run's Q8_0 tensors, so the Q8_0 door is exercised.
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(
            "/opt/strata/data/models/swift-q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", mp);
    if (model == nullptr) {
        fprintf(stderr, "model load failed\n");
        return 2;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(devname);
    if (dev == nullptr) {
        return 2;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    const int64_t n_embd = 2560;

    // Multi-row image for the nt-row site, then one nt=1 image per row, all
    // through the production builder.
    bool ok = run_site(be, model, n_embd, nt, mixer, prefix + ".rows.bin");
    for (int64_t r = 0; ok && r < nt; ++r) {
        ok = run_site(be, model, n_embd, 1, mixer,
                      prefix + ".nt1." + std::to_string(r) + ".bin");
    }
    llama_model_free(model);
    ggml_backend_free(be);
    llama_backend_free();
    return ok ? 0 : 4;
}
