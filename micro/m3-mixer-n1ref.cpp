// m3-mixer-n1ref.cpp — mixer-shape HC block, N-row vs per-row N=1 reference.
//
// The 2026-10-27 M3 micro (mtp-hc-n3-trio-m3-20261027/src/m3-micro.cpp) builds a
// trunk site only: it always calls ggml_mul_mat(w_inject, xn) and never reaches
// the output mixer / MTP head mixer shapes. A mixer site is one where the
// caller passes inject == nullptr, so after the HC mean there is no inject
// projection and the HC chain's last consumer of xn is the gated tail. The tail
// storage fix changes lifetime exactly there (lo and xn are both retained), and
// that shape has had no per-row reference — this harness supplies one.
//
// It is a copy of the trunk micro's block with two differences:
//   1. MIXER=1 drops the inject path, reshapes the residual to the trunk's
//      [n_embd, hc] view and appends the output-scale projection the real
//      mixer has after the HC chain (output matrix F32 [E, VOCAB_SLICE]);
//   2. every output (lo, mixed, logits) is written for the requested nt rows
//      and, with NT1=1, the same site is rebuilt with nt=1 and a per-row seed
//      (M3_ROW0..), which is the established "row r reproduced with a single
//      row" convention of chk-N1-r<k>.
//
// Usage:  m3-mixer-n1ref <libdir> <nt> <sites> <out_prefix> [dev]
//   writes  <out_prefix>.rows.bin   f32 [sites][3][nt][*}
//           <out_prefix>.nt1.bin    the same layout with nt=1 per row
//           <out_prefix>.nt1.<r>.bin  single-row images, one per r (cmp input)
// Env: MIXER=1 selects the mixer shape (default: trunk shape, for control).
//      nt selects the multi-row window (2..4).  The N=1 pass is always 1 row.
//      GGML_TP5_HC_N3_TAIL_KEEP / GGML_TP5_HC_N3_TAIL_DIAG_SCOPE /
//      TP5_HC_ROW_EPILOGUE are read by the library under test; the harness only
//      builds the graph, so the leg is selected entirely by the environment.
// Exit: 0 on success, 2 no device, 3 allocator, 4 compute, 5 output file.
// This file is authored, not built.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        hipError_t _e = (call);                                                       \
        if (_e != hipSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,            \
                    hipGetErrorString(_e));                                           \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

namespace {

constexpr int E       = 2560;   // n_embd
constexpr int HC      = 4;      // dsv4_hc_mult
constexpr int HD      = E * HC; // hc_dim
constexpr int LO      = 320;    // HC down width
constexpr int VOCAB   = 1024;   // output-matrix slice: shape, not numerics, is under test

unsigned rs = 0x1234u;
float frand() {
    rs = rs * 1103515245u + 12345u;
    return float((rs >> 8) & 0xffff) / 65535.0f;
}

struct W {
    ggml_tensor * norm = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * out  = nullptr; // mixer output matrix, F32 [E, VOCAB]
};

// Build the HC block once for `nt` rows. `mixer` selects the inject-free site;
// lo, mixed and the site output are appended to `outs` per site.
void build_block(ggml_context * ctx, ggml_cgraph * g, const std::vector<W> & ws,
                 ggml_tensor * x0, int nt, bool mixer, bool row_epi,
                 std::vector<ggml_tensor *> & outs) {
    ggml_tensor * const x = x0;
    for (const W & w : ws) {
        ggml_tensor * xn = ggml_rms_norm(ctx, x, 1e-6f);
        xn = ggml_mul(ctx, xn, w.norm);
        xn = ggml_reshape_2d(ctx, xn, HD, nt);

        ggml_tensor * lo = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, w.down, xn), 0.25f));
        ggml_tensor * up = ggml_mul_mat(ctx, w.up, lo);
        if (row_epi && nt >= 2 && nt <= 4) {
            // Same lifetime pin the qualified row-epilogue path sets; it is part
            // of the site the guard is asked about, so the harness mirrors it.
            ggml_set_output(up);
        }
        ggml_tensor * gated = ggml_mul(ctx, xn, ggml_sigmoid(ctx, up));
        gated = ggml_reshape_3d(ctx, gated, E, HC, nt);

        ggml_tensor * st[HC];
        for (int c = 0; c < HC; ++c) {
            st[c] = ggml_view_2d(ctx, gated, E, nt, ggml_row_size(GGML_TYPE_F32, E) * HC,
                                 ggml_row_size(GGML_TYPE_F32, E) * c);
            ggml_build_forward_expand(g, st[c]);
        }
        ggml_tensor * mixed = st[0];
        for (int c = 1; c < HC; ++c) {
            mixed = ggml_add(ctx, mixed, st[c]);
        }
        mixed = ggml_scale(ctx, mixed, 0.25f);

        // The stream the guard sees next: for the mixer site it is the output
        // projection (which reads `mixed`), for the trunk control it is `mixed`
        // itself. Either way the tail's window ends after `mixed`.
        ggml_tensor * site_out = mixed;
        if (mixer) {
            // No inject consumer: xn dies at `gated`, which is the property the
            // mixer site has in production. The output projection keeps the same
            // tail-window structure with a cheap F32 matrix and is the tensor the
            // N=1 reference compares.
            site_out = ggml_mul_mat(ctx, w.out, mixed);
            ggml_build_forward_expand(g, site_out);
        } else {
            ggml_build_forward_expand(g, mixed);
        }
        outs.push_back(lo);
        outs.push_back(mixed);
        outs.push_back(site_out);
    }
}

std::vector<float> read_tensor(ggml_backend_t be, ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <libdir> <nt> <sites> <out_prefix> [dev]\n", argv[0]);
        return 1;
    }
    const char * libdir = argv[1];
    const int nt = atoi(argv[2]);
    const int S = atoi(argv[3]);
    const std::string prefix = argv[4];
    const char * devname = argc > 5 ? argv[5] : "ROCm0";
    const bool mixer = std::getenv("MIXER") != nullptr && std::atoi(std::getenv("MIXER")) == 1;
    const bool row_epi = std::getenv("TP5_HC_ROW_EPILOGUE") != nullptr &&
            std::strcmp(std::getenv("TP5_HC_ROW_EPILOGUE"), "1") == 0;

    if (nt < 1 || nt > 4) {
        fprintf(stderr, "nt must be 1..4 (2..4 is the tail's multi-row window)\n");
        return 1;
    }
    ggml_backend_load_all_from_path(libdir);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(devname);
    if (dev == nullptr) {
        return 2;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    const size_t mem = size_t(96) * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<W> ws(S);
    for (W & w : ws) {
        w.norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, HC);
        w.down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, HD, LO);
        w.up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, LO, HD);
        w.out  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, VOCAB);
    }
    // Multi-row input; the per-row N=1 pass rebuilds its own context below so
    // the two graphs never share allocator state.
    ggml_tensor * x_nt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, HC, nt);
    ggml_set_input(x_nt);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, be);

    rs = 0x12345u;
    for (W & w : ws) {
        std::vector<float> nf(E * HC);
        for (float & v : nf) v = 1.0f + 0.1f * frand();
        ggml_backend_tensor_set(w.norm, nf.data(), 0, nf.size() * sizeof(float));
        std::vector<float> f(size_t(HD) * LO);
        for (float & v : f) v = 0.02f * frand();
        std::vector<uint8_t> q(ggml_nbytes(w.down));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, LO, HD, nullptr);
        ggml_backend_tensor_set(w.down, q.data(), 0, q.size());
        for (float & v : f) v = 0.05f * frand();
        q.assign(ggml_nbytes(w.up), 0);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, HD, LO, nullptr);
        ggml_backend_tensor_set(w.up, q.data(), 0, q.size());
        std::vector<float> of(size_t(E) * VOCAB);
        for (float & v : of) v = 0.01f * frand();
        ggml_backend_tensor_set(w.out, of.data(), 0, of.size() * sizeof(float));
    }

    // Row r uses a seed that does not depend on nt, mirroring the trunk micro's
    // "row r reproducible from an nt=1 run" convention.
    const int row0 = std::getenv("M3_ROW0") != nullptr ? atoi(std::getenv("M3_ROW0")) : 0;
    std::vector<float> xin(size_t(E) * HC * nt);
    for (int r = 0; r < nt; ++r) {
        rs = 0xabc00u + uint32_t(row0 + r) * 7919u;
        for (int i = 0; i < E * HC; ++i) {
            xin[size_t(r) * E * HC + i] = frand();
        }
    }
    ggml_backend_tensor_set(x_nt, xin.data(), 0, xin.size() * sizeof(float));

    // Multi-row graph: sites in sequence, each contributing lo and mixed rows.
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 64 * S + 64, false);
    std::vector<ggml_tensor *> outs;
    build_block(ctx, g, ws, x_nt, nt, mixer, row_epi, outs);
    for (ggml_tensor * t : outs) {
        ggml_set_output(t);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, g)) {
        return 3;
    }
    for (int i = 0; i < 3; ++i) {
        if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) {
            return 4;
        }
    }
    FILE * f = fopen((prefix + ".rows.bin").c_str(), "wb");
    if (f == nullptr) {
        return 5;
    }
    for (ggml_tensor * t : outs) {
        std::vector<float> v = read_tensor(be, t);
        fwrite(v.data(), sizeof(float), v.size(), f);
    }
    fclose(f);

    // Per-row N=1 pass: one graph per row, nt = 1, same row seed.
    for (int r = 0; r < nt; ++r) {
        ggml_init_params ip1 = { mem, nullptr, /*no_alloc=*/true };
        ggml_context * c1 = ggml_init(ip1);
        std::vector<W> w1(ws.size());
        for (size_t i = 0; i < ws.size(); ++i) {
            w1[i].norm = ggml_new_tensor_2d(c1, GGML_TYPE_F32, E, HC);
            w1[i].down = ggml_new_tensor_2d(c1, GGML_TYPE_Q8_0, HD, LO);
            w1[i].up   = ggml_new_tensor_2d(c1, GGML_TYPE_Q8_0, LO, HD);
            w1[i].out  = ggml_new_tensor_2d(c1, GGML_TYPE_F32, E, VOCAB);
        }
        ggml_tensor * x1 = ggml_new_tensor_3d(c1, GGML_TYPE_F32, E, HC, 1);
        ggml_set_input(x1);
        ggml_backend_buffer_t b1 = ggml_backend_alloc_ctx_tensors(c1, be);
        rs = 0x12345u;
        for (size_t i = 0; i < ws.size(); ++i) {
            std::vector<float> nf(E * HC);
            for (float & v : nf) v = 1.0f + 0.1f * frand();
            ggml_backend_tensor_set(w1[i].norm, nf.data(), 0, nf.size() * sizeof(float));
            std::vector<float> f(size_t(HD) * LO);
            for (float & v : f) v = 0.02f * frand();
            std::vector<uint8_t> q(ggml_nbytes(w1[i].down));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, LO, HD, nullptr);
            ggml_backend_tensor_set(w1[i].down, q.data(), 0, q.size());
            for (float & v : f) v = 0.05f * frand();
            q.assign(ggml_nbytes(w1[i].up), 0);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, HD, LO, nullptr);
            ggml_backend_tensor_set(w1[i].up, q.data(), 0, q.size());
            std::vector<float> of(size_t(E) * VOCAB);
            for (float & v : of) v = 0.01f * frand();
            ggml_backend_tensor_set(w1[i].out, of.data(), 0, of.size() * sizeof(float));
        }
        std::vector<float> one(xin.begin() + size_t(r) * E * HC, xin.begin() + size_t(r + 1) * E * HC);
        ggml_backend_tensor_set(x1, one.data(), 0, one.size() * sizeof(float));

        ggml_cgraph * g1 = ggml_new_graph_custom(c1, 64 * S + 64, false);
        std::vector<ggml_tensor *> o1;
        build_block(c1, g1, w1, x1, 1, mixer, row_epi, o1);
        for (ggml_tensor * t : o1) {
            ggml_set_output(t);
        }
        ggml_gallocr_t ga1 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga1, g1)) {
            return 3;
        }
        for (int i = 0; i < 3; ++i) {
            if (ggml_backend_graph_compute(be, g1) != GGML_STATUS_SUCCESS) {
                return 4;
            }
        }
        FILE * fr = fopen((prefix + ".nt1." + std::to_string(r) + ".bin").c_str(), "wb");
        if (fr == nullptr) {
            return 5;
        }
        for (ggml_tensor * t : o1) {
            std::vector<float> v = read_tensor(be, t);
            fwrite(v.data(), sizeof(float), v.size(), fr);
        }
        fclose(fr);
        ggml_gallocr_free(ga1);
        ggml_backend_buffer_free(b1);
        ggml_free(c1);
    }

    printf("{\"nt\":%d,\"sites\":%d,\"mixer\":%d,\"row_epilogue\":%d,\"outs_per_site\":3}\n",
           nt, S, int(mixer), int(row_epi));
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
