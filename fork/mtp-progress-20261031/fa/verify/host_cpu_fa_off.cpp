// verify/host_cpu_fa_off.cpp — CPU compute-path check for FA OFF == stock and
// for the runtime bound actually consumed by the CPU kernel.
//
// NOT EXECUTED in this session (user discipline: code + static review only).
// Exact build/run commands: ../RUNBOOK.md.
//
// Uses only the public ggml API, so the identical source builds against the
// stock tree and the baseline tree. Bit-exact comparison of the "A <dump>" line
// between the two builds is the OFF == stock evidence for the CPU attention
// compute path (the mask-only checks are NOT claimed as sufficient on their own).
//
// Variants printed for every tree:
//   A  no src[5], slot 5 untouched : stock graph state
//   B  slot 5 = -1 / 300           : baseline only; ggml-cpu must ignore slot 5
//                                    (slot 5 is consumed by the CUDA launcher)
//   C  src[5] = I32[1] bound       : baseline only; ggml-cpu consumes the raw
//                                    I32 value fail-closed: accepted domain is
//                                    [-1, k->ne[1]] where -1 is the documented
//                                    off sentinel (stock full width) and 0 is a
//                                    legal zero-length region; anything outside
//                                    the domain must GGML_ABORT (each case runs
//                                    in a forked child; pass = child died by
//                                    SIGABRT).
//
// A/B must hash identically on both trees. C is checked against an independent
// scalar reference for bounds in (0, n_kv]; out-of-domain bounds are no longer
// clamped and must abort.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void expect(bool ok, const std::string & what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        ++g_fail;
    }
}

constexpr int D  = 16;  // head size
constexpr int N  = 8;   // query tokens
constexpr int H  = 2;   // heads
constexpr int KV = 512; // built KV width
constexpr int NO_SLOT = INT32_MIN;

float det(int seed) {
    uint32_t s = (uint32_t) seed * 2654435761u + 12345u;
    s ^= s >> 13;
    s *= 1274126177u;
    return ((float) (s & 0xffff) / 65535.0f) - 0.5f;
}

uint32_t bits(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, 4);
    return b;
}

struct inputs {
    std::vector<float>    q, k, v;
    std::vector<uint16_t> mask;
};

inputs make_inputs(int n_q = N) {
    inputs in;
    in.q.resize((size_t) D * n_q * H);
    in.k.resize((size_t) D * KV * H);
    in.v.resize((size_t) D * KV * H);
    in.mask.resize((size_t) KV * n_q);
    for (size_t i = 0; i < in.q.size(); ++i) { in.q[i] = det((int) i); }
    for (size_t i = 0; i < in.k.size(); ++i) { in.k[i] = det((int) i + 1000); }
    for (size_t i = 0; i < in.v.size(); ++i) { in.v[i] = det((int) i + 2000); }
    for (int t = 0; t < n_q; ++t) {
        for (int j = 0; j < KV; ++j) {
            // causal, plus a second visible band far above the row position:
            // keep j <= t, and keep 400 < j <= 400 + t
            const bool keep = (j <= t) || (j > 400 && j <= 400 + t);
            in.mask[(size_t) j + (size_t) KV * t] = keep ? 0x0000 : 0xfc00;
        }
    }
    return in;
}

// n_q selects the query-token count (N for the checked one-chunk path,
// 1 for the split-KV path with KV >= 512).
void reference(const inputs & in, int bound, std::vector<float> & out, int n_q = N) {
    out.assign((size_t) D * n_q * H, NAN);
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < n_q; ++t) {
            std::vector<double> s((size_t) bound, 0.0);
            double maxs = -INFINITY;
            for (int j = 0; j < bound; ++j) {
                double acc = 0.0;
                for (int d = 0; d < D; ++d) {
                    acc += (double) in.q[(size_t) d + (size_t) D * (t + (size_t) n_q * h)] *
                                 in.k[(size_t) d + (size_t) D * (j + (size_t) KV * h)];
                }
                const uint16_t mb = in.mask[(size_t) j + (size_t) KV * t];
                const double   mv = mb == 0xfc00 ? -INFINITY : 0.0;
                s[j] = acc * (1.0 / std::sqrt((double) D)) + mv;
                maxs = std::max(maxs, s[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < bound; ++j) {
                s[j] = std::exp(s[j] - maxs);
                sum += s[j];
            }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int j = 0; j < bound; ++j) {
                    acc += s[j] * (double) in.v[(size_t) d + (size_t) D * (j + (size_t) KV * h)];
                }
                out[(size_t) d + (size_t) D * (h + (size_t) H * t)] = (float) (acc / sum);
            }
        }
    }
}

struct run_result {
    std::vector<float> out;
    std::string        dump; // "A <hex>" / "C <hex>" payload, bit-exact
};

// attach_bound: create I32[1] src[5] with the given value
// slot5:        when != NO_SLOT, write op params slot 5 (baseline only)
// n_q:          query-token count; 1 (with KV >= 512) selects the split-KV
//               dispatch path, N selects the checked one-chunk path
run_result run_fa(const inputs & in, bool attach_bound, int bound, int slot5, int n_q = N) {
    run_result res;

    ggml_init_params ip = { 512u * 1024u * 1024u, nullptr, true /* no_alloc */ };
    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, KV, H, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, KV, H, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, n_q, 1, 1);

    ggml_tensor * bound_t = nullptr;
    if (attach_bound) {
        bound_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(bound_t);
    }

    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / std::sqrt((float) D), 0.0f, 0.0f);
    ggml_set_name(fa, "fa_probe");
    if (attach_bound) {
        fa->src[5] = bound_t;
    }
#ifdef WITH_NKV_TRUE_LEN
    if (slot5 != NO_SLOT) {
        ggml_flash_attn_ext_set_n_kv_runtime(fa, slot5);
    }
#else
    (void) slot5;
#endif

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    ggml_backend_cpu_set_n_threads(backend, 1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    ggml_backend_tensor_set(q, in.q.data(), 0, in.q.size() * sizeof(float));
    ggml_backend_tensor_set(k, in.k.data(), 0, in.k.size() * sizeof(float));
    ggml_backend_tensor_set(v, in.v.data(), 0, in.v.size() * sizeof(float));
    ggml_backend_tensor_set(m, in.mask.data(), 0, in.mask.size() * sizeof(uint16_t));
    if (attach_bound) {
        const int32_t b = bound;
        ggml_backend_tensor_set(bound_t, &b, 0, sizeof(b));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    GGML_ASSERT(st == GGML_STATUS_SUCCESS);

    res.out.resize((size_t) D * n_q * H);
    ggml_backend_tensor_get(fa, res.out.data(), 0, res.out.size() * sizeof(float));

    char tmp[32];
    for (float f : res.out) {
        std::snprintf(tmp, sizeof(tmp), "%08x", bits(f));
        res.dump += tmp;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return res;
}

std::string label(const char * v, int bound, int slot5) {
    std::string s = v;
    if (bound >= 0)   { s += " src5=" + std::to_string(bound); }
    if (slot5 >= 0)   { s += " slot5=" + std::to_string(slot5); }
    return s;
}

} // namespace

int main() {
    const inputs in = make_inputs();

    // A: stock graph state
    const run_result a = run_fa(in, false, -1, NO_SLOT);
    std::printf("A %s\n", a.dump.c_str());
    expect(true, "A stock graph state ran");

#ifdef WITH_NKV_TRUE_LEN
    // B: slot 5 written with the off sentinel and with a length. ggml-cpu does
    // not read op params slot 5, so both must equal A bit-exactly.
    const run_result b_off = run_fa(in, false, -1, -1);
    expect(b_off.dump == a.dump, "B slot5=-1 ignored by ggml-cpu (dump == A)");
    const run_result b_len = run_fa(in, false, -1, 300);
    expect(b_len.dump == a.dump, "B slot5=300 ignored by ggml-cpu (dump == A)");

    // C: src[5] runtime bound is consumed. Full width must equal A exactly.
    const run_result c_full = run_fa(in, true, KV, NO_SLOT);
    std::printf("C %s\n", c_full.dump.c_str());
    expect(c_full.dump == a.dump, "C src5=n_kv identical to stock graph state");

    for (int bound : { 1, 255, 256, 257, 511 }) {
        const run_result c = run_fa(in, true, bound, NO_SLOT);

        std::vector<float> ref;
        reference(in, bound, ref);

        bool ok = true;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!(std::fabs(c.out[i] - ref[i]) <= 1e-4f)) {
                ok = false;
                break;
            }
        }
        expect(ok, label("C bound consumed", bound, NO_SLOT));
    }

    // Fail-closed channel separation: the src[5] I32[1] tensor lane accepts
    // [0, built KV width] only (the documented -1 off sentinel belongs to the
    // op-param channel, which ggml-cpu never reads). On BOTH dispatch paths
    // (checked one-chunk for n_q = N, split-KV for n_q = 1 with KV >= 512):
    // bound = 0 is a legal empty region -> all-zero output; every negative and
    // every value above the built KV width must GGML_ABORT in a forked child.
    for (int n_q : { N, 1 }) {
        const inputs inq = make_inputs(n_q);
        const run_result z = run_fa(inq, true, 0, NO_SLOT, n_q);
        bool z_all_zero = true;
        for (float f : z.out) {
            z_all_zero = z_all_zero && (f == 0.0f);
        }
        expect(z_all_zero, "C src5=0 all-zero output (n_q=" + std::to_string(n_q) + ")");
        for (int bound : { -1, -2, INT32_MIN, KV + 1, INT32_MAX }) {
            const pid_t pid = fork();
            expect(pid >= 0, "C out-of-domain fork (n_q=" + std::to_string(n_q) + ")");
            if (pid == 0) {
                run_fa(inq, true, bound, NO_SLOT, n_q); // must GGML_ABORT inside ggml-cpu
                _exit(0);                               // reached only on a fail-open regression
            }
            int wstatus = 0;
            waitpid(pid, &wstatus, 0);
            const bool aborted = WIFSIGNALED(wstatus) && WTERMSIG(wstatus) == SIGABRT;
            expect(aborted, "C src5=" + std::to_string(bound) + " aborts (n_q=" + std::to_string(n_q) + ")");
        }
    }

    // split-KV positive: single-token query over the full envelope, aligned
    // with the independent scalar reference (1e-4)
    {
        const inputs in1 = make_inputs(1);
        std::vector<float> ref1;
        reference(in1, KV, ref1, 1);
        const run_result s_full = run_fa(in1, true, KV, NO_SLOT, 1);
        bool ok = true;
        for (size_t i = 0; i < ref1.size() && ok; ++i) {
            if (!(std::fabs(s_full.out[i] - ref1[i]) <= 1e-4f)) {
                ok = false;
            }
        }
        expect(ok, "split-KV full-width result aligned with scalar reference (n_q=1, src5=512)");
    }
#else
    std::printf("NOTE stock tree build: src[5]/op-param variants are baseline-only\n");
#endif

    std::printf("%s host_cpu_fa_off failures=%d\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
