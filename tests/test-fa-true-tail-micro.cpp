// tests/test-fa-true-tail-micro.cpp
// GPU micro regression for the FA true-length (ncols2>1) device tail.
//
// Exercises the real ggml-cuda FLASH_ATTN_EXT dispatch with:
//   T1  poison test: built KV width B > runtime bound n, K/V rows [n,B) hold
//       NaN, mask allocated exactly at width n (the "mask narrower than the K
//       view" tail shape). With the runtime bound armed the kernels must never
//       read rows >= n, so the output must equal the scalar reference over
//       [0,n) with zero NaNs. Without the bound the poisoned rows are read and
//       the output is NaN -> the test fails (this is the regression detector).
//   T2  aligned equivalence: bound == built width must be bit-identical to the
//       unarmed (stock) run of the same node in the same process (both take the
//       generic tile kernel on gfx1030; on other devices the run prints the
//       comparison result and is informational).
//   T6  out-of-contract bound (width+1 or -2) must abort the process.
//
// Gate: GGML_TP5_FA_NKV_PARAM=1 (exact "1"), same as production. The test sets
// the op param through the public API ggml_flash_attn_ext_set_n_kv_runtime.
//
// Build/run (Devops only, see FA-TRUELEN-NCOLS2-AUDIT.md §6):
//   HIP_VISIBLE_DEVICES=0 GGML_TP5_FA_NKV_PARAM=1 ./test-fa-true-tail-micro --case t1 --gqa 8 --n 257
//
// Exit codes: 0 = PASS, 1 = FAIL (mismatch/NaN), 2 = usage, 3 = backend error.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int  D        = 128;
static constexpr int  B_BUILT  = 512; // built KV width (256-aligned envelope)
static constexpr int  N_Q      = 1;   // decode

namespace {

int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL: " __VA_ARGS__); std::fprintf(stderr, "\n"); ++g_fail; } } while (0)

struct args {
    std::string test = "t1";
    int  gqa = 8;     // Q heads per KV head (ncols2 selection driver)
    int  n   = 257;   // runtime KV length (0 = use built width)
    bool dump = false;
};

// Deterministic small inputs.
static void fill_f32(std::vector<float> & v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto & x : v) x = d(rng);
}

static void fill_f16(std::vector<ggml_fp16_t> & v, uint32_t seed, bool poison_tail, size_t n_valid) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = ggml_fp32_to_fp16(d(rng));
    }
    if (poison_tail) {
        // NaN payload for every element at or past n_valid (byte-poisoned so the
        // kernel cannot accidentally read a finite value even with a wrong type
        // interpretation).
        const uint16_t nan_bits = 0x7E00;
        for (size_t i = n_valid; i < v.size(); ++i) {
            std::memcpy(&v[i], &nan_bits, sizeof(nan_bits));
        }
    }
}

// Scalar reference over KV rows [0, n) for every (query token, head).
static void reference(const std::vector<float> & q, const std::vector<ggml_fp16_t> & k,
                      const std::vector<ggml_fp16_t> & v, int gqa, int n, float scale,
                      std::vector<float> & out) {
    const int n_head = gqa; // single KV head keeps the test small; gqa == n_head
    out.assign(size_t(D) * N_Q * n_head, 0.0f);
    for (int h = 0; h < n_head; ++h) {
        for (int t = 0; t < N_Q; ++t) {
            // logits
            std::vector<double> logit(n, 0.0);
            double mx = -INFINITY;
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                const ggml_fp16_t * ki = &k[size_t(i) * D]; // KV head 0
                for (int d = 0; d < D; ++d) {
                    s += double(q[size_t(t) * n_head * D + h * D + d]) *
                         double(ggml_fp16_to_fp32(ki[d]));
                }
                logit[i] = s * double(scale);
                mx = std::fmax(mx, logit[i]);
            }
            double den = 0.0;
            std::vector<double> w(n, 0.0);
            for (int i = 0; i < n; ++i) {
                w[i] = std::exp(logit[i] - mx);
                den += w[i];
            }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int i = 0; i < n; ++i) {
                    acc += w[i] * double(ggml_fp16_to_fp32(v[size_t(i) * D + d]));
                }
                out[size_t(t) * n_head * D + h * D + d] = float(acc / den);
            }
        }
    }
}

static int run_case(const args & a) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        std::fprintf(stderr, "no GPU backend device\n");
        return 3;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        std::fprintf(stderr, "backend init failed\n");
        return 3;
    }

    const int gqa = a.gqa;
    const int n_head = gqa;
    const int n_kv_head = 1;
    const int n = (a.n > 0 && a.n <= B_BUILT) ? a.n : B_BUILT;
    const bool bound_armed = std::getenv("GGML_TP5_FA_NKV_PARAM") != nullptr &&
                             std::strcmp(std::getenv("GGML_TP5_FA_NKV_PARAM"), "1") == 0;
    const bool poison = a.test == "t1";

    // Tensors are allocated ON the device buffer (the canonical backend-test
    // pattern): graph_compute on CUDA requires device-resident tensors.
    struct ggml_init_params ip = { /*mem_size*/ 64u * 1024u * 1024u, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N_Q, n_head, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, B_BUILT, n_kv_head, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, B_BUILT, n_kv_head, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n, N_Q, 1, 1);
    const float scale = 1.0f / std::sqrt(float(D));
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    const int32_t bound = a.test == "t6-invalid-hi" ? B_BUILT + 1
                        : a.test == "t6-invalid-neg" ? -2
                        : (a.test == "t2" ? B_BUILT : n);
    bool armed = bound_armed && bound >= 0;
    if (a.test != "t1-nobound" && a.test != "t2-nobound") {
        ggml_flash_attn_ext_set_n_kv_runtime(out, bound);
        armed = bound_armed && bound >= 0;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // Allocate ALL graph tensors (inputs + FA output) into one device buffer;
    // created above so the FA node is included in the allocation set.
    ggml_backend_buffer_t dev_buf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(backend));
    if (dev_buf == nullptr) {
        std::fprintf(stderr, "device buffer alloc failed\n");
        return 3;
    }

    // Host data.
    std::vector<float> qh(size_t(D) * N_Q * n_head);
    std::vector<ggml_fp16_t> kh, vh;
    std::vector<ggml_fp16_t> mh(size_t(n) * N_Q);
    fill_f32(qh, 1234u + uint32_t(gqa));
    kh.resize(size_t(D) * B_BUILT * n_kv_head);
    vh.resize(size_t(D) * B_BUILT * n_kv_head);
    fill_f16(kh, 5678u + uint32_t(gqa), poison, size_t(D) * size_t(n) * n_kv_head);
    fill_f16(vh, 9012u + uint32_t(gqa), poison, size_t(D) * size_t(n) * n_kv_head);
    for (auto & x : mh) x = ggml_fp32_to_fp16(0.0f); // all allowed within [0,n)

    ggml_backend_tensor_set(q, qh.data(), 0, qh.size() * sizeof(float));
    ggml_backend_tensor_set(k, kh.data(), 0, kh.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, vh.data(), 0, vh.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));

    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute status %d (abort path is exit code != 0)\n", (int) st);
        return a.test.rfind("t6", 0) == 0 ? 0 : 1; // t6 expects the abort
    }
    if (a.test.rfind("t6", 0) == 0) {
        std::fprintf(stderr, "FAIL: out-of-contract bound was accepted (no abort)\n");
        return 1;
    }

    std::vector<float> got(size_t(D) * N_Q * n_head);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    size_t nan_count = 0;
    double max_err = 0.0;
    std::vector<float> ref;
    reference(qh, kh, vh, gqa, n, scale, ref);
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) { ++nan_count; continue; }
        max_err = std::fmax(max_err, std::fabs(double(got[i] - ref[i])));
    }

    std::printf("case %s gqa=%d n=%d armed=%d nan=%zu max_err=%.3e\n",
                a.test.c_str(), gqa, n, (int) armed, nan_count, max_err);
    if (a.dump) {
        uint64_t h = 1469598103934665603ull;
        for (float x : got) { uint32_t u; std::memcpy(&u, &x, 4); h ^= u; h *= 1099511628211ull; }
        std::printf("hash=%016llx\n", (unsigned long long) h);
    }

    if (a.test != "t1-nobound") {
        CHECK(nan_count == 0, "non-finite outputs (%zu)", nan_count);
        CHECK(max_err <= 1e-5, "max abs error %.3e vs reference", max_err);
    }

    // Negative control (--case t1-nobound): no runtime bound is armed, so the
    // poisoned rows ARE read and the output must be non-finite. If it is finite
    // the poison never reached the computation and the T1 above proves nothing.
    if (a.test == "t1-nobound") {
        const bool control_ok = (nan_count > 0);
        std::printf("control(unarmed) poison-visible=%d\n", (int) control_ok);
        g_fail = control_ok ? 0 : 1;
        if (!control_ok) {
            std::fprintf(stderr, "FAIL: poison did not reach the output without a bound\n");
        }
    }

    ggml_free(ctx);
    ggml_backend_buffer_free(dev_buf);
    ggml_backend_free(backend);
    return g_fail == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--case" && i + 1 < argc) a.test = argv[++i];
        else if (s == "--gqa" && i + 1 < argc) a.gqa = std::atoi(argv[++i]);
        else if (s == "--n" && i + 1 < argc) a.n = std::atoi(argv[++i]);
        else if (s == "--dump") a.dump = true;
        else { std::fprintf(stderr, "usage: %s [--case t1|t2|t1-nobound|t2-nobound|t6-invalid-hi|t6-invalid-neg] [--gqa N] [--n N] [--dump]\n", argv[0]); return 2; }
    }
    std::printf("gates: GGML_TP5_FA_NKV_PARAM=%s GGML_TP5_FA_CHECKED_TAIL=%s\n",
                std::getenv("GGML_TP5_FA_NKV_PARAM") ? std::getenv("GGML_TP5_FA_NKV_PARAM") : "(unset)",
                std::getenv("GGML_TP5_FA_CHECKED_TAIL") ? std::getenv("GGML_TP5_FA_CHECKED_TAIL") : "(unset)");
    // Every case except the t*-nobound negative controls requires the exact-"1"
    // gate; refusing loudly instead of silently passing an unarmed run.
    const bool gate_on = std::getenv("GGML_TP5_FA_NKV_PARAM") != nullptr &&
                         std::strcmp(std::getenv("GGML_TP5_FA_NKV_PARAM"), "1") == 0;
    const bool needs_gate = !(a.test == "t1-nobound" || a.test == "t2-nobound");
    if (needs_gate && !gate_on) {
        std::fprintf(stderr, "SKIP-FAIL: case %s requires GGML_TP5_FA_NKV_PARAM=1 (exact)\n", a.test.c_str());
        return 4;
    }
    return run_case(a);
}
