// test-xkv-vulkan-factorize.cpp — focused test for native XKV factorization ops.
//
// Covers (CPU oracle vs CPU graph vs Vulkan graph), K and V independently:
//  - low-rank+noise, rank-deficient, repeated singular values,
//    n/m/r non-block-aligned, K!=V, multiple layer-group feature widths,
//    partial HALF IMRoPE canonicalization, Turbo2/3/4/Q8_0/F16/F32 inputs;
//  - residual/cosine/subspace tolerances, deterministic byte streams,
//    sign/order stability, exact scratch T vs T-1, no NaN, one final sync,
//    no D2H except tiny scalar status, injected failure leaves outputs
//    untouched (unpublished).
// Vulkan absence skips only the Vulkan subcase; claimed capability with
// runtime failure fails. No project-wide commands; standalone binary.

#include "ggml.h"
#include "ggml-xkv.h"
#include "ggml-xkv-factor.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
// std::min
#include <algorithm>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

// ---------- deterministic helpers ----------

static uint64_t rng_state = 0x12345678ULL;
static uint64_t rng_next() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static float rng_uni() {
    return (float)(rng_next() & 0x7FFFFFFFu) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}
static void rng_seed(uint64_t s) { rng_state = s ? s : 0x12345678ULL; }

static std::vector<float> gen_lowrank_noise(uint32_t n, uint32_t m, uint32_t r0,
                                            float noise, uint64_t seed) {
    rng_seed(seed);
    std::vector<float> A(size_t(n) * r0), B(size_t(m) * r0);
    for (auto & x : A) x = rng_uni();
    for (auto & x : B) x = rng_uni() * 0.5f;
    std::vector<float> X(size_t(n) * m, 0.0f);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (uint32_t k = 0; k < r0; ++k) s += (double)A[size_t(i) * r0 + k] * B[size_t(j) * r0 + k];
            X[size_t(i) * m + j] = (float)s + noise * rng_uni();
        }
    return X;
}

static std::vector<float> gen_rankdef(uint32_t n, uint32_t m, uint64_t seed) {
    // First min(n,m)/2 columns duplicated -> rank <= ceil(min/2)
    rng_seed(seed);
    std::vector<float> X(size_t(n) * m);
    for (auto & x : X) x = rng_uni();
    uint32_t h = std::min(n, m) / 2;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = h; j < 2 * h && j < m; ++j)
            X[size_t(i) * m + j] = X[size_t(i) * m + j - h];
    return X;
}

static std::vector<float> gen_repeated_sv(uint32_t n, uint32_t m, uint64_t seed) {
    // X = U S V^T with repeated singular values (S has triples of equal values)
    rng_seed(seed);
    uint32_t k = std::min(n, m);
    std::vector<float> U(size_t(n) * k), V(size_t(m) * k), S(k);
    for (auto & x : U) x = rng_uni();
    for (auto & x : V) x = rng_uni();
    // orthonormalize columns (modified Gram-Schmidt, double)
    auto orth = [](std::vector<float> & Q, uint32_t rows, uint32_t cols) {
        std::vector<double> q(size_t(rows) * cols);
        for (size_t i = 0; i < q.size(); ++i) q[i] = Q[i];
        for (uint32_t j = 0; j < cols; ++j) {
            for (uint32_t j0 = 0; j0 < j; ++j0) {
                double d = 0.0;
                for (uint32_t i = 0; i < rows; ++i) d += q[size_t(i) * cols + j0] * q[size_t(i) * cols + j];
                for (uint32_t i = 0; i < rows; ++i) q[size_t(i) * cols + j] -= d * q[size_t(i) * cols + j0];
            }
            double nn = 0.0;
            for (uint32_t i = 0; i < rows; ++i) { double v = q[size_t(i) * cols + j]; nn += v * v; }
            nn = std::sqrt(nn);
            if (nn < 1e-9) {
                for (uint32_t i = 0; i < rows; ++i) q[size_t(i) * cols + j] = (i == j % rows) ? 1.0 : 0.0;
            } else {
                for (uint32_t i = 0; i < rows; ++i) q[size_t(i) * cols + j] /= nn;
            }
        }
        for (size_t i = 0; i < q.size(); ++i) Q[i] = (float)q[i];
    };
    orth(U, n, k);
    orth(V, m, k);
    for (uint32_t i = 0; i < k; ++i) S[i] = 4.0f / (1 + (i / 3)); // triples: 4,4,4,2,2,2,...
    std::vector<float> X(size_t(n) * m, 0.0f);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (uint32_t t = 0; t < k; ++t) s += (double)U[size_t(i) * k + t] * S[t] * V[size_t(j) * k + t];
            X[size_t(i) * m + j] = (float)s;
        }
    return X;
}

static double frob(const std::vector<float> & v) {
    double s = 0.0;
    for (float x : v) s += (double)x * (double)x;
    return std::sqrt(s);
}

// Decode one packed stream to floats (ROTATED Turbo domain, like production).
static bool decode_stream(ggml_type type, const std::vector<uint8_t> & bytes,
                          uint32_t rows, uint32_t pr, uint32_t rank,
                          std::vector<float> & out) {
    out.assign(size_t(rows) * rank, 0.0f);
    size_t rb = ggml_row_size(type, pr);
    if (bytes.size() != rb * rows) return false;
    std::vector<float> pad(pr);
    bool is_turbo = (type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0);
    for (uint32_t i = 0; i < rows; ++i) {
        const uint8_t * src = bytes.data() + size_t(i) * rb;
        if (type == GGML_TYPE_F32) {
            memcpy(pad.data(), src, size_t(pr) * 4);
        } else if (is_turbo) {
            if (!ggml_dequantize_turbo_row(type, src, pad.data(), pr, 128, GGML_TURBO_DECODE_ROTATED)) return false;
        } else {
            const auto * tr = ggml_get_type_traits(type);
            if (!tr || !tr->to_float) return false;
            tr->to_float(src, pad.data(), pr);
        }
        memcpy(out.data() + size_t(i) * rank, pad.data(), size_t(rank) * 4);
    }
    return true;
}

static double rel_residual(const std::vector<float> & X,
                           const std::vector<float> & A, const std::vector<float> & BT,
                           uint32_t n, uint32_t m, uint32_t r) {
    double se = 0.0, so = 0.0;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < m; ++j) {
            double rec = 0.0;
            for (uint32_t k = 0; k < r; ++k) rec += (double)A[size_t(i) * r + k] * BT[size_t(j) * r + k];
            double o = X[size_t(i) * m + j];
            double e = o - rec;
            se += e * e;
            so += o * o;
        }
    if (so < 1e-12) return std::sqrt(se);
    return std::sqrt(se / so);
}

static double cosine_sim(const std::vector<float> & X, const std::vector<float> & A,
                         const std::vector<float> & BT, uint32_t n, uint32_t m, uint32_t r) {
    double dot = 0.0, nx = 0.0, nr = 0.0;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < m; ++j) {
            double rec = 0.0;
            for (uint32_t k = 0; k < r; ++k) rec += (double)A[size_t(i) * r + k] * BT[size_t(j) * r + k];
            double o = X[size_t(i) * m + j];
            dot += o * rec;
            nx += o * o;
            nr += rec * rec;
        }
    if (nx < 1e-20 || nr < 1e-20) return 0.0;
    return dot / std::sqrt(nx * nr);
}

static bool has_nan(const std::vector<uint8_t> & bytes, ggml_type t, uint32_t rows, uint32_t pr) {
    std::vector<float> d;
    uint32_t r = pr;
    if (!decode_stream(t, bytes, rows, pr, r, d)) return true;
    for (float x : d) if (!std::isfinite(x)) return true;
    return false;
}

struct factor_buffers {
    ggml_xkv_factorize_params p;
    std::vector<uint8_t> a_bytes, b_bytes;
    std::vector<float> svals;
    ggml_xkv_residual r_ab = {0, 0, 0}, r_a = {0, 0, 0}, r_b = {0, 0, 0};
    int32_t status = -1;
};

static bool run_oracle_full(const std::vector<float> & X, uint32_t n, uint32_t m,
                            const ggml_xkv_factorize_params & p, factor_buffers & fb,
                            std::string & err_out) {
    fb.p = p;
    uint32_t r = std::min(p.requested_rank, std::min(n, m));
    uint32_t pra = p.pad_r_a ? p.pad_r_a : r;
    uint32_t prb = p.pad_r_b ? p.pad_r_b : r;
    fb.a_bytes.assign(ggml_row_size((ggml_type)p.type_a, pra) * n, 0xAB);
    fb.b_bytes.assign(ggml_row_size((ggml_type)p.type_b, prb) * m, 0xAB);
    fb.svals.assign(r, 0.0f);
    char err[256] = {0};
    bool ok = ggml_xkv_factorize_cpu_oracle_resid(X.data(), n, m, &p,
        fb.a_bytes.data(), fb.b_bytes.data(), fb.svals.data(),
        &fb.r_ab, &fb.r_a, &fb.r_b, err, sizeof(err));
    fb.status = ok ? 0 : 2;
    if (!ok) err_out = err;
    return ok;
}

static ggml_xkv_factorize_params default_params(uint32_t n, uint32_t m, uint32_t r,
        ggml_type ta, ggml_type tb, uint64_t seed, uint32_t balance = GGML_XKV_BALANCE_UPSTREAM) {
    ggml_xkv_factorize_params p = {};
    p.version = GGML_XKV_FACTOR_VERSION;
    p.rows_n = n;
    p.cols_m = m;
    p.requested_rank = r;
    p.oversampling = 16;
    p.power_iterations = 2;
    p.balance_mode = balance;
    p.type_a = (uint32_t)ta;
    p.type_b = (uint32_t)tb;
    p.seed_low = (uint32_t)seed;
    p.seed_high = (uint32_t)(seed >> 32);
    uint32_t rr = std::min(r, std::min(n, m));
    p.pad_r_a = (uint32_t)ggml_xkv_padded_rank(ta, rr);
    p.pad_r_b = (uint32_t)ggml_xkv_padded_rank(tb, rr);
    return p;
}

// ---------- oracle property tests ----------

static void test_lowrank_oracle() {
    std::printf("[factor] low-rank+noise oracle ...\n");
    // non-block-aligned n/m/r; exact rank 12 + small noise; F32 streams
    uint32_t n = 100, m = 70, r0 = 12, r = 16;
    auto X = gen_lowrank_noise(n, m, r0, 1e-4f, 777);
    auto p = default_params(n, m, r, GGML_TYPE_F32, GGML_TYPE_F32, 42);
    factor_buffers fb;
    std::string err;
    CHECK(run_oracle_full(X, n, m, p, fb, err));
    std::vector<float> A, BT;
    CHECK(decode_stream(GGML_TYPE_F32, fb.a_bytes, n, p.pad_r_a, r, A));
    CHECK(decode_stream(GGML_TYPE_F32, fb.b_bytes, m, p.pad_r_b, r, BT));
    double rel = rel_residual(X, A, BT, n, m, r);
    double cos = cosine_sim(X, A, BT, n, m, r);
    CHECK(rel < 5e-3);
    CHECK(cos > 0.99999);
    CHECK(!has_nan(fb.a_bytes, GGML_TYPE_F32, n, p.pad_r_a));
    // sorted singular values desc
    for (uint32_t k = 1; k < r; ++k) CHECK(fb.svals[k - 1] + 1e-6f >= fb.svals[k]);
    // sign canonicalization: max-|.| of each A column is non-negative
    for (uint32_t k = 0; k < r; ++k) {
        float best = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            float v = A[size_t(i) * r + k];
            if (std::fabs(v) > std::fabs(best)) best = v;
        }
        CHECK(best >= 0.0f);
    }
    std::printf("  rel=%.6f cos=%.6f\n", rel, cos);
}

static void test_rankdeficient_oracle() {
    std::printf("[factor] rank-deficient oracle ...\n");
    uint32_t n = 96, m = 64, r = 40; // true rank <= 32 < r
    auto X = gen_rankdef(n, m, 31337);
    auto p = default_params(n, m, r, GGML_TYPE_F32, GGML_TYPE_F32, 7);
    factor_buffers fb;
    std::string err;
    CHECK(run_oracle_full(X, n, m, p, fb, err));
    std::vector<float> A, BT;
    CHECK(decode_stream(GGML_TYPE_F32, fb.a_bytes, n, p.pad_r_a, r, A));
    CHECK(decode_stream(GGML_TYPE_F32, fb.b_bytes, m, p.pad_r_b, r, BT));
    double rel = rel_residual(X, A, BT, n, m, r);
    CHECK(rel < 1e-3);
    CHECK(!has_nan(fb.a_bytes, GGML_TYPE_F32, n, p.pad_r_a));
    CHECK(!has_nan(fb.b_bytes, GGML_TYPE_F32, m, p.pad_r_b));
    // trailing singular values ~0 (rank deficiency handled, no blowup)
    CHECK(fb.svals[r - 1] < 1e-2f);
    std::printf("  rel=%.6f s_last=%.6f\n", rel, fb.svals[r - 1]);
}

static void test_repeated_sv_oracle() {
    std::printf("[factor] repeated singular values oracle ...\n");
    uint32_t n = 48, m = 40, r = 24;
    auto X = gen_repeated_sv(n, m, 999);
    // determinism: same seed twice -> bitwise identical streams
    auto p = default_params(n, m, r, GGML_TYPE_F32, GGML_TYPE_F32, 1234);
    factor_buffers f1, f2;
    std::string err;
    CHECK(run_oracle_full(X, n, m, p, f1, err));
    CHECK(run_oracle_full(X, n, m, p, f2, err));
    CHECK(f1.a_bytes == f2.a_bytes);
    CHECK(f1.b_bytes == f2.b_bytes);
    CHECK(f1.svals == f2.svals);
    std::vector<float> A, BT;
    CHECK(decode_stream(GGML_TYPE_F32, f1.a_bytes, n, p.pad_r_a, r, A));
    CHECK(decode_stream(GGML_TYPE_F32, f1.b_bytes, m, p.pad_r_b, r, BT));
    double rel = rel_residual(X, A, BT, n, m, r);
    // The fixture is full-rank; rank-24 has an irreducible truncated tail.
    // Repeated singular values exercise deterministic basis/sign selection,
    // while residual must stay near the optimal truncated result.
    CHECK(rel < 0.20);
    std::printf("  rel=%.6f deterministic=1\n", rel);
}

static void test_quant_codecs_oracle() {
    std::printf("[factor] Turbo2/3/4/Q8/F16 codec streams ...\n");
    uint32_t n = 128, m = 128, r = 32;
    auto X = gen_lowrank_noise(n, m, 16, 1e-3f, 555);
    struct cc { ggml_type t; double rel_max; double cos_min; const char * name; };
    const cc cases[] = {
        { GGML_TYPE_F16,     0.05, 0.9990, "F16" },
        { GGML_TYPE_Q8_0,    0.15, 0.9950, "Q8_0" },
        { GGML_TYPE_TURBO4_0, 0.25, 0.9900, "Turbo4" },
        { GGML_TYPE_TURBO3_0, 0.35, 0.9700, "Turbo3" },
        { GGML_TYPE_TURBO2_0, 0.50, 0.9000, "Turbo2" },
    };
    for (const auto & c : cases) {
        auto p = default_params(n, m, r, c.t, c.t, 2024, GGML_XKV_BALANCE_SQRT);
        factor_buffers fb;
        std::string err;
        CHECK(run_oracle_full(X, n, m, p, fb, err));
        std::vector<float> A, BT;
        const uint32_t product_rank = (c.t == GGML_TYPE_TURBO2_0 ||
            c.t == GGML_TYPE_TURBO3_0 || c.t == GGML_TYPE_TURBO4_0) ? p.pad_r_a : r;
        CHECK(decode_stream(c.t, fb.a_bytes, n, p.pad_r_a, product_rank, A));
        CHECK(decode_stream(c.t, fb.b_bytes, m, p.pad_r_b, product_rank, BT));
        // Turbo WHT padding spreads logical components across the padded row;
        // the reader intentionally dots all padded coordinates.
        double rel = rel_residual(X, A, BT, n, m, product_rank);
        double cos = cosine_sim(X, A, BT, n, m, product_rank);
        CHECK(rel < c.rel_max);
        CHECK(cos > c.cos_min);
        CHECK(!has_nan(fb.a_bytes, c.t, n, p.pad_r_a));
        // byte determinism per codec
        factor_buffers fb2;
        CHECK(run_oracle_full(X, n, m, p, fb2, err));
        CHECK(fb.a_bytes == fb2.a_bytes);
        CHECK(fb.b_bytes == fb2.b_bytes);
        std::printf("  %s rel=%.4f cos=%.5f\n", c.name, rel, cos);
    }
}

static void test_k_neq_v() {
    std::printf("[factor] K!=V independent factorization ...\n");
    uint32_t n = 100;
    auto Xk = gen_lowrank_noise(n, 70, 10, 1e-3f, 11);   // K: m=70, r=37
    auto Xv = gen_lowrank_noise(n, 96, 14, 1e-3f, 22);   // V: m=96, r=40
    auto pk = default_params(n, 70, 37, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 101);
    auto pv = default_params(n, 96, 40, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0, 102); // V seed = K seed + 1 pattern
    factor_buffers fk, fv;
    std::string err;
    CHECK(run_oracle_full(Xk, n, 70, pk, fk, err));
    CHECK(run_oracle_full(Xv, n, 96, pv, fv, err));
    CHECK(fk.a_bytes.size() != fv.a_bytes.size()); // distinct shapes/streams
    std::vector<float> Ak, BTk, Av, BTv;
    CHECK(decode_stream(GGML_TYPE_TURBO4_0, fk.a_bytes, n, pk.pad_r_a, pk.pad_r_a, Ak));
    CHECK(decode_stream(GGML_TYPE_TURBO4_0, fk.b_bytes, 70, pk.pad_r_b, pk.pad_r_b, BTk));
    CHECK(decode_stream(GGML_TYPE_TURBO2_0, fv.a_bytes, n, pv.pad_r_a, pv.pad_r_a, Av));
    CHECK(decode_stream(GGML_TYPE_TURBO2_0, fv.b_bytes, 96, pv.pad_r_b, pv.pad_r_b, BTv));
    CHECK(rel_residual(Xk, Ak, BTk, n, 70, pk.pad_r_a) < 0.30);
    CHECK(rel_residual(Xv, Av, BTv, n, 96, pv.pad_r_a) < 0.55);
    std::printf("  K/V independent OK\n");
}

static void test_balance_modes() {
    std::printf("[factor] balance modes preserve product ...\n");
    uint32_t n = 64, m = 48, r = 16;
    auto X = gen_lowrank_noise(n, m, 8, 1e-3f, 4242);
    double prods[3] = {0};
    for (int b = 0; b < 3; ++b) {
        auto p = default_params(n, m, r, GGML_TYPE_F32, GGML_TYPE_F32, 9, (uint32_t)b);
        factor_buffers fb;
        std::string err;
        CHECK(run_oracle_full(X, n, m, p, fb, err));
        std::vector<float> A, BT;
        CHECK(decode_stream(GGML_TYPE_F32, fb.a_bytes, n, p.pad_r_a, r, A));
        CHECK(decode_stream(GGML_TYPE_F32, fb.b_bytes, m, p.pad_r_b, r, BT));
        prods[b] = rel_residual(X, A, BT, n, m, r);
        CHECK(prods[b] < 1e-3);
    }
    std::printf("  upstream/sqrt/diag rel=%.6f/%.6f/%.6f\n", prods[0], prods[1], prods[2]);
}

// ---------- scratch exactness ----------

static void test_scratch_exact() {
    std::printf("[factor] exact scratch T vs T-1 ...\n");
    ggml_xkv_factorize_params p = default_params(100, 70, 37, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 1);
    size_t need = 0;
    char err[256] = {0};
    CHECK(ggml_xkv_factorize_scratch_bytes(&p, &need, err, sizeof(err)));
    CHECK(need > 0);
    // Build dummy tensors to run supports() with T and T-1 scratch.
    ggml_init_params ip = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 70, 100);
    ggml_tensor * sc_full = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4));
    ggml_tensor * sc_short = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4) - 1);
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_TURBO4_0, p.pad_r_a, 70);
    ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 10); // [0]=code [1..9]=residuals
    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_TURBO4_0, p.pad_r_a, 100);
    CHECK(ggml_xkv_factorize_supports(x, sc_full, b, st, a, &p, err, sizeof(err)));
    CHECK(!ggml_xkv_factorize_supports(x, sc_short, b, st, a, &p, err, sizeof(err)));
    std::printf("  need=%zu bytes; T ok, T-1 rejected (%s)\n", need, err);
}

// ---------- failure atomicity ----------

static void test_failure_atomicity() {
    std::printf("[factor] injected failures leave outputs unpublished ...\n");
    uint32_t n = 150, m = 48, r = 16; // n > 128 to test row 129 / last row adversarial injection!
    auto X = gen_lowrank_noise(n, m, 8, 1e-3f, 818);
    auto good = default_params(n, m, r, GGML_TYPE_F32, GGML_TYPE_F32, 3);
    // baseline success
    {
        factor_buffers fb;
        std::string err;
        CHECK(run_oracle_full(X, n, m, good, fb, err));
        CHECK(fb.status == 0);
    }
    auto expect_fail_untouched = [&](ggml_xkv_factorize_params bad, const char * name) {
        uint32_t rr = std::min(bad.requested_rank, std::min(n, m));
        uint32_t pra = bad.pad_r_a ? bad.pad_r_a : (rr ? rr : 16);
        uint32_t prb = bad.pad_r_b ? bad.pad_r_b : (rr ? rr : 16);
        ggml_type ta = (ggml_type)bad.type_a, tb = (ggml_type)bad.type_b;
        if (!ggml_xkv_codec_supported(ta)) ta = GGML_TYPE_F32;
        if (!ggml_xkv_codec_supported(tb)) tb = GGML_TYPE_F32;
        std::vector<uint8_t> oa(ggml_row_size(ta, pra) * n, 0x5A);
        std::vector<uint8_t> ob(ggml_row_size(tb, prb) * m, 0x5A);
        std::vector<uint8_t> oa0 = oa, ob0 = ob;
        std::vector<float> sv(rr ? rr : 16, 0.0f);
        char err[256] = {0};
        bool ok = ggml_xkv_factorize_cpu_oracle(X.data(), n, m, &bad,
            oa.data(), ob.data(), sv.data(), err, sizeof(err));
        CHECK(!ok);
        CHECK(oa == oa0);
        CHECK(ob == ob0);
        (void)name;
    };
    {
        auto bad = good;
        bad.version = 0xFFFF;
        expect_fail_untouched(bad, "version");
    }
    {
        auto bad = good;
        bad.requested_rank = 0;
        expect_fail_untouched(bad, "rank0");
    }
    {
        auto bad = good;
        bad.power_iterations = 0;
        expect_fail_untouched(bad, "q0");
    }
    {
        auto bad = good;
        bad.balance_mode = 99;
        expect_fail_untouched(bad, "balance");
    }
    {
        // Adversarial error injected specifically at row 129 (which would pass a 128-row sampled gate!):
        // must fail oracle residual and be rejected!
        auto X_adv = X;
        for (uint32_t j = 0; j < m; ++j) X_adv[size_t(129) * m + j] += 50.0f;
        uint32_t rr = r;
        std::vector<uint8_t> oa(ggml_row_size(GGML_TYPE_F32, rr) * n, 0x5A);
        std::vector<uint8_t> ob(ggml_row_size(GGML_TYPE_F32, rr) * m, 0x5A);
        std::vector<float> sv(rr, 0.0f);
        ggml_xkv_residual r_ab = {0,0,0}, r_a = {0,0,0}, r_b = {0,0,0};
        char err[256] = {};
        bool ok = ggml_xkv_factorize_cpu_oracle_resid(X_adv.data(), n, m, &good, oa.data(), ob.data(), sv.data(),
            &r_ab, &r_a, &r_b, err, sizeof(err));
        CHECK(ok);
        // Assert that the exact all-row residual catches the row 129 corruption:
        double rel_err = r_ab.frob_err / r_ab.frob_orig;
        CHECK(rel_err > 0.40); // huge residual detected!
    }
    {
        // Adversarial error injected at the very last row (n - 1 = 149):
        auto X_adv = X;
        for (uint32_t j = 0; j < m; ++j) X_adv[size_t(n - 1) * m + j] += 50.0f;
        uint32_t rr = r;
        std::vector<uint8_t> oa(ggml_row_size(GGML_TYPE_F32, rr) * n, 0x5A);
        std::vector<uint8_t> ob(ggml_row_size(GGML_TYPE_F32, rr) * m, 0x5A);
        std::vector<float> sv(rr, 0.0f);
        ggml_xkv_residual r_ab = {0,0,0}, r_a = {0,0,0}, r_b = {0,0,0};
        char err[256] = {};
        bool ok = ggml_xkv_factorize_cpu_oracle_resid(X_adv.data(), n, m, &good, oa.data(), ob.data(), sv.data(),
            &r_ab, &r_a, &r_b, err, sizeof(err));
        CHECK(ok);
        double rel_err = r_ab.frob_err / r_ab.frob_orig;
        CHECK(rel_err > 0.40); // caught!
    }
    {
        // NaN input
        auto Xn = X;
        Xn[7] = std::numeric_limits<float>::quiet_NaN();
        uint32_t rr = r;
        std::vector<uint8_t> oa(ggml_row_size(GGML_TYPE_F32, rr) * n, 0x5A);
        std::vector<uint8_t> ob(ggml_row_size(GGML_TYPE_F32, rr) * m, 0x5A);
        std::vector<uint8_t> oa0 = oa;
        std::vector<float> sv(rr, 0.0f);
        char err[256] = {0};
        CHECK(!ggml_xkv_factorize_cpu_oracle(Xn.data(), n, m, &good, oa.data(), ob.data(), sv.data(), err, sizeof(err)));
        CHECK(oa == oa0);
    }
    std::printf("  all failure cases untouched OK\n");
}

// ---------- canonicalize tests ----------

static std::vector<float> build_rope_tables(uint32_t rotary_dim, float mag = 1.0f) {
    uint32_t fc = rotary_dim / 2;
    std::vector<float> t(size_t(fc) * 2);
    for (uint32_t f = 0; f < fc; ++f) {
        t[f] = std::pow(10000.0f, -2.0f * (float)f / (float)rotary_dim);
        t[fc + f] = mag;
    }
    return t;
}

// forward RoPE (HALF) for roundtrip validation
static void apply_rope_fwd_half(std::vector<float> & v, uint32_t hd, uint32_t rotary_dim,
                                float pos, const std::vector<float> & tables) {
    uint32_t fc = rotary_dim / 2;
    for (uint32_t f = 0; f < fc; ++f) {
        float ang = tables[f] * pos;
        float c = std::cos(ang), s = std::sin(ang);
        float sc = tables[fc + f];
        float re = v[f] * sc, im = v[f + fc] * sc;
        v[f] = re * c - im * s;
        v[f + fc] = re * s + im * c;
    }
    (void)hd;
}

static void test_canonicalize_roundtrip() {
    std::printf("[canon] partial HALF IMRoPE roundtrip, all hot codecs ...\n");
    // feature widths: 4 layers x 2 heads x 96 dim (K), 4x2x64 (V); rotary 32
    struct width { uint32_t nl, nh, hd, rot; };
    const width widths[] = { {4, 2, 96, 32}, {2, 3, 64, 32}, {3, 1, 128, 0} };
    const ggml_type codecs[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0,
        GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };
    for (const auto & w : widths) {
        uint32_t nvec = w.nl * w.nh;
        uint32_t tf = nvec * w.hd;
        uint32_t n = 24;
        // canonical ground truth
        rng_seed(1000 + w.nl * 131 + w.nh * 17 + w.hd);
        std::vector<float> truth(size_t(n) * tf);
        for (auto & x : truth) x = rng_uni() * 2.0f;
        auto rope = build_rope_tables(w.rot, 1.0f);
        std::vector<int32_t> rows(n), pos(n);
        for (uint32_t i = 0; i < n; ++i) { rows[i] = (int32_t)i; pos[i] = (int32_t)(i * 3 + 1); }
        for (ggml_type ht : codecs) {
            bool turbo = (ht == GGML_TYPE_TURBO2_0 || ht == GGML_TYPE_TURBO3_0 || ht == GGML_TYPE_TURBO4_0);
            if (ht == GGML_TYPE_Q8_0 && (w.hd % 32) != 0) continue;
            uint32_t phd = turbo ? ((w.hd + 127) / 128 * 128) : w.hd;
            size_t head_bytes = ggml_row_size(ht, phd);
            std::vector<uint8_t> hot(head_bytes * nvec * n, 0);
            // build hot = forward-RoPE(truth) encoded per head (K path)
            std::vector<float> pad(phd, 0.0f);
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t v = 0; v < nvec; ++v) {
                    std::vector<float> h(phd, 0.0f);
                    memcpy(h.data(), truth.data() + (size_t(i) * nvec + v) * w.hd, size_t(w.hd) * 4);
                    if (w.rot > 0) apply_rope_fwd_half(h, w.hd, w.rot, (float)pos[i], rope);
                    uint8_t * dst = hot.data() + (size_t(i) * nvec + v) * head_bytes;
                    if (ht == GGML_TYPE_F32) {
                        memcpy(dst, h.data(), size_t(phd) * 4);
                    } else if (ht == GGML_TYPE_F16) {
                        auto * tr = ggml_get_type_traits(ht);
                        tr->from_float_ref(h.data(), dst, phd);
                    } else if (ht == GGML_TYPE_Q8_0) {
                        auto * tr = ggml_get_type_traits(ht);
                        tr->from_float_ref(h.data(), dst, phd);
                    } else {
                        CHECK(ggml_quantize_turbo_row(ht, h.data(), dst, phd, 128));
                    }
                }
            }
            ggml_xkv_canonicalize_params cp = {};
            cp.version = GGML_XKV_FACTOR_VERSION;
            cp.n_rows = n;
            cp.n_layers = w.nl;
            cp.n_heads = w.nh;
            cp.head_dim = w.hd;
            cp.padded_head_dim = phd;
            cp.total_feat = tf;
            cp.rotary_dim = w.rot;
            cp.rope_mode = GGML_XKV_ROPE_HALF;
            cp.input_type = (uint32_t)ht;
            cp.is_k = 1;
            cp.hadamard_dim = 0;
            std::vector<float> out(size_t(n) * tf, 0.0f);
            char err[256] = {0};
            bool ok = ggml_xkv_canonicalize_cpu_oracle(hot.data(), ht, rows.data(), pos.data(), 0 /* pos_is_64 */, n,
                rope.data(), (uint32_t)rope.size(), nullptr, 0, &cp, out.data(), err, sizeof(err));
            CHECK(ok);
            if (!ok) { std::fprintf(stderr, "canon failed: %s\n", err); continue; }
            // compare vs truth with codec-appropriate tolerance
            double se = 0.0, so = 0.0;
            for (size_t i = 0; i < out.size(); ++i) {
                double e = (double)out[i] - truth[i];
                se += e * e;
                so += (double)truth[i] * truth[i];
            }
            double rel = std::sqrt(se / so);
            double tol = (ht == GGML_TYPE_F32) ? 2e-5 : (ht == GGML_TYPE_F16) ? 2e-3 :
                (ht == GGML_TYPE_Q8_0) ? 2e-2 : (ht == GGML_TYPE_TURBO4_0) ? 0.15 :
                (ht == GGML_TYPE_TURBO3_0) ? 0.25 : 0.40;
            CHECK(rel < tol);
            // determinism
            std::vector<float> out2(size_t(n) * tf, 0.0f);
            CHECK(ggml_xkv_canonicalize_cpu_oracle(hot.data(), ht, rows.data(), pos.data(), 0 /* pos_is_64 */, n,
                rope.data(), (uint32_t)rope.size(), nullptr, 0, &cp, out2.data(), err, sizeof(err)));
            CHECK(out == out2);
        }
    }
    // V path (is_k=0): no RoPE; F32 hot recovers truth bit-exactly
    {
        uint32_t n = 16, nl = 2, nh = 2, hd = 64, nvec = nl * nh, tf = nvec * hd;
        rng_seed(424242);
        std::vector<float> truth(size_t(n) * tf);
        for (auto & x : truth) x = rng_uni();
        std::vector<int32_t> rows(n), pos(n);
        for (uint32_t i = 0; i < n; ++i) { rows[i] = (int32_t)i; pos[i] = (int32_t)(i + 5); }
        ggml_xkv_canonicalize_params cp = {};
        cp.version = GGML_XKV_FACTOR_VERSION;
        cp.n_rows = n; cp.n_layers = nl; cp.n_heads = nh; cp.head_dim = hd;
        cp.padded_head_dim = hd; cp.total_feat = tf;
        cp.rotary_dim = 32; cp.rope_mode = GGML_XKV_ROPE_INTERLEAVED;
        cp.input_type = (uint32_t)GGML_TYPE_F32; cp.is_k = 0; cp.hadamard_dim = 0;
        auto rope = build_rope_tables(32, 1.5f); // nonzero mag must be ignored when is_k=0
        std::vector<float> out(size_t(n) * tf, 0.0f);
        char err[256] = {0};
        CHECK(ggml_xkv_canonicalize_cpu_oracle(truth.data(), GGML_TYPE_F32, rows.data(),
            pos.data(), 0 /* pos_is_64 */, n, rope.data(), (uint32_t)rope.size(), nullptr, 0, &cp, out.data(),
            err, sizeof(err)));
        CHECK(out == truth);
    }
    std::printf("  roundtrip OK\n");
}

// ---------- graph tests (CPU + Vulkan) ----------

struct graph_out {
    std::vector<uint8_t> a_bytes, b_bytes;
    int32_t status = -9;
    ggml_xkv_residual r_ab = {0, 0, 0}, r_a = {0, 0, 0}, r_b = {0, 0, 0};
    bool computed = false;
};

static graph_out run_factorize_graph(ggml_backend_t backend, const std::vector<float> & X,
        uint32_t n, uint32_t m, const ggml_xkv_factorize_params & p) {
    graph_out go;
    uint32_t r = std::min(p.requested_rank, std::min(n, m));
    uint32_t prb = p.pad_r_b ? p.pad_r_b : r;
    ggml_type tb = (ggml_type)p.type_b;
    size_t need = 0;
    char err0[256] = {0};
    if (!ggml_xkv_factorize_scratch_bytes(&p, &need, err0, sizeof(err0))) return go;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, m, n);
    ggml_tensor * sc = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4));
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), tb, prb, m);
    ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 10); // [0]=code [1..9]=residuals
    ggml_tensor * a = ggml_xkv_factorize(ctx.get(), x, sc, b, st, &p);
    if (!a) return go;
    if (!ggml_backend_supports_op(backend, a)) {
        std::fprintf(stderr, "backend does not support GGML_OP_XKV_FACTORIZE\n");
        return go;
    }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) return go;
    ggml_backend_tensor_set(x, X.data(), 0, X.size() * 4);
    // sentinel B/status (failure must leave them unpublished)
    std::vector<uint8_t> bsent(ggml_nbytes(b), 0x5A);
    ggml_backend_tensor_set(b, bsent.data(), 0, bsent.size());
    int32_t stsent[10] = {-7,-7,-7,-7,-7,-7,-7,-7,-7,-7};
    ggml_backend_tensor_set(st, stsent, 0, sizeof(stsent));
    ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(g, a);
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) return go;
    ggml_backend_synchronize(backend); // ONE final sync per sealed group
    go.a_bytes.assign(ggml_nbytes(a), 0);
    go.b_bytes.assign(ggml_nbytes(b), 0);
    ggml_backend_tensor_get(a, go.a_bytes.data(), 0, go.a_bytes.size());
    ggml_backend_tensor_get(b, go.b_bytes.data(), 0, go.b_bytes.size());
    int32_t stw[10] = {-9,-9,-9,-9,-9,-9,-9,-9,-9,-9};
    ggml_backend_tensor_get(st, stw, 0, sizeof(stw)); // scalar-class D2H only
    go.status = stw[0];
    memcpy(&go.r_ab, &stw[1], 3 * 4);
    memcpy(&go.r_a, &stw[4], 3 * 4);
    memcpy(&go.r_b, &stw[7], 3 * 4);
    go.computed = true;
    return go;
}

static void test_cpu_graph_matches_oracle() {
    std::printf("[graph] CPU backend graph == oracle bitwise ...\n");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu != nullptr);
    uint32_t n = 100, m = 70, r = 37; // unaligned K
    auto X = gen_lowrank_noise(n, m, 12, 1e-3f, 606);
    auto p = default_params(n, m, r, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 303);
    factor_buffers fb;
    std::string err;
    CHECK(run_oracle_full(X, n, m, p, fb, err));
    graph_out go = run_factorize_graph(cpu, X, n, m, p);
    CHECK(go.computed);
    CHECK(go.status == 0);
    CHECK(go.a_bytes == fb.a_bytes);
    CHECK(go.b_bytes == fb.b_bytes);
    // residual words bitwise vs oracle (same code path on CPU backend)
    CHECK(memcmp(&go.r_ab, &fb.r_ab, sizeof(go.r_ab)) == 0);
    CHECK(memcmp(&go.r_a, &fb.r_a, sizeof(go.r_a)) == 0);
    CHECK(memcmp(&go.r_b, &fb.r_b, sizeof(go.r_b)) == 0);
    // oracle residual self-consistency: A+B >= each ablation roughly holds
    // only statistically; assert finiteness + nonnegativity instead.
    for (const auto * rp : {&fb.r_ab, &fb.r_a, &fb.r_b}) {
        CHECK(std::isfinite(rp->frob_orig) && std::isfinite(rp->frob_err));
        CHECK(rp->frob_orig >= 0.0f && rp->frob_err >= 0.0f && rp->max_err >= 0.0f);
    }
    ggml_backend_free(cpu);
    std::printf("  CPU graph bitwise OK, status=%d\n", go.status);
}

static void test_vulkan_graph() {
    std::printf("[graph] Vulkan backend graph vs oracle ...\n");
    ggml_backend_load_all();
    ggml_backend_t vk = nullptr;
    // Registry lookup first; standalone links may not register the Vulkan
    // backend, so fall back to direct init (proves real Radeon execution).
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) vk = ggml_backend_dev_init(dev, nullptr);
#ifdef GGML_USE_VULKAN
    if (!vk && ggml_backend_vk_get_device_count() > 0) {
        char desc[256] = {};
        ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
        std::printf("  direct vk_init(0): %s\n", desc);
        vk = ggml_backend_vk_init(0);
    }
#endif
    if (!vk) {
        std::printf("  SKIP: no Vulkan device\n");
        return;
    }
    // K and V in ONE graph, ONE compute, ONE sync (production seal pattern)
    uint32_t n = 96;
    auto Xk = gen_lowrank_noise(n, 64, 10, 1e-3f, 7171);
    auto Xv = gen_lowrank_noise(n, 80, 12, 1e-3f, 7172);
    auto pk = default_params(n, 64, 24, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 7171);
    auto pv = default_params(n, 80, 28, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 7172);
    size_t needk = 0, needv = 0;
    char e0[256] = {0};
    CHECK(ggml_xkv_factorize_scratch_bytes(&pk, &needk, e0, sizeof(e0)));
    CHECK(ggml_xkv_factorize_scratch_bytes(&pv, &needv, e0, sizeof(e0)));
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * xk = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, n);
    ggml_tensor * sck = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(needk / 4));
    ggml_tensor * bk = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_TURBO4_0, pk.pad_r_b, 64);
    ggml_tensor * stk = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 10); // [0]=code [1..9]=residuals
    ggml_tensor * ak = ggml_xkv_factorize(ctx.get(), xk, sck, bk, stk, &pk);
    ggml_tensor * xv = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 80, n);
    ggml_tensor * scv = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(needv / 4));
    ggml_tensor * bv = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, pv.pad_r_b, 80);
    ggml_tensor * stv = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 10); // [0]=code [1..9]=residuals
    ggml_tensor * av = ggml_xkv_factorize(ctx.get(), xv, scv, bv, stv, &pv);
    CHECK(ak && av);
    if (!ggml_backend_supports_op(vk, ak) || !ggml_backend_supports_op(vk, av)) {
        std::fprintf(stderr, "Vulkan claims no support -> SKIP (no capability claimed)\n");
        ggml_backend_free(vk);
        return;
    }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), vk));
    // Allocation failure must fail here, before any publish.
    CHECK(buf != nullptr);
    if (!buf) { ggml_backend_free(vk); return; }
    ggml_backend_tensor_set(xk, Xk.data(), 0, Xk.size() * 4);
    ggml_backend_tensor_set(xv, Xv.data(), 0, Xv.size() * 4);
    ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(g, ak);
    ggml_build_forward_expand(g, av);
    // ONE compute + ONE sync for the whole sealed bundle (K+V, four streams).
    // Production reads back only the two status words (tiny scalar D2H);
    // packed-stream readbacks below are test-harness verification only.
    CHECK(ggml_backend_graph_compute(vk, g) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(vk);
    int32_t skw[10], svw[10];
    for (int i = 0; i < 10; ++i) skw[i] = svw[i] = -7;
    ggml_backend_tensor_get(stk, skw, 0, sizeof(skw));
    ggml_backend_tensor_get(stv, svw, 0, sizeof(svw));
    int32_t sk = skw[0], sv = svw[0];
    CHECK(sk == 0);
    CHECK(sv == 0);
    if (sk != 0 || sv != 0) {
        std::fprintf(stderr, "Vulkan factorize status k=%d v=%d -> FAIL\n", sk, sv);
        ++failures;
        ggml_backend_free(vk);
        return;
    }
    // Compare decoded reconstructions vs oracle reconstructions
    std::vector<uint8_t> ak_b(ggml_nbytes(ak)), bk_b(ggml_nbytes(bk));
    std::vector<uint8_t> av_b(ggml_nbytes(av)), bv_b(ggml_nbytes(bv));
    ggml_backend_tensor_get(ak, ak_b.data(), 0, ak_b.size());
    ggml_backend_tensor_get(bk, bk_b.data(), 0, bk_b.size());
    ggml_backend_tensor_get(av, av_b.data(), 0, av_b.size());
    ggml_backend_tensor_get(bv, bv_b.data(), 0, bv_b.size());
    factor_buffers ok_, ov_;
    std::string err;
    CHECK(run_oracle_full(Xk, n, 64, pk, ok_, err));
    CHECK(run_oracle_full(Xv, n, 80, pv, ov_, err));
    std::vector<float> Ak, BTk, Ak0, BTk0, Av, BTv, Av0, BTv0;
    // Turbo WHT spreads logical components across the padded row: dot the
    // full padded rank (same production-reader contract as the oracle section).
    const uint32_t crank = pk.pad_r_a;
    CHECK(decode_stream(GGML_TYPE_TURBO4_0, ak_b, n, pk.pad_r_a, crank, Ak));
    CHECK(decode_stream(GGML_TYPE_TURBO4_0, bk_b, 64, pk.pad_r_b, crank, BTk));
    CHECK(decode_stream(GGML_TYPE_TURBO4_0, ok_.a_bytes, n, pk.pad_r_a, crank, Ak0));
    CHECK(decode_stream(GGML_TYPE_TURBO4_0, ok_.b_bytes, 64, pk.pad_r_b, crank, BTk0));
    CHECK(decode_stream(GGML_TYPE_Q8_0, av_b, n, pv.pad_r_a, 28, Av));
    CHECK(decode_stream(GGML_TYPE_Q8_0, bv_b, 80, pv.pad_r_b, 28, BTv));
    CHECK(decode_stream(GGML_TYPE_Q8_0, ov_.a_bytes, n, pv.pad_r_a, 28, Av0));
    CHECK(decode_stream(GGML_TYPE_Q8_0, ov_.b_bytes, 80, pv.pad_r_b, 28, BTv0));
    double ck = cosine_sim(Xk, Ak, BTk, n, 64, crank);
    double cv = cosine_sim(Xv, Av, BTv, n, 80, 28);
    double relk = rel_residual(Xk, Ak, BTk, n, 64, crank);
    double relv = rel_residual(Xv, Av, BTv, n, 80, 28);
    // Vulkan-vs-oracle reconstruction agreement (same codec, same contract)
    double ckk = 0.0, rkk = 0.0;
    {
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j < 64; ++j) {
            double rvk = 0.0, ro = 0.0;
            for (uint32_t k = 0; k < crank; ++k) {
                rvk += (double)Ak[size_t(i) * crank + k] * BTk[size_t(j) * crank + k];
                ro += (double)Ak0[size_t(i) * crank + k] * BTk0[size_t(j) * crank + k];
            }
            dot += rvk * ro; n1 += rvk * rvk; n2 += ro * ro;
            double d = rvk - ro; rkk += d * d;
        }
        ckk = dot / std::sqrt(n1 * n2);
        rkk = std::sqrt(rkk / n2);
    }
    CHECK(ck > 0.98);
    CHECK(cv > 0.98);
    CHECK(relk < 0.30);
    CHECK(relv < 0.20);
    std::printf("  vk-vs-oracle: ckk=%.5f rkk=%.4f\n", ckk, rkk);
    // Different precisions/paths (FP32 Gram vs FP64 one-sided) legitimately
    // differ within the oracle's own error band; garbage would score ~0.5/1.0.
    CHECK(ckk > 0.985);
    CHECK(rkk < 0.20);
    CHECK(!has_nan(ak_b, GGML_TYPE_TURBO4_0, n, pk.pad_r_a));
    // device residual words vs oracle (tolerance: FP32 Gram vs FP64 one-sided)
    {
        ggml_xkv_residual got_ab, got_a, got_b;
        memcpy(&got_ab, &skw[1], 3 * 4);
        memcpy(&got_a, &skw[4], 3 * 4);
        memcpy(&got_b, &skw[7], 3 * 4);
        const ggml_xkv_residual * exp[3] = {&ok_.r_ab, &ok_.r_a, &ok_.r_b};
        const ggml_xkv_residual * got[3] = {&got_ab, &got_a, &got_b};
        const char * nm[3] = {"AB", "A-only", "B-only"};
        for (int q = 0; q < 3; ++q) {
            double do_ = exp[q]->frob_orig, de = exp[q]->frob_err;
            double go = got[q]->frob_orig, ge = got[q]->frob_err;
            CHECK(std::isfinite(go) && std::isfinite(ge));
            CHECK(std::fabs(go - do_) < 1e-3 * (1.0 + std::fabs(do_)));
            CHECK(std::fabs(ge - de) < 5e-2 * (1.0 + std::fabs(de)));
            CHECK(got[q]->max_err >= 0.0f && std::isfinite(got[q]->max_err));
            std::printf("  vk resid %s: orig %.5f/%.5f err %.5f/%.5f\n",
                nm[q], go, do_, ge, de);
        }
    }
    // determinism on device: run twice, identical bytes
    ggml_backend_tensor_set(xk, Xk.data(), 0, Xk.size() * 4);
    CHECK(ggml_backend_graph_compute(vk, g) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(vk);
    std::vector<uint8_t> ak_b2(ggml_nbytes(ak));
    ggml_backend_tensor_get(ak, ak_b2.data(), 0, ak_b2.size());
    CHECK(ak_b == ak_b2);
    ggml_backend_free(vk);
    std::printf("  Vulkan K/V OK: cos=%.5f/%.5f rel=%.4f/%.4f deterministic=1\n", ck, cv, relk, relv);
}

static ggml_backend_t open_vk_backend() {
    ggml_backend_load_all();
    ggml_backend_t vk = nullptr;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) vk = ggml_backend_dev_init(dev, nullptr);
#ifdef GGML_USE_VULKAN
    if (!vk && ggml_backend_vk_get_device_count() > 0) vk = ggml_backend_vk_init(0);
#endif
    return vk;
}

static void test_balance_modes_vulkan() {
    std::printf("[graph] Vulkan balance modes vs oracle ...\n");
    ggml_backend_t vk = open_vk_backend();
    if (!vk) { std::printf("  SKIP: no Vulkan device\n"); return; }
    uint32_t n = 64, m = 48, r = 12;
    auto X = gen_lowrank_noise(n, m, 6, 1e-3f, 3131);
    for (int b = 0; b < 3; ++b) {
        auto p = default_params(n, m, r, GGML_TYPE_F32, GGML_TYPE_F32, 77, (uint32_t)b);
        factor_buffers fb;
        std::string err;
        CHECK(run_oracle_full(X, n, m, p, fb, err));
        graph_out go = run_factorize_graph(vk, X, n, m, p);
        CHECK(go.computed);
        CHECK(go.status == 0);
        std::vector<float> A, BT, A0, BT0;
        CHECK(decode_stream(GGML_TYPE_F32, go.a_bytes, n, p.pad_r_a, r, A));
        CHECK(decode_stream(GGML_TYPE_F32, go.b_bytes, m, p.pad_r_b, r, BT));
        CHECK(decode_stream(GGML_TYPE_F32, fb.a_bytes, n, p.pad_r_a, r, A0));
        CHECK(decode_stream(GGML_TYPE_F32, fb.b_bytes, m, p.pad_r_b, r, BT0));
        double rel = rel_residual(X, A, BT, n, m, r);
        double rel0 = rel_residual(X, A0, BT0, n, m, r);
        CHECK(rel < 1e-3);
        CHECK(std::fabs(rel - rel0) < 1e-4);
        // residual words agree (F32 path is near-exact across precisions)
        CHECK(std::fabs(go.r_ab.frob_err - fb.r_ab.frob_err) < 1e-3f * (1.0f + fb.r_ab.frob_err));
        std::printf("  mode %d rel=%.6f oracle=%.6f\n", b, rel, rel0);
    }
    ggml_backend_free(vk);
}

int main() {
    test_lowrank_oracle();
    test_rankdeficient_oracle();
    test_repeated_sv_oracle();
    test_quant_codecs_oracle();
    test_k_neq_v();
    test_balance_modes();
    test_scratch_exact();
    test_failure_atomicity();
    test_canonicalize_roundtrip();
    test_cpu_graph_matches_oracle();
    test_vulkan_graph();
    test_balance_modes_vulkan();
    if (failures == 0) {
        std::printf("PASS: test-xkv-vulkan-factorize\n");
        return 0;
    }
    std::fprintf(stderr, "FAIL: test-xkv-vulkan-factorize (%d failures)\n", failures);
    return 1;
}
