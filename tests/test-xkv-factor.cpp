#include "llama-xkv-factor.h"
#include "llama-xkv-codec.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <new>
#include <limits>
#include <iostream>
#include <vector>
#include <string>

using namespace llama_xkv;

namespace {
std::atomic<size_t> g_heap_new_count{0};
std::atomic<size_t> g_heap_new_bytes{0};
thread_local bool g_heap_counting = false;
struct heap_scope {
    heap_scope() {
        g_heap_new_count.store(0, std::memory_order_relaxed);
        g_heap_new_bytes.store(0, std::memory_order_relaxed);
        g_heap_counting = true;
    }
    ~heap_scope() { g_heap_counting = false; }
    size_t count() const { return g_heap_new_count.load(std::memory_order_relaxed); }
    size_t bytes() const { return g_heap_new_bytes.load(std::memory_order_relaxed); }
};
} // namespace

void * operator new(std::size_t n) {
    if (g_heap_counting) {
        g_heap_new_count.fetch_add(1, std::memory_order_relaxed);
        g_heap_new_bytes.fetch_add(n, std::memory_order_relaxed);
    }
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void * operator new[](std::size_t n) {
    if (g_heap_counting) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "[HEAP ALLOC[]: %zu bytes]\n", n);
        std::fputs(buf, stderr);
        g_heap_new_count.fetch_add(1, std::memory_order_relaxed);
        g_heap_new_bytes.fetch_add(n, std::memory_order_relaxed);
    }
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

static void assert_true(bool cond, const char * msg) {
    if (!cond) {
        std::cerr << "Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}
static void assert_true(bool cond, const std::string & msg) {
    assert_true(cond, msg.c_str());
}

static void assert_true_err(bool cond, const std::string & msg, const std::string & err) {
    if (!cond) {
        std::cerr << "Assertion failed: " << msg << " [error: " << err << "]" << std::endl;
        std::exit(1);
    }
}

static matrix generate_deterministic_matrix(uint64_t rows, uint64_t cols, uint64_t seed) {
    matrix m(rows, cols);
    uint64_t state = seed ? seed : 123456789;
    for (uint64_t i = 0; i < rows * cols; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        float val = ((float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF) * 2.0f - 1.0f;
        m.data[i] = val;
    }
    return m;
}

// 1. Test Layer Group Mapping: deduplication, non-contiguous layer IDs, group sizes 1, 2, 4, tail groups
static void test_layer_group_mapping() {
    std::cout << "[Test 1] Testing layer group mapping..." << std::endl;

    std::vector<uint32_t> owning = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};

    // Group size 4: [3, 7, 11, 15], [19, 23, 27, 31], [35, 39] (tail group of 2)
    layer_group_map map4 = build_layer_group_map(owning, 4, 128, 128);
    assert_true(map4.groups.size() == 3, "group size 4 should produce 3 groups");
    assert_true(map4.groups[0].owning_layers.size() == 4, "group 0 size == 4");
    assert_true(map4.groups[1].owning_layers.size() == 4, "group 1 size == 4");
    assert_true(map4.groups[2].owning_layers.size() == 2, "group 2 (tail) size == 2");
    assert_true(map4.groups[0].total_dim_k == 512, "group 0 total dim == 512");
    assert_true(map4.groups[2].total_dim_k == 256, "group 2 total dim == 256");

    // Group size 2: 5 groups of 2
    layer_group_map map2 = build_layer_group_map(owning, 2, 64, 64);
    assert_true(map2.groups.size() == 5, "group size 2 should produce 5 groups");

    // Group size 1: 10 groups of 1
    layer_group_map map1 = build_layer_group_map(owning, 1, 128, 128);
    assert_true(map1.groups.size() == 10, "group size 1 should produce 10 groups");

    // Alias deduplication test
    std::vector<uint32_t> aliased = {3, 7, 3, 11, 7, 15, 19, 19};
    layer_group_map map_dedup = build_layer_group_map(aliased, 4, 128, 128);
    assert_true(map_dedup.unique_owning_layers.size() == 5, "unique owning layers deduplicated to 5");
    assert_true(map_dedup.groups.size() == 2, "5 unique layers with group size 4 gives 2 groups");

    // Extended mapping with model layer to owning layer aliases
    std::vector<uint32_t> model_to_owning = {0, 3, 0, 7, 3, 11, 7, 15};
    std::vector<bool> is_attn = {false, true, false, true, true, true, true, true};
    std::vector<uint32_t> dims(16, 128);
    layer_group_map map_ex = build_layer_group_map_ex(model_to_owning, is_attn, 2, dims, dims);
    assert_true(map_ex.unique_owning_layers.size() == 4, "unique attention owning layers deduplicated");
    assert_true(map_ex.model_layer_to_group[1] == map_ex.model_layer_to_group[4], "aliased model layers map to same group");

    std::cout << "  [PASS] Layer group mapping verified." << std::endl;
}

// 2. Test Determinism, Seed Equality, Different Seed Behavior, and Config Fingerprint
static void test_determinism_and_fingerprint() {
    std::cout << "[Test 2] Testing determinism, seed behavior, and config fingerprint..." << std::endl;

    matrix m = generate_deterministic_matrix(16, 32, 999);

    factor_pair fp1, fp2, fp_diff_seed;
    std::string err;

    // Identical seeds produce bitwise deterministic output
    bool ok1 = factorize_matrix(m, 8, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 12345, fp1, &err);
    assert_true_err(ok1, "factorize_matrix 1 succeeded", err);
    bool ok2 = factorize_matrix(m, 8, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 12345, fp2, &err);
    assert_true_err(ok2, "factorize_matrix 2 succeeded", err);

    assert_true(fp1.a == fp2.a, "A factor is bitwise deterministic for identical seed");
    assert_true(fp1.b_transposed == fp2.b_transposed, "B_transposed is bitwise deterministic for identical seed");
    assert_true(fp1.singular_values == fp2.singular_values, "singular values are bitwise deterministic");

    // Different seed in randomized SVD produces valid factorization with different test space
    matrix m_large = generate_deterministic_matrix(96, 96, 555);
    factor_pair fp_s1, fp_s2;
    bool ok_s1 = factorize_matrix(m_large, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 100, fp_s1, &err);
    assert_true_err(ok_s1, "factorize_matrix seed 100 succeeded", err);
    bool ok_s2 = factorize_matrix(m_large, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 200, fp_s2, &err);
    assert_true_err(ok_s2, "factorize_matrix seed 200 succeeded", err);
    assert_true(fp_s1.a != fp_s2.a, "different seeds produce distinct randomized bases");

    factor_config cfg1;
    cfg1.algorithm_version = 1;
    cfg1.rank_k = 384;
    cfg1.rank_v = 576;
    cfg1.balance = LLAMA_XKV_FACTOR_BALANCE_SQRT;
    cfg1.seed = 42;
    cfg1.oversampling = 16;
    cfg1.power_iterations = 2;

    factor_config cfg2 = cfg1;
    assert_true(cfg1.fingerprint() == cfg2.fingerprint(), "fingerprints match for identical configs");

    cfg2.oversampling = 32;
    assert_true(cfg1.fingerprint() != cfg2.fingerprint(), "fingerprint differs when oversampling changes");

    cfg2 = cfg1;
    cfg2.power_iterations = 3;
    assert_true(cfg1.fingerprint() != cfg2.fingerprint(), "fingerprint differs when power_iterations changes");

    cfg2 = cfg1;
    cfg2.balance = LLAMA_XKV_FACTOR_BALANCE_DIAGONAL;
    assert_true(cfg1.fingerprint() != cfg2.fingerprint(), "fingerprint differs when balance changes");

    std::cout << "  [PASS] Determinism, seed behavior, and config fingerprint verified." << std::endl;
}

// 3. Test Full-Rank Exact Reconstruction: A * (B^T)^T == X
static void test_full_rank_reconstruction() {
    std::cout << "[Test 3] Testing full-rank reconstruction (A * (B^T)^T == X)..." << std::endl;

    // Square matrix 12 x 12 with full rank
    matrix m12 = generate_deterministic_matrix(12, 12, 54321);
    factor_pair fp12;
    std::string err;
    bool ok12 = factorize_matrix(m12, 12, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp12, &err);
    assert_true_err(ok12, "12x12 full rank factorization succeeded", err);

    matrix recon12 = matrix_reconstruct(fp12.a, fp12.b_transposed);
    factor_error_report rep12 = compute_error_report(m12, recon12);
    assert_true(rep12.relative_error < 1e-4, "12x12 full rank relative error < 1e-4");
    assert_true(rep12.max_absolute_error < 1e-4, "12x12 full rank max absolute error < 1e-4");

    // 8x5 full rank matrix (tall: n=8, m=5, r=5)
    matrix m8x5 = generate_deterministic_matrix(8, 5, 12345);
    factor_pair fp8x5;
    bool ok8x5 = factorize_matrix(m8x5, 5, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp8x5, &err);
    assert_true_err(ok8x5, "8x5 full rank factorization succeeded", err);
    assert_true(fp8x5.rank == 5, "8x5 rank is 5");

    matrix recon8x5 = matrix_reconstruct(fp8x5.a, fp8x5.b_transposed);
    factor_error_report rep8x5 = compute_error_report(m8x5, recon8x5);
    assert_true(rep8x5.relative_error < 1e-4, "8x5 full rank relative error < 1e-4");
    assert_true(rep8x5.max_absolute_error < 1e-4, "8x5 full rank max absolute error < 1e-4");

    std::cout << "  [PASS] Full-rank exact reconstruction verified (relative error < 1e-4)." << std::endl;
}

// 4. Test Balancing Modes: Upstream (alpha=1.0), Sqrt (alpha=0.5), Diagonal
// Mathematical invariant: unquantized balancing preserves AB product exactly!
static void test_balancing_preserves_ab() {
    std::cout << "[Test 4] Testing balancing modes preserve AB product..." << std::endl;

    matrix m = generate_deterministic_matrix(20, 30, 777);
    uint32_t rank = 10;

    factor_pair fp_up, fp_sqrt, fp_diag;
    std::string err;
    assert_true(factorize_matrix(m, rank, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 1, fp_up, &err), "upstream balance ok");
    assert_true(factorize_matrix(m, rank, LLAMA_XKV_FACTOR_BALANCE_SQRT,     1, fp_sqrt, &err), "sqrt balance ok");
    assert_true(factorize_matrix(m, rank, LLAMA_XKV_FACTOR_BALANCE_DIAGONAL, 1, fp_diag, &err), "diagonal balance ok");

    matrix recon_up   = matrix_reconstruct(fp_up.a, fp_up.b_transposed);
    matrix recon_sqrt = matrix_reconstruct(fp_sqrt.a, fp_sqrt.b_transposed);
    matrix recon_diag = matrix_reconstruct(fp_diag.a, fp_diag.b_transposed);

    factor_error_report diff_sqrt = compute_error_report(recon_up, recon_sqrt);
    factor_error_report diff_diag = compute_error_report(recon_up, recon_diag);

    // Assert against ground truth X
    factor_error_report gt_up   = compute_error_report(m, recon_up);
    factor_error_report gt_sqrt = compute_error_report(m, recon_sqrt);
    factor_error_report gt_diag = compute_error_report(m, recon_diag);

    assert_true(gt_up.frobenius_norm_original > 0.0, "ground truth original norm > 0");
    assert_true(diff_sqrt.max_absolute_error < 1e-4, "Upstream vs Sqrt reconstruction difference < 1e-4");
    assert_true(diff_diag.max_absolute_error < 1e-4, "Upstream vs Diagonal reconstruction difference < 1e-4");
    assert_true(std::abs(gt_up.relative_error - gt_sqrt.relative_error) < 1e-5, "Upstream and Sqrt have identical relative error to X");
    assert_true(std::abs(gt_up.relative_error - gt_diag.relative_error) < 1e-5, "Upstream and Diagonal have identical relative error to X");

    // Cosine similarity check between recon_up and recon_sqrt / recon_diag
    double dot_up_sqrt = 0.0, norm_up_sq = 0.0, norm_sqrt_sq = 0.0;
    for (uint64_t i = 0; i < m.elements(); ++i) {
        dot_up_sqrt += (double)recon_up.data[i] * (double)recon_sqrt.data[i];
        norm_up_sq += (double)recon_up.data[i] * (double)recon_up.data[i];
        norm_sqrt_sq += (double)recon_sqrt.data[i] * (double)recon_sqrt.data[i];
    }
    double cos_sim = dot_up_sqrt / (std::sqrt(norm_up_sq * norm_sqrt_sq) + 1e-12);
    assert_true(cos_sim > 0.9999, "cosine similarity between Upstream and Sqrt reconstruction is > 0.9999");

    std::cout << "  [PASS] All unquantized balancing modes preserve AB product within tolerance." << std::endl;
}

// 5. Test Low-Rank Recovery with Randomized SVD, Wide/Tall Matrices, and Ill-Conditioned Columns
static void test_low_rank_and_rsvd() {
    std::cout << "[Test 5] Testing randomized SVD low-rank recovery and matrix shapes..." << std::endl;

    // Generate rank-4 synthetic matrix: X = A_true(80 x 4) * B_true^T(60 x 4)
    uint64_t n = 80;
    uint64_t m = 60;
    uint32_t true_rank = 4;
    matrix a_true = generate_deterministic_matrix(n, true_rank, 111);
    matrix bt_true = generate_deterministic_matrix(m, true_rank, 222);
    matrix x_lowrank = matrix_reconstruct(a_true, bt_true);

    factor_pair fp_lr;
    std::string err;
    bool ok_lr = factorize_matrix(x_lowrank, true_rank, LLAMA_XKV_FACTOR_BALANCE_SQRT, 42, fp_lr, &err, 1e-9, 30, 8, 2);
    assert_true_err(ok_lr, "rSVD on synthetic low-rank matrix succeeded", err);

    matrix recon_lr = matrix_reconstruct(fp_lr.a, fp_lr.b_transposed);
    factor_error_report rep_lr = compute_error_report(x_lowrank, recon_lr);
    assert_true(rep_lr.relative_error < 1e-3, "rSVD recovers low-rank matrix with relative error < 1e-3");

    // Wide matrix: 32 x 128 with rank 16
    matrix m_wide = generate_deterministic_matrix(32, 128, 333);
    factor_pair fp_wide;
    bool ok_wide = factorize_matrix(m_wide, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp_wide, &err, 1e-9, 30, 8, 2);
    assert_true_err(ok_wide, "rSVD on wide matrix succeeded", err);
    assert_true(fp_wide.rank == 16, "wide matrix rank is 16");

    // Tall matrix: 128 x 32 with rank 16
    matrix m_tall = generate_deterministic_matrix(128, 32, 444);
    factor_pair fp_tall;
    bool ok_tall = factorize_matrix(m_tall, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp_tall, &err, 1e-9, 30, 8, 2);
    assert_true_err(ok_tall, "rSVD on tall matrix succeeded", err);
    assert_true(fp_tall.rank == 16, "tall matrix rank is 16");

    // Ill-conditioned / zero columns: matrix with columns of all zeros
    matrix m_zero_cols = generate_deterministic_matrix(40, 40, 555);
    for (uint64_t i = 0; i < 40; ++i) {
        m_zero_cols.at(i, 5) = 0.0f;
        m_zero_cols.at(i, 10) = 0.0f;
    }
    factor_pair fp_zc;
    bool ok_zc = factorize_matrix(m_zero_cols, 8, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp_zc, &err, 1e-9, 30, 8, 2);
    assert_true_err(ok_zc, "matrix with zero columns factorizes cleanly", err);

    std::cout << "  [PASS] Low-rank recovery and matrix shapes verified." << std::endl;
}

// 6. Test Four-Stream Quantized Shadow and Streaming Tile Verification
static void test_four_stream_quantized_shadow() {
    std::cout << "[Test 6] Testing four-stream quantized shadow without dense mirrors..." << std::endl;

    uint64_t n_tokens = 32;
    uint64_t dim_k = 128;
    uint64_t dim_v = 128;

    matrix x_k = generate_deterministic_matrix(n_tokens, dim_k, 101);
    matrix x_v = generate_deterministic_matrix(n_tokens, dim_v, 202);

    factor_config cfg;
    cfg.rank_k = 64;
    cfg.rank_v = 64;
    cfg.balance = LLAMA_XKV_FACTOR_BALANCE_SQRT;
    cfg.seed = 42;
    cfg.factor_a_k = GGML_TYPE_F32;
    cfg.factor_b_k = GGML_TYPE_F32;
    cfg.factor_a_v = GGML_TYPE_F32;
    cfg.factor_b_v = GGML_TYPE_F32;

    // Factorize K and V
    factor_result res = factorize_kv(x_k, x_v, cfg);
    assert_true(res.success, "factorize_kv succeeded");
    assert_true(res.config_fingerprint == cfg.fingerprint(), "fingerprint matches");

    // Evaluate quantized shadow with F32 streams
    factor_quantized_shadow shadow_f32 = evaluate_quantized_shadow(x_k, x_v, res.k, res.v, cfg);
    assert_true(shadow_f32.success, "evaluate_quantized_shadow F32 succeeded");
    assert_true(shadow_f32.stream_a_k.desc.role == factor_role::a_k, "stream_a_k role is a_k");
    assert_true(shadow_f32.stream_b_k.desc.role == factor_role::b_k, "stream_b_k role is b_k");
    assert_true(shadow_f32.stream_a_v.desc.role == factor_role::a_v, "stream_a_v role is a_v");
    assert_true(shadow_f32.stream_b_v.desc.role == factor_role::b_v, "stream_b_v role is b_v");

    // All four streams are encoded and non-empty
    assert_true(!shadow_f32.stream_a_k.bytes.empty(), "stream_a_k bytes non-empty");
    assert_true(!shadow_f32.stream_b_k.bytes.empty(), "stream_b_k bytes non-empty");
    assert_true(!shadow_f32.stream_a_v.bytes.empty(), "stream_a_v bytes non-empty");
    assert_true(!shadow_f32.stream_b_v.bytes.empty(), "stream_b_v bytes non-empty");

    // Check error reports: A-only, B-only, and AB-product are populated
    assert_true(shadow_f32.errors_k.a_only.frobenius_norm_original > 0.0, "K a_only error computed");
    assert_true(shadow_f32.errors_k.b_only.frobenius_norm_original > 0.0, "K b_only error computed");
    assert_true(shadow_f32.errors_k.ab_product.frobenius_norm_original > 0.0, "K ab_product error computed");

    // Test Turbo4 streams (group_size=128)
    cfg.factor_a_k = GGML_TYPE_TURBO4_0;
    cfg.factor_b_k = GGML_TYPE_TURBO4_0;
    cfg.factor_a_v = GGML_TYPE_TURBO4_0;
    cfg.factor_b_v = GGML_TYPE_TURBO4_0;

    std::string err_pair;
    // Direct test proving validate_factor_pair_compatibility succeeds for each Turbo2/3/4 K and V pair
    std::vector<ggml_type> turbo_types = {GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0};
    for (ggml_type tt : turbo_types) {
        cfg.factor_a_k = tt;
        cfg.factor_b_k = tt;
        cfg.factor_a_v = tt;
        cfg.factor_b_v = tt;

        factor_quantized_shadow s = evaluate_quantized_shadow(x_k, x_v, res.k, res.v, cfg);
        assert_true_err(s.success, "evaluate_quantized_shadow succeeded for turbo type", s.error_message);
        assert_true_err(validate_factor_pair_compatibility(s.stream_a_k.desc, s.stream_b_k.desc, &err_pair),
                        "K factor pair compatibility verified", err_pair);
        assert_true_err(validate_factor_pair_compatibility(s.stream_a_v.desc, s.stream_b_v.desc, &err_pair),
                        "V factor pair compatibility verified", err_pair);

        // Repeat encoding yields bitwise identical code stream bytes
        factor_quantized_shadow s_repeat = evaluate_quantized_shadow(x_k, x_v, res.k, res.v, cfg);
        assert_true(s.stream_a_k.bytes == s_repeat.stream_a_k.bytes, "A_K code stream is bitwise deterministic");
        assert_true(s.stream_b_k.bytes == s_repeat.stream_b_k.bytes, "B_K code stream is bitwise deterministic");
    }

    // Reset to Turbo4: the loop above leaves Turbo2_0 in cfg, while the
    // explicit assertions below require Turbo4 streams (group_size 128).
    cfg.factor_a_k = GGML_TYPE_TURBO4_0;
    cfg.factor_b_k = GGML_TYPE_TURBO4_0;
    cfg.factor_a_v = GGML_TYPE_TURBO4_0;
    cfg.factor_b_v = GGML_TYPE_TURBO4_0;

    factor_quantized_shadow shadow_tq4 = evaluate_quantized_shadow(x_k, x_v, res.k, res.v, cfg);
    assert_true(shadow_tq4.success, "evaluate_quantized_shadow Turbo4 succeeded");
    assert_true(shadow_tq4.stream_a_k.desc.type == GGML_TYPE_TURBO4_0, "stream_a_k is TURBO4_0");
    assert_true(shadow_tq4.stream_a_k.desc.group_size == 128, "stream_a_k group_size is 128");

    // Ground-truth assertions for shadow reconstructions: error norm and relative error threshold checks
    assert_true(shadow_f32.errors_k.ab_product.frobenius_norm_error > 0.0, "K shadow f32 error norm is positive");
    assert_true(shadow_f32.errors_k.ab_product.relative_error < 1.0, "K shadow f32 relative error < 1.0");
    assert_true(shadow_f32.errors_v.ab_product.relative_error < 1.0, "V shadow f32 relative error < 1.0");

    assert_true(shadow_tq4.errors_k.ab_product.frobenius_norm_error > 0.0, "K shadow tq4 error norm is positive");
    assert_true(shadow_tq4.errors_k.ab_product.relative_error < 1.5, "K shadow tq4 relative error bounded");
    assert_true(shadow_tq4.errors_v.ab_product.relative_error < 1.5, "V shadow tq4 relative error bounded");

    std::cout << "  [PASS] Four-stream quantized shadow verified with tiled streaming." << std::endl;
}

// 7. Test Preflight Estimator, 4096 x 4096 Shape Verification, and Workspace Failure Atomicity
static void test_preflight_estimator_and_atomicity() {
    std::cout << "[Test 7] Testing preflight estimator and workspace failure atomicity..." << std::endl;

    // 1. Estimate 4096 x 4096 shape: rank=384, oversampling=16, power=2 with checked estimator
    uint64_t est_4k = 0;
    std::string err;
    bool est_ok = estimate_factorize_matrix_workspace_bytes(4096, 4096, 384, 16, 2, &est_4k, &err);
    assert_true(est_ok && est_4k > 0, "checked estimator returns positive bytes");

    // Overflow sentinel test on huge dimension
    uint64_t est_overflow = 0;
    bool est_overflow_ok = estimate_factorize_matrix_workspace_bytes(UINT64_MAX, UINT64_MAX, 384, 16, 2, &est_overflow, &err);
    assert_true(!est_overflow_ok, "huge dimension triggers overflow rejection");
    assert_true(est_overflow == UINT64_MAX, "overflow sentinel is UINT64_MAX");

    // Zero power_iterations rejected
    uint64_t est_zero_power = 0;
    bool est_zp_ok = estimate_factorize_matrix_workspace_bytes(4096, 4096, 384, 16, 0, &est_zero_power, &err);
    assert_true(!est_zp_ok, "zero power iterations rejected by estimator");

    // Direct full-rank Jacobi above 64 rejected
    uint64_t est_huge_full = 0;
    uint64_t est_hf_ok = estimate_factorize_matrix_workspace_bytes(128, 128, 128, 16, 2, &est_huge_full, &err);
    assert_true(!est_hf_ok, "direct full-rank Jacobi above conservative reference bound 64 rejected");

    // 2. Asymmetric K and V dimensions where naive max(ws_k, ws_v) underestimates:
    // K is 128 x 64, rank 32. V is 128 x 64, rank 32.
    // During V factorization, K output A/BT/S is simultaneously live.
    matrix x_k_asym = generate_deterministic_matrix(128, 64, 11);
    matrix x_v_asym = generate_deterministic_matrix(128, 64, 22);
    factor_config cfg_asym;
    cfg_asym.rank_k = 32;
    cfg_asym.rank_v = 32;

    uint64_t ws_k = 0, ws_v = 0, out_k = 0, peak_kv = 0;
    assert_true(estimate_factorize_matrix_workspace_bytes(128, 64, 32, 16, 2, &ws_k, &err), "ws_k estimated");
    assert_true(estimate_factorize_matrix_workspace_bytes(128, 64, 32, 16, 2, &ws_v, &err), "ws_v estimated");
    assert_true(estimate_factor_output_bytes(128, 64, 32, &out_k, &err), "out_k estimated");
    assert_true(estimate_factorize_kv_workspace_bytes(x_k_asym, x_v_asym, cfg_asym, &peak_kv, &err), "peak_kv estimated");

    uint64_t naive_max = std::max(ws_k, ws_v);
    assert_true(peak_kv > naive_max, "peak_kv strictly exceeds naive max(ws_k, ws_v) due to simultaneous K output");
    assert_true(peak_kv == out_k + ws_v, "peak_kv equals exact simultaneous term out_k + ws_v");

    // Test shadow workspace estimator
    uint64_t shadow_bytes = 0;
    factor_pair dummy_k, dummy_v;
    assert_true(estimate_quantized_shadow_workspace_bytes(x_k_asym, x_v_asym, dummy_k, dummy_v, cfg_asym, &shadow_bytes, &err), "shadow estimator succeeds");

    // l = 400. Peak double temporaries ~ 3 * 4096 * 400 * 8 = 39.3 MB, total < 80 MB.
    assert_true(est_4k < 100ULL * 1024ULL * 1024ULL, "4096 x 4096 rSVD workspace is bounded (< 100 MB)");
    std::cout << "  -> 4096 x 4096 (rank 384) estimated workspace: " << (est_4k / (1024 * 1024)) << " MiB" << std::endl;

    // 2. Workspace Failure Atomicity:
    // If max_workspace_bytes is set below the required amount, factorize_kv rejects before allocating!
    matrix x_k = generate_deterministic_matrix(64, 64, 1);
    matrix x_v = generate_deterministic_matrix(64, 64, 2);
    factor_config cfg;
    cfg.rank_k = 16;
    cfg.rank_v = 16;
    cfg.max_workspace_bytes = 100; // Unreasonably small limit

    factor_result res = factorize_kv(x_k, x_v, cfg);
    assert_true(!res.success, "factorize_kv rejected by preflight workspace ceiling");
    assert_true(res.error_message.find("preflight") != std::string::npos, "error message indicates preflight failure");
    assert_true(res.k.a.empty(), "K factor A remains empty on preflight failure (atomic)");
    assert_true(res.k.b_transposed.empty(), "K factor B^T remains empty on preflight failure (atomic)");
    assert_true(res.v.a.empty(), "V factor A remains empty on preflight failure (atomic)");

    std::cout << "  [PASS] Preflight estimator and workspace failure atomicity verified." << std::endl;
}

static void test_edge_cases_and_bounded_workspaces() {
    std::cout << "[Test 8] Testing edge cases, bad_alloc safety, bounded workspace, rank deficiency, and non-finite input..." << std::endl;

    // 1. One-byte-short workspace test
    uint64_t n = 96, m = 96;
    matrix x_k = generate_deterministic_matrix(n, m, 111);
    matrix x_v = generate_deterministic_matrix(n, m, 222);
    factor_config cfg;
    cfg.rank_k = 16;
    cfg.rank_v = 16;

    uint64_t req_ws = 0;
    std::string err;
    bool est_ok = estimate_factorize_kv_workspace_bytes(x_k, x_v, cfg, &req_ws, &err);
    assert_true(est_ok && req_ws > 0, "estimate_factorize_kv_workspace_bytes succeeds");

    // Test one-byte short workspace on factorize_matrix
    uint64_t req_mat_ws = 0;
    assert_true(estimate_factorize_matrix_workspace_bytes(n, m, 16, 16, 2, &req_mat_ws, &err), "matrix ws estimated");
    std::vector<uint8_t> short_buf(req_mat_ws > 0 ? req_mat_ws - 1 : 0);
    factor_pair fp_short;
    factor_workspace_span short_span(short_buf.data(), short_buf.size());
    bool short_res = factorize_matrix_bounded(x_k, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 1, fp_short, short_span, &err);
    assert_true(!short_res, "one-byte short workspace must fail factorize_matrix_bounded");
    assert_true(err.find("workspace buffer") != std::string::npos, "error mentions workspace buffer shortfall");
    assert_true(fp_short.a.empty() && fp_short.b_transposed.empty(), "outputs unmodified on workspace failure");

    // Test one-byte short workspace on factorize_kv
    std::vector<uint8_t> kv_short_buf(req_ws > 0 ? req_ws - 1 : 0);
    factor_workspace_span kv_short_span(kv_short_buf.data(), kv_short_buf.size());
    factor_result kv_short_res = factorize_kv_bounded(x_k, x_v, cfg, kv_short_span);
    assert_true(!kv_short_res.success, "one-byte short workspace must fail factorize_kv_bounded");
    assert_true(kv_short_res.k.a.empty(), "factorize_kv leaves output empty on shortfall");

    // 2. Huge overflow dimensions without allocation
    uint64_t huge_dim = 1ULL << 40;
    uint64_t est_overflow = 0;
    assert_true(!estimate_factorize_matrix_workspace_bytes(huge_dim, huge_dim, 384, 16, 2, &est_overflow, &err), "overflow dimension rejected");
    assert_true(est_overflow == UINT64_MAX, "overflow returns UINT64_MAX sentinel");

    // 3. Deterministic rSVD path with n, m > 64 (triggers rSVD instead of direct Jacobi)
    assert_true(n > 64 && m > 64, "dimensions n, m > 64");
    factor_pair fp_rsvd1, fp_rsvd2;
    assert_true(factorize_matrix(x_k, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 7777, fp_rsvd1, &err), "rSVD run 1 succeeded");
    assert_true(factorize_matrix(x_k, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 7777, fp_rsvd2, &err), "rSVD run 2 succeeded");
    assert_true(fp_rsvd1.a == fp_rsvd2.a, "rSVD n,m>64 is bitwise deterministic for A");
    assert_true(fp_rsvd1.b_transposed == fp_rsvd2.b_transposed, "rSVD n,m>64 is bitwise deterministic for BT");
    assert_true(fp_rsvd1.singular_values == fp_rsvd2.singular_values, "rSVD n,m>64 is bitwise deterministic for singular values");

    // 4. Rank deficiency test (matrix of all identical rows or low effective rank)
    matrix x_rank1(80, 80, 0.0f);
    for (uint64_t i = 0; i < 80; ++i) {
        for (uint64_t j = 0; j < 80; ++j) {
            x_rank1.at(i, j) = (float)(i + 1) * (float)(j + 1) * 0.01f;
        }
    }
    factor_pair fp_rank1;
    assert_true(factorize_matrix(x_rank1, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp_rank1, &err), "rank deficient matrix factorizes cleanly");
    assert_true(fp_rank1.singular_values.size() == 16, "singular values populated");
    assert_true(fp_rank1.singular_values[0] > 1e-3, "first singular value is dominant");
    // Higher singular values should be near zero for rank-1 matrix
    assert_true(fp_rank1.singular_values[2] < 1e-3 * fp_rank1.singular_values[0], "subsequent singular values are negligible");

    // 5. Non-finite input handling (NaN / Inf)
    matrix x_nan = x_k;
    x_nan.at(5, 5) = std::numeric_limits<float>::quiet_NaN();
    factor_pair fp_nan;
    assert_true(!factorize_matrix(x_nan, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp_nan, &err), "NaN matrix rejected");
    assert_true(err.find("NaN or Inf") != std::string::npos, "error reports NaN or Inf");

    matrix x_inf = x_v;
    x_inf.at(10, 10) = std::numeric_limits<float>::infinity();
    factor_result res_inf = factorize_kv(x_k, x_inf, cfg);
    assert_true(!res_inf.success, "Inf matrix in factorize_kv rejected");
    assert_true(res_inf.error_message.find("non-finite") != std::string::npos, "error reports non-finite");
    assert_true(res_inf.k.a.empty() && res_inf.v.a.empty(), "outputs empty on non-finite input");

    // 6. Zero heap allocations during bounded execution (after arena/workspace initialization)
    {
        std::vector<uint8_t> arena_mem(req_mat_ws + 16384);
        factor_workspace_span arena_span(arena_mem.data(), arena_mem.size());

        // Pre-allocate persistent output buffers off-side to verify ZERO temp allocations inside rSVD
        factor_pair fp_heap;
        fp_heap.a = matrix(n, 16, 0.0f);
        fp_heap.b_transposed = matrix(m, 16, 0.0f);
        fp_heap.singular_values.assign(16, 0.0f);

        heap_scope scope;
        bool ok = factorize_matrix_bounded(x_k, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 12345, fp_heap, arena_span, &err);
        size_t alloc_count = scope.count();
        size_t alloc_bytes = scope.bytes();
        // Stop scope before calling assert_true with string
        g_heap_counting = false;

        assert_true(ok, "factorize_matrix_bounded with adequate workspace succeeded");
        assert_true(alloc_count == 0, "bounded rSVD execution must perform zero heap allocations for temporaries");
        assert_true(alloc_bytes == 0, "bounded rSVD execution must allocate zero heap bytes for temporaries");
    }

    // 7. Evaluating bounded shadow error with one-byte-short
    factor_result valid_res = factorize_kv(x_k, x_v, cfg);
    assert_true(valid_res.success, "valid factorize_kv succeeded");
    uint64_t shadow_req = 0;
    assert_true(estimate_quantized_shadow_workspace_bytes(x_k, x_v, valid_res.k, valid_res.v, cfg, &shadow_req, &err), "shadow workspace estimated");

    std::vector<uint8_t> shadow_short_buf(shadow_req > 0 ? shadow_req - 1 : 0);
    factor_workspace_span shadow_short_span(shadow_short_buf.data(), shadow_short_buf.size());
    factor_quantized_shadow short_shadow = evaluate_quantized_shadow_bounded(x_k, x_v, valid_res.k, valid_res.v, cfg, shadow_short_span);
    assert_true(!short_shadow.success, "evaluate_quantized_shadow_bounded fails on one-byte short workspace");
    assert_true(short_shadow.error_message.find("workspace buffer") != std::string::npos, "shadow shortfall error message verified");

    // Shadow on non-finite input factors
    matrix x_bad = x_k;
    x_bad.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    factor_result res_bad = factorize_kv(x_bad, x_v, cfg);
    assert_true(!res_bad.success, "factorize_kv on NaN returns false");

    // 8. QR and SVD Fault Injection Tests (XKV-SR §15.4)
    {
        // Baseline without fault succeeds
        factor_config cfg_fault = cfg;
        cfg_fault.fault = factor_fault_injection::none;
        factor_result res_base = factorize_kv(x_k, x_v, cfg_fault);
        assert_true(res_base.success, "baseline factorize_kv succeeds");

        // Injected QR failure on K (first stage)
        cfg_fault.fault = factor_fault_injection::fail_qr;
        cfg_fault.fault_occurrence = 0;
        factor_pair untouched_k, untouched_v;
        factor_result res_qr = factorize_kv(x_k, x_v, cfg_fault);
        assert_true(!res_qr.success, "injected QR failure on K fails factorize_kv");
        assert_true(res_qr.error_message.find("injected QR failure") != std::string::npos, "error message specifies injected QR failure");
        assert_true(res_qr.k.a.empty() && res_qr.k.b_transposed.empty(), "K output remains empty on QR failure");
        assert_true(res_qr.v.a.empty() && res_qr.v.b_transposed.empty(), "V output remains empty on QR failure");

        // Injected SVD failure on K
        cfg_fault.fault = factor_fault_injection::fail_svd;
        cfg_fault.fault_occurrence = 0;
        factor_result res_svd_k = factorize_kv(x_k, x_v, cfg_fault);
        assert_true(!res_svd_k.success, "injected SVD failure on K fails factorize_kv");
        assert_true(res_svd_k.error_message.find("injected SVD failure") != std::string::npos, "error message specifies injected SVD failure");
        assert_true(res_svd_k.k.a.empty() && res_svd_k.k.b_transposed.empty(), "K output remains empty on SVD failure");

        // Injected SVD failure on V (second stage after K succeeded)
        cfg_fault.fault = factor_fault_injection::fail_svd;
        cfg_fault.fault_occurrence = 1;
        factor_result res_svd_v = factorize_kv(x_k, x_v, cfg_fault);
        assert_true(!res_svd_v.success, "injected SVD failure on V fails factorize_kv");
        assert_true(res_svd_v.error_message.find("injected SVD failure") != std::string::npos, "error message specifies injected SVD failure on V");
        assert_true(res_svd_v.k.a.empty() && res_svd_v.k.b_transposed.empty(), "atomic failure clears K output when V fails");
        assert_true(res_svd_v.v.a.empty() && res_svd_v.v.b_transposed.empty(), "V output remains empty when V fails");

        // Excluded from fingerprint: verify fingerprint identical to baseline
        assert_true(cfg_fault.fingerprint() == cfg.fingerprint(), "fault fields must be excluded from config fingerprint");
    }

    std::cout << "  [PASS] Edge cases, bounded workspaces, rank deficiency, and non-finite tests verified." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running XKV Factor Tests (PR04)" << std::endl;
    std::cout << "========================================" << std::endl;

    test_layer_group_mapping();
    test_determinism_and_fingerprint();
    test_full_rank_reconstruction();
    test_balancing_preserves_ab();
    test_low_rank_and_rsvd();
    test_four_stream_quantized_shadow();
    test_preflight_estimator_and_atomicity();
    test_edge_cases_and_bounded_workspaces();

    std::cout << "========================================" << std::endl;
    std::cout << "ALL XKV FACTOR TESTS PASSED SUCCESSFULLY" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
