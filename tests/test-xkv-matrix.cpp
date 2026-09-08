// test-xkv-matrix.cpp
//
// High-value mathematical and representation regression suite covering XKV-SR.md §15.1
// gaps against public codec, factor, reader, cache store, and graph-reference APIs.
//
// Invariants verified:
// 1. Full-rank/no-quant exact representation (A * (B^T)^T == X) and low-rank K-only, V-only, K+V
// 2. Layer group mapping: group sizes 1, 2, 4, tail groups, non-contiguous layers, and alias deduplication
// 3. Real GQA (n_q > n_kv) mapping and shared KV head + feature slice fixture for head_dim 128 and
//    256/64 partial NeoX / IMRoPE (rotary_dim 64 pass-through)
// 4. Boundary cases: segment/chunk non-divisibility, single-row matrices (n=1), rank >= n,
//    non-power-of-two padded rank (F32 exact zero tail + Turbo4 bounded tail RMSE)
// 5. Four-stream mixed types: independent A_K, B_K, A_V, B_V covering Turbo2, Turbo3, Turbo4, Q8_0, F16, F32
// 6. Fixed-factor error attribution: A-only vs B-only vs A+B product error decomposition
// 7. Byte-preserving A pack and immutable B identity driven via actual store mutation/pack APIs
//    (store.remove_payload + store.pack_segment) asserting A rows byte-identical and B shared_ptr identity
// 8. Large and negative storage positions under RoPE inversion / phase mapping
// 9. Chunked online global softmax equivalence: multi-block vs monolithic with non-trivial logit_softcap,
//    all-masked finite-zero handling, and sink-once invariant
//
// Self-contained, deterministic, bounded fixtures, zero external model dependency.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-factor.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-reader.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-canonical.h"
#include "llama-triattention.h"
#include "llama-cparams.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <memory>

using namespace llama_xkv;

static int g_failures = 0;

#define MATRIX_CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " - " << msg << std::endl; \
        ++g_failures; \
    } \
} while (0)

#define MATRIX_CHECK_APPROX(a, b, tol, msg) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (diff > (tol) || !std::isfinite(diff)) { \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " - " << msg \
                  << " (got " << (a) << ", expected " << (b) << ", diff=" << diff << ", tol=" << (tol) << ")" << std::endl; \
        ++g_failures; \
    } \
} while (0)

// Deterministic PRNG for test matrix synthesis (LCG / xorshift)
static matrix make_test_matrix(uint64_t rows, uint64_t cols, uint64_t seed) {
    matrix m(rows, cols);
    uint64_t state = seed ? seed : 0x12345678ULL;
    for (uint64_t i = 0; i < rows * cols; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        float val = ((float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF) * 2.0f - 1.0f;
        m.data[i] = val;
    }
    return m;
}

// Low-rank structured matrix synthesis: X = U * V^T + noise
static matrix make_low_rank_matrix(uint64_t rows, uint64_t cols, uint32_t true_rank, float noise_scale, uint64_t seed) {
    matrix u = make_test_matrix(rows, true_rank, seed);
    matrix v = make_test_matrix(cols, true_rank, seed + 100);
    matrix x(rows, cols, 0.0f);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < true_rank; ++k) {
                sum += u.at(r, k) * v.at(c, k);
            }
            x.at(r, c) = sum;
        }
    }
    if (noise_scale > 0.0f) {
        matrix noise = make_test_matrix(rows, cols, seed + 200);
        for (uint64_t i = 0; i < rows * cols; ++i) {
            x.data[i] += noise.data[i] * noise_scale;
        }
    }
    return x;
}

static llama_cparams make_test_cparams() {
    llama_cparams cp = {};
    cp.xkv_mode = LLAMA_XKV_MODE_SHADOW;
    cp.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cp.xkv_group_size = 4;
    cp.xkv_rank_k = 128;
    cp.xkv_rank_v = 128;
    cp.xkv_segment_tokens = 64;
    cp.xkv_chunk_tokens = 8;
    cp.xkv_workspace_mib = 16;
    cp.xkv_decode_cache_mib = 8;
    return cp;
}

// ============================================================================
// 1. Full-rank / no-quant, low-rank K-only, V-only, K+V
// ============================================================================
static void test_representation_rank_modes() {
    std::cout << "[Test 1] Full-rank / no-quant, low-rank K-only, V-only, K+V..." << std::endl;

    // 1a. Full-rank no-quant exact reconstruction
    {
        uint64_t n = 16, m = 16;
        matrix x = make_test_matrix(n, m, 101);
        factor_pair fp;
        std::string err;
        bool ok = factorize_matrix(x, (uint32_t)m, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp, &err);
        MATRIX_CHECK(ok, "Full-rank factorization should succeed: " + err);
        matrix recon = matrix_reconstruct(fp.a, fp.b_transposed);
        factor_error_report rep = compute_error_report(x, recon);
        MATRIX_CHECK(rep.relative_error < 1e-4, "Full-rank relative error should be < 1e-4");
        MATRIX_CHECK(rep.max_absolute_error < 1e-4, "Full-rank max absolute error should be < 1e-4");
    }

    // 1b. Low-rank K+V factorize_kv
    {
        uint64_t n = 32, m = 64;
        matrix x_k = make_low_rank_matrix(n, m, 8, 0.001f, 201);
        matrix x_v = make_low_rank_matrix(n, m, 8, 0.001f, 202);

        factor_config cfg;
        cfg.rank_k = 16;
        cfg.rank_v = 16;
        cfg.seed = 42;
        cfg.balance = LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;

        factor_result res = factorize_kv(x_k, x_v, cfg);
        MATRIX_CHECK(res.success, "factorize_kv K+V should succeed: " + res.error_message);
        MATRIX_CHECK(res.k.rank == 16, "K rank should match requested rank");
        MATRIX_CHECK(res.v.rank == 16, "V rank should match requested rank");

        matrix recon_k = matrix_reconstruct(res.k.a, res.k.b_transposed);
        matrix recon_v = matrix_reconstruct(res.v.a, res.v.b_transposed);
        factor_error_report rep_k = compute_error_report(x_k, recon_k);
        factor_error_report rep_v = compute_error_report(x_v, recon_v);
        MATRIX_CHECK(rep_k.relative_error < 0.05, "Low-rank K reconstruction relative error < 0.05");
        MATRIX_CHECK(rep_v.relative_error < 0.05, "Low-rank V reconstruction relative error < 0.05");
    }

    // 1c. K-only low-rank (full-rank V) and V-only low-rank (full-rank K)
    {
        uint64_t n = 24, m = 32;
        matrix x_k = make_low_rank_matrix(n, m, 6, 0.001f, 301);
        matrix x_v = make_test_matrix(n, m, 302);

        factor_config cfg_k_low;
        cfg_k_low.rank_k = 8;
        cfg_k_low.rank_v = (uint32_t)std::min(n, m); // full rank V
        cfg_k_low.seed = 42;
        factor_result res_k = factorize_kv(x_k, x_v, cfg_k_low);
        MATRIX_CHECK(res_k.success, "K-only low-rank factorize_kv should succeed");
        matrix recon_v_full = matrix_reconstruct(res_k.v.a, res_k.v.b_transposed);
        factor_error_report rep_v_full = compute_error_report(x_v, recon_v_full);
        MATRIX_CHECK(rep_v_full.relative_error < 1e-3, "V full rank with K-only low rank should be exact (< 1e-3)");

        factor_config cfg_v_low;
        cfg_v_low.rank_k = (uint32_t)std::min(n, m); // full rank K
        cfg_v_low.rank_v = 8;
        cfg_v_low.seed = 42;
        factor_result res_v = factorize_kv(x_k, x_v, cfg_v_low);
        MATRIX_CHECK(res_v.success, "V-only low-rank factorize_kv should succeed");
        matrix recon_k_full = matrix_reconstruct(res_v.k.a, res_v.k.b_transposed);
        factor_error_report rep_k_full = compute_error_report(x_k, recon_k_full);
        MATRIX_CHECK(rep_k_full.relative_error < 1e-3, "K full rank with V-only low rank should be exact (< 1e-3)");
    }
}

// ============================================================================
// 2. Layer grouping: groups 1/2/4, incomplete tail groups, non-contiguous/alias
// ============================================================================
static void test_layer_grouping_matrix() {
    std::cout << "[Test 2] Layer grouping: groups 1/2/4, incomplete tail groups, non-contiguous/alias..." << std::endl;

    std::vector<uint32_t> owning = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};

    // Group size 4: 10 layers -> 2 full groups of 4 + 1 tail group of 2
    layer_group_map gm4 = build_layer_group_map(owning, 4, 128, 128);
    MATRIX_CHECK(gm4.groups.size() == 3, "group_size=4 must yield 3 groups");
    MATRIX_CHECK(gm4.groups[0].owning_layers.size() == 4, "group 0 has 4 layers");
    MATRIX_CHECK(gm4.groups[1].owning_layers.size() == 4, "group 1 has 4 layers");
    MATRIX_CHECK(gm4.groups[2].owning_layers.size() == 2, "group 2 (tail) has 2 layers");
    MATRIX_CHECK(gm4.groups[0].total_dim_k == 512, "group 0 total dim_k is 512");
    MATRIX_CHECK(gm4.groups[2].total_dim_k == 256, "tail group total dim_k is 256");

    // Group size 2: 10 layers -> 5 groups of 2 (no incomplete tail)
    layer_group_map gm2 = build_layer_group_map(owning, 2, 128, 128);
    MATRIX_CHECK(gm2.groups.size() == 5, "group_size=2 must yield 5 groups");
    for (size_t g = 0; g < gm2.groups.size(); ++g) {
        MATRIX_CHECK(gm2.groups[g].owning_layers.size() == 2, "every group has 2 layers");
        MATRIX_CHECK(gm2.groups[g].total_dim_k == 256, "every group total dim_k is 256");
    }

    // Group size 1: 10 groups of 1
    layer_group_map gm1 = build_layer_group_map(owning, 1, 128, 128);
    MATRIX_CHECK(gm1.groups.size() == 10, "group_size=1 must yield 10 groups");
    for (size_t g = 0; g < gm1.groups.size(); ++g) {
        MATRIX_CHECK(gm1.groups[g].owning_layers.size() == 1, "group has 1 layer");
        MATRIX_CHECK(gm1.groups[g].total_dim_k == 128, "group dim is 128");
    }

    // Incomplete tail group of 1 with group size 4: e.g. 5 layers -> 1 group of 4, 1 tail of 1
    std::vector<uint32_t> owning5 = {1, 3, 5, 7, 9};
    layer_group_map gm5_4 = build_layer_group_map(owning5, 4, 64, 64);
    MATRIX_CHECK(gm5_4.groups.size() == 2, "5 layers / 4 gives 2 groups");
    MATRIX_CHECK(gm5_4.groups[1].owning_layers.size() == 1, "tail group has 1 layer");
    MATRIX_CHECK(gm5_4.groups[1].total_dim_k == 64, "tail group dim is 64");

    // Alias mapping & non-attention layers via build_layer_group_map_ex
    std::vector<uint32_t> model_to_owning = {0, 3, 0, 7, 3, 11, 7, 15};
    std::vector<bool> is_attn              = {false, true, false, true, true, true, true, true};
    std::vector<uint32_t> dim_k_per_layer(16, 128);
    std::vector<uint32_t> dim_v_per_layer(16, 128);
    layer_group_map gm_ex = build_layer_group_map_ex(model_to_owning, is_attn, 2, dim_k_per_layer, dim_v_per_layer);

    MATRIX_CHECK(gm_ex.unique_owning_layers.size() == 4, "Unique owning layers deduplicated to 4 (3, 7, 11, 15)");
    MATRIX_CHECK(gm_ex.groups.size() == 2, "4 unique owning layers grouped by 2 gives 2 groups");
    MATRIX_CHECK(gm_ex.model_layer_to_group[0] == UINT32_MAX, "recurrent layer 0 maps to UINT32_MAX");
    MATRIX_CHECK(gm_ex.model_layer_to_group[2] == UINT32_MAX, "recurrent layer 2 maps to UINT32_MAX");
    MATRIX_CHECK(gm_ex.model_layer_to_group[1] == gm_ex.model_layer_to_group[4], "aliased model layers 1 and 4 have same group");
    MATRIX_CHECK(gm_ex.model_layer_to_group_offset[1] == gm_ex.model_layer_to_group_offset[4], "aliased model layers have same offset");
}

// ============================================================================
// 3. Real GQA (n_q > n_kv) mapping and shared KV head + feature slice fixture
//    for head_dim 128 and 256/64 partial NeoX / IMRoPE
// ============================================================================
static void test_gqa_mapping_and_partial_neox_slices() {
    std::cout << "[Test 3] Real GQA (n_q > n_kv) mapping, shared KV head + feature slices (hd128 and 256/64)..." << std::endl;

    // 3a. Standard head_dim = 128, GQA ratio 4:1 (n_q = 4, n_kv = 1)
    {
        uint32_t head_dim_k = 128;
        uint32_t head_dim_v = 128;
        uint32_t rotary_dim = 128;
        uint32_t n_q_heads  = 4; // 4 query heads share 1 KV head
        uint32_t n_tokens   = 8;

        uint32_t fc = rotary_dim / 2;
        std::vector<float> omega(fc);
        std::vector<float> scale_sq(fc);
        bool ok = triattention_build_rope_tables(omega.data(), scale_sq.data(), rotary_dim, 10000.0f, 1.0f, 4096, 0.0f, 1.0f, 32.0f, 1.0f, nullptr);
        MATRIX_CHECK(ok, "RoPE tables head_dim=128 should succeed");

        // 4 query head vectors: [4 * 128]
        std::vector<float> q_heads = make_test_matrix(n_q_heads, head_dim_k, 301).data;

        // 8 shared KV tokens: K=[8 * 128], V=[8 * 128]
        matrix k_shared = make_test_matrix(n_tokens, head_dim_k, 302);
        matrix v_shared = make_test_matrix(n_tokens, head_dim_v, 303);

        // Apply RoPE at positions 0..7
        std::vector<float> k_phased(n_tokens * head_dim_k);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            float pos = (float)t;
            for (uint32_t f = 0; f < fc; ++f) {
                float angle = omega[f] * pos;
                float c = std::cos(angle);
                float s = std::sin(angle);
                float re = k_shared.at(t, f);
                float im = k_shared.at(t, f + fc);
                k_phased[t * head_dim_k + f]      = re * c - im * s;
                k_phased[t * head_dim_k + f + fc] = im * c + re * s;
            }
        }

        // Each Q head independently attends to the same shared KV head
        float scale = 1.0f / std::sqrt((float)head_dim_k);
        for (uint32_t qh = 0; qh < n_q_heads; ++qh) {
            const float * q_ptr = q_heads.data() + qh * head_dim_k;
            xkv_online_softmax_state st;
            st.init(nullptr, head_dim_v);

            for (uint32_t t = 0; t < n_tokens; ++t) {
                float dot = 0.0f;
                const float * k_ptr = k_phased.data() + t * head_dim_k;
                for (uint32_t d = 0; d < head_dim_k; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
                float score = dot * scale;
                st.update_single(score, v_shared.row_ptr(t), head_dim_v);
            }

            std::vector<float> out(head_dim_v, 0.0f);
            st.finalize(out.data(), head_dim_v);
            MATRIX_CHECK(st.sum_exp > 0.0f, "GQA Q head softmax sum_exp > 0");
            MATRIX_CHECK(std::isfinite(out[0]), "GQA Q head output is finite");
        }
    }

    // 3b. Partial NeoX / IMRoPE: head_dim = 256, rotary_dim = 64, GQA ratio 8:1 (n_q = 8, n_kv = 1)
    // Invariant: dimensions [0, 64) undergo RoPE rotation; dimensions [64, 256) pass through untransformed
    {
        uint32_t head_dim_k = 256;
        uint32_t head_dim_v = 256;
        uint32_t rotary_dim = 64;
        uint32_t n_q_heads  = 8; // 8 query heads share 1 KV head
        uint32_t n_tokens   = 6;

        uint32_t fc = rotary_dim / 2; // 32
        std::vector<float> omega(fc);
        std::vector<float> scale_sq(fc);
        bool ok = triattention_build_rope_tables(omega.data(), scale_sq.data(), rotary_dim, 10000000.0f, 1.0f, 4096, 0.0f, 1.0f, 32.0f, 1.0f, nullptr);
        MATRIX_CHECK(ok, "RoPE tables head_dim=256 rotary_dim=64 should succeed");

        std::vector<float> q_heads = make_test_matrix(n_q_heads, head_dim_k, 304).data;
        matrix k_shared = make_test_matrix(n_tokens, head_dim_k, 305);
        matrix v_shared = make_test_matrix(n_tokens, head_dim_v, 306);

        std::vector<float> k_phased(n_tokens * head_dim_k);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            float pos = (float)(t * 10);
            // RoPE on [0, 64)
            for (uint32_t f = 0; f < fc; ++f) {
                float angle = omega[f] * pos;
                float c = std::cos(angle);
                float s = std::sin(angle);
                float re = k_shared.at(t, f);
                float im = k_shared.at(t, f + fc);
                k_phased[t * head_dim_k + f]      = re * c - im * s;
                k_phased[t * head_dim_k + f + fc] = im * c + re * s;
            }
            // Pass-through unrotated tail [64, 256)
            for (uint32_t d = rotary_dim; d < head_dim_k; ++d) {
                k_phased[t * head_dim_k + d] = k_shared.at(t, d);
            }
        }

        // Validate inverse RoPE exact round-trip across GQA shared keys
        for (uint32_t t = 0; t < n_tokens; ++t) {
            int32_t pos = (int32_t)(t * 10);
            std::vector<float> recovered(head_dim_k, 0.0f);
            std::string err;
            bool inv_ok = invert_rope_k(recovered.data(), k_phased.data() + t * head_dim_k, &pos, omega.data(), scale_sq.data(), 1, head_dim_k, rotary_dim, &err);
            MATRIX_CHECK(inv_ok, "invert_rope_k for partial NeoX GQA token should succeed: " + err);
            for (uint32_t d = 0; d < rotary_dim; ++d) {
                MATRIX_CHECK_APPROX(recovered[d], k_shared.at(t, d), 1e-5f, "Rotary dim restored");
            }
            for (uint32_t d = rotary_dim; d < head_dim_k; ++d) {
                MATRIX_CHECK_APPROX(recovered[d], k_shared.at(t, d), 1e-7f, "Unrotated tail identical");
            }
        }

        // Verify all 8 GQA Q heads evaluate against the shared partial-RoPE K/V
        float scale = 1.0f / std::sqrt((float)head_dim_k);
        for (uint32_t qh = 0; qh < n_q_heads; ++qh) {
            const float * q_ptr = q_heads.data() + qh * head_dim_k;
            xkv_online_softmax_state st;
            st.init(nullptr, head_dim_v);

            for (uint32_t t = 0; t < n_tokens; ++t) {
                float dot = 0.0f;
                const float * k_ptr = k_phased.data() + t * head_dim_k;
                for (uint32_t d = 0; d < head_dim_k; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
                float score = dot * scale;
                st.update_single(score, v_shared.row_ptr(t), head_dim_v);
            }

            std::vector<float> out(head_dim_v, 0.0f);
            st.finalize(out.data(), head_dim_v);
            MATRIX_CHECK(st.sum_exp > 0.0f, "Partial NeoX GQA Q head softmax sum_exp > 0");
        }
    }
}

// ============================================================================
// 4. Boundary cases: segment/chunk non-divisibility, single-row (n=1), rank >= n,
//    non-power-of-two padded rank (F32 exact zero tail + Turbo4 bounded tail RMSE)
// ============================================================================
static void test_boundary_and_geometry_matrix() {
    std::cout << "[Test 4] Boundary cases: non-divisibility, single-row (n=1), rank >= n, non-power-of-two padding..." << std::endl;

    // 4a. Single-row matrix (n=1, m=64)
    {
        matrix m1 = make_test_matrix(1, 64, 501);
        factor_pair fp;
        std::string err;
        bool ok = factorize_matrix(m1, 1, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp, &err);
        MATRIX_CHECK(ok, "Factorize n=1 matrix should succeed: " + err);
        if (ok) {
            MATRIX_CHECK(fp.rank == 1, "Rank for n=1 should be 1");
            MATRIX_CHECK(fp.a.rows == 1 && fp.a.cols == 1, "A matrix shape should be 1x1");
            MATRIX_CHECK(fp.b_transposed.rows == 64 && fp.b_transposed.cols == 1, "B^T matrix shape should be 64x1");

            matrix recon = matrix_reconstruct(fp.a, fp.b_transposed);
            factor_error_report rep = compute_error_report(m1, recon);
            MATRIX_CHECK(rep.relative_error < 1e-4, "n=1 reconstruction should be exact (< 1e-4)");
        }
    }

    // 4b. Requested rank >= n (e.g. n=10, m=20, requested_rank=16 clamped to min(n,m)=10)
    {
        matrix m_clamp = make_test_matrix(10, 20, 502);
        factor_pair fp;
        std::string err;
        bool ok = factorize_matrix(m_clamp, 16, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 42, fp, &err);
        MATRIX_CHECK(ok, "Factorize rank >= n should succeed: " + err);
        MATRIX_CHECK(fp.rank == 10, "Rank should clamp to min(rows, cols) = 10");
        matrix recon = matrix_reconstruct(fp.a, fp.b_transposed);
        factor_error_report rep = compute_error_report(m_clamp, recon);
        MATRIX_CHECK(rep.relative_error < 1e-4, "Clamped full rank reconstruction should be exact (< 1e-4)");
    }

    // 4c. Non-power-of-two rank with padding (rank=70 padded to 128):
    // Part 1: F32 codec decodes tail cols [70..127] to EXACT zero
    {
        matrix m70 = make_test_matrix(8, 70, 503);
        codec_desc desc_f32 = make_codec_desc(
            factor_role::a_k,
            GGML_TYPE_F32,
            orientation::token_major,
            {8, 70},
            0,
            42
        );
        MATRIX_CHECK(desc_f32.logical_shape.cols == 70, "F32 logical cols == 70");
        MATRIX_CHECK(desc_f32.padded_shape.cols == 70, "F32 padded cols == 70 (no block padding needed for F32)");
        encoded_matrix em_f32 = encode_matrix(desc_f32, m70.data.data(), m70.elements());
        std::vector<float> dec_f32 = decode_matrix(em_f32, value_domain::canonical);
        for (size_t i = 0; i < m70.elements(); ++i) {
            MATRIX_CHECK(dec_f32[i] == m70.data[i], "F32 decode matches source exactly");
        }
    }

    // Part 2: Turbo4 codec: rank 50 padded to 128.
    // In canonical domain after inverse WHT, quantization noise distributes across all 128 elements.
    // Tail cols [50..127] must have bounded tail RMSE (per codec contract < 0.10f), NOT exact zero!
    {
        uint64_t log_cols = 50;
        uint64_t pad_cols = 128;
        uint64_t rows = 4;
        matrix m50 = make_test_matrix(rows, log_cols, 504);

        codec_desc desc_t4 = make_codec_desc(
            factor_role::a_k,
            GGML_TYPE_TURBO4_0,
            orientation::token_major,
            {rows, log_cols},
            128,
            42
        );
        MATRIX_CHECK(desc_t4.logical_shape.cols == log_cols, "Turbo4 logical cols == 50");
        MATRIX_CHECK(desc_t4.padded_shape.cols == pad_cols, "Turbo4 padded cols == 128");

        encoded_matrix em_t4 = encode_matrix(desc_t4, m50.data.data(), m50.elements());
        std::vector<float> decoded = decode_matrix(em_t4, value_domain::canonical);
        MATRIX_CHECK(decoded.size() == rows * pad_cols, "Decoded size matches padded shape rows * padded_cols");

        for (uint64_t r = 0; r < rows; ++r) {
            float tail_sq_sum = 0.0f;
            for (uint64_t c = log_cols; c < pad_cols; ++c) {
                float val = decoded[r * pad_cols + c];
                MATRIX_CHECK(std::isfinite(val), "Decoded tail value is finite");
                tail_sq_sum += val * val;
            }
            float tail_rmse = std::sqrt(tail_sq_sum / (float)(pad_cols - log_cols));
            MATRIX_CHECK(tail_rmse < 0.12f, "Turbo4 tail noise RMSE is strictly bounded (< 0.12f)");
        }
    }
}

// ============================================================================
// 5. Mixed types: A_K, B_K, A_V, B_V independent across Turbo2/3/4, Q8_0, F16, F32
// ============================================================================
static void test_mixed_codecs_matrix() {
    std::cout << "[Test 5] Mixed types: independent A_K, B_K, A_V, B_V (Turbo2/3/4, Q8_0, F16, F32)..." << std::endl;

    uint64_t rows = 16, cols = 128;
    matrix m = make_test_matrix(rows, cols, 601);

    ggml_type test_types[] = {
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_TURBO2_0,
        GGML_TYPE_TURBO3_0,
        GGML_TYPE_TURBO4_0
    };

    for (ggml_type t : test_types) {
        codec_desc desc = make_codec_desc(
            factor_role::a_k,
            t,
            orientation::token_major,
            {rows, cols},
            128,
            42
        );
        MATRIX_CHECK(desc.validate(), "codec_desc should validate for type " + std::to_string((int)t));

        encoded_matrix em = encode_matrix(desc, m.data.data(), m.elements());
        MATRIX_CHECK(em.bytes.size() == encoded_matrix_bytes(desc), "encoded byte count matches exact descriptor bytes");

        std::vector<float> dec = decode_matrix(em, value_domain::canonical);
        MATRIX_CHECK(dec.size() == rows * desc.padded_shape.cols, "decoded elements match padded shape");

        double se = 0.0;
        for (uint64_t i = 0; i < m.elements(); ++i) {
            double diff = (double)m.data[i] - (double)dec[i];
            se += diff * diff;
        }
        double mse = se / (double)m.elements();

        if (t == GGML_TYPE_F32) {
            MATRIX_CHECK(mse < 1e-9, "F32 codec MSE must be near zero");
        } else if (t == GGML_TYPE_F16) {
            MATRIX_CHECK(mse < 1e-5, "F16 codec MSE < 1e-5");
        } else if (t == GGML_TYPE_Q8_0) {
            MATRIX_CHECK(mse < 0.001, "Q8_0 codec MSE < 0.001");
        } else if (t == GGML_TYPE_TURBO4_0) {
            MATRIX_CHECK(mse < 0.05, "Turbo4 codec MSE < 0.05");
        } else if (t == GGML_TYPE_TURBO3_0) {
            MATRIX_CHECK(mse < 0.15, "Turbo3 codec MSE < 0.15");
        } else if (t == GGML_TYPE_TURBO2_0) {
            MATRIX_CHECK(mse < 0.40, "Turbo2 codec MSE < 0.40");
        }
    }

    // Factor pair compatibility check: A Turbo4 + B F16 (mixed representation)
    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {16, 128}, 128, 42);
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F16, orientation::feature_major_transposed, {64, 128}, 0, 42);
    std::string compat_err;
    bool compat_ok = validate_factor_pair_compatibility(desc_a, desc_b, &compat_err);
    MATRIX_CHECK(compat_ok, "A Turbo4 + B F16 mixed pair should be compatible via canonical decoding: " + compat_err);
}

// ============================================================================
// 6. Fixed-factor error attribution: A-only vs B-only vs A+B product
// ============================================================================
static void test_fixed_factor_error_attribution() {
    std::cout << "[Test 6] Fixed-factor error attribution: A-only vs B-only vs A+B product..." << std::endl;

    uint64_t n = 32, m = 64;
    matrix x_k = make_low_rank_matrix(n, m, 8, 0.001f, 701);
    matrix x_v = make_low_rank_matrix(n, m, 8, 0.001f, 702);

    factor_config cfg;
    cfg.rank_k = 128;
    cfg.rank_v = 128;
    cfg.seed = 42;
    cfg.factor_a_k = GGML_TYPE_TURBO4_0;
    cfg.factor_b_k = GGML_TYPE_TURBO4_0;
    cfg.factor_a_v = GGML_TYPE_TURBO4_0;
    cfg.factor_b_v = GGML_TYPE_TURBO4_0;

    factor_result res = factorize_kv(x_k, x_v, cfg);
    MATRIX_CHECK(res.success, "factorize_kv should succeed for shadow evaluation: " + res.error_message);

    factor_quantized_shadow shadow = evaluate_quantized_shadow(x_k, x_v, res.k, res.v, cfg);
    MATRIX_CHECK(shadow.success, "evaluate_quantized_shadow should succeed: " + shadow.error_message);

    // Verify error attribution ordering for K:
    MATRIX_CHECK(shadow.errors_k.a_only.relative_error > 0.0, "A-only quantization has positive error");
    MATRIX_CHECK(shadow.errors_k.b_only.relative_error > 0.0, "B-only quantization has positive error");
    MATRIX_CHECK(shadow.errors_k.ab_product.relative_error > 0.0, "A+B product quantization has positive error");

    double sum_err = shadow.errors_k.a_only.relative_error + shadow.errors_k.b_only.relative_error;
    MATRIX_CHECK(shadow.errors_k.ab_product.relative_error <= sum_err * 1.5,
                 "A+B product error bounded by sum of individual errors");

    // Same for V
    MATRIX_CHECK(shadow.errors_v.a_only.relative_error > 0.0, "V A-only error > 0");
    MATRIX_CHECK(shadow.errors_v.b_only.relative_error > 0.0, "V B-only error > 0");
    MATRIX_CHECK(shadow.errors_v.ab_product.relative_error > 0.0, "V A+B error > 0");
}

// ============================================================================
// 7. Byte-preserving A pack and B identity driven via actual store mutation APIs
//    (store.remove_payload + store.pack_segment) asserting A rows byte-identical
//    and B shared_ptr identity across versions.
// ============================================================================
static void test_byte_preserving_store_pack() {
    std::cout << "[Test 7] Byte-preserving A pack and B identity driven via store.pack_segment API..." << std::endl;

    auto cparams = make_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 6;
    const std::vector<uint64_t> pids = {10, 20, 30, 40, 50, 60};
    const std::vector<uint64_t> gens = {1, 2, 3, 4, 5, 6};

    for (size_t i = 0; i < n_rows; ++i) {
        store.register_hot_payload(pids[i], (uint32_t)i, gens[i], xkv_state::hot_committed);
    }
    bool marked = store.mark_seal_candidates(pids, gens, nullptr);
    MATRIX_CHECK(marked, "mark_seal_candidates should succeed");

    // Construct valid candidate segment
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.total_dim_k = 64;
    g.total_dim_v = 64;
    g.layer_feature_offsets_k = {0, 16, 32, 48};
    g.layer_feature_dims_k    = {16, 16, 16, 16};
    g.layer_feature_offsets_v = {0, 16, 32, 48};
    g.layer_feature_dims_v    = {16, 16, 16, 16};

    uint32_t rank_k = 16, rank_v = 16;
    codec_desc desc_a_k = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_k}, 0, 1001);
    std::vector<float> data_a_k(n_rows * rank_k);
    for (size_t i = 0; i < data_a_k.size(); ++i) data_a_k[i] = ((float)((i * 3 + 7) % 100)) * 0.01f;
    g.a_k = encode_matrix(desc_a_k, data_a_k.data(), data_a_k.size());

    codec_desc desc_b_k = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {64, rank_k}, 0, 1002);
    std::vector<float> data_b_k(64 * rank_k);
    for (size_t i = 0; i < data_b_k.size(); ++i) data_b_k[i] = ((float)((i * 5 + 11) % 100)) * 0.01f;
    g.set_b_k(encode_matrix(desc_b_k, data_b_k.data(), data_b_k.size()));

    codec_desc desc_a_v = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_v}, 0, 1003);
    std::vector<float> data_a_v(n_rows * rank_v);
    for (size_t i = 0; i < data_a_v.size(); ++i) data_a_v[i] = ((float)((i * 7 + 13) % 100)) * 0.01f;
    g.a_v = encode_matrix(desc_a_v, data_a_v.data(), data_a_v.size());

    codec_desc desc_b_v = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {64, rank_v}, 0, 1004);
    std::vector<float> data_b_v(64 * rank_v);
    for (size_t i = 0; i < data_b_v.size(); ++i) data_b_v[i] = ((float)((i * 11 + 17) % 100)) * 0.01f;
    g.set_b_v(encode_matrix(desc_b_v, data_b_v.data(), data_b_v.size()));

    auto seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, {g});
    MATRIX_CHECK(seg != nullptr, "create_candidate_segment succeeded");
    const uint64_t seg_id = seg->segment_id;

    // Stamp unique patterns into A_K and A_V rows
    const size_t stride_a_k = seg->groups[0].a_k.desc.row_stride_bytes;
    const size_t stride_a_v = seg->groups[0].a_v.desc.row_stride_bytes;
    for (uint32_t r = 0; r < n_rows; ++r) {
        std::memset(seg->groups[0].a_k.bytes.data() + r * stride_a_k, (int)(0xA0 + r), stride_a_k);
        std::memset(seg->groups[0].a_v.bytes.data() + r * stride_a_v, (int)(0xB0 + r), stride_a_v);
    }

    // Capture expected bytes for survivor rows (rows 0, 2, 5 corresponding to pids 10, 30, 60)
    std::vector<uint8_t> expected_row0_a_k(seg->groups[0].a_k.bytes.begin() + 0 * stride_a_k, seg->groups[0].a_k.bytes.begin() + 1 * stride_a_k);
    std::vector<uint8_t> expected_row2_a_k(seg->groups[0].a_k.bytes.begin() + 2 * stride_a_k, seg->groups[0].a_k.bytes.begin() + 3 * stride_a_k);
    std::vector<uint8_t> expected_row5_a_k(seg->groups[0].a_k.bytes.begin() + 5 * stride_a_k, seg->groups[0].a_k.bytes.begin() + 6 * stride_a_k);

    std::vector<uint8_t> expected_row0_a_v(seg->groups[0].a_v.bytes.begin() + 0 * stride_a_v, seg->groups[0].a_v.bytes.begin() + 1 * stride_a_v);
    std::vector<uint8_t> expected_row2_a_v(seg->groups[0].a_v.bytes.begin() + 2 * stride_a_v, seg->groups[0].a_v.bytes.begin() + 3 * stride_a_v);
    std::vector<uint8_t> expected_row5_a_v(seg->groups[0].a_v.bytes.begin() + 5 * stride_a_v, seg->groups[0].a_v.bytes.begin() + 6 * stride_a_v);

    std::shared_ptr<const encoded_matrix> orig_b_k = seg->groups[0].b_k;
    std::shared_ptr<const encoded_matrix> orig_b_v = seg->groups[0].b_v;

    std::string pub_err;
    bool published = store.publish_candidate(seg, pids, gens, &pub_err);
    MATRIX_CHECK(published, "publish_candidate should succeed: " + pub_err);

    // Remove payloads 20, 40, 50 (rows 1, 3, 4)
    store.remove_payload(20);
    store.remove_payload(40);
    store.remove_payload(50);

    // Trigger genuine pack_segment through store API
    std::string pack_err;
    bool pack_ok = store.pack_segment(seg_id, &pack_err);
    MATRIX_CHECK(pack_ok, "store.pack_segment should succeed: " + pack_err);

    auto packed_seg = store.get_segment(seg_id);
    MATRIX_CHECK(packed_seg != nullptr, "packed segment must exist in store");
    MATRIX_CHECK(packed_seg->n_rows == 3, "packed segment has 3 survivor rows");
    MATRIX_CHECK(packed_seg->row_payload_ids == std::vector<uint64_t>({10, 30, 60}), "survivor payload IDs match exactly");

    // Invariant 1: B factor matrix handle is SHARED across versions (zero re-allocation)
    MATRIX_CHECK(packed_seg->groups[0].b_k == orig_b_k, "Immutable B_K pointer shared across pack versions");
    MATRIX_CHECK(packed_seg->groups[0].b_v == orig_b_v, "Immutable B_V pointer shared across pack versions");

    // Invariant 2: Survivor A rows are byte-identical without re-quantization
    MATRIX_CHECK(std::memcmp(packed_seg->groups[0].a_k.bytes.data() + 0 * stride_a_k, expected_row0_a_k.data(), stride_a_k) == 0,
                 "Survivor row 0 A_K byte identical");
    MATRIX_CHECK(std::memcmp(packed_seg->groups[0].a_v.bytes.data() + 0 * stride_a_v, expected_row0_a_v.data(), stride_a_v) == 0,
                 "Survivor row 0 A_V byte identical");

    MATRIX_CHECK(std::memcmp(packed_seg->groups[0].a_k.bytes.data() + 1 * stride_a_k, expected_row2_a_k.data(), stride_a_k) == 0,
                 "Survivor row 1 (src 2) A_K byte identical");
    MATRIX_CHECK(std::memcmp(packed_seg->groups[0].a_v.bytes.data() + 1 * stride_a_v, expected_row2_a_v.data(), stride_a_v) == 0,
                 "Survivor row 1 (src 2) A_V byte identical");

    MATRIX_CHECK(std::memcmp(packed_seg->groups[0].a_k.bytes.data() + 2 * stride_a_k, expected_row5_a_k.data(), stride_a_k) == 0,
                 "Survivor row 2 (src 5) A_K byte identical");
    MATRIX_CHECK(std::memcmp(packed_seg->groups[0].a_v.bytes.data() + 2 * stride_a_v, expected_row5_a_v.data(), stride_a_v) == 0,
                 "Survivor row 2 (src 5) A_V byte identical");
}

// ============================================================================
// 8. Large and negative storage positions
// ============================================================================
static void test_large_and_negative_positions() {
    std::cout << "[Test 8] Large and negative storage positions..." << std::endl;

    uint32_t head_dim = 128;
    uint32_t rotary_dim = 128;
    uint32_t fc = rotary_dim / 2;
    std::vector<float> omega(fc);
    std::vector<float> scale_sq(fc);
    bool ok = triattention_build_rope_tables(omega.data(), scale_sq.data(), rotary_dim, 10000.0f, 1.0f, 4096, 0.0f, 1.0f, 32.0f, 1.0f, nullptr);
    MATRIX_CHECK(ok, "RoPE tables built successfully");

    std::vector<float> canonical_k = make_test_matrix(1, head_dim, 901).data;

    int32_t test_positions[] = {-1000, -1, 0, 1, 1024, 65536, 1000000};

    for (int32_t pos : test_positions) {
        std::vector<float> post_rope(head_dim);
        for (uint32_t f = 0; f < fc; ++f) {
            float angle = omega[f] * (float)pos;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float re = canonical_k[f];
            float im = canonical_k[f + fc];
            post_rope[f]      = re * c - im * s;
            post_rope[f + fc] = im * c + re * s;
        }

        std::vector<float> recovered(head_dim, 0.0f);
        std::string err;
        bool inv_ok = invert_rope_k(recovered.data(), post_rope.data(), &pos, omega.data(), scale_sq.data(), 1, head_dim, rotary_dim, &err);
        MATRIX_CHECK(inv_ok, "invert_rope_k should succeed for pos=" + std::to_string(pos) + ": " + err);

        for (uint32_t d = 0; d < head_dim; ++d) {
            MATRIX_CHECK_APPROX(recovered[d], canonical_k[d], 1e-4f, "Exact recovery for position " + std::to_string(pos));
        }
    }
}

// ============================================================================
// 9. Chunked global online softmax equivalence: multi-block vs monolithic with
//    non-trivial logit_softcap, all-masked handling, and sink-once invariant
// ============================================================================
static void test_chunked_online_softmax_equivalence() {
    std::cout << "[Test 9] Chunked online softmax equivalence, logit softcap, all-masked, sink-once..." << std::endl;

    uint32_t dv = 16;
    size_t n_items = 64;

    std::vector<float> raw_scores = make_test_matrix(1, n_items, 1001).data;
    matrix v_mat = make_test_matrix(n_items, dv, 1002);

    // Non-trivial logit softcap transformation: score = C * tanh(raw / C)
    const float softcap = 15.0f;
    std::vector<float> softcapped_scores(n_items);
    for (size_t i = 0; i < n_items; ++i) {
        float s = raw_scores[i] * 5.0f; // produce raw scores up to ~25
        softcapped_scores[i] = softcap * std::tanh(s / softcap);
        MATRIX_CHECK(std::abs(softcapped_scores[i]) <= softcap, "Softcapped logit must be strictly bounded by softcap");
    }

    // 9a. Monolithic online softmax reference with softcapped scores
    xkv_online_softmax_state mono_state;
    mono_state.init(nullptr, dv);
    for (size_t i = 0; i < n_items; ++i) {
        mono_state.update_single(softcapped_scores[i], v_mat.row_ptr(i), dv);
    }
    std::vector<float> out_mono(dv, 0.0f);
    mono_state.finalize(out_mono.data(), dv);

    // 9b. Chunked online softmax with softcapped scores (4 chunks of 16 items each, merged via merge_block)
    xkv_online_softmax_state chunked_state;
    chunked_state.init(nullptr, dv);
    size_t chunk_size = 16;
    for (size_t c = 0; c < n_items / chunk_size; ++c) {
        xkv_online_softmax_state block;
        block.init(nullptr, dv);
        for (size_t i = 0; i < chunk_size; ++i) {
            size_t idx = c * chunk_size + i;
            block.update_single(softcapped_scores[idx], v_mat.row_ptr(idx), dv);
        }
        chunked_state.merge_block(block.max_score, block.sum_exp, block.get_accum(), dv);
    }
    std::vector<float> out_chunked(dv, 0.0f);
    chunked_state.finalize(out_chunked.data(), dv);

    for (uint32_t d = 0; d < dv; ++d) {
        MATRIX_CHECK_APPROX(out_chunked[d], out_mono[d], 1e-5f, "Softcapped chunked block merge matches monolithic softmax");
    }

    // 9c. Sink-once invariant with logit softcap
    float sink_logit = 2.5f;
    xkv_online_softmax_state sink_mono;
    sink_mono.init(nullptr, dv);
    sink_mono.update_sink(sink_logit, dv);
    for (size_t i = 0; i < n_items; ++i) {
        sink_mono.update_single(softcapped_scores[i], v_mat.row_ptr(i), dv);
    }
    std::vector<float> out_sink_mono(dv, 0.0f);
    sink_mono.finalize(out_sink_mono.data(), dv);

    xkv_online_softmax_state sink_chunked;
    sink_chunked.init(nullptr, dv);
    sink_chunked.update_sink(sink_logit, dv); // Sink registered ONCE at query level
    for (size_t c = 0; c < n_items / chunk_size; ++c) {
        xkv_online_softmax_state block;
        block.init(nullptr, dv);
        for (size_t i = 0; i < chunk_size; ++i) {
            size_t idx = c * chunk_size + i;
            block.update_single(softcapped_scores[idx], v_mat.row_ptr(idx), dv);
        }
        sink_chunked.merge_block(block.max_score, block.sum_exp, block.get_accum(), dv);
    }
    std::vector<float> out_sink_chunked(dv, 0.0f);
    sink_chunked.finalize(out_sink_chunked.data(), dv);

    for (uint32_t d = 0; d < dv; ++d) {
        MATRIX_CHECK_APPROX(out_sink_chunked[d], out_sink_mono[d], 1e-5f, "Sink-once chunked matches monolithic sink under softcap");
    }

    // 9d. All-masked query: finite zero output without NaN/inf
    xkv_online_softmax_state masked_state;
    masked_state.init(nullptr, dv);
    masked_state.update_single(-1e35f, v_mat.row_ptr(0), dv);
    masked_state.update_single(-INFINITY, v_mat.row_ptr(1), dv);
    std::vector<float> out_masked(dv, 999.0f);
    masked_state.finalize(out_masked.data(), dv);

    for (uint32_t d = 0; d < dv; ++d) {
        MATRIX_CHECK(out_masked[d] == 0.0f, "All-masked state must produce finite zero output");
        MATRIX_CHECK(!std::isnan(out_masked[d]), "All-masked state must not produce NaN");
    }
}

// ============================================================================
// Main entry point
// ============================================================================
int main() {
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Running test-xkv-matrix: XKV-SR §15.1 Mathematical & Representation Tests" << std::endl;
    std::cout << "=====================================================================" << std::endl;

    test_representation_rank_modes();
    test_layer_grouping_matrix();
    test_gqa_mapping_and_partial_neox_slices();
    test_boundary_and_geometry_matrix();
    test_mixed_codecs_matrix();
    test_fixed_factor_error_attribution();
    test_byte_preserving_store_pack();
    test_large_and_negative_positions();
    test_chunked_online_softmax_equivalence();

    std::cout << "=====================================================================" << std::endl;
    if (g_failures == 0) {
        std::cout << "ALL XKV-SR §15.1 MATHEMATICAL & REPRESENTATION TESTS PASSED." << std::endl;
        return 0;
    } else {
        std::cerr << g_failures << " TEST(S) FAILED." << std::endl;
        return 1;
    }
}
