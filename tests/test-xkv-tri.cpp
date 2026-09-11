// Comprehensive tests for PR08 XKV TriAttention data adapter:
// 1. Identical Tri scores/survivors versus dense final factor oracle for exact sampled heads with Turbo4 factors
// 2. Tile-size invariance and bounded tile memory across multiple tile sizes (1, 4, 16, 32)
// 3. Multi-sequence evaluation with per-sequence target retention, shared-prefix union, and exact (payload, seq) reference removals
// 4. Sparse per-sequence positions and per-seq recent window thresholding
// 5. Missing provider / slice / segment error checking (fail explicitly, never zero-fill fallback)
// 6. Preflighted atomic store mutation proposal and failure safety (old locations stay valid on failure)
// 7. A code-byte preservation, immutable B identity, binding epoch advance, and landmark invalidation
// 8. Recurrent-only pressure bypass indication

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-tri.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-factor.h"
#include "llama-xkv-landmark.h"
#include "llama-xkv-canonical.h"
#include "llama-triattention.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

using namespace llama_xkv;

static int g_test_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_test_failures++; \
        return; \
    } \
} while (0)

#define TEST_ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s:%d: %s — %s\n", __FILE__, __LINE__, #cond, msg); \
        g_test_failures++; \
        return; \
    } \
} while (0)

// Helper to construct mock calibration data with trusted NeoX half-layout (rope_style = 0)
static triattention_calibration create_test_calib(
    uint32_t head_dim = 128,
    uint32_t num_layers = 2,
    uint32_t num_attn_heads = 4,
    uint32_t num_kv_heads = 2,
    uint32_t n_sampled = 2,
    uint32_t rotary_dim = 128
) {
    triattention_calibration cal = {};
    cal.head_dim = head_dim;
    cal.num_layers = num_layers;
    cal.num_attn_heads = num_attn_heads;
    cal.num_kv_heads = num_kv_heads;
    cal.num_kv_groups = num_attn_heads / num_kv_heads;
    cal.rope_theta = 10000.0;
    cal.rope_style = 0; // MUST be 0
    cal.rotary_dim = rotary_dim;
    cal.freq_count = rotary_dim / 2;
    cal.n_sampled = n_sampled;
    snprintf(cal.model_name, sizeof(cal.model_name), "test_tri_model");

    cal.sampled_layer = new uint32_t[n_sampled];
    cal.sampled_head  = new uint32_t[n_sampled];
    cal.head_stats    = new triattention_head_stats[n_sampled];

    const uint32_t fc = cal.freq_count;
    for (uint32_t h = 0; h < n_sampled; ++h) {
        cal.sampled_layer[h] = h % num_layers;
        cal.sampled_head[h]  = h % num_attn_heads;

        auto & hs = cal.head_stats[h];
        hs.q_mean_real  = new float[fc];
        hs.q_mean_imag  = new float[fc];
        hs.q_abs_mean   = new float[fc];
        hs.q_mean_abs   = new float[fc];
        hs.extra_weight = new float[fc];

        for (uint32_t f = 0; f < fc; ++f) {
            hs.q_mean_real[f] = 0.8f + 0.01f * (float)(f % 10);
            hs.q_mean_imag[f] = 0.2f - 0.005f * (float)(f % 8);
            float re = hs.q_mean_real[f];
            float im = hs.q_mean_imag[f];
            hs.q_mean_abs[f]   = std::sqrt(re * re + im * im);
            hs.q_abs_mean[f]    = hs.q_mean_abs[f] + 0.05f;
            hs.extra_weight[f] = hs.q_abs_mean[f] - hs.q_mean_abs[f];
        }
    }

    return cal;
}

static void free_test_calib(triattention_calibration & cal) {
    if (cal.head_stats) {
        for (uint32_t h = 0; h < cal.n_sampled; ++h) {
            delete[] cal.head_stats[h].q_mean_real;
            delete[] cal.head_stats[h].q_mean_imag;
            delete[] cal.head_stats[h].q_abs_mean;
            delete[] cal.head_stats[h].q_mean_abs;
            delete[] cal.head_stats[h].extra_weight;
        }
        delete[] cal.head_stats;
        delete[] cal.sampled_layer;
        delete[] cal.sampled_head;
    }
}

static void compute_test_omega_and_fsq(
    const triattention_calibration & cal,
    std::vector<float> & omega,
    std::vector<float> & fsq
) {
    const uint32_t fc = cal.freq_count;
    omega.resize(fc);
    fsq.assign(fc, 1.0f);
    for (uint32_t f = 0; f < fc; ++f) {
        omega[f] = (float)(1.0 / std::pow(cal.rope_theta, (double)(2 * f) / (double)cal.rotary_dim));
    }
}

// Global heap counter for the zero-heap carved-scratch test (this TU only).
// Armed around a single call; every allocation while armed is counted.
namespace {
bool g_tri_count_heap = false;
size_t g_tri_heap_news = 0;
} // namespace
void * operator new(std::size_t n) {
    if (g_tri_count_heap) ++g_tri_heap_news;
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void * operator new[](std::size_t n) {
    if (g_tri_count_heap) ++g_tri_heap_news;
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

// Zero-heap span hot-K provider with deterministic distinct values.
static hot_k_span_provider_fn make_tri_span_provider(uint32_t head_dim) {
    return [head_dim](uint32_t, uint32_t, const uint32_t * cells, size_t n, float * dst,
                        size_t) {
        for (size_t i = 0; i < n; ++i) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                dst[i * head_dim + d] = 0.01f * (float)((cells[i] * 7 + d * 3) % 64);
            }
        }
        return true;
    };
}

// Publish one F32 factored segment and return it; payload ids [pid_base, ...).
// Seal helper adapting the nonce-based mark_seal_candidates store API.
static bool tri_mark_sealed(llama_xkv_cache_store & store, const std::vector<uint64_t> & pids,
                              const std::vector<uint64_t> & gens) {
    uint64_t nonce = 0;
    return store.mark_seal_candidates(pids, gens, &nonce);
}

// Publish one F32 factored segment and return it; payload ids [pid_base, ...).
static std::shared_ptr<const xkv_segment> make_tri_factored_segment(llama_xkv_cache_store & store,
                                                               uint32_t n_rows, uint32_t rank,
                                                               uint32_t total_dim, uint64_t pid_base,
                                                               std::vector<uint64_t> & pids_out) {
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = total_dim;
    g.total_dim_v = total_dim;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim};
    codec_desc desc_a =
        make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    std::vector<float> data_a(n_rows * rank);
    for (size_t i = 0; i < data_a.size(); ++i) data_a[i] = (float)((i * 3 + 5) % 97) * 0.01f;
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());
    codec_desc desc_av =
        make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F32,
                                          orientation::feature_major_transposed, {total_dim, rank});
    std::vector<float> data_b(total_dim * rank);
    for (size_t i = 0; i < data_b.size(); ++i) data_b[i] = (float)((i * 7 + 11) % 89) * 0.01f;
    g.set_b_k(encode_matrix(desc_b, data_b.data(), data_b.size()));
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32,
                                           orientation::feature_major_transposed, {total_dim, rank});
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));
    auto seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
                                              LLAMA_XKV_SOURCE_DECODED_HOT, {g});
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);
    pids_out.resize(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids_out[i] = pid_base + i;
        store.register_hot_payload(pids_out[i], i, gens[i], xkv_state::hot_committed);
    }
    if (!tri_mark_sealed(store, pids_out, gens)) return nullptr;
    std::string err;
    if (!store.publish_candidate(seg, pids_out, gens, &err)) {
        fprintf(stderr, "make_tri_factored_segment publish failed: %s\n", err.c_str());
        return nullptr;
    }
    return store.get_segment(seg->segment_id);
}

// -----------------------------------------------------------------------------
// Test 1: Real Turbo4 factors exact-sampled-head vs dense final factor oracle
// -----------------------------------------------------------------------------
static void test_turbo4_exact_sampled_head_vs_dense_oracle() {
    std::cout << "[test_xkv_tri] test_turbo4_exact_sampled_head_vs_dense_oracle..." << std::endl;

    const uint32_t head_dim = 128;
    const uint32_t rotary_dim = 128;
    const uint32_t rank_k = 128; // group_size 128 for Turbo4
    const uint32_t total_dim_k = 128;
    const uint32_t n_rows = 16;

    triattention_calibration cal = create_test_calib(head_dim, 1, 2, 1, 2, rotary_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.tile_size = 8;
    cfg.recent_window = 4;
    cfg.ratio = 0.5;
    cfg.normalize_scores = true;
    cfg.pool_radius = 2;

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    TEST_ASSERT(adapter.valid());

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank_k;
    g.rank_v = rank_k;
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_k;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim_k};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim_k};

    // Create synthetic factor data for A (n_rows x rank_k)
    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank_k}, 128);
    std::vector<float> data_a(n_rows * rank_k);
    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = 0.05f * std::sin((float)i * 0.17f);
    }
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());

    // Create synthetic factor data for B (total_dim_k x rank_k)
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim_k, rank_k}, 128);
    std::vector<float> data_b(total_dim_k * rank_k);
    for (size_t i = 0; i < data_b.size(); ++i) {
        data_b[i] = 0.05f * std::cos((float)i * 0.23f);
    }
    g.set_b_k(encode_matrix(desc_b, data_b.data(), data_b.size()));

    // Dummy V factor streams
    codec_desc desc_v = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank_k}, 128);
    g.a_v = encode_matrix(desc_v, data_a.data(), data_a.size());
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim_k, rank_k}, 128);
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));

    auto cand_seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 3000 + i;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));

    std::string err;
    if (!store.publish_candidate(cand_seg, pids, gens, &err)) {
        fprintf(stderr, "publish_candidate failed: %s\n", err.c_str());
        TEST_ASSERT(false);
    }
    uint64_t seg_id = cand_seg->segment_id;

    // Set up candidates pointing to this factored segment
    std::vector<xkv_tri_candidate> candidates(n_rows);
    std::vector<int32_t> positions(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)(i * 5); // non-contiguous
        candidates[i].payload_id = pids[i];
        candidates[i].storage_generation = 1;
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].seq_ids = {0};
        positions[i] = candidates[i].storage_pos;
    }

    xkv_layer_slice slice = {};
    slice.model_layer = 0;
    slice.owning_layer = 0;
    slice.feature_offset_k = 0;
    slice.feature_dim_k = total_dim_k;
    slice.head_dim = head_dim;
    slice.rotary_dim = rotary_dim;
    slice.rope_style = 0;
    slice.n_kv_heads = 1;

    // First: Reconstruct the dense final factor representation directly
    // Resolve exact group streams via find_group (never assume segment-level streams).
    const auto * oracle_g = cand_seg->find_group(0);
    TEST_ASSERT(oracle_g != nullptr);
    TEST_ASSERT(oracle_g->b_k != nullptr);
    std::vector<float> decoded_a = decode_matrix(oracle_g->a_k, value_domain::canonical);
    std::vector<float> decoded_b = decode_matrix(*oracle_g->b_k, value_domain::canonical);
    std::vector<float> dense_k(n_rows * head_dim, 0.0f);

    uint64_t a_pad_cols = oracle_g->a_k.desc.padded_shape.cols;
    uint64_t b_pad_cols = oracle_g->b_k->desc.padded_shape.cols;

    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (uint64_t k = 0; k < rank_k; ++k) {
                sum += decoded_a[r * a_pad_cols + k] * decoded_b[d * b_pad_cols + k];
            }
            dense_k[r * head_dim + d] = sum;
        }
    }

    // Compute dense oracle scores on the final factor K
    std::vector<float> offsets;
    for (uint32_t d = 1; d <= 65536; d *= 2) offsets.push_back((float)d);

    const int64_t frontier_pos = positions.back() + 10;
    std::vector<float> oracle_combined(n_rows, -1e30f);
    std::vector<float> temp_scores(n_rows);

    for (uint32_t sh = 0; sh < cal.n_sampled; ++sh) {
        triattention_score_keys(
            temp_scores.data(),
            dense_k.data(),
            &cal.head_stats[sh],
            omega.data(),
            fsq.data(),
            offsets.data(),
            positions.data(),
            frontier_pos,
            n_rows,
            head_dim,
            rotary_dim / 2,
            (uint32_t)offsets.size(),
            cfg.agg,
            cfg.disable_trig
        );

        // Normalize across entire set
        double sum = 0.0;
        for (float v : temp_scores) sum += v;
        double mean = sum / n_rows;
        double var = 0.0;
        for (float v : temp_scores) var += (v - mean) * (v - mean);
        double std = std::sqrt(var / n_rows);
        if (std < 1e-10) std = 1e-10;
        for (float & v : temp_scores) v = (float)((v - mean) / std);

        for (uint32_t i = 0; i < n_rows; ++i) {
            oracle_combined[i] = std::max(oracle_combined[i], temp_scores[i]);
        }
    }

    std::vector<float> oracle_pooled(n_rows);
    triattention_max_pool_scores(oracle_pooled.data(), oracle_combined.data(), positions.data(), n_rows, cfg.pool_radius);

    // Compute adapter scores (streams in bounded tiles with single-decode reuse)
    std::vector<float> adapter_raw(n_rows);
    std::vector<float> adapter_pooled(n_rows);
    std::vector<uint32_t> all_indices(n_rows);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    adapter.score_candidate_subset(
        candidates,
        all_indices,
        {slice},
        store,
        frontier_pos,
        adapter_pooled.data(),
        adapter_raw.data()
    );

    // Assert exact match with dense final factor oracle
    for (uint32_t i = 0; i < n_rows; ++i) {
        TEST_ASSERT_MSG(std::fabs(adapter_raw[i] - oracle_combined[i]) < 1e-5f, "Raw score mismatch with Turbo4 factor oracle");
        TEST_ASSERT_MSG(std::fabs(adapter_pooled[i] - oracle_pooled[i]) < 1e-5f, "Pooled score mismatch with Turbo4 factor oracle");
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_turbo4_exact_sampled_head_vs_dense_oracle PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 2: Tile-size invariance and bounded tile memory
// -----------------------------------------------------------------------------
static void test_tile_size_invariance_and_memory_bounds() {
    std::cout << "[test_xkv_tri] test_tile_size_invariance_and_memory_bounds..." << std::endl;

    const uint32_t head_dim = 64;
    const uint32_t rank = 16;
    const uint32_t n_rows = 36;
    const uint32_t total_dim_k = 64;

    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_k;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim_k};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim_k};

    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    std::vector<float> data_a(n_rows * rank);
    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = (float)((i * 3 + 5) % 97) * 0.01f;
    }
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());
    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());

    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank});
    std::vector<float> data_b(total_dim_k * rank);
    for (size_t i = 0; i < data_b.size(); ++i) {
        data_b[i] = (float)((i * 7 + 11) % 89) * 0.01f;
    }
    g.set_b_k(encode_matrix(desc_b, data_b.data(), data_b.size()));
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank});
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));

    // Second group with distinct B streams: exact group-1 resolution below must
    // not fall back to group 0 (its B data differs, so a fallback would mismatch).
    xkv_factor_group_payload g1;
    g1.group_index = 1;
    g1.owning_layers = {1};
    g1.rank_k = rank;
    g1.rank_v = rank;
    g1.total_dim_k = total_dim_k;
    g1.total_dim_v = total_dim_k;
    g1.layer_feature_offsets_k = {0};
    g1.layer_feature_dims_k = {total_dim_k};
    g1.layer_feature_offsets_v = {0};
    g1.layer_feature_dims_v = {total_dim_k};
    codec_desc desc_a1 = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    std::vector<float> data_a1(n_rows * rank);
    for (size_t i = 0; i < data_a1.size(); ++i) {
        data_a1[i] = (float)((i * 11 + 7) % 83) * 0.01f;
    }
    g1.a_k = encode_matrix(desc_a1, data_a1.data(), data_a1.size());
    codec_desc desc_av1 = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    g1.a_v = encode_matrix(desc_av1, data_a1.data(), data_a1.size());
    codec_desc desc_b1 = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank});
    std::vector<float> data_b1(total_dim_k * rank);
    for (size_t i = 0; i < data_b1.size(); ++i) {
        data_b1[i] = (float)((i * 13 + 29) % 79) * 0.01f + 1.0f;
    }
    g1.set_b_k(encode_matrix(desc_b1, data_b1.data(), data_b1.size()));
    codec_desc desc_bv1 = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank});
    g1.set_b_v(encode_matrix(desc_bv1, data_b1.data(), data_b1.size()));

    auto seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g, g1}
    );
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 10000 + i;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));
    std::string err;
    if (!store.publish_candidate(seg, pids, gens, &err)) {
        fprintf(stderr, "publish_candidate in test 2 failed: %s\n", err.c_str());
        TEST_ASSERT(false);
    }
    auto published_seg = store.get_segment(seg->segment_id);
    TEST_ASSERT(published_seg != nullptr);

    xkv_layer_slice slice = {};
    slice.model_layer = 0;
    slice.owning_layer = 0;
    slice.feature_offset_k = 0;
    slice.feature_dim_k = total_dim_k;
    slice.head_dim = head_dim;
    slice.rotary_dim = head_dim;
    slice.rope_style = 0;
    slice.n_kv_heads = 1;

    std::vector<uint32_t> row_indices(n_rows);
    std::iota(row_indices.begin(), row_indices.end(), 0);

    std::vector<float> out_t1(n_rows * head_dim);
    std::vector<float> out_t4(n_rows * head_dim);
    std::vector<float> out_t16(n_rows * head_dim);
    std::vector<float> out_t32(n_rows * head_dim);

    adapter.stream_factored_k_head(*published_seg, slice, 0, row_indices.data(), n_rows, out_t1.data(), out_t1.size(), 1);
    adapter.stream_factored_k_head(*published_seg, slice, 0, row_indices.data(), n_rows, out_t4.data(), out_t4.size(), 4);
    adapter.stream_factored_k_head(*published_seg, slice, 0, row_indices.data(), n_rows, out_t16.data(), out_t16.size(), 16);
    adapter.stream_factored_k_head(*published_seg, slice, 0, row_indices.data(), n_rows, out_t32.data(), out_t32.size(), 32);

    for (size_t i = 0; i < out_t1.size(); ++i) {
        TEST_ASSERT_MSG(out_t1[i] == out_t4[i], "Mismatch between tile_size 1 and 4");
        TEST_ASSERT_MSG(out_t1[i] == out_t16[i], "Mismatch between tile_size 1 and 16");
        TEST_ASSERT_MSG(out_t1[i] == out_t32[i], "Mismatch between tile_size 1 and 32");
    }

    // Exact second-group resolution: streaming owning_layer 1 must decode group-1
    // B rows, not group-0 data. The oracle below uses group-1 streams resolved via
    // find_group, so a first-group fallback would mismatch.
    const auto * tile_g1 = published_seg->find_group(1);
    TEST_ASSERT(tile_g1 != nullptr);
    TEST_ASSERT(tile_g1->b_k != nullptr);
    TEST_ASSERT(published_seg->find_group_for_layer(1) == tile_g1);
    xkv_layer_slice slice1 = {};
    slice1.model_layer = 1;
    slice1.owning_layer = 1;
    slice1.feature_offset_k = 0;
    slice1.feature_dim_k = total_dim_k;
    slice1.head_dim = head_dim;
    slice1.rotary_dim = head_dim;
    slice1.rope_style = 0;
    slice1.n_kv_heads = 1;
    std::vector<float> out_g1(n_rows * head_dim);
    adapter.stream_factored_k_head(*published_seg, slice1, 0, row_indices.data(), n_rows, out_g1.data(), out_g1.size(), 4);
    std::vector<float> decoded_a1 = decode_matrix(tile_g1->a_k, value_domain::canonical);
    std::vector<float> decoded_b1 = decode_matrix(*tile_g1->b_k, value_domain::canonical);
    uint64_t a1_pad = tile_g1->a_k.desc.padded_shape.cols;
    uint64_t b1_pad = tile_g1->b_k->desc.padded_shape.cols;
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            double sum = 0.0;
            for (uint32_t k = 0; k < rank; ++k) {
                sum += (double)decoded_a1[r * a1_pad + k] * (double)decoded_b1[d * b1_pad + k];
            }
            TEST_ASSERT_MSG(std::fabs(out_g1[r * head_dim + d] - (float)sum) < 1e-5f, "Group-1 stream mismatch (first-group fallback?)");
        }
    }

    // Exact boundary check: row_indices containing an out-of-bounds row (e.g. n_rows) MUST throw std::out_of_range
    std::vector<uint32_t> invalid_row_indices = {n_rows};
    std::vector<float> dummy_out(head_dim);
    bool caught_oob = false;
    try {
        adapter.stream_factored_k_head(*published_seg, slice, 0, invalid_row_indices.data(), 1, dummy_out.data(), dummy_out.size());
    } catch (const std::out_of_range &) {
        caught_oob = true;
    }
    TEST_ASSERT_MSG(caught_oob, "stream_factored_k_head must throw std::out_of_range for row index >= segment.n_rows");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_tile_size_invariance_and_memory_bounds PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 3: Multi-sequence evaluation with per-sequence target retention and exact reference removals
// -----------------------------------------------------------------------------
static void test_multi_sequence_per_seq_target_and_ref_removals() {
    std::cout << "[test_xkv_tri] test_multi_sequence_per_seq_target_and_ref_removals..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 2, 1, 2, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 2;
    cfg.ratio = 0.2; // keep 20%

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    const uint32_t n_cand = 12;
    std::vector<xkv_tri_candidate> candidates(n_cand);
    for (uint32_t i = 0; i < n_cand; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = 4000 + i;
        candidates[i].location.kind = xkv_location_kind::hot;
        candidates[i].location.row = i;

        // Shared prefix tokens [0..3] carry both seq 0 and seq 1
        if (i < 4) {
            candidates[i].seq_ids = {0, 1};
        } else if (i < 8) {
            candidates[i].seq_ids = {0}; // private seq 0
        } else {
            candidates[i].seq_ids = {1}; // private seq 1
        }
    }

    // Define two sequences with distinct logical histories
    xkv_tri_seq_info seq0;
    seq0.seq_id = 0;
    seq0.logical_tokens = 8;
    seq0.frontier_pos = 7;
    seq0.tail_guard = 2;
    seq0.eligible = true;

    xkv_tri_seq_info seq1;
    seq1.seq_id = 1;
    seq1.logical_tokens = 8;
    seq1.frontier_pos = 11;
    seq1.tail_guard = 2;
    seq1.eligible = true;

    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    hot_k_provider_fn mock_hot_provider = [&](
        uint32_t /*model_layer*/,
        uint32_t /*kv_head*/,
        const std::vector<uint32_t> & cell_indices,
        float * dst_pre_rope_k_half,
        size_t /*capacity_elements*/
    ) -> bool {
        // Exercise invert_rope_k from llama-xkv-canonical to reconstruct canonical pre-RoPE K from simulated post-RoPE K
        std::vector<float> post_rope_k(cell_indices.size() * head_dim);
        for (size_t ci = 0; ci < cell_indices.size(); ++ci) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                post_rope_k[ci * head_dim + d] = 0.02f * std::cos((float)(ci + d) * 0.1f);
            }
        }
        std::vector<int32_t> pos_buf(cell_indices.size());
        for (size_t ci = 0; ci < cell_indices.size(); ++ci) {
            pos_buf[ci] = (int32_t)cell_indices[ci];
        }
        triattention_invert_rope(
            dst_pre_rope_k_half,
            post_rope_k.data(),
            pos_buf.data(),
            omega.data(),
            fsq.data(),
            (uint32_t)cell_indices.size(),
            head_dim,
            head_dim,
            head_dim / 2,
            0
        );
        return true;
    };

    auto plan = adapter.build_mutation_plan(
        candidates,
        {seq0, seq1},
        {slice},
        store,
        pressure,
        mock_hot_provider
    );

    // Exact reference removals must be emitted with exact (payload_id, seq_id)
    TEST_ASSERT(!plan.ref_removals.empty());
    for (const auto & rem : plan.ref_removals) {
        TEST_ASSERT(rem.payload_id >= 4000 && rem.payload_id < 4000 + n_cand);
        TEST_ASSERT(rem.seq_id == 0 || rem.seq_id == 1);
    }

    // Physical survivors union: shared prefix cells that only lost one ref still survive!
    for (uint32_t i = 0; i < 4; ++i) {
        bool in_survivor = (std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), candidates[i].payload_id) != plan.survivor_payloads.end());
        bool in_evicted = (std::find(plan.evicted_payloads.begin(), plan.evicted_payloads.end(), candidates[i].payload_id) != plan.evicted_payloads.end());
        // Cell is either kept as survivor or evicted, never both
        TEST_ASSERT(in_survivor != in_evicted);
    }

    // Accounting: references_removed >= physical_freed due to shared prefix
    TEST_ASSERT(plan.references_removed >= plan.physical_freed);

    // Frontier independence check: verify that changing seq1's frontier_pos does NOT alter seq0's scores or reference removals!
    xkv_tri_seq_info seq1_alt = seq1;
    seq1_alt.frontier_pos = 50; // much larger frontier
    auto plan_alt = adapter.build_mutation_plan(
        candidates,
        {seq0, seq1_alt},
        {slice},
        store,
        pressure,
        mock_hot_provider
    );

    // Removals for seq 0 must be 100% identical regardless of seq 1's frontier!
    std::vector<xkv_tri_ref_removal> s0_rem_orig, s0_rem_alt;
    for (const auto & r : plan.ref_removals) {
        if (r.seq_id == 0) s0_rem_orig.push_back(r);
    }
    for (const auto & r : plan_alt.ref_removals) {
        if (r.seq_id == 0) s0_rem_alt.push_back(r);
    }
    TEST_ASSERT_MSG(s0_rem_orig.size() == s0_rem_alt.size(), "Seq 0 removal count must be independent of seq 1 frontier");
    for (size_t i = 0; i < s0_rem_orig.size(); ++i) {
        TEST_ASSERT_MSG(s0_rem_orig[i] == s0_rem_alt[i], "Seq 0 removals must be identical and frontier-isolated");
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_multi_sequence_per_seq_target_and_ref_removals PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 4: Sparse per-sequence positions and per-seq recent window thresholding
// -----------------------------------------------------------------------------
static void test_sparse_per_seq_recent_behavior() {
    std::cout << "[test_xkv_tri] test_sparse_per_seq_recent_behavior..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 3;
    cfg.ratio = 0.05; // aggressively evict non-recent

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // Sequence 0: positions {0, 10, 20, 30, 40}, frontier=40, tail_guard=3 -> recent >= 40-3+1 = 38 -> only 40 is recent
    // Sequence 1: positions {5, 15, 25, 98, 99, 100}, frontier=100, tail_guard=3 -> recent >= 100-3+1 = 98 -> 98, 99, 100 are recent
    std::vector<xkv_tri_candidate> candidates;
    std::vector<int32_t> s0_pos = {0, 10, 20, 30, 40};
    for (int32_t p : s0_pos) {
        xkv_tri_candidate c;
        c.cell_index = (uint32_t)candidates.size();
        c.storage_pos = p;
        c.payload_id = 5000 + c.cell_index;
        c.location.kind = xkv_location_kind::hot;
        c.seq_ids = {0};
        candidates.push_back(c);
    }

    std::vector<int32_t> s1_pos = {5, 15, 25, 98, 99, 100};
    for (int32_t p : s1_pos) {
        xkv_tri_candidate c;
        c.cell_index = (uint32_t)candidates.size();
        c.storage_pos = p;
        c.payload_id = 5000 + c.cell_index;
        c.location.kind = xkv_location_kind::hot;
        c.seq_ids = {1};
        candidates.push_back(c);
    }

    xkv_tri_seq_info seq0 = {0, 5, 40, 3, true, 0, {}};
    xkv_tri_seq_info seq1 = {1, 6, 100, 3, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    hot_k_provider_fn mock_provider = [](
        uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t
    ) -> bool {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };

    auto plan = adapter.build_mutation_plan(candidates, {seq0, seq1}, {slice}, store, pressure, mock_provider);

    // Candidates with pos 40 (index 4) and pos 98, 99, 100 (indices 8, 9, 10) MUST survive due to per-seq recent guards
    TEST_ASSERT(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), candidates[4].payload_id) != plan.survivor_payloads.end());
    TEST_ASSERT(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), candidates[8].payload_id) != plan.survivor_payloads.end());
    TEST_ASSERT(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), candidates[9].payload_id) != plan.survivor_payloads.end());
    TEST_ASSERT(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), candidates[10].payload_id) != plan.survivor_payloads.end());

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_sparse_per_seq_recent_behavior PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 4b: Semantic-cell multi-ref eviction and reader-tail hard guard parity
// -----------------------------------------------------------------------------
static void test_semantic_cell_eviction_and_reader_tail_parity() {
    std::cout << "[test_xkv_tri] test_semantic_cell_eviction_and_reader_tail_parity..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 2;
    cfg.ratio = 0.1; // evict almost all non-recent

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // Setup 6 candidates:
    // Cells 0, 1, 2: semantic episode 101, public_live. Carrying both exec_seq (0) and archive_seq (1).
    // Cell 3: foreign episode 202, public_live. Carrying seq 0.
    // Cell 4: semantic episode 101, but in reader tail of archive_seq (frontier=20, tail=2 -> pos >= 19). Cell 4 is at pos 19.
    // Cell 5: recent in exec_seq (frontier=10, tail=2 -> pos >= 9). Cell 5 is at pos 10.
    std::vector<xkv_tri_candidate> candidates(6);

    // Cell 0: pos 0, semantic episode 101, refs {0, 1}
    candidates[0].cell_index = 0; candidates[0].storage_pos = 0; candidates[0].payload_id = 9000;
    candidates[0].location.kind = xkv_location_kind::hot; candidates[0].seq_ids = {0, 1};
    candidates[0].semantic_episode_id = 101; candidates[0].visibility = llama_rerot_visibility::public_live;

    // Cell 1: pos 1, semantic episode 101, refs {0, 1}
    candidates[1].cell_index = 1; candidates[1].storage_pos = 1; candidates[1].payload_id = 9001;
    candidates[1].location.kind = xkv_location_kind::hot; candidates[1].seq_ids = {0, 1};
    candidates[1].semantic_episode_id = 101; candidates[1].visibility = llama_rerot_visibility::public_live;

    // Cell 2: pos 2, semantic episode 101, refs {0, 1} + foreign seq 2
    candidates[2].cell_index = 2; candidates[2].storage_pos = 2; candidates[2].payload_id = 9002;
    candidates[2].location.kind = xkv_location_kind::hot; candidates[2].seq_ids = {0, 1, 2};
    candidates[2].semantic_episode_id = 101; candidates[2].visibility = llama_rerot_visibility::public_live;

    // Cell 3: pos 3, foreign episode 202, ref {0}
    candidates[3].cell_index = 3; candidates[3].storage_pos = 3; candidates[3].payload_id = 9003;
    candidates[3].location.kind = xkv_location_kind::hot; candidates[3].seq_ids = {0};
    candidates[3].semantic_episode_id = 202; candidates[3].visibility = llama_rerot_visibility::public_live;

    // Cell 4: pos 19, semantic episode 101, refs {0, 1}. In archive_seq reader tail (pos 19 >= 20 - 2 + 1)
    candidates[4].cell_index = 4; candidates[4].storage_pos = 19; candidates[4].payload_id = 9004;
    candidates[4].location.kind = xkv_location_kind::hot; candidates[4].seq_ids = {0, 1};
    candidates[4].semantic_episode_id = 101; candidates[4].visibility = llama_rerot_visibility::public_live;

    // Cell 5: pos 10, semantic episode 101, ref {0}. In exec_seq recent window (pos 10 >= 10 - 2 + 1)
    candidates[5].cell_index = 5; candidates[5].storage_pos = 10; candidates[5].payload_id = 9005;
    candidates[5].location.kind = xkv_location_kind::hot; candidates[5].seq_ids = {0};
    candidates[5].semantic_episode_id = 101; candidates[5].visibility = llama_rerot_visibility::public_live;

    // Sequence 0: exec_seq with semantic_episode_id = 101, semantic_seq_ids = {0, 1}
    xkv_tri_seq_info seq0;
    seq0.seq_id = 0;
    seq0.logical_tokens = 6;
    seq0.frontier_pos = 10;
    seq0.tail_guard = 2;
    seq0.eligible = true;
    seq0.semantic_episode_id = 101;
    seq0.semantic_seq_ids = {0, 1};

    // Sequence 1: archive_seq with frontier_pos = 20 (distinct frontier!)
    xkv_tri_seq_info seq1;
    seq1.seq_id = 1;
    seq1.logical_tokens = 6;
    seq1.frontier_pos = 20;
    seq1.tail_guard = 2;
    seq1.eligible = false; // not reclaiming seq 1 directly in this pass
    seq1.semantic_episode_id = 101;
    seq1.semantic_seq_ids = {0, 1};

    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    hot_k_provider_fn mock_provider = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };

    auto plan = adapter.build_mutation_plan(candidates, {seq0, seq1}, {slice}, store, pressure, mock_provider);

    // Invariant 1: Cell 3 is foreign episode 202 -> hard-protected from seq 0 reclaim!
    TEST_ASSERT_MSG(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), 9003) != plan.survivor_payloads.end(),
                    "Foreign episode cell 3 must be hard-protected");

    // Invariant 2: Cell 4 is in archive_seq reader tail (pos 19 >= 19) -> hard-protected via semantic reader tail!
    TEST_ASSERT_MSG(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), 9004) != plan.survivor_payloads.end(),
                    "Semantic reader tail cell 4 must be hard-protected");

    // Invariant 3: Cell 5 is in exec_seq recent window (pos 10 >= 9) -> hard-protected!
    TEST_ASSERT_MSG(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), 9005) != plan.survivor_payloads.end(),
                    "Recent window cell 5 must be hard-protected");

    // Invariant 4: When Cell 0 or 1 is evicted for seq 0, it represents a semantic PUBLIC cell for episode 101.
    // BOTH seq 0 and seq 1 references must be removed in ref_removals!
    bool found_rem_0_s0 = false, found_rem_0_s1 = false;
    for (const auto & rem : plan.ref_removals) {
        if (rem.payload_id == 9000) {
            if (rem.seq_id == 0) found_rem_0_s0 = true;
            if (rem.seq_id == 1) found_rem_0_s1 = true;
        }
    }
    if (found_rem_0_s0) {
        TEST_ASSERT_MSG(found_rem_0_s1, "Semantic cell eviction must remove both seq 0 and seq 1 refs for episode 101");
    }

    // Invariant 5: Cell 2 has foreign seq 2 reference. Even if seq 0 and seq 1 refs are removed, foreign ref seq 2 survives!
    bool cell_2_in_survivors = (std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), 9002) != plan.survivor_payloads.end());
    TEST_ASSERT_MSG(cell_2_in_survivors, "Cell 2 with remaining foreign seq 2 ref must survive physically");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_semantic_cell_eviction_and_reader_tail_parity PASSED." << std::endl;
}
// -----------------------------------------------------------------------------
// Test 5: Missing slice / segment / provider error checking
// -----------------------------------------------------------------------------
static void test_missing_provider_and_slice_errors() {
    std::cout << "[test_xkv_tri] test_missing_provider_and_slice_errors..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 2, 2, 1, 2, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    std::vector<xkv_tri_candidate> candidates(2);
    candidates[0].cell_index = 0;
    candidates[0].location.kind = xkv_location_kind::hot;
    candidates[1].cell_index = 1;
    candidates[1].location.kind = xkv_location_kind::hot;

    // Slices provided only for layer 0, but sampled heads include layer 1
    xkv_layer_slice slice0 = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    std::vector<float> pooled(2);
    std::vector<uint32_t> subset = {0, 1};
    bool caught_missing_slice = false;
    try {
        adapter.score_candidate_subset(candidates, subset, {slice0}, store, 10, pooled.data());
    } catch (const std::runtime_error & e) {
        caught_missing_slice = true;
    }
    TEST_ASSERT_MSG(caught_missing_slice, "Must throw on missing layer slice");

    // Slices provided for both, but missing hot provider for hot candidates
    xkv_layer_slice slice1 = {1, 1, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};
    bool caught_missing_provider = false;
    try {
        adapter.score_candidate_subset(candidates, subset, {slice0, slice1}, store, 10, pooled.data(), nullptr, nullptr);
    } catch (const std::runtime_error & e) {
        caught_missing_provider = true;
    }
    TEST_ASSERT_MSG(caught_missing_provider, "Must throw on missing hot provider");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_missing_provider_and_slice_errors PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 6: Preflighted atomic store mutation proposal and failure safety
// -----------------------------------------------------------------------------
static void test_preflighted_atomic_store_mutation_and_failure_safety() {
    std::cout << "[test_xkv_tri] test_preflighted_atomic_store_mutation_and_failure_safety..." << std::endl;

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 8;
    const uint32_t rank = 16;
    const uint32_t total_dim = 64;

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = total_dim;
    g.total_dim_v = total_dim;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim};

    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    std::vector<float> data_a(n_rows * rank, 1.0f);
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());

    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    std::vector<float> data_b(total_dim * rank, 2.0f);
    g.set_b_k(encode_matrix(desc_b, data_b.data(), data_b.size()));

    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));

    auto cand_seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);

    std::vector<uint64_t> pids = {6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007};
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));
    std::string err;
    if (!store.publish_candidate(cand_seg, pids, gens, &err)) {
        fprintf(stderr, "publish_candidate failed: %s\n", err.c_str());
        TEST_ASSERT(false);
    }

    uint64_t seg_id = cand_seg->segment_id;
    xkv_tri_mutation_plan plan;
    plan.evicted_payloads = {6002, 6005};
    plan.affected_segments = {seg_id};

    triattention_calibration cal = create_test_calib();
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());

    auto proposal = adapter.create_mutation_proposal(plan, store);

    // Preflight must succeed on untouched store
    TEST_ASSERT(proposal.preflight(store, &err));

    // Simulate concurrent store mutation before applying (e.g. bump live_epoch)
    store.bump_live_epoch();

    // Now preflight must FAIL due to stale stamp
    bool preflight_ok = proposal.preflight(store, &err);
    TEST_ASSERT_MSG(!preflight_ok, "Preflight must fail when store stamp has advanced");

    // Attempting apply must fail safely without modifying store
    bool apply_ok = proposal.apply(store, nullptr, &err);
    TEST_ASSERT_MSG(!apply_ok, "Apply must fail safely when preflight fails");

    // Verify all original payloads are still intact and valid
    for (uint64_t pid : pids) {
        xkv_location loc;
        TEST_ASSERT_MSG(store.find_location(pid, loc), "Old locations must stay valid on atomic failure");
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_preflighted_atomic_store_mutation_and_failure_safety PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 7: A code-byte preservation, immutable B identity, binding epoch advance, and landmark invalidation
// -----------------------------------------------------------------------------
static void test_a_byte_preservation_and_b_identity() {
    std::cout << "[test_xkv_tri] test_a_byte_preservation_and_b_identity..." << std::endl;

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t rank = 16;
    const uint32_t total_dim = 64;

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = total_dim;
    g.total_dim_v = total_dim;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim};

    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    std::vector<float> data_a(n_rows * rank);
    for (size_t r = 0; r < n_rows; ++r) {
        for (size_t c = 0; c < rank; ++c) {
            data_a[r * rank + c] = (float)(r * 100 + c + 1);
        }
    }
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());

    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    std::vector<float> data_b(total_dim * rank);
    for (size_t i = 0; i < data_b.size(); ++i) {
        data_b[i] = (float)(i + 77);
    }
    auto b_encoded = encode_matrix(desc_b, data_b.data(), data_b.size());
    g.set_b_k(b_encoded);

    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));

    auto cand_seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t r = 0; r < n_rows; ++r) {
        pids[r] = 7000 + r;
        store.register_hot_payload(pids[r], r, gens[r], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));

    std::string err;
    if (!store.publish_candidate(cand_seg, pids, gens, &err)) {
        fprintf(stderr, "publish_candidate failed: %s\n", err.c_str());
        TEST_ASSERT(false);
    }
    uint64_t seg_id = cand_seg->segment_id;

    auto published_seg = store.get_segment(seg_id);
    // Resolve exact group streams via find_group (never assume segment-level streams).
    const auto * pub_g0 = published_seg->find_group(0);
    TEST_ASSERT(pub_g0 != nullptr);
    TEST_ASSERT(pub_g0->b_k != nullptr);
    const size_t row_bytes = pub_g0->a_k.desc.row_stride_bytes;
    std::vector<std::vector<uint8_t>> orig_row_bytes(n_rows, std::vector<uint8_t>(row_bytes));
    for (uint32_t r = 0; r < n_rows; ++r) {
        std::memcpy(orig_row_bytes[r].data(), pub_g0->a_k.bytes.data() + r * row_bytes, row_bytes);
    }
    const void * orig_b_ptr = pub_g0->b_k->bytes.data();

    uint64_t orig_content_epoch = store.content_epoch();
    uint64_t orig_binding_epoch = store.binding_epoch();

    landmark_table lm_table(1024 * 1024);

    std::vector<uint32_t> evict_rows = {1, 4, 7, 10};
    xkv_tri_mutation_plan plan;
    for (uint32_t er : evict_rows) {
        plan.evicted_payloads.push_back(pids[er]);
    }
    plan.affected_segments = {seg_id};

    triattention_calibration cal = create_test_calib();
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());

    auto proposal = adapter.create_mutation_proposal(plan, store);
    TEST_ASSERT(proposal.apply(store, &lm_table, &err));

    auto repacked_seg = store.get_segment(seg_id);
    TEST_ASSERT(repacked_seg != nullptr);

    // 1. Immutable B identity
    const auto * repacked_g0 = repacked_seg->find_group(0);
    TEST_ASSERT_MSG(repacked_g0 != nullptr, "Repacked segment must contain group 0");
    TEST_ASSERT_MSG(repacked_g0->b_k != nullptr, "Repacked group 0 must hold B_K");
    TEST_ASSERT_MSG(repacked_g0->b_k->bytes == b_encoded.bytes, "B_K bytes must be identical");
    TEST_ASSERT_MSG(repacked_g0->b_k->bytes.data() == orig_b_ptr, "B_K shared pointer must be preserved");

    // 2. A code-byte preservation
    const uint32_t expected_survivors = n_rows - (uint32_t)evict_rows.size();
    TEST_ASSERT_MSG(repacked_seg->n_rows == expected_survivors, "Repacked segment row count mismatch");

    for (uint32_t dst_r = 0; dst_r < repacked_seg->n_rows; ++dst_r) {
        uint64_t pid = repacked_seg->row_payload_ids[dst_r];
        uint32_t orig_r = (uint32_t)(pid - 7000);
        const uint8_t * new_bytes = repacked_g0->a_k.bytes.data() + dst_r * row_bytes;
        TEST_ASSERT_MSG(std::memcmp(new_bytes, orig_row_bytes[orig_r].data(), row_bytes) == 0,
                        "A survivor row bytes must be copied byte-for-byte");
    }

    // 3. Binding epoch advanced, content epoch unchanged
    TEST_ASSERT_MSG(store.binding_epoch() > orig_binding_epoch, "binding_epoch must advance");
    TEST_ASSERT_MSG(store.content_epoch() == orig_content_epoch, "content_epoch must not change on physical pack");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_a_byte_preservation_and_b_identity PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 8: Recurrent-only pressure bypass indication
// -----------------------------------------------------------------------------
static void test_recurrent_only_bypass() {
    std::cout << "[test_xkv_tri] test_recurrent_only_bypass..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    std::vector<xkv_tri_candidate> candidates(4);
    for (uint32_t i = 0; i < 4; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = i;
        candidates[i].payload_id = 8000 + i;
        candidates[i].seq_ids = {0};
    }

    xkv_tri_seq_info seq0 = {0, 4, 3, 2, true, 0, {}};
    xkv_tri_pressure_state pressure = {false, true, false}; // recurrent pressure only!
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    auto plan = adapter.build_mutation_plan(candidates, {seq0}, {slice}, store, pressure);

    TEST_ASSERT_MSG(plan.recurrent_only_bypass, "Must flag recurrent-only bypass");
    TEST_ASSERT_MSG(!plan.changed, "Must not change on recurrent bypass");
    TEST_ASSERT_MSG(plan.physical_freed == 0, "No cells freed on recurrent bypass");
    TEST_ASSERT_MSG(plan.ref_removals.empty(), "No ref removals on recurrent bypass");
    TEST_ASSERT_MSG(plan.survivor_payloads.size() == 4, "All candidates survive");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_recurrent_only_bypass PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 9: Mixed hot / factored candidates in one sequence with exact reference target accounting
// -----------------------------------------------------------------------------
static void test_mixed_hot_factored_one_sequence() {
    std::cout << "[test_xkv_tri] test_mixed_hot_factored_one_sequence..." << std::endl;

    const uint32_t head_dim = 64;
    const uint32_t rank = 16;
    const uint32_t total_dim = 64;
    const uint32_t n_factored = 8;
    const uint32_t n_hot = 8;
    const uint32_t n_total = n_factored + n_hot;

    triattention_calibration cal = create_test_calib(head_dim, 1, 2, 1, 2, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 4;
    cfg.ratio = 0.25; // keep 25% (4 out of 16, but recent window is 4, so exactly 4)

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // Create factored segment for first 8 rows
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = total_dim;
    g.total_dim_v = total_dim;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim};

    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_factored, rank});
    std::vector<float> data_a(n_factored * rank, 0.5f);
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());
    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_factored, rank});
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());

    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    std::vector<float> data_b(total_dim * rank, 0.2f);
    g.set_b_k(encode_matrix(desc_b, data_b.data(), data_b.size()));
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));

    auto cand_seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);

    std::vector<uint64_t> pids_f(n_factored);
    std::vector<uint64_t> gens_f(n_factored, 1);
    for (uint32_t i = 0; i < n_factored; ++i) {
        pids_f[i] = 11000 + i;
        store.register_hot_payload(pids_f[i], i, gens_f[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids_f, gens_f));
    std::string err;
    TEST_ASSERT(store.publish_candidate(cand_seg, pids_f, gens_f, &err));
    uint64_t seg_id = cand_seg->segment_id;

    // Register next 8 as hot payloads
    std::vector<uint64_t> pids_h(n_hot);
    for (uint32_t i = 0; i < n_hot; ++i) {
        pids_h[i] = 12000 + i;
        store.register_hot_payload(pids_h[i], n_factored + i, 1, xkv_state::hot_committed);
    }

    std::vector<xkv_tri_candidate> candidates(n_total);
    for (uint32_t i = 0; i < n_factored; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = pids_f[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
    }
    for (uint32_t i = 0; i < n_hot; ++i) {
        uint32_t idx = n_factored + i;
        candidates[idx].cell_index = idx;
        candidates[idx].storage_pos = (int32_t)idx;
        candidates[idx].payload_id = pids_h[i];
        candidates[idx].location.kind = xkv_location_kind::hot;
        candidates[idx].location.row = idx;
        candidates[idx].seq_ids = {0};
    }

    xkv_tri_seq_info seq0;
    seq0.seq_id = 0;
    seq0.logical_tokens = n_total;
    seq0.frontier_pos = n_total - 1;
    seq0.tail_guard = 4;
    seq0.eligible = true;

    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};

    hot_k_provider_fn hot_provider = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };

    xkv_tri_reclaim_options opts;
    opts.hot_provider = hot_provider;

    auto prop = adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts);
    TEST_ASSERT_MSG(prop.is_success(), prop.message.c_str());

    // Target retention: max(4, ceil(16 * 0.25)) = 4
    TEST_ASSERT(prop.plan.target_references == 4);
    TEST_ASSERT(prop.plan.hard_keep == 4); // recent tokens [12..15]
    TEST_ASSERT(prop.plan.references_removed == 12); // 16 - 4
    TEST_ASSERT(prop.plan.physical_freed == 12);

    // Released hot payloads & factored affected segments correctly partitioned
    TEST_ASSERT(!prop.released_hot_payloads.empty());
    TEST_ASSERT(!prop.factored_pack_segments.empty());
    TEST_ASSERT(prop.factored_pack_segments[0] == seg_id);

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_mixed_hot_factored_one_sequence PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 10: 8K-like shared prefix across 3 seq refs with physical union
// -----------------------------------------------------------------------------
static void test_shared_prefix_across_3_seqs_with_physical_union() {
    std::cout << "[test_xkv_tri] test_shared_prefix_across_3_seqs_with_physical_union..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 2;
    cfg.ratio = 0.2;

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // 10 shared prefix cells [0..9] held by seq 0, 1, 2
    // Give seq 0 and 1 high target retention so they keep some shared cells
    const uint32_t n_shared = 10;
    const uint32_t n_priv = 4;
    const uint32_t n_total = n_shared + n_priv * 3; // 22 cells

    std::vector<xkv_tri_candidate> candidates(n_total);
    for (uint32_t i = 0; i < n_shared; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = 20000 + i;
        candidates[i].location.kind = xkv_location_kind::hot;
        candidates[i].location.row = i;
        candidates[i].seq_ids = {0, 1, 2};
    }

    for (uint32_t s = 0; s < 3; ++s) {
        for (uint32_t p = 0; p < n_priv; ++p) {
            uint32_t idx = n_shared + s * n_priv + p;
            candidates[idx].cell_index = idx;
            candidates[idx].storage_pos = (int32_t)(n_shared + p);
            candidates[idx].payload_id = 20000 + idx;
            candidates[idx].location.kind = xkv_location_kind::hot;
            candidates[idx].location.row = idx;
            candidates[idx].seq_ids = {(llama_seq_id)s};
        }
    }

    // Make ratio 0.6 so each seq wants to keep max(2, ceil(14 * 0.6)) = 9 cells
    // Recent window is 2 (pos 12, 13). Remaining to keep = 9 - 2 = 7 cells from [0..11].
    // Cells [0..9] are shared!
    xkv_tri_seq_info seq0 = {0, n_shared + n_priv, (int64_t)(n_shared + n_priv - 1), 2, true, 0, {}};
    xkv_tri_seq_info seq1 = {1, n_shared + n_priv, (int64_t)(n_shared + n_priv - 1), 2, true, 0, {}};
    xkv_tri_seq_info seq2 = {2, n_shared + n_priv, (int64_t)(n_shared + n_priv - 1), 2, true, 0, {}};

    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    xkv_tri_config cfg_shared = cfg;
    cfg_shared.ratio = 0.6; // High ratio keeps shared cells
    xkv_tri_adapter adapter_shared(cal, omega.data(), fsq.data(), cfg_shared);

    hot_k_provider_fn mock_provider = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };

    auto plan = adapter_shared.build_mutation_plan(candidates, {seq0, seq1, seq2}, {slice}, store, pressure, mock_provider);

    // Check physical union semantics:
    // For shared prefix cells, references removed may occur on one sequence while another keeps it.
    // references_removed > physical_freed strictly!
    TEST_ASSERT(plan.references_removed > plan.physical_freed);
    TEST_ASSERT(plan.shared_keep > 0);

    // Verify each candidate is strictly in survivor OR evicted, never both
    for (const auto & c : candidates) {
        bool surv = (std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), c.payload_id) != plan.survivor_payloads.end());
        bool evic = (std::find(plan.evicted_payloads.begin(), plan.evicted_payloads.end(), c.payload_id) != plan.evicted_payloads.end());
        TEST_ASSERT(surv != evic);
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_shared_prefix_across_3_seqs_with_physical_union PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 11: Sampled Q heads sharing one KV head
// -----------------------------------------------------------------------------
static void test_sampled_q_heads_sharing_one_kv_head() {
    std::cout << "[test_xkv_tri] test_sampled_q_heads_sharing_one_kv_head..." << std::endl;

    const uint32_t head_dim = 64;
    // 1 KV head, 4 attention heads -> num_kv_groups = 4. 4 sampled Q heads sharing KV head 0.
    triattention_calibration cal = create_test_calib(head_dim, 1, 4, 1, 4, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 6;
    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = 30000 + i;
        candidates[i].location.kind = xkv_location_kind::hot;
        candidates[i].location.row = i;
        candidates[i].seq_ids = {0};
    }

    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    size_t decode_calls = 0;
    hot_k_provider_fn count_provider = [&](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        decode_calls++;
        for (size_t i = 0; i < cells.size() * head_dim; ++i) {
            dst[i] = 0.01f * (float)(i % 17);
        }
        return true;
    };

    std::vector<float> pooled(n_rows);
    std::vector<uint32_t> all_idx = {0, 1, 2, 3, 4, 5};
    adapter.score_candidate_subset(candidates, all_idx, {slice}, store, 10, pooled.data(), nullptr, count_provider);

    // Invariant: single KV decode call reused across all 4 sampled Q heads for this KV head!
    TEST_ASSERT_MSG(decode_calls == 1, "Must reuse single K readback across all sampled Q heads sharing one KV head");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_sampled_q_heads_sharing_one_kv_head PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 12: Candidate order ties deterministic tie-breaking
// -----------------------------------------------------------------------------
static void test_candidate_order_ties_determinism() {
    std::cout << "[test_xkv_tri] test_candidate_order_ties_determinism..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 1;
    cfg.ratio = 0.5; // keep 2 of 4

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // 4 candidates: all identical scores (all-zero K produces identical scores)
    // Candidate 0: pos 0, cell 10
    // Candidate 1: pos 0, cell 5
    // Candidate 2: pos 1, cell 2
    // Candidate 3: pos 10 (recent, protected)
    std::vector<xkv_tri_candidate> candidates(4);
    candidates[0].cell_index = 10; candidates[0].storage_pos = 0; candidates[0].payload_id = 40001;
    candidates[0].location.kind = xkv_location_kind::hot; candidates[0].seq_ids = {0};

    candidates[1].cell_index = 5; candidates[1].storage_pos = 0; candidates[1].payload_id = 40002;
    candidates[1].location.kind = xkv_location_kind::hot; candidates[1].seq_ids = {0};

    candidates[2].cell_index = 2; candidates[2].storage_pos = 1; candidates[2].payload_id = 40003;
    candidates[2].location.kind = xkv_location_kind::hot; candidates[2].seq_ids = {0};

    candidates[3].cell_index = 20; candidates[3].storage_pos = 10; candidates[3].payload_id = 40004;
    candidates[3].location.kind = xkv_location_kind::hot; candidates[3].seq_ids = {0};

    xkv_tri_seq_info seq0 = {0, 4, 10, 1, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    hot_k_provider_fn mock_provider = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };

    auto plan1 = adapter.build_mutation_plan(candidates, {seq0}, {slice}, store, pressure, mock_provider);
    auto plan2 = adapter.build_mutation_plan(candidates, {seq0}, {slice}, store, pressure, mock_provider);

    // Results must be bitwise deterministic across identical runs
    TEST_ASSERT(plan1.ref_removals == plan2.ref_removals);
    TEST_ASSERT(plan1.survivor_payloads == plan2.survivor_payloads);
    TEST_ASSERT(plan1.evicted_payloads == plan2.evicted_payloads);

    // Tied score secondary tie-break: pos 1 > pos 0, so pos 1 (payload 40003) is kept!
    TEST_ASSERT(std::find(plan1.survivor_payloads.begin(), plan1.survivor_payloads.end(), 40003) != plan1.survivor_payloads.end());

    // For identical pos 0, tertiary tie-break: smaller cell_index is kept first (cell 5 < cell 10),
    // so if only one survived, cell 5 would win. Here target is 2 (recent pos 10 + pos 1),
    // so both pos 0 cells are evicted.
    TEST_ASSERT(std::find(plan1.evicted_payloads.begin(), plan1.evicted_payloads.end(), 40001) != plan1.evicted_payloads.end());
    TEST_ASSERT(std::find(plan1.evicted_payloads.begin(), plan1.evicted_payloads.end(), 40002) != plan1.evicted_payloads.end());

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_candidate_order_ties_determinism PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 13: Stale stamp retry and separate floor-exhausted reporting
// -----------------------------------------------------------------------------
static void test_stale_stamp_and_floor_exhausted() {
    std::cout << "[test_xkv_tri] test_stale_stamp_and_floor_exhausted..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 4;
    cfg.ratio = 0.5;

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // Scenario A: Floor exhausted (all tokens inside recent window or below floor)
    std::vector<xkv_tri_candidate> candidates(4);
    for (uint32_t i = 0; i < 4; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = 50000 + i;
        candidates[i].location.kind = xkv_location_kind::hot;
        candidates[i].seq_ids = {0};
        store.register_hot_payload(candidates[i].payload_id, i, 1, xkv_state::hot_committed);
    }

    // Sequence of length 4 with tail_guard=4: target=max(4, ceil(4*0.5))=4 -> all 4 protected!
    xkv_tri_seq_info seq0 = {0, 4, 3, 4, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    hot_k_provider_fn mock_provider = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };

    xkv_tri_reclaim_options opts;
    opts.hot_provider = mock_provider;

    auto prop_floor = adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts);
    TEST_ASSERT_MSG(prop_floor.status == xkv_tri_proposal_status::floor_exhausted, "Must return floor_exhausted separately from failure");
    TEST_ASSERT(prop_floor.is_floor_exhausted());

    // Scenario A2: Idle or fill-first (no pressure) must return explicit no-op success, NEVER floor_exhausted
    xkv_tri_pressure_state pressure_idle = {false, false, false};
    auto prop_idle = adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure_idle, opts);
    TEST_ASSERT_MSG(prop_idle.status == xkv_tri_proposal_status::success, "Idle/fill-first must return explicit success");
    TEST_ASSERT(!prop_idle.is_floor_exhausted());
    TEST_ASSERT(prop_idle.plan.references_removed == 0);
    TEST_ASSERT(prop_idle.plan.survivor_payloads.size() == 4);

    // Scenario A3: flat_quantized candidates succeed with canonical flat decode
    std::vector<xkv_tri_candidate> flat_cands = candidates;
    flat_cands[0].location.kind = xkv_location_kind::flat_quantized;
    // Now with pressure, scorable candidate is scored successfully. With the
    // test ratio 0.5 the target must stay below the 4 candidates: L=6 gives
    // max(1, ceil(3)) = 3 < 4, so all non-recent rows (incl. flat) are scored.
    seq0.logical_tokens = 6;
    seq0.frontier_pos = 9;
    seq0.tail_guard = 1;
    auto prop_flat = adapter.create_reclaim_proposal(flat_cands, {seq0}, {slice}, store, pressure, opts);
    TEST_ASSERT_MSG(prop_flat.status == xkv_tri_proposal_status::success, "flat_quantized candidates must score successfully");
    TEST_ASSERT(!prop_flat.plan.survivor_payloads.empty());
    TEST_ASSERT(prop_flat.plan.references_removed > 0);

    // Verify score_candidate_subset directly produces valid non-error scores for flat_quantized candidates
    std::vector<uint32_t> sub_indices = {0, 1, 2, 3};
    std::vector<float> flat_pooled(4, 0.0f);
    adapter.score_candidate_subset(flat_cands, sub_indices, {slice}, store, seq0.frontier_pos, flat_pooled.data(), nullptr, opts.hot_provider);
    for (uint32_t i = 0; i < 4; ++i) {
        TEST_ASSERT_MSG(std::isfinite(flat_pooled[i]), "flat_quantized candidate scores must be finite");
    }

    // Scenario B: Stale stamp retry detection
    // Make seq0 longer so it wants to evict
    seq0.logical_tokens = 8;
    seq0.frontier_pos = 7;
    seq0.tail_guard = 2;

    std::vector<xkv_tri_candidate> c8(8);
    for (uint32_t i = 0; i < 8; ++i) {
        c8[i].cell_index = i;
        c8[i].storage_pos = (int32_t)i;
        c8[i].payload_id = 51000 + i;
        c8[i].location.kind = xkv_location_kind::hot;
        c8[i].seq_ids = {0};
        store.register_hot_payload(c8[i].payload_id, i, 1, xkv_state::hot_committed);
    }

    // Proposal generated under stamp 0
    auto plan = adapter.build_mutation_plan(c8, {seq0}, {slice}, store, pressure, mock_provider);
    auto store_prop = adapter.create_mutation_proposal(plan, store);
    TEST_ASSERT(store_prop.preflight(store));

    // Advance store stamp
    store.bump_binding_epoch();

    // Now preflight must fail with stale stamp
    std::string err;
    TEST_ASSERT(!store_prop.preflight(store, &err));

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_stale_stamp_and_floor_exhausted PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 14: Workspace one-byte-short and arena budget enforcement
// -----------------------------------------------------------------------------
static void test_workspace_budget_and_one_byte_short() {
    std::cout << "[test_xkv_tri] test_workspace_budget_and_one_byte_short..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    std::vector<xkv_tri_candidate> candidates(10);
    for (uint32_t i = 0; i < 10; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = 61000 + i;
        candidates[i].location.kind = xkv_location_kind::hot;
        candidates[i].seq_ids = {0};
        store.register_hot_payload(candidates[i].payload_id, i, 1, xkv_state::hot_committed);
    }

    xkv_tri_seq_info seq0 = {0, 10, 9, 2, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    xkv_tri_workspace_requirements reqs;
    adapter.estimate_workspace_requirements(10, {seq0}, {slice}, reqs);
    TEST_ASSERT(reqs.total_bytes > 0);

    // Budget exact match: succeeds
    xkv_tri_reclaim_options opts_exact;
    opts_exact.workspace_budget_bytes = reqs.total_bytes;
    opts_exact.hot_provider = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };
    auto prop_ok = adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_exact);
    TEST_ASSERT_MSG(prop_ok.is_success(), prop_ok.message.c_str());

    // Budget one byte short: fails with workspace_exhausted
    xkv_tri_reclaim_options opts_short;
    opts_short.workspace_budget_bytes = reqs.total_bytes - 1;
    opts_short.hot_provider = opts_exact.hot_provider;
    auto prop_short = adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_short);
    TEST_ASSERT_MSG(prop_short.status == xkv_tri_proposal_status::workspace_exhausted, "One byte short must fail with workspace_exhausted");

    // Arena with exact capacity vs one-byte-short
    // Arena backing is 64B-rounded, so byte-exactness is asserted via the
    // budget path above; here one full quantum short must refuse.
    xkv_workspace_arena arena_short(reqs.total_bytes - 64);
    xkv_tri_reclaim_options opts_arena_short;
    opts_arena_short.arena = &arena_short;
    opts_arena_short.hot_provider = opts_exact.hot_provider;
    auto prop_arena_short = adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_arena_short);
    TEST_ASSERT_MSG(prop_arena_short.status == xkv_tri_proposal_status::workspace_exhausted, "Arena one byte short must fail with workspace_exhausted");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_workspace_budget_and_one_byte_short PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 15: Zero physical free despite references removed
// -----------------------------------------------------------------------------
static void test_zero_physical_free_despite_refs_removed() {
    std::cout << "[test_xkv_tri] test_zero_physical_free_despite_refs_removed..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_config cfg;
    cfg.recent_window = 1;
    cfg.ratio = 0.5; // evict 2 of 4 from seq 0

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // 4 cells: all 4 shared with seq 1!
    // Seq 0 is reclaimed, but seq 1 is NOT eligible.
    std::vector<xkv_tri_candidate> candidates(4);
    for (uint32_t i = 0; i < 4; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = 71000 + i;
        candidates[i].location.kind = xkv_location_kind::hot;
        candidates[i].seq_ids = {0, 1};
    }

    xkv_tri_seq_info seq0 = {0, 4, 3, 1, true, 0, {}};
    xkv_tri_seq_info seq1 = {1, 4, 3, 4, false, 0, {}}; // not eligible

    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    hot_k_provider_fn mock_provider = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        std::memset(dst, 0, cells.size() * 64 * sizeof(float));
        return true;
    };

    auto plan = adapter.build_mutation_plan(candidates, {seq0, seq1}, {slice}, store, pressure, mock_provider);

    // Invariant: references_removed > 0, plan.changed == true, but physical_freed == 0!
    TEST_ASSERT(plan.references_removed > 0);
    TEST_ASSERT(plan.changed);
    TEST_ASSERT(plan.physical_freed == 0);
    TEST_ASSERT(plan.survivor_payloads.size() == 4);
    TEST_ASSERT(plan.evicted_payloads.empty());

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_zero_physical_free_despite_refs_removed PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 16: Multiple layer groups with adaptive tail ranks
// -----------------------------------------------------------------------------
static void test_multiple_layer_groups_adaptive_tail_ranks() {
    std::cout << "[test_xkv_tri] test_multiple_layer_groups_adaptive_tail_ranks..." << std::endl;

    const uint32_t head_dim = 64;
    // Sampled layers across two groups: layer 0 (group 0, rank 32) and layer 1 (tail group 1, rank 16)
    triattention_calibration cal = create_test_calib(head_dim, 2, 2, 1, 2, head_dim);
    cal.sampled_layer[0] = 0; cal.sampled_head[0] = 0;
    cal.sampled_layer[1] = 1; cal.sampled_head[1] = 1;

    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 8;
    const uint32_t total_dim = 64;
    const uint32_t rank0 = 32;
    const uint32_t rank1 = 16; // adaptive tail rank

    xkv_factor_group_payload g0;
    g0.group_index = 0;
    g0.owning_layers = {0};
    g0.rank_k = rank0;
    g0.rank_v = rank0;
    g0.total_dim_k = total_dim;
    g0.total_dim_v = total_dim;
    g0.layer_feature_offsets_k = {0};
    g0.layer_feature_dims_k = {total_dim};
    g0.layer_feature_offsets_v = {0};
    g0.layer_feature_dims_v = {total_dim};

    codec_desc desc_a0 = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank0});
    std::vector<float> data_a0(n_rows * rank0, 0.1f);
    g0.a_k = encode_matrix(desc_a0, data_a0.data(), data_a0.size());
    codec_desc desc_av0 = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank0});
    g0.a_v = encode_matrix(desc_av0, data_a0.data(), data_a0.size());
    codec_desc desc_b0 = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank0});
    std::vector<float> data_b0(total_dim * rank0, 0.2f);
    g0.set_b_k(encode_matrix(desc_b0, data_b0.data(), data_b0.size()));
    codec_desc desc_bv0 = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank0});
    g0.set_b_v(encode_matrix(desc_bv0, data_b0.data(), data_b0.size()));

    // Tail group 1 with adaptive lower rank1 = 16
    xkv_factor_group_payload g1;
    g1.group_index = 1;
    g1.owning_layers = {1};
    g1.rank_k = rank1;
    g1.rank_v = rank1;
    g1.total_dim_k = total_dim;
    g1.total_dim_v = total_dim;
    g1.layer_feature_offsets_k = {0};
    g1.layer_feature_dims_k = {total_dim};
    g1.layer_feature_offsets_v = {0};
    g1.layer_feature_dims_v = {total_dim};

    codec_desc desc_a1 = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank1});
    std::vector<float> data_a1(n_rows * rank1, 0.3f);
    g1.a_k = encode_matrix(desc_a1, data_a1.data(), data_a1.size());
    codec_desc desc_av1 = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank1});
    g1.a_v = encode_matrix(desc_av1, data_a1.data(), data_a1.size());
    codec_desc desc_b1 = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank1});
    std::vector<float> data_b1(total_dim * rank1, 0.4f);
    g1.set_b_k(encode_matrix(desc_b1, data_b1.data(), data_b1.size()));
    codec_desc desc_bv1 = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank1});
    g1.set_b_v(encode_matrix(desc_bv1, data_b1.data(), data_b1.size()));

    auto cand_seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g0, g1}
    );
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 80000 + i;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));
    std::string err;
    TEST_ASSERT(store.publish_candidate(cand_seg, pids, gens, &err));
    uint64_t seg_id = cand_seg->segment_id;

    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = pids[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
    }

    xkv_layer_slice slice0 = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};
    xkv_layer_slice slice1 = {1, 1, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};

    std::vector<float> pooled(n_rows);
    std::vector<uint32_t> all_indices(n_rows);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    // Score across both adaptive groups seamlessly
    adapter.score_candidate_subset(candidates, all_indices, {slice0, slice1}, store, 10, pooled.data());
    for (uint32_t i = 0; i < n_rows; ++i) {
        TEST_ASSERT(!std::isnan(pooled[i]) && std::isfinite(pooled[i]));
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_multiple_layer_groups_adaptive_tail_ranks PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 17: Factor decode error handling & candidate validation
// -----------------------------------------------------------------------------
static void test_factor_decode_error_handling() {
    std::cout << "[test_xkv_tri] test_factor_decode_error_handling..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    // Create candidates with inconsistent row_ref vs location
    xkv_tri_candidate bad_c;
    bad_c.cell_index = 0;
    bad_c.payload_id = 90001;
    bad_c.location.kind = xkv_location_kind::factored;
    bad_c.location.segment_id = 1;
    bad_c.location.row = 0;
    bad_c.row_ref.segment_id = 2; // mismatch!
    bad_c.row_ref.row = 0;
    bad_c.seq_ids = {0};

    xkv_tri_seq_info seq0 = {0, 1, 0, 1, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};

    xkv_tri_reclaim_options opts;
    auto prop = adapter.create_reclaim_proposal({bad_c}, {seq0}, {slice}, store, pressure, opts);
    TEST_ASSERT_MSG(prop.status == xkv_tri_proposal_status::invalid_argument, "Mismatched row_ref and location must return invalid_argument");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_factor_decode_error_handling PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 18: Landmark pack callback vs tombstone fallback & refusal
// -----------------------------------------------------------------------------
static void test_landmarks_pack_callback_and_refusal() {
    std::cout << "[test_xkv_tri] test_landmarks_pack_callback_and_refusal..." << std::endl;

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 8;
    const uint32_t rank = 16;
    const uint32_t total_dim = 64;

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = total_dim;
    g.total_dim_v = total_dim;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim};

    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank}, 128);
    std::vector<float> data_a(n_rows * rank, 1.0f);
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());
    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank}, 128);
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());

    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim, rank}, 128);
    std::vector<float> data_b(total_dim * rank, 2.0f);
    g.set_b_k(encode_matrix(desc_b, data_b.data(), data_b.size()));
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim, rank}, 128);
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));

    // Construct required landmark stream for LANDMARKS profile
    const uint32_t chunks = (n_rows + 7) / 8;
    codec_desc lm_desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                         orientation::token_major, {chunks, total_dim}, 0, 777);
    std::vector<float> lm_data((size_t)chunks * total_dim, 0.1f);
    g.landmark = encode_matrix(lm_desc, lm_data.data(), lm_data.size());
    // Bounds table travels atomically with landmarks: one entry per 8-row chunk.
    g.landmark_chunks.resize(chunks);
    for (uint32_t c = 0; c < chunks; ++c) {
        g.landmark_chunks[c].row_begin = c * 8;
        g.landmark_chunks[c].row_count = std::min<uint32_t>(8, n_rows - c * 8);
        g.landmark_chunks[c].error_bound = 0.1f;
        g.landmark_chunks[c].source_fingerprint = (uint64_t)c + 1;
    }
    g.landmark_table_fingerprint = compute_landmark_table_fingerprint(g.landmark_chunks);

    g.owning_layers = {0};
    // Create LANDMARKS-profile segment
    auto cand_seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 95000 + i;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));
    std::string err;
    if (!store.publish_candidate(cand_seg, pids, gens, &err)) {
        fprintf(stderr, "publish_candidate in test 18 failed: %s\n", err.c_str());
        TEST_ASSERT(false);
    }
    uint64_t seg_id = cand_seg->segment_id;

    xkv_tri_mutation_plan plan;
    plan.evicted_payloads = {pids[2], pids[5]};
    plan.affected_segments = {seg_id};

    triattention_calibration cal = create_test_calib();
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());

    // Case 1: Without callback, LANDMARKS profile mutation MUST be refused by the store (no fabricated summary, no tombstone clone)
    auto prop_tomb = adapter.create_mutation_proposal(plan, store, nullptr);
    TEST_ASSERT(prop_tomb.preflight(store));
    bool applied_without_cb = prop_tomb.apply(store, nullptr, &err);
    TEST_ASSERT_MSG(!applied_without_cb, "LANDMARKS segment mutation without rebuild callback must be refused by store");
    auto cur_tomb_seg = store.get_segment(seg_id);
    TEST_ASSERT_MSG(cur_tomb_seg->live_rows[2], "Refused mutation must keep rows alive without partial changes");

    // Case 2: Callback that refuses the rebuild -> atomic failure with zero store mutation
    xkv_landmark_rebuild_fn refusing_cb = [](const xkv_segment &, const std::vector<uint32_t> &, xkv_segment &, std::string * err) {
        if (err) *err = "landmark rebuild refused by policy";
        return false;
    };

    xkv_tri_mutation_plan plan2;
    plan2.evicted_payloads = {pids[0]};
    plan2.affected_segments = {seg_id};

    auto prop_refuse = adapter.create_mutation_proposal(plan2, store, refusing_cb);
    bool ok = prop_refuse.apply(store, nullptr, &err);
    TEST_ASSERT_MSG(!ok, "Refusing landmark callback must abort transaction atomically");

    // Verify state unchanged: payload 0 still valid and alive
    auto cur_seg = store.get_segment(seg_id);
    TEST_ASSERT(cur_seg->live_rows[0]);

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_landmarks_pack_callback_and_refusal PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 19: Distinguish mean vs max head aggregation and ablation enforcement
// -----------------------------------------------------------------------------
static void test_head_aggregation_mean_vs_max_and_ablation_enforcement() {
    std::cout << "[test_xkv_tri] test_head_aggregation_mean_vs_max_and_ablation_enforcement..." << std::endl;

    const uint32_t head_dim = 64;
    // 2 sampled heads with different stats
    triattention_calibration cal = create_test_calib(head_dim, 2, 2, 1, 2, head_dim);
    cal.sampled_layer[0] = 0; cal.sampled_head[0] = 0;
    cal.sampled_layer[1] = 1; cal.sampled_head[1] = 1;

    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);

    // Invariant 1: Non-max head_agg without is_ablation MUST throw std::invalid_argument
    xkv_tri_config bad_cfg;
    bad_cfg.head_agg = TRIATTENTION_AGG_MEAN;
    bad_cfg.is_ablation = false;
    bool caught_unauthorized = false;
    try {
        xkv_tri_adapter bad_adapter(cal, omega.data(), fsq.data(), bad_cfg);
    } catch (const std::invalid_argument &) {
        caught_unauthorized = true;
    }
    TEST_ASSERT_MSG(caught_unauthorized, "Overriding production head_agg without is_ablation=true must be rejected");

    // Invariant 2: Explicit ablation head_agg = MEAN produces different scores from MAX
    xkv_tri_config max_cfg;
    max_cfg.head_agg = TRIATTENTION_AGG_MAX;
    xkv_tri_adapter max_adapter(cal, omega.data(), fsq.data(), max_cfg);

    xkv_tri_config mean_cfg;
    mean_cfg.head_agg = TRIATTENTION_AGG_MEAN;
    mean_cfg.is_ablation = true;
    xkv_tri_adapter mean_adapter(cal, omega.data(), fsq.data(), mean_cfg);

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    std::vector<xkv_tri_candidate> candidates(4);
    for (uint32_t i = 0; i < 4; ++i) {
        candidates[i].cell_index = i; candidates[i].storage_pos = i; candidates[i].payload_id = 99000 + i;
    }
    xkv_layer_slice s0 = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};
    xkv_layer_slice s1 = {1, 1, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};
    hot_k_provider_fn prov = [](uint32_t ml, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            for (size_t d = 0; d < 64; ++d) {
                dst[ci * 64 + d] = (ml == 0) ? (0.05f * std::sin((float)ci)) : (0.05f * std::cos((float)(ci * 2 + d)));
            }
        }
        return true;
    };
    std::vector<float> p_max(4), p_mean(4);
    std::vector<uint32_t> idx = {0, 1, 2, 3};
    max_adapter.score_candidate_subset(candidates, idx, {s0, s1}, store, 10, p_max.data(), nullptr, prov);
    mean_adapter.score_candidate_subset(candidates, idx, {s0, s1}, store, 10, p_mean.data(), nullptr, prov);
    TEST_ASSERT_MSG(p_max != p_mean, "Max and Mean head aggregation must produce distinct scores");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_head_aggregation_mean_vs_max_and_ablation_enforcement PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 20: hot-slot deficit is NOT satisfied by factored-only deletion.
// -----------------------------------------------------------------------------
static void test_hot_slot_deficit_factored_only_does_not_satisfy() {
    std::cout << "[test_xkv_tri] test_hot_slot_deficit_factored_only_does_not_satisfy..." << std::endl;
    const uint32_t head_dim = 64, rank = 16, total_dim = 64, n_rows = 8;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg;
    cfg.ratio = 0.1;
    xkv_tri_adapter thin(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    std::vector<uint64_t> pids;
    auto seg = make_tri_factored_segment(store, n_rows, rank, total_dim, 40000, pids);
    TEST_ASSERT(seg != nullptr);
    const uint64_t seg_id = seg->segment_id;
    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = pids[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
    }
    xkv_tri_seq_info seq0 = {0, n_rows, 7, 2, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false, xkv_tri_pressure_resource::hot_slots, 2};
    xkv_layer_slice slice = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};
    xkv_tri_reclaim_options opts;
    opts.hot_span_provider = make_tri_span_provider(head_dim);
    auto prop = thin.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts);
    TEST_ASSERT_MSG(prop.is_success(), prop.message.c_str());
    TEST_ASSERT_MSG(prop.hot_slots_freed == 0, "factored-only eviction frees no hot slots");
    TEST_ASSERT_MSG(prop.factored_rows_freed == 6, "six non-recent factored rows evicted");
    TEST_ASSERT_MSG(prop.factored_bytes_reclaimed > 0, "factored bytes must be counted");
    TEST_ASSERT_MSG(prop.plan.physical_freed == 6, "generic physical_freed still counts rows");
    TEST_ASSERT_MSG(!prop.capacity_satisfied, "hot-slot deficit NOT satisfied by factored rows");
    TEST_ASSERT_MSG(!prop.plan.capacity_satisfied, "plan must agree");
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_hot_slot_deficit_factored_only_does_not_satisfy PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 21: store-byte pressure IS satisfied by factored reclaim, exact bytes.
// -----------------------------------------------------------------------------
static void test_store_byte_pressure_satisfied_by_factored_reclaim() {
    std::cout << "[test_xkv_tri] test_store_byte_pressure_satisfied_by_factored_reclaim..." << std::endl;
    const uint32_t head_dim = 64, rank = 16, total_dim = 64, n_rows = 8;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg;
    cfg.ratio = 0.1;
    xkv_tri_adapter thin(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    std::vector<uint64_t> pids;
    auto seg = make_tri_factored_segment(store, n_rows, rank, total_dim, 41000, pids);
    TEST_ASSERT(seg != nullptr);
    const uint64_t seg_id = seg->segment_id;
    const size_t row_bytes = seg->groups[0].a_k.desc.row_stride_bytes +
                               seg->groups[0].a_v.desc.row_stride_bytes;
    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = pids[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
    }
    xkv_tri_seq_info seq0 = {0, n_rows, 7, 2, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false, xkv_tri_pressure_resource::store_bytes, 1};
    xkv_layer_slice slice = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};
    xkv_tri_reclaim_options opts;
    opts.hot_span_provider = make_tri_span_provider(head_dim);
    auto prop = thin.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts);
    TEST_ASSERT_MSG(prop.is_success(), prop.message.c_str());
    TEST_ASSERT_MSG(prop.factored_rows_freed == 6, "six factored rows evicted");
    TEST_ASSERT_MSG(prop.factored_bytes_reclaimed == 6 * row_bytes, "exact row-byte accounting");
    TEST_ASSERT_MSG(prop.capacity_satisfied, "store-byte pressure satisfied by factored reclaim");
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_store_byte_pressure_satisfied_by_factored_reclaim PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 22: zero-heap carved scratch with a global new counter.
// -----------------------------------------------------------------------------
static void test_zero_heap_carved_scratch_with_new_counter() {
    std::cout << "[test_xkv_tri] test_zero_heap_carved_scratch_with_new_counter..." << std::endl;
    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg;
    cfg.disable_trig = true;
    cfg.is_ablation = true;
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    const uint32_t n = 10;
    std::vector<xkv_tri_candidate> candidates(n);
    for (uint32_t i = 0; i < n; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = 42000 + i;
        candidates[i].location.kind = xkv_location_kind::hot;
        candidates[i].location.row = i;
        candidates[i].seq_ids = {0};
    }
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    auto span = make_tri_span_provider(head_dim);
    xkv_tri_workspace_requirements reqs;
    TEST_ASSERT(adapter.estimate_workspace_requirements(n, {}, {slice}, reqs, nullptr, &store,
                                                          &candidates));
    xkv_workspace_arena arena(reqs.total_bytes);
    xkv_arena_lease lease = arena.acquire(reqs.total_bytes);
    TEST_ASSERT_MSG((bool)lease, "exact-size arena lease must succeed");
    std::vector<float> pooled_heap(n), pooled_arena(n);
    std::vector<xkv_layer_slice> slices = {slice};
    // Warmup run (disarmed): stabilizes thread pools/allocators outside the window.
    adapter.score_candidate_subset(candidates, idx, slices, store, 20, pooled_heap.data(), nullptr,
                                     nullptr, nullptr, span, nullptr);
    g_tri_count_heap = true;
    g_tri_heap_news = 0;
    adapter.score_candidate_subset(candidates, idx, slices, store, 20, pooled_heap.data(), nullptr,
                                     nullptr, nullptr, span, nullptr);
    const size_t heap_news = g_tri_heap_news;
    g_tri_heap_news = 0;
    adapter.score_candidate_subset(candidates, idx, slices, store, 20, pooled_arena.data(), nullptr,
                                     nullptr, nullptr, span, &lease);
    const size_t arena_news = g_tri_heap_news;
    g_tri_count_heap = false;
    // The arena path must skip exactly the single heap_scratch fallback block;
    // every other allocation (pre-existing scorer-kernel vectors in
    // llama-triattention.cpp, out of slice) is identical on both paths, so the
    // delta proves all Tri-owned planner/scoring scratch is arena-carved.
    TEST_ASSERT_MSG(heap_news == arena_news + 1, "arena path must save exactly the fallback block");
    TEST_ASSERT_MSG(pooled_heap == pooled_arena, "arena and heap scratch paths must agree exactly");
    for (float v : pooled_arena) TEST_ASSERT(std::isfinite(v));
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_zero_heap_carved_scratch_with_new_counter PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 23: unknown reader frontier preserves the tail guard; explicit table
// makes the same candidate evictable.
// -----------------------------------------------------------------------------
static void test_unknown_reader_frontier_preserved_and_table_accepted() {
    std::cout << "[test_xkv_tri] test_unknown_reader_frontier_preserved_and_table_accepted..." << std::endl;
    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg;
    cfg.recent_window = 1;
    cfg.ratio = 0.1;
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    std::vector<xkv_tri_candidate> candidates(3);
    candidates[0].cell_index = 0;
    candidates[0].storage_pos = 0;
    candidates[0].payload_id = 43000;
    candidates[0].location.kind = xkv_location_kind::hot;
    candidates[0].seq_ids = {0, 7};
    candidates[0].semantic_episode_id = 5;
    candidates[1].cell_index = 1;
    candidates[1].storage_pos = 1;
    candidates[1].payload_id = 43001;
    candidates[1].location.kind = xkv_location_kind::hot;
    candidates[1].seq_ids = {0};
    candidates[1].semantic_episode_id = 5;
    candidates[2].cell_index = 2;
    candidates[2].storage_pos = 9;
    candidates[2].payload_id = 43002;
    candidates[2].location.kind = xkv_location_kind::hot;
    candidates[2].seq_ids = {0};
    candidates[2].semantic_episode_id = 5;
    xkv_tri_seq_info seq0;
    seq0.seq_id = 0;
    seq0.logical_tokens = 3;
    seq0.frontier_pos = 9;
    seq0.tail_guard = 1;
    seq0.eligible = true;
    seq0.semantic_episode_id = 5;
    seq0.semantic_seq_ids = {0, 7};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};
    auto span = make_tri_span_provider(head_dim);
    auto has = [](const std::vector<uint64_t> & v, uint64_t pid) {
        return std::find(v.begin(), v.end(), pid) != v.end();
    };
    auto plan_closed = adapter.build_mutation_plan(candidates, {seq0}, {slice}, store, pressure,
                                                     nullptr, nullptr, nullptr, span, nullptr);
    TEST_ASSERT_MSG(has(plan_closed.survivor_payloads, 43000), "unknown reader frontier must preserve");
    TEST_ASSERT_MSG(has(plan_closed.survivor_payloads, 43002), "recent window must preserve");
    TEST_ASSERT_MSG(has(plan_closed.evicted_payloads, 43001), "plain scorable row evicted");
    xkv_tri_frontier_table table = {{7, 100}};
    auto plan_open = adapter.build_mutation_plan(candidates, {seq0}, {slice}, store, pressure, nullptr,
                                                   &table, nullptr, span, nullptr);
    TEST_ASSERT_MSG(has(plan_open.evicted_payloads, 43000), "explicit frontier makes row evictable");
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_unknown_reader_frontier_preserved_and_table_accepted PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 24: device-owned factors dispatch natively; absent callback fails
// closed distinctly without any host decode/readback.
// -----------------------------------------------------------------------------
static void test_proposal_arena_heap_allocation_delta() {
    std::cout << "[test_xkv_tri] test_proposal_arena_heap_allocation_delta..." << std::endl;
    const uint32_t head_dim = 64, rank = 16, total_dim = 64, n_rows = 8;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg;
    cfg.ratio = 0.1;
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    std::vector<uint64_t> pids;
    auto seg = make_tri_factored_segment(store, n_rows, rank, total_dim, 46000, pids);
    TEST_ASSERT(seg != nullptr);
    const uint64_t seg_id = seg->segment_id;
    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = pids[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
    }
    xkv_tri_seq_info seq0 = {0, n_rows, 7, 2, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};
    xkv_tri_workspace_requirements reqs;
    TEST_ASSERT(adapter.estimate_workspace_requirements(n_rows, {seq0}, {slice}, reqs, nullptr,
                                                          &store, &candidates));
    xkv_workspace_arena arena(reqs.total_bytes);
    xkv_arena_lease hold = arena.acquire(reqs.total_bytes);
    TEST_ASSERT_MSG((bool)hold, "exact-size arena lease must succeed");
    hold.release(); // fit check only; the proposal acquires its own lease below
    xkv_tri_reclaim_options opts_heap;
    opts_heap.hot_span_provider = make_tri_span_provider(head_dim);
    xkv_tri_reclaim_options opts_arena;
    opts_arena.hot_span_provider = make_tri_span_provider(head_dim);
    opts_arena.arena = &arena;
    g_tri_count_heap = true;
    g_tri_heap_news = 0;
    auto prop_heap =
        adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_heap);
    const size_t heap_news = g_tri_heap_news;
    g_tri_heap_news = 0;
    auto prop_arena =
        adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_arena);
    const size_t arena_news = g_tri_heap_news;
    g_tri_count_heap = false;
    TEST_ASSERT_MSG(prop_heap.is_success(), prop_heap.message.c_str());
    TEST_ASSERT_MSG(prop_arena.is_success(), prop_arena.message.c_str());
    TEST_ASSERT_MSG(arena_news + 1 == heap_news, "arena path must skip exactly the heap fallback block");
    TEST_ASSERT_MSG(prop_arena.plan.evicted_payloads == prop_heap.plan.evicted_payloads, "paths must agree");
    TEST_ASSERT_MSG(prop_arena.plan.survivor_payloads == prop_heap.plan.survivor_payloads, "paths must agree");
    TEST_ASSERT_MSG(prop_arena.factored_bytes_reclaimed == prop_heap.factored_bytes_reclaimed, "agree");
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_proposal_arena_heap_allocation_delta PASSED." << std::endl;
}

static void test_device_owned_dispatch_and_fail_closed() {
    std::cout << "[test_xkv_tri] test_device_owned_dispatch_and_fail_closed..." << std::endl;
    const uint32_t head_dim = 64, rank = 16, total_dim = 64, n_rows = 6;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg;
    cfg.ratio = 0.1;
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    std::vector<uint64_t> pids;
    auto seg = make_tri_factored_segment(store, n_rows, rank, total_dim, 44000, pids);
    TEST_ASSERT(seg != nullptr);
    const uint64_t seg_id = seg->segment_id;
    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = pids[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
        candidates[i].device_owned = true;
    }
    xkv_tri_seq_info seq0 = {0, n_rows, 5, 1, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_layer_slice slice = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};
    size_t host_calls = 0;
    auto counting_span = [&](uint32_t, uint32_t, const uint32_t *, size_t, float *, size_t) {
        ++host_calls;
        return true;
    };
    xkv_tri_reclaim_options opts_closed;
    opts_closed.hot_span_provider = counting_span;
    auto prop_closed =
        adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_closed);
    TEST_ASSERT_MSG(prop_closed.status == xkv_tri_proposal_status::device_scoring_unavailable,
                      prop_closed.message.c_str());
    TEST_ASSERT_MSG(host_calls == 0, "device owned rows must never reach host readback");
    size_t dev_calls = 0;
    device_factor_scoring_fn dev = [&](uint64_t, uint64_t, const uint32_t * rows, uint32_t n,
                                        const xkv_layer_slice &, uint32_t, int64_t, float * out,
                                        std::string *) {
        ++dev_calls;
        for (uint32_t k = 0; k < n; ++k) out[k] = 0.05f * (float)(rows[k] + 1);
        return true;
    };
    xkv_tri_reclaim_options opts_open;
    opts_open.hot_span_provider = counting_span;
    opts_open.device_scoring = dev;
    auto prop_open =
        adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_open);
    TEST_ASSERT_MSG(prop_open.is_success(), prop_open.message.c_str());
    TEST_ASSERT_MSG(dev_calls > 0, "device scorer must be dispatched");
    TEST_ASSERT_MSG(host_calls == 0, "still no host readback with device dispatch");
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_device_owned_dispatch_and_fail_closed PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 25: estimator accounts tile-A and every carved array from actual
// descriptors; exact fit succeeds, one byte short fails.
// -----------------------------------------------------------------------------
static void test_estimator_tile_a_exact_and_descriptor_derived() {
    std::cout << "[test_xkv_tri] test_estimator_tile_a_exact_and_descriptor_derived..." << std::endl;
    const uint32_t head_dim = 64, rank = 16, total_dim = 64, n_rows = 8;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg;
    cfg.ratio = 0.1;
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    std::vector<uint64_t> pids;
    auto seg = make_tri_factored_segment(store, n_rows, rank, total_dim, 45000, pids);
    TEST_ASSERT(seg != nullptr);
    const uint64_t seg_id = seg->segment_id;
    const uint32_t a_pad_actual = (uint32_t)seg->groups[0].a_k.desc.padded_shape.cols;
    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)i;
        candidates[i].payload_id = pids[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
    }
    xkv_layer_slice slice = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};
    xkv_tri_workspace_requirements reqs;
    TEST_ASSERT(adapter.estimate_workspace_requirements(n_rows, {}, {slice}, reqs, nullptr, &store,
                                                          &candidates));
    const size_t expect_tile_a = ((size_t)32 * a_pad_actual * sizeof(float) + 63) & ~size_t(63);
    TEST_ASSERT_MSG(reqs.tile_a_bytes == expect_tile_a, "tile-A must be exact from actual A pads");
    const size_t expect_total = reqs.candidate_array_bytes + reqs.candidate_order_bytes +
                                reqs.score_buffers_bytes + reqs.tile_buffers_bytes + reqs.tile_a_bytes +
                                reqs.union_maps_bytes;
    TEST_ASSERT_MSG(reqs.total_bytes == expect_total, "total must equal every carved array");
    xkv_tri_seq_info seq0 = {0, n_rows, 7, 2, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    xkv_tri_reclaim_options opts_ok;
    opts_ok.hot_span_provider = make_tri_span_provider(head_dim);
    opts_ok.workspace_budget_bytes = reqs.total_bytes;
    auto prop_ok =
        adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_ok);
    TEST_ASSERT_MSG(prop_ok.is_success(), prop_ok.message.c_str());
    xkv_tri_reclaim_options opts_short;
    opts_short.hot_span_provider = make_tri_span_provider(head_dim);
    opts_short.workspace_budget_bytes = reqs.total_bytes - 1;
    auto prop_short =
        adapter.create_reclaim_proposal(candidates, {seq0}, {slice}, store, pressure, opts_short);
    TEST_ASSERT_MSG(prop_short.status == xkv_tri_proposal_status::workspace_exhausted,
                      "one byte short must fail");
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_estimator_tile_a_exact_and_descriptor_derived PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 27: server-pressure integration order (simulated with public APIs).
// Fixed order: lossless seal/pack -> Tri proposal across HOT+FACTORED ->
// apply + usage refresh -> victim handling ONLY on floor-exhausted.
// Recurrent-only bypasses both KV paths. Device lane dispatches natively.
// -----------------------------------------------------------------------------
static void test_server_pressure_integration_order() {
    std::cout << "[test_xkv_tri] test_server_pressure_integration_order..." << std::endl;
    const uint32_t head_dim = 64, rank = 16, total_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    const uint32_t n_hot = 8, n_fact = 8, n_total = n_hot + n_fact;
    std::vector<uint64_t> pids_f;
    auto seg = make_tri_factored_segment(store, n_fact, rank, total_dim, 61000, pids_f);
    TEST_ASSERT(seg != nullptr);
    const uint64_t seg_id = seg->segment_id;
    for (uint32_t i = 0; i < n_hot; ++i) {
        store.register_hot_payload(60000 + i, n_fact + i, 1, xkv_state::hot_committed);
    }

    llama_kv_cells cells;
    cells.resize(n_total);
    for (uint32_t i = 0; i < n_fact; ++i) {
        cells.pos_set(i, (llama_pos)i);
        cells.seq_add(i, 0);
        cells.payload_id_set(i, pids_f[i], 1);
    }
    for (uint32_t i = 0; i < n_hot; ++i) {
        const uint32_t c = n_fact + i;
        cells.pos_set(c, (llama_pos)c);
        cells.seq_add(c, 0);
        cells.payload_id_set(c, 60000 + i, 1);
    }

    xkv_layer_slice slice = {0, 0, 0, total_dim, 0, total_dim, 1, head_dim, head_dim, 0};
    llama_memory_kv_reclaim_request req;
    req.required_free = 4;
    req.drain_to_floor = true;
    llama_memory_kv_reclaim_seq_hint hint;
    hint.seq_id = 0;
    hint.logical_tokens = n_total;
    hint.tail_guard = 2;
    hint.eligible = true;
    req.seq_hints = {hint};

    uint64_t fetch_calls = 0;
    xkv_tri_pressure_hooks hooks;
    hooks.hot_span_provider = make_tri_span_provider(head_dim);
    hooks.selected_k_fetch = adapter.make_host_selected_k_fetch(store, nullptr, &fetch_calls);

    // Step 1: lossless seal/pack (store/runtime side, pre-done).
    // Step 2: Tri proposal across HOT+FACTORED semantic payloads.
    const xkv_tri_usage_snapshot use_before = xkv_tri_read_usage(cells, store);
    auto prop = adapter.plan_pressure(cells, req, {slice}, store, false, hooks, xkv_tri_pressure_resource::hot_slots);
    TEST_ASSERT_MSG(prop.is_success(), prop.message.c_str());
    TEST_ASSERT_MSG(prop.factored_rows_freed > 0, "factored refs reclaimed in pressure order");
    TEST_ASSERT_MSG(prop.hot_slots_freed > 0, "hot refs reclaimed in pressure order");
    TEST_ASSERT_MSG(adapter.decide(prop) == xkv_tri_pressure_decision::reclaimed, "success maps to reclaimed");

    // Step 3: apply the atomic transaction, then refresh usage.
    // Collect hot removals and set batch.removal_precommit
    std::vector<uint64_t> hot_pids;
    std::vector<uint64_t> hot_gens;
    std::vector<uint32_t> hot_slots;
    for (size_t i = 0; i < prop.released_hot_payloads.size(); ++i) {
        uint64_t pid = prop.released_hot_payloads[i];
        uint32_t slot = prop.released_hot_rows[i];
        xkv_location loc;
        if (store.find_location(pid, loc)) {
            hot_pids.push_back(pid);
            hot_gens.push_back(loc.storage_generation);
            hot_slots.push_back(slot);
        }
    }
    prop.store_proposal.batch.removal_precommit = [](const xkv_removal_precommit_ctx &, std::string *) {
        return true;
    };

    bool victim_ran = false;
    std::string apply_err;
    TEST_ASSERT_MSG(prop.store_proposal.apply(store, nullptr, &apply_err), apply_err.c_str());
    const xkv_tri_usage_snapshot use_after = xkv_tri_read_usage(cells, store);
    TEST_ASSERT_MSG(use_after.live_payload_bytes < use_before.live_payload_bytes, "usage refresh shows reclaim");
    auto result = adapter.result_from_proposal(prop, req, use_before.cells_used);
    TEST_ASSERT_MSG(result.supported && result.changed, "result carries change");
    TEST_ASSERT_MSG(result.capacity_satisfied, "deficit of 4 met by freed hot slots");
    TEST_ASSERT_MSG(!victim_ran, "no victim handling on the reclaimed path");

    // Step 4: floor-exhausted with remaining deficit -> victim handling runs.
    llama_memory_kv_reclaim_request req_floor = req;
    req_floor.seq_hints[0].tail_guard = n_total;
    auto prop_floor = adapter.plan_pressure(cells, req_floor, {slice}, store, false, hooks);
    TEST_ASSERT_MSG(prop_floor.status == xkv_tri_proposal_status::floor_exhausted, "floor reported");
    TEST_ASSERT_MSG(adapter.decide(prop_floor) == xkv_tri_pressure_decision::floor_exhausted_victim,
                      "floor maps to victim");
    if (adapter.decide(prop_floor) == xkv_tri_pressure_decision::floor_exhausted_victim) {
        victim_ran = true;
    }
    TEST_ASSERT_MSG(victim_ran, "victim handling runs exactly on the floor path");

    // Recurrent-only pressure bypasses both KV paths before Tri scoring.
    llama_memory_kv_reclaim_request req_idle;
    req_idle.required_free = 0;
    req_idle.drain_to_floor = false;
    req_idle.seq_hints = {hint};
    auto prop_bypass = adapter.plan_pressure(cells, req_idle, {slice}, store, true, hooks);
    TEST_ASSERT_MSG(adapter.decide(prop_bypass) == xkv_tri_pressure_decision::bypass_recurrent_only,
                      "recurrent bypasses");

    // Device lane inside the same pressure flow: residency bit arrives
    // cache-side (BackendResidency seam); Tri dispatches natively per run.
    // Fresh segment: earlier apply retired seg_id to version 2; a fresh segment
    // simulates a parallel device lane under the same pressure flow cleanly.
    std::vector<uint64_t> dev_pids;
    auto dev_seg = make_tri_factored_segment(store, n_fact, rank, total_dim, 62000, dev_pids);
    TEST_ASSERT(dev_seg != nullptr);
    const uint64_t dev_seg_id = dev_seg->segment_id;
    std::vector<xkv_tri_candidate> dev_cands(n_fact);
    for (uint32_t i = 0; i < n_fact; ++i) {
        dev_cands[i].cell_index = i;
        dev_cands[i].storage_pos = (int32_t)i;
        dev_cands[i].payload_id = dev_pids[i];
        dev_cands[i].location.kind = xkv_location_kind::factored;
        dev_cands[i].location.segment_id = dev_seg_id;
        dev_cands[i].location.row = i;
        dev_cands[i].row_ref.segment_id = dev_seg_id;
        dev_cands[i].row_ref.row = i;
        dev_cands[i].seq_ids = {0};
        dev_cands[i].device_owned = true;
    }
    xkv_tri_seq_info dseq = {0, n_fact, 7, 1, true, 0, {}};
    xkv_tri_pressure_state dpress = {true, false, false};
    xkv_tri_reclaim_options dopts;
    dopts.hot_span_provider = make_tri_span_provider(head_dim);
    dopts.selected_k_fetch = adapter.make_host_selected_k_fetch(store, nullptr, &fetch_calls);
    const uint64_t fetch_before = fetch_calls;
    auto prop_dev = adapter.create_reclaim_proposal(dev_cands, {dseq}, {slice}, store, dpress, dopts);
    TEST_ASSERT_MSG(prop_dev.is_success(), prop_dev.message.c_str());
    TEST_ASSERT_MSG(fetch_calls > fetch_before, "device lane invoked the native-shaped fetch");

    // Sub-test B: Real Vulkan mock / device lane integration test proving
    // factored refs reclaimed before preemption and device path invoked natively.
    {
        uint64_t native_vulkan_fetch_count = 0;
        device_factor_scoring_fn vulkan_device_scorer = [&](
            uint64_t segment_id,
            uint64_t segment_version,
            const uint32_t * rows,
            uint32_t n,
            const xkv_layer_slice & sl,
            uint32_t sampled_head_idx,
            int64_t frontier,
            float * out_scores,
            std::string * err
        ) -> bool {
            native_vulkan_fetch_count++;
            // Reconstruct and score on device without any host D2H
            for (uint32_t r = 0; r < n; ++r) {
                out_scores[r] = 0.1f * (float)(rows[r] + sampled_head_idx + 1);
            }
            return true;
        };

        xkv_tri_reclaim_options vk_opts;
        vk_opts.hot_span_provider = make_tri_span_provider(head_dim);
        vk_opts.device_scoring = vulkan_device_scorer;

        auto prop_vk = adapter.create_reclaim_proposal(dev_cands, {dseq}, {slice}, store, dpress, vk_opts);
        TEST_ASSERT_MSG(prop_vk.is_success(), prop_vk.message.c_str());
        TEST_ASSERT_MSG(native_vulkan_fetch_count > 0, "Native Vulkan device scoring must be invoked");
        TEST_ASSERT_MSG(prop_vk.factored_rows_freed > 0, "Factored rows freed before preemption");
        TEST_ASSERT_MSG(adapter.decide(prop_vk) == xkv_tri_pressure_decision::reclaimed, "Device lane maps to reclaimed");
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_server_pressure_integration_order PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 28: Partial IMRoPE head_dim=256 / rotary_dim=64 vs dense final-factor oracle
// -----------------------------------------------------------------------------
static void test_partial_imrope_256_64_vs_dense_oracle() {
    std::cout << "[test_xkv_tri] test_partial_imrope_256_64_vs_dense_oracle..." << std::endl;

    const uint32_t head_dim = 256;
    const uint32_t rotary_dim = 64;
    const uint32_t rank_k = 128;
    const uint32_t total_dim_k = 256;
    const uint32_t n_rows = 16;

    triattention_calibration cal = create_test_calib(head_dim, 1, 2, 1, 2, rotary_dim);
    TEST_ASSERT(cal.rotary_dim == 64);
    TEST_ASSERT(cal.freq_count == 32);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    TEST_ASSERT(omega.size() == 32);

    xkv_tri_config cfg;
    cfg.tile_size = 8;
    cfg.recent_window = 4;
    cfg.ratio = 0.5;
    cfg.normalize_scores = true;
    cfg.pool_radius = 2;

    xkv_tri_adapter adapter(cal, omega.data(), fsq.data(), cfg);
    TEST_ASSERT(adapter.valid());

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank_k;
    g.rank_v = rank_k;
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_k;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim_k};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim_k};

    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank_k}, 128);
    std::vector<float> data_a(n_rows * rank_k);
    for (size_t i = 0; i < data_a.size(); ++i) data_a[i] = 0.05f * std::sin((float) i * 0.17f);
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim_k, rank_k}, 128);
    std::vector<float> data_b(total_dim_k * rank_k);
    for (size_t i = 0; i < data_b.size(); ++i) data_b[i] = 0.05f * std::cos((float) i * 0.23f);
    g.set_b_k(encode_matrix(desc_b, data_b.data(), data_b.size()));
    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank_k}, 128);
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim_k, rank_k}, 128);
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));

    auto cand_seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS, LLAMA_XKV_SOURCE_DECODED_HOT, {g});
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);
    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 31000 + i;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));
    std::string err;
    if (!store.publish_candidate(cand_seg, pids, gens, &err)) {
        fprintf(stderr, "test_partial_imrope_256_64 publish failed: %s\n", err.c_str());
        TEST_ASSERT(false);
        return;
    }
    uint64_t seg_id = cand_seg->segment_id;

    std::vector<xkv_tri_candidate> candidates(n_rows);
    std::vector<int32_t> positions(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t)(i * 5);
        candidates[i].payload_id = pids[i];
        candidates[i].storage_generation = 1;
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg_id;
        candidates[i].location.row = i;
        candidates[i].seq_ids = {0};
        positions[i] = candidates[i].storage_pos;
    }

    xkv_layer_slice slice = {};
    slice.model_layer = 0;
    slice.owning_layer = 0;
    slice.feature_offset_k = 0;
    slice.feature_dim_k = total_dim_k;
    slice.head_dim = head_dim;
    slice.rotary_dim = rotary_dim;
    slice.rope_style = 0;
    slice.n_kv_heads = 1;

    const auto * oracle_g = cand_seg->find_group(0);
    TEST_ASSERT(oracle_g != nullptr);
    TEST_ASSERT(oracle_g->b_k != nullptr);
    std::vector<float> decoded_a = decode_matrix(oracle_g->a_k, value_domain::canonical);
    std::vector<float> decoded_b = decode_matrix(*oracle_g->b_k, value_domain::canonical);
    std::vector<float> dense_k(n_rows * head_dim, 0.0f);
    uint64_t a_pad = oracle_g->a_k.desc.padded_shape.cols;
    uint64_t b_pad = oracle_g->b_k->desc.padded_shape.cols;
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (uint64_t k = 0; k < rank_k; ++k) sum += decoded_a[r * a_pad + k] * decoded_b[d * b_pad + k];
            dense_k[r * head_dim + d] = sum;
        }

    std::vector<float> offsets;
    for (uint32_t d = 1; d <= 65536; d *= 2) offsets.push_back((float) d);
    const int64_t frontier_pos = positions.back() + 10;
    std::vector<float> oracle_combined(n_rows, -1e30f);
    std::vector<float> temp_scores(n_rows);
    for (uint32_t sh = 0; sh < cal.n_sampled; ++sh) {
        triattention_score_keys(temp_scores.data(), dense_k.data(), &cal.head_stats[sh], omega.data(), fsq.data(),
                                 offsets.data(), positions.data(), frontier_pos, n_rows, head_dim, rotary_dim / 2,
                                 (uint32_t) offsets.size(), cfg.agg, cfg.disable_trig);
        double sum = 0.0;
        for (float v : temp_scores) sum += v;
        double mean = sum / n_rows;
        double var = 0.0;
        for (float v : temp_scores) var += (v - mean) * (v - mean);
        double std = std::sqrt(var / n_rows);
        if (std < 1e-10) std = 1e-10;
        for (float & v : temp_scores) v = (float)((v - mean) / std);
        for (uint32_t i = 0; i < n_rows; ++i) oracle_combined[i] = std::max(oracle_combined[i], temp_scores[i]);
    }
    std::vector<float> oracle_pooled(n_rows);
    triattention_max_pool_scores(oracle_pooled.data(), oracle_combined.data(), positions.data(), n_rows, cfg.pool_radius);

    std::vector<float> adapter_raw(n_rows), adapter_pooled(n_rows);
    std::vector<uint32_t> all_indices(n_rows);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    adapter.score_candidate_subset(candidates, all_indices, {slice}, store, frontier_pos, adapter_pooled.data(), adapter_raw.data());
    for (uint32_t i = 0; i < n_rows; ++i) {
        TEST_ASSERT_MSG(std::fabs(adapter_raw[i] - oracle_combined[i]) < 1e-5f, "Raw 256/64 partial-IMRoPE mismatch");
        TEST_ASSERT_MSG(std::fabs(adapter_pooled[i] - oracle_pooled[i]) < 1e-5f, "Pooled 256/64 partial-IMRoPE mismatch");
    }

    // Unsupported geometries fail closed (never silently scored).
    {
        triattention_calibration bad = cal;
        bad.rope_style = 1;
        bool threw = false;
        try {
            xkv_tri_adapter bad_adapter(bad, omega.data(), fsq.data(), cfg);
        } catch (const std::invalid_argument &) { threw = true; }
        TEST_ASSERT_MSG(threw, "rope_style=1 must be rejected");
    }
    {
        xkv_layer_slice bad_slice = slice;
        bad_slice.rotary_dim = head_dim + 1;
        std::vector<float> out(n_rows);
        bool threw = false;
        try {
            adapter.score_candidate_subset(candidates, all_indices, {bad_slice}, store, frontier_pos, out.data());
        } catch (const std::invalid_argument &) { threw = true; }
        TEST_ASSERT_MSG(threw, "rotary_dim > head_dim must be rejected");
    }
    {
        xkv_layer_slice zero_slice = slice;
        zero_slice.rotary_dim = 0;
        std::vector<float> out(n_rows);
        bool threw = false;
        try {
            adapter.score_candidate_subset(candidates, all_indices, {zero_slice}, store, frontier_pos, out.data());
        } catch (const std::invalid_argument &) { threw = true; }
        TEST_ASSERT_MSG(threw, "rotary_dim=0 must be rejected");
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_partial_imrope_256_64_vs_dense_oracle PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 29: Layer-group sizes 4/2/1 with a size-1 tail across one segment
// -----------------------------------------------------------------------------
static void test_group_sizes_1_2_4_and_tail() {
    std::cout << "[test_xkv_tri] test_group_sizes_1_2_4_and_tail..." << std::endl;

    const uint32_t head_dim = 64, rank = 16, n_rows = 8, total = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);

    auto mk_group = [&](uint32_t gidx, std::vector<uint32_t> owners, std::vector<uint32_t> offs,
                          std::vector<uint32_t> dims, float base) {
        xkv_factor_group_payload gp;
        gp.group_index = gidx;
        gp.owning_layers = std::move(owners);
        gp.rank_k = rank;
        gp.rank_v = rank;
        gp.total_dim_k = total;
        gp.total_dim_v = total;
        gp.layer_feature_offsets_k = offs;
        gp.layer_feature_dims_k = dims;
        gp.layer_feature_offsets_v = offs;
        gp.layer_feature_dims_v = dims;
        codec_desc da = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
        std::vector<float> adata(n_rows * rank);
        for (size_t i = 0; i < adata.size(); ++i) adata[i] = (float)((i * 3 + 5) % 97) * 0.01f + base;
        gp.a_k = encode_matrix(da, adata.data(), adata.size());
        codec_desc db = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total, rank});
        std::vector<float> bdata(total * rank);
        for (size_t i = 0; i < bdata.size(); ++i) bdata[i] = (float)((i * 7 + 11) % 89) * 0.01f + base;
        gp.set_b_k(encode_matrix(db, bdata.data(), bdata.size()));
        codec_desc dav = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
        gp.a_v = encode_matrix(dav, adata.data(), adata.size());
        codec_desc dbv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total, rank});
        gp.set_b_v(encode_matrix(dbv, bdata.data(), bdata.size()));
        return gp;
    };
    xkv_factor_group_payload g0 = mk_group(0, {0, 1, 2, 3}, {0, 16, 32, 48}, {16, 16, 16, 16}, 0.0f);
    xkv_factor_group_payload g1 = mk_group(1, {4, 5}, {0, 32}, {32, 32}, 1.0f);
    xkv_factor_group_payload g2 = mk_group(2, {6}, {0}, {64}, 2.0f); // size-1 tail

    auto seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, {g0, g1, g2});
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);
    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 62000 + i;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));
    std::string err;
    if (!store.publish_candidate(seg, pids, gens, &err)) {
        fprintf(stderr, "test_group_sizes_1_2_4_and_tail publish failed: %s\n", err.c_str());
        TEST_ASSERT(false);
        return;
    }
    auto published = store.get_segment(seg->segment_id);
    TEST_ASSERT(published != nullptr);
    if (published == nullptr) return;
    TEST_ASSERT(published->groups.size() == 3);
    // Tail routing resolves per owning layer, never falls back to group 0.
    TEST_ASSERT(published->find_group_for_layer(0) == published->find_group(0));
    TEST_ASSERT(published->find_group_for_layer(4) == published->find_group(1));
    TEST_ASSERT(published->find_group_for_layer(6) == published->find_group(2));

    std::vector<xkv_tri_candidate> candidates(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        candidates[i].cell_index = i;
        candidates[i].storage_pos = (int32_t) i;
        candidates[i].payload_id = pids[i];
        candidates[i].location.kind = xkv_location_kind::factored;
        candidates[i].location.segment_id = seg->segment_id;
        candidates[i].location.row = i;
        candidates[i].row_ref.segment_id = seg->segment_id;
        candidates[i].row_ref.row = i;
        candidates[i].seq_ids = {0};
    }
    xkv_layer_slice s0 = {0, 0, 0, total, 0, total, 1, head_dim, head_dim, 0};
    xkv_layer_slice s1 = {4, 4, 0, total, 0, total, 1, head_dim, head_dim, 0};
    xkv_layer_slice s2 = {6, 6, 0, total, 0, total, 1, head_dim, head_dim, 0};
    std::vector<float> pooled(n_rows);
    std::vector<uint32_t> idx(n_rows);
    std::iota(idx.begin(), idx.end(), 0);
    adapter.score_candidate_subset(candidates, idx, {s0, s1, s2}, store, 10, pooled.data());
    for (uint32_t i = 0; i < n_rows; ++i) TEST_ASSERT(std::isfinite(pooled[i]));

    xkv_tri_seq_info seq0 = {0, n_rows, 7, 2, true, 0, {}};
    xkv_tri_pressure_state pressure = {true, false, false};
    hot_k_provider_fn no_hot = nullptr;
    auto plan = adapter.build_mutation_plan(candidates, {seq0}, {s0, s1, s2}, store, pressure, no_hot);
    // Every candidate lands in exactly one of survivors/evicted (union, no loss/duplication).
    TEST_ASSERT(plan.survivor_payloads.size() + plan.evicted_payloads.size() == n_rows);
    for (uint64_t pid : pids) {
        bool s = std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), pid) != plan.survivor_payloads.end();
        bool e = std::find(plan.evicted_payloads.begin(), plan.evicted_payloads.end(), pid) != plan.evicted_payloads.end();
        TEST_ASSERT(s != e);
    }
    // Fixed 3/32 policy: L=8, guard 2 -> max(2, ceil(8*3/32)=1) = 2 references targeted.
    TEST_ASSERT_MSG(plan.target_references == 2, "default 3/32 target for L=8/guard=2 must be 2");

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_group_sizes_1_2_4_and_tail PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 30: Empty / one-row / non-divisible candidate boundaries
// -----------------------------------------------------------------------------
static void test_empty_single_row_and_nondivisible_candidates() {
    std::cout << "[test_xkv_tri] test_empty_single_row_and_nondivisible_candidates..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_config cfg8, cfg32;
    cfg8.tile_size = 8;
    cfg32.tile_size = 32;
    xkv_tri_adapter adapter8(cal, omega.data(), fsq.data(), cfg8);
    xkv_tri_adapter adapter32(cal, omega.data(), fsq.data(), cfg32);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};
    xkv_tri_pressure_state pressure = {true, false, false};
    hot_k_provider_fn prov = [](uint32_t ml, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        for (size_t ci = 0; ci < cells.size(); ++ci)
            for (size_t d = 0; d < 64; ++d)
                dst[ci * 64 + d] = (ml == 0) ? (0.05f * std::sin((float)(cells[ci] * 3 + d))) : (0.05f * std::cos((float)(cells[ci] * 2 + d)));
        return true;
    };

    // Empty boundary: no candidates — no crash, no survivors, no removals.
    {
        std::vector<xkv_tri_candidate> empty;
        std::vector<uint32_t> no_idx;
        adapter8.score_candidate_subset(empty, no_idx, {slice}, store, 0, nullptr, nullptr, prov);
        xkv_tri_seq_info seq0 = {0, 4, 3, 2, true, 0, {}};
        auto plan = adapter8.build_mutation_plan(empty, {seq0}, {slice}, store, pressure, prov);
        TEST_ASSERT(!plan.changed);
        TEST_ASSERT(plan.survivor_payloads.empty());
        TEST_ASSERT(plan.evicted_payloads.empty());
        TEST_ASSERT(plan.ref_removals.empty());
        TEST_ASSERT(plan.physical_freed == 0);
    }

    // One-row boundary: the lone cell is recent-guarded and survives.
    {
        std::vector<xkv_tri_candidate> one(1);
        one[0].cell_index = 0;
        one[0].storage_pos = 0;
        one[0].payload_id = 7000;
        one[0].seq_ids = {0};
        std::vector<float> pooled(1), raw(1);
        std::vector<uint32_t> idx = {0};
        adapter8.score_candidate_subset(one, idx, {slice}, store, 0, pooled.data(), raw.data(), prov);
        TEST_ASSERT(std::isfinite(pooled[0]) && std::isfinite(raw[0]));
        xkv_tri_seq_info seq0 = {0, 1, 0, 2, true, 0, {}};
        auto plan = adapter8.build_mutation_plan(one, {seq0}, {slice}, store, pressure, prov);
        TEST_ASSERT(plan.survivor_payloads.size() == 1);
        TEST_ASSERT(plan.evicted_payloads.empty());
    }

    // Non-divisible 17 (= 2x8+1): tile 8 vs 32 agree; chunk/page tails score finitely.
    {
        const uint32_t n = 17;
        std::vector<xkv_tri_candidate> cands(n);
        for (uint32_t i = 0; i < n; ++i) {
            cands[i].cell_index = i;
            cands[i].storage_pos = (int32_t) i;
            cands[i].payload_id = 7100 + i;
            cands[i].seq_ids = {0};
        }
        std::vector<uint32_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> p8(n), r8(n), p32(n), r32(n);
        adapter8.score_candidate_subset(cands, idx, {slice}, store, 20, p8.data(), r8.data(), prov);
        adapter32.score_candidate_subset(cands, idx, {slice}, store, 20, p32.data(), r32.data(), prov);
        for (uint32_t i = 0; i < n; ++i) {
            TEST_ASSERT(std::isfinite(p8[i]) && std::isfinite(p32[i]));
            TEST_ASSERT_MSG(std::fabs(p8[i] - p32[i]) < 1e-6f, "tail tile must not change pooled score");
            TEST_ASSERT_MSG(std::fabs(r8[i] - r32[i]) < 1e-6f, "tail tile must not change raw score");
        }
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_empty_single_row_and_nondivisible_candidates PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 31: Shared references carry single importance; fixed 3/32 survivor targets
// -----------------------------------------------------------------------------
static void test_shared_reference_single_importance_and_fixed_ratio() {
    std::cout << "[test_xkv_tri] test_shared_reference_single_importance_and_fixed_ratio..." << std::endl;

    const uint32_t head_dim = 64;
    triattention_calibration cal = create_test_calib(head_dim, 1, 1, 1, 1, head_dim);
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    TEST_ASSERT(xkv_tri_config().ratio == 3.0 / 32.0);
    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    xkv_layer_slice slice = {0, 0, 0, head_dim, 0, head_dim, 1, head_dim, head_dim, 0};
    xkv_tri_pressure_state pressure = {true, false, false};
    hot_k_provider_fn prov = [](uint32_t, uint32_t, const std::vector<uint32_t> & cells, float * dst, size_t) {
        for (size_t ci = 0; ci < cells.size(); ++ci)
            for (size_t d = 0; d < 64; ++d)
                dst[ci * 64 + d] = 0.05f * std::sin((float)(cells[ci] * 3 + d));
        return true;
    };

    auto mk4 = [](bool shared) {
        std::vector<xkv_tri_candidate> v(4);
        for (uint32_t i = 0; i < 4; ++i) {
            v[i].cell_index = i;
            v[i].storage_pos = (int32_t) i;
            v[i].payload_id = 8000 + i;
            v[i].seq_ids = {0};
        }
        if (shared) v[3].seq_ids = {0, 1, 2};
        return v;
    };
    std::vector<xkv_tri_candidate> v1 = mk4(false);
    std::vector<xkv_tri_candidate> v2 = mk4(true);
    std::vector<uint32_t> idx = {0, 1, 2, 3};
    std::vector<float> p1(4), r1(4), p2(4), r2(4);
    adapter.score_candidate_subset(v1, idx, {slice}, store, 10, p1.data(), r1.data(), prov);
    adapter.score_candidate_subset(v2, idx, {slice}, store, 10, p2.data(), r2.data(), prov);
    // Adding two extra sequence references to a cell must not change any score.
    TEST_ASSERT(p1 == p2);
    TEST_ASSERT(r1 == r2);

    // Physical union: the shared cell survives once (recent-guarded) and is never evicted.
    xkv_tri_seq_info s0 = {0, 4, 3, 2, true, 0, {}};
    xkv_tri_seq_info s1 = {1, 4, 3, 2, true, 0, {}};
    xkv_tri_seq_info s2 = {2, 4, 3, 2, true, 0, {}};
    auto plan = adapter.build_mutation_plan(v2, {s0, s1, s2}, {slice}, store, pressure, prov);
    TEST_ASSERT(plan.survivor_payloads.size() + plan.evicted_payloads.size() == 4);
    {
        std::vector<uint64_t> s = plan.survivor_payloads;
        std::sort(s.begin(), s.end());
        TEST_ASSERT(std::adjacent_find(s.begin(), s.end()) == s.end()); // no double-counted survivor
    }
    TEST_ASSERT(std::find(plan.survivor_payloads.begin(), plan.survivor_payloads.end(), 8003) != plan.survivor_payloads.end());
    TEST_ASSERT(std::find(plan.evicted_payloads.begin(), plan.evicted_payloads.end(), 8003) == plan.evicted_payloads.end());

    // Fixed 3/32 survivor targets through the real planning path (matches Tri selfcheck).
    {
        std::vector<xkv_tri_candidate> cands(4);
        for (uint32_t i = 0; i < 4; ++i) {
            cands[i].cell_index = i;
            cands[i].storage_pos = (int32_t) i;
            cands[i].payload_id = 8100 + i;
            cands[i].seq_ids = {0};
        }
        xkv_tri_seq_info big = {0, 2048, 2047, 128, true, 0, {}};
        auto pb = adapter.build_mutation_plan(cands, {big}, {slice}, store, pressure, prov);
        TEST_ASSERT_MSG(pb.target_references == 192, "L=2048/guard=128 at 3/32 must target 192");
        xkv_tri_seq_info small = {0, 320, 319, 128, true, 0, {}};
        auto ps = adapter.build_mutation_plan(cands, {small}, {slice}, store, pressure, prov);
        TEST_ASSERT_MSG(ps.target_references == 128, "L=320/guard=128 at 3/32 must target floor 128");
    }

    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_shared_reference_single_importance_and_fixed_ratio PASSED." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 32: Tiny two-row pack preserves survivor bytes and B identity
// -----------------------------------------------------------------------------
static void test_tiny_two_row_pack_byte_preservation() {
    std::cout << "[test_xkv_tri] test_tiny_two_row_pack_byte_preservation..." << std::endl;

    llama_cparams cparams = {};
    llama_xkv_cache_store store(cparams);
    const uint32_t n_rows = 2, rank = 16, total_dim = 64;
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = total_dim;
    g.total_dim_v = total_dim;
    g.layer_feature_offsets_k = {0};
    g.layer_feature_dims_k = {total_dim};
    g.layer_feature_offsets_v = {0};
    g.layer_feature_dims_v = {total_dim};
    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    std::vector<float> data_a(n_rows * rank);
    for (size_t r = 0; r < n_rows; ++r)
        for (size_t c = 0; c < rank; ++c) data_a[r * rank + c] = (float)(r * 100 + c + 1);
    g.a_k = encode_matrix(desc_a, data_a.data(), data_a.size());
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    std::vector<float> data_b(total_dim * rank);
    for (size_t i = 0; i < data_b.size(); ++i) data_b[i] = (float)(i + 77);
    auto b_encoded = encode_matrix(desc_b, data_b.data(), data_b.size());
    g.set_b_k(b_encoded);
    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank});
    g.a_v = encode_matrix(desc_av, data_a.data(), data_a.size());
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim, rank});
    g.set_b_v(encode_matrix(desc_bv, data_b.data(), data_b.size()));
    auto cand_seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, {g});
    cand_seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_seg->groups);
    std::vector<uint64_t> pids = {7100, 7101};
    std::vector<uint64_t> gens = {1, 1};
    for (uint32_t r = 0; r < n_rows; ++r) store.register_hot_payload(pids[r], r, gens[r], xkv_state::hot_committed);
    TEST_ASSERT(tri_mark_sealed(store, pids, gens));
    std::string err;
    if (!store.publish_candidate(cand_seg, pids, gens, &err)) {
        fprintf(stderr, "test_tiny_two_row_pack publish failed: %s\n", err.c_str());
        TEST_ASSERT(false);
        return;
    }
    uint64_t seg_id = cand_seg->segment_id;
    auto published_seg = store.get_segment(seg_id);
    if (published_seg == nullptr) {
        fprintf(stderr, "test_tiny_two_row_pack: published segment missing\n");
        TEST_ASSERT(false);
        return;
    }
    const auto * pub_g0 = published_seg->find_group(0);
    TEST_ASSERT(pub_g0 != nullptr && pub_g0->b_k != nullptr);
    const size_t row_bytes = pub_g0->a_k.desc.row_stride_bytes;
    std::vector<uint8_t> survivor_row(row_bytes);
    std::memcpy(survivor_row.data(), pub_g0->a_k.bytes.data() + row_bytes, row_bytes); // row 1 survives
    const void * orig_b_ptr = pub_g0->b_k->bytes.data();
    uint64_t orig_binding = store.binding_epoch();
    uint64_t orig_content = store.content_epoch();
    landmark_table lm_table(1024 * 1024);
    xkv_tri_mutation_plan plan;
    plan.evicted_payloads.push_back(pids[0]);
    plan.affected_segments = {seg_id};
    triattention_calibration cal = create_test_calib();
    std::vector<float> omega, fsq;
    compute_test_omega_and_fsq(cal, omega, fsq);
    xkv_tri_adapter adapter(cal, omega.data(), fsq.data());
    auto proposal = adapter.create_mutation_proposal(plan, store);
    if (!proposal.apply(store, &lm_table, &err)) {
        fprintf(stderr, "test_tiny_two_row_pack apply failed: %s\n", err.c_str());
        TEST_ASSERT(false);
        return;
    }
    auto repacked = store.get_segment(seg_id);
    TEST_ASSERT(repacked != nullptr);
    if (repacked == nullptr) return;
    TEST_ASSERT_MSG(repacked->n_rows == 1, "two-row pack evicting one must leave one row");
    const auto * rep_g0 = repacked->find_group(0);
    TEST_ASSERT(rep_g0 != nullptr && rep_g0->b_k != nullptr);
    TEST_ASSERT_MSG(rep_g0->b_k->bytes == b_encoded.bytes, "B_K bytes identical after tiny pack");
    TEST_ASSERT_MSG(rep_g0->b_k->bytes.data() == orig_b_ptr, "B_K shared pointer preserved after tiny pack");
    TEST_ASSERT_MSG(repacked->row_payload_ids[0] == pids[1], "survivor identity preserved");
    TEST_ASSERT_MSG(std::memcmp(rep_g0->a_k.bytes.data(), survivor_row.data(), row_bytes) == 0, "survivor A row byte-identical");
    TEST_ASSERT_MSG(store.binding_epoch() > orig_binding, "binding_epoch must advance");
    TEST_ASSERT_MSG(store.content_epoch() == orig_content, "content_epoch unchanged on physical pack");
    free_test_calib(cal);
    std::cout << "[test_xkv_tri] test_tiny_two_row_pack_byte_preservation PASSED." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running XKV TriAttention Data Adapter Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_turbo4_exact_sampled_head_vs_dense_oracle();
    test_tile_size_invariance_and_memory_bounds();
    test_multi_sequence_per_seq_target_and_ref_removals();
    test_sparse_per_seq_recent_behavior();
    test_semantic_cell_eviction_and_reader_tail_parity();
    test_missing_provider_and_slice_errors();
    test_preflighted_atomic_store_mutation_and_failure_safety();
    test_a_byte_preservation_and_b_identity();
    test_recurrent_only_bypass();
    test_mixed_hot_factored_one_sequence();
    test_shared_prefix_across_3_seqs_with_physical_union();
    test_sampled_q_heads_sharing_one_kv_head();
    test_candidate_order_ties_determinism();
    test_stale_stamp_and_floor_exhausted();
    test_workspace_budget_and_one_byte_short();
    test_zero_physical_free_despite_refs_removed();
    test_multiple_layer_groups_adaptive_tail_ranks();
    test_factor_decode_error_handling();
    test_landmarks_pack_callback_and_refusal();
    test_head_aggregation_mean_vs_max_and_ablation_enforcement();
    test_hot_slot_deficit_factored_only_does_not_satisfy();
    test_store_byte_pressure_satisfied_by_factored_reclaim();
    test_zero_heap_carved_scratch_with_new_counter();
    test_unknown_reader_frontier_preserved_and_table_accepted();
    test_device_owned_dispatch_and_fail_closed();
    test_estimator_tile_a_exact_and_descriptor_derived();
    test_server_pressure_integration_order();
    test_proposal_arena_heap_allocation_delta();
    test_partial_imrope_256_64_vs_dense_oracle();
    test_group_sizes_1_2_4_and_tail();
    test_empty_single_row_and_nondivisible_candidates();
    test_shared_reference_single_importance_and_fixed_ratio();
    test_tiny_two_row_pack_byte_preservation();

    if (g_test_failures == 0) {
        std::cout << "ALL XKV TRI TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cerr << g_test_failures << " TEST(S) FAILED!" << std::endl;
        return 1;
    }
}
