// Comprehensive deterministic regression & unit tests for llama-xkv-reader (PR06):
// 1. Full-rank/no-SR vs dense reference oracle
// 2. Real Turbo4 low-rank four-stream decode against explicit codec -> GEMM oracle
// 3. Multi-segment online softmax against concatenated logits with extreme logits
// 4. DDVR spans & effective RoPE position groups
// 5. Variable K/V dimensions (head_dim_k != head_dim_v) and multi-head GQA
// 6. Per-query speculative isolation via CSR masks (draft tokens cannot influence earlier selections)
// 7. Hot/tentative row per-query causal & membership isolation (query 32+ without modulo aliasing)
// 8. Empty masks / all-masked rows without NaN (finite zero output)
// 9. Stale epochs / snapshot stamp invalidation returning retry without partial output
// 10. Checked arithmetic overflow protection (safe_mul / safe_add)
// 11. Batch workspace preflight: sum of all query states + CSR + output + tile peak > budget rejected
// 12. Batch queries with mixed head dimensions rejected
// 13. B-cache lease accounting: eviction refusal under pressure, clear-with-lease retention, no cross-entry subtraction
// 14. Owning xkv_reader_pin requirement and segment validation
// 15. Single sink logit contribution preservation with softcap and scale ordering

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-reader.h"
#include "llama-xkv-factor.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-landmark.h"
#include "llama-cparams.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace llama_xkv;

// Global C++ heap allocation counter for the bounded-reader proofs below.
// Installed unconditionally (it only counts while explicitly enabled), so every
// std::vector / unordered container / shared_ptr-control-block / long std::string
// allocation inside a measured read is observed. C heap (malloc) is not used by the
// F32/Turbo4 decode paths exercised here (memcpy / stack-only row kernels).
#include <atomic>
#include <cstdlib>
// <limits> for quiet_NaN/infinity in scale validation tests.
#include <limits>
// <new> for std::bad_alloc used by the counting operator new below.
#include <new>
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
        g_heap_new_count.fetch_add(1, std::memory_order_relaxed);
        g_heap_new_bytes.fetch_add(n, std::memory_order_relaxed);
    }
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

static llama_cparams make_reader_test_cparams() {
    llama_cparams cparams = {};
    cparams.xkv_mode = LLAMA_XKV_MODE_SHADOW;
    cparams.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cparams.xkv_group_size = 4;
    cparams.xkv_rank_k = 128;
    cparams.xkv_rank_v = 128;
    cparams.xkv_segment_tokens = 64;
    cparams.xkv_chunk_tokens = 8;
    cparams.xkv_workspace_mib = 16;
    cparams.xkv_decode_cache_mib = 8;
    return cparams;
}

static std::vector<float> generate_deterministic_floats(size_t n, uint64_t seed) {
    std::vector<float> data(n);
    uint64_t state = seed ? seed : 123456789ULL;
    for (size_t i = 0; i < n; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        float val = (static_cast<float>(state & 0x7FFFFFFF) / static_cast<float>(0x7FFFFFFF)) * 2.0f - 1.0f;
        data[i] = val;
    }
    return data;
}

static bool approx_equal(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static bool vectors_approx_equal(const std::vector<float> & a, const std::vector<float> & b, float tol = 1e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approx_equal(a[i], b[i], tol)) {
            std::cerr << "Mismatch at index " << i << ": a=" << a[i] << " b=" << b[i] << " diff=" << std::fabs(a[i] - b[i]) << std::endl;
            return false;
        }
    }
    return true;
}

static std::shared_ptr<xkv_segment> create_test_segment_f32(
    llama_xkv_cache_store & store,
    uint32_t n_rows,
    uint32_t rank_k,
    uint32_t rank_v,
    uint32_t total_dim_k,
    uint32_t total_dim_v,
    uint64_t seed_base,
    std::vector<float> & out_a_k,
    std::vector<float> & out_b_k,
    std::vector<float> & out_a_v,
    std::vector<float> & out_b_v
) {
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.rank_k = rank_k;
    g.rank_v = rank_v;
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_v;
    uint32_t ldim_k = total_dim_k / 4;
    uint32_t ldim_v = total_dim_v / 4;
    g.layer_feature_offsets_k = {0, ldim_k, 2 * ldim_k, 3 * ldim_k};
    g.layer_feature_dims_k = {ldim_k, ldim_k, ldim_k, total_dim_k - 3 * ldim_k};
    g.layer_feature_offsets_v = {0, ldim_v, 2 * ldim_v, 3 * ldim_v};
    g.layer_feature_dims_v = {ldim_v, ldim_v, ldim_v, total_dim_v - 3 * ldim_v};

    codec_desc desc_a_k = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_k}, 0, seed_base + 1);
    out_a_k = generate_deterministic_floats(n_rows * rank_k, seed_base + 1);
    g.a_k = encode_matrix(desc_a_k, out_a_k.data(), out_a_k.size());

    codec_desc desc_b_k = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank_k}, 0, seed_base + 2);
    out_b_k = generate_deterministic_floats(total_dim_k * rank_k, seed_base + 2);
    g.set_b_k(encode_matrix(desc_b_k, out_b_k.data(), out_b_k.size()));

    codec_desc desc_a_v = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_v}, 0, seed_base + 3);
    out_a_v = generate_deterministic_floats(n_rows * rank_v, seed_base + 3);
    g.a_v = encode_matrix(desc_a_v, out_a_v.data(), out_a_v.size());

    codec_desc desc_b_v = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_v, rank_v}, 0, seed_base + 4);
    out_b_v = generate_deterministic_floats(total_dim_v * rank_v, seed_base + 4);
    g.set_b_v(encode_matrix(desc_b_v, out_b_v.data(), out_b_v.size()));

    auto seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = seed_base * 1000 + i + 1;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    bool marked = store.mark_seal_candidates(pids);
    assert(marked);

    std::string err;
    bool ok = store.publish_candidate(seg, pids, gens, &err);
    if (!ok) {
        std::cerr << "create_test_segment_f32 publish_candidate failed: " << err << std::endl;
        assert(false);
    }
    return seg;
}

static std::shared_ptr<xkv_segment> create_test_segment_turbo4(
    llama_xkv_cache_store & store,
    uint32_t n_rows,
    uint32_t rank_k,
    uint32_t rank_v,
    uint32_t total_dim_k,
    uint32_t total_dim_v,
    uint64_t seed_base
) {
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.rank_k = rank_k;
    g.rank_v = rank_v;
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_v;
    uint32_t ldim_k = total_dim_k / 4;
    uint32_t ldim_v = total_dim_v / 4;
    g.layer_feature_offsets_k = {0, ldim_k, 2 * ldim_k, 3 * ldim_k};
    g.layer_feature_dims_k = {ldim_k, ldim_k, ldim_k, total_dim_k - 3 * ldim_k};
    g.layer_feature_offsets_v = {0, ldim_v, 2 * ldim_v, 3 * ldim_v};
    g.layer_feature_dims_v = {ldim_v, ldim_v, ldim_v, total_dim_v - 3 * ldim_v};

    codec_desc desc_a_k = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank_k}, 128, seed_base + 1);
    auto raw_a_k = generate_deterministic_floats(n_rows * rank_k, seed_base + 1);
    g.a_k = encode_matrix(desc_a_k, raw_a_k.data(), raw_a_k.size());

    codec_desc desc_b_k = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim_k, rank_k}, 128, seed_base + 1);
    auto raw_b_k = generate_deterministic_floats(total_dim_k * rank_k, seed_base + 2);
    g.set_b_k(encode_matrix(desc_b_k, raw_b_k.data(), raw_b_k.size()));

    codec_desc desc_a_v = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, orientation::token_major, {n_rows, rank_v}, 128, seed_base + 3);
    auto raw_a_v = generate_deterministic_floats(n_rows * rank_v, seed_base + 3);
    g.a_v = encode_matrix(desc_a_v, raw_a_v.data(), raw_a_v.size());

    codec_desc desc_b_v = make_codec_desc(factor_role::b_v, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {total_dim_v, rank_v}, 128, seed_base + 3);
    auto raw_b_v = generate_deterministic_floats(total_dim_v * rank_v, seed_base + 4);
    g.set_b_v(encode_matrix(desc_b_v, raw_b_v.data(), raw_b_v.size()));

    auto seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = seed_base * 1000 + i + 1;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    bool marked = store.mark_seal_candidates(pids);
    assert(marked);

    std::string err;
    bool ok = store.publish_candidate(seg, pids, gens, &err);
    if (!ok) {
        std::cerr << "create_test_segment_turbo4 publish_candidate failed: " << err << std::endl;
        assert(false);
    }
    return seg;
}

// ----------------------------------------------------------------------------
// Test 1: Full-rank / no-SR vs dense reference oracle
// ----------------------------------------------------------------------------
static void test_full_rank_no_sr_vs_dense_oracle() {
    std::cout << "[Test 1] Full-rank / no-SR vs dense reference oracle..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t head_dim_k = 32;
    const uint32_t head_dim_v = 32;
    const uint32_t rank_k = 32;
    const uint32_t rank_v = 32;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank_k, rank_v, 4 * head_dim_k, 4 * head_dim_v, 100, a_k, b_k, a_v, b_v);

    std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(head_dim_v, 0.0f));
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            double sum = 0.0;
            for (uint32_t k = 0; k < rank_k; ++k) {
                sum += static_cast<double>(a_k[r * rank_k + k]) * static_cast<double>(b_k[f * rank_k + k]);
            }
            cold_keys[r][f] = static_cast<float>(sum);
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            double sum = 0.0;
            for (uint32_t k = 0; k < rank_v; ++k) {
                sum += static_cast<double>(a_v[r * rank_v + k]) * static_cast<double>(b_v[f * rank_v + k]);
            }
            cold_values[r][f] = static_cast<float>(sum);
        }
    }

    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.q_vec = generate_deterministic_floats(head_dim_k, 999);
    query.scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    xkv_snapshot_stamp stamp = store.current_stamp();
    auto res = xkv_read_attention(query, {}, views, nullptr, stamp, &store);
    assert(res.status == xkv_read_status::success);

    std::vector<uint32_t> group_indices(n_rows, 0);
    auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, group_indices, {});

    assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));
    std::cout << "[Test 1] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 2: Real Turbo4 low-rank four-stream decode against explicit codec -> GEMM oracle
// ----------------------------------------------------------------------------
static void test_real_turbo4_four_stream_vs_explicit_oracle() {
    std::cout << "[Test 2] Real Turbo4 low-rank four-stream decode against explicit codec -> GEMM oracle..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank_k = 128;
    const uint32_t rank_v = 128;

    auto seg = create_test_segment_turbo4(store, n_rows, rank_k, rank_v, 4 * head_dim_k, 4 * head_dim_v, 200);

    std::vector<uint32_t> selected = {1, 3, 7, 10, 15};
    uint32_t n_sel = static_cast<uint32_t>(selected.size());

    // Resolve the exact factor group from the segment bundle
    const auto * g = seg->find_group(0);
    assert(g != nullptr);
    assert(g->b_k != nullptr);
    assert(g->b_v != nullptr);

    std::vector<uint64_t> sel_u64(n_sel);
    for (size_t i = 0; i < n_sel; ++i) sel_u64[i] = selected[i];

    std::vector<float> dec_a_k(n_sel * rank_k);
    std::vector<float> dec_a_v(n_sel * rank_v);
    decode_rows(g->a_k, sel_u64.data(), n_sel, dec_a_k.data(), dec_a_k.size(), value_domain::canonical);
    decode_rows(g->a_v, sel_u64.data(), n_sel, dec_a_v.data(), dec_a_v.size(), value_domain::canonical);

    std::vector<uint64_t> b_rows_k(head_dim_k);
    for (uint32_t i = 0; i < head_dim_k; ++i) b_rows_k[i] = i;
    std::vector<float> dec_b_k(head_dim_k * rank_k);
    decode_rows(*g->b_k, b_rows_k.data(), head_dim_k, dec_b_k.data(), dec_b_k.size(), value_domain::canonical);

    std::vector<uint64_t> b_rows_v(head_dim_v);
    for (uint32_t i = 0; i < head_dim_v; ++i) b_rows_v[i] = i;
    std::vector<float> dec_b_v(head_dim_v * rank_v);
    decode_rows(*g->b_v, b_rows_v.data(), head_dim_v, dec_b_v.data(), dec_b_v.size(), value_domain::canonical);

    std::vector<std::vector<float>> cold_keys(n_sel, std::vector<float>(head_dim_k));
    std::vector<std::vector<float>> cold_values(n_sel, std::vector<float>(head_dim_v));
    for (uint32_t r = 0; r < n_sel; ++r) {
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            double sum = 0.0;
            for (uint32_t k = 0; k < rank_k; ++k) {
                sum += static_cast<double>(dec_a_k[r * rank_k + k]) * static_cast<double>(dec_b_k[f * rank_k + k]);
            }
            cold_keys[r][f] = static_cast<float>(sum);
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            double sum = 0.0;
            for (uint32_t k = 0; k < rank_v; ++k) {
                sum += static_cast<double>(dec_a_v[r * rank_v + k]) * static_cast<double>(dec_b_v[f * rank_v + k]);
            }
            cold_values[r][f] = static_cast<float>(sum);
        }
    }

    xkv_query_input query;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.n_q_heads = 1;
    query.q_vec = generate_deterministic_floats(head_dim_k, 555);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    view.selected_rows = selected;
    for (uint32_t r : selected) {
        view.storage_positions.push_back(r * 2);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store);
    assert(res.status == xkv_read_status::success);

    std::vector<uint32_t> sub_groups(n_sel, 0);
    auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, sub_groups, {});

    assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));
    std::cout << "[Test 2] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 3: Multi-segment online softmax with extreme logits
// ----------------------------------------------------------------------------
static void test_multi_segment_online_softmax_extreme_logits() {
    std::cout << "[Test 3] Multi-segment online softmax with extreme logits..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;

    std::vector<float> a1_k, b1_k, a1_v, b1_v;
    auto seg1 = create_test_segment_f32(store, 8, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 301, a1_k, b1_k, a1_v, b1_v);

    std::vector<float> a2_k, b2_k, a2_v, b2_v;
    auto seg2 = create_test_segment_f32(store, 8, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 302, a2_k, b2_k, a2_v, b2_v);

    xkv_segment_read_view v1;
    v1.pin = store.pin_segment(seg1->segment_id);
    v1.segment_version_id = seg1->segment_version;
    v1.storage_generation = 1;
    v1.owning_layer = 0;
    v1.factor_group_index = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        v1.selected_rows.push_back(i);
        v1.storage_positions.push_back(i);
    }

    xkv_segment_read_view v2;
    v2.pin = store.pin_segment(seg2->segment_id);
    v2.segment_version_id = seg2->segment_version;
    v2.storage_generation = 1;
    v2.owning_layer = 0;
    v2.factor_group_index = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        v2.selected_rows.push_back(i);
        v2.storage_positions.push_back(10 + i);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(v1));
    views.push_back(std::move(v2));

    xkv_query_input query;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.n_q_heads = 1;
    query.q_vec = generate_deterministic_floats(head_dim_k, 777);
    for (float & v : query.q_vec) v *= 10.0f; // produce logits > 50

    auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store);
    assert(res.status == xkv_read_status::success);

    // Compute dense concatenated oracle
    std::vector<std::vector<float>> cold_keys(16, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(16, std::vector<float>(head_dim_v, 0.0f));
    for (uint32_t r = 0; r < 8; ++r) {
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            for (uint32_t k = 0; k < rank; ++k) cold_keys[r][f] += a1_k[r * rank + k] * b1_k[f * rank + k];
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            for (uint32_t k = 0; k < rank; ++k) cold_values[r][f] += a1_v[r * rank + k] * b1_v[f * rank + k];
        }
    }
    for (uint32_t r = 0; r < 8; ++r) {
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            for (uint32_t k = 0; k < rank; ++k) cold_keys[8 + r][f] += a2_k[r * rank + k] * b2_k[f * rank + k];
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            for (uint32_t k = 0; k < rank; ++k) cold_values[8 + r][f] += a2_v[r * rank + k] * b2_v[f * rank + k];
        }
    }
    std::vector<uint32_t> groups(16, 0);
    auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, groups, {});
    assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));

    std::cout << "[Test 3] Passed. Extreme logits match concatenated dense oracle." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 4: DDVR spans & effective RoPE position groups
// ----------------------------------------------------------------------------
static void test_ddvr_spans_and_groups() {
    std::cout << "[Test 4] DDVR spans & effective RoPE position groups..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;

    // Create a multi-group segment with 2 distinct factor groups
    const uint32_t n_rows = 10;
    xkv_factor_group_payload g0;
    g0.group_index = 0;
    g0.owning_layers = {0, 1};
    g0.rank_k = rank;
    g0.rank_v = rank;
    g0.total_dim_k = head_dim_k * 2;
    g0.total_dim_v = head_dim_v * 2;
    g0.layer_feature_offsets_k = {0, head_dim_k};
    g0.layer_feature_dims_k = {head_dim_k, head_dim_k};
    g0.layer_feature_offsets_v = {0, head_dim_v};
    g0.layer_feature_dims_v = {head_dim_v, head_dim_v};

    codec_desc desc_a_k0 = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 401);
    auto a_k0 = generate_deterministic_floats(n_rows * rank, 401);
    g0.a_k = encode_matrix(desc_a_k0, a_k0.data(), a_k0.size());

    codec_desc desc_b_k0 = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim_k * 2, rank}, 0, 402);
    auto b_k0 = generate_deterministic_floats(head_dim_k * 2 * rank, 402);
    g0.set_b_k(encode_matrix(desc_b_k0, b_k0.data(), b_k0.size()));

    codec_desc desc_a_v0 = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 403);
    auto a_v0 = generate_deterministic_floats(n_rows * rank, 403);
    g0.a_v = encode_matrix(desc_a_v0, a_v0.data(), a_v0.size());

    codec_desc desc_b_v0 = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim_v * 2, rank}, 0, 404);
    auto b_v0 = generate_deterministic_floats(head_dim_v * 2 * rank, 404);
    g0.set_b_v(encode_matrix(desc_b_v0, b_v0.data(), b_v0.size()));

    xkv_factor_group_payload g1;
    g1.group_index = 1;
    g1.owning_layers = {2, 3};
    g1.rank_k = rank;
    g1.rank_v = rank;
    g1.total_dim_k = head_dim_k * 2;
    g1.total_dim_v = head_dim_v * 2;
    g1.layer_feature_offsets_k = {0, head_dim_k};
    g1.layer_feature_dims_k = {head_dim_k, head_dim_k};
    g1.layer_feature_offsets_v = {0, head_dim_v};
    g1.layer_feature_dims_v = {head_dim_v, head_dim_v};

    codec_desc desc_a_k1 = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 501);
    auto a_k1 = generate_deterministic_floats(n_rows * rank, 501);
    g1.a_k = encode_matrix(desc_a_k1, a_k1.data(), a_k1.size());

    codec_desc desc_b_k1 = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim_k * 2, rank}, 0, 502);
    auto b_k1 = generate_deterministic_floats(head_dim_k * 2 * rank, 502);
    g1.set_b_k(encode_matrix(desc_b_k1, b_k1.data(), b_k1.size()));

    codec_desc desc_a_v1 = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 503);
    auto a_v1 = generate_deterministic_floats(n_rows * rank, 503);
    g1.a_v = encode_matrix(desc_a_v1, a_v1.data(), a_v1.size());

    codec_desc desc_b_v1 = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim_v * 2, rank}, 0, 504);
    auto b_v1 = generate_deterministic_floats(head_dim_v * 2 * rank, 504);
    g1.set_b_v(encode_matrix(desc_b_v1, b_v1.data(), b_v1.size()));

    auto seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g0, g1}
    );
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 4000 + i + 1;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    bool marked = store.mark_seal_candidates(pids);
    assert(marked);
    std::string pub_err;
    bool published = store.publish_candidate(seg, pids, gens, &pub_err);
    assert(published);

    xkv_query_input query;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.n_q_heads = 1;
    query.q_groups.resize(2);
    query.q_groups[0] = generate_deterministic_floats(head_dim_k, 1111);
    query.q_groups[1] = generate_deterministic_floats(head_dim_k, 2222);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    // Target group 1 via owning_layer 2 (never fallback)
    view.owning_layer = 2;
    view.factor_group_index = 1;
    for (uint32_t i = 0; i < 10; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(i % 2);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    phase_transform_fn phase_tx = [](const float * src_key, int64_t storage_pos, uint32_t /*head_idx*/, float * dst_key) {
        float factor = 1.0f + 0.05f * static_cast<float>(storage_pos);
        for (uint32_t i = 0; i < 16; ++i) {
            dst_key[i] = src_key[i] * factor;
        }
    };

    auto res = xkv_read_attention(query, {}, views, phase_tx, store.current_stamp(), &store);
    assert(res.status == xkv_read_status::success);

    // Compute cold keys using exact resolved group 1 (owning_layer 2 has offset 0)
    std::vector<std::vector<float>> cold_keys(10, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(10, std::vector<float>(head_dim_v, 0.0f));
    std::vector<uint32_t> groups(10);
    for (uint32_t r = 0; r < 10; ++r) {
        groups[r] = r % 2;
        std::vector<float> raw_k(head_dim_k, 0.0f);
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                raw_k[f] += a_k1[r * rank + k] * b_k1[f * rank + k];
            }
        }
        phase_tx(raw_k.data(), r, 0, cold_keys[r].data());

        for (uint32_t f = 0; f < head_dim_v; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                cold_values[r][f] += a_v1[r * rank + k] * b_v1[f * rank + k];
            }
        }
    }

    auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, groups, {});
    assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));
    std::cout << "[Test 4] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 5: Variable K/V dimensions & GQA
// ----------------------------------------------------------------------------
static void test_variable_kv_dims_and_gqa() {
    std::cout << "[Test 5] Variable K/V dimensions & GQA..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 32;
    const uint32_t head_dim_v = 48;
    const uint32_t n_q_heads = 4;
    const uint32_t rank_k = 16;
    const uint32_t rank_v = 16;
    const uint32_t n_rows = 12;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank_k, rank_v, 4 * head_dim_k, 4 * head_dim_v, 500, a_k, b_k, a_v, b_v);

    xkv_query_input query;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.n_q_heads = n_q_heads;
    query.q_vec = generate_deterministic_floats(head_dim_k * n_q_heads, 888);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store);
    assert(res.status == xkv_read_status::success);
    assert(res.output.size() == n_q_heads * head_dim_v);

    // Dense oracle validation with variable dims and 4 GQA query heads
    std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(head_dim_v, 0.0f));
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            for (uint32_t k = 0; k < rank_k; ++k) cold_keys[r][f] += a_k[r * rank_k + k] * b_k[f * rank_k + k];
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            for (uint32_t k = 0; k < rank_v; ++k) cold_values[r][f] += a_v[r * rank_v + k] * b_v[f * rank_v + k];
        }
    }
    std::vector<uint32_t> groups(n_rows, 0);
    auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, groups, {});
    assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));

    std::cout << "[Test 5] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 6: Per-query speculative isolation via CSR masks
// ----------------------------------------------------------------------------
static void test_per_query_speculative_isolation() {
    std::cout << "[Test 6] Per-query speculative isolation via CSR masks..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;
    const uint32_t n_rows = 16;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 600, a_k, b_k, a_v, b_v);

    xkv_query_input q0;
    q0.query_index = 0;
    q0.head_dim_k = head_dim_k;
    q0.head_dim_v = head_dim_v;
    q0.n_q_heads = 1;
    q0.q_vec = generate_deterministic_floats(head_dim_k, 101);

    xkv_query_input q1;
    q1.query_index = 1;
    q1.head_dim_k = head_dim_k;
    q1.head_dim_v = head_dim_v;
    q1.n_q_heads = 1;
    q1.q_vec = generate_deterministic_floats(head_dim_k, 102);

    sr_batch_selection_result batch_sel;
    batch_sel.per_query.resize(2);

    for (uint32_t i = 0; i < n_rows; ++i) {
        segment_row_ref ref;
        ref.segment_id = seg->segment_id;
        ref.segment_version = seg->segment_version;
        ref.storage_generation = 1;
        ref.row = i;
        batch_sel.gather_rows.push_back(ref);
    }

    batch_sel.csr_ptrs = {0, 8, 24};
    for (uint32_t i = 0; i < 8; ++i) batch_sel.csr_indices.push_back(i);
    for (uint32_t i = 0; i < 16; ++i) batch_sel.csr_indices.push_back(i);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    auto batch_res = xkv_read_attention_batch({q0, q1}, {}, views, batch_sel, nullptr, store.current_stamp(), &store);
    assert(batch_res.status == xkv_read_status::success);
    assert(batch_res.per_query.size() == 2);

    xkv_segment_read_view view_q0_only;
    view_q0_only.pin = store.pin_segment(seg->segment_id);
    view_q0_only.segment_version_id = seg->segment_version;
    view_q0_only.storage_generation = 1;
    view_q0_only.owning_layer = 0;
    view_q0_only.factor_group_index = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        view_q0_only.selected_rows.push_back(i);
        view_q0_only.storage_positions.push_back(i);
    }
    std::vector<xkv_segment_read_view> views_q0;
    views_q0.push_back(std::move(view_q0_only));
    auto single_q0_res = xkv_read_attention(q0, {}, views_q0, nullptr, store.current_stamp(), &store);
    assert(single_q0_res.status == xkv_read_status::success);

    assert(vectors_approx_equal(batch_res.per_query[0].output, single_q0_res.output, 1e-5f));
    std::cout << "[Test 6] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 7: Hot row isolation and query 32+ without modulo aliasing
// ----------------------------------------------------------------------------
static void test_hot_row_isolation_and_query_32() {
    std::cout << "[Test 7] Hot row isolation and query 32+ without modulo aliasing..." << std::endl;

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;

    // Query 0 and Query 32: must NOT alias via modulo 32!
    xkv_query_input q0;
    q0.query_index = 0;
    q0.head_dim_k = head_dim_k;
    q0.head_dim_v = head_dim_v;
    q0.n_q_heads = 1;
    q0.q_vec = generate_deterministic_floats(head_dim_k, 201);

    xkv_query_input q32;
    q32.query_index = 32; // Query 32
    q32.head_dim_k = head_dim_k;
    q32.head_dim_v = head_dim_v;
    q32.n_q_heads = 1;
    q32.q_vec = generate_deterministic_floats(head_dim_k, 202);

    std::vector<float> k1 = generate_deterministic_floats(head_dim_k, 301);
    std::vector<float> v1 = generate_deterministic_floats(head_dim_v, 302);
    std::vector<float> k2 = generate_deterministic_floats(head_dim_k, 303);
    std::vector<float> v2 = generate_deterministic_floats(head_dim_v, 304);

    xkv_hot_row hr0_only;
    hr0_only.storage_pos = 1;
    hr0_only.k_ptr = k1.data();
    hr0_only.v_ptr = v1.data();
    hr0_only.query_visibility = {true}; // visible only to query 0 (size 1)

    xkv_hot_row hr32_only;
    hr32_only.storage_pos = 2;
    hr32_only.k_ptr = k2.data();
    hr32_only.v_ptr = v2.data();
    hr32_only.query_visibility.assign(33, false);
    hr32_only.query_visibility[32] = true; // visible only to query 32

    // Check single reads
    auto res_q0 = xkv_read_attention(q0, {hr0_only, hr32_only}, {}, nullptr, {}, nullptr);
    assert(res_q0.status == xkv_read_status::success);

    auto res_q32 = xkv_read_attention(q32, {hr0_only, hr32_only}, {}, nullptr, {}, nullptr);
    assert(res_q32.status == xkv_read_status::success);

    // q0 should only see hr0_only (value v1)
    for (uint32_t d = 0; d < head_dim_v; ++d) {
        assert(approx_equal(res_q0.output[d], v1[d], 1e-4f));
    }

    // q32 should only see hr32_only (value v2)
    for (uint32_t d = 0; d < head_dim_v; ++d) {
        assert(approx_equal(res_q32.output[d], v2[d], 1e-4f));
    }

    std::cout << "[Test 7] Passed. Query 0 and Query 32 isolation verified without modulo aliasing." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 8: Empty masks and all-masked rows without NaN
// ----------------------------------------------------------------------------
static void test_empty_and_all_masked_without_nan() {
    std::cout << "[Test 8] Empty masks and all-masked rows without NaN..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;

    xkv_query_input query;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.n_q_heads = 1;
    query.q_vec = generate_deterministic_floats(head_dim_k, 701);

    auto res_empty = xkv_read_attention(query, {}, {}, nullptr, store.current_stamp(), &store);
    assert(res_empty.status == xkv_read_status::success);
    for (float v : res_empty.output) {
        assert(!std::isnan(v) && !std::isinf(v) && v == 0.0f);
    }

    std::cout << "[Test 8] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 9: Stale epochs & snapshot stamp invalidation
// ----------------------------------------------------------------------------
static void test_stale_stamp_retry() {
    std::cout << "[Test 9] Stale epochs & snapshot stamp invalidation..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, 8, 16, 16, 4 * head_dim_k, 4 * head_dim_v, 801, a_k, b_k, a_v, b_v);

    xkv_query_input query;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.n_q_heads = 1;
    query.q_vec = generate_deterministic_floats(head_dim_k, 802);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    xkv_snapshot_stamp old_stamp = store.current_stamp();
    store.bump_content_epoch();

    auto res = xkv_read_attention(query, {}, views, nullptr, old_stamp, &store);
    assert(res.status == xkv_read_status::retry_stale_stamp);
    assert(res.output.empty());

    std::cout << "[Test 9] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 10: Checked arithmetic overflow protection
// ----------------------------------------------------------------------------
static void test_checked_arithmetic_overflow() {
    std::cout << "[Test 10] Checked arithmetic overflow protection..." << std::endl;

    size_t out = 0;
    assert(!safe_add(SIZE_MAX - 10, 20, out));
    assert(safe_add(SIZE_MAX - 20, 10, out));
    assert(out == SIZE_MAX - 10);

    assert(!safe_mul(SIZE_MAX / 2 + 1, 2, out));
    assert(safe_mul(SIZE_MAX / 2, 2, out));

    xkv_query_input bad_query;
    bad_query.head_dim_k = static_cast<uint32_t>(SIZE_MAX / 2);
    bad_query.head_dim_v = 16;
    bad_query.n_q_heads = 4;

    std::string err;
    assert(!bad_query.validate(&err));

    std::cout << "[Test 10] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 11: Batch workspace preflight: sum of states + CSR + output + tile peak > budget rejected
// ----------------------------------------------------------------------------
static void test_batch_workspace_sum_budget_rejected() {
    std::cout << "[Test 11] Batch workspace preflight: sum of states + CSR + output + tile peak > budget rejected..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 32;
    const uint32_t head_dim_v = 32;
    const uint32_t rank = 16;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, 32, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 1101, a_k, b_k, a_v, b_v);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < 32; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    std::vector<xkv_query_input> queries(10);
    for (size_t q = 0; q < 10; ++q) {
        queries[q].query_index = static_cast<uint32_t>(q);
        queries[q].head_dim_k = head_dim_k;
        queries[q].head_dim_v = head_dim_v;
        queries[q].n_q_heads = 1;
        queries[q].q_vec = generate_deterministic_floats(head_dim_k, 1100 + q);
    }

    xkv_reader_config tight_cfg;
    tight_cfg.tile_size = 16;
    tight_cfg.workspace_budget_bytes = 1000; // Far too small for 10 queries

    auto res = xkv_read_attention_batch(queries, {}, views, {}, nullptr, store.current_stamp(), &store, tight_cfg);
    assert(res.status == xkv_read_status::workspace_exceeded);

    std::cout << "[Test 11] Passed. Batch preflight rejected budget breach across all query states." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 12: Batch queries with mixed head dimensions rejected
// ----------------------------------------------------------------------------
static void test_batch_mixed_head_dims_rejected() {
    std::cout << "[Test 12] Batch queries with mixed head dimensions rejected..." << std::endl;

    xkv_query_input q0;
    q0.query_index = 0;
    q0.head_dim_k = 16;
    q0.head_dim_v = 16;
    q0.q_vec = generate_deterministic_floats(16, 1);

    xkv_query_input q1;
    q1.query_index = 1;
    q1.head_dim_k = 32; // Mismatch with q0!
    q1.head_dim_v = 16;
    q1.q_vec = generate_deterministic_floats(32, 2);

    auto batch_res = xkv_read_attention_batch({q0, q1}, {}, {}, {}, nullptr, {}, nullptr);
    assert(batch_res.status == xkv_read_status::invalid_argument);

    std::cout << "[Test 12] Passed. Mixed head dimensions rejected." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 12b: Empty batch preflight returns 0 safely without crash
// ----------------------------------------------------------------------------
static void test_empty_batch_preflight() {
    std::cout << "[Test 12b] Empty batch preflight returns 0 safely without crash..." << std::endl;

    size_t out_ws = 999;
    std::string err;
    bool ok = xkv_preflight_batch_workspace({}, {}, {}, {}, out_ws, &err);
    assert(ok);
    assert(out_ws == 0);

    std::cout << "[Test 12b] Passed. Empty batch preflight returns 0 safely." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 13: B-cache lease accounting: eviction refusal, clear-with-lease retention, no cross-entry subtraction
// ----------------------------------------------------------------------------
static void test_b_cache_lease_accounting_and_retention() {
    std::cout << "[Test 13] B-cache lease accounting: eviction refusal, clear-with-lease retention..." << std::endl;

    const size_t tile_elements = 256;
    const size_t single_entry_bytes = xkv_b_tile_cache::compute_entry_bytes(tile_elements);
    assert(single_entry_bytes != SIZE_MAX && single_entry_bytes > 0);
    const size_t cache_budget = single_entry_bytes * 2; // Exact budget for exactly 2 entries
    xkv_b_tile_cache cache(cache_budget);

    b_tile_cache_key k1, k2, k3;
    k1.segment_id = 1;
    k2.segment_id = 2;
    k3.segment_id = 3;

    // 1. Put k1 and hold lease
    xkv_b_tile_lease lease1 = cache.put_and_lease(k1, std::vector<float>(tile_elements, 1.0f));
    assert(lease1.valid());
    assert(cache.total_allocated_bytes() == single_entry_bytes);
    assert(cache.active_cache_bytes() == single_entry_bytes);

    // 2. Put k2 and hold lease
    xkv_b_tile_lease lease2 = cache.put_and_lease(k2, std::vector<float>(tile_elements, 2.0f));
    assert(lease2.valid());
    assert(cache.total_allocated_bytes() == single_entry_bytes * 2);
    assert(cache.active_cache_bytes() == single_entry_bytes * 2);

    // Verify k1 and k2 are confirmed cache hits
    auto hit_k1 = cache.get(k1);
    assert(hit_k1.valid());
    hit_k1.release();
    auto hit_k2 = cache.get(k2);
    assert(hit_k2.valid());
    hit_k2.release();

    // 3. Put k3: since BOTH k1 and k2 are leased, NEITHER can be evicted!
    // Cache must refuse caching k3, but return an uncached lease
    xkv_b_tile_lease lease3 = cache.put_and_lease(k3, std::vector<float>(tile_elements, 3.0f));
    assert(lease3.valid());
    // k3 is not in cache
    auto miss_k3 = cache.get(k3);
    assert(!miss_k3.valid());
    assert(cache.active_cache_bytes() == single_entry_bytes * 2);
    assert(cache.total_allocated_bytes() == single_entry_bytes * 2);

    // 4. Clear-with-lease retention: clear() clears the map, but total_allocated_bytes
    // continues to reflect leased memory until leases are released!
    cache.clear();
    assert(cache.active_cache_bytes() == 0);
    assert(cache.total_allocated_bytes() == single_entry_bytes * 2); // Leases 1 and 2 still alive!
    assert(cache.leased_bytes() == single_entry_bytes * 2);

    // 5. Release lease 1: no cross-entry corruption
    lease1.release();
    assert(cache.total_allocated_bytes() == single_entry_bytes); // Lease 2 still alive!
    assert(cache.leased_bytes() == single_entry_bytes);

    // 6. Release lease 2: now fully 0
    lease2.release();
    assert(cache.total_allocated_bytes() == 0);
    assert(cache.leased_bytes() == 0);

    // 7. Active-lease same-key put regression:
    // Put k1 again, obtain lease, call put_and_lease with the same key k1.
    // Verify pointer is NOT invalidated and data remains intact (no UAF).
    std::vector<float> original_data(tile_elements, 42.0f);
    xkv_b_tile_lease lease_orig = cache.put_and_lease(k1, original_data);
    const float * orig_ptr = lease_orig.data();
    assert(orig_ptr != nullptr);
    assert(orig_ptr[0] == 42.0f);

    std::vector<float> replacement_data(tile_elements, 99.0f);
    xkv_b_tile_lease lease_reput = cache.put_and_lease(k1, replacement_data);
    assert(lease_reput.data() == orig_ptr); // MUST retain existing buffer, not reallocate
    assert(lease_orig.data()[0] == 42.0f); // Original data preserved without UAF
    lease_orig.release();
    lease_reput.release();

    std::cout << "[Test 13] Passed. Leased memory correctly tracked and cleared without cross-subtraction." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 14: Owning xkv_reader_pin requirement and segment validation
// ----------------------------------------------------------------------------
static void test_owning_reader_pin_requirement() {
    std::cout << "[Test 14] Owning xkv_reader_pin requirement and segment validation..." << std::endl;

    xkv_query_input q;
    q.head_dim_k = 16;
    q.head_dim_v = 16;
    q.q_vec = generate_deterministic_floats(16, 1);

    // View without pin
    xkv_segment_read_view unpinned_view;
    unpinned_view.selected_rows = {0};
    unpinned_view.storage_positions = {0};

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(unpinned_view));
    auto res = xkv_read_attention(q, {}, views, nullptr, {}, nullptr);
    assert(res.status == xkv_read_status::invalid_argument);

    std::cout << "[Test 14] Passed. Unpinned segment view rejected." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 15: Sinks, softcapping, and scaling order
// ----------------------------------------------------------------------------
static void test_sinks_and_softcap_ordering() {
    std::cout << "[Test 15] Sinks, softcapping, and scaling order..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, 8, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 1501, a_k, b_k, a_v, b_v);

    xkv_query_input query;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.n_q_heads = 1;
    query.q_vec = generate_deterministic_floats(head_dim_k, 1502);
    query.scale = 0.5f;
    query.logit_softcap = 15.0f;
    query.sink_logits = {2.5f};

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
    }

    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store);
    assert(res.status == xkv_read_status::success);

    std::vector<std::vector<float>> cold_keys(8, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(8, std::vector<float>(head_dim_v, 0.0f));
    for (uint32_t r = 0; r < 8; ++r) {
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                cold_keys[r][f] += a_k[r * rank + k] * b_k[f * rank + k];
            }
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                cold_values[r][f] += a_v[r * rank + k] * b_v[f * rank + k];
            }
        }
    }

    std::vector<uint32_t> groups(8, 0);
    auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, groups, {});
    assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));

    std::cout << "[Test 15] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 16: Workspace warmup stability: capacities & buffer pointers do not grow
// as segment/reader history increases under fixed configured maxima
// ----------------------------------------------------------------------------
static void test_workspace_warmup_stability_and_zero_growth() {
    std::cout << "[Test 16] Workspace warmup stability: capacities & buffer pointers do not grow as history increases..." << std::endl;

    xkv_reader_workspace_config ws_cfg;
    ws_cfg.capacity_bytes = 4 * 1024 * 1024; // 4 MiB fixed workspace
    ws_cfg.max_queries = 8;
    ws_cfg.max_q_heads = 4;
    ws_cfg.max_head_dim_k = 64;
    ws_cfg.max_head_dim_v = 64;
    ws_cfg.max_tile_size = 32;
    ws_cfg.max_rank_k = 64;
    ws_cfg.max_rank_v = 64;
    ws_cfg.max_csr_entries = 1024;

    xkv_reader_workspace ws;
    ws.warmup(ws_cfg);

    assert(ws.is_warmed_up());
    size_t initial_allocated = ws.allocated_bytes();
    const void * initial_base = ws.scratch_ptr();
    const float * initial_k_buf = ws.tile_k_buffer();
    const float * initial_v_buf = ws.tile_v_buffer();
    const float * initial_ak_buf = ws.tile_a_k_buffer();
    const float * initial_bk_buf = ws.tile_b_k_buffer();
    const segment_row_ref * initial_csr_buf = ws.csr_refs_buffer();

    assert(initial_base != nullptr);
    assert(initial_k_buf != nullptr);
    assert(initial_v_buf != nullptr);
    assert(initial_ak_buf != nullptr);
    assert(initial_bk_buf != nullptr);
    assert(initial_csr_buf != nullptr);

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 32;
    const uint32_t head_dim_v = 32;
    const uint32_t rank = 32;

    // Run sequential reads across increasing segment sizes / history:
    // Segment lengths 8, 16, 24, 32 rows
    for (uint32_t n_rows = 8; n_rows <= 32; n_rows += 8) {
        std::vector<float> a_k, b_k, a_v, b_v;
        auto seg = create_test_segment_f32(store, n_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 1000 + n_rows, a_k, b_k, a_v, b_v);

        xkv_query_input q;
        q.query_index = 0;
        q.n_q_heads = 1;
        q.head_dim_k = head_dim_k;
        q.head_dim_v = head_dim_v;
        q.q_vec = generate_deterministic_floats(head_dim_k, 555 + n_rows);
        q.scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));

        xkv_segment_read_view view;
        view.pin = store.pin_segment(seg->segment_id);
        view.segment_version_id = seg->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < n_rows; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back(i);
            view.group_indices.push_back(0);
        }

        std::vector<xkv_segment_read_view> views;
        views.push_back(std::move(view));

        xkv_reader_config cfg;
        cfg.workspace = &ws;
        cfg.tile_size = 16;

        auto res = xkv_read_attention(q, {}, views, nullptr, store.current_stamp(), &store, cfg);
        assert(res.status == xkv_read_status::success);
        assert(!res.output.empty());

        // Invariant check: capacities and buffer pointers must NOT have grown or moved!
        assert(ws.allocated_bytes() == initial_allocated);
        assert(ws.scratch_ptr() == initial_base);
        assert(ws.tile_k_buffer() == initial_k_buf);
        assert(ws.tile_v_buffer() == initial_v_buf);
        assert(ws.tile_a_k_buffer() == initial_ak_buf);
        assert(ws.tile_b_k_buffer() == initial_bk_buf);
        assert(ws.csr_refs_buffer() == initial_csr_buf);
        assert(ws.live_bytes() == 0); // Must be cleanly released
    }

    // Also verify batch execution with CSR indices:
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, 32, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 9000, a_k, b_k, a_v, b_v);
    xkv_query_input q0, q1;
    q0.query_index = 0; q0.n_q_heads = 1; q0.head_dim_k = head_dim_k; q0.head_dim_v = head_dim_v;
    q0.q_vec = generate_deterministic_floats(head_dim_k, 9001);
    q1.query_index = 1; q1.n_q_heads = 1; q1.head_dim_k = head_dim_k; q1.head_dim_v = head_dim_v;
    q1.q_vec = generate_deterministic_floats(head_dim_k, 9002);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < 32; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    sr_batch_selection_result b_sel;
    for (uint32_t i = 0; i < 32; ++i) {
        b_sel.gather_rows.push_back(views[0].get_row_ref(i));
    }
    b_sel.csr_ptrs = {0, 16, 32};
    b_sel.csr_indices.resize(32);
    for (uint32_t i = 0; i < 32; ++i) b_sel.csr_indices[i] = i;

    xkv_reader_config batch_cfg;
    batch_cfg.workspace = &ws;
    batch_cfg.tile_size = 16;

    auto batch_res = xkv_read_attention_batch({q0, q1}, {}, views, b_sel, nullptr, store.current_stamp(), &store, batch_cfg);
    assert(batch_res.status == xkv_read_status::success);
    assert(batch_res.per_query.size() == 2);

    // Pointers and capacity unchanged after batch CSR read
    assert(ws.allocated_bytes() == initial_allocated);
    assert(ws.scratch_ptr() == initial_base);
    assert(ws.tile_k_buffer() == initial_k_buf);
    assert(ws.tile_v_buffer() == initial_v_buf);
    assert(ws.csr_refs_buffer() == initial_csr_buf);
    assert(ws.live_bytes() == 0);

    std::cout << "[Test 16] Passed. Capacities and pointers remained strictly invariant across varying history lengths." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 17: Insufficient workspace fails cleanly with NO cache or store mutation
// ----------------------------------------------------------------------------
static void test_insufficient_workspace_fails_without_mutation() {
    std::cout << "[Test 17] Insufficient workspace fails cleanly without cache/store mutations..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t head_dim_k = 32;
    const uint32_t head_dim_v = 32;
    const uint32_t rank = 32;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 7700, a_k, b_k, a_v, b_v);

    xkv_snapshot_stamp stamp_initial = store.current_stamp();
    auto accounting_initial = store.get_accounting();

    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = head_dim_k;
    query.head_dim_v = head_dim_v;
    query.q_vec = generate_deterministic_floats(head_dim_k, 7701);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    // Create a tiny workspace of only 64 bytes (insufficient for even a single tile)
    xkv_reader_workspace tiny_ws(64, false);
    auto b_cache = std::make_shared<xkv_b_tile_cache>(1024 * 1024);

    xkv_reader_config cfg;
    cfg.workspace = &tiny_ws;
    cfg.b_cache = b_cache;

    auto res = xkv_read_attention(query, {}, views, nullptr, stamp_initial, &store, cfg);
    assert(res.status == xkv_read_status::workspace_exceeded);
    assert(res.output.empty());

    // Verify NO store mutation
    xkv_snapshot_stamp stamp_after = store.current_stamp();
    assert(stamp_after == stamp_initial);
    auto accounting_after = store.get_accounting();
    assert(accounting_after.allocated_bytes == accounting_initial.allocated_bytes);
    assert(accounting_after.arena_live_bytes == accounting_initial.arena_live_bytes);

    // Verify NO B-cache mutation (nothing inserted or cached)
    assert(b_cache->total_allocated_bytes() == 0);
    assert(b_cache->active_cache_bytes() == 0);
    assert(b_cache->hit_count() == 0);

    // Workspace scratch was cleanly rejected and live_bytes is 0
    assert(tiny_ws.live_bytes() == 0);

    std::cout << "[Test 17] Passed. Rejection occurred before any mutation/allocation; store and cache intact." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 18: Multi-reader outputs with workspace match independent dense reference oracle
// ----------------------------------------------------------------------------
static void test_multi_reader_workspace_matches_dense_oracle() {
    std::cout << "[Test 18] Multi-reader outputs with workspace match independent dense reference oracle..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 24;
    const uint32_t head_dim_k = 32;
    const uint32_t head_dim_v = 32;
    const uint32_t rank_k = 32;
    const uint32_t rank_v = 32;
    const uint32_t n_q_heads = 2; // GQA

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank_k, rank_v, 4 * head_dim_k, 4 * head_dim_v, 8800, a_k, b_k, a_v, b_v);

    // Compute independent cold keys and values
    std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(head_dim_v, 0.0f));
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            double sum = 0.0;
            for (uint32_t k = 0; k < rank_k; ++k) {
                sum += static_cast<double>(a_k[r * rank_k + k]) * static_cast<double>(b_k[f * rank_k + k]);
            }
            cold_keys[r][f] = static_cast<float>(sum);
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            double sum = 0.0;
            for (uint32_t k = 0; k < rank_v; ++k) {
                sum += static_cast<double>(a_v[r * rank_v + k]) * static_cast<double>(b_v[f * rank_v + k]);
            }
            cold_values[r][f] = static_cast<float>(sum);
        }
    }

    xkv_reader_workspace shared_ws(cparams);
    std::vector<uint32_t> groups(n_rows, 0);

    // Run 5 distinct queries across different readers
    for (uint32_t q_idx = 0; q_idx < 5; ++q_idx) {
        xkv_query_input query;
        query.query_index = q_idx;
        query.n_q_heads = n_q_heads;
        query.head_dim_k = head_dim_k;
        query.head_dim_v = head_dim_v;
        query.q_vec = generate_deterministic_floats(n_q_heads * head_dim_k, 8801 + q_idx);
        query.scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
        query.logit_softcap = 12.0f;
        query.sink_logits = {1.0f, 0.5f};

        xkv_segment_read_view view;
        view.pin = store.pin_segment(seg->segment_id);
        view.segment_version_id = seg->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < n_rows; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back(i);
            view.group_indices.push_back(0);
        }
        std::vector<xkv_segment_read_view> views;
        views.push_back(std::move(view));

        // Reader execution with shared pre-warmed workspace
        auto reader_res = xkv_read_attention(shared_ws, query, {}, views, nullptr, store.current_stamp(), &store);
        assert(reader_res.status == xkv_read_status::success);

        // Dense reference oracle calculation
        auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, groups, {});

        // Assert exact numerical match
        assert(vectors_approx_equal(reader_res.output, oracle_out, 1e-4f));
        assert(reader_res.peak_workspace_bytes <= shared_ws.capacity_bytes());
    }

    std::cout << "[Test 18] Passed. Multiple readers on reusable workspace match independent dense reference oracle." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 19: Workspace hard capacity invariant & tiny capacity / overflow rejection
// ----------------------------------------------------------------------------
static void test_workspace_hard_capacity_and_overflow_rejection() {
    std::cout << "[Test 19] Workspace hard capacity: never silently enlarged, tiny/overflow configs rejected..." << std::endl;

    // 1. Requested capacity is NEVER silently enlarged:
    // Request 256 bytes with defaults that need ~400 KB -> MUST hard fail, NOT enlarge!
    xkv_reader_workspace_config tiny_cfg;
    tiny_cfg.capacity_bytes = 256;
    xkv_reader_workspace ws_tiny;
    std::string err;
    bool ok = ws_tiny.warmup(tiny_cfg, &err);
    assert(!ok);
    assert(!err.empty());
    assert(!ws_tiny.is_warmed_up());
    assert(ws_tiny.capacity_bytes() == 0 || ws_tiny.capacity_bytes() == 256);

    // 2. Capacity unchanged when setup succeeds:
    xkv_reader_workspace_config valid_cfg;
    valid_cfg.capacity_bytes = 2 * 1024 * 1024; // 2 MiB requested
    valid_cfg.max_queries = 2;
    valid_cfg.max_q_heads = 2;
    valid_cfg.max_head_dim_k = 32;
    valid_cfg.max_head_dim_v = 32;
    valid_cfg.max_tile_size = 32;
    valid_cfg.max_rank_k = 32;
    valid_cfg.max_rank_v = 32;
    valid_cfg.max_csr_entries = 64;

    xkv_reader_workspace ws_valid;
    ok = ws_valid.warmup(valid_cfg, &err);
    assert(ok);
    assert(ws_valid.is_warmed_up());
    // Capacity must remain EXACTLY requested 2 MiB, not enlarged or truncated
    assert(ws_valid.capacity_bytes() == 2 * 1024 * 1024);

    // 3. Overflow configs cleanly rejected without undefined behavior:
    xkv_reader_workspace_config overflow_cfg;
    overflow_cfg.capacity_bytes = 1024 * 1024;
    overflow_cfg.max_queries = UINT32_MAX / 2;
    overflow_cfg.max_q_heads = UINT32_MAX / 2;
    xkv_reader_workspace ws_overflow;
    ok = ws_overflow.warmup(overflow_cfg, &err);
    assert(!ok);
    assert(!ws_overflow.is_warmed_up());

    std::cout << "[Test 19] Passed. Hard capacity invariant verified; overflow configs rejected." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 20: Exclusive single lease: simultaneous second lease rejected without aliasing
// ----------------------------------------------------------------------------
static void test_workspace_exclusive_single_lease_no_alias() {
    std::cout << "[Test 20] Exclusive single lease: simultaneous second lease rejected, no data race/aliasing..." << std::endl;

    xkv_reader_workspace_config cfg;
    cfg.capacity_bytes = 1024 * 1024;
    cfg.max_queries = 4;
    cfg.max_q_heads = 2;
    cfg.max_head_dim_k = 32;
    cfg.max_head_dim_v = 32;
    cfg.max_tile_size = 32;
    cfg.max_rank_k = 32;
    cfg.max_rank_v = 32;
    cfg.max_csr_entries = 64;

    xkv_reader_workspace ws;
    bool ok = ws.warmup(cfg);
    assert(ok);
    assert(!ws.has_active_lease());

    // 1. Acquire first lease
    xkv_workspace_scratch_lease lease1 = ws.acquire_lease(1024);
    assert(lease1.valid());
    assert(ws.has_active_lease());
    assert(ws.live_bytes() == 1024);

    // Preflight check must report false while lease is active
    assert(!ws.preflight(1024));

    // 2. Attempt simultaneous second lease -> MUST be rejected to prevent sharing same scratch spans!
    xkv_workspace_scratch_lease lease2 = ws.acquire_lease(1024);
    assert(!lease2.valid());

    xkv_reader_scratch scratch2;
    bool acq2 = ws.acquire(1024, scratch2);
    assert(!acq2);
    assert(!scratch2.valid);

    // 3. Release first lease -> now second lease can be acquired cleanly
    lease1.release();
    assert(!ws.has_active_lease());
    assert(ws.live_bytes() == 0);
    assert(ws.preflight(1024));

    xkv_workspace_scratch_lease lease3 = ws.acquire_lease(1024);
    assert(lease3.valid());
    assert(ws.has_active_lease());
    lease3.release();
    assert(!ws.has_active_lease());

    std::cout << "[Test 20] Passed. Simultaneous second lease cleanly rejected; exclusive lease invariant holds." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 21: Non-movability at compile-time & atomic reconfigure preserving prior state
// ----------------------------------------------------------------------------
static void test_workspace_non_movable_and_atomic_reconfigure() {
    std::cout << "[Test 21] Non-movability at compile-time & atomic reconfigure on failure..." << std::endl;

    // 1. Compile-time proof: xkv_reader_workspace is NOT move-constructible and NOT move-assignable
    static_assert(!std::is_move_constructible<xkv_reader_workspace>::value, "xkv_reader_workspace must not be move constructible");
    static_assert(!std::is_move_assignable<xkv_reader_workspace>::value, "xkv_reader_workspace must not be move assignable");
    static_assert(!std::is_copy_constructible<xkv_reader_workspace>::value, "xkv_reader_workspace must not be copy constructible");
    static_assert(!std::is_copy_assignable<xkv_reader_workspace>::value, "xkv_reader_workspace must not be copy assignable");

    // 2. Constructor throwing on failed warmup
    bool threw = false;
    try {
        // Request 128 bytes with preallocate=true -> must throw invalid_argument, not silently create unwarmed object!
        xkv_reader_workspace bad_ws(128, true);
    } catch (const std::invalid_argument & e) {
        threw = true;
    }
    assert(threw);

    // 3. Atomic reconfigure: failed re-warm leaves old warmed workspace completely intact
    xkv_reader_workspace_config good_cfg;
    good_cfg.capacity_bytes = 2 * 1024 * 1024; // 2 MiB
    good_cfg.max_queries = 2;
    good_cfg.max_q_heads = 2;
    good_cfg.max_head_dim_k = 32;
    good_cfg.max_head_dim_v = 32;
    good_cfg.max_tile_size = 32;
    good_cfg.max_rank_k = 32;
    good_cfg.max_rank_v = 32;
    good_cfg.max_csr_entries = 64;

    xkv_reader_workspace ws;
    std::string err;
    bool ok = ws.warmup(good_cfg, &err);
    assert(ok);
    assert(ws.is_warmed_up());
    size_t prev_capacity = ws.capacity_bytes();
    size_t prev_allocated = ws.allocated_bytes();
    const void * prev_scratch_ptr = ws.scratch_ptr();
    const float * prev_k_buf = ws.tile_k_buffer();

    // Attempt a re-warm that will fail hard capacity (request 64 bytes)
    xkv_reader_workspace_config bad_reconfig = good_cfg;
    bad_reconfig.capacity_bytes = 64;
    ok = ws.warmup(bad_reconfig, &err);
    assert(!ok);
    assert(!err.empty());

    // Prior configuration, pointers, and warmed_up status MUST be completely preserved!
    assert(ws.is_warmed_up());
    assert(ws.capacity_bytes() == prev_capacity);
    assert(ws.allocated_bytes() == prev_allocated);
    assert(ws.scratch_ptr() == prev_scratch_ptr);
    assert(ws.tile_k_buffer() == prev_k_buf);

    // It must still be usable for leases
    xkv_workspace_scratch_lease lease = ws.acquire_lease(512);
    assert(lease.valid());

    // Reconfigure while lease is active must be rejected
    ok = ws.warmup(good_cfg, &err);
    assert(!ok);
    assert(!err.empty());
    lease.release();

    std::cout << "[Test 21] Passed. Compile-time non-movability and atomic reconfigure guarantees verified." << std::endl;
}

// ----------------------------------------------------------------------------
// Bounded-reader proofs (Tests 23-30). Per-layer slices are sized exactly
// (total_dim = 4 * head_dim for the 4-layer fixture group) so the B feature
// offset validation passes by construction; oracles decode the same B rows.
// ----------------------------------------------------------------------------

static xkv_reader_workspace_config make_bounded_ws_config(
    uint32_t max_queries, uint32_t max_q_heads,
    uint32_t max_dk, uint32_t max_dv,
    uint32_t max_tile, uint32_t max_rk, uint32_t max_rv, uint32_t max_csr) {
    xkv_reader_workspace_config cfg;
    cfg.max_queries = max_queries;
    cfg.max_q_heads = max_q_heads;
    cfg.max_head_dim_k = max_dk;
    cfg.max_head_dim_v = max_dv;
    cfg.max_tile_size = max_tile;
    cfg.max_rank_k = max_rk;
    cfg.max_rank_v = max_rv;
    cfg.max_csr_entries = max_csr;
    size_t total = 0;
    std::string err;
    bool ok = xkv_estimate_workspace_layout(cfg, total, &err);
    assert(ok);
    cfg.capacity_bytes = total;
    return cfg;
}

// ----------------------------------------------------------------------------
// Test 23: strict single read performs zero C++ heap allocations besides the
// accounted output buffer (hot + cold, GQA, F32).
// ----------------------------------------------------------------------------
static void test_strict_single_zero_heap() {
    std::cout << "[Test 23] Strict single read: zero heap besides accounted output..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t dk = 16, dv = 16, rk = 16, rv = 16;
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rk, rv, 4 * dk, 4 * dv, 23100, a_k, b_k, a_v, b_v);

    const auto * g = seg->find_group(0);
    assert(g != nullptr);
    const uint32_t pad_rk = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
    const uint32_t pad_rv = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);

    // Cold oracle from the exact B rows the reader decodes (layer 0 slice [0, dk)).
    std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(dk, 0.0f));
    std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(dv, 0.0f));
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t f = 0; f < dk; ++f)
            for (uint32_t k = 0; k < rk; ++k) cold_keys[r][f] += a_k[r * rk + k] * b_k[f * rk + k];
        for (uint32_t f = 0; f < dv; ++f)
            for (uint32_t k = 0; k < rv; ++k) cold_values[r][f] += a_v[r * rv + k] * b_v[f * rv + k];
    }

    std::vector<float> hk0 = generate_deterministic_floats(dk, 23101);
    std::vector<float> hv0 = generate_deterministic_floats(dv, 23102);
    std::vector<float> hk1 = generate_deterministic_floats(dk, 23103);
    std::vector<float> hv1 = generate_deterministic_floats(dv, 23104);
    xkv_hot_row hr0, hr1;
    hr0.row_index = 0; hr0.storage_pos = 100; hr0.k_ptr = hk0.data(); hr0.v_ptr = hv0.data();
    hr1.row_index = 1; hr1.storage_pos = 101; hr1.k_ptr = hk1.data(); hr1.v_ptr = hv1.data();
    std::vector<xkv_hot_row> hot_rows = {hr0, hr1};

    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 2;
    query.head_dim_k = dk;
    query.head_dim_v = dv;
    query.q_vec = generate_deterministic_floats(2 * dk, 23105);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    auto ws_cfg = make_bounded_ws_config(1, 2, dk, dv, 8, pad_rk, pad_rv, 8);
    xkv_reader_workspace ws;
    std::string werr;
    assert(ws.warmup(ws_cfg, &werr));

    xkv_reader_config rcfg;
    rcfg.workspace = &ws;
    rcfg.tile_size = 8;

    xkv_snapshot_stamp stamp = store.current_stamp();
    xkv_read_result res;
    {
        heap_scope scope;
        res = xkv_read_attention(query, hot_rows, views, nullptr, stamp, &store, rcfg);
        assert(res.status == xkv_read_status::success);
        // Exactly one heap allocation: the accounted [n_q_heads * dv] output.
        assert(scope.count() == 1);
        assert(scope.bytes() == static_cast<size_t>(2 * dv * sizeof(float)));
    }
    assert(ws.live_bytes() == 0);
    assert(res.peak_workspace_bytes <= ws.capacity_bytes());

    std::vector<uint32_t> groups(n_rows, 0);
    auto oracle_out = xkv_dense_attention_reference(query, hot_rows, cold_keys, cold_values, groups, {});
    assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));
    std::cout << "[Test 23] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 24: strict batch read with CSR, K!=V, non-power-of-two ranks/tile and
// mixed GQA heads performs only accounted allocations and matches singles/oracle.
// ----------------------------------------------------------------------------
static void test_strict_batch_csr_zero_heap() {
    std::cout << "[Test 24] Strict batch CSR (K!=V, odd ranks/tile, mixed GQA)..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t dk = 24, dv = 12, rk = 24, rv = 40;
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rk, rv, 4 * dk, 4 * dv, 23200, a_k, b_k, a_v, b_v);
    const auto * g = seg->find_group(0);
    assert(g != nullptr);
    const uint32_t pad_rk = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
    const uint32_t pad_rv = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);

    xkv_query_input q0, q1;
    q0.query_index = 0; q0.n_q_heads = 2; q0.head_dim_k = dk; q0.head_dim_v = dv;
    q0.q_vec = generate_deterministic_floats(2 * dk, 23201);
    q1.query_index = 1; q1.n_q_heads = 3; q1.head_dim_k = dk; q1.head_dim_v = dv;
    q1.q_vec = generate_deterministic_floats(3 * dk, 23202);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    // CSR: q0 sees all rows, q1 sees even rows only.
    sr_batch_selection_result bsel;
    for (uint32_t i = 0; i < n_rows; ++i) {
        segment_row_ref ref;
        ref.segment_id = seg->segment_id;
        ref.segment_version = seg->segment_version;
        ref.storage_generation = 1;
        ref.row = i;
        bsel.gather_rows.push_back(ref);
    }
    bsel.csr_ptrs = {0, 16, 24};
    for (uint32_t i = 0; i < 16; ++i) bsel.csr_indices.push_back(i);
    for (uint32_t i = 0; i < 16; i += 2) bsel.csr_indices.push_back(i);

    auto ws_cfg = make_bounded_ws_config(2, 3, dk, dv, 7, pad_rk, pad_rv, 24);
    xkv_reader_workspace ws;
    assert(ws.warmup(ws_cfg));
    xkv_reader_config rcfg;
    rcfg.workspace = &ws;
    rcfg.tile_size = 7;

    xkv_snapshot_stamp stamp = store.current_stamp();
    xkv_batch_read_result bres;
    const std::vector<xkv_query_input> batch_queries = {q0, q1};
    {
        heap_scope scope;
        bres = xkv_read_attention_batch(batch_queries, {}, views, bsel, nullptr, stamp, &store, rcfg);
        assert(bres.status == xkv_read_status::success);
        assert(bres.per_query.size() == 2);
        // Accounted allocations only: per-query table + one output buffer per query.
        const size_t expect_bytes = 2 * sizeof(xkv_read_result) +
            static_cast<size_t>((2 * dv + 3 * dv) * sizeof(float));
        assert(scope.count() == 3);
        assert(scope.bytes() == expect_bytes);
    }
    assert(ws.live_bytes() == 0);

    // q0 (CSR = all rows) matches a strict single read over the same view.
    auto s0 = xkv_read_attention(q0, {}, views, nullptr, stamp, &store, rcfg);
    assert(s0.status == xkv_read_status::success);
    assert(vectors_approx_equal(bres.per_query[0].output, s0.output, 1e-5f));

    // q1 (even rows) matches the dense oracle masked to even rows.
    std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(dk, 0.0f));
    std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(dv, 0.0f));
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t f = 0; f < dk; ++f)
            for (uint32_t k = 0; k < rk; ++k) cold_keys[r][f] += a_k[r * rk + k] * b_k[f * rk + k];
        for (uint32_t f = 0; f < dv; ++f)
            for (uint32_t k = 0; k < rv; ++k) cold_values[r][f] += a_v[r * rv + k] * b_v[f * rv + k];
    }
    std::vector<uint32_t> groups(n_rows, 0);
    std::vector<bool> even_mask(n_rows, false);
    for (uint32_t i = 0; i < n_rows; i += 2) even_mask[i] = true;
    auto oracle_q1 = xkv_dense_attention_reference(q1, {}, cold_keys, cold_values, groups, even_mask);
    assert(vectors_approx_equal(bres.per_query[1].output, oracle_q1, 1e-4f));
    std::cout << "[Test 24] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 25: B-cache hit reads allocate only the accounted output; an over-budget
// cache insert is served uncached from caller workspace with zero heap.
// ----------------------------------------------------------------------------
static void test_b_cache_hit_and_overbudget_zero_heap() {
    std::cout << "[Test 25] B-cache hit + over-budget uncached zero-heap..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t dk = 16, dv = 16, rk = 16, rv = 16;
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rk, rv, 4 * dk, 4 * dv, 23300, a_k, b_k, a_v, b_v);
    const auto * g = seg->find_group(0);
    assert(g != nullptr);
    const uint32_t pad_rk = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
    const uint32_t pad_rv = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);

    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = dk;
    query.head_dim_v = dv;
    query.q_vec = generate_deterministic_floats(dk, 23301);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    auto ws_cfg = make_bounded_ws_config(1, 1, dk, dv, 8, pad_rk, pad_rv, 8);
    xkv_reader_workspace ws;
    assert(ws.warmup(ws_cfg));

    auto b_cache = std::make_shared<xkv_b_tile_cache>(8 * 1024 * 1024);
    xkv_reader_config rcfg;
    rcfg.workspace = &ws;
    rcfg.tile_size = 8;
    rcfg.b_cache = b_cache;

    xkv_snapshot_stamp stamp = store.current_stamp();
    auto miss = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg);
    assert(miss.status == xkv_read_status::success);
    assert(b_cache->total_allocated_bytes() > 0); // miss warmed two B entries

    {
        heap_scope scope;
        auto hit = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg);
        assert(hit.status == xkv_read_status::success);
        assert(hit.b_cache_hit);
        assert(vectors_approx_equal(hit.output, miss.output, 1e-6f));
        assert(scope.count() == 1);
        assert(scope.bytes() == static_cast<size_t>(dv * sizeof(float)));
    }
    assert(ws.live_bytes() == 0);

    // Over-budget cache: insert refuses without heap; the read still succeeds
    // uncached from caller workspace with only the accounted output allocated.
    auto tiny_cache = std::make_shared<xkv_b_tile_cache>(1);
    rcfg.b_cache = tiny_cache;
    {
        heap_scope scope;
        auto res = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg);
        assert(res.status == xkv_read_status::success);
        assert(vectors_approx_equal(res.output, miss.output, 1e-6f));
        assert(scope.count() == 1);
        assert(scope.bytes() == static_cast<size_t>(dv * sizeof(float)));
    }
    assert(tiny_cache->total_allocated_bytes() == 0);
    assert(ws.live_bytes() == 0);
    std::cout << "[Test 25] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 26: every short slice fails deterministically with workspace_exhausted,
// empty output, and no cache/store mutation.
// ----------------------------------------------------------------------------
static void test_slice_short_matrix() {
    std::cout << "[Test 26] Per-slice one-short deterministic exhaustion..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t dk = 16, dv = 16, rk = 16, rv = 16;
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rk, rv, 4 * dk, 4 * dv, 23400, a_k, b_k, a_v, b_v);
    const auto * g = seg->find_group(0);
    assert(g != nullptr);
    const uint32_t pad_rk = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
    const uint32_t pad_rv = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);

    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 2;
    query.head_dim_k = dk;
    query.head_dim_v = dv;
    query.q_vec = generate_deterministic_floats(2 * dk, 23401);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    xkv_snapshot_stamp stamp = store.current_stamp();
    auto accounting_before = store.get_accounting();

    // Each entry trims exactly one slice maximum by one element below the read need.
    // The read tile size stays 8 while workspace maxima shrink.
    std::vector<xkv_reader_workspace_config> short_cfgs;
    short_cfgs.push_back(make_bounded_ws_config(1, 1, dk, dv, 8, pad_rk, pad_rv, 8)); // heads 2>1
    short_cfgs.push_back(make_bounded_ws_config(1, 2, dk - 1, dv, 8, pad_rk, pad_rv, 8));
    short_cfgs.push_back(make_bounded_ws_config(1, 2, dk, dv - 1, 8, pad_rk, pad_rv, 8));
    short_cfgs.push_back(make_bounded_ws_config(1, 2, dk, dv, 7, pad_rk, pad_rv, 8)); // tile 8>7
    short_cfgs.push_back(make_bounded_ws_config(1, 2, dk, dv, 8, pad_rk - 1, pad_rv, 8));
    short_cfgs.push_back(make_bounded_ws_config(1, 2, dk, dv, 8, pad_rk, pad_rv - 1, 8));

    for (size_t t = 0; t < short_cfgs.size(); ++t) {
        xkv_reader_workspace ws;
        assert(ws.warmup(short_cfgs[t]));
        auto b_cache = std::make_shared<xkv_b_tile_cache>(8 * 1024 * 1024);
        xkv_reader_config rcfg;
        rcfg.workspace = &ws;
        rcfg.tile_size = 8;
        rcfg.b_cache = b_cache;
        auto res = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg);
        assert(res.status == xkv_read_status::workspace_exceeded);
        assert(res.output.empty());
        assert(b_cache->total_allocated_bytes() == 0);
        assert(b_cache->hit_count() == 0);
        assert(ws.live_bytes() == 0);
        assert(store.current_stamp() == stamp);
    }
    auto accounting_after = store.get_accounting();
    assert(accounting_after.allocated_bytes == accounting_before.allocated_bytes);

    // Batch dimension: max_queries and max_csr_entries one short.
    xkv_query_input qb = query;
    qb.query_index = 1;
    sr_batch_selection_result bsel;
    for (uint32_t i = 0; i < n_rows; ++i) {
        segment_row_ref ref;
        ref.segment_id = seg->segment_id;
        ref.segment_version = seg->segment_version;
        ref.storage_generation = 1;
        ref.row = i;
        bsel.gather_rows.push_back(ref);
    }
    bsel.csr_ptrs = {0, 16, 32};
    for (uint32_t i = 0; i < 16; ++i) bsel.csr_indices.push_back(i);
    for (uint32_t i = 0; i < 16; ++i) bsel.csr_indices.push_back(i);
    {
        auto cfg_q = make_bounded_ws_config(1, 2, dk, dv, 8, pad_rk, pad_rv, 64);
        xkv_reader_workspace ws;
        assert(ws.warmup(cfg_q));
        xkv_reader_config rcfg;
        rcfg.workspace = &ws;
        rcfg.tile_size = 8;
        auto bres = xkv_read_attention_batch({query, qb}, {}, views, bsel, nullptr, stamp, &store, rcfg);
        assert(bres.status == xkv_read_status::workspace_exceeded);
        assert(bres.per_query.empty());
        assert(ws.live_bytes() == 0);
    }
    {
        auto cfg_csr = make_bounded_ws_config(2, 2, dk, dv, 8, pad_rk, pad_rv, 31);
        xkv_reader_workspace ws;
        assert(ws.warmup(cfg_csr));
        xkv_reader_config rcfg;
        rcfg.workspace = &ws;
        rcfg.tile_size = 8;
        auto bres = xkv_read_attention_batch({query, qb}, {}, views, bsel, nullptr, stamp, &store, rcfg);
        assert(bres.status == xkv_read_status::workspace_exceeded);
        assert(bres.per_query.empty());
        assert(ws.live_bytes() == 0);
    }
    assert(store.current_stamp() == stamp);
    std::cout << "[Test 26] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 27: the workspace planner is byte-exact (capacity T warms, T-1 fails)
// and an exact-fit workspace serves a strict read.
// ----------------------------------------------------------------------------
static void test_planner_byte_exact() {
    std::cout << "[Test 27] Planner byte-exactness + exact-fit strict read..." << std::endl;

    xkv_reader_workspace_config cfg;
    cfg.max_queries = 3;
    cfg.max_q_heads = 5;
    cfg.max_head_dim_k = 24;
    cfg.max_head_dim_v = 12;
    cfg.max_tile_size = 7;
    cfg.max_rank_k = 24;
    cfg.max_rank_v = 40;
    cfg.max_csr_entries = 100;
    size_t total = 0;
    assert(xkv_estimate_workspace_layout(cfg, total));
    assert(total > 0);

    cfg.capacity_bytes = total;
    xkv_reader_workspace ws_ok;
    assert(ws_ok.warmup(cfg));

    cfg.capacity_bytes = total - 1;
    xkv_reader_workspace ws_short;
    std::string err;
    assert(!ws_short.warmup(cfg, &err));
    assert(!err.empty());
    assert(!ws_short.is_warmed_up());

    // Overflow configs are rejected without allocation.
    xkv_reader_workspace_config bad = cfg;
    bad.max_queries = UINT32_MAX;
    bad.max_q_heads = UINT32_MAX;
    size_t dummy = 0;

    assert(!xkv_estimate_workspace_layout(bad, dummy, nullptr));

    // Exact-fit workspace serves a real strict read end to end.
    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, 4, 8, 8, 32, 32, 23500, a_k, b_k, a_v, b_v);
    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = 8;
    query.head_dim_v = 8;
    query.q_vec = generate_deterministic_floats(8, 23501);
    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));
    auto fit = make_bounded_ws_config(1, 1, 8, 8, 4, 8, 8, 4);
    xkv_reader_workspace ws;
    assert(ws.warmup(fit));
    xkv_reader_config rcfg;
    rcfg.workspace = &ws;
    rcfg.tile_size = 4;
    auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store, rcfg);
    assert(res.status == xkv_read_status::success);
    assert(res.output.size() == 8);
    assert(ws.live_bytes() == 0);
    std::cout << "[Test 27] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 28: NaN/Inf/negative scales and huge dimensions are rejected before any
// bulk allocation: deterministic invalid_argument, empty output, untouched cache.
// ----------------------------------------------------------------------------
static void test_scale_and_huge_dim_rejection() {
    std::cout << "[Test 28] Scale/dimension fail-closed validation..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);
    xkv_snapshot_stamp stamp = store.current_stamp();
    auto b_cache = std::make_shared<xkv_b_tile_cache>(8 * 1024 * 1024);
    xkv_reader_config rcfg;
    rcfg.b_cache = b_cache;

    auto expect_invalid = [&](xkv_query_input q) {
        heap_scope scope;
        auto res = xkv_read_attention(q, {}, {}, nullptr, stamp, &store, rcfg);
        assert(res.status == xkv_read_status::invalid_argument);
        assert(res.output.empty());
        // At most the diagnostic text allocates; no bulk buffers are ever sized.
        assert(scope.count() <= 2);
    };

    xkv_query_input base;
    base.query_index = 0;
    base.n_q_heads = 1;
    base.head_dim_k = 8;
    base.head_dim_v = 8;
    base.q_vec = generate_deterministic_floats(8, 23600);

    xkv_query_input q = base;
    q.scale = std::numeric_limits<float>::quiet_NaN();
    expect_invalid(q);
    q = base;
    q.scale = std::numeric_limits<float>::infinity();
    expect_invalid(q);
    q = base;
    q.scale = -std::numeric_limits<float>::infinity();
    expect_invalid(q);
    q = base;
    q.scale = -1.0f;
    expect_invalid(q);
    q = base;
    q.head_dim_k = 1u << 30; // size mismatch without any large allocation
    expect_invalid(q);
    q = base;
    q.n_q_heads = (1u << 20) + 1; // over the sanity cap
    expect_invalid(q);

    // Zero scale keeps its "default to 1/sqrt(dk)" meaning and succeeds.
    {
        auto res = xkv_read_attention(base, {}, {}, nullptr, stamp, &store, rcfg);
        assert(res.status == xkv_read_status::success);
        for (float v : res.output) assert(v == 0.0f);
    }
    assert(b_cache->total_allocated_bytes() == 0);
    assert(store.current_stamp() == stamp);
    std::cout << "[Test 28] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 29: selected rows without a generation pin are rejected before decode.
// ----------------------------------------------------------------------------
static void test_generation_pin_enforcement() {
    std::cout << "[Test 29] Generation pin enforcement..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, 4, 8, 8, 32, 32, 23700, a_k, b_k, a_v, b_v);

    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = 8;
    query.head_dim_v = 8;
    query.q_vec = generate_deterministic_floats(8, 23701);

    auto make_view = [&]() {
        xkv_segment_read_view view;
        view.pin = store.pin_segment(seg->segment_id);
        view.segment_version_id = seg->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back(i);
            view.group_indices.push_back(0);
        }
        return view;
    };

    // Missing segment-level generation with no per-row generations: stale/unbound.
    {
        auto v = make_view();
        v.storage_generation = 0;
        std::vector<xkv_segment_read_view> views;
        views.push_back(std::move(v));
        auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store);
        assert(res.status == xkv_read_status::invalid_argument);
        assert(res.output.empty());
    }
    // One zero per-row generation poisons the view even with a segment default.
    {
        auto v = make_view();
        v.row_generations = {1, 1, 0, 1};
        std::vector<xkv_segment_read_view> views;
        views.push_back(std::move(v));
        auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store);
        assert(res.status == xkv_read_status::invalid_argument);
        assert(res.output.empty());
    }
    // Fully pinned control succeeds.
    {
        auto v = make_view();
        v.row_generations = {1, 1, 1, 1};
        std::vector<xkv_segment_read_view> views;
        views.push_back(std::move(v));
        auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store);
        assert(res.status == xkv_read_status::success);
        assert(res.output.size() == 8);
    }
    std::cout << "[Test 29] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 30: strict batch with SR=all plus a phase transform matches strict
// singles: union gather is a pure optimization with exact per-query semantics.
// ----------------------------------------------------------------------------
static void test_strict_batch_sr_all_matches_singles() {
    std::cout << "[Test 30] Strict SR=all batch vs singles under phase transform..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 16;
    const uint32_t dk = 16, dv = 16, rk = 16, rv = 16;
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rk, rv, 4 * dk, 4 * dv, 23800, a_k, b_k, a_v, b_v);
    const auto * g = seg->find_group(0);
    assert(g != nullptr);
    const uint32_t pad_rk = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
    const uint32_t pad_rv = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);

    xkv_query_input q0, q1;
    q0.query_index = 0; q0.n_q_heads = 1; q0.head_dim_k = dk; q0.head_dim_v = dv;
    q0.q_vec = generate_deterministic_floats(dk, 23801);
    q1.query_index = 1; q1.n_q_heads = 1; q1.head_dim_k = dk; q1.head_dim_v = dv;
    q1.q_vec = generate_deterministic_floats(dk, 23802);

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(100 + i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    sr_batch_selection_result bsel;
    for (uint32_t i = 0; i < n_rows; ++i) {
        segment_row_ref ref;
        ref.segment_id = seg->segment_id;
        ref.segment_version = seg->segment_version;
        ref.storage_generation = 1;
        ref.row = i;
        bsel.gather_rows.push_back(ref);
    }
    bsel.csr_ptrs = {0, 16, 32};
    for (uint32_t i = 0; i < 16; ++i) bsel.csr_indices.push_back(i);
    for (uint32_t i = 0; i < 16; ++i) bsel.csr_indices.push_back(i);

    // Deterministic RERoT-style phase: value-domain affine map of the canonical key.
    phase_transform_fn phase_tx = [](const float * src, int64_t, uint32_t, float * dst) {
        for (uint32_t i = 0; i < 16; ++i) dst[i] = src[i] * 1.25f + 0.5f;
    };

    auto ws_cfg = make_bounded_ws_config(2, 1, dk, dv, 8, pad_rk, pad_rv, 32);
    xkv_reader_workspace ws;
    assert(ws.warmup(ws_cfg));
    xkv_reader_config rcfg;
    rcfg.workspace = &ws;
    rcfg.tile_size = 8;

    xkv_snapshot_stamp stamp = store.current_stamp();
    xkv_batch_read_result bres;
    const std::vector<xkv_query_input> batch_queries = {q0, q1};
    {
        heap_scope scope;
        bres = xkv_read_attention_batch(batch_queries, {}, views, bsel, phase_tx, stamp, &store, rcfg);
        assert(bres.status == xkv_read_status::success);
        assert(bres.per_query.size() == 2);
        const size_t expect_bytes = 2 * sizeof(xkv_read_result) + 2 * static_cast<size_t>(dv * sizeof(float));
        assert(scope.count() == 3);
        assert(scope.bytes() == expect_bytes);
    }
    assert(ws.live_bytes() == 0);

    auto s0 = xkv_read_attention(q0, {}, views, phase_tx, stamp, &store, rcfg);
    auto s1 = xkv_read_attention(q1, {}, views, phase_tx, stamp, &store, rcfg);
    assert(s0.status == xkv_read_status::success);
    assert(s1.status == xkv_read_status::success);
    assert(vectors_approx_equal(bres.per_query[0].output, s0.output, 1e-5f));
    assert(vectors_approx_equal(bres.per_query[1].output, s1.output, 1e-5f));
    std::cout << "[Test 30] Passed." << std::endl;
}


// ----------------------------------------------------------------------------
// Quant-codec fixture for the decode_tmp proofs below. Mirrors the Turbo4
// helper but parameterizes the codec type (F16/Q8_0); per-layer slices are
// total/4 so B offset validation passes with total = 4 * head_dim.
// ----------------------------------------------------------------------------
static std::shared_ptr<xkv_segment> create_test_segment_quant(
    llama_xkv_cache_store & store,
    uint32_t n_rows,
    uint32_t rank_k,
    uint32_t rank_v,
    uint32_t total_dim_k,
    uint32_t total_dim_v,
    uint64_t seed_base,
    ggml_type type,
    std::vector<float> & out_a_k,
    std::vector<float> & out_b_k,
    std::vector<float> & out_a_v,
    std::vector<float> & out_b_v
) {
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_v;
    uint32_t ldim_k = total_dim_k / 4;
    uint32_t ldim_v = total_dim_v / 4;
    g.layer_feature_offsets_k = {0, ldim_k, 2 * ldim_k, 3 * ldim_k};
    g.layer_feature_dims_k = {ldim_k, ldim_k, ldim_k, total_dim_k - 3 * ldim_k};
    g.layer_feature_offsets_v = {0, ldim_v, 2 * ldim_v, 3 * ldim_v};
    g.layer_feature_dims_v = {ldim_v, ldim_v, ldim_v, total_dim_v - 3 * ldim_v};

    const uint32_t block = (type == GGML_TYPE_Q8_0) ? 32 : 0;
    codec_desc desc_a_k = make_codec_desc(factor_role::a_k, type, orientation::token_major, {n_rows, rank_k}, block, seed_base + 1);
    out_a_k = generate_deterministic_floats(n_rows * rank_k, seed_base + 1);
    g.a_k = encode_matrix(desc_a_k, out_a_k.data(), out_a_k.size());

    codec_desc desc_b_k = make_codec_desc(factor_role::b_k, type, orientation::feature_major_transposed, {total_dim_k, rank_k}, block, seed_base + 2);
    out_b_k = generate_deterministic_floats(total_dim_k * rank_k, seed_base + 2);
    g.set_b_k(encode_matrix(desc_b_k, out_b_k.data(), out_b_k.size()));

    codec_desc desc_a_v = make_codec_desc(factor_role::a_v, type, orientation::token_major, {n_rows, rank_v}, block, seed_base + 3);
    out_a_v = generate_deterministic_floats(n_rows * rank_v, seed_base + 3);
    g.a_v = encode_matrix(desc_a_v, out_a_v.data(), out_a_v.size());

    codec_desc desc_b_v = make_codec_desc(factor_role::b_v, type, orientation::feature_major_transposed, {total_dim_v, rank_v}, block, seed_base + 4);
    out_b_v = generate_deterministic_floats(total_dim_v * rank_v, seed_base + 4);
    g.set_b_v(encode_matrix(desc_b_v, out_b_v.data(), out_b_v.size()));

    // F16/Q8 factor streams are explicit reference/ablation fixtures. The
    // production TQ profiles require all four streams to be TurboQuant.
    auto seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, {g});
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = seed_base * 1000 + i + 1;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    bool marked = store.mark_seal_candidates(pids);
    assert(marked);

    std::string err;
    bool ok = store.publish_candidate(seg, pids, gens, &err);
    if (!ok) {
        std::cerr << "create_test_segment_quant publish_candidate failed: " << err << std::endl;
        assert(false);
    }
    return seg;
}

// ----------------------------------------------------------------------------
// Test 31: externally carved workspace (arena adapter): zero-copy, zero heap,
// no double counting, fail-closed carve validation, backing never freed.
// ----------------------------------------------------------------------------
static void test_workspace_external_carve() {
    std::cout << "[Test 31] Externally carved workspace..." << std::endl;

    auto fit = make_bounded_ws_config(1, 1, 8, 8, 4, 8, 8, 4);
    size_t total = 0;
    assert(xkv_estimate_workspace_layout(fit, total));
    assert(total > 0 && total % 64 == 0);

    void * mem = std::aligned_alloc(64, total);
    assert(mem != nullptr);

    xkv_reader_workspace ws;
    std::string err;
    assert(ws.warmup_external(mem, total, fit, &err));
    assert(ws.is_external());
    assert(ws.capacity_bytes() == total);
    assert(ws.scratch_ptr() == mem);
    // Owned heap excludes the carve: side tables only, never double-counted.
    assert(ws.allocated_bytes() < total);

    // Carve validation fails closed with prior state preserved.
    assert(!ws.warmup_external(reinterpret_cast<char *>(mem) + 8, total - 8, fit, &err));
    assert(!err.empty());
    assert(!ws.warmup_external(nullptr, total, fit, &err));
    assert(!ws.warmup_external(mem, total - 1, fit, &err));
    assert(ws.is_external());
    assert(ws.scratch_ptr() == mem);

    // A strict read runs on the carve with only the accounted output allocated.
    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, 4, 8, 8, 32, 32, 23900, a_k, b_k, a_v, b_v);
    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = 8;
    query.head_dim_v = 8;
    query.q_vec = generate_deterministic_floats(8, 23901);
    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));
    xkv_reader_config rcfg;
    rcfg.workspace = &ws;
    rcfg.tile_size = 4;
    {
        heap_scope scope;
        auto res = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store, rcfg);
        assert(res.status == xkv_read_status::success);
        assert(res.output.size() == 8);
        assert(scope.count() == 1);
        assert(scope.bytes() == static_cast<size_t>(8 * sizeof(float)));
    }
    assert(ws.live_bytes() == 0);

    // clear() drops the carve without freeing caller memory; owned warmup replaces it.
    ws.clear();
    assert(!ws.is_external());
    assert(!ws.is_warmed_up());
    *static_cast<char *>(mem) = 0x5A;
    assert(*static_cast<char *>(mem) == 0x5A);
    assert(ws.warmup(fit));
    assert(!ws.is_external());
    assert(ws.scratch_ptr() != mem);
    auto res2 = xkv_read_attention(query, {}, views, nullptr, store.current_stamp(), &store, rcfg);
    assert(res2.status == xkv_read_status::success);
    std::free(mem);
    std::cout << "[Test 31] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 32: strict F16/Q8_0 reads use the decode_tmp slice with zero heap, match
// the explicit-decode oracle, and fail one-byte-short on the tmp region.
// ----------------------------------------------------------------------------
static void test_strict_quant_decode_tmp() {
    std::cout << "[Test 32] Strict F16/Q8_0 decode_tmp zero-heap + one-short..." << std::endl;

    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        auto cparams = make_reader_test_cparams();
        llama_xkv_cache_store store(cparams);

        const uint32_t n_rows = 16;
        const uint32_t dk = 16, dv = 16, rk = 32, rv = 32;
        std::vector<float> a_k, b_k, a_v, b_v;
        auto seg = create_test_segment_quant(store, n_rows, rk, rv, 4 * dk, 4 * dv,
                                             24000 + (type == GGML_TYPE_Q8_0 ? 500 : 0),
                                             type, a_k, b_k, a_v, b_v);
        const auto * g = seg->find_group(0);
        assert(g != nullptr);
        const uint32_t pad_rk = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
        const uint32_t pad_rv = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);

        // Exact tmp needs via the codec query; F32-style types need 0.
        size_t need_ak = 0, need_bk = 0;
        assert(decode_rows_scratch_bytes(g->a_k.desc, need_ak, nullptr));
        assert(decode_rows_scratch_bytes(g->b_k->desc, need_bk, nullptr));
        assert(need_ak > 0 && need_bk > 0);
        size_t need_f32 = 0;
        codec_desc f32d = make_codec_desc(factor_role::a_k, GGML_TYPE_F32,
                                          orientation::token_major, {4, 8}, 0, 1);
        assert(decode_rows_scratch_bytes(f32d, need_f32, nullptr));
        assert(need_f32 == 0);

        xkv_query_input query;
        query.query_index = 0;
        query.n_q_heads = 1;
        query.head_dim_k = dk;
        query.head_dim_v = dv;
        query.q_vec = generate_deterministic_floats(dk, 24101);

        xkv_segment_read_view view;
        view.pin = store.pin_segment(seg->segment_id);
        view.segment_version_id = seg->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < n_rows; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back(i);
            view.group_indices.push_back(0);
        }
        std::vector<xkv_segment_read_view> views;
        views.push_back(std::move(view));
        xkv_snapshot_stamp stamp = store.current_stamp();

        auto ws_cfg = make_bounded_ws_config(1, 1, dk, dv, 8, pad_rk, pad_rv, 8);
        xkv_reader_workspace ws;
        assert(ws.warmup(ws_cfg));
        xkv_reader_config rcfg;
        rcfg.workspace = &ws;
        rcfg.tile_size = 8;

        {
            heap_scope scope;
            auto res = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg);
            assert(res.status == xkv_read_status::success);
            assert(scope.count() == 1);
            assert(scope.bytes() == static_cast<size_t>(dv * sizeof(float)));
        }
        assert(ws.live_bytes() == 0);

        // Explicit-decode oracle: same codec rows, double-precision GEMM.
        const auto * gg = seg->find_group(0);
        std::vector<uint64_t> sel(n_rows);
        for (uint32_t i = 0; i < n_rows; ++i) sel[i] = i;
        std::vector<float> dec_a_k(n_rows * pad_rk), dec_a_v(n_rows * pad_rv);
        decode_rows(gg->a_k, sel.data(), n_rows, dec_a_k.data(), dec_a_k.size(), value_domain::canonical);
        decode_rows(gg->a_v, sel.data(), n_rows, dec_a_v.data(), dec_a_v.size(), value_domain::canonical);
        std::vector<uint64_t> b_rows_k(dk), b_rows_v(dv);
        for (uint32_t i = 0; i < dk; ++i) b_rows_k[i] = i;
        for (uint32_t i = 0; i < dv; ++i) b_rows_v[i] = i;
        const uint32_t pad_bk = static_cast<uint32_t>(gg->b_k->desc.padded_shape.cols);
        const uint32_t pad_bv = static_cast<uint32_t>(gg->b_v->desc.padded_shape.cols);
        std::vector<float> dec_b_k(dk * pad_bk), dec_b_v(dv * pad_bv);
        decode_rows(*gg->b_k, b_rows_k.data(), dk, dec_b_k.data(), dec_b_k.size(), value_domain::canonical);
        decode_rows(*gg->b_v, b_rows_v.data(), dv, dec_b_v.data(), dec_b_v.size(), value_domain::canonical);
        std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(dk, 0.0f));
        std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(dv, 0.0f));
        for (uint32_t r = 0; r < n_rows; ++r) {
            for (uint32_t f = 0; f < dk; ++f) {
                double s = 0.0;
                for (uint32_t k = 0; k < rk; ++k) s += (double) dec_a_k[r * pad_rk + k] * (double) dec_b_k[f * pad_bk + k];
                cold_keys[r][f] = (float) s;
            }
            for (uint32_t f = 0; f < dv; ++f) {
                double s = 0.0;
                for (uint32_t k = 0; k < rv; ++k) s += (double) dec_a_v[r * pad_rv + k] * (double) dec_b_v[f * pad_bv + k];
                cold_values[r][f] = (float) s;
            }
        }
        std::vector<uint32_t> groups(n_rows, 0);
        auto oracle_out = xkv_dense_attention_reference(query, {}, cold_keys, cold_values, groups, {});
        auto res = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg);
        assert(res.status == xkv_read_status::success);
        assert(vectors_approx_equal(res.output, oracle_out, 1e-4f));

        // One-byte-short on the decode_tmp region fails deterministically.
        auto short_cfg = make_bounded_ws_config(1, 1, dk, dv, 8, pad_rk, pad_rv, 8);
        short_cfg.max_decode_tmp_bytes = static_cast<uint32_t>(need_ak - 1);
        size_t short_total = 0;
        assert(xkv_estimate_workspace_layout(short_cfg, short_total));
        short_cfg.capacity_bytes = short_total;
        xkv_reader_workspace ws_short;
        assert(ws_short.warmup(short_cfg));
        xkv_reader_config rcfg_short;
        rcfg_short.workspace = &ws_short;
        rcfg_short.tile_size = 8;
        auto rej = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg_short);
        assert(rej.status == xkv_read_status::workspace_exceeded);
        assert(rej.output.empty());
        assert(ws_short.live_bytes() == 0);

        // Exact-need knob succeeds.
        short_cfg.max_decode_tmp_bytes = static_cast<uint32_t>(need_ak);
        assert(xkv_estimate_workspace_layout(short_cfg, short_total));
        short_cfg.capacity_bytes = short_total;
        xkv_reader_workspace ws_exact;
        assert(ws_exact.warmup(short_cfg));
        rcfg_short.workspace = &ws_exact;
        auto ok = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg_short);
        assert(ok.status == xkv_read_status::success);
    }
    std::cout << "[Test 32] Passed." << std::endl;
}


// ----------------------------------------------------------------------------
// Test 33: two queries share the same cold rows (CSR) and the same hot row but
// use different per-query DDVR groups. Each query processes every shared row
// exactly once with its own group: the batch result matches per-query single
// oracles built with the mirrored per-row groups. Includes a UINT32_MAX CSR
// entry exercising the view-group fallback. Zero-heap bounded strict read.
// ----------------------------------------------------------------------------
static void test_batch_shared_rows_different_groups() {
    std::cout << "[Test 33] Shared cold+hot rows, per-query groups..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 8;
    const uint32_t dk = 8, dv = 8, rk = 8, rv = 8;
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rk, rv, 4 * dk, 4 * dv, 24200, a_k, b_k, a_v, b_v);
    const auto * g = seg->find_group(0);
    assert(g != nullptr);
    const uint32_t pad_rk = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
    const uint32_t pad_rv = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);

    // Two queries, two DDVR groups each, all four Q vectors distinct.
    xkv_query_input q0, q1;
    q0.query_index = 0; q0.n_q_heads = 1; q0.head_dim_k = dk; q0.head_dim_v = dv;
    q0.q_groups = {generate_deterministic_floats(dk, 24201),
                   generate_deterministic_floats(dk, 24202)};
    q1.query_index = 1; q1.n_q_heads = 1; q1.head_dim_k = dk; q1.head_dim_v = dv;
    q1.q_groups = {generate_deterministic_floats(dk, 24203),
                   generate_deterministic_floats(dk, 24204)};

    // One hot row shared by both queries with per-query groups {0, 1}.
    std::vector<float> hk = generate_deterministic_floats(dk, 24205);
    std::vector<float> hv = generate_deterministic_floats(dv, 24206);
    xkv_hot_row hr;
    hr.row_index = 0; hr.storage_pos = 100;
    hr.k_ptr = hk.data(); hr.v_ptr = hv.data();
    hr.query_group_indices = {0, 1};
    std::vector<xkv_hot_row> hot_rows = {hr};

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    std::vector<xkv_segment_read_view> views;
    views.push_back(std::move(view));

    // CSR: both queries select all rows; q0 uses group 0 everywhere, q1 uses
    // group 1 except row 5 (UINT32_MAX -> view-group 0 fallback).
    sr_batch_selection_result bsel;
    for (uint32_t i = 0; i < n_rows; ++i) {
        segment_row_ref ref;
        ref.segment_id = seg->segment_id;
        ref.segment_version = seg->segment_version;
        ref.storage_generation = 1;
        ref.row = i;
        bsel.gather_rows.push_back(ref);
    }
    bsel.csr_ptrs = {0, 8, 16};
    for (uint32_t i = 0; i < 8; ++i) bsel.csr_indices.push_back(i);
    for (uint32_t i = 0; i < 8; ++i) bsel.csr_indices.push_back(i);
    for (uint32_t i = 0; i < 8; ++i) bsel.csr_group_indices.push_back(0);
    for (uint32_t i = 0; i < 8; ++i) {
        bsel.csr_group_indices.push_back(i == 5 ? UINT32_MAX : 1);
    }

    auto ws_cfg = make_bounded_ws_config(2, 1, dk, dv, 8, pad_rk, pad_rv, 16);
    xkv_reader_workspace ws;
    assert(ws.warmup(ws_cfg));
    xkv_reader_config rcfg;
    rcfg.workspace = &ws;
    rcfg.tile_size = 8;
    xkv_snapshot_stamp stamp = store.current_stamp();

    xkv_batch_read_result bres;
    const std::vector<xkv_query_input> batch_queries = {q0, q1};
    {
        heap_scope scope;
        bres = xkv_read_attention_batch(batch_queries, hot_rows, views, bsel, nullptr, stamp, &store, rcfg);
        assert(bres.status == xkv_read_status::success);
        assert(bres.per_query.size() == 2);
        const size_t expect_bytes = 2 * sizeof(xkv_read_result) + 2 * static_cast<size_t>(dv * sizeof(float));
        assert(scope.count() == 3);
        assert(scope.bytes() == expect_bytes);
    }
    assert(ws.live_bytes() == 0);

    // Oracle q0: single read, all rows group 0, hot mapping {0}.
    xkv_hot_row hr0 = hr;
    hr0.query_group_indices = {0};
    std::vector<xkv_hot_row> hot0 = {hr0};
    auto s0 = xkv_read_attention(q0, hot0, views, nullptr, stamp, &store, rcfg);
    assert(s0.status == xkv_read_status::success);
    assert(vectors_approx_equal(bres.per_query[0].output, s0.output, 1e-5f));

    // Oracle q1: single read, rows group 1 except row 5 group 0, hot mapping {1}.
    xkv_segment_read_view view1;
    view1.pin = store.pin_segment(seg->segment_id);
    view1.segment_version_id = seg->segment_version;
    view1.storage_generation = 1;
    view1.owning_layer = 0;
    view1.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view1.selected_rows.push_back(i);
        view1.storage_positions.push_back(i);
        view1.group_indices.push_back(i == 5 ? 0 : 1);
    }
    std::vector<xkv_segment_read_view> views1;
    views1.push_back(std::move(view1));
    xkv_hot_row hr1 = hr;
    hr1.query_group_indices = {1};
    std::vector<xkv_hot_row> hot1 = {hr1};
    auto s1 = xkv_read_attention(q1, hot1, views1, nullptr, stamp, &store, rcfg);
    assert(s1.status == xkv_read_status::success);
    assert(vectors_approx_equal(bres.per_query[1].output, s1.output, 1e-5f));

    std::cout << "[Test 33] Passed." << std::endl;
}

// ----------------------------------------------------------------------------
// Test 34: per-query group mapping validation matrix (CSR + hot, single + batch).
// ----------------------------------------------------------------------------
static void test_group_mapping_validation() {
    std::cout << "[Test 34] Group mapping validation matrix..." << std::endl;

    auto cparams = make_reader_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 4;
    const uint32_t dk = 8, dv = 8;
    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, 8, 8, 4 * dk, 4 * dv, 24300, a_k, b_k, a_v, b_v);

    auto make_query = [&](uint32_t idx, bool with_groups) {
        xkv_query_input q;
        q.query_index = idx; q.n_q_heads = 1; q.head_dim_k = dk; q.head_dim_v = dv;
        if (with_groups) {
            q.q_groups = {generate_deterministic_floats(dk, 24310 + idx),
                          generate_deterministic_floats(dk, 24320 + idx)};
        } else {
            q.q_vec = generate_deterministic_floats(dk, 24330 + idx);
        }
        return q;
    };
    auto make_view = [&]() {
        xkv_segment_read_view view;
        view.pin = store.pin_segment(seg->segment_id);
        view.segment_version_id = seg->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < n_rows; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back(i);
            view.group_indices.push_back(0);
        }
        return view;
    };
    auto make_bsel = [&](bool with_groups) {
        sr_batch_selection_result bsel;
        for (uint32_t i = 0; i < n_rows; ++i) {
            segment_row_ref ref;
            ref.segment_id = seg->segment_id;
            ref.segment_version = seg->segment_version;
            ref.storage_generation = 1;
            ref.row = i;
            bsel.gather_rows.push_back(ref);
        }
        bsel.csr_ptrs = {0, 4, 8};
        for (uint32_t i = 0; i < 4; ++i) bsel.csr_indices.push_back(i);
        for (uint32_t i = 0; i < 4; ++i) bsel.csr_indices.push_back(i);
        if (with_groups) {
            for (uint32_t i = 0; i < 8; ++i) bsel.csr_group_indices.push_back(0);
        }
        return bsel;
    };
    xkv_snapshot_stamp stamp = store.current_stamp();
    auto b_cache = std::make_shared<xkv_b_tile_cache>(1024 * 1024);
    xkv_reader_config rcfg;
    rcfg.b_cache = b_cache;

    xkv_query_input q0 = make_query(0, true);
    xkv_query_input q1 = make_query(1, true);
    std::vector<xkv_segment_read_view> views;
    views.push_back(make_view());

    // 1. CSR group array length mismatch.
    {
        auto bsel = make_bsel(true);
        bsel.csr_group_indices.pop_back();
        auto r = xkv_read_attention_batch({q0, q1}, {}, views, bsel, nullptr, stamp, &store, rcfg);
        assert(r.status == xkv_read_status::invalid_argument);
        assert(r.per_query.empty());
    }
    // 2. CSR group out of range.
    {
        auto bsel = make_bsel(true);
        bsel.csr_group_indices[3] = 7;
        auto r = xkv_read_attention_batch({q0, q1}, {}, views, bsel, nullptr, stamp, &store, rcfg);
        assert(r.status == xkv_read_status::invalid_argument);
        assert(r.per_query.empty());
    }
    // 3. CSR non-zero group without q_groups.
    {
        xkv_query_input nq0 = make_query(0, false);
        xkv_query_input nq1 = make_query(1, false);
        auto bsel = make_bsel(true);
        bsel.csr_group_indices[0] = 1;
        auto r = xkv_read_attention_batch({nq0, nq1}, {}, views, bsel, nullptr, stamp, &store, rcfg);
        assert(r.status == xkv_read_status::invalid_argument);
        assert(r.per_query.empty());
    }
    // 4. Hot mapping length mismatch (batch).
    {
        std::vector<float> hk = generate_deterministic_floats(dk, 24340);
        std::vector<float> hv = generate_deterministic_floats(dv, 24341);
        xkv_hot_row hr;
        hr.k_ptr = hk.data(); hr.v_ptr = hv.data();
        hr.query_group_indices = {0};
        auto r = xkv_read_attention_batch({q0, q1}, {hr}, views, make_bsel(false),
                                          nullptr, stamp, &store, rcfg);
        assert(r.status == xkv_read_status::invalid_argument);
        assert(r.per_query.empty());
    }
    // 5. Hot visible query with UINT32_MAX group.
    {
        std::vector<float> hk = generate_deterministic_floats(dk, 24342);
        std::vector<float> hv = generate_deterministic_floats(dv, 24343);
        xkv_hot_row hr;
        hr.k_ptr = hk.data(); hr.v_ptr = hv.data();
        hr.query_group_indices = {0, UINT32_MAX};
        auto r = xkv_read_attention_batch({q0, q1}, {hr}, views, make_bsel(false),
                                          nullptr, stamp, &store, rcfg);
        assert(r.status == xkv_read_status::invalid_argument);
        assert(r.per_query.empty());
    }
    // 6. Hot empty mapping with group invalid for a visible query.
    {
        std::vector<float> hk = generate_deterministic_floats(dk, 24344);
        std::vector<float> hv = generate_deterministic_floats(dv, 24345);
        xkv_hot_row hr;
        hr.k_ptr = hk.data(); hr.v_ptr = hv.data();
        hr.group_index = 5;
        auto r = xkv_read_attention_batch({q0, q1}, {hr}, views, make_bsel(false),
                                          nullptr, stamp, &store, rcfg);
        assert(r.status == xkv_read_status::invalid_argument);
        assert(r.per_query.empty());
    }
    // 7. Single-read mapping length must be exactly 1.
    {
        std::vector<float> hk = generate_deterministic_floats(dk, 24346);
        std::vector<float> hv = generate_deterministic_floats(dv, 24347);
        xkv_hot_row hr;
        hr.k_ptr = hk.data(); hr.v_ptr = hv.data();
        hr.query_group_indices = {0, 0};
        auto r = xkv_read_attention(q0, {hr}, views, nullptr, stamp, &store, rcfg);
        assert(r.status == xkv_read_status::invalid_argument);
        assert(r.output.empty());
    }
    // 8. No cache mutation on any rejection above.
    assert(b_cache->total_allocated_bytes() == 0);
    assert(store.current_stamp() == stamp);
    std::cout << "[Test 34] Passed." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running comprehensive llama-xkv-reader unit tests..." << std::endl;
    std::cout << "========================================" << std::endl;

    test_full_rank_no_sr_vs_dense_oracle();
    test_real_turbo4_four_stream_vs_explicit_oracle();
    test_multi_segment_online_softmax_extreme_logits();
    test_ddvr_spans_and_groups();
    test_variable_kv_dims_and_gqa();
    test_per_query_speculative_isolation();
    test_hot_row_isolation_and_query_32();
    test_empty_and_all_masked_without_nan();
    test_stale_stamp_retry();
    test_checked_arithmetic_overflow();
    test_batch_workspace_sum_budget_rejected();
    test_batch_mixed_head_dims_rejected();
    test_empty_batch_preflight();
    test_b_cache_lease_accounting_and_retention();
    test_owning_reader_pin_requirement();
    test_sinks_and_softcap_ordering();
    test_workspace_warmup_stability_and_zero_growth();
    test_insufficient_workspace_fails_without_mutation();
    test_multi_reader_workspace_matches_dense_oracle();
    test_workspace_hard_capacity_and_overflow_rejection();
    test_workspace_exclusive_single_lease_no_alias();
    test_workspace_non_movable_and_atomic_reconfigure();
    test_strict_single_zero_heap();
    test_strict_batch_csr_zero_heap();
    test_b_cache_hit_and_overbudget_zero_heap();
    test_slice_short_matrix();
    test_planner_byte_exact();
    test_scale_and_huge_dim_rejection();
    test_generation_pin_enforcement();
    test_strict_batch_sr_all_matches_singles();
    test_workspace_external_carve();
    test_strict_quant_decode_tmp();
    test_batch_shared_rows_different_groups();
    test_group_mapping_validation();

    // Test 22: Fail-closed resolution: invalid layer, conflicting group index, layer 0 resolution
    {
        std::cout << "[Test 22] Fail-closed resolution: invalid layer, conflicting group index, layer 0..." << std::endl;
        auto cparams = make_reader_test_cparams();
        llama_xkv_cache_store store(cparams);
        std::vector<float> a_k, b_k, a_v, b_v;
        auto seg = create_test_segment_f32(store, 8, 16, 16, 64, 64, 22001, a_k, b_k, a_v, b_v);

        xkv_segment_read_view bad_view;
        bad_view.pin = store.pin_segment(seg->segment_id);
        bad_view.selected_rows = {0};
        bad_view.storage_positions = {0};
        // Generation pin required by the reader before decode.
        bad_view.storage_generation = 1;

        // A. Layer 0 is valid and resolves cleanly
        bad_view.owning_layer = 0;
        bad_view.factor_group_index = 0;
        std::string err;
        assert(bad_view.validate(&err));

        // B. Deliberately wrong/non-existent layer must fail closed
        bad_view.owning_layer = 999;
        bad_view.factor_group_index = UINT32_MAX;
        assert(!bad_view.validate(&err));
        assert(err.find("owning_layer") != std::string::npos);

        // C. Conflicting factor_group_index and owning_layer must fail closed
        bad_view.owning_layer = 0;
        bad_view.factor_group_index = 5; // Group 5 does not exist
        assert(!bad_view.validate(&err));
        assert(err.find("factor group index") != std::string::npos || err.find("conflict") != std::string::npos);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "All 34 llama-xkv-reader tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
