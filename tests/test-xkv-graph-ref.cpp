#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-graph-ref.h"
#include "llama-xkv-reader.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-landmark.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace llama_xkv;

static llama_cparams make_graph_ref_cparams() {
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
            std::cerr << "Mismatch at index " << i << ": a=" << a[i] << " b=" << b[i]
                      << " diff=" << std::fabs(a[i] - b[i]) << std::endl;
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
    g.layer_feature_offsets_k = {0, total_dim_k / 4, total_dim_k / 2, 3 * total_dim_k / 4};
    g.layer_feature_dims_k = {total_dim_k / 4, total_dim_k / 4, total_dim_k / 4, total_dim_k / 4};
    g.layer_feature_offsets_v = {0, total_dim_v / 4, total_dim_v / 2, 3 * total_dim_v / 4};
    g.layer_feature_dims_v = {total_dim_v / 4, total_dim_v / 4, total_dim_v / 4, total_dim_v / 4};

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
    assert(ok);
    return seg;
}

// ----------------------------------------------------------------------------
// Test 1: Combined Hot + Factored attention matching dense oracle in real GGML graph
// ----------------------------------------------------------------------------
static void test_graph_hot_and_factored() {
    std::cout << "Running test_graph_hot_and_factored..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;
    const uint32_t n_cold_rows = 6;
    const uint32_t n_hot_rows = 4;
    const uint32_t n_queries = 1;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_cold_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 100, a_k, b_k, a_v, b_v);

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = n_queries;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;

    // Hot rows
    for (uint32_t h = 0; h < n_hot_rows; ++h) {
        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = h;
        hd.storage_pos = n_cold_rows + h;
        hd.group_index = 0;
        hd.is_valid = true;
        hd.k_data = generate_deterministic_floats(head_dim_k, 500 + h);
        hd.v_data = generate_deterministic_floats(head_dim_v, 600 + h);
        snap->hot_data.push_back(std::move(hd));
    }

    // Factored segment view
    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_cold_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    snap->segment_views.push_back(std::move(view));

    // Setup GGML context and graph
    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 777);
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);
    assert(op_handle != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);

    // Compute graph on CPU
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    // Compare with dense oracle
    std::vector<xkv_hot_row> oracle_hot_rows;
    for (const auto & hd : op_handle->snapshot()->hot_data) {
        xkv_hot_row hr;
        hr.row_index = hd.row_index;
        hr.storage_pos = hd.storage_pos;
        hr.k_ptr = hd.k_data.data();
        hr.v_ptr = hd.v_data.data();
        hr.is_valid = true;
        oracle_hot_rows.push_back(hr);
    }

    std::vector<std::vector<float>> cold_keys(n_cold_rows, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(n_cold_rows, std::vector<float>(head_dim_v, 0.0f));
    for (uint32_t r = 0; r < n_cold_rows; ++r) {
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

    xkv_query_input qi;
    qi.query_index = 0;
    qi.n_q_heads = 1;
    qi.head_dim_k = head_dim_k;
    qi.head_dim_v = head_dim_v;
    qi.q_vec = q_data;

    std::vector<float> expected = xkv_dense_attention_reference(qi, oracle_hot_rows, cold_keys, cold_values, {}, {});

    std::vector<float> actual(head_dim_v);
    std::memcpy(actual.data(), out_tensor->data, head_dim_v * sizeof(float));

    assert(vectors_approx_equal(actual, expected, 1e-4f));

    ggml_free(ctx);
    std::cout << "test_graph_hot_and_factored PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 2: DDVR groups and phase transform support
// ----------------------------------------------------------------------------
static void test_graph_ddvr_groups() {
    std::cout << "Running test_graph_ddvr_groups..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;
    const uint32_t n_rows = 8;
    const uint32_t n_ddvr_groups = 2;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 200, a_k, b_k, a_v, b_v);

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = 1;
    snap->n_ddvr_groups = n_ddvr_groups;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;

    snap->phase_tx = [](const float * src_key, int64_t storage_pos, uint32_t /*head_idx*/, float * dst_key) {
        float factor = 1.0f + 0.02f * static_cast<float>(storage_pos);
        for (uint32_t i = 0; i < 16; ++i) {
            dst_key[i] = src_key[i] * factor;
        }
    };

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(i % n_ddvr_groups);
    }
    snap->segment_views.push_back(std::move(view));

    // Q tensor has shape [head_dim_k, n_ddvr_groups] for 1 query
    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q_g0 = generate_deterministic_floats(head_dim_k, 111);
    std::vector<float> q_g1 = generate_deterministic_floats(head_dim_k, 222);
    std::vector<float> q_tensor_data;
    q_tensor_data.insert(q_tensor_data.end(), q_g0.begin(), q_g0.end());
    q_tensor_data.insert(q_tensor_data.end(), q_g1.begin(), q_g1.end());

    struct ggml_tensor * q_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim_k, n_ddvr_groups);
    std::memcpy(q_tensor->data, q_tensor_data.data(), q_tensor_data.size() * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    // Dense reference with phase transform applied to keys
    std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(head_dim_k, 0.0f));
    std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(head_dim_v, 0.0f));
    std::vector<uint32_t> groups(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) {
        groups[r] = r % n_ddvr_groups;
        std::vector<float> raw_k(head_dim_k, 0.0f);
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                raw_k[f] += a_k[r * rank + k] * b_k[f * rank + k];
            }
        }
        for (uint32_t f = 0; f < head_dim_v; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                cold_values[r][f] += a_v[r * rank + k] * b_v[f * rank + k];
            }
        }
        float factor = 1.0f + 0.02f * static_cast<float>(r);
        for (uint32_t f = 0; f < head_dim_k; ++f) {
            cold_keys[r][f] = raw_k[f] * factor;
        }
    }

    xkv_query_input qi;
    qi.query_index = 0;
    qi.n_q_heads = 1;
    qi.head_dim_k = head_dim_k;
    qi.head_dim_v = head_dim_v;
    qi.q_groups = { q_g0, q_g1 };

    std::vector<float> expected = xkv_dense_attention_reference(qi, {}, cold_keys, cold_values, groups, {});
    std::vector<float> actual(head_dim_v);
    std::memcpy(actual.data(), out_tensor->data, head_dim_v * sizeof(float));

    assert(vectors_approx_equal(actual, expected, 1e-4f));

    ggml_free(ctx);
    std::cout << "test_graph_ddvr_groups PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 3: Dk != Dv and Softcap and Sink logit
// ----------------------------------------------------------------------------
static void test_graph_dk_neq_dv_softcap_sink() {
    std::cout << "Running test_graph_dk_neq_dv_softcap_sink..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 32;
    const uint32_t head_dim_v = 16;
    const uint32_t rank_k = 8;
    const uint32_t rank_v = 8;
    const uint32_t n_rows = 6;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank_k, rank_v, 4 * head_dim_k, 4 * head_dim_v, 300, a_k, b_k, a_v, b_v);

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = 1;
    snap->logit_softcap = 15.0f;
    snap->scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
    snap->query_sink_logits = { { 2.5f } }; // Sink logit 2.5
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;

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
    snap->segment_views.push_back(std::move(view));

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 333);
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);
    assert(out_tensor->ne[0] == head_dim_v);
    assert(out_tensor->ne[1] == 1);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    // Compute expected result via direct batch reader call with same parameters
    xkv_query_input qi;
    qi.query_index = 0;
    qi.n_q_heads = 1;
    qi.head_dim_k = head_dim_k;
    qi.head_dim_v = head_dim_v;
    qi.q_vec = q_data;
    qi.logit_softcap = 15.0f;
    qi.scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
    qi.sink_logits = { 2.5f };

    xkv_segment_read_view view2;
    view2.pin = store.pin_segment(seg->segment_id);
    view2.segment_version_id = seg->segment_version;
    view2.storage_generation = 1;
    view2.owning_layer = 0;
    view2.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view2.selected_rows.push_back(i);
        view2.storage_positions.push_back(i);
        view2.group_indices.push_back(0);
    }

    std::vector<xkv_segment_read_view> views2;
    views2.push_back(std::move(view2));
    auto expected_res = xkv_read_attention(qi, {}, views2, nullptr, store.current_stamp(), &store);
    assert(expected_res.status == xkv_read_status::success);

    std::vector<float> actual(head_dim_v);
    std::memcpy(actual.data(), out_tensor->data, head_dim_v * sizeof(float));

    assert(vectors_approx_equal(actual, expected_res.output, 1e-4f));

    ggml_free(ctx);
    std::cout << "test_graph_dk_neq_dv_softcap_sink PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 4: SR CSR Selection and Multi-Query Future Isolation
// ----------------------------------------------------------------------------
static void test_graph_sr_csr_and_future_isolation() {
    std::cout << "Running test_graph_sr_csr_and_future_isolation..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;
    const uint32_t n_rows = 10;
    const uint32_t n_queries = 2;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 400, a_k, b_k, a_v, b_v);

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = n_queries;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;

    // Query 0 causal limit pos 4 (can only see rows 0..4)
    // Query 1 causal limit pos 9 (can see rows 0..9)
    snap->query_causal_limits = { 4, 9 };

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
    snap->segment_views.push_back(std::move(view));

    // Setup CSR batch selection:
    // gather_rows: [row 0..9]
    // Q0 selects rows {0, 1, 2, 7} (7 will be rejected by causal cutoff 4)
    // Q1 selects rows {5, 6, 7} (all visible)
    for (uint32_t i = 0; i < n_rows; ++i) {
        segment_row_ref ref;
        ref.segment_id = seg->segment_id;
        ref.segment_version = seg->segment_version;
        ref.storage_generation = 1;
        ref.row = i;
        snap->sr_selection.gather_rows.push_back(ref);
    }
    // CSR ptrs: size n_queries + 1 = 3
    snap->sr_selection.csr_ptrs = { 0, 4, 7 };
    snap->sr_selection.csr_indices = { 0, 1, 2, 7,  5, 6, 7 };

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q0_data = generate_deterministic_floats(head_dim_k, 501);
    std::vector<float> q1_data = generate_deterministic_floats(head_dim_k, 502);
    std::vector<float> q_tensor_data;
    q_tensor_data.insert(q_tensor_data.end(), q0_data.begin(), q0_data.end());
    q_tensor_data.insert(q_tensor_data.end(), q1_data.begin(), q1_data.end());

    struct ggml_tensor * q_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim_k, n_queries);
    std::memcpy(q_tensor->data, q_tensor_data.data(), q_tensor_data.size() * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    // Compare with reader batch directly
    xkv_query_input qi0;
    qi0.query_index = 0;
    qi0.head_dim_k = head_dim_k;
    qi0.head_dim_v = head_dim_v;
    qi0.causal_limit_pos = 4;
    qi0.q_vec = q0_data;

    xkv_query_input qi1;
    qi1.query_index = 1;
    qi1.head_dim_k = head_dim_k;
    qi1.head_dim_v = head_dim_v;
    qi1.causal_limit_pos = 9;
    qi1.q_vec = q1_data;

    xkv_segment_read_view view2;
    view2.pin = store.pin_segment(seg->segment_id);
    view2.segment_version_id = seg->segment_version;
    view2.storage_generation = 1;
    view2.owning_layer = 0;
    view2.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view2.selected_rows.push_back(i);
        view2.storage_positions.push_back(i);
        view2.group_indices.push_back(0);
    }

    sr_batch_selection_result sel2;
    for (uint32_t i = 0; i < n_rows; ++i) {
        segment_row_ref ref;
        ref.segment_id = seg->segment_id;
        ref.segment_version = seg->segment_version;
        ref.storage_generation = 1;
        ref.row = i;
        sel2.gather_rows.push_back(ref);
    }
    sel2.csr_ptrs = { 0, 4, 7 };
    sel2.csr_indices = { 0, 1, 2, 7,  5, 6, 7 };

    std::vector<xkv_segment_read_view> views2;
    views2.push_back(std::move(view2));
    auto batch_res = xkv_read_attention_batch({qi0, qi1}, {}, views2, sel2, nullptr, store.current_stamp(), &store);
    assert(batch_res.status == xkv_read_status::success);

    std::vector<float> actual_q0(head_dim_v);
    std::vector<float> actual_q1(head_dim_v);
    const float * out_ptr = static_cast<const float *>(out_tensor->data);
    std::memcpy(actual_q0.data(), out_ptr, head_dim_v * sizeof(float));
    std::memcpy(actual_q1.data(), out_ptr + head_dim_v, head_dim_v * sizeof(float));

    assert(vectors_approx_equal(actual_q0, batch_res.per_query[0].output, 1e-4f));
    assert(vectors_approx_equal(actual_q1, batch_res.per_query[1].output, 1e-4f));

    ggml_free(ctx);
    std::cout << "test_graph_sr_csr_and_future_isolation PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 5: Empty/All-Masked rows produces finite zero without NaN
// ----------------------------------------------------------------------------
static void test_graph_empty_masked_finite_zero() {
    std::cout << "Running test_graph_empty_masked_finite_zero..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = 1;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;
    // No hot rows, no segment views

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 999);
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    const float * out_ptr = static_cast<const float *>(out_tensor->data);
    for (uint32_t i = 0; i < head_dim_v; ++i) {
        assert(std::isfinite(out_ptr[i]));
        assert(out_ptr[i] == 0.0f);
    }

    ggml_free(ctx);
    std::cout << "test_graph_empty_masked_finite_zero PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 6: Stale stamp fail-closed behavior
// ----------------------------------------------------------------------------
static void test_graph_stale_stamp_fail_closed() {
    std::cout << "Running test_graph_stale_stamp_fail_closed..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = 1;
    // Set a stale stamp
    snap->expected_stamp = store.current_stamp();
    snap->expected_stamp.content_epoch += 99; // intentionally mismatched
    snap->store = &store;

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 999);
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);

    // Fill output with garbage to test fail-closed zeroing
    std::memset(out_tensor->data, 0x7F, ggml_nbytes(out_tensor));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Must be rejected with retry_stale_stamp and output zeroed out
    assert(!op_handle->succeeded());
    assert(op_handle->status() == xkv_read_status::retry_stale_stamp);

    const float * out_ptr = static_cast<const float *>(out_tensor->data);
    for (uint32_t i = 0; i < head_dim_v; ++i) {
        assert(out_ptr[i] == 0.0f);
    }

    ggml_free(ctx);
    std::cout << "test_graph_stale_stamp_fail_closed PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 7: Bounded workspace preflight check and cleanup
// ----------------------------------------------------------------------------
static void test_graph_workspace_limit_fail_closed() {
    std::cout << "Running test_graph_workspace_limit_fail_closed..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;
    const uint32_t n_rows = 64;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 700, a_k, b_k, a_v, b_v);

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = 1;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;

    // Imposing an impossibly tiny budget: 64 bytes
    snap->reader_config.workspace_budget_bytes = 64;

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
    snap->segment_views.push_back(std::move(view));

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 701);
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(!op_handle->succeeded());
    assert(op_handle->status() == xkv_read_status::workspace_exceeded);

    const float * out_ptr = static_cast<const float *>(out_tensor->data);
    for (uint32_t i = 0; i < head_dim_v; ++i) {
        assert(out_ptr[i] == 0.0f);
    }

    ggml_free(ctx);
    std::cout << "test_graph_workspace_limit_fail_closed PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 8: Uncomputed initial status is fail-closed
// ----------------------------------------------------------------------------
static void test_graph_uncomputed_initial_status() {
    std::cout << "Running test_graph_uncomputed_initial_status..." << std::endl;

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = 16;
    snap->head_dim_v = 16;
    snap->n_q_heads = 1;
    snap->n_queries = 1;

    xkv_graph_op_handle handle(std::move(snap));
    // Before any compute call: must NOT be succeeded, must have invalid_argument status
    assert(!handle.succeeded());
    assert(handle.status() == xkv_read_status::invalid_argument);
    assert(handle.error_message() == "uncomputed");

    std::cout << "test_graph_uncomputed_initial_status PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 9: Malformed or truncated Q tensor rejection
// ----------------------------------------------------------------------------
static void test_graph_malformed_truncated_q() {
    std::cout << "Running test_graph_malformed_truncated_q..." << std::endl;

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t n_queries = 2;

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Truncated Q tensor: only allocate 1 query worth of elements (16 floats) instead of 2 (32 floats)
    struct ggml_tensor * q_truncated = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 888);
    std::memcpy(q_truncated->data, q_data.data(), head_dim_k * sizeof(float));

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = n_queries;

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_truncated, std::move(snap), op_handle);
    assert(out_tensor != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Truncated Q tensor must fail-closed
    assert(!op_handle->succeeded());
    assert(op_handle->status() == xkv_read_status::invalid_argument);

    // Output destination must be fully zeroed
    const float * out_ptr = static_cast<const float *>(out_tensor->data);
    size_t total_out = head_dim_v * n_queries;
    for (size_t i = 0; i < total_out; ++i) {
        assert(out_ptr[i] == 0.0f);
    }

    // Non-F32 Q tensor (e.g. F16)
    struct ggml_tensor * q_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head_dim_k, n_queries);
    auto snap2 = std::make_unique<xkv_graph_snapshot>();
    snap2->head_dim_k = head_dim_k;
    snap2->head_dim_v = head_dim_v;
    snap2->n_q_heads = 1;
    snap2->n_queries = n_queries;

    std::shared_ptr<xkv_graph_op_handle> op_handle2;
    struct ggml_tensor * out_tensor2 = xkv_build_graph_attention_ref(ctx, q_f16, std::move(snap2), op_handle2);
    assert(out_tensor2 != nullptr);

    struct ggml_cgraph * gf2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf2, out_tensor2);
    ggml_graph_compute_with_ctx(ctx, gf2, 1);

    assert(!op_handle2->succeeded());
    assert(op_handle2->status() == xkv_read_status::invalid_argument);

    ggml_free(ctx);
    std::cout << "test_graph_malformed_truncated_q PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 10: Non-contiguous / strided Q tensor rejection
// ----------------------------------------------------------------------------
static void test_graph_non_contiguous_strided_q() {
    std::cout << "Running test_graph_non_contiguous_strided_q..." << std::endl;

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t n_queries = 2;

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create a 2D tensor [head_dim_k, 4] then take a view or transpose to make it non-contiguous
    struct ggml_tensor * q_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim_k, 4);
    std::vector<float> q_data = generate_deterministic_floats(head_dim_k * 4, 1234);
    std::memcpy(q_base->data, q_data.data(), q_data.size() * sizeof(float));

    // Transposed tensor [4, head_dim_k] is not contiguous for dims
    struct ggml_tensor * q_transposed = ggml_transpose(ctx, q_base);
    assert(!ggml_is_contiguous(q_transposed));

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = 4;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = n_queries;

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_transposed, std::move(snap), op_handle);
    assert(out_tensor != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Must be rejected because Q is not contiguous
    assert(!op_handle->succeeded());
    assert(op_handle->status() == xkv_read_status::invalid_argument);
    assert(op_handle->error_message().find("contiguous") != std::string::npos);

    ggml_free(ctx);
    std::cout << "test_graph_non_contiguous_strided_q PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 11: Unequal / variable DDVR group counts and offsets across queries
// ----------------------------------------------------------------------------
static void test_graph_variable_ddvr_groups_and_offsets() {
    std::cout << "Running test_graph_variable_ddvr_groups_and_offsets..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t rank = 16;
    const uint32_t n_rows = 6;
    const uint32_t n_queries = 2;

    std::vector<float> a_k, b_k, a_v, b_v;
    auto seg = create_test_segment_f32(store, n_rows, rank, rank, 4 * head_dim_k, 4 * head_dim_v, 800, a_k, b_k, a_v, b_v);

    // Query 0 has 1 DDVR group
    // Query 1 has 2 DDVR groups
    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = n_queries;
    snap->query_ddvr_group_counts = { 1, 2 };
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;

    // Hot rows:
    // Row 0: group 0 (visible to Q0 and Q1)
    // Row 1: group 1 (only visible to Q1)
    xkv_graph_snapshot::hot_row_data hd0;
    hd0.row_index = 0;
    hd0.storage_pos = n_rows;
    hd0.group_index = 0;
    hd0.is_valid = true;
    hd0.query_visibility = { true, true };
    hd0.k_data = generate_deterministic_floats(head_dim_k, 810);
    hd0.v_data = generate_deterministic_floats(head_dim_v, 820);
    snap->hot_data.push_back(std::move(hd0));

    xkv_graph_snapshot::hot_row_data hd1;
    hd1.row_index = 1;
    hd1.storage_pos = n_rows + 1;
    hd1.group_index = 1;
    hd1.is_valid = true;
    hd1.query_visibility = { false, true }; // only Q1 can see group 1
    hd1.k_data = generate_deterministic_floats(head_dim_k, 830);
    hd1.v_data = generate_deterministic_floats(head_dim_v, 840);
    snap->hot_data.push_back(std::move(hd1));

    // Factored segment view: all rows group 0
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
    snap->segment_views.push_back(std::move(view));

    // Total Q elements needed: (1 + 2) * head_dim_k = 3 * 16 = 48 floats
    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q0_g0 = generate_deterministic_floats(head_dim_k, 850);
    std::vector<float> q1_g0 = generate_deterministic_floats(head_dim_k, 860);
    std::vector<float> q1_g1 = generate_deterministic_floats(head_dim_k, 870);

    std::vector<float> q_tensor_data;
    q_tensor_data.insert(q_tensor_data.end(), q0_g0.begin(), q0_g0.end());
    q_tensor_data.insert(q_tensor_data.end(), q1_g0.begin(), q1_g0.end());
    q_tensor_data.insert(q_tensor_data.end(), q1_g1.begin(), q1_g1.end());

    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q_tensor_data.size());
    std::memcpy(q_tensor->data, q_tensor_data.data(), q_tensor_data.size() * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    // Verify against individual reader batch run
    xkv_query_input qi0;
    qi0.query_index = 0;
    qi0.head_dim_k = head_dim_k;
    qi0.head_dim_v = head_dim_v;
    qi0.q_groups = { q0_g0 };

    xkv_query_input qi1;
    qi1.query_index = 1;
    qi1.head_dim_k = head_dim_k;
    qi1.head_dim_v = head_dim_v;
    qi1.q_groups = { q1_g0, q1_g1 };

    xkv_segment_read_view view2;
    view2.pin = store.pin_segment(seg->segment_id);
    view2.segment_version_id = seg->segment_version;
    view2.storage_generation = 1;
    view2.owning_layer = 0;
    view2.factor_group_index = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view2.selected_rows.push_back(i);
        view2.storage_positions.push_back(i);
        view2.group_indices.push_back(0);
    }

    std::vector<xkv_hot_row> hot_rows2 = op_handle->snapshot()->build_hot_rows();
    std::vector<xkv_segment_read_view> views2;
    views2.push_back(std::move(view2));
    auto batch_res = xkv_read_attention_batch({qi0, qi1}, hot_rows2, views2, {}, nullptr, store.current_stamp(), &store);
    assert(batch_res.status == xkv_read_status::success);

    std::vector<float> actual_q0(head_dim_v);
    std::vector<float> actual_q1(head_dim_v);
    const float * out_ptr = static_cast<const float *>(out_tensor->data);
    std::memcpy(actual_q0.data(), out_ptr, head_dim_v * sizeof(float));
    std::memcpy(actual_q1.data(), out_ptr + head_dim_v, head_dim_v * sizeof(float));

    assert(vectors_approx_equal(actual_q0, batch_res.per_query[0].output, 1e-4f));
    assert(vectors_approx_equal(actual_q1, batch_res.per_query[1].output, 1e-4f));

    // Negative case: hot row visible to query 0 with group index 1 (which query 0 does not have)
    auto snap_neg = std::make_unique<xkv_graph_snapshot>();
    snap_neg->head_dim_k = head_dim_k;
    snap_neg->head_dim_v = head_dim_v;
    snap_neg->n_q_heads = 1;
    snap_neg->n_queries = n_queries;
    snap_neg->query_ddvr_group_counts = { 1, 2 };
    snap_neg->expected_stamp = store.current_stamp();
    snap_neg->store = &store;

    xkv_graph_snapshot::hot_row_data hd_bad;
    hd_bad.row_index = 0;
    hd_bad.group_index = 1; // Q0 only has 1 group (group 0)
    hd_bad.query_visibility = { true, true }; // visible to Q0 -> must fail validation!
    hd_bad.k_data = generate_deterministic_floats(head_dim_k, 991);
    hd_bad.v_data = generate_deterministic_floats(head_dim_v, 992);
    snap_neg->hot_data.push_back(std::move(hd_bad));

    std::shared_ptr<xkv_graph_op_handle> op_handle_neg;
    struct ggml_tensor * out_neg = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap_neg), op_handle_neg);
    struct ggml_cgraph * gf_neg = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_neg, out_neg);
    ggml_graph_compute_with_ctx(ctx, gf_neg, 1);

    assert(!op_handle_neg->succeeded());
    assert(op_handle_neg->status() == xkv_read_status::invalid_argument);

    ggml_free(ctx);
    std::cout << "test_graph_variable_ddvr_groups_and_offsets PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 12: Hot row validation, dimension mismatch, and explicit bound enforcement
// ----------------------------------------------------------------------------
static void test_graph_hot_row_validation_and_bounds() {
    std::cout << "Running test_graph_hot_row_validation_and_bounds..." << std::endl;

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t n_queries = 1;

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 911);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    // Case A: Hot row dimension mismatch (k_data has 15 floats instead of 16)
    {
        auto snap = std::make_unique<xkv_graph_snapshot>();
        snap->head_dim_k = head_dim_k;
        snap->head_dim_v = head_dim_v;
        snap->n_queries = n_queries;

        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = 0;
        hd.k_data = generate_deterministic_floats(head_dim_k - 1, 912); // dimension mismatch!
        hd.v_data = generate_deterministic_floats(head_dim_v, 913);
        snap->hot_data.push_back(std::move(hd));

        std::shared_ptr<xkv_graph_op_handle> op_handle;
        struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_tensor);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        assert(!op_handle->succeeded());
        assert(op_handle->status() == xkv_read_status::invalid_argument);
        assert(op_handle->error_message().find("head_dim_k") != std::string::npos);
    }

    // Case B: max_hot_rows bound exceeded
    {
        auto snap = std::make_unique<xkv_graph_snapshot>();
        snap->head_dim_k = head_dim_k;
        snap->head_dim_v = head_dim_v;
        snap->n_queries = n_queries;
        snap->max_hot_rows = 2; // bound is 2

        for (uint32_t i = 0; i < 3; ++i) { // provide 3 rows
            xkv_graph_snapshot::hot_row_data hd;
            hd.row_index = i;
            hd.k_data = generate_deterministic_floats(head_dim_k, 920 + i);
            hd.v_data = generate_deterministic_floats(head_dim_v, 930 + i);
            snap->hot_data.push_back(std::move(hd));
        }

        std::shared_ptr<xkv_graph_op_handle> op_handle;
        struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_tensor);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        assert(!op_handle->succeeded());
        assert(op_handle->status() == xkv_read_status::invalid_argument);
        assert(op_handle->error_message().find("max_hot_rows") != std::string::npos);
    }

    // Case C: max_hot_bytes bound exceeded
    {
        auto snap = std::make_unique<xkv_graph_snapshot>();
        snap->head_dim_k = head_dim_k;
        snap->head_dim_v = head_dim_v;
        snap->n_queries = n_queries;
        // Each row is (16 + 16) * 4 = 128 bytes. Bound is 100 bytes.
        snap->max_hot_bytes = 100;

        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = 0;
        hd.k_data = generate_deterministic_floats(head_dim_k, 940);
        hd.v_data = generate_deterministic_floats(head_dim_v, 950);
        snap->hot_data.push_back(std::move(hd));

        std::shared_ptr<xkv_graph_op_handle> op_handle;
        struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_tensor);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        assert(!op_handle->succeeded());
        assert(op_handle->status() == xkv_read_status::invalid_argument);
        assert(op_handle->error_message().find("max_hot_bytes") != std::string::npos);
    }

    // Case D: query_visibility size mismatch
    {
        auto snap = std::make_unique<xkv_graph_snapshot>();
        snap->head_dim_k = head_dim_k;
        snap->head_dim_v = head_dim_v;
        snap->n_queries = 2; // 2 queries

        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = 0;
        hd.query_visibility = { true }; // size 1 != 2
        hd.k_data = generate_deterministic_floats(head_dim_k, 960);
        hd.v_data = generate_deterministic_floats(head_dim_v, 970);
        snap->hot_data.push_back(std::move(hd));

        struct ggml_tensor * q_2q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k * 2);
        std::shared_ptr<xkv_graph_op_handle> op_handle;
        struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_2q, std::move(snap), op_handle);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_tensor);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        assert(!op_handle->succeeded());
        assert(op_handle->status() == xkv_read_status::invalid_argument);
        assert(op_handle->error_message().find("query_visibility") != std::string::npos);
    }

    ggml_free(ctx);
    std::cout << "test_graph_hot_row_validation_and_bounds PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 13: Explicit dependency graph ordering and validation
// ----------------------------------------------------------------------------
static void test_graph_explicit_dependency_ordering() {
    std::cout << "Running test_graph_explicit_dependency_ordering..." << std::endl;

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint32_t n_queries = 1;

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Source 0: Q
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 1001);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    // Source 1: Dependency tensor produced by earlier graph op (e.g. ggml_scale)
    struct ggml_tensor * dep_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    std::vector<float> dep_data = generate_deterministic_floats(8, 1002);
    std::memcpy(dep_input->data, dep_data.data(), 8 * sizeof(float));
    struct ggml_tensor * dep_tensor = ggml_scale(ctx, dep_input, 2.0f);

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim_k;
    snap->head_dim_v = head_dim_v;
    snap->n_q_heads = 1;
    snap->n_queries = n_queries;

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(
        ctx,
        q_tensor,
        std::move(snap),
        op_handle,
        { dep_tensor } // explicit dependency
    );
    assert(out_tensor != nullptr);
    assert(out_tensor->src[0] == q_tensor);
    assert(out_tensor->src[1] == dep_tensor);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);

    // Verify graph ordering: dep_tensor must be computed BEFORE out_tensor
    int idx_dep = -1;
    int idx_out = -1;
    int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == dep_tensor) idx_dep = i;
        if (node == out_tensor) idx_out = i;
    }
    assert(idx_dep >= 0);
    assert(idx_out >= 0);
    assert(idx_dep < idx_out);

    // Compute graph
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    // Verify dep_tensor was indeed computed
    const float * dep_out = static_cast<const float *>(dep_tensor->data);
    for (size_t i = 0; i < 8; ++i) {
        assert(approx_equal(dep_out[i], dep_data[i] * 2.0f, 1e-4f));
    }

    ggml_free(ctx);
    std::cout << "test_graph_explicit_dependency_ordering PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 14: Userdata lifetime safety
// ----------------------------------------------------------------------------
static void test_graph_userdata_lifetime() {
    std::cout << "Running test_graph_userdata_lifetime..." << std::endl;

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    std::vector<float> q_data = generate_deterministic_floats(16, 2001);
    std::memcpy(q_tensor->data, q_data.data(), 16 * sizeof(float));

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = 16;
    snap->head_dim_v = 16;
    snap->n_q_heads = 1;
    snap->n_queries = 1;

    // Build graph op and obtain shared_ptr handle
    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);
    assert(op_handle != nullptr);

    // Simulate runtime graph-input owner (e.g. llm_graph_input_xkv) holding a shared_ptr
    std::shared_ptr<xkv_graph_op_handle> graph_owner = op_handle;

    // Caller alias is destroyed or moved away before compute
    op_handle.reset();
    assert(op_handle == nullptr);

    // Compute graph while retained graph_owner keeps handle alive
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Must have computed successfully
    assert(graph_owner->succeeded());
    const float * out_ptr = static_cast<const float *>(out_tensor->data);
    for (size_t i = 0; i < 16; ++i) {
        assert(out_ptr[i] == 0.0f);
    }

    // Null userdata check: invoking callback with null userdata safely zeroes dst
    std::memset(out_tensor->data, 0xAA, ggml_nbytes(out_tensor));
    xkv_graph_op_handle::ggml_custom_op_callback(out_tensor, 0, 1, nullptr);
    for (size_t i = 0; i < 16; ++i) {
        assert(out_ptr[i] == 0.0f);
    }

    // Release final owner only after graph execution has completely finished
    graph_owner.reset();
    assert(graph_owner == nullptr);

    ggml_free(ctx);
    std::cout << "test_graph_userdata_lifetime PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 15: Independent fixed golden output (hand-calculated reference)
// ----------------------------------------------------------------------------
static void test_graph_independent_fixed_golden() {
    std::cout << "Running test_graph_independent_fixed_golden..." << std::endl;

    // Fixed parameters:
    // head_dim_k = 2, head_dim_v = 2, n_q_heads = 1, n_queries = 1
    // Q = [1.0, 2.0]
    // Scale = 1.0 (override)
    // Hot Row 0: K = [0.5, 0.5], V = [3.0, 4.0] -> dot = 1.0*0.5 + 2.0*0.5 = 1.5
    // Hot Row 1: K = [1.0, -0.5], V = [5.0, 6.0] -> dot = 1.0*1.0 + 2.0*(-0.5) = 0.0
    // Softmax:
    // max_score = 1.5
    // row 0: exp(1.5 - 1.5) = 1.0 -> weights = 1.0 * [3.0, 4.0] = [3.0, 4.0]
    // row 1: exp(0.0 - 1.5) = exp(-1.5) = 0.22313016 -> weights = 0.22313016 * [5.0, 6.0] = [1.1156508, 1.33878096]
    // sum_exp = 1.0 + 0.22313016 = 1.22313016
    // out[0] = (3.0 + 1.1156508) / 1.22313016 = 4.1156508 / 1.22313016 = 3.3648512
    // out[1] = (4.0 + 1.33878096) / 1.22313016 = 5.33878096 / 1.22313016 = 4.3648512
    const float expected_v0 = 3.3648512f;
    const float expected_v1 = 4.3648512f;

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = 2;
    snap->head_dim_v = 2;
    snap->n_q_heads = 1;
    snap->n_queries = 1;
    snap->scale = 1.0f; // override scale to 1.0

    xkv_graph_snapshot::hot_row_data r0;
    r0.row_index = 0;
    r0.storage_pos = 0;
    r0.storage_generation = 1;
    r0.group_index = 0;
    r0.is_valid = true;
    r0.k_data = { 0.5f, 0.5f };
    r0.v_data = { 3.0f, 4.0f };
    snap->hot_data.push_back(std::move(r0));

    xkv_graph_snapshot::hot_row_data r1;
    r1.row_index = 1;
    r1.storage_pos = 1;
    r1.storage_generation = 1;
    r1.group_index = 0;
    r1.is_valid = true;
    r1.k_data = { 1.0f, -0.5f };
    r1.v_data = { 5.0f, 6.0f };
    snap->hot_data.push_back(std::move(r1));

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float * q_ptr = static_cast<float *>(q_tensor->data);
    q_ptr[0] = 1.0f;
    q_ptr[1] = 2.0f;

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    assert(op_handle->succeeded());

    const float * actual_out = static_cast<const float *>(out_tensor->data);
    assert(approx_equal(actual_out[0], expected_v0, 1e-4f));
    assert(approx_equal(actual_out[1], expected_v1, 1e-4f));

    ggml_free(ctx);
    std::cout << "test_graph_independent_fixed_golden PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 16: Hot payload store location, generation, and state validation
// ----------------------------------------------------------------------------
static void test_graph_hot_store_bindings_validation() {
    std::cout << "Running test_graph_hot_store_bindings_validation..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim_k = 16;
    const uint32_t head_dim_v = 16;
    const uint64_t pid = 77701;
    const uint32_t slot_row = 3;
    const uint64_t gen = 5;

    // Register hot payload in store
    store.register_hot_payload(pid, slot_row, gen, xkv_state::hot_committed);

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_k);
    std::vector<float> q_data = generate_deterministic_floats(head_dim_k, 555);
    std::memcpy(q_tensor->data, q_data.data(), head_dim_k * sizeof(float));

    // Case A: Correct bindings -> Success
    {
        auto snap = std::make_unique<xkv_graph_snapshot>();
        snap->head_dim_k = head_dim_k;
        snap->head_dim_v = head_dim_v;
        snap->n_q_heads = 1;
        snap->n_queries = 1;
        snap->expected_stamp = store.current_stamp();
        snap->store = &store;

        xkv_graph_snapshot::hot_row_data hd;
        hd.payload_id = pid;
        hd.row_index = slot_row;
        hd.storage_pos = 10;
        hd.storage_generation = gen;
        hd.expected_state = xkv_state::hot_committed;
        hd.k_data = generate_deterministic_floats(head_dim_k, 601);
        hd.v_data = generate_deterministic_floats(head_dim_v, 602);
        snap->hot_data.push_back(std::move(hd));

        std::shared_ptr<xkv_graph_op_handle> op_handle;
        struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_tensor);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        assert(op_handle->succeeded());
    }

    // Case B: Generation mismatch -> Fail-closed
    {
        auto snap = std::make_unique<xkv_graph_snapshot>();
        snap->head_dim_k = head_dim_k;
        snap->head_dim_v = head_dim_v;
        snap->n_q_heads = 1;
        snap->n_queries = 1;
        snap->expected_stamp = store.current_stamp();
        snap->store = &store;

        xkv_graph_snapshot::hot_row_data hd;
        hd.payload_id = pid;
        hd.row_index = slot_row;
        hd.storage_pos = 10;
        hd.storage_generation = gen + 1; // generation mismatch!
        hd.expected_state = xkv_state::hot_committed;
        hd.k_data = generate_deterministic_floats(head_dim_k, 603);
        hd.v_data = generate_deterministic_floats(head_dim_v, 604);
        snap->hot_data.push_back(std::move(hd));

        std::shared_ptr<xkv_graph_op_handle> op_handle;
        struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_tensor);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        assert(!op_handle->succeeded());
        assert(op_handle->status() == xkv_read_status::invalid_argument);
        assert(op_handle->error_message().find("generation mismatch") != std::string::npos);
    }

    // Case C: Row mismatch -> Fail-closed
    {
        auto snap = std::make_unique<xkv_graph_snapshot>();
        snap->head_dim_k = head_dim_k;
        snap->head_dim_v = head_dim_v;
        snap->n_q_heads = 1;
        snap->n_queries = 1;
        snap->expected_stamp = store.current_stamp();
        snap->store = &store;

        xkv_graph_snapshot::hot_row_data hd;
        hd.payload_id = pid;
        hd.row_index = slot_row + 1; // row mismatch!
        hd.storage_pos = 10;
        hd.storage_generation = gen;
        hd.expected_state = xkv_state::hot_committed;
        hd.k_data = generate_deterministic_floats(head_dim_k, 605);
        hd.v_data = generate_deterministic_floats(head_dim_v, 606);
        snap->hot_data.push_back(std::move(hd));

        std::shared_ptr<xkv_graph_op_handle> op_handle;
        struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_tensor);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        assert(!op_handle->succeeded());
        assert(op_handle->status() == xkv_read_status::invalid_argument);
        assert(op_handle->error_message().find("row mismatch") != std::string::npos);
    }

    ggml_free(ctx);
    std::cout << "test_graph_hot_store_bindings_validation PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// Test 17: Second factor group exact resolution (no first-group fallback)
// ----------------------------------------------------------------------------
static void test_graph_second_group_exact_resolution() {
    std::cout << "Running test_graph_second_group_exact_resolution..." << std::endl;

    llama_cparams cparams = make_graph_ref_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t head_dim = 16;
    const uint32_t rank = 8;
    const uint32_t n_rows = 6;

    // Two-group bundle with distinct B data per group. Group 1 owns layers {2, 3}
    // with contiguous offsets {0, 8} summing to total_dim 16.
    xkv_factor_group_payload g0;
    g0.group_index = 0;
    g0.owning_layers = {0, 1};
    g0.rank_k = rank;
    g0.rank_v = rank;
    g0.total_dim_k = head_dim;
    g0.total_dim_v = head_dim;
    g0.layer_feature_offsets_k = {0, head_dim / 2};
    g0.layer_feature_dims_k = {head_dim / 2, head_dim / 2};
    g0.layer_feature_offsets_v = {0, head_dim / 2};
    g0.layer_feature_dims_v = {head_dim / 2, head_dim / 2};
    std::vector<float> a_k0 = generate_deterministic_floats(n_rows * rank, 5101);
    std::vector<float> b_k0 = generate_deterministic_floats(head_dim * rank, 5102);
    std::vector<float> a_v0 = generate_deterministic_floats(n_rows * rank, 5103);
    std::vector<float> b_v0 = generate_deterministic_floats(head_dim * rank, 5104);
    g0.a_k = encode_matrix(make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 5101), a_k0.data(), a_k0.size());
    g0.set_b_k(encode_matrix(make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim, rank}, 0, 5102), b_k0.data(), b_k0.size()));
    g0.a_v = encode_matrix(make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 5103), a_v0.data(), a_v0.size());
    g0.set_b_v(encode_matrix(make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim, rank}, 0, 5104), b_v0.data(), b_v0.size()));

    xkv_factor_group_payload g1;
    g1.group_index = 1;
    g1.owning_layers = {2, 3};
    g1.rank_k = rank;
    g1.rank_v = rank;
    g1.total_dim_k = head_dim;
    g1.total_dim_v = head_dim;
    g1.layer_feature_offsets_k = {0, head_dim / 2};
    g1.layer_feature_dims_k = {head_dim / 2, head_dim / 2};
    g1.layer_feature_offsets_v = {0, head_dim / 2};
    g1.layer_feature_dims_v = {head_dim / 2, head_dim / 2};
    std::vector<float> a_k1 = generate_deterministic_floats(n_rows * rank, 5201);
    std::vector<float> b_k1 = generate_deterministic_floats(head_dim * rank, 5202);
    std::vector<float> a_v1 = generate_deterministic_floats(n_rows * rank, 5203);
    std::vector<float> b_v1 = generate_deterministic_floats(head_dim * rank, 5204);
    g1.a_k = encode_matrix(make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 5201), a_k1.data(), a_k1.size());
    g1.set_b_k(encode_matrix(make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim, rank}, 0, 5202), b_k1.data(), b_k1.size()));
    g1.a_v = encode_matrix(make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank}, 0, 5203), a_v1.data(), a_v1.size());
    g1.set_b_v(encode_matrix(make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {head_dim, rank}, 0, 5204), b_v1.data(), b_v1.size()));

    auto seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g0, g1}
    );
    seg->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(seg->groups);

    std::vector<uint64_t> pids(n_rows);
    std::vector<uint64_t> gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) {
        pids[i] = 52000 + i + 1;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }
    assert(store.mark_seal_candidates(pids));
    std::string pub_err;
    assert(store.publish_candidate(seg, pids, gens, &pub_err));

    // Exact lookup must resolve owning_layer 2 to group 1, never group 0.
    const auto * resolved = seg->find_group_for_layer(2);
    assert(resolved != nullptr);
    assert(resolved->group_index == 1);
    assert(seg->find_group(1) == resolved);

    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = head_dim;
    snap->head_dim_v = head_dim;
    snap->n_q_heads = 1;
    snap->n_queries = 1;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;

    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 2;
    view.factor_group_index = 1;
    for (uint32_t i = 0; i < n_rows; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    snap->segment_views.push_back(std::move(view));

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> q_data = generate_deterministic_floats(head_dim, 5299);
    struct ggml_tensor * q_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    std::memcpy(q_tensor->data, q_data.data(), head_dim * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> op_handle;
    struct ggml_tensor * out_tensor = xkv_build_graph_attention_ref(ctx, q_tensor, std::move(snap), op_handle);
    assert(out_tensor != nullptr);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(op_handle->succeeded());

    // Dense oracle from group-1 streams only: a first-group fallback would mismatch.
    std::vector<std::vector<float>> cold_keys(n_rows, std::vector<float>(head_dim, 0.0f));
    std::vector<std::vector<float>> cold_values(n_rows, std::vector<float>(head_dim, 0.0f));
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t f = 0; f < head_dim; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                cold_keys[r][f] += a_k1[r * rank + k] * b_k1[f * rank + k];
            }
        }
        for (uint32_t f = 0; f < head_dim; ++f) {
            for (uint32_t k = 0; k < rank; ++k) {
                cold_values[r][f] += a_v1[r * rank + k] * b_v1[f * rank + k];
            }
        }
    }
    xkv_query_input qi;
    qi.query_index = 0;
    qi.n_q_heads = 1;
    qi.head_dim_k = head_dim;
    qi.head_dim_v = head_dim;
    qi.q_vec = q_data;
    std::vector<uint32_t> groups(n_rows, 0);
    std::vector<float> expected = xkv_dense_attention_reference(qi, {}, cold_keys, cold_values, groups, {});
    std::vector<float> actual(head_dim);
    std::memcpy(actual.data(), out_tensor->data, head_dim * sizeof(float));
    assert(vectors_approx_equal(actual, expected, 1e-4f));

    ggml_free(ctx);
    std::cout << "test_graph_second_group_exact_resolution PASSED" << std::endl;
}

int main() {
    std::cout << "=== Running test-xkv-graph-ref ===" << std::endl;

    test_graph_hot_and_factored();
    test_graph_ddvr_groups();
    test_graph_dk_neq_dv_softcap_sink();
    test_graph_sr_csr_and_future_isolation();
    test_graph_empty_masked_finite_zero();
    test_graph_stale_stamp_fail_closed();
    test_graph_workspace_limit_fail_closed();
    test_graph_uncomputed_initial_status();
    test_graph_malformed_truncated_q();
    test_graph_non_contiguous_strided_q();
    test_graph_variable_ddvr_groups_and_offsets();
    test_graph_hot_row_validation_and_bounds();
    test_graph_explicit_dependency_ordering();
    test_graph_userdata_lifetime();
    test_graph_independent_fixed_golden();
    test_graph_hot_store_bindings_validation();

    test_graph_second_group_exact_resolution();

    std::cout << "=== All test-xkv-graph-ref tests PASSED successfully ===" << std::endl;
    return 0;
}
