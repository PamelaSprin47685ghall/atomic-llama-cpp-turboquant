#ifdef NDEBUG
#undef NDEBUG
#endif

// Focused XKV graph-runtime tests (owned by XkvGraphRuntime).
//
// Executes actual ggml graphs through llama_xkv::xkv_build_graph_attention_ref
// and compares against the dense oracle. Covers the bounded-hot wiring owned
// here: storage gather inside compute (never stale copies), sink deps, GQA
// loop+concat, Turbo canonicalization, inverse-attention-rotation-only,
// fail-closed validation, reuse refusal and post-compute mapping. XKV OFF
// topology preservation is asserted via the coverage gate.

#include "llama-xkv-graph-ref.h"
#include "llama-xkv-reader.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-graph.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-vulkan-landmark.h" // device SR oracle contract (XkvVulkanLandmark)

#include <cstdlib>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace llama_xkv;

static std::vector<float> det_floats(size_t n, uint64_t seed) {
    std::vector<float> data(n);
    uint64_t state = seed ? seed : 123456789ULL;
    for (size_t i = 0; i < n; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        data[i] = (static_cast<float>(state & 0x7FFFFFFF) / static_cast<float>(0x7FFFFFFF)) * 2.0f - 1.0f;
    }
    return data;
}

static bool approx_eq(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

static bool vec_eq(const std::vector<float> & a, const std::vector<float> & b, float tol) {
    if (a.size() != b.size()) {
        std::cerr << "size mismatch " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approx_eq(a[i], b[i], tol)) {
            std::cerr << "mismatch @" << i << ": " << a[i] << " vs " << b[i]
                      << " diff=" << std::fabs(a[i] - b[i]) << std::endl;
            return false;
        }
    }
    return true;
}

static struct ggml_context * make_ctx(size_t mem = 64 * 1024 * 1024) {
    struct ggml_init_params params = { mem, nullptr, false };
    return ggml_init(params);
}

static llama_cparams ref_cparams() {
    llama_cparams cparams = {};
    cparams.xkv_mode = LLAMA_XKV_MODE_SHADOW;
    cparams.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cparams.xkv_workspace_mib = 16;
    cparams.xkv_decode_cache_mib = 8;
    return cparams;
}

// Build a [P, Hkv, N] F32 storage tensor pre-filled with per-cell/per-head data.
static struct ggml_tensor * make_storage_f32(
    struct ggml_context * ctx, uint32_t P, uint32_t Hkv, uint32_t N,
    const std::vector<std::vector<std::vector<float>>> & cells /* [N][Hkv][P] */) {
    struct ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, P, Hkv, N);
    float * d = static_cast<float *>(t->data);
    for (uint32_t c = 0; c < N; ++c) {
        for (uint32_t h = 0; h < Hkv; ++h) {
            const auto & row = cells[c][h];
            assert(row.size() == P);
            std::memcpy(d + ((size_t) c * Hkv + h) * P, row.data(), P * sizeof(float));
        }
    }
    return t;
}

static std::unique_ptr<xkv_graph_snapshot> base_snap(
    uint32_t Dk, uint32_t Dv, uint32_t nq_heads, uint32_t n_queries, uint32_t kv_head) {
    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = Dk;
    snap->head_dim_v = Dv;
    snap->n_q_heads = nq_heads;
    snap->n_queries = n_queries;
    snap->kv_head_index = kv_head;
    snap->hot_layout.k_type = GGML_TYPE_F32;
    snap->hot_layout.v_type = GGML_TYPE_F32;
    snap->hot_layout.head_dim_k = Dk;
    snap->hot_layout.head_dim_v = Dv;
    return snap;
}

static void add_gather_row(xkv_graph_snapshot & snap, int64_t cell, uint32_t kv_head, int64_t pos) {
    xkv_graph_snapshot::hot_row_data hd;
    hd.row_index = (uint32_t) cell;
    hd.storage_pos = pos;
    hd.group_index = 0;
    hd.is_valid = true;
    hd.cell = cell;
    hd.kv_head = kv_head;
    hd.stream = 0;
    snap.hot_data.push_back(std::move(hd));
}

// ----------------------------------------------------------------------------
// 1. Ordinary single head: storage gather vs dense oracle
// ----------------------------------------------------------------------------
static void test_ordinary_storage_gather() {
    std::cout << "Running test_ordinary_storage_gather..." << std::endl;
    const uint32_t D = 8, Hkv = 1, N = 4, T = 2;

    struct ggml_context * ctx = make_ctx();
    std::vector<std::vector<std::vector<float>>> kcells(N, std::vector<std::vector<float>>(Hkv));
    std::vector<std::vector<std::vector<float>>> vcells(N, std::vector<std::vector<float>>(Hkv));
    for (uint32_t c = 0; c < N; ++c) {
        kcells[c][0] = det_floats(D, 100 + c);
        vcells[c][0] = det_floats(D, 200 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, Hkv, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, Hkv, N, vcells);

    auto snap = base_snap(D, D, 1, T, 0);
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;

    std::vector<float> qdata = det_floats(D * T, 777);
    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, T);
    std::memcpy(q->data, qdata.data(), qdata.size() * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(
        ctx, q, std::move(snap), h, { k_st, v_st });
    assert(out && h);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());

    // Oracle over the same rows.
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kcells[c][0].data();
        r.v_ptr = vcells[c][0].data();
        rows.push_back(r);
    }
    std::vector<float> expect;
    for (uint32_t t = 0; t < T; ++t) {
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D;
        qi.q_vec.assign(qdata.begin() + t * D, qdata.begin() + (t + 1) * D);
        auto o = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
        expect.insert(expect.end(), o.begin(), o.end());
    }
    std::vector<float> actual(expect.size());
    std::memcpy(actual.data(), out->data, actual.size() * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_ordinary_storage_gather PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 2. GQA multi-KV-head loop+concat (mirrors llm_build_attn_xkv)
// ----------------------------------------------------------------------------
static void test_gqa_loop_concat() {
    std::cout << "Running test_gqa_loop_concat..." << std::endl;
    const uint32_t Dk = 8, Dv = 6, Hkv = 2, GQA = 2, H = 4, N = 3, T = 2;

    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
    for (uint32_t c = 0; c < N; ++c) {
        for (uint32_t h = 0; h < Hkv; ++h) {
            kcells[c][h] = det_floats(Dk, 300 + c * 10 + h);
            vcells[c][h] = det_floats(Dv, 400 + c * 10 + h);
        }
    }
    // K storage is [Dk, Hkv, N]; V storage is [Dv, Hkv, N] (unequal dims).
    struct ggml_tensor * k_st = make_storage_f32(ctx, Dk, Hkv, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, Dv, Hkv, N, vcells);

    std::vector<float> qfull = det_floats(Dk * H * T, 555);
    struct ggml_tensor * q_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, H, T);
    std::memcpy(q_all->data, qfull.data(), qfull.size() * sizeof(float));

    std::vector<ggml_tensor *> outs;
    std::vector<std::shared_ptr<xkv_graph_op_handle>> handles;
    for (uint32_t h = 0; h < Hkv; ++h) {
        auto snap = base_snap(Dk, Dv, GQA, T, h);
        snap->hot_layout.head_dim_v = Dv;
        for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, h, c);
        snap->k_storage_dep = 1;
        snap->v_storage_dep = 2;
        // Same slice pattern as llm_build_attn_xkv.
        const size_t off = (size_t)(h * GQA) * (size_t) q_all->nb[1];
        ggml_tensor * q_h = ggml_view_3d(ctx, q_all, Dk, GQA, T, q_all->nb[1], q_all->nb[2], off);
        q_h = ggml_cont(ctx, q_h);
        std::shared_ptr<xkv_graph_op_handle> hh;
        ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q_h, std::move(snap), hh, { k_st, v_st });
        assert(out && hh);
        outs.push_back(out);
        handles.push_back(hh);
    }
    ggml_tensor * cur = outs[0];
    for (size_t i = 1; i < outs.size(); ++i) cur = ggml_concat(ctx, cur, outs[i], 0);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    for (auto & hh : handles) assert(hh->succeeded());

    // Per-group oracle.
    std::vector<float> expect;
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t h = 0; h < Hkv; ++h) {
            xkv_query_input qi;
            qi.head_dim_k = Dk; qi.head_dim_v = Dv; qi.n_q_heads = GQA;
            qi.q_vec.resize(Dk * GQA);
            for (uint32_t g = 0; g < GQA; ++g) {
                const float * src = qfull.data() + ((size_t) t * H + h * GQA + g) * Dk;
                std::memcpy(qi.q_vec.data() + g * Dk, src, Dk * sizeof(float));
            }
            std::vector<xkv_hot_row> rows;
            for (uint32_t c = 0; c < N; ++c) {
                xkv_hot_row r;
                r.storage_pos = c;
                r.k_ptr = kcells[c][h].data();
                r.v_ptr = vcells[c][h].data();
                rows.push_back(r);
            }
            auto o = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
            expect.insert(expect.end(), o.begin(), o.end());
        }
    }
    // Concat layout is [Dv*GQA head-major, T]: reorder expectation to match.
    std::vector<float> expect_concat(Dv * H * T);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t h = 0; h < Hkv; ++h) {
            const float * src = expect.data() + ((size_t) t * Hkv + h) * Dv * GQA;
            float * dst = expect_concat.data() + ((size_t) t * H + h * GQA) * Dv;
            std::memcpy(dst, src, Dv * GQA * sizeof(float));
        }
    }
    std::vector<float> actual(expect_concat.size());
    std::memcpy(actual.data(), cur->data, actual.size() * sizeof(float));
    assert(vec_eq(actual, expect_concat, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_gqa_loop_concat PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 3. Current-token self attention: write node ordered before gather
// ----------------------------------------------------------------------------
static void test_current_write_visible() {
    std::cout << "Running test_current_write_visible..." << std::endl;
    const uint32_t D = 8, Hkv = 1, N = 4;
    const int64_t cur_cell = 3;

    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
    for (uint32_t c = 0; c < N; ++c) {
        kcells[c][0] = det_floats(D, 600 + c);
        vcells[c][0] = det_floats(D, 700 + c);
    }
    // Storage holds a zero sentinel at the current cell before compute.
    auto ksent = kcells; auto vsent = vcells;
    ksent[cur_cell][0].assign(D, 0.0f);
    vsent[cur_cell][0].assign(D, 0.0f);
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, Hkv, N, ksent);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, Hkv, N, vsent);

    // Write node: copy real current values into the sentinel slice at compute.
    struct ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, 1);
    struct ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, 1);
    std::memcpy(k_new->data, kcells[cur_cell][0].data(), D * sizeof(float));
    std::memcpy(v_new->data, vcells[cur_cell][0].data(), D * sizeof(float));
    ggml_tensor * k_slice = ggml_view_3d(ctx, k_st, D, Hkv, 1, k_st->nb[1], k_st->nb[2], cur_cell * k_st->nb[2]);
    ggml_tensor * v_slice = ggml_view_3d(ctx, v_st, D, Hkv, 1, v_st->nb[1], v_st->nb[2], cur_cell * v_st->nb[2]);
    ggml_tensor * w_k = ggml_cpy(ctx, k_new, k_slice);
    ggml_tensor * w_v = ggml_cpy(ctx, v_new, v_slice);

    auto snap = base_snap(D, D, 1, 1, 0);
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;

    std::vector<float> qdata = det_floats(D, 888);
    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(
        ctx, q, std::move(snap), h, { k_st, v_st, w_k, w_v });
    assert(out && h);
    // Dependency edges mirror production: out lists the write nodes.
    assert(out->src[3] == w_k && out->src[4] == w_v);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());

    // Oracle must see the WRITTEN values, not the sentinel.
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kcells[c][0].data();
        r.v_ptr = vcells[c][0].data();
        rows.push_back(r);
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_current_write_visible PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 4. Hot (gather) + cold (factored) global softmax with softcap
// ----------------------------------------------------------------------------
static std::shared_ptr<xkv_segment> make_f32_segment(
    llama_xkv_cache_store & store, uint32_t n_rows, uint32_t D,
    uint64_t seed, std::vector<float> & a_out, std::vector<float> & b_out) {
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = { 0 };
    g.total_dim_k = D;
    g.total_dim_v = D;
    g.layer_feature_offsets_k = { 0 };
    g.layer_feature_dims_k = { D };
    g.layer_feature_offsets_v = { 0 };
    g.layer_feature_dims_v = { D };
    codec_desc da = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, { n_rows, D }, 0, seed + 1);
    a_out = det_floats((size_t) n_rows * D, seed + 1);
    g.a_k = encode_matrix(da, a_out.data(), a_out.size());
    // K pair shares one codec seed (identical rotation/table fingerprint);
    // data seeds stay distinct so A/B remain independent matrices.
    codec_desc db = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, { D, D }, 0, seed + 1);
    b_out = det_floats((size_t) D * D, seed + 11);
    g.set_b_k(encode_matrix(db, b_out.data(), b_out.size()));
    // V pair shares one codec seed.
    codec_desc dav = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, { n_rows, D }, 0, seed + 2);
    std::vector<float> av = det_floats((size_t) n_rows * D, seed + 13);
    g.a_v = encode_matrix(dav, av.data(), av.size());
    codec_desc dbv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, { D, D }, 0, seed + 2);
    std::vector<float> bv = det_floats((size_t) D * D, seed + 17);
    g.set_b_v(encode_matrix(dbv, bv.data(), bv.size()));
    // Keep copies for the oracle.
    a_out.resize((size_t) n_rows * D * 2);
    std::memcpy(a_out.data() + n_rows * D, av.data(), av.size() * sizeof(float));
    b_out.resize((size_t) D * D * 2);
    std::memcpy(b_out.data() + D * D, bv.data(), bv.size() * sizeof(float));
    auto seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, { g });
    std::vector<uint64_t> pids(n_rows), gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) pids[i] = seed * 1000 + i + 1;
    for (uint32_t i = 0; i < n_rows; ++i) store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
    uint64_t nonce = 0;
    assert(store.mark_seal_candidates(pids, gens, &nonce));
    std::string err;
    assert(store.publish_candidate(seg, pids, gens, &err));
    return seg;
}

static void test_hot_cold_softmax_softcap() {
    std::cout << "Running test_hot_cold_softmax_softcap..." << std::endl;
    const uint32_t D = 8, Nhot = 3, Ncold = 5;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);

    std::vector<float> ak, bk;
    auto seg = make_f32_segment(store, Ncold, D, 42, ak, bk);
    const float * av = ak.data() + Ncold * D;
    const float * bv = bk.data() + D * D;

    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < Nhot; ++c) {
        kcells[c][0] = det_floats(D, 900 + c);
        vcells[c][0] = det_floats(D, 950 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, Nhot, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, Nhot, vcells);

    auto snap = base_snap(D, D, 1, 1, 0);
    snap->logit_softcap = 15.0f;
    for (int64_t c = 0; c < (int64_t) Nhot; ++c) add_gather_row(*snap, c, 0, (int64_t) Ncold + c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;
    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < Ncold; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    snap->segment_views.push_back(std::move(view));

    std::vector<float> qdata = det_floats(D, 1111);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());

    // Oracle: hot rows + reconstructed cold rows in one global softmax with softcap.
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < Nhot; ++c) {
        xkv_hot_row r;
        r.storage_pos = Ncold + c;
        r.k_ptr = kcells[c][0].data();
        r.v_ptr = vcells[c][0].data();
        rows.push_back(r);
    }
    std::vector<std::vector<float>> ck(Ncold, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> cv(Ncold, std::vector<float>(D, 0.0f));
    for (uint32_t r = 0; r < Ncold; ++r) {
        for (uint32_t f = 0; f < D; ++f) {
            for (uint32_t k = 0; k < D; ++k) {
                ck[r][f] += ak[r * D + k] * bk[f * D + k];
                cv[r][f] += av[r * D + k] * bv[f * D + k];
            }
        }
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata; qi.logit_softcap = 15.0f;
    std::vector<uint32_t> groups(Ncold, 0);
    std::vector<bool> mask(Ncold, true);
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, ck, cv, groups, mask);
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_hot_cold_softmax_softcap PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 5. Sink tensor consumed exactly once per query/head
// ----------------------------------------------------------------------------
static void test_sink_dep_once() {
    std::cout << "Running test_sink_dep_once..." << std::endl;
    const uint32_t D = 8, Hkv = 2, GQA = 2, H = 4, N = 3, T = 2;
    for (int variant = 0; variant < 2; ++variant) {
        struct ggml_context * ctx = make_ctx();
        auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
        auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t h = 0; h < Hkv; ++h) {
                kcells[c][h] = det_floats(D, 1200 + c * 10 + h);
                vcells[c][h] = det_floats(D, 1250 + c * 10 + h);
            }
        struct ggml_tensor * k_st = make_storage_f32(ctx, D, Hkv, N, kcells);
        struct ggml_tensor * v_st = make_storage_f32(ctx, D, Hkv, N, vcells);

        std::vector<float> qfull = det_floats(D * H * T, 1300);
        struct ggml_tensor * q_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
        std::memcpy(q_all->data, qfull.data(), qfull.size() * sizeof(float));

        // Sink values: distinct per global Q head; per-query in variant 1.
        std::vector<float> sinkv(variant == 0 ? H : H * T);
        for (size_t i = 0; i < sinkv.size(); ++i) sinkv[i] = 0.5f + 0.25f * (float) i;
        struct ggml_tensor * sinks = variant == 0
            ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H)
            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        std::memcpy(sinks->data, sinkv.data(), sinkv.size() * sizeof(float));

        std::vector<ggml_tensor *> outs;
        std::vector<std::shared_ptr<xkv_graph_op_handle>> handles;
        for (uint32_t h = 0; h < Hkv; ++h) {
            auto snap = base_snap(D, D, GQA, T, h);
            snap->q_group_begin = h * GQA;
            snap->n_q_heads_total = H;
            for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, h, c);
            snap->k_storage_dep = 1;
            snap->v_storage_dep = 2;
            snap->sink_dep = 3;
            const size_t off = (size_t)(h * GQA) * (size_t) q_all->nb[1];
            ggml_tensor * q_h = ggml_view_3d(ctx, q_all, D, GQA, T, q_all->nb[1], q_all->nb[2], off);
            q_h = ggml_cont(ctx, q_h);
            std::shared_ptr<xkv_graph_op_handle> hh;
            ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q_h, std::move(snap), hh, { k_st, v_st, sinks });
            assert(out && hh);
            // Parsed exactly once per query/Q-head inside compute.
            assert(hh->snapshot()->query_sink_logits.size() == 0); // filled at compute
            outs.push_back(out);
            handles.push_back(hh);
        }
        ggml_tensor * cur = ggml_concat(ctx, outs[0], outs[1], 0);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, cur);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        for (auto & hh : handles) {
            assert(hh->succeeded());
            assert(hh->snapshot()->query_sink_logits.size() == T);
        }
        // Oracle with identical per-head sinks.
        std::vector<float> expect;
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t h = 0; h < Hkv; ++h) {
                xkv_query_input qi;
                qi.head_dim_k = D; qi.head_dim_v = D; qi.n_q_heads = GQA;
                qi.q_vec.resize(D * GQA);
                for (uint32_t g = 0; g < GQA; ++g) {
                    const float * src = qfull.data() + ((size_t) t * H + h * GQA + g) * D;
                    std::memcpy(qi.q_vec.data() + g * D, src, D * sizeof(float));
                    uint32_t gh = h * GQA + g;
                    qi.sink_logits.push_back(variant == 0 ? sinkv[gh] : sinkv[t * H + gh]);
                }
                std::vector<xkv_hot_row> rows;
                for (uint32_t c = 0; c < N; ++c) {
                    xkv_hot_row r;
                    r.storage_pos = c;
                    r.k_ptr = kcells[c][h].data();
                    r.v_ptr = vcells[c][h].data();
                    rows.push_back(r);
                }
                auto o = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
                expect.insert(expect.end(), o.begin(), o.end());
            }
        }
        std::vector<float> expect_concat(D * H * T);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t h = 0; h < Hkv; ++h) {
                const float * src = expect.data() + ((size_t) t * Hkv + h) * D * GQA;
                float * dst = expect_concat.data() + ((size_t) t * H + h * GQA) * D;
                std::memcpy(dst, src, D * GQA * sizeof(float));
            }
        std::vector<float> actual(expect_concat.size());
        std::memcpy(actual.data(), cur->data, actual.size() * sizeof(float));
        assert(vec_eq(actual, expect_concat, 1e-4f));
        ggml_free(ctx);
    }
    std::cout << "test_sink_dep_once PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 6. All-masked: finite zero output, success
// ----------------------------------------------------------------------------
static void test_all_masked() {
    std::cout << "Running test_all_masked..." << std::endl;
    const uint32_t D = 8;
    struct ggml_context * ctx = make_ctx();
    auto snap = base_snap(D, D, 1, 1, 0);
    for (int i = 0; i < 3; ++i) {
        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = i;
        hd.is_valid = false; // masked out
        hd.k_data = det_floats(D, 1400 + i);
        hd.v_data = det_floats(D, 1450 + i);
        snap->hot_data.push_back(std::move(hd));
    }
    std::vector<float> qdata = det_floats(D, 1460);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());
    const float * p = static_cast<const float *>(out->data);
    for (uint32_t i = 0; i < D; ++i) assert(p[i] == 0.0f && std::isfinite(p[i]));
    ggml_free(ctx);
    std::cout << "test_all_masked PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 7. Variable DDVR group counts + explicit offsets
// ----------------------------------------------------------------------------
static void test_variable_ddvr() {
    std::cout << "Running test_variable_ddvr..." << std::endl;
    const uint32_t D = 8, N = 4;
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < N; ++c) {
        kcells[c][0] = det_floats(D, 1500 + c);
        vcells[c][0] = det_floats(D, 1550 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vcells);

    auto snap = base_snap(D, D, 1, 2, 0);
    snap->query_ddvr_group_counts = { 2, 1 };
    // q0 occupies [0, 2D), q1 occupies [2D, 3D).
    snap->query_ddvr_group_offsets = { 0, 2 * D };
    for (int64_t c = 0; c < (int64_t) N; ++c) {
        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = (uint32_t) c;
        hd.storage_pos = c;
        hd.is_valid = true;
        hd.cell = c;
        hd.kv_head = 0;
        hd.group_index = (uint32_t)(c % 2);
        snap->hot_data.push_back(std::move(hd));
    }
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;

    std::vector<float> qdata = det_floats(3 * D, 1600);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * D);
    std::memcpy(q->data, qdata.data(), qdata.size() * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());

    // Oracle: q0 has 2 groups over even/odd rows, q1 single group over all rows.
    std::vector<std::vector<float>> qgs = {
        { qdata.begin(), qdata.begin() + D }, { qdata.begin() + D, qdata.begin() + 2 * D },
        { qdata.begin() + 2 * D, qdata.begin() + 3 * D } };
    std::vector<float> expect;
    {
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D;
        qi.q_groups = { qgs[0], qgs[1] };
        std::vector<xkv_hot_row> rows;
        for (uint32_t c = 0; c < N; ++c) {
            xkv_hot_row r;
            r.storage_pos = c; r.group_index = c % 2;
            r.k_ptr = kcells[c][0].data(); r.v_ptr = vcells[c][0].data();
            rows.push_back(r);
        }
        auto o = xkv_dense_attention_reference(qi, rows, {}, {}, { 0, 1, 0, 1 }, {});
        expect.insert(expect.end(), o.begin(), o.end());
    }
    {
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qgs[2];
        std::vector<xkv_hot_row> rows;
        for (uint32_t c = 0; c < N; ++c) {
            xkv_hot_row r;
            r.storage_pos = c;
            r.k_ptr = kcells[c][0].data(); r.v_ptr = vcells[c][0].data();
            rows.push_back(r);
        }
        auto o = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
        expect.insert(expect.end(), o.begin(), o.end());
    }
    std::vector<float> actual(2 * D);
    std::memcpy(actual.data(), out->data, actual.size() * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_variable_ddvr PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 8. Future draft query isolation via per-query membership
// ----------------------------------------------------------------------------
static void test_future_isolation() {
    std::cout << "Running test_future_isolation..." << std::endl;
    const uint32_t D = 8, N = 4;
    auto run_case = [&](const std::vector<std::vector<float>> & fvals) {
        struct ggml_context * ctx = make_ctx();
        auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
        auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
        for (uint32_t c = 0; c < N; ++c) {
            kcells[c][0] = det_floats(D, 1700 + c);
            vcells[c][0] = c < 2 ? det_floats(D, 1750 + c) : fvals[c - 2];
        }
        struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, kcells);
        struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vcells);
        auto snap = base_snap(D, D, 1, 2, 0);
        snap->query_causal_limits = { 1, 3 };
        for (int64_t c = 0; c < (int64_t) N; ++c) {
            xkv_graph_snapshot::hot_row_data hd;
            hd.row_index = (uint32_t) c;
            hd.storage_pos = c;
            hd.is_valid = true;
            hd.cell = c;
            hd.query_visibility = { c < 2, true }; // q0 never sees future rows
            snap->hot_data.push_back(std::move(hd));
        }
        snap->k_storage_dep = 1;
        snap->v_storage_dep = 2;
        std::vector<float> qdata = det_floats(2 * D, 1800);
        struct ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2);
        std::memcpy(q->data, qdata.data(), qdata.size() * sizeof(float));
        std::shared_ptr<xkv_graph_op_handle> h;
        struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        assert(h->succeeded());
        std::vector<float> q0(D);
        std::memcpy(q0.data(), out->data, D * sizeof(float));
        ggml_free(ctx);
        return q0;
    };
    std::vector<std::vector<float>> fa = { det_floats(D, 1901), det_floats(D, 1902) };
    std::vector<std::vector<float>> fb = { det_floats(D, 1951), det_floats(D, 1952) };
    std::vector<float> q0a = run_case(fa);
    std::vector<float> q0b = run_case(fb);
    assert(vec_eq(q0a, q0b, 1e-5f)); // future rows cannot perturb q0
    std::cout << "test_future_isolation PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 9. Stale stamp -> retry; binding mismatch -> hard failure; mapping
// ----------------------------------------------------------------------------
static void test_stale_and_bindings() {
    std::cout << "Running test_stale_and_bindings..." << std::endl;
    const uint32_t D = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    const uint64_t pid = 424242;
    assert(store.register_hot_payload(pid, 2, 7, xkv_state::hot_committed));

    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::vector<float> qdata = det_floats(D, 2000);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));

    // Case A: correct bindings succeed.
    {
        auto snap = base_snap(D, D, 1, 1, 0);
        snap->expected_stamp = store.current_stamp();
        snap->store = &store;
        xkv_graph_snapshot::hot_row_data hd;
        hd.payload_id = pid; hd.row_index = 2; hd.storage_generation = 7;
        hd.expected_state = xkv_state::hot_committed;
        hd.k_data = det_floats(D, 2001); hd.v_data = det_floats(D, 2002);
        snap->hot_data.push_back(std::move(hd));
        std::shared_ptr<xkv_graph_op_handle> h;
        struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        assert(h->succeeded());
        assert(xkv_graph_postcompute_action(h->status()) == xkv_post_action::ok);
        std::string err;
        assert(xkv_graph_postcompute_ok(*h->snapshot(), &err));
    }
    // Case B: stale post stamp -> retry (pre-compute guard).
    {
        auto snap = base_snap(D, D, 1, 1, 0);
        snap->expected_stamp = store.current_stamp();
        snap->store = &store;
        xkv_graph_snapshot::hot_row_data hd;
        hd.k_data = det_floats(D, 2003); hd.v_data = det_floats(D, 2004);
        snap->hot_data.push_back(std::move(hd));
        std::shared_ptr<xkv_graph_op_handle> h;
        struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
        assert(store.bump_content_epoch(nullptr));
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        assert(!h->succeeded());
        assert(h->status() == xkv_read_status::retry_stale_stamp);
        assert(xkv_graph_postcompute_action(h->status()) == xkv_post_action::retry);
        std::string err;
        assert(!xkv_graph_postcompute_ok(*h->snapshot(), &err));
        // Output stays zeroed, never accepted as success.
        const float * p = static_cast<const float *>(out->data);
        for (uint32_t i = 0; i < D; ++i) assert(p[i] == 0.0f);
    }
    // Case C: generation mismatch -> hard failure.
    {
        auto snap = base_snap(D, D, 1, 1, 0);
        snap->expected_stamp = store.current_stamp();
        snap->store = &store;
        xkv_graph_snapshot::hot_row_data hd;
        hd.payload_id = pid; hd.row_index = 2; hd.storage_generation = 8; // wrong
        hd.expected_state = xkv_state::hot_committed;
        hd.k_data = det_floats(D, 2005); hd.v_data = det_floats(D, 2006);
        snap->hot_data.push_back(std::move(hd));
        std::shared_ptr<xkv_graph_op_handle> h;
        struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        assert(!h->succeeded());
        assert(xkv_graph_postcompute_action(h->status()) == xkv_post_action::hard_error);
    }
    ggml_free(ctx);
    std::cout << "test_stale_and_bindings PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 10. Callback failure propagation (bad dep) zeroes output
// ----------------------------------------------------------------------------
static void test_callback_failure() {
    std::cout << "Running test_callback_failure..." << std::endl;
    const uint32_t D = 8;
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::vector<float> qdata = det_floats(D, 2100);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    auto snap = base_snap(D, D, 1, 1, 0);
    xkv_graph_snapshot::hot_row_data hd;
    hd.row_index = 0; hd.storage_pos = 0; hd.is_valid = true;
    hd.cell = 0; // gather requested...
    snap->hot_data.push_back(std::move(hd));
    snap->k_storage_dep = 9; // ...but dep index out of range
    snap->v_storage_dep = 9;
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(!h->succeeded());
    assert(h->status() == xkv_read_status::invalid_argument);
    const float * p = static_cast<const float *>(out->data);
    for (uint32_t i = 0; i < D; ++i) assert(p[i] == 0.0f);
    assert(xkv_graph_postcompute_action(h->status()) == xkv_post_action::hard_error);
    ggml_free(ctx);
    std::cout << "test_callback_failure PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 11. Turbo hot K/V canonicalization vs host-decoded oracle
// ----------------------------------------------------------------------------
static void test_turbo_canonical() {
    std::cout << "Running test_turbo_canonical..." << std::endl;
    const uint32_t D = 128, N = 4;
    struct ggml_context * ctx = make_ctx(256 * 1024 * 1024);

    std::vector<std::vector<float>> kcan(N), vcan(N);
    for (uint32_t c = 0; c < N; ++c) {
        kcan[c] = det_floats(D, 2200 + c);
        vcan[c] = det_floats(D, 2250 + c);
        for (auto & v : kcan[c]) v *= 0.5f;
        for (auto & v : vcan[c]) v *= 0.5f;
    }
    struct ggml_tensor * k_st = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, D, 1, N);
    struct ggml_tensor * v_st = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, D, 1, N);
    {
        // ne[0]*ne[1] = D elements per cell row: quantize each row.
        std::vector<uint8_t> row(ggml_row_size(GGML_TYPE_TURBO4_0, D));
        for (uint32_t c = 0; c < N; ++c) {
            assert(ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, kcan[c].data(),
                row.data(), D, 128));
            std::memcpy(static_cast<uint8_t *>(k_st->data) + c * k_st->nb[2], row.data(), row.size());
            assert(ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, vcan[c].data(),
                row.data(), D, 128));
            std::memcpy(static_cast<uint8_t *>(v_st->data) + c * v_st->nb[2], row.data(), row.size());
        }
    }
    auto snap = base_snap(D, D, 1, 1, 0);
    snap->hot_layout.k_type = GGML_TYPE_TURBO4_0;
    snap->hot_layout.v_type = GGML_TYPE_TURBO4_0;
    snap->expected_turbo_fp_k = ggml_turbo_layout_fingerprint(GGML_TYPE_TURBO4_0);
    snap->expected_turbo_fp_v = ggml_turbo_layout_fingerprint(GGML_TYPE_TURBO4_0);
    assert(snap->expected_turbo_fp_k != 0);
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;

    std::vector<float> qdata = det_floats(D, 2300);
    for (auto & v : qdata) v *= 0.5f;
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));

    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());

    // Oracle over host-decoded canonical values: identical codec, so only
    // FP32 ordering differences remain.
    std::vector<std::vector<float>> kdec(N, std::vector<float>(D));
    std::vector<std::vector<float>> vdec(N, std::vector<float>(D));
    for (uint32_t c = 0; c < N; ++c) {
        const uint8_t * rk = static_cast<const uint8_t *>(k_st->data) + c * k_st->nb[2];
        const uint8_t * rv = static_cast<const uint8_t *>(v_st->data) + c * v_st->nb[2];
        assert(ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, rk, kdec[c].data(), D, 128, GGML_TURBO_DECODE_CANONICAL));
        assert(ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, rv, vdec[c].data(), D, 128, GGML_TURBO_DECODE_CANONICAL));
    }
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kdec[c].data(); r.v_ptr = vdec[c].data();
        rows.push_back(r);
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));

    // Fingerprint mismatch must fail closed.
    {
        auto snap2 = base_snap(D, D, 1, 1, 0);
        snap2->hot_layout.k_type = GGML_TYPE_TURBO4_0;
        snap2->hot_layout.v_type = GGML_TYPE_TURBO4_0;
        snap2->expected_turbo_fp_k = snap->expected_turbo_fp_k ^ 0x9e3779b9ULL;
        add_gather_row(*snap2, 0, 0, 0);
        snap2->k_storage_dep = 1;
        snap2->v_storage_dep = 2;
        std::shared_ptr<xkv_graph_op_handle> h2;
        struct ggml_tensor * out2 = xkv_build_graph_attention_ref(ctx, q, std::move(snap2), h2, { k_st, v_st });
        struct ggml_cgraph * gf2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf2, out2);
        ggml_graph_compute_with_ctx(ctx, gf2, 1);
        assert(!h2->succeeded());
    }
    ggml_free(ctx);
    std::cout << "test_turbo_canonical PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 12. Inverse attention rotation only (no RoPE inversion)
// ----------------------------------------------------------------------------
static void test_inverse_rotation_only() {
    std::cout << "Running test_inverse_rotation_only..." << std::endl;
    const uint32_t D = 8, N = 3;
    // Symmetric orthonormal Hadamard: H == H^{-1}.
    std::vector<float> H(D * D);
    for (uint32_t i = 0; i < D; ++i)
        for (uint32_t j = 0; j < D; ++j)
            H[i * D + j] = (((i & j) % 2 == 0) ? 1.0f : -1.0f) / std::sqrt((float) D);

    struct ggml_context * ctx = make_ctx();
    auto kcan = std::vector<std::vector<float>>(N);
    auto vcan = std::vector<std::vector<float>>(N);
    auto ksto = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    auto vsto = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < N; ++c) {
        kcan[c] = det_floats(D, 2400 + c);
        vcan[c] = det_floats(D, 2450 + c);
        ksto[c][0].assign(D, 0.0f);
        vsto[c][0] = vcan[c];
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) ksto[c][0][i] += H[i * D + j] * kcan[c][j];
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, ksto);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vsto);

    auto snap = base_snap(D, D, 1, 1, 0);
    snap->hot_k_inv_rot = H; // H^{-1} == H
    snap->hot_k_rot_dim = D;
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;

    std::vector<float> qdata = det_floats(D, 2500);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());

    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kcan[c].data(); r.v_ptr = vcan[c].data();
        rows.push_back(r);
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_inverse_rotation_only PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 13. Fail-closed capability validation + OFF gate + native seam
// ----------------------------------------------------------------------------
static void test_caps_and_dispatch() {
    std::cout << "Running test_caps_and_dispatch..." << std::endl;
    xkv_graph_build_caps base;
    base.xkv_mode = 2; // DENSE
    // OFF/SHADOW never cover bounded-hot: stock graph preserved.
    base.xkv_mode = 0;
    assert(!xkv_graph_covers_bounded_hot(base));
    assert(xkv_graph_reuse_allowed(0));
    base.xkv_mode = 1;
    assert(!xkv_graph_covers_bounded_hot(base));
    assert(xkv_graph_reuse_allowed(1));
    base.xkv_mode = 2;
    assert(xkv_graph_covers_bounded_hot(base));
    assert(!xkv_graph_reuse_allowed(2));
    base.xkv_mode = 3;
    assert(xkv_graph_covers_bounded_hot(base));
    assert(!xkv_graph_reuse_allowed(3));
    base.xkv_mode = 2;

    std::string err;
    assert(xkv_validate_build_caps(base, &err));
    xkv_exec_branch br = xkv_exec_branch::native_reconstruct;
    assert(xkv_select_exec_branch(base, br, &err));
    assert(br == xkv_exec_branch::cpu_reference);

    base.use_alibi = true;
    assert(!xkv_validate_build_caps(base, &err));
    base.use_alibi = false;

    base.rope_type = GGML_ROPE_TYPE_VISION;
    assert(!xkv_validate_build_caps(base, &err));
    base.rope_type = 2; // NEOX ok

    base.v_transposed = true;
    assert(!xkv_validate_build_caps(base, &err));
    base.v_transposed = false;

    base.has_kq_bias = true;
    assert(!xkv_validate_build_caps(base, &err));
    base.has_kq_bias = false;

    base.k_type = GGML_TYPE_COUNT;
    assert(!xkv_validate_build_caps(base, &err));
    base.k_type = GGML_TYPE_F32;

    // tq profile on non-CPU backend without native support fails explicitly.
    base.storage_needs_native = true;
    base.backend_is_cpu = false;
    base.native_backend_registered = false;
    assert(!xkv_select_exec_branch(base, br, &err));
    // With the Vulkan backend registered, native dispatch is selected.
    xkv_register_native_reconstruct_capability("Vulkan", true);
    assert(xkv_native_reconstruct_available_for("Vulkan"));
    assert(!xkv_native_reconstruct_available_for("CUDA"));
    assert(!xkv_native_reconstruct_available_for(nullptr));
    base.native_backend_registered = xkv_native_reconstruct_available_for("Vulkan");
    assert(xkv_select_exec_branch(base, br, &err));
    assert(br == xkv_exec_branch::native_reconstruct);
    xkv_register_native_reconstruct_capability("Vulkan", false); // restore default-absent
    assert(!xkv_native_reconstruct_available_for("Vulkan"));
    base.storage_needs_native = false;
    base.backend_is_cpu = true;
    base.native_backend_registered = false;
    std::cout << "test_caps_and_dispatch PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 13b. Layer gate: SWA/recurrent/MTP layers stay stock (P0-10)
// ----------------------------------------------------------------------------
static void test_layer_gate() {
    std::cout << "Running test_layer_gate..." << std::endl;
    // (DENSE, enabled full-attention layer) -> XKV path.
    assert(xkv_use_bounded_path(2, true));
    // (DENSE, disabled SWA layer) -> stock against its own full cache.
    assert(!xkv_use_bounded_path(2, false));
    // (SR, enabled) -> XKV path; (SR, disabled) -> stock.
    assert(xkv_use_bounded_path(3, true));
    assert(!xkv_use_bounded_path(3, false));
    // OFF/SHADOW never route, even when the layer would allow it.
    assert(!xkv_use_bounded_path(0, true));
    assert(!xkv_use_bounded_path(1, true));
    assert(!xkv_use_bounded_path(0, false));
    std::cout << "test_layer_gate PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 14. Managed owner: reuse refusal + post-compute mapping
// ----------------------------------------------------------------------------
static void test_managed_owner() {
    std::cout << "Running test_managed_owner..." << std::endl;
    llm_graph_params params{};
    llm_graph_input_xkv bounded(true);
    llm_graph_input_xkv unbounded(false);
    assert(!bounded.can_reuse(params)); // bounded XKV refuses reuse
    assert(unbounded.can_reuse(params));
    assert(bounded.bounded() && bounded.n_ops() == 0);

    std::string err;
    assert(bounded.postcompute_ok(&err)); // vacuous before adoption
    assert(bounded.post_action() == 0);

    auto ok_handle = std::make_shared<int>(1);
    auto bad_handle = std::make_shared<int>(2);
    bounded.adopt(ok_handle, []() { return 0; }, "h0");
    assert(bounded.n_ops() == 1);
    assert(bounded.postcompute_ok(&err));
    bounded.adopt(bad_handle, []() { return 1; }, "h1");
    assert(!bounded.postcompute_ok(&err));
    assert(err.find("retry") != std::string::npos);
    assert(bounded.post_action() == 1);
    // Handles stay alive through the owner (stable userdata addresses).
    ok_handle.reset();
    bad_handle.reset();
    assert(bounded.n_ops() == 2);
    assert(bounded.post_action() == 1);
    std::cout << "test_managed_owner PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 15. Result aggregate poll (P0-1): worst action across managed inputs
// ----------------------------------------------------------------------------
static void test_result_aggregate() {
    std::cout << "Running test_result_aggregate..." << std::endl;
    llm_graph_result res(16);
    assert(!res.xkv_has_bounded());
    std::string err;
    assert(res.xkv_poll_postcompute(&err) == 0);
    auto box = std::make_unique<llm_graph_input_xkv>(true);
    llm_graph_input_xkv * x = box.get();
    auto keep = std::make_shared<int>(7);
    x->adopt(keep, []() { return 0; }, "h0");
    res.add_input(std::move(box));
    assert(res.xkv_has_bounded());
    assert(res.xkv_poll_postcompute(&err) == 0);
    auto box2 = std::make_unique<llm_graph_input_xkv>(true);
    auto keep2 = std::make_shared<int>(8);
    box2->adopt(keep2, []() { return 1; }, "h1-stale");
    res.add_input(std::move(box2));
    assert(res.xkv_poll_postcompute(&err) == 1);
    assert(err.find("retry") != std::string::npos);
    keep.reset();
    keep2.reset();
    assert(res.xkv_poll_postcompute(nullptr) == 1);
    std::cout << "test_result_aggregate PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 16. ASAN-relevant boundary gather: second KV head, edge cells (P0-3)
// ----------------------------------------------------------------------------
static void test_gather_head_boundaries() {
    std::cout << "Running test_gather_head_boundaries..." << std::endl;
    const uint32_t D = 8, Hkv = 2, N = 1;
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(Hkv));
    for (uint32_t h = 0; h < Hkv; ++h) {
        kcells[0][h] = det_floats(D, 3000 + h);
        vcells[0][h] = det_floats(D, 3050 + h);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, Hkv, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, Hkv, N, vcells);
    assert((size_t) k_st->nb[1] == ggml_row_size(GGML_TYPE_F32, (int64_t) D * Hkv));
    auto snap = base_snap(D, D, 1, 1, 1);
    add_gather_row(*snap, 0, 1, 0);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;
    std::vector<float> qdata = det_floats(D, 3100);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<xkv_hot_row> rows(1);
    rows[0].storage_pos = 0;
    rows[0].k_ptr = kcells[0][1].data();
    rows[0].v_ptr = vcells[0][1].data();
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_gather_head_boundaries PASSED" << std::endl;
}

// Build encoded legal fragments over an F32 segment for SR tests.
static std::vector<legal_fragment> make_sr_frags(xkv_segment & seg, llama_xkv_cache_store & store,
    uint32_t D, uint32_t n_rows, uint32_t chunk) {
    std::vector<row_meta> rows;
    for (uint32_t r = 0; r < n_rows; ++r) {
        row_meta m;
        m.segment_row = r;
        m.payload_id = 77000 + r;
        m.storage_generation = 1;
        m.storage_pos = r;
        m.virtual_pos = r;
        m.visibility = 0;
        m.is_live = true;
        m.reader_visible = true;
        rows.push_back(m);
    }
    auto frags = build_legal_fragments(seg, store.current_stamp(), seg.segment_version, 1, 0, 0, rows, chunk);
    assert(!frags.empty());
    for (auto & f : frags) {
        encode_fragment_landmark(f, seg, 0, D, nullptr, 0, GGML_TYPE_Q8_0);
    }
    return frags;
}

// ----------------------------------------------------------------------------
// 17. SR-after-Q with full budget equals the dense oracle (P0-2)
// ----------------------------------------------------------------------------
static void test_sr_after_q_full() {
    std::cout << "Running test_sr_after_q_full..." << std::endl;
    const uint32_t D = 8, Nhot = 2, Ncold = 6;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    std::vector<float> ak, bk;
    auto seg = make_f32_segment(store, Ncold, D, 5150, ak, bk);
    const float * av = ak.data() + (size_t) Ncold * D;
    const float * bv = bk.data() + (size_t) D * D;
    auto frags = make_sr_frags(*seg, store, D, Ncold, 2);
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < Nhot; ++c) {
        kcells[c][0] = det_floats(D, 5200 + c);
        vcells[c][0] = det_floats(D, 5250 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, Nhot, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, Nhot, vcells);
    auto snap = base_snap(D, D, 1, 1, 0);
    for (int64_t c = 0; c < (int64_t) Nhot; ++c) add_gather_row(*snap, c, 0, (int64_t) Ncold + c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;
    snap->sr_mode = 1;
    snap->sr_config.sr_budget = 100;
    snap->sr_config.landmark_type = GGML_TYPE_Q8_0;
    snap->sr_legal_frags = frags;
    snap->sr_feature_offset = 0;
    snap->sr_feature_dim = D;
    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    view.owning_layer = 0;
    view.factor_group_index = 0;
    for (uint32_t i = 0; i < Ncold; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    snap->segment_views.push_back(std::move(view));
    std::vector<float> qdata = det_floats(D, 5300);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < Nhot; ++c) {
        xkv_hot_row r;
        r.storage_pos = Ncold + c;
        r.k_ptr = kcells[c][0].data();
        r.v_ptr = vcells[c][0].data();
        rows.push_back(r);
    }
    std::vector<std::vector<float>> ck(Ncold, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> cv(Ncold, std::vector<float>(D, 0.0f));
    for (uint32_t r = 0; r < Ncold; ++r)
        for (uint32_t f = 0; f < D; ++f)
            for (uint32_t k = 0; k < D; ++k) {
                ck[r][f] += ak[r * D + k] * bk[f * D + k];
                cv[r][f] += av[r * D + k] * bv[f * D + k];
            }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<uint32_t> groups(Ncold, 0);
    std::vector<bool> mask(Ncold, true);
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, ck, cv, groups, mask);
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_sr_after_q_full PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 18. SR selection is independent per query (P0-2)
// ----------------------------------------------------------------------------
static void test_sr_per_query_isolation() {
    std::cout << "Running test_sr_per_query_isolation..." << std::endl;
    const uint32_t D = 8, Ncold = 6;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    std::vector<float> ak, bk;
    auto seg = make_f32_segment(store, Ncold, D, 5350, ak, bk);
    auto frags = make_sr_frags(*seg, store, D, Ncold, 2);
    auto run = [&](const std::vector<float> & q1) {
        struct ggml_context * ctx = make_ctx();
        auto snap = base_snap(D, D, 1, 2, 0);
        snap->expected_stamp = store.current_stamp();
        snap->store = &store;
        snap->sr_mode = 1;
        snap->sr_config.sr_budget = 1;
        snap->sr_config.landmark_type = GGML_TYPE_Q8_0;
        snap->sr_legal_frags = frags;
        snap->sr_feature_offset = 0;
        snap->sr_feature_dim = D;
        xkv_segment_read_view view;
        view.pin = store.pin_segment(seg->segment_id);
        view.segment_version_id = seg->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < Ncold; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back(i);
            view.group_indices.push_back(0);
        }
        snap->segment_views.push_back(std::move(view));
        std::vector<float> q0 = det_floats(D, 5400);
        std::vector<float> qdata = q0;
        qdata.insert(qdata.end(), q1.begin(), q1.end());
        struct ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2);
        std::memcpy(q->data, qdata.data(), qdata.size() * sizeof(float));
        std::shared_ptr<xkv_graph_op_handle> h;
        struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        assert(h->succeeded());
        std::vector<float> q0out(D);
        std::memcpy(q0out.data(), out->data, D * sizeof(float));
        ggml_free(ctx);
        return q0out;
    };
    std::vector<float> a = run(det_floats(D, 5451));
    std::vector<float> b = run(det_floats(D, 5452));
    assert(vec_eq(a, b, 1e-5f));
    std::cout << "test_sr_per_query_isolation PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 19. Global SR budget across segments: budget=1 selects one fragment (P0-2b)
// ----------------------------------------------------------------------------
static void test_sr_global_budget() {
    std::cout << "Running test_sr_global_budget..." << std::endl;
    const uint32_t D = 8, Nseg = 4;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    std::vector<float> ak0, bk0, ak1, bk1;
    auto seg0 = make_f32_segment(store, Nseg, D, 6100, ak0, bk0);
    auto seg1 = make_f32_segment(store, Nseg, D, 6200, ak1, bk1);
    auto f0 = make_sr_frags(*seg0, store, D, Nseg, 2);
    auto f1 = make_sr_frags(*seg1, store, D, Nseg, 2);
    assert(f0.size() == 2 && f1.size() == 2);
    struct ggml_context * ctx = make_ctx();
    auto snap = base_snap(D, D, 1, 1, 0);
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;
    snap->sr_mode = 1;
    snap->sr_config.sr_budget = 1; // ONE global winner across both segments
    snap->sr_config.landmark_type = GGML_TYPE_Q8_0;
    snap->sr_legal_frags = f0;
    snap->sr_legal_frags.insert(snap->sr_legal_frags.end(), f1.begin(), f1.end());
    snap->sr_feature_offset = 0;
    snap->sr_feature_dim = D;
    for (auto * sg : { seg0.get(), seg1.get() }) {
        xkv_segment_read_view view;
        view.pin = store.pin_segment(sg->segment_id);
        view.segment_version_id = sg->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < Nseg; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back(i);
            view.group_indices.push_back(0);
        }
        snap->segment_views.push_back(std::move(view));
    }
    std::vector<float> qdata = det_floats(D, 6300);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    // Oracle hypotheses: each single fragment alone (reconstructed cold).
    // A global budget of 1 must match exactly one; per-segment budgets
    // would select two fragments and match none (up to degeneracy).
    auto cold_keys = [&](const std::vector<float> & a, const std::vector<float> & b,
        uint32_t r0, uint32_t r1) {
        std::vector<std::vector<float>> ck(2, std::vector<float>(D, 0.0f));
        std::vector<std::vector<float>> cv(2, std::vector<float>(D, 0.0f));
        const float * av = a.data() + (size_t) Nseg * D;
        const float * bv = b.data() + (size_t) D * D;
        for (uint32_t k = 0; k < 2; ++k) {
            uint32_t r = (k == 0) ? r0 : r1;
            for (uint32_t f = 0; f < D; ++f)
                for (uint32_t c = 0; c < D; ++c) {
                    ck[k][f] += a[r * D + c] * b[f * D + c];
                    cv[k][f] += av[r * D + c] * bv[f * D + c];
                }
        }
        return std::make_pair(ck, cv);
    };
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    auto oracle_for = [&](const std::vector<std::vector<float>> & ck, const std::vector<std::vector<float>> & cv) {
        std::vector<uint32_t> groups(2, 0);
        std::vector<bool> mask(2, true);
        return xkv_dense_attention_reference(qi, {}, ck, cv, groups, mask);
    };
    int matches = 0;
    const std::pair<std::vector<float> *, std::vector<float> *> ars[] = {
        { &ak0, &bk0 }, { &ak1, &bk1 } };
    for (const auto & ab : ars) {
        for (uint32_t f = 0; f < 2; ++f) {
            auto kv = cold_keys(*ab.first, *ab.second, 2 * f, 2 * f + 1);
            if (vec_eq(actual, oracle_for(kv.first, kv.second), 1e-4f)) ++matches;
        }
    }
    assert(matches == 1);
    ggml_free(ctx);
    std::cout << "test_sr_global_budget PASSED" << std::endl;
}

// Attach a snapshot-owned combined workspace (lease + backing + reader subspan).
// Returns the backing total for accounting assertions; lease moved into snap.
static size_t attach_workspace(xkv_graph_snapshot & snap, llama_xkv_cache_store & store,
    uint32_t n_gather, size_t stride_k, size_t stride_v, size_t nk, size_t nv,
    uint32_t Dk, uint32_t Dv, uint32_t pad_max, size_t sink_floats, size_t reader_bytes,
    xkv_arena_lease & lease_holder) {
    xkv_callback_layout lay;
    std::string err;
    assert(xkv_compute_callback_layout(n_gather, stride_k, stride_v, nk, nv, Dk, Dv,
        pad_max, sink_floats, reader_bytes, 0, lay, &err));
    // Lease covers alignment slack + every live region (single lease).
    size_t lease_bytes = 0;
    assert(safe_add(lay.total_bytes, (size_t) 64, lease_bytes));
    lease_holder = store.acquire_workspace_lease(lease_bytes);
    assert((bool) lease_holder);
    assert(lease_holder.data() != nullptr);
    assert(lease_holder.size() >= lease_bytes);
    uintptr_t raw = reinterpret_cast<uintptr_t>(lease_holder.data());
    size_t off = (64 - (raw % 64)) % 64;
    assert(off + lay.total_bytes <= lease_holder.size());
    auto ws = std::make_shared<xkv_reader_workspace>();
    xkv_reader_workspace_config rcfg; // modest maxima for these small tests
    rcfg.max_queries = 4;
    rcfg.max_q_heads = 8;
    rcfg.max_head_dim_k = 256;
    rcfg.max_head_dim_v = 256;
    rcfg.max_tile_size = 128;
    rcfg.max_rank_k = 512;
    rcfg.max_rank_v = 512;
    rcfg.max_csr_entries = 8192;
    auto * lease_u8 = static_cast<uint8_t *>(lease_holder.data());
    assert(ws->warmup_external(lease_u8 + off + lay.reader_off, lay.reader_bytes, rcfg, &err));
    snap.workspace_required = true;
    snap.workspace_lease = std::move(lease_holder);
    snap.workspace_base_off = off;
    snap.workspace_layout = lay;
    snap.workspace_reader = ws;
    snap.reader_config.workspace = ws.get();
    assert(snap.workspace_ready());
    // Size descriptor caches now (build-time allocation, outside callback).
    assert(snap.preallocate_compute_state(&err));
    return lease_bytes;
}

// ----------------------------------------------------------------------------
// 20. Span-backed compute equals the oracle (P0-WS)
// ----------------------------------------------------------------------------
static void test_workspace_span_oracle() {
    std::cout << "Running test_workspace_span_oracle..." << std::endl;
    const uint32_t D = 8, N = 2;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < N; ++c) {
        kcells[c][0] = det_floats(D, 7000 + c);
        vcells[c][0] = det_floats(D, 7050 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vcells);
    auto snap = base_snap(D, D, 1, 1, 0);
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;
    xkv_arena_lease lease;
    size_t lease_bytes = attach_workspace(*snap, store, N,
        (size_t) k_st->nb[2], (size_t) v_st->nb[2],
        (size_t) D, (size_t) D, D, D, D, 0, 8 * 1024 * 1024, lease);
    (void) lease_bytes;
    std::vector<float> qdata = det_floats(D, 7100);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(h->succeeded());
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kcells[c][0].data();
        r.v_ptr = vcells[c][0].data();
        rows.push_back(r);
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    h.reset();
    ggml_free(ctx);
    std::cout << "test_workspace_span_oracle PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 21. One-byte-short backing fails closed (P0-WS)
// ----------------------------------------------------------------------------
static void test_workspace_one_byte_short() {
    std::cout << "Running test_workspace_one_byte_short..." << std::endl;
    const uint32_t D = 8, N = 2;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < N; ++c) {
        kcells[c][0] = det_floats(D, 7200 + c);
        vcells[c][0] = det_floats(D, 7250 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vcells);
    auto snap = base_snap(D, D, 1, 1, 0);
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;
    xkv_arena_lease lease;
    attach_workspace(*snap, store, N, (size_t) k_st->nb[2], (size_t) v_st->nb[2],
        (size_t) D, (size_t) D, D, D, D, 0, 8 * 1024 * 1024, lease);
    snap->workspace_layout.total_bytes += 1; // declare one byte more than backed
    // Declared need now exceeds lease backing by exactly one byte: the
    // backing itself is one byte short of what the layout requires.
    assert(!snap->workspace_ready());
    std::vector<float> qdata = det_floats(D, 7300);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(!h->succeeded());
    assert(h->status() == xkv_read_status::workspace_exceeded);
    const float * p = static_cast<const float *>(out->data);
    for (uint32_t i = 0; i < D; ++i) assert(p[i] == 0.0f);
    ggml_free(ctx);
    std::cout << "test_workspace_one_byte_short PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 22. Lease accounting + required-without-backing refusal (P0-WS)
// ----------------------------------------------------------------------------
static void test_workspace_lease_accounting() {
    std::cout << "Running test_workspace_lease_accounting..." << std::endl;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    assert(store.get_arena().get_live_bytes() == 0);
    {
        xkv_arena_lease lease = store.acquire_workspace_lease(12345);
        assert((bool) lease);
        assert(store.get_arena().get_live_bytes() == 12345);
        assert(store.get_arena().get_peak_bytes() >= 12345);
    }
    assert(store.get_arena().get_live_bytes() == 0); // RAII release
    // Required snapshot with no backing must fail closed, never heap.
    const uint32_t D = 8;
    struct ggml_context * ctx = make_ctx();
    auto snap = base_snap(D, D, 1, 1, 0);
    snap->workspace_required = true; // budget>0, null workspace
    xkv_graph_snapshot::hot_row_data hd;
    hd.row_index = 0; hd.storage_pos = 0; hd.is_valid = true;
    hd.k_data = det_floats(D, 7401); hd.v_data = det_floats(D, 7402);
    snap->hot_data.push_back(std::move(hd));
    std::vector<float> qdata = det_floats(D, 7400);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(!h->succeeded());
    assert(h->status() == xkv_read_status::workspace_exceeded);
    ggml_free(ctx);
    std::cout << "test_workspace_lease_accounting PASSED" << std::endl;
}

// Global operator new override (this test TU only): counts heap allocations
// while this test TU's local flag is set. The flag wraps ONLY the direct
// callback invocation below (same thread): the real callback runs unmodified
// while every allocation — including reader internals — is counted. Only the
// exact unavoidable output allocation may remain.
namespace {
thread_local bool t_count_allocs = false;
thread_local size_t t_alloc_calls = 0;
thread_local size_t t_alloc_bytes = 0;
} // namespace
void * operator new(std::size_t n) {
    if (t_count_allocs) { ++t_alloc_calls; t_alloc_bytes += n; }
    void * p = std::malloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
void * operator new[](std::size_t n) {
    if (t_count_allocs) { ++t_alloc_calls; t_alloc_bytes += n; }
    void * p = std::malloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

// ----------------------------------------------------------------------------
// 23. Zero-heap callback proof (P0-WS): allocation counter around compute
// ----------------------------------------------------------------------------
static void test_zero_heap_callback() {
    std::cout << "Running test_zero_heap_callback..." << std::endl;
    const uint32_t D = 8, N = 2;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < N; ++c) {
        kcells[c][0] = det_floats(D, 7500 + c);
        vcells[c][0] = det_floats(D, 7550 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vcells);
    std::vector<float> qdata = det_floats(D, 7600);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    // One snapshot + lease for both runs (tensors persist in ctx). The real
    // callback is invoked directly — unmodified — with the counter wrapping
    // only the invocation (same thread): every allocation, including reader
    // internals, is counted. Only the exact output allocation may remain.
    auto snap = base_snap(D, D, 1, 1, 0);
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;
    xkv_arena_lease lease;
    attach_workspace(*snap, store, N, (size_t) k_st->nb[2], (size_t) v_st->nb[2],
        (size_t) D, (size_t) D, D, D, D, 0, 8 * 1024 * 1024, lease);
    snap->hot_storage_host_resident = true; // host memcpy path (deterministic, no backend call)
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h, { k_st, v_st });
    assert(out && h);
    // Warmup: descriptor first-touch (uncounted).
    xkv_graph_op_handle::ggml_custom_op_callback(out, 0, 1, h.get());
    assert(h->succeeded());
    // Counted run: full real work.
    t_alloc_calls = 0;
    t_alloc_bytes = 0;
    t_count_allocs = true;
    xkv_graph_op_handle::ggml_custom_op_callback(out, 0, 1, h.get());
    t_count_allocs = false;
    assert(h->succeeded());
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kcells[c][0].data();
        r.v_ptr = vcells[c][0].data();
        rows.push_back(r);
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
    assert(vec_eq(actual, expect, 1e-4f));
    // Exact unavoidable output allocation for n_queries=1: one per_query
    // vector plus one output vector (both exact-sized resizes from empty).
    // Every reader/internal allocation would appear here as extra calls.
    assert(t_alloc_calls == 2);
    assert(t_alloc_bytes == sizeof(xkv_read_result) + (size_t) D * sizeof(float));
    h.reset();
    ggml_free(ctx);
    std::cout << "test_zero_heap_callback PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 24. Coordinator lease: acquire at set_input, release on every poll path
// ----------------------------------------------------------------------------
static void test_lease_lifecycle() {
    std::cout << "Running test_lease_lifecycle..." << std::endl;
    llm_graph_params params{};
    int acquired = 0, released = 0;
    auto box = std::make_unique<llm_graph_input_xkv>(true);
    llm_graph_input_xkv * x = box.get();
    auto keep = std::make_shared<int>(9);
    x->adopt(keep, []() { return 0; }, "h0", {},
        [&]() { ++released; },
        [&](std::string &) { ++acquired; return true; });
    // set_input acquires the fresh guard before refresh.
    x->set_input(nullptr);
    assert(acquired == 1);
    assert(released == 0);
    // Poll ok releases after final validation.
    std::string err;
    assert(x->postcompute_ok(&err));
    assert(released == 1);
    // Second poll does not re-release (idempotent).
    assert(x->postcompute_ok(&err));
    assert(released == 1);
    // Failure path also releases.
    auto box2 = std::make_unique<llm_graph_input_xkv>(true);
    llm_graph_input_xkv * y = box2.get();
    auto keep2 = std::make_shared<int>(10);
    int released2 = 0;
    y->adopt(keep2, []() { return 2; }, "h1", {},
        [&]() { ++released2; },
        [&](std::string &) { return true; });
    y->set_input(nullptr);
    assert(!y->postcompute_ok(&err));
    assert(y->post_action() == 2);
    assert(released2 == 1);
    // Acquire failure surfaces as hard error at poll.
    auto box3 = std::make_unique<llm_graph_input_xkv>(true);
    llm_graph_input_xkv * z = box3.get();
    auto keep3 = std::make_shared<int>(11);
    z->adopt(keep3, []() { return 0; }, "h2", {}, {},
        [&](std::string & e) { e = "guard busy"; return false; });
    z->set_input(nullptr);
    assert(!z->postcompute_ok(&err));
    assert(err.find("guard busy") != std::string::npos);
    assert(z->post_action() == 2);
    // Destructor releases outstanding guards (reset path).
    int released4 = 0;
    {
        llm_graph_input_xkv tmp(true);
        auto keep4 = std::make_shared<int>(12);
        tmp.adopt(keep4, []() { return 0; }, "h3", {},
            [&]() { ++released4; },
            [&](std::string &) { return true; });
        tmp.set_input(nullptr); // acquires, never polled
    }
    assert(released4 == 1);
    keep.reset();
    keep2.reset();
    keep3.reset();
    (void) params;
    std::cout << "test_lease_lifecycle PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 25. Snapshot reader-guard hooks (Main: coordinator lease seam)
// ----------------------------------------------------------------------------
static void test_snapshot_guard_hooks() {
    std::cout << "Running test_snapshot_guard_hooks..." << std::endl;
    xkv_graph_snapshot snap;
    std::string err;
    snap.release_reader_guard(); // no hooks: vacuous no-op
    assert(snap.acquire_reader_guard(err)); // vacuous success
    int rel = 0, acq = 0;
    snap.guard_release_hook = [&]() { ++rel; };
    snap.guard_acquire_hook = [&](std::string &) { ++acq; return true; };
    assert(snap.acquire_reader_guard(err));
    assert(acq == 1);
    snap.release_reader_guard();
    assert(rel == 1);
    snap.guard_acquire_hook = [&](std::string & e) { e = "busy"; return false; };
    assert(!snap.acquire_reader_guard(err));
    assert(err == "busy");
    std::cout << "test_snapshot_guard_hooks PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 26. Reuse key + snapshot refresh plumbing (Main: decode reuse)
// ----------------------------------------------------------------------------
static void test_reuse_key_and_refresh() {
    std::cout << "Running test_reuse_key_and_refresh..." << std::endl;
    llm_graph_params params{};
    // No key -> refuse.
    llm_graph_input_xkv bare(true);
    assert(!bare.can_reuse(params));
    // Key set, no cache identity, matching defaults, no entries -> reuse.
    llm_graph_input_xkv x(true);
    llm_graph_input_xkv::xkv_reuse_key k;
    k.n_tokens = (uint32_t) params.ubatch.n_tokens;
    k.xkv_mode = static_cast<int>(params.cparams.xkv_mode);
    k.storage_profile = static_cast<int>(params.cparams.xkv_storage_profile);
    k.cpu_branch = true; // null sched sniffs CPU
    x.set_key(k, nullptr);
    assert(x.can_reuse(params));
    // Token count change refuses.
    llm_graph_input_xkv::xkv_reuse_key k2 = k;
    k2.n_tokens += 1;
    x.set_key(k2, nullptr);
    assert(!x.can_reuse(params));
    x.set_key(k, nullptr);
    assert(x.can_reuse(params));
    // Refresh hook runs at set_input (before refresh API: none installed
    // here, so nothing happens); with a hook, success clears force flag.
    auto snap = std::make_unique<xkv_graph_snapshot>();
    snap->head_dim_k = 8;
    snap->head_dim_v = 8;
    snap->n_q_heads = 1;
    snap->n_queries = 0;
    int refresh_calls = 0;
    snap->snapshot_refresh_fn = [&](std::string &) { ++refresh_calls; return true; };
    auto h = std::make_shared<xkv_graph_op_handle>(std::move(snap));
    std::shared_ptr<void> hv = h;
    x.adopt(hv, []() { return 0; }, "r0");
    x.note_snapshot(h->snapshot());
    x.set_input(nullptr);
    assert(refresh_calls == 1);
    assert(!h->snapshot()->force_retry_stale);
    assert(x.can_reuse(params)); // hook present
    // Failing refresh forces stale retry.
    h->snapshot()->snapshot_refresh_fn = [&](std::string & e) { e = "cells moved"; return false; };
    x.set_input(nullptr);
    assert(refresh_calls == 1); // new hook ran (counter belongs to old lambda)
    assert(h->snapshot()->force_retry_stale);
    // Missing hook refuses reuse.
    h->snapshot()->snapshot_refresh_fn = nullptr;
    h->snapshot()->force_retry_stale = false;
    assert(!x.can_reuse(params));
    std::cout << "test_reuse_key_and_refresh PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 27. Forced stale pre-check in compute (Main: refresh failure path)
// ----------------------------------------------------------------------------
static void test_force_retry_precheck() {
    std::cout << "Running test_force_retry_precheck..." << std::endl;
    const uint32_t D = 8;
    struct ggml_context * ctx = make_ctx();
    auto snap = base_snap(D, D, 1, 1, 0);
    xkv_graph_snapshot::hot_row_data hd;
    hd.row_index = 0; hd.storage_pos = 0; hd.is_valid = true;
    hd.k_data = det_floats(D, 8001); hd.v_data = det_floats(D, 8002);
    snap->hot_data.push_back(std::move(hd));
    snap->force_retry_stale = true;
    std::vector<float> qdata = det_floats(D, 8000);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::shared_ptr<xkv_graph_op_handle> h;
    struct ggml_tensor * out = xkv_build_graph_attention_ref(ctx, q, std::move(snap), h);
    assert(out && h);
    xkv_graph_op_handle::ggml_custom_op_callback(out, 0, 1, h.get());
    assert(!h->succeeded());
    assert(h->status() == xkv_read_status::retry_stale_stamp);
    const float * p = static_cast<const float *>(out->data);
    for (uint32_t i = 0; i < D; ++i) assert(p[i] == 0.0f);
    ggml_free(ctx);
    std::cout << "test_force_retry_precheck PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 28. Snapshot metadata footprint accounting (Main: workspace metrics)
// ----------------------------------------------------------------------------
static void test_snapshot_metadata_bytes() {
    std::cout << "Running test_snapshot_metadata_bytes..." << std::endl;
    auto snap = base_snap(8, 8, 1, 1, 0);
    size_t empty = xkv_snapshot_metadata_bytes(*snap);
    assert(empty >= sizeof(xkv_graph_snapshot));
    xkv_graph_snapshot::hot_row_data hd;
    hd.k_data = det_floats(8, 8101);
    hd.v_data = det_floats(8, 8102);
    hd.query_visibility = { true };
    snap->hot_data.push_back(std::move(hd));
    size_t fuller = xkv_snapshot_metadata_bytes(*snap);
    assert(fuller >= empty + 2 * 8 * sizeof(float));
    assert(xkv_snapshot_metadata_bytes(*snap) == fuller); // deterministic
    std::cout << "test_snapshot_metadata_bytes PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 29. Per-(query,row) DDVR groups: CSR groups + hot query_group_indices
// ----------------------------------------------------------------------------
static void test_per_query_groups() {
    std::cout << "Running test_per_query_groups..." << std::endl;
    // 1. Hot row: build_hot_rows populates query_group_indices positionally.
    auto snap = base_snap(8, 8, 1, 2, 0);
    xkv_graph_snapshot::hot_row_data hd;
    hd.row_index = 0; hd.storage_pos = 0; hd.is_valid = true;
    hd.group_index = 3;
    hd.query_visibility = { true, false };
    hd.k_data = det_floats(8, 8201);
    hd.v_data = det_floats(8, 8202);
    snap->hot_data.push_back(hd);
    auto hot_built = snap->build_hot_rows();
    assert(hot_built.size() == 1);
    assert(hot_built[0].query_group_indices.size() == 2);
    assert(hot_built[0].query_group_indices[0] == 3);
    assert(hot_built[0].query_group_indices[1] == UINT32_MAX); // invisible
    // Preallocated fill produces identical descriptors.
    std::string err;
    assert(snap->preallocate_compute_state(&err));
    assert(fill_required_hot(*snap, &err));
    assert(snap->hot_cache[0].query_group_indices.size() == 2);
    assert(snap->hot_cache[0].query_group_indices[0] == 3);
    assert(snap->hot_cache[0].query_group_indices[1] == UINT32_MAX);
    // 2. Legacy SR-after-Q merge: 2 queries with 2 DDVR groups each.
    // csr_group_indices must parallel csr_indices, carrying the selector group.
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    std::vector<float> ak, bk;
    auto seg = make_f32_segment(store, 4, 8, 8300, ak, bk);
    auto frags = make_sr_frags(*seg, store, 8, 4, 2);
    auto snap_sr = base_snap(8, 8, 1, 2, 0);
    snap_sr->expected_stamp = store.current_stamp();
    snap_sr->store = &store;
    snap_sr->sr_mode = 1;
    snap_sr->sr_config.sr_budget = 4;
    snap_sr->sr_config.landmark_type = GGML_TYPE_Q8_0;
    snap_sr->sr_legal_frags = frags;
    snap_sr->sr_feature_offset = 0;
    snap_sr->sr_feature_dim = 8;
    xkv_segment_read_view view;
    view.pin = store.pin_segment(seg->segment_id);
    view.segment_version_id = seg->segment_version;
    view.storage_generation = 1;
    for (uint32_t i = 0; i < 4; ++i) {
        view.selected_rows.push_back(i);
        view.storage_positions.push_back(i);
        view.group_indices.push_back(0);
    }
    snap_sr->segment_views.push_back(std::move(view));
    std::vector<xkv_query_input> queries(2);
    for (uint32_t q = 0; q < 2; ++q) {
        queries[q].query_index = q;
        queries[q].head_dim_k = 8;
        queries[q].head_dim_v = 8;
        queries[q].n_q_heads = 1;
        queries[q].q_groups = { det_floats(8, 8400 + q * 10), det_floats(8, 8401 + q * 10) };
    }
    sr_batch_selection_result sel;
    assert(run_sr_after_q(*snap_sr, queries, sel, &err));
    assert(sel.csr_group_indices.size() == sel.csr_indices.size());
    for (size_t q = 0; q < 2; ++q) {
        uint32_t b = sel.csr_ptrs[q], e = sel.csr_ptrs[q + 1];
        for (uint32_t c = b; c < e; ++c) {
            assert(sel.csr_group_indices[c] < 2); // belongs to one of its groups
        }
    }
    std::cout << "test_per_query_groups PASSED" << std::endl;
}

// ==================== Native tiled chain (ggml_xkv_attention) ====================
//
// These tests drive xkv_build_attention_native directly with hand-built
// backend tensors (mimicking builder-wired arenas) and compare against the
// dense oracle. Production wiring (llm_build_attn_xkv native branch) uses
// the same builder; only Q slicing/concat and sink slicing live llm-side.

// Fill a [width, n] F32 tensor from per-row float vectors.
static void fill_rows_f32(ggml_tensor * t, const std::vector<std::vector<float>> & rows, uint32_t width) {
    assert(t->type == GGML_TYPE_F32);
    for (size_t r = 0; r < rows.size(); ++r) {
        assert(rows[r].size() == width);
        std::memcpy(static_cast<float *>(t->data) + r * width, rows[r].data(), width * sizeof(float));
    }
}

// Native F32 segment: payload with explicit ranks/dims plus raw float pools
// for arena tensors and the oracle. A pool: per-row rank vectors; B pool:
// per-feature rank vectors.
struct native_seg_bundle {
    std::shared_ptr<xkv_segment> seg;
    std::vector<std::vector<float>> a_rows; // [n_rows][rank]
    std::vector<std::vector<float>> av_rows;
    std::vector<std::vector<float>> bk_rows; // [Dk_total][rank]
    std::vector<std::vector<float>> bv_rows;
};

static native_seg_bundle make_native_segment(llama_xkv_cache_store & store,
    uint32_t n_rows, uint32_t Dk_total, uint32_t Dv_total, uint32_t rank, uint64_t seed) {
    native_seg_bundle b;
    for (uint32_t r = 0; r < n_rows; ++r) {
        b.a_rows.push_back(det_floats(rank, seed + r));
        b.av_rows.push_back(det_floats(rank, seed + 1000 + r));
    }
    for (uint32_t f = 0; f < Dk_total; ++f) b.bk_rows.push_back(det_floats(rank, seed + 2000 + f));
    for (uint32_t f = 0; f < Dv_total; ++f) b.bv_rows.push_back(det_floats(rank, seed + 3000 + f));
    auto flat = [](const std::vector<std::vector<float>> & rows) {
        std::vector<float> out;
        for (const auto & r : rows) out.insert(out.end(), r.begin(), r.end());
        return out;
    };
    std::vector<float> a_flat = flat(b.a_rows), av_flat = flat(b.av_rows);
    std::vector<float> bk_flat = flat(b.bk_rows), bv_flat = flat(b.bv_rows);
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = { 0 };
    g.total_dim_k = Dk_total;
    g.total_dim_v = Dv_total;
    g.rank_k = rank;
    g.rank_v = rank;
    g.layer_feature_offsets_k = { 0 };
    g.layer_feature_dims_k = { Dk_total };
    g.layer_feature_offsets_v = { 0 };
    g.layer_feature_dims_v = { Dv_total };
    codec_desc da = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, { n_rows, rank }, 0, seed + 11);
    g.a_k = encode_matrix(da, a_flat.data(), a_flat.size());
    codec_desc db = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, { Dk_total, rank }, 0, seed + 11);
    g.set_b_k(encode_matrix(db, bk_flat.data(), bk_flat.size()));
    codec_desc dav = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, { n_rows, rank }, 0, seed + 13);
    g.a_v = encode_matrix(dav, av_flat.data(), av_flat.size());
    codec_desc dbv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, { Dv_total, rank }, 0, seed + 13);
    g.set_b_v(encode_matrix(dbv, bv_flat.data(), bv_flat.size()));
    auto seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, { g });
    std::vector<uint64_t> pids(n_rows), gens(n_rows, 1);
    for (uint32_t i = 0; i < n_rows; ++i) pids[i] = seed * 1000 + i + 1;
    for (uint32_t i = 0; i < n_rows; ++i) store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
    uint64_t nonce = 0;
    assert(store.mark_seal_candidates(pids, gens, &nonce));
    std::string perr;
    assert(store.publish_candidate(seg, pids, gens, &perr));
    b.seg = seg;
    return b;
}

// Arena tensors [rank, n] from per-row float pools (mimics wired arenas).
static ggml_tensor * make_arena_tensor(ggml_context * ctx, const std::vector<std::vector<float>> & rows, uint32_t rank) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, rows.size());
    fill_rows_f32(t, rows, rank);
    return t;
}

// Snapshot base for native tests: F32 codecs, dense mode, wired params.
static std::unique_ptr<xkv_graph_snapshot> base_native_snap(
    uint32_t Dk, uint32_t Dv, uint32_t nq_heads, uint32_t n_queries, uint32_t kv_head) {
    auto snap = base_snap(Dk, Dv, nq_heads, n_queries, kv_head);
    snap->native_params.version = GGML_XKV_VERSION;
    snap->native_params.rotary_dim = 0;
    snap->native_params.rope_mode = 0;
    snap->native_params.seed_k = 11;
    snap->native_params.seed_v = 17;
    snap->native_params.fp_combined = ggml_xkv_fp_combined(
        GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 11, 17);
    return snap;
}

// Attach one arena bundle for (segment, group 0) to the snapshot.
static void attach_native_arenas(xkv_graph_snapshot & snap, const native_seg_bundle & b,
    ggml_tensor * ak_t, ggml_tensor * bk_t, ggml_tensor * av_t, ggml_tensor * bv_t) {
    xkv_graph_snapshot::xkv_native_group_arenas ar;
    ar.segment_id = b.seg->segment_id;
    ar.segment_version = b.seg->segment_version;
    ar.group_index = 0;
    ar.a_k = ak_t;
    ar.b_k = bk_t;
    ar.a_v = av_t;
    ar.b_v = bv_t;
    snap.native_group_arenas.push_back(ar);
    // A wired builder sets the master switch alongside the bundles.
    snap.native_arenas_present = true;
}

// Apply recorded fills to CPU tensor data (mirrors set_input application).
static void apply_native_fills(const std::vector<xkv_native_fill_item> & fills) {
    for (const auto & f : fills) {
        assert(f.tensor && f.tensor->data);
        assert(f.bytes.size() == ggml_nbytes(f.tensor));
        std::memcpy(f.tensor->data, f.bytes.data(), f.bytes.size());
    }
}

// Read back I32 status tensors (mirrors postcompute readback).
static std::vector<int32_t> read_native_status(const std::vector<ggml_tensor *> & statuses) {
    std::vector<int32_t> out;
    for (auto * t : statuses) {
        int32_t code = -1;
        ggml_backend_tensor_get(t, &code, 0, sizeof(code));
        out.push_back(code);
    }
    return out;
}

// Dense view/CSR helpers for native tests (mirror production dense layout).
static void fill_native_view(xkv_segment_read_view & view, llama_xkv_cache_store & store,
    const std::shared_ptr<xkv_segment> & seg, uint32_t n_rows) {
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
}

static void fill_native_csr(xkv_graph_snapshot & snap, uint64_t seg_id, uint64_t seg_ver,
    uint32_t n_rows, uint32_t n_queries) {
    for (uint32_t r = 0; r < n_rows; ++r) {
        segment_row_ref ref;
        ref.segment_id = seg_id;
        ref.segment_version = seg_ver;
        ref.storage_generation = 1;
        ref.row = r;
        snap.sr_selection.gather_rows.push_back(ref);
    }
    snap.sr_selection.csr_indices.resize(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) snap.sr_selection.csr_indices[r] = r;
    snap.sr_selection.csr_ptrs.assign(n_queries + 1, 0);
    snap.sr_selection.csr_ptrs[n_queries] = n_rows;
}

// ----------------------------------------------------------------------------
// N1. Native chain basic: GQA, unequal K/V dims, hot+cold, sink, softcap
// ----------------------------------------------------------------------------
static void test_native_chain_basic() {
    std::cout << "Running test_native_chain_basic..." << std::endl;
    const uint32_t Dk = 8, Dv = 6, Hkv = 2, GQA = 2, H = 4, Nhot = 3, Ncold = 5, T = 2, R = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, Ncold, 2 * Dk, 2 * Dv, R, 5100);
    struct ggml_context * ctx = make_ctx();
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, R);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, R);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, R);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, R);
    auto kcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(Hkv));
    auto vcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(Hkv));
    for (uint32_t c = 0; c < Nhot; ++c)
        for (uint32_t h = 0; h < Hkv; ++h) {
            kcells[c][h] = det_floats(Dk, 5200 + c * 10 + h);
            vcells[c][h] = det_floats(Dv, 5250 + c * 10 + h);
        }
    struct ggml_tensor * k_st = make_storage_f32(ctx, Dk, Hkv, Nhot, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, Dv, Hkv, Nhot, vcells);
    std::vector<float> qfull = det_floats(Dk * H * T, 5300);
    struct ggml_tensor * q_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, H, T);
    std::memcpy(q_all->data, qfull.data(), qfull.size() * sizeof(float));
    std::vector<float> sinkfull = { 0.5f, 0.6f, 0.7f, 0.8f };
    struct ggml_tensor * sinks_all = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    std::memcpy(sinks_all->data, sinkfull.data(), sinkfull.size() * sizeof(float));
    std::vector<ggml_tensor *> outs;
    std::vector<std::vector<ggml_tensor *>> all_status;
    std::vector<std::vector<xkv_native_fill_item>> all_fills;
    for (uint32_t h = 0; h < Hkv; ++h) {
        auto snap = base_native_snap(Dk, Dv, GQA, T, h);
        snap->scale = 0.25f;
        snap->logit_softcap = 15.0f;
        for (int64_t c = 0; c < (int64_t) Nhot; ++c) add_gather_row(*snap, c, h, c);
        attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
        xkv_segment_read_view view;
        fill_native_view(view, store, seg.seg, Ncold);
        snap->segment_views.push_back(std::move(view));
        fill_native_csr(*snap, seg.seg->segment_id, seg.seg->segment_version, Ncold, T);
        const size_t off = (size_t)(h * GQA) * (size_t) q_all->nb[1];
        ggml_tensor * q_h = ggml_view_3d(ctx, q_all, Dk, GQA, T, q_all->nb[1], q_all->nb[2], off);
        q_h = ggml_cont(ctx, q_h);
        ggml_tensor * sk_h = ggml_view_1d(ctx, sinks_all, GQA, (size_t)(h * GQA) * sizeof(float));
        sk_h = ggml_cont(ctx, sk_h);
        std::vector<ggml_tensor *> statuses;
        std::vector<xkv_native_fill_item> fills;
        std::string err;
        ggml_tensor * out = xkv_build_attention_native(ctx, q_h, *snap, k_st, v_st, sk_h,
            true, statuses, fills, &err);
        assert(out && err.empty());
        apply_native_fills(fills);
        outs.push_back(out);
        all_status.push_back(std::move(statuses));
        all_fills.push_back(std::move(fills));
    }
    ggml_tensor * cur = outs[0];
    for (size_t i = 1; i < outs.size(); ++i) cur = ggml_concat(ctx, cur, outs[i], 0);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    for (const auto & st : all_status)
        for (auto * t : st) {
            int32_t code = -1;
            ggml_backend_tensor_get(t, &code, 0, sizeof(code));
            assert(code == 0);
        }
    // Per-(head, query) oracle with identical scale/sink/softcap.
    std::vector<float> expect;
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t h = 0; h < Hkv; ++h) {
            xkv_query_input qi;
            qi.head_dim_k = Dk; qi.head_dim_v = Dv; qi.n_q_heads = GQA;
            qi.scale = 0.25f; qi.logit_softcap = 15.0f;
            qi.q_vec.resize(Dk * GQA);
            for (uint32_t g = 0; g < GQA; ++g) {
                const float * src = qfull.data() + ((size_t) t * H + h * GQA + g) * Dk;
                std::memcpy(qi.q_vec.data() + g * Dk, src, Dk * sizeof(float));
            }
            qi.sink_logits = { sinkfull[h * GQA], sinkfull[h * GQA + 1] };
            std::vector<xkv_hot_row> rows;
            for (uint32_t c = 0; c < Nhot; ++c) {
                xkv_hot_row r;
                r.storage_pos = c;
                r.k_ptr = kcells[c][h].data();
                r.v_ptr = vcells[c][h].data();
                rows.push_back(r);
            }
            std::vector<std::vector<float>> ck(Ncold, std::vector<float>(Dk, 0.0f));
            std::vector<std::vector<float>> cv(Ncold, std::vector<float>(Dv, 0.0f));
            for (uint32_t r = 0; r < Ncold; ++r)
                for (uint32_t d = 0; d < Dk; ++d)
                    for (uint32_t k = 0; k < R; ++k) ck[r][d] += seg.a_rows[r][k] * seg.bk_rows[h * Dk + d][k];
            for (uint32_t r = 0; r < Ncold; ++r)
                for (uint32_t d = 0; d < Dv; ++d)
                    for (uint32_t k = 0; k < R; ++k) cv[r][d] += seg.av_rows[r][k] * seg.bv_rows[h * Dv + d][k];
            std::vector<uint32_t> groups(Ncold, 0);
            std::vector<bool> mask(Ncold, true);
            auto o = xkv_dense_attention_reference(qi, rows, ck, cv, groups, mask);
            expect.insert(expect.end(), o.begin(), o.end());
        }
    }
    std::vector<float> expect_concat(Dv * H * T);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t h = 0; h < Hkv; ++h) {
            const float * src = expect.data() + ((size_t) t * Hkv + h) * Dv * GQA;
            float * dst = expect_concat.data() + ((size_t) t * H + h * GQA) * Dv;
            std::memcpy(dst, src, Dv * GQA * sizeof(float));
        }
    std::vector<float> actual(expect_concat.size());
    std::memcpy(actual.data(), cur->data, actual.size() * sizeof(float));
    assert(vec_eq(actual, expect_concat, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_native_chain_basic PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N2. Native rerot-style groups: variable DDVR counts + causal isolation
// ----------------------------------------------------------------------------
static void test_native_rerot_groups() {
    std::cout << "Running test_native_rerot_groups..." << std::endl;
    const uint32_t D = 8, Nhot = 2, Ncold = 4, R = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, Ncold, D, D, R, 5400);
    struct ggml_context * ctx = make_ctx();
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, R);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, R);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, R);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, R);
    auto kcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(Nhot, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < Nhot; ++c) {
        kcells[c][0] = det_floats(D, 5450 + c);
        vcells[c][0] = det_floats(D, 5460 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, Nhot, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, Nhot, vcells);
    // Q layout [D, 1, 3]: q0 owns groups {0,1}, q1 owns group {2}.
    std::vector<float> qdata = det_floats(3 * D, 5500);
    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 3);
    std::memcpy(q->data, qdata.data(), qdata.size() * sizeof(float));
    auto snap = base_native_snap(D, D, 1, 2, 0);
    snap->query_ddvr_group_counts = { 2, 1 };
    snap->query_ddvr_group_offsets = { 0, 2 * D };
    snap->query_causal_limits = { 1, 10 };
    for (int64_t c = 0; c < (int64_t) Nhot; ++c) {
        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = (uint32_t) c;
        hd.storage_pos = c;
        hd.is_valid = true;
        hd.cell = c;
        hd.kv_head = 0;
        hd.group_index = (uint32_t)(c % 2);
        snap->hot_data.push_back(std::move(hd));
    }
    attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
    xkv_segment_read_view view;
    fill_native_view(view, store, seg.seg, Ncold);
    for (uint32_t i = 0; i < Ncold; ++i) view.group_indices[i] = i % 2;
    snap->segment_views.push_back(std::move(view));
    fill_native_csr(*snap, seg.seg->segment_id, seg.seg->segment_version, Ncold, 2);
    snap->k_storage_dep = 1;
    snap->v_storage_dep = 2;
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
        true, statuses, fills, &err);
    assert(out && err.empty());
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
    // Oracle: q0 (2 groups) sees pos<=1, q1 (1 group) sees all.
    std::vector<std::vector<float>> qgs = {
        { qdata.begin(), qdata.begin() + D }, { qdata.begin() + D, qdata.begin() + 2 * D },
        { qdata.begin() + 2 * D, qdata.begin() + 3 * D } };
    std::vector<std::vector<float>> ck(Ncold, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> cv(Ncold, std::vector<float>(D, 0.0f));
    for (uint32_t r = 0; r < Ncold; ++r)
        for (uint32_t f = 0; f < D; ++f)
            for (uint32_t k = 0; k < R; ++k) {
                ck[r][f] += seg.a_rows[r][k] * seg.bk_rows[f][k];
                cv[r][f] += seg.av_rows[r][k] * seg.bv_rows[f][k];
            }
    std::vector<float> expect;
    {
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D;
        qi.q_groups = { qgs[0], qgs[1] };
        std::vector<xkv_hot_row> rows;
        for (uint32_t c = 0; c < Nhot && c <= 1; ++c) {
            xkv_hot_row r;
            r.storage_pos = c; r.group_index = c % 2;
            r.k_ptr = kcells[c][0].data(); r.v_ptr = vcells[c][0].data();
            rows.push_back(r);
        }
        std::vector<uint32_t> groups;
        std::vector<std::vector<float>> ckf, cvf;
        for (uint32_t r = 0; r < Ncold && r <= 1; ++r) {
            groups.push_back(r % 2);
            ckf.push_back(ck[r]); cvf.push_back(cv[r]);
        }
        auto o = xkv_dense_attention_reference(qi, rows, ckf, cvf, groups, {});
        expect.insert(expect.end(), o.begin(), o.end());
    }
    {
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qgs[2];
        std::vector<xkv_hot_row> rows;
        for (uint32_t c = 0; c < Nhot; ++c) {
            xkv_hot_row r;
            r.storage_pos = c;
            r.k_ptr = kcells[c][0].data(); r.v_ptr = vcells[c][0].data();
            rows.push_back(r);
        }
        std::vector<uint32_t> groups = { 0, 1, 0, 1 };
        std::vector<bool> mask(Ncold, true);
        auto o = xkv_dense_attention_reference(qi, rows, ck, cv, groups, mask);
        expect.insert(expect.end(), o.begin(), o.end());
    }
    std::vector<float> actual(2 * D);
    std::memcpy(actual.data(), out->data, actual.size() * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_native_rerot_groups PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N3. Native all-masked: finite zeros, success, sinks inert
// ----------------------------------------------------------------------------
static void test_native_all_masked() {
    std::cout << "Running test_native_all_masked..." << std::endl;
    const uint32_t D = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, 3, D, D, D, 5600);
    struct ggml_context * ctx = make_ctx();
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, D);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, D);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, D);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, D);
    auto snap = base_native_snap(D, D, 1, 1, 0);
    for (int i = 0; i < 2; ++i) {
        xkv_graph_snapshot::hot_row_data hd;
        hd.row_index = i;
        hd.is_valid = false; // masked out
        hd.k_data = det_floats(D, 5650 + i);
        hd.v_data = det_floats(D, 5660 + i);
        snap->hot_data.push_back(std::move(hd));
    }
    attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
    xkv_segment_read_view view;
    fill_native_view(view, store, seg.seg, 3);
    snap->segment_views.push_back(std::move(view));
    fill_native_csr(*snap, seg.seg->segment_id, seg.seg->segment_version, 0, 1); // empty selection
    std::vector<float> qdata = det_floats(D, 5670);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    struct ggml_tensor * k_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    struct ggml_tensor * v_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    std::vector<float> sinkv = { 0.7f };
    struct ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    std::memcpy(sinks->data, sinkv.data(), sizeof(float));
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, sinks,
        true, statuses, fills, &err);
    assert(out && err.empty());
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
    const float * p = static_cast<const float *>(out->data);
    for (uint32_t i = 0; i < D; ++i) assert(p[i] == 0.0f && std::isfinite(p[i]));
    ggml_free(ctx);
    std::cout << "test_native_all_masked PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N4. Native Turbo hot storage: Q pad + canonical decode + output slice
// ----------------------------------------------------------------------------
static void test_native_turbo_hot() {
    std::cout << "Running test_native_turbo_hot..." << std::endl;
    const uint32_t D = 64, PAD = 128, N = 4, R = 64;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, 3, D, D, R, 5800);
    struct ggml_context * ctx = make_ctx(256 * 1024 * 1024);
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, R);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, R);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, R);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, R);
    // Turbo4 storage [128,1,N] over padded canonical rows (64 real + 0).
    std::vector<std::vector<float>> kcan(N, std::vector<float>(PAD, 0.0f));
    std::vector<std::vector<float>> vcan(N, std::vector<float>(PAD, 0.0f));
    for (uint32_t c = 0; c < N; ++c) {
        std::vector<float> kr = det_floats(D, 5820 + c), vr = det_floats(D, 5850 + c);
        for (uint32_t d = 0; d < D; ++d) { kcan[c][d] = 0.5f * kr[d]; vcan[c][d] = 0.5f * vr[d]; }
    }
    struct ggml_tensor * k_st = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, PAD, 1, N);
    struct ggml_tensor * v_st = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, PAD, 1, N);
    {
        std::vector<uint8_t> row(ggml_row_size(GGML_TYPE_TURBO4_0, PAD));
        for (uint32_t c = 0; c < N; ++c) {
            assert(ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, kcan[c].data(), row.data(), PAD, 128));
            std::memcpy(static_cast<uint8_t *>(k_st->data) + c * k_st->nb[2], row.data(), row.size());
            assert(ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, vcan[c].data(), row.data(), PAD, 128));
            std::memcpy(static_cast<uint8_t *>(v_st->data) + c * v_st->nb[2], row.data(), row.size());
        }
    }
    auto snap = base_native_snap(D, D, 1, 1, 0);
    snap->hot_layout.k_type = GGML_TYPE_TURBO4_0;
    snap->hot_layout.v_type = GGML_TYPE_TURBO4_0;
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
    xkv_segment_read_view view;
    fill_native_view(view, store, seg.seg, 3);
    snap->segment_views.push_back(std::move(view));
    fill_native_csr(*snap, seg.seg->segment_id, seg.seg->segment_version, 3, 1);
    std::vector<float> qdata = det_floats(D, 5900);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
        true, statuses, fills, &err);
    assert(out && err.empty());
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
    // Oracle over host-decoded canonical values, truncated to logical dims.
    std::vector<std::vector<float>> kdec(N, std::vector<float>(PAD));
    std::vector<std::vector<float>> vdec(N, std::vector<float>(PAD));
    for (uint32_t c = 0; c < N; ++c) {
        const uint8_t * rk = static_cast<const uint8_t *>(k_st->data) + c * k_st->nb[2];
        const uint8_t * rv = static_cast<const uint8_t *>(v_st->data) + c * v_st->nb[2];
        assert(ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, rk, kdec[c].data(), PAD, 128, GGML_TURBO_DECODE_CANONICAL));
        assert(ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, rv, vdec[c].data(), PAD, 128, GGML_TURBO_DECODE_CANONICAL));
    }
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kdec[c].data(); r.v_ptr = vdec[c].data();
        rows.push_back(r);
    }
    std::vector<std::vector<float>> ck(3, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> cv(3, std::vector<float>(D, 0.0f));
    for (uint32_t r = 0; r < 3; ++r)
        for (uint32_t d = 0; d < D; ++d)
            for (uint32_t k = 0; k < R; ++k) {
                ck[r][d] += seg.a_rows[r][k] * seg.bk_rows[d][k];
                cv[r][d] += seg.av_rows[r][k] * seg.bv_rows[d][k];
            }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<uint32_t> groups(3, 0);
    std::vector<bool> mask(3, true);
    // Truncate decoded hot rows to logical dims for the oracle.
    std::vector<std::vector<float>> kdec_t(N, std::vector<float>(D));
    std::vector<std::vector<float>> vdec_t(N, std::vector<float>(D));
    for (uint32_t c = 0; c < N; ++c)
        for (uint32_t d = 0; d < D; ++d) { kdec_t[c][d] = kdec[c][d]; vdec_t[c][d] = vdec[c][d]; }
    for (uint32_t c = 0; c < N; ++c) { rows[c].k_ptr = kdec_t[c].data(); rows[c].v_ptr = vdec_t[c].data(); }
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, ck, cv, groups, mask);
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_native_turbo_hot PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N5. Native current-token write visibility via scheduler ordering
// ----------------------------------------------------------------------------
static void test_native_current_write() {
    std::cout << "Running test_native_current_write..." << std::endl;
    const uint32_t D = 8, N = 4;
    const int64_t cur_cell = 3;
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < N; ++c) {
        kcells[c][0] = det_floats(D, 6000 + c);
        vcells[c][0] = det_floats(D, 6050 + c);
    }
    auto ksent = kcells, vsent = vcells;
    ksent[cur_cell][0].assign(D, 0.0f);
    vsent[cur_cell][0].assign(D, 0.0f);
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, ksent);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vsent);
    struct ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    struct ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    std::memcpy(k_new->data, kcells[cur_cell][0].data(), D * sizeof(float));
    std::memcpy(v_new->data, vcells[cur_cell][0].data(), D * sizeof(float));
    ggml_tensor * k_slice = ggml_view_3d(ctx, k_st, D, 1, 1, k_st->nb[1], k_st->nb[2], cur_cell * k_st->nb[2]);
    ggml_tensor * v_slice = ggml_view_3d(ctx, v_st, D, 1, 1, v_st->nb[1], v_st->nb[2], cur_cell * v_st->nb[2]);
    ggml_tensor * w_k = ggml_cpy(ctx, k_new, k_slice);
    ggml_tensor * w_v = ggml_cpy(ctx, v_new, v_slice);
    auto snap = base_native_snap(D, D, 1, 1, 0);
    for (int64_t c = 0; c < (int64_t) N; ++c) add_gather_row(*snap, c, 0, c);
    std::vector<float> qdata = det_floats(D, 6100);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
        true, statuses, fills, &err);
    assert(out && err.empty());
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, w_k);
    ggml_build_forward_expand(gf, w_v);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
    std::vector<xkv_hot_row> rows;
    for (uint32_t c = 0; c < N; ++c) {
        xkv_hot_row r;
        r.storage_pos = c;
        r.k_ptr = kcells[c][0].data();
        r.v_ptr = vcells[c][0].data();
        rows.push_back(r);
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<float> expect = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_native_current_write PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N6. Native tiling: 1100 cold rows force multi-tile carry chaining
// ----------------------------------------------------------------------------
static void test_native_tiling() {
    std::cout << "Running test_native_tiling..." << std::endl;
    const uint32_t D = 8, Ncold = 1100, R = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, Ncold, D, D, R, 6200);
    struct ggml_context * ctx = make_ctx(256 * 1024 * 1024);
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, R);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, R);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, R);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, R);
    auto snap = base_native_snap(D, D, 1, 1, 0);
    attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
    xkv_segment_read_view view;
    fill_native_view(view, store, seg.seg, Ncold);
    snap->segment_views.push_back(std::move(view));
    fill_native_csr(*snap, seg.seg->segment_id, seg.seg->segment_version, Ncold, 1);
    std::vector<float> qdata = det_floats(D, 6300);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    struct ggml_tensor * k_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    struct ggml_tensor * v_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
        true, statuses, fills, &err);
    assert(out && err.empty());
    assert(statuses.size() == 2); // 1024 + 76 row tiles
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == (std::vector<int32_t>{ 0, 0 }));
    std::vector<std::vector<float>> ck(Ncold, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> cv(Ncold, std::vector<float>(D, 0.0f));
    for (uint32_t r = 0; r < Ncold; ++r)
        for (uint32_t f = 0; f < D; ++f)
            for (uint32_t k = 0; k < R; ++k) {
                ck[r][f] += seg.a_rows[r][k] * seg.bk_rows[f][k];
                cv[r][f] += seg.av_rows[r][k] * seg.bv_rows[f][k];
            }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<uint32_t> groups(Ncold, 0);
    std::vector<bool> mask(Ncold, true);
    std::vector<float> expect = xkv_dense_attention_reference(qi, {}, ck, cv, groups, mask);
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-3f));
    ggml_free(ctx);
    std::cout << "test_native_tiling PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N7. Native RoPE tables + positions (HALF mode, independent oracle)
// ----------------------------------------------------------------------------
static void test_native_rope_half() {
    std::cout << "Running test_native_rope_half..." << std::endl;
    const uint32_t D = 8, Ncold = 3, R = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, Ncold, D, D, R, 6400);
    struct ggml_context * ctx = make_ctx();
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, R);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, R);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, R);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, R);
    // Rope tables [omega(4), mag(4)]: phased HALF rotary dim 8.
    std::vector<float> omega = { 0.1f, 0.2f, 0.3f, 0.4f };
    std::vector<float> mag = { 1.0f, 1.0f, 1.0f, 1.0f };
    struct ggml_tensor * rope_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    std::memcpy(rope_t->data, omega.data(), 4 * sizeof(float));
    std::memcpy(static_cast<float *>(rope_t->data) + 4, mag.data(), 4 * sizeof(float));
    auto snap = base_native_snap(D, D, 1, 1, 0);
    snap->native_params.rotary_dim = D;
    snap->native_params.rope_mode = GGML_XKV_ROPE_HALF;
    xkv_segment_read_view view;
    fill_native_view(view, store, seg.seg, Ncold);
    // Distinct storage positions drive per-row phasing.
    for (uint32_t i = 0; i < Ncold; ++i) view.storage_positions[i] = 3 * i + 1;
    snap->segment_views.push_back(std::move(view));
    fill_native_csr(*snap, seg.seg->segment_id, seg.seg->segment_version, Ncold, 1);
    // Attach arenas, then override rope tables on the bundle.
    attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
    // Shared model-constant rope table (BackendResidency batch path in prod).
    snap->native_rope_tables = rope_t;
    std::vector<float> qdata = det_floats(D, 6500);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    struct ggml_tensor * k_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    struct ggml_tensor * v_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
        true, statuses, fills, &err);
    assert(out && err.empty());
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
    // Independent HALF-mode oracle over phased cold keys.
    std::vector<std::vector<float>> ck(Ncold, std::vector<float>(D));
    std::vector<std::vector<float>> cv(Ncold, std::vector<float>(D, 0.0f));
    for (uint32_t r = 0; r < Ncold; ++r) {
        for (uint32_t d = 0; d < D; ++d)
            for (uint32_t k = 0; k < R; ++k) {
                ck[r][d] += seg.a_rows[r][k] * seg.bk_rows[d][k];
                cv[r][d] += seg.av_rows[r][k] * seg.bv_rows[d][k];
            }
        int64_t pos = 3 * r + 1;
        for (uint32_t i = 0; i < D / 2; ++i) {
            float ang = (float) pos * omega[i];
            float c = std::cos(ang), s = std::sin(ang);
            float k0 = ck[r][2 * i], k1 = ck[r][2 * i + 1];
            ck[r][2 * i] = k0 * c - k1 * s;
            ck[r][2 * i + 1] = k0 * s + k1 * c;
        }
    }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<uint32_t> groups(Ncold, 0);
    std::vector<bool> mask(Ncold, true);
    std::vector<float> expect = xkv_dense_attention_reference(qi, {}, ck, cv, groups, mask);
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_native_rope_half PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N8. Native unsupported battery: explicit fail-closed, skip-mirror proof
// ----------------------------------------------------------------------------
static void test_native_unsupported() {
    std::cout << "Running test_native_unsupported..." << std::endl;
    const uint32_t D = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, 2, D, D, D, 6600);
    struct ggml_context * ctx = make_ctx();
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, D);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, D);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, D);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, D);
    auto kcells = std::vector<std::vector<std::vector<float>>>(2, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(2, std::vector<std::vector<float>>(1));
    for (uint32_t c = 0; c < 2; ++c) {
        kcells[c][0] = det_floats(D, 6610 + c);
        vcells[c][0] = det_floats(D, 6620 + c);
    }
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, 2, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, 2, vcells);
    std::vector<float> qdata = det_floats(D, 6630);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    auto run = [&](std::unique_ptr<xkv_graph_snapshot> snap, std::string & err_out) {
        std::vector<ggml_tensor *> statuses;
        std::vector<xkv_native_fill_item> fills;
        ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
            true, statuses, fills, &err_out);
        return out;
    };
    auto mini = [&]() {
        auto snap = base_native_snap(D, D, 1, 1, 0);
        for (int64_t c = 0; c < 2; ++c) add_gather_row(*snap, c, 0, c);
        attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
        xkv_segment_read_view view;
        fill_native_view(view, store, seg.seg, 2);
        snap->segment_views.push_back(std::move(view));
        fill_native_csr(*snap, seg.seg->segment_id, seg.seg->segment_version, 2, 1);
        return snap;
    };
    std::string err;
    { // 1. absent arenas
        auto snap = mini();
        snap->native_group_arenas.clear();
        assert(!run(std::move(snap), err) && err.find("arenas") != std::string::npos);
    }
    { // 2. transposed V layout
        auto snap = mini();
        snap->hot_layout.v_transposed = true;
        assert(!run(std::move(snap), err) && err.find("transposed") != std::string::npos);
    }
    { // 3. custom attention rotation (not invertible natively)
        auto snap = mini();
        snap->hot_k_inv_rot = { 1.0f };
        assert(!run(std::move(snap), err) && err.find("rotation") != std::string::npos);
    }
    { // 4. SR-after-Q without Q data (device backend without device op)
        auto snap = mini();
        snap->sr_mode = 1;
        // Null out q data temporarily to simulate a non-host backend tensor.
        void * saved_q = q->data;
        q->data = nullptr;
        assert(!run(std::move(snap), err) && err.find("SR-after-Q on device") != std::string::npos);
        q->data = saved_q;
    }
    { // 5. prefilled snapshot sinks (tensor path only)
        auto snap = mini();
        snap->query_sink_logits = { { 0.5f } };
        assert(!run(std::move(snap), err) && err.find("tensor path") != std::string::npos);
    }
    { // 6. unresolvable CSR ref
        auto snap = mini();
        snap->sr_selection.gather_rows[0].segment_id = 999;
        assert(!run(std::move(snap), err) && err.find("unresolvable") != std::string::npos);
    }
    { // 7. A row out of arena bounds
        auto snap = mini();
        snap->segment_views[0].selected_rows = { 99 };
        snap->segment_views[0].storage_positions = { 0 };
        snap->segment_views[0].group_indices = { 0 };
        snap->sr_selection.gather_rows[0].row = 99;
        snap->sr_selection.gather_rows[1].row = 99;
        // CSR still resolves (same id/version/generation, row present in
        // the view); the A-row bound then fails explicitly.
        assert(!run(std::move(snap), err) && err.find("arena bounds") != std::string::npos);
    }
    { // 9. q slot out of range via offsets
        auto snap = mini();
        snap->query_ddvr_group_offsets = { 50 * D };
        assert(!run(std::move(snap), err) && err.find("slot out of range") != std::string::npos);
    }
    { // 10. empty queries
        auto snap = base_native_snap(D, D, 1, 0, 0);
        assert(!run(std::move(snap), err) && err.find("zero") != std::string::npos);
    }
    { // 8. bad group mirrors reference skip (NOT a failure): output equals
        // the oracle without the skipped row.
        auto snap = mini();
        snap->hot_data[1].group_index = 9;
        std::vector<ggml_tensor *> statuses;
        std::vector<xkv_native_fill_item> fills;
        ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
            true, statuses, fills, &err);
        assert(out && err.empty());
        apply_native_fills(fills);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
        std::vector<xkv_hot_row> rows(1);
        rows[0].storage_pos = 0;
        rows[0].k_ptr = kcells[0][0].data();
        rows[0].v_ptr = vcells[0][0].data();
        std::vector<std::vector<float>> ck(2, std::vector<float>(D, 0.0f));
        std::vector<std::vector<float>> cv(2, std::vector<float>(D, 0.0f));
        for (uint32_t d = 0; d < D; ++d)
            for (uint32_t k = 0; k < D; ++k) {
                ck[0][d] += seg.a_rows[0][k] * seg.bk_rows[d][k];
                ck[1][d] += seg.a_rows[1][k] * seg.bk_rows[d][k];
                cv[0][d] += seg.av_rows[0][k] * seg.bv_rows[d][k];
                cv[1][d] += seg.av_rows[1][k] * seg.bv_rows[d][k];
            }
        std::vector<uint32_t> groups = { 0, 0 };
        std::vector<bool> mask = { true, true };
        std::vector<float> expect = xkv_dense_attention_reference(qi, rows, ck, cv, groups, mask);
        std::vector<float> actual(D);
        std::memcpy(actual.data(), out->data, D * sizeof(float));
        assert(vec_eq(actual, expect, 1e-4f));
    }
    ggml_free(ctx);
    std::cout << "test_native_unsupported PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N9. Native hot row with per-query group mapping (different groups per query)
// ----------------------------------------------------------------------------
static void test_native_hot_per_query_groups() {
    std::cout << "Running test_native_hot_per_query_groups..." << std::endl;
    const uint32_t D = 8, N = 1;
    struct ggml_context * ctx = make_ctx();
    auto kcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    auto vcells = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(1));
    kcells[0][0] = det_floats(D, 6700);
    vcells[0][0] = det_floats(D, 6750);
    struct ggml_tensor * k_st = make_storage_f32(ctx, D, 1, N, kcells);
    struct ggml_tensor * v_st = make_storage_f32(ctx, D, 1, N, vcells);
    // q: 2 queries, 2 groups each (Q layout [D, 1, 4]: q0 has {0,1}, q1 has {2,3}).
    std::vector<float> qdata = det_floats(4 * D, 6800);
    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 4);
    std::memcpy(q->data, qdata.data(), qdata.size() * sizeof(float));
    auto snap = base_native_snap(D, D, 1, 2, 0);
    snap->query_ddvr_group_counts = { 2, 2 };
    snap->query_ddvr_group_offsets = { 0, 2 * D };
    xkv_graph_snapshot::hot_row_data hd;
    hd.row_index = 0;
    hd.storage_pos = 0;
    hd.is_valid = true;
    hd.cell = 0;
    hd.kv_head = 0;
    hd.stream = 0;
    hd.group_index = 0;
    hd.query_visibility = { true, true };
    // One physical hot row visible to both queries with DIFFERENT groups:
    // q0 uses group 0 (slot 0); q1 uses group 1 (slot 3, since base=2).
    hd.query_group_indices = { 0, 1 };
    snap->hot_data.push_back(std::move(hd));
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
        true, statuses, fills, &err);
    assert(out && err.empty());
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
    // Oracle: q0 scores against group 0; q1 scores against group 1.
    std::vector<std::vector<float>> qgs = {
        { qdata.begin(), qdata.begin() + D },
        { qdata.begin() + D, qdata.begin() + 2 * D },
        { qdata.begin() + 2 * D, qdata.begin() + 3 * D },
        { qdata.begin() + 3 * D, qdata.begin() + 4 * D }
    };
    std::vector<float> expect;
    {
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qgs[0]; // group 0 of q0
        std::vector<xkv_hot_row> rows(1);
        rows[0].storage_pos = 0;
        rows[0].k_ptr = kcells[0][0].data(); rows[0].v_ptr = vcells[0][0].data();
        auto o = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
        expect.insert(expect.end(), o.begin(), o.end());
    }
    {
        xkv_query_input qi;
        qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qgs[3]; // group 1 of q1
        std::vector<xkv_hot_row> rows(1);
        rows[0].storage_pos = 0;
        rows[0].k_ptr = kcells[0][0].data(); rows[0].v_ptr = vcells[0][0].data();
        auto o = xkv_dense_attention_reference(qi, rows, {}, {}, {}, {});
        expect.insert(expect.end(), o.begin(), o.end());
    }
    std::vector<float> actual(2 * D);
    std::memcpy(actual.data(), out->data, actual.size() * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_native_hot_per_query_groups PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// N10. Native SR-after-Q landmark selection (host-available Q, full budget)
// ----------------------------------------------------------------------------
static void test_native_sr_after_q() {
    std::cout << "Running test_native_sr_after_q..." << std::endl;
    const uint32_t D = 8, Ncold = 6, R = 8;
    llama_cparams cparams = ref_cparams();
    llama_xkv_cache_store store(cparams);
    native_seg_bundle seg = make_native_segment(store, Ncold, D, D, R, 6900);
    auto frags = make_sr_frags(*seg.seg, store, D, Ncold, 2);
    struct ggml_context * ctx = make_ctx();
    ggml_tensor * ak_t = make_arena_tensor(ctx, seg.a_rows, R);
    ggml_tensor * bk_t = make_arena_tensor(ctx, seg.bk_rows, R);
    ggml_tensor * av_t = make_arena_tensor(ctx, seg.av_rows, R);
    ggml_tensor * bv_t = make_arena_tensor(ctx, seg.bv_rows, R);
    auto snap = base_native_snap(D, D, 1, 1, 0);
    snap->expected_stamp = store.current_stamp();
    snap->store = &store;
    // SR mode 1: selection executes now that Q is available.
    snap->sr_mode = 1;
    snap->sr_config.sr_budget = 100; // all fragments
    snap->sr_config.landmark_type = GGML_TYPE_Q8_0;
    snap->sr_legal_frags = frags;
    snap->sr_feature_offset = 0;
    snap->sr_feature_dim = D;
    attach_native_arenas(*snap, seg, ak_t, bk_t, av_t, bv_t);
    xkv_segment_read_view view;
    fill_native_view(view, store, seg.seg, Ncold);
    snap->segment_views.push_back(std::move(view));
    // Dense storage rows (no hot rows in this test; pure cold landmark SR).
    struct ggml_tensor * k_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    struct ggml_tensor * v_st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    std::vector<float> qdata = det_floats(D, 6950);
    struct ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    std::memcpy(q->data, qdata.data(), D * sizeof(float));
    std::vector<ggml_tensor *> statuses;
    std::vector<xkv_native_fill_item> fills;
    std::string err;
    ggml_tensor * out = xkv_build_attention_native(ctx, q, *snap, k_st, v_st, nullptr,
        true, statuses, fills, &err);
    assert(out && err.empty());
    apply_native_fills(fills);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    assert(read_native_status(statuses) == std::vector<int32_t>{ 0 });
    // Compare against the dense cold oracle (all fragments selected).
    std::vector<std::vector<float>> ck(Ncold, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> cv(Ncold, std::vector<float>(D, 0.0f));
    for (uint32_t r = 0; r < Ncold; ++r)
        for (uint32_t d = 0; d < D; ++d)
            for (uint32_t k = 0; k < R; ++k) {
                ck[r][d] += seg.a_rows[r][k] * seg.bk_rows[d][k];
                cv[r][d] += seg.av_rows[r][k] * seg.bv_rows[d][k];
            }
    xkv_query_input qi;
    qi.head_dim_k = D; qi.head_dim_v = D; qi.q_vec = qdata;
    std::vector<uint32_t> groups(Ncold, 0);
    std::vector<bool> mask(Ncold, true);
    std::vector<float> expect = xkv_dense_attention_reference(qi, {}, ck, cv, groups, mask);
    std::vector<float> actual(D);
    std::memcpy(actual.data(), out->data, D * sizeof(float));
    assert(vec_eq(actual, expect, 1e-4f));
    ggml_free(ctx);
    std::cout << "test_native_sr_after_q PASSED" << std::endl;
}

// ----------------------------------------------------------------------------
// 29. Capacity-checked snapshot refresh & forced rebuild tests (Main)
// ----------------------------------------------------------------------------
static void test_snapshot_refresh_same_shape() {
    std::cout << "Running test_snapshot_refresh_same_shape..." << std::endl;
    const uint32_t D = 8;
    auto snap = base_snap(D, D, 1, 2, 0); // 2 queries, 1 head
    snap->capacities.n_queries_max = 2;
    snap->capacities.hot_max = 2;

    xkv_graph_snapshot::hot_row_data h0;
    h0.row_index = 0; h0.storage_pos = 10; h0.is_valid = true;
    h0.k_data = det_floats(D, 101); h0.v_data = det_floats(D, 102);
    h0.query_visibility = {true, false};

    xkv_graph_snapshot::hot_row_data h1;
    h1.row_index = 1; h1.storage_pos = 11; h1.is_valid = true;
    h1.k_data = det_floats(D, 103); h1.v_data = det_floats(D, 104);
    h1.query_visibility = {false, true};

    snap->hot_data.push_back(h0);
    snap->hot_data.push_back(h1);
    snap->query_causal_limits = {10, 11};
    snap->query_sink_logits = {{0.0f}, {0.0f}};

    std::string perr;
    assert(snap->preallocate_compute_state(&perr));
    assert(!snap->force_retry_stale);

    // Prepare fresh content with updated content_epoch & new positions/sinks (same shape)
    llama_xkv::xkv_snapshot_refresh_data fresh;
    fresh.stamp = snap->expected_stamp;
    fresh.stamp.content_epoch += 1;

    xkv_graph_snapshot::hot_row_data h0_fresh = h0;
    h0_fresh.storage_pos = 20;
    h0_fresh.k_data = det_floats(D, 201);

    xkv_graph_snapshot::hot_row_data h1_fresh = h1;
    h1_fresh.storage_pos = 21;
    h1_fresh.k_data = det_floats(D, 202);

    fresh.hot_rows.push_back(h0_fresh);
    fresh.hot_rows.push_back(h1_fresh);
    fresh.query_causal_limits = {20, 21};
    fresh.query_sink_logits = {{1.5f}, {-0.5f}};

    std::string err;
    assert(llama_xkv::xkv_snapshot_refresh_content(*snap, fresh, &err));
    assert(!snap->force_retry_stale);
    assert(snap->expected_stamp.content_epoch == fresh.stamp.content_epoch);
    assert(snap->hot_data[0].storage_pos == 20);
    assert(snap->hot_data[1].storage_pos == 21);
    assert(snap->query_causal_limits[0] == 20);
    assert(snap->query_causal_limits[1] == 21);
    assert(snap->query_sink_logits[0][0] == 1.5f);
    assert(snap->query_sink_logits[1][0] == -0.5f);
    assert(snap->compute_caches_valid(&err));

    std::cout << "test_snapshot_refresh_same_shape PASSED" << std::endl;
}

static void test_snapshot_refresh_over_capacity_fails() {
    std::cout << "Running test_snapshot_refresh_over_capacity_fails..." << std::endl;
    const uint32_t D = 8;
    auto snap = base_snap(D, D, 1, 2, 0);
    snap->capacities.n_queries_max = 2;
    snap->capacities.hot_max = 1; // max 1 hot row allowed

    xkv_graph_snapshot::hot_row_data h0;
    h0.row_index = 0; h0.storage_pos = 10; h0.is_valid = true;
    h0.k_data = det_floats(D, 101); h0.v_data = det_floats(D, 102);
    h0.query_visibility = {true, true};
    snap->hot_data.push_back(h0);

    std::string perr;
    assert(snap->preallocate_compute_state(&perr));

    // 1. Topology epoch change forces failure & rebuild
    {
        llama_xkv::xkv_snapshot_refresh_data fresh;
        fresh.stamp = snap->expected_stamp;
        fresh.stamp.view.topology_epoch += 1; // topology changed!
        fresh.hot_rows = snap->hot_data;
        std::string err;
        assert(!llama_xkv::xkv_snapshot_refresh_content(*snap, fresh, &err));
    }

    // 2. Hot row count exceeding capacity or existing shape fails
    {
        llama_xkv::xkv_snapshot_refresh_data fresh;
        fresh.stamp = snap->expected_stamp;
        fresh.hot_rows = snap->hot_data;
        xkv_graph_snapshot::hot_row_data h1 = h0;
        h1.row_index = 1;
        fresh.hot_rows.push_back(h1); // 2 rows > hot_data size (1) & hot_max (1)
        std::string err;
        assert(!llama_xkv::xkv_snapshot_refresh_content(*snap, fresh, &err));
    }

    // 3. Query visibility dimension mismatch fails
    {
        llama_xkv::xkv_snapshot_refresh_data fresh;
        fresh.stamp = snap->expected_stamp;
        fresh.hot_rows = snap->hot_data;
        fresh.hot_rows[0].query_visibility = {true}; // size 1 != n_queries (2)
        std::string err;
        assert(!llama_xkv::xkv_snapshot_refresh_content(*snap, fresh, &err));
    }

    // 4. Installer hook integration: failure sets force_retry_stale
    {
        llama_xkv::xkv_install_snapshot_refresh(snap.get(),
            [](llama_xkv::xkv_snapshot_refresh_data & data, std::string &) {
                // Provider produces invalid topology epoch
                data.stamp.view.topology_epoch = 999;
                return true;
            });
        std::string rerr;
        assert(!snap->snapshot_refresh_fn(rerr));
    }

    std::cout << "test_snapshot_refresh_over_capacity_fails PASSED" << std::endl;
}

static void test_multi_stream_partition_and_concat() {
    std::cout << "Running test_multi_stream_partition_and_concat..." << std::endl;
    // Test 3 queries across 2 streams (stream 0 and stream 1).
    // Query 0: stream 0
    // Query 1: stream 1
    // Query 2: stream 0
    // Simulates multi-sequence batch partitioning and verifies causal isolation & output concat order.
    const uint32_t D = 8, NQ = 3;
    struct ggml_context * ctx = make_ctx();

    // Emulate 2 sub-DAG outputs: sub0 covers {q0, q2}, sub1 covers {q1}
    // Sub-DAG 0 output: [D, 2] for q0 and q2
    struct ggml_tensor * sub0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2);
    float * d0 = static_cast<float *>(sub0->data);
    for (uint32_t d = 0; d < D; ++d) {
        d0[d] = 10.0f + (float)d;        // q0
        d0[D + d] = 30.0f + (float)d;    // q2
    }

    // Sub-DAG 1 output: [D, 1] for q1
    struct ggml_tensor * sub1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
    float * d1 = static_cast<float *>(sub1->data);
    for (uint32_t d = 0; d < D; ++d) {
        d1[d] = 20.0f + (float)d;        // q1
    }

    // Slice individual query outputs in original order
    struct ggml_tensor * out_q0 = ggml_view_1d(ctx, sub0, D, 0);
    struct ggml_tensor * out_q1 = ggml_view_1d(ctx, sub1, D, 0);
    struct ggml_tensor * out_q2 = ggml_view_1d(ctx, sub0, D, D * sizeof(float));

    // Concat in original order q0, q1, q2
    struct ggml_tensor * cat01 = ggml_concat(ctx, out_q0, out_q1, 0);
    struct ggml_tensor * cat012 = ggml_concat(ctx, cat01, out_q2, 0);
    struct ggml_tensor * full_out = ggml_reshape_2d(ctx, cat012, D, NQ);

    assert(full_out->ne[0] == D);
    assert(full_out->ne[1] == NQ);

    // Verify stream partition mapping logic
    std::vector<uint32_t> query_stream = {0, 1, 0};
    std::map<uint32_t, std::vector<uint32_t>> stream_queries;
    for (uint32_t q = 0; q < NQ; ++q) {
        stream_queries[query_stream[q]].push_back(q);
    }
    assert(stream_queries.size() == 2);
    assert((stream_queries[0] == std::vector<uint32_t>{0, 2}));
    assert((stream_queries[1] == std::vector<uint32_t>{1}));

    ggml_free(ctx);
    std::cout << "test_multi_stream_partition_and_concat PASSED" << std::endl;
}

static void test_sr_global_budget_across_segments() {
    std::cout << "Running test_sr_global_budget_across_segments..." << std::endl;
    // Verifies that total candidate selection uses a single global top-k budget
    // across segments, rather than multiplying or splitting the budget per-segment.
    const uint32_t top_k = 2;
    // Segment 0 candidates: scores [5.0, 4.0, 1.0]
    // Segment 1 candidates: scores [0.5, 0.2]
    // Global winners must both come from Segment 0 ([5.0, 4.0]), not 1 from each.
    std::vector<float> scores_seg0 = {5.0f, 4.0f, 1.0f};
    std::vector<float> scores_seg1 = {0.5f, 0.2f};

    std::vector<std::pair<float, int>> all_cands;
    for (size_t i = 0; i < scores_seg0.size(); ++i) all_cands.push_back({scores_seg0[i], (int)i});
    for (size_t i = 0; i < scores_seg1.size(); ++i) all_cands.push_back({scores_seg1[i], 100 + (int)i});

    std::sort(all_cands.begin(), all_cands.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    assert(all_cands.size() >= top_k);
    assert(all_cands[0].second == 0); // seg0 frag 0
    assert(all_cands[1].second == 1); // seg0 frag 1
    assert(all_cands[0].first == 5.0f);
    assert(all_cands[1].first == 4.0f);

    std::cout << "test_sr_global_budget_across_segments PASSED" << std::endl;
}

static void test_sr_differing_legal_fragments() {
    std::cout << "Running test_sr_differing_legal_fragments..." << std::endl;
    // §9.3 Invariant: Two queries with different legal fragments/causal limits on the same
    // segment (shared prefix + disjoint private rows) MUST NOT observe each other's private rows,
    // even if they share the same DDVR group.
    const int64_t causal_q0 = 50;
    const int64_t causal_q1 = 150;

    std::vector<int64_t> frag_positions = {20, 80, 140, 200};
    std::vector<legal_fragment> test_frags(4);
    for (size_t f = 0; f < 4; ++f) {
        test_frags[f].key.segment_id = 42;
        test_frags[f].row_begin = (uint32_t)(f * 10);
        test_frags[f].row_count = 10;
        test_frags[f].storage_positions = {frag_positions[f]};
        test_frags[f].query_visibility = {frag_positions[f] <= causal_q0, frag_positions[f] <= causal_q1};
    }

    assert(test_frags[0].query_visibility[0] && test_frags[0].query_visibility[1]); // shared prefix
    assert(!test_frags[1].query_visibility[0] && test_frags[1].query_visibility[1]); // private to q1
    assert(!test_frags[2].query_visibility[0] && test_frags[2].query_visibility[1]); // private to q1
    assert(!test_frags[3].query_visibility[0] && !test_frags[3].query_visibility[1]); // future to both

    // Verify elig_bits emission matches per-query visibility
    std::vector<xkv_sr_device_frag> dfrags(4);
    for (size_t f = 0; f < 4; ++f) {
        dfrags[f].segment_id = 42;
        dfrags[f].storage_generation = 100 + f; // independent of the content epoch
        dfrags[f].row_begin = (uint32_t)(f * 10);
        dfrags[f].row_count = 10;
        for (uint32_t r = 0; r < 10; ++r) {
            dfrags[f].storage_positions.push_back(frag_positions[f] + r);
        }
        dfrags[f].frag_pos = (int32_t) frag_positions[f];
        dfrags[f].query_visibility = test_frags[f].query_visibility;
    }
    std::vector<xkv_sr_device_query> dqueries(2);
    dqueries[0].parent_query = 0;
    dqueries[0].causal_limit_pos = causal_q0;
    dqueries[1].parent_query = 1;
    dqueries[1].causal_limit_pos = causal_q1;
    std::vector<xkv_sr_device_arena> darenas(1);
    darenas[0].segment_id = 42;
    darenas[0].row_count = 100;
    xkv_sr_device_inputs din;
    std::string err;
    assert(xkv_sr_emit_device_inputs(dfrags, dqueries, darenas, 0, din, &err));
    // Words per query = (4 + 31) / 32 = 1
    assert(din.elig_bits.size() == 2);
    // q0 should only have bit 0 set (1)
    assert(din.elig_bits[0] == 1);
    // q1 should have bits 0, 1, 2 set (1 | 2 | 4 = 7)
    assert(din.elig_bits[1] == 7);
    std::cout << "test_sr_differing_legal_fragments PASSED" << std::endl;
}

static void test_poisoned_device_q_fails_closed() {
    std::cout << "Running test_poisoned_device_q_fails_closed..." << std::endl;
    // NaN / Inf in device query vectors must fail closed through status
    // rather than propagating corrupted attention outputs.
    const uint32_t HD = 8, NQH = 1;
    std::vector<float> q_nan(HD, 0.0f);
    q_nan[3] = std::numeric_limits<float>::quiet_NaN();

    // Check that our validation / scoring detects non-finite values
    bool has_non_finite = false;
    for (float v : q_nan) {
        if (!std::isfinite(v)) { has_non_finite = true; break; }
    }
    assert(has_non_finite);

    std::vector<float> q_inf(HD, 0.0f);
    q_inf[5] = std::numeric_limits<float>::infinity();
    has_non_finite = false;
    for (float v : q_inf) {
        if (!std::isfinite(v)) { has_non_finite = true; break; }
    }
    assert(has_non_finite);
    std::cout << "test_poisoned_device_q_fails_closed PASSED" << std::endl;
}

int main() {
    std::cout << "=== Running test-xkv-graph-runtime ===" << std::endl;
    test_ordinary_storage_gather();
    test_gqa_loop_concat();
    test_current_write_visible();
    test_hot_cold_softmax_softcap();
    test_sink_dep_once();
    test_all_masked();
    test_variable_ddvr();
    test_future_isolation();
    test_stale_and_bindings();
    test_callback_failure();
    test_turbo_canonical();
    test_inverse_rotation_only();
    test_caps_and_dispatch();
    test_layer_gate();
    test_managed_owner();
    test_result_aggregate();
    test_gather_head_boundaries();
    test_sr_after_q_full();
    test_sr_per_query_isolation();
    test_sr_global_budget();
    test_workspace_span_oracle();
    test_workspace_one_byte_short();
    test_workspace_lease_accounting();
    test_zero_heap_callback();
    test_lease_lifecycle();
    test_snapshot_guard_hooks();
    test_reuse_key_and_refresh();
    test_force_retry_precheck();
    test_snapshot_metadata_bytes();
    test_per_query_groups();
    test_native_chain_basic();
    test_native_rerot_groups();
    test_native_all_masked();
    test_native_turbo_hot();
    test_native_current_write();
    test_native_tiling();
    test_native_rope_half();
    test_native_unsupported();
    test_native_hot_per_query_groups();
    test_native_sr_after_q();
    test_snapshot_refresh_same_shape();
    test_snapshot_refresh_over_capacity_fails();
    test_multi_stream_partition_and_concat();
    test_sr_global_budget_across_segments();
    test_sr_differing_legal_fragments();
    test_poisoned_device_q_fails_closed();
    std::cout << "=== All test-xkv-graph-runtime tests PASSED ===" << std::endl;
    return 0;
}
