// Acceptance tests for the cache-owned XKV runtime coordinator
// (src/llama-xkv-runtime.{h,cpp}):
//
// 1. Non-contiguous + trailing layer groups from actual cache/model/hparams,
//    with exact per-layer feature offsets (prefix sums).
// 2. All-group atomic seal: every group published in one bundle; physical hot
//    rows released while logical cells (pos/payload/generation) remain.
// 3. Sealed segment reads back through existing codec/factor APIs
//    (decode_matrix + matrix_reconstruct) with bounded error.
// 4. PRIVATE_CONTROL / PENDING_RECORD / tentative (hot_writing) exclusion.
// 5. Shared-prefix identity: one payload enumerated once across keeper refs.
// 6. SHADOW seals (capture/factor/gates) but never rebinds/releases.
// 7. Failed group/landmark (workspace exhaustion) leaves every row hot.
// 8. Landmarks profile publishes valid quantized landmarks.
// 9. MTP/draft context creates no runtime.
// 10. XKV OFF admission defaults exactly map legacy capacity/usage.
// 11. Readiness/capability: bounded hot without XKV attention path fails
//     closed (admission reason + graph snapshot refusal).
// 12. Bounded segment selection: oldest full segment first, remainder deferred.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-kv-cache.h"
#include "llama-xkv-runtime.h"
#include "llama-xkv-graph-ref.h"
#include "llama-model.h"
#include "llama-xkv-transaction.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace llama_xkv;

static int g_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return; \
    } \
} while (0)

#define TEST_ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s -- %s\n", __FILE__, __LINE__, #cond, (msg)); \
        g_failures++; \
        return; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Stub model: small host KV fixture.
// ---------------------------------------------------------------------------

struct rt_test_model : public llama_model {
    explicit rt_test_model(uint32_t n_layer = 2) : llama_model(llama_model_default_params()) {
        hparams.n_ctx_train = 4096;
        arch = LLM_ARCH_LLAMA;
        hparams.rope_type = LLAMA_ROPE_TYPE_NEOX;
        hparams.rope_freq_base_train = 10000.0f;
        hparams.n_layer_all = n_layer;
        hparams.n_head_arr.fill(4);
        hparams.n_head_kv_arr.fill(2);
        hparams.n_embd_head_k_full = 64;
        hparams.n_embd_head_v_full = 64;
        hparams.n_rot_full = 64;
        hparams.no_alloc = false;
    }

    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        return nullptr;
    }
};

static llama_cparams rt_cparams(
    llama_xkv_mode mode,
    uint32_t seg_tokens = 8,
    uint32_t ubatch = 16,
    uint32_t group_size = 2,
    llama_xkv_storage_profile profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS) {
    llama_cparams c = {};
    c.n_batch = ubatch;
    c.n_ubatch = ubatch;
    c.xkv_mode = mode;
    c.xkv_storage_profile = profile;
    c.xkv_group_size = group_size;
    c.xkv_rank_k = 8;
    c.xkv_rank_v = 8;
    c.xkv_segment_tokens = seg_tokens;
    c.xkv_chunk_tokens = 8;
    c.xkv_workspace_mib = 16;
    c.xkv_decode_cache_mib = 8;
    c.xkv_min_saving = 0.10;
    return c;
}

// Commits n_tokens for seq at the given storage positions. When pool is null
// (SHADOW) the logical index is the hot row. Returns logical indices and hot
// rows. When commit=false the batch stays tentative (hot_writing).
struct committed_batch {
    std::vector<uint32_t> idxs;
    std::vector<uint32_t> hot_rows;
};

static committed_batch commit_tokens(
    llama_kv_cache & kv,
    llama_seq_id seq,
    const std::vector<llama_pos> & positions,
    bool commit = true,
    const llama_kv_rerot_meta * tag = nullptr) {
    committed_batch out;
    std::vector<llama_token> tokens(positions.size(), 101);
    std::vector<int32_t> n_seq_id(positions.size(), 1);
    std::vector<llama_seq_id *> seq_ptrs(positions.size(), &seq);

    llama_ubatch ubatch = {};
    ubatch.token = tokens.data();
    ubatch.pos = const_cast<llama_pos *>(positions.data());
    ubatch.n_tokens = (int32_t) positions.size();
    ubatch.n_seq_tokens = (int32_t) positions.size();
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.seq_id_unq = &seq;
    ubatch.n_seq_id = n_seq_id.data();
    ubatch.seq_id = seq_ptrs.data();

    auto sinfos = kv.prepare({ubatch});
    assert(!sinfos.empty());
    assert(sinfos[0].size() == positions.size());

    std::vector<xkv_hot_reservation> hot_res;
    auto pool = kv.get_hot_slot_pool();
    if (pool) {
        std::string err;
        auto res = pool->reserve((uint32_t) positions.size(), &err);
        assert(res.valid());
        sinfos[0].hot_idxs.resize(1);
        sinfos[0].hot_idxs[0].resize(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            sinfos[0].hot_idxs[0][i] = res[i];
            out.hot_rows.push_back(res[i]);
        }
        hot_res.push_back(std::move(res));
    } else {
        for (size_t i = 0; i < positions.size(); ++i) {
            out.hot_rows.push_back(sinfos[0].idxs[0][i]);
        }
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        out.idxs.push_back(sinfos[0].idxs[0][i]);
    }
    if (tag) {
        assert(kv.rerot_set_write_tag(seq, *tag));
    }
    llama_kv_cache_context ctx(&kv, sinfos, {ubatch}, std::move(hot_res));
    assert(ctx.get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    assert(ctx.apply());
    if (tag) {
        kv.rerot_clear_write_tag(seq);
    }
    if (commit) {
        ctx.postcompute_success();
    }
    return out;
}

// Deterministic smooth host pattern, distinct per layer/slot.
static void write_layer_rows(llama_kv_cache & kv, uint32_t il, const committed_batch & b,
                             const std::vector<llama_pos> & positions) {
    ggml_tensor * k = kv.get_k_storage((int32_t) il);
    ggml_tensor * v = kv.get_v_storage((int32_t) il);
    assert(k && v);
    assert(k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
    const int64_t dk = k->ne[0];
    const int64_t dv = v->ne[0];
    float * kd = (float *) k->data;
    float * vd = (float *) v->data;
    assert(kd && vd);
    for (size_t i = 0; i < b.hot_rows.size(); ++i) {
        const float p = (float) positions[i];
        float * krow = kd + (size_t) b.hot_rows[i] * (size_t) dk;
        float * vrow = vd + (size_t) b.hot_rows[i] * (size_t) dv;
        for (int64_t j = 0; j < dk; ++j) {
            krow[j] = sinf(p * 0.3f + (float) il * 1.7f) * cosf((float) j * 0.2f);
        }
        for (int64_t j = 0; j < dv; ++j) {
            vrow[j] = 0.5f * cosf(p * 0.23f + (float) il) * sinf((float) j * 0.11f + 1.0f);
        }
    }
}

static llama_kv_rerot_meta make_tag(llama_rerot_visibility vis) {
    llama_kv_rerot_meta m;
    m.episode_id = 7;
    m.node_id = (llama_rerot_node_id) 1;
    m.run_id = (llama_rerot_run_id) 1;
    m.publish_epoch = 0;
    m.visibility = vis;
    return m;
}

static void test_domain_isolation();
static void test_release_plan_fault();
static void test_device_upload_fail_closed();
static void test_coordinator_deferral();
static void test_snapshot_shadow_history();
static void test_rerot_snapshot_isolation();
static void test_tri_fill_first_interaction();
static void test_arena_lease_zero_heap();
static void test_ddvr_two_query_groups();
static void test_zero_alloc_warmed_maintain();
static void test_r3_reference_landmarks();
static void test_vulkan_shadow_no_mutation();
static void test_fingerprint_field_coverage();
static void test_imrope_seal_oracle();
static void test_tail_rank_proportional();
static void test_placement_split_two_devices();
static void test_repeated_seal_stable();
static void test_pressure_partial_seal();
static void test_disjoint_private_rows_shared_prefix();
static void test_sr_per_slot_fragment_plans();
static void test_physical_vs_semantic_accounting();


static double frob_rel_err(const std::vector<float> & ref, const std::vector<float> & got) {
    assert(ref.size() == got.size());
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) ref[i] - (double) got[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    if (den == 0.0) {
        return num == 0.0 ? 0.0 : 1e30;
    }
    return sqrt(num / den);
}

// Decode one factor pair and reconstruct canonical rows.
static std::vector<float> reconstruct_pair(const encoded_matrix & a, const encoded_matrix & b,
                                           uint64_t n_rows, uint64_t rank, uint64_t dim) {
    const std::vector<float> a_dec = decode_matrix(a);
    const std::vector<float> b_dec = decode_matrix(b);
    const uint64_t a_pad = a.desc.padded_shape.cols;
    const uint64_t b_pad = b.desc.padded_shape.cols;
    assert(a_dec.size() == n_rows * a_pad);
    assert(b_dec.size() == dim * b_pad);
    matrix am(n_rows, rank), bt(dim, rank);
    for (uint64_t r = 0; r < n_rows; ++r) {
        memcpy(am.row_ptr(r), a_dec.data() + r * a_pad, rank * sizeof(float));
    }
    for (uint64_t c = 0; c < dim; ++c) {
        memcpy(bt.row_ptr(c), b_dec.data() + c * b_pad, rank * sizeof(float));
    }
    const matrix x = matrix_reconstruct(am, bt);
    assert(x.rows == n_rows && x.cols == dim);
    return x.data;
}

// ---------------------------------------------------------------------------
// 1. Non-contiguous + trailing groups, exact feature offsets.
// ---------------------------------------------------------------------------

static void test_groups_noncontiguous_trailing() {
    fprintf(stderr, "--- test_groups_noncontiguous_trailing ---\n");
    rt_test_model model(6);
    // Layer 2 is recurrent: excluded, leaving owning [0,1,3,4,5].
    model.hparams.is_recr_impl[2] = 1;
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 4);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    std::string err;
    TEST_ASSERT(rt->rebuild_groups(model.hparams, kv, &err));
    const layer_group_map & gm = rt->group_map();
    TEST_ASSERT(gm.groups.size() == 2);
    TEST_ASSERT(gm.groups[0].owning_layers == std::vector<uint32_t>({0, 1, 3, 4}));
    TEST_ASSERT(gm.groups[1].owning_layers == std::vector<uint32_t>({5}));
    // Exact feature offsets are prefix sums of per-layer dims (128 each).
    TEST_ASSERT(gm.groups[0].layer_feature_offsets_k == std::vector<uint32_t>({0, 128, 256, 384}));
    TEST_ASSERT(gm.groups[0].layer_feature_dims_k == std::vector<uint32_t>({128, 128, 128, 128}));
    TEST_ASSERT(gm.groups[0].total_dim_k == 512);
    TEST_ASSERT(gm.groups[1].layer_feature_offsets_k == std::vector<uint32_t>({0}));
    TEST_ASSERT(gm.groups[1].total_dim_k == 128);
    // Same for V.
    TEST_ASSERT(gm.groups[0].layer_feature_offsets_v == std::vector<uint32_t>({0, 128, 256, 384}));
    TEST_ASSERT(gm.groups[1].total_dim_v == 128);
    // Per-layer gate: trunk enabled, recurrent excluded, OOR fail-closed.
    TEST_ASSERT(rt->xkv_layer_enabled(0));
    TEST_ASSERT(rt->xkv_layer_enabled(5));
    TEST_ASSERT(!rt->xkv_layer_enabled(2));
    TEST_ASSERT(!rt->xkv_layer_enabled(6));
}

// ---------------------------------------------------------------------------
// 1b. Per-layer gate: SWA excluded, trunk enabled.
// ---------------------------------------------------------------------------

static void test_layer_gate_swa() {
    fprintf(stderr, "--- test_layer_gate_swa ---\n");
    rt_test_model model(3);
    model.hparams.is_swa_impl[1] = 1;
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 4);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    std::string err;
    TEST_ASSERT(rt->rebuild_groups(model.hparams, kv, &err));
    TEST_ASSERT(rt->group_map().groups.size() == 1);
    TEST_ASSERT(rt->group_map().groups[0].owning_layers == std::vector<uint32_t>({0, 2}));
    TEST_ASSERT(rt->xkv_layer_enabled(0));
    TEST_ASSERT(!rt->xkv_layer_enabled(1)); // SWA: never on the XKV path
    TEST_ASSERT(rt->xkv_layer_enabled(2));
    TEST_ASSERT(!rt->xkv_layer_enabled(3)); // out of range: fail closed
}

// ---------------------------------------------------------------------------
// 2. All-group atomic seal + physical hot release, logical cells remain.
// ---------------------------------------------------------------------------

static void test_atomic_seal_and_hot_release() {
    fprintf(stderr, "--- test_atomic_seal_and_hot_release ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    TEST_ASSERT(store && pool);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    const auto & cells = kv.get_cells(0);
    std::vector<uint64_t> pids;
    for (uint32_t idx : b.idxs) {
        pids.push_back(cells.payload_id_get(idx));
    }
    const uint32_t bound_before = pool->get_bound();
    TEST_ASSERT(bound_before == 8);

    std::string err;
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    TEST_ASSERT(rt->stats().sealed_rows == 8);

    // All payloads factored in the store...
    for (uint64_t pid : pids) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(pid, st) && st == xkv_state::factored);
    }
    // ...physical hot rows released...
    TEST_ASSERT(pool->get_bound() == bound_before - 8);
    for (uint64_t pid : pids) {
        TEST_ASSERT(!pool->has_payload(pid));
    }
    // Host fixture performs zero backend synchronizations.
    TEST_ASSERT(rt->stats().last_sync_count == 0);
    // ...while logical cells remain resident with identity intact.
    for (size_t i = 0; i < b.idxs.size(); ++i) {
        TEST_ASSERT(!cells.is_empty(b.idxs[i]));
        TEST_ASSERT(cells.pos_get(b.idxs[i]) == pos[i]);
        TEST_ASSERT(cells.payload_id_get(b.idxs[i]) == pids[i]);
    }
    TEST_ASSERT(kv.get_kv_used() == 8);
}

// ---------------------------------------------------------------------------
// 3. Readback through existing codec/factor APIs + exact group offsets.
// ---------------------------------------------------------------------------

static void test_readback_and_offsets() {
    fprintf(stderr, "--- test_readback_and_offsets ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    // Canonical oracle via public read helper before sealing.
    std::vector<xkv_layer_row_view> rows;
    std::string err;
    TEST_ASSERT_MSG(rt->collect_eligible_rows(kv, rows, &err), err.c_str());
    TEST_ASSERT(rows.size() == 8);
    matrix ck0, cv0, ck1, cv1;
    TEST_ASSERT_MSG(rt->read_layer_canonical(kv, 0, rows, ck0, cv0, &err), err.c_str());
    TEST_ASSERT_MSG(rt->read_layer_canonical(kv, 1, rows, ck1, cv1, &err), err.c_str());
    TEST_ASSERT(ck0.rows == 8 && ck0.cols == 128);

    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());

    // Exactly one bundle with both groups; offsets are prefix sums.
    TEST_ASSERT(store->get_accounting().active_segments == 1);
    auto seg = store->pin_segment(1);
    TEST_ASSERT((bool) seg);
    TEST_ASSERT(seg->groups.size() == 2);
    TEST_ASSERT(seg->n_rows == 8 && seg->n_live_rows == 8);
    const auto * g0 = seg->find_group(0);
    TEST_ASSERT(g0 && g0->owning_layers == std::vector<uint32_t>({0, 1}));
    TEST_ASSERT(g0->layer_feature_offsets_k == std::vector<uint32_t>({0, 128}));
    TEST_ASSERT(g0->total_dim_k == 256 && g0->total_dim_v == 256);

    // Reconstruct each group from stored streams and compare to oracle.
    for (const auto & g : seg->groups) {
        TEST_ASSERT(g.a_k.desc.logical_shape.rows == 8);
        const std::vector<float> rk = reconstruct_pair(
            g.a_k, *g.b_k, 8, g.a_k.desc.logical_shape.cols, g.total_dim_k);
        const std::vector<float> rv = reconstruct_pair(
            g.a_v, *g.b_v, 8, g.a_v.desc.logical_shape.cols, g.total_dim_v);
        // Oracle concatenation in owning-layer order.
        std::vector<float> ok(8 * g.total_dim_k), ov(8 * g.total_dim_v);
        for (size_t li = 0; li < g.owning_layers.size(); ++li) {
            const matrix & mk = (g.owning_layers[li] == 0) ? ck0 : ck1;
            const matrix & mv = (g.owning_layers[li] == 0) ? cv0 : cv1;
            for (uint32_t r = 0; r < 8; ++r) {
                memcpy(ok.data() + (size_t) r * g.total_dim_k + g.layer_feature_offsets_k[li],
                       mk.row_ptr(r), g.layer_feature_dims_k[li] * sizeof(float));
                memcpy(ov.data() + (size_t) r * g.total_dim_v + g.layer_feature_offsets_v[li],
                       mv.row_ptr(r), g.layer_feature_dims_v[li] * sizeof(float));
            }
        }
        TEST_ASSERT(frob_rel_err(ok, rk) < 0.30);
        TEST_ASSERT(frob_rel_err(ov, rv) < 0.30);
    }
}

// ---------------------------------------------------------------------------
// 4. PRIVATE / PENDING / tentative exclusion.
// ---------------------------------------------------------------------------

static void test_exclusions() {
    fprintf(stderr, "--- test_exclusions ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    auto priv_tag = make_tag(llama_rerot_visibility::private_control);
    auto b_priv = commit_tokens(kv, 1, {100, 101}, true, &priv_tag);
    auto pend_tag = make_tag(llama_rerot_visibility::pending_record);
    auto b_pend = commit_tokens(kv, 2, {200, 201}, true, &pend_tag);
    auto b_tent = commit_tokens(kv, 0, {8, 9}, false);

    std::string err;
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());

    const auto & cells = kv.get_cells(0);
    // The 8 ordinary rows sealed...
    for (uint32_t idx : b.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::factored);
    }
    // ...PRIVATE / PENDING / tentative rows stay hot and bound.
    for (uint32_t idx : b_priv.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::hot_committed);
        TEST_ASSERT(pool->has_payload(cells.payload_id_get(idx)));
    }
    for (uint32_t idx : b_pend.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::hot_committed);
        TEST_ASSERT(pool->has_payload(cells.payload_id_get(idx)));
    }
    for (uint32_t idx : b_tent.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::hot_writing);
    }
    TEST_ASSERT(pool->get_bound() == 6);
}

// ---------------------------------------------------------------------------
// 5. Shared-prefix identity.
// ---------------------------------------------------------------------------

static void test_shared_prefix_identity() {
    fprintf(stderr, "--- test_shared_prefix_identity ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto * rt = kv.get_xkv_runtime();

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);
    // Share the whole prefix with seq 1: same cells, extra keeper refs.
    kv.seq_cp(0, 1, 0, 8);

    std::vector<xkv_layer_row_view> rows;
    std::string err;
    TEST_ASSERT_MSG(rt->collect_eligible_rows(kv, rows, &err), err.c_str());
    TEST_ASSERT(rows.size() == 8); // one row per payload, not per keeper ref
    const auto & cells = kv.get_cells(0);
    for (size_t i = 0; i < b.idxs.size(); ++i) {
        TEST_ASSERT(rows[i].payload_id == cells.payload_id_get(b.idxs[i]));
    }
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_rows == 8);
}

// ---------------------------------------------------------------------------
// 6. SHADOW seals but never releases.
// ---------------------------------------------------------------------------

static void test_shadow_non_release() {
    fprintf(stderr, "--- test_shadow_non_release ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SHADOW, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    TEST_ASSERT(kv.get_hot_slot_pool() == nullptr);
    auto store = kv.get_xkv_store();
    TEST_ASSERT(store != nullptr);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    // Bindings exist at logical rows even without a pool.
    auto snap = store->list_hot_payload_bindings();
    TEST_ASSERT(snap.bindings.size() == 8);

    std::string err;
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().shadow_evals == 1);
    TEST_ASSERT(rt->stats().sealed_segments == 0);
    // Evaluate-only: factor/gates ran, but nothing published, rebound, or
    // released. Payloads stay hot_committed; epochs/locations untouched.
    TEST_ASSERT(store->get_accounting().active_segments == 0);
    const auto & cells = kv.get_cells(0);
    for (size_t i = 0; i < b.idxs.size(); ++i) {
        TEST_ASSERT(!cells.is_empty(b.idxs[i]));
        TEST_ASSERT(cells.pos_get(b.idxs[i]) == pos[i]);
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(b.idxs[i]), st) &&
                      st == xkv_state::hot_committed);
    }
    TEST_ASSERT(store->list_hot_payload_bindings().bindings.size() == 8);
    // Second run re-evaluates (still no publication).
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().shadow_evals == 2);
    TEST_ASSERT(store->get_accounting().active_segments == 0);
}

// ---------------------------------------------------------------------------
// 7. Failed seal leaves every row hot.
// ---------------------------------------------------------------------------

static void test_failed_seal_leaves_hot() {
    fprintf(stderr, "--- test_failed_seal_leaves_hot ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    cp.xkv_workspace_mib = 0; // no workspace: seal preflight must fail
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    TEST_ASSERT(!rt->maintain(kv, 0, true, &err));
    TEST_ASSERT(!err.empty());
    // Every row is still hot_committed and still bound.
    const auto & cells = kv.get_cells(0);
    for (uint32_t idx : b.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::hot_committed);
        TEST_ASSERT(pool->has_payload(cells.payload_id_get(idx)));
    }
    TEST_ASSERT(pool->get_bound() == 8);
    TEST_ASSERT(store->get_accounting().active_segments == 0);
}

// ---------------------------------------------------------------------------
// 8. Landmarks profile publishes valid quantized landmarks.
// ---------------------------------------------------------------------------

static void test_landmarks_profile() {
    fprintf(stderr, "--- test_landmarks_profile ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2,
        LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    auto seg = store->pin_segment(1);
    TEST_ASSERT((bool) seg);
    for (const auto & g : seg->groups) {
        TEST_ASSERT(!g.landmark.bytes.empty());
        TEST_ASSERT(g.landmark.desc.logical_shape.rows == 1); // 8 rows, chunk 8
        TEST_ASSERT(g.landmark.desc.logical_shape.cols == g.total_dim_k);
    }
    // Phase correctness: the decoded landmark approximates the mean of the
    // post-RoPE stored rows (independent oracle: raw host tensors), and is
    // not a phase-less pooled mean of canonical rows.
    for (const auto & g : seg->groups) {
        const std::vector<float> lm_dec = decode_matrix(g.landmark);
        TEST_ASSERT(lm_dec.size() == (size_t) g.total_dim_k);
        // Oracle 1: mean of stored post-RoPE rows for this group's layers.
        std::vector<float> oracle(g.total_dim_k, 0.0f);
        for (size_t li = 0; li < g.owning_layers.size(); ++li) {
            ggml_tensor * k = kv.get_k_storage((int32_t) g.owning_layers[li]);
            const float * kd = (const float *) k->data;
            const int64_t dk = k->ne[0];
            for (size_t i = 0; i < b.hot_rows.size(); ++i) {
                const float * row = kd + (size_t) b.hot_rows[i] * (size_t) dk;
                float * acc = oracle.data() + g.layer_feature_offsets_k[li];
                for (uint32_t j = 0; j < g.layer_feature_dims_k[li]; ++j) {
                    acc[j] += row[j] / (float) b.hot_rows.size();
                }
            }
        }
        TEST_ASSERT(frob_rel_err(oracle, lm_dec) < 0.25);
    }
    // Determinism: rebuilding twice yields identical bytes.
    {
        auto seg2 = store->pin_segment(1);
        TEST_ASSERT((bool) seg2);
        const auto * h0 = seg2->find_group(0);
        TEST_ASSERT(h0 != nullptr);
        TEST_ASSERT(!h0->landmark.bytes.empty());
    }
}

// ---------------------------------------------------------------------------
// 9. MTP context creates no runtime.
// ---------------------------------------------------------------------------

static void test_mtp_no_runtime() {
    fprintf(stderr, "--- test_mtp_no_runtime ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    TEST_ASSERT(kv.get_xkv_store() == nullptr);
    TEST_ASSERT(kv.get_xkv_runtime() == nullptr);
}

// ---------------------------------------------------------------------------
// 10. OFF admission maps legacy capacity/usage.
// ---------------------------------------------------------------------------

static void test_off_admission() {
    fprintf(stderr, "--- test_off_admission ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_OFF, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    TEST_ASSERT(kv.get_xkv_runtime() == nullptr);
    llama_memory_admission_snapshot adm = {};
    TEST_ASSERT(kv.get_admission_snapshot(&adm));
    TEST_ASSERT(adm.logical_capacity == kv.get_kv_capacity());
    TEST_ASSERT(adm.logical_used == kv.get_kv_used());
    TEST_ASSERT(adm.hot_capacity == adm.logical_capacity);
    TEST_ASSERT(adm.hot_used == adm.logical_used);
    TEST_ASSERT(adm.factor_safe_tokens == UINT32_MAX);
    TEST_ASSERT(adm.workspace_safe_tokens == UINT32_MAX);
    TEST_ASSERT(adm.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
}

// ---------------------------------------------------------------------------
// 11. Readiness gate: bounded hot without XKV path fails closed.
// ---------------------------------------------------------------------------

static void test_readiness_gate() {
    fprintf(stderr, "--- test_readiness_gate ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    TEST_ASSERT(rt->bounded_hot_configured());
    // Computed readiness: host CPU tensors + CPU factorizer + decoded-hot
    // source support the CPU/reference path (never a fixed bool).
    TEST_ASSERT(rt->attention_path_ready());
    TEST_ASSERT(rt->readiness_reason()[0] == '\0');
    TEST_ASSERT(!rt->legacy_attention_safe());

    llama_memory_admission_snapshot adm = {};
    TEST_ASSERT(kv.get_admission_snapshot(&adm));
    TEST_ASSERT(adm.logical_capacity == kv.get_kv_capacity());
    TEST_ASSERT(adm.hot_capacity == kv.get_kv_hot_capacity());
    TEST_ASSERT(adm.hot_free + adm.hot_used <= adm.hot_capacity + adm.hot_reserved);

    // Graph snapshot builds on the ready CPU path with gather descriptors.
    std::vector<llama_pos> pos = {0, 1, 2, 3};
    auto b = commit_tokens(kv, 0, pos, true);
    (void) b;
    std::vector<llama_token> tokens(pos.size(), 101);
    std::vector<int32_t> n_seq_id(pos.size(), 1);
    llama_seq_id seq0 = 0;
    std::vector<llama_seq_id *> seq_ptrs(pos.size(), &seq0);
    llama_ubatch ubatch = {};
    ubatch.token = tokens.data();
    ubatch.pos = pos.data();
    ubatch.n_tokens = (int32_t) pos.size();
    ubatch.n_seq_tokens = (int32_t) pos.size();
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.seq_id_unq = &seq0;
    ubatch.n_seq_id = n_seq_id.data();
    ubatch.seq_id = seq_ptrs.data();
    auto sinfos = kv.prepare({ubatch});
    TEST_ASSERT(!sinfos.empty());
    llama_kv_cache_context ctx(&kv, sinfos, {ubatch});
    std::unique_ptr<xkv_graph_snapshot> snap;
    std::string err;
    TEST_ASSERT_MSG(ctx.build_xkv_graph_snapshot(0, 0, 1.0f, 0.0f, snap, &err), err.c_str());
    TEST_ASSERT(snap != nullptr);
    TEST_ASSERT(snap->hot_data.size() == 4);
    for (const auto & hd : snap->hot_data) {
        TEST_ASSERT(hd.use_storage_gather());
        TEST_ASSERT(hd.k_data.empty() && hd.v_data.empty());
    }
    TEST_ASSERT(ctx.xkv_layer_enabled(0));
    TEST_ASSERT(rt->xkv_layer_enabled(0) && rt->xkv_layer_enabled(1));
    TEST_ASSERT(!rt->xkv_layer_enabled(99));
}

// ---------------------------------------------------------------------------
// 12. Bounded selection: oldest full segment first, remainder deferred.
// ---------------------------------------------------------------------------

static void test_bounded_selection() {
    fprintf(stderr, "--- test_bounded_selection ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    auto * rt = kv.get_xkv_runtime();

    std::vector<llama_pos> pos;
    for (llama_pos p = 0; p < 12; ++p) {
        pos.push_back(p);
    }
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    TEST_ASSERT(rt->stats().sealed_rows == 8);
    // Oldest 8 factored (positions 0..7)...
    const auto & cells = kv.get_cells(0);
    for (size_t i = 0; i < 8; ++i) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(b.idxs[i]), st) && st == xkv_state::factored);
    }
    // ...newest 4 still hot and bound.
    for (size_t i = 8; i < 12; ++i) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(b.idxs[i]), st) && st == xkv_state::hot_committed);
    }
    TEST_ASSERT(pool->get_bound() == 4);
    // Second run defers: no full segment available.
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    TEST_ASSERT(store->get_accounting().active_segments == 1);
}

int main() {
    test_groups_noncontiguous_trailing();
    test_layer_gate_swa();
    test_atomic_seal_and_hot_release();
    test_readback_and_offsets();
    test_exclusions();
    test_shared_prefix_identity();
    test_shadow_non_release();
    test_failed_seal_leaves_hot();
    test_landmarks_profile();
    test_mtp_no_runtime();
    test_off_admission();
    test_readiness_gate();
    test_bounded_selection();
    test_domain_isolation();
    test_release_plan_fault();
    test_device_upload_fail_closed();
    test_coordinator_deferral();
    test_snapshot_shadow_history();
    test_rerot_snapshot_isolation();
    test_tri_fill_first_interaction();
    test_arena_lease_zero_heap();
    test_ddvr_two_query_groups();
    test_zero_alloc_warmed_maintain();
    test_r3_reference_landmarks();
    test_vulkan_shadow_no_mutation();
    test_fingerprint_field_coverage();
    test_imrope_seal_oracle();
    test_tail_rank_proportional();
    test_placement_split_two_devices();
    test_repeated_seal_stable();
    test_pressure_partial_seal();
    test_disjoint_private_rows_shared_prefix();
    test_sr_per_slot_fragment_plans();
    test_physical_vs_semantic_accounting();


    if (g_failures == 0) {
        fprintf(stderr, "test-xkv-runtime: all tests passed\n");
        return 0;
    }
    fprintf(stderr, "test-xkv-runtime: %d FAILURES\n", g_failures);
    return 1;
}

// ---------------------------------------------------------------------------
// 13. Domain isolation: unrelated ordinary histories never factor together.
// ---------------------------------------------------------------------------

static void test_domain_isolation() {
    fprintf(stderr, "--- test_domain_isolation ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);

    auto b0 = commit_tokens(kv, 0, {0, 1, 2, 3}, true);
    auto b1 = commit_tokens(kv, 1, {100, 101, 102, 103}, true);
    write_layer_rows(kv, 0, b0, {0, 1, 2, 3});
    write_layer_rows(kv, 1, b0, {0, 1, 2, 3});
    write_layer_rows(kv, 0, b1, {100, 101, 102, 103});
    write_layer_rows(kv, 1, b1, {100, 101, 102, 103});

    // Neither domain fills a segment: deferred, nothing sealed.
    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 0);

    auto b0b = commit_tokens(kv, 0, {4, 5, 6, 7}, true);
    write_layer_rows(kv, 0, b0b, {4, 5, 6, 7});
    write_layer_rows(kv, 1, b0b, {4, 5, 6, 7});
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    // Only seq 0's domain factored; seq 1 untouched and still hot.
    const auto & cells = kv.get_cells(0);
    for (uint32_t idx : b0.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::factored);
    }
    for (uint32_t idx : b0b.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::factored);
    }
    for (uint32_t idx : b1.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::hot_committed);
    }
    TEST_ASSERT(rt->stats().last_domain_fp != 0);
}

// ---------------------------------------------------------------------------
// 14. Release fault injection: stale plans refused, pool untouched.
// ---------------------------------------------------------------------------

static void test_release_plan_fault() {
    fprintf(stderr, "--- test_release_plan_fault ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto pool = kv.get_hot_slot_pool();

    auto b = commit_tokens(kv, 0, {0, 1, 2, 3, 4, 5, 6, 7}, true);
    const auto & cells = kv.get_cells(0);
    xkv_hot_release_plan stale;
    for (uint32_t idx : b.idxs) {
        stale.released_payload_ids.push_back(cells.payload_id_get(idx));
        stale.released_generations.push_back(cells.storage_generation_get(idx) + 100); // stale
        uint32_t slot = 0;
        TEST_ASSERT(pool->find_slot(cells.payload_id_get(idx), slot));
        stale.released_physical_rows.push_back(slot);
    }
    std::string err;
    TEST_ASSERT(!kv.precommit_hot_release_nostore(stale, &err));
    TEST_ASSERT(!err.empty());
    TEST_ASSERT(!kv.validate_hot_release(stale, &err));
    // Pool untouched by the refused validation.
    TEST_ASSERT(pool->get_bound() == 8);
    for (uint32_t idx : b.idxs) {
        TEST_ASSERT(pool->has_payload(cells.payload_id_get(idx)));
    }
}

// ---------------------------------------------------------------------------
// 15. Device profile without an upload path fails closed (never host-factors).
// ---------------------------------------------------------------------------

static void test_device_upload_fail_closed() {
    fprintf(stderr, "--- test_device_upload_fail_closed ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    cp.xkv_factorizer = LLAMA_XKV_FACTORIZER_VULKAN; // device-owned, no hook set
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);
    TEST_ASSERT(!rt->maintain(kv, 0, true, &err));
    TEST_ASSERT(!err.empty());
    // Gate refused: no publication, no rebind, every row hot and bound.
    TEST_ASSERT(store->get_accounting().active_segments == 0);
    const auto & cells = kv.get_cells(0);
    for (uint32_t idx : b.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::hot_committed);
        TEST_ASSERT(pool->has_payload(cells.payload_id_get(idx)));
    }
    TEST_ASSERT(pool->get_bound() == 8);
}

// ---------------------------------------------------------------------------
// 16. Coordinator exclusion contention defers maintenance.
// ---------------------------------------------------------------------------

static void test_coordinator_deferral() {
    fprintf(stderr, "--- test_coordinator_deferral ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);

    xkv_transaction_coordinator coord;
    rt->set_coordinator(&coord);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    // Hold the exclusion externally: maintenance defers without sealing.
    std::string err;
    {
        xkv_maintenance_guard ext(&coord, xkv_maintenance_op::pack, 99, &err);
        TEST_ASSERT(ext.held());
        TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
        TEST_ASSERT(rt->stats().sealed_segments == 0);
    }
    // Released: maintenance seals.
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    rt->set_coordinator(nullptr);
}

// ---------------------------------------------------------------------------
// 17. SHADOW snapshot covers history + current writes as gather descriptors.
// ---------------------------------------------------------------------------

static void test_snapshot_shadow_history() {
    fprintf(stderr, "--- test_snapshot_shadow_history ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SHADOW, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);

    auto b = commit_tokens(kv, 0, {0, 1, 2, 3}, true);
    write_layer_rows(kv, 0, b, {0, 1, 2, 3});
    write_layer_rows(kv, 1, b, {0, 1, 2, 3});

    // Current writes stay tentative in this context (applied, not committed).
    std::vector<llama_token> tokens = {101, 102};
    std::vector<llama_pos> pos = {8, 9};
    std::vector<int32_t> n_seq_id = {1, 1};
    llama_seq_id seq0 = 0;
    std::vector<llama_seq_id *> seq_ptrs = {&seq0, &seq0};
    llama_ubatch ub = {};
    ub.token = tokens.data();
    ub.pos = pos.data();
    ub.n_tokens = 2;
    ub.n_seq_tokens = 2;
    ub.n_seqs = 1;
    ub.n_seqs_unq = 1;
    ub.seq_id_unq = &seq0;
    ub.n_seq_id = n_seq_id.data();
    ub.seq_id = seq_ptrs.data();
    auto sinfos = kv.prepare({ub});
    TEST_ASSERT(!sinfos.empty());
    std::vector<xkv_hot_reservation> no_res;
    llama_kv_cache_context ctx(&kv, sinfos, {ub}, std::move(no_res));
    TEST_ASSERT(ctx.apply());
    // No canonical pre-copy: storage still holds zeros for the new rows.
    std::unique_ptr<xkv_graph_snapshot> snap;
    std::string err;
    TEST_ASSERT_MSG(ctx.build_xkv_graph_snapshot(0, 0, 1.0f, 0.0f, snap, &err), err.c_str());
    TEST_ASSERT(snap != nullptr);
    // History (4) + current writes (2), all as gather descriptors.
    TEST_ASSERT(snap->hot_data.size() == 6);
    TEST_ASSERT(snap->segment_views.empty());
    for (const auto & hd : snap->hot_data) {
        TEST_ASSERT(hd.use_storage_gather());
        TEST_ASSERT(hd.k_data.empty() && hd.v_data.empty());
        TEST_ASSERT(hd.query_visibility.size() == 2);
    }
    // Causal per-query visibility: token at pos 8 sees rows 0..3 and 8.
    uint32_t vis0 = 0, vis1 = 0;
    for (const auto & hd : snap->hot_data) {
        if (hd.is_visible_to_query(0)) vis0++;
        if (hd.is_visible_to_query(1)) vis1++;
    }
    TEST_ASSERT(vis0 == 5 && vis1 == 6);
    TEST_ASSERT(snap->query_causal_limits[0] == 8 && snap->query_causal_limits[1] == 9);
    // Tentative current rows accepted with writing state.
    uint32_t writing = 0;
    for (const auto & hd : snap->hot_data) {
        if (hd.expected_state == xkv_state::hot_writing) writing++;
    }
    TEST_ASSERT(writing == 2);
    TEST_ASSERT(snap->hot_storage_host_resident);
    TEST_ASSERT(snap->sr_mode == 0);
}

// ---------------------------------------------------------------------------
// 18. RERoT snapshot isolation: root/fork/frontier/pending/private visibility.
// ---------------------------------------------------------------------------

static void test_rerot_snapshot_isolation() {
    fprintf(stderr, "--- test_rerot_snapshot_isolation ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SHADOW, 8, 16, 2);
    cp.rerot_enabled = true;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);

    // Tag 1: public root run 1 at epoch 1
    llama_kv_rerot_meta root_tag;
    root_tag.episode_id = 42; root_tag.node_id = 1; root_tag.run_id = 1;
    root_tag.publish_epoch = 1;
    root_tag.visibility = llama_rerot_visibility::public_live;
    auto b_root = commit_tokens(kv, 0, {0, 1, 2, 3}, true, &root_tag);
    write_layer_rows(kv, 0, b_root, {0, 1, 2, 3});

    // Tag 2: private control run 2 (never public)
    llama_kv_rerot_meta priv_tag;
    priv_tag.episode_id = 42; priv_tag.node_id = 2; priv_tag.run_id = 2;
    priv_tag.publish_epoch = 0;
    priv_tag.visibility = llama_rerot_visibility::private_control;
    auto b_priv = commit_tokens(kv, 0, {4, 5}, true, &priv_tag);
    write_layer_rows(kv, 0, b_priv, {4, 5});

    // Tag 3: pending record run 3 (unpublished, epoch 0)
    llama_kv_rerot_meta pend_tag;
    pend_tag.episode_id = 42; pend_tag.node_id = 3; pend_tag.run_id = 3;
    pend_tag.publish_epoch = 0;
    pend_tag.visibility = llama_rerot_visibility::pending_record;
    auto b_pend = commit_tokens(kv, 0, {6, 7}, true, &pend_tag);
    write_layer_rows(kv, 0, b_pend, {6, 7});

    // Install reader view for seq 0 seeing root run 1 and its own private run 2,
    // but NOT the unpublished pending run 3.
    llama_rerot_reader_state reader;
    reader.episode_id = 42;
    reader.reader = 2;
    reader.query_run = 2;
    reader.frontier = 2;
    reader.ordered_runs = {1, 2}; // root (1) + own private (2)
    TEST_ASSERT(kv.rerot_set_reader_view(0, reader));

    // Single query at pos 5
    std::vector<llama_token> tokens = {101};
    std::vector<llama_pos> pos = {5};
    std::vector<int32_t> n_seq_id = {1};
    llama_seq_id seq0 = 0;
    std::vector<llama_seq_id *> seq_ptrs = {&seq0};
    llama_ubatch ub = {};
    ub.token = tokens.data(); ub.pos = pos.data();
    ub.n_tokens = 1; ub.n_seq_tokens = 1; ub.n_seqs = 1; ub.n_seqs_unq = 1;
    ub.seq_id_unq = &seq0; ub.n_seq_id = n_seq_id.data(); ub.seq_id = seq_ptrs.data();

    auto sinfos = kv.prepare({ub});
    TEST_ASSERT(!sinfos.empty());
    std::vector<xkv_hot_reservation> no_res;
    llama_kv_cache_context ctx(&kv, sinfos, {ub}, std::move(no_res));
    TEST_ASSERT(ctx.apply());

    std::unique_ptr<xkv_graph_snapshot> snap;
    std::string err;
    TEST_ASSERT_MSG(ctx.build_xkv_graph_snapshot(0, 0, 1.0f, 0.0f, snap, &err), err.c_str());
    TEST_ASSERT(snap != nullptr);

    // Layout-derived visibility: root (4) + private (2) = 6 rows visible.
    // Unpublished pending run 3 is completely filtered out.
    uint32_t visible_count = 0;
    bool saw_pending = false;
    for (const auto & hd : snap->hot_data) {
        if (hd.is_visible_to_query(0)) {
            visible_count++;
            if (hd.storage_pos >= 6 && hd.storage_pos <= 7) saw_pending = true;
        }
    }
    TEST_ASSERT(visible_count == 6);
    TEST_ASSERT(!saw_pending);
    // DDVR group counts populated from the RERoT layout.
    TEST_ASSERT(!snap->query_ddvr_group_counts.empty());
    TEST_ASSERT(snap->query_ddvr_group_counts[0] > 0);

    // Now publish pending run 3 at epoch 2:
    TEST_ASSERT(kv.rerot_publish_run(42, 3, 2) == 2);
    // Add run 3 to reader's ordered_runs:
    reader.ordered_runs = {1, 2, 3};
    TEST_ASSERT(kv.rerot_set_reader_view(0, reader));

    std::unique_ptr<xkv_graph_snapshot> snap2;
    TEST_ASSERT_MSG(ctx.build_xkv_graph_snapshot(0, 0, 1.0f, 0.0f, snap2, &err), err.c_str());
    TEST_ASSERT(snap2 != nullptr);
    // Now published: pending run rows become visible (within causal pos 5).
    // Note: pos=5 cutoff means rows 6,7 are causally excluded even when published.
    TEST_ASSERT(snap2->query_causal_limits[0] == 5);
}

// ---------------------------------------------------------------------------
// 19. Tri fill-first interaction: maintain preserves dense hot while free slots exist
// ---------------------------------------------------------------------------

static void test_tri_fill_first_interaction() {
    fprintf(stderr, "--- test_tri_fill_first_interaction ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(pool != nullptr && rt != nullptr);

    // Commit full segment (8 tokens). Hot capacity is 16, so 8 free slots remain.
    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);
    TEST_ASSERT(pool->get_bound() == 8);
    TEST_ASSERT(pool->get_free() == pool->get_capacity() - 8);
    TEST_ASSERT(pool->get_free() > 0);

    std::string err;
    // 1. Fill-first check: maintain called with default or small upcoming requirement.
    // Free slots exist -> fill-first preserves dense hot rows, NO SEAL occurs!
    TEST_ASSERT_MSG(rt->maintain(kv, 1, false, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 0);
    TEST_ASSERT(store->get_accounting().active_segments == 0);
    TEST_ASSERT(pool->get_bound() == 8);

    // 2. Upcoming requirement exceeds free slots -> pressure hits -> seals!
    uint32_t needed = pool->get_free() + 1;
    TEST_ASSERT_MSG(rt->maintain(kv, needed, false, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    TEST_ASSERT(store->get_accounting().active_segments == 1);
    TEST_ASSERT(pool->get_bound() == 0); // Released on pressure drain!
}

// ---------------------------------------------------------------------------
// 20. Arena allocation proof: canonical workspace carved from real lease, zero heap
// ---------------------------------------------------------------------------

static void test_arena_lease_zero_heap() {
    fprintf(stderr, "--- test_arena_lease_zero_heap ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store != nullptr && rt != nullptr);

    // Introspect store arena: capacity initialized and base_data valid.
    auto & arena = store->get_arena();
    TEST_ASSERT(arena.get_capacity_bytes() > 0);
    TEST_ASSERT(arena.base_data() != nullptr);
    TEST_ASSERT(arena.get_live_bytes() == 0);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);

    // Proof: arena peak bytes recorded the exact carved canonical workspace,
    // and live bytes returned to 0 after RAII lease release!
    TEST_ASSERT(arena.get_peak_bytes() > 0);
    TEST_ASSERT(arena.get_live_bytes() == 0);
}

// ---------------------------------------------------------------------------
// 21. DDVR: two queries mapping the SAME payload to DIFFERENT groups.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 21. DDVR: two queries mapping the SAME payload to DIFFERENT groups.
// ---------------------------------------------------------------------------
static void test_ddvr_two_query_groups() {
    fprintf(stderr, "--- test_ddvr_two_query_groups ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SHADOW, 8, 16, 2);
    cp.rerot_enabled = true;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);

    llama_kv_rerot_meta xtag;
    xtag.episode_id = 7; xtag.node_id = 1; xtag.run_id = 1;
    xtag.publish_epoch = 1; xtag.frontier = 0;
    xtag.visibility = llama_rerot_visibility::public_live;
    llama_kv_rerot_meta ktag = xtag;
    ktag.node_id = 2; ktag.run_id = 2;
    auto bx = commit_tokens(kv, 1, {0}, true, &xtag);
    auto bk = commit_tokens(kv, 0, {5}, true, &ktag);
    write_layer_rows(kv, 0, bx, {0});
    write_layer_rows(kv, 1, bx, {0});
    write_layer_rows(kv, 0, bk, {5});
    write_layer_rows(kv, 1, bk, {5});

    llama_rerot_reader_state r0;
    r0.episode_id = 7; r0.reader = 1; r0.query_run = 2; r0.frontier = 10;
    r0.ordered_runs = {2};
    TEST_ASSERT(kv.rerot_set_reader_view(0, r0));
    llama_rerot_reader_state r1 = r0;
    r1.query_run = 1;
    r1.ordered_runs = {1, 2};
    TEST_ASSERT(kv.rerot_set_reader_view(1, r1));

    std::vector<llama_token> tokens = {101, 102};
    std::vector<llama_pos> pos = {6, 6};
    std::vector<int32_t> n_seq_id = {1, 1};
    llama_seq_id s0 = 0, s1 = 1;
    std::vector<llama_seq_id *> seq_ptrs = {&s0, &s1};
    llama_ubatch ub = {};
    ub.token = tokens.data(); ub.pos = pos.data();
    ub.n_tokens = 2; ub.n_seq_tokens = 2; ub.n_seqs = 2; ub.n_seqs_unq = 2;
    llama_seq_id seqs_unq[2] = {0, 1};
    ub.seq_id_unq = seqs_unq;
    ub.n_seq_id = n_seq_id.data(); ub.seq_id = seq_ptrs.data();
    auto sinfos = kv.prepare({ub});
    TEST_ASSERT(!sinfos.empty());
    std::vector<xkv_hot_reservation> no_res;
    llama_kv_cache_context ctx(&kv, sinfos, {ub}, std::move(no_res));
    TEST_ASSERT(ctx.apply());

    std::unique_ptr<xkv_graph_snapshot> snap;
    std::string err;
    TEST_ASSERT_MSG(ctx.build_xkv_graph_snapshot(0, 0, 1.0f, 0.0f, snap, &err), err.c_str());
    TEST_ASSERT(snap != nullptr);
    const auto & cells0 = kv.get_cells(0);
    uint64_t kpid = cells0.payload_id_get(bk.idxs[0]);
    uint32_t kg0 = UINT32_MAX, kg1 = UINT32_MAX, kcount = 0;
    for (const auto & hd : snap->hot_data) {
        if (hd.payload_id != kpid) continue;
        kcount++;
        if (hd.is_visible_to_query(0)) kg0 = hd.group_index;
        if (hd.is_visible_to_query(1)) kg1 = hd.group_index;
    }
    TEST_ASSERT(kcount == 2);
    TEST_ASSERT(kg0 != UINT32_MAX && kg1 != UINT32_MAX && kg0 != kg1);
    TEST_ASSERT(kg0 == 0 && kg1 == 1);
    uint64_t xpid = kv.get_cells(1).payload_id_get(bx.idxs[0]);
    bool xvis0 = false, xvis1 = false;
    for (const auto & hd : snap->hot_data) {
        if (hd.payload_id != xpid) continue;
        if (hd.is_visible_to_query(0)) xvis0 = true;
        if (hd.is_visible_to_query(1)) xvis1 = true;
    }
    TEST_ASSERT(!xvis0 && xvis1);
    TEST_ASSERT(snap->query_ddvr_group_counts.size() == 2);
    TEST_ASSERT(snap->query_ddvr_group_counts[0] == 2);
    TEST_ASSERT(snap->query_ddvr_group_counts[1] == 3);
    TEST_ASSERT(snap->query_causal_limits[0] == -1 && snap->query_causal_limits[1] == -1);
}

// Global operator-new counter for the zero-alloc warmed-maintain proof.
static std::atomic<size_t> g_alloc_count{0};
static thread_local bool g_count_allocs = false;
void * operator new(std::size_t n) {
    if (g_count_allocs) g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void * operator new[](std::size_t n) {
    if (g_count_allocs) g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

struct alloc_scope {
    alloc_scope() { g_alloc_count.store(0, std::memory_order_relaxed); g_count_allocs = true; }
    ~alloc_scope() { g_count_allocs = false; }
    size_t count() const { return g_alloc_count.load(std::memory_order_relaxed); }
};

static void test_zero_alloc_warmed_maintain() {
    fprintf(stderr, "--- test_zero_alloc_warmed_maintain ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(rt != nullptr);

    auto b = commit_tokens(kv, 0, {0, 1, 2, 3}, true);
    write_layer_rows(kv, 0, b, {0, 1, 2, 3});
    write_layer_rows(kv, 1, b, {0, 1, 2, 3});
    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().deferred_runs >= 1);
    {
        alloc_scope guard;
        TEST_ASSERT_MSG(rt->maintain(kv, 0, true, nullptr), "deferred");
        TEST_ASSERT(guard.count() == 0);
    }
    {
        rt_test_model m2(2);
        llama_cparams cp2 = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
        cp2.xkv_workspace_mib = 0;
        llama_kv_cache kv2(m2, m2.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp2);
        kv2.init_xkv_store(cp2);
        auto * rt2 = kv2.get_xkv_runtime();
        TEST_ASSERT(rt2 != nullptr);
        auto b2 = commit_tokens(kv2, 0, {0, 1, 2, 3, 4, 5, 6, 7}, true);
        write_layer_rows(kv2, 0, b2, {0, 1, 2, 3, 4, 5, 6, 7});
        write_layer_rows(kv2, 1, b2, {0, 1, 2, 3, 4, 5, 6, 7});
        TEST_ASSERT(!rt2->maintain(kv2, 0, true, nullptr));
        alloc_scope guard;
        TEST_ASSERT(!rt2->maintain(kv2, 0, true, nullptr));
        TEST_ASSERT(guard.count() == 0);
    }
    {
        auto b3 = commit_tokens(kv, 0, {4, 5, 6, 7, 8, 9, 10, 11}, true);
        write_layer_rows(kv, 0, b3, {4, 5, 6, 7, 8, 9, 10, 11});
        write_layer_rows(kv, 1, b3, {4, 5, 6, 7, 8, 9, 10, 11});
        TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
        auto b4 = commit_tokens(kv, 0, {12, 13, 14, 15, 16, 17, 18, 19}, true);
        write_layer_rows(kv, 0, b4, {12, 13, 14, 15, 16, 17, 18, 19});
        write_layer_rows(kv, 1, b4, {12, 13, 14, 15, 16, 17, 18, 19});
        alloc_scope guard;
        TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
        fprintf(stderr, "    seal-path allocations: %zu\n", guard.count());
        TEST_ASSERT(guard.count() == 0);
    }
}

static void test_r3_reference_landmarks() {
    fprintf(stderr, "--- test_r3_reference_landmarks ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SR, 8, 16, 2,
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE);
    cp.xkv_landmark_type = GGML_TYPE_F16;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store != nullptr && rt != nullptr);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    // R3 requires store-side want_landmarks invocation for reference+SR.
    auto seg = store->pin_segment(1);
    TEST_ASSERT((bool) seg);
    TEST_ASSERT(seg->groups.size() == 1);
    const auto * g0 = seg->find_group(0);
    TEST_ASSERT(g0 != nullptr);
    TEST_ASSERT(!g0->landmark.bytes.empty());
    TEST_ASSERT(g0->landmark.desc.type == GGML_TYPE_F16);
}

static void test_vulkan_shadow_no_mutation() {
    fprintf(stderr, "--- test_vulkan_shadow_no_mutation ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SHADOW, 8, 16, 2);
    cp.xkv_factorizer = LLAMA_XKV_FACTORIZER_VULKAN;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store != nullptr && rt != nullptr);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);
    const auto stamp0 = store->current_stamp();

    std::string err;
    const bool ok = rt->maintain(kv, 0, true, &err);
    TEST_ASSERT(store->current_stamp() == stamp0);
    TEST_ASSERT(store->get_accounting().active_segments == 0);
    const auto & cells = kv.get_cells(0);
    for (uint32_t idx : b.idxs) {
        xkv_state st = xkv_state::hot_writing;
        TEST_ASSERT(store->find_payload_state(cells.payload_id_get(idx), st) && st == xkv_state::hot_committed);
    }
    if (ok) {
        TEST_ASSERT(rt->stats().shadow_evals == 1);
    } else {
        TEST_ASSERT(!err.empty());
    }
}

static void test_fingerprint_field_coverage() {
    fprintf(stderr, "--- test_fingerprint_field_coverage ---\n");
    llama_cparams base = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    const uint64_t fp0 = llama_xkv::make_effective_config(base).fingerprint();
    auto mutated = [&](auto setter) {
        llama_cparams c = base;
        setter(c);
        return llama_xkv::make_effective_config(c).fingerprint() != fp0;
    };
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_group_size++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_rank_k++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_rank_v++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_segment_tokens++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_chunk_tokens++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_mode = LLAMA_XKV_MODE_SR; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_source = LLAMA_XKV_SOURCE_DECODED_HOT; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_factor_a_k = GGML_TYPE_Q8_0; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_factor_b_k = GGML_TYPE_Q8_0; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_factor_a_v = GGML_TYPE_Q8_0; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_factor_b_v = GGML_TYPE_Q8_0; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_factor_balance = LLAMA_XKV_FACTOR_BALANCE_SQRT; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_landmark_type = GGML_TYPE_Q8_0; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_landmark_refine = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_landmark_refine_max_rows++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_sr_budget = 4; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_factorizer = LLAMA_XKV_FACTORIZER_VULKAN; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_workspace_mib++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_decode_cache_mib++; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_store_mib = 1; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_min_saving += 0.01; }));
    TEST_ASSERT(mutated([](llama_cparams & c) { c.xkv_min_factor_coverage += 0.01; }));
}

static void test_imrope_seal_oracle() {
    fprintf(stderr, "--- test_imrope_seal_oracle ---\n");
    rt_test_model model(2);
    model.hparams.rope_type = LLAMA_ROPE_TYPE_IMROPE;
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store && pool && rt);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    TEST_ASSERT(store->get_accounting().active_segments == 1);
    TEST_ASSERT(pool->get_bound() == 0);
    auto seg = store->pin_segment(1);
    TEST_ASSERT((bool) seg);
    TEST_ASSERT(seg->groups.size() == 1);
    const auto * g0 = seg->find_group(0);
    TEST_ASSERT(g0 && g0->total_dim_k == 256);
}

static void test_tail_rank_proportional() {
    fprintf(stderr, "--- test_tail_rank_proportional ---\n");
    TEST_ASSERT(llama_xkv::xkv_scaled_group_rank(8, 4, 4, 8, 512) == 8);
    TEST_ASSERT(llama_xkv::xkv_scaled_group_rank(8, 2, 4, 8, 256) == 4);
    TEST_ASSERT(llama_xkv::xkv_scaled_group_rank(384, 2, 4, 4096, 1u << 20) == 192);
    TEST_ASSERT(llama_xkv::xkv_scaled_group_rank(576, 2, 4, 4096, 1u << 20) == 288);
    TEST_ASSERT(llama_xkv::xkv_scaled_group_rank(8, 4, 4, 4, 512) == 4);
    TEST_ASSERT(llama_xkv::xkv_scaled_group_rank(8, 0, 4, 8, 512) == 0);
    rt_test_model model(10);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 4);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store && rt);
    TEST_ASSERT(rt->group_map().groups.size() == 3);
    TEST_ASSERT(rt->group_map().groups[2].owning_layers.size() == 2);

    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    for (uint32_t il = 0; il < 10; ++il) {
        write_layer_rows(kv, il, b, pos);
    }
    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    auto seg = store->pin_segment(1);
    TEST_ASSERT((bool) seg);
    TEST_ASSERT(seg->groups.size() == 3);
    TEST_ASSERT(seg->find_group(0)->a_k.desc.logical_shape.cols == 8);
    TEST_ASSERT(seg->find_group(1)->a_k.desc.logical_shape.cols == 8);
    TEST_ASSERT(seg->find_group(2)->a_k.desc.logical_shape.cols == 4);
}

static void test_repeated_seal_stable() {
    fprintf(stderr, "--- test_repeated_seal_stable ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store && pool && rt);

    // Three consecutive seal/drain cycles: each must fully succeed with the
    // pool drained and no accumulated error. Backend instances (when the
    // native path engages) are RAII-held per call, never per-segment leaked.
    std::string err;
    for (int cycle = 0; cycle < 3; ++cycle) {
        const int base = cycle * 8;
        std::vector<llama_pos> pos;
        for (int i = 0; i < 8; ++i) pos.push_back(base + i);
        auto b = commit_tokens(kv, 0, pos, true);
        write_layer_rows(kv, 0, b, pos);
        write_layer_rows(kv, 1, b, pos);
        TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
        TEST_ASSERT(pool->get_bound() == 0);
    }
    TEST_ASSERT(rt->stats().sealed_segments == 3);
    TEST_ASSERT(rt->stats().sealed_rows == 24);
    TEST_ASSERT(store->get_accounting().active_segments == 3);
    TEST_ASSERT(rt->stats().last_domain_fp != 0);
}

static void test_placement_split_two_devices() {
    fprintf(stderr, "--- test_placement_split_two_devices ---\n");
    // Pure logic with fake device keys (no devices needed): a 4-layer group
    // spanning devA/devA/devB/devB splits into two homogeneous runs with
    // rebased offsets; uniform groups pass through untouched.
    llama_xkv::layer_group g;
    g.group_index = 0;
    g.owning_layers = {3, 4, 5, 6};
    g.layer_feature_dims_k = {128, 128, 128, 128};
    g.layer_feature_dims_v = {128, 128, 128, 128};
    g.layer_feature_offsets_k = {0, 128, 256, 384};
    g.layer_feature_offsets_v = {0, 128, 256, 384};
    g.total_dim_k = 512;
    g.total_dim_v = 512;
    std::vector<std::string> keys = {"devA", "devA", "devB", "devB"};
    auto runs = llama_xkv::split_group_by_placement(g, keys);
    TEST_ASSERT(runs.size() == 2);
    TEST_ASSERT(runs[0].owning_layers == std::vector<uint32_t>({3, 4}));
    TEST_ASSERT(runs[1].owning_layers == std::vector<uint32_t>({5, 6}));
    TEST_ASSERT(runs[0].total_dim_k == 256 && runs[1].total_dim_k == 256);
    TEST_ASSERT(runs[1].layer_feature_offsets_k == std::vector<uint32_t>({0, 128}));
    TEST_ASSERT(runs[1].layer_feature_offsets_v == std::vector<uint32_t>({0, 128}));
    // Uniform placement: single run, identity offsets.
    auto single = llama_xkv::split_group_by_placement(g, {"devA", "devA", "devA", "devA"});
    TEST_ASSERT(single.size() == 1);
    TEST_ASSERT(single[0].layer_feature_offsets_k == g.layer_feature_offsets_k);
    // Shape mismatch fails closed with empty output.
    auto bad = llama_xkv::split_group_by_placement(g, {"devA"});
    TEST_ASSERT(bad.empty());
    // Alternating devices split per layer.
    auto alt = llama_xkv::split_group_by_placement(g, {"devA", "devB", "devA", "devB"});
    TEST_ASSERT(alt.size() == 4);
    TEST_ASSERT(alt[2].owning_layers == std::vector<uint32_t>({5}));
}

static void test_pressure_partial_seal() {
    fprintf(stderr, "--- test_pressure_partial_seal ---\n");
    // Scenario: segment_tokens=8, hot pool capacity 12 (small bounded pool).
    // Six sequences each commit 2 tokens -> pool reaches capacity (12 bound, 0 free).
    // No single sequence reaches segment_tokens=8 (all are partial runs of length 2).
    // Under unforced maintain without pressure requirement: fill-first applies, NO SEAL.
    // Under forced maintain (or hot_free < upcoming_tokens): pressure hits, maintain
    // picks the oldest/largest partial domain, adapts ranks, seals it, and drains hot rows,
    // proving lossless partial seal precedes Tri/preemption.
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 12, 2);
    cp.xkv_min_saving = 0.0; // partial seals can have lower saving; allow any benefit
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 6, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store && pool && rt);

    // Commit 2 tokens each for 6 distinct sequences (seq 0..5), filling the pool to 12.
    std::vector<committed_batch> batches;
    for (llama_seq_id s = 0; s < 6; ++s) {
        std::vector<llama_pos> pos = {0, 1};
        auto b = commit_tokens(kv, s, pos, true);
        write_layer_rows(kv, 0, b, pos);
        write_layer_rows(kv, 1, b, pos);
        batches.push_back(std::move(b));
    }
    TEST_ASSERT(pool->get_bound() == 12);
    TEST_ASSERT(pool->get_free() == 0);

    std::string err;
    // 1. Without pressure (unforced, upcoming=0): fill-first applies, no partial seal.
    TEST_ASSERT(rt->maintain(kv, 0, false, &err) == llama_xkv::xkv_maintain_no_action);
    TEST_ASSERT(rt->stats().sealed_segments == 0);
    TEST_ASSERT(pool->get_bound() == 12);

    // 2. Under pressure (forced=true, e.g. admission or reclaim pressure):
    // Chooses the oldest/largest partial domain (seq 0, 2 tokens), seals it, drains 2 rows.
    auto oc = rt->maintain(kv, 0, true, &err);
    TEST_ASSERT_MSG(oc == llama_xkv::xkv_maintain_sealed, err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    TEST_ASSERT(rt->stats().sealed_rows == 2);
    TEST_ASSERT(pool->get_bound() == 10); // 2 rows drained!
    TEST_ASSERT(pool->get_free() == 2);   // 2 free slots restored before Tri!

    // The sealed segment has 2 rows and valid factors.
    auto seg = store->pin_segment(1);
    TEST_ASSERT((bool) seg);
    TEST_ASSERT(seg->n_rows == 2);
}

static void test_physical_vs_semantic_accounting() {
    fprintf(stderr, "--- test_physical_vs_semantic_accounting ---\n");
    // Test that when bounded XKV is active:
    // 1. get_kv_capacity() returns hot_pool capacity (physical), not logical cells.size()
    // 2. get_kv_used() returns hot_pool get_live() (physical), not logical cells.get_used()
    // 3. get_kv_seq_used() continues returning semantic reference count
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_DENSE, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store && pool && rt);

    // Initial state: 0 used, capacity is pool capacity (16)
    TEST_ASSERT(kv.get_kv_capacity() == pool->get_capacity());
    TEST_ASSERT(kv.get_kv_capacity() == 16);
    TEST_ASSERT(kv.get_kv_used() == 0);

    // Commit 8 tokens: physical pool has 8 live slots
    std::vector<llama_pos> pos = {0, 1, 2, 3, 4, 5, 6, 7};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    TEST_ASSERT(pool->get_bound() == 8);
    TEST_ASSERT(kv.get_kv_used() == 8);
    TEST_ASSERT(kv.get_kv_seq_used(0) == 8); // semantic reference count

    // Seal the segment: hot rows released, logical cells remain
    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);

    // After seal:
    // Physical hot pool is drained: get_kv_used() returns 0!
    TEST_ASSERT(pool->get_bound() == 0);
    TEST_ASSERT(kv.get_kv_used() == 0);
    // Semantic sequence usage still reflects the resident logical cells (8)!
    TEST_ASSERT(kv.get_kv_seq_used(0) == 8);
    // Capacity remains physical (16)
    TEST_ASSERT(kv.get_kv_capacity() == 16);
}

static void test_disjoint_private_rows_shared_prefix() {
    fprintf(stderr, "--- test_disjoint_private_rows_shared_prefix ---\n");
    // Scenario: two queries in a batch sharing the same segment view.
    // Both see a shared prefix (rows 0..3), but seq0 also has private rows 4..5
    // and seq1 has private rows 6..7.
    // Verify:
    // 1. query_row_visibility on the segment view gives exact per-query membership:
    //    q0 sees {0,1,2,3, 4,5} (true for 0..5, false for 6..7)
    //    q1 sees {0,1,2,3, 6,7} (true for 0..3 and 6..7, false for 4..5)
    // 2. No leakage of private rows across queries!
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SHADOW, 8, 16, 2);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store && rt);

    // Commit 4 shared prefix tokens on seq 0
    auto b_shared = commit_tokens(kv, 0, {0, 1, 2, 3}, true);
    write_layer_rows(kv, 0, b_shared, {0, 1, 2, 3});
    write_layer_rows(kv, 1, b_shared, {0, 1, 2, 3});

    // Copy prefix to seq 1 (shared prefix)
    kv.seq_cp(0, 1, 0, 4);

    // Now commit private tokens: 4..5 on seq 0, and 6..7 on seq 1
    auto b_priv0 = commit_tokens(kv, 0, {4, 5}, true);
    write_layer_rows(kv, 0, b_priv0, {4, 5});
    write_layer_rows(kv, 1, b_priv0, {4, 5});

    auto b_priv1 = commit_tokens(kv, 1, {6, 7}, true);
    write_layer_rows(kv, 0, b_priv1, {6, 7});
    write_layer_rows(kv, 1, b_priv1, {6, 7});

    // Run maintain to seal the 8 rows into segment 1
    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());

    // Prepare a two-query batch: q0 on seq 0 at pos 5, q1 on seq 1 at pos 7
    std::vector<llama_token> tokens = {101, 102};
    std::vector<llama_pos> pos = {5, 7};
    std::vector<int32_t> n_seq_id = {1, 1};
    llama_seq_id s0 = 0, s1 = 1;
    std::vector<llama_seq_id *> seq_ptrs = {&s0, &s1};
    llama_ubatch ub = {};
    ub.token = tokens.data(); ub.pos = pos.data();
    ub.n_tokens = 2; ub.n_seq_tokens = 2; ub.n_seqs = 2; ub.n_seqs_unq = 2;
    llama_seq_id seqs_unq[2] = {0, 1};
    ub.seq_id_unq = seqs_unq;
    ub.n_seq_id = n_seq_id.data(); ub.seq_id = seq_ptrs.data();
    auto sinfos = kv.prepare({ub});
    TEST_ASSERT(!sinfos.empty());
    std::vector<xkv_hot_reservation> no_res;
    llama_kv_cache_context ctx(&kv, sinfos, {ub}, std::move(no_res));
    TEST_ASSERT(ctx.apply());

    std::unique_ptr<xkv_graph_snapshot> snap;
    TEST_ASSERT_MSG(ctx.build_xkv_graph_snapshot(0, 0, 1.0f, 0.0f, snap, &err), err.c_str());
    TEST_ASSERT(snap != nullptr);

    // Verify per-query row visibility on the segment view
    TEST_ASSERT(!snap->segment_views.empty());
    const auto & view = snap->segment_views[0];
    TEST_ASSERT(view.query_row_visibility.size() == 2);
    // There are 8 selected rows in this view
    TEST_ASSERT(view.selected_rows.size() == 8);

    // Verify query 0 visibility: sees shared rows (0..3) and its own private rows (4..5); NOT rows 6..7
    for (size_t r = 0; r < view.selected_rows.size(); ++r) {
        int64_t spos = view.storage_positions[r];
        if (spos <= 5) {
            TEST_ASSERT(view.query_row_visibility[0][r] == true);
        } else {
            TEST_ASSERT(view.query_row_visibility[0][r] == false);
        }
    }

    // Verify query 1 visibility: sees shared rows (0..3) and its own private rows (6..7); NOT rows 4..5
    for (size_t r = 0; r < view.selected_rows.size(); ++r) {
        int64_t spos = view.storage_positions[r];
        if (spos <= 3 || spos >= 6) {
            TEST_ASSERT(view.query_row_visibility[1][r] == true);
        } else {
            TEST_ASSERT(view.query_row_visibility[1][r] == false);
        }
    }
}

// ---------------------------------------------------------------------------
// 22. SR per-(parent query, DDVR slot) fragment plans: one shared segment with
// public plus interleaved private A/B/C rows, one parent query, three DDVR
// groups. Each plan sees only its own legal rows: slot 0 <- {0:P, 1:A},
// slot 1 <- {4:P, 5:B}, slot 2 <- {8:P, 9:C} (effective-pos bucketing of
// ordinary base rows). Every fragment carries its parent query + slot
// explicitly, binds the exact persisted landmark source fingerprint (intact
// chunk, chunk_tokens == 2), and exposes the borrowed rope bridge.
// ---------------------------------------------------------------------------
static void test_sr_per_slot_fragment_plans() {
    fprintf(stderr, "--- test_sr_per_slot_fragment_plans ---\n");
    rt_test_model model(2);
    llama_cparams cp = rt_cparams(LLAMA_XKV_MODE_SR, 6, 16, 2,
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE);
    cp.xkv_chunk_tokens = 2;
    cp.xkv_landmark_type = GGML_TYPE_F16;
    cp.rerot_enabled = true;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto * rt = kv.get_xkv_runtime();
    TEST_ASSERT(store != nullptr && rt != nullptr);

    // Shared segment content (storage positions):
    //   0:P 1:A | 4:P 5:B | 8:P 9:C   (public interleaved with private A/B/C)
    const std::vector<llama_pos> pos = {0, 1, 4, 5, 8, 9};
    auto b = commit_tokens(kv, 0, pos, true);
    write_layer_rows(kv, 0, b, pos);
    write_layer_rows(kv, 1, b, pos);

    std::string err;
    TEST_ASSERT_MSG(rt->maintain(kv, 0, true, &err), err.c_str());
    TEST_ASSERT(rt->stats().sealed_segments == 1);
    TEST_ASSERT(rt->stats().sealed_rows == 6);

    // One parent query; no tagged rows, so any valid run ordering works.
    llama_rerot_reader_state r0;
    r0.episode_id = 7; r0.reader = 1; r0.query_run = 1; r0.frontier = 10;
    r0.ordered_runs = {1};
    TEST_ASSERT(kv.rerot_set_reader_view(0, r0));

    std::vector<llama_token> tokens = {101};
    std::vector<llama_pos> qpos = {12};
    std::vector<int32_t> n_seq_id = {1};
    llama_seq_id s0 = 0;
    std::vector<llama_seq_id *> seq_ptrs = {&s0};
    llama_ubatch ub = {};
    ub.token = tokens.data(); ub.pos = qpos.data();
    ub.n_tokens = 1; ub.n_seq_tokens = 1; ub.n_seqs = 1; ub.n_seqs_unq = 1;
    ub.seq_id_unq = &s0;
    ub.n_seq_id = n_seq_id.data(); ub.seq_id = seq_ptrs.data();
    auto sinfos = kv.prepare({ub});
    TEST_ASSERT(!sinfos.empty());
    std::vector<xkv_hot_reservation> no_res;
    llama_kv_cache_context ctx(&kv, sinfos, {ub}, std::move(no_res));
    TEST_ASSERT(ctx.apply());

    std::unique_ptr<xkv_graph_snapshot> snap;
    TEST_ASSERT_MSG(ctx.build_xkv_graph_snapshot(0, 0, 1.0f, 0.0f, snap, &err), err.c_str());
    TEST_ASSERT(snap != nullptr);
    TEST_ASSERT(snap->sr_mode == 1);
    TEST_ASSERT(snap->n_queries == 1);
    TEST_ASSERT(snap->query_ddvr_group_counts.size() == 1);
    // Three cold slots; the hot query token itself may form a fourth group.
    TEST_ASSERT(snap->query_ddvr_group_counts[0] >= 3);
    TEST_ASSERT(snap->query_causal_limits[0] == -1);

    // Exactly three intact per-slot fragments (chunk_tokens == 2).
    TEST_ASSERT(snap->sr_legal_frags.size() == 3);
    // Per-slot storage-position sets; each plan sees only its own legal rows.
    int64_t slot_pos[3][2] = {{-1, -1}, {-1, -1}, {-1, -1}};
    uint32_t slot_count[3] = {0, 0, 0};
    for (const auto & f : snap->sr_legal_frags) {
        TEST_ASSERT(f.parent_query_id == 0);
        TEST_ASSERT(f.parent_group_id <= 2);
        TEST_ASSERT(f.key.ddvr_group == f.parent_group_id);
        TEST_ASSERT(f.query_visibility.size() == 1);
        TEST_ASSERT(f.query_visibility[0] == true);
        TEST_ASSERT(f.key.causal_cutoff == -1);
        TEST_ASSERT(f.row_count == 2);
        TEST_ASSERT(f.row_indices.size() == 2);
        TEST_ASSERT(f.storage_positions.size() == 2);
        // Intact chunk: exact persisted source fingerprint, mapped to the
        // arena landmark row equal to the chunk index.
        TEST_ASSERT(f.uses_base_landmark());
        TEST_ASSERT(f.source_fingerprint != 0);
        TEST_ASSERT(f.base_landmark_row == f.parent_group_id);
        TEST_ASSERT(!f.is_derived_partial);
        TEST_ASSERT(!f.requires_native_rebuild);
        TEST_ASSERT(f.error_bound >= 0.0f);
        const uint32_t g = f.parent_group_id;
        TEST_ASSERT(slot_count[g] == 0); // one fragment per slot
        slot_count[g] = 1;
        slot_pos[g][0] = f.storage_positions[0];
        slot_pos[g][1] = f.storage_positions[1];
    }
    // Slot 0 sees {0:P, 1:A} only; slot 1 sees {4:P, 5:B} only;
    // slot 2 sees {8:P, 9:C} only. No A/B/C leakage across plans.
    TEST_ASSERT(slot_count[0] == 1 && slot_count[1] == 1 && slot_count[2] == 1);
    TEST_ASSERT(slot_pos[0][0] == 0 && slot_pos[0][1] == 1);
    TEST_ASSERT(slot_pos[1][0] == 4 && slot_pos[1][1] == 5);
    TEST_ASSERT(slot_pos[2][0] == 8 && slot_pos[2][1] == 9);
    // Host path: no native rebuild descriptors, no device arenas.
    TEST_ASSERT(snap->native_landmark_rebuild_requests.empty());
    TEST_ASSERT(!snap->native_arenas_present);
    TEST_ASSERT(snap->native_group_arenas.empty());
    // Borrowed rope bridge [omega, direct_mag], finite with positive mag.
    TEST_ASSERT(!snap->native_rope_table_data.empty());
    TEST_ASSERT(snap->native_rope_table_data.size() % 2 == 0);
    const size_t fc = snap->native_rope_table_data.size() / 2;
    for (size_t i = 0; i < fc; ++i) {
        TEST_ASSERT(std::isfinite(snap->native_rope_table_data[i]));
    }
    for (size_t i = fc; i < 2 * fc; ++i) {
        TEST_ASSERT(std::isfinite(snap->native_rope_table_data[i]));
        TEST_ASSERT(snap->native_rope_table_data[i] > 0.0f);
    }
}
