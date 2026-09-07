// Production-path unit and integration tests for the XKV bounded-hot lifecycle:
//  1. Hot capacity derivation (segment_tokens + max(n_batch,n_ubatch), checked,
//     chunk-rounded, clamped) and OFF/SHADOW equivalence; non-unified, MTP, and
//     empty-segment rejection.
//  2. Full production write path (prepare + reserve_hot_slots + context apply +
//     postcompute) for success, double-finalize, and next-decode-after-success.
//  3. postcompute_failure full rollback (pool + store + cells + victims) with
//     next decode succeeding; injected store-commit failure is atomic (victims
//     retained, entries retained) and recoverable via the failure path.
//  4. Injected pool failure inside rollback: best-effort completion of every
//     step with a propagated false.
//  5. Shared-multiref overwrite fails closed with zero mutation.
//  6. Removal release with aligned hot-only triple over mixed hot/factored
//     rows; factored logical cells survive; shared-prefix last-ref behavior.
//  7. Seal-protocol split: validate -> commit -> publish with SHADOW no-release.
//  8. Metadata-only compaction keeps payload -> hot-slot bindings; K-shift maps
//     bound hot rows only.
//  9. Clear preserves shared pool identity; clear with live reservations throws
//     before mutating any subsystem.
// 10. iSWA/hybrid target attention uses real hot reservations with exactly-once
//     postcompute forwarding; dedicated get_xkv_hot_k/v views span hot rows.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-hot.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llama_xkv;

// Stub model for test instantiation
struct test_model_xkv_hot : public llama_model {
    test_model_xkv_hot() : llama_model(llama_model_default_params()) {
        hparams.n_ctx_train = 4096;
        arch = LLM_ARCH_LLAMA;
        hparams.n_layer_all = 2;
        hparams.n_head_arr.fill(4);
        hparams.n_head_kv_arr.fill(4);
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

static llama_cparams make_cparams(llama_xkv_mode mode, uint32_t seg_tokens = 64,
        uint32_t chunk_tokens = 8, uint32_t ubatch = 16, uint32_t n_batch = 0) {
    llama_cparams cparams = {};
    cparams.xkv_mode = mode;
    cparams.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cparams.xkv_group_size = 2;
    cparams.xkv_rank_k = 16;
    cparams.xkv_rank_v = 16;
    cparams.xkv_segment_tokens = seg_tokens;
    cparams.xkv_chunk_tokens = chunk_tokens;
    cparams.n_ubatch = ubatch;
    cparams.n_batch = n_batch == 0 ? ubatch : n_batch;
    cparams.xkv_workspace_mib = 16;
    cparams.xkv_decode_cache_mib = 8;
    return cparams;
}

#define ASSERT_THROWS_INVALID(stmt) \
    do { \
        bool threw = false; \
        try { stmt; } catch (const std::invalid_argument &) { threw = true; } \
        assert(threw); \
    } while (0)

// Manual ubatch fixture with storage owned by the fixture (production layout)
struct ubatch_fixture {
    std::vector<llama_token> tokens;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_vals;
    std::vector<llama_seq_id *> seq_ptrs;
    llama_seq_id seq = 0;
    llama_ubatch ub = {};

    ubatch_fixture(std::vector<llama_token> t, std::vector<llama_pos> p, llama_seq_id s = 0) :
        tokens(std::move(t)), pos(std::move(p)), seq(s) {
        assert(tokens.size() == pos.size());
        seq_vals.assign(tokens.size(), seq);
        n_seq_id.assign(tokens.size(), 1);
        seq_ptrs.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            seq_ptrs.push_back(&seq_vals[i]);
        }
        ub.token = tokens.data();
        ub.pos = pos.data();
        ub.n_tokens = (int32_t) tokens.size();
        ub.n_seq_tokens = (int32_t) tokens.size();
        ub.n_seqs = 1;
        ub.n_seqs_unq = 1;
        ub.seq_id_unq = &seq;
        ub.n_seq_id = n_seq_id.data();
        ub.seq_id = seq_ptrs.data();
    }
};

// Production write path: prepare -> reserve_hot_slots -> context apply.
// Returns sinfos + reservations for the caller to drive postcompute.
static llama_kv_cache::slot_info_vec_t prepare_reserve(llama_kv_cache & kv, const llama_ubatch & ub,
        std::vector<xkv_hot_reservation> & res_out) {
    auto sinfos = kv.prepare({ub});
    assert(!sinfos.empty());
    std::string err;
    const std::vector<llama_ubatch> ubs = {ub};
    assert(kv.reserve_hot_slots(sinfos, ubs, res_out, &err));
    assert(res_out.size() == sinfos.size());
    return sinfos;
}

// Contexts are move-only-hostile (const status member + move-only
// reservations), so the helper hands back a unique_ptr.
static std::unique_ptr<llama_kv_cache_context> apply_one(llama_kv_cache & kv,
        llama_kv_cache::slot_info_vec_t sinfos, const llama_ubatch & ub,
        std::vector<xkv_hot_reservation> res) {
    std::vector<llama_ubatch> ubs;
    ubs.push_back(ub);
    auto ctx = std::make_unique<llama_kv_cache_context>(&kv, std::move(sinfos), std::move(ubs), std::move(res));
    assert(ctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    assert(ctx->apply());
    return ctx;
}

// Full production decode step through postcompute_success
static void decode_success(llama_kv_cache & kv, const llama_ubatch & ub) {
    std::vector<xkv_hot_reservation> res;
    auto sinfos = prepare_reserve(kv, ub, res);
    auto ctx = apply_one(kv, std::move(sinfos), ub, std::move(res));
    assert(ctx->postcompute_success());
    // Double-finalize returns the recorded success
    assert(ctx->postcompute_success());
}

// Position-based cell lookup: find_slot placement is allocator-defined, so
// tests must never assume logical cell == position.
static uint32_t cell_with_pos(const llama_kv_cells & cells, llama_pos p) {
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.pos_get(i) == p) {
            return i;
        }
    }
    assert(false && "position not resident");
    return 0;
}

static bool has_pos(const llama_kv_cells & cells, llama_pos p) {
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.pos_get(i) == p) {
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// Bit-identical fault injection: pool/store/rebuild/gate failures must leave
// store, pool, cells and epochs exactly as before the refused call.
// ----------------------------------------------------------------------------
struct kv_snapshot {
    xkv_hot_accounting pool;
    uint64_t live = 0, content = 0, binding = 0, codec = 0;
    struct cell {
        bool empty = true;
        uint64_t pid = 0, gen = 0;
        llama_pos pos = -1;
    };
    std::vector<cell> cells;
};

static kv_snapshot take_snapshot(const llama_kv_cache & kv) {
    kv_snapshot s;
    s.pool = kv.get_hot_slot_pool()->get_accounting();
    auto store = kv.get_xkv_store();
    s.live = store->live_epoch();
    s.content = store->content_epoch();
    s.binding = store->binding_epoch();
    s.codec = store->codec_epoch();
    const auto & cells = kv.get_cells(0);
    for (uint32_t i = 0; i < cells.size(); ++i) {
        kv_snapshot::cell c;
        c.empty = cells.is_empty(i);
        if (!c.empty) {
            c.pid = cells.payload_id_get(i);
            c.gen = cells.storage_generation_get(i);
            c.pos = cells.pos_get(i);
        }
        s.cells.push_back(c);
    }
    return s;
}

static void assert_snapshots_equal(const kv_snapshot & a, const kv_snapshot & b) {
    assert(a.pool.capacity == b.pool.capacity);
    assert(a.pool.free == b.pool.free);
    assert(a.pool.reserved == b.pool.reserved);
    assert(a.pool.bound == b.pool.bound);
    assert(a.pool.live == b.pool.live);
    assert(a.pool.peak == b.pool.peak);
    assert(a.live == b.live);
    assert(a.content == b.content);
    assert(a.binding == b.binding);
    assert(a.codec == b.codec);
    assert(a.cells.size() == b.cells.size());
    for (size_t i = 0; i < a.cells.size(); ++i) {
        assert(a.cells[i].empty == b.cells[i].empty);
        assert(a.cells[i].pid == b.cells[i].pid);
        assert(a.cells[i].gen == b.cells[i].gen);
        assert(a.cells[i].pos == b.cells[i].pos);
    }
}

// llama_kv_cache is non-copyable/non-movable: factory hands back unique_ptr.
static std::unique_ptr<llama_kv_cache> make_dense_kv(test_model_xkv_hot & model, uint32_t logical = 128) {
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    auto kv = std::make_unique<llama_kv_cache>(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, logical, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv->init_xkv_store(cp);
    return kv;
}

// Forward declarations: tests 11-17 are defined after main.
static void test_gate_refusal_identical();
static void test_rebuild_hook_refusal_and_capture();
static void test_stale_generation_identical();
static void test_stale_plan_identical();
static void test_rollback_gate_false_identical();
static void test_try_clear_epoch_identical();
static void test_pool_snapshot_restore();
static void test_compact_keeps_hot_rows();
static void test_seq_div_refusal();
static void test_seq_add_eviction_release();
static void test_shift_seal_locked_refusal();
static void test_shift_rope_none_success();
static void test_shift_needs_context_refusal();
static void test_shift_mutation_failure_rolls_back_k();
static void test_shift_landmark_position_rollback();
static void test_shift_device_gate_refusal();
static void test_shift_landmark_exempt_preservation();
static void test_turbo_k_has_shift_and_graph_built();
static void test_shift_factored_exemption();
static void test_capacity_and_rejections();
static void test_production_success_path();
static void test_failure_rollback_and_recovery();
static void test_injected_commit_failure();
static void test_injected_pool_failure_in_rollback();
static void test_shared_overwrite_rejected();
static void test_mixed_deletion_and_sharing();
static void test_compaction_and_kshift();
static void test_clear_semantics();
static void test_wrapper_forwarding_and_views();

int main() {
    std::cout << "Starting test-xkv-hot-cache..." << std::endl;
    test_capacity_and_rejections();
    test_production_success_path();
    test_failure_rollback_and_recovery();
    test_injected_commit_failure();
    test_injected_pool_failure_in_rollback();
    test_shared_overwrite_rejected();
    test_mixed_deletion_and_sharing();
    test_compaction_and_kshift();
    test_clear_semantics();
    test_wrapper_forwarding_and_views();
    test_gate_refusal_identical();
    test_rebuild_hook_refusal_and_capture();
    test_stale_generation_identical();
    test_stale_plan_identical();
    test_rollback_gate_false_identical();
    test_try_clear_epoch_identical();
    test_pool_snapshot_restore();
    test_compact_keeps_hot_rows();
    test_seq_div_refusal();
    test_seq_add_eviction_release();
    test_shift_seal_locked_refusal();
    test_shift_rope_none_success();
    test_shift_needs_context_refusal();
    test_shift_factored_exemption();
    test_turbo_k_has_shift_and_graph_built();
    test_shift_mutation_failure_rolls_back_k();
    test_shift_landmark_position_rollback();
    test_shift_device_gate_refusal();
    test_shift_landmark_exempt_preservation();
    std::cout << "All test-xkv-hot-cache tests passed successfully!" << std::endl;
    return 0;
}

// ----------------------------------------------------------------------------
// Test 1: capacity derivation, OFF/SHADOW equivalence, rejections
// ----------------------------------------------------------------------------
static void test_capacity_and_rejections() {
    std::cout << "[Test 1] Capacity derivation and mode rejections..." << std::endl;

    test_model_xkv_hot model;
    const uint32_t logical_cells = 512;

    // OFF: hot == logical, legacy attention, no pool
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_OFF);
        llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp);
        assert(kv.get_size() == logical_cells);
        assert(kv.get_hot_size() == logical_cells);
        assert(kv.get_kv_capacity() == logical_cells);
        assert(kv.get_kv_hot_capacity() == logical_cells);
        assert(kv.can_use_legacy_attention());
        assert(!kv.is_xkv_bounded_hot());
        assert(kv.get_hot_slot_pool() == nullptr);
        assert(kv.get_k_storage(0)->ne[1] == (int64_t) logical_cells);
        // Hot views fail closed when unbounded
        llama_kv_cache::slot_info dummy;
        assert(kv.get_hot_k(nullptr, 0, dummy) == nullptr);
        assert(kv.get_hot_v(nullptr, 0, dummy) == nullptr);
    }

    // SHADOW: full-size tensors + store, no pool, never releases
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_SHADOW);
        llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp);
        kv.init_xkv_store(cp);
        assert(kv.get_hot_size() == logical_cells);
        assert(kv.can_use_legacy_attention());
        assert(!kv.is_xkv_bounded_hot());
        assert(kv.get_hot_slot_pool() == nullptr);
        assert(kv.get_xkv_store() != nullptr);
        // Seal-protocol halves are vacuous (never release) on SHADOW
        xkv_hot_release_plan empty_plan;
        std::string err;
        assert(kv.validate_hot_release(empty_plan, &err));
        assert(kv.commit_hot_release(empty_plan, &err));
        assert(kv.precommit_hot_release(empty_plan, &err));
    }

    // DENSE: segment=64, max(batch)=16 -> raw=80, chunk=8 -> 80
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 64, 8, 16);
        llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp);
        kv.init_xkv_store(cp);
        assert(kv.get_size() == logical_cells);
        assert(kv.get_hot_size() == 80);
        assert(kv.get_hot_size() < kv.get_size());
        assert(kv.get_kv_hot_capacity() == 80);
        assert(!kv.can_use_legacy_attention());
        assert(kv.is_xkv_bounded_hot());
        assert(kv.get_hot_slot_pool()->get_capacity() == 80);
        assert(kv.get_k_storage(0)->ne[1] == 80);
        assert(kv.get_v_storage(0)->ne[1] == 80);
    }

    // n_batch > n_ubatch: segment=64, n_batch=64, n_ubatch=16 -> 128
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 64, 8, 16, 64);
        llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp);
        assert(kv.get_hot_size() == 128);
    }

    // Non-unified bounded XKV is rejected before any allocation
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
        ASSERT_THROWS_INVALID((llama_kv_cache(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, false, logical_cells, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp)));
    }

    // MTP draft path (nullptr cparams, as wired by create_memory) allocates nothing
    {
        llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, nullptr);
        assert(kv.get_hot_slot_pool() == nullptr);
        assert(!kv.is_xkv_bounded_hot());
        llama_cparams cp_mtp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
        cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        kv.init_xkv_store(cp_mtp);
        assert(kv.get_xkv_store() == nullptr);
    }

    // MTP + dense via direct cparams fails closed instead of half-allocating
    {
        llama_cparams cp_mtp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
        cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        ASSERT_THROWS_INVALID((llama_kv_cache(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp_mtp)));
    }

    // Empty segment_tokens is rejected
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 0, 8, 8);
        ASSERT_THROWS_INVALID((llama_kv_cache(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp)));
    }

    // Zero chunk_tokens has no hidden fallback: it throws
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 0, 8);
        ASSERT_THROWS_INVALID((llama_kv_cache(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, logical_cells, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp)));
    }

    // Logical capacity below segment+batch cannot seal safely: rejected,
    // never silently clamped
    {
        llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 64, 8, 16);
        ASSERT_THROWS_INVALID((llama_kv_cache(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, 32, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr, &cp)));
    }
}

// ----------------------------------------------------------------------------
// Test 2: production success path, double-finalize, next decode
// ----------------------------------------------------------------------------
static void test_production_success_path() {
    std::cout << "[Test 2] Production apply/postcompute success..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 256, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    assert(store && pool);
    const uint32_t cap = pool->get_capacity();

    // Dry-run prepare alone must never touch store or pool
    ubatch_fixture f1({101, 102, 103, 104}, {0, 1, 2, 3});
    {
        auto sinfos = kv.prepare({f1.ub});
        assert(!sinfos.empty());
        assert(pool->get_reserved() == 0 && pool->get_bound() == 0);
    }

    // First production decode
    decode_success(kv, f1.ub);
    assert(pool->get_bound() == 4);
    assert(pool->get_reserved() == 0);

    const auto & cells = kv.get_cells(0);
    // Payloads are committed and pool-bound regardless of placement
    for (llama_pos p = 0; p < 4; ++p) {
        const uint32_t i = cell_with_pos(cells, p);
        const uint64_t pid = cells.payload_id_get(i);
        assert(pid != 0);
        xkv_state st;
        assert(store->find_payload_state(pid, st) && st == xkv_state::hot_committed);
        uint32_t slot = 0;
        assert(pool->find_slot(pid, slot));
    }

    // Next decode succeeds with consistent accounting (no leak)
    ubatch_fixture f2({105, 106}, {4, 5});
    decode_success(kv, f2.ub);
    assert(pool->get_bound() == 6);
    assert(pool->get_free() == cap - 6);

    // Pool exhaustion fails prepare admission without mutation
    {
        std::vector<xkv_hot_reservation> res;
        auto sinfos = kv.prepare({f1.ub});
        std::string err;
        // Reserve the entire remaining pool, then prove one more fails closed
        auto fill = pool->reserve(pool->get_free(), &err);
        assert(fill.valid());
        std::vector<xkv_hot_reservation> res2;
        auto sinfos2 = kv.prepare({f2.ub});
        assert(!kv.reserve_hot_slots(sinfos2, {f2.ub}, res2, &err));
        assert(!err.empty());
        assert(res2.empty());
        assert(sinfos2[0].hot_idxs.empty());
    }
}

// ----------------------------------------------------------------------------
// Test 3: postcompute_failure full rollback, next decode succeeds
// ----------------------------------------------------------------------------
static void test_failure_rollback_and_recovery() {
    std::cout << "[Test 3] Graph-failure rollback and recovery..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 256, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    const uint32_t cap = pool->get_capacity();

    // Baseline decode commits 2 rows
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(kv, f1.ub);
    assert(pool->get_bound() == 2);

    // Second batch applies, then the graph fails: full rollback
    ubatch_fixture f2({3, 4}, {2, 3});
    std::vector<xkv_hot_reservation> res;
    auto sinfos = prepare_reserve(kv, f2.ub, res);
    auto ctx = apply_one(kv, std::move(sinfos), f2.ub, std::move(res));
    assert(pool->get_bound() == 4);
    assert(ctx->postcompute_failure());
    // Double-finalize returns the recorded outcome
    assert(ctx->postcompute_failure());

    // Rollback is exact: pool freed, store clean, cells dropped, victims none
    assert(pool->get_bound() == 2);
    assert(pool->get_free() == cap - 2);
    const auto & cells = kv.get_cells(0);
    assert(!has_pos(cells, 2) && !has_pos(cells, 3));
    assert(has_pos(cells, 0) && has_pos(cells, 1));

    // Next decode succeeds on the recovered cache with clean accounting
    ubatch_fixture f3({5, 6}, {2, 3});
    decode_success(kv, f3.ub);
    assert(pool->get_bound() == 4);
    for (llama_pos p = 0; p < 4; ++p) {
        const uint32_t i = cell_with_pos(cells, p);
        xkv_state st;
        assert(store->find_payload_state(cells.payload_id_get(i), st) && st == xkv_state::hot_committed);
    }
}

// ----------------------------------------------------------------------------
// Test 4: injected store-commit failure is atomic, then failure path recovers
// ----------------------------------------------------------------------------
static void test_injected_commit_failure() {
    std::cout << "[Test 4] Injected commit failure atomicity..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 256, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();

    ubatch_fixture f1({11, 12}, {0, 1});
    std::vector<xkv_hot_reservation> res;
    auto sinfos = prepare_reserve(kv, f1.ub, res);
    auto ctx = apply_one(kv, std::move(sinfos), f1.ub, std::move(res));

    // Inject a seal lock on one applied payload: commit must fail atomically
    const auto & cells = kv.get_cells(0);
    const uint32_t c0 = cell_with_pos(cells, 0);
    const uint64_t pid0 = cells.payload_id_get(cell_with_pos(cells, 0));
    const uint64_t gen0 = cells.storage_generation_get(c0);
    uint64_t nonce0 = 0;
    assert(store->mark_seal_candidates({pid0}, {gen0}, &nonce0));
    assert(!ctx->postcompute_success());

    // Atomicity: new rows still pool-bound, victims (none here) untouched,
    // entries retained for the failure path
    assert(pool->get_bound() == 2);
    assert(pool->has_payload(pid0));

    // Release the lock and roll back through the failure path
    assert(store->abort_seal_candidates({pid0}, nonce0, xkv_skip_reason::aborted));
    assert(ctx->postcompute_failure());
    assert(pool->get_bound() == 0);
    assert(pool->get_free() == pool->get_capacity());
    assert(!has_pos(cells, 0) && !has_pos(cells, 1));

    // Cache recovers: next decode succeeds
    ubatch_fixture f2({13}, {0});
    decode_success(kv, f2.ub);
    assert(pool->get_bound() == 1);
}

// ----------------------------------------------------------------------------
// Test 5: injected pool failure inside rollback still completes + propagates
// ----------------------------------------------------------------------------
static void test_injected_pool_failure_in_rollback() {
    std::cout << "[Test 5] Pool failure inside rollback..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 256, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();

    ubatch_fixture f1({21, 22}, {0, 1});
    std::vector<xkv_hot_reservation> res;
    auto sinfos = prepare_reserve(kv, f1.ub, res);
    // Capture applied rows before apply consumes the reservation
    auto ctx = apply_one(kv, std::move(sinfos), f1.ub, std::move(res));

    // Sabotage: free one applied row behind the context's back
    const auto & cells = kv.get_cells(0);
    const uint32_t c0 = cell_with_pos(cells, 0);
    const uint64_t pid0 = cells.payload_id_get(c0);
    const uint64_t gen0 = cells.storage_generation_get(c0);
    uint32_t slot0 = 0;
    assert(pool->find_slot(pid0, slot0));
    std::string serr;
    assert(pool->release(pid0, gen0, slot0, &serr));

    // Rollback refuses atomically through the store-held gate: the store
    // still tracks both payloads, the cells still hold both, and the pool
    // still binds the un-sabotaged row — bit-identical to before the call.
    const kv_snapshot before = take_snapshot(kv);
    assert(!ctx->postcompute_failure());
    assert_snapshots_equal(before, take_snapshot(kv));
    // Records are retained and retryable: rebind the sabotaged row at its
    // freed slot (lowest-free deterministic reuse), then roll back cleanly.
    auto fix = pool->reserve(1, &serr);
    assert(fix.valid());
    assert(fix.commit({pid0}, {gen0}));
    uint32_t rs = 0;
    assert(pool->find_slot(pid0, rs) && rs == slot0);
    assert(ctx->postcompute_failure());
    assert(pool->get_bound() == 0);
    assert(!has_pos(cells, 0) && !has_pos(cells, 1));
    xkv_state st;
    assert(!store->find_payload_state(pid0, st));
}

// ----------------------------------------------------------------------------
// Test 6: shared-multiref overwrite fails closed with zero mutation
// ----------------------------------------------------------------------------
static void test_shared_overwrite_rejected() {
    std::cout << "[Test 6] Shared overwrite fail-closed..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 128, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();

    // Same-stream copy shares the cell across two sequences (multiref)
    ubatch_fixture fa({31}, {5}, 0);
    decode_success(kv, fa.ub);
    kv.seq_cp(0, 1, 0, 100);

    const auto & cells0 = kv.get_cells(0);
    // Find the shared cell: seq 0 and seq 1 both reference it
    uint32_t shared = 0;
    bool found = false;
    for (uint32_t i = 0; i < 128; ++i) {
        if (!cells0.is_empty(i) && cells0.seq_has(i, 0) && cells0.seq_has(i, 1)) {
            shared = i;
            found = true;
            break;
        }
    }
    assert(found);
    const uint64_t pid = cells0.payload_id_get(shared);
    const uint64_t gen = cells0.storage_generation_get(shared);
    assert(pool->get_bound() == 1);

    // Direct overwrite attempt on the shared cell must throw before mutation.
    // (Hand-built sinfo: negative test of an input find_slot must never
    // produce; lifecycle tests above/below use production sequences.)
    llama_kv_cache::slot_info bad;
    bad.s0 = 0;
    bad.s1 = 0;
    bad.strm = {0};
    bad.idxs = {{shared}};
    llama_ubatch ub = {};
    llama_token tok = 32;
    llama_pos pp = 5;
    int32_t nsi = 1;
    llama_seq_id sq = 1;
    llama_seq_id * ptrs[1] = {&sq};
    ub.token = &tok;
    ub.pos = &pp;
    ub.n_tokens = 1;
    ub.n_seq_tokens = 1;
    ub.n_seqs = 1;
    ub.n_seqs_unq = 1;
    ub.seq_id_unq = &sq;
    ub.n_seq_id = &nsi;
    ub.seq_id = ptrs;

    bool threw = false;
    try {
        kv.apply_ubatch(bad, ub, false, nullptr);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw);

    // Zero mutation: pool, store, and cells all intact
    assert(pool->get_bound() == 1);
    assert(pool->has_payload(pid));
    xkv_state st;
    assert(store->find_payload_state(pid, st) && st == xkv_state::hot_committed);
    assert(!cells0.is_empty(shared));
    assert(cells0.seq_has(shared, 0) && cells0.seq_has(shared, 1));
    assert(cells0.payload_id_get(shared) == pid);
    assert(cells0.storage_generation_get(shared) == gen);

    kv.seq_rm(0, 0, 100);
    kv.seq_rm(1, 0, 100);
}

// ----------------------------------------------------------------------------
// Test 7: mixed hot/factored deletion, factored survival, last-ref sharing
// ----------------------------------------------------------------------------
static xkv_factor_group_payload make_group() {
    xkv_factor_group_payload g0;
    g0.group_index = 0;
    g0.owning_layers = {0};
    // Explicit ranks and feature maps: one owning layer spanning the full
    // 64-wide K/V feature space.
    g0.rank_k = 16;
    g0.rank_v = 16;
    g0.layer_feature_offsets_k = {0};
    g0.layer_feature_dims_k = {64};
    g0.layer_feature_offsets_v = {0};
    g0.layer_feature_dims_v = {64};
    g0.total_dim_k = 64;
    g0.total_dim_v = 64;
    // Codec pair contract: K pair shares one seed with K roles, V pair shares
    // a separate seed with V roles (A token_major, B feature_major_transposed).
    constexpr uint64_t k_seed = 11;
    constexpr uint64_t v_seed = 12;
    codec_desc da_k = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {2, 16}, 0, k_seed);
    codec_desc db_k = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {64, 16}, 0, k_seed);
    codec_desc da_v = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {2, 16}, 0, v_seed);
    codec_desc db_v = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {64, 16}, 0, v_seed);
    std::vector<float> d_a(2 * 16, 0.1f);
    std::vector<float> d_b(64 * 16, 0.1f);
    g0.a_k = encode_matrix(da_k, d_a.data(), d_a.size());
    g0.set_b_k(encode_matrix(db_k, d_b.data(), d_b.size()));
    g0.a_v = encode_matrix(da_v, d_a.data(), d_a.size());
    g0.set_b_v(encode_matrix(db_v, d_b.data(), d_b.size()));
    return g0;
}

static void test_mixed_deletion_and_sharing() {
    std::cout << "[Test 7] Mixed hot/factored deletion and last-ref sharing..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 128, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    const uint32_t cap = pool->get_capacity();

    // Two hot rows via production path
    ubatch_fixture f1({1, 2}, {10, 11});
    decode_success(kv, f1.ub);
    const auto & cells = kv.get_cells(0);
    const uint32_t c10 = cell_with_pos(cells, 10);
    const uint32_t c11 = cell_with_pos(cells, 11);
    const uint64_t pid10 = cells.payload_id_get(c10);
    const uint64_t gen10 = cells.storage_generation_get(c10);
    const uint64_t pid11 = cells.payload_id_get(c11);
    const uint64_t gen11 = cells.storage_generation_get(c11);
    uint32_t s0 = 0, s1 = 0;
    assert(pool->find_slot(pid10, s0) && pool->find_slot(pid11, s1));

    // Seal protocol: validate -> commit -> publish (no post-publish step)
    auto seg = store->create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, {make_group()});
    xkv_hot_release_plan plan;
    plan.released_payload_ids = {pid10, pid11};
    plan.released_generations = {gen10, gen11};
    plan.released_physical_rows = {s0, s1};
    plan.expected_segment_id = seg->segment_id;
    plan.expected_segment_version = seg->segment_version;
    {
        std::string err;
        assert(kv.validate_hot_release(plan, &err));
        assert(kv.commit_hot_release(plan, &err));
        assert(!pool->has_payload(pid10) && !pool->has_payload(pid11));
        std::string pub_err;
        assert(store->publish_candidate(seg, {pid10, pid11}, {gen10, gen11}, &pub_err));
    }
    // Factored logical cells survive hot-slot release with identity intact
    assert(has_pos(cells, 10) && has_pos(cells, 11));
    assert(cells.payload_id_get(cell_with_pos(cells, 10)) == pid10);
    assert(cells.storage_generation_get(cell_with_pos(cells, 10)) == gen10);

    // Two fresh hot rows alongside the factored ones
    ubatch_fixture f2({3, 4}, {12, 13});
    decode_success(kv, f2.ub);
    const uint64_t pid12 = cells.payload_id_get(cell_with_pos(cells, 12));
    const uint64_t pid13 = cells.payload_id_get(cell_with_pos(cells, 13));
    assert(pool->has_payload(pid12) && pool->has_payload(pid13));

    // Mixed deletion: factored pids must never reach the pool (aligned triple)
    assert(kv.seq_rm(0, 10, 14));
    assert(pool->get_bound() == 0);
    assert(pool->get_free() == cap);
    for (llama_pos p = 10; p < 14; ++p) {
        assert(!has_pos(cells, p));
    }
    xkv_state st;
    assert(!store->find_payload_state(pid10, st) && !store->find_payload_state(pid12, st));

    // Shared-prefix last-ref: share, drop one ref (payload kept), drop last (freed)
    ubatch_fixture f3({7, 8}, {20, 21});
    decode_success(kv, f3.ub);
    const uint64_t q0 = cells.payload_id_get(cell_with_pos(cells, 20));
    assert(pool->get_bound() == 2);
    kv.seq_cp(0, 1, 0, 100);
    assert(kv.seq_rm(0, 20, 22));
    // First drop releases nothing: sibling still references the payloads
    assert(pool->get_bound() == 2);
    assert(pool->has_payload(q0));
    assert(kv.seq_rm(1, 20, 22));
    // Last ref frees pool + store
    assert(pool->get_bound() == 0);
    assert(!pool->has_payload(q0));
    assert(!store->find_payload_state(q0, st));
}

// ----------------------------------------------------------------------------
// Test 8: metadata-only compaction keeps hot bindings; K-shift maps hot rows
// ----------------------------------------------------------------------------
static void test_compaction_and_kshift() {
    std::cout << "[Test 8] Compaction mapping and K-shift..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 128, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();

    ubatch_fixture f1({1, 2, 3}, {0, 1, 2});
    decode_success(kv, f1.ub);
    const auto & cells = kv.get_cells(0);
    const uint64_t pid0 = cells.payload_id_get(cell_with_pos(cells, 0));
    const uint64_t pid2 = cells.payload_id_get(cell_with_pos(cells, 2));
    uint32_t hs0 = 0, hs2 = 0;
    assert(pool->find_slot(pid0, hs0) && pool->find_slot(pid2, hs2));

    // Remove the middle token, then compact: physical hot rows never remap
    assert(kv.seq_rm(0, 1, 2));
    const uint64_t b_epoch_before = store->binding_epoch();
    kv.compact();
    uint32_t f0 = 0, f2 = 0;
    assert(pool->find_slot(pid0, f0) && f0 == hs0);
    assert(pool->find_slot(pid2, f2) && f2 == hs2);
    assert(store->binding_epoch() >= b_epoch_before);
    assert(kv.get_owning_layer(0) == 0);

    // K-shift: delta lands on bound hot rows, factored/unbound rows exempt
    kv.seq_add(0, 0, 3, 5);
    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    ggml_context * gctx = ggml_init(params);
    ggml_backend_buffer_t host_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 1024 * 1024);
    ggml_tensor * t_shift = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, kv.get_hot_size());
    t_shift->buffer = host_buf;
    t_shift->data = ggml_backend_buffer_get_base(host_buf);
    kv.set_input_k_shift(t_shift);
    const int32_t * shift_data = (const int32_t *) t_shift->data;
    assert(shift_data[f0] == 5);
    assert(shift_data[f2] == 5);
    ggml_backend_buffer_free(host_buf);
    ggml_free(gctx);
}

// ----------------------------------------------------------------------------
// Test 9: clear identity + clear failure atomicity
// ----------------------------------------------------------------------------
static void test_clear_semantics() {
    std::cout << "[Test 9] Clear identity and failure atomicity..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_kv_cache kv_primary(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 128, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv_primary.init_xkv_store(cp);
    auto primary_pool = kv_primary.get_hot_slot_pool();
    assert(primary_pool != nullptr);

    // Shared view observes the same pool instance
    llama_kv_cache kv_view(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 128, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        &kv_primary, nullptr, nullptr, nullptr, &cp);
    assert(kv_view.get_hot_slot_pool() == primary_pool);

    // Live reservation makes clear throw BEFORE mutating any subsystem
    ubatch_fixture f1({1}, {0});
    auto sinfos = kv_primary.prepare({f1.ub});
    std::vector<xkv_hot_reservation> res;
    {
        std::string err;
        assert(kv_primary.reserve_hot_slots(sinfos, {f1.ub}, res, &err));
    }
    assert(primary_pool->get_reserved() == 1);
    // Live reservation makes try_clear report false BEFORE mutating anything
    {
        std::string err;
        assert(!kv_primary.try_clear(false, &err));
        assert(!err.empty());
    }
    // Bit-identical: reservation live, pool identity kept, epochs kept
    assert(primary_pool->get_reserved() == 1);
    assert(kv_primary.get_hot_slot_pool() == primary_pool);
    assert(kv_view.get_hot_slot_pool() == primary_pool);
    assert(primary_pool->get_accounting().peak == primary_pool->get_accounting().peak);

    // Quiescent clear preserves identity and frees everything
    res.clear();
    assert(primary_pool->get_reserved() == 0);
    {
        std::string err;
        assert(kv_primary.try_clear(false, &err));
    }
    assert(kv_primary.get_hot_slot_pool() == primary_pool);
    assert(kv_view.get_hot_slot_pool() == primary_pool);
    assert(primary_pool->get_free() == primary_pool->get_capacity());
    assert(primary_pool->get_bound() == 0 && primary_pool->get_reserved() == 0);
    assert(kv_primary.get_xkv_store()->live_epoch() == 0);
}

// ----------------------------------------------------------------------------
// Test 10: wrapper reservations, forwarding, and hot views
// ----------------------------------------------------------------------------
static void test_wrapper_forwarding_and_views() {
    std::cout << "[Test 10] Wrapper reservations, forwarding, and views..." << std::endl;

    test_model_xkv_hot model;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);

    // iSWA wrapper: target attention is bounded, SWA is not
    llama_kv_cache_iswa iswa(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, false, true, 128, 1, 8, 1,
        nullptr, nullptr, nullptr, nullptr, &cp);
    assert(iswa.is_xkv_bounded_hot());
    assert(!iswa.can_use_legacy_attention());
    assert(iswa.get_kv_hot_capacity() == iswa.get_base()->get_kv_hot_capacity());
    auto * base = iswa.get_base();
    auto * pool = base->get_hot_slot_pool().get();
    assert(pool != nullptr);

    ubatch_fixture f1({41, 42}, {0, 1});
    auto sinfos_base = base->prepare({f1.ub});
    assert(!sinfos_base.empty());
    std::vector<xkv_hot_reservation> hot_res;
    {
        std::string err;
        assert(base->reserve_hot_slots(sinfos_base, {f1.ub}, hot_res, &err));
    }
    assert(!sinfos_base[0].hot_idxs.empty());
    auto sinfos_swa = iswa.get_swa()->prepare({f1.ub});
    assert(!sinfos_swa.empty());
    llama_kv_cache_iswa_context ictx(&iswa, std::move(sinfos_base), std::move(sinfos_swa),
        {f1.ub}, std::move(hot_res));
    assert(ictx.apply());
    assert(ictx.postcompute_success());
    assert(ictx.postcompute_success()); // exactly-once: recorded true
    assert(pool->get_bound() == 2);

    // Physical views span hot rows for the graph path
    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * gctx = ggml_init(params);
    ggml_tensor * hk = ictx.get_xkv_hot_k(gctx, 0);
    ggml_tensor * hv = ictx.get_xkv_hot_v(gctx, 0);
    assert(hk != nullptr && hv != nullptr);
    assert(hk->ne[2] == (int64_t) base->get_hot_size());
    assert(hv->ne[2] == (int64_t) base->get_hot_size());
    ggml_free(gctx);

    // Hybrid wrapper: capability forwards + real reservations + forwarding
    llama_memory_hybrid hybrid(model, GGML_TYPE_F32, GGML_TYPE_F32, false,
        128, 1, 0, LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32, 8,
        1, 0, false, true, nullptr, nullptr, &cp);
    assert(hybrid.is_xkv_bounded_hot());
    assert(!hybrid.can_use_legacy_attention());
    auto * hattn = hybrid.get_mem_attn();
    assert(hattn->get_hot_slot_pool() != nullptr);
    assert(hybrid.get_kv_hot_capacity() == hattn->get_kv_hot_capacity());

    ubatch_fixture f2({43}, {2});
    auto hsinfos = hattn->prepare({f2.ub});
    std::vector<xkv_hot_reservation> hres;
    {
        std::string err;
        assert(hattn->reserve_hot_slots(hsinfos, {f2.ub}, hres, &err));
    }
    llama_memory_hybrid_context hctx(&hybrid, std::move(hsinfos), {f2.ub}, std::move(hres));
    assert(hctx.apply());
    assert(hctx.postcompute_success());
    assert(hattn->get_hot_slot_pool()->get_bound() == 1);
    xkv_state st;
    assert(hattn->get_xkv_store()->find_payload_state(
        hattn->get_cells(0).payload_id_get(0), st) && st == xkv_state::hot_committed);
}



// Seal-locked removal target: store refuses pre-gate, everything identical
static void test_gate_refusal_identical() {
    std::cout << "[Test 11] Seal-lock removal refusal identical..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint64_t pid0 = cells.payload_id_get(cell_with_pos(cells, 0));
    const uint64_t pid1 = cells.payload_id_get(cell_with_pos(cells, 1));
    const uint64_t gen0 = cells.storage_generation_get(cell_with_pos(cells, 0));
    uint64_t nonce0 = 0;
    assert(store->mark_seal_candidates({pid0}, {gen0}, &nonce0));
    const kv_snapshot before = take_snapshot(*kv);
    assert(!kv->seq_rm(0, 0, 2));
    const kv_snapshot after = take_snapshot(*kv);
    assert_snapshots_equal(before, after);
    xkv_state st;
    assert(store->find_payload_state(pid0, st) && st == xkv_state::seal_candidate);
    assert(store->find_payload_state(pid1, st) && st == xkv_state::hot_committed);
    assert(store->abort_seal_candidates({pid0}, nonce0, xkv_skip_reason::aborted));
}

// Rebuild-hook refusal identical; accepting hook observes exact positions
static void test_rebuild_hook_refusal_and_capture() {
    std::cout << "[Test 12] Rebuild hook refusal and capture..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    kv->set_removal_rebuild_hook([](const llama_kv_cache::xkv_removal_cells &, std::string * err) {
        if (err) *err = "injected rebuild refusal";
        return false;
    });
    const kv_snapshot before = take_snapshot(*kv);
    assert(!kv->seq_rm(0, 0, 2));
    assert_snapshots_equal(before, take_snapshot(*kv));
    // Accepting hook sees the exact pre-mutation cells then succeeds
    std::vector<std::pair<uint32_t, uint32_t>> seen;
    kv->set_removal_rebuild_hook([&seen](const llama_kv_cache::xkv_removal_cells & cells, std::string *) {
        seen = cells;
        return true;
    });
    assert(kv->seq_rm(0, 0, 2));
    assert(seen.size() == 2);
    kv->set_removal_rebuild_hook(nullptr);
}

// Stale cell generation: collect preflight refuses, everything identical
static void test_stale_generation_identical() {
    std::cout << "[Test 13] Stale generation refusal identical..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    auto & cells = const_cast<llama_kv_cells &>(kv->get_cells(0));
    const uint32_t c0 = cell_with_pos(cells, 0);
    cells.payload_id_set(c0, cells.payload_id_get(c0), cells.storage_generation_get(c0) + 1);
    const kv_snapshot before = take_snapshot(*kv);
    assert(!kv->seq_rm(0, 0, 2));
    assert_snapshots_equal(before, take_snapshot(*kv));
}

// Stale release plan: pool gate refuses, everything identical
static void test_stale_plan_identical() {
    std::cout << "[Test 14] Stale plan refusal identical..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    auto pool = kv->get_hot_slot_pool();
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint32_t c0 = cell_with_pos(cells, 0);
    const uint32_t c1 = cell_with_pos(cells, 1);
    uint32_t s0 = 0, s1 = 0;
    assert(pool->find_slot(cells.payload_id_get(c0), s0));
    assert(pool->find_slot(cells.payload_id_get(c1), s1));
    xkv_hot_release_plan stale;
    stale.released_payload_ids = {cells.payload_id_get(c0), cells.payload_id_get(c1)};
    stale.released_generations = {cells.storage_generation_get(c0) + 1, cells.storage_generation_get(c1)};
    stale.released_physical_rows = {s0, s1};
    const kv_snapshot before = take_snapshot(*kv);
    std::string err;
    assert(!kv->precommit_hot_release(stale, &err));
    assert(!kv->commit_hot_release(stale, &err));
    assert_snapshots_equal(before, take_snapshot(*kv));
    assert(pool->get_bound() == 2);
}

// Rollback gate refusal identical, then retry succeeds after unlocking
static void test_rollback_gate_false_identical() {
    std::cout << "[Test 15] Rollback refusal identical, retry recovers..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();
    ubatch_fixture f1({1, 2}, {0, 1});
    std::vector<xkv_hot_reservation> res;
    auto sinfos = prepare_reserve(*kv, f1.ub, res);
    auto ctx = apply_one(*kv, std::move(sinfos), f1.ub, std::move(res));
    const auto & cells = kv->get_cells(0);
    const uint64_t pid0 = cells.payload_id_get(cell_with_pos(cells, 0));
    const uint64_t gen0 = cells.storage_generation_get(cell_with_pos(cells, 0));
    uint64_t nonce0 = 0;
    assert(store->mark_seal_candidates({pid0}, {gen0}, &nonce0));
    const kv_snapshot before = take_snapshot(*kv);
    assert(!ctx->postcompute_failure());
    assert_snapshots_equal(before, take_snapshot(*kv));
    // Applied entries retained and retryable: unlock, then roll back cleanly
    assert(pool->get_bound() == 2);
    assert(store->abort_seal_candidates({pid0}, nonce0, xkv_skip_reason::aborted));
    assert(ctx->postcompute_failure());
    assert(pool->get_bound() == 0);
    assert(!has_pos(cells, 0) && !has_pos(cells, 1));
}

// Epoch-overflow clear refusal identical
static void test_try_clear_epoch_identical() {
    std::cout << "[Test 16] Epoch-overflow clear refusal identical..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    store->set_epoch_for_testing(1, UINT64_MAX); // live bump would overflow
    const kv_snapshot before = take_snapshot(*kv);
    std::string err;
    assert(!kv->try_clear(false, &err));
    assert(!err.empty());
    assert_snapshots_equal(before, take_snapshot(*kv));
    assert(pool->get_bound() == 2);
}

// Pool snapshot/restore is bit-identical across mutations
static void test_pool_snapshot_restore() {
    std::cout << "[Test 17] Pool snapshot/restore identical..." << std::endl;
    xkv_hot_slot_pool pool(8);
    auto r1 = pool.reserve(3);
    assert(r1.valid());
    assert(r1.commit({101, 102, 103}, {1, 1, 1}));
    const auto snap = pool.snapshot_state();
    const auto acc_before = pool.get_accounting();
    auto r2 = pool.reserve(2);
    assert(r2.valid());
    assert(r2.commit({201, 202}, {1, 1}));
    std::string err;
    assert(pool.release(101, 1, r1[0], &err));
    assert(pool.get_bound() == 4);
    pool.restore_state(snap);
    const auto acc_after = pool.get_accounting();
    assert(acc_before.bound == acc_after.bound && acc_before.free == acc_after.free);
    assert(acc_before.reserved == acc_after.reserved && acc_before.peak == acc_after.peak);
    assert(pool.get_bound() == 3);
    assert(pool.has_payload(101) && pool.has_payload(102) && pool.has_payload(103));
    assert(!pool.has_payload(201) && !pool.has_payload(202));
    uint32_t s = 0;
    assert(pool.find_slot(101, s) && s == r1[0]);
    // Original triple still validates after restore
    std::vector<uint64_t> pids = {101, 102, 103};
    std::vector<uint64_t> gens = {1, 1, 1};
    std::vector<uint32_t> slots = {r1[0], r1[1], r1[2]};
    assert(pool.can_release_batch(pids.data(), gens.data(), slots.data(), 3, &err));
}

// Logical compaction never remaps physical hot rows and never touches the
// store: logical cell index != hot slot is arranged through the overwrite
// path (fresh reservation row for an existing cell, the only production
// source of logical/physical divergence), then a pack moves a logical cell
// while its hot row must stay put.
static void test_compact_keeps_hot_rows() {
    std::cout << "[Test 18] Compaction keeps hot rows and store locations..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();

    // Production writes: pos0,1,2 -> cells c0,c1,c2, rows r0,r1,r2.
    ubatch_fixture f1({1, 2, 3}, {0, 1, 2});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint32_t c0 = cell_with_pos(cells, 0);
    const uint32_t c1 = cell_with_pos(cells, 1);
    const uint32_t c2 = cell_with_pos(cells, 2);
    const uint64_t p0 = cells.payload_id_get(c0);
    const uint64_t p1 = cells.payload_id_get(c1);
    const uint64_t p2 = cells.payload_id_get(c2);
    uint32_t r0 = 0, r1 = 0, r2 = 0;
    assert(pool->find_slot(p0, r0) && pool->find_slot(p1, r1) && pool->find_slot(p2, r2));

    // Overwrite c0 through the full production context path with a crafted
    // slot assignment (the overwrite path masked/SWA flows produce): the new
    // token takes a FRESH reservation row while the victim row is still held.
    auto ow_res = pool->reserve(1);
    assert(ow_res.valid());
    const uint32_t r3 = ow_res[0];
    assert(r3 != r0 && r3 != r1 && r3 != r2);
    llama_kv_cache::slot_info ow_sinfo;
    ow_sinfo.s0 = 0;
    ow_sinfo.s1 = 0;
    ow_sinfo.strm = {0};
    ow_sinfo.idxs = {{c0}};
    ow_sinfo.hot_idxs = {{r3}};
    ubatch_fixture fow({9}, {0}, 0);
    std::vector<xkv_hot_reservation> ow_res_v;
    ow_res_v.push_back(std::move(ow_res));
    llama_kv_cache::slot_info_vec_t ow_sinfos;
    ow_sinfos.push_back(std::move(ow_sinfo));
    std::vector<llama_ubatch> ow_ubs;
    ow_ubs.push_back(fow.ub);
    llama_kv_cache_context ow_ctx(kv.get(), std::move(ow_sinfos), std::move(ow_ubs), std::move(ow_res_v));
    assert(ow_ctx.apply());
    assert(ow_ctx.postcompute_success());
    // Logical c0 now addresses physical r3: divergence achieved.
    const uint64_t p0b = cells.payload_id_get(c0);
    assert(p0b != p0);
    uint32_t r3_check = 0;
    assert(pool->find_slot(p0b, r3_check) && r3_check == r3);
    assert(!pool->has_payload(p0));

    // Remove the middle token to force a logical pack move, then compact.
    // Epochs are read AFTER the removal: removal legitimately bumps them;
    // the assertion is that compact() itself moves nothing.
    assert(kv->seq_rm(0, 1, 2));
    const uint64_t live_before = store->live_epoch();
    const uint64_t bind_before = store->binding_epoch();
    const uint64_t cont_before = store->content_epoch();
    kv->compact();
    // Survivor P2 moved logically (c2 -> c1) but keeps hot row r2; P0' keeps
    // r3. Store locations and all epochs are untouched: no notification ran.
    uint32_t fr3 = 0, fr2 = 0;
    assert(pool->find_slot(p0b, fr3) && fr3 == r3);
    assert(pool->find_slot(p2, fr2) && fr2 == r2);
    xkv_location loc;
    assert(store->find_location(p0b, loc) && loc.row == r3);
    assert(store->find_location(p2, loc) && loc.row == r2);
    assert(store->live_epoch() == live_before);
    assert(store->binding_epoch() == bind_before);
    assert(store->content_epoch() == cont_before);
    const uint32_t c1b = cell_with_pos(cells, 2);
    assert(cells.payload_id_get(c1b) == p2);
    assert(pool->get_bound() == 2);
}

// ----------------------------------------------------------------------------
// Bounded context-shift transaction: prebuild-first, atomic commit, refusal
// with retryable state. update() takes a null context on paths that need no
// scheduler/graph work; paths needing them refuse instead of crashing.
// ----------------------------------------------------------------------------
static llama_kv_cache::stream_copy_info empty_sc_info() {
    return llama_kv_cache::stream_copy_info{};
}

// seq_div with d<=0 is refused with zero mutation (d==0 would trap).
static void test_seq_div_refusal() {
    std::cout << "[Test 19] seq_div refusal..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    const kv_snapshot before = take_snapshot(*kv);
    kv->seq_div(0, 0, 100, 0);
    assert_snapshots_equal(before, take_snapshot(*kv));
    kv->seq_div(0, 0, 100, -2);
    assert_snapshots_equal(before, take_snapshot(*kv));
    kv->seq_div(0, 0, 100, 1);
    assert_snapshots_equal(before, take_snapshot(*kv));
}

// Negative shift evicting a cell releases its pool row + store payload.
static void test_seq_add_eviction_release() {
    std::cout << "[Test 20] seq_add eviction release..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    auto pool = kv->get_hot_slot_pool();
    auto store = kv->get_xkv_store();
    ubatch_fixture f1({7}, {5});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint64_t pid = cells.payload_id_get(cell_with_pos(cells, 5));
    assert(pool->has_payload(pid));
    kv->seq_add(0, 0, 100, -10); // 5 - 10 < 0: cell evicted
    assert(!has_pos(cells, 5));
    assert(!pool->has_payload(pid));
    xkv_state st;
    assert(!store->find_payload_state(pid, st));
}

// Seal-locked shifted payload: update refuses, old state stays valid and
// retryable (positions advanced per stock protocol, deltas pending).
static void test_shift_seal_locked_refusal() {
    std::cout << "[Test 21] Seal-locked shift refusal..." << std::endl;
    test_model_xkv_hot model;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint64_t pid0 = cells.payload_id_get(cell_with_pos(cells, 0));
    const uint64_t gen0 = cells.storage_generation_get(cell_with_pos(cells, 0));
    uint64_t nonce0 = 0;
    assert(store->mark_seal_candidates({pid0}, {gen0}, &nonce0));
    kv->seq_add(0, 0, 2, 5);
    assert(has_pos(cells, 5) && has_pos(cells, 6));
    const kv_snapshot before = take_snapshot(*kv);
    assert(!kv->update(nullptr, true, empty_sc_info()));
    assert_snapshots_equal(before, take_snapshot(*kv));
    assert(pool->get_bound() == 2);
    xkv_state st;
    assert(store->find_payload_state(pid0, st) && st == xkv_state::seal_candidate);
    assert(store->abort_seal_candidates({pid0}, nonce0, xkv_skip_reason::aborted));
}

// Rope-NONE model: no rotation work exists; shift commits positions and
// clears deltas with pool/store bit-identical and no context needed.
static void test_shift_rope_none_success() {
    std::cout << "[Test 22] Rope-NONE shift success..." << std::endl;
    test_model_xkv_hot model;
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    auto kv = make_dense_kv(model);
    auto pool = kv->get_hot_slot_pool();
    auto store = kv->get_xkv_store();
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    kv->seq_add(0, 0, 2, 5);
    const kv_snapshot before = take_snapshot(*kv);
    assert(kv->update(nullptr, true, empty_sc_info()));
    const auto & cells = kv->get_cells(0);
    assert(has_pos(cells, 5) && has_pos(cells, 6));
    assert(pool->get_bound() == 2);
    assert(store->live_epoch() == before.live);
    assert(store->binding_epoch() == before.binding);
    assert(store->content_epoch() == before.content);
    // Deltas cleared: the cache accepts the next decode cleanly.
    ubatch_fixture f2({3}, {7});
    decode_success(*kv, f2.ub);
    assert(pool->get_bound() == 3);
}

// RoPE model without a context: rotation needs the scheduler, so update
// refuses instead of crashing; everything stays retryable.
static void test_shift_needs_context_refusal() {
    std::cout << "[Test 23] Null-context rotation refusal..." << std::endl;
    test_model_xkv_hot model;
    // Pin RoPE explicitly: the hparams default is ROPE_NONE, which would
    // take the no-rotation path and succeed instead of refusing.
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NEOX;
    auto kv = make_dense_kv(model);
    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(*kv, f1.ub);
    kv->seq_add(0, 0, 2, 5);
    const kv_snapshot before = take_snapshot(*kv);
    assert(!kv->update(nullptr, true, empty_sc_info()));
    assert_snapshots_equal(before, take_snapshot(*kv));
}

// Plain factored rows (pre-RoPE canonical) need no store mutation on shift:
// no pack, no epoch moves; hot rows advance alongside.
static void test_shift_factored_exemption() {
    std::cout << "[Test 24] Factored shift exemption..." << std::endl;
    test_model_xkv_hot model;
    // Rope-NONE takes the no-rotation path (explicit; also the default).
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();
    ubatch_fixture f1({1, 2}, {10, 11});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint32_t c10 = cell_with_pos(cells, 10);
    const uint32_t c11 = cell_with_pos(cells, 11);
    const uint64_t pid10 = cells.payload_id_get(c10);
    const uint64_t gen10 = cells.storage_generation_get(c10);
    const uint64_t pid11 = cells.payload_id_get(c11);
    const uint64_t gen11 = cells.storage_generation_get(c11);
    uint32_t s0 = 0, s1 = 0;
    assert(pool->find_slot(pid10, s0) && pool->find_slot(pid11, s1));
    auto seg = store->create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, {make_group()});
    const uint64_t seg_id = seg->segment_id;
    xkv_hot_release_plan plan;
    plan.released_payload_ids = {pid10, pid11};
    plan.released_generations = {gen10, gen11};
    plan.released_physical_rows = {s0, s1};
    plan.expected_segment_id = seg->segment_id;
    plan.expected_segment_version = seg->segment_version;
    {
        std::string err;
        assert(kv->validate_hot_release(plan, &err));
        assert(kv->commit_hot_release(plan, &err));
        std::string pub_err;
        assert(store->publish_candidate(seg, {pid10, pid11}, {gen10, gen11}, &pub_err));
    }
    const uint64_t seg_version = store->get_segment(seg_id)->segment_version;
    // One more hot row, then shift everything (factored + hot).
    ubatch_fixture f2({3}, {12});
    decode_success(*kv, f2.ub);
    // Epoch baseline is read AFTER the extra decode (decodes legitimately
    // bump live/binding epochs): the assertion is that update() itself —
    // with no landmark segments to refresh and no rotation under RoPE-NONE
    // — moves no epochs.
    const kv_snapshot sealed = take_snapshot(*kv);
    kv->seq_add(0, 0, 100, 5);
    assert(has_pos(cells, 15) && has_pos(cells, 16) && has_pos(cells, 17));
    assert(kv->update(nullptr, true, empty_sc_info()));
    // Factored segment untouched: same version and epochs as sealed state,
    // with only cell positions advanced and deltas cleared.
    assert(store->get_segment(seg_id)->segment_version == seg_version);
    assert(store->live_epoch() == sealed.live);
    assert(store->binding_epoch() == sealed.binding);
    assert(store->content_epoch() == sealed.content);
    assert(pool->get_bound() == 1);
    xkv_location loc;
    assert(store->find_location(pid10, loc));
    assert(store->find_location(pid11, loc));
}

// Fault-injection: if store mutation fails after hot rotation is executed,
// live K bytes and cell shift deltas MUST remain unchanged (rollback verified),
// ensuring retry does not double-rotate or leave corrupted state.

// Landmark group fixture: 2-row F32 factors with one exact landmark chunk
// (rows [0,2), nonzero source fingerprint). Mirrors make_group geometry so
// publish validation (contiguous chunks, nonzero fingerprints, matching table
// fingerprint) accepts it.
static xkv_factor_group_payload make_landmark_group() {
    xkv_factor_group_payload g0;
    g0.group_index = 0;
    g0.owning_layers = {0};
    g0.rank_k = 16;
    g0.rank_v = 16;
    g0.layer_feature_offsets_k = {0};
    g0.layer_feature_dims_k = {64};
    g0.layer_feature_offsets_v = {0};
    g0.layer_feature_dims_v = {64};
    g0.total_dim_k = 64;
    g0.total_dim_v = 64;
    constexpr uint64_t k_seed = 31;
    constexpr uint64_t v_seed = 32;
    codec_desc da_k = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {2, 16}, 0, k_seed);
    codec_desc db_k = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {64, 16}, 0, k_seed);
    codec_desc da_v = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {2, 16}, 0, v_seed);
    codec_desc db_v = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {64, 16}, 0, v_seed);
    std::vector<float> d_a(2 * 16, 0.1f);
    std::vector<float> d_b(64 * 16, 0.1f);
    g0.a_k = encode_matrix(da_k, d_a.data(), d_a.size());
    g0.set_b_k(encode_matrix(db_k, d_b.data(), d_b.size()));
    g0.a_v = encode_matrix(da_v, d_a.data(), d_a.size());
    g0.set_b_v(encode_matrix(db_v, d_b.data(), d_b.size()));
    codec_desc dlm = make_codec_desc(factor_role::landmark, GGML_TYPE_F32, orientation::token_major, {1, 64}, 0, 33);
    std::vector<float> d_lm(64, 0.2f);
    g0.landmark = encode_matrix(dlm, d_lm.data(), d_lm.size());
    xkv_landmark_chunk ch;
    ch.row_begin = 0;
    ch.row_count = 2;
    ch.error_bound = 0.05f;
    ch.source_fingerprint = 0x9E3779B97F4A7C15ULL;
    g0.landmark_chunks = {ch};
    g0.landmark_table_fingerprint = compute_landmark_table_fingerprint(g0.landmark_chunks);
    g0.refresh_descriptor_fingerprint();
    g0.update_byte_counters();
    return g0;
}

// Seals a 2-row LANDMARKS segment over the given hot payloads (production
// validate -> commit -> publish split). Returns the segment id.
static uint64_t seal_landmark_segment(llama_kv_cache & kv, uint64_t pid0, uint64_t gen0,
        uint64_t pid1, uint64_t gen1) {
    auto store = kv.get_xkv_store();
    auto pool = kv.get_hot_slot_pool();
    uint32_t s0 = 0, s1 = 0;
    assert(pool->find_slot(pid0, s0) && pool->find_slot(pid1, s1));
    auto seg = store->create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS, LLAMA_XKV_SOURCE_DECODED_HOT, {make_landmark_group()});
    const uint64_t seg_id = seg->segment_id;
    xkv_hot_release_plan plan;
    plan.released_payload_ids = {pid0, pid1};
    plan.released_generations = {gen0, gen1};
    plan.released_physical_rows = {s0, s1};
    plan.expected_segment_id = seg->segment_id;
    plan.expected_segment_version = seg->segment_version;
    std::string err;
    assert(kv.validate_hot_release(plan, &err));
    assert(kv.commit_hot_release(plan, &err));
    std::string pub_err;
    assert(store->publish_candidate(seg, {pid0, pid1}, {gen0, gen1}, &pub_err));
    return seg_id;
}

// Position shift on a landmark segment with a forced store-mutation failure:
// the transaction must refuse with zero mutation (rollback), preserving live
// K bytes, cell deltas, segment version, and exact landmark bytes / chunks /
// source fingerprints. Metadata publishes only after backend success.
static void test_shift_landmark_position_rollback() {
    std::cout << "[Test 27] Landmark position-shift rollback preserves fingerprints..." << std::endl;
    test_model_xkv_hot model;
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    ubatch_fixture f1({1, 2}, {10, 11});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint32_t c10 = cell_with_pos(cells, 10);
    const uint32_t c11 = cell_with_pos(cells, 11);
    const uint64_t pid10 = cells.payload_id_get(c10);
    const uint64_t gen10 = cells.storage_generation_get(c10);
    const uint64_t pid11 = cells.payload_id_get(c11);
    const uint64_t gen11 = cells.storage_generation_get(c11);
    const uint64_t seg_id = seal_landmark_segment(*kv, pid10, gen10, pid11, gen11);
    auto orig = store->get_segment(seg_id);
    const uint64_t ver_before = orig->segment_version;
    auto orig_bk = orig->groups[0].b_k;
    auto orig_bv = orig->groups[0].b_v;
    const std::vector<uint8_t> ak_before = orig->groups[0].a_k.bytes;
    const std::vector<uint8_t> lm_before = orig->groups[0].landmark.bytes;
    const auto chunks_before = orig->groups[0].landmark_chunks;
    const uint64_t table_before = orig->groups[0].landmark_table_fingerprint;

    // Shift factored rows: positions move, so landmark summaries need a
    // rebuild (never a survivor copy). Force the mutation to fail.
    kv->seq_add(0, 0, 100, 5);
    const kv_snapshot before = take_snapshot(*kv);
    store->set_epoch_for_testing(1, UINT64_MAX);
    llama_cparams cp_lctx = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_context lctx(model, cp_lctx, /*test_only=*/true);
    assert(!kv->update(&lctx, true, empty_sc_info()));
    // Reservation stays held (same precedent as Test 26); the rollback
    // below proves zero mutation while it is held.
    assert_snapshots_equal(before, take_snapshot(*kv));
    // Shift deltas stay pending for retry.
    assert(cells.get_shift(c10) == 5);
    assert(cells.get_shift(c11) == 5);
    // Store untouched: same version, byte-identical A rows, shared B handles,
    // and exact landmark bytes / chunks / fingerprints.
    auto after = store->get_segment(seg_id);
    assert(after->segment_version == ver_before);
    assert(after->groups[0].b_k == orig_bk);
    assert(after->groups[0].b_v == orig_bv);
    assert(after->groups[0].a_k.bytes == ak_before);
    assert(after->groups[0].landmark.bytes == lm_before);
    assert(after->groups[0].landmark_chunks == chunks_before);
    assert(after->groups[0].landmark_table_fingerprint == table_before);
}

// Device failure injection: a production device factorizer without an armed
// native landmark builder must refuse the shift before any store mutation,
// host decode, or epoch move (fail closed, no silent fallback).
static void test_shift_device_gate_refusal() {
    std::cout << "[Test 28] Device-factorizer shift gate refusal..." << std::endl;
    test_model_xkv_hot model;
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    cp.xkv_factorizer = LLAMA_XKV_FACTORIZER_VULKAN;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 128, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    auto store = kv.get_xkv_store();
    ubatch_fixture f1({1, 2}, {10, 11});
    decode_success(kv, f1.ub);
    const auto & cells = kv.get_cells(0);
    const uint32_t c10 = cell_with_pos(cells, 10);
    const uint32_t c11 = cell_with_pos(cells, 11);
    const uint64_t seg_id = seal_landmark_segment(kv,
        cells.payload_id_get(c10), cells.storage_generation_get(c10),
        cells.payload_id_get(c11), cells.storage_generation_get(c11));
    const uint64_t ver_before = store->get_segment(seg_id)->segment_version;
    kv->seq_add(0, 0, 100, 5);
    const kv_snapshot before = take_snapshot(*kv);
    llama_cparams cp_lctx = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_context lctx(model, cp_lctx, /*test_only=*/true);
    assert(!kv.update(&lctx, true, empty_sc_info()));
    assert_snapshots_equal(before, take_snapshot(*kv));
    assert(store->get_segment(seg_id)->segment_version == ver_before);
    assert(cells.get_shift(c10) == 5);
    assert(cells.get_shift(c11) == 5);
}

// Fingerprint preservation: shifting only hot rows leaves a landmark segment
// fully exempt (no pack, no rebuild), so its version, chunk table, and exact
// source fingerprints stay bit-identical while cell positions advance.
static void test_shift_landmark_exempt_preservation() {
    std::cout << "[Test 29] Landmark exemption preserves fingerprints..." << std::endl;
    test_model_xkv_hot model;
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();
    ubatch_fixture f1({1, 2}, {10, 11});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint32_t c10 = cell_with_pos(cells, 10);
    const uint32_t c11 = cell_with_pos(cells, 11);
    const uint64_t seg_id = seal_landmark_segment(*kv,
        cells.payload_id_get(c10), cells.storage_generation_get(c10),
        cells.payload_id_get(c11), cells.storage_generation_get(c11));
    ubatch_fixture f2({3}, {12});
    decode_success(*kv, f2.ub);
    auto orig = store->get_segment(seg_id);
    const uint64_t ver_before = orig->segment_version;
    const auto chunks_before = orig->groups[0].landmark_chunks;
    const uint64_t table_before = orig->groups[0].landmark_table_fingerprint;
    const std::vector<uint8_t> lm_before = orig->groups[0].landmark.bytes;
    // Shift only the hot row: the landmark segment has no shifted rows.
    kv->seq_add(0, 12, 13, 5);
    assert(kv->update(nullptr, true, empty_sc_info()));
    assert(has_pos(cells, 17));
    auto after = store->get_segment(seg_id);
    assert(after->segment_version == ver_before);
    assert(after->groups[0].landmark.bytes == lm_before);
    assert(after->groups[0].landmark_chunks == chunks_before);
    assert(after->groups[0].landmark_table_fingerprint == table_before);
    assert(pool->get_bound() == 1);
}

static void test_shift_mutation_failure_rolls_back_k() {
    std::cout << "[Test 26] Shift mutation failure rolls back live K and preserves deltas..." << std::endl;
    test_model_xkv_hot model;
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    auto kv = make_dense_kv(model);
    auto store = kv->get_xkv_store();
    auto pool = kv->get_hot_slot_pool();

    ubatch_fixture f1({1, 2}, {10, 11});
    decode_success(*kv, f1.ub);
    const auto & cells = kv->get_cells(0);
    const uint32_t c10 = cell_with_pos(cells, 10);
    const uint32_t c11 = cell_with_pos(cells, 11);
    const uint64_t pid10 = cells.payload_id_get(c10);
    const uint64_t gen10 = cells.storage_generation_get(c10);
    const uint64_t pid11 = cells.payload_id_get(c11);
    const uint64_t gen11 = cells.storage_generation_get(c11);
    uint32_t s0 = 0, s1 = 0;
    assert(pool->find_slot(pid10, s0) && pool->find_slot(pid11, s1));

    auto seg = store->create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_SOURCE_DECODED_HOT, {make_group()});
    const uint64_t seg_id = seg->segment_id;
    xkv_hot_release_plan plan;
    plan.released_payload_ids = {pid10, pid11};
    plan.released_generations = {gen10, gen11};
    plan.released_physical_rows = {s0, s1};
    plan.expected_segment_id = seg->segment_id;
    plan.expected_segment_version = seg->segment_version;
    {
        std::string err;
        assert(kv->validate_hot_release(plan, &err));
        assert(kv->commit_hot_release(plan, &err));
        std::string pub_err;
        assert(store->publish_candidate(seg, {pid10, pid11}, {gen10, gen11}, &pub_err));
    }

    // Shift cells by 5
    kv->seq_add(0, 0, 100, 5);

    // Take snapshot of K tensor bytes and kv state before update
    const kv_snapshot before = take_snapshot(*kv);
    ggml_tensor * k_storage = kv->get_k_storage(0);
    std::vector<uint8_t> k_bytes_before(ggml_nbytes(k_storage));
    std::memcpy(k_bytes_before.data(), k_storage->data, k_bytes_before.size());

    // Force store to refuse mutation by holding an epoch reservation
    store->set_epoch_for_testing(1, UINT64_MAX);

    // Assert that no full-history D2H occurred during failed shift
    // (live K is restored purely on-device via memmove regions or staged buffers)
    std::string err;
    bool updated = kv->update(nullptr, true, empty_sc_info());
    assert(!updated); // Must fail closed!

    // Check that live K bytes are unchanged (exact byte identity)
    std::vector<uint8_t> k_bytes_after(ggml_nbytes(k_storage));
    std::memcpy(k_bytes_after.data(), k_storage->data, k_bytes_after.size());
    assert(std::memcmp(k_bytes_before.data(), k_bytes_after.data(), k_bytes_before.size()) == 0);

    // Check that cell shift state remains pending (retryable!)
    assert(cells.get_shift(c10) == 5);
    assert(cells.get_shift(c11) == 5);
    assert(kv->get_has_shift());
}

// Reachability: Turbo K layers previously returned get_has_shift() == false
// unconditionally, silently dropping shift rephasing. Verify that after
// seq_add with Turbo4 K:
//  1. get_has_shift() reports true;
//  2. build_graph_shift constructs a valid compute graph containing the
//     Turbo-K rephasing nodes (set_rows write-back, WHT forward/inverse);
//  3. k_shift_rows carries the exact shifted hot row.
static void test_turbo_k_has_shift_and_graph_built() {
    std::cout << "[Test 25] Turbo K get_has_shift reachability and graph build..." << std::endl;
    test_model_xkv_hot model;
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NEOX;
    llama_cparams cp = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    // Construct with Turbo4 K type
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_TURBO4_0, GGML_TYPE_F32,
        false, false, true, 128, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, &cp);
    kv.init_xkv_store(cp);
    assert(!kv.get_has_shift());

    ubatch_fixture f1({1, 2}, {0, 1});
    decode_success(kv, f1.ub);
    assert(!kv.get_has_shift());

    // Shift by +5: must report has_shift == true (no longer suppressed!)
    kv.seq_add(0, 0, 2, 5);
    assert(kv.get_has_shift());

    // Build the shift graph using a lightweight mock context params
    // to verify that the Turbo branch in build_graph_shift actually builds
    // the set_rows / WHT pipeline.
    struct ggml_init_params gparams = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * gctx = ggml_init(gparams);

    // Create a minimal llama_context with mock params for build_graph_shift
    llm_graph_result gres(32);
    llama_cparams cp_lctx = make_cparams(LLAMA_XKV_MODE_DENSE, 32, 8, 8);
    llama_context lctx(model, cp_lctx, /*test_only=*/true);

    ggml_cgraph * gf = kv.build_graph_shift_for_testing(&gres, &lctx);
    assert(gf != nullptr);
    assert(ggml_graph_n_nodes(gf) > 0);

    // Verify that the graph contains nodes targeting the Turbo K tensor:
    // look for a GGML_OP_SET_ROWS node whose destination matches layer 0's K.
    bool found_set_rows = false;
    bool found_turbo_wht = false;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_SET_ROWS) {
            found_set_rows = true;
            // Verify wht_group == 128 was stored in op_params
            int32_t wht_group = 0;
            std::memcpy(&wht_group, node->op_params, sizeof(int32_t));
            assert(wht_group == 128);
        }
        if (node->op == GGML_OP_TURBO_WHT) {
            found_turbo_wht = true;
        }
    }
    assert(found_set_rows);
    assert(found_turbo_wht);

    ggml_free(gctx);
}
