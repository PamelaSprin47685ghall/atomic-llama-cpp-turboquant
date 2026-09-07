// Focused unit tests for XKV-SR admission accounting (§11):
// OFF mapping, tightest-domain reasons, hot_free authority, planner token
// bounds, reserve/store overflow, hybrid recurrent gating (sequence slots,
// not token slots), maintenance status names, and JSON/Prometheus fields.
// Hermetic: no model needed.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-memory.h"

#include "fit.h"
#include "llama.h"
#include "server-task.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Minimal fake memory: legacy getters only, XKV OFF semantics.
struct fake_mem : llama_memory_i {
    uint32_t kv_cap   = 0;
    uint32_t kv_used  = 0;
    uint32_t recr_cap = 0;
    uint32_t recr_used = 0;

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override { return nullptr; }
    llama_memory_context_ptr init_full() override { return nullptr; }
    llama_memory_context_ptr init_update(llama_context *, bool) override { return nullptr; }
    bool get_can_shift() const override { return true; }
    uint32_t get_kv_capacity() const override { return kv_cap; }
    uint32_t get_kv_used() const override { return kv_used; }
    uint32_t get_recurrent_capacity() const override { return recr_cap; }
    uint32_t get_recurrent_used() const override { return recr_used; }
    void clear(bool) override {}
    bool seq_rm(llama_seq_id, llama_pos, llama_pos) override { return true; }
    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}
    void seq_keep(llama_seq_id) override {}
    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override {}
    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override {}
    llama_pos seq_pos_min(llama_seq_id) const override { return -1; }
    llama_pos seq_pos_max(llama_seq_id) const override { return -1; }
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }
    void state_write(llama_io_write_i &, llama_seq_id = -1, llama_state_seq_flags = 0) const override {}
    void state_read(llama_io_read_i &, llama_seq_id seq_id = -1, llama_state_seq_flags = 0) override {}
};

static void test_off_mapping() {
    std::cout << "[off_mapping] starting..." << std::endl;
    fake_mem m;
    m.kv_cap = 1024;
    m.kv_used = 100;
    m.recr_cap = 8;
    m.recr_used = 3;

    llama_memory_admission_snapshot snap = {};
    CHECK(m.get_admission_snapshot(&snap));

    // Exact legacy-derived values, zero store/workspace, unconstrained planners.
    CHECK(snap.logical_capacity == 1024);
    CHECK(snap.logical_used == 100);
    CHECK(snap.hot_capacity == 1024); // OFF: hot backing == logical
    CHECK(snap.hot_used == 100);
    CHECK(snap.hot_free == 924);
    CHECK(snap.hot_reserved == 0);
    CHECK(snap.factor_live_bytes == 0);
    CHECK(snap.factor_reserved_bytes == 0);
    CHECK(snap.factor_budget_bytes == 0);
    CHECK(snap.factor_free_bytes == 0);
    CHECK(snap.factor_safe_tokens == UINT32_MAX);
    CHECK(snap.workspace_live_bytes == 0);
    CHECK(snap.workspace_peak_bytes == 0);
    CHECK(snap.workspace_budget_bytes == 0);
    CHECK(snap.workspace_free_bytes == 0);
    CHECK(snap.workspace_safe_tokens == UINT32_MAX);
    CHECK(snap.recurrent_capacity == 8);
    CHECK(snap.recurrent_used == 3);
    // Token bound never mixes recurrent sequence slots: min(924, 924).
    CHECK(snap.safe_next_ubatch == 924);
    // Tightest domain wins even when positive (tie -> logical first).
    CHECK(snap.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
    CHECK(!snap.recurrent_blocks_new_seq);

    // Empty memory exposes no domain.
    fake_mem e;
    llama_memory_admission_snapshot esnap = {};
    CHECK(!e.get_admission_snapshot(&esnap));
    CHECK(!m.get_admission_snapshot(nullptr));

    // Full logical cache: zero bound with a real reason (never clamp to 1).
    fake_mem f;
    f.kv_cap = 64;
    f.kv_used = 64;
    llama_memory_admission_snapshot fsnap = {};
    CHECK(f.get_admission_snapshot(&fsnap));
    CHECK(fsnap.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
    CHECK(fsnap.safe_next_ubatch == 0);
    std::cout << "[off_mapping] done." << std::endl;
}

static llama_memory_admission_snapshot make_free_snapshot() {
    llama_memory_admission_snapshot s = {};
    s.logical_capacity = 1000;
    s.logical_used = 100;
    s.hot_capacity = 1000;
    s.hot_used = 100;
    s.hot_free = 900;
    s.factor_safe_tokens = UINT32_MAX;
    s.workspace_safe_tokens = UINT32_MAX;
    s.recurrent_capacity = 16;
    s.recurrent_used = 4;
    return s;
}

static void test_tightest_reason() {
    std::cout << "[tightest_reason] starting..." << std::endl;

    CHECK(std::string(llama_memory_limit_reason_name(LLAMA_MEMORY_LIMIT_NONE)) == "none");
    CHECK(std::string(llama_memory_limit_reason_name(LLAMA_MEMORY_LIMIT_LOGICAL_CELLS)) == "logical_cells");
    CHECK(std::string(llama_memory_limit_reason_name(LLAMA_MEMORY_LIMIT_HOT_SLOTS)) == "hot_slots");
    CHECK(std::string(llama_memory_limit_reason_name(LLAMA_MEMORY_LIMIT_FACTOR_STORE)) == "factor_store");
    CHECK(std::string(llama_memory_limit_reason_name(LLAMA_MEMORY_LIMIT_WORKSPACE)) == "workspace");

    // Reason names the minimum bound while still positive (tie -> logical).
    llama_memory_admission_snapshot s = make_free_snapshot();
    llama_memory_admission_finalize(s);
    CHECK(s.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
    CHECK(s.safe_next_ubatch == 900);
    CHECK(!s.recurrent_blocks_new_seq);

    // Provided hot_free is authoritative (reserved never over-admitted).
    s = make_free_snapshot();
    s.hot_free = 100;
    s.hot_reserved = 50;
    llama_memory_admission_finalize(s);
    CHECK(s.limit_reason == LLAMA_MEMORY_LIMIT_HOT_SLOTS);
    CHECK(s.safe_next_ubatch == 100);

    // Planner token bounds limit even with nonzero free bytes.
    s = make_free_snapshot();
    s.factor_live_bytes = 100;
    s.factor_budget_bytes = 1000;
    s.factor_free_bytes = 900;
    s.factor_safe_tokens = 50;
    llama_memory_admission_finalize(s);
    CHECK(s.limit_reason == LLAMA_MEMORY_LIMIT_FACTOR_STORE);
    CHECK(s.safe_next_ubatch == 50);

    s = make_free_snapshot();
    s.workspace_safe_tokens = 30;
    llama_memory_admission_finalize(s);
    CHECK(s.limit_reason == LLAMA_MEMORY_LIMIT_WORKSPACE);
    CHECK(s.safe_next_ubatch == 30);

    // Genuine exhaustion: zero bound, real reason.
    s = make_free_snapshot();
    s.logical_used = 1000;
    llama_memory_admission_finalize(s);
    CHECK(s.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
    CHECK(s.safe_next_ubatch == 0);

    // Unknown snapshot: no finite domain -> NONE.
    llama_memory_admission_snapshot u = {};
    u.factor_safe_tokens = UINT32_MAX;
    u.workspace_safe_tokens = UINT32_MAX;
    llama_memory_admission_finalize(u);
    CHECK(u.limit_reason == LLAMA_MEMORY_LIMIT_NONE);
    CHECK(u.safe_next_ubatch == 0);
    CHECK(!u.recurrent_blocks_new_seq);

    std::cout << "[tightest_reason] done." << std::endl;
}

static void test_recurrent_gating() {
    std::cout << "[recurrent_gating] starting..." << std::endl;

    // Full recurrent slots never enter the token minimum: a continuation
    // batch on existing sequences proceeds on the token bound.
    llama_memory_admission_snapshot s = make_free_snapshot();
    s.recurrent_used = 16;
    llama_memory_admission_finalize(s);
    CHECK(s.safe_next_ubatch == 900);
    CHECK(s.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
    CHECK(s.recurrent_blocks_new_seq);

    // Hybrid combine preserves the attention token bound/reason and only
    // recomputes the separate new-sequence gate.
    const llama_memory_admission_snapshot attn = make_free_snapshot();
    const llama_memory_admission_snapshot bound = llama_memory_admission_combine(attn, 2, 2);
    CHECK(bound.logical_capacity == 1000);
    CHECK(bound.logical_used == 100);
    CHECK(bound.hot_free == 900);
    CHECK(bound.factor_safe_tokens == UINT32_MAX);
    CHECK(bound.safe_next_ubatch == 900);
    CHECK(bound.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
    CHECK(bound.recurrent_capacity == 2);
    CHECK(bound.recurrent_used == 2);
    CHECK(bound.recurrent_blocks_new_seq);

    const llama_memory_admission_snapshot ok = llama_memory_admission_combine(attn, 2, 0);
    CHECK(ok.safe_next_ubatch == 900);
    CHECK(ok.limit_reason == LLAMA_MEMORY_LIMIT_LOGICAL_CELLS);
    CHECK(!ok.recurrent_blocks_new_seq);

    // Fake hybrid: full recurrent + existing-seq token batch proceeds.
    fake_mem h;
    h.kv_cap = 512;
    h.kv_used = 12;
    h.recr_cap = 1;
    h.recr_used = 1;
    llama_memory_admission_snapshot hsnap = {};
    CHECK(h.get_admission_snapshot(&hsnap));
    CHECK(hsnap.safe_next_ubatch == 500);
    CHECK(hsnap.recurrent_blocks_new_seq);

    std::cout << "[recurrent_gating] done." << std::endl;
}

static void test_reserve_overflow() {
    std::cout << "[reserve_overflow] starting..." << std::endl;

    // OFF yields zeros.
    {
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_OFF;
        common_xkv_fit_reserve r = {};
        CHECK(common_xkv_fit_reserve_bytes(&c, 0, &r));
        CHECK(r.workspace_bytes == 0);
        CHECK(r.decode_cache_bytes == 0);
        CHECK(r.factor_scratch_bytes == 0);
        CHECK(r.total_bytes == 0);
    }
    CHECK(!common_xkv_fit_reserve_bytes(nullptr, 0, nullptr));

    // Workspace-once model: total is workspace (once); decode is a
    // sub-budget inside it and scratch must fit the remainder.
    {
        llama_context_params c = {};
        const llama_xkv_params d = llama_xkv_default_params();
        c.xkv_mode             = LLAMA_XKV_MODE_SR;
        c.xkv_workspace_mib    = d.workspace_mib;     // 256
        c.xkv_decode_cache_mib = d.decode_cache_mib; // 64
        c.xkv_rank_k           = d.rank_k;            // 384
        c.xkv_rank_v           = d.rank_v;            // 576
        c.xkv_segment_tokens   = d.segment_tokens;    // 4096
        common_xkv_fit_reserve r = {};
        const uint64_t scratch = 10ull * 1024ull * 1024ull; // synthetic exact scratch
        CHECK(common_xkv_fit_reserve_bytes(&c, scratch, &r));
        CHECK(r.workspace_bytes == 256ull * 1024ull * 1024ull);
        CHECK(r.decode_cache_bytes == 64ull * 1024ull * 1024ull);
        CHECK(r.factor_scratch_bytes == scratch);
        CHECK(r.total_bytes == r.workspace_bytes); // once, never the sum
        // Scratch breaching the remainder fails closed.
        common_xkv_fit_reserve bad = {};
        CHECK(!common_xkv_fit_reserve_bytes(&c, 200ull * 1024ull * 1024ull, &bad));
        CHECK(bad.total_bytes == UINT64_MAX);
        // Decode alone breaching workspace fails closed.
        llama_context_params c2 = c;
        c2.xkv_decode_cache_mib = 512;
        common_xkv_fit_reserve bad2 = {};
        CHECK(!common_xkv_fit_reserve_bytes(&c2, scratch, &bad2));
    }

    // Overflow fails closed.
    {
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_SR;
        c.xkv_workspace_mib = UINT32_MAX;
        c.xkv_decode_cache_mib = UINT32_MAX;
        c.xkv_rank_k = UINT32_MAX;
        c.xkv_rank_v = UINT32_MAX;
        c.xkv_segment_tokens = UINT32_MAX;
        common_xkv_fit_reserve r = {};
        CHECK(!common_xkv_fit_reserve_bytes(&c, 0, &r));
        CHECK(r.total_bytes == UINT64_MAX);
    }
    std::cout << "[reserve_overflow] done." << std::endl;
}

static void test_store_budget() {
    std::cout << "[store_budget] starting..." << std::endl;

    // OFF and SHADOW-reference keep no persistent store: separate from peaks.
    {
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_OFF;
        c.xkv_store_mib = 512;
        CHECK(common_xkv_store_budget_bytes(&c) == 0);
    }
    {
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_SHADOW;
        // SHADOW publishes no segments for ANY profile.
        c.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
        c.xkv_store_mib = 512;
        CHECK(common_xkv_store_budget_bytes(&c) == 0);
    }
    {
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_SR;
        CHECK(common_xkv_store_budget_bytes(&c) == 0); // 0 = auto-derive
        CHECK(common_xkv_store_budget_bytes(nullptr) == 0);
    }
    // Explicit budget converts with checked arithmetic.
    {
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_SR;
        c.xkv_store_mib = 512;
        CHECK(common_xkv_store_budget_bytes(&c) == 512ull * 1024ull * 1024ull);
    }
    std::cout << "[store_budget] done." << std::endl;
}

static void test_maintenance_status() {
    std::cout << "[maintenance_status] starting..." << std::endl;
    CHECK(std::string(llama_memory_maintenance_status_name(LLAMA_MEMORY_MAINTENANCE_PROGRESS)) == "progress");
    CHECK(std::string(llama_memory_maintenance_status_name(LLAMA_MEMORY_MAINTENANCE_NO_ACTION)) == "no_action");
    CHECK(std::string(llama_memory_maintenance_status_name(LLAMA_MEMORY_MAINTENANCE_FLOOR_EXHAUSTED)) == "floor_exhausted");
    CHECK(std::string(llama_memory_maintenance_status_name(LLAMA_MEMORY_MAINTENANCE_RETRY_STALE)) == "retry_stale");
    CHECK(std::string(llama_memory_maintenance_status_name(LLAMA_MEMORY_MAINTENANCE_ERROR)) == "error");
    std::cout << "[maintenance_status] done." << std::endl;
}

static void test_partition_budget() {
    std::cout << "[partition_budget] starting..." << std::endl;
    // Exact-sum split across uneven multi-GPU placement.
    {
        const uint64_t w[] = { 100, 300, 600 };
        uint64_t s[] = { 0, 0, 0 };
        CHECK(common_xkv_partition_budget(1000, w, s, 3));
        CHECK(s[0] == 100 && s[1] == 300 && s[2] == 600);
    }
    // Remainder lands on the largest weight; shares still sum exactly.
    {
        const uint64_t w[] = { 1, 1, 1 };
        uint64_t s[] = { 0, 0, 0 };
        CHECK(common_xkv_partition_budget(100, w, s, 3));
        CHECK(s[0] + s[1] + s[2] == 100);
        CHECK(s[0] == 34 && s[1] == 33 && s[2] == 33);
    }
    // Zero-weight devices get nothing; CPU-only (all zero) splits equally.
    {
        const uint64_t w[] = { 0, 500, 0 };
        uint64_t s[] = { 9, 9, 9 };
        CHECK(common_xkv_partition_budget(1000, w, s, 3));
        CHECK(s[0] == 0 && s[1] == 1000 && s[2] == 0);
    }
    {
        const uint64_t w[] = { 0, 0 };
        uint64_t s[] = { 0, 0 };
        CHECK(common_xkv_partition_budget(7, w, s, 2));
        CHECK(s[0] + s[1] == 7);
    }
    // Degenerate inputs.
    CHECK(common_xkv_partition_budget(100, nullptr, nullptr, 0));
    {
        uint64_t s = 0;
        CHECK(!common_xkv_partition_budget(100, nullptr, &s, 1));
        CHECK(!common_xkv_partition_budget(100, &s, nullptr, 1));
    }
    {
        const uint64_t w[] = { 5 };
        uint64_t s[] = { 9 };
        CHECK(common_xkv_partition_budget(0, w, s, 1));
        CHECK(s[0] == 0);
    }
    // Single huge total still partitions exactly (no N× overreserve).
    {
        const uint64_t w[] = { 3, 1 };
        uint64_t s[] = { 0, 0 };
        const uint64_t total = 10ull * 1024ull * 1024ull * 1024ull;
        CHECK(common_xkv_partition_budget(total, w, s, 2));
        CHECK(s[0] + s[1] == total);
    }
    std::cout << "[partition_budget] done." << std::endl;
}

static void test_scratch_estimator() {
    std::cout << "[scratch_estimator] starting..." << std::endl;
    // Exact estimator on reference dims: succeeds and covers at least the
    // old toy rank*segment bound (toy omitted B/shadow/landmarks/capture).
    {
        uint64_t bytes = 0;
        CHECK(common_xkv_scratch_for_group(4096, 8, 4096, 4096, 384, 576, GGML_TYPE_Q8_0, &bytes));
        CHECK(bytes >= (384ull + 576ull) * 4096ull * sizeof(float));
        uint64_t again = 0;
        CHECK(common_xkv_scratch_for_group(4096, 8, 4096, 4096, 384, 576, GGML_TYPE_Q8_0, &again));
        CHECK(again == bytes); // deterministic
    }
    // Larger groups cost more; smaller ranks cost less.
    {
        uint64_t big = 0;
        uint64_t small = 0;
        CHECK(common_xkv_scratch_for_group(4096, 8, 8192, 8192, 384, 576, GGML_TYPE_F32, &big));
        CHECK(common_xkv_scratch_for_group(4096, 8, 1024, 1024, 64, 64, GGML_TYPE_F32, &small));
        CHECK(big > small);
    }
    // Rejections: bad dims, bad landmark type, null out, overflow.
    {
        uint64_t bytes = 0;
        CHECK(!common_xkv_scratch_for_group(0, 8, 4096, 4096, 384, 576, GGML_TYPE_Q8_0, &bytes));
        CHECK(!common_xkv_scratch_for_group(4096, 8, 0, 4096, 384, 576, GGML_TYPE_Q8_0, &bytes));
        CHECK(!common_xkv_scratch_for_group(4096, 8, 4096, 4096, 0, 576, GGML_TYPE_Q8_0, &bytes));
        CHECK(!common_xkv_scratch_for_group(4096, 8, 4096, 4096, 384, 576, GGML_TYPE_Q4_0, &bytes));
        CHECK(!common_xkv_scratch_for_group(4096, 8, 4096, 4096, 384, 576, GGML_TYPE_Q8_0, nullptr));
        CHECK(!common_xkv_scratch_for_group(UINT32_MAX, 1, UINT64_MAX / 2, UINT64_MAX / 2, 384, 576, GGML_TYPE_F32, &bytes));
    }
    // Model-backed estimator fails closed without a model file.
    {
        uint64_t bytes = 0;
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_SR;
        CHECK(!common_xkv_scratch_bytes("/nonexistent-model.gguf", nullptr, &c, &bytes));
        CHECK(!common_xkv_scratch_bytes(nullptr, nullptr, nullptr, nullptr));
    }
    std::cout << "[scratch_estimator] done." << std::endl;
}

static void test_store_mib_math() {
    std::cout << "[store_mib_math] starting..." << std::endl;
    constexpr uint64_t MiB = 1024ull * 1024ull;
    CHECK(common_xkv_store_mib_for_bytes(0, 0.5) == 0);
    CHECK(common_xkv_store_mib_for_bytes(100 * MiB, 0.5) == 50);
    CHECK(common_xkv_store_mib_for_bytes(100 * MiB + 1, 0.5) == 51); // rounds up
    CHECK(common_xkv_store_mib_for_bytes(100 * MiB, 0.0) == 100);
    CHECK(common_xkv_store_mib_for_bytes(100 * MiB, 1.0) == 100); // degenerate saving
    CHECK(common_xkv_store_mib_for_bytes(100 * MiB, -1.0) == 100);
    CHECK(common_xkv_store_mib_for_bytes(1, 0.99) == 1); // minimum 1 MiB
    CHECK(common_xkv_store_mib_for_bytes(UINT64_MAX, 0.0) == 0); // overflow
    // COW overlap: base plus one capped segment share, never less than base.
    const uint32_t base = common_xkv_store_mib_for_bytes(100 * MiB, 0.5);
    const uint32_t with_ov = common_xkv_store_mib_with_overlap(100 * MiB, 0.5, 4096, 65536);
    CHECK(with_ov >= base);
    CHECK(with_ov <= base + base); // segment share capped at base
    CHECK(common_xkv_store_mib_with_overlap(0, 0.5, 4096, 65536) == 0);
    CHECK(common_xkv_store_mib_with_overlap(100 * MiB, 0.5, 0, 65536) == base); // no seg info
    CHECK(common_xkv_store_mib_with_overlap(100 * MiB, 0.5, 4096, 0) == base);
    std::cout << "[store_mib_math] done." << std::endl;
}

static void test_runtime_snapshot_default() {
    std::cout << "[runtime_snapshot_default] starting..." << std::endl;
    // Unbound memory reports false: server shows empty/not_evaluated and
    // never copies request values into effective fields.
    fake_mem m;
    m.kv_cap = 64;
    llama_memory_xkv_runtime_snapshot rsnap = {};
    rsnap.armed = true; // must be zeroed on failure
    CHECK(!m.get_xkv_runtime_snapshot(&rsnap));
    CHECK(!rsnap.armed);
    CHECK(!m.get_xkv_runtime_snapshot(nullptr));
    std::cout << "[runtime_snapshot_default] done." << std::endl;
}

static void test_runtime_abi() {
    std::cout << "[runtime_abi] starting..." << std::endl;
    // Self-description: filler-sized, version 1, observed mode carried
    // (profile alone cannot tell SHADOW/DENSE/SR).
    llama_memory_xkv_runtime_snapshot rs = {};
    CHECK(rs.struct_size == 0 && rs.version == 0); // zero-init contract
    CHECK(!rs.armed);
    // Public skip codes mirror the store order; server names them.
    CHECK(LLAMA_MEMORY_XKV_SKIP_REASON_COUNT == 10);
    CHECK(std::string(llama_memory_xkv_skip_reason_name(LLAMA_MEMORY_XKV_SKIP_NONE)) == "none");
    CHECK(std::string(llama_memory_xkv_skip_reason_name(LLAMA_MEMORY_XKV_SKIP_NO_SAVING)) == "no_saving");
    CHECK(std::string(llama_memory_xkv_skip_reason_name(LLAMA_MEMORY_XKV_SKIP_LANDMARK_REQUIRED)) == "landmark_required");
    CHECK(std::string(llama_memory_xkv_skip_reason_name(LLAMA_MEMORY_XKV_SKIP_REASON_COUNT)) == "unknown");
    std::cout << "[runtime_abi] done." << std::endl;
}

static void test_planner_policy() {
    std::cout << "[planner_policy] starting..." << std::endl;
    using M = server_task_result_metrics;
    // Deficit-derived bound: ceil(deficit/seg) + 4 domains, never fixed.
    // A legal deficit needing >3 segments is allowed (old fixed-3 wrong).
    CHECK(M::xkv_maintain_max_attempts(false, true, 0, 0, 100, 8) == 0); // OFF
    CHECK(M::xkv_maintain_max_attempts(true, true, 900, 1000, 100, 8) == 0); // covered
    CHECK(M::xkv_maintain_max_attempts(true, true, 0, 1000, 1000, 8) == 125 + 4);
    CHECK(M::xkv_maintain_max_attempts(true, true, 0, 64, 1000, 8) == 125 + 4);
    CHECK(M::xkv_maintain_max_attempts(true, false, 0, 1000, 100, 8) == 125 + 4); // unknown->cap/seg
    CHECK(M::xkv_maintain_max_attempts(true, false, 0, 0, 100, 8) == 4);
    CHECK(M::xkv_maintain_max_attempts(true, true, 0, 1000, 100, 0) == 4); // seg 0
    using M2 = server_task_result_metrics;
    // Fill-first: hot covering required skips sealing.
    CHECK(!M2::xkv_should_maintain(false, false, true, 0, 100, 0, 10)); // OFF
    CHECK(!M2::xkv_should_maintain(true, false, true, 100, 100, 0, 10)); // covered
    CHECK(!M2::xkv_should_maintain(true, false, true, 200, 100, 0, 10)); // surplus
    CHECK(M2::xkv_should_maintain(true, false, true, 99, 100, 0, 10)); // short
    CHECK(M2::xkv_should_maintain(true, false, false, 0, 0, 0, 10)); // unknown
    CHECK(M2::xkv_should_maintain(true, true, true, 1000, 10, 0, 10)); // SHADOW eval
    CHECK(!M2::xkv_should_maintain(true, true, true, 0, 100, 10, 10)); // bound out
    CHECK(M2::xkv_should_maintain(true, true, true, 0, 100, 9, 200)); // large bound ok
    std::cout << "[planner_policy] done." << std::endl;
}

static void test_metrics_fields() {
    std::cout << "[metrics_fields] starting..." << std::endl;

    // OFF: empty, no JSON block content, no Prometheus output.
    {
        server_xkv_metrics xm;
        CHECK(xm.empty());
        std::ostringstream os;
        xm.to_prometheus(os);
        CHECK(os.str().empty());
    }

    server_xkv_metrics xm;
    xm.has_admission = true;
    xm.safe_next_ubatch = 7;
    xm.limit_reason = "hot_slots";
    xm.logical_capacity = 1000;
    xm.logical_used = 993;
    xm.hot_capacity = 64;
    xm.hot_used = 57;
    xm.hot_free = 7;
    xm.hot_reserved = 2;
    xm.factor_safe_tokens = 50;
    xm.workspace_peak_bytes = 1ull << 20;
    xm.workspace_budget_bytes = 256ull << 20;
    xm.workspace_free_bytes = 255ull << 20;
    xm.workspace_safe_tokens = 30;
    xm.recurrent_capacity = 4;
    xm.recurrent_used = 1;
    xm.requested_profile = "tq-factors-landmarks";
    xm.effective_profile = "tq-factors";
    xm.source = "decoded-hot";
    xm.codec_fingerprint = 0x1234;
    xm.backend_fingerprint = 0x5678;
    xm.source_fingerprint = 0x9abc;
    xm.profile_fingerprint = 0xdef0;
    xm.rank_k = 384;
    xm.rank_v = 576;
    xm.factor_streams = 4;
    xm.actual_bytes = 100;
    xm.nominal_bytes = 400;
    xm.hot_bytes = 10;
    xm.flat_bytes = 20;
    xm.factor_ak_bytes = 1;
    xm.factor_bk_bytes = 2;
    xm.factor_av_bytes = 3;
    xm.factor_bv_bytes = 4;
    xm.factor_payload_bytes = 5;
    xm.factor_metadata_bytes = 6;
    xm.factor_padding_bytes = 7;
    xm.landmark_payload_bytes = 8;
    xm.landmark_metadata_bytes = 9;
    xm.landmark_exception_bytes = 10;
    xm.index_bytes = 11;
    xm.codec_shared_bytes = 12;
    xm.decode_tile_cache_bytes = 13;
    xm.capture_bytes = 14;
    xm.candidate_bytes = 15;
    xm.snapshot_pinned_bytes = 16;
    xm.allocator_live_bytes = 17;
    xm.allocator_reserved_bytes = 18;
    xm.device_peak_bytes = 19;
    xm.host_peak_bytes = 20;
    xm.unique_payloads = 21;
    xm.aliased_payloads = 22;
    xm.baseline_same_rows_bytes = 23;
    xm.covered_compressed_bytes = 18;
    xm.factored_baseline_byte_coverage = 0.5;
    xm.factor_quant_ratio = 4.0;
    xm.net_extra_compression_ratio = 1.25;
    xm.seal_seconds = 0.1;
    xm.factor_quant_seconds = 0.2;
    xm.landmark_quant_seconds = 0.3;
    xm.select_seconds = 0.4;
    xm.refine_seconds = 0.5;
    xm.reconstruct_seconds = 0.6;
    xm.read_seconds = 0.7;
    xm.pack_seconds = 0.8;
    xm.segments_sealed = 9;
    xm.segments_skipped_by_reason["no_saving"] = 3;
    xm.segments_skipped_by_reason["codec_error"] = 1;
    xm.sr_selected_rows = 30;
    xm.sr_fragments = 5;
    xm.effective_chunk_size = 8;
    xm.landmark_refine_rows = 40;
    xm.landmark_refine_cap_hits = 2;
    xm.spec_stale_total = 6;
    xm.transaction_abort_total = 7;
    xm.synchronize_total = 11;
    xm.compression_goal_met = true;
    xm.throttle_total = 2;
    xm.throttle_reason = "hot_slots";
    xm.observed = true;
    xm.ratios_evaluated = true;
    xm.seal_timers_evaluated = true;
    xm.quant_timers_evaluated = true;
    xm.net_extra_compression_ratio_reserved = 1.20;
    xm.requested_mode = "sr";
    xm.effective_mode = "sr";
    xm.requested_a_k = "turbo4_0";
    xm.effective_a_k = "turbo4_0";
    xm.requested_b_k = "turbo4_0";
    xm.effective_b_k = "turbo4_0";
    xm.requested_a_v = "turbo2_0";
    xm.effective_a_v = "turbo2_0";
    xm.requested_b_v = "turbo2_0";
    xm.effective_b_v = "turbo2_0";
    xm.requested_landmark = "q8_0";
    xm.effective_landmark = "q8_0";
    xm.requested_factorizer = "cpu-reference";
    xm.effective_factorizer = "cpu-reference";
    xm.requested_balance = "upstream";
    xm.effective_balance = "upstream";
    xm.requested_seed = "0x584b565352303031";
    xm.effective_seed = "0x584b565352303031";
    xm.model_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    xm.tri_calibration_fingerprint = 0x1234567890ABCDEFULL;
    xm.tri_calibration_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    xm.tri_ratio_str = "3/32";
    xm.tri_recent_window = 128;
    xm.tri_scorer_valid = true;
    xm.compression_goal_evaluated = true;
    xm.graph_timings_evaluated = true;
    xm.pack_timer_evaluated = true;
    xm.sr_counters_evaluated = true;
    xm.spec_counters_evaluated = true;
    CHECK(!xm.empty());

    const json j = xm.to_json();
    for (const char * key : {
            "xkv_admission", "xkv_requested_profile", "xkv_effective_profile",
            "xkv_requested_mode", "xkv_effective_mode",
            "xkv_requested_a_k", "xkv_effective_a_k",
            "xkv_requested_b_k", "xkv_effective_b_k",
            "xkv_requested_a_v", "xkv_effective_a_v",
            "xkv_requested_b_v", "xkv_effective_b_v",
            "xkv_requested_landmark", "xkv_effective_landmark",
            "xkv_requested_factorizer", "xkv_effective_factorizer",
            "xkv_requested_balance", "xkv_effective_balance",
            "xkv_requested_seed", "xkv_effective_seed",
            "xkv_model_sha256",
            "tri_calibration_fingerprint", "tri_calibration_sha256", "tri_ratio", "tri_recent_window", "tri_scorer_valid",
            "xkv_compression_goal_evaluated",
            "xkv_source",
            "xkv_codec_fingerprint", "xkv_backend_fingerprint", "xkv_source_fingerprint",
            "xkv_profile_fingerprint", "xkv_rank_k", "xkv_rank_v", "xkv_factor_streams",
            "xkv_actual_bytes", "xkv_nominal_bytes", "xkv_hot_bytes", "xkv_flat_bytes",
            "xkv_factor_ak_bytes", "xkv_factor_bk_bytes", "xkv_factor_av_bytes", "xkv_factor_bv_bytes",
            "xkv_factor_payload_bytes", "xkv_factor_metadata_bytes", "xkv_factor_padding_bytes",
            "xkv_landmark_payload_bytes", "xkv_landmark_metadata_bytes", "xkv_landmark_exception_bytes",
            "xkv_index_bytes", "xkv_codec_shared_bytes", "xkv_decode_tile_cache_bytes",
            "xkv_capture_bytes", "xkv_candidate_bytes", "xkv_snapshot_pinned_bytes",
            "xkv_allocator_live_bytes", "xkv_allocator_reserved_bytes",
            "xkv_device_peak_bytes", "xkv_host_peak_bytes",
            "xkv_unique_payloads", "xkv_aliased_payloads", "xkv_baseline_same_rows_bytes",
            "xkv_covered_compressed_bytes",
            "xkv_factored_baseline_byte_coverage", "xkv_factor_quant_ratio",
            "xkv_net_extra_compression_ratio", "xkv_seal_seconds", "xkv_factor_quant_seconds",
            "xkv_landmark_quant_seconds", "xkv_select_seconds", "xkv_refine_seconds",
            "xkv_reconstruct_seconds", "xkv_read_seconds", "xkv_pack_seconds",
            "xkv_segments_sealed", "xkv_segments_skipped_by_reason",
            "xkv_sr_selected_rows", "xkv_sr_fragments", "xkv_effective_chunk_size",
            "xkv_landmark_refine_rows", "xkv_landmark_refine_cap_hits",
            "xkv_spec_stale_total", "xkv_transaction_abort_total", "xkv_synchronize_total",
            "xkv_compression_goal_met", "xkv_throttle_total", "xkv_throttle_reason" }) {
        CHECK(j.contains(key));
    }
    CHECK(j.contains("xkv_graph_timings_evaluated"));
    CHECK(j.contains("xkv_pack_timer_evaluated"));
    CHECK(j.contains("xkv_sr_counters_evaluated"));
    CHECK(j.contains("xkv_spec_counters_evaluated"));
    CHECK(j["xkv_admission"]["safe_next_ubatch"] == 7);
    CHECK(j["xkv_admission"]["limit_reason"] == "hot_slots");
    CHECK(j["xkv_segments_skipped_by_reason"]["no_saving"] == 3);
    CHECK(j["xkv_compression_goal_met"] == true);
    CHECK(j["xkv_source"] == "decoded-hot");
    CHECK(j["xkv_covered_compressed_bytes"] == 18);

    std::ostringstream os;
    xm.to_prometheus(os);
    const std::string prom = os.str();
    CHECK(prom.find("llamacpp:xkv_source_info{source=\"decoded-hot\"}") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_covered_compressed_bytes 18") != std::string::npos);
    for (const char * needle : {
            "llamacpp:xkv_safe_next_ubatch", "llamacpp:xkv_hot_free",
            "llamacpp:xkv_factor_ak_bytes", "llamacpp:xkv_factor_bk_bytes",
            "llamacpp:xkv_factor_av_bytes", "llamacpp:xkv_factor_bv_bytes",
            "llamacpp:xkv_workspace_peak_bytes", "llamacpp:xkv_decode_tile_cache_bytes",
            "llamacpp:xkv_seal_seconds", "llamacpp:xkv_factor_quant_seconds",
            "llamacpp:xkv_read_seconds", "llamacpp:xkv_pack_seconds",
            "llamacpp:xkv_sr_selected_rows", "llamacpp:xkv_sr_fragments",
            "llamacpp:xkv_synchronize_total", "llamacpp:xkv_compression_goal_met",
            "llamacpp:xkv_limit_reason{reason=\"hot_slots\"}",
            "llamacpp:xkv_segments_skipped_total{reason=\"codec_error\"} 1",
            "llamacpp:xkv_segments_skipped_total{reason=\"no_saving\"} 3",
            "llamacpp:xkv_profile_info{requested=\"tq-factors-landmarks\",effective=\"tq-factors\"}",
            "llamacpp:xkv_mode_info{requested_mode=\"sr\",effective_mode=\"sr\"}",
            "llamacpp:xkv_mode{mode=\"sr\"}",
            "llamacpp:xkv_codec_profile_info{a_k=\"turbo4_0\",b_k=\"turbo4_0\",a_v=\"turbo2_0\",b_v=\"turbo2_0\",landmark=\"q8_0\",factorizer=\"cpu-reference\",balance=\"upstream\"}",
            "llamacpp:xkv_seed_info{seed=\"0x584b565352303031\"}",
            "llamacpp:model_artifact_info{sha256=\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
            "llamacpp:tri_calibration_info{fingerprint=\"1234567890abcdef\",valid=\"true\",sha256=\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
            "llamacpp:tri_config_info{ratio=\"3/32\",recent_window=\"128\"}",
            "llamacpp:xkv_compression_goal_evaluated 1",
            "llamacpp:xkv_throttle_reason{reason=\"hot_slots\"}" }) {
        CHECK(prom.find(needle) != std::string::npos);
    }
    // Fingerprints are exact lowercase hex info series, never float gauges.
    CHECK(prom.find("llamacpp:xkv_codec_info{fingerprint=\"0000000000001234\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_profile_fingerprint_info{fingerprint=\"000000000000def0\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_codec_fingerprint ") == std::string::npos);
    // Source fingerprint series is xkv_source_fingerprint_info (xkv_source_info carries {source}).
    CHECK(prom.find("llamacpp:xkv_source_fingerprint_info{fingerprint=\"0000000000005678\"} 1") != std::string::npos);
    // Live evaluated flags emit 1 when evaluated.
    CHECK(prom.find("llamacpp:xkv_graph_timings_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_pack_timer_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_sr_counters_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_spec_counters_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_ratios_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_timers_evaluated 1") != std::string::npos);
    // HELP/TYPE for skipped reasons emitted once, not per entry.
    {
        const std::string tag = "# HELP llamacpp:xkv_segments_skipped_total";
        CHECK(prom.find(tag) != std::string::npos);
        CHECK(prom.find(tag, prom.find(tag) + 1) == std::string::npos);
    }
    CHECK(j["xkv_admission"]["factor_safe_tokens"] == 50);
    CHECK(j["xkv_admission"]["workspace_safe_tokens"] == 30);
    std::cout << "[metrics_fields] done." << std::endl;
}

int main() {
    test_off_mapping();
    test_tightest_reason();
    test_recurrent_gating();
    test_reserve_overflow();
    test_store_budget();
    test_maintenance_status();
    test_runtime_snapshot_default();
    test_runtime_abi();
    test_partition_budget();
    test_scratch_estimator();
    test_store_mib_math();
    test_planner_policy();
    test_metrics_fields();

    if (g_failures != 0) {
        std::fprintf(stderr, "test-xkv-admission: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("test-xkv-admission: all tests OK\n");
    return 0;
}
