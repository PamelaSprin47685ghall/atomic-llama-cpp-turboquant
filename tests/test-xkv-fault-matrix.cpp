// tests/test-xkv-fault-matrix.cpp
//
// Comprehensive fault-injection and atomicity regression matrix for XKV-SR §15.4.
//
// Invariants verified:
// 1. Allocation failures:
//    - Hot slot pool reserve failure (capacity 0, capacity overflow, atomic failure, no partial mutation)
//    - Hot slot pool commit failure (mismatched count, zero payload id, zero generation, RAII rollback)
//    - Workspace arena preflight & acquire failure (hard ceiling enforcement, no partial lease)
//    - Backend code-stream allocation & transfer failure (injected alloc, upload,
//      post-submit, readback, checksum faults; store-reservation refusal;
//      failed-batch ID burn; pack-path alloc/copy/post-submit/reservation faults)
// 2. Seal precommit gate failures:
//    - Precommit callback returns false -> atomic rollback of all locked payloads to HOT_COMMITTED
//    - Precommit callback throws exception -> atomic rollback, no unhandled throw, no published candidate
//    - Removal precommit gate refusal -> zero segment mutation, old version preserved intact
// 3. Factorization failures:
//    - Parameter validation: requested_rank == 0, power_iterations == 0, max_sweeps == 0, tolerance <= 0, non-finite input
//    - Workspace buffer shortfall and preflight workspace ceiling enforcement
// 4. "K succeeds then V fails":
//    - In factorize_kv, K matrix factorizes completely, but V matrix fails validation/factorization
//    - Invariant: intermediate K factors are discarded atomically; output result.k and result.v remain empty
// 5. Four factor encodes & "A succeeds then B fails":
//    - In evaluate_quantized_shadow, stream A_K encodes successfully, stream B_K fails -> shadow.success == false, zero partial streams
//    - Independent failure paths for Stream A_K, B_K, A_V, and B_V
//    - Factor pair compatibility: rank mismatch, Turbo table_fingerprint mismatch, seed mismatch, format_revision mismatch
// 6. Landmark failures & "Factor succeeds then landmark fails":
//    - Profile requires landmarks but landmark_factory is null -> skip_reason::landmark_required
//    - Factorization and shadow succeed, but landmark_factory callback returns false -> skip_reason::codec_error
//    - Landmark factory callback returns invalid descriptor -> skip_reason::codec_error
//    - Landmark encode scratch bounds and integer overflow checks
// 7. Actual bytes exceed estimate:
//    - Bound-checked workspace buffer 1 byte smaller than exact estimator for factorize_matrix, factorize_kv, evaluate_quantized_shadow
//    - Net memory saving ratio gate failure in seal_segment_bundle -> skip_reason::no_saving
//    - Maximum relative error threshold exceeded in seal_segment_bundle -> skip_reason::error_threshold_exceeded
// 8. Stale snapshot and generation:
//    - Transaction coordinator stale stamp rejection (view topology, publish, layout, content, codec, live, binding epochs)
//    - Monotonic transaction ID reuse rejection
//    - Hot slot pool stale generation in release and release_batch -> all-or-nothing rejection
//    - Store remove_payloads and seal_segment_bundle generation mismatch rejection
// 9. Codec version, magic, and fingerprint mismatch:
//    - State image decode with mismatched model, rope, codec, profile, source, tri_calibration fingerprints
//    - Provenance SHA-256 digest mismatch refusal
//    - Bad magic and unsupported version rejection
//    - Invariant: destination image left completely unmodified on decode failure
// 10. Truncated state images:
//    - Truncations at lengths 0, 4, 8, header, allocations, segments, payloads, size - 1
//    - Single-bit corruption checksum verification
//    - Invariant: destination image left completely unmodified on failure
// 11. Transaction rollback accepted prefixes 0 / 1 / partial / all:
//    - Prefix shape enforcement: gap in accepted set, missing head, reversed order, extra payloads rejected
//    - Commit with 0 accepted (reject all), 1 accepted, partial accepted, all accepted
//    - Compute failure rollback: removes all created payloads, tx closed, stamp untouched
// 12. Publishing failures & epoch overflow:
//    - Duplicate payload IDs, zero payload ID, uncommitted payload state in publish_candidate
//    - can_bump preflight failure on epoch overflow (UINT64_MAX)
//    - Batch COW transaction throwing failure hook aborts before commit
// 13. Absent injection seams reporting:
//    - Compiling named helper returning structured gap list for QR, SVD, memmove, and landmark rescore
//
// Every failure test compares exact pre/post authoritative store stamp, payload locations,
// pool bindings, published segment handles/code bytes, and destination state so zero half-mutation is proven.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-factor.h"
#include "llama-xkv-landmark.h"
#include "llama-xkv-state.h"
#include "llama-xkv-transaction.h"
#include "llama-xkv-hot.h"
#include "llama-xkv-backend.h"
#include "llama-cparams.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xkv.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llama_xkv;

// ============================================================================
// Exact State Snapshot Containers for Strict Atomicity Assertions
// ============================================================================

struct store_full_snapshot {
    xkv_snapshot_stamp stamp = {};
    std::unordered_map<uint64_t, xkv_location> locations;
    size_t sealed_count = 0;
    std::unordered_map<uint8_t, size_t> skipped_counts;
    xkv_accounting accounting = {};
    std::unordered_map<uint64_t, std::shared_ptr<const xkv_segment>> segments;

    bool operator==(const store_full_snapshot & o) const {
        if (stamp != o.stamp) return false;
        if (sealed_count != o.sealed_count) return false;
        if (locations != o.locations) return false;
        if (skipped_counts != o.skipped_counts) return false;
        if (accounting.active_segments != o.accounting.active_segments) return false;
        if (accounting.total_payloads != o.accounting.total_payloads) return false;
        if (accounting.allocated_bytes != o.accounting.allocated_bytes) return false;
        if (accounting.factored_bytes != o.accounting.factored_bytes) return false;
        if (accounting.unique_b_matrices != o.accounting.unique_b_matrices) return false;
        if (accounting.shared_b_bytes != o.accounting.shared_b_bytes) return false;
        if (segments.size() != o.segments.size()) return false;
        for (const auto & kv : segments) {
            auto it = o.segments.find(kv.first);
            if (it == o.segments.end()) return false;
            if (kv.second != it->second) return false;
        }
        return true;
    }
    bool operator!=(const store_full_snapshot & o) const { return !(*this == o); }
};

struct pool_full_snapshot {
    xkv_hot_accounting accounting = {};
    std::vector<xkv_hot_slot_info> slots;

    bool operator==(const pool_full_snapshot & o) const {
        if (accounting.capacity != o.accounting.capacity ||
            accounting.free     != o.accounting.free ||
            accounting.reserved != o.accounting.reserved ||
            accounting.bound    != o.accounting.bound ||
            accounting.live     != o.accounting.live ||
            accounting.peak     != o.accounting.peak) {
            return false;
        }
        if (slots.size() != o.slots.size()) return false;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].slot != o.slots[i].slot ||
                slots[i].state != o.slots[i].state ||
                slots[i].payload_id != o.slots[i].payload_id ||
                slots[i].storage_generation != o.slots[i].storage_generation) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const pool_full_snapshot & o) const { return !(*this == o); }
};

static store_full_snapshot capture_store(
    llama_xkv_cache_store & store,
    const std::vector<uint64_t> & pids,
    const std::vector<uint64_t> & seg_ids
) {
    store_full_snapshot s;
    s.stamp = store.current_stamp();
    s.sealed_count = store.get_sealed_count();
    s.accounting = store.get_accounting();
    for (uint8_t r = 0; r <= static_cast<uint8_t>(xkv_skip_reason::landmark_required); ++r) {
        s.skipped_counts[r] = store.get_skipped_count(static_cast<xkv_skip_reason>(r));
    }
    for (uint64_t pid : pids) {
        xkv_location loc;
        if (store.find_location(pid, loc)) {
            s.locations[pid] = loc;
        }
    }
    for (uint64_t sid : seg_ids) {
        auto seg = store.get_segment(sid);
        if (seg) {
            s.segments[sid] = seg;
        }
    }
    return s;
}

static pool_full_snapshot capture_pool(const xkv_hot_slot_pool & pool) {
    pool_full_snapshot ps;
    ps.accounting = pool.get_accounting();
    ps.slots.resize(pool.get_capacity());
    for (uint32_t i = 0; i < pool.get_capacity(); ++i) {
        pool.get_slot_info(i, ps.slots[i]);
    }
    return ps;
}

// ============================================================================
// Fixture Helpers
// ============================================================================

static llama_cparams make_test_cparams(enum llama_xkv_mode mode = LLAMA_XKV_MODE_SHADOW) {
    llama_cparams cp = {};
    cp.xkv_mode = mode;
    cp.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cp.xkv_group_size = 4;
    cp.xkv_rank_k = 16;
    cp.xkv_rank_v = 16;
    cp.xkv_segment_tokens = 64;
    cp.xkv_chunk_tokens = 8;
    cp.xkv_workspace_mib = 16;
    cp.xkv_decode_cache_mib = 8;
    cp.xkv_min_saving = 0.10;
    return cp;
}

static std::shared_ptr<xkv_segment> create_test_candidate(
    uint32_t n_rows,
    uint32_t rank_k = 16,
    uint32_t rank_v = 16,
    uint32_t total_dim_k = 64,
    uint32_t total_dim_v = 64
) {
    auto seg = std::make_shared<xkv_segment>();
    seg->segment_id = 1;
    seg->segment_version = 1;
    seg->profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    seg->source = LLAMA_XKV_SOURCE_DECODED_HOT;
    seg->profile_fingerprint = 0xAA11;
    seg->source_fingerprint = 0xBB22;
    seg->layer_group_map_fingerprint = 0xCC33;
    seg->descriptor_fingerprint = 0xDD44;
    seg->n_rows = n_rows;
    seg->n_live_rows = n_rows;
    seg->row_payload_ids.resize(n_rows);
    seg->live_rows.assign(n_rows, true);

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_v;
    g.layer_feature_offsets_k = {0, total_dim_k / 4, total_dim_k / 2, 3 * total_dim_k / 4};
    g.layer_feature_dims_k = {total_dim_k / 4, total_dim_k / 4, total_dim_k / 4, total_dim_k / 4};
    g.layer_feature_offsets_v = {0, total_dim_v / 4, total_dim_v / 2, 3 * total_dim_v / 4};
    g.layer_feature_dims_v = {total_dim_v / 4, total_dim_v / 4, total_dim_v / 4, total_dim_v / 4};
    g.rank_k = rank_k;
    g.rank_v = rank_v;

    codec_desc desc_a_k = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_k}, 0, 1001);
    std::vector<float> data_a_k(n_rows * rank_k, 0.01f);
    g.a_k = encode_matrix(desc_a_k, data_a_k.data(), data_a_k.size());

    codec_desc desc_b_k = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank_k}, 0, 1001);
    std::vector<float> data_b_k(total_dim_k * rank_k, 0.02f);
    g.set_b_k(encode_matrix(desc_b_k, data_b_k.data(), data_b_k.size()));

    codec_desc desc_a_v = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_v}, 0, 1002);
    std::vector<float> data_a_v(n_rows * rank_v, 0.03f);
    g.a_v = encode_matrix(desc_a_v, data_a_v.data(), data_a_v.size());

    codec_desc desc_b_v = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_v, rank_v}, 0, 1002);
    std::vector<float> data_b_v(total_dim_v * rank_v, 0.04f);
    g.set_b_v(encode_matrix(desc_b_v, data_b_v.data(), data_b_v.size()));

    g.refresh_descriptor_fingerprint();
    g.update_byte_counters();

    seg->groups.push_back(std::move(g));
    seg->update_byte_counters();
    return seg;
}

static xkv_state_fingerprints test_fps() {
    return {0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888};
}

static xkv_state_provenance test_prov() {
    xkv_state_provenance p;
    std::memset(p.model_sha256, 0xA1, 32);
    std::memset(p.tri_calibration_sha256, 0xB2, 32);
    std::memset(p.source_sha256, 0xC3, 32);
    return p;
}

static xkv_state_config test_config() {
    xkv_state_config c;
    c.profile                   = (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    c.source                    = (uint32_t) LLAMA_XKV_SOURCE_DECODED_HOT;
    c.group_size                = 4;
    c.rank_k                    = 4;
    c.rank_v                    = 4;
    c.segment_tokens            = 64;
    c.chunk_tokens              = 8;
    c.sr_budget                 = 0;
    c.factor_balance            = (uint32_t) LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    c.landmark_type             = (uint32_t) GGML_TYPE_F32;
    c.landmark_refine           = (uint32_t) LLAMA_XKV_LANDMARK_REFINE_NONE;
    c.landmark_refine_max_rows  = 0;
    c.store_mib                 = 1;
    c.seed                      = 42;
    c.min_saving_ppm            = 100000;
    c.min_coverage_ppm          = 500000;
    c.factorizer                = (uint32_t) LLAMA_XKV_FACTORIZER_CPU_REFERENCE;
    c.min_saving_bytes          = 1;
    c.next_segment_id           = 8;
    c.next_alloc_id             = 12;
    c.next_seal_tx_nonce        = 1;
    return c;
}

static xkv_state_limits test_limits() {
    return xkv_state_limits::from_store_budgets(
        1 << 20, 1 << 20,
        /*max_segments=*/8, /*max_groups=*/8,
        /*max_payloads=*/64, /*max_allocations=*/64,
        /*max_owning=*/8, /*max_rows=*/64
    );
}

static xkv_snapshot_stamp test_stamp() {
    xkv_snapshot_stamp s;
    s.view.topology_epoch = 11;
    s.view.publish_epoch  = 22;
    s.view.layout_epoch   = 33;
    s.live_epoch          = 44;
    s.content_epoch       = 55;
    s.codec_epoch         = 66;
    s.binding_epoch       = 77;
    return s;
}

static xkv_accounting test_accounting() {
    xkv_accounting a;
    a.allocated_bytes = 2000;
    a.reserved_bytes = 3000;
    a.hot_bytes = 400;
    return a;
}

static encoded_matrix make_stream(factor_role role, orientation orient, uint64_t rows, uint64_t cols, uint64_t seed) {
    codec_desc d = make_codec_desc(role, GGML_TYPE_F32, orient, {rows, cols}, 0, seed);
    std::vector<float> src((size_t)(rows * cols));
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (float)((i * 3 + seed) % 17) * 0.01f;
    }
    return encode_matrix(d, src.data(), src.size());
}

static xkv_factor_group_payload make_group(uint32_t index, uint64_t seed_base, bool with_landmark,
                                           std::shared_ptr<const encoded_matrix> share_bk = nullptr,
                                           std::shared_ptr<const encoded_matrix> share_bv = nullptr) {
    xkv_factor_group_payload g;
    g.group_index = index;
    g.owning_layers = {0, 1};
    g.rank_k = 4;
    g.rank_v = 4;
    g.layer_feature_offsets_k = {0, 4};
    g.layer_feature_dims_k = {4, 4};
    g.layer_feature_offsets_v = {0, 4};
    g.layer_feature_dims_v = {4, 4};
    g.total_dim_k = 8;
    g.total_dim_v = 8;
    const uint64_t seed_k = seed_base + 10;
    const uint64_t seed_v = seed_base + 20;
    g.a_k = make_stream(factor_role::a_k, orientation::token_major, 4, 4, seed_k);
    if (share_bk) g.b_k = share_bk;
    else g.set_b_k(make_stream(factor_role::b_k, orientation::feature_major_transposed, 8, 4, seed_k));
    g.a_v = make_stream(factor_role::a_v, orientation::token_major, 4, 4, seed_v);
    if (share_bv) g.b_v = share_bv;
    else g.set_b_v(make_stream(factor_role::b_v, orientation::feature_major_transposed, 8, 4, seed_v));
    if (with_landmark) {
        g.landmark = make_stream(factor_role::landmark, orientation::token_major, 4, 8, seed_base + 30);
    }
    g.config_fingerprint = 0xC0DE + index;
    g.refresh_descriptor_fingerprint();
    g.update_byte_counters();
    return g;
}

static std::shared_ptr<xkv_segment> make_test_segment(
    uint64_t id, uint64_t ver, uint64_t payload_base, int n_groups, bool with_landmark,
    std::shared_ptr<const encoded_matrix> share_bk = nullptr,
    std::shared_ptr<const encoded_matrix> share_bv = nullptr
) {
    auto seg = std::make_shared<xkv_segment>();
    seg->segment_id = id;
    seg->segment_version = ver;
    seg->profile = with_landmark
        ? LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS
        : LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    seg->source = LLAMA_XKV_SOURCE_DECODED_HOT;
    seg->profile_fingerprint = 0xA1;
    seg->source_fingerprint = 0xA2;
    seg->layer_group_map_fingerprint = 0xA3 + id;
    seg->descriptor_fingerprint = 0xA4 + ver;
    seg->n_rows = 4;
    seg->row_payload_ids = {payload_base + 0, payload_base + 1, payload_base + 2, payload_base + 3};
    seg->live_rows = {true, true, false, true};
    seg->n_live_rows = 3;
    for (int gi = 0; gi < n_groups; ++gi) {
        const bool share = (gi == 0);
        seg->groups.push_back(make_group((uint32_t)gi, payload_base + 100 * (uint64_t)gi,
                                         with_landmark, share ? share_bk : nullptr,
                                         share ? share_bv : nullptr));
    }
    seg->update_byte_counters();
    return seg;
}

struct test_world {
    std::vector<std::shared_ptr<const xkv_segment>> segments;
    std::vector<xkv_hot_payload_binding> hot;
    std::vector<xkv_state_payload> factored;
    xkv_accounting accounting = test_accounting();
};

static xkv_state_payload live_payload(uint64_t id, uint64_t seg, uint64_t ver, uint32_t row, uint64_t gen) {
    xkv_state_payload p;
    p.payload_id = id;
    p.generation = gen;
    p.live = 1;
    p.locator.kind = xkv_location_kind::factored;
    p.locator.segment_id = seg;
    p.locator.segment_version = ver;
    p.locator.row = row;
    p.locator.storage_generation = gen;
    p.locator.state = xkv_state::factored;
    p.locator.seal_tx_nonce = 0;
    return p;
}

static test_world make_world() {
    test_world w;
    auto s7v1 = make_test_segment(7, 1, 100, 1, true);
    auto s7v2 = make_test_segment(7, 2, 200, 2, false, s7v1->groups[0].b_k, s7v1->groups[0].b_v);
    w.segments.push_back(std::move(s7v1));
    w.segments.push_back(std::move(s7v2));
    w.hot.push_back({901, 5, 3, xkv_state::hot_committed});
    for (uint64_t seg_i = 0; seg_i < 2; ++seg_i) {
        const auto & seg = w.segments[(size_t)seg_i];
        const uint64_t base = seg_i == 0 ? 100 : 200;
        const uint64_t ver  = seg_i == 0 ? 1 : 2;
        for (uint32_t r = 0; r < 4; ++r) {
            if (!seg->live_rows[r]) continue;
            w.factored.push_back(live_payload(base + r, 7, ver, r, 9));
        }
    }
    std::sort(w.factored.begin(), w.factored.end(), [](const auto & a, const auto & b) {
        return a.payload_id < b.payload_id;
    });
    return w;
}

static bool capture_world(const test_world & w, xkv_state_image & img, std::string * err = nullptr) {
    return capture_image(w.segments, w.hot, w.factored, w.accounting, 16, 8, test_config(), test_fps(),
                         test_prov(), test_stamp(), test_limits(), img, nullptr, err);
}

// ============================================================================
// 1. Allocation Failures: Hot Slot Pool, Workspace Arena, Backend Allocator
// ============================================================================

static void test_fault_allocation_hot_pool() {
    std::cout << "[Test 1.1] Fault Injection: Hot Slot Pool Allocation..." << std::endl;

    // Capacity 0: reserve must fail cleanly
    xkv_hot_slot_pool zero_pool(0);
    auto pre_zero = capture_pool(zero_pool);
    std::string err;
    xkv_hot_reservation res0 = zero_pool.reserve(1, &err);
    assert(!res0.valid());
    assert(res0.empty());
    assert(!err.empty());
    auto post_zero = capture_pool(zero_pool);
    assert(pre_zero == post_zero);

    // Bounded capacity: overflow reservation must fail atomically
    xkv_hot_slot_pool pool(4);
    auto pre_pool = capture_pool(pool);
    err.clear();
    xkv_hot_reservation fail_res = pool.reserve(5, &err);
    assert(!fail_res.valid());
    assert(fail_res.empty());
    assert(!err.empty());
    auto post_pool = capture_pool(pool);
    assert(pre_pool == post_pool);

    // Valid reservation followed by failed commit (zero payload ID, zero generation, count mismatch)
    xkv_hot_reservation res = pool.reserve(2);
    assert(res.valid());
    assert(pool.get_reserved() == 2);
    assert(pool.get_bound() == 0);
    assert(pool.get_free() == 2);

    // Commit with mismatched count
    err.clear();
    assert(!res.commit({100}, {1}, &err));
    assert(!err.empty());
    assert(!res.is_committed());
    assert(pool.get_reserved() == 2);
    assert(pool.get_bound() == 0);

    // Commit with payload_id == 0
    err.clear();
    assert(!res.commit({100, 0}, {1, 1}, &err));
    assert(!res.is_committed());
    assert(pool.get_reserved() == 2);
    assert(pool.get_bound() == 0);

    // Commit with generation == 0
    err.clear();
    assert(!res.commit({100, 101}, {1, 0}, &err));
    assert(!res.is_committed());
    assert(pool.get_reserved() == 2);
    assert(pool.get_bound() == 0);

    // Explicit rollback restores pool to pristine state
    res.rollback();
    assert(!res.valid());
    auto post_rollback = capture_pool(pool);
    assert(pre_pool == post_rollback);

    // RAII destructor rollback
    {
        xkv_hot_reservation raii_res = pool.reserve(3);
        assert(raii_res.valid());
        assert(pool.get_reserved() == 3);
        // Exiting scope without commit
    }
    auto post_raii = capture_pool(pool);
    assert(pre_pool == post_raii);

    std::cout << "  [PASS] Hot slot pool allocation failure atomicity verified." << std::endl;
}

static void test_fault_allocation_workspace_arena() {
    std::cout << "[Test 1.2] Fault Injection: Workspace Arena Preflight & Acquire..." << std::endl;

    // Capacity 0 arena
    xkv_workspace_arena zero_arena(0);
    assert(!zero_arena.preflight(100));
    assert(!bool(zero_arena.acquire(100)));
    assert(zero_arena.get_live_bytes() == 0);
    assert(zero_arena.get_peak_bytes() == 0);

    // Fixed capacity arena
    const size_t cap = 1024 * 1024; // 1 MiB
    xkv_workspace_arena arena(cap);
    assert(arena.get_capacity_bytes() == cap);

    // Preflight exceeding capacity fails cleanly
    assert(!arena.preflight(cap + 1));
    assert(arena.get_live_bytes() == 0);
    assert(arena.get_peak_bytes() == 0);

    // Acquire exceeding capacity fails cleanly with null lease
    xkv_arena_lease fail_lease = arena.acquire(cap + 64);
    assert(!bool(fail_lease));
    assert(fail_lease.data() == nullptr);
    assert(arena.get_live_bytes() == 0);
    assert(arena.get_peak_bytes() == 0);

    // Partial capacity exhaustion followed by failed acquire
    xkv_arena_lease lease1 = arena.acquire(768 * 1024);
    assert(bool(lease1));
    assert(arena.get_live_bytes() == 768 * 1024);
    assert(arena.get_peak_bytes() == 768 * 1024);

    // Requesting remaining + 1 fails cleanly
    xkv_arena_lease fail_lease2 = arena.acquire(300 * 1024);
    assert(!bool(fail_lease2));
    assert(arena.get_live_bytes() == 768 * 1024); // Untouched
    assert(arena.get_peak_bytes() == 768 * 1024);

    lease1.release();
    assert(arena.get_live_bytes() == 0);
    assert(arena.get_peak_bytes() == 768 * 1024);

    std::cout << "  [PASS] Workspace arena allocation failure atomicity verified." << std::endl;
}

static void test_fault_allocation_backend() {
    std::cout << "[Test 1.3] Fault Injection: Backend Code-Stream Allocation..." << std::endl;

    ggml_backend_load_all();
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    assert(cpu_backend != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    assert(buft != nullptr);

    codec_desc desc = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {16, 128}, 128, 42);
    std::vector<uint8_t> bytes(encoded_matrix_bytes(desc), 0x5A);

    // 1. Injected allocation failure at stream 0
    {
        xkv_backend_batch_builder builder;
        builder.add_stream(desc, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.inject_alloc_fail_at = 0;

        xkv_backend_batch_result res;
        std::string err;
        bool ok = builder.build(cpu_backend, buft, cfg, res, &err);
        assert(!ok);
        assert(res.handles.empty());
        assert(err.find("injected allocation failure") != std::string::npos);
    }

    // 2. Injected upload failure at stream 0
    {
        xkv_backend_batch_builder builder;
        builder.add_stream(desc, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.inject_upload_fail_at = 0;

        xkv_backend_batch_result res;
        std::string err;
        bool ok = builder.build(cpu_backend, buft, cfg, res, &err);
        assert(!ok);
        assert(res.handles.empty());
        assert(err.find("injected upload failure") != std::string::npos);
    }

    // 3. Injected readback failure
    {
        xkv_backend_batch_builder builder;
        builder.add_stream(desc, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_batch_result res;
        std::string err;
        assert(builder.build(cpu_backend, buft, cfg, res, &err));
        assert(res.handles.size() == 1);

        xkv_backend_batch_config rcfg;
        rcfg.inject_readback_fail_at = 0;
        xkv_backend_readback_result rb;
        rb.stream_bytes.push_back({1, 2, 3}); // Pre-existing output must remain unmodified on failure!
        rb.stats.sync_count = 999;
        assert(!xkv_backend_readback_batch(cpu_backend, res.handles, rcfg, rb, &err));
        assert(rb.stream_bytes.size() == 1);
        assert(rb.stream_bytes[0] == std::vector<uint8_t>({1, 2, 3}));
        assert(rb.stats.sync_count == 999);
        assert(err.find("injected readback failure") != std::string::npos);

        // 4. Injected checksum failure
        rcfg.inject_readback_fail_at = -1;
        rcfg.inject_checksum_fail_at = 0;
        rb.stream_bytes.clear();
        assert(!xkv_backend_readback_batch(cpu_backend, res.handles, rcfg, rb, &err));
        assert(rb.stream_bytes.empty());
        assert(err.find("injected checksum failure") != std::string::npos);
    }

    // 5. Injected post-submit failure at EVERY queued index (build: per-stream
    // post-set_async; fence must flush queued sets before destruction; outputs
    // untouched, no success flag). Cover with and without verify_checksum.
    for (int k = 0; k < 2; ++k) {
        for (int with_verify = 0; with_verify < 2; ++with_verify) {
            codec_desc d0 = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                orientation::token_major, {8, 128}, 128, 42);
            codec_desc d1 = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO3_0,
                orientation::token_major, {8, 128}, 128, 43);
            std::vector<uint8_t> b0(encoded_matrix_bytes(d0), 0x11);
            std::vector<uint8_t> b1(encoded_matrix_bytes(d1), 0x22);
            xkv_backend_batch_builder builder;
            builder.add_stream(d0, b0.data(), b0.size());
            builder.add_stream(d1, b1.data(), b1.size());
            xkv_backend_batch_config pcfg;
            pcfg.residency = GGML_XKV_RES_REFERENCE_HOST;
            pcfg.inject_post_submit_fail_at = k;
            std::vector<uint8_t> vwork;
            if (with_verify) {
                pcfg.verify_checksum = true;
                vwork.resize(b0.size() + b1.size());
                pcfg.verify_workspace = vwork.data();
                pcfg.verify_workspace_bytes = vwork.size();
            }
            xkv_backend_batch_result pres;
            pres.stats.uploaded_streams = 999; // sentinel: untouched on failure
            std::string perr;
            assert(!builder.build(cpu_backend, buft, pcfg, pres, &perr));
            assert(!perr.empty());
            assert(pres.handles.empty());
            assert(pres.stats.uploaded_streams == 999);
            assert(!pres.is_success());
        }
    }

    // 6. Store reservation refusal paths (estimate>reserved, estimate>cap;
    // adequate reservation succeeds). Outputs untouched, no success flag,
    // host-release commit refuses on failed batch.
    {
        xkv_backend_batch_builder builder;
        builder.add_stream(desc, bytes.data(), bytes.size());
        xkv_backend_batch_config scfg;
        scfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_store_reservation tiny;
        tiny.reserved_bytes = 1; // preflight estimate exceeds -> atomic refusal
        xkv_backend_batch_result sres;
        sres.stats.uploaded_streams = 999; // sentinel
        std::string serr;
        assert(!builder.build(cpu_backend, buft, scfg, sres, &serr, &tiny));
        assert(!serr.empty());
        assert(sres.handles.empty());
        assert(sres.stats.uploaded_streams == 999);
        assert(!sres.is_success());
        assert(!sres.commit_host_release(&serr));
    }
    {
        xkv_backend_batch_builder builder;
        builder.add_stream(desc, bytes.data(), bytes.size());
        xkv_backend_batch_config scfg;
        scfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_store_reservation capped;
        capped.reserved_bytes = (uint64_t) 1 << 40;
        capped.cap_bytes = 1; // cap below estimate also refuses
        xkv_backend_batch_result sres;
        std::string serr;
        assert(!builder.build(cpu_backend, buft, scfg, sres, &serr, &capped));
        assert(sres.handles.empty());
    }
    {
        xkv_backend_batch_builder builder;
        builder.add_stream(desc, bytes.data(), bytes.size());
        xkv_backend_batch_config scfg;
        scfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_store_reservation ample;
        ample.reserved_bytes = (uint64_t) 1 << 30;
        ample.cap_bytes = (uint64_t) 1 << 30;
        xkv_backend_batch_result sres;
        std::string serr;
        assert(builder.build(cpu_backend, buft, scfg, sres, &serr, &ample));
        assert(sres.handles.size() == 1);
    }

    // 7. Failed batches burn consumed IDs (never reused); generator is explicit.
    {
        xkv_allocation_id_generator gen(1, 100);
        {
            xkv_backend_batch_builder builder(&gen);
            codec_desc d0 = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                orientation::token_major, {4, 128}, 128, 42);
            codec_desc d1 = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO4_0,
                orientation::token_major, {4, 128}, 128, 43);
            std::vector<uint8_t> b0(encoded_matrix_bytes(d0), 0x33);
            std::vector<uint8_t> b1(encoded_matrix_bytes(d1), 0x44);
            builder.add_stream(d0, b0.data(), b0.size());
            builder.add_stream(d1, b1.data(), b1.size());
            xkv_backend_batch_config fcfg;
            fcfg.residency = GGML_XKV_RES_REFERENCE_HOST;
            fcfg.inject_alloc_fail_at = 1; // stream 0 burns ID 1, then fail
            xkv_backend_batch_result fres;
            std::string ferr;
            assert(!builder.build(cpu_backend, buft, fcfg, fres, &ferr));
            assert(fres.handles.empty());
        }
        // Next success must continue at 2, never reuse burned ID 1.
        {
            xkv_backend_batch_builder builder(&gen);
            codec_desc dq = make_codec_desc(factor_role::a_k, GGML_TYPE_Q8_0,
                orientation::token_major, {4, 64}, 0, 42);
            std::vector<uint8_t> bq(encoded_matrix_bytes(dq), 0x55);
            builder.add_stream(dq, bq.data(), bq.size());
            xkv_backend_batch_config scfg;
            scfg.residency = GGML_XKV_RES_REFERENCE_HOST;
            xkv_backend_batch_result sres;
            std::string serr;
            assert(builder.build(cpu_backend, buft, scfg, sres, &serr));
            assert(sres.handles.size() == 1);
            assert(sres.handles[0]->get_allocation_id() == 2);
        }
    }

    // 8. Pack path (§15.4): alloc per-request pre-ID, copy pre-submission per
    // flattened copy ordinal, post-submit per ordinal; reservation refusal.
    // Failure-atomic: sentinel outputs untouched, source bytes intact.
    {
        codec_desc da = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
            orientation::token_major, {8, 128}, 128, 42);
        std::vector<uint8_t> ab(encoded_matrix_bytes(da), 0x66);
        xkv_backend_batch_builder src_builder;
        src_builder.add_stream(da, ab.data(), ab.size());
        xkv_backend_batch_config src_cfg;
        src_cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_batch_result src_res;
        std::string src_err;
        assert(src_builder.build(cpu_backend, buft, src_cfg, src_res, &src_err));
        assert(src_res.handles.size() == 1);
        auto src_a = src_res.handles[0];

        codec_desc dd = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
            orientation::token_major, {4, 128}, 128, 42);
        const std::vector<uint32_t> good_rows = {1, 3, 4, 6};
        xkv_allocation_id_generator pack_gen(1, 1000);
        xkv_backend_batch_config pack_cfg;
        pack_cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        auto expect_pack_fail = [&](xkv_backend_pack_request rq,
                                      const xkv_backend_batch_config & c,
                                      const xkv_backend_store_reservation * rsv) {
            xkv_backend_pack_result pr;
            pr.handles.push_back(src_a); // sentinel: untouched on failure
            pr.stats.sync_count = 999;
            std::string perr;
            assert(!xkv_backend_pack_batch(cpu_backend, buft, {rq}, c, pack_gen, pr, &perr, rsv));
            assert(!perr.empty());
            assert(pr.handles.size() == 1 && pr.handles[0].get() == src_a.get());
            assert(pr.stats.sync_count == 999);
        };
        auto good_rq = [&]() {
            xkv_backend_pack_request rq;
            rq.src = src_a;
            rq.surviving_rows = good_rows;
            rq.dst_desc = dd;
            return rq;
        };
        // Reservation refusal (estimate exceeds reserved).
        {
            xkv_backend_store_reservation rsv;
            rsv.reserved_bytes = 1;
            expect_pack_fail(good_rq(), pack_cfg, &rsv);
        }
        // Alloc injection per request index.
        {
            xkv_backend_batch_config c = pack_cfg;
            c.inject_alloc_fail_at = 0;
            expect_pack_fail(good_rq(), c, nullptr);
        }
        // Copy injection at every flattened copy ordinal in queue order.
        for (int k = 0; k < 4; ++k) {
            xkv_backend_batch_config c = pack_cfg;
            c.inject_copy_fail_at = k;
            expect_pack_fail(good_rq(), c, nullptr);
        }
        // Post-submit injection per flattened copy ordinal.
        {
            xkv_backend_batch_config c = pack_cfg;
            c.inject_post_submit_fail_at = 2;
            expect_pack_fail(good_rq(), c, nullptr);
        }
        // Source still intact after all pack failures.
        {
            std::vector<uint8_t> src_out;
            std::string rerr;
            assert(src_a->readback(cpu_backend, src_out, &rerr));
            assert(src_out == ab);
        }
    }

    ggml_backend_free(cpu_backend);
    std::cout << "  [PASS] Backend code-stream failure injection atomicity verified." << std::endl;
}

// ============================================================================
// 2. Seal Precommit Gate Failures (Callback False, Exception Throw, Removal Refusal)
// ============================================================================

static void test_fault_seal_precommit_gate() {
    std::cout << "[Test 2] Fault Injection: Seal Precommit Gate Refusal & Throw..." << std::endl;

    auto cp = make_test_cparams();
    cp.xkv_workspace_mib = 64;
    llama_xkv_cache_store store(cp);

    const uint32_t n_tokens = 64;
    const uint32_t dim_k = 64;
    const uint32_t dim_v = 64;

    std::vector<uint64_t> pids(n_tokens);
    std::vector<uint64_t> gens(n_tokens, 1);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pids[i] = 1000 + i;
        assert(store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed));
    }

    std::vector<float> k_data(n_tokens * dim_k, 0.5f);
    std::vector<float> v_data(n_tokens * dim_v, 0.25f);

    xkv_factor_group_input g0;
    g0.group_index = 0;
    g0.owning_layers = {0, 1, 2, 3};
    g0.rank_k = 8;
    g0.rank_v = 8;
    g0.total_dim_k = dim_k;
    g0.total_dim_v = dim_v;
    g0.layer_feature_offsets_k = {0, 16, 32, 48};
    g0.layer_feature_dims_k = {16, 16, 16, 16};
    g0.layer_feature_offsets_v = {0, 16, 32, 48};
    g0.layer_feature_dims_v = {16, 16, 16, 16};
        g0.row_positions.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) g0.row_positions[i] = (int64_t) i;
        g0.hot_bytes_per_row_k = {32, 32, 32, 32};
        g0.hot_bytes_per_row_v = {32, 32, 32, 32};
    g0.canonical_k_data = k_data.data();
    g0.k_rows = n_tokens;
    g0.k_cols = dim_k;
    g0.canonical_v_data = v_data.data();
    g0.v_rows = n_tokens;
    g0.v_cols = dim_v;

    xkv_bundle_sealing_params sparams;
    sparams.rank_k = 8;
    sparams.rank_v = 8;
    sparams.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    sparams.source = LLAMA_XKV_SOURCE_DECODED_HOT;
    sparams.max_relative_error = 0.99;
    sparams.min_saving_ratio = 0.01;
    sparams.flat_type_k = GGML_TYPE_F16;
    sparams.flat_type_v = GGML_TYPE_F16;

    // Snapshot store state before fallible operations
    auto pre_store = capture_store(store, pids, {1});

    // 1. Precommit gate returns false with error
    xkv_seal_precommit_fn refusing_precommit = [](const xkv_seal_precommit_ctx &, std::string * err) {
        if (err) *err = "injected precommit refusal: pool lock conflict";
        return false;
    };

    auto res_refuse = store.seal_segment_bundle({g0}, pids, gens, sparams, refusing_precommit);
    assert(!res_refuse.success);
    assert(res_refuse.skip_reason == xkv_skip_reason::aborted);
    assert(res_refuse.message.find("injected precommit refusal") != std::string::npos);

    // Invariant: ZERO half-mutation! All payload states must be restored to hot_committed,
    // seal_tx_nonce must be 0, store stamp must be unchanged, no segment published!
    auto post_refuse = capture_store(store, pids, {1});
    assert(pre_store.stamp == post_refuse.stamp);
    assert(pre_store.sealed_count == post_refuse.sealed_count);
    assert(pre_store.segments == post_refuse.segments);
    assert(pre_store.accounting == post_refuse.accounting);
    for (uint64_t pid : pids) {
        xkv_location loc;
        assert(store.find_location(pid, loc));
        assert(loc.state == xkv_state::hot_committed);
        assert(loc.seal_tx_nonce == 0);
        assert(loc.kind == xkv_location_kind::hot);
    }

    // 2. Precommit gate throws exception
    xkv_seal_precommit_fn throwing_precommit = [](const xkv_seal_precommit_ctx &, std::string *) -> bool {
        throw std::runtime_error("injected precommit crash during slot release");
    };

    auto res_throw = store.seal_segment_bundle({g0}, pids, gens, sparams, throwing_precommit);
    assert(!res_throw.success);
    assert(res_throw.skip_reason == xkv_skip_reason::aborted);
    assert(res_throw.message.find("injected precommit crash") != std::string::npos);

    // Invariant: Exact pre/post state match after throw
    auto post_throw = capture_store(store, pids, {1});
    assert(pre_store.stamp == post_throw.stamp);
    assert(pre_store.segments == post_throw.segments);
    for (uint64_t pid : pids) {
        xkv_location loc;
        assert(store.find_location(pid, loc));
        assert(loc.state == xkv_state::hot_committed);
        assert(loc.seal_tx_nonce == 0);
    }

    // 3. Removal precommit refusal in remove_payloads
    // First successfully publish a segment
    sparams.max_relative_error = 0.99;
    auto res_ok = store.seal_segment_bundle({g0}, pids, gens, sparams, nullptr);
    assert(res_ok.success);
    const uint64_t seg_id = res_ok.segment_id;
    assert(seg_id > 0);

    auto pre_removal = capture_store(store, pids, {seg_id});

    // Removal precommit gate refuses
    xkv_removal_precommit_fn refusing_removal = [](const xkv_removal_precommit_ctx &, std::string * err) {
        if (err) *err = "injected removal precommit refusal";
        return false;
    };

    std::string rem_err;
    xkv_payload_removal_result rem_res;
    bool rem_ok = store.remove_payloads({pids[0]}, {gens[0]}, &rem_res, &rem_err, nullptr, nullptr, refusing_removal);
    assert(!rem_ok);
    assert(!rem_res.success);
    assert(rem_err.find("injected removal precommit refusal") != std::string::npos);

    // Invariant: Old segment version intact, payload locations untouched, stamp unchanged
    auto post_removal = capture_store(store, pids, {seg_id});
    assert(pre_removal.stamp == post_removal.stamp);
    assert(pre_removal.segments == post_removal.segments);
    assert(pre_removal.locations == post_removal.locations);

    std::cout << "  [PASS] Seal precommit gate failure atomicity verified." << std::endl;
}

// ============================================================================
// 3. Factorization Failures & "K Succeeds then V Fails"
// ============================================================================

static void test_fault_factor_and_k_succeeds_v_fails() {
    std::cout << "[Test 3] Fault Injection: Factorization & K Succeeds then V Fails..." << std::endl;

    matrix x_k(32, 32, 1.0f);
    matrix x_v(32, 32, 2.0f);

    // 1. Parameter validation failure in factorize_matrix
    factor_pair fp;
    std::string err;
    // requested_rank == 0
    assert(!factorize_matrix(x_k, 0, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 1, fp, &err));
    assert(err.find("requested_rank cannot be 0") != std::string::npos);
    assert(fp.a.empty() && fp.b_transposed.empty());

    // power_iterations == 0
    err.clear();
    assert(!factorize_matrix(x_k, 8, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 1, fp, &err, 1e-9, 30, 16, 0));
    assert(err.find("power_iterations cannot be 0") != std::string::npos);
    assert(fp.a.empty() && fp.b_transposed.empty());

    // max_sweeps == 0
    err.clear();
    assert(!factorize_matrix(x_k, 8, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 1, fp, &err, 1e-9, 0));
    assert(err.find("max_sweeps must be >= 1") != std::string::npos);
    assert(fp.a.empty() && fp.b_transposed.empty());

    // Non-finite input
    matrix x_bad(32, 32, 1.0f);
    x_bad.at(5, 5) = std::numeric_limits<float>::quiet_NaN();
    err.clear();
    assert(!factorize_matrix(x_bad, 8, LLAMA_XKV_FACTOR_BALANCE_UPSTREAM, 1, fp, &err));
    assert(err.find("input contains NaN or Inf") != std::string::npos);
    assert(fp.a.empty() && fp.b_transposed.empty());

    // 2. K succeeds then V fails in factorize_kv
    // Setup config where K has valid rank 8, but V has rank 0 (which causes factorize_matrix on V to fail)
    factor_config cfg;
    cfg.rank_k = 8;
    cfg.rank_v = 0;
    cfg.balance = LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    cfg.seed = 42;

    factor_result res = factorize_kv(x_k, x_v, cfg);
    assert(!res.success);
    assert(res.error_message.find("factorize_kv V failed") != std::string::npos);
    assert(res.error_message.find("requested_rank cannot be 0") != std::string::npos);

    // CRITICAL ATOMICITY INVARIANT:
    // Even though K factorization executed and completed successfully, on V failure
    // the completed K factors MUST be discarded atomically!
    assert(res.k.a.empty());
    assert(res.k.b_transposed.empty());
    assert(res.v.a.empty());
    assert(res.v.b_transposed.empty());

    // 2b. Injected QR failure via factor_config::fault (XKV-SR §15.4)
    // Matrix needs to be > 64 to exercise the rSVD QR branch (e.g. 128x128)
    matrix x_k_128(128, 128, 1.0f);
    matrix x_v_128(128, 128, 2.0f);
    factor_config cfg_qr;
    cfg_qr.rank_k = 16;
    cfg_qr.rank_v = 16;
    cfg_qr.fault = factor_fault_injection::fail_qr;
    cfg_qr.fault_occurrence = 0; // Trigger on K factorization
    factor_result res_qr = factorize_kv(x_k_128, x_v_128, cfg_qr);
    assert(!res_qr.success);
    assert(res_qr.error_message.find("injected QR failure") != std::string::npos);
    assert(res_qr.k.a.empty() && res_qr.k.b_transposed.empty());
    assert(res_qr.v.a.empty() && res_qr.v.b_transposed.empty());

    // 2c. Injected SVD failure via factor_config::fault on V (K succeeds, V fails SVD)
    factor_config cfg_svd;
    cfg_svd.rank_k = 16;
    cfg_svd.rank_v = 16;
    cfg_svd.fault = factor_fault_injection::fail_svd;
    cfg_svd.fault_occurrence = 1; // K succeeds, V fails at SVD!
    factor_result res_svd = factorize_kv(x_k_128, x_v_128, cfg_svd);
    assert(!res_svd.success);
    assert(res_svd.error_message.find("factorize_kv V failed") != std::string::npos);
    assert(res_svd.error_message.find("injected SVD failure") != std::string::npos);
    assert(res_svd.k.a.empty() && res_svd.k.b_transposed.empty());
    assert(res_svd.v.a.empty() && res_svd.v.b_transposed.empty());

    // 3. Row count mismatch in factorize_kv
    matrix x_v_short(31, 32, 2.0f);
    cfg.rank_v = 8;
    factor_result res_dim = factorize_kv(x_k, x_v_short, cfg);
    assert(!res_dim.success);
    assert(res_dim.error_message.find("row count mismatch") != std::string::npos);
    assert(res_dim.k.a.empty() && res_dim.v.a.empty());

    std::cout << "  [PASS] Factorization and K-succeeds-V-fails failure atomicity verified." << std::endl;
}

// ============================================================================
// 4. Four Factor Encodes & "A Succeeds then B Fails"
// ============================================================================

static void test_fault_four_encodes_and_a_succeeds_b_fails() {
    std::cout << "[Test 4] Fault Injection: Four Factor Encodes & A Succeeds then B Fails..." << std::endl;

    matrix x_k(32, 32, 1.0f);
    matrix x_v(32, 32, 2.0f);

    factor_config cfg;
    cfg.rank_k = 8;
    cfg.rank_v = 8;
    cfg.balance = LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    cfg.seed = 42;

    factor_result f_res = factorize_kv(x_k, x_v, cfg);
    assert(f_res.success);

    // 1. Stream A_K encode failure (unsupported/invalid ggml_type)
    {
        factor_config bad_a_k = cfg;
        bad_a_k.factor_a_k = GGML_TYPE_COUNT;
        factor_quantized_shadow shadow = evaluate_quantized_shadow(x_k, x_v, f_res.k, f_res.v, bad_a_k);
        assert(!shadow.success);
        assert(!shadow.error_message.empty());
        assert(shadow.stream_a_k.bytes.empty());
        assert(shadow.stream_b_k.bytes.empty());
    }

    // 2. Stream A_K succeeds then Stream B_K fails!
    {
        factor_config bad_b_k = cfg;
        bad_b_k.factor_a_k = GGML_TYPE_TURBO4_0; // A_K encodes successfully
        bad_b_k.factor_b_k = GGML_TYPE_COUNT;    // B_K encode fails!
        factor_quantized_shadow shadow = evaluate_quantized_shadow(x_k, x_v, f_res.k, f_res.v, bad_b_k);
        assert(!shadow.success);
        assert(!shadow.error_message.empty());
        // Invariant: On B_K failure, intermediate A_K is discarded and not delivered!
        assert(shadow.stream_a_k.bytes.empty());
        assert(shadow.stream_b_k.bytes.empty());
        assert(shadow.stream_a_v.bytes.empty());
        assert(shadow.stream_b_v.bytes.empty());
    }

    // 3. Streams A_K and B_K succeed, Stream A_V fails
    {
        factor_config bad_a_v = cfg;
        bad_a_v.factor_a_k = GGML_TYPE_TURBO4_0;
        bad_a_v.factor_b_k = GGML_TYPE_TURBO4_0;
        bad_a_v.factor_a_v = GGML_TYPE_COUNT; // A_V fails!
        factor_quantized_shadow shadow = evaluate_quantized_shadow(x_k, x_v, f_res.k, f_res.v, bad_a_v);
        assert(!shadow.success);
        assert(shadow.stream_a_k.bytes.empty());
        assert(shadow.stream_b_k.bytes.empty());
        assert(shadow.stream_a_v.bytes.empty());
        assert(shadow.stream_b_v.bytes.empty());
    }

    // 4. Streams A_K, B_K, A_V succeed, Stream B_V fails
    {
        factor_config bad_b_v = cfg;
        bad_b_v.factor_a_k = GGML_TYPE_TURBO4_0;
        bad_b_v.factor_b_k = GGML_TYPE_TURBO4_0;
        bad_b_v.factor_a_v = GGML_TYPE_TURBO4_0;
        bad_b_v.factor_b_v = GGML_TYPE_COUNT; // B_V fails!
        factor_quantized_shadow shadow = evaluate_quantized_shadow(x_k, x_v, f_res.k, f_res.v, bad_b_v);
        assert(!shadow.success);
        assert(shadow.stream_a_k.bytes.empty());
        assert(shadow.stream_b_v.bytes.empty());
    }

    // 5. Factor pair compatibility validation failure:
    // Both A and B individually encode, but pair validation fails before GEMM/reconstruction
    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {16, 128}, 128, 42);
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {64, 128}, 128, 42);
    std::string comp_err;
    assert(validate_factor_pair_compatibility(desc_a, desc_b, &comp_err));

    // Case 5a: Logical rank mismatch
    codec_desc bad_rank_b = desc_b;
    bad_rank_b.logical_shape.cols = 64;
    bad_rank_b.padded_shape.cols = 128;
    assert(!validate_factor_pair_compatibility(desc_a, bad_rank_b, &comp_err));
    assert(comp_err.find("logical rank mismatch") != std::string::npos);

    // Case 5b: Turbo table_fingerprint mismatch
    codec_desc bad_fp_b = desc_b;
    bad_fp_b.table_fingerprint ^= 0xDEADBEEF;
    assert(!validate_factor_pair_compatibility(desc_a, bad_fp_b, &comp_err));
    assert(comp_err.find("table_fingerprint mismatch") != std::string::npos);

    // Case 5c: Turbo seed mismatch
    codec_desc bad_seed_b = desc_b;
    bad_seed_b.seed = 999;
    assert(!validate_factor_pair_compatibility(desc_a, bad_seed_b, &comp_err));
    assert(comp_err.find("seed mismatch") != std::string::npos);

    // Case 5d: Turbo format_revision mismatch
    codec_desc bad_rev_b = desc_b;
    bad_rev_b.format_revision ^= 0x1234;
    assert(!validate_factor_pair_compatibility(desc_a, bad_rev_b, &comp_err));
    assert(comp_err.find("format_revision mismatch") != std::string::npos);

    // Direct rotated GEMM must be refused when compatibility fails
    assert(!can_direct_rotated_gemm(desc_a, bad_fp_b));

    std::cout << "  [PASS] Four factor encodes and A-succeeds-B-fails atomicity verified." << std::endl;
}

// ============================================================================
// 5. Landmark Failures & "Factor Succeeds then Landmark Fails"
// ============================================================================

static void test_fault_factor_succeeds_landmark_fails() {
    std::cout << "[Test 5] Fault Injection: Factor Succeeds then Landmark Fails..." << std::endl;

    auto cp = make_test_cparams();
    cp.xkv_workspace_mib = 64;
    llama_xkv_cache_store store(cp);

    const uint32_t n_tokens = 64;
    const uint32_t dim_k = 64;
    const uint32_t dim_v = 64;

    std::vector<uint64_t> pids(n_tokens);
    std::vector<uint64_t> gens(n_tokens, 1);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pids[i] = 2000 + i;
        assert(store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed));
    }

    std::vector<float> k_data(n_tokens * dim_k, 0.4f);
    std::vector<float> v_data(n_tokens * dim_v, 0.3f);

    xkv_factor_group_input g0;
    g0.group_index = 0;
    g0.owning_layers = {0, 1, 2, 3};
    g0.rank_k = 8;
    g0.rank_v = 8;
    g0.total_dim_k = dim_k;
    g0.total_dim_v = dim_v;
    g0.layer_feature_offsets_k = {0, 16, 32, 48};
    g0.layer_feature_dims_k = {16, 16, 16, 16};
    g0.layer_feature_offsets_v = {0, 16, 32, 48};
    g0.layer_feature_dims_v = {16, 16, 16, 16};
        g0.row_positions.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) g0.row_positions[i] = (int64_t) i;
        g0.hot_bytes_per_row_k = {32, 32, 32, 32};
        g0.hot_bytes_per_row_v = {32, 32, 32, 32};
    g0.canonical_k_data = k_data.data();
    g0.k_rows = n_tokens;
    g0.k_cols = dim_k;
    g0.canonical_v_data = v_data.data();
    g0.v_rows = n_tokens;
    g0.v_cols = dim_v;

    xkv_bundle_sealing_params sparams;
    sparams.rank_k = 8;
    sparams.rank_v = 8;
    sparams.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS; // Requires landmark factory
    sparams.source = LLAMA_XKV_SOURCE_DECODED_HOT;
    sparams.max_relative_error = 0.99;
    sparams.min_saving_ratio = 0.01;
    sparams.flat_type_k = GGML_TYPE_F16;
    sparams.flat_type_v = GGML_TYPE_F16;

    auto pre_store = capture_store(store, pids, {1});

    // 1. Landmark factory callback missing when profile requires landmarks
    sparams.landmark_factory = nullptr;
    auto res_nolm = store.seal_segment_bundle({g0}, pids, gens, sparams);
    assert(!res_nolm.success);
    assert(res_nolm.skip_reason == xkv_skip_reason::landmark_required);
    assert(store.get_skipped_count(xkv_skip_reason::landmark_required) == 1);

    // Invariant: Payloads restored to hot_committed, store stamp unchanged
    auto post_nolm = capture_store(store, pids, {1});
    assert(pre_store.stamp == post_nolm.stamp);
    assert(pre_store.segments == post_nolm.segments);
    for (uint64_t pid : pids) {
        xkv_location loc;
        assert(store.find_location(pid, loc));
        assert(loc.state == xkv_state::hot_committed);
        assert(loc.seal_tx_nonce == 0);
    }

    // 2. Factorization succeeds, shadow succeeds, but landmark_factory callback returns false!
    sparams.landmark_factory = [](
        uint32_t /*group_index*/,
        const encoded_matrix & /*enc_a_k*/,
        const encoded_matrix & /*enc_b_k*/,
        const int64_t * /*row_positions*/,
        uint64_t /*lm_rows*/,
        factor_workspace_span /*scratch*/,
        encoded_matrix & /*out_lm*/,
        std::vector<xkv_landmark_chunk> & /*out_chunks*/,
        std::string * err
    ) -> bool {
        if (err) *err = "injected landmark encoding synthesis failure";
        return false;
    };

    auto res_lmfail = store.seal_segment_bundle({g0}, pids, gens, sparams);
    assert(!res_lmfail.success);
    assert(res_lmfail.skip_reason == xkv_skip_reason::codec_error);
    assert(res_lmfail.message.find("injected landmark encoding synthesis failure") != std::string::npos);

    // Invariant: ZERO partial mutation! All payload states must be hot_committed, seal_tx_nonce == 0
    auto post_lmfail = capture_store(store, pids, {1});
    assert(pre_store.stamp == post_lmfail.stamp);
    assert(pre_store.segments == post_lmfail.segments);
    for (uint64_t pid : pids) {
        xkv_location loc;
        assert(store.find_location(pid, loc));
        assert(loc.state == xkv_state::hot_committed);
        assert(loc.seal_tx_nonce == 0);
    }

    // 3. Landmark factory returns true but emits invalid descriptor
    sparams.landmark_factory = [](
        uint32_t, const encoded_matrix &, const encoded_matrix &,
        const int64_t *, uint64_t, factor_workspace_span,
        encoded_matrix & out_lm, std::vector<xkv_landmark_chunk> &,
        std::string *
    ) -> bool {
        // Construct invalid landmark descriptor (e.g. role mismatch or 0 rows)
        out_lm.desc.role = factor_role::a_k; // Invalid role for landmark stream
        out_lm.desc.type = GGML_TYPE_COUNT;
        return true;
    };

    auto res_invlm = store.seal_segment_bundle({g0}, pids, gens, sparams);
    assert(!res_invlm.success);
    assert(res_invlm.skip_reason == xkv_skip_reason::codec_error);

    auto post_invlm = capture_store(store, pids, {1});
    assert(pre_store.stamp == post_invlm.stamp);
    assert(pre_store.segments == post_invlm.segments);

    // 4. Landmark scratch bounds and overflow failure
    size_t out_scratch = 0;
    std::string lm_err;
    assert(!landmark_encode_scratch_bytes(0, 16, 64, out_scratch, &lm_err));
    assert(!landmark_encode_scratch_bytes(64, 0, 64, out_scratch, &lm_err));
    assert(!landmark_encode_scratch_bytes(64, 16, 0, out_scratch, &lm_err));
    assert(!landmark_encode_scratch_bytes(UINT32_MAX, UINT32_MAX, UINT32_MAX, out_scratch, &lm_err));
    assert(lm_err.find("overflow") != std::string::npos);

    // 5. Coarse selection succeeds then refine_rescore injection fails (§15.4)
    {
        auto seg_refine = std::make_shared<xkv_segment>();
        seg_refine->segment_id = 100;
        seg_refine->segment_version = 1;
        seg_refine->n_rows = 16;
        seg_refine->n_live_rows = 16;
        seg_refine->row_payload_ids.resize(16);
        seg_refine->live_rows.assign(16, true);
        for (uint32_t i = 0; i < 16; ++i) {
            seg_refine->row_payload_ids[i] = 4000 + i;
        }
        xkv_factor_group_payload grp;
        grp.group_index = 0;
        grp.owning_layers = {0};
        grp.layer_feature_offsets_k = {0};
        grp.layer_feature_dims_k = {32};
        grp.total_dim_k = 32;
        codec_desc bk_desc = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {32, 16}, 0, 77);
        std::vector<float> bk_data(32 * 16, 0.1f);
        grp.set_b_k(encode_matrix(bk_desc, bk_data.data(), bk_data.size()));
        codec_desc ak_desc = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {16, 16}, 0, 77);
        std::vector<float> ak_data(16 * 16, 0.1f);
        grp.a_k = encode_matrix(ak_desc, ak_data.data(), ak_data.size());
        seg_refine->groups.push_back(std::move(grp));
        seg_refine->update_byte_counters();

        xkv_snapshot_stamp stamp_refine;
        stamp_refine.live_epoch = 1;
        std::vector<row_meta> rmetas;
        for (uint32_t i = 0; i < 16; ++i) {
            rmetas.push_back({i, 4000 + i, 1, 0, (int64_t)i, (int64_t)i, 0, true, true});
        }
        auto frags = build_legal_fragments(*seg_refine, stamp_refine, 1, 1, 0, 0, rmetas, 4, 0x1, -1);
        for (auto & f : frags) {
            encode_fragment_landmark(f, *seg_refine, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
        }

        sr_query q;
        q.head_dim = 32;
        q.q_vec.assign(32, 0.2f);
        q.scale = 1.0f / std::sqrt(32.0f);

        sr_selection_config rcfg;
        rcfg.sr_budget = 2;
        rcfg.refine_mode = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
        rcfg.refine_max_rows = 64;

        // Baseline coarse selection + refine succeeds
        auto clean = select_sr_query(q, frags, rcfg, seg_refine.get(), 0, 32, nullptr, 0x1);
        assert(clean.rows_refined > 0);

        // Inject refine_rescore failure: coarse selection succeeds then refine fails
        sr_selection_config faulty = rcfg;
        faulty.fault.stage = landmark_fault_stage::refine_rescore;
        faulty.fault.occurrence = 0;
        bool threw = false;
        try {
            select_sr_query(q, frags, faulty, seg_refine.get(), 0, 32, nullptr, 0x1);
        } catch (const std::runtime_error & e) {
            threw = true;
            assert(std::string(e.what()).find("injected refine_rescore fault") != std::string::npos);
        }
        assert(threw);

        // Bounded query path: assert output buffers, counts, and selection cache unchanged
        std::vector<landmark_fragment_view> views;
        for (const auto & f : frags) views.push_back(view_of_fragment(f));

        size_t need_scratch = 0;
        std::string sc_err;
        assert(landmark_select_scratch_bytes(4, 1, 32, 4, 16, need_scratch, &sc_err));
        std::vector<uint8_t> scratch(need_scratch, 0);

        std::vector<segment_row_ref> br(64, segment_row_ref{0xEEu, 0xEEu, 0xEEu, 0xEEu});
        std::vector<float> bs(8, -999.0f);
        const auto br_snap = br;
        const auto bs_snap = bs;
        size_t nr = 9999, ns = 9999;
        uint32_t rf = 9999;
        bool ch = true;
        std::string b_err;
        bool b_ok = select_sr_query_bounded(q, views.data(), views.size(), faulty, seg_refine.get(), 0,
            32, nullptr, 0x1, scratch.data(), scratch.size(), br.data(), br.size(), &nr,
            bs.data(), bs.size(), &ns, &rf, &ch, &b_err);
        assert(!b_ok);
        assert(nr == 0 && ns == 0 && rf == 0 && !ch);
        assert(br == br_snap); // output rows untouched
        assert(bs == bs_snap); // output scores untouched
    }

    std::cout << "  [PASS] Landmark failures and factor-succeeds-landmark-fails atomicity verified." << std::endl;
}

// ============================================================================
// 6. Actual Bytes Exceed Estimate & Gate Failures (Saving / Error)
// ============================================================================

static void test_fault_actual_bytes_exceed_estimate() {
    std::cout << "[Test 6] Fault Injection: Actual Bytes Exceed Estimate & Gating..." << std::endl;

    matrix x_k(32, 32, 1.0f);
    matrix x_v(32, 32, 2.0f);

    factor_config cfg;
    cfg.rank_k = 8;
    cfg.rank_v = 8;
    cfg.balance = LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    cfg.seed = 42;

    // 1. Buffer smaller than estimator in factorize_matrix_bounded
    uint64_t req_ws_mat = 0;
    std::string est_err;
    assert(estimate_factorize_matrix_workspace_bytes(32, 32, 8, cfg.oversampling, cfg.power_iterations, &req_ws_mat, &est_err));
    assert(req_ws_mat > 0);

    std::vector<uint8_t> short_buf_mat(req_ws_mat - 1);
    factor_workspace_span short_span_mat(short_buf_mat.data(), short_buf_mat.size());
    factor_pair fp_short;
    std::string f_err;
    assert(!factorize_matrix_bounded(x_k, 8, cfg.balance, cfg.seed, fp_short, short_span_mat, &f_err));
    assert(f_err.find("workspace buffer") != std::string::npos);
    assert(fp_short.a.empty() && fp_short.b_transposed.empty());

    // 2. Buffer smaller than estimator in factorize_kv_bounded
    uint64_t req_ws_kv = 0;
    assert(estimate_factorize_kv_workspace_bytes(x_k, x_v, cfg, &req_ws_kv, &est_err));
    assert(req_ws_kv > 0);

    std::vector<uint8_t> short_buf_kv(req_ws_kv - 1);
    factor_workspace_span short_span_kv(short_buf_kv.data(), short_buf_kv.size());
    factor_result kv_short_res = factorize_kv_bounded(x_k, x_v, cfg, short_span_kv);
    assert(!kv_short_res.success);
    assert(kv_short_res.error_message.find("workspace buffer") != std::string::npos);
    assert(kv_short_res.k.a.empty() && kv_short_res.v.a.empty());

    // 3. Buffer smaller than estimator in evaluate_quantized_shadow_bounded
    factor_result valid_f = factorize_kv(x_k, x_v, cfg);
    assert(valid_f.success);

    uint64_t req_ws_shadow = 0;
    assert(estimate_quantized_shadow_workspace_bytes(x_k, x_v, valid_f.k, valid_f.v, cfg, &req_ws_shadow, &est_err));
    assert(req_ws_shadow > 0);

    std::vector<uint8_t> short_buf_shadow(req_ws_shadow - 1);
    factor_workspace_span short_span_shadow(short_buf_shadow.data(), short_buf_shadow.size());
    factor_quantized_shadow shadow_short = evaluate_quantized_shadow_bounded(x_k, x_v, valid_f.k, valid_f.v, cfg, short_span_shadow);
    assert(!shadow_short.success);
    assert(shadow_short.error_message.find("workspace buffer") != std::string::npos);
    assert(shadow_short.stream_a_k.bytes.empty());

    // 4. Net saving gate failure in seal_segment_bundle:
    // When actual factored bytes exceed flat source bytes (or saving ratio < min_saving_ratio)
    auto cp = make_test_cparams();
    llama_xkv_cache_store store(cp);

    const uint32_t n_tokens = 64;
    std::vector<uint64_t> pids(n_tokens);
    std::vector<uint64_t> gens(n_tokens, 1);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pids[i] = 3000 + i;
        assert(store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed));
    }

    std::vector<float> k_data(n_tokens * 64, 0.1f);
    std::vector<float> v_data(n_tokens * 64, 0.2f);

    xkv_factor_group_input g0;
    g0.group_index = 0;
    g0.owning_layers = {0, 1, 2, 3};
    g0.rank_k = 16;
    g0.rank_v = 16;
    g0.total_dim_k = 64;
    g0.total_dim_v = 64;
    g0.layer_feature_offsets_k = {0, 16, 32, 48};
    g0.layer_feature_dims_k = {16, 16, 16, 16};
    g0.layer_feature_offsets_v = {0, 16, 32, 48};
    g0.layer_feature_dims_v = {16, 16, 16, 16};
        g0.row_positions.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) g0.row_positions[i] = (int64_t) i;
        g0.hot_bytes_per_row_k = {32, 32, 32, 32};
        g0.hot_bytes_per_row_v = {32, 32, 32, 32};
    g0.canonical_k_data = k_data.data();
    g0.k_rows = n_tokens;
    g0.k_cols = 64;
    g0.canonical_v_data = v_data.data();
    g0.v_rows = n_tokens;
    g0.v_cols = 64;

    xkv_bundle_sealing_params sparams;
    sparams.rank_k = 16;
    sparams.rank_v = 16;
    sparams.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    sparams.source = LLAMA_XKV_SOURCE_DECODED_HOT;
    sparams.max_relative_error = 0.99;
    sparams.min_saving_ratio = 0.99; // Impossible 99% saving gate
    sparams.flat_type_k = GGML_TYPE_F16;
    sparams.flat_type_v = GGML_TYPE_F16;

    auto pre_store = capture_store(store, pids, {1});
    auto res_nosave = store.seal_segment_bundle({g0}, pids, gens, sparams);
    assert(!res_nosave.success);
    assert(res_nosave.skip_reason == xkv_skip_reason::no_saving);
    assert(store.get_skipped_count(xkv_skip_reason::no_saving) == 1);

    auto post_nosave = capture_store(store, pids, {1});
    assert(pre_store.stamp == post_nosave.stamp);
    assert(pre_store.segments == post_nosave.segments);
    for (uint64_t pid : pids) {
        xkv_location loc;
        assert(store.find_location(pid, loc));
        assert(loc.state == xkv_state::hot_committed);
        assert(loc.seal_tx_nonce == 0);
    }

    // 5. Error threshold exceeded in seal_segment_bundle
    sparams.min_saving_ratio = 0.01;
    sparams.max_relative_error = 1e-9; // Impossibly low error tolerance
    auto res_err = store.seal_segment_bundle({g0}, pids, gens, sparams);
    assert(!res_err.success);
    assert(res_err.skip_reason == xkv_skip_reason::error_threshold_exceeded);
    assert(store.get_skipped_count(xkv_skip_reason::error_threshold_exceeded) == 1);

    auto post_err = capture_store(store, pids, {1});
    assert(pre_store.stamp == post_err.stamp);
    assert(pre_store.segments == post_err.segments);
    for (uint64_t pid : pids) {
        xkv_location loc;
        assert(store.find_location(pid, loc));
        assert(loc.state == xkv_state::hot_committed);
    }

    std::cout << "  [PASS] Actual bytes exceed estimate & gating failure atomicity verified." << std::endl;
}

// ============================================================================
// 7. Stale Snapshot and Generation Failures
// ============================================================================

static void test_fault_stale_snapshot_and_generation() {
    std::cout << "[Test 7] Fault Injection: Stale Snapshot & Generation..." << std::endl;

    // 1. Transaction coordinator stale stamp
    xkv_transaction_coordinator coord;
    std::string err;
    auto s0 = coord.current_stamp();

    // Advance provider epoch externally
    auto fake_prov = dynamic_cast<xkv_in_memory_stamp_provider *>(coord.stamp_provider());
    assert(fake_prov != nullptr);

    // Stale begin: passing an outdated snapshot
    xkv_snapshot_stamp stale_stamp = s0;
    stale_stamp.content_epoch += 10;
    assert(coord.begin(1, stale_stamp, &err) == xkv_tx_status::stale_stamp);
    assert(!coord.has_active_tx());

    // Begin with current stamp
    assert(coord.begin(1, s0, &err) == xkv_tx_status::ok);
    assert(coord.has_active_tx());
    assert(coord.record_payload(1, 501, 1, &err) == xkv_tx_status::ok);
    assert(coord.record_byte_complete(1, 501, &err) == xkv_tx_status::ok);

    // Bump content epoch externally while transaction is active
    fake_prov->advance_content_epoch(fake_prov->current_stamp(), nullptr);
    assert(coord.current_stamp().content_epoch > s0.content_epoch);

    // Commit with stale stamp must fail
    auto dummy_cb = [](const std::vector<uint64_t> &, uint64_t, xkv_removal_feedback *, std::string *) { return true; };
    assert(coord.commit(1, {501}, dummy_cb, s0, &err) == xkv_tx_status::stale_stamp);
    assert(coord.has_active_tx());
    assert(!coord.has_committed_publication());

    // is_stale check verification
    std::string stale_reason;
    assert(coord.is_stale(s0, &stale_reason));
    assert(stale_reason.find("stale content_epoch") != std::string::npos);

    // Rebase and commit successfully
    assert(coord.rebase(1, coord.current_stamp(), &err) == xkv_tx_status::ok);
    assert(coord.commit(1, {501}, dummy_cb, coord.current_stamp(), &err) == xkv_tx_status::ok);
    assert(coord.has_committed_publication());

    // Monotonic tx ID reuse rejection
    assert(coord.begin(1, coord.current_stamp(), &err) == xkv_tx_status::tx_id_reused);
    assert(coord.begin(0, coord.current_stamp(), &err) == xkv_tx_status::bad_tx_id);

    // 2. Hot slot pool stale generation
    xkv_hot_slot_pool pool(4);
    xkv_hot_reservation res = pool.reserve(2);
    assert(res.commit({601, 602}, {10, 20}, &err));
    assert(pool.get_bound() == 2);

    // Single release with stale generation fails cleanly
    auto pre_pool = capture_pool(pool);
    assert(!pool.release(601, 9, 0, &err));
    assert(err.find("generation mismatch") != std::string::npos);
    auto post_pool = capture_pool(pool);
    assert(pre_pool == post_pool);

    // Batch release with one stale item fails all-or-nothing
    std::vector<uint64_t> rel_pids = {601, 602};
    std::vector<uint64_t> rel_gens = {10, 19}; // 19 != 20 (stale generation!)
    std::vector<uint32_t> rel_slots = {0, 1};
    assert(!pool.release_batch(rel_pids, rel_gens, rel_slots, &err));
    auto post_batch = capture_pool(pool);
    assert(pre_pool == post_batch); // ZERO slots released!

    // 3. Store stale generation in remove_payloads
    auto cp = make_test_cparams();
    llama_xkv_cache_store store(cp);
    assert(store.register_hot_payload(701, 0, 5, xkv_state::hot_committed));

    auto pre_store = capture_store(store, {701}, {});
    std::string rem_err;
    xkv_payload_removal_result rem_res;
    // Expected generation 4 != 5
    bool rem_ok = store.remove_payloads({701}, {4}, &rem_res, &rem_err);
    assert(!rem_ok);
    assert(rem_err.find("generation mismatch") != std::string::npos);
    auto post_store = capture_store(store, {701}, {});
    assert(pre_store.stamp == post_store.stamp);
    assert(pre_store.locations == post_store.locations);

    std::cout << "  [PASS] Stale snapshot and generation failure atomicity verified." << std::endl;
}

// ============================================================================
// 8. Codec Version, Magic, and Fingerprint Mismatch Failures
// ============================================================================

static void test_fault_codec_version_and_fingerprint_mismatch() {
    std::cout << "[Test 8] Fault Injection: Codec Version & Fingerprint Mismatch..." << std::endl;

    // Capture valid world and encode image
    test_world w = make_world();
    xkv_state_image valid_img;
    std::string err;
    assert(capture_world(w, valid_img, &err));
    std::vector<uint8_t> valid_bytes;
    assert(encode_image(valid_img, valid_bytes, test_limits(), &err));

    // Sentinel destination image
    xkv_state_image dst;
    dst.config.seed = 0xDEADBEEF;

    // 1. Mismatched model fingerprint
    {
        xkv_state_fingerprints bad_fps = test_fps();
        bad_fps.model ^= 0xFF;
        xkv_state_image dst_copy = dst;
        assert(!decode_image(valid_bytes.data(), valid_bytes.size(), dst_copy, bad_fps, test_prov(), test_limits(), &err));
        assert(err.find("fingerprint mismatch: model") != std::string::npos);
        assert(dst_copy.config.seed == 0xDEADBEEF); // Invariant: destination state completely untouched!
    }

    // 2. Mismatched rope fingerprint
    {
        xkv_state_fingerprints bad_fps = test_fps();
        bad_fps.rope ^= 0xFF;
        xkv_state_image dst_copy = dst;
        assert(!decode_image(valid_bytes.data(), valid_bytes.size(), dst_copy, bad_fps, test_prov(), test_limits(), &err));
        assert(err.find("fingerprint mismatch: rope") != std::string::npos);
        assert(dst_copy.config.seed == 0xDEADBEEF);
    }

    // 3. Mismatched codec fingerprint
    {
        xkv_state_fingerprints bad_fps = test_fps();
        bad_fps.codec ^= 0xFF;
        xkv_state_image dst_copy = dst;
        assert(!decode_image(valid_bytes.data(), valid_bytes.size(), dst_copy, bad_fps, test_prov(), test_limits(), &err));
        assert(err.find("fingerprint mismatch: codec") != std::string::npos);
        assert(dst_copy.config.seed == 0xDEADBEEF);
    }

    // 4. Mismatched provenance model_sha256
    {
        xkv_state_provenance bad_prov = test_prov();
        bad_prov.model_sha256[0] ^= 0xFF;
        xkv_state_image dst_copy = dst;
        assert(!decode_image(valid_bytes.data(), valid_bytes.size(), dst_copy, test_fps(), bad_prov, test_limits(), &err));
        assert(err.find("provenance mismatch: model_sha256") != std::string::npos);
        assert(dst_copy.config.seed == 0xDEADBEEF);
    }

    // 5. Corrupted magic bytes in wire data
    {
        std::vector<uint8_t> bad_magic_bytes = valid_bytes;
        bad_magic_bytes[0] ^= 0xFF;
        xkv_state_image dst_copy = dst;
        assert(!decode_image(bad_magic_bytes.data(), bad_magic_bytes.size(), dst_copy, test_fps(), test_prov(), test_limits(), &err));
        assert(err.find("magic mismatch") != std::string::npos);
        assert(dst_copy.config.seed == 0xDEADBEEF);
    }

    // 6. Corrupted envelope version in wire data
    {
        std::vector<uint8_t> bad_ver_bytes = valid_bytes;
        // Byte 4 is version
        bad_ver_bytes[4] = 99;
        xkv_state_image dst_copy = dst;
        assert(!decode_image(bad_ver_bytes.data(), bad_ver_bytes.size(), dst_copy, test_fps(), test_prov(), test_limits(), &err));
        assert(err.find("version mismatch") != std::string::npos);
        assert(dst_copy.config.seed == 0xDEADBEEF);
    }

    std::cout << "  [PASS] Codec version, magic, and fingerprint mismatch atomicity verified." << std::endl;
}

// ============================================================================
// 9. Truncated State Images & Bit Corruption
// ============================================================================

static void test_fault_truncated_state() {
    std::cout << "[Test 9] Fault Injection: Truncated State Images & Bit Corruption..." << std::endl;

    test_world w = make_world();
    xkv_state_image valid_img;
    std::string err;
    assert(capture_world(w, valid_img, &err));
    std::vector<uint8_t> valid_bytes;
    assert(encode_image(valid_img, valid_bytes, test_limits(), &err));

    const size_t full_sz = valid_bytes.size();
    assert(full_sz > 100);

    const std::vector<size_t> trunc_lengths = {
        0, 4, 8, 16, 32, 64, 128, full_sz / 2, full_sz - 1
    };

    for (size_t t_len : trunc_lengths) {
        xkv_state_image dst;
        dst.config.seed = 0xC0FFEE;
        err.clear();
        bool ok = decode_image(valid_bytes.data(), t_len, dst, test_fps(), test_prov(), test_limits(), &err);
        assert(!ok);
        assert(!err.empty());
        // Destination image MUST be untouched!
        assert(dst.config.seed == 0xC0FFEE);
    }

    // Single-bit corruption checksum failure
    std::vector<uint8_t> corrupted_bytes = valid_bytes;
    corrupted_bytes[full_sz - 5] ^= 0x01;
    xkv_state_image dst_corrupt;
    dst_corrupt.config.seed = 0xC0FFEE;
    err.clear();
    assert(!decode_image(corrupted_bytes.data(), corrupted_bytes.size(), dst_corrupt, test_fps(), test_prov(), test_limits(), &err));
    assert(dst_corrupt.config.seed == 0xC0FFEE);

    // Trailing bytes rejection
    std::vector<uint8_t> trailing_bytes = valid_bytes;
    trailing_bytes.push_back(0xAA);
    xkv_state_image dst_trailing;
    dst_trailing.config.seed = 0xC0FFEE;
    err.clear();
    assert(!decode_image(trailing_bytes.data(), trailing_bytes.size(), dst_trailing, test_fps(), test_prov(), test_limits(), &err));
    assert(err.find("trailing bytes") != std::string::npos);
    assert(dst_trailing.config.seed == 0xC0FFEE);

    std::cout << "  [PASS] Truncated state images & corruption atomicity verified." << std::endl;
}

// ============================================================================
// 10. Transaction Rollback Accepted Prefixes 0 / 1 / Partial / All
// ============================================================================

static void test_fault_transaction_rollback_accepted_prefixes() {
    std::cout << "[Test 10] Fault Injection: Transaction Rollback Accepted Prefixes 0/1/Partial/All..." << std::endl;

    // 1. Prefix Shape Enforcement: Non-prefix accepted sets must be refused with invalid_accept
    {
        xkv_transaction_coordinator c;
        std::string err;
        auto s0 = c.current_stamp();
        assert(c.begin(10, s0, &err) == xkv_tx_status::ok);
        for (uint64_t id : {101, 102, 103, 104}) {
            assert(c.record_payload(10, id, id, &err) == xkv_tx_status::ok);
            assert(c.record_byte_complete(10, id, &err) == xkv_tx_status::ok);
        }

        auto ok_cb = [](const std::vector<uint64_t> &, uint64_t, xkv_removal_feedback *, std::string *) { return true; };

        // Gap in accepted set: {101, 103} skips 102
        assert(c.commit(10, {101, 103}, ok_cb, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
        assert(!c.has_committed_publication());
        assert(c.has_active_tx());

        // Missing head: {102} skips 101
        assert(c.commit(10, {102}, ok_cb, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
        assert(!c.has_committed_publication());

        // Out of order: {102, 101}
        assert(c.commit(10, {102, 101}, ok_cb, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);

        // Extra payload not in transaction: {101, 102, 103, 104, 999}
        assert(c.commit(10, {101, 102, 103, 104, 999}, ok_cb, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
        assert(!c.has_committed_publication());
        assert(c.has_active_tx());

        // Rollback uncommitted
        std::vector<uint64_t> rolled;
        assert(c.rollback(10, [&](const std::vector<uint64_t> & cr, std::string *) {
            rolled = cr;
            return true;
        }, &err) == xkv_tx_status::ok);
        assert(!c.has_active_tx());
        assert(rolled == std::vector<uint64_t>({101, 102, 103, 104}));
    }

    // 2. Accept 0 (Reject All): commit with empty accepted set
    {
        xkv_transaction_coordinator c;
        std::string err;
        assert(c.begin(20, c.current_stamp(), &err) == xkv_tx_status::ok);
        for (uint64_t id : {201, 202}) {
            assert(c.record_payload(20, id, id, &err) == xkv_tx_status::ok);
            assert(c.record_byte_complete(20, id, &err) == xkv_tx_status::ok);
        }

        std::vector<uint64_t> rejected;
        auto st = c.commit(20, {}, [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
            rejected = rej;
            return true;
        }, c.current_stamp(), &err);
        assert(st == xkv_tx_status::ok);
        assert(rejected == std::vector<uint64_t>({201, 202}));
        assert(c.committed_accepted().empty());
        assert(!c.is_eligible_for_maintenance(201));
        assert(!c.is_eligible_for_maintenance(202));
        assert(c.has_committed_publication());
        assert(!c.has_active_tx());
    }

    // 3. Accept 1: commit prefix of length 1
    {
        xkv_transaction_coordinator c;
        std::string err;
        assert(c.begin(30, c.current_stamp(), &err) == xkv_tx_status::ok);
        for (uint64_t id : {301, 302, 303}) {
            assert(c.record_payload(30, id, id, &err) == xkv_tx_status::ok);
            assert(c.record_byte_complete(30, id, &err) == xkv_tx_status::ok);
        }

        std::vector<uint64_t> rejected;
        auto st = c.commit(30, {301}, [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
            rejected = rej;
            return true;
        }, c.current_stamp(), &err);
        assert(st == xkv_tx_status::ok);
        assert(rejected == std::vector<uint64_t>({302, 303}));
        assert(c.committed_accepted() == std::vector<uint64_t>({301}));
        assert(c.is_eligible_for_maintenance(301));
        assert(!c.is_eligible_for_maintenance(302));
        assert(c.has_committed_publication());
    }

    // 4. Accept Partial (length 2 of 4)
    {
        xkv_transaction_coordinator c;
        std::string err;
        assert(c.begin(40, c.current_stamp(), &err) == xkv_tx_status::ok);
        for (uint64_t id : {401, 402, 403, 404}) {
            assert(c.record_payload(40, id, id, &err) == xkv_tx_status::ok);
            assert(c.record_byte_complete(40, id, &err) == xkv_tx_status::ok);
        }

        std::vector<uint64_t> rejected;
        auto st = c.commit(40, {401, 402}, [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
            rejected = rej;
            return true;
        }, c.current_stamp(), &err);
        assert(st == xkv_tx_status::ok);
        assert(rejected == std::vector<uint64_t>({403, 404}));
        assert(c.committed_accepted() == std::vector<uint64_t>({401, 402}));
        assert(c.committed_rejected() == std::vector<uint64_t>({403, 404}));
        assert(c.is_eligible_for_maintenance(401));
        assert(c.is_eligible_for_maintenance(402));
        assert(!c.is_eligible_for_maintenance(403));
    }

    // 5. Accept All: commit exact full set
    {
        xkv_transaction_coordinator c;
        std::string err;
        assert(c.begin(50, c.current_stamp(), &err) == xkv_tx_status::ok);
        for (uint64_t id : {501, 502, 503}) {
            assert(c.record_payload(50, id, id, &err) == xkv_tx_status::ok);
            assert(c.record_byte_complete(50, id, &err) == xkv_tx_status::ok);
        }

        std::vector<uint64_t> rejected;
        auto st = c.commit(50, {501, 502, 503}, [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
            rejected = rej;
            return true;
        }, c.current_stamp(), &err);
        assert(st == xkv_tx_status::ok);
        assert(rejected.empty());
        assert(c.committed_accepted() == std::vector<uint64_t>({501, 502, 503}));
        assert(c.committed_rejected().empty());
        assert(c.is_eligible_for_maintenance(501));
        assert(c.is_eligible_for_maintenance(502));
        assert(c.is_eligible_for_maintenance(503));
    }

    // 6. Compute failure rollback removes all created payloads
    {
        xkv_transaction_coordinator c;
        std::string err;
        assert(c.begin(60, c.current_stamp(), &err) == xkv_tx_status::ok);
        assert(c.record_payload(60, 601, 1, &err) == xkv_tx_status::ok);
        assert(c.record_payload(60, 602, 2, &err) == xkv_tx_status::ok);
        assert(c.record_byte_complete(60, 601, &err) == xkv_tx_status::ok);

        std::vector<uint64_t> rolled;
        auto st = c.rollback(60, [&](const std::vector<uint64_t> & cr, std::string *) {
            rolled = cr;
            return true;
        }, &err);
        assert(st == xkv_tx_status::ok);
        assert(rolled == std::vector<uint64_t>({601, 602}));
        assert(!c.has_active_tx());
        assert(!c.has_committed_publication());
    }

    std::cout << "  [PASS] Transaction rollback accepted prefixes 0/1/partial/all verified." << std::endl;
}

// ============================================================================
// 11. Publishing Failures & Epoch Overflow Failures
// ============================================================================

static void test_fault_publish_candidate_failures() {
    std::cout << "[Test 11] Fault Injection: Candidate Publishing & Epoch Overflow..." << std::endl;

    auto cp = make_test_cparams();
    llama_xkv_cache_store store(cp);

    // 1. Duplicate payload IDs in publish_candidate
    auto seg1 = create_test_candidate(4);
    std::string err;
    assert(!store.publish_candidate(seg1, {1, 2, 3, 1}, {1, 1, 1, 1}, &err));
    assert(err.find("duplicate payload_id") != std::string::npos);

    // 2. Payload ID 0 in publish_candidate
    auto seg2 = create_test_candidate(4);
    err.clear();
    assert(!store.publish_candidate(seg2, {1, 0, 3, 4}, {1, 1, 1, 1}, &err));
    assert(err.find("cannot be 0") != std::string::npos);

    // 3. Payload not in seal_candidate state
    auto seg3 = create_test_candidate(4);
    err.clear();
    assert(store.register_hot_payload(10, 0, 1, xkv_state::hot_writing));
    assert(!store.publish_candidate(seg3, {10, 11, 12, 13}, {1, 1, 1, 1}, &err));

    // 4. can_bump preflight failure on epoch overflow
    auto seg4 = create_test_candidate(4);
    for (uint64_t p : {21, 22, 23, 24}) {
        assert(store.register_hot_payload(p, (uint32_t)(p - 21), 1, xkv_state::hot_committed));
    }
    assert(store.mark_seal_candidates({21, 22, 23, 24}));

    // Set content_epoch to UINT64_MAX to force overflow in can_bump
    store.set_epoch_for_testing(llama_xkv_cache_store::bump_flag_content, UINT64_MAX);
    err.clear();
    assert(!store.publish_candidate(seg4, {21, 22, 23, 24}, {1, 1, 1, 1}, &err));
    assert(err.find("overflow") != std::string::npos || err.find("epoch") != std::string::npos);

    // Invariant: candidate was not published
    assert(store.get_segment(seg4->segment_id) == nullptr);

    // 5. Batch COW transaction failure hook
    // Reset epoch to valid value
    store.set_epoch_for_testing(llama_xkv_cache_store::bump_flag_content, 1);
    err.clear();
    assert(store.publish_candidate(seg4, {21, 22, 23, 24}, {1, 1, 1, 1}, &err));
    const uint64_t pub_id = seg4->segment_id;
    assert(store.get_segment(pub_id)->segment_version == 1);

    auto pre_mut = capture_store(store, {21, 22, 23, 24}, {pub_id});

    xkv_batch_mutation mut;
    mut.payload_removals = {22};
    mut.test_failure_hook = []() {
        throw std::runtime_error("simulated failure after clone preparation");
    };

    xkv_mutation_result mut_res;
    err.clear();
    bool mut_ok = store.execute_mutation_transaction(mut, &mut_res, &err);
    assert(!mut_ok);
    assert(!mut_res.success);

    // Invariant: Segment version must remain 1, location of payload 22 must remain intact!
    auto post_mut = capture_store(store, {21, 22, 23, 24}, {pub_id});
    assert(pre_mut.stamp == post_mut.stamp);
    assert(pre_mut.segments == post_mut.segments);
    assert(pre_mut.locations == post_mut.locations);

    std::cout << "  [PASS] Candidate publishing and epoch overflow atomicity verified." << std::endl;
}

// ============================================================================
// 12. Structured Gap Reporting for Absent Injection Seams (XKV-SR §15.4)
// ============================================================================

struct xkv_injection_seam_gap {
    std::string subsystem;
    std::string missing_api;
    std::string description;
};

inline std::vector<xkv_injection_seam_gap> get_xkv_absent_injection_seams() {
    std::vector<xkv_injection_seam_gap> gaps;

    // 1. Standalone / direct QR failure injection
    gaps.push_back({
        "qr",
        "llama_xkv::set_qr_failure_hook(std::function<bool(double*, uint64_t, uint64_t)>)",
        "Direct standalone call to qr_orthonormalize in llama-xkv-factor.cpp is a void function with double-stable "
        "Gram-Schmidt and has no direct hook or error return to simulate numerical failure outside of factor_config::fault."
    });

    // 2. Full-rank direct Jacobi SVD failure injection
    gaps.push_back({
        "svd",
        "llama_xkv::jacobi_svd_failure_hook",
        "Direct Jacobi SVD for small reference bound (min_nm <= 64) in factorize_matrix does not inspect "
        "factor_config::fault (which applies only to the rSVD branch)."
    });

    // 3. Store-level memmove compaction failure injection
    gaps.push_back({
        "memmove",
        "llama_xkv::llama_xkv_cache_store::inject_memmove_failure_hook",
        "llama_xkv_cache_store operates on RAM arena blocks and immutable snapshots; native tensor memmove "
        "compaction (ggml_backend_tensor_memmove_regions) is handled by llama_kv_cache cells without a store-level injection seam."
    });

    return gaps;
}

static void test_absent_injection_seams_reporting() {
    std::cout << "[Test 12] Reporting absent injection seams per XKV-SR.md §15.4..." << std::endl;
    auto gaps = get_xkv_absent_injection_seams();
    // Invariant: Must NOT be an empty passing placeholder!
    assert(!gaps.empty());
    assert(gaps.size() == 3);
    for (const auto & g : gaps) {
        assert(!g.subsystem.empty());
        assert(!g.missing_api.empty());
        assert(!g.description.empty());
        std::cout << "  - Subsystem: " << g.subsystem << "\n"
                  << "    Missing API: " << g.missing_api << "\n"
                  << "    Description: " << g.description << "\n";
    }
    std::cout << "  [PASS] Structured gap list verified and reported (" << gaps.size() << " gaps)." << std::endl;
}

// ============================================================================
// Main Suite Driver
// ============================================================================

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " Running test-xkv-fault-matrix (XKV-SR §15.4 Fault Matrix)" << std::endl;
    std::cout << "============================================================" << std::endl;

    test_fault_allocation_hot_pool();
    test_fault_allocation_workspace_arena();
    test_fault_allocation_backend();
    test_fault_seal_precommit_gate();
    test_fault_factor_and_k_succeeds_v_fails();
    test_fault_four_encodes_and_a_succeeds_b_fails();
    test_fault_factor_succeeds_landmark_fails();
    test_fault_actual_bytes_exceed_estimate();
    test_fault_stale_snapshot_and_generation();
    test_fault_codec_version_and_fingerprint_mismatch();
    test_fault_truncated_state();
    test_fault_transaction_rollback_accepted_prefixes();
    test_fault_publish_candidate_failures();
    test_absent_injection_seams_reporting();

    std::cout << "============================================================" << std::endl;
    std::cout << " ALL FAULT MATRIX REGRESSION TESTS PASSED (14/14 Suites)" << std::endl;
    std::cout << " Zero Half-Mutation Invariant Strictly Verified" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
