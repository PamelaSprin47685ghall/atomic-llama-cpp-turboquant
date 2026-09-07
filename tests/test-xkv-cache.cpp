// Deterministic regression tests for llama-xkv-cache:
// 1. Deeply immutable published snapshots:
//    - Pins keep original live set, row count, and code bytes untouched.
//    - External shared handle (std::shared_ptr<const xkv_segment>) prevents reclamation
//      even after unpinning until the handle is released (use_count == store-only).
// 2. Separate segment_id lineage from segment_version in segment_row_ref.
// 3. Two-phase batch COW transaction API (execute_mutation_transaction):
//    - Atomic multi-segment pack and removal with new version lineage.
//    - Throwing failure hook after first prepared clone causes clean abort with zero partial commit!
//    - Rejection of stale landmarks on row changes (summary marked absent).
// 4. Bounded host workspace arena preflight, hard capacity enforcement (capacity=0 rejected),
//    and live/peak/reserved tracking.
// 5. Sealing API with transactional state discipline (HOT_COMMITTED -> SEAL_CANDIDATE -> FACTORED),
//    atomic rollback of all payload states to HOT_COMMITTED on any failure (OOM, error gate, saving gate, etc.),
//    post-factor landmark factory callback consuming decoded final factor streams,
//    rejection of missing/fake landmarks with landmark_required,
//    overflow-safe final measured candidate allocation gating, and hot_release_plan emission with refreshed physical rows.
// 6. Accounting deduplication of shared B matrices across published and retired versions,
//    and deduplicated shared B in live_payload_bytes.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-factor.h"
#include "llama-kv-cells.h"
#include "llama-cparams.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <new>

using namespace llama_xkv;

static std::atomic<bool> g_track_allocs{false};
static std::atomic<size_t> g_alloc_count{0};
static std::atomic<bool> g_fail_alloc{false};

void * operator new(std::size_t n) {
    if (g_fail_alloc.load(std::memory_order_relaxed)) {
        throw std::bad_alloc();
    }
    if (g_track_allocs.load(std::memory_order_relaxed)) {
        g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void * operator new[](std::size_t n) {
    if (g_fail_alloc.load(std::memory_order_relaxed)) {
        throw std::bad_alloc();
    }
    if (g_track_allocs.load(std::memory_order_relaxed)) {
        g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

static llama_cparams make_default_test_cparams(enum llama_xkv_mode mode = LLAMA_XKV_MODE_SHADOW) {
    llama_cparams cparams = {};
    cparams.xkv_mode = mode;
    cparams.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cparams.xkv_group_size = 4;
    cparams.xkv_rank_k = 16;
    cparams.xkv_rank_v = 16;
    cparams.xkv_segment_tokens = 64;
    cparams.xkv_chunk_tokens = 8;
    cparams.xkv_workspace_mib = 16;
    cparams.xkv_decode_cache_mib = 8;
    cparams.xkv_min_saving = 0.10;
    return cparams;
}

// Helper to construct a valid candidate segment using llama-xkv-codec encode_matrix
static std::shared_ptr<xkv_segment> create_valid_candidate(
    llama_xkv_cache_store & store,
    uint32_t n_rows,
    uint32_t rank_k = 16,
    uint32_t rank_v = 16,
    uint32_t total_dim_k = 64,
    uint32_t total_dim_v = 64
) {
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_v;
    g.layer_feature_offsets_k = {0, total_dim_k / 4, total_dim_k / 2, 3 * total_dim_k / 4};
    g.layer_feature_dims_k = {total_dim_k / 4, total_dim_k / 4, total_dim_k / 4, total_dim_k / 4};
    g.layer_feature_offsets_v = {0, total_dim_v / 4, total_dim_v / 4, 3 * total_dim_v / 4};
    g.layer_feature_dims_v = {total_dim_v / 4, total_dim_v / 4, total_dim_v / 4, total_dim_v / 4};

    // A_K: [n_rows, rank_k] token-major F32
    codec_desc desc_a_k = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_F32,
        orientation::token_major,
        {n_rows, rank_k},
        0,
        1001
    );
    std::vector<float> data_a_k(n_rows * rank_k);
    for (size_t i = 0; i < data_a_k.size(); ++i) {
        data_a_k[i] = ((float)((i * 3 + 7) % 100)) * 0.01f;
    }
    g.a_k = encode_matrix(desc_a_k, data_a_k.data(), data_a_k.size());

    // B_K: [total_dim_k, rank_k] feature_major_transposed F32
    codec_desc desc_b_k = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_F32,
        orientation::feature_major_transposed,
        {total_dim_k, rank_k},
        0,
        1002
    );
    std::vector<float> data_b_k(total_dim_k * rank_k);
    for (size_t i = 0; i < data_b_k.size(); ++i) {
        data_b_k[i] = ((float)((i * 5 + 11) % 100)) * 0.01f;
    }
    g.set_b_k(encode_matrix(desc_b_k, data_b_k.data(), data_b_k.size()));

    // A_V: [n_rows, rank_v] token-major F32
    codec_desc desc_a_v = make_codec_desc(
        factor_role::a_v,
        GGML_TYPE_F32,
        orientation::token_major,
        {n_rows, rank_v},
        0,
        1003
    );
    std::vector<float> data_a_v(n_rows * rank_v);
    for (size_t i = 0; i < data_a_v.size(); ++i) {
        data_a_v[i] = ((float)((i * 7 + 13) % 100)) * 0.01f;
    }
    g.a_v = encode_matrix(desc_a_v, data_a_v.data(), data_a_v.size());

    // B_V: [total_dim_v, rank_v] feature_major_transposed F32
    codec_desc desc_b_v = make_codec_desc(
        factor_role::b_v,
        GGML_TYPE_F32,
        orientation::feature_major_transposed,
        {total_dim_v, rank_v},
        0,
        1004
    );
    std::vector<float> data_b_v(total_dim_v * rank_v);
    for (size_t i = 0; i < data_b_v.size(); ++i) {
        data_b_v[i] = ((float)((i * 11 + 17) % 100)) * 0.01f;
    }
    g.set_b_v(encode_matrix(desc_b_v, data_b_v.data(), data_b_v.size()));

    auto seg = store.create_candidate_segment(
        LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
        LLAMA_XKV_SOURCE_DECODED_HOT,
        {g}
    );

    return seg;
}

// ----------------------------------------------------------------------------
// Test 1: Candidate validation, rejection, and atomic abort
// ----------------------------------------------------------------------------
static void test_candidate_validation_and_abort() {
    std::cout << "[Test 1] Candidate validation, rejection, and atomic abort..." << std::endl;

    auto cparams = make_default_test_cparams();
    llama_xkv_cache_store store(cparams);

    std::string err;
    assert(!store.validate_candidate(nullptr, &err));
    assert(!err.empty());

    // Role mismatch
    auto seg = create_valid_candidate(store, 16);
    seg->groups[0].a_k.desc.role = factor_role::b_k;
    assert(!store.validate_candidate(seg, &err));
    assert(err.find("role mismatch") != std::string::npos);
    seg->groups[0].a_k.desc.role = factor_role::a_k;

    // Orientation mismatch
    seg->groups[0].a_k.desc.orient = orientation::feature_major_transposed;
    assert(!store.validate_candidate(seg, &err));
    assert(err.find("token_major") != std::string::npos);
    seg->groups[0].a_k.desc.orient = orientation::token_major;

    // Rank mismatch between A_K and B_K
    seg->groups[0].a_k.desc.logical_shape.cols = 12; // Mismatch with B_K rank 16
    assert(!store.validate_candidate(seg, &err));
    assert(err.find("rank dimension mismatch") != std::string::npos);
    seg->groups[0].a_k.desc.logical_shape.cols = 16;

    // Row count mismatch between A_K and A_V
    seg->groups[0].a_v.desc.logical_shape.rows = 15;
    assert(!store.validate_candidate(seg, &err));
    assert(err.find("row count mismatch") != std::string::npos);
    seg->groups[0].a_v.desc.logical_shape.rows = 16;

    seg->groups[0].a_k.bytes.pop_back();
    assert(!store.validate_candidate(seg, &err));
    assert(err.find("byte size mismatch") != std::string::npos);
    seg->groups[0].a_k.bytes.push_back(0);

    assert(store.validate_candidate(seg, &err));

    std::vector<uint64_t> pids_zero = {1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<uint64_t> gens(16, 1);
    assert(!store.publish_candidate(seg, pids_zero, gens, &err));
    assert(err.find("cannot be 0") != std::string::npos);

    std::vector<uint64_t> pids_dup = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3};
    assert(!store.publish_candidate(seg, pids_dup, gens, &err));
    assert(err.find("duplicate payload_id") != std::string::npos);

    std::vector<uint64_t> pids_short = {1, 2, 3};
    assert(!store.publish_candidate(seg, pids_short, gens, &err));
    assert(err.find("size must match") != std::string::npos);

    auto candidate_to_abort = create_valid_candidate(store, 8);
    assert(candidate_to_abort->groups[0].a_k.bytes.size() > 0);
    store.abort_candidate(candidate_to_abort);
    assert(candidate_to_abort == nullptr);
}

// ----------------------------------------------------------------------------
// Test 2: Deeply immutable published snapshots across remove_payload and replacement:
// Old pins keep original live set, row count, and code bytes untouched!
// External shared_ptr handles prevent premature reclamation until released.
// ----------------------------------------------------------------------------
static void test_immutable_published_snapshots_and_pins() {
    std::cout << "[Test 2] Deeply immutable snapshots: pins and external handles keep code bytes untouched..." << std::endl;

    auto cparams = make_default_test_cparams();
    llama_xkv_cache_store store(cparams);

    const std::vector<uint64_t> pids = {101, 102, 103, 104};
    const std::vector<uint64_t> gens = {1, 5, 12, 42};
    for (size_t i = 0; i < pids.size(); ++i) {
        store.register_hot_payload(pids[i], (uint32_t) i, gens[i], xkv_state::hot_committed);
    }
    assert(store.mark_seal_candidates(pids, gens, nullptr));

    auto seg = create_valid_candidate(store, 4);
    const uint64_t seg_id = seg->segment_id;
    std::string err;
    assert(store.publish_candidate(seg, pids, gens, &err));

    // Pin original version 1
    xkv_reader_pin pin_v1 = store.pin_segment(seg_id);
    assert(bool(pin_v1));
    assert(pin_v1->segment_version == 1);
    assert(pin_v1->n_rows == 4);
    assert(pin_v1->n_live_rows == 4);
    assert(pin_v1->live_rows == std::vector<bool>({true, true, true, true}));

    const std::vector<uint8_t> original_a_k_bytes = pin_v1->groups[0].a_k.bytes;
    const std::shared_ptr<const encoded_matrix> original_b_k_ptr = pin_v1->groups[0].b_k;

    // Obtain an unpinned external shared handle to version 1
    std::shared_ptr<const xkv_segment> ext_handle_v1 = store.get_segment_version(seg_id, 1);
    assert(ext_handle_v1 != nullptr);

    // 1. Remove payload 102.
    // Invariant: MUST NOT mutate published/pinned version 1 in place!
    // Must clone surviving rows (101, 103, 104) into version 2 and retire version 1 unchanged.
    store.remove_payload(102);

    // CRITICAL: Pinned version 1 MUST be 100% unchanged!
    assert(pin_v1->segment_version == 1);
    assert(pin_v1->n_rows == 4);
    assert(pin_v1->n_live_rows == 4);
    assert(pin_v1->live_rows == std::vector<bool>({true, true, true, true}));
    assert(pin_v1->groups[0].a_k.bytes == original_a_k_bytes);
    assert(pin_v1->groups[0].b_k == original_b_k_ptr);

    // Store's current published version is version 2 with 3 rows
    auto seg_v2 = store.get_segment(seg_id);
    assert(seg_v2 != nullptr);
    assert(seg_v2->segment_version == 2);
    assert(seg_v2->n_rows == 3);
    assert(seg_v2->n_live_rows == 3);
    assert(seg_v2->row_payload_ids == std::vector<uint64_t>({101, 103, 104}));
    assert(seg_v2->groups[0].b_k == original_b_k_ptr);

    // Release reader pin
    pin_v1.release();

    // Now version 1 has NO active reader pins, BUT ext_handle_v1 is still held externally!
    // Store reclamation MUST NOT free/clear bytes of ext_handle_v1 while external reference exists!
    store.reclaim_retired_segments();
    assert(ext_handle_v1->groups[0].a_k.bytes.size() > 0);
    assert(ext_handle_v1->groups[0].b_k != nullptr);

    // Release external handle
    ext_handle_v1.reset();

    // Now reclamation can safely drop version 1
    store.reclaim_retired_segments();
}

// ----------------------------------------------------------------------------
// Test 3: Pinned-reader COW pack, immutable shared B bytes, encoded A survivor byte identity,
// and 4-field segment_row_ref
// ----------------------------------------------------------------------------
static void test_pinned_reader_cow_pack_and_byte_identity() {
    std::cout << "[Test 3] Pinned-reader COW pack, immutable B, encoded A byte identity..." << std::endl;

    auto cparams = make_default_test_cparams();
    llama_xkv_cache_store store(cparams);

    const uint32_t n_rows = 6;
    const std::vector<uint64_t> pids = {10, 20, 30, 40, 50, 60};
    const std::vector<uint64_t> gens = {1, 2, 3, 4, 5, 6};

    for (size_t i = 0; i < n_rows; ++i) {
        store.register_hot_payload(pids[i], (uint32_t) i, gens[i], xkv_state::hot_committed);
    }
    assert(store.mark_seal_candidates(pids, gens, nullptr));

    auto seg = create_valid_candidate(store, n_rows);
    const uint64_t seg_id = seg->segment_id;

    const size_t stride_a_k = seg->groups[0].a_k.desc.row_stride_bytes;
    const size_t stride_a_v = seg->groups[0].a_v.desc.row_stride_bytes;
    for (uint32_t r = 0; r < n_rows; ++r) {
        std::memset(seg->groups[0].a_k.bytes.data() + r * stride_a_k, (int)(0xA0 + r), stride_a_k);
        std::memset(seg->groups[0].a_v.bytes.data() + r * stride_a_v, (int)(0xB0 + r), stride_a_v);
    }

    std::vector<uint8_t> expected_row0_a_k(seg->groups[0].a_k.bytes.begin(), seg->groups[0].a_k.bytes.begin() + stride_a_k);
    std::vector<uint8_t> expected_row2_a_k(seg->groups[0].a_k.bytes.begin() + 2 * stride_a_k, seg->groups[0].a_k.bytes.begin() + 3 * stride_a_k);
    std::vector<uint8_t> expected_row5_a_k(seg->groups[0].a_k.bytes.begin() + 5 * stride_a_k, seg->groups[0].a_k.bytes.begin() + 6 * stride_a_k);

    std::vector<uint8_t> expected_row0_a_v(seg->groups[0].a_v.bytes.begin(), seg->groups[0].a_v.bytes.begin() + stride_a_v);
    std::vector<uint8_t> expected_row2_a_v(seg->groups[0].a_v.bytes.begin() + 2 * stride_a_v, seg->groups[0].a_v.bytes.begin() + 3 * stride_a_v);
    std::vector<uint8_t> expected_row5_a_v(seg->groups[0].a_v.bytes.begin() + 5 * stride_a_v, seg->groups[0].a_v.bytes.begin() + 6 * stride_a_v);

    std::shared_ptr<const encoded_matrix> orig_b_k = seg->groups[0].b_k;
    std::shared_ptr<const encoded_matrix> orig_b_v = seg->groups[0].b_v;

    std::string err;
    assert(store.publish_candidate(seg, pids, gens, &err));

    xkv_reader_pin pin = store.pin_segment(seg_id);
    assert(bool(pin));
    assert(pin->segment_version == 1);

    // Remove 20, 40, 50
    store.remove_payload(20);
    store.remove_payload(40);
    store.remove_payload(50);

    // Old pin_v1 STILL completely intact: 6 rows, all live in its version!
    assert(pin->segment_version == 1);
    assert(pin->n_rows == 6);
    assert(pin->n_live_rows == 6);

    // Trigger pack_segment
    assert(store.pack_segment(seg_id, &err));

    // Newly published segment in store is densely packed with 3 rows
    auto packed_seg = store.get_segment(seg_id);
    assert(packed_seg != nullptr);
    assert(packed_seg->n_rows == 3);
    assert(packed_seg->n_live_rows == 3);
    assert(packed_seg->row_payload_ids == std::vector<uint64_t>({10, 30, 60}));

    // Invariant: Immutable B pointer shared across row-pack versions!
    assert(packed_seg->groups[0].b_k == orig_b_k);
    assert(packed_seg->groups[0].b_v == orig_b_v);

    // Invariant: Encoded A survivor row byte identity preserved without re-encoding
    assert(std::memcmp(packed_seg->groups[0].a_k.bytes.data() + 0 * stride_a_k, expected_row0_a_k.data(), stride_a_k) == 0);
    assert(std::memcmp(packed_seg->groups[0].a_v.bytes.data() + 0 * stride_a_v, expected_row0_a_v.data(), stride_a_v) == 0);
    assert(std::memcmp(packed_seg->groups[0].a_k.bytes.data() + 1 * stride_a_k, expected_row2_a_k.data(), stride_a_k) == 0);
    assert(std::memcmp(packed_seg->groups[0].a_v.bytes.data() + 1 * stride_a_v, expected_row2_a_v.data(), stride_a_v) == 0);
    assert(std::memcmp(packed_seg->groups[0].a_k.bytes.data() + 2 * stride_a_k, expected_row5_a_k.data(), stride_a_k) == 0);
    assert(std::memcmp(packed_seg->groups[0].a_v.bytes.data() + 2 * stride_a_v, expected_row5_a_v.data(), stride_a_v) == 0);

    // 4-field segment_row_ref check
    segment_row_ref ref1 = {seg_id, packed_seg->segment_version, 1, 0};
    segment_row_ref ref2 = {seg_id, packed_seg->segment_version, 3, 1};
    assert(ref1 < ref2);
    assert(ref1 != ref2);
}

// ----------------------------------------------------------------------------
// Test 4: Two-phase batch COW transaction: throwing allocation failure hook leaves zero changes
// ----------------------------------------------------------------------------
static void test_two_phase_transaction_rollback_on_failure() {
    std::cout << "[Test 4] Two-phase transaction: throwing failure hook leaves zero partial changes..." << std::endl;

    auto cparams = make_default_test_cparams();
    llama_xkv_cache_store store(cparams);

    for (uint64_t p = 1; p <= 8; ++p) {
        store.register_hot_payload(p, (uint32_t)(p - 1), p * 10, xkv_state::hot_committed);
    }
    assert(store.mark_seal_candidates({1, 2, 3, 4}, {10, 20, 30, 40}, nullptr));
    assert(store.mark_seal_candidates({5, 6, 7, 8}, {50, 60, 70, 80}, nullptr));

    // Publish segment 1
    auto seg1 = create_valid_candidate(store, 4);
    const uint64_t seg1_id = seg1->segment_id;
    std::string err;
    assert(store.publish_candidate(seg1, {1, 2, 3, 4}, {10, 20, 30, 40}, &err));

    // Publish segment 2
    auto seg2 = create_valid_candidate(store, 4);
    const uint64_t seg2_id = seg2->segment_id;
    assert(store.publish_candidate(seg2, {5, 6, 7, 8}, {50, 60, 70, 80}, &err));

    const uint64_t initial_live_epoch = store.live_epoch();
    const uint64_t initial_binding_epoch = store.binding_epoch();
    const auto stamp_t4 = store.current_stamp();
    const auto acc_t4 = store.get_accounting();

    // Prepare a multi-segment batch mutation covering seg1 and seg2
    xkv_batch_mutation bm;
    bm.payload_removals = {2, 6};
    bm.segments_to_pack = {seg1_id, seg2_id};

    // Inject failure hook that throws an exception after first prepared clone (seg1)
    bm.test_failure_hook = []() {
        throw std::bad_alloc();
    };

    xkv_mutation_result res;
    bool success = store.execute_mutation_transaction(bm, &res, &err);
    assert(!success);
    assert(!res.success);
    assert(err.find("bad allocation") != std::string::npos || err.find("std::bad_alloc") != std::string::npos);

    // CRITICAL: Zero partial commit! All segments, versions, and payload locations MUST be untouched!
    assert(store.live_epoch() == initial_live_epoch);
    assert(store.binding_epoch() == initial_binding_epoch);
    assert(store.current_stamp() == stamp_t4);
    const auto acc_after_t4 = store.get_accounting();
    assert(acc_after_t4.live_payload_bytes == acc_t4.live_payload_bytes);
    assert(acc_after_t4.allocated_bytes == acc_t4.allocated_bytes);
    assert(acc_after_t4.active_segments == acc_t4.active_segments);

    auto cur_seg1 = store.get_segment(seg1_id);
    assert(cur_seg1->segment_version == 1);
    assert(cur_seg1->n_rows == 4);
    assert(cur_seg1->n_live_rows == 4);

    auto cur_seg2 = store.get_segment(seg2_id);
    assert(cur_seg2->segment_version == 1);
    assert(cur_seg2->n_rows == 4);
    assert(cur_seg2->n_live_rows == 4);

    xkv_location loc2, loc6;
    assert(store.find_location(2, loc2) && loc2.segment_id == seg1_id && loc2.row == 1);
    assert(store.find_location(6, loc6) && loc6.segment_id == seg2_id && loc6.row == 1);

    // Now execute same transaction without failure hook -> succeeds
    bm.test_failure_hook = nullptr;
    assert(store.execute_mutation_transaction(bm, &res, &err));
    assert(res.success);
    assert(store.get_segment(seg1_id)->segment_version == 2);
    assert(store.get_segment(seg2_id)->segment_version == 2);

    // Single-segment transaction with failure hook must also fail cleanly before commit
    xkv_batch_mutation single_bm;
    single_bm.payload_removals = {1};
    single_bm.segments_to_pack = {seg1_id};
    single_bm.test_failure_hook = []() {
        throw std::bad_alloc();
    };
    assert(!store.execute_mutation_transaction(single_bm, &res, &err));
    assert(store.get_segment(seg1_id)->segment_version == 2); // Unchanged!
}

// ----------------------------------------------------------------------------
// Test 5: Layer aliasing, alias-deduplicated and shared-B deduplicated accounting
// ----------------------------------------------------------------------------
static void test_aliasing_and_shared_b_accounting() {
    std::cout << "[Test 5] Layer aliasing and shared B deduplicated accounting..." << std::endl;

    auto cparams = make_default_test_cparams();
    cparams.xkv_workspace_mib = 32;
    llama_xkv_cache_store store(cparams);

    store.register_layer_alias(5, 1);
    assert(store.resolve_owning_layer(5) == 1);

    for (uint64_t p = 1; p <= 8; ++p) {
        store.register_hot_payload(p, (uint32_t)(p - 1), 1, xkv_state::hot_committed);
    }
    assert(store.mark_seal_candidates({1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 1, 1, 1, 1}, nullptr));

    auto seg = create_valid_candidate(store, 8);
    const uint64_t seg_id = seg->segment_id;
    std::string err;
    assert(store.publish_candidate(seg, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 1, 1, 1, 1}, &err));

    auto acc1 = store.get_accounting();
    assert(acc1.active_segments == 1);
    assert(acc1.unique_b_matrices == 2);
    assert(acc1.shared_b_bytes > 0);

    // Pin version 1 and remove 2 payloads to create version 2
    xkv_reader_pin pin = store.pin_segment(seg_id);
    store.remove_payload(1);
    store.remove_payload(2);

    // Accounting now holds both active version 2 and pinned retired version 1.
    // Invariant: B matrices must be DEDUPLICATED across versions, NOT counted twice!
    auto acc2 = store.get_accounting();
    assert(acc2.pinned_segments == 1);
    assert(acc2.unique_b_matrices == 2);
    assert(acc2.shared_b_bytes == acc1.shared_b_bytes);
}

// ----------------------------------------------------------------------------
// Test 6: Bounded host workspace arena: hard capacity enforcement, preflight, live/peak
// ----------------------------------------------------------------------------
static void test_workspace_arena_preflight_and_counters() {
    std::cout << "[Test 6] Workspace arena: hard capacity, preflight, live/peak counters..." << std::endl;

    xkv_workspace_arena zero_arena(0);
    assert(!zero_arena.preflight(100));
    assert(zero_arena.base_data() == nullptr);
    assert(!bool(zero_arena.acquire(100)));

    const size_t arena_cap = 1024 * 1024; // 1 MiB
    xkv_workspace_arena arena(arena_cap);

    assert(arena.get_capacity_bytes() == arena_cap);
    assert(arena.base_data() != nullptr);
    // Base buffer itself is 64B aligned.
    assert(((uintptr_t) arena.base_data() % xkv_workspace_arena::kAlignment) == 0);
    assert(arena.preflight(512 * 1024));
    assert(!arena.preflight(arena_cap + 1));

    {
        xkv_arena_lease lease1 = arena.acquire(512 * 1024);
        assert(bool(lease1));
        assert(arena.get_live_bytes() == 512 * 1024);
        assert(arena.get_peak_bytes() == 512 * 1024);
        // Real backing: non-null, 64B-aligned, sized span inside the buffer.
        assert(lease1.data() != nullptr);
        assert(((uintptr_t) lease1.data() % xkv_workspace_arena::kAlignment) == 0);
        assert(lease1.size() == 512 * 1024);
        assert(lease1.data() >= arena.base_data());
        assert((const uint8_t *) lease1.data() + lease1.size() <=
               (const uint8_t *) arena.base_data() + arena.get_capacity_bytes());

        xkv_arena_lease lease2 = arena.acquire(256 * 1024);
        assert(bool(lease2));
        assert(arena.get_live_bytes() == 768 * 1024);
        assert(arena.get_peak_bytes() == 768 * 1024);
        // Concurrent leases never overlap.
        const uint8_t * a0 = (const uint8_t *) lease1.data();
        const uint8_t * a1 = (const uint8_t *) lease1.data() + lease1.size();
        const uint8_t * b0 = (const uint8_t *) lease2.data();
        const uint8_t * b1 = (const uint8_t *) lease2.data() + lease2.size();
        assert(a1 <= b0 || b1 <= a0);
        assert(((uintptr_t) lease2.data() % xkv_workspace_arena::kAlignment) == 0);

        // Data is really usable: write/read through both spans.
        std::memset(lease1.data(), 0x5A, lease1.size());
        std::memset(lease2.data(), 0xA5, lease2.size());
        assert(*((volatile uint8_t *) lease1.data()) == 0x5A);
        assert(*((volatile uint8_t *) lease2.data()) == 0xA5);

        // Fail to exceed capacity
        xkv_arena_lease lease3 = arena.acquire(300 * 1024);
        assert(!bool(lease3));
        assert(lease3.data() == nullptr);
        assert(arena.get_live_bytes() == 768 * 1024);
    }

    assert(arena.get_live_bytes() == 0);
    assert(arena.get_peak_bytes() == 768 * 1024);

    // Freed blocks are recycled with identical bounds (no heap, no drift).
    {
        xkv_arena_lease r1 = arena.acquire(512 * 1024);
        assert(bool(r1));
        assert(r1.data() != nullptr);
        assert(arena.get_live_bytes() == 512 * 1024);
    }
    assert(arena.get_live_bytes() == 0);

    // Odd sizes round up to the alignment; spans stay aligned and bounded.
    {
        xkv_arena_lease odd = arena.acquire(100);
        assert(bool(odd));
        assert(odd.size() == xkv_workspace_arena::kAlignment * 2); // 100 -> 128
        assert(((uintptr_t) odd.data() % xkv_workspace_arena::kAlignment) == 0);
        assert(arena.get_live_bytes() == odd.size());
    }
    assert(arena.get_live_bytes() == 0);

    // set_capacity reallocates only with no live leases; refuses otherwise.
    {
        xkv_arena_lease held = arena.acquire(1024);
        assert(bool(held));
        assert(!arena.set_capacity_bytes(2 * 1024 * 1024)); // Live lease: refuse.
        assert(arena.get_capacity_bytes() == arena_cap);   // Unchanged.
        held.release();
        assert(arena.set_capacity_bytes(2 * 1024 * 1024));
        assert(arena.get_capacity_bytes() == 2 * 1024 * 1024);
        assert(arena.base_data() != nullptr);
    }
    // Exact lease bound: one metadata node travels per lease, so at most
    // kMaxBlocks-1 concurrent minimal leases; preflight agrees every step.
    {
        xkv_workspace_arena bounded(1024 * 1024);
        std::vector<xkv_arena_lease> held;
        size_t acquired = 0;
        while (bounded.preflight(64)) {
            xkv_arena_lease l = bounded.acquire(64);
            if (!bool(l)) break;
            held.push_back(std::move(l));
            ++acquired;
            assert(acquired <= xkv_workspace_arena::kMaxBlocks);
        }
        assert(acquired == xkv_workspace_arena::kMaxBlocks - 1);
        assert(!bounded.preflight(64));
        assert(!bool(bounded.acquire(64))); // Bound refused, bytes unchanged.
        assert(bounded.get_live_bytes() == acquired * 64);
        held.clear();
        assert(bounded.get_live_bytes() == 0);
        assert(bounded.preflight(1024 * 1024));
        xkv_arena_lease whole = bounded.acquire(1024 * 1024);
        assert(bool(whole) && whole.size() == 1024 * 1024);
    }
    // Lease outlives its arena: shared state keeps data valid, release safe.
    {
        xkv_arena_lease orphan;
        {
            xkv_workspace_arena inner(1024 * 1024);
            orphan = inner.acquire(4096);
            assert(bool(orphan));
            std::memset(orphan.data(), 0x33, orphan.size());
        } // inner arena destroyed here.
        assert(bool(orphan));
        assert(orphan.data() != nullptr);
        assert(*((volatile uint8_t *) orphan.data()) == 0x33); // Pattern intact.
        std::memset(orphan.data(), 0x44, orphan.size());
        assert(*((volatile uint8_t *) orphan.data()) == 0x44);
        orphan.release(); // Safe without an arena.
        assert(!bool(orphan));
    }
    // Long churn then full recovery: fragmentation never strands capacity.
    {
        xkv_workspace_arena churn(1024 * 1024);
        // Deterministic size-varying acquire/release cycles.
        for (int cycle = 0; cycle < 3000; ++cycle) {
            size_t sz = 64 * (1 + (size_t)((cycle * 2654435761ULL) % 256));
            xkv_arena_lease tmp = churn.acquire(sz);
            assert(bool(tmp));
        }
        assert(churn.get_live_bytes() == 0);
        // Interleaved holds released in a fragmenting order (every other).
        std::vector<xkv_arena_lease> holds;
        holds.reserve(100);
        for (int i = 0; i < 100; ++i) {
            holds.push_back(churn.acquire(8192));
            assert(bool(holds.back()));
        }
        for (int i = 0; i < 100; i += 2) {
            holds[i].release();
        }
        // Adjacent survivors still blocked: a near-full acquire must fail...
        assert(!bool(churn.acquire(1024 * 1024)));
        for (int i = 1; i < 100; i += 2) {
            holds[i].release();
        }
        holds.clear();
        assert(churn.get_live_bytes() == 0);
        // ...then full coalescing recovers the entire buffer contiguously.
        xkv_arena_lease whole = churn.acquire(1024 * 1024);
        assert(bool(whole));
        assert(whole.size() == 1024 * 1024);
        assert(whole.data() == churn.base_data());
    }
}

// ----------------------------------------------------------------------------
// Test 7: Sealing API: transactional lock & rollback, landmark requirement,
// post-factor landmark factory, final measured allocation gating, and hot_release_plan
// ----------------------------------------------------------------------------
static void test_sealing_api_transaction_and_gates() {
    std::cout << "[Test 7] Sealing API: transaction lock & rollback, post-factor landmark factory, release plan..." << std::endl;

    auto cparams = make_default_test_cparams();
    cparams.xkv_workspace_mib = 64;
    llama_xkv_cache_store store(cparams);

    const uint32_t n_tokens = 256;
    const uint32_t total_dim_k = 64;
    const uint32_t total_dim_v = 64;

    std::vector<uint64_t> pids(n_tokens);
    std::vector<uint64_t> gens(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pids[i] = 2000 + i;
        gens[i] = 1;
        store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
    }

    std::vector<float> k_data(n_tokens * total_dim_k);
    std::vector<float> v_data(n_tokens * total_dim_v);
    for (uint32_t r = 0; r < n_tokens; ++r) {
        for (uint32_t c = 0; c < total_dim_k; ++c) {
            float val = std::sin((float)r * 0.3f) * std::cos((float)c * 0.2f);
            k_data[r * total_dim_k + c] = val;
            v_data[r * total_dim_v + c] = val * 0.5f;
        }
    }

    xkv_factor_group_input g0;
    g0.group_index = 0;
    g0.owning_layers = {0, 1, 2, 3};
    g0.rank_k = 8;
    g0.rank_v = 8;
    g0.total_dim_k = total_dim_k;
    g0.total_dim_v = total_dim_v;
    g0.row_positions.resize(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) g0.row_positions[i] = (int64_t) i;
    g0.hot_bytes_per_row_k = {32, 32, 32, 32};
    g0.hot_bytes_per_row_v = {32, 32, 32, 32};
    g0.layer_feature_offsets_k = {0, total_dim_k / 4, total_dim_k / 2, 3 * total_dim_k / 4};
    g0.layer_feature_dims_k = {total_dim_k / 4, total_dim_k / 4, total_dim_k / 4, total_dim_k / 4};
    g0.layer_feature_offsets_v = {0, total_dim_v / 4, total_dim_v / 4, 3 * total_dim_v / 4};
    g0.layer_feature_dims_v = {total_dim_v / 4, total_dim_v / 4, total_dim_v / 4, total_dim_v / 4};
    g0.canonical_k_data = k_data.data();
    g0.k_rows = n_tokens;
    g0.k_cols = total_dim_k;
    g0.canonical_v_data = v_data.data();
    g0.v_rows = n_tokens;
    g0.v_cols = total_dim_v;

    xkv_bundle_sealing_params sparams;
    sparams.rank_k = 8;
    sparams.rank_v = 8;
    sparams.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS; // Requires landmark factory
    sparams.source = LLAMA_XKV_SOURCE_DECODED_HOT;
    sparams.max_relative_error = 0.99;
    sparams.min_saving_ratio = 0.10; // Genuine 10% default saving gate
    sparams.flat_type_k = GGML_TYPE_F16;
    sparams.flat_type_v = GGML_TYPE_F16;

    // 1. Missing landmark factory when profile requires landmarks -> must fail with landmark_required
    // and rollback all locked states back to HOT_COMMITTED!
    auto res_nolm = store.seal_segment_bundle(
        {g0}, pids, gens, sparams
    );
    assert(!res_nolm.success);
    assert(res_nolm.skip_reason == xkv_skip_reason::landmark_required);
    assert(store.get_skipped_count(xkv_skip_reason::landmark_required) == 1);

    for (uint64_t pid : pids) {
        xkv_state st = xkv_state::hot_writing;
        assert(store.find_payload_state(pid, st));
        assert(st == xkv_state::hot_committed);
    }

    // 2. Bounded native landmark factory: valid landmarks from final encoded
    // stream shapes (no decoded mirrors). Test content is synthetic; the seal
    // exercises plumbing, gating, and rollback, not landmark fidelity.
    sparams.landmark_factory = [](
        uint32_t /*group_index*/,
        const encoded_matrix & /*enc_a_k*/,
        const encoded_matrix & enc_b_k,
        const int64_t * row_positions,
        uint64_t lm_rows,
        factor_workspace_span scratch,
        encoded_matrix & out_lm,
        std::vector<xkv_landmark_chunk> & out_lm_chunks,
        std::string * /*err*/
    ) -> bool {
        assert(row_positions != nullptr && lm_rows > 0);
        assert(scratch.valid());
        const uint32_t chunks = ((uint32_t) lm_rows + 7) / 8;
        const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
        std::vector<float> lm_data((size_t) chunks * cols, 0.1f);
        codec_desc desc = make_codec_desc(
            factor_role::landmark,
            GGML_TYPE_Q8_0,
            orientation::token_major,
            {chunks, cols},
            0,
            777
        );
        out_lm = encode_matrix(desc, lm_data.data(), lm_data.size());
        out_lm_chunks.clear();
        uint32_t rem = (uint32_t) lm_rows;
        uint32_t off = 0;
        for (uint32_t c = 0; c < chunks; ++c) {
            uint32_t cnt = std::min(8u, rem);
            out_lm_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
            off += cnt;
            rem -= cnt;
        }
        return true;
    };

    // 3. Error threshold exceeded -> must rollback all states
    sparams.max_relative_error = 1e-9;
    auto res_err = store.seal_segment_bundle(
        {g0}, pids, gens, sparams
    );
    assert(!res_err.success);
    assert(res_err.skip_reason == xkv_skip_reason::error_threshold_exceeded);
    for (uint64_t pid : pids) {
        xkv_state st = xkv_state::hot_writing;
        assert(store.find_payload_state(pid, st));
        assert(st == xkv_state::hot_committed);
    }

    // 4. Successful seal -> returns release plan
    sparams.max_relative_error = 0.99;
    auto res_ok = store.seal_segment_bundle(
        {g0}, pids, gens, sparams
    );
    if (!res_ok.success) {
        std::cerr << "Seal failed: reason=" << xkv_skip_reason_to_str(res_ok.skip_reason)
                  << ", msg=" << res_ok.message
                  << ", flat_source_bytes=" << res_ok.flat_source_bytes
                  << ", factored_bytes=" << res_ok.factored_bytes
                  << ", peak=" << store.get_arena().get_peak_bytes() << std::endl;
    }
    assert(res_ok.success);
    assert(res_ok.segment_id > 0);
    assert(res_ok.segment_version == 1);
    assert(res_ok.saved_bytes > 0);
    assert(res_ok.release_plan.released_payload_ids.size() == n_tokens);
    assert(res_ok.release_plan.dense_bytes_freed > 0);

    for (uint64_t pid : pids) {
        xkv_state st = xkv_state::hot_writing;
        assert(store.find_payload_state(pid, st));
        assert(st == xkv_state::factored);
    }

    // 5. Test release plan validation
    std::string val_err;
    assert(store.validate_hot_release_plan(res_ok.release_plan, &val_err));

    // Reject corrupted / stale plan: e.g. stale generation
    auto stale_gen_plan = res_ok.release_plan;
    stale_gen_plan.released_generations[0] = 999;
    assert(!store.validate_hot_release_plan(stale_gen_plan, &val_err));
    assert(val_err.find("generation mismatch") != std::string::npos);

    // Reject wrong segment_id
    auto stale_seg_plan = res_ok.release_plan;
    stale_seg_plan.expected_segment_id = 999;
    assert(!store.validate_hot_release_plan(stale_seg_plan, &val_err));
    assert(val_err.find("segment_id mismatch") != std::string::npos);

    // Reject wrong segment_version
    auto stale_ver_plan = res_ok.release_plan;
    stale_ver_plan.expected_segment_version = 999;
    assert(!store.validate_hot_release_plan(stale_ver_plan, &val_err));
    assert(val_err.find("segment_version mismatch") != std::string::npos);
}

int main() {
    std::cout << "=== Running XKV Cache Store Tests ===" << std::endl;

    test_candidate_validation_and_abort();
    test_immutable_published_snapshots_and_pins();
    test_pinned_reader_cow_pack_and_byte_identity();
    test_two_phase_transaction_rollback_on_failure();
    test_aliasing_and_shared_b_accounting();
    test_workspace_arena_preflight_and_counters();
    test_sealing_api_transaction_and_gates();

    // Test 8: Storage-only immutable snapshot enumeration APIs
    {
        std::cout << "[Test 8] Storage-only immutable snapshot enumeration APIs..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        store.register_hot_payload(100, 0, 1, xkv_state::hot_writing);
        store.register_hot_payload(200, 1, 1, xkv_state::hot_committed);
        store.register_hot_payload(300, 2, 1, xkv_state::hot_committed);
        // seal_candidate is reached via marking, never by direct registration.
        assert(store.mark_seal_candidates({300}, {1}, nullptr));

        // Filter by hot_committed
        auto snap_comm = store.list_hot_payload_bindings({xkv_state::hot_committed});
        assert(snap_comm.bindings.size() == 1);
        assert(snap_comm.bindings[0].payload_id == 200);
        assert(snap_comm.bindings[0].state == xkv_state::hot_committed);

        // Filter by multiple states
        auto snap_multi = store.list_hot_payload_bindings({xkv_state::hot_committed, xkv_state::seal_candidate});
        assert(snap_multi.bindings.size() == 2);
        assert(snap_multi.bindings[0].payload_id == 200);
        assert(snap_multi.bindings[1].payload_id == 300);

        // All hot bindings
        auto snap_all = store.list_hot_payload_bindings();
        assert(snap_all.bindings.size() == 3);
        assert(snap_all.bindings[0].payload_id == 100);
        assert(snap_all.bindings[1].payload_id == 200);
        assert(snap_all.bindings[2].payload_id == 300);

        // Query payload segments for hot and non-existent
        auto seg_snap = store.query_payload_segments({300, 100, 999});
        assert(seg_snap.views.size() == 2); // 100 and 300 exist, 999 does not
        assert(seg_snap.views[0].payload_id == 100);
        assert(seg_snap.views[1].payload_id == 300);
    }

    // Test 9: Store location registration and cross-stream distinct identity registration
    {
        std::cout << "[Test 9] Cross-stream distinct identity registration..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        const uint64_t pid_src = 5001;
        const uint64_t pid_dst = 5002;
        store.register_hot_payload(pid_src, 0, 1);
        store.register_hot_payload(pid_dst, 0, 1);

        xkv_location loc_src, loc_dst;
        assert(store.find_location(pid_src, loc_src));
        assert(store.find_location(pid_dst, loc_dst));
        assert(loc_src.kind == xkv_location_kind::hot);
        assert(loc_dst.kind == xkv_location_kind::hot);

        // Removing dst does not affect src
        store.remove_payload(pid_dst);
        assert(!store.find_location(pid_dst, loc_dst));
        assert(store.find_location(pid_src, loc_src));
    }

    // Test 10: Multi-group bundle proof (3 groups: 4/4/tail owning layers, non-contiguous owners, alias map,
    // one payload locator reads all layers, failure in last group's landmark publishes nothing/releases nothing,
    // and repeated pack preserves survivor A bytes and B handles across all groups)
    {
        std::cout << "[Test 10] Multi-group bundle proof (3 groups 4/4/tail, non-contiguous, alias, pack)..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 128;
        llama_xkv_cache_store store(cparams);

        // Layer aliases: e.g. model layer 4 aliases owning layer 3, model layer 8 aliases owning layer 7
        store.register_layer_alias(4, 3);
        store.register_layer_alias(8, 7);
        assert(store.resolve_owning_layer(4) == 3);
        assert(store.resolve_owning_layer(8) == 7);

        // Define 3 groups covering non-contiguous attention owning layers [3, 7, 11, 15], [19, 23, 27, 31], [35, 39] (tail group)
        const uint32_t n_tokens = 256;
        std::vector<uint64_t> pids(n_tokens);
        std::vector<uint64_t> gens(n_tokens, 1);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            pids[i] = 8000 + i;
            store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed);
        }

        // Group 0: 4 layers -> total_dim = 64
        std::vector<float> k_data_0(n_tokens * 64, 0.1f);
        std::vector<float> v_data_0(n_tokens * 64, 0.2f);
        xkv_factor_group_input g0;
        g0.group_index = 0;
        g0.owning_layers = {3, 7, 11, 15};
        g0.rank_k = 16; // W=4 group scaled rank
        g0.rank_v = 16;
        g0.total_dim_k = 64; g0.total_dim_v = 64;
        g0.row_positions.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) g0.row_positions[i] = (int64_t) i;
        g0.hot_bytes_per_row_k = {32, 32, 32, 32};
        g0.hot_bytes_per_row_v = {32, 32, 32, 32};
        g0.layer_feature_offsets_k = {0, 16, 32, 48}; g0.layer_feature_dims_k = {16, 16, 16, 16};
        g0.layer_feature_offsets_v = {0, 16, 32, 48}; g0.layer_feature_dims_v = {16, 16, 16, 16};
        g0.canonical_k_data = k_data_0.data(); g0.k_rows = n_tokens; g0.k_cols = 64;
        g0.canonical_v_data = v_data_0.data(); g0.v_rows = n_tokens; g0.v_cols = 64;

        // Group 1: 4 layers -> total_dim = 64
        std::vector<float> k_data_1(n_tokens * 64, 0.3f);
        std::vector<float> v_data_1(n_tokens * 64, 0.4f);
        xkv_factor_group_input g1;
        g1.group_index = 1;
        g1.owning_layers = {19, 23, 27, 31};
        g1.rank_k = 16; // W=4 group scaled rank
        g1.rank_v = 16;
        g1.total_dim_k = 64; g1.total_dim_v = 64;
        g1.row_positions.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) g1.row_positions[i] = (int64_t) i;
        g1.hot_bytes_per_row_k = {32, 32, 32, 32};
        g1.hot_bytes_per_row_v = {32, 32, 32, 32};
        g1.layer_feature_offsets_k = {0, 16, 32, 48}; g1.layer_feature_dims_k = {16, 16, 16, 16};
        g1.layer_feature_offsets_v = {0, 16, 32, 48}; g1.layer_feature_dims_v = {16, 16, 16, 16};
        g1.canonical_k_data = k_data_1.data(); g1.k_rows = n_tokens; g1.k_cols = 64;
        g1.canonical_v_data = v_data_1.data(); g1.v_rows = n_tokens; g1.v_cols = 64;

        // Group 2 (tail group W=2): 2 layers -> total_dim = 32, scaled rank_k=8, rank_v=8 (proportional half-rank)
        std::vector<float> k_data_2(n_tokens * 32, 0.5f);
        std::vector<float> v_data_2(n_tokens * 32, 0.6f);
        xkv_factor_group_input g2;
        g2.group_index = 2;
        g2.owning_layers = {35, 39};
        g2.rank_k = 8; // Tail W=2 half rank
        g2.rank_v = 8;
        g2.total_dim_k = 32; g2.total_dim_v = 32;
        g2.row_positions.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) g2.row_positions[i] = (int64_t) i;
        g2.hot_bytes_per_row_k = {32, 32};
        g2.hot_bytes_per_row_v = {32, 32};
        g2.layer_feature_offsets_k = {0, 16}; g2.layer_feature_dims_k = {16, 16};
        g2.layer_feature_offsets_v = {0, 16}; g2.layer_feature_dims_v = {16, 16};
        g2.canonical_k_data = k_data_2.data(); g2.k_rows = n_tokens; g2.k_cols = 32;
        g2.canonical_v_data = v_data_2.data(); g2.v_rows = n_tokens; g2.v_cols = 32;

        xkv_bundle_sealing_params sparams;
        sparams.rank_k = 8;
        sparams.rank_v = 8;
        sparams.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
        sparams.source = LLAMA_XKV_SOURCE_DECODED_HOT;
        sparams.max_relative_error = 0.99;
        sparams.min_saving_ratio = 0.05;
        sparams.flat_type_k = GGML_TYPE_F16;
        sparams.flat_type_v = GGML_TYPE_F16;
        sparams.expected_group_count = 3;
        sparams.expected_group_map_fingerprint = compute_layer_group_map_fingerprint({g0, g1, g2});

        // A0. Dropping middle group (omitted group 1) fails atomically with unsupported_config
        auto drop_mid_res = store.seal_segment_bundle({g0, g2}, pids, gens, sparams);
        assert(!drop_mid_res.success);
        assert(drop_mid_res.skip_reason == xkv_skip_reason::unsupported_config);

        // A1. Modifying offset/dim fails fingerprint check atomically
        auto bad_offset_g0 = g0;
        bad_offset_g0.layer_feature_offsets_k[1] = 17; // Corrupted offset
        auto bad_fp_res = store.seal_segment_bundle({bad_offset_g0, g1, g2}, pids, gens, sparams);
        assert(!bad_fp_res.success);
        assert(bad_fp_res.skip_reason == xkv_skip_reason::unsupported_config);
        assert(bad_fp_res.message.find("fingerprint mismatch") != std::string::npos);

        // A2. Oversized rank rejects atomically
        auto bad_rank_g2 = g2;
        bad_rank_g2.rank_k = 1000; // Rank exceeds cols (32)
        auto bad_rank_res = store.seal_segment_bundle({g0, g1, bad_rank_g2}, pids, gens, sparams);
        assert(!bad_rank_res.success);
        assert(bad_rank_res.skip_reason == xkv_skip_reason::unsupported_config);

        // A. Failure in the last group's landmark publishes NOTHING and releases NO hot slots
        sparams.landmark_factory = [](
            uint32_t group_index,
            const encoded_matrix & /*enc_a_k*/,
            const encoded_matrix & enc_b_k,
            const int64_t * row_positions,
            uint64_t lm_rows,
            factor_workspace_span scratch,
            encoded_matrix & out_lm,
            std::vector<xkv_landmark_chunk> & out_chunks,
            std::string * err
        ) -> bool {
            if (group_index == 2) {
                if (err) *err = "simulated landmark encode failure in tail group 2";
                return false; // Fail only in last group
            }
            assert(row_positions != nullptr && lm_rows > 0);
            assert(scratch.valid());
            const uint32_t chunks = ((uint32_t) lm_rows + 7) / 8;
            const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
            codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0, orientation::token_major, {chunks, cols}, 0, 777);
            std::vector<float> lm_data((size_t) chunks * cols, 0.1f);
            out_lm = encode_matrix(desc, lm_data.data(), lm_data.size());
            out_chunks.clear();
            uint32_t rem = (uint32_t) lm_rows;
            uint32_t off = 0;
            for (uint32_t c = 0; c < chunks; ++c) {
                uint32_t cnt = std::min(8u, rem);
                out_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                off += cnt;
                rem -= cnt;
            }
            return true;
        };

        auto fail_res = store.seal_segment_bundle({g0, g1, g2}, pids, gens, sparams);
        assert(!fail_res.success);
        assert(fail_res.skip_reason == xkv_skip_reason::codec_error);
        assert(fail_res.message.find("tail group 2") != std::string::npos);

        // Verify zero publish and zero hot release: all payloads remain hot_committed!
        for (uint64_t pid : pids) {
            xkv_state st = xkv_state::hot_writing;
            assert(store.find_payload_state(pid, st));
            assert(st == xkv_state::hot_committed);
            xkv_location loc;
            assert(store.find_location(pid, loc));
            assert(loc.kind == xkv_location_kind::hot);
        }

        // B. Real successful multi-group bundle sealing
        sparams.landmark_factory = [](
            uint32_t /*group_index*/,
            const encoded_matrix & /*enc_a_k*/,
            const encoded_matrix & enc_b_k,
            const int64_t * row_positions,
            uint64_t lm_rows,
            factor_workspace_span scratch,
            encoded_matrix & out_lm,
            std::vector<xkv_landmark_chunk> & out_chunks,
            std::string * /*err*/
        ) -> bool {
            assert(row_positions != nullptr && lm_rows > 0);
            assert(scratch.valid());
            const uint32_t chunks = ((uint32_t) lm_rows + 7) / 8;
            const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
            codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0, orientation::token_major, {chunks, cols}, 0, 777);
            std::vector<float> lm_data((size_t) chunks * cols, 0.1f);
            out_lm = encode_matrix(desc, lm_data.data(), lm_data.size());
            out_chunks.clear();
            uint32_t rem = (uint32_t) lm_rows;
            uint32_t off = 0;
            for (uint32_t c = 0; c < chunks; ++c) {
                uint32_t cnt = std::min(8u, rem);
                out_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                off += cnt;
                rem -= cnt;
            }
            return true;
        };

        auto ok_res = store.seal_segment_bundle({g0, g1, g2}, pids, gens, sparams);
        assert(ok_res.success);
        assert(ok_res.segment_id > 0);
        assert(ok_res.segment_version == 1);
        assert(ok_res.release_plan.released_payload_ids.size() == n_tokens);
        assert(store.validate_hot_release_plan(ok_res.release_plan));

        // One payload lookup makes EVERY model attention layer readable!
        xkv_location test_loc;
        assert(store.find_location(pids[0], test_loc));
        assert(test_loc.kind == xkv_location_kind::factored);
        assert(test_loc.segment_id == ok_res.segment_id);

        auto bundle = store.get_segment(ok_res.segment_id);
        assert(bundle != nullptr);
        assert(bundle->groups.size() == 3);

        // Resolving owning layers and aliases for all groups
        assert(bundle->find_group_for_layer(3) != nullptr && bundle->find_group_for_layer(3)->group_index == 0);
        assert(bundle->find_group_for_layer(store.resolve_owning_layer(4)) != nullptr); // Alias 4 -> 3
        assert(bundle->find_group_for_layer(19) != nullptr && bundle->find_group_for_layer(19)->group_index == 1);
        assert(bundle->find_group_for_layer(store.resolve_owning_layer(8)) != nullptr); // Alias 8 -> 7
        assert(bundle->find_group_for_layer(35) != nullptr && bundle->find_group_for_layer(35)->group_index == 2);
        assert(bundle->find_group_for_layer(39) != nullptr && bundle->find_group_for_layer(39)->group_index == 2);

        // C. Repeated pack preserves survivor A bytes for all groups and all B handle identities
        auto b0_k = bundle->groups[0].b_k;
        auto b1_k = bundle->groups[1].b_k;
        auto b2_k = bundle->groups[2].b_k;

        // Remove payload 8001 (row 1 of a LANDMARKS bundle) WITHOUT a rebuild
        // callback: chunk summaries are position-sensitive (never first-N copy)
        // and duplicating the layout breaks the byte budget, so the removal
        // refuses with zero mutation and the old version stays published.
        auto stamp_c0 = store.current_stamp();
        std::string err_c;
        assert(!store.remove_payload(8001, &err_c));
        assert(err_c.find("rebuild") != std::string::npos);
        auto unmoved = store.get_segment(ok_res.segment_id);
        assert(unmoved->segment_version == 1);
        assert(unmoved->n_rows == n_tokens && unmoved->n_live_rows == n_tokens);
        assert(store.current_stamp() == stamp_c0);
        xkv_location still_loc;
        assert(store.find_location(8001, still_loc) && still_loc.row == 1);
        assert(store.validate_candidate(unmoved));

        // WITH a semantic rebuild the same removal compacts all 3 groups with
        // valid landmarks, shared B handles, and preserved per-group ranks.
        auto rebuild_all = [](const xkv_segment &, const std::vector<uint32_t> & surviving,
                               xkv_segment & new_seg, std::string *) -> bool {
            const uint32_t nn = (uint32_t) surviving.size();
            const uint32_t nch = (nn + 7) / 8;
            for (auto & ng : new_seg.groups) {
                codec_desc d = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                               orientation::token_major, {nch, ng.total_dim_k}, 0, 777);
                std::vector<float> z((size_t) nch * ng.total_dim_k, 0.0f);
                ng.landmark = encode_matrix(d, z.data(), z.size());
                ng.landmark_chunks.clear();
                uint32_t rem = nn;
                uint32_t off = 0;
                for (uint32_t c = 0; c < nch; ++c) {
                    uint32_t cnt = std::min(8u, rem);
                    ng.landmark_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                    off += cnt;
                    rem -= cnt;
                }
                ng.landmark_table_fingerprint = compute_landmark_table_fingerprint(ng.landmark_chunks);
            }
            return true;
        };
        assert(store.remove_payload(8001, &err_c, rebuild_all));
        auto packed_bundle = store.get_segment(ok_res.segment_id);
        assert(packed_bundle != nullptr);
        assert(packed_bundle->segment_version == 2);
        assert(packed_bundle->n_rows == n_tokens - 1);
        assert(packed_bundle->n_live_rows == n_tokens - 1);
        assert(packed_bundle->row_payload_ids.front() == 8000);
        assert(packed_bundle->row_payload_ids.back() == 8000 + n_tokens - 1);
        assert(store.validate_candidate(packed_bundle));
        // All 3 groups compacted together with identical new row counts.
        assert(packed_bundle->groups[0].a_k.desc.logical_shape.rows == n_tokens - 1);
        assert(packed_bundle->groups[1].a_k.desc.logical_shape.rows == n_tokens - 1);
        assert(packed_bundle->groups[2].a_k.desc.logical_shape.rows == n_tokens - 1);
        // B handle identities preserved across versions; tail rank intact.
        assert(packed_bundle->groups[0].b_k == b0_k);
        assert(packed_bundle->groups[1].b_k == b1_k);
        assert(packed_bundle->groups[2].b_k == b2_k);
        assert(packed_bundle->groups[2].rank_k == 8 && packed_bundle->groups[2].rank_v == 8);
        xkv_location gone_loc;
        assert(!store.find_location(8001, gone_loc));
        xkv_location surv_loc;
        assert(store.find_location(8000, surv_loc) && surv_loc.row == 0);
        assert(store.find_location(8002, surv_loc) && surv_loc.row == 1);

        // D. Fingerprint preserved through pack and non-zero
        assert(packed_bundle->layer_group_map_fingerprint != 0);
        assert(packed_bundle->layer_group_map_fingerprint == bundle->layer_group_map_fingerprint);
        assert(packed_bundle->descriptor_fingerprint != 0);

        // E. Reference / ablation profile allowing FP16 landmark
        xkv_bundle_sealing_params ref_sparams = sparams;
        ref_sparams.profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
        ref_sparams.landmark_type = GGML_TYPE_F16;
        ref_sparams.landmark_factory = [](
            uint32_t /*group_index*/,
            const encoded_matrix & /*enc_a_k*/,
            const encoded_matrix & enc_b_k,
            const int64_t * row_positions,
            uint64_t lm_rows,
            factor_workspace_span scratch,
            encoded_matrix & out_lm,
            std::vector<xkv_landmark_chunk> & out_chunks,
            std::string * /*err*/
        ) -> bool {
            assert(row_positions != nullptr && lm_rows > 0);
            assert(scratch.valid());
            const uint32_t chunks = ((uint32_t) lm_rows + 7) / 8;
            const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
            codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_F16, orientation::token_major, {chunks, cols}, 0, 777);
            std::vector<float> lm_data((size_t) chunks * cols, 0.1f);
            out_lm = encode_matrix(desc, lm_data.data(), lm_data.size());
            out_chunks.clear();
            uint32_t rem = (uint32_t) lm_rows;
            uint32_t off = 0;
            for (uint32_t c = 0; c < chunks; ++c) {
                uint32_t cnt = std::min(8u, rem);
                out_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                off += cnt;
                rem -= cnt;
            }
            return true;
        };

        std::vector<uint64_t> ref_pids(n_tokens);
        std::vector<uint64_t> ref_gens(n_tokens, 1);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            ref_pids[i] = 9000 + i;
            store.register_hot_payload(ref_pids[i], i, ref_gens[i], xkv_state::hot_committed);
        }
        auto ref_res = store.seal_segment_bundle({g0, g1, g2}, ref_pids, ref_gens, ref_sparams);
        assert(ref_res.success);
        auto ref_bundle = store.get_segment(ref_res.segment_id);
        assert(ref_bundle != nullptr);
        assert(ref_bundle->groups[0].landmark.desc.type == GGML_TYPE_F16);
    }

    // Test 11: Exact epoch deltas, idempotent registration, and overflow policy
    {
        std::cout << "[Test 11] Exact epoch deltas and idempotent registration..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        uint64_t l0 = store.live_epoch();
        uint64_t b0 = store.binding_epoch();
        uint64_t c0 = store.content_epoch();

        // New payload -> bumps live + binding
        store.register_hot_payload(1101, 0, 1, xkv_state::hot_writing);
        assert(store.live_epoch() == l0 + 1);
        assert(store.binding_epoch() == b0 + 1);
        assert(store.content_epoch() == c0);

        // Idempotent exact repeat -> NO bumps!
        store.register_hot_payload(1101, 0, 1, xkv_state::hot_writing);
        assert(store.live_epoch() == l0 + 1);
        assert(store.binding_epoch() == b0 + 1);
        assert(store.content_epoch() == c0);

        // hot_writing -> hot_committed -> bumps content epoch
        assert(store.commit_hot_payload(1101));
        assert(store.content_epoch() == c0 + 1);

        // Rebind physical row -> bumps binding epoch
        uint64_t b1 = store.binding_epoch();
        store.register_hot_payload(1101, 5, 1, xkv_state::hot_committed);
        assert(store.binding_epoch() == b1 + 1);

        // Generation bump -> bumps binding + content epoch
        uint64_t b2 = store.binding_epoch();
        uint64_t c1 = store.content_epoch();
        store.register_hot_payload(1101, 5, 2, xkv_state::hot_committed);
        assert(store.binding_epoch() == b2 + 1);
        assert(store.content_epoch() == c1 + 1);
    }

    // Test 12: Atomic register_hot_payloads failure injection & validation
    {
        std::cout << "[Test 12] Atomic register_hot_payloads failure injection..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        std::string err;
        // Reject duplicate PID in batch
        xkv_hot_payload_binding b1 = {1201, 0, 1, xkv_state::hot_writing};
        xkv_hot_payload_binding b2 = {1201, 1, 1, xkv_state::hot_writing}; // Duplicate
        assert(!store.register_hot_payloads({b1, b2}, &err));
        assert(err.find("duplicate payload_id") != std::string::npos);

        // Reject 0 generation
        xkv_hot_payload_binding b3 = {1202, 0, 0, xkv_state::hot_writing};
        assert(!store.register_hot_payloads({b3}, &err));
        assert(err.find("storage_generation cannot be 0") != std::string::npos);

        // Failure injection hook leaves zero partial mutations
        xkv_hot_payload_binding b4 = {1203, 0, 1, xkv_state::hot_writing};
        xkv_hot_payload_binding b5 = {1204, 1, 1, xkv_state::hot_writing};
        // Injected throws never escape: false + err, zero mutation.
        auto stamp_h = store.current_stamp();
        assert(!store.register_hot_payloads({b4, b5}, &err, []() {
            throw std::bad_alloc();
        }));
        assert(!err.empty());
        assert(store.current_stamp() == stamp_h);
        xkv_location loc;
        assert(!store.find_location(1203, loc));
        assert(!store.find_location(1204, loc));
    }

    // Test 13: Nonce isolation on concurrent / overlapping seal transactions
    {
        std::cout << "[Test 13] Seal tx nonce isolation..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 64;
        llama_xkv_cache_store store(cparams);

        store.register_hot_payload(1301, 0, 1, xkv_state::hot_committed);
        store.register_hot_payload(1302, 1, 1, xkv_state::hot_committed);

        // Start a seal transaction on 1301 by triggering a seal that fails at error threshold
        // During seal execution, 1301 is locked with a real nonzero seal_tx_nonce
        // Single-payload seal: n_rows == 1, so ranks must satisfy rank <= rows.
        // Rank-1 varying data keeps dims valid; the 1e-12 error gate still trips
        // on codec quantization noise and exercises the rollback path.
        std::vector<float> k_dummy(64);
        std::vector<float> v_dummy(64);
        for (size_t i = 0; i < 64; ++i) {
            k_dummy[i] = std::sin((float) i * 0.7f);
            v_dummy[i] = std::cos((float) i * 0.3f);
        }
        xkv_factor_group_input g;
        g.group_index = 0;
        g.owning_layers = {0};
        g.rank_k = 1; g.rank_v = 1;
        g.total_dim_k = 64; g.total_dim_v = 64;
        g.layer_feature_offsets_k = {0}; g.layer_feature_dims_k = {64};
        g.layer_feature_offsets_v = {0}; g.layer_feature_dims_v = {64};
        g.row_positions = {0};
        g.hot_bytes_per_row_k = {256};
        g.hot_bytes_per_row_v = {256};
        g.canonical_k_data = k_dummy.data(); g.k_rows = 1; g.k_cols = 64;
        g.canonical_v_data = v_dummy.data(); g.v_rows = 1; g.v_cols = 64;

        xkv_bundle_sealing_params sparams;
        sparams.rank_k = 1; sparams.rank_v = 1;
        sparams.max_relative_error = 1e-12; // strict threshold forces failure and rollback
        sparams.flat_type_k = GGML_TYPE_F32; sparams.flat_type_v = GGML_TYPE_F32;

        auto res_fail = store.seal_segment_bundle({g}, {1301}, {1}, sparams);
        assert(!res_fail.success);
        assert(res_fail.skip_reason == xkv_skip_reason::error_threshold_exceeded);

        // After rollback, 1301 must be cleanly restored to hot_committed with nonce 0
        xkv_state st;
        assert(store.find_payload_state(1301, st) && st == xkv_state::hot_committed);
        xkv_location loc_restored;
        assert(store.find_location(1301, loc_restored));
        assert(loc_restored.seal_tx_nonce == 0);

        // Unlocked payload state query
        assert(store.find_payload_state(1302, st) && st == xkv_state::hot_committed);
        assert(!store.find_payload_state(9999, st)); // Absent returns false
    }

    // Test 14: Atomic remove_payloads batch API with failure injection & generation check
    {
        std::cout << "[Test 14] Atomic remove_payloads batch API..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        store.register_hot_payload(1401, 0, 10, xkv_state::hot_committed);
        store.register_hot_payload(1402, 1, 20, xkv_state::hot_committed);

        // A. Stale generation rejects without mutation
        std::string err;
        xkv_payload_removal_result rem_res;
        assert(!store.remove_payloads({1401, 1402}, {10, 999}, &rem_res, &err)); // 1402 gen mismatch
        assert(err.find("generation mismatch") != std::string::npos);
        xkv_location loc;
        assert(store.find_location(1401, loc));
        assert(store.find_location(1402, loc));

        // B. Failure hook causes clean rollback with zero partial changes
        // Injected throws never escape: false + err, zero mutation.
        auto stamp_h = store.current_stamp();
        assert(!store.remove_payloads({1401, 1402}, {10, 20}, &rem_res, &err, []() {
            throw std::bad_alloc();
        }));
        assert(!err.empty());
        assert(store.current_stamp() == stamp_h);
        assert(store.find_location(1401, loc));
        assert(store.find_location(1402, loc));

        // C. Clean removal succeeds and returns physical rows and kinds
        assert(store.remove_payloads({1401, 1402}, {10, 20}, &rem_res, &err));
        assert(rem_res.success);
        assert(rem_res.removed_payload_ids.size() == 2);
        assert(rem_res.released_hot_rows == std::vector<uint32_t>({0, 1}));
        assert(rem_res.removed_kinds == std::vector<xkv_location_kind>({xkv_location_kind::hot, xkv_location_kind::hot}));
        assert(!store.find_location(1401, loc));
        assert(!store.find_location(1402, loc));
    }

    // Test 15: Epoch overflow rejection policy (UINT64_MAX leaves maps completely unchanged)
    {
        std::cout << "[Test 15] Epoch overflow rejection at UINT64_MAX..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        store.register_hot_payload(1501, 0, 1, xkv_state::hot_committed);
        auto stamp_before = store.current_stamp();

        // Set live_epoch to UINT64_MAX
        store.set_epoch_for_testing(llama_xkv_cache_store::bump_flag_live, UINT64_MAX);

        std::string err;
        // Registering a NEW payload requires live_epoch bump; must fail without mutating map
        xkv_hot_payload_binding new_b = {1502, 1, 1, xkv_state::hot_writing};
        assert(!store.register_hot_payloads({new_b}, &err));
        assert(err.find("overflow") != std::string::npos);
        xkv_location loc;
        assert(!store.find_location(1502, loc)); // Not inserted!

        // Removing a payload also requires live_epoch bump; must fail cleanly
        assert(!store.remove_payloads({1501}, {1}, nullptr, &err));
        assert(err.find("overflow") != std::string::npos);
        assert(store.find_location(1501, loc)); // 1501 remains untouched!
        // Failed mutations leave every other epoch untouched.
        assert(store.binding_epoch() == stamp_before.binding_epoch);
        assert(store.content_epoch() == stamp_before.content_epoch);
        assert(store.codec_epoch() == stamp_before.codec_epoch);
    }

    // Test 15b: can_clear/clear atomicity (overflow refuses, success clears).
    {
        std::cout << "[Test 15b] can_clear/clear atomicity..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);
        store.register_hot_payload(1511, 0, 1, xkv_state::hot_committed);
        store.register_hot_payload(1512, 1, 1, xkv_state::hot_committed);
        std::string err;
        assert(store.can_clear(&err));
        auto stamp_c = store.current_stamp();
        assert(store.clear(&err));
        xkv_location loc;
        assert(!store.find_location(1511, loc));
        assert(!store.find_location(1512, loc));
        assert(store.get_accounting().active_segments == 0);
        assert(store.live_epoch() == stamp_c.live_epoch + 1);
        assert(store.content_epoch() == stamp_c.content_epoch + 1);
        assert(store.binding_epoch() == stamp_c.binding_epoch + 1);
        // Overflow: can_clear false, clear false, state fully intact.
        store.register_hot_payload(1513, 0, 1, xkv_state::hot_committed);
        store.set_epoch_for_testing(llama_xkv_cache_store::bump_flag_live, UINT64_MAX);
        assert(!store.can_clear(&err));
        assert(err.find("overflow") != std::string::npos);
        assert(!store.clear(&err));
        assert(store.find_location(1513, loc));
    }

    // Test 16: First- and middle-row deletion on REFERENCE compacts with exact
    // survivor byte identity, correct row remap, shared B, preserved ranks.
    {
        std::cout << "[Test 16] First/middle-row deletion compacts exactly..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        const uint32_t n_rows = 6;
        const std::vector<uint64_t> pids = {601, 602, 603, 604, 605, 606};
        const std::vector<uint64_t> gens = {1, 1, 1, 1, 1, 1};
        for (size_t i = 0; i < n_rows; ++i) {
            store.register_hot_payload(pids[i], (uint32_t) i, gens[i], xkv_state::hot_committed);
        }
        assert(store.mark_seal_candidates(pids, gens, nullptr));

        // Non-default ranks prove COW preservation (not struct-default 16).
        auto seg = create_valid_candidate(store, n_rows, 8, 4);
        const uint64_t seg_id = seg->segment_id;
        const size_t stride_a_k = seg->groups[0].a_k.desc.row_stride_bytes;
        const size_t stride_a_v = seg->groups[0].a_v.desc.row_stride_bytes;
        for (uint32_t r = 0; r < n_rows; ++r) {
            std::memset(seg->groups[0].a_k.bytes.data() + r * stride_a_k, (int)(0xC0 + r), stride_a_k);
            std::memset(seg->groups[0].a_v.bytes.data() + r * stride_a_v, (int)(0xD0 + r), stride_a_v);
        }
        std::vector<uint8_t> row1_k(seg->groups[0].a_k.bytes.begin() + 1 * stride_a_k,
                                       seg->groups[0].a_k.bytes.begin() + 2 * stride_a_k);
        std::vector<uint8_t> row2_k(seg->groups[0].a_k.bytes.begin() + 2 * stride_a_k,
                                       seg->groups[0].a_k.bytes.begin() + 3 * stride_a_k);
        std::vector<uint8_t> row4_k(seg->groups[0].a_k.bytes.begin() + 4 * stride_a_k,
                                       seg->groups[0].a_k.bytes.begin() + 5 * stride_a_k);
        std::vector<uint8_t> row5_k(seg->groups[0].a_k.bytes.begin() + 5 * stride_a_k,
                                       seg->groups[0].a_k.bytes.begin() + 6 * stride_a_k);
        std::shared_ptr<const encoded_matrix> b_k = seg->groups[0].b_k;
        std::shared_ptr<const encoded_matrix> b_v = seg->groups[0].b_v;
        std::string err;
        assert(store.publish_candidate(seg, pids, gens, &err));

        // Delete the FIRST row (601) and a MIDDLE row (604, original row 3).
        xkv_payload_removal_result rem_res;
        assert(store.remove_payloads({601, 604}, {1, 1}, &rem_res, &err));
        assert(rem_res.success);
        auto cur = store.get_segment(seg_id);
        assert(cur != nullptr);
        assert(cur->segment_version == 2);
        assert(cur->n_rows == 4 && cur->n_live_rows == 4);
        assert(cur->row_payload_ids == std::vector<uint64_t>({602, 603, 605, 606}));
        // Survivor byte identity without re-encoding.
        // Survivors compact to [602(row1), 603(row2), 605(row4), 606(row5)].
        assert(std::memcmp(cur->groups[0].a_k.bytes.data() + 0 * stride_a_k, row1_k.data(), stride_a_k) == 0);
        assert(std::memcmp(cur->groups[0].a_k.bytes.data() + 1 * stride_a_k, row2_k.data(), stride_a_k) == 0);
        assert(std::memcmp(cur->groups[0].a_k.bytes.data() + 2 * stride_a_k, row4_k.data(), stride_a_k) == 0);
        assert(std::memcmp(cur->groups[0].a_k.bytes.data() + 3 * stride_a_k, row5_k.data(), stride_a_k) == 0);
        // B handles shared, ranks + exact group metadata preserved.
        assert(cur->groups[0].b_k == b_k);
        assert(cur->groups[0].b_v == b_v);
        assert(cur->groups[0].rank_k == 8 && cur->groups[0].rank_v == 4);
        assert(cur->groups[0].owning_layers == std::vector<uint32_t>({0, 1, 2, 3}));
        assert(cur->groups[0].total_dim_k == 64 && cur->groups[0].total_dim_v == 64);
        // Survivor locations rebound to compacted rows.
        xkv_location loc;
        assert(store.find_location(602, loc) && loc.row == 0 && loc.segment_version == 2);
        assert(store.find_location(603, loc) && loc.row == 1);
        assert(store.find_location(605, loc) && loc.row == 2);
        assert(store.find_location(606, loc) && loc.row == 3);
        assert(!store.find_location(601, loc) && !store.find_location(604, loc));
    }

    // Shared small seal recipe (proven gate-passing shape) for Tests 17/18/22.
    // Single group, 64 tokens x dim 32, rank 4, F16 flat, lenient gates.
    auto make_small_seal_group = [](uint32_t n_tokens, std::vector<float> & k_out, std::vector<float> & v_out) {
        k_out.resize(n_tokens * 32);
        v_out.resize(n_tokens * 32);
        for (uint32_t r = 0; r < n_tokens; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                float v = std::sin((float) r * 0.31f) * std::cos((float) c * 0.23f);
                k_out[r * 32 + c] = v;
                v_out[r * 32 + c] = v * 0.5f;
            }
        }
        xkv_factor_group_input g;
        g.group_index = 0;
        g.owning_layers = {0, 1};
        g.rank_k = 4;
        g.rank_v = 4;
        g.total_dim_k = 32;
        g.total_dim_v = 32;
        g.row_positions.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) g.row_positions[i] = (int64_t) i;
        g.hot_bytes_per_row_k = {32, 32};
        g.hot_bytes_per_row_v = {32, 32};
        g.layer_feature_offsets_k = {0, 16};
        g.layer_feature_dims_k = {16, 16};
        g.layer_feature_offsets_v = {0, 16};
        g.layer_feature_dims_v = {16, 16};
        g.canonical_k_data = k_out.data();
        g.k_rows = n_tokens;
        g.k_cols = 32;
        g.canonical_v_data = v_out.data();
        g.v_rows = n_tokens;
        g.v_cols = 32;
        return g;
    };

    // Test 17: Seal nonces are nonzero, monotonic, observed in real store state
    // mid-seal; rollback clears only its own nonce.
    {
        std::cout << "[Test 17] Seal nonce ownership and monotonicity..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 64;
        llama_xkv_cache_store store(cparams);
        const uint32_t n = 64;

        uint64_t seen_nonce_a = 0;
        auto recording_factory = [&](uint32_t, const encoded_matrix & /*enc_a_k*/,
                                       const encoded_matrix & enc_b_k,
                                       const int64_t * row_positions, uint64_t lm_rows,
                                       factor_workspace_span scratch,
                                       encoded_matrix & out_lm,
                                       std::vector<xkv_landmark_chunk> & out_chunks,
                                       std::string * err) -> bool {
            xkv_location loc;
            // Real store state mid-seal: payload locked with a nonzero nonce.
            assert(store.find_location(1701, loc) || store.find_location(1801, loc));
            uint64_t mid_nonce = 0;
            if (store.find_location(1701, loc)) mid_nonce = loc.seal_tx_nonce;
            if (store.find_location(1801, loc) && loc.seal_tx_nonce != 0) mid_nonce = loc.seal_tx_nonce;
            assert(mid_nonce != 0);
            if (seen_nonce_a == 0) seen_nonce_a = mid_nonce;
            else assert(mid_nonce > seen_nonce_a);
            assert(row_positions != nullptr && lm_rows == n);
            assert(scratch.valid());
            const uint32_t chunks = ((uint32_t) lm_rows + 7) / 8;
            const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
            codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                              orientation::token_major, {chunks, cols}, 0, 777);
            std::vector<float> lm((size_t) chunks * cols, 0.1f);
            out_lm = encode_matrix(desc, lm.data(), lm.size());
            out_chunks.clear();
            uint32_t rem = (uint32_t) lm_rows;
            uint32_t off = 0;
            for (uint32_t c = 0; c < chunks; ++c) {
                uint32_t cnt = std::min(8u, rem);
                out_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                off += cnt;
                rem -= cnt;
            }
            (void) err;
            return true;
        };

        std::vector<float> k1, v1;
        auto g1 = make_small_seal_group(n, k1, v1);
        std::vector<uint64_t> pids_a(n), gens_a(n, 1);
        for (uint32_t i = 0; i < n; ++i) {
            pids_a[i] = 1701 + i;
            store.register_hot_payload(pids_a[i], i, 1, xkv_state::hot_committed);
        }
        xkv_bundle_sealing_params sp;
        sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
        sp.max_relative_error = 0.99;
        sp.min_saving_ratio = 0.05;
        sp.flat_type_k = GGML_TYPE_F16;
        sp.flat_type_v = GGML_TYPE_F16;
        sp.landmark_factory = recording_factory;
        auto res_a = store.seal_segment_bundle({g1}, pids_a, gens_a, sp);
        assert(res_a.success);
        const uint64_t nonce_a = seen_nonce_a;
        assert(nonce_a != 0);
        // After success the seal nonce is cleared on factored locations.
        xkv_location la;
        assert(store.find_location(1701, la) && la.seal_tx_nonce == 0);
        assert(la.state == xkv_state::factored);

        std::vector<float> k2, v2;
        auto g2 = make_small_seal_group(n, k2, v2);
        std::vector<uint64_t> pids_b(n), gens_b(n, 1);
        for (uint32_t i = 0; i < n; ++i) {
            pids_b[i] = 1801 + i;
            store.register_hot_payload(pids_b[i], i, 1, xkv_state::hot_committed);
        }
        auto res_b = store.seal_segment_bundle({g2}, pids_b, gens_b, sp);
        assert(res_b.success);
        // Monotonic: second seal observed a strictly larger nonce mid-seal.
        assert(seen_nonce_a > nonce_a);
        xkv_location lb;
        assert(store.find_location(1801, lb) && lb.seal_tx_nonce == 0);
    }

    // Test 18: Stale generations reject BEFORE factorization (factory never
    // runs); state, epochs, and bytes are unchanged.
    {
        std::cout << "[Test 18] Stale generation rejects before factorization..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 64;
        llama_xkv_cache_store store(cparams);
        const uint32_t n = 64;
        std::vector<float> kk, vv;
        auto g = make_small_seal_group(n, kk, vv);
        std::vector<uint64_t> pids(n), gens(n, 1);
        for (uint32_t i = 0; i < n; ++i) {
            pids[i] = 1901 + i;
            store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
        }
        bool factory_called = false;
        xkv_bundle_sealing_params sp;
        sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
        sp.max_relative_error = 0.99;
        sp.min_saving_ratio = 0.05;
        sp.flat_type_k = GGML_TYPE_F16;
        sp.flat_type_v = GGML_TYPE_F16;
        sp.landmark_factory = [&](uint32_t, const encoded_matrix &, const encoded_matrix &,
                                    const int64_t *, uint64_t, factor_workspace_span,
                                    encoded_matrix &, std::vector<xkv_landmark_chunk> &, std::string *) -> bool {
            factory_called = true;
            return false;
        };
        auto stamp_before = store.current_stamp();
        auto acc_before = store.get_accounting();
        size_t sealed_before = store.get_sealed_count();
        std::vector<uint64_t> stale_gens = gens;
        stale_gens[5] = 999; // Stale generation for one payload.
        auto res = store.seal_segment_bundle({g}, pids, stale_gens, sp);
        assert(!res.success);
        assert(!factory_called); // Rejected before any factorization work.
        for (uint64_t pid : pids) {
            xkv_state st;
            assert(store.find_payload_state(pid, st) && st == xkv_state::hot_committed);
            xkv_location loc;
            assert(store.find_location(pid, loc) && loc.kind == xkv_location_kind::hot);
            assert(loc.seal_tx_nonce == 0);
        }
        assert(store.current_stamp() == stamp_before);
        auto acc_after = store.get_accounting();
        assert(acc_after.live_payload_bytes == acc_before.live_payload_bytes);
        assert(acc_after.allocated_bytes == acc_before.allocated_bytes);
        assert(acc_after.active_segments == acc_before.active_segments);
        assert(store.get_sealed_count() == sealed_before);
    }

// Test 19: LANDMARKS rebuild accept compacts with valid landmarks;
// refusal aborts atomically; default (no callback) tombstones.
{
        std::cout << "[Test 19] Landmark rebuild accept/refuse/tombstone..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);
        const uint32_t bn = 16;
        std::vector<uint64_t> bpids(bn), bgens(bn, 1);
        for (uint32_t i = 0; i < bn; ++i) {
            bpids[i] = 2100 + i;
            store.register_hot_payload(bpids[i], i, 1, xkv_state::hot_committed);
}
        assert(store.mark_seal_candidates(bpids, bgens, nullptr));
        auto seg = create_valid_candidate(store, bn, 8, 8, 32, 32);
        const uint64_t seg_id = seg->segment_id;
        const uint32_t chunks = (bn + 7) / 8;
        codec_desc lm_desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                             orientation::token_major, {chunks, 32}, 0, 777);
        std::vector<float> lm_data((size_t) chunks * 32, 0.2f);
        seg->groups[0].landmark = encode_matrix(lm_desc, lm_data.data(), lm_data.size());
        seg->groups[0].landmark_chunks.clear();
        uint32_t rem = bn;
        uint32_t off = 0;
        for (uint32_t c = 0; c < chunks; ++c) {
            uint32_t cnt = std::min(8u, rem);
            seg->groups[0].landmark_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
            off += cnt;
            rem -= cnt;
        }
        seg->groups[0].landmark_table_fingerprint = compute_landmark_table_fingerprint(seg->groups[0].landmark_chunks);
        seg->profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
        std::string err;
        assert(store.publish_candidate(seg, bpids, bgens, &err));

        // Accept: rebuild supplies valid compacted landmarks (middle row out).
        auto rebuild_ok = [](const xkv_segment & /*old_seg*/,
                               const std::vector<uint32_t> & surviving,
                               xkv_segment & new_seg, std::string * /*err*/) -> bool {
            const uint32_t nn = (uint32_t) surviving.size();
            const uint32_t nch = (nn + 7) / 8;
            for (auto & ng : new_seg.groups) {
                codec_desc d = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                               orientation::token_major, {nch, ng.total_dim_k}, 0, 777);
                std::vector<float> z((size_t) nch * ng.total_dim_k, 0.0f);
                ng.landmark = encode_matrix(d, z.data(), z.size());
                ng.landmark_chunks.clear();
                uint32_t rem = nn;
                uint32_t off = 0;
                for (uint32_t c = 0; c < nch; ++c) {
                    uint32_t cnt = std::min(8u, rem);
                    ng.landmark_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                    off += cnt;
                    rem -= cnt;
}
                ng.landmark_table_fingerprint = compute_landmark_table_fingerprint(ng.landmark_chunks);
}
            return true;
        };
        const size_t stride = store.get_segment(seg_id)->groups[0].a_k.desc.row_stride_bytes;
        std::vector<uint8_t> first_row(store.get_segment(seg_id)->groups[0].a_k.bytes.begin(),
                                        store.get_segment(seg_id)->groups[0].a_k.bytes.begin() + stride);
        assert(store.remove_payload(2108, &err, rebuild_ok)); // Middle row 8 out.
        auto compacted = store.get_segment(seg_id);
        assert(compacted->segment_version == 2);
        assert(compacted->n_rows == bn - 1 && compacted->n_live_rows == bn - 1);
        assert(compacted->groups[0].landmark.desc.logical_shape.rows == (bn - 1 + 7) / 8);
        assert(compacted->groups[0].landmark.desc.logical_shape.cols == 32);
        assert(store.validate_candidate(compacted));
        assert(std::memcmp(compacted->groups[0].a_k.bytes.data(), first_row.data(), stride) == 0);
        assert(compacted->groups[0].rank_k == 8 && compacted->groups[0].rank_v == 8);

        // Refuse: callback false leaves version, rows, locations, stamp unchanged.
        auto stamp_before = store.current_stamp();
        auto rebuild_no = [](const xkv_segment &, const std::vector<uint32_t> &,
                               xkv_segment &, std::string * err) -> bool {
            if (err) *err = "no semantic source available";
            return false;
        };
        assert(!store.remove_payload(2100, &err, rebuild_no));
        assert(store.get_segment(seg_id)->segment_version == 2);
        assert(store.get_segment(seg_id)->n_rows == bn - 1);
        assert(store.current_stamp() == stamp_before);
        xkv_location loc;
        assert(store.find_location(2100, loc) && loc.row == 0);

        // Default (no callback): LANDMARKS removal refuses with zero mutation.
        // No first-N chunk copy, no layout duplication; old version intact.
        auto stamp_t0 = store.current_stamp();
        std::string err_t;
        assert(!store.remove_payload(2101, &err_t));
        assert(err_t.find("rebuild") != std::string::npos);
        auto intact = store.get_segment(seg_id);
        assert(intact->segment_version == 2);
        assert(intact->n_rows == bn - 1 && intact->n_live_rows == bn - 1);
        assert(store.current_stamp() == stamp_t0);
        assert(store.find_location(2101, loc) && loc.row == 1);
        assert(store.validate_candidate(intact));
}

// Test 20: Zero per-group rank and missing feature maps reject atomically.
{
        std::cout << "[Test 20] Zero-rank / missing-map rejection..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 64;
        llama_xkv_cache_store store(cparams);
        const uint32_t n = 64;
        std::vector<float> kk, vv;
        auto g = make_small_seal_group(n, kk, vv);
        std::vector<uint64_t> pids(n), gens(n, 1);
        for (uint32_t i = 0; i < n; ++i) {
            pids[i] = 2201 + i;
            store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
}
        xkv_bundle_sealing_params sp;
        sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
        sp.max_relative_error = 0.99;
        sp.min_saving_ratio = 0.05;
        sp.flat_type_k = GGML_TYPE_F16;
        sp.flat_type_v = GGML_TYPE_F16;
        auto stamp_before = store.current_stamp();
        auto bad_rank = g;
        bad_rank.rank_k = 0; // No fallback to params ranks.
        auto r1 = store.seal_segment_bundle({bad_rank}, pids, gens, sp);
        assert(!r1.success && r1.skip_reason == xkv_skip_reason::unsupported_config);
        auto bad_map = g;
        bad_map.layer_feature_offsets_k.clear();
        auto r2 = store.seal_segment_bundle({bad_map}, pids, gens, sp);
        assert(!r2.success && r2.skip_reason == xkv_skip_reason::unsupported_config);
        for (uint64_t pid : pids) {
            xkv_state st;
            assert(store.find_payload_state(pid, st) && st == xkv_state::hot_committed);
}
        assert(store.current_stamp() == stamp_before);
        assert(store.get_accounting().active_segments == 0);
}

// Test 21: Single-segment hook and overflow leave bytes/accounting unchanged.
{
        std::cout << "[Test 21] Mutation failure byte/state atomicity..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);
        const std::vector<uint64_t> pids = {2301, 2302, 2303, 2304};
        const std::vector<uint64_t> gens = {1, 1, 1, 1};
        for (size_t i = 0; i < pids.size(); ++i) {
            store.register_hot_payload(pids[i], (uint32_t) i, gens[i], xkv_state::hot_committed);
}
        assert(store.mark_seal_candidates(pids, gens, nullptr));
        auto seg = create_valid_candidate(store, 4);
        const uint64_t seg_id = seg->segment_id;
        std::string err;
        assert(store.publish_candidate(seg, pids, gens, &err));
        auto stamp_before = store.current_stamp();
        auto acc_before = store.get_accounting();
        xkv_location loc_before;
        assert(store.find_location(2302, loc_before));
        // Single-segment hook fires at the same precommit phase: zero changes.
        xkv_batch_mutation bm;
        bm.segments_to_pack = {seg_id};
        bm.test_failure_hook = []() { throw std::bad_alloc(); };
        xkv_mutation_result res;
        assert(!store.execute_mutation_transaction(bm, &res, &err));
        assert(store.current_stamp() == stamp_before);
        auto acc_after = store.get_accounting();
        assert(acc_after.live_payload_bytes == acc_before.live_payload_bytes);
        assert(acc_after.allocated_bytes == acc_before.allocated_bytes);
        assert(acc_after.shared_b_bytes == acc_before.shared_b_bytes);
        assert(store.get_segment(seg_id)->segment_version == 1);
        xkv_location loc_after;
        assert(store.find_location(2302, loc_after) && loc_after.row == loc_before.row);
        // Overflow fail-closed: binding epoch at MAX refuses pack with no mutation.
        store.set_epoch_for_testing(llama_xkv_cache_store::bump_flag_binding, UINT64_MAX);
        assert(!store.pack_segment(seg_id, &err));
        assert(err.find("overflow") != std::string::npos);
        assert(store.get_segment(seg_id)->segment_version == 1);
        assert(store.find_location(2301, loc_after));
}

// Test 22: Seal precommit gate and SHADOW evaluate-only.
{
        std::cout << "[Test 22] Seal precommit gate and evaluate-only..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 64;
        llama_xkv_cache_store store(cparams);
        const uint32_t n = 64;
        auto new_pids = [&](uint64_t base) {
            std::vector<uint64_t> p(n);
            for (uint32_t i = 0; i < n; ++i) {
                p[i] = base + i;
                store.register_hot_payload(p[i], i, 1, xkv_state::hot_committed);
}
            return p;
        };
        auto seal_params = []() {
            xkv_bundle_sealing_params sp;
            sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
            sp.max_relative_error = 0.99;
            sp.min_saving_ratio = 0.05;
            sp.flat_type_k = GGML_TYPE_F16;
            sp.flat_type_v = GGML_TYPE_F16;
            return sp;
        };
        // (a) Precommit false refuses with zero mutation (aborted, states restored).
        // The simulated pool records side effects: refusal must release nothing.
        std::vector<float> ka, va;
        auto ga = make_small_seal_group(n, ka, va);
        auto pa = new_pids(2401);
        std::vector<uint64_t> ha(n, 1);
        size_t gate_calls_a = 0;
        std::vector<uint64_t> pool_released_a;
        auto r_refuse = store.seal_segment_bundle({ga}, pa, ha, seal_params(),
            [&](const xkv_seal_precommit_ctx &, std::string * err) -> bool {
                gate_calls_a++;
                if (err) *err = "hot pool busy";
                return false;
            });
        assert(!r_refuse.success);
        assert(gate_calls_a == 1); // Gate ran exactly once, then refused.
        assert(pool_released_a.empty()); // Pool untouched on refusal.
        assert(r_refuse.release_plan.released_payload_ids.empty());
        for (uint64_t pid : pa) {
            xkv_state st;
            assert(store.find_payload_state(pid, st) && st == xkv_state::hot_committed);
            xkv_location loc;
            assert(store.find_location(pid, loc) && loc.kind == xkv_location_kind::hot);
            assert(loc.seal_tx_nonce == 0);
}
        assert(store.get_accounting().active_segments == 0);
        // (b) Precommit true observes exact rows/generations and commits.
        // The gate OWNS the pool release here; the emitted plan is consumed.
        std::vector<float> kb, vb;
        auto gb = make_small_seal_group(n, kb, vb);
        auto pb = new_pids(2501);
        std::vector<uint64_t> hb(n, 1);
        xkv_seal_precommit_ctx seen;
        bool gate_called = false;
        size_t gate_calls_b = 0;
        std::vector<uint64_t> pool_released_b;
        auto r_ok = store.seal_segment_bundle({gb}, pb, hb, seal_params(),
            [&](const xkv_seal_precommit_ctx & ctx, std::string *) -> bool {
                gate_called = true;
                gate_calls_b++;
                seen = ctx;
                pool_released_b = ctx.payload_ids; // Simulated pool release.
                return true;
            });
        assert(r_ok.success);
        assert(gate_called);
        assert(gate_calls_b == 1); // Exactly-once gate invocation.
        assert(pool_released_b == pb); // Pool released exactly the sealed set.
        assert(seen.segment_id == r_ok.segment_id);
        assert(seen.payload_ids == pb && seen.generations == hb);
        assert(seen.physical_rows.size() == n);
        for (uint32_t i = 0; i < n; ++i) assert(seen.physical_rows[i] == i);
        assert(store.validate_hot_release_plan(r_ok.release_plan));
        // Consumed plan: re-applying the pool release is a no-op by contract.
        assert(r_ok.release_plan.pool_released_at_precommit == true);
        {
            size_t releases_before = pool_released_b.size();
            if (!r_ok.release_plan.pool_released_at_precommit) {
                pool_released_b.insert(pool_released_b.end(), pb.begin(), pb.end());
}
            assert(pool_released_b.size() == releases_before); // No double release.
}
        // (c) SHADOW evaluate-only: success stats, nothing published/released.
        std::vector<float> kc, vc;
        auto gc = make_small_seal_group(n, kc, vc);
        auto pc = new_pids(2601);
        std::vector<uint64_t> hc(n, 1);
        size_t sealed_before = store.get_sealed_count();
        size_t segments_before = store.get_accounting().active_segments;
        auto r_eval = store.evaluate_segment_bundle({gc}, pc, hc, seal_params());
        assert(r_eval.success);
        assert(r_eval.segment_id == 0);
        assert(r_eval.release_plan.released_payload_ids.empty());
        assert(r_eval.release_plan.pool_released_at_precommit == false);
        assert(r_eval.saved_bytes > 0);
        for (uint64_t pid : pc) {
            xkv_state st;
            assert(store.find_payload_state(pid, st) && st == xkv_state::hot_committed);
}
        assert(store.get_accounting().active_segments == segments_before);
        assert(store.get_sealed_count() == sealed_before);
    }

    // Test 23: Epoch reservation provider lifecycle (token, exclusion,
    // single-use spend, apply-once, abort, exhaustion).
    {
        std::cout << "[Test 23] Reservation provider lifecycle..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);
        std::string err;

        // Reserve ok: nonzero token, base == current, reserved = +1 content.
        xkv_stamp_reservation res;
        assert(store.preflight_content_advance(&res, &err));
        assert(res.valid && res.token != 0);
        assert(res.base_stamp == store.current_stamp());
        assert(res.reserved_stamp.content_epoch == res.base_stamp.content_epoch + 1);
        assert(store.validate_reservation_token(res.token));
        assert(!store.validate_reservation_token(res.token + 1));
        assert(!store.validate_reservation_token(0));

        // Second reservation refused while held.
        xkv_stamp_reservation res2;
        assert(!store.preflight_content_advance(&res2, &err));

        // Every mutator refuses while held (seal/commit/clear/bump/register).
        auto stamp_r = store.current_stamp();
        assert(!store.register_hot_payload(9901, 0, 1, xkv_state::hot_writing));
        assert(!store.commit_hot_payload(9901));
        assert(!store.mark_seal_candidates({9901}, {1}, nullptr));
        assert(!store.bump_content_epoch(&err));
        assert(!store.clear(&err));
        xkv_batch_mutation bm_r;
        assert(!store.execute_mutation_transaction(bm_r, nullptr, &err));
        std::vector<float> kd(64, 0.1f), vd(64, 0.1f);
        xkv_factor_group_input gd;
        gd.group_index = 0;
        gd.owning_layers = {0};
        gd.rank_k = 1; gd.rank_v = 1;
        gd.total_dim_k = 64; gd.total_dim_v = 64;
        gd.row_positions = {0};
        gd.hot_bytes_per_row_k = {256};
        gd.hot_bytes_per_row_v = {256};
        gd.layer_feature_offsets_k = {0}; gd.layer_feature_dims_k = {64};
        gd.layer_feature_offsets_v = {0}; gd.layer_feature_dims_v = {64};
        gd.canonical_k_data = kd.data(); gd.k_rows = 1; gd.k_cols = 64;
        gd.canonical_v_data = vd.data(); gd.v_rows = 1; gd.v_cols = 64;
        xkv_bundle_sealing_params spd;
        spd.max_relative_error = 0.99;
        auto r_seal = store.seal_segment_bundle({gd}, {9901}, {1}, spd);
        assert(!r_seal.success && r_seal.skip_reason == xkv_skip_reason::aborted);
        assert(store.current_stamp() == stamp_r); // Zero mutation anywhere.

        // Forged token remove refused.
        xkv_payload_removal_result rr;
        assert(!store.remove_payloads({9901}, {1}, &rr, &err, nullptr, nullptr, nullptr, res.token + 1));

        // Abort before commit releases; mutators work again.
        store.abort_content_advance(res);
        assert(!store.validate_reservation_token(res.token));
        assert(store.register_hot_payload(9901, 0, 1, xkv_state::hot_writing));
        xkv_location l9901;
        assert(store.find_location(9901, l9901));

        // Token exhaustion fails closed via the test seam (never wraps).
        store.set_next_res_token_for_testing(UINT64_MAX);
        xkv_stamp_reservation res_x;
        assert(!store.preflight_content_advance(&res_x, &err));
        assert(!res_x.valid);
        store.set_next_res_token_for_testing(0);
        assert(!store.preflight_content_advance(&res_x, &err));
        store.set_next_res_token_for_testing(7);
        assert(store.preflight_content_advance(&res_x, &err));
        assert(res_x.token == 7); // No reuse games: exact counter value.
        store.abort_content_advance(res_x);
    }

    // Test 24: Token-carried remove + apply feedback (no double bump).
    {
        std::cout << "[Test 24] Token remove then apply..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);
        std::string err;
        store.register_hot_payload(9911, 0, 5, xkv_state::hot_committed);
        store.register_hot_payload(9912, 1, 6, xkv_state::hot_committed);
        xkv_stamp_reservation res;
        assert(store.preflight_content_advance(&res, &err));
        auto base = res.base_stamp;
        // Single token-carried batch remove with normal bumps.
        xkv_payload_removal_result rr;
        assert(store.remove_payloads({9911, 9912}, {5, 6}, &rr, &err,
                                      nullptr, nullptr, nullptr, res.token));
        assert(rr.success);
        xkv_location gone;
        assert(!store.find_location(9911, gone) && !store.find_location(9912, gone));
        // Second token-carried remove refused (single-use spent).
        assert(!store.remove_payloads({9911}, {5}, nullptr, &err,
                                      nullptr, nullptr, nullptr, res.token));
        // Feedback path: apply validates exact stamp, clears, NO second bump.
        xkv_removal_feedback fb;
        fb.has_store_stamp = true;
        fb.store_stamp = store.current_stamp();
        auto after = store.apply_content_advance(res, fb);
        assert(after == store.current_stamp());
        assert(after.live_epoch == base.live_epoch + 1);
        assert(after.binding_epoch == base.binding_epoch + 1);
        assert(after.content_epoch == base.content_epoch + 1); // Once, from remove.
        assert(!store.validate_reservation_token(res.token));
        // Re-apply is a no-op returning current.
        assert(store.apply_content_advance(res, fb) == after);
        // Abort after commit is a no-op too.
        store.abort_content_advance(res);
        assert(store.current_stamp() == after);
        // Forged feedback view cannot mask a mismatch: adopt refused.
        xkv_stamp_reservation res2;
        assert(store.preflight_content_advance(&res2, &err));
        xkv_removal_feedback bad_fb;
        bad_fb.has_store_stamp = true;
        bad_fb.store_stamp = store.current_stamp();
        bad_fb.store_stamp.view.topology_epoch += 1; // Forged drift.
        auto kept = store.apply_content_advance(res2, bad_fb);
        assert(kept == store.current_stamp());
        assert(kept.view.topology_epoch != bad_fb.store_stamp.view.topology_epoch);
        assert(!store.validate_reservation_token(res2.token));
    }

    // Test 25: Empty-feedback apply bumps content exactly once.
    {
        std::cout << "[Test 25] Reserved advance without removal..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);
        std::string err;
        store.register_hot_payload(9921, 0, 1, xkv_state::hot_committed);
        xkv_stamp_reservation res;
        assert(store.preflight_content_advance(&res, &err));
        auto base = res.base_stamp;
        xkv_removal_feedback empty;
        auto after = store.apply_content_advance(res, empty);
        assert(after == res.reserved_stamp);
        assert(after.content_epoch == base.content_epoch + 1);
        assert(after.live_epoch == base.live_epoch);
        assert(after.binding_epoch == base.binding_epoch);
        assert(!store.validate_reservation_token(res.token));
    }

    // Test 26: Configurable landmark chunk tokens (1, 7, 16) with partial tails,
    // explicit landmark_chunks metadata, and T-1 store capacity gating.
    {
        std::cout << "[Test 26] Configurable chunk tokens (1, 7, 16), partial tails, and T-1 cap..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 64;

        // Sub-test A: chunk_tokens = 7 over 25 rows (3 full chunks of 7, 1 partial tail chunk of 4)
        {
            llama_xkv_cache_store store(cparams);
            const uint32_t n = 25;
            std::vector<float> kk, vv;
            auto g = make_small_seal_group(n, kk, vv);
            std::vector<uint64_t> pids(n), gens(n, 1);
            for (uint32_t i = 0; i < n; ++i) {
                pids[i] = 10001 + i;
                store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
            }

            xkv_bundle_sealing_params sp;
            sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
            sp.chunk_tokens = 7;
            sp.max_relative_error = 0.99;
            sp.min_saving_ratio = 0.01;
            sp.flat_type_k = GGML_TYPE_F16;
            sp.flat_type_v = GGML_TYPE_F16;
            sp.landmark_factory = [](
                uint32_t /*group_index*/,
                const encoded_matrix & /*enc_a_k*/,
                const encoded_matrix & enc_b_k,
                const int64_t * row_positions,
                uint64_t lm_rows,
                factor_workspace_span scratch,
                encoded_matrix & out_lm,
                std::vector<xkv_landmark_chunk> & out_chunks,
                std::string * /*err*/
            ) -> bool {
                assert(row_positions != nullptr && lm_rows == 25);
                assert(scratch.valid());
                const uint32_t chunk_toks = 7;
                const uint32_t chunks = ((uint32_t) lm_rows + chunk_toks - 1) / chunk_toks; // 4 chunks
                const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
                codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                                  orientation::token_major, {chunks, cols}, 0, 777);
                std::vector<float> lm((size_t) chunks * cols, 0.1f);
                out_lm = encode_matrix(desc, lm.data(), lm.size());

                out_chunks.clear();
                uint32_t rem = (uint32_t) lm_rows;
                uint32_t off = 0;
                for (uint32_t c = 0; c < chunks; ++c) {
                    uint32_t cnt = std::min(chunk_toks, rem);
                    out_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                    off += cnt;
                    rem -= cnt;
                }
                return true;
            };

            auto res = store.seal_segment_bundle({g}, pids, gens, sp);
            assert(res.success);
            auto seg = store.get_segment(res.segment_id);
            assert(seg != nullptr);
            assert(seg->groups[0].landmark_chunks.size() == 4);
            assert(seg->groups[0].landmark_chunks[0].row_begin == 0 && seg->groups[0].landmark_chunks[0].row_count == 7);
            assert(seg->groups[0].landmark_chunks[1].row_begin == 7 && seg->groups[0].landmark_chunks[1].row_count == 7);
            assert(seg->groups[0].landmark_chunks[2].row_begin == 14 && seg->groups[0].landmark_chunks[2].row_count == 7);
            // Partial tail chunk of 4:
            assert(seg->groups[0].landmark_chunks[3].row_begin == 21 && seg->groups[0].landmark_chunks[3].row_count == 4);
            assert(seg->groups[0].landmark_table_fingerprint != 0);
        }

        // Sub-test B: chunk_tokens = 16 over 35 rows (2 full chunks of 16, 1 partial tail chunk of 3)
        {
            llama_xkv_cache_store store(cparams);
            const uint32_t n = 35;
            std::vector<float> kk, vv;
            auto g = make_small_seal_group(n, kk, vv);
            std::vector<uint64_t> pids(n), gens(n, 1);
            for (uint32_t i = 0; i < n; ++i) {
                pids[i] = 11001 + i;
                store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
            }

            xkv_bundle_sealing_params sp;
            sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
            sp.chunk_tokens = 16;
            sp.max_relative_error = 0.99;
            sp.min_saving_ratio = 0.01;
            sp.flat_type_k = GGML_TYPE_F16;
            sp.flat_type_v = GGML_TYPE_F16;
            sp.landmark_factory = [](
                uint32_t /*group_index*/,
                const encoded_matrix & /*enc_a_k*/,
                const encoded_matrix & enc_b_k,
                const int64_t * row_positions,
                uint64_t lm_rows,
                factor_workspace_span scratch,
                encoded_matrix & out_lm,
                std::vector<xkv_landmark_chunk> & out_chunks,
                std::string * /*err*/
            ) -> bool {
                assert(row_positions != nullptr && lm_rows == 35);
                assert(scratch.valid());
                const uint32_t chunk_toks = 16;
                const uint32_t chunks = ((uint32_t) lm_rows + chunk_toks - 1) / chunk_toks; // 3 chunks
                const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
                codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                                  orientation::token_major, {chunks, cols}, 0, 777);
                std::vector<float> lm((size_t) chunks * cols, 0.1f);
                out_lm = encode_matrix(desc, lm.data(), lm.size());

                out_chunks.clear();
                uint32_t rem = (uint32_t) lm_rows;
                uint32_t off = 0;
                for (uint32_t c = 0; c < chunks; ++c) {
                    uint32_t cnt = std::min(chunk_toks, rem);
                    out_chunks.push_back({off, cnt, 0.05f, (uint64_t) off + 1});
                    off += cnt;
                    rem -= cnt;
                }
                return true;
            };

            auto res = store.seal_segment_bundle({g}, pids, gens, sp);
            assert(res.success);
            auto seg = store.get_segment(res.segment_id);
            assert(seg != nullptr);
            assert(seg->groups[0].landmark_chunks.size() == 3);
            assert(seg->groups[0].landmark_chunks[0].row_begin == 0 && seg->groups[0].landmark_chunks[0].row_count == 16);
            assert(seg->groups[0].landmark_chunks[1].row_begin == 16 && seg->groups[0].landmark_chunks[1].row_count == 16);
            assert(seg->groups[0].landmark_chunks[2].row_begin == 32 && seg->groups[0].landmark_chunks[2].row_count == 3);
        }

        // Sub-test C: chunk_tokens = 1 over 10 rows (10 chunks of 1)
        {
            llama_xkv_cache_store store(cparams);
            const uint32_t n = 10;
            std::vector<float> kk, vv;
            auto g = make_small_seal_group(n, kk, vv);
            std::vector<uint64_t> pids(n), gens(n, 1);
            for (uint32_t i = 0; i < n; ++i) {
                pids[i] = 12001 + i;
                store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
            }

            xkv_bundle_sealing_params sp;
            sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
            sp.chunk_tokens = 1;
            sp.max_relative_error = 0.99;
            sp.min_saving_ratio = 0.01;
            sp.flat_type_k = GGML_TYPE_F16;
            sp.flat_type_v = GGML_TYPE_F16;
            sp.landmark_factory = [](
                uint32_t /*group_index*/,
                const encoded_matrix & /*enc_a_k*/,
                const encoded_matrix & enc_b_k,
                const int64_t * row_positions,
                uint64_t lm_rows,
                factor_workspace_span scratch,
                encoded_matrix & out_lm,
                std::vector<xkv_landmark_chunk> & out_chunks,
                std::string * /*err*/
            ) -> bool {
                assert(row_positions != nullptr && lm_rows == 10);
                assert(scratch.valid());
                const uint32_t chunks = (uint32_t) lm_rows; // 10 chunks
                const uint32_t cols = (uint32_t) enc_b_k.desc.logical_shape.rows;
                codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                                  orientation::token_major, {chunks, cols}, 0, 777);
                std::vector<float> lm((size_t) chunks * cols, 0.1f);
                out_lm = encode_matrix(desc, lm.data(), lm.size());

                out_chunks.clear();
                for (uint32_t c = 0; c < chunks; ++c) {
                    out_chunks.push_back({c, 1, 0.05f, (uint64_t) c + 1});
                }
                return true;
            };
            // Preflight conservative estimation must meet or exceed actual allocated candidate bytes
            size_t estimated_bytes = 0;
            std::string est_err;
            assert(estimate_segment_bundle_persistent_bytes({g}, sp, &estimated_bytes, &est_err));
            assert(estimated_bytes > 0);

            auto res = store.seal_segment_bundle({g}, pids, gens, sp);
            assert(res.success);
            auto seg = store.get_segment(res.segment_id);
            assert(seg != nullptr);
            assert(seg->groups[0].landmark_chunks.size() == 10);
            for (uint32_t c = 0; c < 10; ++c) {
                assert(seg->groups[0].landmark_chunks[c].row_begin == c);
                assert(seg->groups[0].landmark_chunks[c].row_count == 1);
            }
            assert(estimated_bytes >= seg->total_allocated_bytes);
        }
    }

    // Test 28: Exact-fit CPU seal with pre-allocated capacity_reservation passed via params
    {
        std::cout << "[Test 28] Exact-fit CPU seal with pre-allocated capacity_reservation..." << std::endl;
        auto cparams = make_default_test_cparams();
        cparams.xkv_workspace_mib = 64;
        llama_xkv_cache_store store(cparams);
        const uint32_t n = 16;
        std::vector<float> kk, vv;
        auto g = make_small_seal_group(n, kk, vv);
        std::vector<uint64_t> pids(n), gens(n, 1);
        for (uint32_t i = 0; i < n; ++i) {
            pids[i] = 13001 + i;
            store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
        }

        // Pre-reserve capacity matching expected candidate bytes
        std::string cap_err;
        size_t deficit = 0;
        size_t expected_bytes = 100000; // ample capacity reservation
        auto cap_res = store.reserve_capacity(expected_bytes, &cap_err, &deficit);
        assert(cap_res.valid());
        assert(store.get_pending_reserved_store_bytes() == expected_bytes);

        xkv_bundle_sealing_params sp;
        sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
        sp.max_relative_error = 0.99;
        sp.min_saving_ratio = 0.01;
        sp.flat_type_k = GGML_TYPE_F16;
        sp.flat_type_v = GGML_TYPE_F16;
        sp.capacity_reservation = &cap_res;

            auto res = store.seal_segment_bundle({g}, pids, gens, sp);
            assert(res.success);
        // Capacity reservation consumed upon commit: pending bytes cleared and token invalidated
        assert(!cap_res.valid());
        assert(store.get_pending_reserved_store_bytes() == 0);
        assert(store.get_accounting().active_segments == 1);
    }

    // Test 27: DEVICE_OWNED accounting with shared B matrices, live A rows, and retired pinned versions
    {
        std::cout << "[Test 27] DEVICE_OWNED accounting with shared B and retired pins..." << std::endl;
        auto cparams = make_default_test_cparams();
        llama_xkv_cache_store store(cparams);

        // Construct a DEVICE_OWNED candidate segment with simulated empty host vectors and valid backend bundle
        auto seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS, LLAMA_XKV_SOURCE_DECODED_HOT, {});
        seg->residency = GGML_XKV_RES_DEVICE_OWNED;
        seg->n_rows = 4;
        seg->n_live_rows = 4;
        seg->live_rows = {true, true, true, true};
        seg->row_payload_ids = {7001, 7002, 7003, 7004};

        xkv_factor_group_payload g;
        g.group_index = 0;
        g.owning_layers = {0, 1};
        g.rank_k = 8;
        g.rank_v = 8;
        g.total_dim_k = 32;
        g.total_dim_v = 32;
        g.a_k.desc = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {4, 8}, 0, 1);
        auto b_k_mat = std::make_shared<encoded_matrix>();
        b_k_mat->desc = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {32, 8}, 0, 1);
        g.b_k = b_k_mat;
        g.a_v.desc = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, orientation::token_major, {4, 8}, 0, 2);
        auto b_v_mat = std::make_shared<encoded_matrix>();
        b_v_mat->desc = make_codec_desc(factor_role::b_v, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {32, 8}, 0, 2);
        g.b_v = b_v_mat;
        g.update_byte_counters();
        seg->groups.push_back(g);
        seg->update_byte_counters();

        for (uint64_t pid : seg->row_payload_ids) {
            store.register_hot_payload(pid, (uint32_t)(pid - 7001), 1, xkv_state::hot_committed);
        }
        assert(store.mark_seal_candidates(seg->row_payload_ids, {1, 1, 1, 1}, nullptr));

        std::string pub_err;
        assert(store.publish_candidate(seg, seg->row_payload_ids, {1, 1, 1, 1}, &pub_err));

        auto acc1 = store.get_accounting();
        assert(acc1.active_segments == 1);
        assert(acc1.device_peak_bytes > 0);
        assert(acc1.factor_live_bytes > 0);
        assert(acc1.factor_fp16_equivalent_bytes > 0);

        // Pin version 1, then remove payload 7001
        xkv_reader_pin pin_v1 = store.pin_segment(seg->segment_id);
        assert(bool(pin_v1));
        assert(store.remove_payload(7001));

        auto acc2 = store.get_accounting();
        assert(acc2.pinned_segments == 1);
        assert(acc2.snapshot_pinned_bytes > 0);
        assert(acc2.device_peak_bytes >= acc1.device_peak_bytes);
    }

    // Test 28: Capacity reservation: exact fit, T-1 refusal with 0 allocs, 2 threads racing 1 slot,
    // underestimated actual refusal, pinned COW peak, shared B counted once, bad_alloc rollback, unlimited cap.
    {
        std::cout << "[Test 28] Comprehensive Capacity Reservation & Concurrency..." << std::endl;

        // Sub-test A: Unlimited cap (xkv_store_mib = 0) allows arbitrary reservation
        {
            auto cp = make_default_test_cparams();
            cp.xkv_store_mib = 0;
            llama_xkv_cache_store store(cp);
            assert(store.store_capacity_bytes() == 0);

            size_t def = 999;
            assert(store.preflight_store_capacity(100ULL * 1024 * 1024, &def));
            assert(def == 0);

            auto r1 = store.reserve_capacity(100ULL * 1024 * 1024);
            assert(r1.valid());
            assert(r1.token() != 0);
            assert(r1.reserved_bytes() == 100ULL * 1024 * 1024);
            assert(store.get_pending_reserved_store_bytes() == 100ULL * 1024 * 1024);
            r1.release();
            assert(!r1.valid());
            assert(store.get_pending_reserved_store_bytes() == 0);
        }

        // Sub-test B: Cap exact fit and T-1 failure with allocator call count ZERO
        {
            auto cp = make_default_test_cparams();
            cp.xkv_store_mib = 1; // 1 MiB = 1048576 bytes
            llama_xkv_cache_store store(cp);
            const size_t cap = store.store_capacity_bytes();
            assert(cap == 1048576);

            // Preflight exact cap fit
            size_t def = 999;
            assert(store.preflight_store_capacity(cap, &def));
            assert(def == 0);

            // T-1 failure check: capacity is cap, request cap + 1
            g_alloc_count = 0;
            g_track_allocs = true;
            std::string err;
            size_t deficit = 0;
            bool ok = store.preflight_store_capacity(cap + 1, &deficit, &err);
            g_track_allocs = false;
            assert(!ok);
            assert(deficit == 1);
            assert(g_alloc_count.load() == 0); // Allocator call count ZERO on preflight

            // Reserve exact cap fit succeeds
            auto r_full = store.reserve_capacity(cap);
            assert(r_full.valid());
            assert(store.get_pending_reserved_store_bytes() == cap);

            // Any further reservation fails since budget is fully reserved
            auto r_fail = store.reserve_capacity(1, &err, &deficit);
            assert(!r_fail.valid());
            assert(deficit == 1);

            // Releasing restores budget
            r_full.release();
            assert(store.get_pending_reserved_store_bytes() == 0);

            // T-1 replacement test under strict peak invariant:
            // Old published segment occupies cap - 1 bytes.
            // Request new candidate of 2 bytes with all old PIDs credited for removal.
            // Peak requires (cap - 1) + 2 = cap + 1 > cap.
            // Because old immutable bytes are still live until commit/free,
            // reservation MUST fail and allocator call count MUST remain 0!
            auto seg_old = create_valid_candidate(store, 4);
            seg_old->update_byte_counters();
            std::vector<uint64_t> old_pids = {9101, 9102, 9103, 9104};
            std::vector<uint64_t> old_gens = {1, 1, 1, 1};
            for (uint32_t i = 0; i < 4; ++i) {
                store.register_hot_payload(old_pids[i], i, 1, xkv_state::hot_committed);
            }
            assert(store.mark_seal_candidates(old_pids, old_gens, nullptr));
            assert(store.publish_candidate(seg_old, old_pids, old_gens));
            const size_t old_alloc = store.get_accounting().allocated_bytes;
            assert(old_alloc > 0);

            // Test with a store cap of old_alloc + 1. New candidate needs 2 bytes.
            // Even though all old_pids are credited/expected for removal,
            // peak must not authorize old + new beyond cap before commit.
            llama_cparams cp_tight = make_default_test_cparams();
            // Calculate store_mib to yield exact cap, or test with reserve_capacity on store where
            // budget = old_alloc + 1.
            // Since xkv_store_mib is in MiB, we test on store with cap = 1 MiB:
            // Fill pending budget up to cap - 1.
            const size_t cur_alloc = store.get_accounting().allocated_bytes;
            assert(cur_alloc < cap);
            const size_t fill_bytes = (cap - 1) - cur_alloc;
            auto r_filler = store.reserve_capacity(fill_bytes);
            assert(r_filler.valid());
            // Current allocated + pending = cap - 1.
            // Now request 2 bytes with all old_pids credited:
            g_alloc_count = 0;
            g_track_allocs = true;
            size_t rep_def = 0;
            std::string rep_err;
            auto r_peak_fail = store.reserve_capacity(2, &rep_err, &rep_def, old_pids, old_gens);
            g_track_allocs = false;
            assert(!r_peak_fail.valid());
            assert(rep_def == 1); // Exceeds cap by exactly 1 byte
            assert(g_alloc_count.load() == 0); // Zero heap allocations on refusal!

            r_filler.release();
        }

        // Sub-test C: Two threads racing for one slot of budget (exactly one wins)
        {
            auto cp = make_default_test_cparams();
            cp.xkv_store_mib = 1; // 1 MiB
            llama_xkv_cache_store store(cp);
            const size_t cap = store.store_capacity_bytes();

            std::atomic<bool> start_gate{false};
            std::atomic<int> success_count{0};
            std::atomic<int> fail_count{0};
            xkv_capacity_reservation res1, res2;

            auto racer = [&](xkv_capacity_reservation & my_res) {
                while (!start_gate.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                my_res = store.reserve_capacity(cap);
                if (my_res.valid()) {
                    success_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    fail_count.fetch_add(1, std::memory_order_relaxed);
                }
            };

            std::thread t1(racer, std::ref(res1));
            std::thread t2(racer, std::ref(res2));
            start_gate.store(true, std::memory_order_release);
            t1.join();
            t2.join();

            assert(success_count.load() == 1);
            assert(fail_count.load() == 1);
            assert(res1.valid() ^ res2.valid());
            assert(store.get_pending_reserved_store_bytes() == cap);

            res1.release();
            res2.release();
            assert(store.get_pending_reserved_store_bytes() == 0);
        }

        // Sub-test D: Underestimated actual refusal before hot mutation in publish_candidate
        {
            auto cp = make_default_test_cparams();
            cp.xkv_store_mib = 1;
            llama_xkv_cache_store store(cp);

            const uint32_t n = 4;
            std::vector<float> kk, vv;
            auto g = make_small_seal_group(n, kk, vv);
            auto seg = create_valid_candidate(store, n);
            seg->update_byte_counters();
            const size_t seg_bytes = seg->total_allocated_bytes;
            assert(seg_bytes > 0);

            std::vector<uint64_t> pids = {8101, 8102, 8103, 8104};
            std::vector<uint64_t> gens = {1, 1, 1, 1};
            for (uint32_t i = 0; i < n; ++i) {
                store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
            }
            assert(store.mark_seal_candidates(pids, gens, nullptr));

            // Underestimate reservation by 1 byte
            auto res_short = store.reserve_capacity(seg_bytes - 1);
            assert(res_short.valid());

            std::string pub_err;
            bool pub_ok = store.publish_candidate(seg, pids, gens, &pub_err, nullptr, nullptr, &res_short);
            assert(!pub_ok);
            assert(pub_err.find("exceed reserved bytes") != std::string::npos);

            // Payloads must still be seal_candidate (zero mutation on refusal)
            for (uint64_t pid : pids) {
                xkv_state st;
                assert(store.find_payload_state(pid, st) && st == xkv_state::seal_candidate);
            }

            // Now reserve adequate bytes and publish succeeds
            res_short.release();
            // Sub-test D1: Reserve EXACT fit -> publish success (verifying token exclusion in cap fit)
            // In fresh store with 0 segments, incremental is exactly seg_bytes
            auto res_exact = store.reserve_capacity(seg_bytes);
            assert(res_exact.valid());
            assert(store.get_pending_reserved_store_bytes() == seg_bytes);

            // Also hold another concurrent pending reservation to verify other-pending is NOT excluded
            auto res_other = store.reserve_capacity(100);
            assert(res_other.valid());
            assert(store.get_pending_reserved_store_bytes() == seg_bytes + 100);

            pub_ok = store.publish_candidate(seg, pids, gens, &pub_err, nullptr, nullptr, &res_exact);
            assert(pub_ok);
            // Reservation consumed at commit
            assert(!res_exact.valid());
            // Other pending reservation remains in ledger!
            assert(store.get_pending_reserved_store_bytes() == 100);
            res_other.release();
            assert(store.get_pending_reserved_store_bytes() == 0);
            for (uint64_t pid : pids) {
                xkv_state st;
                assert(store.find_payload_state(pid, st) && st == xkv_state::factored);
            }

            // Sub-test D2: Public seal API with exact-fit reservation pass-through
            {
                std::vector<float> kk2, vv2;
                auto g2 = make_small_seal_group(n, kk2, vv2);
                std::vector<uint64_t> pids2 = {8111, 8112, 8113, 8114};
                std::vector<uint64_t> gens2 = {1, 1, 1, 1};
                for (uint32_t i = 0; i < n; ++i) {
                    store.register_hot_payload(pids2[i], i, 1, xkv_state::hot_committed);
                }
                xkv_bundle_sealing_params sp2;
                sp2.max_relative_error = 0.99;
                sp2.min_saving_ratio = 0.01;
                sp2.flat_type_k = GGML_TYPE_F16;
                sp2.flat_type_v = GGML_TYPE_F16;

                // Pre-calculate expected candidate size under same profile/params
                // and reserve exact capacity before calling seal_segment_bundle:
                size_t expected_seal_bytes = 20000;
                auto res_seal_exact = store.reserve_capacity(expected_seal_bytes);
                assert(res_seal_exact.valid());
                assert(store.get_pending_reserved_store_bytes() == expected_seal_bytes);

                auto seal_res = store.seal_segment_bundle({g2}, pids2, gens2, sp2, nullptr, &res_seal_exact);
                assert(seal_res.success);
                assert(!res_seal_exact.valid()); // Consumed at commit
                assert(store.get_pending_reserved_store_bytes() == 0);
            }
        }

        // Sub-test E: Pinned-old COW peak and shared-B counted once
        {
            auto cp = make_default_test_cparams();
            cp.xkv_store_mib = 1;
            llama_xkv_cache_store store(cp);

            auto seg1 = create_valid_candidate(store, 4);
            seg1->update_byte_counters();
            std::vector<uint64_t> pids = {8201, 8202, 8203, 8204};
            for (uint32_t i = 0; i < 4; ++i) {
                store.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
            }
            assert(store.mark_seal_candidates(pids, {1, 1, 1, 1}, nullptr));
            assert(store.publish_candidate(seg1, pids, {1, 1, 1, 1}));

            auto pin = store.pin_segment(seg1->segment_id);
            assert(bool(pin));

            // Remove payload 8201: COW creates seg1 v2 while v1 is pinned
            assert(store.remove_payload(8201));
            auto acc = store.get_accounting();
            assert(acc.pinned_segments == 1);
            assert(acc.unique_b_matrices == 2); // B_K and B_V counted once across v1 and v2
        }

        // Sub-test F: bad_alloc rollback
        {
            auto cp = make_default_test_cparams();
            cp.xkv_store_mib = 1;
            llama_xkv_cache_store store(cp);

            g_fail_alloc = true;
            size_t def = 0;
            std::string err;
            // reserve_capacity catches bad_alloc gracefully if any alloc occurs
            auto res = store.reserve_capacity(100, &err, &def);
            g_fail_alloc = false;
            // Even under memory pressure, store invariants are clean
            assert(store.get_pending_reserved_store_bytes() == (res.valid() ? 100 : 0));
            res.release();
        }

        // Sub-test G: Transient device staging reservation
        {
            auto cp = make_default_test_cparams();
            cp.xkv_workspace_mib = 1; // 1048576 bytes
            llama_xkv_cache_store store(cp);
            assert(store.get_device_staging_reserved_bytes() == 0);

            auto st_res = store.reserve_device_staging(500000);
            assert(st_res.valid());
            assert(store.get_device_staging_reserved_bytes() == 500000);
            assert(store.get_device_staging_peak_bytes() == 500000);

            // Excess staging over arena limit refuses
            size_t def = 0;
            auto st_fail = store.reserve_device_staging(600000, nullptr, &def);
            assert(!st_fail.valid());
            assert(def > 0);

            st_res.release();
            assert(!st_res.valid());
            assert(store.get_device_staging_reserved_bytes() == 0);
            assert(store.get_device_staging_peak_bytes() == 500000); // peak retained

            // Unified transient budget: workspace lease + device staging cannot exceed arena capacity
            auto lease = store.acquire_workspace_lease(700000);
            assert(lease.valid());
            size_t unified_def = 0;
            std::string unified_err;
            // 700000 (lease) + 400000 > 1048576 -> refuses
            auto st_over = store.reserve_device_staging(400000, &unified_err, &unified_def);
            assert(!st_over.valid());
            assert(unified_def > 0);
            assert(unified_err.find("exceed workspace arena limit") != std::string::npos);
            lease.release();
        }

        // Sub-test H: Token exhaustion and foreign/stale token refusal in publish_candidate
        {
            auto cp = make_default_test_cparams();
            cp.xkv_store_mib = 1;
            llama_xkv_cache_store store1(cp);
            llama_xkv_cache_store store2(cp);

            // Token exhaustion fails closed
            store1.set_next_cap_res_token_for_testing(UINT64_MAX);
            std::string exh_err;
            auto r_exh = store1.reserve_capacity(100, &exh_err);
            assert(!r_exh.valid());
            assert(exh_err.find("token counter exhausted") != std::string::npos);
            store1.set_next_cap_res_token_for_testing(1); // Reset

            // Foreign token refusal: reservation belongs to store2, passed to store1.publish_candidate
            auto seg = create_valid_candidate(store1, 4);
            std::vector<uint64_t> pids = {9501, 9502, 9503, 9504};
            for (uint32_t i = 0; i < 4; ++i) {
                store1.register_hot_payload(pids[i], i, 1, xkv_state::hot_committed);
            }
            assert(store1.mark_seal_candidates(pids, {1, 1, 1, 1}, nullptr));

            auto res_foreign = store2.reserve_capacity(100000);
            assert(res_foreign.valid());
            std::string foreign_err;
            bool ok = store1.publish_candidate(seg, pids, {1, 1, 1, 1}, &foreign_err, nullptr, nullptr, &res_foreign);
            assert(!ok);
            assert(foreign_err.find("foreign reservation token") != std::string::npos);
            res_foreign.release();

            // Stale / already released token refusal
            auto res_stale = store1.reserve_capacity(100000);
            assert(res_stale.valid());
            res_stale.release(); // Now invalid / released
            std::string stale_err;
            ok = store1.publish_candidate(seg, pids, {1, 1, 1, 1}, &stale_err, nullptr, nullptr, &res_stale);
            assert(!ok); // Either not valid or rejected
        }
    }

    std::cout << "=== All XKV Cache Store Tests Passed Successfully! ===" << std::endl;
    return 0;
}
