// Deterministic unit and regression tests for llama_xkv_hot:
// 1. Atomic reserve failure (insufficient capacity, capacity 0, overflow-safe bounds, clean no-op).
// 2. RAII move-only reservation rollback (scope exit, explicit rollback, move assignment/construction).
// 3. Successful commit binding exact nonzero payload IDs and generations.
// 4. Stale release rejection without partial mutation (all-or-nothing batch release, wrong payload ID,
//    wrong generation, wrong expected slot, duplicate payload ID).
// 5. Deterministic slot reuse (freed slots are reused in deterministic order).
// 6. Generation ABA protection (reused slot with new generation rejects stale release attempt).
// 7. Full accounting bounds verification (free, reserved, bound, live, peak, reset_peak).
// 8. Multi-threaded concurrency: two-thread no-duplicate allocation and safe concurrent reservation/commit/release.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-hot.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <unordered_set>

using namespace llama_xkv;

// ----------------------------------------------------------------------------
// Test 1: Capacity 0 and atomic reserve failure
// ----------------------------------------------------------------------------
static void test_capacity_zero_and_atomic_failure() {
    std::cout << "[Test 1] Capacity 0 and atomic reserve failure..." << std::endl;

    // Capacity 0 pool
    xkv_hot_slot_pool zero_pool(0);
    assert(zero_pool.get_capacity() == 0);
    assert(zero_pool.get_free() == 0);
    assert(zero_pool.get_reserved() == 0);
    assert(zero_pool.get_bound() == 0);
    assert(zero_pool.get_live() == 0);
    assert(zero_pool.get_peak() == 0);

    std::string err;
    xkv_hot_reservation zero_res = zero_pool.reserve(1, &err);
    assert(!zero_res.valid());
    assert(zero_res.empty());
    assert(!err.empty());

    // Zero count reservation on capacity 0 pool is a valid empty reservation
    xkv_hot_reservation noop_res = zero_pool.reserve(0, &err);
    assert(!noop_res.valid()); // empty reservation has valid() == false because slots are empty
    assert(noop_res.empty());

    // Pool with capacity 8
    xkv_hot_slot_pool pool(8);
    assert(pool.get_capacity() == 8);
    assert(pool.get_free() == 8);

    // Requesting 9 slots should fail atomically
    err.clear();
    xkv_hot_reservation fail_res = pool.reserve(9, &err);
    assert(!fail_res.valid());
    assert(fail_res.empty());
    assert(!err.empty());
    assert(pool.get_free() == 8);
    assert(pool.get_reserved() == 0);
    assert(pool.get_live() == 0);

    // Reserve 5 slots succeeds
    xkv_hot_reservation res1 = pool.reserve(5);
    assert(res1.valid());
    assert(res1.size() == 5);
    assert(pool.get_free() == 3);
    assert(pool.get_reserved() == 5);
    assert(pool.get_live() == 5);
    assert(pool.get_peak() == 5);

    // Requesting 4 slots should fail atomically (only 3 free)
    err.clear();
    xkv_hot_reservation fail_res2 = pool.reserve(4, &err);
    assert(!fail_res2.valid());
    assert(!err.empty());
    // Invariant: Failed reservation must not mutate any counters or state!
    assert(pool.get_free() == 3);
    assert(pool.get_reserved() == 5);
    assert(pool.get_live() == 5);
    assert(pool.get_peak() == 5);

    // Requesting 3 slots succeeds
    xkv_hot_reservation res2 = pool.reserve(3);
    assert(res2.valid());
    assert(res2.size() == 3);
    assert(pool.get_free() == 0);
    assert(pool.get_reserved() == 8);
    assert(pool.get_live() == 8);
    assert(pool.get_peak() == 8);

    // Now 0 free; requesting 1 slot fails
    err.clear();
    xkv_hot_reservation fail_res3 = pool.reserve(1, &err);
    assert(!fail_res3.valid());
    assert(!err.empty());
}

// ----------------------------------------------------------------------------
// Test 2: Move-only RAII reservation and rollback
// ----------------------------------------------------------------------------
static void test_raii_and_rollback() {
    std::cout << "[Test 2] RAII reservation and rollback..." << std::endl;

    xkv_hot_slot_pool pool(10);

    // Sub-scope destruction rolls back reserved slots
    {
        xkv_hot_reservation res = pool.reserve(4);
        assert(res.valid());
        assert(pool.get_reserved() == 4);
        assert(pool.get_free() == 6);
        assert(pool.get_peak() == 4);
        // Exiting scope without commit
    }

    assert(pool.get_reserved() == 0);
    assert(pool.get_bound() == 0);
    assert(pool.get_free() == 10);
    assert(pool.get_live() == 0);
    assert(pool.get_peak() == 4); // Peak retained

    // Explicit rollback
    xkv_hot_reservation res = pool.reserve(6);
    assert(res.valid());
    assert(pool.get_reserved() == 6);
    res.rollback();
    assert(!res.valid());
    assert(res.empty());
    assert(pool.get_reserved() == 0);
    assert(pool.get_free() == 10);

    // Calling rollback again is safe no-op
    res.rollback();
    assert(!res.valid());

    // Move construction transfers reservation ownership
    xkv_hot_reservation r1 = pool.reserve(3);
    const std::vector<uint32_t> r1_slots = r1.slots();
    xkv_hot_reservation r2(std::move(r1));
    assert(!r1.valid());
    assert(r1.empty());
    assert(r2.valid());
    assert(r2.slots() == r1_slots);
    assert(pool.get_reserved() == 3);

    // Move assignment transfers ownership and rolls back previous reservation in target
    xkv_hot_reservation r3 = pool.reserve(2);
    assert(pool.get_reserved() == 5);
    r3 = std::move(r2); // r3 previously held 2 slots, which should be rolled back!
    assert(!r2.valid());
    assert(r3.valid());
    assert(r3.slots() == r1_slots);
    assert(pool.get_reserved() == 3); // 5 - 2 = 3
    assert(pool.get_free() == 7);
}

// ----------------------------------------------------------------------------
// Test 3: Commit validation, double-commit rejection, and lookups
// ----------------------------------------------------------------------------
static void test_commit_and_lookups() {
    std::cout << "[Test 3] Commit validation, duplicate rejection, and lookups..." << std::endl;

    xkv_hot_slot_pool pool(10);
    xkv_hot_reservation res = pool.reserve(3);
    assert(res.valid());

    const std::vector<uint32_t> slots = res.slots();
    assert(slots.size() == 3);

    // Zero payload ID rejection
    std::string err;
    std::vector<uint64_t> pids_zero = {1001, 0, 1003};
    std::vector<uint64_t> gens = {1, 1, 1};
    assert(!res.commit(pids_zero, gens, &err));
    assert(err.find("is 0") != std::string::npos);
    assert(!res.is_committed());
    assert(pool.get_reserved() == 3);
    assert(pool.get_bound() == 0);

    // Duplicate payload ID within batch rejection
    std::vector<uint64_t> pids_dup = {1001, 1002, 1001};
    err.clear();
    assert(!res.commit(pids_dup, gens, &err));
    assert(err.find("duplicate") != std::string::npos);
    assert(!res.is_committed());
    assert(pool.get_reserved() == 3);
    assert(pool.get_bound() == 0);

    // Zero generation rejection (0 is the free/reserved sentinel; the store
    // rejects generation 0 on register/remove, so the pool must match)
    std::vector<uint64_t> pids_ok = {1001, 1002, 1003};
    std::vector<uint64_t> gens_zero = {1, 0, 1};
    err.clear();
    assert(!res.commit(pids_ok, gens_zero, &err));
    assert(err.find("is 0") != std::string::npos);
    assert(!res.is_committed());
    assert(pool.get_reserved() == 3);
    assert(pool.get_bound() == 0);

    // Size mismatch rejection
    std::vector<uint64_t> pids_short = {1001, 1002};
    std::vector<uint64_t> gens_short = {1, 1};
    err.clear();
    assert(!res.commit(pids_short, gens_short, &err));
    assert(err.find("does not match") != std::string::npos);

    // Successful commit
    std::vector<uint64_t> valid_pids = {1001, 1002, 1003};
    std::vector<uint64_t> valid_gens = {10, 20, 30};
    assert(res.commit(valid_pids, valid_gens, &err));
    assert(res.is_committed());
    assert(pool.get_reserved() == 0);
    assert(pool.get_bound() == 3);
    assert(pool.get_live() == 3);

    // Double-commit rejection
    err.clear();
    assert(!res.commit(valid_pids, valid_gens, &err));
    assert(!err.empty());

    // Destroying committed reservation must NOT rollback bound slots!
    res.rollback();
    assert(pool.get_bound() == 3);
    assert(pool.get_free() == 7);

    // Check lookups
    for (size_t i = 0; i < valid_pids.size(); ++i) {
        uint32_t found_slot = 999;
        assert(pool.find_slot(valid_pids[i], found_slot));
        assert(found_slot == slots[i]);
        assert(pool.has_payload(valid_pids[i]));

        xkv_hot_slot_info info;
        assert(pool.find_payload(valid_pids[i], info));
        assert(info.slot == slots[i]);
        assert(info.state == xkv_slot_state::bound);
        assert(info.payload_id == valid_pids[i]);
        assert(info.storage_generation == valid_gens[i]);

        xkv_hot_slot_info slot_info;
        assert(pool.get_slot_info(slots[i], slot_info));
        assert(slot_info.slot == slots[i]);
        assert(slot_info.state == xkv_slot_state::bound);
        assert(slot_info.payload_id == valid_pids[i]);
        assert(slot_info.storage_generation == valid_gens[i]);
    }

    // Lookup non-existent payload
    uint32_t nonexistent_slot = 0;
    assert(!pool.find_slot(99999, nonexistent_slot));
    assert(!pool.has_payload(99999));
    assert(!pool.find_slot(0, nonexistent_slot));
    assert(!pool.has_payload(0));

    // Attempting to commit an already bound payload ID in a new reservation must fail
    xkv_hot_reservation res2 = pool.reserve(1);
    std::vector<uint64_t> conflict_pid = {1002};
    std::vector<uint64_t> conflict_gen = {1};
    err.clear();
    assert(!res2.commit(conflict_pid, conflict_gen, &err));
    assert(err.find("already bound") != std::string::npos);
    assert(pool.get_reserved() == 1);
    assert(pool.get_bound() == 3);
}

// ----------------------------------------------------------------------------
// Test 4: Stale release rejection without partial mutation
// ----------------------------------------------------------------------------
static void test_stale_release_rejection() {
    std::cout << "[Test 4] Stale release rejection without partial mutation..." << std::endl;

    xkv_hot_slot_pool pool(10);
    xkv_hot_reservation res = pool.reserve(4);

    const std::vector<uint32_t> slots = res.slots();
    const std::vector<uint64_t> pids = {201, 202, 203, 204};
    const std::vector<uint64_t> gens = {1, 2, 3, 4};
    assert(res.commit(pids, gens));

    assert(pool.get_bound() == 4);
    assert(pool.get_free() == 6);

    // 1. Rejection: wrong generation
    std::string err;
    assert(!pool.release(202, 999 /* wrong gen */, slots[1], &err));
    assert(err.find("generation mismatch") != std::string::npos);
    // Invariant: Slot 202 must still be intact and bound
    assert(pool.has_payload(202));
    assert(pool.get_bound() == 4);

    // 2. Rejection: wrong expected_slot
    err.clear();
    assert(!pool.release(202, 2, slots[0] /* wrong slot */, &err));
    assert(err.find("payload_id mismatch") != std::string::npos);
    assert(pool.has_payload(202));
    assert(pool.get_bound() == 4);

    // 3. Rejection: slot out of range
    err.clear();
    assert(!pool.release(202, 2, 9999, &err));
    assert(err.find("out of range") != std::string::npos);
    assert(pool.get_bound() == 4);

    // 4. Rejection: wrong payload_id for slot
    err.clear();
    assert(!pool.release(999, 2, slots[1], &err));
    assert(err.find("payload_id mismatch") != std::string::npos);
    assert(pool.get_bound() == 4);

    // 5. Batch release all-or-nothing:
    // If one item has wrong generation, ZERO slots in the batch are released!
    std::vector<xkv_hot_release_item> batch_items = {
        {201, gens[0], slots[0]},        // OK
        {202, gens[1], slots[1]},        // OK
        {203, 9999 /* stale */, slots[2]},// FAILS
        {204, gens[3], slots[3]},        // OK
    };

    err.clear();
    assert(!pool.release_batch(batch_items, &err));
    assert(err.find("generation mismatch") != std::string::npos);

    // CRITICAL: All-or-nothing! None of 201, 202, 203, 204 should have been freed!
    assert(pool.get_bound() == 4);
    assert(pool.get_free() == 6);
    for (size_t i = 0; i < pids.size(); ++i) {
        assert(pool.has_payload(pids[i]));
        uint32_t s = 0;
        assert(pool.find_slot(pids[i], s));
        assert(s == slots[i]);
    }

    // 6. Valid batch release of first two items succeeds
    std::vector<xkv_hot_release_item> valid_batch = {
        {201, gens[0], slots[0]},
        {202, gens[1], slots[1]},
    };
    assert(pool.release_batch(valid_batch, &err));
    assert(pool.get_bound() == 2);
    assert(pool.get_free() == 8);
    assert(!pool.has_payload(201));
    assert(!pool.has_payload(202));
    assert(pool.has_payload(203));
    assert(pool.has_payload(204));

    // 7. Double-release rejection: attempting to release 201 again fails
    err.clear();
    assert(!pool.release(201, gens[0], slots[0], &err));
    assert(err.find("not bound") != std::string::npos);
}

// ----------------------------------------------------------------------------
// Test 5: Slot reuse and ABA generation protection
// ----------------------------------------------------------------------------
static void test_slot_reuse_and_aba_protection() {
    std::cout << "[Test 5] Slot reuse and generation ABA protection..." << std::endl;

    xkv_hot_slot_pool pool(4);
    xkv_hot_reservation r1 = pool.reserve(2);
    assert(r1.valid());
    const uint32_t s0 = r1[0];
    const uint32_t s1 = r1[1];
    assert(s0 == 0 && s1 == 1); // Min-heap allocates lowest indices first

    // Commit payload 10 on slot 0 with generation 1
    // Commit payload 20 on slot 1 with generation 1
    assert(r1.commit({10, 20}, {1, 1}));

    // Release slot 0 (payload 10, gen 1)
    assert(pool.release(10, 1, s0));
    assert(!pool.has_payload(10));
    assert(pool.get_free() == 3);

    // Reserve 1 slot -> must reuse slot 0 (lowest available)
    xkv_hot_reservation r2 = pool.reserve(1);
    assert(r2.valid());
    assert(r2[0] == s0);

    // Re-bind slot 0 to payload 30 with new generation 2 (simulating new cell/generation)
    assert(r2.commit({30}, {2}));
    assert(pool.has_payload(30));

    // Stale consumer tries to release slot 0 with old payload 10 and generation 1
    std::string err;
    assert(!pool.release(10, 1, s0, &err));
    assert(err.find("payload_id mismatch") != std::string::npos);
    assert(pool.has_payload(30));

    // Stale consumer tries to release slot 0 with current payload 30 but OLD generation 1
    err.clear();
    assert(!pool.release(30, 1 /* stale gen */, s0, &err));
    assert(err.find("generation mismatch") != std::string::npos);
    assert(pool.has_payload(30));

    // Release with correct new generation 2 succeeds
    assert(pool.release(30, 2, s0));
    assert(!pool.has_payload(30));
    assert(pool.get_free() == 3);
}

// ----------------------------------------------------------------------------
// Test 6: Accounting bounds and peak tracking
// ----------------------------------------------------------------------------
static void test_accounting_and_peak() {
    std::cout << "[Test 6] Accounting bounds and peak tracking..." << std::endl;

    xkv_hot_slot_pool pool(16);

    auto acc = pool.get_accounting();
    assert(acc.capacity == 16);
    assert(acc.free == 16);
    assert(acc.reserved == 0);
    assert(acc.bound == 0);
    assert(acc.live == 0);
    assert(acc.peak == 0);

    // Reserve 6 slots
    xkv_hot_reservation r1 = pool.reserve(6);
    acc = pool.get_accounting();
    assert(acc.free == 10);
    assert(acc.reserved == 6);
    assert(acc.bound == 0);
    assert(acc.live == 6);
    assert(acc.peak == 6);

    // Commit 6 slots
    std::vector<uint64_t> pids = {1, 2, 3, 4, 5, 6};
    std::vector<uint64_t> gens = {1, 1, 1, 1, 1, 1};
    assert(r1.commit(pids, gens));

    acc = pool.get_accounting();
    assert(acc.free == 10);
    assert(acc.reserved == 0);
    assert(acc.bound == 6);
    assert(acc.live == 6);
    assert(acc.peak == 6);

    // Reserve 4 more slots -> live reaches 10
    xkv_hot_reservation r2 = pool.reserve(4);
    acc = pool.get_accounting();
    assert(acc.free == 6);
    assert(acc.reserved == 4);
    assert(acc.bound == 6);
    assert(acc.live == 10);
    assert(acc.peak == 10);

    // Rollback r2 -> live drops to 6, peak remains 10
    r2.rollback();
    acc = pool.get_accounting();
    assert(acc.free == 10);
    assert(acc.reserved == 0);
    assert(acc.bound == 6);
    assert(acc.live == 6);
    assert(acc.peak == 10);

    // Reset peak -> resets peak to current live (6)
    pool.reset_peak();
    acc = pool.get_accounting();
    assert(acc.peak == 6);
}

// ----------------------------------------------------------------------------
// Test 7: Concurrency: Two threads reserving, committing, and releasing
// Guarantee: No duplicate slot allocation, no data race, strict accounting bounds.
// ----------------------------------------------------------------------------
static void test_concurrency_no_duplicate_allocation() {
    std::cout << "[Test 7] Concurrency: two-thread no-duplicate allocation..." << std::endl;

    const uint32_t capacity = 32;
    xkv_hot_slot_pool pool(capacity);

    const int iterations = 1000;
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};

    auto thread_worker = [&](uint64_t pid_base) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        for (int i = 0; i < iterations && !failed.load(std::memory_order_relaxed); ++i) {
            const uint32_t count = (i % 4) + 1; // 1 to 4 slots
            xkv_hot_reservation res = pool.reserve(count);
            if (!res.valid()) {
                // Contention or full pool is expected, retry
                std::this_thread::yield();
                continue;
            }

            // Verify slots allocated to this reservation are all distinct
            std::unordered_set<uint32_t> seen_slots;
            for (uint32_t s : res.slots()) {
                if (s >= capacity || !seen_slots.insert(s).second) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }

            std::vector<uint64_t> pids;
            std::vector<uint64_t> gens;
            for (uint32_t k = 0; k < count; ++k) {
                pids.push_back(pid_base + (uint64_t) i * 10 + k + 1);
                gens.push_back(1);
            }

            // Half the time commit then release; half the time let RAII rollback
            if (i % 2 == 0) {
                if (!res.commit(pids, gens)) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }

                // Verify bound slots in pool
                for (size_t k = 0; k < count; ++k) {
                    uint32_t s = 0;
                    if (!pool.find_slot(pids[k], s) || s != res[k]) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }

                // Release bound slots
                std::vector<xkv_hot_release_item> items;
                for (size_t k = 0; k < count; ++k) {
                    items.push_back({pids[k], gens[k], res[k]});
                }
                if (!pool.release_batch(items)) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            } else {
                // RAII rollback
                res.rollback();
            }
        }
    };

    // Thread 1 uses payload IDs starting at 10,000,000
    // Thread 2 uses payload IDs starting at 20,000,000
    std::thread t1(thread_worker, 10000000ULL);
    std::thread t2(thread_worker, 20000000ULL);

    start.store(true, std::memory_order_release);

    t1.join();
    t2.join();

    assert(!failed.load());

    // After both threads finish, all reservations/bindings must have been cleaned up!
    auto acc = pool.get_accounting();
    assert(acc.reserved == 0);
    assert(acc.bound == 0);
    assert(acc.live == 0);
    assert(acc.free == capacity);
    assert(acc.peak <= capacity);
}

// ----------------------------------------------------------------------------
// Test 8: Pool reset preserving object identity and shared-view test
// ----------------------------------------------------------------------------
static void test_reset_preserving_identity_and_shared_view() {
    std::cout << "[Test 8] Reset preserving object identity and shared view..." << std::endl;

    auto pool_ptr = std::make_shared<xkv_hot_slot_pool>(8);
    const void * original_address = pool_ptr.get();

    // Create multiple shared holders
    auto holder_view1 = pool_ptr;
    auto holder_view2 = pool_ptr;
    assert(holder_view1.get() == original_address);
    assert(holder_view2.get() == original_address);

    // Reserve and commit 4 slots
    xkv_hot_reservation res = pool_ptr->reserve(4);
    assert(res.valid());
    std::vector<uint64_t> pids = {101, 102, 103, 104};
    std::vector<uint64_t> gens = {1, 1, 1, 1};
    assert(res.commit(pids, gens));

    assert(holder_view1->get_bound() == 4);
    assert(holder_view2->get_free() == 4);
    assert(holder_view1->has_payload(102));

    // Try to reset while an active uncommitted reservation exists
    xkv_hot_reservation active_res = pool_ptr->reserve(2);
    assert(active_res.valid());
    std::string err;
    assert(!pool_ptr->reset(&err));
    assert(err.find("active live reservations") != std::string::npos);
    // Invariant: Pool was not mutated
    assert(holder_view1->get_bound() == 4);
    assert(holder_view1->get_reserved() == 2);
    assert(holder_view1->has_payload(102));

    // Rollback active reservation
    active_res.rollback();
    assert(pool_ptr->get_reserved() == 0);

    // Now perform reset
    assert(pool_ptr->reset(&err));

    // Check object identity is preserved
    assert(pool_ptr.get() == original_address);
    assert(holder_view1.get() == original_address);
    assert(holder_view2.get() == original_address);

    // Both holders must see fully cleared state
    assert(holder_view1->get_capacity() == 8);
    assert(holder_view1->get_free() == 8);
    assert(holder_view1->get_bound() == 0);
    assert(holder_view1->get_reserved() == 0);
    assert(holder_view1->get_live() == 0);
    assert(holder_view1->get_peak() == 0); // Peak policy: 0 on reset
    assert(!holder_view1->has_payload(101));
    assert(!holder_view2->has_payload(102));

    // All slots are immediately reusable from both views
    xkv_hot_reservation reuse_res = holder_view2->reserve(8);
    assert(reuse_res.valid());
    assert(reuse_res.size() == 8);
    assert(holder_view1->get_reserved() == 8);
    assert(holder_view1->get_free() == 0);

    std::vector<uint64_t> new_pids = {201, 202, 203, 204, 205, 206, 207, 208};
    std::vector<uint64_t> new_gens = {2, 2, 2, 2, 2, 2, 2, 2};
    assert(reuse_res.commit(new_pids, new_gens));

    for (uint64_t pid : new_pids) {
        assert(holder_view1->has_payload(pid));
    }
    assert(holder_view2->get_bound() == 8);
}

int main() {
    std::cout << "=== Running test-xkv-hot ===" << std::endl;

    test_capacity_zero_and_atomic_failure();
    test_raii_and_rollback();
    test_commit_and_lookups();
    test_stale_release_rejection();
    test_slot_reuse_and_aba_protection();
    test_accounting_and_peak();
    test_concurrency_no_duplicate_allocation();
    test_reset_preserving_identity_and_shared_view();

    std::cout << "=== All test-xkv-hot tests passed successfully! ===" << std::endl;
    return 0;
}
