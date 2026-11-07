#include <cstdio>
#include <cstdint>

#include "tq/alloc/arena.h"

using namespace tq::alloc;

namespace {

int failures = 0;

#define CHECK(expr)                                                            \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,        \
                         __LINE__, #expr);                                     \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

void test_i1_disjoint_live_leases() {
    Slot storage[4];
    Arena arena(storage, 4, 7, 3, 1000);

    Lease a;
    Lease b;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 10}, a).ok());
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {10, 20}, b).ok());
    CHECK(a.slot != b.slot);
    CHECK(!a.extent.overlaps(b.extent));
    CHECK(arena.live_count() == 2);
    CHECK(arena.live_bytes() == 20);

    Lease bad;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {5, 15}, bad)
              .code == Code::overlap);
    CHECK(!bad.live);
    CHECK(arena.live_count() == 2);

    CHECK(arena.release(a).ok());
    CHECK(arena.release(b).ok());
    CHECK(arena.live_count() == 0);
}

void test_i2_reuse_after_previous_end() {
    Slot storage[2];
    Arena arena(storage, 2, 1, 0, 1000);

    Lease a;
    Lease b;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {10, 20}, a).ok());
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {20, 30}, b).ok());
    CHECK(arena.release(a).ok());
    CHECK(arena.release(b).ok());

    // A reused slot starts at or after the previous lease end. The extent is
    // half-open, so begin == last_end is valid.
    Lease reuse;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {30, 40}, reuse)
              .ok());
    CHECK(arena.release(reuse).ok());

    // Every free slot has last_end >= 30, so a lower begin is out of order.
    Lease low;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {5, 8}, low).code ==
          Code::out_of_order);
    CHECK(!low.live);
}

void test_i3_footprint_bound() {
    Slot storage[2];
    Arena arena(storage, 2, 2, 0, 100);

    Lease a;
    Lease b;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 60}, a).ok());
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {60, 100}, b).ok());
    CHECK(arena.live_bytes() == 100);
    CHECK(arena.capacity() == 2);

    // The footprint is fully leased. One more byte exceeds the footprint.
    // A single extent above the footprint is rejected as out of range.
    Lease over;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 1}, over)
              .code == Code::over_capacity);
    Lease outside;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 101}, outside)
              .code == Code::out_of_range);
    CHECK(arena.live_bytes() == 100);

    // Hot path: repeated acquire and release does not grow the arena.
    CHECK(arena.release(a).ok());
    CHECK(arena.release(b).ok());
    CHECK(arena.advance_window(1).ok());

    // The full footprint is valid when it is one extent.
    Lease whole;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 100}, whole)
              .ok());
    CHECK(arena.release(whole).ok());

    std::uint64_t offset = 0;
    std::uint32_t window = 1;
    for (int i = 0; i < 1000; ++i) {
        if (offset + 10 > 100) {
            CHECK(arena.advance_window(++window).ok());
            offset = 0;
        }
        Lease loop;
        CHECK(arena.acquire(Lifetime::request, ReuseClass::cold,
                            {offset, offset + 10}, loop)
                  .ok());
        CHECK(arena.release(loop).ok());
        offset += 10;
    }
    CHECK(arena.capacity() == 2);
    CHECK(arena.live_bytes() == 0);
    CHECK(arena.live_count() == 0);
}

void test_i4_generation_match() {
    Slot storage[2];
    Arena arena(storage, 2, 3, 0, 100);

    Lease a;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 10}, a).ok());

    Lease forged_slot = a;
    forged_slot.slot_gen += 1;
    CHECK(arena.release(forged_slot).code == Code::stale_lease);

    Lease forged_arena_gen = a;
    forged_arena_gen.arena_gen += 1;
    CHECK(arena.release(forged_arena_gen).code == Code::stale_lease);

    Lease forged_arena = a;
    forged_arena.arena += 1;
    CHECK(arena.release(forged_arena).code == Code::wrong_arena);

    CHECK(arena.live_count() == 1);
    CHECK(arena.release(a).ok());
}

void test_i5_window_match() {
    Slot storage[2];
    Arena arena(storage, 2, 4, 2, 100);

    Lease a;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 10}, a).ok());
    CHECK(arena.advance_window(3).code == Code::busy);

    CHECK(arena.release(a).ok());
    CHECK(arena.advance_window(3).ok());
    CHECK(arena.window() == 3);
    CHECK(arena.generation() == 2);

    // A lease from the previous window is rejected.
    Lease stale = a;
    CHECK(arena.release(stale).code == Code::wrong_window);

    Lease b;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 10}, b).ok());
    CHECK(b.window == 3);
    CHECK(arena.release(b).ok());
}

void test_i6_deferred_reuse_after_drain() {
    Slot storage[1];
    Arena arena(storage, 1, 5, 0, 100);

    Lease a;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::frame, {0, 10}, a).ok());
    CHECK(arena.release(a).ok());
    CHECK(arena.slot_at(0)->state == State::draining);

    // The slot is not reusable before drain.
    Lease blocked;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::frame, {10, 20}, blocked)
              .code == Code::no_slot);

    CHECK(arena.drain(0).ok());
    CHECK(arena.slot_at(0)->state == State::free);
    CHECK(arena.acquire(Lifetime::request, ReuseClass::frame, {10, 20}, blocked)
              .ok());
    CHECK(arena.release(blocked).ok());
    CHECK(arena.drain(0).ok());
}

void test_i7_quarantine_and_cold_reserve() {
    Slot storage[2];
    Arena arena(storage, 2, 6, 0, 100);

    Lease a;
    CHECK(arena.acquire(Lifetime::plan, ReuseClass::cold, {0, 10}, a).ok());
    CHECK(arena.quarantine(a).ok());
    CHECK(arena.cold_reserve() == 1);
    CHECK(arena.live_count() == 0);
    CHECK(arena.slot_at(a.slot)->state == State::quarantined);
    CHECK(arena.stats(Lifetime::plan).quarantines == 1);

    // The quarantined extent is bad storage and stays unavailable.
    Lease bad;
    CHECK(arena.acquire(Lifetime::plan, ReuseClass::cold, {0, 10}, bad).code ==
          Code::overlap);
    CHECK(arena.acquire(Lifetime::plan, ReuseClass::cold, {10, 20}, bad).ok());
    CHECK(arena.release(bad).ok());

    // A repeated quarantine is rejected and does not add reserve.
    CHECK(arena.quarantine(a).code == Code::duplicate_release);
    CHECK(arena.cold_reserve() == 1);
}

void test_i8_repeated_release() {
    Slot storage[2];
    Arena arena(storage, 2, 7, 0, 100);

    Lease a;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 10}, a).ok());
    CHECK(arena.release(a).ok());
    CHECK(arena.live_bytes() == 0);

    CHECK(arena.release(a).code == Code::duplicate_release);
    CHECK(arena.live_bytes() == 0);
    CHECK(arena.live_count() == 0);
    CHECK(arena.slot_at(a.slot)->state == State::free);
}

void test_convenience_acquire() {
    Slot storage[3];
    Arena arena(storage, 3, 8, 0, 100);

    Lease zero;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, std::size_t{0},
                        zero)
              .code == Code::invalid_argument);
    Lease huge;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, std::size_t{101},
                        huge)
              .code == Code::out_of_range);

    Lease a;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, std::size_t{10}, a)
              .ok());
    CHECK(a.slot == 0);
    CHECK(a.extent.begin == 0);
    CHECK(a.extent.end == 10);
    CHECK(arena.release(a).ok());

    // Free slots: slot0 last_end=10, slot1 and slot2 last_end=0. The
    // smallest last_end wins.
    Lease b;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, std::size_t{5}, b)
              .ok());
    CHECK(b.slot == 1);
    CHECK(b.extent.begin == 0);
    CHECK(b.extent.end == 5);

    // slot2 has the smallest last_end but [0,5) overlaps b. The next valid
    // candidate is slot0 at [10,15).
    Lease c;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, std::size_t{5}, c)
              .ok());
    CHECK(c.slot == 0);
    CHECK(c.extent.begin == 10);
    CHECK(c.extent.end == 15);

    // 10 live bytes plus 95 would exceed the footprint.
    Lease over;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, std::size_t{95},
                        over)
              .code == Code::over_capacity);

    // A single-slot arena has no free slot while the lease is live.
    Slot one_storage[1];
    Arena one(one_storage, 1, 9, 0, 100);
    Lease held;
    CHECK(one.acquire(Lifetime::request, ReuseClass::cold, std::size_t{1}, held)
              .ok());
    Lease none;
    CHECK(one.acquire(Lifetime::request, ReuseClass::cold, std::size_t{1}, none)
              .code == Code::no_slot);
    CHECK(one.release(held).ok());
}

void test_lifetime_accounting() {
    Slot storage[4];
    Arena arena(storage, 4, 10, 0, 1000);

    Lease a;
    Lease b;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {0, 10}, a).ok());
    CHECK(arena.acquire(Lifetime::step, ReuseClass::cold, {10, 30}, b).ok());
    CHECK(arena.stats(Lifetime::request).live_bytes == 10);
    CHECK(arena.stats(Lifetime::request).live_count == 1);
    CHECK(arena.stats(Lifetime::request).leases == 1);
    CHECK(arena.stats(Lifetime::step).live_bytes == 20);
    CHECK(arena.stats(Lifetime::step).live_count == 1);
    CHECK(arena.stats(Lifetime::step).high_water == 20);
    CHECK(arena.stats().live_bytes == 30);
    CHECK(arena.stats().high_water == 30);
    CHECK(arena.stats().leases == 2);
    CHECK(arena.high_water() == 30);

    CHECK(arena.release(a).ok());
    CHECK(arena.stats(Lifetime::request).live_bytes == 0);
    CHECK(arena.stats(Lifetime::request).releases == 1);
    CHECK(arena.stats().live_bytes == 20);
    CHECK(arena.high_water() == 30);

    CHECK(arena.release(b).ok());
    CHECK(arena.stats(Lifetime::step).releases == 1);
    CHECK(arena.stats().releases == 2);
    CHECK(arena.high_water() == 30);

    Lease q;
    CHECK(arena.acquire(Lifetime::plan, ReuseClass::cold, {40, 50}, q).ok());
    CHECK(arena.quarantine(q).ok());
    CHECK(arena.stats(Lifetime::plan).quarantines == 1);
    CHECK(arena.stats().quarantines == 1);
    CHECK(arena.cold_reserve() == 1);
}

void test_instances_are_independent() {
    Slot first_storage[2];
    Slot second_storage[2];
    Arena first(first_storage, 2, 11, 0, 100);
    Arena second(second_storage, 2, 12, 0, 100);

    Lease a;
    CHECK(first.acquire(Lifetime::request, ReuseClass::cold, {0, 10}, a).ok());
    CHECK(second.live_count() == 0);
    CHECK(second.live_bytes() == 0);
    CHECK(second.stats(Lifetime::request).live_count == 0);

    Lease b;
    CHECK(second.acquire(Lifetime::step, ReuseClass::cold, {0, 10}, b).ok());
    CHECK(first.release(a).ok());
    CHECK(second.live_count() == 1);
    CHECK(second.live_bytes() == 10);
    CHECK(second.release(b).ok());
}

void test_external_values_cannot_mutate_state() {
    Slot storage[2];
    Arena arena(storage, 2, 13, 0, 100);

    Extent input{0, 10};
    Lease a;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, input, a).ok());
    input.begin = 50;
    input.end = 60;
    CHECK(arena.slot_at(a.slot)->extent.begin == 0);
    CHECK(arena.slot_at(a.slot)->extent.end == 10);
    CHECK(arena.live_bytes() == 10);

    // The returned lease is a value. A changed copy is rejected.
    Lease copy = a;
    copy.extent.begin = 99;
    CHECK(arena.release(copy).code == Code::invalid_lease);
    CHECK(arena.live_count() == 1);

    copy = a;
    copy.reuse = ReuseClass::frame;
    CHECK(arena.release(copy).code == Code::invalid_lease);
    CHECK(arena.live_count() == 1);

    copy = a;
    copy.lifetime = Lifetime::model;
    CHECK(arena.release(copy).code == Code::invalid_lease);
    CHECK(arena.live_count() == 1);

    CHECK(arena.release(a).ok());
}

void test_arguments_and_drain_errors() {
    Slot storage[1];
    Arena arena(storage, 1, 14, 0, 100);

    Lease a;
    CHECK(arena.acquire(Lifetime::request, ReuseClass::cold, {5, 5}, a).code ==
          Code::invalid_argument);
    CHECK(arena.acquire(static_cast<Lifetime>(7), ReuseClass::cold, {0, 1}, a)
              .code == Code::invalid_argument);
    CHECK(arena.acquire(Lifetime::request, static_cast<ReuseClass>(3), {0, 1},
                        a)
              .code == Code::invalid_argument);
    CHECK(arena.drain(0).code == Code::not_draining);
    CHECK(arena.drain(1).code == Code::invalid_argument);
    CHECK(arena.slot_at(1) == nullptr);
}

} // namespace

int main() {
    test_i1_disjoint_live_leases();
    test_i2_reuse_after_previous_end();
    test_i3_footprint_bound();
    test_i4_generation_match();
    test_i5_window_match();
    test_i6_deferred_reuse_after_drain();
    test_i7_quarantine_and_cold_reserve();
    test_i8_repeated_release();
    test_convenience_acquire();
    test_lifetime_accounting();
    test_instances_are_independent();
    test_external_values_cannot_mutate_state();
    test_arguments_and_drain_errors();

    if (failures != 0) {
        std::fprintf(stderr, "%d arena checks failed\n", failures);
        return 1;
    }
    std::printf("arena tests passed\n");
    return 0;
}
