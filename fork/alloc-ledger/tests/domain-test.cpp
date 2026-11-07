#include <cstdio>
#include <cstdint>

#include "tq/alloc/domain.h"

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

void test_cancel_drain_reuse() {
    Slot storage[6];
    Arena arena(storage, 6, 20, 0, 1000);
    Domain domain(arena);

    CHECK(domain.state() == DomainState::idle);

    Lease a;
    CHECK(domain.lease(Class::frame, {0, 10}, a).ok());
    CHECK(domain.state() == DomainState::active);

    CHECK(domain.cancel(a).ok());
    CHECK(domain.state() == DomainState::draining);
    CHECK(domain.pending_drains() == 1);
    CHECK(arena.slot_at(a.slot)->state == State::draining);

    // A draining domain refuses a nonblocking lease.
    Lease blocked;
    CHECK(domain.try_lease(Class::frame, {10, 20}, blocked).code == Code::busy);
    CHECK(!blocked.live);

    CHECK(domain.drain().ok());
    CHECK(domain.pending_drains() == 0);
    CHECK(domain.state() == DomainState::idle);

    Lease reused;
    CHECK(domain.try_lease(Class::frame, {10, 20}, reused).ok());
    CHECK(reused.slot == a.slot);
    CHECK(domain.state() == DomainState::active);

    CHECK(domain.release(reused).ok());
    CHECK(domain.state() == DomainState::draining);
    CHECK(domain.drain().ok());
    CHECK(domain.state() == DomainState::idle);
}

void test_quarantine_transition() {
    Slot storage[6];
    Arena arena(storage, 6, 21, 0, 1000);
    Domain domain(arena);

    Lease a;
    CHECK(domain.lease(Class::staging, {0, 10}, a).ok());
    CHECK(domain.quarantine(a).ok());
    CHECK(domain.state() == DomainState::quarantined);
    CHECK(domain.cold_reserve() == 1);
    CHECK(arena.slot_at(a.slot)->state == State::quarantined);

    // A repeated quarantine changes nothing.
    CHECK(domain.quarantine(a).code == Code::duplicate_release);
    CHECK(domain.cold_reserve() == 1);

    // The domain can still lease from the remaining slots.
    Lease b;
    CHECK(domain.lease(Class::cold, {20, 30}, b).ok());
    CHECK(domain.state() == DomainState::active);
    CHECK(domain.release(b).ok());
    CHECK(domain.state() == DomainState::idle);
}

void test_release_defers_frame_reuse() {
    Slot storage[6];
    Arena arena(storage, 6, 22, 0, 1000);
    Domain domain(arena);

    Lease a;
    CHECK(domain.lease(Class::frame, {0, 10}, a).ok());
    CHECK(domain.release(a).ok());
    CHECK(domain.state() == DomainState::draining);

    Lease blocked;
    CHECK(domain.try_lease(Class::frame, {10, 20}, blocked).code == Code::busy);

    CHECK(domain.drain().ok());
    CHECK(domain.state() == DomainState::idle);
    CHECK(domain.try_lease(Class::frame, {10, 20}, blocked).ok());
    CHECK(blocked.slot == a.slot);
    CHECK(domain.release(blocked).ok());
    CHECK(domain.drain().ok());
}

void test_cold_release_is_immediate() {
    Slot storage[2];
    Arena arena(storage, 2, 23, 0, 1000);
    Domain domain(arena);

    Lease a;
    CHECK(domain.lease(Class::cold, {0, 10}, a).ok());
    CHECK(domain.release(a).ok());
    CHECK(domain.state() == DomainState::idle);
    CHECK(domain.pending_drains() == 0);
    CHECK(arena.slot_at(a.slot)->state == State::free);

    // cancel applies to frame and staging leases only.
    Lease b;
    CHECK(domain.lease(Class::cold, {10, 20}, b).ok());
    CHECK(domain.cancel(b).code == Code::invalid_argument);
    CHECK(domain.release(b).ok());
}

void test_stale_and_cross_window_rejected() {
    Slot storage[4];
    Arena arena(storage, 4, 24, 0, 1000);
    Domain domain(arena);

    Lease a;
    CHECK(domain.lease(Class::cold, {0, 10}, a).ok());
    CHECK(domain.release(a).ok());

    Lease stale = a;
    CHECK(domain.release(stale).code == Code::duplicate_release);

    // A changed copy is rejected and the live lease stays valid.
    Lease live;
    CHECK(domain.lease(Class::frame, {10, 20}, live).ok());
    Lease copy = live;
    copy.slot_gen += 1;
    CHECK(domain.release(copy).code == Code::stale_lease);
    CHECK(arena.live_count() == 1);
    CHECK(domain.release(live).ok());
    CHECK(domain.drain().ok());

    CHECK(arena.advance_window(1).ok());
    CHECK(domain.generation() == arena.generation());
    CHECK(domain.release(stale).code == Code::wrong_window);
    CHECK(domain.cancel(stale).code == Code::invalid_argument);
}

void test_instances_are_independent() {
    Slot first_storage[2];
    Slot second_storage[2];
    Arena first_arena(first_storage, 2, 25, 0, 1000);
    Arena second_arena(second_storage, 2, 26, 0, 1000);
    Domain first(first_arena);
    Domain second(second_arena);

    Lease a;
    CHECK(first.lease(Class::frame, {0, 10}, a).ok());
    CHECK(first.state() == DomainState::active);
    CHECK(second.state() == DomainState::idle);
    CHECK(second.arena().live_count() == 0);

    Lease b;
    CHECK(second.lease(Class::frame, {0, 10}, b).ok());
    CHECK(first.cancel(a).ok());
    CHECK(second.state() == DomainState::active);
    CHECK(second.release(b).ok());
    CHECK(second.drain().ok());
}

} // namespace

int main() {
    test_cancel_drain_reuse();
    test_quarantine_transition();
    test_release_defers_frame_reuse();
    test_cold_release_is_immediate();
    test_stale_and_cross_window_rejected();
    test_instances_are_independent();

    if (failures != 0) {
        std::fprintf(stderr, "%d domain checks failed\n", failures);
        return 1;
    }
    std::printf("domain tests passed\n");
    return 0;
}
