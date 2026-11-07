#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "tq/alloc/events.h"

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

static_assert(std::is_trivially_copyable<LeaseEvent>::value,
              "LeaseEvent must be trivially copyable");
static_assert(std::is_trivially_copyable<PlanEvent>::value,
              "PlanEvent must be trivially copyable");

void test_lease_event_value_semantics() {
    LeaseEvent event;
    CHECK(event.ts_ns == 0);
    CHECK(event.leg_id == 0);
    CHECK(event.seq == 0);
    CHECK(event.lifetime == Lifetime::process);
    CHECK(event.reuse_class == ReuseClass::frame);
    CHECK(event.site == 0);
    CHECK(event.bytes == 0);
    CHECK(event.slot == 0);
    CHECK(event.slot_gen == 0);
    CHECK(event.arena_gen == 0);
    CHECK(event.window == 0);
    CHECK(event.request == 0);
    CHECK(event.action == LeaseAction::lease);
    CHECK(event.from == State::free);
    CHECK(event.to == State::free);

    event.ts_ns = 1000;
    event.leg_id = 3;
    event.seq = 7;
    event.lifetime = Lifetime::step;
    event.reuse_class = ReuseClass::staging;
    event.site = 42;
    event.bytes = 4096;
    event.slot = 1;
    event.slot_gen = 2;
    event.arena_gen = 5;
    event.window = 4;
    event.request = 9;
    event.action = LeaseAction::quarantine;
    event.from = State::leased;
    event.to = State::quarantined;

    LeaseEvent copy = event;
    copy.seq = 8;
    copy.to = State::free;
    CHECK(event.seq == 7);
    CHECK(event.to == State::quarantined);
    CHECK(copy.seq == 8);
    CHECK(copy.to == State::free);
    CHECK(copy.lifetime == Lifetime::step);
    CHECK(copy.reuse_class == ReuseClass::staging);
}

void test_plan_event_value_semantics() {
    PlanEvent event;
    CHECK(event.ts_ns == 0);
    CHECK(event.leg_id == 0);
    CHECK(event.seq == 0);
    CHECK(event.plan_key.n_tokens == 1);
    CHECK(event.kind == PlanKind::miss);
    CHECK(event.graph_exec_create == 0);
    CHECK(event.graph_exec_destroy == 0);
    CHECK(event.relay_uid == 0);
    CHECK(event.arena_gen == 0);

    event.ts_ns = 2000;
    event.leg_id = 4;
    event.seq = 11;
    event.plan_key.model = 12;
    event.plan_key.window = 2;
    event.plan_key.n_tokens = 4;
    event.plan_key.n_outputs = 8;
    event.plan_key.rs_class = -2;
    event.plan_key.n_kv_bucket = 255;
    event.plan_key.alloc_gen = 19;
    event.kind = PlanKind::replace;
    event.graph_exec_create = 21;
    event.graph_exec_destroy = 22;
    event.relay_uid = 23;
    event.arena_gen = 24;

    PlanEvent copy = event;
    copy.kind = PlanKind::hit;
    copy.plan_key.model = 13;
    CHECK(event.kind == PlanKind::replace);
    CHECK(event.plan_key.model == 12);
    CHECK(copy.plan_key.model == 13);
    CHECK(copy.plan_key.n_kv_bucket == 255);
}

} // namespace

int main() {
    test_lease_event_value_semantics();
    test_plan_event_value_semantics();

    if (failures != 0) {
        std::fprintf(stderr, "%d event checks failed\n", failures);
        return 1;
    }
    std::printf("event tests passed\n");
    return 0;
}
