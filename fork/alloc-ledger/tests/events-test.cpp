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

static_assert(std::is_trivially_copyable<AllocEvent>::value,
              "AllocEvent must be trivially copyable");
static_assert(std::is_trivially_copyable<PlanEvent>::value,
              "PlanEvent must be trivially copyable");
static_assert(std::is_trivially_copyable<LegManifest>::value,
              "LegManifest must be trivially copyable");

void test_alloc_event_value_semantics() {
    AllocEvent event;
    CHECK(event.seq == 0);
    CHECK(event.window == 0);
    CHECK(event.arena_gen == 0);
    CHECK(event.slot == 0);
    CHECK(event.cls == Class::frame);
    CHECK(event.from == State::free);
    CHECK(event.to == State::free);
    CHECK(event.bytes == 0);
    CHECK(event.tick == 0);

    event.seq = 7;
    event.window = 3;
    event.arena_gen = 2;
    event.slot = 1;
    event.cls = Class::staging;
    event.from = State::free;
    event.to = State::leased;
    event.bytes = 4096;
    event.tick = 11;

    AllocEvent copy = event;
    copy.seq = 8;
    copy.to = State::quarantined;
    CHECK(event.seq == 7);
    CHECK(event.to == State::leased);
    CHECK(copy.to == State::quarantined);
}

void test_plan_event_value_semantics() {
    PlanEvent event;
    CHECK(event.seq == 0);
    CHECK(event.kind == PlanEvent::Kind::miss);
    CHECK(event.tick == 0);

    event.seq = 3;
    event.kind = PlanEvent::Kind::evict;
    event.key.model = 4;
    event.key.window = 2;
    event.key.footprint = 8192;
    event.tick = 9;

    PlanEvent copy = event;
    copy.kind = PlanEvent::Kind::hit;
    copy.key.model = 5;
    CHECK(event.kind == PlanEvent::Kind::evict);
    CHECK(event.key.model == 4);
    CHECK(copy.key.model == 5);
}

void test_leg_manifest_value_semantics() {
    LegManifest manifest;
    CHECK(manifest.model == 0);
    CHECK(manifest.window == 0);
    CHECK(manifest.footprint == 0);
    CHECK(manifest.slots == 0);
    CHECK(manifest.reserved == 0);
    CHECK(manifest.generation == 0);

    manifest.model = 12;
    manifest.window = 1;
    manifest.footprint = 65536;
    manifest.slots = 8;
    manifest.reserved = 1;
    manifest.generation = 5;

    LegManifest copy = manifest;
    copy.reserved = 3;
    CHECK(manifest.reserved == 1);
    CHECK(copy.reserved == 3);
    CHECK(copy.footprint == 65536);
}

} // namespace

int main() {
    test_alloc_event_value_semantics();
    test_plan_event_value_semantics();
    test_leg_manifest_value_semantics();

    if (failures != 0) {
        std::fprintf(stderr, "%d event checks failed\n", failures);
        return 1;
    }
    std::printf("event tests passed\n");
    return 0;
}
