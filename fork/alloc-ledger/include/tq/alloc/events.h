#pragma once

#include <cstdint>
#include <type_traits>

#include "tq/alloc/arena.h"
#include "tq/alloc/plan_cache.h"

namespace tq::alloc {

// In-memory lease actions. These values are not the WS0 allocation actions.
// A C adapter maps them to the external schema.
enum class LeaseAction : std::uint8_t {
    lease = 0,
    release = 1,
    drain = 2,
    quarantine = 3,
    cancel = 4,
    advance_window = 5,
};

// In-memory plan cache actions. A C adapter maps them to the WS0 PlanEvent
// action values. The mapping lives in README.md, not in this header.
enum class PlanKind : std::uint8_t {
    hit = 0,
    miss = 1,
    insert = 2,
    replace = 3,
    evict = 4,
    flush = 5,
    pin = 6,
    unpin = 7,
};

// Value record of one slot state change inside this library. This is a
// memory event. It is not the WS0 AllocEvent record. The WS0 schema observes
// uprobe and kprobe data and owns a different class, action, and window
// vocabulary. The C adapter writes the external JSON.
struct LeaseEvent {
    std::uint64_t ts_ns = 0;
    std::uint32_t leg_id = 0;
    std::uint64_t seq = 0;
    Lifetime lifetime = Lifetime::process;
    ReuseClass reuse_class = ReuseClass::frame;
    std::uint32_t site = 0;
    std::uint64_t bytes = 0;
    std::uint32_t slot = 0;
    std::uint64_t slot_gen = 0;
    std::uint64_t arena_gen = 0;
    std::uint32_t window = 0;
    std::uint32_t request = 0;
    LeaseAction action = LeaseAction::lease;
    State from = State::free;
    State to = State::free;
};

// Value record of one plan cache operation. This is a memory event. The C
// adapter serializes it to the WS0 PlanEvent schema, including the
// plan_key string from PlanKey::describe() or PlanKey::hash().
struct PlanEvent {
    std::uint64_t ts_ns = 0;
    std::uint32_t leg_id = 0;
    std::uint64_t seq = 0;
    PlanKey plan_key;
    PlanKind kind = PlanKind::miss;
    std::uint64_t graph_exec_create = 0;
    std::uint64_t graph_exec_destroy = 0;
    std::uint64_t relay_uid = 0;
    std::uint64_t arena_gen = 0;
};

static_assert(std::is_trivially_copyable<LeaseEvent>::value,
              "LeaseEvent must be a value type");
static_assert(std::is_trivially_copyable<PlanEvent>::value,
              "PlanEvent must be a value type");

} // namespace tq::alloc
