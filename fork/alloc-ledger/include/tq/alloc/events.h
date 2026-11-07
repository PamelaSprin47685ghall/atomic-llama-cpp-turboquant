#pragma once

#include <cstdint>
#include <type_traits>

#include "tq/alloc/arena.h"
#include "tq/alloc/plan_cache.h"

namespace tq::alloc {

// Value record of one slot state change. This header defines data only. The
// caller decides when to store or transmit a record.
struct AllocEvent {
    std::uint64_t seq = 0;
    std::uint32_t window = 0;
    std::uint64_t arena_gen = 0;
    std::uint32_t slot = 0;
    Class cls = Class::frame;
    State from = State::free;
    State to = State::free;
    std::uint64_t bytes = 0;
    std::uint64_t tick = 0;
};

// Value record of one plan cache operation.
struct PlanEvent {
    enum class Kind : std::uint8_t {
        hit = 0,
        miss = 1,
        insert = 2,
        replace = 3,
        evict = 4,
        flush = 5,
        pin = 6,
        unpin = 7,
    };

    std::uint64_t seq = 0;
    Kind kind = Kind::miss;
    PlanKey key;
    std::uint64_t tick = 0;
};

// Value description of one allocation leg. reserved counts cold reserve
// units. generation is the arena generation at capture time.
struct LegManifest {
    std::uint64_t model = 0;
    std::uint32_t window = 0;
    std::uint64_t footprint = 0;
    std::uint32_t slots = 0;
    std::uint32_t reserved = 0;
    std::uint64_t generation = 0;
};

static_assert(std::is_trivially_copyable<AllocEvent>::value,
              "AllocEvent must be a value type");
static_assert(std::is_trivially_copyable<PlanEvent>::value,
              "PlanEvent must be a value type");
static_assert(std::is_trivially_copyable<LegManifest>::value,
              "LegManifest must be a value type");

} // namespace tq::alloc
