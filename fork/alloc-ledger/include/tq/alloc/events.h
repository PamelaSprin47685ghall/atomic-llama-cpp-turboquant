#pragma once

#include <cstdint>
#include <type_traits>

#include "tq/alloc/arena.h"
#include "tq/alloc/plan_cache.h"

namespace tq::alloc {

enum class AllocAction : std::uint8_t {
    lease = 0,
    release = 1,
    drain = 2,
    quarantine = 3,
    cancel = 4,
    advance_window = 5,
};

enum class PlanAction : std::uint8_t {
    hit = 0,
    miss = 1,
    insert = 2,
    replace = 3,
    evict = 4,
    flush = 5,
    pin = 6,
    unpin = 7,
};

enum class Verdict : std::uint8_t {
    unknown = 0,
    pass = 1,
    fail = 2,
};

// Value record of one slot state change. site is an opaque allocation-site
// id. This header defines data only. The caller decides when to store or
// transmit a record.
struct AllocEvent {
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
    AllocAction action = AllocAction::lease;
    State from = State::free;
    State to = State::free;
};

// Value record of one plan cache operation. The exec handles are opaque
// counters, not pointers.
struct PlanEvent {
    std::uint64_t ts_ns = 0;
    std::uint32_t leg_id = 0;
    std::uint64_t seq = 0;
    PlanKey plan_key;
    PlanAction action = PlanAction::miss;
    std::uint64_t graph_exec_create = 0;
    std::uint64_t graph_exec_destroy = 0;
    std::uint64_t relay_uid = 0;
    std::uint64_t arena_gen = 0;
};

// Value description of one allocation leg. SHA fields hold stable 64-bit
// digests. closure4 holds four closure values. reserved counts cold reserve
// units. generation_span is the number of arena generations the leg saw.
struct LegManifest {
    std::uint32_t leg_id = 0;
    std::uint64_t build_sha = 0;
    std::uint64_t closure4[4] = {};
    std::uint64_t model_sha = 0;
    std::uint64_t env_sha = 0;
    std::uint64_t platform_fingerprint = 0;
    double loadavg = 0.0;
    bool gpu_idle = false;
    std::uint64_t iova_last = 0;
    double decode_submit_p50 = 0.0;
    double decode_submit_p95 = 0.0;
    double tok_s = 0.0;
    bool all_511 = false;
    double ttft = 0.0;
    std::uint64_t generation_span = 0;
    Verdict verdict = Verdict::unknown;
};

static_assert(std::is_trivially_copyable<AllocEvent>::value,
              "AllocEvent must be a value type");
static_assert(std::is_trivially_copyable<PlanEvent>::value,
              "PlanEvent must be a value type");
static_assert(std::is_trivially_copyable<LegManifest>::value,
              "LegManifest must be a value type");

} // namespace tq::alloc
