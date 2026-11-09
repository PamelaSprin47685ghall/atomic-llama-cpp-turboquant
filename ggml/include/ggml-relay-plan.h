#pragma once

// TP5 pillar-5 plan/frame/pool policy. Header-only, no HIP, no allocation.
//
// This header owns the pure decisions that relay.hip and the compose runtime
// both need:
//   * the K=8 plan cache identity and LRU eviction,
//   * the frame LRU keyed by graph uid and plan generation,
//   * the exact channel-pool layout and its budget,
//   * the per-window plan-slot table for channel_state ownership,
//   * the environment switch matrix with the legacy defaults.
//
// The owners keep their own storage; this header only names slots and
// offsets. A decision here never frees a live owner: eviction candidates are
// unpinned plan slots and idle channel slots.

#include "ggml-relay.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace ggml_relay_plan {

// One provider owns at most eight prepared plans and eight captured frames.
constexpr std::size_t plan_capacity      = 8;
constexpr std::size_t frame_capacity_max = 8;
// The relay channel table is fixed at 256 slots; the T4/MTP working set
// uses at most 181 of them.
constexpr std::size_t channel_capacity   = 256;
// control is 16 cache-line words (generation, 5 ready, 5 ack, 5 error).
constexpr std::size_t control_bytes      = 16 * 64;
constexpr std::size_t pool_budget_default = 68ull << 20;
constexpr std::size_t pool_baseline       = 8ull << 20;
// 256 * (1024 + 65536 * 4 * 6) bytes, the pre-pillar-5 fixed pool.
constexpr std::size_t pool_bytes_legacy   = 402915328;

// Seven coarse identity fields plus the channel count. Two epochs with the
// same key must be interchangeable for every captured graph. `channels` is
// part of the identity: a 180- and a 181-channel plan never share a slot.
//
// This is NOT tq::alloc::PlanKey. The F2 allocation ledger keys seven fields
// (model, window, n_tokens, n_outputs, rs_class, n_kv_bucket, alloc_gen) and
// owns arena placement. This relay plan adds `channels` because one allocation
// plan can back relay graphs with different channel tables. The two domains
// stay separate; the C adapters map each one explicitly.
struct plan_key {
    std::uint64_t model      = 0;
    std::uint32_t window     = 0;
    std::uint32_t n_tokens   = 0;
    std::uint32_t n_outputs  = 0;
    std::int32_t  rs_class   = 0;
    std::uint32_t n_kv_bucket = 0;
    std::uint64_t alloc_gen  = 0;
    std::uint32_t channels   = 0;
    // Relay-internal second half of the identity: a digest of every terminal
    // binding (tensor/data pointers, ne, nb) the plan was prepared with. The
    // seven llama fields are coarse; a graph rebuild can keep them identical
    // while the terminal layout or storage changes. Including the binding in
    // the key makes "same key, different binding" a cache miss instead of a
    // fail-closed key bug. It stays 0 on the C boundary; relay.hip fills it.
    std::uint64_t binding    = 0;

    bool operator==(const plan_key & other) const noexcept {
        return model == other.model && window == other.window && n_tokens == other.n_tokens &&
               n_outputs == other.n_outputs && rs_class == other.rs_class &&
               n_kv_bucket == other.n_kv_bucket && alloc_gen == other.alloc_gen &&
               channels == other.channels && binding == other.binding;
    }
    bool operator!=(const plan_key & other) const noexcept { return !(*this == other); }

    static plan_key from_c(const ggml_relay_plan_key & value) noexcept {
        plan_key key;
        key.model       = value.model;
        key.window      = value.window;
        key.n_tokens    = value.n_tokens;
        key.n_outputs   = value.n_outputs;
        key.rs_class    = value.rs_class;
        key.n_kv_bucket = value.n_kv_bucket;
        key.alloc_gen   = value.alloc_gen;
        key.channels    = value.channels;
        return key;
    }
    ggml_relay_plan_key to_c() const noexcept {
        ggml_relay_plan_key value = {};
        value.model       = model;
        value.window      = window;
        value.n_tokens    = n_tokens;
        value.n_outputs   = n_outputs;
        value.rs_class    = rs_class;
        value.n_kv_bucket = n_kv_bucket;
        value.alloc_gen   = alloc_gen;
        value.channels    = channels;
        return value;
    }
};

// Seven lifetime counters. `stale_generation` counts valid slots replaced by
// an insert; it is a diagnostic, not a correctness signal.
struct plan_stats {
    std::uint64_t hits             = 0;
    std::uint64_t misses           = 0;
    std::uint64_t inserts          = 0;
    std::uint64_t evictions        = 0;
    std::uint64_t pinned_refusals  = 0;
    std::uint64_t flushes          = 0;
    std::uint64_t stale_generation = 0;
};

struct plan_slot {
    bool          valid      = false;
    plan_key      key        = {};
    std::uint64_t generation = 0; // unique per insert; never reused
    std::uint64_t last_used  = 0;
    std::uint32_t pins       = 0;
    std::uint32_t channels   = 0;
    std::uint32_t index      = 0; // slot position, indexes the binding snapshot
};

// Exact terminal binding snapshot for one plan slot. The key's `binding`
// digest is only a fast filter: a hit must also pass this full comparison.
// A 64-bit digest collision therefore fails closed instead of reusing a
// different binding. 256 channels x 5 ranks x (2 pointers + 4 ne + 4 nb).
struct binding_entry {
    const ggml_tensor * terminal        = nullptr;
    void *              data            = nullptr;
    std::int64_t        ne[GGML_MAX_DIMS] = {};
    std::size_t         nb[GGML_MAX_DIMS] = {};
};

using binding_set = std::array<std::array<binding_entry, GGML_RELAY_RANKS>, channel_capacity>;

inline void snapshot_binding(binding_set & out, const ggml_tensor * const * terminals,
                             std::size_t count) noexcept {
    for (std::size_t c = 0; c < count; ++c) {
        for (std::size_t r = 0; r < GGML_RELAY_RANKS; ++r) {
            const ggml_tensor * t = terminals[c * GGML_RELAY_RANKS + r];
            binding_entry & e = out[c][r];
            e.terminal = t;
            e.data     = t->data;
            std::memcpy(e.ne, t->ne, sizeof(e.ne));
            std::memcpy(e.nb, t->nb, sizeof(e.nb));
        }
    }
}

inline bool binding_matches(const binding_set & set, const ggml_tensor * const * terminals,
                            std::size_t count) noexcept {
    for (std::size_t c = 0; c < count; ++c) {
        for (std::size_t r = 0; r < GGML_RELAY_RANKS; ++r) {
            const ggml_tensor * t = terminals[c * GGML_RELAY_RANKS + r];
            const binding_entry & e = set[c][r];
            if (e.terminal != t || e.data != t->data ||
                std::memcmp(e.ne, t->ne, sizeof(e.ne)) != 0 ||
                std::memcmp(e.nb, t->nb, sizeof(e.nb)) != 0) {
                return false;
            }
        }
    }
    return true;
}

// Fixed-capacity plan cache. Hit touches the slot and returns it without
// pinning: begin() pins, finish() releases. A miss inserts into an empty slot
// or evicts the least recently used unpinned slot. Every slot pinned fails
// closed (returns null) instead of silently dropping a plan.
class plan_cache {
public:
    static constexpr std::size_t capacity = plan_capacity;

    const plan_slot * find(const plan_key & key) const noexcept {
        for (const auto & slot : slots_) {
            if (slot.valid && slot.key == key) {
                return &slot;
            }
        }
        return nullptr;
    }

    // On a miss, `evicted` receives the generation displaced from the victim
    // slot (0 for an empty victim). The caller must retire that generation's
    // frames and channel slots before using the returned slot.
    plan_slot * acquire(const plan_key & key, std::uint64_t epoch,
                        std::uint64_t * evicted = nullptr) noexcept {
        if (evicted != nullptr) {
            *evicted = 0;
        }
        if (plan_slot * hit = mutable_find(key)) {
            ++stats_.hits;
            hit->last_used = epoch;
            return hit;
        }
        ++stats_.misses;
        const int index = pick_victim();
        if (index < 0) {
            ++stats_.pinned_refusals;
            return nullptr;
        }
        plan_slot & slot = slots_[static_cast<std::size_t>(index)];
        if (slot.valid) {
            ++stats_.evictions;
            ++stats_.stale_generation;
            if (evicted != nullptr) {
                *evicted = slot.generation;
            }
        }
        slot.valid      = true;
        slot.key        = key;
        slot.generation = ++next_generation_;
        slot.last_used  = epoch;
        slot.pins       = 0;
        slot.channels   = key.channels;
        slot.index      = static_cast<std::uint32_t>(index);
        ++stats_.inserts;
        return &slot;
    }

    bool pin(std::uint64_t generation) noexcept {
        for (auto & slot : slots_) {
            if (slot.valid && slot.generation == generation) {
                ++slot.pins;
                return true;
            }
        }
        return false;
    }

    bool release(std::uint64_t generation) noexcept {
        for (auto & slot : slots_) {
            if (slot.valid && slot.generation == generation) {
                if (slot.pins == 0) {
                    return false;
                }
                --slot.pins;
                return true;
            }
        }
        return false;
    }

    bool flush(std::uint64_t generation) noexcept {
        for (auto & slot : slots_) {
            if (slot.valid && slot.generation == generation) {
                if (slot.pins != 0) {
                    ++stats_.pinned_refusals;
                    return false;
                }
                slot = {};
                ++stats_.flushes;
                return true;
            }
        }
        return false;
    }

    bool flush_all() noexcept {
        for (const auto & slot : slots_) {
            if (slot.valid && slot.pins != 0) {
                ++stats_.pinned_refusals;
                return false;
            }
        }
        for (auto & slot : slots_) {
            if (slot.valid) {
                ++stats_.flushes;
            }
            slot = {};
        }
        return true;
    }

    std::size_t live() const noexcept {
        std::size_t count = 0;
        for (const auto & slot : slots_) {
            count += slot.valid ? 1 : 0;
        }
        return count;
    }
    std::size_t pinned() const noexcept {
        std::size_t count = 0;
        for (const auto & slot : slots_) {
            count += slot.valid && slot.pins != 0 ? 1 : 0;
        }
        return count;
    }
    bool empty() const noexcept { return live() == 0; }

    const plan_slot * at(std::size_t index) const noexcept {
        return index < capacity && slots_[index].valid ? &slots_[index] : nullptr;
    }
    const plan_stats & stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = {}; }

private:
    plan_slot * mutable_find(const plan_key & key) noexcept {
        for (auto & slot : slots_) {
            if (slot.valid && slot.key == key) {
                return &slot;
            }
        }
        return nullptr;
    }

    // Empty slot first; otherwise the oldest unpinned slot. Ties break on the
    // lowest index so eviction is deterministic.
    int pick_victim() const noexcept {
        int best = -1;
        for (std::size_t i = 0; i < capacity; ++i) {
            const plan_slot & slot = slots_[i];
            if (!slot.valid) {
                return static_cast<int>(i);
            }
            if (slot.pins != 0) {
                continue;
            }
            if (best < 0 || slot.last_used < slots_[static_cast<std::size_t>(best)].last_used) {
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    std::array<plan_slot, capacity> slots_{};
    plan_stats                      stats_{};
    std::uint64_t                   next_generation_ = 0;
};

// Frame metadata LRU. The provider owns the native graph/exec storage and
// mirrors its per-frame uid, plan generation and last use here. A frame is a
// uid fast-path candidate only when both uid and plan generation match; the
// full signature remains the second-level check. The provider tests its
// current frame directly; find() exposes the same predicate for the CPU
// policy test and for callers that search every slot.
class frame_lru {
public:
    explicit frame_lru(std::size_t capacity = 2) noexcept : capacity_(capacity) {
        if (capacity_ > frame_capacity_max) {
            capacity_ = frame_capacity_max;
        }
    }

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t used() const noexcept {
        std::size_t count = 0;
        for (const auto & entry : entries_) {
            count += entry.used ? 1 : 0;
        }
        return count;
    }

    int find(std::uint64_t uid, std::uint64_t plan_generation) const noexcept {
        if (uid == 0) {
            return -1;
        }
        for (std::size_t i = 0; i < capacity_; ++i) {
            const entry & e = entries_[i];
            if (e.used && e.uid == uid && e.plan_generation == plan_generation) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // Empty frame slot first, then the least recently used frame that is not
    // the current one. Returns -1 when every slot is the current frame.
    int victim(int current) const noexcept {
        for (std::size_t i = 0; i < capacity_; ++i) {
            if (!entries_[i].used) {
                return static_cast<int>(i);
            }
        }
        int best = -1;
        for (std::size_t i = 0; i < capacity_; ++i) {
            if (static_cast<int>(i) == current) {
                continue;
            }
            if (best < 0 || entries_[i].last_used < entries_[static_cast<std::size_t>(best)].last_used) {
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    void assign(int index, std::uint64_t uid, std::uint64_t plan_generation, std::uint64_t now) noexcept {
        if (index < 0 || static_cast<std::size_t>(index) >= capacity_) {
            return;
        }
        entry & e          = entries_[static_cast<std::size_t>(index)];
        e.used             = true;
        e.uid              = uid;
        e.plan_generation  = plan_generation;
        e.last_used        = now;
    }

    void touch(int index, std::uint64_t now) noexcept {
        if (index >= 0 && static_cast<std::size_t>(index) < capacity_) {
            entries_[static_cast<std::size_t>(index)].last_used = now;
        }
    }

    void clear(int index) noexcept {
        if (index >= 0 && static_cast<std::size_t>(index) < capacity_) {
            entries_[static_cast<std::size_t>(index)] = {};
        }
    }

    bool valid(int index) const noexcept {
        return index >= 0 && static_cast<std::size_t>(index) < capacity_ &&
               entries_[static_cast<std::size_t>(index)].used;
    }
    std::uint64_t uid(int index) const noexcept {
        return valid(index) ? entries_[static_cast<std::size_t>(index)].uid : 0;
    }
    std::uint64_t plan_generation(int index) const noexcept {
        return valid(index) ? entries_[static_cast<std::size_t>(index)].plan_generation : 0;
    }

private:
    struct entry {
        bool          used            = false;
        std::uint64_t uid             = 0;
        std::uint64_t plan_generation = 0;
        std::uint64_t last_used       = 0;
    };

    std::array<entry, frame_capacity_max> entries_{};
    std::size_t                           capacity_ = 2;
};

// The control block starts every channel slot and is a 64-byte aligned object
// whose words are 64 bytes apart. Every slot is padded to this boundary so the
// next control block keeps that alignment.
constexpr std::size_t pool_alignment = 64;
static_assert((pool_alignment & (pool_alignment - 1)) == 0, "pool alignment must be a power of two");
static_assert(control_bytes % pool_alignment == 0, "control bytes must preserve pool alignment");

// Exact channel layout. offset[c] is the byte offset of channel c's control
// word; capacity[c] is the payload capacity in F32 elements per rank. Each
// slot is padded to `pool_alignment` bytes, so offset[c] is always aligned
// and the control word can be used as a control object. The total is
// sum(round_up(control_bytes + capacity * 4 * (ranks + 1), pool_alignment)).
struct pool_layout {
    std::array<std::size_t, channel_capacity> offset{};
    std::array<std::size_t, channel_capacity> capacity{};
    std::size_t bytes = 0;
    std::size_t count = 0;

    void clear() noexcept {
        offset.fill(0);
        capacity.fill(0);
        bytes = 0;
        count = 0;
    }
};

// Bytes for one channel slot: its control block plus its payload, rounded up
// so the next control block stays aligned. `elements` is the exact F32 element
// count; the padding is never exposed as capacity.
inline bool pool_slot_bytes(std::size_t elements, std::size_t stride,
                            std::size_t & bytes) noexcept {
    bytes = 0;
    if (stride == 0 || elements == 0 || elements > (SIZE_MAX - control_bytes) / stride) {
        return false;
    }
    const std::size_t raw = control_bytes + elements * stride;
    if (raw > SIZE_MAX - (pool_alignment - 1)) {
        return false;
    }
    bytes = (raw + pool_alignment - 1) & ~(pool_alignment - 1);
    return true;
}

// Byte need for a plan. Returns false on overflow or an invalid element count.
inline bool pool_need(const std::size_t * elements, std::size_t count, std::size_t ranks,
                      std::size_t & need) noexcept {
    need = 0;
    if (elements == nullptr || count == 0 || count > channel_capacity || ranks == 0 ||
        ranks > SIZE_MAX / sizeof(float) - 1) {
        return false;
    }
    const std::size_t stride = sizeof(float) * (ranks + 1);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t slot = 0;
        if (!pool_slot_bytes(elements[i], stride, slot)) {
            return false;
        }
        if (slot > SIZE_MAX - need) {
            return false;
        }
        need += slot;
    }
    return true;
}

// Compute the exact layout when it fits `budget`. The layout is rejected when
// the byte need exceeds the budget; the caller fails closed and names the
// actual value. `out` is cleared on failure.
inline bool compute_pool_layout(const std::size_t * elements, std::size_t count, std::size_t ranks,
                                std::size_t budget, pool_layout & out) noexcept {
    out.clear();
    std::size_t need = 0;
    if (!pool_need(elements, count, ranks, need) || need > budget) {
        return false;
    }
    const std::size_t stride = sizeof(float) * (ranks + 1);
    std::size_t       offset = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t slot = 0;
        if (!pool_slot_bytes(elements[i], stride, slot)) {
            out.clear();
            return false;
        }
        out.offset[i]   = offset;
        out.capacity[i] = elements[i];
        offset += slot;
    }
    out.bytes = offset;
    out.count = count;
    return true;
}

// True when every plan channel fits the existing per-channel capacity.
inline bool layout_covers(const pool_layout & layout, const std::size_t * elements,
                          std::size_t count) noexcept {
    if (layout.count < count) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (elements[i] > layout.capacity[i]) {
            return false;
        }
    }
    return true;
}

enum class pool_growth {
    keep,       // the current layout covers the plan
    grow,       // the plan fits the budget and needs a new layout
    fail_budget // the plan cannot fit the budget at all
};

inline pool_growth pool_growth_decision(const pool_layout & layout, const std::size_t * elements,
                                        std::size_t count, std::size_t ranks, std::size_t budget,
                                        std::size_t & need) noexcept {
    if (layout_covers(layout, elements, count)) {
        need = layout.bytes;
        return pool_growth::keep;
    }
    if (!pool_need(elements, count, ranks, need)) {
        return pool_growth::fail_budget;
    }
    return need > budget ? pool_growth::fail_budget : pool_growth::grow;
}

// Environment switch matrix. Every switch defaults to the legacy behavior for
// plan cache, frames, pool and channel table. graph_uid is parsed and logged;
// the relay uid selection does not read it.
struct config {
    bool          plan_union    = false;
    std::size_t   frames        = 2;
    bool          pool_exact    = false;
    bool          slot_channels = false;
    bool          verify_uid    = false;
    bool          plan_stats    = false;
    bool          graph_uid     = false; // GGML_TP5_GRAPH_UID=1 assigns the llama parent uid; relay uid selection ignores it
    bool          plan_flush    = false;
};

inline bool parse_switch(const char * value, bool & out) noexcept {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
        out = false;
        return true;
    }
    if (std::strcmp(value, "1") == 0) {
        out = true;
        return true;
    }
    return false;
}

inline bool parse_config(const char * plan_union, const char * frames, const char * pool_exact,
                         const char * slot_channels, const char * verify_uid,
                         const char * plan_stats, const char * graph_uid,
                         const char * plan_flush, config & out,
                         char * error, std::size_t error_size) noexcept {
    if (error != nullptr && error_size != 0) {
        error[0] = '\0';
    }
    const auto reject = [&](const char * message) {
        if (error != nullptr && error_size != 0) {
            std::snprintf(error, error_size, "%s", message);
        }
        return false;
    };
    if (!parse_switch(plan_union, out.plan_union)) {
        return reject("GGML_TP5_PLAN_UNION must be 0 or 1");
    }
    if (frames == nullptr || frames[0] == '\0' || std::strcmp(frames, "0") == 0) {
        out.frames = 2;
    } else {
        char * end = nullptr;
        const long value = std::strtol(frames, &end, 10);
        if (end == frames || (end != nullptr && *end != '\0') ||
            value < 2 || value > static_cast<long>(frame_capacity_max)) {
            return reject("GGML_TP5_RELAY_FRAMES must be 0 or 2..8");
        }
        out.frames = static_cast<std::size_t>(value);
    }
    if (!parse_switch(pool_exact, out.pool_exact)) {
        return reject("GGML_TP5_POOL_EXACT must be 0 or 1");
    }
    if (slot_channels == nullptr || slot_channels[0] == '\0') {
        out.slot_channels = out.plan_union;
    } else if (!parse_switch(slot_channels, out.slot_channels)) {
        return reject("GGML_TP5_RELAY_SLOT_CHANNELS must be 0 or 1");
    }
    if (!parse_switch(verify_uid, out.verify_uid)) {
        return reject("GGML_TP5_RELAY_VERIFY_UID must be 0 or 1");
    }
    if (!parse_switch(plan_stats, out.plan_stats)) {
        return reject("GGML_TP5_RELAY_PLAN_STATS must be 0 or 1");
    }
    if (graph_uid == nullptr || graph_uid[0] == '\0') {
        out.graph_uid = false; // OFF does not force uid 0: sched/meta still assign uids
    } else if (!parse_switch(graph_uid, out.graph_uid)) {
        return reject("GGML_TP5_GRAPH_UID must be 0 or 1");
    }
    if (!parse_switch(plan_flush, out.plan_flush)) {
        return reject("GGML_TP5_RELAY_PLAN_FLUSH must be 0 or 1");
    }
    return true;
}

// Per-window channel ownership. Eight plan slots, each holding 256 channel
// states. A slot is bound to one plan generation; retirement clears it only
// after its readers ended. Two windows own independent tables, so the same
// generation number in both never shares state.
template <typename ChannelState>
class channel_slots {
public:
    static constexpr std::size_t slot_capacity     = plan_capacity;
    static constexpr std::size_t channel_capacity_ = channel_capacity;

    struct slot {
        std::uint64_t generation = 0; // 0 = empty
        std::uint64_t last_used  = 0;
        std::array<std::unique_ptr<ChannelState>, channel_capacity> states;
    };

    slot * find(std::uint64_t generation) noexcept {
        if (generation == 0) {
            return nullptr;
        }
        for (auto & candidate : slots_) {
            if (candidate.generation == generation) {
                return &candidate;
            }
        }
        return nullptr;
    }

    const slot * find(std::uint64_t generation) const noexcept {
        return const_cast<channel_slots *>(this)->find(generation);
    }

    // Bind a generation to a slot, reusing an existing binding. An empty slot
    // is preferred; otherwise the least recently used slot for which
    // `idle(slot)` is true is cleared and rebound. Returns null when every
    // slot still holds a live reader.
    template <typename IdleFn>
    slot * acquire(std::uint64_t generation, std::uint64_t epoch, IdleFn && idle) noexcept {
        if (generation == 0) {
            return nullptr;
        }
        if (slot * existing = find(generation)) {
            existing->last_used = epoch;
            return existing;
        }
        for (auto & candidate : slots_) {
            if (candidate.generation == 0) {
                candidate.generation = generation;
                candidate.last_used  = epoch;
                return &candidate;
            }
        }
        slot * victim = nullptr;
        for (auto & candidate : slots_) {
            if (!idle(candidate)) {
                continue;
            }
            if (victim == nullptr || candidate.last_used < victim->last_used) {
                victim = &candidate;
            }
        }
        if (victim == nullptr) {
            return nullptr;
        }
        victim->states = {};
        victim->generation = generation;
        victim->last_used  = epoch;
        return victim;
    }

    // Drop one generation's slot. The caller proves every held reader ended.
    bool retire(std::uint64_t generation) noexcept {
        slot * existing = find(generation);
        if (existing == nullptr) {
            return false;
        }
        *existing = {};
        return true;
    }

    std::size_t live() const noexcept {
        std::size_t count = 0;
        for (const auto & candidate : slots_) {
            count += candidate.generation != 0 ? 1 : 0;
        }
        return count;
    }

    std::size_t live_channels() const noexcept {
        std::size_t count = 0;
        for (const auto & candidate : slots_) {
            if (candidate.generation == 0) {
                continue;
            }
            for (const auto & owner : candidate.states) {
                count += owner ? 1 : 0;
            }
        }
        return count;
    }

    bool empty() const noexcept { return live() == 0; }

    void clear() noexcept {
        for (auto & candidate : slots_) {
            candidate = {};
        }
    }

    slot & at(std::size_t index) noexcept { return slots_[index]; }
    const slot & at(std::size_t index) const noexcept { return slots_[index]; }

private:
    std::array<slot, slot_capacity> slots_{};
};

} // namespace ggml_relay_plan
