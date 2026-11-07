#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tq/alloc/arena.h"

namespace tq::alloc {

// Identity of one allocation plan. The footprint is a property of the plan,
// not part of its identity. n_tokens is 1..4. n_kv_bucket indexes one of 256
// bucket cells.
struct PlanKey {
    std::uint64_t model = 0;
    std::uint32_t window = 0;
    std::uint8_t n_tokens = 1;
    std::uint16_t n_outputs = 0;
    std::int8_t rs_class = 0;
    std::uint32_t n_kv_bucket = 0;
    std::uint64_t alloc_gen = 0;

    constexpr bool operator==(const PlanKey & other) const noexcept {
        return model == other.model && window == other.window &&
               n_tokens == other.n_tokens && n_outputs == other.n_outputs &&
               rs_class == other.rs_class &&
               n_kv_bucket == other.n_kv_bucket &&
               alloc_gen == other.alloc_gen;
    }
    constexpr bool operator!=(const PlanKey & other) const noexcept {
        return !(*this == other);
    }
    constexpr bool valid() const noexcept {
        return n_tokens >= 1 && n_tokens <= 4 && n_kv_bucket < 256;
    }

    // Fixed text for logs. No allocation. The caller may copy the array.
    std::array<char, 192> describe() const noexcept;
};

// Cached plan value. The cache owns its copy. The uid fields are opaque
// handles. They are not pointers and no A type is involved.
struct PlanEntry {
    PlanKey key;
    std::uint64_t footprint = 0;
    std::uint32_t slots = 0;
    std::uint32_t reserved = 0;
    std::uint64_t generation = 0;
    std::uint64_t graph_uid = 0;
    std::uint64_t sched_uid = 0;
    std::uint64_t meta_uid = 0;
    std::uint64_t relay_uid = 0;
    std::uint64_t arena_gen = 0;
};

struct PlanStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t inserts = 0;
    std::uint64_t replaces = 0;
    std::uint64_t evictions = 0;
    std::uint64_t flushes = 0;
    std::uint64_t pin_rejects = 0;
};

// Fixed-capacity LRU cache. Capacity is 8. The cache never allocates.
// find touches an entry. insert of an existing key replaces the value and
// keeps its pin. A full cache evicts the least recently used unpinned entry.
class PlanCache {
public:
    static constexpr std::size_t capacity = 8;

    PlanCache() noexcept = default;

    // Return the entry and mark it most recently used, or null on a miss.
    const PlanEntry * find(const PlanKey & key) noexcept;

    // Insert or replace. Return invalid_argument for an invalid key. Return
    // full when every entry is pinned and no slot is free. A replacement
    // does not change size or pin state.
    Result insert(const PlanEntry & entry) noexcept;

    // Set the pin flag of an existing key. Return missing when absent.
    Result pin(const PlanKey & key, bool pinned) noexcept;

    Result erase(const PlanKey & key) noexcept;

    void flush_all() noexcept;

    std::size_t size() const noexcept { return size_; }
    PlanStats stats() const noexcept { return stats_; }

    // Immutable view of a stored entry, or null for an unused position.
    const PlanEntry * at(std::size_t index) const noexcept {
        return index < capacity && nodes_[index].used ? &nodes_[index].entry
                                                      : nullptr;
    }

private:
    struct Node {
        PlanEntry entry;
        std::uint64_t tick = 0;
        bool pinned = false;
        bool used = false;
    };

    Node nodes_[capacity];
    std::size_t size_ = 0;
    std::uint64_t clock_ = 0;
    PlanStats stats_;
};

} // namespace tq::alloc
