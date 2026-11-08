#pragma once

// TP5 llama-side per-shape plan cache policy. Header-only, no model/backend
// dependency, so the CPU test pins the same decisions the context uses.
//
// One slot names one graph shape: the ubatch dimensions, the output width,
// the KV target width, the graph type and the allocator generation. A hit
// lets the caller reuse that slot's llm_graph_result and scheduler instead of
// rebuilding the graph. can_reuse() remains the second-level check: a key
// collision only selects a candidate, it never proves reuse.
//
// The key is deliberately coarse. It must be stable across epochs that can
// share a graph, and it must change when the placement generation changes
// (alloc_gen), because a scheduler allocated in an older generation must not
// back a graph in a newer one.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace llama_shape_cache {

struct key {
    uint32_t n_tokens     = 0;
    uint32_t n_seq_tokens = 0;
    uint32_t n_outputs    = 0;
    uint32_t n_seqs       = 0;
    uint32_t n_seqs_unq   = 0;
    uint32_t equal_seqs   = 0;
    uint32_t n_kv_target = 0; // padded KV width the graph executes at, 0 if none
    uint32_t gtype       = 0;
    uint64_t alloc_gen   = 0;

    bool operator==(const key & other) const noexcept {
        return n_tokens == other.n_tokens && n_seq_tokens == other.n_seq_tokens &&
               n_outputs == other.n_outputs && n_seqs == other.n_seqs &&
               n_seqs_unq == other.n_seqs_unq && equal_seqs == other.equal_seqs &&
               n_kv_target == other.n_kv_target && gtype == other.gtype &&
               alloc_gen == other.alloc_gen;
    }
    bool operator!=(const key & other) const noexcept { return !(*this == other); }
};

struct stats {
    uint64_t hits      = 0;
    uint64_t misses    = 0;
    uint64_t inserts   = 0;
    uint64_t evictions = 0;
    uint64_t flushes   = 0;
    uint64_t rekeys    = 0;
};

// GGML_TP5_LLAMA_PLAN_CACHE: unset/0 = off, 1..capacity_max = number of shape
// slots. The value is the capacity; there is no hidden default tier.
constexpr std::size_t capacity_max = 8;

inline std::size_t parse_capacity(const char * value) noexcept {
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || end == nullptr || *end != '\0') {
        return 0;
    }
    if (parsed < 1 || parsed > static_cast<long>(capacity_max)) {
        return 0;
    }
    return static_cast<std::size_t>(parsed);
}

// Fixed-capacity LRU over shape keys. The slots own no graph state; the
// caller keeps a parallel array of entries and frees the entry named by
// `evicted` before installing a new one.
class cache {
public:
    struct result {
        int  index   = -1; // slot to use, -1 when the cache is off
        bool hit     = false;
        int  evicted = -1; // slot whose old entry must be freed, -1 if none
    };

    cache() noexcept = default;
    explicit cache(std::size_t capacity) noexcept : capacity_(capacity) {
        if (capacity_ > capacity_max) {
            capacity_ = capacity_max;
        }
    }

    std::size_t capacity() const noexcept { return capacity_; }
    bool enabled() const noexcept { return capacity_ != 0; }

    // Hit: refresh the slot and return it. Miss: take an empty slot, or the
    // least recently used one (its entry is reported as evicted). A cache
    // with zero capacity returns index -1 so the caller keeps the legacy
    // single-scheduler path.
    result acquire(const key & k, uint64_t epoch) noexcept {
        result out;
        if (capacity_ == 0) {
            return out;
        }
        for (std::size_t i = 0; i < capacity_; ++i) {
            slot & candidate = slots_[i];
            if (candidate.valid && candidate.k == k) {
                candidate.last_used = epoch;
                out.index = static_cast<int>(i);
                out.hit   = true;
                ++stats_.hits;
                return out;
            }
        }
        ++stats_.misses;
        int chosen = -1;
        for (std::size_t i = 0; i < capacity_; ++i) {
            if (!slots_[i].valid) {
                chosen = static_cast<int>(i);
                break;
            }
        }
        if (chosen < 0) {
            for (std::size_t i = 0; i < capacity_; ++i) {
                if (chosen < 0 || slots_[i].last_used < slots_[static_cast<std::size_t>(chosen)].last_used) {
                    chosen = static_cast<int>(i);
                }
            }
            out.evicted = chosen;
            ++stats_.evictions;
        }
        slot & target = slots_[static_cast<std::size_t>(chosen)];
        target.valid     = true;
        target.k         = k;
        target.last_used = epoch;
        out.index        = chosen;
        ++stats_.inserts;
        return out;
    }

    // The selected entry was rebuilt for a different shape (can_reuse failed
    // on a key collision). Keep the slot, replace its identity.
    void rekey(int index, const key & k, uint64_t epoch) noexcept {
        if (index < 0 || static_cast<std::size_t>(index) >= capacity_) {
            return;
        }
        slot & target    = slots_[static_cast<std::size_t>(index)];
        target.valid     = true;
        target.k         = k;
        target.last_used = epoch;
        ++stats_.rekeys;
    }

    // Drop one slot. The caller still owns and frees the entry.
    void invalidate(int index) noexcept {
        if (index >= 0 && static_cast<std::size_t>(index) < capacity_) {
            slots_[static_cast<std::size_t>(index)] = {};
        }
    }

    // Drop every slot. The caller must free the entries separately.
    void flush() noexcept {
        for (auto & candidate : slots_) {
            candidate = {};
        }
        ++stats_.flushes;
    }

    std::size_t live() const noexcept {
        std::size_t count = 0;
        for (std::size_t i = 0; i < capacity_; ++i) {
            count += slots_[i].valid ? 1 : 0;
        }
        return count;
    }

    const key * at(std::size_t index) const noexcept {
        return index < capacity_ && slots_[index].valid ? &slots_[index].k : nullptr;
    }

    const stats & get_stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = {}; }

private:
    struct slot {
        bool     valid     = false;
        key      k         = {};
        uint64_t last_used = 0;
    };

    std::array<slot, capacity_max> slots_{};
    std::size_t                    capacity_ = 0;
    stats                          stats_{};
};

} // namespace llama_shape_cache
