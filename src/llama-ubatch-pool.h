#pragma once

// TP5 per-token host pooling: ubatch payload slots.
//
// A llama_ubatch owns its token/position/sequence arrays through this handle.
// Handles are intrusive reference counts over fixed pool slots: a slot returns
// to the pool when its last copy dies. A generation counter rejects a stale
// release instead of freeing the next occupant. When the pool is disabled,
// unconfigured, or exhausted, the handle owns a heap payload exactly like the
// old shared_ptr path.

#include "llama.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

struct llama_ubatch_data {
    std::vector<llama_token>    token;
    std::vector<float>          embd;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id *> seq_id;      // these point into seq_id_data below
    std::vector<llama_seq_id>   seq_id_unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         output;
    std::vector<int32_t>        decision_order;

    std::vector<llama_seq_id> seq_id_data;
};

class ubatch_data_pool;

class ubatch_data_handle {
public:
    ubatch_data_handle() = default;
    // Implicit on purpose: a caller that already builds a heap payload (the
    // dsv4 raw-write path, for example) keeps the old shared_ptr semantics.
    ubatch_data_handle(std::shared_ptr<llama_ubatch_data> data);
    ubatch_data_handle(const ubatch_data_handle & other);
    ubatch_data_handle & operator=(const ubatch_data_handle & other);
    ubatch_data_handle(ubatch_data_handle && other) noexcept;
    ubatch_data_handle & operator=(ubatch_data_handle && other) noexcept;
    ~ubatch_data_handle();

    llama_ubatch_data * get() const;
    llama_ubatch_data * operator->() const { return get(); }
    llama_ubatch_data & operator*() const { return *get(); }
    explicit operator bool() const { return get() != nullptr; }

    void reset();

    bool     pooled()     const { return pool_ != nullptr; }
    uint32_t slot()       const { return slot_; }
    uint32_t generation() const { return gen_; }

private:
    friend class ubatch_data_pool;

    ubatch_data_handle(ubatch_data_pool * pool, uint32_t slot, uint32_t gen);

    ubatch_data_pool * pool_ = nullptr;
    uint32_t slot_ = 0;
    uint32_t gen_  = 0;

    std::shared_ptr<llama_ubatch_data> fallback_;
};

class ubatch_data_pool {
public:
    struct stats_t {
        std::uint64_t acquired  = 0; // handles handed out from a slot
        std::uint64_t released  = 0; // handles that dropped the last reference
        std::uint64_t fallback  = 0; // heap payloads created by the caller
        std::uint64_t exhausted = 0; // acquire calls that found no free slot
        std::uint64_t stale     = 0; // rejected stale releases
    };

    ubatch_data_pool() = default;
    ~ubatch_data_pool();

    ubatch_data_pool(const ubatch_data_pool &)             = delete;
    ubatch_data_pool & operator=(const ubatch_data_pool &) = delete;

    // Fixed-capacity slots. The first successful call allocates every slot.
    // A later call may only grow the pool while no handle is live.
    bool configure(uint32_t capacity);

    bool     configured() const;
    uint32_t capacity()   const;

    // Empty handle when the pool is not configured or exhausted.
    ubatch_data_handle acquire();

    // Count one caller-side heap payload created after an empty acquire().
    void note_fallback();

    stats_t  stats() const;
    uint32_t in_use() const;

    // Test-only: exercise the stale-release rejection for a slot/generation
    // pair that no longer owns the slot. Returns true when the release was
    // rejected.
    bool debug_release_stale(uint32_t slot, uint32_t gen);

private:
    friend class ubatch_data_handle;

    struct slot_t {
        llama_ubatch_data data;
        uint32_t gen = 0;
        uint32_t refs = 0;
        bool in_use = false;
    };

    void retain(uint32_t slot);
    void release(uint32_t slot, uint32_t gen);
    llama_ubatch_data * object(uint32_t slot, uint32_t gen) const;

    mutable std::mutex mutex_;
    std::vector<slot_t> slots_;
    uint32_t in_use_ = 0;
    stats_t stats_;
};
