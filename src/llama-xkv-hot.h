#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llama_xkv {

// Lifecycle state of a physical hot slot
enum class xkv_slot_state : uint8_t {
    free     = 0,
    reserved = 1,
    bound    = 2,
};

const char * xkv_slot_state_to_str(xkv_slot_state state);

// Accounting snapshot for the hot slot pool
struct xkv_hot_accounting {
    uint32_t capacity = 0;
    uint32_t free     = 0;
    uint32_t reserved = 0;
    uint32_t bound    = 0;
    uint32_t live     = 0; // reserved + bound (active physical allocations)
    uint32_t peak     = 0; // High-water mark of live slots
};

// Information about a single physical slot
struct xkv_hot_slot_info {
    uint32_t       slot               = 0;
    xkv_slot_state state              = xkv_slot_state::free;
    uint64_t       payload_id         = 0;
    uint64_t       storage_generation = 0;
};

// Item for batch release validation and execution
struct xkv_hot_release_item {
    uint64_t payload_id    = 0;
    uint64_t generation    = 0;
    uint32_t expected_slot = 0;

    bool operator==(const xkv_hot_release_item & o) const noexcept {
        return payload_id == o.payload_id &&
               generation == o.generation &&
               expected_slot == o.expected_slot;
    }
};

class xkv_hot_slot_pool;

// Move-only RAII handle representing reserved physical slots in the pool.
// If destroyed without a successful commit, automatically rolls back
// reserved slots to the free pool.
class xkv_hot_reservation {
public:
    xkv_hot_reservation() noexcept = default;
    ~xkv_hot_reservation();

    xkv_hot_reservation(const xkv_hot_reservation &) = delete;
    xkv_hot_reservation & operator=(const xkv_hot_reservation &) = delete;

    xkv_hot_reservation(xkv_hot_reservation && other) noexcept;
    xkv_hot_reservation & operator=(xkv_hot_reservation && other) noexcept;

    // Reservation validity and slot inspection
    bool valid() const noexcept { return pool_ != nullptr && !slots_.empty(); }
    explicit operator bool() const noexcept { return valid(); }
    bool empty() const noexcept { return slots_.empty(); }
    size_t size() const noexcept { return slots_.size(); }
    const std::vector<uint32_t> & slots() const noexcept { return slots_; }
    uint32_t operator[](size_t idx) const { return slots_[idx]; }

    // Commit binds exact nonzero payload IDs and generations.
    // Transition: reserved -> bound.
    // Guaranteed: no allocations after mutation begins.
    bool commit(const std::vector<uint64_t> & payload_ids,
                const std::vector<uint64_t> & generations,
                std::string * err = nullptr);

    bool commit(const uint64_t * payload_ids,
                const uint64_t * generations,
                size_t count,
                std::string * err = nullptr);

    // Rollback uncommitted reserved slots back to free state immediately.
    void rollback() noexcept;

    bool is_committed() const noexcept { return committed_; }

private:
    friend class xkv_hot_slot_pool;
    xkv_hot_reservation(xkv_hot_slot_pool * pool, std::vector<uint32_t> slots) noexcept;

    xkv_hot_slot_pool *   pool_      = nullptr;
    std::vector<uint32_t> slots_;
    bool                  committed_ = false;
};

// Bounded physical hot-slot pool with transactional reservations and
// strict all-or-nothing binding and release.
class xkv_hot_slot_pool {
public:
    explicit xkv_hot_slot_pool(uint32_t capacity = 0);
    ~xkv_hot_slot_pool() = default;

    xkv_hot_slot_pool(const xkv_hot_slot_pool &) = delete;
    xkv_hot_slot_pool & operator=(const xkv_hot_slot_pool &) = delete;

    // Reserve N slots atomically or none.
    // Returns move-only RAII reservation handle.
    xkv_hot_reservation reserve(uint32_t count, std::string * err = nullptr);

    // Commit a reservation (can be called via res.commit() or directly on pool)
    bool commit(xkv_hot_reservation & res,
                const std::vector<uint64_t> & payload_ids,
                const std::vector<uint64_t> & generations,
                std::string * err = nullptr);

    bool commit(xkv_hot_reservation & res,
                const uint64_t * payload_ids,
                const uint64_t * generations,
                size_t count,
                std::string * err = nullptr);

    // Rollback a reservation
    void rollback(xkv_hot_reservation & res) noexcept;

    // Read-only preflight: returns true when a commit with these payload IDs and
    // generations would succeed against the current pool state. Performs zero
    // mutation; the cache uses it to validate before mutating any subsystem.
    bool can_commit(const xkv_hot_reservation & res,
                    const uint64_t * payload_ids,
                    const uint64_t * generations,
                    size_t count,
                    std::string * err = nullptr) const;

    // Read-only preflight for batch release with identical acceptance to
    // release_batch but zero mutation.
    bool can_release_batch(const uint64_t * payload_ids,
                            const uint64_t * generations,
                            const uint32_t * expected_slots,
                            size_t count,
                            std::string * err = nullptr) const;

    // True while uncommitted reservations are outstanding. Cache clear
    // preflights this before mutating any subsystem so a rejected clear
    // leaves store, pool, and cells untouched.
    bool has_active_reservations() const;

    // Full pool state snapshot for clear-atomicity. snapshot_state copies
    // (may throw OOM: take it during preflight, before any mutation);
    // restore_state rebuilds the table through the noexcept insert path so
    // a failed clear commit can put the pool back bit-identically.
    struct pool_snapshot {
        std::vector<xkv_hot_slot_info> slots;
        std::vector<uint32_t> free_heap;
        std::vector<uint64_t> table_keys;
        std::vector<uint32_t> table_vals;
        size_t table_mask = 0;
        uint32_t capacity = 0;
        uint32_t reserved_count = 0;
        uint32_t bound_count = 0;
        uint32_t peak_count = 0;
    };
    pool_snapshot snapshot_state() const;
    void restore_state(const pool_snapshot & snap) noexcept;

    // Release a single bound slot. Validates payload_id, generation, and expected_slot.
    bool release(uint64_t payload_id,
                 uint64_t generation,
                 uint32_t expected_slot,
                 std::string * err = nullptr);

    // Batch release with all-or-nothing semantics: if any item fails validation,
    // zero mutations are applied and false is returned.
    bool release_batch(const std::vector<xkv_hot_release_item> & items,
                       std::string * err = nullptr);

    bool release_batch(const std::vector<uint64_t> & payload_ids,
                       const std::vector<uint64_t> & generations,
                       const std::vector<uint32_t> & expected_slots,
                       std::string * err = nullptr);

    bool release_batch(const uint64_t * payload_ids,
                       const uint64_t * generations,
                       const uint32_t * expected_slots,
                       size_t count,
                       std::string * err = nullptr);

    // Reset the pool preserving object identity:
    // Fails (without mutation) if there are any active reservations (reserved_count_ > 0).
    // Cache clear is responsible for quiescence and must treat rejection as an invariant failure.
    // Preventing reset while reservations are outstanding protects against ABA rollback corruption.
    // Under lock:
    // - Clears all bound/reserved slot records and hash table entries
    // - Rebuilds free heap in-place with retained capacity and zero allocations
    // - Resets live counts (reserved = 0, bound = 0)
    // - Peak policy: peak is reset to 0 (documented clean lifecycle reset)
    bool reset(std::string * err = nullptr);

    // Lookups
    bool find_slot(uint64_t payload_id, uint32_t & out_slot) const;
    bool find_payload(uint64_t payload_id, xkv_hot_slot_info & out_info) const;
    bool get_slot_info(uint32_t slot, xkv_hot_slot_info & out_info) const;
    bool has_payload(uint64_t payload_id) const;

    // Accounting
    xkv_hot_accounting get_accounting() const;
    uint32_t get_capacity() const;
    uint32_t get_free() const;
    uint32_t get_reserved() const;
    uint32_t get_bound() const;
    uint32_t get_live() const;
    uint32_t get_peak() const;
    void reset_peak();

private:
    friend class xkv_hot_reservation;

    // Locked validators: single source of truth for commit/release acceptance.
    // Called with mtx_ held; perform zero mutation and zero allocation.
    bool validate_commit_locked(const xkv_hot_reservation & res,
                                const uint64_t * payload_ids,
                                const uint64_t * generations,
                                size_t count,
                                std::string * err) const noexcept;
    bool validate_release_locked(const uint64_t * payload_ids,
                                 const uint64_t * generations,
                                 const uint32_t * expected_slots,
                                 size_t count,
                                 std::string * err) const noexcept;

    // Internal rollback called by reservation destructor / rollback
    void rollback_slots_locked(const std::vector<uint32_t> & slots) noexcept;

    // Internal open-addressing hash table for payload_id -> slot index mapping
    struct hash_entry {
        uint64_t key = 0; // 0 indicates empty
        uint32_t val = 0;
    };

    const hash_entry * table_find(uint64_t key) const noexcept;
    hash_entry * table_find_mut(uint64_t key) noexcept;
    void table_insert(uint64_t key, uint32_t val) noexcept;
    void table_remove(uint64_t key) noexcept;

    mutable std::mutex mtx_;

    uint32_t capacity_       = 0;
    uint32_t reserved_count_ = 0;
    uint32_t bound_count_    = 0;
    uint32_t peak_count_     = 0;

    // Per-slot status
    std::vector<xkv_hot_slot_info> slots_;

    // Min-heap for deterministic lowest-index free slot allocation without dynamic allocation
    std::vector<uint32_t> free_heap_;

    // Pre-allocated flat hash table for O(1) lookups with zero heap allocations during commit/release
    std::vector<hash_entry> table_;
    size_t table_mask_ = 0;
};

} // namespace llama_xkv
