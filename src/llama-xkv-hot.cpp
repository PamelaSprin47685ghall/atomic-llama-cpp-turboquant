#include "llama-xkv-hot.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace llama_xkv {

const char * xkv_slot_state_to_str(xkv_slot_state state) {
    switch (state) {
        case xkv_slot_state::free:     return "free";
        case xkv_slot_state::reserved: return "reserved";
        case xkv_slot_state::bound:    return "bound";
        default:                       return "unknown";
    }
}

// ----------------------------------------------------------------------------
// xkv_hot_reservation
// ----------------------------------------------------------------------------

xkv_hot_reservation::xkv_hot_reservation(xkv_hot_slot_pool * pool, std::vector<uint32_t> slots) noexcept
    : pool_(pool), slots_(std::move(slots)), committed_(false) {}

xkv_hot_reservation::~xkv_hot_reservation() {
    rollback();
}

xkv_hot_reservation::xkv_hot_reservation(xkv_hot_reservation && other) noexcept
    : pool_(other.pool_), slots_(std::move(other.slots_)), committed_(other.committed_) {
    other.pool_ = nullptr;
    other.slots_.clear();
    other.committed_ = false;
}

xkv_hot_reservation & xkv_hot_reservation::operator=(xkv_hot_reservation && other) noexcept {
    if (this != &other) {
        rollback();
        pool_ = other.pool_;
        slots_ = std::move(other.slots_);
        committed_ = other.committed_;
        other.pool_ = nullptr;
        other.slots_.clear();
        other.committed_ = false;
    }
    return *this;
}

void xkv_hot_reservation::rollback() noexcept {
    if (pool_ && !committed_ && !slots_.empty()) {
        pool_->rollback(*this);
    }
    slots_.clear();
    pool_ = nullptr;
    committed_ = false;
}

bool xkv_hot_reservation::commit(const std::vector<uint64_t> & payload_ids,
                                 const std::vector<uint64_t> & generations,
                                 std::string * err) {
    if (!pool_) {
        if (err) *err = "xkv_hot_reservation::commit: reservation is invalid or empty";
        return false;
    }
    return pool_->commit(*this, payload_ids, generations, err);
}

bool xkv_hot_reservation::commit(const uint64_t * payload_ids,
                                 const uint64_t * generations,
                                 size_t count,
                                 std::string * err) {
    if (!pool_) {
        if (err) *err = "xkv_hot_reservation::commit: reservation is invalid or empty";
        return false;
    }
    return pool_->commit(*this, payload_ids, generations, count, err);
}

// ----------------------------------------------------------------------------
// xkv_hot_slot_pool: Helper hash table functions
// ----------------------------------------------------------------------------

static inline uint64_t hash_uint64(uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

const xkv_hot_slot_pool::hash_entry * xkv_hot_slot_pool::table_find(uint64_t key) const noexcept {
    if (key == 0 || table_.empty()) {
        return nullptr;
    }
    size_t idx = hash_uint64(key) & table_mask_;
    while (true) {
        if (table_[idx].key == key) {
            return &table_[idx];
        }
        if (table_[idx].key == 0) {
            return nullptr;
        }
        idx = (idx + 1) & table_mask_;
    }
}

xkv_hot_slot_pool::hash_entry * xkv_hot_slot_pool::table_find_mut(uint64_t key) noexcept {
    if (key == 0 || table_.empty()) {
        return nullptr;
    }
    size_t idx = hash_uint64(key) & table_mask_;
    while (true) {
        if (table_[idx].key == key) {
            return &table_[idx];
        }
        if (table_[idx].key == 0) {
            return nullptr;
        }
        idx = (idx + 1) & table_mask_;
    }
}

void xkv_hot_slot_pool::table_insert(uint64_t key, uint32_t val) noexcept {
    assert(key != 0);
    assert(!table_.empty());
    size_t idx = hash_uint64(key) & table_mask_;
    while (table_[idx].key != 0 && table_[idx].key != key) {
        idx = (idx + 1) & table_mask_;
    }
    table_[idx].key = key;
    table_[idx].val = val;
}

void xkv_hot_slot_pool::table_remove(uint64_t key) noexcept {
    if (key == 0 || table_.empty()) {
        return;
    }
    size_t idx = hash_uint64(key) & table_mask_;
    while (table_[idx].key != key) {
        if (table_[idx].key == 0) {
            return;
        }
        idx = (idx + 1) & table_mask_;
    }

    // Classic open addressing backward-shift deletion
    table_[idx].key = 0;
    table_[idx].val = 0;
    size_t curr = idx;
    size_t next = (curr + 1) & table_mask_;

    while (table_[next].key != 0) {
        size_t ideal = hash_uint64(table_[next].key) & table_mask_;
        // Check if ideal is cyclically strictly between curr and next
        // (i.e. next's element can be shifted into curr)
        bool can_shift = false;
        if (curr <= next) {
            can_shift = (ideal <= curr || ideal > next);
        } else {
            can_shift = (ideal <= curr && ideal > next);
        }
        if (can_shift) {
            table_[curr] = table_[next];
            table_[next].key = 0;
            table_[next].val = 0;
            curr = next;
        }
        next = (next + 1) & table_mask_;
    }
}

// ----------------------------------------------------------------------------
// xkv_hot_slot_pool: Implementation
// ----------------------------------------------------------------------------

xkv_hot_slot_pool::xkv_hot_slot_pool(uint32_t capacity)
    : capacity_(capacity) {
    if (capacity_ > 0) {
        slots_.resize(capacity_);
        for (uint32_t i = 0; i < capacity_; ++i) {
            slots_[i].slot = i;
            slots_[i].state = xkv_slot_state::free;
            slots_[i].payload_id = 0;
            slots_[i].storage_generation = 0;
        }

        // Min-heap of available slot indices:
        // By inserting in reverse and making heap with std::greater, lowest slot index is popped first
        free_heap_.resize(capacity_);
        for (uint32_t i = 0; i < capacity_; ++i) {
            free_heap_[i] = capacity_ - 1 - i;
        }
        std::make_heap(free_heap_.begin(), free_heap_.end(), std::greater<uint32_t>());

        // Preallocate open-addressing table with power-of-two size and max load factor <= 0.5
        size_t table_size = 16;
        while (table_size < (size_t) capacity_ * 2) {
            table_size <<= 1;
        }
        table_.resize(table_size, {0, 0});
        table_mask_ = table_size - 1;
    }
}

xkv_hot_reservation xkv_hot_slot_pool::reserve(uint32_t count, std::string * err) {
    if (count == 0) {
        // Zero reservation is empty but valid (no-op)
        return xkv_hot_reservation(this, {});
    }

    std::lock_guard<std::mutex> lock(mtx_);

    if (capacity_ == 0) {
        if (err) *err = "xkv_hot_slot_pool::reserve: pool capacity is 0";
        return xkv_hot_reservation();
    }

    // Atomic preflight: live = reserved + bound
    const uint64_t current_live = (uint64_t) reserved_count_ + bound_count_;
    if (current_live + count > capacity_ || free_heap_.size() < count) {
        if (err) {
            *err = "xkv_hot_slot_pool::reserve: insufficient capacity (requested " +
                   std::to_string(count) + ", available " +
                   std::to_string(capacity_ - current_live) + ", capacity " +
                   std::to_string(capacity_) + ")";
        }
        return xkv_hot_reservation();
    }

    // Allocate lowest-index available slots from the min-heap
    std::vector<uint32_t> allocated_slots;
    allocated_slots.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        std::pop_heap(free_heap_.begin(), free_heap_.end(), std::greater<uint32_t>());
        uint32_t slot = free_heap_.back();
        free_heap_.pop_back();

        assert(slot < capacity_);
        assert(slots_[slot].state == xkv_slot_state::free);

        slots_[slot].state = xkv_slot_state::reserved;
        slots_[slot].payload_id = 0;
        slots_[slot].storage_generation = 0;

        allocated_slots.push_back(slot);
    }

    reserved_count_ += count;
    const uint32_t new_live = reserved_count_ + bound_count_;
    if (new_live > peak_count_) {
        peak_count_ = new_live;
    }

    return xkv_hot_reservation(this, std::move(allocated_slots));
}

bool xkv_hot_slot_pool::commit(xkv_hot_reservation & res,
                              const std::vector<uint64_t> & payload_ids,
                              const std::vector<uint64_t> & generations,
                              std::string * err) {
    if (payload_ids.size() != generations.size()) {
        if (err) *err = "xkv_hot_slot_pool::commit: payload_ids and generations size mismatch";
        return false;
    }
    return commit(res, payload_ids.data(), generations.data(), payload_ids.size(), err);
}

bool xkv_hot_slot_pool::validate_commit_locked(const xkv_hot_reservation & res,
                                               const uint64_t * payload_ids,
                                               const uint64_t * generations,
                                               size_t count,
                                               std::string * err) const noexcept {
    // Note: std::string assignment may allocate, but validation performs
    // zero pool-state mutation, which is the property preflight relies on.
    try {
        if (res.pool_ != this || res.committed_) {
            if (err) *err = "xkv_hot_slot_pool::commit: reservation does not belong to this pool or already committed";
            return false;
        }
        if (res.slots_.size() != count) {
            if (err) {
                *err = "xkv_hot_slot_pool::commit: reservation size (" +
                       std::to_string(res.slots_.size()) + ") does not match payload count (" +
                       std::to_string(count) + ")";
            }
            return false;
        }
        if (count == 0) {
            return true;
        }
        if (!payload_ids || !generations) {
            if (err) *err = "xkv_hot_slot_pool::commit: null payload_ids or generations array";
            return false;
        }
        // 1. All payload IDs must be nonzero.
        // 2. All generations must be nonzero (0 is the free/reserved sentinel).
        // 3. All payload IDs in this commit must be distinct.
        // 4. No payload ID can already be bound in the pool.
        // 5. All slots in reservation must currently be in reserved state.
        for (size_t i = 0; i < count; ++i) {
            const uint64_t pid = payload_ids[i];
            if (pid == 0) {
                if (err) *err = "xkv_hot_slot_pool::commit: payload_id at index " + std::to_string(i) + " is 0";
                return false;
            }
            if (generations[i] == 0) {
                if (err) *err = "xkv_hot_slot_pool::commit: generation at index " + std::to_string(i) + " is 0";
                return false;
            }
            for (size_t j = 0; j < i; ++j) {
                if (payload_ids[j] == pid) {
                    if (err) *err = "xkv_hot_slot_pool::commit: duplicate payload_id " + std::to_string(pid) + " in commit batch";
                    return false;
                }
            }
            if (table_find(pid) != nullptr) {
                if (err) *err = "xkv_hot_slot_pool::commit: payload_id " + std::to_string(pid) + " is already bound in pool";
                return false;
            }
            const uint32_t slot = res.slots_[i];
            if (slot >= capacity_ || slots_[slot].state != xkv_slot_state::reserved) {
                if (err) *err = "xkv_hot_slot_pool::commit: slot " + std::to_string(slot) + " is not in reserved state";
                return false;
            }
        }
    } catch (...) {
        if (err) *err = "xkv_hot_slot_pool::commit: validation threw";
        return false;
    }
    return true;
}

bool xkv_hot_slot_pool::validate_release_locked(const uint64_t * payload_ids,
                                                const uint64_t * generations,
                                                const uint32_t * expected_slots,
                                                size_t count,
                                                std::string * err) const noexcept {
    try {
        for (size_t i = 0; i < count; ++i) {
            const uint64_t pid           = payload_ids[i];
            const uint64_t gen           = generations[i];
            const uint32_t expected_slot = expected_slots[i];
            if (pid == 0) {
                if (err) *err = "xkv_hot_slot_pool::release_batch: payload_id at index " + std::to_string(i) + " is 0";
                return false;
            }
            for (size_t j = 0; j < i; ++j) {
                if (payload_ids[j] == pid) {
                    if (err) *err = "xkv_hot_slot_pool::release_batch: duplicate payload_id " + std::to_string(pid) + " in release batch";
                    return false;
                }
                if (expected_slots[j] == expected_slot) {
                    if (err) *err = "xkv_hot_slot_pool::release_batch: duplicate expected_slot " + std::to_string(expected_slot) + " in release batch";
                    return false;
                }
            }
            if (expected_slot >= capacity_) {
                if (err) *err = "xkv_hot_slot_pool::release_batch: expected_slot " + std::to_string(expected_slot) + " out of range [0, " + std::to_string(capacity_) + ")";
                return false;
            }
            const auto & slot_info = slots_[expected_slot];
            if (slot_info.state != xkv_slot_state::bound) {
                if (err) {
                    *err = "xkv_hot_slot_pool::release_batch: slot " + std::to_string(expected_slot) +
                           " is not bound (state: " + xkv_slot_state_to_str(slot_info.state) + ")";
                }
                return false;
            }
            if (slot_info.payload_id != pid) {
                if (err) {
                    *err = "xkv_hot_slot_pool::release_batch: slot " + std::to_string(expected_slot) +
                           " payload_id mismatch (expected " + std::to_string(pid) + ", actual " +
                           std::to_string(slot_info.payload_id) + ")";
                }
                return false;
            }
            if (slot_info.storage_generation != gen) {
                if (err) {
                    *err = "xkv_hot_slot_pool::release_batch: slot " + std::to_string(expected_slot) +
                           " generation mismatch (expected " + std::to_string(gen) + ", actual " +
                           std::to_string(slot_info.storage_generation) + ")";
                }
                return false;
            }
            const hash_entry * entry = table_find(pid);
            if (!entry || entry->val != expected_slot) {
                if (err) {
                    *err = "xkv_hot_slot_pool::release_batch: lookup table mismatch for payload_id " +
                           std::to_string(pid);
                }
                return false;
            }
        }
    } catch (...) {
        if (err) *err = "xkv_hot_slot_pool::release_batch: validation threw";
        return false;
    }
    return true;
}

bool xkv_hot_slot_pool::can_commit(const xkv_hot_reservation & res,
                                   const uint64_t * payload_ids,
                                   const uint64_t * generations,
                                   size_t count,
                                   std::string * err) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return validate_commit_locked(res, payload_ids, generations, count, err);
}

bool xkv_hot_slot_pool::can_release_batch(const uint64_t * payload_ids,
                                          const uint64_t * generations,
                                          const uint32_t * expected_slots,
                                          size_t count,
                                          std::string * err) const {
    if (count == 0) {
        return true;
    }
    if (!payload_ids || !generations || !expected_slots) {
        if (err) *err = "xkv_hot_slot_pool::release_batch: null input array";
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    return validate_release_locked(payload_ids, generations, expected_slots, count, err);
}

bool xkv_hot_slot_pool::has_active_reservations() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return reserved_count_ > 0;
}

xkv_hot_slot_pool::pool_snapshot xkv_hot_slot_pool::snapshot_state() const {
    std::lock_guard<std::mutex> lock(mtx_);
    pool_snapshot snap;
    snap.slots = slots_;
    snap.free_heap = free_heap_;
    snap.table_keys.reserve(table_.size());
    snap.table_vals.reserve(table_.size());
    for (const auto & e : table_) {
        snap.table_keys.push_back(e.key);
        snap.table_vals.push_back(e.val);
    }
    snap.table_mask = table_mask_;
    snap.capacity = capacity_;
    snap.reserved_count = reserved_count_;
    snap.bound_count = bound_count_;
    snap.peak_count = peak_count_;
    return snap;
}

void xkv_hot_slot_pool::restore_state(const pool_snapshot & snap) noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    // Geometry is fixed per pool; a snapshot always comes from snapshot_state
    // on this same pool. On geometry mismatch (foreign snapshot: programming
    // error) refuse rather than corrupt; noexcept forbids reporting.
    if (snap.capacity != capacity_ || snap.slots.size() != slots_.size() ||
        snap.free_heap.size() + snap.reserved_count + snap.bound_count != snap.capacity ||
        snap.table_keys.size() != table_.size()) {
        return;
    }
    // Element-wise copies only: no allocation, cannot fail.
    for (size_t i = 0; i < slots_.size(); ++i) {
        slots_[i] = snap.slots[i];
    }
    free_heap_.clear();
    for (uint32_t s : snap.free_heap) {
        free_heap_.push_back(s);
    }
    for (auto & e : table_) {
        e.key = 0;
        e.val = 0;
    }
    table_mask_ = snap.table_mask;
    for (size_t i = 0; i < snap.table_keys.size(); ++i) {
        table_[i].key = snap.table_keys[i];
        table_[i].val = snap.table_vals[i];
    }
    reserved_count_ = snap.reserved_count;
    bound_count_ = snap.bound_count;
    peak_count_ = snap.peak_count;
}

bool xkv_hot_slot_pool::commit(xkv_hot_reservation & res,
                              const uint64_t * payload_ids,
                              const uint64_t * generations,
                              size_t count,
                              std::string * err) {
    // Empty commits bind nothing but still consume the reservation.
    if (count == 0 && res.pool_ == this && !res.committed_ && res.slots_.empty()) {
        res.committed_ = true;
        return true;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    // Single-lock validate-then-mutate: acceptance is decided solely by
    // validate_commit_locked, so preflight (can_commit) and commit agree.
    if (!validate_commit_locked(res, payload_ids, generations, count, err)) {
        return false;
    }


    // Phase 2: Apply mutations.
    // Transition: reserved -> bound.
    // Guaranteed no dynamic allocation or failure from this point forward.
    for (size_t i = 0; i < count; ++i) {
        const uint32_t slot = res.slots_[i];
        const uint64_t pid  = payload_ids[i];
        const uint64_t gen  = generations[i];

        slots_[slot].state              = xkv_slot_state::bound;
        slots_[slot].payload_id         = pid;
        slots_[slot].storage_generation = gen;

        table_insert(pid, slot);
    }

    assert(reserved_count_ >= count);
    reserved_count_ -= static_cast<uint32_t>(count);
    bound_count_    += static_cast<uint32_t>(count);

    res.committed_ = true;
    return true;
}

void xkv_hot_slot_pool::rollback(xkv_hot_reservation & res) noexcept {
    if (res.pool_ != this || res.committed_ || res.slots_.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mtx_);
        rollback_slots_locked(res.slots_);
    }
    res.slots_.clear();
    res.committed_ = false;
    res.pool_ = nullptr;
}

void xkv_hot_slot_pool::rollback_slots_locked(const std::vector<uint32_t> & slots) noexcept {
    for (uint32_t slot : slots) {
        if (slot < capacity_ && slots_[slot].state == xkv_slot_state::reserved) {
            slots_[slot].state = xkv_slot_state::free;
            slots_[slot].payload_id = 0;
            slots_[slot].storage_generation = 0;

            free_heap_.push_back(slot);
            std::push_heap(free_heap_.begin(), free_heap_.end(), std::greater<uint32_t>());

            assert(reserved_count_ > 0);
            reserved_count_--;
        }
    }
}

bool xkv_hot_slot_pool::release(uint64_t payload_id,
                               uint64_t generation,
                               uint32_t expected_slot,
                               std::string * err) {
    xkv_hot_release_item item = {payload_id, generation, expected_slot};
    return release_batch(&item.payload_id, &item.generation, &item.expected_slot, 1, err);
}

bool xkv_hot_slot_pool::release_batch(const std::vector<xkv_hot_release_item> & items,
                                     std::string * err) {
    if (items.empty()) {
        return true;
    }
    std::vector<uint64_t> pids;
    std::vector<uint64_t> gens;
    std::vector<uint32_t> slots;
    pids.reserve(items.size());
    gens.reserve(items.size());
    slots.reserve(items.size());

    for (const auto & item : items) {
        pids.push_back(item.payload_id);
        gens.push_back(item.generation);
        slots.push_back(item.expected_slot);
    }
    return release_batch(pids.data(), gens.data(), slots.data(), items.size(), err);
}

bool xkv_hot_slot_pool::release_batch(const std::vector<uint64_t> & payload_ids,
                                     const std::vector<uint64_t> & generations,
                                     const std::vector<uint32_t> & expected_slots,
                                     std::string * err) {
    if (payload_ids.size() != generations.size() || payload_ids.size() != expected_slots.size()) {
        if (err) *err = "xkv_hot_slot_pool::release_batch: vector size mismatch";
        return false;
    }
    return release_batch(payload_ids.data(), generations.data(), expected_slots.data(), payload_ids.size(), err);
}

bool xkv_hot_slot_pool::release_batch(const uint64_t * payload_ids,
                                     const uint64_t * generations,
                                     const uint32_t * expected_slots,
                                     size_t count,
                                     std::string * err) {
    if (count == 0) {
        return true;
    }

    if (!payload_ids || !generations || !expected_slots) {
        if (err) *err = "xkv_hot_slot_pool::release_batch: null input array";
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx_);

    // Phase 1: All-or-nothing preflight validation via the shared validator.
    // Any violation causes clean failure without any state changes.
    if (!validate_release_locked(payload_ids, generations, expected_slots, count, err)) {
        return false;
    }

    // Phase 2: Execute release mutation.
    // Zero possibility of failure or reallocation.
    for (size_t i = 0; i < count; ++i) {
        const uint64_t pid           = payload_ids[i];
        const uint32_t expected_slot = expected_slots[i];

        slots_[expected_slot].state              = xkv_slot_state::free;
        slots_[expected_slot].payload_id         = 0;
        slots_[expected_slot].storage_generation = 0;

        table_remove(pid);

        free_heap_.push_back(expected_slot);
        std::push_heap(free_heap_.begin(), free_heap_.end(), std::greater<uint32_t>());
    }

    assert(bound_count_ >= count);
    bound_count_ -= static_cast<uint32_t>(count);

    return true;
}

bool xkv_hot_slot_pool::reset(std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (reserved_count_ > 0) {
        if (err) {
            *err = "xkv_hot_slot_pool::reset: active live reservations exist (" +
                   std::to_string(reserved_count_) + " slots reserved); cache not quiescent";
        }
        return false;
    }

    // Clear all slot metadata in-place
    for (uint32_t i = 0; i < capacity_; ++i) {
        slots_[i].state = xkv_slot_state::free;
        slots_[i].payload_id = 0;
        slots_[i].storage_generation = 0;
    }

    // Rebuild free heap in-place with retained capacity and zero allocations
    free_heap_.resize(capacity_);
    for (uint32_t i = 0; i < capacity_; ++i) {
        free_heap_[i] = capacity_ - 1 - i;
    }
    std::make_heap(free_heap_.begin(), free_heap_.end(), std::greater<uint32_t>());

    // Clear open-addressing lookup table in-place
    if (!table_.empty()) {
        std::fill(table_.begin(), table_.end(), hash_entry{0, 0});
    }

    // Reset counts
    reserved_count_ = 0;
    bound_count_    = 0;
    // Peak policy: reset peak to 0 on explicit whole-pool lifecycle reset
    peak_count_     = 0;

    return true;
}

bool xkv_hot_slot_pool::find_slot(uint64_t payload_id, uint32_t & out_slot) const {
    if (payload_id == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    const hash_entry * entry = table_find(payload_id);
    if (!entry) {
        return false;
    }
    out_slot = entry->val;
    return true;
}

bool xkv_hot_slot_pool::find_payload(uint64_t payload_id, xkv_hot_slot_info & out_info) const {
    if (payload_id == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    const hash_entry * entry = table_find(payload_id);
    if (!entry) {
        return false;
    }
    const uint32_t slot = entry->val;
    assert(slot < capacity_);
    out_info = slots_[slot];
    return true;
}

bool xkv_hot_slot_pool::get_slot_info(uint32_t slot, xkv_hot_slot_info & out_info) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (slot >= capacity_) {
        return false;
    }
    out_info = slots_[slot];
    return true;
}

bool xkv_hot_slot_pool::has_payload(uint64_t payload_id) const {
    if (payload_id == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    return table_find(payload_id) != nullptr;
}

xkv_hot_accounting xkv_hot_slot_pool::get_accounting() const {
    std::lock_guard<std::mutex> lock(mtx_);
    xkv_hot_accounting acc;
    acc.capacity = capacity_;
    acc.reserved = reserved_count_;
    acc.bound    = bound_count_;
    acc.live     = reserved_count_ + bound_count_;
    acc.free     = (capacity_ >= acc.live) ? (capacity_ - acc.live) : 0;
    acc.peak     = peak_count_;
    return acc;
}

uint32_t xkv_hot_slot_pool::get_capacity() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return capacity_;
}

uint32_t xkv_hot_slot_pool::get_free() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint32_t live = reserved_count_ + bound_count_;
    return (capacity_ >= live) ? (capacity_ - live) : 0;
}

uint32_t xkv_hot_slot_pool::get_reserved() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return reserved_count_;
}

uint32_t xkv_hot_slot_pool::get_bound() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return bound_count_;
}

uint32_t xkv_hot_slot_pool::get_live() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return reserved_count_ + bound_count_;
}

uint32_t xkv_hot_slot_pool::get_peak() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return peak_count_;
}

void xkv_hot_slot_pool::reset_peak() {
    std::lock_guard<std::mutex> lock(mtx_);
    peak_count_ = reserved_count_ + bound_count_;
}

} // namespace llama_xkv
