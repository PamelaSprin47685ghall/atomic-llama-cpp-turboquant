#pragma once

#include <cstddef>
#include <cstdint>

#include "tq/alloc/arena.h"

namespace tq::alloc {

enum class DomainState : std::uint8_t {
    idle = 0,
    active = 1,
    draining = 2,
    quarantined = 3,
};

// Lease lifecycle over one arena. The domain owns the pending drain queue and
// the reported state. It does not place data and does not allocate.
class Domain {
public:
    static constexpr std::size_t max_pending = 16;

    explicit Domain(Arena & arena) noexcept : arena_(&arena) {}

    // Drain pending slots first, then acquire. A successful acquire makes the
    // domain active.
    Result lease(Lifetime lifetime, ReuseClass reuse, Extent extent,
                 Lease & out) noexcept;
    Result lease(Lifetime lifetime, ReuseClass reuse, std::size_t bytes,
                 Lease & out) noexcept;

    // Acquire only when the domain is not draining.
    Result try_lease(Lifetime lifetime, ReuseClass reuse, Extent extent,
                     Lease & out) noexcept;
    Result try_lease(Lifetime lifetime, ReuseClass reuse, std::size_t bytes,
                     Lease & out) noexcept;

    // Return a live frame or staging lease and queue its slot for drain.
    Result cancel(const Lease & lease) noexcept;

    // Return a live lease. Frame and staging slots queue for drain. A cold
    // slot becomes free at once.
    Result release(const Lease & lease) noexcept;

    // Drain every queued slot. Idempotent when the queue is empty.
    Result drain() noexcept;

    // Return a live lease and quarantine its slot. One cold reserve unit
    // becomes available.
    Result quarantine(const Lease & lease) noexcept;

    // Drain pending slots, then advance the arena window. Returns busy when a
    // live lease still holds the arena.
    Result advance_window(std::uint32_t next) noexcept;

    DomainState state() const noexcept { return state_; }
    std::uint64_t generation() const noexcept { return arena_->generation(); }
    std::size_t pending_drains() const noexcept { return pending_count_; }
    std::uint32_t cold_reserve() const noexcept { return arena_->cold_reserve(); }
    std::uint64_t high_water() const noexcept { return arena_->high_water(); }
    ArenaStats stats() const noexcept { return arena_->stats(); }
    LifetimeStats stats(Lifetime lifetime) const noexcept {
        return arena_->stats(lifetime);
    }

    Arena & arena() noexcept { return *arena_; }
    const Arena & arena() const noexcept { return *arena_; }

private:
    Result enqueue(std::uint32_t slot) noexcept;
    void settle() noexcept;

    Arena * arena_;
    std::uint32_t pending_[max_pending] = {};
    std::size_t pending_count_ = 0;
    DomainState state_ = DomainState::idle;
};

} // namespace tq::alloc
