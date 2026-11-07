#include "tq/alloc/arena.h"

namespace tq::alloc {
namespace {

constexpr bool valid_class(Class cls) noexcept {
    return cls == Class::frame || cls == Class::staging || cls == Class::cold;
}

constexpr bool deferred(Class cls) noexcept {
    return cls == Class::frame || cls == Class::staging;
}

} // namespace

Arena::Arena(Slot * storage, std::size_t capacity, std::uint32_t id,
             std::uint32_t window, std::uint64_t footprint_limit) noexcept
    : storage_(storage),
      capacity_(storage == nullptr ? 0 : capacity),
      id_(id),
      window_(window),
      footprint_limit_(footprint_limit) {
    for (std::size_t i = 0; i < capacity_; ++i) {
        storage_[i] = Slot{};
    }
}

Result Arena::acquire(Class cls, Extent extent, Lease & out) noexcept {
    if (!valid_class(cls)) {
        return failure(Code::invalid_argument);
    }
    if (extent.empty()) {
        return failure(Code::invalid_argument);
    }
    if (extent.end > footprint_limit_) {
        return failure(Code::out_of_range);
    }
    // I3: the sum of live extents stays inside the footprint. A valid extent
    // ends inside the footprint and does not overlap a live extent, so the
    // sum cannot grow. This check catches an inconsistent live_bytes_ before
    // it can issue a lease.
    if (live_bytes_ > footprint_limit_ ||
        extent.size() > footprint_limit_ - live_bytes_) {
        return failure(Code::over_capacity);
    }

    // I1: a live or quarantined extent cannot be issued again. A quarantined
    // extent is bad storage, so no other slot may claim those bytes.
    for (std::size_t i = 0; i < capacity_; ++i) {
        const Slot & slot = storage_[i];
        if ((slot.state == State::leased || slot.state == State::quarantined) &&
            slot.extent.overlaps(extent)) {
            return failure(Code::overlap);
        }
    }

    // I2: a reused slot must start at or after the previous lease end. The
    // extent is half-open, so begin == last_end does not overlap.
    std::size_t chosen = capacity_;
    bool out_of_order = false;
    for (std::size_t i = 0; i < capacity_; ++i) {
        Slot & slot = storage_[i];
        if (slot.state != State::free) {
            continue;
        }
        if (extent.begin < slot.last_end) {
            out_of_order = true;
            continue;
        }
        chosen = i;
        break;
    }
    if (chosen == capacity_) {
        return failure(out_of_order ? Code::out_of_order : Code::no_slot);
    }

    Slot & slot = storage_[chosen];
    slot.extent = extent;
    slot.cls = cls;
    slot.state = State::leased;
    ++slot.gen;
    live_bytes_ += extent.size();
    ++live_count_;

    out = Lease{};
    out.arena = id_;
    out.window = window_;
    out.arena_gen = generation_;
    out.slot = static_cast<std::uint32_t>(chosen);
    out.slot_gen = slot.gen;
    out.extent = extent;
    out.cls = cls;
    out.live = true;
    return success();
}

Result Arena::check(const Lease & lease) const noexcept {
    if (!lease.live) {
        return failure(Code::invalid_lease);
    }
    if (lease.arena != id_) {
        return failure(Code::wrong_arena);
    }
    if (lease.window != window_) {
        return failure(Code::wrong_window);
    }
    if (lease.arena_gen != generation_) {
        return failure(Code::stale_lease);
    }
    if (lease.slot >= capacity_) {
        return failure(Code::invalid_lease);
    }
    const Slot & slot = storage_[lease.slot];
    if (slot.gen != lease.slot_gen) {
        // A later lease holds the slot: the lease is stale. Otherwise this is
        // a repeated release of a lease that already returned.
        return failure(slot.state == State::leased ? Code::stale_lease
                                                   : Code::duplicate_release);
    }
    if (slot.state != State::leased) {
        return failure(Code::invalid_lease);
    }
    if (slot.extent != lease.extent || slot.cls != lease.cls) {
        return failure(Code::invalid_lease);
    }
    return success();
}

Result Arena::release(const Lease & lease) noexcept {
    const Result checked = check(lease);
    if (!checked) {
        return checked;
    }
    Slot & slot = storage_[lease.slot];
    slot.state = deferred(slot.cls) ? State::draining : State::free;
    slot.last_end = slot.extent.end;
    ++slot.gen;
    live_bytes_ -= slot.extent.size();
    --live_count_;
    return success();
}

Result Arena::drain(std::size_t slot) noexcept {
    if (slot >= capacity_) {
        return failure(Code::invalid_argument);
    }
    if (storage_[slot].state != State::draining) {
        return failure(Code::not_draining);
    }
    storage_[slot].state = State::free;
    return success();
}

Result Arena::quarantine(const Lease & lease) noexcept {
    const Result checked = check(lease);
    if (!checked) {
        return checked;
    }
    Slot & slot = storage_[lease.slot];
    slot.state = State::quarantined;
    slot.last_end = slot.extent.end;
    ++slot.gen;
    live_bytes_ -= slot.extent.size();
    --live_count_;
    ++cold_reserve_;
    return success();
}

Result Arena::advance_window(std::uint32_t next) noexcept {
    if (live_count_ != 0) {
        return failure(Code::busy);
    }
    window_ = next;
    ++generation_;
    for (std::size_t i = 0; i < capacity_; ++i) {
        if (storage_[i].state == State::draining) {
            storage_[i].state = State::free;
        }
        storage_[i].last_end = 0;
    }
    return success();
}

} // namespace tq::alloc
