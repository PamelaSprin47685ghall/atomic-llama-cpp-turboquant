#include "tq/alloc/arena.h"

namespace tq::alloc {
namespace {

constexpr bool valid_reuse(ReuseClass reuse) noexcept {
    return reuse == ReuseClass::frame || reuse == ReuseClass::staging ||
           reuse == ReuseClass::cold;
}

constexpr bool valid_lifetime(Lifetime lifetime) noexcept {
    return static_cast<std::size_t>(lifetime) < lifetime_count;
}

constexpr bool deferred(ReuseClass reuse) noexcept {
    return reuse == ReuseClass::frame || reuse == ReuseClass::staging;
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

Result Arena::acquire(Lifetime lifetime, ReuseClass reuse, Extent extent,
                      Lease & out) noexcept {
    if (!valid_lifetime(lifetime) || !valid_reuse(reuse)) {
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
        const Slot & slot = storage_[i];
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
    return issue(chosen, lifetime, reuse, extent, out);
}

Result Arena::acquire(Lifetime lifetime, ReuseClass reuse, std::size_t bytes,
                      Lease & out) noexcept {
    if (!valid_lifetime(lifetime) || !valid_reuse(reuse)) {
        return failure(Code::invalid_argument);
    }
    if (bytes == 0) {
        return failure(Code::invalid_argument);
    }
    const std::uint64_t need = static_cast<std::uint64_t>(bytes);
    if (need > footprint_limit_) {
        return failure(Code::out_of_range);
    }
    if (live_bytes_ > footprint_limit_ ||
        need > footprint_limit_ - live_bytes_) {
        return failure(Code::over_capacity);
    }

    std::size_t chosen = capacity_;
    std::uint64_t chosen_begin = 0;
    bool saw_free = false;
    bool saw_overlap = false;

    for (std::size_t i = 0; i < capacity_; ++i) {
        const Slot & slot = storage_[i];
        if (slot.state != State::free) {
            continue;
        }
        saw_free = true;
        const std::uint64_t begin = slot.last_end;
        if (begin > footprint_limit_ || need > footprint_limit_ - begin) {
            continue;
        }
        const Extent candidate{begin, begin + need};
        bool blocked = false;
        for (std::size_t j = 0; j < capacity_; ++j) {
            const Slot & other = storage_[j];
            if ((other.state == State::leased ||
                 other.state == State::quarantined) &&
                other.extent.overlaps(candidate)) {
                blocked = true;
                break;
            }
        }
        if (blocked) {
            saw_overlap = true;
            continue;
        }
        if (chosen == capacity_ || begin < chosen_begin) {
            chosen = i;
            chosen_begin = begin;
        }
    }
    if (chosen == capacity_) {
        if (!saw_free) {
            return failure(Code::no_slot);
        }
        return failure(saw_overlap ? Code::overlap : Code::out_of_range);
    }
    return issue(chosen, lifetime, reuse,
                 Extent{chosen_begin, chosen_begin + need}, out);
}

Result Arena::issue(std::size_t index, Lifetime lifetime, ReuseClass reuse,
                    Extent extent, Lease & out) noexcept {
    Slot & slot = storage_[index];
    slot.extent = extent;
    slot.lifetime = lifetime;
    slot.reuse = reuse;
    slot.state = State::leased;
    ++slot.gen;

    const std::uint64_t size = extent.size();
    live_bytes_ += size;
    ++live_count_;
    ++leases_;
    if (live_bytes_ > high_water_) {
        high_water_ = live_bytes_;
    }
    LifetimeStats & stats = lifetime_[static_cast<std::size_t>(lifetime)];
    stats.live_bytes += size;
    ++stats.live_count;
    ++stats.leases;
    if (stats.live_bytes > stats.high_water) {
        stats.high_water = stats.live_bytes;
    }

    out = Lease{};
    out.arena = id_;
    out.window = window_;
    out.arena_gen = generation_;
    out.slot = static_cast<std::uint32_t>(index);
    out.slot_gen = slot.gen;
    out.extent = extent;
    out.lifetime = lifetime;
    out.reuse = reuse;
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
    if (slot.extent != lease.extent || slot.lifetime != lease.lifetime ||
        slot.reuse != lease.reuse) {
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
    slot.state = deferred(slot.reuse) ? State::draining : State::free;
    slot.last_end = slot.extent.end;
    ++slot.gen;

    const std::uint64_t size = slot.extent.size();
    live_bytes_ -= size;
    --live_count_;
    ++releases_;
    LifetimeStats & stats = lifetime_[static_cast<std::size_t>(slot.lifetime)];
    stats.live_bytes -= size;
    --stats.live_count;
    ++stats.releases;
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
    ++drains_;
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

    const std::uint64_t size = slot.extent.size();
    live_bytes_ -= size;
    --live_count_;
    ++quarantines_;
    ++cold_reserve_;
    LifetimeStats & stats = lifetime_[static_cast<std::size_t>(slot.lifetime)];
    stats.live_bytes -= size;
    --stats.live_count;
    ++stats.quarantines;
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

ArenaStats Arena::stats() const noexcept {
    ArenaStats result;
    result.live_bytes = live_bytes_;
    result.live_count = live_count_;
    result.high_water = high_water_;
    result.leases = leases_;
    result.releases = releases_;
    result.drains = drains_;
    result.quarantines = quarantines_;
    result.cold_reserve = cold_reserve_;
    result.capacity = capacity_;
    result.footprint_limit = footprint_limit_;
    return result;
}

LifetimeStats Arena::stats(Lifetime lifetime) const noexcept {
    const std::size_t index = static_cast<std::size_t>(lifetime);
    return index < lifetime_count ? lifetime_[index] : LifetimeStats{};
}

} // namespace tq::alloc
