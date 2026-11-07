#pragma once

#include <cstddef>
#include <cstdint>

namespace tq::alloc {

// Storage class of a leased extent. Frame and staging extents defer reuse
// until drain. Cold extents return to the free list on release.
enum class Class : std::uint8_t {
    frame = 0,
    staging = 1,
    cold = 2,
};

// State of one slot. A slot is a reusable position in the footprint, not a
// physical address.
enum class State : std::uint8_t {
    free = 0,
    leased = 1,
    draining = 2,
    quarantined = 3,
};

enum class Code : std::uint8_t {
    ok = 0,
    invalid_argument,
    invalid_lease,
    stale_lease,
    wrong_window,
    wrong_arena,
    out_of_range,
    over_capacity,
    overlap,
    out_of_order,
    no_slot,
    not_draining,
    duplicate_release,
    busy,
    full,
    missing,
};

struct Result {
    Code code = Code::ok;

    constexpr bool ok() const noexcept { return code == Code::ok; }
    constexpr explicit operator bool() const noexcept { return ok(); }
};

constexpr Result success() noexcept { return Result{}; }
constexpr Result failure(Code code) noexcept { return Result{code}; }

// Half-open byte range [begin, end). An empty extent has end <= begin.
struct Extent {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;

    constexpr std::uint64_t size() const noexcept {
        return end > begin ? end - begin : 0;
    }
    constexpr bool empty() const noexcept { return end <= begin; }
    constexpr bool overlaps(const Extent & other) const noexcept {
        return begin < other.end && other.begin < end;
    }
    constexpr bool operator==(const Extent & other) const noexcept {
        return begin == other.begin && end == other.end;
    }
    constexpr bool operator!=(const Extent & other) const noexcept {
        return !(*this == other);
    }
};

// One reusable position. last_end is the end of the previous lease on this
// slot. A new lease may use the slot only when its begin >= last_end.
struct Slot {
    Extent extent;
    std::uint64_t last_end = 0;
    std::uint64_t gen = 0;
    Class cls = Class::frame;
    State state = State::free;
};

// Value returned to the caller. It is a capability, not an alias: the arena
// keeps its own copy and validates every field before a state change.
struct Lease {
    std::uint32_t arena = 0;
    std::uint32_t window = 0;
    std::uint64_t arena_gen = 0;
    std::uint32_t slot = 0;
    std::uint64_t slot_gen = 0;
    Extent extent;
    Class cls = Class::frame;
    bool live = false;
};

// Slot and lease bookkeeping. The caller supplies the slot storage. Arena
// never calls malloc or free and never places data inside an extent.
class Arena {
public:
    // Initialize every slot to free. storage must hold capacity slots and
    // must outlive the arena.
    Arena(Slot * storage, std::size_t capacity, std::uint32_t id,
          std::uint32_t window, std::uint64_t footprint_limit) noexcept;

    Arena(const Arena &) = delete;
    Arena & operator=(const Arena &) = delete;

    // Issue a lease for a free slot. Reject an invalid, out-of-range,
    // over-capacity, overlapping, or out-of-order extent. out is unchanged
    // on failure.
    Result acquire(Class cls, Extent extent, Lease & out) noexcept;

    // Return a live lease. Frame and staging slots enter draining. Cold slots
    // become free. A repeated or stale release changes nothing.
    Result release(const Lease & lease) noexcept;

    // Move a draining slot to free. Other states are rejected.
    Result drain(std::size_t slot) noexcept;

    // Return a live lease and mark its slot unusable. One cold reserve unit
    // becomes available for accounting.
    Result quarantine(const Lease & lease) noexcept;

    // Move to a new window and invalidate every outstanding lease. Requires
    // no live lease. Draining slots become free. Quarantined slots remain.
    Result advance_window(std::uint32_t next) noexcept;

    std::uint32_t id() const noexcept { return id_; }
    std::uint32_t window() const noexcept { return window_; }
    std::uint64_t generation() const noexcept { return generation_; }
    std::uint64_t footprint_limit() const noexcept { return footprint_limit_; }
    std::uint64_t live_bytes() const noexcept { return live_bytes_; }
    std::size_t live_count() const noexcept { return live_count_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::uint32_t cold_reserve() const noexcept { return cold_reserve_; }

    const Slot * slots() const noexcept { return storage_; }
    const Slot * slot_at(std::size_t index) const noexcept {
        return index < capacity_ ? &storage_[index] : nullptr;
    }

private:
    Result check(const Lease & lease) const noexcept;

    Slot * storage_;
    std::size_t capacity_;
    std::uint32_t id_;
    std::uint32_t window_;
    std::uint64_t generation_ = 1;
    std::uint64_t footprint_limit_;
    std::uint64_t live_bytes_ = 0;
    std::size_t live_count_ = 0;
    std::uint32_t cold_reserve_ = 0;
};

} // namespace tq::alloc
