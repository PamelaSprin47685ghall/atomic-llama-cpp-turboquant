#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace tq::scheduling {

class admission_budget;

// A reservation names one use of a slot. Copies do not release that slot.
class reservation {
public:
    reservation(const reservation &) = default;
    reservation & operator=(const reservation &) = default;
    reservation(reservation &&) noexcept = default;
    reservation & operator=(reservation &&) noexcept = default;

    std::uint32_t slot() const noexcept { return slot_; }
    std::uint64_t request() const noexcept { return request_; }
    std::uint64_t generation() const noexcept { return generation_; }
    std::uint64_t tokens() const noexcept { return tokens_; }

private:
    friend class admission_budget;
    struct domain {};
    reservation(std::shared_ptr<const domain> owner, std::uint32_t slot, std::uint64_t request,
                std::uint64_t generation, std::uint64_t tokens) noexcept;

    std::shared_ptr<const domain> owner_;
    std::uint32_t slot_;
    std::uint64_t request_;
    std::uint64_t generation_;
    std::uint64_t tokens_;
};

// Reserve a complete request's upper token bound before model work starts.
// This object owns accounting only. The caller owns and releases model memory.
class admission_budget {
public:
    admission_budget(std::uint64_t tokens, std::uint32_t slots);
    admission_budget(const admission_budget &) = delete;
    admission_budget & operator=(const admission_budget &) = delete;

    // Zero demand, demand above capacity, or a duplicate active request is invalid.
    // A full budget returns nullopt. All failed calls leave the budget unchanged.
    std::optional<reservation> try_acquire(std::uint64_t request, std::uint64_t tokens);

    // Call only after the consumer has released the associated model state.
    // Foreign, moved-from, released, and stale reservations return false.
    bool release(const reservation & held) noexcept;

    std::uint64_t capacity() const noexcept { return capacity_; }
    std::uint64_t reserved_tokens() const noexcept { return reserved_; }
    std::uint64_t available_tokens() const noexcept { return capacity_ - reserved_; }
    std::uint32_t slot_count() const noexcept { return std::uint32_t(entries_.size()); }
    std::uint32_t active_count() const noexcept { return active_; }

private:
    struct entry {
        std::uint64_t request = 0;
        std::uint64_t generation = 0;
        std::uint64_t tokens = 0;
    };

    std::shared_ptr<const reservation::domain> owner_;
    std::vector<entry> entries_;
    std::uint64_t capacity_;
    std::uint64_t reserved_ = 0;
    std::uint64_t generation_ = 0;
    std::uint32_t active_ = 0;
};

} // namespace tq::scheduling
