#pragma once

#include <cstdint>
#include <limits>

// Checked u64 arithmetic shared by the ledger and the plan compiler. Every
// derived count and offset must fail before it can wrap (guide 32.5, 54.8).
namespace tq::relay::detail {

inline bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t & out) noexcept {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

inline bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t & out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

// Ticket sequence arithmetic. given "issued_next" is the ticket value that
// was just issued (i.e. the value of next_ticket before the issue), return
// the following next_ticket value. 0 is the exhausted sentinel: it is never
// a valid ticket, so no valid ticket ever wraps back into 0 unnoticed.
constexpr std::uint64_t advance_ticket(std::uint64_t issued_next) noexcept {
    if (issued_next == 0 || issued_next == std::numeric_limits<std::uint64_t>::max()) {
        return 0;
    }
    return issued_next + 1;
}

}  // namespace tq::relay::detail
