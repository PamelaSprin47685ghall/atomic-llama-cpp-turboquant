#pragma once

#include <cstdint>
#include <memory>

namespace tq::relay {

// Full control-plane identity of one collective message (guide 32.1).
// The ticket is a per-channel monotonically increasing sequence. Tickets
// start at 1; 0 means "never issued". The ticket is the hot-path value;
// the remaining fields are checked by the host ledger and must not be
// compressed away unless a ChannelBinding proves the compression is safe.
struct message_id {
    std::uint64_t run_id            = 0;
    std::uint64_t plan_id           = 0;
    std::uint64_t window_id         = 0;
    std::uint64_t window_generation = 0;
    std::uint64_t channel_id        = 0;
    std::uint64_t segment_id        = 0;
    std::uint64_t ticket            = 0;
};

inline bool operator==(const message_id & a, const message_id & b) noexcept {
    return a.run_id == b.run_id && a.plan_id == b.plan_id && a.window_id == b.window_id &&
           a.window_generation == b.window_generation && a.channel_id == b.channel_id && a.segment_id == b.segment_id &&
           a.ticket == b.ticket;
}

inline bool operator!=(const message_id & a, const message_id & b) noexcept {
    return !(a == b);
}

// A lease names one reservation of one bank. Copies do not release that bank;
// retirement happens only through explicit completion facts (guide 32.8).
// Construction is private: only the issuing channel_ledger can mint a lease,
// so callers cannot hand-roll a "plausible-looking" handle (guide 6.1, 11.2).
class lease {
  public:
    lease(const lease &)                 = default;
    lease & operator=(const lease &)     = default;
    lease(lease &&) noexcept             = default;
    lease & operator=(lease &&) noexcept = default;

    const message_id & message() const noexcept { return message_; }

    std::uint64_t allocation_generation() const noexcept { return allocation_generation_; }

    std::uint32_t slot() const noexcept { return slot_; }

  private:
    friend class channel_ledger;

    struct domain {};

    lease(std::shared_ptr<const domain> owner,
          message_id                    message,
          std::uint64_t                 allocation_generation,
          std::uint32_t                 slot) noexcept;

    std::shared_ptr<const domain> owner_;
    message_id                    message_;
    std::uint64_t                 allocation_generation_ = 0;
    std::uint32_t                 slot_                  = 0;
};

// Result of asking the ledger for the next bank (guide 32.8).
enum class reserve_code {
    accepted,            // a fresh message identity and bank lease were granted
    wait_for_capacity,   // the next bank still has live readers; retry later
    sequence_exhausted,  // no further ticket can be issued without wrapping
    invalid_message,     // the same (window, generation, segment) is already active
    stopped,             // the channel has failed or drained
};

// Result of recording one completion fact on an existing lease.
// Values other than ok do not modify ledger state.
enum class ledger_result {
    ok,
    stale_ticket,    // lease ticket is not the bank's current ticket (guide 32.6)
    wrong_identity,  // run/plan/channel/window/segment or issuer mismatch: integrity violation
    not_ready,       // a required prior completion fact is still missing
    duplicate,       // this completion was already recorded, or arrived after retirement
    invalid_rank,    // rank is outside the channel membership
    stopped,         // the channel has failed or drained
};

const char * to_string(reserve_code code) noexcept;
const char * to_string(ledger_result result) noexcept;

}  // namespace tq::relay
