#pragma once

#include <tq/relay/message.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tq::relay {

// One bank's observable state, for diagnostics. Ready and consumed sets use
// 32-bit masks, so this channel model is limited to at most 32 ranks.
struct bank_state {
    bool          occupied              = false;
    std::uint64_t ticket                = 0;
    std::uint64_t window_id             = 0;
    std::uint64_t window_generation     = 0;
    std::uint64_t segment_id            = 0;
    std::uint64_t allocation_generation = 0;
    bool          input_released        = false;  // the CPU has finished reading all rank inputs
    bool          output_published      = false;  // the output generation is visible to consumers
    std::uint32_t ready_ranks           = 0;      // producers that published this generation
    std::uint32_t consumed_ranks        = 0;      // consumers that finished their last read
};

struct reserve_reply {
    reserve_code         code = reserve_code::stopped;
    std::optional<lease> granted;
};

// Owns one channel's bank lifecycle (guide 30.4 SlotLedger, 32.4, 50.4).
//
// The first version uses the conservative combined-bank model: one bank
// carries the whole message, input and output together. A bank can be
// reused only after its input has been fully read by the CPU, its output
// has been published, and every registered consumer has finished its last
// read (guide 32.3, 50.5). Separate input and output pools need two
// independent lease bindings and are not claimed by this object.
//
// This is a single-threaded, trusted, single-issuer ownership model. The
// ledger receives facts only after the transport has proven them; calling
// mark_drained does not make any device stop (guide 35, 50.5). Wrong
// identity and stale ticket results are returned as values so the caller
// can decide which integrity violations must escalate to fail-stop.
class channel_ledger {
  public:
    // run_id and plan_id must be positive. ranks must be in [1, 32];
    // banks must be positive. Invalid configurations throw.
    channel_ledger(std::uint64_t run_id,
                   std::uint64_t plan_id,
                   std::uint64_t channel_id,
                   std::uint32_t ranks,
                   std::uint32_t banks);

    channel_ledger(const channel_ledger &)             = delete;
    channel_ledger & operator=(const channel_ledger &) = delete;

    // Reserve the next bank for one (window, generation, segment).
    // allocation_generation identifies the caller's storage arena and is
    // recorded in the lease; a later mismatch is an identity violation.
    reserve_reply reserve(std::uint64_t window_id,
                          std::uint64_t window_generation,
                          std::uint64_t segment_id,
                          std::uint64_t allocation_generation);

    // Record one producer's published input for this message.
    ledger_result producer_ready(const lease & held, std::uint32_t rank);
    // Record that the CPU finished reading every required input. Only this
    // fact releases the input side of the bank (guide 32.2).
    ledger_result reduce_done(const lease & held);
    // Record that the output payload and its generation word are visible.
    // The caller must have its submission-sealed precondition (guide 52.4);
    // the ledger enforces the internal ordering only.
    ledger_result output_published(const lease & held);
    // Record one consumer's last read of this generation (guide 32.3).
    ledger_result consumer_done(const lease & held, std::uint32_t rank);

    // Sticky fail-stop. The first non-empty reason is preserved (guide 35,
    // 50.4 group 4); later failures do not overwrite it. After fail(), the
    // channel refuses new reservations and completion records.
    void fail(const std::string & reason);

    // External drain proof: set only when all submitted native work is
    // proven finished. A false argument changes nothing (guide 35.6).
    bool mark_drained(bool all_native_work_done);

    bool failed() const noexcept { return failed_; }

    const std::string & failure_reason() const noexcept { return failure_; }

    bool drained() const noexcept { return drained_; }

    // A timeout does not prove that a device stopped using these pages.
    bool may_release_storage() const noexcept { return drained_; }

    // Next ticket to issue; 0 means the sequence is exhausted.
    std::uint64_t next_ticket() const noexcept { return next_ticket_; }

    std::uint32_t rank_count() const noexcept { return ranks_; }

    std::uint32_t bank_count() const noexcept { return std::uint32_t(banks_.size()); }

    std::uint32_t active_bank_count() const noexcept;
    bank_state    bank(std::uint32_t slot) const noexcept;

  private:
    struct slot_state {
        bool          occupied              = false;
        std::uint64_t ticket                = 0;
        std::uint64_t window_id             = 0;
        std::uint64_t window_generation     = 0;
        std::uint64_t segment_id            = 0;
        std::uint64_t allocation_generation = 0;
        bool          input_released        = false;
        bool          output_published      = false;
        std::uint32_t ready                 = 0;
        std::uint32_t consumed              = 0;
    };

    ledger_result check_identity(const lease & held, const slot_state *& slot);
    bool          active_duplicate(std::uint64_t window_id,
                                   std::uint64_t window_generation,
                                   std::uint64_t segment_id) const noexcept;

    std::shared_ptr<const lease::domain> owner_;
    std::uint64_t                        run_id_;
    std::uint64_t                        plan_id_;
    std::uint64_t                        channel_id_;
    std::uint32_t                        ranks_;
    std::uint32_t                        member_mask_;
    std::vector<slot_state>              banks_;
    std::uint64_t                        next_ticket_ = 1;
    bool                                 failed_      = false;
    std::string                          failure_;
    bool                                 drained_ = false;
};

}  // namespace tq::relay
