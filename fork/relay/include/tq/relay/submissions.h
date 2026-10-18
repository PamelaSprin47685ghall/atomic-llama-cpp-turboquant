#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tq::relay {

// State of one native submission attempt (guide 35.2). An attempt is
// registered as unknown_in_flight BEFORE the provider call, because a later
// allocation or event-record failure must not lose track of work that may
// already be running (guide 35.3, 52.5).
enum class submit_state {
    not_submitted,      // a proven no-side-effect rejection
    accepted,           // the provider accepted the submission
    unknown_in_flight,  // side effects cannot be ruled out
    completed,          // external native completion proof arrived
};

const char * to_string(submit_state state) noexcept;

// Handle of one registered attempt. Values are issued by run_submissions;
// copies only name the attempt, they do not change its state.
struct attempt_id {
    std::uint64_t value = 0;
};

inline bool operator==(attempt_id a, attempt_id b) noexcept {
    return a.value == b.value;
}

inline bool operator!=(attempt_id a, attempt_id b) noexcept {
    return a.value != b.value;
}

struct attempt_state {
    std::uint64_t window_id = 0;
    std::uint32_t rank      = 0;
    submit_state  state     = submit_state::not_submitted;
    bool          resolved  = false;  // the provider call returned a definite answer
};

// Records per-rank native submission attempts for one run and derives the
// window facts of guide 52.4 and 52.5:
//   in_flight_rank_mask = accepted ranks U unknown ranks
//   sealed              = all attempts resolved, no unknown, all ranks accepted, run healthy
//   drained             = no attempt remains in flight
// accepted and unknown attempts stay in flight until a completed() fact
// arrives; not_submitted attempts have nothing to complete. A completed()
// fact on an unknown attempt is legitimate: the caller may later obtain
// coverage for a submission whose immediate outcome was unclear (guide
// 52.12 F10). This object owns accounting only; the caller owns the actual
// native handles and retained owners.
class run_submissions {
  public:
    // ranks must be in [1, 32]. Invalid configurations throw.
    explicit run_submissions(std::uint64_t run_id, std::uint32_t ranks);

    run_submissions(const run_submissions &)             = delete;
    run_submissions & operator=(const run_submissions &) = delete;

    // Register one attempt BEFORE the provider call. Throws for an invalid
    // rank; returns nullopt after fail().
    std::optional<attempt_id> begin_attempt(std::uint64_t window_id, std::uint32_t rank);

    // Record the provider's answer. Each returns false for an unknown,
    // already resolved, or foreign attempt id.
    bool resolve_accepted(attempt_id id) noexcept;
    bool resolve_no_effect(attempt_id id) noexcept;
    bool resolve_unknown(attempt_id id) noexcept;

    // Record external native completion proof. Valid for accepted and
    // unknown attempts; false otherwise.
    bool completed(attempt_id id) noexcept;

    // Sticky fail-stop; the first non-empty reason is preserved.
    void fail(const std::string & reason);

    bool failed() const noexcept { return failed_; }

    const std::string & failure_reason() const noexcept { return failure_; }

    std::uint32_t rank_count() const noexcept { return ranks_; }

    std::uint32_t accepted_rank_mask(std::uint64_t window_id) const noexcept;
    std::uint32_t unknown_rank_mask(std::uint64_t window_id) const noexcept;
    std::uint32_t in_flight_rank_mask(std::uint64_t window_id) const noexcept;
    bool          attempts_resolved(std::uint64_t window_id) const noexcept;
    bool          sealed(std::uint64_t window_id) const noexcept;
    // True for a window with no attempt, and for a window whose every
    // attempt left the in-flight set. The caller decides whether a window
    // that was never submitted counts as releasable (guide 35.5).
    bool          drained(std::uint64_t window_id) const noexcept;

    std::optional<attempt_state> attempt(attempt_id id) const noexcept;

  private:
    struct record {
        std::uint64_t window_id = 0;
        std::uint32_t rank      = 0;
        submit_state  state     = submit_state::unknown_in_flight;
        bool          resolved  = false;
    };

    std::uint64_t       run_id_;
    std::uint32_t       ranks_;
    std::vector<record> attempts_;
    bool                failed_ = false;
    std::string         failure_;
};

}  // namespace tq::relay
