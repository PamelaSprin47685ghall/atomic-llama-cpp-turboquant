#pragma once

// TP5 relay verification ledger.
//
// The pillar-5 uid fast path can be verified on every submission
// (GGML_TP5_RELAY_VERIFY_UID=1). Verification is invisible today: a run with
// the switch on prints the same plan/frame counters as one with the switch
// off, because those counters describe frame selection, not the verification
// itself. Without a bound counter there is no way to tell "verification was
// requested but the fast path never resolved" from "verification described
// every submission" - and the two have very different costs.
//
// This ledger binds three counts to the one describe call that verification
// performs. relay.hip and the CPU test call the same verify_submission()
// below, so the test exercises the production request/resolve/describe/
// execute order rather than a second copy of it:
//
//   requested  the submission asked for verification (the switch is on),
//   resolved   the uid fast path resolved the frame this submission reuses,
//   executed   the verification describe returned (it ran, either outcome).
//
// Invariant: executed <= resolved <= requested. A requested-but-unresolved
// submission (no fast path) is visible as requested > resolved; a resolved
// submission that failed before the describe is resolved > executed.
//
// Cost: three predictable branches per submission, no clock, no allocation,
// no atomics. The provider serializes submissions with its own mutex, so the
// ledger is plain memory owned by one provider.
//
// The counters are cumulative for the provider lifetime. begin(), plan
// flushes and frame retirement do not clear them; destroy() reads them once.

#include <cstdint>

namespace ggml_relay_verify {

struct counts {
    std::uint64_t requested = 0;
    std::uint64_t resolved  = 0;
    std::uint64_t executed  = 0;
};

// One provider's ledger. The per-submission flags are reset by request(), so
// a stage can never be counted twice for the same submission and a stage can
// never be counted without the stage before it.
struct ledger {
    counts total;

    bool this_requested = false;
    bool this_resolved  = false;
    bool this_executed  = false;

    // First call of every accepted submission, before any stage below.
    // Only a submission that actually reached the relay counts: a refused
    // submit (busy provider, malformed stage list) never reaches this call.
    void request(bool verify_enabled) noexcept {
        this_requested = verify_enabled;
        this_resolved  = false;
        this_executed  = false;
        if (verify_enabled) {
            ++total.requested;
        }
    }

    // The uid fast path resolved the frame that this submission will reuse.
    // Ignored unless the submission requested verification and the fast path
    // is real: a slow-path describe is not a verification.
    void resolve(bool fast) noexcept {
        if (!this_requested || !fast || this_resolved) {
            return;
        }
        this_resolved = true;
        ++total.resolved;
    }

    // Immediately after the verification describe returns, whatever it
    // returned. Ignored unless this submission resolved a frame to verify.
    void execute() noexcept {
        if (!this_resolved || this_executed) {
            return;
        }
        this_executed = true;
        ++total.executed;
    }

    // The three counts must stay nested. TP5_PLAN_STATS reports this check on
    // the prepare (every 64 epochs) and destroy lines when the stats switch is
    // enabled; it is a diagnostic, not a hard assertion.
    bool consistent() const noexcept {
        return total.executed <= total.resolved && total.resolved <= total.requested;
    }
};

// One accepted submission. It owns the order and the fast/switch gate shared
// by relay.hip and the CPU test:
//   request -> resolve -> (fast && requested ? describe + execute : nothing)
// `did_describe` (when non-null) reports whether the describe callback ran, so
// the caller can keep the signature check on the same branch. The callback may
// throw; in that case execute() is not reached, matching the production
// describe call.
template <typename Describe>
bool verify_submission(ledger & l, bool requested, bool fast, Describe && describe,
                       bool * did_describe = nullptr) {
    l.request(requested);
    l.resolve(fast);
    const bool verify = fast && requested;
    if (did_describe != nullptr) {
        *did_describe = verify;
    }
    if (!verify) {
        return true;
    }
    const bool described = describe();
    l.execute();
    return described;
}

} // namespace ggml_relay_verify
