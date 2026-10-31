// DUALWIN-DEVICE-WAIT (GGML_TP5_DW_DEVICE_WAIT=1): gate parsing, peer
// selection and the wait/decision state machine for the overlay
// mtp-async runtime's production consumption points (target decode /
// verified-prefix replay). This is the overlay product's own copy: the fork
// tp5-compose runtime no longer ships or includes this header (its symmetric
// per-epoch peer wait was a same-phase rendezvous with no cross-window data
// dependency and was removed), so there is no longer a fork mirror to keep
// byte-identical. `dw_event_trace.h` is still mirrored by both runtimes. The
// header holds everything testable without a GPU: the exact-"1" gate, the
// one-window-has-no-peer rule, the release/acquire pairing contract and the
// watermark that makes every record consumed at most once and never waits on
// an unrecorded event.
//
// HONEST SCOPE: whether the wait's stream/device actually covers relay work
// is unverified (inherited from the r7/r8 wording); the wait can only delay
// a consumption behind work the peer already submitted — it never weakens an
// exit predicate. Deadlock freedom does NOT rest on assertion: per the
// CUDA/HIP event contract (cudaEventRecord: "cudaStreamWaitEvent() use[s]
// the most recently captured state at the time of the API call, and [is]
// not affected by later calls to cudaEventRecord()"), a wait enqueued at
// host time T binds to a record issued at T_R <= T (the release/acquire
// pairing below guarantees the record's host call returned before T); any
// wait contained in that captured work was enqueued before T_R and binds
// strictly earlier, so binding times strictly decrease along any
// wait->captured-work->wait chain — finite, hence acyclic. Re-records cannot
// retarget an enqueued wait.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace dw_device_wait {

// Exact "1", read once at first call (static initialization at the runtime's
// gate aliases; unset/"0"/"2"/"1x" keep every site a counted no-op).
inline bool gate() {
    static const bool on = [] {
        const char * v = std::getenv("GGML_TP5_DW_DEVICE_WAIT");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

// Peer selection: the cyclic neighbour (the dualwin pair for two windows).
inline std::uint32_t peer_of(std::uint32_t wi, std::uint32_t windows) noexcept {
    return windows > 1u ? (wi + 1u) % windows : wi;
}

inline bool has_peer(std::uint32_t wi, std::uint32_t windows) noexcept {
    return windows > 1u && peer_of(wi, windows) != wi;
}

// What the runtime knows about the peer at a consumption point. All fields
// are cheap reads; peer_event comes from the event pointers and same_device
// from the first-child devices resolved once at event creation
// (single-threaded setup, before any decode worker runs).
// NOTE: the tree Meta source no longer exports
// ggml_backend_meta_first_child_device (the fork runtime that used it removed
// its peer-wait layer), and this overlay's own Meta build never defined it.
// This overlay runtime therefore needs that resolution supplied by its own
// Meta patch/build (a public, read-only first-simple-child query) before it
// can link against either Meta source; the planner below is unchanged.
struct context {
    bool event_gate  = false;  // GGML_TP5_DW_BACKEND_EVENT=1 (events exist)
    bool have_peer   = false;  // windows > 1
    bool peer_event  = false;  // the peer's event pointer is non-null
    bool self_event  = false;  // this window's event pointer is non-null
    bool same_device = false;  // both windows resolved the SAME first-child
                               // device (unknown device never waits)
};

inline bool peer_usable(const context & c) noexcept {
    return c.event_gate && c.have_peer && c.peer_event && c.self_event && c.same_device;
}

enum class action { none, wait, skip_unavailable, skip_unrecorded };

// The production decision. peer_available is the peer's released publication
// count (acquire load in the caller); peer_consumed is this window's
// watermark — written ONLY by this window's own thread. A wait is planned
// iff the peer is usable AND published something not yet consumed; the
// watermark then advances to the observed value, so each published record is
// decided exactly once and a stale/empty count never issues a wait. The
// caller issues the non-blocking ggml_backend_event_wait enqueue only on
// action::wait.
inline action plan(const context & c, std::uint64_t peer_available,
                   std::uint64_t & peer_consumed) noexcept {
    if (!peer_usable(c)) {
        return action::skip_unavailable;
    }
    if (peer_available > peer_consumed) {
        peer_consumed = peer_available;
        return action::wait;
    }
    return action::skip_unrecorded;
}

}  // namespace dw_device_wait
