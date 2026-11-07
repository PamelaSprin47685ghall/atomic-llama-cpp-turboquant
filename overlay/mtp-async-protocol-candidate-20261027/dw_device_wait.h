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
// STATUS: the planner is NOT wired to a cross-window wait, and it must not be.
// The overlay runtime's two consumption points (target decode, verified-prefix
// replay) keep this gate fail-closed for that shape, for two independent
// reasons:
//   (1) no dependency — each window owns a private Meta relay provider, its own
//       context, KV/recurrent state and requests; only read-only weights are
//       shared. Blocking one window on the other's epoch would be a same-phase
//       rendezvous that turns the pair into lockstep, and it would order
//       nothing the windows actually share;
//   (2) no legal handle — a relay rank lane's completion is a HIP hipEvent_t.
//       ggml_backend_event_t is NOT a HIP handle: it is a struct whose first
//       member is the device pointer that ggml_backend_event_synchronize
//       dereferences. Lending a HIP handle as a ggml event is a type confusion
//       (it segfaults on first use), not a shortcut. Legal ggml events come
//       only from ggml_backend_event_new on a child device — which is exactly
//       what ggml_backend_meta_output_record does, one event per contributing
//       child, i.e. every rank of THIS window for a split output.
// What the runtime does instead is inside one window: it records this window's
// output-copy events after decode and host-completes them after the relay
// drain (llama_tp5_output_copy_record / llama_tp5_output_event_wait). The
// planner below stays as the reviewed decision table and is still consulted
// with a peerless context, where it can only answer skip_unavailable.
//
// HONEST SCOPE: a wait can only delay a consumption behind work already
// submitted — it never weakens an exit predicate. Because the runtime's wait
// is a host wait on its own recorded events, no wait edge enters a device
// stream and no cycle can form through one. A wait enqueued on a device stream
// would need the CUDA/HIP event contract to argue acyclicity (cudaEventRecord:
// "cudaStreamWaitEvent() use[s] the most recently captured state at the time
// of the API call, and [is] not affected by later calls to cudaEventRecord()");
// that shape is not used here and must not be reintroduced.
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

// What the runtime knows at a consumption point. All fields are cheap reads.
// The runtime fills a PEERLESS context: have_peer/peer_event/same_device stay
// false, so peer_usable() is false and every plan is skip_unavailable. The
// fields are kept because they are the reviewed contract for any future
// same-window use, not because a peer is ever named today.
struct context {
    bool event_gate  = false;  // GGML_TP5_DW_BACKEND_EVENT=1 (events exist)
    bool have_peer   = false;  // windows > 1
    bool peer_event  = false;  // never set: no cross-window wait exists
    bool self_event  = false;  // this window's event pointer is non-null
    bool same_device = false;  // never set: no cross-window wait exists
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
// decided exactly once and a stale/empty count never issues a wait. On
// action::wait the caller would complete the recorded events as a HOST wait;
// it never enqueues ggml_backend_event_wait on any stream. No caller reaches
// that branch today: every context is peerless.
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
