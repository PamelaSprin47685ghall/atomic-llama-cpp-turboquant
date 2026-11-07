// DUALWIN-COMPOSE-EVENT (GGML_TP5_DW_COMPOSE_EVENT=1): exact-"1" gate for the
// production-path ggml backend event record/wait wired into the fork compose
// runtime (fork/tp5-compose/src/runtime.cpp).
//
// Semantics: exact value "1" only, read once at static initialization (the
// runtime aliases gate() into a file-scope const before main; later env
// changes are never observed). unset, "0", "2", "1x", " 1", "10" and every
// other spelling keep the OFF path: no allocation, no event, no record/wait
// call, no receipt, no new branch body executed — OFF equals the applied
// baseline/stock in source structure.
//
// ON path (real calls stay in the runtime and in ggml):
// llama_tp5_output_copy_record() records one device event per child stream
// that the asynchronous logits D2H copy of the just-decoded epoch used. ggml
// resolves the stream set from the same split state the readback uses and
// records through each contributing child backend/device: MIRRORED reads the
// first child alone; a split axis copies from every child with a non-empty
// chunk. PARTIAL/unknown output, a host-resident output or a missing child
// event interface make the record fail closed (it returns 0 and leaves no
// allocated event behind).
//
// The record is ordered behind that copy on the same stream, so the paired
// llama_tp5_output_event_wait() — deferred until after the relay drain, right
// before the host reads logits — proves the host row is readable. That wait IS
// the epoch's host synchronization: the scheduler-wide llama_synchronize /
// syncing logits getter is skipped, rows are resolved with
// llama_get_logits_ith_no_sync, and only the eval bookkeeping of a normal
// synchronize() runs (and only when the wait actually released a barrier).
//
// NOT a Meta event interface. The Meta device deliberately keeps stock
// nullptr event_new/event_free/event_synchronize and the Meta backend keeps
// stock nullptr event_record/event_wait. A non-null Meta event_new would make
// ggml_backend_sched allocate and use its own events for every Meta backend in
// the process (replacing its stock ggml_backend_synchronize fallback), and the
// Meta forwarder only reaches child 0 — neither can order all five relay
// ranks. The barrier therefore talks to the contributing child devices
// directly and cannot change any other graph's synchronization.
//
// NO cross-window wait. Each window owns a private Meta relay provider, its
// own context, KV and requests; the model weights are read-only shared state,
// so no writable state crosses windows. A symmetric per-epoch peer event wait
// would be a same-phase rendezvous with no data dependency (and a rank-0-only
// one through the deleted Meta forwarder). It was removed together with its
// per-window event, release publication and consumption watermark. The relay
// dual-completion drain is unchanged and remains the runtime's only exit
// predicate.
//
// Real dependencies that are NOT removed: the output-copy event wait itself
// (it blocks the host on device events), the relay dual-completion drain, the
// five-rank submission gate, and the stock synchronize kept fail-closed when
// no sound output barrier exists (missing child event interface, PARTIAL/
// unknown or host-resident output). That fallback is counted in
// fallback_no_output_barrier and is never deferred to a later consumer.
//
// The receipt counts host calls; it is NOT device completion evidence and does
// not prove the hardware stream mapping.
#pragma once

#include <cstdlib>
#include <cstring>

namespace dw_compose_event {

inline bool gate() {
    static const bool on = [] {
        const char * v = std::getenv("GGML_TP5_DW_COMPOSE_EVENT");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

}  // namespace dw_compose_event
