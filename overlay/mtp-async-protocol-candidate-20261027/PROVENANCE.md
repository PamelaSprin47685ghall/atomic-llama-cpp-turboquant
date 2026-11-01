# Overlay provenance — mtp-async-protocol-candidate-20261027

## Layering (all files are real; no diff is synthesized)

    base/runtime.ldl.cpp --(runtime.ldl.patch)--> base/runtime.cpp          --(runtime.patch)--> runtime.cpp
    base/ggml-backend-meta.ldl.cpp --(meta.ldl.patch)--> base/ggml-backend-meta.cpp --(meta.patch)--> ggml-backend-meta.cpp

| artifact | sha256 | why it exists |
|---|---|---|
| `base/runtime.ldl.cpp` | `5656748435f8e44b2147cdf34c203d3ecc71b29dd67b6151e2202e563b3ba5f6` | original pre-image at archive time (overlay root `base/runtime.cpp` at commit f1c5fafc0) |
| `runtime.ldl.patch` | `be9e73501a608605bfbf5adb386af519ee3799c033c4cacbc8c37eedc69bce1a` | the archived patch verbatim (overlay root `runtime.patch` at f1c5fafc0) |
| `base/runtime.cpp` | `86acb16fff12eb9c21ecc499f29b4156777c893618079a00ed1f32f580f30c95` | the BUILT source (2177 lines) = `runtime.ldl.cpp` + `runtime.ldl.patch`; equals the 2026-10-27 `build-receipt.json` `runtime_sha256` |
| `runtime.patch` | `8173485a237d59d9787fa11e002c5d1b8b08047bb42fb8d39c442570ac7a2de9` | `base/runtime.cpp` -> delivered `runtime.cpp` (32 hunks) |
| `runtime.cpp` | `f29f3b0809cbb1959f2a50ce8a29b6e5af54f93de06eef73d4139bcecfc38688` | delivered source (3592 lines) |
| `base/ggml-backend-meta.ldl.cpp` | `29c8fdf349e1c333524e3ae9b820c887ad29fc837a9f6ecf92770e384db28cd2` | original pre-image at archive time (overlay root `base/ggml-backend-meta.cpp` at f1c5fafc0) |
| `meta.ldl.patch` | `b525c6906cbc004522fea76f1e75b79597371e9159c9cb6458a5a5434f5d867d` | archived Meta patch verbatim (overlay root `meta.patch` at f1c5fafc0) |
| `base/ggml-backend-meta.cpp` | `e744a6dc582b24f76c1d483a7edfd5fb69b69a67f3b8175d64129759736e4109` | the BUILT Meta source (4309 lines) = `ggml-backend-meta.ldl.cpp` + `meta.ldl.patch`; the 2026-10-27 receipt `meta_sha256` / `meta_sha256_built_source` |
| `meta.patch` | `eb17e98b173d419cc1b1daa5953284be9da063cd94b3efb3bfd2df38a6342ac8` | `base/ggml-backend-meta.cpp` -> delivered `ggml-backend-meta.cpp` (2 hunks: the two first-child accessors) |
| `ggml-backend-meta.cpp` | `44461aad1c517a8bc29ae64d8d96636a32e2da7e4418c8f96f72a8971cffdc52` | delivered Meta source (4344 lines) |
| `headers/ggml-backend.h` | `1d52503425e7a14f85181c9aacc1d0e1258c4a26399180fc7df08dc661bea7c7` | delivered copy of the recorded public header (== tree `ggml/include/ggml-backend.h`); `headers/README.md` owns the closure table |
| `headers/llama.h` | `c0aad9efd9eb1167b1240da95b674e7d28997f16e4964dc56fdc5b87c10f3510` | delivered copy of the recorded public header (== tree `include/llama.h`) |

`build-receipt.json` keeps the 2026-10-27 step/binary/library facts untouched. `runtime_sha256`
and `meta_sha256` now name the delivered sources (the files a rebuild would actually compile);
`meta_sha256_built_source` preserves the hash the recorded build really hashed. The `provenance`
block states the built/delivered split for both objects.

## First-simple-child accessor (why the delivered runtime did not compile)

`runtime.cpp` (DUALWIN-DEVICE-WAIT/BACKEND-EVENT) resolves this window's first simple child.
The value it needs is a DEVICE (for the wait site's device-identity comparison), the argument
the Meta layer needs to find it is a BACKEND:

```cpp
dw_be.device = ggml_backend_meta_first_child_device(native.back());   // Meta backend in, child DEVICE out
auto * dw_be_dev = ggml_backend_get_device(native.back());            // Meta device: only for ggml_backend_event_new
```

That split matches the neighbours: `ggml_backend_is_meta` and
`ggml_backend_meta_simple_backend(backend, index)` both take the Meta BACKEND, and
`ggml_backend_get_device(backend)` returns a DEVICE. An earlier delivery passed the Meta
*device* (`ggml_backend_get_device(native.back())`) to an accessor that takes a backend — the
type error Devops hit. It is fixed at the call site; the declaration and both definitions
already took `ggml_backend_t meta_backend` and stay as they are.

The historical source called `ggml_backend_meta_first_child_device(dw_be_dev)` — a Meta
*device* argument — against an API that takes a Meta *backend*. Neither Meta source defined
the symbol, and the recorded include set has no `ggml-backend-impl.h`, where the sibling child
accessors live.
The delivered tree therefore declares it in the PUBLIC `ggml/include/ggml-backend.h` (next to
`ggml_backend_meta_output_record/wait`) and defines the SAME body in both Meta sources:

```cpp
ggml_backend_dev_t ggml_backend_meta_first_child_device(ggml_backend_t meta_backend) {
    if (meta_backend == nullptr || !ggml_backend_is_meta(meta_backend)) return nullptr;
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    if (backend_ctx == nullptr || backend_ctx->backend_configs.empty() ||
            backend_ctx->backend_configs[0].backend == nullptr) return nullptr;
    return ggml_backend_get_device(backend_ctx->backend_configs[0].backend);
}
```

Ownership/lifetime: a Meta backend enumerates its children in its Meta device's simple-device
order (`ggml_backend_meta_context` ctor: `backend_configs[j]` <-> `ggml_backend_meta_dev_simple_dev(meta_dev, j)`),
and each child backend is `ggml_backend_dev_init`'d from one of those devices. The returned
`ggml_backend_dev_t` belongs to the child's own backend registry (`ggml_backend_get_device`),
NOT to the Meta backend, so it stays valid for device-identity comparison while that registry
lives; no Meta-backend teardown frees it. Caller must not free it and must not use it after
registry shutdown. The probe is null-safe where the sibling accessors `GGML_ASSERT`, so an
ungated call cannot abort; it allocates, records and prints nothing, so OFF behaviour is
unchanged.

`dw_device_wait.h` documented the previous gap in a note; that note now states the live wiring
(host rank-event wait, see "Rank events" below) and the `context` fields it fills. The planner
itself is unchanged (`peer_usable()` false => `skip_unavailable`, never waits).

## Why the ON leg skipped every event (root cause, resolved)

Observed ON receipt: `created=0 record_ok=0 skipped_unavailable=1`, `wait_calls=0`.

The event was created as `ggml_backend_event_new(ggml_backend_get_device(meta_backend))` — i.e.
from the **Meta device**. `ggml_backend_event_new` is:

```cpp
if (device == NULL || device->iface.event_new == NULL) return NULL;
```

and the Meta device interface deliberately sets `event_new/event_free/event_synchronize` to
`nullptr` (`ggml/src/ggml-backend-meta.cpp`, `ggml_backend_meta_device_iface`) because a
non-null Meta `event_new` would make `ggml_backend_sched` allocate and use its own events for
every Meta backend in the process. So the creation call *must* return NULL, `dw_be.event`
stayed null, and every later site (`record`, and `dw_device_wait::peer_usable()` ->
`skip_unavailable`) was skipped. The separate `wait_skip_unavailable=13` follows from the same
null event; `first_child_device()` was never the problem (it returned the child device, but a
device only *templates* a backend — the child **backend** is what carries
`event_record`/`event_wait`).

Fix (production path, gates unchanged):

- new public probe `ggml_backend_meta_first_child_backend(meta_backend)` (same null-safe,
  allocation-free contract as `..._first_child_device`; returns `backend_configs[0].backend`,
  owned by the Meta backend until `ggml_backend_meta_free`);
- the window setup resolves that child backend once and creates the event from
  `ggml_backend_get_device(child_backend)`;
- `ggml_backend_event_record(...)` and `ggml_backend_event_wait(...)` now take that **child
  backend** (the Meta backend has no `event_record`; the event's device IS the child device, so
  the record lands on the same stream the event was created for);
- the receipt prints the actual pointers
  (`child_backend= meta_backend= child_device= meta_device=`), so a still-skipping run names
  its own cause instead of leaving it to inference.

Nothing about the gates changed: `GGML_TP5_DW_BACKEND_EVENT` / `GGML_TP5_DW_DEVICE_WAIT` are
still exact-"1", still read once, and every skip is still counted — the numbers must move from
`created=0` to `created=1` with `record_ok == record_calls`; a skip is still a skip, never a
success.

## Cross-window device wait is FAIL-CLOSED (reviewer P1 outcome — SUPERSEDED, see "Rank events")

SchedulerReview found that the ON leg's `event_wait(peer.backend, peer.event)` was the PEER's
stream waiting on the PEER's own event (a self-wait: it constrains nothing for this window), and
that only rank 0 was ever covered while TP5 relay work spans five ranks. Both are confirmed in
the source: `ggml_backend_meta_first_child_backend/device` resolve `backend_configs[0]`, and the
children are the five rank backends `relay_i->create(simple_backends...)` is built from.

A wait can only be issued here if three facts are TRUE AND DEMONSTRATED: (1) an event covering
every rank's relay work exists, (2) the wait is enqueued on THIS window's own stream against the
peer's event, (3) the stream the record lands on carries the relay compute. None holds: a Meta
backend deliberately exports no event interface (only child 0's event is reachable), the child
stream an event can be recorded on carries the asynchronous logits D2H copy issued after decode,
and enqueueing through the peer's backend lands the wait on the peer's stream. Therefore the
runtime now makes **no** device wait: both consumption points count
`wait_skip_unavailable` and issue nothing, the receipt banner says
`mode=fail_closed waits_issued=0`, and the per-window availability counter that only fed the
removed planner is gone. `wait_calls>0` is impossible while the gate is on, and a zero is not a
pass. The device event record (child backend, rank 0) stays as an honest, separately-labelled
receipt; it is not a dependency.

> **Superseded by the "Rank events" round below.** The three conditions above were unmet for the
> *enqueue* shape (`ggml_backend_event_wait` on the peer's backend). The live design does not
> try to satisfy them: it records one event per rank lane in the provider and completes them on
> the HOST, so coverage is five ranks, nothing is enqueued on any stream, and no self-wait
> exists. The paragraph above describes what the code did before that round; it is kept so the
> reasoning that produced it stays auditable, not as a description of the current runtime.

## Cross-window device dependency — REMOVED (no gate, no ABI bump)

**Boundary.** The two windows own disjoint WRITABLE device state: each has its own Meta relay
provider (`relay_i->create(simple_backends…)` runs once per Meta backend, so `p.mutex`, epochs,
channel banks, `rank[r].scratch`, `channel_pool` and `abort_device[]` are provider-private), its
own llama context with its own KV/recurrent state, and only the weights are shared (read-only).
Window A's device work writes nothing window B reads. There is therefore **no cross-window data
dependency**, and a stream wait for one would be the same-phase rendezvous the dual-window proof
forbids.

**What existed briefly and is now gone.** An earlier revision introduced
`GGML_TP5_DW_XWIN_DEVICE` with per-rank tail events, an ABI-3 provider member and a
`ggml_backend_xstream_wait` primitive. Reviewer findings: the HIP facts make it unrepairable
(`hipEventRecord` keeps only the latest record, so a missed epoch can never be waited on again;
HIP cannot roll back an already enqueued `hipStreamWaitEvent`, so "five ranks or none" is not
expressible), and since there is no dependency at all the gate was a pseudo-switch. It was
first reduced to an observation and then **removed entirely**:

| removed | where |
|---|---|
| `GGML_TP5_DW_XWIN_DEVICE` gate, `dw_xwin_device.h`, per-window rank/observation state, `TP5_DW_XWIN_DEVICE` receipt | `runtime.cpp` |
| `ggml_backend_relay_tail_epoch` (+ its Meta forwarder and public declaration) | `ggml-backend-meta.cpp`, `ggml/include/ggml-relay.h` |
| provider `tail_epoch` member, `tail_epoch()` reader, `p.tail_epoch` publish, ABI 3 bump (`..._MIN`, version branches) | `ggml/src/ggml-cuda/relay.hip`, `ggml/src/ggml-relay-impl.h`, `ggml-backend-meta.cpp` |
| `ggml_backend_xstream_wait`, the backend `xstream_wait` iface hook, `ggml/include/ggml-xstream.h` | `ggml-backend.cpp`, `ggml-backend-impl.h`, `ggml-cuda.cu` |

That removal round left relay provider ABI at **2**. The round after it (below) appends one
member and bumps the ABI to **3**; the ABI-2 prefix keeps its offsets, and the completeness
check in `ggml-backend-meta.cpp` rejects an older provider explicitly rather than degrading.

**Kept, because they have real consumers:** `ggml_backend_meta_first_child_device` /
`..._first_child_backend` (the `GGML_TP5_DW_BACKEND_EVENT` leg resolves them). The fallback that
creates the per-window event on the first child that can actually create one is written inline in
`runtime.cpp` over the existing public accessors
(`ggml_backend_meta_n_backends` / `..._simple_backend` + `ggml_backend_get_device`), so no new Meta
symbol and no header change is needed.

## Rank events: RETRACTED (type confusion + same-phase lockstep)

An intermediate revision exposed the relay provider's per-rank HIP events through a provider vtable
member (`rank_events`, "ABI 3") and completed them from the peer window's decode thread. It was
withdrawn for two independent reasons, both fatal:

1. **Type confusion.** `ggml_backend_event_t` is `struct ggml_backend_event *`, i.e. a struct whose
   first member is a `device` pointer; `ggml_backend_event_synchronize` dereferences
   `event->device->iface`. A relay lane's `hipEvent_t` is a raw HIP handle. Casting one into the
   other "works" only because both are pointer-sized, and the first `synchronize` on it
   dereferences the HIP handle as a struct - a guaranteed crash. `sizeof` equality is not
   type-safety.
2. **Same-phase lockstep.** The two windows own disjoint writable device state (separate providers,
   contexts, KV/recurrent state; only read-only weights are shared). Blocking one window on the
   other's epoch created a cross-window rendezvous with no data to order.

What is in the tree now:

| piece | state |
|---|---|
| relay provider ABI | **2** again; `ggml-relay-impl.h` is byte-identical to HEAD |
| `relay.hip` (tree and overlay) | the busy-wait stage only (`relay.ldl.hip`); no `publish` event, no `rank_events`, `++p.status.cpu_reductions` unconditional as before |
| `ggml_backend_meta_rank_events` / `..._rank_wait` | removed from both Meta sources and the public header |
| `GGML_TP5_DW_DEVICE_WAIT` | fail-closed for cross-window waits; see `dw_device_wait.h` and the gate comment in `runtime.cpp` |

## What replaces it: one window's own output-copy barrier

The only sound wait is on events a window really owns. The overlay runtime now uses the same
Meta-level barrier as the fork compose runtime, per window:

- `llama_tp5_output_copy_record(ctx)` -> `ggml_backend_meta_output_record`, which calls the real
  `ggml_backend_event_new(child_dev)` and `ggml_backend_event_record(event, child_backend)` on
  every child stream the logits D2H copy used - one child for a MIRRORED output, every
  contributing child (all five ranks) for a split output. These are genuine
  `ggml_backend_event_t` values, never HIP handles.
- `llama_tp5_output_event_wait(ctx)` completes them on the host after the relay drain, right
  before a logits row is read; a 0 from `record` keeps the stock `llama_synchronize`
  (fail-closed, counted as `wait_skip_unavailable`).
- Logits rows are then resolved with `llama_get_logits_ith_no_sync` only when that barrier
  exists; otherwise the stock synchronizing getters stay exactly as before.

`GGML_TP5_DW_DEVICE_WAIT` therefore needs no separate mechanism: it is the runtime's switch for
this same-window barrier, and its receipt is
`TP5_DW_DEVICE_WAIT ... mode=same_window_output_copy_barrier`.

**Still unproven (do not read this as completion of the dual-window work):** asynchronous send,
busy-wait behaviour, per-window provider independence and actual GPU overlap all still need real
measurement. Two windows sharing one physical GPU contend for the same SMs/queues; any overlap
number must be read against that contention, and nothing here asserts it.

## Header closure (see headers/README.md)

`compile.argv.json` has exactly three `-I` roots; the ggml/llama public headers come from
`.../headers/build-tp5-dual50-20261022/hc-router-owner-a-install/include`. That copy of
`ggml-backend.h` and `llama.h` differs from this tree's delivered headers ONLY by the
candidate's own declarations, so they are closed by `headers/base/<h>` + `headers/<h>.patch`
-> `headers/<h>` (== tree delivered header). `ggml-cuda.h` was dropped from `runtime.cpp`:
nothing from it was used (the q8-cache probe resolves through `get_proc_address` with a local
function type, as in `fork/tests/a-seams/*`), so the recorded include set is sufficient and no
new `-I` is required. No unrelated header content is touched.

## History of the earlier inconsistency

The 2026-10-27 build hashed the BUILT sources. The delivered `runtime.cpp` is a later iteration:
the dualwin rounds (DEVICE-WAIT, BACKEND-EVENT, OVERLAP-METER, COURIER, SERIALCUT[-TRIM],
MUTEX-MEET, COMBO, dual completion verification) were src-overlayed after that build and the
archived patch was never regenerated — the 1774-line gap Devops found. Both chains above are
now three-stage and reproducible in both directions.

## HIP relay candidate (hip-relay/)

`hip-relay/` has its own three-stage chain — `relay.orig.hip` -> `relay.ldl.hip` (busy-wait,
the lane's archived one-line patch) -> `relay.hip` (delivered: the same source plus the
DUALWIN-SERIALCUT-TRIM and DUALWIN-OVERLAP-METER host-gate adoption, 189+/1-). See
`hip-relay/PROVENANCE.md`; the earlier `relay.patch` only carried the first edge, which is why
`orig + patch` stopped at `1badd280` instead of the delivered `318c9b50`.

## Verify (CPU; runs no build — for Devops)

    cd <this dir>
    patch -o /tmp/f.cpp -p0 base/runtime.cpp -i runtime.patch && sha256sum /tmp/f.cpp runtime.cpp
    patch -o /tmp/r.cpp -R -p0 runtime.cpp -i runtime.patch && sha256sum /tmp/r.cpp base/runtime.cpp
    patch -o /tmp/mf.cpp -p0 base/ggml-backend-meta.cpp -i meta.patch && sha256sum /tmp/mf.cpp ggml-backend-meta.cpp
    patch -o /tmp/mr.cpp -R -p0 ggml-backend-meta.cpp -i meta.patch && sha256sum /tmp/mr.cpp base/ggml-backend-meta.cpp
    patch -o /tmp/ldl.cpp -p0 base/runtime.ldl.cpp -i runtime.ldl.patch          # 86acb16f… (built)
    patch -o /tmp/ml.cpp  -p0 base/ggml-backend-meta.ldl.cpp -i meta.ldl.patch   # e744a6dc… (built)

The fork tree has its own self-contained runtime (`fork/tp5-compose/src/runtime.cpp`, 1828
lines) and ships no `base/`+patch pair; the overlay is not a fork patch series.
