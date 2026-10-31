# REPORT — real-path dual-window: remove same-phase meet, blocking doorbell and unnecessary synchronization

Status: **static-only deliverable (code + docs).** No compilation, syntax check,
mock, fixture, harness, test or GPU run was performed in this task (work rules:
12022 forbids all compilation/testing; 13022 HIP is not to be burned and no GPU
lock may be taken without Ashley's authorization). Nothing here claims hardware
overlap, device completion, or that the code builds.

## 1. Where the work is

- Worktree: `/home/kunweiz/work/lead-device-wait-20261030/wt`
  (copy of the read-only predecessor `dualwin-device-wait-live-r8-085001/wt`,
  HEAD = stock `f1c5fafc0449b22969ddabfcc9c361940d94d033`; the 085001 candidate
  is in the git index).
- `patch.diff` — cumulative diff stock → **this** tree (16 files).
- `net-increment.patch` — this round only, 085001 candidate → this tree (11 files).
- Predecessors were read only: `dualwin-device-wait-live-r8-{084454,085001}`,
  `dualwin-event-device-wait-r8`, `dualwin-async-*`, `dualwin-mutex-meet`.

## 2. Lineage decision (avoids duplicate/incompatible work)

`080724`/`084454` and `085001` are two increments over the same 052046 base and
both touch the same files, so they cannot be stacked. `084454` (three-gate host
replace) is marked EXPERIMENTAL / NOT DEPLOYABLE by its own report and is **not**
continued here. The overlay async-protocol candidate is a **separate link
product**: its Meta copy has submit-ready/submitted hooks but no event interface,
the tree Meta has the opposite, and the overlay runtime calls
`ggml_backend_meta_first_child_device`, which the overlay Meta never defined.
Therefore the real path here is the **085001 fork compose line**, and no overlay
source was mixed in.

## 3. What was removed, and why it was not a real dependency

### 3.1 Same-phase meet — per-epoch symmetric peer `event_wait` (removed)

The 052046/085001 fork created one event per window from the Meta device
(`ggml_backend_event_new`, which forwarded to the first child), recorded it after
every `llama_decode`, published an availability counter with release, and had the
other window enqueue a device `ggml_backend_event_wait` for the peer's latest
unconsumed record — with a consumption watermark from `dw_device_wait.h`.

Static facts that make it a same-phase rendezvous with no data dependency:

- each window builds its own `llama_context`, KV, requests and a private Meta
  relay provider (`relay_service` keyed per window: `dual_wait_*(wi, native[wi])`,
  `retired(wi, …)`); only the model weights are shared and they are read-only;
- the wait is symmetric in both windows at the same point of the epoch
  (post-decode, pre-consumption), i.e. lockstep T with T;
- even if a cross-window dependency existed, the Meta forwarder reaches child 0
  only, so the wait could never be a five-rank barrier.

Removed with it: the per-window event, the release/acquire publication array, the
watermark, and all of their receipt counters. Deleted
`fork/tp5-compose/src/dw_device_wait.h` (no fork consumer left); the overlay keeps
its own copy and its header note now states there is no fork mirror.

### 3.2 Blocking doorbell — teardown `event_synchronize` (removed)

The fork synchronized the peer event on the host before `llama_synchronize`,
only to drain the records/waits above. With the peer layer gone there is no event
to drain; the two `llama_synchronize` calls per window are kept (lifecycle
boundary around the host-only `seq_rm`, not hot path).

### 3.3 Unnecessary synchronization at the scheduler level (removed)

The Meta **device** registered `event_new/event_free/event_synchronize` and the
Meta **backend** registered `event_record/event_wait` whenever
`GGML_TP5_DW_BACKEND_EVENT=1` **or** `GGML_TP5_DW_COMPOSE_EVENT=1`. A non-null
Meta-device `event_new` makes `ggml_backend_sched` allocate and use its own events
for every Meta backend in the process
(`ggml/src/ggml-backend.cpp`: `ggml_backend_event_new(backends[b]->device)`),
replacing its stock `ggml_backend_synchronize` fallback — an unverified,
process-wide synchronization change that the compose gate never needed.

Both interfaces are now stock `nullptr`; the only Meta event surface is
`ggml_backend_meta_output_record/output_wait`, which calls the contributing child
devices directly and is confined to the gate. `ggml_backend_meta_first_child_device`
(052046) lost its last in-tree consumer and is removed from the header and
definition; the overlay header records that its own build must supply it.

The gate is now a single exact-`"1"` `GGML_TP5_DW_COMPOSE_EVENT`
(`ggml_backend_meta_compose_event_gate`); `GGML_TP5_DW_BACKEND_EVENT` no longer
affects the tree.

## 4. What was strictly kept (real dependencies)

|Dependency|Where|Why it stays|
|---|---|---|
|logits D2H copy completion before the host read|`runtime.cpp:1472-1476`, `1590-1598`; `llama-context.cpp:1644-1677`|the async copy is enqueued inside `llama_decode`; the host reads logits at commit|
|relay dual-completion drain (`finished && retired`)|`runtime.cpp:1517-1580`|shipped exit predicate; relay tail is device-side but courier retirement is CPU|
|five-rank submission gate|`runtime.cpp:1436-1462`|the relay protocol itself|
|stock `llama_synchronize` when no output barrier exists|`runtime.cpp:1499-1517` (`fallback_no_output_barrier`)|fail-closed; deliberately **not** deferred to a later consumer|
|relay `process_ubatch` leading synchronize|`llama-context.cpp:2554`|retained input batch still owns the host graph buffers|
|teardown `llama_synchronize` around `seq_rm`|`runtime.cpp:1745-1750`|lifecycle boundary (fail-closed), not frequency-critical|

## 5. Event ownership, record/wait order, errors, exit, dual-window lifecycle

1. **Owner** — the events are owned by the Meta backend context
   (`output_copy_events`, one slot per child) and created lazily by
   `ggml_backend_meta_output_record`; released by `ggml_backend_meta_output_wait`
   (synchronize + free + null) and by `ggml_backend_meta_free` at context
   destruction. If the relay retains its native owners, the whole Meta owner is
   retained instead (no free behind a live reader).
2. **Record/wait order and event targets** (reviewed against the real readback,
   not assumed) — `ggml_backend_meta_get_tensor_async` copies MIRRORED from child 0
   with `ggml_backend_tensor_get_async`, and a split axis from every child with
   `simple_tensor->nb[axis+1] != 0` with `ggml_backend_tensor_get_2d_async`;
   `output_record` uses the identical predicate and the identical child backend
   objects (`backend_configs[j].backend`). Both CUDA copy paths and
   `ggml_backend_cuda_event_record` use `cuda_ctx->stream()`. Strict relay admits
   exactly one child stream (`lanes_per_rank = 1`, lane stream
   `ctx->streams[device][0]`, `curr_stream_no == 0`, no other stream may exist,
   relay compute rejected otherwise), so the copy, the record and the relay lane
   provably share one stream. The readback backend is the Meta backend because
   relay requires a single scheduler split and rejects non-Meta compute nodes.
   The wait runs after the relay drain and immediately before the first host read;
   `require(dw_proven == dw_copy_streams)` makes a mismatch fatal.
3. **Error path (fixed this round)** — if an epoch is abandoned between record
   and wait (drain failure/abort), its events stayed non-null. A later epoch with
   a different split shape would then have its wait release the stale events too,
   `output_wait` would return more completions than that epoch recorded, and the
   exact-coverage `require` would abort a perfectly valid run. `output_record` now
   releases never-waited events first, without waiting (nothing consumed their
   logits; destroying a recorded event does not cancel stream work; the next
   relay `process_ubatch` synchronize drains the stream).
4. **Accounting on no barrier (fixed this round)** — `tp5_output_copy_wait()`
   used to run `synchronize_accounting()` even when `output_wait` released no
   event (gate off / refused record), i.e. it reported the queued tokens as
   synchronized with no host wait. It now runs the bookkeeping only when
   `complete > 0`.
5. **Exit** — the context owns the events; teardown synchronizes the streams and
   destroys the contexts; `ggml_backend_meta_free` releases any remainder. The
   removed teardown event synchronize was the only blocking wait on the peer
   notification; no exit predicate changed.
6. **Dual-window lifecycle** — worker threads still join before teardown; the only
   cross-window state left is the process-wide submission mutex (protocol) and the
   read-only model. No peer watermark, no atomic publication array, no shared
   event survive.

## 6. Receipt changes

Per window, `TP5_DW_COMPOSE_EVENT window=…` now prints `epochs`
(`decode_calls`), `output_event_syncs`, `host_sync_replaced`,
`fallback_no_output_barrier`, `no_sync_logits_calls`. Structural identities:
`output_event_syncs == host_sync_replaced` and
`output_event_syncs + fallback_no_output_barrier == epochs`. The removed peer
counters (`created`, `record_calls/ok`, `sync_calls`, `free_calls`,
`wait_calls/ok`, `wait_skip_*`, `skipped_unavailable`) no longer exist; the peer
wait never had a receipt in the fork.

## 7. Not claimed / still unproven

- No build, link or syntax result; no unit/CPU/harness execution; no GPU run.
- No device trace and no kernel interval intersection: no overlap, latency or
  throughput claim. `output_event_syncs` / `host_sync_replaced` are host call
  counts. The comment that used to say the drain "already covered the wait time,
  so this returns without adding the full device latency" was **removed** as
  unsupported; the code now states the deferral is a placement and any saving is
  unmeasured.
- The stream/target reasoning above is a **source-level** result. It still needs
  the device trace for: that the reference logits tensor resolves to a split
  state with a non-empty target set (otherwise `output_event_syncs == 0`,
  `fallback_no_output_barrier == epochs` and the run is merely unchanged); that
  the installed ROCm version's event semantics match the CUDA contract used in
  the reasoning; that no other relay-provider component reorders the copy; and
  that per-epoch re-record/free of one event per child is race-free on that
  runtime (one record/wait window per worker is a design argument, not a measured
  property).
- Every missing precondition makes `output_record` return 0 and keeps the stock
  synchronize: non-relay context, no graph, host-resident output, PARTIAL/unknown
  split, missing child event interface, `GGML_CUDA_NO_PEER_COPY` (device
  `event_new` returns nullptr), gate off. That is the fail-closed path, not a
  guarantee that the replacement engages.
- `GGML_TP5_RSZ_RT` is absent from the tree and was not used; sampling
  (`greedy.selected`) was not modified.
