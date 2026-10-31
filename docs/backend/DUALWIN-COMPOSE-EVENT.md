# Dual-window compose epochs: device-event output barrier

`GGML_TP5_DW_COMPOSE_EVENT=1` (exact value `"1"`) changes how the TP5 compose
runtime (`fork/tp5-compose/src/runtime.cpp`) completes a decode epoch on the
host. With the gate ON, in relay mode, the per-epoch scheduler-wide
`llama_synchronize` / synchronizing logits getter is replaced by device events
recorded on the child streams after the asynchronous output copy. Every other
value — unset, `0`, `2`, `1x`, `" 1"`, `10` — keeps the stock path: no
allocation, no event, no record/wait/synchronize call, no receipt, and no new
branch body executed.

There is **no cross-window wait** in this design. Each window owns a private
Meta relay provider, its own context, KV cache and requests; only the model
weights are shared, and they are read-only. No writable state crosses windows,
so a symmetric per-epoch peer event wait would be a same-phase rendezvous with
no data dependency. The earlier peer layer (per-window event created from the
Meta device, per-epoch record + release publication, first-child
`event_wait`, and the teardown blocking `event_synchronize`) was removed; the
epoch's only real cross-boundary dependency is this window's own logits D2H
copy.

## What is replaced

Stock per epoch, in each window's decode worker:

1. `llama_decode(ctx, batch)` — submits the graph, then (inside the same call)
   issues the asynchronous logits D2H copy with
   `ggml_backend_tensor_get_async`, which for a Meta output enqueues one copy per
   contributing child stream.
2. `llama_synchronize(ctx)` (or `llama_get_logits_ith`, which synchronizes
   internally) — `ggml_backend_sched_synchronize`, i.e. a host synchronization
   of every backend in the scheduler, which for a relay window means all five
   rank children.
3. The relay dual-completion drain (device tail + CPU courier retirement).
4. Read the logits row on the host.

Gate ON keeps 1, 3 and 4 and replaces 2 with two steps:

- `llama_tp5_output_copy_record(ctx)`, called immediately after `llama_decode`
  returned. `ggml_backend_meta_output_record` resolves the same split state the
  D2H readback uses and records one event per **exactly** the child streams that
  copy touched (the first child for a mirrored tensor, every contributing child
  for a split axis), through each child backend's `event_record` on that child's
  own device. Each record is ordered behind that stream's copy.
- `llama_tp5_output_event_wait(ctx)`, deliberately deferred until after the relay
  drain and immediately before the host reads logits. It
  `cudaEventSynchronize`s the recorded events, releases them, and runs the
  context's `synchronize` bookkeeping — the bookkeeping only when at least one
  event was actually synchronized. Rows are then resolved with
  `llama_get_logits_ith_no_sync`, which performs only the host-side output
  reorder and row resolution.

`ggml` reports how many streams the record covered. The runtime requires the
wait to release exactly that many, so a record/wait mismatch is a hard `require`
failure rather than a silent read of incomplete data.

## Why the events are a complete barrier for the logits

The relay's `finished` / `retired` counters are **not** used for this and cannot
be: the D2H copy is issued by the scheduler inside `llama_decode`, after the
relay provider recorded its per-lane tail events. The chain is instead:

1. `ggml_backend_meta_output_record(backend, tensor)` computes the split state
   with the same internal resolver the readback uses (compute tensors derive
   their state from their sources, so the name-based device callback is not an
   equivalent query). MIRRORED reads the first child alone; a split axis copies
   from every child with a non-empty chunk. PARTIAL/unknown output is refused.
2. The target set is the readback's own target set, by construction:
   `ggml_backend_meta_get_tensor_async` (mirrored) calls
   `ggml_backend_tensor_get_async` on `ggml_backend_meta_simple_backend(0)`, and
   its split-axis branch calls `ggml_backend_tensor_get_2d_async` on
   `ggml_backend_meta_simple_backend(j)` for every child with
   `simple_tensor->nb[axis+1] != 0`. `output_record` uses the identical
   predicate and the identical backend objects (`backend_configs[j].backend`).
3. Stream identity. Both CUDA copy paths (`ggml_backend_cuda_get_tensor_async`
   and `..._get_tensor_2d_async`) use `cuda_ctx->stream()`, and
   `ggml_backend_cuda_event_record` uses the same `cuda_ctx->stream()`. In relay
   mode that stream is provably the only stream of the child: strict relay
   requires `curr_stream_no == 0` and rejects any other existing
   `ctx->streams[device][s]` (`ggml/src/ggml-cuda/relay.hip`), the relay lane
   count is one, the lane stream is `ctx->streams[device][0]`, and
   `ggml_backend_cuda_graph_compute` fails relay work when `curr_stream_no != 0`.
   The copy, the record and the relay lane therefore share one stream, and the
   record is ordered behind the copy and behind that rank's relay work.
4. `cudaEventSynchronize` on those events therefore proves every copy of this
   epoch completed and the host logits rows are readable.

The readback backend is the Meta backend itself, not a child: `llama_context`
requires a single scheduler split in relay mode and rejects any non-Meta compute
node (`src/llama-context.cpp`, `tp5-relay requires a single native Meta graph` /
`tp5-relay refuses non-Meta compute node`).

If any step cannot be satisfied — non-relay context, no graph, host-resident or
PARTIAL output, a missing child event interface, gate off — the record returns
0, no event is left allocated, the ON path keeps the stock synchronize, and the
epoch is counted in `fallback_no_output_barrier`.

## Meta event interface: deliberately absent

The Meta device keeps stock `nullptr` `event_new` / `event_free` /
`event_synchronize`, and the Meta backend keeps stock `nullptr` `event_record` /
`event_wait`. This is a correctness requirement, not an omission:

- A non-null **Meta-device** `event_new` makes `ggml_backend_sched` allocate and
  use its own events for every Meta backend in the process
  (`ggml/src/ggml-backend.cpp`: `ggml_backend_event_new(backends[b]->device)`),
  replacing its stock `ggml_backend_synchronize` fallback. That is a
  process-wide synchronization change no gate here can scope.
- A Meta-**backend** `event_record` / `event_wait` forwards to child 0 only, so
  it can never order all five relay ranks; using it as a scheduler barrier would
  be incomplete.

The output barrier therefore calls the contributing child devices directly and
is confined to `GGML_TP5_DW_COMPOSE_EVENT`. `ggml_backend_meta_first_child_device`
was removed with the peer layer because it had no remaining consumer in the
tree.

## Ownership, ordering and lifecycle

- The events are owned by the Meta backend context (`output_copy_events`), one
  slot per child. They are created lazily by `ggml_backend_meta_output_record`,
  and every successful record is released by the paired
  `ggml_backend_meta_output_wait` (synchronize + free + null).
- A record is issued **after** `llama_decode` returned, on the same child stream
  the D2H copy was enqueued on, so the wait is ordered behind the copy.
- If an epoch is abandoned between record and wait (a throw on the error/abort
  path), its events are still non-null at the next record. `output_record`
  releases those stale events without waiting: nothing consumed their logits,
  destroying a recorded event does not cancel stream work, and the next decode's
  relay `process_ubatch` synchronize drains the stream. This keeps a later,
  differently-shaped epoch's wait count exact.
- `ggml_backend_meta_free` releases anything still owned when the context is
  destroyed; if the relay retains its native owners, the whole Meta owner is
  retained instead, and the events are not freed behind a live reader.
- The runtime's teardown keeps its two `llama_synchronize` calls per window
  around the host-only `seq_rm`; they are lifecycle boundaries, not part of the
  replaced per-epoch barrier.

## What is still a host wait

`llama_tp5_output_event_wait` blocks the host on the recorded device events. It
replaces a scheduler-wide synchronize (every backend, every rank stream, plus
the CPU backend) with exactly the streams the output copy used, and its coverage
is proven by the chain above. The relay dual-completion drain is unchanged and
remains the runtime's only exit predicate; the epoch's completion still requires
the device tail (`status.finished`, driven by real per-lane HIP tail events) and
CPU courier retirement.

## Receipt

With the gate ON the runtime prints one `TP5_DW_COMPOSE_EVENT window=…` line per
window:

|Field|Meaning|
|---|---|
|`epochs`|the window's decode epochs (same-work baseline; equals `decode_calls`)|
|`output_event_syncs`|epochs whose host barrier was the recorded output-copy events|
|`host_sync_replaced`|same count under the contract name (one stock `llama_synchronize` skipped each)|
|`fallback_no_output_barrier`|ON-path epochs that kept the stock synchronize because no sound output-copy barrier was available|
|`no_sync_logits_calls`|logits rows resolved without a synchronizing getter|

`output_event_syncs == host_sync_replaced` and
`output_event_syncs + fallback_no_output_barrier == epochs` are structural
identities of the current code. They are host call counts: they are not
device-completion evidence, they do not prove the hardware stream mapping, and
an absent receipt is not a zero count. The removed peer counters
(`record_calls`, `wait_calls`, `sync_calls`, `free_calls`, …) no longer exist;
the peer wait never received a receipt of its own here.

## Verification status and limits

Implemented and reviewed statically. Under the current work rules no
compilation, syntax check, mock, fixture, harness or GPU run was executed, so
"compiles", "links" and "works on device" are not claimed.

What the source shows (still not a hardware result): the target set of
`output_record` is the readback's target set (same resolver, same predicate, same
child backend objects); the copy and the record use the same `cuda_ctx->stream()`;
and strict relay admits exactly one stream per child with `curr_stream_no == 0`,
so copy, record and relay lane share it; every missing precondition returns 0 and
keeps the stock synchronize.

What remains **unproven** and must not be asserted as guaranteed:

- that the reference run's logits tensor actually resolves to a split state with
  a non-empty target set — a mismatch makes the record return 0, which keeps the
  run correct but silently disables the replacement (it would show up as
  `output_event_syncs == 0`, `fallback_no_output_barrier == epochs`);
- that HIP event capture/synchronize semantics on the installed ROCm version
  match the CUDA contract used in the reasoning, in particular that
  `cudaEventSynchronize` on an event recorded after an async D2H copy proves that
  copy completed (no device trace exists);
- that no other component of the relay provider reorders or defers the copy on
  the same stream outside the reviewed code paths;
- that re-recording a single event object per epoch across a long run and freeing
  it from a different thread of control is race-free on the target HIP runtime —
  the code keeps one record/wait window per window worker, but that is a design
  argument, not a measured property;
- deadlock freedom, latency, or any overlap/time benefit. `wait_*` and
  `host_sync_*` are call counts; the deferral past the drain is a placement, not
  a measured saving.

Acceptance therefore still requires the fork RUNBOOK arms: stock, gate OFF, and
`GGML_TP5_DW_COMPOSE_EVENT=1` with two windows in relay mode, compared token by
token and logits byte for byte, checking that the receipt reports
`output_event_syncs == host_sync_replaced > 0`,
`fallback_no_output_barrier == 0`, and that OFF matches stock byte for byte.
