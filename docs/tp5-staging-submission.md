# TP5 staging submission boundary

The persistent H2D staging pool can fail after a copy was accepted. The
completion is then unknown: the destination must not be replayed and the
resources must stay alive. This note records the delivered boundary and its
limits.

## Capability negotiation

The backend exposes two optional functions through the existing
`ggml_backend_reg_i::get_proc_address` hook. The backend iface layout does not
change, so an older plugin cannot be read past its end.

- `ggml_backend_submission_error(ggml_backend_t) -> int`
  - `0`: clean.
  - nonzero: first native error of an unknown submission.
- `ggml_backend_submission_pending(ggml_backend_t) -> bool`
  - true while the destination may still be in flight.
  - never clears for an unprovable submission.

Only the meta backend implements them today, from the **process-global**
staging pool. The functions ignore the backend argument: one staging fault in
one Meta context makes every Meta backend report the error/pending state until
process exit. Other backends report `0`/`false`.

The CUDA backend's host-only staging `api` table is versioned. Version 2 added
`mark_partial` after the complete v1 prefix; version 3 appends `kind_count`
after `mark_partial`. A v1 table cannot publish a partial submission; a v2
table cannot advertise its kind domain. Both are **rejected at registration**
before any later field is read. A rejected registration does not replace an
existing table: if a v3 table is already registered it remains in use; if
none is registered, staging stays unregistered and the original copy path is
used. This is an explicit mode, not a claim of equivalent performance. A v3
table is accepted only when `mark_partial` is
present and `kind_count` covers the host's full kind domain. Mixed v1/v2/v3
installs are explicitly unsupported; a provider with a smaller kind domain is
rejected instead of receiving kind values it would treat as invalid. The CPU
pool test mirrors the v1 prefix and static-asserts that v3 appends
`kind_count` after `mark_partial`.

## Semantics

- `ggml_backend_tensor_copy_async` returns `bool`:
  - The helper queries the optional error/pending capability on **both**
    backends before it does anything. If either reports an unknown submission,
    it returns `false` and performs no copy. This is what keeps the blocking
    fallback from overwriting a destination an earlier copy may still write.
  - `true`: the async copy was accepted and ordered, or the async callback
    rejected with no effects and the helper completed the blocking copy. The
    helper re-queries both backends before returning `true`; a concurrent
    publication turns the result into `false`.
  - `false`: an unknown submission already existed (no copy was attempted), or
    the copy was accepted but cannot be proven. The destination must not be
    replayed, freed or overwritten, and the source, destination and both
    backends must stay alive until the process/context owner resolves it. The
    helper never clears the state.
- The scheduler queries **both** capabilities after an accepted
  `cpy_tensor_async`. A nonzero error or a pending answer stops the split,
  keeps the input copy slots, and returns `GGML_STATUS_FAILED`. Pending with no
  recorded error is reachable: a staging pool that fails before recording a
  native error reports `pending = true`, `error = 0`.
- `ggml_backend_sched_has_unproven(sched)` queries every backend's error and
  pending state while they are alive, folds either answer into the cached
  state, and returns true. `reset`, `reserve_size`,
  `reserve_size_preserve`, `reserve`, `reserve_min`, `alloc_graph`,
  `release_buffers`, `graph_compute`, `graph_compute_async`,
  `replan_graph`, `update_split_copies`, `refresh_split_copy_views` and
  `set_moe_cache` all refuse while that state is set. `replan_graph` and
  `update_split_copies` return false; `refresh_split_copy_views` and
  `set_moe_cache` log and mutate nothing; the compute entry points return
  `GGML_STATUS_FAILED`.
- A failed split compute queries **every** backend, not only the split that
  failed, because the Meta capability is process-global. This covers a fault
  first observed outside the scheduler copy path. The callback-evaluation
  split path applies the same rule.
- The synchronous `ggml_backend_sched_graph_compute` wrapper checks the state
  after the async call. When unproven it does **not** synchronize and returns
  `GGML_STATUS_FAILED`, even if the async call itself reported success. A
  successful submission without a pending/error answer still synchronizes as
  before.
- `ggml_backend_sched_free` uses the **cached** state only. It never
  dereferences the backends: the caller may destroy them before the scheduler
  (llama_context does), and a probe there would be a use-after-free on a
  normal shutdown.
- The Meta allreduce/reduce path checks both the generic helper result and the
  **Meta backend's own** error/pending state at graph-compute entry, before
  each child copy and before the copy-back. The child CUDA registry does not
  expose the staging capability, so the helper bool alone is not sufficient
  there. When the helper is called with a Meta backend as either operand, its
  pre-check and post-check do query that registry and return false for either
  capability answer, including pending with no error.
- The scheduler's input-copy scope does not flush on an unknown return path,
  whether the fault was observed at the accepted copy or at a failed split
  compute. The pre-copy slot wait is separate; no additional drain is
  performed after the fault.
- A mirrored submission that queues some ranks and then gets a clean rejection
  is not a clean rejection of the whole copy. The pool publishes the same
  fail-stop state as a native fatal: `mark_partial` disables the pool,
  quarantines the slot and records the destination as unknown. `first_error`
  stays `0` because a rejection is not a native error; `submission_pending`
  is the signal, so the async helper returns false and no later compute runs.
  The synchronous setter has no return value: its caller cannot observe the
  partial write, but every later staging submit and Meta graph compute sees
  the fail-stop state. A concurrent duplicate `mark_partial` for the same lease
  keeps the gate closed but does not count the destination twice.
- Staging faults do not abort: sync paths stop in place and the async path
  returns to the scheduler. The pool quarantines the pinned slot, records the
  unknown destinations (`TP5_STAGING_UNKNOWN` in the diagnostic dump), and
  keeps the first native error.

## Callers and source/binary contract

- All live in-repo callers are compiled against the bool declaration:
  - `ggml/src/ggml-backend-meta.cpp` checks the helper result before building
    the reduce node or returning the copy-back status; `false` returns
    `GGML_STATUS_FAILED` and leaves the aux destination allocated.
  - The scheduler input-copy paths call `ggml_backend_i::cpy_tensor_async`
    directly. There, `false` is a clean rejection with no effects and the
    scheduler may perform its blocking copy; it is not the public helper's
    `false`.
  - `tests/test-tp5-staging-scheduler.cpp` asserts both helper outcomes,
    including the sticky refusal before any callback.
    `tests/test-moe-cache.cpp` checks the CPU fallback's `true` result.
- `overlay/mtp-async-protocol-candidate-20261027/` is a frozen provenance
  snapshot of the pre-cutover candidate. Its `void` declaration and ignored
  returns are historical bytes; this tree does not compile them. Do not edit
  them or rebuild them against the current library.
- The return-type change is source-incompatible. External consumers must
  rebuild against the installed header and honor `false`. The symbol keeps its
  name; a caller compiled against the old `void` prototype can invoke it on the
  tested CPU ABI but cannot observe the result. There is no void compatibility
  entry, version gate or runtime API-version probe, so an old external binary's
  behavior is not qualified.

## Synchronous source identity

The scheduler's blocking input fallback (`ggml_backend_tensor_copy(input,
input_cpy)`) reaches the Meta buffer through `set_tensor`, which only receives
the destination. With `n_copies == 1` that destination has no input flag and
its name is `"%s#%s#%d"`, so the synchronous lane cannot classify the original
graph input. The union index names use an exact match and would fall to the
generic key, or be rejected by the 64 KiB envelope when oversized.

The Meta buffer now implements the existing `ggml_backend_buffer_i::cpy_tensor`
slot; no interface field is added. `ggml_backend_tensor_copy` tries it first
only for a host source and a Meta destination. The implementation preflights
the same conditions as `cpy_tensor_async` (host non-Meta source, non-view Meta
destination, same layout, both contiguous, mirrored single segment) and then
calls the synchronous set path with the source tensor as `staging_meta`.
Eligibility and the kind therefore use the source's input flag and build-time
name while the upload still targets the destination. A direct
`ggml_backend_tensor_set` of the decorated destination keeps the old behavior:
the name is not parsed.

A false return is a pure preflight rejection with no side effect, so the
caller may fall back to `set_tensor`. After the first possible child write the
entry returns true even when staging publishes a fatal or partial submission:
the existing sticky `submission_pending`/`submission_error` state is the only
error signal, and the blocking fallback never replays a destination that may
have been partially written. The CPU test `test-tp5-staging-sync-src` covers
the source kind, mirrored bytes, a non-input source, a direct set without
source identity, and a partial mirrored submission that must not be replayed.

## Ownership and lifetime

- **Scheduler input copies**: the scheduler owns the copy tensors and retains
  them while the cached state is unproven. The caller must not free the
  scheduler or its backends while pending.
- **Graph arena and copy metadata**: `replan_graph` keeps the old gallocr,
  tensor addresses and plan sizes; `update_split_copies` and
  `refresh_split_copy_views` change no copy/view metadata. The live split
  graph is not re-bound behind in-flight work.
- **MoE cache session**: `set_moe_cache` keeps the current session and its
  provider resources. It neither destroys nor replaces them while unproven.
- **Staging submit gate**: `enter_submit()` holds `submit_mutex_` across the
  meta fallback's backend set path. A CUDA simple tensor on that path calls
  `pool::note_source()` on the same thread, so note_source must not `try_lock`
  the owned mutex (undefined) and must not take `mutex_` while the gate is
  held: the fatal publication holds `mutex_` and waits for `submit_mutex_`.
  The pool publishes `submit_gate_held_` in `enter_submit()` and clears it in
  `leave_submit()`; note_source reads it, skips, and counts a fast miss. A
  successful try_lock would drop the gate, so the flag is the contract, not a
  hint. The flag alone is not the deadlock proof: a gate can start between
  the flag load and the pool lock (TOCTOU), so note_source uses
  `mutex_.try_lock()` and counts a fast miss when it is contended. It never
  waits for `mutex_`; a blocking lock would close the cycle
  note_source -> mutex_ -> submit gate -> note_source when the gate holder
  waits for the note. The gate flag is rechecked after the try_lock so a gate
  that started in that window still counts as a fast miss.
- **llama_context / llama_free**: `llama_free` queries
  `llama_context::has_unproven()` (and the relay status) while every member is
  still alive. If either is pending it does **not** delete the context: it
  logs, calls `retain_for_process()` (detach the caller-owned threadpools,
  clear the CPU backend's raw threadpool pointer, increment the model's
  retained-context count) and returns. The whole context
  and all its members stay allocated for process exit, so the pending copy
  cannot race a member destructor (memory, kv_stream_phase_arena,
  cvec/loras, balloc, sampling, graph results, scheduler, backends).
  `has_unproven()` covers the main scheduler and every prefab envelope slot
  scheduler (plus the recorded in-flight slot list); a slot-only fault retains
  the whole context too. A duplicate `llama_free` on a retained context is
  ignored and does not increment the count again.
  `llama_context::~llama_context()` refuses a direct delete of a
  scheduler-unproven context (aborts) because a direct delete cannot keep the
  external model and context-owned buffers alive. A relay-only pending context
  is retained by `llama_free`; a direct delete is unsupported and is not
  covered by that abort. On a clean state the destructor frees the scheduler
  **before** the backend members are destroyed, matching the actual reverse
  member order.
- **Model**: `llama_model` carries a live-context reference count and a
  retained-context count. The live count and the free request are read and
  changed under a lifetime mutex; the retained count is atomic and is
  consulted in the same critical sections.
  `llama_init_from_model` acquires one live reference before it reads the
  model. A clean `llama_free` releases it; a retained context never does. A
  `llama_model_free` before `llama_free` therefore defers instead of deleting
  under the live context. The last clean release deletes the model when a free
  was already requested. `llama_model_free` and `release_context()` share the
  mutex, so two free paths cannot both delete. A second free request is
  ignored. A retained pending context and the model are process-owned; there
  is no release operation by design, and the retained context must not be used
  again.
- **External borrows**: LoRA adapters and samplers are caller-owned raw
  pointers stored in the context. A retained context never computes or samples
  again, so no retained member dereferences them. `llama_adapter_lora_free`
  after retention erases the adapter from the model and marks it process-owned
  instead of deleting it, because submitted native work may still read the
  adapter tensors; a duplicate call is ignored. Samplers may be freed after
  `llama_free` returns. Cvec input data is copied into context-owned tensors
  by `llama_set_adapter_cvec`, so it is not borrowed after that call. The
  model-owned LoRA set is released only by the deferred model free.
- **Meta owner**: `ggml_backend_meta_free` retains the whole Meta owner
  (context, child backends, buffers) while `submission_pending` is true.
- **Public helper callers**: `ggml_backend_tensor_copy_async` does not retain
  anything. On `false` the caller is the owner of the source, destination and
  both backends until process exit; there is no public release operation for an
  unknown submission. A pre-existing unproven state returns `false` before any
  copy, so the blocking fallback cannot overwrite a destination behind an
  in-flight copy.
- **Threadpools** are caller-owned. `retain_for_process` detaches them and
  clears the CPU backend's pointer without pausing the pool, so the caller may
  free them; the retained context does not compute again.
- The process exit is the final owner. No API claims native completion. There
  is no operation that clears a cached unproven state.

## Limits

- No drain is claimed. `submission_pending` stays true for unknown work; the
  destination and staging are retained, not proven empty.
- The submit gate (`enter_submit`/`leave_submit`) is the only synchronization
  boundary between a fallback copy and the fatal publication. It does not make
  an already queued native copy complete.
- The capability scope is global, not per backend/context. A fault in one Meta
  context makes every Meta backend report pending/error; a clean context's
  scheduler and llama_context therefore retain too. This is a fail-stop policy,
  not per-context isolation.
- Pending with zero error is a runtime state, not a test-only state: the
  staging pool can be disabled before it records `first_error`. The public
  helper and the scheduler therefore treat pending alone as unproven.
- The helper's pre-check and post-check are point-in-time queries. A fault
  published after the final check but before the caller acts is not observed;
  the capability is sticky, so a later query or scheduler gate sees it, but a
  fallback copy that ran in that window may already have overwritten the
  destination. No lock spans the caller's action.
- The helper observes only the registries of the two backends passed to it. A
  fault exposed only by an unexposed child capability is invisible; the Meta
  reduce path therefore keeps its own Meta-backend check.
- A caller that ignores `ggml_backend_sched_has_unproven` and frees the
  scheduler's backends first is a contract violation; the cached free path
  avoids dereferencing them, but it cannot protect a destination the caller
  itself freed.
- `ggml_backend_submission_pending` means **accepted but unprovable**, not
  ordinary ordered in-flight work. A backend that reports normal in-flight
  work here would make every accepted copy fail-stop and retain the
  context/model. The capability contract is strict for that reason; the Meta
  staging implementation reports only `disabled || unknown_count > 0`.
- The live-context reference protects a model that already has a context.
  `llama_model_free` is not made safe against a concurrent
  `llama_init_from_model` that is creating the first context: destroying the
  model and its lifetime mutex while another thread enters the constructor
  needs external serialization. The caller must serialize creation with the
  final free.
- A second free of the same model or context handle after the first one
  completed cannot be detected without a global handle registry. The retained
  path is idempotent; a clean double free remains caller undefined behavior.
- External borrows (LoRA adapters, samplers) are not reference-counted. They
  are safe to free after retention only because the retained context is never
  used again. Cvec input data is copied, not borrowed. Calling any context API
  after `llama_free` is a caller contract violation.
- The LoRA-adapter borrow arm constructs a bare adapter, registers it on a
  real CPU context, retains the context and then frees the adapter. It proves
  the erase-from-retained-model path, the process-owned retention and the
  duplicate-free path. It also checks the ordinary delete path without a
  retained context. It does not load adapter weights and does not compute
  with them; full LoRA arithmetic is outside this test. Freeing an adapter
  before retention, while the context can still compute, is a use-after-free
  by contract.
- The adapter decision uses the model's retained-context count, not a
  per-adapter reference. Once any context on the model is retained, every
  released adapter is process-owned; this is intentionally conservative.
  The count is atomic and is read without the model lifetime mutex, which is
  safe only under the documented rule that borrows are released after
  `llama_free` returns, not concurrently with model teardown.
- The retained adapter allocation prevents freeing tensors that submitted
  native work may still read; it is not a proof that the in-flight kernel
  finished. CPU tests cannot observe that ordering, and no GPU run is claimed.
- `llama_adapter_lora_free` and the final `llama_model_free` are not
  internally serialized against each other. The caller must not run them
  concurrently; the retained-context path keeps the model alive, so the
  documented free-after-retention order is safe without a new lock.
- Meta owners retained on pending leak intentionally; the process owns them.
- The Meta reduce-path pending gate is not exercised by the CPU tests below:
  it needs a Meta backend with the CUDA staging provider. It mirrors the
  scheduler rule that either capability answer is unproven, but this branch
  records no GPU runtime evidence.
- The prefab-slot-unproven test is model-gated. Without `LLAMA_TEST_MODEL`, or
  when prefab refuses the context, it prints `prefab-skip`. When prefab is
  enabled it pins `LLAMA_PREFAB_ENVELOPES=1` for the constructor (the caller's
  value is restored afterwards), then decodes one token on that exact-fit
  envelope: the slot scheduler is created lazily by that ubatch, and only then
  is it marked unproven. The test proves the slot state (`sched`, `built`, a
  clean `has_unproven`) before it faults the slot.
- A prefab envelope mask has `n_env` rows, but only `ubatch->n_tokens` rows are
  real. The fill therefore takes an explicit real-row count
  (`llama_kv_cache::fill_input_kq_mask`); the stock entry point keeps its
  `n_tps == n_tokens` equality as a stock-layout contract, and the prefab
  wrapper writes the real rows and replicates the last one into the pad rows.
  A regression that reuses the stock row count on the envelope mask would read
  past the ubatch's token arrays, so the row count is an invariant, not an
  assertion to relax.
- The sub-envelope test compares stock, exact-envelope and padded-envelope
  logits. A GGUF whose raw F32 tensors are not finite (the
  `/tmp/min-llama.gguf` generator writes arbitrary bytes, so it contains
  NaN/Inf) cannot support a numeric comparison: the test prints
  `numeric-skip` and falls back to raw-bit agreement with the stock arm. Use a
  finite-weight GGUF (for example `/tmp/min-llama-finite.gguf`) for the
  numeric arm.
- Stock multi-token decode with the non-unified KV cache (`kv_unified=false`,
  the default context parameter, `n_seq_max=1`) still hits the stock
  `n_tps == n_tokens` assertion in `set_input_kq_mask`: the batch allocator
  reports `n_seq_tokens=1` for a 3-token single-sequence batch while the mask
  has one row. The sub-envelope test sets `kv_unified=true` for the stock
  baseline, and compares padded against exact prefab under `kv_unified=false`
  without a stock arm. That stock lane is outside this fix.
- A backend that reports neither capability cannot report an unknown
  submission; the scheduler has no way to detect one.
- The retained process-owned allocations are intentional. Tests that force a
  pending state therefore leave their scheduler and arena allocated; run them
  with `ASAN_OPTIONS=detect_leaks=0`.

## Tests

```
# scheduler boundary (ctest, CPU)
cmake -S . -B build-ws4-sched-ctest -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build build-ws4-sched-ctest -j --target test-tp5-staging-scheduler
ctest --test-dir build-ws4-sched-ctest -R test-tp5-staging-scheduler --output-on-failure

# staging submit-gate / note_source lock ownership (ctest, CPU)
cmake --build build-ws4-sched-ctest -j --target test-tp5-staging-note-source-lock
ctest --test-dir build-ws4-sched-ctest -R test-tp5-staging-note-source-lock --output-on-failure

# Meta partial-enqueue publication (ctest, CPU)
cmake --build build-ws4-sched-ctest -j --target test-tp5-staging-meta-partial
ctest --test-dir build-ws4-sched-ctest -R test-tp5-staging-meta-partial --output-on-failure

# ASAN variant: catches the backends-before-scheduler UAF and the
# release_buffers-while-pending free.
cmake -S . -B build-ws4-sched-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_HIP=OFF \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF \
      -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
      -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
      -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address"
cmake --build build-ws4-sched-asan -j --target test-tp5-staging-scheduler
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-ws4-sched-asan \
      -R test-tp5-staging-scheduler --output-on-failure

# pending-context/model ownership (ctest); the context part needs a model:
#   LLAMA_TEST_MODEL=/path/to/model.gguf ctest -R test-tp5-context-retain
cmake --build build-ws4-sched-ctest -j --target test-tp5-context-retain
LLAMA_TEST_MODEL=/path/to/model.gguf ctest --test-dir build-ws4-sched-ctest \
      -R test-tp5-context-retain --output-on-failure

# prefab sub-envelope mask/logits (ctest); use a finite-weight GGUF for the
# numeric arm. The context-retain fixture (/tmp/min-llama.gguf) still runs the
# raw-bit fallback.
cmake --build build-ws4-sched-ctest -j --target test-tp5-prefab-subenvelope
LLAMA_TEST_MODEL=/path/to/finite-model.gguf ctest --test-dir build-ws4-sched-ctest \
      -R test-tp5-prefab-subenvelope --output-on-failure

# staging pool (standalone)
g++ -std=c++17 -O2 -pthread -I ggml/include -I ggml/src \
    tests/test-tp5-staging-pool.cpp -o /tmp/test-tp5-staging-pool
/tmp/test-tp5-staging-pool
```

The scheduler test covers: success, rejected, fatal (no replay), a pending
state without a local failure, and pending published by an accepted copy with
no error. It checks the public helper true/false contract for error, pending
and rejected copies, the sticky refusal before any callback once a backend is
unproven, and the no-op copy path; the pending entry gates (`alloc_graph`, `reserve`,
`reserve_size`, `release_buffers`, `reset`, `compute`); that a failed split
compute on both the normal and callback paths becomes fail-stop and does not
flush deferred input copies; that the synchronous wrapper skips `synchronize`
and never returns success while unproven; that `replan_graph` leaves tensor
layout, addresses and plan size
unchanged; that `update_split_copies`/`refresh_split_copy_views` change no copy
metadata (observed through the diagnostic dependency view); that
`set_moe_cache` neither destroys nor replaces the session; no input-scope
flush on the unknown path; and the production backends-before-scheduler
destruction order (the fake records any capability probe after `iface.free`).

The staging lock test covers the meta fallback path where `enter_submit()` is
held while the backend set path calls `note_source()` on the same thread. It
checks that note_source does not try_lock the owned gate, does not drop it
while a second entrant waits, and does not take `mutex_` while a fatal
publication is waiting for the gate; the same-thread and concurrent calls are
counted as fast misses. It also constructs the check-then-lock TOCTOU: a
waiter holds `mutex_` inside `event_sync`, note_source sees
`submit_gate_held_=false` and must return without waiting for the pool lock,
then the gate holder waits for the note and the waiter fails into the fatal
publication. The blocking design fails this case with the explicit
`note_source blocked on mutex_ (TOCTOU cycle)` message. The fatal path then
completes after `leave_submit()` with the unknown-work record intact.

The staging pool test covers the slot lifecycle with injected primitives: a
leased slot is never reused, both slots may be pending, a blocking acquire
waits the pending events before reuse, and `acquire(wait_pending=false)` does
not wait. Before reporting `no_slot`, that non-blocking acquire reclaims a
pending slot whose recorded events have already completed; it polls the
events instead of synchronizing them, so the async lane rotates its two slots
without blocking the pipeline. Each pending rank event is tracked separately:
a slot is not reclaimed until every recorded rank event has completed, an
active lease is skipped even when its event has completed, and recording a
reused slot resets that event's completion. The dump reports `reaps` (slots
returned to rotation) and `queries` (backend event-query calls, at most one
per pending rank event examined and stopping at the first not-ready or
error); neither is a latency measurement. A query error is fatal, like a
failed wait. A not-ready query is a state, not a fault, and the adapter does
not clear the runtime's sticky error. A slot whose events are still in flight
stays pending and the async lane falls back, which is the intended transient
collision. The dump splits `no_slot` into the two reachable classes:
`no_slot_leased` (all usable slots for that key were leased) and
`no_slot_pending` (at least one usable unleased slot was still pending and
its events were not all complete). A quarantined/faulted pool is disabled at
acquire entry, so a fault is not counted as `no_slot`; a free slot is chosen
by `pick_free_locked` and a too-small slot fails as `oversize`, so there is
no third class. The dump also reports `leases/no_slot/reaps/queries` per kind
on `TP5_STAGING_KIND` lines. The global `misses` counter remains the
note_source pointer check, not a staging miss. The test pins the split and
the per-kind sums against a deterministic lease collision and a deterministic
pending miss. It also asserts the frozen v1 `api` prefix at compile time
and covers a clean rejection after one rank was queued: `mark_partial` stops
the pool, `submission_pending` is true, `first_error` stays 0, the slot is
quarantined, and later upload/enqueue/wait/enter_submit/acquire calls fail. A
held submit gate blocks `mark_partial` until it is released, and two concurrent
publications for the same lease count one unknown destination and one partial
fault.

`queries` is a backend call count, not a code invariant and not invertible to
the `no_slot` composition: a successful acquire can query an incomplete slot
before reclaiming another, the synchronous path issues no queries, and
`pending_ranks` can hold 1..16 ranks. Use `no_slot_leased`,
`no_slot_pending` and the per-kind counters directly. The earlier
five-rank/five-query inverse applied only to the five-rank normal run; it is
not a code invariant.

The two union index inputs (`union_inp_k_idxs`, `union_inp_v_idxs`) have
dedicated kinds. Both are graph inputs on the same graph; before this change
both classified as generic and could occupy both generic slots. In the async
lane each now has its own double buffer and keeps the 64 KiB per-slot envelope it had as a
generic input, so eligibility is unchanged. Fixed limits: 4 MiB total pinned
pool (`kMaxPoolBytes`), 2 MiB per request (`kMaxStagingBytes`), 2 slots per
(window, kind), and 64 KiB per generic/union slot. The split is not free: it
adds two double buffers, i.e. up to 4 x 64 KiB = 256 KiB of additional pinned
capacity in the active window 0 (up to 16 slot records and 1 MiB across all
four window arrays), plus up to one lazily created event per rank per new slot
(5 active ranks -> up to 20 additional events for the four window-0 slots).
The 4 MiB cap still bounds pinned bytes; the extra event handles and slot
metadata are outside that byte cap. No GPU run has measured whether this
reduces `no_slot`; the CPU test only proves the mapping and per-kind
independence. Reuse still requires every recorded rank event to complete; the
split adds no synchronization and does not reuse a slot earlier.

The CPU test also covers 1/5/16 pending-rank reap query counts, a synchronous
all-leased miss with zero queries, and the four-window union-kind budget
(16 slots x 64 KiB = 1 MiB under a 1 MiB cap; the next growth is oversize,
not no_slot). It also checks that a scheduler-decorated union name stays
generic, that a non-INPUT union input does not stage, that `no_slot` sums to
`no_slot_leased + no_slot_pending`, and that a faulted pool does not count
`no_slot`.

The exact union kinds are async-only in the current scheduler. The async lane
classifies the scheduler's real source tensor (`staging_meta`); the sync lane
classifies the destination copy, named "%s#%s#%d" (ggml-backend.cpp), and for
n_copies==1 that copy is not INPUT. Do not parse the decorated name and do
not relax INPUT: the sync fallback stays generic or unstaged. A future sync
fix needs a scoped source-identity channel from the scheduler; the copy loop
already holds the real source tensor, but the public sync set interface passes
only the destination. The dependency is ggml-backend.cpp's input-copy loop
(batch and non-batch paths) plus a meta set_tensor hook; the benefit is
correct kind classification and the real INPUT flag in the sync lane without
name parsing. Not implemented here.

Per-window staging is not wired. The compose worker's `tl_tp5_window`
(`fork/tp5-compose/src/runtime.cpp`) is a real per-window thread identity,
but it lives in the compose module and is not visible to ggml-base. Passing
it in requires a registered hook; this change does not guess a window and
keeps the existing window-0 behavior. A per-window split needs that hook and
a separate measurement.

The Meta partial test builds a real Meta backend over two CPU child devices and
injects a v3 staging table whose enqueue queues rank 0 and rejects rank 1. It
covers both the synchronous buffer-set lane (`merge_wait` + `enqueue_sync`)
and the asynchronous lane (`acquire_nowait` + `enqueue_async`). It asserts the
partial branch calls `mark_partial` before releasing the lease, that
`submission_pending` becomes true with `first_error == 0`, and that the next
set hits the fatal submit gate without another enqueue. It also checks that
registration rejects v1, rejects a v2 table, rejects a v3 table without
mark_partial, rejects a v3 table whose kind_count is smaller than the host's
kind domain, accepts the complete v3 table and rejects a future version. It
also registers version-only four-byte v1/v2 tables (ASan checks the read
boundary), checks that a rejected table does not replace an already accepted
one, and accepts a v3 table whose kind_count is larger than the host's.

The context-retain test covers, without a model: the live-context reference
(free-before-context deferral, last-release delete, new-context refusal after
a free request), retained-count deferral, and a clean model free. With
`LLAMA_TEST_MODEL` it additionally covers: model-free-before-context on clean
and pending contexts, the CPU backend threadpool pointer cleared by
`retain_for_process`, a sampler borrow freed after retention, a bare LoRA
adapter registered on the context and freed after retention (its stale pointer
is never read by the duplicate free or the deferred model free), a prefab
envelope slot built by a real one-token decode and then marked unproven
retaining the context, a duplicate `llama_free` that keeps the retained count
at one, and a real CPU context retained whole. The model part reports
`context-skip` without a model; the prefab arm reports `prefab-skip` when
prefab refuses the context.

The prefab sub-envelope test covers the mask fill for `n_tokens < n_env`: it
decodes a 3-token batch and two 1-token batches through a stock baseline, an
exact-envelope arm (envelopes 1 and 3) and a padded arm (envelope 4 only). It
requires `n_sub_envelope == 3`, `n_pad_rows == 7` and no stock fallback in the
padded arm, and compares the user-visible logits of every step. Under the
default non-unified cache it repeats the padded/exact comparison inside the
prefab lane, which is the configuration of the retention fixture.

## Rejection diagnostics

With `GGML_TP5_STAGING_DIAG=1`, a rejected MIRRORED candidate used to print one
line per candidate. The weight load alone produced 512 `no_input_flag` lines,
each through unbuffered stderr (FILE lock plus one write syscall), so the
diagnostic added I/O to the load path and buried the signal.

Rejections are counted per `(lane, reason)` with one relaxed atomic increment
on the hot path. Each class embeds a fixed example record (`char[GGML_MAX_NAME]`
name, size, flags, type) and an atomic state EMPTY/WRITING/READY. The single
thread that wins the EMPTY->WRITING compare_exchange writes the record in
place and publishes READY with release. A dump loads the state with acquire
and reads the fields only in READY; WRITING prints `(pending)` and EMPTY prints
`(none)`, so a race never fabricates a consistent snapshot. There is no
allocation and no blocking. The exit dump prints one summary line per class:

```
TP5_STAGING_REJECT lane=sync reason=no_input_flag count=512 example_name=... flags=0x0 size=... type=...
```

`GGML_TP5_STAGING_DIAG_VERBOSE=1` restores the per-candidate trace (capped at
512 lines) for an explicitly requested run. The reason classification
(`null`, `no_input_flag`, `zero_size`, `size`, `slot_size`, `layout`) is
derived from `ggml_cuda_staging::staging_candidate_reason`, which returns the
reason of the same `staging_candidate_decision` that `staging_candidate` reads.
One decision classifies the name once and carries both the kind and the
reason, so the eligibility predicate and the diagnostic cannot drift apart,
and an accepted or slot-rejected candidate is not looked up twice. `slot_size`
is the 64 KiB per-slot envelope shared by the generic and the two union index
kinds, so an oversized `union_inp_k_idxs` or `union_inp_v_idxs` is reported as
`slot_size`, never as `other`. Do not re-derive the rules in the diagnostic: a
second copy of them classified those union rejections as `other`.

Cost: the hot path performs one relaxed increment and no I/O; the example is
written once per class with no allocation. The summary costs at most one line per
`(lane, reason)` at exit and holds no lock while printing. With the flag off,
`staging_log_reject` returns after the cached flag check, as before. This
change touches only `ggml-backend-meta.cpp`; the slot-pressure work owns
`staging.h`/`acquire` and is not modified.

### Example lifetime and dump ordering

The first example of each class is a fixed `char[GGML_MAX_NAME]` copy inside
the class record, written in place by the CAS winner. The source tensor may be
renamed, freed, or its arena reused before the exit dump; the summary still
prints the name captured at rejection time. A test records a rejection,
renames the tensor, releases the buffer, context and backend, and then dumps:
it requires the original name and rejects the mutated one. The same Meta test
records an oversized `union_inp_k_idxs` input and requires
`reason=slot_size` in the dump, so the consumer path, not the predicate in
isolation, proves that union over-limit rejections never surface as `other`.

The dump loads the counter (relaxed) and the state (acquire) without holding a
lock, so it never blocks the rejection path and never holds a lock across
`fprintf`. Memory order: the CAS uses acq_rel on success and relaxed on
failure; the writer publishes with a release store of READY; the reader's
acquire load of READY orders the example fields before their use. A concurrent
first rejection can be in WRITING or not started when the dump reads the
state; the dump then prints `(pending)` or `(none)` for that class while the
count stays exact. The table is a namespace-scope static, so it is alive when
the `atexit` dump runs. The pool dump still takes the pool mutex before the
Meta summary; that ordering is unchanged.

The summary test is registered as `test-tp5-staging-reject-summary` and the
concurrent test as `test-tp5-staging-reject-concurrent` in
`tests/CMakeLists.txt` (labels `cpu;tp5;staging`, CPU only). Both run through
CTest. The concurrent test allows intermediate `(none)`/`(pending)` lines and
asserts only the final dump (READY, exact count, no `(none)`/`(pending)`).

### Integration with the staging union kind

The reason classifier shares the staging constants and name mapping
(`classify_name`, `kMaxStagingBytes`, `kGenericMaxBytes`). When the staging
union-kind change lands, any new rejection condition in `staging_candidate`
must get a reason here; otherwise those candidates fall into `other`, which is
reported and counted instead of `unknown`. Keep the generic 64 KiB overflow
classification aligned with `kGenericMaxBytes` in `staging.h`.
