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

## Semantics

- `ggml_backend_tensor_copy_async` returns `bool`:
  - `true`: accepted and ordered, or completed synchronously. A rejected async
    copy also returns `true`: the helper synchronizes both backends and performs
    the blocking copy.
  - `false`: **only** the accepted-but-unprovable case. The destination must
    not be replayed, freed or overwritten, and the source, destination and both
    backends must stay alive until the process/context owner resolves it.
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
  there. `ggml_backend_tensor_copy_async` itself returns false for either
  capability answer, including pending with no error.
- The scheduler's input-copy scope does not flush on an unknown return path,
  whether the fault was observed at the accepted copy or at a failed split
  compute. The pre-copy slot wait is separate; no additional drain is
  performed after the fault.
- Staging faults do not abort: sync paths stop in place and the async path
  returns to the scheduler. The pool quarantines the pinned slot, records the
  unknown destinations (`TP5_STAGING_UNKNOWN` in the diagnostic dump), and
  keeps the first native error.

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
- **llama_context / llama_free**: `llama_free` queries
  `llama_context::has_unproven()` (and the relay status) while every member is
  still alive. If either is pending it does **not** delete the context: it
  logs, calls `retain_for_process()` (detach the caller-owned threadpools,
  increment the model's retained-context count) and returns. The whole context
  and all its members stay allocated for process exit, so the pending copy
  cannot race a member destructor (memory, kv_stream_phase_arena,
  cvec/loras, balloc, sampling, graph results, scheduler, backends).
  `llama_context::~llama_context()` refuses a direct delete of a pending
  context (aborts) because a direct delete cannot keep the external model and
  context-owned buffers alive. On a clean state the destructor frees the
  scheduler **before** the backend members are destroyed, matching the actual
  reverse member order.
- **Model**: `llama_model` carries `n_retained_contexts`. Every retained
  context increments it; `llama_model_free` defers and logs instead of
  deleting while it is nonzero. The model, its pimpl buffers and the retained
  context are process-owned; the caller must not use the freed pointer but the
  underlying model is not deleted. There is no release operation for a
  retained context by design.
- **Meta owner**: `ggml_backend_meta_free` retains the whole Meta owner
  (context, child backends, buffers) while `submission_pending` is true.
- **Public helper callers**: `ggml_backend_tensor_copy_async` does not retain
  anything. On `false` the caller is the owner of the source, destination and
  both backends until process exit; there is no public release operation for an
  unknown submission.
- **Threadpools** are caller-owned. `retain_for_process` detaches them so the
  caller may free them; the retained context does not compute again.
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
- A caller that ignores `ggml_backend_sched_has_unproven` and frees the
  scheduler's backends first is a contract violation; the cached free path
  avoids dereferencing them, but it cannot protect a destination the caller
  itself freed.
- Meta owners retained on pending leak intentionally; the process owns them.
- The Meta reduce-path pending gate is not exercised by the CPU tests below:
  it needs a Meta backend with the CUDA staging provider. It mirrors the
  scheduler rule that either capability answer is unproven, but this branch
  records no GPU runtime evidence.

## Tests

```
# scheduler boundary (ctest, CPU)
cmake -S . -B build-ws4-sched-ctest -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build build-ws4-sched-ctest -j --target test-tp5-staging-scheduler
ctest --test-dir build-ws4-sched-ctest -R test-tp5-staging-scheduler --output-on-failure

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

# staging pool (standalone)
g++ -std=c++17 -O2 -pthread -I ggml/include -I ggml/src \
    tests/test-tp5-staging-pool.cpp -o /tmp/test-tp5-staging-pool
/tmp/test-tp5-staging-pool
```

The scheduler test covers: success, rejected, fatal (no replay), a pending
state without a local failure, and pending published by an accepted copy with
no error. It checks the public helper true/false contract for error, pending
and rejected copies; the pending entry gates (`alloc_graph`, `reserve`,
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

The context-retain test covers: `llama_model_free` deferral while a retained
context exists, a clean model free, and (with `LLAMA_TEST_MODEL`) a real CPU
context marked unproven that `llama_free` retains whole, followed by a
deferred model free. Without a model it reports `context-skip` and still
passes the model-retention checks.

Remaining limits, not claimed closed:

- The Meta reduce-path error/pending gate has no runtime test here. It needs a
  Meta backend with the CUDA staging provider and a fault injection; the CPU
  test builds neither.
- There is no drain and no release operation. A pending scheduler, Meta owner,
  context and model remain process-owned; the process exit is the final owner.
- A backend that reports neither capability cannot report an unknown
  submission; the scheduler has no way to detect one.
- The retained process-owned allocations are intentional. Tests that force a
  pending state therefore leave their scheduler and arena allocated; run them
  with `ASAN_OPTIONS=detect_leaks=0`.
