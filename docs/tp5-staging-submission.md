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
- The scheduler queries `submission_error` after an accepted
  `cpy_tensor_async`. A nonzero answer stops the split, keeps the input copy
  slots, and returns `GGML_STATUS_FAILED`. It also queries a failed split
  compute for the same state, so a fault first observed on the backend side
  still becomes the scheduler's fail-stop state.
- `ggml_backend_sched_has_unproven(sched)` queries every backend while they
  are alive and caches the answer in the scheduler. `reset`,
  `reserve_size`, `reserve_size_preserve`, `reserve`, `alloc_graph`,
  `release_buffers` and `graph_compute` all refuse while that state is set.
- `ggml_backend_sched_free` uses the **cached** state only. It never
  dereferences the backends: the caller may destroy them before the scheduler
  (llama_context does), and a probe there would be a use-after-free on a
  normal shutdown.
- The Meta allreduce/reduce path checks both the generic helper result and the
  **Meta backend's own** `submission_error` before each child copy and before
  the copy-back. The child CUDA registry does not expose the staging
  capability, so the helper bool alone is not sufficient there.
- The scheduler's input-copy scope does not flush on the unknown return path.
  The pre-copy slot wait is separate; no additional drain is performed after
  the fault.
- Staging faults do not abort: sync paths stop in place and the async path
  returns to the scheduler. The pool quarantines the pinned slot, records the
  unknown destinations (`TP5_STAGING_UNKNOWN` in the diagnostic dump), and
  keeps the first native error.

## Ownership and lifetime

- **Scheduler input copies**: the scheduler owns the copy tensors and retains
  them while the cached state is unproven. The caller must not free the
  scheduler or its backends while pending.
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
- The process exit is the final owner. No API claims native completion.

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
- A caller that ignores `ggml_backend_sched_has_unproven` and frees the
  scheduler's backends first is a contract violation; the cached free path
  avoids dereferencing them, but it cannot protect a destination the caller
  itself freed.
- Meta owners retained on pending leak intentionally; the process owns them.

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
state without a local failure, the public helper true/false contract, the
pending entry gates (`alloc_graph`, `reserve`, `reserve_size`,
`release_buffers`, `reset`, `compute`), no input-scope flush on the unknown
path, and the production backends-before-scheduler destruction order (the fake
records any capability probe after `iface.free`).

The context-retain test covers: `llama_model_free` deferral while a retained
context exists, a clean model free, and (with `LLAMA_TEST_MODEL`) a real CPU
context marked unproven that `llama_free` retains whole, followed by a
deferred model free. Without a model it reports `context-skip` and still
passes the model-retention checks.

Known open points from the independent review that are not claimed closed:
`replan_graph`, `refresh_split_copy_views`, `set_moe_cache` and the
synchronous `ggml_backend_sched_graph_compute` wrapper do not all gate on the
unproven state; the compute failure currently checks only the failed split
backend; Meta reduce checks `submission_error` but not `submission_pending`
(pending-with-zero-error is a scheduler-test-only state today).
