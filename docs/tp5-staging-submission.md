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
  slots, and returns `GGML_STATUS_FAILED`. Later computes fail without
  submitting; `ggml_backend_sched_reset` and `reserve` are refused while the
  fault is pending.
- `ggml_backend_sched_free` checks `submission_pending` on **every** backend,
  regardless of whether that scheduler observed the fault itself. If any is
  pending, the copy slots and their buffers are retained. The process owns them
  until exit.
- The Meta allreduce/reduce path checks the return value of every
  `ggml_backend_tensor_copy_async` call. A false result stops before building or
  launching the reduce node that would consume an unproven destination; the
  Meta `graph_compute` returns `GGML_STATUS_FAILED`. There is no `(void)`
  caller left.
- staging faults do not abort: sync paths stop in place and the async path
  returns to the scheduler. The pool quarantines the pinned slot, records the
  unknown destinations (`TP5_STAGING_UNKNOWN` in the diagnostic dump), and
  keeps the first native error.

## Ownership and lifetime

- **Scheduler input copies**: the scheduler owns the copy tensors and retains
  them while any backend reports pending. The caller must not free the
  scheduler, its backends or the destination graph/context while pending.
- **Meta owner**: `ggml_backend_meta_free` retains the whole Meta owner
  (context, child backends, buffers) while `submission_pending` is true instead
  of freeing behind an unknown child-buffer destination. This is retention, not
  a drain.
- **Public helper callers**: `ggml_backend_tensor_copy_async` does not retain
  anything. On `false` the caller is the owner of the source, destination and
  both backends until process exit; there is no public release operation for an
  unknown submission.
- The process exit is the final owner. No API claims native completion.

## Limits

- No drain is claimed. `submission_pending` stays true for unknown work; the
  destination and staging are retained, not proven empty.
- The submit gate (`enter_submit`/`leave_submit`) is the only synchronization
  boundary between a fallback copy and the fatal publication. It does not make
  an already queued native copy complete.
- The capability scope is global, not per backend/context. A fault in one Meta
  context makes every Meta backend report pending/error; a clean context's
  `ggml_backend_sched_free` therefore retains its copy slots too. This is a
  fail-stop policy, not per-context isolation.
- Freeing a backend/context behind an unknown submission is a contract
  violation, not something the API can prevent for a caller that ignores the
  pending capability.

## Tests

```
# staging pool (standalone)
g++ -std=c++17 -O2 -pthread -I ggml/include -I ggml/src \
    tests/test-tp5-staging-pool.cpp -o /tmp/test-tp5-staging-pool
/tmp/test-tp5-staging-pool

# scheduler boundary (ctest)
cmake -S . -B build-ws4-sched-ctest -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build build-ws4-sched-ctest -j --target test-tp5-staging-scheduler
ctest --test-dir build-ws4-sched-ctest -R test-tp5-staging-scheduler --output-on-failure
```
