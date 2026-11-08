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

Only the meta backend implements them today (from the staging pool state).
Other backends report `0`/`false`.

## Semantics

- `ggml_backend_tensor_copy_async` returns `bool`:
  - `true`: accepted and ordered, or performed synchronously.
  - `false`: rejected with no side effect (the helper performed the blocking
    copy), or accepted with an unprovable completion. In the second case the
    caller must not replay the destination.
- The scheduler queries `submission_error` after an accepted
  `cpy_tensor_async`. A nonzero answer stops the split, keeps the input copy
  slots, and returns `GGML_STATUS_FAILED`. Later computes fail without
  submitting; `ggml_backend_sched_reset` and `reserve` are refused while the
  fault is pending.
- `ggml_backend_sched_free` refuses to free while any backend reports
  `submission_pending`. The process owns the destination and staging until
  exit.
- staging faults do not abort: sync paths stop in place and the async path
  returns to the scheduler. The pool quarantines the pinned slot, records the
  unknown destinations (`TP5_STAGING_UNKNOWN` in the diagnostic dump), and
  keeps the first native error.

## Limits

- No drain is claimed. `submission_pending` stays true for unknown work; the
  destination and staging are retained, not proven empty.
- The submit gate (`enter_submit`/`leave_submit`) is the only synchronization
  boundary between a fallback copy and the fatal publication. It does not make
  an already queued native copy complete.
- Scheduler behavior tests are owned by the scheduler test task; this change
  ships the pool boundary tests only.

## Tests

```
g++ -std=c++17 -O2 -pthread -I ggml/include -I ggml/src \
    tests/test-tp5-staging-pool.cpp -o /tmp/test-tp5-staging-pool
/tmp/test-tp5-staging-pool
```
