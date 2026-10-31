// INSTRUMENTATION-INTERFACE.md — correlation contract for the 12022 driver
//
// Driver : driver/nout_floor1_driver.cpp (this repo)
// Runtime: instrumented llama_context build — an INDEPENDENT COPY of the
//          floor1 tree (tree-patched) with nout-instrument-v1.patch applied;
//          the original tree and the instrumentation worker's worktree are
//          never built from or modified
// Merge key: (pid, ctx, step)
//   pid      = getpid() of the driver process
//   ctx      = llama_nout_instrument_context_id(ctx) — the runtime records
//              the same id in process_ubatch; the driver tags each arm
//              ("OFF"/"ON") via llama_nout_instrument_set_tag right after
//              creation and records ctx_id in every step event
//   request  = 1-based request ordinal
//   step     = 1-based decode ordinal WITHIN the request; EXPECTED to equal
//              the runtime's dec= counter per context (NOUT1_FLOOR_TRACE,
//              ON arm only) — but event order is authoritative, the counter
//              is not assumed (see Runtime-side contract item 1)
//
// The LLAMA_NOUT_INSTR instrumentation patch must be applied to the PATCHED
// COPY of the tree (tree-patched) BEFORE building libllama and this driver;
// the driver links against that build and calls
// llama_nout_instrument_set_tag on every context it creates. The original
// tree is never patched in place.
//

## Driver event schema (JSONL, one object per line, via --event-log PATH; '-' = stdout, which is the DEFAULT sink)

Every event carries: pid, ts_us, plus:

| run_meta     | model, n_vocab, n_ctx, n_batch, n_ubatch, n_seq_max, threads, grid_tokens, k, gen, requests, seed, tol, probe_empty_rows |
| context_new  | arm, n_ctx, n_batch, n_ubatch, n_seq_max, n_seq_max_actual, n_threads, flash_attn, kv_unified, n_samplers, gpu_offload_supported |
| context_new_failed | arm (context creation returned nullptr; run exits 3, no summary) |
| model_load_failed | path (run exits 3) |
| request_begin| request, seq, steps (no arm: both arms execute the same request) |
| step_begin   | arm, request, step, phase, n_tokens, n_out, pos0, ctx_id      |
| decode_error | arm, request, step, ret (decode/API failure only; the run aborts — there is NO success-side decode event, a clean step just proceeds to step_end) |
| row_missing  | arm, request, step, row (a flagged row was unreadable; run FAILs) |
| step_end     | arm, request, step, n_out_seen, n_kv_est, bucket_crossed, trace_seen, trace_reuse, trace_miss, trace_dec, trace_n_out, trace_stored |
| row_probe    | arm, request, step, row, non_null (--probe-empty-rows only, 0-output steps) |
| kv_bucket_cross | arm, request, step, n_kv_est (recorded; never a verdict by itself) |
| kv_cleanup   | arm, request, seq, removed                                    |
| compare      | request, step, n_out, rows_off, rows_on, rows_match, argmax_ok, finite, bitexact, max_abs, max_rel, tol, pass |
| temporal_check | request, req_step, step, phase, rows_eq, argmax_eq, checksum_eq (request 3 only) |
| sampler_phase | arm, request, action (add/replace/remove/remove_noop), accepted |
| sampler_phase_skipped | reason (replace is skipped when the add was rejected on either arm) |
| request_end  | request, steps, mismatches (no arm)                           |
| summary      | steps, compares, mismatches, gen_div, decode_err, api_err, row_probe_bad, bucket_cross, kv_rm_miss, on_trace_dec, on_trace_reuse, on_builds, on_builds_fb, on_bind_zero, on_compute_skip, on_gen, pass |

Note: run_meta was malformed before the driver's run_meta fix (repair-driver:
literal `"model":,` plus an extra "" argument shifting every later field);
the field list above is the post-fix target, as is context_new's
n_seq_max_actual — re-verify both against a post-edit driver snapshot before
relying on them.
## Runtime-side contract (confirmed with the instrumentation worker)

The instrumented build writes one JSON object per line (single write() on an
O_APPEND fd) to the file named by LLAMA_NOUT_INSTR; filter strictly by pid+ctx.

1. `step` is the per-context process_ubatch ordinal, 1-based. With the
   driver's exactly-one-ubatch-per-decode guarantee it coincides with the
   decode ordinal; event order is authoritative (throw paths can consume a
   step, so the trace dec= counter is NOT assumed).
2. Out-of-band events carry NO step field (ctx_init, ctx_free, lane,
   reserve_build) — treat as step 0.
3. step_outcome is emitted only after graph_compute returns; decision is one
   of {hit, fresh, fallback, failed_compute}. failed_compute (graph_compute
   failed) is never a hit (c.compute_fails); c.hits/c.hits_narrowed count
   only post-success final hits. The OFF arm receives full instrumentation
   (it is independent of the floor1 lane).
4. step_end carries status (success|failed|alloc_failed|aborted|unknown) +
   n_kv + a counters snapshot. A step with no step_end, or two step_outcomes
   for one step, is an instrumentation error: fail the run.
5. Miss-reason fine buckets (first/ubatch/outputs/samplers/cparams/inputs)
   exist only with LLAMA_NOUT_FLOOR1_TRACE=1 on the ON arm; otherwise
   reason="miss_untraced" — not evidence.
6. If no ctx_init appears at all, the run is INVALID (env unset or the event
   file could not be opened).

Correlation rule: the driver's (arm, request, step) events merge 1:1 with the
instrumentation's (pid, ctx, step) events — identical call order by
construction (single thread, sequential decode); the driver's ctx_id field
carries llama_nout_instrument_context_id(ctx).

Two channels, do not conflate: the LLAMA_NOUT_INSTR JSONL fires on BOTH arms
(the step guard is constructed unconditionally at the top of
process_ubatch; only the env switch gates it). The NOUT1_FLOOR_TRACE stdout
stats fire on the ON arm ONLY (the OFF arm's floor1 lane is disarmed and
stays silent). OFF-arm JSONL events are expected; OFF-arm stdout trace lines
are not.
## Output comparison rules (report:60)

- Per token index: llama_get_logits_ith returns the full vocab row; the driver
  compares OFF vs ON rows bit-exact on the single-thread CPU path.
- 0-output steps read NO logits rows (report:60); --probe-empty-rows asserts
  row 0 is unreadable there instead.
- Missing rows fail the step (row_missing, always enforced). Extra rows are
  NOT probed: the driver reads exactly the n_out flagged rows and never
  tests whether row n_out is readable when n_out > 0; the only extra-row
  evidence is the 0-output probe (row 0), so "no extra rows" on grid/k
  steps is unverified — do not claim it.
- Tolerance (--tol F) relaxes bit-exactness to max-abs diff F with per-element
  max abs/rel reported; it never widens post hoc (report:60).

## Build (13022 only — 12022 forbids compilation)

    # BLOCKED until 13022's serial-lock path/protocol is confirmed (lock
    # held before ANY command; nothing runs on 12022) and a real GGUF
    # model is staged. CPU-only: no GPU on either host.
    # Original tree is NEVER patched in place: copy first, patch the copy.
    cp -a /home/kunweiz/work/lead-nout-reuse-20261030/tree \
          /home/kunweiz/work/lead-nout-reuse-20261030/tree-patched
    cd /home/kunweiz/work/lead-nout-reuse-20261030/tree-patched
    git apply --check /home/kunweiz/work/lead-nout-instrument-20261030/nout-instrument-v1.patch
    git apply /home/kunweiz/work/lead-nout-instrument-20261030/nout-instrument-v1.patch
    TREE=/home/kunweiz/work/lead-nout-reuse-20261030/tree-patched
    BUILD=/home/kunweiz/work/lead-nout-driver-20261030/build
    g++ -std=c++17 -O2 -Wall -Wextra -I"$TREE"/include driver/nout_floor1_driver.cpp \
        -L"$BUILD" -lllama -o "$BUILD"/nout_floor1_driver
    ldd "$BUILD"/nout_floor1_driver   # record actual dynamic resolution

libllama.so is built on 13022 from tree-patched (the patched COPY, never
the original tree) with the latest nout-instrument-v1.patch applied, then
staged into $BUILD; the driver binary is written to $BUILD too, never into
any source tree. LLAMA_NOUT_INSTR is set to an absolute path before the
run; the driver records ctx_id on every step event and tags each context
OFF/ON via llama_nout_instrument_set_tag.
Missing dependencies until the executor provides them: the 13022
serial-lock protocol (BLOCKED until confirmed); a real GGUF model;
libllama built on 13022 from tree-patched.
