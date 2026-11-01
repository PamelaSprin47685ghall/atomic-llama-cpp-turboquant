# N3 verify logits — interface, evidence, and what is still open

Written 2026-10-31 for the 13022 MTP lane. Everything below is a reading of the source on this
host; nothing here was built, run, or measured by the author.

## 1. Where the qualified row-tail / MTP run lives

| item | path |
|---|---|
| qualified configuration (ON 56.77 / 58.93 tok/s) | `$E/mtp-hc-row-tail-candidate-20261025/` |
| its sources (4 files) and build receipt | `.../sources/`, `.../build-receipt.json` (base = `mtp-hc-row-epilogue-candidate-20261025`) |
| its binary | `.../install/bin/tq-tp5-compose` |
| its formal command + environment | `.../inputs/command.json` |
| the N3 loop it runs | `$E/mtp-async-protocol-candidate-20261027/runtime.cpp` (verify block, acceptance, rollback) |
| the MTP graph it executes | `src/models/qwen4exp.cpp` → `llama_model_qwen4exp::graph_mtp` |
| the trunk tree this lane builds on | `/home/kunweiz/work/merge-fa-20261031` @ `03024e721` |

`$E` = `/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024`.

The row-tail lane is a **build lane over a different tree**: it carries `qwen4exp.cpp`,
`llama-model.cpp`, `ggml-cuda.cu`, `binbcast.cu` and a receipt whose base is another lane, not
the merge tree. The two share the model and the runtime design, not the history.

## 2. How the trunk + MTP model is loaded

* The two trunk shards carry 48 blocks and **no** `blk.48.nextn.*` and no
  `qwen4exp.nextn_predict_layers`.
* The trained head is a separate draft-only export,
  `$E/mtp-qwen4exp-nextn-self-contained-q2_0.gguf` (31 tensors).
* `work/trunk-mtp-pair-20261031/pair_mtp.py` attaches it as a **third shard** and edits only
  metadata: `block_count 48 → 49`, `nextn_predict_layers 0 → 1`, one trailing element in
  `attention.compress_ratios` and `attention.recurrent_layers`, `split.count 2 → 3`,
  `split.tensors.count 1224 → 1255`, `general.alignment 32`. Payload bytes are never re-encoded
  (FICLONERANGE / copy_file_range), the trunk files are opened read-only and fingerprinted.
* `qwen4exp.load_arch_hparams` reads `nextn_predict_layers` first, because
  `n_layer() == n_layer_all − n_layer_nextn` drives every per-layer array; `load_arch_tensors`
  marks the tail layer `TENSOR_SKIP` unless `ml.load_mtp` is set, and `graph_mtp` asserts
  `n_layer_nextn == 1` and `hparams.n_embd_out() == hc · n_embd`.
* A **draft context** is a second `llama_context` over the same model with
  `ctx_type = LLAMA_CONTEXT_TYPE_MTP`; its memory is a plain attention cache filtered to the
  NextN layer (`create_memory`, `mtp_on_hybrid_qwen` branch).
* After this lane's change a **default context** of the same model filters its attention cache
  to the trunk layers (`il < n_layer()`), so no context allocates a slot set for a layer its
  graph never touches.

## 3. Why the target logits are missing today (three independent reasons)

1. **`tq-tp5-compose` has no speculative path.** `fork/tp5-compose/src/runtime.cpp` pushes one
   input token per active request per step (`batch.push(input, consumed[local], local, output)`)
   and the container writes one full-vocabulary row per **emitted** token
   (`main.cpp`, `write_logits`). Its widest batch is therefore N1-per-request.
2. **The N3 lane keeps its verify logits in memory.** `runtime.cpp` reads
   `llama_get_logits_ith(ctx, k)` for `k < 1 + drafts` for acceptance and for the rollback KL
   check, then discards them; nothing is written, and the report has no field for a verify row.
3. **The trunk tree refused the model outright.** Before this lane:
   * `llama-context.cpp` rejected any Swift profile context with `n_layer_nextn != 0`;
   * `GGML_TP5_SHARED_EXPERT_SHARD` and `GGML_TP5_HC_CONTEXT_OWNER` — both armed by the
     qualified row-tail command — required `n_layer_all == 48 && n_layer_nextn == 0`.

## 4. The interface this lane adds

```c
int llama_decode_n3_verify_logits(
        struct llama_context * ctx,          // default (target) context
        llama_seq_id           seq_id,
        llama_pos              p,            // rows are p, p+1, p+2 of that one sequence
        const llama_token      tokens[3],    // current, draft 0, draft 1
        const int32_t          needs_logits[3],   // NULL copies all three
        float * const          out_logits[3]);    // n_vocab floats each
```

* It executes the N3 graph shape (3 rows, 1 sequence, consecutive positions, every row output)
  through the ordinary `llama_decode`, then copies the rows of **that** graph after it retired.
* The copies never feed acceptance, sampling or the next step: `N3_CAPTURE=0` and `N3_CAPTURE=1`
  runs of the same input must produce identical trajectories, which is gate G3 in the lane.
* It refuses: `n_layer_nextn != 1`, an MTP context, a row count outside `[2,3]`, a sequence or
  position outside the context, a vocabulary violation, a row that wants logits with no
  destination, and a context that does not export **unmasked** NextN residuals (the draft head's
  input). A relay context whose `n_ubatch` is smaller than the row count is refused too.
* A failed decode leaves the memory modules as `llama_decode` leaves them (the failed ubatch's
  positions are removed).

## 5. The N1 reference, and why it is not a plain N1 decode

A one-row decode of the same sequence at the same position, later in time, is **not** the same
state: the GDN recurrent cell, the PLE n-gram history (`ple_ngram_size = 3`, so the history
depends on the two preceding tokens) and the indexer frontier all differ. The harness therefore:

1. snapshots the target's **full sequence state** (`llama_state_seq_get_data_ext`, which carries
   attention KV, GDN state, PLE history and the indexer cache) **before** the N3 step;
2. restores it into a third context of the same window and decodes the same rows **one at a
   time** (1×1 ubatches);
3. discards that context's state; the target is never touched between the snapshot and the step.

Rows are labelled `(window, seq, step, row, input_position, input_token)` on both sides, so the
gate aligns them exactly and refuses any misalignment instead of repairing it.

## 6. Blockers that remain (stated, not worked around)

* **B1 — no comparison against the qualified row-tail binary.** That binary is built from a
  different tree (§1). This lane proves the merge tree's N3 shape; it says nothing about the
  56.8 tok/s lane until the row-tail patches are re-applied on top of this tree, which is a
  separate job.
* **B2 — nothing here is measured.** No build, no run, no GPU by the author. Every gate in
  `patch/result.py` is `UNPROVEN` until Devops runs `patch/start.sh` on 13022.
* **B3 — the harness is not wired into the lane runtime.** `runtime.cpp`'s verify step still
  reads its rows directly. Publishing them in the container is the follow-up.
* **B4 — no independent implementation.** The N1 side is the same kernels at a different batch
  shape; it is a shape reference, not a second implementation of the model.
* **B5 — no timing.** `no_perf` is set; a capture leg is evidence for numerics only.
