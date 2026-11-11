# Conservative Pillar-5 candidate — validation record 2026-11-11

Status: the conservative branch is the measured Pillar-5 candidate for the
delivery gate. The formal 512x2 contract reaches the >=50 tok/s/request target
with P5 OFF; P5 ON is still slower, so the default remains OFF.

## Base and merged commits

Base: `044ab7d04a251a4ce5bccccc9dab1f861ac7e5cf` (the same binary that was
measured at 49-51 tok/s earlier). The branch is `feat/p5-conservative-20261111`
and the measured code commit is `b11ad6dbb08f4ad2d595e447c2a9f8cf032a0dac`.
The delivery branch adds only this record and the re-signed exception manifest.

Seven performance commits are merged in order:

1. `82fc40a88` perf(staging): key the pool by the staging thread
2. `4e06d6448` perf(batch): keep the split index scratch across ubatches
3. `16fa527fc` fix(batch): validate the per-token sequence count before the seq_id walk
4. `6aab27309` perf(tp5): drop redundant NextN drains in the MTP readback path
5. `77f02583b` fix(tp5): keep outputs_released in step after prepare
6. `2bfdd2dac` fix(tp5): port full-accept LateBind reuse with completion proof
7. `b11ad6dbb` fix(tp5): cover both positive LateBind binds with the completion proof

The four-slot relay shape-cache default was deliberately dropped: this base has
no shape cache compatible with `sched_active()`, and
`GGML_TP5_LLAMA_PLAN_CACHE=4` aborts at
`src/llama-context.cpp` with `GGML_ASSERT(backend_res != nullptr)` in
`ggml_backend_sched_get_tensor_backend(sched_active(), t_logits)`. The cache is
therefore unusable on this baseline and is not part of the candidate.

## Measurement setup

- Same binary, same machine, kernel `7.0.14-r28b6` (kernel #6), ROUNDS=2.
- Formal 512x2 contract: warmup round followed by measured rounds.
- Build options: `GGML_HIP_GRAPHS=OFF`, `GGML_CUDA_NO_PEER_COPY=ON`,
  `GGML_HIP_MMQ_FMA=OFF`.
- P5 OFF: the runtime default; P5 ON: 13 opt-in switches, without
  `GGML_TP5_LLAMA_PLAN_CACHE`.

## Results

| Arm | Measured steady tok/s (run 1 / run 2) | TTFT ms (run 1 / run 2) |
| --- | --- | --- |
| P5 OFF | 54.408 / 52.555 | 902.9 / 989.0 |
| P5 ON (13 switches) | 49.737 / 50.738 | 947.3 / 881.6 |

The P5 OFF arm clears the formal >=50 tok/s/request target on both measured
rounds. The same-day baseline before this candidate was 44.28 / 48.95 tok/s.

P5 ON remains 5-9% slower than OFF on this candidate. The per-worker staging
window and the batch/readback fixes are retained, but the remaining P5
switches do not pay for themselves at 512x2. The default stays OFF; P5 remains
opt-in for experiments only.

No GPU run, build, or runtime change was made while writing this record; the
numbers are the devops-light-runner's measured output for the commit above.
