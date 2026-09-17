# Changelog

One section per release, keyed by the exact tag. `verify-version` refuses to cut
a release whose tag has no section here, so this gets written *before* the tag is
pushed — the release notes are generated from it verbatim.

Write it for the person who downloads the build: what they get, what changed for
them, what to watch out for. Not a commit dump — the notes already carry the full
commit list underneath.

Releases before `b10269-1.5.0` predate this file; see the git history.

## Unreleased

Earlier TP5 measurements using `--no-mmap` ran with an empty lazy PLE table.
Those historical throughput and GPU-to-GPU equivalence results are not
full-model correctness acceptance; repaired-model measurements are required.

### Fixed

- **TP5 replay scheduling and shared transfer recordings.** Warm epoch chains
  reuse submission templates without skipping live binding, HC recipe, or full
  command-buffer sequence validation. Compatible reductions share owned P1
  recordings while retaining private P2 recipes and GPU retirement lifetimes.
  The measured cold collective recording allocation count falls from 1860 to
  930; paired no-MTP request medians are 39.00 versus 39.91 tok/s, not the 50
  tok/s target. Canonical 280-step logits remain bitwise unchanged. The server
  launcher adds `--baseline` and non-launching `--baseline --print-config` for
  fixed single-slot no-MTP comparisons with environment and binary provenance.

- **Whole-region Vulkan decode and collective boundaries.** GDN, attention,
  routed/shared MoE, and HC regions preserve their native arithmetic and cache
  ownership while eliminating intermediate dispatches. Terminal producers can
  emit the exact F16 wire representation alongside their F32 output; repeated
  reductions without a new producer still repack current data. Sparse mask
  compaction can share attention preparation, with independent-buffer fallback.
  The verified five-card checkpoint records 669 compute + 96 collective
  dispatches per token and bitwise-identical 280-step canonical logits. HC
  SUM/normalization fusion is numerically checked but is not an independently
  proven throughput improvement. The no-MTP 50 tok/s target remains unmet;
  further 8-to-6 dispatch lowering is paused at the user's request.

- **Bounded meta reconstruction and replay validation.** Reduction subgraphs
  allocate graph/hash capacity for their own spans rather than the entire model.
  Deferral analysis indexes nodes through the existing graph identity hash and
  stops ancestor traversal at the current reduction boundary. Replay validates
  current live bindings, shapes, operation parameters, offsets, and Vulkan
  buffers without dereferencing retired captured tensor addresses; splitting
  a formerly shared source still invalidates its recorded bindings. See `TP5.md`
  for measured cold/warm costs and the remaining collective-recording bottleneck.
  The later shared-P1 implementation and its measurements are listed above.

- **CPU attention honors explicit F32 accumulation.** F16 value caches no
  longer force F16 accumulators when `GGML_PREC_F32` is requested. Both direct
  decode and split-KV paths retain F32 value sums through online-softmax
  rescaling. A constant-value regression drops from roughly 3.88e-4 error to
  1.19e-7; the default-precision path and cache storage types are unchanged.

- **Vulkan replay preserves pending uploads.** Pinned host uploads and event
  waits on the ordinary compute context are flushed asynchronously before
  recording or submitting replay buffers, rather than discarded by a context
  reset. A minimal pinned-upload regression failed before the repair. The real
  five-GPU layer model now produces bitwise-identical replay/non-replay logits
  across 11 positions and the complete 248320-token vocabulary.

- **Lazy tensor payload and lifetime.** On-demand tables remain mapped after
  loader retirement, including when other weights use non-mmap loading.
  Preallocated fallback buffers are loaded instead of silently left empty.
  This fixes both a CPU PLE gather crash and zero PLE results under `--no-mmap`.
  Regression coverage includes mapped-only, eager-fallback, and mixed contexts.
  The real GPU PLE gather now matches all 2560 CPU elements bitwise; CPU mmap
  and non-mmap loading match over 11×248320 logits. Full CPU/GPU model parity
  remains a separate acceptance check.

- **Ordinary Qwen4EXP recurrence and Vulkan HC fold.** Indexed hybrid memory now
  forwards explicit RERoT capacities instead of enabling grouped recurrence when
  RERoT is off, and preserves the requested recurrent capacity. Four-stream HC
  folds now match the native 11-node multiply/reshape/reduce graph, with the
  original floating-point order. Graph optimization preserves the fold and its
  input lifetimes; Vulkan roots are at least 16-byte aligned. Unaligned views
  and externally observed intermediates retain the unfused path. Real-model
  profiling confirms 1057 fused calls per rank across the 11-position tape,
  with all 11×248320 logits bitwise unchanged. Complete 171-token counting
  reaches 5.69–5.72 tok/s after the PLE repair, not the 100 tok/s goal.

- **Qwen4EXP tensor-split GDN layout.** Native Q/K/V projections, convolution
  histories, and per-head decay parameters now use the same repeated-segment
  ownership. Column-parallel matmul and convolution preserve that layout instead
  of collapsing it into contiguous rank slices. A five-rank CPU regression
  matches 10,240 convolution outputs and 48 decay gates exactly; bounded
  five-GPU counting and arithmetic requests now produce correct text. This is
  not full-model numerical or 100 tok/s acceptance; see `TP5.md` for limits.

- **Vulkan indexed replay and GDN fusion.** Device-side gather/scatter operations
  may replay with changing indices and cache writes; the new six-round regression
  checks all cache rows exactly. Native-layout tensor inference again uses normal
  fused-GDN capability detection. Bounded, non-speculative five-GPU counting
  improved from 3.17 to 4.27 tok/s while preserving the tested answers; these are
  short samples, not sustained-throughput acceptance. Replay profile counters now
  distinguish queue API calls from submission batches correctly.

- **Vulkan TP5 R=2 correctness and fused submission.** Both reduction phases now
  use the same aligned two-bank mailbox mapping. Replay plans retain their
  buffers, validate source bindings and scratch generations, and register
  externally submitted work for synchronization. Adjacent SUM/compute/PUSH
  stages are fused without dropping required memory dependencies. Five-RX-6800
  F16/F32 numerical and dependent-replay checks pass; warm 96-stage paired
  median speedups are 1.660×/1.462× (microbenchmark, not end-to-end or tail-latency
  guarantees). See `TP5.md` for protocol, complete timing samples, and limits.

### Added

- **Owned asynchronous input snapshots; backend plugin ABI 3.** Supporting
  backends capture host bytes before returning and order destination writes
  after prior device work, so callers may immediately reuse the host storage.
  Composite backends check all destinations before capture. Backends without
  this capability retain the existing synchronized path. Out-of-tree backend
  plugins must be rebuilt for the new interface version.

- **Opt-in replicated TP5 attention/GDN.** `GGML_TP5_REPLICATE_ATTN=1`
  mirrors native attention weights and caches while retaining expert channel
  partitioning. The real five-GPU model executes 48 rather than 96 reductions;
  complete 171-token counting requests measure 5.35–5.40 tok/s, not 100 tok/s.
  Peak observed VRAM is about 13.25 GiB per card. CPU/GPU full-model numerical
  acceptance remains open. The header-only inspector supports the matching
  `--replicate-attention` mode and fixes double-counted KV layer instances.

- **Single-submission decode chain includes the model tail.** The final
  non-reducing subgraph now travels inside each rank's one batched submission
  instead of a separate async submit per rank, so the completion signal and the
  tail it must order are published together. The hot path records five queue
  API calls and 485 submission batches per decode graph; these are not driver
  submit counts. Mesh tests verify a dependent GPU tail against a CPU oracle
  and reject malformed tails before submission. Short model requests are
  correct, but no isolated throughput improvement from tail batching is proven.
  After correcting an earlier pre-change artifact mix-up, a fresh capture
  hitting the five-call path matches non-replay over 11×248320 logits bitwise.

- **TP5 replay-chain GPU timing.** `GGML_TP5_GPU_TIMING=N` captures one
  timeline chain with per-rank SUM/compute/PUSH intervals and inter-batch gaps.
  The same capture reports host queue-call durations and compute, collective,
  and marker command-buffer counts after every rank has been submitted.
  Queries are availability-only and never reused in flight. Extra command
  buffers and barriers perturb timing; use separate runs for throughput.


- **FlashPrefill V2 sparse prefill (experimental, opt-in, default off;
  implementation and compile delivered, runtime acceptance NOT RUN).**
  New `--flashprefill off|auto|required` policy plus tuning flags
  (`--flashprefill-alpha`, `--flashprefill-block-q`, `--flashprefill-block-k`,
  `--flashprefill-sink-blocks`, `--flashprefill-window-blocks`,
  `--flashprefill-dense-tail-tiles`, `--flashprefill-tail-scope`,
  `--flashprefill-min-kv`, `--flashprefill-full-attn-layers`,
  `--flashprefill-mean-correction`, `--flashprefill-exact-all`), backed by new
  GGML ops (`GGML_OP_FLASH_PREFILL_POOL/SELECT/ATTN`), CPU reference kernels,
  Vulkan native kernels, always-emitted `/metrics` series
  (`llamacpp:flashprefill_*`, zeros while OFF; GPU totals merged only after
  successful graph slices; GPU dispatch times 0 until the timestamp hook
  reports; required-mode enforcement is graph-side, metrics only counts),
  and `scripts/flashprefill-matrix.py` /
  `scripts/flashprefill-quality.py` harnesses. Only ordinary prefill and
  bounded RERoT teacher-forced injections are sparse-eligible (ordinary
  dispatch shared over supported KV shapes; raw-Q RERoT hooks Qwen3.5
  dense/MoE only, including current Ornith weights); decode, MTP
  draft/verify,
  embeddings, rerank, out-of-family RERoT, recurrent/SWA layers,
  transposed-V or multi-stream layouts, special KQ bias, and unsupported
  backends keep the existing dense path. TriAttention (`3/32`) semantics and the RERoT
  active-lane MTP pause are unchanged. See `PREFILL.md`.

### Notes

- **Acceptance not run.** The server, 4 new tests and 7 existing test
  targets all compile/link clean (`build-prefill/bin/llama-server`;
  baseline frozen separately), but no tests, benchmarks, quality gates, or
  deployment have been executed. No performance or quality claims are made;
  the Definition-of-Done checklist in `PREFILL.md` §21.3 remains fully
  unchecked. Hashes: see `build-prefill-evidence/manifest.json` and
  `binary-libraries.sha256`.
- State restores across contexts/processes/restarts are deliberately rejected
  (enabled-only 64-byte policy envelope; re-prefill on mismatch; no sidecar
  file). Qwen3.5 dense/MoE family only; other architectures get no automatic
  sparse routing.

## b10269-1.6.0

### Added

- **Windows AMD ROCm archive** `llama-turboquant-windows-x64-rocm.zip`. HIP
  backend for RDNA2 through RDNA4 and Ryzen AI 300 / Ryzen AI Max (gfx1030,
  gfx1100/1101/1102/1103, gfx1150/1151, gfx1200/1201), self-contained: the HIP
  runtime and the BLAS DLLs it links are bundled (about 100 MB), only a current
  Adrenalin driver is needed, no ROCm SDK install. Until now Windows + Radeon
  meant the Vulkan build; HIP is the faster prompt-processing path on these
  GPUs. Everything in the archive is Authenticode-signed, the AMD DLLs with
  AMD's signature where they ship one.
- **Linux ROCm archive** now also targets gfx950 (Instinct MI350/MI355X),
  gfx1150 (Ryzen AI 300) and gfx1103 (Radeon 780M/760M).
- `docs/amd/`: what ships for AMD, the ROCm vs Vulkan benchmark plan for the
  TurboQuant KV cache, and the open items. `scripts/bench-amd.sh` runs that
  matrix per backend build, `scripts/bench-amd-report.py` renders one table.

### Changed

- ROCm builds no longer pass `GGML_HIP_ROCWMMA_FATTN`; upstream removed the
  rocWMMA flash-attention path and the flag was a no-op.

### Notes

- **The Windows ROCm archive is beta.** It is built, signed and checked for
  completeness in CI, but has not been run on a Radeon yet. `llama-server
  --list-devices` must show a HIP device; if it does not, or if loading fails,
  report the GPU, driver version and the error, and use the Vulkan archive in
  the meantime.
- Atomic Chat does not select the Windows ROCm backend yet; the client change
  follows separately. Until then it is a manual download.

## b10269-1.5.1

### Fixed

- **Ling-3.0-flash (BailingMoeV3) no longer emits garbage token bursts.** The
  model is trained with clamped SwiGLU activations in its late layers, and the
  per-layer limits live in `config.json` under `expert_swiglu_limit_list` and
  `share_expert_swiglu_limit_list`. The public HF modeling code ignores those
  keys and so did this port, which caused deterministic transient logit
  collapse - output like `count += 1eville` dropped into otherwise fine
  generations. Measured at roughly -20 pass@1 on HumanEval (72.6% -> 93%+ with
  the fix); the garbage-token repro is eliminated.

### Notes

- **Re-convert your Ling-3.0-flash GGUF to get the fix.** The clamp limits are
  written by the converter into two new KVs (`{arch}.swiglu_clamp_exp` and
  `{arch}.swiglu_clamp_shexp`); a GGUF produced before this release does not
  carry them, and the runtime then defaults to no clamping. Re-download the
  quant or re-run `conversion/bailingmoe.py`.
- Both KVs are optional and default to zero, so existing GGUFs and every other
  architecture are unaffected. The graph needed no change - the SwiGLU clamp
  branches in `build_ffn` / `build_moe_ffn` already trigger on a nonzero
  per-layer limit, matching the vLLM `SwigluStepAndMul` semantics.

## b10269-1.5.0

### Added

- **NVIDIA DGX Spark (GB10) support.** New archive
  `llama-turboquant-linux-arm64-cuda-13.3`, built natively for aarch64 with
  CUDA 13.3 and sm_121 SASS. Other arm64 NVIDIA machines (GH200, GB200, Jetson
  Thor) run it too, JITing the kernels from PTX on first launch. This is the
  first Linux arm64 build the fork ships — until now arm64 meant macOS only.
- **BailingMoeV3 (Ling 3.0) architecture support**, including the KDA gate
  handling.

### Changed

- **Linux CUDA archives are roughly half the size** — 1657 → 956 MB (12.4) and
  1879 → 1028 MB (13.3) measured across both the `.zip` and `.tar.gz`. The zips
  were storing `libcublas.so` → `.so.13` → `.so.13.5.1.27` as three full copies
  because `zip` followed the symlinks.
- **CUDA 13.3 builds ship Ampere PTX (`80-virtual`).** A100/H100/B200 were
  falling back to the Turing PTX floor, which silently disabled `cp.async` and
  the Ampere MMA path — both gated on `__CUDA_ARCH__ >= 800`. Those cards get
  Ampere-class kernels now. No architecture lost support in this release.
- Windows CUDA builds got their architecture lists pinned, all runner cores, a
  ccache that can actually hold a CUDA build, and 7-Zip instead of
  `Compress-Archive`. Release turnaround drops accordingly.

### Notes

- The DGX Spark archive has **not yet been validated on real GB10 hardware** —
  it is built and arch-checked in CI (`cuobjdump` asserts sm_121 SASS is
  present), but nobody has run it on a Spark yet. Treat this one as beta and
  report back.
- The CUDA 13.3 archives now use `-compress-mode=size`. Kernel SASS is
  unchanged and inference speed is unaffected; the fatbin is decompressed once
  at module load. It needs a driver from the CUDA 12.4 era or newer.
