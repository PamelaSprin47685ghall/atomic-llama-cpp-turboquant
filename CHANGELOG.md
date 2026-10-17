# Changelog

One section per release, keyed by the exact tag. `verify-version` refuses to cut
a release whose tag has no section here, so this gets written *before* the tag is
pushed — the release notes are generated from it verbatim.

Write it for the person who downloads the build: what they get, what changed for
them, what to watch out for. Not a commit dump — the notes already carry the full
commit list underneath.

Releases before `b10269-1.5.0` predate this file; see the git history.

## Unreleased

**Necessity certificate v4: soundness fixes and a split bound**
  `scripts/tp5-necessity-certificate.py` v4 supersedes v3 after audit
  found three defects: dense bytes per layer were 90.8 MB in v3 but the
  audited 1222-tensor inventory gives 63.82 MB; the L2 retention credit
  divided by the full 11.9 GB weight pool instead of the per-token active
  set (under-crediting caching, invalid as a lower bound); and v3's
  single 476 tok/s number claimed per-row validity under the
  perfect-acceptance MTP relaxation, but weight DRAM traffic is read once
  per verify CYCLE and amortizes across committed rows, so R3 cannot
  bound the full allowed set. v4 reports a split certificate:
  L_global = 5667 tok/s (FP32 ALU bound, valid for the full allowed set)
  and L_nomtp = 730 tok/s (active-weight DRAM bound, valid only for the
  no-speculative-batching class). A serial-latency term was investigated
  and rejected: every candidate coupling has a legal implementation that
  hides it under the relaxed allowed set. 14 tests in
  `tests/test-tp5-necessity-certificate.py` pin the defect fixes.

- **First nontrivial global necessity certificate for the TP5 task.**
  `scripts/tp5-necessity-certificate.py` derives a per-token time lower
  bound valid across the full allowed strategy set (lossless
  representation, caching, partitioning, prefetch, fusion, perfect-acceptance
  MTP) from four unavoidable resources: output emission, causal cut-set,
  active-weight DRAM traffic, and FP32 ALU work. v3 counts both the
  10/512 active experts AND every dense weight (attention, PLE, HC,
  router, shared experts) that must be read each token: 1076.3
  MB/GPU/token of necessary DRAM at the 512 GB/s spec ceiling, plus
  12.1 GFLOP/token of FP32 work. The binding resource certifies
  S_hi = 476 tok/s; demonstrated S_lo is 66.34 tok/s (gap 7.2x, up from
  null -> 2395 -> 476 through honest tightening).

- **Native AllGather executes dynamic active rows on real five-rank hardware.**
  A scalar ragged-row ADD producer guards every element against the frame's
  live token count, the submitted frame is the authoritative extent source for
  indirect-dispatch patching, and RELAY chain-cache reuse is restricted to the
  chain whose compiled commands are actually resident in the per-rank
  scratch. A five-rank fixture verifies 4→1→3→2→1 active rows across two
  mailbox banks: exact scores, untouched inactive source tails, published
  status metadata, and downstream softmax, with asymmetric frames rejected
  before submission.

- **Lossless BF16 router weights run through the native F32 GEMV pipeline.**
  The converted packed-BF16 buffer dispatches with the generic one-row F32
  workgroup geometry, so accumulation and reduction order match the F32 path
  bit-for-bit (verified on synthetic full-vocabulary scores). The dispatch
  reopens the compute submission after the one-time conversion synchronize.

- **One-time router weight conversions no longer corrupt TP5 replay definitions.**
  The conversion's temporary transfer/compute command buffers are suspended
  from definition-tape capture, kept out of the replay pending-buffer list,
  and excluded from the definition dispatch count; the first-conversion path
  no longer synchronizes mid-graph, so the five ranks' shared submission is
  never split during recording. Verified by real-model fixed-token parity:
  3607 files (logits, state, all 3600 router trace tensors including per-rank
  inputs) byte-identical between F32 and lossless-BF16 router weights.

- **Vulkan converted router weights refresh after strided uploads.**
  Synchronous and asynchronous 2D writes now invalidate the full checked byte
  span, including updates to nonzero-offset weight views. A native regression
  verifies every score after both write paths and warm replay; previously the
  converted cache could retain the old weight despite a successful upload.

- **Lossless BF16 router weights qualified with a measured throughput win.**
  Preregistered eight paired blocks (ABBA, 16 independent processes, frozen
  binary, single env-var difference) show +1.14% pooled tok/s, 95% CI
  [0.79%, 1.48%], with byte-identical outputs in every block. The mode stays
  opt-in (`GGML_VK_ROUTER_WEIGHTS=bf16`); production default changes need
  separate qualification.

- **Meta graph views no longer reuse a different slice with the same parent UID.**
  Cached child graphs and prepared native definitions now include the node-array
  base alongside UID and node count. Evaluation callbacks can visit equal-sized
  slices without silently replaying the preceding operation. A two-backend CPU
  reproducer now executes ADD after SQR correctly; regression coverage also
  preserves same-slice reuse with updated runtime inputs. This fixes observer
  correctness, not a claimed GPU speedup or router qualification.

- **LateBind validates current-epoch Q payloads and reduces all live HC rows.**
  Exact F32 publication and every Q8DOT variant now publish payload checksums,
  epochs and active extents; normal relay and L2 validate these before retiring
  ready or reducing Q. Real five-GPU production observations caught bytes still
  changing after ready. The extent gate also exposed legacy two-row chains
  reducing only one row; row counts now follow each stage's actual HC payload,
  including the next mailbox bank. Seven modes each completed eighty real
  fixed-prefix requests afterward with no observed post-validation payload
  changes or abnormal exits. Existing timeout/numerical gates remain; these
  diagnostics do not establish universal generation quality or new throughput.
  All eight LateBind modes across disabled/auto/forced MMVQ subsequently passed
  independent semantic review of 144 real consecutive MTP1 requests, with normal
  exits. Deeper interactions and paired production performance remain separate
  qualification requirements; occasional auto-policy budget-tail rerecording
  is retained as an optimization target, not grounds to remove configurations.

- **Failed TP5 cleanup releases pending P1 owners before their Vulkan pools.**
  A real sidecar-rejected model load crashed while freeing a retained P1 command
  buffer after its command pool was destroyed. Teardown now releases drained
  in-flight owners while pools are live and unlocks the communicator mutex
  before deleting it. Real five-rank and partial-four-rank GPU submissions at
  logical epoch zero complete their fences and payload writes before owners
  expire; an unsubmitted rank remains unsignaled and untouched.

- **POSIX server shutdown no longer reenters the queue mutex from a signal handler.**
  SIGINT/SIGTERM now notify a normal shutdown thread through a pipe; the
  second-signal immediate-exit escape remains. An early termination cannot be
  overwritten when the queue loop starts. Production queue regressions cover
  lock interruption, early stop, second-signal escape and handler restoration;
  four real five-GPU model exits completed within the unchanged ten-second
  limit, including the previously force-killed output-rejected configuration.
  Its degenerate output remains rejected, not promoted as a speed candidate.

- **New zero-prefix requests discard the preceding MTP hidden-state carry.**
  After cache/checkpoint adjustments, a request with no retained prompt prefix
  resets only its sequence's speculative state before prefill. Cached prefixes
  and other slots are unchanged. A passive observer of six real five-GPU
  requests found nonzero carry on all five reused requests before the fix and
  zero carry on all six afterward; reference outputs and draft counts remain
  unchanged. This fixes request-state isolation, not a claimed speedup or a
  numerical explanation for every LateBind output difference.

- **TP5 plan identity includes the actual Vulkan buffer, not just its extent.**
  Same-offset/size views into distinct live buffers no longer qualify the old
  native definition or reuse its captured descriptors. Real five-device
  admission rejects the rebound target, with before/after evidence; pending
  two-bank and partial-rank native owner draining also passes at logical epoch
  zero. Single-prompt exact/aggressive completions recover natural EOS, but
  repeated multiprompt quality is not established by that smoke. The opt-in
  relay probe checks real rebound bindings and shares consumer progress across
  mailbox banks, avoiding a false previous-bank completion on even chains.

- **TP5 prepared decode uses a qualified two-bank definition, not a bank-only topology lock.**
  Linear lowering checks compute command identity, plan keys/indices, capacity,
  capture/isolation and resource generations against one definition recorded for
  both mailbox polarities. Replaced definitions retain both command-pool owners
  until actual native submissions complete. Graph rebuilds invalidate prepared
  handles; a failed native submission cannot fall through to fallback execution.
  TP5 warmup covers decode and configured MTP verification prefixes, restores
  memory/sampler/performance state and aborts model loading on failure;
  `--no-warmup` remains an explicit opt-out. CPU control-plane and single-device
  replay/MMVQ smoke checks passed. Five-GPU binding/drain, repeated-request
  prewarm equivalence, L2 and deep/single-row transition checks now pass.
  The output-qualified 131-configuration search retained off/disabled/MTP1;
  four independent current-server blocks did not establish a significant
  speed advantage over the best qualified no-MTP reference. No new speedup
  or mathematical global optimum is claimed.

Earlier TP5 measurements using `--no-mmap` ran with an empty lazy PLE table.
Those historical throughput and GPU-to-GPU equivalence results are not
full-model correctness acceptance; repaired-model measurements are required.


- **Vulkan matvec row tile spills at 13-18 columns; prefill gained 28%.**
  The multi-column matvec pipelines kept the 4x row tile from the wide-
  dispatch widening at every column count, so at NUM_COLS 13..18 the
  per-thread accumulator array (13-18 x 8 floats) spilled past the register
  budget. Measured on Bonsai PQ2_0 / RX 6800, prefill throughput fell off a
  4x cliff exactly at 13 columns (112.8 t/s at npp=12 versus 28.2 at npp=13)
  and the n=16 prefill shape ran at 35 GB/s effective weight bandwidth
  against 129-141 GB/s for the n=6 decode shapes. Halving the row tile for
  those widths (rm_wide 2) keeps the accumulator in registers: npp=13..18
  now all measure 111-116 t/s, the single-stream n=16 prefill gains 28%
  (86.6 -> 110.9 t/s at npl=1). Decode is untouched (1-8 column pipelines
  unchanged); repeated runs confirm 1/4/6/8-lane decode at
  32.6/81.6/97.6-98/102.5 t/s, matching the pre-change baseline within
  noise. Greedy 6-lane decode remains byte-identical to the single-lane
  path; test-backend-ops passes (368 s).

  Routing small-N decode through the scalar mul_mm GEMM instead was
  measured with a temporary env hook and rejected: 3-7x slower than the
  matvec at these shapes on this no-coopmat RADV device (TG 33.4 -> 4.4
  t/s at one lane). The matvec stays the decode path; the GEMM threshold is
  not widened.

- **RERoT grouped-head attention is now the DEFAULT.** The
  `GGML_VK_REROT_GROUPED_HEADS path (one subgroup per Q head, KV tile
  staged once per GQA group) was qualified on a full episode for the first
  time. Attention cost drops from 18.7 s to 4.2 s per episode (per-call
  460 us -> ~150-200 us at n_kv 4-5k) and the indexed shader's efficiency
  gap versus ordinary FA narrows from 2.7x (273 -> 478 GFLOPS/s at the
  production shape). Precision gates pass on both paths (indexed error
  < 1.2e-7 vs F64; test-rerot-attn 0 failures).

  The earlier qualification run FAILED, and the two defects responsible
  were both found, fixed, and re-verified -- neither was in the grouped
  shader itself:
  (1) The synthesis lane leaked a `<|fim_prefix|>become` tool call as
  the public answer, spelling the reserved marker character-by-character
  (the token-level sampling ban only covers the single-token spelling).
  Fixed by KV-level occlusion of the advertised tools block from the
  planner and synthesis readers (workers keep it; see the occlusion
  commit). The deterministic leak seed is now clean 3/3.
  (2) Every third back-to-back episode died 42 ms after start with a bare
  rerot_resource_exhausted. Pinned by backtrace + slot-state dump to a
  force-sealed worker whose host slot was never released (an inverted
  branch in the lane hard stop), leaving a GENERATING slot pointing at
  a deleted episode; the bare propagation then killed the next innocent
  episode. Fixed by releasing the slot on seal success; the failing
  three-episode sequence now completes on scalar and grouped alike.

  With both fixed, the complete paired run (seeds 777/12345/20260923
  back-to-back, thinking wire, Bonsai PQ2_0 / RX 6800) passes 3/3 on the
  grouped path, so grouped is now the default. Devices or shapes that
  cannot stage the group keep the single-head scalar path;
  GGML_VK_REROT_GROUPED_HEADS=0 restores Br=1 for A/B.

- **Multi-lane decode is now worth its lanes, and RERoT finally beats serial.**
  Two Vulkan matvec fixes and two scheduling bounds. The shader dequantized each
  weight block once per (column,row) pair, because the generic loop nests columns
  outside rows and the existing shared-dequant fast path only covered
  `num_rows == 1`, which quantized pipelines never reach; and every workgroup
  re-reads the whole activation vector, which at six columns is 1.07 GB against
  23.7 MB of weights. The loop is inverted for multi-column dispatches and PQ2_0
  matvec pipelines widen their row tile when the dispatch is wide.
  `llama-batched-bench` decode t/s: 33.10 -> 33.47 at one lane (no regression),
  51.09 -> 82.54 at four, 68.99 -> 98.90 at six, 63.03 -> 103.40 at eight;
  scaling against one lane goes 2.08x -> 2.95x at six and eight lanes stops
  regressing. Output is bitwise identical (greedy decode, six concurrent requests
  versus one).

  Parallel thinking then has to fit a token budget to pay off, so
  `--rerot-lane-wrap-tokens` (default 256) gives a long worker lane, and the
  synthesis lane at three times that allowance, escalating in-voice reminders to
  close. Without them lanes ran past 1240 tokens each and the request never
  finished; with a worker-only bound the synthesis took over as the unbounded
  phase. On the geography prompt against a fixed eight-item checklist:
  serial 134.37 s / 3085 tokens / full coverage; RERoT at 256 **101.51 s / 2959
  tokens / full coverage** (1.32x faster on fewer total tokens, terser answer);
  at 512 it regresses to 251.65 s / 5575 tokens; at 128 it finishes in 53.37 s but
  coverage collapses to 1/8, which is what the checklist is there to catch.
  Earlier drafts of this entry quoted an "end-to-end tok/s" built from
  `timings.predicted_n`, which counts lane tokens and therefore flatters RERoT;
  the numbers above are wall clock at equal coverage. Details in `RERoT.md` §22.6.

- **RERoT planning wire: parallel thinking aspects instead of Mermaid indentation.**
  New `--rerot-plan-wire thinking` injects one instruction sentence plus
  `<ul>Let me see what can be thought through on its own here...`, then constrains
  generation with a six-rule GBNF and stops at the first `</ul>`. Items are bare
  `<li>` sub-titles in words with no punctuation at all (path characters `/ _ -`
  stay legal), so the wire cannot drift into indentation, outline numbering or
  `&lt;` escaping. Execution is the flat shape the pen cohort wants: synthetic
  root plus one peer worker per aspect. `json` / `mindmap` / `todo` remain
  selectable controls.

  Two structural facts drove the design and are worth knowing before tuning it.
  First, the unit is an *aspect* because that is what the model's own
  unconstrained thinking is made of: an RERoT-off capture on a real
  agent-coding task spends about 25 of 41 paragraphs circling one unknown about
  the codebase and the rest covering the requested deliverables. Second, the
  escape token that ends a free-form region is not merely unlikely under the
  model card's thinking-mode sampling, it is masked: measured on the real probe
  context, `<` sits at rank 2 with p=0.086 while the head is empty and falls
  below 1.2e-3 (2.7e-3 of the top token) after one sentence, i.e. under
  `min_p=0.05` it leaves the pool. An unbounded region therefore never
  terminates — one head ran 1749 tokens with no list, one item ran 1670 tokens
  with no close. The server answers this in two ways: when the lane ends a head
  line the newline is replaced by a reminder plus that same newline (capped), and
  item text carries no punctuation, so a label cannot hold prose.

  The `become` frame was rewritten for all tree wires: the tool carries a
  one-clause description, the tool reply is a data echo, and the contract is the
  lane's own first-person sentence at the start of its thinking, ending on a colon
  so the thought continues. Second person in the reply does not survive a shared
  KV (6/6 mindmap workers dismissed the handoff and re-answered the whole
  request; 3/3 coding workers claimed the same node; a numeric tag only made the
  tag the anchor), and a seed ending in a full stop made every worker sample
  `</think>` immediately while the synthesis lane emitted a `become` tool call as
  the public answer. On a single RX 6800 with Bonsai-2 27B PQ2_0 and six pens, the
  wire planned a `/healthz` endpoint task in 286 probe tokens (8 aspects, 7
  workers each on their own) and a geography prompt in 148 (3 aspects, 3
  workers). Every run hit its 60 s client deadline: there is no completed
  episode, no public answer and no throughput comparison. Evidence, including the
  distribution measurement and the natural-thinking baseline:
  `artifacts/rerot-thinking-wire/`; scope and limits in `RERoT.md` §22.5.

- **Meta rank-local asynchronous readback.** Eligible contiguous F32 vocabulary
  shards now copy into persistent per-device pinned host slots, submit every
  rank before waiting, and assemble at the existing consumption boundary.
  Pending reads retire before a subsequent read, slot growth, fallback, or
  backend destruction; unsupported layouts retain the original getter.
  Five-device byte/lifetime regressions and MTP acceptance prefixes 0–6 pass.
  `GGML_META_ASYNC_READBACK=0` provides a same-binary control; timing evidence
  and the unchanged model/sampler contract are recorded in `TP5.md`. Five paired
  ABBA blocks measured 72.25→74.27 decode tok/s (+2.80%); all 60 retained cold
  and steady requests were exact 171-token natural stops. This is not 100 tok/s.

- **Vulkan skinny MMVQ projections keep distinct input identities.** The converted
  activation cache now keys on the graph tensor, not a temporary stack view
  reused by successive projections. Forced MMVQ could otherwise give the second
  projection the first input's values. A sparse-weight numerical regression
  checks distinct inputs and changed inputs across cold execution and replay;
  it fails with the old backend and passes with the repair.

- **TP5 counting acceptance preserves exact request evidence.** The cross-matrix
  runner now records the submitted request alongside the full response, including
  responses missing token usage. Counting gates no longer trim whitespace before
  claiming a byte-for-byte match. The 171-token workload explicitly disables
  thinking and prompt reuse; a reasoning-enabled request is a different workload
  and needs its own context/output budget. A 40-character response preview is
  not evidence of a miscount. See the September 24 correctness investigation in
  `TP5.md` for the restored non-MTP binary and full-response evidence.

- **MM-R1 logical leaves no longer inherit a physical pen budget.** Removed
  the incremental mindmap probe guard and parser node/leaf/wire-byte/depth/
  label budgets; the JSON probe retains its separate 512-token policy. After one English
  instruction, the mindmap probe injects the Mermaid `mindmap` header, the
  literal `root` line, and four child-indentation spaces; grammar generates
  the subtree at any even indentation depth; the parser runs only once the
  closing fence arrives, rather than rescanning every generated token.
  `root` is a synthetic container, excluded from logical node and worker
  counts. Persisted RERoT state advances to v8 rather than misreading older
  trees. Bonsai-2 on Vulkan completed a 21-leaf plan on six pens with the
  model's 262144-token context; the full 51-leaf Raft stress reached synthesis
  but its HTTP client deadline expired before a public answer. See `RERoT.md`
  §22.3 for the exact scope and artifacts.

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

- **Vulkan async output copies retain their buffers.** A queued readback of
  four logits rows could outlive the graph allocator's source VkBuffer as
  RERoT switched between six active pens and a seventh parked worker. Pinned
  host transfers now retain both VkBuffer owners until the queue fence retires
  the recording context. The Bonsai-2 seven-leaf GPU request that previously
  triggered a CPG use-after-free now completes with the correct answer and no
  new GPUVM fault. Mindmap remains opt-in; its paired wall time on that prompt
  is slower than the JSON baseline, which chose a simple single-answer route.

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



