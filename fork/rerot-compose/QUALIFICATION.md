# Qualifying an installed TQRerotCompose

This document defines reproducible checks and the evidence each check produces.
It does not record a successful run. Executed results, source identities, and
historical failures remain in `fork/VALIDATION.md` and its linked receipts.

## Installed package boundary

C builds from this directory with installed public `llama`, `TQRerot`, `TQRelay`,
`TQText`, and `TQSampling` packages. The sampling shortlist is an independent B
result; C retains A's native sampler chain and random state. C does not link the
other `TQCompose` program to obtain its different sampling policy.

C requests `llama_context_params.batch_invariant_matmul = true` before native
context creation, including the first shared-prefix prefill. The request
reaches applicable quantized dense `MUL_MAT` and expert-indexed `MUL_MAT_ID`
projection nodes as `GGML_PREC_BATCH_INVARIANT` (mode 20). It takes precedence
over a projection builder's accumulator-precision request. A's default
parameter remains `false`, preserving ordinary optimized quantized matrix
selection for other consumers.

The tested mode preserves a backend's native single-query quantized arithmetic
across query batches. The native Vulkan device/type/K policy can still select
Q8 activation conversion and MMVQ. This is separate from C's resident F32 K/V
and disabled FA reader policy. CPU and Vulkan results have separate numerical
scope; the request does not establish general cross-backend bitwise equality.

Use the complete source-isolation command to verify the pinned A archive and
its exact reviewed overlay bytes, independent B/C builds, and relocated public
consumers:

```sh
python3 fork/tests/isolation.py \
  --output build-rerot-isolation-candidate-01 --jobs 4 \
  --rerot-model /models/instruct.gguf \
  --rerot-ubatch-sizes 128,1 --rerot-phase-modes release,retain
```

Without `--a-prefix`, this command rebuilds a CPU A from the pin and verified
overlay snapshot. It removes its source/build copies and original installation
prefixes before testing the relocated `delivered` tree. Inspect `commands.json`,
`verified-boundary.json`, `result.json`, and the nested
`relocated-real-rerot/receipt.json`. A failed command retains its log and command
record even if the final `result.json` was not written. The run never deletes
the user's repository or an input installation.

`--a-prefix /installed/A` can test installed compatibility and relocation with
an existing backend build. It reports `a_source_provenance: reused_unverified`:
the copied installation's source pin, overlays, and build provenance have not
been proved by that run. Neither form creates an operating-system sandbox.

Omitting `--rerot-model` leaves real generation explicitly skipped. Supplying
it currently runs four relocated cases: two microbatch sizes and two phase
modes, with one repetition. Isolation does not pass the strict cross-shape flag
or register the optional real cancellation test. Run the following matrix on
the relocated executable when those additional conclusions are needed.

## Quantized projection and reader gates

Build the public A seam consumers against the same installed A candidate that
C will load. The dense fixture retains its historical `test-vulkan-...` target
name but also builds against a CPU-only A installation. The ID fixture uses
only public GGML APIs. Neither fixture silently chooses another device when
the requested device is unavailable.

CPU:

```sh
cmake -S fork/tests/a-seams -B build-rerot-a-seams-cpu-candidate-01 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/installed/cpu
cmake --build build-rerot-a-seams-cpu-candidate-01 --parallel 4 \
  --target test-vulkan-quantized-shape-invariance test-mul-mat-id-shape-invariance \
           test-rerot-reader-views test-rerot-attention-oracle
ctest --test-dir build-rerot-a-seams-cpu-candidate-01 \
  --output-on-failure --no-tests=error \
  -R '^(cpu-quantized-shape-invariance|mul-mat-id-shape-invariance-cpu)$'

build-rerot-a-seams-cpu-candidate-01/test-rerot-reader-views \
  --model /models/instruct.gguf --gpu-layers 0 --threads 4 --kv-type f32 \
  --require-cross-shape-equivalence
build-rerot-a-seams-cpu-candidate-01/test-rerot-attention-oracle --gpu-layers 0
```

Vulkan:

```sh
cmake -S fork/tests/a-seams -B build-rerot-a-seams-vulkan-candidate-01 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/installed/vulkan
cmake --build build-rerot-a-seams-vulkan-candidate-01 --parallel 4 \
  --target test-vulkan-quantized-shape-invariance test-mul-mat-id-shape-invariance \
           test-rerot-reader-views test-rerot-attention-oracle
ctest --test-dir build-rerot-a-seams-vulkan-candidate-01 \
  --output-on-failure --no-tests=error \
  -R '^(vulkan-quantized-shape-invariance|mul-mat-id-shape-invariance-vulkan)$'

build-rerot-a-seams-vulkan-candidate-01/test-rerot-reader-views \
  --model /models/instruct.gguf --gpu-layers 99 --threads 4 --kv-type f32 \
  --require-cross-shape-equivalence
build-rerot-a-seams-vulkan-candidate-01/test-rerot-attention-oracle --gpu-layers 99
```

The registered Vulkan operator tests request `Vulkan0`. To select another
reported native device, run each fixture binary with that device name as its
positional argument. The model-based reader commands use A's model placement;
retain actual model/device and positive-offload evidence for them. A positive
`--gpu-layers` request alone does not identify which device executed the work.

| Gate | Reference and required comparison |
| --- | --- |
| Dense quantized projections | Native `GGML_PREC_DEFAULT` N=1 computations supply the reference; candidate nodes request mode 20. Compare full output bits across the fixture's query counts, head layouts, order, and one/two bias additions. |
| Expert-indexed quantized projections | Native DEFAULT N=1 computations omit the precision setter entirely. Compare full output bits for IQ3_S, IQ4_NL, Q8_0, and Q4_K, shared/per-slot activations, padded I32 ID rows, repeated and reordered expert IDs, `ADD_ID`, shared per-slot scale, and the 65,540-slot periodic boundary. |
| Strict full-model reader | Pass the existing same-shape, causal-effect, state, and explicitly required cross-shape gates; retain full-vocabulary differences and the actual model/backend identity. |
| Independent attention oracle | Compare the real tiny-GGUF ordinary/reader results with independent F64 calculations under the original bounds. This oracle does not represent every operation of the full production model. |

Both projection fixtures accept `--native-matmul` as a separate control that
uses the ordinary default path for candidate batches. Preserve its real result,
including a numerical failure. The ID fixture omits every DEFAULT setter call
so this control can run against the retained pre-ID-extension library; that
library's old setter assertion did not accept `MUL_MAT_ID`. A successful
mode-20 run is its own candidate result and does not rewrite the old control.

Record the full footer and exit status from each gate. The strict reader flag
is required here even when the executable is also registered as a CTest;
running its default CTest command alone does not require cross-shape equality.
Operator passes qualify the tested shapes and native reference profile. They
do not replace the complete generation matrix below.

## Real generation matrix

Build and install each candidate with `fork/build.py --backend cpu` or
`--backend vulkan`. Complete the separate A reader numerical gates before
promoting a full-generation result. Use a fresh output directory for every
candidate; both real harnesses reject an existing output directory.

The compact generation matrix is `ubatch=128,1 × release,retain × repeat=2`:
eight complete episodes per backend. Keep `--parallel 2` for the default
coverage fixture. Two independent writers occupy the pens. A third independent
observer must wait until one writer seals, so it necessarily sees a completed
non-ancestor. A fourth worker depends on all three and checks dependency
admission. Synthesis then waits for every worker. No relative response length
or chosen order of writer completion is required.

CPU:

```sh
python3 fork/rerot-compose/tests/real-whiteboard.py \
  --binary build-rerot-cpu/install/bin/tq-rerot-compose \
  --model /models/instruct.gguf \
  --gpu-layers 0 --split-mode none --parallel 2 \
  --ctx-size 4096 --total-kv 4096 --batch-size 512 \
  --ubatch-sizes 128,1 --phase-modes release,retain --repeat 2 \
  --require-cross-shape-equivalence \
  --output build-rerot-matrix-cpu-candidate-01
```

Vulkan, with the actual device name from the installed backend:

```sh
python3 fork/rerot-compose/tests/real-whiteboard.py \
  --binary build-rerot-vulkan/install/bin/tq-rerot-compose \
  --model /models/instruct.gguf \
  --gpu-layers 99 --devices Vulkan0 --split-mode layer --parallel 2 \
  --ctx-size 4096 --total-kv 4096 --batch-size 512 \
  --ubatch-sizes 128,1 --phase-modes release,retain --repeat 2 \
  --require-cross-shape-equivalence \
  --output build-rerot-matrix-vulkan-candidate-01
```

The driver resolves the actual native microbatch size and requires it to match
each requested shape. A GPU run must also show positive model-layer offload.
Inspect `config.native_context.batch_invariant_matmul_requested` and retain it
with the receipt. It must be `true`; the recorded source identifies C's
configuration before context creation and prefill. The getter reports the
retained request, not a history of every operation's effective policy. The
public projection gates above supply separate operator-invariance evidence.
The cross-shape flag makes differing greedy trajectories fail the candidate
run. Omitting it records cross-shape observations as `NOT_QUALIFIED`, even when
each shape's phase and repetition controls pass. CPU and Vulkan trajectories
are not compared to each other by these commands.

| Claim checked by the real trace | Required observation |
| --- | --- |
| Incremental shared work | Exact prior public key identities; repeated BODY growth while a peer remains active |
| Frozen cohort | Every live pen appears once; one current BODY key per reader; current peer keys remain absent |
| Completed non-ancestor visibility | A recorded reader/peer witness, with the peer already sealed and at least one committed BODY token |
| FRAME publication | The final reader view contains its complete causal FRAME; `frame_tokens_by_stage` records its actual size |
| Dependency admission and pen lifetime | Every parent has sealed before admission; a live stage retains its pen; synthesis starts last |
| Private source end | Completed public tapes retain their keys but omit each source-end key from peers |
| Phase release | `config.phase_memory` matches the command; exactly one release at each FRAME-to-BODY transition; every `after` is zero |
| Retained compute | `config.phase_memory` is false and no release event occurs |
| Output completion | Tokens belong to their committed frontier, text offsets are contiguous, and each stage closes through native EOG |

The trace records the last view of each FRAME, not every internal wave's A
input. Intermediate FRAME causality, attention arithmetic, and native state
ownership therefore also need the public A seam tests and the executor's
lifecycle checks. A custom `--stage` graph keeps the full public-key assertions
but may not witness a completed non-ancestor; its receipt records that coverage
as `NOT_QUALIFIED` when absent. The default fixture requires the witness.

## Cancellation and output failure

Run the process failure checks on each candidate backend. This example chooses
`ubatch=1` to cover the small submission shape separately from the native CPU
abort test:

```sh
python3 fork/rerot-compose/tests/real-interrupt.py \
  --binary build-rerot-cpu/install/bin/tq-rerot-compose \
  --model /models/instruct.gguf --gpu-layers 0 --split-mode none \
  --ubatch-size 1 --cases sigint,broken-pipe \
  --output build-rerot-interrupt-cpu-candidate-01

python3 fork/rerot-compose/tests/real-interrupt.py \
  --binary build-rerot-vulkan/install/bin/tq-rerot-compose \
  --model /models/instruct.gguf \
  --gpu-layers 99 --devices Vulkan0 --split-mode layer \
  --ubatch-size 1 --cases sigint,broken-pipe \
  --output build-rerot-interrupt-vulkan-candidate-01
```

Each driver injects after two observed dual-worker, non-EOG BODY cohorts.
SIGINT must terminate through C with status 130 and a terminal incomplete
cancellation error. Closing the only stdout read end must reach C's output
error path with status 1. Any timeout or forced cleanup fails the check. These
commands use the driver's phase-release default and do not inject during a
FRAME or test SIGTERM. The observed process boundary cannot identify the exact
GPU instruction at signal delivery or independently prove every device drained.

The model-dependent native cancellation test additionally covers cancellation
during model loading, an actual CPU graph abort, sticky stop, drain, and transfer
of an exception raised by the cancellation query. Register it explicitly:

```sh
cmake -S fork/rerot-compose -B build-rerot-cancellation \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/installed/cpu \
  -DBUILD_TESTING=ON -DTQ_REROT_CANCEL_TEST_MODEL=/models/instruct.gguf
cmake --build build-rerot-cancellation --parallel 4 \
  --target test_rerot_compose_cancellation
ctest --test-dir build-rerot-cancellation --output-on-failure --no-tests=error \
  -R '^tq_rerot_compose.cancellation$'
```

The model-free `tq_rerot_compose.whiteboard_validator` CTest checks deliberately
invalid traces, including missing or misplaced release, retained bytes, lost
completed-peer keys, early dependency admission, and false ancestor witnesses.
Those fixtures test the checker and never count as model execution.

## Projection performance and dispatch

Measure the cost of the consistency request separately from correctness.
The [quantized projection timing procedure](../tests/a-seams/VULKAN_QUANTIZED_PERFORMANCE.md)
compares DEFAULT and mode 20 in the same normal Release Vulkan installation,
using resident `SCALE -> MUL_MAT` graphs at N=9/32/128. It records the first
computation and 15 consecutive warm computations per mode, with a second run
that reverses which mode starts first. The source and weights are shared
within each pair, and every sample rewrites the consumed activation.

Use its clean runtime environment and exact command sequence; retain all raw
samples, medians, dispersion, mode order, build options, and resolved library
and device identities. A warm-median mode-20/DEFAULT ratio greater than one
reports extra time for that tested resident graph. The timing branch checks
finite, nontrivial, stable outputs within each mode; its performance result
does not supply the independent DEFAULT N=1 equivalence gate above. It covers
dense projections, not ID-operator or whole-model throughput.

The separate [MMVQ dispatch procedure](../tests/a-seams/VULKAN_MMVQ.md) requires
actual dispatch records in completed DEBUG graph intervals and the matching
numerical gate. Shader creation, environment flags, and native tensor trace
names alone do not prove MMVQ execution. The existing trace checker covers its
own N<=8 fixture, not every mode-20 dense or ID graph. DEBUG and timestamp
diagnostics retain separate receipts from normal Release timing.

Whole-model elapsed time and token throughput require the actual completed
model workload, placement, phase mode, prompt, and native EOG result. A
DEFAULT run that fails the strict generation contract cannot be advertised
as an equivalent faster implementation. Report any observed cost of the
qualified policy with its workload and candidate identity, rather than
generalizing the synthetic operator ratio.

## Capability and artifact evidence

`config.native_context` reports resolved batch capacities, K/V types, the
reader precision policy and its reason, `batch_invariant_matmul_requested`,
the configured FA state, the current pipeline-parallel setting, and the model
loader's selected devices. The batch-invariance field reads A's public context
getter and proves only that the request remains configured. The frontier and
result events read the pipeline setting again because A can disable it after
graph allocation. The `devices` inventory, requested placement, selected
devices, actual model buffers, and actual kernel activity are distinct observations.

| Capability | Evidence still needed beyond the generation trace |
| --- | --- |
| Reader numerical correctness | Strict full-vocabulary seam comparisons and the independent attention oracle for the exact model/backend scope |
| Quantized projection consistency | Independent native DEFAULT N=1 operator comparisons for the actual tested backend/type/layout; a retained mode request or a short prefill diagnostic is insufficient |
| Multi-device PP | Real buffers on multiple devices, enabled native pipeline state throughout the relevant execution, and actual execution evidence; `layer` alone is a request |
| MMVQ | An observed native kernel dispatch and a controlled timing comparison for the executed batch shape |
| F16, mixed, or quantized reader KV | A supported precision path and its strict permutation/restore/numerical results; this C requests resident F32 K and V |
| Whole-model TP5, LateBind, MTP | A complete installed execution provider with correct shared-frontier validation, invalidation, completion ownership, and drain |
| Automatic planning, reasoning/tool routing | Their own product contracts and executable composition beyond the explicit `assistant-eot` DAG |

`--require-multi-device` verifies positive model-buffer logs on at least two
non-CPU devices. It does not by itself assert PP overlap or a speedup. The
unavailable `--tp5`, `--latebind`, and `--mtp` modes remain errors; a refusal is
not an implementation of their execution behavior.

Each real receipt hashes its driver, executable, and every GGUF shard. Add
`--artifact /actual/installed/library.so` once for every installed A/B shared
library relevant to the run, and retain the loader-resolved library paths with
the build record. `--source-commit` is a caller-provided annotation, not source
provenance verification. Keep the exact exception manifest, source digests,
build options, and precision-environment settings beside the new receipt.

Append the new run identity and result to `fork/VALIDATION.md` and its evidence
index. Keep previous FAIL receipts, commands, hashes, and their original scope
unchanged, including F16 permutation/restore failures, earlier cross-shape
failures, the GPU independent-oracle failure, and unsupported Vulkan row split.
A later passing candidate may qualify its own bytes and executed scope; it
does not change what the earlier candidate did.
