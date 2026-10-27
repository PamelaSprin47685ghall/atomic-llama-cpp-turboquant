<!-- two-requests-current-status-20261021 -->
## 2026-10-21：双请求保留版本与最终测量；每路 47 tok/s 目标未达到

保留版本、证据和入口见[双请求报告](../../build-tp5-two-requests-20261021/TWO-REQUESTS-REPORT.md)；以下旧正文保留。本次增量为 3 个 A 源文件和 1 个 C 源文件，公共 API 不变；使用原始模型权重、每请求 262144 容量的 Q4_0 K/V，并保留 Swift 共享权重派生 view 生命周期修复及五 rank host 提交顺序 gate。

同 prompt 双请求组（n=2 个独立进程），每路 512 token：两路几何均值为 **37.629332 / 37.624558 tok/s**，aggregate 几何均值 **74.923025 tok/s**，范围 **74.847236–74.998891**。相同 C 版本的单请求观测（n=1）为 **46.071845 tok/s**；aggregate 对单请求的 **1.626222×** 仅作描述性比较。不同 prompt 的一个双请求案例为 **34.571716 / 34.899258 tok/s**，aggregate **69.062687 tok/s**。**两个请求各 47 tok/s（合计 94 tok/s）的目标未达到。** 历史 **47.34 tok/s** 只属于旧 F16 单请求配置。

完整 16-token trace 已通过，49 组五 rank 提交保持一致次序，但该完整 tracing 的共同生成期 GPU overlap 为 **0**。另一次相同保留版本的 kernel-only 采样已通过：共同输出阶段 GPU0–3 观察到跨请求算术/算术与算术/consume 的真实交集，GPU4 为零；含预填充的完整活动阶段五卡均有局部交集。由此可确认存在实际 GPU 重叠，但不能宣称五卡共同生成期都已重叠或通信完全隐藏。各次观测和证据分别保留，详见双请求报告。

---

# TQTP5Compose

<!-- TP5_NATIVE_SPEED_20261021_FINAL -->
## Latest retained result: native TP5, 2026-10-21

The current retained native TP5 configuration is the 2026-10-21 four-pair-qualified build. Older “current”, “final”, and gap-failure entries below are historical. The user's relaxed numerical policy applies to native acceptance; explicit strict-reference modes retain their separate gates.

The retained launcher is `build-tp5-gap-repair-20261021/run-retained.py`, using C `native-policy-compose-install` with A `native-pad1024-install`. The numerical-policy receipt is `original-weights-native-backend-activations-ordered-f32`. Original weights, CPU ordered F32 relay, all five ranks, ownership, and drain remain required.

Four balanced pairs against fixed master `46a6fd8a129091741f9ebc6431f7f610abe1ccfc` passed: geometric-mean throughput **47.343475 vs 45.366353 token/s**, paired ratio **1.043581249**, two-sided 95% CI **[1.037961336, 1.049231589]**. Every run includes all 511 intervals from 512 natural outputs.

Read the [final report](../../build-tp5-native-speed-20261021/NATIVE-SPEED-REPORT.md) for the natural-generation checks, full-capacity evidence, installed identities, and the distinction between capacity allocation and a filled long-context benchmark.

<!-- /TP5_NATIVE_SPEED_20261021_FINAL -->

`C` combines installed llama/GGML `A`, `TQRelay`, `TQScheduling`, `TQText`, and `TQSampling`.
It does not copy model arithmetic, include private A headers, or make a B library
call another participant. The installed executable is `tq-tp5-compose`.

Current numerical/performance evidence is in the
[single-request repair](#single-request-performance-repair-2026-10-19).
Historical sections retain their original image identities.

## Finite greedy composition (2026-10-21)

C now calls the independent public `TQSampling::sampling` package for complete-row finite greedy selection. The operation checks every logit, rejects either infinity and every NaN, and selects the smallest id on equal maxima, preserving the selected input bits. It owns its result and uses runtime AVX2 dispatch with a scalar fallback. The separate shortlist API keeps its unique-maximum and negative-infinity-mask contract. B includes no model dependency, and C's installed package declares the transitive sampling dependency.

The current Swift repair identity and limits are in `build-tp5-gap-repair-20261021/REPAIR-REPORT.md` at the repository root and the dated entry in `../VALIDATION.md`. Latest scoped performance is 32.731905 token/s, still below the fixed master's earlier approximately 45.30 token/s; it is not a master noninferiority qualification. Historical sections below retain their original identities.

## Supported execution contract

The initial strict profile is the real **Qwen3-8B Q8_0** model: 36 layers,
4096 hidden dimensions, 32 query heads, 8 KV heads, head dimension 128, and
12288 intermediate features. Five distinct HIP **RX 6800 / gfx1030** devices
share the model's tensor computation. Other model architectures, adapters,
host-routed graph splits and incompatible geometry are refused, not mirrored
into a fake five-rank execution. Ordinary reference inference is a separately
named mode and never a post-submission fallback.

The HIP build keeps generic `GGML_HIP_GRAPHS=OFF`. The strict provider owns a
separate, bounded whole-epoch HIP capture/replay on each rank's stream zero.
Each rank has its own lane and bounded scratch; the two mutable contexts do not
share input tensors, output arrays, graph arenas, streams or KV state. They share
only the model's read-only weights. Up to two windows support up to four
independent requests, with at most 512 consumed/generation tokens reserved per
request. Each request has one local sequence writer. Every model
call fits one microbatch, with one known input token per active request.

A prepares every rank/stage and mapped communication allocation before starting
an epoch. Each partial-sum boundary uses two bounded single-workgroup kernels:
producer copy plus all-writer publication; exact-epoch wait plus output copy and
all-reader acknowledgement. Payload copies use four-value transfers with scalar
tails, without imposing 16-byte alignment on terminal views. All stages and
continuations are submitted without waiting for the CPU reducer. The CPU service
starts before any model call, serves all windows, and may publish only after the
frame is sealed. It sums F32 rank contributions in rank order. There is no CPU
continuation launch, GPU allreduce fallback or P2P requirement.

An exact frame first executes directly to warm backend resources, then captures
the complete model/relay chain. All five executables exist before the first
actual graph launch; captured/replayed execution seals normal CPU publication
only after all five launches are accepted.
At most two definitions are cached. Their signatures include ordered topology,
source/view ancestry, types, dimensions, strides, op parameters, flags, fusion
use counts, real storage addresses and the numeric policy. A changed KV geometry
or binding misses rather than reusing stale kernel arguments. Captured relay
kernels read an activation word updated only after prior native and CPU-reader
drain; they never capture a stale epoch scalar. Graphs die before their owned
scratch and mapped storage. Partial launches cannot fall back.

Hot polling reads mapped protocol words, not five sets of HIP tail events on
every spin. Snapshot and close retain actual native completion queries. C folds
independent elements with SSE2 where available, preserving each element's
rank-0-to-4 F32 addition order and finite-input/result checks. The decoded-Q8
projection uses four output rows per block and a fixed subgroup reduction,
without LDS barriers or activation requantization.

The numerical policy is explicit:
`q8_0-decoded-weights-f32-activations-ordered-f32`. Q8_0 weight blocks retain
their original bytes and scale interpretation; the HIP projection kernel reads
those blocks directly and multiplies their decoded values by F32 activations.
Activations are not requantized to Q8_1. This avoids amplifying tiny partial-sum
rounding changes through discontinuous activation quantization. The single-GPU
reference uses the same declared policy, not A's ordinary MMVQ approximation.
Neither policy is claimed bitwise equivalent to a different floating-point
accumulation order; the complete fixed-trajectory numerical gates remain.

Before either window starts, each provider reserves 72 mapped channel slots,
each with capacity for four complete 4096-element F32 rows, plus 128 MiB of
native scratch and a 1 MiB quantization cache per rank. Runtime shape changes
only rebind validated active extents; they do not allocate mapped memory behind
another window's waits. Actual channel and transfer counters exclude unused
capacity. The logical token reservation does not stand in for those physical
allocations.

Native graph construction/instantiation has additional runtime-owned storage;
it is not included in the fixed scratch figures above. The two-entry bound is
not a measured total VRAM-peak claim.

Strict context creation keeps K and V caches in F32 and selects A's canonical
V-cache layout for both attention settings. Flash Attention defaults to off
(`--flash-attn off`); explicit on remains supported. Historical pre-replay
performance below is not a performance claim for the repaired native image.
Ordinary attention is also the explicit numerical reference, using its existing GPU transpose and contiguous-
view path rather than an unsupported split-row write into the transposed cache.
Both reference and relay contexts use the selected setting, never auto mode.
Context preparation checks A's public resolved-setting getter and refuses a
mismatch instead of silently falling back. The cache's own layout remains part
of its state format.
For this explicit decoded-F32 profile, Flash Attention uses the existing fused
online F32 implementation with F32 Q/K/V operands and accumulation. The profile
preserves F32 cache views through graph construction instead of casting K/V to
F16 before attention. Ordinary FA routing and precision are unchanged; they do
not inherit this profile's precise-F32 contract. The shared online F32 kernel
now excludes negative-infinity masked keys before loading K/V or computing QK,
so masked nonfinite values cannot contaminate visible outputs. This improves
shared banded-kernel masked-value semantics too; it is not a claim that every
other backend behavior is unchanged. Finite unmasked accumulation order is preserved.
The on and off paths still have different attention accumulation orders and
require separate numerical qualification.
Input embedding offload is requested explicitly at model load. Shared model
split metadata is synchronized; per-window graph rebuilds do not rotate another
window's mutable arena or clear a shared cache underneath a reader.

Explicit `--flash-attn on` has passed the real five-RX-6800 TP5
numerical and lifecycle gates for this Qwen3-8B Q8_0 profile. The four-request,
eight-token same-setting reference/relay comparison checked 32 full-vocabulary
rows (4,861,952 logits): zero failures under `0.002 + 0.001 * abs(reference)`,
maximum absolute difference `2.193450927734375e-5`. The 2-second delayed-window,
cancellation and injected-abort cases also passed: peers progressed, requests
1/2/3 survived cancellation, both native completed masks were 31 after abort,
and all reservations closed. The separate actual FA-on trace recorded 720 fused
launches, 144 on each physical device; the actual FA-off trace recorded zero FA
launches. These are executed profile-specific facts,
not qualification for other models, arbitrary shapes or a speedup. Exact
identities and retained early failures are in the
[TP5 machine index](../audit/tp5-acceptance-20261019.json) and
[validation record](../VALIDATION.md).

Qualification does not make FA the performance default: the paired two-window
measurement found on/off speed ratio `0.87325` (95% CI `0.83966..0.90818`), about
12.7% slower with FA on. The default therefore remains off, with fully qualified
FA available explicitly. Earlier treating the user's speed concern as a demand
to enable FA by default was an assumption, not a measured result.

Embeddings and norm parameters retain A's explicit mirrored layout; the large
attention and FFN projections are tensor-split. Whole KV groups stay together:
five nonempty head-group allocations contain 1/2/1/2/2 of the 8 groups, and Q
heads follow those groups. FFN input partitions keep the original Q8 blocks.
This is not a claim of zero physical replicas, exact five-way timing balance,
or arbitrary model support.

## Build and install

From the repository root, with the selected ROCm SDK installed:

```sh
python3 fork/build.py --backend hip --tp5 \
  --hip-root /absolute/path/to/rocm --build-dir build-tp5 --jobs 8
```

`--tp5` selects gfx1030, disables HIP graph capture and includes Flash Attention
compilation so the CLI can explicitly select either attention path. The pipeline verifies all A changes,
installs A, and builds B and C from their installed dependencies. The HIP runtime
is an external dependency, not bundled; its imported path is included in A's
runtime search path. Set the right `--hip-root` for the target installation.
There is no environment-variable switch that silently enables strict mode for
ordinary callers: A's public `tp5_relay` context parameter defaults to false.

For a standalone C build, provide an installation containing the matching A,
`TQRelay`, `TQScheduling`, and `TQText` packages:

```sh
cmake -S fork/tp5-compose -B build-tp5-c \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/packages \
  -DCMAKE_INSTALL_PREFIX=/absolute/path/to/packages
cmake --build build-tp5-c --parallel 4
ctest --test-dir build-tp5-c --output-on-failure --no-tests=error
cmake --install build-tp5-c
```

The installed acceptance helper is exactly
`<prefix>/share/TQTP5Compose/real-tp5.py` (capitalization is significant), not a
guessed lowercase documentation path. It can be invoked with Python against the
installed `<prefix>/bin/tq-tp5-compose`.

The public C target is `TQTP5Compose::runtime`; its owned options and results are
in `<tq/tp5/runtime.h>`. The CLI is the isolated GPU execution domain. An unproved
native drain exits that process without destructors that would free in-flight
owners. Do not host this fail-stop operation inside an unrelated long-lived
process and assume it can recover a broken device in place.

## Model identity

The qualification weight is obtained from
[unsloth/Qwen3-8B-GGUF at a6adef130ffb23ddaf1a62fec9dced968c9bc482](https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/a6adef130ffb23ddaf1a62fec9dced968c9bc482):
`Qwen3-8B-Q8_0.gguf`, 8709519168 bytes, SHA-256
`0cfbf745760f07a76ddeb358dd025a27f2e11d1ca9c9a4169a373d52990fe86e`.
The Qwen3 base model uses Apache-2.0; model distribution terms are separate from
this package's MIT license. Do not infer a new weight file's identity from its
basename. Keep the model file stable throughout load and execution.

## Run

Create a new output directory, then run four requests through two windows:

```sh
mkdir -p results-tp5
build-tp5/install/bin/tq-tp5-compose \
  --model /absolute/path/Qwen3-8B-Q8_0.gguf \
  --prompt 'Explain tensor parallel inference.' \
  --prompt 'Write a short description of a tree.' \
  --prompt 'What is a CPU reduction?' \
  --prompt 'List three prime numbers.' \
  --windows 2 --predict 32 --ctx-size 512 --deadline 300 --flash-attn off \
  --output results-tp5/run.json
```

This CLI uses raw prompt text and exact greedy selection of the full finite
vocabulary, with smaller global token IDs winning ties. It does not silently
apply a chat template, stochastic sampler, grammar or speculative decoding.
Prompts are tokenized with A's declared special-token insertion. `--ignore-eos`
is available for fixed-work qualification; ordinary runs stop on EOG or budget.
Text is assembled from the accepted tokens with bounded UTF-8 stream handling.

`--windows 1` executes the same strict five-rank model in a single context and
is the controlled TP baseline. `--mode reference --windows 1` executes the same
decoded-F32 model on one HIP device; `--windows 2` can match the two-row contexts
for a shape diagnostic. This reference has no CPU/partial-offload fallback.
The modes have different explicit execution labels. They do not satisfy strict
relay merely because they return a plausible sentence.

`--flash-attn` accepts exactly `on` or `off` (default `off`), including in reference
mode. Select `--flash-attn on` for the qualified fused F32 path; `off` is both the
measured faster default and ordinary-attention numerical reference. Do not compare it to an on-path relay and call that a
same-setting reference comparison. There is no retry with attention disabled
when the enabled path fails preparation or execution.

The report contains every request's input token, consumed position, sampled and
emitted token, commit timestamp, final reason and text. It also records actual
rank graph/node/relay-launch counts, endpoint payload bytes, native completion,
CPU input and last-reader retirements, physical BDF mapping and final reservation
counts. `options.flash_attention` records the request; `run.flash_attention`
records the resolved setting verified for every context. These booleans are not
per-kernel launch receipts. GGML compute nodes are not presented as HIP kernel counts.

## Full-logits and lifecycle acceptance

`-DTQ_TP5_BUILD_HIP_TESTS=ON` also builds `tq_tp5_flash_f32_tests`, registered as
`tq-tp5-flash-f32-oracle`. On every discovered ROCm GPU it compares every F32
attention output against an independent F64 stable-softmax oracle at fixed
absolute tolerance `2e-5`. The fixture covers head dimension 128, GQA group 4,
1..4 query rows, 256 KV positions, distinct causal F16 masks, and full 32-head
plus local 4/8-head shapes. Every shape runs with F32 Q/K/V in three fixtures:
F16-exact dyadic operands isolate accumulation error, while non-F16-exact F32
operands also detect unintended operand rounding. Visible V values are positive
and near 1..2; the always-masked tail contains distinct finite values near 8..9.
The third fixture puts NaNs in K/V only at always-masked positions 237..255;
visible outputs must remain finite and within the same oracle bound.
Each case prints its operand dtype and fixture. The fixed tolerance applies to
the precise F32-input profile, not ordinary raw F16-input FA. Source bytes and guards
outside the backend's complete allocation (including its FA conversion tail)
must remain unchanged. This real-device arithmetic test is separate from
full-model numerical/lifecycle qualification and does not establish performance.

```sh
python3 fork/tp5-compose/tests/real-tp5.py \
  --binary /absolute/path/to/packages/bin/tq-tp5-compose \
  --model /absolute/path/Qwen3-8B-Q8_0.gguf \
  --output-dir results-tp5/acceptance \
  --predict 8 --deadline 300 --delay-ms 2000 --flash-attn off --atol 0.002 --rtol 0.001
```

The output directory must not already exist. Each case has one attempt; errors
are preserved rather than retried until a pass appears. Optional repeatable
`--expected-bdf` arguments pin the five-rank mapping. This runner executes a
single-GPU reference, replays its emitted tokens in strict dual-window teacher
forcing, and compares every raw logit with predeclared tolerances. `--flash-attn`
defaults to `off` and is forwarded to every case, including abort; successful
reports must record that exact resolved setting. Run again with `--flash-attn on`
and a distinct output directory to qualify the enabled path separately.
Each comparison uses the same setting for reference and relay; the
numeric tolerances are unchanged between settings. The runner separately
checks a delayed CPU publication, request cancellation and a reached abort
injection with both native-drain receipts and reservation cleanup. Candidate tolerances and trace timings are not a performance default
promotion or a general model-quality guarantee.

The five-device arithmetic receipts additionally passed 170 Q8/F32 cases and
180 FA cases; the latter's maximum F64-oracle absolute error was
`2.24640083e-6` against its fixed `2e-5` bound. Existing `acceptance-fa-on`,
`acceptance-fa-precise` and `acceptance-fa-cast` failures remain evidence rather
than being overwritten by the final `acceptance-fa-final` pass. The earlier
`acceptance-fa-f32` receipt retains its candidate identity. The 85-command
archive isolation pass belongs to a 45-path pre-FA candidate, not the final
48-path A manifest. The final HIP package has separate successful relocation and
installed-consumer receipts below; they do not rerun the full archive flow. No blanket model,
backend, long-context, MTP, LFM2 MoE TP5 or throughput claim follows here.

Final-source FA-off acceptance passed the complete eight-token main/slow/cancel/
abort matrix. FA-on also passed a 32-token **main-only** comparison: 128 rows,
19,447,808 logits, maximum absolute difference `9.059906005859375e-5`, and zero
nonfinite values or tolerance failures under the same `.002/.001` gate. This
does not extend the lifecycle matrix to 32 tokens or qualify long contexts.
The earlier-candidate **host-only ASan** receipt passed main plus abort with four
tokens/request. The current-source `asan-long-final.json` additionally completed
the original four 193-token prompts through two contexts with two generated
tokens/request, FA on, exit 0 and zero final reservations. Leak detection was
disabled. These are neither a GPU sanitizer nor evidence that all
lifecycle cases are sanitizer-clean or that allocations never leak. The final
HIP build completed seven-B/seven-C build/install stages and 11/11 A seams; its
subsequent guard rejected a concurrently changed root-note digest, separately
from those runtime/build results. Final identity and guard receipts belong in
the machine index.

The real 193-token prompt workload is retained. Strict preflight now skips
legitimate empty/no-output prefill tensors instead of rejecting them; it does
not request artificial prefill logits or shorten prompts to make a gate pass.

## Historical pre-replay installed-package and relocated-consumer proof

`build-tp5-runtime/hip-package-proof/summary.json` passed its selected contracts.
All 15 components (A + seven B + seven C) were freshly installed to stage, then
renamed to relocated. The original stage and old install prefix were genuinely
unavailable throughout consumer configure/build and all device calls, and were
restored only after the proof. The user source tree was not moved; the external
consumer used only `find_package(TQTP5Compose)`, installed public headers and
imported targets. Compile-path checks found neither old-prefix imports nor
private source headers; `ldd` resolved the relocated libraries.

The external two-request consumer executed explicit FA on through two contexts,
five native ranks and zero final reservations. A direct two-request CLI run used
default off. A single-request CLI automatically selected one window/off, and
three requests in one window with explicit on also passed. The relocated helper
at `share/TQTP5Compose/real-tp5.py` passed main-only full-logits acceptance with
predict4. The initial harness's wrong guessed helper path exited 2 before its
child started; that failure remains recorded, and correcting the actual installed
path did not repeat successful setup.

The final package's `final-dispatch-summary.json` binds five actual BDFs to 720
on-path fused launches (144/device) and zero off-path FA launches. Its
`performance-image-binding.json` confirms byte-identical native libraries to all
12 measured performance processes. The published CLI changed the default to off;
every performance case supplied its setting explicitly, with no kernel change
after measurement. Source guard and final machine-index status remain separate
receipts, not a blanket all-gates PASS.

A combined scenario requested a delay on window 0 and observed abort on window 1.
Both native completed masks were 31, unknown masks 0, reservations zero, with
exit 1 and no success report. The exact held-borrow branch was not separately
instrumented, so this receipt does not claim branch coverage of that path.

## Single-request performance repair (2026-10-19)

`build-tp5-performance-20261019/paired-selected/summary.json` confirms the repaired
image against the preserved installed pre-replay image. Four independent paired
blocks each run baseline/off/on in a frozen order, with one request, the same
Qwen3-8B Q8_0 file, five physical ranks, five prompt tokens and 64 fixed teacher
outputs. No profiler, full-logit capture, compiler or concurrent GPU workload
runs during measurement. The endpoint is 63 actual inter-token intervals;
capture costs inside those intervals are not dropped.

| Image / attention | Geometric-mean single-request token/s |
| --- | ---: |
| Preserved pre-replay image / off | 9.1784 |
| Repaired image / off | **54.9584** |
| Repaired image / on | 49.9261 |

Off/baseline speed ratio is **5.98781**, paired-log-ratio 95% Student-t interval
**[5.29900, 6.76615]**, df=3. The earlier 11.30575 baseline and 56.23309 repaired
single runs were screening observations, not the denominator/numerator used for
this confirmation. On/off is **0.90844**, interval **[0.88025, 0.93752]**; off
therefore remains the default. These are request rates, never aggregate
throughput divided by a request count. They exclude model/context startup,
network and pre-admission queueing; 63 intervals do not qualify population p99.

The same repaired image also reran the prior four-request workload, each with
193 prompt tokens and 32 teacher outputs, in three paired blocks. One/two-window
per-stream rates are 38.4234/39.5584; the two/one ratio is 1.02954, interval
[0.75777, 1.39878]. This does **not** confirm a new concurrency speedup. The
existing window-selection policy is retained: one for a single request, two for
multiple requests unless explicitly selected. Do not carry the historical
1.71597 two-window ratio below into claims about this native image.

Master's retained no-MTP report records 64.77485/67.14624 token/s on
APEX-I-Compact mixed-precision weights with aggressive model-specific LateBind
and Vulkan. It is not this dense Qwen3-8B Q8_0/HIP/F32-activation workload, so no
cross-model speed ratio is claimed. The transferable implementation gap was
whole-chain native submission, not a reason to copy its framework or change
this model's precision.

The selected image passes both attention modes' complete 16-token
main/slow/cancel/epoch-8-abort matrix. Each full-vector comparison checks
64 rows / 9,723,904 logits under the unchanged `.002 + .001*abs(reference)`
gate: zero nonfinite/out-of-tolerance values. A separate 288-output single-request
comparison checks 43,757,568 values through input position 291, including actual
KV-256-to-512 recapture: maximum absolute difference 0.0005264282, zero failures.
Its HIP API trace records five late capture begins and subsequent graph launches;
it is numerical/lifecycle evidence, not a performance measurement. Host C/C++
ASan main plus epoch-8 abort also passes; leak detection is off and HIP translation
units/device accesses are not sanitizer-covered. The five-device Q8/F32 oracle
now covers 1,210 cases; the existing attention oracle covers 180 cases.

## Historical pre-replay per-stream performance and default selection

The preceding release image's
`build-tp5-runtime/per-stream-fa-final/summary.json` records 12 processes in
three paired blocks, each with the same four requests, 193 prompt tokens/request
and 32 fixed teacher outputs/request. These are geometric-mean **per-stream**
rates, not total process throughput:

| Attention | Windows | Per-stream token/s | Admission-to-first-commit TTFT (ms) |
| --- | ---: | ---: | ---: |
| off | 1 | 8.9853 | 25568.1 |
| off | 2 | **15.4185** | 14059.6 |
| on | 1 | 7.7510 | 26719.4 |
| on | 2 | 13.4642 | 14748.4 |

The paired on/off two-window ratio is `0.87325`, 95% CI
`[0.83966, 0.90818]`, using per-block means of request log ratios and Student t
with 2 degrees of freedom. FA on is about **12.7% slower** here, so it is not
promoted as the performance default. Off remains the measured fastest qualified
mode; on remains explicit and fully numerically/lifecycle qualified.

For the selected off mode, two/one-window per-stream ratio is `1.71596659`,
95% paired-t CI `[1.28598327, 2.28971978]`, from the same three blocks
(`per-stream-fa-final/default-decision.json`). This is a workload-specific
two-window benefit, not a universal concurrency or FA speedup claim.

Only three independent blocks were measured. Each request contributes 31
inter-token intervals; p99 is near the maximum, not a stable population-tail
estimate. TTFT runs from request admission to first committed token and excludes
model/context startup, pre-admission queueing and network transport. Measurements ran without profiler, full-logit
capture or concurrent hardware work. No universal model/hardware/long-context
speed claim follows. The original `per-stream-fa` failure and the slower
`per-stream-fa-prefill-fixed` candidate remain preserved; the final result does
not erase either attempt.

`--capture-logits` publishes `.logits.f32le` and `.logits.rows.jsonl` companions.
The binary has no ambiguous row header: the JSONL and final report declare exact
vocabulary, count, byte offset and request-major/token-index order. Every row has
its consumed input identity. A report is published only after complete companion
files; existing files and symlinks at output names are never overwritten.
`--teacher-file` contains one ASCII-space-separated token row per prompt, exactly
`--predict` nonnegative IDs each, within a 1 MiB input bound. The final teacher
token is emitted but is not consumed by a nonexistent next model step.

## Failure and resource ownership

Input completion and output completion are different facts. After reducing all
rank inputs, C retires input leases; it retains output leases until A reports
all actual last reads. The next epoch cannot replace a still-borrowed view or
unretired output. Requests are not simultaneously written by two windows.

`--cancel-request N --cancel-after K` cancels request N after K accepted tokens.
It stops new input for that request; its peer and other window continue. Already
submitted work drains under the original shape before state removal. Every
request's state is removed and synchronized before returning its reservation.

`--delay-window N --delay-publish-ms M` is a controlled once-per-run experiment,
not a normal-path sleep. The CPU holds that window's borrowed view while serving
the other window. `--abort-window N --abort-channel M` is explicit failure
injection. `--abort-epoch E` selects a positive epoch (default 1), allowing
post-warmup replay failures to be exercised. Reaching it must return a nonzero
result, not a success report; the marker records the actual window/channel/epoch.
The normal deadline requests abort, then permits a bounded 45-second drain grace.
Failure to prove completion returns a retained-resource fail-stop exit (125);
that is not a clean-abort pass. No timeout fabricates last-reader acknowledgement,
clears active control pages or replays the graph through another mode.

See [the current validation record](../VALIDATION.md) for actually executed
commands and evidence. Source existence, `--help`, CPU-only self-tests and
platform probes alone do not establish this full-model profile's qualification.
