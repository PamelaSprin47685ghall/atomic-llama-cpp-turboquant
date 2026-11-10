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

The PLE/KV diagnostic trace uses the versioned public
`llama_memory_hybrid_idx_get_trace` query from `<llama.h>`; `runtime.cpp`
includes only installed public A/B headers. The Meta submit-hook declarations
live in `<ggml-backend.h>`; the obsolete `tp5-meta-submit-ready.h` has been
removed, so new code includes `<ggml-backend.h>` directly.
The NextN/MTP embedding entry points are public in `<llama.h>`; `runtime.cpp`
carries no manual declarations for them.

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

### Host pooling

The worker-local MTP batch pool reuses four one-token batches; the token storage
is inline and the pool is allocated once per worker. The piece scratch keeps at
least 256 bytes, grows for a piece longer than the current buffer, and never
shrinks, so a token piece does not zero-fill the buffer again. With
`TP5_TEXT_POOL=1`, each request pre-reserves a conservative
emitted-text bound, and `push_into` grows the text geometrically past it. The pool flags
(`TP5_UBATCH_POOL`, `TP5_KV_SNAPSHOT_POOL`, `TP5_DECODE_SCRATCH_POOL`,
`TP5_MTP_BATCH_POOL`, `TP5_TEXT_POOL`) are exact-`1` and default off; the
default path keeps the pre-pool allocations.

With `TP5_KV_SNAPSHOT_POOL` or `TP5_DECODE_SCRATCH_POOL` on, the pooled KV
prepare trials every ubatch except the last only when the cache has no SWA
(`n_swa == 0` and `swa_type == LLAMA_SWA_TYPE_NONE`). No later `find_slot`
observes the last placement, so its output slot is written directly and the
snapshot/apply/restore is not spent on it. With SWA active, every ubatch is
trialled: `apply_ubatch()`'s `seq_rm` purge removes cells outside the trial
slot and is not restored, so deferring it would change what earlier ubatches'
graphs read. The reference path (`prepare_reference`) keeps the full trial
sequence.

The CPU counterexample compares cache state on a real `llama_kv_cache`; the
non-SWA control checks cache-state equivalence only and does not prove the
last trial is skipped. A prepare trial activates a per-stream cell journal:
`apply_ubatch()` and its SWA `seq_rm` purge record every mutated cell,
including cells outside `sinfo.idxs`. The journal records each cell's
pos/ext/seq/shift; a generation mark deduplicates a cell. The trial never
changes the stream's `has_shift` flag, so the journal does not save it. A throw replays the journal in reverse and restores the
heads, so the full write domain returns to the pre-prepare state; the original
exception propagates. Two layers: the global journal owns the nodes of the pre-prepare state and
the reverse replay restores the full transaction on an exception; the journal
keeps all trial entries in one flat vector plus each trial's range, so a trial
owns its snapshot state and the nodes it removed without a per-trial buffer.

The failure counterexamples use the test allocator as a probe: it compares the
real cells at every allocation and arms the fault only after the target
stream(s) have changed, recording the allocation number instead of pinning an
index. A no-op replay fails the full-state check.

The `n_swa=0` case covers the journal replay branch only; it does not execute
an SWA purge. The SWA purge path is covered by the separate SWA purge tests.

The recovery state records one trial index per stream: a stream that an
earlier ubatch did not touch keeps its own count. All states' indices are one
flat block allocated once per prepare, not a vector per trial; the block is
allocated only when the journal is active (SWA), the pooled path reuses the
decode scratch's block, and the block size and the per-stream count are
checked before the write. All throwing setup before apply (the per-stream
trial index, `begin_trial()` and the cell snapshots) happens before the state
enters the recovery stack and before any mutation, so a setup throw cannot
leave a half-filled state for the catch or the restore. A throw from
`apply_ubatch()` after a mutation is recovered by the journal replay. `apply_ubatch()`
never calls `pos_add()`/`pos_div()`, so `has_shift` cannot change during a
trial and the journal does not save or restore it. A journal belongs to one stream: the trial entries carry the stream tag and the search rejects a
foreign tag, so the same cell/position in another stream cannot supply a node.
A `seq_pos` node is keyed by (sequence, position), not by the cell: when two
cells share the key, the extracted node can live in another cell's entry. The
search checks the cell's global entry, then the flat trial entries newest
first, and only then all global entries for the shared key. When the trial's
first mutation creates the global entry, the trial entry is created lazily
from it, so later mutations' used/`seq_pos` nodes are owned too. The restore
compares positions as well as sequence bits: a purge followed by reuse moves a
cell, so the same sequence bit at a different position still moves its
`seq_pos` entry. A normal restore with no trial entry restores the real
snapshot state, used or empty; it no longer assumes an empty snapshot. The
snapshot copy keeps a non-zero `shift`, so there is no zero-shift
precondition. The per-cell copy carries pos/ext/seq/shift; the stream
`has_shift` flag is not part of the copy contract, and a prepare trial does
not change it. The journal is active for every prepare: the non-SWA path uses the same
reversible log, so the mutation extracts the live used/seq_pos nodes and the
replay re-inserts them without allocating; no ordered-tree nodes are cloned
per old pos/used. The old `set` stays for the restore_cells helper. The
non-trial record and every replay walk only the set sequence bits, not all
`LLAMA_MAX_SEQ` bits; the trial-record path still scans all `LLAMA_MAX_SEQ`
bits. The common one- or two-sequence node handles live inline
in the journal entry, so a touched cell does not allocate a per-cell vector;
wider masks spill to a heap vector. The replay passes the entry directly to
the restore, and each node-handle insert uses a `begin()`/`end()` hint. A
failed map insert means another cell already supplies the same (sequence,
position) key; the restore increments that count and drops the redundant
node, so the path stays allocation-free. The retired capacity test's five-pair
microbench for 1024 cells reported the old `cp_into`+`set` median about
0.135 ms and the journal median about 0.142 ms per iteration (ratio about
1.05, single observation). That is historical local evidence, not a stable
performance claim and not part of the long-term test suite; the journal
removes the clone allocations and the
remaining per-mutation record and replay are close to the old allocating
restore. `prepare()` runs once per ubatch in the decode loop for both SWA and
non-SWA, so the cost is per ubatch (amortized over the ubatch's tokens), not
per token, but it is on the decode path. The non-SWA journal uses the same
12 B/cell seen/trial arrays and the same per-trial entries as the SWA journal.
The fallback increments an existing map entry
through an iterator, never through `operator[]`, and `seq_pos_dec` checks its
entry with `GGML_ASSERT`; a missing entry aborts instead of dereferencing
`end()`. A node returned to the
containers by a normal restore can be re-extracted and re-owned by a later
prepare's journal; the replay inserts the extracted node handles, allocates
nothing, and a third prepare can own them again. `pos_set` inserts into `used`
before it writes the position, and every container that grows after a mutation
is reserved before the take, so a reserve failure leaves the nodes in the
journal where the replay finds them. A forward `seq_add` inserts its
`seq_pos` entry before the seq bit, so a failed insert leaves the cell
unchanged. Prepare exceptions are recovered by the journal; K-shift async
failure still uses the explicit `shift_failed` fail-stop path, and
`clear(true)` recovers it. `clear()` is synchronous, never throws, and does not
block on a retained context object; the caller owns the ordering. With SWA,
`prepare()` runs on the decode path, so the journal marks, the flat
`seen_` is one 8 B word per cell (32-bit generation + 32-bit entry index) and
`trial_seen_` is 4 B per cell; together 12 B/cell, 3 MiB per stream for 262144
cells. The entries, the flat trial entries and the ranges hold the distinct
nodes. The source search is O(1) for the cell's own entry and the newest trial
entries; the worst case is a linear scan over the flat trial entries. The `bytes()` ledger covers the journal marks, the global entries, the flat
trial entries, the ranges and the discard; the recovery snapshots
(`states`/`v_cells`), the non-SWA snapshot nodes and the flat trial index
block are separate prepare storage and are not included. The retired capacity
test's 2.2 MiB/1,105,920-byte figures predate the 32-bit index widening and
were not rerun; they remain historical observations. It printed
single-observation record/replay/trial/scan microbenchmarks; those remain
historical observations, not stable performance claims, and the long-term
suite no longer contains that microbenchmark.

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
`TQRelay`, `TQSampling`, `TQScheduling`, and `TQText` packages:

```sh
cmake -S fork/tp5-compose -B build-tp5-c \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/packages \
  -DCMAKE_INSTALL_PREFIX=/absolute/path/to/packages
cmake --build build-tp5-c --parallel 4
ctest --test-dir build-tp5-c --output-on-failure --no-tests=error
cmake --install build-tp5-c
```

Do not add source-tree include paths (`ggml/include` or `src`) to this build.
The installed packages provide every public header; a source-tree path hides
private-header dependencies and breaks the installed-consumer contract.

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

## MTP batch pool contract (2026-11-09)

`TP5_MTP_BATCH_POOL=1` enables the worker-local pool in `src/host_pool.h`.
Each worker owns four lazy slots; a slot allocates its one-token `llama_batch`
on first acquire and keeps the token inline. A lease returns its slot when it
leaves scope, including on exception unwinding. When the flag is off or every
slot is leased, the fallback lease owns the same heap batch that the pre-pool
path allocated and frees it on release. `llama_batch_init` checks every
allocation, including the outer `seq_id` array, and reports failure with null
fields instead of dereferencing a failed allocation. `llama_batch_free` is
safe on a partial batch because it checks each pointer and stops the `seq_id`
walk at the first null entry. Every caller in this tree checks the result with
`llama_batch_is_complete` and frees the partial batch; functions with an error
channel return or propagate it, and the void entry points with no error channel
(`llama_context::opt_epoch` and perplexity's void scorers) fail-stop with
`GGML_ABORT`, recorded at the call site. The MTP pool and the target
`row_batch` throw `std::bad_alloc`; a failed pool slot stays unready, so a
later acquire retries.

`row_batch` also keeps its own capacity. `llama_batch` does not store the
extent it was initialised with, so `row_batch::push` rejects a row that would
write past `token[]`, `pos[]`, `n_seq_id[]`, `seq_id[]` or `logits[]` with
`std::logic_error`. Every current caller stays within its allocation; the
guard exists so a later miscounted row cannot become a silent heap overwrite.

The pool is not thread safe: exactly one worker owns it, and the pool outlives
every lease. `lease::set` requires a non-null embedding row whose width equals
the pool width. An MTP decode reads the NextN hidden row, not the token, so a
missing or narrower row would decode stale bytes, and a wider row would
overrun the slot. The constructor refuses a non-positive width. The target and
draft NextN widths are checked once (`10240`) before the worker pool exists.

`TP5_ALLOC_PROBE=1` records lease and release values up to 4096 events per
pool; the log stops at the cap. With the pool enabled, 10k sequential leases
perform no heap allocation after the first slot is warm. The CPU test
`tq-tp5-host-pool` covers reuse, fallback, exhaustion, exception release, the
probe cap, pool destruction, the `set` contract, and a failure injected at
each of the six real `llama_batch_init` allocation points (direct, pooled, and
fallback), plus the fallback token. Injection uses the test-only `LD_PRELOAD`
interposer in `tests/fail-malloc.c`, which arms a countdown immediately before
the real call and adds no production hook or ABI. Every failure must report
`std::bad_alloc` and the next acquire must succeed; without the preload the
normal path skips injection and still passes. The contract is documented in
`include/llama.h`.

`mtp_batch_pool::stats()` reports the same reuse without the probe log:
`pooled_leases`, `fallback_leases`, `slot_inits` and `exhausted`. A run that
keeps the pool enabled must show `fallback_leases == 0` and
`slot_inits <= slots`; a nonzero `exhausted` means the slot count is below the
worker's concurrent lease depth. The destructor asserts that no lease is live,
because a lease that outlives the pool would write into freed batch storage.

## MTP target state checkpoint (2026-11-09)

The non-native-trim MTP verify path (`GGML_TP5_MTP_RS_RING=0`,
`GGML_TP5_MTP_ON_DEVICE_CHECKPOINT=0`) keeps one host checkpoint per worker
across verify epochs. The checkpoint is an `mtp_state_checkpoint`
(`src/mtp_state_checkpoint.h`): it owns the storage, the current snapshot size,
and an explicit valid flag. A capture grows the storage only when the new
snapshot is larger; a same-size or smaller capture reuses the existing bytes
without zero-filling them. `llama_state_seq_get_data_ext` must write exactly
the requested size, otherwise the checkpoint stays invalid even though older
bytes remain in the storage. A restore passes the captured size, not the
storage size, and consumes the checkpoint; full acceptance invalidates it. A
failed, short or throwing restore also leaves it invalid, so an older snapshot
is never restored as the new one. The native-trim path
(`GGML_TP5_MTP_RS_RING=1`) does not create a host checkpoint. The CPU test
`tq-tp5-mtp-state-checkpoint` covers same-size retention, growth, shrink,
short capture, short restore, full acceptance, exception unwinding, zero size,
and the shrink-after-invalidate and shrink-after-exception epoch sequences.

## MTP target greedy cache (2026-11-09)

The MTP verify epoch caches each target logits row's
`tq::sampling::select_finite_greedy` result for the duration of that epoch
(`src/mtp_greedy_cache.h`). The cache is local: a later target decode can
overwrite the logits buffer, so it is never reused across epochs.

- Rows below the accepted prefix use the verify id. The replay branch refreshes
  the cached entry with the replay result; the id is required to equal the
  verify id, and the stored logit bits become the replay bits. The commit uses
  only the id.
- The replay mismatch row (row `accepted`) is refreshed with the replay scan.
  Its argmax may change: the replay gate only requires the replay and original
  distributions to be close, not the top-1, so the commit must use the replay
  result, not the verify id.
- The native-trim mismatch row has no replay and uses the verify result.
- The full-accept new row `n` is never verified and is always scanned.
- The LateBind prefetch `g0` seeds row 0. It is usable for an accepted row or
  the native-trim mismatch; an `accepted == 0` replay refreshes row 0 before
  the commit.
- The cache stores the exact `greedy_result` (usable, id, logit bits). The
  finite/unique-maximum/signed-zero tie rules of `select_finite_greedy` and
  every `require(usable)` check are unchanged. Cached rows do not perform the
  greedy scan and do not read the logits buffer unless `capture_logits` is
  enabled; the capture path still reads the row through the same getter and
  output-borrow proof. `no_sync_logits_calls` counts rows resolved through the
  no-sync path under the output barrier; a cached greedy commit row without
  `capture_logits` does not read the logits and is not counted, while a cached
  row with `capture_logits` on still reads through the capture path and counts.
  The field can be 0 when all commit rows are cached and `capture_logits` is
  off. It is not a barrier proof. The `real_h` NextN read, the `verifier_rows`
  copy and the replay distribution gate are unchanged.

The CPU unit test `tq-tp5-mtp-greedy-cache` (`tests/test-mtp-greedy-cache.cpp`)
covers the deterministic cache boundary/semantics: partial mismatch with a
changed replay argmax and refreshed bits, the verifier copy not aliasing the
refreshed entry, the accepted-prefix id staying the verify id while the bits
become the replay result, the full-accept new-row guard (including a direct
cache of row n), the LateBind g0 rules, signed-zero/tie and NaN results, and
row bounds. It does not compile or execute runtime.cpp and does not prove the
runtime wiring, the runtime cancellation/exception path,
`no_sync_logits_calls`, `capture_logits`, or the real end-to-end. Those remain
unverified here (no GPU/full).

The unit target has 32 checks; the reviewed unit run is 32 PASS / 0 FAIL. The
real MTP runtime and GPU path were not run.

## LateBind verdict reuse and full acceptance (2026-11-09)

The LateBind prefetch (`TP5_MTP_LATEBIND_DRAFT1=1`) assumes verdict `a=0`.
Verdict reuse is on by default. `GGML_TP5_LATEBIND` unset and
`TP5_MTP_LATEBIND_REUSE_ACCEPT` unset means reuse on. `GGML_TP5_LATEBIND=0`
turns reuse off and `GGML_TP5_LATEBIND=1` turns it on. The legacy
`TP5_MTP_LATEBIND_REUSE_ACCEPT` is an override only while
`GGML_TP5_LATEBIND` is unset: `1` is on and `0` is off. A legacy value other
than `1` under a set master gate fails loudly. `GGML_TP5_LATEBIND=0` does not
disable the prefetch; `TP5_MTP_LATEBIND_DRAFT1` does. The `a>0` bind requires
the native-trim path (`GGML_TP5_MTP_RS_RING=1`). The `accepted==0` base bind
does not require native trim.

An `a>0` epoch binds the prefetched draft step 1 only when the commit token
equals the prefetched draft token and the exact-input guard proves that the
serial row-0 decode would consume bitwise-identical inputs: the same
position, local row, a prefetch that exported outputs, and a bitwise-equal
`real_h` NextN residual. The prefetch state is per epoch, so the guard cannot
cross an epoch. The guard is the only additional authority for reuse; the
accepted count is not a second gate. The full-checkpoint path
(`GGML_TP5_MTP_RS_RING=0`) keeps the serial decode: the guard refuses reuse,
the replay path discards the prefetch, and row 0 decodes serially. This
section is therefore about native-trim reuse, not every LateBind path.

Full acceptance (`accepted == n`) is a legal bind on that path. It leaves
row 0's serial inputs unchanged. A stop on the last row `n` is classified in
the commit loop, after row 0's bind, and keeps the same full-accept verdict.
The stop cleanup trims the draft attention KV from the stopped request's
source token, including a stop-row-0 prefetch that the bind site reused.
Partial acceptance and a stop before the last row latch `full_accept=false`;
the same guard decides their bind.

The earlier code aborted when the guard passed on a full-accept native-ring
epoch (`require(!lb_reuse_full_accept)`) unless the opt-in
`GGML_TP5_LB_SELFCHECK_REUSE` was set. That was a false invariant: full
acceptance is the normal case when the drafts are right, and the abort fired
before the row-n stop was classified. The opt-in branch also checked
`lb_reuse_verdict_n`, `actual_accepted` and `lb_reuse_full_accept` in one
three-term `require`. Those terms come from the same verify epoch
(`verdict_n == n`, `full_accept == (accepted == n)`), so the check is a
same-epoch consistency echo, not an independent safety gate. The guard is the
authority. The old selfcheck env var is no longer read.

The bind carries an explicit completion proof before the next row can reuse
the draft context. When row 0 exports outputs (`draft_output=true`), the
synchronizing `llama_get_logits_ith` / `llama_get_embeddings_nextn_ith`
getters prove completion on that row. When `TP5_MTP_LAST_DRAFT_LOGITS=1` and
row 0 is not the last row (`draft_output=false`), the getter does not run; the
bind calls `llama_synchronize(dctx)` and clears `draft_pending` before the
next decode. The draft context is single-device with no relay, so this host
synchronize is the completion proof. This removes the earlier "next decode
submitted with no explicit synchronize" window.

Do not read the `draft_output` flag as "row 0's outputs are never read":
`draft_output = !last_draft_logits || last_target_row`. With
`TP5_MTP_LAST_DRAFT_LOGITS` unset (default false), row 0 has
`draft_output=true`, so its prefetched outputs are read by the next getter.
Only with `TP5_MTP_LAST_DRAFT_LOGITS=1` and `accepted>=1` does row 0 have
`draft_output=false`.

No runtime/GPU run was performed for this tree. The full-accept bind, the
exact-input guard, the completion proof, and the row-n stop cleanup are
supported by code review only. `llama_synchronize` is a host synchronization
of the single-device draft context; it is not a native relay receipt. The
physical KV deletion after the stop trim and the GPU path remain unproven. No
new CPU test was added; this tree has no CPU runtime path that reaches the
LateBind bind.

## Legacy relay channel map reclamation (2026-11-09)

When neither `GGML_TP5_PLAN_UNION=1` nor
`GGML_TP5_RELAY_SLOT_CHANNELS=1` selects the slotized lane, a window owns one
`channel_state` per `(plan generation, channel)` in `src/relay_channel_map.h`.
A provider plan generation can advance when a graph is rebuilt and its terminal
binding changes (the MTP 1-token / n+1-token alternation is one trigger). The
map previously never erased an entry, so every generation added its channels
for the provider lifetime.

Reclamation runs only in the legacy map branch, before the next generation's
first `INPUT_READY` is inserted. `prepare()` already required the provider to
be idle (finished, no borrow, every channel released). The same courier
services older `OUTPUT_RELEASED` events before the next `INPUT_READY`, and a
deferred `delayed_event` is handled before the next poll. `OUTPUT_RELEASED`
still finds its entry and never triggers reclamation; failure or abort leaves
the map to `finish_ledgers()`.

Reclamation is state-driven and preserves availability. When a newer
`INPUT_READY` arrives, `legacy_channel_map` advances its admitted generation;
an older entry is retired only on a ledger receipt: `channel_ledger::retirement()`
returns `bank_released` when every occupied bank recorded input release, output
publication and every consumer completion, or `no_reservations` when no bank was
ever reserved. `not_ready` is retained, as is any entry with a held lease. The
reclaim callback does not call `mark_drained` and does not turn a bool or
`held == false` into a native-drain proof; the callback only records the
receipt (`map_retired_bank` or `map_retired_no_reservations`). A held or not-ready old
entry stays for its later `OUTPUT_RELEASED`, and a later generation retries.
The provider idle requirement at prepare remains the ordering proof; the
native-drain proof for entries that remain at close is still `finish_ledgers`.
A stale `INPUT_READY` older than the admitted generation is rejected
explicitly and creates no old ledger.

The existing `TP5_DW_COURIER_HIT` line prints `map_retired_bank` and
`map_retired_no_reservations`, the two receipt classes retired through this path.

The CPU-only `tq-tp5-relay-channel-map` test drives the same
`INPUT_READY -> held lease -> OUTPUT_RELEASED -> newer INPUT_READY` chain
through `legacy_channel_map`, with real `channel_ledger` leases (reserve, five
producer-ready records, reduce done, output published, five consumer-done
records). It observes map size/generation, held retention, the retired
`bank_state` drain facts (`input_released`, `output_published`,
`consumed_ranks`), the `bank_released` / `no_reservations` proofs, an
incomplete bank that is retained as `not_ready`, the explicit stale refusal,
and that `OUTPUT_RELEASED` never reclaims. It is not a pure helper-counter
test, but it is still a narrow host test: no GPU, full-model, or MTP runtime
evidence is claimed.

## MTP multi-row stop contract (2026-11-09)

One MTP verification batch commits `1 + N` rows (`N` in 1..3): the source
token plus the accepted draft prefix and its correction row. A stop can occur
before the last row: an EOG token, the request's cancel threshold, or the
`predict` budget. The committed prefix is then shorter than the speculative
batch.

The runtime does not merely suppress the later text pushes. During
verification it classifies the emitted token of every row that could be
committed and records the first stop row. The target rollback path then keeps
only the committed prefix:

- `mtp_native_trim` (the bounded GDN ring) rewinds the target
  attention/PLE/indexer/recurrent suffix from `source_position + valid_rows`.
- The full checkpoint path restores the pre-batch target state and replays only
  the `valid_rows` committed inputs, so the target state never retains the
  speculative suffix. The replay preserves the original numerical contract:
  accepted draft rows must match exactly; the correction row may differ only
  inside the KL/rel/overlap gate; a stop row must match because its token
  decides the stop. A full accept does not replay the extra row.
- The commit loop advances `consumed` by the committed rows only, emits only
  those tokens, and finishes the UTF-8 stream exactly once. The post-loop
  `apply_commit_trims` remains a safety net for a commit-time stop that the
  verification row classification did not predict.

The draft context is speculative. The MTP draft context is single-sequence:
`mtp_batch_pool::lease::set` always writes `seq_id` 0, and the option
validation requires two windows with one prompt per window. The stop cleanup
therefore enforces `ids.size() == 1 && local == 0`; this restates the existing
T4 smoke contract and adds no new restriction on legal inputs. A future
multi-request MTP path must add per-request draft sequences before removing
that guard. The trim start is the stopped request's first row position
(`stopped_source_position`), not necessarily `batch.pos[0]`. On any MTP stop
the runtime synchronizes a pending draft decode and trims the predicted
attention KV from that source token, including a stop-row-0 prefetch that the
bind site reused. The service-round boundary remains the final clear. It is
not part of the request's committed target state.

The CPU-only `tq-tp5-commit-stop` test exercises the production-used
`classify_commit_stop`, `commit_stop_state`, `has_later_row`,
`stopped_source_position`, `replay_token_must_match`, and
`apply_commit_trims` helpers, plus the real `generation_stream`. It covers
cancel/EOG/budget precedence, `ignore_eos`, the target trim position with and
without a later row, state reset, the replay contract, trim application, the
stopped-request source position on a multi-request row map, and the
finished-stream push rejection. It does not execute `run()`, a model,
checkpoint restore/replay, `llama_memory_seq_rm`, teacher forcing, or a draft
context; those remain covered only by the real-model acceptance suite when it
is run. The stop-row-0 draft KV cleanup is code-inspected here; this CPU test
cannot create a draft context.
