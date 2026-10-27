<!-- two-requests-current-status-20261021 -->
## 2026-10-21：双请求保留版本与最终测量；每路 47 tok/s 目标未达到

保留版本、证据和入口见[双请求报告](../build-tp5-two-requests-20261021/TWO-REQUESTS-REPORT.md)；以下旧正文保留。本次增量为 3 个 A 源文件和 1 个 C 源文件，公共 API 不变；使用原始模型权重、每请求 262144 容量的 Q4_0 K/V，并保留 Swift 共享权重派生 view 生命周期修复及五 rank host 提交顺序 gate。

同 prompt 双请求组（n=2 个独立进程），每路 512 token：两路几何均值为 **37.629332 / 37.624558 tok/s**，aggregate 几何均值 **74.923025 tok/s**，范围 **74.847236–74.998891**。相同 C 版本的单请求观测（n=1）为 **46.071845 tok/s**；aggregate 对单请求的 **1.626222×** 仅作描述性比较。不同 prompt 的一个双请求案例为 **34.571716 / 34.899258 tok/s**，aggregate **69.062687 tok/s**。**两个请求各 47 tok/s（合计 94 tok/s）的目标未达到。** 历史 **47.34 tok/s** 只属于旧 F16 单请求配置。

完整 16-token trace 已通过，49 组五 rank 提交保持一致次序，但该完整 tracing 的共同生成期 GPU overlap 为 **0**。另一次相同保留版本的 kernel-only 采样已通过：共同输出阶段 GPU0–3 观察到跨请求算术/算术与算术/consume 的真实交集，GPU4 为零；含预填充的完整活动阶段五卡均有局部交集。由此可确认存在实际 GPU 重叠，但不能宣称五卡共同生成期都已重叠或通信完全隐藏。各次观测和证据分别保留，详见双请求报告。

---

# Exact upstream exceptions

<!-- TP5_NATIVE_SPEED_20261021_FINAL -->
## Latest retained result: native TP5, 2026-10-21

The current retained native TP5 configuration is the 2026-10-21 four-pair-qualified build. Older “current”, “final”, and gap-failure entries below are historical. The user's relaxed numerical policy applies to native acceptance; explicit strict-reference modes retain their separate gates.

The overlay still contains **51 existing paths**, with no added path. Ten A-source hashes and their retention/removal rationale were refreshed in `upstream-exceptions.json`. Relative to saved entry snapshots, those ten A files total +156/−132 lines; C runtime totals +4/−3. The production patch includes the prior native-policy change. It is not the full fork-versus-upstream diff.

Public C signatures and public struct layouts are unchanged. Private padding configuration/helper declarations, the `tp5-relay-native` backend parameter, and weak C++ async/future symbols are intentional changes. The BF16 fast path also admits ordinary non-strict RDNA2 use when its guards match.

See the [final report](../build-tp5-native-speed-20261021/NATIVE-SPEED-REPORT.md), [inventory](../build-tp5-native-speed-20261021/current-change-inventory.json), and [patch index](../build-tp5-native-speed-20261021/patch-index.json).

<!-- /TP5_NATIVE_SPEED_20261021_FINAL -->

A is the commit recorded in `upstream.json`, plus the explicitly reviewed files in `upstream-exceptions.json`. B modules stay independent of A and their sibling B modules; C uses installed public interfaces. An A exception is coupled work whose continued presence must be justified. The manifest does not turn that work into an independent B module.

The current manifest contains **48 reviewed A paths in total**, including the two new Q8/F32 kernel files; do not count those files twice. The saved `final-attention-boundary` receipt exits 0; later concurrent changes to the root status note correctly caused the HIP pipeline's final guard to reject its old digest. Final source identity and guard results are indexed separately from successful build stages. The current seven-B/seven-C topology includes the separately installed [TP5 consumer](tp5-compose/README.md). Runtime qualification and exact candidate identities are recorded in the [TP5 index](audit/tp5-acceptance-20261019.json) and [validation record](VALIDATION.md); older counts and results below retain their original candidate identities.

The manifest has exactly this schema; placeholder hashes are not accepted:

```json
{
  "version": 1,
  "upstream_commit": "full upstream commit ID matching upstream.json",
  "files": [{
    "path": "include/llama.h",
    "sha256": "64 lowercase hexadecimal digits of the reviewed file bytes",
    "mode": "100644",
    "consumers": ["fork/batch-compose"],
    "reason": "Describe the public behavior required by the named consumer.",
    "removal_condition": "State the upstream API or migration that removes this exception."
  }]
}
```

Use `100755` only for executable regular files. Paths are repository relative and canonical. Deletions, symlink overlays, duplicate paths, unused entries, and A files with `skip-worktree` or `assume-unchanged` are rejected. The visible changed A path set must equal the manifest set; each live file and any changed staged version must match its digest and mode. Fork files remain outside this A gate. Ignored build products are not source overlays; the manifest must not be used as an automatic enrollment mechanism for whatever happens to be in the worktree.

After a deliberate A change, review the complete changed file, its named consumers, and the removal condition, then update its digest. Neither the build script nor isolation runner regenerates or accepts a hash for you. Run `python3 fork/check-upstream.py` before committing. The Git repository itself and its configuration are trusted inputs; this guard is a development boundary gate, not a sandbox for hostile build scripts.

## Claims and executable evidence

- **AXIOM:** the full upstream commit and the manifest entries have been reviewed. A digest proves identity with those recorded bytes, not functional correctness.
- **LEMMA:** the gate rejects undeclared changes and mismatched file content or modes, including an unreviewed staged version hidden by a reviewed worktree. `tests/test-upstream-boundary.py` exercises real temporary Git repositories.
- **SPEC:** default `tests/isolation.py` archives the pinned upstream commit, applies only the bytes captured by successful verification, builds A and its public seam tests, then builds seven B and seven C projects independently using installed package exports. It removes its own intermediate source/build copies and original install prefixes before compiling relocated consumers and running installed programs; it does not remove the user's repository. The relocated TQRerot consumer exercises a two-worker publication through the installed public header and library, and the installed RERoT C runs help / CLI checks. TP5's installed public consumer and help checks do not execute the five-device model. `--model /absolute/model.gguf` enables the existing batch/phase, stream/stop, and teacher-forced logits checks. The independent `--rerot-model /absolute/instruct.gguf` option enables actual shared-whiteboard model checks after relocation; absence is recorded as a skip, not a pass.

The logits check also consumes the installed batch program to check the same submitted prompt tokens, final maximum logit, and selected token within 1e-5. This independent check does not obtain a second full-vocabulary row; repeat tapes and scalar prefix/suffix controls compare all vocabulary values. The stream/stop check uses actual model output and compares matched submission configurations. Neither check establishes HTTP transport behavior or equality between arbitrary batch shapes.

The RERoT model harness checks actual frontier events, frozen peer prefixes, immediate per-token publication, stage completion, and output ownership. It retains the command, binary/model hashes, raw output and diagnostics, timing, and validation receipt. Template, microbatch sizes, phase-release modes, and native device placement are explicit options. Multi-device qualification additionally requires the real configuration and device evidence. Event checks do not replace the independent attention/logits oracle or establish a kernel hit, TP5 execution, or performance benefit.

The new calibration parser requires only the C++17 standard library. Its imatrix C consumer links installed `ggml::ggml-base` and TQCalibration, and uses the installed `GGML_INCLUDE_DIR` exported by the pinned package. It has no private-header or sibling-C dependency, no llama model, and no inference context. A's package configuration still discovers its enabled backends; this dependency choice does not make an incomplete A installation supported. Separate unchanged-writer and actual quantizer-loader checks qualify format interoperability. A loader accepting a fixture is not model quantization or quality qualification.

## Current exception responsibilities

The exact path set, complete file digests, consumers, and removal conditions are in [upstream-exceptions.json](upstream-exceptions.json). Read final code totals and run status in [VALIDATION.md](VALIDATION.md#final-release-gates); old release totals do not describe later working changes.

The workflow under `.github/workflows/` is fork build infrastructure at the path GitHub requires. Its manifest entry does not add an inference-engine dependency. The Meta async offset correction adds no API and has independent failing get/set regressions. The earlier recurrent clear/reuse qualification uses A without importing the old eager-clear path; the later independent recurrent capacity work is a separate exception and needs separate qualification.

### TP5 bounded whole-epoch native replay (current)

The single-stream repair keeps public relay status/event ABI 1 and its layout
unchanged. Only the private provider interface moves to version 2: Meta supplies
the complete prepared stage-major graph array; the HIP owner warms, captures
and replays it with two exact-signature entries. All five executables exist
before any replay launch. Actual accepted/unknown masks, submission sealing,
CPU borrows and native completion remain separate facts. The activation word
changes only after prior native/CPU readers retire. An unterminated capture
cannot satisfy `finished`, even when no graph launch was accepted.

Two single-workgroup relay kernels replace five per boundary. Every producer
writer fences and meets before publication; each consumer acquires the exact
generation and all readers finish before acknowledgement. Four-value copies
preserve scalar tails and four-byte terminal alignment. Hot poll/publish avoids
native tail queries; snapshot/close still proves native completion. The Q8/F32
projection uses a barrier-free four-output block without changing weights or
requantizing activations. No B dependency or new public replay API is added.

Current complete numerical/lifecycle matrices, KV-transition recapture, late
abort, host sanitizer, paired performance and relocated installation evidence
are indexed by `fork/audit/tp5-performance-20261019.json`. The fresh package's
executable and native libraries are byte-identical to the confirmation image.
The following records retain the preceding pre-replay image's identities.

### TP5 native relay and precise F32 attention (pre-replay history)

The `fork/tp5-compose` consumer needs A to own real five-rank model execution, native payload storage, complete pre-submission, exact-epoch publication, actual last-reader acknowledgement, and partial-submit/native-drain facts. The public opt-in `tp5_relay` context and relay backend interfaces expose those facts without importing A's private headers into C. C owns ordered CPU F32 reduction and composes the independent TQRelay, TQScheduling, and TQText packages; A does not call a B service or own C's scheduling policy. Shared model metadata must remain stable while each mutable context owns its input, KV, graph arena, and native execution state. Unknown completion retains native owners rather than freeing in-flight storage.

The declared `q8_0-decoded-weights-f32-activations-ordered-f32` profile adds a qualified Q8/F32 projection path without requantizing activations to Q8_1. Its Flash Attention path reuses the existing fused online F32 attention implementation and preserves F32 Q/K/V operands through model graph construction. It does not substitute separate QK/softmax/V operations or insert an F16 rounding boundary. Ordinary FA routing and precision selection are unchanged, but the shared online/banded F32 kernel now skips negative-infinity masked keys before loading K/V or computing QK, improving masked-nonfinite semantics while preserving finite unmasked accumulation order; do not claim that all other backend behavior is unchanged. `--flash-attn on|off` defaults to **off** in C because it is the measured faster qualified mode; explicit on retains full numerical/lifecycle qualification. Each initialized context's public resolved getter must match the request; there is no silent fallback.

This scope has real Qwen3-8B Q8_0 execution on five RX 6800/gfx1030 devices: the final `acceptance-fa-final` eight-token four-request full-logits, delayed-consumer, cancellation, and injected-abort contracts pass with unchanged preregistered `.002/.001` tolerances. The separate five-device arithmetic gates pass 170 Q8/F32 and **180 FA cases**, including always-masked K/V NaNs, under the unchanged `2e-5` bound (max error `2.24640083e-6`). The FA-on trace records 720 fused launches, 144 per physical device. These facts do not extend support to LFM2 MoE, other models/backends, long contexts, MTP, RERoT+TP5, or universal performance claims. Early FA precision/cast failures and all prior sanitizer and Vulkan FA/TOPK receipts remain intact.

Additional receipts passed the complete eight-token FA-off numerical/lifecycle matrix, while the FA-off trace contains zero FA launches. The FA-on 32-token main-only comparison passed 128 rows / 19,447,808 logits with maximum absolute difference `9.059906005859375e-5` and no numerical failures under the same threshold. Current-source host-only ASan completed four original 193-token prompts through two contexts with two generated tokens/request, FA on and exit 0 (`asan-long-final.json`); earlier four-token main plus abort sanitizer receipts retain their candidate identities. Leak detection is disabled: neither result establishes full-matrix sanitizer, GPU memory-safety, or leak qualification. Strict preflight now skips legitimate empty tensors during no-output prefill; the original 193-token workload remains, with no artificial logits. The HIP build completed all seven-B/seven-C build/install stages and eleven public A seam tests before the root-note digest rejection described above.

The default decision is evidence-based rather than a demand to enable FA: `per-stream-fa-final` measured twelve processes in three paired blocks with four fixed 193-token prompts and 32 teacher outputs/request. Per-stream geometric-mean token/s are off/one-window **8.9853**, off/two-window **15.4185**, on/one-window **7.7510**, on/two-window **13.4642**. The paired two-window on/off ratio is **0.87325**, 95% CI **[0.83966,0.90818]**: on is about **12.7% slower**, not promoted as the performance default. Off two/one-window ratio is **1.71596659**, 95% CI **[1.28598327,2.28971978]**. Both intervals use paired per-block log ratios, Student t df=2, with only three independent blocks. There are 31 ITLs/request, so p99 is not a stable population tail. TTFT is admission-to-first-commit, excluding model/context startup, queueing and network. The original `per-stream-fa` failure and slower `per-stream-fa-prefill-fixed` candidate remain intact; earlier provisional on-default decisions are not rewritten as measured successes.

The final HIP package separately passed selected relocation and installed-consumer contracts (`hip-package-proof/summary.json`). All fifteen components (A + seven B + seven C) were freshly installed to stage, then renamed to relocated. The original stage and old install prefix were genuinely absent during consumer configure/build and all GPU calls, then restored after proof. The user's source tree was not moved: the external consumer used only `find_package(TQTP5Compose)`, installed public headers and imported targets. Compile-path validation excludes old-prefix imports and private source headers; `ldd` resolves relocated libraries. Its two-request explicit-on/two-context execution used five native ranks and closed budgets, while direct default-off two-request, automatic one-window/off single-request, and one-window/on three-request CLI cases also passed. The actual installed helper `share/TQTP5Compose/real-tp5.py` passed relocated main-only/predict4 full-logits checks. The initial guessed-path failure remains recorded; correcting the install path did not repeat completed setup.

Final package traces bind the five physical BDFs to on=720 (144/device) and off=0 FA launches. `performance-image-binding.json` establishes identical native library images between the published product and twelve performance processes; only the CLI default selection changed, while every measured case specified on/off explicitly. A combined W0-delay/W1-abort scenario also passed with both completed masks=31, unknown=0, reservations=0, exit 1 and no success report. The exact held-borrow branch was not separately instrumented; this is selected-scenario evidence, not branch-coverage proof.

Remove these exceptions when pinned upstream provides equivalent prepared five-rank HIP execution, decoded-Q8/F32 arithmetic, fused F32 attention without operand rounding, independent mutable contexts, exact publication/last-reader/native-drain contracts, and the installed full-model numerical, lifecycle, real-dispatch, and relocation consumers pass without the overlays. Manifest digests prove reviewed-byte identity, not these runtime properties. The 85-command isolation PASS belongs to the saved 45-path pre-FA candidate, not a final 48-path archive-isolation rerun; the final HIP package has its own successful relocation/public-consumer receipts above. Current 48-path source-guard identity and the final machine index remain separate final receipts, not an inferred blanket all-gates PASS.

### Vulkan build-only header dependency

The DevOps acceptance retains one build-only Vulkan exception in
`ggml/src/ggml-vulkan/CMakeLists.txt`: the already required
`SPIRV-Headers::SPIRV-Headers` package supplies private include directories for
the backend. Finding its config in a non-system prefix did not previously make
`spirv/unified1/spirv.hpp` visible to the compiler. The include requirement does
not enter the installed runtime interface. Remove this exception when the pin
builds with that declared prefix, without `CPATH` or `CPLUS_INCLUDE_PATH`, and
the installed public consumers still pass. Exact bytes and the current path
count remain in the manifest; runtime evidence is separate in `VALIDATION.md`.

### RERoT reader and recurrent interfaces

The RERoT implementation requires A to do work that an installed C consumer cannot perform through ordinary sequence masks alone. A owns the physical KV cells, RoPE geometry, attention graph, and recurrent state. Its public reader view describes each query's logical sequence, storage position, and virtual position, together with the corresponding visible key identities and positions. A resolves those identities to its actual resident cells. Identity-phase spans reuse the native batched KQ scores; shifted spans reconstruct the query from its pre-RoPE source and original RoPE parameters. Visible scores are scattered into their physical KV slots, with hidden slots set to negative infinity. One global softmax per query/head then feeds the native batched V operation. This preserves the native cache layout, physical reduction order, and batch geometry for those operations; V is no longer gathered per reader. The helper uses existing GGML operations and A's scheduler, and neither calls TQRerot nor accepts a B-owned runtime service. Qualification remains tied to the specific built and installed A in each receipt.

The recurrent exception separates logical sequence tails from the `n_rs_slots` physical pool, retains native copy-on-write ownership, and preflights required physical capacity. A public recurrent-only retirement operation lets a sealed stage release its recurrent/convolution state while retaining its public attention keys. RERoT C requests `W + 2` logical sequences and `P + 1` recurrent slots for `W` workers and `P` active pens. The extra slot keeps the immutable base lineage. New stages copy that base through A's public interface; reusing a pen does not reuse the previous stage's sampler or recurrent history.

These interfaces have named consumers in `fork/rerot-compose` and public A seam tests. The B library remains C++17 standard-library-only. C supplies owned B views to A and creates a local `run_submissions(call, 1)` ledger for each native submission. The ledger owns that call's receipt until actual `llama_synchronize`, completion, and view cleanup; completed history does not accumulate across the episode. Unique call identities and the executor's failure state remain persistent. Cancellation is checked before another submission, after synchronized chunks/waves, and before B publication. On failure C drains the actual work and stops; it does not promise rollback of native writes or publish a partial B cohort. The context parameter structure has changed, so the installed header, A library, and consumers must be rebuilt together.

The RERoT work adds no GPU kernels. The manifest and final validation receipt remain authoritative for the exact changed paths and bytes; earlier line-count estimates do not describe later attention-helper revisions. Remove the corresponding exception when pinned upstream supplies equivalent resident reader views, reader phase handling, independent recurrent capacity, and retirement semantics that satisfy the public consumers. An upstream replacement must preserve the token frontier contract; moving A's private structures into B is not a removal strategy.

The independent B's Release, UBSan, and ASan receipts retain their original candidate identities. The current CPU and Vulkan installations passed their component checks. The hardening candidate passed both sixteen-case CPU and GPU attention oracles; the DevOps acceptance independently reran the GPU oracle. Legal single-layer GGUF fixtures with zero FFN weights compare F32 A execution against independent F64 forward/logits calculations: Llama NORMAL and Qwen2 NEOX, four Q heads / two KV heads, RoPE amplitude 1 / 1.375, forward/reverse row order, and serial execution of the same frozen cohort. Earlier eight-case results and the old GPU ordinary-calibration failure remain historical evidence, not the current candidate's status. There is still no A partial-rotary or GPTNeoX qualification, and B's mathematical tests cannot grant one. Exact candidate and artifact identities are in [VALIDATION.md](VALIDATION.md).

The release reader profile accepts resident F32 K and F32 V only; C requests those types and disables FA. F16 and mixed reader types are rejected by the public capability check. Ordinary A decoding keeps its existing F16 support. The DevOps LFM2.5 F32 reader runs passed all 33 strict groups and 12 explicitly required cross-shape comparisons, covering hidden/pending, permutation, restore, invalid input, and pool checks. Maximum equivalence and cross-shape logits differences were 0 on CPU and `1.52587890625e-5` on Vulkan, without changing the thresholds. The exact grouping and receipts are recorded in [VALIDATION.md](VALIDATION.md). These relational/state results do not replace the independent oracle or the separate complete-generation runs.

F16 reader experiments failed strict state gates: CPU reordering/restore reached `0.11799526`, and Vulkan row permutation reached `0.0003576`. The receipts remain evidence for the E23 precision-policy rewrite, including physical permutation and native single-sequence restore reallocation. Explaining a floating-point difference does not waive the gate. Multi-GPU PP has not run, and there is no RERoT speedup claim. Turbo/quantized KV readers, the RERoT+TP5 combination, LateBind, and MTP remain outside this reader implementation; the separately qualified Qwen3 TP5 consumer does not extend reader support. Final support and release decisions belong in [VALIDATION.md](VALIDATION.md), not in the manifest.

### Existing Vulkan cache corrections

Vulkan conversion caching has two distinct ownership obligations:

1. A private shared-scratch write invalidates the conversion identity stored in that scratch. Both large softmax partials and masked-attention bitmaps can overwrite it. A later scratch writer must also wait for prior scratch readers, including the no-split attention path's last bitmap read.
2. A tensor write that overlaps the cached conversion's original source storage invalidates that conversion, even if the source tensor pointer is unchanged. Exact aliases and every output selected by a fused-write mask must be considered. A memory barrier alone cannot refresh a cached copy.

These responsibilities belong to A's private cache and GPU recording logic. A public C graph consumer cannot repair them or own their state. Keep the A edits explicit, limited to those contracts, and remove them when pinned upstream provides equivalent behavior. The 2026-10-18 final rebuilt and reinstalled library passed all seven seam CTests, including the scratch and source-write fixtures. That receipt retains its original source identity; it does not qualify the later RERoT overlay. The exact source and manifest identities and remaining broad-suite status are tracked in the central [release gates](VALIDATION.md#final-release-gates).

The regression consumers are documented in [vulkan-scratch-reuse-protocol.md](tests/a-seams/vulkan-scratch-reuse-protocol.md) and [vulkan-activation-cache-protocol.md](tests/a-seams/vulkan-activation-cache-protocol.md). They call installed public GGML/Vulkan APIs and run actual device operations. They do not use a private bridge, scheduler CPU fallback, or replacement kernels. Fixture success does not prove all model graphs or GPU devices correct.

Current A already generates F32 and F16 Turbo SET_ROWS variants. Preserve that upstream support; do not register a new exception for the old fork's F16 rejection or attach unrelated logging. Its bounded packed-row regression tests existing behavior. The separate G14 device-specific pinned allocation/free/getter candidate remains unmigrated and unqualified.

## Source and runtime provenance

`--a-prefix` is a convenience for repeated package work. The receipt labels it `reused_unverified`: the reused A package's source commit, overlay content, and build provenance have not been established by that run. Its public seam and consumer tests still run. Only the default archive/build path proves the stated source construction. Separate source copies are not an operating system sandbox.

A new production A change requires a newly verified overlay and the relevant rebuilt, reinstalled runtime checks. An earlier successful receipt retains its original identity. The 2026-10-18 delivery completed its final archive isolation, targeted GPU checks, and the normal CPU and Vulkan tools builds. Its final full GPU suite passed 25932 of 25933 cases and still failed one FA case, with reported ERR 0.009055835 above its 0.0005 limit. TOPK did not reproduce in that final run, but its earlier full-suite failure remains unexplained and recorded. These are historical results, not a rerun of the RERoT batch. The source/manifest identities and command index are recorded separately from the Git commit history. Neither a matching manifest nor successful targeted cases mean every release gate is green. The actual decisions and remaining evidence belong in [VALIDATION.md](VALIDATION.md#final-release-gates), with immutable historical records retained.
