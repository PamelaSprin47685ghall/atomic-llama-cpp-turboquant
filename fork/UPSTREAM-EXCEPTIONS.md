# Exact upstream exceptions

A is the commit recorded in `upstream.json`, plus the explicitly reviewed files in `upstream-exceptions.json`. B modules stay independent of A and their sibling B modules; C uses installed public interfaces. An A exception is coupled work whose continued presence must be justified. The manifest does not turn that work into an independent B module.

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
- **SPEC:** default `tests/isolation.py` archives the pinned upstream commit, applies only the bytes captured by successful verification, builds A and its public seam tests, then builds seven B and six C projects independently using installed package exports. It removes its own intermediate source/build copies and original install prefixes before compiling relocated consumers and running installed programs; it does not remove the user's repository. The relocated TQRerot consumer exercises a two-worker publication through the installed public header and library, and the installed RERoT C runs help / CLI checks. `--model /absolute/model.gguf` enables the existing batch/phase, stream/stop, and teacher-forced logits checks. The independent `--rerot-model /absolute/instruct.gguf` option enables actual shared-whiteboard model checks after relocation; absence is recorded as a skip, not a pass.

The logits check also consumes the installed batch program to check the same submitted prompt tokens, final maximum logit, and selected token within 1e-5. This independent check does not obtain a second full-vocabulary row; repeat tapes and scalar prefix/suffix controls compare all vocabulary values. The stream/stop check uses actual model output and compares matched submission configurations. Neither check establishes HTTP transport behavior or equality between arbitrary batch shapes.

The RERoT model harness checks actual frontier events, frozen peer prefixes, immediate per-token publication, stage completion, and output ownership. It retains the command, binary/model hashes, raw output and diagnostics, timing, and validation receipt. Template, microbatch sizes, phase-release modes, and native device placement are explicit options. Multi-device qualification additionally requires the real configuration and device evidence. Event checks do not replace the independent attention/logits oracle or establish a kernel hit, TP5 execution, or performance benefit.

The new calibration parser requires only the C++17 standard library. Its imatrix C consumer links installed `ggml::ggml-base` and TQCalibration, and uses the installed `GGML_INCLUDE_DIR` exported by the pinned package. It has no private-header or sibling-C dependency, no llama model, and no inference context. A's package configuration still discovers its enabled backends; this dependency choice does not make an incomplete A installation supported. Separate unchanged-writer and actual quantizer-loader checks qualify format interoperability. A loader accepting a fixture is not model quantization or quality qualification.

## Current exception responsibilities

The exact path set, complete file digests, consumers, and removal conditions are in [upstream-exceptions.json](upstream-exceptions.json). Read final code totals and run status in [VALIDATION.md](VALIDATION.md#final-release-gates); old release totals do not describe later working changes.

The workflow under `.github/workflows/` is fork build infrastructure at the path GitHub requires. Its manifest entry does not add an inference-engine dependency. The Meta async offset correction adds no API and has independent failing get/set regressions. The earlier recurrent clear/reuse qualification uses A without importing the old eager-clear path; the later independent recurrent capacity work is a separate exception and needs separate qualification.

### RERoT reader and recurrent interfaces

The RERoT implementation requires A to do work that an installed C consumer cannot perform through ordinary sequence masks alone. A owns the physical KV cells, RoPE geometry, attention graph, and recurrent state. Its public reader view describes each query's logical sequence, storage position, and virtual position, together with the corresponding visible key identities and positions. A resolves those identities to its actual resident cells. Identity-phase spans reuse the native batched KQ scores; shifted spans reconstruct the query from its pre-RoPE source and original RoPE parameters. Visible scores are scattered into their physical KV slots, with hidden slots set to negative infinity. One global softmax per query/head then feeds the native batched V operation. This preserves the native cache layout, physical reduction order, and batch geometry for those operations; V is no longer gathered per reader. The helper uses existing GGML operations and A's scheduler, and neither calls TQRerot nor accepts a B-owned runtime service. Qualification remains tied to the specific built and installed A in each receipt.

The recurrent exception separates logical sequence tails from the `n_rs_slots` physical pool, retains native copy-on-write ownership, and preflights required physical capacity. A public recurrent-only retirement operation lets a sealed stage release its recurrent/convolution state while retaining its public attention keys. RERoT C requests `W + 2` logical sequences and `P + 1` recurrent slots for `W` workers and `P` active pens. The extra slot keeps the immutable base lineage. New stages copy that base through A's public interface; reusing a pen does not reuse the previous stage's sampler or recurrent history.

These interfaces have named consumers in `fork/rerot-compose` and public A seam tests. The B library remains C++17 standard-library-only. C supplies owned B views to A and creates a local `run_submissions(call, 1)` ledger for each native submission. The ledger owns that call's receipt until actual `llama_synchronize`, completion, and view cleanup; completed history does not accumulate across the episode. Unique call identities and the executor's failure state remain persistent. Cancellation is checked before another submission, after synchronized chunks/waves, and before B publication. On failure C drains the actual work and stops; it does not promise rollback of native writes or publish a partial B cohort. The context parameter structure has changed, so the installed header, A library, and consumers must be rebuilt together.

The RERoT work adds no GPU kernels. The manifest and final validation receipt remain authoritative for the exact changed paths and bytes; earlier line-count estimates do not describe later attention-helper revisions. Remove the corresponding exception when pinned upstream supplies equivalent resident reader views, reader phase handling, independent recurrent capacity, and retirement semantics that satisfy the public consumers. An upstream replacement must preserve the token frontier contract; moving A's private structures into B is not a removal strategy.

The independent B has passed all four CTest groups in Release, UBSan, and ASan builds. The latest physical-score/native-KQ/native-V helper has passed the sixteen-case CPU oracle. Legal single-layer GGUF fixtures with zero FFN weights compare F32 A execution against independent F64 forward/logits calculations: Llama NORMAL and Qwen2 NEOX, four Q heads / two KV heads, RoPE amplitude 1 / 1.375, forward/reverse row order, and serial execution of the same frozen cohort. The earlier eight-case result with maximum absolute logits error `7.72386e-7` retains its original helper identity. The GPU oracle still fails its ordinary-calibration gate; there is no independent GPU mathematical qualification. There is also no A partial-rotary or GPTNeoX qualification, and B's mathematical tests cannot grant one.

The release reader profile accepts resident F32 K and F32 V only; C requests those types and disables FA. F16 and mixed reader types are rejected by the public capability check. Ordinary A decoding keeps its existing F16 support. The final LFM2.5 F32 reader state tests passed all 33 strict groups: 31 full-logits equivalence groups and two peer causal-effect groups, covering hidden/pending, permutation, restore, invalid input, and pool checks. Including the added same-shape reversed-serial and visible-serial comparisons, the equivalence groups recorded maximum absolute logits differences of 0 on CPU and `0.0000133514404296875` on Vulkan. The final grouping and receipts are recorded in [VALIDATION.md](VALIDATION.md). These relational/state results do not replace the independent GPU oracle or qualify complete C generation.

F16 reader experiments failed strict state gates: CPU reordering/restore reached `0.11799526`, and Vulkan row permutation reached `0.0003576`. The receipts remain evidence for the E23 precision-policy rewrite, including physical permutation and native single-sequence restore reallocation. Explaining a floating-point difference does not waive the gate. Multi-GPU PP has not run, and there is no RERoT speedup claim. Turbo/quantized KV readers, a whole-model TP5 provider, LateBind, and MTP remain outside this implementation. Final support and release decisions belong in [VALIDATION.md](VALIDATION.md), not in the manifest.

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
