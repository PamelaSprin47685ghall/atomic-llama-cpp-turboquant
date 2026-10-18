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
- **SPEC:** default `tests/isolation.py` archives the pinned upstream commit, applies only the bytes captured by successful verification, builds A and its public seam tests, then builds five B and five C projects independently using installed package exports. Intermediate source/build copies and original install prefixes are removed before relocated consumers run. Package/help checks prove build and loading boundaries; `--model /absolute/model.gguf` additionally runs the actual batch/phase, stream/stop, and teacher-forced logits checks after relocation.

The logits check also consumes the installed batch program to check the same submitted prompt tokens, final maximum logit, and selected token within 1e-5. This independent check does not obtain a second full-vocabulary row; repeat tapes and scalar prefix/suffix controls compare all vocabulary values. The stream/stop check uses actual model output and compares matched submission configurations. Neither check establishes HTTP transport behavior or equality between arbitrary batch shapes.

The new calibration parser requires only the C++17 standard library. Its imatrix C consumer links installed `ggml::ggml-base` and TQCalibration, and uses the installed `GGML_INCLUDE_DIR` exported by the pinned package. It has no private-header or sibling-C dependency, no llama model, and no inference context. A's package configuration still discovers its enabled backends; this dependency choice does not make an incomplete A installation supported. Separate unchanged-writer and actual quantizer-loader checks qualify format interoperability. A loader accepting a fixture is not model quantization or quality qualification.

## Current exception responsibilities

The exact path set, complete file digests, consumers, and removal conditions are in [upstream-exceptions.json](upstream-exceptions.json). Read final code totals and run status in [VALIDATION.md](VALIDATION.md#final-release-gates); old release totals do not describe later working changes.

The workflow under `.github/workflows/` is fork build infrastructure at the path GitHub requires. Its manifest entry does not add an inference-engine dependency. The Meta async offset correction adds no API and has independent failing get/set regressions. Recurrent lifecycle qualification reuses A and adds public tests without importing the old eager-clear path.

Vulkan conversion caching has two distinct ownership obligations:

1. A private shared-scratch write invalidates the conversion identity stored in that scratch. Both large softmax partials and masked-attention bitmaps can overwrite it. A later scratch writer must also wait for prior scratch readers, including the no-split attention path's last bitmap read.
2. A tensor write that overlaps the cached conversion's original source storage invalidates that conversion, even if the source tensor pointer is unchanged. Exact aliases and every output selected by a fused-write mask must be considered. A memory barrier alone cannot refresh a cached copy.

These responsibilities belong to A's private cache and GPU recording logic. A public C graph consumer cannot repair them or own their state. Keep the A edits explicit, limited to those contracts, and remove them when pinned upstream provides equivalent behavior. The final rebuilt and reinstalled library passed all seven seam CTests, including the scratch and source-write fixtures. The exact source and manifest identities and remaining broad-suite status are tracked in the central [release gates](VALIDATION.md#final-release-gates).

The regression consumers are documented in [vulkan-scratch-reuse-protocol.md](tests/a-seams/vulkan-scratch-reuse-protocol.md) and [vulkan-activation-cache-protocol.md](tests/a-seams/vulkan-activation-cache-protocol.md). They call installed public GGML/Vulkan APIs and run actual device operations. They do not use a private bridge, scheduler CPU fallback, or replacement kernels. Fixture success does not prove all model graphs or GPU devices correct.

Current A already generates F32 and F16 Turbo SET_ROWS variants. Preserve that upstream support; do not register a new exception for the old fork's F16 rejection or attach unrelated logging. Its bounded packed-row regression tests existing behavior. The separate G14 device-specific pinned allocation/free/getter candidate remains unmigrated and unqualified.

## Source and runtime provenance

`--a-prefix` is a convenience for repeated package work. The receipt labels it `reused_unverified`: the reused A package's source commit, overlay content, and build provenance have not been established by that run. Its public seam and consumer tests still run. Only the default archive/build path proves the stated source construction. Separate source copies are not an operating system sandbox.

A new production A change requires a newly verified overlay and the relevant rebuilt, reinstalled runtime checks. An earlier successful receipt retains its original identity. This batch completed final archive isolation, targeted GPU checks, and the normal CPU and Vulkan tools builds. The final full GPU suite passed 25932 of 25933 cases and still failed one FA case, with reported ERR 0.009055835 above its 0.0005 limit. TOPK did not reproduce in that final run, but its earlier full-suite failure remains unexplained and recorded. The source/manifest identities and command index are recorded separately from the Git commit history. Neither a matching manifest nor successful targeted cases mean every release gate is green. The actual decisions and remaining evidence belong in [VALIDATION.md](VALIDATION.md#final-release-gates), with immutable historical records retained.
