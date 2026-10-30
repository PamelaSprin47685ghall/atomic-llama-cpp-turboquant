# Vulkan activation source mutation

This qualification is separate from the shared-scratch overwrite correction. It concerns a cached conversion whose original source bytes change while the source tensor pointer remains the same. A memory barrier orders the source write; it does not refresh the backend's converted copy or its cache key.

## Public-API probe

Build `vulkan-activation-cache.cpp` as `test-vulkan-activation-cache` against the actual installed A public headers and libraries, linking `ggml::ggml` and `ggml::ggml-vulkan`. Record the A revision and working diff, loaded library paths, compiler, GPU, and driver with each run.

```sh
test-vulkan-activation-cache 0 control
test-vulkan-activation-cache 0 inplace
```

`all` runs both cases. No model, scheduler, CPU fallback, private backend struct, or replacement implementation participates.

The probe allocates F32 storage with 514 elements, shaped `[257, 2]`. A shared view has shape `[256, 2]`, a row stride of 257 elements, and a nonzero byte offset of `sizeof(float)`. The view fits exactly inside that storage. Both matmuls retain this same view pointer; they do not use the SCALE result pointer as a new input identity. Two distinct F16 weight tensors have equal values and shape `[256, 32]`.

The explicit graph order is:

1. The first matmul reads the original activation view.
2. `ggml_scale_inplace` doubles the complete underlying contiguous F32 storage.
3. The second matmul reads the original activation view again.

Public graph inspection checks this order. The separate negative control doubles another allocated tensor instead of the activation storage. Independent same-GPU reference graphs read original input bytes and input bytes doubled before submission, using the same shapes and view offset. Their matmul outputs must be finite, positive, and distinct before comparisons begin.

The probe compares all 64 outputs from each matmul against its independent reference. It also checks all 514 SCALE results and all 514 activation-storage values against their known contents. The comparison tolerance is zero and nonfinite values fail. Matmul outputs start as NaNs. SCALE is not poisoned, because its output aliases its initialized input; an omitted SCALE instead fails the known doubled-data comparison. Buffers and metadata survive computation, synchronization, and readback.

The nonzero view includes each backing row's last element. The independent reference uses the same layout, so changing from the earlier zero-offset probe changes valid numerical outputs. Do not reuse an absolute error figure across these probe versions.

## Correction and type boundaries

The backend invalidates the converted-activation identity when a recorded tensor write overlaps the cached source's storage range. For ordinary Vulkan buffers, it reuses A's existing `ggml_vk_tensors_overlap` operation with `elementwise=false`, so exact aliases count as overlap. The range calculation includes the view offset and the tensor's byte span. It clears `prealloc_y_last_pipeline_used`, `prealloc_y_last_tensor_used`, and `prealloc_y_last_decode_vector_staging` together.

The new traversal runs after the operation switch in `ggml_vk_build_graph`, before recording `tensor_ctxs[node_idx]` and before any context submission. Every output selected by the fused-write mask is considered. This location is deliberate: a traversal that stops once it finds an ordinary synchronization dependency can miss later fused writes, and invalidating before the operation records its own conversion can leave a newly repopulated identity stale after a subsequent output write. The new traversal describes the cache state after the operation is recorded. It does not move the existing tensor-hazard bookkeeping or introduce a new allocation or cache family.

The existing overlap helper casts both buffers' contexts to Vulkan buffer contexts. The new caller therefore checks both the written tensor and the cached source before calling it. Missing buffers, aggregate multi-buffer wrappers, and non-Vulkan buffer types conservatively clear the cache identity. Only ordinary Vulkan buffers proceed to the range comparison. All these checks use existing A buffer predicates.

The multi-buffer check is needed to make the type condition accurate: an aggregate wrapper inherits its first child's buffer type, but its context has the aggregate layout. `ggml_backend_buffer_is_vk` alone identifies the buffer type and cannot distinguish that wrapper. The source review did not establish a valid public graph that directly uses an aggregate wrapper as a computable tensor buffer, and the tests do not construct one. This is a local type guard, not a claim that a wrapper graph bug was reproduced. Similarly, the conservative non-Vulkan branch avoids introducing a UMA/host-context cast; no UMA runtime qualification follows from the RX 6800 fixture.

The disjoint control establishes that scaling independent storage does not change the mathematical result. It does not measure cache-hit frequency or prove that an implementation retained every optimization. For comparable ordinary Vulkan buffers, source review supplies the reason to preserve the identity after disjoint writes. Unknown storage conservatively loses the optimization.

## Recorded before and after evidence

The observed runs used the RX 6800, GCC 16.2.1, and the workspace `/home/kunweiz/Desktop/atomic-llama-cpp-turboquant-clean-20261018`. The activation-source baselines already include the separate 12-line scratch correction. These results therefore isolate an additional defect rather than repeating the earlier scratch-overwrite baseline.

### Initial zero-offset probe

The initial zero-offset probe passed its disjoint control. In the in-place case, the SCALE and activation-storage checks passed for all 514 values, but the later matmul differed in all 64 outputs. The reported maximum absolute error was approximately 16.0306; one reported pair was expected 31.8711 versus actual 15.9355. This isolates the stale converted activation from an omitted or incorrect SCALE operation.

### Frozen nonzero-offset probe, before the source correction

The frozen probe source SHA256 is:

```text
5d64a8a5eab81562ae07827fe959240591c277f80ec6ff316b316192a9ade020
```

Its view offset is `sizeof(float)`, asserted nonzero in the test. The actual baseline was recorded under:

```text
build-clean-vulkan/details-activation-offset-before-d3kmq975/
```

| Command or check | Observed result |
|---|---|
| `test-vulkan-activation-cache 0 control` | Exit 0; all comparisons passed |
| `test-vulkan-activation-cache 0 inplace` | Exit 1 |
| In-place SCALE output | All 514 values correct |
| In-place activation storage | All 514 values correct |
| In-place later matmul | All 64 values wrong; maximum absolute error approximately 15.4662 |

One reported later-matmul pair was expected 25.835 versus actual 12.9175. These figures belong to the frozen nonzero-offset version; the zero-offset error figures above are retained only as earlier evidence. Keep the original stdout, stderr, exit status, and installed-library identity with this receipt.

### Final combined candidate: actual runtime qualification passed

The source correction and the existing scratch correction together produce the final candidate with `ggml/src/ggml-vulkan/ggml-vulkan.cpp` SHA256:

```text
6453d62fe781b6452406476f88ba4145fc55af7c7207b68357b1d8032a128c6a
```

The final candidate was actually rebuilt, installed, and run on the RX 6800. All seven seam CTests passed. The recorded stdout is:

```text
build-clean-vulkan/details-final-gpu-pii1wep9/a-seams-test.stdout
```

Both activation-source cases used the frozen four-byte-offset probe. Every comparison below had zero mismatches and maximum absolute error zero:

| Case | First matmul | Later matmul | SCALE result | Activation storage |
|---|---|---|---|---|
| Disjoint control | All 64 values exact | All 64 values exact | All 514 values exact | All 514 values exact |
| In-place mutation | All 64 values exact | All 64 values exact | All 514 values exact | All 514 values exact |

The same final candidate also passed all five scratch-reuse cases and all 16 packed SET_ROWS comparisons. The no-split attention fixture's 524288 closed-form comparisons had maximum absolute error zero. These final-candidate results are distinct from the preceding scratch-only qualification.

The actual Vulkan library and activation-test executable identities for this run are:

```text
Vulkan library SHA256:
5529dd36ee499771c7b93c56c0aca1d16baf28bb8beb5c07f623914a038a0453
test-vulkan-activation-cache SHA256:
a74790c17850097ac38d41a67ed02ed3248a9a1fc0e4142f644e4dbec91792f1
```

This runtime qualification uses the real installed library. A syntax-only build does not supply the before/after evidence above.

## Limits

The observed baseline establishes a public GGML graph correctness defect. The corrected run qualifies the stated fixtures. It does not establish all activation-cache coherence behavior, all overlapping or strided layouts, all fusion combinations, host-side writes during execution, graph capture/replay, UMA, aggregate-wrapper execution, other backends, or a particular production model trigger. The available recurrent-state clearing call is not itself evidence of this exact stale-cache sequence. No real-model throughput or quality claim follows from these probes.

The broader Vulkan operator gate on the preceding scratch-corrected version reported 25931 of 25933 passing, with an FA failure that reproduced in isolated before/after runs and a TOPK failure that did not reproduce in either isolated rerun. Both failures remain recorded. The full operator-gate results for the final candidate are tracked in [VALIDATION.md](../../VALIDATION.md). Passing these narrow cache fixtures does not imply that the complete Vulkan operator gate passed.
