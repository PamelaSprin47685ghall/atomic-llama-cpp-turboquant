# Vulkan scratch reuse and SET_ROWS qualification

## Scope

These tests call the installed public GGML and Vulkan APIs. They run actual GPU operations directly, without a backend scheduler or CPU fallback. They do not use private backend structs, replacement kernels, a model, or the discarded TP5/XKV runtimes.

The retained A correction owns one internal contract: overwriting `prealloc_y` invalidates the cached activation identity, and a later writer must wait until prior scratch readers finish. C cannot maintain that private cache or its GPU dependencies through the public graph API. The exception can be removed when the pinned upstream provides this contract itself.

There are two separate migration decisions:

- Retain scratch lifetime correctness. Both masked attention's bitmap and large softmax's reduction partials overwrite the activation scratch. Clear all three cached identity fields at those writes. The no-split masked-attention path must also mark the scratch as requiring synchronization after its final read.
- Reuse current A's F16 SET_ROWS support. The current host registration, shader generation, and runtime selection include F16 sources for Turbo2, Turbo3, and Turbo4 with I32 and I64 indices. Do not port the old F16 rejection. The packed-row test checks this existing behavior without changing A.

## Build and run

Build the seam tests against a complete installed A package from the revision being qualified. Record that revision, any A working diff, the actual library paths, compiler, GPU, driver, and relevant Vulkan environment settings with the run. Link both test targets to `ggml::ggml` and `ggml::ggml-vulkan`.

Run the baseline modes separately so a failure retains its own stdout, stderr, and exit status:

```sh
test-vulkan-scratch-reuse 0 controls
test-vulkan-scratch-reuse 0 softmax
test-vulkan-scratch-reuse 0 flash-mask
test-vulkan-scratch-reuse 0 flash-no-split
test-vulkan-set-rows-types 0
```

After rebuilding and reinstalling the corrected A library, rebuild or relink the tests against that same installation, then run:

```sh
test-vulkan-scratch-reuse 0 all
test-vulkan-set-rows-types 0
```

Exit 0 requires every comparison in the selected mode to pass. Unsupported operations or failed Vulkan initialization are failures, not skipped successes. A syntax-only build does not qualify the tests.

## Scratch fixtures and assertions

Each combined graph explicitly contains matmul, the middle operation, then another matmul. Both matmuls read the same noncontiguous F32 view of shape `[256, 2]`, backed by rows of 257 elements. Their distinct F16 weight tensors have equal data and shape `[256, 32]`. Each result has 64 F32 values. Public graph inspection checks the operation order, and all output arrays begin as NaNs.

| Mode or case | Middle operation | Purpose |
|---|---|---|
| `controls`: small softmax | One row with 16384 columns | Negative control below the large-softmax threshold |
| `controls`: unmasked FA | Width 64, 32 queries, 4096 keys, one query/KV head | Negative control without a mask bitmap |
| `softmax` | One row with 32768 columns | Exercises the three-stage reduction's scratch overwrite |
| `flash-mask` | The same one-head FA with a 4096 by 32 half mask | Exercises the mask bitmap's scratch overwrite |
| `flash-no-split` | Width 64, 32 queries, 4096 keys, 256 query heads and one KV head | Targets a final bitmap read without the split-K reduction barrier on RX 6800 |

All attention cases use F32 queries, F16 K/V, and explicitly requested F32 accumulation. Masked cases retain keys 0 through 2047 and mask the rest with negative infinity. The original one-head cases keep V constant across keys. The no-split target chooses a channel scale from `0.25, 0.5, 1.0`, cycling by channel index. V is zero for keys 0 through 1023, twice that scale for keys 1024 through 2047, and eight times that scale for masked keys 2048 through 4095. Since Q and K are zero, the result is exactly the channel scale. All values and the visible-band sums are binary powers or small integer multiples of them. A changed set of visible key bands can therefore change the result without requiring the backend to retain a small fractional increment in a large accumulation.

Each case compares both matmul results against an independent same-GPU matmul with identical shapes and inputs. It also compares every middle-operation result against an independent same-GPU invocation. Those comparisons require finite, exactly equal float values; they are not a comparison of serialized float bits. A separate closed-form check uses zero tolerance for uniform softmax and absolute tolerance `2e-4` for attention. The no-split target compares all 524288 attention outputs. It does not sample selected coordinates or tokens.

The public API does not expose the private FA tuning state. The mask path requires a supported mask size, a selected block-column width no larger than 256 for these fixtures, and the backend's architecture condition. The 256-head case avoids the `N <= 8` GQA rearrangement and provides at least 256 workgroups before splitting. Record the actual device and confirm the selected path from the pinned source and device tuning when qualifying a different GPU. The `no_split_target` label is a target condition, not a runtime probe of private state.

## Synchronization review

The normal graph overlap tracker sees tensor storage, not the private scratch allocation. Cache invalidation and execution ordering are therefore separate obligations.

Large softmax already marks Y as requiring synchronization after its last dispatch. Split-K attention already inserts a barrier after the dispatch that reads the bitmap, before reducing its separate split-K output. No-split attention has no such reduction barrier; after reading an optimized mask it must mark Y as requiring synchronization. The next matmul then records the existing pre-copy barrier before replacing Y. The conversion's barrier after the copy alone cannot order an earlier bitmap read before that copy.

A passing one-head fixture cannot prove the no-split dependency. Even a passing no-split numerical run does not prove that a missing GPU dependency is harmless; pair the actual-operation test with this source-level ownership review. Keep any synchronization-validation-layer output separate from ordinary numerical results, and do not claim such validation ran unless it did.

## Recorded before and after evidence

The actual-operation runs below used the RX 6800 and an A build compiled with GCC 16.2.1. The workspace was `/home/kunweiz/Desktop/atomic-llama-cpp-turboquant-clean-20261018`. Counts describe full output comparisons, not selected samples. Preserve the exact machine logs, loaded library paths, and source revision alongside these rounded figures.

### Before the scratch correction

| Fixture | Observed result |
|---|---|
| Small softmax negative control | Passed |
| Unmasked one-head FA negative control | Passed |
| 32768-column softmax | Later matmul wrong in 32 of 64 values; maximum absolute error approximately 3380.01 |
| Masked one-head FA | Later matmul wrong in 32 of 64 values; maximum absolute error approximately `2.70e13` |

The combined graphs and their isolated controls use the same GPU, operators, shapes, and initialized inputs. In particular, a correct middle operation cannot make the later matmul legitimately depend on the scratch bytes left behind by that operation.

The no-split target was added during the subsequent synchronization review. Its first probe used eight visible bands with fractional offsets. On the installation before the scratch correction, all 524288 attention values matched the independent same-GPU invocation, but 475136 failed the closed-form tolerance, with maximum absolute error 0.0146484. One reported pair was expected 1.00781 versus actual 1.00293. Its later matmul also differed in 32 of 64 values, with maximum absolute error approximately `2.70297e13`.

That first no-split probe had an unsuitable numerical control: its attention discrepancy also occurred in isolation. It was not evidence of a bitmap synchronization race, and its closed-form failure must not be counted as a correction made by this patch. The final fixture replaces those fractional bands with the binary-scaled values described above; the tolerance remains `2e-4`.

The binary-scaled fixture was then run on the installation before the scratch correction. Its independent FA output passed the closed-form check for all 524288 values, with maximum absolute error zero. The combined graph's later matmul still differed in 32 of 64 values, with maximum absolute error approximately `2.70297e13`. This is a valid independent numerical control for the cache-overwrite regression. It does not by itself demonstrate a missing-dependency race.

That separate baseline receipt is:

```text
build-clean-vulkan/details-no-split-binary-before-t1gqpmbg/
```

The `test-vulkan-scratch-reuse` executable used for that baseline has SHA256:

```text
6bc2694ded21d8e8aff4001a61de2d660eb7c1f7068caddd927d4a1b7b0257e4
```

### After the scratch-only correction

The actual rebuilt and installed A version containing the 12-line scratch correction passed all five cases of `test-vulkan-scratch-reuse 0 all`, including the final binary-scaled no-split fixture. The same qualification passed all 16 packed scatter comparisons in `test-vulkan-set-rows-types 0`.

These passing numerical comparisons establish the stated fixtures. The reason to retain the no-split dependency additionally comes from the synchronization review above; the run does not establish that the earlier installation exhibited a write-after-read race.

The broader Vulkan operator gate on that scratch-corrected version reported 25931 of 25933 passing. Its FA failure also reproduced in isolated before/after runs. Its TOPK failure from the broader gate did not reproduce in either isolated rerun. Both failures remain recorded. The full operator-gate results for subsequent versions are tracked in [VALIDATION.md](../../VALIDATION.md); the narrower seam qualification is not a claim that the complete Vulkan operator gate passed.

### Final combined candidate: actual runtime qualification passed

The subsequent activation-source correction is documented in [vulkan-activation-cache-protocol.md](vulkan-activation-cache-protocol.md). It invalidates cached conversions after recording a graph operation, considers every selected fused output, and guards both buffers before using A's Vulkan range-overlap helper. Unknown buffer types and aggregate multi-buffer wrappers conservatively invalidate the identity. The guard does not constitute runtime qualification of UMA or of tensors stored directly in aggregate wrappers.

The final combined candidate has `ggml/src/ggml-vulkan/ggml-vulkan.cpp` SHA256:

```text
6453d62fe781b6452406476f88ba4145fc55af7c7207b68357b1d8032a128c6a
```

The recorded baseline and the combined-candidate run before the final EOF cleanup used this `vulkan-scratch-reuse.cpp` source SHA256:

```text
ff348961b324c76b1bc3bf4e5d5928ec1935e9112d3224ca7dc40a71e3068e34
```

The baseline receipt records the same test-source SHA256 shown above. The baseline and pre-cleanup combined-candidate executable identities are recorded separately. That candidate was actually rebuilt, installed, and run on the RX 6800. All seven seam CTests passed. The recorded stdout is:

```text
build-clean-vulkan/details-final-gpu-pii1wep9/a-seams-test.stdout
```

| Final check | Actual result |
|---|---|
| All five scratch cases | Passed; every first/later matmul and every middle-operation output matched its independent same-GPU reference exactly, with zero mismatches |
| Binary-scaled no-split FA | All 524288 closed-form comparisons passed; maximum absolute error zero at tolerance `0.0002` |
| The two one-head FA fixtures | Closed-form checks passed; reported maximum absolute error `8.73804e-05` at tolerance `0.0002` |
| SET_ROWS source types | All 16 complete packed-byte scatter comparisons exact |
| Nonzero-offset activation-source control and in-place cases | Both passed; all first/later matmul, SCALE, and storage comparisons had zero mismatches and maximum absolute error zero |

The actual Vulkan library and scratch-test executable identities for this run are:

```text
Vulkan library SHA256:
5529dd36ee499771c7b93c56c0aca1d16baf28bb8beb5c07f623914a038a0453
test-vulkan-scratch-reuse SHA256:
d2e0cad9220416a5d0a9b69ed4250782be58dded2ff52f67d3ff14e2ae270e01
```

These are results from the final combined candidate, in addition to the earlier scratch-only run. The seven passing seam CTests and the broader backend operator gate are separate records; consult [VALIDATION.md](../../VALIDATION.md) for the latter's final-version result and retained failures.

## Existing SET_ROWS qualification

`test-vulkan-set-rows-types` covers Q8_0, Turbo2, Turbo3, and Turbo4 destinations with both I32 and I64 indices, using width 256. It first obtains each quantized row through the same GPU's F32 identity SET_ROWS, then compares F32 and F16 scatter writes into destination rows 5, 1, and 3. The input values are exactly half-representable. The complete destination byte comparison also checks that the other four rows retain their sentinel bytes. Identity rows must be written and distinct, so a missing operation or wrong row permutation cannot pass by symmetry.

The initial RX 6800 run passed all 16 packed scatter comparisons across eight destination/index cases. This qualifies existing source-type and row-addressing behavior in that bounded fixture. It does not independently establish quantizer error, all dimensions and strides, duplicate-index behavior, or correctness on another GPU. Duplicate destination indices are outside the public SET_ROWS contract and are not used.

## Limits

These are backend correctness tests. They do not establish model quality, affected real-model graph frequency, throughput gains, CUDA/Metal behavior, or universal Vulkan-device support. The same-GPU reference avoids attributing normal CPU/GPU numerical differences to this cache defect. Public buffers and contexts remain alive through synchronization and readback; the test does not depend on asynchronous host-memory lifetime assumptions.

### Final source formatting check

The final cleanup removed one empty line at EOF from this test and `vulkan-set-rows-types.cpp`. A byte comparison verified that these were the only changes. The delivered scratch source SHA256 is `27a1166a6f0ad7cac443a4a8461d4bfc4404b5493e1af18a1eb98c916fe3cc1c`. Both test targets were rebuilt and all seven seam CTests passed again; see `build-clean-vulkan/details-eof-seams-20261018.log` and the `eof_only_normalization` record in [the qualification report](../../audit/details-qualification-20261018.json). The rebuilt scratch executable and loaded Vulkan library retain the digests recorded above. Earlier baseline, isolation and full-backend runs retain their original identities.
