# Vulkan Turbo WHT regression

This test links the installed public `ggml::ggml`, `ggml::ggml-cpu`, and
`ggml::ggml-vulkan` targets.
It uses only installed public headers. Execute `test-vulkan-wht [device-index]`
with an actual Vulkan device; absence of Vulkan is a failure, not a passing CPU
fallback. CMake adds this target only when the installed Vulkan target is available.

## Observed before / after

On the actual RX 6800 / RADV device, the identical 12-case public-op test
failed 10 cases against pinned A before the patch. Only the unscaled group-128
forward/inverse controls passed. After the two-file A fix all 12 cases passed;
the maximum absolute difference from the public CPU reference was 0 on this
device. The 513-row cases exercise the old group-scaled X/Y dispatch holes;
the four scale cases cover forward input and inverse output scaling.

Reproduce the baseline in a separate build of the pinned upstream commit, then
rebuild with the two reviewed WHT files and run the same test. Do not overwrite
an existing build or reinterpret missing Vulkan as a successful CPU fallback.
See `../../VALIDATION.md` for the retained raw log locations.

Every run constructs a public `ggml_turbo_wht` graph and executes it directly
through the selected CPU or Vulkan backend. No mixed scheduler or copied CPU
kernel serves as the reference. Output storage begins as NaN so skipped writes
are deterministic failures. Checks remain active in Release builds. Each value
must be finite and match CPU within `2e-5 + 2e-5 * abs(reference)`; this is a
numeric correctness regression, not a bit-for-bit logits parity claim.

All row widths remain 128, so this patch does not broaden Vulkan's existing
`ne[0] % 128 == 0` support contract. Groups use the same sign-array prefix as A's
CPU kernel. Testing 32/64-wide physical rows, noncontiguous scales, malformed
scale lengths, and Z-split grids would be a separate support-expansion task.

## Build and run

```sh
cmake -S fork/tests/a-seams -B build-a-seams \
  -DCMAKE_PREFIX_PATH=/path/to/installed-A-with-reviewed-seams
cmake --build build-a-seams --target test-vulkan-wht
build-a-seams/test-vulkan-wht
```

The final shader's out-of-bounds workgroup return is uniform (`base >= ne`):
all lanes exit together before any shared-memory barrier. Shape support remains
limited to the explicitly checked physical rows and scale layouts.
