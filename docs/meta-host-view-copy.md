# Meta host<->shard virtual-address copy

## Scope

This work fixes the CPU Meta buffer copy path in a private worktree created from
`r28b-pillar5-refine` at `52cd0404d`. No other tree, no integration branch and
no GPU path were changed.

## Root cause

1. `ggml_backend_tensor_copy` took the generic host fast paths when
   `ggml_backend_buffer_is_host(buf)` was true. A CPU Meta buffer reports
   `is_host=true` when every child is a host buffer, but the top-level Meta
   tensor data is the virtual sentinel `0x2000000000000000`, not a host
   address. A Meta-to-Meta copy could therefore dereference the sentinel.
2. The host<->Meta view slow path did not use the real logical coordinates.
   `ggml_backend_meta_buffer_init_tensor_impl` scaled `view_offs`
   proportionally by `view_ne / parent_ne`. That is wrong when a non-zero view
   starts inside a child shard and crosses a rank boundary. For example, a
   64-element parent split `[32, 32]` and a view `[8, 40)` needs child widths
   `[24, 8]` and child starts `[8, 0]`; the proportional rule produced
   `[16, 16]` and `[2, 0]`.
3. `ggml_backend_meta_get_split_state` applied the same proportional rule to
   views, so the split widths and the simple-tensor offsets disagreed.
4. A 1-D partial view hit the row-stride assertions in `set_tensor`/`get_tensor`
   before any 1-D special handling (`nb[1]` is the parent row size, not the
   view size).
5. `ggml_backend_meta_alloc_ctx_tensors_from_buft` initialized the simple
   tensors before assigning the Meta base, so the root simple-tensor data
   offset was computed from the post-init sentinel.

## Fix

- `ggml/src/ggml-backend.cpp`
  - Empty copies are a no-op.
  - Meta buffers are detected before the host fast paths; the host fast paths
    are skipped when either side is Meta. The host->Meta staging identity fast
    path (`ggml_backend_buffer_copy_tensor`) is still tried first.
  - Meta copies use the blocking `get`/`set` fallback; a failed fallback
    allocation aborts instead of dereferencing null.
- `ggml/src/ggml-backend-meta.cpp`
  - Added `ggml_backend_meta_view_simple_split`: for a contiguous view of a
    simply split parent, intersect the view's absolute range along the split
    axis with each child slice. The result is the exact per-child width and the
    exact per-child view start.
  - `handle_view` returns that exact state and skips the proportional
    post-adjustment.
  - `init_tensor_impl` uses the same helper for the simple-tensor
    `view_offs`, so widths and offsets cannot drift.
  - `set_tensor`/`get_tensor` use a 1-D logical range mapping for
    `ne[1] == 1`, before the row-stride assertions. It supports non-zero
    offsets and segmented/repeated split states.
  - Empty set/get/memset are no-ops.
  - The Meta allocation assigns `t->data = get_base(meta_buf)` before
    `init_tensor_impl`, then restores the sentinel.
- `tests/CMakeLists.txt`
  - Registered `test-meta-host-view-copy`.

The host->Meta staging identity fast path is unchanged. The fast path still
rejects views and non-mirrored destination splits; those fall back to the
corrected slow path.

## Test

`tests/test-meta-host-view-copy.cpp` uses two real CPU child devices behind a
Meta device. It is deterministic and does not use a GPU or a model. It covers:

- host non-view -> Meta non-view and the reverse;
- host non-zero view -> Meta non-zero view and the reverse, where the view
  `[8, 40)` crosses the `[32, 32]` child boundary;
- Meta -> Meta with different source/destination splits (`[32, 32]` vs
  `[16, 48]`);
- empty copy and empty get;
- tensor alias and view alias no-ops.

Build and run, only the corresponding target:

```bash
cmake -S tree -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build -j 44 --target test-meta-host-view-copy
ctest --test-dir build -R '^test-meta-host-view-copy$' --output-on-failure
```

Result: `1/1` passed.

The first run exposed an error in the test assertion, not in the production
copy: the host->Meta check compared `got[8..40)` with `view_src[8..40)` while
`view_src` holds only the 32 view values. The observed bytes were already
correct (`got[8..40) == view_src[0..32)`). The test was corrected to compare
the two ranges explicitly. No test was deleted and no assertion was relaxed.

## Unverified

- No GPU / five-rank Meta run.
- No full test suite; only `test-meta-host-view-copy` was run.
- No compose binary rebuild or end-to-end model load.
- `memset_tensor` is not covered by the new test; its split-condition change is
  the same as the candidate and was not exercised.
- The exact helper handles contiguous, simply split parents. Segmented,
  repeated, mirrored, unaligned and non-contiguous views keep the existing
  heuristic.
