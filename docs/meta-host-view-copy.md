# Meta host<->shard virtual-address copy

## Scope

Private worktree from `r28b-pillar5-refine` at `52cd0404d`. No other tree, no
integration branch, no GPU path and no full test suite were changed.

## Root cause

1. `ggml_backend_tensor_copy` took the generic host fast paths when
   `ggml_backend_buffer_is_host(buf)` was true. A CPU Meta buffer reports
   `is_host=true` when every child is a host buffer, but the top-level Meta
   tensor data is the virtual sentinel `0x2000000000000000`. The sentinel must
   not be dereferenced. Only the side that is actually dereferenced needs the
   exclusion: a host source may call `tensor_set` on a Meta destination, and a
   host destination may call `tensor_get` from a Meta source.
2. The host<->Meta view slow path used proportional scaling
   (`view_offs * view_ne / parent_ne` and the same rule for split widths).
   That is wrong for a non-zero view that starts inside a child shard or
   crosses a rank boundary.
3. A 1-D partial view reached the row-stride assertions before the 1-D logical
   range mapping (`nb[1]` is the parent row size, not the view size).
4. A non-contiguous view has `ggml_nbytes(view)` equal to the **strided span**,
   not the compact logical size. The copy path must use the view's physical
   strides for the data buffer and copy only the logical rows, skipping gaps.

## Exact mapping model

For a parent split state on axis `a`, the logical axis is the sequence:

```
for s in segments:
  for r in 0..nr[s]-1:
    for j in 0..n_bufs-1:
      chunk of ne[s*n_bufs+j] elements
```

The child simple tensor for device `j` stores only that device's chunks, in
the same `s,r` order. A view is intersected with this sequence:

- `view_start` is derived from `view_offs` and the parent stride along `a`.
- For axis 0 the offset is converted to elements through the quantization
  block size (`blck_size`), and every child width must be block-aligned.
- For axis >= 1 an unaligned offset is split into a floor row/block plus a
  `within` byte offset; `within` is added to every child view start.
- Each intersecting parent repeat becomes one view segment. Identical
  consecutive segments are folded into `nr`.
- Child view starts are bytes into the child simple tensor. For axis 0 they
  use `(elements / blck_size) * nb[0]`; for axis >= 1 they use
  `elements * nb[axis]`, plus `within`.
- If the view intersects more than 16 distinct segment patterns the state is
  not representable and the helper returns false rather than silently using
  the proportional rule.

## Counterexamples

1. Single segment `[32, 32]`, view `[8, 40)`: exact child widths `[24, 8]`,
   child starts `[8, 0]`; proportional gives `[16, 16]` and wrong starts.
2. Segmented `[16,16] | [16,16]`, view `[8, 56)`: exact view segments
   `[8,16]` and `[16,8]`, child starts `[8, 0]`. The first child chunk is
   `[8,16)` of segment 0; the second is `[32,48)` of segment 1. They are
   contiguous in child 0 at offsets `[8,32)`.
3. Repeated `8/8 x 4`, view `[4, 36)`: exact view segments `[4,8]`, `[8,8]`,
   `[4,0]`, child starts `[4, 0]`. Child 0's chunks are at child offsets
   `[4,8)`, `[8,16)`, `[16,20)`; child 1's at `[0,8)`, `[8,16)`.
4. Q8_0 parent of 128 elements split `[64,64]`, view offset 34 bytes
   (1 block) and length 64 elements: exact widths `[32,32]`, child starts
   `34` and `0` bytes. Proportional byte/element mixing is wrong.
5. Non-contiguous/unaligned 2D 4x4, split axis 1 `[2,2]`, view 2x2 at byte
   offset 20: exact row widths `[1,1]`, child starts `20` and `4` bytes.
   The host view has `nb[1]=16`, `nb[2]=32`; `ggml_nbytes(view)=24`, not 16.
   The full-view copy reads row 0 at offset 0 and row 1 at offset 16, and
   writes child rows at the exact child starts. A compact 16-byte copy would
   write the second row into the wrong place.

## Fix

- `ggml/src/ggml-backend.cpp`
  - Empty copies are a no-op.
  - The host fast paths exclude only the sentinel-bearing side:
    source-host requires `!src_meta`; destination-host requires `!dst_meta`.
  - The host->Meta staging identity hook remains first. Remaining cases try
    the destination buffer cpy hook, then the blocking `get`/`set` fallback;
    a failed fallback allocation aborts instead of dereferencing null.
- `ggml/src/ggml-backend-meta.cpp`
  - Added `ggml_backend_meta_view_exact_split` for segmented/repeated
    parents, non-contiguous views, unaligned axis >= 1, and quantization
    blocks. It replaces the proportional view state and child offsets.
  - `handle_view` uses the exact state and skips the proportional
    post-adjustment.
  - `init_tensor_impl` uses the exact child starts in bytes.
  - Added `ggml_backend_meta_view_copy`:
    - full-view copies accept `size == ggml_nbytes(tensor)`, use the tensor's
      physical `nb` for the data buffer, copy only logical rows, and skip
      gaps;
    - partial 1-D copies use the existing compact logical range mapping;
    - axis 0 uses the quantization block size for chunk and row offsets.
  - Empty set/get/memset are no-ops.
  - The Meta allocation assigns `t->data = get_base(meta_buf)` before
    `init_tensor_impl`, then restores the sentinel. This is defensive
    ordering, not a proven root cause: on the first CPU Meta allocation round
    the simple buffers are still null.
- `tests/CMakeLists.txt`
  - Registered `test-meta-host-view-copy`.

## Test

`tests/test-meta-host-view-copy.cpp` uses two real CPU child devices behind a
Meta device. It is deterministic and does not use a GPU or a model. It covers:

- host non-view -> Meta non-view and the reverse;
- host non-zero view -> Meta non-zero view and the reverse, view `[8, 40)`
  crossing the `[32, 32]` boundary;
- Meta -> Meta with different splits (`[32, 32]` vs `[16, 48]`);
- segmented parent `[16,16] | [16,16]`, view `[8, 56)`;
- repeated parent `8/8 x 4`, view `[4, 36)`;
- Q8_0 parent split `[64,64]`, view at byte offset 34, length 64, both
  directions, plus a partial `set/get` at view offset 34 / size 34;
- non-contiguous and unaligned 2D axis-1 view (4x4 parent, 2x2 view at byte
  offset 20);
- empty copy/get and alias no-ops.

Build and run, only the corresponding target, small incremental build:

```bash
cmake -S tree -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build -j 8 --target test-meta-host-view-copy
ctest --test-dir build -R '^test-meta-host-view-copy$' --output-on-failure
```

Final run after the last fix: `1/1` passed.

Two test-setup corrections were needed during bring-up; neither relaxed the
production assertion:

- the host->Meta view check compared `got[8..40)` with `view_src[8..40)` while
  `view_src` held only the 32 view values; it now compares the two ranges
  explicitly;
- the non-contiguous 2D view was initially filled with a compact 16-byte
  buffer, but its physical span is 24 bytes (`nb[1]=16`). The test now writes
  the host root at the view's positions; the copy still has to skip the gaps
  and write the two child rows at the exact starts.

## Unverified / limits

- No GPU / five-rank Meta run.
- No full test suite; only `test-meta-host-view-copy` was run.
- No compose binary rebuild or end-to-end model load.
- `memset_tensor` is not covered by the new test.
- Axis 2/3 full-view copy is not implemented in `ggml_backend_meta_view_copy`;
  only axes 0 and 1 are handled.
- Partial non-contiguous multi-row ranges that are not whole rows/blocks are
  not covered; the compact partial path remains for 1-D.
- A view intersecting more than 16 distinct segment patterns is not
  representable in `ggml_backend_meta_split_state`; the helper returns false
  rather than silently using the proportional rule.
- No timing/performance measurement of the asymmetric host fast-path guard.
