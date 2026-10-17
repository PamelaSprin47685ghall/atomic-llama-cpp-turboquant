# GAP-1 终局判别：sourcetrace 裁决（r8）

## 判定：假设 B 坐实——源数据非零、写入成功、但 dispatch 时被覆盖为零

## 决定性证据（r8，sourcetrace + write-then-readback + W dump 三方对照）

### 1. 源数据非零（假设 A 排除）

MIRRORED fan-out 的源（GGUF mmap 数据）前 16 字节 hex：
```
blk.0.hc_attn_norm.weight  src_hex=00c0913f0000ec3e0004813f0018873f  (F32, 非零)
blk.0.hc_attn_down.weight  src_hex=2d3bd83b1cbb9abb63ba40bdb2bcbcbb  (BF16, 非零)
blk.0.hc_attn_up.weight    src_hex=81bee3bc02be44bdb83d623e42bd593d  (BF16, 非零)
```
**源是真实权重字节，不是零填充区。** 假设 A（loader offs/mapping 指向零区）排除。

### 2. 写入成功（write-then-readback 非零且与源逐字节一致）

```
blk.0.hc_attn_down.weight rank0:
  src_hex=      2d3bd83b1cbb9abb63ba40bdb2bcbcbb
  readback_hex= 2d3bd83b1cbb9abb63ba40bdb2bcbcbb  ← 完全一致
（全部 5 rank × 全部 hc_ 张量：readback 与 src 逐字节一致）
```
**`ggml_backend_tensor_set` 的写入真实到达了 per-rank simple tensor 的内存。** 写入机制完全正常。

### 3. 但 dispatch 时 W dump 仍全零

```
r0-r4 W_down: 0/64 全零；lo 0/320 全零；mixed 0/2560 全零（与 r2-r7 完全一致）
```

### 4. loader else-branch 打印未出现（插桩位置笔误，如实记录）

loader 的打印在 `if (use_mmap)` 的 else 分支内，但 runner 传 `--no-mmap`，实际走非 mmap 路径（`if (use_mmap)` 的大 else）。sourcetrace 的 meta.cpp MIRRORED 打印不受影响（ggml_backend_tensor_set 从非 mmap 路径同样触发 fan-out）。

## 结论

**写入链完整且正确**：源非零 → MIRRORED fan-out 执行 → per-rank simple tensor 收到正确字节（readback 证实）。**但加载完成到 dispatch 之间，该内存被覆盖为零。**

## 覆盖窗口与候选清零者

覆盖发生在「加载完成」之后、「dispatch 读取」之前。候选（按嫌疑排序）：

1. **ggml-alloc 图构建**：图构建期的 tensor allocator（`ggml_alloc` / `ggml_gallocr`）可能重新分配或 zero-init per-rank buffer 上的内存区域，覆盖已写入的权重。
2. **Vulkan 缓存失效/重分配**：`ggml_vk_cache_invalidate_all` 或 buffer 池管理可能释放并重新分配 per-rank buffer，新分配 = 零页。
3. **meta buffer 的 stc_compute 初始化**：`ggml_backend_meta_buffer_init_tensor_impl(stc_compute_0/1, ...)` 在图构建时可能触碰同一内存区域。

**修复方向**：在加载完成后、图构建前/后各做一次 readback（定位覆盖发生的精确阶段），或在图构建路径上对 hc_ 权重张量的 buffer 做 write-protect 观察。这属 Engineer 下一轮工作。

## 附注

r8 的 sourcetrace 打印了 384 张量 × 5 rank = 1920 对 src/readback 行——**全部一致，零失败**。写入链的每个 rank 每个张量都正确搬运了字节。覆盖是加载后的全局性事件，不是个别张量的偶发问题。
