# GAP-1 归因链最后一环：phasestrace 裁决（r9）

## 判定：覆盖窗口收窄到 graph_compute 执行内部——且「覆盖」可能是「不可见」而非「清零」

## 决定性证据（r9，三时点 W readback + W dump）

```
[phasestrace] SAVED tensor=0x59c02e441ed0 for blk.0.hc_attn_down.weight rank0
[phasestrace] T1_post_all_load W_down_r0 readback=2d3bd83b1cbb9abb63ba40bdb2bcbcbb
[phasetrace] T2_first_graph_compute W_down_r0 readback=2d3bd83b1cbb9abb63ba40bdb2bcbcbb
```

| 时点 | 读取路径 | 结果 |
|---|---|---|
| T1（全部加载完成） | `ggml_backend_tensor_get`（host 路径） | `2d3bd83b…` **非零** |
| T2（首次 graph_compute 入口） | `ggml_backend_tensor_get`（host 路径） | `2d3bd83b…` **非零（与 T1 逐字节相同）** |
| T3（dispatch W dump） | `vk staging copy`（GPU 路径） | **全零** |

## 关键发现：T1/T2 与 T3 用的是不同的读取路径

T1/T2 用 `ggml_backend_tensor_get`（经 meta buffer → simple tensor → 底层 buffer 的 host 接口）——**host 路径**。
T3 用 `ggml_vk_tensor_subbuffer` + `vkCmdCopyBuffer`（Vulkan staging）——**GPU 路径**。

**数据在 host 路径始终可见（T1=T2=非零），但在 GPU 路径不可见（T3=零）。**

这推翻了 r8 的「覆盖」假设——数据可能从未被覆盖。更可能的解释是：**MIRRORED fan-out 的 host 写入从未到达 GPU 可见的内存。**

## 剩余嫌疑（最终收窄）

1. **非 HOST_COHERENT 内存**：per-rank Vulkan buffer 的 host mapping 写入后，device memory 不可见（需 `vkFlushMappedMemoryRanges`）。T1/T2 的 host readback 看到的是 host mapping 缓存，GPU 读的是 device memory——两者不同步。
2. **host write 路径与 GPU read 路径指向不同 VkBuffer**：`ggml_backend_tensor_set` 走的 buffer 链与 `ggml_vk_tensor_subbuffer` 解析的 VkBuffer 不同（r7 已见 subbuffer buf 与 tensor->buffer 是不同间接层）。
3. **graph_compute 内的 buffer 重映射**：graph_compute 入口时数据在 host mapping 可见，但在执行过程中 buffer 被重新映射/绑定，导致 GPU 读到不同位置。

## 修复方向（供 Engineer）

1. **检查 `ggml_backend_vk_buffer` 的 memory type**：确认 per-rank simple tensor 的 VkBuffer 分配时是否用了 HOST_VISIBLE|HOST_COHERENT。若非 COHERENT，在 MIRRORED fan-out 的 `ggml_backend_tensor_set` 后加 `vkFlushMappedMemoryRanges`。
2. **对比 host write 与 GPU read 的 VkBuffer 链**：在 `ggml_backend_tensor_set` 内打印实际写入的 VkDeviceMemory/VkBuffer，与 `ggml_vk_tensor_subbuffer` 解析的对比。
3. **在 T2 与 T3 之间加一个 GPU 路径 readback**（在 graph_compute 内、hc_segment 之前用 vk staging 读一次）——区分「从未到达 GPU」vs「graph_compute 内被清零」。

## 附注

- T2 的 `ggml_backend_tensor_get` 与 T1 完全一致（`2d3bd83b1cbb9abb63ba40bdb2bcbcbb`）——数据在 host 路径从加载到首次计算稳定存在。
- r8 的「1920 对 readback 全部一致」用的也是 host 路径（`ggml_backend_tensor_get`）——与 T3 的 GPU 路径不同。
- 编译修复：`extern "C"` 声明从函数体内移到文件头（函数内 `extern` 声明 C 链接函数在 C++ 中不合法）。
