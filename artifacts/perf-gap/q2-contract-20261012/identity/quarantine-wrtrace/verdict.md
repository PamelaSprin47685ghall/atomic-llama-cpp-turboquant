# GAP-1 r11 verdict：write/read 链对照

## 判定：情形 A（ggml 层身份匹配）+ 新发现（buffer 非 HostVisible，写入走 staging 路径）

## 决定性证据（r11，同一张量 blk.0.hc_attn_down.weight rank0）

**GAP1-W（写入链，ggml_backend_vk_buffer_set_tensor）：**
```
ggml_buf=0x62d2febe8900  dev_buffer(vk_buffer obj)=0x62d2f8fb5a30  ptr=(nil)  vk_ptr_base=0x1000  off=100597760  size=6553600
```

**GAP1-R（读取链，ggml_vk_tensor_subbuffer）：**
```
ggml_buf=0x62d2febe8900  dev_buffer(context)=0x62d2febe88b0  ptr=(nil)  vk_ptr_base=0x1000  off=100597760  size=6553600
```

## 对照分析

| 维度 | GAP1-W | GAP1-R | 匹配 |
|---|---|---|---|
| ggml_buf (ggml_backend_buffer_t) | 0x62d2febe8900 | 0x62d2febe8900 | **✓ 完全一致** |
| vk_ptr_base | 0x1000 | 0x1000 | **✓ 一致** |
| offset (vk_tensor_offset) | 100597760 | 100597760 | **✓ 一致** |
| size | 6553600 | 6553600 | **✓ 一致** |
| ptr (host mapping) | nil | nil | **✓ 一致（均非 HostVisible）** |
| dev_buffer | 0x62d2f8fb5a30 (vk_buffer obj) | 0x62d2febe88b0 (context obj) | 不同对象但不同间接层 |

## 关键新发现

**`ptr=(nil)`——per-rank VkBuffer 非 HostVisible！** 这意味着：
- `ggml_vk_buffer_write_2d` 走 **staging buffer + command submission 路径**（非直接 memcpy）
- 写入是同步的（代码提交后等 fence）
- 读取（W dump 的 vkCmdCopyBuffer）也从同一 VkBuffer 读

**但 W dump 仍全零。** ggml 层身份完全匹配（buffer/offset/base/size），问题出在 staging 写入是否真实到达 device VkBuffer。

## 剩余嫌疑（最终）

1. **staging 写入路径缺陷**：`ggml_vk_buffer_write_2d` 的非 HostVisible 分支可能未正确提交/等待
2. **vk_buffer 的 dev_buffer 字段在写入与读取时不同**：GAP1-W 的 vk_buffer 对象 (0x62d2f8fb5a30) 与 W dump subbuffer 的 vk_buffer 对象 (0x62d2febe9f70) 是不同对象——但 ggml_buf 相同意味着它们应该通过 context 链解析到同一 VkBuffer

## 下一步

在 staging 路径的 `ggml_vk_ctx_end(subctx)` 后加一个 GPU readback（vk staging read），验证写入是否真实到达 device memory。若到达 → 读取链有 bug；若未到达 → staging 写入有 bug。
