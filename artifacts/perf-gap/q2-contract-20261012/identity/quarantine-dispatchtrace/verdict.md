# GAP-1 最终判别：dispatch 侧 W_down 张量身份回溯（r7 verdict）

## 判定：新形态——身份完全匹配，写入执行，但数据在 dispatch 时不可见

## 决定性证据（r7，同一 run，同一张量 blk.47.hc_ffn_down.weight，逐 rank 对照）

### 加载期 MIRRORED fan-out 写入目标（meta.cpp:2330）
```
rank0: simple{buf=0x6490fe8fb9f0 data=0x3e872980} set(data,0,6553600)
rank1: simple{buf=0x6490fe8fdc10 data=0x3edac980} set(data,0,6553600)
rank2: simple{buf=0x6490fe8ffe50 data=0x3e94ad80} set(data,0,6553600)
rank3: simple{buf=0x6490fe902090 data=0x3e872980} set(data,0,6553600)
rank4: simple{buf=0x6490fe9042d0 data=0x3e30f900} set(data,0,6553600)
```

### dispatch 侧 W_down 张量（ggml_vk_hc_segment 内）
```
rank0: tensor=0x6490f94c8410 buffer=0x6490fe8fb9f0 data=0x3e872980 (no view_src)
rank1: tensor=0x6490f953ef10 buffer=0x6490fe8fdc10 data=0x3edac980 (no view_src)
rank2: tensor=0x6490f95b5a10 buffer=0x6490fe8ffe50 data=0x3e94ad80 (no view_src)
rank3: tensor=0x6490f962c510 buffer=0x6490fe902090 data=0x3e872980 (no view_src)
rank4: tensor=0x6490f96a3010 buffer=0x6490fe9042d0 data=0x3e30f900 (no view_src)
```

### 逐 rank 对照表
| rank | MIRRORED buf | dispatch buffer | 匹配 | MIRRORED data | dispatch data | 匹配 |
|---|---|---|---|---|---|---|
| 0 | 0x6490fe8fb9f0 | 0x6490fe8fb9f0 | **✓** | 0x3e872980 | 0x3e872980 | **✓** |
| 1 | 0x6490fe8fdc10 | 0x6490fe8fdc10 | **✓** | 0x3edac980 | 0x3edac980 | **✓** |
| 2 | 0x6490fe8ffe50 | 0x6490fe8ffe50 | **✓** | 0x3e94ad80 | 0x3e94ad80 | **✓** |
| 3 | 0x6490fe902090 | 0x6490fe8fb9f0→0x6490fe902090 | **✓** | 0x3e872980 | 0x3e872980 | **✓** |
| 4 | 0x6490fe9042d0 | (同上模式) | **✓** | 0x3e30f900 | (同) | **✓** |

**buffer 与 data 指针在全部 5 rank 完全一致。** dispatch 张量无 view_src（standalone per-rank simple tensor，非 VIEW 非 clone）。

### 但 W dump 仍全零
W dump 的 subbuffer 显示 `buf=0x6490fe8fb730, off=1049041280`——这是 `ggml_vk_tensor_subbuffer` 经 Vulkan 后端解析后的底层 VkBuffer（与 `tensor->buffer` 的 `ggml_backend_buffer_t` 是不同间接层）。

## r5/r6 结论修正

**r5/r6 的「身份分叉」结论是跨 run 比较的伪影。** r5 的 dispatch buf（0x5705644297d0）与加载期 buf（0x570564428160）比较的是不同 run 的地址；r7 在同一 run 内比较，buffer 与 data 指针完全匹配。r6 的守卫未生效不是因为「路由分叉」，而是因为根本没有分叉——守卫保护的问题不存在。

## 剩余嫌疑（收窄到最后一个环节）

身份匹配 + 写入调用 + W 仍全零 → **写入未到达 dispatch 可见的内存**。候选：

1. **`ggml_backend_tensor_set` 静默失败**：simple tensor 的 buffer（per-rank Vulkan buffer）可能不支持 set_tensor（返回错误被吞掉），或写入到 host staging 而非 device memory。
2. **host mapping 非 coherent**：simple tensor 的 data 指针（0x3e872980）是 host mapping，但底层 VkBuffer 的 memory type 不是 HOST_COHERENT——host 写入对 GPU 不可见，需要 explicit flush。
3. **`ggml_vk_tensor_subbuffer` 解析到不同底层 VkBuffer**：subbuffer buf=0x6490fe8fb730 与 tensor->buffer=0x6490fe8fb9f0 是不同间接层——如果 subbuffer 解析链经过的 buffer 对象与 host write 的目标不同（例如 host write 走 CPU buffer 而 subbuffer 走 GPU buffer），数据就不通。

**修复方向（供 Engineer）**：在 `ggml_backend_tensor_set` 的 simple tensor 路径加返回值检查 + 在 write 后立即读回验证（host 侧 readback 确认写入到达）；检查 simple tensor buffer 的 VkMemoryType flags 是否 HOST_COHERENT。

## 附注

- dispatchtrace 的 site=2 显示 `output_hc_down.weight`（head 层），data=0xb000（低地址）——不同张量不同位置，模式一致。
- 加载期 alloc trace 补充了 tensor 本体指针与 meta buffer（`buffer=0x6490f9271c90`），确认 dispatch 张量确实是 stc_static 的 per-rank simple tensor（非 meta 张量本体）。
