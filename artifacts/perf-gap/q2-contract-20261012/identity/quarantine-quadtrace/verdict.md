# r12 裁决：GAP-1 fork 位置三实验（quadtrace）

运行：arms/sitedump-fused-r12/（GGML_VK_CMD_REPLAY=0、GGML_VK_HC_DUMP_SITES=2、GGML_VK_GAP1_READBACK=1、GGML_VK_GAP1_TRACE=1）
库：libggml-vulkan.so.0.18.1 sha256=f6eaf122f8199cacd407c99f88e2513318cdfcf30206b11c3dc4b7d46c6207de（rebuild 树增量编译）

## 实验 1｜四元组对照（GAP1-QUAD，写侧 W vs dump 侧 DUMP）

写侧前 20 条 BF16 写（output_hc_down/up ×5 rank、blk.0.hc_attn_down/up ×5 rank）与 dump 侧 20 条（site1=blk.47.hc_ffn、site2=output_hc，各 down/up ×5 rank）对照：

**output_hc（site2）——完全匹配**（5 rank 逐一核对）：
| rank | 写侧 (ggml_buf, ctx, dev_buffer, vkhandle) | dump 侧 | 判定 |
|---|---|---|---|
| 0 | 0x64cbdca613c0 / 0x64cbdca61370 / 0x64cbd6e2e4f0 / 0x64cbd73cfff0 | 同 | 一致 |
| 1 | 0x64cbdca63500 / 0x64cbdca62d80 / 0x64cbdca63240 / 0x64cbdca63290 | 同 | 一致 |
| 2 | 0x64cbdca65720 / 0x64cbdca64fa0 / 0x64cbdca65460 / 0x64cbdca654b0 | 同 | 一致 |
| 3 | 0x64cbdca67960 / 0x64cbdca671e0 / 0x64cbdca676a0 / 0x64cbdca676f0 | 同 | 一致 |
| 4 | 0x64cbdca69ba0 / 0x64cbdca69420 / 0x64cbdca698e0 / 0x64cbdca69930 | 同 | 一致 |

**offset 也一致**：写侧 output_hc_down off=40960、output_hc_up off=6594560（GAP1-W）；dump 侧 sub_off=40960/6594560（GAP1-DUMPID）。

blk.47.hc_ffn（site1）的 dump 四元组不在写侧前 20 条内——写侧只覆盖前 20 条 BF16 写，blk.47 的写在 20 条之后（预期，非 fork 证据）。GAP1-R（读链 subbuffer 解析）显示 blk.0.hc_attn_down 的读 offset=100597760 与写侧一致——读链 offset 无错位。

**实验 1 结论：写入与 dump 读取解析到同一 (VkBuffer, offset)。fork 排除（候选 1 meta 影子、候选 2 view_src、候选 3 打印伪影全部排除）。**

## 实验 2｜fence 后读回（GAP1-FENCE，200 条全部非零）

write_2d 非 HostVisible 分支的写 fence 退休后，从同一 (dst buffer, offset) 立即读回 64 字节：
- seq 6-10（off=40960，即 output_hc_down 的写位置，5 buffer）：first8=39b63b42 3d003c3b 3c483c75 3cddbd04 b9c1bc45 3c81bb06 3cc53b7d 3d78bb90 —— **非零**
- seq 11+（off=6594560，即 output_hc_up 的写位置）：非零
- 全部 200 条无一条全零。

**实验 2 结论：staging 写入真实到达 device buffer。写入失效假设排除。**

## 实验 3｜dump 侧张量身份（GAP1-DUMPID）

dump 的张量对象指针/data/view_offs 已打印（20 条）；r7 dispatchtrace 的张量身份对照链仍在树上。dump 机制本身有效：同一机制读出 normalized 非零（10240/10240，5 rank），W 读出零不是 dump 伪影。

## 综合裁决：**情形 A——写后覆盖（fork 排除）**

完整证据链（同一 buffer A、同一 offset 40960，output_hc_down 权重）：
1. 写侧 set_tensor → write_2d 写入 (buffer A, off=40960)。
2. FENCE：写后立即读回同一位置——**非零**。
3. r9 phasestrace（本轮复现）：T1_post_all_load readback=2d3bd83b… 非零；**T2_first_graph_compute readback=2d3bd83b… 仍非零**（首次图计算入口时权重在位）。
4. dump（首个 HC 段 dispatch 时）：同一 (buffer, offset) 读出——**全零**（W/lo/mixed 5 rank 全零；normalized 非零证明读取路径有效）。

**覆盖窗口锁定：T2（首次 graph_compute 入口）→ T3（首个 HC 段 dispatch）之间，即首次图执行过程中，某操作清零了权重 buffer。**

## 覆盖者候选（下一轮定位方向）

已静态排除：
- vk 层 init_tensor（22776：no-op，仅 view_src 断言）；
- COUNT_EQUAL memset（16586：dst 是小计数 buffer）；
- prealloc_add_rms_partials memset（21854：独立 prealloc buffer）；
- bst 小 memset（18847/18911/18978/19007：4×int32 计数器）；
- 读链 subbuffer 解析错位（GAP1-R offset 一致）。

剩余候选（按嫌疑排序）：
1. **图执行中 dispatch 的某算子把 dst 写进权重区**——某激活张量的 data 被 meta 层 compute 容器解析到权重 buffer 区间（写侧 fork 而非读侧 fork；r5-r10 的守卫只保护读侧所有权）。GAP1-R 只追踪 BF16 张量，激活（F32）的 subbuffer 解析未追踪——这是盲区。
2. ggml_vk_cache_invalidate_all / buffer 池释放重分配（首次图执行时 cache 初始化路径）。
3. stc_compute 容器 lazy 初始化触碰权重区。

下一轮建议：对权重区间加写入 watchpoint（所有 buffer_write/memset/copy dispatch 路径检查 offset 是否落在 HC 权重区间并打印调用者），或在 graph_compute 内分段 readback 二分定位覆盖节点。

## 请求状态（如实记录）

- request-*.json 未产出（请求未完成）；teardown returncode=-6（SIGABRT，诊断插桩进程的退出路径）；FP-TMISS=0；runner 退出码 0（dump 采集完成即达成本轮目的）。诊断轮数据不进排名。

## 判定表结论

偏差首次出现环节：**权重上传正确（写到达、T2 时点在位）→ 首次图执行过程中被覆盖 → HC 段 dispatch 读零 → lo/mixed 全零 → 乱码**。GAP-1 的最终缺陷形态不是「权重从未到达」而是「到达后被首次图执行中的某操作清零」。
