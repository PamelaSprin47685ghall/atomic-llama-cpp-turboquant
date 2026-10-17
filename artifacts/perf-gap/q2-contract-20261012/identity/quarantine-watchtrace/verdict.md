# r13 watch/bisect 裁决（GAP 1 覆盖者定位轮）

运行：arms/sitedump-fused-r13/（env：GGML_VK_CMD_REPLAY=0、GGML_VK_HC_DUMP_SITES=2、GGML_VK_GAP1_WATCH=1、GGML_VK_GAP1_BISECT=1；注：GGML_VK_GAP1_TRACE 与 GGML_VK_GAP1_READBACK 未设——r13 缺 GAP1-R/FENCE 行属 env 配置差异，非代码缺失；r12 已采集同型数据）。
teardown returncode -6（SIGABRT）——与 r12 相同的插桩树 teardown 形态，非新异常。

## 基线形态复现确认

✅ 复现：dump（T3）weightsA-down/up 全 5 rank 全零（0/64）、lo/mixed 全零、normalized 10240/10240 非零——与 r5-r12 形态一致。无观察者效应（bisect 探针未改变 T3 结果）。

## 三重证据

1. **bisect 探针**（blk.0.hc_attn_down，watch 表登记区间 buf=rank0 第一组权重 buffer，off=100597760）：600 次批次提交边界同步读回（ggml_vk_buffer_read）**全部 nonzero=1**（first2=3bd83b2d bb9abb1c，真实 BF16 权重字节）。无 1→0 翻转。
2. **watch HIT 分布**：400 条 COPY-HIT-W2DA + 200 条 DISPATCH-HIT。**MEMSET-HIT=0、SUBBUFFER-HIT=0、COPY-HIT-BC=0**。
   - 400 条 W2DA 全部是加载期合法 fan-out 写入（写入自身权重区间，匿名 actor 无法区分 self——前 25 条证据：output_hc/blk.0 系，每张量 5 条=5 rank；400 条限流被 1920 个 fan-out 写入的前 400 打满）。
   - 200 条 DISPATCH-HIT 全部是 rms_norm_mul_rope_f32_f32 的超大 binding 噪音（binding_off=534783232、binding_range=536084224≈512MB，覆盖全 buffer 与所有权重区间相交）。
3. **cache-invalidate-all：0 次**——候选 (2)（cache invalidate/buffer 池）正式排除。

## fork 排除（写读两侧位置完全一致的直接证明）

output_hc_down rank0 对照（唯一两侧都有四元组打印的张量）：
- 写侧 [GAP1-QUAD] tag=W：ggml_buf=0x6302fea2d2d0 ctx=0x6302fea2d280 dev_buffer=0x6302f8dfa6c0 vkhandle=0x6302f939c1c0；[GAP1-W] off=40960+0+0 size=6553600。
- dump 侧 [GAP1-QUAD] tag=DUMP：ggml_buf=0x6302fea2d2d0 ctx=0x6302fea2d280 dev_buffer=0x6302f8dfa6c0 vkhandle=0x6302f939c1c0；[hc-dump] site2 off=40960 size=6553600。
- **四元组与 offset 完全一致**——写与 dump 读指向同一 VkBuffer 同一位置。vk 层 buffer/offset fork 排除（r12 裁决的「写后覆盖」方向维持，r11 的双 vk_buffer 指针观察系打印语义不对齐的伪差异）。

## 决定性新发现——覆盖是选择性的，不是全局的

- r12 FENCE（写入时刻读回）：output_hc_down off=40960 在 5 个 rank 写入后立即读回**全部非零**（first8=39b63b42 3d003c3b 3c483c75 3cddbd04 b9c1bc45 3c81bb06 3cc53b7d 3d78bb90——真实 BF16 位模式）→ 写入成功到达设备。
- r13 dump（T3）：同一位置读**零**。
- r13 bisect：**同一 rank0 buffer 的另一区间**（off=100597760，blk.0.hc_attn_down）在全部 600 个批次边界（贯穿 T2→T3→执行结束）**保持非零**。
- 结论：同一 VkBuffer 内，off=40960 区间（output_hc）被清零，off=100597760 区间（blk.0.hc_attn）幸存。**r8 轮「384 张量同时清零/全局覆盖」的推断被修正——r8 只观察了 dump 的 2 个 site（blk.47.hc_ffn 与 output_hc），「全局性」是外推不是观察。覆盖是选择性的。**
- 覆盖窗口锁定：加载完成（r12 fence 非零时刻）→ 首个 HC site dispatch（r13 dump 零时刻）。

## 插桩盲区披露（限流设计缺陷）

- COPY-HIT-W2DA 家族限流 400 被加载期 fan-out 打满 → **T2 之后的 write_2d_async 写入行为未监控**。
- DISPATCH-HIT 家族限流 200 被 rms_norm 噪音打满 → **T2 之后 dispatch 写 dst 若触碰权重区间，HIT 不再打印**。
- 因此「三大家族（memset/subbuffer/BC）零命中」只排除这三个家族；候选 (1)（激活 dst 错位写进权重区间）与候选 (3)（stc_compute lazy init 产物被 dispatch 写入）的覆盖动作走 dispatch 写 dst 路径——**恰在盲区，未被排除也未被证实**。charge 预设的绿分支结论「覆盖者在 collective 侧」推断过强：剩余嫌疑 = dispatch 写 dst 家族（候选 1/3）+ collective peer copy/relay，r13 数据无法进一步区分。

## 候选收窄结论

| 候选 | 状态 |
|---|---|
| (1) 激活 dst 写进权重区间 | 保持（盲区，未排除） |
| (2) cache invalidate/buffer 池 | **排除**（invalidate=0） |
| (3) stc_compute lazy init 被 dispatch 写 | 保持（盲区，未排除） |
| collective peer copy/relay | 不能排除（但非唯一剩余） |

## r14 建议（交 Manager 裁定）

1. 修正 watch 限流：W2DA 跳过加载期（首个图计算开始后才计数）或跳过「写入区间=权重自身区间」的 self 写；DISPATCH 过滤超大 binding 噪音（binding_range > 1MB 单独计数）。
2. bisect 探针扩到三个位置：output_hc(40960) + blk.47.hc_ffn(61012480) + blk.0.hc_attn(100597760)——直接抓被清区间的 1→0 翻转节点 + 确认选择性模式（buffer 开头 vs ffn/output 家族）。
3. 或在 T2→T3 窗口对 off=40960 做逐图节点 readback——二分收敛清零节点。

本轮不实施修复：覆盖者指纹未明确捕获（盲区），不满足「指纹明确指向具体错位机理」的自修条件。
