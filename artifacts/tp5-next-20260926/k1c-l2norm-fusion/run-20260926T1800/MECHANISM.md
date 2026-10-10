# K1c 融合候选机制说明（修订版：GDN_SEGMENT 结构性不可用已证实）

日期：2026-09-26。run-id：20260926T1800。

## 关键新发现（推翻"已有 GDN 融合覆盖此段"的假设）

`gdn_segment_pattern`（27 op，SSM_CONV→…→MUL_MAT）与真实 verify 图
**逐节点完全对齐**（stage0 node65-91 已逐位核对），但
`ggml_vk_can_fuse_gdn_segment` 对真实图**结构性返回 false**：

- 真实 GDN：`K=7`（n_rs_seq=n_max=6 的 rollback 快照 bank）、`n_tokens=2`、
  `active_tokens=0`、dst `1280×898`（2 token + 7×128 状态行）。
- multistep 分支要求 `active_param>0 && n_time<=8`：active=0 → 拒绝。
- single-step 分支要求 dst `head_values×129`（K=1、n_time=1 布局）：898≠129 → 拒绝。

**结论：36 个 PLE 层 × 27 op 的整段以逐 op dispatch 运行**——这是
K1b "44 dispatch/层"的完整形态。历史 GDN 融合失败记录（fullrow/K7，
TP5.md §199-201）是**主动扩展守卫后的数值失败**，与本发现（守卫从未
接受 K>1 图）不冲突：那些实验是强行改守卫后失败，本发现是**现状守卫
根本不命中**。

## 候选（修订）：L2_NORM q/k 对合并 —— 最小、bit-exact、不碰 GDN

在 27-op 段内选唯一纯计算、无状态语义的融合点：

```
67 VIEW q_conv-0        (128×4×2, conv_output_silu 的 view)
68 L2_NORM q_conv_predelta-0   ← dispatch A（8 rows × 128）
69 VIEW k_conv-0        (128×4×2, 同源 view，offset 128*4*4B)
70 L2_NORM k_conv_predelta-0   ← dispatch B（8 rows × 128）
```

- 每层 2 次 → 36 层 = **72 次/stage**；5 个同构 stage；draft 链另有
  36 层×1 对（n_tokens=1 形状）。每次 ~5-8 us launch+barrier。
- 保守收益：verify 链 5 stage × 36 层 × 6 us ≈ **1.1 ms/verify**；
  n=8 19 cycles ≈ **20 ms/请求**；加上 draft 链 ~10 ms → 上限 ~30 ms。
- **不碰 GATED_DELTA_NET**（历史数值失败域）、不碰 conv/快照（S1 域）、
  不碰 gating 小算子（SIGMOID/SOFTPLUS 公式融合改变浮点路径的风险
  留给后续候选）。

### bit-exact 论证

合并方案 = 一个 dispatch 处理两个 src view（各 8 行）+ 两个 dst。
每行仍由完整 512 线程 WG 独立归约（行间无交互），行内 i0 网格序与
树形归约与原 kernel 逐一相同 → **输出位级相同**。q/k view 同 buffer
不重叠（offset 0 / offset 2048B），无需拷贝。

### 实现路径（Vulkan dispatcher，仿 SSM_CONV_SILU 先例）

1. pattern：`{VIEW, L2_NORM, VIEW, L2_NORM}`（4 节点，i..i+3），
   outputs = {i+1, i+3}（两个 L2_NORM 结果都是外部消费——GDN 的 src，
   不能省略）。
2. 新 shader `l2_norm_pair.comp`：binding 0=A0(q view)、1=A1(k view)、
   2=D0、3=D1；push constant 复用 `vk_op_unary_push_constants` +
   第二组 ne/nb/misalign 打包进 param2/param3 与保留字段——空间不够
   （128B 上限）。改用专用 push constant struct（两组 unary 参数
   ≈ 2×60B > 128B）→ **必须精简**：q/k 同形（ne/nb 相同），第二组
   只需 misalign offset 差值（1 uint）→ 可行：第一组完整参数 +
   `pair_a_misalign`、`pair_d_misalign` 两个 uint。
3. dispatch sizing：`elements = {nr0 + nr1(=2×nr0), 1, 1}`，row ≥ nr0
   走第二组基址。nr = ggml_nrows(q_view)（8）→ 16 WGs。
4. 执行器 `ggml_vk_l2_norm_pair`：仿 `ggml_vk_ssm_conv` 的
   fused-dst 选择（dst0=nodes[i+1], dst1=nodes[i+3]）。
5. `ggml_vk_op_get_pipeline`：`GGML_OP_L2_NORM && num_additional_fused_ops==3`
   → `pipeline_l2_norm_pair_f32`。
6. write mask：bit1|bit3（两个 L2_NORM 节点都写内存）。
   `op_srcs_fused_elementwise`：VIEW 是零成本空 op，全 false 安全。
7. 守卫（`ggml_vk_can_fuse_l2_norm_pair`）：
   - 两 L2_NORM 的 eps（op_params[0]）位级相同；
   - 两 src 同 type=F32、同 ne、同 nb、view_src 同一 tensor、
     `get_misalign_bytes` 均 0（或可表示）；
   - 两 dst 同 type/同形/连续；
   - `ggml_can_fuse_subgraph(cgraph, i, pattern, {i+1, i+3})`
     （VIEW 被外部 src 引用没关系——outputs 只列 L2_NORM）。
     注：VIEW 的 use_count 包含两个 L2_NORM（都在子图内）→ 可省略。

### 风险与退出

1. push constant 128B 上限 → 用"同形假设"压缩参数；若仍超限，
   退化为两 dispatch 共享一次 barrier（收益减半）或放弃。
2. `ggml_can_fuse_subgraph` 对 VIEW 节点的 use_count 检查：q_conv view
   的唯一消费者是其 L2_NORM（在子图内）→ 通过；若 conv_output_silu
   另有消费者（v_conv view 是独立 VIEW 节点，不是 q view 的消费者）→
   不影响。**实测确认**。
3. 误命中非 PLE 路径：守卫要求两 src view 同源同形 → attention 的
   q/k l2_norm（Qwen attention 无 l2_norm；其他模型若有同构对也会
   正确合并——数学等价，无需排除）。
4. ABBA 未检出（CI 含 1）→ 非劣即保留（-72 dispatch/stage 的维护
   价值）；预注册，不反复抽样。

### 验收

1. 同形数值对拍：Vulkan0 单测（复用 test-vulkan-command-replay 的
   PLE 图样板），fused vs unfused F32 位级全等。
2. canonical 171/171 + 默认路径（MTP-off 不含此 pattern，天然不命中）。
3. 5-block ABBA（A=完美版 B=+pair）。
