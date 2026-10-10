# K1c 裁决：L2_NORM q/k 对融合——非劣通过，未检出独立提速，按预注册保留

日期：2026-09-26。run-id：20260926T1800。工单：AGENTS §5.3 算子融合（"剩余
需要新融合模式 + 浮点次序资格"）。

## 改了什么

PLE 每层 conv_silu 后的 q/k 两个 L2_NORM（同源同形 view）合并为单次 dispatch：

- 新 shader `l2_norm_pair.comp`：4 binding（A/D/A_pair/D_pair），行号
  `[0,p.ne)` 走组 0、`[p.ne,2p.ne)` 走组 1；每组内归约顺序与 `l2_norm.comp`
  完全一致（同 BLOCK_SIZE 网格 + 同树形）→ **bit-exact**。第二组基址经
  param2/param3 位传（floatBitsToUint 解码）。
- 融合守卫 `ggml_vk_can_fuse_l2_norm_pair`（两种节点序均接受：
  `L2_NORM,VIEW,L2_NORM` 与 `L2_NORM,L2_NORM`；同 eps 位级、同 view_src、
  同形同步长、misalign ≤16bit）。**不用** `ggml_can_fuse_subgraph`——其
  view_src 内部性规则对"源 view 在区域外"的 PLE 图结构性拒绝。
- 执行器 `ggml_vk_l2_norm_pair`：仿 SSM_CONV_SILU 先例，直接 4-buffer
  dispatch；`GGML_VK_DISABLE_L2_PAIR` 为动态退选开关。
- harness 新增 `native-mtp-fused-no-l2pair` 控制臂（同二进制 env 退选）。

## 实际命中

- 合成图：`--l2-pair-only` 3 形状（rows 8/6/3）融合臂 vs 未融合臂
  **逐字节相等**；`--gdn-only` 6 次命中且 GDN 状态输出字节级通过；
  全套 `test-vulkan-command-replay` **27/27**。
- 真实五卡模型：`GGML_VK_LOG_L2_PAIR=1` 单请求 8+ 次命中
  （rows=8 verify / rows=12 catch-up 形态），171/171 精确自然停止。
- 链 dispatch：verify 链 **4274→4196**（-78）、barriers **3519→3485**。

## ABBA 裁决（A=未融合控制臂, B=融合默认臂，同完美版二进制）

|指标|A: 未融合|B: 融合|配对统计|
|---|---:|---:|---|
|decode tok/s|83.256|83.194|ratio 0.9993 CI [0.9902, 1.0083]，**非劣通过**|
|committed tok/s|50.877|50.954|ratio 1.0016 CI [0.9860, 1.0171]，非劣**未**过（CI 下界 <0.99）|
|predicted_ms|2053.98|2055.58|+1.6 ms（噪声带内）|
|正确性|20/20 精确|20/20 精确|40/40，5/5 blocks|

**预注册解读**（MECHANISM.md）：非劣即保留、正收益才宣传。decode 非劣
通过；committed 非劣差 0.4%（CI 下界 0.986 vs 0.99 门）——按预注册的保守
读法，这是"未检出独立提速"而非劣证据在两指标间不一致。融合保留为默认
成员（-78 dispatch/-34 barriers 有维护价值、bit-exact 已证、无回退路径
损失），**不宣传提速**；`GGML_VK_DISABLE_L2_PAIR=1` 提供显式退选。

## 为什么没有提速（诚实归因）

-78 dispatch/链 ÷ 96 stage ≈ 每 stage 不到 1 个 launch 节省；按 B1 已证
的 <0.1 us/barrier 和 ~5 us/dispatch 前端成本，理论上限 ~0.4 ms/链 ×
~25 链 ≈ 10 ms/请求——在 ±15 ms 会话噪声内不可检出。K1b 的"44 dispatch/
层"账里，l2 对只占 2/44；大头（gating 小算子、per-slot CPY）分别在
SIGMOID/SOFTPLUS 公式风险域与 S1 快照域，本轮未触碰。

## 边界与回退

- MTP-off、其他模型：无 q/k l2 对 pattern 天然不命中；守卫要求同源同形
  同 eps，误命中即数学等价。
- `GGML_VK_DISABLE_L2_PAIR=1` 退选后行为与完美版基线一致（A 臂即证明）。
- 未 commit/push；候选树 `/var/tmp/tp5-candidate-k1a`，构建
  `/var/tmp/tp5-final-build`（server SHA 与基线相同源+本轮增量）。

## 对 100 tok/s 门

无变化（88.1 tok/s 为 n=8 臂成绩，本融合在其上叠加 ~0）。剩余缺口仍映射
已关闭家族；下一个有证据的候选是 gating 链 SIGMOID/SOFTPLUS 公式融合
（需逐算子浮点资格，收益上限 ~25 ms/请求）——是否开工由用户决定。
