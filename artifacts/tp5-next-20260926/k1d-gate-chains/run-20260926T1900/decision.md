# K1d 裁决：PLE 路由门链融合——双指标非劣通过，按预注册保留

日期：2026-09-26。run-id：20260926T1900。工单：AGENTS §5.3（K1c 之后的
下一个纯计算、状态无关、可保位序的融合点）。

## 改了什么

`build_ple` 的路由门链（每 PLE 层 1 条，9 个 dispatch，全部 ne≤2560）合并为
单 dispatch：

```
MUL(key,query) → SUM_ROWS → SCALE(1/√d) → SGN → ABS → CLAMP(1e-6,∞) → SQRT → MUL → SIGMOID
```

- 新 shader `ple_router_gate.comp`：每 WG 一行；乘积用 **`precise`** 禁止
  FMA 收缩（乘法必须先舍入到 f32 再累加——与未融合链 MUL 落显存后
  SUM_ROWS 读回完全一致）；树形和照抄 sum_rows.comp（同 BLOCK_SIZE=
  subgroup_size spec constant、同跨步、同树）；标量链公式逐字复制
  unary.comp/scale.comp。
- 守卫 `ggml_vk_can_fuse_ple_router_gate`：9 节点模式+边检查（**ABS 消费
  SCALE 而非 SGN**——真实图 DFS 序为 MUL,SUM_ROWS,SCALE,SGN,ABS,CLAMP,
  SQRT,MUL,SIGMOID，与 K1b segments dump 逐节点一致）；span 内多消费
  合法（SCALE 被 SGN+ABS 双消费，use-count 按 span 内扫描计数）；
  CLAMP max 允许 INFINITY；`GGML_VK_DISABLE_PLE_GATE` 动态退选。
- 执行器 `ggml_vk_ple_router_gate`：专用 push constants（行数/列数/三
  基偏移/scale/bias/clamp min/max）；dispatch 挂在 **GGML_OP_MUL case**
  （链首 op），不是 SUM_ROWS——融合框架按链首节点进入 case。

## 调试过程中修掉的 4 个真 bug

1. 谓词边表初版写错（ABS→SGN）；用 K1b dump 逐节点核对修正。
2. 执行器基址偏移：SUM_ROWS case 的 node_idx 是链中第 2 节点，链首在
   -1；改挂 MUL case 后基址即 node_idx。
3. dispatch 位置：初版挂在 SUM_ROWS case → 融合匹配后 MUL case 落入
   snake_dispatch_fused 崩溃；移到 MUL case 的 hc_fold4 分支之后。
4. **FMA 收缩**：合成对拍 row0 差 1 ulp——驱动把 x*y 收缩进累加；加
   `precise float prod = x*y` 后逐位一致。

## 附带发现（独立于本工单，已记录未修）

全 suite 下 **未融合路径** 曾读出脏数据（2560×4 输入 rows 2-3 与 CPU 参考
不符，两臂同错）——replay 缓存与先前测试的形状/缓冲碰撞。改用 2549（质数
列数）后消失。这是 replay-cache 的既有风险面，不是本融合引入；单列工单
待立。

## 实际命中（真实五卡模型，dispatch 计数为证）

|stage 类型|K1c 基线|K1d|Δ|
|---|---:|---:|---:|
|verify(55)|4196|4188|**-8**|
|draft(60)|4206|4198|**-8**|
|catch-up(5)|4238|4230|**-8**|
|prefill(55)|4613|4605|**-8**|

每 stage 恰 1 条路由链、-8 dispatch、-8 barrier，全部 stage 类型、全 rank
命中。`GGML_VK_LOG_PLE_GATE` 确认 rows=8（verify）/rows=108（prefill）。

## 验证

- `--ple-gate-only`：3 形状（2549×4/128×1/3571×3）融合臂 vs env 退选臂
  **逐字节相等**（precise 修复后）。
- 全套 command-replay **28/28**（含 GDN 状态保护线 `--gdn-only` 通过）。
- CPU 回归 5/5（sampling/arg-parser/mtp-workspace/tp5-plan/target-capacity）。
- 真实五卡 smoke：**12/12 精确 171/171 自然停止**。

## ABBA 裁决（A=env 退选, B=默认融合, 同二进制, 5 blocks × 2 repeats）

|指标|A|B|配对统计|
|---|---:|---:|---|
|decode tok/s|83.039|83.060|ratio **1.0003** CI [0.9967, 1.0038]，非劣**通过**|
|committed tok/s|50.765|50.750|ratio 0.9997 CI [0.9929, 1.0065]，非劣**通过**|
|predicted_ms|2059.33|2058.81|-0.5 ms（噪声带内）|
|正确性|20/20|20/20|40/40，5/5 blocks|

**预注册解读**（MECHANISM.md）：非劣即保留、CI 整体>1 才宣传提速。双指标
非劣通过 → **保留为默认成员**；CI 含 1 → 不宣传提速。与 K1c 一致：每
stage -8 dispatch 的维护价值成立，端到端收益在 ±10 ms 噪声内不可检出。

## 边界与回退

- 链 B（GDN alpha/beta 门控 ADD+SOFTPLUS+MUL+SIGMOID，ne=10/1）**未做**：
  K1b 账里每层仅 4 个 tiny dispatch，且 alpha/beta 两链间隔 3 节点需跨
  RESHAPE 匹配；链 A 已证明该家族收益不可检出，链 B 预期更小，按
  MECHANISM 预注册的"链 A 先行、链 B 视成本"规则关闭。
- MoE CLAMP/DIV 与 hc 侧 SCALE-SIGMOID-SCALE：0.75 元素 dispatch，转发
  READ 语义改写风险大于收益，未触碰。
- `GGML_VK_DISABLE_PLE_GATE=1` 退选后行为与基线一致（A 臂即证明）。
- 未 commit/push；候选树 `/var/tmp/tp5-candidate-k1a`，构建
  `/var/tmp/tp5-final-build`。

## 对 100 tok/s 门

无变化（~0 ms 检出）。剩余缺口仍映射已关闭家族；下一个有证据候选需新
瓶颈证据（G1 复测或新融合模式），无证据即停。
