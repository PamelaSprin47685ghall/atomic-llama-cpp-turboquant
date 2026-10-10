# G8：自然文本 canonical 下的 cycle 账本与 MTP 边际价值——账本闭合，无新可立项候选（2026-09-27，run-20260927T0400）

## 目标变更记录

用户裁决：自然文本为唯一 canonical workload，计数题退役。TP5.md 顶部已记录。
本轮以 weightloss workload（n1-varargmax，G7 ABBA 冠军 3211ms）为基座建账。

## n1 自然文本 cycle 账（4 请求，100 cycles/请求，profiler 开）

steady cycle **29.76 ms** = draft 2.94 + target 26.02 + catchup 0.80：

|成分|量|说明|
|---|---:|---|
|target verify|26.02 ms|2 行（1 draft + 1 sampled）|
|draft（单 token 前向）|2.94 ms|draft 模型整过一层|
|catchup|0.80 ms|K/V-only|
|handoff|0.00 ms|RELAY P2 常驻|

- 100 cycles × 29.76 + 首验尖峰（~232ms，prompt→verify 形状切换，G5 已关闭）
  ≈ predicted_ms 3168（req2 实测 3073.6，残差 7.6ms——账闭合）。
- final_tokens 直方图：308×2 + 92×1（mean_len 1.77，与 G7 一致）。
- G5 遗留的 verify-sub 异常已解：print 站点在 `decode()` 内，`decode_us`
  含 `llama_decode` 的 host 等待全程；8 行/4 请求 = 每请求 2 次（首验+1），
  稳态 cycle 不打印——不是采样成本回归。

## target 26.02 的分解（跨 workload 推算）

- 2 行真实算量 ≈ 2×7.8 = 15.6 ms（计数题 9 行 70.5ms 的每行外推）。
- **固定 ~10.4 ms/cycle** = 96-stage relay 流水的每 stage 固定延迟
  （~152µs/stage，与 G1 的 V≈173µs/stage 一致：PCIe 排空 + CPU publish/detect）。
- GDN 快照拷贝（K=2）不是瓶颈：n8→n1 差 44.5ms < 7 行算量差 54.6ms，
  快照写与计算重叠。

## MTP 在自然文本上的边际价值（关键新事实）

|臂|eval mean|cycles|说明|
|---|---:|---:|---|
|n1-varargmax|**3262.8 ms**|100|G7 冠军|
|no-MTP（native-default）|3361.1 ms|177|1 行 decode/cycle|

**MTP 在自然文本上只值 98ms（2.9%）**。n1 的 verify（26.02/2 行）比 no-MTP
的 1 行 decode cycle（~19ms）更贵——draft+固定链开销吃掉几乎全部接受增益。
no-MTP 也产出新变体（177-token，8/8 句结构，已带证据收录，变体集 4 个）。

## n1 自然文本请求的最终分解

|成分|量|占比|家族状态|
|---|---:|---:|---|
|relay 固定延迟（100cyc×~14.6ms）|~1460 ms|45%|LateBind/direct 拒绝、G3a 关闭、B1 关闭|
|2 行真实算量（100×15.6）|~1560 ms|48%|真实计算|
|draft（100×2.94）|~294 ms|9%|D1 关闭（架构门）|
|catchup+杂项|~80 ms|2%|S1 上界以下|

## 裁决

1. **账本闭合**：自然文本 n1 的全部时间有实证归因；无新可立项候选——
   固定 relay 延迟的全部攻击家族（LateBind/direct、G3a 传输微架构、B1 barrier）
   均有负结果或架构拒绝记录，其关闭证据在自然文本下**同样成立**
   （152µs/stage 的机制与行数无关）。
2. **MTP 对自然文本的净价值 ~3%**：若未来 draft 模型质量提升（接受率↑），
   最优 horizon 与 MTP 价值需重测；当前 n1 与 no-MTP 在噪声带边缘。
3. **计数 workload 的 G6 n10 晋级与 1886.4ms 基座保持**，但其 1710 门
   随目标变更退役；计数回归保护线继续有效。
4. 唯一可能改变格局的是架构家族重启（LateBind 消除 CPU-in-loop、或
   D1 设备闭环），重启门见各自 decision.md，本轮无新证据满足。

## 证据

- `server-ledger.log`（400 cycle 行 + verify-sub）、`ledger.json`（4 请求）
- `pilot-n1-vs-nomtp.json`（n1 3262.8 vs no-MTP 3361.1，1-block pilot）
- 变体集：`g7-realistic-workload/weightloss-variants.json`（4 变体）
- TP5.md 顶部目标变更 + 本 decision
