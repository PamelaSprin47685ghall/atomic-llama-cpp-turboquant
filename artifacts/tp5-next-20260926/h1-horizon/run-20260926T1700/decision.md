# H1 horizon 裁决（run-20260926T1700）

日期：2026-09-26。工单：AGENTS §9.2 H1。**结果：n=8 通过完整 ABBA 资格，
成为首个有配对统计支撑的真实提速成员（+6.2% decode tok/s）；n=10 被自有
ABBA 拒绝。**

## 证据链（全部同完美版二进制 /var/tmp/tp5-final-build，SHA b92fda129…）

### ABBA #1：A=n=6 vs B=n=8（5 blocks × 2 repeats，40 请求，全部逐字精确 171/171）

|指标|A: n=6|B: n=8|配对统计|
|---|---:|---:|---|
|decode tok/s|82.96|**88.11**|ratio **1.0620** CI **[1.0579, 1.0662]**（整体 >1）|
|committed tok/s|50.76|52.46|+1.70 CI [+1.43, +1.96]，5/5 块 B>A|
|predicted_ms 均值|2061.26|**1940.84**|**-120.4 ms**|
|非劣 0.99 门|—|通过|且为显著正收益|

机制：接受率 1.00000（152/152）、mean len 9.00；19 cycles × 102.7ms vs 25 × 82.7ms；
**verify 边际 token 成本 10.0ms < 平均 11.8ms**——融合 96-stage 链的固定 relay
交接开销被更多 token 摊销。历史 n=7/n=10 负结果（前融合时代）确认被推翻。

### ABBA #2：A=n=8 vs B=n=10（5 blocks × 40 请求，全部精确）

|指标|A: n=8|B: n=10|配对统计|
|---|---:|---:|---|
|committed tok/s|52.76|52.30|diff **-0.46** CI **[-0.91, -0.01]**（整体 <0）|
|decode tok/s|88.47|87.86|ratio 0.9931 CI [0.9852, 1.0010]|
|非劣 0.99 门|—|**双指标均失败**|n=10 拒绝|

n=10 烟测优势（-11ms）未在配对中存活——两次 ABBA 的 n=8 臂分别为 1940.8/
1933.0ms，证实 ~±8ms 会话级漂移；单次烟测不能给候选排名（§10.3 纪律的又一实证）。

## 裁决

1. **`--spec-draft-n-max 8` 为 ABBA 资格成员**（variant `native-mtp-fused-n8`）。
   默认 n=6 不动；晋级默认须用户授权。
2. **n=10 拒绝**；n≥12 由周期模型（n=10 实测点重拟合后 n=12 预测 1944.6ms）
   与 n=10 的实测劣化共同关闭，不再扫描。
3. H1"自适应"策略（按前缀存活分布动态选 horizon）保留为未开工大工单；
   本轮交付其前置数据：计数题接受率 ~1.0 下的静态最优 n=8。
4. 边界重申（§9.2/§11.1）：计数题接受率不可外推到真实文本；低接受率 workload
   的最优 horizon 更小；171=9×19 整除对 n=8 有利，不同输出长度最优值不同。

## 对 100 tok/s 门的影响

n=8 后 canonical decode 88.1 tok/s / 1940.8ms，仍差 1710ms 门 **230.8ms**。
剩余成分全部映射已拒绝/已关闭家族（relay 交接 ~340ms→LateBind/direct、draft
~320ms→D1 互斥、barrier <8.4ms→B1 关闭、snapshot ~24ms→S1 <噪声）。n=8 的
+120ms 是摊销效应，不改变家族裁决。

## 归档

- `abba/abba5.json`（n6 vs n8）、`abba/abba5-n10.json`（n8 vs n10）、两组
  `abba5*_logs/`（20 server 日志，含 mean len/acceptance 机制证据）
- 烟测：`n8-server.log`、`n10-server.log`、`n6-control-server.log`
- harness 变更：`run-tp5-cross-matrix.py` 新增 `native-mtp-fused-n8`/`n10` 变体
