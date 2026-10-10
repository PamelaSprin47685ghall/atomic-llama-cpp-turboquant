# G7：真实 workload（减肥建议）——horizon 排名完全反转，n=1 以 1.65× 赢下正式 ABBA（2026-09-27，run-20260927T0330）

## 用户方向修正

计数题（"1..60"）接受率 1.0 是**可预测 workload**：draft 每步都命中，
horizon 越大摊销越好。用户指出这不真实，要求改用自然问题（减肥建议）且
token 数对齐。G7 全轮按此执行。

## Workload 构建（prompt_n=31，自然 stop，~171 tokens）

- 迭代措辞得到 `请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。`
  ——server tokenize 确认 prompt_n=31（chat 模板后），与计数题完全同形。
- 回答自然 stop 于 174 tokens（8 句结构），不截断。
- **发现：自然文本在 TP5 下非字节稳定**——跨 rank argmax 归约的近 tie 会让
  单个词翻转（观测 171/174 两种长度、词级差异）。计数题永不触发。验证策略
  从单一 ground truth 改为**记录变体集合成员资格**（新变体 fail-closed，
  调查后带证据收录）。n6 产生过一个 178-token 新变体（8/8 句结构、8/8 要点
  命中，draft 截断改变近 tie 选择所致），已收录。
- harness 新增 `--workload weightloss`（`scripts/run-tp5-cross-matrix.py`）：
  prompt_n=31 严格、自然 stop、completion∈[150,200] 带、变体集匹配。
  计数 workload 行为零改动（默认 count）。

## 真实接受率与 horizon 扫描（全部 1-block pilot，2 repeats）

|horizon|mean_len|acceptance|eval_ms（steady 均值）|
|---|---:|---:|---:|
|n=10|2.63|0.163|7056|
|n=8|2.60|0.200|5973|
|n=6|2.68|0.279|5278|
|n=4|—|—|4315|
|n=3|2.45|0.484|3893|
|n=2|2.09|0.547|3691|
|**n=1**|**1.77**|**0.770**|**3233**|

**排名完全反转**：计数题最优 n=10（G6 刚晋级），真实文本最优 n=1。
机制：接受概率集中在 draft 前两步（n10 时 84% 的 draft 行被拒），大 horizon
只增加被拒行的 verify 成本，不增加提交。mean_len 由 draft 质量决定，
horizon 截断只砍尾部概率。

## 正式 ABBA：n6 vs n1（5 blocks × 2 repeats，40/40 全部有效）

- **40/40**：exact 变体匹配、自然 stop、prompt_n=31。
- A(n6) eval mean **5280.1 ms** [5239–5336]；B(n1) **3211.2 ms** [3095–3403]。
- **零重叠**（B max 3403 < A min 5239）。
- decode tok/s 33.71 → **55.49**（+21.78，CI [19.99, 23.56] 全正）；
  ratio **1.646** CI [1.592, 1.700]；committed ratio 1.470 CI [1.431, 1.509]。
- 非劣 0.99 门通过，且为大幅显著正收益。

## 裁决

1. **n=1 是 weightloss workload 的 ABBA 资格成员**。计数 workload 的
   n=10 晋级（G6）不变——两个 workload 各自的最优 horizon 不同，
   这正是 AGENTS §9.2"不同 workload 最优 horizon 不同"的实证。
2. **自适应 horizon（H1 大工单）的真实价值被量化**：静态 horizon 在两个
   workload 间相差 2.2×（7056/3211）。按请求特征（可预测 vs 自然文本）
   选 horizon 的收益远超任何剩余的执行优化。这是下一工单的首选。
3. **不修改默认**：`--spec-draft-n-max` 保持用户显式控制；两个已验证
   变体（n8-varargmax 计数冠军、n1-varargmax 真实文本冠军）均入 harness。
4. 1710 门是计数 workload 的门，G7 不改变其 1886.4ms 现状；G6 的 n10
   晋级（−18.7ms）仍然成立。

## 边界

- 单一自然问题，接受率 0.77@n1 不能外推所有自然文本（代码/数学/翻译
  各有分布）；但"计数题接受率不可外推"已被本 round 双向证实。
- n1 的 mean_len 1.77 意味着 77% 的 cycle 提交 2 token——MTP 仍有效，
  但接近无 MTP 的边界；未测 n0（golden-relay）对照，留作 H1 前置数据。
- 变体验证依赖记录集合；新模型/新 seed/新采样参数需重建变体集。

## 证据

- `weightloss-variants.json`（3 变体）、`weightloss-ground-truth.txt`
- `smoke-weightloss.json`（SMOKE_PASS）
- `run-20260927T0330/pilot-{n10-vs-n6,n6-vs-n8,n6-vs-n4,n4-vs-n2,n2-vs-n1,n1-vs-n3}.json`（扫描曲线）
- `run-20260927T0330/abba5-n6-vs-n1.json`（正式 ABBA + 20 server 日志）
- harness：`--workload {count,weightloss}`、变体 n1/n2/n3/n4/n6/n12-varargmax
