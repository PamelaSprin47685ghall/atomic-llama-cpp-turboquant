# h1 vs h2 配对块预注册（Preregistration）

冻结时间：h2 pilot 资格通过后、任何配对块 GPU 运行前写入。一经写入即冻结。

## 主指标

- **池化 tok/s**（S 口径）：runner 的 `pooled_tps`（measured 3 请求 completion_tokens/generation_ms 池化）。
- **generation_ms 均值**：measured 3 请求算术均值。

## 样本设计

- 8 个独立配对块 × 2 臂（h1/h2）× 3 个 measured 请求（warmup 2 + guard 1 不参与统计）。
- 每臂全新独立进程（runner 自动 GPU 锁串行）。
- 输出目录：`arms/pair-NN-h1/` 与 `arms/pair-NN-h2/`（NN=01..08，事先不存在；注意 pair-NN-h1 与 h0/h1 轮的目录名不冲突——本轮使用 h2 专属块目录前缀 pair-NN，h1 臂目录为 pair-NN-h1）。

## 轮序

- 奇数块（1,3,5,7）：先 h1 后 h2。
- 偶数块（2,4,6,8）：先 h2 后 h1。

## 臂定义

两臂除 horizon 外全部参数一致（同 bin=冻结 measured-bin、同模型、同 draft、同 v2 契约、同 env 集、--threads 22 --ctx 1024 --warmup 2 --repeats 3 --max-tokens 400 --profile none）：

- **h1**：`--horizon 1 --latebind off --mmvq disabled`
- **h2**：`--horizon 2 --latebind off --mmvq disabled`

对照基准 = h1（当前最强已晋级臂，V5-V9 全过）。

## 判定规则（预注册，跑分后不得更改）

1. 按块聚合：每臂取池化 tok/s 与 generation_ms 均值。
2. 每块计算 h2/h1 池化 tok/s 比值。
3. 8 个比值取**几何均值**与 **95% t-CI**（t 分布，df=7，t(0.975,7)=2.3646）。
4. E6 判读：**CI 下界 > 1 → h2 显著更快**（成为新候选，V8/V9 待后续）；**CI 跨 1 或上界 < 1 → 暂停加深**（ΔC/ΔK 不利），h1 保持最强臂，不测 h3。

## 质量门（每臂，任一失败该块作废并保留原始记录，不重跑不删除）

- measured 3 请求全部 n=160 且 finish_reason=stop。
- 三次正文 SHA256 全部为 `c2d7bdd3f18f0d9a496087157c8773f26296076c0f979c60bce3ef509effe62c`（与 h0/h1 逐字一致——greedy 分布保持）。
- measured 相位 fallback 为 0 行。
- runner 退出码 0、server 正常退出（returncode 0、非强杀）。

## 二进制与身份

- 二进制：`artifacts/perf-gap/q2-baseline-20261010T0401/measured-bin/`（冻结，binary_SHA256=e9aca365…，与 h0/h1 全部资格轮同一二进制）。
- 模型/契约：与 h1 资格链完全相同（identity/ 已登记）。
