# h0 vs h1 配对块预注册（Preregistration）

冻结时间：执行前写入，先于任何 GPU 运行。本文件一经写入即冻结，跑分期间不得修改。

## 主指标

- **池化 tok/s**（S 口径）：runner 的 `pooled_tps`，由 measured 相位 3 请求的 completion_tokens/generation_ms 池化得出。
- **generation_ms 均值**：measured 相位 3 请求的 generation_ms 算术均值。

## 样本设计

- 8 个独立配对块 × 2 臂（h0/h1）× 3 个 measured 请求（另有 warmup 2 + guard 1，不参与统计）。
- 每臂一个全新独立进程（runner 自动取 GPU 锁串行）。
- 输出目录：`arms/pair-NN-h0/` 与 `arms/pair-NN-h1/`（NN = 01..08，事先不存在）。

## 轮序

- 奇数块（1,3,5,7）：先 h0 后 h1。
- 偶数块（2,4,6,8）：先 h1 后 h0。
- A/B 交替防顺序偏差。

## 臂定义

两臂除 horizon 外全部参数一致（同 bin、同模型、同 draft、同契约、同 env 集、--threads 22 --ctx 1024 --warmup 2 --repeats 3 --max-tokens 400 --profile none）：

- **h0**：`--horizon 0 --latebind off --mmvq disabled`
- **h1**：`--horizon 1 --latebind off --mmvq disabled`

## 判定规则（预注册，跑分后不得更改）

1. 按块聚合：每臂取池化 tok/s 与 generation_ms 均值。
2. 每块计算 h1/h0 池化 tok/s 比值。
3. 8 个比值取**几何均值**与 **95% t-CI**（t 分布，df=7）。
4. 结论表述：几何均值及 CI 是否包含 1；CI 完全高于 1 视为 h1 显著更快，CI 跨 1 视为未达显著（不宣称等价）。

## 质量门（每臂逐项检查，任何一臂失败则该块作废并保留原始记录，不重跑不删除）

- `length_gate.passed == true`，measured 相位 3 请求全部 n=160 且 `finish_reason=stop`。
- 三次 measured 请求正文 SHA256 全部为 `c2d7bdd3f18f0d9a496087157c8773f26296076c0f979c60bce3ef509effe62c`（与 q2-baseline h0 资格轮逐字一致）。
- measured 相位 fallback 为 0 行。
- runner 退出码 0、server 进程正常退出（returncode 0、非强杀）。

## 二进制与身份

- 二进制：`artifacts/perf-gap/q2-baseline-20261010T0401/measured-bin/llama-server`（冻结快照，binary_SHA256 = `e9aca36509cbd2d6a4d448dc08d49f21907b9d5f37084d4c94929328ae019c6c`，全部 DSO 哈希已登记于该轮 binary-identity.json）。
- 模型：原 Q2_0 两分片（SHA256 与 AGENTS.md §1.1 一致，已在本轮 identity/ 复核）。
- 契约：q2-contract-20261012 v2（question/system 哈希已冻结于 contract-id.json）。
- h0/h1 的全部既有资格（pilot/A-A/b1 单臂）均基于此同一二进制。
