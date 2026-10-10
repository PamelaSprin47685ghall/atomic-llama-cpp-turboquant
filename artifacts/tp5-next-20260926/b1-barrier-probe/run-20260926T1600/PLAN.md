# B1 barrier 成本探针（run-20260926T1600）

## 假设
稳态链 compute_us (~33ms/chain) 内含 ~3497 条 pipelineBarrier。此前 ~2-3us/条是猜测；
真实成本决定 B1（barrier 审计与合并）是否有可回收空间（20-100ms vs ~0）。

## 探针设计（零正确性风险）
`GGML_TP5_BARRIER_PROBE=N`（默认 1 = 原样）：在两个 barrier 发射点
（`ggml_vk_sync_buffers` 全缓冲依赖、`ggml_vk_segment_barrier` 段尾依赖）把同参数
pipelineBarrier 重复发射 N 次。重复 barrier 只过度同步（相同 src/dst/access mask），
语义不变——probe=N 下输出必须与 N=1 逐位等价，这本身就是正确性 sanity。

覆盖范围说明：这两处是稳态链 CB 的主要 barrier 源（其余 8 处 pipelineBarrier 调用点
属于 readback/indirect-update/诊断路径，不在 96-stage 热链内）。若 N=2 使
compute_us 增量 ≈ 3497×c，则单条成本 c 可反推；若增量不可测（<2ms/chain），
则 barrier 不是可回收成分，B1 死刑。

## 测量
- 同一 server、同一 canonical 请求、GGML_TP5_GPU_TIMING=40（稳态 exec）。
- N=1 与 N=2 各一次，比较 [tp5-gpu-timing] compute_us 与逐 stage compute_us 分布。
- 探针本身扰动：额外 barrier 增加 GPU 前端命令量，但这是被测对象本身。
- 判据：Δcompute_us / 3497 = 单条成本；>1.5us/条 → B1 立项；<0.3us/条 → B1 撤销。

## 证据链
- server: /var/tmp/tp5-final-build/bin/llama-server（源 /var/tmp/tp5-candidate-k1a，增量构建）
- 模型: 同完美版 canonical 配置
- 输出: probe1-gpu-timing.log / probe2-gpu-timing.log + canonical 文本比对
