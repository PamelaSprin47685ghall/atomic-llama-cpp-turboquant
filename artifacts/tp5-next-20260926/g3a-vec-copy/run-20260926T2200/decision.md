# G3a：RELAY P1 host-push 传输微架构——负结果关闭（2026-09-26，run-20260926T2200）

## 改了什么

针对 G2 账本中 P2 spin 24.2 ms/链（96 stage × 254 µs，链 span 的 37%）的归因假设
"flag 可见性 V ≈ 173 µs/stage 由 TRANSFER 拷贝+双屏障引入"，在 fallback RELAY P1
记录路径（`tp5_relay_submit` P1 record，ggml-vulkan-collective.cpp ~:3128）做了
两个单因素 opt-in 候选：

1. **cap**：`GGML_TP5_RELAY_VEC_COPY_MAX`（默认 16384 = 原行为）。调高后 92 KB
   verify payload 从 `vkCmdCopyBuffer`（TRANSFER DMA + mb_host 界障）切到
   `tp5_copy_u128` compute-shader 多 WG 拷贝（grid-stride，ceil(n_uvec4/64) WG）。
2. **fold**：`GGML_TP5_RELAY_VEC_COPY_FOLD`。新增 `tp5_copy_u128_fold` 管线
   （同 shader + `TP5_COPY_FOLD_READY` 定义；binding 2 = 同一 host import 内
   `star_rank_stride-64` 处的 ready word），单 WG 拷贝后在 shader 内发布 ready
   word，替换 `vkCmdFillBuffer` + mb_ready 界障对。flag 偏移与 CPU 轮询位置
   （`star_host + (i+1)*rank_stride - 64`）逐位对齐已验证。

数值路径：两种变体都是字节相同拷贝，无数学顺序改变。

## 实测结果（真机 5×RX 6800，canonical payload，n=8，每臂 1 warmup + 3 测量）

server 侧 `eval time`（171 tok）配对：

|臂|配置|mean|配对差|
|---|---|---:|---|
|A（pilot 1）|默认|1944.68 ms|—|
|B（pilot 1）|cap=1M|1933.52 ms|−11.2 ms（−0.57%，含 +4.6 离散对）|
|A2（pilot 2）|默认|1987.55 ms|—|
|C（pilot 2）|cap=1M+fold|2042.72 ms|**+55.2 ms（+82.0/+6.2/+77.3）**|

- 全部 24 请求 171/171 精确、finish=stop。
- 会话漂移：A 与 A2 相隔约 40 分钟，均值差 +42.9 ms（2.2%）→ 噪声带 ±20–40 ms，
  B 的 −11.2 ms 不具资格。
- fold 明确回归（3 对中 2 对 +77~82 ms）：单 WG 把 92 KB 串行推过 1 个 CU 的
  uncached GTT 存储，MLP 不足，比 DMA 引擎慢 ~30 µs/stage；省下的 FILL+界障
  （~10–20 µs）被完全吞掉。

## 裁决

- **fold：拒绝**。机制（省一次 FILL+barrier）真实但量级太小，单 WG 序列化代价更大。
  多 WG fold 需要 cross-WG 完成计数器，为已证 ≤20 µs 的收益加机制，不值。
- **cap-only：无显著收益**（−11 ms 在噪声带内）。P2 spin 的主导项是 92 KB payload
  的 PCIe 排空 + CPU reduce/publish 延迟，界障开销不是大头——G2 的 V≈173 µs
  归因修正为"PCIe 排空 + 检测/发布"，非"TRANSFER 界障"。
- **G3a 家族（RELAY P1 host-push 传输微架构）关闭**。两个 env 保留为 opt-in 诊断
  旋钮，默认行为与基线逐位一致（cap 默认 16384、fold 默认关）。不注册 harness
  variant，不做 ABBA（pilot 未过关）。

## 对 100 tok/s 缺口的意义

n=8 干净 1940.8 ms 中 P2 spin 24.2 ms/链 × 19 = 458 ms 是最大非计算项，但本轮
证明其在现有 RELAY 架构（CPU-in-loop + host-import GTT）内不可由传输微优化回收；
彻底消除 = LateBind/direct 家族（§2.2 已拒绝）。剩余可回收项全部映射已关闭家族或
workload 固定项（见 G2-N8-LEDGERS.md 最终分解）。

## 证据

- `g3a-pilot.py`、`pilot.json`（A/B）、`pilot-fold.json`（C/A2）、`server-{A,B,C,A2}.log`
  （含全部 eval time 行与 acceptance 1.00000 记录）
- 候选 diff：`ggml-vulkan-collective.cpp`（env cap + scaled dispatch + fold 管线/
  记录/teardown）、`vulkan-shaders/tp5_copy_u128.comp`（FOLD_READY 变体）、
  `vulkan-shaders-gen.cpp`（注册 `tp5_copy_u128_fold`）
