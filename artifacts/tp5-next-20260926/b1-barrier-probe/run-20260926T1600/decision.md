# B1 barrier 成本探针裁决（run-20260926T1600）

日期：2026-09-26。工单：B1（barrier 审计与合并）——上轮唯一新量化的候选。

## 方法

`GGML_TP5_BARRIER_PROBE=N`（默认 1=原样）：在两个 barrier 发射点
（`ggml_vk_sync_buffers` 全缓冲依赖、`ggml_vk_segment_barrier` 段尾依赖）将同参数
`vkCmdPipelineBarrier` 重复发射 N 次。重复 barrier 仅过度同步（相同 src/dst/access
mask），语义不变。probe=2 使稳态链 barrier 数 3497 → 6610（+3113）。
probe=1 完整复现历史 barrier 计数（3485/3497/3519/4239），设置有效性已证。

## 结果

|指标|probe=1|probe=2|差值|
|---|---:|---:|---:|
|compute_us 均值（5 rank，稳态链）|31928.8|31791.6|**-137.2 us（噪声内）**|
|span_us 均值|50846.3|50774.0|-72.3 us|
|端到端 171-token eval 均值（n=45）|2067.23 ms|2072.37 ms|+5.1 ms（sd≈56，噪声）|
|canonical 正确性|45/45 逐字精确|45/45 逐字精确|过度同步位安全，符合预期|

**单条 barrier 成本上界：噪声带宽 ~±300us/chain ÷ 3113 条 ≈ <0.1 us/条。**

## 裁决

1. **B1 关闭（不可回收）**。RADV/RDNA2 上预录 CB 内的 pipelineBarrier 是前端
   元数据，不触发流水线排空；3497 条 × <0.1us < 0.35ms/chain，全请求 <8.4ms。
   上轮"~2-3us/条 → 170-250ms/请求"的假设被实验杀死。
2. 稳态 stage 周期的未归因残差（~80-110us/stage）重新归因为 **dispatch 前端
   发射开销**（44 launches/stage × ~1-2us）+ 算子固有延迟——属 dispatch 粒度/
   融合家族（既有 GDN/ATTN/MOE/HC 融合已收获容易部分），不是 barrier。
3. 至此 356ms 缺口的全部成分均有实证归因，且均映射到已拒绝/已关闭家族：
   - relay 交接串行化 ~340ms（含链头填充 ~113ms）→ LateBind/direct 家族（§2.2 拒绝）
   - draft ~320ms → D1（架构互斥已证）
   - barrier <8.4ms → B1 本轮关闭
   - snapshot ~24ms → S1（<噪声，已降级）
   - producer 33ms 真实计算 → K1 已收割，剩余需新融合模式+浮点资格（大工单）

## 探针旋钮处置

`GGML_TP5_BARRIER_PROBE` 为诊断旋钮，默认 1 与原 CB 逐命令等价
（probe=1 barrier 计数与历史完全一致）。保留在候选工作树中作为 B1 死因证据；
不进入默认路径，无行为影响。

## 证据

- `probe1-server.log` / `probe2-server.log`：45/45 exact、GPU timing、
  linear-definition barrier 计数（3497→6610）
- 源：`/var/tmp/tp5-candidate-k1a`（探针补丁仅 2 处 for 循环包裹）
- 构建：`/var/tmp/tp5-final-build`（增量，DSO 含探针字符串已验证）
