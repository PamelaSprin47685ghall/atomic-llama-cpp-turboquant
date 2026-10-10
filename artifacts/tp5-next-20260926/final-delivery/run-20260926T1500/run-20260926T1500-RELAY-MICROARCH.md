# 继续优化（第二轮）：relay 交接微架构完整解剖（run-20260926T1500）

日期：2026-09-26。目标：继续在完美版之上寻找可验收优化。

## 本轮完成的全链路源码解剖（零新跑测，全部基于已有稳态数据）

### 1. 稳态链的真实结构（修正前一轮两个结论）

- **p1_rec 22-27ms 是首图定义成本**：`[tp5-profile] 96 allreduces` 四行全部来自前 4 个 exec（chain 定义期，
  `meta-chain-miss`×4 → `PREDEFINED TRUTH` 锁定后不再出现）。稳态 chain 100% 复用预定义 CB，
  **不存在**每轮 528ms CPU 重录（前轮恐惧已排除）。
- **P2 普通路径不是 64 线程 relay_copy**：`tp5_relay_copy_f32`（1 WG×64）只是"少数非 HC 前缀边界"的 fallback。
  绝大多数 stage 走 `qwen4_hc_segment_norm` 的 `TP5_HC_RELAY` 变体：**dispatch(2,1,1)×1024 lanes**，
  自旋等 doorbell 后做 RMS/inject/norm，实际工作仅 ~5us——并行度足够，**不是欠并行问题**（已排除）。
- **bcast 内存已是最优配置**：DEVICE_LOCAL+HOST_COHERENT+DEVICE_COHERENT_AMD（cached 优先，
  uncached 仅 fallback）。`GGML_TP5_RELAY_FORCE_UNCACHED`/`FORCE_FLUSH` 均为诊断旋钮。

### 2. 稳态每 stage 周期的完整分解（数据自洽）

| 成分 | 量 | 性质 |
|---|---:|---|
| producer compute | ~344 us | 真实 GPU 工作（MoE 重 ~700 / GDN 轻 ~230 交替） |
| P2 spin（等 CPU publish） | ~140 us | 等待（gpu_spin 稳态 27-67us = BAR doorbell 写延迟，硬件/驱动属性） |
| P2 实际工作 | ~5 us | 2048 线程足够 |
| flag 路径（GPU→host RAM→CPU poll） | ~30 us | PCIe posted write + poll |
| CPU reduce | 10 us | AVX2 |
| publish 路径（CPU→VRAM BAR） | 27-67 us | BAR 写延迟 |
| **CB barrier drain** | **~72-110 us** | **3519 barriers/chain ≈ 36/stage，每条 ~2-3us 流水线排空** |
| 合计 | ~530-630 us/重 stage，~230-280/轻 stage | 96 stage ≈ 51ms span ✓ |

### 3. 新量化出的可归因项：CB barrier drain

- `tp5-linear-definition`：dispatches=4274、**barriers=3519**（每 chain，每 rank）——接近 1:1。
- 每条 `vkCmdPipelineBarrier` 在 compute 队列强制流水线排空（RADV ~2-3us）。
- 96 stage × 36 barriers × ~2-3us ≈ **7-10ms/chain 的排空调度成本**，× 24 chain ≈ **170-250ms/请求**。
- 这是本轮唯一新量化、且不映射到已拒绝家族的成分（不是 LateBind、不是 D1、不是 S1）。

## 裁决

**barrier 削减是一个真实的新工单候选**，但它属于 shader/CB 级改造：
1. 需要逐 barrier 证明可合并/可降级为 execution-only 依赖（不改数学、不改同步语义）；
2. AGENTS §2.2 对此类工单的门槛：先有"瓶颈变化与成本模型"——本轮已补齐成本模型
   （36 barriers/stage、~2-3us/条、7-10ms/chain）；
3. 但实施规模是全链 CB 生成器审计（3519 条逐一分类），超出"最小可验收补丁"范围，
   且每条 barrier 都对应既有同步契约（W1 修复史、GDN/HC/relay 的可见性要求）——
   动一条都可能破坏保活/可见性契约。

**结论：记录为下一阶段唯一有证据的新工单（B1：barrier 审计与合并），
本轮不实施。** 诚实上限：~230ms/请求仅当全部 barrier 可去除——实际不可能，
多数 dispatch 是真实依赖算子，既有 GDN/ATTN/MOE/HC 融合模式已收获容易的部分；
剩余需要新融合模式 + 每个模式的浮点次序资格（§5.3）。其余成分（producer 33ms
真实计算、P2 spin 13.4ms 等待、flag/publish ~60us/stage 物理延迟）均为
架构固有或已拒绝家族。

另：`GGML_TP5_MTP_MAX_CAPACITY=1` 实测 fail（`MTP execution exceeds its
predefined capacity (tokens=2, sequences=2)`），capacity-aware lowering
已知未完成，非可用旋钮。

## 证据

- 源码：`qwen4_hc_segment_norm.comp:429-446`（P2 relay 主循环）、
  `ggml-vulkan-collective.cpp:3960-3978`（relay_copy fallback dispatch(1,1,1)）、
  `:4060-4090`（hc_sum dispatch(2,1,1)）、`:2379-2404`（bcast 内存属性）、
  `:7290-7330`（chain 提交 + 96×star_handoff 循环）
- 稳态数据：`k1b/server-steady85.log`（linear-definition barriers=3519、
  PREDEFINED TRUTH 300+ hits、4 条 collective 行全部位于 chain 定义期）、
  `g1-ledgers`（gpu_spin 27-67us 稳态、ready_all 222-720us、cpu_data ~10us）
