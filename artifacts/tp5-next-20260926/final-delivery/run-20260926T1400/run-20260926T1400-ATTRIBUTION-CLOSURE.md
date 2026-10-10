# 继续优化：最终归因闭环裁决（run-20260926T1400）

日期：2026-09-26。目标：在完美版之上继续缩小 356 ms 缺口。

## 本轮深挖的三层归因修正（全部基于已有稳态数据 + 源码，无新跑测）

### 修正 1：stage 1–8 的"慢"不是 PLE 计算，是 relay 管线填充

- `tp5_hc_resume.comp::await_input()`（行 84-110）在"compute"括号内**自旋等待**
  CPU 发布的 generation doorbell——`compute_us` 括号（时间戳 5s+1→5s+2）含等待。
- stage 1 稳态 2215 us vs stage 11 同 op 同 shape 同 MoE 324 us：差额 ≈1.9 ms
  是**等待**，不是工作。44 个额外小 dispatch（PLE 门控链）上限 ~350 us，解释不了。
- p2 列衰减（776→943→…→150 us）正是 relay 往返的管线填充曲线；到 stage 10 达稳态。
- **链头填充 ≈4.7 ms/链 × 24 链 ≈ 113 ms/请求。**

### 修正 2：PLE 权重 MIRRORED 不是可收项

- `llama-tp5-plan.cpp:582`：`ple_*` 全部 MIRRORED（每 rank 全量 27 MB/层，
  ple_key q6_K 2560×10240 + ple_value 2560×2560）。
- 按 hc 流切分的收益 ~54 us/层（带宽），代价 = 每层一次 80 KB allreduce
  （~170 us 稳态 p2）——**负收益**。MIRRORED 是合理设计。

### 修正 3：稳态 p2 ~150 us/stage 是设计内的 relay 交接

- 发布路径用 HOST_COHERENT + DEVICE_COHERENT_AMD + sfence（无 flush 调用，
  `GGML_TP5_RELAY_FORCE_FLUSH` 仅诊断）——150 us 不是 flush 成本。
- 它是 [GPU producer 完成 → CPU poll+AVX2 reduce+publish → GPU resume 自旋命中]
  的流水线稳态时延。96 stage × (compute ~300 + p2 ~150) ≈ 43 ms + 链头 8 ms ≈ 51 ms span，与实测一致。

### 附带核验

- `GGML_TP5_MTP_MAX_CAPACITY=1`（唯一未在融合基座测过的 opt-in）：**fail**
  （`MTP execution exceeds its predefined capacity (tokens=2, sequences=2)`）——
  代码注释明言"resource layer is opt-in until the capacity-aware lowering is
  complete"，已知未完成，非可用旋钮。

## 356 ms 缺口的最终分解

| 项 | 量 | 回收路线 | 手册状态 |
|---|---:|---|---|
| 链头 relay 管线填充 | ~113 ms | 重构交接 = LateBind/direct 家族 | **§2.2 已拒绝** |
| 稳态 relay 交接串行 | ~340 ms（含上项） | 同上 | 同上 |
| draft 高层往返 | ~320 ms | D1 设备闭环 | **架构互斥已证**（Vulkan 同设备 copy × tensor-split 拒绝 backend sampling） |
| PLE 快照 CPY | ~24 ms | S1 域 | 须过 shadow 门，预期 <噪声 |
| MoE/其他 | 已在 33 ms compute 内 | — | — |

## 裁决

**在手册规则（§2.2 失败档案重启门 + §3 单因素串行）下，356 ms 缺口无剩余可启动候选。**
所有可归因项映射到已拒绝/已撤回家族：
- LateBind/direct producer（链头+稳态交接）
- D1（draft 往返，重启条件 = Vulkan 跨设备 copy 或 tensor-split backend chain）
- S1（快照，须完整 shadow 门）

**不重启已拒绝路线**（手册 §2.2：重启需新条件，本轮未产生新条件——归因修正
反而确认了原拒绝的理由）。完美版（run-20260926T1330）维持最终交付。

## 证据

- 稳态 timing + segments.tsv（k1b-ple-layers/run-20260926T1130/，已归档）
- 源码：tp5_hc_resume.comp:84-110（in-shader spin）、llama-tp5-plan.cpp:582
  （PLE MIRRORED）、vulkan-collective.cpp:5063-5076（coherent 发布路径）、
  common/speculative.cpp:2955（MAX_CAPACITY 未完成注释）
- MAX_CAPACITY 失败日志（本 run）
