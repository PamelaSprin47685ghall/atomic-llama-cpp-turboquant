# K1b-1 预实现裁决：conv tap 补丁撤销，PLE 成本重新归因

日期：2026-09-26。run-id：20260926T1300。承接 K1B-VERDICT 的方向 1。

## 深挖结果（segment dump 逐 CPY/逐 dispatch 分析，stage 2 vs stage 10 全量 diff）

PLE 层每层的超额成本分解（每 verify，每层）：

| 成分 | 量 | 类别 |
|---|---|---|
| **`cache_ple_r` 7× CPY(92160 F32)** | **2.52 MB 写** | **S1 状态快照域**（n_rs_seq=6 + 1 个 rollback 槽的卷积历史回写，`build_conv_state_at` per-slot 循环） |
| 44 个额外小 dispatch（MUL+9/CONT+9/ADD+5/RMS_NORM+2/门控链 8 个逐元素） | ~220–350 us | dispatch 数开销 |
| conv tap CONT+MUL（4 tap × 80KB）+ 权重列 CPY | ~1.4 MB 小算子 | K1b-1 原目标 |

**修正 K1B-VERDICT 的方向判断**：conv tap 物化只占 PLE 超额的小头（~1.4 MB 小算子），
不是 1.4 ms/层的主要来源。占大头的是：

1. **per-slot 卷积历史 CPY（2.52 MB/层）**——这是 `n_rs_seq+1` 份 rollback 快照，
   属手册 §7 S1 的明确范畴（"conv_state per-slot CPY 属 S1 范畴不动"，K1B-VERDICT
   方向 2 已标注）。减少它须过 S1a 字节账 + S1b shadow replay 门。
2. **44 个小 dispatch 的固定开销**——门控点积链（SUM_ROWS/SQRT/SGN/SIGMOID/CLAMP/ABS）
   与分组 norm 的逐算子 dispatch。可做算子融合，但每个 dispatch ~5-8 us，
   融合上限 ~250 us/层 × 4 层 × 25 链 ≈ 25 ms/请求——值得但属算子融合工程，
   非最小补丁。

## 裁决

**撤销 K1b-1 conv tap 补丁（未实现即撤销，零代码浪费）。**

- conv tap 优化目标错位：80KB×4 的 CONT 不是 1.4 ms/层的来源。
- PLE 真正的可收项按手册归属：per-slot CPY → S1（须先过 shadow 门）；
  小 dispatch 融合 → 算子融合工单（收益上限 ~25 ms/请求，需先做 dispatch
  成本账确认 5-8 us 假设）。
- **不启动 S1**：G1 已裁决 catch-up/恢复项仅 ~1 ms/cycle、~24 ms/请求总量，
  PLE 快照 CPY 虽在 verify 链内（非 catch-up），但 S1 的完整门（shadow replay
  oracle、逐位对拍、事务协议）成本远超本轮剩余空间，且手册 §3 排序明确
  "S1 降级观察"。

## 对"最终没有办法"的收尾判定

剩余可归因项全部落入三类：
1. S1 域（须过 shadow 门，工程量大、预期收益 < 噪声或需完整 S1a-d）
2. 算子融合（44 dispatch/层 × 4 PLE 层，上限 ~25 ms/请求，需新工单+逐算子账）
3. D1 域（架构互斥已证，须 Vulkan 跨设备 copy 或 tensor-split backend chain）

**356 ms 缺口在当前架构+手册规则下无最小可验收补丁可兑现。** 进入收尾：
整理完美版（基座 + K1a 非劣成员 + cstddef 修复 + RERoT 可用性修复），
全部已验收、零回退、默认路径行为与 83 基座一致。

## 证据

- `segments.tsv`（本 run-id 已归档于 k1b-ple-layers/run-20260926T1130/）
- stage2 vs stage10 op diff：VIEW+22/RESHAPE+14/CPY+11/MUL+9/CONT+9/ADD+5/
  RMS_NORM+2/逐元素+8（Vulkan 上 VIEW/RESHAPE/TRANSPOSE 为零 dispatch 空算子，
  `ggml_vk_is_empty`）
- CPY 清单：stage2 独有 7× cache_ple_r(92160) + 4× ple_conv1d 权重列(10240)；
  cache_r/cache_s 两 stage 共有（GDN 域，非 PLE）
