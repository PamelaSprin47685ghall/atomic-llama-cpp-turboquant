# E1/S1/D1/V1/H1 裁决记录（基于 G1 三张账）

日期：2026-09-26。run-id：20260926T0421。依据：`artifacts/tp5-next-20260926/g1-ledgers/`。

## E1：exact 程序与资源复用 — **退选（无可收账）**

G1 阶段表证据：
- 每请求 `graphs reused = 24`（=cycles−1，25 cycle 中 1 次首建）；无 phase-reset
  重复建图迹象。`ubatch_execution_phase` 的 predefined MTP 豁免在融合基座实际生效
  （`[tp5-meta] SUBMIT_EPOCH_CHAIN SUCCESS: predefined truth locked!` hits 1→5）。
- 首建 ~200–550 ms 属进程首请求（cycle1 target 551.8 ms 含 build+alloc+首执行），
  不占 warm 时间。warm generation 时间无重复 release/bind 可省。
- 手册 §6 的担忧（"native K/V-only 不获益于豁免"）实测不成立风险面：K/V-only catch-up
  走独立 graph type 但每请求仅 ~1 ms/cycle、~24 ms/请求总量，即便有重复 reset 也低于噪声。

结论：不立项。若未来 G1 复测出现 `graphs reused < cycles−1` 再重启。

## S1：GDN shadow → 精确恢复 — **降级为观察项**

G1 证据：catch-up 稳态 ~0.95–0.99 ms/cycle（24–25 cycle 请求合计 ~24 ms/2075 ms ≈ 1.2%）。
K/V-only 路径已免输出投影。GDN 快照成本未在 producer 表中显示独立大头
（stage 0–9 的 compute 是 matvec/attention 入口，非快照拷贝）。
S1a 字节账的预期收益上限 < 请求时间的 1.2%，低于 ABBA 噪声（±0.8%）。
结论：不进入实现阶梯；仅在 D1 改变执行结构后重估。

## D1：两步设备闭环 — **保留为下一主攻方向（需过生死门）**

G1 证据：draft 13.3–14.3 ms/cycle × 24–25 cycle ≈ **320–350 ms/请求（15–17%）**，
是 target verify 之外最大互斥项；历史 445 ms host sampler 外围与本次 13.3 ms/cycle
一致（含 GPU 等待）。K1a 已证明 stage 头部 matvec 无可收账，D1 成为剩余最大可归因项。
但按手册 §8.3 生死门：先 n=2 两步原型对照同 horizon 原实现，无高层
`llama_decode/common_sampler_sample` 往返；两步不赢即停。本轮未实现（单因素串行原则：
K1a GPU 实验期间不并行改执行结构）。

## V1/H1 — **未启动**

G1 producer 表未显示全词表 D2H 回读是可归因大头（rank-local pinned readback 已在融合基座
生效；ready 等待集中在 stage 0–9 的 producer compute）。V1 的前提（G1 证明该边界仍值得）
不成立。H1 依赖执行结构优化完成，未动。

## 下一轮排序

1. D1 两步原型生死门（唯一 >10% 可归因项）。
2. 若 D1 两步失败：K1b 针对 stage 0–9 具体 shape 的 staging/行 tile 单因素实验。
3. E1/S1/V1/H1 维持退选/观察。
