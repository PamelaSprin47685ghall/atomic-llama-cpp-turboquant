# G1 三张账：融合基座（16e28b8ac 干净重建）当前关键路径

日期：2026-09-26。测量对象：`/var/tmp/tp5-control-16e28b8/build-review/bin/llama-server`
（16e28b8ac + 1 行 `<cstddef>` 修干净构建，SHA256 `7bc567b4c2b2…`）。
插桩方式：手动起服务（绕过 harness 环境白名单），env 仅含
`GGML_TP5_PROFILE=1 GGML_TP5_MTP_PROFILE=1 GGML_TP5_PROFILE_RELAY_STAGES=1 GGML_TP5_GPU_TIMING=6`。
未插桩对照：harness smoke `82.55 tok/s / 2071.450 ms`（`/var/tmp/tp5-g0-20260926/control-fused-default-smoke.json`）。

## 观测扰动（插桩 vs 未插桩）

| 运行 | predicted_ms | decode tok/s |
|---|---:|---:|
| 未插桩 harness smoke（稳态） | 2071.450 | 82.55 |
| 插桩手动 server 稳态 #1 | 2098.626 | 81.48 |
| 插桩手动 server 稳态 #2 | 2136.222 | 80.05 |

插桩开销约 **27–65 ms/请求（1.3–3.1%）**。所有绝对数只作归因，不作吞吐声明。

## 请求分相账（warm 请求，`[tp5-mtp-cycle]`，24–25 cycles）

- `avg_draft_us=13.7–14.3 ms`，`avg_target_us=76.6–90.1 ms`（含首 cycle 551 ms 的图首建），
  `avg_catchup_us=0.95–0.99 ms`，`avg_total=91.4–105.4 ms`。
- 稳态 cycle（cycle≥4）：draft ≈13.3 ms、target ≈61.3 ms、catchup ≈1.0 ms、总 ≈75.8 ms。
- 接受率 eff≈1.00；`draft_tokens=6 accepted≈6 final=7`。
- **draft 占稳态 cycle 约 17.6%（13.3/75.8）**，target verify 约 80.9%，catch-up 约 1.3%。

## 阶段表（host 相位）

- 每请求 24 graphs reused（`print_timing`），25 cycles 中 1 次首建（cycle1 target 551.8 ms 内
  含 build+alloc+首执行）；无 phase-reset 重复建图迹象：`graphs reused` 每请求 +24，
  与 cycles-1 完全一致。
- `[tp5-profile] 96 allreduces`（warm 窗口）：p1_rec 21.1–26.7 ms、p1_sub 14.4–17.9 ms、
  p1_wait 32.2–184.9 ms（波动大，含 GPU 排队等待）、p2 全 0（RELAY 模式 P2 常驻）。
  COLLECTIVE TOTAL 69–200 ms/96-stage 链。
- `[tp5-relay-profile]`（每 exec 一条链 96 stages，全 fallback direct=0）：
  ready_us 46.0–46.3 ms、ready_max 2.07 ms、skew 11.0–11.6 ms、cpu_data 3.3 ms、
  handoff 49.4–49.6 ms、route 28–29 us、submit 0.57 ms。

## Producer 表（per-stage，exec=82/83 warm 窗口）

- ready_all 累计 **46.2–46.3 ms/96 stages**；first-ready 累计 34.8 ms；skew 累计 11.4 ms。
- Top 慢 stage（ready_all_us）：stage 1（2.07 ms）、2（1.67）、5（1.50）、3/7（1.49–1.51）、
  9（0.99–1.29）、6（1.22）、4（1.16–1.18）、8（1.13–1.14）、0（0.77）。
  stage 0–9 合计约 12.9 ms，其余 86 个 stage 各 ~0.42–0.49 ms（均匀）。
- GPU 时间戳账（`GGML_TP5_GPU_TIMING=6` 捕获一轮链）：
  - 每 rank 96-stage span 50.6–51.2 ms；compute 31.0–33.3 ms；p2(sum) 16.1–17.8 ms；
    push 12 us；gaps 8 us；tail_compute 0.58–0.61 ms。
  - **慢 stage 集中在 stage 0–8**（最慢 rank compute 964–2237 us，spread 最高 524 us）；
    其余 stage 每个约 200–300 us 且五 rank 均衡（spread 8–16 us）。
  - 96 stages 慢 rank compute 合计 **33.2 ms**。
- 归因：**producer（rank-local compute）是主头**；CPU relay bookkeeping（cpu_data 3.3 ms、
  route+submit <1 ms）不是。ready skew（11.4 ms）主要来自 stage 0–9 的 compute 不均衡，
  不是通信。

## 内存表（运行中 rocm-smi 快照）

| 卡 | VRAM used |
|---|---:|
| GPU0（draft+target host rank） | 14.86 GB |
| GPU1 | 12.44 GB |
| GPU2 | 12.42 GB |
| GPU3 | 12.40 GB |
| GPU4 | 12.30 GB |

- GPU0 比其余 rank 多 ~2.4–2.5 GB：单卡 MTP draft 权重（Q4_K_M draft ~2.79 GB 文件，
  部分驻留）+ host 侧 readback/输出缓冲。
- GTT 无法直接读（`/sys/kernel/debug/amdgpu` 不可访问）；系统 MemAvailable 85.9 GB，
  无 GTT 紧张迹象。此边界记录，不作权重迁移推断。

## 结论（G1 完成门对照）

1. **大头可归因**：warm 请求 ~2075 ms 中，25 cycles × 稳态 75.8 ms ≈ 1895 ms + 首建 ~200 ms
   + prompt ~50 ms。稳态 cycle 的 61 ms target verify 里，96-stage 链 GPU span ~51 ms、
   draft 13.3 ms、catch-up 1 ms。残差（cycle 内非链、非 draft 时间）约 10 ms/请求，公开。
2. **观测扰动已量化**：1.3–3.1%，见上表。
3. **排名明确选择下一项**：
   - target verify 的 96-stage 链中 **stage 0–9 贡献 ~12.9 ms ready 等待**，其中 stage 1–8
     的最慢 rank compute 显著高于其余 stage（1.0–2.2 ms vs 0.2–0.3 ms）——这些是链头部的大
     matvec/attention 入口 stage，正是 K1 多行 matvec 复用的命中面（七行 target verify 的
     skinny-as-batch 投影走 `mul_mat_vec` `NUM_COLS=7`，当前基座每 (列,行) 对重复读权重）。
   - **E1（phase/程序复用）无账可收**：每请求 graphs reused=cycles-1，无重复 reset/release
     迹象；首建 ~200 ms 属进程首请求，不占 warm 时间。
   - **S1（GDN 快照）**：catch-up 仅 ~1 ms/cycle、~24 ms/请求，且 K/V-only 路径已免输出；
     快照成本在 target verify 内未见独立大头。
   - **D1（draft 闭环）**：draft 13.3 ms/cycle × 24 ≈ **320 ms/请求** 是第二大互斥项，
     但其生死门要求先过两步原型；K1 若兑现 stage 头部收益，优先级更高且风险低。
4. **下一步**：K1a——把整合版已有的多行 matvec 权重复用最小增量移植进 candidate
   worktree，先同形核数值对拍，再七行 target 命中统计，再整请求 ABBA。
