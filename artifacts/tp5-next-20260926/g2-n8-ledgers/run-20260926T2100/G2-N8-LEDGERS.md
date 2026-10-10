# G2 三张账：n=8 horizon（H1 资格配置）当前关键路径

日期：2026-09-26。run-id：20260926T2100。对象：完美版二进制
`/var/tmp/tp5-final-build/bin/llama-server`（SHA b92fda129…，K1a+K1c+K1d 默认开，
K1e opt-in 关）+ `--spec-draft-n-max 8`。方法与 G1（run-20260926T0421）一致：
手动插桩服务（`GGML_TP5_PROFILE=1 GGML_TP5_MTP_PROFILE=1
GGML_TP5_PROFILE_RELAY_STAGES=1 GGML_TP5_GPU_TIMING=6`），warmup+3 请求。

## 观测扰动

插桩稳态 2039–2065 ms vs 未插桩 ABBA n=8 稳态 1937.8 ms（H1 abba5.json warm 均值）
——插桩开销 ~100–125 ms/请求（5%）。绝对数只作归因，不作吞吐声明。
（n=6 时 G1 插桩开销仅 27–65 ms；n=8 下 relay-stage 逐 stage 日志量翻倍
（每请求 19 cycles×96 stages），扰动更大。）

## 请求分相账（n=8，warm cycle≥3，76 cycles = 4 请求）

|成分|n=8 实测|n=6 旧值（G1）|
|---|---:|---:|
|draft/cycle|**21.1 ms**|13.5 ms|
|target verify/cycle|**74.5 ms**|69.3–70.3 ms|
|catch-up/cycle|0.98 ms|0.88–1.00 ms|
|总/cycle|**~96.5 ms**（warm 均值）|83.7–85.4 ms|
|cycles/请求|19|25|
|draft_tokens/accepted|8/8（eff 1.00）|6/6|

**n=8 稳态请求模型**：19 × 96.5 ≈ 1834 ms + 首建 ~610 ms(c1) 摊销 + prompt。
与插桩实测 2039–2065 ms 自洽。

## 关键修正：draft 占比从 16% 升到 **21.9%**

n=8 下 draft = 21.1/96.5 = **21.9%**（n=6 为 16.1%）。draft 绝对值
13.5→21.1 ms（+56%）：8 步高层逐步往返 vs 6 步。**draft 现在是 target verify
之外最大单项：19 cycles × 21.1 ≈ 401 ms/请求**（n=6 时 ~337 ms）。
D1 家族（设备闭环消除高层往返）的可收账**变大**了，但其重启门
（Vulkan 跨设备 copy / tensor-split backend chain）未变，仍关闭。

## 阶段表（96-stage 链，warm exec=86）

- `plan=96/0`（全部 predefined 复用）、`cb_alloc=0`、`creplay=0/0`——
  E1 结论在 n=8 下不变：无重复建图/分配可收。
- `[tp5-profile] 96 allreduces`：p1_rec 21–28 ms、p1_sub 13.8–16.1 ms、
  p1_wait 46–277 ms（含 GPU 排队）、p2=0（RELAY P2 常驻）。
- relay-stage 账（exec=86，elems=23040=9 行×2560）：
  - **ready_all 累计 56.6 ms/链**（n=6 为 46.3）；first 44.4 ms；skew ~12 ms。
  - stage 0–9 合计 **14.7 ms**（n=6 为 12.9）；其余 86 stage 均值 487 us
    合计 41.9 ms（n=6 为 ~0.42–0.49 ms/stage 均匀）。

## Producer 表（GPU 时间戳，最后捕获链）

|量|n=8|n=6 旧值|
|---|---:|---:|
|每 rank span（96 stage）|62.8–65.4 ms|50.6–51.2 ms|
|每 rank compute|37.8–39.0 ms|31.0–33.3 ms|
|慢 rank compute 合计|**39.8 ms**|33.2 ms|
|stage 0–9 慢 rank compute 合计|11.26 ms|~12.9（ready 口径）|
|stage 10+ 均值|332 us|200–300 us|

- 每链 GPU span 64.6 ms vs CPU 侧 target_us 74.5 ms/cycle：**~10 ms/cycle
  是链外开销**（draft 侧同步、readback、sampler 提交间隙），19 cycles ≈ 190 ms/请求。
- compute 增量（33→38 ms）来自 9 行 verify 的真实算量（+29%），符合
  9/7 行比例（+29%）——**无异常回归**。

## 内存表

未重取（n=8 不改变权重/KV/snapshot 布局，仅 active rows 7→9；G1 内存表
仍有效：GPU0 14.86 GB、其余 12.3–12.4 GB）。TP5_LATE_CAPACITY_ROWS
覆盖 9 行（定义容量 ≥10）。

## 结论与排名

1. **n=8 下 draft 401 ms/请求（21.9%）是最大可归因单项**，但映射 D1 家族
   （架构互斥已证，重启门未满足）——不立项。
2. relay 交接（ready_all 56.6 ms/链 × 19 链 ≈ 1075 ms 中的等待成分）仍是
   结构性最大项，映射 LateBind/direct 家族（§2.2 拒绝）——不立项。
3. **stage 0–9 头部 14.7 ms/链**（n=6 12.9）：K1a 已收割多行 matvec；
   剩余是 9 行下的大 K 投影真实算量。stage 1 慢 rank compute 2509 us
   仍为最大单 stage。
4. E1（无重复建图）、B1（barrier <0.1us/条）结论不变。
5. **无新可立项候选**。n=8 改变了各项占比但未改变任何家族裁决；
   剩余缺口（1940.8−1710=230.8 ms）的成分在当前架构下全部映射已关闭家族。

## 与 100 tok/s 门的差距

n=8 未插桩 1940.8 ms / 88.1 tok/s。若 draft 全消（不可能，D1 互斥）：
~401 ms → 1540 ms 达标。现实可收空间需要新架构证据（D1 重启门或
relay 家族重启门），当前无。

## 证据

- `server-n8-instrumented.log`（7872 条 relay-stage、GPU 时间戳、cycle 账）
- `measured.json`（4 请求全部精确 171/171）
- `g2-run.py`（复现脚本）

## 补充发现（同日深挖）：每请求一次的 ~300ms target 尖峰 = prompt 重评估

- 尖峰周期：req1 c6=223ms、每请求首个 MTP cycle（c20/c39/c58）≈ 297–306ms。
- 该 cycle 的链 elems=69120 = 3×23040（27 行 × 2560）且 n_stages=97——是 31 token
  prompt（cache_prompt=False，n_past=0，server-context.cpp:6405）在链内重评估。
  LCP 槽选择显示 f_sim_best=1.000 但 f_keep=0.153——相似槽被选中但 KV 不可复用。
- **每行成本 8.5ms ≈ 稳态 verify 7.2ms/行**——无 kernel 异常；是工作量体积。
  （n=6 基线同款尖峰：req2/req3 cycle1 target 275–279ms，非回归。）
- small-Q8 特例（Q8_0 ne1≤320，27∈(18,32]）确实把 HC down 投影路由到
  NUM_COLS=27 matvec，但 48×0.41MB×27 重读仅 ~528MB/链 ≈ 1.3ms——不是尖峰来源。
- 裁决：**canonical payload 的固定成本**，改 cache_prompt 即改 workload，不立项。

## n=8 干净 1940.8 ms 的最终分解（G2 三张账合成）

|成分|量|占比|家族状态|
|---|---:|---:|---|
|19 链 GPU span（96 stage relay 流水）|~1235 ms|63.7%|其中 producer compute ~722（真实算量）；CPU-publish 自旋 ~96×100us×19 ≈ 182 → LateBind/direct（§2.2 拒绝）|
|draft 高层往返（19×21ms）|~400 ms|20.6%|D1（架构互斥已证；占比从 n=6 的 16% 升到 21.9%）|
|prompt 重评估（27 行链内）|~230 ms|11.9%|canonical workload 固定成本|
|catch-up + 杂项|~75 ms|3.9%|S1 上界（GDN 快照 ~50ms 上界，含在链内）|

**结论：n=8 下全部成分均有实证归因且映射已关闭/已拒绝家族或 workload 固定项。
本轮无新可立项候选；S1 的真实上界从"24ms catch-up"修正为"~50ms/请求
（1GB/链 S-state 快照写在 36 GDN stage 内）"但仍低于其 shadow 门成本。
D1 重启门不变。**
