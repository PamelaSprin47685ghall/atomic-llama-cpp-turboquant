# P0-A：对照漂移根因——环境（单卡 DPM 降级），非代码；LB1/LB2 平反（2026-09-27，run-20260927T0900）

## 工单

评审二轮把"解释 22.50→27.48 的 +4.98 漂移"列为最高优先。本轮用三步
实验把它钉死了。

## 第一步：干净 A/A（修复 abc.py 全部缺陷）

新 harness `aa-run.py`：env 白名单（只 PATH/HOME/LANG + RADV_DEBUG）、
二进制/DSO sha 记录、每臂前后 amd-smi 快照、3 轮 × 2 臂交错 × 4 请求。

**结果**：27.468 ± 0.105 ms/tok（18 样本，95% 带 [27.26, 27.68]），
36/36 同一输出变体（156 tok，hash 一致）。当前二进制**稳定**——
"27.48"不是噪声，是当前环境的真实水平。

## 第二步：profiler 假设检验（交错 2×2）

LB1 session（22.50）带 `GGML_TP5_PROFILE=1`；abc/aa 都没带。
交错测 profiled vs unprofiled：

- profiled：28.86 / 28.86
- unprofiled：27.39 / 27.47

profiler 让它**更慢** 1.4——方向反了，profiler 假设死亡。
但注意：10:36 二进制 profiled = 22.50，今天 profiled = 28.86——
**同 env 下二进制间差 +6.36**，回归真实存在。

## 第三步：源码二分（决定性）

回退 collective.cpp/hpp + 4 个 shader 文件到 HEAD（剥掉 LB1+LB2 全部
plumbing；K1c/K1d gen 条目保留），重建，同 env 测：

**HEAD 态（无 LB1 无 LB2）= 29.14 ms/tok——比带 LB1+LB2 的（28.86）还慢。**

**LB1/LB2 被平反**：回归不在 LateBind 代码里。回退方向错误，已恢复
LB2 源码（备份于 `/tmp/shader-bisect/*_LB2.*`）。

## 根因：单卡 DPM 降级（环境）

```
card1 (0f:00.0): max sclk 2475 MHz
card2 (09:00.0): max sclk 1200 MHz   ← 降级
card3 (0c:00.0): max sclk 2475 MHz
card4 (06:00.0): max sclk 2475 MHz
card5 (03:00.0): max sclk 2475 MHz
```

- 今天 profiled 日志的 per-rank ready 时间：rank2 = 527µs vs 其余
  373-384µs（1.38×）——慢卡在链里，锁步拖慢全链。
- 非温度（edge 31-50°C，远低于 throttle）；是 amdgpu DPM hysteresis
  在当天连续 4h+ 负载后把一张卡的 boost 档位压掉了。
- 时间线吻合：10:36 是**过夜空闲后**首个 GPU 负载（全卡满档 = 22.50）；
  之后所有测量都在单卡降级态（27.5-29.1）。
- watchdog 规则禁止动时钟/驱动：**记录为环境条件，不修**。卡可能在
  长空闲后自行恢复；驱动级复位被规则禁止。

## 对已有结论的影响

|结论|状态|
|---|---|
|LB1 零效应（27.486 vs 27.485 配对）|**有效**（同臂同降级态配对）|
|LB2 净负 +3.91|**有效**（同上）|
|LB0 的 +2.24（20.41→22.65）|**有效**（同 session 配对，当时可能全卡正常——但两臂同态，配对差有效）|
|一切**绝对值**跨 session 比较|**全部作废**（时钟态不同）|
|G8 的 20.41 基线|属于全卡正常态；当前不可复现直到卡恢复|
|"LB2 代码导致 feature-off 回退"|**撤回**（二分平反）|

## 测量协议增补（写入交接 §6）

**每次正式测量前记录五卡 `pp_dpm_sclk` 尾档**（一行 shell）；任一卡
max sclk < 2000MHz 时该 session 标记为"降级态"，绝对值不得与正常态
比较，只允许 session 内配对差。

## 证据

- `aa-run.py` / `aa.json`（A/A：27.468±0.105 + GPU 快照 + sha）
- `aa-prof.py` / `aa-prof.json`（profiler 交错：28.86 vs 27.43）
- `bisect-run.py` / `bisect.json`（HEAD 态 29.14）+ `server-headk1-*.log`
- per-rank ready：headk1-1.log `[tp5-relay-stage] ready_rank_us`（rank2 527 vs 373-384）
- DPM 表：`/sys/class/drm/card{1..5}/device/pp_dpm_sclk`
- LB2 源码备份：`/tmp/shader-bisect/`（collective/hpp/latebind/publish/resume/gen 的 LB2 版）


---

## 【终局修正 13:20】降级触发时刻与因果链（用户追问后查内核日志钉死）

**触发时刻：2026-09-27 11:47:16**——今天唯一一次 GPU wedge（此前自
2026-09-26 起 0 次）。内核日志：五卡**同时** `ring gfx_0.0.0 timeout` →
五卡 `Starting ring reset` → `device wedged, but recovered through reset`。

**因果链（替换此前"binary timeline"与"hysteresis"两个理论）**：

1. 我的 LB2 rho-v2 实现有双发布竞态（rho 块 fall-through 后旧
   Q-doorbell 块仍执行；LO_U 等 word 0、旧块发布 word 2 → GPU P2
   自旋到 ring timeout）。11:46 rho-v2 正在运行。
2. 锁步链五卡一起 wedge → 驱动复位五卡。
3. 复位后 card2 DPM 表显示降级（1200MHz）——用户指出 sysfs 显示本身
   可能是已知的状态同步假象（`restore-gpu-highspeed.sh` 注释即为此
   而写）；本机有 eagle-gpu-watchdog + max-performance + gpu-hard-unlock
   全套恢复机制，13:11 我已对 card2 执行 SBR（合法路径）。
4. 11:47 后所有测量（abc 27.5、P0A 27.47、bisect 29.14）都在
   post-wedge 态；10:36 的 22.50 是 pre-wedge 态。

**责任归属**：降级由**我自己的实验 bug 触发**，不是环境自发现象。
此前"环境条件、不可修"的表述撤回——本机有标准恢复机制。

**有效性不变**：session 内配对差（LB1 零效应、LB2 +3.91）与 LB0
+2.24（pre-wedge）均有效；跨态绝对值比较作废的结论不变。

**教训**：①跑有竞态风险的 GPU 实验前先想清楚失败模式对整机的影响
（五卡锁步 = 一个 bug wedge 全部）；②发现异常先查 `journalctl -k` 与
本机已有的 watchdog/恢复脚本，再下"环境"结论。


---

## 【R1 修正 13:30】"card2 降级"判断本身重开——显示不可信

用户指出：card2 的 sysfs **历来显示 1200MHz**，与实际算力无关
（restore-gpu-highspeed.sh 注释、TP5.md 09-23 记录"静态 pp_dpm_sclk
可能显示 1200MHz，静态标称值不是负载算力"、ALU 探针实测 17 TFLOPS
均证实此为已知显示假象）。

**因此"post-wedge = card2 DPM 降级"的机制判断不成立**。维持的事实：
- +4.98（22.50→27.48）真实存在（bisect 证明非代码）；
- 分界是 11:47:16 五卡 wedge/reset（journalctl -k 唯一事件）；
- **机制未明**：wedge/reset 后遗症的哪种成分（PCIe 链路重协商 /
  shader cache 失效 / 内存时钟态 / 其它）造成 +4.98——未知。
- post-wedge rank2 ready 527µs vs 373-384 的 1.38× skew 是否异常
  需与 pre-wedge 同口径对比后才能定（G1/G2 历史本有 11-12ms/链 skew）。

**重开动作（R1，OPEN）**：pre/post-wedge 各跑一次同配置 profiled
请求，对比五卡 ready 分布——skew 两态一致则 +4.98 另有来源；
skew 仅 post 出现则真实慢卡（但与 sysfs 无关，用 ALU 探针定位）。

**协议 rev2**：钟态分界 = journalctl -k wedge/reset 事件；sysfs sclk
仅记录不判断。eagle-dpm-guard（轮询 sysfs 的自动纠正）已撤——对显示
假象会反复误触发 SBR。改为方案 C：eagle 的 restore_clock_card
（事件触发路径）已升级为 OD 重写序列（manual→OD s1 2475→commit→
pin DPM→high），备份与 diff 归档于 eagle-hardening/run-20260927T1320/。
