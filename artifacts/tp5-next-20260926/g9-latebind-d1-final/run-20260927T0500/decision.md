# G9：LateBind 实测关闭 + D1 随 MTP 退役正式关闭——收官（2026-09-27，run-20260927T0500）

## 工单来源

用户裁决：不考虑 MTP；把 LateBind 和 D1 坐实结束。G8 账本已证明自然文本下
relay 固定延迟（45%）是最大项且映射 LateBind 家族——本轮用**新测量**把它
从"历史拒绝"变成"当前源码 + 自然文本 workload 下的实测拒绝"。

## LateBind 实测（natural text，no-MTP，`--tp5 qwen4exp-af`）

|臂|结果|ms/token|
|---|---|---:|
|baseline（RELAY F32）|3×177-token 完成，3653/3621/3632ms|**20.6**|
|`--tp5-latebind exact`（F32 sidecar）|3×156-token 完成（合法变体：重结合改变近 tie 选择）|**21.0**|
|`GGML_TP5_LATEBIND=hc-down`（aggressive-q8 默认）|**进程在首个 latebind decode 死亡**（53 行日志，无 eval 输出）|—|

**exact 模式 +0.4 ms/token（净负）**。机制实测：每个 decode exec 的
sidecar Q 等待 **1981µs 均值**（median 1972，463 样本）/exec——Y-ready 的
节省覆盖不了它。这与 TP5.md 历史结论（"exact 减少部分 Y ready，却在 Y 发布
后重新串行等 GPU Q，当前没有净加速"）完全一致，但在自然文本 + 无 MTP +
当前源码下重新坐实。

**aggressive-q8（默认档）在首个 latebind stage 崩溃**——fail-closed，无资格。

## 为什么 LateBind 不可收（结构归因）

LateBind 的设计目标是**解耦搬运时机**（Q 充分统计量提前算），不是消除
CPU-in-loop 交接。152µs/stage 的固定交接延迟（G8 账本：45% 的自然文本时间）
全部发生在 **P2 GPU 自旋等 CPU 发布 doorbell → CPU 轮询 ready → CPU 归约 →
CPU 发布 epoch** 的锁步节拍里；LateBind 只是把其中一段 W_down 提前，并
**新增**了 sidecar 的独立等待。G3a（传输微架构）、B1（barrier）的负结果
覆盖其余成分。

**彻底消除 = GPU 端自主接力（P2 直接消费 peer-ready，不经 CPU）**：源码中
`tp5_gpuflag`/`GGML_TP5_SYNC=gpu` 路径存在但被明确标注"experimental and
unsafe under current driver memory model; falling back to timeline"
（ggml-vulkan-collective.cpp 同步模式解析）——RADV/AMDGPU 的跨设备
Device-scope 可见性不保证，这是驱动内存模型级的墙，不是工程量问题。

## D1 正式关闭（随 MTP 退役）

D1（两步设备闭环）的全部价值前提是消除 draft 的高层逐步往返（计数题下
21.9% 占比）。用户裁决放弃 MTP 后：
- **无消费者**：no-MTP 路径没有 draft、没有 spec verify，D1 的 device-side
  token/embedding 闭环机制无对象。
- S1（GDN 精确恢复）同理服务于 MTP 拒绝恢复，无 MTP 时 GDN 快照仅剩
  K=2 常规更新，无恢复语义需求。
- 未来若重启 MTP（更强 draft 模型），D1/S1 重启门见各自 decision.md。

## 最终格局（自然文本 canonical）

|项|状态|
|---|---|
|最优配置|no-MTP（`--tp5 qwen4exp-af`）或 n1-varargmax（−98ms，噪声带边缘）|
|时间构成|relay 固定延迟 45% + 行算量 48% + 杂项 7%|
|全部执行优化家族|LateBind（本轮实测关）、G3a（关）、B1（关）、D1/S1（随 MTP 退役关）、G4c（无 MTP 下 readback 仅 1 行，架构门不值）|
|剩余路径|仅 GPU 端自主接力（驱动内存模型墙，源码自证 unsafe）或换 draft 模型重开 MTP 账本|

**收官裁决：在五张 RX 6800 + RADV + 本模型组合下，自然文本延迟的执行侧
优化空间已穷尽。** 45% 的 relay 固定延迟受限于 CPU-in-loop 架构 +
跨设备可见性驱动保证，不是可渐进修复的实现问题。

## 证据

- `server-nomtp-stage.log`（baseline 3×177-token + `[tp5-numerical-mode] reference`）
- `server-nomtp-latebind-exact.log`（3×156-token + sidecar_wait 463 样本均值 1981µs）
- `server-nomtp-latebind.log`（aggressive-q8 53 行死亡日志，`[tp5-latebind] hc-down active` 后无输出）
- `probe.json`/`probe2.json`、`probe-run.py`/`probe2.py`
- 源码：`ggml-vulkan-collective.cpp`（gpuflag fallback 注释、tp5_star_handoff 锁步、P2 `tp5_relay_copy_f32.comp` doorbell 自旋）


---

## 【勘误 2026-09-27 LB0】本文三处归因错误，判词撤回

用户复核发现并经本人重验（日志+源码），本文以下结论**作废**：

1. **"只动 1 个 HC site"——错。** 每条链 `late=47`（exact 日志 :54
   `[tp5-linear-definition] stages=48 late=47`）。463 行 profile 是 463 次
   exec 的汇总，非 site 数。
2. **"sidecar Q 等待 ~2ms/site，与 40µs 窗口差 50 倍"——错。**
   `[tp5-latebind-profile]` 的 `sidecar_wait_us` 是**链级累计**
   （`ggml-tp5-profile.cpp:121`；`ggml-vulkan-collective.cpp:5780` 在
   stage 循环内累加；`q_spin_n=230=46×5` 佐证）。1981µs/47 ≈ **42.2µs/site**
   （等待）+ 2.5µs/site（数据）。baseline 单行 `elems=2560` 的
   `cpu_data` 实测 **5.19µs**（25201 样本）——且 CPU 归约区间≠完整可重叠
   窗口，不能做比例。
3. **"Q 收缩压进单 WG/单 CU"——错。** Q 派发
   `tp5_cmd_dispatch(cmd, late.late_rank=320, 1, 1)` = 320 WG × 256 线程
   （:3994、`tp5_hc_latebind.comp:52 local_size_x=256`）。单 WG 的只有 LO
   consumer（读 Q、乘 ρ、合并、SiLU），不是整个 W_down 收缩。

**保留**：exact +0.49ms/tok 的负结果（但两臂 profiler 负载不对称——
baseline 多打 25k+ 条 stage 日志、生成轨迹不同 177 vs 156 token——
属诊断级证据，不构成家族否定）；aggressive-q8 崩溃（未定位根因）；
`GGML_TP5_SYNC=gpu` unsafe 警告（限定于 peer-polling 协议）。

**撤回**："LateBind 家族实测关闭"、"执行侧空间已穷尽"、"驱动内存模型墙
= 45% 不可动"。G8 的 45% 账本身需重核（多行外推 + 口径混用 + 分项加总
3394≠3263）。

**后续路线**（见修订版 交接.md §2/§4）：LB0 修证据 → LB1 解除 `z_p/Y`
存储别名（exact 的 Q/norm 串行是 buffer 复用约束，非方程要求）→ LB2 RMS
充分统计量+一次发布（`‖x_b‖² = a_b + 2s_b·Σd + s_b²·c`，ρ 在 relay 内
合成）→ LB3 producer 投影复合。测量纪律：关 profiler、对称日志、保留
"同数学重排不重叠"对照臂。
