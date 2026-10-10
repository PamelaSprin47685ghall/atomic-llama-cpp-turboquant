# LB1/LB2：LateBind 存储别名与 RMS 充分统计量——实现、测量、关闭（2026-09-27，run-20260927T0730）

## 工单来源

用户复核推翻 G9 的三处误读后（见 LB0 勘误），沿 LateBind 路线推进到底：
LB1 = 解除 `z_p/Y` 存储别名（WAR 串行）；LB2 = RMS 充分统计量 + relay
一次发布（消除 Q 二次 doorbell）。全部在自然文本 no-MTP 单 token 链上。

## 实现（全部 env-gated，默认 off = 逐位原行为）

- **LB1**（`GGML_TP5_LATEBIND_ZSNAP=1`）：每 rank 私有 z_p 快照 buffer；
  late_pre 在 Q 前拷贝 `trefs→zsnap`；Q 的 LocalZ4 绑定重定向到快照；
  exact 线性链的 WAR barrier 移除；WAR schedule 验证器新增 `READ_ZSNAP`
  语义（快照自身的 TRANSFER→COMPUTE barrier 排序 Q 读）。
- **LB2**（`GGML_TP5_LATEBIND_RHO=1`，依赖 LB1）：
  - Q kernel 变体 `tp5_hc_late_q_stats`（12 绑定）：row-0 WG 在收缩后
    折入 `a_b=⟨R_b,R_b⟩`、`d_b=⟨R_b,z_p⟩`、`s_b`（12 words/token）。
  - publisher 变体 `tp5_hc_publish_stats`：Q sidecar + stats 一次发布。
  - CPU handoff：Q-ready 轮询（保留——Q 在 P1 后 ~115µs，CPU 5µs 归约后
    必须防陈旧读）→ stats 跨 rank 折叠（~60 标量读）→ `c=‖y‖²` **折入
    Y 归约循环本身**（`tp5_star_sum_broadcast_f32_cc_avx2`，读已付，
    只加 FMA）→ ρ 发布到 bank stats 区（12 words/token）。
  - LO consumer 变体 `tp5_hc_resume_lo_u`（单 doorbell word 0）：GPU 自己
    折 `u=0.25·Σ_b ρ_b·Q_b` + SiLU——**u 不经 CPU**。
  - frame-null 链（普通 decode）`late_rows=0` 的行数推导：`n_elems/width`
    （此前除零 SIGFPE 已修）。

## 过程事故（全部修复，留档）

1. rho 初版把 c 计算 + u 折叠全放 CPU 标量循环 → **446 ms/tok**——
   bcast 是 BAR 映射 VRAM，逐字访问 ~0.3-0.7µs/word，~40k words/site。
   **用户当场指出"这么重的计算你压给 CPU"**——正确。重构：c 折入 Y
   归约（读已付）、u 折叠回 GPU LO、CPU 只剩 stats（~30 读）+ ρ 发布。
2. rho 块曾与旧双 doorbell 块串联执行（双发布 + u 被旧块覆写）→ 修为
   rho 块自包含、旧块 rho 互斥。
3. 行数推导丢失导致除零 SIGFPE（exit -8）→ 恢复 `n_elems/width` 推导。

## 最终三臂测量（同二进制、同 session、3 请求/臂、全部自然 stop）

|臂|配置|ms/tok|对比|
|---|---|---:|---|
|ctl|exact legacy（双 doorbell）|**27.48**（4288ms/156tok）|基线|
|zsnap|LB1 别名解除|**27.49**（4343/158）|**+0.01（零效应）**|
|rho|LB2 单 doorbell|**31.40**（5212/166）|**+3.92（净负）**|

（ctl 从 LB1 session 的 22.5 漂到 27.5——session 漂移或管线 plumbed；
同 session 配对比较为准。）

## 归因（账闭合）

LB1 零效应的原因：**串行化的不是 barrier，是单 GPU 队列顺序**
（`P1(Y)→late_pre(Q)→P2(norm)`）。barrier 删除后 norm 仍排在 Q 后。

LB2 净负 +3.92 ms/tok = ~83µs/site，成分：
|项|量|
|---|---:|
|Q kernel 内 stats 折叠（residual+z 的额外整遍读，关键路径上）|~+100µs GPU|
|stats sidecar 发布（publisher +12 words/token）|~+几 µs|
|CPU stats 折叠 + ρ 发布（uncached BAR ~90 words）|~+30µs|
|LO_U 从 bank 读 4×Q+ρ（旧路径读本地 rho_buf）|~+几 µs|
|**省下的二次 doorbell（Q-ready 等待）**|**−42µs**|

净 ≈ +83µs。**即使把 stats 折叠完美融进 Q 的收缩循环（同 loads，~免费），
也只能收回 ~100µs，rho 仍 ≥ ctl**——因为 doorbell 本身只值 42µs，而
publication + LO_U 读 bank 的成本是结构性的。

## 结构教训（写给下一轮）

**"二次 doorbell"从来不是纯开销——它是 Q kernel（115µs GPU）的完成
信号，而无论谁折 u，Q 都在关键路径上。** LB2 把折叠移上 GPU，但新增
的 stats 发布成本超过 doorbell 节省。LB1/LB2 的数学（充分统计量分离）
是对的，但在这个 workload 的成本结构下（单 token、Q 115µs、doorbell
42µs），**可收空间 < 必付成本**。关闭。

## 裁决

1. **LB1 关闭**：零效应（队列顺序才是串行根源，非 barrier）。
2. **LB2 关闭**：净负 +3.92 ms/tok，账闭合（预测=实测）。
3. 全部代码保留为 env-gated opt-in（`ZSNAP`/`RHO`，默认逐位原行为），
   CPU 回归 4/4 通过；不注册 harness variant，不做 ABBA（pilot 未过关）。
4. LateBind 家族最终状态：**exact 实现无收益（LB0 对称基线 +2.24）、
   LB1 零效应、LB2 净负、aggressive-q8 崩溃**——四条路径全部实测关闭。
   剩余未测的只有 LB3（producer 投影复合 `C_{p,b}`），重启门：需要
   stats 零成本化的 kernel 融合证据。

## 证据

- `abc.json` + `server-abc-{ctl,zsnap,rho}.log`（三臂终测）
- `server-rho-v{1..4}.log`（446→73→31→31 ms/tok 的实现迭代轨迹，
  含 `[lb2-dbg]` 计时行：v1 stats_copy=403µs rho_u=12420µs/site）
- 源码：`ggml-vulkan-collective.cpp`（zsnap/rho env、cc_avx2 归约、
  rho handoff）、`tp5_hc_latebind.comp`（Q_STATS）、`tp5_hc_publish.comp`
  （PUBLISH_STATS）、`tp5_hc_resume.comp`（LO_U）


---

## 【评审二轮修订 2026-09-27】本文四处结论降级（v3）

外部评审指出并经本人对源码逐条核实成立：

1. **"Q 固定 115µs"作废**：`late_pre_us` 逐 stage 波动 515-598µs（stage
   8-12）vs 58-59µs（stage 14/20），且计时区间 ts3..ts4 包含
   `emit(p2, 0, norm_end)`（norm 段）——不是纯 Q kernel 时间。
   "stats 给 Q 加 100µs"引用了错误口径。
2. **"doorbell=42µs 可省"作废**：42µs 是等 Q 算完；LB2 保留了 Q-ready
   轮询，Q 仍须算完（`t_LO ≥ max(t_Q, t_ρ)` 与门铃数无关）。单发布只省
   发布/检测/恢复，不能抵扣整个 Q-ready 区间。本文"账闭合"降为
   **初步归因**——83−100=−17µs 的自相矛盾正是分项未闭合的证据。
3. **LB1 结论限定**：实测证明的是"只解别名不改编排无收益"；不是
   "重叠做不到"——提交顺序不建立执行依赖，真重叠未验证。另 LB1 两轮
   出现 156 vs 158 tok 输出差异，只改存储同步不应改数学——**正确性
   问题先于性能结论**。
4. **"都是 BAR 映射"作废**：star_host 是 posix_memalign 主机内存经
   host-import；bcast 是 device-local VRAM（cached 优先）。教训应表述为
   "CPU 统计量在产生处保留主机副本，禁止写回映射区再读回"。
5. **LB3 无逻辑前置**（本文"重启门=stats 零成本"撤回）：LB3 只改 Q 的
   生产方式，可保留双门铃；门槛是"新增非重叠成本 < 减少的关键路径
   成本"，不是零。

**修订后裁决**：LB2 当前实现不晋级（+3.91 实测成立）；后续优先级 =
修 control 漂移（+4.98，比回退还大）→ 定位 LB1 输出差异 → 调度真
改造/组件消融/LB3 独立原型。全文见修订版 `交接.md` v3。
