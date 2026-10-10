# LB0：证据修正与对称基线——G9 归因勘误成立，exact 回归成分定位（2026-09-27，run-20260927T0600）

## 目的

按用户复核意见修正 G9 的三处误读，并用**对称 profiler 两臂**重测，给出
LateBind exact 每链每 site 的真实成本表，为 LB1/LB2 定靶。

## G9 误读勘误（全部核实成立）

|G9 说法|核实|证据|
|---|---|---|
|只动 1 个 site|**错**：每链 `late=47`|exact 日志 :54|
|~2ms/site Q 等待，50 倍窗口差|**错**：1981µs 是链累计，**42.2µs/site**|`ggml-tp5-profile.cpp:121` 链级打印 + `:5780` 循环累加 + `q_spin_n=230=46×5`|
|Q 压进单 WG/单 CU|**错**：320 WG × 256 线程|`:3994` dispatch `late.late_rank=320`、`tp5_hc_latebind.comp:52`|
|（附）baseline 重叠窗口 40µs|**修正**：单行 elems=2560 的 `cpu_data` 实测 **5.19µs**（25201 样本）|G9 baseline 日志重解析|

## LB0 对称重测（两臂同 profiler、同日志级别、自然文本 no-MTP）

|臂|steady eval|tokens|ms/tok|
|---|---:|---:|---:|
|base|3611.9ms|177|**20.41**|
|exact|3532.7ms|156|**22.65（+2.24）**|

与 G9 非对称测量（+0.49）不同——**对称 profiler 下回归更大**。G9 的
两臂日志负载不对称（baseline 多打 25k 条），其 +0.49 低估了真实差距。

## 每链每 site 成本表（exact 臂，21298 条 stage 级样本）

|量|均值|中位|p90|max|
|---|---:|---:|---:|---:|
|sidecar_wait（Q ready 轮询）|**42.1µs**|42.4|43.3|205.6|
|sidecar_data（Q 归约）|2.7µs|—|—|—|
|y_publish / q_publish|0.1µs|—|—|—|
|q_spin（LO 消费者自旋）|44.1 iters|—|—|—|

- 47 site × 42.1µs = **1979µs/链** 的 Q 等待——**这是 LB1/LB2 的靶子**。
- 仅 33/21298 个 site >100µs（首链填充期），稳态高度均匀（p90=43.3）。
- 时序语义（`:5084–5121`）：Y 归约 → 发布 word 0（0.1µs）→ CPU 轮询五卡
  Q-ready flag → Q 归约（2.7µs）→ 发布 word 2。**42µs 是 GPU Q kernel
  （320 WG）完成 + flag 穿 host-import VRAM 可见的全程**，从 Y 发布后起算。

## +2.24 ms/tok 的分解（账闭合）

|成分|量|机制|
|---|---:|---|
|CPU handoff 窗口增量|+1.16 ms/链|链 total_us 14.25→15.41ms。等待从 ready-poll **搬家**到 Q-poll：Y 提前发布（producer 少了 Q 计算）使 ready 等待缩短，但 Q 等待是新增串行段|
|链外 GPU 增量|~+1.08 ms/链|47 个 Q kernel dispatch、LO 消费者自旋 44 iters、norm→LO 拆分 barrier、exact 的 scatter/publish 段|

## 对 LB1/LB2 的直接推论

1. **42µs/site 的等待不是"窗口装不下"**——是 Q kernel 在 Y 发布后才启动
   （GPU 队列序：producer → P1(Y) → late_pre(Q)），其完成时间天然落在
   CPU 归约 Y（5µs）之后。LB1 解除 `z_p/Y` 别名后 Q 可与 producer 同段
   执行，等待才有机会被 producer 算量吸收。
2. **LB2 的靶子明确**：ρ 进 relay 后，word 2 发布、Q-poll、LO 的
   norm→LO 依赖链全部消失，42µs/site × 47 ≈ 2ms/链是上限账；代价是
   relay 多算 4 个标量 + 320 维合成（CPU 侧 ~几 µs）。
3. q_spin 44 iters 说明 LO 消费者平均等 ~44 轮 flag 轮询——LB2 消除
   这个二次等待。

## 测量纪律确认

- 两臂同 profiler 同日志级别；exact 生成 156-token 合法变体（轨迹不同，
  ms/tok 口径比较）。
- 正式资格仍需：关 profiler、5-block ABBA、"同数学重排不重叠"对照臂。

## 证据

- `server-lb0-base.log` / `server-lb0-exact.log`（对称 stage 级日志）
- `lb0.json`（两臂 3 请求）、`lb0-run.py`
- 源码时序：`ggml-vulkan-collective.cpp:5084–5121`（Y 发布→Q 轮询→Q 归约→word 2）
