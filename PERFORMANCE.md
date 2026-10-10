# 性能调优总纲：TP5 × LateBind × MMVQ × MTP

更新：2026-10-09。适用：本仓库 Qwen4EXP-AF、五张 RX 6800、Vulkan/RADV，目标是单路自然文本生成的低延迟。

**理论口径已重置。撤回“274→95–104→64.74–67.68 tok/s”构成必要上界收紧的推导。** 独立微基准仍是某个实现的耗时，现有复制/分片、host relay与dispatch数量不自动具有理论必要性。预测与实测接近不等于最优性。先读 [必要成本理论重建](docs/performance-necessity.md)；既有实验数值保留，但不得据此宣布67接近硬件极限或关闭新设计路线。

本次源清单复核：1224张量中，去除两张当前CPU/索引表后的选中原始编码量为4.614GB，当前rank布局的副本访问总量为9.198GB。两者都是条件清单，尚非最小DRAM流量定理。全设计空间硬件上界及最优性Gap均**未获证明**，不是用另一个拟合值补空。已有24份测量原始响应复核仍给出基线67.5681 tok/s；这是实测一侧，不是理论下界。本轮没有新增GPU跑分。

用户最新口径：**先不论 RERoT，不限制 CPU 为单线程；其他配置可变。** 最新真机证据在 `artifacts/perf-gap/run-20261009T0145/`：同 APEX 权重、相同分片、无 MTP、aggressive，紧凑 top-k 采样使 **64.77 → 67.15 tok/s**；四对净增益 **3.66%，95% CI [2.64%,4.69%]**。新路径通过 `GGML_TP5_GREEDY_HOST=1` 显式启用，默认仍关闭。

实际理论值、真实 tok/s、已收回和剩余差距，先读 [最新测量报告](artifacts/perf-gap/run-20261009T0145/REPORT.md)。冻结实现成本预测为66.64 tok/s，独立实测67.15；七个留出进程最大误差2.23%，没有把未解释余量回填入模型。**这不是将权重流式参考273.86证明收紧成硬件极限67。** 八点交互扫描没有超过2%筛选门槛的新候选。最初基线发生STAR timeout，根因未修复；新成绩只取得有限范围性能证据，不代表全配置可靠性。

上段是01:45轮次的历史成本预测，不是最新理论或全局最优证据；后续07:30轮次原始配对保留在 `artifacts/perf-gap/run-20261009T0730/`。理论必要性一律以页首重建入口为准，禁止把“独立于整体跑分”误当作“对全部实现不可避免”。

此前同模型 LateBind off→aggressive 的59.66→65.17独立比较见 [00:40报告](artifacts/perf-gap/run-20261009T0040/REPORT.md)，不与本轮跨会话分数相减凑叠加收益。模型转换、缓存实验与探索见 [10-08报告](artifacts/perf-gap/run-20261008T1845/REPORT.md)。原Q2_0的BF16 HC没有命中LateBind；默认无MTP的attention复制布局不是最强对照。旧mtp-strong四对确认中途加载失败，未完成，不能当作MTP正式胜负结论。

**从本页开始，不再从某一轮“终局交接”开始。先证明运行正确、路径确实命中、计时口径一致，再讨论速度。优化对象是每个实际交付 token 的整条关键路径，不是开关数量、shader 行数或单个漂亮峰值。**

本页是当前方法、配置和工作顺序的统一入口。[TP5.md](TP5.md)、[MMVQ+MTP.md](MMVQ+MTP.md) 是保留的历史实验账，包含不同日期的更正，不构成今天的执行指令。源码与原始证据优先于本页；代码变化后须重新取得资格。

历史整理范围：最初的文档清理没有运行 GPU，审计起点为 `master@9be06d5ab` 加已有未提交改动；当时部分工作区外原始请求未能读取。随后 10-08、10-09 的实际 GPU 实验与各自身份已写入上方独立报告。下面较早的“源码已核实”“历史记录”“待实测”不自动获得最新资格。

## 1. 先把目标写清楚

### 1.1 三个“单”不是一回事

| 名称 | 本页含义 | 控制方法与边界 |
|---|---|---|
| 单路 Decode | 同时只有一条用户输出序列 | `-np 1`，无并发请求；这是主目标 |
| 单 CPU 计算线程 | ggml 主计算线程预算为 1 | 显式 `-t 1`；不等于 HTTP、驱动、调度等整个进程只有一个线程 |
| 单 token target 调用 | 一次 target 只处理一个有效 token | 关闭 MTP 时常见；MTP 单路生成仍会出现多行验证 |

TP5 是五个 GPU rank 参与一个 target，不是五路用户吞吐。MMVQ 多列也不是多用户并发。必须把 CPU 线程数、用户并发数、实际验证行数分别填进结果。

历史 55–63 tok/s 等记录不能自动标成 `-t 1` 成绩；没有实际命令、运行身份和线程证据，就只写“单路生成历史记录”。把整个进程用 `taskset` 压到一个逻辑 CPU 是另一种更强约束，可能让 relay、驱动与服务线程互相争用；不得拿它与只设 `-t 1` 混测。

### 1.2 验收指标

主指标为同一质量标准下，稳态生成总时间除以公开交付的 completion token 数：

```text
ms/token = Σ generation_ms / Σ public_completion_tokens
token/s  = 1000 / (ms/token)
```

另报首 token 延迟、完整请求墙钟、prompt eval 时间、输出长度、自然 EOS/预算停止、p50/p95 请求延迟。服务器 `predicted_ms` 与客户端墙钟不是同一计时范围；不要相减后把全部差值叫 CPU 开销。先按进程/配对块聚合，再做统计，不能把同一进程内的 token 当独立样本。

固定自然问题是现成的回归输入：

> 请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。

它只是工程测试文本，不是医学建议。固定模板、关闭 thinking、temperature=0、seed=42、cache_prompt=false；主排名必须自然 EOS。开放式中英文问题、长上下文、代码/算术作为保护集。预算停止可以是保护集的正常结果，不能冒充主排名的完整答案。计数 `1..60` 继续做回归，但其高接受率不能代表自然文本速度。

## 2. 当前能说什么，不能说什么

router 输出行分片当前仍是默认关闭的实验原型。`router-slice-fixed-bin` 的实际 144-stage
原生链保留 94 个 LateBind site；6-token prefill 加四次固定后续 token 的完整词表 logits
及序列化状态逐字节对拍一致。修复 meta 的同父 UID/同节点数 graph view 错复用后，
48 层五次执行的 2640 份完整 router 中间量和 2400 份物理 rank 输入也逐字节一致；
旧 observer 失败原样保留，不能用它作数值资格证据。
这些窄范围结果没有覆盖自然文本资格：同一冻结程序的 h0 参考 pilot 为 67.0260 tok/s，
router 分片臂在第一条 27-token prompt 的 warmup 请求中因 AllGather 源契约拒绝而 HTTP 500。
失败发生在 meta 定义期；没有取得候选吞吐，不晋级、不删除失败，也不放宽 extent 门。
原始材料见 `artifacts/perf-gap/frontier-20261009T154118Z/router-pilot-{A,B}/`、
`router-fixed-{trace,state}-comparison.json`。Sampler/RNG、active 小于 capacity、长测和
全局必要性证书仍未由这些对拍取得资格。

后续查明该定义期失败来自分块 prefill 最后一层合法的空 score/gather；空算子不建立
collective，正载荷的 source/extent/epoch 门保持不变。CPU live→empty-head→live 回归
先失败、修复后通过。修复后同一冻结程序的顺序 B/A 自然文本 pilot 为参考
66.6914、router 行分片 61.8741 tok/s，降幅 7.22%；六次对应正文及 token 数一致，
measured 无 fallback、两进程正常退出，冷 warmup/guard 的 fallback 仍保留。
这不是八个独立配对块的确认结果；该具体行分片候选没有收益，不晋级。
证据见 `router-empty-head-{A,B}/` 和 `router-empty-head-review.json`，早期 HTTP 500
及独立诊断保留；后续通过不抹去旧失败。

2026-10-10 补充：router 行分片所需的 native AllGather 动态行数能力已在真机五卡上
单独取得资格（与该行分片候选的胜负无关）。`gather-capacity-final.json`：
shards=[103,103,102,102,102]、C=4、active 4→1→3→2→1、双 bank 交替；五步全部
验证精确 scores、P1 源 bank 未触碰的 inactive tail、status[9..12] 元数据与下游
softmax；不对称 active/capacity 帧在提交前被拒。为此修复了三个真实缺陷：
RELAY chain 缓存复用未限制在“其编译命令仍驻留 per-rank scratch”的链上，
陈旧 start-bank 的复用会重提交绑定另一 mailbox bank 的 P1/P2；新增标量 ragged-row
ADD producer（逐元素 live-row guard）；提交帧作为 indirect-dispatch 补丁的权威
extent 源转发。同一轮还修复：converted router 权重 strided 写入失效不足
（failing-before/passing-after 见 `router-strided-write-proof.json`）；无损 BF16
router 权重改用通用 F32 GEMV 几何的 native BF16 pipeline，全 512 分数与 generic
F32 逐位一致（`router-bit-native-comparison.json`，合成权重，非真实 target 状态）。
这些是能力与数值修复，不是吞吐收益声明；行分片候选仍不晋级。

同日：无损 BF16 router 权重候选完成真实模型固定 token 对拍。同一冻结快照
`router-bf16-native-bin/`（f32 与 bf16 仅差 `GGML_VK_ROUTER_WEIGHTS`），
6-token prefill 加四次 teacher-forced decode，五步全部 248320 词表 logits、
序列化 state 及全部 3600 份 router trace 张量（含逐 rank 物理输入）逐字节一致；
`router-bf16-parity-comparison.json`，两臂各 3607 个文件零差异。为使该路径在
TP5 replay 定义下可运行，修复了四个真实缺陷：转换的临时 transfer/compute
CB 会污染活跃定义 tape（挂起 capture）；转换 CB 被误登记进 replay pending 列表
（改用无 vk_ctx 绑定的临时上下文）；mid-graph 提交重复登记（去重）；首次转换
路径的两次 mid-graph synchronize 会拆分共享提交并使五 rank 集体执行失序
（仅 LRU 驱逐保留 drain）。自然文本吞吐、sampler/RNG 与长测资格仍未取得，
该候选默认关闭。随后同一快照的顺序 A/B 自然文本 pilot：f32 67.2561、
bf16 67.9025 tok/s（+0.96%），六次对应正文与 token 数完全一致，两进程正常
退出；bf16 臂 guard 请求也从 52.61 升至 61.59 tok/s，与 router 权重流量减半
的方向一致。+0.96% 低于预登记 2% 晋级门，单进程噪声未排除；这只是 pilot
信号，不是晋级结论，未做八个独立配对块确认前不改变默认关闭状态。证据
`router-bf16-natural-{A,B}/` 与 `router-bf16-natural-review.json`。

随后完成预登记的八个配对块确认（ABBA 轮换、16 个独立进程、同一冻结快照、
唯一差异 `GGML_VK_ROUTER_WEIGHTS`）：八块全部 bf16 胜出，几何均值比
**1.01135，95% CI [1.00788, 1.01482]**，即 +1.14% [0.79%, 1.48%]；
16 个进程全部正常退出，每块两臂六次对应正文与 token 数完全一致。
f32 臂 pooled 均值约 66.79，bf16 约 67.55 tok/s。证据
`router-bf16-pair-{1..8}-*/` 与 `router-bf16-paired-confirmation.json`。
CI 下界 1.0079 > 1，配对块证据支持正收益；该候选作为合格候选保留，
仍默认关闭——生产默认值改变需单独取得资格，且 sampler/RNG、
capacity>active、长测与自然 EOS 分布资格未由本实验覆盖。

本轮继续优化的证据保存在 `artifacts/perf-gap/frontier-20261009T154118Z/`。冻结参考 A/A 为
67.1607 与 67.2925 tok/s，差 0.196%；六次对应正文及 token 数一致，measured 无 fallback，
但 warmup 与 guard 的图缓存未命中仍保留，不能据此宣布完整状态或可靠性资格通过。
同一候选二进制的固定 CPU relay tile pilot 为 tile0 67.4520、tile2 67.0784、tile4 66.2113 tok/s，
未取得收益；该候选的生产接入已删除，源码归档与原始结果保留，没有补做正式配对确认。
冻结 h1 的 native submit 失败根因仍未闭合；增加失败日志后的另一 h1 进程完成，但六次对应正文
均不同于 h0，未通过 exact output 门，不晋级。失败日志不是重试或保护门放宽。
2026-10-10 更新：首个非平凡全局必要性证书已建立并迭代收紧（此前
对允许策略集 A（无损表示、任意缓存/分区/预取/融合、完美 MTP 接受）
逐资源求必要性下界——R1 输出发射、R2 因果割集、R3 活跃权重 DRAM、
R4 FP32 ALU——取跨资源 max 得每 token 时间下界。v2（`necessity-
certificate-v2.json`）只计活跃专家，得 S_hi = 2395 tok/s。v3（`necessity-
certificate-v3.json`）曾补上稠密权重项并给出 S_hi = 476 tok/s，但该单一数字
经审计存在三处缺陷，已被 v4 撤回并拆分：稠密字节实为每层 63.82MB（非
90.8MB，来自 1222 张量审计清单）；L2 保留额度须按活跃集而非全池计（对
实现最大宽容）；且 MTP 行批验证使权重 DRAM 流量按周期摊销，R3 在完整
允许集 A 下不成立。v4（`frontier-20261010T000000Z/necessity-
certificate-v4.json`）给出分层界限：**L_global = 5667 tok/s**（R4 FP32 ALU
绑定，对完整 A 有效）与 **L_nomtp = 730 tok/s**（R3 绑定，仅对无 MTP 条件
类 A_nomtp 有效，700.9MB/GPU/token 必要 DRAM）。串行链 R5 调查结论：
完全放松集合（任意融合/缓存/完美 draft）下无可证延迟耦合项，收敛杠杆在
U 侧。原 Q2_0 参考（50.8 tok/s，DIAGNOSTIC_ONLY）对 L_nomtp 的 Gap =
14.4x，仍未收敛；不计算伪 Match 比例。测试
`tests/test-tp5-necessity-certificate.py` 14 项通过。
`necessity-status.json` 已更新为 v4 状态（同目录）。
2026-10-10 第二轮（`artifacts/perf-gap/frontier-20261010T000000Z/`）：契约 v5（190 token
自然 stop，长度门 160–240 首次通过）下完成原 Q2_0 h0/h1 的 8 独立进程配对确认
（ABBA/BABA、同一冻结二进制、唯一差异 `--horizon 1`）：几何均值比
**1.13215，95% CI [1.12320, 1.14118]**，即 h1 相对 h0 +13.2% [12.3%, 14.1%]；
全部对应正文（含 warmup/guard）逐字一致，measured 全部 190 token 自然 stop、
零 issue、rc=0 无强杀；接受率 0.731。旧 APEX 类 h1 正文不一致失败在 Q2_0
上不复现。h1 晋级为该契约下最强已知臂（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`），
生产默认不变。周期账诊断（D2/D3/D5）同时定住瓶颈：每 decode step 17.6ms
中 GPU 忙仅 ~0.65ms，96 stage × ~160μs 为主机↔GPU 互等轮询；每 stage 每
rank ~35 dispatch + ~29 barrier，RADV 派发固定开销占主导（launch-bound）。
TIMELINE 传输 A/B 为 3.33 tok/s（慢 15×，臂 FAILED），确认 RELAY epoch 链为
当前最优传输。S_lo（h1，v5 契约）54.76 tok/s，对 L_nomtp Gap = 13.3×，
仍未收敛；下一杠杆为 MTP 深度扫描与派发数削减。
同轮深度扫描（C-h2/h3-v5，单因素 horizon，正文均与 h1 一致）：h2 51.12、
h3 45.06 tok/s——周期成本随验证行数超线性增长（h1→h2：行数 1.18×，
周期成本 1.38×），MTP 摄销在 h1 饱和，**h1 确认为该 draft 在 Q2_0 上的
最优深度**。派发数削减入口已定位（ADD+RMS 融合模板 + multi_add_rms.comp；
目标 HC 流混合元素链 74 节点/stage → ~10 核），需 C++/shader 工单带对拍，
未在本轮实施。
HC-Q8 派生 target 臂（E1/E2，另一数值路径不拼账）：off 56.78、aggressive
62.32 tok/s（同 target 内 +9.7%）；**aggressive 命中 late=94 site**、派发
3391→2853（−16%）——直接验证 launch-bound 诊断与 LateBind 杠杆；长度门
139<160 失败按近似路径记，不晋级主排名。原 Q2_0 同等收益需 BF16
LateBind（工单 A-native）。
原 Q2_0 F 系列（同 target）：h1+mmvq-auto 无收益且正文变（近似路径不
晋级）；h1+aggressive（BF16 HC 不 capture，late=0，但 gtt-push 传输
+0.8% 与 hc-down lowering 派发 4117→3647 共同 +3.7%）8 进程配对确认
**1.02622 [1.01703, 1.03550]**，正文与 h0 逐步一致。**最强合格臂
 h1+aggressive 池化 56.47 tok/s；本轮原 target 爬升 +16.8%**
（48.37→56.47，逐步正文一致）。S_lo=56.47，对 L_nomtp Gap = 12.9×。
同轮工单 A-native 代码落地：BF16 HC 段融合/LateBind shader 变体、capture gate、
collective EXACT_F32 BF16 路径与 `tp5_hc_late_q_bf16` Q 核全部实现（新快照
`measured-bin-v2/`）。capture 命中 late=94、派发 2702，但 exact 链在 stage 0 Q
sidecar 握手挂起；二分排除 Q 核（强制 Q8 Q 同挂）、排除段融合与 collective
回归（G4/G5/G7 对照全绿）。改动全部 opt-in，默认与 HC-Q8 路径无回归；
BF16 LateBind 标记 OPEN 待 GPU 侧内省继续（证据
`frontier-20261010T000000Z/G1..G7`）。
### 2.1 源码层面的确定事实

`common/common.cpp::common_tp5_apply_env()` 对五个不同 RX 6800、`qwen4exp-af` 配置采用 RELAY/F32，并为合格路径启用 replay/linear lowering。MTP 默认单卡 draft；无显式 horizon 时仍设置 `n_max=6`，不是近期自然文本搜索偏好的 1。MTP 还会改变 attention replicate/headmap、KV-only、small-Q8-columns 等默认行为。因此所有正式命令都要显式指定 MTP 深度，并保存 resolved 配置。[S1]

`--tp5-latebind` 的 CLI 值是 `off / exact / aggressive`。八个实验家族中的 q8first、pack-once、ap、l2、l2-ap 是 CLI 与环境变量的组合，不是另外五个 CLI 值。旧手册中固定四行容量、旧工作组几何和“全家族永远关闭”的说法不再作为当前事实。[S1][S3]

### 2.2 性能证据账：模型、日期、分母必须一起看

| 日期 / 范围 | 既有记录 | 本页允许的解释 |
|---|---|---|
| 09-26，计数负载 | 约 83 tok/s，h=6 等旧融合配置 | 历史回归，不是自然文本或严格 CPU 单线程上限 |
| 09-28，自然文本旧源码 | aggressive 对 off 的 ms/cycle 比 0.9579，95% CI [0.9178, 0.9979] | 当轮记录支持约 4.2% 周期节省；不能跨源码继承 |
| 10-07，输出合格子集搜索 | off/disabled/h1；C/R=1.034315，CI [0.996803, 1.073239] | 未检出显著收益；C 与 B 配置相同，不是新配置优化 |
| 10-08，修复后独立确认记录 | R 无 MTP 55.62；B off/disabled/h1 58.41；C aggressive/auto/h1 59.97 tok/s | 记录报告 C/R=1.0715 [1.0340,1.1104]；C/B=1.0322 [0.9862,1.0804]，后者未显著 |
| 10-08，另一个 Q2_0 模型记录 | R off/disabled/h1 58.01；A aggressive/disabled/h1 60.90；M aggressive/auto/h1 61.51 tok/s | 记录报告 A/R 延迟比 0.9565 [0.9315,0.9821]；M/A=0.9951 [0.9429,1.0502]，不能宣称 MMVQ 另有稳定收益 |

上表来自历史账和保留证据索引，不是本次重新跑出的数据。10-08 两组记录不能拼成同一个实验：Q2_0 与原模型的 GGUF、分片、量化布局不同；第一组 C 同时有 LateBind、MMVQ 和 MTP，不能把 C/R 的全部差值归给其中一项。四块三臂轮序也不应仅因文件名带 ABBA 就视为完整两臂 A–B–B–A 设计，须检查实际顺序和统计脚本。

10-08 文档同时保留“全配置修复进行中”和后续确认报告。当前应按覆盖矩阵读：已报告 `8 家族 × 3 MMVQ 策略 × h1` 的 144 个连续请求，以及深 MTP16 的八家族检查；这不等于所有策略、所有深度、所有相位的笛卡尔积都已验收。部分 auto 变体预算尾段的 fallback 曾未闭合，不能因为后续摘要写“完成”就抹掉这一项。

**目前没有经本次独立复核的绝对极限 tok/s，也没有“四项全开必然最快”的证据。** 工作起点是明确的参考臂 off/disabled/h1，加一个无 MTP 对照；aggressive/disabled/h1、aggressive/auto/h1 是需要在当前模型上确认的候选，不在本次整理中静默修改生产默认值。

## 3. 整条执行链如何连接

```text
用户请求 / chat template
  → target prefill，建立 KV / recurrent 状态
  → 单路生成周期
      ├─ 无 MTP：target 单行 → 采样 → 提交一个 token
      └─ 有 MTP：draft h 步 → target 验证若干有效行
                  → 接受前缀 / 拒绝 / bonus
                  → rollback、KV-only catch-up、carry 更新
                  → 提交若干公开 token

每次 target 执行内部
  → ggml 图、meta TP5 分片与完整定义资格
  → 五 rank Vulkan 子图 / replay
  → producer → P1 发布部分 Y → host RELAY 归约 / 广播
  → consumer 在首次使用数据处等待
  → 下一层，最后接 vocab / sampler / readback

LateBind：在合格 HC 边界提前算 Q，延后绑定全局 Y / rho
MMVQ：在合格收缩内部，把 activation 量化为 Q8_1 再整数点积
MTP：改变 target 的实际行数、draft 成本和每周期产出
TP5：改变每 rank 分片、复制、同步和最慢 rank 的等待
```

同一条 GPU 队列里“先提交”不自动等于执行不能重叠；反过来，删除一个 barrier 也不证明发生了重叠。必须检查数据依赖、内存别名、调度与真实时间线。GPU timestamp 包围的 consumer 可能在自旋等 CPU，不能把这段全部算作纯计算。[S2][S3][R2]

## 4. 四项交互，逐项拆开

### 4.1 TP5 × MTP：验证变宽，拓扑也可能变了

无 MTP 与有 MTP 的产品配置可能分别采用 replicated / non-replicated attention，历史曾看到 48 / 96 stage 链。stage 数、实际 collective 数和 LateBind 命中 site 数都要从当前日志数，不能固定写死为层数的某个倍数。[S1][E1]

单卡 draft 较少付出五卡同步，但 draft 的 hidden/token 往返仍有成本。把 draft 改为五卡以求全设备直连，同时改变了模型布局和 sampling 能力，不是免费删除一次回读。当前 device-hidden 门要求 target/draft 使用相同顺序的五 rank、tensor split、RELAY/F32；不得与默认单卡 draft 混配。[S1]

### 4.2 MMVQ × MTP：复用收益与接受率损失必须同时计入

MMVQ 在这里指 activation Q8_1 + 量化权重收缩，**不是重新量化 GGUF**。disabled 仍可使用原生量化权重 GEMV、多列复用及融合内核，不等于把整个模型改成纯 F32 或逐列重复读取。[S2]

当前 `ggml_vk_should_use_mmvq()` 的重要选择顺序是：先检查类型与设备能力；再按 forced/disabled；auto 排除非 Intel 的 Q6_K；其余受支持类型在 n≥8 时允许；窄列 k<2048 不选；AMD 的窄列 Q8_0 偏好仅 GCN，不是 RDNA2。这个门使用实际 lowering 后的 n/k，不是用户配置的 h。融合 attention 的内部 selector 还可能看到 n=1。[S2]

一次 h1 请求有两行验证，不等于每个矩阵都在 MMVQ 多列快路。记录 `role、phase、权重类型、M/N/K、mat_vec_cols、outer_batches、融合区域、实际 pipeline`。small-Q8 可以把列改成外层 batch；先解释形状，不要为制造 MMVQ 命中而把已合格的原生路径关慢。

量化输入会改变 target logits，也可能改变 MTP 接受长度。投机验证只能相对于**实际使用的 target**工作；它不能恢复被 target activation 量化改掉的分布。原始 speculative decoding 的分布保持结论有算法与 target 不变等前提，不是本 fork 四开关任意组合的自动正确性证明。[R3]

### 4.3 LateBind 的数学机会在哪里

令 p 为 GPU rank，b 为 HC stream，当前 token 的全局 activation 为 `Y=Σ_p z_p`。简化写作：

```text
x_b       = R_b + s_b Y
D_b       = W_down,b · diag(gamma_b)
Q_p,b     = D_b (δ_p0 R_b + s_b z_p)
Σ_p Q_p,b = D_b x_b
rho_b     = 1 / sqrt(||x_b||² / width + eps)
u         = (1/4) Σ_b rho_b · Σ_p Q_p,b     （本模型四流）
下一步     = SiLU(u) → W_up / fold
```

线性部分可以在 Y 收齐以前计算，RMS 所需的 rho 晚一些到。这是 LateBind 的机会，不是消灭全部 allreduce。Q 本身的计算、发布、归约、LO 合成及同步都收费。收益须由关键路径的实际缩短覆盖这些新增成本。[S3][S4]

exact 指 Q 不额外走 Q8 activation 网格，不表示整条图逐位不变。浮点归约重排会影响末位。aggressive 的 LateBind Q8_0 与 MMVQ 的 Q8_1 是两个独立的量化位置；不能说“关闭 MMVQ 就关闭了所有激活量化”。[S2][S4]

RMS、SiLU、量化都不是一般可交换的线性运算。特别是 `||Σ z_p||²` 不等于 `Σ||z_p||²`。L2 统计量路线可利用：

```text
||R_b+s_bY||² = a_b + 2s_b d_b + s_b² c
a_b=||R_b||²，d_b=<R_b,Y>，c=||Y||²
```

其中跨 rank 交叉项不能丢。当前 L2=td 在 ACT_Q8/Q8DOT 携带统计量，由 relay 合成 rho、u、SiLU 与 LO Q8，UP 从 bank 消费；它不是旧 `GGML_TP5_LATEBIND_RHO` 实验的别名。L2 省掉部分 GPU consumer/fanout，又增加 CPU 工作，必须重测。[S3]

### 4.4 LateBind × MTP：有效行数是协议，不是优化提示

`capacity_rows` 是允许的上限，`active_rows` 才是本次执行的有效行。EOS、预算不足、draft 全拒绝、零提案、catch-up 和相位切换都会改变它。

10-08 修复记录中的真实反例是 `main_elems=5120，width=2560`，实际两行，却按 `late_count=1280` 只处理了一行 Q。当前代码按各 stage 的 Y extent 推导行数，并为下一 bank 发布下一 stage 自己的行数。不能删掉这些检查换取少一次分支。[S3][E1]

缓存身份必须包含实际 VkBuffer、offset、size，以及定义/计划/容量/命令的完整契约。两个同时存活的不同 VkBuffer 即使 offset/size 相同，也不是同一绑定。图形状近似、节点数相同、某个 `k==1` 都不足以授权复用。[S2][S3]

### 4.5 LateBind × MMVQ：不允许低层选择破坏高层契约

融合区域先取得自己的合法节点范围，再在区域内部选择浮点或整数收缩。没有合法 MMVQ pipeline 应回到该区域允许的原生收缩，而不是拆掉整个已合格融合以凑命中。[S2]

输入量化 scratch 的缓存键还须识别源内容更新：同 buffer、同 shape 不代表本周期 activation 没变。量化结果跨行/跨 epoch 复用必须有实际数据世代与所有权依据。源码/ISA 相同的重写不能算作新的优化；流量下降而时间上升也可能是寄存器、占用率或同步代价变大，不能只选一个计数器宣布成功。[S2][E2]

## 5. 不能牺牲的正确性底座

发布顺序必须维持 `producer → P1 发布主 Y → late_pre` 的有效依赖，使 CPU 尽早开始主归约。不能为了简化顺序把主发布放到耗时 Q 之后；也不能在 Q 尚未读完 z_p 时让 norm 覆写同一存储。[S3]

ready 标志不等于 host 已看到完整 payload。当前 Q 的 epoch、live extent、payload 校验与 L2 独立 tag 都有实际故障背景。coherent 内存不取消同步义务；XOR 校验可以捕捉某些破损，不能取代可见性契约，也不是任意错误都能被检测的证明。[S3][R2]

两个 bank 的复用必须等真实读者完成。资源退休和异常退出按**实际 native submission / fence**排空，不能只看逻辑 epoch 或 allreduce 计数。部分 rank 已提交后的失败不可再 fallback 重放一遍。只有确定尚未提交且合法的回退才保留。[S3]

`GGML_GALLOC_KEEP_CHUNKS` 的 chunk 保留关系到 prefill/decode 切换后 replay 缓存是否连坐失效。换取释放显存的策略可单独评估，但不能把释放得更干净当默认性能改进。模型、KV、draft、scratch、两相位 chunk、两 bank 和 retired owners 都要计入 VRAM 峰值。[S5]

跨请求只清应清的状态。零前缀新请求不能继承上次 MTP carry；有效 cached prefix、其他 slot、sampler/RNG/grammar/checkpoint 又不能被全局重置破坏。退出信号、失败 drain、正常析构同样属于性能测试资格，而不是跑完分数后才管的杂事。[S6]

## 6. 单路 Decode 的极限：先建立可以证伪的模型

### 6.1 硬件账与理论账分开

本次机器盘点见单 socket Xeon E5-2699A v4，22 物理核/44 逻辑线程，AVX2；五个 AMD 设备当前 PCIe 链路均为 8 GT/s ×16。Vulkan rank 与 DRM card 编号必须另做映射，不能按数字猜。RX 6800 官方规格给出 16 GB 显存、最高 512 GB/s 显存带宽、128 MB Infinity Cache 和最高约 16.17 TFLOP/s FP32。[R1]

这些不是本机当前负载的可达带宽或算力。CPU 的 sysfs 最大频率、GPU 的 DPM 档位也不能代替实际负载吞吐。PCIe 端点宽度不说明全部上游链路可同时满速；记录 BDF、桥接拓扑和每 rank ready 分布，不用五张卡的标称值直接相加。

### 6.2 下界、工程预测、实测值要有三个名字

对阶段 s、rank r，先指定缓存边界和算术类型，写资源时间下界：

```text
t_mem(s,r)  = 必须经过该缓存/显存边界的字节数 / 该资源带宽上界
t_math(s,r) = 对应运算数 / 对应类型计算吞吐上界
t_local_lb = max(t_mem, t_math)
```

以不可避免的字节数和真正的资源上界计算，才可以叫理论下界。用实测有效带宽和当前流量计算的是**条件预测**；它可以指导优化，但不是全局极限。FP32 TFLOP/s 不能直接替代整数 dot、解包、地址运算的吞吐。Roofline 的作用是限制资源可能性，不是给所有 kernel 一个“峰值乘利用率”的万能系数。[R4]

TP5 阶段要看 `max_r t_local(s,r)`，不是五 rank 平均时间。整图下界是依赖图最长路径上的下界之和：

```text
T_target >= max_{依赖路径 P} Σ_{v∈P} t_lb(v)
```

只有相互串行且不重叠的边才能相加。host 等 ready、GPU 自旋、同一段 GPU timestamp 很可能在描述同一段等待。不能把它们加三次，也不能从长时间线中简单删去全部 wait 得到可实现极限。

通信的条件模型可以写为 `启动/可见性延迟 + 载荷/BW + CPU fold + fanout`，但只有暴露在关键路径上的部分才增加周期。小载荷通常更需要测延迟，而不是只看 PCIe GB/s。把一次 collective 的延迟乘以当前实测次数，再判断多少能重叠；不要拿旧链 96 stages 替代当前调用计数。

### 6.3 权重字节账怎么填

按 **role × phase × stage × rank × shape** 统计：权重真实 block bytes、参与的 MoE expert 并集、复制矩阵、KV 读取、激活、量化 scratch、状态快照、读回和写回。MTP 不同 token 选中不同 expert 时，验证多行的权重流量不一定是单行一次读取。

本仓保留的小型形状账给出：Q8_0 每 32 权重 34 字节，Q6_K 每 256 权重 210 字节；HC 的 10240×320 Q8_0 矩阵为 3,481,600 字节，五分片 Q6 vocab 的每 rank 49664×2560 为 104,294,400 字节。[E3]

仅作算术示例：假定后者完整从显存读一次，按 512 GB/s 得 `104294400/512e9=0.2037 ms`。这**不是词表 kernel 的实测时间或全模型下界**：缓存命中可能减少显存流量，重复加载可能增加流量，归约、解量化、采样与同步还未入账。同理不能把总 GGUF 文件大小除以五卡带宽当每 token 延迟。

该形状账的原记录混有 prefill/decode、只生成 16 token，融合标签还可能少算其他矩阵。其 logical bytes 不等于 VRAM counters，更不能除以另一场 171-token 输出外推吞吐。[E3]

### 6.4 MTP 的盈亏线

对每个生成周期 j 记录：draft 时间 D、target verify 时间 V、catch-up/rollback 状态维护 C、采样与未重叠 host 时间 H，以及公开提交 token 数 K。使用不重叠区间：

```text
T_cycle = D + V + C + H
t_mtp   = Σ T_cycle / Σ K
S_mtp   = t_single · E[K] / E[T_cycle]
盈利条件：E[K] > E[T_cycle] / t_single
```

`draft_accepted/draft_n` 不是 K。bonus、零提案、全拒绝、末轮 EOS、预算和被隐藏的内部 token 都会让它们不同。周期边界不齐时保留未归因残差，而不是强行让账相加为零。

仅在理想化的固定 h、无边界截断模型中：

```text
E[K] = 1 + Σ_{i=1..h} P(前 i 个 draft 全部被接受)
若条件接受率恒为 α：E[K] = 1+α+...+α^h
```

这不是把实测总接受率直接代入任意深度的许可。需要保存接受前缀长度分布，并按实际有效行数分桶。h1 的简化盈亏线为 `α > T_cycle/t_single - 1`。

假设单行 18 ms，h1 每周期 29 ms，则至少需要约 1.611 个提交 token/周期；理想 h1 对应 α>0.611。若 MMVQ 把周期减到 27 ms，却让 K 从 1.75 降至 1.55，则从 16.57 变为 17.42 ms/token，kernel 更快而产品更慢。这是演算，不是本机新增测量。

### 6.5 60、80、100 tok/s 分别要求什么

| 目标 | 单个公开 token 时间 | 若每周期平均提交 1.75 token，周期预算 |
|---|---:|---:|
| 60 tok/s | 16.667 ms | 29.167 ms |
| 80 tok/s | 12.500 ms | 21.875 ms |
| 100 tok/s | 10.000 ms | 17.500 ms |

这张表是预算，不是承诺。以 60 tok/s 为假设起点，达到 100 tok/s 要消除约 40% 总时间；一个占 5% 的环节即使免费，整机最多约 `1/(1-0.05)=1.0526×`。用 Amdahl 上限先筛选工单，避免再花一轮只削低于噪声的小项。

理论最高速度至少需要完整的逐 rank 必需流量、实际依赖图、不可隐藏的通信、验证行分布和接受率约束。目前资料不足以为当前 dirty 源码填齐这些量；正确输出是**带假设的区间和缺测项**，而不是把历史最高分叫硬件极限。

配套 [scripts/tp5-perf-model.py](scripts/tp5-perf-model.py) 只做上述周期、盈亏线、Amdahl 与预算计算，不启动 GPU，也不会替缺失输入编数字。

```bash
# 下列输入是本节演算假设，不是机器采样值。
python3 scripts/tp5-perf-model.py cycle --single-ms 18 \
  --draft-ms 3 --verify-ms 25 --other-ms 1 --committed 1.75 --horizon 1
python3 scripts/tp5-perf-model.py budget \
  --target-tps 100 --committed 1.75 --baseline-tps 60
python3 scripts/tp5-perf-model.py amdahl --fraction 0.05 --eliminate
python3 scripts/tp5-perf-model.py resource --bytes 104294400 --bandwidth-gbs 512
```

计算结果统一标记 `CALCULATION_NOT_BENCHMARK`。把真实周期账代入时，各项必须来自同一批周期，且没有重复计时。`--committed` 是实际公开提交 token/周期，不是 draft 接受率。

### 6.6 RERoT 关闭，CPU 线程数是可调因素

本任务固定单路请求、RERoT 关闭，不要求 `-t 1`。本轮对照先用 `-np 1 -t 22`；线程数、draft、attention 布局和上下文都可作为明确记录的实验因素。不能只给控制臂限制 CPU 线程来制造加速。进程线程清单可以用 `ps -L -p "$SERVER_PID" -o pid,tid,psr,pcpu,comm` 核对；记录服务 PID 时注意它不是外层 flock 的 PID。

线程敏感性可比较 `-t 4/8/22/44`，其他参数与 workload 不变。它不替代 relay 暴露时间和 GPU 依赖链分析，也不应把整个进程绑到一个物理核。目标是用户实际拿到的 token/s，不是人为的单核竞赛。

## 7. 可复制的配置表

### 7.1 八个 LateBind 家族

每臂从干净环境开始。表中“无”表示不传入该变量，不是把所有旧变量都赋值 0；有些旧开关按存在性判断。[S1][S3]

| 家族 | `--tp5-latebind` | 额外环境变量 |
|---|---|---|
| off | off | 无 |
| exact | exact | 无 |
| aggressive | aggressive | 无；合格路径默认 gtt-push |
| q8first | aggressive | `GGML_TP5_LATEBIND_Q8_ORDER=first` |
| pack-once | aggressive | `GGML_TP5_LATEBIND_PACK_ONCE=1` |
| ap | aggressive | 上面两项同时设置 |
| l2 | aggressive | `GGML_TP5_LATEBIND_L2=td`、`GGML_TP5_LATE_TRANSPORT=gtt-push` |
| l2-ap | aggressive | l2 加 q8first、pack-once |

不要把 `GGML_TP5_LATEBIND_RHO`、ZSNAP、P1A、FUSED_FINALIZE 等历史诊断开关混入这些家族。不要主动设置 `GGML_VK_HC_DOT` 来再加一层加速：它可能改变匹配与实际执行臂。L2 的 `...L2_CHECK=1` 只用于数值门，正式速度测量不带它。

### 7.2 三个 MMVQ 策略

| 策略 | `GGML_VK_DISABLE_MMVQ` | `GGML_VK_FORCE_MMVQ` |
|---|---:|---:|
| disabled | 1 | 0 |
| auto | 0 | 0 |
| forced | 0 | 1 |

两个值都显式给出，避免 TP5 默认注入或继承环境改变被测臂。forced 仍不越过类型、输入、对齐和 pipeline 合法性门；启动日志写 forced 也不能替代 target-verify kernel 命中证据。[S1][S2]

### 7.3 基础启动配方

在仓库根目录执行。先设置本机实际路径，不要照搬历史日志中的可执行文件。以下是启动骨架，不是已验收的最优配置；上下文、batch、KV 类型须与待比较臂保持一致。

```bash
export BUILD="$PWD/build-perf"
export MODEL="/absolute/path/to/target-first-shard.gguf"
export MTP_MODEL="/absolute/path/to/matched-mtp.gguf"

# 构建与 GPU 测量分开进行；已有合格构建可直接令 BUILD 指向它。
cmake -S . -B "$BUILD" -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" -j 8

# 核对当前二进制的选项与五张物理卡映射。
"$BUILD/bin/llama-server" --help
"$BUILD/bin/llama-server" --list-devices

# 参考臂：RERoT 关闭、单路、主 CPU 计算线程 22、MTP 深度 1。
# 此锁在整个前台服务生命周期内持有；不要同时跑另一组 GPU 实验。
flock -n /var/tmp/tp5-gpu.lock env -i \
  PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 RADV_DEBUG=nobolist \
  GGML_VK_DISABLE_MMVQ=1 GGML_VK_FORCE_MMVQ=0 \
  "$BUILD/bin/llama-server" \
  -m "$MODEL" -md "$MTP_MODEL" \
  -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4 \
  --split-mode tensor --tp5 qwen4exp-af --tp5-sync relay --tp5-wire f32 \
  --tp5-latebind off --spec-type draft-mtp --spec-draft-n-max 1 \
  --fit off -ngl 999 -np 1 -t 22 -c 4096 -b 512 -ub 512 \
  --host 127.0.0.1 --port 8600
```

只有确认上述 Vulkan0..4 对应五张 RX 6800 后才能使用设备列表。动态库解析应由构建的 RPATH 或明确记录的库路径保证；不要无意继承另一个 build 的 `LD_LIBRARY_PATH`。若驱动/安装需要额外环境项，明确加入白名单并记录，不从整个 shell 继承。

无 MTP 对照删除 `-md` 和 draft 参数，改 `--spec-type none`；其余固定，记录因产品默认引起的 attention 布局差异。aggressive 候选只改该 CLI；auto 候选只改两个 MMVQ 环境值。只测一个因素时，不能顺便改变模型、采样、线程、上下文或 draft 卡。

### 7.4 最小自然请求

在另一个终端对这一个已持锁的服务发送请求，不再启动第二个 GPU 服务：

```bash
curl --fail-with-body --max-time 180 http://127.0.0.1:8600/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。"}],"temperature":0,"seed":42,"max_tokens":400,"cache_prompt":false,"chat_template_kwargs":{"enable_thinking":false},"stream":false}'
```

这个调用只检查服务，不产生性能资格。正式 runner 必须保存原始响应、单调时钟起止、timings、usage、停止原因、完整文本 SHA256、该请求是否 warmup，以及退出状态。原始输出不删词、不去 tag 后重新判定 EOS。

### 7.5 现有 harness 的使用边界

`scripts/run-tp5-cross-matrix.py` 仍有维护价值，但默认是 count、旧 golden 对照和旧路径；自然模式还依赖保留的 `weightloss-variants.json`。它适合**已有 oracle 覆盖的回归**，不是新模型/新量化的通用语义验收器。[S7]

```bash
python3 scripts/run-tp5-cross-matrix.py --list-variants
python3 scripts/run-tp5-cross-matrix.py \
  --variant-a qmp-b0-h1-off --variant-b qmp-b1-h1-auto \
  --workload weightloss --bin "$BUILD/bin/llama-server" \
  --model "$MODEL" --mtp-model "$MTP_MODEL" --dry-run
```

dry-run 必须显示 NOT_MEASURED。该对照固定 LateBind off，只检验 MTP h1 内 MMVQ 因素。新合法输出被旧 oracle 拒绝时，先保存并独立审查，不自动扩白名单。八家族自由生成资格未迁入可访问的受维护 runner 前，不把历史一次性脚本当生产入口。

## 8. 实验顺序：从资格到归因，再到选优

### 第一步：冻结身份与机器状态

记录 HEAD、dirty diff 摘要/校验和、未跟踪源码、CMake cache、编译器、shader 编译器、Mesa/RADV、内核、server 与**实际映射 DSO**、模型全部分片及 draft SHA256。记录五卡 BDF/rank 映射、PCIe 拓扑、温度/功耗/频率观测、最近 reset/wedge 事件、CPU 亲和性与 governor。

HEAD 相同不代表 binary 相同；binary 相同不代表动态库相同；包里有 identity 文件不代表本次重新核对过。禁止同时编译和测 GPU，禁止为了通过测试去改 watchdog、复位保护、超时或时钟。

### 第二步：正确性资格矩阵

先做 off/disabled 无 MTP 与 h1 的冷/暖、重复请求、正常退出。再逐家族加入 exact/aggressive 等，分别覆盖 disabled/auto/forced。随后测 h2/h3/h4/h8/h16 的容量边界与实际行数变化，不能只测满载前缀。

每个配置至少覆盖零前缀新请求、保留前缀、prefill↔decode、两 bank 极性、全拒绝、部分接受、零提案、短预算与自然 EOS。数值检查使用同输入/同位置，检查中间张量、固定前缀 logits、有限值和独立参考；L2 rho 的 `1e-6` 门只证明所测标量范围，不证明全 logits 或文本质量。

别把八家族当成需要淘汰到只剩一个才算正确。实现故障应该修；性能选择才允许某个合法配置更慢。协议失败、GPU fault、非正常退出立即中止本轮，不能只把该臂标低分继续压测。

### 第三步：三张账与 A/A

第一张是请求账：输入/输出、时间、质量、身份。第二张是周期账：D/V/C/H/K 与接受前缀。第三张是执行账：逐 rank/stage 的纯计算、暴露等待、通信、kernel 命中、字节/计数器和资源峰值。

先跑交错 A/A 验证同配置漂移；插桩开关也要做同配置对照。历史 ±3–5% 只是某轮分辨率，不是这台机器永恒的测量下限。若 A/A 漂移大于候选预算，先改进阻塞/配对设计和环境控制，不先写新 kernel。

### 第四步：最小交互实验

固定一个 LateBind 家族，先做四格：

| | MMVQ disabled | MMVQ auto |
|---|---|---|
| 无 MTP | A0 | A1 |
| MTP h1 | B0 | B1 |

以同一延迟尺度 L 计算 `I=(L_B0-L_B1)-(L_A0-L_A1)`。I>0 只表示 MMVQ 在 MTP 内省得更多，不能单独证明组合最快。组合仍须赢过独立确认的最佳合格无 MTP 产品配置。

再比较 off 与 aggressive 两个家族的四格结果，最后沿 h 轴扩展。机制归因实验固定 token 轨迹、实际行数和布局；产品实验允许各自合法默认布局，但必须披露差异。不要把这两类实验混称单因素。

### 第五步：筛选与独立确认分离

筛选用小样本选候选，不用于最后显著性结论。正式确认重新启动独立进程块；两臂用 ABBA/BAAB，三臂用预先平衡轮序。多个请求嵌套在同一进程内，先形成块统计。对每块 latency ratio 取 log，再报告几何均值及 95% CI；小块数要用合适的 t 区间或预注册的块 bootstrap，不能一律套 1.96。

速度比与延迟比方向相反。延迟比 [l,u] 对应速度比 [1/u,1/l]，不要只翻中心。CI 包含 1 表示未检出差异，不表示已证明等价；非劣需要事先给出非劣界。遍历 131 配置后最好的 screen 值有选择偏差，必须另做确认。

同一答案变短可以提高请求吞吐，却不代表每 token 计算更快；语义退化或提前 EOS 更不能晋级。多提示总分须预先规定聚合方式，且单独报告保护集回归，不只挑对自己有利的问题。

## 9. 按收益上限排工单，而不是按开关排

| 优先级 | 工作 | 立项证据 | 完成条件 |
|---|---|---|---|
| P0 | 统一当前八家族资格与身份 | 原始请求、退出、行数/绑定/发布反例 | 矩阵无未解释故障，最新证据进入受维护目录 |
| P0 | 稳定测量，闭合周期账 | A/A、逐 rank ready、插桩扰动、残差 | 候选预期收益大于分辨率；计时不重计 |
| P1 | h1/h2 与选择性 MMVQ | target-verify 实际 pipeline、K 分布、量化耗时 | 减少整周期每交付 token 成本，质量通过 |
| P1 | 最慢 rank 与复制算子 | 持续 rank skew、路由/HC/PLE/vocab 分项 | 固定拓扑下真正缩短关键路径 |
| P2 | LateBind LO 合成与发布重叠 | 当前 L2 CPU 段暴露时间、Q/ρ 依赖 | 少付出的时间大于新调度、传输与占用率成本 |
| P2 | 小 kernel 融合 / 权重重用 | 热点实际时间、ISA、VRAM/L2 counters | 改善真实形状，不只缓存驻留微基准 |
| P3 | GPU 自主通信 / 多步设备 draft | 跨设备可见性与采样/布局机制已成立 | 独立设计评审、故障边界和端到端资格 |

对每个工单先写五句话：当前瓶颈多少毫秒；最多能省多少；改哪个机制；用哪个不削弱的对照；什么结果算失败。没有这五句，先不改生产代码。

历史负结果不是永不重开的禁令，但重开须有新机制、新源码、新形状或旧归因被反例推翻。Q6 subgroup/row tile、只改 GLSL 不改 ISA、移除 barrier 却提高寄存器压力等旧尝试，先查保留 decision，避免原样重复。[E2]

## 10. 常见异常按这个顺序查

| 现象 | 先查 | 不要做 |
|---|---|---|
| 突然掉速，所有臂一起慢 | GPU reset/wedge、慢 rank、实际 DSO、A/A | 仅凭 sysfs 频率给某个补丁定罪 |
| auto 开了却没快 | 实际 role/phase/shape/pipeline、AMD Q6 门、small-Q8 lowering | 把启动日志当 kernel 命中 |
| 第一次快，第二请求坏 | 零前缀 MTP carry、cache/epoch、buffer 身份、bank 退休 | 全局清全部 slot 状态掩盖问题 |
| prefill/decode 后反复回退 | chunk free 引发 cache invalidation、完整定义、活动行数 | 放宽绑定或伪造 replay 命中 |
| exact 也输出退化 | 同输入 logits、Q 发布与 extent、alias、归约误差 | 看到 exact 就免除质量检查 |
| L2 scalar 对拍好但文本差 | 全路径 logits、量化、状态、采样和语义 | 把 rho 对拍偷换成整模型等价 |
| draft 接受率更高却更慢 | D/V/C/H/K 全账、行数与 expert 并集 | 只优化 accepted/drafted |
| kernel 更快但请求更慢 | 覆盖比例、接受长度、通信/同步、资源回退 | 拿微基准比例乘全请求 |
| SIGTERM 不退出 | 信号上下文、队列锁、实际 pending submission | 延长超时或强杀后当正常退出 |

## 11. 仓库结构与清理纪律

维护三个层次即可：本页放当前方法和路线；`scripts/`、`tests/` 放受维护程序；`artifacts/` 放必要的当前原始证据和紧凑历史裁决。清理说明见 [artifacts/README.md](artifacts/README.md)。不再每次追加一个“最终最终交接”。

新实验使用一个目录，包含 manifest、原始请求、机器可读汇总和短 decision。完整 stdout 只留定位故障必需的部分；被确认取代的构建日志、逐 stage 大日志与硬编码旧 binary 的一次性 runner 应清理，不再永久打包一份源树和库的副本。保留失败原因、条件和重开门，不必永远保留每次重复输出。

Git 历史可恢复的 tracked 旧文件可以删除；未跟踪的唯一成果不能假称在 Git 中可恢复。本次清理不使用 reset/clean/stash，不删除当前 dirty 核心代码或测试，不动模型、系统恢复服务、生产启动器。运行中的程序、被活跃 harness 引用的 fixture 和未合并 dirty worktree 都不是无用内容。

本次盘点只有 master 一个本地分支，三个旧 detached worktree 都有未提交修改，故没有安全可删的分支/worktree。不能为清理数字好看而强删。之后删除 worktree 要先确认无进程、无独有提交、无未提交内容，再使用 Git 的 worktree 管理命令；不直接递归删除目录。

## 12. 源码与证据索引

行号会漂移，按文件和符号定位。

| 编号 | 入口 | 查什么 |
|---|---|---|
| S1 | [common/common.cpp](common/common.cpp)，`common_tp5_apply_env`；[common/arg.cpp](common/arg.cpp) | 默认值、CLI/env 优先级、TP5/MTP admission |
| S2 | [ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp)，`ggml_vk_should_use_mmvq`、`ggml_vk_mul_mat_vec_q_f16` | MMVQ 类型与 shape 门、融合、scratch/replay |
| S3 | [collective.cpp](ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp)，`tp5_star_handoff`、`linear_prepared_for`、`tp5_latebind_l2_td_requested`；[rows.h](ggml/src/ggml-vulkan/ggml-vulkan-tp5-rows.h) | 发布、bank、行数、绑定与资源退休 |
| S4 | [LateBind shader](ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_latebind.comp)、[consumer shader](ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_resume.comp) | Q/RMS/LO 数学与当前工作组布局 |
| S5 | [ggml-alloc.c](ggml/src/ggml-alloc.c)，`GGML_GALLOC_KEEP_CHUNKS` | 相位切换、chunk 保留与缓存寿命 |
| S6 | [speculative.cpp](common/speculative.cpp)、[server-context.cpp](tools/server/server-context.cpp)、[server-signal.h](tools/server/server-signal.h) | 接受/回滚、跨请求 carry、退出边界 |
| S7 | [cross-matrix runner](scripts/run-tp5-cross-matrix.py)、[tests/CMakeLists.txt](tests/CMakeLists.txt) | 现有回归与其覆盖边界 |
| E1 | [TP5 历史账](TP5.md) 的日期节 | 10-07/08 修复、模型区别、性能声明的来源 |
| E2 | [MMVQ decision](artifacts/mmvq-mtp/run-20260928T1700/decision.md)、[组合实验账](MMVQ+MTP.md) | ISA、流量、资格与负结果 |
| E3 | [形状与逻辑字节账](artifacts/mmvq-mtp/run-20260928T1700/model-kernel-source-byte-rank-20260929.json) | 真实矩阵 shape；不是 VRAM 测量 |
| E4 | [10-07 保留证据索引](artifacts/mmvq-mtp/run-20260928T1700/takeover-qualified-search-20261007.proof.json) | archive 成员与旧验证声明；本次未重验包体 |

理论参考，核对日期 2026-10-08：

- R1：[AMD RX 6800 官方规格](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html)。
- R2：[Khronos Vulkan 同步与缓存控制](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)。跨设备设计还须明确所使用的外部内存与同步扩展，不能把单设备 scope 当跨卡保证。
- R3：[Leviathan、Kalman、Matias，Fast Inference from Transformers via Speculative Decoding，ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)。
- R4：[Williams、Waterman、Patterson，Roofline，Berkeley 技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)。

最后的判断准则只有一个：**在相同质量与明确资源约束下，用可复核的证据证明，每个实际交付 token 更快了。**
