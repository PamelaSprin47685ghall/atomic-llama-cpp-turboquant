# TP5 / LateBind / MMVQ / MTP：理论速度、实测速度与差距

更新提示（2026-10-09）：下文 `mtp-strong` 的四对独立确认没有完成。实际只完成第一对，第二块 B 加载时出现 `RELAY previous GPU wait failed on rank 0` 并退出 1；不得据此宣布 MTP 显著优于或劣于强无 MTP 对照。后续另行注册、同 APEX 权重同分片无 MTP 的 LateBind 四对比较已完成，见 [10-09 报告](../run-20261009T0040/REPORT.md)。本页保留原始实验来源，不把探索数据改称最新确认。

本轮目标：RERoT 关闭，普通单路生成，不限制 CPU 为单线程。实验目录使用 UTC 日期 2026-10-08；日本时间为 2026-10-09。

这里的实测来自本目录新构建、新启动的真实五卡服务，不沿用旧战报。理论数值是明确假设下的权重流式带宽参考，不是已经达到的速度，也不是任意算法的绝对上限。

## 1. 已经得到的结果

四个独立配对进程块确认：原 GSQ-RCO-Q2_0 的 MTP1 基线为 **57.17 tok/s**；只将 HC 权重转成 Q8、实际启用 aggressive LateBind 后，为 **62.28 tok/s**。配对速度比为 **1.08949，95% CI [1.06529, 1.11423]**。这项组合变化有当前短提示上的吞吐证据。

它不是“原模型算法逐位不变的 9% 提速”。HC 权重精度、实际 kernel 路线和输出都改变了。原回答 156 token，新回答 167 token；平均生成时间 2728.81 → 2681.48 ms，完整请求墙钟 3896.69 → 3886.89 ms。完整回答墙钟没有显著改善。不能把归一化 tok/s 的增益直接解释成回答同一道题快了 9%。

本轮另一项源码改动是完整命令定义缓存。它让连续请求的定义重录由每请求 6 次降为 0；**prefill 时间减少 5.34%**，但 Decode 速度比的 95% CI 包含 1，不能宣称 Decode 提速。默认仍关闭。

## 2. 理论 tok/s 与实际 tok/s

AMD 官方给 RX 6800 的标称显存带宽为 512 GB/s、Infinity Cache 为 128 MB。[外部 1] 本轮还实测了四个独立 64 MiB 输入与输出的 F32 流式读写，每卡工作集 512 MiB，超过该缓存容量。五卡有效读写带宽约 **390–403 GB/s**。这不是 GEMV 的实测读带宽，不把两者混为一谈。

| 模型与执行方式 | 512 GB/s 下的权重流式参考 tok/s | 代入本机流式读写带宽的敏感性范围 | 无插桩实际 tok/s | 实测资格 |
|---|---:|---:|---:|---|
| 原 Q2_0，无 MTP，默认 attention 复制 | 167 | 127–132 | 48.41 | 探索 |
| 原 Q2_0，MTP1，attention 分片 | 320–335 | 244–264 | **57.17** | 四块独立确认的原模型臂 |
| APEX，无 MTP，显式 attention 分片，LateBind aggressive | 274 | 209–216 | 64.19 | 强无 MTP 对照探索 |
| APEX，MTP1，LateBind aggressive | 337–365 | 257–287 | 65.13（MMVQ disabled）；65.43（auto） | 探索；差异不是独立 MMVQ 收益证据 |
| HC-Q8 衍生模型，无 MTP，显式 attention 分片，LateBind aggressive | 345 | 263–272 | 63.38 | 探索 |
| HC-Q8 衍生模型，MTP1，LateBind aggressive | 390–414 | 297–326 | **62.28** | 四块独立确认；不是综合质量认证 |

APEX 的强无 MTP/MTP1 独立配对确认单独保存在 `mtp-strong-b*-*/`，其最终统计应与上述探索数区分，不混成一个样本池。

### 理论值怎么得到

记每周期真正公开提交的 token 数为 k，阶段 s、rank r 需要读取的权重字节为 B(s,r)，对应带宽为 beta(r)。理想化成本为：

```text
target_weight_ms = 1000 × sum_s max_r [ B(s,r) / beta(r) ]
draft_weight_ms  = 1000 × B_draft / beta_draft
ideal_tok_s     = 1000 × k / (target_weight_ms + draft_weight_ms)
```

上式用一致的字节/秒单位；计算器内部 GB/s 按十进制处理。逐阶段取最慢 rank，不把五张卡的执行时间相加，也不把总模型大小简单除以五卡带宽。Roofline 本来就是资源需求与资源吞吐之间的条件模型，而非某个软件的完成时间保证。[外部 2]

原 Q2_0、MTP1 的 target 权重流量参考为每周期关键 rank 约 **2.084–2.216 GB**，单卡 draft 约 **0.623 GB**。512 GB/s 对应 target **4.070–4.329 ms**，draft **1.216 ms**；每周期公开产出 **156/88 = 1.7727 token**，因此约 **320–335 tok/s**。

HC-Q8 衍生模型的 target 对应 **1.484–1.617 GB**，draft 不变；每周期公开产出 **167/98 = 1.7041 token**，得 **390–414 tok/s**。

范围来自两行 verify 的专家集合：同一层每行激活 10/512 个专家，两行完全重合与完全不重合分别按 10 与 20 个专家计算。非专家权重假设两行完全复用。HC/router 的复制、QSA 的 Q/K 复制、GDN 的 Q/K 复制按当前 plan 几何计入，不假装所有权重均分为五份。

28.8 GB 的 PLE 表按本模型 16 个索引行读取，不按每 token 扫完整表。它仍有 CPU 查表、缓存缺失、页错误或 I/O 成本；这些没有计入上面的权重参考。实际 GGUF 类型、大小和 tensor 目录见 `model-metadata.json`、`apex-metadata.json`、`hcq8-metadata.json`。

上述参考还没有包含激活读写、KV/recurrent 状态、解包指令、寄存器/LDS 压力、dispatch、barrier、GPU/CPU 接力与状态维护。尤其 LateBind 自己的 sidecar、打包副本和额外工作不是零成本。**不能把“参考 400、实际 62”的全部差额叫作已经证明可消除的软件浪费。** 实际跨 VRAM 边界的字节数和各算子的指令吞吐尚未完整计量。

## 3. 一个真正的根因：原 Q2_0 没有执行 LateBind

原文件名带 Q2_0，但它是混合精度 GGUF；HC down/up 主要为 BF16。当前 `ggml_vk_hc_segment()` 的 LateBind capture 要求相应权重是 Q8_0。

真实五卡日志证据：

```text
原 Q2_0 + --tp5-latebind aggressive:
mode=reference reason=no-late-tensors
late=0 q8_fast=0

APEX / 新 HC-Q8 模型 + aggressive:
mode=aggressive-q8
late=94 q8_fast=94
```

见 `q2-route-aggressive/server.log`、`apex-latebind-hit/server.log` 和 `hcq8-screen-h1-disabled-aggressive/server.log`。旧记录里“Q2_0 上打开 aggressive 更快”的时间差，不能直接归因于 LateBind；开关名不是命中证明。

本轮没有放宽类型检查硬跑 BF16，而是明确产生一个实验衍生模型：仅将 **290 个 HC down/up/inject 张量**用当前 GGML CPU 量化器转换为 Q8_0，其余 **934 个张量的源/目标字节逐一 SHA256 对拍一致**。原模型文件没有修改。总大小减少 **599,654,400 字节**。

每个变更权重张量的最大相对 L2 误差不超过 **0.00922**；这是权重误差，不是 logits 误差。转换脚本为 `scripts/tp5-hc-q8-variant.py`，native helper 为 `tests/tp5-hc-quant-helper.cpp`，完整转换证明为 `hcq8-conversion.json`。

同一新模型探索中，LateBind off 为 **60.13 tok/s**，aggressive 为 **64.10 tok/s**；但正式确认只证明“原 Q2_0/off → HC-Q8/aggressive”这一组合的收益，不把探索拆分结果冒充独立因果证据。

对应的 96-stage 定义中，rank0 的 dispatch 大约由 **4052 降到 2820**，barrier 由 **3490 降到 2446**。这是定义层命令数量，不代表每种 active-row 轨迹均执行相同工作，更不代表整体必然提速 30%。

## 4. TP5、LateBind、MMVQ、MTP 的交互

### APEX 八格探索

| LateBind | MTP | MMVQ disabled tok/s | MMVQ auto tok/s |
|---|---|---:|---:|
| off | 无 | 55.31 | 55.04 |
| off | 深度 1 | 60.69 | 60.52 |
| aggressive | 无 | 57.28 | 56.35 |
| aggressive | 深度 1 | 65.13 | 65.43 |

这八格使用默认产品布局：无 MTP 默认复制 attention，MTP1 默认分片并设置 headmap。它衡量的是产品配置交互，不是一个孤立 kernel 的因果收益。

补充更公平的控制后：无 MTP 也显式设置 attention 分片与 QSA/GDN headmap，APEX 已达 **64.19 tok/s**，HC-Q8 达 **63.38 tok/s**。所以不能用 57.28 作为“最强无 MTP”，再把差额全部算到 MTP 身上。强对照另做独立确认，不删掉这个打破原解释的结果。

### 接受率会吃掉一部分执行收益

HC 独立确认中，在 h=1、p_min=0 且实际计数满足公开 N = proposed + accepted 的请求上：

| 项目 | 原 Q2_0/off | HC-Q8/aggressive |
|---|---:|---:|
| 每请求公开 token | 156 | 167 |
| 每请求提案周期 | 88 | 98 |
| 每请求接受草稿 | 68 | 69 |
| 接受率 | 77.27% | 70.41% |
| 每周期公开产出 k | 1.7727 | 1.7041 |
| 由请求总生成时间核算的平均周期 ms | 31.009 | 27.362 |

周期缩短约 **11.76%**，每周期产出同时减少约 **3.87%**，最终 tok/s 净增约 **8.95%**。这个例子说明，核更快不保证净收益同样大；必须把分子成本和分母产出一起算。

MTP 验证的是当前 target 的数值路径；当 target 本身的权重或激活量化改变时，不能指望 MTP 把它自动还原为原 target。[外部 3 的分布保持结论以既定 target 为前提。]

### 深度、MMVQ 与线程的实际选择

原 Q2_0 的自然文本探索：h1 **56.94**、h2 **49.42**、h4 **41.40 tok/s**。同一目标下，增加提案深度没有换来足够的公开产出；不能用计数题的高接受率代替自然文本。

MMVQ 对 Q2_0 h1 的 disabled/auto/forced 为 **56.94/55.74/57.49**。APEX aggressive+h1 的 disabled/auto 为 **65.13/65.43**。这些小差异尚无独立确认；同时全局策略会影响多种 role/phase，不把 auto 日志当作所有 target verify kernel 已命中的证明。

APEX h1 aggressive 的 L2 分支为 **60.42 tok/s**，低于非 L2 探索的 **65.13**。本轮保留 aggressive 作为候选，不把“选项更多”当作“更快”。

同一 APEX h1 aggressive/disabled，CPU 线程 4/8/22 的探索为 **63.39/63.60/65.13 tok/s**。这不是 CPU 必须单线程的任务，也不是已经证明 22 是全局最优。

## 5. 剩余 GAP 逐项算账

吞吐差不能相加，统一换成每个公开 token 的毫秒。

### 轻量周期诊断：同一次请求内相加

HC-Q8/aggressive/MTP1 的一次实测诊断请求：167 token，生成时间 **2584.268 ms**。不是正式确认分数，详见 `hcq8-light-profile/request-2.json` 和相应日志。

| 非重叠一级账 | 请求总 ms | ms/公开 token | 占生成时间 |
|---|---:|---:|---:|
| draft | 239.815 | 1.4360 | 9.28% |
| target 验证 | 2300.945 | 13.7781 | 89.04% |
| catch-up / 状态维护 | 37.085 | 0.2221 | 1.44% |
| 一级账残差 | 6.423 | 0.0385 | 0.25% |
| 合计 | 2584.268 | 15.4747 | 100% |

这里的 target 是服务端记录的验证路径区间，包含相关 host 调度/处理，不是“纯 GPU ALU 时间”。它与下一段 GPU timestamp 的分类是两层账，不能跨层直接相加。

另一次同配置诊断为 2780.221 ms，说明会话内仍有漂移；不只选最快诊断作为正式速度。`profiles.json` 同时保留全部样本。

**第一项：权重类型与路径没有命中。已经处理。** 原 Q2_0 的 BF16 HC 阻止 LateBind；实验 HC-Q8 模型让 94 个 site 真正执行。正式归一化耗时从 **17.4923 降到 16.0567 ms/token**，实际收回 **1.4356 ms/token**，但有精度和输出长度变化。

**第二项：相位切换重录。重录已消除，Decode 大缺口没有消除。** 新完整定义缓存按原有全部身份条件匹配两 bank 程序，容量至多四个闲置定义。实际 6→0 重录；prefill 时间比 **0.94659 [0.92709,0.96650]**。Decode 速度比 **1.01039 [0.97994,1.04179]**，不显著。代码开关 `GGML_TP5_DEFINITION_CACHE=4`，默认 0。没有放松 buffer、形状、容量、epoch、原生提交排空或资源寿命契约。

**第三项：首个 verify 仍异常长。尚未消除。** 轻量候选诊断首个 target 周期为 **227.89 ms**，普通周期中位约 **21.089 ms**；其差约 206.8 ms/请求，即 **1.24 ms/token** 的待归因预算。原 Q2_0 开完整定义缓存、重录归零后首轮仍为 210–240 ms。因此不能把它全部说成编译或重录，也不能未归因就宣称能全数收回。

**第四项：target 内部的计算、等待与通信。仍是大头。** 单条 GPU timestamp 捕获显示 HC-Q8/aggressive 某 rank 的 chain span 约 **24.47 ms**，其中 native compute 区间 **14.22 ms**、late_pre **4.56 ms**、P2 消费/等待区间 **4.38 ms**、P1 **1.26 ms**。P2 不是纯 PCIe 传输；compute 也不是纯算术。tail_compute 是包含于其他区间的细分，不能再加一次。不同 rank 时间不能累加成请求时间。这次只把大类区间辨认出来，尚未把完整模型的真实 VRAM 事务、解包、LDS/寄存器和每条 barrier 分开闭合。

**第五项：draft 与状态维护。不是主要缺口。** 在上面的同一次诊断里，它们合计约 **1.66 ms/token**；哪怕不切实际地全删，也无法把当前约 62 tok/s 变为 100。降低 draft 数值成本还可能降低接受率，净收益要重新算 k。

**第六项：TP5 的复制/分片布局与 MTP 绑在一起。已经补上强控制。** 无 MTP 的默认复制布局本来偏慢。显式分片后 Q2_0 的无 MTP 从 **48.41 到 51.51**，APEX/aggressive 从 **57.28 到 64.19**，HC-Q8/aggressive 从 **56.13 到 63.38 tok/s**（均为探索）。它减少权重冗余，但仍需承担更多交接边界；不能仅用带宽公式决定布局。

### 距离 100 tok/s 还有多少

按正式 HC 候选 **62.28 tok/s**，当前约 **16.06 ms/token**，100 tok/s 需要 **10 ms/token**，还要减少 **6.06 ms/token，即约 37.7% 的总生成时间**。

本轮已收回的是约 1.44 ms/token，而不是全部差距。100 没有达到；320–414 的权重参考更没有达到。要继续缩短时间，优先级是 target 的真实瓶颈与首轮长延迟，而非继续堆草稿深度或把所有环境变量打开。

把上述同一次 2584.268 ms 的诊断作为算术起点，还可得到更贴近当前实现的条件预算：

| 假设能够完全消除的时间 | 条件估算 tok/s | 不是实测的原因 |
|---|---:|---|
| 只消除首周期相对普通周期的 206.801 ms 超额 | 70.24 | 首轮超额尚未完整归因，不保证可全部消除 |
| 只把 draft 与状态维护的 276.900 ms 全部删除 | 72.38 | 真实生成不能免费获得草稿和状态维护 |
| 上面两项都全部删除 | 79.50 | 两个过于乐观假设同时成立仍不够 100 |

这些不是新硬件上限，更不是已经做到的速度。它们只说明：要跨过 100，普通 target 周期本身还必须明显变短；“再省一点 host 开销”不是充分方案。

## 6. 运行与复核

所有真实生成都使用五张物理 RX 6800，`flock /var/tmp/tp5-gpu.lock` 独占，不同时编译、转换或进行模型全量哈希。没有改时钟、功耗、watchdog、驱动、PCI 拓扑，没有重置 GPU。服务只终止自己创建的 PID；所有正式确认均正常退出，无强杀，测量期 fallback=0。

统一负载为 31-token 的规范提示、temperature=0、seed=42、关闭 thinking、cache_prompt=false、自然 EOS。固定 400-token 预算是保护上限，不把预算停止作为规范提示的成功。每个正式进程预热 2 次、测量 3 次，另跑不计分的哈希表/数组保护题。

实际构建为 `build-perf-gap/`，Ninja、Vulkan、RelWithDebInfo，显式 O2；HEAD 为 `9be06d5ab`，含接手前未提交代码，不能把 HEAD 单独视为全部二进制身份。每臂 manifest 与实际映射的 inference DSO SHA256 都保留。原 Q2_0 与 draft 已全文件计算 SHA；APEX 的本轮短实验保留元数据、文件 stat 与路径身份，不冒充已新做六份文件的全量哈希。

关键入口：

```text
scripts/tp5-gap-bench.py          真实服务逐请求证据与受控退出
scripts/tp5-gap-report.py         配对统计、周期账、唯一输出汇总
scripts/tp5-roofline.py           可复算的权重流式模型
scripts/tp5-gguf-metadata.py      GGUF 元数据和 tensor 目录
scripts/tp5-hc-q8-variant.py      显式 HC-Q8 衍生模型
```

例如，复测已生成的 HC-Q8 候选：

```bash
python3 scripts/tp5-gap-bench.py \
  --bin build-perf-gap/bin/llama-server \
  --model models/perf-gap-q2-hcq8/Q2-HCQ8-00001-of-00002.gguf \
  --draft "$HOME/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" \
  --horizon 1 --latebind aggressive --mmvq disabled --threads 22 \
  --warmup 2 --repeats 3 \
  --out "artifacts/perf-gap/recheck-$(date -u +%Y%m%dT%H%M%SZ)"
```

这会进行新的实际 GPU 实验，不是 dry-run；输出目录必须不存在。不把一次复测当作新晋级证据。

## 7. 没有隐去的边界

这些是短提示性能结果，不是所有上下文、提示和采样配置的最优证明。正文连贯性及已发现的事实错误见 `OUTPUT-REVIEW.md`；原基线的数组索引访问解释存在错误。部分近似路径冷暖输出不同，本轮未完成根因归类；没有把 warmup 输出删除。

新模型只保证未修改张量字节一致，不保证全 logits、困惑度或多领域质量等价，因此不默认替换原模型。数值误差阈值、超时与正常退出要求没有为跑分放宽。

初始 CPU/control 检查有一项已有失败：`test-mtp-workspace` 的 `prefill_phase == 1` 与当前 `ubatch_execution_phase()` 行为不一致。本轮未改该断言或函数来把测试涂绿。其他六项通过；新的完整定义缓存另过真实五卡身份/保活检查，完整 Vulkan replay 的 29 项逐项通过。最终命令及状态应查同目录验证日志。

没有 commit/push，没有声称工作树干净。性能改动、实验模型与报告均留在当前工作树，原有未提交工作保留。

## 外部资料与本地证据

外部 1：AMD RX 6800 官方规格（512 GB/s、128 MB Infinity Cache、16 GB）。
https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-6000-series/amd-radeon-rx-6800.html

外部 2：NERSC，Roofline Performance Model。
https://docs.nersc.gov/tools/performance/roofline/

外部 3：Leviathan 等，Fast Inference from Transformers via Speculative Decoding，arXiv:2211.17192。
https://arxiv.org/abs/2211.17192

本地数字的依据是本目录原始响应、`summary.json`、`profiles.json`、`*-roofline.json`、`stream-*.json`、转换证明和两份预注册文件；外部资料不为本机测量结果背书。
