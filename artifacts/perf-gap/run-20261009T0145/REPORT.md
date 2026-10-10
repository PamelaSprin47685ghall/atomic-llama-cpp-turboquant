# 预测下压、实测上增：TP5 普通生成的闭账与实现

日期：2026-10-09。RERoT 关闭，普通单路生成，22 个 CPU 计算线程，五张 RX 6800。
模型仍为未修改的 APEX-I-Compact；attention 分片与 headmap 保持开启，主对照
不使用 MTP，MMVQ disabled，LateBind aggressive。未改变时钟、驱动、功耗、
看门狗或同步/数值容差。

## 1. 已经确认的结果

| 项目 | A：原普通采样 | B：紧凑 top-k 采样 |
|---|---:|---:|
| 四对独立进程合并 Decode | 64.77485 tok/s | **67.14624 tok/s** |
| 每公开 token 时间 | 15.43809 ms | **14.89287 ms** |
| 每篇回答平均生成时间 | 2717.104 ms | **2621.145 ms** |
| 每篇回答平均客户端墙钟 | 3901.996 ms | **3807.721 ms** |
| 每篇测量回答 token 数 | 176 | 176 |
| 测量期 fallback | 0 | 0 |

配对速度比 **1.036613，95% CI [1.026421, 1.046905]**，即 **+3.66%**，
区间 **[+2.64%, +4.69%]**；生成时间缩短 **3.53%**，客户端墙钟缩短约 **2.42%**。
实际每 token 收回 **0.54522 ms**。本轮与上一轮 65.17 不直接作跨会话因果比较。

| 配对块 | A tok/s | B tok/s |
|---|---:|---:|
| 1 | 64.5043 | 67.4352 |
| 2 | 64.9942 | 66.9845 |
| 3 | 64.6366 | 66.7917 |
| 4 | 64.9671 | 67.3779 |

实验顺序 AB、BA、AB、BA；每进程两次预热、三次测量、一题不排名的保护题。
统计单位是四个独立进程对，采用逐进程合并请求成本，再对配对 log-ratio 用
Student t（df=3）计算双侧区间，不把 token 当作独立样本。
源码、模型、实际映射 DSO 均记录；只有 `GGML_TP5_GREEDY_HOST=0/1` 是因素。

三道新题另做 A/B 对照，并用于冻结模型的外部验证。四组正式配对加这三组
新题共 **78 个真实请求、36 个测量请求、14 次正常退出**；逐请求正文全数
匹配，测量期没有回退。自动原始证据核查见 [raw-audit.json](raw-audit.json)，
九种唯一正文的阅读范围见 [OUTPUT-REVIEW.md](OUTPUT-REVIEW.md)。

## 2. “Match”到底匹配了什么

这里严格区分三件事：

| 层次 | 本轮数值 | 能说明什么 |
|---|---:|---|
| 理想权重流式参考 | 273.86 tok/s；代入已测流式带宽为 208.60–215.55 | 只计权重读取，不是完整实现的速度预测 |
| 冻结的实现成本预测，176-token 回答 | **66.64332 tok/s** | 分别计入首轮、稳态、同步回读与采样后的条件预测 |
| 四对独立、无插桩实测 | **67.14624 tok/s** | 当前限定配置的实际暖态成绩 |

预测生成时间 2640.925 ms，独立实测平均 2621.145 ms，预测时间误差约
**+0.75%**。不是把“273.86”直接改成“67.15”冒充硬件极限，也不是把实际总
时间减去理论之后剩下的量塞回模型。旧的权重参考仍保留原来的定义。

本轮达到的是**实现成本模型的留出验证 Match**：开测前登记每独立进程误差
不超过 5%，冻结后不重新拟合。七个未参加校准的候选进程全部通过，最大绝对
时间误差 **2.23%**。这不证明 GPU 已被榨干，更不证明整个硬件下界已收紧到 67。

### 2.1 先拆首轮与稳态，不能只用平均 token 成本

低扰动插桩只读 host 时钟，没有添加 GPU 等待或新的 GPU 命令。`decode` 是
`ubatch` 的父区间，父子不能相加；采样中的同步等待与 CPU 选择分别记录。
一次 176-token 请求实际包含 **175 次单行 Decode**：第一个公开 token 从
prefill logits 采样，发生在 generation 计时开始之前。

以三份单独校准请求为依据，冻结参数为：

| 不重叠区间 | 第一个单行 Decode+sample | 后续每个单行 Decode+sample |
|---|---:|---:|
| memory/setup | 0.1380 ms | 0.00284 ms |
| graph build/reuse | 2.0253 ms | 0.00583 ms |
| scheduler graph allocation | **104.5110 ms** | 0 |
| input update | 0.0230 ms | 0.01225 ms |
| graph_compute 调用区间 | **126.3393 ms** | **13.28747 ms** |
| ubatch 收尾 | 0.0060 ms | 0.00266 ms |
| Decode 父区间其余部分 | 0.2423 ms | 0.17979 ms |
| sample synchronize | 0.2717 ms | 0.25990 ms |
| sample select | 0.0863 ms | 0.08422 ms |
| 合计 | **233.6430 ms** | **13.83495 ms** |

`scheduler graph allocation` 包括 scheduler 分图与分配流程，不等于纯 malloc。
`graph_compute` 包含首次定义/录制、host 调度、relay 和 GPU 等待，**不是纯 GPU
执行时间**。这是可预测的一级成本账，尚不是全部 shader/指令的底层成本证明。

冻结公式为：

```text
T_generation(N) = 233.643 + (N - 2) × 13.834954  [ms], N >= 3
predicted_tok_s = 1000 × N / T_generation(N)
```

N 是实际公开回答长度；模型预测耗时，不预测回答内容或回答长度。它只适用于
该配置、暖态、首个标量图重建一次的普通 Decode，不适用于 MTP、任意长上下文
或冷启动。校准阶段未解释的 server 余量约 **0.24%**，保留在 JSON 中，**没有
作为经验修正项加入公式**。

### 2.2 真正未参加校准的新题

模型于 **02:15:23 UTC** 冻结，随后才启动四对确认与三个新题进程；校准与
验证使用不同进程。规范提示的四个候选进程均通过，下面额外检验新题和长度变化。

| 未用于校准的题目 | token 数 | 预测 tok/s | 实测 tok/s | 预测时间误差 |
|---|---:|---:|---:|---:|
| 数据库索引，三句话 | 93 | 62.3064 | **62.1699** | -0.219% |
| 线程与进程，六句话 | 149 | 65.7146 | **67.1766** | +2.225% |
| 排查程序变慢，八句话 | 172 | 66.5227 | **67.4267** | +1.359% |

没有重新选择题目、删除慢点、调整容差或回填常数。原始冻结模型与完整验证在
[frozen-model-B.json](frozen-model-B.json)、[heldout-validation.json](heldout-validation.json)。

### 2.3 再做一份不使用候选 B 计时的预测

为了避免“测了 B，再预测 B”的循环，又用**对照 A 的分段计时 + 独立 CPU
微基准**构造因果预测：其他区间保持不动，只从每个标量样本的采样阶段减去
`487.665 - 80.645 = 407.020 μs`。负值直接拒绝，不钳制到零。

| 新题 | 仅用 A+CPU 微基准预测 tok/s | B 实测 tok/s | 预测时间误差 |
|---|---:|---:|---:|
| 数据库索引 | 62.1971 | 62.1699 | -0.044% |
| 线程与进程 | 65.4798 | 67.1766 | +2.591% |
| 排查变慢 | 66.2564 | 67.4267 | +1.766% |

这份模型于 **02:23:54 UTC** 冻结，只用其后尚未运行的新题验证。登记时已看过
前两个规范题 B 进程，因此它们明确不算这份模型的验证样本；公式输入没有 B
计时。三题仍全部在 5% 门内。见 [counterfactual-register.json](counterfactual-register.json)、
[frozen-counterfactual.json](frozen-counterfactual.json)、[counterfactual-validation.json](counterfactual-validation.json)。

## 3. 实际优化：删除没有必要的全词表候选整理

原先普通 no-MTP 的 CPU 贪心采样仍为全部 **248,320 个词表项**构造 candidate
数组，再执行 top-k、top-p、min-p、temperature、dist。现有 MTP 专用 argmax
路径并未覆盖这个普通 Decode 场景。

新实现先直接从 logits 保留严格排序的 top-k 小集合。k 限定为 2..128；AVX2
一次筛八个分数，只有超过当前最小候选的元素才进入小堆。无 AVX2 的平台有
标量回退。随后仍执行**原来的真实 sampler chain**，不是直接返回 argmax
并模拟随机状态。

启用要求是 TP5、无 MTP、已证明合格的贪心链、无 grammar/reasoning budget、
无概率输出，且 top-k 是第一个实际有效过滤器。最高分必须有限且唯一。
最高分并列、NaN/Inf、未支持的过滤顺序等情况回原路径。默认关闭，通过
`GGML_TP5_GREEDY_HOST=1` 显式启用；MTP 和其他默认路径没有顺手改动。

开发时第一版“直接 argmax 后补一次随机抽样”被回归测试拒绝：原 dist 在只剩
一个候选时不会抽随机数。该失败版本没有参加 GPU 测试。修订为保留真实采样链
后，**400 组唯一/并列分数的 token 对拍，以及后续 100 次依赖 RNG 的抽样全部一致**。

| 验证层 | 原路径 | 新路径 | 含义 |
|---|---:|---:|---|
| 独立 CPU 微基准，248,320 词表，200 次 | 487.665 μs | 80.645 μs | 不是整机 tok/s |
| 真模型诊断中的 sample select 均值 | 约 581 μs | 约 84 μs | 净少约 0.50 ms/标量样本 |
| 真模型 sample synchronize | 约 269 μs | 约 260 μs | 大体未变，未冒称回读被消除 |
| 独立无插桩配对 Decode | 64.77485 tok/s | 67.14624 tok/s | 已确认端到端净收益 |

每 token 的总节省 0.54522 ms 与采样阶段约 0.50 ms 的节省相符，但不把两种
实验中小幅不同的时间差强行解释成另一个已证明的优化。

## 4. 尚未消掉的 GAP

成本模型 Match 不等于硬件极限 Match。一级账中最大的区间仍是普通稳态
`graph_compute` 的约 **13.29 ms/步**；这个区间里 GPU 算术、量化解包、命令
调度与 CPU relay 还没有被低扰动地逐项分开。不能拿它当“已知全部可消除的
等待”，也不能将其改名为不可突破的硬件下界。

现有校准参数只做条件预算：若消掉整个首轮相对稳态的额外成本，176-token
回答的模型值约为 **72.69 tok/s**；稳态无限长、其他条件不变的形式极限约
**72.28 tok/s**。两者都是**当前实现成本模型**的演算，不是新测量或硬件上限。
上下文增长会改变稳态成本，不能真的外推到无限上下文。

到 100 tok/s，当前独立实测还须从 **14.89287 降到 10 ms/token**，减少
**4.89287 ms/token，约 32.85%**。CPU 采样已经不是能兑现这个预算的大头；
继续推进必须攻克 graph_compute 内部和首轮分配/定义，而不是只增加草稿深度。

## 5. 可靠性缺口没有被隐藏

最初按十点计划启动的**原基线**在第三个请求失败：stage 1、epoch 47423、
position 159，`STAR payload timeout on rank 0`；随后四张卡由驱动自动复位，
服务退出 -6。没有强杀，也没有手工复位或改时钟。整个初始扫描按计划停止，
两个预热分数没有当作测量成绩。记录见 [FAILURE.md](FAILURE.md) 与
`screen-baseline/`。

之后只增加了失败路径的两个 bank/rank 状态快照与受控 host 时钟插桩，没有
放宽任何同步门。七请求诊断、十二请求校准以及后面的十四进程均正常完成，
但**没有复现并修复该间歇性故障的根因**。因此本报告只晋级有限范围的性能
证据；新快路仍为 opt-in，不宣称已取得全配置生产可靠性资格。

原 aggressive 冷暖正文差异也仍存在；新的 A/B 在对应冷/暖请求中逐正文相同，
只证明本补丁没有新增这些差异，不证明基础执行路径数值完全等价。

## 6. 复现与源码入口

当前被测构建为 `build-perf-gap/bin/llama-server`。测量时 executable SHA256
为 `e9aca36509cbd2d6a4d448dc08d49f21907b9d5f37084d4c94929328ae019c6c`；共享库
另按实际进程 maps 全部核查，不能只认这个 launcher 的 hash。HEAD 仍是
`9be06d5ab` 加原有及本轮未提交修改，没有 commit/push，也未重置旧工作。

```bash
python3 scripts/tp5-gap-bench.py \
  --bin build-perf-gap/bin/llama-server \
  --model "$HOME/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf" \
  --draft "$HOME/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" \
  --horizon 0 --latebind aggressive --mmvq disabled --threads 22 \
  --warmup 2 --repeats 3 \
  --env GGML_TP5_REPLICATE_ATTN=0 --env GGML_TP5_QSA_HEADMAP=1 \
  --env GGML_TP5_GDN_HEADMAP=1 --env GGML_VK_SMALL_Q8_COLUMNS=1 \
  --env GGML_TP5_GREEDY_HOST=1 \
  --out "artifacts/perf-gap/recheck-$(date -u +%Y%m%dT%H%M%SZ)"
```

`--horizon 0` 不加载 draft，runner 仍要求给出存在的 draft 路径。一次运行不等于
四组配对资格。正式排名不要传 `GGML_DECODE_COST=1`；诊断状态已在 runner 和
报告器双重检查，旧的误标 host-profile 记录只保留原始标签，不参加性能排名。

实现位于 `common/sampling-unique-argmax.h`、`common/sampling.cpp` 与
`tools/server/server-context.cpp`；host 时间账在 `src/llama-decode-cost.h`；
模型冻结/验证在 `scripts/tp5-match-model.py`，原始请求审计在
`scripts/tp5-match-audit.py`。冻结文件拒绝被覆盖，验证拒绝混入旧进程、插桩
分数、不同 binary/DSO/模型身份、回退请求或不一致的原始响应。

本报告中的现场数据以本目录原始 JSON/日志为依据；旧权重流式参考的推导见
[上一轮报告](../run-20261009T0040/REPORT.md)。它与本轮的实现成本预测不是同一个量。

## 7. 在新基线上继续做八点交互扫描

四组独立配对及三个新题 A/B 完成后，另行登记并完成八个配置的探索。全部
使用相同 APEX 权重、五卡分片、22 线程。普通 Decode 开启新采样快路；MTP1
继续使用其既有采样实现。每进程两次预热、两次测量、一题保护，八次正常退出，
40 份成功响应，16 份测量请求 fallback=0。

| 配置 | 探索 tok/s | 相对同轮基线 | 测量输出 token 数 |
|---|---:|---:|---:|
| aggressive / disabled / h0 | 66.5395 | 基线 | 176 |
| q8first / disabled / h0 | 67.1675 | +0.944% | 176 |
| pack-once / disabled / h0 | 65.9538 | -0.880% | 176 |
| AP / disabled / h0 | 66.0175 | -0.785% | 176 |
| L2 / disabled / h0 | 52.3194 | -21.371% | 175 |
| aggressive / auto / h0 | 66.0886 | -0.678% | 176 |
| aggressive / forced / h0 | 65.8915 | -0.974% | 176 |
| aggressive / disabled / h1 | 61.2994 | -7.875% | 158 |

**没有一点超过预登记的 2% 筛选门槛，因此没有另选赢家，也没有改变生产默认。**
每点只有一个进程，不能用这张表声称 q8first 显著更快，或 auto/forced 与基线
严格等效。MTP1 每份测量请求提案89、接受68，但输出158 token、每 token 速度
仍较低；不能因整篇答案更短而宣布更快，也不能把这些计数直接当作完整周期账。

L2 的慢点保留，没有因它慢就删除配置。当前扫描只证明这个具体实现、负载、
上下文下的观察，尚未把其额外耗时逐项归因为 CPU 合成、传输或 GPU 执行。
因此不重演“一个试验慢，所以整个路线永久无用”的旧结论。

新出现的12种输出已经逐份读取：自然结束、结构完整，未见乱码或空列表循环。
MMVQ、L2、MTP 的数值路径会改变正文，不能把这些输出说成全部与基线逐字一致。
此处仅作探索质量保护，不冒充全面质量评估。见
[interaction-summary.json](interaction-summary.json)、[interaction-new-outputs.json](interaction-new-outputs.json)。

## 8. 最终验证与交付范围

本轮总计 **139 份成功的真实模型响应、25 次正常进程退出，另有最初1个失败
进程和1份失败请求**。失败没有被后来的通过覆盖。正式采样补丁的原始配对审计
仍单列为78份响应/14个进程，不能拿全部探索与预热扩充正式统计样本量。

最终 CTest **11/11 通过**，其中三组 Python 模块共 **46 个测试**；另含真实
采样链400组 token对拍、100次后续 RNG 抽样检查、host 时钟开/关测试及五项
控制面回归。测试构建前后已比对推理 executable/DSO，**没有变化**。实际被测
文件另外保存在 `qualified-bin/`，原始补丁前文件保存在 `reference-bin/`。

没有运行整个已有生产测试集，未修复此前已知 `test-mtp-workspace` 失败；本轮
STAR timeout 根因也未解决。没有 commit/push，没有改旧 dirty 工作，没有
人为调整时钟、功耗、驱动、看门狗、同步或数值门限。所有本轮模型进程已经退出。

验证记录：[verification.json](verification.json)、[final-tests.log](final-tests.log)。
数据交付入口：[delivery-summary.json](delivery-summary.json)。
