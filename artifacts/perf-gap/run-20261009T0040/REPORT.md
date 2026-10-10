# RERoT 之外：理论 tok/s、真实 tok/s 与 GAP

日期：2026-10-09（UTC）。任务口径：普通单路生成，RERoT 关闭，不限制 CPU 为单线程。本轮实际使用 22 个 CPU 计算线程、五张 RX 6800、Vulkan/RADV。没有改时钟、功耗、驱动、看门狗、超时或数值容差。

## 1. 先给结果

本轮在同一份未修改的 APEX-I-Compact 权重、相同 attention 分片布局、MMVQ disabled、无 MTP 下，只改变 LateBind：

| 项目 | A：off | B：aggressive |
|---|---:|---:|
| 四对独立进程合并 Decode tok/s | 59.6591 | **65.1734** |
| 每个公开 token 的生成成本 | 16.7619 ms | **15.3437 ms** |
| 每篇回答公开 token 数 | 178 | 176 |
| 每篇回答平均生成时间 | 2983.62 ms | 2700.49 ms |
| 每篇回答平均客户端墙钟 | 3995.27 ms | 3873.11 ms |
| 测量期 fallback | 0 | 0 |

配对吞吐比 B/A = **1.092434**，双侧 95% CI **[1.085383, 1.099531]**。即吞吐提高 **9.24%**，每 token 时间减少 **8.46%**，实际收回 **1.4182 ms/token**。

这不是把某次最快的 65.72 拿来当成绩，也不是拿默认复制 attention 的慢对照来给 MTP 记功。四个进程块顺序是 AB、BA、AB、BA；每进程两次预热、三次测量、一次不参与排名的保护题。统计单位是独立进程对，不是单个 token。

| 配对块 | A tok/s | B tok/s |
|---|---:|---:|
| 1 | 59.5872 | 65.4301 |
| 2 | 59.6361 | 65.0793 |
| 3 | 59.7256 | 64.9420 |
| 4 | 59.6879 | 65.2445 |

48 份实际请求逐份核对了原始响应、公开 token 数、计时、正文 SHA256、停止原因和参数；其中 24 份是测量请求。八个进程均正常退出，没有强杀，所有测量请求 fallback=0。结果见 [summary.json](summary.json)、[raw-audit.json](raw-audit.json)、[preregister.json](preregister.json)。

**质量边界：同一份权重不等于同一条数值路径。** aggressive 改变激活量化和计算顺序，两臂不是同一 token 轨迹。A 测量正文只有一种 hash，B 也只有一种；但 B 的首个冷请求为 182 token，后续为 176，差异未归因。已读正文未见乱码、空列表循环或截断，不代表全 logits、困惑度、多领域能力和冷暖等价已经合格。本结果是短提示暖态性能证据，不能据此默认替换生产配置。

完整回答墙钟的配对 B/A = **0.969422 [0.958816, 0.980145]**，约缩短 **3.06%**，不是 9.24%。客户端墙钟、prompt 时间和 generation 时间也没有在所有请求中完全相加闭合；未归因区间不能被悄悄并入 GPU 时间。

## 2. 理论 tok/s 与实测 tok/s 放在同一张表

这里的“理论”明确指：按实际 GGUF 权重与分片结构计算、假设需要的权重每周期只从显存读取一次的理想流式参考。它没有收齐计算指令、激活/KV/状态流量与同步代价，**不是完整实现可达上限的证明**。

| 模型与配置 | 每卡 512 GB/s 的权重流式参考 tok/s | 代入 390–403 GB/s 的敏感性范围 | 实际 Decode tok/s | 证据等级 |
|---|---:|---:|---:|---|
| APEX，分片，无 MTP，LateBind off | 273.86 | 208.60–215.55 | **59.66** | 本轮四对 |
| APEX，分片，无 MTP，LateBind aggressive | 273.86 | 208.60–215.55 | **65.17** | 本轮四对 |
| 原 GSQ-RCO-Q2_0，MTP1，LateBind 实际未命中 | 319.71–335.39 | 243.53–263.99 | **约 57.17** | 前一轮四对，保留原来源 |
| HC-Q8 衍生模型，MTP1，aggressive | 约 390–414 | 约 297–326 | **约 62.28** | 前一轮四对，改变 HC 权重精度 |
| APEX，MTP1，aggressive，MMVQ disabled/auto | 337.06–364.71 | 约 257–287 | 65.13 / 65.43 | 前一轮探索，不作独立晋级结论 |

本轮 A/B 的理论栏相同，是因为这个模型只计算理想权重读取，根本没有包含 LateBind 要减少的等待及其自身附加成本；不能为了贴合实测给理论栏事后加减一个经验百分比。

RX 6800 的官方规格是每卡最高 512 GB/s、128 MB Infinity Cache。[外部 1] 前一轮 `stream-*.json` 的 512 MiB 工作集流式读写实测约 390–403 GB/s；这是读加写带宽，不是本模型 GEMV 的真实读带宽。将它代入只是敏感性分析，不是又一次模型测试。

### 2.1 APEX 的 273.86 是怎样算出来的

令 B(s,r) 为阶段 s、rank r 的权重需求，beta(r) 为相应带宽，k 为每周期真正交付的 token 数：

```text
T_target_weight = sum_s max_r [ B(s,r) / beta(r) ]
T_cycle_ideal   = T_target_weight + T_draft_weight
theory_tok_s   = k / T_cycle_ideal
```

阶段间串行、rank 间并行，所以先对 rank 取最大值，再对阶段相加；不是五张卡时间相加，也不是把全部模型文件大小除以五卡带宽。

实际 APEX 目录与 TP5 plan 给出，无 MTP、attention 分片后的阶段关键 rank 权重总量为 **1,869,594,000 字节**。因此：

```text
512 GB/s：1.869594 / 512 秒 = 3.651551 ms/token → 273.856 tok/s
390 GB/s：4.793831 ms/token → 208.601 tok/s
403 GB/s：4.639191 ms/token → 215.555 tok/s
实际 B：15.343671 ms/token → 65.173 tok/s
```

名义带宽参考与实际差 **11.6921 ms/token**；代入流式校准后的参考差 **10.5498–10.7045 ms/token**。这部分尚未由无扰动的逐项成本账完整解释，不能宣称都是可消除的软件浪费。

模型已计入 HC/router 复制、QSA/GDN 的 head 分片及 Q/K 复制。MoE 每行按 10/512 个专家读取；两行 MTP 验证按专家集合完全重合到完全不重合估计范围。非专家权重假设两行理想复用。28.8 GB 的 PLE 表按索引行访问，不按每 token 扫完整表；CPU 查表成本没有纳入上述流式参考。输入和计算脚本见前一轮的 `apex-metadata.json`、`apex-roofline.json` 与 `scripts/tp5-roofline.py`。

Roofline 的完整使用还要把算术强度、指令吞吐与实际访问的存储层级对应起来；一个带宽商不是端到端性能承诺。[外部 2]

## 3. GAP 逐项解释：哪些已经收回，哪些还没有

### 3.1 开关开了，路径却没有执行

原 Q2_0 是混合精度模型，HC down/up 为 BF16；当前 LateBind capture 要求相应 Q8_0 权重。原始日志确实是：

```text
mode=reference reason=no-late-tensors
late=0 q8_fast=0
```

证据：前一轮 `q2-route-aggressive/server.log`。所以此前仅凭 Q2_0 的 aggressive 开关与时间差认定 LateBind 加速，是归因错误。

前一轮已通过显式衍生模型改变 290 个 HC 张量，其他 934 个张量逐一字节校验不变，使 94 个 site 命中 LateBind。原 Q2_0/off/MTP1 → HC-Q8/aggressive/MTP1 的四对结果约 **57.17 → 62.28 tok/s**。这是改变 HC 精度与执行路径的组合收益，不是原模型数值不变的优化。本轮新 APEX 对照则避免再次更换权重，只验证 LateBind 模式自身的产品效果。

**状态：路径误判已排除；已有实际吞吐收益。未完成综合质量等价证明。**

### 3.2 attention 布局与 MTP 被绑在一起

原默认配置中，无 MTP 时 attention 复制，开启 MTP 后 attention 分片并开启 headmap。只比较这两个默认值，会把布局优化记到 MTP 头上。

本轮两臂都显式开启分片与 headmap，均无 MTP；因此 **59.66 → 65.17** 不含“开启 MTP 顺带改变布局”的混杂因素。前一轮 APEX 默认 no-MTP/aggressive 的 57.28 与显式分片的 64.19 只是探索，不能与本轮配对增益继续相加。

**状态：已建立正确的强对照；布局与 LateBind 的收益不再混算。**

### 3.3 LateBind 确有净收益，但自身不是免费

本轮同模型同布局单因素的实际净收益是 **1.4182 ms/token**。aggressive 既改变算术，也改变依赖安排，因此不能进一步声称这 1.4182 全来自隐藏通信。

正式配对结束后，另做一次不排名的 GPU 捕获，选择第 400 条 chain。rank0 的时间如下：

| 同一次捕获的区间 | ms/捕获链 |
|---|---:|
| native compute | 10.516 |
| LateBind late_pre | 5.176 |
| P2 等待及消费 | 3.853 |
| P1 发布 | 1.163 |
| 已记录间隙 | 0.023 |
| 其余 span 差额 | 0.029 |
| 总 span | **20.762** |

这些不是正常生成的逐 token 均值。该次带插桩请求只有 **59.65 tok/s**，相对于正式 **65.17** 有明显扰动；捕获还改变了定义录制。不能把 20.762 ms 的区间比例强行缩放到 15.344 ms，伪造“完整闭账”。P2 包含等待和消费，不等于 PCIe 拷贝；native compute 不等于纯 ALU；tail_compute 已含于其他区间；不同 rank 的 span 互相重叠，不能相加。日志没有给出 rows 字段，解析键中的 rows0 不代表执行了零有效行。

**状态：已验证净收益，并定位仍大的区间；每个区间还能省多少尚未获得无扰动因果证据。**

### 3.4 MMVQ 变快，可能同时使 MTP 每周期少产出

正确目标不是某个矩阵核时间，而是：

```text
ms/交付 token = (draft + target verify + 状态维护 + 其他未重叠时间) / k
```

前一轮 HC 变化的实际例子：原模型每请求 88 个提案周期、156 个公开 token，k=1.7727；新模型 98 个周期、167 个公开 token，k=1.7041。周期时间 **31.009 → 27.362 ms**，少了 11.76%；每周期产出却少了 3.87%，最终净吞吐只增加约 8.95%。

MMVQ 的激活量化也可能通过 target/draft logits 改变接受行为。前一轮 APEX aggressive/MTP1 的 disabled/auto 探索为 **65.13/65.43**，不足以声称稳定额外收益。本轮正式对照固定 disabled，没有假装已经测出三策略的最优者。

推测解码的分布保持结论针对既定 target；更换 target 权重或激活数值路径后，验证器不会自动恢复原 target。[外部 3]

**状态：没有独立 MMVQ 净增益证据，不把它算入已收回 GAP。**

### 3.5 深 MTP 增加验证成本，强无 MTP 对照尚未被正式击败

前一轮原 Q2_0 的 h1/h2/h4 探索为 **56.94/49.42/41.40 tok/s**。该负载上更多草稿没有换来足够的公开产出。

更重要的是，前一轮 `mtp-strong` 原计划四对，但只完成第一对：no-MTP 65.04、MTP1 62.71；第二块 B 在加载时发生 `RELAY previous GPU wait failed on rank 0`，实验停止。**该确认未完成，不得说 MTP 显著更慢，也不得说已经证实 MTP 比最强 no-MTP 更快。** 本轮没有替换那个失败点，而是单独注册并完成新的 LateBind 比较。

**状态：MTP 与最强 no-MTP 的比较仍有缺口；当前可独立报告的是 no-MTP 配置的 65.17。**

### 3.6 重录已经减少，但没有证据说明 Decode 大幅受益

前一轮完整定义缓存 `GGML_TP5_DEFINITION_CACHE=4` 保留全部身份和 native retirement 规则，使每请求定义重录 **6 → 0**。四对 prefill 时间比 **0.94659 [0.92709,0.96650]**，确有减少；Decode 吞吐比 **1.01039 [0.97994,1.04179]**，区间含 1。

不能将“重录次数归零”写成“Decode GAP 已消除”。该缓存默认仍关闭。本轮只修改报表解析器，使无 MTP 的 GPU 捕获不再被遗漏，没有夹带生产缓存、shader 或默认参数改动。

**状态：消掉了重复定义工作；尚无显著 Decode 加速证据。**

### 3.7 首轮长延迟与 target 稳态成本仍需攻克

前一轮 HC-Q8/MTP1 的轻量诊断，同一次 167-token 请求：

| 不重叠一级账 | 请求总 ms | ms/token |
|---|---:|---:|
| draft | 239.815 | 1.4360 |
| target verify（含相关 host 处理） | 2300.945 | 13.7781 |
| catch-up / 状态维护 | 37.085 | 0.2221 |
| 残差 | 6.423 | 0.0385 |
| 总计 | **2584.268** | **15.4747** |

其中首个 target 周期 227.890 ms、普通周期中位 21.089 ms，超额约 **206.801 ms/请求，1.2383 ms/token**。它是待归因预算，不是已经证明可全删的时间。完整定义缓存开了、重录归零后，原 Q2_0 首周期仍在 210–240 ms，因此“全部是录制开销”的解释不成立。

对上述同一次诊断作条件演算：只删除全部首周期超额是 **70.24 tok/s**；只令 draft 和状态维护免费是 **72.38 tok/s**；两者同时免费是 **79.50 tok/s**。这些都不是实测、也不是硬件上限；它们说明，超过 100 还必须显著缩短普通 target 周期。

后续工作应按这个顺序验收：先对首周期的图建立、分配、输入更新、真实 GPU 执行和回读逐段计时；再对高占比 shape 记录真实 VRAM 事务和 ISA/寄存器/LDS 数据；最后做保持数值/同步契约的局部替换。每项都要证明端到端净收益，不能按源码行数或 barrier 数量估计加速。

**状态：剩余主要 GAP 未消除，也没有证据证明路线已经穷尽。**

## 4. 现在离目标还有多远

以本轮 **65.1734 tok/s** 为起点，到 100 tok/s 需要 **15.3437 → 10 ms/token**，还差 **5.3437 ms/token**，即减少 **34.83%** 的当前生成时间。

本轮确认已经省下的 1.4182 ms/token 与前一轮 HC 换精度省下的约 1.44 ms/token 不是可相加的两个独立插件：模型、对照与优化覆盖不同。不能用它们凑一个虚假的叠加速度。

当前结论是：**有真实的、独立配对确认的收益；没有把全部理论与实际 GAP 消除。** 权重流式参考 274，实际 65.17，差距中只有部分被识别和验证。实际 VRAM 流量、算术指令成本、流水重叠和冷暖数值差异仍需进一步闭合。

## 5. 重现与证据

本轮构建使用既有 `build-perf-gap/bin/llama-server`，HEAD `9be06d5ab` 加原有 dirty 修改；没有重新构建。每个进程保存 executable SHA256 和实际映射 inference DSO SHA256，配对分析确认相同。APEX 六分片记录 path/stat/inode/mtime 身份一致，本轮没有新做六文件全量 SHA256。六个关键源码文件测量前后指纹一致。

复测性能候选（新的实际 GPU 运行，不是 dry-run）：

```bash
python3 scripts/tp5-gap-bench.py \
  --bin build-perf-gap/bin/llama-server \
  --model "$HOME/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf" \
  --draft "$HOME/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" \
  --horizon 0 --latebind aggressive --mmvq disabled --threads 22 \
  --warmup 2 --repeats 3 \
  --env GGML_TP5_REPLICATE_ATTN=0 \
  --env GGML_TP5_QSA_HEADMAP=1 --env GGML_TP5_GDN_HEADMAP=1 \
  --env GGML_VK_SMALL_Q8_COLUMNS=1 \
  --out "artifacts/perf-gap/recheck-$(date -u +%Y%m%dT%H%M%SZ)"
```

`--horizon 0` 不加载 draft；runner 仍要求提供已存在的 draft 路径用于输入检查。输出目录须不存在，独占 GPU 锁、空闲及五卡检查由 runner 执行。一次复测不等于新的四对资格。

本轮对 `scripts/tp5-gap-report.py` 的修改仅修复无 MTP 的 GPU 时间报告被丢弃的问题；缺失 MTP 指标写 null，不伪造为零。对应工具测试 **14/14 通过**。没有声称整个已有 dirty 源码的全部测试通过，前一轮 `test-mtp-workspace` 已记录的失败仍未在本轮修复。

证据索引：本目录 [summary.json](summary.json)、[raw-audit.json](raw-audit.json)、[profiles.json](profiles.json)、[unique-outputs.json](unique-outputs.json)，`lb-b*-*/` 内逐请求原始响应与日志，`gpu-ledger/` 为不排名诊断。前一轮证据保留在 [run-20261008T1845](../run-20261008T1845/REPORT.md)，不把其历史实验改称本轮新测。

## 外部依据

外部 1：AMD RX 6800 官方规格，512 GB/s、128 MB Infinity Cache。
https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-6000-series/amd-radeon-rx-6800.html

外部 2：NERSC，Roofline Performance Model，带宽、算术强度和资源边界。
https://docs.nersc.gov/tools/performance/roofline/

外部 3：Leviathan 等，Fast Inference from Transformers via Speculative Decoding，固定 target 下的推测解码。
https://arxiv.org/abs/2211.17192

外部资料不为本机 tok/s 数据背书；本机数字均来自上述本地原始响应或明确标出的条件演算。
