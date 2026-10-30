## 2026-10-30：TP5 双请求 MTP ready snapshot 源码交付

此版本移植 2026-10-25 ready snapshot 的合并源码补丁；默认保留 post-submit unlock 关闭的配置。历史正式运行在五张 RX 6800、F32 relay、Flash Attention、两路各 262144 KV 和 512 token 下，包含全部 511 个间隔，观察到 51.23 / 52.09 tok/s；短测 logits 与参考逐位一致。结果来自历史隔离安装，不是本次提交的重建测试，也不保证其他环境达到该性能。

快照历史复现闭包位于原机器 `mtp-endtoend-20261024/mtp-ready-snapshot-consolidated-20261025/`；它复用历史对象和本机模型，不能据此宣称本提交已从干净源码完整重建或通过设备验收。row-tail 56.77 / 58.93 tok/s 是另一候选，未包含于此版本。未合入未验收的 FA true-len、provider 归属、prefab 或 LateBind 默认开启候选。

<!-- two-requests-current-status-20261021 -->
## 2026-10-21：双请求保留版本与最终测量；每路 47 tok/s 目标未达到

保留版本、证据和入口见[双请求报告](../build-tp5-two-requests-20261021/TWO-REQUESTS-REPORT.md)；以下旧正文保留。本次增量为 3 个 A 源文件和 1 个 C 源文件，公共 API 不变；使用原始模型权重、每请求 262144 容量的 Q4_0 K/V，并保留 Swift 共享权重派生 view 生命周期修复及五 rank host 提交顺序 gate。

同 prompt 双请求组（n=2 个独立进程），每路 512 token：两路几何均值为 **37.629332 / 37.624558 tok/s**，aggregate 几何均值 **74.923025 tok/s**，范围 **74.847236–74.998891**。相同 C 版本的单请求观测（n=1）为 **46.071845 tok/s**；aggregate 对单请求的 **1.626222×** 仅作描述性比较。不同 prompt 的一个双请求案例为 **34.571716 / 34.899258 tok/s**，aggregate **69.062687 tok/s**。**两个请求各 47 tok/s（合计 94 tok/s）的目标未达到。** 历史 **47.34 tok/s** 只属于旧 F16 单请求配置。

完整 16-token trace 已通过，49 组五 rank 提交保持一致次序，但该完整 tracing 的共同生成期 GPU overlap 为 **0**。另一次相同保留版本的 kernel-only 采样已通过：共同输出阶段 GPU0–3 观察到跨请求算术/算术与算术/consume 的真实交集，GPU4 为零；含预填充的完整活动阶段五卡均有局部交集。由此可确认存在实际 GPU 重叠，但不能宣称五卡共同生成期都已重叠或通信完全隐藏。各次观测和证据分别保留，详见双请求报告。

---

# 发布候选变更说明

<!-- TP5_NATIVE_SPEED_20261021_FINAL -->
## Latest retained result: native TP5, 2026-10-21

The current retained native TP5 configuration is the 2026-10-21 four-pair-qualified build. Older “current”, “final”, and gap-failure entries below are historical. The user's relaxed numerical policy applies to native acceptance; explicit strict-reference modes retain their separate gates.

Four balanced pairs against fixed master `46a6fd8a129091741f9ebc6431f7f610abe1ccfc` passed: geometric-mean throughput **47.343475 vs 45.366353 token/s**, paired ratio **1.043581249**, two-sided 95% CI **[1.037961336, 1.049231589]**. Every run includes all 511 intervals from 512 natural outputs.

Retained changes enable the native backend policy, BF16 load4, existing HC fusions and Q2 short-row reuse, parallel per-rank graph instantiation, and guarded logical attention padding of 1024. Physical KV capacity and CPU rank-ordered F32 relay are unchanged. The F32 router extension was withdrawn; initial direct warm is retained.

The incremental production patch changes ten existing A files (+24 net lines) and one C file (+1); the 51-path overlay set is unchanged. See the [final report](../build-tp5-native-speed-20261021/NATIVE-SPEED-REPORT.md).

<!-- /TP5_NATIVE_SPEED_20261021_FINAL -->

## 2026-10-19：TP5 单流性能修复，原数值门不变

同一 Qwen3-8B Q8_0、五 RX 6800、固定单请求轨迹，四个独立配对 block 确认旧/新几何平均为 **9.1784/54.9584 token/s**；速率比 **5.98781**，95% CI **[5.29900,6.76615]**。主终点是 64 输出的 63 个真实提交间隔，不含模型加载、网络或排队。此前 11.30575/56.23309 是筛选单次值，不混入确认统计。

- A 私有 provider 接收完整模型/RELAY 链，预热后进行至多两个精确绑定缓存项的整步原生重放；五卡 executable 准备完成后才实际 launch。通用 `GGML_HIP_GRAPHS` 仍关闭，未放宽原图/KV 失效条件。
- 忙轮询不再反复切五个设备查询尾事件；snapshot/close 仍收集真实完成证据。每边界五 kernel 合为两个有界单 workgroup kernel，保留全写者发布和最后读者 ack，四值搬运不假定 view 的 16-byte 对齐。
- 原 Q8 权重直接解码乘 F32 activation；投影改为无 LDS barrier 的四输出 block。CPU SIMD 只并行不同元素，各元素保持 rank 0→4 F32 加法和完整有限值检查。

最终匹配轨迹的稳态四间隔观察到 0 次 `hipLaunchKernel` API、20 次 `hipGraphLaunch`；五个物理 BDF 都执行真实模型投影和 RELAY。性能来自无 profiler 的配对进程，而不是 launch 计数折算。

FA on/off 各自完整 16-token 主路径、慢消费者、取消和第 8 epoch abort 通过，原 `.002 + .001*abs(reference)` 门下均零超差。288-token 单请求跨 KV 256→512，43,757,568 值最大绝对差 **0.0005264282**、零超差，并实际重新捕获后继续重放。1210 个 Q8/F32 GPU oracle、180 个 attention oracle、保序 CPU 回归及 host C/C++ ASan 主路径/第 8 epoch abort 通过；ASan 不覆盖 HIP 翻译单元/设备访问，leak detection 关闭。

FA on/off 比值 **0.90844**、CI **[0.88025,0.93752]**，默认仍 off。新四请求 one/two-window 对照没有确认额外并发收益；保留现有自动窗口策略，不沿用旧镜像的 1.71597 倍结论。master 的 60–67 token/s 来自另一份 APEX-I-Compact 混合精度/MoE、激进 LateBind/Vulkan 负载，不作为当前 dense Q8_0/F32 路径的同模型加速比；未复制旧框架或移植不适用的 HC/LateBind。

A + 七 B + 七 C 已重新安装、搬迁，在原 prefix/stage 不存在时用公共 target 构建外部消费者，运行双窗真实模型、默认单流和安装后的全 logits/第 8 epoch abort。发布 CLI/native 镜像与性能确认逐字节相同；原 prefix 已恢复。精确证据见 [当前机器索引](audit/tp5-performance-20261019.json) 和 [当前 C 文档](tp5-compose/README.md#single-request-performance-repair-2026-10-19)。以下保留原生重放前记录。

## 2026-10-19：真实五卡 TP5、合格 F32 Flash Attention 与实测 off 默认

新增独立 C [`tq-tp5-compose`](tp5-compose/README.md)：真实 Qwen3-8B Q8_0 在五张
RX 6800 / HIP gfx1030 上执行，CPU 按 rank 顺序归约 F32，完整 producer / wait /
consumer 整链预提交，两个 context 保持独立可变输入与 KV。现有组件为七 B、七 C。
A pin 不变；当前 manifest 共 **48 个已审 A 路径**，包含两个新增 Q8/F32 文件，
不是“48 修改另加二”。精确身份见 [机器索引](audit/tp5-acceptance-20261019.json) 与
[本轮验证](VALIDATION.md#2026-10-19tp5-五卡-hipflash-attention-数值与生命周期验收)。

- `--flash-attn on|off` 默认 **off**，严格解析；off 是实测最快的合格模式和显式数值参考，
  不是错误回退。显式 on 仍有完整数值与生命周期资格，但没有性能默认晋级。
  每个 context 的 resolved 设置必须与请求相同，报告区分请求与执行设置。
- 精确 profile 复用已有融合 online F32 FA，模型图保留 F32 Q/K/V，不增加 F16 舍入
  边界，也不拆成 QK / softmax 替代图。普通 FA 分派与精度不变；共享 online/banded F32
  kernel 在 K/V 读取前跳过 -INF masked key，改善 masked NaN 语义，不笼统声称其他后端
  一切不变。有限未屏蔽项的累加顺序保留。
- 完整八 token 四请求的 `main/slow/cancel/abort` 门全部通过：32 行、4,861,952 logits，
  固定 `.002 + .001 * abs(reference)` 门下超差和非有限值均为 0，max_abs
  `2.193450927734375e-5`。慢消费者延迟 2 秒时另一窗口前进；取消后请求 1/2/3 继续；
  abort 退出 1，两窗口真实 native completed mask 均为 31，最后预算为 0。
- 最终 `acceptance-fa-final` 重跑完整八 token 全场景通过；五设备 Q8/F32 oracle 170 例与
  最终 FA F64 oracle **180 例**通过（含 always-masked K/V NaN），max_abs `2.24640083e-6`
  小于未改动的 `2e-5` 门。FA-on trace 记录 720 次
  融合 FA launch，五物理设备各 144 次；FA-off trace 为 0。这不是吞吐或性能晋级证据。
- 最终 FA-off 八 token main/slow/cancel/abort 全部通过。FA-on 的 32-token main-only
  对照另检查 128 行 / 19,447,808 logits，同一数值门下非有限/超差 0，max_abs
  `9.059906005859375e-5`；不扩写为 32-token 生命周期全矩阵或长上下文支持。
- 当前源码 host-only ASan 对四个原始 193-token prompt、两个 context、每请求生成两 token
  退出 0；此前四 token main+abort ASan 保留其独立候选身份。Leak detection 关闭，不代表
  GPU sanitizer、全场景 ASan 或无泄漏。Strict preflight 只跳过合法无输出 prefill 空 tensor，
  不造 logits、不缩短原始工作负载。最终 HIP 已完成七 B / 七 C 构建安装与 11/11
  A seams；其末尾 guard 曾因根说明并发修改而拒绝旧 digest，最终身份回执另列。
- 来源隔离的 85 条通过命令属于保存的 **45-path 前 FA 候选**，不冒充当前 48-path
  构建的完整重跑。当前最终 HIP 包已有独立的搬迁/公开消费者证明，见下文；源码 guard
  与最终机器索引看各自最终回执，不以这些选定合同代替所有门全绿。
- `hip-package-proof/summary.json` 通过选定合同：15 组件（A+七 B+七 C）fresh install
  到 stage 后改名 relocated，原 stage 与旧 install 前缀在消费者配置/构建和所有 GPU
  调用期间均不可用，之后才恢复。外部项目仅安装公共头与 `find_package(TQTP5Compose)`
  imported targets；编译路径无旧前缀/私有头，`ldd` 指向 relocated libraries。
- 外部 consumer 两请求显式 on、双 context、真实五 rank、预算 0；直接 CLI 两请求
  默认 off、单请求自动一窗口/off、三请求一窗口/on 均通过。搬迁 helper 的实际路径
  **`share/TQTP5Compose/real-tp5.py`** 下 predict4 main-only 完整 logits 门通过。
  首次 harness 猜错路径、子测试启动前退出 2 的失败保留；只改路径续跑，未重复成功 setup。
- 最终发布包 trace 绑定五个实际 BDF：on 720 次、每卡 144 次，off 0；native libraries
  与 12 个性能进程镜像相同。发布 CLI 的默认选择不同，但所有性能 case 显式传 on/off。
- W0 请求 delay、W1 观察到 abort 的组合场景通过：两窗口 completed=31、unknown=0，
  预算 0、exit 1、无成功输出。精确 held-borrow 分支未单独 instrument，不宣称该分支覆盖。

### 逐流数据与默认模式裁决

`per-stream-fa-final/summary.json`：12 个进程、三个配对 block，固定四请求，每请求
193 prompt token 与 32 teacher 输出。每流几何平均速率（不是总吞吐）为：

| FA | 窗口 | 每流 token/s |
| --- | ---: | ---: |
| off | 1 | 8.9853 |
| off | 2 | **15.4185** |
| on | 1 | 7.7510 |
| on | 2 | 13.4642 |

两窗口 on/off 速率比 `0.87325`，95% CI `[0.83966,0.90818]`，on 慢约 **12.7%**。
因此默认 off、保留显式 qualified on。配对区间采用 Student t df=2，仅 n=3 独立 block；
每请求只有 31 个 ITL，p99 不是稳定尾部估计。TTFT 为 admission-to-first-commit，排除
模型加载和网络。`per-stream-fa` 原失败与 `per-stream-fa-prefill-fixed` 较慢候选均保留。
用户关心速度不是要求默认 on，先前将两者等同的假设已纠正。

Off 两/一窗口速率比 `1.71596659`，95% 配对 t 区间 `[1.28598327,2.28971978]`，
仍来自同一 n=3 固定工作负载，不是通用并发收益。TTFT 还排除 context startup 和
admission 前 queueing，不能理解为端到端时延。

模型固定为 8,709,519,168 bytes，SHA-256
`0cfbf745760f07a76ddeb358dd025a27f2e11d1ca9c9a4169a373d52990fe86e`。
旧 FA-on、累加精度与 F16-cast 尝试的失败，以及原 Q8/ASan、Vulkan FA/TOPK 记录不删除，
数值门不放宽。支持不扩展到 LFM2 MoE TP5、其他模型/后端、长上下文、MTP 或 RERoT+TP5；
没有通用加速或 FA 性能晋级结论；上述固定工作负载的逐流测量不能扩大为通用性能保证。
后文各批次维持自己的历史身份。

## 2026-10-19：DevOps 交付收尾

现有 CPU / Vulkan 包、CLI、真实模型与 HTTP 入口完成本轮重验；独立 HIP 探针完成
五张 RX 6800 的 2000 轮运行。来源、具体门、失败前反例与新回执见
[验证记录](VALIDATION.md#2026-10-19devops-收尾与当前交付验收) 和
[机器索引](audit/devops-acceptance-20261019.json)。后文各批次仍保留自己的身份。

- 修复 Vulkan 找到非系统 SPIRV-Headers 后仍缺少编译 include 的接线。只有这一项新增
  A 例外，manifest 现为 35 路径；私有头依赖不传给安装消费者，不修改推理数学。
- HIP 工具的 build/install RPATH 来自导入 runtime；搬迁安装不再依赖
  `LD_LIBRARY_PATH`。定标 kernel 使用单写者与映射设备地址，未知完成时不做危险清理。
- bench 校验强制四项测量的名字、单位、样本数，并检查与平台报告的共同字段。
  缺项、配错轮数和非法 Unicode 字段名不再被接受或通过 traceback 退出。
- 默认构建只运行 probe 的 CPU checks，显式关闭额外 HIP discovery；独立 HIP 构建、
  CLI、安装与硬件运行仍是单独的门。空输出路径在设备访问前拒绝。
- 隔离安装清理编译器与加载器搜索覆盖、禁用 CMake 环境/package-root 搜索，检查导入
  位置。新增直接消费 Relay 和只链接 Compose 公共 target 的验收；污染环境下，
  来源隔离、删除副本、搬迁与真实模型消费 80 条命令全部通过。
- 文档统一指向当前 hardening 与本轮 DevOps 回执，不再将早期“正在验证”或 GPU oracle
  失败误当成后续候选的现状。旧失败未删除，数值门未放宽。

本轮五卡 **host fetch_add 仍为不可依赖能力**：全部实际返回错值；单写者 load/store
平台门通过不等于全部系统原子可用。模型用例仍限定单卡 Vulkan 与 CPU；没有新增整模型
TP5、HIP/CUDA 模型、多设备 PP、吞吐提升或远端 GitHub Actions 成功声明。

## 2026-10-19：RERoT 共享白板与独立执行包

这次加入可实际运行的 `TQRerot::rerot` 和 `tq-rerot-compose`。每个正在运行的插入点
每轮推进一个 BODY token，同轮统一使用冻结的已提交 peer 视图；原生计算全部完成后
共同发布，下一轮立即互相看见新增 token。多个插入点可以在各自结束前持续交流，
DAG 的边只规定开始条件，最终 synthesis 消费全部已提交工作。

### 职责与维护成本

| 层 | 本次交付 |
|---|---|
| B：`fork/rerot` | 仅依赖 C++17 标准库的 DAG、不可变 reader 视图、public/private 行、每 token 事务与失败 / 取消身份校验；可独立安装消费 |
| C：`fork/rerot-compose` | 链接已安装的 llama、TQRerot、TQRelay、TQText；原生 tokenization、模板、模型采样、通过 A 使用统一 resident KV、按波执行 FRAME、实际同步后 BODY 发布、阶段计算内存释放与 JSON 事件 |
| A：明确登记的例外 | 公共 reader attention views、实际 KV 位置绑定、reader-relative RoPE / 全局 attention、逻辑序列与物理 recurrent 池分离及 recurrent-only retirement；本次 14 路径 +642 / -52，无新增通用 GPU kernel 文件 |

A 的 identity span 保留 native batched KQ，可见分数按物理 KV 位置排列，所有 span 共享
一次全局 softmax 和 native batched V。C 每个 A 调用使用局部 TQRelay 提交账本，实际完成
并清理 reader 后销毁；不积累随生成长度增长的历史票据。固定 A pin 未变，当前 28 个
例外路径的 hash、消费者、理由和移除条件在 [manifest](upstream-exceptions.json) 中登记。

首次发布限定为 **resident F32 K / F32 V、C 关闭 FA 的 assistant-eot profile**。
权重量化仍由 A 执行。F16 reader 的剩余重排 / restore 失败没有获得容差豁免，而是关闭该
profile 并写入重写候选；普通 A 的 F16 解码继续可用。这会增加 reader KV 存储成本，
没有新的峰值 VRAM 或吞吐收益声明。

### 实际通过的检查

- B 的独立构建、四组测试及 ASan + UBSan 通过，覆盖 543 个 DAG、3,007 个状态和
  3,904 个 reader；独立 F64 oracle 比较 3,200 个 reader/head。
- 实际 A 图的 16 个 CPU 独立 F64 案例通过，使用单层、零 FFN 的 F32 GGUF fixture；
  真实 LFM2.5 的 CPU / RX 6800 F32 reader
  各 33 项严格检查通过，等价组最大差分别为 0 和 `0.0000133514404296875`。
  GPU 的独立 F64 oracle 尚未取得资格。
- CPU / 单卡 Vulkan 各八次完整生成通过。每次两个活跃 worker 都观察到对方至少
  103 次前缀增长；固定 microbatch 下，阶段内存开关及重复运行保持同一完整 token 轨迹。
- 两端的真实 SIGINT / broken pipe 检查通过，分别受控退出 130 / 1；ordinary recurrent
  生命周期回归续算差均为 0。CPU 4/4、Vulkan 8/8 公开接缝 CTest 通过。
- 完整来源隔离 77/77 条命令通过：固定 A archive + verified overlays、七 B / 六 C
  独立安装，删除构建副本并搬迁前缀后，B 消费者、CLI 和四次实际 RERoT 生成通过。

精确命令、模型 / 库身份、数值门与保留的失败见
[验证记录](VALIDATION.md#2026-10-19rerot-逐-token-共享白板) 和
[机器索引](audit/rerot-qualification-20261019.json)。这些结果不代表所有后端、模型或
此前全量 GPU 遗留问题均已通过。

### 与 TP5、PP 和优化组合的关系

复用的是现有 TP5/TQRelay 架构中已能由真实 A 调用消费的提交、完成和 drain 生命周期。
当前只有一个 A 提交域；整模型 TP5 provider 尚未暴露为可安装消费者。
`--tp5`、`--latebind`、`--mtp` 明确拒绝，相关后续协议保留在
[重写候选](REWRITE-CANDIDATES.md)。世界 revision 更新后，旧世界的 draft / logits
必须重新验证或作废，不能跨提交边界直接复用。

原生设备放置与调度交给 A。RX 6800 的 `none` 和 `layer` 实际生成通过；
`row` 因该 Vulkan 后端不支持 split buffers 在加载时失败。多设备 PP 未实测。
QKV / FFN 保留上游量化权重和多行分派，但没有本次 RERoT 的 MMVQ kernel 命中或加速证明。
跨 microbatch 的 logits 与 greedy 轨迹实际不同，明确未取得跨形状等价资格。

构建和运行方式见 [RERoT C README](rerot-compose/README.md)。这一版本使用明确的
`--stage` / `--after` DAG 和模型原生 assistant-EOT 模板；自动 planner、tool /
reasoning-end 路由、HTTP 接入与量化 KV reader 没有混入该 profile。

## 2026-10-18：历史候选记录

以下保留上一批次的范围和证据；其中“当前”“本批次”、五 B / 五 C 与旧 manifest 数字
仅指该历史批次。其失败与未完成调查继续有效，不被本次 RERoT 专项覆盖。



## 本批次：增量文本、校准互操作与后端缓存正确性

本批次从 `6de088714f35f84427d9952d713f9d6cdffc668b` 继续整理有价值的小功能；固定上游 A 仍是 `cda0f9256665845cd9d6f3a0176880883f653a3d`。当前维护五个独立 B 库和五个 C 编排包。新增能力由独立的格式、文本对象与真实公开接口消费者承担；Vulkan 私有缓存的修复保留为有明确消费者和回归用例的 A 例外。

**本页记录已取得的开发与运行证据，不代表最终发布门槛全部通过。** scratch 与源数据原地修改两个修复已通过最终专门 GPU 用例，最终 CPU 来源隔离和普通 tools 构建通过。最终全量 GPU 为 25932/25933，仍有一个 FA 失败；较早 TOPK 失败的归因也未闭合。普通 Vulkan 构建入口也已通过，实际身份和统一状态见 [Final release gates](VALIDATION.md#final-release-gates)。A 的文件集合和精确内容以 [manifest](upstream-exceptions.json) 为准。

### 新增与重新划分的职责

| 变化 | 维护合同与使用价值 |
|---|---|
| B `TQText` | 每个对象拥有增量 UTF-8 和 stop 状态，返回拥有自身文本的结果与字节游标；对输入分块保持一致，不依赖 A 或兄弟 B。 |
| C `tq-batch-compose` 的文本输出 | 用真实模型 token piece 驱动 B；支持重复 `--stop`、`--json --stream`，在匹配确定后结束该请求的后续 decode，并按现有序列与预算合同释放状态。 |
| B `TQCalibration` | 有界逐层读取 WXQCAL03，校验几何、统计人口、数值、截断和精确文件结束；各层结果拥有自己的数据。 |
| C `tq-imatrix-compose` | 明确选择完整范围、partial 与 unseen expert 策略，保留原始平方和及可精确表示的计数，用已安装的 A 公共 GGUF/GGML 元数据接口导出 imatrix。 |
| Vulkan scratch 复用 | 共享工作区被大 softmax 或 masked attention 改写时失效转换缓存；复核 no-split 路径最后一次读取与下一次写入的同步责任。该阶段专门用例通过，范围见下表。 |
| 既有 Turbo SET_ROWS | 复核固定 A 已有的 F16 支持，以真实 packed-row 对照验证；不迁入旧 fork 的 F16 拒绝条件，也不把上游已有功能记作新增。 |
| 源数据原地修改修复 | 同一个 source view 的底层值被 SCALE 改写后，A 对重叠写入失效转换缓存。非零偏移 probe 修复前第二次 matmul 64/64 错，重建重装后 matmul 与源存储的完整比较误差为 0；独立于 scratch 覆盖问题验收。 |

五个 B 为 Sampling、Corpus、Scheduling、Text、Calibration，均只依赖 C++17 标准库。五个 C 为 compose、corpus-compose、batch-compose、logits-compose、imatrix-compose。新 imatrix C 仅链接安装后的 `ggml::ggml-base` 和 TQCalibration；A 的 package config 仍会查找启用的 backend，配置时需要完整 A 安装。其他消费者的依赖边界见 [README.md](README.md)。

### 文本输出是用途重建

`TQText` 保留有用的 UTF-8 边界、跨块 stop 与输出生命周期合同，没有宣称复现旧 `server:Q14` 的非流式跳扫描 fastpath 或速度收益。`server:T09` 只提供了跨块终止符机制的参考；完整 thinking/lane parser 没有迁入。

`tokens` 保留已采样的非 EOG token，`--ignore-eog` 时也保留 EOG；stop 抑制文本不回删计数。匹配确定后 C 不再提交该请求的 decode。若最终 UTF-8 修复才完成 stop，承载未完成字节的早先 token 可能已经提交给 A。每个成功请求恰有一个终止文本事件，按字节游标拼接的增量等于最终文本；stdout 失败令运行失败。此合同没有包含 HTTP SSE、网络送达、取消、重试或多 choice usage。

以下为使用示例，不是本批次的实际验证命令回执：

```sh
build-clean-fork/install/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt --parallel 2 \
  --ctx-size 2048 --predict 64 --stop '<end>' --stop '###' --json --stream

build-clean-fork/install/bin/tq-imatrix-compose \
  --input calibration.bin --output imatrix.gguf \
  --dataset 'my-selected-calibration-range'
```

### 校准导出保留测量与估计的区别

默认要求生产者报告 complete、声明的总量为正且 `decoded == total`，并拒绝没有观测值的 expert。旧 writer 的最后一批可超出所选节点上限；`--allow-partial` 会保留原始计数并记录每个实际原因，不能用它跳过结构或数值错误。`--unseen global-mean` 是显式估计，记录填补的 expert ID、所用策略与“填补行不是测量”的元数据，绝不把 count 1 当作真实观测数。

导出保留 raw sum2；A 才按 expert count 归一化。计数必须是可精确表示的 F32 整数且不超过 INT32_MAX；平方和及 A 中实际 F32 除法的溢出、非零下溢都拒绝。一个 parsed layer、有限转换缓冲和私有磁盘 spool 控制内存；完整验证后才无覆盖发布新文件。完整格式、平台、磁盘容量和失败合同见 [imatrix-compose/README.md](imatrix-compose/README.md)。

### 本批次已观察的验证结果

| 检查 | 实际结果与范围 |
|---|---|
| 最终 CPU archive / isolation | `build-isolated-details-qualified-20261018` 的 62 条命令通过；五个 B、五个 C 独立构建、安装，移除原副本和前缀后搬迁，真实 batch/logits/stream 与 imatrix 消费通过。 |
| 两个新增 B | TQText、TQCalibration 的 ASan 与 UBSan 测试通过；不据此补写较早 logits writer 的 ASan 资格。 |
| 旧 writer 与实际 A loader | 原 `calib-dump.cpp/.h` 与旧提交 `2777543b2358c88cd1c2000f7fa342b764a0e213` 完全相同；生产格式 fixture 经新 C 产生 12 tensors，A quantizer loader 读入 6 entries / 9 chunks。随后故意缺失模型以 exit 1 结束，只证明格式互操作与 loader 消费。 |
| 最终 Vulkan 接缝 | 7/7 CTest 通过，含 5 个 scratch 用例、2 个 source-alias 用例及 16 次 SET_ROWS packed 比较；no-split 检查全部 524288 输出及闭式结果。source-alias 的 4-byte-offset probe 修复前 64/64 错，最大绝对差 15.4662，修复后完整比较误差为 0。 |
| 最终 Vulkan 文本消费者 | 10 条命令，其中 6 个真实模型配置各 3 个请求，共 18 个请求通过，脚本用时约 14.5 秒；stream 开关对照保持同配置提交形状。该时间不是吞吐基准。 |
| 已完成的全量 CPU backend 测试 | 27639 / 27639 通过，保留该次运行身份；最终广覆盖复验状态另列。 |
| 最终全量 Vulkan backend 测试 | 25932 / 25933，exit 1，仍有同一 mask+sinks、Q8_0 K/V、head dimension 256 的 FA 失败，报告 ERR 0.009055835 超过 0.0005 门限。最终 TOPK 未复现；较早 +12 full 中的 TOPK 失败保留，归因仍未闭合，没有因此排除回归。 |
| 普通 tools 构建入口 | 实际 CPU 与 Vulkan `build.py --tools` 流程均 exit 0，分别约 67.84 秒和 7.31 秒；这是构建回执，不是推理性能测量。 |
| Vulkan K/V 执行 smoke | 9 个 K/V 组合的 Prefill / Decode 共 18 行结果，exit 0，只有 1 repeat；这只是执行检查，不提供可靠吞吐或收益结论。 |
| 设备范围 | 本批次实际 GPU 后端是 Vulkan；没有 CUDA 设备，CUDA 未取得运行资格。 |

最终专项已对包含两个修复的 A 重新运行，但不能抵消最终全量 GPU 中保留的 FA 失败，也不能撤销较早未定位的 TOPK 失败。最初复杂 V fixture 观察到的 `0.0146484` 闭式偏差保留为历史反例；改用可精确构造的 binary fixture 后仍使用原来的误差门。这个过程没有证明曾实际复现 no-split 缺少同步的竞态。协议、映射与证据边界见 [本批次 findings](audit/details-findings-20261018.md)、[VALIDATION.md](VALIDATION.md) 和 [本批次机器摘要](audit/details-qualification-20261018.json)。

### 对 A 的维护成本

最终已核对 17 个 A 源码文件合计 +357/-43；另有 49 行自有 CI，manifest 共 18 路径、合计 +406/-43。本批次在既有 A 差异上新增 29 行。文件数本身不证明正确性；每项理由、消费者、删除条件和真实回归仍随 manifest 与验证记录维护。

### 仍保留的有用候选

`small:S04` 的格式和 imatrix 导出已由 B/C 重建；`small:S05` 的真实 callback 采集、DFS、checkpoint 及模型契约仍待迁。`server:Q15/Q16/Q17` 的首 chunk 状态、多 choice usage 与最终 SSE 事件，完整 thinking/lane parser、HTTP 取消和恢复，均没有被 CLI 文本输出替代。`backend:G14` 的正确设备 pinned host 分配/释放/查询仍是候选；G09 的正确性工作不能为它取得多 GPU 资格。

已保留的 Prefill/Decode 内存、统一物理 KV、准入、采样、corpus 和 logits 能力继续维护，各自先前测量保留原身份。没有新吞吐、峰值 VRAM、模型量化质量或跨形状一致性结论；逐项清单见 [MIGRATION.md](MIGRATION.md)。

## 同日较早发布候选记录：起点 3746ebf82585

下面保留该批次原文。其“本轮”“当前”、三个 B / 四个 C、47 条命令与 A 差异数字仅指起点 `3746ebf82585f47c88fb48097f6ae45cb972b031` 的较早资格，不能用于声称本页新增代码或最终 A 已完成相同验证。Meta graph-view UID 的源码纠正及 admission API 名称修正已保留。

本轮在 `clean/fork-20261018` 上继续处理有价值的小功能。固定上游A仍为 `cda0f9256665845cd9d6f3a0176880883f653a3d`，本轮起点为 `3746ebf82585f47c88fb48097f6ae45cb972b031`。交付三个独立B库与四个C编排包；新运行时能力都有真实消费者、失败合同、安装验证和可回查证据。本轮没有推送、打tag或发布发行包。

### 新增与调整

| 变化 | 设计与实际结果 |
|---|---|
| token/槽位准入 | `tq::scheduling::admission_budget` 同时限制已知token需求和驻留请求数，拥有自己的账目。domain/generation拒绝外来、陈旧、重复释放；失败不改变账目。 |
| 真实排队与槽位复用 | `tq-batch-compose --parallel` 将稳定请求ID与可复用A序列槽分开，FIFO准入、保持Prefill选择、消费全部输出并同步后删除序列，最后回收B预算。队列可长于驻留槽位。 |
| 有界完整logits采集 | 新 `tq-logits-compose` 用真实token做teacher forcing，记录TP5LOG2完整词表行。只调用安装后的A公开API；默认逐token，支持明确前缀。 |
| 可预测的文件发布 | 预检尺寸、逐行有限值、有限缓冲、失败终止、完成后无覆盖发布。原文件和symlink保留；stdout失败准确报告已完整发布的tape。 |
| Meta async偏移 | 一个A文件+4/-4，将逻辑chunk起点换算为child tensor偏移；原A非零get/set分别失败，修复后48个真实用例通过。 |
| recurrent生命周期 | 复用当前A，新增公开接口的clear、共享、邻居、dirty槽位复用及合法恢复回归；没有复制旧eager-clear实现。 |
| 独立构建与CI | 安装/搬迁验证加入第四个C和admission外部消费者，`--tools`补建A所需mtmd，新增独立CI配置。 |

完整用法在 [README.md](README.md)；各模块的README定义输入、所有权、退出码和平台范围。具体测试、命令与数值在 [VALIDATION.md](VALIDATION.md)，可移交机器摘要在 [audit/release-qualification-20261018.json](audit/release-qualification-20261018.json)。

#### B 只拥有预算，C 执行模型操作

准入需求是已知上界 `prompt_tokens + predict - 1`。B不探测显存，不持有A context，不代替A清理状态，也不因reservation析构就回收尚在使用的模型资源。C只有在真实序列删除成功后才释放reservation。失败的释放不会令容量被重复计入。

当前C是确定输入、有限生成长度的批处理程序。token或槽位不足时等待；无法容纳的单请求提前失败。请求生命周期的不可恢复错误令本次进程失败。本轮没有把这个合同扩写成在线HTTP取消、压力重试、抢占、LRU或未知长度生成的保证。

Prefill选择策略与FIFO准入分开：已经准入的提示由B保持当前Prefill，C在提示期间暂停Decode；队列资格按输入顺序。持续提示流的公平性没有获得证明。阶段内存仍只释放context自己的compute allocator，保留模型、KV/recurrent、输出和sampler；报告的字节不是进程峰值VRAM。

#### logits 文件可以复查，身份需要随运行保存

默认在每个输入token之后保存一行完整logits；该行预测下一个token。`--prefill N`保存前缀最后一行，随后逐token保存。输入始终来自文件，不由模型采样反馈。输出大小在创建context前检查；writer只有16KiB转换缓冲。

TP5LOG2保留既有二进制格式，头中没有模型、tokenizer、构建或设备身份。必须同时保存输入/模型摘要、命令、stdout、后端日志和构建版本。相同token ID和词表大小不能证明这些身份相同。

发布依赖POSIX同目录hard link与调用者控制的目录。完整文件出现前不会暴露部分目标；存在目标不覆盖。完整发布后stdout写入失败会退出1并说明文件已发布，文件保持完整。没有Windows、目录断电持久性或强制终止自动清理的承诺。

### 实际验证

本轮使用LFM2.5-8B-A1B-UD-IQ4_NL，同一个模型SHA-256在摘要中固定。分别运行CPU与单张RX6800 Vulkan；GPU模型日志实际确认25层卸载。

| 检查 | 通过范围 |
|---|---|
| 准入/阶段/复用 | 每后端7个运行、60个成功请求；同配置开关、串行容量对照、同槽位重复波次均通过，两后端共120请求。 |
| Meta传输 | 48例，axes0/1/2/mirror、F32/F16、不均分片、偏移、get/set与guard；真实参与者是两个CPU child backend。 |
| recurrent | 每后端24邻居快照、8共享快照检查，合法恢复继续最大绝对差0；模型范围为LFM2.5 hybrid convolution/KV，`n_rs_seq=0`。 |
| logits完整行 | 每后端15×128000个值；重复tape按位相同，同标量形状的suffix/prefix全行差0。 |
| logits独立C对照 | 相同提示token与提交形状，最终greedy token/最大logit在1e-5内通过；这项独立对照不提供第二份全词表行。 |
| 文件与错误 | 真实文件故障、竞争发布、symlink、缺行/多行、NaN、上下文与文件预算、stdout失败均有检查。 |
| 独立交付 | 47个命令通过：从固定A archive重建，三个B与四个C各自构建/安装，移除源码/构建副本和原前缀，再搬迁、编译消费者与运行真实模型。 |
| 常规入口 | CPU默认、Vulkan `--tools`完整构建退出0；边界12项与比较器20项通过。 |

本地检查通过不等于GitHub已经执行CI。新工作流已配置，尚未推送；没有远端CI或发布状态可宣称。

### 保留的已知数值限制

这次没有隐藏一个不符合最初预期的结果：LFM2.5三路与单路的greedy输出存在差异。该现象已在准入改动前的C中、关闭阶段释放时复现；单路只改变KV容量则输出相同。原始失败记录、旧二进制摘要、结果token和复现脚本均保留。因而本轮对照固定实际提交形状，跨形状观察另列。

完整logits诊断同样观察到chunked与scalar前缀的最终行差异，最大绝对差CPU为 `2.6373519897460938`，Vulkan为 `0.9928340911865234`。这些数值不能被称作已经证明无害的舍入误差；数值原因尚未定位，也没有承诺跨后端或跨批形状结果一致。固定相同形状的重复、行选择和状态复用检查仍保留严格判定。

recurrent测试也有明确范围：没有一般GDN、MTP、PLE或损坏snapshot事务恢复资格。源码审查发现坏restore可能先改变目的状态，不能把合法恢复成功写成所有错误恢复都不改变状态。

### 为什么有些旧细节没有直接搬入

| 旧细节 | 当前判断 |
|---|---|
| M33单cell清零优化 | 旧907–935ms到12.8–16.8ms减少的是旧fork新增的清零。当前A已有整buffer清零与图内延迟初始化；搬入旧路径会增加写与fence，因此保留目的、复用A、增加生命周期回归。 |
| Meta graph-view cache-key | 当前 A 会复用非零图 UID，但 graph view 的 UID 为零。旧 fork 才让 view 继承父图 UID；旧 key 修复与该优化共同设计，不能孤立搬入。 |
| pinned-readback retire | 当前Meta路径没有旧pending transaction；不孤立引入退休队列或pin。需要真实异步消费者、所有权与设备验证后再设计。 |

auto-fit/slack、PP reserve、idle LRU/cache affinity、物理recurrent槽位、取消和随机状态恢复、流式stop/usage、在线corpus、state envelope、imatrix与模型兼容等继续保留逐项候选，不按大模块名称删除。详见 [MIGRATION.md](MIGRATION.md)。

### 对 A 的维护成本

当前A代码差异为17文件、+328/-43；另有49行自有CI，manifest总计18个例外。每个文件登记精确内容、消费者、理由与删除条件。自有代码都在 `fork/`，三个B只依赖C++17标准库，C通过安装包调用公开A/B接口。原版server的小范围私有调度修改仍是明确例外。

上游公开结构体扩展不承诺跨版本ABI；更新A时须重建消费者。可用 [check-upstream.py](check-upstream.py) 检查内容边界，再执行独立交付与相关真实回归。原master和 `archive/pre-clean-20261018` 保留旧实现；没有为了清理分支删除历史功能。

本轮没有测量新吞吐、峰值VRAM或reset加速。CUDA、HIP、Metal、Windows、macOS、多GPU和异构Meta仍没有这次运行资格。714路径的审计记录保持原阅读深度，不因本轮完成的功能而扩大“已审完”的范围。
