<!-- two-requests-current-status-20261021 -->
## 2026-10-21：双请求保留版本与最终测量；每路 47 tok/s 目标未达到

保留版本、证据和入口见[双请求报告](../build-tp5-two-requests-20261021/TWO-REQUESTS-REPORT.md)；以下旧正文保留。本次增量为 3 个 A 源文件和 1 个 C 源文件，公共 API 不变；使用原始模型权重、每请求 262144 容量的 Q4_0 K/V，并保留 Swift 共享权重派生 view 生命周期修复及五 rank host 提交顺序 gate。

同 prompt 双请求组（n=2 个独立进程），每路 512 token：两路几何均值为 **37.629332 / 37.624558 tok/s**，aggregate 几何均值 **74.923025 tok/s**，范围 **74.847236–74.998891**。相同 C 版本的单请求观测（n=1）为 **46.071845 tok/s**；aggregate 对单请求的 **1.626222×** 仅作描述性比较。不同 prompt 的一个双请求案例为 **34.571716 / 34.899258 tok/s**，aggregate **69.062687 tok/s**。**两个请求各 47 tok/s（合计 94 tok/s）的目标未达到。** 历史 **47.34 tok/s** 只属于旧 F16 单请求配置。

完整 16-token trace 已通过，49 组五 rank 提交保持一致次序，但该完整 tracing 的共同生成期 GPU overlap 为 **0**。另一次相同保留版本的 kernel-only 采样已通过：共同输出阶段 GPU0–3 观察到跨请求算术/算术与算术/consume 的真实交集，GPU4 为零；含预填充的完整活动阶段五卡均有局部交集。由此可确认存在实际 GPU 重叠，但不能宣称五卡共同生成期都已重叠或通信完全隐藏。各次观测和证据分别保留，详见双请求报告。

---

# 验证记录

<!-- TP5_NATIVE_SPEED_20261021_FINAL -->
## Latest retained result: native TP5, 2026-10-21

The current retained native TP5 configuration is the 2026-10-21 four-pair-qualified build. Older “current”, “final”, and gap-failure entries below are historical. The user's relaxed numerical policy applies to native acceptance; explicit strict-reference modes retain their separate gates.

Four balanced pairs against fixed master `46a6fd8a129091741f9ebc6431f7f610abe1ccfc` passed: geometric-mean throughput **47.343475 vs 45.366353 token/s**, paired ratio **1.043581249**, two-sided 95% CI **[1.037961336, 1.049231589]**. Every run includes all 511 intervals from 512 natural outputs.

Independent audit recomputed all 8 × 511 raw intervals. Five ranks, original model weights, actual 262144 capacity, natural token chaining, finite outputs, drain and zero final ownership budgets passed. This qualifies the specified one-window workload, not a filled-262144-history benchmark or a new complete multi-window/fault suite.

See [final report](../build-tp5-native-speed-20261021/NATIVE-SPEED-REPORT.md) and [audit index](audit/tp5-native-speed-20261021.json).

<!-- /TP5_NATIVE_SPEED_20261021_FINAL -->

## 2026-10-21：TP5 有效改动已落地，master 性能目标仍未达到

最新保留镜像与完整证据见 `build-tp5-gap-repair-20261021/REPAIR-REPORT.md` 和 `fork/audit/tp5-gap-repair-20261021.json`。A 为 `fa-prefix-install`，C 为 `finite-greedy-compose-install/bin/tq-tp5-compose`；`run-retained.py` 显式绑定这两个已安装产物。后文旧镜像的“当前”“最终”保留历史身份，不覆盖本条更新。

保留 gfx1030 的 tile32 attention 与稳定前缀并行指数、HC BF16 K320 的保序交错 rows8、indexed Q2 K128 的保序 rows2，以及独立 B sampling 的 finite-greedy API。A 的新增生产改动仍限定在三个既有 overlay 文件；C 通过公开安装包组合 TQSampling。relay 的临时设备时钟诊断已撤出生产源码，K256 Q3/Q5 展开候选因增量速度下降而撤回。

最终 A+C 在冻结输入轨迹下的512行、127139840个完整 logits 逐位相同，超门/非有限/greedy差异均0，五个实际rank正常drain、预算归零。原F64 attention oracle实际五卡320例通过。B由只含sampling的副本独立构建，Release AVX2与scalar-only各2/2测试通过；C由只含自身源码的副本消费已安装A/B构建，新的外部安装消费者configure/build/run通过。上述安装组合证据不扩为Swift长程状态、填满262k历史或全部多窗口故障资格。

单次相邻自然greedy增量对：finite-greedy **31.663348→32.247912 token/s**，其提交阶段均值 **0.944519→0.058066 ms/输出**；最终attention前缀 **31.581164→32.731905 token/s**。两对使用不同独立进程，不相乘、不提供置信区间，也不是最终master非劣效资格。固定master较早四组平衡运行几何均值为45.300541 token/s；目标仍未完成。原reference、误差门、权重编码、F32 activation/query、F16 KV、CPU rank顺序归约、512输出/511完整提交间隔均未改变。

一次仅用于归因的设备时钟诊断显示各rank归约间工作约24–26ms/步，显式等待约0.74–2.92ms/步；前者包含计算、调度和尾部，不能称纯kernel计算。36个GDN层各两次join、12个FA层各一次join，最重段覆盖工作不同且mirrored工作可能越过命名terminal，不能把段差直接归因于FA。诊断与可复现离线解析均不提供性能资格。完整取舍、失败候选、精确源/二进制身份以新报告为准。

## 2026-10-20：Swift 同模型迁移，尚未验收

本轮目标改为 `/opt/strata/data/models/swift-q2_0` 的实际两分片 Swift Qwen4EXP：单请求、五张 RX 6800、context **262144**、实际生成 **512 token**，速度不低于固定 master。下文 Qwen3-8B/Q8_0 的已完成资格只保留历史身份，不覆盖当前 Swift 源码或安装镜像。当前输入和比较合同见 `build-tp5-swift-20261020/model-catalog.json`、`comparison-contract.json`。

已实际分配完整 context/sequence/KV 容量，各为 262144；Swift 使用 F16 KV、F32 query 与算术。28.8 GB 的原 IQ4_NL PLE 表由模型持有，CPU 只在输入准备阶段读取所选行，后续模型计算仍在 GPU。混合量化投影使用原编码权重与 F32 activation；没有重新量化 activation、模型遗漏或提交后回退。

当前单次筛选的 512 输出 / 511 commit 间隔结果如下；不是独立配对确认，也不构成速度达标。

| 镜像与条件 | token/s | 回执 |
| --- | ---: | --- |
| 固定 master，同一 Swift，context262144 | 43.589 | `master-full-context-512` |
| Swift PLE 输入复用后 | 12.4483 | `swift-ple-reuse-512` |
| 完整签名线性收集、steady-clock 等待 | 13.3428 | `swift-steady-hash-512` |
| 紧凑混合投影、已退休通道不再轮询 | 15.8257 | `swift-compact-retired-512` |
| 按 Meta 原有定义寿命复用已校验执行实例 | 18.6287 | `swift-owned-definition-512` |
| 既有异步接口批量准备 host 输入 | 20.4254 | `swift-input-batch-512` |
| 同一输入批处理，原串行 attention，全 logits | 20.6144 | `swift-input-batch-serial-fa-512` |
| 四值复用原量化块元数据，全 logits | 21.4559 | `swift-packed-four-full-logits-512` |

回执均在 `build-tp5-swift-20261020/`。定义复用运行正常退出，523 epochs、84 channels，输入/输出退休各 43921，关闭后预算与活动请求均为 0。这里只证明完整容量已分配，并未从已填满 262k 的历史开始生成。下面的选定512行全 logits 已取得局部数值证据；长程状态、生命周期、安装搬迁和最终同模型配对性能仍未完成，不能称为合格发布。

设备等待曾在首 epoch 误报超时。恢复完整图签名后仍失败，排除“只因跳过图检查”的归因；临时精确诊断实际读到 `clock64 start=8217203, now=1253691`，无符号差值下溢触发 error3。失败实际提交掩码均为31、unknown=0，真实 drain 后预算归零。现在使用 HIP 的 `wall_clock64()`，并在 provider 创建时查询每 rank 的 `hipDeviceAttributeWallClockRate`，将 600 秒后备界限换算成稳定时钟 tick；正常 deadline/abort 和完成证明不变。五卡独立15秒探针均报告100000 kHz，修复后的上述512输出运行通过；临时控制字段已移除。原失败保留在 `swift-exact-timer-8`，不能因为后续成功而删除。

先前首 epoch 失败的候选保留原记录，但不能再将该失败归因于定义失效。确认设备时钟根因后，执行器按 Meta 原有 graph UID/重建边界管理定义：每次重新物化都调用 prepare，即使 terminal 地址未变，仍重新检查完整拓扑、use count、tensor/source/view 布局与地址；两次 prepare 之间只允许输入内容变化，复用已经检查的执行实例。上述18.6287是这一路径的实际512输出筛选，输出与此前逐步校验路径相同。上传、并行 launch、替换 copy kernel 等未确认收益的分叉没有留在生产路径。

已另取得原路径五卡 CPU-staged 的512输出完整参考，`swift-staged-reference-512` 保存508559360字节 logits，并记录所用动态库身份。原 BF16 改为紧凑 F32 累加的候选在相同输入轨迹下比较 **512行 / 127139840值**，超门与非有限值均为0，argmax差异0，max_abs **0.002171039581298828**；门保持 `.002 + .001*abs(reference)`。记录为 `swift-bf16-full-logits-512-comparison.json`，不是其它分片方案的资格。该候选单独筛选17.6174 token/s，未确认性能收益，不据此晋级默认。

上述整类 BF16 替换已撤下，精确候选源码与原结果保存在 `bf16-compact-withdrawn.json`。重复分组 Q/O 分片在完整轨迹超门7值，也已撤下，见 `gqa-repeated-withdrawn.json`。attention 的分区归并、四 key 保序预取及禁用内部展开的预取版本均未通过原512行数值门；最后两者分别超门270693值、max_abs 0.3245208263397217。保序预取从第0行已有微小差异，第426行出现大偏差；禁用展开未改变其数值结果，尚未定位具体编译器/算术原因。`key-prefetch-withdrawn.json` 保留失败源码和三个比较回执；生产路径恢复原串行 attention，不能以相同greedy输出或局部F64算例通过抵消整模型失败。

输入上传批处理只复用既有 backend异步复制接口：包含USER输入的全host非权重输入组可一起排队，空view不产生读写，首个非空复制仍先等旧copy-slot；所有上传在模型分发和调用返回前排空。实际仍是每token 70次H2D复制，所观察decode CPU时间约9.53→2.09ms。真实两CPU子后端合同已通过完整55例，以及独立zero/get/set模式16/24/24例；这是数值、拒绝边界及源寿命证据，不是HIP故障注入。`swift-input-batch-serial-fa-512-comparison.json` 的全部127139840值与冻结参考相同，max_abs为0；由此排除该输入批处理是上述attention失败的原因。

保留的量化四值解码候选只复用相邻四码共有的Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS元数据和打包读取，原Q8/Q2解码、四条F32 FMA链及归约顺序不变。`swift-packed-four-full-logits-512-comparison.json` 同样逐值相同；局部混合投影、attention和ordered-fold CTests实际通过。21.4559仍只是单次筛选，不是独立配对收益确认，也远未满足master速度门；当前候选身份保存在 `packed-four-qualified-screen.json`。

固定 master 的 HC 融合也补做了显式环境对照。`master-hc-full-logits/{unfused,default}` 各实际生成16输出；真正移除 disable 变量后，默认融合在第1个生成 token 开始分歧并产生异常文本。共同输入前缀的两行 max_abs 分别为1.927751和15.541243；之后输入轨迹已不同，不能将余下行当作成对数值证据。后续仍以原 segment-disabled43.589筛选为速度参照，不使用错误融合路径降低正确性要求，也不以此掩盖当前性能缺口。

## 2026-10-19：TP5 单流性能修复与整步原生重放

当前机器索引为 [`audit/tp5-performance-20261019.json`](audit/tp5-performance-20261019.json)。后续旧 TP5 章节保留原生重放前的镜像、数字和范围；不把它们作为本轮新二进制的验收。A pin 不变，已审覆盖仍为 48 个路径；新精确 SHA 以 manifest 和当前 source guard 为准。

### 问题、最小修复与实际命中

冻结 master 为 `46a6fd8a129091741f9ebc6431f7f610abe1ccfc`，未切换或重置用户工作树。其 `artifacts/perf-gap/run-20261009T0145/REPORT.md` 确有 no-MTP 单请求 ordinary/compact **64.77485/67.14624 token/s**，但模型是 APEX-I-Compact 混合精度/MoE、激进 HC/LateBind、Vulkan，不是本轮 dense Qwen3-8B Q8_0/HIP。这里只继承完整预提交/重放等有效机制，不复制旧框架，不以模型差异代替修复。

原路径每 token 重复数千 host kernel API；RELAY 忙轮询还反复查询尾事件、切五设备。图 reserve 日志属于启动规划，未发现每 token 重建，所以没有放宽 context/KV 失效规则。新 A 私有 provider 接收完整 prepared graph 数组，用至多两个精确签名缓存项预热、捕获、重放，所有五个 executable 准备完成后才实际提交。公共 relay ABI 1/layout 不变，私有 provider ABI 2。捕获的 epoch 是 drain 后更新的稳定 activation word，不是旧标量；未结束的 capture 不能冒充 finished。每边界 producer-copy/ready 与 wait/copy/last-reader-ack 合为两个有界单 workgroup，四值搬运不引入 16-byte view 对齐要求。Q8 投影保持原权重解码与 F32 activation，以无 LDS barrier 的四输出 block 替代旧归约；C 的 SSE2 fold 保留各元素 rank 0→4 的 F32 顺序与有限值门。

`final-trace-summary.json` 绑定相同八输出输入轨迹、12 epochs 和五个实际 BDF：

| 观察量（profiler 扰动，不能当速度） | 原路径 | 最终路径 |
| --- | ---: | ---: |
| `hipLaunchKernel` API | 66,940 | 17,970 |
| `hipGraphLaunch` API | 0 | 50 |
| `hipEventQuery` | 158,874 | 545 |
| `hipSetDevice` | 390,944 | 4,449 |
| profiler GPU0 实际 kernel dispatch | 13,888 | 11,308 |

最终短 trace 包含直接预热/捕获。其最后四个实际 ITL 为 **0 kernel API / 20 graph launches**，即每步五个 native graph launch；不是停止了 GPU 计算。每个物理 BDF 都有 3,020 次 Q8/F32 投影和各 860 次 producer/consumer kernel。实际 GPU 工作继续执行，`rank_graphs` 是 logical prepared stages、`rank_relay_nodes` 是 accepted kernel nodes，不冒充 host API 次数或完成证明。

### 独立进程确认，不使用 aggregate / 请求数

`build-tp5-performance-20261019/paired-selected/preregistration.json` 在执行前冻结了两套安装镜像、模型、teacher 和顺序。四个 block 每个分别启动原路径/off/on，唯一请求、五 prompt tokens、64 个相同 teacher 输出、ctx512、threads4、无 profiler/dump/编译/并行 GPU 作业。主终点 `(64-1)/(last_commit-first_commit)` 保留其中所有 capture 成本，不删慢间隔；各 block 的 log speed ratio 以 Student-t df=3 给区间。

| 同模型单请求条件 | 几何平均 token/s |
| --- | ---: |
| 保留的原安装镜像 / FA off | 9.1784 |
| 最终修复镜像 / FA off | **54.9584** |
| 最终修复镜像 / FA on | 49.9261 |

off/原路径 **5.98781**，95% CI **[5.29900,6.76615]**；on/off **0.90844**，CI **[0.88025,0.93752]**，所以 FA 默认继续 off。此前 11.30575、56.23309 仅筛选单次值，未混作确认统计。样本为每请求 63 个间隔，不能宣称总体 p99；时间不含模型/context 初始化、网络和准入前排队。

另六进程、三配对 block 复用旧四请求×193 prompt/32 teacher 负载：一窗/两窗每流 38.4234/39.5584，two/one **1.02954**，CI **[0.75777,1.39878]**，没有确认额外并发收益。保留现有自动窗口策略（单请求一窗，多请求两窗），不把旧 1.71597 倍作为当前结论。尝试缩窄控制轮询 frontier 的单次候选退化，已撤销；相应试验回执保留，不增加调参开关。

### 原数值、状态与失败合同

另外直接读取已保留、未重生成的原镜像单 GPU reference，逐请求匹配共同前八个输入行和全部词表值，避免新 reference 与新 TP5 共用优化 kernel 而相互自证。`cross-generation-numerical-comparison.json` 每次比较 32 行 / 4,861,952 值；当前 reference / TP5 相对原参考 max abs 分别 `1.52587890625e-5` / `2.193450927734375e-5`，原 `.002/.001` 门均零超差、零非有限、argmax 分歧 0。性能确认的十二个单流进程也全部 sampled==emitted，teacher 没有替换它们实际选出的 greedy token。

`selected-qualification/full-off` 和 `full-on` 各运行完整 16-token main/slow/cancel/epoch-8-abort。每模式 64 行 / 9,723,904 个 logits，在原 `.002 + .001*abs(reference)` 门下零非有限、零超差；max abs 分别 `6.961822509765625e-5` / `5.53131103515625e-5`。慢消费者持有输出时 peer 先产生三个 token；取消请求不阻止其余三请求；晚期 abort 不发布成功，实际 attempted/accepted/completed=31、unknown=0、finished=1 后才移除状态和归还预算。

`selected-qualification/long-comparison.json` 为另一个单请求、288 个完整输出、43,757,568 个值，输入位置 4→291；max abs `0.00052642822265625`、零超差、argmax 分歧 0。实际 HIP API trace 有 15 个 capture begins / 1,445 graph launches，其中跨 input position255 后有五个新 captures；证明真实 KV 256→512 重新绑定后继续运行，不伪装固定地址即可重放。这个带 trace 的运行不参与性能统计。

最终 CTest 包含五物理 GPU 的 **1,210 Q8/F32 oracle cases**（K/N/tokens/四字节非十六字节对齐/尾部与拒绝保护），**180 F32 attention cases**，保序 CPU fold 的 signed-zero/消去/非有限/溢出/unaligned/tail 回归以及 CLI help。独立 dyadic-F64 Q8 oracle 和 F64 attention oracle 保持原门。`asan-qualification` 的 host C/C++ main + epoch8 abort、16-token 全 logits 通过；未见 ASan/runtime error。C/C++ flags 有 address sanitizer，HIP flags 未启用；不声称覆盖 HIP 翻译单元、GPU 访存、竞态或 leaks（detect_leaks=0）。两个只读专项审查没有发现可行动缺陷，不充当设备资格。

### 新安装、搬迁和镜像绑定

`package-proof/summary.json` 的 A + 七 B + 七 C 共 15 组件全新安装后搬迁。原 prefix 和 stage 在 consumer configure/build 以及所有 GPU 调用时真实不存在；用户源码树未搬走。外部消费者只用 `find_package(TQTP5Compose)`、安装公共头和 `TQTP5Compose::runtime`，compile/link 路径不含 A/C 私有头或原 prefix，`ldd` 解析到 relocated。它显式 FA on、双窗、两请求各八输出；另测默认 FA off 单请求八输出、安装后的 main+epoch8 abort（16-token，原门零超差）及零/越界/未配对 abort-epoch 提交前拒绝。它不是第二次完整 slow/cancel 矩阵。

`performance-image-binding.json` 验证发布 CLI 和所有 native 库与确认镜像逐字节相同。原 prefix 已在 finally 恢复，隐藏路径和 stage 均不存在；新 relocated 包继续保留。原始命令、退出、stderr、tapes、逐流事件、统计与 payload 摘要都在 `build-tp5-performance-20261019/`。没有新增库、通用执行器、低精度 fallback、host continuation 或 GPU allreduce；支持范围仍为该模型/量化/五卡平台、普通 greedy decode、每请求 512 reservation、最多四请求/双窗，不扩为 MTP、LoRA、RERoT+TP5、其它架构或长上下文。

## 2026-10-19：TP5 五卡 HIP、Flash Attention 数值与生命周期验收

本节是完整模型运行，不复用下文独立 HIP 平台探针或旧候选的 PASS。当前 A pin 仍为
`cda0f9256665845cd9d6f3a0176880883f653a3d`，已审 manifest 共 **48 个 A 路径**，
其中已包括两个新增 Q8/F32 kernel 文件，不能再加二。
`build-tp5-runtime/final-attention-boundary/stdout.log` 记录 48，配套 `exit.json` 为 0。
七个独立 B、七个 C 的源代码和安装接口保留各自边界；C 的模型入口是
[TQTP5Compose](tp5-compose/README.md)。完整身份、命令与回执由
[本轮机器索引](audit/tp5-acceptance-20261019.json) 绑定，不能由目录名推导另一构建通过。

### 实际模型与执行路径

- 权重：`Qwen3-8B-Q8_0.gguf`，**8,709,519,168 bytes**，SHA-256
  `0cfbf745760f07a76ddeb358dd025a27f2e11d1ca9c9a4169a373d52990fe86e`。
- 设备：五张 RX 6800 / HIP gfx1030；rank BDF 顺序为
  `0000:0f:00.0, 0000:0c:00.0, 0000:06:00.0, 0000:09:00.0, 0000:03:00.0`。
- 五卡分担真实模型张量计算，CPU 按 rank 顺序归约 F32，producer / wait / consumer
  整链在 epoch 开始时预提交；两个 context 的输入、KV、可变 arena 和原生执行域独立。
- 数值策略为 `q8_0-decoded-weights-f32-activations-ordered-f32`；K/V cache 与实际
  attention 操作数保持 F32。FA 默认 `off`，采用本工作负载实测最快的合格模式；显式
  `on` 已取得完整数值/生命周期资格，但不因功能合格而晋级为性能默认。
  两种模式均不得静默回退，报告检查每个 context 的 resolved 设置。
- 复用 A 已有的 fused online F32 attention kernel，保留 F32 操作数边界；不是将 FA
  改名为拆开的 QK / softmax / V 图。普通 FA 分派与精度选择不变；共享 online/banded
  F32 kernel 在 K/V 读取前跳过 -INF masked key，避免无用 QK 与 masked NaN 污染，
  保持有限未屏蔽项累加顺序。这改善共享 masked-nonfinite 语义，不宣称其他后端一切不变。
- 原四个 193-token prompt 保留；strict preflight 跳过合法无输出 prefill 的空 tensor，
  不制造 logits、不缩短 prompt。原 `per-stream-fa` 失败与 `per-stream-fa-prefill-fixed`
  较慢候选分别保留，不能用换工作负载的方法隐藏它们。

### 本轮实际通过的门

最终结果位于 `build-tp5-runtime/acceptance-fa-final/summary.json`。`main,slow,cancel,abort`
全部执行，每请求预测八 token；`all_contracts_passed=true`，不是缩短成一 token 的冒烟。

| 门 | 实际结果 | 范围与限制 |
| --- | --- | --- |
| 同设置单卡参考与 TP5 teacher forcing | **32 行 / 4,861,952 logits**，非有限值 0、超差 0、argmax 行差 0 | 固定 `abs(diff) <= 0.002 + 0.001 * abs(reference)`；最大绝对差 `2.193450927734375e-5`。不代表全模型 F64 oracle 或任意 prompt/形状等价 |
| 双窗口与慢消费者 | 两个 context 实际重叠；window 0 延迟 2000 ms，首个延迟 token 前另一窗口提交 2 token | 不是仅以线程数或设备繁忙推导独立前进 |
| 请求取消 | 请求 0 在接受两 token 后取消；请求 **1、2、3** 随后仍有提交 | 同窗口幸存请求也继续；结束后 token 与 active-request 预算均为 0 |
| 注入 abort 与真实 native drain | 达到 window 0/channel 0 注入，受控退出 **1**，无成功报告 | 两窗口 attempted/accepted/completed 均为 **31**，unknown=0、finished=1；请求预算归零。不是 watchdog kill 或 fail-stop exit 125 |
| Q8/F32 HIP arithmetic oracle | 五设备 **170 例通过** | 独立精确 dyadic 点积、padding/source 保留与非法图拒绝，不能代替完整模型门 |
| F32 FA 独立 arithmetic oracle | 最终 `fa-final-oracles/summary.json` 五设备 **180 例通过**，最大绝对差 `2.24640083e-6`，非有限/超差均为 0 | F64 stable-softmax；固定绝对门 `2e-5`；F16-exact、non-F16-exact、always-masked K/V NaN 三组，rows 1..4、full/local heads、源与分配 guard 保留 |
| 最终已发布安装包实际 kernel trace | FA-on **720 次 fused FA launch**，物理 GPU 0..4 各 **144 次**；off 为 **0** | `hip-package-proof/final-dispatch-summary.json` 将两个 trace 绑定上述五个实际 BDF；实际 kernel 为 `flash_attn_ext_banded_f32<128, 4>`，报告布尔值本身不等于 kernel receipt |
| FA-off 完整八 token 数值与生命周期 | `acceptance-fa-off-final` 的 main/slow/cancel/abort 全部通过；32 行 / 4,861,952 logits，max_abs `2.193450927734375e-5`，非有限/超差 0 | 与同为 off 的单卡参考比较，仍用 `.002/.001` 门；实际 off trace 的 FA launch 为 **0** |
| FA-on 32 token 扩展数值 | `acceptance-fa-f32-long` 的 main 通过；**128 行 / 19,447,808 logits**，max_abs `9.059906005859375e-5`，非有限/超差/argmax 行差均为 0 | 同一 `.002/.001` 门；这是 main-only，不冒充 32-token 生命周期全矩阵或长上下文资格 |
| 先前候选 host ASan | `acceptance-fa-asan` 的 **main + abort** 通过；每请求四 token，16 行 / 2,430,976 logits，max_abs `2.193450927734375e-5` | 保留其原候选身份；host-only、leak detection 关闭，非全四场景，也不检测 GPU kernel 越界或证明没有泄漏 |
| 当前源码长 prompt host ASan | `asan-long-final.json` 使用原四个 **193-token prompt**、两个 context、FA on、每请求生成两 token，退出 **0**，最后预算为 0 | 当前无输出 prefill 修复路径；host-only、leak detection 关闭，不扩写成全场景 sanitizer、GPU sanitizer 或无泄漏资格 |
| 最终 HIP 构建与安装阶段 | 七 B / 七 C 均构建、测试、安装完成，A public seams **11/11** 通过 | `final-hip-build` 随后的 guard 因本轮并发修改 `AGENTS.md` 而拒绝旧 digest，整条命令当时退出 1；属于预期身份更新，不是模型或 seam 回归，也不写成流水线 exit 0 |
| 请求延迟叠加另一窗口 abort | `hip-package-proof/delay-abort-valid/summary.json` 通过选定场景：W0 请求延迟、W1 观察到 abort，受控退出 1 | 两窗口 completed=31、unknown=0、finished=1，预算归零且无成功报告；**未单独 instrument 精确 held-borrow 分支，不声称该分支已覆盖** |

完整来源隔离另有 **85 条命令通过**，但保存的
`build-tp5-runtime/isolation/upstream-exceptions.json` 是 **45-path** 的前 FA 候选。
它证明该候选七 B / 七 C 的 archive 构建、移除副本与搬迁消费，不证明最终 48-path
FA 构建完整重跑了同一 archive 流程。当前最终 HIP 安装包已另行取得下述真实搬迁与
安装消费者回执，不能把两个不同证明混为一谈。当前 48-path source guard、完整源码身份
与最终机器索引见各自最终回执；不凭局部 PASS 提前宣布所有门全绿。当前源码 ASan 和
on/off trace 也有独立回执，不由早期 PASS 推导，不扩大 sanitizer 覆盖范围。

### 最终 HIP 安装包、搬迁与公开消费者

`build-tp5-runtime/hip-package-proof/summary.json` 为 **passed-selected-contracts**：
将 **A + 七 B + 七 C 共 15 个组件**重新安装到 stage，再改名为 relocated。消费者
configure/build 和全部 GPU 调用期间，原 stage 与旧 install 前缀均真实不可用，证明后才恢复。
原用户源码树未搬走；证明的是消费者只用安装接口，而非整个源码树隔离重建。

- 外部项目仅 `find_package(TQTP5Compose)`，使用 installed headers/imported targets；
  `compile-path-validation.json` 未发现旧前缀或私有源码头，`ldd` 指向 relocated libraries。
- 外部 consumer 两请求显式 FA on、两个 context，实际五 rank 执行并以预算 0 结束；
  直接 CLI 的两请求默认 off 也执行成功。单请求 CLI 自动选一窗口/off，三请求一窗口
  显式 on 均实际通过，而不是只运行 `--help`。
- 实际安装 helper 路径是 **`share/TQTP5Compose/real-tp5.py`**。从搬迁前缀调用该 helper，
  main-only、predict4 的完整 logits 门通过；不扩写为搬迁后全生命周期重跑。
- 最初 harness 猜错 share 路径、在启动子测试前退出 2 的失败仍保留；查安装清单改正路径
  后只续跑未完成步骤，没有重复已完成 setup 或覆盖失败日志。
- `final-dispatch-summary.json` 的最终包 on/off trace 为 720/0，on 每卡 144 次，验证五个 BDF。
- `performance-image-binding.json` 为 **identical-native-images**：最终产品的
  `libllama.so`、`libggml-base.so`、`libggml-hip.so` 与 12 个性能进程使用的原生镜像相同。
  发布 CLI 的默认选项改为 off，但每个性能 case 都显式传 on/off；没有在测量后换 kernel。

### 逐流性能与默认模式

`build-tp5-runtime/per-stream-fa-final/summary.json` 实际记录 **12 个进程、三个配对 block**。
每次四个固定请求，每请求 193 prompt token、32 个固定 teacher 输出；没有 profiler、
full-logits capture 或并发硬件工作。表中是每流几何平均速率，不是总吞吐：

| FA | 窗口 | 每流 token/s | admission-to-first-commit TTFT（ms） |
| --- | ---: | ---: | ---: |
| off | 1 | 8.9853 | 25568.1 |
| off | 2 | **15.4185** | 14059.6 |
| on | 1 | 7.7510 | 26719.4 |
| on | 2 | 13.4642 | 14748.4 |

两窗口 FA on/off 速率比 **0.87325**，95% CI **[0.83966, 0.90818]**，约慢 **12.7%**。
区间采用每 block 四请求 log-ratio 均值配对、Student t df=2。只有三个独立 block，
不是把每个 token 当独立样本；每请求仅 31 个 ITL，p99 接近最大值，不是稳定总体尾部估计。
TTFT 从请求 admission 到首个 token commit，**排除模型加载和网络**。
因此默认选择 off；显式 on 保留完整功能资格，但没有 FA 性能晋级。
用户询问速度并不等于要求默认启用 FA，先前的相反假设已纠正。

同一三个配对 block 的 off 模式两/一窗口速率比 **1.71596659**，95% CI
**[1.28598327,2.28971978]**（Student t df=2），见 `per-stream-fa-final/default-decision.json`。
这支持该固定工作负载下的两窗口默认，不扩写为通用并发收益。上述 TTFT 还排除
model/context startup 与 admission 前 queueing，不能解读为端到端请求延迟。

### 失败证据与支持边界

`acceptance-fa-on`、`acceptance-fa-precise`、`acceptance-fa-cast` 中的早期 FA 数值失败
全部保留：只修累加精度而仍让模型图先转 F16，不能取得当前 F32 profile 的资格。
早期 Q8、ASan、`acceptance-fa-f32` 的反例和通过回执也保留原候选身份。没有通过改输入、放宽阈值、
忽略 logits、关闭 FA 或缩短生命周期场景得到本轮 PASS。

当前只授予上述 Qwen3-8B Q8_0、五张 RX 6800、每请求不超过 512 token 的已执行范围。
LFM2 MoE 的 TP5 仍不支持；其他模型/后端、长上下文、MTP、RERoT+TP5、HTTP 取消恢复
和多设备 PP 不从这里获得资格。下文历史 Vulkan FA 与 TOPK 失败继续保留。
逐流数据只适用于上述固定短测；没有通用加速、FA 性能默认晋级、峰值显存或稳定 p99 结论。

复现构建入口为 `python3 fork/build.py --backend hip --tp5 --hip-root /absolute/path/to/rocm
--build-dir build-tp5 --jobs 8`；完整模型命令和显式 `.002/.001` 数值门见
[C 的验收说明](tp5-compose/README.md#full-logits-and-lifecycle-acceptance)。使用新的输出目录，
不覆盖失败证据。后文日期相同的旧批次仍保持各自源码与运行身份。

## 2026-10-19：DevOps 收尾与当前交付验收

本节是本次实际重验，不复用后文历史 PASS。起点为
`90aa90e2a7d12aeb181dd11c9c04e367034dc2a8`，最终代码以
[机器索引](audit/devops-acceptance-20261019.json) 中的工作树源码、manifest、二进制、
安装库和原始回执 SHA-256 为准。A pin 不变；例外由 34 增为 **35 个路径**，
唯一新增的 A 改动是 Vulkan CMake 传播已声明 SPIRV-Headers 的私有 include 目录。
没有更改模型数学、数值门或旧失败记录。GitHub workflow 保持原内容；本次验证本地
入口，不声称远端 Actions 已执行。

环境为 Linux x86_64、GCC 15.2、CMake 4.2.3、Python 3.14.4、Mesa 26.0.8，
HIP 7.15.26333 / AMD Clang 23。平台探针实际使用五张 RX 6800；模型 Vulkan 用例
用 `GGML_VK_VISIBLE_DEVICES=0` 限定单卡，不能写成五卡模型或 PP 资格。
重新取得固定模型版本 `unsloth/LFM2.5-8B-A1B-GGUF@37563684c4a625e9958906fd924e15095cf9c896`
中的 `LFM2.5-8B-A1B-UD-IQ4_NL.gguf`，实际 SHA-256 为
`a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a`，
与历史权重身份相同；本次结果仍属于新构建和新执行。

### 发现、复现与修复

| 问题 | 失败前实际观察 | 修复与通过证据 |
| --- | --- | --- |
| 非系统 SPIRV-Headers 未进入编译 include | 设置 `CMAKE_PREFIX_PATH` 后 configure 成功，但编译在 `spirv/unified1/spirv.hpp` 失败 | A 的 Vulkan target 读取已找到的 imported target include 属性；未添加公开或安装期头依赖。相同前缀、不设置 CPATH 的完整 Vulkan tools 构建和公开消费者通过 |
| HIP 工具遗漏 runtime 搜索目录 | 构建成功，CLI CTest 因找不到 `libamdhip64.so.7` 退出 127 | 两个工具使用导入 HIP runtime 的真实目录生成 build/install RPATH；3/3 CTest 通过，安装前缀搬迁后清除 LD_LIBRARY_PATH 仍完成五卡探针和定标 |
| bench 证据缺项与配错仍被接受 | 原 validator 接受仅一项测量；完整 bench 的 rounds=1234 与 report=2000 配对仍退出 0 | 强制四项测量、单位和样本数，并检查共同设备/版本/轮数字段；相同反例退出 1，未放松任何硬件门 |
| 拒绝非法 Unicode key 时输出崩溃 | 未知 `"\ud800"` key 触发 UnicodeEncodeError，只有 traceback | 对字段名也校验 Unicode scalar；同反例稳定退出 1、stderr 为空，顶层和嵌套反例进入自测 |
| CPU 入口意外启用 HIP discovery | 原探针在 HIP 可发现时，即使 BUILD_TESTING=OFF 仍初始化 HIP compiler | `build.py` 显式关闭 probe HIP 构建并转发测试选项；单独 CPU-only/tests-OFF 冒烟在可发现 HIP 的环境中不生成 HIP/compiler 或测试可执行文件 |
| 包消费受外部搜索环境污染且覆盖有缺口 | 人工 CPATH 头文件可使真实编译失败；旧流程继承该搜索环境 | 隔离入口清理编译器/加载器路径、禁用 CMake 环境和 package-root 搜索，验证导入库位于搬迁前缀；污染环境下完整 80 命令通过，新增 Relay 与 Compose 传递依赖消费者通过 |

定标 kernel 另外修复了 64 个线程竞争写同一个输出字的问题：只有一个线程写入映射
设备地址；不能证明完成时不释放仍可能在飞的页。五卡实际定标退出 0；这不构成
故障设备的 DMA 终止或时钟速率通用保证。probe CLI 的空输出/bench 路径在设备访问前
退出 2，真实 HIP CLI 回归通过。

初次新增 Relay 安装消费者错误地要求最后读者完成后 occupied 立即归零，实际 B 合同是
在下一次 reserve 时复用。修正的是消费者断言，没有削弱 B 的最后读者或 credit 门。
早期 workflow 编辑与新增 A 文件尚未登记的中间状态也被 guard 拒绝；workflow 已恢复，
最终 35-path guard 通过。所有失败尝试保留在本轮 build 目录和索引中。

### 本轮实际通过的门

各行分母不同，不相加成总测试数。原始日志与结构化回执位于 `build-devops-20261019/`，
摘要和身份清单由上述机器索引绑定。

| 门 | 实际结果 | 证明范围 |
| --- | --- | --- |
| A 边界 / 全词表 reader / 指南 CPU 程序 | 12/12、20/20；四个内嵌程序全部执行成功 | 包含固定旧工具的合成反例；不表示旧工具漏洞已被改写或 GPU 已通过 |
| 普通 CPU / Vulkan `fork/build.py --tools` | 两端均退出 0；七 B、六 C、CPU probe checks 及 server/bench/mtmd 构建安装成功 | Vulkan 只使用声明的非系统 SPIRV-Headers 前缀，不依赖隐式头文件搜索环境 |
| 安装 A public seams | CPU 11/11；Vulkan 安装 20/20 CTest 通过 | Vulkan 安装的集合同时含 CPU 与 GPU 测项，不是 20 项全部在 GPU 执行 |
| 完整来源隔离与搬迁 | **80/80 命令通过**，未使用 `--a-prefix` | A 从 pin archive 加 35 个已审 overlay 构建；七 B / 六 C 的源码、build 副本及原前缀删除后，消费者、转换和全部六个 CLI 实际运行 |
| 搬迁后的真实 CPU 模型 | batch/phase、完整 logits、stream/stop 和四例 RERoT 通过 | RERoT 的合同、phase 门 PASS；重复与跨形状 greedy 资格仍是 NOT_QUALIFIED，不能因观察到相同轨迹而提升 |
| CPU / Vulkan reader | 两端各 33 个严格组与 12 个显式跨形状对照通过 | 72 次 decode / 169 行，等价与跨形状 max_abs：CPU 0，Vulkan `1.52587890625e-5`；原门限未改 |
| 独立 GPU attention oracle | 16/16 案例通过，实际 offload | 单层、零 FFN 的 Llama/Qwen2 fixture；reader max_abs `6.539480768563699e-7`；不代表全模型 F64 参考 |
| 完整 Vulkan RERoT | **8/8 episode 通过** | ubatch128/1 × release/retain × repeat2；合同、phase、同形状重复和显式跨形状 greedy 四项资格均 PASS |
| 真实采样 / corpus | CPU、Vulkan greedy 与正温对照均 24/24 token 相同 | 六请求 prefix/reference 实际 decode 44/120；CPU 总 NLL 差 `8.881784197001252e-16`，Vulkan 0，门仍为 1e-5 |
| 真实取消与输出失败 | CPU native cancellation 通过；CPU / Vulkan SIGINT 均 130，broken pipe 均 1 | 实际 live BODY 后注入；四项进程用例无 timeout、无强制清理；不扩写为 GPU kernel 内抢占 |
| HTTP 服务 | 三路 SSE 完成 48、48、8 token；实际 offload 25 层，KV=1536，两个 phase cycle | 第三请求插入时前两流均已收到至少两个块且未完成；非法 phase 参数在模型加载前拒绝 |
| 独立 HIP 安装及五卡平台 | HIP 3/3 CTest；搬迁安装实际 2000 轮，qualification 与 five_rank_status 均 pass | [原始平台报告](audit/devops-fivecard-probe-20261019.json) 与 [bench](audit/devops-fivecard-bench-20261019.json) 经最终安装 validator 验证；只取得相应平台合同 |

五卡各自的 host 系统 fetch_add 都实际得到 **448 而不是 7**，因此该能力观察保留 fail；
CAS 观察 pass。它们按原合同不参与正常单写者 load/store 资格，不得改写成“所有原子均可用”。
bench 与定标、其它验收工作存在并行执行，本轮不从这些时长作性能晋级或吞吐收益声明。

### 可复现入口与裁决

```sh
python3 fork/build.py --backend cpu --tools --build-dir build-devops-20261019/cpu --jobs 8
CMAKE_PREFIX_PATH=/home/kunweiz/deps/prefix python3 fork/build.py \
  --backend vulkan --tools --build-dir build-devops-20261019/vulkan --jobs 8
python3 fork/tests/isolation.py --output build-devops-recheck --jobs 8 \
  --model /absolute/path/LFM2.5-8B-A1B-UD-IQ4_NL.gguf \
  --rerot-model /absolute/path/LFM2.5-8B-A1B-UD-IQ4_NL.gguf
```

第二行的依赖前缀是本机实际路径，其他机器替换为自己声明的 SPIRV-Headers 安装。
HIP 配置、安装工具命令、进程与模型矩阵的完整参数由机器索引和每项回执保存。
输出目录必须新建；不要覆盖失败证据。当前 CPU/Vulkan 可交付组件与独立 HIP 探针的
DevOps 门通过；**不把这次验收称作整模型 TP5/mapped-relay 发布**。多设备模型 PP、
HIP/CUDA 模型路径、MTP、HTTP 取消恢复及后文历史全量 GPU 遗留门没有被本轮代验。

## 2026-10-19：RERoT hardening 与最终候选验收

本节记录本轮冻结源码对应的当前候选；后文早期 RERoT、TQRelay 与其他批次的 PASS、FAIL、
命令和数字继续属于各自原候选。当前机器索引为
[audit/rerot-hardening-qualification-20261019.json](audit/rerot-hardening-qualification-20261019.json)，
旧索引 [audit/rerot-qualification-20261019.json](audit/rerot-qualification-20261019.json)
保留原始身份。下列结果必须与新索引中的源码、安装库、消费者二进制、驱动、环境和原始输出
hash 一起使用，不能只凭相同目录名或测试名称提升另一构建的资格。

固定 A 仍为 `cda0f9256665845cd9d6f3a0176880883f653a3d`。
当前 [upstream-exceptions.json](upstream-exceptions.json) 为 **34 个路径**，SHA-256 为
`d52ab562be8ac3fcc331fdc731267fd18341539f980f72a6f80257855b3ae715`。
本轮实现提交为 A `81276fa6c`、B `23e1460af`、C `e60d999cb`；完整提交与产物身份由
新索引记录。GPU 数值和完整生成门使用新安装候选 `install-vulkan-expert`；
较早的 `install-vulkan-invariant` 及其负例继续保留。

模型仍为 LFM2.5-8B-A1B-UD-IQ4_NL，4,358,844,000 bytes，SHA-256
`a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a`。
本机为 AMD Ryzen 5 7500F 与单张 RX 6800（RADV NAVI21）。GPU 资格使用实际 Vulkan
offload 和模型 buffer 记录；枚举到设备、请求 `layer` 或打印 context 配置均不等于多设备执行。

### 数值策略、发布与失败合同

本轮继续使用一份 resident F32 K / F32 V 和 A 原生 recurrent owner。BODY 每 frontier
让每个运行中的插入点推进一个 token，读同一已提交 peer revision，整组同步完成后共同发布。
BODY 强制公开；private source-end 不进入其他 reader 的白板。多 token FRAME 仍在一个冻结
peer 世界内分 wave 执行并整组提交；中途失败不会发布半个 FRAME 或 cohort。

C 在创建 context、执行共同 base Prefill 之前设置
`batch_invariant_matmul=true`，并通过公开 getter 确认请求。**A 默认值仍为 false。**
`GGML_PREC_BATCH_INVARIANT=20` 为量化 dense `MUL_MAT` 和 expert-indexed
`MUL_MAT_ID` 请求原生 DEFAULT N1 的 dot 合同：保留相同后端的 activation 量化选择、
单输出点积及归约策略，其中可能使用 Q8 activations。它与 true-F32 的
`GGML_PREC_F32_PEDANTIC` 分开，不表示跨后端、跨模型或所有 provider 按位一致。

在已验收的 Vulkan 布局中，显式模式把 dense 查询列和 ID 的 token/selected-slot 对映射为
并行 GPU workgroups，只在硬件 grid 上限处分块；ID row padding、共享或逐 slot 输入和
融合 bias/scale 各有公开回归。它不展开逐 token 的 host 提交循环。未请求该模式时保留
原生大 N 量化矩阵路径；通用小 N MMVQ 修正单独保留 vendor/type/K 判断和 force/disable
优先级。getter 只报告 context 请求，实际 kernel 命中和性能分别验收。

图中标记的 F32 dense projection/router、F32 KQ/V 与 reader products 使用显式 PEDANTIC 合同；
CPU 排除相应 SGEMM 捷径，Vulkan 保留 F32 操作数并固定对应归约策略，包含仅 head 间有
padding 的输入。MoE 输出统一 expert 轴求和顺序，正常推理使用实际传入的 expert 数，
warmup 仍保留完整 expert 分配图。CPU mode20 的量化 dense 路径同时绕过两次 SGEMM，
固定 single-output dot，并保留转换与线程 chunk 调度；当前 x86 结果不扩写到其他 CPU
架构、extra-compute provider 或 MoE cache。

C 分开校验逻辑 reader 容量、物理 KV、逻辑 sequence ID 与 recurrent 物理槽位。
已接受的原生调用先同步排空，再释放 reader 和提交票据；发生过部分原生写入的失败会终止
episode。公开 unsupported-reason API 与 capability/setter 使用同一拒绝谓词，报告实际
不支持的模型、cache 或 context 条件。native sampler chain 继续拥有选择、accept 和 RNG；
compact shortlist 的使用不把 B 变成原生 sampler 的替代实现。

### 已完成的最终候选门

本表各行是不同分母，不应相加成一个“总测试数”。对应命令与产物 hash 由上面的新索引绑定。

| 检查 | 已确认结果 | 实际范围 |
| --- | --- | --- |
| 新 Vulkan 安装的 A public seams | **20/20 CTest PASS** | 测试集合包含 CPU 与 Vulkan 测项；不能把安装前缀名字解释为每一项都在 GPU 上运行 |
| GPU dense 量化跨形状 | **597 graphs / 90 bitwise comparisons / 18 bias graphs PASS** | native DEFAULT 单列独立参考；覆盖实际 fixture 的类型、batch、行重排和 head broadcasting；不以候选模式自身生成参考 |
| CPU / GPU expert-indexed 量化跨形状 | **每端 1,114 graphs / 431 bitwise comparisons PASS** | IQ3_S、IQ4_NL、Q8_0、Q4_K；共享/逐 slot 输入、padded ID rows、重复与变化的 expert IDs、正反序、ADD_ID 与逐 slot scale；含 65,540 个 flattened slots 的硬件 grid 边界 |
| 最终 GPU F32 reader | **33 个严格组 + 12 个跨形状对照 PASS** | 31 个全词表等价组、2 个 peer 因果效应组；`equivalence_max_abs` 和 `cross_shape_max_abs` 均为 `1.52587890625e-5`，原数值阈值未放宽 |
| 最终 CPU 独立 F64 attention oracle | **16 例 PASS，7 项 unsupported checks PASS** | reader 最大绝对误差 `6.75223405188774e-7`，ordinary 最大绝对误差 `3.7782220707605063e-7` |
| 最终 GPU 独立 F64 attention oracle | **16 例 PASS** | reader 最大绝对误差 `6.539480768563699e-7`，ordinary 最大绝对误差 `6.109807473286821e-7`；实际 offload。该次 GPU receipt 的 unsupported checks 为 0，不能写成 GPU 拒绝路径已跑 |
| GPU Prefill trace 控制 | **32 行 greedy mismatch 为 0** | 同一 token 输入的 ubatch128/1；保留 traced/untraced 控制和微小 F32 残差。greedy 一致不等于全部中间 tensor 按位一致，也不将残差的具体算子根因写成已独立闭合 |
| 原生 C tests | **5/5 CTest PASS** | 包含模型相关 native cancellation；该取消测试实际使用 CPU，覆盖真实 native 中断与已接受调用的清理，不能归因成 GPU kernel 内取消实测 |
| 最终 CPU 完整 C 严格矩阵 | **8/8 完整 episode PASS**；每例 **764 frontiers / 208 live peer reads** | 合同、phase、重复与跨形状四项 qualification 均 PASS；08:20:27 完成，driver 用时 `264.06855 s` |
| 最终 GPU 完整 C 严格矩阵 | **8/8 完整 episode PASS**；每例 **748 frontiers / 208 live peer reads** | 与 CPU 各自运行四 worker / 两 pen、`ubatch=128,1 × release,retain × repeat=2`，各自通过严格 greedy 门，并由原生 EOG 完成 worker 和 synthesis；不要求 CPU 与 GPU 的生成轨迹相同 |
| 实际 CPU / GPU SIGINT 与 broken pipe | **四项 PASS** | SIGINT 均 exit 130，pipe 均 exit 1；无 timeout / forced cleanup。对应时延单列于下文，不据此断言信号位于某个 GPU kernel 内或证明多设备 drain |
| 最终源码的纯 CPU 隔离 A / C | **A seams 11/11、C 5/5 PASS** | 来自新的 `isolation-expert/delivered`；C 的 native cancellation 实际在 CPU 执行 |
| 隔离安装的真实 CPU reader | **33 组 + 12 跨形状 PASS**；两个 max_abs 均 **0** | 同一新 delivered 安装的 72 次 decode / 169 行输入，单独绑定其安装产物身份 |
| 新源码隔离、搬迁与真实消费 | **08:10:50 完成，PASS；搬迁后 4 例真实 RERoT PASS** | A 从固定 upstream archive 加 34-path verified overlay 重新编译为纯 CPU；未复用 `--a-prefix`。删除 A/B/C 源码与 build 副本后，重定位安装消费者并实际运行 |

CPU / GPU 的 F64 attention oracle 都使用合法单层、零 FFN fixture，含 ordinary 校准与
reader logits，覆盖 fixture 中的 Llama NORMAL、Qwen2 NEOX、GQA、幅度及 query 顺序。
它们是独立数学参考，不能替代 8B 全模型的 F64 参考。以上误差来自最终候选，不复用旧 GPU
oracle 数值。

新隔离的四例 receipt 取得的是合同与 phase 资格。其 repeat 与 cross-shape 字段仍为
`NOT_QUALIFIED`，即使观察到相同轨迹也不提升这份 receipt 的声明；最终 CPU 严格八例
使用自己的完整 receipt 取得四项资格。source/build 隔离与安装搬迁证明包边界，不构成
操作系统沙箱或新的模型质量结论。

本次 isolation receipt 共 77 条命令、13 个隔离模块：B 为 sampling、corpus、scheduling、
text、calibration、relay、rerot；C 为 compose、corpus-compose、batch-compose、logits-compose、
imatrix-compose、rerot-compose。这里的 7 B / 6 C 只描述该 receipt；relay-probe 没有计入，
a-seams 是另列的安装消费者检查，不构成第八个 B 或第七个 C。

### 实际退出、kernel 命中与 Release 成本

进程测试在真实运行后注入信号或关闭 pipe。下表的 `seconds_after_injection` 以秒计并
四舍五入至六位小数；完整原始数值见新索引。四项都自行退出，无 timeout 或强制清理。
这些是进程生命周期证据，独立于 CTest 中实际在 CPU 执行的 native cancellation 测试。

| 后端 | SIGINT exit / seconds_after_injection | broken pipe exit / seconds_after_injection |
| --- | --- | --- |
| CPU | `130 / 0.160301` | `1 / 0.215547` |
| Vulkan | `130 / 0.099831` | `1 / 0.100161` |

DEBUG 另作实际 kernel 命中验收。MMVQ 的 auto / force / disable / both 四组 fixture 与
checker 均 PASS：auto/force 的 Q4_K / Q4_0 在 N1/2/8 实际命中 MMVQ，IQ4_NL 使用原生 F32 fallback，
both 时 disable 优先。ID 整 fixture 的 1,114 图也通过；原始 DEBUG 记录严格解析出
**8 对相邻分派**，每对均为 `mul_mat_vec_id_q8_0_f32`，workgroups 先为
`(3,65535,1)`、后为 `(3,5,1)`。这直接覆盖 65,540 个 selected slots 超过设备 Y 上限、
且 chunk 落在 token 内部的边界。DEBUG 日志只证明执行路径，不用于下面的性能数值。

正常 Release 性能另运行六个进程，DEBUG 关闭，测量期间没有本任务的其他重构或实模型
负载，没有锁频。原始 samples、cold/warm、IQR、库与命令身份在
`performance-vulkan-expert`，由新索引绑定。性能执行成功并不预设优化普遍获益。

dense 对照为 IQ4_NL、Q8_0、Q4_K，K=2048、M=1025、N=9/32/128，输入为设备驻留的
`SCALE → MUL_MAT` graph；计时包含 compute 与 synchronize。两次运行采用相反首模式，
每次 9 个 case / 18 个 mode / 288 次 graph compute，每 mode 连续执行 1 cold + 15 warm。
下表是两次运行内各类型的 **invariant/default warm median 比值范围**；比值大于 1 表示
显式批形状不变策略用时更多。

| N | invariant/default median 范围 | 对这些 graph 的观察 |
| --- | --- | --- |
| 9 | `0.469399–0.499107` | 显式模式用时较低 |
| 32 | `1.092569–1.311400` | 为相同原生单列 dot 合同付出额外用时 |
| 128 | `1.857106–2.019109` | 显式模式约为 DEFAULT 用时的 1.86–2.02 倍 |

小 N MMVQ 对照按 auto / disable 的 ABBA 顺序运行四个正常 Release 进程。
Q4_K N8 的 auto 为 `63.05–63.54 µs`，disable 为 `77.14–77.38 µs`；
Q4_0 N8 两者均约 `77 µs`。一些小 N 与 IQ4_NL 负控制存在明显的主机或时钟波动，
因此不作普遍提速结论。这里测的是 resident graph 的局部计算成本，不是完整模型吞吐、
end-to-end RERoT 用时或多设备收益，也不以 DEBUG 日志中的时长替代 Release 测量。

### 旧失败、当前限制与复现

原 F32 CPU/Vulkan 跨形状失败、GPU tiny ordinary 校准失败、F32 head-padding 反例、
早期 CPU observer 预算失败、Vulkan shape-policy 完整生成失败及 dense-only Prefill 的
expert-ID 差异都保留原候选与原回执。本轮通过来自新的数值路径、实际公共算子对照和严格完整
生成门，没有放宽 numeric/greedy 阈值、伪造 EOG 或把失败分组改称通过。

当前 reader 仍限定支持列表内的 resident F32 K/V。F16/mixed、Turbo/量化 KV、FA reader、
SWA、partial RoPE 与 GPTNeoX 模型架构等不因完整 Qwen2 NEOX fixture 通过而取得新资格。
多设备 PP、整模型 TP5、LateBind、MTP、自动 planner、reasoning-end/tool、HTTP 集成、
CUDA/HIP RERoT 及多模型质量继续保留各自的实现和硬件门。单卡 `layer` 成功不等于
多设备 PP；既有 Vulkan `row` 的原生 split-buffer 拒绝也保持原身份。当前结果不是整个
Vulkan 算子库重验，后文全量 FA/TOPK 的历史失败不因这组专项 PASS 被抹去。

复现入口及当前参数合同见
[rerot-compose/QUALIFICATION.md](rerot-compose/QUALIFICATION.md)。
每个候选使用新的输出目录，保存原始 stdout/stderr、执行命令、完整模型 shard hash、
消费者和安装库 hash；启用严格跨形状门。需要模型的测试须真实提供模型，不能把 skipped
记成通过。重新安装公开 context 参数发生变化的 A 后，应以相匹配的头文件和库重建消费者。

## 历史批次的记录身份

以下各节按原文保留。其中“本批次”“最终”“当前”、28-path manifest、旧数值和未闭合门
均指相应历史候选。当前 hardening 的资格以上一节及新机器索引为准；新 PASS 不改写旧 FAIL，
旧 PASS 也不能替代当前候选的独立记录。

## 2026-10-19：RERoT 逐 token 共享白板

本批次起点为 `4ef817eecff5bb512a65d91bacd2d6b028b40ed5`，固定上游仍为
`cda0f9256665845cd9d6f3a0176880883f653a3d`。本节记录新的 RERoT 实现，
后文较早批次的数字、失败与资格保持原身份。机器索引见
[audit/rerot-qualification-20261019.json](audit/rerot-qualification-20261019.json)。

**交付范围：独立 TQRerot B、使用安装包的原生 C，以及 resident F32 K / F32 V 的 A reader 接口。**
真实模型为 LFM2.5-8B-A1B-UD-IQ4_NL，模型文件 4,358,844,000 bytes，
SHA-256 `a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a`。
实际设备为 AMD Ryzen 5 7500F CPU 与一张 AMD Radeon RX 6800（RADV NAVI21）；
Vulkan C 的模型日志记录实际 offload 25 层及 Vulkan0 模型 buffer。设备枚举本身不计为运行。

### 白板语义及实现边界

每个 BODY frontier 让每个正在运行的插入点推进一个 token。所有 reader 持有上一轮已提交
世界的不可变视图，当前自己的 token 参加因果计算，其他 reader 当前尚未提交的 token 不可见。
整组原生调用完成并同步后共同发布；下一 frontier 立即看到其他插入点刚提交的 token。
已采样、尚未 decode 的 token 不会提前出现在白板。实际多 token FRAME 按每序列每调用一行的
wave 执行，所有 wave 使用同一个冻结 peer 世界；FRAME 完成后才共同发布。

B 拥有 DAG、reader 相对顺序、public/private 行、token publication 和事务身份，不依赖 A、
GGML 或其他 B。C 通过安装后的 llama、TQRerot、TQRelay、TQText 公共接口运行；每次真实
A 提交使用局部 `tq::relay::run_submissions(call, 1)`，票据覆盖接受、同步、完成登记和
reader 清理，随后释放，不随 token 数累积历史记录。这里的 1 是一个不透明的 A 提交域，
不伪装成五个已完成的 GPU rank。

A 保存一份实际 resident KV。identity span 复用原生 batched KQ，真正改变相位的 span
从 pre-RoPE Q 与原始 RoPE 参数重建 reader-relative Q，再与 writer-phase K 计算分数；
分数放回原物理 KV 槽，隐藏槽为负无穷，
再做一次全局 softmax 和原生 batched V。reader 视图深拷贝、无效 setter 保留旧视图、
图不跨 reader frontier 复用。recurrent 的逻辑序列身份与物理状态池分开；sealed worker
回收 recurrent 状态而保留其他 reader 仍需读取的 attention KV。

### 已执行的严格门

| 检查 | 最终实际结果 | 范围 |
|---|---|---|
| 独立 B | Release 警告作为错误、独立 Debug ASan + UBSan + 标准库断言均 4/4 通过 | 543 个四节点 DAG、3,007 个合法状态、3,904 个 reader；3,200 个独立 F64 reader/head 比较最大误差 `7.77156e-16`，错误相位反例误差 `1.70309` |
| 实际 A 对独立 F64 | 16 个 CPU 案例通过，reader 最大绝对 logits 差 `7.238989769842163e-7`，ordinary 校准最大差 `4.1964194408272704e-7` | 单层、FFN 权重置零、F32 权重 / KV 的 GGUF fixture，通过实际 A 图执行；Llama NORMAL / Qwen2 NEOX、4 Q / 2 KV heads、幅度 1 / 1.375、正反序、batch / serial；原门 `3e-5 + 1e-5 * abs(reference)`。不是 8B 全模型 F64 对照；Vulkan 安装包中的默认 oracle 也实际在 CPU 执行 |
| 真实 F32 reader | CPU / Vulkan 各 33 个严格门通过：31 个全词表等价组与 2 个 peer 因果效应组 | 每组 128,000 logits；等价组最大差 CPU 0、Vulkan `0.0000133514404296875`；包含新增 reversed-serial 对 visible-serial 同形状对照 |
| reader 生命周期 | 两端各 72 次真实 decode、169 行输入、6 个 invalid view 拒绝、1 个满 recurrent 池拒绝及随后复用 | 包含隐藏 peer、ordinary identity、未提交 / 当前 peer 隔离、下一轮生效、行重排、retire / replacement、restore 前后与独立 control；原门 `2e-4 + 1e-5 * max(abs(a), abs(b))` 未改 |
| 已有 ordinary recurrent 回归 | 两端各 24 个邻居快照、8 个共享快照检查通过；续算最大差均 0 | 各 24 次 decode、189 行输入；验证默认 recurrent 配置，不使用新 reader 模式 |
| 原生公开接缝 | CPU 4/4、Vulkan 8/8 CTest 通过 | 含本次 CPU oracle 及既有 compute-memory、FA precision、meta transfer、Vulkan 专项；不是全量 GPU 算子重跑 |
| C CLI | 两端各 28 个输入、选项及输出合同检查通过 | 一项 CTest 驱动这些检查；不把 CLI 测试计为模型执行 |
| 实际 C 完整生成 | CPU 8/8、Vulkan 8/8 通过 | 每端 `ubatch=128,1 × release,retain × repeat=2`；两个并发 writer、依赖阶段和 synthesis 都以原生 EOG 完成，无预算超限强行补终止符 |
| 相互逐 token 可见 | 每次运行均观察到两个活跃 worker 之间连续增长 | CPU 每方向 103 次增长、206 次 live peer 读取；Vulkan 每方向 103 或 104 次、206 或 208 次读取。事件合同不代替上述 A 数学与因果 logits 检查 |
| 阶段内存与重复性 | 每端固定 microbatch 下，release / retain 的完整 greedy token 轨迹相同；两次重复轨迹相同 | 每次 release 运行出现 3 次实际 compute release，after 均为 0。ubatch 128 的首次 CPU / Vulkan before 分别为 75,561,120 / 89,946,060 bytes；这是公开接口的计算 buffer 字节，不是峰值 VRAM 或 KV 大小 |
| 实际 SIGINT / broken pipe | CPU / Vulkan 各两例通过 | 观察到两轮真实双 worker BODY 后注入；SIGINT 退出 130 并输出 incomplete cancellation error；关闭唯一 stdout 读端后退出 1，走 output-write-failed 路径；无超时或强杀兜底 |
| 完整来源与搬迁隔离 | `build-isolation-rerot-20261019` 的 77/77 条命令通过 | A 从固定 pin archive 加 verified overlay 重建；七 B、六 C 独立构建安装，移除 source / build 副本和原前缀后搬迁，B 公共消费者、C CLI 与四例真实 RERoT 生成通过 |
| 原生 split 选项 | RX 6800 单卡 `none` 与 `layer` 实际生成通过 | 额外 layer 用例为 512 frontiers、208 次 live peer 读取；不是多设备 PP 资格 |

阶段内存对照只比较相同调用形状。完整 C 的默认每阶段预算是 256；验收驱动显式使用
worker / synthesis 各 512，将该模型实际生成的 reasoning token 也计入预算。
测试没有通过伪造 EOG、替换为已完成答案或跳过未完成阶段取得通过。

取消检查覆盖提交前、已同步 chunk / wave 后、commit 前及最终完成前；已接受调用先由原生
同步收束再释放 owned reader 状态。外部进程测试的退出码和事件不证明信号恰好落在某个
GPU kernel 内，也不独立证明每个设备的 drain。原生同步合同与 C 代码审查、进程测试分别保留。

### 仍未取得资格的范围

| 项目 | 实际状态与处理 |
|---|---|
| F16 / mixed KV reader | capability 与 setter 明确拒绝，CPU / Vulkan 的负向门又实际验证普通 F16 prefill 和续算仍可工作。较早 F16 CPU permutation / restore 最大差 `0.1179952621459961`、Vulkan permutation `0.00035762786865234375` 未过门，保留原 FAIL 回执；精度策略与物理位置重排 / restore 后重分配列入 E23 重写候选 |
| 跨 batch / serial 数值等价 | 最终两端各 12 个原阈值对照全部观察为 FAIL，最大差 CPU `2.1814355850219727`、Vulkan `4.442938804626465`；完整 C 跨 ubatch 的 greedy 轨迹也不同。默认显式 `NOT_QUALIFIED`，`--require-cross-shape-equivalence` 可恢复严格退出门；同形状门不受此分组影响 |
| GPU 独立 F64 oracle | 实际 offload 的 tiny oracle 在 ordinary attention 校准即失败；显式 `GGML_VK_DISABLE_F16=1` 后仍失败。原有两次 exit 1 和日志保留，不用 CPU 16 例或 Vulkan 关系门代替 GPU 独立数学资格 |
| Vulkan row split | 真实加载失败：`device Vulkan0 does not support split buffers`。保持失败及底层原因，不做隐式模式回退 |
| 多设备 PP、整模型 TP5 | 当前 C 复用已有 TQRelay 提交寿命协议，并把原生设备放置交给 A；没有已安装的整模型 TP5 provider，也没有多 GPU 实测。`--tp5` 明确拒绝，不能把单卡 ledger 或 layer 模式运行扩写成五卡模型资格 |
| MMVQ | QKV / FFN 保留 A 的量化权重与多行调用分派；本次不改通用 GPU kernel。没有针对该组合的实际 kernel 命中或性能收益资格，microbatch 改变还会改变分派形状 |
| LateBind / MTP | 当前安装接口缺少完整 provider，C 明确拒绝 `--latebind` / `--mtp`。共享世界每轮变化后，旧 revision 的 draft / logits 不再能直接提交；后续需真实刷新、验证、作废与 drain 协议，相关设计只保留文本候选 |
| 其他范围 | Turbo / 量化 KV reader、streamed KV、SWA、partial rotary / GPTNeoX A 路径、自动 planner、reasoning-end / tool 路由、HTTP 集成、CUDA / HIP RERoT、多模型质量与吞吐收益均未由本批次取得资格 |

A 的当前例外 manifest 共 28 路径，其 SHA-256 为
`b980037891817e8fc82b1c31bd92af33805c02c24ca185089d4fcf2f21717a65`。
本次相对起点修改 14 个 A 源码路径，合计 +642 / -52；没有新的通用 GPU kernel 文件修改。
每项 hash、消费者、理由与移除条件随 manifest 维护。下面较早的全量 Vulkan FA 失败、
TOPK 未闭合调查及未运行平台状态不因本次专项通过而改变。

### 回查与复现

最终 native 构建、reader、F16 拒绝、ordinary recurrent 与 CTest 的命令和日志位于
`build-rerot-runtime-20261019/`。完整生成的两份 receipt 位于
`build-rerot-whiteboard-{cpu,vulkan}-f32-20261019/receipt.json`，中断记录位于
`build-rerot-interrupt-{cpu,vulkan}-f32-20261019/receipt.json`；其模型、二进制、附加库和
原始输出 hash 随回执保存。单卡 layer / row 记录与旧失败回执也在机器索引中分别标识。
搬迁隔离的 `result.json`、`commands.json`、`verified-boundary.json` 与实际模型 receipt
位于 `build-isolation-rerot-20261019/`；本次没有运行可选的旧 batch / logits / stream 模型矩阵，
它们在该回执中明确为 skipped。

下列命令是复现入口，输出目录必须是新的目录。真实模型需由运行者提供，不自动下载：

```sh
python3 fork/build.py --backend vulkan --build-dir build-rerot --jobs 8

python3 fork/rerot-compose/tests/real-whiteboard.py \
  --binary build-rerot/install/bin/tq-rerot-compose \
  --model /path/to/compatible-model.gguf --gpu-layers 99 \
  --ubatch-sizes 128,1 --phase-modes release,retain --repeat 2 \
  --output build-rerot-whiteboard-proof

python3 fork/rerot-compose/tests/real-interrupt.py \
  --binary build-rerot/install/bin/tq-rerot-compose \
  --model /path/to/compatible-model.gguf --gpu-layers 99 \
  --output build-rerot-interrupt-proof

python3 fork/tests/isolation.py --output build-rerot-isolated --jobs 8 \
  --rerot-model /path/to/compatible-model.gguf
```



2026-10-18 批次起点为 `6de088714f35f84427d9952d713f9d6cdffc668b`；固定 A 为 `cda0f9256665845cd9d6f3a0176880883f653a3d`。该批次新增 TQText、TQCalibration 和 imatrix C，扩展真实 batch C 的文本消费，并对 Vulkan 缓存的两个不同失效来源分别建立回归。六个 B 与五个 C 的职责见 [README.md](README.md)，旧 feature ID 与证据细分见 [audit/details-findings-20261018.md](audit/details-findings-20261018.md)。

本页先列 2026-10-19 TQRelay 批次已取得的结果，随后保留 2026-10-18 批次的 Final release gates 与更早历史；其中原模型摘要、HTTP、corpus、logits 与数值记录保持各自批次身份，没有被改写成本批次结果。

## 2026-10-19 批次：TQRelay（TP5 纯 CPU 合同）

本批次新增第六个 B 库 `fork/relay`（安装目标 `TQRelay::relay`），把《TP5-最新设计
保姆级指南》第 30–39、50、52、54 章中与设备无关的合同实现为 C++17：`channel_ledger`
（通道 bank 生命周期、精确票据、两种 credit、fail-stop 与排空事实）、
`run_submissions`（逐 rank 原生提交登记、封口与排空推导）、`compile_glu_plan`
（五 rank 合成 GLU 分片计划）。没有 A 文件变化，`fork/upstream-exceptions.json` 不变。
指南第 50、54 章内嵌的四个 Python 程序与四个 JSON 输入按记录 SHA-256 原样提取到
`fork/relay/tests/python-reference/`，全部匹配。

| 检查 | 实际结果 | 可以支持的结论 |
|---|---|---|
| Release 构建与测试 | `build-relay-20261018`，GCC 16.2.1，`-Wall -Wextra -Wpedantic -Werror`，4/4 ctest 通过 | ledger 所有权反例、提交登记边界、局部计划数字与 dense GLU 独立参考按代码路径通过 |
| ASan + UBSan | `build-relay-san-20261018`，4/4 ctest 通过 | 本库已执行用例无地址/未定义行为报告；不扩写到线程、GPU 或模型 |
| 完整构建流水线 | `build-clean-fork-relay-20261019`：A 固定 archive + 六个 B + 五个 C + a-seams，exit 0，末道边界复查通过 | 新库进入 `fork/build.py` 常规入口后整条流水线成立 |
| 完整来源隔离 | `build-isolation-relay-20261019`：63/63 条命令通过，`isolated_modules` 含 `relay`，relay 从自身源码副本构建、测试（含 Python 交叉对照）、安装并进入搬迁后交付集合 | 独立库边界与安装消费成立；不构成操作系统沙箱 |
| M0c 单卡平台探针（16 用例扩展版） | `fork/relay-probe`（HIP 构建，`build-relay-probe-20261019`）在本机 RX 6800（gfx1030，ROCm 7.2.4）实际运行 2000 轮：双向热复用、生产延迟排序、精确票据、多工作组最后到达、abort、五消费者输出 credit、延迟最后读者（H08）、双 bank 轮换、可变有效长度（H05）、单卡五流 RELAY 环全部 pass；host fetch_add 产生错误值（448 而非 7）记 fail，host CAS pass；五卡用例 not_run（本机 1 张）。证据报告 SHA-256 0cdd86bd…9cb8a | 单卡平台事实成立：单写者 load/store 协议可用且无需 host RMW；五消费者 credit 与双 bank 是软件协议层验证，不构成五卡或整链资格，换工具链/拓扑后需重跑 |
| 探针 bench 测量（`--bench`） | 独立 `tp5.relay.bench.v1` 文件（SHA-256 062c2806…a9de0ad），2000 轮 median：host 发布→设备 ack 9,690 ns、设备生产→host ready 7,770 ns、五流整圈 29,930 ns、host 五路折减 65.7 GB/s | 单设备含 launch 时间的观察值，供指南 14 成本模型使用；不是资格门，五卡数字待真机 |
| 设备等待预算修正 | 全量运行实测发现旧代码每次 ack 未命中把 2e9 ticks 预算一次性睡满（每轮 ~0.9 s，2000 轮预计数小时，600 s 上限内无产出）；改为累计预算等待（clock64 实测 ~2.3 GHz，预算 24e9 ticks ≈ 8 s，超过最长 5 s host 截止期），失败路径 abort+排空防晚写污染 | 全量 2000 轮从数小时级降到约 20 秒；防抖语义保留（设备等待超过 host 截止期，瞬时负载尖峰不再造成设备侧假超时） |
| 五卡跨设备用例实现加固（无新硬件证据） | 提交 35d41c8fc 之后的加固：`run_five_rank` 从单轮裸写升级为 guard 三明治 + 按 `--rounds` 多轮热复用（每轮 payload 随代次变化）+ 逐 rank/轮次失败归因 + 每轮同步后复位；`five_rank_cross_device` 现在登记进每张设备的证据记录（注册项可达）；`free_device_context` 与 `run_bench` 补上入口 `hipSetDevice`（五卡机上 `hipFree`/launch 作用于当前设备，缺失会解错设备）；`--rounds` 拒绝尾随垃圾与溢出。单卡 2000 轮 smoke 复跑 qualification=pass、five_rank=not_run、16 用例分布不变。机器可读索引见 [audit/relay-probe-evidence-20261019.json](audit/relay-probe-evidence-20261019.json) | 单设备路径无回归；跨设备路径第一次执行只能在真实五卡机器上发生，本行不含跨设备可见性证据。报告 SHA-256 0cdd86bd…9cb8a 是加固前的产物，其 five_rank_status=not_run 依然准确，不改写旧报告字节 |
| 探针交接批次：两个新用例 + 证据路径抗故障化 + 原子发布 | 注册表 16→18：新增 `offset_addressing`（指南 39.5 H06，字节 72、8 字节对齐但跨 cache line 的内部槽双向往返，arena 前后 guard 抓基址/越界算术）与 `multi_group_ordering`（指南 39.6 P01/P02，确定性反序延迟使最大编号组先完成，陷阱观察 ready 保持未发布，8 轮全部观察到陷阱窗口）。HIP 错误改为异常：逐用例捕获记 fail、坏设备 setup 失败仍以 `device_identity=fail` 留在报告、`run_five_rank` 任何 HIP 失败按 rank 归因且照常写报告、bench 失败不再丢失报告。证据文件改同目录临时文件 + fsync + 硬链接原子发布，已有文件/符号链接拒绝覆盖；CLI 拒绝 `--output` 与 `--bench` 同路径。单卡 2000 轮权威重跑 qualification=pass、17 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 3ff3a3b6…f03a6、bench SHA-256 e2a66eec…33de0；无 HIP 构建 config-test 1/1、relay release/san 4/4、check-upstream 通过、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-handoff.json](audit/relay-probe-evidence-20261019-handoff.json) | 单卡新增用例通过；故障路径由逐项代码评审与 CLI 拒绝验证保证，五卡机首次运行前不声称已实测。16 用例报告 0cdd86bd…9cb8a 保留为被取代历史，字节不改写 |
| 探针证据校验器与归因加固批次 | 新增 `fork/relay-probe/validate-report.py`：报告与 bench 的严格 JSON 解析（重复键、非有限数、未知字段、尾随数据、bool 冒充整数全拒绝）+ 指南 53.4 终态一致性机器检查（注册表精确覆盖、rank 与下标一致、five_rank 状态与设备数一致、qualification 与逐用例结果重算一致、无设备报告的合法形状、bench min≤median≤p90≤max 且测量名唯一），`--self-test` 覆盖 20 余个反例，两份权威证据实测 exit 0。五卡归因记录塑形抽成纯函数 `five_rank_case_records`（rank 0–4 记录、failed_rank==5 哨兵、失败详情只归因到该 rank），config-test 覆盖成功、rank 2 归因、哨兵三种形状——该路径真机上首次执行，先在此闭合其逻辑。stdout 改无缓冲（_IONBF，C 标准"尽快传输"保证）：config-test 新增 fork+SIGKILL 存活测试实测"杀进程不丢已打印证据行"，兑现 README 的可恢复承诺。`fork/build.py` 默认流水线接入探针模块（install=False：无 HIP 机器只构建并运行 config-test，有 ROCm 时顺带构建探针），AGENTS.md 例外条款相应修订；isolation.py 继续排除（无 HIP 时没有可安装 target）。校验器 `--self-test` 挂进 ctest（tq_relay_probe.validator），覆盖五卡 fail 报告形状及其不一致反例，流水线与真机 ctest 都先证明校验器。最终二进制 2000 轮权威重跑 qualification=pass、17 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 75dbd439…dec4c、bench SHA-256 29efb0c1…e80d，校验器对两份新证据 exit 0；relay release/san 4/4、无 HIP 构建 ctest 2/2、HIP 构建 ctest 2/2、check-upstream exit 0、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-validator.json](audit/relay-probe-evidence-20261019-validator.json) | 单卡平台事实与前批一致；五卡路径新增逻辑只经 CPU 测试闭合，跨设备可见性资格仍待真机首次运行。3ff3a3b6…f03a6 报告保留为被取代历史，字节不改写 |
| 探针有界同步批次：挂死设备不再让运行永不返回 | 全部 17 处裸 `hipDeviceSynchronize`/`hipStreamSynchronize`（逐用例检查、run_case/fail_case 排空、五卡逐 rank 排空与 teardown、bench 排空）替换为 `sync_with_deadline`：记录 event 后 host 截止期轮询，健康队列微秒级返回，停转队列 60 s 后变成该用例的 fail 并继续收集证据（指南 35.8；SPEC 7 的"只有设备枚举失败才可能无报告"由此补上"挂死也能出报告"）。超时分支不再是交班后首次执行的死代码：新增强制用例 `native_sync_deadline`（注册表 18→19）用 stall kernel 把队列停转超过 2 s 短截止期，实测触发有界失败、停止字恢复等待者、drain 完成，本机 pass。注册表 validator/config-test 同步；报告 limits 记入 60 s/2 s 两个截止期。最终二进制 2000 轮权威重跑 qualification=pass、18 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 c0a21839…2e1f、bench SHA-256 04473ee1…c375b，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 除已登记 HIP 误报外无告警。机器可读索引见 [audit/relay-probe-evidence-20261019-syncdeadline.json](audit/relay-probe-evidence-20261019-syncdeadline.json) | 有界同步语义在本机真实设备上验证（超时分支实际执行）；"挂死设备有界失败"是软件行为+event 查询活性的组合事实，五卡机上的最终确认仍待真机运行。75dbd439…dec4c 报告保留为被取代历史，字节不改写 |
| 探针 P 系列批次：同流逆序识别与双窗口独立前进（单卡可测子集） | 指南 39.6 的完成与前进必做用例在本机补上软件层覆盖：新增强制用例 `same_stream_inverse_dependency`（P04，消费者排在 payload 生产者之前的同流逆序，流序保证早读写入前载荷，验证字 0xDEADBEEF 确定性抓住早读而非静默消费垃圾，生产者后写精确，运行有界）与 `dual_window_independent_progress`（指南 31.1 与 P05/P06/P08 软件层：两个独立通道各自控制块/槽/种子，窗口 A 被扣留票据、64 个驻留等待工作组、延迟最后读者三种形态停摆时窗口 B 独立完成 528 代消费，逐代金丝雀证明 B 从未写 A 槽、A 的 ack 半读期始终扣留，释放后精确完成并复用）。device_context 增第二通道（cb2/slot_b），bare_waiter_kernel 用累计预算占驻留。注册表 19→21，validator/config-test 同步；单卡 2000 轮权威重跑 qualification=pass、20 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 b6baeb81…f5be、bench SHA-256 d51f2e08…fa16e，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-pcases.json](audit/relay-probe-evidence-20261019-pcases.json) | 两个新用例证明单设备上软件协议层的逆序识别与跨通道隔离；它们是五卡机上双窗口设备时间线的软件前提，跨设备资格仍由 five_rank_cross_device 给出。c0a21839…2e1f 报告保留为被取代历史，字节不改写 |
| 探针服务循环批次：P07/P09/P10、F04、F07（单卡可测的最后一格） | 指南 39.6 完成与前进表在本机闭合最后三个软件层格子：新增强制用例 `service_loop_independent_progress`（P07/P09/P10：独立 worker 线程驱动 B 通道 256 代而观察者只读状态；两个同时齐备的通道背靠背发布，确定性慢读者半读期内快通道完成其验证读，双向都验证；关停时带驻留 GPU 等待者，stop 字与两条流排空全在 worker 退出前完成，观察者只 join）、`nonfinite_reduction_gate`（F04/指南 41.6：host 五路折减新增发布前门——rank 权重模 2^64 和为 1，精确恒等式确定性抓住任何损坏的部分和，翻转位与 quiet-NaN 位型都在发布前被拒绝且输出槽零写入；five_stream 用例与 bench 共用同一门）与 `host_wait_deadline_no_reuse`（F07/指南 35.8：读完却永不确认的消费者触发 host 有界等待 400 ms 截止期——spin_until 超时返回分支从交班后死代码变成活设备实测路径；槽位字节不变，停止字恢复等待者）。注册表 21→24，validator/config-test 同步；单卡 2000 轮权威重跑 qualification=pass、23 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 01533600…fc73、bench SHA-256 05c3f207…f75e，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 除已登记 HIP 误报外无告警。机器可读索引见 [audit/relay-probe-evidence-20261019-serviceloop.json](audit/relay-probe-evidence-20261019-serviceloop.json) | 三个新用例证明单设备上服务循环进度独立、无队头串行与关停排空顺序，以及归约发布门与 host 有界等待；它们是真机双窗口调度与 CPU 服务循环的软件前提。剩余 F 系（F01/F03/F06/F08/F09/F11/F12）在健康单卡上不可注入，软件路径经代码评审、不上设备。b6baeb81…f5be 报告保留为被取代历史，字节不改写 |
| 探针定标与折减门批次：时钟标定入库 + 折减门下沉 + bench 多样本 | 三个收口：① `fold_partials` 与 rank 权重从 main.cpp 下沉到证据层（evidence.h/evidence.cpp），无 HIP 的 config-test 新增 `check_fold_gate` 钉住全部反例。审查中先抓出真缺陷再修正：权重原为 {1, 2^32, 2^32, 2^32, −3·2^32}——rank 1–3 同权，行复制对同权 rank 对不可见；改为一组两两互异且和为 1 mod 2^64 的奇数权重 {1, 3, 5, 7, 2^64−15}：奇乘是 64 位双射，任意位损坏（含幂权重会移出字的高位）必然改变折减，行复制对任意 rank 对可见；门改全有或全无（先验证全部词再写 out，拒绝不留部分结果）。活设备用例 `nonfinite_reduction_gate` 同步改钉高位翻转 + quiet-NaN 注入。② 定标方法入库：新增 HIP 门控诊断工具 `tq-clock-rate`（每设备 5 次固定 tick 自旋 + event 轮询墙钟截止期），本机三次独立运行 median 2320.1–2320.4 MHz，复核 2.3 GHz 预算锚；main.cpp 注释与运维 README 交叉引用。③ bench 折减带宽从两遍取优单样本改为 32 样本 min/median/p90/max（单样本被调度噪声支配，min 与 median 实测相差约 1.6 倍）。单卡 2000 轮权威重跑（全新树）qualification=pass、23 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 b40dd940…6b26、bench SHA-256 a1505a8c…7b2c，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-calib.json](audit/relay-probe-evidence-20261019-calib.json)
| 探针五卡审计批次：not_run 路径抓出真缺陷 + 五卡核索引单卡钉住 | 对唯一未在活设备跑过的五卡路径逐行审计，抓到三个缺陷并全部修复：① **真缺陷（活设备复现）**：`rank_write_kernel` 向基址发布 ready——五个 rank 全员踩写 `ready[0]`，rank 1–4 的 ready 永不到达，五卡机第一轮就会等满 5 s 并以虚假 stall 归因失败（/tmp 复现驱动实测 ready=[1,0,0,0,0]）；修复为 `release_store(&ready[rank], generation)`。② `run_five_rank` 的 setup 失败预检查是死条件：它检查 `record.ordinal < 0`，而 run_probe 的 catch 总会先把 ordinal 写正——五卡机上坏卡不会被识别，会向坏 context 发核再以误导性超时归因；改为按 `device_identity` 结果与 `mapped` 能力位判定，坏卡在核启动前被精确归因（bench 同补 mapped 守卫）。③ 零设备分支退出码为 0（与枚举失败分支的 1 不一致，qualification=fail 却退出 0 会骗过只看退出码的脚本）；改为与枚举失败一致的 1（本机 ROCm 枚举失败走报错分支，零设备分支为其他运行时的防御性一致化）。**新增单卡钉住用例 `five_rank_indexing`（注册表 24→25，validator/config-test 同步）**：五条流在一张设备上跑与五卡用例完全相同的核与 guard 布局，逐轮验证每个 rank 的 payload 与自己的 ready 字；两个用例共享同一套轮次协议 `run_rank_rounds`（抽取自 run_five_rank，五卡机上跑的循环正是单卡钉住的那份代码）。回注缺陷版的临时工作树实测：`five_rank_indexing=fail`，detail 精确归因 rank 1 5 s 超时，退出 1、运行有界——用例对缺陷的判别力在活设备上被证实。运维语义实测：HIP_VISIBLE_DEVICES 的空白/空值形态会让枚举报错（报告照写、退出 1），`0,1,2,3,4` 与尾随逗号可用，已写入运维 README。单卡 2000 轮权威重跑（全新树 build-relay-probe-20261019g）qualification=pass、24 用例/设备（仅 rmw_add 预期 fail）、five_rank=not_run、five_rank_indexing=pass、新报告 SHA-256 65146f98…b4533、bench SHA-256 70400732…03c3，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 除已登记误报外零新增（本班新增 rank_stream padding 与空指针推断两处即时修掉，既有 C++ 风格结论补登记进 AGENTS.md 例外清单）。机器可读索引见 [audit/relay-probe-evidence-20261019-fivecard.json](audit/relay-probe-evidence-20261019-fivecard.json) |
| 探针交付加固批次：CLI 严格化 + 定标工具映射内存合同 + 安装完整 + 校验器抗畸形 | 四个收口：① **文档-实现不一致（实测抓出）**：README 承诺 `--rounds` 只收纯十进制整数，实际 `strtoul` 静默放行 `02000`、`+2000`、`" 2000"`；解析器现在要求首字符为 1–9，六种形状（0、02000、+2000、前导空白、-1、溢出）全部退出 2，只有纯数字放行（本机逐项实测）。② **定标工具依赖平台意外（实测确认）**：`tq-clock-rate` 用 `hipHostMallocDefault` 却由核写入该字——本机实测该分配返回 flags=0（不含 Mapped）时核写入照样落地，属契约外行为，不可依赖（AXIOM 1）；改为映射分配 + 逐卡校验 Mapped 标志 + acquire 读取，修复后速率不变（2257.1–2320.4 MHz），另补 `--help` 退出 0。③ 安装包完整：`cmake --install` 原装出两个二进制与文档却漏掉运维签收必需的 `validate-report.py`；现一并安装并做过独立前缀冒烟（装出二进制 50 轮 qualification=pass，装出校验器对装出二进制写出的报告 exit 0）。④ 校验器抗畸形树：`--self-test` 新增固定种子结构化变异模糊（报告与 bench 各 500 次逐层变异，任何结果必须落在"仍合法"或 EvidenceError 内）；本次交付前先用 6000 次变异跑过一遍，零意外异常后入库。config-test 补钉 `write_file_exclusive` 的符号链接拒绝（链接与被指文件原样保留）与目标目录不存在的干净拒绝。权威证据以最终源码在全新树重跑（注册表仍 25，新报告/bench SHA 见索引）。机器可读索引见 [audit/relay-probe-evidence-20261019-delivery.json](audit/relay-probe-evidence-20261019-delivery.json) |
| 探针交付加固批次：CLI 严格化 + 定标工具映射内存合同 + 安装完整 + 校验器抗畸形 | 四个收口：① **文档-实现不一致（实测抓出）**：README 承诺 `--rounds` 只收纯十进制整数，实际 `strtoul` 静默放行 `02000`、`+2000`、`" 2000"`；解析器现在要求首字符为 1–9，六种形状（0、02000、+2000、前导空白、-1、溢出）全部退出 2，只有纯数字放行（本机逐项实测）。② **定标工具依赖平台意外（实测确认）**：`tq-clock-rate` 用 `hipHostMallocDefault` 却由核写入该字——本机实测该分配返回 flags=0（不含 Mapped）时核写入照样落地，属契约外行为，不可依赖（AXIOM 1）；改为映射分配 + 逐卡校验 Mapped 标志 + acquire 读取，修复后速率不变（2257.1–2320.4 MHz），另补 `--help` 退出 0。③ 安装包完整：`cmake --install` 原装出两个二进制与文档却漏掉运维签收必需的 `validate-report.py`；现一并安装并做过独立前缀冒烟（装出二进制 50 轮 qualification=pass，装出校验器对装出二进制写出的报告 exit 0）。④ 校验器抗畸形树：`--self-test` 新增固定种子结构化变异模糊（报告与 bench 各 500 次逐层变异，任何结果必须落在"仍合法"或 EvidenceError 内）；本次交付前先用 6000 次变异跑过一遍，零意外异常后入库。config-test 补钉 `write_file_exclusive` 的符号链接拒绝（链接与被指文件原样保留）与目标目录不存在的干净拒绝。权威证据以最终源码在全新树重跑（注册表仍 25，新报告/bench SHA 见索引）。机器可读索引见 [audit/relay-probe-evidence-20261019-delivery.json](audit/relay-probe-evidence-20261019-delivery.json) |

较早的单卡探针报告（SHA-256 9d1d1429…684112，12 用例版）按证据身份保留为被取代的历史记录；
新报告覆盖并取代其资格结论，不修改其字节。
| 实时跨语言对照 | `tq_relay.python_crosscheck` 通过：C++ 移植与已验证的 `tp5_local_plan.py`（SHA-256 59129982…59e657）在四个固定 fixture 上 decision、错误码、totals、逐 rank 字节/MAC、middle/output 区间全部一致 | C++ 端口与固定比较对象一致；不是 GPU 或真实量化资格 |
| Python 参考基线 | `tp5_protocol_spec.py` PASS 8 checks；`output_credit_model.py` broken 32 状态 16 条不安全边 / fixed 17 状态 0 条；`tp5_local_plan.py --self-test` 16 组通过 | 提取产物可运行且与指南记录输出一致 |
| 安装与搬迁消费 | 安装到 `/tmp/tqrelay-prefix-*`，从空目录 `find_package(TQRelay)` 构建消费者，搬迁前缀后重建再运行 `consumer ok` | 安装包自足，无绝对路径泄漏；不证明操作系统沙箱 |
| A 边界 | `fork/check-upstream.py` 通过，无未登记 A 变化 | 本批次未改 A |

本批次的通过范围是单线程可信调用方的 CPU 所有权模型与合成存储几何。它不构成
HIP 内存序、五卡前进性、模型数值或性能资格；指南 M0c 平台探针与 A 的最小执行
接口仍是下一步。通道账本的预算合同沿用 `TQScheduling::admission_budget`（指南
第 50 章场景组 5、6 的既有实现），不在 TQRelay 内重复。

## Final release gates

最终格式收尾仅删除 `vulkan-scratch-reuse.cpp` 与 `vulkan-set-rows-types.cpp` 各一个文件尾空行，逐字节验证其余内容不变。随后重新构建接缝测试，7/7 再次通过，日志为 `build-clean-vulkan/details-eof-seams-20261018.log`。交付源码摘要及这次重建单独记在 [machine-readable qualification](audit/details-qualification-20261018.json) 的 `eof_only_normalization`；此前 62 条 isolation、完整 GPU sweep 和 before/after 反例保留原始运行身份。A、B/C 生产源码与 exception manifest 均未因该格式收尾改变。

**当前状态：最终专项、完整来源隔离和 CPU / Vulkan 普通 tools 构建已完成；最终全量 GPU 仍有一个 FA 失败，较早 TOPK 失败归因未闭合。** 以下统一登记本批次实际身份与验证状态。较早失败保持原记录；它们不能被定向用例成功或一次 manifest 检查覆盖。

| 门槛 | 当前可回查事实 | 状态与边界 |
|---|---|---|
| 最终源码文件身份 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` SHA-256 为 `6453d62fe781b6452406476f88ba4145fc55af7c7207b68357b1d8032a128c6a` | 这是一个源码文件的摘要，不是整 A 目录摘要或 commit。固定 A pin 不变，clean 提交历史以 Git log 为准。 |
| 最终 manifest 身份 | `fork/upstream-exceptions.json` SHA-256 为 `830a973a4e373d122d54f236984785de2d1b9cfeb3a8a67716a998c74b9f63dc` | 下述最终 archive 与 GPU 专项使用该内容身份；各文件集合和完整摘要以 manifest 为准。 |
| 完整 CPU 来源隔离 | `build-isolated-details-qualified-20261018`，62/62 条命令通过，2026-10-18 11:57:10 UTC 结束 | 从 pin archive 与经验证 overlay 重建；五 B / 五 C 独立安装，删除 source/build 副本及原前缀，搬迁后的实际 batch、logits、stream、imatrix 消费通过。没有使用 `reused_unverified` 代替。 |
| 最终 Vulkan 专项 | `build-clean-vulkan/details-final-gpu-pii1wep9`，7/7 接缝 CTest 通过 | 含 5 个 scratch case、2 个 source-alias case、16 次 SET_ROWS packed 比较；有真实重建重装后的执行记录，未转借较早版本的结果。 |
| 源数据原地修改 before/after | 冻结的 4-byte-offset probe 修复前第二个 matmul 64/64 错、最大绝对差 15.4662；修复后 control/inplace 均通过 | 每个 matmul 的全部 64 个输出，以及 SCALE/源存储各 514 个值的相应检查误差为 0；数值身份与较早零偏移 probe 分开保留。 |
| 最终 Vulkan 文本消费者 | 同一最终 GPU 目录：10 条命令，6 个模型配置各 3 个请求，脚本约 14.5 秒 | 18 个真实模型请求通过；脚本用时不是吞吐基准，也不提供 HTTP SSE 资格。 |
| 最终全量 GPU backend | 25932/25933，exit 1，用时 454.338607 秒 | 仅剩同一 mask+sinks、Q8_0 K/V、head dimension 256 的 FA case，ERR 0.009055835 > 0.0005；最终 TOPK 未复现，不能撤销 +12 full 的原失败或视为归因闭合。 |
| 普通 CPU tools 入口 | `build-clean-fork/details-build-entry-mob1rldf/result.json`，`build.py --tools` exit 0，用时 67.8365955 秒 | 这是实际常规构建入口的执行回执。 |
| 最终 A 差异统计 | 17 个 A 源码文件 +357/-43；另有 49 行自有 CI，共 18 路径 +406/-43；本批次新增 A 代码 29 行 | 最终 Git numstat 已核对；计数不代替内容摘要或运行资格。 |
| 普通 Vulkan tools 入口 | `build-clean-vulkan/details-build-entry-4byg2csx/result.json`，`build.py --tools` exit 0，用时 7.313101 秒 | 这是实际常规构建入口的执行回执。 |
| 可移交证据索引 | [audit/details-qualification-20261018.json](audit/details-qualification-20261018.json) | 本批次起点、源码与 manifest 摘要、命令、库/设备身份及结果分别记录；不借用旧 release receipt 取得新资格。全量 FA 失败和旧 TOPK 未定位结论继续保留。 |

“+12”只指下文远程日志使用的 scratch 修正阶段简称，不是最终 A 差异总数或提交身份。GitHub CI、CUDA 或发布流程未执行的部分保持未执行；成功的本地检查不能代替这些外部结果。

## 本批次已执行的检查

| 检查 | 实际结果 | 可以支持的结论 |
|---|---|---|
| 最终 CPU 来源隔离与安装 | 62/62 条命令通过，五个 B、五个 C 从各自源码副本构建、安装并搬迁消费 | 已在上述最终源码/manifest 身份上重建 archive + verified overlay，实际消费者通过。 |
| 新 B sanitizer | TQText、TQCalibration 的 ASan 与 UBSan 测试通过 | 两个新独立库的已执行用例通过；不扩写到全部项目、模型或早先未取得资格的 logits ASan。 |
| 生产 writer 互操作 | `2777543b2358c88cd1c2000f7fa342b764a0e213` 的原 calibration writer 两个文件逐字节相同，实际写出所有 flags 的 fixture；新 C 的 12 tensors 检查通过 | 旧生产格式与新 reader/exporter 互操作，而非历史真实模型采集质量。 |
| 实际 A imatrix loader | `llama-quantize` 读入 6 importance entries、9 chunks；模型文件被故意设置为不存在，后续阶段 exit 1 | 真实 A loader 接受并消费生成格式；整个量化命令没有成功，不宣称完成模型量化或质量验证。 |
| 最终 Vulkan 接缝 | 7/7 CTest 通过，含全部 5 个 scratch 用例、2 个 source-alias 用例与 16 次 SET_ROWS packed 比较 | 实际 RX 6800 Vulkan 操作的指定图与布局；不是全部缓存组合或全部设备。 |
| 最终 Vulkan stream/stop 模型运行 | 10 条命令，6 个模型配置各 3 个请求，共 18 个成功模型请求，脚本约 14.5 秒 | 实际模型 token、同配置 stream 开关、序列释放和文本事件合同；不是 10 次模型运行、吞吐基准或 HTTP SSE 资格。 |
| 已完成的全量 CPU backend ops | 27639/27639 通过 | 保留该次运行身份；最后广覆盖复验见统一门槛表，不能推导未运行平台。 |
| 最终全量 Vulkan backend ops | 25932/25933，exit 1，454.338607 秒 | 同一 FA case 仍失败；最终 TOPK 未复现，较早 +12 full 的 TOPK 失败与未定位状态保持记录。 |
| 普通 tools 构建 | CPU exit 0，67.8365955 秒；Vulkan exit 0，7.313101 秒 | 两个常规构建入口均实际完成；不把构建时间解释成推理吞吐。 |
| Vulkan K/V smoke | 9 个 K/V 组合的 pp/tg 共 18 行，exit 0，1 repeat | 这些配置的一次执行通过，不是统计可靠的性能比较或收益证明。 |
| CUDA | 没有可用 CUDA 设备 | 无运行资格；构建选项存在不代表执行通过。 |

各阶段构建与数值结果以实际运行身份为限。当前结果未提供新的吞吐、延迟分位数、峰值显存、模型 NLL 或量化质量测量。实际 GPU 资格来自 Vulkan 执行，不从 `--gpu-layers` 请求参数推断。

## 增量文本：独立合同与真实消费者

B `TQText` 拥有 UTF-8 尾部和 stop 匹配状态，不调用 A、I/O 或兄弟库。输入分块不会改变匹配结果；每个非法字节及最终未完成字节各转为一个 U+FFFD。停止字符串非空、必须是合法 UTF-8，限制为最多 32 个、每个 4096 bytes、总量 65536 bytes。同一字节位置完成的匹配选最长，相同字符串按配置顺序。

真实 C 用 A 返回的 token piece 推进 B。`--stream` 要求 `--json`；每个成功请求恰有一个 terminal text 事件，`offset_bytes` 连续，拼接增量等于 `result.text`。文本抑制不回删采样 token。匹配确定后才停止后续 decode；最终 UTF-8 修复可能在已提交过早先 token 后确定 stop，因此不宣称任何 stop 所在 token 都从未进入 A。C 消费本 batch 的全部输出、同步并删除序列后，才回收预算和槽位。

GPU 的六种实际模型配置为 serial-buffered、serial-stream、stop-buffered、stop-stream、parallel-buffered、parallel-stream，每种三个请求。其余命令检查参数失败，不能作为额外模型请求计数。stop 用实际模型基线文本形成，并验证跨 delta 的匹配；对照保持同配置模型提交形状。独立 B 单测负责分块、无效 UTF-8、匹配冲突、实例和结果所有权，不能将模型 fixture 的成功扩写为它单独覆盖所有字符路径。

上述真实模型 streaming 对照显式使用 `--ignore-eog` 和 `--no-phase-memory`。因此它不单独验证自然 EOG 时的最后文本边界，也不构成 stream/stop 与阶段释放组合的模型资格；finish-time UTF-8 与 stop 合同由 B 单测覆盖，阶段释放保留其独立实际回归。

该工作重建 `server:Q14` 的有用文本目的，不宣称复刻旧无 stop / 非 stream fastpath 或性能收益。`server:T09` 只涉及跨块 terminator 机制。`Q15/Q16/Q17`、完整 thinking/lane parser、HTTP SSE、网络送达、取消和重试仍有各自待迁合同。

## WXQCAL03：旧生产 writer 与真实 A loader

实际互操作使用旧提交 `2777543b2358c88cd1c2000f7fa342b764a0e213` 的 `wanxiangqi/lib/calib-dump.cpp` 与 `calib-dump.h`，不修改这两个文件。fixture driver 调用原 `wxq::write_calib` 写出两层小数据，包含全部三个统计 flags，decoded 与 declared total 均为 9。未找到可用于本次验证的历史真实校准样本；这个 fixture 是原生产 writer 生成的格式样本，不是训练或推理模型的测量。

新 C 的实际输出包含 12 个 F32 GGUF tensors。独立标准库 reader 检查形状、次序、typed metadata 和值；另由实际 A `llama-quantize` 读取为 6 个 importance entries 与 9 chunks。故意不存在的模型随后令命令退出 1，这是有意限定的 loader qualification。成功消费格式不证明 Tensor 命名适合任意模型、收集统计来自相同模型，或任何量化质量收益。

C 使用安装后的 `ggml::ggml-base` 与 TQCalibration；不加载模型，不创建 inference context，不初始化 GPU。实际 GGML 元数据 context 仍存在。A 的 package config 会继续发现该安装启用的 backend，因此不能删除 backend 安装文件来证明依赖缩减。

关键数值合同与 A 的实际消费保持一致：

- 保留每个 expert 的 raw sum2，不预先归一化；A 用 F32 sum2 / F32 count 求 importance。
- count 必须为精确 F32 整数且不超过 INT32_MAX；`chunk_count` 也为正且不超过 INT32_MAX，`chunk_size` 为 1。既考虑 F32 精度，也考虑 A 使用 `std::lround` 的可交换范围。
- 拒绝源非有限值、负平方和、F32 溢出与非零下溢，并检查实际 F32 除法不会把非零 importance 静默变成零。
- routing 计数与 expert activation 计数是不同统计人口，不能机械要求 routecounts 等于 `exp_n`。
- 默认拒绝不完整所选范围和未观测 expert。`--allow-partial` 保留所有实际原因，包括旧末 batch 的 overshoot；`--unseen global-mean` 明确标记 estimate、原始策略和精确填补 ID，不把填补 count 当作测量。

Reader 在分配前检查层大小，协方差分块验证后丢弃；C 一次持有一层、16 KiB 转换缓冲和私有 payload spool。输出完整验证、关闭和 sync 后以同目录 hard link 无覆盖发布。spool 与 staging 共存需要约两倍最终 payload 磁盘空间。平台为小端 POSIX、调用者控制目录；不承诺信号清理或断电目录持久性。stdout 后续失败准确报告完整文件已发布。

完整复现命令、旧 writer 源码摘要记录方式、可执行测试与失败合同见 [imatrix-compose/README.md](imatrix-compose/README.md)。本节不借用较早 corpus writer 的文件摘要作为 calibration writer 身份。`small:S04` 的格式与导出已重建；`small:S05` 的真实 callback 采集、DFS、checkpoint 和模型质量验证仍待迁。

## Vulkan：scratch 覆盖、同步与既有 SET_ROWS

本轮实际 probe 调用已安装 A 的公开 GGML/Vulkan API，直接运行 GPU，不使用 scheduler CPU fallback、私有头、模型或替代 kernel。原图使用同一个不连续 F32 activation view，依次执行 matmul、中间操作、第二个 matmul；两个 F16 权重内容相同但拥有不同 tensor。

原 scratch 修复前，small-softmax 和无 mask FA 控制通过。32768-column softmax 后的第二个 matmul 有 32/64 值不符，最大绝对差约 3380.01；masked one-head FA 后有 32/64 不符，最大绝对差约 `2.70e13`。这两个操作会覆盖 `prealloc_y`，必须同时清除转换缓存的 pipeline、source 和 decode-layout 三个身份字段。

执行顺序是另一项责任。large softmax 已标记最后 dispatch 后需要同步，split-K FA 已有读取 mask bitmap 后的 reduction barrier；no-split 路径需要在最后读取后标记 Y 待同步，让下一次覆盖 Y 的 matmul 使用已有 pre-copy barrier。数值 probe 与源码责任检查分别提供证据，没有声称数值测试复现了缺同步竞态或运行过 validation-layer 同步检查。

初版 no-split probe 用复杂 fractional V：全部 524288 值与同 GPU 独立中间操作相同，但其中 475136 值未通过闭式门限，最大偏差 0.0146484。该失败是独立控制的数值限制，不能归因为 scratch 同步。最终 fixture 采用 binary-scaled V、Q/K 为零和明确 mask，使预期值可精确构造，保留原 attention absolute tolerance `2e-4`。scratch 修正阶段全部五个用例通过，包含 source-write 修复的最终安装重新运行也通过；no-split 对全部 524288 输出做同 GPU 独立对照与闭式检查。公开 API 不暴露私有 FA tuning 状态，no-split 名称表达该设备与源码分析对应的目标条件，不是运行时私有状态探针。

`backend:G44` 的旧建议建立在“只编译 F32 source”前提上，固定 A 源码实际生成 F32/F16 到 Turbo2/3/4 的 I32/I64 SET_ROWS 变体。因此保留 A 已有支持。实际测试覆盖 Q8_0、Turbo2/3/4 四种目的类型及两种 index 类型，共八个配置，每个比较 F32 和 F16 scatter，共 16 次 packed 比较；同时检查其他行 sentinel。这个结果不独立量化 quantizer 误差，也不涵盖所有 stride、重复 index 或其他 GPU。

详细图、参数和源码责任见 [vulkan-scratch-reuse-protocol.md](tests/a-seams/vulkan-scratch-reuse-protocol.md)。`backend:G14` 的设备 pinned ownership 没有迁入，不能借本节取得多 GPU 资格。

## Vulkan：源数据原地修改是另一个反例

scratch 修正阶段的后续审查发现，同一 source view 指针的底层 storage 被 `ggml_scale_inplace` 改写，缓存 key 仍可命中旧转换结果。初始零偏移 probe 的 disjoint control 通过；inplace 图中 SCALE 和源存储全部 514 值正确变为两倍，但第二个 matmul 的 64/64 个值错误，最大绝对差约 16.0306。记录的一个值为预期 31.8711、实际 15.9355。

这建立了与 scratch 覆盖不同的公开图正确性问题：barrier 可以排序原始 storage 写入，不能刷新另一个 converted copy。冻结的后续 probe 使用 4-byte-offset 的同一 view、独立同 GPU 原始/预先加倍 reference，并比较全部 matmul 与 storage 值。其修复前第二个 matmul 有 64/64 值错误、最大绝对差 15.4662；重建重装最终 A 后，control 与 inplace 两个模式通过，每个 matmul 的全部 64 个输出，以及 SCALE/源存储各 514 个值的相应比较误差为 0。非零偏移改变了有效数据，所以两个 probe 的误差数字分别保留。实际最终来源见 [Final release gates](#final-release-gates)。

修复的 A 合同是：记录的 tensor write 只要与 cached source 的 storage range 重叠，就失效全部三个转换身份字段，包含 exact alias 和每个 fused-write 输出；不能在找到普通同步依赖后提前停止检查。无需引入新 B 或新的缓存类型。测试和范围见 [vulkan-activation-cache-protocol.md](tests/a-seams/vulkan-activation-cache-protocol.md)。该反例不证明某个模型或 recurrent clear 实际触发了这个序列，也不覆盖所有融合、stride、外部 host 修改或图捕获。

## 全量 Vulkan：最终仍失败，旧反例不撤销

scratch 修正的 +12 阶段全量 backend ops 为 25931/25933。FA 失败配置包含 mask、sinks、Q8_0 K/V 和 head dimension 256；对应单项在 baseline 与 +12 都失败。该次 TOPK 也在全量运行中失败，对应单项在两边均没有复现。

最终包含 source-write 修复的完整 GPU suite 为 25932/25933，exit 1，用时 454.338607 秒，只剩同一 FA case，测试报告 ERR 0.009055835 超过门限 0.0005。TOPK 在这次完整运行中没有复现。重复运行结果的变化没有定位旧 TOPK 的原因，原失败仍保留。

FA 两边失败是限定 case 的对照，TOPK 单项不复现保留了测试顺序、状态或非确定性等未定位可能；不能把它们合并成“已排除回归”。本页不猜测根因，也不以 99% 以上通过率替代明确的失败处置。最终测试状态和普通构建入口回执统一登记于 [Final release gates](#final-release-gates)。

## 同日较早发布候选记录：起点 3746ebf82585

以下保留原文。其“本轮”“当前”及原记录中的 18 项 manifest、三个 B / 四个 C、47 条隔离命令，仅指起点 `3746ebf82585f47c88fb48097f6ae45cb972b031` 的较早批次。本节内部还保留更早的 16 个 A 文件、三 B / 三 C、41 条命令的历史。旧 receipt 不承担本页新增能力和最终 A 的来源证明。

当前 A 固定为 `cda0f9256665845cd9d6f3a0176880883f653a3d`。登记例外共18个文件：17个 A 代码文件合计 +328/-43，另有49行自有 CI 工作流。本轮新增的生产 A 修改只有 Meta 异步偏移一个文件的 +4/-4；其余新行为由 B、C 或独立测试承担。

本轮可移交摘要为 [audit/release-qualification-20261018.json](audit/release-qualification-20261018.json)，记录关键源码摘要、模型身份、实际命令回执、反例和适用范围。原始 stdout、stderr、二进制 tape 与完整命令保留在对应 `build-*` 目录。本页后半保留同日上一轮记录，并明确标为历史验证，不能将其未复跑的部分自动当作本轮结果。

### 本轮实际参与者

- Linux x86-64，GCC 16.2.1；主测试 Python 3.14，CMake CLI 测试使用 Python 3.12.13。
- 模型：`/home/kunweiz/models/lfm25/LFM2.5-8B-A1B-UD-IQ4_NL.gguf`。
- 模型 SHA-256：`a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a`，本轮重新核对。
- Vulkan：单张 AMD Radeon RX 6800，RADV / Mesa 26.2.2。模型日志实际确认25层卸载；没有把请求的 GPU 参数当作执行证据。

### 本轮验收概览

| 检查 | 实际结果 | 主要证据 |
|---|---|---|
| A 边界与反例 | 18项内容 / mode / 消费者匹配，12个临时 Git 仓库测试通过 | manifest、`check-upstream.py`、`tests/test-upstream-boundary.py` |
| 有界预算 B | 2/2 CTest；双资源、失败保持、陈旧 / 外来 / 重复释放及整数边界 | `scheduling/tests/admission-test.cpp`、独立安装测试 |
| 队列与槽位复用 C | CPU、Vulkan各7个运行、60个请求成功；固定配置的阶段开关对照相同 | `build-release-admission-qualified-{cpu,vulkan}/` |
| Meta async 偏移 | 原 A 零偏移16控制通过、非零 get/set 分别断言；修复后48例通过 | `build-release-meta-before/summary.json`、Meta 协议 |
| recurrent 生命周期 | CPU、Vulkan各24邻居 / 8共享快照检查，恢复继续最大绝对差0 | `build-release-seams-{cpu,vulkan}/` |
| 完整 logits 采集 | 每后端15行×128000值，重复 tape 按位一致；标量形状行选择差0 | `build-release-logits-real-{cpu,vulkan}/` |
| logits 文件和 CLI | CPU、Vulkan链接构建均2/2，C++17严格警告；独立 UBSan writer通过 | `logits-compose/tests/` |
| 独立比较器 | 20项真实 CLI 测试通过 | `tools/tests/test-compare-logits.py` |
| 独立交付 | 47个命令成功，重新构建 A、三个 B、四个 C，搬迁后消费者及真实模型通过 | `build-isolation-release-20261018/result.json` |
| 构建入口 | CPU默认、Vulkan `--tools` 完整流水线均退出0 | `build-release-cpu.log`、`build-release-vulkan.log` |
| GitHub CI | 已配置无模型完整隔离流水线；本地对应流程通过 | 工作流尚未推送或在 GitHub 执行 |

构建、纯库测试、真实模型、数值对照和性能测试是不同证据。本轮没有吞吐或峰值显存测量，不是上游全部测试套件的通过声明。

### 准入、排队与真实状态释放

`admission_budget` 独立拥有 token 和驻留槽位预算。C 用实际 `llama_n_ctx_kv()` 初始化预算，每次预留已知上界 `prompt_tokens + predict - 1`。请求 ID 与 A 的可复用 seq ID 分开；reservation 带 owner domain 与 generation。队列可长于 batch 和驻留槽位数。

C 消费完当前批次所有输出并同步后，先成功调用 `llama_memory_seq_rm()`，再释放预算、增加可用槽位。B 的 reservation 析构不会代替 C 删除真实状态。容量不足留在 FIFO 队列；不能分配的单请求明确失败。本轮没有引入 HTTP 取消、抢占、LRU、失败重试或未知长度流式增长。

`batch-compose/tests/test_admission.py` 在每个后端执行以下七个运行。每请求135个提示 token，生成16个 token，实际 decode 行数150。

| 配置 | 请求数 | 驻留上限 / 物理 KV | 验证 |
|---|---:|---|---|
| three-wide，phase off/on | 每臂9 | 3 / 2048 | 相同配置两臂完整输入行与生成结果相同 |
| token-pressure，phase off/on | 每臂9 | 3 / 256，实际仅驻留1 | token预算等待、回收、固定配置结果相同 |
| slot-pressure，phase off/on | 每臂9 | 1 / 2048 | 槽位预算等待，和上述串行运行的结果相同 |
| repeat-wave | 6 | 3 / 2048 | 同三个提示在相同槽位复用后与第一波新槽位完全相同 |

合计每后端 `6×9+6=60` 个成功请求；两后端共120个。检查真实提交行、token反馈、位置、请求完成、generation、预算守恒，以及 phase release 前正数 / 后零的计算分配字节。等待计数按调度观察点累积，不能当作独立请求数或延迟样本。这里的120不是吞吐样本。

#### 保留的跨批形状反例

最初要求并行与串行 greedy token 完全一致的测试失败。随后在本轮准入修改前的原 C 二进制上，关闭 phase release、保持提示与模型相同，独立复现三路与单路的结果差异；单路 KV2048 与 KV256 则完全相同。

原 C 来自本轮起点 `3746ebf82585f47c88fb48097f6ae45cb972b031`。其 CPU、Vulkan二进制摘要、实际命令、确切 UTF-8 stdin 与返回 token 保存在本轮摘要的 `shape_counterexample_before_admission`。CPU组选取原fixture的3/4/5，Vulkan选取0/1/2；已将全部提示token行逐条对上原运行记录。它们后来被完整安装覆盖，复现时应从该提交重建。最初失败记录仍在 `build-release-admission-cpu/run-cga4dqjq` 与 `build-release-admission-vulkan/run-cc0q5fad`，没有改写成成功。

本轮队列测试的 CPU 在请求4的第5个生成 token 出现差异，Vulkan在请求1、4、7出现差异。开关 phase release不改变各自结果。这个证据建立了修改前已存在的批形状敏感性，**没有定位数值原因**，也不能据此认定差异无害。验收明确比较同配置形状、串行容量对照和同槽位重复；跨形状差异继续单列。

### logits：全行采集与有限的独立对照

新 C `tq-logits-compose` 用真实 tokenizer 产生输入 token，以 teacher forcing 调用 A。默认逐 token 保存完整词表行；前缀模式只保存前缀最终行，再保存后续各行。每次 decode 后同步，行内非有限值立即失败。完整尺寸先检查，writer不保存历史行。

CPU、Vulkan各执行11条命令：完整采集、重复采集、标量形状suffix、标量形状prefix-only、chunked-prefix、旧比较器、新文件拒绝覆盖、上下文不足、输出尺寸不足、独立 batch C对照，以及 Linux `/dev/full` stdout故障。

| 验证内容 | CPU | Vulkan |
|---|---:|---:|
| 输入 token / 词表 / 完整行 | 15 / 128000 / 15 | 15 / 128000 / 15 |
| 重复 tape | 全部192万值及文件字节相同 | 相同 |
| 同标量形状suffix全行最大绝对差 | 0 | 0 |
| 同标量形状prefix-only全行最大绝对差 | 0 | 0 |
| 独立 batch C：相同提示行，最终greedy token / 最大logit，门限1e-5 | 通过 | 通过 |
| chunked vs scalar最终全行最大绝对差，仅诊断 | 2.6373519897460938 | 0.9928340911865234 |
| stdout写失败 | 非零退出；已完整发布的tape保留且字节正确 | 相同 |

独立 batch C只提供最终greedy token和最大logit，因此这个独立对照**没有比较第二个实现的全部词表值**。重复tape、suffix/prefix选择与chunked/scalar诊断则读取并比较实际完整行。chunked/scalar差异不能称作已证明无害的舍入误差；没有跨后端相等或模型质量声明。

Writer单测直接检查little-endian字节、负零、尺寸、token范围、NaN、缺行/多行、已存在文件、symlink、竞争发布者、临时文件样式目标名和真实文件大小限制故障。写或sync失败是终止状态；不能捕获后继续发布部分文件。完整文件先发布，stdout随后失败则明确报出“文件已完整发布”。独立UBSan检查通过；ASan环境未取得可靠资格，本记录不宣称ASan通过。

TP5LOG2头不含模型、tokenizer、构建和设备身份，应连同命令、日志及摘要保存。发布使用POSIX同目录hard link与调用者控制的目录；不承诺Windows、恶意目录写者防护、断电目录持久性或信号时临时文件清理。接口和复现命令见 [logits-compose/README.md](logits-compose/README.md)。

### Meta：公开异步传输的 before / after

新回归直接创建两个真实CPU child backend，覆盖split axes0/1/2与mirrored、F32/F16、3+7不均分片、full/middle/tail、独立get和set，以及前后guard与未触及字节。

原A的零偏移16个控制通过，非零get、set单独运行均在旧 `offset == 0` 断言处以SIGABRT(-6)失败。A修复只移除该过度限制，并把逻辑chunk起点换算为每个child自己的偏移。修复后48例通过。CPU安装的全部A接缝为4/4，Vulkan链接安装为5/5；**Meta用例在两次运行中都使用CPU child**，没有取得GPU异步、多GPU或异构Meta的运行资格。

连续、单segment、chunk对齐与repeat条件保持。没有新增API、后台队列、cache、pin或所有权协议。完整形状与检查方式见 [meta-transfers-protocol.md](tests/a-seams/meta-transfers-protocol.md)。

### recurrent：直接复用当前 A

源码复查表明：当前A在构造与 `clear(true)` 时整buffer清零；`seq_rm` / `seq_keep` 修改元数据，通过空闲zero row与图内初始化/聚集取得新状态。旧M33性能记录减少的是旧fork自己加的eager clearing；把它搬到当前A会增加写入和fence，本轮没有修改A的recurrent代码。

公开接口测试使用上述LFM2.5 hybrid convolution/KV模型，实际 `n_rs_seq=0`。CPU、Vulkan各执行24次decode、189行输入；24个邻居快照与8个共享快照检查通过，合法恢复后同形状teacher-forced继续的最大绝对差为0。覆盖三个槽位两次复用、COW存活者、dirty槽位恢复、`clear(false)` / `clear(true)`再填充，以及不支持的部分回滚失败后原快照不变。batch输入在显式同步前始终存活。

这不等于一般GDN/recurrent行数、MTP、PLE、HTTP取消或跨进程snapshot兼容性证明。源码另发现坏restore可能在完整验证前改变目的状态；**没有宣称损坏snapshot恢复具有事务性，也没有在本轮修复它**。操作边界见 [recurrent-lifecycle-protocol.md](tests/a-seams/recurrent-lifecycle-protocol.md)。

### 独立交付、构建入口和 CI

本轮实际运行：

```sh
python3 fork/tests/isolation.py \
  --output build-isolation-release-20261018 --jobs 4 \
  --model /home/kunweiz/models/lfm25/LFM2.5-8B-A1B-UD-IQ4_NL.gguf
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
python3 fork/build.py --backend vulkan --tools --build-dir build-clean-vulkan --jobs 4
```

47个隔离命令全部成功，A来源标记为 `built_from_pinned_archive_and_verified_overlay_bytes`，没有使用 `--a-prefix` 复用捷径。A从pin的Git archive导出，只覆盖18个经验证的完整文件；A源码没有 `fork/`。三个B、四个C从各自源码副本构建，B不调用A或兄弟B，C只查找安装包。

删除原源码/构建副本和原安装前缀后，搬迁后的真实包消费者仍能编译运行；新admission API参与外部消费者编译，安装后的batch模型与logits模型测试通过。logits的比较器和测试脚本在删除源码前拷出。这个流程证明安装消费与来源隔离，不等于操作系统沙箱。

两个真实构建入口退出0。Vulkan `--tools` 路径补建A自身tools安装要求的 `mtmd`，从C的构建脚本解决缺失安装依赖，没有改A的CMake。工具可执行文件位于对应 `upstream/bin/`。

新增 `.github/workflows/fork-extensions.yml` 配置只读权限、独立无模型CPU隔离检查和失败回执上传；没有改上游原工作流。本地完整流程已通过，GitHub工作流尚未推送或执行，不能称作远端CI绿灯。真实模型、Vulkan与可选recurrent资格仍由显式本地命令提供。

### 本轮范围与后续候选

没有新的整模型吞吐、reset加速、峰值VRAM或自动容量适配结论。CUDA、HIP、Metal、Windows、macOS、多GPU与异构Meta没有取得本轮资格。

auto-fit/slack、PP reserve、idle LRU、物理recurrent槽、HTTP取消/恢复、流式stop/usage、在线corpus、状态envelope、校准/imatrix和其他局部算子继续逐项列在 [MIGRATION.md](MIGRATION.md)。714路径的审计分母与546完整改动内容/168选定范围的阅读记录保持不变，不能用这些测试覆盖所有历史实现。

### 同日上一轮验证记录（保留原证据边界）

以下段落中的“本次”指本轮起点之前的小功能补全；当时为16个A代码例外、三个B和三个C，隔离流程41步。本轮重跑项目以上面的记录为准。其可移交摘要仍是 [audit/validation-20261018.json](audit/validation-20261018.json)。保留旧数值用于回查，不将未重跑的HTTP、corpus互操作或采样测量写成本轮新结果。

#### Prefill / Decode：真正发生了什么

`fork/batch-compose/tests/real-batch.py` 启动真实 C，使用已安装的 A 与 B。先让两个请求生成两批，再接纳第三个提示；逐条检查真实 `llama_decode` 输入的请求 ID、位置、token 和 logits 标记。验证 Prefill 期间选择提示请求、完成后原 Decode 请求继续，且已采样 token 真正成为后续模型输入。

每个后端各运行释放开启、关闭两臂。配置为每请求逻辑窗口 512、物理 KV 1024、batch 16、ubatch 8，每请求生成 12 token。A 的公开 getter 验证逻辑 / 物理设置确实分离；每臂都提交 22 批、145 行，开关两臂的所有生成 token 和输入行完全相同。Vulkan 两臂的实际卸载记录均为 25 层。

| 当前计算分配器字节数 | CPU | Vulkan |
|---|---:|---:|
| 关闭释放时维持的工作区 | 1,726,848 | 2,846,956 |
| 第一次 PP→TG 释放后 | 0 | 0 |
| 第一段两请求 Decode 重新分配后 | 1,074,368 | 1,168,472 |
| 插入 Prefill 后第二次 PP→TG 释放后 | 0 | 0 |
| 三请求 Decode 重新分配后 | 1,611,456 | 1,738,356 |

这里读的是 `llama_memory_compute_bytes()`，表示 context 的计算分配器当前拥有的 buffer 字节数。它排除模型、KV、recurrent、输出和 provider cache，不是显卡驻留量、进程峰值或系统回收显存。这个小负载证明了释放与按实际图重建；它没有测量吞吐收益，也不证明扩大物理 KV 后注意力工作区不增长。

同一脚本还实际运行不足的物理 KV 配置与缺失模型，要求 A 明确失败。非法配置不能用模拟日志或伪造成功替代。

##### 分配器生命周期

`tests/a-seams/compute-memory.cpp` 使用真实 GGML CPU 后端：大图分配、排队异步读回、由 release 自己完成 fence、字节归零、重复 release、小图重建、再扩回大图。它检查保留输入/权重、scheduler 和 callback 身份，并执行重建后的计算。Release 构建中的失败检查始终有效。

API 前置检查拒绝 metadata-only context、活动 optimizer 和共享 CUDA phase arena；正常路径同步、失效两个可复用图结果，再释放分配器。模型 / KV / recurrent / output 的所有权保留由源代码边界与真实输出对照支持；没有宣称跨 context 共池或 target / draft 联合分配已经实现。

#### 原版 HTTP server

`tests/server-phase.py` 只启动并清理自己创建的 localhost server。真实设置为 logical 512、physical KV 1536、3 slots、Vulkan。两个 SSE 流各收到至少两个生成块后插入第三个长提示；初始两条流当时均未在客户端结束。

三条请求分别完成 48、48、8 个 token，槽位和 limit 结束原因正确。日志出现 `prefill → generation → prefill → generation`，实际卸载 25 层，`n_ctx_kv=1536`。关闭 Prefill 优先但开启阶段释放的组合，在加载不存在的模型前就报出专用参数错误。

初次 smoke 的三条请求也都完成，但默认日志隐藏 loader 信息，导致设备证据断言失败。最终脚本启用 `--verbose --log-colors off`，保留原来的正数卸载 / 物理容量断言，完整复跑通过，未放宽检查。详见 `build-server-phase-verified-20261018/summary.json` 与 `server.log`。

HTTP 接收时间不能证明内部批次严格暂停；逐批暂停 / 恢复的证据来自上一节 C 的真实批次输入。此次 server smoke 没有证明 MTP、语法、随机 RNG、取消或压力重试等全部协作组合。阶段释放后，A 析构时可能打印实际计算 buffer 与初始预留预测不同的诊断；当前观察接口和结果以实际分配及输出为准。

#### 两个小型正确性修复的 before / after

##### CPU：显式 F32 FlashAttention

针对 F16 V、显式 F32 precision 的实际 GGML CPU 操作，在相同测试中：

| KV 长度 | 固定 A 原实现最大绝对误差 | 修复后最大绝对误差 |
|---|---:|---:|
| 256 | 0.000388145447 | 0.00000011920929 |
| 1024 | 0.000388026237 | 0.00000011920929 |

原实现退出失败；四行级别的累加路径修复后通过。用例为 `tests/a-seams/flash-attn-precision.cpp`。这不是一次整模型质量评分，也不应归为无收益的量化实验。

##### Vulkan：公开 Turbo WHT

同一个 `tests/a-seams/vulkan-wht.cpp` 直接分别调用真实 CPU 与 Vulkan backend，没有混合调度器的 CPU fallback。12 例覆盖 group 32 / 64 / 128、正反变换、513 行跨工作组以及既有 scale 参数。

原 A 失败 10 / 12，仅未缩放 group128 的两个控制组通过。修复两个 A 文件后 12 / 12 通过，本设备最大绝对误差为 0。输出先填 NaN，遗漏写入会失败；所有数值须有限。当前支持仍限制在 128 对齐物理行和已验证 scale 布局，shader 越界返回保持整个 workgroup 一致。

记录：`build-clean-fork/wht-after.log`、`wht-vulkan-build.log`、`wht-vulkan-install.log`；完整复现合同见 [tests/a-seams/vulkan-wht-protocol.md](tests/a-seams/vulkan-wht-protocol.md)。

#### 生产 corpus 格式与旧入口复核

互操作脚本从旧提交 `2777543b2358c88cd1c2000f7fa342b764a0e213` 导出真正的 `wanxiangqi/server/server-corpus-dump.h/.cpp`，不改其代码，与已安装 A 头文件及独立 TQCorpus 编译运行。旧 writer 产生混合和空请求 corpus，新 B 读取 / 保存后逐字节比较，最后由旧 reader 读取；验证前后原输入文件保持不变。

覆盖四种 kind、64 位时间戳、空请求、重复请求、前缀终点与追加。旧源码 SHA-256：

- `.h`：`0ce1d80aff1b75aab67c2d3c36a8f40e632ffbeca93c214e8c4c7995d0ef6209`
- `.cpp`：`f031635a31c9c4a9c40633d99b09d70365659371beff6b353a3e743fe97518e7`

本次修改 A 与 corpus 后重新运行 `tests/real-model.py`，没有仅引用第一轮结果：

| 当前构建上的真实对照 | CPU | Vulkan |
|---|---|---|
| greedy compact / 完整行 | 24 / 24 token 相同，compact 实际执行 24 次 | 相同 |
| 正温、固定 seed 的完整采样 | 24 / 24 token 相同 | 相同 |
| 6 请求前缀 / 独立重放 | 实际 decode 44 / 120 token | 实际 decode 44 / 120 token |
| 整体 mean NLL 绝对差 | 0 | 0 |
| 最大逐请求及反转顺序差 | 1.8947806286936005e-15 | 9.473903143468002e-16 |
| 预设 NLL 误差门 | 1e-5 | 1e-5 |
| 上下文错误、拒绝覆盖、失败后文件不变 | 通过 | 通过 |

两种不同后端之间没有宣称 logits 相同；NLL 数字只是本测试语料的条件分数。当前 CPU 的 prefix / reference 均为 5.2485678755225615，Vulkan 均为 5.15723864954567。误差门没有根据结果放宽。

#### A / B / C 独立构建与搬迁

```sh
python3 fork/tests/isolation.py \
  --output build-isolation-small-20261018 --jobs 4 \
  --model /absolute/path/to/LFM2.5-8B-A1B-UD-IQ4_NL.gguf
```

以上为可替换模型路径的复现示例，实际命令保存在该目录 `commands.json`。本轮没有使用 `--a-prefix`：A 从 pin 的 Git archive 构建，只覆盖 manifest 已验证的 16 个完整文件，导出 A 不含 `fork/`，其自身 `test-sampling` 通过。

三个 B 分别从只含自身代码的副本构建 / 测试 / 安装；三个 C 分别只消费安装依赖。流程删除源码和构建副本、搬迁安装前缀，再编译实际包消费者，并执行搬迁后的真实 batch 模型比较。41 个命令全部成功。B 的输入 / 输出所有权和实例隔离测试不需要模型服务；C 的模型证明使用真正 A 和 B。

`--a-prefix` 是可选便捷路径，记录为 `reused_unverified`，不能拿来证明该次重新构建了对应 A。独立源码副本和安装消费也不等于操作系统沙箱。

#### 性能资料的边界

第一轮实际 CPU 采样微基准仍作为单一阶段证据保留：248,320 logits、top-k40、唯一最大值，每臂预热 10、计时 200 次，4 个 AB / BA 配对；完整行约 252–312 µs / row，compact 约 29.9–30.0 µs / row。每次使用真实 A sampler 检查 token。这不是整模型的加速倍数。

旧 fork 的 +3.66% compact、TP5 LateBind、router 与 MMVQ / MTP 数据是归档历史，不是这次新分支测量；具体取舍见 [MIGRATION.md](MIGRATION.md)。本轮阶段内存测试、WHT 与 F32 正确性测试都没有给出新吞吐结论。

#### 未覆盖与明确限制

没有证明全部模型、采样链、极长上下文或跨设备组合；CUDA、HIP、Metal、Windows、macOS 未取得本次运行资格。没有证明多 GPU / 五卡 TP5、RERoT 多 reader 共享 KV、FlashPrefill / XKV 整模型质量、设备隐藏层交接、自动显存 fit、压力抢占、跨 context 共池或随机 MTP 恢复。

Corpus 未提供 tokenizer 身份、并发录制、崩溃事务和多分片读取；比较器的文件格式也不携带完整构建 / 模型身份。逐文件静态审计中已识别但未运行的能力保留自己的证据等级，不能因本页中其他测试通过而自动获得资格。


## 2026-10-21 最新保留检查点：QSA full-coverage 回移

当前入口为 build-tp5-gap-repair-20261021/run-retained.py；它使用 qsa-full-coverage-install 的 A 与原 finite-greedy-compose-install 的 C。以下早期章节的旧入口/旧速率属于历史证据。本检查点仍未宣称追平 master，四组平衡 master 非劣门尚未通过。

固定 master 46a6fd8a129091741f9ebc6431f7f610abe1ccfc 已有全覆盖 QSA 快路径；当前 A 原先无条件计算无效排名。本次仅回移 full-coverage 逻辑，继续原始 indexer key 投影、F16 cache 写入和同一 QSA/F32 attention 路径，KV extent 变化使旧图失效；超过预算恢复原 selective 路径。未移植 pooled-cache、MTP 或 PLE API 迁移。另保留小型 relay idle-event 修正，避免已交付全部事件的 worker 空转抢 snapshot mutex；abort 优先可见。

完整 512 步、127139840 logits bitwise equal，error_count=0、nonfinite_count=0、native_drained=true、budgets_zero=true。原始权重、五卡、F32 activation、F16 KV、ctx/sequence=262144、teacher 和数值门均保持。一次相邻无录制 pair：32.7820521 → 35.5374418 tok/s，ratio=1.08405177（+8.405%）。这是 QSA+idle-event 组合的增量比较，不是 master 非劣证明，不能将组合收益精确分给单项。

真实五卡 A/B/A fixture 共 9 epochs，输入/选中终点/另一终点各 11565 项逐位检查通过；最后事件交付后的 idle abort 仍返回失败，已有五卡 drain 证明和安全销毁通过。QSA 跨预算切换已逐行分析：raw-key 写入在 early return 之前，旧 set_input_qsa 只写临时输入、没有持久状态推进，can_reuse 比较完整 n_kv。尚未另跑超过 2051 KV 的完整模型，不能把该边界表述为已实测。

本轮撤回：shared-merge 全512有2个数值越门；shared-pair 数值通过但32.5804→32.8259仅+0.7535%，不足以保留双段私有协议；native capture 并行虽全512逐位一致、每卡解除约380条边，却32.5159→27.6277（-15.03%），调度代码全部撤回。所有失败/撤回源码与 receipts 保存在各自前缀。一次混用旧 Meta/model 对象导致 ABI 初始化失败也保留，已改为内容恢复后刷新 mtime 重编；当前安装与当前源对应。

证据：qsa-full-coverage-{identity.json,full512/full-metrics.json,pair/summary.json,hazards}；设计 qsa-full-coverage-design.md；共享支路与调度撤回见 shared-pair-decision.json、native-parallel-decision.json。A 例外仍为原 51 路径，本次更新两个既有路径的 exact hash/理由/移除条件。

## 2026-10-21 当前保留状态：原 Q2/F32 gate/up 融合

在 QSA 全覆盖与 idle relay 基线上，复用上游三节点融合、外部使用者和别名检查，只补既有原编码权重/F32 路线的执行入口。48 个路由 FFN 每层由两次投影与 SwiGLU 三次 dispatch 合为一次，四链 FMA、实际 combine/XOR 树和原 SiLU 算术保持。共享专家因 gate/up 类型不同，未扩大覆盖。

真实五卡 30 个定向用例通过，35840 个有限值逐位检查、F64 oracle、padding、无效 ID、外部中间结果使用均通过；完整 512 行/127139840 logits 与原冻结参照逐位一致，native drain/预算门通过。单次相邻无逐算子录制比较为 **35.559735 → 36.241402 tok/s（+1.91696%）**。真实模型五卡融合计数在后端关闭时读取，并逐项证明读取时间不在 511 个 commit 间隔内。详见 `build-tp5-gap-repair-20261021/precise-glu-retained-checkpoint.json` 与 `precise-glu-pair/summary.json`。

当前 wrapper 已指向 `precise-glu-install`。历史固定 master 45.300541 tok/s；尚未达到非劣效目标。两项 attention 尝试均撤回：双输出 CTA 比例0.998978；独立分母 wave 比例1.002414，未提供足够保留收益。其数值通过和单次比较证据完整保留，没有追加调参网格。上游例外仍为51条，无新公共接口。
