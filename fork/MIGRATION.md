# 功能迁移与取舍

## 固定身份与原则

A 为 AtomicBot-ai/atomic-llama-cpp-turboquant 的 master，`cda0f9256665845cd9d6f3a0176880883f653a3d`，标记 `b10298-2.0.0`。旧 fork 为 `2777543b2358c88cd1c2000f7fa342b764a0e213`，由 `archive/pre-clean-20261018` 保留。旧分支比该 A 多 527 个提交；这些提交没有重新作为祖先进入干净分支。

完整路径差异为 4,634 项，其中 `artifacts/` 3,908 项、`reports/` 12 项，剩余 **714 个代码、工具、测试、文档和构建文件**。第一轮使用的缩减清单不是全库分母，先前“547 文件”的说法在本记录中撤回。逐文件范围、实际阅读深度和证据见 [AUDIT.md](AUDIT.md) 与 [audit/files.json](audit/files.json)。有记录不等于每行都做过语义证明。

迁移按具体行为作决定。一个大模块暂未迁入，不构成丢弃其小工具、数学函数、协议或内存策略的理由；发现旧实现错误，也不构成功能目的无价值的证明。无内容证据的项目保持待审或保留候选。

## 2026-10-19：最初的 TP5 后续文本审计

用户已重新实现 TP5，当前路线以 [relay](relay/README.md)、[relay-probe](relay-probe/README.md) 和各自运行记录为准。其余有价值的细节已整理为 [REWRITE-CANDIDATES.md](REWRITE-CANDIDATES.md)：33 组详细条目、20 条后端小项、10 条模型/调度补充，以及原论文与最低验收条件。

该轮审计的全部新增候选只作文本记录，没有迁入未证实工作的旧代码，也没有新增模型/设备资格。它另纠正在线 corpus 的事件时机、重复 uint64 saver、M29 状态恢复归因及普通输出容量的已有 A 能力；原总账和历史运行身份保持不变。后续实际实现按下面的独立更新记录，不能把文本审计的身份改成实现通过。

## 后续 RERoT 重写：已实现的边界与验证范围

用户明确要求的是插入点之间逐 token 可见的共享白板。当前重写由独立 [TQRerot](rerot/README.md)、A 的公开 reader / recurrent 接缝和 [rerot-compose](rerot-compose/README.md) 承接，没有直接迁入旧 `llama-rerot.*` 或 server 运行时。E11 / E23 的有用合同已进入新实现；旧自动 planner、reasoning-end / tool profile 等相邻功能仍保留文本候选。

本轮 hardening 的最终结果以 [新机器索引](audit/rerot-hardening-qualification-20261019.json) 和 [VALIDATION.md 顶部记录](VALIDATION.md) 绑定的源码、安装库与消费者身份为准。当前 A 为固定上游加 **34-path overlay**；本轮实现提交为 A `81276fa6c`、B `23e1460af`、C `e60d999cb`，完整提交与产物 hash 由新索引记录。下面的“最终”只指这些实际执行的候选，早期失败与数据保留原身份。

| 部分 | 已实现的职责 | 已有证据与剩余范围 |
| --- | --- | --- |
| B：DAG / episode / reader view | 独立 C++17 库；有界 pen、稳定 stage / run 身份、immutable public 前缀、own causal / private / source-end、checked arithmetic、陈旧票据拒绝、cohort 原子 commit / abort；BODY 必须 public | 本轮 B 随新源码隔离构建并由搬迁后的 C 消费。早期 Release、UBSan、ASan 各 4/4 CTest，以及四 worker DAG 的 543 个图、3,007 个合法状态、3,904 个 reader 和 F64 oracle 的 3,200 个 reader/head 比较、最大绝对误差 `7.77156e-16`，继续属于其原批次，不改称本轮全部重跑 |
| A：真实 reader attention | 一个 context / KV owner；公开 view 分开存储位置与虚拟位置。identity span 复用原生 batched KQ，其余 span 重建 reader Q；可见分数 scatter 到物理 KV 槽，隐藏槽置负无穷，作一个全局 softmax 后消费原生 batched V。本次 reader 仅支持 resident F32 K / F32 V；公开 reason API 说明实际拒绝条件 | 最终 CPU 与实际 offload 的 GPU 独立 F64 logits oracle 各 16 例 PASS，含 ordinary 校准。单层合法 GGUF、零 FFN、Llama NORMAL / Qwen2 NEOX、GQA、幅度及 query 顺序的范围不变；CPU 另有 7 项 unsupported checks，GPU 该次为 0。完整 Qwen2 NEOX fixture 不提供 A partial rotary / GPTNeoX 架构资格 |
| A：量化批形状数值策略 | 新公开 context 请求 `batch_invariant_matmul` 默认 false。C 从创建 context、共同 Prefill 之前请求 true；`GGML_PREC_BATCH_INVARIANT=20` 覆盖量化 dense 与 expert-indexed products，以同后端原生 DEFAULT N1 dot 为参考，可能仍用 Q8 activations；F32 PEDANTIC 是另一合同 | GPU dense 597 graphs / 90 bitwise comparisons / 18 bias graphs PASS；CPU 与 GPU ID 各 1,114 graphs / 431 bitwise comparisons PASS，参考为未设置候选 precision 的 native DEFAULT N1。Vulkan 的列与 selected slots 留在 GPU 批量执行，仅在硬件 grid 上限处分块；DEFAULT 保留大 N 优化矩阵路径。context getter 只报告请求，当前结果不扩为所有后端、模型、布局或 provider 的保证 |
| A：独立 recurrent 物理池 | 逻辑 sequence tails 与 `n_rs_slots` 物理容量分离，实际 COW / 容量预检由 A 拥有；公开 recurrent-only retirement 保留已封存阶段的 attention keys | C 使用 `W + 2` 个逻辑 ID、`P + 1` 个 recurrent 槽，其中一槽保存 immutable base。最终 CPU / GPU reader 各通过 31 个全 logits 等价组、2 个 peer 因果效应组及 12 个跨形状对照；两个 max_abs 均为 CPU 0、GPU `1.52587890625e-5`。CPU 结果来自新隔离安装，72 次 decode / 169 行；不借旧 clear/reuse 扩大资格 |
| C：真实生成与发布 | 显式 DAG、`assistant-eot` 原生模板、FRAME token waves、每阶段独立 native sampler / recurrent lineage、每 BODY token 同步后共同发布、阶段工作区释放；实际消费安装后的 A、TQRerot、TQRelay、TQText、TQSampling | 最终 CPU / GPU 各 8/8 完整 episode 通过合同、phase、重复与跨形状四项资格；四 worker / 两 pen、`ubatch=128,1 × release,retain × repeat=2`，原生 EOG 完成 worker 与 synthesis。CPU 每例 764 frontiers / 208 live peer reads，GPU 每例 748 / 208；严格门是各自后端内的对照，不要求两端轨迹相同 |
| C：提交、退出与设备边界 | 每次 submit 用局部 `run_submissions(call, 1)` 登记一个真实 A 域，在实际 `llama_synchronize` 后完成并释放票据；已接受调用先排空再清理 reader，失败状态跨调用保留。取消在同步 chunk / wave 边界及 B commit 前检查 | 原生 C CTest 5/5 PASS，其中 native cancellation 实际在 CPU 执行；真实 CPU / GPU 的 SIGINT 与 broken pipe 均通过，无 timeout / forced cleanup。复用 A 单设备 / layer / row placement 接线，多 GPU PP 尚未实跑；单卡 `layer` 与信号退出不能代替五卡 provider、GPU kernel 内取消或多设备 drain 证明 |
| A / B / C：源码与安装消费 | 固定 upstream archive 加 verified overlay 重新编译纯 CPU A；各库与 C 独立构建，随后删除 A/B/C 源码和 build 副本，再搬迁安装并消费 | 新 `isolation-expert/delivered` 未复用 `--a-prefix`：A seams 11/11、C 5/5 与搬迁后 4 例真实 RERoT PASS。隔离四例 receipt 只有合同和 phase 资格，repeat / cross-shape 仍为 `NOT_QUALIFIED`；最终严格八例有单独 receipt。源码隔离不等于操作系统沙箱 |

共享白板的发布单位是本轮每个运行阶段已经完成 forward 的一个 BODY token。所有 member 读相同的 peer revision，完成后一次 commit，下一轮立即看到 peer 刚提交的 token。刚采样但还没有计算 KV 的后继 token 不提前发布。多 token FRAME 以波次完成并保持同一冻结 peer 快照；这不延迟正常 BODY 的逐 token 共享。DAG 只控制阶段启动，public 可见性还包含已结束的非祖先阶段。

当前 C 是全驻留、无持久恢复的显式任务消费者。共同 base 只 Prefill 一次，新 worker 从该 base 建立 lineage，synthesis 在全部 worker 结束后独立启动；private source-end 不进入 peer 的白板。失败可能已经写过原生 microbatch，C 排空后终止 episode，没有发布半个 cohort，也没有声称可原地重试。逻辑 reader 容量、物理 KV、逻辑 sequence ID 与 recurrent 槽位分别校验。旧协议和更快的设备路径只有满足同一 token frontier 合同后才能另行接入。

数值修复针对已经找到的形状边界：图中 F32 dense/router 与 attention products 使用显式 PEDANTIC；MoE 保持相同 expert 求和顺序及实际 expert 数；CPU mode20 的量化 dense 路径绕过两次 SGEMM 并固定 single-output dot，转换与 chunk 并行保留。当前 x86 结果不扩大到其他 CPU 架构、extra-compute provider 或 MoE cache。GPU dense / ID 的点积参考、padding、融合 bias/scale 与硬件 Y 边界分别有公开算子和实际 DEBUG 分派证据；没有把请求配置当作 kernel 命中证明。

正常 Release 的 resident graph 测量也已完成。IQ4_NL / Q8_0 / Q4_K、K=2048、M=1025 的 invariant/default warm median 比值范围，在 N9 为 `0.469399–0.499107`，N32 为 `1.092569–1.311400`，N128 为 `1.857106–2.019109`。这记录了显式策略在较大 N 的成本，A 默认优化矩阵路径继续保留。MMVQ 的 auto/disable ABBA 对照在 Q4_K N8 观察到较低用时，Q4_0 N8 相近，部分小 N 与 IQ4_NL 负控制有主机或时钟波动；不推导普遍提速或完整 RERoT 吞吐收益。形状、cold/warm、控制顺序、samples 与产物身份集中见 [VALIDATION.md](VALIDATION.md) 和新索引，DEBUG 时长不作性能数值。

早期 8 例 CPU oracle 的 `7.72386e-7`、原 GPU ordinary 校准失败及 reader 同形状 `0.0000133514404296875`，仍归属 [旧索引](audit/rerot-qualification-20261019.json)。本轮先后失败的跨形状、F32 head-padding、dense-only Prefill 与 expert-ID 候选也保留原回执；最终 PASS 来自实际修复和原有严格门，没有放宽 numeric/greedy 阈值。F16/mixed 或量化 KV、FA reader、SWA、partial RoPE、GPTNeoX 架构、CUDA/HIP RERoT、多模型质量、HTTP 集成、自动 planner、reasoning/tool、whole-model TP5、LateBind、MTP 与多 GPU PP 继续各自验收；当前 F32 / 单机结果不补齐这些边界。

## 已迁入并有真实消费者的能力

表中的此前 CPU / Vulkan、HTTP、corpus 与 logits 数值沿用各自原运行身份；没有因新增库而把旧结果记为本轮重跑。2026-10-18 细节迁移批次已完成五个 B、五个 C 的 CPU archive/isolation、两个新增 B 的 sanitizer 测试和校准格式互操作；最终 Vulkan 同形状流式模型对照与 scratch / 输入别名专项回归也已通过；完整算子门的失败记录另列，不合并成全后端通过。具体命令与范围见 [VALIDATION.md](VALIDATION.md)。

| 行为 | 当前实现 | 保留的合同与证据 |
|---|---|---|
| 精确 compact top-k | B `TQSampling`；C `compose` 与 `rerot-compose` | B 拥有数值结果；C 用真实 A 采样链，保留 native sampler 的 accept / RNG 状态。本轮 RERoT 的 compact shortlist 与完整输入回退消费原生链，配对回归见新索引；此前 CPU / Vulkan 的 greedy、正温 token 对照保留原运行身份。 |
| 请求 token trie | B `TQCorpus`；C `corpus-compose` | 共享前缀、重复权重、请求顺序和前缀终点；真实 prefix / full replay 的逐请求 NLL 对照通过。 |
| 旧生产 corpus 小合同 | 同上，另有归档 writer / reader 互操作测试 | nodes ID 从 1 开始；kind、64 位 timestamp、空请求和缺失空流正确处理；真实旧 writer → 新 B → 旧 reader 字节往返通过。 |
| Prefill 优先 | 独立 B `TQScheduling`；C `batch-compose` | 保持当前提示、按调用者提供的候选顺序选择；B 不隐含 FIFO。C 另行拥有 FIFO 准入。有待处理已准入提示时暂停 Decode，完成后恢复。 |
| 有界 token / 槽位准入 | B `admission_budget`；C 真实使用 | 完整需求预留、token 与驻留槽位双约束；domain / generation 防止陈旧或外来释放，失败保持账目。C 在真实序列删除成功后才回收预算。 |
| 队列与序列槽位复用 | C `batch-compose --parallel` | 请求 ID 与 A 的 seq ID 分离；每后端七个模型运行、60 请求，相同配置开关对照、同槽位重复 canary 通过。跨批形状差异另存诊断。 |
| 逻辑窗口与物理 KV 分离 | A 的 `n_ctx_kv` / `llama_n_ctx_kv`；C 配置与使用 | 复用已有统一 KV 分配器，只将物理容量传入 attention KV 构造；不修改逻辑窗口 / RoPE 或 recurrent 容量。真实 logical512 / physical1024 通过。 |
| Prefill / Decode 计算工作区 | A 的同步释放 / 当前字节观察 API；C 显式调用 | 同步、失效可重建图、释放 galloc，保留模型 / KV / recurrent / 输出；下一批按实际形状重建。CPU / Vulkan 两臂输出、位置、token 相同，分配器字节确实归零。 |
| 原版 server 的 Prefill / phase 开关 | 五个 common / server 文件的有限协作补丁 | 保留实际 HTTP 入口；prefill 默认 on、phase memory 默认 off，错误组合加载前拒绝。三路 SSE、插入 Prefill、25 层卸载与 KV1536 实测通过。 |
| 普通 server task 默认槽位 | `server-task.h` 一行 | `id_slot=-1`，普通 task 移动不读取未初始化整数。与原版 server 请求路径共同验证。 |
| CPU 显式 F32 attention | `ggml-cpu/ops.cpp` 小修复 | F16 V 也尊重请求的 F32 累加。真实公开操作用例在原 A 失败，修复后通过。 |
| Vulkan 公开 Turbo WHT | host / shader 两个文件 | 既有 group32 / 64 / 128、正反 scale 与实际 grid；原 A 12 例失败 10 例，修复后 12 例全过。 |
| Vulkan scratch 身份与顺序 | A 现有后端内的 12 行修复；公开 GGML 回归 | FA bitmap / large Softmax 写入后清除 activation 身份；no-split bitmap 读取后登记同步需要。最终 5 场景通过，独立与组合全量比较，binary-step no-split 的 524288 个闭式值一致。 |
| Vulkan 输入原地修改 | 同一 A 文件的 17 行修复；公开 GGML 回归 | 在算子记录后检查全部 fused writes，复用已有范围判断并保守处理未知 buffer 类型；4 字节 view offset 的最后 MM 从 64/64 错误变为全部一致，不重叠写入控制通过。 |
| Meta 异步非零偏移 | A 一个文件 +4/-4；公开 GGML 回归 | 从逻辑 chunk 起点换算子张量偏移，axes0/1/2 与 mirrored；原零偏移16控制通过，非零 get/set 分别断言，修复后48真实双 CPU child 用例通过。 |
| recurrent 清除 / 共享 / 复用 | 直接使用当前 A；新增公开接口资格测试 | 当前 A 已整 buffer 清零与图内延迟初始化。CPU/Vulkan 在 LFM2.5 hybrid convolution/KV、n_rs_seq=0 上均通过24邻居、8共享快照比较和同形状恢复；不携带旧 eager-clear 补丁，不承诺损坏状态恢复的事务性。 |
| 公开 logits 采集 | C `logits-compose` | 不采样的真实 token 输入、TP5LOG2 全词表行、有界写入、完整后无覆盖发布；CPU/Vulkan 重复 tape 按位一致，同标量形状行选择的全词表比较通过；独立 C 只对照相同提示输入、最终 greedy token / 最大 logit（1e-5），没有第二份全词表参考。 |
| 独立 logits 比较器 | `fork/tools/compare-logits.py` | 保留 TP5LOG2 文件兼容，stdlib、逐行有界读取、token 身份和完整性校验，数值 / bitwise / argmax 分开；20 个真实 CLI 测试通过。 |
| 增量 UTF-8 / stop | B `TQText`；C `batch-compose --stream --stop` | B 拥有跨块状态和文本结果；C 保留原 token 记账，在匹配确定后停止后续 decode、发出一次终止事件并回收请求。重建用途，不声称旧非流式 fastpath 或吞吐收益。 |
| WXQCAL03 校准格式 | B `TQCalibration` | 显式 LE、逐层有界读取、统计人口与有限值校验、精确 EOF、拥有自身数据的结果。B 不调用 A；与 TQText 的独立测试均通过 ASan / UBSan。 |
| 校准统计导出 imatrix | C `imatrix-compose`，使用已安装 B 和 A `ggml::ggml-base` | raw sum2 / exact F32 counts、显式 partial / unseen 策略、元数据记录估计与范围、无覆盖发布。原 writer fixture 经新 C 产生 12 tensors，实际 A loader 读为 6 entries / 9 chunks；不证明模型量化质量。 |
| 源码与安装包边界 | build、manifest guard、isolation、独立 CI | 2026-10-18 的五个 B、五个 C 及 CPU archive/isolation 62 条命令保持原身份。本次新隔离 receipt 为 77 条命令、7 B / 6 C；不包含 relay-probe，a-seams 另作安装消费者检查，不是全仓程序分母。从固定上游加 34-path overlay 新编译纯 CPU A，完成源码删除、安装搬迁和真实 RERoT 消费，未复用 `--a-prefix`。新 receipt 的四例合同/phase 与最终八例严格生成范围分开；A 内容以 manifest 为准，CI 尚待推送后实际执行。 |

当前 A 例外的精确文件、内容、消费者与退出条件见 [upstream-exceptions.json](upstream-exceptions.json)。原版 HTTP server 的私有调度循环仍属于 A 例外，不能把它包装成“已经独立的 B”。当前七个 B 均只依赖 C++17 标准库，各自能独立构建和消费；C 显式组合它们与 A。TQRerot 不包含 A 或 TQRelay 依赖，跨库协作发生在 C。

### 文本与校准：关闭的是具体子项

`server:Q14` 原来是非流式、无 stop 时跳过逐 token 扫描的 fastpath。现在的 TQText / batch C 定义了切块不变的 UTF-8 修复、stop 选择、字节游标和输出失败合同；没有把旧性能目的变成未经测量的收益。`server:T09` 只有跨块终止符这一机制可作为参考，完整 thinking / lane parser、Q15/Q16/Q17 的首块与 usage、HTTP 取消和重试仍需独立迁移。用法见 [TQText](text/README.md) 和 [batch C](batch-compose/README.md)。

`small:S04` 的格式读取和 imatrix 导出已迁，`small:S05` 的模型 callback、DFS 采集人口、周期 checkpoint 与 partial-save 尚未迁。默认导出要求 `complete`、正 total 和 `decoded == total`，拒绝未见专家；显式 `--allow-partial` 保留全部范围原因，`--unseen global-mean` 记录估计行及专家 ID。旧 writer 最后一批可能 overshoot total，新实现不截断统计或伪造完整标志。有限值、精确 F32 计数和 A 归一化后非零值保留仍是硬约束，选项不绕过它们。

2026-10-18 的互操作使用与 `2777543b2358c88cd1c2000f7fa342b764a0e213` 完全相同的旧 writer `.cpp/.h` 生成生产格式 fixture。实际 A quantizer 完成 loader 消费后，在故意缺失模型的阶段以 exit 1 结束；该结果不等于完整模型量化成功，也不是历史模型校准数据的质量验证。转换不需要 NumPy 或推理 context，完整边界见 [TQCalibration](calibration/README.md)、[imatrix C](imatrix-compose/README.md) 和 [VALIDATION.md](VALIDATION.md)。

### 内存能力必须分开看

| 子功能 | 当前状态 |
|---|---|
| 同一 context 中统一 KV / sequence 存储 | 上游 A 已有，直接复用 |
| 物理 attention KV token 容量独立于逻辑窗口 | 已补回，支持范围有显式检查 |
| Prefill / Decode 计算工作区释放与重建 | 已补回单 context 生命周期 |
| Prefill 优先任务选择 | 已补回 B、C 和原版 server 入口 |
| Prefill 最大并发 / 初始 reserve 单序列图 | 独立候选，未因已有优先策略就宣称完成 |
| 基于设备真实容量的 auto-fit / slack / 预算 | 保留目的，旧 solver 不照搬 |
| recurrent 物理槽位、COW / 容量预检 | E11 已由 A 的 `n_rs_slots` 与逻辑 tails 分离实现；RERoT C 请求 base 加 P 个驻留槽，已有 LFM2.5 F32 reader 状态门，其他模型/配置仍须单独验证。`n_ctx_kv` 参数本身不改变 recurrent 容量 |
| 已知 token 需求与驻留请求数的 admission | 已实现独立 B；C 有真实压力队列与槽位复用 |
| 压力下 idle KV LRU、抢占与恢复 | 仍是独立候选；当前 C 不自动驱逐或重试 |
| target / draft 跨 context 共享 allocator | 待明确共享所有权和同步协议，当前未实现 |

`--total-kv` 当前接受非负整数，非零启用 unified；不接受旧 `auto` 语法。C 根据已知 token 需求选择默认值，不是显存探测。`llama_memory_compute_bytes()` 只报告计算分配器拥有的字节；没有把小测试的工作区缩小写成整机峰值显存下降。

## 有价值、尚未迁入的小能力

下表是实际内容审查后的候选。它们不是“因性能低而删除”；每项需要自己的真实消费者、边界和必要验证。具体代码位置及未读范围见各 [审计分册](AUDIT.md)。

| 组别 | 应保留的独立行为 | 下一步边界 / 验证 |
|---|---|---|
| 容量策略 | 设备维度 bounded fit、显存余量、PP reserve 限制；已知 token 预算已迁 | B 只算拥有值的预算 / 探测计划；C 拥有真实 A context 并执行探测；不能拿 metadata-only 假分配当峰值测量 |
| 槽位与池 | idle LRU、cache-key 亲和、payload 身份；预算 reservation 的 domain / generation 已迁，RERoT recurrent 物理槽已实现 | 继续定义一般缓存的共享引用、回收、实际占用与恢复；新 recurrent 池按真实 hybrid 场景取得自己的资格。旧 `llama-xkv-hot` 的元数据责任可独立提取，不等于接管 A 的 tensor |
| recurrent 清零 | M33 已重新判断：复用当前 A，不迁旧 eager clear | 历史907–935 ms →12.8–16.8 ms减少的是旧 fork 自己加入的清零。当前 A 已有整 buffer / lazy init；加入旧路径会增加写与 fence。公开生命周期测试通过；HTTP取消、GDN / MTP仍需自己的验证 |
| 重试与输出 | 已输出文本游标、预算 / usage 聚合、首块语义、重试恢复 | C 拥有 HTTP 请求与 sampler；必须验证随机状态、取消和部分输出，不能只比较最后 token 总数 |
| 请求 / 文本协议 | cache affinity、按 lane 拼完整行、thinking parser、usage 只计算一次 prompt | UTF-8 / stop 用途已由 TQText 和 batch C 重建；完整多 choice / SSE / 取消仍由真实 HTTP C 验收，旧非流式 fastpath 的性能需另测 |
| 在线 corpus 录制 | 入队前记录模板/tokenization 后的请求输入、种类与时间戳，供校准与回放；不是请求成功结束后的生成输出 | B 的生产格式已经恢复；在线事件身份、失败/重试和重启完整性仍需 C。离线构建不等于旧 server 钩子，见 [E04](REWRITE-CANDIDATES.md#e04--在线输入-corpus事件身份与重启完整性) |
| 退出与失败 | signal 自管道、正常线程清理、第二信号、LoRA 失败传播 | RERoT C 的真实 CPU / GPU SIGINT 与 broken pipe、实际 CPU native cancellation 已取得本轮局部资格；第二信号、HTTP 队列及 LoRA 失败传播等仍须各自验收，不能以进程正常退出代替全部生命周期证明 |
| grammar / sampler | grammar 替换、clone 后结果所有权、实例级 Turbo 配置、额外 RNG draw 语义 | RERoT 的 compact / full 路径已由 C 消费公开 A native sampler 并验证配对行为；相邻 grammar 替换、通用 clone 所有权与实例配置仍保留各自合同，不能以一组采样链对照关闭所有候选 |
| 模型 / GGUF | BailingMoE3 名称及缺省兼容、缺失 MTP 权重、uint64 PLE hash | 上游实际消费者和 GGUF 元数据校验；每个兼容点独立提交，不整包恢复所有图修改 |
| 校准采集与复用 | small:S05 的公共回调激活采集、DFS 统计人口、trie 复用、周期 checkpoint 和异常保存 | small:S04 格式读取 / imatrix 导出已迁；采集 C 仍须验证实际模型 callback 的名称、shape、dtype、分支顺序与计数，不以转换器代替采集 |
| 独立证据工具 | GGUF 头读取、离线性能报告、请求 / 构建身份 | 比较器与公开采集器已迁；格式仍无完整模型 / 构建身份，必须保存旁侧运行证据。其他工具需各自验收统计分母、有限值和失败输出 |
| 工作区计划 | 64 KiB 容量向上取整、阶段 scratch 最大值、间接 dispatch 算术 | 纯值 B；容量取整不能改变 dispatch 几何；实际 buffer 与设备生命周期由 C / A 拥有 |
| 状态封装 | 有界 state envelope、scope / policy / adapter / nonce 校验 | B 可封装字节，C 包装 A 的 opaque state；需明确端序，同进程身份不能冒充持久跨进程模型身份 |
| 数学参考与 reader 精度 | 无符号 Hadamard、其他 GQA / DDVR 参考与高效 reader 路径；RERoT 可见性和 F32 reader 相位已有独立 B 与实际 A 消费者 | 本轮最终 F32 CPU / GPU oracle、同形状及跨形状 reader 门按上节范围通过。早期 F16 reader 的 CPU 重排/restore `0.11799526`、Vulkan permutation `0.0003576` 未过严格门，仍保留失败，当前不发布 F16/mixed reader。A partial rotary、Turbo KV 与其他设备组合仍单独验证 |
| 稀疏 / 合并参考 | mean / energy 选择、online softmax 三元组归并、有限 mask 保留 | 单独证明数学合同和异常回退；局部参考不自动获得稀疏 attention 整模型质量资格 |
| Vulkan pinned host 所有权（G14） | 使用 buft 对应设备分配、释放与查询，而不是硬编码 device 0 | 保留独立候选；alloc/free/getter 必须由同一设备 owner 闭合，不能借 G09 或已有 WHT 回归代替 |
| 后端算子 / 生命周期 | SCALE F16、MROPE / IMROPE 融合、字节 memmove、readback pin / 可见性、device staging 清理；meta 分片非零偏移已迁 | 公开既有算子可小补丁；overlap、stream / fence 和多 context 生命周期需对应证明 |
| CPU / CUDA 小优化 | 既有 Q2_0 VNNI、Turbo 小查询优先 native VEC | 与私有 PQ2_0 / TP5 分开评估；本轮没有 CUDA 或该 CPU kernel 的性能测量 |
| 评测与参考 | 性能预算、评测 split 与其他协议 fixtures；新 B 的四节点 DAG / reader 顺序已有穷举 oracle | 保留测试目的，修正错答案、泄漏分组、NaN 假通过和提前 PASS 后才能作证据；数学 oracle 不代替真实任务质量或性能 |

### 已检查、但当前 A 不具备旧优化前提

Meta graph-view cache-key（FBE15）的前提是旧 fork 让 graph view 继承父图 UID。当前 A 会保存并复用非零图 UID，但其 `ggml_graph_view()` 将 view UID 设为零，view 因而触发重建。仅搬旧 view key 修复不能改善这个路径；未来若引入 view UID 复用，必须同时定义父图 / 切片身份与生命周期。旧 pinned-readback retire（FBE12）依赖的 pending transaction 在当前 Meta 路径也不存在。两者保留功能目的，未来要将新 fast path、所有权、验证和收益一并设计；不孤立搬入旧修补语句。具体依据见 [meta-transfers-protocol.md](tests/a-seams/meta-transfers-protocol.md)。

### 重新核对后保留 A 的支持范围

G44 原候选建议 Turbo SET_ROWS 拒绝 F16 source，前提是只编译 F32 shader。固定 A 的 `vulkan-shaders-gen.cpp` 实际同时生成 F32/F16 source 到 Turbo 输出的 I32/I64 索引变体，因此不迁旧拒绝条件。这是纠正先前静态判断的前提，没有改换 A 基线，也不把其已有能力算成新 fork 功能；相邻 G19 日志不随它迁入。本轮另加入 16 项完整 packed-byte 回归验证 F32/F16 和 I32/I64 的组合，保留未写入的 guard 行。G09 的共享 scratch 身份与新发现的输入别名修改按各自的反例与修复回执验收。

714 文件的登记不代表全部语义已证明，也不代表所有候选已迁完。2026-10-18 的结果关闭文本合同、校准格式与导出、当时的 CPU 独立交付，以及有明确反例的 Vulkan scratch / 输入别名子项；同形状 CPU / Vulkan streaming 单独验证。后续 RERoT 重写关闭的实现范围与已完成 oracle 见本文新增章节。此前准入、logits 采集、偏移和 recurrent 生命周期仍保留各自原证据；每批最终资格以 [VALIDATION.md](VALIDATION.md) 为准。

## 舍弃的具体实验补丁

仅下列已有收益依据的实验组合在本次选择中明确不迁。数据来自归档 `PERFORMANCE.md`、`MMVQ+MTP.md`，不是本轮重新测量，不能为其他小改动批量背书。

| 实验 | 历史证据 | 决定 |
|---|---|---|
| router row split | 同一 pilot 66.6914 → 61.8741 tok/s，约 -7.22%，未晋级 | 不携带该实验补丁 |
| BF16 router | 8 配对块 / 16 进程，约 +1.14%，原记95% CI [0.79%,1.48%]，低于原2%门槛 | 不为它保留跨模型 / 后端耦合 |
| MMVQ + MTP 联合实验选择器 | 131 配置，最终候选等于原基线；C/R=1.034315，原记95% CI [0.996803,1.073239] 包含1 | 不携带该联合选择器、强制模式和控制复杂度；不否定整个 MMVQ / MTP 家族 |

旧诊断工具不再一概归为“无用并舍弃”。比较器、公开 logits 采集和校准转换已按各自范围交付，离线报告、真实校准采集和小参考实现继续保留候选。硬件档案、历史报告与实验输出保留在旧分支，作为来源证据，不进默认构建。

## 不复制的旧错误实现

静态审计指出的合并风险按具体语句和状态边界记录，而不是宣称都已在运行中复现：共享 allocator 初始化随后被普通初始化覆盖；默认 KEEP_CHUNKS 跳过 trim；KV cells 特殊成员遗漏新增身份字段；Turbo K shift 新分支落在已有 `continue` 后；recurrent / KV 失败预检仍有部分先行修改；某些 Qwen4 MTP 图路径重复归一化、RoPE 或 gather。

测试和脚本也有可见失效条件：并发容量脚本实际串行且并发代码不可达、写 PASS 早于最后检查；部分数值比较让 NaN 被 `std::max` 吞掉；新增 CHECK 在最后失败汇总之后执行；小样本 t 临界值错误；评测算术答案错误或 dev / blind 家族交叉。详见审计分册。当前验收采用直接 A 操作、真实输入行、真实设备日志和严格有限值检查。

五个旧 reader 问题另有实际纯函数反例，包括缺键 / null 漏报、非有限参考答案误判、LP 容差伪下界、坏 SSE JSON 未纳入 error，以及多行 PID 只读首行。输入、输出与来源摘要见 [audit/reader-counterexamples-20261018.json](audit/reader-counterexamples-20261018.json)，不代表运行了旧模型或硬件恢复脚本。

这些结论支持“重新实现有用合同”，不支持“所有旧功能都无效”。未跑过的旧路径不写成已复现崩溃或已测性能下降。

## 较大能力的状态

| 能力 | 当前判断 | 重新进入的条件 |
|---|---|---|
| TP5 / LateBind / relay / device loop | 已进入用户重写路线：独立 [TQRelay](relay/README.md) 与 [relay-probe](relay-probe/README.md)。旧私有 runtime 保留归档，不再作为待迁移项；59.66 →65.17 tok/s 仍仅属旧运行身份 | 平台探针已经落地。单设备协议、五 rank 索引 fixture、真实五设备 probe 和整模型五卡负载分别记录资格；以当前模块文档与 [VALIDATION.md](VALIDATION.md) 为准，不用历史收益或局部 probe 替代新路线的完整运行证据 |
| RERoT | 已重写独立 B、原生 A reader / recurrent 接缝和显式 DAG 的 `assistant-eot` C；一 context / owner、逐 BODY token 共享、全局 softmax、独立 recurrent / RNG 与原子发布有真实消费者；本次 reader 限定 F32 K/V。C 在 Prefill 前请求量化 dense / ID 的批形状不变策略，A 默认 false | 最终 CPU / GPU 各 16 例 F64 oracle、33 组 + 12 个跨形状 reader 对照、8 例完整 C 严格矩阵通过；固定上游加 34-path overlay 的新纯 CPU 隔离交付和搬迁消费通过，四例隔离 receipt 不升级为重复/跨形状资格。当前范围、Release 局部成本与原始负例见新索引；多 GPU PP、F16 reader、自动 planner、reasoning/tool、HTTP、LateBind、MTP、整模型 TP5 provider、Turbo KV 仍单列 |
| FlashPrefill / TriAttention | 独立数学、格式和状态封装有价值；历史 online value accumulation 的局部 prefill 约 +31%，但 greedy 输出变化、部分检索失败，整体验收未闭合 | 固定模型 / 上下文 / 质量门，公开消费者和最小接缝，避免把局部 kernel 测试当整模型证明 |
| XKV-SR | 槽位元数据、codec、预算和验证可拆；完整量化存储 / 图 / 事务运行时暂未迁 | payload、实际驻留、质量、吞吐与共享状态失败回滚同时验收 |
| DFlash / DSpark direct-hidden | 旧记录有内部交接与 probe；15–40%是目标，缺乏本次可用的独立整模型配对证明 | 真正 target hidden → draft 设备消费、数值与生命周期验证，再测相对合格基线的收益 |

精确文档差异还确认，speculative aliases、chained / adaptive MTP、默认 draft 深度和 draft CPU / KV 参数已经属于 A；本 fork 在对应文档新增的是 direct-hidden 段。已有能力直接复用，不再冒充自创 B 重写。

上游已有模型、量化、后端、server、bench 和 speculative 继续由 A 提供。当前 clean 分支不宣称与旧完整 fork 功能等价，但有价值的小能力都保留独立入口或可追溯的迁移候选，不按大模块名称抹去。

## 回查

```sh
git show archive/pre-clean-20261018:PERFORMANCE.md
git show archive/pre-clean-20261018:wanxiangqi/server/server-corpus-dump.cpp
git diff cda0f9256665845cd9d6f3a0176880883f653a3d \
  archive/pre-clean-20261018 -- src/llama-kv-cache.cpp
```

原 master、旧实现、用户未跟踪实验资料和模型均保留。当前 clean 分支的每一项迁入、候选或拒绝都能回到固定原提交与实际测试范围；没有删除旧分支来掩盖尚未迁移的工作。
