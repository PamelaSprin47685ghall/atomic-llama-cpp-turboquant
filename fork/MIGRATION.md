# 功能迁移与取舍

## 固定身份与原则

A 为 AtomicBot-ai/atomic-llama-cpp-turboquant 的 master，`cda0f9256665845cd9d6f3a0176880883f653a3d`，标记 `b10298-2.0.0`。旧 fork 为 `2777543b2358c88cd1c2000f7fa342b764a0e213`，由 `archive/pre-clean-20261018` 保留。旧分支比该 A 多 527 个提交；这些提交没有重新作为祖先进入干净分支。

完整路径差异为 4,634 项，其中 `artifacts/` 3,908 项、`reports/` 12 项，剩余 **714 个代码、工具、测试、文档和构建文件**。第一轮使用的缩减清单不是全库分母，先前“547 文件”的说法在本记录中撤回。逐文件范围、实际阅读深度和证据见 [AUDIT.md](AUDIT.md) 与 [audit/files.json](audit/files.json)。有记录不等于每行都做过语义证明。

迁移按具体行为作决定。一个大模块暂未迁入，不构成丢弃其小工具、数学函数、协议或内存策略的理由；发现旧实现错误，也不构成功能目的无价值的证明。无内容证据的项目保持待审或保留候选。

## 2026-10-19：TP5 重写后的文本候选

用户已重新实现 TP5，当前路线以 [relay](relay/README.md)、[relay-probe](relay-probe/README.md) 和各自运行记录为准。其余有价值的细节已整理为 [REWRITE-CANDIDATES.md](REWRITE-CANDIDATES.md)：33 组详细条目、20 条后端小项、10 条模型/调度补充，以及原论文与最低验收条件。

本次所有新增候选均只作文本记录，不迁入未证实工作的旧代码，不新增模型/设备资格。新文档另纠正在线 corpus 的事件时机、重复 uint64 saver、M29 状态恢复归因及普通输出容量的已有 A 能力；原总账和历史运行身份保持不变。

## 已迁入并有真实消费者的能力

表中的此前 CPU / Vulkan、HTTP、corpus 与 logits 数值沿用各自原运行身份；没有因新增库而把旧结果记为本轮重跑。2026-10-18 细节迁移批次已完成五个 B、五个 C 的 CPU archive/isolation、两个新增 B 的 sanitizer 测试和校准格式互操作；最终 Vulkan 同形状流式模型对照与 scratch / 输入别名专项回归也已通过；完整算子门的失败记录另列，不合并成全后端通过。具体命令与范围见 [VALIDATION.md](VALIDATION.md)。

| 行为 | 当前实现 | 保留的合同与证据 |
|---|---|---|
| 精确 compact top-k | B `TQSampling`；C `compose` | B 拥有数值结果；C 用真实 A 采样链，保留 distribution sampler 的状态。此前 CPU / Vulkan 的 greedy、正温 token 对照通过。 |
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
| 源码与安装包边界 | build、manifest guard、isolation、独立 CI | 五个 B、五个 C 分别构建并消费安装包；本轮完整 CPU archive/isolation 62 条命令通过，包含删除源码与前缀搬迁后的消费。A 内容以 manifest 为准；CI 尚待推送后实际执行。 |

当前 A 例外的精确文件、内容、消费者与退出条件见 [upstream-exceptions.json](upstream-exceptions.json)。原版 HTTP server 的私有调度循环仍属于 A 例外，不能把它包装成“已经独立的 B”。五个 B 均只依赖 C++17 标准库，各自能独立构建和消费；C 显式组合它们与 A。

### 文本与校准：关闭的是具体子项

`server:Q14` 原来是非流式、无 stop 时跳过逐 token 扫描的 fastpath。现在的 TQText / batch C 定义了切块不变的 UTF-8 修复、stop 选择、字节游标和输出失败合同；没有把旧性能目的变成未经测量的收益。`server:T09` 只有跨块终止符这一机制可作为参考，完整 thinking / lane parser、Q15/Q16/Q17 的首块与 usage、HTTP 取消和重试仍需独立迁移。用法见 [TQText](text/README.md) 和 [batch C](batch-compose/README.md)。

`small:S04` 的格式读取和 imatrix 导出已迁，`small:S05` 的模型 callback、DFS 采集人口、周期 checkpoint 与 partial-save 尚未迁。默认导出要求 `complete`、正 total 和 `decoded == total`，拒绝未见专家；显式 `--allow-partial` 保留全部范围原因，`--unseen global-mean` 记录估计行及专家 ID。旧 writer 最后一批可能 overshoot total，新实现不截断统计或伪造完整标志。有限值、精确 F32 计数和 A 归一化后非零值保留仍是硬约束，选项不绕过它们。

本轮互操作使用与 `2777543b2358c88cd1c2000f7fa342b764a0e213` 完全相同的旧 writer `.cpp/.h` 生成生产格式 fixture。实际 A quantizer 完成 loader 消费后，在故意缺失模型的阶段以 exit 1 结束；该结果不等于完整模型量化成功，也不是历史模型校准数据的质量验证。转换不需要 NumPy 或推理 context，完整边界见 [TQCalibration](calibration/README.md)、[imatrix C](imatrix-compose/README.md) 和 [VALIDATION.md](VALIDATION.md)。

### 内存能力必须分开看

| 子功能 | 当前状态 |
|---|---|
| 同一 context 中统一 KV / sequence 存储 | 上游 A 已有，直接复用 |
| 物理 attention KV token 容量独立于逻辑窗口 | 已补回，支持范围有显式检查 |
| Prefill / Decode 计算工作区释放与重建 | 已补回单 context 生命周期 |
| Prefill 优先任务选择 | 已补回 B、C 和原版 server 入口 |
| Prefill 最大并发 / 初始 reserve 单序列图 | 独立候选，未因已有优先策略就宣称完成 |
| 基于设备真实容量的 auto-fit / slack / 预算 | 保留目的，旧 solver 不照搬 |
| recurrent 物理槽位、COW / 事务预检 | 独立候选，当前物理 KV 参数不改变它 |
| 已知 token 需求与驻留请求数的 admission | 已实现独立 B；C 有真实压力队列与槽位复用 |
| 压力下 idle KV LRU、抢占与恢复 | 仍是独立候选；当前 C 不自动驱逐或重试 |
| target / draft 跨 context 共享 allocator | 待明确共享所有权和同步协议，当前未实现 |

`--total-kv` 当前接受非负整数，非零启用 unified；不接受旧 `auto` 语法。C 根据已知 token 需求选择默认值，不是显存探测。`llama_memory_compute_bytes()` 只报告计算分配器拥有的字节；没有把小测试的工作区缩小写成整机峰值显存下降。

## 有价值、尚未迁入的小能力

下表是实际内容审查后的候选。它们不是“因性能低而删除”；每项需要自己的真实消费者、边界和必要验证。具体代码位置及未读范围见各 [审计分册](AUDIT.md)。

| 组别 | 应保留的独立行为 | 下一步边界 / 验证 |
|---|---|---|
| 容量策略 | 设备维度 bounded fit、显存余量、PP reserve 限制；已知 token 预算已迁 | B 只算拥有值的预算 / 探测计划；C 拥有真实 A context 并执行探测；不能拿 metadata-only 假分配当峰值测量 |
| 槽位与池 | idle LRU、cache-key 亲和、物理 recurrent 槽、payload 身份；预算 reservation 的 domain / generation 已迁 | 先定义分配、释放、共享引用与失败回滚；旧 `llama-xkv-hot` 的槽位元数据责任可独立提取，不等于接管 A 的 tensor |
| recurrent 清零 | M33 已重新判断：复用当前 A，不迁旧 eager clear | 历史907–935 ms →12.8–16.8 ms减少的是旧 fork 自己加入的清零。当前 A 已有整 buffer / lazy init；加入旧路径会增加写与 fence。公开生命周期测试通过；HTTP取消、GDN / MTP仍需自己的验证 |
| 重试与输出 | 已输出文本游标、预算 / usage 聚合、首块语义、重试恢复 | C 拥有 HTTP 请求与 sampler；必须验证随机状态、取消和部分输出，不能只比较最后 token 总数 |
| 请求 / 文本协议 | cache affinity、按 lane 拼完整行、thinking parser、usage 只计算一次 prompt | UTF-8 / stop 用途已由 TQText 和 batch C 重建；完整多 choice / SSE / 取消仍由真实 HTTP C 验收，旧非流式 fastpath 的性能需另测 |
| 在线 corpus 录制 | 入队前记录模板/tokenization 后的请求输入、种类与时间戳，供校准与回放；不是请求成功结束后的生成输出 | B 的生产格式已经恢复；在线事件身份、失败/重试和重启完整性仍需 C。离线构建不等于旧 server 钩子，见 [E04](REWRITE-CANDIDATES.md#e04--在线输入-corpus事件身份与重启完整性) |
| 退出与失败 | signal 自管道、正常线程清理、第二信号、LoRA 失败传播 | C 生命周期职责；错误必须传给真实调用者，不能靠测试环境替代实际队列 |
| grammar / sampler | grammar 替换、clone 后结果所有权、实例级 Turbo 配置、额外 RNG draw 语义 | 使用公开 A sampler 原语；消除对私有 distribution 对象的强转，专测后续随机状态 |
| 模型 / GGUF | BailingMoE3 名称及缺省兼容、缺失 MTP 权重、uint64 PLE hash | 上游实际消费者和 GGUF 元数据校验；每个兼容点独立提交，不整包恢复所有图修改 |
| 校准采集与复用 | small:S05 的公共回调激活采集、DFS 统计人口、trie 复用、周期 checkpoint 和异常保存 | small:S04 格式读取 / imatrix 导出已迁；采集 C 仍须验证实际模型 callback 的名称、shape、dtype、分支顺序与计数，不以转换器代替采集 |
| 独立证据工具 | GGUF 头读取、离线性能报告、请求 / 构建身份 | 比较器与公开采集器已迁；格式仍无完整模型 / 构建身份，必须保存旁侧运行证据。其他工具需各自验收统计分母、有限值和失败输出 |
| 工作区计划 | 64 KiB 容量向上取整、阶段 scratch 最大值、间接 dispatch 算术 | 纯值 B；容量取整不能改变 dispatch 几何；实际 buffer 与设备生命周期由 C / A 拥有 |
| 状态封装 | 有界 state envelope、scope / policy / adapter / nonce 校验 | B 可封装字节，C 包装 A 的 opaque state；需明确端序，同进程身份不能冒充持久跨进程模型身份 |
| 数学参考 | 无符号 Hadamard、RoPE 平移、RERoT mask / 可见性、GQA / DDVR 参考 | 可成为独立 B / 对照实现；不声称已改变生产 KV 可见性，也不与 signed Turbo WHT 混为一项 |
| 稀疏 / 合并参考 | mean / energy 选择、online softmax 三元组归并、有限 mask 保留 | 单独证明数学合同和异常回退；局部参考不自动获得稀疏 attention 整模型质量资格 |
| Vulkan pinned host 所有权（G14） | 使用 buft 对应设备分配、释放与查询，而不是硬编码 device 0 | 保留独立候选；alloc/free/getter 必须由同一设备 owner 闭合，不能借 G09 或已有 WHT 回归代替 |
| 后端算子 / 生命周期 | SCALE F16、MROPE / IMROPE 融合、字节 memmove、readback pin / 可见性、device staging 清理；meta 分片非零偏移已迁 | 公开既有算子可小补丁；overlap、stream / fence 和多 context 生命周期需对应证明 |
| CPU / CUDA 小优化 | 既有 Q2_0 VNNI、Turbo 小查询优先 native VEC | 与私有 PQ2_0 / TP5 分开评估；本轮没有 CUDA 或该 CPU kernel 的性能测量 |
| 评测与参考 | DAG 参考、reader 顺序、性能预算、评测 split 与协议 fixtures | 保留测试目的，修正错答案、泄漏分组、NaN 假通过和提前 PASS 后才能作证据 |

### 已检查、但当前 A 不具备旧优化前提

Meta graph-view cache-key（FBE15）的前提是旧 fork 让 graph view 继承父图 UID。当前 A 会保存并复用非零图 UID，但其 `ggml_graph_view()` 将 view UID 设为零，view 因而触发重建。仅搬旧 view key 修复不能改善这个路径；未来若引入 view UID 复用，必须同时定义父图 / 切片身份与生命周期。旧 pinned-readback retire（FBE12）依赖的 pending transaction 在当前 Meta 路径也不存在。两者保留功能目的，未来要将新 fast path、所有权、验证和收益一并设计；不孤立搬入旧修补语句。具体依据见 [meta-transfers-protocol.md](tests/a-seams/meta-transfers-protocol.md)。

### 重新核对后保留 A 的支持范围

G44 原候选建议 Turbo SET_ROWS 拒绝 F16 source，前提是只编译 F32 shader。固定 A 的 `vulkan-shaders-gen.cpp` 实际同时生成 F32/F16 source 到 Turbo 输出的 I32/I64 索引变体，因此不迁旧拒绝条件。这是纠正先前静态判断的前提，没有改换 A 基线，也不把其已有能力算成新 fork 功能；相邻 G19 日志不随它迁入。本轮另加入 16 项完整 packed-byte 回归验证 F32/F16 和 I32/I64 的组合，保留未写入的 guard 行。G09 的共享 scratch 身份与新发现的输入别名修改按各自的反例与修复回执验收。

本轮没有把714文件的登记当成全部语义已证明，也没有重新声称所有候选已迁完。新增结果关闭文本合同、校准格式与导出、CPU 独立交付，以及有明确反例的 Vulkan scratch / 输入别名子项；同形状 CPU / Vulkan streaming 单独验证。此前准入、logits 采集、偏移和 recurrent 生命周期仍保留各自原证据。完整 Vulkan 新结果以 [VALIDATION.md](VALIDATION.md) 为准。

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
| RERoT | 普通多请求、corpus 前缀重放或独立 parser 不具备其多 reader 共享 KV / 可见性语义 | 明确同 context、同 owner、动态可见性、recurrent / RNG 隔离协议，验证真实 reader 行为 |
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
