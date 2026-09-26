# TP5 融合版下一阶段执行手册

更新：2026-09-26。适用：五张 RX 6800、Qwen4EXP TP5 + MTP，仓库 `~/atomic-llama-cpp-turboquant`。

**当前已验证基座约 83 tok/s，尚未达到 100 tok/s。下一步先核验已有的多行 matvec 改动，再推进执行资源复用、GDN 精确恢复与真正的设备端草稿闭环；不要重新从 72 tok/s 的旧账开工。**

本文件替换旧路线图，不重复保留两套互相冲突的实施顺序。历史性能与失败记录仍查 `TP5.md`、Git 历史和 `TP5_FUSION_HANDOFF_2026-09-26.md`。后者是最新性能战报，但其分支/推送状态已落后于本次看到的本地 Git 引用，修正见第 1 节。

## 0. 开工规则与本轮授权边界

本轮仅审阅和更新规划：不恢复编译、模型服务或 GPU 实验，不修改生产代码，不提交或推送。以下命令是后续明确恢复实施时的操作说明，不是现在启动跑测的授权。

后续执行者必须先读本文件，再读战报。遇到冲突，按以下顺序处理：用户最新要求；当前源码/提交和对应原始证据；本文件；历史叙述。**提交已合并不等于已测，单次 smoke 不等于性能资格，功能非劣不等于独立提速。**

- 不切换、重置、清理或覆盖当前 dirty 工作树；不对用户改动执行 `reset --hard`、`clean` 或自动 stash。实验另建工作树，基座固定到提交，不追随可移动分支。
- 不停止 watchdog，不改 GPU 时钟/功耗/驱动/PCI 拓扑，不用故障复位制造更快样本。只管理本次 harness 创建的服务进程，不广域 kill 用户任务。
- 一次只允许一组 GPU 实验。正式测量期间不并行编译、哈希大模型、拷贝权重、运行其他 GPU 负载。取证在测量前后进行。
- 保留非 MTP、grammar、penalty、JSON、reasoning budget、RERoT 和其他模型的原回退路径。不要以 TP5 优化为名删除共享 Vulkan 的保活、稳定 tensor 身份及同步契约。
- 不绕过工具的目录访问限制。无法读取某个 worktree/artifact，就记录边界；可审阅本仓库可见的提交对象，但不得声称检查了不可访问的二进制或日志。

## 1. 基座、源码与证据：先把三个版本分清

### 1.1 本次核验的本地 Git 状态

| 对象 | 本次看到的状态 | 用途/限制 |
|---|---|---|
| 当前目录 | `work/mtp-100`，HEAD `aa9e7f5de`，已有大量未提交实验 | 不是 83 tok/s 的性能对照，不在这里继续堆实验 |
| 已验收融合提交 | `16e28b8ac63c9fd5a54611070c6405f064f69a21`；分支 `work/mtp-fused-83`；tag `tp5-fused-83-20260926` | 后续实验的固定性能/行为基座 |
| 整合提交 | `5742c9c59`，父提交为 `9e4926382` 与 `16e28b8ac` | 已发生源码 merge，不等于完整 TP5 验证通过 |
| 本地 `origin/master` 与 `work/mtp-fused-master` | 均指向 `9b3c33881`，即整合提交之后的战报文档提交 | 本次未 fetch/查询真实远端；只陈述本地引用，不再照抄“尚未 merge/未 push” |

战报 §4 的 `origin/master=9e4926382` 和“未 push”属于较早快照。当前没有新增证据证明整合版已完成构建、canonical、状态与性能验收，因此其资格仍为 **待验证**。不得据此自动发布、部署或修改远端历史。

原融合 worktree 记录为 `/var/tmp/tp5-fusion-82-wt`。本次工具不允许打开该目录，因此审阅的是仓库中的融合提交及可访问源码；没有重解析 `/var/tmp` 的 ABBA 原始结果，没有新跑性能。以下历史成绩均明确来自战报，而非本次复测。

### 1.2 必须使用的历史性能口径

| 证据 | 数值 | 可以得出的结论 |
|---|---|---|
| 历史 canonical `native-mtp-kv-only` | `2065.730 ms / 82.779 tok/s`，prompt31、171/171、自然 stop、全文正确 | 已有有效成绩；旧 binary 本体未保留，不能假装仍可直接复用 |
| 融合 checkpoint 五块 ABBA | A mean `83.45`、B mean `83.41 tok/s`；B/A `0.9995`，95% CI `[0.9919, 1.0072]` | checkpoint 满足既定 0.99 非劣界；没有可测的独立正收益 |
| 融合单次 smoke | `2058.120 ms / 83.09 tok/s` | 功能与路线核验，不替代 ABBA |
| 无调优参数默认 smoke | `2076.556 ms / 82.35 tok/s` | 默认解析/功能通过，不用单次值与 ABBA 均值比较 |
| 当前实验目录的 78–79 tok/s smoke | 不是融合基座 | 不可降格选择弱对照以制造收益 |

证据路径见战报 §1–3、§8；其中关键结果是 `fusion-checkpoint-abba5.json`。新实验必须同时保存 server、实际加载 DSO、shader/构建配置及源码身份，不能仅记 HEAD 或 server 文件名。

### 1.3 距离目标还有多少

最终门仍为相同 canonical：全文精确 `1..60`、prompt31、`predicted_n=completion_tokens=171`、自然 `stop`，无 profiler 的 server `predicted_ms < 1710`。

以 **2065.730 ms 这一单次历史样本**作工程量级估算，还需减少 `355.730 ms`，即约 `17.22%` 的请求时间。若新的测量仍为 25 cycles，等价于每轮约 `14.23 ms`；25 轮来自历史诊断，必须在融合版重新确认，不能假定不变。不要把平均 tok/s 的倒数当作平均请求时间。

旧的 `2437.917 ms` 分相账、`514.051 ms` draft、`90.072 ms` 图外和 `95.056 ms` 验收只保留为历史定位线索。W1、K/V-only、ARGMAX、checkpoint 已改变执行路径，禁止拿这些旧数直接承诺新收益。

## 2. 已完成项与负结果：不再列成“从零实现”

### 2.1 融合基座必须保护的成员

RELAY/F32、linear lowering、单卡 draft、n=6、hierarchical ARGMAX、rank-local pinned readback、small-Q8 columns、non-replicated attention、QSA/GDN headmap、K/V-only catch-up、durable sampler checkpoint 均按战报属于融合成员。

源码审阅进一步确认：

- `common/speculative.cpp` 在 backend sampling 可用且 `p_min<=0` 时挂接 greedy backend chain；`draft()` 仍逐步 decode、等待所选 token、重建 batch。因此“再把草稿 argmax 搬到 GPU”不是新任务，**消除高层逐步控制依赖**才是。
- `src/models/qwen4exp.cpp::graph_mtp` 的 `LLM_GRAPH_TYPE_DECODER_MTP_KV` 分支完成 K/V 旋转和缓存写入后直接返回；后面的 Q、attention、输出投影、MoE、LM head 不在该数学路径内。不能再把删掉它们当作待兑现大收益。
- W1 的持久 pinned slots、全部 rank 入队后消费已经实现。剩余 fallback 必须先证明动态命中、语义范围和关键路径成本，不能重复实现另一套所有权。
- checkpoint 是已经保留的状态管理优化，不再从旧 clone 的 6.363 ms 推算它的“预期收益”。

默认启动只需保留模型/模式身份：`--tp5 qwen4exp-af -md <mtp.gguf> --spec-type draft-mtp`。不要强迫用户重新填写已默认的 draft 卡、n=6 或一长串性能环境变量。显式退选仍保留，详见战报 §3。

### 2.2 失败档案与重启门

| 路线 | 已知结论 | 只有满足什么新条件才能重启 |
|---|---|---|
| strict GPU target top-1 | audit-free pilot 比关闭慢 22.425 ms | 新账证明存在可省的关键路径；消除新增同步/全量物化；采样状态契约先通过 |
| host-hidden unrolled chain | 3488.112 ms；同 horizon 普通 n=2 为 3019.149 ms | token、embedding、hidden、索引和 mask 真正闭环，且两步同形对照先赢 |
| CPU stateless greedy / compact top-k | 正确但 pilot 回退 | 新机制有字节/工作量和整请求证据，不只换名字再试 |
| device feedback / external token fence / staged relay draft | 未证明净收益，已撤回/退选 | 不再依赖逐 token host embedding materialization 与高层调度 |
| direct GDN snapshot injection | 状态语义不安全 | 先有独立 shadow 恢复 oracle、所有权与事务协议；不直接复活旧写入补丁 |
| GDN target-capacity | fail-closed | inactive rows、KV、rollback、collective 全部证明后才准入 |
| horizon、五卡 draft、挪 rank、Q2/Q3、LateBind 等扫描 | 已有负结果或驻留风险 | 先有瓶颈变化与成本模型；不原样重跑参数矩阵 |
| device embedding placement | 孤立高值 smoke，无独立资格 | 作为完整 D1 闭环的组成部分验证，不单独晋级默认 |

同样保护已修复的 skinny matvec 输入转换缓存身份问题：不得将栈上临时 view 地址重新用作跨调用缓存键。不要照抄整合分支 AGENTS 中更早的“生产源码尚未修复”叙述；查 `TP5.md` 的 `ef48a9d40` 修复与对应回归。

## 3. 实际实施顺序与交付清单

编号为本轮新编号，括号中仅标旧 W 工单的对应关系。**先通过 G0/G1，再选择有证据的改造；不要同时改 shader、状态协议、采样器和 horizon。**

| 优先级/工单 | 第一件事 | 可交付结果 | 停止条件 |
|---|---|---|---|
| P0 / G0 基座与整合资格 | 固定 `16e28b8ac`，区分整合版和 dirty 实验 | manifest、干净构建、行为门、可重建对照 | 对照无法重现约 83 的合理水平，先查来源，不用 79 作新基座 |
| P0 / G1 当前关键路径（W0） | 测融合版，而不是拼历史数字 | 同请求互斥账、stage/shape 排名、驻留表 | profiler 扰动/归因不清时，不依据它宣称净收益 |
| P1 / K1 现成多行 matvec 增量（W6 的计算侧） | 隔离上游已存在的 shader 改动并统计命中 | 同形核数值、实际七行 target、完整请求对照 | 不命中、VGPR/spill 恶化或整请求不赢即退选 |
| P1 / E1 exact 程序与资源复用（W6） | 计数 phase reset、release、bind、record | 明确 phase 身份、稳定 arena、冷热分账 | 不能只少一次 reset，却留下旧 tensor/descriptor 引用 |
| P1 / S1 GDN shadow → 精确恢复（W3） | 保留旧快照，先逐层 shadow replay | 状态 oracle、日志字节、拒绝恢复成本 | 任一位置状态不等价，禁止删除旧快照 |
| P2 / D1 两步设备闭环 → 六步（W5） | 先证明当前单卡 draft 可两步无高层往返 | 完整输入/状态 ABI、两步及六步整请求收益 | 两步不赢，不继续复制六份图或切五卡 |
| P3 / V1 target 小候选（W2） | 证明默认 sampler 的完整可观察语义 | 单次候选发布、状态一致、有效 fallback | 只在换了采样链的合成 workload 上快，不算 canonical 优化 |
| P3 / H1 自适应 horizon（W7） | 获取新执行结构下的前缀存活分布 | 有边际成本的策略 | 仅凭接受率高就增大 horizon，拒绝 |

K1、E1、S1 的设计/CPU 测试可独立推进；GPU 实验串行。D1 的契约设计不必等全部工作结束，但大规模实现必须经过两步生死门。每个候选单独 qualification；最终组合仍须重新测，不能相加各自最佳收益。

## 4. G0/G1：基座、最小观测与收益归因

### 4.1 G0 的具体步骤

1. 保存当前 branch、HEAD、dirty diff、未跟踪文件清单；不要修改它们。固定 control 为 `16e28b8ac`。另建 integration 候选检查 `9b3c33881`，但不要混称为 control。
2. 在新的 build 目录干净构建；记录 compiler、CMake、shader compiler、构建选项和动态库搜索路径。特别验证 ARGMAX stage1/stage2 文件、descriptor 数与 shader generator 对应；不能借旧生成文件掩盖干净构建失败。
3. 先运行已有单测，再做 canonical/default route 与行为门。保留实际模型分片身份、量化、KV 类型、context、batch、sampler chain、默认解析结果、设备 UUID/BDF、driver 和 watchdog 状态。
4. 同机新 control 的 A/A pilot 用于发现环境漂移/测量异常，不赋予性能提升资格。恢复约 83 tok/s 的稳定参照后才测候选；偏低时检查 DSO 混用、构建差异、环境继承和驻留，不把旧最快值拼进新表。
5. 整合版先过构建和行为门，再对 control 做配对性能。若失败，隔离共享 shader 与 server/RERoT 改动；不声称无冲突 merge 已兼容所有行为。

### 4.2 G1 只收集能推动决策的账

每条记录至少关联 `request/cycle/phase/step/rank/stage/rows/generation`。第一轮只收阶段级信息，请求结束统一输出，不逐 op 打日志、不强插每算子等待来破坏 replay。

CPU 互斥账包括：prepare/build/bind/record、enqueue、等待、真正 sampler CPU 独占工作、accept/commit、K/V catch-up。GPU 账按各 rank 记录 compute、copy、producer-ready、CPU publish 对应边界和 consumer resume。不能相加五卡时间作为请求延迟，也不能把 CPU 等待和被等待的 GPU 工作再相加。

战报的 96-stage relay 诊断为 producer-ready cumulative 46.403 ms、CPU relay work 3.225 ms、prefix transfer 0.106 ms。这是**特定旧诊断范围**，不是新的 83 tok/s 请求完整分账。它足以提示优先查 producer，但不足以给 CPU relay 承诺某个整请求节省数。

输出三张可决策表：

- **阶段表**：每 phase 的调用数、rebuild/reuse、release/alloc、bind/record、首次与稳态时间；特别区分 native K/V-only 和 predefined 路径。
- **producer 表**：按 stage 的关键路径贡献排序，列各 rank 的 shape、量化、实际 kernel、最慢 rank、ready skew。进一步区分大家都慢与某 rank 慢，不直接把 skew 命名为通信或 DPM。
- **内存表**：权重、持久 KV/GDN、snapshot bank、临时 gdn_out、日志、target/draft scratch、readback slot 的 owner/live range/每卡字节；分别记录逻辑分配与 VRAM/GTT，不凭 GTT 猜某张权重发生迁移。

跨 CPU/GPU 关联使用支持的 calibrated timestamps 并保存 deviation；不支持时保留各时钟域的局部时长与因果关系，不直接比较原始 tick。正式性能关闭 profiler。需要硬件 counters 时另开诊断，不把多 pass 的 counter 采样当正式吞吐。[外部契约 A、B]

**G1 完成门：** 能解释当前总请求的大部分时间且残差公开；有未插桩对照证明观测扰动的量级；排名能明确选择 K1/E1/S1/D1 的下一项。没有可归因的大头，就不要再堆全局 flag。

## 5. K1：首先核验已有的七行 matvec 改动

### 5.1 本次发现的具体增量

`git diff 16e28b8ac 9b3c33881` 显示，整合版的 `mul_mat_vec.comp` 已把原来 `num_rows == 1 && NUM_COLS > 1` 的复用快路扩展到多输出行。入口仍受 `K_PER_ITER == 8`、非 `MUL_MAT_ID`、非 `HC_DOWN_Q8_DOT` 等编译条件约束。

必须区分三件事：

1. **权重跨列复用**：整合版已有相应循环调整，不应重新从零实现，也不能未经命中统计断言所有 Qwen 投影都会获益。
2. **activation staging**：实际代码条件是 `NUM_COLS <= 4`。附近注释写到 8，但不是当前执行条件。七列路径仍在每个输出行中加载 B；“只把 4 改成 7/8”不是已证明正确且更快的方案。
3. **输出行 tile 扩大**：`ggml-vulkan.cpp` 的 `rm_wide` 改动只应用到 PQ2_0 对应 pipeline。不能把它的注释数字外推到本模型所有量化类型，也不能混淆输出行 tile 和 MTP 的 token 行数。

### 5.2 实施阶梯

先列实际七行执行中每类权重的 `(type, M, N, K, strides, rows-per-WG, NUM_COLS, flags)`，记录 guard 命中和关键路径占比。普通 `MUL_MAT`、专家 `MUL_MAT_ID`、HC fused region、small-Q8 要分开；MoE 的专家维度不是 horizon。

做独立候选 K1a：从 `16e28b8ac` 出发，只移植/评估多行复用相关最小增量，保持编译和所有其他默认不变。完整整合版 B 另做兼容验证；不要用 B 相对 A 的收益冒充 K1a 独立收益。

K1a 通过后，才选 K1b：对真实高占比的七列 shape 比较当前 per-row B load、有限列 staging、合理 row tile。每次只改变一个维度；记录寄存器、spill、LDS/occupancy（工具可用时）、真实字节模型与同形时间。寄存器占用可能限制并发，不能把“少 load”或“更高 occupancy”直接等同净收益。[外部契约 C]

MoE 另统计每轮七个 token 的专家并集与复用次数。只有实际存在复用且重排不改变每 token 的专家选择/累加顺序，才讨论 grouping；不能把普通矩阵快路直接套到 `MUL_MAT_ID`。

### 5.3 验收与退出

先对齐真实 shape 和尾块：1、2、4、7 行及非整齐权重分片，覆盖连续不同输入、cold/replay、HC fold stride、small-Q8 和被排除的 fallback。逐元素比较，并明确是否改了浮点运算次序；改变数学顺序的候选单列数值资格。

再看完整七行 target 和 MTP-off 单行保护线，最后完整 canonical ABBA。新核不命中就关闭该任务；局部快但整请求慢则退选。禁止取消既有 small-Q8 特例或强开 MMVQ 来掩盖问题。

## 6. E1：先修 exact phase 生命周期，不先放宽 capacity

源码入口：`llama_context::process_ubatch()`、`ubatch_execution_phase()`、`llm_graph_result::can_reuse()`、Meta binding 与 Vulkan replay。

当前 native K/V-only 有独立 graph type；phase helper 对普通路径仍按 `n_tokens > n_seqs` 分类。phase 改变会 `gf_res_prev->reset()` 后 `ggml_backend_sched_release_buffers()`；predefined MTP 有豁免，但这不等于 native K/V-only 同样获益。这是**可核验的重复建图/分配风险**，其在 83 基座上的动态成本尚未测定。

实施顺序：

1. 在融合 control 记录 `DRAFT_ONE → CATCHUP_KV(rows) → DRAFT_ONE` 和 `TARGET_VERIFY(7) → partial → 7` 的 phase、graph UID、release/alloc、buffer generation 和实际 bind 时间。先确认发生几次、占多少，而不是引用旧首图约 100 ms。
2. 建立显式的 draft、target verify、K/V-only 程序身份。第一版保持 **exact rows**；cache key 包含 graph type、真实形状/布局、headmap、精度、设备、内存 plan generation 及必要的 KV/attention 结构。positions、token 和 active metadata 只有在契约允许时才作为运行输入。
3. 把可复用的执行定义与临时 arena 所有权分开。先解决安全保活与绑定计划，不急于同时存多份完整 scratch。记录新常驻内存；串行 phase 才可能复用同一 arena，重叠执行不能无条件 alias。
4. 为输入 slot、descriptor、tensor view 和命令引用建立退休规则。旧 GPU 提交结束前不重写、不释放；异常、取消、shape 变化、context 销毁都要退回安全路径。

不要为了少 reset 保留失效的旧 tensor 指针；不要复制一个 graph-result slot 就宣称有稳定双图；不要全局关闭 graph reuse，也不要通过伪造 UID 命中 cache。是否需要多个 exact shape 条目由实测频次和驻留预算决定，不一次缓存所有可能长度。

**验收：** 反复 1→7→1、accepted=0…6、零输出 catch-up、context 扩展、取消后下一请求、销毁时 pending copy；guard/状态均正确。把进程首请求、请求内首 shape、steady replay 分开报告。仅降低服务启动成本的收益不能算作已降低 warm generation 时间。

## 7. S1：GDN 精确恢复，先 shadow，后减少快照

### 7.1 先建立状态合同

`src/models/delta-net-base.cpp::build_recurrent_attn()` 当前用 `ggml_gated_delta_net_ext` 产生 `gdn_out`，再从其中的 snapshot view 拷贝到 recurrent bank；`K=n_rs_seq+1`、`n_written=min(executed_rows,K)`。共享/RERoT 状态另有分支。第一版只覆盖目标的单序列、非共享 GDN 路径，其他路径保留原实现。

逐层列清：状态输入、工作状态、snapshot 顺序、`rs_idx`、卷积历史、普通 attention KV 有效范围、QSA 相关缓存、target hidden、sampler checkpoint，以及每次 accept 后的提交边界。**accepted=0 不自动等于消费了零个 target row**；anchor/bonus/correction 的映射必须来自现有 workspace/server oracle，禁止自行写 a 或 a+1。

### 7.2 四级实现门

**S1a：只读字节与成本模型。** 按每层、每 rank 的真实 headmap 计算 `state_bytes = sizeof(F32) × S_v × S_v × H_v_local`，再算实际快照数、临时物化、bank copy 及峰值 live allocation。逻辑写入、copy 读写、allocator 峰值不能混为一个数；先查后端是否已消除了某次 copy。

**S1b：shadow replay。** 保留全部原快照。增加起点状态与逐步转移必要输入的只读日志，至少审阅 k/v/有效 gate/beta 和对应 head/layout/position 参数。复用原 GDN 的 F32 状态更新次序，在独立 shadow buffer 恢复每一个合法前缀，与原快照逐层逐位比较。不是重跑投影、MoE 和 LM head，也不是用代数简化改变舍入顺序。

多行 kernel 与单步 kernel 未必有相同浮点顺序；未证明等价前不能只重复调用现成单步核冒称 exact。日志必须保存实际消费的精度和布局，不能通过低精度重算输入来“压缩”后又称精确恢复。

**S1c：事务式恢复。** 仍保留原实现作为 oracle，验证起点 checkpoint、最终 working state、replay log 的 owner/generation。接受时按合法前缀发布，拒绝时恢复相应前缀，取消/失败时恢复或明确失效整个可复用槽位。卷积历史和 KV 有效区间同时处理，不只恢复 GDN 矩阵。

**S1d：减少生产快照。** 前三门通过后，才在受限路径改成起点/终点＋日志或实测更划算的稀疏 checkpoint。旧全快照路径保留为回退；这不是重新启用已撤回的 direct snapshot injection。若专门研究消除中间 CPY，也必须先有相同的所有权/状态门，单独立项。

### 7.3 收益与停止条件

按实际事件区分 model mismatch、EOG、长度限制、取消、请求结束且不复用。不要把 token 接受率当作无恢复概率；不能只测计数题接近全接受的情况。

`净收益 = 少物化/少写快照及驻留改善 − 日志开销 − Σ P(恢复到前缀 j) × 恢复成本(j)`。

输出每个前缀的恢复时间、日志字节、峰值内存以及短/长上下文的真实事件分布。只有结束且明确不再复用状态，才可跳过无消费者的恢复；保留 prompt cache 时必须恢复或失效，不能留脏状态。

任何一层、一个拒绝位置或下一请求不一致，停在 shadow，不删快照。若日志/恢复抵消收益，保留较简单的全快照实现。S1 的时间收益属于 target verify/恢复总项，不能与“target 提速”再相加一次。

## 8. D1：两步设备闭环先过生死门，再做六步

### 8.1 执行统一不等于所有算子强切五卡

保留当前单卡 draft 的权重/放置作为第一版，和 TP5 target 统一提交、状态和生命周期契约。先证明去掉高层往返有收益，再决定是否需要多卡 draft。五卡放置本身已有负结果，不是完成闭环的标志。

设备端仍有严格自回归因果关系。每一步依赖前一步 token/hidden，不得把六个草稿 token 当作独立并行 batch。可用有限步录制或有界命令程序，不使用永久满占用自旋 kernel。

### 8.2 写代码前必须交的输入表

| 输入/输出 | 每步规则 | 必须证明的事 |
|---|---|---|
| token | 第一步来自已提交 token，后续来自前步 sampler | 不再逐 token 回 CPU 才能做 embedding；无效 token 不得触发越界读取 |
| embedding | 与该 token 对应的原权重行 | 检查实际 backend placement，不因图上存在 GET_ROWS 就假定设备闭环；不得盲目复制巨量 embedding 权重 |
| hidden | 第一步用正确 carry，后续用本步 `t_h_nextn` | rank/宽度/owner/generation 正确，不 alias 未退休 scratch |
| position、K/V index、mask | 每步与原单步执行相同 | 第一版可一次预制有界的六步 metadata，但不得漏掉环形 KV、上下文边界或按步有效范围 |
| candidate、valid/EOG/cancel | 按 reference 的停止与提交语义处理 | 不能只有 dispatch=0，却仍写无效 KV 或让其他 rank 等不存在的 generation |
| catch-up | 验收后使用 target 真实 hidden | 复用已验证 K/V-only，不能因 token 全接受就直接删掉 |

源码入口为 `common_speculative_impl_draft_mtp::draft/commit`、`qwen4exp::graph_mtp`、`llm_graph_input_embd_h::set_input`、attention/KV input setter 与 backend replay。当前 setter 仍接受 host ubatch，所以只加 token CPY 或 external fence 不构成完整方案。

### 8.3 两步原型的硬门

先用相同权重、context、sampler、放置和 **n=2** 对照原实现：第一步 candidate → 第二步 embedding/MTP 必须在没有高层逐 token `llama_decode/common_sampler_sample` 往返的情况下完成。检查所有实际 enqueue、readback、host wait，而不是只计 API 名称。

原型包含有效的 attention/KV 位置和 mask；不得以少做状态更新换速度。记录计算、真实 CPU 控制、额外 copy、首次录制和驻留。两步只是执行机制筛选，不与 n=6 的 83 tok/s 混比。

两步完整请求不优于同 horizon 对照，或节省的边界成本小于新增开销，就停；不要机械扩到六步。通过后扩展 n=6，再与固定融合 control 做行为/性能全门。

### 8.4 内存、同步与回退

计算 scratch 按 live range 复用，token/hidden/metadata 用有界 slots。六份参数/命令不应自然膨胀成六份大 scratch。共享执行资源不能通过 `ctx_other=target` 偷换共享模型状态；target/draft 权重也必须逐张量核对后才允许 alias。

shader 写入下一步输入后，满足 compute-write → compute-read 的可见性；如果写的是间接 dispatch 参数，另满足 compute-write → indirect-command-read，而不只加普通 shader-read barrier。是否支持具体 API/扩展以设备能力为准。[外部契约 A、D]

第一版只覆盖明确证明的单序列、Qwen 单 MTP head、无共享 KV、受支持 greedy/backend sampling 条件。`p_min>0`、其他采样器/模型、并发、状态复用不满足契约时，在修改状态之前回退。执行后出错不能直接从半提交状态“重跑 fallback”；必须先恢复事务。

## 9. V1/H1：最后兑现的小候选与策略

### 9.1 target 小候选不是“temp=0 就可绕过采样器”

`common/sampling.cpp` 的普通 chain 默认末尾仍添加 `dist`。即便当前输出 token 相同，RNG、history、grammar、reasoning budget、候选观察接口的状态都可能不同。不能为命中新快路把 canonical 的 chain 改成另一个 greedy workload。

只有证明原链完整可观察行为等价的路径才准入；否则保持原路径。测试必须包含 checkpoint/rollback 后继续采样，不能只比当前 argmax。无效 logits 的策略、全无效词表、NaN/Inf、有符号零和相同最大值的 global-id tie 都以 reference 为准。

候选 ABI 一次产生 `(value, global_token_id, valid, generation)`，支持不等长/空 vocab shard。不得先回读 index，再由 CPU 决定第二次 logit 回读。第一版可以在 CPU 合并小候选并顺序验收，不强求所有逻辑都上 GPU。

以词表 248320、七行 F32 为例，全 logits 原始载荷为 6,952,960 字节；五 rank × 七行 × F32/I32 pair 是 280 字节，实际还要 valid/header/alignment。它只减少回读/物化，**不会免去 LM head 全词表计算**。所有字节数仍应由实际张量校验。

只有 G1 证明此边界仍值得优化且新的端到端 pilot 改善，才继续。原 strict candidate、CPU compact 的负结果保持有效，不用“理论上少拷贝”推翻它们。

### 9.2 自适应 horizon

先固定 n=6 完成执行优化。随后记录每个候选位置的前缀存活概率、新增 draft/verify 成本、恢复分布、峰值显存和公开输出 token，而不是只记 accepted 总数。

选择 horizon 的目标是提高 `E[公开提交 token] / E[完整 cycle 时间]`；多猜一步的边际收益要覆盖额外计算、恢复和驻留成本。合法策略不得使用尚不可得的 target 结果作弊。更换量化、卡布局、context 和 horizon 是不同因素，分别实验。

## 10. 可直接执行的后续操作模板

以下仅在明确恢复实施、工具允许相应目录、G0 的前置检查完成后使用。不得覆盖已存在的同名 worktree/build/artifact；缺少模型、工具链或证据时先停止该步骤并报告。

### 10.1 建立固定 control 与候选

```bash
set -euo pipefail
REPO="$HOME/atomic-llama-cpp-turboquant"
CONTROL_WT="$HOME/tp5-control-20260926"
CANDIDATE_WT="$HOME/tp5-candidate-20260926"
BASE=16e28b8ac63c9fd5a54611070c6405f064f69a21

git -C "$REPO" status --short
git -C "$REPO" show --no-patch --format=fuller "$BASE"
test ! -e "$CONTROL_WT" && test ! -e "$CANDIDATE_WT"
git -C "$REPO" worktree add --detach "$CONTROL_WT" "$BASE"
git -C "$REPO" worktree add -b work/tp5-next-20260926 "$CANDIDATE_WT" "$BASE"
```

这建立的是单因素实验起点，不是把 current dirty 或整合版整包抄进去。若任务专门验证整合版，另选固定 `9b3c33881`，保留相同 control。

干净构建的最小模板如下；实际应先恢复已验收 toolchain/选项，不能声称这些默认值必然与历史构建完全一致。A/B 使用相同配置，build 目录必须新建。

```bash
: "${BUILD_JOBS:=4}"
for WT in "$CONTROL_WT" "$CANDIDATE_WT"; do
    test ! -e "$WT/build-review"
    cmake -S "$WT" -B "$WT/build-review" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON
    cmake --build "$WT/build-review" --parallel "$BUILD_JOBS"
    ctest --test-dir "$WT/build-review" -N
    ctest --test-dir "$WT/build-review" --output-on-failure --no-tests=error \
      -R '^(test-sampling|test-arg-parser|test-mtp-workspace|test-tp5-plan|test-target-capacity)$'
done
```

不把 `ctest -N` 当通过，不把零测试或 return 77 的 GPU skip 当通过。GPU 定向测试按现有测试入口及所需 fixture 运行：ARGMAX、skinny/replay、rank-local readback `--vulkan`、GDN multistep、recurrent rollback；必须记录 backend 身份。新增状态/两步程序行为不由旧单测自动覆盖。

### 10.2 正式性能模板

先完成模型身份、五卡独占、watchdog、DSO、白名单环境与行为门。下面使用固定 control 的 harness，同一个已存在的 `native-mtp-fused-default` variant、两份二进制比较，避免误用脚本默认的不同 sync/wire 对照。

```bash
: "${TARGET_MODEL:?设为同一已验收 target GGUF 的首分片}"
: "${MTP_MODEL:?设为同一已验收 MTP GGUF}"
: "${RUN_DIR:?设为本实验新的证据目录}"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"

python3 "$CONTROL_WT/scripts/run-tp5-cross-matrix.py" \
  --variant-a native-mtp-fused-default \
  --variant-b native-mtp-fused-default \
  --bin "$CONTROL_WT/build-review/bin/llama-server" \
  --bin-b "$CANDIDATE_WT/build-review/bin/llama-server" \
  --model "$TARGET_MODEL" --mtp-model "$MTP_MODEL" \
  --blocks 5 --repeats 2 --count-to 60 \
  --host 127.0.0.1 --port 8097 --max-wall-seconds 2400 \
  --output "$RUN_DIR/abba.json"
```

候选必须在 B 中实际命中新机制；若以 opt-in 方式开发，需要显式加入同因素 B variant 并确认 resolved route，不能关着 feature 做虚假 ABBA。先用 `--list-variants`/`--dry-run` 检查 harness；dry run 是 NOT_MEASURED。`--smoke` 只验路线；不足五块须用 `--pilot`，不能升级为正式收益。

harness 的 Git provenance 来自脚本所在仓库，不天然证明 B 的源码身份。额外保存 A/B 各自的 commit、dirty patch、未跟踪源码、CMake 配置、server 与所有实际 DSO 哈希。不要让两份 server 意外加载同一目录的旧 DSO。

### 10.3 统计与发布门

五个 ABBA block 是独立配对单元；A1/B1/B2/A2 内的重复请求不能伪装成更多独立 block。按 block 计算差值/比值和预先选定的置信区间，报告完整样本、mean/median、离散度和冷/热分类。保留首请求，不能丢弃坏样本后再称稳定。

普通性能晋级要求完整行为通过、实际机制命中、整请求改善，并由预注册配对统计支持；仅非劣可因明确的正确性/维护价值保留，但不得宣传提速。小于噪声只称未检出收益，不反复抽样直到碰到显著。

100 tok/s 的声明必须满足原 workload、无 profile、逐字正确、自然停止和 171/171；对应声明类别的预定正式样本都要满足 `predicted_ms<1710`，并给出完整分布。若仅 warm 达标，必须写“热请求达标”，不能隐去 cold 或把启动预计算说成免费。额外记录 client wall/首 token，但不替代 server 判据。

成功合并后重新跑组合回归和正式对照，不相加 K1/S1/E1/D1 的局部最好值。未经用户明确要求不 commit/push；即使已授权推送，也先完成该实际提交的资格验证。

## 11. 正确性矩阵、实验记录与源码索引

### 11.1 每个相关工单必须覆盖的矩阵

| 类别 | 覆盖要求 | 比较内容 |
|---|---|---|
| 接受/拒绝 | 全接受、accepted=0…6、真实 mismatch 与定向强制位置 | token/position、hidden carry、target/draft K/V、各层 GDN、卷积历史、有效范围 |
| 结束/复用 | EOG、长度限制、stream cancel、取消后短/长请求、prompt-cache 复用 | 槽位有效性、generation、恢复/失效策略、下一请求输出与状态 |
| 形状/寿命 | 1→7→1、partial rows、零 logits catch-up、扩容、pending readback、context 销毁 | guard bytes、旧指针/descriptor、未完成提交、无越界/悬空引用 |
| 采样 | 原 canonical chain、有效 penalty、grammar、JSON、reasoning budget、随机采样与 rollback | 当前 token 以及后续 RNG/history/grammar 状态，unsupported 路径回退 |
| 分片/数值 | 不等长/空 shard、tie、NaN/Inf、尾块、headmap、不同输入连续 replay | 与明确 reference 对拍；布局先规范化；浮点次序改变单列资格 |
| 保护线 | MTP-off、既有 shared Vulkan/RERoT 回归、其他模型不命中新路径 | 不改变默认语义，不把单卡合成题作为五卡模型背书 |

强制拒绝是定向测试手段，不替代真实模型拒绝；合成全接受性能不能冒充真实 workload。状态优化至少比较所有受影响层与下一次转移，不只看最终计数文本。

### 11.2 每项实验留一个完整档案

在新的 `artifacts/tp5-next-<日期>/<工单>/<run-id>/` 保存 manifest、单因素 diff、编译/单测日志、完整请求/响应、未插桩 ABBA、独立诊断、内存表、状态对拍、全部失败和最终裁决。临时日志可放 scratch，但正式资格证据应入库或明确可持久访问；不要再次只留下 SHA 前缀和已经消失的旧 binary。

每份 `decision.md` 回答：改了什么；此前为什么失败、本次机制何处不同；实际命中什么；节省哪个互斥时间项/字节；新增多少内存/恢复成本；正确性和 fallback 是否通过；完整请求结果；保留还是撤回。没有证据就写待测，不补猜测数字。

### 11.3 源码/本地证据索引

- 最新性能：`TP5_FUSION_HANDOFF_2026-09-26.md` §1–3、§5–9；分支更正以本文件 §1 的本地 Git 核验为准。
- 已完成 W1 与状态回归记录：`TP5.md` 开头“正确性优先的 MTP 续做”，`artifacts/tp5-mtp-readback-20260924/`。
- 基座/整合差异：`git diff 16e28b8ac 9b3c33881 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp`。
- 草稿与 catch-up：`common/speculative.cpp`、`common/speculative-mtp-workspace.h`、`src/models/qwen4exp.cpp`、`src/llama-graph.cpp`。
- 状态：`src/models/delta-net-base.cpp`、`src/llama-memory-recurrent.{cpp,h}`、GDN shaders 与 `tests/test-vulkan-gdn-multistep.cpp`、`tests/test-recurrent-state-rollback.cpp`。
- 程序/回读：`src/llama-context.{cpp,h}`、`ggml/src/ggml-backend-meta.cpp`、`ggml/src/ggml-vulkan/ggml-vulkan{,-collective}.cpp`。
- 采样：`common/sampling.{cpp,h}`、`tools/server/server-context.cpp`、ARGMAX stage1/stage2 shaders。
- 测试/跑测：`tests/CMakeLists.txt`、`tests/test-mtp-workspace.cpp`、`tests/test-tp5-plan.cpp`、`tests/test-target-capacity.cpp`、`tests/test-tp5-rank-local-readback.cpp`、`tests/test-vulkan-command-replay.cpp`、`scripts/run-tp5-cross-matrix.py`。

外部契约仅解释 API/硬件约束，不证明本工程获得了性能收益；查阅日期 2026-09-26：

- A：Khronos《Synchronization and Cache Control》与《Synchronization Examples》：`https://docs.vulkan.org/spec/latest/chapters/synchronization.html`、`https://docs.vulkan.org/guide/latest/synchronization_examples.html`。
- B：Khronos《Calibrated Timestamps》：`https://docs.vulkan.org/samples/latest/samples/extensions/calibrated_timestamps/README.html`。
- C：AMD GPUOpen《Occupancy explained》：`https://gpuopen.com/learn/occupancy-explained/`。
- D：Khronos《Indirect Dispatch: Building Parameters on the GPU》：`https://github.khronos.org/Vulkan-Site/tutorial/latest/Advanced_Vulkan_Compute/07_GPU_Driven_Pipelines/02_indirect_dispatch.html`。

**下一次恢复工作的首个交付不是新的 flag：是固定基座、现成 matvec 增量的命中清单，以及融合版当前的 phase/producer/内存三张账。随后用最小可验收补丁兑现约 350 ms 的缺口；无证据就停止该候选，不把实现规模当作进展。**
