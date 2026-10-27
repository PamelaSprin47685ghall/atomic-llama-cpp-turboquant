# 最后 8 个 server / memory 合同文件的实际内容补审

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

本轮只处理 root 指定的 8 个文件。8 项都已从“仅名称/符号盘点”提升为实际内容审阅：4 个新增小头文件全文读取，1 个修改头文件读取全部已提供内容，3 个大文件读取完整的关键函数、结构体或声明族。大文件的未读部分仍逐段登记，不能把这次补审理解为完整实现审计。

发现的有价值小能力均登记为**保留候选**，不因它们原先属于 MTP、FlashPrefill、XKV 或 RERoT 而整体舍弃。本轮没有移植功能、修改远程源码、运行测试或模型；最终实际保留状态和验证结果由 root 的迁移总账及 `VALIDATION` 决定。

## 证据与覆盖

源码基准为原 fork `2777543b2358c88cd1c2000f7fa342b764a0e213` 相对上游 `cda0f9256665845cd9d6f3a0176880883f653a3d` 的精确 diff。本文所有行号是 **diff 新侧的原 fork 源码行号**，不是 diff 文本行号，也不是干净分支行号。

| 输入 | SHA256 |
| --- | --- |
| `audit-common.diff` | `e8a17e1c05e0bedfca48e9e5d2b801ad72d3da338618dd531ee18e5eaef989ef` |
| `audit-server.diff` | `30dd21577728279db5251bdb8f0b4d54a56498fa616f9b5f90072d4687d1cbb9` |
| `audit-memory.diff` | `515668ba2ca9ee5ee6967e0705345ae7e0a143bb808e2892172afb19c8d7e0e2` |

| 文件 | 实际读到的行数 / 已提供行数 | 深度 | 发现 |
| --- | ---: | --- | --- |
| `common/speculative.cpp` | 636 / 1534 | 完整关键函数及明确标出的初始化/accept 局部块；不是整文件 | FC002–004、FC011 |
| `common/speculative.h` | 68 / 68 | 全部修改内容与上下文；未提供的原文件其他部分不算已读 | FC002–003 |
| `include/llama-flashprefill.h` | 251 / 251 | 新增头文件全文，主要是合同和声明 | FC006 |
| `include/llama-predefined-hidden.h` | 53 / 53 | 新增头文件全文，含行计划与设备状态 API 合同 | FC004 |
| `include/llama-predefined.h` | 31 / 31 | 新增头文件全文，含资源准备与能力查询合同 | FC005 |
| `src/llama-kv-cache.h` | 750 / 1132 | 完整声明族、结构体和选定 inline 函数；不是全部实现 | FC007–008 |
| `src/llama-turbo-config.h` | 20 / 20 | 新增头文件全文及完整 inline 实现 | FC001 |
| `tools/server/server-rerot.h` | 956 / 1276 | 完整 parser / mux / checkpoint / episode 结构及选定运行时、持久化声明族 | FC009–012 |

合计：8 个实际内容审阅路径，4 个新增全文、1 个全部已提供修改内容、3 个选择性内容审阅。已提供 4365 行中实际读取 2765 行，剩余 1600 行明确登记为未读；仅用途/符号盘点项为 0，实读范围不确定项为 0。

机器总账为 `audit-final-contract-coverage.json`，顶层 `files[]` 严格只有这 8 条。每条含实际读区、已提供但未读区、未提供的 unchanged 间隙、未知 unchanged 尾部、实际审阅单元、行为、依赖和保留候选。modified 文件的分母是**已提供源码行数**，不能拿它表示完整文件长度。导航时的符号索引未计入内容审阅。

## FC001：按缓存实例解析 Turbo 默认模式，是独立的小正确性修复

**证据：** `src/llama-turbo-config.h:8–20`，完整函数 `llama_turbo_layer_adaptive_mode`。

该函数每次根据本次 `type_v`、层数和显式环境值解析模式。没有函数局部 `static` 保存第一次结果，因此第一次 fit 探测或第一个模型不会替后续缓存决定策略。未提供 override 时，仅 `GGML_TYPE_TURBO2_0` 且层数至少为 8 返回模式 7，其余返回 0；有 override 则直接用 `std::atoi`。这不是只有注释的设计：20 行中的实际函数确实没有全局缓存状态。

**保留边界：** 值得作为独立 B 策略，输入由 C 适配；如果模式仅能从 A 的缓存构造处生效，也可以是一处很小、明确的 A 调用点修复。它的价值是多模型、多个上下文和 fit 探测之间的正确隔离，不依赖性能跑分。

**限制：** `atoi` 不是严格配置解析器；空串/非数字可变成 0，数字前缀可被接受，没有可报告的溢出检查。迁移时可以决定是否改变该输入合同，但不能悄悄把它描述为严格校验。本轮也没有检查模式最终如何替换各层 K/V 类型。

## FC002：周期统计可独立保留，但当前字段不等于成功产出

**证据：** `common/speculative.h:88–122`；`common/speculative.cpp:30–59,2197–2213,3222–3315,4080–4088`。

值类型记录 draft、target verify、catch-up、hidden handoff 四个阶段、草稿与接受 token 数。格式化函数返回拥有自身存储的 `std::string`，对零草稿/零周期给出 0；汇总接受率使用总接受数除以总草稿数，而不是对每个周期的百分比取平均。`record_cycle_settle` 汇总这些值并无条件清空当前周期计数与暂存的 target 时间，避免跨请求、跨周期残留。它的 `total_us` 是四项时间之和，不能自动解释为从请求入口到出口的 wall time。

**需要保留的边界：** `cycle_settle_guard` 的析构会在 `commit` 提前失败时仍结算周期；随后 `record_cycle_settle` 仍生成 `final_tokens = accepted_tokens + 1`，没有成功/失败字段。因此日志可以说明一个已开始的尝试周期，却不能单凭 `final_tokens` 证明整个 catch-up、审计保存和提交链条都成功。`+1` 也带有“本周期一个已采样 token”的口径，迁移到多序列消费者时必须重新对齐真实产出。

**保留边界：** 统计结构、格式化、清零和汇总规则适合 B；C 应在真实草稿、校验、提交位置产生数据。若需要“成功产出吞吐”，应从真正成功的提交/输出事件计数，或显式区分尝试与成功状态。这里没有把已有清洁采样 CLI 冒称为已拥有 MTP 统计生产链。

## FC003：延迟 catch-up、暂停与恢复有实际合同，错误恢复并非原子操作

**证据：** `common/speculative.h:58–63,77–86`；`common/speculative.cpp:218–235,2215–2288,2693–2763,3222–3265,3847–3901,4070–4078`。

读取了完整 `commit`。它先检查预定义容量，再逐行从 workspace 取得待提交 token/位置/hidden，清除 staged 元数据后对 draft 真正 decode；零行时清除 staged 状态并成功返回。容量、非法行、decode 失败以及可选的真实 sequence-state 审计文件保存失败，都返回 false。外层 `common_speculative_process` 与 `common_speculative_commit` 明确分离；defer mask 复用已分配存储，仅当对应上轮实现和非空草稿存在时进行该序列延迟处理。

暂停功能也不是空名字：`set_paused` 在进入暂停时清理本序列工作区与 sampler，并且只有 draft 独立拥有内存时才删除 draft 序列，避免从共享内存中删除 target 的状态。它为 RERoT 等 draft 不能表达的历史保留了明确的关闭镜像路径。基类的 optional export/pause/commit 默认函数大多是 no-op，不能由统一签名推断所有算法都支持这些能力。

**观察到的恢复边界：** `set_state` 在 `data.size()` 校验之前已经调用 `workspace->reset_sequence`、清除 generation、可能 reset device carry，并清除其他本序列状态。错误长度会记录警告后返回，旧状态已经变化；其返回类型还是 `void`。所以不能把它当成“失败时输出不变”的原子恢复 API。相比之下，该文件中的 `get_state` 导出一个 hidden row，且拒绝无效序列。

**另一个边界：** 外层循环用 `result = result && impl->process/commit(...)`，第一次失败后会短路，后续实现不会再执行。这适用于停止后续工作，但不能当作“失败也尝试清理所有实现”的循环。所读 `commit` 也不是跨 target/draft 的完整回滚事务：它可能在清除 staged 元数据并进行 decode 后返回失败。

**保留边界：** 拥有内存的 accepted-row 计划、恢复前验证、明确 pause/reset/commit 状态机值得保留为 B/C 组成部分；真正设备 hidden 和 draft 上下文操作仍须真实接口。完整 MTP `process`/`draft` 循环和 adaptive controller 未在本轮重审。

## FC004：设备 hidden 行计划与 generation 防旧读，不能用普通请求替代

**证据：** `include/llama-predefined-hidden.h:9–49`；`common/speculative.cpp:1134–1189,1774–1799,2679–2691,2693–2738`。

新增头把 `RESULT`、`CARRY`、`SEED` 与虚拟 `INPUT` 分开：RESULT 只能由成功图输出捕获写入，显式 copy 更新 CARRY/SEED；RESULT 源需要 generation，因此同样行数和形状的下一轮验证结果不能被旧 accepted-row 计划误读。INPUT 是完整覆盖 token 行区间且无洞、无重叠的计划，直接供图输入，不再保存一份 packed 输入矩阵。host carry get/set 明确只用于 checkpoint/reset，一个 hidden row 的字节数必须准确。

所读 MTP 消费者实际把 SEED 作为第一行、target RESULT 作为随后行，并携带 verification generation；接受后又选择正确 target 行拷入 CARRY。startup 路径拒绝多序列、chain heads 与共享内存，并在后端 enable 成功后才设置 device_hidden。DFlash 的另一个完整 startup 函数要求 target/draft 处于同一个非 CPU 单设备，检查 F32 几何与设备 copy 能力；启动阶段检查失败会回退 flag，使 host 提取路径保持可用。源码注释同时明确，首次 decode 之后不能沿用同样回退假设，因为 host buffer 可能未分配。

**保留边界：** generation、行范围覆盖、源/目标角色可作为 B 的独立受检计划；真正持久设备 hidden 和图输入捕获需要一组明确的 A 小接口，由 C 真正消费。函数位于旧 fork 的 `include/` 不代表它已存在于干净上游公共 API 中。

**限制：** 本轮没有审查设备 copy、分配和图捕获实现，没有证明端到端避免 readback 或获得任何加速。range 中的 source 是借用 context 指针，不转移上下文所有权；消费者必须负责生命周期。用 CPU 的多次普通 completion 请求不能复现这里的设备行与私有状态语义。

## FC005：容量准备与图能力分开，保留 truthful capability 查询

**证据：** `include/llama-predefined.h:10–27`，全文头合同。

该 API 声明 startup 共享 high-water compute pool 和不可变 capacity record，并仅在相同 buffer type 之间复用分配；同一组 GPU 的不同 meta model wrapper 仍可能有独立条目。graph reserve 是尺寸准备，不执行模型或 shader。非 TP5 / 非 Qwen 的已有 MTP 路径可以返回成功，同时没有挂载 session；`get_capacity` 返回 false 才是明确的“没有 session”查询结果。

这个区分值得保留：资源准备成功，不意味着所有 Vulkan operator 都接受最大容量框架。头文件明确禁止仅据 reserve 成功就把 `ne == rows` 放宽成 `<=`。

**保留边界：** 容量值、查询结果和真实逻辑行数的独立合同可在 B/C 保留；共享 scheduler/buffer 分配本身依赖 A。这里是完整头声明审阅，不是资源分配实现验证。

## FC006：FlashPrefill 策略合同本身已接近独立 B

**证据：** `include/llama-flashprefill.h:1–251`，全文。

它只依赖标准 C 头，使用 forward declaration，不引入 `llama.h` 或 GGML wire 头。config 与 row 是带 version/struct_size 的值，reserved 字节要求为零；exec 借用调用者 rows，仅在消费调用期间有效，不能保存指针、释放或修改。该分界适合独立库，避免把后端 tensor 所有权带进 B。

合同还包含若干容易被大主题吞掉的小行为：

- 默认 OFF；仅 PREFILL 与 REROT_TEACHER_FORCED 允许稀疏候选。decode、MTP draft/verify、replay、embedding、rerank、多模态等走 dense。
- dense tail 可以按一次调用，也可以按冻结的逻辑 prompt 区间。边界是否已知有单独 flag，区间尾端为 exclusive，不能由当前 batch 形状猜出整段 prompt。
- alpha 定义为能量阈值因子，不是稀疏比例。mean correction 是生产合同，关闭它是 ablation；exact_all 是调试新路径的开关。
- 失败类型和检查顺序固定，fallback 原因可以作为指标。policy fingerprint 覆盖显式策略字段及 model/adapter 标识，NULL 与空串域区分；它是 FNV-1a 身份值，不是密码学认证。

**必须保留的消费者校验：** `route_is_dense` 对越界枚举也返回 false，因此不能用 `!route_is_dense(...)` 就批准稀疏执行；必须检查合法的具体 route。另有一处合同措辞需要在迁移时对齐真实 validator：前文说 `n_rows == 0` 允许 `rows == NULL`，成员注释写成 `NULL iff n_rows == 0`，两者对空 view 的非 NULL 指针是否允许并不完全等价。本轮没有取得 validator 的新证据，不能自行增加限制。

**保留边界：** 值、验证器、策略 fingerprint、角色与 fallback 决策是 B 候选；C 提供真实角色和逻辑边界。稀疏执行、KV fragment 和后端 lowering 仍须单独 A 接口。头注释声称 pure/no-allocation，不等于本轮实测了实现；helper 定义不在这 8 文件范围。

## FC007：KV 事务小合同应保留，不能把声明当成已证明的原子实现

**证据：** `src/llama-kv-cache.h:142–184,304–399,600–716,769–790,1059–1075,1102–1173,1292–1306`。

头中并非只有大规模 XKV 改动的“标签”。受害行快照保存 sequence 的完整多引用集合、位置/shift、RERoT metadata、payload ID/generation，以及是否存在于 hot pool 和 store。这保留了 factored 行只在 store、不占 hot slot 的区别，不能把两个删除列表混成一个。applied entries、pending victims 和 RAII reservation 是 context 拥有的值；旧行释放被延后到成功 compute 之后，失败时仍可恢复其引用和元数据。

释放合同明确分为 read-only validate 和 commit gate。store 持锁时的 gate 只能调用 pool，不能再回调 store；顺序是 store → pool。预先验证涉及 store 时应在外层执行。removal rebuild hook 提供一次语义重建入口，而不是先改 cells 后再问 store 是否允许。

还有可独立登记的小接口：实际 logical layer 的存储类型查询避免为检查而创建 tensor view；materialize 使分配边界可见；compaction 必须先 synchronize；Turbo 位置移位仍需真实 phase transform 和重数量化，不能因为 WHT 就把 shift 当 no-op；状态 trailer 仅在需要保存 factored 行时出现，OFF 和 PARTIAL_ONLY 合同要求不增加字节。

**保留边界：** transaction plan 的拥有值、准确快照、锁顺序、重复 postcompute 调用的既定结果和明确失败状态，值得 B 或必要的小 A memory lifecycle seam 保留。不能在完全不触碰实际 KV 所有者的情况下，仅用一个外部装饰器实现这些物理语义。

**实现证据限制：** 核心操作在本头中是声明。`postcompute_success/failure` 注释声称 exactly-once；rollback 一处概括为 atomic，另一处明确说所有步骤都尝试并返回 combined status。报告记录的是需要维持的合同，不是已证明跨 store/pool/cells 真正原子的实现。本轮未读其 cpp 或执行失败注入/并发测试。

## FC008：每个 owner 的 epoch、布局失效和共享所有权是独立小能力

**证据：** `src/llama-kv-cache.h:834–892,931–965,978–1004,1209–1244,1325–1332`。

实际 inline `fp_bump` 只有 `fp_active` 为真才递增实例 epoch；未启用路径保留一个分支，不进行计数写入。这里没有饱和或特殊 wrap 处理，不应加入源码没有的保证。RERoT world 的有效性和 generation 属于缓存 owner；lazy sequence ownership 列与持续 scratch 减少每 frontier 重建对象。读取到的 `rerot_world_set_meta` 真实捕获更新异常并把 world 标成无效，下次由完整重建处理；无效 world 上的 incremental upsert/meta 更新直接返回。

布局声明以**本 context 拥有的 ubatch**、role、capacity 为 key，owner stamp 验证 freshness；next/apply 后保守失效。build false 同时可能代表 policy 不适用与硬错误，调用者必须区分 reason 和 error，不能把所有 false 都静默吞成同一状态。

所有权声明还显示 store、runtime 和 coordinator 在共享同一 store 的缓存之间用 shared_ptr 共持有，避免每个 alias 自己维护互不知情的 coordinator。每个物理 device 只有一个受锁保护的 executor entry。`kv_stream_runtime` 在 `ctxs_bufs` 前声明，C++ 逆序析构意味着 buffer 成员先析构，符合其回调仍需要 runtime 存活的设计。

**限制：** `kv_stream_runtime_owner` 的主体在 diff 未提供的 unchanged 区间 `899–928`，本轮未审其自定义释放实现；只确认成员顺序和声明合同。cache 是否在每一种实际 mutation 中 bump stamp、shared runtime 的初始化是否完整，也需要对应 cpp 的实现证据。

**保留边界：** 按 owner 的无全局污染 key、失效策略、异常后丢弃派生缓存、scratch 与共享 executor 管理，均是保留候选。真正的 metadata mutation 和图消费仍需要 A 显式接点。

## FC009：保留 token 边界 parser 和逐 lane 的完整行输出

**证据：** `tools/server/server-rerot.h:16–172,198–275`，完整 parser / marker / line mux 声明族与值结构。

planner 是 byte-oriented，KV 可见性是 token/cell-oriented。一个 sampled token 的任意 suffix 可能成为 `<ol>` 或 close marker 前缀时，整个 token 保持 PENDING；只有判断完成后才通过步骤事件要求真正 KV 消费者重新发布。parser snapshot 拥有 opener/list/item/error 数据，恢复会拒绝坏状态、错误前缀、无 items 的 complete 等不变量，而不是修补。

line mux 拥有每个 lane 的阻塞片段与 partial line。某 lane 的 PENDING 只阻塞自身；PRIVATE 字节和其中的换行不形成外部边界；公开完整行按调用的完成顺序发出，lane terminal/fork 才允许把公开残余补一个换行结算。这是独立的流式输出合同，值得从大 RERoT 主题中救出。

**保留边界：** parser 和 mux 可成为 B，改由明确的 visibility adapter 访问逻辑文档；C 接到真实逐 token 事件。现有头直接使用 `llama_rerot_document` 和 visibility 枚举，不能仅复制头就声称独立。实现和已有测试的其他审计证据可以继续引用，但本轮新增证据是声明、状态所有权和小 inline predicate。

## FC010：逻辑持久化与真实 sampler 连续性必须分别保真

**证据：** `tools/server/server-rerot.h:297–341,464–477,497–630,958–979,1100–1258`。

fence checkpoint 的实际 `complete` 要求 prepared、rows 非空且 cursor 恰好等于大小；prebranch 的 `valid` 用 captured 标志和有效 seq/watermark，recurrent 字节为空也合法，不能拿 episode ID 是否存在替代捕获状态。`clear` 确实清除所有 checkpoint 字段。

持久化合同保存 logical episode：节点、run、queue、epochs、counter、parser snapshot 以及 sampler/MTP 扩展 bytes；采用 little-endian wire，检查 magic/version/caps/model-RoPE-Tri fingerprints 与拓扑/可见性/parser 不变量。load 声明失败不改输出，save 不产出部分 blob。physical KV cell/key 不进入文件，virtual position 通过逻辑 ownership 重建；exec/parked/archive seq 仍作为 lineage handle 保留。context envelope 同样不含 physical tensor bytes，需要与真实上下文恢复、views 安装配合。

源码非常明确地否认一种常见的错误替代：seed + accepted-token history 只能校验编码，不能单独恢复 RNG offset、grammar、reasoning budget 或 adaptive state。完整 sampler 连续性只在 transport 拥有 sampler clone 的进程内 demotion 路径成立；不能重新 seed 后就说恢复了同一运行。opaque MTP/sampler blob 也不等于此 runtime 能独立重新构造对象。

public-run 截断合同也包含小行为：保留公共 system/user prefix、FRAME 与活跃依赖；无可删除行返回成功且 tokens_removed 为 0，让上层拒绝资源请求，不能强行删掉 worker 结果。context epoch 只能前进，旧 MTP view stamp 需要变 stale。

**保留边界：** owned codec、严格长度/拓扑检查、metadata 与明确恢复次序值得 B/C 保留；完整运行连续性需要真实 KV/recurrent/sampler 提供者。这份头的 fingerprint 注释描述 arch、训练 context 和 RoPE/Tri 配置，不构成对所有权重或 tokenizer 的内容 hash 证明。所有 codec 大函数在本轮仅审声明合同，没有再次运行 interop。

## FC011：物理执行槽和逻辑任务身份应独立，调度小策略可保留

**证据：** `common/speculative.cpp:3531–3565`；`tools/server/server-rerot.h:277–295,414–478,845–891,940–955,1080–1098`。

MTP 初始化改动使 draft 的 sequence、output 与 recurrent 上限按真实 pen/parallel 数量配置，不照抄 target 为 parked document 节点准备的大逻辑 arena。可选 detached 单设备 draft 只改变自己的 split mode，不改变 target 的多设备设置。这是具体的内存配置行为，应作为 C 候选保留。

server 头把 lane 与 internal parked/archive/probe 的 sequence ID 分到不同区间，且按角色保存分配标记，避免同一 physical KV row 被两种角色重复领用。pen scheduler 的合同是先按 age 给当前无 pen 的 runnable person 一个，再在剩余 ready children 中 round-robin；单 person 可以用满 pen。完成 admission 不在逐行 commit 内自行调度；同 frontier 的退出按稳定 tree-path 顺序处理，HTTP/SSE 响应仍归 episode root request，而不是临时 lane slot。

**保留边界：** role-aware ID allocator、公平顺序、root response identity 可作为 B/C 的小状态机/规则；实际 `allocate_pen` / `schedule_pens` 实现在本轮没读，不能从成员数组或注释推断所有非法状态都已被防住。物理绑定仍依赖 A 的真实内存语义。

## FC012：chronicle → canonical 的历史映射，属于独立的实用小功能候选

**证据：** `tools/server/server-rerot.h:1260–1275`，完整声明族。

用户实时看到的 reasoning 是各 lane 完成顺序的 chronicle；下一轮如果原样带回这些文字，server 需要映射到 canonical PAC-DFS 文档顺序，才能与内部历史一致。合同提供 register、resolve、clear 三个接口，resolve 对未知、篡改或编辑的 reasoning 返回 `nullopt`，不猜测或擅自重新排列。

**保留边界：** exact-history registry 和拥有内存的 identity mapping 可在 B，C 在输出和下一轮输入接入。这比整体复制 RERoT runtime 小得多，仍应列入保留候选。此处只有声明；实际 key、ownership、eviction、并发、碰撞处理须由已审 cpp 或之后针对实现的工作决定，当前不能宣称这些性质已验证。

## 未读范围与停止边界

本轮按 root 的 8 文件清单关闭“只按名字扫描”的问题，未扩展到其他实现文件。3 个选择性大文件仍有以下**已提供但未读**区间，JSON 也分别列出了未提供的 unchanged 段：

- `common/speculative.cpp`：`663–704, 708–718, 1018–1031, 1118–1133, 1247–1260, 1262–1271, 1275–1287, 1291–1415, 1505–1523, 1577–1588, 1598–1612, 1620–1648, 1652–1662, 1678–1697, 1712–1728, 1745–1755, 1761–1773, 1809–1823, 1904–1913, 1925–2013, 2058–2196, 2290–2307, 2362–2372, 2490–2513, 2523–2543, 2549–2584, 2614–2626, 2647–2665, 3460–3499, 3698–3732, 3794–3819, 4129–4138`，共 898 行。
- `src/llama-kv-cache.h`：`83–95, 201–217, 224–258, 401–564, 569–586, 726–738, 741–768, 804–825, 973–977, 1005–1027, 1087–1099, 1245–1275`，共 382 行。
- `tools/server/server-rerot.h`：`637–836, 894–934, 1001–1079`，共 320 行。

`common/speculative.h` 的 68 行全部已读，但原文件 `1–53`、`69–75` 以及最后一段 hunk 后的未知尾部未提供，不能标记整文件完成。其余 4 个新增小头无未读区域。

这些范围不是隐含舍弃清单。它们只是本轮证据界限；root 的已有实现审计可以提供互补证据，但本补充账本不会把其他阶段的阅读、测试或注释承诺混入当前实读统计。
