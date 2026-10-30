# 内存与相位功能审计：功能清单、依赖与迁移边界

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

基线：上游 `AtomicBot-ai/atomic-llama-cpp-turboquant`，commit `cda0f9256665845cd9d6f3a0176880883f653a3d`；旧 fork HEAD `2777543b2358c88cd1c2000f7fa342b764a0e213`。资料是根代理提取的完整差异 `audit-memory.diff`、`audit-common.diff`、`audit-kv.diff`，辅助 `audit-phase-search.txt`、`audit-small.diff` 与权威 `audit-inventory.json`。后者列出 4634 个差异文件，排除 `artifacts/` 和 `reports/` 后为 714 个；初始 `audit-old-numstat.txt` 是缩减范围，不用它当全库分母。下述旧 fork 行号来自 unified diff 的 **new-side 行号**；“上游已有”只在差异 context/old-side 或干净 A 片段直接确认后使用。这里没有实测旧 fork 的任何显存或吞吐收益。

本报告优先回答用户点名的三项，随后逐项记录普通内存行为，不把它们统归为 TP5、XKV 或 RERoT。`audit-memory-coverage.json` 明确逐文件、逐 hunk 与 KV 成员函数的覆盖程度。KV cpp 全部 hunk 已分类，KV cells 与 hybrid-idx 全部新增块已阅读；XKV 大型事务/序列化、FlashPrefill 大型 planner 按接口、状态阶段和依赖审查，没有伪称逐行证明所有扩展实现正确。测试证据与静态审阅分开。

## 一、三个核心功能的结论

| 功能 | 上游现状与真正新增 | 是否保留 | A / B / C 最小边界 |
| --- | --- | --- | --- |
| Prefill / Decode 分离工作区 | 上游已有 CUDA KV-stream phase arena，已有 `llama_set_decode_phase`。fork 另加通用 scheduler/allocator pool 与相位 trim，试图让 CPU/Vulkan 及 target/MTP 共用内存。两者不是同一功能。 | 保留通用相位释放的目的；先实现单 context 生命周期 API。跨 context 共享池另列，不作为相位分离的前提。 | A 拥有同步、图失效、allocator 销毁与重建；B 只持有相位策略值；C 在已知 PROMPT→GENERATION 边界显式调用 A。新的 `llama_memory_release_compute(ctx)` 返回 bool，不能修改旧 `llama_set_decode_phase` 的语义。 |
| 统一 KV 池 | `kv_unified` 和统一 sequence 存储本就存在。fork 真新增 `n_ctx_kv`：物理池容量独立于单序列逻辑窗口 `n_ctx_seq`，再配自动 fit、容量观测及 recurrent physical-slot 分离。 | 应保留，不能因其在巨大缓存文件中出现就丢弃。 | A 增加物理容量输入、验证和查询，并把现有 KV ctor 的 size 参数接到该物理值；B 是纯值容量规划；C 负责 CLI、请求 admission、逻辑窗口。无需为普通 dense pool 移植 XKV/TriAttention。 |
| Prefill 优先 | 当前两份内存/common diff 不能证明 server queue 的完整优先级行为。`n_parallel_pp=1` 是 PP 并行度/内存预留限制，不能冒充任务调度优先级。 | 应保留真实调度语义，由 server 审计补证。 | 队列策略适合独立 B，C 调 A 的 decode；无需在 A graph/allocator 内硬编码请求优先级。需要定义 prompt 与 generation 同时到达、continuation、饥饿保护和 admission 边界。 |

### 1.1 相位最小 seam 的精确约束

1. 调用线程必须独占 context；API 先同步这个 context 所有 scheduler 后端，不能仅依赖“有无已排队 token”的统计值。
2. 清空 `gf_res_prev` 和 `gf_res_reserve` 内部图状态，同时清掉 scheduler 的 tensor/backend/copy 分配缓存。保留 graph-result 对象，以便正常下一次 build 使用。
3. 仅重新创建 `ggml_gallocr_new_n(sched->bufts, sched->n_backends)` 并释放旧 galloc，保留 scheduler 本身、MoE cache session、模型、KV、host output。下一次真实图重新分配，不再立刻 reserve 最大 PP 图。
4. CUDA 共享 phase arena 的 allocator 使用外部布局；应在任何状态变化前返回 false。训练图/optimizer 持有 allocator tensor 的模式同理拒绝，除非补足专用生命周期协议。
5. 返回成功表示 allocator 所拥有的 buffer 已释放；不能等同于驱动物理空闲显存增加。用 `ggml_backend_sched_get_buffer_size` 测拥有容量，以同一 context 下一 decode 的 token/logits/KV 不变测语义。
6. 不在普通 phase release 中加跨 context pool、generation registry、sealed capacity 或 tensor 指针登记。那些机制有独立同步和失败原子性成本。

### 1.2 n_ctx_kv 最小 seam

`src/llama-context.cpp:962,1053–1087` 显式区分物理输入与逻辑窗口；`src/llama-model.cpp::create_memory`（`audit-small.diff`，约 2660–3080）把 attention 各 ctor 已有的 KV size 参数从 `n_ctx_seq` 改为 `n_ctx_kv`。普通 `llama_kv_cache` 已接受该 size 参数，因此关键新增不是重写整个 cache，而是容量解析和传参。

保留旧零值兼容；对 unified dense attention，物理容量至少覆盖一个完整逻辑窗口；256 对齐必须做溢出验证。non-unified 使用旧布局并报告输入是否忽略。旧 fork 在 `llama-model.cpp:2794` 明确拒绝 DeepSeek4 动态 unified capacity；不能在没有 architecture 测试时去掉。TriAttention 允许物理小于逻辑，需要另外的有损回收语义，不属于最小 dense pool。

## 二、必须剔除的旧合并残留与不安全路径

| ID | 证据 | 直接行为 / 风险 | 迁移决定 |
| --- | --- | --- | --- |
| R01 | `src/llama-context.cpp:2092–2117`；回退路径 `2320–2328` | 创建 shared scheduler，给 memory 写 scheduler 指针后，又无条件 `sched.reset(ggml_backend_sched_new(...))`。前者被销毁，shared pool 失效；memory 可能留存悬空 scheduler 指针。 | 不移植这段组装；新单-context API 无额外 scheduler 指针回注。 |
| R02 | `ggml/src/ggml-backend.cpp:2306–2308,2326` | 条件 shared/ordinary galloc 初始化之后再次无条件普通 galloc 赋值。首次 allocator 泄漏，shared registry/ref 生命周期不闭合，功能也失效。 | 不基于此实现宣称共享池已可用；重新实现小 seam。 |
| R03 | `ggml/src/ggml-alloc.c:828–834` | `GGML_GALLOC_KEEP_CHUNKS` 默认 1，trim 默认完全跳过。因此仅有相位代码不证明默认释放 PP 高水位。 | 显式 API 和明确策略，不能用隐式环境默认掩盖行为。 |
| R04 | `ggml/src/ggml-alloc.c:836–851,1299–1339` | 先 free 原 chunk，再尝试替代分配；失败时 generation 没有随破坏动作递增。已绑定 tensor 会置空，但依赖 generation 的 allocator 有失败状态复杂性。 | 如果将来保留跨-context pool，需失败前预分配或在失效动作时立即递增 generation，并验证回滚。 |
| R05 | `src/llama-context.cpp:935–939,958–959` | 新 output 最大值限制随后被旧赋值覆盖，内存 fit 与真实输出预算可能不同。 | 以行为为单位重新连接，不拷贝整个构造块。 |
| R06 | `common/fit.cpp:3536–3558` | 注释称最终真实 dry probe，实际调用 metadata probe；最多三次调整后，没有“最后一次成功”检查即返回成功。 | 新 solver 必须有显式 ACCEPT 状态与成功证据，不能按循环耗尽视作成功。 |
| R07 | `common/fit.cpp:3629–3636` | fallback 检查 `n_probe > n_min`，但未检查较大 probe 失败后 `data_probe` 是否为空便按 device 索引访问。 | 保留有界搜索目的，丢弃该重复分支；probe 失败应是具名结果。 |
| R08 | `common/fit.cpp:3540` | 两个 `uint64_t Shmem` 直接相减可能下溢；同函数其他位置已有 clamp，合并后不一致。 | 使用有符号/饱和差值；全局 host pressure 仅为提示，不能代替归属清晰的容量测量。 |
| R09 | `common/fit.cpp:1407–1425,1451,1668` | global logger 暂时指向栈对象，恢复和 model/context 释放主要依赖正常末尾；context 创建失败等异常路径有泄漏/日志恢复风险。 | C 的 probe 使用局部 RAII，必要 logger scope 同步；B 不持有 A 指针。 |
| R10 | `common/fit.cpp:941–973` | 静态 metadata-model 缓存无锁、key 不含文件版本及全部 placement 参数、生命周期原始指针。并行 probe/模型替换不安全。 | 改 C 中单会话拥有的 metadata cache，完整 key，明确析构。 |
| R11 | `ggml/src/ggml-backend.cpp:2440–2468`；`llama-context.cpp:3707–3738` | release_buffers 自身不 fence；相位切换调用点也没有局部明确同步。锁只能防 CPU 并发，不等同 GPU 已完成。 | 新 API 自带同步；不依赖调用链偶然同步。 |

这些是代码可直接支持的风险，并非声称已经触发了崩溃。旧注释中的性能数字不作为本审计的性能证据。

## 三、普通小功能：逐项保留和分层

| ID | 功能、行为与证据 | 上游差异 | 独立边界与建议 |
| --- | --- | --- | --- |
| M01 | 独立 PP 最大序列数：`common/common.h:499`；`common/common.cpp:2238`；`src/llama-context.cpp:2069–2083`，PP synthetic shape 与 TG streams 对齐。 | 新增 `n_seq_max_pp`，上游仅 n_seq_max。 | 保留为 C 的 PP concurrency/batch limit；若 A 最大 reserve 需单独 PP shape，再登记很小的参数 seam。不能把它当 prefill-priority。LCM/加法需防溢出。 |
| M02 | TG reserve 结束态：`llama-context.cpp:2340–2374`，no_alloc 按 PP/TG 每 device max 计量，实际尾部保留 TG graph。 | 旧上游尾部再次 reserve PP 最大图。 | phase release 与启动 reserve 是两件事。先保留实测 PP 峰值预算，C 显式相位释放；不谎报内存峰值。 |
| M03 | `llama-context.cpp:2412–2430` 保留无条件 scheduler fence，新增 predefined-hidden executor fence 和统计 early-return。 | **普通 scheduler synchronize 原本上游已有**（old-side 1439–1444；干净 A 完整体再次确认）。不能将新增注释误认为修复；新增仅是隐藏执行器额外生命周期。 | 新释放 API 复用上游 synchronize。若后续引入另一个 hidden executor，要明确扩展 fence 范围。 |
| M04 | physical KV CLI：`common/arg.cpp:1777–1800` 增 `-kv/--kv-size/--total-kv N|auto`，`common/common.h:490–492`。 | 新增容量控制，非新增 unified storage 本身。 | C 自己 CLI，B 参数验证/规划值；尽量不改 A common。 |
| M05 | physical KV 查询：`include/llama.h:881` 和 `llama-context.cpp:1096–1103`。 | 新增观测。 | 最小 A getter；C 清楚展示单序列逻辑窗口、物理池 token 数、seq 并发。不能只显示 n_ctx。 |
| M06 | ordinary recurrent 物理容量独立逻辑 sequence ID：`llama-memory-recurrent.cpp:38,54–79,671–895`；include/llama.h:626。 | 新增 `n_seq_recurrent` 和独立 tails；原有 n_rs_seq / replay 机制已经上游有。 | 应单独保留，不以 RERoT grouped layout 为前提。需 A metadata+ctor seam，B 规划物理 slots，C admission。纯 dense KV 最小补丁可暂不包含它，但必须在功能清单单列。 |
| M07 | recurrent metadata COW 和 `find_slot` preflight：`llama-memory-recurrent.cpp:370–399,743–753,815–823`；prepare snapshot `645–666`。 | COW 思路已有；新的 tails 映射、低物理容量 preflight/恢复是新增。 | 物理容量小于逻辑 ID 空间时必须带上。测试容量刚好/不足、共享 parent 后 write、prepare 失败 metadata 不变、parked ID。 |
| M08 | 内存 admission 纯值快照：`llama-memory.h:156–211,275–294`。逻辑/hot/factor/workspace 用 token-bound 最小值；recurrent 满只阻止新序列，不能阻止已有序列 continuation。 | 新增。 | 非 XKV 情况只需 capacity/used+recurrent gate。A 导出 owned POD；B `decide(snapshot, request)`；C 执行。保留区分 token 和 sequence 的语义，避免 max/min 混单位。 |
| M09 | hybrid getters 与 admission 合并：`llama-memory-hybrid.cpp:163–226`，iswa 对应 wrapper。 | 新增公开可观测容量。 | 小 A 虚函数+wrapper seam；不把独立纯值 planner 放进 A 内存类。 |
| M10 | 内存 clear 可报告失败：`llama-memory.h:337–349`；hybrid `233–237`。 | 新增 try_clear；原始 clear void 无失败通道。 | 单列为缓存事务需要的能力；普通内存不是必需。默认只 catch std::exception 不证明 C ABI 任意异常都安全。 |
| M11 | hybrid 删除顺序改为 attention 先成功再 recurrent：`llama-memory-hybrid.cpp:240–247`。 | 新增，针对 bounded attention 可能拒绝释放。 | 注意只避免 attention 失败的半更新，不能保证 recurrent 随后失败时整体事务回滚。必要时 validate 两边再 commit，不应宣称原子。 |
| M12 | component-selective rm/cp：`llama-memory.h:356–389`；hybrid `255–277`。 | 新增 attention / recurrent 分别操作。 | 是保留混合模型 fork/park 功能的 A seam；普通统一池不依赖它。B 只能表达操作计划，A 内部完成模型一致性。 |
| M13 | owned MTP 工作区：`common/speculative-mtp-workspace.h:12–188`，所有序列共用一份 rows×width 验证矩阵，只留每序列 pending/seed 两行；range 不重叠、accept 更新 commit prefix，reset 不 shrink。`194–240` 同时拥有 token+embedding batch，禁止 copy，返回 borrowed llama_batch。 | 两个独立新 helper；backing buffers 生命周期价值与 direct-hidden backend 无关。 | workspace 用 int32 token/position 可做纯值 B；llama_batch view adapter 留 C。host_hidden=false 只保留 row metadata。不可对 borrowed batch 调 llama_batch_free。该 240 行文件已完整阅读。 |
| M14 | shared-pool retain/sealed policy：`ggml-alloc.h:64–74`；`ggml-alloc.c:1282–1297`。 | 新增区分 reset logical plan、retain physical capacity、禁止增长。 | 本质有价值，但本轮单 context phase release 不需要。未来 pool 单独提交，清楚公开 fence/owner/fail contracts。 |
| M15 | scheduler owner fence：`ggml-backend.cpp:2178–2188`；pool generation `2521–2569`。 | 新增 target/MTP scheduler serialization。 | 只在跨 context 共用物理 chunks 才需要；不能伪称 B 独立 pool 时实际调用 A allocator 或让 A 反调 B。 |
| M16 | fusion 额外活跃依赖：`ggml-backend.cpp:1696–1780`，backend optimization 后收集 add_alloc_dep 并加入 no-payload nodes。 | 新增自定义 backend lifetime seam。 | 如果保留相应 custom kernels，不能只移植 kernel 而丢掉活跃依赖。A 中小显式 hook+有寿命的声明；B 不通过私有 graph 伪装独立。普通相位释放无需它。 |
| M17 | 异步 host-input snapshot：`ggml-backend.cpp:464,1943–1953`。 | 新增可选 backend capability，契约为返回前持有 host bytes。 | 可独立评测小性能项，默认 fallback；不能将普通 async copy 宣称已拥有输入。 |
| M18 | device memmove batch preflight：`ggml-backend.cpp:663–722`。 | 新增多region越界/溢出验证，host memmove fallback。 | 仅 cache compaction 等需用；从 Tri 大功能中单列可复用 primitive。driver 复制不是简单 CPU memcpy，可单独 capability。 |
| M19 | `GGML_DISABLE_INPLACE` 可关 in-place，view parent ownership 检查：`ggml-alloc.c:947–980`。 | 新增调试/兼容行为。 | ownership 修复可能独立有价值；env 开关适合保留到诊断工具，不能替代精确 alias lifetime 验证。 |
| M20 | 输出上限与 graph hash 空间：`llama-context.cpp:935–959`，`ggml-alloc.c:1153`。 | 新输出预算被覆盖；hash 安全余量增加。 | 需要具体 graph-size 压力证据再保留，不机械带入 2x+512 魔数。 |

## 四、auto-fit 是独立功能，但旧求解器不宜原样移植

### 4.1 应保留的普通能力

`common/common.cpp:1451–1528` 在 `--total-kv auto` 保持逻辑 context 不被 fit 自动削短，以 metadata / dry probe 搜索物理 pool，并纳入 MTP 额外 context。`1405–1428` 提出预期活跃长度约为逻辑窗口的一半，得到 `round(2*physical/logical)` 的 concurrency 目标；它是统计假设，不能作为所有请求都能填满 context 的保证。`1818–1823` 将求解后的 slot 数和 PP 最大并行 1 传回参数。

`common/fit.cpp::common_fit_recurrent_capacity:3693–3886` 则独立搜索 physical recurrent slots，不需要 RERoT brains/hands。这个 ordinary 功能不能随着 grouped layout 一起归档。

最小干净结构是 **B 的有界求解状态机**：输入设备容量快照、逻辑窗口、alignment、每次 A 测量返回的资源值；返回 `next_probe`、`accept` 或 `reject`。**C** 创建/销毁 A probe context 并提交测量值。B 不收 A callback、不包含 ggml/llama header。A 最多增加可靠容量报告和必要 dry-measure seam。

### 4.2 应拒绝照搬的预算逻辑

| 证据 | 问题 | 要求 |
| --- | --- | --- |
| `common/fit.cpp:1467–1518` | dry probe 触碰/清空所有 KV 后 decode 两个 dummy token；未检查 decode 返回值，未在内存读数前明确 fence。 | 一次测量要标明是否 metadata / committed / active；检查 decode 状态并同步后观测。 |
| `llama-memory-recurrent.cpp:583`、hybrid `416–418` | `materialize` 是清 buffer，不是无副作用的 residency 查询；用于 live context 会破坏状态。 | 若保留，限制到新建 probe 的明示生命周期；不能作为公共任意时刻“刷新显存读数”接口。 |
| `common/fit.cpp:1542–1587` | 当实际计费只有一个 device 时按 `model->dev_layer` 层数重新摊派；有异构层/CPU memory 时可能与真实 buffer placement 不符。 | A 返回拥有的 per-buffer byte 分类；C 不用层数猜覆盖实际证据。 |
| `common/fit.cpp:2949–3127` | metadata probe 的可行性决定忽略 model/headroom，且允许 `free+512MiB` slack，注释却称 resident probe。 | 选择一致预算口径；模型是否已扣除必须是显式字段，不能在不同阶段来回解释。 |
| `common/fit.cpp:3144,3166` | free_base 在 devs 未填入时初始化，实际 probe 后又 assign 0；注释称“保留 probe 前 baseline”不成立。 | 测量快照必须具备采样时刻和用途。 |
| `common/fit.cpp:3331–3343,3835` | 搜索斜率/最终 recurrent 可行性对 compute、headroom、model 的计入方式不一致；metadata free 差常为零。 | B 统一全量预算模型，不能对不同候选采用不同口径。 |
| `common/fit.cpp:3190,3443` | 全局 Shmem/GTT 阈值保护混入其他进程影响，固定 2GiB/8GiB/64MiB 适合单机实验但不可无条件泛化。 | C 的平台观测保留为带来源的 advisory/abort policy，B 输入明确，不硬编码全机事实。 |

Linux `/sys` 物理 VRAM/GTT 读取（`common/fit.cpp:1000–1118`）可作为 C 平台 adapter 保留；driver budget 与 resident bytes 应分别命名，并支持无 sysfs 回退。XKV workspace、factor store、FlashPrefill packed cache 的**恰好一次**预算原则值得保留，但各扩展的具体公式不应该成为普通 KV fit 的必选依赖。

## 五、扩展专属项与覆盖边界

- RERoT grouped recurrent layout（`llama-memory-recurrent.cpp:148–181,903–1624`）把共享 brain S 与 per-hand D 分开，D 使用 F32 有独立数值目的；它不是 ordinary recurrent 物理容量分离的必要部分。需把参数、layout、state IO、backend ops、commit frontier 作为完整扩展验证。
- XKV hot-slot 预留和 `postcompute_success/failure`（hybrid `132–144,519–545`；iswa 对应实现）是异步执行成功后发布/失败回滚协议，不能将“exactly once forwarding”当作多内存域原子提交。普通 dense pool 不依赖它。
- sparse positions 与 reclaim（`llama-memory.h:526–538`）是有损 KV 回收 API；调用方不能按 pos_min..pos_max 推定所有 token 都在。普通物理/逻辑容量分离保持 dense 语义。
- `src/llama-kv-cells.h` 的全部新增块已阅读；功能细分和 copy/identity 风险见第七节，不能因为注释声称 stable identity 就假定实现满足。
- `src/llama-memory-hybrid-idx.cpp` 的全部新增块已阅读；raw indexer key、pooled dirty watermark、sparse reserve、状态尾部和预算分别登记，见第七节。
- `src/llama-kv-cache.cpp` 另由完整 `audit-kv.diff` 审查。抽取出 167 个完整成员函数，39 个忽略注释/空白后的代码与上游相同，16 个有完整 old-side 对照且改变，其余 112 个是新增或 old-side 定义未在差异范围内。分类器保留这个区别，没有把巨型移动 hunk 的每条 `+` 当成自创功能。构造/片段 hunk 另行检查，不包含在这 167 个完整函数数目里。
- Prefill 优先的最终证据应来自 server scheduler agent；本报告只确认内存接口适合与 C 调度分开。

## 六、验收协议

**相位释放**：真实 A CPU scheduler 大图分配→compute→记录 owned bytes→release→owned bytes 0→小图重新 build/compute且输出正确；同一 scheduler 再大图可增长；重复 release 安全；外部权重/input buffer 不被释放。llama 真 GGUF 对照同 prompt 和每个 continuation token/logit；记录 release 前/后/下一 token allocator bytes；测试 PP→TG→新 PP。CUDA phase arena / optimizer 模式返回 false 并保持状态。GPU 回归至少单独验证已启用后端的异步同步与图缓存行为，不用 CPU 通过代替。

**统一池**：固定 logical window，改变 physical capacity，验证 getter与分配尺寸；多 seq 共用池容量守恒、部分 seq 删除后复用、边界拒绝无状态残留、dense 不允许 physical<logical；未指定参数与旧 A parity。hybrid 若引入独立 recurrent slots，测试逻辑 ID 大于 physical slot 数、COW、prepare失败回滚、满 recurrent slots仍可继续已有序列。

**fit**：有界次数、失败 probe 无数据时不索引、失败不 ACCEPT、单调性违反可报告；每设备预算统一，reserve恰好一次，noGPU/fallback明确；真实 probe 以 RAII 释放模型/context/logger，失败decode不接受。报告峰值预算与当前 owned/resident 读数分别呈现。

**分层**：A 补丁逐个登记 API 契约和不可外移的私有生命周期原因；B 独立编译且不含 A header、symbol/callback；C 组装真实 A+B。隔离检查允许具名 A seams，不再把“零 A 改动”当成放弃功能的理由。

## 七、KV / cells / hybrid-idx 补充审查结果

### 7.1 确認哪些其实已在上游

KV `find_slot` 的完整 old/new 函数为 198/199 行，唯一增量是结果初始化的 `hot_idxs`；普通统一空槽搜索与 sequence 布局均已在 A。`get_size`、`get_n_kv`、`get_n_kv_pos_contiguous`、`kv_stream_resize_pool`、`kv_stream_adapt`、`set_input_kq_mask`、`build_rope_shift`、普通 KV state data 写出等完整函数代码相同。MTP cache 的 `other`/shared `v_cells_impl`、目标比 draft 默认 capacity 小时跟随目标的保护、`n_layer_all` 的 nextn 层覆盖也属于差异中的上游 context，不算新 fork 能力。

`hybrid-idx` 的 PLE n-gram history 及其原有 sequence rm/cp/shift/persistence 是上游已有。本 fork 新增另一路 KV token history 与 pooled indexer 缓存，不能把前者已有支持也算作自有新增。

### 7.2 进一步拆出的独立小功能

| ID | 原文件 / 行号 / 符号 | 行为与保留判断 | A / B / C 依赖 |
| --- | --- | --- | --- |
| M21 | KV cpp `4645–4678`，`get_kv_capacity/used/seq_used`；cells `867–871,1306–1323` | 总使用量数物理 cells，per-seq 使用量由维护计数直接查询；共享 prefix 在总量中计一次、在各 seq 中均计一次。保留普通观测价值。 | A-owned counters/query；B admission 接收数值，不能把每 seq 数量相加冒充总物理占用。 |
| M22 | KV cpp `318–330`，`llama_turbo_layer_adaptive_mode` | 新代码声称按 cache instance 解析 V 分层精度，避免进程级配置串扰。但当前差异只见新变量用于日志；旧 `kv_adaptive_mode` switch 仍保留，不能声称实际路由修复已接通。 | 纯类型选择规则可在 B，真实 tensor ctor 的 type 连接须 A seam。由类型/kernel 审计确认，不盲目移植日志块。 |
| M23 | KV cpp `2473–2644`；cells `38–50,547–638` | Native compaction：used cell 连续段形成 dst≤src 的 pack plan；K/V按行/stride分组，backend memmove 后再提交 metadata；转置量化 V 明确不支持。普通 dense cache 也适用，不仅有损回收。 | B 可生成纯 move plan；A 必须拥有 layout/view、buffer mutation 与同步。当前统一池基本功能不需要先实现压紧。跨设备失败并非自动可回滚。 |
| M24 | KV cpp `1400–1404`，`seq_div` | 拒绝 d≤0 避免除零/负位置；独立小输入保护。保留价值低成本，但最好明确 API 错误，不仅 void 静默返回。 | 小 A 参数验证，与 XKV 无关。 |
| M25 | KV cpp `7392–7493`，`get_k/get_v` | 新添 stream span clamp 防 view 越界，但 s0 非法时改为0、ns裁短会改变 attention 数据语义。不能当“安全 bounds fix”直接带走。 | 要在 slot planning 找出错误源并拒绝；普通 A unified 分配未改，不需要该掩盖。 |
| M26 | KV cpp `7727–7862` | 固定 capacity graph 的 inactive index tail 重定向到空 cell，避免 padded `set_rows` 写坏 slot0/history；无空 cell 则拒绝。 | 只对 padding/capacity replay graph 有意义；需要 physical reserved sink slot 或明确 free-cell preflight。bounded XKV 的 logical idx 不一定是 hot row，不能复用此简化映射。 |
| M27 | KV cpp `3394–3424,10551–10565` | MTP read-only graph 预先拿到 occupancy、max-position slot，不依赖 `apply()` 修改内存后才得 shape。 | 小 A 只读 snapshot 能力；C host-MTP 编排可以利用。普通 decode 不必带上 direct-hidden executor。 |
| M28 | cells `18–24,760–795`；KV cpp `4198–4215,8259–8308` | cell 记录 token，按 position≤p 取 predecessor（同 position 取最高 physical idx）；embedding行用 PLE image/eos pad。支持 KV 驱动 PLE n-gram。 | 不是纯 host-token 文本缓存等价物：同位置、sparse/vision语义必须保持；A metadata/state seam，B可做纯查询。既有 PLE host history 迁移需单独检验。 |
| M29 | KV cpp `10197–10205,10258–10265` | per-seq state restore 后回填保存 ext，避免不使用 M-RoPE 的 indexer mirror 丢 x/y；has_cell_ext 扩展到 PLE token。 | mirror ext 修复可单列。扩大 ext 的序列化结构需要版本/兼容检查，不能凭新增字段静默改变旧state协议。 |
| M30 | KV cpp `10518–10526` | reserve mask span 改 min(pool,n_ctx_train)，意图把 storage capacity 与 scratch 分开。**不建议保留该固定 clamp**：普通 unified attention 仍按物理占用 span 建 view，多序列即使各自窗口小，也不代表图只扫描一个窗口。 | 当前 A reserve 保留保守形状；phase release 下一实际图自行分配。不能承诺 scratch 被 logical window 封顶。 |
| M31 | cells `55–257` 及各 mutator；KV cpp `4886–4944,6166–6254` | opt-in (object stamp,generation) 使派生布局重用；epoch不按“正好+1”推断，copy新stamp、move转移。 | 小 A mutation-version seam + B owned layout cache + C snapshot；目前身份实现有 R12/R13，需修正后才能用。 |
| M32 | KV cpp `4896–4923,5130–5400,6041–6164` | ordinary FlashPrefill fragment planner按stream、BN块、查询相关membership拆分；允许文本等价IMRoPE `[p,p,p,0]`，真正2D/ALiBi/不支持SWA回退。与 RERoT teacher planner 分开。 | 基于 owned cell/query POD 可提纯 B；A 提供版本化快照与内核执行 seam，C做路由。原函数直接读私有 cells，不能直接改目录冒充独立。 |
| M33 | recurrent cpp `1013–1127` | 单cell非grouped缓存用一次buffer_clear替代逐层逐snapshot set；一般路径合并为strided set_2d并在前后fence。 | 有独立生命周期/减少提交价值。清除策略留A，B只表达释放哪个逻辑资源；无需引入整个 grouped layout。 |
| M34 | hybrid-idx cpp `58–80` | raw indexer K 不做 RoPE shift；Turbo K/V请求时 indexer强制F16以使 get_rows 在支持设备执行，attention KV仍保留请求类型。 | 两个模型专用小修复，与用户普通统一池无关；应保留候选并用QSA模型校验，不能覆盖所有cache类型。 |
| M35 | hybrid-idx cpp `89–168,686–719,982–1032` | 每层每position-block常驻F32 pooled K，按owner buffer type分配；partial块加备用行+padding dustbin；complete-prefix watermark只重算dirty块。 | 缓存实体与device writes必须A拥有，B可做dirty-range计划；C调度不能保存裸physical index跨变动。需修复预算、all-seq失效和跨seqowner问题后保留。 |
| M36 | hybrid-idx cpp `636–659,1182–1200` | full restore清watermark，partial restore按live pos clamp；graph失败使pooled cache失效，避免信任未提交行。 | 必须与M35整体同提交，不能只移植缓存速度路径。base failure短路导致子ctx failure不调用也需审查。 |
| M37 | hybrid-idx cpp `1130–1138,1237–1267` | reserve context也创建indexer child context与stream数，确保预留真正的sparse graph；dirty表在真实restore后按需要增长，并处理M-RoPE live-cell上界。 | 模型专用A graph-shape seam；是独立于新内存pool的首请求分配修复。 |
| M38 | cells `812–835`；KV cpp `3626–4147,10680–10766` | 多引用cell完整membership snapshot，overwrite victims延迟至compute成功后释放，失败对新entries及victims回滚。 | 独立事务语义，普通dense池无须此store protocol；确需时要A-owned transaction message而非 A→B callback。 |

### 7.3 补查发现的风险（不移植）

| ID | 直接证据 | 风险及边界 |
| --- | --- | --- |
| R12 | cells `130–219` 的四个 copy/move 特殊成员没有处理 `payload_id/storage_generation`；字段 `1278–1279` | copy产生空身份、assignment保留目的旧身份；prepare/compaction中的值复制可能丢失payload归属。**明确遗漏**；不影响当前未移植cells扩展的最小seam。 |
| R13 | cells `1163–1178,1289–1294` | restore可把counter推进MAX，再经fetch_add wrap；MAX-1状态即可构造，不需要运行2^64次。与“不复用”契约冲突。stamp new_stamp在wrap后也不真正sticky；seq_restore未独立bump。 |
| R14 | KV cpp `8443–8585` | 新 `if(is_turbo_k)` 的inverse-WHT/RoPE/set_rows修复前仍有上游turbo `continue`，新分支不可达。独立host `shift_turbo_keys:6666–6728` 存在不证明执行路由接通。 |
| R15 | hybrid-idx cpp `148–160,322–331` | 新 pooled buffers 未计入 memory_breakdown；fit可低估常驻缓存。当前最小seam不引入这个cache。 |
| R16 | hybrid-idx cpp `695–711` | pooled_rm 对 seq_id<0 写 map[-1] 而不清每个真实seq watermark；all-seq removal可留下可信度过期。应统一all-seq invalidation语义。 |
| R17 | hybrid-idx cpp `585–633` | 写出删PLE history尾部、正常读取也删，但 catch cleanup后调用reader。可能是向KV-token表示迁移的未清理残留；异常路径不该继续读旧尾部。不能借机把上游已具备的PLE历史能力丢失。 |
| R18 | recurrent cpp `1893–1915,2176–2215` | 原有 PLE P tensor 行读写后又新加一套逐层 has_p/type/row-size/data块；含P的state重复写/读，无P也加header。需删重复或明确新格式，不能当独立“新增PLE支持”。 |
| R19 | KV cpp `1256–1273` vs cells `677–709` | seq_keep 预清单仅选“不含目标且seq_count==1”，但真正cells.seq_keep会清所有不含目标的cell，包含多引用其他seq的cell。bounded pool/store有残留风险；这是 last-reference 释放清单不一致。 |
| R20 | KV cpp `1328–1376` | negative-shift removal preflight失败只log，随后仍修改cells；release失败也只log。注释承认延续leak行为，与前后“fail closed”承诺不一致。 |
| R21 | recurrent cpp `1200–1237` | release_episode 清除一个seq引用后，无论cell是否仍被其他seq引用，都调用clear_hand_row；若共享tail跨存活引用，会清掉剩余使用者的状态。需要合法共享图验证后限定clear条件。 |
| R22 | KV cpp `1475–1488` vs clear `988–998` | materialize直接清所有ctx buffer但不恢复同buffer中的turbo rotation/InnerQ常量。即使probe新建也不能将后续dummy decode视作正确模型计算的证据。 |
| R23 | hybrid-idx cpp `89–168,691–719,1000–1031` | pooled数据按position row共享，watermark按seq分别保存；仅“single stream”并不等于“single logical sequence”。代码未见在切换写入seq时验证pooled row内容owner。多seq interleave需独立准确性测试，不能默认其与unified多并发兼容。 |

### 7.4 巨型 KV hunk 中的扩展能力分解

下列均已定位入口、输入/输出状态与私有依赖；这一级别是**功能与边界审查**，没有声称完成底层codec/backend每条分支的验证。

| new-side 范围 | 能力 | 迁移归属与检验点 |
| --- | --- | --- |
| `202–275,698–719,1491–1547` | bounded hot physical rows 与 logical cells 分开；segment+batch headroom，RAII预留 | 扩展专属A内存布局；和用户普通 unified physical-pool/logical-window 分离不是同一个开关。不能混淆 hot row 与logical index。 |
| `1700–2371` | bounded context shift：按hot/factored分类，landmark COW，device staging，commit/rollback | 单独事务扩展；需要coordinator、factor store、device执行器、RoPE精度共同契约，不是几个普通seq_add wrapper。 |
| `2647–3387` | 校准驱动KV回收、owner去重、required_free按物理数、保护public/pending分类、回收后压紧 | B可做纯评分/eviction计划，A通过具名事务提交；需要量化误差与缓存准确率证据后决定保留。 |
| `3626–4147,4493–4635` | overwrite victim生命周期、last-reference释放、pool+store precommit、快照回滚 | 无法纯外部decorator保证原子性；先修R12/R19/R20，再考虑作为集中登记A seam。 |
| `4886–6259` | mutation tracking、ordinary fragment合法集、text-IMRoPE gating、RERoT单次resident scan、多view共享、owned layout缓存 | ordinary与RERoT planner可分别抽取B；C拿A snapshot并下发layout。不能让B持私有llama_kv_cells引用。 |
| `6284–6385` | rotation参数/上下文查询、pre-RoPE capture资格检查 | 资格需committed、nonempty、属该seq、排除private/pending；属于数据导出seam，非任意内部缓存读取钩子。 |
| `6397–6664,6730–7329` | episode/run发布与重分类、archive keeper引用、reader frontier/world增量、active episode禁止普通state保存 | 多运行可见性扩展，A-owned cell标签与B owned逻辑计划通过C协调；不把它当普通KV池基础。 |
| `8917–10154` | config/模型内容/校准内容fingerprint、per-seq裁切factor segment、批量readback、device-owned restore、ordinal重绑定、预算上限、decoded tile invalidation | 可以把纯state envelope/validation/plan变为B，但A必须拥有实际device resources与原子安装。模型完整内容hash不是metadatahash；R13 counter仍需修复。 |
| `10551–10766` | read-only MTP shape预设；hot_writing→hot_committed；成功后释放victim与失败回滚 | 两个不同功能拆开。MTP shape小seam可独立，store事务不必一起迁移。 |
| `11017–11205` | shared-store family共享一个runtime/coordinator，per-device执行器复用，admission/runtime快照，单segment maintenance迭代 | C可以循环maintenance消息；A保持设备、cache ownership；B做有限策略，不通过callback回注私有cache。 |

## 八、本轮实际 seam 的审查状态

已向根代理交付 `phase-memory.patch`，根代理确认已应用且 CPU A 增量编译通过。随后根代理报告真实 CPU C phase 对照通过：on/off 各执行 3 个请求、22 个 batch、145 行，输出一致；验证 logical 512 / physical 1024，并观测两次实际释放到 0。独立低层 allocator 测试的完整日志与最终命令由根代理汇总。补丁只新增 `llama_memory_release_compute`、`llama_memory_compute_bytes`、必要的context方法与 `ggml_backend_sched_release_buffers`；没有移植旧共享pool/cells/XKV类。测试代码使用公开ggml API且 Release 下检查仍生效。

当前 safety contracts：同context独占；先拒绝metadata-only/optimizer/CUDA joint arena；同步完整A scheduler；同时重置prev/reserve图；只重建galloc；不给`set_decode_phase`加隐藏副作用；getter按buft去重只数当前owned bytes。`alloc_splits` 已确认冷galloc会按实际graph reserve，MoE provider session不重建。encoder-decoder 的 cross 所有权已通过干净 A 片段确认：`src/llama-graph.h:79–104` 的有效成员为 `vector<float> v_embd`、`vector<set<llama_seq_id>> seq_ids_enc`，tensor 指针仅是注释；`llama-context.cpp:2521–2560` 在同步后 memcpy 到所拥有的 host vector。此处没有因 graph reset 而悬空的有效 cross tensor 成员。静态所有权检查已关闭，仍不以它代替对应真实模型回归或声称全部 architecture 已验证。

物理KV池可以比单序列窗口大，实际attention physical span和compute峰值也可以随占用增长。这是当前干净实现的真实边界，不应用旧 `n_ctx_train` clamp 隐藏。用户要求的相位释放仍独立成立：主动放掉上一次图的高水位，随后由实际图需求再分配。

## 九、后续 backend 范围补审

`audit-backend-findings.md` 第八节补全了 `llama-kv-cache-iswa.cpp/.h` 的14/8个修改hunks、DSV4三个hunks以及DSA/MSA统计转发。它确认base/SWA的cparams分配、base容量统计、bounded-hot reservation和postcompute生命周期，也明确 `try_clear/seq_rm` 的base-first短路不保证失败时两侧事务回滚；tag/view第二侧失败后的clear也不等于恢复原状态。机器证据在 `audit-backend-coverage.json`，本报告coverage的 `supplemental_and_gaps[].followup_review` 指向对应后续状态，不把此前范围缺口篡改为一开始已经深读。
