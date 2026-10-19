# 逐文件与小功能审计

本次审计以固定提交为边界，先枚举全量路径，再核对实际改动、函数、状态字段、CLI 参数、失败出口、文件格式与测试判定。保留决定按具体行为作出，不能由 TP5、RERoT、XKV 等大模块名称代替。

当前迁入状态见 [MIGRATION.md](MIGRATION.md)，实际运行资格见 [VALIDATION.md](VALIDATION.md)。本页和分册用于回答“在哪里看过什么、还有什么价值、哪些证据不能使用”，不替代构建或运行验证。

2026-10-19 的 [剩余重写候选](REWRITE-CANDIDATES.md) 基于用户新 TP5 路线重新去重，记录具体用途、旧入口、证据缺口、原始来源和未来验收。所有新增候选仅为文本。本次还纠正 S03 的重复 uint64 分支、M29 的原有 QSA ext 恢复和普通输出容量归因；历史分册保留原阅读记录，以该文的纠正说明回查。

## 来源和分母

| 对象 | 固定身份或范围 |
|---|---|
| A | `cda0f9256665845cd9d6f3a0176880883f653a3d`，AtomicBot-ai master，`b10298-2.0.0` |
| 旧 fork | `2777543b2358c88cd1c2000f7fa342b764a0e213`，`archive/pre-clean-20261018` |
| 全部变更路径 | 4,634 |
| 历史运行资料 | `artifacts/` 3,908 路径、`reports/` 12 路径；保留在旧分支，未逐一解析所有实验输出 |
| 代码、脚本、测试、文档和构建文件 | **714 个唯一路径** |
| 精确源码输入 | 八份固定 A → 旧 fork 差异快照覆盖全部 714 路径；各快照 SHA-256 和字节数保存在总账 |

第一轮缩减清单里的“547 文件”不是全库分母，已撤回。第一阶段仅做用途索引的记录仍保持原来的证据等级；后续补读另加有归属的记录，不反向改写成第一阶段已经读过。

## 实际审阅深度

以下按每个路径最高的**实际记录**去重统计，不累加重叠分工：

| 深度 | 唯一路径数 | 能说明什么 |
|---|---:|---|
| 新文件全文或已提供的全部改动内容 | 546 | 静态读到全部新增文件，或该旧文件的全部给定变更；不等于未修改全文或运行正确性证明 |
| 选定实际函数、声明族、接口、fixture、断言或文档段落 | 168 | 结论限于记录的真实范围；较大文件的其他内容仍有明确未读区间 |
| 仅符号或变更边界扫描 | 0 | 不能获得实现正确性或无用的判断 |
| 仅用途清单 | 0 | 不能获得基于内容的保留 / 舍弃判断 |
| **合计** | **714** | 每条都保留来源、原始深度和具体范围 |

[audit/files.json](audit/files.json) 是机器总账：每条记录有路径、增删行、分类、可用精确 diff、各阶段 `reviews`、finding 引用和最终去重深度。原始字段、逐 hunk 说明、函数区间、未读补集均保留；不能因为另一份报告读完了某个小合同，就把同文件全部私有实现升级为已验证。

此次专门补读了原先遗漏的短代码、头文件、shader、测试、脚本和配置。对较大的私有运行时与长测试，读到哪些完整函数、哪些只是声明合同都单列。**没有宣称 714 个文件逐行完成语义证明，也没有把尚未迁入或未读代码判成无价值。**

行号通常是固定旧 fork 的源码，或 A → 旧 fork diff 的旧侧 / 新侧行号；分册明确标注 `audit-extra/clean` 的补充 A 接缝片段例外。它们不是当前 clean 分支的行号。

## 审计分册

各分册包含具体函数 / 字段位置、行为、A/B/C 边界、保留建议和风险。标题中的阶段只是证据来源；最终交付入口以本目录 README 为准。

| 分册 | 实际关注点 |
|---|---|
| [服务端与编排](audit/audit-server-findings.md) | Prefill 选择、输出预算、重试、停止字符串、请求统计、状态恢复、采样和进程生命周期 |
| [内存与 KV](audit/audit-memory-findings.md) | 物理 / 逻辑容量、phase 工作区、reserve / fit、KV cells、recurrent、COW、失败预检和 allocator 所有权 |
| [小功能初审](audit/audit-small-findings.md) | corpus、GGUF、模型兼容、校准、数值比较、独立工具和小型正确性修复 |
| [后端改动](audit/audit-backend-findings.md) | CPU / CUDA / Vulkan 的既有算子、WHT、布局、缓存、shader、小型融合及私有图边界 |
| [全库用途索引](audit/audit-purpose-findings.md) | 714 路径的第一阶段用途与候选索引；深度以其后补审及总账为准 |
| [脚本实际内容补审](audit/audit-script-content-findings.md) | 请求、SSE、模型与 DSO 身份、统计、容量评测、验收 gate 和五个实际 reader 反例 |
| [测试实际内容补审](audit/audit-test-content-findings.md) | 真实被测对象、断言是否生效、NaN、预算、失败状态、租约 / snapshot 生命周期和参考算法 |
| [其余代码内容补审](audit/audit-remaining-content-findings.md) | 193 路径：短 shader / header、槽位元数据、状态封装、codec、策略 / 数学及大文件实际接口 |
| [最后的后端漏项](audit/audit-final-backend-findings.md) | 原先仅扫描名称的 14 路径：普通量化接线、线程 scratch、meta 分片偏移与 readback 生命周期等 |
| [最后的接口漏项](audit/audit-final-contract-findings.md) | 八个服务端 / speculative / memory 路径：实例配置、周期统计、parser、恢复合同及所有权声明 |
| [最后的协议与配置漏项](audit/audit-final-protocol-findings.md) | 24 个文档 / JSON：默认值、请求预算、缓存 / 状态协议与历史证据的准确口径 |

## 用户指出的内存与调度能力

| 小行为 | 当前决定 | 为什么单独列出 |
|---|---|---|
| 同 context 的统一 KV 存储 | 复用 A 已有实现 | 上游已有统一分配器，无需复制旧私有 allocator |
| 物理 attention KV 容量与逻辑窗口分离 | **已迁入**最小 A 参数 / getter，C 配置实际使用 | 保持逻辑窗口和 RoPE 语义；物理容量是另一项资源设置 |
| Prefill / Decode 计算工作区分离 | **已迁入**同步释放与实际图重建 | 保留模型、KV / recurrent 和输出所有权；实际分配字节与输出分别验证 |
| Prefill 优先、当前提示保持、槽位顺序 | **已迁入**独立 B 和真实 C；stock server 另有小补丁 | 不是 FIFO；连续提示可能令 Decode 饥饿，不能附带公平性承诺 |
| PP 并发上限、初始 reserve 单序列图 | 独立候选 | 调度上的暂停不自动改变预分配峰值 |
| `auto` fit、每设备 slack、预算求解 | 保留目的，旧 solver 不照搬 | 需要实际分配探测与失败结果；metadata-only 不能代替显存容量证明 |
| 已知 token / 驻留槽位 admission | **已迁入**独立 B 预算和真实 C 队列 / 槽位复用 | owner domain / generation、先删除真实状态再回收；不能把容量值当作显存探测 |
| idle LRU、recurrent 物理槽、抢占恢复 | 独立候选 | 涉及真实所有权、引用、RNG 与部分输出，不能假装由一个 `--total-kv` 参数全部完成 |
| target / draft 跨 context 共池 | 暂不迁入完整旧实现 | 先闭合共享所有权和 fence 协议；当前 release 仅处理单 context 自有 compute |

旧实现中还存在共享 allocator 被随后普通初始化覆盖、KEEP_CHUNKS 跳过 trim、输出 reserve 参数被覆盖等具体问题。应保留功能目的并重做必要合同，而不是因为旧代码有错就丢弃内存模型。

## 藏在大功能中的小能力

以下均已有实际内容位置，详细条件与 finding ID 见各分册；表中“候选”表示未取得当前 clean 分支的真实消费 / 运行资格。

| 小能力 | 当前状态与适合的职责 |
|---|---|
| corpus kind、timestamp、空请求、根节点、重复 / 前缀终点 | 已由 B 保留，并与未修改的旧生产 writer / reader 互操作；在线请求结束录制仍是 C 候选 |
| 有界 logits reader、token 身份、完整行与 argmax 分开比较 | 已迁为独立标准库工具；新 C 通过公开 A 采集完整行。重复与行选择比较全部值，独立 C 只对照最终 greedy token / 最大 logit；同名 TP5 格式不要求私有运行时 |
| task 可选槽位的默认值 | 已用一行 A 修复保留，避免移动普通 task 时读取未初始化整数 |
| F16 V 下的显式 F32 累加 | 已用小型 CPU 修复保留，原 A 的真实用例失败、修复后通过 |
| Turbo WHT 的小 group、scale、跨 workgroup 形状 | 已用两个文件修复，CPU / Vulkan 直接算子对照通过 |
| 64 KiB scratch roundup、checked 容量 / 间接 dispatch、阶段 scratch 最大值 | 可独立成为拥有数值的 B；C / A 执行真实分配，容量算式不改变 dispatch 几何 |
| 热槽 reserve / commit / release、generation 与 payload 身份 | token / 槽位 reservation 已迁 B，带 domain / generation；热 KV payload 与 snapshot 身份仍是候选，不等于 B 已接管 A tensor |
| 固定 state envelope、100-byte LE codec 描述、nonce / adapter / scope | 字节封装 B 与 A opaque state 的 C 包装候选；端序、空指针、持久身份与失败后输出合同须补齐 |
| 每实例 Turbo 默认策略、统计格式化、严格 seed 解析 | 可与私有 runtime 分离；统计需要区分尝试、失败和成功产出 |
| 增量 UTF-8 与跨块 stop | 已由独立 TQText 与 batch C 重新实现；字节游标、一次终止事件、停止 token 计数和输出失败有明确合同。没有宣称旧非流式扫描 fastpath 的速度收益 |
| 按 lane 拼完整行、增量 parser、多 choice 与 SSE | 仍是独立候选；TQText 不提供 thinking parser、usage 聚合、网络送达、取消或重试恢复 |
| 首次失败原因、输入 / 输出游标、请求预算和 usage | 保留状态合同；重试时 token 总数一致不能代替 RNG 与已经发送输出的恢复证明 |
| 无符号 Hadamard / RoPE、RERoT 可见性 / GQA / DDVR 参考 | 独立数学 B 候选；与已修复的 signed Turbo WHT 有不同输入域和含义 |
| mean / energy、online softmax M/L/O、稳定有限 mask compaction | 数学或局部 kernel 候选；局部正确性不能替代整模型稀疏 attention 质量 |
| 单 cell recurrent buffer 清零、一般 strided 清零与前后 fence | M33 减少的是旧 fork 的 eager clearing；当前 A 已整 buffer / lazy init，故不迁旧清零。新增 LFM2.5 hybrid、n_rs_seq=0 的真实生命周期回归；一般 GDN、strided 清零仍未取得资格 |
| 非对齐字节 memmove、meta 分片非零偏移、异步 readback 生命周期 | Meta async 偏移已用 A +4/-4 与48个双 CPU child 用例迁入。memmove 与 GPU readback 仍独立；当前 A 的 graph view UID 为零，且没有旧 pending retire 路径，详见新 Meta 协议 |
| WXQCAL03 格式与 imatrix sum2 / expert counts | 已迁为 TQCalibration 与 imatrix C；原 writer 生产格式互操作及实际 A loader 消费通过，完整性和填补策略显式记录 |
| callback 激活采集、trie 校准复用与 checkpoint | 保留为独立 C 候选；格式转换没有重建采集人口、模型 callback 名称 / 布局、分支统计或异常保存生命周期 |
| 模型名称兼容、PLE uint64 hash | 仍是独立模型 / 格式候选，须对齐 A 的实际元数据消费者；不整包复制图改动 |
| FA / large Softmax 共用 scratch 与 converted-activation 缓存（G09） | 最小 A 修复与 5 个实际 GPU 场景通过；保留小 Softmax、无 mask 和独立算子控制，不据此宣称完整后端通过 |
| 输入经别名原地修改后复用旧 activation | 本轮新反例与最小 A 修复；带 4 字节偏移的图从最后 MM 64/64 错误变为完整输出一致，不重叠写入控制也通过 |
| Turbo SET_ROWS F16 capability（G44） | 复核固定 A 的 shader 生成后保留其 F16 支持，不迁旧拒绝条件；不把这一决定包装成新增后端能力 |
| 构建 / DSO 内容身份、完整请求、失败分级、离线容量与性能工具 | 保留证据工具职责；修正统计分母、非有限值、提前 PASS 和伪并发后才可作为验收 |

### 本轮重新判断的具体范围

finding ID 与来源分册一起引用；例如下表的 `small:S04` 是校准转换，`server:S04` 则是 Prefill reserve。总账中的原始 `source_stage`、范围和当时的 `runtime_verified` 保持原样，本轮结果在 [MIGRATION.md](MIGRATION.md) 和 [VALIDATION.md](VALIDATION.md) 中另行记录。

| 旧 finding | 本轮处理 | 没有一并完成的事项 |
|---|---|---|
| [server:Q14](audit/audit-server-findings.md) | 用 TQText 和真实 batch C 重建文本消费；B 不访问 A，C 拥有采样、detokenize、输出和序列释放 | 原 non-stream fastpath 的性能等价、Q15/Q16/Q17 的首块 / usage / SSE 生命周期 |
| [server:T09](audit/audit-server-findings.md) | 跨 chunk 终止符是可分离的机制；新 stop 规则有自己的独立合同 | thinking parser、lane 文本协议与特定 grammar，不随通用 stop 匹配迁入 |
| [small:S04](audit/audit-small-findings.md) | C++17 B 解析 WXQCAL03；C 使用安装后的公共 gguf/ggml metadata API 输出 imatrix，无 NumPy 运行依赖 | small:S05 的 callback / DFS 采集、周期 checkpoint、真实模型量化及质量 |
| [backend:G09](audit/audit-backend-findings.md) | 从 `ggml-vulkan.cpp#h183` 的缓存身份问题扩展审查 scratch 写入与读后写顺序；最终 5 个专项场景通过 | 最初 fractional V 探针自身的 0.0146484 闭式偏差仍保留，不能算成已修；最终 binary-step fixture 有独立 before/after 和 524288 个闭式比较 |
| 本轮输入别名反例 | 复用 A 的范围重叠判断，在算子记录后检查所有 fused writes；非零 view offset 的 before/after 通过 | 未声称 UMA、全部 fusion / 跨图 cache coherence 或真实模型吞吐已验证；见 [具体记录](audit/details-findings-20261018.md) |
| [backend:G44](audit/audit-backend-findings.md) | `ggml-vulkan.cpp#h290/#h292` 的旧 F16 拒绝不迁；固定 A 已生成 F32/F16 到 Turbo 的 I32/I64 SET_ROWS 变体 | 旧“只编译 F32 shader”前提在本轮复核中被纠正，不是 A 基线更新，也不附带迁入 G19 日志 |

本轮完整 CPU archive/isolation 的 62 条命令通过，新增 TQText / TQCalibration 独立测试通过 ASan 与 UBSan。原校准 writer 的 `.cpp/.h` 与固定旧提交完全相同，真实写出 fixture 后，新 C 的 12 个 GGUF tensors 被 A quantizer loader 读为 6 个 importance entries、9 chunks；随后故意缺失模型的阶段以 exit 1 结束。该证据只关闭格式互操作与实际 loader 消费，不能称作完整模型量化成功。最终 Vulkan 同形状流式模型对照和 7 项 A 接缝 CTest 通过；完整算子门的失败记录与归因边界单列在 [VALIDATION.md](VALIDATION.md)。此前模型与 HTTP 结果保留此前身份。本轮可追溯结果集中在 [details-findings-20261018.md](audit/details-findings-20261018.md) 与 [机器回执](audit/details-qualification-20261018.json)。

## 哪些旧证据不能直接使用

静态源码已经足以定位一些失效条件：并发容量脚本实际串行、并发代码位于 `return` 之后；JSON 先写 PASS 再检查失败；有 CHECK 在最后失败汇总之后执行；`std::max` 比较吞掉 NaN；部分指标只打印不判定；小样本 t 临界值错误；某些评测答案或 dev / blind 分组不正确。

另外五个 reader 问题已实际执行最小反例，输入、输出、原始函数范围和源码 SHA-256 见 [reader-counterexamples-20261018.json](audit/reader-counterexamples-20261018.json)：

| 实际反例 | 观察 |
|---|---|
| 请求一侧缺键，另一侧为 `null` | 不同 canonical payload 返回空 violations |
| 数字答案 `1` 对参考 `1e309` | 非有限参考令错误答案被接受 |
| 真正最优值为 0 的 LP | 容差允许不可行对偶点返回正的“下界” |
| 坏 JSON SSE 行后跟 `stop=true` | `malformed_lines=1` 但 `error=None`；调用者必须另检事件完整性 |
| 多行 fuser PID 交给 bash `read -a` | 只读取第一行；没有执行恢复、信号或硬件操作 |

这些反例只执行选择出的原始纯函数体和 bash 的读取内建命令，没有运行旧模型 / GPU 路径。它们限定旧工具证据的强度，不证明 TP5、RERoT 或 XKV 没有价值。

## 保留和舍弃的界线

本次明确不迁的是三组有历史取舍依据的实验：router row split、低于原晋级门槛的 BF16 router、未优于已选择基线的 MMVQ + MTP 联合实验选择器。具体历史数值、来源和适用范围在 [MIGRATION.md](MIGRATION.md)。

TP5 / LateBind 有历史实质收益；RERoT、FlashPrefill、TriAttention、XKV 各有可拆的小能力。当前不引入其整条私有运行时，但不会把它们归成“无用”或借其他测试的通过宣称完成。旧实现及实验资料仍在归档提交中。

2026-10-18 细节迁移批次交付五个独立 B 与五个 C；用户随后增加第六个 B `TQRelay` 和独立平台探针 `relay-probe`，当前入口与各自范围见 [README.md](README.md)。A 的实际变动由 [upstream-exceptions.json](upstream-exceptions.json) 固定文件、内容、消费者与退出条件，运行资格由 [VALIDATION.md](VALIDATION.md) 逐项给出。该细节批次新增证据另见 [发布候选说明](RELEASE-NOTES.md)，不改写上面的历史源码阅读深度。新增小行为按职责拆成提交；每个包含 A 变动的提交携带对应当时源码的完整文件摘要，不把未来状态的摘要提前塞入旧提交。

## 如何回查

```sh
git show 2777543b2358c88cd1c2000f7fa342b764a0e213:src/llama-kv-cache.cpp
git diff --no-renames cda0f9256665845cd9d6f3a0176880883f653a3d \
  archive/pre-clean-20261018 -- common/arg.cpp
```

以上用于按固定源码和范围复查；diff 的上下文选项不同会改变 patch 的字节摘要。总账的八份输入摘要记录本次实际审阅资料，不替代 Git 源内容身份。

在总账中按 `path` 找到条目，再读其 `reviews` 中的来源分册与范围。局部审阅、旧源码静态发现、历史测量、新分支真实测试是不同证据，保留原有层次即可重新评估每个小功能。
