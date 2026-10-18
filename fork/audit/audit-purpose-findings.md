# 小功能补充审计：714 路径用途清点与独立工具保留

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

基准是旧 fork 2777543b2358c88cd1c2000f7fa342b764a0e213 对 A cda0f9256665845cd9d6f3a0176880883f653a3d。此前的小功能内容审计见 audit-small-findings.md；本报告补充脚本、测试、文档和独立工具，保留小能力的线索，并明确审查深度。

## 1. 覆盖口径及交付状态

权威清单 audit-inventory.json 有 4,634 路径。排除路径前缀 artifacts/ 和 reports/ 后恰为 **714 路径**，没有另外创建或替换一份“714 权威原表”。

最初六份 diff 涵盖 264 路径，另外 450 路径不在这些 diff 中。root 后续提供 audit-scripts-tests.diff，补齐其中 scripts、tests、docs、tools/tp5、pocs 的 **257 个路径**；其字节 SHA256 为：

    ed00f769609b23934f39e63ef743edf2b64c396334262badb5acb6db802749b9

此值已在本机核对。13 个 focused 完整原文件也逐字匹配对应 diff 的新文件内容。当前这些脚本、测试、文档和工具没有缺源码的项目，最初的 focused source 请求全部闭合。

audit-purpose-coverage.json 给每个 714 路径提供独立记录，含用途类别、证据层级、审查范围、来源及其他代理的原始审查声明。统计如下：

| 本代理的审查层级 | 路径数 | 含义 |
|---|---:|---|
| 此前 small diff 的全部给定 hunk | 79 | 每个 hunk 已内容审查，详情保留在原 coverage |
| 完整源码阅读 | 18 | focused 工具、短脚本和测试，逐文件有原行号 |
| 完整新增 header diff | 1 | llama-decode-cost.h 的 48 行 |
| 选定完整函数 | 1 | cross-matrix 的 execute_bench_request |
| 选定 CMake hunk | 1 | greedy/decode-cost 的断言和注册、F32 test 注册 |
| 此前补充源码审查 | 2 | lazy reader 与其测试 |
| 仅用途与符号清点 | 236 | 新 257 范围中，已建立函数/测试/include/headings 索引，未逐函数内容审查 |
| 仅用途清点或其他代理的审查记录 | 376 | 保留其他代理声明的来源与深度，不冒充本代理内容审查 |
| **合计** | **714** | **102 路径有本代理内容阅读；612 路径没有** |

“内容阅读”也不自动意味着完整原文件审计：selected_functions 和 selected_hunks 的边界已经保留。**236 个 only-symbol 项不能在合并 ledger 时升级成“内容审计完成”。** 这些记录使每个文件均有去向和后续线索，但不提供未经内容审查的舍弃结论。

对已发现有价值、尚未迁入的行为，本报告统一使用“保留候选”。保留候选可能仍需参数化、格式检查、修复或运行验证；不因它位于 TP5、RERoT、Wanxiangqi 或研究目录就将其整目录舍弃。

## 2. 已实现的小工具：P01 全词表 logits 比较

原文件 tools/tp5/compare-logits.py:14–79 具有独立用户价值：逐行读取完整词表 logits；要求相同 teacher-forced token 序列；检查截断、尾部数据和 nonfinite；报告逐值误差、bitwise 差异和 argmax 差异。只依赖 Python 标准库，没有 TP5 运行时依赖。

已生成并交付 compare-logits-retain.patch，新增：

- fork/tools/compare-logits.py
- fork/tools/tests/test-compare-logits.py
- fork/tools/README.md

root 已应用，并报告远端真实 CLI 20 tests PASS。本机相同 20 tests 也通过，耗时 1.151 秒。测试实际写入临时二进制 tape 并执行 CLI，不以 mock 替代读取器。

补丁保留 TP5LOG2 magic 及既有主要 JSON 指标语义，并修复原工具的具体边界：

1. 原 compare() Python API 不校验 atol/rtol，只有 CLI 校验；NaN tolerance 可令超差比较绕过阈值判断。新 API 与 CLI 都要求有限、非负数。
2. header 校验非空 prefill 和每个 int32 token 的词表范围；允许自洽的 prefill-only tape，即 steps=0。
3. 所有行都要求 finite。含 nonfinite 的行记录 argmax_unavailable_rows，不将其算作有效 argmax 比较。
4. 新 report 路径排他创建，拒绝覆盖既有报告或输入；输出写失败有结构化失败结果，不抛出 traceback。
5. 文档明确 pass 的含义是 finite 且逐值满足 atol+rtol*abs(reference)。bitwise 和 argmax 另行报告；允许容差时 argmax 改变不自动违反逐值误差阈值。
6. tape 没有 model/tokenizer/build/context-policy 身份，使用者需在旁侧记录。相同词表大小和 token ID 不能证明 tokenizer 相同。

这是一个完整、已接入的独立小功能。其原来所在目录不构成保留整个 TP5 wrapper 的必要条件。

## 3. 已内容审查的保留候选

### P02 通用固定输入 logits 采集器

tools/tp5/tp5-logits.cpp:33–45 的长度/LE header 处理、77–106 的模型初始化和固定 token 构造、118–149 的实际 llama_decode 与完整 logits 保存，都能成为 C 的通用数值采集器。它没有包含私有 TP5 header；当前用到 common 和公开 llama API。若迁入，可复用 A 的 common 源码或改为现有 C 的公开参数/初始化封装。

原 run-logits.py:43–47 硬编码五张 Vulkan 卡与 TP5 flags；41 行调用 timeline_env(binary)，但现存 tp5-bench.py:73 的签名是 timeline_env(binary, mmvq)，会在运行前 TypeError。应保留其“自有进程、超时、日志、新证据路径”的编排意图，重新写通用 C 参数边界。

原 C++ writer 的 exists 检查与输出 open 分离，不能提供原子的不覆盖保证；输出 tape 也不包含身份 metadata。保留候选需要补这两个具体边界。本轮没有迁入或运行该采集器。

### P03 GGUF 目录与模型形状检查

scripts/tp5-gguf-metadata.py:14–77 是较小的独立读取工具：只解析 metadata 和 tensor directory；保留 raw type ID、header 中 type/offset 字段的位置及相邻 tensor 的 storage extent。它不会为了列目录映射全部模型权重。原始 type ID 对跨 fork 的 type 42 解释差异有实际诊断价值。

需修的边界有重复 metadata/tensor 名、alignment 正数、offset/extent 的单调性和文件范围。extent 含 padding，不能称为实际 tensor payload bytes 或 VRAM traffic。原 CLI 的 --out 没有禁止指向输入模型，必须先修输出保护再保留。

tools/tp5/tp5-inspect-model.py 是另外一种能力，不能和上述小目录读取器混为一谈：

- 87–144：header/shard 发现。
- 181–223：完整 GQA 组和 rank-local query/KV 角色。
- 232–267：量化 block 边界下的 head 分配。
- 275–406：Qwen4 特定 tensor role 和静态成本估算。

这些算术/规划部分可保留为离线候选，但当前 parser 缺少严格的 shard 完整性和 metadata 验证；84 行 scalar 读取使用宿主字节序；type 名表和 TYPE_TRAITS 不完整对应；硬编码的 raw type ID 还存在跨 fork 解释问题。464 行 actual_model_validated=True 仅凭 header 检查会夸大结论。应拆分“通用目录读取”和“TP5 特定规划”，使用 A 的类型契约，不能以该脚本的静态估算替代实测内存。

### P04 离线测量报告与工作量验收

scripts/nanbeige-report.py:40–137 不是普通漂亮报表。它拒绝未完成运行、构建变化、模型变化、失序/缺失/重复样本、截断、缓存复用、错误 token 数和不一致的时间/速率；将首请求单独列出，后续请求另算中位数及按总 token/总时间加权的速率。

140–162 的比较要求相同 model SHA 和 request SHA，且每组至少两个后续请求；可选要求所有重复输出完全一致。165–178 还拒绝重复 JSON key 和 nonfinite JSON。旧记录缺身份时仍可显示受限描述，但不会宣称可直接比较。这样的离线报告功能值得保留为独立 B/C。

它仅通过导入调用 nanbeige-audit.py:559–578 的纯 check_completion helper；应提取这个纯 helper，不必把整个 server runner 作为报告依赖。原 tests/test-nanbeige-report.py 的 15 个测试已在本机通过，使用真实临时 JSON 文件，没有 GPU。

### P05 正确的吞吐分母和完整请求

tools/tp5/tp5-bench.py:120–135 用总 committed token / 总时间汇总，排除 warmup，并分别报告 decode 与客户端 wall time。原 tests/test-audit-tools.py:50–59 的 fixture 能区分正确的 55 tok/s 与错误的逐请求速率平均。这是可独立保留的小规则。

scripts/run-tp5-cross-matrix.py:852–1014 的 execute_bench_request 使用完整收到响应后的单调时间，要求完整精确输出、自然 stop、usage.completion_tokens 存在，以及有限正的 predicted_ms。count-60 的 31/171 token oracle 绑定特定模型/tokenizer；不能作为通用 server invariant。非 count-60 分支的校验也较弱，不能直接把此函数的全部判断抽象成普适验收器。

原 bench 中的 sysfs 五卡选择、环境 defaults 和私有 TP5 flags 是 runner 策略。应保留纯“固定输入 + 完整输出 + 相同工作量 + 正确分母”的规则，并与可配置 C runner 分离。

### P06 构建、模型与请求身份的小检查

scripts/nanbeige-audit.py 已完整阅读，其中可独立的行为包括：

| 原函数/行号 | 行为与保留边界 |
|---|---|
| artifact_fingerprint:38–68 | 对本地所选 library alias 目标去重 hash，避免把目录中旧版本孤儿库都当成本次运行内容 |
| _make_dependencies:78–87 | 解析 compiler dependency 文件 |
| _cmake_link_inputs:90–143 | 只读 link recipe/response 文件，处理嵌套并拒绝循环 |
| _archive_members:146–198 | 解析 GNU/BSD ar 名称及长文件名，不执行归档命令 |
| inspect_build_identity:201–242 | 比较 CMake source 的 HEAD 与可执行文件 --version 报告的 commit，并记录 dirty 状态 |
| inspect_build_artifacts:245–384 | 追踪 local object/header/link 时间关系和重复 archive member，暴露被绕过的正常增量构建链 |
| model_fingerprint:387–399 | 内容 hash 前后比较 dev/inode/size/mtime/ctime，拒绝读取过程中模型变化 |
| probe:588–598 | 对实际 tokenized 请求及其采样参数计算规范化 request hash |

这些应作为 C 的独立证据检查候选。时间戳及 commit 匹配不能证明 ABI 一致；本地 alias 集合也不等于完整实际加载 DSO 集合。当前 build-tree 检查依赖特定 CMake Makefiles 布局，Ninja/安装目录的覆盖范围必须如实标明。标准库实现并不消除这些证据边界。

### P07 另一个不兼容的 corpus 生成器

scripts/nanbeige-audit.py:509 写 nodes start_id=0，527 写 request kind=0。它与旧 production recorder 的 canonical header/metadata 不同，只是旧 trie parser 忽略字段时能读取。

新 B 按生产契约拒绝这种目录是预期行为。这类历史目录应重新构建；不能将其当作真实旧 recorder 兼容 fixture，也不能伪造 metadata 来迁移。若保留 prepare 功能，C 应通过统一 token_trie append/save 写出数据，避免继续手拼二进制。模型 /tokenize 调用本身仍是有用的小型 corpus 构建路径。

### P08 配对统计的价值和置信区间缺陷

scripts/tp5-paired-stats-h1.py:21–44 按 measured phase 汇总 tokens/ms；64–68 要求两 arm 完整输出一致、退出正常且没有强制清理；70–90 用 log ratio 的均值表示几何平均加速比。独立的 paired-block 算法值得保留。

但 t_quantile_95:47–51 对表外自由度统一返回 2.0。对于 2 或 3 个 paired blocks，自由度为 1 或 2，正确的双侧 95% t 临界值约为 12.706 和 4.303；旧值会让区间过窄。n=1 仅输出 NaN 区间，n=0 还会在平均时失败。它也使用硬编码历史 artifact 路径，对每个样本的有限正计数/时间缺乏严格检查。

保留候选应修正自由度、样本数和数值校验，并把路径、arm 标签和相同输入身份参数化。历史报告中的既有区间需要按实际样本数复核，本轮没有重算其性能结论。

### P09 双 context 确定性回归的误报风险

tests/test-native-determinism.cpp:95–112、133–160 用同一输入比较两个 context 的所有 logits，是值得保留的正确性属性。

107–109 和 149–151 使用 std::max(current, abs(left-right))，没有先检查 finite。NaN 差值可被 std::max 吞掉，max_diff 保持 0；当 argmax 也没有变化时，测试会误报 exact deterministic。114–127 的 tape 读完没有检查 stream、token 范围或足够的上下文容量。26–42 还含 fork 扩展 context 字段。

应保留该属性，用公开 A context 配置和已修的完整 finite tape 比较实现；不照搬旧 verifier 的错误判断。

### P10 采样 RNG 与 trace 关闭成本

tests/test-greedy-unique-host.cpp:29–56 覆盖 nonfinite/tie fallback、400 行真实 sampler chain 对拍，以及之后 100 个依赖 RNG 的样本。后续 RNG 连续性是独立且重要的小契约；仅比 greedy token ID 无法覆盖它。59–85 的耗时仅是 CPU sampling microbenchmark，不是模型吞吐。

tests/test-decode-cost.cpp:8–19 用明确的 fake clock 检查禁用 trace 时 0 次读时钟、启用时 7 次。src/llama-decode-cost.h:1–48 是小型 host span timer，本体可独立；在 A 内部插入 mark 的位置仍属于 A instrumentation。

已核对 tests/CMakeLists.txt:136–155：原仓库确实对这两个测试设置 -UNDEBUG 或 /UNDEBUG，并为 decode-cost 注册 GGML_DECODE_COST=0/1 两种环境。不存在“原构建忘了启用这些断言”的已证实缺陷；迁移时应携带这个测试条件。嵌套 decode/ubatch span 的时间不可相加或称为纯 GPU 耗时。

### P11 F32 attention 累加回归

tests/test-flash-attn-precision.cpp:1–53 只用公开 ggml/CPU API。它在递增 score 下放入完全可表示的常量正负 F16 V，并要求 256/1024 KV 长度时 F32 attention 输出保持该常量，finite 且最大误差小于 3e-6。

root 已报告其真实 A 验证为修前失败约 3.88e-4、最小四行修复后通过约 1.19209e-7。最终数值、运行命令及 CPU/Vulkan phase 结论以 root 的最终 VALIDATION 为准，本代理未重复执行 GPU 验证。

### P12 纯 DAG 和 frame 参考

scripts/rerot-dag-reference.py:11–39 提供带重复/未知端点检查的 priority Kahn topo；42–64 提供 ready-set 和 cycle-preferred reader view；67–91 是简小的 frame 状态机。112–164 枚举四节点有向图、合法 live state 和 frame composition 的参考检查。

它是标准库逻辑参考，可保留为 B 的数学/调度测试候选。输入 validator 和 frame 完整性还需明确：walk_frames 返回状态，不能只检查其布尔值就认定所有 frame 完成。其自测使用 assert，需要 Python 非 -O 模式。这些纯逻辑检查没有验证私有 KV、GPU 或实际模型语义。

### P13 离线 attention 分布诊断

scripts/rerot-attention-audit.py:14–34 按真实 byte stride 读取 F32/F16/I32 dump；37–50 有 JSD/cosine；53–118 重建 indexed Q/K 分数、softcap、每 head entropy 和 reader 间 overlap。

统计和 stride-aware reader 可独立为 B/C 候选，但 dump 的 producer 仍依赖原 RERoT 观测路径。当前工具只支持其显式 F16/F32 观察约定；需要 exact read、非空/finite/维度/offset 检查及全部 score 参数一致性证明，不能称为所有 attention 形态的完整 CPU oracle。

### P14 通用成本算术

scripts/tp5-perf-model.py:1–152 没有 TP5 runtime、GPU、网络或 subprocess 依赖：

- number/checked:14–28 检查有限数、正数、bool 和算术溢出。
- cycle:31–60 按 committed public tokens 计算周期成本及盈亏平衡。
- budget:63–75 把目标 tps 转为预算。
- amdahl:78–95 计算单一时间份额的变化。
- resource:98–113 使用 memory/compute 的 max，并明确资源边界和重叠假设。

输出显式标为 CALCULATION_NOT_BENCHMARK。它是可以独立保留的小工具候选，不能因文件名含 TP5 一并丢弃。本轮没有迁入，也没有据它产生新的性能结论。

## 4. 236 个符号层项目如何进入后续保留清单

以下是已建立函数/测试名和行号索引的用途线索，**没有逐函数内容审查**。各文件独立记录在 JSON，不能以这里的分组代替其具体审计深度。

| 小职责线索 | 精确候选与索引 | 下一步判断 |
|---|---|---|
| SSE 分片解析、完成边界和输出组装 | flashprefill-matrix.py read_sse_stream:1084–1219、assemble_stream:1222–1296；xkv/run-evaluation-matrix.py parse_sse_stream:230–313 | 可独立 C client/protocol 边界候选；需实际检查断流、UTF-8 分片、终态和多 choice |
| Prometheus label/metric 解析 | flashprefill-matrix.py parse_labels_block:905–915、parse_metrics_text:928–953、selector_delta:972–1005；xkv/collect-runtime-evidence.py parse_prometheus_text:154–248 | 可独立 parser 候选；需区分 counter/gauge/reset、缺值和重复 series |
| 严格 evidence envelope 与队列状态 | experiment-queue.py validate_envelope:141–231、dry_run_evidence_layout:251–348、ExclusiveLock:352–389、enqueue_pack:414–466 | 名称不证明它会执行实验；有 MockTp5Comm 与 mock teardown，需辨别 inert 记录和真正 runner |
| 正常退出与资源未知状态 | test-safe-teardown-mock.py 的 drain_fail、unknown_completion、fabricated_completion、successful_drain；test-t1c-publish-order-mock.py 的 generation/ABA 检查 | 保留契约候选；mock 结果不能作为真实 GPU fence/同步证明 |
| 成功运行所需的实际路径证据 | test-throughput-gate-mock.py:207–498 的缺 counter、没有多 lane、forced frames、未出现六行 decode 等拒绝 case | 防止只有“功能开关已开”而无实际执行的验收误报；需检查生产 counter 来源 |
| 质量评分与区间 | flashprefill-quality.py wilson:399–412、paired_diff_ci:415–437、score_exact/numeric/regex:504–539 | 独立离线评分候选；要核对统计总体、缺失数据、失败样本和 grader 身份 |
| 解析/reader 顺序 | rerot-mindmap-bench.py check_reader_order:275–395；tests/test-rerot-mindmap-parser.cpp、test-rerot-parser.cpp | 保留原 wire/状态机 reference；不能用普通多请求近似私有 KV reader 语义 |
| 多模板和流式隔离 | tests/test-chat.cpp、tests/snapshots/*、rerot-simple-continuation-smoke.py compare_stream:54–83 | 源码/断言检查后拆分普通 server 回归与特性专用测试 |
| XKV 字节成本与编码尺寸 | xkv/memory-model.py checked_add/mul/div:79–102、encoded_matrix_bytes:260–293；tests/test-xkv-factor-encoding-size.cpp | 可独立格式/成本测试候选；不将名义 bits-per-value 当真实 encoded size |
| XKV 关闭时普通路径等价 | xkv/off-regression.py run_side:412–491、compare:493–707；tests/test-xkv-* | 保留 off-path 证据候选；具体私有 fixture 需要按消费路径拆分 |
| 纯 HC/索引参考 | tp5-hc-coordinate-prototype.py reference:70–88、coordinate_owned:91–140；test-tp5-upfold-indexing.py；test-tp5-hc-elementwise-fusion.py | 数学/reference 可能可独立；不能从文件名推断已通过实际 kernel |
| 必要性/LP 证书 | tp5-necessity-audit.py joint_lower:55–62、exact_counterexamples:80–119；tp5-necessity-certificate.py verify_lp_dual:278–304、merge_lower_bounds:438–480 | 保留计算/反例工具候选；审查 premises、单位与不同资源界相加/取 max 的条件 |
| 原始 profile 汇总 | tp5-gap-report.py paired_report:135–197、unique_texts:200–219；tp5-sparse-summary.py summarize:10–51 | 保留离线证据归并候选；检查关联键、遗漏/失败样本、身份和实际 timing 区间 |
| backend 边界回归 | test-backend-ops.cpp、test-backend-memmove.cpp、test-vulkan-flash-attn-capacity.cpp、test-vulkan-command-replay.cpp | 对照对应 A seam 后保留；目前只有符号/用途层记录，F32 test 是已深入检查的例外 |
| P2P 独立验证 | pocs/vulkan-p2p/test-vulkan-p2p-allreduce.cpp | 目录独立可能性较高；仍需读取实际 Vulkan 资源/完成/错误路径，不能以计时注释代替真实计量 |
| 运行脚本与设备恢复 | gpu-idle-audit、watchdog、reset、run-ternary、run-qwen38 等 | 分离只读诊断、启动参数、自有进程监督及设备动作；本轮没有执行或确认这些脚本的完整行为 |

所有 Wanxiangqi 原有校准、转换、fit、ablation 能力的细分仍见 S04/S05/S25/S26。fit-test.py 和 make-transforms.py 具有研究配方价值，其 ledger 处置更新为“修复后保留候选”；原来的缺 import、形状/finite/partial 校验问题仍然存在，当前没有迁入。这一状态保留其潜在价值，也保留未完成的验证。

## 5. 提交记录提供的线索及证据限制

audit-commits.txt 用来发现可能遗漏的小行为。提交标题是定位线索，不证明某一源文件当前仍包含正确实现：

| 提交 | 帮助定位的独立行为 |
|---|---|
| f77498a40、56238d0da、c2914a461 | 离线报告的 provenance gate、模型前后 fingerprint、拒绝截断/不一致样本；已由 P04/P06 实际源码闭合 |
| 1783ad91f、b4a655c02、27981ad97、79990745c | stale linked artifact、runtime commit、runtime library dependency、mixed ABI 证据；已读 P06 的实际边界 |
| bc16e8853 | 首请求与后续吞吐分开；已读 P04 实现和测试 |
| 7ea2d32c9 | A/B/B/A 与多轮统计；只对 P05 所列 helper 作内容结论，整个吞吐 harness 未深审 |
| a381654e3、cb1eee7b7 | FlashAttention capacity tail isolation、GQA 和量化 KV 回归；测试主体仍为符号层 |
| cb382344b | server task 所有权与跨 TU queue 生命周期；由 server/common 分工与具体后续测试审查关联 |
| cbe8014c1 | 独立 P2P 多轮计时和数据字节记账；本分工仍未深审该程序 |
| 6ebbf78ee | Release 下断言仍应生效；P10 的实际 CMake 条件已核对 |

这些关联不会替其他代理生成“已读”声明。合并 714 ledger 时，应使用其他代理报告中的原始 hunk/函数边界补充，保留本报告的 only-symbol 标记。

## 6. 验证与截止点

本轮实际运行了原 nanbeige 离线报告 15 tests，以及保留后的 compare-logits 20 tests。root 随后应用小工具并报告远端相同 20 tests PASS。CPU/Vulkan phase、corpus 原生产 writer/reader 互操作及最终 A/B/C 集成结果，以 root 最终 VALIDATION 为准。

上游 imatrix 的实际消费契约已回写 audit-small-findings.md / audit-small-coverage.json：GGUF .in_sum2/.counts 按 F32 读取，counts 经 lround，按 expert-major 的每专家 count 归一化，chunk_count 不参与权重归一化，三个 metadata key 必需。导出器仍需显式 partial/形状/finite/计数精度规则；本轮没有运行完整量化转换。

此 checkpoint 后不继续扩大候选或实现。完整脚本/测试/文档 diff 已在本机，后续工作的瓶颈是具体函数的内容审查和逐项验证，不是缺少上述 257 个源文件。保留候选与只做用途清点的项目均已进入 coverage，未将它们批量认定为无用或删除。

## 后续内容补审

本分册的714用途表是第一阶段的原始审阅深度快照。后续 scripts-content、tests-content、remaining-content 已继续读取实际内容；请以 files.json 各阶段记录和 ../AUDIT.md 的去重统计为当前覆盖范围。原始612个未内容审阅的计数没有被当作最终计数，也不会逆向篡改为本分册已读。
