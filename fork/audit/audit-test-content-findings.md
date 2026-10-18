# 测试目录实际内容补审

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

## 结论与范围

本轮把旧 fork 的 `tests/` 变更从用途索引推进到实际内容审阅：147 个路径全部有明确的行为记录，142 个在本轮读取代码，5 个沿用此前深审。最新一次聚焦补读仅核对 lazy-mmap 测试的99行全文，关闭此前该项行范围不确定性。大型测试按完整选定函数、关键 fixture、断言及比较器读取；这不表示每个大型文件的每一行都读完。逐路径 JSON 公开已读、未读与 diff 未提供的区间，原始用途报告不被覆盖。

来源为原 fork `2777543b2358c88cd1c2000f7fa342b764a0e213` 对上游 `cda0f9256665845cd9d6f3a0176880883f653a3d` 的精确 diff：`audit-scripts-tests.diff`。文中行号全部是旧 fork 文件的源码行号。总清单仍为 `audit-inventory.json` 的 4634 项；root 的 714 项是排除 `artifacts/`、`reports/` 后的子集，本报告只补其中 147 个测试路径。

没有执行本轮所读的旧测试，没有启动模型或 GPU 任务，也没有操作远端。新 fork 已迁入功能的 CPU/Vulkan、真实模型与旧格式互通结果，以 root 的 `VALIDATION` 为准。下文“测试断言”表示源码确实有该检查，不能改写成“本轮实测通过”。

## 从测试中找回的独立小功能

这里的保留候选只表示已有具体用户价值和代码证据，需要放入后续保留清单；没有迁入不等于舍弃。A 内部状态相关的合同，需要真正的 A 接口或小 seam，不能用普通多请求伪装原始私有 KV 语义。

| 可保留行为 | 精确证据 | 合适边界与依赖 |
|---|---|---|
| 固定容量 arena、对齐、碎片合并、租约跨 facade 生命周期 | `test-xkv-cache.cpp:552–714`，`test-xkv-reader.cpp:1578–1699`，`test-xkv-workspace-bound.cpp:402–590` | std B 候选；拥有存储的 RAII 租约，容量不足保持原配置。实际 reader 接入 C 时仍需明确工作区需求。 |
| 带代数标记的 slot reservation 与借用期间禁止淘汰 | `test-xkv-hot.cpp:277–409,474–567`，`test-xkv-landmark.cpp:979–1014,1698–1790` | std B 候选；确定性复用、ABA 拒绝、pinned LRU、失败不发布部分结果。 |
| 不重用的物理 payload ID，与逻辑引用次数分离 | `test-kv-cells.cpp:724–860,1004–1027`，`test-xkv-state.cpp:1407–1459`，`test-xkv-runtime.cpp:2325–2369` | ID 与预算算法可拆 B；实际物理 KV 占用统计是 A seam 合同，不能把 freed rows 当作 VRAM 已释放字节。 |
| 共享物理内容只计一次重要性、稳定的 budget 与 tie order | `test-xkv-tri.cpp:1519–1581,3108–3182`，`test-xkv-admission.cpp:347–399` | 独立规划/预算 B 候选；实际分数生产和执行回收仍依赖原 Tri/XKV。 |
| 立即捕获 host 输入的 async upload，与 pending readback 生命周期 | `test-vulkan-command-replay.cpp:789–901`，`test-tp5-rank-local-readback.cpp:227–403` | 真实 backend 小合同，应留回归；snapshot async 是旧私有 iface，普通异步上传不能自动承诺相同所有权。 |
| 同 UID 的不同 graph slice 不错误复用；存储依赖延长生命周期 | `test-meta-reduce-boundary.cpp:2067–2151,2315–2367`，`test-alloc.cpp:730–798` | A backend seam 候选；有实际 CPU 图计算与数值核对，不只是缓存键字符串比较。 |
| 独立配置身份、每实例 Turbo group、刷新后的 snapshot 失效 | `test-turbo-config.cpp`，`test-xkv-codec.cpp:706–760`，`test-flashprefill-state.cpp:541–600` | 配置/身份 B 候选；全局 legacy Turbo 开关不能污染另一实例。真正 GPU 并发不由顺序单测证明。 |
| 严格完整数字解析、显式默认值和选项冲突 | `test-arg-parser.cpp:1516–1588`，`test-target-capacity.cpp:91–184` | C 参数层候选；seed 支持完整 uint64 范围，拒绝符号、尾随垃圾、空白及溢出。 |
| 冷启动与重复请求分开；失效样本不能宣称吞吐更快 | `test-nanbeige-audit.py:146–184`，先前深审 `test-nanbeige-report.py`，`test-tp5-gap-tools.py` | 独立 C 离线证据工具候选；保留 identity、请求数量和失败状态。JSON/log 检查不替代真实生产事件。 |
| 模型/库按字节内容辨识；读构建响应文件而不执行 | `test-nanbeige-audit.py:282–344,588–640,698–766`，`test-xkv-state-code-stream.cpp:876–974` | stdlib C 工具或小 B 候选；分片顺序进入 digest，改路径而同字节不改变内容身份；tokenizer 同一性另有语义要求。 |
| 逐字节增量规划协议、完成行复用、首次错误原因保留 | `test-rerot-mindmap-parser.cpp`、`test-rerot-thinking.cpp`、`test-rerot-todo.cpp`，`test-rerot-runtime.cpp:200–294,964–1029` | 纯 parser/serializer/line mux 可拆 B；C 必须明确控制记录与正文界限。scheduler 的 KV 执行不在此纯文本替代范围。 |
| 回收存储上的输入对象初始化，move-only task 与跨 TU queue 所有权 | `test-rerot-recurrent.cpp:941–962`，`test-server-task.cpp:15–166` | 小初始化/所有权修复与回归候选；有 placement-new poison 和真实 setter、真实 queue。 |
| 真正的数学参考与反例 | `test-rerot-attn.cpp:263–286,808–895`，`test-xkv-matrix.cpp:755–844`，`test-rerot-math.cpp:446–527,650–758`，`test-triattention-score.cpp:1164–1259` | 纯参考算法 B/验证资产；分别保留有限性、在线全局 softmax、四码 PQ2、Hadamard/phase 顺序。数学参考不能被宽松的 NaN 误差比较削弱。 |
| HC 折叠索引、BF16 元素顺序、float32 运算顺序反例 | `test-tp5-upfold-indexing.py:58–229,371–427`，`test-tp5-hc-elementwise-fusion.py:492–566` | 独立研究/回归资产；包括明确旧线性索引越界反例与抵消/FMA 差异，不必迁入 TP5 大型编排才能保存。 |
| 校准模型几何兼容与不跨位置空洞 max-pool | `test-triattention-score.cpp:241–292,725–741,1117–1156` | 模型适配 C 候选；使用实际 head dim，过滤 recurrent/SWA 层，检查 rotary pairing。原 scorer 仍有私有依赖。 |
| 动态输入、mask 与候选容量合同 | `test-vulkan-command-replay.cpp:1237–1278`，`test-flashprefill-select.cpp:442–540,661–746`，`test-flashprefill-state.cpp:1963–2033` | 回归资产及规划 B 候选：变化索引必须生效、同分 inclusive、容量不足报错而不截断、外序列不可污染本序列布局。 |

## 对旧测试证据的具体修正

以下问题都是从已读源码得出；它们说明应修复或谨慎使用旧验证器，不表示相应功能没有价值。语法疑点只在 diff 层面记录，没有为归档代码另跑编译。

| 证据问题 | 位置与实际行为 | 对结论的影响 |
|---|---|---|
| 后半段失败没有影响退出码 | `test-fit-residency.cpp:115–118` 在后续 `120–250` 检查之前已检查失败计数，之后仍直接返回 0 | 后续报告的失败不能靠进程成功状态排除；保留 residency policy 时需保留真正终点 gate。 |
| NaN 被 max 误差累积吞掉，或只比较较短向量 | `test-xkv-vulkan-attention.cpp:45–52,432–455`，`test-turbo-k-shift.cpp:82–89`，`test-rerot-shared-block.cpp:402–412`；另见逐文件条目 | 有些全 NaN/长度错误可能显示误差 0。`test-rerot-attn.cpp:263–286`、`test-rerot-recurrent.cpp:223–246` 已提供可保留的正面有限性合同。 |
| 在测试中手写实现后只验证自己 | `test-target-capacity.cpp:520–545` 只模拟 safe slot；`983–987` 手工清零 output 后再断言全零 | 不能声称生产 TARGET 已清理尾部。`91–184` 的真实 input setter 检查仍有独立价值。 |
| 模式选择/格式行为用本地副本或恒等式验证 | `test-tp5-plan.cpp:1148–1161,1163–1255` 使用 test-local resolver；`1636–1637` 先赋相同值再比较 | ABI offset helper 是实际生产调用，但“producer/consumer 模式一致”尚未通过该测试连接生产路径。 |
| FP32 更准确的结论没有数值证据 | `test-tp5-plan.cpp:1709–1713` 仅 `sizeof(float)>2` | 不能据此宣称量化误差严格缩小。应保留格式合同，准确率另做同输入比较。 |
| “T−1 字节”其实少一个 float | `test-xkv-vulkan-factorize.cpp:413–420`，`test-xkv-vulkan-landmark.cpp:1004–1012` | 检查的是 T−4 bytes 的 supports，不是执行路径 T−1 byte。其他 arena 测试有真正 byte/对齐边界，不能互相混写。 |
| “所有 BF16 bit patterns”不是穷举 | `test-tp5-upfold-indexing.py:409–420` 实际 19 种模式重复与滑窗 | 有价值的映射例子，不能称 65536 种穷举；经 Python float 往返还会影响 signaling NaN。 |
| “全 GPU/服务压力”实际只是 metadata 构造 | `test-server-task.cpp:342–429`；多处 runtime null-context fixture | 保留 task/queue policy，小测试不能证明 GPU OOM、SSE、真实 reclaim、生产并发或死锁恢复。 |
| sparse/replay 路径仅打印统计 | `test-vulkan-sparse-attn.cpp:541–621` replay hit 只打印，`get_sparse_count` 参数未消费 | 实际输入变化/CPU parity 有效，但不能仅此证明走了稀疏或 replay 内核。对照 `test-vulkan-command-replay.cpp:1272–1274` 的真实 hit 下界。 |
| 每个指标必须满足与任一满足不同 | `test-vulkan-sparse-attn.cpp:457` 用 `nmse>tol && max_diff>tol` 才失败 | 实际通过合同是任一误差指标足够小；应准确呈现，不写成同时满足两个限制。 |
| 与待测 GPU 同源的 expected | `test-turbo-k-shift.cpp:229` expected absolute RoPE 用同 backend；mesh HC reference 是 unchained GPU 再 CPU half-sum | 能检查执行组织差异，不能单独排除底层算子共有错误。root 的独立 phase/数学验证是另一层证据。 |
| 失败路径或缺失 seam 被标题放大 | `test-xkv-hot-cache.cpp:1714–1801` 可能在 mutation 前失败且 RoPE_NONE；`test-xkv-fault-matrix.cpp:2111–2126` 把缺失 injection seam 列表报告 PASS | 不能写成完成了 on-device K rollback 或所有故障注入；具体保持 pending 状态和缺口列举仍值得保留。 |
| 同数据跨 stream/head 会掩盖路由错误 | `test-xkv-vulkan-attention.cpp:545–558` 两个 stream 相同；`test-rerot-shared-block.cpp:246–262` 各 KV head 被填为相同数据 | 支持 baseline parity，不能验证选中正确 stream/head；移植路由回归需用能区分来源的值。 |
| OFF gate 掩盖数字解析问题 | `test-arg-parser.cpp:983–1062` 不启用 XKV，非默认 OFF 本身可拒绝 | 这组失败不能单独证明每个数字都严格解析；enabled seed 测试 `1516–1588` 有明确隔离价值。 |
| 负面对照改变的不止一个条件 | `test-flashprefill-attn.cpp:638–648` 双 sink 对照未加入 value 项；`test-xkv-tri.cpp:1573–1577` 两个同位置候选都淘汰 | 前者不能单独归因 double-count；后者没真正检查 tertiary cell-ID tie-break。 |
| 部分强断言仅检查一个字段或空循环 | `test-xkv-fault-matrix.cpp:1781–1829` 只检查 sentinel；`test-xkv-reader.cpp:891–895` 未检查 output size；`test-rerot-view.cpp:2375–2382` 无 runs 的 fixture 可能空比 | 报告必须把“完整无修改”“有限全长输出”“完整 view 一致”缩到实际检查范围。 |
| 缺失元数据被视作零错误 | `test-nanbeige-audit.py:634–640` 缺 link.txt 得 `checked_links=0, errors=[]` | 这是缺覆盖，不是构建新鲜度已证实；使用者必须消费 coverage 字段。 |
| 源码字符串顺序不是执行语义 | `test-tp5-upfold-indexing.py:576–768` dump 检查及若干 registry 哨兵 | 保留已知坏样本有价值，但文本搜索不能证明支配关系、同步完成或真正 buffer lifetime。隔离源码不存在时还会 skip。 |
| 研究数据错误和 blind 划分冲突 | `data/rerot-mindmap/eval-set.txt:180–182` 表达式结果应为 7 而非 35；多 family_group 同时出现在 dev/blind；fixture 空文本规则与 parser 不一致 | 测试/研究资产应保留并修正，不能用现有标签直接形成质量结论。 |
| diff 级疑似函数大括号问题 | `test-chat.cpp:7459–7460,7474` | 只记源码片段疑点，未从完整编译单元或编译结果确认；不称已确定编译故障。 |

其他具体限制均在逐文件记录中，包括 fixed `/tmp`/cwd fixture 写入、对 `new` 而非全部分配器的计数、无超时 thread join，以及常规 `assert` 对构建模式的依赖。没有执行这些 fixture。上述限制仅用于判断旧测试证据的强度，不额外扩大当前改造范围。

## 覆盖深度与证据边界

脚本 `audit-test-content-build.py` 只把手工撰写的 `audit-test-content-records.py` 组装成报告，并计算区间并集；它不根据 AST、函数名或断言数量自动判断审阅完成。旧报告中 `purpose_symbol_inventory_only` 仍保留在原文件中，本补充为 root 总账提供更具体的新证据。

JSON 的 `files` 数组每项有 `path`、`finding_ids`、`review_depth`、`behavior`、`dependencies`、`retention_decision`、`failure_conditions_and_limits`，以及精确的可见已读/未读区间。大型文件保留明确未读范围，例如 `test-rerot-runtime.cpp` 的大多数 DAG 生产集成体未读，不能据选中几个函数概括为全运行时已验证。

此前 `test-model-lazy-mmap.cpp` 旧补充记录的行号与本次 diff 不一致；现已按 root 的聚焦补读指令，实际读取当前精确源快照的1–99行全文并关闭该项不确定性。diff 明确为新增文件 `@@ -0,0 +1,99 @@`、blob前缀 `a56fde3d2`；提取源码 SHA256 为 `f545c4964af6eb593ed341e383fe6070bdcccc4d9b6796f738a4fe1b657fa388`。正确函数范围为 `check_lazy_rows:10–84`、`main:86–99`，不再沿用旧记录中超过99行的定位。

当前源码确认：49行由测试手工调用 `loader.lazy.add`；57–66行可选lazy绑定并实际调用 `load_all_data`；71–80行在加载及映射裁剪之后核对首/中/末三行与混合ordinary tensor。临时文件使用 `std::tmpfile()`。这是一项真实加载后映射生命周期回归候选，但不能证明生产模型自动注册lazy，也不测试加载器销毁后的读取。mmap不可用时86–89行整体跳过，非mmap子例同样不会运行；DIRECT子例仍启用mmap，因此不能单独证明直接IO已执行。此次没有运行该测试。其余五个沿用条目均带原有精确范围。

- 本轮手工内容记录：142 个；沿用深审：5 个。
- 本轮完整新文件内容：67 个；全部可见 modified hunks：24 个；选定实际函数/fixture：51 个。
- diff 提供源码共 107083 行；本轮明确读过 28391 行；沿用范围明确的此前已读 543 行。
- 范围已分类的可见未读源码仍有 78149 行；精确范围未重建条目 0 个。

## 逐文件索引

下表给出每个条目的证据 ID、实际已读范围和可见未读范围。完整行为、依赖及失败条件以同 ID 的 JSON 条目为准。modified 文件还会在 JSON 单列 diff 未提供的 unchanged 区间与未知尾部，不能因为“所有 supplied hunks 已读”就称整个文件已读。

| ID | 原文件 | 深度 | 已读源码范围 | 可见未读范围 |
|---|---|---|---|---|
| TC001 | tests/.gitignore | 全部提供的修改 hunks | 18–27 | 无 |
| TC002 | tests/CMakeLists.txt | 全部提供的修改 hunks | 117–209, 249–495, 527–538, 669–675, 759–763, 796–799, 802–807 | 无 |
| TC003 | tests/data/rerot-mindmap/eval-set.txt | 完整新文件 | 1–342 | 无 |
| TC004 | tests/data/rerot-mindmap/fixtures.txt | 完整新文件 | 1–313 | 无 |
| TC005 | tests/fixtures/run-envelope.example.json | 完整新文件 | 1–74 | 无 |
| TC006 | tests/snapshots/deepseek-v3.1.schema | 全部提供的修改 hunks | 3355–3359 | 无 |
| TC007 | tests/snapshots/gemma-3-4b-it.schema | 全部提供的修改 hunks | 1451–1455 | 无 |
| TC008 | tests/snapshots/glm-4.6v.schema | 全部提供的修改 hunks | 4051–4055 | 无 |
| TC009 | tests/snapshots/gpt-oss-120b.schema | 全部提供的修改 hunks | 5596–5600 | 无 |
| TC010 | tests/snapshots/meta-llama-3.1-70b-instruct.schema | 全部提供的修改 hunks | 3895–3899 | 无 |
| TC011 | tests/snapshots/qwen3-0.6b.schema | 全部提供的修改 hunks | 1220–1224 | 无 |
| TC012 | tests/snapshots/qwen3-14b.schema | 全部提供的修改 hunks | 1904–1908 | 无 |
| TC013 | tests/snapshots/qwen3-coder-next.schema | 全部提供的修改 hunks | 2137–2141 | 无 |
| TC014 | tests/snapshots/qwen3.6-27b.schema | 全部提供的修改 hunks | 1–3, 2404–2408 | 无 |
| TC015 | tests/snapshots/step-3.5-flash.schema | 全部提供的修改 hunks | 2449–2453 | 无 |
| TC016 | tests/test-alloc.cpp | 全部提供的修改 hunks | 1–4, 86–115, 129–142, 618–801, 819–826 | 无 |
| TC017 | tests/test-arg-parser.cpp | 选定实际函数/fixture | 319–369, 983–1064, 1516–1591 | 6–11, 230–318, 370–982, 1065–1515, 1784–1892, 1950–2057 |
| TC018 | tests/test-backend-memmove.cpp | 完整新文件 | 1–138 | 无 |
| TC019 | tests/test-backend-ops.cpp | 全部提供的修改 hunks | 1219–1232, 1532–1536, 6558–6620, 7598–7802, 10072–10124, 10253–10267, 10437–10459, 10992–11005, 11014–11057, 11115–11135, 11278–11283, 11370–11386 | 无 |
| TC020 | tests/test-batch-alloc.cpp | 全部提供的修改 hunks | 651–944, 956–964 | 无 |
| TC021 | tests/test-chat.cpp | 全部提供的修改 hunks | 9–15, 1853–1858, 7065–7157, 7187–7461, 7472–7476, 7624–7630, 7636–7641 | 无 |
| TC022 | tests/test-decode-cost.cpp | 沿用前轮内容审阅 | 1–20 | 无 |
| TC023 | tests/test-flash-attn-precision.cpp | 沿用前轮内容审阅 | 1–53 | 无 |
| TC024 | tests/test-flashprefill-attn.cpp | 选定实际函数/fixture | 588–712, 845–958 | 1–587, 713–844, 959–1978 |
| TC025 | tests/test-flashprefill-routing.cpp | 完整新文件 | 1–534 | 无 |
| TC026 | tests/test-flashprefill-select.cpp | 选定实际函数/fixture | 442–542, 661–751 | 1–441, 543–660, 752–2141 |
| TC027 | tests/test-flashprefill-state-envelope.h | 完整新文件 | 1–249 | 无 |
| TC028 | tests/test-flashprefill-state.cpp | 选定实际函数/fixture | 541–605, 1963–2033 | 1–540, 606–1962, 2034–2105 |
| TC029 | tests/test-gpu-recovery-mock.py | 完整新文件 | 1–149 | 无 |
| TC030 | tests/test-gpu-recovery.sh | 完整新文件 | 1–228 | 无 |
| TC031 | tests/test-grammar-indent.cpp | 完整新文件 | 1–180 | 无 |
| TC032 | tests/test-grammar-integration.cpp | 全部提供的修改 hunks | 11–15 | 无 |
| TC033 | tests/test-greedy-unique-host.cpp | 沿用前轮内容审阅 | 1–87 | 无 |
| TC034 | tests/test-json-schema-to-grammar.cpp | 全部提供的修改 hunks | 10–14 | 无 |
| TC035 | tests/test-kv-cells.cpp | 选定实际函数/fixture | 724–861, 1004–1029, 1111–1185 | 1–723, 862–1003, 1030–1110 |
| TC036 | tests/test-llama-archs.cpp | 全部提供的修改 hunks | 114–124, 149–179, 438–443 | 无 |
| TC037 | tests/test-meta-reduce-boundary.cpp | 选定实际函数/fixture | 1093–1218, 2067–2154, 2315–2380 | 1–1092, 1219–2066, 2155–2314, 2381–2419 |
| TC038 | tests/test-model-lazy-mmap.cpp | 完整新文件 | 1–99 | 无 |
| TC039 | tests/test-model-resolution.cpp | 全部提供的修改 hunks | 208–229 | 无 |
| TC040 | tests/test-mtp-workspace.cpp | 完整新文件 | 1–548 | 无 |
| TC041 | tests/test-nanbeige-audit.py | 选定实际函数/fixture | 146–193, 257–347, 588–644, 698–807 | 1–145, 194–256, 348–587, 645–697 |
| TC042 | tests/test-nanbeige-report.py | 沿用前轮内容审阅 | 1–197 | 无 |
| TC043 | tests/test-native-determinism.cpp | 沿用前轮内容审阅 | 1–186 | 无 |
| TC044 | tests/test-predefined-capacity.cpp | 完整新文件 | 1–165 | 无 |
| TC045 | tests/test-predefined-coverage.cpp | 完整新文件 | 1–143 | 无 |
| TC046 | tests/test-qsa-pooled-cache.cpp | 完整新文件 | 1–120 | 无 |
| TC047 | tests/test-quant-type-selection.cpp | 全部提供的修改 hunks | 346–351 | 无 |
| TC048 | tests/test-recurrent-state-rollback.cpp | 全部提供的修改 hunks | 7–20, 74–86, 90–99, 110–118, 121–158, 161–170, 173–228, 236–263 | 无 |
| TC049 | tests/test-rerot-attn.cpp | 选定实际函数/fixture | 198–296, 808–897 | 1–197, 297–807, 898–2113 |
| TC050 | tests/test-rerot-ddvr.cpp | 选定实际函数/fixture | 20–28, 72–120, 140–208, 623–701 | 1–19, 29–71, 121–139, 209–622 |
| TC051 | tests/test-rerot-math.cpp | 选定实际函数/fixture | 446–536, 650–764 | 1–445, 537–649, 765–1766 |
| TC052 | tests/test-rerot-mindmap-parser.cpp | 完整新文件 | 1–343 | 无 |
| TC053 | tests/test-rerot-model-batch.cpp | 完整新文件 | 1–167 | 无 |
| TC054 | tests/test-rerot-model-permute.cpp | 完整新文件 | 1–245 | 无 |
| TC055 | tests/test-rerot-model-single.cpp | 完整新文件 | 1–506 | 无 |
| TC056 | tests/test-rerot-parser.cpp | 选定实际函数/fixture | 1–156, 226–310, 429–470, 630–694 | 157–225, 311–428, 471–629 |
| TC057 | tests/test-rerot-profile.cpp | 选定实际函数/fixture | 1–106, 455–649, 860–983 | 107–454, 650–859 |
| TC058 | tests/test-rerot-q-prep.cpp | 完整新文件 | 1–318 | 无 |
| TC059 | tests/test-rerot-q3-shared-kv-2reader.cpp | 完整新文件 | 1–303 | 无 |
| TC060 | tests/test-rerot-recurrent.cpp | 选定实际函数/fixture | 36–101, 220–256, 878–967 | 1–35, 102–219, 257–877, 968–2005 |
| TC061 | tests/test-rerot-runtime.cpp | 选定实际函数/fixture | 200–296, 964–1032 | 1–199, 297–963, 1033–7877 |
| TC062 | tests/test-rerot-semantic-smoke.py | 完整新文件 | 1–44 | 无 |
| TC063 | tests/test-rerot-shared-block.cpp | 选定实际函数/fixture | 1–56, 230–416 | 57–229, 417–780 |
| TC064 | tests/test-rerot-span-expand.cpp | 完整新文件 | 1–218 | 无 |
| TC065 | tests/test-rerot-thinking.cpp | 完整新文件 | 1–203 | 无 |
| TC066 | tests/test-rerot-todo.cpp | 完整新文件 | 1–231 | 无 |
| TC067 | tests/test-rerot-view.cpp | 选定实际函数/fixture | 829–905, 2333–2445 | 1–828, 906–2332 |
| TC068 | tests/test-server-signal.cpp | 完整新文件 | 1–164 | 无 |
| TC069 | tests/test-server-task.cpp | 完整新文件 | 1–448 | 无 |
| TC070 | tests/test-server-triattention.cpp | 完整新文件 | 1–22 | 无 |
| TC071 | tests/test-spec-direct-hidden.cpp | 完整新文件 | 1–156 | 无 |
| TC072 | tests/test-target-capacity.cpp | 选定实际函数/fixture | 91–186, 504–559, 903–1000 | 1–90, 187–503, 560–902, 1001–1023 |
| TC073 | tests/test-tokenizer-1-bpe.cpp | 全部提供的修改 hunks | 9–13 | 无 |
| TC074 | tests/test-tp5-dependency-ledger.cpp | 完整新文件 | 1–14 | 无 |
| TC075 | tests/test-tp5-gap-tools.py | 完整新文件 | 1–494 | 无 |
| TC076 | tests/test-tp5-hc-coordinate.py | 完整新文件 | 1–86 | 无 |
| TC077 | tests/test-tp5-hc-elementwise-fusion.py | 选定实际函数/fixture | 34–61, 348–427, 485–595 | 1–33, 62–347, 428–484 |
| TC078 | tests/test-tp5-hc-envelope.py | 完整新文件 | 1–22 | 无 |
| TC079 | tests/test-tp5-independent-envelope.py | 完整新文件 | 1–23 | 无 |
| TC080 | tests/test-tp5-match-model.py | 完整新文件 | 1–81 | 无 |
| TC081 | tests/test-tp5-necessity-audit.py | 完整新文件 | 1–313 | 无 |
| TC082 | tests/test-tp5-necessity-certificate.py | 完整新文件 | 1–147 | 无 |
| TC083 | tests/test-tp5-perf-model.py | 完整新文件 | 1–115 | 无 |
| TC084 | tests/test-tp5-plan.cpp | 选定实际函数/fixture | 1099–1257, 1617–1719 | 1–1098, 1258–1616, 1720–1797 |
| TC085 | tests/test-tp5-q-geometry.cpp | 完整新文件 | 1–37 | 无 |
| TC086 | tests/test-tp5-qcheck.cpp | 完整新文件 | 1–61 | 无 |
| TC087 | tests/test-tp5-rank-local-readback.cpp | 完整新文件 | 1–459 | 无 |
| TC088 | tests/test-tp5-row-program.cpp | 完整新文件 | 1–242 | 无 |
| TC089 | tests/test-tp5-sparse-timing.cpp | 完整新文件 | 1–20 | 无 |
| TC090 | tests/test-tp5-submit-pool.cpp | 完整新文件 | 1–31 | 无 |
| TC091 | tests/test-tp5-upfold-indexing.py | 选定实际函数/fixture | 58–232, 371–429, 576–768 | 1–57, 233–370, 430–575, 769–1155 |
| TC092 | tests/test-triattention-kv.cpp | 完整新文件 | 1–244 | 无 |
| TC093 | tests/test-triattention-score.cpp | 选定实际函数/fixture | 241–298, 655–775, 1117–1279 | 1–240, 299–654, 776–1116 |
| TC094 | tests/test-turbo-config.cpp | 完整新文件 | 1–27 | 无 |
| TC095 | tests/test-turbo-quant.c | 全部提供的修改 hunks | 5–63, 132–354 | 无 |
| TC096 | tests/test-vulkan-command-replay.cpp | 选定实际函数/fixture | 1–91, 737–910, 1237–1280 | 92–736, 911–1236, 1281–4628 |
| TC097 | tests/test-vulkan-flash-attn-capacity.cpp | 完整新文件 | 1–411 | 无 |
| TC098 | tests/test-vulkan-gdn-multistep.cpp | 完整新文件 | 1–495 | 无 |
| TC099 | tests/test-vulkan-moe-threshold.cpp | 完整新文件 | 1–318 | 无 |
| TC100 | tests/test-vulkan-scratch.cpp | 完整新文件 | 1–47 | 无 |
| TC101 | tests/test-vulkan-sparse-attn.cpp | 选定实际函数/fixture | 407–468, 541–626, 671–740 | 1–406, 469–540, 627–670, 741–1155 |
| TC102 | tests/test-vulkan-tp5-mesh.cpp | 选定实际函数/fixture | 587–656, 1926–2384 | 1–586, 657–1925, 2385–2713 |
| TC103 | tests/test-vulkan-tp5-output-liveness.cpp | 完整新文件 | 1–142 | 无 |
| TC104 | tests/test-vulkan-tp5-relay.cpp | 完整新文件 | 1–103 | 无 |
| TC105 | tests/test-xkv-admission.cpp | 选定实际函数/fixture | 113–231, 347–400, 489–543 | 1–112, 232–346, 401–488, 544–798 |
| TC106 | tests/test-xkv-backend.cpp | 选定实际函数/fixture | 488–555, 591–616, 848–887 | 1–487, 556–590, 617–847, 888–2605 |
| TC107 | tests/test-xkv-cache.cpp | 选定实际函数/fixture | 552–716 | 1–551, 717–3248 |
| TC108 | tests/test-xkv-canonical.cpp | 选定实际函数/fixture | 31–72, 105–171, 318–335, 444–535, 626–643 | 1–30, 73–104, 172–317, 336–443, 536–625 |
| TC109 | tests/test-xkv-codec.cpp | 选定实际函数/fixture | 1–50, 639–761, 961–1068 | 51–638, 762–960 |
| TC110 | tests/test-xkv-factor-encoding-size.cpp | 完整新文件 | 1–326 | 无 |
| TC111 | tests/test-xkv-factor.cpp | 选定实际函数/fixture | 1–87, 403–653 | 88–402 |
| TC112 | tests/test-xkv-fault-matrix.cpp | 选定实际函数/fixture | 1781–1834, 2111–2158 | 1–1780, 1835–2110 |
| TC113 | tests/test-xkv-fit-residency.cpp | 完整新文件 | 1–255 | 无 |
| TC114 | tests/test-xkv-graph-ref.cpp | 选定实际函数/fixture | 617–724, 1231–1307 | 1–616, 725–1230, 1308–1748 |
| TC115 | tests/test-xkv-graph-runtime.cpp | 选定实际函数/fixture | 1839–1978, 3419–3478 | 1–1838, 1979–3418, 3479–4030 |
| TC116 | tests/test-xkv-hot-cache.cpp | 选定实际函数/fixture | 1340–1427, 1714–1809 | 1–1339, 1428–1713, 1810–1988 |
| TC117 | tests/test-xkv-hot.cpp | 选定实际函数/fixture | 1–31, 277–409, 474–567, 643–657 | 32–276, 410–473, 568–642 |
| TC118 | tests/test-xkv-landmark.cpp | 选定实际函数/fixture | 979–1016, 1698–1791, 1963–2024 | 1–978, 1017–1697, 1792–1962, 2025–2859 |
| TC119 | tests/test-xkv-matrix.cpp | 选定实际函数/fixture | 1–82, 713–872 | 83–712 |
| TC120 | tests/test-xkv-memory-model.cpp | 选定实际函数/fixture | 1–59, 411–497, 564–606 | 60–410, 498–563, 607–689 |
| TC121 | tests/test-xkv-metrics.cpp | 完整新文件 | 1–444 | 无 |
| TC122 | tests/test-xkv-native-seal.cpp | 选定实际函数/fixture | 510–575, 1083–1134 | 1–509, 576–1082, 1135–1252 |
| TC123 | tests/test-xkv-no-dense-mirror.cpp | 选定实际函数/fixture | 1–76, 146–241, 473–535, 581–608 | 77–145, 242–472, 536–580 |
| TC124 | tests/test-xkv-quantized-pack.cpp | 完整新文件 | 1–362 | 无 |
| TC125 | tests/test-xkv-reader.cpp | 选定实际函数/fixture | 876–973, 1578–1732 | 1–875, 974–1577, 1733–3046 |
| TC126 | tests/test-xkv-reconstruct.cpp | 选定实际函数/fixture | 1–63, 165–220, 357–581 | 64–164, 221–356, 582–629 |
| TC127 | tests/test-xkv-runtime.cpp | 选定实际函数/fixture | 2053–2087, 2233–2268, 2325–2371 | 1–2052, 2088–2232, 2269–2324, 2372–2868 |
| TC128 | tests/test-xkv-state-code-stream.cpp | 选定实际函数/fixture | 818–974 | 1–817, 975–1453 |
| TC129 | tests/test-xkv-state.cpp | 选定实际函数/fixture | 775–788, 1378–1460, 1559–1593 | 1–774, 789–1377, 1461–1558, 1594–1628 |
| TC130 | tests/test-xkv-transaction.cpp | 选定实际函数/fixture | 1–49, 290–336, 458–515, 663–711 | 50–289, 337–457, 516–662, 712–1061 |
| TC131 | tests/test-xkv-tri.cpp | 选定实际函数/fixture | 1519–1586, 3108–3187 | 1–1518, 1587–3107, 3188–3321 |
| TC132 | tests/test-xkv-turbo-shift.cpp | 完整新文件 | 1–314 | 无 |
| TC133 | tests/test-xkv-vulkan-attention.cpp | 选定实际函数/fixture | 1–160, 372–570 | 161–371, 571–917 |
| TC134 | tests/test-xkv-vulkan-factorize.cpp | 选定实际函数/fixture | 402–583 | 1–401, 584–997 |
| TC135 | tests/test-xkv-vulkan-landmark-build.cpp | 完整新文件 | 1–479 | 无 |
| TC136 | tests/test-xkv-vulkan-landmark.cpp | 选定实际函数/fixture | 186–281, 545–634, 989–1015 | 1–185, 282–544, 635–988, 1016–1609 |
| TC137 | tests/test-xkv-workspace-bound.cpp | 选定实际函数/fixture | 316–341, 395–466, 505–598, 773–794 | 1–315, 342–394, 467–504, 599–772 |
| TC138 | tests/tp5-command-tick.comp | 完整新文件 | 1–4 | 无 |
| TC139 | tests/tp5-hc-quant-helper.cpp | 完整新文件 | 1–40 | 无 |
| TC140 | tests/tp5-kernel-service.cpp | 完整新文件 | 1–129 | 无 |
| TC141 | tests/tp5-late-service.cpp | 完整新文件 | 1–311 | 无 |
| TC142 | tests/tp5-native-traffic.cpp | 完整新文件 | 1–132 | 无 |
| TC143 | tests/tp5-pack-probe.cpp | 完整新文件 | 1–204 | 无 |
| TC144 | tests/tp5-peer-collective.comp | 完整新文件 | 1–80 | 无 |
| TC145 | tests/tp5-peer-collective.cpp | 完整新文件 | 1–234 | 无 |
| TC146 | tests/tp5-peer-timeline.comp | 完整新文件 | 1–37 | 无 |
| TC147 | tests/tp5-stream-probe.cpp | 完整新文件 | 1–69 | 无 |
