# 补充 193 路径的实际内容审计

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

范围：旧 fork 相对锁定上游 A 的剩余 193 文件，补足此前未传输的内容。审计日期 2026-10-18。

本报告识别可以独立保留的小行为，明确新旧边界和未读范围；没有将 XKV、TP5、RERoT 整块打包为“已审/无用”。本轮不移植、不运行设备测试，不把历史测试声明当成当前 clean 分支的验证。

## 来源、覆盖与可信边界

- 源 diff：audit-remaining.diff，6,627,565 bytes，193 个文件 header、206 个 hunk。
- SHA256：c23d8a63815b0b1f776053e4c52cbf8fe9e929697f2e933331eb1639651ad1f9
- 上游 A：cda0f9256665845cd9d6f3a0176880883f653a3d。旧 fork baseline 为父任务提供的 2777543b2。
- 全文读完 120 个 ≤500 行的小代码/头文件/shader；另全文读热槽池 770 行、FlashPrefill 策略 582 行、两份小元数据。共 124 个新文件全文。
- 10 个现有文件的全部删除/新增/上下文 hunk 读完；并不声称读完它们未修改的全文。
- 59 个大源文件、头文件、研究文档或设备数据只读下面指定范围。JSON 逐路径列出未读补集。
- 这些范围与完整源 inventory 集合完全一致，无漏路径、无多报路径。空白行可在显示时省略，实际代码和注释不省略；此前输出截断的范围在登记全文前补读。

机器账本：audit-remaining-content-coverage.json；可合并顶层 files 数组，字段含 path/read_scope/ranges/review_level/finding_ids/behavior。小脚本 finalize_remaining_audit.py 可从保留的审计输入重复生成。

## 直接影响 fork 取舍的结论

以下行为值得保留目录，且其中多项可组成真正不调用 A 的 B：checked 容量与相位 planner、实际 completion 驱动的 slot 状态机、64 KiB scratch roundup、稳定范围/liveness 数学、KV 变换参考、FlashPrefill 的纯策略/packing/MLO、same-context state envelope、热槽元数据池、固定 LE codec 描述符、Tri 压力/去重/稳定排序、RERoT 数学参考。它们各自的边界缺口在下文逐项列明，不能原样一揽子移植。

prefill/decode compute 释放、统一物理 KV 与逻辑窗口、prefill priority 已在主任务通过明确 A seam 和 C 编排分别处理。本轮新发现的 metadata pool 或相位 planner 不替代这些真实执行能力。较大的物理 KV 池还可能扩大 attention 的实际 compute span，文档不能承诺 scratch 被逻辑窗口自动限制。

GPU 小功能也单列：mask 的稳定压缩、非对齐 word-owned copy、pair L2 舍入保留、PLE precise 乘法、Turbo3 的 2-byte 对齐、staged top-k 的 native tie fallback、小 subgroup 归约、有效 row 先于读写。这些有明确价值，但仍需要 A 后端接线与真实设备回归；不能把文件搬到 B 就称解耦完成。

发现若干旧私有路径的具体缺口：GDN multistep 的 active/rollback/gamma、XKV landmark 的 barrier 一致控制流与全局 status 覆盖、reconstruct 的 layer slot 上界、canonical 64-bit pos low word、profile ring 发布、hot pool 恢复校验、graph refresh 的失败后部分修改。它们没有进入 clean 分支，本报告不将静态疑点升级为该分支的运行故障。

## 建议优先级

| 优先级 | 具体候选 | 实施边界 |
|---|---|---|
| 高 | checked scratch / 相位容量 / slot 状态机；FlashPrefill 纯 route；Tri 压力与稳定顺序 | 自足 B record 与数学；C 执行真实 A 调用 |
| 高 | codec 固定 LE parser；state envelope；热槽 metadata | 先补 null、身份、restore、错误事务性与所有权，不含物理 KV 分配 |
| 中 | MLO、Hadamard/RoPE、RERoT reference、workspace carver | 冻结数学域、有限值、空集合、overflow 和失败输出合同 |
| 中 | memmove、mask compact、tie-preserving top-k、小浮点融合 | 明确最小 A 后端补丁，真实 before/after 回归后再比较收益 |
| 暂不迁 | TP5 collective、LateBind、GDN multistep、XKV device graph、完整稀疏 attention | 私有依赖大，静态缺口未闭合，没有本轮质量/性能资格 |
| 历史 | 研究账、设备导出、BIOS 记录、旧 manifest | 保留必要索引；不作为当前产品功能或验证证据 |

## 逐行为发现

### RMC01 — 64 KiB scratch 容量向上取整

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-scratch.hpp:10–17。

**实际行为：** 在容量限制内做 64 KiB roundup，接近限制时保留 requested，避免取整溢出；返回值不是实际分配或实际空闲显存。

**A/B/C 边界：** 函数只有 size_t 数学，能直接成为 B；C 把结果用于自己拥有的分配。

**风险与证据限制：** requested 本身超过 limit 时仍返回 requested，调用方必须负责拒绝；不能把 capacity 当 dispatch 的有效长度。

**保留建议：** 保留为独立小 B 候选；关联 test-vulkan-scratch.cpp 的实际边界测试。

### RMC02 — Predefined 四相位与 checked 容量规划

**证据：** ggml/include/ggml-predefined.h:1–175; ggml/src/ggml-predefined.cpp:48–211。

**实际行为：** 分别描述 prefill、decode、draft、verify 的有效行、容量和间接 dispatch，工作区为相位需求最大值加 persistent，拒绝时不覆盖输出。

**A/B/C 边界：** 独立 POD、整数运算可移 B；去掉 ggml.h/GGML_API，并由 C 向 A 请求真实执行。

**风险与证据限制：** 规划器不会使 A 的图自动认识 active_rows，也不会释放 A 的 compute buffer；不能替代已经登记的 phase seam。

**保留建议：** 保留纯规划和溢出合同，执行能力单独验收。

### RMC03 — 双 slot 元数据与真实完成值

**证据：** ggml/src/ggml-predefined.cpp:213–323; ggml/include/ggml-predefined.h:1–175。

**实际行为：** slot acquire、submit、retire 区分逻辑 epoch 与各 rank 实际 native completion；部分提交只等待确实提交的 rank，尚未提交的 acquire 可撤销。

**A/B/C 边界：** 状态机可独立 B，C 提交操作并回填真实结果；不让 B 在回调内直接操作 A。

**风险与证据限制：** 调用必须外部串行化；单调 epoch 和完成值不能由 CPU 人造 signal 冒充。

**保留建议：** 保留有限状态机和失败边界设计，不将它包装成现成多卡执行器。

### RMC04 — 批量设备 copy 预检

**证据：** ggml/include/ggml-device-copy.h:1–75; ggml/src/ggml-device-copy.cpp:1–88。

**实际行为：** 批量范围统一预检，明确设备执行合同，不默认主机 fallback。

**A/B/C 边界：** 实际 copy 依赖新增 backend 接口与私有实现，属于最小 A seam 或独立后端项目，不能只移动两个文件。

**风险与证据限制：** 预检可使无效范围整批不提交，但 native 录制或提交中异常不等于撤销已经提交的写入。

**保留建议：** 保留合同与真实 buffer 回归；是否移植 API 由具体消费者决定。

### RMC05 — 非对齐 byte memmove 的单 writer

**证据：** ggml/src/ggml-vulkan/vulkan-shaders/memmove_bytes.comp:16–31。

**实际行为：** 每个 destination word 仅一个 invocation 写，掩码保留首尾未覆盖字节；可服务跨 row 和非对齐 byte 范围。

**A/B/C 边界：** 需要 host 完整范围检查和对 overlap 的两遍 scratch copy，shader 不是完整 memmove。

**风险与证据限制：** u32 offset 加法、向 word 取整后的 storage 范围必须先证明；不能以非重叠 copy 实现代替重叠语义。

**保留建议：** 小而明确的 A 后端候选；关联兄弟审计实际读过的 test-backend-memmove.cpp:24–124。

### RMC06 — 非有限 activation 修整

**证据：** ggml/src/ggml-vulkan/vulkan-shaders/sanitize_nonfinite.comp:1–24。

**实际行为：** NaN 变 0，正负 Infinity 变正负 65504；所有有限输入，包括大于 65504 的值，都保留。

**A/B/C 边界：** 数学策略可 B；GPU dispatch 需要 A hook，且必须明确是有损策略。

**风险与证据限制：** fast-math 可能使非有限判断失效；它不是全范围 F16 clamp，更不是修复上游数值问题的证据。

**保留建议：** 保留为显式策略与边界用例，避免默认悄悄清洗。

### RMC07 — KV Hadamard 与 RoPE 变换小函数

**证据：** src/llama-kv-transform.h:35–93。

**实际行为：** 可选 unsigned normalized Hadamard 与 Turbo signed WHT 是两种不同变换；相位旋转保留已经施加的 YaRN 幅度。

**A/B/C 边界：** Hadamard/RoPE 算法可纯 B；codec decode 需要 C 提供公开 GGML 转换或明示 seam。

**风险与证据限制：** 不能把这份 unsigned 变换当成当前 Turbo WHT，也不能重复施加或删除 YaRN scale。

**保留建议：** 保留数学参考和精确域说明。

### RMC08 — lazy pread 的局部去重与工作线程

**证据：** src/llama-lazy-reader.h:1–147; src/llama.cpp 新版:160–220,381–413。

**实际行为：** 对 row 请求排序，chunk 内重复 row 复用，pread 处理 EINTR，工作线程异常收集后传播。

**A/B/C 边界：** 文件读取与调度可成为拥有 fd 的独立 B；模型 tensor/lazy PLE 接入仍在 A 私有加载链。

**风险与证据限制：** n_threads=0 时没有 worker，错误路径访问 errs[0]；base+row*stride 未作完整 overflow 防护。

**保留建议：** 候选保留读取组件，修正边界后再接 C；没有通用模型加载提速实证。

### RMC09 — persistent hidden result/carry/seed 所有权

**证据：** src/llama-predefined-hidden.h:1–101; src/llama-predefined-hidden.cpp:1–389。

**实际行为：** 独立拥有 result、carry、seed device tensor，检查来源 row 的连续切片和启动容量；目标是图后持久输出和跨阶段交接。

**A/B/C 边界：** 真实依赖 A context、scheduler、backend、hidden tensor 生命周期；不能以 B 回调 A 假装解耦。

**风险与证据限制：** 跨来源 generation=0 可能与初始 synchronized_generation=0 命中 skip-sync；需实际 consumer 合同证明。此项未移入 clean 分支。

**保留建议：** 保留行为目录；只有需要 direct-hidden 时设计显式 A API 并以真实 queued compute 验收。

### RMC10 — predefined context/session 接线

**证据：** src/llama-predefined.cpp:1–123; src/llama-predefined-session.h:1–11; src/llama-ext.h 全部变更 hunk。

**实际行为：** 把 capacity frame 与 context 执行、persistent feature、pre-RoPE Q 暴露和回收诊断接到真实 A。

**A/B/C 边界：** 外部可做 C，但这些新增符号不属于锁定上游 A 的既有公开 API。

**风险与证据限制：** queued 执行失败、借用 tensor 生存期和更新 generation 不可由 decorator 自动解决。

**保留建议：** 如后续消费者需要，集中登记最小 API seam；不整体复制私有 context 扩展。

### RMC11 — TP5 profile 聚合账本

**证据：** ggml/include/ggml-tp5-profile.h:1–99; ggml/src/ggml-tp5-profile.cpp:1–196。

**实际行为：** 区分提交、等待、copy、compute 等阶段及计数，提供较细的性能归因。

**A/B/C 边界：** 记录格式与 owned ledger 可 B；采集实际 backend 时间必须 C 或 A hook。

**风险与证据限制：** 计数 atomics 不保护全局 g_prof 指针和并发 begin/end，多 context 生命周期并非线程安全。

**保留建议：** 保留字段语义，改为显式所有者后复用；历史计时不是当前 clean 性能。

### RMC12 — 地址范围与 graph liveness

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-liveness.hpp:1–91。

**实际行为：** 用减法检查地址区间和非零地址，结合 graph 使用区间识别物理 alias。

**A/B/C 边界：** 范围数学可 B；graph 节点遍历依赖 GGML tensor 及 caller 给出的真实物理范围。

**风险与证据限制：** 仅名称、offset 或 byte size 相同不足以证明是同一个活 buffer。

**保留建议：** 保留 checked 范围函数与 physical identity 原则。

### RMC13 — TP5 command 录制与 native owner

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-command.h:1–280; ggml-vulkan-internal.h:1–223。

**实际行为：** 命令、owner、buffer 身份和后端录制路径集中描述，为 replay 与跨 phase 重绑建立合同。

**A/B/C 边界：** 依赖 Vulkan command pool、private backend structs 和 pipeline 实例，无法只靠公开 llama API 安装。

**风险与证据限制：** 类型声明和工具类不能证明 buffer rebinding 或异步清理正确，必须运行真正 submitted command 回归。

**保留建议：** 保留审计依据；不将整套私有 backend header 放入 B。

### RMC14 — predefined coverage 分类器

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-coverage.h:1–202。

**实际行为：** 按 dispatch 类型、copy 和计数把路径分为不同 coverage 情况。

**A/B/C 边界：** 纯分类逻辑可 B 或测试工具。

**风险与证据限制：** dispatch 数量相等或没有 copy 不能证明状态写入、alias 和 barrier 语义等价。

**保留建议：** 保留诊断，输出应标明这是覆盖分类而非正确性证明。

### RMC15 — 依赖 ledger 的 epoch 合同

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-dependency-ledger.h:1–13。

**实际行为：** 用 epoch/状态顺序表达依赖，不用耗时相近冒充依赖已经满足。

**A/B/C 边界：** 小型独立状态数据可 B。

**风险与证据限制：** 它不替代 native completion，也不保证设备间内存可见性。

**保留建议：** 保留明确的小合同。

### RMC16 — Q geometry 环境值解析

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-q-geometry.h:1–16。

**实际行为：** 有限三个 geometry 选项的确定性解析和默认值。

**A/B/C 边界：** 独立配置解析，可 B；C 决定是否传给真实后端。

**风险与证据限制：** 解析成功不能证明 shape 或 pipeline 在当前设备可用。

**保留建议：** 低成本工具候选，避免为这个小 parser 迁入 TP5。

### RMC17 — Q payload 完整性检查

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-qcheck.h:1–45。

**实际行为：** 结合 epoch、active extent 与 XOR 检测载荷在 handoff 时是否符合预期。

**A/B/C 边界：** 格式/检查器可拆，但旧实现直接使用 AVX2 和特定载荷布局。

**风险与证据限制：** 缺通用 ISA guard/runtime dispatch，XOR 可碰撞，也不是 host/device coherence 的证明。

**保留建议：** 诊断候选，先补 portability，再由 C 使用。

### RMC18 — active row 几何与 mailbox 布局

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-rows.h:1–248。

**实际行为：** 把有效行数、capacity、stage 宽度、两 bank、Q sidecar 的数量与位置分别描述。

**A/B/C 边界：** geometry 数学可 B；mailbox/sidecar 为私有 wire ABI。

**风险与证据限制：** 无 frame 不必然是一行；地址与 epoch 检查自身不提供同步。

**保留建议：** 保留几何和实际 extent 合同；wire 仅归档。

### RMC19 — 稀疏 GPU timestamp 采样

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-sparse-timing.h:1–41。

**实际行为：** 只收集实际选择的 query 范围，不等待未执行的 timestamp slot。

**A/B/C 边界：** 选择索引与差值计算可 B，Vulkan query fetch 归 C/A。

**风险与证据限制：** 未取到时间应保留 absent，不能填 0 当作测量值。

**保留建议：** 保留小诊断能力。

### RMC20 — 固定 worker submit pool 的失败处理

**证据：** ggml/src/ggml-vulkan/ggml-vulkan-tp5-submit-pool.h:1–70。

**实际行为：** 固定五个 worker，错误或 timeout 使 pool 不再复用，并等待 worker 退出。

**A/B/C 边界：** 线程控制可独立，但 native 提交动作仍需 C 提供真实结果。

**风险与证据限制：** timeout 后 join 全部 worker：native call 永久阻塞时不是硬时间上限；已提交部分必须独立 retire。

**保留建议：** 保留失败模型，勿宣称有界取消或通用五卡调度。

### RMC21 — FlashPrefill 独立配置与相位策略

**证据：** src/llama-flashprefill.h:1–139; src/llama-flashprefill.cpp:1–582。

**实际行为：** 完整实现 config/row/exec 校验、模型与 adapter 标识分域 FNV、GQA packed layout、call/logical dense tail、角色 route 和 checked scratch 估计。

**A/B/C 边界：** 没有 A 调用；把自有 enums/POD 自足定义放到 B，C 用 route 选择真实 A 操作。

**风险与证据限制：** required 模式的强制失败由 caller 执行；packed row 超范围仅返回非 tail，caller 必须先验证。FNV 非强身份哈希。

**保留建议：** 高价值独立 B 候选，但保留策略不等于交付 sparse attention。

### RMC22 — FlashPrefill 布局、packing 与 fixture

**证据：** src/llama-flashprefill-pack.h:1–63; llama-flashprefill-fixture.h:1–114; llama-flashprefill-layout.h:1–65,439–503。

**实际行为：** 区分 logical、physical、storage、virtual 位置，描述 run/fragment/row packing；fixture 构造参考输入；准入预算按容量和分裂数约束。

**A/B/C 边界：** record packing 可 B；真实 cells 视图与 graph owner stamp 需要 C/seam。

**风险与证据限制：** 本轮没有全文读 503 行 layout；split=4、run=65 是准入预留，不是当前性能测量。多模态、SWA 等 gate 不可省。

**保留建议：** 保留 packing 小机制与显式能力判断。

### RMC23 — state envelope 与 same-context 恢复

**证据：** src/llama-flashprefill-state.h:1–192; llama-flashprefill-state.cpp:1–269。

**实际行为：** 短 marker 加 64-byte identity body，避免旧 read_string 把 magic 视为约 1.18 GB 长度；匹配 process nonce、context serial、adapter generation、policy。

**A/B/C 边界：** B 可负责 framing，C 包裹 A 的公开 state bytes。

**风险与证据限制：** 实现用 host-endian memcpy，尽管注释称 LE；同进程同 context 身份不等于可持久化模型权重身份。

**保留建议：** 优先候选：先明确 portability 和身份范围，再独立测试截断、过期、上限。

### RMC24 — FlashPrefill 计量单位与 slice 提交

**证据：** src/llama-flashprefill-metrics.h:1–347; llama-flashprefill-metrics.cpp:1–105,650–726。

**实际行为：** 按 slice 汇总实际 row、visible token、exact token、timing，区分 measured 与 absent，失败 slice 不直接当成功量计入。

**A/B/C 边界：** owned counters、JSON/Prometheus 文本可以 B；实际事件从 C 输入。

**风险与证据限制：** 实现只抽查上述边界，未证明全部中间累计路径；不同单位不能直接相除得到任意节省率。

**保留建议：** 保留计量合同与 serializer，当前无需导入 sparse kernel。

### RMC25 — 均值、max-energy 与 online M/L/O 数学

**证据：** ggml/src/ggml-flashprefill.cpp:1382–1776。

**实际行为：** 完整选读 mean、select、MLO init/add/sink/merge/finalize：mean 双精度累计，partial 不进入能量，mandatory/exact_all/容量先判，multiplicity 用 log(count)，空 MLO 是 merge identity。

**A/B/C 边界：** 选读纯指针数学可 B，不要把整份 GGML wire/parser 源称独立。

**风险与证据限制：** 部分 MLO 操作在后续 finite 检查失败前已写 numerator；double finite 转 float 仍可溢出；空侧 merge 的 m/l 验证不完整。

**保留建议：** 保留参考算法；明确错误后输出与 in-place 合同再生产化。

### RMC26 — FlashPrefill GGML wire 与新增 op

**证据：** ggml/include/ggml-flashprefill.h:1–417; vulkan-shaders/flashprefill-interface.h:1–405。

**实际行为：** C++/GLSL 共享字段宏、plan error、capacity 与实际 count、POOL/SELECT/ATTN 数据布局一致定义。

**A/B/C 边界：** 是新增 GGML op 与 private dispatch 合同，不是 A 已存在的公开 hook。

**风险与证据限制：** 冻住的私有 enum 数值不能带入干净 A；只复制 shader 而不同步字段/支持检查不可运行。

**保留建议：** 保留 wire 设计文档和解析用例；完整执行路径暂归档。

### RMC27 — FlashPrefill POOL/SELECT/ATTN/MERGE 执行细节

**证据：** flashprefill_pool.comp:1–249; merge.comp:1–92; select.comp:1–50,597–681; attn.comp:1–50,488–572; ggml-cuda/flashprefill.cu:1–50,1433–1522。

**实际行为：** full/partial block 分开，sink 只入一个 split；SELECT 稳定 compaction 后 tail=-1，按实际 tiles 做 grid-stride；CUDA 使用独立 phase0/phase1 和分 row warp。

**A/B/C 边界：** 依赖 FP wire、新 op、plan 错误提升和 backend barrier，不能靠 B 接口替换 A attention。

**风险与证据限制：** pool 某失败分支写 0 与 NaN 注释相反；merge denominator 排除 Inf L 而 numerator 未同等排除；已有 plan error 直接 return 并不清 output。大 shader 中段未全文读。

**保留建议：** 保留各小行为清单；无当前模型质量/性能资格，不整体迁入。

### RMC28 — 有限 mask 稳定压缩

**证据：** flash_attn_sparse_compact.comp:1–23; flash_attn_sparse_compact.glsl:48–107。

**实际行为：** 保留所有有限 mask，包括 -65504；排除 -Infinity；NaN/+Infinity 或 compact 溢出用 -1 标记密集 fallback。subgroup prefix 保留原始 KV 次序。

**A/B/C 边界：** 算法可 B 参考，实际 GPU fast path 需 A dispatch 接线。

**风险与证据限制：** 把所有负 mask 当不可见会改变 softmax；必须保持原 mask 与物理 KV 索引对应。

**保留建议：** 小而明确的潜在 A 修复/优化，单列待真实 CPU/GPU 对拍。

### RMC29 — GPU 分阶段 unique argmax

**证据：** argmax_stage1.comp:47–95; argmax_stage2.comp:43–123。

**实际行为：** 先分块扫描再归约，仅有限唯一最大值用快结果；任何非有限值、tie 或 -FLT_MAX sentinel 返回原 row 的归约路径。

**A/B/C 边界：** GPU dispatch 需 A seam；纯 CPU unique-max 已在另一独立采样 B 交付。

**风险与证据限制：** 这里连 -Infinity 也 fallback，不能与 CPU 支持 -Infinity mask 的合同混同；tie 不可换成 id 排序。

**保留建议：** 保留算法与两种合同的区别；无新增 GPU 移植。

### RMC30 — pair L2 与 PLE gate 的浮点次序

**证据：** l2_norm_pair.comp:40–80; ple_router_gate.comp:53–81。

**实际行为：** pair L2 沿用单 row 的 tree 次序；PLE 乘法用 precise 避免 FMA 改变已有中间舍入。

**A/B/C 边界：** 需要 A fusion 的严格 shape 与 alias gate。

**风险与证据限制：** 数学等式等价不意味着选词相同；不能无保护替换原 kernel。

**保留建议：** 保留小融合候选，独立对拍之后再决定收益。

### RMC31 — Turbo cooperative-matrix 域与对齐

**证据：** flash_attn_turbo_cm2.glsl:1–46。

**实际行为：** 保留 KV 的 stored WHT domain，load alignment 使用 2 bytes 以覆盖 50-byte Turbo3 block。

**A/B/C 边界：** 依赖当前 A Turbo layout 和 private CM2 dispatch。

**风险与证据限制：** 必须与当前格式及设备 feature 一致；不可引入旧私有类型重编号。

**保留建议：** 可供后续小后端修复参考，本轮不移植。

### RMC32 — Qwen attention MMVQ、投影与 prep

**证据：** qwen4_attention_mmvq.glsl:73–245; qwen4_attention_prep.comp:161–255; qwen4_attention_projections.comp:1–50,749–833。

**实际行为：** Q8 activation 跨投影/多 token 复用，Q/K RMS+RoPE 与 KV write 保留 native F16 中间舍入；分开完整 tile 和尾 tile。

**A/B/C 边界：** 特定 Qwen 架构和 quant mix 的 private fusion，不是通用 BLAS B。

**风险与证据限制：** MAX_TOKENS/min 的截断需 caller 准入；ragged shape、subgroup geometry 和 alias 必须证明。大投影中段未全文读。

**保留建议：** 保留精确候选；没有独立净收益则按用户要求暂舍弃完整融合。

### RMC33 — GDN 单步三段融合

**证据：** qwen4_gdn_segment_prep.comp:1–226; delta.comp:1–180; norm.comp:1–65。

**实际行为：** 将 GDN prep、递推 delta、normalization 分段，正常路线传入 epsilon 并使用共享 gamma，单步写回状态。

**A/B/C 边界：** 依赖模型 recurrent tensor 和 backend fused graph，B 单独文件不可接管 A state。

**风险与证据限制：** 真实回滚、active 行和隐藏状态更新要用 A 模型验证，不能由 shader 名称宣称支持 MTP。

**保留建议：** 保存行为与参考，完整路径不移植。

### RMC34 — GDN multistep scaffold 的不完整合同

**证据：** qwen4_gdn_multistep_prep.comp:156–167; delta.comp:105,145–204; norm.comp:54,79–88。

**实际行为：** 尝试把多步 recurrent 执行合并，但 active=0/超 capacity 被改成 full capacity；多步 delta 未逐步写 rollback snapshot。

**A/B/C 边界：** 仍依赖私有 graph 与 recurrent state。

**风险与证据限制：** prep 在 tid<128 保护前做权重读而 workgroup 512；norm 硬编码 epsilon=1e-6 并按 head 索引 gamma，与单步合同不同。源码 PRODUCTION 注释不是可达路径证明。

**保留建议：** 不作为已完成功能保留；逐缺口登记，未来先做真实反例回归。

### RMC35 — HC 四 stream fold 与伴随 Q8

**证据：** qwen4_hc_segment_norm.comp:271–425; qwen4_hc_up_fold.comp:26–43。

**实际行为：** 按 native 的四 stream 顺序折叠；Q8 packing 在复用 LDS 前有 barrier，避免前一归约仍读取。

**A/B/C 边界：** 特定 HC 图融合，需 private dispatch/shape。

**风险与证据限制：** 不能仅按代数重新排序并称 exact；Q8 中间层是额外数值合同。

**保留建议：** 保留细粒度机制，不整体纳入 clean。

### RMC36 — MoE quant pair 与常量 full-tile

**证据：** qwen4_moe_down_k128.comp:22–45,166–171; qwen4_moe_projections.comp:1–50,480–564。

**实际行为：** IQ4 K128 配对 16-lane 专家，gate/up 共享读取；用常量完整 tile 避免动态 min 改变 RADV loop lowering 与 contraction。

**A/B/C 边界：** 架构/quant/driver 窄包络下的 A 融合。

**风险与证据限制：** 源码对 reduction-order 的预期不是跨 driver 证明；标量 gate 的 subgroup 与 WG 关系需要 host 条件。

**保留建议：** 保留候选与限定，未见当前独立收益前暂不迁。

### RMC37 — router staging 与 native tie fallback

**证据：** qwen4_router.comp:82–183; topk_moe_staged.comp:68–74,131–256。

**实际行为：** router 输入在 expert tile 内复用；top-k 先存 512 logits 避免重叠写，唯一最大值迭代快选，tie 则重走 native value-only bitonic argsort 次序。

**A/B/C 边界：** 真实路由融合依赖 A；tie 参考算法可 B 测试。

**风险与证据限制：** 不能把 tie 改成 id 升序；实现固定一 row、最多 16 selected，若干配置字段未实际使用。

**保留建议：** 独立记录 tie/alias 修复价值，不因属于 TP5 而遗漏。

### RMC38 — TP5 copy、pack、sum、relay 的实际有效长度

**证据：** tp5_add_rows.comp:36–55; tp5_gather_p1.comp:40–80; tp5_predefined_add_rows.comp:19–26; tp5_relay*.comp 全文; tp5_bda_push_f16.comp 全文。

**实际行为：** active 先于 load/store；gather 检查乘积和 epoch；F16 pack、rank 固定次序 sum、两 bank 交接分别存在。

**A/B/C 边界：** 这些是 private transport shader，公开 A 没有安装该 collective 的接口。

**风险与证据限制：** coherent/volatile、XOR、CPU atomic 不自动给出跨设备 visibility；spinMax=UINT_MAX 可能自增回绕；ready 合并若只支持一 WG 不可扩大 grid。

**保留建议：** 保留各小合同和失败用例，整套运输层暂归档。

### RMC39 — 小 subgroup 的 partial 汇总

**证据：** tp5_hc_subgroup.glsl:7–22; tp5_hc_norm_shared.comp:1–117。

**实际行为：** 32-thread helper 显式汇总所有 subgroup partial，即使第一 subgroup 的 lane 数不足以覆盖 partial 数也可逐步处理。

**A/B/C 边界：** 可作为独立 shader 数学组件，但实际 graph 仍需 A 后端。

**风险与证据限制：** 不要把此 helper 的已读逻辑外推到所有旧 HC norm。

**保留建议：** 保留小可移植归约模式，须设备回归再接。

### RMC40 — HC publish/resume/latebind 边界风险

**证据：** tp5_hc_publish.comp:65–107; tp5_hc_resume.comp:57,239–247,275–279; tp5_hc_latebind.comp:875–959。

**实际行为：** publish 包含当前 epoch、live count 与载荷诊断；latebind 输出 RMS sufficient statistics，resume 执行等待后的 residual/norm。

**A/B/C 边界：** 依赖私有 HC/transport ABI 与真实 owner。

**风险与证据限制：** resume partial 数组 WG/32，保护条件却可容许 wave16 的更多 subgroup；latebind 887–889 的 vec2 scatter 尾与每 token 12 words 注释不符，streams=4 时有跨 token 重叠疑点，须消费者复核。

**保留建议：** 风险归档，不能当作可直接独立复制的小库。

### RMC41 — RERoT 独立数学参考

**证据：** ggml-cpu/ggml-cpu-rerot.cpp:1–460; src/llama-rerot-math.h:1–493; math.cpp:1–147,616–751,910–1007。

**实际行为：** CPU oracle 真正 std-only；共享 key 外循环、每 reader 单独 softmax，另有 span/prefix、grid frontier 验证与 naive 验证反例。

**A/B/C 边界：** 纯函数可 B；整份 math.cpp 的 Q-prep 参考会调生产 GGML，必须选取独立部分。

**风险与证据限制：** enum/有限值/有符号位置 overflow 校验不足；attn_state_merge 拒绝空/非空，而 FP MLO 用空 identity；skip-mass bound 未约束负 scale；FNV order signature 不可能保证 iff。

**保留建议：** 高价值参考 B 候选，先冻结数学语义与边界；不宣称 RERoT 生产质量或性能已保留。

### RMC42 — RERoT GPU attention 与辅助 shader

**证据：** ggml-cuda/fattn-rerot.cu:1–390; rerot_q_prep.comp:72–150; rerot_span_expand.comp:23–65; rerot_shared_kv_2reader.comp:59–115; flash_attn_rerot.comp:1–50,646–730。

**实际行为：** indexed entry 范围的全局 softmax、split M/L/O、single-split sink；Q phase prep 与 span 二分展开；inactive/invalid Q 输出 NaN。

**A/B/C 边界：** 需要新增 A graph op、DDVR 视图和 private backend dispatch。

**风险与证据限制：** two-reader shader 是串行 stub，reader 外循环会读 K/V 两次且 VD>128 输出 0；unsupported rope 的 copy fallback 需要 host 拒绝。

**保留建议：** 保留 span 小算法，别将 stub 称作已经完成 shared-load 优化。

### RMC43 — RERoT visibility 与 fragment 边界

**证据：** src/llama-rerot.h:1–65,908–972; src/llama-rerot.cpp:1–50,3717–3806。

**实际行为：** PUBLIC peer 必须来自已提交的旧 frontier；own gated 行同时受 ownership、causal storage、episode 与 node 限制。fragment 在 phase bias、virtual block、visibility/gated 变化时切分。

**A/B/C 边界：** 纯 metadata predicate 可 B，C 管 reader 计划；实际共享 KV/RoPE 图需要显式 A seam。

**风险与证据限制：** 本轮未读大型 cpp 中段；不能只凭这些 predicate 声称整个 DAG 一致性。

**保留建议：** 单列保留算法合同，真实共享执行暂归档。

### RMC44 — RERoT profile ring 的发布缺口

**证据：** src/llama-rerot-profile.h:1–278; profile.cpp:1–35,73–112,551–578。

**实际行为：** 固定容量 event ring、scratch peak 与 profile 入口；scratch peak 的 CAS 小 helper 可独立。

**A/B/C 边界：** owned event recorder 可 B；backend counters 从 C 输入。

**风险与证据限制：** write_idx relaxed 增加早于普通 event 写入，reader 普通读；wrap/reset/并发 push 没有发布保护，全局 profiler 生命周期另有竞态。

**保留建议：** 不能原样宣称 thread-safe；若复用改成显式单 writer 或完整发布协议。

### RMC45 — Tri 的压力、保留数、去重与顺序

**证据：** src/llama-triattention.h:1–492; .cpp:1644–1778; ggml-cuda/triattention-score.cu:1–50,473–562。

**实际行为：** ratio 默认 3/32、tail floor，分别分类 KV/recurrent/maintenance；recurrent-only 不触发 KV reclaim；物理 cell 只评分一次，storage/frontier/cell/index 稳定排序。

**A/B/C 边界：** 这些 policy/helper 可纯 B；真实 scorer 读私有 KV 和量化、CUDA buffer，因此执行和有损淘汰需 C/seam。

**风险与证据限制：** tail_guard 可大于 logical_tokens；hash 容器分配与 32-bit reserve 边界需要定义；GPU readback 明确同步，不是异步免费路径。

**保留建议：** 保留小行为目录，不把纯 helper 等同实际压缩或质量通过。

### RMC46 — 热槽 metadata pool 的 reserve/commit/release

**证据：** src/llama-xkv-hot.h:1–268; src/llama-xkv-hot.cpp:1–770。

**实际行为：** std-only：最小 slot 优先 heap、固定低装载哈希、backward-shift 删除，batch 预检后无分配变更，RAII 撤销未提交 reservation，generation/id/expected slot 匹配。

**A/B/C 边界：** 可以成为真正 B metadata pool；它不分配或共享 A 的物理 KV buffer。C 必须拥有使用该 metadata 的动作。

**风险与证据限制：** pool 要长于 raw-pointer reservation；generation 由 caller 提供；restore_state:452–482 未充分检查 table_vals/mask/heap 等，不能暴露成通用反序列化；noexcept 错误字符串赋值可能再次抛出。

**保留建议：** 高价值候选，先收窄 restore 接口和加固所有权；不要冒称 unified physical KV 已由这个池实现。

### RMC47 — codec 域、布局指纹与固定 LE 描述符

**证据：** src/llama-xkv-codec.h:1–185; .cpp:572–702,736–805。

**实际行为：** 逻辑/填充 shape、role/orientation、stored/decoded domain、seed/table/format revision 分开；固定 100 bytes LE，拒绝尾字节与枚举高位污染，临时完整验证后才覆盖输出。

**A/B/C 边界：** serializer 和独立 domain record 可 B；GGML type/traits 与 Turbo helper 由 C 适配当前公开 A。

**风险与证据限制：** deserialize 对 size 正确而 data=null 未拒绝；混合 Turbo/non-Turbo 仅 canonical decode 可兼容，不能 direct rotated GEMM。

**保留建议：** 保留这一小能力；不得复制旧 GGML 类型号或模型格式 heuristic。

### RMC48 — factor 小数学、矩阵视图与 arena carver

**证据：** src/llama-xkv-factor.h:1–326; .cpp:1–102,360–444,475–501; ggml-xkv-factor.cpp:26–45; xkv_factorize.comp:1–50,835–919; CUDA xkv-factorize.cu:1–50,920–1009。

**实际行为：** 借用/拥有矩阵区分、checked 元素数、watermark 不变的对齐 bump carver；counter-Rademacher 无需常驻 Omega；大 kernel 采用 QR/Jacobi/codec stages。

**A/B/C 边界：** carver、counter RNG、纯矩阵可 B；codec/graph/backend factorizer 仍是新 A op。

**风险与证据限制：** 矩阵外部指针及 SIZE_MAX 边界靠 caller；carver reset/真实 span 必须有效；CPU wrapper 的 vector 分配可抛异常，不能以 never-throws 注释担保 C ABI；CUDA GEMM 用 cuBLAS，注释 same serial order 不是舍入证明；大分解中段没有全文审查。

**保留建议：** 保留小 allocator/纯参考；不把 1856/1009 行模块整体认证。

### RMC49 — canonical transform 与 selected-row reconstruct

**证据：** src/llama-xkv-canonical.h:1–377; .cpp:1–40,833–902; ggml-xkv-reconstruct.cpp:748–817; xkv_canonicalize.comp:1–223; xkv_reconstruct.comp:1–370; CUDA corresponding small .cu 全文。

**实际行为：** 明确 Turbo stored→canonical、可选 Hadamard、逆 RoPE、selected factor row 重建；exact bytes 先检查 block 整除再调 GGML geometry；fragment refine 做有限展开。

**A/B/C 边界：** 变换数学/checked bytes 可 B+C；实际 KV cell snapshot 和新 reconstruct op 需要 A seam。

**风险与证据限制：** GPU reconstruct 仅检查 ls<0，没有上界就索引 layer_meta；Vulkan canonical 64-bit pos 的 low word 按 signed int 转换丢 2^32；输出 finite 检查在写后；refine 先截断再去重且 u32 乘法未完整 checked。

**保留建议：** 保留独立数学与错误用例，不能原样作为安全 device reader。

### RMC50 — XKV hot/cold tile attention 与 CPU wrapper

**证据：** ggml-xkv-attention.cpp:1–481; ggml-cpu-xkv-attention.cpp:1–83; xkv_attention.comp:1–357; CUDA xkv-attention.cu:1–50,444–533。

**实际行为：** 合并 hot/cold entry，显式 tile carry、first/final、每 query softmax 与 sink 单次加入；后续 tile 保留先前 error。

**A/B/C 边界：** 依赖新 GGML op 和 private graph 构造；online softmax 参考可拆 B。

**风险与证据限制：** 非有限 score 被当 mask，并非所有非有限输入一律失败；CUDA strides 转 u32 需要 supports 的尺寸条件。

**保留建议：** 保留数学合同和 sticky status 行为；完整执行暂不迁。

### RMC51 — landmark 合法性、稳定 top-k 与 bounded rows

**证据：** ggml-vulkan-landmark.h:1–427; ggml-vulkan-landmark.cpp:1–45,1353–1422; xkv_landmark_{score,select,rows,merge}.comp 全文; src/llama-xkv-landmark.cpp:1–40,3246–3315; CUDA xkv-landmark.cu:936–1025。

**实际行为：** 先 legality 再 score，score-desc/id-asc 严格次序；arena filter 后 cap、按(row,slot)去重、compact ordinal window 和 valid=0 padding；固定 tile 避免 nq*history 临时矩阵。

**A/B/C 边界：** bounded selection/record ordering 可 B；真实 phase-aware scoring 与 factor decode 依赖新 op。

**风险与证据限制：** score 的 local failed 只 lane0 赋值却用于 barrier 外 break，可能非一致控制流；merge 各 query 写 status=0 会覆盖别的失败，CUDA 同样；dup 静默跳过与 collision-fail 注释不符。

**保留建议：** 保留 bounded 算法设计与明确缺陷；未通过真实 GPU 失败回归前不迁。

### RMC52 — immutable backend store 与实际 byte 计量

**证据：** src/llama-xkv-backend.h:1–85,759–833; .cpp:1–40,3434–3503; cache.h:1–85,1279–1353; cache.cpp:1–40,5485–5554; runtime.cpp:4767–4823。

**实际行为：** 逻辑 bytes、padded bytes、实际 unique allocation bytes 分开；清空先检 epoch/reserve，再 retire 不可变段；report 按 allocation ID 去重。

**A/B/C 边界：** immutable ownership/元数据可 B；实际 GGML buffer、cells、内容/绑定 epoch 为私有 A 体系。

**风险与证据限制：** canonical device reader 尾部初始化 status 却不读取就返回 true；计量 sum 有 unchecked overflow；预检和真实 native failure 不能混为事务原子性。

**保留建议：** 保留正确的计量口径和 preflight 原则，完整 store 不迁。

### RMC53 — reader pin、B tile 缓存与同形 content refresh

**证据：** src/llama-xkv-reader.h:1–70,735–799; .cpp:1–40,3420–3489; graph-ref.h:1–85,700–774; .cpp:1–40,5116–5185。

**实际行为：** immutable reader pin 与 owned hot rows；B tile cache key 排除 hot content/binding epoch，仅按 segment/codec/device retire，防止热提交导致 steady miss；topology 和内容刷新分开。

**A/B/C 边界：** cache identity 数据可 B，但真实 graph snapshot/新 op 与原始 tensor 指针绑定在 A。

**风险与证据限制：** refresh 在预分配可能失败前已更新 stamp/hot_data 等，并非失败输出不变；raw capture 需要稳定地址；大型中段没有全文读。

**保留建议：** 保留这些小 cache/owner 合同，并先修 fail 后状态才能独立复用。

### RMC54 — state import plan 与 transaction 协调

**证据：** src/llama-xkv-state.h:1–70,555–619; .cpp:1–40,2775–2844; transaction.h:1–424; .cpp:1–40,1303–1372。

**实际行为：** state frame 有显式容量/校验和/身份，高水位不倒退；先构造临时 import plan，最后调用唯一 authoritative commit；transaction/maintenance 生命周期分开。

**A/B/C 边界：** 可以提炼 B 状态转移计划，由 C 执行 A 动作；当前 callback+private cache header 并非已独立。

**风险与证据限制：** build_import_plan 自身拿 image fingerprints 验 image，调用方仍须比较 live 身份；guard release 忽略 end 失败，需协调器保持可恢复状态；本轮未全文读大 cpp。

**保留建议：** 保留无重编码恢复和高水位合同；不引入隐藏回调依赖。

### RMC55 — XKV/Tri pressure 单位与 commit 后失效

**证据：** src/llama-xkv-tri.h:1–70,494–558; .cpp:1–40,1913–1982; runtime.h:1–70,734–798。

**实际行为：** hot deficit 只由热槽减少满足，store deficit 按 factored bytes，recurrent 压力独立；mutation 成功后才使相关 landmark 失效。

**A/B/C 边界：** 单位和计划可 B，实际删除语义仍由 A cells/store 执行。

**风险与证据限制：** tri_round64(n+63) 可在极限回绕，不如已识别的 checked carver；preflight 不能替代 commit 时的 stamp 校验。

**保留建议：** 保留单位分离与 commit 顺序，小函数另行加固。

### RMC56 — native seal 的 persistent/最大 transient 分离

**证据：** src/llama-xkv-seal.h:1–234; .cpp:1–40,1785–1854; ggml-xkv-landmark-build.cpp:517–586; private Vulkan .inl 全文。

**实际行为：** 最终 compact 输出长期拥有；按最大 group 图复用 transient；状态与 telemetry 成功后零 copy adoption；landmark 每 row 自己的相位再平均。

**A/B/C 边界：** 生命周期设计可用于 C 自己拥有的 GGML graph，但不能在无 seam 时接管 A 私有 model KV。

**风险与证据限制：** 旧 CPU landmark oracle 把 nonfinite error-bound 换成 1e-4，失去保守上界；CUDA build 超 UINT32_MAX grid 时跳过却继续 finalize，需先保证 caps。

**保留建议：** 保留真实内存计量与所有权思想，完整 seal 暂归档。

### RMC57 — TP5 split planner 与 collective 边界

**证据：** src/llama-tp5-plan.h:1–224; .cpp:1–50,1084–1173; ggml-vulkan-collective.hpp:1–65,683–747; .cpp:1–50,11605–11694; ggml-vulkan-relay.h:1–29。

**实际行为：** 按 quant block 对齐 head 分片，验证全局/local Q→KV 对应；collective 明确 native drain、withheld timeout 和 toy relay timing 非模型 tok/s。

**A/B/C 边界：** 纯分片算法可 B，现 source 仍依赖 private arch/hparams/meta split 与 Vulkan/DRM；不是 GPU-free 就等于 A-free。

**风险与证据限制：** planner 末尾假定 inner head-map 长度；LCM 算术与 ranks 前提靠 caller；11694 行 transport 仅接口/尾部检查，绝无全实现正确性证明。

**保留建议：** 保留算法/验证合同，私有运输层明确归档。

### RMC58 — 现有文件的小补丁逐 hunk 分解

**证据：** src/llama.cpp 七 hunks; src/llama-quant.cpp 三 hunks; src/CMakeLists.txt 三 hunks; ext/DSA/MSA 与根文档全部 hunks。

**实际行为：** llama.cpp 是 TP5 opt-in/device gate、lazy mode、来源路径去重与调整准备时机；quant 三处都是 PQ2_0 整合；DSA/MSA 只声明 get_kv_capacity、get_kv_used、get_kv_seq_used，分别查询物理容量、已用槽和序列占用。

**A/B/C 边界：** 这些属于各自明确 feature，不可整文件 cherry-pick；CMake 把全部私有模块和 cudart 加进 A，不是独立构建。

**风险与证据限制：** 来源路径去重不是文件内容身份；新增 ABI 与旧 enum 编号不能混进 clean；旧 AGENTS/CHANGELOG 大段替换上游内容。

**保留建议：** 按小行为保留目录；上游根文档不照搬历史任务书，新源码仅按已登记 seam 修改。

### RMC59 — 历史性能、质量与方案文档

**证据：** MMVQ+MTP.md、PERFORMANCE.md、PREFILL.md、RERoT.md、TP5.md、XKV-SR.md 各:1–65。

**实际行为：** 文档真实包含阶段、artifact、模型/设备条件与后续更正；已撤回若干 universal lower-bound 或 LateBind 命中归因。PREFILL/RERoT 记载历史上游 total-kv 和已有 phase selector。

**A/B/C 边界：** 可作为历史索引，不是当前 clean 的执行说明或验证证据。

**风险与证据限制：** 本轮只读开头 65 行；不能声称完整研究文本核对。必须以锁定 A 源码与新测试分辨上游已有/新增。

**保留建议：** 保留必要的带版本历史索引，避免把 20k 行研究账带回主产品文档。

### RMC60 — 配置草案、必要成本与设备历史数据

**证据：** schema.toml:1–46; necessity-current.json:1–88; tp5-manifest.json:1–45,8674–8713; Vulkan profile:1–60,11000–11039; 维修建议刷BIOS.md:1–65,211–240。

**实际行为：** TOML 是 strategy/DAG scratch schema；necessity JSON 明示许多全设计空间下界尚未证明；manifest 固定旧模型与 unsupported 列表；设备导出和维修文档是历史快照。

**A/B/C 边界：** schema/benchmark 口径可用于独立工具，设备状态数据不属于 B/C runtime。

**风险与证据限制：** DAG 分支没有 required payload，也不验证唯一 id/环；manifest actual_model_validated 字段不证明当前 binary；硬件操作文字不构成本轮授权或需要执行的动作。

**保留建议：** 归档历史与未证明结论；不携带本机 profile/维修说明作为通用 fork 功能。

## 逐文件内容范围

以下行号全部是旧 fork diff 的 new-side 行号。F=新文件全文；H=所有修改 hunk；S=仅列出的实际范围。S 之外内容未逐行审查；H 未改区域不计入。路径和 finding ID 的详细行为/依赖/风险亦完整保存在 JSON。

| 路径 | 模式 | 实际范围 | 发现 |
|---|---|---|---|
| .gitignore | H | 3–9, 154–164 | RMC58 |
| AGENTS.md | H | 1–121 | RMC58 |
| CHANGELOG.md | H | 10–502 | RMC58 |
| MMVQ+MTP.md | S | 1–65 | RMC59 |
| PERFORMANCE.md | S | 1–65 | RMC59 |
| PREFILL.md | S | 1–65 | RMC59 |
| README.md | H | 1–7 | RMC58 |
| RERoT.md | S | 1–65 | RMC59 |
| TP5.md | S | 1–65 | RMC59 |
| VP_VULKANINFO_AMD_Radeon_RX_6800_(RADV_NAVI21)_26_0_8.json | S | 1–60, 11000–11039 | RMC60 |
| XKV-SR.md | S | 1–65 | RMC59 |
| ggml/include/ggml-device-copy.h | F | 1–75 | RMC04 |
| ggml/include/ggml-flashprefill.h | F | 1–417 | RMC26, RMC25 |
| ggml/include/ggml-predefined.h | F | 1–175 | RMC02, RMC03 |
| ggml/include/ggml-tp5-profile.h | F | 1–99 | RMC11 |
| ggml/include/ggml-vulkan-landmark.h | F | 1–427 | RMC51 |
| ggml/include/ggml-xkv-factor.h | F | 1–180 | RMC48 |
| ggml/include/ggml-xkv-landmark-build.h | F | 1–139 | RMC56 |
| ggml/include/ggml-xkv.h | F | 1–408 | RMC49, RMC50 |
| ggml/src/ggml-cpu/ggml-cpu-rerot.cpp | F | 1–460 | RMC41 |
| ggml/src/ggml-cpu/ggml-cpu-xkv-attention.cpp | F | 1–83 | RMC50 |
| ggml/src/ggml-cpu/ggml-cpu-xkv-factor.cpp | F | 1–118 | RMC48 |
| ggml/src/ggml-cpu/ggml-cpu-xkv-factor.h | F | 1–21 | RMC48 |
| ggml/src/ggml-cpu/ggml-cpu-xkv-landmark-build.cpp | F | 1–79 | RMC56 |
| ggml/src/ggml-cpu/ggml-cpu-xkv-landmark-build.h | F | 1–20 | RMC56 |
| ggml/src/ggml-cpu/ggml-cpu-xkv-landmark.cpp | F | 1–164 | RMC51 |
| ggml/src/ggml-cpu/ggml-cpu-xkv-landmark.h | F | 1–21 | RMC51 |
| ggml/src/ggml-cpu/ggml-cpu-xkv.cpp | F | 1–79 | RMC49 |
| ggml/src/ggml-cuda/fattn-rerot.cu | F | 1–390 | RMC42 |
| ggml/src/ggml-cuda/fattn-rerot.cuh | F | 1–7 | RMC42 |
| ggml/src/ggml-cuda/flashprefill.cu | S | 1–50, 1433–1522 | RMC27 |
| ggml/src/ggml-cuda/flashprefill.cuh | F | 1–11 | RMC27 |
| ggml/src/ggml-cuda/triattention-score.cu | S | 1–50, 473–562 | RMC45 |
| ggml/src/ggml-cuda/triattention-score.cuh | F | 1–19 | RMC45 |
| ggml/src/ggml-cuda/xkv-attention.cu | S | 1–50, 444–533 | RMC50 |
| ggml/src/ggml-cuda/xkv-attention.cuh | F | 1–6 | RMC50 |
| ggml/src/ggml-cuda/xkv-canonicalize.cu | F | 1–300 | RMC49 |
| ggml/src/ggml-cuda/xkv-factorize.cu | S | 1–50, 920–1009 | RMC48, RMC56 |
| ggml/src/ggml-cuda/xkv-factorize.cuh | F | 1–12 | RMC48, RMC56 |
| ggml/src/ggml-cuda/xkv-landmark-build.cu | S | 1–50, 562–651 | RMC56 |
| ggml/src/ggml-cuda/xkv-landmark.cu | S | 1–50, 936–1025 | RMC51 |
| ggml/src/ggml-cuda/xkv-landmark.cuh | F | 1–12 | RMC51 |
| ggml/src/ggml-cuda/xkv-reconstruct.cu | F | 1–313 | RMC49 |
| ggml/src/ggml-cuda/xkv-reconstruct.cuh | F | 1–6 | RMC49 |
| ggml/src/ggml-device-copy.cpp | F | 1–88 | RMC04 |
| ggml/src/ggml-flashprefill.cpp | S | 1382–1776 | RMC26, RMC25 |
| ggml/src/ggml-predefined.cpp | F | 1–323 | RMC02, RMC03 |
| ggml/src/ggml-tp5-profile.cpp | F | 1–196 | RMC11 |
| ggml/src/ggml-vulkan-landmark.cpp | S | 1–45, 1353–1422 | RMC51 |
| ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp | S | 1–50, 11605–11694 | RMC57, RMC38 |
| ggml/src/ggml-vulkan/ggml-vulkan-collective.hpp | S | 1–65, 683–747 | RMC57, RMC38 |
| ggml/src/ggml-vulkan/ggml-vulkan-internal.h | F | 1–223 | RMC13 |
| ggml/src/ggml-vulkan/ggml-vulkan-relay.h | F | 1–29 | RMC57 |
| ggml/src/ggml-vulkan/ggml-vulkan-scratch.hpp | F | 1–17 | RMC01 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-command.h | F | 1–280 | RMC13 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-coverage.h | F | 1–202 | RMC14 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-dependency-ledger.h | F | 1–13 | RMC15 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-liveness.hpp | F | 1–91 | RMC12 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-q-geometry.h | F | 1–16 | RMC16 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-qcheck.h | F | 1–45 | RMC17 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-rows.h | F | 1–248 | RMC18 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-sparse-timing.h | F | 1–41 | RMC19 |
| ggml/src/ggml-vulkan/ggml-vulkan-tp5-submit-pool.h | F | 1–70 | RMC20 |
| ggml/src/ggml-vulkan/ggml-vulkan-xkv-factor.inl | F | 1–201 | RMC48, RMC56 |
| ggml/src/ggml-vulkan/ggml-vulkan-xkv-landmark-build.inl | F | 1–96 | RMC56 |
| ggml/src/ggml-vulkan/vulkan-shaders/argmax_stage1.comp | F | 1–96 | RMC29 |
| ggml/src/ggml-vulkan/vulkan-shaders/argmax_stage2.comp | F | 1–123 | RMC29 |
| ggml/src/ggml-vulkan/vulkan-shaders/dequant_pq2_0.comp | F | 1–30 | RMC58 |
| ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_rerot.comp | S | 1–50, 646–730 | RMC42 |
| ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.comp | F | 1–23 | RMC28 |
| ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.glsl | F | 1–111 | RMC28 |
| ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_turbo_cm2.glsl | F | 1–46 | RMC31 |
| ggml/src/ggml-vulkan/vulkan-shaders/flashprefill-interface.h | F | 1–405 | RMC26 |
| ggml/src/ggml-vulkan/vulkan-shaders/flashprefill_attn.comp | S | 1–50, 488–572 | RMC27 |
| ggml/src/ggml-vulkan/vulkan-shaders/flashprefill_merge.comp | F | 1–92 | RMC27, RMC25 |
| ggml/src/ggml-vulkan/vulkan-shaders/flashprefill_pool.comp | F | 1–249 | RMC27 |
| ggml/src/ggml-vulkan/vulkan-shaders/flashprefill_select.comp | S | 1–50, 597–681 | RMC27 |
| ggml/src/ggml-vulkan/vulkan-shaders/l2_norm_pair.comp | F | 1–83 | RMC30 |
| ggml/src/ggml-vulkan/vulkan-shaders/memmove_bytes.comp | F | 1–32 | RMC05 |
| ggml/src/ggml-vulkan/vulkan-shaders/ple_router_gate.comp | F | 1–83 | RMC30 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_attention_mmvq.glsl | F | 1–245 | RMC32 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_attention_prep.comp | F | 1–255 | RMC32 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_attention_projections.comp | S | 1–50, 749–833 | RMC32 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_gdn_multistep_delta.comp | F | 1–204 | RMC34 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_gdn_multistep_norm.comp | F | 1–92 | RMC34 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_gdn_multistep_prep.comp | F | 1–280 | RMC34 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_gdn_segment_delta.comp | F | 1–180 | RMC33 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_gdn_segment_norm.comp | F | 1–65 | RMC33 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_gdn_segment_prep.comp | F | 1–226 | RMC33 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_hc_segment_norm.comp | F | 1–457 | RMC35 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_hc_up_fold.comp | F | 1–44 | RMC35 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_moe_down_k128.comp | F | 1–353 | RMC36 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_moe_projections.comp | S | 1–50, 480–564 | RMC36 |
| ggml/src/ggml-vulkan/vulkan-shaders/qwen4_router.comp | F | 1–184 | RMC37 |
| ggml/src/ggml-vulkan/vulkan-shaders/rerot_q_prep.comp | F | 1–151 | RMC42 |
| ggml/src/ggml-vulkan/vulkan-shaders/rerot_shared_kv_2reader.comp | F | 1–118 | RMC42 |
| ggml/src/ggml-vulkan/vulkan-shaders/rerot_span_expand.comp | F | 1–66 | RMC42, RMC43 |
| ggml/src/ggml-vulkan/vulkan-shaders/sanitize_nonfinite.comp | F | 1–24 | RMC06 |
| ggml/src/ggml-vulkan/vulkan-shaders/topk_moe_staged.comp | F | 1–257 | RMC37 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_add_rows.comp | F | 1–57 | RMC38, RMC18 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_bda_push_f16.comp | F | 1–138 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_copy_u128.comp | F | 1–49 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_gather_p1.comp | F | 1–81 | RMC38, RMC18 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_gpuflag.comp | F | 1–128 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_latebind.comp | S | 1–50, 875–959 | RMC40 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_norm_shared.comp | F | 1–117 | RMC39 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_publish.comp | F | 1–108 | RMC40 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_resume.comp | F | 1–458 | RMC40 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_subgroup.glsl | F | 1–24 | RMC39 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_p2p_push_f16.comp | F | 1–37 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_pack_f16.comp | F | 1–27 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_pack_f16_vec.comp | F | 1–29 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_predefined_add_rows.comp | F | 1–27 | RMC38, RMC03 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_relay.comp | F | 1–144 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_relay_copy_f32.comp | F | 1–90 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_sum_f16.comp | F | 1–82 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/tp5_sum_f32.comp | F | 1–80 | RMC38 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_attention.comp | F | 1–357 | RMC50 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_canonicalize.comp | F | 1–223 | RMC49 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_factorize.comp | S | 1–50, 835–919 | RMC48 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_landmark_build.comp | S | 1–50, 453–537 | RMC56 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_landmark_merge.comp | F | 1–135 | RMC51 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_landmark_rows.comp | F | 1–340 | RMC51 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_landmark_score.comp | F | 1–302 | RMC51 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_landmark_select.comp | F | 1–183 | RMC51 |
| ggml/src/ggml-vulkan/vulkan-shaders/xkv_reconstruct.comp | F | 1–370 | RMC49 |
| ggml/src/ggml-xkv-attention.cpp | F | 1–481 | RMC50 |
| ggml/src/ggml-xkv-factor.cpp | S | 1–45, 827–896 | RMC48 |
| ggml/src/ggml-xkv-landmark-build.cpp | S | 1–45, 517–586 | RMC56 |
| ggml/src/ggml-xkv-reconstruct.cpp | S | 1–45, 748–817 | RMC49 |
| necessity-current.json | F | 1–88 | RMC60 |
| schema.toml | F | 1–46 | RMC60 |
| src/CMakeLists.txt | H | 10–37, 54–81, 93–102 | RMC58 |
| src/llama-ext.h | H | 8–14, 161–227, 235–304 | RMC58, RMC09, RMC10 |
| src/llama-flashprefill-fixture.h | F | 1–114 | RMC22 |
| src/llama-flashprefill-layout.h | S | 1–65, 439–503 | RMC22 |
| src/llama-flashprefill-metrics.cpp | S | 1–105, 650–726 | RMC24 |
| src/llama-flashprefill-metrics.h | F | 1–347 | RMC24 |
| src/llama-flashprefill-pack.h | F | 1–63 | RMC22 |
| src/llama-flashprefill-state.cpp | F | 1–269 | RMC23 |
| src/llama-flashprefill-state.h | F | 1–192 | RMC23 |
| src/llama-flashprefill.cpp | F | 1–582 | RMC21 |
| src/llama-flashprefill.h | F | 1–139 | RMC21 |
| src/llama-kv-cache-dsa.h | H | 47–56 | RMC58 |
| src/llama-kv-cache-msa.h | H | 44–53 | RMC58 |
| src/llama-kv-transform.h | F | 1–93 | RMC07 |
| src/llama-lazy-reader.h | F | 1–147 | RMC08 |
| src/llama-predefined-hidden.cpp | F | 1–389 | RMC09 |
| src/llama-predefined-hidden.h | F | 1–101 | RMC09 |
| src/llama-predefined-session.h | F | 1–11 | RMC10 |
| src/llama-predefined.cpp | F | 1–123 | RMC10, RMC02 |
| src/llama-quant.cpp | H | 442–448, 551–557, 917–923 | RMC58 |
| src/llama-rerot-math.cpp | S | 1–147, 616–751, 910–1007 | RMC41 |
| src/llama-rerot-math.h | F | 1–493 | RMC41 |
| src/llama-rerot-profile.cpp | S | 1–35, 73–112, 551–578 | RMC44 |
| src/llama-rerot-profile.h | F | 1–278 | RMC44 |
| src/llama-rerot.cpp | S | 1–50, 3717–3806 | RMC43 |
| src/llama-rerot.h | S | 1–65, 908–972 | RMC43 |
| src/llama-tp5-plan.cpp | S | 1–50, 1084–1173 | RMC57 |
| src/llama-tp5-plan.h | F | 1–224 | RMC57 |
| src/llama-triattention.cpp | S | 1–35, 1644–1779, 1843–1902 | RMC45 |
| src/llama-triattention.h | F | 1–492 | RMC45 |
| src/llama-xkv-backend.cpp | S | 1–40, 3434–3503 | RMC52 |
| src/llama-xkv-backend.h | S | 1–85, 759–833 | RMC52 |
| src/llama-xkv-cache.cpp | S | 1–40, 5485–5554 | RMC52 |
| src/llama-xkv-cache.h | S | 1–85, 1279–1353 | RMC52 |
| src/llama-xkv-canonical.cpp | S | 1–40, 833–902 | RMC49 |
| src/llama-xkv-canonical.h | F | 1–377 | RMC49 |
| src/llama-xkv-codec.cpp | S | 1–40, 572–702, 736–805 | RMC47 |
| src/llama-xkv-codec.h | F | 1–185 | RMC47 |
| src/llama-xkv-factor.cpp | S | 1–102, 360–444, 475–501 | RMC48 |
| src/llama-xkv-factor.h | F | 1–326 | RMC48 |
| src/llama-xkv-graph-ref.cpp | S | 1–40, 5116–5185 | RMC53 |
| src/llama-xkv-graph-ref.h | S | 1–85, 700–774 | RMC53 |
| src/llama-xkv-hot.cpp | F | 1–770 | RMC46 |
| src/llama-xkv-hot.h | F | 1–268 | RMC46 |
| src/llama-xkv-landmark.cpp | S | 1–40, 3246–3315 | RMC51 |
| src/llama-xkv-landmark.h | S | 1–85, 562–636 | RMC51 |
| src/llama-xkv-reader.cpp | S | 1–40, 3420–3489 | RMC53 |
| src/llama-xkv-reader.h | S | 1–70, 735–799 | RMC53 |
| src/llama-xkv-runtime.cpp | S | 1–40, 4756–4825 | RMC52, RMC55 |
| src/llama-xkv-runtime.h | S | 1–70, 734–798 | RMC52, RMC55 |
| src/llama-xkv-seal.cpp | S | 1–40, 1785–1854 | RMC56 |
| src/llama-xkv-seal.h | F | 1–234 | RMC56 |
| src/llama-xkv-state.cpp | S | 1–40, 2775–2844 | RMC54 |
| src/llama-xkv-state.h | S | 1–70, 555–619 | RMC54 |
| src/llama-xkv-transaction.cpp | S | 1–40, 1303–1372 | RMC54 |
| src/llama-xkv-transaction.h | F | 1–424 | RMC54 |
| src/llama-xkv-tri.cpp | S | 1–40, 1913–1982 | RMC55 |
| src/llama-xkv-tri.h | S | 1–70, 494–558 | RMC55 |
| src/llama.cpp | H | 1–5, 47–62, 160–219, 233–241, 277–285, 381–406, 430–440 | RMC58, RMC08, RMC57 |
| tp5-manifest.json | S | 1–45, 8674–8713 | RMC60 |
| 维修建议刷BIOS.md | S | 1–65, 211–240 | RMC60 |

## 与其他审计的衔接

- 先前 memory 报告覆盖 KV/cells/hybrid/allocator 修改；backend 报告覆盖已修改 GGML、Vulkan、格式等。这里补的是未传输的新增/遗漏文件内容，不覆盖或改写它们的已验证结果。
- test-backend-memmove.cpp 的真实 buffer/无效后段整批拒绝、test-vulkan-scratch.cpp 的 roundup overflow、test-tp5-rank-local-readback.cpp 的 pending 双读/fallback/free retire，均由兄弟代理完成内容审查。本报告仅关联语义，不把这些路径重复计入 193。
- 主任务已经运行的 CPU allocator、C phase parity、真实 server smoke、WHT before/after 等属于主任务证据；没有用于冒充这里 59 个大文件的完整资格。

## 尚未证明的部分

59 个 S 文件中段/其余内容的全语义、私有模块全部路径可达性、当前模型质量和多卡净收益、所有错误注入的 native 退役与设备内存可见性，均未由本报告证明。机器账本的 unread_ranges 给出逐路径精确未读补集。任何后续选中的候选应围绕其具体剩余风险补最小必要验证，而不是把整套旧实验重新搬回 clean 分支。
