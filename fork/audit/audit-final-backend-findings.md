# 最后 14 个 backend / graph 路径的内容补审

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

## 范围、证据和阅读深度

本报告仅补审最终总账中 highest_recorded_depth 为 symbol_or_boundary_screen_only，且路径以 ggml/ 或 src/llama-graph.h 开头的 **14 个路径**。没有把范围扩大到其他源码、没有运行任何测试、没有访问远程环境，也没有移植代码。

实际阅读为 **13 个路径的全部已给变更 hunks，加上 ggml-backend-meta.cpp 的 33/74 个 hunks**，合计 **126/167 hunks**。Meta 剩余 **41 个 hunks** 的准确旧、新行号保存在 coverage 的 unread_ranges 中。这里的“全部”仅指相对基线的 diff 内容，**不表示读过文件中未出现在 diff 的原有正文**。特别是 graph header 的声明和注释，不等于其 out-of-line 实现已经验证。

- 基线 A：cda0f9256665845cd9d6f3a0176880883f653a3d。
- 旧 fork：2777543b2（沿用原始审计输入固定 head）。
- 输入：audit-backend.diff，2,015,787 bytes，SHA256 **2f5ce648a9b8ffa01ecb15e97f72a7397c7b55b5f28ed0f21eed01cea58993b7**。
- 行号：以下引用均为旧 fork diff 的新侧行号；不是当前干净分支的行号。
- 逐 hunk 阅读日志：audit-final-backend-read-log.json。两次长输出的中部截断已经用窄范围重读闭合；没有把被截断部分登记为已读。
- 本轮不产生“全部旧功能可以丢弃”的结论。未移植候选与明确未读区域继续保留在账本。

## 最值得单独登记的小行为

| ID | 行为及依据 | 保留边界与当前判断 |
|---|---|---|
| FBE01 | 普通 CPU SET_ROWS / SOFT_MAX / ROPE / ROPE_BACK 的工作区逐线程 padding；ggml-cpu.c:3188–3201 | 与 TP5/XKV 算法无关。应单独核对 A 中消费者实际 stride 后再做最小 A 修复；本轮没有证明旧总工作区必然不足。 |
| FBE06 | 显式 group=128、无共享 group 设置依赖的 Turbo row codec，且允许编码字节非对齐；ggml-turbo-quant.c:390–594 | 是可单独评估的 codec seam。不能把全部旧量化入口称为已可重入。 |
| FBE11 | meta 非零全局行偏移转换成每 shard 的局部偏移；meta.cpp:3317–3319、3597–3599 | 直接涉及普通已存在的 axis-0/1 异步读写。最小 A 候选，不需要私有量化类型或完整 TP5。 |
| FBE12 | 每次新读取、fallback、同步与销毁之前完成前次 pinned readback 的 host assembly；meta.cpp:2954–2989、3368–3514、3616–3626 | 生命周期合同应独立保留；实际 fast path 还需要设备 host-buffer 与 flush 能力。 |
| FBE15 | graph view 缓存键除 UID 外比较节点数量及 nodes 基址；meta.cpp:3675–3679 | 不能只因父 graph UID 相同就复用另一 slice。最小 A 候选。 |
| FBE16 | 外部 CPU view 快捷路径只允许 empty op 且不允许 meta buffer；meta.cpp:3779–3785 | 防止真实 CPY 或 host-backed meta 的虚拟地址被当成真实 host tensor。最小 A 候选。 |
| FBE14 | 对所有 shard 预检后才开始设备同 rank 拷贝；meta.cpp:2666–2696、3193–3238 | 可独立保留预检原则；返回 false 不等于所有执行期错误都有回滚。 |
| FBE20 | graph reuse 区分容量、有效行数、输出与输入来源；llama-graph.h:1410–1605 | 对固定容量执行有价值，但不是只改比较条件即可安全获得的普通 graph reuse 优化。 |

前六项尤其不能因为所在文件主要服务私有扩展就整体归档。这里只给出保留候选，是否增加当前 clean fork 补丁由根任务结合真实回归与维护成本决定。

## CPU 与量化

### FBE01 — 普通工作区的 cache-line 间距

证据：ggml/src/ggml-cpu/ggml-cpu.c:3188–3201，ggml_graph_plan 的既有操作分支。

F16 源转其他格式的 SET_ROWS 从每线程 `sizeof(float) * ne0` 改为 `sizeof(float) * ne0 + CACHE_LINE_SIZE`。SOFT_MAX、ROPE、ROPE_BACK 从单纯的行宽乘线程数改为每线程行宽加 CACHE_LINE_SIZE_F32，再在总量加一个 CACHE_LINE_SIZE。这些分支本来就存在，不是新私有 op 的 scratch。

真正应保留的是规划器与消费者逐线程实际地址步长一致。单看这一组 diff 不能排除外围已有额外总 padding，不能直接宣称它修复所有越界；本轮没有越出清单重读 ops.cpp 的完整消费者，也没有跑 ASan。最小候选应同时给非 cache-line 对齐行宽、1/多线程、不同数据格式的真实 compute 回归，而不是只测试重写的尺寸公式。B 可以独立持有 checked sizing 算法；实际 A 分配器采用哪个大小仍是一个明确的小 A seam。

### FBE02 — 空 threadpool barrier 和新增私有 op 调度

证据：ggml-cpu.c:663–671、2222–2271、2333–2350、2683–2714、3267–3401。

新增 `if (tp == NULL) return;` 允许某些单线程参考内核直接调用 barrier。它不是“任何无 threadpool 多线程调用都安全”的保证；外部调用者必须保证确实没有其他 worker 需要同步。

同一文件还注册了 RERoT、FlashPrefill 和 XKV 操作及任务数。XKV reconstruct、attention 等走单任务，FlashPrefill 使用多线程。部分 XKV workspace 估算失败后退回固定 4096/2048 float 的大小；该 fallback 本身不是内核任意输入都能安全执行的证据。FlashPrefill 规划器有多处 SIZE_MAX 饱和分支，但不能据此把所有前置乘法、对齐加法也称为已证明防溢出。应把通用 barrier 和规划行为独立登记，把私有 op 注册与其内核、输入验证共同归档。

### FBE03 — CPU capability 探测在分配前只读元数据

证据：ggml/src/ggml-cpu/ggml-cpu.cpp:423–586 的完整新增 ggml_backend_cpu_supports_flash_prefill，632–644 的路由。

该探测只读 type、shape、nb、op_params，不读取 tensor data，能在图尚未分配时调用。它验证版本、DK/DV 上限、头映射、config 数值及布尔模式、K/V codec 能力，并排除 TQ3_1S/TQ4_1S 权重域格式。POOL、SELECT、ATTN 各自检查相应元数据和 sink 条件；已有 extra-buffer 委托的上层顺序保留。

这个小行为是“分配前可用的能力预检”，不应只写成“新增 FP backend”。但代码并未构成内核所有前置条件的形式证明；例如若干跨输入维度组合还需与真实 kernel 一致性测试。B 可以拥有独立 POD 形状规则，C 负责能力选择；最终 ggml type/op 的支持回答属于 A 后端 seam。新增 CPU snapshot callback 为 NULL，不代表 CPU 获得了新的异步 snapshot 实现。

### FBE04 — PQ2_0 独立布局、reference codec 与 CPU dot

证据：ggml/src/ggml-common.h:99–107、194–209；ggml-quants.c:109–154、496–521、2317–2337、5753–5762；ggml-cpu/quants.c:30–39、226–280；ggml-cpu/arch/arm/quants.c:294–375；ggml-cpu.c:272–283。

这是明确独立的 group-128 PQ2_0，块为一个 half scale 加 32 个量化字节，共 34 bytes；不是修改当前 group-64 Q2_0 的布局。reference 编码以块内 amax 为 scale，round(w/scale) 后编码；解码等级为 {-1,0,1,2}*scale。对通常的有限非零输入，amax 归一化意味着 reference 编码实际主要落在前三个等级，不能仅因类型名就称为质量提升。

generic dot 每个 128 元素 PQ 块对应四个 32 元素 Q8_0 块，各自应用 Q8 scale。ARM 完整新函数把字节展开、位移、mask 后做 NEON dot，再应用相同的 scale 结构；非 NEON 调用 generic。ARM 这次增加的确实仅是 PQ2，不能登记成对普通 Q2 的新 ARM 优化；NEON 和 scalar 的浮点归约顺序也没有在本轮证明位级相同。

`quantize_pq2_0` 的非空 quant_weights 分支仍逐行调用无权重 reference，没有使用权重值，所以不是新的 importance-weighted 量化功能。dimension 的部分检查是普通 assert，不应当作所有 Release 输入都受到动态验证。保留建议是只有存在明确旧 PQ2 文件消费者时，单独做 codec / 格式转换工具或独立 B codec；不能把私有 enum 值 142 和 GGML_TYPE_COUNT=143 原样加入干净 A。

### FBE05 — architecture alias 与构建整合不是孤立可搬的优化

证据：ggml/src/ggml-cpu/arch-fallback.h 全部七个 hunks；ggml/src/ggml-cpu/CMakeLists.txt:52–63。

多个架构增加 PQ2 generic alias。同时 x86 分支删除了普通 ggml_vec_dot_q2_0_q8_0_generic 映射到公共名字的 alias；这允许架构专用实现拥有该符号，脱离对应实现单搬可能造成缺失或错误的链接分派。本轮不重新扩大到 x86 文件，其实现证据应引用此前 backend 审计，而不是假装在这里又读了一遍。

CMake 的六个新增源文件是 XKV reconstruct/attention/factor/landmark/build 和 CPU RERoT 的构建接入。它们不是一个无依赖的通用 CPU 库，也不是应一并保留的新性能功能。

### FBE06 — 显式、可重入的 Turbo row codec 与字节所有权

证据：ggml/src/ggml-turbo-quant.c 全部十个 hunks，尤其 279–364、390–594、598–664；公开合同 ggml/include/ggml.h:2829–2880。

新增 ggml_quantize_turbo_row / ggml_dequantize_turbo_row 接受 type、元素数、group 和 decode domain。它们拒绝空指针、非正/非 128 倍数元素数和 group!=128。TURBO2/3 使用显式 group helper；每个编码块先在局部正确对齐对象中产生，再 memcpy 到字节数组。解码先把可能非对齐的编码字节复制到局部块；canonical 域最后做 inverse WHT，rotated 域保留旋转值。

这是真正可单独讨论的 API 和对齐行为，当前格式即可受益，不依赖 XKV 才有意义。但原有 quantize_row_turbo2/3_ref wrapper 仍读取原来的 global group 配置，不能称“全库可重入”。新 API 也没有带 src/dst byte span 长度，调用者仍需保证缓冲区实际容量；bool 不是有限值/数值质量验证。TURBO4 旧非四位模式的兼容性还需调用者检查 fingerprint，不能假设所有编译模式都可交换同一编码。

可行边界：A 暴露一个窄且明确的 codec seam，B 使用稳定自有字节域与格式标识；或 B 完全拥有自身 codec 算法，C 做显式转换。不得在 B 头文件中偷引 A 的私有 block 类型。

### FBE07 — WHT export、兼容 wrapper 与布局 fingerprint

证据：ggml-turbo-quant.c:22–30、225–301、526–594、717–723。

WHT 实体改名为 ggml_turbo_wht_row / ggml_turbo_wht_inverse_row，保留旧 turbo_cpu_fwht wrapper，避免旧符号调用点立即失效。这里没有给任意 group size 增加完整公开验证；不能因为函数被 export 就扩大它的 group 合同，尤其不能与前轮已经实测修复的 Vulkan WHT workgroup stride 混为一项。

fingerprint 逐字节固定顺序散列类型、旋转种子/维数、sign、block 大小、centroid 位模式及 T4 模式。不支持的类型/旧 T4 模式返回 0。用途是编码兼容性拒绝，不是密码学证明、输入内容校验或质量指标。若移植 codec，应将 fingerprint 作为版本/兼容合同单独保存；不能顺便带入私有类型 renumber。

## 公共声明、CUDA 与 Vulkan 编译矩阵

### FBE08 — ggml.h 中的小合同要区别于 op 数量

证据：ggml/include/ggml.h 的全部 11 个 hunks。

除私有 PQ2 类型值与一组 XKV/FP/RERoT op 枚举外，值得独立记录的是：

- 2612–2623：普通 flash attention 的 n_kv_max 限定 setter，0 表示禁用；它是有限 mask/稀疏 KV 扫描上界的 API 候选，但必须和 backend 对该参数的消费共同验证。
- 2785–2824：GDN active_tokens 与 capacity 的显式区分；RBB 则是另一个共享 recurrent 状态语义，不能混同。
- 2540–2600：RERoT entries + query offsets 的全局 softmax 合同、Q position 的 capacity stride 和 inactive group 行为。
- 1399–1472：FlashPrefill metadata、pool、plan 与 dependency-only K/V 边；依赖边不能因不读 payload 而被优化掉。
- 3090–3098：graph UID getter/setter 只提供身份数据；不替代 view 基址与 node-count 检查。

这是“声明内容已读”，不是所有导出函数的实现全部验证。没有证据说明这些新合同均可通过现在上游稳定 API 获得。任何实际需要 A 消费的新字段、op 参数、tensor flag 都要登记 A seam。

### FBE09 — CUDA GDN 改动是完整 RBB solver，不是普通小 kernel 修复

证据：ggml/src/ggml-cuda/gated_delta_net.cu:291–530，单个 hunk 已全部阅读。

新增完整 gated_delta_net_rbb_cuda 和 launcher：一线程处理 (head,state-column)，最多 64 writers、state size 128，double packed Cholesky 解 sqrt(beta) 对称缩放系统。density 模式计算有效 writer 间 cos² 冗余，合并 shared brain，同时生成 writer hand 与当前 native readout；N=1 走单 writer 特例。launcher 要求 one-token、contiguous state/native，并排除现有 fused cache；主入口由 op_params[1] 选择该路径。

这有明确语义，但不应称为免费提高普通 GDN 性能。单线程 packed double factor 自身已有约 16 KiB 状态，还不算其他数组，GPU local-memory/占用率成本未测。边界断言、beta/shape 和矩阵正定合同必须一起成立；代码不是一般失败可返回的数值求解 API。本轮没有 CUDA 执行证据。

B 可以独立拥有 CPU 数学参考；要在 A graph 上走这个 CUDA op，仍需 type/op 参数、tensor 布局和 backend 支持 seam。只保留研究候选及其数学/边界测试，不带入普通推理默认路径。

### FBE10 — shader generator 中分离的构建行为

证据：ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp 全部 15 个 hunks。

小而独立的生成项包括 IQ2_S/IQ3_XXS grouped MoE matvec 三个归约变体（818–821、1989–1996），pair L2 / PLE router（880–881），F16 scale（1560–1561），memmove_bytes / sanitize_nonfinite（1564–1565），两阶段 argmax（1697–1698），FA output gate / split-K gate（1042–1050），有限 mask compact（1063）。这些是前轮已经审查对应 kernel 的构建连接点，不是这份 generator 自身完成了算法证明。

其他生成族仍需分别登记：FlashPrefill pool/select/attn/merge；RERoT span-expand/Q-prep/indexed attention；XKV reconstruct/factor/canonicalize/landmarks；TP5 wire/relay/latebind；Qwen HC/GDN/router/MoE。HC workgroup reassociation 和 Q8 companion 路径有明确近似合同，不能借“fused”标签称为位级等价。GDN cached/norm、BF16 HC 原始模型支持与固定四列输出 producer 是不同功能，不应当都合并成 TP5 sum。

两个构建层行为尤其可保留为原则：

1. 1082–1155 等处从 C++ 单一 capacity/epoch 常量生成 shader define，避免 host 与 unrolled loop 容量悄悄不一致；直接移植该 include 会同时引入私有 TP5 header，应重新做自己的独立生成配置。
2. 1499–1507 区分 generator 自身是否继承 C++ capability define 与设备运行期 integer-dot capability，始终生成 Qwen MMVQ 变体。它解决的是缺失编译变体，不是证明旧 glslc 均能编译整数点积扩展；运行期 gate 与构建工具能力仍要单独验证。

## Meta 后端：已读完整函数及独立小行为

本节覆盖 33 个完整给定 hunks，但没有完整阅读整个 ggml_backend_meta_graph_compute 或 split-state 推导大函数。完整新函数包含 headmap set/get、mapped helper 组、cpy_tensor、context 的 release/retire/shutdown 成员、predefined setter、device_copy_ranges、snapshot upload、try_get_tensor_async_pinned、synchronize、graph_optimize。已有函数只有 diff 提供的局部改变时，以下明确按局部行为引用。

### FBE11 — 异步 shard 的非零 offset 修正

证据：ggml-backend-meta.cpp:3240–3246、3288–3319、3501–3521、3562–3601。

旧异步 set/get 要求 offset==0，并把全局 offset 传给每个 shard。新代码移除该入口限制，普通 axis-0/1 分支将起始逻辑行 `i_start = offset / full_row_bytes` 转成 `i_start * shard_row_bytes`。数据指针依然是调用者给的局部传输起点，各 shard 按同一行的分片偏移读写。对应 synchronous 分支给定上下文已经使用局部 simple_offset；本轮看到的修复主要针对 async。

这是具体、可测试、可最小保留的普通 meta 修复。验证应使用两种不等宽 shard、非零起始行、跨多行、目的 guard 和 sync/async 对拍。只改一边会留下另一方向的错误。不需要 TP5 head map、PQ2 或 graph chain。

### FBE12 — pinned readback 的单事务退役

证据：2846–2873 的状态；2954–2989 retire_pending_readback；3002–3011 shutdown；3368–3514 的完整 fast function 与 caller；3616–3626 synchronize。

fast path 限于 contiguous F32、axis-0、单 segment、非 mapped/replica、完整行、至少两个 child。它检查每个 child 的 2D async get 与 owner-local flush callback，检查 shard 行宽之和；每 rank 分配其设备要求的 host buffer 类型并保留 slot。数据先读入 slots，统一 child 同步后才按 row stride 拼入 caller destination。

关键小行为是入口先 retire 旧事务，即使接下来是 size==0、mirrored fallback 或需要扩大 slots。synchronize 若完成退役就不重复同步；shutdown 在 child device 被销毁前完成退役、清 graph 状态、释放 slot 和 device-owned bufs。使用者仍必须使 caller destination 生存到同步边界；backend 不能替 caller 保护已销毁内存。

此原则应独立保留。相关测试的内容证据由 portable_features 代理提供：test-tp5-rank-local-readback.cpp:227–280、286–354、360–403 分别覆盖双次未同步读取、pending 后 fallback、析构退役。本代理没有重读或运行该清单外测试，不能将其算成本轮源码阅读或实际通过。

### FBE13 — snapshot 上传全 child 预检，且普通 async 不等于 snapshot

证据：3336–3366 完整 ggml_backend_meta_set_tensor_snapshot_async。

只接受 contiguous mirrored tensor 和有效 offset/size。它先 dry_run 每个 child 的 snapshot callback，全部可用后才捕获任何一份数据，避免异构 meta 上传一部分后又普通 fallback。执行阶段假定预检后 capture 不再失败，以 GGML_ASSERT 检查返回值；这依赖明确的 backend callback 合同，不提供任意资源失败下的回滚。

这是可复用的小异步所有权功能，但新增 iface 字段是 A seam。普通 set_tensor_async 仍不能让调用者立即覆写源数据；必须明确使用 capture/snapshot API。B 不应通过访问私有 iface 冒充独立实现。

### FBE14 — 同 rank copy 与 device-copy ranges 的整批预检

证据：2666–2696、3193–3238 的完整新增函数。

cpy_tensor 先确认 src/dst 都是 meta、rank 数一致、每个简单 tensor 存在且 layout 相同，再逐 rank 执行设备 copy，能力不足时调用通用 tensor_copy。ranges 版本只支持 mirrored 对应物理设备，先映射所有 ranges，所有 rank dry_run 成功后才开始记录。

这保留了“后面 rank 不支持时不要先写前面 rank”的有用行为。但 ranges 的真实 record 失败仍返回 false，此前 rank 可能已记录；不能说 false 恒表示零修改，也不能自动转另一条路径重做。独立 B 可负责纯 range 验证和计划；真实设备复制由 C 调用 A 的窄 API。后端接入不能绕过 scheduler/queue 生命周期。

### FBE15 — cache 的 graph view 身份与失效顺序

证据：2881–2898；3655–3695；4415–4473。

cached definition 属于每个 backend/session，key 除 uid 还保存 n_nodes 和 nodes 基址。共享父 UID、相同节点数但不同 view slice 不再误复用。重建前先使 predefined valid=false 并清空 CB/tensor handles，然后才重置图结构。节点基址和数量本身仍不检测“原地修改同一数组中节点内容且 UID 没变”的非法复用，不能夸成完整内容 hash。

最小可保留项是普通 cache 的 view 身份及先失效后释放。private epoch chain 需要更多 native lifetime 验证。本轮仅静态阅读；portable_features 的 test-meta-reduce-boundary.cpp:2315–2367 是外部代理提供的针对相同 parent UID 不同 view 的测试内容证据，非本轮执行结果。

### FBE16 — host view 快捷路径排除真实 copy 和 meta 虚拟地址

证据：3777–3788 局部 graph_compute 改动。

原分支仅看 view_src 为 op NONE 且 host-backed。新分支还要求当前 op 是 empty，且 view_src buffer 不是 meta；CPU 上的 s_copy_main 这类真实外部 host view 可以直接复用，CPY 等真实操作和 host-backed meta 必须得到 rank-local tensor。理由是 meta 全局 data 地址只是虚拟标记。这个范围限制是有具体地址语义的普通安全候选，不是 TP5 算法专属性能项。

### FBE17 — 初始化、通信失败和 teardown 的可观察行为

证据：2905–2952、3002–3011、3045–3102、3179–3191、4703–4719、4780–4785。

child 初始化失败会释放已有 child 并令 meta init 返回 nullptr；广告 comm_init 的 backend 必须同时提供 allreduce/free，否则在创建 communicator 前失败。native comm init 拒绝后不悄悄转 host relay；native submit/allreduce 已尝试但失败时返回 FAILED，避免不明提交状态再 fallback。普通无 native communicator 的 CPU 路径仍使用 generic fallback。

release_comm 可以调用可选 free_safe；drain 失败会保留整个 meta/child 对象，日志明确指出未销毁，而不是让驱动继续访问已释放 buffers。这是有代价的受控资源保留，不是“无泄漏 teardown”。这些合同必须按 API 能力及是否已提交分别登记，不能把 strict native failure 策略无条件改成上游普通 fallback 的新默认。

### FBE18 — 映射区间、owner 选择、懒初始化和 mirrored 缺席

证据：550–585、595–796 的完整 helper；2419–2646；2761–2784；mapped async 3248–3278、3523–3553、indexed async 3566–3587。

mapped helper 把逻辑非连续 spans 映到 rank-local 连续片段；slice 不允许把分离的本地存储伪装成一个连续 view。canonical interval 先收集所有端点、排序去重，再为每个相邻区间选择最低 rank 的完整 owner，避免重叠 replica 同时写同一 host 字节或漏掉部分重叠的尾段。这个区间算法可拆成不含 ggml 类型的 B；张量 nb、quant block、rank-local source 的连接仍属于 C/A seam。

不能把 helper 称为完整无溢出 verifier：span_start+span_len、local_len 累加和重缩放乘法没有在该函数中逐个 checked arithmetic，部分条件用断言。外部已约束的小整数域可以令它安全，但该前提需要明确。

simple_tensor lookup 会对非 weight 缺失项尝试 lazy init；op NONE 且非 view 的 weight 若未在 static container 注册则 abort，避免未加载 shadow。另一个 guard 跳过已归属其他 meta static container 的 tensor。源注释自己说明最初怀疑的权重身份分叉后来未复现，因此应登记为防御性 guard，不能宣称已经证明修复了一个真实旧 bug。

同步 mirrored 写跳过不存在的 simple tensor，读从第一个存在的 rank 取值；若所有 rank 均不存在，读分支本身没有新错误返回。这依赖非空 owner invariant，不能把 skip-null 简写成“所有缺 shard 情形都安全”。

### FBE19 — hooks 的隔离与 allocation-only 优化入口

证据：66–99、110–114、268–273、3105–3171、3628–3653、4680–4694、4725–4754。

headmap hook 以 device/session 存储并复用相应 userdata，3-bit packed head map 有 magic/count 字段；这避免全局 head mapping 污染其他模型。但 native chain callback pfn_submit_chain/pfn_get_cbs，以及旧 flush 探测仍有 process-static 状态，不能把整个 meta 后端称为已彻底消除跨实例污染。异构 registry 或并发场景仍应隔离或验证。

predefined setters 先检查所有 child entry point 再逐个设置。其无回滚行为依赖“同样的输入预检后所有 setter 不失败”，不是任意第三方 backend 的保证。

graph_optimize 新入口按 registry 去重调用可选 ggml_backend_graph_alloc_deps，只添加分配依赖，不直接调用 child graph_optimize 重排跨 rank reduction 边界。可复用的是把 allocation dependency advice 与计算图重排分开；采用该 proc 名/参数仍是明确的新 A seam。

## Graph header 中被大功能掩盖的小合同

### FBE20 — 固定容量执行的 reuse 与 active tail

证据：llama-graph.h:244–265、1410–1605、1769–1780、2228–2233。

out_ids 新增 capacity_mode；graph params 分开保存 capacity rows/outputs 和 active frame。allow_reuse 对版本、非零 capacity、active<=capacity、两边 capacity 一致进行判断。只有受限 MTP 或 target predefined 路径才允许有效 token 数、每序列 token 数与 n_outputs 变化，并相应跳过原 sampler-row 比较。ubatch.device_embd 身份加入比较，避免把不同设备输入来源当成同一图。卷积 tail helper 声明明确排除 inactive capacity suffix。

这些是具体的功能拼图。只抄“跳过 n_outputs/采样器比较”会破坏未完全降成 capacity-aware 的图，必须和所有 consumer 对 active token、output ID、conv/recurrent tail 的处理成组迁移。本轮只读了 header 中实际比较代码和声明，未新增对 out-of-line builder 的验证。

### FBE21 — XKV handle、lease 与缓存新鲜度

证据：llama-graph.h:332–435 的完整新增类声明和 inline key equality。

图输入以 shared_ptr 持有 custom-op handles；entry 分别保存 status、validity、acquire/release/fill callbacks。状态区分 ok/retry/hard error，约定 postcompute 必须读状态，不能把零输出误认成功；lease 在所有 poll 结果和析构释放，set_input 前重新 acquire。key 比较 codec/factorizer/budget/chunk/mode/token 数；不比较 n_kv 的依据是固定容量 hot view invariant。

这些所有权/错误合同可独立设计成 B 的 lease/status 辅助机制，但 actual snapshot、mctx、GGML input 都是私有依赖。类顶部“禁用 reuse”的旧注释与后面按 stamp 允许 reuse 的新合同存在差异，以实际 can_reuse 实现为准；本轮没有把注释当执行证明。

### FBE22 — RERoT staging 的容量和位置 stride

证据：llama-graph.h:473–645。

key 分 group cap、entry cap、token、backend variant、frontier mode，默认 group bucket32、entry bucket256；虚拟位置、物理 storage base、counts/visibility 被设计为每次刷新输入数据。staging vector 属于 input 实例，避免 pipeline 期间全局共享 staging 被覆盖。Q 多坐标按 capacity stride 放置，RoPE 在 WHT 前，IMRoPE 第四坐标固定 0。GPU expand 输出作为 attention 明确的依赖边。

“只初始化一次的零 tail”需要认真保留其真实消费者边界：前轮较长 live prefix 在下一轮缩短后可能成为非零 inactive suffix，不能仅凭 header 注释宣称所有尾部恒为零；有效 offsets/capacity consumer 的行为必须由实现与 poison-tail 测试证明。B 可以拥有 span layout 及 position 数学，实际 input tensors 和 kernel 仍需要 A seam。

### FBE23 — FlashPrefill 的数据/拓扑拆分、orphan 输入与混合格式

证据：llama-graph.h:665–912、927–984、1282–1290、1434–1482、1576–1601。

key 明确比较每 layer K/V type、heads、维数、各种 caps、role、reserve 标志；rows_equal 的完整 inline 实现仅比较 presence/size/role/prefill_known，seq、position、epoch 是每 submit 刷新的数据。wants_companion 声明用于使 dense→sparse 状态改变时重建，而不是固定容量下永久复用 dense graph。used_by_graph 防止向未被任何 op 消费、因此 scheduler 没有分配的 orphan metadata 上传。cur_rows 用 shared ownership 保持提交期间源数据存活。

这些是小的真实工程合同，不能都用“FP 可舍弃”遮掉。但 out-of-line refresh、cap guards 与标志设置是否完整仍需其实现证据。B 可持有纯 route/layout key，C 负责决定启用及 snapshot；A 需接入 metadata 图输入和 op 节点，不能靠 decorator 自动获得私有 allocation 生命周期。

### FBE24 — 指标语义和资源相位的边界

证据：llama-graph.h:728–799、1656–1674、1698–1704、1997–2056。

summary 区分 shared metadata 一次计数、各 layer pinned plan 之和、每 backend transient pool 的峰值，避免把所有 layer pool 全部累加；route 以可见合法 tokens 和真实 row incidences 计数，不能用物理 KV capacity 假装可见上下文。plan/error word 约定在完成边界读取，避免逐层 sync；reserve sizing 与 live graph 的 key 隔离，设计 dense 与 unsupported/corrupt 路由分别登记。

这些 header 度量字段不是驱动实际可用 VRAM，也不是释放实际 bytes 的证据。当前 clean phase seam 的真实 scheduler buffer getter 和 root 已有实测与此是不同证据。物理 KV pool 变大可能扩大实际 attention/compute span；不能据这里的逻辑 visible count 向用户承诺 scratch 上限永远由逻辑 window 决定。

## 逐路径覆盖清单

| 原文件 | 本轮内容范围 | Finding IDs |
|---|---|---|
| ggml/include/ggml.h | 全部 11 个变更 hunks，声明合同与 enum 变化 | FBE04, FBE06, FBE08 |
| ggml/src/ggml-backend-meta.cpp | 33/74 hunks；完整新增 helper/生命周期/API 函数加 graph_compute 选定分支；41 hunks 未读 | FBE11–FBE19 |
| ggml/src/ggml-common.h | 全部 2 hunks | FBE04 |
| ggml/src/ggml-cpu/CMakeLists.txt | 全部 1 hunk | FBE05 |
| ggml/src/ggml-cpu/arch-fallback.h | 全部 7 hunks | FBE05 |
| ggml/src/ggml-cpu/arch/arm/quants.c | 全部 1 hunk，完整新 PQ2 dot | FBE04 |
| ggml/src/ggml-cpu/ggml-cpu.c | 全部 9 hunks | FBE01, FBE02, FBE04 |
| ggml/src/ggml-cpu/ggml-cpu.cpp | 全部 4 hunks，完整 capability 函数 | FBE03 |
| ggml/src/ggml-cpu/quants.c | 全部 2 hunks | FBE04 |
| ggml/src/ggml-cuda/gated_delta_net.cu | 全部 1 hunk，完整 CUDA RBB solver/launcher | FBE09 |
| ggml/src/ggml-quants.c | 全部 4 hunks | FBE04 |
| ggml/src/ggml-turbo-quant.c | 全部 10 hunks | FBE06, FBE07 |
| ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp | 全部 15 hunks，生成注册行为 | FBE10 |
| src/llama-graph.h | 全部 26 hunks；声明和 inline 比较代码已读，未冒充所有实现已读 | FBE20–FBE24 |

Meta 未读的 41 个 hunks 是索引 **8–42、63–68（从 0 开始）**：包括 split-state 推导的大量 switch 分支、部分 tensor init/clone 与 mapped upload、部分 graph lowering/reduction 子图构造。它们没有在本轮被升级为内容已读；准确范围随 coverage 一起保留。报告引用的 generic 修复候选并不证明这些未读分支安全，亦不构成可整体丢弃它们的理由。
