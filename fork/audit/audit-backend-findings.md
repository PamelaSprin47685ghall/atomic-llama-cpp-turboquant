# Backend 改动审计：通用小功能、明确 A seam 与私有扩展边界

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

作者：`boundary_audit`；日期：2026-10-18。

比较基准是上游 `cda0f9256665845cd9d6f3a0176880883f653a3d`，旧 fork 为 `2777543b2358c88cd1c2000f7fa342b764a0e213`。本文行号使用 `audit-backend.diff` 的旧 fork **新侧**行号；“A 行号”特指该 diff 的旧侧，不是当前干净分支的行号。逐 hunk 的精确旧/新范围、文件名、阅读深度、发现编号见 `audit-backend-coverage.json`。

## 1. 可以立即据此作的决定

**保留已经证明的 WHT 正确性修复。** 它修正上游已经公开支持的 `GGML_OP_TURBO_WHT` 的 32/64 分组、符号前缀、可选 scale 和工作组定位，不需要新增 GGML op、量化类型或私有 enum。根 agent 在真实 RX6800 上运行了本报告附的公共 GGML CPU/Vulkan 对拍：修复前 12 例中 10 例失败，只有无 scale 的 128 分组正向/反向通过；修复后 12 例全部通过，当前设备的最大绝对误差为 0。这是本轮真实回归证据，不能被归入“无价值的 TP5/XKV 改动”。[G01]

**不要把全部 one-line workgroup-stride 修改视为同一个现存上游 bug。** 普通逐元素、逐行、quantized CPY、RoPE 和 WHT 分别有不同的 host dispatch 契约。其中不少旧常数在当前 A 的固定 512 slab 布局下是正确的；新布局需要配套新公式。WHT 的分组契约与实际 grid 不一致则已经用真实测试重现。批量搬运 shader 的短 diff 会把配套主机布局、私有 MoE 行映射或 relay ABI 一并带入。[G01–G05、G21]

**将剩余独立的小行为逐项保留在候选清单，不按三个大项目名一次丢弃。** 比较有实际意义的下一批是 FA 覆盖 activation scratch 后的缓存失效、正确设备的 pinned host allocator、SCALE F16、MROPE/IMROPE 融合、CUDA 小批 TurboQuant VEC 内存策略、现有 Q2_0 的 x86 VNNI 实现，以及模型卸载后的 device staging 生命周期。它们的证据与风险各不相同，本文没有把静态合理性写成“已经验证有效”。[G06–G09、G14、G18、G31–G32]

**现有 phase seam 不依赖整个私有 backend runtime。** 已实现的 `llama_memory_release_compute()` 是明确的上下文生命周期边界：同步、失效 graph、释放实际 galloc buffer；它保留 scheduler、MoE/KV 和已拥有的输出。不要为保留这个小功能引入 TP5 command replay 或跨 context 同池。根 agent 已验证 CPU C phase 开/关的 3 请求、22 batch、145 行输出一致，逻辑 512 / 物理 1024 KV 与两次实际 compute bytes 归零。详见 `audit-memory-findings.md`；这里不重复声称 GPU phase 已验证。[G46]

## 2. 覆盖范围与证据强度

输入 `audit-backend.diff` 为 2,015,787 字节，包含 **126 个修改文件、883 个 diff hunks**；独立的 `audit-backend-hunks.json` 是对应 hunk 清单。仓库权威 inventory 是 `audit-inventory.json`，总 4634 个改动文件，排除 `artifacts/`、`reports/` 后 714 个。本文覆盖的是这 126 个文件，不能据此宣布 714 个文件都完成了语义深审。

机器清单区分以下状态：

| 深度 | 含义 | 不代表什么 |
|---|---|---|
| `all_changed_statements_static_review` | 文件中此次所有修改语句已读；小 shader 的模板相同项也逐个核对了公式 | 未改动整个源文件已审；所有 hardware/variant 已运行 |
| `whole_hunk_static_review` | 某一完整 hunk 的语句已读 | 文件中其它巨大 hunk 也已逐行审查 |
| `selected_statements_and_function_boundaries` | 阅读了有关语句和函数接口/调用边界 | 全函数所有实现分支已证实正确 |
| `symbol_and_hunk_boundary_screen_only` | 核对所有 hunk 标题、变更范围和新增符号用途，登记依赖与后续审查范围 | “该文件都是 TP5，所以可以丢弃”或“文件已经深读” |
| `prior_memory_audit_*` | 对应 allocator / scheduler 机制的细读证据在 memory 审计中 | 自动覆盖 backend 新增的其它异步接口 |

修改 shader 文件中，除大型 `vulkan-shaders-gen.cpp` 外，其余均阅读了修改语句，包括 `flash_attn.comp` 的 sparse 路径、`gated_delta_net.comp` 的 RBB 路径、matvec 的新行路由和融合分支。**新建 shader 文件**不在这个 126 文件的修改 diff 内；例如 FlashPrefill、relay、专用 qwen4 shader 只是从调用边界登记，不能宣称已经深读其本体。`ggml-vulkan.cpp` 的 302 hunks 保留逐 hunk 深度标记，巨大录制、融合、私有导出区没有伪报为全部语义深读。

测试只引用根 agent 实际完成的结果。旧源码注释中的“7 倍”“OOM”“illegal instruction”“字节一致”等描述均为历史注释，未当作干净分支的实测结果。

## 3. 已抽取并证明的 WHT 修复 [G01]

### 3.1 A 原本承诺什么

根 agent 提供的 `audit-focused/A-wht-0.txt` 包含 CPU `ggml_compute_forward_turbo_wht_f32()` 和两份 128 项 sign 表；`A-wht-1.txt` 包含公开 `ggml_turbo_wht()` 构造器。

公开构造器接受 F32 contiguous 张量、正向/反向 `direction`，分组 32/64/128，并要求行长度可以被分组整除。CPU 为每个分组取同一份 sign 表的前 `group_size` 项。可选 scale 在正向变换前逐组应用，在反向变换后应用。公开 API 因此已经定义了这些情况的含义；32/64 分组不是新 fork 才加的功能。

上游 Vulkan capability 当时仅承诺行长度可以被 128 整除，但这不排斥行宽 128 上的 group 32/64。原 shader 固定 128 butterfly 和 sign 位置，且没有 scale binding。原 host 用 `ne/group_size` 规划元素数，pipeline denominator 与 shader 又按 128/固定 512 处理。32/64 分组会产生不同的变换，某些跨 Y dispatch 还会漏写有效输出。

### 3.2 最小修改边界

本地 `vulkan-wht-fix.patch` 只涉及：

1. `ggml/src/ggml-vulkan/vulkan-shaders/turbo_wht.comp`：有效 group size、组内 butterfly、sign 前缀、可选 scale、实际 dispatch flatten。
2. `ggml/src/ggml-vulkan/ggml-vulkan.cpp`：pipeline binding/push constant、传入 `src[1]`、按完整 128-lane workgroup 调度、严格 capability 验证。

对应旧 fork 新侧是 Vulkan pipeline 7888 附近、host WHT 18776–18816、graph dispatch 22561 附近、capability 28453 附近，以及 WHT shader 的三个 hunks。补丁不包含新 GGML type/ops、TP5 头、XKV、RERoT 或源码重写脚本。根 agent 在集成时将 shader 的返回条件改为 `if (base >= ne)`，保证整个 128-lane 工作组一致进入或跳过带 barrier 的变换；原 A 的行宽 `%128` 支持边界继续成立。

### 3.3 测试能证明什么

`vulkan-wht-regression.patch` 添加 `fork/tests/a-seams/vulkan-wht.cpp` 和协议。测试只使用公共 GGML/CPU/Vulkan 头，分别直接运行两个 backend，**没有 scheduler fallback**。输出先写 NaN，因而漏写不会被偶然的零内存掩盖。覆盖正/反方向、32/64/128 分组、非平凡 scale、513 行的跨 Y dispatch；所有行宽为 128，严格留在 A 已有的 Vulkan 行宽支持范围内。

| 实测阶段 | 用例数 | 失败数 | 说明 |
|---|---:|---:|---|
| 原锁定 A | 12 | 10 | 只有无 scale 的 group128 正/反控制组通过 |
| 最小 WHT 修复后 | 12 | 0 | RX6800 上相对公共 GGML CPU reference 的最大绝对误差为 0 |

这是操作级正确性回归，不是整模型速度证明，也不能外推到所有 Vulkan 驱动。它足以支持把这两个 A 文件的改动作为独立、可 review 的正确性提交保留。

## 4. One-line shader 修改的真实分类 [G02–G05、G21]

### 4.1 逐行工作组 [G02]

`argmax.comp:18`、`cumsum.comp:22`、`l2_norm.comp:11`、`norm.comp:11`、`rms_norm_back.comp:16`、`soft_max.comp:40`、`soft_max_back.comp:17`、`solve_tri.comp:27`、`sum_rows.comp:15` 将旧的 `z*262144+y*512+x` 换成 `((z*NumWorkGroups.y+y)*NumWorkGroups.x+x)`。两个 convolution shader 则把 Y/Z 拼接的固定 512 改成 `NumWorkGroups.y`。

这把 shader 与实际工作组网格绑定，适合新 indirect/dynamic dispatch。但若 A host 在跨 Z 时恒定使用 X=512、Y=512，旧公式原本等价。必须先证明某个 A 合法 shape 的 host 不满足这一条件，才称为“修复上游现存 bug”。不能只看数值常量而宣布旧公式错误。

### 4.2 逐元素 512 slab [G03]

`generic_binary_head.glsl:36`、`generic_unary_head.glsl:25`、`multi_add.comp:129`、concat、conv2d_dw、glu、optimizer、out_prod、pad、silu_back、upscale 的新公式仍然保留 **每 X slab 512 元素**：

`(WorkGroupID.z * NumWorkGroups.y + WorkGroupID.y) * 512 + GlobalInvocationID.x`。

它不是“适用于任意 grid 的完全通用 flatten”。这组函数仍需要 `ggml_vk_op_f32()` 的 512 元素分块约定。`generic_unary_head.glsl` 同一 hunk 还无条件加入了两个 L2 pair bindings；若只需要网格修复，不能顺便把 pair 接口也迁入。

### 4.3 Quantized CPY [G04]

`copy_from_quant.comp:19`、`copy_to_quant.comp:901/929` 的索引改为从实际工作组数乘 `QUANT_K`。Vulkan host 新侧 `16734–16760` 同时修改调度单位：F32→quant 按 `32*block_size` 合并工作；quant→F32 按 block 数规划并乘 pipeline 的工作组 denominator。

旧公式与旧调度是在另一组单位下相互配套的。只搬 shader 或只搬 host 会改变覆盖范围。Turbo SET_ROWS 的其它四处 group flatten 也属于这一系列。若未来移植，应成套做元素覆盖、尾部、不同量化 block 大小的 public GGML 测试。

### 4.4 RoPE [G05]

四个 `rope_{norm,neox,multi,vision}.comp:5` 把固定 32768 改为实际 `NumWorkGroups.x`。Vulkan host `16567–16578` 将超大行数从固定 cap32768 改成设备实际 `maxComputeWorkGroupCount[0]`。

旧 host 固定 cap 与旧 shader 是配套的；新 host 与四个新 shader 也是一组修改。价值是适配设备工作组上限/减少分块，不是单行文字本身说明上游必错。未来验证应跨实际 X 上限前后，并包含多轴 position 模式。

### 4.5 Matvec 的 `mat_vec_first_row()` [G21]

IQ1、IQ2、IQ3、Q2_K–Q6_K、TQ4_1S 等 shader 的 one-line 替换不是上述 generic stride 修复。`mul_mat_vec_base.glsl:156–394` 中 helper 区分 HC_UP_FOLD、MUL_MAT_ID_GROUPED、routed MoE、Z 中的 token 编号以及 TP5 relay rows。组内偏移、`expert_i1` 来源、列是否 active、bias/scale expert slot、输出 offset 也一并变化。

这一系列只有在配套 route metadata 和 host dispatch 变化下有意义；单独抽 helper 会把私有执行协议穿进普通 matvec。应按“MoE 路由与批处理优化”完整登记，不能按“shader 一行安全修复”迁入。

## 5. 普通小行为清单

下表引用 `Vulkan:hNNN` 时使用原始数组的 **0-based hunk 编号**；JSON 还同时提供稳定 `path#hNNN`（1-based）标识和精确行号，避免两者混淆。没有实测的条目均为静态候选。

| 编号 | 真实行为、证据位置 | 与 A 的关系、保留判断 |
|---|---|---|
| G06 | Vulkan `ggml_vk_op_get_pipeline:15683`、supports_op:28362，加 F16 SCALE；shader generator 1557 附近生成相应类型 | 扩大既有 op 的合法 GPU 支持。小而可分；需 paired F16 output/capability 测试，暂不跟 WHT 搭售 |
| G07 | `rms_norm.comp:118` 加 MROPE/IMROPE 分支；Vulkan 24480/24565 扩大 RMS→mul→RoPE、RoPE→SET_ROWS 融合资格 | 新融合支持，不是基础 RoPE 修复。必须 host/shader 配套；需要 unfused oracle 覆盖分段 position/SET_ROWS |
| G08 | `quantize_q8_1.comp:82` 去掉已经 round 的值上的第二次 round | 等价清理候选，收益不明，优先舍弃以减少 A 差异 |
| G09 | `ggml_vk_flash_attn:14900` mask-opt 把 bitmap 写入 `prealloc_y` 后清 converted-activation cache 三个字段 | 高价值正确性候选。A 的 graph_cleanup 旧侧15970已存在这三个字段；如果同一 graph 后面重新用同一 activation，旧 key 可能错误命中。尚未完成合法图 before/after，不宣称已证实 |
| G10 | `ggml_vk_buffer_write_2d` / memset，Vulkan:h135/h141（新侧11502/11755附近），同步写从 transfer queue 改为 compute queue并加前后 barrier | 为与既有 compute 读写排队。可能可独立，但需先定义 A 的 buffer-level API 调用同步责任与跨 context 行为；不能把任意不合法并发当回归 |
| G11 | Vulkan:h29/h133/h136/h137，readback context 保留 buffer owner；transfer-write→host-read 可见性 barrier | 异步 lifetime/visibility 候选；需源指针、buffer、context 的有效生存期协议。phase seam 已同步之后才释放，不以此为理由强行引入 replay runtime |
| G12 | `ggml_vk_compute_forward:22936` 显式接收刚刚录制的 `vk_context&`，不从最早 tensor 的弱映射取回 | scratch 中途增容可能提交/retire 先前 context，改为提交真实当前 context。很有针对性的候选，但存在多处调用/cleanup 依赖；未来用强制 scratch growth 图证明 |
| G13 | `ggml_backend_vk_buffer_type_get_alignment:23583` 取 `max(16,minStorageBufferOffsetAlignment)` | 保证 root vec4 对齐。只有 min descriptor alignment=4 的设备及实际 vec4 访问形状能验证；view 仍需自己的未对齐路径。未验证，不普遍声称现有 A 错 |
| G14 | Vulkan 23622–23675、27458–27512，pinned host allocation/free/alignment/max-size 从硬编码 device0 改为 buft 的实际 device | 独立 multi-GPU ownership 功能。应为每设备创建稳定 buft，free 还原同设备；保留候选，不能只改 alloc 而不改 free/getter |
| G15 | Vulkan 23342/23722，`size==0 || n_copies==0`，strided write span 乘法上界 | 零行 no-op/溢出防护。其中 span 计算为新 router cache invalidation 服务；只抽 needed guard。应先核对公开 wrapper 是否已挡零参数 |
| G16 | Vulkan:h42，只有没有 almost-ready pending 时直接阻塞 `waitForFences` | 减少 polling CPU 消耗的独立策略，仍需 async fence/notification 状态矩阵，源注释不足以证明延迟无回归 |
| G17 | `ggml_backend_vk_get_device_memory:27385` 从 budget−usage 换 heap-size−usage 并饱和到0 | 一处同时改变“driver budget”与“physical heap”语义。下溢 guard 可评估独立修复；不能把物理 heap 容量宣传成实际可安全分配预算 |
| G18 | `ggml_backend_vk_free:23685` 销毁 device-level sync_staging；cleanup还按 env 重新设定起始 prealloc 容量 | 回收大 host-visible staging 有内存价值，但 device 是共享所有者；单 context free 时必须证明别的 context/同步复制不再使用它。旧写法不是可直接照搬的通用卸载修复 |
| G19 | env `=0` 语义、device enumeration catch、view/tensor/argsort logs、null shader/descriptor guard、PEP feature、large alloc override | 各自独立登记见§6；调试噪声与无验证 override 不作为核心保留项 |
| G20 | Vulkan max small cols8→18、Q8/Q6 skinny调优、outer-batch flatten、multi-column staged activation、MMVQ fixed K/rows | 真正通用小批性能功能，不必归 TP5；但 reduction/寄存器/shape策略依赖硬件。`mul_mat_vec.comp:189–319`、`387–409` 和 vecq/functions 是具体实现。本轮无性能证明，不自动迁入 |
| G21 | count_experts、mul_mm_id_funcs、mul_mm/cm2/mmq 以及 host13300附近：一次生成 expert→row routes，singleton/grouped/indirect execution | 可独立于 TP5 的 MoE 性能项目；需整体 metadata ABI 和 row-id位宽上界，不是一个 safe helper |
| G22 | `flash_attn.comp:780–1169` 从mask compaction metadata读取有穷/非负无穷项，负无穷才跳过；dense fallback | 与 TP5 头映射可分的 sparse attention 功能；需要 compactor新文件、scratch计划、GQA/stream/分块/sink/非有限mask测试。不要把“保留所有finite mask”直接等同于逐bit dense一致 |
| G23 | FA split-k reduce 加 attention gate，L2 pair融合 | 可以独立于大研究功能；仍需 host识别合法图、alias检查、allocator lifetime 与实际新shader。当前只登记，不伪装为B可直接hook现有llama图 |
| G24 | `gated_delta_net.comp` WORKGROUP_SIZE与多subgroup；另加 active_tokens与RBB块求解 | 多subgroup是调度优化；active rows属于predefined runtime；RBB改变多writer状态演化数学，必须分开。RBB不是上游 recurrent 正确性修复 |
| G25 | HC up/down/fold/combine/mix专门融合，R320、BF16 aligned path，选配Q8 activation | 部分修复只修新加入的HC融合（例如aligned up行映射），不能按A老bug迁入；Q8 activation明确近似，须质量与速度证明 |
| G26 | 巨大TP5录制、cached command、epoch/relay/wire、active/capacity frame、resource pin/retire | 私有后端所有权系统，边界登记；不能靠decorator假独立。见§7 |
| G27 | RERoT indexed FA、span expand、Q prep、head grouping、metadata bucket | 新执行语义与新op/binding；纯文本/调度策略可B化，native KV视图/graph构造需明确A seam |
| G28 | XKV reconstruct/attention/factor/canonicalize/landmark等 native handle路径 | 新op与native storage owner；device affinity gate是该功能的必要约束，不是通用A修复 |
| G29 | FlashPrefill pool/select/attention；CPU/CUDA/Vulkan计划错误与fallback | 新feature及plan ABI；众多其它backend的 `return false` 只是兼容该新增op，不能误计为十几个独立性能功能 |
| G30 | `ggml-cpu/ops.cpp:8646/8670/8728/8769`，F16 V仅在未请求GGML_PREC_F32时用F16累加 | 现有public精度契约的独立修复；portable_features+root已给公共GGML CPU regression before FAIL/after PASS。与同hunk附近TP5 headmap分开移植 |
| G31 | `ggml-cuda/fattn.cu:3130` Turbo K/V且≤16queries并可用vector kernel，优先VEC避免大F16 temp | 既有TurboQuant的小批内存/性能策略；非RERoT。缺本轮CUDA实测，作为候选而非弃用结论 |
| G32 | x86 `quants.c:552–632` 新现有Q2_0×Q8_0 VNNI/标量实现；后半为PQ2_0版本 | 现有Q2_0部分可从私有类型拆开，需确认arch-fallback及源符号链接；变动不同CPU ISA上的累加策略，不宣称无损速度提升 |
| G33 | PQ2_0 128分组格式、Q2_0/PQ2_0编号移动、gguf type42 remap、RPC_COUNT修改 | 私有wire兼容负担。WHT/phase/B采样均不需要它；保留A类型编号，不能按“兼容旧文件”的注释盲改解析 |
| G34 | `ggml-turbo-quant.c:225–717` group-aware row quant/dequant、公开WHT row helper、layout fingerprint | 可作为独立算法库候选，但当前函数签名仍带ggml_type和私有布局；应独立描述格式/字节协议，C显式适配。不是直接把A源码目标改名为B |
| G35 | memmove regions、device-copy ranges、snapshot async、sanitize-nonfinite等公开/后端接口 | 四种语义不同，详见§7；不可用“直接hidden支持”一项掩盖所有约束 |
| G36 | graph_optimize 的 add_alloc_dep、API version2→3、allocator/source lifetime | 跨后端ABI变化；保留融合才可能需要，phase trim本身不要求整套ABI升版 |
| G37 | `llama-batch.cpp:444/801/860/888` 原batch source-row tracking由ubatch data_t拥有 | 独立小观察能力：OFF无分配，reserve没有source map，clear保留开关。若C掌握原batch分组可在C实现；要知道A内部split精确映射须一个explicit callback/getter seam |
| G38 | `llama-kv-cache-iswa.*` base/SWA失败转发、bounded-hot reserve/postcompute、RERoT tags/views | base与SWA组合语义，不能将普通clear/seq_rm视作事务；见§8，已补全本体diff阅读 |
| G39 | DSA/MSA getter转发，DSV4 compressed HCA/indexer stream参数补齐 | 容量统计与共享stream配置的小机制；不能从base容量直接推总字节，n_ctx_kv仍需按memory family定义最小契约 |
| G40 | llama-graph 的direct hidden、source rows、graph reuse、Hadamard memo、RERoT/XKV/FlashPrefill构造 | 只对特定接口/生命周期细读，其它巨大本体本报告保留边界审查标记，不能一并声明无价值 |
| G41 | backend interface null字段传播、CUDA compiler选择、Vulkan硬编码drm依赖等 | 机械兼容/构建便捷性；硬编码系统路径和重复link行不进入干净A |
| G42 | `ggml.c:7993/8066` graph UID getter/setter/view继承；1894 view错误日志 | UID为新replay identity用途，需证明graph slice不会碰撞；错误日志只提高诊断，不解决lifetime越界 |
| G43 | Vulkan FA module/shared-memory精度计算、Turbo cm2独立decode、split-k scratch bucket | 子机制分别评估。512 bucket主要稳定replay cache，但可能多留scratch；F32 module与shared memory计算必须配对验证 |
| G44 | Vulkan `supports_op:28136` Turbo SET_ROWS明确拒绝F16 source | capability与实际只编译F32 shader一致的强候选；需查询/调度fallthrough测试。不把新stderr日志一起带入 |
| G45 | 新hierarchical argmax/staged top-k shader管线 | 另一个独立sampling/router性能点；新shader本体尚未在本范围深读，不能用B shortlist已测试代替GPU top-k证明 |
| G46 | shared compute pool/phase trim/capacity retain+seal | 完整分类见memory报告M14–M17与风险；本轮采用小显式release seam而非旧共享池全部实现 |

## 6. 容易被“小修复”文字误导的项目

### 6.1 几个并不充分的修复证据

Vulkan `pipelineExecutableInfo` 的 enable 只有在实际 feature query 支持时才成立；扩展名出现不等于feature必为true。null SPIR-V指针绕过vector赋值，也不自动使后面的SPIR-V解析或pipeline有效；允许null VkBuffer则还依赖descriptor布局和device feature。此类条件不能凭几行guard认定安全。[G19]

`GGML_VK_LARGE_ALLOC` 通过env扩大驱动暴露的allocation/range上限，在旧注释已经提到device lost的背景下尤其不能当成“修复限制”。干净分支应保留真实device caps，除非专门的设备证据支持另一个合法API路径。[G19]

环境变量从“只要存在”为true变为`atoi(value)!=0`，会让 `=0`、空值及非数字值有新含义。这是用户接口选择，不能混作透明的性能优化。CUDA CMake 的 `/usr/bin/gcc` fallback以及Vulkan固定`/usr/include/libdrm`、重复link line也只是旧机器环境选择，不应污染A。[G19、G41]

### 6.2 CUDA allocation 注释与实际 diff 不一致

`fattn.cu:3258–3271` 的注释声称“修复Turbo temp under-allocation”。但相对当前锁定A，旧侧已经对TILE/MMA置 `need_f16_K=true; need_f16_V=true;`；新侧是改成“源不是F16才true”。因此不能将它报告成锁定A漏配Turbo buffer的已证实修复。合理的独立行为是前面的“小batch优先VEC”策略；是否节省预留、哪些源码路径已避免F16重复转换，必须读完整调用链和运行分配统计后再判断。[G31]

### 6.3 Device staging回收与phase trim不同

device-level `sync_staging` 是backend资源，galloc临时图buffer是另一类资源。当前phase API对后者提供实际bytes；它不保证Vulkan driver“显存free”同步上升，也不保证device staging被回收。旧fork在一个backend析构中清device共享staging，缺少“其它context已不再用”的局部证明。要保留该回收，应有device-level owner/refcount或明确device drained边界，不应把它暗藏进release_compute。[G18、G46]

## 7. A/B/C 可执行边界：按能力而不是改目录

### 7.1 可以真正独立的 B

已经完成的 `fork/sampling` 是标准库类型和算法，不调用A头/API；C取公开logits并编排真实A sampler链，B的compact输出只是一个选择性输入优化。这是独立库的可检验定义。

同样可成为B的是纯host策略、独立行号/公平调度数据结构、独立格式明确的量化/WHT算法、统计分析。B的测试应只依赖它自己的输入输出。B若持有`ggml_tensor*`，经A callback访问private context，或者链接A内部vulkan symbol，并不会因为位于`fork/`就获得独立性。

### 7.2 必须明确登记的 A 接缝

| 需要的能力 | 当前可行接缝 | 不应采用的替代做法 |
|---|---|---|
| phase结束后归还compute buffer并准确观察 | 已实现显式`llama_memory_release_compute`、`llama_memory_compute_bytes` | 改旧`llama_set_decode_phase`的暗含语义；打印估算VRAM代替实测 |
| 物理KV容量与逻辑window分离 | root负责`n_ctx_kv`字段、getter与memory constructor链，family边界显式验证 | 把物理pool限制为`n_ctx_train`来伪装scratch不增长 |
| 获得A内部ubatch与原batch的准确行映射 | opt-in owned row map + public callback/getter，或让C直接掌握切分 | 从token/pos重复内容猜row，借用临时vector裸指针 |
| 把hidden state直接从一模型后端传给另一模型 | 公共可拥有tensor/lease描述、byte/layout/device约束、native queue lifetime接口 | 公开void*却让B依赖A私有tensor布局；B回调A等同假独立 |
| 在A现有FA/Matmul内部替換dispatch/fusion | 少量显式backend hook需完整supports/alias/allocator lifetime contract；或把整块graph构造由C拥有 | 编译前regex重写A源文件；偷改enum/backend vtable而不登记 |

### 7.3 四种新copy能力需要分别定义 [G35]

`ggml_backend_tensor_memmove_regions` 是同buffer、ordered regions、overlap可处理的移动；先验证整组边界，CPU host直接memmove，GPU可以用bounded scratch。它主要服务KV物理压缩，属于比`seq_rm`更底层的storage seam。

`device_copy_ranges` 则主动拒绝src/dst以及各range交叠，不能把它当memmove。Vulkan实现还拒绝写weights、限制同device、对齐和buffer边界，并插入既有compute stream；这是direct hidden/state复制更合适的独立语义。

`set_tensor_snapshot_async` 的“snapshot”表示在调用时拥有host输入bytes，调用者立即可以释放/复用指针。Vulkan旧实现用`updateBuffer`，只支持4字节对齐、最多65536字节，dry-run不得排队；它与普通借用source pointer的async set不能混为一谈。META需要先验证全部destination再做任何一份capture。

`sanitize_nonfinite` 明确改变数值，不能作为通用“正确性清理”隐藏调用；它若只服务一种研究状态维护，需要独立开关和可观察计数。它不是direct copy不可分离的必需项。

这四组都需要native queue/lifetime支持，但可以用明确的小公共接口表达。旧fork通过backend ABI、collective、private proc-address一次扩张实现，不代表最小干净fork也必须搬入那些大系统。

## 8. 补齐 memory 审计的 ISWA/DSV4 小机制

本轮完整读取 `llama-kv-cache-iswa.cpp/.h` 所有14/8个hunks以及DSV4三个hunks，解决memory报告早期`outside_initial_diff_not_yet_reviewed`的范围缺口；以本报告和JSON为后续证据，不能回写成早期已经完成。

- **base有bounded cparams，SWA明确传nullptr。** ISWA constructor新侧26–131把cparams仅给base，SWA保留无bounded hot配置；SWA仍只有在size_swa==size_base时才传入stream staging/phase arena/maximum pool。[G38]
- **容量和用量getter只转发base。** 新侧457–481报告base物理容量/热容量/逻辑used/seqused；不能当base+SWA总bytes。`can_use_legacy_attention`是两者AND，`is_xkv_bounded_hot`是OR。[G38]
- **`try_clear`和`seq_rm`只是base-first短路，不是事务。** base拒绝时SWA不改；但base成功、SWA后拒绝时没有回滚base。因此不能对用户承诺false=全无改动。这个风险比方法名字更重要。[G38]
- **write-tag/reader-view也不是旧状态恢复。** 如果第二侧失败，代码clear第一侧；若第一侧以前已有一个有效tag/view，会被clear而非恢复。publish/reclassify/add-ref先分别preflight，两边count一致依赖串行调用且中间不变；需要在上下文同步契约中明确。[G38]
- **bounded-hot prepare与postcompute有独立生命周期。** 两种batch split都先reserve base hot slots；context拥有reservation。apply/next重置finalized，success/failure仅在两侧都成功时finalize，失败保留重试。这个逻辑只在引入hot reservations后需要，普通n_ctx_kv seam不应顺手带入。[G38]
- **DSV4的HCA/LID补传已有stream参数。** 新侧1262–1286，除了新增cparams位置以nullptr占位，HCA、LID也补上stage bytes、phase arena、maximum pool；这是真实配置传播变化，不是新统一物理池。需要按具体compressed memory family验证支持，不能凭“都叫KV”套用同容量规则。[G39]
- **DSA/MSA统计转发分别取MLA/base。** `llama-kv-cache-dsa.cpp:161–178`、`msa.cpp:157–174`没有引入新分配器，只增容量/使用量可观察性。[G39]

物理KV池放大后，attention可覆盖的真实物理span及相关scratch可能增大。文档应说逻辑window约束每条请求的语义窗口，不能承诺它自动给compute scratch设置上限。当前root的512逻辑/1024物理CPU测试证明的是该具体配置的容量与输出一致，不是所有memory family的峰值恒定。

## 9. 审查深度未升级为完整语义审计的文件

`audit-backend-coverage.json` 的 `boundary_only_files` 和 `partially_reviewed_files` 是机器可读的最终准确列表。它们不是“舍弃清单”。重点范围如下：

| 范围 | 已做到 | 仍需针对将来保留项做到 |
|---|---|---|
| `ggml/src/ggml-backend-meta.cpp` | 所有74个hunk边界与新增mapped span/indexed replica/async owner/collective接口索引 | 逐range、多设备、slice/stride与rollback实际数据正确性；不能全归TP5后跳过 |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 所有302个hunk索引；普通dispatch、读写、内存、capability、WHT等指定hunk细读 | 巨大private replay/region fusion/导出区只作接口边界/部分语句审查；新shader本体与完整retirement证明未做 |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | 所有15个hunk的pipeline变体与新增shader依赖边界 | 巨大注册区及分别新建shader不是本体全审 |
| CPU/CUDA新RBB/XKV/FlashPrefill实现 | 全部modified-file/hunk与入口登记，CPU F32精度、CUDA VEC策略细读 | 新算子完整数值/边界与GPU设备执行，不能用public API存在代替证明 |
| `src/llama-graph.cpp/.h` | 原direct hidden/owned cross输出及reuse生命周期审查，加所有diff符号边界 | 大型RERoT/XKV/FlashPrefill graph建图、Hadamard memo及模型shape组合仍需按保留范围细审 |
| `ggml-turbo-quant.c`与private量化类型 | explicit group/layout helpers及type依赖边界 | 外部格式规范、不同group字节互操作、quant/dequant错误状态全测 |

coverage的完成含义是“本轮交付的范围、发现和缺口已完整登记”，不是“所有私有实现已证明可弃或可迁”。与root的714路径总账合并时，必须保留这些深度字段；不应把有一行record就统计成深读。

## 10. 交付与复现

- `audit-backend-findings.md`：本文，46项细分能力/候选与边界。
- `audit-backend-coverage.json`：126文件、883hunks，精确新旧行号、发现编号、阅读深度。
- `backend-audit-input/finalize_backend_coverage.py`：可复现生成器，校验126/883和权威inventory路径。
- `vulkan-wht-regression.patch`：公共GGML CPU/Vulkan操作对拍，test-only；根已实测。
- `vulkan-wht-fix.patch`：两A文件最小修复提案；根应用时作uniform workgroup return微调，当前干净分支是最终权威实现。
- `audit-memory-findings.md` / `audit-memory-coverage.json`：前序memory机制深审与phase seam实际验证证据。

复现清单生成：`python3 backend-audit-input/finalize_backend_coverage.py`。此命令不访问远程、不改A，也不运行GPU测试。GPU实际build/test由root的干净分支日志和提交中的测试协议复现。
