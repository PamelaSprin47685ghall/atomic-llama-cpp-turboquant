# TP5 TARGET 验证容量化设计（M3 定稿，供评审与实施依循）

- 状态：设计定稿待评审，未实施。本文只定契约与分区闭环，不改代码。
- 约束：只读现状已盘点；本文只新增本文档；不构建、不运行、不改代码。
- 适用对象：qwen4exp 48 层有状态主干的 TARGET 验证 phase（`GGML_PREDEFINED_TARGET`），单序列第一阶段。

## 0. Manager 边界决策（本文的硬前提）

M3 实施必须以以下三条为不可谈判前提，本文所有双长度契约与分区判据都由它们推导：

1. **验证容量复用 MTP 会话的 `verify_tokens`，单一容量。** TARGET 的输出容量等于 token 容量（`capacity_outputs == capacity_rows == verify_tokens`），符合 `ggml_predefined_make_frame` 对 TARGET 的 `outputs == tokens` 约束。不以 `n_batch` 为容量规格，不另设 TARGET 独立容量。
   - 依据：`ggml/src/ggml-predefined.cpp:102-116` 对通用请求的 `outputs <= tokens` 检查与 TARGET 的 `outputs != tokens` 拒绝分支；`src/llama-context.cpp:2477-2488` 把 `n_outputs == n_tokens && n_tokens <= capacity->verify_tokens` 识别为 TARGET（含零候选的单 token decode 退化情形）；`ggml/include/ggml-predefined.h:29-51` 的 `capacity.verify_tokens / limits.output_rows` 定义。
2. **第一阶段只支持单序列（`n_seq_max == 1` 前提）。** 多序列明确列为边界，留待后续。`equal_seqs` 与单序列断言是准入条件，不是可顺手放宽的优化项。
   - 依据：`src/llama-graph.cpp:4117-4123` 的容量分支 `GGML_ASSERT(n_seqs == 1 && n_seqs_unq == 1)`；`src/models/qwen4exp.cpp:build_layer_attn_linear` 的 `equal_seqs` 与 `n_tokens == n_seq_tokens * n_seqs` 断言族（现状仍按实际行构图，见 §2）。
3. **prefill 保持精确路径不变；容量化只覆盖 TARGET 验证 phase，不把 prefill 抬成容量图。**
   - 依据：`src/llama-context.cpp:2489-2519`——非 MTP 非 TARGET 一律 `GGML_PREDEFINED_PREFILL`，且 `Target/prefill stays exact until the 48-layer stateful trunk has independently completed capacity lowering`（`predefined_capacity_rows_current = ubatch.n_tokens`）；`TP5-MTP-PREDEFINED.md §10` 的 phase 豁免仅限预定义 MTP。prefill 的精确构图、phase 分离、buffer 生命周期一律不动。

违反任一条即 out-of-scope，必须停下并交回 Manager，不得在实施中自行放宽。

## 1. 双长度契约（active vs capacity）

### 1.1 术语

- `capacity`：地址稳定维度。图定义、物理张量形状、stride、KV/rollback 槽位上限、RELAY payload 上限按它分配。TARGET 第一阶段恒为 `verify_tokens`。
- `active`：有效工作维度。本轮真正要算的行数。`frame.active_tokens`（有效 token 行）、`frame.active_outputs`（有效输出行）。TARGET 约束下 `active_outputs == active_tokens`（`ggml-predefined.cpp:116`），且 `1 <= active <= capacity`。
- `inactive`：`[active, capacity)` 后缀。已分配地址，但本轮不得产生任何语义效果。

### 1.2 在 trunk 图中的承载方式

| 承载点 | capacity 侧 | active 侧 | 现状与改动方向 |
|---|---|---|---|
| `frame` 字段 | `capacity` 由会话定义隐含（`verify_tokens`） | `frame.active_tokens / active_outputs / active_sequences / context_tokens / draft_tokens / epoch / slot`（`ggml-predefined.h:68-84`） | frame 生成已就绪（`llama-context.cpp:2493-2505`）；TARGET 的 `capacity_rows` 尚未抬到 `verify_tokens`（`2514-2519` 仍精确），这是 M3 第一改动点 |
| 张量形状 | 所有按行定形的输入/中间张量第二维（或等价行维）固定为 `capacity`：`tokens [capacity]`、`embd [n_embd, capacity]`、`pos [capacity * n_pos_per_embd]`、`out_ids [capacity_outputs]`、`K/V idx / mask` 按放大后的 `shape_ubatch` 建图 | 形状不表达 active；active 只出现在间接参数与掩码中 | 输入壳已容量就绪：`llama-graph.cpp:2695/2697`（`n_tokens_capacity / n_outputs_capacity`）、`3614/3619/3712/3745-3749`；KV 输入已有单序列容量分支 `4117-4128`；主干 attention/GDN/MoE 仍按实际行（见 §2），须逐区下沉 |
| dispatcher 间接参数 | 物理 stride/workgroup 上限按 `capacity` 派生；`ggml_predefined_dispatch_arguments` 只允许按 `frame[extent] * scale / divisor` 上取整（`ggml-predefined.h:96-113`） | 实际派发 workgroups 按 `active_tokens / active_outputs` 计算；`OUTPUTS` 派发跳过非活跃后缀 | `llama-predefined.cpp` 注释与 `llama-graph.cpp:234-286` 已声明“OUTPUTS 派发跳过非活跃”；Vulkan 侧 `matmul_indirect` 要求 `n == expected_capacity`（`ggml-vulkan.cpp:10150-10174`），`update_predefined_dispatches` 在 `!predefined_complete && active != capacity` 时 fail-closed（`27488-27544`）：间接分类未全即不得执行小前缀 |
| 状态写入掩码 | KV `k_idxs/v_idxs`、recurrent/conv/indexer 写入索引按 `capacity` 建地址空间 | 提交时只写 `[0, active)` 行；`set_input` 只写活跃前缀，非活跃后缀清零且语义无害 | `out_ids::set_input:234-286` 已示范（容量模式下清零后缀、越界抛错）；KV/conv/recurrent 写入仍按 `ubatch.n_tokens` 逐行（`llama-kv-cache-dsv4.cpp:467-556`、qwen4exp `build_conv_state_at:1601-1650`），须收敛 |
| RELAY/消费者长度 | payload/bank 上限按 `capacity`（`verify_tokens * width` 元素） | 包头 `active_elements = width * active_rows`（`collective.cpp:6433-6440`），P2 按包头消费 | `submit_epoch_chain:6310-6331` 校验 `active <= capacity`；`6796-6808` 明确 `active < capacity` 时仅 RELAY direct producer 算最大容量执行，legacy P1 直接拒绝。TARGET 下沉后必须全 stage direct，否则 fail-closed |

### 1.3 不变量（实施期任何阶段都不得违反）

1. **inactive 行不得读写。** 不得进入 KV、`recurrent`/conv/indexer 状态、collectives 求和、采样 `get_rows` 源。读越界与写越界同等严重（flash 内核行循环、QSA TopK、MoE 打包都须服从此条）。
2. **push-constant 不得把容量当有效。** shader 可见的行截断参数必须是 active 派生（`token >= active_rows` 截断，如 `tp5_hc_resume.comp:113/125/205/221/257`、`tp5-rows.h:114-115`）；`late.capacity_rows` 这类容量字段只用于地址/上限校验，不得直接作为执行行数。
3. **stride 属于不可变容量布局。** 任何 shader 不得从 `active_tokens` 推导物理 stride（`ggml-predefined.h:64-67` 注释即此意）。
4. **无间接分类即无小前缀执行。** `active != capacity` 时若 `predefined_complete` 未达成或任一 stage 非 direct，fail-closed 回拒绝，不得静默复制 padding 行（`collective.cpp:6796-6808` 现行语义即标准）。
5. **容量行变化不改变图定义。** 同一 `verify_tokens` 下 `1..verify_tokens` 行都命中同一定义；复用键只允许 active 可变（见 §4）。

## 2. 区域切片：改动点、闭合判据、依赖与可复用机制

推进顺序即依赖顺序：输入/位置 → Attention（含 FA/QSA）→ GDN/recurrent → MoE → terminal producer → RELAY/消费者。后一步依赖前一步的行契约，不得跳序。

每区的闭环判据统一为四句：**输入只写活跃前缀且后缀无害；计算不触碰非活跃地址；状态只提交活跃行；通信长度按活跃元素算。** 下表“改动点”指 M3 必须做的事，“现状”指只读盘点已确认的起点。

### 区 0 — 输入/位置（地基，先行）

- 现状：输入壳已容量就绪（`llama-graph.cpp:2695/2697/3614/3619/3712/3745-3749`，`can_reuse:98-106/146-159/183-190/288-295`）；`set_input` 仍按 `ubatch.n_tokens` 写前缀，`out_ids` 容量模式已清零后缀（`234-286`）；TARGET 的 `capacity_rows` 仍恒等于真实行（`llama-context.cpp:2514-2519`）。
- 改动点：
  1. `process_ubatch` 中 TARGET 分支抬为 `predefined_capacity_rows_current = capacity->verify_tokens`，`predefined_capacity_outputs_current = capacity->verify_tokens`（注意不是 `max(1, active)`；输出容量恒等于 token 容量，见 §0 决策 1）。
  2. `set_input`（tokens/embd/pos）只写 `[0, active)` 前缀，非活跃后缀保持分配但清零或保持无害值；越界（`active > capacity`）抛错而非截断。
  3. `pos` 仍用真实 `ubatch.pos` 前缀；`context_tokens` 取 `max(pos+1)`（现行 `2434-2457` 语义保留），不得把容量当上下文长度。
- 闭合判据：同一定义下 `active = 1..verify_tokens` 均可提交；`out_ids` 非活跃后缀全零；`can_reuse` 在容量相等、active 可变时命中（见 §4）；CPU 契约见 §3.1。
- 可复用：`frame/active_rows/indirect dispatch/row program/predefined_* 校验` 全套；MTP `graph_mtp:601-610` 的 `rows = n_tokens_capacity` 建图范式。
- 依赖：无前置；是所有后区的地基。

### 区 1 — Attention（含 Flash Attention 与 indirect 规则）

- 现状：通用 KV 输入已有单序列容量分支（`build_attn_inp_kv_impl:4117-4128`：`shape_ubatch.n_tokens/n_seq_tokens` 放大到 capacity 再建 `k_idxs/v_idxs/mask`）；但 qwen4exp 主干 Q/K/V 投影视图行数写死（`qwen4exp.cpp:1214/1222/1226-1230` 用成员 `n_tokens` 做 `view_3d/reshape/cont`，`build_qkvz:760-768`、`build_layer_attn:1199-1205` 跟实际行走）；Flash 路径 `llama-graph.cpp:3880-3906` 走 `ggml_flash_attn_ext`，尚缺与 matmul indirect 同等的 frame 间接入口（matmul 间接见 `ggml-vulkan.cpp:10150-10174`）；QSA（`qwen4exp.cpp:884/888-889/913-942`）的 `width/top_k/cell_blk/bias/dirty` 依赖 `n_kv` 与压缩比，`build_attn_qsa` 含 mask 改写与 `set_rows`。
- 改动点：
  1. 主干 Q/K/V 投影与 `build_qkvz` 行维改按 `n_tokens_capacity` 建图，QSA 以外的 attention 计算行由 indirect/掩码截断到 active。
  2. KQ mask 按放大 `shape_ubatch` 建图（复用区 0 的 KV 分支），flash 内核的行循环必须服从 mask/行截断，不得读 `[active, capacity)`。
  3. QSA：`n_tps/top_k/cell_blk` 仍按 `n_kv` 定形（不是行容量决定），但 QSA 的行输入必须只消费 active 前缀；`full_coverage` 与 pooled cache 拓扑在验证行变化时保持同一定义（风险见 §5）。
  4. 所有 push-constant 行参数统一为 active 派生；容量只进上限校验。
- 闭合判据：`active < capacity` 时 flash/QSA 内核不读非活跃行（真机证据）；KQ mask 形状按 capacity，语义按 active；KV `apply` 只推进 active 行对应的位置。
- 可复用：`build_attn_inp_kv_impl` 容量分支；`predefined_mtp_dynamic` 的“容量相等、active 可变”复用思想（TARGET 版见 §4）。
- 依赖：区 0 的容量与 frame；状态写入（KV 范围）与区 2 联动。

### 区 2 — GDN/recurrent（含 conv/indexer 状态写入）

- 现状：风险最大。`build_layer_attn_linear:1275-1295` 断言 `equal_seqs` 且 `n_tokens == n_seq_tokens * n_seqs`，`beta/alpha/gate:1312-1328` reshape 全用实际行；`build_conv_state_at:1601-1650` 按 `n_seqs` 读卷积行并按 `slot*mem_size+kv_head` 写回；GDN 快照写入按实际行切分（`delta-net-base:525-958`）；`dsv4 state_write_idxs/pos:467-556` 按 `ubatch.n_tokens` 逐行生成；`qwen4exp.cpp:1689` 注释提醒显式 pin 会拆图并影响 fused GDN 派发。
- 改动点：
  1. reshape/门控/卷积读写的行维改按 `capacity` 建地址，提交掩码只含 `[0, active)`。
  2. `state_write_idxs/pos` 生成只覆盖活跃行；`apply` 与回滚槽按 active 提交，多写或少写都算失败。
  3. fused GDN 派发键必须按容量命中（形状派发），active 只进间接参数；显式 pin/拆图行为在容量图下重验。
- 闭合判据：`active=1` 与 `active=verify_tokens` 在同一定义下交替执行后，recurrent/KV 状态与精确图逐行一致；回滚后重放一致（真机必验）。
- 可复用：frame epoch/slot/generation；`submit_epoch_chain` 的 frame 版本/phase/active 校验。
- 依赖：区 0/1 的行契约；是区 5 之前必须先闭合的状态门。

### 区 3 — MoE（打包/阈值/路由）

- 现状：`build_moe_ffn:3240-3368` 取 `cur->ne[1]` 为 `n_tokens`，`logits[n_expert, n_tokens]`、`argsort_top_k[n_expert_used, n_tokens]`、`weights` 全随行；主干 `qwen4exp.cpp:1429-1443` 只传 `n_expert/n_expert_used`；专家打包、分组 topk、共享专家门控无容量概念。
- 改动点：路由张量物理形状固定为容量（第二维 `= verify_tokens`），打包只处理活跃行并只归约活跃行的专家输出；阈值数字本身不动，动的是行参与集合。
- 闭合判据：路由/打包/归约不读不写非活跃行；`active` 变化不改变图定义；CPU 形状契约 + 真机位级（与精确图同输入同输出位级一致）。
- 可复用：row program（`tp5-rows.h:11` 的 `none/dense_columns/add`，`ggml-vulkan.cpp:12146/17047` 定义，`27546-27598` 按 `active_elements` 算 row 参数并校验 `capacity_elements`）。
- 依赖：区 0/1 的 active 定义；输出侧（区 4）的行裁剪与之对齐。

### 区 4 — terminal producer（RESULT 直写门）

- 现状：主干末端 `result_output:557-559` 对实际行做 `build_lora_mm(model.output)`，元素数 `= width * 实际行`，对不上 `width * 容量行`；末层 `get_rows` 裁剪（`qwen4exp.cpp:501-509/542-545`，`embeddings_nextn/masked` 控制的延迟 gather）与 `llama-graph.cpp:5449-5502` 采样循环仍按 `ubatch.output/n_tokens` 逐行；LateBind 7 组 push 常量用 `late.capacity_rows`，而行程序执行规则是 `token >= active_rows` 截断——两者必须统一为 active + indirect 重补。
- 改动点：
  1. 输出张量达到容量元素数（`width * verify_tokens`）并具备 direct 行派发；`key.n_elems` 对齐容量。
  2. 末层 gather/out_ids 与采样侧改按 `frame.active_outputs` 选行（TARGET 下 `active_outputs == active_tokens`）。
  3. LateBind push 常量：执行行数改 active 派生，容量只做上限；缺失的 indirect 重补必须补齐。
- 闭合判据：`RESULT` 交接的 generation 匹配规则下，非活跃行永不触碰（`token >= active` 截断）；`active_elems = width * active` 进 payload 头（与区 5 对接）。
- 可复用：`collective.cpp:6322-6331` 的 frame 校验；`tp5-rows.h:114-119` 的 `latebind_token_is_active/active_q_elements`。
- 依赖：区 0-3 的 active；是 RELAY 的前置门。

### 区 5 — RELAY/消费者（传输与下一消费）

- 现状：Vulkan 侧行程序与 `submit_epoch_chain` 校验已就位（`collective.cpp:6310-6331/6433-6440/6796-6818`）；LateBind/RELAY 侧 7 组 push constant 全带 `active_rows`、以 `token >= active` 截断；`terminal producer/direct-host` 与 P1 epilogue、`hidden RESULT/CARRY/SEED` generation 匹配已就位；`capacity_rows = 4` 一致才命中否则 fail-closed（工作记录语境，TARGET 下对应 `verify_tokens`）。
- 改动点：
  1. TARGET 验证图的 `active_elems` 进 payload 头与 P2（`6433-6440` 语义），hidden RESULT 交接按 generation 消费。
  2. 全 stage 必须为 RELAY direct（`tp5_relay_direct_stage` 全绿），否则 `active < capacity` 时 fail-closed，不得回退 legacy P1（现行 `6796-6808` 即此语义，TARGET 下沉后同样适用）。
  3. 下一消费者（MTP catch-up/draft 衔接、采样、跨 context hidden copy/synchronize）按 `frame.active_*` 消费，不得按容量多读一行。
- 闭合判据：`active < capacity` 时 RELAY direct payload 位级一致；五卡 reduce 只读活跃元素；跨 context hidden 拷贝长度按 active；teardown 排空仍走原生有界路径（AGENTS 安全门不因容量化改变）。
- 可复用：全部传输侧机制；`frame epoch/slot` 与 native completion 排空。
- 依赖：区 0-4 全部闭合后方可闭合本区。

## 3. 测试与证据（CPU 契约 vs 真机必验，严格分开）

### 3.1 CPU 契约（可静态/单机断言，无需 GPU）

- 形状：输入 `tokens/embd/pos/out_ids` 行维 `== verify_tokens`；KV `k_idxs/v_idxs/mask` 按放大 `shape_ubatch` 建图；MoE 路由第二维 `== verify_tokens`；terminal 输出 `n_elems == width * verify_tokens`。
- frame：`ggml_predefined_make_frame` 对 TARGET 的 `tokens == sequences + draft`、`outputs == tokens`（`llama-context.cpp:2477-2488` 与 `ggml-predefined.cpp:116` 同义）；`active <= capacity`；`context_tokens = max(pos+1)`。
- `out_ids`：容量模式下非活跃后缀全零、越界抛错（`llama-graph.cpp:234-286` 语义推广到 tokens/embd/pos 的 `set_input`）。
- 复用键：容量相等、active 可变时命中；`definition_uid`/epoch/slot/generation 单调；`predefined_mtp_dynamic` 的 TARGET 对应键（见 §4）行为有单测覆盖。
- 间接参数：`ggml_predefined_dispatch_arguments` 对 TOKENS/OUTPUTS extent 按 active 算 workgroups，空输出零 work；`update_predefined_dispatches` 在 `!predefined_complete && active != capacity` 时拒绝。
- `test-predefined-capacity` 类契约：同一定义下 `1..verify_tokens` 行的形状/键/参数计算全过。

以上任何一项失败都是实现 bug，不得用“真机再看”推迟。

### 3.2 必须真机验证项（CPU 仿真不得代替）

- flash 行跳过：`active < capacity` 时 flash 内核不读非活跃行，无越界 fault（`dmesg` 零新增是必要非充分条件，须结合位级比对）。
- GDN fused 派发：容量形状下命中同一 fused 路径，active 只进间接参数；显式 pin/拆图行为重验。
- recurrent 回滚一致性：`active` 高低交替 + 回滚重放后，KV/recurrent 状态与精确图逐行一致。
- RELAY 位级：`active < capacity` 时 RELAY direct payload（F16/F32 wire）位级一致；P2 按包头 `active_elems` 消费。
- 五卡 reduce 只读活跃：reduce 求和不读非活跃元素（FD 增量语义与现有 mesh 测试同等严格）。
- 跨 context hidden copy/synchronize：长度按 active，无多读多写。
- 安全门回归：RELAY 有界自旋上限固定（不得动态递增）、teardown 原生有界排空、watchdog 不关闭——容量化不得引入新的自旋或 `vkDeviceWaitIdle` 式伪造完成（见 AGENTS.md 真机安全门）。

## 4. 与 MTP 单定义的关系（不新增图家族）

- **不新增 `verify_2/3/4` 图家族。** TARGET 与 MTP 共享“一份定义、active 可变”思想，但不是同一张图：MTP 是 `LLM_GRAPH_TYPE_DECODER_MTP` 的 `capacity_rows = verify_tokens / outputs = 1` 定义；TARGET 是 decoder 主干的 `capacity_rows = verify_tokens / capacity_outputs = verify_tokens` 定义。行数变化走 frame，不走新图。
- **复用键对应关系。** 现行 `llama-graph.h:1447-1481`：
  - `predefined_enabled` 下要求 `capacity_rows/outputs` 相等且 `active <= capacity`（`1451-1465`）——TARGET 直接沿用，无需改语义，只需把 TARGET 的 `capacity_rows/outputs` 从精确值抬到 `verify_tokens`。
  - `predefined_mtp_dynamic`（`1467-1469`）目前仅豁免 MTP 的 `n_tokens/n_seq_tokens` 与 `n_outputs`/sampler 输出比对（`1473-1474/1501-1509`）。TARGET **不得**复用该豁免字面条件（MTP 的输出语义是“至多一行”，TARGET 是“outputs == tokens”），而应新增一个对称的 TARGET 单定义键（命名待实施定，语义：`gtype == DECODER`（非 MTP）`&& phase == TARGET && capacity == verify_tokens` 时允许 `n_tokens/active` 可变、容量相等即复用）。本文只定语义，不定名字与补丁。
- **phase 豁免不扩展。** MTP 的 phase-0 豁免（`llama-context.cpp:2543-2553`）是 MTP 单定义专属；TARGET 验证仍走 `GGML_PREDEFINED_TARGET` phase，不得借用 MTP 豁免，也不得把 DEFAULT/prefill 路径卷入。

## 5. 风险与开放问题

1. **QSA `full_coverage` 与 pooled cache。** 验证行变化时是否保持同一拓扑、同一复用键。风险：拓扑随行数分裂则单定义破裂。实施时须逐个确认，必要时把覆盖模式纳入键或显式拒绝非常态组合（fail-closed）。
2. **PLE n-gram 行数。** PLE 哈希/投影在 host 侧按实际行计算（`build_inp_ple/build_ple` 路径），容量行下的行数与 gather 方式尚无对应改造。风险：PLE 层成为漏网的精确行假设。须在区 1/2 内显式收敛或声明 PLE 组合暂不支持。
3. **`hybrid_idx`。** 多行验证下单序列假设（`equal_seqs`、indexer 与 attention cell 一一对应，`qwen4exp.cpp:448-452`）是否仍成立。风险：indexer 状态写入行数错位。须在区 2 内按 active 收敛并真机验证。
4. **flash mask 越界。** flash 内核内部行循环是否真正服从 mask/行截断而不读 `[active, capacity)`。风险：静态 mask 形状对了但内核仍越界。必须真机证据（fault + 位级），不得仅凭形状断言。
5. **`predefined_complete` 门槛。** 覆盖口径“今天只比较派发数”（`ggml-vulkan-tp5-coverage.h:7`）。风险：把“数够了”当成“分类全了”，放行未间接化的算子。门槛必须是在所有参与算子完成 indirect 分类后才置位；之前一律 fail-closed。

## 6. 需要 Manager 再拍板的事项

1. **TARGET 单定义复用键的命名与豁免条件**（§4 第 2 点）：是否新增与 `predefined_mtp_dynamic` 对称的 TARGET 键，精确的 `gtype/phase/capacity` 组合写成什么。若 Manager 决定复用同一布尔名，需明确 MTP 输出语义差异如何隔离。
2. **`verify_tokens` 取值冻结**：MTP 会话的 `verify_tokens` 一旦作为 TARGET 容量，即不可在会话中途改变；改变即新定义。是否接受“会话级不可变”作为硬规则。
3. **PLE/QSA 非常态组合的处置**：若实施中发现某 PLE 或 QSA 覆盖组合无法在单定义下收敛，是 fail-closed 拒绝该组合（推荐），还是扩大本期范围。需 Manager 预授权 fail-closed 策略，避免实施中自行扩大 scope。
4. **真机证据门槛**：§3.2 的每一项是否均为合入前置，还是允许分批合入（例如先合区 0-1 的 CPU 契约部分）。本文推荐：任何允许 `active < capacity` 执行的合入，必须同时具备 §3.2 的 RELAY/状态一致性证据；纯 CPU 契约合入不得打开 `capacity_rows = verify_tokens` 的 TARGET 开关。

## 附：关键源码索引（评审定位用，不复制规范）

- 容量准入与 frame 发布：`src/llama-context.cpp:2422-2535`；TARGET 识别：`2477-2488`；MTP 容量：`2507-2513`；TARGET 精确现状：`2514-2519`；phase 豁免：`2543-2553`。
- frame/容量 ABI：`ggml/include/ggml-predefined.h:12-17/44-84/96-113`；TARGET 行数校验：`ggml/src/ggml-predefined.cpp:90-116`。
- 图容量行/复用键：`src/llama-graph.h:1447-1481/1726-1729`；图构造：`src/llama-graph.cpp:2692-2697/3614-3619/3681/3712/3745-3749`；`out_ids::set_input:234-286`；KV 容量分支：`4101-4129`；flash：`3880-3906`。
- qwen4exp 主干：`src/models/qwen4exp.cpp:429-562`（trunk/gather/result）、`574-619`（MTP 头容量范式 `601-610`）、`760-768/884-942/1199-1230/1275-1328/1429-1443/1601-1650`。
- Vulkan/RELAY：`ggml-vulkan.cpp:10150-10174/27488-27544/27546-27598`；`ggml-vulkan-collective.cpp:6310-6331/6433-6440/6796-6818`；`ggml-vulkan-tp5-rows.h:44-119`；shader 截断：`tp5_hc_resume.comp:113-257`、`tp5_add_rows.comp:16-50`、`tp5_relay_copy_f32.comp:21-83`。
- 设计主文档：`TP5.md`、`RERoT.md`、`docs/TP5-MTP-PREDEFINED.md §10`。

## 7. 当前进展与剩余闭合

### 7.1 区域进展与提交状态
- **输入 / 位置壳（区 0）**：`09d1a8d64`。`llm_graph_input_embd` / `pos` / `out_ids` 容量尺寸分配、有效前缀写入与无效后缀置零完成，超出抛错。
- **Attention + KV 尾部安全（区 1）**：`62409ebff`。`qwen4exp` 注意力 Q/K/V/gate 及 QKVZ 容量形状展开；KV 缓存尾部 `[active, capacity)` 重定向至空闲安全槽位；QSA fail-closed 准入门禁接入。
- **GDN CPU 方案 B（区 2 CPU）**：`14d9b321a`。`active_tokens` 注入 `op_params[2]`（`op_params[1] == 0` 避开冲突），CPU 时间维按 `effective_active` 截断且真实前向通过脏数据隔离。
- **MoE 区域（区 3）**：`e5fd879ed`。盘点与第 13 组测试确认生产代码无需修改（算子形状天然由输入行维驱动，逐行独立无跨行归约，脏数据逐比特隔离）。
- **Terminal Producer 区域（区 4）**：`354b5b1f8`。盘点与第 14 组测试确认生产代码无需修改（`inp_out_ids` 容量分配、D2H 活跃截断、采样器活跃行消费、LateBind 行程序截断天然就绪）。
- **RELAY / 消费者通道**：代码盘点确认天然以活跃 payload 元素消费，无需额外修改。

### 7.2 GDN GPU 侧状态与单卡验证事实
- **着色器与管线接线**：多步变长内核 `qwen4_gdn_multistep_delta.comp`（`e09beffc7`）、`prep`/`norm`（`a74a03725`）、管线句柄创建与基于 `op_params[2]` 的容量分派接线、以及 `ggml_vk_can_fuse_gdn_segment` 容量匹配放宽（`5407c211b`）已完成结构闭环。
- **单卡真机验证通过（最新事实）**：单卡 GDN 多步容量融合已在真实 RX 6800 上通过单算子密闭回归测试 `tests/test-vulkan-gdn-multistep.cpp`（提交 `e508881d7`）：
  - 覆盖范围：活跃步 $A \in \{1, 2, 4\}$、容量 $C = 4$；
  - 核心断言证据：
    1. 尾部脏数据隔离：变异无效输入步 $[A, C)$，GPU 最终循环状态位级一致（bit-identical）；
    2. 数值精度：活跃步输出与最终循环状态和 CPU 参考实现严格对齐（parity 残差约 $10^{-6}$ 量级）；
    3. 融合派发命中：执行日志确认捕获 `GDN_SEGMENT` 融合派发，段耗时约为 82–99 $\mu$s/段；
    4. 单步传统路径回归 100% 通过。
  - 修复链提交支撑（按序）：
    - `bff0c009b`：Q/K/V 交织视图 matcher 约束放宽；
    - `3d3822c90`：循环状态切片偏移修正为 `n_time * head_values`；
    - `d6579d1bb`：prep 卷积输入布局对齐；
    - `a2638e056`：segment_project 多行分派支持；
    - `2d0b67c15`：norm 树状规约与 gamma 头偏移修复。
- **生产准入护栏仍维持**：尽管单卡算子级验证全通，生产准入函数 `evaluate_target_capacity_admission`（`src/llama-context.cpp:2432-2440`）对 GDN/recurrent 模型依然保持严格的 **fail-closed** 拦截；解禁的前置条件保持不变。

### 7.3 剩余闭合前置
1. **多卡与模型级一致性真机验证**：在 5 卡 TP5 / 完整模型端到端上下文下验证多步 GDN 执行、`dmesg` 零新增 GPUVM fault / timeout，以及跨卡 AllReduce 后的状态连续性。
2. **解除 GDN fail-closed 决策**：在多卡/全模型级实测证据确凿后，方可由负责人裁决解除 `evaluate_target_capacity_admission` 中的循环状态 fail-closed 门禁。
3. **容量路径启用的端到端证据**：端到端吞吐测速与端到端文本生成正确性校验（须在真机获批前提下执行）。

### 7.4 保持不匹配的未闭合回退项
以下场景在图匹配与准入层显式保持不匹配或 fail-closed 回退至精确单定义 / 传统路径：
- 多序列场景（`n_seqs != 1 || n_seqs_unq != 1`）；
- 容量超界（`capacity_rows > 8`）；
- 缓存段变体（`fused_gdn_cached_segment`）；
- RBB 模式（`op_params[1] != 0`）；
- Chunking 分块循环路径。

### 7.5 M4 首步：LateBind 内核运行时有效行（已合入）

**问题（M3 不变量 2 的现存违反）**：LateBind 全部内核（inject / ACT_Q8 / Q / Q8DOT /
norm / LO / UP_Q8DOT）的 push constant 把 `late.capacity_rows` 当作执行行数烘焙，
1 行 MTP draft 在容量 4 定义下执行 4 行的 late 工作——CPU 侧 payload/sidecar 归约
早已按 active 计数，GPU 侧却按容量付费。

**修复（单一运行时行源，零重录）**：

1. **RELAY 广播 header word 5 作为运行时有效行的唯一发布点。** CPU 在提交前写入
   （chain 路径在 `tp5_relay_submit_epoch_chain` 首个 bank；后续 bank 在 pre-arm
   `tp5_relay_arm_bank` 传递信用验证后写入；one-shot 路径提交前写入，无帧则写 0）。
   Word 5 位于 64 字节 header 内，与 generation（word 0/2）、counter（word 1）、
   n_elems（word 3）互不重叠，relay probe 的 word 4 也不受影响。
2. **全部 7 个 late 内核改为读 word 5 截断 token 循环**，push constant 的容量仅作
   word 5 == 0 时的回退（非预定义 legacy 路径行为不变）。Q8DOT 经既有 QMailbox
   binding 读 `qmail[5]`；norm/LO 经既有 Inbox binding 读 `inbox[5]`；
   inject/ACT_Q8/Q/UP 新增一个只读 RowsHeader binding（绑定号各自追加，DSL 计数
   3→4 / 5→6 / 6→7 / 4→5）。
3. **inject 描述符改为每 bank**（原为每 rank 单份，因新增 bcast 依赖必须随 bank 切换），
   `late_inject_ds` 尺寸与释放路径同步改为 `n_ranks * BANKS`。

**写入时机与安全**：chain 首个 bank 在 `ensure_armed_epoch` 验证空闲后写入；后续
bank 在 pre-arm 传递信用（上一读者完成）后、且下一消费者必须等到 generation 发布
才能通过门铃等待的窗口内写入。无新增自旋、无 `vkDeviceWaitIdle`、无动态上限重试。

**验证**：`test-tp5-plan` 新增 `test_tp5_latebind_runtime_rows_protocol`（word 5
位置/发布值/legacy 回退/1→4→1→2 序列）；15/15 ctest 全绿（含 5 卡 mesh
RELAY/STAR 96 轮 + 变异输入 + 延迟生产者）；GPU 空闲、内核零新增错误。
relay probe 的 `consumer failed` 为基线既有行为（stash 前后一致），非本轮回归。

**仍未闭合**：latebind GPU 路径的端到端真机验证需要真实模型（HC 图 + MTP 会话），
须按 AGENTS.md 安全门获批后进行；本轮仅覆盖传输层回归与 CPU 协议测试。

### 7.6 P1-A 对照器具修复：sidecar 格式跟随 Q8DOT 生产者（已合入）

**问题（三处，均静态可证）**：

1. **sidecar 格式键错位**：`late_sidecar_f16` 原以 `numerical_mode ==
   AGGRESSIVE_Q8` 为键，但 F16 sidecar 是 **Q8DOT 生产者的属性**——P1-A
   派发同一枚 Q8DOT（F16 pair 写 bcast VRAM + qctrl 发布）。错位后 P1-A
   拿到 F32/exact 消费者：handoff 轮询 `status[6]`（仅 exact-path publisher
   写入）必超时；即使等到也从 host_import F32 区读数据（错缓冲区）；
   LO_Q8 把 F16 packed word 按 F32 位解码（数值全错）。
2. **ACT_Q8 可见性 barrier 只盖 1 行**：分配为 `capacity_rows(4) × streams ×
   width`，shader 写 token < capacity_rows 行，barrier 却只覆盖
   `streams × width`（1 行）——token 2..4 的写入对 Q8DOT 无序。LO_Q8→
   UP_Q8DOT 的 `lo_q8_ready` 同病（`late_rank` vs `4 × late_rank`）。
3. **P1-A 链上 norm_ready barrier 未发射**：`BARRIER_NORM_ACT` 只是验证器
   标签；实际命令（`mb_norm_lo` 全局 barrier，位于 p2[norm_end..lo_begin)）
   在 P1-A 发射序列中被跳过——ACT_Q8 可能在 norm 的 sum_output 写入可见
   前读 local_z（同一 trefs 缓冲区）。

**修复**（单文件 `ggml-vulkan-collective.cpp`）：

- `plan.late_sidecar_f16 = plan.late_q8_fast`（格式跟随 Q8 生产者路径）；
- ACT_Q8/LO_Q8 barrier 尺寸改为 `capacity_rows × …`（与分配一致）；
- P1-A 发射从 `emit(p2, 0, norm_end)` 改为 `emit(p2, 0, lo_begin)`，把
  norm_ready barrier 带进命令流；split 校验同步加 `lo_begin` 边界。

**P1-A 发射结构确认**（调研结论）：LO_Q8/UP_Q8DOT 不在 P1-A 段内发射，而是
作为下一 stage 的 `emit(incoming.p2, lo_begin)` 尾段——顺序契约
（norm→ACT_Q8→Q8DOT→LO_Q8→UP_Q8DOT）跨 stage 边界成立，与验证器一致。

**验证**：`test-tp5-plan` 新增 `test_tp5_sidecar_format_keying`（Q8 生产者
⟹ F16 sidecar + qctrl ready 轮询，对 AGGRESSIVE_Q8 与 P1A 双模式断言）；
15/15 ctest 全绿；GPU 空闲、零新增内核错误。P1-A 端到端真机收益测量
仍需模型会话（安全门未批），本轮修复的是对照器具的可信性前提。

### 7.7 P1-B 首步：Q8 权重定义期打包布局（已合入）

**问题**：Q8DOT/UP_Q8DOT 的内层点积循环每 k 迭代对权重块执行两次
`pack_q8_pair`（i16→i32 位拼接，约 6–8 条 ALU），而 `dotPacked4x8EXT` 本体仅
1–2 条 VALU。权重流量（W_down 3.5MB + W_up 3.5MB/rank/token，是 activation 的
320 倍）使该打包开销在每 token 重复支付。

**改动**（路线图 P1-B 首步，位级等价重排，无需精度校准）：

1. **新布局 `late_q8_packed`**：`float d; uint qs_words[8]` = 36B/块（vs
   `late_q8_0` 34B）。块数与索引不变（仅元素尺寸变）；索引 0 保留为持久
   done 标志，数据块 i 位于索引 i+1。
2. **一次性转换 shader `TP5_LATE_PACK_WEIGHTS`**：读 `late_q8_0` 写 packed；
   首次执行后由 done 标志自禁用（权重为不可变模型数据，plan 持有目标缓冲
   全生命周期）。
3. **发射位置**：pack dispatch 记录在 `cmd_late_pre` 头部（`mb_in` 后、
   inject 前），经 pre tape 进入线性链——每 stage 重放均为单次 guard 读空转。
4. **标志初始化**：plan 创建期（冷路径）以专用 fill CB + fence wait 将两块
   packed 缓冲的前 4 字节清零——防未初始化设备内存或先前 plan 释放的缓冲
   别名导致 flag==1 而跳过本次 pack（读到旧权重）。
5. **描述符**：Q8DOT 绑定 0 与 UP_Q8DOT 绑定 0 改接 packed 缓冲；
   activation 侧维持 `late_q8_0`（ACT_Q8 每块写一次，打包成本被全部输出行摊销）。

**验证**：`test-tp5-plan` 新增 `test_tp5_packed_weight_layout`（36B/块、W_down/W_up
共享块数 102400、边界、flag 索引不与数据重叠）；全套 all passed；15/15 ctest
全绿（含 5 卡 mesh RELAY/STAR）；GPU 空闲、零新增内核错误。**真机收益测量
（内层循环 ALU 消除 vs +6% 显存）需模型会话，安全门未批**。

### 7.8 RADV codegen 证据通道修复与首次实测（已合入）

**两处器具修复**（路线图 §4.3 “看实际 RADV codegen，而不是 shader 名字”的前提）：

1. **主路径 `GGML_VK_PIPELINE_STATS` 一直无效**：`pep_features` 值初始化后
   `pipelineExecutableInfo` 从未被置 `VK_TRUE`——扩展启用了但 feature 未开，
   `vkGetPipelineExecutableStatisticsKHR` 按规范属无效调用。修复后主路径
   实测出 `mul_mat_vec_f32_f32_f32`：VGPR 32 / SGPR 108 / 零 spill。
2. **TP5 collective kernel 无统计通道**：`make_late` 工厂不捕获统计。新增
   `GGML_TP5_PIPELINE_STATS`（过滤子串，空串全匹配）＋ `tp5_print_pipeline_statistics`，
   pipeline 以 `CAPTURE_STATISTICS` 位创建；caps 新增
   `pipeline_executable_properties` 探测（`ggml_vk_tp5_device_caps`）。

**首次 RADV 实测**（RX 6800，`--sync relay --rounds 1`，5 卡 mesh 创建路径）：

| kernel | VGPR | SGPR | spill S/V | LDS | code size |
|---|---|---|---|---|---|
| tp5_hc_late_inject | 32 | 108 | 0/0 | 1024 | 1144 |
| tp5_hc_late_q | 64 | 108 | 0/0 | 1024 | 2316 |
| tp5_hc_publish | 8 | 108 | 0/0 | 0 | 120 |
| tp5_hc_resume_norm | 32 | 108 | 0/0 | 1024 | 18920 |
| tp5_hc_resume_lo | 8 | 108 | 0/0 | 1024 | 1340 |
| tp5_hc_late_pack | 24 | 108 | 0/0 | 0 | 452 |
| tp5_hc_late_act_q8 | 64 | 108 | 0/0 | 2048 | 1952 |
| tp5_hc_late_q_q8dot | 64 | 108 | 0/0 | 5120 | 2676 |
| tp5_hc_resume_lo_q8 | 64 | 108 | 0/0 | 2048 | 3996 |
| tp5_hc_late_up_q8dot | 64 | 108 | 0/0 | 4096 | 6152 |

（SGPR 108 为 RADV 对全部 compute pipeline 的固定开销报告；spill S/V = Spilled SGPRs/VGPRs。）

**结论**：全部 latebind kernel 零 spill——P1-B 打包布局与现有几何在寄存器压力下
健康，§4.1 的几何搜索（row/wave/K tile 候选）不会先撞寄存器場。`resume_norm`
code size 18920 是最大者，是几何优化的下一候选。**注意**：此表来自 5 卡 mesh
创建路径（pipeline 创建期统计），不是模型会话吞吐证据。

### 7.9 P1-B 第二步：激活侧同布局打包（已合入）

**问题**：P1-B 首步只打包了权重；Q8DOT/UP_Q8DOT 的内层循环对 activation 侧
仍每 k 执行 `pack_q8_pair`（i16→i32 位拼接，2 ALU/4 元素）。activation 流量小
（10.9KB/token），但它在**每个消费者每次读时**重复支付。

**改动**（位级等价除 d 精度外，见下）：

1. `TP5_LATE_ACT_Q8`（tp5_hc_latebind.comp）与 `TP5_RESUME_LO_Q8`
   （tp5_hc_resume.comp）输出改 `late_q8_packed`（36B/块，**无 flag 块**：
   per-token 瞬态，每 epoch 重写，自禁用协议无意义；索引 i 即块 i）；
2. Q8DOT 绑定 1、UP_Q8DOT 绑定 1 改读 packed；内层循环变纯
   `dotPacked4x8EXT(w.qs_words[k], a.qs_words[k])`——零打包 ALU；
3. 主机侧 `tp5_late_q8_packed_bytes()` 统一四处字节计算（分配×2 + barrier×2）。

**数值变化（非回归，是精度提升）**：块尺度 d 从 f16（11 位尾数）拓宽为 f32
（24 位）。原路径 `float16_t(d)` 对尺度做舍入；packed 布局存全 f32。int8 载荷
不变，aggressive 路径的去量化误差严格缩小。

**RADV codegen 实测**（同 5 卡 mesh 创建路径）：

| kernel | code 前→后 | Δ |
|---|---|---|
| tp5_hc_late_q_q8dot | 2676 → 1696 | **−36.6%** |
| tp5_hc_late_up_q8dot | 6152 → 5044 | **−18.0%** |
| tp5_hc_late_act_q8 | 1952 → 2064 | +5.7% |
| tp5_hc_resume_lo_q8 | 3996 → 4000 | +0.1% |

两个点积消费者代码量大幅缩减；两个生产者开销微增。VGPR 全部维持 64、零 spill。
**代价**：activation 显存 +6%（36B vs 34B/块，46080B vs 43520B @max_rows=4）。
真机收益仍需模型会话（安全门未批）。

### 7.10 §4.1 几何第一步：Q8DOT 行归并 LDS 往返 → wave 内 shuffle 树（已合入）

**问题**：`late_q_q8dot` 每行 16 条归并 lane 恰是 wave32 的半波（rows_per_wg=16
over 256 线程 → 每 subgroup 恰 2 行），但旧实现走 LDS 往返：`row_partial[lid]`
写 → barrier → lane16==0 串行加 16 项 → `group_out` → barrier → 32 lane 发布。
两次 barrier＋一次 16 项串行加＋2KB LDS 全部只为发现 16 个已就绪的行和。

**改动**（tp5_hc_latebind.comp，单 kernel）：

1. 行内归并变 `subgroupShuffleXor` 掩码 1/2/4/8 的 4 步半波树——零 LDS、零 barrier；
2. 跨半波一次 `subgroupShuffleXor(acc, 16u)` 把偶行的和送到奇行 lane；
3. 发布变每 subgroup 前 4 lane（一 lane 一 stream）打包 F16 对——8 subgroup ×
   4 stream = 与旧版完全相同的 32 个 packed word，地址映射不变；
4. `row_partial`/`group_out` shared 数组删除，LDS 只剩 `publish_last` 一个字。

**RADV codegen 实测**：

| 指标 | 改前 | 改后 | Δ |
|---|---|---|---|
| code size | 1696 | **1248** | **−26.4%** |
| LDS | 5120B | **1024B** | **−80%** |
| VGPR / spill | 64 / 0 | 64 / 0 | 不变 |

**语义**：归并仍是 16 项的树状重结合（旧版是 16 项串行链）——aggressive 模式
本就声明重结合自由；发布顺序与 word 地址逐一保持。

### 7.11 两处小修（前轮交接遗留）

1. **packed 权重 range check**：`2u * max_storage_buffer_range` → 单 range——
   绑定 range 直接是 packed_bytes，无 split，双倍上限是错的宽松（3.7MB 从不
   触发，但检查应诚实）；
2. **pack clear fence**：`UINT64_MAX` → 2s 有界（与 relay handoff timeout 同
   惯例）——冷路径但绝不无界，挂死的 clear 必须 fail-closed。
