# TP5 工单 A：原 Q2_0 BF16 HC LateBind 源码改动与对拍方案

状态：源码改动已落盘，**未经编译与 GPU 验证**（Engineer 无执行权限；编译、对拍与跑分由 DevOps 执行）。
对应 AGENTS.md §8 工单 A（BF16_LATEBIND）。日期：2026-10-12。

## 1. A2：原 Q2_0 未命中 LateBind 的完整原因链

静态源码结论（`ggml_vk_hc_segment`，ggml-vulkan.cpp ~16742）：

1. **类型截断（根因）**：`capture_late` 原条件要求 `down_node->src[0]->type == GGML_TYPE_Q8_0` 且 up 同条件。原 Q2_0 的 HC down/up 权重为 BF16（AGENTS.md §1.2），在 gate 处被直接拒绝。
2. **capture 前提链**（均须成立，改动前后不变）：`first == 0 && fuse_inject && replay_recording && pipeline_hc_sum_f16 存在 && recorded_dispatches == 0 && 无 rms-partials && 单流单序列`；`GGML_TP5_LATEBIND=hc-down|1|on`；`late_rows_safe`（active/capacity rows ≤ VK_TP5_DIRECT_COLUMN_TILE）；`lo/normalized/mixed 为 F32`；`normalized 宽度 = 4×norm 宽度`；`late_rank ∈ (0, 512]`；streams ∈ [1,4]。
3. **下游分派缺失**：即便 gate 放行，原 collective 侧 `tp5_resolve_numerical_mode` 无 BF16 权重分支；`tp5_hc_late_q` 的 Q contraction kernel 只实现 Q8_0 block 解码（`late_q8_0` 结构、34B/32elem），BF16 字节会被误解析成 Q8 块——不是"慢"，是"错"。
4. **fused segment 侧同样缺失 BF16 变体选择**：`hc_bf16_weights` 之前不存在，fused down_silu/project_fold 只会选 Q8DOT/WG 管线。

## 2. A3：选型结论——A-native（原生 BF16 直接参与提前计算）

**选 A-native，弃 A-convert。理由：**

| 维度 | A-native（BF16 直接读） | A-convert（转 Q8_0 后融合） |
|---|---|---|
| 数值语义 | BF16 位展开精确（`u<<16`），权重零转换误差；与 reference 路径读同一份字节 | 引入量化误差，原 Q2_0 变成近似 target，需走 A_quality 质量门，丧失严格对拍资格 |
| 代码结构 | 仓库已有 `pipeline_hc_down_silu_bf16/project_fold_bf16` 与 shader `TP5_LATE_Q_BF16` 变体骨架；只需 gate 放行 + 分派接线 | 需要权重转换管线（类似 `tp5-hc-q8-variant.py` 的运行时版），改动面大且产生第二份权重存储 |
| AGENTS.md 红线 | 不改 target 数值身份 | 混淆"原 Q2_0"与"HC-Q8 衍生"两条数值路径（§1.2 明令不得拼账） |
| 带宽代价 | BF16 2B/elem > Q8_0 ~1.06B/elem，Q contraction 读带宽约 2× | 带宽更省 |

带宽劣势接受理由：工单 A 的目标是**打通原 Q2_0 的严格数值 LateBind 路径**（先正确）；Q8 化的近似加速属于 HC-Q8 衍生 target 的既有账目，不在本工单混淆。若后续实测 BF16 Q contraction 成为瓶颈，可在近似资格下另行立项（A_convert 工单），与本工单解耦。

## 3. 源码改动清单

全部改动已落盘；**Q8_0 路径行为不变**（HC-Q8 衍生模型既有资格不受影响），理由见各条。

### 3.1 ggml-vulkan.cpp

- **`ggml_vk_hc_segment` capture gate（~16768-16796）**：
  - down/up 类型条件放宽为 `Q8_0 || (BF16 && 对应 bf16 pipeline 存在)`；
  - 新增两条族守卫：(a) `down_node->src[0]->type == up_node->src[0]->type`（down/up 必须同类型，防止混合族把 BF16 kernel 绑到 Q8_0 字节上）；(b) `inject->src[0]->type == GGML_TYPE_Q8_0`（late INJECT kernel 只实现 Q8_0 解码；原 Q2_0 的 inject 权重按静态证据为 Q8_0，非 Q8_0 时保持 NO_HIT 而非误解析，BF16 INJECT 变体是后续独立工单）。
  - Q8_0 权重时所有新增条件均为恒真或与原条件等价 → 逐位行为不变。
- **fused 段 pipeline 选择（~16812-16822）**：`hc_bf16_weights`（down 为 BF16）时 down/up pipeline 选 `pipeline_hc_down_silu_bf16` / `pipeline_hc_project_fold_bf16`（声明 ~1087-1088，创建 ~6672-6677，与 native 版同 binding 数/同 push constants，无条件创建）。族守卫保证 up 同为 BF16。
- pipeline 声明与创建为仓库既有资产（本次工单接通选择逻辑）。

### 3.2 ggml-vulkan-collective.cpp（最小改动，理由与方案）

任务边界允许在"必须改 collective 才能成事"时做最小改动。必须改的理由：LateBind 的 Q contraction dispatch 与 numerical mode 解析都在 collective 侧，gate 放行后若 collective 不接 BF16，BF16 权重会被 Q8 kernel 误解析（静默数值垃圾）。最小改动四点：

1. **`tp5_numerical_spec::bf16_late_weights` 字段（~144）**：BF16 late 权重强制 `EXACT_F32` 数值模式（~167-172）——Q8 激活量化/Q8DOT 整数收缩/packed repack 均假设 Q8_0 权重块，BF16 下全部不适用。
2. **BF16 检测（~4175-4177）**：由记录的 `down_weight.size == late_rank × streams × width × 2`（2B/elem）判定；2B/elem 与 34B/32elem 的字节数不可能相等，判据无歧义。
3. **LB2/RHO fail-closed（~4178-4188）**：`bf16_late_weights && GGML_TP5_LATEBIND_RHO` 时 `c.fail(...)` 拒绝建 plan——rho 模式的 Q-stats/LO_U kernel 解析 Q8_0 块，BF16 下是结构性错乱；fail-closed 让失败可见而非隐藏（红线 8）。该组合在改动前本身就是错误解析，不存在"既有正确行为"被破坏。
4. **Q contraction 分派（~5238-5239）**：`late_q_selected = (bf16_late_weights && r.late_q_pipe_bf16) ? r.late_q_pipe_bf16 : r.late_q_pipe`。`late_q_pipe_bf16` 声明 ~564、创建 ~2183-2185（`make_late(7, 20, tp5_hc_late_q_bf16_*, r.late_q_dsl, r.late_q_layout, ...)`，复用 Q8 版同一 descriptor set/layout——binding 数与语义一致，仅 binding 0 的权重元素类型不同）、销毁 ~2789。rho_mode 分支不受影响（BF16+rho 已在 plan 期拒绝）。

### 3.3 vulkan-shaders/tp5_hc_latebind.comp（仓库既有 BF16 变体，本次核实）

- `TP5_LATE_Q_BF16`（~17-23）：`tp5_bf16_to_fp32(u) = uintBitsToFloat(u << 16)`，精确位展开。
- binding 0 权重布局（~727-739）：`uint16_t w[]`（2B/elem、无 block scale）vs Q8 版 `late_q8_0 w[]`。
- Q contraction 解码（~825-841）：8 个连续 u16 展开，`dot(b,v)` FMA 链与 per-lane/subgroup 归约顺序对齐 Q8 路径；正确省去 Q8 版的 `w[ib].d` block-scale 乘法。
- `vulkan-shaders-gen.cpp` ~994 已注册 `TP5_LATE_Q_BF16` 变体。

### 3.4 未改动

qwen4exp.cpp 的 `build_hc_mix/build_hc_combine`：图结构无需类型分派（gate 在 vk 层按张量实际类型判断）；同步/内存语义（DEVICE_UNCACHED、coherent、barrier、epoch/bank/reader-done）零改动；未删任何失败路径。

## 4. 单 site 对拍方案（供 DevOps 执行）

### 4.0 前置

1. 编译（`build-perf-gap` 或等价配置），确认 `tp5_hc_late_q_bf16` pipeline 创建成功（无 SPIRV 编译错误）。
2. 模型身份：原 Q2_0 两分片 SHA256 按 AGENTS.md §1.1 核对。
3. 全程 `GGML_TP5_LATEBIND_RHO` 不设（BF16+RHO 是 fail-closed 拒绝，属预期失败，单独立臂记录）。

### 4.1 步骤一：NO_HIT 基线（先证 gate 行为）

原 Q2_0、`GGML_TP5_LATEBIND=hc-down`，跑 1 个短请求（如 `--max-tokens 8` 的 teacher-forced 回放或自然请求）。检查 server.log：

- late site 计数 > 0（现有 `late_sites`/capture 统计日志）→ gate 放行成功，进 4.2。
- site = 0 → **NO_HIT**，按 §5 分类记录原因，本工单在该模型上记失败/待查，不得宣称提速。

同时跑 HC-Q8 衍生模型同配置，确认其 late site 数与改动前基线一致（Q8_0 路径不变性）。

### 4.2 步骤二：单 site 中间量对拍（teacher-forced 固定 token 回放）

同一二进制、同一模型，两臂仅差 `GGML_TP5_LATEBIND`（off vs hc-down）：

1. **reference 臂**：LateBind off。原 Q2_0 的 fused segment 走 `pipeline_hc_down_silu_bf16`（本改动接通的 fused BF16 变体）→ 这同时验证了 fused BF16 变体自身与旧 Q8DOT 路径的关系：若需隔离，可加第三臂（改动前二进制 + off）确认 fused BF16 与旧路径的中间量一致性。
2. **candidate 臂**：`GGML_TP5_LATEBIND=hc-down`。
3. 固定 prompt + teacher-forced 回放固定 token 序列（temperature=0、seed=42、`enable_thinking=false`、`cache_prompt=false`），逐 site dump 下列观测量（用现有 profile/插桩入口；缺失的量标 `NOT_COLLECTED`，不得事后补造）：

| 量 | 位置 | 判定 |
|---|---|---|
| X（residual+s·Y 后的 HC 输入） | resume 后 | 代数等价（浮点重排） |
| RMS 统计 rho_b | resume | 代数等价 |
| gamma·X（G） | fused/Q kernel 输入 | 代数等价 |
| down 中间 d[k]（Σ_b rho·W_down·G） | Q 部分和相加处 | 代数等价 |
| SiLU(u) | fused | 逐位应一致（同 kernel） |
| up gate | fused | 逐位应一致（同 kernel） |
| inject 权重输出 | late INJECT | 代数等价 |
| fold/mixed 输出 | fused | 代数等价 |
| 下一次 residual | 层出口 | 代数等价 |
| recurrent state / KV | 层间 | 代数等价 |
| **全词表 logits** | 每步 | 见下 |
| **采样 token 序列** | 全序列 | 见下 |

**判定**：LateBind 的本质是把 `W_down·gamma·(R+s·z)` 的线性部分提前到跨 rank 求和之前——数学恒等，但浮点归约顺序改变。因此：
- logits **逐位一致** → 严格路径成立（`A_strict`），可进自然 EOS pilot。
- logits 不一致但逐 token 相对误差 ≤ 预注册容差（建议首检 1e-3，量级参考 HC-Q8 LateBind 历史对拍）且 token 序列不发散 → **如实标注为近似路径**（浮点重排类），需 Manager 裁定按 `A_quality` 走质量门（AGENTS.md §2 两集合不得偷换）后方可进性能排名。
- token 序列发散或误差超容差 → 停止，保留证据，回机制排查（不跑大规模排名）。

单 site（首层首个 HC site）先做，通过后扩全层、再扩 h0/h1 行数。

### 4.3 步骤三：无插桩完整请求

对拍合格后，关 profile，按 AGENTS.md §5 runner 跑自然 EOS pilot（160-240 token 契约、质量门），再决定是否进 §14 配对块。**插桩成绩只进诊断账。**

## 5. NO_HIT 判定标准

开启 `GGML_TP5_LATEBIND=hc-down` 后，若 server.log 中 late site 计数为 0（或 collective plan 显示 `REFERENCE`/`NO_LATE_TENSORS`），即 **NO_HIT**：只记录"环境变量改变而 site=0"，不宣称 LateBind 提速、不把该配置计入性能样本。按 gate 条件分类原因：

| 拒绝原因 | 日志/判定特征 |
|---|---|
| inject 权重非 Q8_0 | 若实测 `hc_attn_inject` 为 BF16 → 本工单边界，需 BF16 INJECT 后续工单 |
| down/up 族不一致 | 类型混排模型 |
| bf16 pipeline 未创建 | pipeline 创建失败/设备不支持 |
| late_rows 超限 | active/capacity > VK_TP5_DIRECT_COLUMN_TILE |
| capture 前提不满足 | replay_recording/首层/fuse_inject 等结构条件 |
| BF16+RHO | fail-closed 显式报错（预期失败，非 NO_HIT，单独立臂） |

## 6. 风险与需 Manager 裁定的边界

1. **inject 权重类型未实测**：静态证据（fused FUSE_INJECT kernel 现状解析 Q8_0 且原 Q2_0 基线输出正常 + AGENTS.md"down/up 主要为 BF16"措辞）推断 inject 为 Q8_0；若 DevOps 实测为 BF16，gate 安全拒绝（NO_HIT），需立 BF16 INJECT 工单。
2. **近似路径风险**：浮点重排使 logits 可能非逐位一致；若对拍显示差异，本工单交付的是"近似资格候选"，晋级与否及质量门归 Manager。
3. **collective.cpp 改动**：超出任务默认写入区（任务预授权"必须改时说明理由与最小方案"）；实际改动为字段+检测+fail-closed+四处分派接线，无协议/同步语义改动。
4. **编译与运行验证全部缺失**：本工单只交付源码与方案；`NOT_RUN` 项见 §4。
