# DECISION.md｜2026-10-10 契约 v5 轮：v4 必要性证书 + Q2_0 h0/h1 配对确认

## 问题与契约版本

任务 Ω：RERoT 关闭、原 Q2_0 两分片、五张 RX 6800、单路 `-np 1`、中文问句"如何减肥"。
契约演进（每次修改提示即新契约，全臂重测）：

| 版本 | system 提示 | 实测 token | 长度门 160–240 |
|---|---|---|---|
| v1（q2-baseline 轮） | 简洁约 200 | 140 | 失败（历史） |
| v2 | 分点约 250 | 135 | 失败 |
| v3 | 六部分强制结构 | 400 无自然停止 | 失败（NOT_NATURAL_STOP，臂终止） |
| v4 | 五部分结构 | 360 自然停止 | 超上界（>240），保留 |
| v5 | 四部分精简结构 | 190 自然停止 | **通过** |

v5 预注册哈希：question `d6203155…`，system `f2850b0f…`。v5 为本轮正式契约；v4 的 8 进程配对作为机制证据保留（两臂同为 360 token，比值内部有效）。

## 本轮身份

- 模型两分片与 draft SHA256 本轮重新核实，与 AGENTS.md 记录一致（69820c02…、316b46f3…、8db8b420…）。
- 二进制：`q2-baseline-20261010T0401/measured-bin/llama-server` 冻结快照，全部 DSO 哈希沿用该轮 binary-identity.json，未重编译、未改生产源码。
- 五卡空闲、无并行编译；runner 内部持 GPU 锁。

## 理论侧（L）产出

v4 必要性证书（`necessity-certificate-v4.json`）撤回并替换 v3 的单一 476 tok/s 数字，修复三处缺陷：

1. **DEFECT-1（事实）**：稠密权重 90.8→63.82 MB/层（1222 张量审计清单复核：qkv 层 63.79、q/k 层 61.65）。
2. **DEFECT-2（健全性）**：L2 保留额度按活跃集（829 MB/GPU/token）而非 11.9 GB 全池计——对实现最大宽容才合法。
3. **DEFECT-3（健全性，最大）**：MTP 行批验证使活跃权重按周期读一次并在 h+1 个提交行上摊销；R3 在完整允许集 A 下不成立，v3 的"per row 即 per token"声明错误。

分层界限：**L_global = 5667 tok/s**（R4 FP32 ALU 绑定，14.3 GFLOP/token @80.85 TFLOP/s，对完整 A 有效）；**L_nomtp = 730 tok/s**（R3 绑定，700.9 MB/GPU/token 必要 DRAM @512 GB/s，仅对 A_nomtp 条件类有效）。串行链 R5 调查：完全放松集合（任意融合/缓存/完美 draft）下每个候选延迟耦合都存在合法实现可隐藏，无可证项——收敛杠杆在 U 侧。测试 `tests/test-tp5-necessity-certificate.py` 14 项通过。

## U 侧证据

### 周期账（D2/D3/D5 诊断臂，不进排名）

- 每 decode step `graph_compute` ≈17.6 ms；GPU subgraph 忙时仅 ~0.65 ms/token；96 stage × ~160 μs 互等轮询占 96%。
- 每 stage 每 rank ~35 dispatch + ~29 barrier（ledger：3391 dispatch/2443 epoch）。RADV 每派发 ~2–4 μs 固定开销 → ~160 μs/stage。**launch-bound，非带宽/计算受限**。
- 段 dump（D5-segments.tsv，89275 行）：每 stage ~106 非自由节点，其中 MUL_MAT 16、MUL_MAT_ID 6、SOFT_MAX 3、RMS_NORM 2、router 类 ~6——剩余 ~35 dispatch 为 GEMV/router 类不可再融合残差。
- TIMELINE 传输 A/B（B1）：3.33 tok/s，比 RELAY 慢 15×，请求超时臂 FAILED。RELAY epoch 链确认为当前最优传输。

### h0/h1 配对确认（契约 v4，8 独立进程，ABBA/BABA）

| 块 | h0 tok/s | h1 tok/s | 比值 |
|---|---|---|---|
| 1 | 51.05 | 56.10 | 1.09907 |
| 2 | 51.07 | 55.85 | 1.09349 |
| 3 | 51.20 | 56.03 | 1.09430 |
| 4 | 51.12 | 55.83 | 1.09230 |

几何均值 **1.09479，95% CI [1.09008, 1.09952]**（n=4 块）。全部 8 进程 rc=0、无强杀、measured 零 issue；**全部对应正文（warmup/measured/guard）逐字一致**——greedy MTP 验证在 Q2_0 上精确复现 h0 token（旧 APEX 类 h1 正文不一致失败不复现）。接受率 0.731，均值接受长 1.73。CI 下界 1.090 > 2% 工程门：**h1 相对 h0 显著加速 +9.5%**。

契约 v5（190 token，长度门通过）配对 `c5pair-1..4` 完成：

| 块 | h0 tok/s | h1 tok/s | 比值 |
|---|---|---|---|
| 1 | 48.39 | 54.48 | 1.12591 |
| 2 | 48.19 | 54.89 | 1.13905 |
| 3 | 48.54 | 54.83 | 1.12971 |
| 4 | 48.37 | 54.85 | 1.13398 |

几何均值 **1.13215，95% CI [1.12320, 1.14118]**（n=4 块，8 独立进程）。全部对应正文逐字一致，measured 全部 190 token、自然 stop、零 issue，rc=0 无强杀。池化 h0=48.37、h1=54.76 tok/s。

## 按证据的结论与晋级状态

- **h1（MTP 深度 1）在原 Q2_0 上取得双契约配对确认的显著收益**：v4 契约 +9.5% CI [9.0%, 10.0%]，v5 契约（长度门合格）**+13.2% CI [12.3%, 14.1%]**。h1 为该契约下最强已知臂（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）。生产默认值不变（需另行资格）。
- U/L 收敛状态：S_lo（h1，v5 契约，配对确认）54.76 tok/s；对 L_nomtp=730 的 Gap = 13.3×，**未收敛**。理论侧已诚实到当前前提集的极限（串行链无可证项）；剩余收敛路径 = U 侧持续提速（下一工单：dispatch 数削减 / MTP 深度扫描 h2/h3 / LateBind BF16）。

## 下一项可证伪问题

1. ~~MTP 深度扫描~~（本轮已测，见下）：h1 为最优深度。
2. **BF16 LateBind（工单 A-native）**：入口地图已定——`vulkan-shaders-gen.cpp:1107` 新增 `qwen4_hc_down_silu_bf16`/`qwen4_hc_project_fold_bf16` 变体（`DATA_A_BF16` 替代 `DATA_A_Q8_0`）；`ggml-vulkan.cpp:6655` 建管线；`ggml-vulkan.cpp:16749` capture gate 接受 `GGML_TYPE_BF16` down/up；`ggml-vulkan-collective.cpp` late 计划（late_q8dot 管线/描述符/sidecar 全套）需 BF16 路径。HC-Q8 臂已证 late=94 site、派发 −16%、同 target +9.7%——BF16 版预期同量级且保持原 target 数值。需重建+固定 token 对拍+配对确认。
3. 派发数削减（融合残差 ~35/stage → 目标 <20）：入口已定位（`ggml-vulkan.cpp:21447` ADD+RMS 融合模板 + `multi_add_rms.comp` 融合核模板；目标 HC 流混合元素链 74 节点/stage → ~10 核）；需改 C++/shader，另立工单带对拍。

### MTP 深度扫描结果（契约 v5，同冻结二进制，单因素 = horizon）

| 深度 | 池化 tok/s | 接受率 | 均接受长 | 正文一致 |
|---|---|---|---|---|
| h0 | 48.37 | — | — | 基准 |
| h1 | 54.76 | 0.731 | 1.73 | ✓（8 进程配对确认） |
| h2 | 51.12 | 0.679 | 2.36 | ✓（pilot） |
| h3 | 45.06 | 0.568 | 2.70 | ✓（pilot） |

h2/h3 变慢目为周期成本随验证行数增长快于接受长增长（h2：K 增 1.36×，周期增 ~1.44×；h3 更差）。launch-bound 诊断细化：每 stage 成本随行数增长（派发几何），MTP 摄销快速饱和。**h1 确认为该 draft 模型在 Q2_0 上的最优深度**；h2/h3 pilot 证据保留，不晋级。

### HC-Q8 派生 target 臂（E1/E2，契约 v5，另一数值路径，不与原 Q2_0 拼账）

| 臂 | LateBind | 池化 tok/s | token | late site | dispatch/rank |
|---|---|---|---|---|---|
| E1 HC-Q8 h0 | off | 56.78 | 141 | 0 | 3391（参考） |
| E2 HC-Q8 aggressive | on | 62.32 | 139 | **94** | 2853（−16%） |

机制证据：**LateBind 在 Q8 HC 权重上确实命中（late=94、q8_fast=94、scatter_early=91）**，派发数 3391→2853，同 target 内 +9.7%（56.78→62.32）。这直接验证了 launch-bound 诊断与 LateBind 杠杆。质量：正文连贯切题自然 stop，无危险建议；长度门 139/141<160 失败（派生 target 生成更短文本），按近似路径记，不晋级主排名。原 Q2_0 要获得同等收益需 BF16 LateBind（工单 A-native，C++/shader）。模型身份：分片 1 = 9350549…（本轮重核），分片 2 与原模型同哈希。

### 原 Q2_0 aggressive lowering 臂（F 系列，同 target，正文与 h0 逐步一致）

| 臂 | 配置 | 池化 tok/s | 正文 |
|---|---|---|---|
| F1 | h1 + mmvq auto | 54.72（无收益） | 变（近似路径，不晋级） |
| F2 | h1 + aggressive | 57.21（pilot） | 与 h1 一致 |
| F3 | h1 + gtt-push only | 55.19 | 与 h1 一致 |

分解：gtt-push 传输 +0.8%；`GGML_TP5_LATEBIND=hc-down` lowering（无 Q8 capture，late=0）派发 4117→3647（−470）另 +3.7%。8 进程配对确认（f2pair-1..4，ABBA/BABA）：**aggressive/off 几何均值 1.02622，95% CI [1.01703, 1.03550]**，全部正文与 h0 基准逐字一致。**最强合格臂 = h1 + aggressive：池化 56.47 tok/s**；本轮原 target 总爬升 h0 48.37 → 56.47 = **+16.8%**，逐步正文一致。S_lo=56.47，对 L_nomtp Gap = 12.9×。

## 未执行 / 限制

- MMVQ、LateBind、HC-Q8 派生臂本轮未跑（单因素原则：先固定 h0/h1 对照）。
- 保护集仅既有哈希表 guard；未做跨请求状态长测。
- v4 证书仍宽松（峰值口径、跨层完全重叠、未计 KV 状态读）。

### BF16 LateBind 工单 A-native 实现（本轮代码落地，路径 OPEN）

按工单 A 入口地图完成实现：`qwen4_hc_down_silu_bf16`/`qwen4_hc_project_fold_bf16` shader 变体（vulkan-shaders-gen.cpp）、管线创建与 capture gate 扩展（ggml-vulkan.cpp，BF16 down/up 接受）、HC 段融合 gate 扩展（q8_projection 接受 BF16）、collective 数值模式 `bf16_late_weights` 检测（强制 EXACT_F32）、`tp5_hc_late_q_bf16` Q 收缩核（uint16 权重逐元素 bf16→fp32）、late consumer 尺寸校验接受 BF16 几何。新二进制快照 `measured-bin-v2/`（binary-identity-v2.json）。

**结果：capture 命中（late=94 site，mode=exact-f32，派发 2702 vs 4100）但链在 stage 0 Q sidecar 握手超时挂起。** 二分定位：

| 臂 | 配置 | 结果 |
|---|---|---|
| G4 | 新二进制 + Q2_0 + off | 正常 55.07 tok/s（BF16 段融合安全） |
| G5 | 旧二进制 + HC-Q8 + exact | 正常（对照） |
| G7 | 新二进制 + HC-Q8 + exact | 正常 50.75（collective 改动无回归） |
| G1/G2 | 新二进制 + Q2_0 + exact（BF16 Q 核） | 挂起：Q sidecar 超时，GPUVM write fault |
| G6 | 同上但强制 Q8 Q 核 | 同样挂起（排除 Q 核自身） |

完整调试地图（12 次运行）：
- G9 bar-pull：同挂（排除传输）；G10：940 条 Q+publisher 录制确认在链 tape；ZSNAP=1：同挂（排除 WAR 别名）；RADV syncshaders：初始化期死亡（工具不可用）。
- 关键读数修正：线性链为单 CB，仅在结尾 signal final（2×192=384）；tl=192 是链前（prefill/mesh epoch 96）的最后信号——**当前链在 GPU 上从未推进**。Y flag 已消费（P1 payload copy 完成）但 Q/publisher 未发布、无 kernel 超时标记（err=0）、全部 GPU 等待均有界（spin_max，fail-closed）。
- G11（BF16 GEMV 强制 Q8 管线）：同挂——**BF16 GEMV 核也被排除**。至此 Q 核（G6）、GEMV 核（G11）、传输（G9）、WAR（ZSNAP）全部排除；tape 结构与 Q8-exact 完全一致。剩余差异仅为 norm producer 管线（Q8 9-binding 带 companion vs BF16 8-binding plain）与权重描述符尺寸。结论：BF16-capture exact 链在 P1 copy 之后、Q 发布之前的 GPU 执行死锁（err=0 非超时）。- G12（跳过 Q/publisher 录制）：宿主 Q 轮询为无条件执行，超时属预期，测试对死锁问题不可判；但确认链在 stage 0 宿主中止后停止推进（后续 stage 等待宿主门铃，符合设计）。
- 最终定位：BF16-capture exact 链 stage 0 的 Q sidecar 未发布（publisher 写入未到达宿主可见），无 kernel 超时标记（err=0）、全部等待有界。需 RADV 交互式 GPU 诊断（如 compute shader 挂起转储）确定 publisher 是否执行及其写入路径。
- 补充排除（本轮）：hc_sum 12/13-binding 两变体等待均有界（spin_max + fail-closed，err=0 排除超时路径）；stage 0 的 incoming 发射在 `s>0` 分支内（s=0 跳过，无 reference→late 标记问题）；G12 跳 Q 测试因宿主轮询无条件而不可判。远程可验证的全部组件已排除；剩余唯一未验证假设为 GPU 端执行顺序/可见性异常，需交互式工具。全部改动为 opt-in（默认路径 G4/G7 证明无回归）；BF16 LateBind 路径标记 OPEN，需 GPU 侧内省（l2-diag 全零 status）继续调试。CPU/ctest 全绿（含 test-tp5-rank-local-readback 重建后通过）。
