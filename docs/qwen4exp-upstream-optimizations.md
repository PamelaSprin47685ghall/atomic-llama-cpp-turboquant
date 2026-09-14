# 上游 Qwen4 (Qwen3.8-Flash-Next / qwen4exp) 优化与前沿 PR 跟踪文档

> **当前硬件门：暂停后续模型/压力测试。** 本次boot出现新的内存控制器MCE（RAS第103条，07:39:47 UTC，corrected patrol scrub，channel1、地址0x0740e000）；历史同通道还有大量纠错/OVERFLOW。所有测试实例已停，待现场内存通道维护与稳定性确认后继续完整集成和60/100 tok/s验收；未宣布目标完成。完整证据与边界统一见 `TP5.md` 的2026-09-14安全门，不能把之前局部“内核无新增日志”扩展为全程无MCE。

> **2026-09-14 验收范围纠正**：`-md` 只设置本地 sidecar 路径，不能据“模型已加载”或普通 TP5 回答正确证明 MTP 实际参与。此前没有 draft 计数的 `9.9` / `1..12` TP5 证据不得用作 PR #27836 的 GPU 投机验收。显式 `--spec-type draft-mtp` 曾在第二请求的 `cache_k_l48 (view)` 恢复中因复制分片 1632/544 字节不匹配中止；旧 canonical readback 的 clamp 亦会错拼两个头。现已按显式 replica 起点同时修复写入、读回、异步偏移和 reshape 映射，CPU 回归及真实五卡连续两次32-token请求通过（文本相同，draft分别15/32、13/36接受，1.47/1.43 tok/s）。这只关闭该恢复故障，完整质量、长稳及性能兼容仍 **OPEN**；证据见 `TP5.md` 和 `/tmp/tp5-reseat-connectivity/mtp-restored-request-{0,1}.json`。
>
> 重新插卡连通性与通信计时重复计数修正见 `../TP5.md` 的 2026-09-14 节。当前短中文比较请求错误回答 `9.11`；60/100 tok/s 质量与性能目标均未通过。旧 CPU/GPU 局部证据只代表对应受测场景。

> **[硬件总线恢复与实测状态更新 - 2026-09-13]**
>
> 1. **硬件总线恢复与局部验证解除**：
>    - 在完成硬件总线 `bus 44 mask 48` 修复后，系统恢复单设备与真实多卡受控测试能力，成功采集到 PR #28422 与 PR #28457 的独立 GPU 算子证明（GPU Operator Proof）。
>    - **真实 TP5 六分卷端到端实测**：全 6 分卷（all 6 shards）Qwen3.8 在实际 5 卡 TP5 上完成端到端加载与推理测试（配置：`Q8_0 K` / `TURBO4_0 V`，`ctx512`，`b32`，`np1`，`direct PLE`，`F32 SYNCFD`，`relay=off`，`replay=off`）。模型端到端正确回答 `"9.9"` 与 `"1..12"`。
>    - **吞吐实测与目标未闭环**：实测端到端解码吞吐为 **2.25 tok/s / 2.55 tok/s**。**TP5 > 60 tok/s 与全集成 > 100 tok/s 目标均未达成，必须严格保持 OPEN 状态**。
> 2. **有界算子证明 vs 完整兼容/性能区分**：
>    - 当前取得的 GPU 成果严格属于**有界算子调度与正确性证明（bounded operator proof）**，**绝不代表全套测试套件通过（not full suite passed）**，亦不代表达到生产级全量兼容与预期性能。
>    - **未决与在修项（OPEN）**：原生 GPU 检查点机制（Native GPU Checkpoint）当前仍然保持 **OPEN** 状态；真实模型下的时间线与命令重放问题（timelines / replay real model bug）正在修复中。
> 3. **安全与基准规范**：
>    - 历史五卡旧 TP5 测试（`s64 repeated 3` 测得的 `3.02 tok/s`）已确认为**无效吞吐基准（INVALID throughput baseline）**，严禁作为基准引用或用于任何性能外推。

---

## 目录
1. [核心机制认知与架构校正](#1-核心机制认知与架构校正)
   * 1.1 True Vulkan Sparse Flash Attention 复杂度与压实机制
   * 1.2 Indexer V 缓冲区的物理分配边界
   * 1.3 基础设施依赖：PR #27301 (`add_alloc_dep`) 与 TP5 跨设备拷贝
2. [上游 7 大核心 PR 实施与分层验证状态矩阵 (截至 2026-09-13)](#2-上游-7-大核心-pr-实施与分层验证状态矩阵-截至-2026-09-13)
3. [七大核心 PR 逐项跟踪与证据链](#3-七大核心-pr-逐项跟踪与证据链)
   * 3.1 PR #28330：Indexer 彻底免除 V Cache 物理分配
   * 3.2 PR #27970：真 Sparse Flash Attention（跳过全量 KV）
   * 3.3 PR #28699：QSA Indexer 增量池化键缓存（Incremental Pooled-Key Cache）
   * 3.4 PR #28457：Vulkan Qwen Small-M 矩阵乘对调与 Split-K 放宽
   * 3.5 PR #28422：Vulkan Top-K MoE 算子前向融合
   * 3.6 PR #28136：超大 PLE 表直接读取（Direct Reads for Lazy PLE）
   * 3.7 PR #27836：NextN / MTP Draft Head 原生投机采样支持
4. [5× RX 6800 TP5 跨卡通信与实测证据](#4-5-rx-6800-tp5-跨卡通信与实测证据)
5. [上游已知陷阱与当前未闭环问题清单](#5-上游已知陷阱与当前未闭环问题清单)

---

## 1. 核心机制认知与架构校正

### 1.1 True Vulkan Sparse Flash Attention 复杂度与压实机制
- **机制原理**：Vulkan 端的真稀疏 Flash Attention 不是全量 $O(2048)$，亦非任意离散索引直输，而是由独立的压实着色器 `flash_attn_sparse_compact.comp` 与主 FA 着色器协同工作：
  1. **掩码压实（Mask Compaction）**：压实着色器扫描长度为 $L$ 的掩码行，过滤提取活跃条目生成紧凑元数据缓冲区（时间复杂度 $O(L)$）；
  2. **稀疏计算（Sparse FA）**：计算内核根据元数据仅加载并计算活跃的 KV 块（时间复杂度 $O(\text{active KV})$）；
  3. **实际约束与溢出回落**：QSA 提示上限可达到 2051（并非恒定 $\le 2048$）。当有限条目数超过 $n_{\text{kv\_max}}$ 时，触发溢出标志（overflow flag），内核安全回落至针对全量 KV 的 Dense 遍历，绝非截断。
- **综合复杂度**：**$O(L)$ 掩码压实 + $O(\text{active KV})$ 实际注意力计算**。

### 1.2 Indexer V 缓冲区的物理分配边界
- **物理事实**：PR #28330 的实质改动是**移除了 Indexer 的 V 缓冲区物理内存分配**（使 `hparams_idx` 仅分配 Key 缓存）。
- **带宽边界**：该项优化**绝不保证带来 50% 的总显存带宽削减**。整网主干注意力的全量 KV 缓存、Q/K/V 投影权重以及 MoE 路由权重仍占主要带宽，严禁宣称全网显存带宽减半。

### 1.3 基础设施依赖：PR #27301 (`add_alloc_dep`) 与 TP5 跨设备拷贝
1. **PR #27301 内存分配依赖**：
   - 官方 PR：[PR #27301](https://github.com/ggml-org/llama.cpp/pull/27301)（`ggml: allow passing alloc dependencies in graph_optimize`）。
   - 核心机制：在 `ggml_backend_graph_optimize_params` 中引入 `add_alloc_dep` 回调，声明分支与融合节点的张量生命周期依赖，避免图优化器过早复用内存池。
   - **强依赖**：**PR #28422（Top-K MoE 前向融合）严格依赖 PR #27301**，否则输出缓冲区易被输入节点提前覆盖。
2. **TP5 设备拷贝与跨卡通信**：
   - 5 卡 AllReduce 跨卡通信依赖设备间传输（Device-to-Device / Host Staging Copy）与同步栅栏。
   - 在真实 TP5 全量 6 分卷实测中，已打通 `Q8_0 K` / `TURBO4_0 V` 结合 `F32 SYNCFD` 通信链路。

---

## 2. 上游 7 大核心 PR 实施与分层验证状态矩阵 (截至 2026-09-13)

全部 7 项 PR 补丁均已移植至本地工作树并编译。状态严格划分为**本地工作树实现**、**CPU 验证**、**GPU 验证状态**及**结论**：

| PR 编号与链接 | 核心功能与模块 | 本地工作树实现 | CPU 验证状态 | GPU 验证状态 | 当前结论与说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[PR #28330](https://github.com/ggml-org/llama.cpp/pull/28330)** | Indexer 免除 V Cache 分配 (`llama-memory-hybrid-idx.cpp`) | 已移植并编译 | **CPU 已验证** | **TP5 全模型实测集成** | 仅 V 缓冲区物理移除，非 50% 总带宽削减 |
| **[PR #27970](https://github.com/ggml-org/llama.cpp/pull/27970)** | 真 Sparse Flash Attention (`ggml-vulkan.cpp`, `flash_attn_sparse_compact.comp`) | 已移植并编译 | **CPU 已验证** (配置容差内对齐) | **GPU0 部分通过** (11 稀疏算子 + 6 重放用例) | 7th transfer 回归未测；多卡命令重放修复中 |
| **[PR #28699](https://github.com/ggml-org/llama.cpp/pull/28699)** | QSA Indexer 增量池化键缓存 (`qwen4exp.cpp`, `llama-memory-hybrid-idx.cpp`) | 已移植并编译 | **CPU 已验证** (Tiny CPU 误差 0) | **TP5 全模型实测集成** | 上游长文本增益本地未单独测定 |
| **[PR #28457](https://github.com/ggml-org/llama.cpp/pull/28457)** | Vulkan Small-M 矩阵乘对调与 Split-K 放宽 (`ggml-vulkan.cpp`) | 已移植并编译 | 不适用 (Vulkan 专用) | **GPU 算子证明通过** (M1 swap 1/1, M32 split_k=3 1/1) | 有界算子证明通过；上游测速本地未测量 |
| **[PR #28422](https://github.com/ggml-org/llama.cpp/pull/28422)** | Vulkan Top-K MoE 算子前向融合 (`ggml-vulkan.cpp`) | 已移植并编译 | **PR #27301 分配器 CPU 通过** | **GPU 算子证明通过** (256x22 top6 32/32 配对全过) | 有界算子调度验证通过；上游测速本地未测量 |
| **[PR #28136](https://github.com/ggml-org/llama.cpp/pull/28136)** | 超大 PLE 表直接读取 (`llama-context.cpp`, `--lazy-mode on-direct`) | 已移植并编译 | **CPU 已验证** (6-shard 主干输出 9.9) | **TP5 全模型实测集成** (Direct PLE 运行) | 真实 5 卡 TP5 跑通，冷启动未单独微基准测试 |
| **[PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836)** | NextN / MTP Draft Head 原生投机采样 (`qwen4exp.cpp`) | 已移植并编译 | **历史 CPU 已验证**；复制 KV 传输/恢复回归通过 | **连续两请求 draft 参与及缓存复用通过** | 1.47/1.43 tok/s；GPU 完整质量、长稳及性能 OPEN |

---

## 3. 七大核心 PR 逐项跟踪与证据链

### 3.1 PR #28330：Indexer 彻底免除 V Cache 物理分配
- **GitHub PR**: [https://github.com/ggml-org/llama.cpp/pull/28330](https://github.com/ggml-org/llama.cpp/pull/28330)
- **Commit**: `311d4211b` | **模块**: `src/llama-memory-hybrid-idx.cpp`
- **原理**: QSA Indexer 仅需 Key 计算打分，不读取 Value。通过将 `hparams_idx` 伪装为单向 Key 缓存，彻底取消其实际 V 缓冲区的分配。
- **性能宣称状态**: 上游宣称的“带宽节约”**在本地标记为未经验证（UNVERIFIED HERE）**；仅 V 缓冲区物理分配被移除，不能推导出整体带宽减半。
- **验证证据**:
  - **CPU 证据**：CPU 完整回归及 ASAN 通过；兼容 QSA Dirty Checkpoint（Session Version 10, Seq Version 3）。
  - **GPU 状态**：在真实 TP5 6 分卷全模型推理中成功集成运行。

### 3.2 PR #27970：真 Sparse Flash Attention（跳过全量 KV）
- **GitHub PR**: [https://github.com/ggml-org/llama.cpp/pull/27970](https://github.com/ggml-org/llama.cpp/pull/27970)
- **模块**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.comp`, `src/models/qwen4exp.cpp`
- **原理与实现**:
  独立压实着色器 `flash_attn_sparse_compact.comp` 执行 $O(L)$ 掩码压实，主 FA 内核仅读取活跃的 KV 条目；当活跃条目超限时触发溢出标志，回落至全量 Dense 遍历而非截断。
- **性能宣称状态**: 上游报告的加速比**在本地标记为未经验证（UNVERIFIED HERE）**。
- **验证证据**:
  - **CPU 证据**：在相同量化输入字节（quantized input bytes）下，GPU 与 CPU 数值对比在配置的容差（configured tolerance）内通过。
  - **GPU0 历史证据**：
    - `test-vulkan-sparse-attn`：单卡 GPU0 上 11 项数值用例全部通过（包含 Q8_0 与 TURBO4_0、-65504.0f 极值边界、全掩码置零、超限溢出回落及非零视图偏移），掩码变异下捕获 3 次命令重放命中。
    - `test-vulkan-command-replay`：6 项数值与生命周期测试通过（同图偏移、145 个子图驻留、270 超量图驱逐与复用）。
  - **未决项**：第 7 项传输回归测试（7th transfer regression）尚未在 GPU 上运行；真实模型下多卡命令重放与时间线仍在修复中。

### 3.3 PR #28699：QSA Indexer 增量池化键缓存 (Incremental Pooled-Key Cache)
- **GitHub PR**: [https://github.com/ggml-org/llama.cpp/pull/28699](https://github.com/ggml-org/llama.cpp/pull/28699)
- **模块**: `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`
- **原理**: 过去每步 Decode 均对所有历史块重复计算 `indexer_k_pooled`；PR #28699 固化已完成块的池化结果，新 token 仅增量计算尾部未满块。
- **性能宣称状态**: 上游报告的性能提升**在本地标记为未经验证（UNVERIFIED HERE）**。
- **验证证据**:
  - **CPU 证据**：Tiny CPU 架构全量重算 vs 增量缓存 Logits 误差为 0；IQ4 PLE mmap 及 direct 模式与 cached QSA 对比全量重算 Logits 误差为 0；脏 PLE/QSA 检查点恢复（ASAN 及 Release）在修复 `p_l` 与 KV 序列化后全部通过。
  - **GPU 状态**：在真实 TP5 6 分卷全模型推理中成功集成运行。

### 3.4 PR #28457：Vulkan Qwen Small-M 矩阵乘对调与 Split-K 放宽
- **GitHub PR**: [https://github.com/ggml-org/llama.cpp/pull/28457](https://github.com/ggml-org/llama.cpp/pull/28457)
- **Commit**: `6788edb4f` | **模块**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- **原理**: Decode 阶段（$M=1$）将矩阵乘 $A^T \cdot B$ 对调为 $B^T \cdot A$ 对齐连续访存；放宽 `split_k` 准入为 $N \ge \text{wg\_denom}$；优化 Qwen 相关的 coopmat2 Tile 选型。
- **性能宣称状态**: 上游报告的超高加速比**在本地标记为未经验证（UNVERIFIED HERE）**。
- **GPU 算子证明实测证据（GPU Operator Proof）**:
  - **M=1 输入对调验证**（证据源：`/tmp/tp5-smallm-swap.log`）：
    - 测试配置：`type_a=f16, type_b=f32, m=1, n=512, k=2048`，测试结果 `1/1 tests passed`，进程正常退出（exit code 0）。
    - DAP 调试器观察证实：在 `vec_q_f16` 路径下实际触发 `swap_inputs = true`；着色器执行清洁（kernel clean，单次派发耗时 102.44 µs）。
  - **Small-M split_k 放宽验证**（证据源：`/tmp/tp5-smallm-splitk.log`）：
    - 测试配置：`type_a=f32, type_b=f32, m=32, n=509, k=2112`，测试结果 `1/1 tests passed`，进程正常退出（exit code 0）。
    - DAP 调试器观察证实：实际配置 `split_k = 3`（非 7）；着色器执行清洁（耗时 1453.44 µs）。
  - **范围界定**：本项证明确立了算子级调度与对调分块逻辑在 RX 6800 上的正确性，属于有界算子证明，不代表全量端到端加速达成。

### 3.5 PR #28422：Vulkan Top-K MoE 算子前向融合
- **GitHub PR**: [https://github.com/ggml-org/llama.cpp/pull/28422](https://github.com/ggml-org/llama.cpp/pull/28422)
- **Commit**: `50182a53f` | **模块**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- **原理与依赖**: 强依赖 PR #27301 的 `add_alloc_dep` 声明生命周期依赖；在单次 Compute Shader 派发中融合 MoE 路由打分、激活与 Top-K 提取。
- **性能宣称状态**: 上游报告的测速增益**在本地标记为未经验证（UNVERIFIED HERE）**。
- **GPU 算子证明实测证据（GPU Operator Proof）**:
  - **算子配对正确性验证**（证据源：`/tmp/tp5-topk-gpu-proof.log`）：
    - 针对 `TOPK_MOE(ne=[256,22,1,1], n_expert_used=6)` 形状，穷举 `with_norm` (0/1)、`bias_probs` (0/1)、`gating_func` (0/1/2/3)、`scale_w` (0.0/2.0) 共计 32 组配对配置。
    - 在 Vulkan0 (AMD Radeon RX 6800) 上测试，**32/32 配对用例全部通过（OK）**，与 Oracle 比对一致。
  - **融合派发证明**（证据源：`/tmp/tp5-topk-fusion-dispatch.log`）：
    - 在 `with_norm=1, bias_probs=0, gating_func=0, scale_w=0` 评测用例中，Vulkan Timings 证实实际派发执行了融合着色器 `TOPK_MOE_EARLY_SOFTMAX_NORM SOFT_MAX`（单次微基准执行耗时 65.39 µs）。
  - **范围界定**：证明了 Vulkan 融合着色器的调度和数值正确性，属于调度证明（proof of dispatch），冷启动微用例耗时不代表全局持续推理吞吐。

### 3.6 PR #28136：超大 PLE 表直接读取 (Direct Reads for Lazy PLE)
- **GitHub PR**: [https://github.com/ggml-org/llama.cpp/pull/28136](https://github.com/ggml-org/llama.cpp/pull/28136)
- **模块**: `src/llama-context.cpp`, `--lazy-mode on-direct`
- **原理**: 针对超大 `per_layer_token_embd` 权重引入 direct 读取，绕过操作系统虚拟内存 Demand Paging 缺页开销。
- **性能宣称状态**: 上游冷启动提升宣称**在本地标记为未经验证（UNVERIFIED HERE）**。
- **验证证据**:
  - **CPU 证据**：真实 6 分卷 Qwen3.8 模型在 CPU 上使用 Direct PLE 路径运行（`F16KV`, `ctx512`, `b/ub32`, `np1`），端到端推理正确输出 `"9.9"`；Tiny CPU Logits 绝对误差为 0。
  - **GPU 实测证据**：在真实 TP5 6 分卷（Qwen3.8-Flash-Next）全模型加载中采用 direct PLE 模式成功完成推理。

### 3.7 PR #27836：NextN / MTP Draft Head 原生投机支持
- **GitHub PR**: [https://github.com/ggml-org/llama.cpp/pull/27836](https://github.com/ggml-org/llama.cpp/pull/27836)
- **模块**: `src/models/qwen4exp.cpp`, `src/llama-context.cpp`
- **原理**: 为 Qwen3.8 打通 `--spec-type draft-mtp` 原生投机采样通路，由自带 NextN Draft Head 进行投机与验证。
- **性能宣称状态**: 上游理论投机加速比**在本地标记为未经验证（UNVERIFIED HERE）**。
- **验证证据**:
  - **CPU 证据**：Sidecar CPU `draft-mtp`（`nmax 2`）正确回答 `"9.9"` (draft 2/2)；长文本序列正确回答 `"1 2 3 4 5 6 7 8 9 10 11 12"` (draft 18/18)；Tiny MTP 1536 状态误差为 0。
  - **GPU 实测证据**：全 6 分卷真实 TP5 集群上端到端正确输出 `"9.9"` 与 `"1..12"`，但端到端吞吐受限（2.25/2.55 tok/s）。

---

## 4. 5× RX 6800 TP5 跨卡通信与实测证据

1. **真实多卡端到端实测表现**：
   - 硬件总线修复后，5× RX 6800 集群执行了 Qwen3.8 全 6 分卷模型端到端评测（`Q8_0 K` / `TURBO4_0 V`，`ctx512`，`b32`，`np1`，`direct PLE`，`F32 SYNCFD`，`relay=off`，`replay=off`）。
   - **正确性**：端到端正确生成答案 `"9.9"` 及长序列 `"1 2 3 4 5 6 7 8 9 10 11 12"`。
   - **性能与未达标**：实测解码吞吐为 **2.25 tok/s / 2.55 tok/s**。与项目既定目标相差甚远，**TP5 > 60 tok/s 与全集成 > 100 tok/s 目标均未达成**。
2. **CPU 侧基准与底层通信验证**：
   - **非对称行拷贝**：CPU 5-rank 非对称拷贝测试中，Row 1 保持 100% 精确匹配（3072 浮点数跨 5 卡），Row 0 和 Row 2 未受影响。
   - **门控归约求和**：默认 CPU 归约门控求和 1 次 AllReduce 验证通过；目标 96 次连续归约在目标硬件上未经验证。
   - **分配器与回滚**：`test-alloc`（含 `test_graph_optimize_alloc_dep`）及循环检查点恢复全部通过。
3. **有界算子证明与全局性能区分**：
   - PR #28422（Top-K MoE 融合）与 PR #28457（Small-M 对调与 Split-K）均已取得清晰的单算子隔离调度证明（`/tmp/tp5-topk-gpu-proof.log`, `/tmp/tp5-topk-fusion-dispatch.log`, `/tmp/tp5-smallm-swap.log`, `/tmp/tp5-smallm-splitk.log`）。
   - 证明属于调度存在性与数值正确性（proof of dispatch / correctness），冷测试单次用例耗时不代表整体端到端加速。

---

## 5. 上游已知陷阱与当前未闭环问题清单

### 5.1 上游已知未闭环的 Issue 与陷阱
1. **Issue #28497: Top-K 非确定性边界选择**：打分相近时并行排序存在平局打散非确定性，相同 Prompt 运行可能产生轻微数值发散（需稳定 Tie-Break 机制）。
2. **Issue #27856 / #28734: QSA 线性衰减**：若未同步增量键池化与真稀疏 FA，长文本下单步 Decode 耗时随上下文增加呈线性劣质化。
3. **Issue #27993: RPC 跨节点多卡超 2K token 乱码**：跨节点切片在 Prompt > 2048 时可能触发同步空洞（单机多卡走 Vulkan P2P/Host 不受此影响）。

### 5.2 本地工程未闭环目标与待决事项
- [ ] **核心性能指标保持 OPEN**：
  - **TP5 > 60 tok/s 目标：未达成（NOT ACHIEVED），保持 OPEN。**
  - **全集成 > 100 tok/s 目标：未达成（NOT ACHIEVED），保持 OPEN。**
  - 历史五卡旧 TP5 测试测得的 `3.02 tok/s`（`s64 repeated 3`）已被明确确认为无效基准；当前真实 6-shard 端到端实测为 `2.25/2.55 tok/s`。
- [ ] **系统与功能未闭环项（OPEN）**：
  - **原生 GPU 检查点（Native GPU Checkpoint）**：当前保持 **OPEN** 状态；
  - **时间线与命令重放修复（Timelines & Command Replay Fix）**：真实模型下的重放与时间线同步缺陷正在修复中；
  - **第 7 项传输回归测试（7th transfer regression）**：尚未在 GPU 端执行测试。
