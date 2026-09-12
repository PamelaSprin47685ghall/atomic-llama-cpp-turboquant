# 上游 Qwen4 (Qwen3.8-Flash-Next / qwen4exp) 优化与前沿 PR 跟踪文档

本文档系统性整理上游官方（`ggml-org/llama.cpp`）针对 **Qwen4-Exp / Qwen3.8-Flash-Next / QSA（Qwen Sparse Attention）** 的核心优化 PR、设计原理及演进路线，作为当前仓库后续针对性性能升级的技术基线。

---

## 目录
1. [核心认知：当前实现的两大致命性能瓶颈](#1-核心认知当前实现的两大致命性能瓶颈)
2. [上游已合并 / 正在合并的核心优化 PR](#2-上游已合并--正在合并的核心优化-pr)
   * 2.1 PR #28330：Indexer 彻底免除 V Cache 物理分配
   * 2.2 PR #27970：真 Sparse Flash Attention（跳过全量 KV）
   * 2.3 PR #28699：QSA Indexer 增量池化键缓存（Incremental Pooled-Key Cache）
   * 2.4 PR #28457：Vulkan Qwen Small-M 矩阵乘对调与 Split-K 放宽
   * 2.5 PR #28422：Vulkan Top-K MoE 算子前向融合
   * 2.6 PR #28136：超大 PLE 表直接读取（Direct Reads for Lazy PLE）
   * 2.7 PR #27836：NextN / MTP Draft Head 投机采样原生支持
3. [上游已知未闭环的 Issue 与陷阱清单](#3-上游已知未闭环的-issue-与陷阱清单)
4. [对本工程（5× RX 6800 Vulkan）的落地优先级建议](#4-对本工程5-rx-6800-vulkan的落地优先级建议)

---

## 1. 核心认知：当前实现的两大致命性能瓶颈

在当前本地分支中，长上下文下 Prefill 与 Decode 性能急剧劣化的根因已通过源码与上游比对完全证实：

1. **“假稀疏（Fake Sparse）”与全量 Attention 访存**：
   - 现存实现虽然在 `build_qsa_top_k` 中计算出了 Top-K 个最相关的块（约 2048 个 token），但并没有通过索引进行紧凑抽取，而是构造了一个长达全池尺寸（如 10k～200k）的全量 Dense Mask，将未选中的位置填为 `-INF`。
   - 随后的 Flash Attention 算子仍然需要读取并遍历整个上下文范围的所有 Key 和 Value。既承担了 Indexer 的打分与排序开销，又完全承受了 $O(L)$ 的全量 KV 访存。
2. **Indexer 全量键池化反复计算**：
   - 每一轮 step 中，QSA 需要对历史所有块做均值池化（`indexer_k_pooled`）。在上下文增长时，该计算开销随上下文长度线性上升，完全抵消了稀疏注意力的收益。

---

## 2. 上游已合并 / 正在合并的核心优化 PR

### 2.1 PR #28330: Indexer 彻底免除 V Cache 物理分配
- **Commit**: `311d4211b`
- **模块**: `src/llama-memory-hybrid-idx.cpp`
- **原理**:
  QSA Indexer 仅根据 Key 进行打分与 Top-K 选择，在整个生命周期中**完全不读取、不计算 Indexer 的 Value 缓存**。
  上游通过将 `hparams_idx` 伪装成单向缓存（仿照 MLA 模式），彻底取消了 Indexer V 缓冲区的创建与显存分配。
- **收益**:
  - Indexer 缓存显存占用直接**减半**；
  - 减少 50% 的物理显存带宽抢占与缓存行污染。

### 2.2 PR #27970: 真 Sparse Flash Attention（跳过全量 KV）
- **Commit / PR**: PR #27970 (`CUDA + ggml: add sparse-fa for DSV4/GLM`)，扩展到 Qwen QSA
- **模块**: `ggml/src/ggml-cuda/`, `ggml/src/ggml-vulkan/`, `src/llama-graph.cpp`
- **原理**:
  改变当前 `Full KV -> Dense Mask (-INF) -> Full FA` 的假稀疏逻辑。
  Flash Attention 算子直接接收 `sparse_indices`（即 Indexer 选出的 Top-K 索引列表），算子内部仅加载并处理那活跃的 2048 个 token 的 K/V。
- **实测收益**:
  在长上下文（32K～1M）下，Prefill 获得 **1.2×～2.2×** 吞吐提升，Decode 获得 **1.56×** 提升，注意力复杂度从 $O(L)$ 彻底收敛至恒定的 $O(2048)$。

### 2.3 PR #28699: QSA Indexer 增量池化键缓存 (Incremental Pooled-Key Cache)
- **Status**: 正在上游审查中 (Opened Sept 10, 2026 by Rhonstin)
- **模块**: `src/models/qwen4exp.cpp`, `src/llama-kv-cache-idx.cpp`
- **原理**:
  过去每次生成 step，Indexer 都要重算整个上下文所有 block 的 `indexer_k_pooled` 均值与 RoPE。
  PR #28699 为 Indexer 引入了增量池化缓存：为每个已完成的 block 只池化一次并固化结果，新 token 到来时仅增量处理当前 incomplete 的尾部 block。
- **实测收益**:
  - 63k 上下文 Decode 速度提升 **+9.3%**；
  - 114k 上下文 Decode 速度提升 **+9.4%**；
  - 完全消除了长上下文下 CPU/GPU 对历史键的重复归约。

### 2.4 PR #28457: Vulkan Qwen Small-M 矩阵乘对调与 Split-K 放宽
- **Commit**: `6788edb4f`
- **模块**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- **原理**:
  1. 在 Decode 阶段（$M=1$），将矩阵乘 $A^T \cdot B$ 对调为 $B^T \cdot A$，使访存模式完全与连续显存跨步对齐，大幅度提升 GPU L2 Cache 命中率；
  2. 放宽 `split_k` 的准入条件：过去要求 $M \ge \text{wg\_denom}$ 才允许拆分 K 轴并行归约，现放宽为只要 $N \ge \text{wg\_denom}$ 即可开启，彻底激活小 batch 宽隐藏层时的并行度；
  3. 为 Qwen 特别优化了 Cooperative Matrix 2 (coopmat2) 的 Tile 选型。

### 2.5 PR #28422: Vulkan Top-K MoE 算子前向融合
- **Commit**: `50182a53f`
- **模块**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- **原理**:
  引入前向分配依赖图分析（`add_alloc_dep`），将 512 选 10 的路由打分、Softmax/Sigmoid、Top-K 排序直接融合至单次 Compute Shader 派发中完成，省去多次中间回写与显存往返。

### 2.6 PR #28136: 超大 PLE 表直接读取 (Direct Reads for Lazy PLE)
- **Status**: Open (PR #28136)
- **原理**:
  针对 26.8 GiB / 51B 规模的超大 `per_layer_token_embd` 权重，引入 `--lazy-mode on-direct` 绕过宿主操作系统的 demand-paging 缺页中断，在冷启动 Prefill 阶段带来最高 **2×** 的冷启动吞吐提升。

### 2.7 PR #27836: NextN / MTP Draft Head 原生投机支持
- **Commit / PR**: PR #27836
- **模块**: `src/models/qwen4exp.cpp`, `src/llama-context.cpp`
- **原理**:
  为 Qwen3.8-Flash-Next 正式打通 `--spec-type draft-mtp` 原生投机采样通路，使主干模型直接驱动自带的 NextN Draft Head 进行投机验证。

---

## 3. 上游已知未闭环的 Issue 与陷阱清单

在采纳上述上游成果时，必须规避以下已知踩坑点：

1. **Issue #28497: Top-K 非确定性边界选择**：
   当不同 block 的打分极为接近时，部分后端的并行排序算法会出现非确定性平局打散，导致相同 prompt 多次运行产生轻微的发散（可引入稳定 tie-break 修复）。
2. **Issue #27856 / #28734: QSA 线性衰减**：
   在长上下文场景下，若未同步 PR #28699（增量键池化）与 PR #27970（真稀疏 FA），随着 token 数增加，Decode 速度会不可逆地单调下降。
3. **Issue #27993: RPC 跨节点多卡超 2K token 乱码**：
   如果将 Qwen4 切片跨 RPC 宿主部署，当 prompt 超过 2048 时可能触发上下文同步空洞（单机多卡走 Vulkan P2P/Host 不受此影响）。

---

## 4. 对本工程（5× RX 6800 Vulkan）的落地优先级建议

针对我们当前 5× 16GB RX 6800 环境，建议按以下梯队逐项落地：

- [ ] **第一梯队（极低风险，立竿见影）**：
  - **PR #28330**：Indexer 去掉 V cache 分配（仅改 4 行代码，显存开销直接斩半）。
  - **PR #28457**：Vulkan Decode $M=1$ 输入对调与 Small-M `split_k` 放宽（直接提升单步 Decode 速率）。
  - **PR #28422**：Vulkan Top-K MoE 算子融合。
- [ ] **第二梯队（核心算法跨越，彻底解决长文本性能）**：
  - **PR #28699**：落地 QSA 增量池化键缓存，彻底消除 10k～100k 上下文下的重复池化开销。
  - **PR #27970**：在 Vulkan 后端落地真 Sparse Flash Attention（直接消费 Top-K 索引列表，废除全量 Dense Mask 回落）。
