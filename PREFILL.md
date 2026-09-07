# FlashPrefill V2 全兼容移植：保姆级实战指南

> 本文是开发与验收指南。代码实现与编译交付完成；运行验收 NOT RUN：所有测试/压测/质量门均未运行，无性能结论，无部署动作。
>
> 目标：在 atomic / llama.cpp 的 AMD Vulkan 主线上移植 FlashPrefill V2 的选块与均值补偿，让它与 TurboQuant、InnerQ、TriAttention、RERoT、现有 MTP、统一 KV、RAM swap、抢占及恢复机制共存。不能靠禁用现有功能来换取“兼容”。
>
> 实现状态（2026-09-07）：代码实现与编译交付完成——server、4 个新增测试和 7 个既有测试目标均编译/链接通过，binary 见 `build-prefill/bin/llama-server`（基线冻结在独立构建，与 candidate 分离）。覆盖 CLI、配置/角色 API、GGML POOL/SELECT/ATTN、CPU/Vulkan kernel、路由、fit 预算、state envelope、`/metrics` 序列与 matrix/quality 脚本（详见 §7.5、§10.2、§12）。适用包络：普通 sparse dispatch 为共享 `build_attn`（支持的普通 KV 形状经公共调用方即可路由，无按架构逐个挂钩）；raw-Q RERoT 钩子仍仅 Qwen3.5 dense/MoE 族（含当前 Ornith）。族外 RERoT 与不支持形状无自动 sparse。运行验收 NOT RUN：测试仅编译，benchmark 与生产发布门均未运行；hash 见 `build-prefill-evidence/manifest.json` 与 `binary-libraries.sha256`，此处不预填任何假值。§21.3 复选框全部未勾选——实现不等于验收，任何“通过”表述都必须有第 18 节证据支撑。

## 0. 基线、范围与阅读顺序

### 0.1 本次建树记录

| 项目 | 值 |
| --- | --- |
| 源分支 | 本地 `master`，不是自动抓取后的 `origin/master` |
| 固定基线 | `4fc5c2213644f70a580649a33b192eb3077a3802` |
| 新分支 | `feat/flashprefill-v2` |
| 兄弟 worktree | `$HOME/Desktop/atomic-llama-cpp-turboquant-prefill` |
| 文档 | 新 worktree 根目录 `PREFILL.md` |
| 核对日期 | 2026-09-07 |
| 上游代码快照 | `qhfan/FlashPrefillv2@75b58f2ecdba1c269a87dd34d8f1ae57bef50c57` |

创建时源目录存在未提交修改；新树只继承上述已提交基线，不携带那些修改。后续移植其他分支的改进时，要重新跑 OFF 回归和组合矩阵，不能继续沿用这里的基线结论。

先读 [AGENTS.md](AGENTS.md) 和 [CLAUDE.md](CLAUDE.md)。AGENTS 中既有历史验证记录，也有剩余发布阻断项；不能因为某段写过“通过”，就把本文新增组合的测试标为通过。

本文三种措辞有意区分：**现有**指已在固定基线源码找到；**拟新增**指接下来要实现；**验收要求**指必须取得证据，不能把要求当成测试结果。**已实现**指代码实现与编译交付完成、运行验收 NOT RUN：`--flashprefill-*` 参数、新 GGML op（`GGML_OP_FLASH_PREFILL_POOL/SELECT/ATTN`）、`test-flashprefill-routing` / `test-flashprefill-state`、以及 `scripts/flashprefill-matrix.py` / `scripts/flashprefill-quality.py` 均属此类。**验收要求**一节的措辞保持原样，不因代码存在而改成结果。

### 0.2 按什么顺序做

| 想做什么 | 阅读位置 |
| --- | --- |
| 理解不能破坏的语义 | 第 1–3 节 |
| 设计数据结构、图和 Vulkan kernel | 第 4–9 节 |
| 处理状态恢复、显存与参数 | 第 10–12 节 |
| 按阶段实际开工 | 第 13 节 |
| 构建、启动隔离实例和测试 | 第 14–17 节 |
| 验收、排错和交付 | 第 18–21 节 |
| 查上游出处与移植差异 | 第 22 节 |

第一交付目标是当前 Vulkan/Ornith 类 hybrid 模型路径。CPU 用于小规模参考与单元测试；CUDA、Metal、RPC 上新增原生稀疏 kernel 不在这一轮范围内。既有普通 attention 在这些后端的能力必须保留，既有 RERoT 的 capability gate 也必须保留。

### 0.3 先纠正三个容易误判的地方

**不是链接一个现成库。** 上游是 Hopper SM90 的 CUDA/CuTe attention 和 Triton 索引实现，使用 BF16/FP8；本项目要重写 Vulkan 执行路径。FP8 支持不代表 Turbo2/3/4 支持。移植算法，不把 CUDA、PyTorch、Triton 或 SGLang 变成本项目的新运行时依赖。[S1]

**现有 server 已区分 prefill/decode。** `server_context_impl::select_inference_mode()`、`update_inference_mode()` 已存在。应复用它们，把真实执行阶段传入 llama graph，而不是重新造一套调度器，更不是用 `n_tokens > 1` 猜阶段。参见 [server-context.cpp](tools/server/server-context.cpp)，基线约 5593–5634 行。

**当前 RERoT 活跃 lane 会暂停 MTP drafting。** 基线在 `rerot_owns_slot(slot.id)` 分支处理 stale stamp 后直接返回；`common_rerot_validate_stage0()` 也说明了这个限制。不能把“RERoT + MTP 开关均开启”写成“活跃 lane 内已经执行投机验证”。本功能应保留既有可用阶段的 MTP，不能新增全局禁用；解除活跃 lane 的既有限制是另一个有独立验收门的扩展任务。参见 [server-context.cpp](tools/server/server-context.cpp)，基线约 5930–5957 行，以及 [common.h](common/common.h) 中上述 gate。

## 1. 固定产品契约：先写成断言，再写 kernel

### 1.1 三种稀疏概念不能混为一谈

```text
TriAttention：决定哪些 KV 仍驻留，必要时确实删除 sequence reference。
RERoT：决定某个 reader 能看见哪些 run，以及每个 run 对应的 Q 相位。
FlashPrefill：只在“驻留且可见”的集合中划分精算块与均值补偿块。
```

对一条 query row r，定义：

```text
V(r) = resident KV ∩ sequence/reader visibility ∩ causal/SWA legality
E(r) = 精算 token 集合
U(r) = 用均值代理的 fragment 集合

E(r) 与 U(r) 覆盖的 token 不相交。
生产 mean-correction 模式下，两者并集必须覆盖 V(r)。
已删除、不属于当前 reader、尚未发布或位于未来的 token 不得进入补偿。
```

这里的 dense 表示“对当前合法驻留集合做完整 attention”，不等于被 Tri 淘汰前的 FullKV。FlashPrefill 不能恢复已删除历史，也不能声称与 FullKV 数值等价。

### 1.2 十条不可退让的约束

1. **OFF 隔离。** 默认关闭。OFF 不分配 FlashPrefill scratch，不构建新 op，不更新新统计缓存，不改变旧调度、旧缓存兼容策略或旧 attention 图。
2. **Tri 语义不变。** 保持 fill-first、首次 pressure drain、固定 `3/32`、recent window `128`、sticky maintenance 和既有 fallback 次序。不改校准，不偷偷提高 residency。
3. **唯一 metadata owner 不变。** `llama_kv_cells` 仍然是 physical cell 的唯一事实来源。允许建立可丢弃的派生索引，不允许再造一个独立维护 cell 生命周期的容器。
4. **只改适用的 prefill attention。** recurrent/DeltaNet、decode、MTP draft、target verification、embedding/rerank 和未支持的特殊 attention 保持原路径。
5. **一个归一化。** 同一 query/head 的精算块、所有补偿块、所有合法 RERoT 相位组，最终进入同一个 softmax 分母。
6. **真实布局。** 读取实际 `type/ne/nb`、KV head 映射、V layout、padding、RoPE 与 InnerQ 参数，不按模型名称写死维度。
7. **GPU 热路径。** 选块、均值统计、索引生成与补偿留在 GPU；不得每层/每头同步搬完整 KV 回 CPU。
8. **可追踪失效。** KV 写入、reclaim、compaction、restore、rollback、发布和 context shift 都有明确的统计/索引失效规则。
9. **不能假全开。** 全开验收必须出现适用 prefill 的实际稀疏工作，并证明 Tri/RERoT/原有 MTP 各自在合法阶段工作；不能因 `tri_enabled` 或 `rerot_enabled` 就永久绕过 FlashPrefill。
10. **质量和性能分别验收。** 稀疏率、HTTP 200、短答案正确，都不能代替任务质量和端到端提速证据。

注意：Tri 的 fill-first 约束是“不提前删除 KV”。在仍保留全部 KV 时使用 FlashPrefill 近似计算，不改变这个约束，但会改变数值，因此必须单独评测质量。

Tri 的数值契约仍是 `target(L) = max(128, ceil(3*L/32))`；已压缩 sequence 在驻留引用数超过 `target + 128` 后按既有逻辑 maintenance。L 沿用原 server 的 logical length，不能再叠加 `n_decoded`，也不能改成 FlashPrefill 的精算 token 数。允许超过 target 的既有 hard/shared guard 行为及其计数口径继续服从 AGENTS。

## 2. 先认清现有代码入口

下表是在固定基线核对的导航点。行号只作辅助，后续以符号搜索为准。

| 层次 | 现有文件/符号 | 本功能要接的地方 |
| --- | --- | --- |
| 参数 | [common/arg.cpp](common/arg.cpp)、[common/common.h](common/common.h) | 添加配置、校验与帮助，不改变旧开关 |
| C API/上下文 | [include/llama.h](include/llama.h)、[src/llama-context.cpp](src/llama-context.cpp)、[src/llama-cparams.h](src/llama-cparams.h) | 配置下传、执行阶段、graph reuse key |
| batch/ubatch | [src/llama-batch.h](src/llama-batch.h)、[src/llama-batch.cpp](src/llama-batch.cpp) | 切分时保留阶段与逻辑 prompt 边界 |
| 普通图 | [src/llama-graph.cpp](src/llama-graph.cpp)：`build_attn_mha()`、`build_attn()` | 新旧 attention 分派；保留 inverse WHT 等收尾 |
| RERoT 图 | 同文件：`build_rerot_q_groups()`、`build_attn_rerot()` | 接入已经按 reader 相位旋转的 Q，不能二次 RoPE |
| 模型入口 | [src/models/qwen35.cpp](src/models/qwen35.cpp) | full-attention 与 recurrent 分支的边界 |
| KV | [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp)：`reclaim_kv()`、`compact()`、`rerot_build_attn_layout()` | 构建合法驻留视图，失效/重建派生统计 |
| KV metadata | [src/llama-kv-cells.h](src/llama-kv-cells.h) | 唯一 cell owner，不复制维护第二套状态 |
| RERoT 语义 | [src/llama-rerot.h](src/llama-rerot.h)、[src/llama-rerot.cpp](src/llama-rerot.cpp) | reader view、group、entry、publish/layout epoch |
| Tri scorer | [src/llama-triattention.cpp](src/llama-triattention.cpp) | 原样复用；不是 FlashPrefill 选块器 |
| GGML op | [ggml/include/ggml.h](ggml/include/ggml.h)、[ggml/src/ggml.c](ggml/src/ggml.c) | 新 op、形状校验、名称表和参数序列化 |
| CPU 对照 | [ggml/src/ggml-cpu/ops.cpp](ggml/src/ggml-cpu/ops.cpp) | 复用 RERoT attention 参考语义，新增小规模 oracle |
| Vulkan dispatch | [ggml/src/ggml-vulkan/ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp) | pipeline、supports_op、buffer、barrier、scratch |
| Vulkan shader | [flash_attn_base.glsl](ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl)、[flash_attn_dequant.glsl](ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl) | 在线 softmax、GQA 和量化读取 |
| shader 生成 | [vulkan-shaders-gen.cpp](ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp) | 注册并编译新变体，不能只新增未引用的文件 |
| fitting | [common/fit.cpp](common/fit.cpp)：`common_fit_rerot_capacities()` | 新 scratch 进入真实 B/P/K 联合预算 |
| server | [tools/server/server-context.cpp](tools/server/server-context.cpp)、[server-rerot.cpp](tools/server/server-rerot.cpp) | 阶段、重试、缓存、恢复、frontier 与 metrics |
| 现有测试 | [tests/CMakeLists.txt](tests/CMakeLists.txt)、[test-rerot-attn.cpp](tests/test-rerot-attn.cpp) | 添加新测试，保留旧对照 |

建议先运行这些只读搜索，不要从整个大文件头开始盲读：

```bash
git grep -n -E 'build_attn_mha|build_attn_rerot|build_rerot_q_groups' -- src/llama-graph.cpp
git grep -n -E 'rerot_build_attn_layout|reclaim_kv|compact\(' -- src/llama-kv-cache.cpp
git grep -n -E 'select_inference_mode|rerot_owns_slot|llama_decode\(' -- tools/server/server-context.cpp
git grep -n -E 'FLASH_ATTN_EXT_REROT|flash_attn_mask_opt' -- ggml/src/ggml-vulkan/ggml-vulkan.cpp
git grep -n -E 'rerot|triattention|memmove|arg-parser' -- tests/CMakeLists.txt
```

## 3. 要移植的算法：不是 top-k，也不是最后加一个平均 V

### 3.1 明确三种块大小

上游快照使用三种容易混淆的单位：[S2]

```text
G = Hq / Hkv                         # GQA 比例
BM = k_block_m                      # packed Q rows，而非 token 数
BN = k_block_n                      # 选块/均值统计的逻辑 K block
BT = attention 执行 tile            # 上游为 64 个 K token

packed row r:
    query_position = r // G
    query_head     = kv_head * G + r % G
```

上游示例 `BM=128, BN=128`，一个选中的 BN 块通常展开成两个 BT=64 的执行 tile。均值补偿仍应按 BN 块计一次，不能跟着两条执行 tile 重复计数。Vulkan 的物理执行 tile 可以按设备调整，但必须保留并测试单位转换。

`BM` 不一定整除 `G`。tile 可以从某个 GQA group 中间开始或结束；不能用 `BM/G` 的整数截断代替精确 packed-row 映射。

### 3.2 块均值与阈值选块

对合法块 J，先从实际缓存内容得到解量化后的 K/V，再计算：

```text
n_J    = 实际有效 token 数，不含 padding、空 cell 或别的 reader 的 token
kbar_J = sum(K_j) / n_J
vbar_J = sum(V_j) / n_J
```

上游选择器使用 tile 级能量，而不是逐 head top-k 或平均 logits：[S2]

```text
z[r,J] = scale * dot(Q_r, kbar_J)
M      = max(z[r,J])，只遍历有效候选
S[J]   = sum_r exp(z[r,J] - M)
keep[J] = S[J] >= alpha * max_J S[J]
```

再并入强制保留的 sink/local/边界块和指定的 dense Q tiles。上游 `abs_threshold` 对应这里的 alpha，示例为 `0.1`；它不是“保留 10%”或“丢掉 90%”的比例。

上游采用 `exp2` 和 `scale * log2(e)`。Vulkan 可以选自然指数或底数 2，但整条路径必须统一，尤其是下面的 `log(n_J)`。

初版可用额外 F32 score buffer，先保证结果正确。之后再移植单次 QK 计算、chunk running max 和紧凑索引复用的优化，不能第一天就让同一片内存同时承担三个未验证的角色。

### 3.3 均值补偿必须进入同一个 softmax

对精算集合 E 和补偿块 U，正确目标是：

```text
exact_logit_j = scale * dot(Q, K_j)
proxy_logit_J = scale * dot(Q, kbar_J) + log(n_J)

output = [sum_E exp(exact_logit_j) * V_j
          + sum_U exp(proxy_logit_J) * vbar_J]
         / [sum_E exp(exact_logit_j) + sum_U exp(proxy_logit_J)]
```

公式里的指数实际要用减 max 的稳定算法。它是零阶近似，不是任意 K/V 上的精确恒等式。上游 FP64 参考测试就是将真实 token 与补偿代理并入一个分母。[S3]

禁止以下写法：

```text
softmax(精算块) @ V + mean(跳过的 V)       # 错：两个量不在同一分母
只修正分子、不修正分母                     # 错
每个 RERoT span 归一化，然后平均            # 错
proxy_logit = dot(Q, kbar)                  # 错：漏掉有效 token 数
精算 BN 的一半、再按整个 BN 做补偿           # 错：重复计算
```

对使用底数 2 的 shader：`proxy_logit = scaled_dot_log2 + log2(n_J)`。

### 3.4 在线合并与 split-K

每个独立计算段保存未归一化三元组 `(m, l, o)`：

```text
m = 当前最大 logit
l = sum exp(logit - m)
o = sum exp(logit - m) * value

合并 A、B：
m' = max(m_A, m_B)
l' = exp(m_A-m') * l_A + exp(m_B-m') * l_B
o' = exp(m_A-m') * o_A + exp(m_B-m') * o_B
最终 output = o' / l'
```

精算和补偿可以先用不同 GPU dispatch 计算上述三元组，再统一合并，作为过渡原型。不能把两个已归一化输出直接相加。优化版应在稀疏 kernel 内融合补偿，减少 launch 与中间结果带宽。

split-K 的每个真实 token/补偿块必须恰好归属一个 split；模型自带的 attention sink 项只能计一次。空段应有明确 identity，避免 `-inf - -inf` 产生 NaN。

### 3.5 两种 sink、softcap 和边界

上游 `attention_sink=2` 表示强制精算前两个 K block。GGML `sinks` tensor 则是模型自带的 softmax sink 项。二者不是同一件事，都要按原语义保留。

若模型使用 logit softcap，拟新增路径先按原 attention 的规则变换 dot，再给代理加 `log(n_J)`。不要把 multiplicity 也塞进 tanh。这是本项目对非标准 score 的扩展，必须有独立 oracle；未经验证的 ALiBi、额外 KQ bias 或特殊 attention 应明确走现有正确路径，不套无 bias 公式。

上游参考对 causal 补偿采取保守条件：

```text
(J + 1) * BN <= prefix_len + q_pos_min_of_tile
```

对角线附近的块由精算路径覆盖，不做整块补偿。不要在对照阶段擅自加 `+1`。推广到 sparse position/RERoT 后，用真实 visibility 判断，不能继续用 `prefix_len = kv_len - q_len` 代替所有情况。[S3]

## 4. 第一大坑：chunked prefill 可能让功能完全不工作

### 4.1 `last_n_blocks` 实际保护的是 packed Q tiles

上游 `last_n_blocks=8` 的含义是当前传入 query 区间最后 8 个 packed Q tiles 保持 dense，而不是最后 8 个 K block。[S2]

例如仅为说明单位，假设 `G=4, BM=128, ubatch=256`：

```text
Q tiles = ceil(256 * 4 / 128) = 8
last_n_blocks = 8
结果：整个 ubatch 都走 dense。
```

一个长 prompt 被切成许多个这种 ubatch，若每次调用都重新套“最后 8 块”，会全程 dense。程序可以返回 200，开关也可以显示开启，但没有一条 query 真正得到稀疏加速。实际 G 必须从模型读取，不能照抄这个例子。

### 4.2 兼容参考与本项目生产策略分开

拟定义 `tail_scope`：

| 模式 | 含义 | 用途 |
| --- | --- | --- |
| `call` | 以本次送入 op 的 query 区间末尾计算 dense tail | 对齐固定上游参考 |
| `logical-prompt` | 以当前请求已知的逻辑 prefill 区间末尾计算 dense tail，跨 ubatch 保留坐标 | 本项目长 prompt 开发主线 |

`logical-prompt` 是明确的工程扩展，不得标为与上游任意 chunk 划分逐位等价。冻结 prompt 区间起点、终点和 GQA packing 规则；从 server 到 `batch.get_view()` 再到 ubatch 的所有切分，都保留原区间坐标。

prefix cache 命中后，只对真正需要计算的 suffix 建立逻辑 prefill 区间，并记录该策略。不要把“cached token 数”“本次处理 token 数”“完整对话长度”三个量混用。

不知道逻辑 prefill 终点的调用者，不得猜测：显式使用 `call` 或回到 dense，记录原因。RERoT 的 teacher-forced 私有注入区间也必须有确定边界，不能把整个动态 episode 当作已知长度 prompt。

`tail_scope`、dense-tail 数量和 packing 规则要进入 policy fingerprint。修改 ubatch 大小可能改变近似选块和深层 KV 数值；不要求不同 chunk 划分的 sparse 输出完全相同，但要求边界合法、结果有限并通过任务质量门。

## 5. 执行阶段：复用 atomic，不重写 atomic

### 5.1 配置与本次执行角色是两回事

拟新增一个执行描述，语义类似下表，具体 C/C++ 名称可调整：

| 角色 | FlashPrefill 行为 |
| --- | --- |
| 普通 prompt prefill | 满足长度、布局、backend 条件时启用 |
| 从合法 cache/state 恢复后重新 prefill 的 suffix | 重建视图后按同一策略判断 |
| 有确定边界的 RERoT teacher-forced 长注入 | 专门通过 RERoT 验收后允许 |
| 普通逐 token decode，包括多 sequence 合批 | 保留旧 attention |
| MTP draft / target verification / speculative replay | 保留旧 attention；不得仅因多行就稀疏化 |
| RERoT frontier 生成 | 保留旧 attention，不将多个 lane 的生成行冒充 prefill |
| embedding / rerank / 不支持的多模态区段 | 保留原有正确路径并说明原因 |
| 外部 `llama_decode()`，没有阶段说明 | 保守 dense；提供显式扩展接口而非猜测 |

先使用 server 现有 `inference_mode` 和 slot 状态构造描述，再在 `llama_context`/ubatch 下传。需要补充的是可靠的角色、逻辑 prompt 边界与 source row 映射，而不是另一套 prefill 调度器。

### 5.2 batch 切片与重试都必须保留语义

检查 `pre_decode()`、`batch.render()`、`batch.get_view(off,n)`、`decode()` 和内部 ubatch 拆分。每次缩小 batch 重试后，阶段、token offset、output row、sequence、reader view 都必须仍然对应同一行。

推荐用只读、具有明确生命周期的 sidecar 或版本化扩展接口；不要未经 ABI 评估就给公共 `llama_batch` 塞一个裸指针。默认旧调用路径不受影响。阶段信息在提交异步 graph 后不能指向已离开作用域的 vector。

当前 server 主线区分 prefill/decode；不要为套上游 continuous batching 范式而强行合并两种模式。基线 `rerot_batch_active()` 还明确拒绝普通行与 RERoT 行同处一个 ubatch，必须保持。

对未来或外部调用产生的混合角色 batch，应明确支持逐行路由，或者在执行前走正确 dense 路径；不能悄悄重新排列 recurrent token、采样行或 MTP acceptance indices。OFF 时尤其不能引入额外拆批或同步。

### 5.3 避免图缓存复用错路

图 topology/reuse key 至少区分：

```text
FlashPrefill OFF / exact-all debug / sparse
普通 / RERoT
输入与输出形状、dtype、stride 变体、head mapping
metadata capacity bucket、pool precision、kernel variant
影响 topology 的执行角色/路由与配置
```

每次变化的 epoch、实际 count、物理 indices 和逻辑坐标优先作为 input tensor 数据更新，而不是每次重建 graph。不能因复用了上一个 dense graph 导致 sparse 开关无效，也不能把前一个 reader 的输入数组留到下一次。

## 6. 数据结构：合法 fragment 在前，近似计算在后

### 6.1 不把 physical index 当 logical position

Tri compaction 后 `physical[0..used)` 可以紧凑，但保留 token 的逻辑位置仍有洞。RERoT 还有 storage position 与 reader-relative virtual position。至少区分：

```text
physical cell index       → 读 K/V bytes 的地址
logical/original position → 普通 causal/SWA 关系
storage position         → K 已写入的 RoPE 相位
reader virtual position  → RERoT 可见布局中的位置
```

`get_n_kv()` 还可能为了 graph reuse 对容量取 padding。它不是有效 token 数，更不是补偿权重。

### 6.2 拟新增派生表示

建议在 `src/llama-flashprefill.{h,cpp}` 中定义短生命周期 planning 类型；它们不是新的 KV owner。

```text
row descriptor:
    source query row、head packing、sequence / reader identity
    execution role、prefill interval、logical position
    visibility/view stamp、phase-group mapping

fragment descriptor:
    cell-reference offset/count，或经过验证的连续 physical range
    logical block identity、有效 token 数
    所属合法 visibility domain / run / phase domain
    row legality 描述、必须精算的边界标记

pool:
    每层、每 KV head、每 fragment 的 k_mean / v_mean

plan:
    每 query tile / KV head 的 exact-fragment 索引
    可补偿 fragment 索引或可推导的互补集合
    counts、capacity、overflow/error 状态
```

同一个 fragment 能用于一个 tile 的共享补偿，前提是其完整 token 集合对该 tile 中所有使用它的行都合法，且每行能得到与其匹配的 Q phase group。不能跨 sequence/reader 随意共享统计。

### 6.3 先选简单正确的 partition

普通路径先按逻辑 K block 划分，再与驻留引用、可见范围相交。对 partial causal/SWA 边界，初版强制精算，不尝试用一个整块均值代表不同 query 的不同子集。

RERoT 再按 run、可见性边界及有效相位域拆 fragment。对 irregular sparse positions，允许 fragment 使用 cell reference 数组，但同一行不能重复包含同一 physical token。

fragment 内计数必须来自真实成员。若 fragmented 布局导致近乎每个 token 一个 fragment，容量估计和性能决策要按实际 F，而不是强行假设 `F = ceil(K/BN)`。这个场景应保持正确，并可能因成本过高选择 dense；不能溢出后截断列表。

### 6.4 RERoT 布局前处理也要降成本

基线 `llama_kv_cache::rerot_build_attn_layout()` 逐 query 扫描 physical KV，再生成 `(physical key, Q-group)` entry。只把 shader 改成稀疏，仍可能留下 `O(Q × Kphysical)` 的 CPU 布局展开与索引传输。

生产设计应从现有 owner/reader view 派生可复用的 run/span/fragment 表，让相同视图中的 query 使用范围与 group 映射；不要为了选择少数块，先生成所有 query×key entries。

初版可以用旧布局做小规模 oracle，证明新 fragment 表展开后与旧合法 entry 集合完全一致。随后优化派生过程，但不能绕过 visibility 判断或维护第二份独立的 cell 生命周期。

## 7. GGML 图与 op 接口怎么落地

### 7.1 推荐先拆三段，成熟后再融合

以下名字是建议接口，不是现有 API：

```text
GGML_OP_FLASH_PREFILL_POOL
    实际 K/V + fragment/cell descriptors → pooled K/V

GGML_OP_FLASH_PREFILL_SELECT
    Q groups + pool + row/block legality + config → 有界稀疏计划

GGML_OP_FLASH_PREFILL_ATTN
    Q/K/V + pool + plan + phase/visibility inputs + sinks → attention output
```

每个 op 的 source 数必须符合实际 `GGML_MAX_SRC` 限制。必要时将整数描述打包成一个 I32 metadata tensor，再创建 views；不要任意增加 source slot，也不要把 host pointer 塞进 op_params 给 GPU 使用。

为 schema 加 version，并固定字段单位、字节对齐、I32/I64 范围及 signedness。所有 byte-size/offset 计算使用 checked 64-bit 运算；在转换成 shader 可用的整数前检查边界。

### 7.2 图里的次序必须真实成立

```text
Q/K/V projection + 原有 RoPE/旋转
→ 当前 ubatch K/V 写入缓存
→ pool 看见已经写入的缓存
→ select 看见 pool 与最新 descriptors
→ sparse attention 看见索引及缓存
→ 原有 inverse WHT / 去 padding / 输出投影 / gate
```

现有 `cpy_k()`/`cpy_v()` 通过图展开和缓存别名完成写入。新增 pool 不能只是“代码写在 cpy 后面”就假设没有调度重排；必须确认 GGML 依赖、别名和 backend barrier，必要时增加显式依赖表示。

先保持 output 的形状、precision、layout 和原 attention 完全相同。尤其不能漏掉 `res->add_fused_node()` 等现有可观测/调度关联，或绕过 LoRA 输出投影。

### 7.3 注册工作不能漏一半

新增 op 时逐项搜索并更新：GGML enum/name/symbol 表、构造与形状检查、CPU dispatch、Vulkan `supports_op`、pipeline 创建与 dispatch、graph clone/调试复制、必要的 scheduler capability 查询、测试和 CMake 源文件注册。

已有 `GGML_OP_FLASH_ATTN_EXT_REROT` 的旧语义不得被复用为“有时精确、有时近似但看不出来”。可以共享 shader helper，但新旧 op 或明确的版本化 mode 必须可区分。共享库 ABI 改动后，测试与部署要使用同一构建的一整套库。

### 7.4 exact-all 调试模式（已实现，验收 NOT RUN）

`--flashprefill-exact-all`：选择所有合法 fragment，不使用补偿，但仍经过新 descriptors、新 op 和新 Vulkan dispatch（op 参数 `exact_all`，plan 头回显；length gate 可 bypass，backend 缺失仍 dense）。

它应对齐“相同驻留 K/V、相同 reader view”上的旧 dense attention。这个 gate 没过，就不要调 alpha，不要用均值补偿掩盖 index、phase 或 stride 错误。

### 7.5 支持包络（已实现，诚实范围）

普通 sparse dispatch 为共享 `build_attn`（`attn_kv*` 公共路径，支持的普通 KV 形状经公共调用方即可路由，无按架构逐个挂钩）；raw-Q RERoT 钩子仍仅 Qwen3.5 dense/MoE 族（`src/models/qwen35.cpp` 与 `qwen35moe.cpp`，含当前 Ornith），shape-gated，无按模型名写死的维度。**族外 RERoT 与不支持形状走原有旧路径**，不自动获得 sparse——不得把本节写成全架构覆盖。第一交付目标仍是当前 Ornith。

钩子内的显式 dense 门（AUTO 静默 dense；REQUIRED 对能力缺口抛错，不伪装）：recurrent 层（backstop，正常走独立 linear helper）、SWA 层、`full_attn_layers` 前缀内层、MTP context、embedding/pooling 调用、special KQ bias（ALiBi 类加项）、非 GQA 层、跨层 head-mapping 漂移（以 `il_first` 定键）、非 F32 Q、Q/KV 形状不匹配（含 `Dk/Dv` 漂移）、transposed V cache（`nb[1] > nb[2]`）、multi-stream cache（`ne[3] != 1`）、无 sparse backend。RERoT 路要求 raw Q + rope sections，普通路要求 roped Q；layout stale（`eligible`/`freshness`/`n_queries`/`n_kv` 任一漂移）直接抛错，不降级猜测。CPU kernel bodies 已全部实现并编译（graph 侧旧依赖注记过时，以此为准）。metrics lazy `unique_ptr` 存储与实际 scratch producer 已落地。不承诺任何未来工作为已完成。

## 8. Vulkan 实现顺序与性能底线

### 8.1 先做原生基础版本

1. **Pool shader。** 按 fragment/head 读取实际 K/V，使用原生量化解码，F32 求和。先存 F32 means；验证后再考虑 F16 means。尾部 count=0 不发出代理。
2. **Select shader。** 先做清晰的 score/max/threshold/count/prefix-scan/write 步骤。保持确定性顺序，禁止越界、重复和无效尾 tile。
3. **Attention shader。** 精算被选 fragment，补偿其合法互补集合，并用同一 `(m,l,o)` 合并。先无 split，再加 split 与 merge。
4. **融合和复用。** 减少中间 buffer、融合补偿、复用 pooled history、GQA 协作读取，最后才调 subgroup/tile。

这些阶段应使用 backend-native dispatch。不能借“原型”之名在真实模型请求里启用全量 KV CPU readback。CPU oracle 只接受小规模测试输入。

### 8.2 不照抄 Hopper 的执行假设

本机发布目标是 AMD Vulkan；不能把 warpgroup、TMA、WGMMA、SM90 register budget 当成可用能力。应从设备实际 subgroup 大小、shared memory、buffer alignment 和支持的标量/矩阵路径选变体。

优先复用现有 `flash_attn_dequant.glsl` 与 RERoT grouped attention 的量化读取、GQA 分摊和 F32 reduction。性能变体必须保留无特定 subgroup 能力时的正确路径；调优 key 使用设备能力与 tensor 形状，不写死一个显卡型号。

### 8.3 GPU 索引生命周期

不要为知道实际 nnz 而每层把 count 同步读回 CPU。用上界分配 + GPU counts/offsets + 受 count 约束的 dispatch；需要 indirect dispatch 时核对设备能力和既有 backend 接口。

最简单可靠的容量策略是为一个 capacity bucket 预留最坏合法列表。若太大，使用分段索引/分段执行，而不是把整个 Q×K attention mask 存出来。必须有 overflow 检查与诊断；不允许取 `min(nnz,capacity)` 然后继续推理。

Pool→Select→Attention→Merge 的写后读需要 Vulkan 正确的 stage/access barrier。不要把 `barrier()` 当作跨 workgroup 或跨 dispatch 的全局同步。调试计时可在请求完成后读取 timestamp query，不能为 metrics 增加每层等待。

### 8.4 不能复用“全零 mask 跳块”冒充 FlashPrefill

现有 `flash_attn_mask_opt` 优化的是既有 mask 中完全不可见的区域；FlashPrefill 跳过的是合法区域里的近似块，并且要补偿其概率质量。两者可以共享调度机制，不能共享错误的“跳掉就不贡献”语义。

## 9. TurboQuant、InnerQ 与 RERoT 相位

### 9.1 在正确的数值域取均值

对 Turbo K，Q 已按现有路径完成 reader-relative RoPE，再做 WHT/InnerQ 对应变换；K 从缓存读取时处于已存储的域。新 pool 应对**实际缓存 bytes 解码后的数值**求均值。

```text
mean(dequantize(真实缓存 K/V))      # 参考与实现应对齐这个目标
quantize(mean(原始未量化 K/V))      # 不是同一个数值目标
```

不能把均值先 inverse-RoPE 到一个随便选的坐标再混合，也不能在 sparse shader 内对已经 pre-phased 的 Q 再做一次 RoPE。

### 9.2 保留既有旋转顺序

```text
RERoT Q:
    raw Q → 每个 reader/phase group 的 RoPE → 必要的 padding → WHT → dot

Turbo V output:
    精算与补偿在存储域完成同一次归一化 → inverse WHT → 去 padding
```

InnerQ 的 scale/inverse-scale 应复用 graph 当前选取的 tensor，不额外乘一次。`kq_scale` 也从现有 graph 传入；不能因为 padding 后 D 变大，就改成 `1/sqrt(D_padded)`。

K/V 类型不对称时分别判断：K 非 Turbo、V Turbo 仍需正确处理 V 的输出逆变换。K/V head dim 不同时，mean buffer、寄存器切片和 output shape 都要分别使用 Dk/Dv。

### 9.3 RERoT 的均值不能跨错相位

同一 query 对不同 run 可能使用不同 Q group。基线 graph 已根据 reader view 生成有效相位，K 留在 writer storage 相位不移动。

因此，一个 pooled fragment 只能代表可由同一有效相位域解释的 K；每条 query 使用该 fragment 时，要找到它自己的对应 Q group。可以共享 K/V 统计，但不代表所有 query 可以共享同一个 Q 向量。

如果两个 run 的相位域、visibility 或 reader 资格不同，应拆 fragment。绝不能为了凑满 128 个 token 把它们揉成一个 mean。

所有 phase group 的精算与代理 logits 仍合并到原 query/head 的同一个分母。相位组数不是输出 query 数。

### 9.4 必测类型与位置组合

覆盖 `f16/f16`、`turbo4/turbo2`、`turbo3/turbo3`，再覆盖现有后端允许的 K/V 非对称配置。非法组合由 capability gate 拒绝，不虚构支持。

维度至少含 64、128、256，以及实际代码允许的需要 padding 的形状、Dk≠Dv、GQA=1/2/4/8 和不整除 BM 的 GQA 测例。Ornith 的 partial IMRoPE 必须使用当前可信的 front-back half pairing，不得恢复旧 even/odd pairing 误解。

## 10. 状态、失效与恢复：不保存垃圾，也不复用错误语义

### 10.1 先区分两类状态

**派生状态**包括 pooled means、fragment 表、selected indices、scratch。它们可以全部丢弃，再从合法驻留 KV 重建。第一版不把这些 GPU 工作区写入 RAM cache/checkpoint。

**语义状态**包括产生现有 KV 的 prefill policy。虽然 FlashPrefill 不删除 KV，但早期层的近似 attention 会影响后续层的 K/V，因此其 cache 不能无条件冒充 dense-prefill cache。

已定稿的 policy envelope（`src/llama-flashprefill-state.h/.cpp`，启用时写在既有 KV 字节之前；OFF 读写原样 legacy 路径）：有界长度前缀 marker + 64B body，共 89B framed（marker 在前，使旧 reader 只做小分配后干净失败，无超大分配风险；无兼容垫片，未发布的 raw-64B 前缀按外来字节拒绝）。body 内容：算法/schema 版本、scope（full/seq，跨 scope 拒绝复用）、config policy fingerprint（全部近似字段 + model 身份，adapter 以 generation 域参与）、随机 128-bit process nonce（每进程固定，与 context serial 联合标识“同一进程内同一 live context”）、per-context serial（启用时非零）、adapter generation（每次有效 LoRA/cvec 变更递增，饱和值恒无效）。envelope 只带身份，不带 means/plans/GPU 指针/KV 内容/prompt 文本。启用 seq blob/file 有各自独立 magic；OFF 格式不变，互读安全拒绝。原 Tri/RERoT fingerprint 原样保留，envelope 包在外层先写先验。实现在树内，验收 NOT RUN。

### 10.2 缓存复用契约（已定稿，验收 NOT RUN）

OFF 的旧运行与旧 cache 格式继续按原规则工作。启用 FlashPrefill 的 state 读取先验 envelope 再碰任何 KV 字节：legacy 无头字节、未知版本/scope、任何身份字段不匹配一律拒绝（抛错，调用方返回 0），调用方必须 re-prefill。跨 context、跨进程、跨 restart 的复用被**故意拒绝**——没有 `.flashprefill` sidecar 文件，RAM/slot 恢复只认“同 nonce + 同 serial + 同 policy + 同 adapter generation”，其余一律拒绝后 re-prefill，不静默混基线。

进程内 prompt cache 的 key 同样按 policy 指纹 + context serial + adapter generation 隔离（OFF key 为 0，走 legacy 无隔离路径）。A/B 测试使用独立进程/缓存目录，避免 dense 复用 sparse-prefill 的后层 KV，或者反过来让 sparse 复用 dense KV 而虚报质量。

有效 adapter 变更：generation 递增 + 既有 graph 失效 + 派生重建 + RAM/blob policy 隔离使旧权重 KV 永不复用；无全局 `memory.clear`/`synchronize`（那会摧毁其他常驻 slot），per-slot KV/LoRA 生命周期保持基线原样。唯一无处可落的组合——启用 policy 叠加活跃 RERoT episode——在应用**之前**零变更拒绝（adapter/generation/KV/graph/schedule 全不动，C API 返回已有错误码）；server 侧对该请求报错，不静默沿用旧权重。

配置在 context 生命周期内不可变已实现（`common_context_params_to_llama()` 按值传入，构造时严格校验，坏配置直接拒绝 context 创建）。按请求覆盖不在本轮范围；若未来新增，必须把 fingerprint 与 graph reuse 一并补全。

### 10.3 失效表必须落实到调用点

| 事件 | 必须做什么 |
| --- | --- |
| K/V append | 标记本层尾 fragment/新 fragment 为 dirty；当前 graph 必须读到本次写入 |
| `seq_rm` / rollback | 删除对应派生成员资格；不能让未接受 draft 进入新 pool |
| Tri reclaim | 按新 per-seq keep-set 重建 fragment/count/pool；不沿用原 BN 权重 |
| `compact()` | 原 physical index 全部视为不可信；等待事务完成后换用新映射 |
| shared-prefix fork | 共享 immutable bytes 不等于共享 reader membership；分别验证引用集合 |
| RERoT publish/retract/frontier | 更新可见集合与 selected plan；公开前的内容不得影响别的 reader 的 pool |
| RERoT layout/context shift | 使 phase/位置相关计划失效；保持既有 MTP view-stamp 失效规则 |
| RAM demote/restore | 丢弃 GPU 派生状态；恢复完成、校验通过后重建，不保存 raw GPU pointer |
| seq clear / slot 复用 / 取消 | 清理对应派生引用；新请求不能继承旧 reader 计划 |
| 模型/adapter/相关缩放配置变化 | policy/context generation 改变，旧统计不能复用 |
| allocation resize/rebind | buffer generation 改变，地址相同也不能证明内容仍然有效 |

初版可以用一个统一 memory mutation generation 做保守全失效。它必须由现有 owner/内存操作统一推进，不能分散在 server 的若干成功路径里。增量优化时再拆 content/layout/view generation，并补充漏失效测试。

### 10.4 异步与失败路径

MTP worker、主 graph 和 compaction 不能读写同一块过期 scratch。复用 buffer 前遵守现有 scheduler/worker 的完成关系；不能因地址仍然存在就提前回收。

缩小 batch 的 retry 不得重复提交已经成功部分的 KV、重复累加成功 metrics、再次发送已输出 token。stats cache 只在对应 graph 成功后标为 valid。

内存移动/图执行失败后，不得继续用“可能一半新一半旧”的 means 提供答案。沿用现有错误处理和一致性恢复路径；非法 metadata、OOB、stale view 不是可掩盖的普通性能 fallback。

## 11. 显存与 auto-fit：没有第二份完整浮点 KV

### 11.1 显存预算公式

令 F 为实际 fragment 数、Hkv 为 KV head 数、R 为 query/head 工作段数量、S 为 split 数，则至少预算：

```text
pool bytes       ≈ F * Hkv * (Dk + Dv) * sizeof(mean_dtype)
index bytes      ≈ offsets + 最坏合法 exact/correction entries + error/count buffers
descriptor bytes ≈ fragment 表 + cell refs + row/phase mappings
split bytes      ≈ S * R * (Dv + 2) * sizeof(float)
scratch peak     = 上述 live range 交叠的峰值 + 对齐/临时 reduction 空间
```

仅作算术示例：连续 `K=65536, BN=128, F=512, Hkv=4, Dk=Dv=256`，F32 K/V means 约 4 MiB/层。保留十层的统计约 40 MiB；若逐层安全复用则不同。碎片化会增加 F，所以这个数字不能写进 reserve 常量。

### 11.2 初版与优化版选择

初版逐层重建并复用有界 scratch，生命周期容易验证，但多个 prefill chunk 反复读取历史 KV 可能很慢。优化版可保留封闭 fragment 的 means，只更新 append/reclaim 影响的部分；它增加显存与失效复杂度，要测净收益。

二者都禁止创建长期驻留的完整 F16/F32 K/V 镜像。特别是 Turbo2/4 路径，完整解压很容易把压缩节省全部吃掉。

### 11.3 接入已有 fitting，而不是额外藏一笔钱

当前 RERoT 使用 B people / P pens / K KV 的联合 fitting。新增预算应进入同一候选配置的峰值模型，覆盖实际 `-b/-ub`、GQA packing、RERoT span 容量、dtype 与 split 变体。

核对 existing graph allocation 已经计入的部分，避免 reserve 与 graph memory 双重计数。probe 和最终 runtime 必须使用同一 sizing helper；不要 probe 通过后再懒分配一个未计入预算的大 buffer。

OFF 的 helper 返回 0。开启后可以合理减少最终 K/B/P，但不能隐藏变化；记录实际 capacity 与新增 scratch。也不能为了给 scratch 腾空间而改变 Tri 的 `3/32`、hard guards 或 pressure 顺序。

`--rerot --total-kv auto` 在当前基线不接受显式 `-np`，因为 B/P 自动推导；第 15 节的 all-on 启动模板因此不带 `-np`。

## 12. 配置与 metrics（CLI/计数器/端点键已实现，阈值未冻结，验收 NOT RUN）

### 12.1 配置：已实现的 CLI（值仍是开发起点，非生产冻结值）

下表 CLI 均已实现（`common/arg.cpp`，`LLAMA_ARG_FLASHPREFILL*` 环境变量同名支持；`--help` 文本由 `llama-gen-docs` 生成进 `tools/server/README.md`，请勿手改）。默认值即冻结 v1 默认：`off / alpha 0.1 / BM 128 / BN 128 / sink 2 / window 4 / tail 8 / logical-prompt / min_kv 1024 / full_attn_layers 0 / mean-correction on / exact-all off`。数值仍是开发起点，不是经过 Ornith 质量标定的生产默认；最终只能在验收报告中冻结生产值。

| 参数 | 语义（实现对照） |
| --- | --- |
| `--flashprefill off\|auto\|required` | 默认 `off`；auto 按明确规则选择；required 用于发现应支持却未支持的 eligible 场景 |
| `--flashprefill-alpha F` | 阈值因子，有限数且 `0 < F <= 1`；**不是稀疏比例，也不是 residency 目标**，不调节 Tri 的 `3/32` |
| `--flashprefill-block-q N` | BM：packed Q rows（`token * GQA + subhead`），`1..256` 任意整数；不是普通 query token 数 |
| `--flashprefill-block-k N` | BN：均值/选择逻辑块，`64..1024` 且为 64 倍数；执行 tile 换算见 §3.1 |
| `--flashprefill-sink-blocks N` | 强制精算的合法 prefix blocks（开发起点 2） |
| `--flashprefill-window-blocks N` | 强制精算的 local/partial-visibility blocks（开发起点 4） |
| `--flashprefill-dense-tail-tiles N` | 尾部保持 dense 的 packed-Q tile 数（开发起点 8）；必须同时解释 tail_scope |
| `--flashprefill-tail-scope call\|logical-prompt` | `call` 以本次调用 query 区间末尾计 tail（对齐上游参考）；`logical-prompt` 以冻结逻辑 prefill 区间末尾计 tail（长 prompt 生产主线，跨 ubatch 保留坐标；边界未知时走 dense，不猜测） |
| `--flashprefill-min-kv N` | 按 resident-visible tokens 判断（非物理容量、非 Tri 已删历史）；阈值通过第 18 节实测冻结 |
| `--flashprefill-full-attn-layers N` | 前 N 个 **eligible full-attention layers**，而非前 N 个模型层；开发先取 0；recurrent/DeltaNet 层永不进入 sparse |
| `--flashprefill-mean-correction on\|off` | 正式模式为 on；off 仅用于消融，不得作为完成 V2 的配置 |
| `--flashprefill-exact-all` | 调试：经过新路径但精算全部合法 token；length gate 可 bypass，backend 缺失仍 dense |

已实现的角色/执行 API（`include/llama.h` + `include/llama-flashprefill.h`，policy v1 冻结）：只有 `PREFILL` 与 `REROT_TEACHER_FORCED`（有确定边界）可 sparse；`DECODE / MTP_DRAFT / MTP_VERIFY / SPECULATIVE_REPLAY / REROT_FRONTIER / EMBEDDING / RERANK / MULTIMODAL / UNKNOWN` 全部保守 dense。policy 经 `common_context_params_to_llama()` 按值传入后**对 context lifetime 不可变**；空 exec（OFF）走普通 `llama_decode()`，显式 exec 走 `llama_decode_with_flashprefill()`（NULL exec 即 dense；`n_rows != n_tokens` 拒绝 `-1`）。外部无阶段说明的 `llama_decode()` 永不猜测、一律 dense。基线 RERoT 活跃 lane 暂停 MTP drafting 的限制原样保留（见 §0.3）。后端：CPU 参考 kernel 与 Vulkan 原生 shader（`flashprefill_pool/select/attn.comp`）支持；CUDA/Metal 明确返回不支持（`return false`），走设计内 dense，不伪装、不报错掩盖。plan 为 24-word 头（`GGML_FLASHPREFILL_PLAN_HEADER_WORDS`），`visible/exact_tokens` 为 64 位（头 word 16..19）；metrics 只计成功完成 graph/slice 后的实际 GPU 工作，不计理论 eligible（见 §12.2）。

`required` 仍然允许 decode、dense-tail、短上下文等设计内的 dense 路由；不允许把“不支持 RERoT/Turbo/Tri”伪装成常规 fallback。能力不足在执行前报明原因。损坏的索引和跨 reader 数据是错误，不因 auto 模式就静默吞掉。

auto 的性能判定应由 shape、可见规模和事先测得的成本规则驱动；不能根据模型回答好坏实时修改 alpha。改变 policy 必须显式记录。

### 12.2 指标：已实现的计数器与冻结的端点键（验收 NOT RUN）

已实现并接入端点（`src/llama-flashprefill-metrics.h/.cpp` 纯聚合器 + `server-context.cpp` 内 `get_metrics` 序列化，`server-task.h/.cpp` 以 `fp_*` 键透传；实现已在树内，验收 NOT RUN）：server 侧 `fp_eligible_rows`（sparse-eligible 角色 + 已知冻结边界，按成功 decode 切片计数；重试/失败/取消不计）与 `fp_dense_by_reason[7]`（`DENSE_OFF..DENSE_CAPACITY` 七档，成功路径 only；OFF 全零）；policy 标识 `fp_policy_fingerprint` / `fp_has_policy`（`llama_flashprefill_fingerprint` over 配置 + model/adapter 身份，用于 state/cache 隔离，OFF 时无）。GPU 决定量（plan 头 word 11..14 `sparse/dense_rows`、`selected/corrected`，word 16..19 `visible/exact_tokens` 64 位）在既有 graph completion 边界之后读取，**仅对成功 slice 做事务性合并**（`merge()` 每成功 slice 恰一次；plan error word 非零则整 slice 失败、无输出、零合并；重试/部分失败/取消永不合并）；不计理论 eligible。graph 结果访问器：`get_flashprefill_plans()`（completion 边界一次性读 plan 统计，永不逐层）与 `get_flashprefill_summary()`（CPU 构建期路由计数，无同步）。`/metrics` **恒常发射** flashprefill 序列：OFF 时全零且零计数成本，只有 `fingerprint_info` 按 policy 门控（有 policy 才出现）。冻结的 `/metrics` 序列（`llamacpp:` 前缀）：

```text
flashprefill_eligible_rows_total                          # source rows，eligible 角色 + 已知边界
flashprefill_sparse_rows_total                            # packed rows，plan 确认的稀疏 GPU 工作
flashprefill_dense_packed_rows_total                      # packed rows，plan dense（packed 口径分母）
flashprefill_dense_rows_total{reason}                     # packed rows × 10 档 presentation 原因：
                                                          # decode / mtp_verify / role_other / short /
                                                          # tail / unknown / unsupported / highcost / no_plan /
                                                          # full_attention_layer（设计内 dense 全前缀/SWA 层：
                                                          # graph summary 实际 query-head-layer 计数，零 plan 包含在内，
                                                          # 同计入 dense_packed 分母；永不进 NO_PLAN/失效计数）
flashprefill_server_dense_rows_total{reason}              # server 侧 × 7 档冻结路由（DENSE_OFF..DENSE_CAPACITY）
flashprefill_selected_blocks_total                        # plan exact uses（头 13）
flashprefill_corrected_blocks_total                       # plan proxy uses（头 14）
flashprefill_visible_tokens_total                         # plan use-record 求和（头 16/17，64 位）
flashprefill_exact_tokens_total                           # plan exact-use 求和（头 18/19，64 位）
flashprefill_pool_rebuild_total{reason}                   # slice / exact_all（2 档，观测值）
flashprefill_plan_invalidations_total{reason}             # bypassed / empty_snapshot / no_plan（3 档；no_plan 恒为零：
                                                          # 每个无 plan 结果都有权威判定，无需失效；残余未分类行计入 dense 侧 no_plan 桶）
flashprefill_scratch_bytes / flashprefill_scratch_peak_bytes  # 本 slice 实际 sizing（live 覆盖，peak 取 max）
flashprefill_layout_seconds                               # 仅 host 实测 layout 时间（chrono；未测量则省略，不造零）
flashprefill_pool_seconds                                 # GPU pool dispatch 时间（timestamp hook 报告前为 0，不造数）
flashprefill_select_seconds                               # GPU select dispatch 时间（同上）
flashprefill_attention_seconds                            # GPU attention dispatch 时间，含 merge（同上）
flashprefill_policy_fingerprint_info{fingerprint}         # 单序列 policy 身份（仅有 policy 时出现）
```

提交边界是**单次 end-of-call 同步**：每 ubatch 排队 96B plan 头快照（`ggml_backend_tensor_get_async`，跟随既有 logits/output 读），call 结束同步一次后逐头解析（`parse_plan_header_stats`），全部 plan error 检查先于提交；真正的 plan 错误使整 call 失败（清理后返回 `-3`），无输出、零合并。fold 不再丢弃零 plan summary；finalize 只为未完成的读做同步。`short` / `unsupported` / `highcost` 三档取 graph 侧权威 per-ubatch 判定（零 plan 时也记录），永不从全局 KV 充满度或 required 启发式推断。metrics 侧无 REQUIRED fail-closed：required 约束由 graph 侧抛错执行，metrics 只计数。`NO_PLAN` 为残余桶（无 plan 且无已分类原因的 dense）兼 AUTO 回退；其余序列名不变，dense 桶基数为 10。逐层 timestamp 等待仍被禁止；GPU dispatch 三项由 per-backend Vulkan timestamp 查询在同步后采样累加（`ggml_backend_vk_flashprefill_times`，attn 含 merge），某后端未报告前该项为 0——0 表示“未测量”，不是“零耗时”。

构建状态：最终构建通过，无运行时验证；hash 见 `build-prefill-evidence/manifest.json` 与 `binary-libraries.sha256`（新构建同路径刷新）。

不要把 sequence ID、prompt 文本或无界 reader ID 放进 metric label。为稀疏率明确分母：只能用合法驻留可见 token/块，不能用总物理容量或已经被 Tri 删除的历史。

完整请求结束时记录 dense 原因分布。验收脚本必须检查 `sparse_rows > 0`、实际发生跳块、`corrected_blocks > 0`，不能只 grep “enabled”。常规 OFF 不应产生新计数或 GPU 统计开销。

日志至少能区分：`off`、`decode`、`mtp_verify`、`short_context`、`dense_tail`、`unknown_prompt_boundary`、`unsupported_layout`、`high_estimated_cost` 与真正的执行错误。

## 13. 分阶段实战：每一阶段都有退出门

不要同时改 kernel、RERoT visibility、Tri scorer 和 server 抢占。下面的阶段顺序是为了让每个错误都有可定位的上一版对照。

### Phase 0：保存基线和测试条件

**做什么：** 确认 worktree/branch；构建未实现 FlashPrefill 的固定基线；保存 binary 和共享库 hash、完整启动参数、模型/校准 hash、设备/驱动、现有测试结果。第 14 节给出命令。

使用一个短 prompt、一个不触发 Tri 的长 prompt、一个确实触发 reclaim 的 prompt，分别保存原行为。重型 GPU 测试只在隔离设备或维护窗口执行，不能为赶进度与已占满显存的生产实例竞争。

**退出门：** 基线可以重复运行；已知失败和 SKIP 单独记录。AGENTS 中未完成的旧 release gate 不能因为这次建了新分支就自动消失。

**不要做：** 为清理环境执行 `git reset --hard`、删除原 worktree、覆盖系统 binary，或把源目录未提交修改抄进来。

### Phase 1：配置与阶段 plumbing，功能仍默认 OFF

**改哪里：** `common/arg.cpp`、`common/common.h`、`include/llama.h`、`src/llama-cparams.h`、`src/llama-context.*`、batch/graph inputs 和 server 下传位置。

**做什么：** 添加版本化配置、范围校验、执行角色和逻辑 prefill 边界；把 OFF 与常规 dense 原因接通。任何 API 默认初始化都必须回到 OFF。先不接 GPU kernel。

**测试：** 默认值、NaN/Inf alpha、负数、整数溢出、不支持 block 大小、缺失 prompt boundary、多 token target verification、多个单 token decode 合批、batch slice/retry。

**退出门：** `test-arg-parser` 和拟新增 `test-flashprefill-routing` 通过；OFF 与固定基线走同样的图和资源路径。不能因只有帮助文本就宣称已支持。

### Phase 2：纯参考算法与 descriptors

**改哪里：** 拟新增 `src/llama-flashprefill.{h,cpp}`、`tests/test-flashprefill-select.cpp`、`tests/test-flashprefill-attn.cpp`。

**做什么：** 用小数组实现块统计、PackGQA 映射、max-energy threshold、强制精算集、exact/proxy 合并。先使用 double/reference 数学，后实现实际 F32 统计语义。

把普通和 RERoT 的合法集合都从现有结构派生；新 descriptor 展开后与旧 entry 集合逐项对照。暂不追求性能。

**退出门：** 第 16 节的结构性测试全过；给定固定 selected plan 时，reference sparse+correction 与 proxy-token 显式拼接计算一致。

### Phase 3：新 GGML/Vulkan 路径只做 exact-all

**改哪里：** GGML op 注册、CPU dispatch、Vulkan dispatch/shader/CMake、`build_attn_mha()` 和 `build_attn_rerot()`。

**做什么：** 接入新 descriptors 和 graph inputs，所有合法 fragment 都精算。先 F16 K/V，随后至少跑一次真实 Turbo4/2 输入以暴露布局错误。

**退出门：** 同一驻留 KV 和 reader view 上，新 exact-all 与旧 dense reference 对齐；涵盖 sparse positions、多个 RERoT 相位、非连续 physical layout、sinks、Dk≠Dv 和尾块。OFF 图未改变。

**失败定位：** 这阶段误差不能归因于“近似算法”。只查输入映射、读写依赖、dtype/stride、phase、padding 和归一化。

### Phase 4：F16 Vulkan 选块与均值补偿

**改哪里：** 新 pool/select/attention shader，Vulkan 有界工作区及 barrier。

**做什么：** 先正确多 dispatch，再融合。跑 `tail_scope=call` 对齐上游边界，随后实现并测试 `logical-prompt` 的跨 ubatch 策略。

**退出门：** 给定同一 plan/means 的 GPU 与 CPU oracle 数值对齐；全选退化测试通过；有实际跳块与非零 correction；无 CPU KV readback、无每层 count 同步。

**特别检查：** 对照 `ub=128/256/512` 和不同 GQA。不能只用一个很大的单次 prefill tensor 证明 kernel 会稀疏，随后忽略真实 server 的小 ubatch。

### Phase 5：TurboQuant / InnerQ / GQA 全部接通

**做什么：** 按第 9 节接入真实量化读取、WHT/InnerQ 和输出逆变换。pool 的输入必须来自实际缓存，而不是量化前暂存 tensor。

**退出门：** f16/f16、turbo4/turbo2、turbo3/turbo3 及约定的非对称配置，通过 exact-all 和 sparse-oracle 两类测试。均值统计不创建完整浮点 KV 镜像。

错误归因要分开：量化误差、pool 存储精度误差、稀疏近似误差、实现误差不能合并成一个“总误差还行”。

### Phase 6：TriAttention + RERoT 的真正组合

**做什么：** 完成 fragment 可见性/相位划分、Tri reclaim 后重建、compaction 后 remap。先用旧 indexed layout 作 oracle，再消除长 prefill 的逐 query 全 KV 展开。

运行如下序列，保持 FlashPrefill 开启：

```text
dense resident prefill
→ KV pressure
→ Tri drain
→ 新 sparse-position prefill chunk
→ sticky maintenance
→ RERoT public commit / private reader / 多相位读取
→ context/layout 改变
→ 再次 prefill
```

**退出门：** descriptor 展开集合始终等于当前合法集合；不同 reader 的隐藏值不能进入对方 pool；Tri ratio/pressure 顺序不变；RERoT 适用的长 prefill 真实执行 sparse+correction，而不是因为开了 RERoT 就整条请求绕过。

### Phase 7：MTP、RAM、checkpoint、抢占与取消

**做什么：** 落实第 10 节失效表。复用现有 MTP worker、checkpoint 和 view stamp，不把 verification 稀疏化，不让 draft 的临时 KV 污染 pool。

**退出门：** 所有约定状态序列成功；失败注入有明确定义；全请求只提交一次成功状态、只输出一次终止事件。RERoT 既有 lane 暂停 drafting 的行为被明确记录，没有因 FlashPrefill 新增扩大暂停范围。

**更强目标：解除活跃 RERoT lane 的 MTP 暂停。** 这不是在 FlashPrefill 条件里去掉一个 return 就能完成。若要求该阶段也真正并行投机，必须增加独立工作包：冻结 draft 所见 reader stamp；禁止未完成 frontier 的跨视图接受；对 stale view 完整恢复 KV/recurrent/sampler 状态；只发布已接受前缀；处理多个 lane 的公平调度、取消和预算。单独证明每个 lane 有真实 draft/verify/accept，以及 publish/layout 变化时正确重采样。在该门通过前，报告里只能写“保留基线的阶段性 MTP 兼容”，不能写“活跃 lane MTP 已完成”。

### Phase 8：性能优化与 B/P/K auto-fit

**做什么：** profile layout、pool、select、attention、recurrent、MLP/MoE、排队时间。决定是否保留跨 chunk 的 means；优化 GQA 重用、fragment 表、GPU scan 和 scratch live ranges。

**退出门：** auto-fit 使用同一真实峰值模型；首次大 prefill 和大 drain 不 OOM；全 slot warmup 后 VRAM 不持续增长；有同配置端到端收益，而非只展示 kernel 理论 FLOPs。

先做性能盈亏平衡，不要先调 alpha 追求漂亮稀疏率。全开场景中 Tri 已压缩 KV、hybrid 只有部分层做 attention，收益上限与 dense Transformer 不同。

### Phase 9：组合质量、零回归与交付

**做什么：** 第 17–19 节全部门禁；冻结生产策略；更新帮助、文档和机器可读报告。每阶段一个可审查提交，不将全部修改压成无法二分的大补丁。

**退出门：** 第 21 节全部适用项有证据。暂未实现的后端、模型变体和既有 MTP 限制明确列出；不能用“全开”一词覆盖这些差别。

## 14. 可复制的工作区与构建命令

### 14.1 在新 worktree 内开始

以下命令不会替你创建第二个 worktree；本次 worktree 已建好。后续执行者从这里进入即可：

```bash
set -euo pipefail
cd "$HOME/Desktop/atomic-llama-cpp-turboquant-prefill"
git branch --show-current
git rev-parse HEAD
git status --short
git worktree list
```

首次执行预期 branch 为 `feat/flashprefill-v2`，HEAD 为第 0 节基线。开发提交后 HEAD 自然会改变，不要为了匹配这里而 reset。

查看上游时使用固定快照，只作参考，不在 AMD 机器执行其安装脚本：

```bash
export FP_UPSTREAM_SHA=75b58f2ecdba1c269a87dd34d8f1ae57bef50c57
export FP_UPSTREAM_DIR="${FP_UPSTREAM_DIR:-$HOME/FlashPrefillv2-reference}"
# 已存在则不 clone、不 checkout；下面仅只读检查既有目录。
test ! -e "$FP_UPSTREAM_DIR" && \
    git clone https://github.com/qhfan/FlashPrefillv2.git "$FP_UPSTREAM_DIR"
git -C "$FP_UPSTREAM_DIR" rev-parse --is-inside-work-tree
git -C "$FP_UPSTREAM_DIR" show "$FP_UPSTREAM_SHA:flashprefill_ops/test_mean_correction.py"
```

不要在原子化源码树中 vendor 整个 SGLang/CUTLASS 目录。翻译或复用源代码时保留适用的 Apache/BSD notices；本项目许可证不能自动覆盖上游各子目录的许可证。[S1]

### 14.2 编译固定基线，保存为对照

必须在开始改功能代码前完成这一小节。文档变化不会影响推理 binary。Ninja、CMake、C++ 编译器、Vulkan SDK/glslc 应已可用；缺依赖时先诊断，不更改生产驱动。

```bash
export REPO="$(git rev-parse --show-toplevel)"
export BASE_BUILD="$REPO/build-prefill-baseline"
export JOBS="${JOBS:-4}"
export EVIDENCE="${EVIDENCE:-$REPO/build-prefill-evidence}"
mkdir -p "$EVIDENCE"

cmake -S "$REPO" -B "$BASE_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON

cmake --build "$BASE_BUILD" -j "$JOBS" --target \
    llama-server test-arg-parser test-kv-cells test-triattention-score \
    test-rerot-attn test-rerot-runtime test-backend-memmove

ctest --test-dir "$BASE_BUILD" -N > "$EVIDENCE/baseline-test-list.txt"
git rev-parse HEAD > "$EVIDENCE/baseline-source.sha"
git diff --binary > "$EVIDENCE/baseline-working-tree.patch"
find -L "$BASE_BUILD/bin" -maxdepth 1 -type f -exec sha256sum '{}' + \
    > "$EVIDENCE/baseline-binaries.sha256"
```

先跑 CPU/轻量门，再安排 GPU 门：

```bash
ctest --test-dir "$BASE_BUILD" --output-on-failure --no-tests=error \
    -R '^test-(arg-parser|kv-cells|triattention-score)$'

# 含 GPU/backend 检查；只在允许占用设备的窗口执行。
ctest --test-dir "$BASE_BUILD" --output-on-failure --no-tests=error \
    -R '^test-(rerot-attn|rerot-runtime|backend-memmove)$'
```

测试进程 exit 0 不代表一定跑到了 Vulkan。检查输出是否真的枚举/执行目标 backend；没有设备、被跳过或只跑 CPU 的情况写成 SKIP，不算 Vulkan PASS。

基线构建完成后，不要在改了功能源码的状态下重建 `BASE_BUILD`。候选版本使用独立 `build-prefill`，并在 A/B 前复核基线 hashes。

### 14.3 构建候选版本和新测试

以下新 targets 需要在 Phase 1–7 添加到 `tests/CMakeLists.txt` 后才存在：

```bash
export BUILD="$REPO/build-prefill"
cmake -S "$REPO" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON

cmake --build "$BUILD" -j "$JOBS" --target \
    llama-server test-arg-parser test-kv-cells test-triattention-score \
    test-rerot-attn test-rerot-runtime test-backend-memmove \
    test-flashprefill-routing test-flashprefill-select \
    test-flashprefill-attn test-flashprefill-state

ctest --test-dir "$BUILD" --output-on-failure --no-tests=error \
    -R '^test-flashprefill-'
git diff --check
```

运行新 target 前缺 target 应直接暴露出来，不要把失败命令后面加 `|| true`。显式查看已构建 binary 的 `--help`，确认参数已经真实注册。

## 15. 隔离启动与真实请求配方

### 15.1 前置准备

本节是实现完成后的运行手册，不是此次已经执行的测试。不要向生产端口发压力请求，不要自动停止或重启生产服务。

先在 shell 设置模型和校准文件，使用实际存在的文件；不提供猜测的个人绝对路径：

```bash
: "${MODEL_GGUF:?请先 export MODEL_GGUF=模型GGUF的实际路径}"
: "${TRI_STATS:?请先 export TRI_STATS=可信Tri校准文件的实际路径}"
: "${BUILD:?请先完成候选版本构建并设置 BUILD}"
test -r "$MODEL_GGUF" && test -r "$TRI_STATS"
sha256sum "$MODEL_GGUF" "$TRI_STATS"
LD_LIBRARY_PATH="$BUILD/bin" "$BUILD/bin/llama-server" --help
ldd "$BUILD/bin/llama-server"
```

Ornith 校准须遵守 AGENTS：可信文件 hash 为 `e95dae507d1f4a64e29be160c5281f8a4308a3332dc9c9176e1a3a0af32e50e2`，旧 `7fbf...` 文件不能部署。这个 hash 仅适用于对应 Ornith 校准，不适用于其他模型。

`LD_LIBRARY_PATH` 只指向本次 build 的库目录，检查 `ldd` 和实际进程 maps，防止 candidate binary 加载旧生产共享库。不能只比较 server 主文件 hash。

### 15.2 首个受控 Tri + Turbo + FlashPrefill 压力实例

以下参数沿用已有小 KV 压力配方，并添加拟新增 FlashPrefill 参数。先检查端口未占用；在前台运行，完成后只终止这个 shadow 进程。

```bash
export PORT="${PORT:-18086}"
ss -ltn "sport = :$PORT"

LD_LIBRARY_PATH="$BUILD/bin" "$BUILD/bin/llama-server" \
    -m "$MODEL_GGUF" -ngl 99 -fa on \
    --host 127.0.0.1 --port "$PORT" --metrics \
    --kv-unified -c 8192 --total-kv 2048 -np 1 -b 512 -ub 256 \
    -ctk turbo4 -ctv turbo2 \
    --triattention-stats "$TRI_STATS" \
    --flashprefill required --flashprefill-alpha 0.1 \
    --flashprefill-tail-scope logical-prompt \
    --flashprefill-dense-tail-tiles 8 \
    --flashprefill-mean-correction on
```

这不是 all-on gate，只用于把 Tri drain、尾块策略和 Turbo 数值问题拆出来。不要默认原有 3519-token 例子的精确日志数字永远不变；断言应基于实际 logical token 数、合法 hard/shared guards 和当前容量。

### 15.3 生成一个可控长度的压力请求

下面只使用 Python 标准库和已有原生 HTTP API。`/tokenize` 用于得到真实 token 数，`/completion` 接受 token ID 数组，避免把字符数当 token 数。它是资源压力 smoke，不是任务质量测试。[L1]

在另一个终端设置同一 `PORT`，运行：

```bash
export PORT="${PORT:-18086}"
export TARGET_TOKENS="${TARGET_TOKENS:-3600}"
export EVIDENCE="${EVIDENCE:-build-prefill-evidence}"
python3 - <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

base = 'http://127.0.0.1:' + os.environ['PORT']
target = int(os.environ['TARGET_TOKENS'])
assert 1024 <= target <= 7000, '该脚本针对 8192-context 的受控 smoke'
out = Path(os.environ['EVIDENCE'])
out.mkdir(parents=True, exist_ok=True)

def post(route, payload):
    req = urllib.request.Request(
        base + route, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=600) as res:
        return json.load(res)

text = ''.join(f'Record {i:06d}: neutral reference material.\n' for i in range(target))
prefix = post('/tokenize', {'content': text, 'add_special': True})['tokens']
tail = post('/tokenize', {
    'content': '\nThe marker is PREFILL_OK. Repeat the marker only.\n',
    'add_special': False})['tokens']
assert all(isinstance(t, int) for t in prefix + tail)
assert len(prefix) >= target and len(tail) < target
tokens = prefix[:target-len(tail)] + tail
payload = {'prompt': tokens, 'n_predict': 32, 'temperature': 0,
           'seed': 42, 'cache_prompt': False, 'stream': False}
(out / 'pressure-request.json').write_text(json.dumps(payload), encoding='utf-8')
start = time.perf_counter()
result = post('/completion', payload)
record = {'input_tokens': len(tokens), 'wall_seconds': time.perf_counter()-start,
          'response': result}
(out / 'pressure-result.json').write_text(
    json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
with urllib.request.urlopen(base + '/metrics', timeout=30) as res:
    (out / 'pressure-metrics.txt').write_bytes(res.read())
print(json.dumps({'input_tokens': len(tokens), 'wall_seconds': record['wall_seconds'],
                  'content': result.get('content'), 'timings': result.get('timings')},
                 ensure_ascii=False, indent=2))
PY
```

验收时额外确认：发生真实 drain/maintenance；稀疏与补偿 counters 增长；无越界/NaN/OOM；三者不能只满足其中一个。raw completion 不保证采用 chat template，因此 marker 输出不作为语言模型质量门。

### 15.4 真正的 all-on 自动容量实例

需要与模型匹配的 MTP 能力。先确认该 GGUF 的既有 MTP 启动方式，在本例的 `--spec-type mtp` 之外需要额外 assistant 文件时按现有模型文档补齐。

```bash
LD_LIBRARY_PATH="$BUILD/bin" "$BUILD/bin/llama-server" \
    -m "$MODEL_GGUF" -ngl 99 -fa on \
    --host 127.0.0.1 --port "$PORT" --metrics \
    --kv-unified -c 262144 --total-kv auto -b 512 -ub 256 \
    -ctk turbo4 -ctv turbo2 \
    --triattention-stats "$TRI_STATS" \
    --rerot --rerot-frontier strong \
    --spec-type mtp --draft-max 2 \
    --flashprefill required --flashprefill-alpha 0.1 \
    --flashprefill-tail-scope logical-prompt \
    --flashprefill-dense-tail-tiles 8 \
    --flashprefill-mean-correction on
```

不指定 `-np`。读取最终拟合的 B/P/K，而不是沿用 AGENTS 中某次历史启动得到的容量。根据实际 K 构造 resident history 超过容量的多请求负载，不能固定“6 个请求一定够”。

先验证无 FlashPrefill 的相同配置能启动和完成请求；若 baseline 本身失败，标记基线缺陷，不随便关掉某个已有功能再称为 all-on。MTP active-lane 限制按 Phase 7 如实报告。

### 15.5 一个开关一个对照，别用一条命令测所有原因

至少保留这些独立配置：基线旧 binary；新 binary + FlashPrefill OFF；F16 + FlashPrefill；Turbo + FlashPrefill；Turbo + Tri + FlashPrefill；Turbo + RERoT + FlashPrefill；Turbo + MTP + FlashPrefill；全部开启。

要关闭 Tri，需要去掉 `--triattention-stats`，因为该参数本身就会启用 Tri。不能只去掉 `--triattention` 然后以为在测 FullKV。RERoT 的 `--rerot-frontier` 同样会隐式启用 RERoT。

## 16. 数值和结构测试：别拿“误差小”掩盖越权

### 16.1 一个不依赖 GPU 的补偿自检

以下 Python 片段用于理解/检查第 3 节的稳定合并，可直接运行；它不是生产实现，也不覆盖整个选择器。

```python
import math

def merge_attention(items, value_dim):
    """items: (logit, value_vector, multiplicity), 全部处于同一 score 域。"""
    m = -math.inf
    l = 0.0
    o = [0.0] * value_dim
    for logit, value, count in items:
        if count <= 0:
            raise ValueError('不允许空代理或负 multiplicity')
        if len(value) != value_dim or not math.isfinite(logit):
            raise ValueError('非法输入')
        if not all(math.isfinite(x) for x in value):
            raise ValueError('非有限 value')
        z = logit + math.log(count)
        m_new = max(m, z)
        a = math.exp(m - m_new) if l else 0.0
        b = math.exp(z - m_new)
        o = [a*x + b*y for x, y in zip(o, value)]
        l = a*l + b
        m = m_new
    return [x/l for x in o] if l else [0.0] * value_dim

# 一个真实 token 的 V=0；一个包含两个等价 token 的代理，其 mean V=3。
a = merge_attention([(0.0, [0.0], 1), (0.0, [3.0], 2)], 1)
b = merge_attention([(0.0, [0.0], 1), (0.0, [3.0], 1), (0.0, [3.0], 1)], 1)
assert abs(a[0] - 2.0) < 1e-12
assert abs(a[0] - b[0]) < 1e-12
# 大 logit 不溢出。
assert math.isfinite(merge_attention([(10000.0, [2.0], 128)], 1)[0])
print('mean-correction identity: PASS')
```

### 16.2 必须拆成三套误差对照

| 对照 | 回答的问题 | 不能用什么替代 |
| --- | --- | --- |
| 新 exact-all vs 旧 dense，使用同一量化缓存 | 接口、布局、phase、mask、kernel 是否正确 | 稀疏输出“看起来还行” |
| GPU sparse vs CPU sparse oracle，同一 plan/means | 均值补偿和实现是否正确 | 与 FullKV 的总体误差 |
| 实际 sparse 推理 vs dense-prefill 推理 | 近似对真实任务质量的影响 | 单层误差/marker smoke |

selector 本身独立检查 selected set。GPU 浮点在阈值附近可能产生 tie 差异：用有明确 margin 的样例做集合严格对照；对 threshold tie 单独规定确定性规则。不能通过宽松 output tolerance 吞掉选择器错误。

小规模有界输入可先使用 `atol=1e-4, rtol=1e-3` 作为 F32 累积实现的探索性门槛，再按已有 dense kernel 的实测误差确定更合适的冻结门槛。这不是所有量化/极端 logit 的通用保证。报告最大误差所在 row/head/维度、相对误差、NaN 数和输入尺度；任何改动容差都要解释原因。

### 16.3 结构性测试清单

| 类别 | 至少覆盖 |
| --- | --- |
| count/边界 | 0/1/63/64/65/127/128/129 tokens；空 padding；partial final block |
| GQA | G=1/2/4/8；BM 不能整除 G；partial packed tile；不同 head 的热点不同 |
| selection | 全等 scores、极端 logit、阈值 tie、热点块、全选、只剩 mandatory 块 |
| compensation | 全选 U=空；所有 V 相同；块内 K/V 相同；混合 count；漏乘/双乘 count 的反例 |
| mask | causal 边界、非零 prefix、SWA 边界、全 mask 行、sink tensor 与 sink blocks 同时存在 |
| split | split=1 与多个 split 的三元组合并；每个代理只出现一次；空 split |
| layout | 打乱 physical 索引、不同 strides、非连续 view、Dk≠Dv、padding |
| Tri | 非连续 logical positions；部分 shared refs 删除；compact 前后相同合法数值 |
| RERoT | 多 reader、多 run、多 phase、strong/lag1、publish 前后、private/pending 不可见 |
| state | clear→reuse、append→rollback、save→restore 到不同物理布局、graph capacity bucket 变化 |
| resources | 超容量/整数溢出拒绝；小显存；unknown backend；无 GPU 的测试不得伪装 GPU PASS |

强制加入“污染探针”：给不属于 reader A 的 K/V 设置极大值，保持 A 的合法输入不变，A 的 descriptors/pool/plan/output 必须不变；修改未接受 draft 的内容后 rollback，不能影响后续重建结果。

对 causal 的 value/mask 检查同样不能包含未来 token。注意上游 tile 级选择聚合多个 query 的统计，本身不保证任意拆分/重新 packing 的逐 token 结果相同；不要把合法的 tile 近似差异与跨 reader 越权混为一谈。

### 16.4 拟新增测试 target 的职责

```text
test-flashprefill-routing  参数、阶段、切片、tail_scope、OFF/no-op
test-flashprefill-select   PackGQA、能量阈值、mandatory 集、GPU/CPU 索引
test-flashprefill-attn     exact-all、mean correction、dtype/stride/phase/split
test-flashprefill-state    owner 派生、epoch、reclaim/compact/restore/rollback
```

为新 attention 测试实现显式 backend 选择和 required-backend 选项：CI 指定 Vulkan 时，没有 Vulkan 必须失败或明确 SKIP 并阻止该门通过，不能自动跑 CPU 后返回“全通过”。

## 17. 全兼容矩阵与状态序列

### 17.1 分层覆盖，不盲目跑所有笛卡尔积

`scripts/flashprefill-matrix.py` 与 `scripts/flashprefill-quality.py` 均已交付源码（仅编译检查 `py_compile`，未运行验收）：前者输入显式配置矩阵和固定请求集（`--matrix` + `--requests`，均为 `schema_version: 1` JSON；未知键直接拒绝），输出逐用例 JSONL、原始回复、metrics 差量、exit status 和失败原因；后者跑 A/B/C/D 四路对照（FullKV×dense/sparse、Tri(3/32)×dense/sparse，RERoT 固定 OFF，可选 paired RERoT/MTP 扩展），plan 与阈值必须事前声明。两脚本均未执行——不要在交付报告里列成已运行。请求 payload 均原样透传、无新增请求字段：FlashPrefill/Tri 开关是 server 启动期配置（`startup.argv`），按 `cache_domain` 隔离；不同 policy 的 KV 缓存永不互用。

第一层做 pairwise：FlashPrefill 与 Turbo、Tri、RERoT、MTP、RAM/cache 各自组合。第二层做主生产组合：Turbo4/2 + Tri(3/32) + RERoT strong + 既有 MTP + FlashPrefill。第三层专测异常状态转换。

| 场景 | 验收重点 |
| --- | --- |
| 新 binary，FlashPrefill OFF，既有组合全部开启 | 输出/行为/资源与固定旧基线对齐 |
| 长 prefill，小 ubatch，Tri OFF | 不被 last-tail 策略全部变成 dense |
| 首次 Tri drain 后继续 prefill | 重新统计真实驻留 fragment/count，策略不变 |
| Tri floor exhausted | 先 Tri，再既有 demotion/preemption；限制 batch 后请求完成 |
| RERoT 长 teacher-forced 区间 | 有真实稀疏计算，不跨 reader/phase 补偿 |
| 多 people / 多 pens | 单人多笔、多人均衡、每人一笔；不新增饥饿、串线或 orphan |
| MTP + FlashPrefill，RERoT OFF | draft/verify/accept 真实发生，verify 不走 sparse |
| 全开 | 记录 RERoT 活跃 lane 暂停的既有限制；其余合法阶段不退化 |
| recurrent-only pressure | 不触发 Tri KV reclaim；也不错误重建/淘汰 FlashPrefill KV 状态 |
| 非生成任务 | embedding/rerank 语义不变；不消耗新增 prefill sparse 资源 |
| unsupported 后端/特殊 mask | auto 明确 dense；required 对应能力缺口明确报错；RERoT 原 gate 不被绕过 |

### 17.2 shared prefix 三分支

按 AGENTS 的共享前缀场景建立至少三个 sequence，使用真正的 shared physical refs，不只是三个相同字符串但三个独立 KV 副本。

```text
共同 prefix
├─ A：继续长 prefill，触发 Tri maintenance
├─ B：保留部分公共 token，同时拥有不同私有后缀
└─ C：保存到 RAM，再恢复到另一组 physical indices
```

记录 `shared_keep > 0`、per-seq references、physical used。对同一个 shared cell，一方的删除不能让另一方的均值 count 或内容失效；相反，只要成员资格变化，就不能复用错误的 sequence-level pool。

### 17.3 每个状态链都要测“恢复后再工作”

```text
A. prefill → drain → save → erase → restore → append → maintenance
B. prefill → MTP checkpoint → draft → partial accept → rollback → append
C. prefill → RAM demotion → 其他请求复用 cell → restore → append
D. RERoT private → public commit → view invalidation → next reader query
E. long episode → context shift → phase/layout invalidation → continue
F. prefill → capacity shortage → smaller retry → complete streaming
G. multi-person → pen yield/preemption → resume → finish
H. mid-prefill cancel → release → new request reuses same slot/buffer
```

不能只检查 save/restore API 返回成功。恢复后至少触发一次新的稀疏 prefill/重建，并与无中断对照或相同恢复状态的 dense oracle 比较。

### 17.4 streaming 与可见输出

对使用 chat template 的真实请求，保存完整 SSE 并组装文本。每条成功 stream 必须只有一个 terminal finish 和一个 `[DONE]`；取消/错误流按既有协议检查，不强迫伪装成功终止。

RERoT 的内部 lane trace 不能泄漏成普通 content delta。JSON Schema、tool calling 和 final fence 的约束恢复不能因新 op/重试而丢失。输出预算、采样器索引、已接受 draft 和已公开 run 均不得重复提交。

## 18. 性能与质量：要证明什么，怎么留证据

### 18.1 先估算上限，再解释实测

对当前工作负载，若 attention 占原耗时比例 f，attention 本体加速 s，新增前处理/调度占原总耗时比例 h，则一个粗略估计是：

```text
端到端加速 ≈ 1 / ((1-f) + f/s + h)
```

这是性能分解，不是本项目预测值。hybrid recurrent、MoE、投影和 scheduler 都不会因减少 attention blocks 自动变快。Tri 压缩后可见 KV 更短，s 和 h 也可能改变。

上游论文的硬件、dtype 和服务栈与本项目不同，不把其 H20 结果当成 AMD/Vulkan 的收益承诺。[S4]

### 18.2 测试尺寸与 chunk sweep

在机器容量允许时，覆盖 logical prompt 4K/16K/32K/64K/128K；同时记录每个阶段实际 resident-visible KV，尤其是 Tri drain 前后。

对每个代表长度，扫实际可用的 `ub=128/256/512`，再按设备余量增加更大 ubatch。固定 `-b`、其他功能开关、模型、seed、采样和输出预算。不能让 sparse 用更大的 batch、dense 用更小的 batch，然后把差异全归于算法。

每个配置先 warmup，再交替进行至少 5 次独立测量；显存允许时增加重复数。单请求取 median，吞吐和尾延迟使用足够数量的并发请求，保存原始样本而非仅报一个最优值。

### 18.3 两组性能实验不能混在一起

**固定容量实验：** 两边固定相同 K/B/P、batch 和并发，用来回答算法本身是否划算。

**真实 auto-fit 实验：** 两边分别按真实资源自动拟合，用来回答用户实际得到的 TTFT、并发容量、吞吐、OOM 风险是否改善。FlashPrefill scratch 使容量变化也应算进结果，不能隐藏。

冷 prefix cache、暖 prefix cache、恢复后的 suffix 分开报告。A/B 之间不共享可能受近似影响的持久 KV。模型权重可以相同，运行状态必须隔离。

### 18.4 分项指标与建议门槛

记录 host layout、pool、select/scan、exact attention、correction、merge、Tri score/pack、recurrent/MLP、prefill wall time、TTFT、decode tok/s、端到端 wall time、峰值显存和拟合容量。

建议在 Phase 0 冻结以下项目门槛，数值是开发建议，不是已经达到的结果：

| 门 | 建议判据 |
| --- | --- |
| OFF 性能 | 重复测量后无显著回退；可先以 median 变化不超过 3% 作为排查线，必须结合噪声 |
| OFF 内存 | 无 FlashPrefill 分配；相同 graph 的资源策略不变 |
| 主长上下文负载 | 全管线 prefill median 至少约 10% 改善，或事先约定的绝对 TTFT 改善 |
| 短上下文/decode | 应路由 dense，无实质新增开销；变化超过约定噪声线则调查 |
| all-on | 不是仅 sparse-only 提速；主要部署组合也要有净收益 |
| correctness/resource | 0 OOB、0 NaN、0 跨 reader 污染、0 未处理 OOM、0 重复 stream |

没有净收益时，先定位 layout/pool/tail policy/fragmentation，不先提高 alpha。可以保持 opt-in 或仅适用部分 shapes，但必须公开条件；不能为了“总是启用”让用户稳定变慢。

### 18.5 质量的四路对照

同一量化模型、template、seed、采样和输出预算，至少比较：

```text
A = FullKV + dense prefill
B = FullKV + FlashPrefill
C = Tri(3/32) + dense prefill
D = Tri(3/32) + FlashPrefill
```

先固定 RERoT OFF，隔离两种近似的影响；再对 C/D 增加相同 RERoT 配置。MTP 影响执行路径与采样行为，也要在合法阶段做配对验证，不能跨不同采样策略直接比较答案。

测试内容沿用 AGENTS 的 AIME24/25、MATH-500、长上下文 retrieval/needle、多轮聊天、代码仓库问答、真实生产 prompt，并增加位置敏感、多个 needle、相似干扰项和 shared-prefix 分支题。若使用 RULER/LongBench，也冻结数据版本与模板。

先确定每个任务集的允许退化阈值，再运行。可将总体下降 1 个百分点、关键 retrieval 下降 2 个百分点作为初始讨论线，但样本量小的 AIME 不能机械套百分点判断，必须同时看逐题变化和置信区间。不得因总平均正常而忽略某个任务或长度段断崖。

分别保存题目 ID、配置 fingerprint、完整输出、评分器版本、逐题分数、token 预算耗尽/超时情况和失败样本。HTTP 200 但答案错误不是通过；增加输出预算得到更好分数必须对全部对照同步调整。

若 C 本身相对 A 有既有质量问题，单独报告；不能让 D“只比 C 差一点”掩盖整体问题，也不能把 C 的旧问题全算到 FlashPrefill 头上。任何新显著断崖都阻止发布，不通过修改 Tri ratio 掩盖。

### 18.6 每个实验目录至少保留这些文件

```text
manifest.json              # 来源、配置、版本、硬件、校准、policy
commands.txt               # 完整 build/启动/测试命令
requests.jsonl             # 每条请求、case id、seed、预算
responses.jsonl            # 每题原始回复与完成状态
scores.jsonl               # 评分器版本与逐题分数
metrics-before.txt
metrics-after.txt
server.log
timings.jsonl              # warmup 标记、单次时延、分项 GPU/CPU 时间
memory.jsonl               # 资源/容量/峰值随时间
binary-libraries.sha256
summary.json               # pass/fail/skip，不能只有平均吞吐
```

`manifest.json` 至少记录本项目 commit 与 dirty patch hash、上游 SHA、模型与 calibration hash、完整 params、tail_scope、pool precision、driver/device、实际 K/B/P、batch/ubatch、所有 fallback 原因以及 baseline/candidate 身份。

## 19. 长稳与失败注入

按 AGENTS 的既有交付门，至少覆盖 100 次 drain/maintenance、20 次 save/erase/restore、10 次 floor exhaustion/preemption，以及实际 fitted slots 的多小时运行；FlashPrefill 在其中保持适用阶段开启。

新增观察点：fragment/pool/index 缓存是否不断累积；第一次 graph capacity 扩容后显存是否稳定；取消时是否有 pending GPU 工作持有已释放的 descriptors；恢复后是不是一直重建全部历史 means 造成吞吐断崖。

失败注入至少覆盖：计划 capacity 不足、checkpoint policy 不匹配、restore 到不同 physical layout、graph reuse 输入代次变化、RERoT publish 导致旧 view stale、提交后取消、recurrent-only shortage。注入点应位于测试 harness，不在生产默认路径增加随机失败。

Vulkan validation layers 仅用于调试/测试，不与正式性能数字混用；有 validation error 就先修，不以“最终答案正确”豁免。记录 warmup 后 VRAM/RSS 的趋势，不能只看测试结束时进程退出释放了内存。

## 20. 排错表：先查哪个地方

| 症状 | 优先检查 |
| --- | --- |
| enabled 但 sparse_rows=0 | 阶段没下传、logical boundary 丢失、last 8 Q tiles 覆盖全部 ubatch、min-KV/layer gate |
| 有 selected index，但速度没变 | shader 仍扫全 KV/全 mask、旧 RERoT O(QK) layout、pool 每 chunk 全重建、过多同步 |
| Tri drain 后突然输出异常 | count 仍按原 BN、physical map stale、被删除 token 仍在 mean、compaction 后未更新代次 |
| RERoT only 错 | mean 跨 phase/run、Q 二次 RoPE、每组各自归一化、混入不可见 pending/private 内容 |
| Turbo only 错 | mean 数值域错误、InnerQ 重复缩放、V inverse WHT 漏掉/两次、scale 按 padded D 算 |
| 非对称 K/V 错 | 只按 K 类型判断 V 收尾、Dk/Dv 混用、V stride 或 head mapping 不正确 |
| 稀疏输出幅度系统偏小 | 漏掉 log(count)、模型 sinks 重复计入、分母/分子计数不一致 |
| 尾部或 split 错 | logical block 与 execution tile 混用、partial count、代理重复、空 split 的 NaN |
| restore 后首次请求错，第二次正常 | 旧 GPU pointer/epoch、pool validity 过早标记、持久 policy 没隔离 |
| all-on MTP acceptance 为 0 | 先确认是否处于基线 RERoT active-lane 暂停段，不把既有限制误诊为 FP kernel 问题 |
| OFF 也变慢/占显存 | 配置未默认关闭、reserve 总是非零、新 graph inputs/同步在 OFF 仍执行 |
| auto-fit 成功，首次长 prefill OOM | reserve 没覆盖 worst-case fragments/packed rows/splits、probe 与 runtime helper 不同 |
| 稀疏率很好但质量断崖 | 先跑 exact-all 和同-plan oracle，再查 alpha、dense-tail 及两种近似叠加 |

排错必须保存可复现输入，不只贴一张日志截图。若只能在真实模型上复现，截取相关层的小规模 Q/K/V 与 descriptors 用于离线测试时，确保不导出敏感生产 prompt 或无关用户内容。

## 21. 提交、交付与最终清单

### 21.1 建议提交边界

```text
docs(prefill): pin baseline and integration contract
feat(prefill): add configuration and execution-phase plumbing
test(prefill): add selector and correction references
feat(ggml): add exact-all prefill operation
feat(vulkan): add sparse selection and mean correction
feat(prefill): support TurboQuant and InnerQ layouts
feat(prefill): integrate Tri and RERoT fragment views
fix(prefill): invalidate derived state across restore and rollback
perf(prefill): size scratch in auto-fit and optimize chunk reuse
test(prefill): add all-on quality and resource gates
```

名字只是建议。每个 commit 必须与真实变更相符，不使用“完美”“全部通过”之类没有证据的提交说明。开始合并主分支新改动时，先记录 merge/rebase 前后的源 commit，并重跑影响矩阵。

### 21.2 不把部署当成默认的收尾动作

完成开发后先交付可复现的 candidate、文档与验收目录。经明确授权才替换生产 binary/共享库/配置；切换前检查加载路径，切换后核对实际进程加载的一整套库和健康/真实请求。

回滚包至少包括旧 binary、匹配的共享库与配置。新 cache schema/policy 的状态不能未经校验交给旧程序读取；必要时丢弃受影响缓存并重新 prefill。不能为了省一次 prefill 而冒险继续不兼容的 checkpoint。

此次编写指南没有执行上述部署动作，也没有创建任何新 kernel、参数或测试 target。

### 21.3 Definition of Done

- [ ] 固定基线与新 binary OFF 的自动化回归通过，OFF 无新增分配/同步/图节点。
- [ ] 执行阶段和 logical prompt 边界可靠，不误把 decode/MTP verification 当 prefill。
- [ ] 小 ubatch 长 prompt 中确实存在 sparse rows 和 mean correction，不是假开启。
- [ ] 新 exact-all 与旧 dense 对齐；GPU sparse 与同-plan CPU oracle 对齐。
- [ ] 普通、Turbo/InnerQ、partial IMRoPE、非对称布局和尾块均通过约定测试。
- [ ] Tri fill-first/3/32/sticky/floor 顺序不变，reclaim/compaction 后重新建立正确统计。
- [ ] RERoT reader 隔离、相位和全局 softmax 正确，适用 prefill 实际运行新路径。
- [ ] 既有 MTP 合法阶段不退化，draft/verify 仍走原路径；active-lane 限制单独列明。
- [ ] 若交付目标要求解除该限制，活跃 RERoT lane 的真实 MTP 扩展门也通过。
- [ ] shared-prefix、RAM swap、checkpoint、rollback、context shift、抢占和取消矩阵通过。
- [ ] 派生缓存失效完备，持久/进程内 cache policy 隔离，拒绝不兼容恢复。
- [ ] 所有 scratch 进入同一 auto-fit 预算，首次大 prefill/drain 不 OOM。
- [ ] 质量 A/B 无约定外退化，保存逐题输出与评分，没有靠改 Tri ratio 掩盖问题。
- [ ] 完整管线和主部署组合有约定净收益，报告真实 fallback 与容量变化。
- [ ] 多小时压力无泄漏、stale GPU state、死锁、5xx 或重复 stream。
- [ ] 所有适用旧 release blocker 仍有清晰状态；不能被新功能报告覆盖。
- [ ] 最终源码/构建/库/配置 hashes 与测试报告一致，未做的测试写为 NOT RUN 或 SKIP。

代码实现与编译交付完成、验收未运行：以上复选框**全部保持未勾选**（运行验收 NOT RUN，无一门可标通过）。最终源码/构建/库/配置 hashes 与测试报告一致性见 `build-prefill-evidence/manifest.json` 与 `binary-libraries.sha256`；未做的测试一律写为 NOT RUN 或 SKIP，不得因测试 target 存在而改写。

### 21.4 工作量与范围控制

按当前 Vulkan 主线、保留基线阶段性 MTP 语义的范围，预估新增及实质修改约 9,000–16,000 行（含测试），熟悉本项目和 Vulkan 的开发者约 5–9 周；这是计划区间，不是实测承诺。

有限 F16/普通 prefill 原型约 2,000–4,000 行、5–10 个工作日，但它不等于全兼容交付。最大变量是 RERoT fragment/layout 优化、实际 AMD kernel 盈亏、跨 chunk 统计缓存和组合质量。

若额外要求所有其他 GPU 后端都有原生 kernel，或解除当前 RERoT 活跃 lane 的 MTP 暂停，必须在上述范围外单独拆解和估时，不能把这些内容藏在“兼容”两个字里。

## 22. 出处、固定快照与移植差异表

本文依据固定本地源码、上游公开仓库的固定 SHA，以及作者论文摘要核对。没有把新闻截图里的性能数字当作本机测试结果。

| 标识 | 原始来源 | 用途 |
| --- | --- | --- |
| S1 | [FlashPrefill V2 README，固定快照][S1] | Hopper/CUDA/CuTe、依赖、示例参数、许可说明 |
| S2 | [上游 Triton 索引实现，固定快照][S2] | PackGQA、均值、tile-energy 阈值、tail tiles、BN/BT 转换 |
| S3 | [上游 mean-correction FP64 测试，固定快照][S3] | exact/proxy 单一分母、log(count)、causal correction 边界 |
| S4 | [作者论文 arXiv:2608.19758][S4] | 方法与硬件实验范围，非本项目性能证据 |
| L1 | [本项目 server API 文档][L1] | 原生 tokenization/completion、cache 和 slot 接口 |
| L2 | [本项目 AGENTS.md][L2] | Tri 固定契约、可信校准、旧 release gates |

实现中必须维护以下差异，不假称上游已经解决了本项目的特殊场景：

| 上游/参考语义 | 本项目移植策略 | 单独验收 |
| --- | --- | --- |
| Hopper kernel | AMD Vulkan 原生实现，CPU oracle | 设备/布局/性能 |
| BF16/FP8 输入 | F16 与现有 TurboQuant K/V 读取 | 数值域与量化 oracle |
| paged logical KV | unified physical cells + sparse logical positions | 映射、count、reclaim/compact |
| 常规 causal query | RERoT reader-relative phase/visibility | 多 phase 全局 softmax、隔离 |
| 本次调用最后 N 个 Q tiles dense | call 参考模式 + logical-prompt 扩展模式 | chunk sweep、质量、cache fingerprint |
| 示例 full-attention-layer 配置 | 本项目明确 eligible full-attention 层计数 | hybrid 模型的真实层覆盖 |
| 无本项目的 Tri 淘汰 | 只补偿当前驻留集合 | FullKV/Tri × dense/sparse 四路对照 |
| 上游 serving backend | 保留 atomic 既有 prefill/decode 调度 | OFF 零回归、抢占/恢复/预算 |

[S1]: https://github.com/qhfan/FlashPrefillv2/blob/75b58f2ecdba1c269a87dd34d8f1ae57bef50c57/README.md
[S2]: https://github.com/qhfan/FlashPrefillv2/blob/75b58f2ecdba1c269a87dd34d8f1ae57bef50c57/flashprefill_ops/flash_block_sparse_index_triton.py
[S3]: https://github.com/qhfan/FlashPrefillv2/blob/75b58f2ecdba1c269a87dd34d8f1ae57bef50c57/flashprefill_ops/test_mean_correction.py
[S4]: https://arxiv.org/abs/2608.19758
[L1]: tools/server/README.md
[L2]: AGENTS.md

## 23. 实现总结与编译交付记录

### 23.1 本次交付口径

本节汇总 FlashPrefill V2 的代码实现及已记录的编译结果。用户后续明确要求：

> 只写代码，不验证不验收，编译通过即可。

因此，本次交付结论为：**代码实现与编译交付完成，不等于运行验收或生产发布通过。** 测试目标已编译、链接，但未执行正式测试套件；未运行真实模型请求、质量 A/B、性能压测或长稳测试，也未替换生产 binary、共享库或服务配置。第 21.3 节的运行验收复选框保持未勾选，AGENTS 中既有发布阻断项不因本次代码交付而关闭。

前文“此次只新增指南、未创建 kernel/参数/测试”的表述属于指南最初编写阶段的历史记录；本节记录其后的实现交付，不能把两阶段混为一谈。

### 23.2 已实现的代码内容

| 模块 | 主要实现 | 关键文件 |
| --- | --- | --- |
| 配置与公开接口 | 默认 OFF；版本化配置、参数校验、显式执行描述及 policy fingerprint；保留公共 `llama_batch` 布局 | `include/llama-flashprefill.h`、`include/llama.h`、`common/arg.cpp`、`common/common.cpp` |
| 阶段与切片传递 | 复用原 prefill/decode 调度；传递 source row、逻辑 prefill 起止位置；切片、内部 ubatch 拆分及缩小 batch 重试保留对应关系 | `src/llama-batch.*`、`src/llama-context.*`、`tools/server/server-context.cpp` |
| 参考数学与 wire schema | 有界 I32 descriptors/plan、检查过的容量计算、PackGQA tile 能量阈值、mandatory 集、exact/proxy 单一 softmax 与稳定合并 | `ggml/include/ggml-flashprefill.h`、`ggml/src/ggml-flashprefill.cpp`、`src/llama-flashprefill.*` |
| GGML 与 CPU | 注册独立 POOL/SELECT/ATTN op、构造与形状检查、CPU dispatch、实际缓存解量化与 reference 运算 | `ggml/include/ggml.h`、`ggml/src/ggml.c`、`ggml/src/ggml-cpu/ops.cpp` |
| Vulkan 原生执行 | GPU pool、选块与有界索引、精算加均值补偿、split/merge、跨 dispatch barrier、lazy pipeline、执行错误状态与时间戳采集 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`、`vulkan-shaders/flashprefill_*.comp`、`flashprefill-interface.h` |
| 普通 attention 图 | 在共享 `build_attn(llm_graph_input_attn_kv *)` 接入符合条件的普通 KV attention；使用已经完成 RoPE 的 Q，不重复旋转 | `src/llama-graph.*` |
| RERoT 图与可见性 | 从现有 cell owner 和 reader view 派生 compact fragments/uses；按 reader、run、有效相位划分；过滤后计算 virtual positions；raw-Q 路径接入 Qwen3.5 dense/MoE | `src/llama-kv-cache.*`、`src/llama-rerot.*`、`src/llama-flashprefill-layout.h`、`src/models/qwen35*.cpp` |
| TurboQuant / InnerQ | 在实际缓存解量化域取均值；保留 RoPE→WHT→dot、原 `kq_scale`、V inverse WHT、非对称 Dk/Dv、padding 与输出投影顺序 | `src/llama-graph.cpp`、既有 `flash_attn_dequant.glsl`、新 CPU/Vulkan op |
| 状态与失效 | opt-in owner generation、每次提交的派生视图重建、policy 隔离、有界状态 framing，以及 restore/rollback/clear/布局变化后的失效处理 | `src/llama-kv-cells.h`、`src/llama-flashprefill-state.*`、`src/llama-context.cpp` |
| fitting 与观测 | 共用 admission/sizing 规则；graph reserve 保持实际 plan 生命周期；单独计入 backend-private split workspace；暴露 counters、原因、scratch 和阶段耗时 | `common/fit.cpp`、`src/llama-flashprefill-metrics.*`、server metrics 序列化代码 |
| 回归与验收工具代码 | routing/select/attention/state 测试源文件；矩阵执行器、逐题质量/性能记录工具和状态矩阵配置 | `tests/test-flashprefill-*.cpp`、`scripts/flashprefill-matrix.py`、`scripts/flashprefill-quality.py`、`scripts/flashprefill-state-matrix.json` |

### 23.3 集成中落实的关键约束

- **OFF 隔离：** 默认关闭；跳过新图输入、op、GPU 工作区和统计采集。异步 readback 容器也延迟到实际使用时才分配。
- **真实阶段：** 普通外部 `llama_decode()` 未提供执行描述时保持 dense；decode、MTP draft/verification、recurrent、embedding/rerank 不按 token 数猜成 prefill。
- **chunked prefill：** 支持 `call` 与 `logical-prompt` tail scope；后者保留跨 ubatch 的冻结区间，避免每个小 ubatch 都被误判为完整 dense tail。
- **合法集合：** ordinary fragments 只考虑当前 query 的 reader membership；不把其他 idle sequence 的私有或多模态 cell 放进本 reader 的 pool。RERoT 按过滤后的可见集合确定相位，不先展开整份 query×physical-KV entry 表。
- **一个归一化：** 精算 token、合法补偿代理和 RERoT 相位组进入同一个分母；代理包含有效 `count` 的对数项，softcap 先作用于 dot，再加 multiplicity。
- **边界精算：** partial visibility 的 mandatory 约束按 use/tile 生效，不因某个早期 query 只看见部分 fragment，就把该 fragment 在所有后续 tile 中永久标成精算。
- **真实依赖与布局：** pool 通过实际 source edges 依赖本 ubatch 的 K/V 写入；K/V 按原 attention 约定 permute；RERoT Q 使用 capacity-static groups 与 `(p,p,p,0)` 的 IMRoPE 坐标布局。
- **图复用：** 路由、容量和 kernel 形态进入 reuse key；位置、reader 映射、实际 counts 等作为输入数据刷新。固定 physical padding 下也能从 dense 转入 sparse，不依赖“只有重建图才会产生”的缓存来判断是否重建。
- **安全 fallback：** dense RERoT fallback 按需创建旧 span 输入；未被任何新 op 消费的 sparse 输入不向未分配 buffer 上传。损坏、越界或 stale metadata 不作为普通 AUTO 性能 fallback 吞掉。
- **异步成功提交：** plan 标为 graph outputs；每层只排队读取 96B header 到稳定存储，在确有 pending reads 时统一完成同步、检查错误，再提交成功统计。失败释放前完成 pending copies，不因 vector 扩容或提前清理造成悬空地址。
- **状态格式：** 启用时使用 89B framed policy envelope（长度前缀、marker、64B body），并区分启用后的 sequence blob/file magic；OFF 原格式保持不变。不兼容数据在进入 KV 恢复前拒绝，避免旧 reader 把新 magic 当成超大字符串长度。
- **KV 生命周期：** FlashPrefill 不负责删 KV。adapter policy 更新只退役相关策略身份和派生状态，不新增全局 `memory.clear()`；原 per-slot KV/LoRA 生命周期仍由既有代码负责。
- **保留基线契约：** 不改 Tri 的 `3/32`、fill-first、sticky maintenance 与 fallback 顺序；不新增全局 MTP 禁用。RERoT 活跃 lane 暂停 drafting 的原有限制保留。
- **真实计数：** 路由原因来自 graph，而非全局 physical KV 用量猜测；全 dense layer policy 单独计数。`mean_correction=off` 的消融不把未执行的代理贡献记成已完成 correction。

### 23.4 编译结果与可复现命令

记录配置为 `RelWithDebInfo`、Vulkan ON、server/tests ON。以下 **12 个目标**均完成编译与链接：1 个 server、4 个新增测试、7 个既有测试。

```bash
cmake -S . -B build-prefill -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON

cmake --build build-prefill -j 4 --target \
    llama-server \
    test-arg-parser test-kv-cells test-batch-alloc \
    test-triattention-score test-rerot-attn test-rerot-runtime \
    test-backend-memmove \
    test-flashprefill-routing test-flashprefill-select \
    test-flashprefill-attn test-flashprefill-state
```

两份 Python harness 还完成了字节码编译，状态矩阵 JSON 完成语法解析；这些都不是 harness 执行或模型验收。生成的 harness 字节码已清理。

构建产物和记录：

| 项目 | 路径 |
| --- | --- |
| Candidate server | `build-prefill/bin/llama-server` |
| 配套共享库与测试 binary | `build-prefill/bin/` |
| 冻结基线 | `build-prefill-baseline/` |
| 机器可读清单 | `build-prefill-evidence/manifest.json` |
| 程序及共享库 SHA-256 | `build-prefill-evidence/binary-libraries.sha256` |
| 源文件 SHA-256 | `build-prefill-evidence/source-files.sha256` |
| 实现到文件的映射 | `build-prefill-evidence/implementation-map.json` |
| 构建日志 | `build-prefill-evidence/build-full.log`、`build-final.log` |
| 执行范围与偏离记录 | `build-prefill-evidence/verification-scope.json` |

清单记录的 server SHA-256：

```text
1504f7ca4cb538fcad340647683f23b3437a7e4a387c24e3971371e4276406c1
```

该值仅标识已记录的构建，不能替代配套共享库校验，也不是永久版本常量。本节属于构建后的**文档追加**；原 manifest/source-files 中的 `PREFILL.md` checksum 对应追加前的文档快照，本次不改写历史构建证据。

### 23.5 支持边界与未执行项目

- 原生新增执行后端为 **CPU reference 和 Vulkan**；本轮不提供 CUDA、Metal、RPC 等后端的原生 FlashPrefill kernel。
- 普通路径通过共享 KV attention 入口按实际 shape/backend 能力选择，不等于所有架构、特殊 mask、transposed V 或 multi-stream 变体都自动支持。
- RERoT raw-Q 模型钩子位于 Qwen3.5 dense/MoE 族，包含当前 Ornith。未解除活跃 lane 的既有 MTP drafting 暂停。
- 不支持的布局、backend 和超出 admission 的资源形态按已记录规则保持 dense 或在 REQUIRED 下明确报错；不得把这些范围写成已验证支持。
- admission 与开发参数不是经过任务质量、吞吐或 TTFT 测量冻结的生产策略。
- 状态身份绑定当前 context/process 与 adapter generation；跨 context、跨重启或不匹配恢复需重新 prefill。开启 policy 时，活跃 RERoT episode 的有效 adapter 变更在应用前拒绝。
- OFF 零回归运行、exact-all 数值对照、GPU/CPU sparse oracle、真实模型 all-on、shared-prefix/RAM/checkpoint/rollback/context-shift 矩阵、质量 A/B、性能收益、长稳和生产部署均为 **NOT RUN / NOT PERFORMED**。测试源码存在、目标编译成功，不代表这些项目已经通过。

本次总结只补充实现与编译交付记录，不修改前文发布门，也不替代后续按第 16–19 节取得的运行证据。
