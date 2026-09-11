# xKV-SR 完整兼容实施方案：低秩因子 TurboQuant 叠加压缩版

适用仓库：`PamelaSprin47685ghall/atomic-llama-cpp-turboquant`  
静态审阅基线：`5298ea46061c4780d173de866bdbba42f5918135`（master，2026-09-04）  
方案日期：2026-09-06；修订版本：r2（四路因子 TurboQuant、摘要量化与真实内存验收）

> 本文是拟实施设计，不是已经完成、构建或压测通过的补丁。下文新增的文件、参数、类型和测试名均为建议；现有入口沿用原版对上述提交的静态审阅记录，本次修订不代表重新审计当前远端代码。xKV 的论文结果不直接代表本方案在 Ornith、Vulkan、RERoT、TurboQuant 组合下的效果。

本次修订的交付目标：**先减少需要保存的元素数，再对低秩产物 A_K、B_K、A_V、B_V 本身做 TurboQuant；SR 摘要也提供独立低比特存储。** FP16 因子只是正确性基线，不再是满足压缩目标的最终交付。保留原有全局统一淘汰和全部兼容范围。

“倍率相乘”仅指相同基线、相同因子形状下的理想数据载荷；量化元数据、padding、摘要、原格式段、热区、共享 snapshot 和工作区全部计入实测。第 2 节的 4.75 MiB 是名义算例，不是对目标模型的实测承诺。

## 0. 先锁定范围

最终交付：不训练、不修改模型权重；继续全局统一淘汰；保留 RERoT 的 PUBLIC/PRIVATE/PENDING、PAC-DFS、DDVR、共享引用、虚拟位置、frontier、final fence；保留 MTP/推测解码、recurrent/hand seed、RAM swap、前缀共享、context shift、取消和抢占；**原有 TurboQuant 热缓存不变，生产低秩段必须实现 A_K/B_K/A_V/B_V 四路 TurboQuant 存储、按需解码重建，并交付摘要量化路径**。不训练任何 router、量化码本或模型权重；允许确定性的分解、重缩放与配置校准。

默认后端路线是 **CPU 参考 + Vulkan 生产**。当前 `llm_graph_input_attn_rerot::select_variant()` 显式拒绝 CUDA/Metal/RPC；不能声称现有 RERoT 已经在 CUDA 跑通。CUDA 是额外后端项目，不是本方案默认前提。现有不支持的组合保持明确拒绝，不静默切回普通注意力。

仓库 AGENTS.md 记载的模型验证基线为 Ornith、Vulkan、K=Turbo4/V=Turbo2，并有 partial IMRoPE；这只是仓库记录，不是本次重新验证的线上部署状态。保留现有受信任校准文件的 SHA-256 验证，不用文件名或路径代替校验。

### 0.1 不改变的现有契约

- `llama_kv_cells` 仍是唯一 cell 元数据所有者。不能再造一套位置数组、引用数组、独立 token 生命周期回调。
- `llama_memory_i::reclaim_kv()` 仍是有损淘汰入口；iSWA/hybrid 包装继续只向 attention cache 转发。
- Tri 的 fill-first、首次 KV 压力 drain、sticky maintenance、floor exhausted、atomic fallback 顺序不变。
- 默认 Tri 保留比仍为 `3/32`，recent window 为 128，normalized max/union 与 local pooling 保持现有实现；不能因为加入低秩就静默调高保留比例。
- recurrent-only pressure 不调用 Tri，不调用 xKV 来“解决”recurrent 容量短缺。
- `XKV OFF` 必须走旧路径；`Tri OFF + XKV ON` 是用户显式启用低秩、没有 Tri 淘汰的另一配置，不冒充旧路径数值完全不变。

### 0.2 存储交付等级与禁止替代项

| 等级 | 冷段实际存储 | 用途 | 是否完成本次目标 |
|---|---|---|---|
| `reference` | FP32/FP16 基线；允许显式 mixed-codec 消融 | shadow、数值对照 | 否 |
| `tq-factors` | 四路 A/B 均为 TurboQuant；摘要可 FP16 | 因子叠加压缩对照 | 仅完成因子部分 |
| `tq-factors-landmarks` | 四路 A/B 均为 TurboQuant；摘要为经验证的 Q8/TurboQuant 等低比特格式 | 最终生产候选 | 还须通过资源、质量、兼容性门 |

`FACTORED` 仍是存储状态；上述等级由段的 codec descriptor 表示。A-only 量化、B-only 量化是消融配置，不能标记为“四路因子量化已交付”。原格式 `FLAT_TQ` 段允许存在，但必须报告其占比；所有段都跳过低秩不算完成叠加压缩。

禁止用以下方式替代目标：量化重建后的整份 KV 而长期保留 FP16 因子；量化因子之外常驻完整解码 A/B；把完整历史或解码副本转移到 host 后只报告 GPU 下降；通过降低 Tri 保留比、关闭 RERoT/MTP 或增加未申报的淘汰获得“量化收益”。

### 0.3 三种结果必须分开

1. **控制语义兼容**：引用、可见性、位置、回滚、发布和资源生命周期没有改变。
2. **数值质量**：低秩、因子量化、SR 和 Tri 叠加后的质量经过本模型评测；不承诺无损。
3. **资源收益**：与当前 TurboQuant + Tri 基线比较真实内存、吞吐和延迟；不与 FP16 基线比较后再声称是额外收益。

## 1. 总体架构

```text
现有 RERoT / server / recurrent 控制层
                |
                v
llama_kv_cells：唯一语义元数据、序列引用、storage_pos、RERoT tags
                |
                v
共享 KV store：payload 定位与后端存储，不拥有第二份语义元数据
  ├─ HOT / FLAT_TQ：原有 TurboQuant 或原有其他 KV 格式
  ├─ FACTORED：TQ(A_K/A_V) + TQ(B_K/B_V)，按组/层/头描述布局
  ├─ SR 摘要：独立 Q8/TQ 码流 + 必要的误差/fragment 元数据
  └─ 全局有界临时区：解码 tile、重建、摘要生成、分解工作区
                |
                v
reader snapshot + 现有 DDVR layout
                |
                v
低比特摘要打分 -> 按 query / KV head 的 SR 选择
                |
                v
仅解码 A[selected] 和 B[layer/head/tile] -> 重建 -> 全局注意力归一化
```

不要另写一套 RERoT 调度器。不要把低秩实现藏在返回普通 dense 张量的 `get_k()` 中。引入显式 storage/read 接口，让原有 dense 路径作为一个实现，新路径作为另一个实现。

## 2. 第一项工作：把乘法收益变成可执行的字节验收

### 2.1 哪一部分可以相乘

对一份跨层拼接矩阵 `X ∈ R^(n×m)`，采用 `A ∈ R^(n×r)`、`B ∈ R^(r×m)`。在不计 padding、元数据和辅助存储的条件下：

```text
M_X_fp16       = 2*n*m
M_factor_fp16  = 2*(n*r + r*m)
C_lowrank      = n*m / (n*r + r*m)
M_factor_bbit  = (b/8)*(n*r + r*m)
C_product      = C_lowrank * (16/b)
```

这是元素数与位宽的算术关系，不是性能或质量保证。K/V rank、A/B 位宽不同，必须逐项加权后计算，不能统一乘 4。**相对 FP16 的组合倍率不等于相对现有 TurboQuant 的额外倍率。** 已经是 4-bit 的基线不能再把同一次 16→4-bit 收益算两次。

xKV v2 附录 D.5 已给出与 round-to-nearest 量化叠加的实验；它证明的是该组合方向有实证，不是目标仓库的 TurboQuant 因子、量化摘要和全开功能已经复现。本文不把论文倍率写成发布承诺。[^xkv-paper]

### 2.2 统一的实际字节计算接口

一个组有 W 个真实 KV owning layer，每层 K/V 宽度允许不同，一段有 n 个存活 token。建议新增唯一入口：

```text
encoded_matrix_bytes(codec_desc, logical_shape, storage_layout)
estimate_segment_bytes(candidate, allocator_layout)
measure_segment_bytes(published_segment)
```

编码器、估算器、allocator、state writer 必须复用同一个 row-size/block-size 实现。所有乘法与长度转换检查整数溢出。估算器不能把 Turbo4/Turbo2 直接视为 0.5/0.25 byte；包含实际 block norm、scale、码流、padding、row stride、码本/旋转表及其共享计费规则。

```text
M_A = bytes(A_K[n,rK]) + bytes(A_V[n,rV])
M_B = bytes(B_K[rK,sum(D_K,l)]) + bytes(B_V[rV,sum(D_V,l)])
M_L = bytes(实际合法 fragments 的量化 landmarks)
M_aux = row/fragment 索引 + codec 元数据 + 可选误差界/保护项
M_cold_new = M_A + M_B + M_L + M_aux + 冷段原格式例外

M_old_same_rows = sum(同一批唯一 payload 在各 owning layer 的实际原 KV 字节)
saved_fraction = 1 - M_cold_new / M_old_same_rows
```

原基线的 scale/padding 同样计入；aliased layer、共享 prefix 只计一次。比较对象保持同一 survivor set、logical positions 和配置，不靠额外删 token 获益。若保护项原本已存在，两侧统一计入。

分别报告 payload live bytes、allocator reserved bytes、设备峰值与 host 峰值。页未释放、旧密集缓冲区未解除常驻时，不得把 payload 减少等同于 reserved 显存下降。总峰值还包含热区、tentative、分解/编码/重建/压实工作区、capture、旧 snapshot pins、候选新页、graph buffers、模型和 recurrent/brain/hand。

### 2.3 同一算例：因子量化是怎样把 19 MiB 降下来的

假设 W=4、D_K=D_V=1024、n=4096、rK=384、rV=576、每 8 个 token 一个完整合法摘要块。只计算名义 payload，不含任何量化元数据、padding、索引、热区、例外或临时工作区：

| 存储方案 | 原 KV 或因子 | SR 摘要 | 合计 | 相对原 K4/V2 的名义额外倍率 |
|---|---:|---:|---:|---:|
| 原始 FP16 KV | 64 MiB | 0 | 64 MiB | — |
| 原始 KV：名义 K4/V2 | 12 MiB | 0 | 12 MiB | 1.00× |
| A/B FP16、摘要 FP16 | 15 MiB | 4 MiB | 19 MiB | 0.63×，变大 |
| A/B 全部 4-bit、摘要 FP16 | 3.75 MiB | 4 MiB | 7.75 MiB | 1.55× |
| A/B 全部 4-bit、摘要 8-bit | 3.75 MiB | 2 MiB | 5.75 MiB | 2.09× |
| A/B 全部 4-bit、摘要 4-bit | 3.75 MiB | 1 MiB | **4.75 MiB** | **2.53×** |

其中 A_K+A_V 与 B_K+B_V 各有 3,932,160 个元素；摘要共 2,097,152 个元素。因子从 FP16 到名义 4-bit 恰好减少 4 倍，但整个表最后一行相对原 FP16 是 `64/4.75≈13.47×`，不是把所有中间倍率随意相乘。

将此表写成 `test-xkv-memory-model` 的**名义计算测试**，精确核对整数 byte 数。另写真实 codec/allocator 测试，用实际 block 大小得出另一张表。禁止把名义测试通过当成真实显存或 4-bit 摘要质量通过。

### 2.4 逐段收益门必须使用最终量化表示

流程调整为：FP 分解候选 → 因子布局/位宽候选 → 摘要布局/位宽候选 → 字节与质量检查 → 最终编码/实测 → 原子发布。**不能因 FP16 因子比旧 TQ 大，就提前否决尚未估算的 TQ 因子版本。**

- 首个全量化候选为四路 `turbo4_0`；摘要先以 `q8_0` 验证，再测试 `turbo4_0`。这只是试验起点，不能未经校准就写成模型默认值。
- 支持配置允许的 rank/位宽/摘要类型候选；按真实字节选择通过质量门的方案。不动态训练参数，不为了多压一点强行降低 rank。
- 建议开发起始门为 `min_saved_fraction=0.10`，即相对同段原 TQ 至少省 10%；这是工程起点，可显式修改，不是研究结论。
- 段内量化质量指标只用于拒绝明显坏候选；不能替代任务质量验收。摘要低比特若不合格，按已声明候选改用 Q8，或保留 `FLAT_TQ`，并报告实际 profile。
- 不满足收益、可见性、支持矩阵或峰值预算的段保留原格式；不改变 Tri 策略，不关闭 RERoT/MTP。
- `c_eff` 使用真实 fragment 数。碎片增加或 row padding 变多时重算收益；不得假定一直 `n/8`。
- 估算阶段先预检峰值，最终 publish 前按实际分配校验。OOM、数值失败或实际无收益时释放候选、保留原段。
- 日志包括 `skip_reason=small_segment|no_saving|workspace|rank|visibility|unsupported|quant_error|landmark_error` 和请求/实际 codec。

对可近似按行线性计算的情况，令 `R_old` 为旧缓存每行字节、`R_A/R_L/R_idx` 为候选每行成本、`F` 为 B 与固定元数据，则目标最小节省 s 对应：

```text
n * ((1-s)*R_old - R_A - R_L - R_idx) >= F
```

若括号非正，单靠增大 segment 无法过门。此式只用于解释盈亏点；实际 gate 用完整布局和 allocator 计算。不能越过可见性域合并段来摊薄 B。

### 2.5 冷段收益与整机收益分开验收

假设可低秩化部分占原缓存字节 p、该部分实际额外倍率为 C，额外常驻/工作区为 H：

```text
M_new / M_baseline = (1-p) + p/C + H/M_baseline
```

这只是各项不重叠时的解释模型。`p` 按基线字节而非 token 数定义。即使某些冷段达到名义 2.53×，若只覆盖 80% 原缓存、暂忽略 H，整体也仅约 1.94×；实际以 allocator 计数为准。

PR00 建立两种报告：

1. 固定 KV/source/survivor trace 的表示对照：隔离低秩、因子量化、摘要量化各自贡献，不重算 Tri 淘汰决策。
2. 同工作负载与资源配置的真实全开运行：报告语义/质量、TTFT、吞吐、live/reserved/peak、低秩字节覆盖率及原格式比例；不要求不同有损配置生成完全相同的文本。

发布门要求在预先指定的代表性工作负载上，扣除全部辅助开销后相对当前 TQ+Tri 确实有额外收益，并满足质量和峰值预算。只存量化因子、全部段跳过、只减少文件大小或搬到 host，都不能单独算 `compression_goal_met`。

## 3. 数据身份与存储契约

### 3.1 四种身份不能混用

| 名称 | 含义 | 是否可作为长期引用 |
|---|---|---|
| semantic identity | episode/node/run/逻辑 token 次序，以及普通 prefix 的 provenance | 可以 |
| payload identity | 某个真实 K/V 内容实例的稳定 ID，属于 cache/model namespace | 可以 |
| physical cell / factor row | 当前存储槽、A 的行号 | 不可以，pack 后会变 |
| reader virtual position | 当前 reader 的语义视图坐标 | 仅在该 view epoch 内 |

相同 token 文本、相同 storage position 不代表相同 K/V 内容。不同模型的 target/draft 更不能据此合并。只有原生 share-map 明确授权的 payload 才共享。

稳定 payload ID 与 generation 的分配、拷贝、删除语义应由 `llama_kv_cells` 的既有操作集中维护。store 只维护 `payload -> location`，不复制 storage_pos、visibility、seq refs 等语义字段。

### 3.2 建议新增的内部类型（接口契约，不是可编译补丁）

```cpp
enum class xkv_location_kind { hot, flat_quantized, factored };

struct xkv_location {
    xkv_location_kind kind;
    uint64_t segment_id;
    uint32_t row;
    uint64_t storage_generation;
};

struct xkv_snapshot_stamp {
    llama_rerot_view_stamp view; // 复用现有 view stamp
    uint64_t live_epoch;        // 永久保留集合变化
    uint64_t content_epoch;     // 低秩重拟合/重新量化改变表示值
    uint64_t codec_epoch;       // codec/scale/rotation 解释变化
};
```

另设 storage/binding epoch 处理纯搬家和重新绑定。纯物理重排与数值表示变化不是同一件事；首版可保守失效更多缓存，但不得漏失效。

不要直接扩展已有公开 C struct 破坏 ABI；新 stamp 可先做内部组合类型，通过版本化 state/API 暴露。

### 3.3 共享存储句柄

现有构造逻辑会让 `other` cache 共享 `v_cells_impl`，并通过 layer share-map 共享 K/V 张量。低秩后必须升级为共享 **store/owner handle**，不能只保留旧 tensor 指针。

- 同一 payload 的 A/B 只记一份物理内存。
- model layer ID 先映射到 owning layer，再决定 xKV group。
- aliased layer 不重复参与 SVD，不重复释放，不重复计费。
- 不同模型的非共享层保持独立表示，即使 cell registry 共享。
- retention refs 仍归 cells 管理；store 的 snapshot pin 只是缓冲区生命周期保护，不能冒充语义 keeper ref。
- server 继续只保存 run 等逻辑 ID，不保存 A row、cell 或设备指针。

## 4. 分组与低秩表示

### 4.1 分解形式

对组 g 内相同的 n 个 token，沿 feature 维拼接：

```text
X_K = concat(K_pre[l0], ..., K_pre[lW-1], feature_axis)
X_V = concat(V[l0],     ..., V[lW-1],     feature_axis)
X_K ≈ A_K B_K
X_V ≈ A_V B_V
```

A 的行对应 token；B 沿 owning layer / KV head / feature 分片。K 和 V 使用独立因子、独立 rank、独立 codec，不共享一个 A 冒充原方法。

全局保留集合 I 后：`X_l[I] ≈ A[I] B_l`。只删行不要求重新 SVD，也不要求修改 B。生产存储保存 `Q(A)`、`Q(B)`，重建使用 `D(Q(A[I])) @ D(Q(B_l))`，其中 Q/D 是完整编码/解码而非直接把整数 code 相乘。

A 的量化块严格位于同一 token 行内。这样统一淘汰只复制幸存行的原有码流及随行元数据，不重算 norm/scale、不重新量化；幸存 token 的重建值应保持不变。共享的 group codec 不依赖行号重新生成随机量。

### 4.2 当前 hybrid 模型的分组

仓库记载的 full-attention 层为 `3,7,11,15,19,23,27,31,35,39`。建议实验候选：

```text
[3,7,11,15], [19,23,27,31], [35,39]
```

这是“按 attention owning layer 分组”，不是四个连续 Transformer block。论文在其他模型上观察到的低秩收益不能直接外推。必须比较 group=1/2/4，尾组 rank 按实际维度处理。

不得把 recurrent/DeltaNet 状态塞进 KV-SVD。group 还需要按后端设备、RoPE 参数、attention 类型、维度和原生 layer sharing 边界切开。不同配置未经验证不强拼；无法压缩的组继续原格式。

## 5. 冷热分段与可见性隔离

### 5.1 状态机

```text
HOT_WRITING -> HOT_COMMITTED -> SEAL_CANDIDATE
                                      |
                         有界 FP 分解、候选布局与位宽
                                      |
                  因子 TQ 编码 + 摘要编码 + 质量/字节检查
                               |                   |
                            未过门               过门
                               |                   |
                            FLAT_TQ       FACTORED（量化码流）
                                                   |
                                  原子换 handle，旧读者释放后回收旧页
```

生产版不把所有生成 token 永远留在热区。已确认且不再变化的旧段可以滚动封存；这属于对论文主要 prefill 场景的工程扩展，必须单独评测。[^xkv-paper] `reference` 可以发布 FP 因子供调试，但生产压缩 profile 只在最终量化产物通过检查后发布；禁止先长期发布 FP16 再等待未来后台量化。

三种粒度独立：

- factor segment：一个 B 对应的 token 段，决定 B 的摊销成本；
- allocation page：A 行和原格式 KV 的分配/回收粒度；
- SR chunk：选择摘要的粒度，例如 8，但实际可被可见性/phase 边界切碎。

不要把它们全部设置成 chunk=8。不要每接受一个 token 就重做全历史 SVD。

### 5.2 严格的封存条件

仅封存各 owning layer 已完成写入、已 commit、不属于未决 speculative 分支、没有 in-flight 写入的段。

默认不跨不同 model namespace、episode、互不可见 owner/private 域做 SVD。PRIVATE/PENDING 先保留原格式；公开且稳定后再进入候选。不同 run 只有证明可见性等价且已越过 publication/causal watermark 才允许共同分解，否则按 run 分段。

这一条不只是 mask 问题：把不可见 token 放进 SVD 会改变公共 B，进而改变可见 token 的近似值。即使注意力最后 mask 掉了不可见行，也已经引入跨可见域的影响。

同样，不能用未接受的 draft 后缀拟合当前 prefix 的因子或摘要。被冻结的旧-frontier snapshot 仍需保留旧表示。

### 5.3 数据来源的两条路径

**兼容路径 `decoded_hot`：** 对原格式热 KV 做完整 codec 解码，K 去掉已有变换并逆 RoPE，再分解；V 恢复规范特征域后分解。这条路线可以先避免修改所有模型的 pre-RoPE capture 入口，但其输入已经有原 TurboQuant 误差，不能当作论文直接压原始 K 的复现。

**高保真路径 `prerope_capture`：** 在模型完成 K 相关 norm/scale、尚未 RoPE 和 TurboQuant 前，捕获有界 staging 数据。只保留当前待封存范围；capture buffer 计入 peak。不能取未经过 K norm 的线性投影直接当作正确输入。

两个模式分别记日志和做质量 A/B。不得长期留一份完整高精度历史来维持所谓兼容。封存时只对当前受预算限制的段/分解 tile 解码；不能为了量化因子反而长期保留全部 decoded-hot 或 capture 数据。

若输入来自 `decoded_hot`，总误差包括原 KV TurboQuant、低秩截断、因子 TurboQuant、摘要选择与 Tri；标记 `source_codec`。不得把原 KV 与因子两次量化说成质量上只量化一次。

### 5.4 候选发布的具体步骤

1. 固定已提交 payload 集合和安全 watermark，预检新旧并存时的 peak；只 pin 所需源页。
2. 在有界 scratch 中分解；按第 8 节完成可选因子重缩放和四路编码。原段仍可读。
3. 用最终码流解码出的 K/V 检查误差；据此生成/编码对应摘要，检查 source/layout/content fingerprint。
4. 实测因子、摘要、例外和元数据字节；任一 gate 失败，释放候选并保留原段。
5. 在既有维护/事务边界原子发布 handle、codec 与 content epoch；不能逐层提前可见。
6. 旧 snapshot 释放后回收旧 KV backing，立即复用/释放 candidate FP、SVD、编码与摘要 FP 暂存。

读取中的 B、scale 或旋转表不可原地替换。重复封存、取消、抢占和序列销毁必须共用此提交/回收纪律。

## 6. 保留 DDVR，不为每个 reader 重写 K

现有 `build_rerot_q_groups()` 使用：

```text
effective_query_pos = query_virtual_pos + storage_base - virtual_base
```

K 仍位于 storage position；Q 按每个 DDVR group 的 effective position 做 RoPE。

新冷缓存读路径：

```text
量化摘要解码/打分 -> 本 query 的合法 selected rows
TQ decode A[selected] + TQ decode B[layer/head/tile]
           -> reconstruct K_pre -> RoPE(storage_pos)
           -> 现有 optional rotation / Turbo domain 适配
           -> 与现有 DDVR query group 做 dot product
```

不要把 `virtual_pos` 烧进共享 A/B。不要用物理槽位代替 storage_pos。不要同时把 K 转到 virtual position、又对 Q 使用上述 DDVR 位移，否则会重复补偿。

复用当前 RoPE 参数和实现：partial rotary_dim、NeoX pairing、IMRoPE text coordinates、YaRN 频率与幅度缩放、freq_factors。仓库已修复过 IMRoPE pairing 错误，新增路径不得重犯。

### 6.1 摘要必须跟随真实 phase 分段

landmark 的概念仍是 post-RoPE key 的块代表值。但只能在相同 DDVR effective-position group、相同可见性条件下使用它。

- 先用现有 layout 判定 query 能看到什么，再给合法候选打分。
- run 边界、存储空洞导致的 phase-delta 变化、当前 query 的 causal 截断都可能切开 chunk。
- 对部分可见的块，拆成合法 fragment，或从合法行重新构造摘要；不能只在最终 attention 阶段补 mask。
- 可见集合改变时重建受影响摘要；全局保留集合改变时不能继续使用旧 chunk 索引。
- 摘要缓存 key 至少含 owning layer/head、segment、live/content/codec epoch、phase/layout 条件；不是只有 reader ID。
- 把实际 fragment/landmark 数纳入内存模型和性能指标。
- 量化摘要从**最终已编码因子所代表的 K** 构造，而不是无记录地沿用量化前的 FP 摘要；原格式保护项则使用该项真正的存储表示。若试验采用另一来源，必须独立 profile 和质量对照。
- 摘要也有完整 codec/domain 描述；反旋转在打分前完成，或把 query 做严格等价变换。不得用原始 Q 直接乘未知旋转域的摘要码流。
- 基础摘要与 reader/phase fragment 的派生缓存合用有界预算。不能随 reader 数保留无限多份 FP16 摘要；可以按需重建合法 fragment，但候选摘要扫描不能变成每步完整 KV 重建。
- 量化摘要的失效键同时包含 factor content/codec 与 landmark codec；同一 batch 的不同 query 不能共享未经合法性检查的部分块摘要。

## 7. SR 选择与注意力正确性

### 7.1 两套集合

```text
Tri survivor set：所有层统一，决定永久还存哪些 payload。
SR selected set：每个 query、layer、KV head 独立，决定本次重建哪些行。
```

可复用论文的 landmark 点积、GQA 内 max pooling、top-k、outlier 保护，但将候选域约束到现有 reader layout。

对一个逻辑 query/KV head，候选预算在其所有合法 DDVR spans 之间统一比较；不要每个 span 各选一个完整 budget 导致总预算成倍膨胀。

hot/recent、outlier 和动态选择集合去重。同一个逻辑 token 因多 keeper refs 出现多次时只计一次；不能因为向量相同而合并两个不同逻辑出现位置。

### 7.2 推测验证的 query 不能互相影响选择

一次验证 m 个 draft token 时，第 i 个 query 的选择只能使用自身可见前缀与自身 Q。不能把未来 draft query 的分数 max/union 后当成前面 query 的注意力集合。

可以为重建效率计算所有 selected rows 的并集，但每个 query/head 必须保存自己的 membership/CSR mask；并集只是物理 gather 优化。

### 7.3 全局 softmax

热 KV、冷 KV、不同 segment、不同 DDVR group 都属于同一个逻辑 query/head 的注意力归一化。

参考实现维护各部分未归一化状态 `(m_j, z_j, u_j)`：

```text
m = max_j(m_j)
z = sum_j(exp(m_j-m) * z_j)
u = sum_j(exp(m_j-m) * u_j)
out = u / z
```

或者在已有局部 normalized output 与 LSE 的接口上作等价 log-sum-exp 合并。绝不能简单相加或平均各段 attention output。

- V 输出先统一到同一特征域；Turbo inverse rotation 不遗漏、不重复。
- attention sink 按每个 query/head 只计一次，不随 segment 重复。
- softcap、attention scale、causal/SWA mask 顺序沿用当前语义。
- 全 mask、空段、零有效行时处理 `-inf`，不能产生 `-inf - -inf` 的 NaN。
- 不支持拆分状态输出的现有 FA，需要新增 partial-state op，或先做统一分块参考 attention，不能假设现成 kernel 已能合并。

## 8. 核心交付：量化低秩产物，而不是重建后的整份 KV

### 8.1 原 KV codec 不改

保留每层实际 `type_k/type_v`、auto-asymmetric、layer-adaptive、head padding、InnerQ 和所有现有旋转配置。不要从全局 cache type 推导每层类型。

建议增加严格定义的接口：

```text
read_k_canonical(payloads, owning_layer, heads) -> 规范 pre-RoPE K
read_v_canonical(payloads, owning_layer, heads) -> 规范 V
```

接口必须明确返回域；旧解码器返回 WHT-domain 还是 canonical-domain 不能靠猜。K 的完整逆变换还要考虑 optional attn rotation 和 RoPE 幅度缩放。用当前底层实现建立 golden fixtures。

原 KV codec 可以复用实现给因子使用，但配置状态不得共用可变全局变量。热 KV 仍沿用原路径，冷段中的因子不再被当成某个原始 head 的 KV 张量。

### 8.2 四路独立 codec 与 descriptor

K/V、A/B 分别配置：`A_K`、`B_K`、`A_V`、`B_V`，每个发布段都保存实际 codec。拟新增接口如下，属于契约草案而非可直接编译的代码：

```text
factor_encoded_bytes(desc, shape, layout) -> checked byte count
encode_factor_tiles(src, dst, desc, workspace) -> status
validate_factor_encoding(src_reference, encoded, limits) -> error report
decode_factor_rows(encoded_A, row_ids, dst_tile, workspace) -> status
decode_factor_b_tile(encoded_B, layer_head_slice, rank_tile, dst_tile) -> status
```

`xkv_factor_codec_desc` 至少记录：format/version、逻辑与 padded 维度、storage orientation、row/tile stride、block size、量化码本/参数版本、norm/scale 存储、旋转范围与 seed/表指纹、解码输出域、singular-value 分配/重缩放约定。编解码所需数据必须可随 state 恢复，不从当前进程的默认全局配置推测。

低秩 rank 不是原 head_dim。不能把原 head 的 InnerQ、向量 norm 或 WHT 参数原样用于 factor row；不允许 factor 初始化修改热缓存的全局 scale。不能因为一个 dtype 同名，就假定其 attention kernel 能直接消费因子。

TurboQuant 论文区分重建误差与内积估计目标。本方案的首条路径是“解码向量后做 AB”，应先验证重建误差导向的 codec；原 KV 路径中的内积校正不能不经证明直接照搬。仓库实现与论文算法的具体差异仍须在 PR03/10 的 fixtures 中确认。[^turbo-paper]

### 8.3 固定量化方向，保证删除与 SR 读取简单

首版建议采用以下布局，后续更改必须版本化：

| 因子 | 数学形状 | 存储建议 | 量化块边界 |
|---|---|---|---|
| A_K/A_V | `n × r` | token-major，行内沿 rank 编码 | 不跨 token 行 |
| B_K/B_V | `r × sum(D_l)` | 按 owning layer/head 分块，块内可存 `B^T` 的 feature-major 行 | 不跨不相关 layer/head；每 feature 行沿 rank 编码 |

存储 `B^T` 只是布局选择，计算仍是 `A[selected] @ B_lh`。不要额外常驻一份完整 B 转置镜像。tail rank 按 codec 支持的 block 大小 pad；零填充坐标经完整解码后不参与逻辑 GEMM。计入实际 padding，不能假定所有 rank 都恰好适配。

旋转首版可限于受支持的行内 block；非二次幂 rank 不自动等同于支持整行 WHT。group/tail/head/padding 测试通过后才能放开整行配置。小块会改变开销与质量，必须记录实际 block 配置。解码 tile 必须覆盖完整的量化/旋转 block；若使用整行旋转，不能假装一个任意 rank 子片段可以独立解码。

A 的 row scale、norm 和码流跟随 payload 移动。`gather/pack(I)` 对幸存行进行字节复制，不重新编码；B 与不可变码本共享。codec 的随机量不能从压实后的物理 row ID 重新派生，否则同一 token 在 pack 后数值会变。

### 8.4 无训练的因子预处理与误差核算

对截断 SVD `X≈UΣV^T`，可选择：

```text
A = U Σ^alpha
B = Σ^(1-alpha) V^T
```

论文 v2 的形式为 `alpha=1`；对照 `alpha=0.5` 是本方案的量化预处理候选，不是模型训练，也不是已证明更好。[^xkv-paper] `A'=A D, B'=D^-1 B` 的对角重缩放也可作为候选。零奇异值、极小 scale、溢出与截断要显式处理；统计只来自已提交且可见性兼容的源段。

必须先验证不量化时乘积不变，再评估量化后的乘积误差。保存最终已变换的因子及 descriptor；不能在恢复时再把 Σ 或 D 乘一次。

```text
A_hat = A + ΔA
B_hat = B + ΔB
A_hat B_hat - AB = ΔA B + A ΔB + ΔA ΔB
```

因此 A/B 单独的 MSE 不足以代表质量。校验项至少包含 K/V 重建误差、注意力 logits/输出误差、SR 选择变化、最终任务质量；B 的误差会影响多行，不能默认 B 比 A 更适合低位宽。`decoded_hot` 还需计入 source 误差。

head-domain WHT、rank-domain 旋转与 RoPE 分离。首版对 A/B 完整解码到约定坐标系再 GEMM。后续仅当一个兼容的固定正交 R 作用于公共 latent 轴时，才可采用 `A'=AR, B'=R^T B` 的匹配变换来减少逆旋转；逐行各不相同的 R 不能这样抵消，量化误差也不会随旋转自动抵消。位置相关 RoPE 不吸收入固定 B。

### 8.5 开发顺序与最终配置要求

| 阶段 | 因子 | 摘要 | 作用 |
|---|---|---|---|
| R0 | FP32/FP16 A/B | FP16 | 数值 oracle/shadow |
| R1 | A=TQ4、B=FP16/Q8，及 B-only 对照 | FP16 | 分离误差来源，不算最终交付 |
| R2 | A_K/B_K/A_V/B_V 全部 TQ4 | FP16 | 验证真正的因子乘法收益 |
| R3 | 四路 TQ4 | Q8 | 最终 profile 的首个质量候选 |
| R4 | 四路独立 TQ4/TQ3/TQ2 | TQ4 或经验证的其他低比特 | 寻找质量约束下的额外收益 |

`TQ4` 表示目标仓库实际 Turbo4 编码，最终名称/版本按实现登记。不要用普通 RTN/Q4 的结果冒充 TurboQuant 因子已经支持。

R0/R1 可以合并为开发阶段，但生产发布不能止步于 A 量化、B 长期 FP16。R4 的 4-bit 摘要必须实现并评估；若不通过质量门，可以交付 R3 并明确注明，不能宣称达到第 2.3 节最后一行。位宽候选、允许的例外和 gate 在运行前明确，任何变更报告实际 profile。

### 8.6 摘要量化与 top-k 安全措施

摘要单独 codec，不能随着 A/B 量化而假定它已经变小。从最终已编码 K 表示按第 6.1 节构造合法 landmark；在有界 FP 暂存中编码后释放暂存，不留全历史 FP16 摘要镜像。

先完成 Q8 对照，再加 Turbo4。需要独立测量 `FP 摘要→量化摘要` 导致的 top-k/chunk recall、检索关键 token 命中与最终质量变化，不能只测平均向量误差。必要的原格式 outlier/保护项、误差界和 fragment table 全计入字节。

可选的边界候选重评分：若摘要 L 的解码误差上界为 ε，则线性打分满足 `|qᵀ(L_hat-L)| <= ||q||₂ ε`。ε 按实际编码误差保守向上记录，scale/GQA 聚合也纳入区间计算。只有区间接近 top-k 边界的候选才从最终因子重建摘要重新打分。

该重评分是新增设计，不是保证免费或保证任务无损。限定每 query 的重建行/候选数和共享 workspace；达到上限后采用显式确定策略并记录 `refine_cap_hit`，不能仍声称与 FP 摘要 top-k 严格等价。参考重建只消除摘要量化误差，不消除低秩或因子误差。未来 draft query 不参与前面 query 的重评分决策。

默认不得通过每步重建所有冷 K 来规避摘要量化误差。若任务只能靠这样才能通过，当前 profile 不具备 SR 性能资格。

### 8.7 有界解码，不允许双份常驻

生产读取：

```text
低比特合法摘要 -> 每 query/layer/head 的 selected rows
    -> gather 编码 A 行
    -> 解码受限数量的 A 行、当前 B layer/head/rank tile
    -> FP16/BF16 tile，FP32 参考/必要累加
    -> 重建当前 K/V tile，执行 DDVR/注意力
    -> 复用 tile，保留 online softmax 状态
```

工作区按 tile 大小与受准入控制的并发分配，而不是按总历史 n 或 segment 数分配。SR 请求较多时流式分块，不能为每 query/reader 都常驻一套完整解码因子。`dense` 模式亦采用有界分块，不等于永久恢复全部 KV。

B 的解码复用只允许使用**全局共享、有硬字节上限的 tile cache**；默认为关闭或采用明确预算，不得每个段各开一个预算。key 包含 store/segment、B content/codec、tile、设备与输出域；失效遵守 snapshot。它占用的内存必须计入 workspace/cache 和净收益。仅有量化 B 的文件大小下降、VRAM 又放回全份 FP16 B，不算 B 的驻留节省。

禁止常驻：完整解码 A、全部 B 的 FP16 镜像、全部冷 KV、全部 FP16 landmarks。源 KV、FP candidate、编码输出短暂并存只允许在峰值 preflight 后且生命周期有结束点；allocator 缓存保留的 reservation 仍需报告。

### 8.8 三个最先写的 golden tests

1. `full_rank_no_quant`：完整重建对齐源表示，关闭 SR/低比特，隔离基础错误。
2. `factor_tq_decode_then_gemm`：固定同一 A/B，CPU 逐步解码 GEMM 与后端 quantized reconstruction 对齐；包括四路位宽、padding、存储转置和旋转域。
3. `evict_copy_codes`：固定全局 I，比较先解码再取 I 与复制量化行后解码；结果/幸存码流不因 pack 变化，B 不变，语义位置和引用不变。

只有上述测试通过，才把因子融合内核接入真实 RERoT/MTP 图。

## 9. TriAttention 只增加数据适配，不重写策略

“全局统一”在这里指同一共享 KV 所有者域中的各层使用同一最终 resident 集合和行排列；不把不同 person/sequence 的既有预算、公平性与可见性规则改成一个不分域的全局 top-k。保留现有 policy 输出，再统一应用到各层表示。

现有 scorer 的 sampled-head 精确统计、normalized max/union、历史长度口径、3/32 floor、recent 128、local clustering 全部保留。

对 cold key，在触发 reclaim 时从最终 TurboQuant 因子按 sampled layer/head 分块解码/重建规范 K；复用底层接受 pre-RoPE K 的评分函数，不先做 RoPE 再逆回来。不直接给 A 或未反旋转的 factor code 打 Tri 分数，也不为 scorer 常驻一份完整 K。

注意：scorer 使用的 half-layout 与模型原始向量 layout 需要显式适配。partial RoPE 的非 rotary 维度仍按当前函数约定处理。

分块评分要保持**整个候选集合上的每-head normalization**，不能每块独立 z-score。可用两遍流式统计或受控 score buffer。多个 Q sampled heads 对应同一 KV head 时复用一次 K 解码，但分别应用各自统计。

GPU readback 保持 sampled-layer 批处理，不能退回 per-cell/per-head 同步复制。CPU 参考先正确，Vulkan 优化后再减少传输。

生成一份 mutation plan，沿现有引用删除规则执行；全局 survivor/permutation 对所有 owning layers、groups、共享 target/draft store 一致。逻辑引用变化与物理释放分别计数。不是见到低分 token 就无条件抹掉所有 keeper refs，也不是多引用 token 永远不能淘汰。

因子侧逐行复制 A 的码流及元数据，B 不改，不在 pack 中重算量化 scale。更新 locator/live epoch、失效受影响的量化摘要与 fragment 索引；若仅物理搬家且幸存值不变，不冒充 content 变化。确实需要重新拟合/量化时走新段原子发布，不能隐藏在 ordinary pack 中。

长期淘汰使段变短后 B 摊销变差，记录收益退化。只可在可见性等价且已有维护边界下评估重封存；不为了压缩率撤回已经承诺的旧 snapshot，不偷偷改变 survivor 集合。

## 10. 原子更新、MTP 和 recurrent 一起提交

### 10.1 事务必须覆盖的状态

- tentative KV 与已接受 KV；
- sampler/RNG 与 grammar；
- RERoT parser 的 partial opener/list/end-marker 状态；
- tree/run publication、frontier 与 final-fence 控制状态；
- MTP/draft cache、view stamp、必要 checkpoint；
- recurrent/hand/brain delta 的既有提交路径；
- xKV snapshot、暂存行、stamps 与待发布段。

xKV 不改 recurrent 数学，但必须和已有事务边界一致：只提交接受部分，且只提交一次。

### 10.2 推荐执行顺序

```text
在现有安全边界完成上一轮已提交状态
    -> 预留本轮 hot rows、workspace、回滚余量
    -> 固定 reader/content snapshot
    -> draft 与 target 验证，仅写 tentative/hot 区
    -> 每个 query 用自己的 SR 集合和 causal/layout 条件
    -> 校验 stamp 未过期
    -> 原子提交接受前缀的既有控制状态与 KV
    -> 回滚拒绝后缀及其临时 parser/采样/recurrent/MTP 状态
    -> 释放本轮 snapshot pins
    -> 在已有 frontier/maintenance 安全边界处理 reclaim/pack/seal
    -> 发布新 epoch 和下轮 view
```

未接受 token 禁止进入永久 factor、landmark 或 PUBLIC run。发布、topology、语义 shift 或数值 refactor 使旧 stamp 失效后，废弃旧 draft，重新在有效 target 状态上验证；不能只更新指针假装 draft 仍有效。

维护 tick 以 committed token/frontier 驱动。推测 batch 跨越原本会触发维护的边界时切批，避免 speculative batch size 改变 target 算法。

### 10.3 Copy-on-write / pin 规则

- in-flight reader 看到的 segment、A/B 码流、量化摘要、scale/rotation/quant metadata 不原地改写。
- mutation 先 preflight 全部目标与 scratch，准备完成后才发布。
- 所有读者释放前，旧 buffer 不能回收到 allocator。
- 删除 A 行通常不 refit；需要 refit 时写新段，成功后原子换 handle，并提升 content epoch。
- OOM、SVD/因子编码/摘要编码失败、backend 错误必须保留原段有效，或按已有资源失败路径处理；不能半层成功半层旧数据。
- 改变任一 A/B 的位宽、scale、旋转或因子重缩放，按数值表示变更推进 content/codec epoch；仅 layout/地址变化使用 binding epoch。
- 摘要重新量化或重评分配置变化会改变 SR 的数值路径，也必须进入 snapshot/config stamp；不能只给 factor 更新 epoch。
- 量化输出在事务外准备，在安全边界发布。验证事务固定同一套因子/摘要码流；target 与其原生共享 draft 使用同一 owner handle，不各自异步重新量化。
- Byte-preserving A pack 不算再次有损压缩。state restore 必须恢复原码流，不能默认解码后重新编码。

## 11. 资源模型与 auto-fit

不得继续把“缓存容量”压成一个数字。需要区分：

1. logical context 范围；
2. cell registry 可容纳的 resident 行数；
3. hot/tentative 区可写槽数；
4. 真实 KV/因子设备字节预算；
5. recurrent 的 people/pens/brain/hand 容量。

保留旧 API 在 XKV OFF 时的语义；XKV ON 增加内部 admission snapshot，例如 `free_rows/free_hot_rows/free_bytes/safe_next_ubatch`，审计 server 每个用 `get_kv_capacity()` 推导可写 batch 的位置。

`common_fit_rerot_capacities` 和 `memory_breakdown()` 纳入上述全部内存；不能把论文压缩倍数乘到 total-kv 上就宣布容量扩大。rank 收益取决于输入，动态 admission 必须按当时真实预算处理，不能依赖“所有段都可压缩”。

为热区、speculation、QR/SVD、重建和 native memmove 保留明确预算。无法满足新 batch 的 hot/workspace 需求时，先在安全边界按既有压力策略处理，再限制 ubatch；不要在 decode 中途耗尽。

prefix cache、RAM swap 和 state restore 的容量核算按唯一 payload/store 计费，不能每个 alias 重复计算；snapshot 双版本也不能漏算。

### 11.1 实际减少常驻内存的条件

量化码流发布后，旧 full-size dense tensor 必须解除常驻/归还可复用页；只改 `get_k()` 的逻辑路径、仍保留按原完整 context 分配的 backing，不算显存节省。对不能细粒度释放的原大缓冲区，先完成 page/store 拆分或整体替换；共享引用和 pins 会推迟释放，照实报告。

`factor_bytes` 要拆为四项 A_K/B_K/A_V/B_V，再拆码流/scale/padding；landmark、fragment index、FP exception、decoded-tile cache、source capture、candidate 编码暂存分别可观测。全局共享表只计一次。统计按 allocation ID 去重，不能把已纳入 workspace 的 tile cache 再加一遍，也不能漏掉旧版 pins。

对同一配置分别输出：逻辑 payload 压缩比、allocator reserved 压缩比、运行峰值比。state 文件更小与 GPU 更小是不同指标。host-assisted 分解/编码/重评分的 host RAM 和传输必须报告，禁止通过迁移整份历史制造“乘法节省”。

### 11.2 组合目标的自动检查

新增报告 `compression_gate.json`，内容包括固定基线指纹、实际 storage profile、低秩覆盖的基线字节比例、四路有效 bit/element、摘要有效位宽、source 格式、fixed-trace 与实际全开内存结果、峰值预算和质量 verdict。

`compression_goal_met` 至少要求：非零且达到预设覆盖门的合格量化因子段；四路 codec 与请求一致；没有完整解码镜像；同口径相对 TQ+Tri 的净收益达门；工作区/热区峰值不超额；质量和全部兼容性测试通过。覆盖率/收益阈值由 PR00 的目标工作负载明确写入版本化配置，未设置或未实测的门显示 `not_evaluated`，不能默认通过。

如果只有部分冷段省了而整体 reserved/peak 没有改善，报告局部收益和未达标原因，不能发布为“已达到整体乘法压缩”。

## 12. 后端实施

### 12.1 CPU 参考

提供小规模 deterministic reference：canonical codec、SVD、四路 factor TurboQuant 编/解码、量化摘要、完整/选择重建、SR、DDVR 分段 attention、softmax 合并。CPU SVD 可使用显式依赖的成熟线性代数库；离线 golden fixture 可用 NumPy/PyTorch。误差对照锁定同一候选因子和 codec，不能把不同分解随机种子的差异归咎于后端。

CPU oracle 用于对齐，不等于生产默认把 GPU 全缓存搬回 CPU。所有 reference/host-assisted 模式必须显式记录。

### 12.2 Vulkan

先实现可正确执行的独立算子，再融合：

- 四路 factor TurboQuant encode，包含 layout/padding、norm/scale、旋转与码流写入；不是只实现 decoder；
- encoded A row gather / decode，与 B layer/head/rank tile decode；
- selected GEMM / reconstruction，先对齐独立解码再融合；
- storage-position RoPE 与域适配；
- 最终 factor 表示生成合法 landmark、Q8/TurboQuant 编/解码、phase-aware score/top-k；
- 可选、有界的 landmark 边界重评分；
- indexed hot/cold attention 与 FP32 online softmax；
- A 码流及 row metadata 的字节保持 pack，复用现有 memmove 事务纪律。

后端 op 的输入应通过 ggml tensor/显式 descriptor 表达；不要把生命周期不受 scheduler 管理的裸 host pointer 塞进图。对容量做 bucket，使 selected 数量/view 内容变化主要更新 tensor data，而不是每 token 重建图。

布局/数据 epoch 更新与图拓扑更新分开。buffer 地址变化必须更新 binding；capture 的图不能继续引用已释放 workspace。

融合前的必备中间实现是“有界解码 tile + GEMM”；已经可以实现存储收益，不以完成融合为借口保留完整 decoded B。融合后必须验证使用同一码流、同一选集与累加语义，吞吐收益单独报告，不把节省位宽等同于速度倍数。

能力矩阵需覆盖 codec 的 encode、decode、selected-reconstruct、pack 和 state。CPU 具备某个 factor dtype 不能自动在 Vulkan 开启；host-assisted 量化仅作为显式过渡，生产 profile 要标明驻留与传输。

### 12.3 分解器不是免费的移植

官方 `torch.svd_lowrank` 不能直接在 Vulkan 中调用。建议独立 `factorizer` 接口：CPU reference、Vulkan rSVD、可选 CUDA library 后端。

rSVD 过程：`Y=XΩ -> QR -> 可选 power iterations -> C=QᵀX -> 小矩阵 SVD -> 截断 -> A/B`。随机种子锁定到已提交 segment/配置，避免不可复现实验。QR/小 SVD 必须做稳定性和残差检查。

Vulkan 生产可复用 ggml 的矩阵乘法能力，另实现稳定正交化与小矩阵分解；不能默认 NVIDIA 库可用。host-assisted 小矩阵分解可作为显式过渡模式，但必须报出 readback、host workspace 和耗时，不能称为 VRAM-only 全 GPU。

不要未经验证使用低精度 Gram 特征分解代替稳定 SVD；Gram 会放大条件数问题。失败时保留原格式段。

### 12.4 CUDA 增量

另外补 RERoT backend 能力、DDVR attention、Turbo codec 与 xKV kernels，完成同一套 CPU oracle 对照后才打开 capability gate。不能只把 `UNSUPPORTED` 改成 `SUPPORTED`。

## 13. 保存、恢复、共享与 context shift

在现有 versioned RERoT episode/state 机制上增加 XKV capability/version，不旁路原来对不完整旧格式的拒绝。

新增持久化项：group/owner map、唯一 segment、稳定 payload 对应关系、A_K/B_K/A_V/B_V 的原始码流、各自 codec 描述、live row 映射、必要 source/phase 参数、content/live epoch、实际 storage profile 和配置指纹。量化因子不能只保存类型名，必须保存 norm/scale、rotation/seed/表版本、逻辑/padded 维度、转置与 tile 布局等解码所需数据。

量化基础摘要及其 codec/source fingerprint 应保存；可派生的 reader fragments 可失效后重建。不得在 state 中偷偷放一份完整 FP16 因子/摘要作为恢复捷径。恢复时码流字节保持，不解码再编码；不支持该 codec 的后端应明确拒绝，或执行用户显式选择并重新计费/验收的迁移，不静默扩展成 FP16。

state 还需保存 factor 预处理方案、摘要重评分配置等影响数值的参数。旧解码 tile cache 不持久化；恢复后清空并受同一全局预算约束。

继续沿用现有 server 保存 parser/sampler/MTP/hand seed/queue/tree 的通路。restore 先验证全部 fingerprint、长度和引用闭包，再分配预算并重建 binding，最后原子发布。读失败不留半个 episode。

模型、LoRA、RoPE、Tri 校准、factor codec/config 不匹配要有明确策略：拒绝或按既有安全 re-prefill 机制处理，不偷偷按另一格式解码。

- `seq_cp`：共享已封存前缀的 store handle；后续写入分开。
- `seq_rm/seq_keep/cancel`：只删对应语义引用；最后可释放引用和 pins 消失后才释放物理页。
- RERoT semantic context shift：按现有 run 裁剪、view epoch 更新；不把 A row 号重新解释成 token position。
- 普通 `seq_add/seq_div`：按原有位置变换语义重建/重绑；未实现的结构不能静默接受。
- RAM demote/restore：按唯一 backing store 序列化，别为每个 reader 重建整份 KV。
- 老 frontier 的 recurrent rewind 继续服从原有安全 re-prefill 条件；不能因 xKV 能重建某些 KV 就声称 recurrent 也能回退。

## 14. 按 PR 拆分：因子量化前移，不再等到最后才补

继续保留 PR00–13 的编号。相较 r1，PR03/04 就引入 CPU 因子 codec 与全量化 shadow，PR05 的 allocator 从一开始按最终量化布局核算；PR10 负责后端完善和性能，而不再是第一次实现因子量化。

| PR | 实施内容 | 主要文件/入口 | 必须通过的验收 |
|---|---|---|---|
| 00 | 冻结 TQ+Tri 基线、模型/校准/驱动/build；固定 survivor trace、名义与真实 memory calculator；定义覆盖/净收益/质量门 | 新建 `docs/xkv/`、`scripts/xkv/`、memory-model fixture | 64/12/19/7.75/5.75/4.75 MiB 名义算例准确；实际格式另表；四路 codec/摘要候选的盈亏点可见 |
| 01 | 配置、storage profile、能力矩阵、字节统计接口与空图路径 | `include/llama.h`、`src/llama-cparams.h`、`common/arg.cpp`、构建文件 | XKV OFF 无新增大 buffer；请求/实际 profile 明确；reference 不算达标 |
| 02 | shared store handle、稳定 payload、binding epoch；扩展 locator 不复制语义元数据 | `src/llama-kv-cache.*`、`src/llama-kv-cells.h`、`src/llama-memory.h` | legacy cp/rm/pack/share/MTP alias 不悬空、不重复计费 |
| 03 | canonical adapter；CPU factor codec descriptor、bytes/encode/decode；A token-major/B 分片转置；golden fixtures | 拟新增 `src/llama-xkv-codec.*`、`tests/test-xkv-codec.cpp` | 原 KV 不受 factor 参数污染；四路 TQ2/3/4、padding/rotation 对齐；真实 byte estimator 与 encoder 一致 |
| 04 | factorizer、group owner map、FP shadow、重缩放对照；四路 TQ shadow 并计算乘积误差 | 拟新增 `src/llama-xkv-factor.*`、`src/llama-xkv-cache.*` | full-rank/no-quant 对齐；TQ 因子解码-GEMM 有参考；A-only/B-only/A+B 消融；失败原子性 |
| 05 | hot/flat/factored 分段 allocator；最终 TQ 字节 gate、peak preflight、原子封存与有界 decoded tile cache | xkv cache + memory/fit | 不因 FP16 过大提前否决 TQ；无完整 FP 镜像；旧 dense backing 回收；所有跳过有原因 |
| 06 | 解码最终 TQ 因子接入全量分块 DDVR reader，合并 hot/cold softmax | `src/llama-graph.*`、拟新增 `src/llama-xkv-reader.*` | 多 reader/virtual 位移、sinks/softcap/空 mask 对齐；临时量不随历史线性膨胀 |
| 07 | SR 合法 fragment、per-query GQA top-k；CPU 摘要 Q8/Turbo4 codec 与有界可选重评分 | reader + codec + CPU/backend ops | SR=all 对齐 PR06；摘要来源为最终码流；4-bit 摘要质量独立测；未来 draft query 无影响 |
| 08 | Tri 适配最终因子；统一 survivor plan；A 码流字节保持 pack；量化摘要失效 | `src/llama-triattention.*`、`reclaim_kv/compact` | 原 3/32/128/sticky/metrics 不变；pack 不重新量化；recurrent-only 不触发 |
| 09 | MTP/RERoT 事务、factor/landmark stamps、COW/pin、accepted-only sealing | `src/llama-context.*`、speculative 通路、`tools/server/server-context.cpp` | 接受 0/部分/全部、publish/shift/final fence/取消无污染；编码失败回滚 |
| 10 | Vulkan 四路 factor 与 landmark encode/decode；量化 tile 重建路径、能力/布局版本 | xkv codec + `ggml/src/ggml-vulkan/` shader/注册 | CPU/Vulkan 同码流解码对齐；canonical-GEMM 对照；无全量 A/B FP16 镜像；decode cache 有硬上限 |
| 11 | versioned state/swap/prefix/preemption/people/pens/KV admission，码流原样恢复 | memory state、`common/fit.cpp`、server-rerot/context | 三分支/save-load/RAM demote/取消/恢复通过；无隐式解码重编码；真实字节与能力校验 |
| 12 | Vulkan native 分解、直接低比特重建/selector/attention/pack 优化与图复用 | `ggml/src/ggml-vulkan/` 及 factorizer | 对齐 CPU；封存后 FP 暂存释放；无 per-token 全缓存 D2H；peak 与性能达门 |
| 13 | 四路因子与摘要量化消融、故障注入、全开长运行、allocator 报告和发布文档 | 新测试/脚本、现有 RERoT/Tri 测试 | `compression_gate.json` 完整；不能以局部 payload 数字冒充全局达标；未通过的 4-bit 摘要配置明确标注 |

实施依赖：先 PR00–04；PR05/06 获得有界的真实 TQ 因子读取；PR07/08 加 SR 与淘汰；PR09 完成事务；PR10/11 完成后端与生命周期；PR12/13 完成性能与发布。可拆成更小提交，但每次对照只改变一个误差来源。

PR09 是真实 RERoT/MTP 生产启用前置条件；PR10/11 是量化生产 profile 前置条件；PR12 不能用融合或吞吐掩盖 PR06/07 数值错误。R3/R4 未通过门时保留开发模式，不将 FP16 因子版标记为本次目标已完成。

## 15. 测试计划

### 15.1 数学与表示层

- full-rank/no-quant；低 rank；K-only、V-only、K+V。
- group=1/2/4 与不完整尾组；非连续 attention layer；alias layer。
- head_dim=128 与仓库 partial IMRoPE 的 256/64；不同 GQA 映射。
- segment/chunk/page 非整除；空段、一行、rank>=n、rank padding。
- Turbo2/3/4、Q8/F16 混合；每层 adaptive 类型；A_K/B_K/A_V/B_V codec 独立。
- 同一固定因子做 A-only、B-only、A+B；同一最终因子做 FP16/Q8/Turbo4 摘要对照。
- 因子 rank 与原 head_dim 不同，B 转置布局、非二次幂 rank、block padding、零向量、极小/极大 norm。
- 奇异值分配/对角重缩放在不量化时保持乘积；量化后按乘积而非仅 factor MSE 验收。
- 名义计算器逐 byte 核对第 2.3 节；真实 estimator 对齐实际 encoder/allocator；别名、metadata、padding 不漏算。
- `pack` 前后幸存 A 码流字节不变、B 不变、解码值不变；只更新位置映射与适用的 epoch。
- 摘要 key 的 live/content/codec/phase 失效、top-k 边界重评分、cap hit 和 GQA 区间处理。
- 存储位置非连续、压实重复执行、负 effective query offset、大合法位置。
- global softmax 分块等价、sink 只计一次、softcap、全 mask。

FP32 小规模 oracle 可先设置约 `atol=1e-5, rtol=1e-4` 作为起始门；FP16/量化分别设门。阈值需按数据尺度和现有内核误差确定，不能用一个宽松阈值掩盖位置错误。量化对齐应比较同一 codec 的表示，而不是要求还原原 FP16 向量无损。

### 15.2 语义回归

- 8K shared prefix 分成三支，取消一支，另外两支继续。
- 同一公共 run 对不同 reader 使用不同 virtual order。
- PUBLIC/PENDING/PRIVATE、parser 标记跨 tokenizer token 的发布边界。
- Tri 后 sparse storage gaps，view 重新 dense pack；反复 pack 后保留逻辑身份。
- archive/exec/parked 多引用不增加重要性、不重复存储或输出。
- final acquire fence 与恢复用户 grammar/tool/JSON schema。
- context shift、pen yield/resume、idle demote/active preemption、多 person 隔离。
- 改变不可见 private run 或未接受 draft，不得改变受保护 reader 的表示与结果。
- 量化段在三分支中共享只记一份；一次 B codec 更新不能污染仍 pin 的旧 reader。
- factor/landmark save-load 后码流一致；解码 tile cache 清空后结果不变，cache 命中/替换不改变语义。
- 增加 reader 数不能隐式增加无限 FP16 摘要/因子副本；可合法存在的 snapshot 双版本有预算与释放点。

### 15.3 推测解码

接受 0、1、部分、全部；在拒绝后缀中放入可能触发 parser/publication 的特殊内容；在验证期间触发即将发生的 pressure、shift、publish 和 final fence。

对照的是 **同一压缩配置、同一维护时钟下的串行 target**，而不是原始无压缩 target。核对 logits/接受逻辑、cache 内容、parser、sampler、recurrent checkpoint、frontier 计数；不仅仅检查最终文本“看起来正常”。

### 15.4 故障注入

在新段分配、QR、SVD、四路 factor encode、landmark encode/重评分、memmove、codec descriptor 发布、metadata publish、state read 中分别注入失败。测试 A 编码成功而 B 失败、K 成功而 V 失败、因子成功而摘要失败、实际字节超出估算等场景。旧状态仍有效或按照既有硬失败路径结束；不能留半更新共享缓存。

恢复被截断的 state、错误 codec version、错误模型/RoPE/校准指纹、过期 snapshot、allocator 重用地址。用 ASAN/UBSAN 与压力测试定位越界和悬空；Vulkan 验证层按项目环境显式开启。

### 15.5 性能与质量矩阵

每个消融先用固定输入、source、rank、survivor trace 隔离误差；随后做真实生成。以下 `F` 指四路因子，不把模型权重量化算进 KV 收益：

```text
B0：原有 TQ + Tri，作为主要资源/质量基线
R0：相同 survivor trace + 低秩 F16 因子，dense 分块重建
R1：R0 + 仅 A TurboQuant / 仅 B TurboQuant，分别对照
R2：R0 + 四路因子 TurboQuant，dense 分块重建
R3：R2 + SR，FP16 摘要
R4：R3 + Q8 摘要
R5：R3 + Turbo4 摘要（与 R4 分开，不串联量化）
E0：当前原有 TQ + Tri + RERoT + MTP，真实全开基线
E1：完整 xKV-SR + 四路 TQ + Q8 摘要 + Tri/RERoT/MTP
E2：完整 xKV-SR + 四路 TQ + Turbo4 摘要 + Tri/RERoT/MTP
```

额外保留 Tri OFF、RERoT OFF、MTP OFF 对照用于定位，不以关闭功能代替最终兼容。低比特不同档位、不同 source、不同 rank 单独标记，不把三项同时改变后的差异归咎于某一个。

测试短 prompt、长 prompt/短回答、长生成、多轮 exact detail/coding、RERoT 多分支、固定设备预算下吞吐。记录 TTFT、prefill/decode tok/s、p50/p95 latency、seal/quant/selector/refine/reconstruct/reclaim/pack 分项、draft acceptance、live/reserved/peak、正常完成率和任务质量。

不承诺叠加后无损，也不承诺吞吐按倍率增长。质量门由项目在 PR00 明确；失败时修实现、重新选择已声明参数或标记配置不合格，不在运行中偷偷改变 Tri residency、关闭保护功能或使用未计费 FP16 副本。

### 15.6 资源节省必须有自动化测试

- `test-xkv-memory-model`：名义数学与实际 row-size 分开测试；float 表格由整数 byte 结果生成。
- `test-xkv-factor-encoding-size`：四路 factor 和 landmark 的 estimate/encode/state bytes 对齐，包含所有 pad/scale。
- `test-xkv-no-dense-mirror`：完成封存且释放 pins 后，无完整 decoded A/B/KV/landmark；旧大 KV backing 不继续常驻。
- `test-xkv-workspace-bound`：扩大历史、segment 和 reader 数时，decode tile workspace/cache 仍不超过配置硬上限；必要 metadata 增长另报。
- `test-xkv-quantized-pack`：反复淘汰/压实不会重新编码幸存因子；无隐式 FP16 重建历史。
- `test-xkv-state-code-stream`：恢复保持原始码流、共享去重与真实预算；错误 descriptor/version 拒绝。
- `bench-xkv-compression-gate`：固定 trace 与 E0/E1/E2 实跑均输出净收益、覆盖率、质量和峰值 verdict；`not_evaluated` 不能当 pass。

测试名均为拟新增。名义 4.75 MiB 的测试只证明算术；最终是否达到约 2.53× 的冷段额外收益，要由真实 codec、fragment 结构及任务质量决定。

## 16. 建议新增的参数与指标

以下参数尚不存在，属于拟新增接口：

```text
--xkv off|shadow|dense|sr
--xkv-storage-profile reference|tq-factors|tq-factors-landmarks
--xkv-group-size N
--xkv-rank-k N
--xkv-rank-v N
--xkv-segment-tokens N
--xkv-chunk-tokens N
--xkv-sr-budget N
--xkv-source decoded-hot|prerope-capture
--xkv-factor-a-k TYPE
--xkv-factor-b-k TYPE
--xkv-factor-a-v TYPE
--xkv-factor-b-v TYPE
--xkv-factor-balance upstream|sqrt|diagonal
--xkv-landmark-type TYPE
--xkv-landmark-refine none|boundary
--xkv-landmark-refine-max-rows N
--xkv-workspace-mib N
--xkv-decode-cache-mib N
--xkv-min-saving FRACTION
--xkv-min-factor-coverage FRACTION
--xkv-factorizer cpu-reference|vulkan|vulkan-hybrid|cuda
```

参数可收敛到版本化配置。rank 是每组 rank，不是每层或每头；group 尾部验证上限；logical rank 和 padded rank 分别记录。SR budget 与 Tri retained budget 独立。

`workspace-mib` 是 XKV 临时区总预算，`decode-cache-mib` 为其中一个全局子预算，不是每个段/reader 各分一份。分解峰值、hot reserve、source capture 等不能因没有包含在此临时子区就漏记到总 admission。`min-saving` 定义为第 2.2 节的相对同段原格式节省比例，不是相对 FP16。

首个完整候选：四路 `turbo4_0` + `q8_0` 摘要；第二候选只把摘要换为 `turbo4_0`。因子/摘要类型名称按实际实现能力登记。上述为拟议试验配置，不是当前 CLI 可用的命令，也不是目标模型的已验证默认值。

请求 `tq-factors-landmarks` 时四路因子须为受支持的 TurboQuant，摘要须为受支持的低比特类型。请求与类型矛盾直接报错，不静默标为成功；允许的逐段原格式 fallback 与实际覆盖率照实报告。消融中的 FP16/Q8 B 使用 `reference` 或显式实验 profile，不伪装成四路 TQ 已完成。`XKV OFF` 禁止新建因子/摘要大 buffer。

所有改变数值或选择的参数都进入 fingerprint/stamp。候选位宽集合、质量阈值、允许的摘要例外和 fallback 规则在配置中明示；自动选择不得越出集合，也不能改 Tri 预算。

至少输出如下指标；同一 buffer 不重复计费：

```text
xkv_requested_profile / xkv_effective_profile
xkv_hot_bytes / xkv_flat_bytes
xkv_factor_ak_bytes / xkv_factor_bk_bytes
xkv_factor_av_bytes / xkv_factor_bv_bytes
xkv_factor_payload_bytes / xkv_factor_metadata_bytes / xkv_factor_padding_bytes
xkv_landmark_payload_bytes / xkv_landmark_metadata_bytes / xkv_landmark_exception_bytes
xkv_index_bytes / xkv_codec_shared_bytes
xkv_decode_tile_cache_bytes / xkv_capture_bytes / xkv_candidate_bytes
xkv_workspace_peak_bytes / xkv_snapshot_pinned_bytes
xkv_allocator_live_bytes / xkv_allocator_reserved_bytes
xkv_device_peak_bytes / xkv_host_peak_bytes
xkv_unique_payloads / xkv_aliased_payloads
xkv_baseline_same_rows_bytes / xkv_factored_baseline_byte_coverage
xkv_factor_quant_ratio / xkv_net_extra_compression_ratio
xkv_seal_seconds / xkv_factor_quant_seconds / xkv_landmark_quant_seconds
xkv_select_seconds / xkv_refine_seconds / xkv_reconstruct_seconds
xkv_segments_sealed / xkv_segments_skipped{reason}
xkv_sr_selected_rows / xkv_sr_fragments / xkv_effective_chunk_size
xkv_landmark_refine_rows / xkv_landmark_refine_cap_hits
xkv_spec_stale_total / xkv_transaction_abort_total
xkv_compression_goal_met
```

`factor_quant_ratio` 是同形状 FP16 因子对实际量化因子字节的比值，不含把低秩再算一遍。`net_extra_compression_ratio` 按报告指定的同口径 TQ+Tri 基线与新存储计算；同时给 live/reserved 两种值，不能混用。覆盖率分母按基线字节，不按段数。无可比数据时返回未评估，不给无穷大或默认 pass。

保持已有 `tri_*` 和 `rerot_*` 的计数含义；格式转换不算 Tri 淘汰，编码失败不算用户主动取消。

## 17. 开工与发布

### 17.1 建议的开工命令

这些命令由开发者在本地仓库执行；本方案没有替用户创建分支、修改远端或部署。

```bash
git fetch origin
git worktree add -b feat/xkv-sr ../atomic-xkv-sr \
  5298ea46061c4780d173de866bdbba42f5918135
cd ../atomic-xkv-sr

cmake -S . -B build-xkv -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-xkv -j
ctest --test-dir build-xkv --output-on-failure -R 'rerot|triattention|kv-cells'
git diff --check
```

正式开始前确认当前开发分支是否已有需要保留的新提交；上述 worktree 固定审阅基线只是为了实验可复现，不是要求回退主开发分支。

不要生成不可回溯的“全开”首版。PR00–04 先获得 baseline、identity、codec 和四路量化 shadow 的可比较结果，再按 PR 表推进；不能等 PR10 才发现量化布局不能 pack 或字节计算错误。

开发者执行顺序：先跑旧基线回归；加入名义/真实 memory tests；使用小固定 factor fixtures 对齐 CPU；完成因子量化 shadow；再接 allocator/DDVR/SR。每阶段保存 source、rank、codec 与结果指纹，不直接用不同随机因素的两次生成比较底层 codec 正确性。

### 17.2 最终发布清单

- XKV OFF 与旧基线隔离通过。
- 全开 RERoT/shared refs/virtual positions/MTP/TurboQuant/Tri，不靠禁用功能通过测试。
- 所有受支持后端/模型组合有明确矩阵；未支持组合拒绝而非静默降级。
- 四路 A_K/B_K/A_V/B_V 的 TurboQuant 编码、存储、读取、pack、恢复在生产后端通过；不止量化 A。
- 摘要 Q8/Turbo4 路径已实现并评测；实际交付档位明确。4-bit 摘要若未达质量门，不宣称名义 4.75 MiB 档已达标。
- 真正释放或复用旧 dense backing；无完整 decoded A/B/KV/landmark 镜像；有界 B tile cache 计入预算。
- FP candidate/capture/分解与编码暂存在安全生命周期结束后释放/复用；全局预算不随段/reader 数复制。
- 真实 allocator 统计与报告的显存收益一致，峰值包含 source/candidate/pin/workspace；host 迁移另报。
- 同 survivor/source 对照显示量化因子贡献；真实全开运行通过覆盖率、净收益与质量门，不靠额外淘汰。
- 名义算例、真实 codec bytes、reserved 显存、整机峰值四类报告分开；`compression_gate.json` 无未评估的必需项。
- 同压缩 target 的 speculative 接受、回滚与 serial 对照通过。
- state/swap/preemption/共享分支/semantic shift/grammar/final fence 回归通过。
- 质量和性能相对当前 TQ+Tri 基线通过预先约定的门。
- build、library、model、calibration、config、driver 指纹进入实验报告。

### 17.3 代码量的工程预算

这次范围不适用“单卡 CUDA、单序列原型约 7K 行”的中心估算。按 CPU + Vulkan、完整缓存生命周期、factor codec、MTP/RERoT 事务和测试交付，建议预留：

```text
实现/实质改写：约 10,000–18,000 行
测试/实验工具：约 6,000–10,000 行
合计：约 16,000–28,000 行
```

这是沿用原版的模块级预算，不是测得的补丁规模，也不是本次加强量化目标后的封顶报价。原预算已经包含 factor codec；此次将四路量化、摘要量化、有界解码与字节验收变成必需项，不应简单把整套 codec 再重复计费。Vulkan encoder/稳定分解器、摘要重评分、更多模型路径和生命周期耦合可能使其超出区间。PR00–04 完成后按实际接口重估；新增 CUDA/RERoT 后端另估。

## 审阅依据

下面的仓库路径和现有功能陈述沿用原版静态审阅记录，以开头固定提交为准。本次修订直接编辑已提供方案，并复核下列论文 HTML 版本；没有重新构建目标仓库、执行模型评测或将源码改动提交远端。

- `src/llama-kv-cache.h`：reclaim/compact/state、RERoT resolve/layout、共享缓存与 graph 接口。
- `src/llama-kv-cache.cpp` 构造函数：`other`/`v_cells_impl`、layer share-map、Turbo auto-asymmetric 与 adaptive 类型。
- `src/llama-kv-cells.h`：cell 元数据所有权、copy/set、pack。
- `src/llama-rerot.h`：稳定逻辑 ID、reader view、DDVR groups/entries、visibility。
- `src/llama-graph.cpp`：`select_variant`、`build_rerot_q_groups`、`build_attn_rerot`、graph span reuse。
- `src/llama-triattention.h`：scorer、canonical scoring 函数、全局聚合、RERoT pressure/retention 契约。
- `tools/server/server-rerot.h`：token parser、runtime、view stamp、publication、state fingerprints、archive/final-fence。
- `AGENTS.md`：已记录的 Tri 产品契约、Vulkan/Ornith 校准注意事项、生产验证说明；其中历史验证不能替代本次测试。
- xKV 官方 `xKV_SR/models/kv_cache_xkv.py`：A/B 形状、跨层拼接、SVD wrapper、landmark/outlier 基础实现。
- xKV 论文 `arXiv:2503.18893v2`（2026-05-27）：pre-RoPE 因子、selective reconstruction、GQA landmark selector，附录 D.5 的 RTN 量化联合实验与 D.6 的因子/摘要内存核算。[^xkv-paper]
- TurboQuant 论文 `arXiv:2504.19874v1`：向量量化的旋转、重建与内积目标。本文没有将原 KV 上的质量结果当成因子量化保证。[^turbo-paper]

本文的四路因子 codec 布局、量化摘要门、事务/域隔离、滚动分段、有界 tile cache 和目标仓库后端接入均为拟议工程设计，不是论文已提供的全套实现。第 2.3 节数字为明确参数下的独立存储计算。

[^xkv-paper]: Chi-Chih Chang 等，*xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction*，`arXiv:2503.18893v2`，2026-05-27。相关位置：§3.1–3.2、附录 B.1、D.5、D.6，以及 §7 长生成限制。本次复核 HTML 版本，不将其吞吐或质量数字外推到本工程。

[^turbo-paper]: Amir Zandieh、Majid Daliri、Majid Hadian、Vahab Mirrokni，*TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*，`arXiv:2504.19874v1`。论文算法提供方法依据；目标仓库的实际编码约定、block 布局与后端支持以实现和 golden tests 为准。

## r2 修订摘要

1. 因子 TurboQuant 从后期可选项改为最终生产交付必需项；A-only/B-only/FP16 仅保留为对照。
2. 内存 gate 改为评估最终量化表示；补齐乘法收益、基线去重、名义/实际算例和整机覆盖率计算。
3. 增加四路 codec descriptor、A 行内量化/B 分片布局、字节保持淘汰、独立位宽与无训练重缩放。
4. 摘要增加 Q8/Turbo4 存储与独立质量验收；限制候选重评分、fragment 缓存和解码 tile cache。
5. 量化产物接入 RERoT/DDVR、Tri、MTP snapshot、state 与 allocator 生命周期；不改变全局淘汰策略。
6. PR03/04 前移 codec 与全量化 shadow，PR10 改为后端完善；补充发布 gate、错误注入与禁止双份常驻测试。
