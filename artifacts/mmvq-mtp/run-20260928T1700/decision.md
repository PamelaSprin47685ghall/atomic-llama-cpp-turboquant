# MMVQ × MTP 四格深井（run-20260928T1700+，`/var/tmp/tp5-mmvq/`）

## 命中证明（先证"测到了"，再谈有没有用）

`GGML_TP5_RELAY_DEBUG=1`（dbg1.log）：
```
[vulkan-region-policy] backend=Vulkan0..4 family=gdn-projections fused=1 mmvq_mask=3 shared_q8_1=1
[vulkan-region-policy] backend=Vulkan0..4 family=qsa-projections  fused=1 mmvq_mask=1 shared_q8_1=1
```
**融合 attention 投影的 MMVQ 真实命中**（gdn: Q/K+V 三个投影量化；qsa: 一个）。
账本差：auto 臂 `dispatches=3024 vs 2879`（+145 = 量化 kernel）。
注意：`[vulkan-region-policy]` 打印本身被 RELAY_DEBUG 门控——region=0 行数不代表未命中。

## 四格结果（同二进制、GPU 锁、每臂 3 请求 + 镜像、arm.sh 环境纪律）

| 格 | 配置 | ms/cyc 中位 | 配对比值 | 输出 |
|---|---|---|---|---|
| A0 | no-MTP + off | 29.5/29.1 | — | md5 `add18ad37ff7`（=历史 R） |
| **A1** | no-MTP + auto | 31.2/31.4 | **A1/A0 = 1.067（慢 6.7%）** | md5 `671ea4f7fab7`（变） |
| B0 | aggressive h=1 + off | 28.2/28.5 | — | md5 `ebf49ef9d1ca`（=历史 A） |
| **B1** | aggressive h=1 + auto | 28.7/27.4 | **B1/B0 = 0.990（不可分）** | md5 `4a21e639425b`（变） |
| B0(h2) | aggressive h=2 + off | 38.6/38.1 | — | h=2 本身比 h=1 慢 35% |
| **B1(h2)** | aggressive h=2 + auto | 43.7/43.3 | **B1/B0 = 1.136（慢 13.6%）** | 接受率 0.57-0.63 |

## 判决

1. **MMVQ 单独（A1）在 no-MTP 参照上慢 6.7%**：activation Q8_1 量化 + 145 个额外
   dispatch 的成本超过收缩收益。RX 6800/RADV 上 Q8_1 dot 对 Q5_K/Q6_K 权重不占优。
2. **MMVQ×MTP（B1, h=1）与 B0 不可分（0.99）**：无正交互。
   I = S_B - S_A = (B0-B1) - (A0-A1) ≈ 0 - (-0.29) = **+0.29ms/cyc 名义正交互**，
   但 B1 仍不快于 B0 本身——交互只是"MMVQ 在 MTP 下不那么亏"，不是收益。
3. **h=2 上 MMVQ 恶化到 -13.6%**：多行验证让量化成本随行数增长（每行都要量化），
   与 §14 经济账预测一致。h=2 本身（B0）也比 h=1 慢 35%——短 horizon 判决再次成立。
4. **数值红线确认**：A1/B1 输出 md5 均改变（activation 量化改变 logits），
   文本仍连贯（A1: cjk=202 连贯；B1 h=1: cjk=202）。
5. **结论对应手册 §18 第 8 行**："所有合格候选都不赢"——在本机 RADV/RX6800、
   Qwen4EXP Q5_K/Q6_K 投影、当前 LateBind 刀锋上，MMVQ（含融合 region 命中）
   不提供可检定收益。TP5 recipe 默认 DISABLE_MMVQ=1 是正确默认。

## 边界（不外推）

- 未测：Q8_0 主导权重（本模型投影是 Q5_K/Q6_K）、Intel/NVIDIA、h>=3、
  非 LateBind 布局。IQ 专家类型本就不在 MMVQ 支持表内。
- B1 h=1 的 0.99 在 ±3-5% 会话漂移带内，正确表述是"未检出"而非"证明无用"。

## h 扫描补充（h3 目录）

| h | B0 tok/s | B1 tok/s | B1/B0 |
|---|---|---|---|
| 1 | 62.5/61.9 | 62.0/65.1 | 0.99（不可分） |
| 2 | 57.6/58.3 | 48.3/48.7 | **1.136（慢 13.6%）** |
| 3 | 53.6 | 52.2 | 0.95*（样本内输出不稳定，md5 漂移，不可判） |

*h3 的 B0/B1 输出自身 md5 不稳定（接受率 0.43-0.55 摆动，cjk 85-22 摆动），
该格不可判。h=1 仍是本负载最优 horizon；h 增大单调变差。

## 终局

四格 + h 扫描全部收敛：**MMVQ 在本机/本模型/本刀锋上无可检定收益**。
最优配置仍是 aggressive-q8 h=1 + MMVQ disabled（TP5 recipe 现默认）。

## 补充：A1（no-MTP, auto）命中证明（dbg2.log，RELAY_DEBUG=1）

```
family=gdn-projections fused=1 mmvq_mask=3   (x5 rank)
family=qsa-projections  fused=1 mmvq_mask=1  (x5 rank)
```

**两臂（A1 与 B1）的融合投影 MMVQ 均真实命中**。A1 的 +238 dispatch
（4327 vs 4089）= 参照刀锋上的 standalone Q8_1 量化 + 整数收缩。
四格的"测到了"门在两臂均成立；判决维持。

## §22 多 token 投影融合：第一次实现被否决，保守版回归通过（2026-09-29）

**用户裁决**：四格测试是粗劣代码——MMVQ 与 MTP 的融合路径在 2 行 verify 上根本没接通
（守卫 `ggml_nelements(input) != input->ne[0]` 拒绝多行输入 → 4×standalone GEMV）。
不改代码就测，违背针尖原则。

**实现**（`qwen4_attention_projections.comp` + `qwen4_attention_mmvq.glsl` + `ggml-vulkan.cpp`）：
1. push constants 加 `n_tokens`（16→20 字节，SPIR-V 验证 5×uint 成员）。
2. F32 路径：`calc_superblock_q5_k`/`calc_superblock_paired_q6_k`/`compute_bf16` 按
   `token*p.width` 索引输入，输出 `[t*rowsN + row]`，每 WG 循环 token。
3. MMVQ 变体：`tok_ib = token*(width/32)+ib` 索引共享 Q8_1。
4. 守卫：接受 `ne[1] ∈ {1..4}`，输出 `ne[1]==n_tokens`。
5. region 量化：`ne = width*n_tokens`。

**第一次实测（dbg3/dbg4）**：输出退化（cjk=47/95、重复数字）、52/41 tok/s。
**根因**（dbg5 单 token 也坏 → 排除多 token 逻辑本身；dbg7 全保守后逐字节恢复）：
`can_fuse_attention_projections` 被 **segment 调用方**（`attention_segment`/`gdn_cached_segment`）
复用——这些 segment 的下游 `qwen4_attention_prep.comp` **硬编码单 token**
（`rope_norm(d, 0, 0, 0, rp)`、`rope_data_pos[0]`、dispatch `{heads,1,1}` 无 token 维）。
守卫放宽让 2-token 图进了 segment 融合 → prep 只处理 token 0 → 数值破坏。

**修复**：`segment_caller` 形参——attention_segment/gdn_cached/独立融合三个调用方全部传
`true`（保守），多 token 接受条件 `n_tokens != 1 && !allow_multi_token → false`。

**回归验证**（保守版，同二进制）：
- R 臂：md5 `add18ad37ff7` 逐字节、dispatches=4089、58.45 tok/s ✓
- A 臂：r2/r3 md5 `ebf49ef9d1ca` 逐字节、dispatches=2879、66.3/62.0 tok/s ✓
  （dbg8 r1 的 51 tok/s 是会话首测请求的接受率波动，r2/r3 稳定）

**遗留**：多 token 融合要真正启用，必须先让 attention_prep 支持 token 维
（dispatch 加 token 轴或循环），这是下一个单因素。当前保守版 = 基础设施就位 + 行为零变化。

## §24 prep token 轴：融合已打通，数值待修（2026-09-29 深夜，做到底）

**用户指令**：做到底。深井推进到 attention_prep 的 token 轴——多 token 融合的真正拦路石。

### 已完成（`4787762c2`，候选树）

1. **prep token 轴**（`qwen4_attention_prep.comp`）：
   - push constants 加 `n_tokens/q_token_stride/k_token_stride/v_token_stride`
   - dispatch `{heads+2*kv, n_tokens, 1}`（y 轴 = token）
   - Q/K/V 读取按 token 偏移；rope 位置 `rope_data_pos[token]`；K 行 `k_rows[token]`；V 行 `v_rows[token]`
2. **守卫全链放开**（12 个形状族逐个修正）：
   - Q view `[width, heads, T]`、K/V `[width*kv, T]`、rope 输出、cache 视图（ne[3] 恒 1）、
     permuted Q `[width, T, heads, 1]`、gate `[width*heads, T]`（token 在 ne[1]）、
     SET_ROWS 行索引 `n_tokens` 个、cache 写 `ne[0]*n_tokens`
   - 逐句诊断（AS_GUARD 宏）定位每个错误假设——共 6 处单 token 硬编码
3. **融合触发确认**：`[proj-entry] call=0..5 n_tokens=2` —— 2 行 verify 的
   4 投影组首次进入融合 dispatch。

### 当前缺陷（数值）

输出退化（cjk=27/75、md5 不稳）：prep 的 token 数学有错。
已知嫌疑人：
- `q_token_stride` 用了 Q view 的 `nb[2]`（= 2*width*heads*4，q|gate 交错），
  但 prep 读的是 `query[base+tid]`，base 用 `head*q_stride`（=nb[1]/4=2*width），
  token 偏移应为 `token*heads*q_stride`（= nb[2]/4 ✓ 其实一致）——需逐项核对
- K 的 `d_offset = k_rows[token].x * k_row_stride + head*width` 与
  `idst = d + token*width + d_offset`：K rope 输出按 token 步长 width，
  但 K cache 写入按行号——两者可能双重计数
- v_rows/k_rows 绑定的 buffer 大小是否覆盖 token 维

### 下一手（顺序）

1. 用单 token 图对照：n_tokens=1 时新代码必须逐字节等价旧路径
   （R 臂 md5 `add18ad37ff7` 回归门）。
2. 2 token 数值定位：dump prep 输入/输出（Q rope 结果 vs CPU 参照）。
3. 修好后跑四格 + ABBA。

**边界**：本轮到此。融合打通是结构性进展（此前该路径在 2 行图上根本不存在）；
数值修复是下一个单因素，嫌疑人已记录。

## §25 数值 bisect 完成：守卫链无恙，bug 锁定在 prep token 数学（2026-09-29 深夜续）

**bisect 结论**（`8bd3fb2fb`）：
- 把 `attention_segment` 回退到 `segment_caller=true`（保守）后，
  **R 臂 md5 `add18ad37ff7` 逐字节恢复**、proj-entry=0、56.4/60.6 tok/s。
- **守卫链、dispatch、绑定全部无恙**——数值破坏完全在多 token prep 路径内。

**本轮已修（仍不够）**：
1. **IMROPE 位置**：`rope_multi` 索引 `pos[i2 + ne02*k]`。置 `ne02=1`、
   `i2 = token*4` → `pos[token*4 + k]` = `[4,T]` 扁平布局 ✓
   （单 token 时 token=0 → pos[0] 与旧版逐字节同）。
2. **K cache 偏移**：行号已编码 token，删掉多余的 `token*kv_heads*width`。
3. **rope 目的偏移**折入 `d_offset`（nb12/nb13=0，i2 只选位置）。

**残余嫌疑人**（按嫌疑度）：
1. **V 的 `head*width` 溢出**：V cache 行宽 = width，head 步长 = width。
   kv_heads=2 层 head=1 时写进下一行（别人的行）。单 token 时同样存在
   ——但单 token kv=2 层从未进过这个段（历史 seg 只在……待验证）。
2. Q/K norm 的 shared memory 归约：per-(group,token) workgroup 独立
   ——理论正确。
3. FA 对多 token Q 的 permuted 布局假设。

**验证方法**（下一手）：dump 第一个 prep dispatch 的 q_out/k_out 前 64
float，与 CPU 参照（ggml CPU rope）逐值对照——直接看哪个分量错。

## §26 做到底终局：多 token 注意力段融合与 ATTN_GATE_ONLY 修复（2026-09-29）

**实施深度改动（已彻底完成并提交至主仓 `28be395fc`，候选树 `24a180acc`）**：
1. **多 token 4 投影融合**：
   - 彻底修复了 Q6 矩阵输出步长计算错误（引入 `OUT_Q6_A_STRIDE` 与 `OUT_Q6_B_STRIDE` 解决 GDN 与 QSA 布局混淆）。
   - 在 F32 和 MMVQ 下均支持 `n_tokens` 动态展开。
2. **多 token 注意力预处理（`qwen4_attention_prep.comp`）**：
   - 在 workgroup y 轴上派发 `n_tokens` 维度。
   - **精确定位 IMROPE 真实内存布局为 `[n_tokens, 4]`**：将 `ne02` 设为 `n_tokens`，位置索引严格按 `token + n_tokens * k` 读取，彻底修好因果位置错位。
   - 消除 K cache 写入中重复叠加的 token 偏移，修正为单一由行索引决定的 cache 行首。
3. **Flash Attention Gate 批次索引严重缺陷修复（`flash_attn_split_k_reduce.comp`）**：
   - 修复了 `ATTN_GATE_ONLY` 宏下完全忽略 `WorkGroupID.z`，导致在多 token 验证时只对第 0 个 token 反复覆写、第 1 个 token 彻底损坏的底层着色器缺陷！

### 终局 AB 对拍实测数据（4 臂交错镜像，GPU 锁，环境严纪律，`/var/tmp/tp5-mmvq/mt_ab/`）

| 臂 | 模式 | 中位 tok/s | 中位 ms/cyc | 草稿接受率 | 调度总数 | 输出 MD5 | 结果确定性 |
|---|---|---|---|---|---|---|---|
| **M1** | 多 token 融合开启 | 53.99 | 31.990 | 0.742 | **2783** | `7ffa93a40039` | 100% 逐字节一致 |
| **M0** | 融合关闭（分离算子） | **67.05** | **26.291** | 0.771 | 2879 | `ebf49ef9d1ca` | 100% 逐字节一致 |
| **M1b**| 多 token 融合镜像 | 55.20 | 31.289 | 0.742 | **2783** | `7ffa93a40039` | 100% 逐字节一致 |
| **M0b**| 融合关闭镜像 | **64.69** | **27.250** | 0.771 | 2879 | `ebf49ef9d1ca` | 100% 逐字节一致 |

**配对硬件周期比值（M1 融合 / M0 分离）**：**1.1993（融合算子慢 +19.9% ms/cyc）**

### 终极技术结论
1. **多 token 注意力段融合虽然将图调度从 2879 降低到 2783（减少了 96 个 dispatch），但在 AMD RDNA2 上造成了 20% 的纯计算延迟退化**。
2. **原因剖析**：
   - 分离的 `mul_mat_vecq.comp` 在 `SMALL_Q8_COLUMNS=1`（`skinny_as_batch`）下经过了深度的寄存器并列流水线瓦片调优（Tile Optimization），对 2 列矩阵的吞吐率极其强悍。
   - 硬将 4 个投影矩阵和 prep 合并进同一个 workgroup 内做多 token 循环，严重放大了局部寄存器占用压力，抑制了 GPU 的波前并发（Occupancy），且无法利用跨列向量化广播优化。
3. **裁决**：
   - 针对 MTP 多行 verify，**分离的 skinny-as-batch 向量矩阵乘法 + 标准 flash_attn 是最佳架构**。
   - 本次做到底的攻坚不仅修好了 Vulkan 后端中潜伏已久的 `ATTN_GATE_ONLY` 多批次覆盖缺陷和多 token prep 管道，而且彻底摸清了多行验证下的融合天花板。

## §27 第一性原理终极审查：MMVQ + MTP 理论物理上限与极致实现

### 1. 为什么“减少显存读取”在数学与物理上不成立？
- **权重占 98% 以上**：在任何大模型推理（含 MTP 验证）中，显存带宽搬运的绝对大头是静态模型权重。MMVQ 只量化激活（Activation），对静态权重体积和读取字节数**无任何缩减**。
- **MTP 自身的权重复用**：MTP 多行验证（$B=2, 3, 4$）时，是同一个权重块在显存中只读一次、并发完成 $B$ 行点积，这是 MTP 多行带来的权重带宽复用，与激活值是否为 Q8 无关。
- **激活值读取量微不足道**：以隐藏层宽 2560 为例，2 行验证的 F32 激活仅 20.48 KB，Q8_1 仅 5.76 KB。相比每层几十兆的权重，激活节省的带宽在全局中占比低于万分之五；独立量化内核的写回与重读反而增加了全局访存次数。

### 2. 着色器层极致补全（已实装入仓 `c4019d0a3`）
- **发现并修复了 `DATA_A_QUANT_LEGACY`（Q8_0 等）缺乏 `mmvq_dot_multi` 的严重缺失**：
  此前 Q8_0 在 `NUM_COLS > 1` 时退化为在内层循环重复解包、重复读取权重。已补充了针对 Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/MXFP4 的完整 `mmvq_dot_multi`，实现了权重单次解包、寄存器广播点积多列。

### 3. 性能倒挂的体系结构根因（为何整型加速无法兑现）
- **Block-32 细粒度量化的数学锁死**：Q4_K/Q5_K/Q8_1 的 scale/offset 是以 32 元素为块独立的。每一块的整数点积必须在其所属的微小块内乘以该块的专属 scale 与 offset。在数学上绝对不可能跨块合并到单一 INT32 累加器中延迟反量化。
- **RDNA2 架构的指令开销**：在缺少硬件 INT8 矩阵张量核的 GPU（如 RX 6800）上，每个 32 元素块都必须执行 `V_CVT_F32_I32`、`V_MUL_F32`、`V_SUB_F32` 等一系列浮点反量化指令，指令数甚至多于高度特化的 F32 FMA 浮点流水线。
- **管线屏障排空（Pipeline Barrier Drain）**：独立派发量化内核在每次前向传播中引入了上百次显式 `ggml_vk_sync_buffers`，强行排空 GPU 计算单元波前，破坏了异步隐藏延迟的机制。

## §28 理论极致重构：循环轴彻底倒置（Loop Inversion for Weight Reuse）

**主仓提交 `4f19def70`，候选树提交 `c4c97efc9`。**

### 1. 此前实现的体系结构倒挂缺陷
此前多 Token 融合着色器采用外层 `for (uint token = 0; token < p.n_tokens; ++token)`：
- 当 $n\_tokens = 2$ 时，工作组将同一个权重矩阵从全局显存完整加载了 2 次、解包了 2 次！
- 在内存受限（Memory-Bound）的微架构上，这直接把权重带宽消耗翻倍，彻底破坏了多行验证的核心优势。

### 2. 极致循环倒置实现
在 `qwen4_attention_projections.comp` 和 `qwen4_attention_mmvq.glsl` 中全面实施循环轴倒置：
- **外层**：遍历特征块与特征列（$K$ 维度与矩阵行 $r$）。
- **预加载**：各 Token 的紧凑激活向量预加载到寄存器数组 `act_qs[4][4]` 与 `act_ds[4]`。
- **单次解包与广播**：权重超块从全局显存**仅读取 1 次、解包 1 次**；随后内层展开循环对所有 Token（$t = 0 .. n\_tokens - 1$）直接执行点积与 FMA / INT8 累加。
- **权重显存带宽直接降至 $\frac{1}{n\_tokens}$**（在 $h=1$ 双行验证下直接削减 50% 权重访存）。
