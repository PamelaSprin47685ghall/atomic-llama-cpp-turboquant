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
1. 该版融合减少 96 个 dispatch，但该负载的**端到端** ms/cyc 增加 19.9%；不能将差值全部归于投影计算。
2. 寄存器压力和 occupancy 是待验证的解释；未记录 ISA VGPR 用量、波前占用率或分 kernel 计时。
3. 该 AB 仅适用于当时的着色器和负载；后续 §28–30 的重构须重新验证，不能据此宣称多行融合的硬件天花板。

## §27 第一性原理终极审查：MMVQ + MTP 理论物理上限与极致实现

### 1. 激活量化与权重访存是不同问题
- MMVQ 改变激活的存储格式，不改变已量化权重的**格式或容量**；不同 shader 的权重加载次数可以变化，不能推断实际 VRAM 流量恒定。
- 循环倒置的着色器源码每个权重块遍历一次并对多 token 累加；缓存命中和最终流量仍需硬件测量。
- 宽 2560、两行的 F32 激活为 20.48 KB，Q8_1 为 5.76 KB（32 值 36 字节）；相对于权重的真实带宽占比未测，不能套用固定百分比。

### 2. 着色器层极致补全（已实装入仓 `c4019d0a3`）
- **补全 `DATA_A_QUANT_LEGACY`（Q8_0 等）的 `mmvq_dot_multi`**：
  针对 Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/MXFP4 增加共享解包的点积实现。但此阶段 `mul_mat_vecq.comp` 仍使用 `#if NUM_COLS > 1`；`NUM_COLS` 实际是 GLSL 专门化常量而非预处理宏，故调用点始终不进入该分支。真正启用、验证和适用类型边界见 §36；不能将此阶段的函数定义误认为当时已实现运行时权重复用。

### 3. 整数点积的确定开销与未证实的瓶颈
- **分块标度**：Q5_K/Q6_K 的 32 值子块标度可不同，整数点积须按标度合成；不能无条件将所有子块合并为一个未加权的 INT32 和。
- **指令与寄存器**：着色器在整数点积后乘标度并累加浮点值；实际 RDNA2 ISA 指令数、寄存器数量及收益未测，不能从 GLSL 推断特定 `V_*` 指令数量。
- **同步**：量化输出到消费者之间需要 RAW 依赖；代码的 `ggml_vk_sync_buffers` 是同步点，是否排空整个 GPU 波前没有直接证据。

## §28 理论极致重构：循环轴彻底倒置（Loop Inversion for Weight Reuse）

**主仓提交 `4f19def70`，候选树提交 `c4c97efc9`。**

### 1. 此前实现的体系结构倒挂缺陷
此前多 Token 融合着色器采用外层 `for (uint token = 0; token < p.n_tokens; ++token)`：
- 当 $n\_tokens = 2$ 时，源码重复执行权重加载和解包；硬件的缓存可能消除部分 VRAM 读取，不能由源码直接推断显存流量翻倍。

### 2. 极致循环倒置实现
在 `qwen4_attention_projections.comp` 和 `qwen4_attention_mmvq.glsl` 中全面实施循环轴倒置：
- **外层**：遍历特征块与特征列（$K$ 维度与矩阵行 $r$）。
- **预加载**：各 Token 的紧凑激活向量预加载到寄存器数组 `act_qs[4][4]` 与 `act_ds[4]`。
- **单次解包与广播**：每次源码循环迭代加载、解包权重一次，之后对各 Token 累加。
- 相对旧实现，源码权重加载次数由 $n\_tokens$ 降为 1；不能据此宣称实际显存带宽下降 $1/n\_tokens$。

## §29 深入理论极致审查与微架构优化（主仓 `71ceb8bff`，候选树 `d2fe26200`）

严格执行用户要求：“做到理论极致之前禁止跑程序得结论，再次反复检查优化点”。针对代码库的底层汇编与调度逻辑进行了第二轮逐行审查，挖出并解决了另外两个隐藏缺陷：

### 1. `split_k > 1` 模式下 `ATTN_GATE` 批次偏移遗漏（已修复）
- **缺陷**：在 `flash_attn_split_k_reduce.comp` 的主累加分支中，当 `split_k > 1` 时，输出写回已加上多批次偏移，但读取 `data_g` 时仍漏掉了 `batch_idx * (ne1 * gate_stride)` 偏移，导致多 Token 验证在切分 $K$ 维时长序列下依然存在覆写隐患。
- **修复**：对齐非 ONLY 分支的 `g_offset` 计算，彻底闭合 split-k 多批次因果掩码链。

### 2. 动态循环分支气泡消除（Straight-line Pipelined Unrolling）
- **边界**：`p.n_tokens` 是运行时 push constant；仅乘加内层显式写出 Token 0/1，不能保证所有动态分支或寄存器开销消失。
- **极致重构**：
  - 对单 Token 解码（$N=1$）和双 Token 验证（$N=2$）显式写出部分乘加；两者的真实负载占比未计量。
  - BF16 路径在 §30 补齐；归约和 $N>2$ 回退仍保留运行时循环，SPIR-V 中仍有 `OpLoopMerge`。

### 3. MMVQ 调度选择器精细化重构（`ggml_vk_should_use_mmvq`）
- **调度变更**：旧代码的 `if (n > 1) return true;` 绕过后续 vendor/k 判据；新代码在 $n<8$ 时先检查 $k \ge 2048$，$n\ge8$ 时保持宽批次优先。
- **未证实**：2048 是策略阈值而非物理分界；它对不同 $m$、权重类型及设备的性能最优性未测。

## §30 守卫复审、BF16 收尾与数值烟测（2026-09-29）

### 静态复审

- `ggml_vk_can_fuse_attention_projections` 原来的 `segment_caller=true` 只出现在**独立**投影调用点，恰好禁止其多 token 路径；两个具备多 token 消费者的 segment 却传 `false`。三个调用点均有各自下游形状守卫，删除失效的布尔参数及单 token 限制，使独立投影可以取 2–4 行。
- `qwen4_attention_projections.comp` 的 BF16 QSA indexK 路径此前仍对每次特征迭代使用 `token<p.n_tokens` 动态循环；现在显式计算 Token 0/1、2–3 保持回退循环，权重加载保持在 token 轴外。
- Q5_K/Q6_K 的分块标度计算与标准 `mul_mat_vecq_funcs.glsl` 路径一致；`MMQ` 下 Q6_K 多列已经有 `mmvq_dot_multi`，无须重复实现。Q8_1 x4 的 `width/32` token 步幅要求 width 为 128 的倍数，投影守卫要求 256 的倍数。
- 离线反汇编 `qwen4_qsa_projections[\_mmvq].spv` 仍能找到 `OpLoopMerge`（含回退和归约）；不得宣称所有 N=1/2 控制流消失，更不能据 GLSL 判断 VGPR/occupancy。

### 只做正确性 smoke，不做吞吐判决

- 候选构建 `/var/tmp/tp5-final-build`：`ninja -j 4 bin/llama-server` 成功，四种投影变体生成并链接；既存的其他 C/C++ 警告不在本变更范围。
- 同二进制、五卡、GPU 锁、`h=1`、aggressive-q8、MMVQ disabled；关闭 `ATTENTION_SEGMENT` 和 `GDN_CACHED_SEGMENT` 后，开启**独立投影**与关闭所有投影融合对照：相同请求两边 `predicted_n=161`、接受 59、输出 MD5 均为 `552cd459aa59`，CJK=215、标点=23。rank0 解码图记录 `dispatches=2681` 对 `2825`（减少 144）；这是命中佐证，不是时间测量。
- `h=1`、强制 MMVQ 独立投影：五卡均报告 `mode=forced`、GDN `mmvq_mask=7`、QSA `mmvq_mask=5`；请求 `predicted_n=171`，CJK=246、标点=22、无服务错误。量化后的输出 MD5=`2576e1cb1a2e`，不能与 F32 的 MD5 要求相同。
- `h=3`、MMVQ disabled，**四行 F32/BF16 投影**与禁用独立投影融合对照：同请求两边 `predicted_n=171`、接受 96、MD5 均为 `68b2736fa4a9`，CJK=229、标点=21；显式 Token 0/1 及 2–3 回退都在端到端路径上通过数值回归。
- `h=3`、强制 MMVQ 的四行验证与禁用投影融合对照均完成请求，但融合输出 CJK=100/标点=53，对照 CJK=150/标点=47，MD5 不同。两者都触发历史 h=3 输出不稳区域；文本 MD5 不是量化投影的数值检验。
- **四行逐值对拍**：同一组确定性 2560×32 权重、4×2560 激活，在 Vulkan0 构建四个连续投影节点；强制 MMVQ 分别执行融合与禁用融合的图。QSA 日志 `mmvq_mask=5`，GDN `mmvq_mask=7`，各比较 4×4×32=512 个 F32 输出，无 NaN/Inf；两种图族的 Q5_K 最大绝对误差均不超过 `1.52587890625e-5`，Q6_K 与 QSA BF16 输出逐值相同，512 值中没有误差超过 0.01。该探针只验证这些确定性输入，不等同于生产模型每层数值对拍。

**界限**：F32/BF16 在二行和四行均与分离投影端到端逐字节一致；强制 MMVQ 二行可运行、四行生产请求完成且合成四行投影逐值对拍通过。没有新版本的 ABBA 吞吐、缓存带宽、VGPR 或 occupancy 证据，不能据此宣称达到硬件理论上限。

## §31 指令资源证据与令牌容量专门化（2026-09-29）

### 可重复的编译资源事实

- Vulkan0/RX 6800，`RADV_DEBUG=shaderstats,nocache`，2560×32 四投影小图，`n_tokens=1` 与 4 旧版都复用同一管线 hash：QSA 强制 MMVQ `VGPRs=64`、`Subgroups per SIMD=16`，GDN 同为 64；QSA F32 为 `VGPRs=128`、`Subgroups per SIMD=8`。没有 VGPR spill。这是驱动编译统计，**不是**实际波前占用率或吞吐率。
- 新增 `MAX_TOKENS` 专门化常量 ID3，MMVQ mask 移至 ID4；容量为 1/2/4，`ggml_vk_attention_projections` 为请求行数选择最小足够容量，缺管线仅能向更大容量回退。Q5/Q6/BF16 与 MMVQ 数组、token1 分支和 token2+ 回退以容量限定；push constants 和十个描述符不变。独立融合守卫也按实际行数验证管线容量。
- RADV 编译结果：QSA/GDN **强制 MMVQ** 容量 1/2/4 的 VGPR 为 **40/56/64**，编译器上界 `Subgroups per SIMD` 为 **24/18/16**，均无 spill；QSA/GDN **F32** 为 **64/128/128**，对应 **16/8/8**。QSA N1 MMVQ 指令数 8422→7442，分支数 391→119；这些指令统计不等同于应用加速。
- 默认 TP5 recipe 的五卡服务器打印 `mode=disabled`，所以不能把五卡“未强制”试验称为混合 auto MMVQ；单独 Vulkan 图在不设 FORCE/DISABLE 时，确实命中 QSA mask1/GDN mask3，**本节阶段**其 N1/N2/N4 的 VGPR 为 **64/128/128**。N2 F32 Q6 成对解包寄存器瓶颈在 §32 继续处理。

### 容量映射数值回归

- 同一合成输入，QSA/GDN × F32/强制 MMVQ × N=1/2/3/4，共 **16 张图**：新旧候选库逐输出字节相同；图覆盖 1、2、4 容量与 N3 选择容量4。四投影每图 4×N×32 个 F32，无 NaN/Inf；逐图均零个绝对误差超过 0.001。
- 另对 QSA mask1/GDN mask3 自动混合图 N=1/2/4 比较启用与禁用投影融合：六图均无 NaN/Inf，最大绝对误差 `1.52587890625e-5`，无 GPU 同步或协议失败。
- 完整五卡 aggressive-q8 `h=1`：默认禁用 MMVQ 的 MD5=`552cd459aa59`，强制 MMVQ（五卡 GDN mask7、QSA mask5）的 MD5=`2576e1cb1a2e`；与 §30 更改前同负载逐字节相同。`h=3`、禁用 MMVQ 的四行图 MD5=`68b2736fa4a9`，同样与 §30 一致。

**结论边界**：本节阶段静态资源压力在 N1 和强制 MMVQ N2 有可观下降，N2 自动混合/F32 的瓶颈转交 §32；计时资格见 §33，不能称 24 subgroups 已实际驻留或断言物理极限已测得。

## §32 Q6_K 成对解包的 VGPR 生存期（2026-09-29）

- 原 `calc_superblock_paired_q6_k` 同时持有 alpha/beta 两组 Q6 权重（每组四个 `vec4`），跨 Token 0/1/2–3 计算均存活。候选代码改为完成 A 对所有 token 的累加后才解包 B，再完成 B 的累加；两组权重仍各读取、解包一次，单一工作组、单一 dispatch，B 侧激活从缓冲区再次读取。每个输出矩阵的浮点累加顺序不变；L1/VRAM 流量增加多少不能仅从代码推断。
- `RADV_DEBUG=shaderstats,nocache`，QSA/GDN **F32** 容量 1/2/4 的 VGPR 从 §31 的 **64/128/128** 变为 **64/96/96**，编译器 `Subgroups per SIMD` 从 **16/8/8** 到 **16/10/10**；QSA/GDN 混合 mask1/3 N2 从 128 到 96。强制 MMVQ mask5/7 绕开 F32 Q6，原 **40/56/64** 不变；均无 spill。QSA F32 N2 `Instructions` 3879→4019，额外重读激活和编译器调度是相对资源占用的显式代价，未按此数字推断时间。
- 同一合成四投影输入，与 §31 的快照对拍：QSA/GDN × F32/强制 MMVQ × N=1/2/4，加上混合 mask1/3 × N2，共 14 图，所有输出**逐字节相同**、无 NaN/Inf。五卡 aggressive-q8 实际 MTP `h=1`（禁用 MMVQ）MD5 仍为 `552cd459aa59`，`h=3` 四行仍为 `68b2736fa4a9`，分别接受 59 与 96，未见服务错误。

**取舍**：驱动静态 occupancy 上界提升 8→10 subgroups/SIMD，但激活读取与指令数增加；§33 的 ABBA 不能单独隔离这处变化，且缺硬件缓存/实际波前采样，不宣称本方案端到端更快，也不将其称为绝对物理上限。

## §33 理论审查后的交错镜像资格（2026-09-29）

同一新二进制、五卡、独占 GPU 锁、`RADV_DEBUG=nobolist`、aggressive-q8、`h=1`、禁用上层 attention/gdn-cached segment；A=独立四投影融合，B=只禁用该投影融合。每臂新服务、相同 prompt/seed/温度，`/completion` 返回的 `predicted_per_token_ms`，不含模型加载；原始服务日志 `/var/tmp/tp5-mmvm-seq-*.log`。交错次序 **ABBA / BAAB / ABBA / BAAB**，不是将不同二进制或不同 workload 混算。

- 禁用 MMVQ 的 F32 组：16 臂中第 6 臂 A 的 `pred_n=157`、MD5=`39e053d56624`，其四臂镜像**整组作废**，未拿 157 与 161 比速。剩余三组四臂均 `pred_n=161`、接受59、MD5=`552cd459aa59`、CJK=215、标点=23，且 `barriers=2456/copies=192` 两路一致；A/B 的 `dispatches=2681/2825`。每组 A/B 平均 ms/token：`18.531/18.083`、`19.819/18.159`、`18.266/18.359`，相对差（A 减 B）`+2.47%`、`+9.14%`、`-0.50%`。方向不一致、方差大，**不能从三组判定融合胜负**，更不能用 shaderstats 的静态 occupancy 代替实测加速。
- 强制 MMVQ 的四臂 ABBA：A 的 MD5=`2576e1cb1a2e`、接受63；B 的 MD5=`dd2877d269ff`、接受64，虽均 `pred_n=171` 且连贯，但 spec 接受和 token 内容不同，**不是工作量一致的吞吐对照**。A/B dispatches=`2883/3027`，barriers=`2706/2658`，copies=192；均无服务错误。不能据其吞吐决定 Q8 融合是否优于分离。
- 同服务“热机第二请求”即使传 `cache_prompt=false` 仍由 `pred_n=161`、MD5=`552cd459aa59` 变为 `pred_n=171`、MD5=`ebf49ef9d1ca`，该方案也不具有工作量资格。

**决策**：保留当前融合代码和显式 `GGML_VK_DISABLE_ATTENTION_PROJECTIONS` 回退，不根据不合格或方向不稳的短跑改变默认选择。内核正确性已有合成逐值对拍和实际五卡 smoke；性能最优、缓存流量及硬件极限仍未被上述证据证明。

## §34 五卡真正的 auto MMVQ（2026-09-29）

- `common_tp5_apply_env` 在合格 RELAY 配方里只对**未设置**的 `GGML_VK_DISABLE_MMVQ` 设置 `1`；Vulkan 端把值 `0` 判为“不禁用”。因此 `GGML_VK_DISABLE_MMVQ=0` 能在原有五卡服务上得到 `mode=auto`，不需要改源码或使用全类型强制量化。`h=1` 的 GDN/QSA 日志确认为 `mmvq_mask=3/1`（Q5 用 Q8_1，AMD 上 Q6 仍用 F32），和合成图一致。
- 五卡 `h=1`、同二进制、独占锁、独立四投影融合 A vs 关闭投影融合 B：A `pred_n=171`、接受61、MD5=`6dca351de601`、CJK=235、标点22、账本 `dispatches/barriers/copies=2729/2552/192`；B `pred_n=171`、接受60、MD5=`51a1f815072e`、CJK=219、标点22、账本 `2873/2504/192`。两路均连贯、无服务错误，但 token 与 speculative 工作量均不相同，**不能比较 tok/s 或改变默认**。
- `h=3`、auto mask3/1 独立融合请求 `pred_n=171`、接受105、CJK=173、标点20、无服务错误；四行路径可运行。相比 §32 F32 的 MD5 不同，不能当作逐字节数值等价证明。

**保守决策**：保留 TP5 配方默认 `DISABLE_MMVQ=1`，保留用户显式 `=0` 进入 auto 的完整路径；不凭不同输出、不同接受率的短跑把 auto 转为默认。静态 96-VGPR 混合路径及合成逐值对拍只给代码资源与局部数值证据，实际物理吞吐上限仍未有工作量一致的资格。

## §35 等值小图 GPU 时间戳（2026-09-29）

Vulkan0、四连续投影、width=2560、Q5[0] rows=4096、GDN Q5[1] rows=4096、其余 rows=1024、两行激活；两路径使用相同输入及权重。`GGML_VK_PERF_LOGGER=1` 用 query pool 记录每节点 GPU 时间；每臂 64 次重复，A=四投影融合计一个节点，B=四个分离节点的时间总和。各模式 ABBA 镜像，独占 GPU 锁、无并行编译；原始日志 `/var/tmp/tp5-ts-{mode}-{family}-{arm}-{A|B}.log`。GPU 平均 us/图：

|模式|QSA A/B|GDN A/B|A 相对 B|
|---|---:|---:|---:|
|F32|130.90 / 181.06|178.74 / 224.18|QSA −27.7%，GDN −20.3%|
|auto Q5 (mask1/3)|117.32 / 178.56|151.69 / 186.89|QSA −34.3%，GDN −18.8%|
|强制 MMVQ (mask5/7)|96.45 / 157.69|120.70 / 182.87|QSA −38.8%，GDN −34.0%|

这是该**单图、缓存驻留、64 次复用**条件下的 GPU 投影时间，数值正确性另由 §31–32 的非零确定性输入对拍证明。测量所用二进制先于 §36 的通用 MMVQ 分支启用和 Q6 shuffle；不能用于评价新实现。`GGML_VK_PERF_LOGGER` 会强制提交/等待并关闭图 replay，四节点小图也未模拟 97 层 TP5 队列、权重换页与多卡屏障，因此这些 20–39% 不能外推为服务 tok/s 增益。它解释了“投影核加速但 §33 端到端未定”的两个观察可以同时为真；默认和回退选择不变。

## §36 复审找到的死分支与 Q6 同步瓶颈（2026-09-29）

1. `mul_mat_vecq.comp` 原有 `#if NUM_COLS > 1` 永远为假：`NUM_COLS` 在 `mul_mat_vec_base.glsl` 是 ID2 的 `layout(constant_id=2)` 专门化常量，着色器生成器不定义同名预处理宏；§26 加的 `mmvq_dot_multi` 从未在通用核中调用。改用按 `NUM_COLS` 专门化求值的 GLSL `if`，仅 2–4 列、**存在 `HAS_MMVQ_DOT_MULTI` 的量化类型**进入列预载/共享权重解包。单列、超过四列、IQ1_S/M（缺共享点积 helper）继续原路径，避免 18 列的寄存器常驻或 IQ1 中额外的逐列缓存复制。`#if NUM_COLS > 1` 源码中的 IQ1 K_PER_ITER=32 预载缺失也是**死分支**，不是线上 IQ1 错算；无需为未启用的 IQ1 优化分支引入新副本。
2. Q6_K 配对的 F32 路径原先每个块/输出行经共享 `sccache_a/b` 和 `barrier()` 交换 16 项子块 scale。每 16 个相邻 invocation 的 `ix=tid/16` 相同、`itid=tid%16` 遍历 0..15，`s_offset+{0,2,4,6}` 范围始终为 0..15；在目标 RADV/RDNA2 subgroup32 中，四次 `subgroupShuffle` 分别从同 16-lane 片取 scale。循环 `i=ix+k*it_size` 让尾块要么整个 16-lane 片执行，要么全部不进入：不依赖未参与的 lane。由此删除 Q6 全工作组 barrier、两份 scale LDS 和尾块空转分支；Q5/量化 Q6 计算分支不变。该融合管线仅对 RADV、满足既有 subgroup 布局条件的设备创建。
3. 完整着色器与服务 `ninja -j 4 bin/llama-server` 编译链接成功。QSA/GDN F32 Q6 的 SPIR-V 有 `OpGroupNonUniformShuffle` 且无 `OpControlBarrier`；同 RADV `shaderstats,nocache` QSA/GDN 两 token 容量 VGPR=88、无 spill、`Subgroups per SIMD=10`（§32 之前 F32 N2 VGPR=96、静态上界同为10）。通用四列 Q5_K/Q6_K 各 VGPR=56、无 spill、静态上界18。上述是编译资源，**不是**实际驻留波前或端到端吞吐。
4. Vulkan0 合成图非零权重：先以宽2560（10个 Q6 超块）覆盖 QSA/GDN 的不同行数 33/35/35/49 与 33/35/35/35，N=1/2/3/4 八组，融合与关闭融合后逐字节相同；QSA N2 与原共享 cache 的快照也相同。**纠正尾块资格**：本机 subgroup32，管线 `BLOCK_SIZE=32`、`it_size=2`，10 能整除 2，故另以宽2816（11个 Q6 超块）、同样奇数行数和 N=1/2/3/4 再测八组，全部融合/分离逐字节相同、无 NaN/Inf，覆盖只有 `ix=0` 的 16-lane 片参加尾块而 `ix=1` 不参加的情况。通用强制 MMVQ 的 IQ1_S、IQ1_M、Q4_K、Q5_K、Q6_K、Q4_0 各以 N=1/2/3/4 对拍每个 token 的独立 N1 运行，全部逐值相同、非零且有限。五卡实际 `h=1` 强制 MMVQ GDN/QSA mask7/5 输出 MD5=`2576e1cb1a2e`、CJK246、标点22、接受63，与 §33 对应同负载一致；禁用 MMVQ 的 Q6 shuffle 请求 CJK205、标点21、无错误，输出并未与先前单请求逐字节相同，底层 TP 基线本身非确定，不能将其解释为回归或等价证明。

**当时未判定；Q6 小图已在 §38 补测**：§35 和 §33 的数值均是旧 shader；评价新通用多列/无 barrier 的五卡净收益仍须冻结二进制、同 workload 交错对照，并记录 `pred_n`、接受率、输出与三重账本。即便测得性能优势，也只能证明该配置与负载的改进，而非数学上的全硬件上限。

**剩余访存与指令边界复审**：通用核每次 `iter` 为活动列各取一组 Q8_1 激活；2–4 列的 helper 对每个输出行解包一次权重，再分别执行每列点积，不能删掉随 token 改变的激活或点积。Q6 F32 配对核的 A/B 权重来自不同矩阵；顺序解包使激活在 B 侧再次从 `common_input_v4` 读取，但把两侧激活同时保存在寄存器中会延长生存期，与 §32 的 VGPR 降幅竞争。Q5、Q6 的 2–3 号 token 仍走动态循环；无证据表明展开会改善 RADV 生成码。`NUM_COLS<=4` 是 MTP 工作集的寄存器容量选择，不是硬件最优阈值证明；缓存命中、VMEM 请求合并、真实 occupancy 与五卡端到端收益仍需同负载实测。故没有依据再增加一个未对拍的访存改写或更改默认策略。

补充 CPU 参照验证：重建 `bin/test-backend-ops`，`Vulkan0` 上以 `GGML_VK_FORCE_MMVQ=1` 对 `m=16,k=256,n=2/3/4`、单 batch 的 Q8_0/Q2_K/Q3_K（9/9）及 Q4_0/Q4_1/Q5_0/Q5_1/MXFP4/Q4_K/Q5_K/Q6_K/IQ1_S/IQ1_M（30/30）运行 `MUL_MAT` 后端测试，均通过其 CPU 对照的 NMSE 阈值 `5e-4`。这是小形状容差对拍，不能替代 §36 的逐字节多 token 比较，也不能证明所有形状、量化输入或服务吞吐等价。

## §37 容量 4 token 尾循环的候选复审（2026-09-29）

仅在候选树把 Q5、成对 Q6 的 token 2–3 循环改为 `[[unroll]]` 的容量固定循环，按活动 token 检查后执行；主仓未采用。GLSL 源码少一个动态循环上界，但 QSA SPIR-V 优化后 `OpLoopMerge` 仍在（只多了 `Unroll` 标记），容量 4 的静态 SPIR-V 从 59976 增到 60336 字节。在宽 2816、GDN/QSA 奇数行 33/35/35/35 与 33/35/35/49 的 Vulkan0 融合/分离图上，N=1..4 八组均逐字节一致。匹配的 RADV `shaderstats,nocache` 容量 4 F32 融合管线：原版和候选版 GDN 均为 VGPR=120、8 subgroups/SIMD、6069 instructions、179 branches、488 VMEM；QSA 均为 VGPR=120、8 subgroups/SIMD、4983 instructions、173 branches、352 VMEM，均无 spill。**这不证明机器指令序列完全相同或时间完全相同**，但未见可归因于此改写的静态成本下降；已撤回候选，未按单次 query 时间选胜负。该探针也暴露了当时容量 4 F32 的 120-VGPR 压力；§38 已按相同形状核对共享 scale 版，不能把 §36 容量 2 的 88 VGPR 外推到容量 4。

## §38 Q6 scale 通信的反证与回退（2026-09-29）

§36 将去 barrier/shuffle 当作指令级收益候选，但 **静态 VGPR 不随它单调改善**：在相同容量 4、宽 2816、QSA/GDN F32 探针上，shuffle 均为 VGPR=120、8 subgroups/SIMD，恢复双缓冲 `sccache_a/b` 的 LDS 版均为 VGPR=96、10 subgroups/SIMD，均无 spill。旧 LDS 有 barrier 和更多静态指令，真实占用不能单靠 `Subgroups per SIMD` 推断；先做同图时间戳交错。

固定同一台 RX 6800、同一融合图与确定性非零输入，独占 GPU；两版分别编译、同 `env -i` + `RADV_DEBUG=nobolist`、禁用 MMVQ，`GGML_VK_PERF_LOGGER=1`。每臂每图 65 次提交，丢首个查询，以下为余下 64 次的 GPU query 中位数（µs；**LDS / shuffle**）：
原始逐次 query 和五卡 smoke JSON 保存在 `q6-lds-shuffle-query-20260929.tar.gz`；表中所有中位数可由日志中 `ATTENTION_PROJECTIONS MUL_MAT_VEC q5_K` 行重算。

|宽 / 行数 / token 数|GDN|QSA|
|---|---:|---:|
|2560 / Q5 4096、Q6 1024、BF16 1024 / N=4；交错 shuffle–LDS–LDS–shuffle|296.46、296.46 / 347.54、347.52|204.32、203.58 / 235.90、235.66|
|相同形状 / N=2；各一臂|176.30 / 191.60|123.68 / 130.68|
|相同形状 / N=1；各一臂|115.72 / 120.74|82.82 / 89.66|

另在宽 2816（Q6 共 11 块，只有一个 16-lane 片处理尾块）的奇数行小图 N=1..4，LDS 版融合与分离逐字节相同、无死锁；宽 2560、上述大图 N=4 也逐字节相同。LDS 的未满块入口要求全工作组参加 barrier，仅在 barrier 后让无效片跳过；整块循环后只在确有余数时运行尾块，省去空 barrier。本轮在主仓与候选树恢复 LDS、删去不再使用的 subgroup shuffle 扩展，不做容量 4 特例：本图 N=1/2/4 方向一致且单一实现更可维护。

**资格边界**：64 次重复使权重缓存驻留，查询器强制提交/等待、关闭 replay；数值仅证明这些图的 GPU 执行时间，不代表 97 层 TP5 队列、五卡同步、模型权重换页或服务 tok/s。§36 的 shuffle 数值/静态记录仍属历史实验，不能称当前实现或将 shuffle 编译开销减少直接写成端到端收益。通用多列 MMVQ 的五卡净收益仍未测定。

恢复后重新构建 `bin/llama-server`、`bin/test-vulkan-command-replay`：后者 `--attention-projections-only` 通过 F32 逐字节和 MMVQ 容差对照；最终着色器在宽 2816、N=1..4 的 QSA/GDN 八组以及宽 2560、N=4 的大图两组与分离路径逐字节一致。五张 RX 6800 上以 `--tp5 qwen4exp-af -md ... --spec-type draft-mtp --tp5-latebind off` 单次请求 smoke：HTTP 200，`pred_n=160`、70/89 draft accepted，CJK=241、distinct ratio=0.701、标点=19；账本五 rank `late=0`，解码时 chain 锁定并复用 88 图。启动与 prefill 曾记录 `CRITICAL-FALLBACK`，不能写成全程无回退；这一次 53.80 tok/s 未交错、且 `pred_n`/接受率不匹配旧跑，不足以判断五卡净性能。

## §39 通用 MMVQ 多列 helper 的静态与 query 对照（2026-09-29）

在候选树临时禁用 `HAS_MMVQ_DOT_MULTI` 分支，以先前不可达的逐列路径作 A、当前 2–4 列共享权重路径作 B；已恢复候选源码与主仓一致。`GGML_VK_FORCE_MMVQ=1` 下的单个矩阵收缩，Q5_K/Q6_K、`m=4096,k=2560,n=2/3/4`，固定量化权重与 F32 激活，A/B 六组输出**逐字节一致**，且日志确实走 `MUL_MAT_VEC q5_K/q6_K`。容量 4 的 RADV 静态指标（**helper / 逐列**）分别为 Q5_K VGPR=56/40、instructions=2818/4305，Q6_K VGPR=56/40、instructions=2929/4716；均无 spill。helper 减少重复解包指令却延长跨列寄存器生存期，静态 subgroups/SIMD=18/24；二者都不直接证明实际占用或运行时间。

同 GPU 独占、同图同数据及 `env -i` + `RADV_DEBUG=nobolist`，各臂 65 个 GPU query 丢首个；查询器强制同步、关闭 replay。余 64 次中位数 µs（**helper B / 逐列 A**）：

|列数|Q5_K|Q6_K|
|---|---:|---:|
|N=4：B–A–A–B|86.82、86.84 / 88.60、88.56|87.68、87.74 / 89.10、88.46|
|N=3：各单臂|78.08 / 77.98|78.68 / 77.92|
|N=2：B–A–B|67.04、67.52 / 66.60|69.56、69.44 / 68.56|

逐次原始 query 保存在 `generic-mmvq-multicol-query-20260929.tar.gz`。该重复小图的 N=4 改善约 0.7–1.8 µs，N=2/3 的方向相反或接近噪声；不能仅凭指令降幅断言更快，也不能拿 1 µs 的缓存驻留单卡差值推翻尚未对照的其他量化类型。§40 曾对 Q4_K 做窄列回退，§41 用真实矩阵尺寸交错对照后撤回；**§42 又依据实际服务尺寸将通用 Q5_K 排除出 helper**。TP5 默认仍禁用 MMVQ。未验证真实五卡 MMVQ 净吞吐、接受率/输出等价或理论全硬件最优；需要同 `pred_n`、接受率和账本条件的服务 ABBA 才能做相应结论。

## §40 跨量化族审查发现 Q4_K 窄列回退（2026-09-29；阶段性选择已由 §41 撤回）

按实际编译分支覆盖 `K_PER_ITER=16` 的 Q2_K/Q3_K/Q4_K 和 `K_PER_ITER=8` 的 Q4_0/Q8_0/MXFP4；各做宽 2560、4096 行、N=2/4 的强制 MMVQ 小图，同输入下 helper 与此前逐列路径的 **12/12 组输出逐字节一致**。容量 4 的 RADV 静态量（**helper / 逐列**）如下；两路均无 VGPR spill：

|类型|VGPR|instructions|
|---|---:|---:|
|Q2_K|56 / 48|4367 / 6031|
|Q3_K|56 / 40|3153 / 5263|
|Q4_K|56 / 40|2589 / 3706|
|Q4_0|32 / 32|2127 / 2610|
|Q8_0|40 / 32|1879 / 2333|
|MXFP4|40 / 40|3755 / 5807|

每臂 65 个 GPU query 丢首个、同独占设备与 `env -i`，在其他类型的 N=2/4 中两路中位数差值大多在约 1 µs 内；Q4_K 的 N=2 则 helper=63.88、复测 64.28 µs，逐列=59.84、复测 59.76 µs。N=3 补测 helper=74.08、逐列=71.94 µs；N=4 为 helper=83.80、复测 83.70，逐列=83.30、复测 83.28 µs。Q4_K N=2 的 helper 虽把静态指令从 2155 降到 1778，却把 VGPR 从 32 增到 40；只按“权重少解包”选路径会漏掉该回退。

**当时选择（§41 已撤回）**：Q4_K 的 N=2/3 回到原逐列计算，N=4 仍保留多列 helper（其重复权重解包确被删除，但本图未证明时间收益）；其他量化类型继续原 2–4 列门控。`NUM_COLS` 是专门化常量，门控在管线专门化时解析；没有新增运行时 type 分支。该版本 Q4_K 的 N=2/3/4 热机中位数分别为 59.96/71.68/83.80 µs，输出与选中的旧/新路径逐字节一致；`bin/test-backend-ops` 强制 MMVQ 的 Q4_K × N=2/3/4 三项 CPU 对照均通过。原始 query 和编译资源位于 `generic-mmvq-family-query-20260929.tar.gz`。这些仍是单 GPU、缓存驻留图的局部策略证据，不是其他 GPU 的通用阈值或五卡服务吞吐资格；默认 MMVQ 禁用不变。

该版本另做五卡 `GGML_VK_FORCE_MMVQ=1` + Qwen4EXP TP5/MTP 单请求 smoke：五个 rank 的 `[vulkan-mmvq-policy]` 均为 `forced`，账本五 rank `late=0`，服务 HTTP 200，`pred_n=171`、72/98 draft accepted，CJK=243、distinct ratio=0.679、标点=21；decode chain 锁定并复用 97 图。启动与 prefill 仍有 `CRITICAL-FALLBACK`，不能描述为全程零回退。输出保存在 `q4-forced-five-gpu-smoke-20260929.tar.gz`。单次 52.65 tok/s 不具备性能资格：与 §38 的 F32 smoke 不同 `pred_n`（171 vs 160）、接受数与停止原因，且服务日志未逐个标明 Q4_K N=2/3 实际派发；**直接的 Q4_K 路径正确性由上述 Vulkan 图与 CPU 对拍证明，不由此请求推断**。

## §41 实际 GGUF 形状推翻 Q4_K 单一类型阈值（2026-09-29）

§40 用 `k=2560,m=4096` 单一合成形状下决定所有 Q4_K 的 N=2/3：范围过宽。直接读取六个 Qwen4EXP base GGUF shard 元数据，**0 个 Q4_K tensor**；Q4_K 出现在 MTP GGUF，例如 `blk.48.attn_q.weight` 为 `(k=2560,m=12288)`、`attn_output.weight` 为 `(6144,2560)`、`nextn.eh_proj.weight` 为 `(5120,2560)`，另有 `(10240,320)`、`(2560,640)`、`(2560,512)`。这些是元数据中的真实矩阵尺寸，**不是服务日志已证实的 N=2/3 派发**。所有模型元数据与查询原始日志归档在 `q4-real-model-shapes-20260929.tar.gz`。

同 Vulkan0、强制 MMVQ、固定输入/量化测试权重，对真实尺寸的七组 N=2/3 矩阵比较逐列 A 与共享解包 B：**7/7 组输出逐字节一致**。仍是缓存驻留的合成图，每臂 65 次 GPU query 去首个；关键形状做 A–B–B–A 交错，中位数 µs：

|MTP 元数据对应尺寸 / token 数|逐列 A 两臂|共享 B 两臂|
|---|---:|---:|
|`attn_q`，k=2560,m=12288,N=2|133.76、133.48|137.88、138.28|
|`attn_q`，k=2560,m=12288,N=3|166.04、166.08|166.06、165.96|
|`attn_output`，k=6144,m=2560,N=2|84.52、84.56|76.10、76.16|
|`nextn.eh_proj`，k=5120,m=2560,N=2|67.84、68.12|67.52、67.60|

特别是 `attn_output` 形状，§40 的按类型整体逐列回退在同图上损失约 8.4 µs；而 `attn_q` N=2 保留 helper 约损失 4.4 µs。按矩阵宽度分配两套 Q4_K 管线可以针对这些图选不同路径，但需要新的 host 专门化/资源与其他设备验证，且当前 base GGUF 根本没有 Q4_K；只为了这组单卡缓存驻留 query 引入该复杂度不合算。**撤回 §40 的 Q4_K 专门门控**，恢复 Q4_K N=2–4 的共享解包，不再把 k2560/m4096 特例当通用收益。该版的三种代表尺寸输出与 helper 对照逐字节相同，Q4_K × N=2/3/4 CPU 对照 3/3 通过；默认 TP5 MMVQ 仍关闭。§42 对**另一量化类型 Q5_K** 的通用核做了独立回退，没有改变 Q4_K 门控。没有五卡服务吞吐资格或全硬件极限证明。

## §42 真实派发形状下的 Q5_K 通用 MMVQ 多列回退（2026-09-29）

§39 的 `m=4096,k=2560` 上 Q5_K N=4 helper 略快，但五卡 `GGML_VK_PERF_LOGGER` 核标签和模型分片显示频繁命中的**通用** Q5_K 是 `m=2560,k=1280/1024,n=2/4`；融合 Q5_K attention 投影使用不同 shader，不能混算。对这些实际尺寸造固定量化权重/非零激活的小图，强制 MMVQ，Vulkan0 排他执行：A=暂时禁用 `HAS_MMVQ_DOT_MULTI` 的逐列旧路径，B=共享权重解包，A/B 七组 Q5_K/Q6_K 的输出全部逐字节相同。每臂重复 65 次 query，去首个后取 64 次中位数（µs），关键形状 B–A–A–B 镜像：

|类型与形状|B 两臂|A 两臂|结论仅限图|
|---|---:|---:|---|
|Q5_K `k=1280,m=2560,n=2`|51.86、51.88|45.04、45.16|B 稳定慢约 6.8 µs|
|Q5_K `k=1024,m=2560,n=2`|41.76、41.76|40.36、40.32|B 慢约 1.4 µs|
|Q5_K `k=1280,m=2560,n=4`|60.28、60.28|56.38、56.56|B 慢约 3.8 µs|
|Q6_K `k=2560,m=10240,n=2`|135.18、134.42|133.50、133.60|差值不足以推翻 Q6 其他形状 N=4 的结果|

另测 Q6_K `m=128/2560,k=2560,n=2`、`m=128,n=4`：小行数约 32 µs 两路接近，大行数单臂 B/A=53.24/51.62 µs；没有证据支持对所有 Q6 形状做类型级回退。Q5_K N=2 的 RADV 静态统计 B/A：VGPR=40/32、instructions=2006/2504、Subgroups per SIMD=24/32，无 spill；B 虽少静态指令却增加寄存器生存期，资源与计时的方向相容，不把静态值当实测 occupancy。原始日志和输出归档于 `q5-q6-production-shapes-20260929.tar.gz`。

**当前实现**：仅将 `mul_mat_vecq_funcs.glsl` 中通用 Q5_K 从 `HAS_MMVQ_DOT_MULTI` 编译门移出，重用原 `mmvq_dot_product` 逐列路径；Q4_K/Q6_K 和融合投影保持原实现，无 host 形状专门化。主仓与候选树 shader 文件 SHA256 相同，最终 `bin/llama-server bin/test-backend-ops` 构建成功；Q5_K 三个上述尺寸与原 B 输出逐字节相同，Q6_K `k=2560,m=10240,n=2` 也逐字节相同。Vulkan0 强制 MMVQ `MUL_MAT` Q4_K/Q5_K/Q6_K × N=2/3/4 小图 CPU 对照 9/9 通过（另两条 Q4_K batch case 不受支持，不计入通过）。该版 Q5_K N=2 静态 VGPR=32、最终中位数 `45.08/40.46` µs（分别 k1280/1024），N=4 k1280 为 `56.60` µs，符合 A 镜像。

最终二进制另外以五张 RX 6800、`GGML_VK_FORCE_MMVQ=1`、Qwen4EXP TP5+MTP、`--tp5-latebind off` 运行一次服务请求：HTTP 200，`pred_n=171`、72/98 draft accepted、汉字243、中文标点21；五 rank 解码账本同为 `late=0,dispatches=4337,barriers=4019,copies=192`，图复用97。原始 HTTP 响应在同一归档内；服务启动/预录及 prefill 出现 `CRITICAL-FALLBACK`，不宣称全程无回退。这是当前**确切源码**的端到端连贯性 smoke，不是服务性能对照。

限制：query 强制提交/等待、重复使用缓存驻留权重，不是显存实际流量或五卡 tok/s；Q5_K 在其他 GPU/大矩阵可能反向。默认 TP5 MMVQ 仍禁用；该最终源码尚无五卡同负载服务 ABBA，不能据此宣称五卡净收益或理论/指令级全硬件最优。

## §43 Q8_0 高频图的下界与 Q8_1 激活冗余舍入（2026-09-29）

先于运行时判断，从 §41 归档的五卡 `GGML_VK_PERF_LOGGER` 日志按核标签聚合：**通用** Q8_0 `m=10240,k=320,n=2` 与 `m=320,k=10240,n=2` 各出现 5315 次，`m=4,k=10240,n=2` 出现 5270 次。这是带同步 logger 的累计**核标签次数**，不能直接换算服务时间。`types.glsl` 的 Q8_0 每 32 权重为 2 字节 scale + 32 字节量化值；前两种矩阵各有 `10240×320/32×34 = 3,481,600` 字节，共 6,963,200 字节的**独立权重**，双 token 的 26,214,400 次浮点操作（按乘加各一）对一次逻辑权重读取的强度约 3.76 FLOP/byte。该值既不是物理显存流量也不是实测 roofline：缓存驻留、合并加载和五卡队列仍未知。不同矩阵无跨矩阵权重复用，第二矩阵的激活又依赖第一矩阵输出；不能通过在单一通用 shader 内预加载另一矩阵的权重消除其读取。

Q8_0 共享解包 helper 仍对每列执行必要的整数点积。将现有 `mul_mat_vec_q8_0_q8_1_f32_subgroup_no_shmem.spv` 专门化为 `NUM_ROWS=1,NUM_COLS=2,BLOCK_SIZE=32` 并做 SPIR-V 优化后，可见同一权重 scale/已打包值分别供两个 token 的 dot 使用；没有发现该处另一个可消掉的跨列权重读取。§40 的 Q8 模型尺寸缓存驻留 query：`k=320,m=10240,n=2` helper/逐列约 63.10/63.21 µs；`k=10240,m=320,n=2` 约 39.18/39.57 µs，而 N=4 两种形状方向相反，故不再按 Q8_0 类型整体撤销 helper。这些仍不证明硬件或服务最优。

**可证明的指令冗余在生产端**：`quantize_q8_1.comp` 先令 `vals = round(vals * d_inv)`，随后在写 `i8vec4` 时又算 `round(vals)`。一次舍入后的每个有限分量已为整数，第二次舍入是恒等变换。旧版 x4/subgroup SPIR-V（含 `spirv-opt -O`）实有两条向量 `Round`；只删除第二条，不改缩放、sum、输入/输出布局或同步。新旧 RADV Q8_1 量化管线静态 instructions **90→86**、VALU **56→52**、VGPR **32→32**、无 spill；同图 Q8_0 收缩管线 instructions/VALU **1198/687** 不变。这是编译指令数，不是已测的速度或真实功耗。

强制 MMVQ 在实际形状 `k=320,m=10240,n=2` 与 `k=10240,m=320,n=4` 的新旧 Q8_0 输出逐字节相同；Q5_K 三组、Q6_K 一组与 §42 改动前归档的同输入输出逐字节相同。最终二进制的 Vulkan0 CPU 对照 Q8_0/Q4_K/Q5_K/Q6_K × N=2/3/4 为 12/12 通过，另有四条非受支持 batch case 未计入。五卡当前版本单请求返回 HTTP 200，`pred_n=171`、72/98 draft accepted，内容与 §42 旧版逐字节相同；解码五 rank `late=0,dispatches=4337,barriers=4019,copies=192`，启动和 prefill 仍有 `CRITICAL-FALLBACK`。原始新旧 GPU 数值、RADV 资源、SPIR-V 和最终 HTTP 响应见 `q8-quant-round-proof-20260929.tar.gz`。服务输出相同不是性能资格；未获得同工作量服务 ABBA、真实物理流量或全硬件理论极限证明，TP5 默认 MMVQ 禁用未改。

## §44 Q6_K 词表规模的共享解包与寄存器生存期（2026-09-29）

先按真实五卡日志中的 `q6_K m=49664,n=2,k=2560` 分析理论机会。Q6_K 每 256 权重 210 字节，该分片矩阵至少读取 `49664×2560/256×210 = 104,294,400` 字节独立权重；逐列源码会两次发出权重解包，而现有 helper 每次解包供两个 token 使用。该矩阵约 99.46 MiB，小于 RX 6800 的 128 MiB Infinity Cache 容量，但实际服务中的其他权重可能逐出它，**源码加载次数不能推断 HBM 流量减半**。同一日志的 `m=248320,n=1,k=2560` 至少 521,472,000 权重字节，没有跨列复用；其改善不能靠改 N=2 helper。§42 的 10240 行小矩阵不足以代替这个容量临界形状。

在候选树上仅临时禁用 `HAS_MMVQ_DOT_MULTI` 比较 A=逐列与 B=当前共享解包；另试 C=只在 Q6 helper 中**先解包单行权重、再逐列加载量化激活**，缩短两个激活列的同时生存期。C 保持原有 Q6 的整数点积与浮点累加次序。同一固定 Q6 量化矩阵、非零 F32 激活，Vulkan0 强制 MMVQ，各臂 65 次 GPU query 丢首个；三版在 49664、10240 行 N=2 的输出逐字节一致。`k=2560,m=128,n=2` 的 RADV `shaderstats,nocache` 记录（A/B/C）为 VGPR **32/40/32**、instructions **2773/2162/2218**、VMEM **186/144/144**，均无 spill；C 降低 B 的 VGPR，却没有减少其静态权重访存指令，另多 56 条静态指令。

排他设备、镜像顺序 B–A–A–B–C–B，`k=2560,m=49664,n=2` 余 64 次查询中位数（µs）：B=**552.22/552.40/550.04**，A=**548.18/550.96**，C=**553.38**。较小的 `m=10240` 对照为 B=**134.68/134.64**、A=**133.64/133.46**、C=**133.98**。A 在这两组重复小图中略快于 B，但 A 多解包并对 Q6 N=4 有独立成本，且逐臂差距约 1–4 µs；C 降 VGPR 并未在本轮最大图兑现净收益。**三版均非静态资源与时间的严格支配者**，没有为这两个缓存驻留单卡图增加设备/形状专门化；保留主仓 Q6 helper、TP5 默认禁用 MMVQ。候选源码已撤回并重建 `bin/llama-server bin/test-backend-ops`；恢复后的 B 在大图再通过 64 次查询及逐字节对照。原始日志、输出、探针和被拒绝的 C 补丁归档在 `q6-vocabulary-shape-20260929.tar.gz`。

限制：单卡强制 MMVQ、重复权重、强制同步的 logger 查询不测五卡权重换页、HBM 实际流量、服务吞吐或所有硬件的理论最优；对当前确切源码仍缺同 `pred_n`、接受率、输出及账本的五卡服务 ABBA。`m=49664,n=2` 的 104 MB 只是输入权重字节下界，不是已证明的物理 VRAM 计数或不可改进的 GPU 时间下界。

## §45 Q6_K 精确六位点积的位编码反证（2026-09-29）

再次审查 Q6_K `repack4`：旧代码用三步按字节计算 `u−32`，其中 `u∈[0,63]`。候选先计算 `((u xor 32) << 2)` 的有符号八位表示；它对全部 64 种 `u` 恒等于 `4×(u−32)`，32 位整数点积的 16 项绝对值不超过 262144，末尾算术右移二位可**精确**还原旧整数和，即使结果为负。浮点乘法次序不变。只改通用 Q6 N=1 和共享 helper 的 Q6 N=2–4，融合 attention 投影经同名 `repack4(ib,iqs)` 包装器仍用原签名和原表达式；不是近似 Q6 替代。

在候选树编译后以 Vulkan0 `shaderstats,nocache` 测 `k=2560,m=128`：N=1 旧/候选 instructions **1793/1797**、VALU **1233/1247**、VGPR **32/32**、VMEM **114/114**；N=2 为 instructions **2162/2180**、VALU **1507/1535**、VGPR **40/40**、VMEM **144/144**。两种输出与旧版均逐字节一致。**代数步数下降不代表 RADV 编译后指令下降**，故按事前静态筛选否决；没有进行词表尺寸计时，也没有任何五卡加速主张。候选已撤回，主仓与候选 Q6 函数源码 SHA256 一致，重新构建 `bin/llama-server bin/test-backend-ops`，恢复后的 N=1 小图输出与旧版逐字节相同。补丁和静态日志归档 `q6-scaled-byte-rejected-20260929.tar.gz`。

## §46 Q6_K 词表派发的物理访存计数（2026-09-29）

为了不把 §44 的 104/521 MB **逻辑权重字节**误当成设备流量，改造已验证的独立 `VK_KHR_performance_query` 探针：Q6_K 的 210-byte block 填为 ql=`0x11`、qh=`0x55`、scales=1、d=1，Q8_1 激活全 1，所有输出应精确为 `−15×2560=−38400`。Vulkan0 独占设备，只把计算 dispatch 放在查询区间；上传、热机及读回在区间外。五个计数器需两个 pass；同形状空查询 waves/VMEM/VRAM 全零，真实每次输出逐项核对。RADV counter 12 标称 `VRAM read size`，不是从 GLSL VMEM 静态数量推测的读量。

`k=2560`，单位十进制 MB，B 为**当前** Q6 N=2 helper，A 为临时逐列；cold / warm 各一次，恢复后 B 另测镜像：

|矩阵与查询|逻辑权重 MB|VRAM read MB（cold / warm）|L2 命中率（cold / warm）|
|---|---:|---:|---:|
|`m=49664,n=1` 当前|104.2944|174.932 / 186.314|3.643% / 3.428%|
|`m=49664,n=2` B|104.2944|223.461 / 241.698；镜像 248.327 / 223.512|5.508% / 5.113%；镜像 4.984% / 5.507%|
|`m=49664,n=2` A|104.2944|210.575 / 210.804|5.825% / 5.819%|
|`m=248320,n=1` 当前|521.472|762.137 / 745.847|4.035% / 4.119%|

空查询的 L1/L2 比值为 nan（零事件）；真实图 L1 命中约 0–0.002%。双列 B 的实测 VRAM 读**只比同一 104 MB 矩阵的 N=1 大约 1.2–1.4 倍**，而非两倍；但 B 自身读量仍约 223–248 MB，A 更低约 211 MB。N=1 全词表实测读约 746–762 MB，是 521 MB 独立权重字节的约 1.43–1.46 倍。差值包含未分离的缓存行为/事务粒度，**不是已归因于某一条重复 GLSL load 的可消除字节**。这也解释了为何仅凭 B 的静态 VMEM/解包更少不能证明内存或时钟收益；§44 的匹配单图 A 略快，但不足以为 Q6 N=2 添加设备相关门控并证明五卡净收益，N=4 仍有独立取舍。源码恢复到主仓，重建 `bin/llama-server bin/test-backend-ops`，恢复后的 B 真实 query 再次输出逐项正确。探针、逐次计数与空查询原始输出见 `q6-radv-vocabulary-traffic-20260929.tar.gz`。

**资格边界**：这是固定内容 Q6 矩阵的单 GPU、独立 dispatch；L2 比值不是 Infinity Cache 驻留证明，VRAM 数值不可外推为模型实权重或服务五卡流量。`VRAM read / 理论最少独立字节 > 1` 说明还有**上界空间**，不证明剩余字节全部可消除或已到指令级最优。仍缺确切源码同工作量、同接受率、同输出的五卡服务 ABBA，不声称全局理论极限或净吞吐获益。

## §47 Q6_K 同块 QL/QH 的 subgroup 共享读取（2026-09-29；否决）

针对 §46 高于逻辑下界的真实 VRAM 计数，先证明重复源：通用 Q6 `K_PER_ITER=16`，每 lane 的 `ib=floor(col/32)`、`iqs=4×(tid mod 2)`；`tid` 与 `tid xor 4` 的 `ib_k`、`ql_idx` 相同，仅 nibble shift 不同；`tid` 与 `tid xor 2/4/6` 的 `ib_k`、`qh_idx` 相同，仅 bit shift 不同。每个完整 16-lane Q6 block 中，QL 的 16-bit 源字至少被两 lane 读取、QH 被四 lane 读取。行起点与 Q6 block 对齐，最后不足 512 个 K 元素时有效 lane 按完整 16-lane block 参与迭代；32/64-lane subgroup 的源 lane 均处于同一活动组。候选在通用 MMVQ 对源 lane 有条件加载 QL/QH，再分别以 `subgroupShuffle` 分发。**候选改动的是公共 `mul_mat_vecq_funcs.glsl`，也由融合投影 include；本轮增量构建只生成通用 shader，没有证明候选的融合管线可从零编译。**它已否决，不能将通用图结果写作融合路径的兼容性证明。

在候选树编译、Vulkan0 对照 `k=2560,m=128`，旧版 / QL+QH 候选静态 N=1 instructions **1793/2240**、VGPR **32/32**、VMEM **114/114**；N=2 为 **2162/2617**、VGPR **40/32**、VMEM **144/144**，均无 spill。QH-only N=2 为 **2380 instructions/40 VGPR/144 VMEM**，QL-only 为 **2389/40/144**；单独共享均无寄存器收益。这些是静态指令，带 lane 掩码的加载不能由静态 VMEM 数直接计数。N=1/2 的固定非零量化小图及 N=2 的 `m=49664,k=2560` 实际矩阵尺寸图，QL+QH 与旧版输出逐字节相同。

在排他 RX 6800 上，后者每臂 65 次强制同步 GPU query 去首个：QL+QH 候选 N=2 中位 **646.98 µs**，恢复原版同图 **551.80 µs**（§44 原版两臂 552.22/552.40）；候选约慢 95 µs。另一方面，独立 Q6 perf query 的固定内容 `m=49664,k=2560,n=2` VRAM 读从 §46 原版约 **223–248 MB** 降到候选 **204.98/202.95 MB**（cold/warm），输出均为预期 `−38400`。**实际减少 VRAM 读取仍不足以抵消分支、shuffle 和其它动态成本**；所以不以流量下降宣称胜利，三个候选均撤回。主仓/候选 `mul_mat_vecq.comp`、`mul_mat_vecq_funcs.glsl` 已恢复逐字节一致，最终重建 `bin/llama-server bin/test-backend-ops`；恢复后的实际尺寸图再次通过逐字节核对。补丁、原始 stats、查询与输出归档在 `q6-subgroup-sharing-rejected-20260929.tar.gz`。

仍未穷尽不增加每轮 subgroup 指令的访存布局方案；单卡固定输入及缓存驻留查询不足以说明实模型五卡 HBM、相同接受率下的服务净吞吐或全硬件理论极致。TP5 默认禁用 MMVQ，不改变任何生产 shader。

## §48 Q6_K 相邻输出行复用量化激活的反证（2026-09-29）

通用 `mul_mat_vecq.comp::iter` 每个列先载入一次 Q8_1 激活，再遍历 `NUM_ROWS` 个**独立权重行**并分别累加；增加行容量不会更改同一输出行的 dot 顺序。宽 2560 的 Q8_1 激活独立载荷为 `2560/128×144 = 2880` 字节：两行分别派发时源码载入两次，`NUM_ROWS=2` 可在一个工作组内共享一次，而两行各自的 Q6 权重至少 4200 字节不变。当前 RDNA2 host 的整数 dot Q6 管线 `rm_kq_int=1`，若要量产两行必须新增/改变管线专门化与对应派发计数；**这不是已有服务已经用上的容量**。

先只在独立 Vulkan 探针把 `NUM_ROWS=1/2` 专门化与 `dispatch_x=m/NUM_ROWS` 一起变化，SPIR-V、固定 Q6/Q8_1 字节及输出约束相同；每个输出元素均核对 `−38400`，但常量矩阵不构成任意逐行输入的数值对拍。排他 RX 6800、上传/热机/读回不计入查询，RADV `VRAM read size`（MB 十进制）：

|实际矩阵尺寸，N=1|一行 cold / warm|两行 cold / warm|方向|
|---|---:|---:|---|
|`m=49664,k=2560`|194.990 / 186.875|285.106 / 285.183|两行约多 90–98 MB|
|`m=248320,k=2560`|745.873 / 732.660|1255.947 / 1325.771|两行约多 510–593 MB|

同一 standalone 探针的 `k=256,m=16,n=1` RADV 静态统计：一行 / 两行 instructions **1773/3084**、VMEM **114/200**、VGPR **64/64**、Subgroups/SIMD **16/16**，均无 spill；动态计数的 VMEM load 和 waves 在两行时更少，但**实际 VRAM 读更多**。这一组 standalone 管线未声明生产 host 的 subgroup-size 控制，不能把它的 VGPR 数与 §45 的生产图 VGPR=32 直接比，也不能把流量回退归因到 VGPR 翻倍。安全结论仅为此探针的两种真实矩阵尺寸都未兑现物理读取收益：不引入新的 Q6 生产管线、阈值或五卡服务实验。源码与最终构建未改动；独立探针及全部原始计数归档 `q6-row-tile-rejected-20260929.tar.gz`。

该反例不能证明另一个布局、真实 GGUF 字节或生产同步下也回退，更不能用于宣称理论/指令级最优。§46 的物理读取仍显著超过独立权重字节；需要有可解释的指令/事务改进并且通过匹配五卡负载资格才能推动默认 MMVQ 门控。

## §49 当前源码的五卡 MTP forced/default 交错资格（2026-09-29；**不合格**）

冻结 `/var/tmp/tp5-final-build/bin/llama-server` SHA256 `da35702a87e9918cea3101806a14476bf15b53a4a68bad43d72eab451f1e8fdb`、`libggml-vulkan.so.0.18.1` SHA256 `43b8d18401b5c82f08f16e5fa2a38a4de4b3d8bee95acd415438fec34ee3666b`；构建 `CMAKE_HOME_DIRECTORY=/var/tmp/tp5-candidate-k1a`，主仓当前决策基座提交 `9d05b44cc`。五张 RX 6800 初态忙碌率均为 0、VRAM 约 17 MB，无既存 `llama-server`；一次只让本轮进程在 `/var/tmp/tp5-gpu.lock` 下运行。使用相同主/草稿 GGUF、`--tp5 qwen4exp-af -md ... --spec-type draft-mtp`、relay/F32、LateBind off、greedy seed=12345、`n_predict=171`、同一中文提示；每臂先一次 8-token 热机，再一次 171-token 请求。只给基础 `RADV_DEBUG=nobolist`，forced 臂加 `GGML_VK_FORCE_MMVQ=1`；默认臂由合格 TP5/relay 路径设置 MMVQ disabled。每个子服务只停自己的 PID，四臂均正常退出。

镜像顺序 forced–default–default–forced，同二进制同提示。下表 decode 时间**仅作原始观察，未获比较资格**；SHA 为响应正文 SHA256 前 16 位：

|臂|pred_n / prompt_n|draft accepted/generated|decode ms|正文 SHA 前缀|测量期间 `CRITICAL-FALLBACK`|
|---|---:|---:|---:|---|---:|
|forced-1|171 / 39|129 / 233|5215.742|`84e0e2b8a08bd50c`|2|
|default-1|171 / 39|130 / 235|4018.389|`eb0f6bb83a7a8213`|1|
|default-2|171 / 39|130 / 235|3979.788|`eb0f6bb83a7a8213`|1|
|forced-2|171 / 39|129 / 233|4297.115|`84e0e2b8a08bd50c`|2|

四臂 HTTP 200、五 rank 每组账本内部一致且 `late=0`；forced 测量尾组为每 rank `dispatches=4522,barriers=4017,copies=192`，默认尾组为 `4242,3497,192`，对应模式不同的 dispatch 图，不能要求跨模式账本相同。每个模式内部重复输出和接受数稳定，但**跨模式正文不同，接受量 129/233 对 130/235，且测量期仍有 graph fallback**；所生成 171 token 仅包含 `<think>` 推理片段，没有最终文章。不能用表中毫秒数断言净加速/退化、连贯最终回答或全硬件最优；TP5 默认 MMVQ 禁用不变。原始四份 HTTP JSON、服务日志、资格提取表见 `mtp-current-binary-unqualified-abba-20260929.tar.gz`。

下一步资格条件仍为相同 `pred_n`、实际正文、draft 接受率/数量、五 rank 账本内部一致，且定位解码 graph fallback；理论方面 §46 的 Q6 物理读取下界与 §47–48 的负例也未收敛。不能因多跑相同不合格请求或者单一 GPU ISA 指标宣称已达到极致。

## §50 五卡解码 replay miss 的证据边界（2026-09-29）

只分析 §49 已归档的四份服务日志，不重跑已观测失败。所有 `CRITICAL-FALLBACK` 前第一个 `meta-chain-miss` 都在 **stage=0/rank=0/nodes=94**；startup 缓存 size=0，首次 draft 图已有缓存 size=97 时 fingerprint 报 `hc_init` 的 `ne[2]` 从 2 到 7、`nb[2]=40960` 未变，属于不能安全重用同一录制命令的**明确形状变化**。热机期间缓存 size 依次经过 99/196/293/330/427/524，测量时 default 两臂在 size=621 各有一次 miss，forced 两臂在 621 和 718 各有一次 miss；这些 miss 均记 `scratch_gen=0`，解码时没有 `cache-invalidate-all`，该消息只出现在服务退出清理。

`ggml_vk_tp5_get_cached_cmd_bufs` 先按图指针查，再跨缓存条目查找；要求节点数、scratch generation、完整 tensor fingerprint 均匹配。fingerprint 比对 shape/stride、op 参数、根指针及 view 偏移、实际 VkBuffer handle、wire/relay 专门化；每个录制命令还持有它用过的 buffer owner。重新绑定一个未证实等价的命令缓冲，可能读错模型/激活地址，不能为消除告警松掉这些条件。首次 shape 变化已证实，但**测量期缺少逐项 fingerprint 证据**：各类 `[FP-TMISS]` 诊断在 startup 前十次被门控耗尽，因此只看 size 增长与固定 `scratch_gen`，无法区别未见 token 拓扑、分配地址变化或其它有效 miss。不能把 stage0 的新图指针本身当作根因，因为实现已跨指针搜等价 fingerprint。

没有找到可安全修的 replay 错误，不改缓存 key、不隐藏 fallback，也不以额外预热掩盖首请求成本。若继续攻此项，先在命中该请求的**实际**日志中给测量期 miss 加可验证的字段级诊断，再对照该图的录制/当前 tensor 绑定；在此之前 §49 的 ABBA 仍不合格，默认 MMVQ 门控保持禁用。
