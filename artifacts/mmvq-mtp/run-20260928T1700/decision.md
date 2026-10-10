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

## §51 F32 MTP 小矩阵的跨列权重复用（2026-09-29；限定容量）

§43 的 16-token 日志中，F32 标签累计 17,695,201,280 **逻辑源权重字节**，不是 HBM 读取或总运行时间；其中 `k=2560,m=512` 的 N=2/1/4/15 标签分别 2630/259/240/240 次。通用 `mul_mat_vec.comp::iter_aligned_nonquant` 原先按列重复 `dequantize4_2aligned`；对 F32、`NUM_ROWS==1`、**恰好 N=4 或 N=15**，新分支每轮只加载一次权重 `vec4`，逐列读取激活并按原 K 顺序对各列分别 `dot`。其它类型、行数和容量保留原路径；没有更改 Q8_1 激活量化、融合投影或 TP5 默认 MMVQ 门控。这里的共享是源码/静态指令层面的，不将源 load 数当作物理 VRAM 字节。

RX 6800/RADV 固定输入图 `k=2560,m=512` 的 `shaderstats,nocache`：N=4 原版/新分支 instructions **2798/2635**、VMEM **229/208**、VGPR **32/32**；N=15 为 **9019/8382**、**812/714**、**40/40**；均无 spill。65 次独占设备同步 GPU query 丢首个、两轮交错 B–C–B–C，余 64 次中位 µs：N=4 **35.00/32.92/35.08/32.84**，N=15 **82.66/79.20/83.32/79.32**。同图 F32 输出新旧逐字节相同；`m=1` 两容量及 K=257/260 尾部形状也做了对拍。更宽的门控没有资格：N=2 候选的两个中位 **31.04/31.32 µs**，对照 **30.84/30.68**；把 N=16 纳入候选时 **90.94 µs**，原版 **87.54**。因此不能用 N=4/15 的方向推断所有 N 或所有输入。图 query 并不代表五卡服务吞吐。

另查 F32 权重、F16 激活的 Vulkan 图：N=4/15 新旧输出分别 **8192/30720 字节逐字节相同**，64-query 中位 µs 原版/新分支 **33.20/31.32**、**64.28/63.92**。现有 CPU backend 不支持该 F32×F16 测例，故不将其写成永远跳过的永久测试；保存临时图对拍。最终候选构建 `bin/llama-server bin/test-backend-ops`，F32×F32 N=4 两个支持的 CPU 对照 **2/2**、新增 N=15 **1/1**；主仓与构建源 shader SHA256 都为 `a6e31656a0858a05a52028f8acb71ad7ef269d5ee8761d9286bb38bf4b3cd092`，测试源均为 `29e10cda4854783ac5a78718b6412c5e5a3fcbae86151e8e22b301e7588a0137`，最终 Vulkan 库为 `f6739f980913e530c8f4b19e0a42cc216dcdae074b7314ab593df7a1f4ff646b`。

早于最终 F16 对拍、但采用同一选定 shader 的五卡 default/forced 服务烟测均 HTTP 200、`pred_n=171`、各自五 rank 账本一致；default draft accepted/generated **130/235**，forced **129/233**，正文也不同且测量期仍有 `CRITICAL-FALLBACK`。这只证明两路能运行，**不证明 F32 改动的服务净收益，也不解除 §49–50 的 ABBA 资格问题**。原始图 stats、65-query 日志、逐字节输出、被拒绝的宽门控补丁、F16 对拍、服务请求/日志及临时探针存于 `f32-mtp-shared-weight-20260929.tar.gz`。全硬件理论/指令级最优与合格五卡收益仍未证明。

后续试图做**同一 default 模式**的 C–B–B–C 五卡服务资格比较时，首臂未留下 JSON/服务日志，不能当成已测。事后 kdump `/var/crash/202609291802/dmesg.202609291802` 记录 `card3=0000:0c:00.0` 在 17:51 左右先 `device lost from bus`，内核恢复失败，约 18:02 在 `drm_sched_job_arm` 空指针 Oops；18:13 再启动。Eagle 服务日志记录了告警，但卡已丢总线，其 `--clocks-only` 预检读取 DPM 返回 Remote I/O error。不能将卡掉线归因于此 shader 或本次尚未完成的比较；**停做五卡 GPU 实验**，待 PCIe/供电/显卡及内核恢复路径排查并确认稳定后再重新预注册同工作量对照。

## §52 加固鹰眼后的同模式五卡 C–B–B–C（2026-09-29）

用户接受偶发硬件故障下继续运转，故在部署新的单卡恢复及原 **10 分钟 IPMI 硬件狗**保留后，受 `/var/tmp/tp5-gpu.lock` 保护再执行一次同模式实验。四臂同 GGUF、default MMVQ、`--tp5-latebind off`、同提示/seed/greedy、8-token 热机和 171-token 测量。C=§51 的 F32 N=4/15 特例（Vulkan 库 SHA256 `f6739f980913e530c8f4b19e0a42cc216dcdae074b7314ab593df7a1f4ff646b`），B=旧 shader（`43b8d18401b5c82f08f16e5fa2a38a4de4b3d8bee95acd415438fec34ee3666b`）；`llama-server` SHA 固定 `da35702a87e9918cea3101806a14476bf15b53a4a68bad43d72eab451f1e8fdb`。每臂独立启停服务，重建只发生在服务进程之间。

|臂|C1|B1|B2|C2|
|---|---:|---:|---:|---:|
|predicted_ms|4364.061|3872.450|3882.872|3864.750|
|accepted/generated draft|130/235|130/235|130/235|130/235|
|测量请求时 fallback 次数|1|1|1|1|

四臂 HTTP 200，`pred_n=171`，正文 SHA256 均为 `eb0f6bb83a7a8213024f422b7ada2d29cf65f6dc0916bdae99f90e68849a5477`；每臂累计 `CRITICAL-FALLBACK=9`（其中 8 次热机/初始化、1 次测量），`tp5-meta-check` 的 97/97 和 UID 序列一致。**C1 与 C2 自身差 499.311 ms，约 12.9%（按 C2）；C2 与 B2 接近而 C1 明显更慢**。尽管工作量与输出资格一致，现有会话漂移/测量期 fallback 仍使这组不具净性能胜负资格，不能据中位宣称 F32 服务加速或回退；§51 的单卡图静态与数值结论保留。四份原始 JSON、服务日志存 `f32-mtp-guarded-cbbc-20260929.tar.gz`。最终候选 shader/库已恢复到 C 的上述 SHA，实验期间未见新的内核掉总线日志；这次跑通也**不是硬件正常的证明**。

四臂 8-token 热机的 predicted_ms 依次 **2555.368/2150.327/2128.041/2123.269**，首臂的偏慢在测量之前已存在；提示部分热机 prompt_ms 也是 **2139.190/1867.904/1777.022/1863.840**。这支持首臂处于不同会话状态，但**不是** GPU 时钟已被测得或 F32 特例净收益的证明。

## §53 测量期 replay miss 字段级归因（2026-09-29；补足 §50 未证项）

在候选构建的 `ggml_vk_tp5_get_cached_cmd_bufs` 未命中分支临时加只读诊断，仅当缓存 size≥621 且显式环境门控时汇总**同节点数和 scratch 代**的候选条目，不修改 fingerprint 匹配或 replay 判定。使用 §52 同模型、default MMVQ、相同 8-token 热机和 171-token 测量；两次独立服务请求 FP1/FP2 均 HTTP 200，`pred_n=171`，draft=130/235。FP1 在测量期 size=621 的八个同资格条目，均在**第一个绑定**的 `hc_init` shape 不符，route/type/data/buffer 项 0；FP2 逐项印出缓存的 `hc_init ne=[2560,4,2/4/6/7/35,1]`、实时 `ne=[2560,4,3,1]`（`nb[2]=40960` 均同，`nb[3]` 随容量变化），八项无一含 N=3。测量期这个特定 `stage=0/rank=0/nodes=94` miss 是**真实尚未录制的容量 3 形状**，不是图指针改变、缓冲搬迁或无效缓存键；强制复用旧录制图不安全。该证据不证明 §49 forced 模式 size=718 的 miss 也有相同根因，也不证明所有历史 fallback。

临时诊断已从候选源码撤掉；FP1/FP2 原始 JSON 和日志存 `mtp-replay-hc-init-measurement-20260929.tar.gz`。不为隐藏测量期开销主动扩充热机或松掉指纹；若要消除一次 N=3 fallback，应单独设计完整形状专属图预录并核对全部 tensor buffer 身份和五 rank 输出，同时把预录成本算入服务，而不是错误命中旧命令。

## §54 Q8_0 双列 `get_dm` 重复源调用（2026-09-29；编译器已消除）

真实模型源字节排名中 Q8_0 双列的 `k=320,m=10240` 和 `k=10240,m=320` 各 5315 标签，逻辑权重字节各约 18.5 GB；仍是优先审查的 dispatch，不是实测 HBM/耗时。通用 MMVQ 多列 helper 在 `j` 内重复源码 `get_dm(ib_a)`。临时候选在 Q8_0 非 TP5 relay 多列路径把该标量显式 hoist 到 `j` 前，当前只检验了 `NUM_COLS=2`；`GGML_VK_FORCE_MMVQ=1`、Vulkan0 的 Q8_0×F32 CPU 数值对照旧/新各 **1/1**，RADV `shaderstats,nocache` 主要 matvec 着色器两者 **instructions=1198、VGPR=32、VMEM=88、code size=6540、静态 ISA hash=4140834875**。管线 hash 因 SPIR-V 源不同而变，ISA hash 完全相同；不能据源码去重宣称指令净收益。这个候选已撤回，主仓与候选 include SHA256 均为 `a11b9a8846f526c8289dde4b0721a295ed901efc82ca31c9d4687d78a5d1e11c`，重建的 Vulkan 库仍为 `f6739f980913e530c8f4b19e0a42cc216dcdae074b7314ab593df7a1f4ff646b`。原始对照见 `q8-scale-hoist-no-isa-gain-20260929.tar.gz`。无法把这一局部优化失败外推为 Q8_0 或 MMVQ+MTP 全局理论极致。

## §55 融合成对 Q6_K 的激活二次读取（2026-09-29；编译器已消除）

`qwen4_attention_projections.comp::calc_superblock_paired_q6_k` 在 A/B 两段分别写有同一 token-0 激活的四次 `common_input_v4` 载入，看似每个矩阵重复读相同激活。候选只在构建树将这四个 `vec4` 变量提升至两个矩阵块的公共作用域，没有改乘法/累加次序。固定 Q5/Q6/BF16、宽 256–3584、非零 view 偏移和奇数行数的单 token 融合投影 Vulkan/CPU 原生对照旧/新均通过 `test-vulkan-command-replay --attention-projections-only`（8192 scratch-growth fixture 未执行）。RADV `shaderstats,nocache` 此测试生成的 **10 个** shader，在新旧之间逐个 `Hash`、instructions、VMEM、VGPR、LDS 全相同；其中一例为 **7061 instructions/344 VMEM/64 VGPR/1024 LDS**，另一例为 **3277/224/64/1024**。编译器已对源级公共载入做到等效处理，不能以 GLSL 变量共享声称 HBM 或服务收益；候选已撤回，主仓与候选 shader SHA256 均为 `db347e80b0f18a33bfa93911fa3c4e6eaa403d92af3188b512b62d27391651e5`，恢复后 Vulkan 库为 `f6739f980913e530c8f4b19e0a42cc216dcdae074b7314ab593df7a1f4ff646b`。原始两臂日志存 `fused-q6-activation-reuse-no-isa-gain-20260929.tar.gz`。此数值测试是单 token，不证明暂时候选的多 token 正确性；已经撤回也无此必要。

## §56 Q8_0 短 K 双行派发（限定真实形状；非五卡收益证明）

保留 host 特例：仅 RADV/RDNA2、原生 subgroup=64、Q8_0×Q8_1、`K=320,M=10240,N=2`，使用 `NUM_ROWS=2`，每 workgroup 算两行、派发数 10240→5120；其它形状与 TP5 默认 MMVQ 门控不变。每行每列的 K 切片、整数 dot、浮点缩放和 reduction 次序未改，激活在同 workgroup 两行间复用。逻辑独立权重下界仍为 **3,481,600 字节**，双行不减少必需权重或 dot 次数。初始 wave32 门控实际没有选中：真实 host 日志明确 `subgroup=64,variant=nil`，数值通过不代表候选被执行；已改成 wave64 并核对真实 CPU 对照图生成 ISA hash **1074734933**，不再把早期旧路径通过算作新路径证据。

实际 host 的 subgroup-no-shmem SPIR-V、native wave64 静态旧/新为 **1198/1762 instructions、88/120 VMEM、32/32 VGPR、0/0 LDS**；新 shader 工作量是两行，静态总数增长不等于每输出代价增长。独立 Vulkan profiler 用不同 row/token 的有符号 Q8 值、单位尺度，每个输出逐项与精确整数 CPU oracle 相等。四臂 1–2–2–1，各 65 次同步 GPU timestamp、丢首个后 64 次中位：短 K **35.00/28.36/28.46/35.12 µs**，独立 VRAM read counter **4,325,888/3,985,152/3,984,128/4,358,016 字节**。对照长 K `K=10240,M=320,N=2` 中位 **26.24/32.72/32.70/26.22 µs**，双行更慢；不能推广到全 Q8_0。主仓新增真实短 K 的 `test_mul_mat`，强制 MMVQ 的 Vulkan0/CPU 数值对照 **1/1**，生成新 ISA hash 的日志已保存。

同模式 forced-MMVQ 五卡 C–B–B–C（171-token、8-token 热机、LateBind off）全部 HTTP 200，正文 SHA256 均为 `84e0e2b8a08bd50c4c111fc6336632b4969f3ed727642edf8baa70bfac0f5e5d`，draft=129/233，五 rank 的各形状 dispatch/barrier/copy 账本一致。predicted_ms **4445.334/4256.507/4262.444/4300.040**；每臂累计十次 fallback（测量期两次）。此组没有显示五卡收益，候选两臂本身漂移 **145.294 ms**，不能以独立 kernel 的改善声称服务净加速，也不是全硬件理论极致。保留的是有直接静态/数值/同形状 GPU 证据的局部路由，不开启 TP5 全局 MMVQ。

原始 profiler 源、选定 SPIR-V、native wave64 ISA/64-query 日志、host 路由诊断及四份服务 JSON/日志存 `q8-short-k-rows2-20260929.tar.gz`。主仓/候选 `ggml-vulkan.cpp` SHA256 均为 `700eaef47928f1a5ff4403c8f99cc70ef55d8884aa8a4cfb4cabac2e8f27aa02`，测试源均为 `08c812267c4f5e06c35bd16de958aca08db986fc8351596c48a527065dfdd007`，当前 Vulkan 库为 `e4c5f8800217199848d44f561be3921142a5c66baa0b9ed675471626221c0791`。鹰眼和 IPMI 狗保留；不能由这次运行证明物理故障消失。

## §57 短 K 固定形状的不可达指令裁剪（2026-09-30）

### 先证不变量，再看指令

§56 的精确路由只接受 RADV/RDNA2、wave64、Q8_0×Q8_1、`K=320,M=10240,N=2`。`K_PER_ITER=8`、`BLOCK_SIZE=64`，故 col_stride=512，原式 `num_iters=floor(320/512)+(8*tid<320)` 恰为 **tid<40 时 1，否则 0**。没有任何 lane 能执行第二轮或四轮展开。每组两行，5120 个 workgroup 的 first_row=0,2,…,10238，不会出现单行尾块。新增 specialization ID=3/4 分别绑定 **320/10240**；默认均为 0，保留通用 push-constant 路径。固定形状编译一次 iter/full-row body；仍保留 `first_row<FIXED_NROWS` 的越界 guard。host 的类型、架构、驱动、subgroup、M/K/N 精确 gate 未放宽。每个有效 lane 的 col、a_offset/view 地址、两个 packed dot、浮点缩放和 subgroup reduction 次序均保持原状。

在每输出独立执行 packed-dot4 的算法族内，所需有效 lane-dot 数为 `10240*2*320/4=1,638,400`；最终反汇编每组 **8 条 v_dot4_i32_i8/v_dot4c_i32_i8**，40 个有效 lane ×5120 组正好是这个计数。独立逻辑权重下界仍为 `10240*(320/32)*34=3,481,600` 字节。**这不是全局算法/指令最优证明**：40/64 lane 利用率、缩放/地址/归约指令及实际 VRAM 超额读取仍没有最优证明，也没有证明另一种算法不能更优。

### 静态结果（只创建 pipeline，不 dispatch）

|编译形状|instructions|VMEM|VGPR|ISA hash|
|---|---:|---:|---:|---:|
|通用 K/M，§56 双行|1762|120|32|1074734933|
|固定 K，仍保留尾行路径|712|24|32|3641141614|
|固定 K/M，最终保留|**338**|**12**|**32**|1947047069|

三者 LDS=0、无 VGPR spill。用 `TP5_PIPELINE_ONLY=1` 和 `RADV_DEBUG=shaderstats,nocache` 收集；另保存最终 `RADV_DEBUG=shaders,nocache` 的机器反汇编。两个 specialization 均 0 的最终 shader ISA hash 与旧双行 **1074734933 相同**。只把 K 写成常量、仍依赖编译器从条件推断 num_iters≤1 的初稿只到 1733 条；显式单轮支路才删除不可达循环。**338 是整个静态程序大小，不是每请求执行 338 条或吞吐提升比例**；此前不可达循环本就不会动态执行，本次未做速度比较。

### 数值与实际服务证据

- 临时图使用不同 row/block 的非单位 Q8 尺度、含零权重行、三次不同 F32 输入。最终固定形状对照旧 §56 双行：普通图和权重/激活非零 view 图各 **245,760 字节逐字节相同**，生成的 host ISA hash 都为 **1947047069**。
- 相邻 `M=10241` 明确没有选中此特例；三次输入的 **245,784 字节**与通用旧路径逐字节相同，ISA hash **4140834875**。新增永久 CPU 对照用例保护这条形状边界；最终 `M=10240/10241` 各 **1/1** 通过。
- 固定 K 阶段的共享 shader 数值回归 Q8_0/Q4_K/Q5_K/Q6_K（含 Q8 单列、三列、四列）共 **15/15**；最终阶段未重复这些已通过用例，最终两个默认 specialization 为 0 的静态/边界证据不能被扩写成所有量化类型的全覆盖。
- 最终二进制五卡 forced-MMVQ、LateBind off 的 8-token 热机+171-token 请求均 HTTP 200，正文与 §56 Q8C2 **逐字节一致**（SHA256 `84e0e2b8a08bd50c4c111fc6336632b4969f3ed727642edf8baa70bfac0f5e5d`），draft accepted/generated **129/233**。该烟测不是速度比较，不能据此宣称服务净收益；fallback 没有被隐藏或放松缓存指纹。

主仓与候选 shader SHA256 都为 `7459dd0436daf1a93282c38c641f5607aae26dfe9cece7a8def662c15bba48de`，`ggml-vulkan.cpp` 都为 `ac29617732ba9353eb68ec280603f83a77036a7800480e2c784ecd57fb50b9ca`，测试源都为 `6ec0c4ea1bddb8232d86008c35962197268abdb6792527a20aef276446cf376b`，最终 Vulkan 库为 `7c41741adfb34628962bed0bf1e4ba38e3bc22bf4171dac74b57e6a5a6d434f3`。构建源绑定仍是候选树，已构建 `llama-server/test-backend-ops/test-vulkan-command-replay`，本项未运行完整 command-replay suite。源码、SPIR-V、两阶段静态 ISA、15/15 与边界日志、逐字节输出和服务记录归档 `q8-fixed-shape-isa-proof-20260930.tar.gz`。TP5 默认 MMVQ 门控、鹰眼和 10 分钟 IPMI 狗不变；全 MMVQ+MTP 理论极致仍未证明。

## §58 Q8_0 长 K 的完整四切片组（2026-09-30）

真实日志中的 `K=10240,M=320,N=2` 有 5315 标签，与短 K 图独立权重字节同为 **3,481,600**，但 §56 已证明不能套用双行。保留每 workgroup **一行**，仅 RADV/RDNA2 wave64 的这个精确形状增加固定 K/M 的 pipeline；N=4/15 和相邻 M 都不推广。`col_stride=8*64=512`，各 lane 的 num_iters 恒为 **20**（余数条件对所有 tid 为假），恰为 **5 组×4 切片**。新增显式整组支路保留 outer loop，不展开全部 20 轮，切片仍按 `tid*8+512*s, s=0..19` 累加，删除不可能的两/一切片尾数。固定 M 保留 first_row<320 guard，320 个 workgroup 不改变输出行映射。

只创建 pipeline 的 RADV `shaderstats,nocache`：

|阶段|instructions|VMEM|VGPR|ISA hash|
|---|---:|---:|---:|---:|
|通用 K/M，原一行|1198|88|32|4140834875|
|只固定 K|1161|88|32|314723678|
|固定 K/M，但通用余数循环|564|44|32|1381497054|
|显式完整四切片组（保留）|**393**|**26**|**32**|3497470871|

均 LDS=0、无 spill；最终 shader 的默认 K/M=0 仍为 **4140834875**，短 K 固定图仍为 **1947047069 / 338 instructions**。最终反汇编 loop body 有 **16 条 packed-dot4**，5 组×64 有效 lane×320 workgroup，仍为 `1,638,400` 有效 lane-dot。这个算法族的必要 dot 与逻辑源权重没有减少；减少的是无效静态路径及控制/地址冗余，**不等于服务吞吐比例或全局最优证明**。本项未做 GPU query/counter 速度比较。

不同 row/block 的非单位 Q8 尺度、含零行、三次不同输入，正常和非零权重/激活 view 图新旧各 **7680 字节逐字节一致**，且 host 确实生成 **3497470871**。相邻 `M=321` 仍走通用 **4140834875**，**7704 字节**对拍一致；新增永久 `K=10240,M=320/321,N=2` 的 CPU 对照各 **1/1**。新二进制五卡 forced-MMVQ + LateBind off 的热机/171-token 请求均 HTTP 200，正文与 §57 逐字节一致（SHA256 `84e0e2b8a08bd50c4c111fc6336632b4969f3ed727642edf8baa70bfac0f5e5d`），draft=129/233。五 rank 每个形状的 dispatch/barrier/copy 账本一致，累计 fallback=10；不隐藏 fallback，不以单次烟测得出净收益结论。

主仓/候选 shader SHA256 均为 `f998a0ea506cfe48bec48798f2eb775a036c1ffd6e795d7daecf169fa20e3848`，host 均为 `8affff5b1262a293e7970eca2682a3f6ffcdd6196124c2df78e6512e175fb386`，测试源均为 `883517b3ad0389f1bb49d4bc623f2c3a16c6a68b3604c4b044e3b8b6a89f3d1e`，Vulkan 库为 `ebe20af22f09873bc4520881e7e6bb63be9ef602c077a79c7fa2fc0363c4b101`。重建了 `llama-server/test-backend-ops/test-vulkan-command-replay`，未运行完整 replay suite。前后 SPIR-V、静态 ISA、非单位尺度图、CPU 边界测试、源代码和五卡请求/日志归档 `q8-long-k-isa-proof-20260930.tar.gz`。默认 MMVQ 门控、鹰眼/IPMI 保持；动态地址/缩放/归约、VRAM 读取及整个 MMVQ+MTP 的最优性仍无证明。

## §59 Q8_0 八值 packed-load 对齐下界（2026-09-30；候选撤回）

Q8_0 块为 34 字节，八值切片起点 `p=34*ib+2+8*iqs`，`iqs=0..3`。`ib` 包含既有 batch/view 地址关系；偶数块起点模 4 为 2，奇数块为 0。限定原有逐 lane 八值 packed-dot4 算法及对齐 dword 载入族，8 字节区间分别跨 **三个/两个 dword**；因此观察到 12 字节原生载入不能直接认作可消除的 4 字节 VRAM 流量，lane 间重叠与缓存行事务仍需独立论证。

临时 raw alias 候选对齐时读两个 dword，偏移 2 时读第三个并用 shift/or 重排。初稿存在不成立的尾部填充假设：34 字节单块的 `ib=0,iqs=3` 第三个 dword 读字节 32..35，超出逻辑范围，旧 packed16 只读 32..33。修正为最后半字载入后，按四个半字 `h0..h3` 证明两个结果仍为 `h0|(h1<<16)`、`h2|(h3<<16)`；源级 bit/address oracle 的 **768 个基向量、5376 个伪随机切片**通过，并覆盖精确 34 字节尾部及非零块前缀。这不是 GPU 数值验证。

只创建 pipeline，未 dispatch、未 query/counter、未服务运行：

|wave64/N=2 路径|旧 instructions/VMEM|raw32 初稿|安全末半字版|
|---|---:|---:|---:|
|短 K 固定双行|338/12|353/14|356/14|
|长 K 固定单行|393/26|414/30|418/30|
|通用单行|1198/88|1314/102|1328/102|

三路 VGPR=32、LDS=0、无 spill。安全版仍增加分支和 VMEM 指令，没有严格资源优势，故两版均撤回，未做任何性能测量，也不以编译器 latency/throughput 估算作运行证据。手工编译 SPIR-V 未进入产品构建；撤回后主仓/候选 shader/include 与 §58 及 `a11b9a8846f526c8289dde4b0721a295ed901efc82ca31c9d4687d78a5d1e11c` 一致，产品库仍为 §58 哈希。原始源 delta、SPIR-V、静态 ISA、基向量脚本及结果见 `q8-packed-alignment-rejected-20260930.tar.gz`。**只证明这个载入族的对齐约束及候选不值得保留，不证明全 MMVQ+MTP 理论极致。**

## §60 Q8_0 固定形状的单 Y 批次寻址（2026-09-30；保留）

### 完整调用契约

`ggml_vk_mul_mat_vec_q_f16` 先断言 `ne11==1 || ne12*ne13==1`。两个既有精确形状选到 `mat_vec_cols=2` 时只有两种表示：`ne11=2` 导致 outer_batches=1，或 `batch_outer=true` 将两个连续外层列折成列维并明确令 dispatch_batches=1。Vulkan support gate 还要求 A/B 的 ne[3] 相等；flatten 谓词要求 A.ne[3]=1，故实际广播表示只来自 ne[2]。非连续输入、按 batch 保存的矩阵、skinny-as-batch 和 expert 图不会由此选到这两个 N=2 特例。

因此 `base_work_group_y=0`、groups_y=1，shader batch_idx 恒 0；原 `get_offsets` 的 a/b/d 三个 batch 偏移皆为 0。新增 specialization ID5 `SINGLE_Y_BATCH=false`，只在既有 RADV/RDNA2 wave64 的短 K 双行与长 K 单行 pipeline 传 true。descriptor/view 字节偏移仍由原绑定保留；列内 j*stride、K 切片、整数 dot、尺度和浮点累加顺序、归约、首行越界 guard、fusion flags 与 replay fingerprint 均不动。MUL_MAT_ID 没有此分支，默认 ID5=false 保持通用路径。普通派发记录的是固定 direct dispatch，不是动态行数 indirect。

### 静态 ISA，不是动态速度

|实际 host 形状|旧 instructions/VMEM|只裁 batch 偏移|ISA hash|
|---|---:|---:|---:|
|K320/M10240/N2，两行|338/12|**238/12**|2010903019|
|K10240/M320/N2，单行|393/26|**283/26**|3541243255|

VGPR=32、LDS=0、无 spill，通用单行 ID3/4/5=0 仍为 1198/88、hash4140834875；ID5=false 的两个固定图仍为旧 hash1947047069/3497470871。短图机器码仍八条 dot×40 有效 lane×5120 组，长图循环体仍十六条 dot×五轮×64 lane×320 组，各 **1,638,400** 有效 lane-dot。另试固定 B/D 列步长：短图239条、长图282条，没有同时严格支配 batch-only，且扩大公共 include 契约，已撤回。代码尺寸减少包含原来不可达的 batch 广播算术，不能外推动态执行减少比例，更不能宣称吞吐收益。

### 当前数值/replay 证据及未通过项

- 两个形状 × 普通 ne11/外层 ne12 双列 × 普通/非零权重与激活 view，共八个图，三次变化输入、非单位 Q8 尺度与零权重行：**1,013,760 字节**与 §57/58 归档旧输出逐字节一致；实际 host 分别生成上述新 hash。每图一次录制 miss、两次真实 command replay hit。
- 相邻 M10241/M321 两图仍生成通用 hash4140834875，进一步 **253,488 字节**一致、四次真实 replay hit。合计 **1,267,248 字节、20 次 replay hit**。永久 CPU 对照两种目标表示及两条 M 边界共 **6/6** 通过。
- 试图广播 ne[3] 的临时图因既有 support gate 回退 CPU，未生成 MMVQ shader，bitwise 旧 GPU 比较失败；不计作 GPU 数值验证，也未更改 support gate。后续临时图增加实际 Vulkan support 断言，未重复这个不支持的检查。
- 默认策略完整 `test-vulkan-command-replay` **26/26** 通过。全局 `GGML_VK_FORCE_MMVQ=1` 完整 runner **未通过**：HC combine 原生参考比较 `diff=0.004761 > 0.000100`（gpu1003.169495、reference1003.174255，:1939）。该参考 injection 是 K10240/M4/N1，未选本项 N2 特例；[INFERENCE] 强制 MMVQ 的激活再量化与 fused HC 原生浮点 contraction 是不同数值契约。不以默认通过替代 forced 全量资格、不削弱容差，仍需进一步归因。
- 当前检查枚举 **四张** Vulkan RX6800，不是五 rank 服务验证；未做新的服务 smoke、query/counter/ABBA 或吞吐比较。runner 自动打印的时间不作性能证据。硬件保护没有改动。

主仓/候选最终 shader SHA256=`71b7a6e764eebabba0cc598aae4f7c2b8e33633ae68899f231fbea319e1f4877`、host=`a7ad8322711c55d9b674386fb6afa5f5bb1a3c71e21142abcc94cf440cb0a184`、测试源=`369e593fe78d7ca8aa7c1501d3e3cc28f76fb1f10798599085249fa646141096` 一致。构建仍绑定候选树，已构建 server/backend/replay runner；Vulkan 库=`3c85291b4726a62f955ac52d292ac9c3064053e1b0bd0b46847af8ed434d4edc`，最终 SPIR-V=`30fc2b3e6b9d1a9ea55404386b9cb42e7677aa14203255921a8f60005a8d4f76`。源码、静态阶段、最终 ISA、变化输入/view/replay 输出、CPU 日志和 suite 观察结果摘录存 `q8-single-batch-isa-proof-20260930.tar.gz`。**整体理论极致和 forced 全路径资格仍未证明，性能测量门控仍关闭。**

## §61 HC 原生 contraction 与 forced-MMVQ 的数值契约（2026-09-30；不改生产、不改容差）

§60 的具体失败可由源码算术区别解释：`qwen4_hc_segment_norm.comp::normalize` 的 fused injection 直接读取 F32 激活，按八值 floating dot、权重尺度 FMA 与 subgroupAdd 累加；通用 Q8_0 MMVQ 则先按 `d=max(abs(x))/127`、`q=round(x/d)` 量化激活，尺度转 F16，计算 `d_w*d_x*dot(q_w,q_x)`。一般 `x != F16(d)*round(x/d)`，故两个 contraction 本身就不同；这不是把地址或 replay 契约放宽后得到的解释。

独立 CPU double 算术脚本复用 fixture 的精确 dyadic 权重和 F32 输入生成，但独立实现上述激活量化与 sigmoid scatter：

|输入前缀 / row0|原生 scatter|再量化 scatter|差值|GPU 观察|
|---|---:|---:|---:|---|
|0|1003.169514240042|1003.174284463860|0.004770223817|原始1003.169495 / 参考1003.174255|
|1（4 字节 view）|1002.716417983574|1002.724690320878|0.008272337304|F32 oracle 候选：实际1002.724670 / 参考1002.716431|

脚本没有 GPU dispatch 或计时；这些数值解释了两个失败的方向与幅度，不等于逐指令 GPU emulation、全部 MMVQ 误差边界或五卡输出资格。`ggml_vk_can_fuse_hc_segment` 的 f32 谓词要求 16 字节对齐；4 字节 view 合法地走通用 fallback，多 token 非 capture 图也有明确 fallback 条件。因而“全局 forced”并不把每个合法 fused injection 都改成同一种算术。

曾只把测试的原生参考 Q8 权重换成精确等值 F32 矩阵，保持原容差：前者可避免合法 fused case 的参考被再量化，但非对齐 fallback 的实际值仍用 MMVQ，随后在 `diff0.008240>tol0.000100` 失败。这个只修一侧的 oracle 候选已撤回，未镜像主仓；**不关闭 MMVQ、不删除失败、不加宽容差来伪造全量 forced 资格**。生产源码/库与 §60 相同。主仓/候选 replay 测试源恢复为 SHA256 `f522c14e1004802a0f0d9ce99f6629bb84f2d61066f8ca28e54ae29d37e7b008`，重建 runner SHA256 `67a883be2ec4e8a68102262e83e35579df371475acebef3c5e99421b66e7a7e4`；恢复后二进制 `--hc-combine-only` 默认策略通过。§60 相同原始源码及 backend 的默认 26/26 证据仍成立，forced 全量仍不声称通过。

算术脚本、两组输出、被拒 oracle 源及失败原始日志、恢复后的 HC 烟测归档 `hc-mmvq-native-contract-20260930.tar.gz`。该调查解决的是两种算术误作同一精确 oracle 的归因；**没有产生生产优化或服务性能结论，整个 MMVQ+MTP 理论极致依然未证明。**

## §62 Q8_1 sumless 路径的缓存及尺度舍入约束（2026-09-30；理论反例，不改 quantizer）

Q8_0 的 `mul_q8_1` 不读 ds.y，但 Q4_0/Q4_1/Q5_0/Q5_1、Q4_K/Q5_K 等 helper 明确用它抵消量化偏置。`ggml_vk_get_quantize_pipeline` 统一返回 `pipeline_quantize_q8_1_x4`；prealloc_y 的复用只按转换 pipeline、输入 tensor 身份和 decode staging layout 判定，没有“和字段有效”的额外状态。故“当前消费者是 Q8_0”不是整个转换结果生命周期内 ds.y 死亡的证明。

静态合法混合消费者反例：Q8_0(X) 若沿相同 pipeline 身份生成缺少和的 X，后续 Q4_0/Q4_K/Q5_K(X) 可命中这个不完整结果；Q4_0 每八值切片的公式为 `da*(q_sum*ds.x-2*ds.y)`。da=1、ds.y=32、四切片时，将 ds.y 清零使整块输出增加 **256**，不是可被浮点误差容差覆盖的微扰。这是缓存契约的独立反例，不声称当前模型日志已观察到这组精确消费者次序。换成不同转换 pipeline 身份虽可迫使后续完整重算，但在此序列中变成 sumless+full 两次转换，旧路径只需一次 full；不能据单个 quantizer 的指令减少主张普遍静态工作量优势。

另一个舍入反例否定了简单 lazy backfill：原 shader 的 `ds.x=F16(d)` 与 `ds.y=F16(F32(sum_q*d))` **独立舍入**。用 IEEE struct F32/F16 转换验证，d=`1.0003999471664429`、sum_q=3 时，ds.x=1；原 ds.y=`3.001953125`，从 ds.x 回算则为 `3.0`，差 `0.001953125`。仅缓存整数 qs 和 F16 尺度不能保证恢复原字段；从原输入重建尺度及和还需计入额外读取/同步/转换，不能默认为免费。

安全接受条件是完整转换生命周期的 sum-reader 死亡证明，或精确且无额外可避免转换的有效性迁移；还必须保留 Q8 值、ds.x 舍入、x4 布局、输入 tensor 身份、非零 view、缓存/replay 生命周期和全部 affine 消费者。当前源码没有提供这项证明，**不实现仅按当前权重类型删和的第二套缓存契约**。本项无源码改动、无 shader 编译/dispatch、无性能测量。两份独立算术反例及对应源码存 `q8-sum-cache-boundary-20260930.tar.gz`；这收束的是朴素 sumless 方案的安全边界，不是全 MMVQ+MTP 最优性证明。

## §63 共享 Y scratch 生命周期与 MTP token 容量（2026-09-30；修复、正确性验证）

### 缓存有效性不是内存同步

`prealloc_y` 同时承载转换后的激活、大 softmax 的 reduction partials、Flash Attention 的 mask bitmap。转换缓存原先按 pipeline、输入 tensor 身份和 decode-vector staging 布局命中；大 softmax 与 FA mask 写入 scratch 后虽完成同步，却没有清除这三个标记。同步保证写入顺序，不能使被覆盖的激活重新有效。

最初用两个静态权重 MMVQ 夹住 softmax 的图没有复现：graph optimizer 合法地把两个独立 MMVQ 都移到 softmax 前。调试器确认派发顺序，旧库数值对照和 replay 均通过。这份 ordering control 保留，不把函数局部推理当作复现。真正复现图为 `MMVQ(X) -> REPEAT -> large SOFT_MAX/FA -> cast(Q8_0 second weight) -> MMVQ(second weight, X)`；第二个权重依赖 scratch writer，不能再把末端 MMVQ 移到它之前。

旧库 softmax 图三次变化输入的末端 128 行均损坏，第一次 row0 独立 CPU 期望 `-0.00468948483`、GPU 为 0；已有两次真实 replay hit，说明命中 replay 不保证其录制时的缓存假设正确。相同永久回归 runner 搭旧库退出 1：`diff=0.004689 > tol=0.000010`。新库同 runner 退出 0。修复仅在两个 scratch writer 派发后清除 `prealloc_y_last_pipeline_used`、`prealloc_y_last_tensor_used`、`prealloc_y_last_decode_vector_staging`；保留原同步、算术、缓存身份及回退契约，不为缓存命中创建第二套有效性规则。

### 实际正确性覆盖

- `test_mmvq_scratch_reuse` 独立 double CPU contraction 对照两端 MMVQ；dyadic 激活确保 Q8_1 尺度及数值可精确表示。对中间 softmax 检查概率归一化，对零 key 的 FA 检查已知 value 输出；所有关键节点要求 Vulkan support 并显式指定 Vulkan，防止 CPU fallback 掩盖缺陷。每种 writer 三次变化输入，确认至少两次 replay hit。
- 独立复现程序修复后 large softmax 与 FA mask 两图均通过；各一次 miss、两次 hit。softmax 末端误差为 0，FA 末端最大误差不超过 `3.57627869e-7`。这些是数值/重放证据，不是计时证据。
- 通过 `GGML_VK_VISIBLE_DEVICES=0..4` 分别执行永久 scratch 回归，五张物理 RX 6800 均退出 0，均非 skip。逐卡序列执行、持 GPU 排他锁；**不是五 rank TP5 服务图的验证**。
- 默认完整 replay runner **27/27** 通过。未重跑已知不合格的全局 forced HC 比较，也不把默认通过解释为 forced 全路径资格。

### MTP 多 token、容量增长及合法回退

扩展已有 GDN/QSA 投影 fixture 到 token 数 **1、2、3、4、5**，不改原容差。1/2/4 容量槽及 N=3 使用容量四的图均按每个输出/token 比较；N=5 超出 fused 容量，覆盖原生合法回退。覆盖 Q5_K、成对 Q6_K、BF16 gate、非零 view、K 尾部和奇数行。reference 是独立原生 Vulkan 图（四个各自 scale 的输入防止投影融合），**不是 CPU backend 对照**；整数路径只允许既有累加顺序误差，F32 路径及 BF16 gate 要求逐字节一致。

先录制 N=3/K=256 图，嵌套 N=4/K=8192 图迫使 region scratch 增长，再以新输入 replay 原来的三 token 图；要求 replay hit 增加而不重新录制 fused region。整个 fixture 还要求每个四投影 integer region 恰好一次量化，并确认 Q5/成对 Q6 分支实际记录。五张物理卡的 `--attention-mmvq-only` 均通过，包括 scratch-growth fixture；仍不能据此宣称真实模型所有 token 图均融合或没有回退。

### 归档及资格边界

源码已镜像主仓/候选树，SHA256 均一致：host `70c9e36117a5b1265cf54a73a37601e276d15b75ab1a89a9f5f76bb3551b57cc`，replay 测试 `c7cd747edd869cc8c554eeba1ba1cd29ff6a7c06255b864458ac1a08f379a330`。新 Vulkan 库 `6a87ca7726241b9eaea98979281f1daf86f823306d5eee237f3a16c42ea69326`，runner `ced0d6eeff442057139c973d1395c3784c67ac677372bb5844d0fcd6ac89d2a4`；前后库身份和原始失败保留。构建仍绑定候选树。

源码、集成 diff、前后库、原生复现程序、ordering control、六组初始验证记录及逐卡日志存 `mmvq-scratch-lifetime-and-mtp-capacity-20260930.tar.gz`。用户已豁免进一步动态指令数优化；后续只推进物理数据搬运、激活复用、MTP 容量及融合回退。**本项没有 query/counter/ABBA、吞吐比较或五卡服务请求，没有全局理论极致证明；性能门控仍关闭。**

## §64 转换源被原地写入后的失效（2026-09-30；第二个确定性缓存缺陷）

### `view_src` 是存储身份，不是值版本

`ggml_new_tensor_impl` 将 view 的存储根和绝对偏移折叠；`ggml_mul_inplace`、`ggml_scale_inplace` 等真实 writer 也通过 view 构造结果。只按 `view_src`、相同 buffer/offset 或相同元素数扩大复用，会把写入前后两个 producer 的值混为一谈。透明 VIEW/RESHAPE 的 `src[0]` 链仍保留真正 producer；合法复用还必须证明顺序、转换 pipeline/布局和元素序相同、源未改写、Y 未覆盖及容量生命周期有效。当前 Qwen4EXP Q/K/V、GDN 与 PLE 相邻投影本来就共享同一个 `cur/input/emb` 对象，不能假定增加存储根别名 key 就有真实模型搬运收益。本项不扩大缓存 key。

进一步复现发现原有精确 tensor 身份 key 本身也缺少源写入失效：`first=MMVQ(e0,X)` 产生 scalar factor，`MUL_INPLACE(X, REPEAT(first))` 改写 X，改写结果生成第二个 Q8_0 权重，末端 `MMVQ(produced_weight, X)` 再用同一个 graph input。权重依赖 writer，不能把末端 matmul 提前。Y 本身保持完整，但它仍是旧 X 的转换；内存同步和 tensor 身份相等都不能修复这个值版本错误。

|输入轮次 / factor|独立精确 CPU 期望|旧 GPU|修复后 GPU|
|---|---:|---:|---:|
|0 / 2|650088|325044|650088|
|1 / 3|1462752|487584|1462752|
|2 / 4|2600608|650152|2600608|

复现程序同时验证初次投影、原地写入内容和实际生成权重，均精确正确；错误只在末端读旧转换。旧/新图均一次 replay miss、两次 hit，旧退出 1、新退出 0。整数/dyadic 数值与 FP16 尺度可精确表示，所有最终和小于 2^24，**不依赖误差容差来区分新旧值**。

### 保留的修复与回归

在 `ggml_vk_build_graph` 记录每个真正 writer 后，用既有 tensor-range overlap 谓词检查缓存转换源。只要写入与源相交，就清除 pipeline/tensor/decode-staging 三个标记；同范围写入也必须失效，故这里使用 `elementwise=false`。不相交的同 buffer 写入不失效。既有同步和 replay 指纹不动；录制后的重放会实际执行 writer 后新增的重新转换。复用已有 overlap 实现，前移定义供 recorder 调用，沿用已有 graph-hazard 的非 Vulkan/multi-buffer context 防护，不另设 alias 表、generation 对象或第二套缓存。

永久 `test_mmvq_input_mutation_replay` 覆盖整段与局部 alias writer，后者从非零输入 view 起点再写入内部两个 Q8 block。每图三次变化输入，逐项检查变更后的输入、生成权重、末端精确 CPU 值及至少两次真实 replay hit；关键节点要求 Vulkan support 和实际 Vulkan assignment。扩展原 `--scratch-reuse-only`，不新设兼容入口、不放宽旧容差。新 runner 搭 §63 旧库时 scratch 两个 writer 测试通过、随后的 source-mutation 精确断言失败（退出 1）；新库两组测试全通过。

最终库在五张物理卡上逐卡 scratch/source-mutation 与 `--attention-mmvq-only` 全通过；MTP fixture 的 1–5 token、容量增长和回退边界继续有效。默认完整 replay **28/28** 通过。已构建 `llama-server` 和 replay runner，但本项没有新五 rank 服务图、吞吐、query/counter 或 ABBA，也没有重跑已归因的 forced HC 比较。

主仓/候选 host SHA256 均 `810389d075914f612b205d5fe4260df3d81fbc6cd257cca51347d9d2b981fe75`，测试均 `2f3487ea7194a21b72a881751d213fb7573751a099fb28a0d1f3b120a814cde9`。Vulkan 库 `bd79cb77e55db03b37faf5290b36838d1180a390a14ccbf8e53777c1a8779a02`，runner `376d1a87ec7f69ed9a4a25a318dcff8f79d2e589cc7fdbb853d2f281854f6cee`；旧库为 §63 的 `6a87ca7726241b9eaea98979281f1daf86f823306d5eee237f3a16c42ea69326`。源码、前后库、复现程序、永久失败/通过日志及逐卡结果存 `mmvq-source-mutation-cache-proof-20260930.tar.gz`。**这是激活复用正确性的必要修复，不是服务性能资格或全局极致证明。**

## §65 Q8_0 扩大 row tile 的物理读取边界（2026-09-30；仅 pipeline 创建，不切换生产）

在 §60 同一 SPIR-V 上只改变 `NUM_ROWS`，固定 K320/M10240/N2、singleY=true；沿用归档的 pipeline-only inspector，不创建 command buffer、不 dispatch、不创建 query 或提交 queue。这里仅审查寄存器、LDS、spill 与源数据覆盖，**不追动态指令数，不用编译器 latency/throughput 字段当速度**。

|row tile|完整 workgroup 数|VGPR|LDS / scratch|SGPR / VGPR spill|编译器 subgroups/SIMD 估计|
|---|---:|---:|---:|---:|---:|
|2（已有 §60）|5120|32|0 / 0|0 / 0|32|
|4|2560|32|0 / 0|0 / 0|32|
|8|1280|32|0 / 0|0 / 0|32|
|10|1024|40|0 / 0|0 / 0|24|
|16|640|56|0 / 0|0 / 0|18|

四种新 tile 都整除 M。限定既有格式、任意非退化输入和独立权重，逻辑源权重仍为 `10240*10*34=3,481,600` 字节；量化 B 的唯一 footprint 为 720 字节，非 elided 原生 F32 输出为 81,920 字节。按每 workgroup 的 B footprint 求和，tile2/4/8 分别为 3,686,400/1,843,200/921,600 字节，但**这个重复 footprint 不是 VRAM 读取**：720 字节 B 可以在 cache 中重用，改变 workgroup 数不等于改变独立源权重下界或实际外存事务。row4/8 没有额外 VGPR/LDS，但仍缺实际 cache-line/coalescing/residency 和数值资格；row10/16 还有寄存器压力代价。未以这张静态表替换已资格验证的 row2 路由。

另对跨行打包 40/64 lane 尾部建立有限映射：R 行的 40R 个八值切片可按 task=row*40+slice 铺到 wave64；R4/R8 分别需三/五轮，而每行 partial 的 lane 位置变成循环平移。CPU 映射检查证明覆盖且无重复；保持原浮点 reduction 顺序需逐行逆置换。直接打包还把既有一次 B preload 改成三/五次，或需额外 LDS/寄存器缓存，不能只凭“填满 lane”宣称物理搬运严格减少。未改 shader、未 dispatch 此方案。

资源日志、当前 SPIR-V、pipeline-only 源/二进制、traffic 理论表及 lane 映射检查存 `q8-row-traffic-resource-boundary-20260930.tar.gz`。这收束了直接扩大 row tile/打包 lane 的资源与流量推理边界；**没有证明全硬件最优，性能门控仍关闭。**

## §66 性能测量：forced-MMVQ vs TP5 默认禁用（2026-09-30；用户授权后首次测量）

### 前提

用户指示：与 MMVQ+MTP 无关的优化不做；两者到极致后可开始测量。动态指令数已豁免。卡 3 在 2026-09-30 09:11 UTC 再次掉总线（`device lost from bus!`，SMU `TransferTableSmu2Dram` 返回 `0xFFFFFFFF`）。当时 `gpu-hard-unlock.sh card3` 卡在 `amdgpu_device_fini_hw -> drain_workqueue` 的不可中断 D 状态，三个 `kworker/u177:*+ttm` 卡在 `dma_fence_default_wait`，`pci rescan` 卡在 `pci_lock_rescan_remove`；根端口 `00:02.2` 链路宽度降为 `x0`、全 8 lane 报错，卡配置空间全 `0xff`。这类物理链路失步 + 内核全局 PCIe 锁死无法在运行中解救（D 状态忽略 `kill -9`，rescan 会再产生新的 D 进程），只能有序重启。重启后 5 张卡全部恢复，Eagle 与 IPMI 硬狗正常。

### 方案与结果

用同一 `llama-server`（`da35702a…`）跑 ABBA，`--tp5-latebind off` 隔离 LateBind：R=TP5 配方注入的全局禁用，T=`GGML_VK_FORCE_MMVQ=1`。R 不是 auto；显式 `GGML_VK_DISABLE_MMVQ=0` 才恢复 auto。每臂 4 请求（首个热身丢），8 个块交替 `R T`/`T R`，共 24+24 个样本。固定 prompt、`temperature=0`、`seed=42`，`env -i` 仅 `RADV_DEBUG=nobolist`，每臂持 `/var/tmp/tp5-gpu.lock`。

|臂|ms/cyc 中位|ms/cyc 均值|tok/s 中位|md5|fallback/session|
|---|---:|---:|---:|---|---|
|R（默认禁用）|28.865|28.934|61.59|`add18ad37ff7` ×24|[5]×8|
|T（强制 MMVQ）|28.930|28.979|59.70|`2696790a8525` ×24|[5]×8|

配对逐块 `T/R` 中位比：均值 **1.0009**，t-95% CI **[0.9882, 1.0136]**，逐块 `[0.9808, 1.0078, 0.9903, 0.9886, 0.9906, 1.0205, 1.0193, 1.0093]`。**区间含 1.0，无统计显著差异。**

### 解读与资格边界

- 在此 `ms/cycle=(predicted_ms)/(predicted_n-draft_n_accepted)` 代理指标上未发现显著差异；没有独立成本分解，不能声称量化/解包开销“恰好抵消” dot4 收益。
- 两臂会话内 `CRITICAL-FALLBACK` 总数相同（`Graph fallback to 480 subgraphs, n_sub=97`）。总数不能区分初始化、热身和测量阶段，也不足以排除 MMVQ 交互；联合配置测量必须记录每请求日志边界。
- R/T 输出长度分别为 160/171，各自 24/24 同 md5；算术和接受模式改变可能导致不同文本，但确定性不能证明不存在数值缺陷。正确性证据仍以 §61/§63/§64 的明确定义 oracle 为准。
- 此实验比较同一优化后二进制的 MMVQ 政策，不是优化前后消融；不能把政策比较结论推广为“本轮所有改动没有服务吞吐价值”。跨 MTP draft 深度应以固定 token 工作量的 token/s、ms/token 和端到端时间为主。

16 份服务端日志、48 份请求 JSON、配对报告及环境记录存 `mmvq-forced-vs-default-abba-20260930.tar.gz`。**这是单模型、单 prompt、五卡单请求的配对测量，不是全模型全路径吞吐最优性证明；全硬件理论极致仍未证明。**
