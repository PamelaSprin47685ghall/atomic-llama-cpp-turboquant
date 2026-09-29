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
- **发现并修复了 `DATA_A_QUANT_LEGACY`（Q8_0 等）缺乏 `mmvq_dot_multi` 的严重缺失**：
  此前 Q8_0 在 `NUM_COLS > 1` 时退化为在内层循环重复解包、重复读取权重。已补充了针对 Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/MXFP4 的完整 `mmvq_dot_multi`，实现了权重单次解包、寄存器广播点积多列。

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
