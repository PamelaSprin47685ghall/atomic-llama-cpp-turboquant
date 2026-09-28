# LateBind 深井：三个根因修复（其中两个互相掩盖），aggressive 刀锋从「不可用」到与参照持平（2026-09-28，run-20260928T0330）

> ## 执行摘要（读这一段就够；下文按时间追加，章节号失序，仅作证据链追溯）
>
> **本会话终态：修好的 aggressive 刀锋 = 61–66 tok/s（连贯），与参照 61–63 持平。**
> 会话开始时它是 **5 tok/s 的 `<think>` 循环**。约 **12× 提升**。
>
> **两个独立根因，只修任一个都不行 —— 它们互相掩盖：**
>
> | | 缺陷 | 症状 |
> |---|---|---|
> | **A** | **孔径族**：Host Push 改了载荷绑定与 CPU 读取基址，漏了 6 处（含**武装重置** → Q 完成计数器攀升过 `n_workgroups`，Q 每会话只发布一次；以及**本该抓住它的那个守卫**） | 握手错乱 |
> | **B** | **激活量化器**：`qs_words[lane] = word` —— lane 与字号不是同一编号，32 个量化值只剩 8 个、每个重复四次 | Q 值本身错 |
>
> **第三个根因（在测 L2 代价时被直接抓出）**：`cc` 偏移在 C++ 与着色器**两侧都硬写了 `12u * 16u`**，
> 而真实容量 `VK_TP5_DIRECT_COLUMN_TILE = 4` —— **两侧互相一致，但一致地指向错误位置**，
> 比 CPU 的边界校验**越界 512 字节**，写进 bank 里别人的地盘。后果：ρ 读到垃圾 → `u` 错 →
> **模型在第 28 个 token 提前 EOS**（四次镜像测量 100% 复现；`rho=0` 四次均为完整的 171）。
> 修正为各自从单一来源符号导出。
>
> **另有五处独立缺陷已修**：`mmvq_dot_multi` 守卫（整棵树此前无法从零编译）、两处 push constant layout 不匹配、`plan.late_stats_buf[i]` 越界、Q8DOT 的 LB2 统计量折叠与发布器管网、`rho=` 账本字段（此前 L2 臂无法被断言）。
>
> **三向对齐的验证**：六臂（3 模式 × 镜像）速度全部落在 61.2–66.3、账本分发/屏障/拷贝计数全同（2879/2496/191）、输出逐字节相同。
>
> **一条结构性发现**：L1（重叠）与 L2（ρ）当前互斥 —— 重叠把 Q 发布搬进 P2，而 LB2 的中继必须在 P1→P2 之间读它。解耦设计已找到（统计量只依赖 `residual`/`z_p`，可拆为独立派发留在 P1），但**先量后建**。
>
> **仍开放**：aggressive 与参照持平但未超越；本会话数值是单次直跑/镜像，不是预注册的正式 ABBA。
>
> **本会话的四次同形方法论错误**（均已自行抓回，记录在案）：
> ① 拿有瑕疵的证据撤回了正确结论；② 把自己的 fail-closed 当成被测对象崩溃；
> ③ 拿未镜像的单样本发布了"重叠是 1.6× 惩罚"——**被镜像臂抓住**；
> ④ 以为"两侧硬编码互相对得上"就等于"单一来源"——**被 L2 的提前 EOS 抓住**。


源码 `/var/tmp/tp5-candidate-k1a`（未提交），构建 `/var/tmp/tp5-final-build`（RelWithDebInfo/-O2）。
工作负载：自然文本 `请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。`，
prompt_n=31、temp=0、seed=42、stream=false、n_predict=171、MTP h=1（`--spec-draft-n-max 1`）。未替换计数题。

## 0. 本轮的起点：一句被重复六次的判断

用户判词「**latebind 的深井代码从未被实现完整**」。逐份通读 `TP5-LATEBIND-LINEAR.md`、
`交接.md`(v3)、`下班交接.md`、`缺口.md`、以及 `artifacts/.../g9-latebind-d1-final` 与
`latebind-squeeze` 两份 decision.md 后，该判断成立，且能指到行号：

| 深井层 | 文档主张 | 改动前实况 |
|---|---|---|
| **L1** Q 支路 ∥ norm 支路汇合于 LO | 规格："关键点不是把两个 generation 合并，而是**删除人为的 Q→norm 串行关系**" | exact 路把整个 `pre`（zsnap 拷贝 + late_q + 发布）串行在 norm 之前；aggressive 路只做了 `norm→Q8DOT` 那半段 |
| **L2** RMS 充分统计量 + 一次发布 | `‖R_b+s_b y‖² = a_b + 2s_b·Σ_p d_{p,b} + s_b²·c` | 实现过（LB2/rho），+14.2% 回退，成分归因未闭合 |
| **L3** producer 投影复合 | `C_{p,b} = W_bΓ_b P_p` | 从未实现 |
| **仪表** | 账本必须可区分臂 | `overlap=norm-q` 在 `:6088` 是**硬编码常量**，连 reference 臂都照印；`tp5_get_late_overlap_mode`/`tp5_get_late_transport_mode` 定义齐全但**零调用点**；`tp5_validate_norm_before_q_schedule` 声明于 `.hpp` 但**从不被调用**；`late_q_gtt_*` 只有声明与销毁、无分配 |

**结论：历次"LateBind 零收益"的测量，缺少可分辨的仪器与正确的对照臂，因此不构成对机制本身的判决。**

## 1. 已落地的代码改动

1. **账本去硬编码** `ggml-vulkan-collective.cpp:6079-6095`：`overlap=` 取 `tp5_get_late_overlap_mode()`
   真实值（`legacy|reordered|norm-q`），新增 `transport=bar-pull|gtt-push`。
2. **L1 重叠调度**：zsnap 快照前移到切分标记之前（`:4225-4243`，legacy 下命令序列逐位不变），
   发射期按 `GGML_TP5_LATEBIND_OVERLAP` 三态切分：`emit(pre,…,q_split)` →
   `emit(p2,0,norm_end)`（norm 先发、阻塞等 Y）→ [`reordered` 保留 COMPUTE→COMPUTE 屏障] →
   `emit(pre,q_split)`（Q 收缩+发布落进该窗口）。
3. **校验器接线**：`tp5_validate_norm_before_q_schedule`（定义期硬错误）。它要求 `WRITE_TREFS`(norm)
   早于 `READ_ZSNAP/READ_TREFS/DISPATCH_Q8DOT`(Q) —— **拿它校验旧顺序必然失败**。
4. **Host Push（缺口一）**：`q8dot_infos[2]` 由 `bcast_buf` 改为按 `GGML_TP5_LATE_TRANSPORT=pull|push`
   选择 `host_import_buf`；CPU 侧读源与控制字基址同移；**fanout 目的端保持 `bcast`** ——
   因为 LO 消费者的 Inbox 是 `tp5_hc_resume.comp` 的 binding 0（= `bcast_buf`），
   跟着搬会让 LO 读到过期字节。F32 路本就是同样的不对称（src host RAM / dst bcast）。

## 2. L1 实测（同二进制、交错镜像 R El Er Eo | Eo Er El R、账本断言）

| arm | `overlap=` | `barriers` | r1 / r2 tok/s |
|---|---|---|---|
| R1 reference | `legacy`, `late=0` | 3485 | 64.58 / 64.96 |
| El1 exact+ZSNAP | `legacy` | 2268 | 41.10 / 36.76 |
| Er1 exact+ZSNAP（对照） | `reordered` | **2362** | 39.67 / 38.98 |
| Eo1 exact+ZSNAP（候选） | `norm-q` | 2268 | 39.27 / 40.25 |

- **门控在改命令流**：`Er − El = 2362 − 2268 = 94 = late`，每 site 精确多一道屏障。
- **`Eo − Er = +0.43 tok/s (+1.1%)`，小于臂内离散（Er 0.69、El 4.34）→ 未检出收益。**
- `validation failed = 0`，校验器在重叠调度上通过。

### 2.1 判决（2026-09-28 更正：两处方法论违例已撤回原判词）

**违例一：在回退实现上做针尖对麦芒。** 本轮把 L1 判在 `exact-f32`（39–42 tok/s）上，而 exact 是**回退实现**，
其每 site 一道 HOST 域屏障（94 × ~72 µs）才是它慢 37 % 的根因。LateBind 的**尖端是 `aggressive-q8`**
（`dispatches` 少 36 %、无 HOST 屏障）。在钝刀上判机制生死，就是 D4 记下的错误：
"the fast LateBind modes were never runnable -> the verdict was taken on the slow fallback only".

**违例二：镜像未完成即封口。** `Eo2`（镜像，同臂同配置）= **41.70** 对 `Eo1` = **39.76**，
**差 4.9 %**，纯属 session 漂移。故本 harness 在 exact 路上的分辨率约 **±5 %**，
而 `Er2/El2/R2` 未跑完 —— 本就无权下"收益为零"的判词。

**更正后的判词**：前向半场、在回退实现上，**未检出收益，且待检效应低于该实验分辨率**。

> **L1 仍未结案。** 它的针尖对麦芒检定必须在 `aggressive-q8` 上做，而那需要先让那条路可跑
> —— 即本文件 §1.4 的 Host Push。机制与调度已确证落地（校验器通过、`barriers` 精确 +94），
> **一个已实现且被强制执行的机制，不等于一个已被否定的机制。**

### 2.2 保留意见（不得含糊）

`overlap=` 报的是**门值**，不是**实际调度**。分支被走到与否，由 `validation failed`（校验器只存在于
`if (ovl_sched)` 内）与 `barriers` 计数交叉证明。

## 3. 未完成

- 镜像臂只跑到 `Eo2`（41.70）；`Er2/El2/R2` 未跑 —— 为把 GPU 让给尖端而终止。
- **Host Push 已编译通过**（`BUILD_EXIT=0`，2026-09-28 03:32，`bin/llama-server`）。
  冒烟（`pull` vs `push`，单因素）运行中。预期 `cpu_data_us` 由 328 781 µs 崩向 ~945 µs 量级
  （同一段求和、同一批元素，仅换孔径；AVX2+F16C 快路已确证在跑，2.1 µs/16B 事务是孔径成本而非向量化不足）。
- **下一步的针尖对麦芒**：`reference`(64.8) vs `aggressive-q8 + Host Push`，同二进制、账本断言
  `mode=aggressive-q8` / `late>0` / `q8_fast>0`；再在 aggressive 内部比 `overlap` vs `reordered`。

## 4. Host Push 冒烟（2026-09-28 03:32 构建；单因素 pull vs push，同二进制，n_predict=32）

| arm | 账本 | 结果 |
|---|---|---|
| `pull` | `transport=bar-pull` | **崩溃**：`llama_decode failed, ret=-3` → SIGSEGV |
| `push` | `transport=gtt-push`, `late=94 q8_fast=94` | **跑完** 11.63 tok/s（带 profiler），bytes=47 |

profile（push）：`sidecar_data_us=190/203`、`sidecar_wait_us=0`、`q_spin_n=470`、
`cpu_data_us=934/960`、`cpu_data_excludes_wait=1`、`validation failures=0`、`vk errors=0`。

**`cpu_data_us` 328 781 → ~940 µs（350×）**，与预测的 ~945 µs（F32 路同批元素的开销）吻合；
`sidecar_wait_us=0` —— CPU 已不再等 Q。

判决：

- **Host Push 是 aggressive 路可运行的前提，不是调优项**（同一二进制：pull 崩、push 通）。
- 但 aggressive 仍比 reference（64.8）慢 **5.6×**，而其 `dispatches/barriers`（2879/2496）
  **优于** reference（4089/3519），且已无 HOST 屏障、孔径已修。
  => **孔径从来不是它的瓶颈，只是它的伪装。** 剩余 5.6× 在 GPU 侧内核与协议。
- 连带修正：LB2 当年的 +14.2% 回退有了明确嫌疑犯 —— `交接.md` §3 自陈的
  "CPU 从 device 映射区逐字读回统计量（~0.3–0.7 µs/word）"，正是本次用 350× 修掉的那处。

## 5. 验证协议（2026-09-28 裁定；优先级高于本文件此前任何结论）

**不得逐层对弱基线判词。** 已发生的两次错都是这个形状：
在 `exact-f32` 上判 L1（被 94 道 HOST 屏障主导）、在 `pull` 上判 aggressive（被孔径主导）——
**当系统被别的瓶颈主导时，被检层的效应必然被遮蔽，"未检出"与"检出"同样无信息量。**

规定顺序：

1. **打包实现 L1 + L2 + L3**，使"理论尖端"真正存在；
2. **尖端对参照**（tip-to-tip）：尖端 vs `reference`，同二进制、账本断言；
3. **再消融**：`tip − L1`、`tip − L2`、`tip − L3` —— 单层贡献只在尖端存在后才有意义。

**为何 L1 必须等 L2**：L1 是同 rank 内的派发重叠，不消除 relay 轮次；而每 stage 现在是**两轮**
（Y 轮、Q 轮）。L2 把第二轮折进第一轮（只需 4+4+1 个标量随 Y 同轮发布）。只有当 stage 收敛为
**一轮**、Q 真正落到关键路径上时，L1 的重叠才有对象可叠。

**L2 的边界**（按裁定的 CPU 预算）：充分统计量走 relay（**CPU 求和，可以**），
`ρ` 与 `u` 在 **GPU** 合成（CPU 不越界）。Host Push 是 L2 的前置 —— 二者治的是同一处
（CPU 读 device 映射区）。
- 下一步：把 `c = ‖y‖²` 与 `d_{p,b} = <R_b, z_p>` 折入现有归约循环（读已付），

## 6. L2 真相：不是未实现，是被孔径与一处基址错配关掉的（2026-09-28 晚）

### 6.1 L2 数学本体早已完整

总门 `GGML_TP5_LATEBIND_RHO`（`:203`）。全链齐备：`lb2_cc[16]` `:5522`、
`lb2_cc_mode = relay && late_count && tp5_latebind_rho_enabled()` `:5523`、
`tp5_star_sum_broadcast_f32_cc_avx2(..., lb2_cc_rows, lb2_cc)` `:5529`（`c` 在热路径顺带累出）、
ρ 合成 `:5651-5667`、单次发布 `publish_generation(bank, 0)`（`q_publish_us=0`、
`sidecar_wait_us=0`）；shader/pipeline/描述符集 `late_q_stats` `:467`、`late_publish_stats` `:486`、
`late_lo_u` `:490`、`late_stats_buf` `:909`。

ρ 的恒等式**逐字落地**：`ss = A + 2*s_b*D + s_b*s_b*cc`，其中 `A=a_b`、`D=Σ_p d_{p,b}`（rk>0 累加）、
`cc` 来自 Y 归约。**规格里的 `‖R_b+s_b y‖² = a_b + 2s_b Σ_p d_{p,b} + s_b² c` 与此一模一样。**

### 6.2 真正病因（本轮已修，已编译）

`late_publish_stats_ds` 第 [4] 号绑定把 GPU 产出的原始统计量写进 **`host_import_buf`**，
而其余三端全部在 **bank**：

| 端 | 位置 | 目标 |
|---|---|---|
| CPU 读原始统计量 | `:5638` | `bcast_host[bank] + stats_offset` |
| CPU 写回 ρ | `:5654` | `bcast_host[bank] + stats_offset` |
| 着色器读 ρ | `tp5_hc_resume.comp:351` | `inbox[rho_word_offset + …]`，`inbox` = binding 0 = `bcast_buf` |
| GPU 写统计量 | `:4084` | ❌ 原为 `host_import_buf` |

**写者与读者不在同一内存域** —— 这比规格警告的"各写一套偏移"更深一层：不是偏移不一致，
**是基址属于不同内存域**。这解释了 LB2 "数值与性能都说不通"却始终未归因的形状：
它回读的从来不是它算出的那些数。

修复：`:4089` 改为 `r.bcast_buf[b]`（偏移不变，单源契约不动）。`BUILD_EXIT=0`。

### 6.3 L2 剩余

1. ρ 的合成在 CPU（`:5659`），按裁定"CPU 求和可以、再多不值"应移入 GPU norm shader；
   relay 只发布原始 `a_b / d_{p,b} / c`。
2. 统计量区仍走 `bcast_host` 带宽孔径；若尖端测试显示它是瓶颈，再用 Host Push 同型手法搬迁
   （需同时改 CPU 两侧 + `rho_word_offset` + 着色器读取绑定）。

### 6.4 本会话引入、待撤回的死重量

`y2_out` 参数化加在了**冷路径** `tp5_star_sum_avx2`（`:4827`）与 `tp5_avx2_accumulate_star`（`:4969`）——
二者仅在 `n_elems >= 262144` 时被调用，本模型 `n_elems = 2560` 永不进入；且与既有的
`tp5_star_sum_broadcast_f32_cc_avx2` 功能重复。已编译、默认 `nullptr` 中性。
**撤回配方**：删 `y2_out` 形参、`__m256 y2acc` 声明、循环内 `if (y2_out) y2acc = …`、
块尾水平归约、标量尾的 `y2_local += s*s` 与末尾 `*y2_out += y2_local`；

### 6.5 L2 的理论尖端（本轮定位完成；实现待下一会话）

关键事实：**`late_publish_stats` 已经把 GPU 算出的原始充分统计量写进 bank**（绑定 [3] 源 =
`late_stats_buf`，[4] 目标 = 本会话修复后的 `bcast_buf + stats_offset`）。每 token 12 words 的布局为：

| words | 含义 |
|---|---|
| `t*12 + 2b + 0` | `A_b = ‖R_b‖²` |
| `t*12 + 2b + 1` | `D_b = Σ_p ⟨R_b, z_p⟩` |
| `t*12 + 8 + b` | `s_b` |

**也就是说，着色器合成 ρ 所需的一切输入都已在 bank 里了**（由 GPU 写好，CPU 不参与）。
而当前代码让 CPU 把这 12 words 读出来、算 `ss = A + 2sD + s²cc`、再把 ρ 写回**同一地址**。
**这个 CPU 往返 100% 可删** —— 它正是裁定中「CPU 求和可以、再多不值」的那部分。

**理论尖端三步（每一步都无新协议，单源契约不动）：**

1. `stats_words = 16u * late_rows`（`:5581`，为 `cc` 腾出 `+12`），`stats_bytes` 随之，
   余量校验 `:5584` 仍按 `c.star_rank_stride`。
2. relay 的 publish 循环（`:5640-5675`）从「读 A/D/s + 写 ρ」降为**只写 `st[t*16 + 12] = cc`**：
   每 rank 每 token **1 次写**（原为 12 读 + 12 写）。A/D/s 保持发布器原样，不再被 CPU 触及。
3. `tp5_hc_resume.comp` 在 `:351-354` 处不再读预算好的 ρ，而是读同一位置的 A/D/s/cc 并内联计算
   `ss = A + 2.0*s*D + s*s*cc; rho = ss > 0.0 ? inversesqrt(ss) : 0.0;`

孔径流量由 ~24 次 trans/token/rank 降为 **1 次**（~**24×**）；且 `rho_word_offset = stats_offset/4`
的等式与 `inbox = bcast_buf` 均不变，**着色器读位置一字未动**，只换了它从这三个字里推导什么。


### 6.6 两侧逐字锚点（下一步可直接粘贴）

#### 着色器侧 — `vulkan-shaders/tp5_hc_resume.comp:351-354`

现状（取 ρ，步长 2）：

```glsl
const float rho0 = uintBitsToFloat(inbox[p.rho_word_offset + token * 12u + 0u]);
const float rho1 = uintBitsToFloat(inbox[p.rho_word_offset + token * 12u + 2u]);
const float rho2 = uintBitsToFloat(inbox[p.rho_word_offset + token * 12u + 4u]);
const float rho3 = uintBitsToFloat(inbox[p.rho_word_offset + token * 12u + 6u]);
```

替换为（自 `A/D/s/cc` 内联合成；**只多一行 `cc`，不新增 push constant，不改步长**）：

```glsl
    // L2: rho is derived here from the sufficient statistics the GPU publisher already
    // wrote (A_b = |R_b|^2 at 2b+0, D_b = Sum_p <R_b,z_p> at 2b+1, s_b at 8+b); only cc
    // = |y|^2 is CPU-produced, because only the relay's reduction can form it. This
    // removes the CPU read-back of A/D/s and its rho write-back entirely.
    const uint cc_word = p.rho_word_offset + 12u * TP5_LATE_CAPACITY_ROWS + token;
    const float cc = uintBitsToFloat(inbox[cc_word]);
    const uint so = p.rho_word_offset + token * 12u;
    const float A0 = uintBitsToFloat(inbox[so + 0u]), D0 = uintBitsToFloat(inbox[so + 1u]), s0 = uintBitsToFloat(inbox[so + 8u]);
    const float A1 = uintBitsToFloat(inbox[so + 2u]), D1 = uintBitsToFloat(inbox[so + 3u]), s1 = uintBitsToFloat(inbox[so + 9u]);
    const float A2 = uintBitsToFloat(inbox[so + 4u]), D2 = uintBitsToFloat(inbox[so + 5u]), s2 = uintBitsToFloat(inbox[so + 10u]);
    const float A3 = uintBitsToFloat(inbox[so + 6u]), D3 = uintBitsToFloat(inbox[so + 7u]), s3 = uintBitsToFloat(inbox[so + 11u]);
    const float ss0 = A0 + 2.0 * s0 * D0 + s0 * s0 * cc;
    const float ss1 = A1 + 2.0 * s1 * D1 + s1 * s1 * cc;
    const float ss2 = A2 + 2.0 * s2 * D2 + s2 * s2 * cc;
    const float ss3 = A3 + 2.0 * s3 * D3 + s3 * s3 * cc;
    const float rho0 = ss0 > 0.0 ? inversesqrt(ss0) : 0.0;
    const float rho1 = ss1 > 0.0 ? inversesqrt(ss1) : 0.0;
    const float rho2 = ss2 > 0.0 ? inversesqrt(ss2) : 0.0;
    const float rho3 = ss3 > 0.0 ? inversesqrt(ss3) : 0.0;
```

**与 CPU 现行 `ss = A + 2.0f*s_b*D + s_b*s_b*cc; st = ss > 0 ? 1.0f/sqrtf(ss) : 0` 逐字等价。**
唯一的浮点差异是 `inversesqrt` 与 `1.0f/sqrtf` 的最后一位；若这一位不允许，用 `1.0 / sqrt(ss)`。

#### CPU 侧 — `ggml-vulkan-collective.cpp:5580-5584` 与 `:5640-5675`

1. `:5580` 后补上 `cc` 的基址（与着色器同式，两侧各自导出，**不新增通信字段**）：
   ```cpp
   const size_t cc_offset = stats_offset + 12u * TP5_LATE_CAPACITY_ROWS * sizeof(float);
   ```
   余量校验 `:5584` 改为 `cc_offset + late_rows * sizeof(float) > c.star_rank_stride`。
2. publish 循环（`:5640-5675`）：删掉对 `stats_fold` 的 12 字读取与 ρ 的 12 字写回，
   只保留**每 token 一个字**的发布：`((float *)((char *) c.ranks[rk].bcast_host[bank] + cc_offset))[t] = cc_t;`
   其中 `cc_t` 来自 `lb2_cc[]`（已由 `tp5_star_sum_broadcast_f32_cc_avx2` 在 Y 归约中顺带累出）。
3. **不要**再写 `st[t*12+8+b] = s_b` —— `s_b` 已由 GPU 发布器写入，CPU 覆写是多余孔径操作。

#### 孔径账（预期）

| | 现值 | 尖端 |
|---|---|---|
| 每 rank 每 token | 12 读 + 12 写 | **1 写** |
| 每 stage（12 words × 5 ranks） | ~120 trans | ~5 trans |
| 96 stage 链 | ~17 µs/stage → **1.6 ms** | ~0.07 ms |


### 6.7 状态：已实现并编译（同日晚，`BUILD_EXIT=0`）

§6.6 的两侧锚点已**实际落地**，不是待办：

- CPU `:5599` `const size_t cc_offset = stats_offset + 12u * 16u * sizeof(float);`
  （`stats_words` 同步改为 `late_rows * 12u + 16u`，原余量校验 `:5600` 自动覆盖 cc 尾部）
- CPU 发布块 `:5641-5667`：旧的双层 `stats_fold` 读+ρ 写回**已整块删除**，只剩
  `ccp[t] = lb2_cc[t]`（每 rank 每 token 一次写）
- 着色器 `tp5_hc_resume.comp:351-376`：ρ 由 `A/D/s`（`late_publish_stats` 已写入）与 `cc`
  内联合成，`1.0 / sqrt(ss)` 保留以与 CPU 路径**逐位一致**（否则后续尖端对参照会把 ULP 当成收益）

静态核验：两侧基址 `12u*16u`（CPU 字节 / 着色器字，`rho_word_offset = stats_offset/4`）逐字吻合；
`grep -c "1.0f / sqrtf(ss)"` = **0**、`grep -c stats_fold` = **0** —— 往返无残留。


## 7. L3（`C_{p,b} = W_b * Gamma_b * P_p`）：拦路石是未做出的设计判断，不是未写的代码

### 7.1 规格（逐字引用，非推演）

| 出处 | 内容 |
|---|---|
| `交接.md:112` | LB3 只改 **Q 的生产方式**（复合投影 `C_{p,b}`），**可保留现有 RMS/ρ/双门铃协议**；门槛不是"stats 零成本"而是**新增非重叠成本 < 减少的关键路径成本**；另有权重存储/量化格式/数值资格的设计判断 |
| `交接.md:128` | **P2 LB3 独立原型**：**不依赖 LB2 结论**；先一个**固定线性出口**验证 `C_{p,b}` 的权重/量化/访存代价与 **"Q 更早完成"** 的实际收益 |
| `TP5.md:163` | LB3 producer 投影复合（`C_{p,b}`，**不能推过非线性**，**需独立数值资格**） |

目标因此明确：**让 Q 早一步完成** —— 把 `W_b * Gamma_b` 与位置相关的 `P_p` 复合成一个矩阵，
使 Q 在输入一到就能一次 GEMM 产出，不再等第二个依赖步骤。**LB3 与 LB2 无关**，两者可独立立项。

### 7.2 已锁定的代码事实

- Q 生产者 = `tp5_hc_late_q`（`:1801-1802`，7 bindings / 20 push constants，`make_late(7, 20, …)`），
  binding [0] = `late.down_weight`（即 `W_b`）。
- Q sidecar 载荷 = **4 流 × 320 宽 = 1280 float/token**（"Q 1280→u 320"，`TP5.md:163`）→ `u = 0.25 * Σ_b rho_b * Q_b`。
- 全部 LateBind 阶段来自**单一源文件** `vulkan-shaders/tp5_hc_latebind.comp`，多个 entry/config。
- 双门铃协议（`late_q_control_offset` / `status_word_offset`）与 RMS/ρ 路径**不需要动**。

### 7.3 尚未做出的设计判断（这才是拦路石）

`C_{p,b}` 以**位置**为下标。Q sidecar 是 4 流×320 宽；若把 `C_{p,b}` 物化，每 rank
每 token 行至少需要 **4 个复合矩阵**。若 `P_p` 是 320×320 的一般线性映射，则
`C_{p,b} = W_b(320×K) * Gamma_b(diag) * P_p(320×320)` 仍为 320×K —— **相当于再存一份权重**。
这正是文档要求的"权重存储/量化格式"判断，也是一个补丁解决不了的问题：

1. `C_{p,b}` **物化**（预处理一次、每位置一份）还是**在线复合**？前者用存储/访存换关键路径，
   后者基本不换 —— 两者结论会被"新增非重叠成本 < 减少的关键路径成本"这条门卡住。
2. **量化格式**：若 `C_{p,b}` 以低精存储，数值资格就不能只对 F32 做；反之则与 `down_weight`
   现有格式重复存储。
3. **`Gamma_b` 是否可复合**：`TP5.md:163` 的"不能推过非线性"是一条**前置约束** —— 必须在源码里
   确认 `Gamma_b` 是纯对角门（可复合）还是坐在非线性之后（不可复合）。这一条尚未验证，
   且是 L3 能否成立的前提。

### 7.4 文档指走的第一步

`交接.md:128` 指定先做**固定线性出口**的原型：在一个固定线性出口上单测 `C_{p,b}` 的
权重/量化/访存代价与"Q 更早完成"的真实收益，**不绑定 LB2 结论、不绑定全模型**。
在 `Gamma_b` 可复合性（§7.3-3）得到正面答案之前开工，等于在未验证的代数上建存储方案。

### 7.5 `Gamma_b` 可复合性——已从源码验定（本条解开 §7.3-3）

`tp5_hc_latebind.comp:226-230`（`TP5_LATE_UP_Q8DOT` 块）逐字：

```glsl
const vec4 gate = vec4(1.0) / (vec4(1.0) + exp(-acc));   // Gamma_b = sigmoid(acc)，4 流
const vec4 n = vec4(norm_in[token O + row],              // 4 个流的 P_p 输出
                    norm_in[token O + p.width + row],
                    norm_in[token O + 2u * p.width + row],
                    norm_in[token O + 3u * p.width + row]);
mixed_out[token * p.width + row] = dot(n, gate) * 0.25;  // out = 0.25 * Sum_b Gamma_b * n_b
```

而 `acc` 是 **10-block 行和的线性累加器**（"After three steps every lane of the group
holds the full 10-block row sum"），即 `acc_b` 是输入的线性泛函。

**结论：`Gamma_b` 是每流一个标量的对角缩放**（`b` 在点上被求和，不是元），因此

    W_b * Gamma_b * P_p = Gamma_b * (W_b * P_p)

`Gamma_b` 可交换出去 —— **`W_b * P_p` 可按位置预复合**。
`TP5.md:163` 的"不能推过非线性"得到准确含义：**产生 `Gamma_b` 的 sigmoid 不能被推进矩阵**
（它是标量，必须单独算，每 token 每行仅 4 个），但 `Gamma_b` **自身**可复合。

### 7.6 L3 余下的唯一判断：`W_b * P_p` 的存储与量化

复合对象因此从 `C_{p,b}` 缩为 **`W_b * P_p`（按位置 p 一份）**，与 `Gamma_b` 解耦。
余下的设计判断（也是 `交接.md:112` 说的那一条）：

- `W_b * P_p` **按位置物化**（存 4 份/位置/rank）还是**在线复合**？
- 以何精度存（低精存储则数值资格不能只做 F32；高精则与 `down_weight` 现有格式重复）。
- 门槛仍是 `交接.md:112` 的原句：**新增非重叠成本 < 减少的关键路径成本**。

### 7.7 Q 生产者本体：绑定逐字确认（L3 复合点已定）

`tp5_hc_latebind.comp:410-437`，`#ifdef TP5_LATE_Q` 块（`local_size_x = 32`，
拓扑注释明写 "Match the native HC W_down single-subgroup topology"）：

```glsl
layout(std430, binding = 0) readonly buffer DownWeights { late_q8_0 w[]; };  // W_b，Q8_0 量化
layout(std430, binding = 1) readonly buffer Gamma4     { vec4 gamma4[]; };   // Gamma_b，具名
layout(std430, binding = 2) readonly buffer Residual4  { vec4 residual4[]; }; // R
layout(std430, binding = 3) readonly buffer LocalZ4    { vec4 local_z4[]; };  // z_p -> 位置下标 p
layout(std430, binding = 4) readonly buffer Scatter    { float scatter[]; };
layout(std430, binding = 5) writeonly buffer LateOut   { float q_out[]; };
layout(std430, binding = 6) readonly buffer RowsHeader { uint rows_header[]; }; // 活跃行数
```

**关键确认**：

1. `Gamma4` 以 **`vec4` 逐项**读取 —— 一次拿到 4 个流的门，与 §7.5 验定的
   "每流一个标量的对角缩放"**逐字吻合**。
2. `W_b` = `DownWeights`，格式为 **`late_q8_0`（Q8_0）** —— 这意味着 `交接.md:112` 说的
   "量化格式的设计判断"**不是开放题**：复合产物必须与 `late_q8_0` 共存或另立格式。
3. 位置下标由 **`LocalZ4`（z_p）** 承载 —— 即 `P_p` 的位置依赖就在 Q 生产者内部。
4. `p.streams != 4u` 是硬条件 —— 复合只对**四流形状**成立，其他形状必须保留原路。
5. `p.width & 255u` 必须为 0 —— 复合产物的宽度约束已由现有一致性检查限定。

### 7.8 井底：Q 生产者本体逐字（本节修正 §7.5 的外推）

`tp5_hc_latebind.comp:474-515`（`TP5_LATE_Q` 主循环）：

```glsl
const float s  = scatter4[stream];                     // s_b，每流标量
const vec4  z0 = local_z4[token_z_offset + col/4u];    // z_p，按 token（位置）
vec4 rz0 = z0 * s;                                      // s_b · z_p
if (p.my_rank == 0u) rz0 += residual4[...];             // + R_b
const vec4 b0 = gamma4[global_col/4u] * rz0;            // Gamma ⊙ (R + s*z_p)
rowtmp  = dot(b0, v0) + dot(b1, v1);                    // · W_b
rowtmp *= float(w[ib].d);  acc[stream] += rowtmp;       //   (Q8_0 反量化)
```

即 **`Q_b = W_b * diag(Gamma) * (R_b + s_b * z_p)`**。

**修正 §7.5 的外推**：§7.5 验的是 `TP5_LATE_UP_Q8DOT` 块，那里的 `gate` 是点积产生的
**每流标量**（`dot(n, gate)`）。而 Q 生产者里的 `Gamma` 是 `gamma4[global_col/4u]` ——
**逐列**，即**对角矩阵**。两处形状不同；"可交换出去"只对 UP 块成立，**对 Q 生产者不成立**。
本节结论优先。

### 7.9 L3 的井底：物化按位置线性涨价，门槛几乎必然否决

代数上有一个比"复合"更强的恒等式：

    W_b * diag(Gamma) * (R_b + s_b*z_p) = W_b*diag(Gamma)*R_b + s_b * W_b*diag(Gamma)*z_p

**右端第二项就是现有代码正在计算的东西** —— 即当前关联顺序已是该式的最优形态之一。
"物化 `C_{p,b} = W_b*Gamma*P_p`"的实际含义是：把产生 `z_p` 的投影 `P_p` 折进 `W_b`，
使 Q 不必等 `z_p` 物化（`交接.md:128` 所说的"Q 更早完成"）。

| | 形状 | 规模 |
|---|---|---|
| `W_b` | `rank_dim x width` | 现有一份 Q8_0 权重 |
| `C_{p,b} = W_b*Gamma*P_p` | `rank_dim x width`，**每个位置 p 一份** | **× 位置的权重存储** |

而 `z_p` 是 **per-token（位置）** 的：`local_z4[token*(width/4) + col/4]`，`token < active_rows <= 16`。
**因此物化 = 至多 16 份权重/rank。** 而 `交接.md:112` 的门槛是
**"新增非重叠成本 < 减少的关键路径成本"** —— 16 份权重的存储与访存几乎必然超过节省的关键路径。

**在线复合 `W_b*(Gamma ⊙ z_p)` 正是现有代码**，所以 L3 唯一可行形态是：
**`P_p` 低秩或跨位置共享**，使得复合产物小于一份完整权重。
若 `P_p` 是一般满秩的位置相关矩阵，**L3 在存储上先死，不需实机测试即可否决**。

### 7.10 井底验定：`P_p` 位置无关，且 `Gamma` 卡在两次收缩中间 —— L3 被算术否决

`local_z` 的读取形态全文件一致（`:278` / `:484` / `:595`）：

```glsl
local_z[token * p.width + col]      // token 索引第几个 token（行），不是第几个矩阵
local_z4[token_z_offset + col / 4u] // token_z_offset = token * (p.width / 4u)
```

**`token` 是行索引。** 所以 `z_p = W_z * x_p`，**`P_p = W_z`，对所有位置是同一个矩阵**。
§7.9 预设的"按位置一份权重"不成立 —— 但同时，真正的链条是：

    Q_b = W_b^T * [ Gamma_b ⊙ ( s_b*W_z*x_p + R_b ) ]

**`Gamma_b` 是运行时对角的，坐在 `W_z` 与 `W_b` 中间。** 要复合出 `W_b*Gamma_b*W_z`，
必须在运行时先形成 `W_b * diag(Gamma_b)`：

| 形态 | 每 token 成本 |
|---|---|
| 现有：`Gamma ⊙ v` 后收缩 | `width` 次乘法 |
| 复合 `(W_b*Gamma_b)*W_z` | `rank_dim x width` 次乘法 + 改写整块权重 |
| 复合 `W_b*(Gamma_b*P_p)` | `width x width` + 改写中间量 |

**两种结合都严格更贵。** 原因：待折叠的对角 `Gamma_b` (a) 运行时才产生，(b) 位于两次
收缩之间 —— 这是"对角卡在矩阵积中间"的标准情形，**没有可预计算的复合对象**。

### 7.11 L3 裁决（本文件）

**L3 被算术否决，不需实机测试。** 它声称的复合对象 `C_{p,b} = W_b*Gamma_b*P_p`：

- `P_p = W_z` **位置无关**（§7.10），所以不存在"按位置预复合"的收益；
- `Gamma_b` **运行时对角且位置最内**，任何结合都比现在多花乘法；
- 现有代码 `W_b^T*(Gamma ⊙ (s·z_p + R))` **已经是该式的最优结合**。

这与 `TP5.md:163` 的警告完全一致："**不能推过非线性**"。
更准确地说：**也不能推过运行时对角**。两个约束指向同一件事 —— `Gamma` 两侧的
矩阵乘法不可合并。

### 7.11b 最后一种重构也已堵死：按列共享的 `Gamma` 与 per-token 的 `rho`

`gamma4[global_col / 4u]` **不含 token 索引** —— 即 `Gamma` 在一个 stage 内**跨位置共享**，
逐 stage 变动。据此检查唯一剩下的重组方式（把 4 流合流）：

    u = Sum_b rho_b * W_b^T * [ Gamma_b ⊙ (s_b*z_p + R_b) ]
      = ( Sum_b rho_b * W_b^T * diag(Gamma_b) ) * z_p  +  Sum_b rho_b * W_b^T * (Gamma_b ⊙ R_b)

第一项看似可合流成**单个** `rank_dim x width` 矩阵。**但它不可预计算**：
`rho_b` 是 **per-token** 的（relay 的 4 个标量），所以该矩阵每 token 都不同；
而每 token 现造它需要完整扫过 4 份 Q8_0 权重，**比现有的 4 次收缩更贵**。

**因此 L3 不存在可行的复合形态**：`Gamma_b` 运行时产生（跨位置共享但逐 stage 变）、
`rho_b` 随 token 变，两者都夹在矩阵积中间。这不是"还没找到做法"，
而是**不存在可预计算的复合对象**。

**残留**：若未来 `P_p` 变成真正的按位置矩阵（例如为了跨位置共享而重构），
本节结论需重验；当前源码下不成立。`交接.md:128` 要求的"固定线性出口原型"
**不需做**，因为否决发生在原型之前（代数层）。


## 8. 理论尖端的定义与尖端对参照（2026-09-28 晚）

### 8.1 尖端 = L1 + L2（这是证明的边界，不是缩小的目标）

用户裁定："在理论尖端出现之前禁止测试"。尖端本应为 L1+L2+L3；而 §7.10-7.11
**证明 L3 不可达**（`P_p = W_z` 位置无关；`Gamma_b` 运行时对角且位于两次收缩之间，
任意结合都比现有多花乘法）。**因此尖端就是 L1+L2，且这个边界是推导出来的。**
测量禁令的前提由此满足。

### 8.2 先补可断言性：`rho=` 账本字段

补字段前，`[tp5-linear-definition]` 只带 `late=` / `overlap=` / `transport=`；
**没有任何字段能区分 `GGML_TP5_LATEBIND_RHO` 的开与关**。按纪律，无法被账本断言的臂
就是无效臂 —— 所以 L2 的消融臂在补字段前根本无法成立。

已在打印位增 `rho=%d`（`tp5_latebind_rho_enabled() ? 1 : 0`），`BUILD_EXIT=0`。
它是 L2 从"引擎里存在"走到"可被消融"的那一步。

### 8.3 尖端对参照 harness：`/var/tmp/tp5-tiptip.sh`

骨架直接复用 `/var/tmp/tp5-l1.sh`（已验证的 `flock 9` + `pgrep` 守门、`env -i` 合格默认档、
自然文本、`temp=0 seed=42`、镜像交错、中位数、`validation failed` 计数）。八臂：

| 臂 | `--tp5-latebind` | 环境 | 含义 |
|---|---|---|---|
| `R` | `off` | — | 参照 |
| `T` | `aggressive` | `ZSNAP=1 TRANSPORT=gtt-push OVERLAP=overlap RHO=1` | **尖端** |
| `Tl` | `aggressive` | 同上但 `OVERLAP=legacy` | 消融 L1 |
| `Tr` | `aggressive` | 同上但无 `RHO` | 消融 L2 |

交错：`R T Tl Tr | Tr Tl T R`。三条硬化：

1. **三重断言**：`late=[1-9]` ∧ `overlap=norm-q` ∧ `rho=<期望值>` 同时命中才算有效臂。
2. **消融分离**：`T−R` 是尖端收益；`T−Tl`、`T−Tr` 各自隔离 L1/L2，**不得拿 `T−Tl` 冒充 L1 收益**。
3. **`gtt-push` 是尖端臂的固定环境项**（aggressive 走 `pull` 实测崩溃），不是可选项。

**工作量与自然文本**：prompt 31、`temp=0 seed=42 stream=false`、`n_predict=171`。**禁用计数题。**

### 8.4 本会话引入的死重量已清

`y2_out` 参数化加在**冷路径** `tp5_star_sum_avx2` 与 `tp5_avx2_accumulate_star` 上：
二者仅在 `n_elems >= 262144` 被调用，本模型 `n_elems = 2560` 永不进入；且与既有的
`tp5_star_sum_broadcast_f32_cc_avx2` 功能重复。**已全部撤回**（5 处确定性替换，`y2_out`/`y2_local`/`y2acc`
共 **0** 残留引用）；撤回前逐位确认**无任何调用点传入 `y2_out`**，即它确实是死路。
构建验证待测量结束（纪律：禁止编译与测量并行）。

### 8.5 自查发现并修正：L1 消融臂选错了档（否则重蹈 Eo−El 旧错）

首版 harness 把 L1 消融臂写成 `OVERLAP=legacy`。**这是错的。** 源码：

```cpp
const bool ovl_sched = ovl_mode != tp5_late_overlap_mode::LEGACY &&
                       tp5_latebind_zsnap_enabled() &&
                       q_split > 0 && q_split <= pre.code.size();
```

**`REORDERED` 与 `OVERLAP` 都进 `ovl_sched` 分支** —— 两者只差是否保留 COMPUTE→COMPUTE
屏障（见 §6 / L1 落地笔记）。而 `LEGACY` 根本不在该分支里。因此 `T − Tl(legacy)` 把
**"重排本身"与"并发"两种效应混在一起**，正是用户明令禁止的
`Eo − El` 冒充法（"收益只认 Eo − Er…绝不拿 Eo − El 冒充"）。

**修正**：L1 消融臂改为 `OVERLAP=reordered`（同发射顺序、保留屏障）—— 这样
`T − Tr(reordered)` 才隔离出**并发**本身。账本 `overlap_name` 映射已核实：
`REORDERED -> "reordered"`、`OVERLAP -> "norm-q"`、`default -> "legacy"`。

修正后的八臂（镜像 `R T Tr Tl | Tl Tr T R`）：

| 臂 | `OVERLAP` | `RHO` | 隔离的量 |
|---|---|---|---|
| `T` | `overlap` | 1 | 尖端（`norm-q`） |
| `Tr` | `reordered` | 1 | **消融 L1 的并发**（屏障仍在） |
| `Tl` | `overlap` | 0 | 消融 L2 |

带缺陷的首轮（`bg_12`）已终止，日志归档至 `/var/tmp/tp5-tiptip-defect1/`，

### 8.6 第二处自查缺陷（结构性）：aggressive 被加载门 fail-closed

被终止的首轮留下了有效日志 —— 参照臂跑完了，但**尖端臂在加载期就死**：

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  TP5+MTP LateBind requires five distinct RX 6800 ranks, RELAY/F32, and explicit exact Q
```

根因（`common/common.cpp:486-500`）：

```cpp
} else if (params.tp5.latebind == "aggressive") {
    setenv("GGML_TP5_LATEBIND", "hc-down", 1);
    setenv("GGML_TP5_LATEBIND_EXACT_Q", "0", 1);   // aggressive = 近似 Q
}
const bool exact_ok = exact_q && std::strcmp(exact_q, "1") == 0;
if (!qualified || effective_sync != "relay" || effective_wire != "f32" ||
    (!exact_ok && !allow_approx)) throw ...;
```

**`aggressive` 刻意把 `EXACT_Q` 置 0，因而必须显式 `GGML_TP5_LATEBIND_ALLOW_APPROX=1` 才放行。**
harness 未给 → 三个尖端臂全被 fail-closed。**这是"近似 Q 默认 fail-closed"的安全设计在起作用，
不是 bug**；缺的是 harness 的显式 opt-in。`bg_13` 同样中招，已终止。

### 8.7 `rho=` 字段首次实机验证通过与第一个真数据点

参照臂账本（首次带 `rho=`）：

```
mode=reference desc=reference-no-latebind late=0 rho=0 q8_fast=0 scatter_early=0
overlap=legacy transport=bar-pull sidecar_pub=fused dispatches=4047 barriers=3485 copies=192
```

**`rho=0` 如实上报** —— 字段已活，L2 臂从此可被断言。
参照臂 tok/s：**60.84 / 63.86 / 58.54**（warm/r1/r2）。

### 8.8 第三个自查缺陷：`pred_n` 不一致则 ABBA 作废

参照臂吐 `predicted_n=160` —— **不在已知变体集 {171,174,178} 内**（`交接.md` 开放问题 B、
G7 记录的变体）。不同 `pred_n` 的臂比的是**不同工作量**，直接比 tok/s 是错的。
harness 原本只打印不断言，已增**工作量一致性护栏**：每臂记录 `pred_n`，收尾统一汇总，
任何不一致的臂在判决中不可比。

### 8.9 本节的三个缺陷共同说明一件事

三个缺陷都发生在**测量链自身**（臂档位、加载门、工作量一致性），而非被测对象。
若不做这三项自查，两轮缺陷测量会产出**看似有数字、实则无效**的结论 —— 正是
"单次 smoke 不等于性能资格"要防的东西。带缺陷的两轮均归档且**明确不作为证据**：
`/var/tmp/tp5-tiptip-defect1/`、`/var/tmp/tp5-tiptip-defect2/`。

## 9. 【已证实】aggressive 刀锋输出退化且慢 5× —— 长度配对证据（本轮）

### 9.0-now 同长度配对证据（同 prompt、同 seed、交错、同长度上限区）

`bg_14` 的 `Tl1` 臂（aggressive + `rho=0` + `overlap=norm-q` + `gtt-push`，账本
`late=94 rho=0 q8_fast=94`）跑通并给出结果：

| 臂 | tok/s | `pred_n` | bytes | 内容特征 |
|---|---|---|---|---|
| **R1 reference** | **62 / 60** | 160（自然停） | 288 | 连贯中文 |
| **Tl1 aggressive** | **13 / 8** | 171（达上限） | 201 / 183 | `<think>，健康减重原则。，，。，。` / `。。。。。` / `，，，，` |

**纯标点汤，无实词。** 且**参照自然停在 160，aggressive 撞满 171 上限** ——
退化输出永远遇不到停止符。warm→r1→r2 递减（41→13→8）说明退化在累积。

**因此：aggressive 输出确实退化，且慢约 5×（13/8 vs 62/60）。**
§9.0 的"警报已撤回"**本身被撤回** —— 撤回理由（拿 32 长度比 171）成立，
但**结论是对的**。教训：**不能因为一条证据有瑕疵，就撤回一个正确的结论。**

### 9.0-now-b 第二个独立缺陷：`RHO=1` 在定义期失败

`T1`（尖端）与 `Tr1`（消融 L1）**都无账本**，但日志显示模式已激活：

```
[tp5-numerical-mode] mode=aggressive-q8 desc=aggressive-q8-quantized reason=none wire=f32 late=yes direct=p1
[tp5-latebind] hc-down active producer=p1 q=q8dot-approx mode=aggressive-q8 width=2560 streams=4 ...
```

**唯独 `[tp5-linear-definition]` 未打印** → 定义失败（`c.fail` 路径）。
两臂与跑通的 `Tl1` 的唯一差异是 **`GGML_TP5_LATEBIND_RHO=1`**。
**首要嫌疑：本轮新增的 LB2 几何**（`stats_words = late_rows*12u + 16u` 与
`cc_offset = stats_offset + 12u*16u*sizeof(float)` 及其余量校验）。
**这是本会话自己引入的回归，必须修。**
### 9.0-now-c 根因已定位：**Q8DOT 生产者从未实现 LB2 统计量折叠（实现缺口，非否决）**

| 检查 | 结果 |
|---|---|
| `TP5_LATE_Q_STATS` 出现位置 | 仅 `:436`、`:521` —— **均在 `TP5_LATE_Q`（精确）块内** |
| `TP5_LATE_Q_Q8DOT` 块（`:308-408`）内 `stats_out`/`s_out`/`StatsOut`/`ScatterOut` | **0 命中** |

**链条**：`RHO=1` → relay 等待 Q 着色器写出的 `(a_b, d_b, s_b)` 充分统计量；
而 aggressive 的 Q 生产者是 `TP5_LATE_Q_Q8DOT`，**它没有那段代码** → relay 自旋等待
永不到来的统计量 → 静默挂死（无 `c.fail`、无崩溃栈，日志停在模式行之后）。

**这不是回归，也不是"路线不通"** —— 是 **`aggressive x RHO` 这个组合从未被实现**。
历史记录里的 LB2 回退（+14.2%）来自**精确路**；aggressive 路从未跑通过该组合。

**修正路径（按用户裁定：必须修，不得以 BUG 否决路线）**：

1. **立即止血（fail-closed）**：`RHO=1` 且 Q 为 `q8dot-approx` 时，在计划定义期
   硬错并给出可诊断信息，而不是挂死。这是项目自身纪律（fail-closed）的直接应用。
2. **真正的修复**：给 `TP5_LATE_Q_Q8DOT` 加上与精确块同形的统计量折叠
   —— `a_b = <R_b,R_b>`、`d_b = <R_b,z_p>`、`s_b = scatter`，写入同一
   `stats_out`/`s_out` 布局。`TP5_LATE_ACT_Q8` 块（`:264-306`）**已经逐字读取了
   全部所需输入**（`local_z`、`scatter`、`residual`），折叠的增量最小；
   但需新增 `ResidualStats`/`LocalZStats` 绑定并在 rank 维上归约。
3. 两项都要有臂级账本字段（如 `stats_fold=`）才能被消融断言 —— 同 `rho=` 的教训。

**这是本会话自己引入的回归，必须修。**

---

### 10.0-a 【更正】`bg_14` 里 RHO 臂的"死"是**我自己的 fail-closed**，不是被测对象的失败

`bg_14` 的四个 RHO 臂（`T1`/`Tr1`/`Tr2`/`T2`）被记为 "DIED during load / Segmentation
fault"。核对 backtrace 后：

```
#13 llama_server                        (server.cpp:511)
#12 server_context::load_model          (server-context.cpp:10486)
#11 server_context_impl::load_model     (server-context.cpp:5311)
#10 common_tp5_apply_env (params=...)   at common/common.cpp:499   <-- 那行 throw
#9  __cxa_throw -> #8 std::terminate() -> ggml_uncaught_exception()
```

**`common.cpp:499` 正是本会话加的 fail-closed `throw`**（检查 `RHO=1 && mode==aggressive`，
**不受 `ALLOW_APPROX` 影响**）。即：那四个臂**根本没有进入 LateBind 代码**，
它们在加载期被我的止血补丁拦下。

**因此下表 `T1/Tr1/Tr2/T2` 一行的"Segmentation fault"不得读作"路径崩溃"：**

| 阶段 | `aggressive x RHO` 的真实状态 |
|---|---|
| Host Push 之前 | 真崩过（`llama_decode ret=-3` → SIGSEGV），根因当时未定位 |
| Host Push 之后、我的 fail-closed 之前 | **未测** |
| `bg_14`（带 fail-closed） | **被拦下，未测** |
| **`bg_18`（fail-closed 已删 + 统计量折叠已补）** | **首次真实测试** |

**教训**：止血补丁会**掩盖它本该仅暂时掩盖的那个问题**，并伪造出一个看起来像被测对象失败的信号。
这正是用户"禁止 fail-closed 伪装成修复"的实质理由 —— 不仅是交付问题，也是**证据完整性问题**。

### 10.0 `bg_14` 终局数据（镜像验证成立）

| 臂 | 中位 tok/s | `pred_n` | 结局 |
|---|---|---|---|
| `R1` / `R2` reference | **61.48 / 63.32** | 160 | 正常，`late=0 rho=0` |
| `Tl1` / `Tl2` aggressive `rho=0` | **10.79 / 10.96** | 171 | 运行但退化 |
| `T1`/`Tr1`/`Tr2`/`T2` `rho=1` | — | — | **被本会话的 fail-closed 拦下（见 10.0-a），非路径失败** |

**修正先前诊断**：`aggressive x RHO` 不是"静默挂死"，是**段错误**。
两端都指向同一根因（统计量从未被写）。

八臂镜像一致，工作一致性护栏工作正常（reference 全 `160`、aggressive 全 `171`）。

## 10b. 【根因】aggressive 输出退化的真因：Host Push 漏改了一处重置的**基址**

### 10b.1 症状与排除

`rho=0` 臂（即不涉及 LB2）：输出从有意义词（`健康减重原则`）迅速退化为标点汤；
`pred_n=171`（撞上限，遇不到停止符）；tok/s 逐请求恶化 **41→13→8**。
已系统性排除六道契约（归约几何、激活量化器、权重打包、邮箱协议、打包标志、点积原语+索引）。

### 10b.2 根因

Q sidecar 的发布协议（`tp5_hc_latebind.comp:439-446`，`TP5_LATE_Q_Q8DOT` 块）：

```glsl
const uint prior = atomicAdd(qmail[p.late_word_offset - 15u], 1u);
publish_last = (prior + 1u == p.n_workgroups) ? 1u : 0u;
if (publish_last != 0u) { ... qmail[p.late_word_offset - 16u] = 1u; ... }
```

即**"计数器累加满 WG 数即发布"** —— 因此**必须每 cycle 把计数器归零**。
重置确实存在（`tp5_relay_arm_bank`，每 cycle 经 `:5552` 预武装下一 epoch 调用）：

```cpp
if (c.ranks[i].bcast_host[bank] && c.late_host_offset) {
    auto * qctrl = tp5_late_q_control_ptr(c.ranks[i].bcast_host[bank], c.late_host_offset);
    qctrl[TP5_LATE_Q_READY_WORD]   = 0u;
    qctrl[TP5_LATE_Q_COUNTER_WORD] = 0u;
}
```

**但它的基址硬编码为 `bcast_host`。而在 `GGML_TP5_LATE_TRANSPORT=gtt-push`（Host Push）下，
Q8DOT 把载荷与控制行都写进了 `host_import_buf`。** 于是：

| | 目标 |
|---|---|
| Q8DOT 写控制 | `host_import_buf` |
| relay 重置控制 | **`bcast_buf`** ❌ |

**重置打在了着色器从不写的内存上** → host-import 里的计数器每 cycle 攀升 `n_workgroups`：
cycle 1 时 `0→20` 最后一个 WG 命中 → 发布一次；cycle 2 起 `20→40→60…`，
`prior + 1 == n_workgroups` **永不成立 → 永不发布**；ready 字停在 cycle 1 →
CPU 立即通过 → **LO 永远消费 cycle 1 的陈旧 Q**。

**这就是 41→13→8 的形态**：离 cycle 1 越远，隐状态错得越深。

### 10b.3 为何是本会话引入

Host Push（缺口一）当时改了**载荷绑定**（`q8dot_infos[2]` → `host_import_buf`）与 **CPU 读取基址**
（`qkctrl` 轮询 → `star_host_aligned`），**唯独漏了武装重置这一处**。
载荷侧、读取侧、写入侧三处都跟随 transport 了，第四处没有 —— 典型的"搬了家具忘了一把椅子"。

### 10b.4 已落地的三处修正

| 位置 | 修正 |
|---|---|
| `tp5_relay_arm_bank`（`:5254`） | 重置基址改按 transport 选择（`star_host_aligned[bank] + i*stride` vs `bcast_host[bank]`） |
| relay 的 sidecar 轮询（`:5736`） | `sidecar_base` 补上 transport 判断（原先硬编码 `star_host_aligned`，BAR-Pull+F16 会读错内存域） |
| relay 的消费点（`:5774`） | 看到 ready 后**退休握手**（清 ready 与计数器），对齐 STAR 路径 `*flags[i] = 0u` 的既有竞式 |

### 10b.5 变慢的机制：**尚未定位**（本小节自我更正）

上一版曾把变慢归因于"LO 自旋等待一个永不推进的 Q generation"。**该解释不成立，已撤回。**
反证：`tp5_hc_resume.comp:111-127` 的 `await_input()` 等的是
`inbox[GENERATION_WORD]`，那是 **relay 的 Y 代**（`GENERATION_WORD = 0` 位于 relay 头），
**不是 Q sidecar 的代**。Q 陈旧不影响 Y 代的推进，LO 不会因此自旋。

**已排除的其他候选：**

| 候选 | 反证 |
|---|---|
| PACK_WEIGHTS 每 cycle 重跑（102400 WG） | `:3850-3886` 确认 clear 是计划定义期的**一次性阻塞提交**，done flag 之后自禁用 |
| Q8DOT 核本身重 | 20 个 WG × 256 线程，量级为微秒 |
| CPU 侧 qctrl 轮询变慢 | ready 卡在 1 → CPU **立即通过**，只会更快 |

**真实形态（来自 `bg_14` 原始数据）**：steady 每 token 耗时
reference **15.0 ms 稳定**；aggressive **24 → 75 → 121 ms**（warm → r1 → r2）。
**warm 只慢 1.6×，随后爆炸** —— 这是**跨请求累积**，与"计数器单调攀升"同形，
但传播路径尚未证实。**不写成结论。**

### 10b.7 不是一个漏点，是**一族漏点**：孔径选择被零散应用

发现第一个漏点后做了一次系统性枚举：列出全部 `bcast_host` 使用并逐个归类。
**结果：同一选择器被应用了 4 处、漏了 6 处。**

| # | 位置 | 作用 | 漏掉的后果 |
|---|---|---|---|
| 1 | `tp5_relay_arm_bank` 控制重置 | 每 cycle 归零计数器 | **计数器永攀升 → Q 只发一次 → 退化 + 快得反常** |
| 2 | `tp5_relay_ensure_armed_epoch` 空闲检查 | 断言预武装 bank 干净 | **本该报警的守卫读错内存 → 恒真通过** |
| 3 | `[5] = late_rows`（`arm_bank`） | 发布活跃行数 | Q8DOT 回退到 `p.active_rows`（**上限**） |
| 4 | `[5] = late_rows`（`first_bank`） | 同上 | 同上 |
| 5 | `[5] = one_shot_rows` | 同上 | 同上 |
| 6 | relay sidecar 轮询的 `sidecar_base` | 轮询就绪 | BAR-Pull+F16 读错内存域 |

**第 2 项是其中最讽刺的一项**：唯一能在定义期抓住"Q 只发布一次"的断言，因为同一个遗漏而失效。
**缺陷与其守卫由同一个错误造成** —— 这就是它为何能一路带病上线。

### 10b.7-b 第 3–5 项不是"二选一"，而是"两个都要写"

修正过程中差点自己制造一个新缺陷：`[5]` 的活跃行数**有两个消费者**：

- `tp5_hc_resume.comp` 的 resume/norm/LO 内核读 **`inbox`**（binding 0 = `bcast_buf`）
- `TP5_LATE_Q_Q8DOT` 内核读 **`qmail`**（Host Push 下 = `host_import_buf`）

若把发布改成"按 transport 二选一"，就会把**Y 路的行数打断**。正确形式是**两者都写**（BAR-Pull 下二者重合）。

### 10b.7-c 修法：修集合，不修补丁

新增两个助手，把所有调用点收敛，使这一族无法再漂开：

```cpp
static void * tp5_late_q_aperture(tp5_comm & c, size_t rank, size_t bank);
    // Q 孔径：Host Push 下为 host-import RAM，否则 bcast VRAM
static void tp5_late_publish_rows(tp5_comm & c, size_t rank, size_t bank, uint32_t rows);
    // 活跃行数：两个孔径都写（不同时才写第二个）
```

现已路由经过它们的共 7 个调用点：武装重置、空闲检查、三处行数发布、孔径定义自身、双孔径发布。
**关键：以后新增任何读写 Q sidecar 的代码，必须经过 `tp5_late_q_aperture`，不得再写裸 `bcast_host`。**

### 10b.7-d 测量仪器先校准，再用它出裁决

把"是否连贯"从人眼判断变成了 harness 的自动断言。**先用已知样本校准它，再用它判定：**

| 样本 | 判定 | CJK 字数 | 标点数 |
|---|---|---|---|
| reference r1（已知好） | **COHERENT** ✅ | 241 | 19 |
| aggressive r1（已知坏） | **DEGENERATE?** ✅ | **6** | **143** |
| aggressive r2（已知坏） | **DEGENERATE?** ✅ | 13 | 154 |

**退化不是"略差"：95% 的输出是标点。** 阀值 `cjk>=20` 两侧余量充分（好 241 / 坏 6–13），
无需调参。`bytes` 单独完全分不出来（坏样本也有 ~200 字节）。

**这一项必须记录在案**：本会话已有多次"用有瑕疵的证据得出（撤回或未撤的）结论"，
而仪器本身未经校验正是这类错误的温床。

### 10f.2-a 【新结果】`aggressive + overlap=legacy` 与参照**持平**

在本会话修好两个根因（§10b 孔径族 + §10e 激活量化器索引）后的首次测量：

| 臂 | tok/s | `pred_n` | 连贯性 | 账本 |
|---|---|---|---|---|
| **Alex1** aggressive + `overlap=legacy` | **61 / 64** | 171 | ✅ COHERENT (cjk=187) | `late=94 rho=0 q8_fast=94 overlap=legacy transport=gtt-push` |
| agg（早前）aggressive + `overlap=overlap` | 38 / 51 | 171 | ✅ COHERENT | `overlap=norm-q` |
| reference | 61–63 | 160 | ✅ | `late=0` |

**输出质量**（`<think>` 内的真实推理）：

> 用户要求用八句话简要介绍四个主题：健康减重原则、每周运动安排、饮食建议、睡眠误区。
> 我需要八句话覆盖这四个方面，每主题两句，简洁准确。让我组织：…

### 10f.2-a# 六臂完整数据集（含镜像）—— 本会话最干净的一次测量

同二进制、同 prompt、`temp=0 seed=42 stream=false`、`n_predict=171`、
`GGML_TP5_LATE_TRANSPORT=gtt-push`、ZSNAP=1、aggressive 刀锋：

| 臂 | 中位 tok/s | 镜像中位 | 发散 | 输出 |
|---|---|---|---|---|
| `Alex` legacy | 63.36 | 66.29 | 4.6% | COHERENT cjk=187 |
| `Areo` reordered | 61.17 | 64.96 | 6.2% | COHERENT cjk=187 |
| `Aovl` overlap | 65.15 | 62.64 | 4.0% | COHERENT cjk=187 |

**三条独立确认**：

1. **速度全部落在 61.2–66.3** —— 落在本机已知的 ~5% 会话漂移带内，**统计上不可分**。
2. **账本完全相同**：六臂均为 `dispatches=2879 barriers=2496 copies=191`。
   **命令流可证同一** —— 这是比 tok/s 更硬的证据。
3. **输出逐字节相同**：六臂 cjk=187、镜像 r1==r2 校验通过。

⇒ 与代码结论（该模式在此刀锋上无作用点）**三向对齐**。

### 10f.2-b 【已证伪】"重叠方向相反"是在**单次未镜像样本**上得出的

我曾根据一次 `aggcheck` 测得「aggressive + `overlap=overlap` = 38/51」而宣布
「重叠是 1.6× 惩罚、方向与理论相反」。**该结论建立在两个错误上：**

**错误一：模式在 aggressive 刀锋上根本无作用点。** 穷举 `tp5_get_late_overlap_mode()`：

```
:6309-6360   精确路的 ovl_sched（aggressive 不经过）
:6422        账本名字符串
:6398        WAR 校验守卫
```

**aggressive 分支一处都不经过** —— 它的发射顺序是无条件的。

**错误二：38/51 是冷态/漂移样本。** 该臂内 r1=38、r2=51（**同臂内 34% 散布**），
而镜像测量（`bg_26`）给出：

| 臂 | tok/s | 模式作用点？ |
|---|---|---|
| `Alex1` (legacy) | **61 / 64** | — |
| `Areo1` (reordered) | **60 / 61** | — |
| `Aovl` 待镜像确认 | — | 代码上不可能不同 |

**教训（本会话第三类同形错误）**：

1. 先前：**拿有瑕疵的证据撤回了正确结论**（§9.0）
2. 先前：**把自己的 fail-closed 当成被测对象失败**（§10.0-a）
3. 本次：**拿未镜像的单样本发布了"模式有 1.6× 影响"的结论**

三类都是同一个毛病：**在证据链未闭合时下结论**。镜像臂存在的意义就是抓这个 ——
它们这次抓住了。

**旧小节（已作废的推论）保留于下；仅作错误链追溯用。**

### 10f.2-b-old 重叠方向与理论相反（**已作废**）

重叠把 Q8DOT 挪到 norm **之后**，于是 LO 消费者必须等它；而 legacy 下 Q8DOT 先跑完，
LO 一到就有数据。

**设计假设是"Q 落进 norm 的 Y 等待窗口"。实测该假设不成立 —— 重叠反而把 Q 推到了
LO 之前的关键路径上，代价约 1.6×。**

这同时说明：**关键路径上是 LO 的输入就绪，不是 Q 与 norm 的排队关系。**
若如此，则 L1 的正确定位应是让 Q **更早**完成（而不是与 norm 重叠），
——而 L2（ρ）正是那条路：它把 Q 的充分统计量并在 Y 里一次发布。
**这是后续工作最重要的方向性输入。**

### 10f.2-c 两条刀锋上的门控不对称（逐字回读确认）

逐字回读 `:6288-6302`（此前我两次误读此段，两次均在此纠正）：

```cpp
} else if (outgoing->late_q8_fast) {          // aggressive
    ... q_norm_barrier ...
    if (!emit(p2, 0, norm_end)) return false;        // norm 先
    ... WRITE_TREFS ...
    if (!emit(pre, q_contract)) return false;        // Q 收缩后
    ... DISPATCH_Q8DOT ...
} else {                                       // exact
    const bool ovl_sched = ovl_mode != LEGACY && zsnap && ...;
    if (ovl_sched) { ...重叠... } else { ...legacy... }
}
```

| 刀锋 | 重叠顺序 | 受 `GGML_TP5_LATEBIND_OVERLAP` 约束？ |
|---|---|---|
| **aggressive** | **永远** | ❌ 无门控 |
| **exact** | 可切 | ✅ |

**推论（比先前表述更精确，也更严重）：**

> **aggressive 刀锋没有 legacy 模式可言 —— 它总是重叠。**
> 而 §10c 已证 LB2 需要非重叠调度（发布器必须在 P1 内完成）。
> **因此 `aggressive x RHO` 在结构上不可共存，不是配置问题。**

**对 `bg_26` 的影响**：六臂（Alex/Areo/Aovl × 镜像）将产出**几乎相同的数字**。
这本身是**确认性证据**（aggressive 忽略该变量），而**不是**"重叠无收益"的证据。
**不得据此宣布 L1 无效** —— 该结论只能在精确刀锋上（那里门控真实存在）得出。

### 10f.3 L1/L2 解耦的精确设计

**重叠发射的完整语义**（`:6310-6354`）：

```cpp
const size_t q_split = outgoing->late_q_begin[slot];   // :4360 处 = Q 段起点
if (ovl_sched) {
    emit(pre, inject_end, q_split);   // P1：散列前导 + zsnap
    emit(p2, 0, norm_end);            // P2：norm（等 Y）
    if (REORDERED) { /* 保留 WAR 屏障 */ }
    emit(pre, q_split);               // P2：从 q_split 起的全部内容
}
```

**`late_q_begin` 指向 Q 段起点**，而 aggressive 路的 Q 段开头就是 ACT_Q8，随后 Q8DOT，随后发布器。
所以整段（含统计量）被搬进 P2。

**解耦表：**

| 内容 | 谁消费 | 必须在哪 |
|---|---|---|
| 统计量折叠 + 统计量发布 | **relay**（在 P1→P2 间隙算 ρ） | **P1** |
| Q 收缩 + Q 载荷发布 | **LO 消费者**（norm 之后） | 可移入 P2 ✅ |

**实现三步**：

1. 给 Q8DOT 块加 `TP5_LATE_STATS_ONLY`：定义时 `main()` **只跑统计量折叠**
2. 生成器注册该变体（同 `tp5_hc_late_q_q8dot_stats` 的形状）
3. cpp：新增第二个分裂标记 `late_q_stats_end`（统计量派发结束处），
   **重叠改用 `q_split = late_q_stats_end` 而非 `late_q_begin`** ——
   统计量留在 P1，只有 Q 收缩进入重叠区

**代价**：Q8DOT 的统计量折叠在 `_stats` 变体里会重复一次（Q 段仍带着它）。
若要避免，给 Q 变体加 `TP5_LATE_NO_STATS_FOLD`；但**先量后优** —— 重复的代价远小于一次额外派发。

**前提（已修正）**：原文写"前提是 `Aovl − Areo` 显著大于零"。**该前提本身作废** ——
`bg_26` 的六臂证明 `GGML_TP5_LATEBIND_OVERLAP` 在 aggressive 刀锋上**根本没有作用点**
（账本 `dispatches/barriers/copies` 六臂全同、输出逐字节全同）。

**真正的前提是两个独立的、可用测量回答的问题：**

1. **L2（ρ）本身的代价是多少？** —— 正在用单因素镜像对照 `E0`/`E1` 测。
   若 L2 在天生的 legacy 调度下已是**正收益或中性**，那就无需解耦，直接用它；
   解耦只在"L2 有收益但被 legacy 调度限制住"时才值得做。
2. **重叠加持到 L2 上有额外收益吗？** —— 这是解耦的**唯一**价值主张：
   让 ρ 能上"重叠调度 + 快速刀锋"的组合。

**顺序**：先看 `E0`/`E1`。若 L2 中性，则解耦的收益上限只是"重叠的加成"，
而重叠在 exact 刀锋上的实测贡献（`exact + overlap` 36–43 vs 待测的 `exact + legacy`）
也需先量。**两个未知数都未闭合之前，不写解耦代码。**

## 10h. 【根因】L2 让模型提前 EOS：`cc` 偏移硬编码容量，**越界写 512 字节**

### 10h.1 症状（单因素镜像对照 E0/E1 直接抓出）

镜像交错，四次测量全部一致：

| 臂 | r1 / r2 | `pred_n` | `stop_type` | 输出 |
|---|---|---|---|---|
| `E0` rho=0 | 42.7 / 33.8 | **171 / 171** | `limit` | 完整连贯答案 |
| `E0b` rho=0 镜像 | 34.0 / 36.4 | **171 / 171** | `limit` | 完整连贯答案 |
| `E1` rho=1 | 36.6 / 36.4 | **28 / 28** | **`eos`** | 思考中途戛然而止 |
| `E1b` rho=1 镜像 | 35.2 / 37.4 | **28 / 28** | **`eos`** | 思考中途戛然而止 |

**`rho=1` 让模型在第 28 个 token 就 EOS，且四次测量完全一致（100% 可复现）。**
这不是浮点次序差异（那是 ULP 级），是**语义级**改变
⇒ **L2 首先是个正确性问题，不是性能问题。**

### 10h.2 根因

```
VK_TP5_DIRECT_COLUMN_TILE = 4      ← TP5_LATE_CAPACITY_ROWS 的真值
```

而我在**两侧都硬写了 `12u * 16u = 192` 字**：

```
cpp  :5770  cc_offset   = stats_offset + 12u * 16u * sizeof(float)   // = +768 字节
             stats_bytes = (late_rows*12 + 16) * 4                    // late_rows=4 → 256 字节
             guard: stats_offset + stats_bytes > stride ? fail
glsl :369    cc = inbox[p.rho_word_offset + 12u * 16u + token]
```

**两侧相互一致，但一致地指向了错误位置。** 而那个位置在 CPU 边界校验（256 字节）
**之外 512 字节** —— 写进了 bank 里别人的地盘：

**越界写 → ρ 读到垃圾 → `u` 错 → 提前 EOS** ✅

**这是本会话自己引入的**：`cc` 几何是我为实现 L2 尖端时加的。

### 10h.3 修正：从单一来源导出

```glsl
const float cc = uintBitsToFloat(inbox[p.rho_word_offset + 12u * TP5_LATE_CAPACITY_ROWS + token]);
```
```cpp
const size_t cc_offset = stats_offset + 12u * VK_TP5_DIRECT_COLUMN_TILE * sizeof(float);
```

修正后 `cc` 落在 `stats_offset + 192 .. 208`，完全在 256 字节的受检区间内 ✅。

**教训（本会话第四类同形错误）**：**两个硬编码的拷贝互相对得上，不等于它们是单一来源。**
前三次是"证据未闭合就下结论"，这一次是"两边抄了同一个错"。
防御手段一样：**从各自的单一来源符号导出，而不是写字面量。**


### 10h.4 L2 的第二个缺陷：跨 rank 的 `d` 求和被我一并删掉了

把 ρ 的合成移进 LO_U 着色器时，我删掉了 CPU 侧那段**跨 rank 折叠**的代码：

```cpp
// 被删掉的（原逻辑）
if (rk == 0) { 拷 12 字 } else { for (b<4) fold[t*12 + 2b+1] += src[t*12 + 2b+1]; }
```

而着色器现在读的是**本 rank 的** `d0 = inbox[so+1]`。恒等式要的是
`Σ_ranks ⟨R_b, z_p⟩`，**不是单个 rank 的偏量**。少一个 `Σ` ⇒ ρ 错 ⇒ `u` 错 ⇒ 提前 EOS。

**已修复**：恢复跨 rank 折叠，并把折叠后的 `d` 写回每个 rank 的区域（使所有 rank 的
LO_U 看到同一个全局值），同时保留 `cc` 发布与着色器的内联合成。

### 10h.5 修复后的实测：从「总是失败」变为「不稳定」

单臂 `exact + legacy + rho=1`（含两项修复）：

| 请求 | tok/s | `pred_n` | `stop` |
|---|---|---|---|
| r1 | 38 | **171** | `limit` ✅ |
| r2 | 33 | **27** | `eos` ❌ |

**⇒ 两项修复都有实效（从「四次全部 27」变为「先 171 后 27」），但 L2 仍不稳定。**

**形态是跨请求累积** —— 同一个 session 内先对后错，与那族孔径缺陷同形。
**ρ 路径上至少还有一处未定位的缺陷。不写成结论，作为下一步的入口。**

**L2 的状态：未完成。** 而 L1（以及修好的 aggressive 刀锋）是**已验收且稳定**的 ——
六臂 61–66 tok/s、账本全同、输出逐字节相同。


### 10h.6 已排除的候选（为下一步收窄面）

| 候选 | 反证 |
|---|---|
| `lb2_cc[]` 跨调用累积 | 它是**局部数组且 `= {}` 零初始化**（`:5702`），每次调用重建 |
| `cc` 发布不是每 cycle 执行 | 发布块（`:5854-5895`）在 relay 主路径上，无条件执行 |
| 跨 rank 折叠写回的 `d` 污染下一轮 | 下一轮的 GPU 发布器会用新值覆盖整个 stats 区 |
| `s_b` 未到达 bank | 精确路的 `s_out` 写入 `late_stats_buf` 的 `+max_rows*4*2*4` 处，即每 token 12 字中的第 8..11 字 ✅ |
| 折叠语义与原实现不一致 | 恢复后的折叠与原始 CPU 代码同形（`a`/`s` 取 rank 0，`d` 跨 rank 求和） |

### 10h.7 L2 的当前裁决：**未完成，停止续挖**

**已修两处真缺陷，L2 从「四次全部 27 token」变为「171 后 27」。仍不稳定。**

**停止的理由不是"没价值"，而是边际风险：** 继续在 ρ 路径上做未经充分定位的改动，
会威胁到已验收的 L1 成果（六臂 61–66、账本全同、输出逐字节相同）。
**L2 的下一步入口已写清**（剩余可疑面：`a_b` 只取 rank 0 的偏量是否在数学上成立、
ρ 的近 tie 敏感性），但它需要一轮独立的、与 L1 隔离的工作。

**明确不宣称的东西**：L2 可用；ρ 数学正确；L1+L2 可共存。


### 10h.8 已解：为何「`a` 只取 rank 0、`d` 跨 rank 求和」在数学上是对的

这是原实现的**不对称**，一度看起来像缺陷。它其实来自张量布局：

| 量 | 布局 | 正确取法 |
|---|---|---|
| `R_b`（残差） | **复制**（源于已归约的 TP `y`） | 各 rank 相同 ⇒ `a_b = \|R_b\|²` **取 rank 0 即可** ✅ |
| `z_p`（位置投影） | **分片** | `⟨R_b, z_p⟩ = Σ_ranks ⟨R_b, z_p^rank⟩` **必须跨 rank 求和** ✅ |

**两者都源于同一个布局事实，所以不对称是对的。** 我恢复的折叠语义与原实现一致，
**因此 `a_b` 不是 L2 的病灶 —— 该候选已排除。**

（反证：若 `R` 分片，则 `a` 取 rank 0 是错的；若 `R` 复制，则 `d` 求和是错的。
两者不能同时错而不被历史验证发现，因此“复制 + 分片”是唯一自洽的解释。）

### 10h.9 失败形态已定性：**首次成功，之后全部失败**

连打六次**完全相同**的请求（同 prompt、`temp=0 seed=42`、`n_predict=171`）：

```
req1: 171, stop=limit  OK
req2:  27, stop=eos    FAIL
req3:  27, stop=eos    FAIL
req4:  27, stop=eos    FAIL
req5:  27, stop=eos    FAIL
req6:  27, stop=eos    FAIL
```

**这是无歧义的跨请求累积**：某样东西在启动时正确，**第一次请求之后再也没有恢复**。
`rho=0` 在同一台机器上四次测量全部为 171 —— **所以它与 ρ 路径绑定，不是模型本身的不稳定。**

**形态的含义**：请求 1 记录（record）计划，请求 2+ 重放（replay）它。
失败从**第二次执行**开始 ⇒ 病灶在「**只在首次执行中正确、在重放中不成立**」的那类状态里。

### 10h.11 四次修复尝试的结果（本次为最后一次）

| # | 修复 | 对失败形态的效果 |
|---|---|---|
| 1 | `cc` 偏移从硬编码 `12u*16u` 改为单一来源符号（真值 4） | **有效**：四次全 27 → 首请求 171 |
| 2 | 恢复跨 rank 的 `d` 折叠（搬迁 ρ 合成时被删） | **有效**：仍是首请求 171，但确认了折叠不可省 |
| 3 | `lb2_cc` 尺寸从字面量 16 改为 `VK_TP5_DIRECT_COLUMN_TILE` | **零结果**（守卫未触发） |
| 4 | 容量守卫 `>16u` → `> VK_TP5_DIRECT_COLUMN_TILE` | **零结果**（同上） |

**形态始终为「首请求 171、之后全 27」** —— 修复 3/4 未改变它。

### 10h.14 【根因】统计区的基址：三方里只有中继用了「实时行数」

判定实验的输出（同一运行内并排）：

```
[tp5-pubfp] rank=0 bank=0 idx=0 late_count=5120 dst_off=544832 n_stats=48
[tp5-l2fp]  call=0  bank=0 late_count=1280 stats_off=529472 cc_off=529664
```

**同一基址（`payload_offset = 524352`），但 `late_count` 差一个 `streams` 因子：5120 vs 1280。**

三处计算并排：

| 参与者 | 表达式 | 值 |
|---|---|---|
| 记录期发布器（`:4025`） | `max_rows * streams * late_rank` = 4×4×320 | **5120** ✅ |
| LO_U 的 `rho_word_offset`（`:4523`） | `capacity_rows * streams * late_rank` | **5120** ✅ |
| **中继（`:6740`）** | **`stage_rows * streams * late_rank`** = 1×4×320 | **1280** ❌ |

**`stage_rows`（实时行数）vs `max_rows`（容量）—— 而统计区的位置是固定的**
（LO_U 就是按容量算的）。中继用实时行数算基址，于是每个统计读取都**偏低 4 倍，
落进了 Q 载荷本身** ⇒ 读到零 ⇒ `ss = 0` ⇒ `ρ = 0` ⇒ `u = 0` ⇒ 模型在第 28 个 token 提前 EOS。

**为何是「首请求成功、之后全失败」**：`stage_rows` 在首个请求为容量值（冷路径先跑满载行数），
之后降为真实行数，两者才开始分歧。这把 10h.9 的形态与根因对上了。

**修法**：给中继增加一个**容量口径**的参数 `late_stats_floats`，
**只用于统计区基址**，不碰用于 Q 载荷归约的实时 `late_count`
（两者服务不同目的：载荷只传有效行，统计区位置固定）。

### 10h.15 修正后的实测：从「首后全败」到不稳定（约半数成功）

容量基址修正后，六次连续同请求：

```
req1: 171 OK   req2: 27 FAIL   req3: 171 OK
req4: 27 FAIL  req5: 27 FAIL   req6: 171 OK
```

**从「首后全败」变为约 50% 成功率，且六次输出全部连贯**（cjk=179/31/31/31/31/31）。

**⇒ 基址修正是真的（读取现在落在发布器的区域内），但还有一个独立的不稳定源。**

**两个候选（未区分）**：

1. **握手竞态**：`ρ` 现在由 LO_U 从 bank 读取并内联合成，而它就读在发布/退休的边界上；
2. **数值敏感**：`ρ` 的计算路径与原 CPU 不同（`1/sqrt` 在着色器、`cc` 由 CPU 发），
   模型在近 tie 处翻转决策——但 `rho=0` 在同一机器上极其稳定（四次全 171），
   这说明差异确实来自 ρ 路径。

**判定方法（未做）**：把 `ρ` 的合成退回 CPU（保持已修好的基址），
若不稳定消失则候选 2 成立；否则候选 1 成立。

### 10h.16 【已确认】容量基址修复生效；病灶已收敛到 `d` 一项

```
记录期： late_count=5120  dst_off=544832
中继期： late_count=1280  stats_off=544832   <- 吻合 ✅
sumA=0.0043 / 0.116 / 0.119    <- 非零 ✅（修复前恒为 0）
sumS=0.0014 / 0.0043 / 0.0043  <- 非零 ✅
sumD=0                         <- 仍为零
```

**⇒ 容量基址修复是真的**（读取终于落在发布器写的区域内），**但它暴露了下一个病灶。**

### 10h.17 病灶已定位到单项：`d_b = ⟨R_b, z_p⟩` 恒为零

于是 ρ 实际算的是 `ss ≈ a_b + s_b²·c` —— **缺了 `2·s_b·Σ_p⟨R_b,z_p⟩` 那一项**。
这正好解释「部分正确、约半数成功」：ρ 不再是 0（灾难级），而是一个**缺项的近似**。

**已能区分两个视图：**

| 视图 | 绑定 | 用途 | 状态 |
|---|---|---|---|
| `local_z4` | 3（Q 主循环） | `rz0 = z0 * s` | **活的**（Q 能算出连贯输出） |
| `local_z_stats` | 8（统计量折叠） | `d_b += dot(rv, zv)` | **零** ❌ |

而 `a_b = dot(rv, rv)` 非零 ⇒ `residual`（绑定 7）是活的。
**⇒ 病灶就是：统计量折叠读的那个 `z_p` 视图拿不到数据。**

**下一步（具体到一处）**：核对 `q_infos[8]`（`LocalZStats`）的 buffer/offset 是否真的指向 zsnap，
并与 `q_infos[3]`（`LocalZ4`）比较 —— 两个视图应当看到同一份 `z_p`。

### 10h.18 对 `d = 0` 的推理：精确路的 `z` 可能一直是零

已核实：`q_infos[3]`（`LocalZ4`，主循环）与 `q_infos[8]`（`LocalZStats`，统计量折叠）
**表达式逐字相同、均解析到 `plan.late_zsnap_buf[i]` + 0**，且两者在**同一次内核调用**中读取。

**⇒ 它们不可能一个看到零、另一个看到非零。**

因此 `a_b = dot(rv,rv)` 非零（residual 视图是活的）而 `d_b = dot(rv,zv)` 恰为零，
只能意味着 **`zv ≡ 0`**。

**这意味着：主循环也一样在用z≡0 跑**，即
`rz0 = z0 * s + residual` 在 rank 0 退化为只用 `residual`，而在 rank 1–4 上 **`rz0 ≡ 0`**。
那正是一个**残缺但不死**的 Q —— 与观测到的“部分连贯”完全一致。

**而 aggressive 刀锋不受影响**：它读的是**活的 `trefs`**（`act_q8_infos[2] = {trefs[i].buf,...}`），
不走快照。这解释了两条刀锋为何表现如此不同。

**快照拷贝本身（`:4359-4370`）已读，看似正确**：
`trefs[i].buf + trefs[i].offset -> zsnap + 0`，字节数 = `capacity_rows * width * 4`，
且在 `late_q_begin` 之前（落在 P1 内）✅。

**还需 GPU 侧回读才能定案的点（未做）**：

1. `trefs[i].offset` 是否真的指向 z_p 起点（若偏了，拷过去的就是别的区域）；
2. 快照拷贝与写入 `trefs` 的生产者之间的屏障是否足够
   （`mb_scatter` 是 `WRITE|TRANSFER_WRITE -> READ`，**不包含 `TRANSFER_READ`**）；
3. `late_zsnap_buf` 的分配尺寸（`:3804`）是否 ≥ `zsnap_bytes`。

**候选 2 特别可疑**：拷贝是 TRANSFER 操作，需要保证源（`trefs`，由 compute 写入）
对 **TRANSFER_READ** 可见 —— 而现有屏障只声明了 `SHADER_READ`。

### 10h.19 【已修】zsnap 拷贝前的屏障缺了 TRANSFER_READ

```cpp
// 修前
VkMemoryBarrier mb_scatter{..., 
    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,   // src
### 10h.20-b 【重要】但稳定性大幅改善 —— 三项连续全部成功

```
req1: 39 tok/s pred_n=171 stop=limit  ✅
req2: 47 tok/s pred_n=171 stop=limit  ✅
req3: 48 tok/s pred_n=171 stop=limit  ✅
```

而修复前（同一测试）是 `171/27/171/27/27/171` 的约半数。**⇒ 屏障确实消除了一个真实的竞态。**

**这是一个重要的区分**：`sumD=0` 是**确定性**的（400 次全部恰为零），
而失败形态是**非确定性**的。**确定性不可能来自竞态** —— 因此两者是**两个独立的事实**：

1. 屏障修复 → 消除了非确定性的竞态 → **稳定性改善**
2. `d` 为零 → 一个**确定性**的数值事实（可能本身正确，也可能另有一个静态缺陷）

**不把两者混为一谈。** `n=3` 不足以定案，正在跑 8 次确认。

### 10h.22 八次确认：前段 5/5，随后再次失败

| 请求 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `pred_n` | 171 | 171 | 171 | 171 | 171 | **27** | **27** |

**⇒ 壁垒修复改善了前段（修复前约半数，修复后前 5 次全过），但不稳定仍未消除。**

**形态再次是「先好后坏」——与请求序号相关。** 这与 §10h.9 的定性一致，
只是转折点从第 2 次推迟到了第 6 次。

### 10h.23 L2 的最终裁决（本轮）

**三项修复已落地，其中两项已由指纹证实：**

| # | 修复 | 证据 |
|---|---|---|
| 1 | `cc` 偏移从硬编码 `12u*16u` 改为单一来源（真值 4） | 首请求从恒败变为可用 |
| 2 | **统计区基址改为容量口径**（中继曾用实时行数） | `dst_off == stats_off == 544832`；`sumA/sumS` 从恰为零变为非零 |
| 3 | `mb_scatter` 补上 `TRANSFER_READ`（教科书级同步修复） | 前段 5/5 通过 |

**但仍未完成**：八次中三次失败（第 6、7 次），仍是请求序号相关。

**已确定的下一个入口**：

- `sumD` 恰为零是**确定性**事实（400 次不变），而失败形态是**非确定性**的 ——
  两者是**独立**的两件事，不得混为一谈；
- 失败的**转折点随修复而后移**（第 2 次 → 第 6 次），这是一个强信号：
  存在一个**逐步累积的量**，而每一项修复只是让它积累得更慢；
- 候选入口：握手退役、`qctrl` 计数、以及 `zsnap` 在多次重放后的内容。

**明确不宣称**：L2 可用；ρ 数学正确；L1+L2 可共存。

### 10h.24 【更正上一条】"6/8 成功"高估了 —— 成功的那几次内容也已退化

连贯性探针看的是 `.content`（`</think>` 之后的正文）。同一批八次请求：

```
r1: cjk=179 marks=28   <- 正常答案（参照臂也在这个水平）
r2: cjk=31  marks=77   <- 标点为主
r3: cjk=31  marks=77
r4: cjk=31  marks=77
r5: cjk=31  marks=77
r8: cjk=31  marks=77
```

**⇒ 即使跑满 171 token，"成功"的那些请求内容也已退化。**
参照臂与 `rho=0` 在同 prompt 下是 `cjk=175~241, marks=18~28`。
**`marks=77` 落在退化区**（历史退化样本是 `marks=143~154`，正常是 `18~28`）。

**因此 "6/8 成功" 的说法高估了实情**：按 token 数看 6/8，按**内容质量**看只有 1/8。
**L2 不只是"不稳定"，它产出的内容本身就是坏的**，只是程度从灾难（`ρ=0`）
降到了退化（缺项的 `ρ`）。

**这把 §10h.17 的 `d` 缺项从"可疑"变成了"确定的关键项"** —— 缺掉 `2·s_b·Σ⟨R_b,z_p⟩`
正好会给出这种"能跑但内容退化"的形态。

### 10h.25 L2 的最终裁决（本轮，含上一条的更正）

**三项修复已落地，两项由指纹证实**（`cc` 偏移、统计区容量基址），第三项是教科书级同步修复。
**但 L2 仍不可用**：

- 按 token 数：八次中六次跑满 171，两次提前 EOS；
- **按内容质量：只有一次正常**，其余六次均为标点退化。

**⇒ 病灶就是 §10h.17 定位的那一项（`d` 恰为零），而不是别的。**
它是一个**确定性**事实，与失败的非确定性是两个独立问题；
两者叠加才产生当前现象。

**修好 `d` 是下一轮的第一个动作**，入口已具体到一处：
`trefs[i].offset` 与 `late_zsnap_buf` 的快照拷贝（aggressive 路直接读活 `trefs` 且工作正常，
所以数据在源里是好的，坏在拷进快照那一步）。

### 10h.27 尺寸统一的结果：零结果；`d ≡ 0` 仍未被解释

统一后重跑：`sumD=0` **不变**，连贯形态也不变（r1 正常、r2/r3 退化）。

**⇒ zsnap 尺寸不是病因。** 该统一仍保留（两个同义变量各自维护是真实隐患）。

### 10h.28 本轮 L2 的六次修复尝试总表

| # | 修复 | 结果 |
|---|---|---|
| 1 | `cc` 偏移从硬编码改为单一来源（真值 4） | **有效** — 首请求从恒败变为可用 |
| 2 | 恢复跨 rank 的 `d` 折叠 | **有效** — 与 #1 共同使统计量到达 |
| 3 | `lb2_cc` 尺寸改为容量常量 | 零结果 |
| 4 | 容量守卫 `>16u` → `> VK_TP5_DIRECT_COLUMN_TILE` | 零结果 |
| 5 | `mb_scatter` 补 `TRANSFER_READ` | **有效** — 失败转折点从第 2 次推迟到第 6 次 |
| 6 | zsnap 拷贝与分配尺寸同源 | 零结果 |

**核心未解**：`sumD` 恰为零（确定性、跨 400+ 次调用不变），而 `sumA`/`sumS` 是活的。
**六次尝试均未能改变它。**

### 10h.29 L2 的停止裁决与理由

**停止的理由是工具边界，不是价值判断：**

1. 已穷举了**静态**可判定的候选（引用、屏障、尺寸、单一来源、并发语义）；
2. 剩下来的都要**读运行时 GPU 缓冲内容**（`late_zsnap_buf` vs `trefs` 逐字节对比）
   才能区分「拷进去的就是零」与「数据对但视图读不到」；
3. **继续无插桩的改动会给已验收的 L1 带来风险**（这是本轮反复重申的底线）。

**下一轮的第一个动作（具体到一处）**：加一个回读路径，把 `late_zsnap_buf` 与 `trefs`
各前 64 字节 dump 到 CPU，逐字节对比。

### 10h.31 【结论】判别实验成立：快照就是 `d ≡ 0` 的原因

`exact + rho=1 + legacy + `**`ZSNAP=0`**（直接读活的 `trefs`，与 aggressive 刀锋同读法）：

```
sumA = 0.0043 / 0.116 / 0.119      （不变）
sumD = -0.0977 / -0.114 / -0.0007  <- 从「恰为零」变为非零 ✅
sumS = -0.0070 / -0.0004 / 0.0042  <- 从全正变为有符号 ✅
```

**⇒ `z` 在源里是好的（aggressive 刀锋已经证明），坏的是「拷进快照」那一步。**
这与我先前的推理一致，且现在是**实验事实**，不再是推断。

**为何输出仍退化（r2 cjk=31 marks=77）**：`sumS` 也从“全正”变为有符号 ——
说明快照中读取的 `s_b` **同样**是错的（而 `r2` 是标点为主的退化文本）。

**但稳定性仍未解决**（r1/r2 过、r3/r4 失败）⇒ **存在第二层独立的病因。**

### 10h.32 两层病灶的分层

| 层 | 现象 | 性质 | 状态 |
|---|---|---|---|
| **L1** | `d ≡ 0`（400+ 次不变） | **确定性** | **已定位到快照拷贝** |
| **L2** | 请求序号相关的不稳定 | **非确定性** | **未定位** |

两者叠加产生当前的一切现象。**把确定性的一层与不确定的一层分开，是下一轮的正确切分。**

### 10h.33 第二个判别：拷贝机制本身是否可用

`zsap` 为零有两种可能，而它们的修法完全不同：

| 假设 | 含义 |
|---|---|
| **A. 拷贝本身坏了** | `vkCmdCopyBuffer` 结构性问题（用途标记、尺寸、对齐） |
| **B. 拷贝时 `z_p` 尚未就绪** | 拷贝发生在生产者之前 ⇒ 拷到的是零 |

**已排除 A 的一半**：`trefs[i].buf` 在 `:3553` 已作为 `vkCmdCopyBuffer` 的**源**使用
（wire 打包路径），故 `TRANSFER_SRC` 用途存在。

**新增判别开关 `GGML_TP5_LATEBIND_ZSNAP_LATE=1`**：把快照**推迟到 Q 收缩之前**，
即 norm 覆写 `trefs` **之后**。此时快照内容必然是错的（拷到 y 而非 z_p），
但它能回答唯一关键的问题：

- 若晚期快照的 `d` **非零** ⇒ 拷贝机制可用 ⇒ **假设 B**（早期时点 z_p 未就绪）；
- 若晚期快照的 `d` **仍为零** ⇒ **假设 A**（拷贝机制本身坏了）。

**该方法的价值**：不新建回读路径、不改默认行为（开关门控），
只用一次运行就能把两个假设分开。

### 10h.39 【关键】记录轨迹：拷贝**确实被记录**，操作数正确

```
[tp5-zsnaprec] rank=0 src_buf=... src_off=406640 dst_buf=... bytes=40960 pre_code=6 q_begin=0
```

**拷贝被记录、`dst` 非空、40960 字节 = 容量×宽度×4 —— 与消费端读取范围完全一致。**
（`pre_code=6` 说明记录位置在 `late_q_begin` 之前 ⇒ 落在 P1 前段。）

**而 `ZSNAP=0`（同内核、同索引、改为直接读 `trefs+406640`）是非零的。**

**⇒ 两者唯一差别是「读快照」vs「直接读源」，而结果不同 ⇒
「拷贝产生的内容」≠「消费时刻直接读取该源所得的内容」⇒ 拷贝时点早于源的写入。**

### 10h.40 门控实验：只把统计量折叠的 z 视图改绑活 `trefs`

新增 `GGML_TP5_LATEBIND_STATS_LIVE_Z=1`：**只改 `q_infos[8]`（`LocalZStats`）这一个视图**，
改为绑 `trefs`；Q 主循环的 `q_infos[3]` 与快照照旧。

**这使得一次运行就能分开两个假设：**

| 结果 | 含义 |
|---|---|
| `sumD` **非零** | 快照内容≠源内容 ⇒ **拷贝时点**是病因（唯一变量就是那一个视图） |
| `sumD` **仍为零** | 与 z 无关 ⇒ 病灶在 `d_b` 的计算本身 |

**为何这个门控比之前的判别更干净**：先前 `ZSNAP=0` 同时改了两个视图（主循环 + 统计量折叠），
无法区分“快照坏”与“d 的计算坏”。**这个门控只动一个视图。**

### 10h.41 【确证】门控实验：只改一个视图，`sumD` 从恒零变为非零

```
sumA = 0.0057 / 0.120 / 0.123     （不变）
sumD = -0.127 / -0.122 / -0.0016  <- 从「恰为零」变为非零 ✅
sumS = 0.037  / 0.033  / 0.107    （不变，布局修复保持）
请求： 171 / 171 / 171            <- 三次全部成功 ✅
```

**只把 `q_infos[8]`（`LocalZStats`）改绑活 `trefs`**，其余一律不动：

⇒ **快照内容 ≠ 源内容，病灶就是拷贝时点。**
（先前 `ZSNAP=0` 同时改了两个视图，无法区分「快照坏」与「`d` 计算坏」——
**这个单变量门控把两者分开了。**）

**而非确定性层也随之消失（171/171/171）。**

### 10h.42 L2 的两层病灶已统一为一个根因

> **`zsnap` 的拷贝时点早于 `trefs` 中 `z_p` 的实际写入。**

一切现象由此解释：

| 现象 | 解释 |
|---|---|
| `d ≡ 0`（确定性） | 快照稳定地拷到全零 |
| 请求序号相关的不稳定 | `z_p` 就绪与拷贝的相对时序随之波动，偶尔拷到数据 |
| `ZSNAP=0` 使 `d` 非零 | 直接读源，拿到消费时刻的真实内容 |
| aggressive 刀锋一直正常 | 它不走快照 |

**而记录轨迹已排除“拷贝未记录”**（操作数正确、`dst` 非空、40960 字节）。

### 10h.44 一处我**未能解开**的矛盾（下一轮从这里开始）

两个实验本该等价，结果不同：

| 实验 | 做法 | `sumD` |
|---|---|---|
| `ZSNAP_LATE=1` | 把**拷贝**挪到 Q 派发之前 | **零** |
| `STATS_LIVE_Z=1` | 同位置，**不拷贝**，直接绑源 | **非零** |

**若二者在同一时点、且源已就绪，结果应当一致。它们不一致 ⇒ 至少有一项违反了我对它的理解。**

已排除的解释：

- “拷贝未被记录” —— 记录轨迹显示操作数正确、`dst` 非空、40960 字节（§10h.39）；
- “用途标记非法” —— 两者均有 `TRANSFER_DST`（§10h.34-c）；
- “尺寸不符” —— 拷贝与分配同为 4×width×4（§10h.26）；
- “屏障缺 `TRANSFER_READ`” —— 已补（§10h.19）。

**尚未验证的假设（具体）：**

1. `ZSNAP_LATE` 的插入点是否真的落在 Q 派发之前（它在精确路的 legacy 分支里，而该分支可能整体不被交付）；
2. `late_zsnap_buf` 是否为**每 bank 共用**，导致两个 `idx` 的拷贝互相覆盖；
3. `z_p` 的生产者到底在哪个命令缓冲（注释说“hoisted”，而 `pre_code=6` 显示它不在 pre 段里）。

**我如实记录：在本子问题上我连续做过多次错误推断，静态阅读已不是合适工具。**
下一轮应当先做的一件事是**打印 pre 段前 10 条命令的类型与各标记点的 `code.size()`**，
而不是继续从注释推断。

### 10h.45 假说：快照拷贝始终早于 `trefs` 的写入；而 `STATS_LIVE_Z` 可能**就是修法**

**推理链（已排除项之上）：**

1. `ZSNAP_LATE`（把拷贝挪到 Q 之前）仍为零 ⇒ **即使更晚的拷贝也得到坏快照**；
2. `STATS_LIVE_Z`（同时点、不拷贝、直接读源）非零且三次全成 ⇒ **消费时刻的 `trefs` 是好数据**；
3. 记录轨迹显示拷贝操作数正确（`src_off=406640`、40960 字节、`dst` 非空）⇒ **不是操作数错**。

**⇒ 唯一自洽的解释：`trefs` 在拷贝执行时还未被写入，而到内核执行时已写好。**

### 10h.46 由此得出的两个结论

**(a) 快照时点错了** —— 它发生在 `z_p` 生产者之前。这与代码注释相符：
注释说屏障“must remain after the terminal z_p producer”，而实测 `pre_code=6`
显示拷贝前面只有 6 项（权重打包 + inject），**生产者不在其中**。

**(b) `STATS_LIVE_Z` 可能不是诊断，而是修法。**

这需要推翻一个我一直默认的假设：**统计量折叠该读快照**。
但恒等式本身是：

```
||R_b + s_b·y||² = a_b + 2·s_b·Σ_p⟨R_b, z_p⟩ + s_b²·c
```

其中参与项是 **`y`**（norm 的规约输出），而 `trefs` 正是 norm 写入该值的地方。
**若折叠读的就是 `trefs`，那它读到的就是 `y` —— 而这正是恒等式需要的。**
快照（`z_p`）则是 **Q 收缩** 需要的另一个张量。

**⇒ 两个消费者需要不同的张量**：Q 收缩要 `z_p`（快照），统计量折叠要 `y`（活 `trefs`）。
而当前两者都绑了快照 —— **这可能就是缺陷的实质。**

### 10h.47 验证方法（未做）

在 `STATS_LIVE_Z=1` 下已经得到 **171/171/171 与 `sumD ≠ 0`**。
**要把它从“诊断”转正为“修法”，需要证明：**

1. 折叠读到的确实是 `y` 而非 `z_p`（可用现有原始 dump 的 `w32..35` 与 `stats_buf` 对比）；
2. `sumS` 与 `sumA` 仍与参照同量级；
3. **与参照臂做内容对比**（不仅是连贯性，还要看输出是否与 `rho=0` 一致）。

**这三项都未做。** 因此本轮**不声称它是修法**。

### 10h.48 对比结果：`STATS_LIVE_Z` 产出与参照**同级**的结果

| | tokens | CJK | bytes |
|---|---|---|---|
| `noRho`（rho=0，legacy） | **171** | 181 | 675 |
| **`liveZ`（rho=1 + `STATS_LIVE_Z`）** | **171** | **179** | 673 |

**两者都跑满 171 token、都连贯、内容仅细微差异**（`liveZ` 多出主题清单几行）。

**⇒ `GGML_TP5_LATEBIND_STATS_LIVE_Z=1` 是一条可用的修法**：
`sumD ≠ 0`、171 token、内容与 `rho=0` 同级。

### 10h.49 它到底修了什么（不夸大）

**它做到：** 让统计量折叠从**活的 `trefs`**（即 norm 规约后的 `y`）读取 `z`，
而不是从那个填不对的 `zsnap`。结果：`d` 非零、稳定、171 token、内容可用。

**它未证：**

1. `ρ` 在数学上正确（只证了与前序路径结果同级）；
2. 长时稳定性（至今只有两次短跑：3 次与 1 次，全过）；
3. 与参照的正式 ABBA（本会话所有数字都是单跑/镜像）。

**而它依赖一个待确认的假设**：统计量折叠要的是 `y`（活性 `trefs`）而非 `z_p`（快照）。
这与恒等式 `||R_b + s_b·y||²` 相符（参与项是 `y`），也与 §10h.46 的推理一致。

### 10h.50 一个需要记录的小失误

我先前有一次把 `liveZ` 读成 `cjk=0` —— **那是读文件抢在了写入之前。**
在后台作业尚未回传时不要从中间产物下结论。

### 10h.51 转正：统计量折叠默认改读活 `trefs`

```cpp
// 新默认：统计量折叠绑活 trefs；GGML_TP5_LATEBIND_STATS_LIVE_Z=0 可退回旧行为
if (want_snapshot && zsnap_enabled && zsnap_buf) return {zsnap_buf, 0, size};
return {trefs.buf, trefs.offset, trefs.size};
```

**转正的理由不是“旧绑定坏”这件事本身，而是语义：**

> 恒等式 `||R_b + s_b·y||²` 的参与项是 **`y`**，而 `trefs` 正是 norm 写入 `y` 之处。
> **统计量折叠要 `y`；Q 收缩要 `z_p`（快照）。两个消费者、两个张量。**

旧代码让两者都读快照 —— 这不是“快照没填好”，而是**绑错了张量**。
这也终于把 `ZSNAP_LATE` 的矛盾解释通了：把拷贝挪后仍然绑错了张量，所以仍为零。

### 10h.52 遗留：快照（`zsnap`）现在只服务 Q 收缩

Q 收缩读快照是正确的（它要 `z_p`，而 norm 会覆写 `trefs`）。
但这也意味着 **LB1 快照的时点问题仍未被独立验证** —— 若快照内容确实早于写入，
Q 收缩也在读坏数据（aggressive 刀锋不走快照，因此看不出）。
**已记录为待验证项，不写成结论。**

### 10h.53 【如实】十次验证：5/10 成功 —— 我的转正基于 n=3 的过度乐观

```
req1 171   req2 27    req3 27    req4 171   req5 171
req6 171   req7 171   req8 27    req9 27    req10 27
```

| 指标 | 结果 |
|---|---|
| token 数 | **5/10 = 171** |
| 内容质量（抽查） | 只有 req1 正常（cjk=179 marks=28）；req5/req10 均 cjk=31（退化） |

**⇒ live-z 转正修好了确定性的 `d ≡ 0`（已由指纹证实），但非确定性层完全没动。**

**这是本会话第七次同形错误：在小样本上下结论。**
我在归档里写了 "NOT YET PROVEN: only 3 + 1 short runs"，**随后仍据它转正并对效果作了过度预期。**
写下保留意见不等于真的保留了 —— **必须跑完足够样本再下结论。**

### 10h.54 两个修复的准确边界

| 修复 | 归属层 | 证据 |
|---|---|---|
| 统计量折叠改读活 `trefs` | **确定性层**（`d ≡ 0`） | 指纹：`sumD` 恒零 → -0.127/-0.122；`sumA/sumS` 不变 |
| （未找到） | **非确定性层**（EOS） | 10 次中 5 次失败，与请求序号相关 |

**转正 live-z 在语义上是正确的**（恒等式 `||R_b + s_b·y||²` 要的是 `y`，`trefs` 正是 norm 写入 `y` 之处；
与 `ZSNAP_LATE` 的矛盾也因此解释通了），**但它只修了一层。**

### 10h.55 L2 的最终裁决（本会话）

**五项修复已落地，其中三项由指纹或对比证实：**

1. `cc` 偏移单一来源（真值 4）
2. **统计区基址改容量口径**（三方里惟中继用实时行数）
3. `mb_scatter` 补 `TRANSFER_READ`
4. 生产端统计量步长（vec2 元素/字换算）
5. **统计量折叠改读活 `trefs`（转正）** —— 修好确定性层

**而 L2 仍不可用**：非确定性层未定位，10 次中 5 次提前 EOS。

**下一轮入口**：早已写下的那一处 —— **`ZSNAP_LATE` 与 `STATS_LIVE_Z` 的矛盾仍未解开**
（前者把拷贝挪后仍为零，后者同位置不拷贝非零）。
**下一轮应先打印 `pre` 段命令轨迹（各标记点 `code.size()` 与命令类型），而不是继续从注释推断。**

### 10h.56 【改判】非确定性层不是竞态，是 `ρ` 的**数值敏感**

十次请求的日志扫描：

```
rho stats timeout: 0
sidecar timeout:   0
handoff failed:    0
validation failed: 0
```

**没有任何协议失败、超时或校验错误。**

**⇒ 提前 EOS 不是“等不到数据”，而是模型基于**错误的 logits** 主动停止。**

**这把非确定性层的性质改判了：**

| 旧判词 | 新判词 |
|---|---|
| 握手竞态（偶尔等不到/读到半写） | **数值敏感**：`ρ` 的误差在部分 token 位置翻转了近 tie 的 argmax，退化文本遇不到停止符 ⇒ 或提前 EOS |

**证据链**：

1. 无任何超时/协议错误（本节）；
2. `ρ` 现在确实由真实数据算出（`sumD ≠ 0`，已指纹证实）；
3. 参照与 `rho=0` 在同一机器上极为稳定（多次全 171）—— 所以不稳定性确实来自 `ρ` 路径；
4. `ρ` 的误差来源：`ρ = 1/√(a + 2s·Σd + s²c)` 中，**跨 rank 的 `d` 折叠现在由 CPU 做，而 `a`/`s` 只取 rank 0** —— 而 §10h.8 的推理（`R` 复制、`z` 分片）虽自洽，**未经实验验证**。

### 10h.57 下一轮的入口（具体，且无需新工具）

**验证“`R` 复制 / `z` 分片”这个假设**：在指纹里**逐 rank 打印 `a`/`d`/`s`**（而不是只打 rank 0）。

- 若各 rank 的 `a` 相同 ⇒ `R` 复制成立 ⇒ `a` 取 rank 0 正确；
- 若各 rank 的 `a` 不同 ⇒ **`a` 也必须跨 rank 求和**，而当前只取了 rank 0 ⇒ **这就是 `ρ` 的误差源**。

**这是一次运行就能判定的事，且完全在现有插桩能力内。**

### 10h.59 【实测】结构假设被证实：`a`/`s` 复制，`d` 分片

```
call=0 rank=0 sumA=0.00570567278 sumS=0.0373229757
call=0 rank=1 sumA=0.00570567278 sumS=0.0373229757   <- 完全相同
call=0 rank=2 sumA=0.00570567278 sumS=0.0373229757
call=1 rank=0 sumA=0.120347753   sumS=0.0334747835
call=1 rank=1 sumA=0.120347753   sumS=0.0334747835
call=1 rank=2 sumA=0.120347753   sumS=0.0334747835
```

**⇒ `a` 与 `s` 在三个 rank 上逐位相同（复制），只有 `d` 分片。**

这**由测量证实了 §10h.8 的推断**（“`R` 复制、`z` 分片”），
**因此 CPU 的折叠（`a`/`s` 取 rank 0、`d` 跨 rank 求和）是正确的。**
**该候选排除。**

### 10h.60 于是剩下的唯一解释：归约本身不确定，而 `rho` 放大了它

已知事实的合取：

1. `rho` 的输入正确（本节）；
2. 无任何协议错误/超时/校验失败（§10h.56）；
3. `rho=0` 在同一机器上多次全 171；
4. `rho=1` 时好时坏。

**⇒ 唯一剩下的解释：底层 TP 归约（`atomic/全归约`）在不同运行间给出**微小不同的 Q**；
参照路径容忍它，而 `rho` 改变了 `u` 的尺度，使模型在近 tie 处翻转决策。**

**判定测试（已启动）**：把参照路径（`rho=0`）连打 5 次，**逐字节对比它的输出**。

- 若参照也变但保持连贯 ⇒ **底层确实不确定**，`rho` 只是把它放大 ⇒ L2 的可用性受限于硬件级归约确定性；
- 若参照逐字节一致 ⇒ **不是底层不确定**，而是 `rho` 路径自身引入了变异，需继续查。

### 10h.72 【已实现】参照 `inverse_rho` 的回读探针

新增 `GGML_TP5_LATEBIND_RHO_PROBE=1`：在 relay 的 ρ 发布点，
直接 `vkMapMemory` 读 `c.cached_plans[0].late_rho_mem[0]` 的前 8 个 float ——
**这就是基线 LO 消费者实际使用的参照 `inverse_rho`**（norm 内核写的），
与恒等式的 `ss` 同一次运行并排打印。

**它一次回答两个问题：**

1. **LO 该用哪个 ρ** —— 参照值是 ~1 还是别的量级；
2. **恒等式缺了哪个因子** —— `ss` 与参照的 `ρ⁻²` 的比值是常数还是随 token 变化。

探针是**一次性、冷路径、门控**的（直接映射 device 内存，不得在性能测量中开启）。

### 10h.70 【实测】norm 路径的 width/epsilon 与恒等式的尺度差

```
[tp5-normscale] rank=0 width=2560 epsilon=9.99999997e-07
[tp5-l2ss]      call=0 n=8 ss_min=0.000713209098 ss_max=0.0065356717
```

| | 量 | 除以 width 后 |
|---|---|---|
| norm 路径 | `fold`（其输入近单位 RMS） | **≈ 1** |
| LB2 恒等式 | `ss`（全宽求和，实测 0.0007–0.0065） | **≈ 3.3e-6** |

**相差约 3×10⁵。** 且 `epsilon=1e-6` 在 norm 的尺度上是极小量，在恒等式的尺度上却**大于整个信号** —— 这就是照搬公式失败的原因（§10h.68）。

### 10h.71 【结构性发现】参照的 ρ 一直就在 `late_rho_buf` 里

基线的 **LO 消费者用的就是 norm 内核写的 `inverse_rho`**（`lo = scale·q`，`late_rho_buf` 由 `TP5_RESUME_NORM` 产出）。所以：

- **"LO 折叠该用哪个 ρ"的参照答案已存在** —— `1/RMS(norm 的输入)`；
- LB2 恒等式给的是**校正后向量**的范数，与 norm 给的 **y 本身**的范数是**不同的量**；
- 我的 `ss` 比两者都小几个数量级 ⇒ **恒等式的输入张量（R、y、c）与两条参照路径都不同尺度**。

**下一轮必须先做的一件事**：把 `late_rho_buf` 回读出来（一次 `vkCmdCopyBuffer` 到 host-visible + 一次打印），
与恒等式的 `ss` 并排 —— 这能直接判定"该用哪个 ρ"以及"恒等式缺了哪个因子"。
**在尺度对齐之前，任何 ρ 公式的改动都只是换一种错法。**

### 10h.68 【已测并撤回】`/width + epsilon` 使指标变差

把 LO_U 的 ρ 改为与 norm 路径同式的 `inversesqrt(ss / width + epsilon)`（并给 push constant 补 `width`/`epsilon`）：

| | 完成 171 | 连贯 |
|---|---|---|
| 改动前 | 5/10 | ~1/10 |
| **改动后** | **10/10** | **0/10** |

**提前 EOS 消失了，但输出变成一致地退化。** 指标变差 ⇒ **撤回**（公式已还原，理由就地写在着色器注释里）。

### 10h.69 【新发现】尺度失配 —— 这才是真正的开口

对照两条路径的量级：

| | 量 | 实测 | 除以 `width=2560` |
|---|---|---|---|
| norm 路径 | `fold` = Σ values² | ≈ `width`（值已归一化） | **≈ 1** |
| LB2 恒等式 | `ss` = ‖R + s·y‖² | **0.0085**（min 1.15e-4, max 584） | **≈ 3.3e-6** |

**两者相差约 3×10⁵。** 因此：

- `epsilon`（norm 路径里相对 1 是极小量）**对我的量级毫无约束作用**；
- `1/sqrt(3.3e-6 + ε)` 仍达数百 ⇒ ρ 被又放大一档 ⇒ **输出一致退化**。

**⇒ 真正待答的问题不是"要不要加 epsilon"，而是"为什么 `ss` 与 `fold` 不在同一尺度上"。**

**候选解释（未验证）：**

1. `R`/`y` 在此处的量级本就远小于 norm 路径的 `values`（即参与恒等式的张量未经同一归一化）；
2. 或 `ss` 缺少某个因子（如 `width` 的幂）；
3. 或 `c`（`lb2_cc`）与 `a`/`d` 的缩放来自不同的归约口径。

**这是下一轮的第一个问题**，而且它是可判定的：在指纹里同时打印
`fold`（norm 路径的同 token 值）与 `ss`，直接看两者比例常量。

### 10h.66 【已否决】`ss` 不过零：guard_hits = 0 / 400

```
[tp5-l2ss] call=0 n=8 ss_min=0.000713209098 ss_max=0.0065356717 guard_hits=0
...
全局： guard_hits=0（400/400）   ss_min=1.15e-4   ss_max=584.6
```

**`ss` 从不 ≤0** ⇒ "ρ=0 ⇒ u=0"的机制不成立，**该假说否决**。

### 10h.67 【真缺陷】我丢掉了 ρ 的**逆 RMS 正规化**

基线（`16e28b8ac`）里 norm 路径一直在算：

```glsl
const float scale = inversesqrt(fold / float(p.width) + p.epsilon);
```

**两处关键：**
1. **除以 `p.width`** —— 使它是**均值**而非全宽求和；
2. **加 `p.epsilon`** —— 给倒数一个下界，防止它发散。

**而我改写后的 LB2 合成两项都没有**（LO_U 的 push constant 甚至没有 `width`/`epsilon` 这两个字段）：

```glsl
const float rho0 = ss0 > 0.0 ? 1.0 / sqrt(ss0) : 0.0;   // 无 /width、无 epsilon
```

**后果（由实测数据算出）：**

| `ss` | 我的 `ρ = 1/√ss` | 参照的 `1/√(ss/width+ε)` |
|---|---|---|
| 1.15e-4（最小） | **93** | 约 1–4（有界） |
| 584.6（最大） | 0.041 | 同上 |

**⇒ 我的 ρ 跨越 3 个数量级，而参照被 epsilon 夹在窄带内。**
`u = 0.25·Σ ρ_b·Q_b`，所以这个 25 倍的放大把普通归约噪声放大成足以翻转 argmax 的扰动 ——
**且无任何协议错误**，与观测完全一致。

**修法**：LO_U 的 push constant 补 `width`/`epsilon`（28 → 36 字节），
着色器改为与 norm 路径同式的 `inversesqrt(ss / p.width + p.epsilon)`。

### 10h.64 【关键假说】`ss` 离零很近，可能被归约噪声推过零

从已测数值手算恒等式：

```
a     ~=  0.0057              (rank 间相同，已实测)
2*s*d ~=  2*0.037*(-0.127) = -0.0094
s^2*c ~=  0.037^2 * 88.79  = +0.0122
------------------------------------------------
ss    ~=  0.0085              <- 离零很近
```

**而 LO_U 着色器的守卫是 `rho = ss > 0 ? 1/sqrt(ss) : 0`。**

**⇒ 若 `ss` 随归约噪声跳到 ≤0，则 `rho = 0` ⇒ `u = 0` ⇒ 输出退化 ⇒ 提前 EOS。**

**这能解释全部观测，而且不需要引入"硬件级不可修"的说法：**

| 观测 | 解释 |
|---|---|
| 无任何协议错误/超时 | 确实是数值路径，不是握手 |
| 时好时坏（~50%） | `ss` 在零附近摆动，跨越零线 |
| 参照稳定 | 参照不用 `rho`，不受 `ss` 影响 |
| **`d` 修复后失败率不变** | `d` 把 `ss` 从 0.018 推到 0.0085 —— **更靠近零** |

**最后一项尤其重要**：修正 `d ≡ 0` **把 `ss` 推向了零**，因此失败率未降。

### 10h.65 若假说成立，修法是什么

**不是"加个下限"那么简单** —— `ss ≤ 0` 在数学上应当不可能（它是平方范数），
出现非正说明 `a + 2sd + s²c` 这一表述**抵消严重**（catastrophic cancellation），
而它本可以写成 `||R + s·y||²` 直接算。

**候选修法（待验证）**：

1. 着色器里改用 `||R_b + s_b·y||²` 的**直接形式**（需把 `R` 与 `y` 同时拿进该内核）；
2. 或把 `ss` 改为**高精度累加**，降低抵消；
3. 或（最小改动）守卫改用平坦下限 —— **但这引入偏差，只可作临时措辞**。

**探针正在运行（`tp5-l2ss` 打 `ss_min/ss_max/guard_hits`）。**

### 10h.61 【决定性】参照路径自身也逐字节变化

`rho=0`（参照）连打五次，同 prompt、`temp=0 seed=42`：

```
req1 171 tokens   req2 171 tokens   req3 171 tokens   req4 171 tokens   req5 171 tokens
ref1 != ref2  ref1 != ref3  ref1 != ref4  ref1 != ref5     <- 五次输出互不相同
```

**⇒ 底层 TP 归约本身不确定。** 参照路径**容忍**它（全 171、全部连贯）；
**`rho` 改变了 `u` 的尺度，使模型在近 tie 处随这点差异翻转决策** —— 十次中五次提前 EOS。

这与已记录的 G7 发现一致：**自然文本在 TP5 跨 rank argmax 归约下非字节稳定。**

### 10h.62 L2 的最终裁决（本会话收束）

| 层 | 结论 | 证据 |
|---|---|---|
| **确定性**（`d ≡ 0`） | **已定位并修复**（统计量折叠绑错了张量） | 单变量门控：`sumD` 恒零 → -0.127；原子与 `s` 逐 rank 实测相同 |
| **非确定性**（15–50% 提前 EOS） | **不是 L2 的代码缺陷，是硬件级归约非确定性被 `ρ` 放大** | 参照自身逐字节变化；`rho=1` 无任何协议错误 |

**它同时确定了 L2 的可行性边界：**

```
单纯修代码无法使 L2 稳定 —— 非确定性来自归约，而 ρ 只是把它放大。
要使其健壮，需要：
  (a) 确定性归约（架构级代价），或
  (b) 一种不改变 u 尺度的 ρ 表述方式。
```

**这就是本会话对 L2 的终态结论。** 它不是一个失败，
而是一个**有证据支撑的可行性判定**：确定性层已被修好且可验证，非确定性层超出“改琳代码”的范围。

### 10h.63 本轮修好的，与留下的

**已修好（八项根因 + 三项顺带）：** 孔径族、激活量化器索引、`cc` 偏移、
统计区基址（惟中继用实时行数）、跨 rank `d` 求和、`mb_scatter` 的 `TRANSFER_READ`、
生产端统计量步长、**统计量折叠绑错的张量**；另加 `mmvq` 守卫、两处 push layout、
`late_stats_buf` 越界、Q8DOT 的 LB2 管网、`rho=` 账本字段。

**留下：** L2 的非确定性层（已判为硬件级，非代码级）；
以及一处未解矛盾（`ZSNAP_LATE` 仍为零而 `STATS_LIVE_Z` 非零）——
现已部分解释：后者绑对了张量，前者仍然绑错。

### 10h.58 一次自找的无效运行：忘了诊断门控

我加了“逐 rank 打印 `a`/`s`”的插桩，然后跑了 `/tmp/l2rank.sh` —— **它由 `l2long.sh` 派生，不带 `GGML_TP5_RELAY_DEBUG`。**
整个插桩块被门控关掉，白跑一轮。

**这是同一类错误的第二次：**

1. 先前：指纹门控写成 `call < 8`，而那 8 次恰好全落在**能工作**的请求 1 里（§10h.11-b）；
2. 本次：派生脚本时漏了一个环境变量，整轮无输出。

**模式相同：** 我给插桩加了门控（为了不污染热路径 —— 这是对的），
**但没有建立“门控确实开了”的校验**。

**今后的做法**：任何带门控的插桩，运行时必须**先确认它输出了至少一行**再继续测量；
否则视为无效运行。这与“仪器先校准再用”（§10b.7-d）是同一条原则。

### 10h.43 正确修法（尚未实现；不是“用门控”）

**门控 `GGML_TP5_LATEBIND_STATS_LIVE_Z` 是诊断，不是修复**：它让统计量折叠绕过快照，
但 Q 主循环仍在读快照（其正确性依赖另一个假设）。

**正解二选一：**

1. **把快照移动到 `z_p` 生产者完成之后、norm 覆写之前** —— 需先确认 `z_p` 的写入者；
2. **或给快照加与生产者同代的 generation 门控**，使拷贝必定晚于写入。

**验证方式已就绪**：把门控关掉（回归默认），看 `sumD` 是否仍为零 —— 若修法正确则应非零。

### 10h.34-c 用途标记对比的结果：两者都有 `TRANSFER_DST`，不是病因

```
wire_buf（可用）  TRANSFER_SRC | TRANSFER_DST | STORAGE_BUFFER
latexsnap（存疑） STORAGE_BUFFER | TRANSFER_DST
```

**两者都能作拷贝目标。此路排除。**

### 10h.34-d 我在这一处的推断命中率很差 —— 如实记录

对本缺陷我做了**三次连续的错误推断**：

1. “假设 A（拷贝坏了）已排除” —— 错（§10h.34-b）；
2. 固为“早期时点未就绪” —— 但晚期快照也为零，故未成立；
3. 把“生产端步长”当成充分原因 —— 部分正确（`s` 确实被修好，`sumS` 0.0014→0.0373），
   **但 `d` 仍为零，病情未解。**

**而一个非常精确、且始终未被推翻的事实是：**

> 同一内核、同一描述符集、**同一缓冲**（精确路下 `q_infos[3]` 与 `q_infos[8]` 都是 `zsnap`）：
> `residual` 给出 `a_b ≠ 0`，而 `z` 给出 `d_b ≡ 0`。
> 而 `ZSNAP=0`（同一个内核、同一套索引、改为绑 `trefs`）给出 `d_b ≠ 0`。

**⇒ `late_zsnap_buf` 里就是零。** 而填充它的那次 `vkCmdCopyBuffer` 与
已验证可用的 `:3553` 拷贝**结构相同、用途标记均合法**。

**剩余唯一的差异面**：那次拷贝命令**是否真的被记录进一个会被提交的命令缓冲**，
以及它落在 P1/P2 的哪一段。**这需要打印命令记录轨迹，不是静态阅读能定的。**

### 10h.34-b 【纠正 10h.35】"假设 A 已排除"是**错的**

我曾写："晚期快照下拷贝路径相同、机制可用 ⇒ 排除假设 A（拷贝坏了）"。**该推理不成立：**

已核实 `zsnap_late` 的插入点（`:4489`）位于**精确路的 legacy 分支**，
在 `emit(pre, …)`（含 norm）**之后**、Q 派发之前 —— 即 `trefs` 此时应已被 norm 覆写为 `y`。

**因此若拷贝真的执行，`d = ⟨R, y⟩` 应当非零 —— 而它是零。**

**⇒ `vkCmdCopyBuffer` 写 `late_zsnap_buf` 这一步本身没生效。**

**而我犯的错误是：** `ZSNAP=0` 那一次只证明了"**直接读 `trefs`** 可行"，
**并不能推出"拼接拷贝可行"**。两者是不同的机制（一个是描述符直接读，一个是拷贝）。
**我把它们混为一谈了。**

**修正后的假设表：**

| 假设 | 裁决 |
|---|---|
| A. **拷贝写 `zsnap` 没生效** | **重新成立（且为首要嫌犯）** |
| B. 早期时点 z_p 未就绪 | 未被独立证实（10h.34 的"四次成功"可能是别的原因） |

**下一步（具体）**：验证 `late_zsnap_buf` 的 `VkBufferCopy` 是否真的执行 ——
与已有的、**确实可用**的那处同缓冲拷贝（`:3553` `trefs -> wire_buf`）逐字段对比：
用途标记、offset、尺寸、以及拷贝前后的屏障阶段。

### 10h.34 【决定性】晚期快照：`d` 仍为零（拷贝机制无问题），**但四次全部成功**

```
sumA = 0.0043 / 0.116 / 0.119    （不变）
sumD = 0                        <- 仍为零 ⇒ 拷贝机制本身可用
sumS = 0.0014 / 0.0043 / 0.0044  （不变）
```

| 请求 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 早期快照（默认） | 171 | 171 | **27** | **27** |
| **晚期快照（`ZSNAP_LATE=1`）** | **171** | **171** | **171** | **171** |

### 10h.35 两个假设的裁决与两个结论

| 假设 | 裁决 |
|---|---|
| **A. 拷贝机制本身坏了** | **排除** — 晚期快照下拷贝路径相同、机制可用 |
| **B. 拷贝时 `z_p` 尚未就绪** | **成立** — 把拷贝挪后，行为改变 |

**结论一（确定性层）**：早期时点 `z_p` 未就绪 ⇒ 快照为零 ⇒ `d ≡ 0`。
这与 §10h.18 的推理（aggressive 路读活 `trefs` 且正常）完全自洽。

**结论二（非确定性层，意外收获）**：

> **把拷贝挪后 ⇒ 四次全部成功（171/171/171/171）。**
> **⇒ 那个“请求序号相关的不稳定”，就是“快照时点相对于生产者”的竞态。**

**两层病灶由此统一到一个根因**：快照的**时点**与 `z_p` 生产者的**完成**之间没有保证。
早期时点：拷到零（确定性地 `d ≡ 0`）；偶尔拷到了、偶尔没拷到 ⇒ 表现为不稳定。

### 10h.36 正确的修法（不是“用晚期快照”)

晚期快照**内容必然是错的**（拷到的是 norm 覆写后的 y，而非 `z_p`），
它只用于判别。**它四次全部成功这一点反而揭示了病灶的形状：**
晚期时点处于一个“所有写入都已完成”的位置，所以拷贝总是得到**一致**的快照内容。

**因此正确修法有两条，需分别验证**：

1. **把快照挪到“`z_p` 生产者完成之后、norm 覆写之前”** ——
   需要找到那个精确的屏障点（`z_p` 的生产者是谁要查）；
2. **或给快照加“等待 `z_p` 就绪”的门控**（如与生产者同代的 generation 检查）。

**两者都未实现。** 下一轮从这里开始。

### 10h.37 【根因】原始 12 字 dump 揭示：不是内容，是**步长**

```
w0..11 = 0.000713209  0  0.000713209  0  0.000713209  0  0.000713209  0  0.000713209  0  0.000713209  0
          ↑a     ↑d≡0   ↑a      ↑d≡0    ...
w32..35 = 0.0116  0.0070  0.1100  0.0091     ← s 暂存（位于 +32）
```

**数据就在正确的基础偏移上，但布局错了：只有奇偶交错，缺 `+8..11` 的 `s_b`。**

| | 生产端实际 | 消费端期望 |
|---|---|---|
| `a`/`d` | float `2b` / `2b+1`（**缺 `token*12` 项**） | `12t+2b` / `12t+2b+1` |
| `s_b` | 独立的 `stats_buf + 32 + b`（**分离区域**） | `12t+8+b` |

生产者：`stats_out[token * p.streams + stream]`（vec2 数组 ⇒ 字地址 = 元素×2 = `2·token·4 + 2b`）
—— **`p.streams` 给的是字节/元素步长，不是 12 字的字步长**。

**这与消费者的 12 字视图结合起来，得出两个独立后果：**

1. `t≥1` 时 `a`/`d` 落在错位置（我的 dump 因 `rows=2` 且首 token 对齐而恰好看到了 `t=0`）；
2. `s` **根本不在 `12t+8+b`**，而我在 CPU 侧对 `2b+1` 写入跨 rank 折叠的 `d`，
   **写在了 `a` 的位置上** —— 这直接损坏了 `a`。

**已在两处（Q8DOT 与精确）改为与 12 字契约一致的索引**：
`stats_out[token*(streams*3)/2 + stream]` 与 `stats_out[token*(streams*3)/2 + streams + stream]`。

### 10h.38 本轮方法论的又一次印证

**标量指纹（`sumA`/`sumD`/`sumS`）把“布局错位”误报成了“数据缺失”。**
两个量（“全零”与“读错位置”）在标量下不可区分，在**原始数组 dump** 下一眼可辨。

**教训：累加型指纹对偏移类缺陷是盲的。** 该直接 dump 原始字。

### 10h.30 判别实验：用现成开关绕过"需要新工具"的瓶颈

我先前把下一步定义为"需要 GPU 侧回读工具"。**其实不需要：`GGML_TP5_LATEBIND_ZSNAP` 本身就是那个开关。**

```
ZSNAP=1（默认）  exact 路读 zsnap 快照   ← 失败臂
ZSNAP=0          exact 路读活的 trefs    ← 与 aggressive 刀锋同样的读法，而后者工作正常
```

**⇒ 若 `exact + rho=1 + legacy + ZSNAP=0` 稳定，则快照就是元凶；若仍失败，`z` 就不是病因。**

这比新建回读路径**更弱但更快**：它不能告诉我"快照里是什么"，但它能直接判定"快照是不是那个坏的部分"。
**先用弱手段排除，再上强手段** —— 这是本会话反复验证过的顺序。

### 10h.26 zsnap 的尺寸用了两个变量 —— 已统一

```cpp
// 分配（:3801）   max_rows (= VK_TP5_DIRECT_COLUMN_TILE)
const VkDeviceSize zsnap_bytes = VkDeviceSize(max_rows) * late.width * sizeof(float);

// 拷贝（:4369，修前） late.capacity_rows —— 另一个变量
const VkDeviceSize zsnap_bytes = VkDeviceSize(late.capacity_rows ? late.capacity_rows : 1u)
                                 * late.width * sizeof(float);
```

**同一件事（快照尺寸）用了两个不同变量。** 若二者在某一时刻不同：

- `late.capacity_rows > max_rows` ⇒ **拷贝越出分配区**；
- `late.capacity_rows < max_rows` ⇒ **快照尾部未写**，而消费者读 `active_rows` 行 ⇒ **读到零**。

后者正是本轮的观测（`d ≡ 0` 而 `a` 活的）。**已把拷贝改为与分配同源的 `VK_TP5_DIRECT_COLUMN_TILE`。**

这是本会话**第五次**同形错误（两个同义变量/字面量各自维护），也再次印证：
**单一来源不只是风格，它是正确性。**

### 10h.20 屏障修复的结果：零结果，但保留

修正后重跑：`sumD=0` **不变**，r1 仍为 171。

**⇒ 屏障不是病因。** 但它是**教科书级的正确修复**（`vkCmdCopyBuffer` 读 `trefs` 确实需要
`TRANSFER_READ` 可见性），**保留**——它消除了一类潜在的未定义行为，只是不是当前症状的来源。

**因此 `zv ≡ 0` 是一个真事实，而不是同步误读的伪影。** 即：精确路的 `z` 快照里**就是零**。

### 10h.21 剩余可判定的点（交给下一轮，需 GPU 侧回读）

| # | 待验 | 为何静态分析到此为止 |
|---|---|---|
| 1 | `trefs[i].offset` 是否真的指向 `z_p` 起点 | 需读运行时值，或回读 `late_zsnap_buf` 对比 `trefs` |
| 2 | `trefs` 在快照时刻是否已被生产者写入 | 生产者在更早的阶段，需回读对比 |
| 3 | `late_zsnap_buf` 的分配尺寸（`:3804`）是否 ≥ `zsnap_bytes` | 需读分配表达式的上下文 |

**而一个已知的强线索**：**aggressive 刀锋不走快照、直接读活的 `trefs`，且工作正常** ——
所以 `trefs` 里**有**正确的 `z`。病灶仅在于**把它拷进快照的那一步**。

### 10h.19 【已修】zsnap 拷贝前的屏障缺了 TRANSFER_READ
    VK_ACCESS_SHADER_READ_BIT};                                  // dst ← 缺 TRANSFER_READ
tp5_cmd_barrier(cmd, COMPUTE|TRANSFER, COMPUTE, ...);            // dst stage ← 缺 TRANSFER
// 紧随其后的 LB1 z_p 快照：
vkCmdCopyBuffer(cmd, trefs[i].buf, plan.late_zsnap_buf[i], 1, &zsnap_cp);
```

**`vkCmdCopyBuffer` 是 TRANSFER 操作，它对源（`trefs`）的读取需要 `TRANSFER_READ` 可见性，
而屏障的 dstAccessMask 只声明了 `SHADER_READ`；dstStageMask 也缺 `TRANSFER`。**

⇒ 拷贝可以合法地观察到**陈旧或零**的源字节。

**这与症状精确吻合**：统计量折叠从**活的** residual 视图算出 `a_b = <R_b,R_b> ≠ 0`，
而从快照算出 `d_b = <R_b,z_p> ≡ 0` —— **二者只能同时成立于 `z` 读成零的时候**。
而主循环读的是同一份快照，所以它在 rank 1–4 上同样退化为零，只靠 rank 0 的 `residual` 支撑。

**修正**：dstAccessMask 加 `VK_ACCESS_TRANSFER_READ_BIT`，dstStageMask 加 `VK_ACCESS_TRANSFER_BIT`。

**为何 aggressive 刀锋不受影响**：它根本不走快照（`!plan.late_q8_fast` 才做快照），
直接读活的 `trefs`（`act_q8_infos[2]`）。**这解释了两条刀锋表现差异的根源。**

### 10h.11-b 插桩证据（400 次中继调用）

```
call=0  bank=0 late_count=1280 stats_off=529472 cc_off=529664 rows=2 sumA=0 sumD=0 sumS=0 sumCC=88.79
call=2  bank=1 late_count=1280 stats_off=529472 cc_off=529664 rows=2 sumA=0 sumD=0 sumS=0 sumCC=0.26
...（400 次全部 sumA=sumD=sumS=0）
```

**已确认的部分：**

- `bank` **正确地在 0/1 间轮转** —— 邮箱交替正常
- `stats_off` **恒定** —— 它是 bank 内偏移，`bank` 选缓冲，所以读法正确
- `cc_off = stats_off + 192` ✅ —— **两个偏移相差正好 12×4×4 字节，与我改后的 `cc_offset` 一致**
- 发布器的描述符目标已核实为 `bcast_buf[b]` + `payload_offset(L,false) + late_count*4` ✅ 与中继同源
- `sumCC` 非零且随 stage 变化 ✅ —— **CPU 侧的 cc 归约是活的**

**仍未闭合的矛盾（不写成结论）：**

**若 stats 在**所有** 400 次调用中均为零，则 `ρ` 在**请求 1 也是零** —— 而那一次却正常完成（171 token）。**
两者不能同时成立，因此必然是**以下二选一**：

1. **指纹读错了地址**（记录的 `late_count` 与中继的不同 ⇒ 发布器的 dst 偏到别处）——
   但两者应都源于同一条链，需在记录期同时打两份才能定案；
2. **`u` 对请求 1 的成败不关键，但有一个随请求序号累积的量才是关键** ——
   与 10h.9 的形态一致。

**区分方法（未做）**：在**记录期**同时打印发布器的 `dst` 偏移与 `late_count`，与中继的值并排比较；
若不等，病灶就是偏移不对称（假设 1）。若相等，则假设 2 成立，需转向 LO_U 侧的插桩。

### 10h.12 已排除的候选（本轮穷举）

| 候选 | 反证 |
|---|---|
| `lb2_cc[]` 跨调用累积 | 局部数组且 `= {}` 零初始化 |
| `cc` 发布不是每 cycle | 在 relay 主路径上无条件执行 |
| 跨 rank 折叠写回的 `d` 污染下一轮 | 下一轮的 GPU 发布器覆盖整个 stats 区 |
| `s_b` 未到达 bank | 精确路 `s_out` 写入 `late_stats_buf + max_rows*4*2*4`，即每 token 12 字中的第 8..11 字 |
| `a_b` 只取 rank 0 是错的 | **错。** `R_b` 复制、`z_p` 分片，故 `a` 取 rank 0 正确而 `d` 必须求和（§10h.8） |
| 预武装未发生 ⇒ 控制字未重置 | `arm_next_epoch = epoch+1` 非零，每 cycle 重置（`:6678`/`:7550`） |
| 容量守卫放行越界行数 | 修后零结果，说明它从未触发 |

### 10h.13 下一轮的入口（需要插桩，不是盲改）

**形态是「首次成功、之后全失败」** ⇒ 请求 1 记录计划、请求 2+ 重放它；
病灶在「**只在首次执行中正确、在重放中不成立**」的那类状态里。

**建议的插桩点**（逐 token 导出请求 1 与请求 2 的对比）：

1. 每个 rank 的 `stats_offset` 区域 12 字（`a_b`/`d_b`/`s_b`）
2. CPU 发布的 `cc`
3. LO_U 着色器算出的 `ss_b` 与 `ρ_b`

**若 1 在请求 2 已经错** → 病灶在 GPU 发布器或其上游；
**若 1 对、２ 错** → 在 CPU 的 cc 归约；
**若 1、2 对而 3 错** → 在 LO_U 的读取或合成。

**本轮到此停止：** 继续未定位的改动会威胁已验收的 L1 成果（六臂 61–66、账本全同、输出逐字节相同）。

### 10h.10 同一段代码里发现的不自洽（已修，待验证）

```cpp
float lb2_cc[16u] = {};                                  // 按 16 预留
if (lb2_cc_rows == 0 || lb2_cc_rows > 16u) => fail       // 守卫放行到 16
// 但 late_stats_buf 实际只按 max_rows*4*3*sizeof(float) = 192 字节分配  <- 仅 4 个 token
// 而 stats 布局是每 token 12 字 => 容量就是 4
```

**容量是 4，守卫却放行到 16** —— `lb2_cc_rows` 落在 5..16 时越界读写。
`lb2_cc` 与守卫现已一并从 `VK_TP5_DIRECT_COLUMN_TILE` 导出。

## 10g. 与参照持平之后的下一步：开销在哪里

**现状**：aggressive 60–66 vs 参照 61–63。对称。

**两边做的是同一份数学**，差异在：

| | 参照 (`latebind off`) | aggressive |
|---|---|---|
| ACT_Q8（激活量化） | — | **每 cycle 一个派发 + 一个可见性屏障** |
| Q 收缩 | F32（含在常规链路里） | Q8DOT（**更便宜** — 这是 aggressive 快于 exact 的原因） |
| sidecar 发布 + 中继扇出 | — | 有 |
| LO 消费者 | 含在常规链路里 | **单 workgroup 派发** |
| 重叠收益 | — | Q 落进 norm 的 Y 等待窗口 |

**净额为零 ⇒ 额外开销 ≈ 重叠收益。要超越参照，必须增加收益或减少开销。**

### 10g.1 减少开销的候选

1. **ACT_Q8 折入 Q8DOT** —— 当前是两个派发 + 一个 `act_ready` 屏障。
   叠化后只需一个派发（Q8DOT 自己量化），省一个屏障 + 一次全量 activation 回写/回读。
   **代价**：Q8DOT 的并行几何（16 行/WG，16 lane/行）与 ACT_Q8 的（每 subgroup 一个 32-元素块）
   不同，合并需重新设计归约 —— **不是一行改动**。
2. **L2（ρ）** —— §10c.4 的“一次发布、无第二门铃”。**这就是 §10g.2 要量的东西。**

### 10g.2 纪律：先量 L2 的代价，再决定是否为它做解耦

历史记录的 LB2 回退（+14.2%）**是在已损坏的 device-mapped 孔径上量的**；而该孔径已修（Host Push），
LB2 自身的寻址缺陷也已修（§10b）。**旧数字不能引用。**

已写好单因素对照（镜像交错的 `E0`/`E1`）：同刀锋（exact）、同调度（legacy，LB2 唯一可跑的调度）、
**唯一变量 `GGML_TP5_LATEBIND_RHO`**。先量，再决定解耦是否值得做。

## 10e. 【根因】aggressive 数值变坏的真因：激活量化器的**字索引错位**

### 10e.1 缺陷

`tp5_hc_latebind.comp` 的 `TP5_LATE_ACT_Q8`（激活量化器）尾部：

```glsl
const uint g0 = lane & ~3u;                      // 组基址：lane 0→0, 4→4, 8→8 …
const int q0 = subgroupShuffle(q, g0);           // 从该组取回 4 个元素
... q1/q2/q3 同理 ...
const uint word = (uint(q0)&0xff) | (uint(q1)&0xff)<<8 |
                  (uint(q2)&0xff)<<16 | (uint(q3)&0xff)<<24;
if (lane < 8u) act_q8[out_block].qs_words[lane] = word;   // ← 用 lane 当代字号
```

**lane 与它携带的元素组不是同一编号。** lane L 的字是从 `g0 = L & ~3` 重建的，
即它携带的是**字号为 `L / 4`** 的那组。写成 `qs_words[L]` 后：

| 写入位置 | 实际内容 | 应为 |
|---|---|---|
| `qs_words[0]` | 元素 0–3 ✅ | 0–3 ✅ |
| `qs_words[1..3]` | 元素 **0–3**（重复） ❌ | 4–7 / 8–11 / 12–15 |
| `qs_words[4..7]` | 元素 **4–7**（重复） ❌ | 16–19 / 20–23 / 24–27 / 28–31 |

**32 个量化值里只有 8 个存活，且每个被重复四次。** Q8DOT 于是消费一份残破的激活。

### 10e.2 与握手缺陷的关系：两个**独立**的根因

这与 §10b 的孔径族缺陷**互不相干**：

- 孔径族缺陷 → **握手错乱**（Q 只发布一次 / 读半写数据）—— 已在 §10b.7-c 修复
- 本缺陷 → **Q 值本身错**（激活被截断重复）—— 与握手无关

**两者都修好之前，aggressive 刀锋无论哪一方单独修复都不会变好**，这正是它长期表现为
"怎么修都还是坏的"的原因：两个独立缺陷互相掩盖。

### 10e.3 修正

```glsl
if ((lane & 3u) == 0u) {
    act_q8[out_block].qs_words[lane >> 2u] = word;
}
```

只有每组第 0 条 lane（0,4,8,…,28）写入，写到 `qs_words[lane >> 2]` —— 字 0..7
各得到元素 4j..4j+3，与 §7.9 布局契约、`PACK_WEIGHTS`、以及 Q8DOT 消费端完全一致。

## 10f. 修复验证：aggressive 刀锋**恢复连贯且提速近十倍**

`qs_words[lane]` → `qs_words[lane >> 2]` 修正后（同二进制、同 prompt、同 seed）：

| 臂 | 修复前 | 修复后 |
|---|---|---|
| aggressive r1 | 5 tok/s, `pred_n=11`, `<think>` 循环 | **38 tok/s, `pred_n=161`, COHERENT (cjk=215)** |
| aggressive r2 | — | **51 tok/s, `pred_n=171`, COHERENT (cjk=187)** |

实际输出（连贯中文）：

> 健康减重的核心在于制造合理的热量缺口，而非盲目追求极速掉秤。
> 饮食方面建议遵循高蛋白、低升糖指数及高纤维的原则，确保营养均衡…
> 每周运动应包含至少150分钟的中等强度有氧运动，如快走或慢跑。

**一行索引错误把刀锋从 5 tok/s 的标点/`<think>` 循环变成 38–51 tok/s 的连贯中文。**

### 10f.1 本会话修复的两个**独立**根因，共存才是它长期"怎么修都坏"的原因

| # | 缺陷 | 症状 | 地位 |
|---|---|---|---|
| A | 孔径族：武装重置/行数发布/堡垒检查硬编码 `bcast_host` | 握手错乱：Q 只发布一次、读半写数据 | §10b.7-c |
| B | 激活量化器 `qs_words[lane]` 用 lane 当字号 | Q 值本身错：32 值只剩 8 个且重复四次 | §10e |

**只修 A：握手对了，但 Q 值仍错 → 仍然退化。只修 B：Q 值对了，但握手仍错 → 仍然退化。**
两者互相掩盖，这就是为何该组合反复表现为"看起来无可救药"。

### 10f.2 但刀锋仍未**胜**参照（持平，非超越）

> **本节已被 §10f.2-a 更新**：修正量化器索引后，`aggressive + overlap=legacy` 达到
> **61 / 64 tok/s**，与参照 61–63 **持平**（且输出连贯，cjk=187）。
> 早先的 38–51 是 `overlap=overlap`（重叠）下的数字，而重叠在本实机上**是负收益**。

所以本轮的净结论是：
- **到达持平**（从「不可用」到「与参照同级」）
- **未产生净收益** —— 没有配置超过参照

真实收益的两个候选方向：
1. **L1 的重新定位**：实测表明重叠（Q 后于 norm）是负的；
   正确的方向是让 Q **更早**完成而非与 norm 重叠（§10f.2-b）
2. **L1 与 L2 解耦**（§10f.3），使 ρ 能上快速刀锋

## 10d. 同二进制同工作量的三刀对照（本会话首批可信数值）

同一二进制、同 prompt、`temp=0 seed=42 stream=false`、`n_predict=171`、
`GGML_TP5_LATE_TRANSPORT=gtt-push`、ZSNAP=1：

| 叶片 | 臂 | tok/s | `pred_n` | 连贯性 |
|---|---|---|---|---|
| **reference** | `latebind off` | **61–63** | 160 | ✅ COHERENT (cjk=241) |
| **exact + L1** | `rho=0 overlap=norm-q` | **36 / 43** | 171 | ✅ **COHERENT** (cjk=193/121) |
| **aggressive** | `rho=0 overlap=overlap` | 5 | 11 | ❌ DEGENERATE (`<think>` 重复) |

### 10d.1 两个可用结论

1. **L1（重叠）功能正确**：exact 叶片在 `overlap=norm-q` 下输出连贯（cjk=193），
   与参照词量级相当。**但 36–43 tok/s 慢于参照 61–63** —— 在本次测量下 L1 无净收益。
2. **aggressive 刀锋仍不可用**：孔径族修复**恢复了握手**（不再静默死、
   不再读陈旧 sidecar），但**没有恢复正确性** —— 输出退化为 `<think>` 重复、`pred_n=11`。
   说明 Q8DOT 算出的 Q 本身是错的，**与握手无关**。

### 10d.2 注意：测量纪律与容差

- exact 的 36 / 43 差异达 **19%**，是**同一臂内的漂移**，尚未达到可判决的样品数；
- 本轮是单次直跑，**不是交错 ABBA**，因此**不得**作为正式性能裁决；
- 可以判定的是**量级与方向**：参照 ~62，exact+L1 ~40 —— **慢约 1.5×**，超出任何漂移容差。

## 10c. 【结构性】L1 与 L2 **在调度上互斥** —— "尖端 = L1+L2"的前提不成立

### 10c.1 证据链

`exact + rho=1 + overlap=norm-q` 的诊断输出（五 rank 一致）：

```
status=[0,97,0,0,0,0,0,0]  qctrl=[ready=0 counter=0]
```

- `status[6]=0` → 发布器**从未设置** sidecar-ready
- `status[0]=0` → **P1 根本未完成**
- `status[1]=97` → epoch 已武装 ✅

**而 `exact + rho=1 + overlap=legacy` 完全成功**（`model loaded` / `listening`）。

### 10c.2 代码依据

```
:4360   plan.late_q_begin[idx] = pre.code.size();   // Q 收缩起点
        ... Q 派发 ...
        ... 发布器派发 ...                            // 位于 late_q_begin 之后
:6310   const size_t q_split = outgoing->late_q_begin[slot];
        emit(pre, inject_end, q_split);   // P1：散列前导 + zsnap
        emit(p2, 0, norm_end);            // P2：norm（等 Y）
        emit(pre, q_split);               // P2：[Q 收缩 + **发布器**] ← 整段搬家
```

**重叠调度把 `q_split` 之后 `pre` 的全部内容搬进 P2 流**，而发布器就在其中。
**LB2 的中继必须在 P1→P2 之间拿到那份发布才能算 ρ** → 它必然超时。

### 10c.3 结论

**L1（重叠）与 L2（ρ）不是"尚未一起测"，是当前调度下不能并存。**
因此 **用户 "尖端 = L1+L2+L3" 的前提本身不成立**；实际尖端是
**max(L1, L2)** —— 在两者被解耦之前只能取其一。

### 10c.4 可修点（已定位，未实现）

**关键事实：统计量 `(a_b, d_b)` 是从 `residual` 与 `z_p` 算出的，不依赖 Q 收缩的输出。**

```glsl
const vec4 rv = residual_stats[res_base + col4];
const vec4 zv = local_z_stats[token * (p.width / 4u) + col4];
a_b += dot(rv, rv);  d_b += dot(rv, zv);
```

**所以统计量 + 它的发布完全可以留在 P1，而只把 Q 收缩搬进 P2。**
当前做不到的原因只是：统计量被写在**同一个内核**里（精确路在 `TP5_LATE_Q_STATS`，
aggressive 路在我加的 Q8DOT 折叠里）。

**修法**：把统计量折叠拆成**独立派发**，它留在 `q_split` 之前；Q 收缩与它的载荷发布
搬到 `q_split` 之后。这样 L1 与 L2 同时成立。

## 10b.9 `aggressive x RHO` 崩溃：逐层剥离出**三个独立原因**
### 10b.9 `aggressive x RHO` 崩溃：逐层剥离出**三个独立原因**

用 gdb 实时取栈（core dump 走 apport 不可用），每修一条重测，逐层剥离：

| # | 崩溃点 | 原因 | 归属 |
|---|---|---|---|
| 1 | `tp5_record_plan:4083`（8 视图初始化列表） | `plan.late_stats_buf` **只在 `rho_enabled()` 时 resize**，而我把 8 元素数组写成了无条件构造 → `rho=0` 时对空 vector 取 `[i]` 越界 | **本会话引入** |
| 2 | `tp5_cmd_push`（layout 不匹配） | 我把管道与描述符集切到 `_stats` 变体，却漏了 push constant 的 layout。同一缺陷在精确路（`:4370`）**早已存在** | 两者都是 |
| 3 | `tp5_update_storage_set:4195` → **libvulkan_radeon 内部** | `plan.late_lo_u_ds` 在 `rho_enabled()` 下被 resize（所以 `.empty()` 为假），但它的 `lu_ai` 分配只在 **`else`（精确）分支**—— `late_q8_fast` 分支从未分配 → `VK_NULL_HANDLE` 描述符集交给驱动 | **既有结构性缺口** |

**三条都是"只在 `aggressive x RHO` 下才现形"** —— 这正好解释了为何该组合从未跑通：
它不是被否决的，是**每一层都缺一块，而第一块的缺失掩盖了后两块**。

### 10b.9-b 第 3 条暴露的更大缺口：发布器根本没接进 aggressive 路

追第 3 条时发现结构性问题 —— LB2 的 **GPU 发布器派发**只在 `else`（精确）分支里：

```
if (plan.late_q8_fast) {  ... Q8DOT 派发 ...  }        ← 无任何统计量发布器
else { ... 精确 Q 派发 ...; if (rho_mode) { publish_stats 派发 } }
```

**即使修好三条崩溃，统计量也永远到不了 CPU 读的 bank，relay 会等一个不会来的 generation。**

已补齐三块（均在 `late_q8_fast` 分支内、`rho` 门控）：

1. **分配** `late_q_stats_ds` / `late_publish_stats_ds` / `late_lo_u_ds`（照精确分支同形）
2. **发布器视图**（5 绑定）：另两个绑定指向任意**有效**缓冲（Q 载荷已由 Q8DOT 直接写进 host-import，**不需复制**），`LocalStats` = `late_stats_buf`，`HostStats` = `bcast_buf + stats_offset` —— 后者正是 CPU 读并写回 ρ 的那块
3. **发布器派发**（Q8DOT 之后，带 `late_stats_buf` 的 W→R 屏障），`n_uvec4 = 0` 驱动自拷贝为空（着色器 `for (i = lane; i < p.n_uvec4; ...)` 自然不执行），`n_stats = max_rows * 12`

**设计要点**：自拷贝置零而非另写一份发布器 —— 保证与精确路**同一条发布路径、同一套偏移契约**，避免第二条单源契约。

### 10b.8 机制的本质：不是"陈旧"，是**同步被移除**

载荷由每个 WG **无条件**写入（`tp5_hc_latebind.comp` 的 token 循环内，
`qmail[word] = packHalf2x16(...)`），而 `-15/-16` 的计数器与 ready 字**只承担完成信号**。
因此当重置打偏、计数器攀升、`prior+1 == n_workgroups` 再不成立时：

- ready 字**永久停在 1**
- → **同步被彻底移除**：relay 的 `q_ready` 检查立即通过
- → relay 不再等本轮 Q 内核写完就前进
- → LO 读到**半写或上一轮的 Q**

这比"读到陈旧数据"更准确地刻画缺陷：**它不是数据延迟，而是握手归零后被当成永远就绪。**
危害也因此更重 —— 会让下游在一段**未定义内容**上继续传播。

**可证伪判定**：修完三处后重测 ——
(a) 若输出恢复连贯 **且** 速度回到与分发数相称的量级 → 单一根因成立；
(b) 若只有输出恢复而速度仍逐请求恶化 → 存在第二个独立机制，届时从
`late_pre_us` 的逐 stage 分布（评审二已指出其 58–598 µs 波动）入手定位。

**三处都必须修**：武装重置负责周期起点、轮询基址负责看到新数据、消费退休负责不让旧 ready 泄到下一轮。

## 11. 【已修复】既有的构建破坏缺陷：树从未能从零编译

### 11.1 发现路径

改动 `vulkan-shaders-gen.cpp` 触发**生成器重建 → 全量重生成着色器**，于是暴露：

```
mul_mat_vecq_funcs.glsl:567: error: 'NUM_COLS' : undeclared identifier
mul_mat_vecq_funcs.glsl:567: error: 'NUM_ROWS' : undeclared identifier
mul_mat_vecq_funcs.glsl:568: error: 'K_PER_ITER' : undeclared identifier
```

**此前的构建全部是增量的，复用旧生成头文件，把这个缺陷掩盖了。**

### 11.2 根因

`mmvq_dot_multi`（K1a 多行 matvec 新增）的守卫**只按量化类型**：

```glsl
#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)   // :566
```

而它引用的 `NUM_COLS`/`NUM_ROWS` 是 `mul_mat_vec_base.glsl:69` 的**规范常量**、
`K_PER_ITER` 由 `mul_mat_vecq.comp:14-21` 按类型定义 —— **只有 MMVQ 矩阵向量核那个编译单元具备**。

`qwen4_attention_mmvq.glsl` **四次包含** `mul_mat_vecq_funcs.glsl`（只为拿 Q5/Q6 点积助手），
自带 `cache_b_qs[4]`、用 `NUM_ROWS_QUANT`，**从不需要这三者**，但它恰好也定义了
`DATA_A_Q5_K`/`DATA_A_Q6_K` → 守卫命中 → 在不具备参数的上下文里编译 → 失败。

这是"**共享库被改成了只在特定核心里成立，却没收紧守卫**"的典型形状。

### 11.3 修复

以 **`MMQ`** 作为判别式（`mul_mat_vecq.comp:6` 定义它，qwen4 路径不定义），
仅在 `mmvq_dot_multi` 族（行域 `560-660`）收紧 4 处守卫：

```glsl
#if defined(MMQ) && (defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K))
#elif defined(MMQ) && defined(DATA_A_Q6_K)
#elif defined(MMQ) && defined(DATA_A_Q2_K)
#elif defined(MMQ) && defined(DATA_A_Q3_K)
```

**行域限定是必要的**：同一守卫在文件中另有一处属于 `mmvq_dot_product`，**那是 qwen4 正常使用的，绝不能改**
（首次尝试的全文件替换被计数断言拦下——`(2, ...)`——这正是断言的价値）。
`mul_mat_vecq.comp:66` 的 `#if defined(HAS_MMVQ_DOT_MULTI)` 消费守卫保证回退路径存在。

**验证**：修复后 `mul_mat_vecq` 四组错误全部消失，构建仅剩我自己的 2 个错误。

### 11.4 真修复进度（三文件）

| # | 文件 / 落点 | 内容 | 状态 |
|---|---|---|---|
| 1 | `vulkan-shaders-gen.cpp:1010-1018` | 变体 `tp5_hc_late_q_q8dot_stats` = `{TP5_LATE_Q_Q8DOT, TP5_LATE_Q_STATS, TP5_LATE_CAPACITY_ROWS}` | ✅ |
| 2 | `tp5_hc_latebind.comp:322-331` | Q8DOT 新绑 3-7：`ResidualStatsQ`/`LocalZStatsQ`/`ScatterStatsQ`/`StatsOutQ`/`ScatterOutQ` | ✅ |
| 3 | `tp5_hc_latebind.comp:397-422` | WG0+`lid<32` 折叠 `a_b`/`d_b`/`s_b`，**用 `subgroupAdd` 而非 `tp5_hc_sum32`** | ✅ |
| 4 | `ggml-vulkan-collective.cpp` 12 处替换 | 声明/销毁×3/创建/reset×3/分配三元/描述符 8 视/派发成对切换 | ✅ |
| 5 | `common/common.cpp` | **撤掉 fail-closed**（它只是止血，不得充当修复） | ✅ 已删 |

**关键设计点：分配时按 `tp5_latebind_rho_enabled()` 二选一 layout**，描述符集存入既有的
`plan.late_q8dot_ds` —— 省掉新 plan 字段/释放/resize，且**管道与描述符集必须成对切换**，
否则 8 绑定着色器会读到越界绑定。

**为何用 `subgroupAdd`**：文件头 `:10-16` 明确写着 Q8 变体**有意不 include**
`tp5_hc_subgroup.glsl`，以免把 128 字节共享数组与屏障对塞进 Q8 核。
去 include 它就会违反该设计；我的块恰好是完整一个 wave32，波内归约正是该设计要的形状。

## 9-old-b. （已撤回的"警报已撤回"，保留以便追溯错误链条）

## 9. aggressive 刀锋的输出质量：**警报已撤回**（本节的错误归因由本节自己纠正）

> **结论先行**：§9.1–9.3 曾把 aggressive 的低输出量当作"刀锋坏事"的"重大发现"。
> **它是错的，已撤回。** 错误在于把 `n_predict=32` 的冒烟结果与 `n_predict=171` 的
> 参照臂相比 —— 即本文件正在修的第四类缺陷（不同工作量相比）。

### 9.0 撤回依据（冒烟请求原文）

`/var/tmp/tp5-agg-smoke.sh:14`：

```bash
PORT="${PORT:-8151}"; NP="${NP:-32}"
REQ=$(jq -n --arg p "$PROMPT" --argjson n "$NP" '{prompt:$p, n_predict:$n, temperature:0, seed:42, stream:false}')
```

**请求就是 `n_predict=32`。** 所以 `pred_n=32` 是**长度上限**，不是自然停止。
我曾把 `truncated=false` 当作自然停止的证据 —— **服务端不因 `n_predict` 达限而置 `truncated`**，
那是误读。两臂的 `predicted_n`（32 vs 160）差异**完全由请求参数解释**，不携带质量信息。

**因此以下均不成立**："早期 EOG"、"退化重复"、"输出不可用"、
"先前 5.6× 结论作废"。前者都是对同一次错误归因的推论。

### 9.0b 仍然开放的（不是结论）

32 token 内 47 字节确实偏低（中文 3B/字），但**在没有同长度参照的前提下，
它不能证明任何事**。正处思考期、以空白/换行分段都可能造成低字节/token。

**未决问题的正确形态**：同 `n_predict`、同 prompt、同 seed 下 reference vs aggressive
逐字节对比。`bg_14` 正是这个对照（六个 aggressive 臂均在 171 长度）。
**在它回传前，本文件不对刀锋质量下任何结论。**

---

## 9.5 缺陷排查：aggressive 链四道契约逐字核验 —— **未发现缺陷，路线不否决**

用户裁定："BUG 不代表路不通，是实现有缺陷"、"必须修，禁止以 BUG 否决路线"。
据此对 aggressive 链的四处关键契约做了逐字核验：

| # | 契约 | 位置 | 结论 |
|---|---|---|---|
| 1 | **Q8DOT 归约几何** | `tp5_hc_latebind.comp:374-390` | `subgroupShuffleXor(mask)` 1..8 在半波内求和、`Xor 16` 取另一行、`row0 = 2*SubgroupID` 与 `acc`/`pair_hi` 配对、`packHalf2x16(vec2(acc, pair_hi))` 打 (row0,row1)、`word = late_word_offset + elem/2` 半字索引 —— **自洽** |
| 2 | **激活量化器** | 同上 `:264-305` | `(z·s + R)·Γ` 与精确块 `rz0=z0*s; rz0+=residual4[...]; b0=gamma4[...]*rz0` **完全同序**；每 32 块 `d=amax/127`；`uint(q)&0xff` 对负数取二补字节正确；4×8 打包契约与消费端一致 |
| 3 | **权重打包** | 同上 `:46-79` | `uint16_t(s.qs[2w])` 的加宽写法**与 llama.cpp 自己的 `dequant_funcs_cm2.glsl:264-266` 逐字同构** —— packed16 布局的既定惯用法，非缺陷 |
| 4 | **Q 邮箱协议** | 同上 `:394-408` | 原子计数 + ready 标志、控制行位于 `late_word_offset-15/-16`、与既有 fused HC 模式一致 —— **自洽** |

**结论：没有已证实的缺陷。** 先前"刀锋输出不可用"的警报已撤回（§9.0）；
其依据是拿 `n_predict=32` 的冒烟与 `n_predict=171` 的参照相比 —— 正是本文件
正在修的第四类缺陷（不同工作量相比）。47B/32tok = **1.47 B/token** 与参照
288B/160tok = **1.80 B/token** 比值同级，无异常。

| 5 | **打包完成标志的跨 replay 语义** | 同上 `:47-79` | `dst[0].d` 的完成标志：`barrier()` 只是 workgroup 级，WG0 可在其余 WG 写完前置位 —— **但良性**，因为该标志只门控**后续 replay**，而同一命令缓冲的分发是顺序的，下次 replay 开始时上次已整体完成。索引自洽：写 `dst[block+1]`、消费端读 `wi = ... + block + 1u`、index 0 专作标志。**残留仅是健壮性**：若某次分发被中途取消（device loss/超时），标志可能带着部分数据置位；正常路径不受影响 |

**唯一缺的是一次合法对照**：同 `n_predict`、同 prompt、同 seed 下
reference vs aggressive 的逐字节对比。`bg_14` 正是它（六个 aggressive 臂均在 171，
镜像交错，账本三重断言）。若它们产出与参照同量级的连贯文本，则该问题干净地回到性能层。

---

## 9-old. （以下为已撤回的原始推断，保留以便追溯错误链条）
## 9. 【重大】aggressive 刀锋输出不可用 —— 尖端对参照的前提不成立

### 9.1 证据

历史冒烟产物 `/var/tmp/tp5-agg-smoke/push.json` 仍在：

| 臂 | `predicted_n` | `bytes` | 内容 |
|---|---|---|---|
| reference（本轮新跑 R1） | 160 | **288** | 连贯中文 |
| **aggressive（push.json）** | 32 | **47** | `<think>健康。，，， 1` / `（ 10 句话 2。，` —— 碎片、重复、非文本 |

**这不是"快而略差"，是退化的输出。** 用它跑尖端对参照得到的是双重无效：
既非同一工作量，也非同一质量 —— 正是纪律禁止的"不同东西相比"。

### 9.2 后果

1. **先前那句"aggressive 比参照慢 5.6×、孔径不是它的瓶颈"作废**：
   那个耗时来自一条产出废文本的路径，做的功根本不同。
2. **上一轮的裁定需重新校准**：用户纠正"拿 fallback 判 L1"是对的，但
   **aggressive 刀锋输出不可用，拿它判 L1 同样是违规（另一种）**。
   **在刀锋变正确之前，不存在有效的尖端对参照。**
3. 近似 Q 的全部消融（`rho=`、`overlap=`、`transport=`）**都无法在无效输出上裁决**。

### 9.3 歧义未消（不把它写成结论）

两种读法都有支持，本轮**不能**在两选一之间拍板：

- **读法 A（数值性破坏）**：`q8dot-approx` 的近似使 logits 退化到破坏推理，
  输出进入退化的重复状态。若是这种，这是**设计级否决**，不是 bug。
- **读法 B（呈现层）**：`stream=false` 且内容落在 `<think>` 块内，`content` 可能只
  截取了部分可见文本；47B/32tok 与 288B/160tok 的字节/token 比（1.5 vs 1.8）相差不算悬殊。

**分辨方法（未做）**：同一 `n_predict`、同一 prompt、reference vs aggressive 的
`content` 逐字节对比，并检查 `.reasoning_content` / 完整响应体。
**补充证据（同一轮取证）**：

| | `predicted_n` | `content_len` | `truncated` | 停止方式 |
|---|---|---|---|---|
| aggressive | **32** | 47 | **false** | **自然停止** |
| reference | **160** | 288 | **false** | **自然停止** |

两臂都是自然停止（非长度截断），而 **aggressive 在 32 token 就停了，且文本从
`健康。` 之后立即退化为 `，，，` / ` 1` / `（ 10 句话 2。，`。** 健康模型在第 32 个 token
应当仍在写连贯推理句 —— 这不是截断，是**早期 EOG + 退化重复**。
两臂的 `tokens` 数组均为空（服务端未回传），因此无法用 token 序列直接定案。

**因此读法 A（近似 Q 数值退化）被强烈指认，但仍不写成定论** —— 缺同长度配对对照。
`bg_14` 正好就是那个对照。

### 9.4 对 `bg_14` 的意义

`bg_14` 仍在跑，其六个 aggressive 臂的 `r1 bytes=` 与 `pred_n=` 就是**在该长度上
量化这个缺陷的证据**。若它们离开 288/160 的量级，即坐实读法 A。

---

# 12. lbfin 会话（2026-09-28 07:51–13:13，本节补记）：L2 重设计为 td、q8first/pack-once、最终 ABBA

> 源码树 `/var/tmp/tp5-candidate-k1a`（提交 `9daff2151`，基 `16e28b8ac`），
> 构建 `/var/tmp/tp5-final-build`，全部原始数据 `/var/tmp/tp5-lbfin/`。
> 本节把 06:44 之后未入账的六小时补齐。

## 12.1 §6-1 的完成方式：不是一次性回读，而是链内连续对拍（更强）

`GGML_TP5_LATEBIND_RHO_PROBE` 的一次性 vkMapMemory 探针被**放弃**（device-local 不可映射、
旁路 CB 等不到、vkDeviceWaitIdle 打断链协议——§3.3 已记录三种失败）。替代方案：

- **`GGML_TP5_LATEBIND_L2=td`**（新路径，`ggml-vulkan-collective.cpp:207`）：RMS 充分统计量
  `(a_b,d_b)` 在 **ACT_Q8** 内从 live `R/z_p` 生成（不依赖 Q 收缩——即 §5 的解耦设计落地）、
  由 Q8DOT 折叠进 Q 孔径、relay 由它们合成 `rho_b`、`u`、SiLU 与 Q8_0 LO 块直接写进 bank；
  LO 内核与 F16 Q fanout 消失。
- **`GGML_TP5_LATEBIND_L2_CHECK=1`**（链内校验，`:6147`）：norm 内核把参照 inverse RMS 写进
  host status words 8..15，relay 在同 epoch 对比自己合成的 rho。**尺度问题就地解决**：
  `rho = 1/sqrt(max(0,ss)/width + eps)`（`:6341` 附近）。

**实测（`/var/tmp/tp5-lbfin/v1/Tc.log`，16k+ 样本）**：

```
[tp5-l2-check] samples=24580 max_rel=2.427e-03 mean_rel=3.4e-07 worst: l2=18.1544762 ref=18.198637
```

**§3.3 的 3×10⁵ 尺度失配不复存在**：l2 与 norm 参照同量级（~18），相对误差 2.4e-3
（Q8_0 量化精度内）。旧 LB2 的 `inversesqrt(ss/width+epsilon)` 撤回（§10h.68）失败的原因
也随之清楚：旧路径的 `ss` 输入张量尺度本就错（§10h.71 的判断成立）。

## 12.2 数据可靠性：tag 协议（本会话的关键工程发现）

Q ready word 与统计量 entry 都**不保证 host 观察到 payload**。实测：ready 之后 relay 偶尔
仍读到上上次使用该 bank 留下的 entry（~19k 次交接中 0–4 次）。解法：GPU 用本 epoch 的
`z_p[t][0..1]` + 武装 epoch 生成 tag，entry 第 4 字携带 tag，relay 用自己的 P1 z_p 副本
算同一 tag 并**自旋等待 tag 匹配**（`tp5_l2_relay_form_lo`，`:6206-6280`）。Q payload 侧
同型：每 workgroup 发布 `uvec4(xor(payload), nonce)`，relay 校验后才归约。
`[tp5-l2-tagwait] entries_not_yet_current=0 q_wg_not_yet_current=0`（16384 次交接零等待命中）。

## 12.3 q8first / pack-once

- `GGML_TP5_LATEBIND_Q8_ORDER=first`：ACT_Q8 → WAR → Q8DOT → norm（Q 收缩 workgroup
  先于自旋等 Y 的 norm workgroup 发射）。
- `GGML_TP5_LATEBIND_PACK_ONCE=1`：Q8_0 权重打包每 site 一次而非每行（`barriers` 2496→2406）。

## 12.4 最终 ABBA（预注册，`/var/tmp/tp5-lbfin/abba.sh`，run 12:53–13:13）

4 块 × 4 臂 × 3 测量请求（+warm-up），交错 R A Ap T | T Ap A R × 2，同二进制，
`arm.sh` 环境纪律（env -i、RADV_DEBUG=nobolist、GPU 锁、账本断言）。
报告：`abba_report.py`，配对 per-block 中位 ms/cyc 比值 + t-95% CI。

| 臂 | 配置 | tok/s 中位 | ms/cyc/R（95% CI） | 输出 |
|---|---|---|---|---|
| R | `--tp5-latebind off` | 61.75 | 1.0 | 12/12 同 md5 `add18ad37ff7`，cjk=241 |
| A | aggressive-q8 | 63.27 | 0.9723 [0.9456, 0.9989] | 12/12 同 md5 `ebf49ef9d1ca`，cjk=187 |
| **Ap** | A + q8first + pack-once | **65.39** | **0.9478 [0.9072, 0.9884]** | 12/12 同 md5（与 A 相同），cjk=187 |
| T | A + L2=td | 42.23 | **1.4138 [1.3225, 1.5051]** | 12/12 同 md5 `6aa03b97213c`，cjk=244 |

**判决**：
1. **Ap 首次显著超参照**：ms/cyc -5.2%，CI 不含 1；Ap/A = 0.9748 [0.9522, 0.9973]，
   q8first+pack-once 贡献独立 -2.5%。
2. **T（L2=td）数值正确但性能被否**：rho 对拍 mean_rel 3e-7、输出 12/12 逐字节稳定
   （**超过 R 的稳定性**——R 自身 5 次输出互不相同，§10h.61），但 ms/cyc **+41%**。
   归因（prof 臂，`tp5-l2-cost`，per-handoff）：`lo=7.97µs`（u/SiLU/Q8_0 量化在 CPU）+
   `validate=1.86µs` + `q_reduce=1.03µs` + `stats=0.26µs` + `write=0.48µs` ≈ 11.6µs/handoff
   × ~96 handoffs/chain ≈ 1.1ms/cycle ≈ 全部差额。**瓶颈是 CPU 上的 LO 合成，不是协议**。
3. T 臂 fallback 15 次/会话 vs 其他臂 5：未单独归因（L2 路径 dispatches 2785 vs 2879，
   图形差异本身可解释计数差；非数值错误，输出仍逐字节稳定）。

## 12.5 会话结论与移交

- **aggressive 刀锋 = 65.4 tok/s（Ap），超参照 +5.6% tok/s（61.75→65.39 中位）**，
  输出与 A 逐字节相同、账本全同（dispatches=2879 barriers=2496/2406 copies=191）。
- **L2 数学与协议已闭合**（对拍 2.4e-3、tag 零等待、输出超参照稳定），
  剩余是纯性能问题：LO 合成 8µs/handoff 在 CPU。若要续攻，方向是把
  `tp5_l2_lo_block_avx2` 的 u/SiLU/Q8_0 搬进 Q8DOT 或 LO_U 着色器（GPU 已有全部输入：
  rho 在 stats 区、Q 在孔径）——**先量后建**：11.6µs×96 ≈ 1.1ms/cycle 与实测 +41% 吻合，
  收益上限明确。
- 源码已提交：候选树 `9daff2151`，主仓 `5c4dac988`（逐字节核对 4 个关键文件 diff=0）。

## 12.6 【归因完成】T 臂 fallback 15/会话 vs 其他臂 5 —— 不是数值错误，是 L2 计划键漂移导致的重录循环

逐行核对 `/var/tmp/tp5-lbfin/abba/T1.log` 与 `A1.log`（同一二进制、同请求）：

**现象分解**（T1 全部 15 次回退，warmup 占 2）：
- A1：warmup 2 + req1 3 次，req2–req4 **零**回退（draft 图缓存命中，`cache_size` 稳定在 194）。
- T1：**每个请求边界固定 3 次回退**（请求结束 1 次 + 下个请求开头 2 次），
  `cache_size` 序列 `0 97 0 97 194 | 0 0 97 0 | 0 97 0 | 0 97 0`——**缓存反复清空**。

**机制链**（源码逐点核实）：
1. miss 的都是 `nodes=79` 的图 = **draft（standalone MTP）图**，走 `pfn_get_cbs`
   （`ggml-backend-meta.cpp:4272`）→ `ggml_vk_tp5_get_cached_cmd_bufs`（`ggml-vulkan.cpp:27797`）。
2. T 臂（L2=td）每个 late 计划额外分配 `late_stats_buf`（`collective.cpp:3974-3980`），
   且 `evict_lru_plan`/`cached_plans_overflow` 路径会 `invalidate_chain()` → `linear_chains.clear()`
   + `plans_gen++`（`:1371-1380`），**连带击穿 vk 侧 `cgraph_cmd_cache` 的指纹一致性**
   （fingerprint 校验 `scratch_gen`/`relay_payload_generation`，`:3344-3350`）。
   T 的 96-stage 链每 stage 一个 plan、L2 键更宽 → 计划键漂移频率高于 A，
   每次漂移 = 一次 `tp5_record_plan` 重录（`p1_rec` 非零，T1 14 次 vs A1 4 次）。
3. 重录后的新 CB 句柄与 meta 侧 `stage_compute_cbs` 不再逐位相等 →
   `meta-chain-miss` → `CRITICAL-FALLBACK` 480 子图逐个提交（每次 ~160-180ms，
   `vk-compute-profile` 实测）。

**代价核算**（prof 臂 Tc2 vs At，同 3 请求）：
- 回退事件 11 次/1981ms vs 4 次/867ms → **+370ms/请求**；
- T 的 draft 接受率 0.707 vs A 的 0.771 → 101 vs 97 cycles → +4 cycles ≈ +156ms；
- 每周期 +3.3ms（L2 relay CPU 段，`l2-cost`：validate 2.3 + q_reduce 0.95 + stats 0.28 + lo 1.22 + write 0.23 ≈ 5µs/handoff × ~96 handoff ≈ 0.5ms/cycle——**注意 lo 已从 7.97 降到 1.22µs**（12:11 的 `tp5_l2_lo_block_avx2` 改进，build16），旧 §12.4 的 8µs 数字过时）。
- 合计 ≈ +850ms/请求 ≈ 实测 +907ms。**+41% 全部归因完毕，无未解释残差**。

**结论**：L2=td 的性能问题不在数学、不在协议、也不主要在 CPU LO 合成（已优化到 1.22µs/handoff），
而在**计划缓存稳定性**：L2 的宽键使 `cached_plans` 演化更频繁，每次演化击穿
draft 图的 replay 缓存 → 480-subgraph 回退。修法方向（未实现）：
(a) 把 `late_stats_buf` 绑定纳入计划键或改用稳定全局缓冲，避免键漂移；
(b) `evict_lru_plan` 只作废被逐出的计划、不 `invalidate_chain` 全清。
**先量后建**：单修 (b) 的收益上限 = 消除 ~370ms/请求 ≈ T/R 从 1.414 → ~1.29。

## 12.7 【实证归因，修正 §12.6】T 臂回退循环的真根因：每请求的 compute buffer 换血 → 静默抹掉 vk replay 缓存

用 `GGML_VK_TRACE_CACHE_FREE=1` 重跑 T/A 各 3 请求（`/var/tmp/tp5-lbfin/dbg/Tfree.log`、`Afree.log`，同二进制同请求，GPU 锁）：

**实测**：
- A：全程仅 **6 次** buffer free（都在 req1 开始：warmup 的 5×10265712 + 1×3143680）。req2/req3 **零** free、零 miss、零 fallback。
- T：**38 次** free——每个请求开始 5×9974368（上一个 decode buffer）+ 请求结束 5×(14422144 或 10782912) + 小 buffer。**每请求 ~50MB×5 rank 的 buffer 换血**。

**因果链（逐行时间戳核实）**：
1. T req1 的早期 fallback（cnt3–5，与 A 相同——两臂 req1 都因 warmup 19 行→4 行的行数变化 miss）在 480-subgraph 逐个提交路径上分配了**额外尺寸**的 compute buffer（14422144 ≠ 初始 10265712）。
2. req1 结束时这批 buffer 被 free → 走 `ggml_backend_vk_buffer_free_buffer` 的**静默擦除路径**（`ggml-vulkan.cpp:22031-22053`：entry 的 `buffer_owners` 含被 free 句柄即整条擦除，仅 `GGML_VK_TRACE_CACHE_FREE` 时才打印）→ vk replay 缓存清空（`cache_size 194→0`）。
3. req2 的 catch-up/prefill 图 `pfn_get_cbs` miss → `CRITICAL-FALLBACK` → 又分配新尺寸 buffer（9974368/10782912 交替）→ 请求结束再 free → **自维持循环**。
4. A 逃逸的原因：其 req1 fallback 没有引发 buffer 换血（req1 结束零 free），缓存存活，req2+ 全部走 predefined/chain 快路（`SUBMIT_EPOCH_CHAIN SUCCESS`，meta-check uid 更新但 pfn 命中）。

**§12.6 的修正**：不是"计划键漂移"——计划键与 linear 链参数在 T 的 req1–req3 完全相同（`dispatches=2731 barriers=2362` 逐字相同仍重录）。是 **buffer 生命周期**：fallback 分配的 per-request buffer 在请求边界被 free，缓存条目因 `buffer_owners` 钉着这些 buffer 而连坐擦除。

**代价**（Tc2 vs At，prof 臂）：回退 11 次/1981ms vs 4 次/867ms（+370ms/请求）+ 计划重录 14 vs 4 次（~+250ms/会话）+ 接受率差导致的 +4 cycles（+156ms/请求）。合计 ≈ 实测 +907ms/请求，**无未解释残差**。

**修法方向（未实现，先量后建）**：
- (a) 最小：fallback 路径复用既有 sched buffer（不新分配）——消灭尺寸漂移即消灭换血；
- (b) 或：请求边界的 buffer free 不擦缓存（把 `buffer_owners` 的连坐改为按 binding 精确失效——但 recorded CB 确实绑着被 free 的 buffer，直接不擦是 use-after-free，不可行）；
- (c) 或：让 T 的 req1 早期 fallback 不发生（warmup 与 req1 行数一致的预热），循环根本不启动。
- 收益上限：消除 ~370ms/请求 + 重录 ~250ms → T/R 从 1.414 → ~1.15–1.2。**L2=td 的 CPU 段本身（~5µs/handoff）已不是主要矛盾**（lo 已优化至 1.22µs）。

## 12.8 【已修复并实测】GGML_GALLOC_KEEP_CHUNKS：一行修复消灭 T 臂回退循环 —— L2=td 从 42.2 → 66.2 tok/s

**修复**：`ggml/src/ggml-alloc.c` 的 buffer-pool trim 路径（`:818` 后）加
`GGML_GALLOC_KEEP_CHUNKS`（默认 1）：decode↔prefill 相位切换时**不再 free/shrink
compute chunk**（保留策略；增长路径不受影响）。被 free 的 chunk 正是把 vk replay
缓存连坐抹掉的元凶（§12.7）。

**验证**（同二进制、GPU 锁、`/var/tmp/tp5-lbfin/dbg/`）：

| 臂 | 回退 | tok/s | 备注 |
|---|---|---|---|
| T + KEEP=0（旧行为） | 13/会话 | ~44 | 循环复现（Tkeep0.log） |
| **T + KEEP=1（新默认）** | **req1 后 0** | **66.24**（r2/r3） | Tkeep1.log；61.97 (r1 warm) |
| T + KEEP=1 + L2_CHECK=1 | — | 49.4（带校验开销） | `max_rel=2.17e-07`（旧 2.4e-03） |

**数值**：L2_CHECK 对拍从 2.4e-3 收紧到 **2.17e-7**——旧路径的 2.4e-3 误差部分来自
回退循环期间跨 bank 的陈旧读；修复后 l2 与 norm 参照几乎逐位一致。
输出：会话内 3/3 同 md5（`0e99fb52234b`，n=168 自然停，cjk=255 连贯）。
跨会话与旧 T 输出不同（`6aa03b…` vs `0e99f…`）——底层 TP 归约非确定性的已知现象
（§10h.61：参照路径自身跨会话也不稳定），非缺陷。

**性能判决改写**：修复后 T（L2=td）≈ **66.2 tok/s ≈ Ap（65.39）≈ A+5%**。
原 ABBA 的 T/R=1.414 判决**作废**：那是回退循环的代价，不是 L2 机制的代价。
L2=td 的真实开销 ≈ 每周期 ~1-2ms（CPU 段 5µs/handoff + 计划差异），与 A/Ap 同带。

**遗留**：KEEP_CHUNKS=1 保留 decode+prefill 两套 chunk（VRAM 占用略增，本机不缺）。
需要正式 ABBA 复测（R/A/Ap/T 四臂、预注册、镜像）来出最终数字。

## 12.9 消融补充（修复后，单因素，`/var/tmp/tp5-lbfin/dbg/`）

| 配置 | tok/s（r1/r2/r3） | 输出 |
|---|---|---|
| R（off）+ keep | 60.20 / 59.58 / 60.96 | md5 `add18ad37ff7`（与历史 R 逐字节同） |
| T（L2=td）+ keep | 61.97 / **66.24** / 66.33 | 会话内 3/3 同 md5，cjk=255 连贯 |
| T + q8first + pack-once + keep | 60.39 / 63.23 / 61.80 | 同 T 输出 |

结论：
1. **R 无回归**（60-61 tok/s、md5 不变、req1 后 0 回退）——KEEP_CHUNKS 对非 L2 臂中性。
2. **q8first/pack-once 与 L2=td 不叠加**（63.2 vs 66.2）：L2 路径下 Q 收缩形状不同
   （无 F16 fanout），q8first 的发射顺序优势消失。**L2=td + legacy 顺序即 T 臂最优**。
3. T（66.2）已与 Ap（65.4，无 L2）持平且更高——L2 的 relay CPU 段在修复后
   被链内并行完全掩盖。

**下一层已无单因素可攻**：transport 消融对 L2 是硬约束（fail-closed 要求 gtt-push）；
overlap 在 aggressive 刀锋上无作用点（六臂账本全同，§10f.2-a#）。
剩余变量只有正式 ABBA 复测。

## 12.10 【终局 ABBA】修复后四臂复测（run-20260928T1550，`/var/tmp/tp5-lbfin/abba2/`）

同二进制（含 KEEP_CHUNKS 修复）、预注册 4 块 × 4 臂 × 3 请求、交错镜像、
账本断言全过、arm.sh 环境纪律。

| 臂 | tok/s 中位 | ms/cyc/R（95% CI） | 输出 | 回退 |
|---|---|---|---|---|
| R 参照 | 61.34 | 1.0 | 12/12 同 md5 | 5 |
| A aggressive | 63.41 | **0.9579 [0.9178, 0.9979]** | 12/12 同 md5 | 5 |
| Ap = A+q8first+pack-once | 62.83 | 0.9606 [0.9301, 0.9911] | 与 A 同 | 5 |
| **T = A+L2=td** | 62.19 | **1.0049 [0.9745, 1.0353]** | **12/12 同 md5，cjk=255** | **6** |

**判决**：
1. **A 臂显著超参照 -4.2% ms/cyc**（CI 不含 1）——LateBind 刀锋的净收益正式成立。
2. **T（L2=td）与参照统计不可分**（CI 含 1）：修复后 L2 的全部协议+CPU 代价被链内
   并行掩盖。T/A = 1.05（L2 相对 A 的净代价 ~5%，即每周期 ~1.4ms）。
3. Ap 与 A 不可分（1.0035 CI 含 1）——q8first/pack-once 的旧 +2.5% 未复现
   （上一轮的 0.9748 在本会话漂移带内；不再主张独立收益）。
4. T 输出 cjk=255/marks=22，**超过 R 的连贯度**（241/19），12/12 会话内逐字节稳定。
   T 回退 6 次/会话（vs 修复前 15）：req1 的行数过渡 4 次 + warmup 2 次，之后零。

**LateBind 路线终局**：
- 数值：L2_CHECK max_rel 2.17e-7，rho 尺度问题关闭。
- 性能：A 显著超参照；T（完整 L2）与参照打平、比 A 慢 5%。
- 剩余理论空间：T/A 的 1.4ms/cycle = L2 relay 的 validate+reduce+stats+lo+write
  CPU 段（~5µs × 96 handoff ≈ 0.5ms）+ 计划差异。要吃掉它需把 LO 合成搬进 GPU
  （Q8DOT/LO_U 已有全部输入），预期收益 <5%——**低于本 harness 分辨率（±3-5% 会话漂移），
  除非先修会话漂移本身，否则不可检定**。
- **路线判定：做到无可做**。单因素消融全部收敛（transport 硬约束 fail-closed、
  overlap 无作用点、q8first/pack-once 不叠加），剩余效应低于测量分辨率。
