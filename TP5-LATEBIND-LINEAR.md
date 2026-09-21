# LateBind：连续命令流、按需等待与 aggressive Q

本页对应 2026-09-21 的源码改造。**未编译、未运行 shader、模型或性能测试**；其他工作同时进行，由用户统一构建、验收。不能据此宣称已经恢复或超过 52.47 tok/s。

## 执行入口与数值模式体系

### 三类数值模式定义

根据 P0 路线图"区分真正参考数值路径与 aggressive 路径"的要求，系统在定义期（Plan 构建 / 编译链定义期）严格确定并打印三类数值模式，杜绝逐 token 检查开销：

1. **参考模式（`reference`）**：
   * **定义**：常规非-LateBind 路径。完全遵循原始数学图与标准 AllReduce 传输/规约流程，无任何延迟绑定、解耦或量化；
   * **生效条件**：未设置 `GGML_TP5_LATEBIND`（或非 `hc-down/1/on`）、同步模式非 `RELAY`、传输非 `f32` 线材、或当前 stage 无可延迟绑定的 HC 算子。
2. **F32 LateBind 模式（`exact-f32`）**：
   * **定义**：启用 LateBind 流水线解耦（P1/Scatter/Norm/Q 并行发射），但 Q 充分统计量保持精确 FP32 运算，sidecar 为 FP32，无任何激活量化或整数收缩误差（仅包含 RMS 树重排引入的必要末位浮点舍入）；
   * **生效条件**：满足 LateBind 基础条件，但设置了 `GGML_TP5_LATEBIND_EXACT_Q=1`，或设备缺少硬件整数点积（`integer_dot_product`）、或不支持固定 32-lane wave32、或张量形状不对齐、或 Q8 着色器管线创建未就绪。
3. **Aggressive 模式（`aggressive-q8`）**：
   * **定义**：启用 LateBind 全流水线解耦 + activation Q8_0 量化 + 4-wave32 workgroup Q8xQ8 整数点积收缩 + FP16 sidecar 紧致传输 + LO Q8 激活重构；
   * **生效条件**：满足 LateBind 基础条件，未请求 exact Q，且五卡均支持硬件整数点积与 wave32 控制，且张量形状对齐、Q8 着色器管线创建成功。

### 模式判定与定义期输出示例

定义期对每个构建的 plan 与每条编译的 linear chain 进行一次性静态确认，并输出统一日志：

* **模式判定日志**（每个数值模式首次出现时输出一次）：
  ```text
  [tp5-numerical-mode] mode=aggressive-q8 reason=none wire=f32 late=yes direct=p1
  [tp5-numerical-mode] mode=exact-f32 reason=exact-q-requested wire=f32 late=yes direct=p1
  [tp5-numerical-mode] mode=reference reason=disabled-by-env wire=f32 late=no direct=p1
  ```
* **链定义日志**（每 rank 每条 primary CB 定义时输出一次）：
  ```text
  [tp5-linear-definition] rank=0 stages=96 primary_cbs=1 mode=aggressive-q8 late=96 q8_fast=96 ...
  ```

保留现有 `GGML_TP5_LATEBIND=hc-down`，只用于 RELAY/F32。`GGML_VK_DISABLE_PRODUCER_WIRE=1` 仍然保留 P1，attention replicate 仍独立可开关。

```bash
export GGML_TP5_LATEBIND=hc-down
export GGML_TP5_WIRE=f32
export GGML_VK_DISABLE_PRODUCER_WIRE=1
unset GGML_TP5_LATEBIND_FUSED_FINALIZE
# GGML_TP5_REPLICATE_ATTN 单独选择，不由 LateBind 改写。
```

旧 `GGML_TP5_LATEBIND_FUSED_FINALIZE=1` 会明确报错。它对应共同 generation 的旧 1024-thread 等待流程，不能与新的 Y/Q 独立 generation 混用。

默认 aggressive Q 在设备具备 packed signed 4×8 integer-dot、subgroup-size-control 且可强制 wave32 时启用。目标 Navi21 满足这些条件：local sufficient-statistic activation 先按 Q8_0 block 量化一次，再以 4-row / 4-wave32 workgroup 做 Q8×Q8 integer-dot contraction。日志为 `q=q8dot-approx`。设置

```bash
export GGML_TP5_LATEBIND_EXACT_Q=1
```

可强制回到 F32 sufficient-statistic Q 对照路径。这个开关只保证 Q 不经过 Q8 activation grid；aggressive RMS reduction tree 等其它重关联仍可能改变末位。

OFF 路径不收集命令 IR，不分配 sidecar，也不为 F32 producer-wire 建立 HC 候选切口。F16 原有的显式 fused-consumer 路径单独保留。

## 命令定义与运行

`ggml-vulkan-tp5-command.h` 在定义期保留必要的 Vulkan 指令参数、descriptor pool、pipeline 和 buffer 所有权。指令序列包含 HC norm 完成、HC down 完成两个**语义位置**，不是额外的 Command Buffer。

普通冷图仍按原数学图执行。只有完整定义可用时，collective 才把所选方案直接录入每 rank 的一个 primary CB。LateBind stage 的实际顺序是：

```text
原生 model body / terminal z_p
    ↓
scatter + visibility（若已提前则只保留必要 visibility）
    ↓
P1（若使用 P1；linear LateBind 跳过 P1 内重复的 producer→reader barrier）
    ↓
norm_resume：先准备 residual/gamma → 首次使用 Y 时有界等待 → combine/RMS/norm
    ╲
     ╲ 同一 primary、无 norm→Q execution barrier
      → Q 预计算 → local-VRAM sidecar
                     ↓
                64-lane fused publisher：
                128-bit host sidecar stores + status[6]
                     ↓
                单次 COMPUTE→HOST publish
    ↓
下一消费点：rho→LO barrier → lo_resume 首次使用 Q 时有界等待
    ↓
SiLU/lo → 原生 W_up/fold → 后续模型
```

关键点不是“把两个 generation 合并”，而是**删除人为的 Q→norm 串行关系**。norm dispatch 的语义切点位于 rho→LO barrier 之前，因此后续 Q dispatch 可以在 norm 的四个 workgroup 等待/消费 Y 时进入调度；rho→LO barrier 留在真正的 LO 消费点。

新 primary 中没有 `vkCmdExecuteCommands`，不按“跳过两个 CB”拼接，不在每个 token 上解释 IR。定义期仍保留常规 transport CB 用于原有非整链执行；LateBind 整链不提交这些片段。

定义不完整、出现不可安全保留的事件/图像依赖、源 CB 发生变化时，整链在提交前明确失败，不静默漏录命令。普通模型 fallback 与 LateBind 替换以完整 stage 为单位，五卡资格必须一致。

## 两种 generation 与 Q Sidecar 内存布局

每 bank 的 GPU inbox 头部仍保留主 Relay 控制区（$B$ 起始的 64 字节）：

| 位置 | 新含义 |
| --- | --- |
| inbox word 0 | Y generation |
| inbox word 2 | Q generation |
| host status word 1 | 本 bank 预定 epoch |
| host status word 2 | sticky failure |
| host status word 3 | 最后 inbox 读者完成的 epoch |
| host status word 5 | norm 的 Y 等待迭代数（profile） |
| host status word 6 | Exact 路径 Q 上行 ready（host status 区） |
| host status word 7 | lo 的 Q 等待迭代数（profile） |

### Q Sidecar 内存布局协议（已统一）

为彻底消除与主 Relay Header 中 Y 状态的高频跨 Cache-line 伪共享，Aggressive Q8 路径（`late_q8_fast`）为 Q 分配了独立的 64 字节控制区：
* **控制区物理基址**：$B + 64 + L$（64 字节对齐，紧跟在主 Y 负载之后）。
  * `word 0`（偏移 0 字节）：Q sidecar ready 标志（GPU 写入 `1u`，CPU 轮询等待）。
  * `word 1`（偏移 4 字节）：Q8 workgroup 完成计数器（GPU `atomicAdd`，最后一个工作组置位 word 0）。
* **Q Payload 起点**：$B + 64 + L + 64 = B + 128 + L$。
* **各端对齐保证**：
  * GPU 生产端（`tp5_hc_latebind.comp`）使用 `late_word_offset = (128 + L) / 4`，写控制区于 `late_word_offset - 16u/15u`，写负载于 `late_word_offset`；
  * GPU 消费端（`tp5_hc_resume.comp`）LO 算子读取 `inbox[late_word_offset]`（即 $B + 128 + L$）；
  * CPU 重置（`tp5_relay_arm_bank`）、预武装检查（`tp5_relay_ensure_armed_epoch`）、就绪轮询（`allreduce_stages_internal`）统一对准 $B + 64 + L$ 控制区；
  * CPU 归约源读取与广播写回目标（`late_ptrs[i]`、`late_bcast_f16[i]`）统一对准 $B + 128 + L$，不再覆盖控制区。

### Exact Fallback 路径与 F32-Q RAW 修复

在 Exact Fallback 路径（`!late_q8_fast`）下：
1. **控制与负载协议**：
   * 控制字位于系统 host-imported RAM 的末尾状态区（`status[6]`）；
   * CPU 从系统内存 `star_host_aligned` 读取输入，归约结果广播写回 $B + 64 + L$；
   * LO 消费端 push constant `late_word_offset` 为 $(64 + L) / 4$，对应 $B + 64 + L$。
2. **F32-Q RAW 依赖修复**：
   * 在原有命令编排中，`late_norm` 在 `late_q` 之前发射，导致 `norm` 将 canonical y 写回 `trefs[i].buf` 冲掉了 `late_q` 尚未读取的 `z_p`（同一缓冲区）；
   * 修复后命令流严格重排：`emit(pre, ...)`（包含 scatter、`late_q` 读取 `z_p` 并输出到 sidecar、发布到 host-import）**严格先于** `emit(p2, 0, norm_end)`（`late_norm` 写回 canonical y）；
   * 在两者之间显式插入针对张量缓冲区的 `VK_ACCESS_SHADER_READ_BIT` 到 `VK_ACCESS_SHADER_WRITE_BIT` 执行与内存屏障，彻底消除 RAW 冒险。

CPU 仍然是原有 handoff 循环：Y 收齐后归约、fanout，并立即发布 word 0；随后收齐 Q，归约、fanout，再发布 word 2。不再让已经完成的 Y 等待 Q。

GPU norm 只读 Y generation，lo 只读 Q generation。norm **不写完成 epoch**；lo 读完两份 payload 后才写 word 3。bank 回收仍需真实的后续 producer/P1 发布所建立的传递完成关系。尾部和销毁使用实际已提交 timeline 值有界排空。

scatter 与 inverse-rho 分开分配，不再覆写同一四标量 scratch。提前 scatter 不再只看 scratch 地址：definition 同时保存 normalized tensor 与下一 HC inject-input tensor 的语义身份，只有**同一模型 tensor 且 buffer/offset/size 也吻合**时才前移。Q 输入 barrier 留在 terminal producer 之后，不能随着 scatter 一起前移。

## 消费者实现

`tp5_hc_resume.comp` 提供两个入口：

* `tp5_hc_resume_norm`：4 个 256-thread 工作组，逐 stream 做 reassociated RMS tree。先读 residual，首次使用 Y 时由一个 lane 有界轮询；直接消费 inbox，融合原 P2 copy 与 combine/norm。gamma 在 RMS 后再读，减少跨 reduction 的寄存器驻留；为其他潜在读者顺便保留 canonical Y 写回。
* `tp5_hc_resume_lo`：一个 256-thread 工作组，rho 已可读，首次使用 Q 时才轮询；320 个 SiLU 由每 lane 1–2 个输出完成，后面继续原生 W_up/fold。
* aggressive Q8 path：80 个 128-thread workgroup（R320 时每组 4 row、强制 4×wave32）完成 Q8dot；每组先写 local F32 sidecar 并递增 `status[8]`，最后完成的 workgroup 自己把 sidecar 转为 F16 并写 host-import，然后发布 `status[6]`。因此 fast path 没有单独 publisher dispatch，也没有 Q→publisher dispatch barrier。host sidecar 为 2.5 KiB/rank；CPU 使用现有 F16C/AVX2 五路归约并 fanout F32 给 LO。
* F32-Q fallback：`tp5_hc_publish` 仍作为一个 64-thread publisher，将 4×rank F32 sidecar 从 local VRAM 写入 host-import；同一 workgroup 的 lane 0 发布 `status[6]`。外部只保留一次 COMPUTE→HOST barrier，而且 memory scope 只覆盖 sidecar 与 64B status 两段 host-import range。

本轮仍选择“单独 lo consumer + 原生 W_up/fold”，没有把 320 次公共 SiLU 复制到每个 W_up 输出组。也没有把 norm 各组之间的依赖替换成设备级自旋栅栏；保留必要的计算依赖。

aggressive Q 明确允许改变 reduction tree、row grouping 和量化语义：activation Q8_0 只形成一次，320 个 row 被重组为 80 个 4-row workgroup；每个 row 的四个 stream accumulator 也在同一 wave 中连续处理。RMS consumer 从 512 lane/stream 改为 256 lane/stream，production width=2560 时每 lane 处理 10 个元素；LO 从 64 lane 改为 256 lane。LateBind 时 R320 W_up fixed-tail pipeline 自动启用。旧共同-generation `late_norm/late_lo/late_finalize` shader 变体及其 pipeline/descriptor 资源不再生成或分配。

默认路径已经不是 exact numeric path：Q8 activation grid、F16 sidecar 以及 RMS/Q reduction reassociation 都会引入数值差异。必须用中间张量、logits、长解码和多请求稳定性验收；一次文本相同不能替代误差检查。

## 资源寿命

模型定义持有 immutable descriptor pools、pipeline 和实际使用的 buffers，包括不在 ggml 图节点中的转换权重/临时 buffer。生成的新 primary 拥有独立 command pool，并继续持有模型定义。

定义重建时，旧 primary 不立即释放；在原生排空已提交工作后再释放。清空或淘汰 transport plan 前，先排空实际提交值并释放引用这些 plan 的 primary。发生 partial-submit 或 drain 失败则保留资源，不能仅依据逻辑 `allreduce_calls` 释放。

连续重建积累到 8 份退休定义时，会在下一次定义前原生有界排空并回收，避免关闭 chain reuse 或频繁变形导致内存无限增长。正常暖态 48/96-stage 路径不经过这个重建分支。

## 统一构建后的观察点

定义期应打印每 rank 一行：

```text
[tp5-linear-definition] rank=... stages=... primary_cbs=1 late=...
                        q8_fast=... scatter_early=... overlap=norm-q sidecar_pub=fused
                        dispatches=... barriers=... copies=... timing=...
```

`late` 是实际替换数量；不能只凭环境变量认定已启用。`scatter_early` 说明多少处证明了输入绑定一致并提前计算。没有命中的 stage 保持原数学路径。

诊断时现有 profile 另输出：

```text
[tp5-latebind-profile] sidecar_wait_us=... sidecar_data_us=...
                       y_publish_us=... q_publish_us=...
                       q_spin_avg=... q_spin_max=... q_spin_n=...
                       cpu_data_excludes_wait=1
```

`cpu_data_us` 不再包含等 Q 的耗时。Q GPU 自旋只对完成的样本计数，未就绪的末尾样本不以零补齐。GPU timing 在同一 primary 内写 timestamp，不再插 marker CB；插桩数字不作为无插桩吞吐成绩。

先使用 OMP 恢复的同一启动参数和 171-token chat request 对照 OFF/ON；P1、MMVQ、replicate、五卡时钟都不同时变动。需分别验收正常返回、Q 延迟/缺失的有界退出、第二次请求、bank 重用、图/工作区重建、中间张量/logits，再讨论性能。不要因这份源码完成而跳过安全门或扩大 spin budget。
