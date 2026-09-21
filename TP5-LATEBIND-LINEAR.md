# LateBind：连续命令流与按需等待

本页对应 2026-09-21 的源码改造。**未编译、未运行 shader、模型或性能测试**；其他工作同时进行，由用户统一构建、验收。不能据此宣称已经恢复或超过 52.47 tok/s。

## 执行入口

保留现有 `GGML_TP5_LATEBIND=hc-down`，只用于 RELAY/F32。`GGML_VK_DISABLE_PRODUCER_WIRE=1` 仍然保留 P1，attention replicate 仍独立可开关。

```bash
export GGML_TP5_LATEBIND=hc-down
export GGML_TP5_WIRE=f32
export GGML_VK_DISABLE_PRODUCER_WIRE=1
unset GGML_TP5_LATEBIND_FUSED_FINALIZE
# GGML_TP5_REPLICATE_ATTN 单独选择，不由 LateBind 改写。
```

旧 `GGML_TP5_LATEBIND_FUSED_FINALIZE=1` 会明确报错。它对应共同 generation 的旧 1024-thread 等待流程，不能与新的 Y/Q 独立 generation 混用。

OFF 路径不收集命令 IR，不分配 sidecar，也不为 F32 producer-wire 建立 HC 候选切口。F16 原有的显式 fused-consumer 路径单独保留。

## 命令定义与运行

`ggml-vulkan-tp5-command.h` 在定义期保留必要的 Vulkan 指令参数、descriptor pool、pipeline 和 buffer 所有权。指令序列包含 HC norm 完成、HC down 完成两个**语义位置**，不是额外的 Command Buffer。

普通冷图仍按原数学图执行。只有完整定义可用时，collective 才把所选方案直接录入每 rank 的一个 primary CB：

```text
前一 stage 的 norm_resume：准备 residual/gamma → 等 Y → combine/RMS/norm
    ↓
当前 stage 的 scatter（仅在输入绑定吻合时提前计算）
    ↓
前一 stage 的 lo_resume：准备 rho → 等 Q → SiLU/lo
    ↓
原生 W_up/fold + 后续 attention/MoE/terminal producer
    ↓
P1（若使用 P1；direct 路径在 producer 内发布 Y）
    ↓
Q 预计算 → 本地 VRAM sidecar → 向量上行 → Q ready
    ↓
下一个 stage
```

新 primary 中没有 `vkCmdExecuteCommands`，不按“跳过两个 CB”拼接，不在每个 token 上解释 IR。定义期仍保留常规 transport CB 用于原有非整链执行；LateBind 整链不提交这些片段。

定义不完整、出现不可安全保留的事件/图像依赖、源 CB 发生变化时，整链在提交前明确失败，不静默漏录命令。普通模型 fallback 与 LateBind 替换以完整 stage 为单位，五卡资格必须一致。

## 两种 generation

每 bank 的 GPU inbox 前 64 字节仍为控制区：

| 位置 | 新含义 |
| --- | --- |
| inbox word 0 | Y generation |
| inbox word 2 | Q generation |
| host status word 1 | 本 bank 预定 epoch |
| host status word 2 | sticky failure |
| host status word 3 | 最后 inbox 读者完成的 epoch |
| host status word 5 | norm 的 Y 等待迭代数（profile） |
| host status word 6 | Q 上行 ready |
| host status word 7 | lo 的 Q 等待迭代数（profile） |

CPU 仍然是原有 handoff 循环：Y 收齐后归约、fanout，并立即发布 word 0；随后收齐 Q，归约、fanout，再发布 word 2。不再让已经完成的 Y 等待 Q。

GPU norm 只读 Y generation，lo 只读 Q generation。norm **不写完成 epoch**；lo 读完两份 payload 后才写 word 3。bank 回收仍需真实的后续 producer/P1 发布所建立的传递完成关系。尾部和销毁使用实际已提交 timeline 值有界排空。

scatter 与 inverse-rho 分开分配，不再覆写同一四标量 scratch。提前 scatter 只在 `normalized` 与目标 inject input 的 buffer/offset/size 全部相等时进行。Q 输入 barrier 留在 terminal producer 之后，不能随着 scatter 一起前移。

## 消费者实现

`tp5_hc_resume.comp` 提供两个入口：

* `tp5_hc_resume_norm`：4 个 512-thread 工作组，沿用逐 stream RMS 树。先读 residual/gamma，首次使用 Y 时由一个 lane 有界轮询；直接消费 inbox，融合原 P2 copy 与 combine/norm。为其他潜在读者顺便保留 canonical Y 写回。
* `tp5_hc_resume_lo`：一个 64-thread 工作组，rho 已可读，首次使用 Q 时才轮询；生成 lo，后面继续原生 W_up/fold。

本轮选择“小 lo consumer + 原生 W_up/fold”，没有把 320 次公共 SiLU 复制到每个 W_up 输出组。也没有把 norm 各组之间的依赖替换成设备级自旋栅栏；保留必要的计算依赖。

现有向量化 Q、local-VRAM sidecar、128-bit 上行 copy 保留。Q/inject 的 subgroup 归约补齐多 subgroup 情况；单 subgroup 时仍走直接 subgroupAdd。

Exact 指代线性分解，不保证 F32 位级等价。归约重排、乘加次序和条件数仍需中间张量与 logits 验收；文本相同不替代这些检查。

## 资源寿命

模型定义持有 immutable descriptor pools、pipeline 和实际使用的 buffers，包括不在 ggml 图节点中的转换权重/临时 buffer。生成的新 primary 拥有独立 command pool，并继续持有模型定义。

定义重建时，旧 primary 不立即释放；在原生排空已提交工作后再释放。清空或淘汰 transport plan 前，先排空实际提交值并释放引用这些 plan 的 primary。发生 partial-submit 或 drain 失败则保留资源，不能仅依据逻辑 `allreduce_calls` 释放。

连续重建积累到 8 份退休定义时，会在下一次定义前原生有界排空并回收，避免关闭 chain reuse 或频繁变形导致内存无限增长。正常暖态 48/96-stage 路径不经过这个重建分支。

## 统一构建后的观察点

定义期应打印每 rank 一行：

```text
[tp5-linear-definition] rank=... stages=... primary_cbs=1 late=...
                        scatter_early=... dispatches=... barriers=... copies=... timing=...
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
