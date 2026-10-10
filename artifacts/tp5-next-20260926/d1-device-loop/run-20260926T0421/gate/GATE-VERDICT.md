# D1 生死门裁决（gate 结果）

日期：2026-09-26。run-id：20260926T0421。二进制：candidate-k1a build3
（16e28b8ac + K1a shader + cstddef + D1 v1 设备 token 机制）。
测量：手动 server（同 harness 命令行），n=2，canonical `1..60`，warmup+2 稳态。

## 实测数据

| 臂 | 配置 | draft 布局 | sampler | 稳态 predicted_ms | exact |
|---|---|---|---|---:|---|
| A | 默认 n=2（host 往返） | 单卡 Vulkan0 | greedy backend chain（GPU argmax + 4B 回读） | **2677.5 / 2685.5** | ✅×3 |
| B | `GGML_TP5_MTP_DEVICE_HIDDEN=1` n=2 | 五卡 tensor-split | CPU sampler（`set_sampler` 拒绝 SPLIT_MODE_TENSOR） | 3032.1 / 3181.6 | ✅×3 |
| C | A+B+`GGML_TP5_MTP_DEVICE_TOKENS=1` | 五卡 | — | **fail-closed**：`device MTP tokens require one sequence with a greedy backend chain` | — |

## 结构性结论（源码证据）

D1 两步设备 token 闭环在融合基座架构下**不可达**，不是实现缺陷而是架构互斥：

1. **Vulkan 设备 copy 只支持同物理设备**（`ggml_backend_vk_device_copy_ranges`：
   `c.src->device != ctx->device → return false`）。target 的 hidden RESULT 分布在五张卡上；
   单卡 draft（Vulkan0）无法设备直连接收 → device hidden 通道**要求 draft 也五卡**
   （server gate `matching_ranks`，common.cpp:462-469，历史 five-smoke 同配）。
2. **五卡 draft = tensor split → `set_sampler` 拒绝 backend chain**
   （llama-context.cpp:2245-2253：`backend sampling not supported with SPLIT_MODE_TENSOR`）。
   无 backend chain → draft 图无 `t_sampled` 设备张量 → 设备 token 无来源。
3. 单卡 draft（split NONE，backend chain 可用）与 device hidden（要求五卡 draft）互斥；
   五卡 draft（device hidden 可用）与 backend chain（设备 token 的前提）互斥。

## 生死门判据对照（§8.3 预注册）

- 判据"两步完整请求不优于同 horizon 对照，或节省的边界成本小于新增开销，就停"：
  **触发**。可达的最接近原型（arm B：五卡 draft + device hidden，hidden 已不回 host）
  稳态 3032–3182 ms vs 对照 arm A（单卡 draft host 往返）2678–2686 ms——**慢 13–19%**。
  五卡 draft 的执行开销远超省下的 hidden 回读；与手册 §2.2"五卡 draft 已有负结果"一致。
- "不扩六步、不切五卡"：本裁决未扩六步；五卡 draft 是 device hidden 的强制前提而非选择。

## 裁决

**D1 两步设备闭环原型不通过生死门，撤回，不保留为默认。**

- 机制实现（设备 token slot + decode_hidden 扩展 + set_input 设备分支）已通过全部既有
  回归（CPU 套件 11/11 + GPU 定向 skinny/argmax/readback/mesh/GDN），fail-closed 行为
  在不适配组合下正确拒绝（arm C 验证）。改动保留在 candidate worktree 供后续架构
  变化后重启，不进入 control/默认路径。
- 重启条件（比 §2.2 更强）：需要以下之一成立——
  1. Vulkan 跨设备 copy（或 host-pinned 中转）使单卡 draft 能设备直连五卡 target 的
     RESULT 分片；或
  2. tensor-split 模型支持 backend chain（`t_sampled` 在 meta 分片下可物化）。
  在此之前，draft 的 320 ms/请求归因项只能通过其他机制（如 target verify 内的
  结构改造）间接压缩，不能靠消除 draft 高层往返兑现。

## 与历史失败档案的对照

- host-hidden unrolled chain（3488 ms）：unrolled 图 + host 物化——本原型未重试该路线
  （无 unrolled 图、hidden 走设备通道），机制差异成立，但被架构互斥挡在更早处。
- 本裁决不推翻 G1 的 draft 320 ms 归因；它说明该归因项在当前架构下**不可收**，
  下一轮应回到 producer 表的 stage 0–9（K1b 范畴）或 target verify 结构。

## 证据

- `arm-A-n2-baseline.json`、`arm-B-hidden-only.json`（本目录）
- `server-A-n2-baseline.log`、`server-B-hidden-only.log`、`server-C-combined.log`
- 源码行号：ggml-vulkan.cpp:22646-22680（同设备限制）、llama-context.cpp:2245-2253
  （split 拒绝）、common.cpp:449-469（draft 布局与 device hidden gate）、
  speculative.cpp:2923（单卡 draft split NONE）
