# D1 decision

日期：2026-09-26。run-id：20260926T0421。

## 改了什么

D1 v1 设备 token 闭环机制（candidate worktree，非默认，opt-in `GGML_TP5_MTP_DEVICE_TOKENS=1`）：

- `llama_predefined_hidden_store` 增持久设备 token slot（1×I32，store buffer 所有）
- `decode_predefined_hidden` 接受 `batch.token==NULL`（设备来源；`token_device_sourced` 标记
  跳过 stub 校验）
- `decode_impl` 尾部把 `t_sampled`（greedy argmax 设备张量）异步镜像进 slot
- `llm_graph_input_embd_h::set_input` 新设备分支：token 设备到设备上传
- draft 循环 step-1 分支：token+hidden 全设备直连，无 `llama_decode`/`common_sampler_sample`
  高层往返
- gate：单序列、greedy backend chain、n_max==2；不满足抛错 fail-closed

## 实际命中什么 / 节省什么

**未达命中。** 生死门三臂实测（n=2 canonical，warmup+2）：

- A 对照（单卡 draft host 往返 + GPU argmax + 4B 回读）：2677.5 / 2685.5 ms
- B 可达原型（五卡 draft + device hidden）：3032.1 / 3181.6 ms（**慢 13–19%**）
- C 完整机制（+device tokens）：fail-closed（五卡 draft 无 backend chain）

架构互斥链（源码证据见 GATE-VERDICT.md）：Vulkan 设备 copy 仅同物理设备 →
device hidden 要求五卡 draft；tensor split 拒绝 backend sampling → 五卡 draft 无
设备 sampled 张量。两前提无交集。

## 新增多少成本

无新增常驻（slot 4 字节）；机制代码 ~370 行，全部 opt-in 且 fail-closed。

## 正确性与 fallback

全部既有回归通过（CPU 11/11 + skinny/argmax/rows/transfer/attn-projections/MoE/HC/
rope/GDN/mesh/readback `--vulkan`）；arm A/B 三请求全文精确、171/171、自然 stop；
arm C 在不适配组合下正确抛错拒绝，不污染状态。默认路径（env 未设）零行为变化。

## 完整请求结果 / 裁决

**不通过生死门，撤回原型，不保留为默认。** D1 的 320 ms/请求 draft 归因在当前架构下
不可收；重启条件：跨设备 Vulkan copy 支持，或 tensor-split 支持 backend chain。
改动留在 candidate worktree + 归档 diff，默认与 control 零影响。

## 下一步

G1 producer 表剩余可归因项：stage 0–9 的 rank compute 不均衡（12.9 ms/链 ready 等待）。
K1b 应针对这些 stage 的真实 shape 做 staging/行 tile 单因素实验——先取 stage kernel
明细账（`GGML_VK_PERF_LOGGER` per-op 时间）再立项。
