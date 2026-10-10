# K1e 裁决：配对专家 K=128 MoE down——死门修复后非劣通过，保持 opt-in

日期：2026-09-26。run-id：20260926T2000。工单：AGENTS §5.2 MoE 分组成员
（K1a 模式现成增量验收）。

## 发现与修复

`qwen4_moe_down_k128.comp`（两选中专家共享一个 wave32 的两个 16-lane 半波，
4 行 tile 320→160 线程）及 8 管线变体早已实现并接入，但 opt-in 从未验收。
**根因：管线创建门的 `device->subgroup_size == 32` 子句是死门**——RADV 在
RDNA2 上报 subgroupSize=64（gate 日志实证 ss=64），该门在唯一准入设备类
（AMD RDNA2+RADV）上结构性不可满足，路径从未创建过。作者混淆了 Vulkan
报告的 preferred size 与管线强制的 required full subgroup（create_pipeline
末参 32）。修复：删除该子句，保留能力检查（size_control、min≤32≤max、
require_full_support、arithmetic、shuffle、RDNA2、RADV、WG≥160）——
与仓库其它 wave32 路径同一判据。

## 数值契约（修复后实测）

半波 XOR 蝶形（步 1,2,4,8）独立归约每专家 16 partial；native 在 64-lane
波下为 16 partial + 48 零，步 16/32/48 合并恰加 0.0，实 partial 结合顺序与
16-lane 蝶形一致。fold 尾部 `precise` 有序专家和 + sigmoid 门控 + shared
融合镜像 native `MOE_DOWN_FOLD`。

## 验证链（全部 K128 开启）

- `--moe-output-only` PASSED（含相邻专家同波污染用例；命中日志
  `[k128-hit] m=256 shared=1` ×3）
- `--moe-only` PASSED；**全套 28/28**（clean DSO 复验）
- 真实五卡 smoke：6/6 精确 171/171 自然停止；命中 `m=2560 shared=1`
  （真实 width）；稳态 2036–2055 ms 正常带
- debug logger 已全部移除（DSO strings 零残留）

## ABBA 裁决（A=K128 off, B=K128 on, 同二进制, 5 blocks × 2 repeats）

|指标|A|B|配对统计|
|---|---:|---:|---|
|decode tok/s|83.492|83.527|ratio **1.0004** CI [0.9953, 1.0056] 非劣**通过**|
|committed tok/s|51.081|51.132|ratio **1.0010** CI [0.9936, 1.0084] 非劣**通过**|
|predicted_ms|2048.18|2047.31|-0.9 ms（噪声内）|
|正确性|20/20|20/20|40/40，5/5 blocks|

**预注册解读**：双指标非劣通过 → 合格成员；CI 含 1 → 不宣传提速。
与 K1c/K1d 同族同命：MoE down fold 的线程减半（320→160）不改变权重带宽
（仍全读 10 专家），收益被内存带宽主导的 kernel 吞没，端到端在噪声内。

## 处置

1. **死门修复保留**（这是 bug fix：原代码在目标设备上不可达）。
2. `GGML_VK_MOE_DOWN_K128=1` **保持 opt-in**：非劣但无检出收益，按
   "仅非劣可因正确性/维护价值保留"原则保留为显式变体
   `native-mtp-fused-k128`，不并入默认。默认 n=6 融合路径不变。
3. harness 变体 `native-mtp-fused-no-k128`/`native-mtp-fused-k128` 注册
   供后续复测。
4. 未 commit/push。

## 对 100 tok/s 门

无变化（~1 ms 噪声内）。MoE down 线程几何优化家族至此有实测结论：
带宽主导、无检出收益——后续 MoE 计算侧候选须先给出带宽模型外的证据。

## 证据

[MECHANISM.md](MECHANISM.md)、[abba5.json](abba/abba5.json)、
smoke/smoke-server.log（命中日志）、smoke/smoke.py、abba5_logs/（20 server 日志）
