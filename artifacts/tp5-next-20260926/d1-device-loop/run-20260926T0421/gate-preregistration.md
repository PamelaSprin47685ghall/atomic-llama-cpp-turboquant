# D1 生死门预注册与历史证据对照

日期：2026-09-26。run-id：20260926T0421。

## 历史证据核对（/var/tmp/tp5-mtp-100-20260924、-20260925）

| 机制 | 证据 | 结果 | 对 D1 的含义 |
|---|---|---|---|
| `GGML_TP5_MTP_DEVICE_HIDDEN=1` 五卡 smoke | `device-hidden-five-smoke.json`（head b4e87a01c dirty） | **SMOKE_PASS，2474.427 ms / 69.11 tok/s，全文精确** | 设备 hidden 通道（SEED/RESULT→INPUT copy_ranges）功能正确，可承载两步 hidden；但该 smoke 是 n=6 全 horizon，非两步原型，且无 ABBA 资格 |
| host-hidden unrolled 两步链 | `device-chain-two-exactness.json` + `withdrawn-mtp-chain-repair.json` | 正确但 **3488.112 ms**；撤回 | 撤回原型 hidden 仍逐 token 回 CPU 再物化；失败点是 host 物化+unrolled 图新增同步，不是设备 hidden 通道本身 |
| 普通 n=2 同口径 | `post-chain-withdrawal-smoke.json`（draft_n=114 → n=2，57 cycles） | **3019.149 ms** | 生死门对照数：两步原型必须优于同 horizon 现状 host 往返实现 |

## D1 两步原型设计（与撤回原型的机制差异）

1. **不建 unrolled 大图**。两步 = 同一张 predefined MTP capacity 图执行两次，第二次的
   token/hidden 输入由设备侧填充：
   - 第二步 token：第一步 `t_sampled`（greedy argmax，I32，已在设备）→ 设备侧写入第二步
     `inp->tokens` 行（或经设备 copy_ranges）。
   - 第二步 hidden：第一步 `t_h_nextn` 行（F32，hc_dim 宽，已在设备）→ 既有
     `ggml_backend_device_copy_ranges` 通道（`llm_graph_input_embd_h::set_input` 的
     `ubatch->device_hidden` 分支）直通第二步 `inp->h`。
2. **唯一 host 参与** = cycle 边界 accept/commit 与第一步已提交 token。
3. **不改**：catch-up（K/V-only）、sampler 契约（greedy backend chain、p_min<=0）、
   KV/pos/mask 语义（每步仍按原单步 metadata 执行）。
4. 契约边界（第一版）：单序列、单 MTP head、非共享 KV、greedy backend sampling。
   不满足即回退现状路径，不修改状态。

## 生死门判据（预注册）

- 原型：n=2、同权重/context/sampler/放置，第二步 token+hidden 不回 host。
- 对照：现状 n=2（每步 decode→argmax 回读→hidden 回读→batch 重建）。
- 通过：两步完整请求 predicted_ms 优于对照（配对，噪声内视为未检出即停），
  且新增设备 copy/同步开销 < 节省的往返成本。
- 失败：不扩六步、不切五卡、不保留原型为默认；归档后回退。
- 正确性：全文精确、171/171、自然 stop；accepted 覆盖 0/1/2；EOG；取消后下一请求正确。

## 本轮状态

- 输入表已按 §8.2 交付（`input-table.md`，全部源码行号证据）。
- 原型实现属生产代码改动；本轮已完成 G0/G1/K1a 的 GPU 实验串行窗口，D1 原型
  实现与生死门测量为下一实验窗口的第一项（手册 §3：GPU 实验串行；K1a 五块
  ABBA 已占用本窗口）。
- 未实现即未通过生死门；不宣称任何 D1 收益。
