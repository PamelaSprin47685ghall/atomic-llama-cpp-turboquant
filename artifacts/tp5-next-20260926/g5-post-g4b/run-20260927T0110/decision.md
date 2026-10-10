# G5：post-G4b 账本与首验图重建调查——归因完成，图保留路线关闭（2026-09-27，run-20260927T0110）

## 本轮做了什么

1. **G4a latent double-draw 清理**：`common_sampler_sample` backend 快路径的
   `llama_sampler_dist_draw` 循环已移除（`llama_sampler_dist_backend_set_input`
   已推进 RNG，再画一次会双倍推进非 spec backend slot 的 mt19937 流）。
   重建 + 回归 4/4 通过 + 默认路径 smoke 2/2 exact 171/171。
2. **post-G4b 三张账**（`GGML_TP5_VERIFY_SUBTIMERS` + G4b 开启，4 请求）：
   - verify 窗口 sample 4.36→**2.34 ms/cycle**（G4b 生效；残差 = 9 行 ×
     `llama_get_logits_ith`（synchronize+reorder+行指针）+ 裸 argmax 扫描
     248320 词表 ~0.2ms/行 + readback ~0.75ms/cycle）。
   - steady cycle（n=8）：draft 16.61 / target 70.52 / catchup 0.96 / total
     **88.09 ms**（G2 时 102.7 → G4b 后 88.1）。
   - 请求分解：eval 1886 ms ≈ 19 × 87.1（1655）+ **242 ms 首验尖峰**
     + prompt 1065（steady，workload 固定）。

## 242 ms 首验尖峰归因（修正 G2 结论）

G2 曾把 ~300ms 尖峰归因为"31-token prompt 链内重评估（elems=69120）——
workload 固定成本，不立项"。exec 级分解显示该归因**错了一半**：尖峰 cycle 的
target_us=312 ms 中，GPU 计算无异常（每行 8.5ms≈稳态），大头是 **CPU 侧**：

- prompt（31 行）→ verify（9 行）形状切换：`allow_reuse` 因 n_tokens 不同
  双向失败 → `res->reset()` → **`build_graph` 重建 ~175 ms**（纯 CPU 张量图
  构造，36 层 × 每 layer 数十 op）；
- 新 `definition_uid` → meta 层 `needs_rebuild` → 97 子图 × 5 rank 重建 +
  全量 validation submit（156 ms vs 缓存路径 61 ms）；
- 首链 ready_us 124 ms（vs 稳态 56）。

每请求一次，3/4 请求支付（首个请求在启动时已付）。

## 为什么不可修（本轮实证）

尝试了 alt-slot 图保留（`GGML_TP5_GRAPH_ALT_SLOT`）：跨图保留
`llm_graph_result`、rebuild 前换出保活。**三次崩溃，三次不同根因**：

1. `GGML_ASSERT(backend_h != nullptr)`（t_h_nextn）：swap 后图张量与 sched
   分属两代，`get_tensor_backend` 返回 null；
2. `pre-allocated tensor (node_53) … FLASH_ATTN_EXT`：旧图张量带 stale
   `tensor->buffer`，sched split 视为 pre-allocated 而 abort；
3. 清 stale buffer 后 `GGML_ASSERT(buffer)`（`ggml_backend_buffer_is_host`
   对 split input null buffer）：split 输入的 buffer 归属被清空破坏查询。

**根因是 galloc 所有权模型**：`ggml_gallocr_init_tensor` 对 `data!=NULL &&
buffer!=NULL` 的张量**跳过重分配**——旧图张量的残留地址可能已被另一图的
分配覆盖；清掉又破坏 split 输入归属。跨图复用张量分配需要 galloc 层多图
分配池，超出单工单范围。

唯一安全的消除路径是 prompt/verify 共享 predefined capacity 图
（capacity_rows=31 覆盖 9 行）——但 `evaluate_target_capacity_admission`
对 GDN/recurrent 模型 **fail-closed**（inactive rows 的 GDN 状态会写脏，
AGENTS §7.1 语义），且这是 target-capacity 家族的重启门，非本工单可开。

**裁决：242 ms 首验尖峰确认为 workload 固定成本**（每请求一次
prompt→verify 形状切换 + GDN 模型 capacity fail-closed）。G2 结论数字修正
（300→242 ms，GPU→CPU 主导），裁决维持：不立项。alt-slot 实验代码全部
revert，无残留。

## 当前状态

- 基座（n8 + G4b）：**1886.4 ms（90.4 tok/s）**，距 1710 门 ~176 ms。
- 剩余缺口成分：draft 16.6×19=316 ms（D1 互斥）、prompt 1065 ms
  （workload）、首验尖峰 242 ms（本轮关闭）、verify 稳态 70.5×19=1340 ms
  内含 relay P2 自旋 24 ms/链（G3a 关闭）、readback+扫描 ~14+2 ms
  （G4c 架构门）。
- **全部大项已映射到关闭家族或 workload 固定成本。无新的可立项候选。**

## 证据

- `server-diag.log`（4 请求 sub-timer + cycle 账）、`diag.json`（4/4 exact）
- `server-alt.log` ×3（三次崩溃栈）、`server-default.log` +
  `default-smoke.json`（revert 后默认路径 2/2 exact）
- 源码：`common/sampling.cpp`（double-draw 清理保留）；llama-context
  alt-slot 改动已全部 revert（含误删后恢复的 D1 token 成员声明）
- 回归：test-sampling/mtp-workspace/tp5-plan/arg-parser 4/4
