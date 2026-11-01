# RouterEngineer notes (isolated; not built, not referenced by CMake)

Owner scope per Main (2026-10-31): production sources only — `ggml/src/ggml-alloc.c`,
`src/models/qwen4exp.cpp`, `ggml/src/ggml-cuda/topk-moe.cu`. Examples/verifier/CMake/
test docs belong to RouterFixture. Nothing here is compiled or executed by me.

Status of the candidate: **PARTIAL / UNPROVEN**. No build, no run, no test.

## 1. What this candidate does

`GGML_TP5_MOE_ROUTER_SCRATCH=1` (exact) makes `src/models/qwen4exp.cpp` mark the
direct F32 MoE router projection (`ffn_moe_logits-<il>`, softmax `src[0]`) of every
layer whose token count is in `[2,4]` with `GGML_TENSOR_FLAG_NO_REUSE`
(`:1502-1505` window, `:1524-1545` marking + fail-closed validation).

`ggml/src/ggml-alloc.c` then provides the lifecycle:

| Step | Code | Effect |
|---|---|---|
| plan capture | `ggml_gallocr_reserve_n_impl` `node_alloc->dst.no_reuse` / `src[j].no_reuse` | the saved plan records the retention contract |
| in-place veto | `ggml_gallocr_allocate_node` | a retained tensor (or a tensor whose `view_src` is retained) is never an in-place parent |
| lifetime | `ggml_gallocr_free_node` | retained allocations are not freed during the graph |
| plan validity | `ggml_gallocr_node_needs_realloc` | a saved plan whose flag state differs from the live graph forces re-reserve |
| plan audit | `ggml_gallocr_check_no_reuse_plan` | `GGML_TP5_ROUTER_SCRATCH_AUDIT=1` (exact): whole node+leaf plan; any overlap outside a recognised in-place continuation fails the reserve |

The consumer side needs no exception: `ggml/src/ggml-cuda/ggml-cuda.cu:4467`
(`ggml_nrows(...) == 1 && is_topk_moe`) is untouched, and with the projection
guaranteed unshared `ggml_cuda_check_fusion_memory_ranges` passes for `nrows > 1`
on real pointers. No `ggml_set_output` is used anywhere in this change.

## 2. Evidence lines (allocator, production reserve path)

```
retention audit clean (reserve): N retained allocations, B bytes, rows=R, nodes=M
retention audit enabled (reserve): 0 retained allocations, nodes=M
```

- `(reserve)` = `ggml_gallocr_reserve_n` (the plan that will be executed);
  `(size)` = `ggml_gallocr_reserve_n_size`, a measurement that allocates nothing.
- `rows=` is the common `ne[1]` of the retained projections, `mixed` on disagreement.
- Violations print `retained tensor '…' overlaps '…'` or
  `retained tensor '…' is a view, it cannot be retained` and make the reserve fail.
- Log level is `GGML_LOG_INFO` deliberately: `DEBUG` would hide it in release builds.

**Object audited is a plan, not an execution.** A decode that reuses an
already-reserved plan prints nothing. There is currently **no** existing owner-side
route that ties an audited reserve to the graph `llama_decode` submits
(`gf_res_prev` is private, `gf_res_reserve` is sizing-only, `reserve_n_size` shares
the entry), which is why the gate stays UNPROVEN rather than PASS.

## 3. Withdrawn designs (do not resurrect without Main)

- Scheduler `cb_eval` observer: unavailable on the `tp5_relay` path (llama rejects a
  non-null `cb_eval`), disables the single-split `defer_input_flush` fast path and
  synchronizes per batched range (timing change), and only sees `split->graph`
  nodes, never `sched->graph` input copies. Kept in `examples/` (RouterFixture's
  file) as triage only, `promotion_evidence=0`.
- Owner-declared gallocr phase (`inputs_set`/`inputs_unset`) plus a public
  `ggml_backend_sched_get_galloc`: reverted by Main as a test-only API carrying an
  unproven identity claim. Verified absent from the tree.

## 4. What must be true before this can pass

1. A build of `ggml-alloc.c` / `qwen4exp.cpp` / `topk-moe.cu` (Devops).
2. `retention audit clean (reserve)` with `N = 2 * rows` on the decode reserve.
3. OFF leg: `enabled (reserve): 0 retained allocations`; no `clean` line.
4. N1 legs bitwise unchanged vs stock (the `nrows==1` exception must still be intact).
5. GPU: `topk_moe fused n_rows=3 … no_reuse=1` (fusion consumed the scratch).
6. An owner-side route proving the switch moves the fused path, and no measured
   regression. Until then: PARTIAL, never PASS.

---

# 审计 fail-closed 收口（Reviewer 三项，仅 `ggml/src/ggml-alloc.c`）

hash: `ggml/src/ggml-alloc.c` = `c929aaf59e6da7cb`（前序 `71714f4cb667eaec` / `0ab3db12e60173db` / `4c981b5288a6ea80`）。

## 1. retained view 的违规分支现在可达
此前 `kept->buffer_id < 0` 的守卫在 view 检查之前 `continue`，导致
`retained tensor '…' is a view, it cannot be retained` 永远不可达、违规被静默接受。
现在先判 `node->view_src != NULL` 并直接 `return false`。

## 2. 外部分配 / 预分配张量：明确判 UNPROVEN 并拒绝，不用无关 buffer_id 猜地址
`ggml_gallocr_audit_extent()` 只对「本计划描述过的分配」返回 extent：
`buffer_id < 0`、`t->data != NULL`、`size_max == 0` 一律返回 0（不再用 `get_alloc_size(t)` 反推）。
- 作为 **retained**：`return false` 并打印 `is outside the audited plan (buffer_id=… size_max=…)`。
- 作为 **邻居**（node 或 leaf）：计入 `n_unverifiable` 并跳过该邻居的区间比较；
  循环结束后若计数非零，打印 `retention audit unproven (…): N allocation(s) could not be compared …`
  并 `return false`。
  → 语义：**只要 retained 无法与某个分配证明不相交，整张计划被拒（UNPROVEN）**，而不是打出 clean 行。
- retained 自身 extent 为 0 的兜底路径也改为 `return false`（`has no auditable extent`）。

## 3. 明确审计的影响面（不是纯观测）
函数头注释写明：`GGML_TP5_ROUTER_SCRATCH_AUDIT` 仅 exact `"1"` 生效；启用后一旦判定违规，
`ggml_gallocr_reserve_n` 返回 false（sched reserve 失败 → decode 报 `ALLOC_FAILED`），
`ggml_gallocr_reserve_n_size` 的 `GGML_ASSERT` 会 abort。**启用即可能拒绝本可运行的计划**；
关闭时不做额外计算或分配，OFF ≡ stock 不变。

## 4. 仍未覆盖（明示）
- Meta backend 二次切分的内部子图、relay 多通道子图：本审计只看 `sched->graph` 的 node+leaf 计划。
- 融合 kernel 内部 dispatch：需 GPU 侧 `topk_moe fused n_rows=… no_reuse=1`。
- 「审计的 plan 即 decode 提交的图」：join key = `(graph_uid, plan_id)`。allocator 在证据行
  `retention audit … (reserve) … graph_uid=… plan_id=… fp=…` 里写下它审计的那份计划；`llama_context`
  在 `graph_reserve` 记录 scheduler 盖章的 uid 与 allocator 的 `plan_id`/`fp`，在 `graph_compute`
  对「以 retained inputs 提交」的图打印 `router_scratch plan identity … match=<0|1>`。C4b 要求
  `match=1` 且两侧 `graph_uid`/`plan_id`/`fp` 相同；缺行、`match=0`、身份不一致仍 UNPROVEN。
  设计见 `PLAN-IDENTITY.md`。

## 5. 覆盖口径修正（domain-based，非 data-based）
上一版把「邻居带有自己的 data」也算作无法比较，会导致任何拥有权重/embedding leaf 的真实图在
`AUDIT=1` 下被判 UNPROVEN 并拒绝 —— 与 retention 无关的假失败。现口径：
- **域由 `buffer_id` 决定**（`buffer_id < 0` 或与 retained 不同 buffer ⇒ 不在本计划地址空间，
  比较是空的，跳过不算损失覆盖）；
- `audit_extent` 不再因 `t->data != NULL` 返回 0：同一计划被 replay 时，带 data 的节点仍在域内，
  必须继续比较；只有 `buffer_id < 0` / `size_max == 0` 返回 0；
- 仍属「真正未知」并计入 `n_unverifiable` 的只有一种：**同域（同 buffer_id）但 `size_max == 0`**；
  末尾打印 `retention audit unproven (reserve): N in-domain allocation(s) carry no recorded size` 并拒绝。
