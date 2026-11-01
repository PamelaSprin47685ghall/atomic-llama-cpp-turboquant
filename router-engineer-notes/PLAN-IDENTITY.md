# 执行图 plan identity（C4b）：(graph_uid, plan_id) 联合绑定

命题：**allocator 审计过的那份 plan，就是本次 decode 提交的图所用的 plan。**
链路上有三个对象，修复前没有任何一环被写下来：

| 环 | 对象 | 现状 |
|---|---|---|
| ① 审计对象 | `sched->graph`（`ggml-backend.cpp::ggml_backend_sched_split_graph` 组装） | 审计函数只拿到指针，证据行只有 `N/bytes/rows/nodes` |
| ② reserve 时的源图 | `graph_reserve()` 里的 `gf` | `split_graph` 给它盖 uid（`ggml-backend.cpp:1234`），llama 侧不保存 |
| ③ decode 提交的图 | `gf_res_prev->get_gf()` | 只用于 `retain_inputs` 判断，不打印、不比对 |

## 1. 为什么必须是 `(graph_uid, plan_id)` 而不是单个 uid

- `ggml_graph_cpy` 不拷贝 `uid`（`ggml.c` 只拷 nodes/leafs/order/hash），split 视图的 uid 也是 0，
  所以 `split_graph` 盖章之后**必须显式把 uid 同步给 `sched->graph`**（allocator 审计的正是这份
  copy）。否则 allocator 读到的 `graph->uid` 恒为 0（实测就是这样）。同步点是
  `ggml-backend.cpp` 里 `graph->uid = ggml_graph_next_uid()` 之后的那一行；`sched_reset` 里同时把
  `sched->graph.uid` 归零，避免 reset 之后读到上一张图的章。
- 同一个 `gf` 再次 `ggml_backend_sched_alloc_graph` 时 uid 不变；`alloc_splits` 若因
  `backend_ids_changed` 重新 reserve，allocator 会产出一份**不同的 plan**。只比 uid 会把
  「计划换了、图没换」判成 match=1。
- 因此 join key 是 `(graph_uid, plan_id)`；`plan_id` 每次 planning pass（`reserve_n` 与
  `reserve_n_size` 都算）自增，`fp` 再提供一重「节点序列相同」的交叉校验。

## 2. 实现（三处，默认路径零改变）

**(a0) 放置顺序（`ggml/src/ggml-alloc.c`，与审计开关无关）**

`ggml_gallocr_alloc_graph_impl` 的预分配循环原来只提前分配 `GGML_TENSOR_FLAG_INPUT`；非视图的
`NO_REUSE` 节点因此排到临时算子之后，拿到的是临时算子（`node_9`）刚释放的地址 —— dyn_tallocr 认为
合法，保留契约认为重叠，审计随即拒绝该 plan。现在预分配条件扩为
`INPUT || (NO_REUSE && !is_view)`（节点与其 src 都适用）。`ggml_gallocr_allocate_node` 的 in-place
分支不动：父张量已带 `OUTPUT|NO_REUSE` 的检查，且本代码库里 `NO_REUSE` 只标在拥有自己存储的
投影张量上（`qwen4exp.cpp`、L2 plan leg），把它加成「禁止一切 in-place」只会无谓增大计划。

**(a-1) 空张量不算「无法核验」（`ggml/src/ggml-alloc.c`）**

审计的 node/leaf 邻居循环里，`other->size_max == 0` 原先一律计 `n_unverifiable`。但真实 Qwen4Exp 解码图
（303 节点）里有合法的 0 元素张量（`ggml_is_empty` 为真：`states_extra` 在 `n_rs == n_seqs` 时形状
`[state_size, 0]`），它们的物理大小本来就是 0 字节、任何后端 kernel 都是
no-op，不可能与保留区重叠。现在只有「非空且 `size_max == 0`」才计入未知；`kept` 自身仍不允许为空。
不加这条，2 个保留张量 × 4 个空张量 = 8 次未知，计划被直接拒绝，clean 行永远不会出现。

**(a) allocator — `ggml/src/ggml-alloc.c`（只在 `GGML_TP5_ROUTER_SCRATCH_AUDIT=1` 下写）**

- `struct ggml_gallocr` 增加 `plan_id / plan_graph_uid / plan_fp / plan_n_nodes / plan_n_leafs / plan_entry_size`。
- `ggml_gallocr_reserve_n_impl()` 开头 `plan_id += 1` 并记录 `entry_size = no_alloc`：
  新 pass 一开就作废旧身份，被拒绝的 plan 也留下新 id，避免「旧 id 仍匹配」。
- `ggml_gallocr_check_no_reuse_plan()` 在出证据行前写入 `graph->uid`、`ggml_gallocr_plan_fp(graph)`、
  node/leaf 计数；证据行追加 `graph_uid=… plan_id=… fp=…`。
- 新增只读访问器 `ggml_gallocr_plan_identity()`（`ggml-impl.h`，内部头，不是公共 API）；
  `ggml_gallocr_plan_fp()` 同样是只读 FNV-1a（`graph->nodes` 按序 + node/leaf 计数）。
- 这一步不改变任何分配决策：字段在 plan 定案之后才写，且只在审计开启时写。

**(b) llama — `src/llama-context.h` / `.cpp`**

关键事实：decode 提交的 plan **不是** `graph_reserve` 产生的。产生 `plan_id` 的那次
`ggml_gallocr_reserve_n` 来自 `process_ubatch` 里的
`ggml_backend_sched_alloc_graph` → `alloc_splits`（`backend_ids_changed` 或 alloc 失败时）。
`graph_reserve` 只做 sizing reserve，它规划的图不是随后提交的图，所以它不记录任何身份。

- `llama_context::exec_plan { recorded, graph_uid, fp, plan_id, n_nodes }`，
  由 `record_exec_plan(gf)` 写入：在 `process_ubatch` 中
  `ggml_backend_sched_alloc_graph(sched.get(), gf)` **成功之后**，读 allocator 的
  `plan_id`/`n_nodes` 与 `gf->uid`、`ggml_gallocr_plan_fp(gf)`。
- `graph_compute()`：在 `ggml_backend_sched_graph_compute_async*` **返回之后**、`status == SUCCESS`、
  两个开关都 exact `"1"` 时打印一行，**不再限定 `retain_inputs`**（CPU fixture 上
  `backend_tp5_relay == nullptr`，那条限定会让证据永不出现）：

```
llama_context::graph_compute: router_scratch plan identity exec_uid=U exec_fp=F exec_nodes=M plan_uid=U plan_fp=F plan_id=P plan_nodes=M audit_plan_id=P audit_uid=U audit_fp=F recorded_plan_id=P recorded_uid=U recorded_fp=F entry_size=0 match=<0|1>
```

- `match=1` 需要同时满足：`exec_plan.recorded`、`plan_id != 0`、`plan.graph_uid != 0`、
  `plan.retained > 0`（当前 plan 真的带着保留的 router scratch）、
  `audit_clean_plan_id == plan_id`（allocator 现在持有的 plan 正是它审计通过的那一份）、
  `audit_clean_graph_uid/fp/n_nodes/n_leafs == plan.*`、`entry_size == 0`、
  `exec_uid != 0 && exec_uid == plan.graph_uid && exec_fp == plan.fp`、
  `plan.n_nodes/n_leafs == gf->n_nodes/n_leafs`、`exec_plan.* == plan.*`。
  任一条不满足即 `match=0`。
- 默认路径：`plan_identity_enabled()` 是 `static const bool`（首次调用读一次 env）；
  开关关闭时 `record_exec_plan` 立即返回，`graph_compute` 不读任何字段。

**(b') allocator 侧「审计通过」的记录 — `ggml/src/ggml-alloc.c`**

- `galloc` 增加 `audit_clean_plan_id / audit_clean_graph_uid / audit_clean_fp /
  audit_clean_n_nodes / audit_clean_n_leafs`，以及 `plan_retained`（当前 plan 的保留数）。
- 这些记录**不**在 pass 开头清零，而是保留到下一次被「带保留的审计通过」覆盖。理由是
  `plan_id` 每次 pass 自增：被拒/未审计的 pass 会让 `audit_clean_plan_id != plan_id` 自然成立，
  无需清空；若清空，反而会丢掉「当前装进 buffer 的那份 plan 是否曾被审计」的唯一记录。
- 写入条件：`ggml_gallocr_check_no_reuse_plan()` 打出 clean/enabled 行之后、`return true` 之前，
  且 **`n_retained > 0`**。零保留的 pass（1-row、knob off）不更新记录，因此永远不会被当成
  「已审计的保留 plan」。
- `plan_retained` 在 pass 开头置 -1（未知），审计结束时写成实际保留数；`match` 要求它 > 0。
- `ggml_gallocr_plan_identity()` 同时返回实时身份、`retained` 与这份「已审计」身份。

**(c) 门 — `examples/tp5-router-verify/run-router-cpu.sh` C4b**

- ON 腿：必须有 `router_scratch plan identity` 行，最后一行 `match=1`；其 `exec_uid/plan_id/exec_fp`
  必须等于最后一条 `retention audit clean (reserve): … graph_uid= plan_id= fp=` **且 `graph_uid != 0`**
  ——`graph_uid=0` 的行来自 L2 合成图（`run_plan_leg`，11 节点、不经 `sched_split_graph`），它排在
  L1 之后、写同一个 stderr，不加这条过滤就会拿它去比 decode 的 identity 行；
  且该行 `retained > 0`、`audit_plan_id == plan_id`（allocator 现在持有的 plan 就是审计通过的那一份）。
- ON 腿还必须满足：`retained >= 2`（两个 trunk MoE 层），且 clean 行自己的 `N retained allocations`
  与 identity 行的 `retained=` 相等（A8 注入 `1 retained allocations` 必 FAIL）；并且**逐条**判定而不是
  只看最后一条：`match=1` 的行数必须等于 `retained>0` 的行数且 `>= 3`（`--rows 1,2,3,4` 中 1-row 无保留、
  2/3/4-row 各 2 个保留投影），任何 `retained=-1` 的行都算失败（那次规划没有审计）。
- OFF / 1-row 腿：不得出现 `match=1`，也不得出现正的 `retained=`（没有保留就没有可推广的对象）。
- 缺行、`match=0`、身份不一致、没有 clean `(reserve)` 行 ⇒ 仍 UNPROVEN 并 `fail=1`。
- `VERIFY_REAL` 行改为 `unavailable=1 reason=owner_side_evidence join=router_scratch_plan_identity_line
  promotion=graded_by_driver_C4b`：工具自己不做 join，但不再宣称「无路可走」。

## 3. 覆盖范围：能证到哪、证不到哪

- 能证：**本次提交的图（`gf`）的 uid/fp 与 allocator 当前持有的 plan 相同，而该 plan 正是审计通过的那一份**
  （`audit_clean_plan_id == plan_id`，且 uid/fp/node/leaf 计数全等），并且分配时记下的身份与提交时一致。
  `sched->graph` 含全部输入副本与依赖节点，审计也是整图。
- 证不到：单个 split 内 kernel 级 dispatch（GPU 融合需 `topk_moe fused n_rows=… no_reuse=1`）；
  Meta backend 内部二次切分子图（各自另得 uid，且没有对应 allocator plan）；
  「图与历史基线数值等价」（仍需 logits 比对）。

## 4. 失效路径

| 场景 | 行为 |
|---|---|
| 形状变化 → 再次 reserve | 新 `plan_id`、新 `audit_graph_uid`，`reserved_plan` 同步覆盖；旧 uid 不可能再匹配 |
| 同形状复用（`can_reuse`） | 不重建图、不重跑 reserve；两侧身份都不动 ⇒ match=1，正确 |
| `alloc_graph` 内部 re-reserve | `plan_id` 前进；若发生在打印之前，打印的是新 plan（真 PASS）；若发生在打印之后，该图不是本次证据对象（不打印） |
| `split_only` / `reserve_n_size` | `reserved_plan` 不写；allocator 侧 `entry_size=1` 被 match 排除 |
| 图被重建但形状相同 | 新 uid ⇒ match=0（旧 `rows/nodes` 判据分辨不出） |
| 审计关闭 | `plan_id`、`graph_uid` 保持 0 ⇒ match=0，fail-closed |

## 5. 明确限制（不得据此宣称 PASS）

1. `match=1` 只说明计划身份一致且该计划被审计为不相交；不证明融合 kernel 被 dispatch。
2. 证据只在 `GGML_TP5_MOE_ROUTER_SCRATCH=1` 且 `GGML_TP5_ROUTER_SCRATCH_AUDIT=1` 时打印；
   默认路径不打印、不改时序、不改数值。
3. identity 行读的是**本次提交之后** allocator 持有的 plan，并与 `process_ubatch` 在分配完成时
   记下的 `exec_plan` 交叉核对。若两者之间的 plan 被替换（另一次 re-reserve），
   `audit_clean_plan_id != plan_id` 或 `exec_plan.plan_id != plan.plan_id`，`match=0`，
   fail-closed，不会把「计划换了」报成一致。
4. 证据行里 `retained=-1` 表示「本次 pass 没有审计信息」，`retained=0` 表示「审计过但没有保留」；
   两者都不允许 `match=1`。
5. 未编译、未运行、未测试。

## 6. 代码状态与 hash

见本轮交付的 sha256（由 Main 汇总）。文件：
`ggml/src/ggml-alloc.c`、`ggml/src/ggml-impl.h`、`ggml/src/ggml-backend.cpp`、
`ggml/src/ggml-backend-impl.h`、`src/llama-context.h`、`src/llama-context.cpp`、
`examples/tp5-router-verify/{tp5-router-verify.cpp,run-router-cpu.sh}`。

---

# 附：size_max realloc 判据修复（2026-10-31，仅 `ggml/src/ggml-alloc.c`）

## 改了什么
1. **恢复上游 realloc 判据**：`node_alloc->dst.size_max` 从“对齐后尺寸”改回未对齐的
   `ggml_backend_buft_get_alloc_size(...)`。`ggml_gallocr_node_needs_realloc()` 用未对齐的
   `node_size` 与之比较，原来的对齐值会把 alignment-1 字节内的真实增长吸收掉，导致本应
   re-reserve 的图被静默按旧计划 replay（Release 下断言被编译掉，不 fail-closed）。
2. **审计改用真实分配 extent**：新增 `ggml_gallocr_audit_extent()`（只读局部计算
   `aligned_offset(NULL, size_max, alignment)`），用于 retained extent、node-vs-node 与
   node-vs-leaf 比较。不写回任何 `tensor_alloc` 字段，allocator 判据不受影响。
   该选择让审计**更严格**：落在 alignment 尾部（未对齐约定视为空闲）的邻居现在算重叠。
   `size_max == 0`（预分配）与 `buffer_id < 0`（view）仍被排除，这两类正是
   `aligned_offset(get_alloc_size())` 不能描述预留的情形。

## 给 Devops 的边界验证（消费可见，勿自跑）
目标：证明“同图尺寸增长落在 alignment 余量内”会触发正确的 re-reserve，而不是静默复用旧计划。

1. **诊断开关**：`GGML_SCHED_DEBUG_REALLOC`（既有，`ggml-backend.cpp`）在 `sched->debug_realloc > 0`
   时，对“图规模不变却发生重分配”的情况打印/中止（`ggml_backend_sched_alloc_splits`）。
2. **边界构造（不改生产代码）**：用与正式完全相同的 builder 连续 reserve 两个图，两者的
   `n_nodes`/`n_leafs` 相同，但某个非 retained 节点的 `ne` 只增加 1 个元素（增量
   < `alignment`）。期望：修复后走 `ggml_gallocr_reserve_n`，出现新的
   `retention audit clean (reserve)` 行（`plan_id` 前进）。
3. **对照**：同一构造下把增量做到 > alignment，两种语义都必然 realloc。
4. **回归**：正常 decode（同形状复用）不得出现额外 `retention audit clean (reserve)` 行，
   也不得触发 `GGML_SCHED_DEBUG_REALLOC` 的 “unexpected graph reallocation” 中止；
   retained-inputs 提交的 `plan identity` 行应为 `match=1` 且 `plan_id` 与前一条 clean 行相同。

## 未验证
未编译、未运行、未测试。
