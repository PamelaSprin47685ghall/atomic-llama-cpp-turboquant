# TP5-FAST：60/100 tok/s 的最快实现（纯理论设计，先设计后实验）

本文是对 [TP5.md](TP5.md) 与 [下班交接.md](下班交接.md) 的**理论收敛**：不做新实验，只依据已有实测日志与现有源码，推导“最快的实现形态”，给出逐项预算、每 token 的提交结构、文件级改动清单与实施顺序。所有实验（GPU 加载、压力测试）按 TP5.md 2026-09-14 的硬件安全门推迟到维护窗口之后。

标记约定：

| 标记 | 含义 |
|---|---|
| **实测** | 2026-09-13/14 已有的真实运行日志（本文引用具体数值） |
| **代码事实** | 本仓库当前源码可直接核对（给出文件与符号） |
| **推导** | 由前两类推出，未独立测量 |

---

## 0. 结论速览

1. **当前 323 ms/token（3.09 tok/s）里，数学计算不到 1 ms。** 4B 激活参数 / token ⇒ 全模型约 8 GFLOP；分摊到 5 卡为 1.6 GFLOP/卡，按 RX 6800 可行的 5–8 TFLOP/s 计算约 **0.2–0.3 ms**；每卡每 token 需要流过的权重约 264 MB，按 512 GB/s 约 **0.5 ms**。剩下 ~99% 全是提交次数、等待次数与逐 stage 的 dispatch/barrier 开销。
2. **结构上每 token 有 97 个串行 stage（96 次归约 + 1 个尾部）**，这是 48 层 ×（attention 输出部分和 + FFN 输出部分和）的必然结果，不能靠再合并归约消掉；能动的只有**每 stage 的成本**。
3. 因此 60 tok/s（16.6 ms/token）要求 **每 stage ≤ 171 µs**；100 tok/s（10 ms）要求 **≤ 103 µs**。按 93 节点/stage/rank 的现状（≈1.5 ms/stage），差 10–15 倍，必须同时做两件事：
   - **把主机与内核驱动从关键路径上彻底移除**（每 token 从 ~1600 次 CS、~600 次 wait、~330 次 GEM_CREATE 降到 10 次以内主机操作）；
   - **把每 stage 的 dispatch 数从 ~93 降到 ≤50（60 tok/s）/ ≤30（100 tok/s）**，即算子融合与 HC 四流向量化。
4. 最快形态的名字叫**“每卡每 token 一次提交的预录 epoch 链”**：97 个 stage 的跨卡依赖全部用永久 timeline semaphore 在 GPU 侧表达，每个 rank 每 token 只发一次 `vkQueueSubmit`（提交数组内含 2×97 个 `VkSubmitInfo`），主机在整个 token 内不等待、不导出/导入、不分配。
5. 预期路径（推导，待维护窗口后实测）：323 ms →（F1+F5）60–90 ms →（F2+F3）40–60 ms →（F4 融合）20–35 ms →（F7 MTP 1.5–2×）→ **16.6 ms 以内** →（F8 上游补丁）→ **10 ms 以内**。
6. 理论地板 **7–13 ms/token（77–140 tok/s）**，由 `97 × N_disp × t_disp` 决定；因此 60 tok/s 可达，100 tok/s 必须 N_disp ≤ 30 **且** MTP 与上游融合全部到位。

---

## 1. 临界路径模型

### 1.1 公式

\[
T_{\text{token}} \;=\; \underbrace{\max\bigl(T_{\text{host}},\; T_{\text{gpu}}\bigr)}_{\text{重叠后}}
\;+\;T_{\text{finalize}}
\]

其中

\[
T_{\text{gpu}} \;=\; \sum_{s=0}^{96}\;\max_{i\in\text{rank}}\Bigl(\underbrace{N_{\text{disp}}(s,i)\cdot t_{\text{disp}}}_{\text{本地 dispatch}}
\;+\;\underbrace{N_{\text{barrier}}(s,i)\cdot t_{\text{bar}}}_{\text{barrier/cache flush}}
\;+\;\underbrace{t_{\text{push}}}_{\text{4 路 P2P 写}}\;+\;\underbrace{t_{\text{sum}}}_{\text{求和 kernel}}\Bigr)
\;+\;\sum_{s}t_{\text{edge}}(s)
\]

\[
t_{\text{edge}}(s) \;=\; \text{队列级 wait 处理} + \text{对端写可见延迟} + \text{语义释放/获取}
\]

关键含义：

* 求和外层是 **max over ranks**：每 stage 由最慢的卡决定，其余四卡闲置等待。
* `t_disp`（单次 dispatch 的引擎+驱动成本）与 `t_edge`（跨卡边）是常数级开销，**它们乘的是 97 与 9,000 这两个大数**，所以优化的唯一方向是把这两个乘数或常数压下来。
* 只有当 `T_host ≥ T_gpu` 时主机才是瓶颈；当前两种同步模式的测量都显示 **主机与 GPU 都没被重叠掉**（见 1.3）。

### 1.2 预算表

| 目标 | T/token | 每 stage 预算 | 备注 |
|---|---:|---:|---|
| 现状 | 323 ms | 3330 µs | 2026-09-14 实测 |
| 60 tok/s | 16.6 ms | **171 µs** | 目标一 |
| 100 tok/s | 10.0 ms | **103 µs** | 目标二 |

每 stage 171 µs 的构成（推导，两种情形）：

| 情形 | dispatch 数/rank | t_disp | 本地 dispatch 合计 | 边 + 求和 | stage 合计 |
|---|---:|---:|---:|---:|---:|
| 现状量级（保守 t_disp=16 µs，含未收敛 barrier） | 93 | 16 µs | 1490 µs | ~40 µs | **~1.5 ms（与实测 stage 1.5 ms 吻合）** |
| 修复 barrier 后（t_disp=3 µs） | 50 | 3 µs | 150 µs | ~15 µs | **~165 µs ✅ 60 tok/s** |
| 融合 + MTP（t_disp=3 µs） | 30 | 3 µs | 90 µs | ~12 µs | **~102 µs ✅ 100 tok/s** |

**这张表就是本方案的定量依据**：不融合算子，60 tok/s 无法到达；融合到 30 dispatch，100 tok/s 才有余量。

### 1.3 实测证据（2026-09-14 两个同步模式，MTP 开，ctx512/b32/ub32）

| 指标 | HOST 模式（默认） | TIMELINE 模式 | 来源 |
|---|---:|---:|---|
| decode | **323.2 ms/token**（21 tokens） | **310.2 ms/token** | `/v1/chat/completions` timings |
| prompt | 198.2 ms/token | 190.4 ms/token | 同上 |
| subgraph 调用 | 480 / token（97×5） | 480 / token | `[vk-compute-profile]` |
| vk-compute 主机区间 | 150–165 ms | 150–163 ms | 同上（replay 命中后 314–338 µs/次；未命中 992–1674 µs/次） |
| collective 分项 | p1_rec 4.2 / p1_sub 12.9 / **p1_wait 145.2** / p2_sub 57.0 ms | p1_sub 18.3 / **backpressure 141.0** / p2_sub 14.2 ms | `[tp5-profile]` |
| collective 合计 | ≈219 ms | ≈175 ms | 同上 |

两种模式的 collective 都落到 **≈1.5 ms/stage**，且在一个模式是“等 GPU fence”，在另一个模式是“等 GPU 追到 epoch-4”，说明**这是 GPU 侧的真实推进速率，不是主机线程慢**。同一模型 09-14 的 sysfs 采样显示五卡 busy 仅 **8/23/24/23/13%**（TP5.md 09-14 段），即 GPU 大部分时间在等而不是在算——与“1.5 ms/stage 远大于数学量级”一致。

内核侧账本（TP5.md 09-14 段，4.64 s decode 窗口）：`AMDGPU_CS 23,930`、`SYNCOBJ_TIMELINE_WAIT 9,058`、`SYNCOBJ_WAIT 4,851`、`GEM_CREATE 4,913`。按该窗口约 15 个 token 折算：**≈1,600 次 CS、≈600 次 timeline wait、≈330 次 GEM_CREATE 每 token**。

### 1.4 上界的另一面：不是算力、也不是带宽

| 项 | 每 token 每卡 | 结论 |
|---|---:|---|
| 激活 FLOPs | ~1.6 GFLOP | 0.2–0.3 ms |
| 需要读的权重字节 | ~264 MB | 0.5 ms @512 GB/s |
| P2P 发送字节（96 事件 × 4 对端 × 5 KiB，F16） | ~1.9 MiB | 0.16 ms @12 GB/s |
| 求和 kernel 元素数 | 96 × 2560 | 可忽略 |

⇒ **所有“物理量”加起来不到 1 ms，剩下的 322 ms 全是结构开销。**这就是为什么本方案只谈提交、等待、dispatch、barrier 四项，不谈算力与带宽。

---

## 2. 现状的六个结构性缺陷（代码定位）

| 编号 | 缺陷 | 代码位置（当前工作树） | 量化后果 |
|---|---|---|---|
| D1 | 每 stage 两次主机同步提交 + HOST 模式每 stage 5 次 `vkDeviceWaitIdle` | `ggml-vulkan-collective.cpp::tp5_allreduce_mesh`、`tp5_wait_all_p1/p2`、`tp5_p2p_visibility_barrier` | 480 次 device drain + 970 次 fence wait 每 token |
| D2 | 流水线深度被硬限在 4 个 epoch，且 mailbox 每发送者只有 1 个槽（无 epoch 环） | `tp5_comm::MAX_OUTSTANDING_EPOCHS = 4`、`in_flight_ring[]`、`tp5_setup_workspace` 里 `mailbox_bytes = n_ranks * stride + flags` | 主机最多领先 4 个 stage，之后必然阻塞；drain 路径本身要进内核等 syncobj |
| D3 | 每个 stage 由 meta 后端单独 `graph_compute_async` + flush 提交，collective 再提交 2 次 | `ggml-backend-meta.cpp` 主循环（`ggml_backend_graph_compute_async` + `ggml_backend_vk_flush_async`）、`tp5_record_plan` | 每 rank 每 token ≈291 次提交；驱动/内核每次 15–30 µs |
| D4 | 每 stage 3–4 次全流水线 barrier（`ALL_COMMANDS` + `MEMORY_WRITE`） | `tp5_record_plan` 的 `mb_pre` / `mb_wire` / `mb_p1` 与 P2 的 `mb_post` | 每 token ~400 次 L2 flush/drain，每次 1–3 µs，且阻止引擎流水 |
| D5 | 每 stage ~93 个 dispatch（主因：HC 四流逐流复制计算、MoE 分片、layout/copy 节点） | `src/models/qwen4exp.cpp` HC/QSA/MoE 图 + `src/llama-graph.cpp` | 9,000 dispatch/rank/token，是 1.5 ms/stage 的主体 |
| D6 | 每 token ~330 次 `GEM_CREATE`（staging/描述符池抖动），SYNC_FD 路径还有 480 export + 1920 import 主机账 | `ggml-vulkan.cpp` staging 与 `tp5_export/import`（SYNC_FD 分支） | 内核 76 ms/窗口；主机 syscall 数进入关键路径（TP5.md §13.6 早已警告） |

四条属于“结构”，两条属于“数量”。**D1–D4 是必须先修的，因为它们决定“能不能让 GPU 连续跑”；D5 决定“能跑多快”。**

---

## 3. 最快形态：每卡每 token 一次提交的预录 epoch 链

### 3.1 总体结构（推导 + 现有机制组合）

```text
每个 rank（rank i ∈ 0..4）在 token 开始时发出 1 次 vkQueueSubmit：
  VkSubmitInfo 数组（按 stage 顺序，同一队列内顺序执行，本地依赖自动成立）：

  s = 0..96:
    [S1_s]  waits:  credit[j][s mod R] ≥ s-R   (j≠i, 4 个永久 timeline semaphore)
            cmds : 本地 subgraph 全部节点（预录 CB）
                   + wire pack（canonical f16）
                   + 本地 PUSH（写自己的 mailbox 槽 i）
                   + 4 路 peer PUSH（写对端 mailbox 槽 i，P2P copy）
            signal: push_done[i] = 2s+1

    [S2_s]  waits:  push_done[j] = 2s+1 (j≠i, 4 个导入的 peer timeline)
            cmds : 求和 kernel（固定 rank 顺序）→ 写回本地部分和/结果张量
            signal: consumed[i][s mod R] = s

  最后：tail 子图（LM head 切片）→ 采样前回读一次
```

* **token 内主机只做 5 次 submit + 1 次最终回读**；不再有 fence wait、device idle、export/import、CB 分配、GEM_CREATE。
* **跨卡依赖全部由 semaphore 表达**，因此这不是 TP5.md §13.5 禁止的“一条大 CB 只等头尾 fence”：每 stage 的 4 条远端依赖都在提交数组里显式表达，`S2_s` 的 wait value 就是 §13.3 要求的 `wait[s][j][i]`。
* `s = 96` 之后是尾部（LM head / 采样前处理），它不需要归约（**实测**：97 个子图里只有 96 次归约）。

### 3.2 为什么需要 R 槽 mailbox + credit 环（修 D2）

当前结构里 `mailbox` 每个发送者只有 **一个** 槽（`mailbox_buf` 大小 = `n_ranks*stride + flags`），跨 epoch 复用靠“只允许 4 个 epoch 在飞行 + drain”来兜底，于是主机必然每 4 个 stage 进一次内核等 syncobj。这正是实测里 `cpu_backpressure ≈ 141 ms/token`、以及 600 次/token `SYNCOBJ_TIMELINE_WAIT` 的来源。

改成 `R = 16` 槽（5 卡 × 16 × 5120 B ≈ 400 KiB，可忽略）：

* 发送者在 epoch `s` 使用 `slot = s mod R`；
* 接收者在 epoch `s` 求和完成后 signal `consumed[i][slot] = s`；
* 发送者下一次使用同一槽（epoch `s+R`）前 wait `consumed[j][slot] ≥ s`。

由于 `R=16` 远大于 5 卡之间的正常抖动，**这些 credit wait 在稳态几乎总是已满足**（引擎只花“检查已信号量”的常数成本），主机因此可以一次性把整个 token 排空，`cpu_backpressure` 从 141 ms 掉到 ~0。这也是 §12.3 的“显式状态机”的自然扩展：mailbox 从“单槽 + 全局 drain”变成“R 槽 + 每槽 epoch credit”。

### 3.3 为什么能砍掉 1,600 次 CS（修 D3）

要让“每卡一次提交”成立，必须让 meta 后端**不再自己提交**：它当前对每个 subgraph 调 `graph_compute_async`（Vulkan 后端内部 submit）再 `flush_async`。需要一个新的窄接口（`ggml-vulkan-internal.h` 已存在同样风格的桥梁）：

```cpp
// ggml-vulkan.cpp（新增，仅内部）
VkCommandBuffer ggml_vk_tp5_record_subgraph(ggml_backend_t backend, ggml_cgraph * cgraph);
// 语义：保证 replay 命中（miss 时现场录制并缓存），把该 subgraph 的 CB 交回调用者，不提交。
// 失效规则沿用 TP5.md §14.5：buffer 世代 / 形状 / 量化类型 / wire 类型变化即缓存失效。
```

meta 后端在 TP5 模式下改为：

1. 对每个 stage，向 5 个 rank 各取一次 `record_subgraph`（缓存命中）；
2. 与 collective 的 pack/PUSH/求和 CB 一起组成 §3.1 的提交数组；
3. **一次** `vkQueueSubmit` 交给 collective 的提交函数（每 rank 一次），自身不再 submit/flush。

这样每 token 的主机操作从“291 次 submit/rank”变成“1 次 submit/rank”。

### 3.4 为什么能把 t_disp 从 16 µs 压到 3 µs（修 D4/D5）

* **barrier 精确化**：按 TP5.md §13.7 的模板，把三处 `ALL_COMMANDS_BIT + MEMORY_WRITE` 换成：
  `COMPUTE_SHADER→TRANSFER`（输出给 PUSH）、`TRANSFER→COMPUTE_SHADER`（PUSH 给求和）、
  `COMPUTE_SHADER→COMPUTE_SHADER`（求和给下一 stage）。只有在真正需要跨设备可见性的那一条上做 release/acquire，其余用本地 `MEMORY_BARRIER`（不带 L2 全冲刷）。
* **dispatch 粗化（F4）**：
  - **HC 四流向量化**：`C=4` 的每一组标量算子合成一次 dispatch（把流维度放进 `ne[2]`/workgroup 维度），预计把 HC 段节点数除以 ~4；
  - MoE：gate/up 已合并，进一步把 shared expert 与 routed 的本地部分和**在同一 kernel 内合并**（§6.2 已要求“本地合并后只归约一次”），并消除 zero-size 专家分片产生的空 dispatch；
  - 消除 layout/copy/view 类节点的实际 dispatch（能 alias 的不要真拷）；
  - norm+rope+… 沿用上游已有融合。
  目标 **≤50 dispatch/stage（60 tok/s）/ ≤30（100 tok/s）**，以 `GGML_VK_PERF_LOGGER` 或自建 dispatch 直方图作为证据。

### 3.5 同步模式与主机账本

| 模式 | 每 token 主机操作 | 用途 |
|---|---|---|
| **TIMELINE（新默认）** | 5 submit + 1 回读 + 0 export/import | 生产快路径 |
| SYNC_FD（保留） | 480 export + 1920 import + FD dup/close | 兼容/对照，不进快路径（§13.6 已警告主机账） |
| HOST（保留） | 970 fence wait + 480 device idle | 仅调试与参考正确性 |

`GGML_TP5_SYNC` 的默认值应从 `HOST` 改为 `TIMELINE`（**代码事实**：`ggml_backend_vk_tp5_comm_init` 在未设置时选 HOST），HOST 保留为显式选项。TIMELINE 已在 09-14 的五卡 F16 测试中通过，包括 8 步无中间 host-sync 的依赖链（TP5.md 09-14 段）。

### 3.6 可选的第二期：在途 flag 自旋（F6）

`tp5_sum_f32.comp` 已经带 `use_flags` / `spin_max` push constant，`tp5_flags_byte_offset` 也已分配 flag 区（当前 `use_flags = 0` 未启用）。启用后每 stage 可去掉 4 个远端 push wait：

* 发送者：4 路 P2P copy 完成后写 `flag[slot_i] = seq`（需要 release 语义）；
* 接收者求和 shader：先自旋读 4 个 flag 直到等于本轮 `seq`，再求和（需要 acquire 语义）。

收益：每 token 减少 96×5×4 = 1,920 次队列级 semaphore wait。风险与前提：TP5.md §26.3 明确要求“另写设备/驱动内存模型的完整证明和验证程序”。因此列为**第二期**，前置条件是一个独立验证程序：人工延后一个生产者、消费者在未完成时提交自旋 kernel、校验读到的是**本轮**数据（§13.8 的严格版本），并在五卡上覆盖 PCIe 写顺序、flag 与 payload 的可见性配对。

### 3.7 MTP 与上游补丁的位置

* **F7 MTP**：把“每输出 token 的验证步数”从 1 摊销到 ~1/mean_len（实测 mean len 3.00，接受率 14/14；09-14 另有 15/32、13/36）。它不改变 §3.1 的结构，只是把 1 步变成同结构的 k 步；要求 draft 图也走“预录 + 一次提交”，否则 §1.1 里 `T_host` 又被 draft 路径拉高。
* **F8 上游补丁**（`docs/qwen4exp-upstream-optimizations.md`）：TopK 融合、Split-K、真稀疏注意力、增量池化 KV。它们减少的是**每 token 的 GPU 工作量与部分 dispatch**，在 §1.1 里表现为 `t_disp` 与 `N_disp` 的乘数下降；不集成它们，100 tok/s 的余量不足。

---

## 4. 每 token 账本：现状 vs 最快形态

| 项 | 现状（实测/推导） | 最快形态（目标） |
|---|---:|---:|
| `vkQueueSubmit`（主机） | ~1,455 次（97×(5+10)） | **5 次** |
| 内核 `AMDGPU_CS` | ~1,600 次 | ≤50 次（由驱动对提交数组的打包决定） |
| `SYNCOBJ_*_WAIT`（主机） | ~600 次 | **0**（GPU 侧 wait） |
| `vkDeviceWaitIdle` | 480 次（HOST 模式） | **0** |
| `GEM_CREATE` | ~330 次 | **0**（全部持久化） |
| SYNC_FD export/import | 0（未启用）→ 若启用 2,400 次 | **0** |
| 队列级 semaphore wait（GPU） | 480 × 5 | 96 × 5 × (4 push + 4 credit) = 3,840（第二期用 flag 自旋降到 96 × 5 × 4 = 1,920，且不再有 push wait） |
| per-stage barrier | 3–4 次全冲刷 | 2 次精确 scope |
| dispatch/stage/rank | ~93 | ≤50（F4 后 ≤30） |
| 最终回读 + CPU 采样 | 1 次 | 1 次（不可省，进 `T_finalize`） |

---

## 5. 与两份文档的契约一致性

| 契约 | 本设计是否满足 |
|---|---|
| TP5.md §13.5「不能预录一条大 CB 只在头尾 fence」 | 满足：跨卡依赖按 stage 显式表达为 semaphore wait/signal；预录的是 CB 与描述符，载荷每 epoch 更新 |
| TP5.md §13.3「一道输出边界的提交顺序」 | 满足：S1 覆盖 produce+convert+push+release+signal，S2 覆盖 4 个远端 wait + 求和；差别只是把两次提交合并进同一次 `vkQueueSubmit` 的数组 |
| TP5.md §13.4 SYNC_FD 规则 | 仅 SYNC_FD 模式适用；快路径改用永久 OPAQUE timeline（§26.3 的“不默认使用”被遵守：它只是我们的**快路径选项**，HOST/SYNC_FD 两种参考路径保留，且需独立验证程序） |
| TP5.md §13.6 主机账本 | 满足并强化：给出 before/after 计数表（§4） |
| TP5.md §11.1/§11.3 固定顺序求和、canonical wire | 不变（求和 shader 与 pack 路径不改语义） |
| TP5.md §13.7 barrier 模板 | 被落实（D4 修复即按该表） |
| TP5.md §12 mailbox 所有权/状态机 | 扩展为 R 槽 + epoch credit，需要补 §12.3 的状态机文档 |
| TP5.md §17.4「计算通信重叠是后续项」 | 遵守：本方案先消除开销，不做 GEMM 分块重叠 |
| TP5.md §25 验收红线 | 不变：不删保护、不把 forced token 计入吞吐、先正确后快 |
| 下班交接.md 任务一（批量提交消除 ioctl） | 本方案的任务一（F1） |
| 下班交接.md 任务二（GPU flag 自旋） | 本方案的 F6（第二期，带前置验证） |
| 下班交接.md 任务三（上游补丁） | F8 |

---

## 6. 实施顺序（每步都有可证伪的验收）

> 执行时机：TP5.md 2026-09-14 硬件安全门解除后（现场内存通道检查 + 维护窗口），先纯 CPU 逻辑验证，再单卡，再五卡。

### 阶段 A：主机与内核退出关键路径（F1 + F5）——预期 323 ms → 60–90 ms

1. `ggml-vulkan-collective.cpp`
   - 新增 `tp5_chain_begin/append/submit`：为每 rank 维护一个提交数组（stage 顺序），`append(stage, cb_s1, cb_s2, wait_infos...)`，`submit()` 一次发出；
   - `tp5_allreduce_mesh` 拆成 `tp5_chain_plan`（录制/缓存）与 `tp5_chain_submit`（提交），删除每调用一次的 `tp5_wait_all_p1/p2` 与 `tp5_p2p_visibility_barrier` 在快路径上的调用；
   - 默认 `sync_mode = TIMELINE`（保留 `GGML_TP5_SYNC=host|syncfd` 显式覆盖）。
2. `ggml-vulkan.cpp`：新增 `ggml_vk_tp5_record_subgraph()`（§3.3），并在 `ggml_backend_vk_reg_get_proc_address` 暴露。
3. `ggml-backend-meta.cpp`：TP5 模式下主循环改为“录制 + 追加”，由集体的链提交统一发出；保留原有逐 stage 提交路径作为 `GGML_TP5_CHAIN=0` 的对照。
4. 验收（后续实测）：
   - 每 token `vkQueueSubmit` ≤ 10（可用 LD_PRELOAD 或 `GGML_VK_TP5_SUBMIT_COUNT` 计数）；
   - 每 token `SYNCOBJ_*_WAIT` ≤ 5，`GEM_CREATE` ≈ 0；
   - 固定 prompt 输出与参考逐字一致（`1..12`、`9.9`），96 边界不变；
   - 记录 `[tp5-profile]` 与 GPU busy%。

### 阶段 B：R 槽 mailbox + credit 环（F2）——预期 60–90 ms → 40–60 ms

1. `tp5_setup_workspace`：mailbox 改为 `R × n_ranks × stride + flags`（`R=16` 可配）；
2. `tp5_record_plan`：`slot = epoch mod R`；
3. 新增 `consumed[i][slot]` timeline 值，S2 求和后 signal，S1 在 `s ≥ R` 时 wait 四个 peer 的 credit；
4. **CPU 侧先行验证**：把 epoch 链的状态机抽成纯逻辑参考（沿用 `tests/test-tp5-plan.cpp` 风格），穷举 R=2/4/16 下的复用顺序与危险窗口；
5. 验收：`cpu_backpressure` 从 141 ms 降到 ≤ 5 ms/token；任意人工延后一个 rank 的对抗测试不出现跨 epoch 数据混淆（§13.8 风格）。

### 阶段 C：barrier 精确化（F3）——预期再省 5–8 ms/token

按 §13.7 表改写 `tp5_record_plan` 的三处 barrier；验收：`[vk-compute-profile]` 与 stage 时间下降，且五卡 96 轮数值测试仍逐元素精确。

### 阶段 D：dispatch 粗化（F4）——预期 40–60 ms → 20–35 ms

1. HC 四流向量化（`src/models/qwen4exp.cpp` + 相关 shader）；
2. MoE 本地 routed/shared 合并（`ggml-backend-meta.cpp` 的 `can_defer_linear_partial` 已存在，需确保 qwen4exp 图形态真正命中）；
3. 消除 empty dispatch 与纯 layout copy；
4. 验收：`N_disp/stage` 直方图 ≤50；固定 prompt 输出一致。

### 阶段 E：MTP 提速（F7）与上游补丁（F8）

* MTP：draft 图走同一条链；验收 `draft_n_accepted / draft_n` 与 tok/s；
* 上游：逐补丁单独提交、单独测量（TopK 融合 → Split-K → 稀疏注意力 → 池化 KV），每项记录对 `N_disp` 与 tok/s 的影响。

### 阶段 F（可选）：flag 自旋（F6）

前置：独立的设备/驱动内存模型验证程序（§13.8 的严格版本）通过后，才允许把 4 个 push wait 换成 shader 自旋。

---

## 7. 风险、地板与禁止项

**风险**

| 风险 | 说明 | 缓解 |
|---|---|---|
| 队列级 wait 的引擎成本未知 | 3,840 次/token 的 wait 若每次 2–3 µs，就是 8–11 ms | 先量测；必要时上 F6 |
| 单队列串行放大最慢卡 | stage 由 max over ranks 决定 | 保持各 rank 张量形状一致；对称角色表已保证（`[5,5,5,5,4]`、`[14,14,15,15,14]`） |
| RADV 对提交数组的打包方式 | `vkQueueSubmit(array)` 可能仍按 wait 边界拆 ioctl | 以 CS 计数为准；必要时按 stage 合并 reduce 边界 |
| 大 CB 的 replay 失效 | 张量指针/形状变化会静默退化成录制 | 沿用 §14.5 的 fingerprint；计数 miss |
| 显存 | R 槽 mailbox 400 KiB、链 CB 若干 MiB，均可忽略 | 逐卡预算按 TP5.md §4 重算 |
| 硬件安全门 | MCE 记录（TP5.md 2026-09-14） | 维护窗口后先做只读检查与阶梯压测 |

**理论地板**：`97 × N_disp × t_disp`。若 `t_disp = 3 µs`、`N_disp = 15`，地板 ≈ 4.4 ms/token（≈227 tok/s）；实际还受 `T_finalize`（回读 + CPU 采样，0.5–2 ms）与 stage 间可见性延迟限制，故**7–13 ms/token 是可信地板**。

**禁止项**（与两份文档一致）：不做 host-relay；不用一条大 CB 替代真实依赖；不用 flag 自旋替代验证程序；不把 forced token 算作有效吞吐；不在没有 per-stage 证据时宣布达标；不为了 tok/s 牺牲 96 边界与数值一致性。

---

## 8. 待执行验证清单（安全门解除后按序）

```text
1) build-tp5-cpu: test-tp5-plan / test-meta-reduce-boundary / test-recurrent-state-rollback / test-qsa-pooled-cache
2) 单卡: 阶段 A 的提交计数与固定输出对比（不加载全模型）
3) 五卡: test-vulkan-tp5-mesh（F16/timeline，96 轮 + 对抗轮次）
4) 五卡全模型: 固定 prompt（计数 1..12 / 9.9），逐字一致 + timings
5) 阶段 A–F 每步: [tp5-profile]、[vk-compute-profile]、CS/WAIT/GEM_CREATE 计数、GPU busy%、VRAM 峰值
6) 质量与长稳: TP5.md §25 的功能/性能/兼容矩阵逐项补齐
```

**当前状态（本文写作时）**：`llama-server` 已停止，五卡显存回到 17.2 MB 基线；本文只做设计，未运行任何 GPU 负载。
