# 设计 TriAttention 移植!

**User:** Anonymous  
**Created:** 9/2/2026 10:13:52  
**Updated:** 9/2/2026 11:14:54  
**Exported:** 9/2/2026 11:17:29  
**Link:** [https://chatgpt.com/c/6a97865f-93d8-83ed-8c02-62abff8f5a8c](https://chatgpt.com/c/6a97865f-93d8-83ed-8c02-62abff8f5a8c)  

## Prompt:
9/2/2026, 10:15:47 AM

[@7500F](plugin://dev-6a964ba312d48191b5e2962acc157833@created-by-me-remote) 如果想把/home/kunweiz/Desktop/llama-cpp-turboquant TriAttention 移植到 /home/kunweiz/Desktop/atomic-llama-cpp-turboquant 让后者的 auto kv management (unified 只按实际用量占用 + 自动决定 recurrent 期望槽 + GPU/CPU auto swap) 进一步自动带 TriAttention 的管理，你会如何设计？

## Response:
9/2/2026, 10:15:51 AM

我会先按“管理层抽象 → TriAttention 资源模型 → 分配/迁移策略 → 执行路径”的顺序，把两个仓库的现有实现对齐，重点看 unified KV、recurrent slot 选择、GPU/CPU swap 和 TriAttention 的 cache/state 生命周期在哪里耦合。然后我会给出一个尽量少侵入、能复用现有 auto-kv policy 的移植设计，并标出建议改动点。

两个 checkout 都已经打开了；第一个仓库的根指令明确允许做“结构理解/设计建议”，但不应该替你直接生成大改动代码，所以这里我会停在架构与落点分析，不改文件。现在开始把两个实现的关键符号和生命周期对应起来。

目前已经看到一个关键差异：TriAttention 分支把它直接嵌进 `llama_kv_cache`，而 atomic 分支已经把 context memory 抽象成 `llama_memory_i`，并且 recurrent 容量是独立可探测/可缩放的。也就是说，最不该做的是把旧分支那套 `triattention_st` 原封不动塞回新版 `llama_kv_cache`；更合理的是让 TriAttention 变成 memory policy/capability，挂在现有 memory 抽象上。

从历史提交也能确认 atomic 这套不是普通 upstream KV：`3cff74bdb` 是 unified KV 管理，`75cff73bb` 是按 unified KV 容量自动定 recurrent state，`e4f1af3ae` 又把 server logical slots 限到 fitted recurrent capacity。接下来我会把这三层和 TriAttention 的“预算/淘汰”统一成同一个 pressure model，而不是再加一套独立阈值。

> **Inspected project workspaces, instructions, and navigation skills**
>
> 
>
> **Inspected project structure and attention auto-memory implementation**
>
> 
>
> **Inspected TriAttention and KV cache integration, memory swapping, and auto-swap history**
>
> 
>
> **Designing automatic management**
>
> I’m focusing on integrating TriAttention with the unified KV manager rather than adding another manual budget. The manager should reclaim cache under pressure, then fall back to demotion or preemption only when necessary.
>
> **Protecting shared context**
>
> I’m keeping recent-window and sliding-window KV entries intact, while treating long-range memory separately. Because cells may be shared across sequences, I’m scoring each reference and evicting a physical cell only when no sequence still requires it.
>
> **Protected shared context**
>
> 
>
> **设计统一生命周期**
>
> 我会把 TriAttention 做成 `llama_memory_i` 的可选“回收策略”，不复制旧缓存状态。统一 KV 管理器先统计真实占用，再按压力调用策略；不足时才触发 CPU/GPU swap 或槽位抢占。
>
> **Aligning cache metadata**
>
> I’m treating `llama_kv_cells` as the sole source of truth. TriAttention should query authoritative positions and sequence references, preserving shared prefixes and avoiding the old global-position assumptions.
>
> **Handling persistence**
>
> I’m noting that serialized KV state may relocate physical cells, so TriAttention metadata must be rebuilt from saved positions. Prompt-cache reuse should also mark pruned states as approximate, especially when branching from earlier prefixes.
>
> **Balancing adaptive policies**
>
> I’m separating lossless RAM demotion from lossy TriAttention pruning. Idle states should move to RAM first; active KV can then be pruned under pressure, with preemption remaining the final fallback.
>
> **Designed unified lifecycle, aligned cache metadata, handled persistence, and balanced adaptive policies**
>
> 
>
> **Investigated TriAttention unified KV cache management and GPU/CPU swapping**
>
> 
>
> **Audited sparse TriAttention sequence, cache, kernel, and backend integration viability**
>
> 
>
> Worked for 8m 33s

我会把 TriAttention 从“KV cache 自己偷偷 prune 的功能”改造成 **atomic auto memory manager 的一种 lossy reclaim policy**。核心原则是：**`llama_memory_i` / unified KV 仍然是唯一资源真相，TriAttention 只负责回答“压力来了，哪些 KV cell 最值得丢”**。

这比直接把旧分支的 `triattention_st` 搬进新版 `llama_kv_cache` 稳得多。

### 先说我看到的两个实现之间最关键的差异

| 旧 `llama-cpp-turboquant` TriAttention | `atomic-llama-cpp-turboquant` 应该变成 |
|---|---|
| `triattention_state::cell_positions[kv_size]` 自己复制一份 cell→position | **删除**，直接以 `llama_kv_cells` 为 authoritative metadata |
| 一个全局 `absolute_position` | unified 下必须是 **per-seq frontier/current pos** |
| 一个全局 `prefix_length` | 必须是 **per-seq protected prefix** |
| 只看 `v_cells[0]` | unified stream 中同时管理所有 seq |
| `apply_ubatch()` 后触发 prune | 主要由 `server_context::ensure_next_kv_capacity()` **在分 batch / prepare 前触发** |
| 固定 `budget + interval/slack` 决定何时 prune | auto 模式按 **实际 deficit** 决定这次到底要释放多少 cell |
| TriAttention 自己负责“删除” | memory manager 发 reclaim request，KV cache 原子地完成 metadata eviction |
| 一个 GPU state 基本假设统一 K type | atomic 有 layer-adaptive K type、CPU/GPU mixed placement，要 **per-layer/per-device dispatch** |
| scratch 按完整 `kv_size × sampled_heads` 分配 | chunked scoring，scratch 纳入 auto-fit reserve |
| 只支持直接 `llama_kv_cache` | hybrid → attention 部分；iSWA → base cache；recurrent 不碰 |
| per-KV-head / per-layer-head 最后仍落成整个 token cell 的删除 | V1 明确只承诺 **token/cell-granularity global eviction** |

最后一点挺重要：旧实现虽然有 `PER_KV_HEAD` / `PER_LAYER_HEAD` 名字，但最终 Step 5 仍然得到一个 whole-cell keep set，然后 `cells.rm(i)`。现有 KV layout 并不能真正做到“某个 token 只删某个 KV head”。所以移植第一版我不会把这些模式包装成真正的 head-sparse eviction。

---

## 我会把架构切成三层

**第一层：TriAttention scorer。**

把 `src/llama-triattention.cpp` 里真正有价值的部分保留下来：calibration loader、RoPE inversion、Q/K trigonometric score、TurboQuant dequant、CUDA score kernel。

但 scorer 不再知道 `kv_size`、不再有 `cell_positions`、不再有 `prefix_length`、不决定 prune trigger，也不直接改 KV cache。

它的输入应更接近：

```text
K storage + layer/head
candidate physical cell indices
candidate positions
query/frontier position
calibration
    ↓
importance score[]
```

也就是说它变成纯粹的“importance oracle”。

**第二层：KV reclaim policy。**

挂到 `llama_kv_cache` 上，但最好通过 `llama_memory_i` 暴露一个 generic capability，而不是 public API 里写死 TriAttention，例如概念上：

```text
memory.reclaim_kv(request)
    request:
        cells_needed
        reserve/slack
        per-seq protection hints
        allow_lossy
        reason = capacity_pressure

    result:
        cells_freed
        cells_scored
        exhausted
        elapsed
```

`llama_memory_i` 默认实现就是“不支持 reclaim”，所以 DSA/DSV4/MSA/recurrent 不需要马上改。

`llama_kv_cache` 实现它；`llama_kv_cache_iswa` 转发给 `kv_base`；`llama_memory_hybrid` 转给 `mem_attn`；`llama_memory_hybrid_iswa` 再往 attention/base 转。

这样以后想接 H2O、SnapKV、StreamingLLM 等，也不用再改 server resource manager。

**第三层：server memory pressure planner。**

真正决定“swap 还是 TriAttention 还是 preempt”的地方就是 atomic 现在已经有的：

`tools/server/server-context.cpp::ensure_next_kv_capacity()`。

这是整个设计最合适的控制中心。

---

## auto TriAttention 不应该再以固定 budget 为主要 trigger

atomic 已经知道：

```text
capacity
used
next batch required KV
next batch required recurrent
```

所以 auto TriAttention 最自然的定义其实是：

```text
deficit =
    max(0,
        kv_used
      + kv_required_next
      + kv_safety_reserve
      - kv_capacity)

tri_target_free =
    deficit + reclaim_slack
```

然后 TriAttention **只释放 `tri_target_free` 左右的 cell**。

这样 `--kv-size auto` 和 TriAttention 才真正是一套系统：

```text
当前 unified KV 实际占 48120
下一步需要              1536
capacity               49152
reserve                   512

deficit = 1016

→ 不需要把所有 seq 硬 prune 回 2048
→ TriAttention 只挑大约 1016 + amortization slack 个最低价值 cell
```

这比旧的：

```text
if used >= budget + window:
    prune_to(budget)
```

更符合 atomic 的设计哲学，而且质量损失明显更可控。

我会保留 fixed budget 模式做实验/论文复现，但增加 `budget=auto`，把 **pressure-driven reclaim 作为 atomic 默认模式**。

---

## 和现有 GPU/CPU auto swap 的关系

按当前 atomic master，我看到的 “CPU swap” 本质上是 server 层的 sequence-state demotion：

`prompt_save()` → `llama_state_seq_get_data_ext()` → `server_prompt_cache` RAM → `prompt_clear()` 释放 unified KV；恢复时再 `llama_state_seq_set_data_ext()`。

它不是 KV page 粒度的实时 GPU↔CPU tensor migration。

所以 TriAttention 正好应该插入同一条 pressure ladder。

我会提供两种 policy。**quality-first 默认**：

```text
KV pressure
    │
    ├─ idle slot 存 RAM + clear GPU KV       ← lossless
    │
    ├─ TriAttention reclaim active KV        ← lossy，但不中断请求
    │
    └─ active slot preempt/save/retry         ← 最 disruptive
```

如果目标是最低 latency，则允许配置成：

```text
TriAttention
    ↓ still insufficient
idle GPU→RAM demotion
    ↓ still insufficient
active preemption
```

这样可以用少量质量换掉一次很大的 D2H state copy。

---

## unified 多 sequence 是移植时最需要重写的一块

旧 TriAttention 的 global `absolute_position` 和 `prefix_length` 在这里不能继续用。

对于物理 cell `i`，它可能同时属于多个 sequence，因为 atomic 的 `llama_kv_cells` 本身支持 shared prefix。

因此应该用：

```text
score(cell i)
    = max over seq ∈ refs(i)
        max over sampled heads
            TriScore(K_i, position_i, frontier(seq))
```

这里对 sequence 再做一次 `max` 很重要：一个 shared physical cell 只要对其中一个 sequence 非常重要，就不应该因为另一个 sequence 不重要而把整块删掉。

同样，hard protection 应是：

```text
protected(cell)
    = ANY referenced seq says protected
```

比如某 cell 同时属于 seq 2 和 seq 5，只要它落在 seq 5 的 recent window / protected prefix 内，整个 physical cell 就不能回收。

而且要注意：

`cells.seq_rm(i, seq)` **未必释放 physical capacity**；只要还有别的 seq reference，`get_kv_used()` 不会下降。

所以 pressure reclaim 的 accounting 必须以 **真正变 empty 的 physical cell 数**为准，而不是“删除了多少 seq references”。

---

## `llama_kv_cells` 应成为唯一 metadata source

这一点我会很坚决地改掉。

旧版本这些东西全部删掉：

```text
triattention_state::cell_positions
triattention_on_token_added()
triattention_on_cell_removed()
triattention_on_position_shift()
triattention_on_reset()
```

atomic 现在 `llama_kv_cells` 已经维护：

`pos`、`seq bitset`、`used`、per-seq `seq_pos`、per-seq `seq_used`。

再复制一套只会在这些路径上迟早出 bug：

state save/load、prompt RAM swap、MTP shared cache、`seq_cp()`、context shift、spec rollback、SWA overwrite。

尤其 atomic 的 `state_read_meta()` 恢复 sequence state 时会重新 `find_slot()`，也就是说 **恢复后的 physical cell index 完全可能变化**。旧版 `cell_positions[cell_idx]` 在这种设计下天然不可靠。

TriAttention 无 cell-index persistent state 后，CPU swap/load 后什么都不用修，下一次直接从 `llama_kv_cells` 枚举 resident cell 即可。

---

## 还有一个 atomic contract 必须显式处理

`include/llama.h` 现在对 `llama_memory_seq_pos_min/max()` 写的是：

> `[pos_min, pos_max]` 中所有位置保证存在。

TriAttention 中间删 token 后，这个 contract 就不成立了。

旧分支只靠“保护 recent window，所以 `seq_pos_max` 没变化”解决了 `Y = X + 1` 的下一 token 校验，但没有真正解决这个 API contract。

我会给 memory 增加一个类似：

```text
DENSE_POSITIONS
SPARSE_POSITIONS
```

的 capability，然后审核依赖 dense-layout 的路径。

现有 atomic 已经有一个很好的例子：`get_n_kv_pos_contiguous()` 本来就在逐 cell 检查是否真的是 `cell[pos] == pos`，不满足就回 dense bias path，因此这个优化本身对 TriAttention 是安全的。

第一版再强制：

```text
recent_keep >= 1
```

保证 latest logical position 永远 resident，现有 `seq_pos_max()+1` 的 decode position 逻辑就还能正常工作。以后如果想连 tail 都允许删除，再把 “logical frontier” 从 “max resident position” 彻底拆开。

---

## prefix protection 也必须变成 per-seq

旧代码：

```text
state->prefix_length
```

在 unified server 是错的。

我会让 server 在 reclaim request 里带 sequence hints，例如概念上：

```text
seq_id
protected_prefix_end
recent_keep
minimum_resident
priority
```

因为只有 server slot 才真正知道：

这是 prompt 还是 generation、`n_keep` 是多少、哪个 slot 正在 prefill、哪个 slot 是 idle。

如果继续支持 `protect_prefill=true`，其边界还要跟 `server_prompt_cache_state` 一起保存一个很小的 TriAttention seq metadata。否则一个生成过 10k token 的 RAM state reload 后，已经没法从 `prompt.tokens.size()` 准确反推出“原始 prompt 到哪里结束”。

---

## iSWA / hybrid 我会这样处理

iSWA 其实特别适合这个划分。

atomic 的 `llama_kv_cache_iswa::get_kv_used()` 本来就是返回 `kv_base->get_kv_used()`，SWA cache 是单独的小 recent cache。

所以：

```text
TriAttention → 只 prune kv_base
SWA         → 保持完整 recent window
```

这比再在 TriAttention 内做一层 recent protection更天然。

Hybrid 则是：

```text
llama_memory_hybrid
    ├─ mem_attn  → TriAttention allowed
    └─ mem_recr  → untouched
```

所以 Qwen3-Next / Qwen3.5 这种 attention + recurrent 的模型不会让 TriAttention 碰 recurrent state。

第一版我会明确禁掉 MLA / DSA / DSV4 / MSA / M-RoPE / variable head-dim 等没有验证过的路径，避免“能编译但评分公式已经不成立”。

---

## GPU scorer 不能原封不动搬

旧 CUDA TriAttention 基本上有一个 `triattention_gpu_state.cfg.k_type`，然后依据这个 K type dispatch。

但 atomic 当前 `llama_kv_cache` 有两个新现实：

一是 layer-adaptive TurboQuant，可以不同层 K type 不同；二是 `model.dev_layer(il)` 允许不同层落在 CPU / 不同 GPU。

因此我会给 scorer 建 **per-layer storage descriptor**：

```text
layer
K tensor
logical head_dim
physical/padded head_dim
K type
backend/device
row stride
n_kv_heads
```

评分时按 `(device, k_type)` grouping。

CUDA K 就原地 CUDA score；CPU K 就 CPU score；multi-GPU 每个 device 保留一份很小的 calibration copy。

不要再用：

```text
k_type = k_tensors[0]->type
```

来决定整个 TriAttention runtime。

---

## scratch memory 要和 `--kv-size auto` 联动

这也是直接搬旧代码很容易 OOM 的地方。

旧版初始化就分：

```text
dequant_buf  = kv_size * head_dim
unrot_buf    = kv_size * head_dim
score_buf    = n_sampled * kv_size
```

atomic 的 `common_fit_kv_cache()` 会尽量把剩余 VRAM 吃满。如果 fit 结束后 TriAttention 第一次 prune 再临时分几十/几百 MB GPU scratch，很容易把刚刚算好的 fit 推翻。

我会改成 chunked scoring，例如一次 2K/4K candidates，CPU 只保持：

```text
candidate metadata
combined_score[n_candidates]
small score chunk
```

global mode 根本没必要保存 `n_sampled × n_candidates`。

同时把 CUDA scratch + calibration 的 worst-case reserve 提前计入 `common_fit_kv_cache()` 的 device reserve。这样：

```text
model
+ recurrent
+ KV
+ compute
+ TriAttention scratch reserve
+ runtime headroom
<= device memory
```

才是真正的 auto。

---

## recurrent auto-size 我不会第一版就粗暴改公式

现在 `common_dynamic_recurrent_target()` 大约基于：

```text
expected KV per seq = n_ctx / 2
recurrent target ≈ n_ctx_kv / expected KV per seq
```

TriAttention 上来后，理论上确实可以变成：

```text
expected resident KV per seq
    = expected protected KV
    + expected TriAttention retained KV

recurrent target
    ≈ n_ctx_kv / expected resident KV
```

但这里有一个陷阱：如果 `protect_prefill=true`，长 prompt 全部不能 prune，那么不能直接把 `2048 budget` 当成“每 sequence 只用 2048 KV”。那会严重高估可支持 concurrency。

所以我的路线是：第一版 recurrent sizing **维持现在的保守算法**，TriAttention 仅参与 runtime reclaim；等多-slot telemetry 稳定后，再做 tri-aware joint fitter。

真要做 joint fitter，我不会只改一条公式，而是让 `common_fit_kv_cache()` 和 `common_fit_recurrent_cache()` 联合 probe：给定 recurrent slots `R`，得到最大 KV `C(R)`；根据 TriAttention 的 hard-protection + soft residency estimate 检查 `C(R)` 能否支撑 R；然后找最大的可行 R。因为增加 recurrent slots 本身又会吃 VRAM、反过来减少 KV capacity，这是一个联立问题。

---

## 我建议的落地文件

| 文件 | 建议职责 |
|---|---|
| `src/llama-triattention.{h,cpp}` | 从旧仓迁 scoring/calibration；剥离 cache ownership |
| `ggml/src/ggml-cuda/triattention-score.{cu,cuh}` | 迁 CUDA scorer，改 per-device/per-type dispatch |
| `src/llama-memory.h` | 加 generic lossy KV reclaim capability，默认 no-op |
| `src/llama-kv-cells.h` | 增加高效 used-cell / seq-ref enumeration，保持唯一 metadata |
| `src/llama-kv-cache.{h,cpp}` | candidate 枚举、Tri policy、物理 cell eviction |
| `src/llama-kv-cache-iswa.*` | reclaim → base cache |
| `src/llama-memory-hybrid*.{h,cpp}` | reclaim → attention memory |
| `include/llama.h`, `src/llama-context.cpp` | usage/reclaim capability 和 stats 的 public bridge |
| `common/arg.cpp`, `common/common.h` | Tri 配置；增加 `budget=auto` 等 |
| `common/fit.cpp` | Tri scratch reserve；后续 tri-aware recurrent joint fit |
| `tools/server/server-context.cpp` | 把 Tri 插入 `ensure_next_kv_capacity()` pressure ladder |
| `tools/server/server-task.*` | 必要时保存 per-seq prefix/protection metadata |

实施顺序我会是：

1. **先移 scorer，不接 auto manager。** 用 atomic 当前 KV storage 做单 sequence 离线/manual prune，对齐旧仓 score 和 PPL，global mode only。
2. **去掉所有 duplicate cell tracking。** 让 candidate 完全从 `llama_kv_cells` 生成，并跑 save→clear→restore→再次 prune，确保 physical relocation 不影响结果。
3. **加 generic `reclaim_kv()`，接 `ensure_next_kv_capacity()`。** 先 unified KV only，实现“缺 N 个就释放 N+slack 个”，验证 Tri 能减少 preemption。
4. **补 multi-seq/shared-cell。** per-seq frontier、prefix/recent protection、shared cell `max-over-seqs`；同时测 MTP/shared `v_cells_impl`，保证同一 physical resource 只有一个 policy owner。
5. **接 iSWA/hybrid + RAM swap。** 做 quality-first / latency-first 两种 pressure policy，并测试 prompt-cache restore、context shift、spec rollback。
6. **最后优化 GPU 和 auto fit。** mixed K type、多 GPU、chunk scratch、fit reserve；有真实 occupancy telemetry 后再让 TriAttention 反向影响 recurrent target。

如果只让我选一个最关键的重构点，我会先改 **`ensure_next_kv_capacity()` → generic reclaim hook**。一旦这个层次建立起来，TriAttention 就不再是“另一个 cache manager”，而是 atomic 已有 auto-KV / recurrent / RAM demotion manager 可调用的一种资源回收手段；后面的 unified、swap、hybrid 都会自然很多。

## Prompt:
9/2/2026, 10:35:46 AM

我认为 TriAttention 应该做个开关，如果开了，就积极使用 TriAttention，你可以去查 TriAttention 如果不发生断崖损失的情况下，能应用的最大范围是多少，然后就以这个值为目标，积极运用，而且就 take it as granted，不要畏缩。这个余量用完了，才是原版的 kv auto management。但是显存空着也没有用，所以应该是优先瞎用着，有压力 TriAttention 主动压到 9.35% target residency → 这 10.7x 的免费余量真正耗尽以后 → 才让 atomic 原版 KV auto management 接手。

## Response:
9/2/2026, 10:54:06 AM

对，这个层级更合理。我会把之前的设计顺序彻底反过来：

> **TriAttention 开启后，它不是“内存不够时试一下”的保守回收器，而是 unified KV 的第一层虚拟化机制。**
>
> 先把空闲显存全部当普通 KV 使用；第一次出现压力时，主动把可压缩序列压到 **3/32 = 9.375% residency**；之后持续维持这个目标。只有这层约 **10.67x** 的逻辑容量被真正耗尽，atomic 原有的 idle swap、active preemption、recurrent pressure management 才接手。

## 9.375% 这个目标的依据

准确数字不是 9.35%，而是：

```text
3072 / 32768 = 3 / 32 = 9.375%
32768 / 3072 = 10.6667x
```

论文在 Qwen3-8B、AIME25、最大生成长度 32K 的实验中，TriAttention 使用 3072-token KV budget 时取得和 Full Attention 完全相同的 40.8% 准确率，对应 10.7x KV memory reduction；吞吐量同时提高到 2.5x。这个点是论文给出的最强“同准确率压缩锚点”。([arXiv](https://arxiv.org/pdf/2604.04921))

它不是跨模型、跨任务的数学安全定理。论文中 MATH-500 在 1024/32K，也就是 3.125% residency 时已经接近 Full Attention；AIME24 则用 4096/32K，也就是 12.5% 时更接近 Full Attention；DeepSeek-R1-Distill-Qwen-7B 在附录实验中需要 30% 才完全追平 FullKV。递归记忆实验中，TriAttention 在较低到中等压力下一直接近 Full Attention，到更深的递归压力才开始落后。([arXiv](https://arxiv.org/pdf/2604.04921))

但按你的产品取向，我不会让 runtime 根据这些差异自动退到 20%、30% 或动态试探。**开关打开，就明确把 3/32 当成默认运行契约。**质量测试用于发现实现回归，而不是让线上 policy 畏缩。

---

# 最终控制状态机

```text
TriAttention OFF
    └─ 完全保持 atomic 当前行为

TriAttention ON
    │
    ├─ DENSE_FILL
    │    尽量使用全部 physical KV
    │    不因为目标是 9.375% 就提前空出显存
    │
    ├─ 检测到下一批分配将产生 KV pressure
    │
    ├─ TRI_DRAIN
    │    对所有 eligible sequence 主动压到 3/32 target
    │    不是“只释放刚好够用的 deficit”
    │
    ├─ TRI_MAINTAIN
    │    已压缩的 sequence 进入 sticky compressed 状态
    │    后续每增长一个窗口，再压回动态 3/32 target
    │
    ├─ tri_reclaimable > 0
    │    禁止 atomic 原版 KV eviction/swap/preemption 抢先介入
    │
    └─ tri_reclaimable == 0 且仍有压力
         └─ atomic 原版 management 接手
              idle GPU KV -> CPU/RAM state
              active task preemption/retry
              recurrent-slot pressure handling
```

论文采用每生成 128 个 token 触发一次压缩；其 GQA 处理是先在各 query head 内做 z-score normalization，再以 maximum 聚合，保留任何 query head 认为重要的 key。官方 runtime 同样使用默认 128-token interval/window。([arXiv](https://arxiv.org/pdf/2604.04921))

因此我会让 aggressive 模式强制使用：

```text
normalize_scores = true
aggregate        = max / union
window           = 128
protect_prefill  = false
```

尤其是 `protect_prefill=false`。如果整个 prompt 都是硬保护区，就不可能承诺全历史 9.375% residency。官方 TriAttention runtime 的默认值也是不保护整个 prefill；不过其多轮聊天指南为了更保守，会建议更大的 12K budget。这里按你的要求明确选择 aggressive contract。([GitHub](https://github.com/WeianMao/triattention))

---

# 每个 sequence 的 target

我会使用精确整数比例，不使用浮点 0.0935：

```cpp
rho_num = 3;
rho_den = 32;
```

对 sequence `s`：

```text
L_s = 该 sequence 已提交的逻辑 token 数
W_s = recent window + 当前 in-flight/speculative guard

target_s =
    min(
        L_s,
        max(
            W_s,
            ceil(L_s * 3 / 32)
        )
    )
```

例如：

```text
L_s = 32768
target_s = 3072

L_s = 65536
target_s = 6144

L_s = 131072
target_s = 12288
```

小于约 1365 token 的短序列会被 128-token recent window 主导，因此不会真的压到 9.375%；长序列则会渐近精确落到 3/32。

这里必须新增一个 **per-sequence logical frontier / logical token count**。不能再用 resident KV 的 `seq_pos_max()` 推断逻辑长度，因为 TriAttention 删除中间位置后：

```text
logical history length != resident cell count
```

但不需要复制旧实现的 `cell_positions[kv_size]`。物理 cell 的位置、seq refs、occupied 状态仍全部以 `llama_kv_cells` 为唯一真相；额外状态只有每个 sequence 几个标量。

---

# 第一次有压力时要“压到底”，不是按 deficit 修修补补

atomic 当前的 `ensure_next_kv_capacity()` 会计算下一轮需要多少 KV。TriAttention 开启后，我会把逻辑改成：

```text
kv_pressure =
    physical_used
  + next_batch_required
  + transient_guard
  > physical_capacity
```

一旦 `kv_pressure=true`：

```text
对所有当前 resident、支持 TriAttention 的 sequence：

    score all candidates
    select target_s
    compact survivors
    mark sequence as compressed
```

即使本次只缺 200 个 cell，也不只释放 200 个，而是把整批 eligible sequences 一次性压到各自的 3/32 floor。这样才真正建立一个大的免费 headroom，避免每几十个 token 就重新触发一次全局评分。

已经进入 compressed 状态的 sequence 是 sticky 的：

```text
resident_s > target_s + 128
    => prune back to target_s
```

新建 sequence 仍然从 dense 开始，继续利用当前空闲的物理 KV；等下一次全局压力再加入 compressed 集合。这同时满足：

- 空显存绝不浪费；
- 一旦有压力就充分释放 TriAttention 余量；
- 已经付过压缩代价的序列不会重新无界增长。

---

# 多 sequence 和 shared-prefix 的 target 不能简单相加

unified KV 中，一个 physical cell 可能同时属于多个 sequence。正确做法是：

1. 每个 sequence 独立计算 `target_s`。
2. 对该 sequence 的 resident cells 计算 TriAttention score。
3. 为每个 sequence 选出自己的 top-`target_s`。
4. 最终 physical keep-set 是所有 sequence keep-set 的并集。
5. 一个 shared cell 只要被任意 sequence 选中，就保留。

概念上：

```text
keep(cell) =
    OR over referenced sequences
        cell ∈ top_target(sequence)
```

cell 的评分同样采用：

```text
score(cell, seq)
    = max over calibrated query heads
        normalized TriAttention score

score(cell)
    = max over referenced sequences
        score(cell, seq)
```

因此不会为了一个 sequence 的压缩目标，把另一个 sequence 非常需要的 shared cell 删除。

真正的剩余 Tri 余量定义为：

```text
tri_reclaimable =
    current physical occupied cells
  - physical union keep-set size
```

只要它大于零，atomic 原管理器就不能提前 swap 或 preempt。由于 recent window、in-flight token 和 shared union 的存在，实际物理比例偶尔会高于 9.375%，但这是根据真实约束算出的精确 floor，而不是保守地把 policy 提高到另一个比例。

---

# 必须做真正的 compaction

旧本地移植版目前只是：

```text
标记低分 cell 为空
保留 surviving cell 原来的 physical index
```

这能释放可复用 slot，但不足以完整获得 TriAttention 的 throughput 收益。

atomic 的 `llama_kv_cache::get_n_kv()` 依赖 `used_max_p1()`。如果留下大量空洞，而最新 token 还在高 index：

```text
occupied cells = 3000
used_max_p1    = 30000
```

attention graph 仍可能按接近 30K 的范围计算和 mask。

所以每次 aggressive drain 后需要一个原子 compaction transaction：

```text
old physical index -> new dense physical index

同步移动：
    llama_kv_cells metadata
    每一层 K rows
    每一层 V rows
    shared seq references
    v_heads
    MTP/shared-cache owner mapping
```

压缩后应达到：

```text
used indices = [0, physical_used)
v_head       = physical_used
get_n_kv     ≈ PAD(physical_used, 256)
```

NVIDIA 当前的正式集成也是把 TriAttention 放在 KV cache manager 上，完成 physical cache compaction 后继续使用标准 dense attention，而不是只留下逻辑洞。这个方向与 atomic 的目标一致。([GitHub](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md))

compaction 必须发生在：

```text
llama_synchronize()
之后
下一次 init_batch()/graph prepare()
之前
```

不能继续塞在旧版 `apply_ubatch()` 尾部，否则容易和当前 graph、speculative token、shared MTP rows 冲突。

---

# 在 atomic 里的具体落点

## `src/llama-memory.h`

增加通用能力，而不是把 TriAttention 名字扩散到整个系统：

```text
reclaim_kv_to_target(request) -> result
```

request 至少包含：

```text
drain_to_policy_floor = true
protected in-flight seq/range
confirmed frontier
reason = capacity_pressure
```

result 包含：

```text
physical_before
physical_after
freed
target
reclaimable_remaining
exhausted
```

默认 memory implementation 返回 unsupported。

## `src/llama-kv-cache.cpp`

负责：

```text
从 llama_kv_cells 枚举 candidates
调用 Tri scorer
生成 per-seq top-k 和 union keep-set
删除 losers
执行 K/V compaction
更新 sparse/frontier metadata
```

不再持有旧版 `triattention_state::cell_positions`。

## wrappers

```text
llama_kv_cache_iswa
    reclaim -> kv_base
    SWA recent cache不压

llama_memory_hybrid
    reclaim -> mem_attn
    mem_recr不动

llama_memory_hybrid_iswa
    reclaim -> attention base cache
```

recurrent-only architecture 不适用 TriAttention。

## `tools/server/server-context.cpp`

在 `ensure_next_kv_capacity()` 中把顺序改为：

```text
1. 计算 KV pressure / recurrent pressure

2. KV pressure 且 TriAttention ON
      reclaim all pools to 3/32
      重新读取 usage

3. 仍然 KV 不足，且 tri_reclaimable == 0
      执行当前 try_clear_idle_slots()

4. 仍不足
      执行当前 active slot preemption/retry
```

如果只有 recurrent slot 不足，TriAttention无法释放 recurrent cell，可以直接走原逻辑；这不是提前退缩，而是两个资源维度互不替代。

target context 和 draft context 如果共享 `v_cells_impl`，必须按 physical owner 去重，只调用一次；如果是两个独立 KV pool，则分别计算各自 Tri floor。

---

# `--kv-size auto` 和 recurrent auto-fit 也要承认 10.67x

这部分我也会按你的思路改，不再保守维持原 recurrent 估计。

现在 physical KV fit 仍应尽量使用所有可用显存。TriAttention 不会把 CUDA allocation 缩成 9.375%；它只是让已分配的 unified pool 能反复容纳更多逻辑 token：

```text
physical KV capacity = C_phys
Tri virtual capacity ≈ C_phys * 32 / 3
```

因此原来的 recurrent 期望并发公式应从：

```text
R ≈ 2 * C_phys / n_ctx
```

变为：

```text
R ≈ 2 * C_phys / (rho * n_ctx)

rho = 3/32

R ≈ 2 * C_phys * 32 / (3 * n_ctx)
```

也就是期望并发上限约放大 10.67 倍，再由 `n_seq_max` 和真实 recurrent memory 限制。

不过 recurrent slots 本身会吃显存，继而降低 `C_phys`，因此不能只乘一个常数。应把 `common_fit_kv_cache()` 和 `common_fit_recurrent_cache()` 改成联合搜索：

```text
对候选 recurrent slots R：

    先计算分配 R 个 recurrent slots 后
    剩余显存可容纳的最大 physical KV：C_phys(R)

    expected physical KV per sequence =
        max(128, ceil((n_ctx / 2) * 3 / 32))

    检查：
        R * expected physical KV per sequence
        + batch/spec reserve
        <= C_phys(R)

选择最大的可行 R
```

这让 fitted recurrent capacity 真正建立在 TriAttention 的 virtual KV envelope 上。

此外，TriAttention ON 时，`common_fit_kv_cache()` 不应再要求：

```text
physical n_ctx_kv >= per-sequence n_ctx
```

最低物理 KV 只需要覆盖：

```text
ceil(n_ctx * 3/32)
+ n_ubatch
+ speculative guard
+ compaction/scoring reserve
```

否则一个 128K logical context 仍被强制要求分配 128K physical cells，等于根本没有利用 TriAttention 的容量虚拟化。

---

# GPU/CPU auto swap 的新语义

顺序明确固定为：

```text
TriAttention lossy compression
    ↓ 3/32 envelope exhausted
idle sequence state -> CPU/RAM
    ↓ CPU budget exhausted
active preemption/retry
```

被 TriAttention 压缩后的 sequence 保存到 RAM 时，只保存 surviving KV，加上：

```text
logical frontier
logical token count
compressed epoch
target residency numerator/denominator
calibration fingerprint
```

恢复后仍然是 compressed state，不会假装它是连续 Full KV，也不需要重新生成已删除 cell。

这还会顺带降低 CPU swap 的序列状态大小和传输量。

如果加载时 TriAttention calibration fingerprint 不一致，则拒绝使用这份 sparse state并重新 prefill；不能把用模型 A 评分删过的状态装进模型 B。

---

# 开关语义要硬

我建议最终接口是：

```text
--triattention
--triattention-stats PATH
```

默认内置：

```text
target residency = 3/32
window           = 128
normalize        = on
aggregation      = max/union
protect prefill  = off
policy           = fill-then-drain
```

关闭时：

```text
--no-triattention
```

严格恢复当前 atomic 行为。

开启时如果发生以下情况：

```text
calibration 缺失
calibration 与模型不匹配
cache layout 未实现
RoPE/layout 不支持
compaction backend 不支持
```

启动直接报错，不允许打印 warning 后悄悄退回普通 KV。NVIDIA 当前实现也维护明确 support matrix，并将 TriAttention作为 KV manager 能力，而不是假定所有 SWA、hybrid、MLA layout 都天然可用。([GitHub](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md))

---

最终的资源模型就是：

```text
physical unified KV:
    仍然吃满可用显存

logical unified KV:
    首次压力前按 1:1 dense 使用
    首次压力后按 3:32 residency 维持

atomic fallback:
    只有当所有 eligible sequence 都已经到 3/32 floor，
    且 union hard-keep 之外再无可删 physical cell 时才启动
```

这样 TriAttention 才不是 atomic management 的一个小插件，而是它前面的 **10.67x KV virtual-capacity layer**；原有 auto swap/preemption 则成为这层虚拟容量真正耗尽后的第二级保障。

---

**Sources:**

- [https://arxiv.org/pdf/2604.04921](https://arxiv.org/pdf/2604.04921)
- [https://github.com/WeianMao/triattention](https://github.com/WeianMao/triattention)

## Prompt:
9/2/2026, 10:59:58 AM

好的请你给出实现规划

## Response:
9/2/2026, 10:59:58 AM

# 实现规划

## 一、冻结产品语义

TriAttention 开关的行为先明确为不可歧义的契约：

```text
TriAttention 关闭
    完全维持 atomic 当前行为

TriAttention 开启
    1. unified KV 有空闲时继续 dense 使用，不提前压缩
    2. 下一批将超过 physical KV capacity 时：
       把所有 eligible sequence 一次性压到 3/32 residency floor
    3. 重新检查容量
    4. 仍不足，说明 TriAttention 免费余量已耗尽
    5. 才执行 atomic 原有：
       idle state -> RAM
       active slot preemption/retry
       recurrent pressure management
```

目标不使用浮点近似，而固定为：

```text
target numerator   = 3
target denominator = 32
target residency   = 9.375%
virtual factor     = 10.6667x
recent window      = 128
```

每个 sequence 的目标：

```text
logical_tokens_s = server 已提交逻辑 token 数

tail_guard_s =
    max(
        128,
        当前待提交 token 数,
        speculative rollback 所需尾部长度
    )

target_s =
    min(
        logical_tokens_s,
        max(
            tail_guard_s,
            ceil(logical_tokens_s * 3 / 32)
        )
    )
```

这里的 128 是硬保留窗口，不是周期触发器。显存仍然采用高水位策略：有空间就保留 dense extras，有压力才压到底。

以下原则不做自动妥协：

- 不根据 runtime 质量猜测把目标改成 12.5%、20% 或 30%。
- 不因某次 reclaim 收益不足而静默关闭 TriAttention。
- 不支持的 cache/model/backend 配置在启动阶段明确报错。
- 质量测试失败时修评分、校准或实现，不自动提高 residency 掩盖问题。

---

# 二、总架构

实现拆成四个职责明确的部件。

## 1. `llama_triattention_scorer`

只负责：

```text
K tensor + calibration + candidate cells + positions + sequence frontier
    ->
importance scores
```

它不再负责：

- 跟踪 KV cell 生命周期；
- 保存 `cell_positions[kv_size]`；
- 判断何时 prune；
- 修改 `llama_kv_cells`；
- 调用 `cells.rm()`；
- 管理 unified sequence；
- 决定 atomic fallback。

从旧仓迁移时，保留 calibration loader、RoPE inversion、TurboQuant dequant、三角评分和 CUDA kernel；删除旧的 cache ownership。

## 2. `llama_memory_i` reclaim 契约

在 memory 抽象中增加一个通用的 lossy KV reclaim 能力。概念接口为：

```text
reclaim_kv(context, request) -> result
```

请求至少包含：

```text
required_free_cells
drain_to_policy_floor = true
per-sequence logical length hints
per-sequence tail guards
```

结果至少包含：

```text
supported
changed
physical_before
physical_after
physical_freed
policy_floor
hard_keep
shared_keep
floor_reached
capacity_satisfied
```

默认 memory implementation 返回 unsupported。

转发关系：

```text
llama_kv_cache
    直接实现

llama_kv_cache_iswa
    转发到 kv_base
    kv_swa 不做 TriAttention

llama_memory_hybrid
    转发到 mem_attn
    mem_recr 不动

llama_memory_hybrid_iswa
    转发到 attention 的 base cache

recurrent-only memory
    unsupported
```

实验接口建议先放在 `src/llama-ext.h`，避免一开始扩大稳定 public C API。

## 3. `llama_kv_cache` reclaim 与 compaction

KV cache 负责：

```text
枚举 authoritative cells
-> 按 sequence 评分和选 top-k
-> 修改 sequence references
-> 计算 physical keep-set
-> pack K/V
-> 原子提交新的 cell metadata
```

`llama_kv_cells` 是唯一 cell metadata 真相。不会从旧实现迁移：

```text
triattention_state::cell_positions
triattention_on_token_added
triattention_on_cell_removed
triattention_on_position_shift
triattention_on_reset
```

## 4. server pressure planner

`tools/server/server-context.cpp::ensure_next_kv_capacity()` 是唯一资源优先级控制器。

TriAttention 不在 `apply_ubatch()` 尾部自行触发。它必须在下一次 batch 准备前、没有执行中 graph 时，由 pressure planner 发起同步 reclaim transaction。

---

# 三、分阶段实施

## 阶段 0：工作区隔离和基准固化

当前目标仓库 `master` 的 `common/common.cpp` 有未提交的 recurrent-fit 修改，并且正好与后面的 Tri-aware joint fitting 重叠。

实际实施时应：

- 从当前 HEAD `823dc5097` 新建独立 worktree；
- 不修改当前 dirty checkout；
- 前七个阶段不碰现有 `common/common.cpp` 修改；
- 到自动拟合阶段，再把这份 downward recurrent search 修改作为输入合并。

同时记录以下基准：

| 场景 | 需要记录 |
|---|---|
| Tri OFF，单序列 | logits、PPL、tokens/s、KV used |
| Tri OFF，unified 多 slot | preemption 次数、RAM swap 次数 |
| 当前 auto KV | fitted physical KV、recurrent slots |
| 源仓 TriAttention | 相同 calibration 下的 score 和 keep-set |

这一阶段不改变 runtime 行为。

---

## 阶段 1：迁移纯 CPU scorer

### 修改范围

- 新增 `src/llama-triattention.h`
- 新增 `src/llama-triattention.cpp`
- 修改 `src/CMakeLists.txt`
- 新增 `tests/test-triattention-score.cpp`

### 从旧仓保留

- `.triattention` calibration 解析；
- calibration/model compatibility 检查；
- RoPE frequency 和 geometric offsets；
- pre-RoPE K 恢复；
- MLR/trigonometric score；
- F16、BF16、Q8、TurboQuant dequant；
- per-head z-score normalization；
- global max/union 聚合。

### 从旧仓删除

- `triattention_state::kv_size`
- `cell_positions`
- `absolute_position`
- `prefix_length`
- `should_prune()`
- `prune_impl()` 中的 cell 删除
- 固定 `budget/slack/interval` trigger
- GPU lazy-init
- `n_sampled * kv_size` 的永久 score buffer

### 新 scorer 内存模型

按 sequence、按 sampled head 流式处理：

```text
一个 sequence：
    一个 candidate index vector
    一个 score vector
    一个 combined max vector

处理一个 sampled head：
    score all candidates
    z-score
    combined[i] = max(combined[i], score[i])
    复用 score vector
```

因此 host scratch 从：

```text
O(n_sampled * n_candidates)
```

降为：

```text
O(n_candidates)
```

### 验收门槛

- 同一 calibration、K tensor、position 和 frontier 下，新旧 scorer 的 score 在设定误差内一致；
- keep-set 完全一致，平分时按 position/cell index 做确定性排序；
- TriAttention 未启用时没有行为变化；
- 暂时不接 KV cache。

---

## 阶段 2：增加 sparse reclaim 内部契约

### 修改范围

- `src/llama-memory.h`
- `src/llama-ext.h`
- `src/llama-context.h`
- `src/llama-context.cpp`
- 各 memory wrapper 的头文件和实现

### 内部数据结构

建议定义：

```text
llama_memory_kv_reclaim_seq_hint
    seq_id
    logical_tokens
    tail_guard
    eligible

llama_memory_kv_reclaim_request
    required_free
    drain_to_floor
    seq_hints

llama_memory_kv_reclaim_result
    supported
    changed
    capacity_satisfied
    floor_reached
    physical_before
    physical_after
    physical_freed
    target_references
    hard_keep
    shared_keep
```

同时增加两个能力：

```text
memory exposes sparse positions
physical KV owner identity
```

owner identity 用于 MTP/shared cache 去重。两个 context 如果共享同一套 `v_cells_impl` 和 K/V storage，只允许 physical owner 执行一次 reclaim。

### context transaction

增加一个类似 `llama_context::memory_update()` 的同步入口：

```text
llama_context::memory_reclaim_kv(request)
```

内部顺序：

```text
llama_synchronize()
调用 memory->reclaim_kv()
如果 storage/metadata 发生变化：
    reset scheduler
    invalidate previous graph result
    reserve new worst-case graph
返回 reclaim result
```

### 验收门槛

- 所有现有 memory 类型编译通过；
- 默认实现为 no-op/unsupported；
- Tri OFF 的所有测试保持原状；
- wrapper 转发和 physical owner 去重有单元测试。

---

## 阶段 3：实现原子 KV compaction

这是整个项目的关键基础阶段。只做 `cells.rm()` 会释放 slot，但 `get_n_kv()` 仍由 `used_max_p1()` 决定，无法真正减少 attention 工作范围。

atomic 历史中在 commit `9ebebef62` 之前存在 graph-based KV defrag，可作为机械参考，但不能直接恢复，因为旧实现：

- 只支持单 stream；
- 使用 hparams 宽度，而 atomic 的 TurboQuant cache 可能有实际 padding；
- 没处理 layer-adaptive K/V type；
- metadata 在 graph 成功前就被修改；
- 旧 `llama_kv_cells::mv()` 没处理当前全部字段。

### 修改范围

- `src/llama-kv-cells.h`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `tests/test-kv-cells.cpp`
- 新增 KV pack integration test

### pack plan

生成稳定的映射：

```text
retained old cells，按原 physical index 升序
    ->
new indices [0, retained_count)
```

连续 move 合并为：

```text
src_begin
dst_begin
length
```

所有移动都满足：

```text
dst <= src
```

因此可以原地向低地址 pack，不需要第二份完整 KV buffer。

### transaction 顺序

```text
1. 从原 cells 构造 post-reclaim metadata 副本
2. 从该副本生成 stable pack plan
3. 原 metadata 保持不变
4. 执行所有 K/V downward copies
5. copy 成功后才提交 post metadata
6. v_head = packed_used
7. trailing rows 仅标记为空，不要求清零
8. invalidate graph
```

这样即使 copy graph 失败，原 retained source rows仍位于原位置，metadata 仍指向原 rows；被覆盖的 destination 原本是 hole 或已淘汰 cell。

### tensor 处理

必须使用实际 tensor descriptor：

```text
layer.k->ne[0]
layer.k->nb[]
layer.v->ne[]
layer.v->nb[]
```

不能假定：

```text
hparams.n_embd_k_gqa == physical K row width
```

因为 atomic 存在：

- Turbo head padding；
- layer-adaptive K/V type；
- V transposed 与非 transposed 两种 layout；
- 每层实际 type 不同。

pack 成功后必须满足：

```text
cells.used_max_p1() == cells.get_used()
get_n_kv() == PAD(cells.get_used(), n_pad)
```

### 验收门槛

测试至少覆盖：

- shared cell；
- duplicate position；
- F16/Q8/Turbo2/Turbo3/Turbo4；
- padded head dimension；
- V transposed 和 non-transposed；
- 多层不同 K/V type；
- pack 前后 retained K/V bytes 完全一致；
- pack 后下一 token logits 一致；
- pack 失败不会提交 metadata。

在这一阶段完成前，不接 aggressive eviction。

---

## 阶段 4：单序列 3/32 reclaim

### 配置层

在 `llama_context_params` 末尾增加 Tri 启用信息，至少包括：

```text
triattention_enabled
target_num = 3
target_den = 32
recent_window = 128
score_chunk
```

校准文件路径和完整 scorer 配置仍由 common 层传给最终 context，避免 auto-fit probe 每次加载完整 calibration。

修改：

- `include/llama.h`
- `src/llama-cparams.h`
- `src/llama-context.cpp`
- `common/common.h`
- `common/common.cpp`
- `common/arg.cpp`

CLI 语义：

```text
--triattention
--triattention-stats PATH
--triattention-window 128
--triattention-log
```

`--triattention-stats` 可以隐式启用，但显式 `--triattention` 缺少 stats 时必须启动失败。

### 放宽 physical KV 限制

当前 `llama-context.cpp` 强制：

```text
n_ctx_kv >= n_ctx_seq
```

改为：

```text
Tri OFF:
    维持原限制

Tri ON:
    允许 n_ctx_kv < n_ctx_seq
    但必须满足最小 batch/tail guard
```

### 单序列 reclaim

高水位压力发生时：

```text
logical_tokens =
    request hint
    若无 hint，则使用 seq_pos_max + 1

target =
    max(
        tail_guard,
        ceil(logical_tokens * 3 / 32)
    )
```

保留：

- 最新 `tail_guard`；
- scorer 的 top-k；
- 当前即将使用的 in-flight guard。

不保护整个 prefill。

淘汰后立即执行阶段 3 的 pack。

### llama_decode fallback

server 外的 CLI/API 调用也要可用。`llama_decode()` 在 `init_batch()` 返回 `FAILED_PREPARE` 后，顺序改为：

```text
尚未尝试 Tri reclaim：
    drain to floor
    retry init_batch

仍失败：
    执行现有 memory_update(true)
    retry

仍失败：
    返回原错误
```

### 验收门槛

- 无物理压力时不触发；
- 首次压力时不是只释放 deficit，而是直接压到 floor；
- 32K logical context 的长期目标是 3072 references，另加必要 tail guard 限制；
- pack 后实际 `n_kv` 跟 resident 数下降；
- Tri OFF 完全保持原行为。

---

## 阶段 5：unified 多 sequence 和 shared prefix

这是从旧实现迁移到 atomic 的主要语义重写。

### per-sequence 选择

对每个 sequence 独立计算：

```text
frontier_s
logical_tokens_s
target_s
candidate refs
score(cell, seq)
top_target_s
```

一个 physical cell 可能被多个 sequence 引用。因此先做 reference-level 选择：

```text
selected(cell, seq)
```

再做：

```text
对于 cell 的每个 seq reference：
    未被该 seq 选中 -> 删除该 reference

cell 所有 reference 均被删除：
    physical cell 才真正释放
```

最终 physical keep-set 等价于：

```text
OR over all referenced sequences selected(cell, seq)
```

一个 shared cell 只要任意 sequence 需要，就保留 physical row。

### 准确计量

不能把“删除了多少 sequence reference”当成释放的 KV 容量。

必须分别统计：

```text
references_removed
physical_cells_freed
shared_cells_retained
```

server 容量判断只使用 `physical_cells_freed`。

### wrappers

这一阶段接入：

```text
llama_kv_cache_iswa
    只压 base cache

llama_memory_hybrid
    只压 attention cache

llama_memory_hybrid_iswa
    只压 attention base cache
```

MTP/shared cache 增加 owner 去重，view cache 不允许独立 pack。

### 第一版支持矩阵

支持：

- unified KV；
- 1D standard RoPE；
- plain attention KV；
- iSWA base；
- hybrid 的 standard attention 部分；
- CPU、CUDA 可访问的标准/Turbo K 类型。

启动失败而不是静默 fallback：

- non-unified KV；
- recurrent-only model；
- MLA、DSA、DSV4、MSA；
- M-RoPE/2D positions；
- 不支持的 K type；
- calibration 与模型不匹配；
- shared cache 无法解析 physical owner。

### 验收门槛

- 两个 sequence 共享 1K prefix 时，一个 sequence 不得误删另一个需要的 shared cell；
- 每个 sequence 的 reference 数到达自己的 floor；
- physical keep-set 为各 sequence keep-set 的精确并集；
- owner cache 只执行一次 pack；
- hybrid recurrent state 不发生变化。

---

## 阶段 6：接入 atomic server 优先级

修改核心：

```text
tools/server/server-context.cpp::ensure_next_kv_capacity()
```

新顺序：

```text
读取 target/draft KV usage 和 recurrent usage

计算下一批 required KV/recurrent

容量足够：
    return true

存在 KV pressure 且 Tri ON：
    对每个独立 physical KV owner drain_to_floor
    刷新全部 usage

刷新后容量足够：
    return true

仍有 recurrent-only pressure：
    进入原 recurrent handling

Tri 已到 floor 且 KV 仍不足：
    try_clear_idle_slots()

仍不足：
    find_preemption_victim()
    preempt_slot()
```

关键约束：

```text
只要一次 full drain 尚未完成，
try_clear_idle_slots() 和 preempt_slot() 不得先执行。
```

target 与 draft：

- 两者若共享 physical owner，只 reclaim 一次；
- 两者若独立，分别 reclaim；
- 独立 draft model 需要自己的 calibration；
- MTP 同模型共享 storage 时复用主 calibration/owner。

### server 提供的 seq hints

`ensure_next_kv_capacity()` 根据 slot 构造：

```text
seq_id
logical_tokens = slot.prompt.n_tokens()
tail_guard
eligible
```

`tail_guard` 包含：

- 128 recent window；
- 本轮预计 append tokens；
- speculative maximum；
- rollback/checkpoint 尾部要求。

### 遥测

至少增加：

```text
tri_drain_count
tri_cells_before
tri_cells_after
tri_cells_freed
tri_target_references
tri_shared_keep
tri_hard_keep
tri_score_ms
tri_pack_ms
tri_floor_exhausted_count
atomic_fallback_after_tri_count
```

日志必须能明确区分：

```text
Tri drain satisfied pressure
Tri floor exhausted
fallback caused by recurrent pressure
fallback caused by physical KV pressure
```

### server 测试

基于已有 unified preemption 测试派生：

1. `kv_size=256`、4 slots，Tri OFF：保留当前行为。
2. 相同配置，Tri ON：
   - 首次压力先出现 drain；
   - idle clear/preemption 计数为 0。
3. 提高逻辑总长度直到 floor 总和也超过 physical KV：
   - `floor_exhausted=true`；
   - 原 preemption/retry 路径开始工作；
   - 每个请求的 `n_predict` 预算仍完整；
   - streaming 不重复 token。
4. 仅 recurrent capacity 不足：
   - 不做无意义 KV drain；
   - 直接进入原 recurrent handling。

---

## 阶段 7：sparse state、RAM swap 和 checkpoint

atomic 的 KV state 已经只序列化 occupied cells，并保存每个 cell 的逻辑 position。恢复时会重新分配 physical rows，因此不需要保存 old physical index。

但必须增加 sparse layout 元数据，避免调用者把 sparse state 当成 FullKV state：

```text
layout = sparse
policy = triattention
target = 3/32
recent_window = 128
calibration fingerprint
```

### 审计范围

重点审核：

- `llama-kv-cache.cpp::state_write/state_read`
- `server_slot::prompt_save/prompt_load`
- `server_prompt_cache`
- context checkpoints
- speculative checkpoints
- `seq_rm/seq_cp/seq_keep/seq_add/seq_div`
- `llama_memory_seq_pos_min/max`
- cache reuse/LCP 路径

当前 `llama_memory_seq_pos_min/max` 文档声称 `[min,max]` 中全部 position 存在。TriAttention 会打破该契约，因此需增加明确 capability：

```text
memory_positions_are_sparse()
```

依赖连续 position 的优化必须在 sparse 时关闭或自行验证。

已经安全的路径：

- 下一 token position 依赖 `seq_pos_max+1`，因为 latest tail 被硬保护；
- `get_n_kv_pos_contiguous()` 已逐 cell 验证，不连续时会退回普通 mask。

仍需通过测试确认的路径：

- prompt LCP reuse；
- partial rollback；
- context shift；
- checkpoint restore；
- RAM save/clear/load 后继续生成。

### 验收门槛

完整流程：

```text
dense prefill
-> Tri drain
-> save to RAM
-> clear GPU KV
-> restore到不同 physical indices
-> 继续生成下一 token
```

要求：

- restored positions 和 K/V survivors 一致；
- 下一 token logits 与保存前同一 sparse state 一致；
- 不重新生成已被 TriAttention 淘汰的 token；
- target residency 不因 restore 丢失；
- calibration/layout 不匹配时明确拒绝加载。

---

## 阶段 8：Tri-aware KV 和 recurrent auto-fit

这一阶段才修改现有的 `common/common.cpp` recurrent-fit 区域，并合并当前工作树中的 downward search 变更。

### physical KV 仍然吃满显存

`--kv-size auto` 的最大值搜索逻辑不改变：

```text
尽可能分配更多 physical KV cells
```

TriAttention 只改变最小可行 physical KV 和 expected residency，不会故意留下显存。

### Tri-aware minimum KV

当前 `common_fit_kv_cache()` 使用：

```text
n_min = n_ctx_seq
```

Tri ON 时改为：

```text
tri_floor =
    ceil(n_ctx_seq * 3 / 32)

operational_floor =
    recent_window
  + n_ubatch
  + speculative guard
  + pack/scoring reserve

n_min =
    aligned max(tri_floor, operational_floor)
```

从而允许：

```text
logical context = 128K
physical KV < 128K
```

但仍能在分块 prefill 中持续 drain。

### recurrent expected residency

当前 `common_dynamic_recurrent_target()` 假定 sequence length 均匀分布于 `[0,n_ctx]`，平均 KV residency 约为 `n_ctx/2`。

Tri ON 后改成：

```text
expected resident KV per seq
    = E[max(recent_window, ceil(3L/32))]
      + transient growth guard
```

其中 `L` 在 `[0,n_ctx]` 上按现有均匀假设取期望。实现可以使用整数闭式公式，避免浮点边界抖动。

然后：

```text
target recurrent slots
    = min(
        n_seq_max,
        physical_kv_capacity / expected_resident_kv_per_seq
    )
```

### 联合拟合

因为更多 recurrent slots 会占用显存并压缩 physical KV，不能只把旧结果乘以 10.67。

使用联合 probe：

```text
候选 recurrent slots R
    ->
probe 得到该 R 下最大 physical KV C(R)
    ->
检查 C(R) 是否覆盖 R 个 sequence 的 expected Tri residency
```

选择最大的可行 R。

当前 `common/common.cpp` 未提交修改已经把 recurrent target 搜索改成向下寻找最大可行值；应保留这一方向，只替换 target residency 计算和 Tri-aware `n_min`，而不是重写整段循环。

### auto slots

server 当前在 dynamic KV 下先把 `n_parallel` 设为最大 logical slots，再根据最终 `llama_n_seq_recurrent()` 下调。Tri-aware recurrent fit 完成后，这条机制自然会得到更高的可用 slot 数，不需要再加第二套 slot 推导器。

### 验收门槛

- Tri OFF 拟合结果不变；
- Tri ON 允许 physical KV 小于 logical per-seq context；
- fitted recurrent slots 使用 Tri expected residency；
- 所有 probe 后实际 context 能成功创建；
- 运行到 high-water drain 时不发生首次评分额外 OOM；
- 当前 dirty `common/common.cpp` 修改没有丢失。

---

## 阶段 9：CUDA、多 GPU 和 scratch 纳入 auto-fit

旧 CUDA 实现不能直接复制，因为它把一个 `k_type` 固定在全局 GPU state；atomic 可能每层 K type 不同，并可能分布在不同 device。

### 新 storage descriptor

每个 sampled layer 使用：

```text
model layer id
K tensor
device/backend
K type
logical head dimension
physical padded head dimension
row stride
number of KV heads
need Turbo inverse rotation
```

评分按：

```text
(device, K type, padded head dimension)
```

分组。

### GPU state

- 每个 device 一份 calibration 常量；
- 每个 type group 独立 dispatch；
- 每次只保留一个 sampled head 的 score vector；
- candidate cell indices 和 positions 分 chunk 上传；
- host 侧完成确定性 normalization 和 max/union；
- 后续再考虑把 normalization/top-k 移到 GPU。

### backend fallback

- K 在 host：CPU scorer；
- K 在 CUDA：CUDA scorer；
- 未实现的 device backend：硬开关模式下启动失败；
- 开发验证模式可允许 candidate rows D2H staging，但不作为正式性能路径。

### scratch 与 auto-fit

scoring/pack scratch 不能在第一次 pressure 时临时申请一大块未计入预算的显存。

在 auto-fit probe 前读取 calibration header，计算：

```text
per-device calibration bytes
candidate index buffers
score chunk
normalization buffer
pack graph reserve
```

并纳入 context memory breakdown 或 device reserve。

### 验收门槛

- CPU/CUDA score 和 keep-set 一致；
- layer-adaptive K type 混用正确；
- multi-GPU 每层在所属设备评分；
- 不把完整 K tensor复制到 host；
- 第一次 drain 无额外 OOM；
- pack 后 attention 范围和吞吐随 resident KV 实际下降。

---

# 四、关键不变量

实现过程中始终用这些不变量做断言。

## KV metadata

```text
llama_kv_cells 是唯一 metadata owner
empty cell 没有任何 sequence refs
non-empty cell 至少有一个 sequence ref
seq_get_used(s) 等于实际引用数
get_kv_used() 统计 physical cells，不统计 refs
```

## Tri target

```text
对每个 eligible sequence：

resident_refs_s <= max(tail_guard_s, ceil(3 * logical_tokens_s / 32))

允许高于该值的唯一原因：
    shared physical cell 被其他 sequence 保留
    hard tail/in-flight guard
```

注意 reference target 和 physical target是两个不同指标。

## compaction

```text
pack 前后 retained cell 的：
    position
    ext
    seq refs
    K bytes
    V bytes
全部一致

pack 后：
    occupied physical indices = [0, used)
```

## fallback 顺序

```text
KV pressure:
    Tri full drain
    -> refresh usage
    -> atomic fallback

不得：
    idle clear
    -> 再尝试 Tri

不得：
    preempt active slot
    -> 再发现其实仍有 Tri reclaimable cells
```

## 开关隔离

```text
Tri OFF：
    不创建 scorer
    不分配 Tri scratch
    不放宽 n_ctx_kv 校验
    不启用 sparse semantics
    不改变 auto-fit
```

---

# 五、测试矩阵

| 层级 | 主要测试 |
|---|---|
| scorer | calibration 解析、RoPE inversion、各 K type、源仓 score parity |
| selection | 3/32 top-k、tail guard、确定性平分、不同 sequence frontier |
| cells | move、shared refs、duplicate positions、per-seq counts |
| compaction | K/V byte parity、Turbo padding、V layout、mixed layer type |
| sparse state | save/load、换 physical index、checkpoint、context shift |
| unified | 多 slot、shared prefix、idle/active 混合 |
| server policy | 无压力不压、压力压到底、floor 前不 preempt、floor 后恢复原管理 |
| hybrid | attention 被压、recurrent 不变 |
| speculative | target/draft 独立 owner、MTP shared owner 去重、rollback guard |
| fit | Tri OFF parity、Tri physical minimum、recurrent joint probe |
| backend | CPU、单 CUDA、多 CUDA、host/device mixed layers |
| 长稳测试 | 多轮 drain、RAM swap、restore、preempt/retry、streaming |

---

# 六、最终验收标准

功能完成应同时满足：

1. **Tri OFF 零回归。** 现有 unified KV、recurrent fitting、RAM prompt cache 和 preemption 测试全部通过。

2. **Tri ON 严格执行 fill-first。** physical KV 未到高水位时不发生任何 eviction。

3. **第一次压力严格 drain-to-floor。** 不是只补 deficit，而是把全部 eligible sequences 排到 `3/32` floor。

4. **真正减少计算范围。** drain 后不只是 `get_kv_used()` 下降，`used_max_p1()` 和 graph 的 `n_kv` 也同步下降。

5. **Tri 余量优先。** 只要 full drain 后能够满足下一批，idle RAM demotion 和 active preemption 都不得发生。

6. **floor 耗尽后无缝回到 atomic。** 原有 save/clear/retry 路径继续保证请求完成、predict budget 不丢失、streaming 不重复 token。

7. **state 可迁移。** sparse state 保存、清除、恢复到不同 physical indices 后可以继续生成。

8. **auto-fit 承认 10.67x envelope。** physical KV 可以小于 logical context，recurrent target 使用 `3/32` expected residency，并通过真实 probe 限制。

9. **不静默退让。** calibration、layout、backend 或模型不受支持时启动失败，而不是偷偷切回 FullKV 或提高 residency。

10. **质量问题不改变产品目标。** 3/32 是固定契约；质量门用于发现 scorer、calibration 和 cache 实现错误，不作为 runtime 自动升配机制。

---

# 七、推荐实施依赖链

```text
阶段 1：纯 scorer
        │
        ├──────────────┐
        ▼              ▼
阶段 2：reclaim 契约   阶段 3：compaction
        │              │
        └──────┬───────┘
               ▼
阶段 4：单序列 3/32
               ▼
阶段 5：unified/shared/hybrid
               ▼
阶段 6：server Tri-first policy
               ▼
阶段 7：state/RAM/checkpoint
               ▼
阶段 8：KV + recurrent joint auto-fit
               ▼
阶段 9：CUDA、多 GPU、正式性能门
```

前三个阶段是基础设施，阶段 4 到阶段 7形成完整的 runtime 语义，阶段 8让 auto KV/recurrent 真正利用 10.67x virtual envelope，阶段 9完成生产性能。最重要的发布阻断条件是：**没有可靠 compaction，就不能把 aggressive TriAttention 接到 server pressure planner。**



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)