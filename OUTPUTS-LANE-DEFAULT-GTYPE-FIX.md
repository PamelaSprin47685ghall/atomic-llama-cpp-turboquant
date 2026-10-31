# outputs lane（n_outputs union binding）DEFAULT 上下文死路径修复 — 根因、类型与安全边界、独立验证指引

日期：2026-10-31。树：`/home/kunweiz/work/merge-fa-20261031`（分支 `candidates/fa-truelen-20261031`）。
依据：Devops CPU 实测 `/home/kunweiz/work/lead-reader-isolated-20261031/VERIFY-RESULTS.md` + `out/`（`probe-out-live` / `probe-out-clear` 各一行 FAIL：attempts=3 fallbacks=3，其余计数全 0；gdb：`binder_record_snapshot` 0 命中）。
纪律：本刀只写码/静态审查，**未编译、未运行**；设备/CPU 证据由 Devops 渠道补。

## 0. 根因（已修复）

| 项 | 旧代码 | 事实 |
|---|---|---|
| outputs lane 记录门 | `env_out = … && gtype == LLM_GRAPH_TYPE_DECODER`（`src/llama-context.cpp:3690` 旧行号） | 本树**没有任何路径产生 `LLM_GRAPH_TYPE_DECODER`**：`ctx_type_to_graph_type()` 把 `LLAMA_CONTEXT_TYPE_DEFAULT → LLM_GRAPH_TYPE_DEFAULT`、`LLAMA_CONTEXT_TYPE_MTP → LLM_GRAPH_TYPE_DECODER_MTP`；编码走 `LLM_GRAPH_TYPE_ENCODER`（`llama_encode` 硬编码）。 |
| 后果 | `Env_out` 恒 false | `binder_record_snapshot` 从不被 outputs lane 触发 → `nout_record_valid=false` → 每次 `union_rebind_outputs` 在 `!st.nout_record_valid || set.tensors->empty()` 静默返回（只计 attempts/fallbacks，不进入任何分类拒绝点）。outputs-only 配置（`GGML_TP5_UNION_BINDER=1`、NKV 关）整条 lane 死。 |

## 1. 修复内容（源码，2 处 + 1 处 cov 会计加固）

1. `src/llama-context.cpp`：新增 `static bool llm_graph_type_is_decoder_build(llm_graph_type)`——全枚举覆盖，**DEFAULT + DECODER = 解码构建**（接受），**ENCODER + DECODER_MTP = 拒绝**（fail-closed，不记录快照）。
2. `env_out` 改用该分类器（`llm_graph_type_is_decoder_build(gtype)`），保留原注释并以新注释说明旧比较为何是死路径。
3. `binder_record_snapshot` 的 r8 覆盖分类遍历改为 `if (cov_enabled && nkv_built > 0)`：outputs lane 现在也记录快照（`nkv_built=0`），cov 家族分类必须以 n_kv 宽度为参照，width=0 的记录不做分类（`cov_present` 仍每记录清零），nkv 覆盖会计与 r8 完全一致。

修复后行为链（DEFAULT 上下文、`GGML_TP5_UNION_BINDER=1`、无 backend samplers）：
```
fresh build (DEFAULT) → env_out=true → binder_record_snapshot(gf, 0)
                      → nout_record_valid=true, nout_cap=n_outputs, nout_bound=n_outputs
后续 eval n_outputs 变化 → can_reuse 的 out_ids 键拒绝 → nout_pending
                      → union_rebind_outputs 进入真实判定（见 §2）
成功 → 就地改 executed shapes（[0, nout_cap]，shrink/restore）→ nout_bound=active, applied++
失败 → 计分类拒绝 → 调用方全量重建（exact active count）→ 重记录
```

## 2. 类型覆盖与安全边界（新可达路径逐守卫复核）

| # | 判定点（`union_rebind_outputs`） | 语义 | 修复后状态 |
|---|---|---|---|
| G0 | `res/gf == nullptr` | 无图 | fallback，计 c_fallbacks |
| G1 | `!nout_record_valid \|\| set.tensors->empty()` | 必须有本次 fresh build 的记录 | **现在能过**（DEFAULT 记录） |
| G2 | 图上无 `out_ids` gather 输入（pooling/embedding-only 图） | 无可对账 → 拒绝 | 不变，`refused_skeleton` |
| G3 | `backend_tp5_relay != nullptr \|\| cparams.pipeline_parallel` | relay/PP 契约外 | 不变，`refused_relay/pipeline` |
| G4 | 非 nkv 键不一致（`allow_reuse`，n_outputs 归一） | 只有 n_outputs 可以不同 | 不变，`refused_skeleton` |
| G5 | `n_outputs == nout_bound` | 已是该宽度：no-op | 不变，fallback（不误记拒绝） |
| G6 | `n_outputs > nout_cap` | 越建图宽度：无分配 backing | 不变，`refused_grow` |
| G7 | nkv 一致性（`nkvs>0 && nkv_enabled && record_valid` 且 `nkv_now != record_bound`） | 防“kv 侧在另一宽度上被就地改” | 不变；outputs-only 时 `record_valid=false` → 整段跳过（注释已说明：否则每轮误拒） |
| G8 | attention-view readers（cache 当前状态 + 图捕获输入状态 双检测） | 读者 span 是建图期捕获，不能就地改 | 不变；**现在真正可达**（Devops 的 probe-out-live/clear 的 FAIL 正是此前从未走到这里） |
| G9 | `union_binder_scratch_patch`：skeleton 逐点比对 + 每改变张量 byte extent 收缩检查 + 就地拷 ne/nb/op_params/view_offs | 唯一变更来源必须是 out_ids gather 闭包 | 不变（共享尾部，nkv 路径同款） |
| G10 | 申请/失败回滚：`--c_applied`、`nout_patched` 撤销、`++c_fallbacks` | 计数器恒等式 attempts == applied + fallbacks | 不变（同一代码路径） |

类型矩阵：

| gtype | 产生者 | outputs lane 记录？ | 理由 |
|---|---|---|---|
| `LLM_GRAPH_TYPE_DEFAULT` | `llama_decode` 全部构建（DEFAULT 上下文，prefill+decode） | **是（本修复）** | 本树唯一的普通解码图类型；out_ids gather 闭合、无 sampler 构建（无 sampler 时注册）成立 |
| `LLM_GRAPH_TYPE_DECODER` | 本树无生产者（保留显式标签兼容） | 是 | 模型侧把 DEFAULT/DECODER 视为同类（dflash/t5 的 `build_arch_graph`） |
| `LLM_GRAPH_TYPE_ENCODER` | `llama_encode()` | 否 | 编码图与 out_ids 解码闭包/宽度簿记无关；记录会污染共享 record |
| `LLM_GRAPH_TYPE_DECODER_MTP` | MTP 上下文 | 否 | nextn 图未在该 lane 的设计/验证范围；保持与 r7 逐位一致 |

OFF/等价性：
- 门关（`GGML_TP5_UNION_BINDER` 未设或非 "1"）：无状态槽、无快照、无 rebind，行为 stock；`llm_graph_type_is_decoder_build` 只在 `env_out` 表达式内被求值（短路于 `nout_enabled`）。
- 门开但图无 out_ids / relay / PP / samplers 已挂：行为与旧代码一致（G2/G3/注册期 sampler 守卫）。
- 修复只改变“是否记录/是否进入既有判定”：不新增 patch 形态、不改任何张量内容路径、不降精度、不碰 FA/`GGML_TP5_FA_NKV_PARAM` 复用禁用约定（`llama_ctx_fa_nkv_param_enabled()` 仍在 outputs 派发条件里）。

## 3. 独立 outputs live/clear 验证指引（Devops，CPU tiny 模型 + 真实 llama_decode）

前置（延续既有纪律）：`build-slot` 串行、180 s 上限、退出 75/124 即停；无 GPU、无 RSZ_RT；使用既有 `reader-driver-probe` 构建流程与 tiny 模型，仅新增/复用 scenarios。

### 3.1 观察量（`NKV_BINDER_STATS`，上下文析构时打印；outputs-only 时 nkv 字段为 0）
`attempts / applied / refused_grow / refused_skeleton / refused_relay / refused_pipeline / fallbacks / record_builds`；cov 门另打 `refused_view` 与 COV 行。
判读恒等式：`attempts == applied + fallbacks`；`record_builds` 只在 NKV 门开时增长（outputs-only 应恒 0）。

### 3.2 场景矩阵（outputs-only：`GGML_TP5_UNION_BINDER=1`，NKV 全关，`GGML_TP5_NKV_COV=1` 取计数）

| 场景 | 步骤 | 期望（修复后） | 失败含义 |
|---|---|---|---|
| **O1 no-reader live bind** | prefill n_tokens=N 且 output 全 true → 连续 decode 1 token（n_outputs 每轮变化） | `applied ≥ 1`（shrink 成功）、`attempts == applied + fallbacks`、**每步 LOGITS 与 stock 构建逐位相同**（shape-only patch） | applied=0 → 记录/派发仍死；LOGITS 差 → patch 形态错误 |
| **O2 restore（增长回建图宽）** | 承接 O1，把 n_outputs 提升回 nout_cap 以内（如 batch 中 output 数变多） | `applied ≥ 2`、`refused_grow == 0` | refused_grow>0 → cap 记账错误 |
| **O3 growth past cap** | 让 n_outputs > 上一建图宽度（不重建） | `refused_grow ≥ 1` 且随后全量重建成功、LOGITS 正确 | applied 误记 → fail-open 风险 |
| **O4 readers LIVE** | 与既有 `probe-out-live` 相同（reader decode，视图未清） | `refused_view ≥ 1`（每 reader 轮 +1），`applied == 0` | refused_view=0（本次 Devops 的 FAIL 形态）→ 未到达 reader 检测 |
| **O5 readers CLEARED** | 与既有 `probe-out-clear` 相同（图上视图捕获后清除） | `refused_view ≥ 1`（图侧检测器命中），`applied == 0` | 同上 |
| **O6 双 lane DELTA** | `UNION=1 + NKV=1`，同 reader decode 分别跑“n_outputs 不变”与“n_outputs 变化” | `rv_change − rv_same == 1`（每对）→ outputs lane 的独立贡献；`record_builds` 由 nkv 侧提供 | 差值 0 → outputs 侧仍未独立生效 |
| **O7 OFF ≡ stock** | `UNION` unset | 无 stats 行、退出 0、REFLOGITS 与 stock 构建逐字节相同 | 有 stats/日志差 → OFF 不 stock |

O1 的“逐位相同”依据：repatch 只改 `ne/nb/op_params/view_offs`，executed gather 行数变化不改任何数值路径；tiny 权重下逐位可比。

### 3.3 最小判定脚本骨架（Devops 展开；每步 build-slot 超时即停）
```bash
D=/home/kunweiz/work/lead-reader-isolated-20261031
S=/home/kunweiz/work/<new-run>/out
# O1: 同 driver，n_outputs 变化路径
GGML_TP5_UNION_BINDER=1 GGML_TP5_NKV_COV=1 build-slot "$D/reader-driver-probe" \
  --model nkv-driver-tiny.gguf --scenario out-live-bind | tee "$S/o1.txt"
grep -o 'attempts=[0-9]* applied=[0-9]*' "$S/o1.txt"   # applied >= 1
# O7: OFF 对照（同 driver, 无 UNION）
env -u GGML_TP5_UNION_BINDER build-slot "$D/reader-driver-probe" \
  --model nkv-driver-tiny.gguf --scenario off-stock | tee "$S/o7.txt"
cmp <(grep -E 'REFLOGITS|LOGITS' "$S/o1.txt" | sed 's/[[:space:]]*$//') \
    <(grep -E 'REFLOGITS|LOGITS' "$S/o7.txt" | sed 's/[[:space:]]*$//')
```
注：若希望 O1/O2 在既有 driver 中直接成立，需要 driver 增加“连续两轮 n_outputs 不同”的 out_ids 输入序列（现有 probe-out-* 是 reader 场景，对应 O4/O5）。driver 改动属测试件，建议 Devops 用同一 tiny 模型、`llama_batch` 中 `logits` 标志位切换即可，无需新权重。

## 4. 残余风险 / PARTIAL

1. 本修复**未编译**：`llm_graph_type_is_decoder_build` 的 switch 覆盖 4 个枚举值并带末尾 `return false`（-Wswitch 需编译器确认无新增告警；按现有风格无 default 分支，其余枚举处理相同）。**PARTIAL**。
2. 新可达路径（G1→G6、G8→G10）只有静态复核；设备/CPU 运行证据需 §3 回填。**PARTIAL**。
3. `LLM_GRAPH_TYPE_DEFAULT` 也覆盖 embedding-only 图：G2（无 out_ids）拒绝兜底；未新增记录侧守卫（记录本身对后续键控无害，骨架不符即拒）。**静态结论**。
4. `llm_graph_input_out_ids::can_reuse` 仍以建图期计数为键（stock 注释）：narrow 后同宽度复用会触发一次重建再复用（既有设计，未改）。
5. `binder_record_snapshot` 现在每次 fresh DEFAULT 构建都会记录（outputs lane 的开销：one skeleton+bytes walk），与 nkv lane 相同量级；未做性能证据。
6. 与 `GGML_TP5_FA_NKV_PARAM` 的互斥不变（`llama_ctx_fa_nkv_param_enabled()` 在 outputs 派发条件中）。

## 5. 改动清单（本刀）

| 文件 | 改动 |
|---|---|
| `src/llama-context.cpp` | 新增 `llm_graph_type_is_decoder_build()`（+42 行含注释）；`env_out` 使用分类器；`binder_record_snapshot` 的 cov 遍历加 `nkv_built > 0` 守卫 |
| `FA-TRUELEN-NCOLS2-AUDIT.md` / `tests/test-fa-true-tail-micro.cpp` | 前序 FA stride 修复与本轮无关，保持原样（未覆盖） |
| 本文件 | 根因/类型/安全边界/独立验证指引 |

---

## 6. 第二轮：`probe-out-applied` SIGSEGV 根因与修复（2026-10-31）

Devops 复验（`/home/kunweiz/work/lead-fa-fix-verify-20261031/VERIFY-FA762-RESULTS.txt`）：out-clear 已 PASS（`refused_view=1`）；out-live 在 skeleton 门提前拒绝（`refused_skeleton=2`，需 §3 O6 独立覆盖）；**out-applied（无 reader，真 applied 命中）SIGSEGV**——gdb：`process_ubatch → llm_graph_result::set_inputs → llm_graph_input_attn_kv::set_input → llama_kv_cache::set_input_k_idxs`，`dst == NULL`，发生在 bind 的 re-plan（`sched_reset + alloc_graph`）之后；同批 gates OFF 两树均 rc=0（14 checks）。

### 6.1 根因（已修）
1. **split 副本不是图的节点**：`null_grads` 的 allocator 不遍历 `sched->graph`（splits 的 `ggml_graph_view` 不扩展 live graph），因此 `ggml_backend_sched_split_graph` pass 5 装进 live graph `node->src[j]` 的 split-input 副本（`ggml_dup_tensor_layout(sched->ctx, …)`，op=NONE）**永远不会被 gallocr 分配**，也**从未被任何清理循环清掉** `buffer/data`：`sched->ctx` 只被 `ggml_free`（不调 buffer 析构），副本结构体内容（含上一轮 `buffer` 指针与已被 gallocr 回收的 `data`）原样保留到下一轮。
2. **pass 1 的 pre-allocated 分支**：下一轮 `split_graph` 从 live graph 的 `node->src[]` 看到这些副本，`ggml_backend_sched_backend_from_buffer` 依据其陈旧 buffer 给出 backend（`1.dst`），随后 pass 3/4 的 `ggml_backend_sched_set_if_supported/supports_op` 走 `src[]` 链，把副本当成输入叶/拷贝源；若该 buffer 类型与新后端不匹配则 `GGML_ABORT("pre-allocated tensor …")`，若匹配则副本的 `data` 仍是上一轮 gallocr 的地址。
3. **`src[5]` 冲突**：nkv 输入的 `k_idxs` 在 **splits 之间共享**时，pass 5 会把包含它的节点（`SET_ROWS` 等）的 `src[0]` 改写成该副本指针。而 `FLASH_ATTN_EXT` 的 `src[5]` 在 **nkv 通道**（`GGML_TP5_FA_CHECKED_TAIL/RTLEN`）下是 rtlen I32 张量、在 stock 下为空；另一侧的 `staged_dst.src[5]`（流式 staging）证明该类插槽在别的路径会承载副本。隐含的 `src[5]`/`src[0]` 槽位在 re-plan 中被复用后，`attn_inp_k_idxs` 变成「副本-或其副本」的引用，re-plan 后 `ggml_backend_sched_alloc_graph` 不保证其 `data`，`set_inputs()` 以 `dst=attn_inp_k_idxs`（此时 NULL）调用 `set_input_k_idxs` → SIGSEGV。
4. **为何只有 outputs 窄化路径炸**：它需要 (a) lane 真正记录并 applied（本轮修好后首次可达），(b) split 布局在轮间改变（prefill→decode 的 n_tokens 变化），使同一 src 的拷贝/副本归属改变。nkv lane 的 plain-rebind 用同一机制却不炸——差异正是分裂布局与 `src[5]` 的占用情况；历史 probe 的 `refused_skeleton=2/1` 解释了 out-live/out-clear 提前返回、从未走到这条代码。

### 6.2 修复（源码，未编译）
`ggml/src/ggml-backend.cpp`
- `struct ggml_backend_sched` 新增构建 schema：`schema_t`（首次 split 时按 id 记录的图张量表）、`schema_src[n_nodes*GGML_MAX_SRC]`（每节点每槽的 schema src id，-1=建图无 src）、`schema_n_src[n_nodes]`（最高 schema 槽+1），以及 `graph_src_copies[n_nodes][GGML_MAX_SRC]`（本次 split 为每节点每槽实际装入的副本，pass 5 写入）。
- `ggml_backend_sched_split_graph()` 开头调用新 `ggml_backend_sched_restore_graph()`：用 schema 还原每个节点全部 `src[]` 槽（含把 schema 之外的槽清空）——对未改动的图是恒等操作，对 re-plan 的图是精确还原，pass 5 随后按内容重新生成副本。
- `ggml_backend_sched_restore_graph` 对「节点数与 schema 不同」的图直接跳过（binder 本身也会按其记录的骨架拒绝）。
- 新增公共只读接口 `ggml_backend_sched_get_src_copy(sched, i_node, i_src)`（`ggml/include/ggml-backend.h`）。
- `schema_*` 在 scheduler 生命周期内保留（`ggml_backend_sched_reset` 不清），因此 binder 的 `sched_reset + alloc_graph` 仍能还原到建图 schema。

`src/llama-context.cpp`
- `union_binder_scratch_patch()` 在既有的「记录张量清 buffer/data/extra」循环之后，遍历 `ggml_backend_sched_get_src_copy()` 把**本次 split 装入 live graph 的全部副本**同样清为 `buffer/data/extra = NULL`（副本不在 `gf->n_nodes` 里，旧循环覆盖不到；这正是 Devops 的失败点）。副本的**指针/拓扑不改**，下一轮按内容命中原副本，保证工具调用语义（`sched->ctx` 里副本的 `src[]` 与 `view_src` 需在 re-plan 时仍有效）不被破坏。

### 6.3 安全边界（新改动）
- **只清 `buffer/data/extra`**：不改 `ne/nb/op_params/view_offs/view_src/src`，所以不会改变调度结果的数据布局；gallocr 会在下一轮重新分配。
- **不动 fixed 存储**：副本是拷贝张量，永远不是 weights/KV cache；清理循环只作用于 `graph_src_copies` 里记录的副本。
- **schema 还原只作用于本 scheduler 规划过的图**（`schema_src` 为空即跳过）；其他用户（单元测试）零影响。
- **OFF 不变**：整段只在 `union_binder_scratch_patch`（binder 门内）与 `split_graph` 的还原分支（schema 为空时早返回）里执行；门关无 binder 调用，schema 记录只是首次 split 的一次性 O(n*GGML_MAX_SRC) 填充（无行为变化）。
- **计数恒等式不变**：失败仍在调用方回滚 `c_applied/nout_binds` 并 `++c_fallbacks`。

### 6.4 为什么不是「屏蔽场景/降低条件」
没有跳过 `set_input_k_idxs`、没有在 SIGSEGV 前加空指针守卫把 NULL 静默吞掉、没有把 `probe-out-applied` 从矩阵移除，也没有缩小 `n_outputs` 变化幅度；修的是 re-plan 生命周期本身（副本状态与图 schema 的一致性）。

### 6.5 独立 out-live/out-clear 覆盖指引（Devops，CPU，同上纪律）
| 场景 | 期望 |
|---|---|
| **A1 out-applied（原 SIGSEGV）** | rc=0（不许崩溃）；`applied ≥ 1`；`attempts == applied + fallbacks`；每步 logits 与 OFF 同批逐位相同 |
| **A2 out-applied 二轮** | 连续两次窄化（不同 n_outputs）→ `applied ≥ 2`，两次都 rc=0（验证副本清理可重复） |
| **A3 out-clear** | `refused_view=1`、`applied=0`（保持既有 PASS 不回退） |
| **A4 out-live（cache-side reader）** | 需补一个 n_outputs 不变的 reader decode 前置批次，使 prefill1 的 skeleton 与 view decode 一致；期望 `refused_view ≥ 1`。若仍先撞 skeleton，记录 `refused_skeleton` 差值并改用 **双 lane DELTA**（`rv_change − rv_same == 1`）作独立证据 |
| **A5 nkv plain-rebind** | 保持 PASS（同一 `union_binder_scratch_patch` 尾部，验证新清理不破坏 nkv lane） |
| **A6 OFF≡stock** | REFLOGITS + 7 步 LOGITS 仍逐字节相同 |

失败判读：A1 若仍崩 → 抓新的 gdb 栈确认是否仍在 `set_inputs`；A2 `applied<2` → 副本清理未生效（检查 `graph_src_copies` 是否被 `clear()` 提前清掉）；A4 仍 skeleton → 记录两次 `refused_skeleton` 差值。

### 6.6 残余风险 / PARTIAL
- 全部改动**未编译未测试**；`std::vector` 已由 `ggml-backend.cpp` 现有 include 提供。
- `ggml_backend_sched_restore_graph` 依赖「首次 split 的 hash 顺序 == 后续 split 的次序」这一事实（节点未增删时成立；节点数变化时整段跳过，binder 会因骨架不符回退全量重建）。
- `graph_src_copies` 中记录的副本指针在 `sched->ctx` 释放后不悬空使用：该表在每次 `split_graph` 开头 `clear()`，读它的 `union_binder_scratch_patch` 只在上一轮 split 之后、下一轮 split 之前调用。
- out-live 的 reader 分支仍未由现有 probe 独立命中（见 A4），本轮不声称已证。
