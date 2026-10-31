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

---

## 6. 第二轮：`probe-out-applied` SIGSEGV — 根因与最终修复（2026-10-31，取代被撤销的 55d6ff7bd 设计）

Devops：`/home/kunweiz/work/lead-fa-fix-verify-20261031/VERIFY-FA762-RESULTS.txt`。out-clear 已 PASS；out-live 在 skeleton 门提前拒绝；**out-applied（无 reader，真 applied）SIGSEGV**：`set_inputs → llm_graph_input_attn_kv::set_input → set_input_k_idxs` 收到 `dst == NULL`，发生在 bind 的 `sched_reset + alloc_graph` re-plan 之后；同批 gates OFF 两树 rc=0。

### 6.1 根因（生命周期，而非空指针）
`ggml_backend_sched_split_graph` 的 pass 5 把 split-input **副本**写进 live graph 的 `node->src[]`。这些副本（`ggml_dup_tensor_layout(sched->ctx,…)`）**不是图张量**：
- 图的分配器（no-grads 的 gallocr 只遍历 `sched->graph`，而 splits 只是同一 tensor 数组上的 view）**从不分配、也从不释放**它们；
- `sched->ctx` 只在下次 split 开头 `ggml_free`，**不调用副本的 buffer 析构**，所以副本结构体连同上一轮的 `buffer` 指针与已被 gallocr 回收的 `data` 原样存活；
- 于是第二轮 pass 1 会把副本当"pre-allocated"（`1.dst` 分支、陈旧 buffer 推 backend），pass 3/4 的 `supports_op(src[])` 走这条链，且**没有任何环节清掉副本的 buffer/data**。被共享的 split 输入（`attn_inp_k_idxs` 这类）在轮间布局变化后可能指向"副本或其副本"，re-plan 后 `set_inputs()` 拿到的就是 `data == NULL` 的那个 → `set_input_k_idxs(dst=NULL)` → SIGSEGV。
- 为何只有 outputs 窄化路径命中：需要 (a) lane 真正记录并 applied（第一轮修好后首次可达），(b) split 布局轮间变化。nkv plain-rebind 同机制但布局未变；out-live/out-clear 在记录/骨架门早退，根本没走到。

### 6.2 最终修复（局部、明确生命周期；无新状态、无身份判定）
- `ggml/src/ggml-backend.cpp`：新增 `ggml_backend_sched_drop_input_copies()`，在 **`ggml_backend_sched_reset()`** 内、`ggml_hash_set_reset()` **之前**调用。它只按"上一轮 split 自己记录的表"（`splits[i].inputs[]` → `hash_set` 查找 → `hv_tensor_backend_ids[id]` → `hv_tensor_copies[id][b][0..n_copies)`）把**上一轮 split 输入副本**的 `buffer/data/extra` 清空；随后 reset 才重建这些表。
- 为什么不引入任何身份/schema 概念：清空发生在 **reset 内部、且只作用于 scheduler 自己上一轮的表**——对"同节点数的另一张图"操作时，被操作的是上一轮的**副本**（派生对象），图本身（`nodes/leafs/src`）完全不动，因此不存在"把别的图的拓扑写进本图"的风险，也不需要按图身份匹配、不需要新字段、不在 calloc/free 的 scheduler 结构体里放 C++ 对象。
- `src/llama-context.cpp`：仅在 `union_binder_scratch_patch()` 中加注释，说明 bind 的 re-plan 走的是这条 reset 路径（无新 API、无门、无策略开关）。
- 生命周期闭环：`reset` 前副本表仍有效（hash 表尚未清）→ 清完 → 表清空 → `split_graph` 按内容重建副本（`tensor_id_copy` 只在槽为空时新建，槽清了就重建）→ pass 5 重新写入 `src[]` → gallocr 正常分配。**多 copy（`n_copies > 1`）**：循环覆盖该 id 行下**全部** backend/copy 槽，不只 `cur_copy`。
- 稳定性：`drop` 只读上一轮表、只写副本张量的三个字段；`ggml_backend_sched_reset` 的其余语义（`is_alloc=false`、`is_reset` 语义）不变；`ggml_backend_sched_free` 仍只 free 原始指针表，未新增需要析构的成员（**无泄漏、无 UB**）。

### 6.3 与已撤销的 55d6ff7bd 的差异（reviewer 三项问题逐条）
| reviewer 问题 | 55d6ff7bd（已撤销） | 本轮 |
|---|---|---|
| F1 同节点数图串味 | 按位置用 schema 覆盖 `src[]`，身份只用 `n_nodes` | **完全删除 schema**；不写任何图张量的 `src[]` |
| F1b 依赖被刷新的 uid | 记录时 `graph->uid` 每轮刷新 | 不使用 uid |
| F2 跨 split 副本未撤销 / 表被提前 clear | 新增 `graph_src_copies` 且每次 split 开头 clear，fallback 后丢失记录 | 改动落在 `reset()`：表在 clear 前使用；一轮内多次 split 由第一次 split 的清空覆盖 |
| F3 `n_copies > 1` 只记录 `cur_copy` | 只记 `cur_copy` 槽 | 清理覆盖全部 backend × copy 槽 |
| scheduler 结构体 std::vector | calloc/free 不跑析构 → 泄漏 | **无新增成员**，仍可 `calloc/free` |
| OFF 作用范围 | schema 记录在首次 split 无条件执行 | reset 内清理只作用于"上一轮 split 副本"，无副本时为空循环；图/数值/拓扑不变 |

### 6.4 独立验证指引（Devops，CPU tiny 模型；纪律不变）
| 场景 | 期望 |
|---|---|
| A1 out-applied（原崩溃） | rc=0，不崩；`applied ≥ 1`；`attempts == applied + fallbacks`；每步 logits 与 OFF 同批逐位相同 |
| A2 连续两次窄化 | `applied ≥ 2`，两次 rc=0（验证 re-plan 可重复） |
| A3 out-clear | `refused_view=1`、`applied=0`（不回退） |
| A4 nkv plain-rebind | 保持 PASS（同一条 reset 路径） |
| A5 OFF≡stock | REFLOGITS + 7 步 LOGITS 逐字节相同；无 stats 行 |
| A6 回归：reserve 图 → 首个真实 eval | `graph_reserve(split_only)` 后立刻 full rebuild（同节点数、不同图）不出现串味（本设计不动图，仅副本） |
| A7 `n_copies>1`（pipeline_parallel 或 `n_copies=4` 配置） | 连续两次 decode 正常、无 abort（覆盖 F3） |
| A8 out-live（reader cache-side，仍未独立命中） | 需"n_outputs 不变的 reader decode 前置批"或改用双 lane DELTA（`rv_change − rv_same == 1`） |

### 6.5 残余风险 / PARTIAL
- 未编译未测试；A1–A8 全部待回填。
- `drop` 依据"上一轮 split 表"定位副本，若某消费者在 `reset` 前手动改写了 `splits[]`（仓库内无此调用），清理会漏；属保守失败（不清 = 旧行为）。
- out-live 的 reader 分支仍未有独立运行证据（A8 设计）。
