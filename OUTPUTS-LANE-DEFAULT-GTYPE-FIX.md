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

### 6.2 根因：patch 成功后跳过了输入的 can_reuse（memo 未刷新）
`268f1b059` 的 probe 安全网已撤销（源码回到 `762631033`）。本轮改用控制流定位，结论：

**控制流事实**
- `llm_graph_input_attn_kv::set_input()` 用的是 `members[mctx]`；该成员**只**在 `can_reuse()` 里被赋值（`llama-graph.cpp:647`）。
- `process_ubatch()` 只有在 `res->can_reuse(gparams)` 那**一次**调用（`:3544`）时才会刷新 memo。第三轮 decode 的 `reused=false`，因此`can_reuse()` 根本没跑；随后的 outputs patch 成功把 `reused` 置 `true`（`:3641`），**memo 仍是上一轮的上下文**（已被销毁，`llama_memory_context_ptr` 仅存活一次 decode）。`set_inputs()` 随即被调用 ⇒ 调进已销毁对象 ⇒ Devops 观测的 `this == 0`。

**修复（不做身份地址比较，不新增 init_batch）**
1. `llm_graph_input_i` 新增 `virtual void revalidate(const llm_graph_params &)`（默认无操作）。持 `mctx` 的类覆盖它，只做"用本轮 params 刷新 memo"：`attn_kv`、`attn_k`、`attn_kv_msa`、`attn_kv_iswa`、`attn_k_iswa`、`rs`。
2. `llm_graph_input_out_ids`：新增 `n_outputs_exec`（执行宽度），`can_reuse()` 用**执行宽度**而非建图宽度比较；`revalidate()` 把执行宽度置为 `params.n_outputs`。否则 patch 成功后 `can_reuse` 仍按旧宽度拒绝，patch 永远只能退化成重建（这正是 reviewer 指出的死结）。
3. `llm_graph_result::revalidate_after_patch(params)`：① 逐输入 `revalidate`（刷新 memo）→ ② `params.allow_reuse` → ③ 逐输入 `can_reuse`（**原有的复用/形状合同**）→ ④ 校验 ubatch 分区未被 patch 改动。任一失败返回 false。
4. binder 两条成功路径都由它把关：
   - `n_outputs` 成功：`revalidate_after_patch` 通过才 `reused = true`，否则 `nout_patched=false, reused=false`（走下面的全量重建）；
   - `n_kv` 成功：同样校验，失败则回滚 `c_applied/++c_fallbacks` 并 `reused=false`。

**为什么这是"用现有合同"，不是先用旧参数拒绝 nout**
`can_reuse` 的 n_outputs 键现在比较**执行宽度**；patch 已经把 `out_ids` 改成新宽度，所以"刚 patch 成功"的图不会被自己拒绝。mask 键比较 `kq_mask->ne[0]` 与缓存**当前** `get_n_kv()`，patch 后两者一致。因此校验发生在 patch **之后**、针对**本轮** params，而不是 patch 之前用旧建模参数。

**副作用与回滚（明确）**
- `revalidate()` 有副作用（写 memo/执行宽度），但**所有写入都是本轮正确的值**，失败回退时不会污染：失败路径随后 `res->reset()` + 全量重建，输入对象连同 memo 一起重建。
- `revalidate_after_patch` 在 nkv 失败路径上先 `++c_fallbacks` 再回滚计数，保持 `attempts == applied + fallbacks`。
- 无新增内存/上下文；`revalidate` 不分配、不切分 `balloc`。
- OFF：不进入 binder，`revalidate` 不被调用；`n_outputs_exec` 初始为 0 时键回落到 `n_outputs`，行为与 stock 等价。

**第二轮 review（be99ce508）核实与修正**
| 项 | 核实 | 修正 |
|---|---|---|
| 头文件内联 revalidate 访问前置声明类型（Devops 构建失败） | 成立 | 全部改为类外定义 |
| `allow_reuse` 用**构建** n_outputs 比较 → 变窄 patch 必自拒 | 成立 | 新增 `n_outputs_topology` 参数（默认＝原行为），由 `revalidate_after_patch` 传入**patch 前**快照的 out_ids 绑定宽度 |
| `n_outputs_exec==0` 哨兵 | 成立 | 独立 `bool n_outputs_exec_set` |
| nout 失败缺计数/`nout_bound` 回滚 | 成立 | 补 `--c_applied/--nout_binds/++c_fallbacks` + `nout_bound_prev` 回退 |
| mctx 覆盖不全 | 成立 | 新增 hybrid×3（含子输入）、dsv4、pos_bucket_kv、attn_k_dsa |
| **Devops 实测的编译错误** `llama-context.cpp:3667` 写 `const` peek 指针 | 成立（`union_binder_state_peek()` 返回 `const llama_nkv_ctx_state *`） | 改走可变引用 `llama_nkv_ctx_state_get(this)` |
| reviewer 预测的 `rs` 缺声明 | **实测未报错**；但仍补上声明（类外定义必须有声明，避免编译器差异与 hybrid 前向调用落到基类空实现） | 已补 |
| `n_outputs_topology` 在 revalidate 之后取值 → 自比较恒真 | 成立 | 改为**先快照**（`revalidate` 之前） |
| nkv 成功 + nout revalidate 失败，nkv 的 `c_applied` 不回滚 | 成立 | 失败块内若 `nkv_patched` 同样 `--c_applied/++c_fallbacks` 并清标志（与 alloc 失败路径同语义） |
| `mem_hybrid::can_reuse` 强转错误（既有 bug，新路径可达） | 成立 | 改为 `llama_memory_hybrid_context`，复现侧访问经 `get_recr()` |
| OFF 等值 / 并发 | 无问题 | — |

**第三轮（Devops 1e5a 实测 + reviewer）**
| 项 | 核实 | 修复 |
|---|---|---|
| `llama-context.cpp:3667` 写 const peek 指针 | 实测唯一报错 | 改走 `llama_nkv_ctx_state_get(this)` 可变引用 |
| `llama-graph.cpp:1339` `mem_hybrid::can_reuse` 重复定义 | 实测报错，且新实现被粘在旧函数前 | 删除旧定义；每类恰好一份（正确类型版） |
| `522-532` 的 `s_copy` 等应为 `inp_rs->` | 实测报错，属同一段粘连代码（`llm_graph_input_rs` 无这些成员） | 随重复定义一并删除；现所有 `s_copy*`/`head`/`rs_z`/`replay_len` 访问均带正确前缀 |
| 取上轮 out_ids 宽(2) 与请求(1) 比较仍自拒 | 成立 | 改为**显式证明**：`revalidate_after_patch(params, kind, bound_width)`；被 patch 的那一维按"已证明的绑定宽度"判定，未 patch 的维度仍按图真实绑定宽度判定（不跳过任何检查） |
| hybrid 父调用 `inp_rs->revalidate(params)` 把 hybrid 强转 recurrent（UB） | 成立 | 传 `recr_params.mctx = mctx->get_recr()`（hybrid 与 hybrid_iswa 都有 `get_recr()`） |
| alloc 失败路径 `nout_bound` 未回滚 | 成立 | 失败块补 `nkv_st.nout_bound = nkv_st.nout_bound_prev` |

**第四轮（8645a60d4 复核）**
| 项 | 核实 | 修复 |
|---|---|---|
| `llm_graph_input_rs::can_reuse` 被误删（头文件仍声明 → 链接失败） | **成立**：删重复定义时把 rs 的 stock 键一起删了 | 从 `762631033` **逐字恢复** stock 实现 |
| 头文件声明 vs 定义 | 复核全部输入类的 `set_input`/`can_reuse`/`revalidate`，唯余基类纯虚声明（正确） | 无缺口 |
| 双 lane 联合收窄风险 | 成立：n_kv lane 的 scratch 建在图**当前输出绑定宽度**、只拷 n_kv 耦合形状，联合变化不属其能力范围 | 入口前置判定：`nout_enabled && nout_bound != n_outputs` 时**不进** n_kv lane（＝stock 的 cache miss → 全量重建）；不靠"patch 后再撤销" |
| revalidate 失败必须 `reused=false` 且全量重建 | 已保证：两个失败块都 `reused=false`，随后 `res->reset()` + `build_graph`；计数与 `nout_bound` 一并回滚 | — |
| 不得跳过必要校验 | 保持：被 patch 维度按**已证明的绑定宽度**判定（patch 的骨架+字节检查即证明），其余维度仍逐条跑原 `can_reuse`/`allow_reuse`/分区键 | — |

**绑定的真实性**：：：`revalidate_after_patch` 的拓扑参照取"patch 之前 out_ids 实际绑定的宽度"；n_outputs lane 的绑定值来自 `union_rebind_outputs` 写入的 `nout_bound`（真实绑定），不是本轮请求宽度；比对对象 `params.n_outputs` 同样是真实请求宽度。


| 项 | 核实 | 修正 |
|---|---|---|
| `this->params.allow_reuse` 用**构建** n_outputs 比较，变窄 patch 必自拒（`llama-graph.h:898`） | 成立 | `allow_reuse(other, n_outputs_topology = numeric_limits::max())`；`revalidate_after_patch` 传入 **out_ids 当前的执行宽度**（`get_n_outputs_for_topology()`），其它调用点默认参数＝stock 行为 |
| `n_outputs_exec == 0` 哨兵 | 成立（0 是合法 bound，`union_rebind_outputs` 注释明示） | 改为独立 `bool n_outputs_exec_set` |
| nout lane revalidate 失败缺计数/`nout_bound` 回滚 | 成立 | 失败分支补 `--c_applied`、`--nout_binds`、`++c_fallbacks`，并用新增的 `nout_bound_prev`（成功写入前记录）回滚 `nout_bound` |
| mctx 覆盖不全 | 成立（且 reviewer 指出 hybrid 的 `set_input` 走 `get_attn()->set_input_k_idxs`，`attn_kv` 的 revalidate 对其无效） | 新增 `mem_hybrid` / `mem_hybrid_k` / `mem_hybrid_iswa`（同时刷新子输入 `inp_attn`/`inp_rs`）、`dsv4`（含 `inp_raw->mctx`）、`pos_bucket_kv`、`attn_k_dsa` |
| 头文件内联 revalidate 访问前置声明类型（Devops 构建失败 6 处） | 成立 | **全部改为类外定义**（声明在头，定义在 `llama-graph.cpp`，此时 `llm_graph_params` 已完整） |
| OFF 等值 | 成立 | 无 binder ⇒ `revalidate` 不被调用；`n_outputs_exec_set=false` ⇒ out_ids 键即 stock |
| 并发 | 无新增（context 串行） | — |
| nkv lane 与 nout lane 同轮时 `gparams.n_outputs` 与 `nout_bound` 不一致 | 属实但属既有设计：nkv lane 的校验用本轮请求宽度，其 `allow_reuse` 走默认参数，行为与 stock 相同 | 不修改（避免两条 lane 语义互相渗透），记录为已知边界 |

**未覆盖/仍需证据**
- 第三轮 `reused=false` 的逐帧原因未验证；本修复不依赖它。
- `revalidate` 族不对 `params.mctx` 做 null 检查（调用点 `process_ubatch` 已在 `:3499` 解引用过，非空；encoder 图不构造 attn 输入）。若将来从别处调用需补。
- 全部改动未编译（Devops 上一轮构建失败已按内联→类外修正）；A1/A4/A5/A7 判据待回填。

### 6.5 残余风险 / PARTIAL
- 未编译未测试；A1–A11 全部无运行证据。
- **A9 的风险点**：如果 `init_batch(...)` 每次返回的上下文对象在"同一批 token 分区"下仍被判为不等价，`can_reuse` 会持续拒绝 → 每次都完整重建（正确但慢）。这需要 Devops 的 `n_reused` 证据，本轮不作乐观声明。
- 其它大模型族（hybrid/iswa/msa/dsa）的输入类仍在 `can_reuse` 里改写 `mctx` 指针，同一类风险未修（本 lane 只用 `llama_kv_cache_context`）；如后续在这些架构打开 binder，需要同样加身份判定。
- A6 rc=-1 未定位。

---

## 7. reader 检测器：内部消费者验证（测试件在树外，生产 API 零扩张）

### 7.1 公共 API 无法构造 cache-side（可证）
`src/llama-kv-cache.cpp`：`validate_attention_batch():1984-2010` 要求视图存在时解码行**逐行等于**视图 `(query_seq,query_pos)`；`prepare_attention_views():1938` 要求视图 query cell **尚不存在**；`apply_ubatch():1615-1624` 解码即写该 cell；`seq_rm():847` 清 cell 而 `resolve_attention_view():2024-2027` 依赖它们。⇒ 公共入口只剩"同位置重复解码"，而实现拒绝该形态。

### 7.2 验证方式：树外真实内部消费者（非 mock）
`/home/kunweiz/work/fa-outlive-probe-20261031/test-binder-reader-detectors.cpp`（链接 `libllama`），三用例各建真实 context/KV/graph result：
- **cache-side**：先在无视图下建图（图侧必盲）→ 再 `llama_attention_set_views` 武装缓存 → 经注册的 `n_outputs` 绑定调用 `rebind()` → 断言 `rebind==false`、`c_refused_view +1`、`c_applied` 不变；
- **graph-side**：武装 → 解码视图行（捕获 spans）→ `llama_attention_clear_views` → 同一组断言；
- **no-readers**：控制组，断言检测器**不**触发。
两个检测器通过**构造顺序**隔离，不需要任何合法/非法的解码技巧；观测点只用 binder 自身计数器与返回值。

### 7.3 生产代码零扩张
`llama_context::graph_params_for_test()` 属于**仅测试扩 API**，已撤销（`1eea533b3`）；`#ifdef` 保护下的 friend 方案同样未采用。测试自建 `llm_graph_params`（字段均为 struct 公有成员）+ 现有公开访问器（`get_cparams()`、`get_sched()`）+ 已有的内部绑定入口，因此**没有为测试向生产类添加访问器、friend 或行为**。

### 7.4 状态（PARTIAL）
- 测试**未编译未运行**；三用例全部未验。
- 不可构造的状态输出 `SKIP: <reason>` 并计入 skipped，**不计 PASS**。
- 需 `GGML_TP5_UNION_BINDER=1`（n_outputs 绑定仅门内注册）。
