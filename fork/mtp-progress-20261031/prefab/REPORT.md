# REPORT — 080352 单一基线冻结、谱系差异与点名缺陷复核

## 1. 结论（先行）

1. **单一基线：080352 谱系。** 源码树 `/home/kunweiz/work/prefab-share-live-r8-080352/wt`
   （HEAD `f1c5fafc0449b22969ddabfcc9c361940d94d033`，stock `f1c5faf` 上的 16 文件补丁工作树）。
2. **本次新增修复：0 行 / 0 个 hunk。** 经逐点复核，用户点名的两个缺陷
   （IMROPE pos2d 守卫、MTP KV first_layer_id）**已经由 080352 基线 patch 修复**；
   没有发现仍存在的、需要新增代码的已证实缺陷。按用户要求，**不把基线已有修复冒充为本次新增**。
3. **085000 谱系不采用、不移植。** 其 REPORT 自述无任何成功构建（build-slot 3 次失败/退出 75），
   运行结论全部来自模拟；其 patch 仍保留两处点名缺陷（见 §3/§4）。
4. **冻结交付：** 本目录 `patch.diff`，173,262 字节 / 16 文件，SHA-256
   `ddf7e97a9f26f52582f0991a8b78327e26d9e95b199dc98291ac2e418aef0aee`；
   在干净 stock `f1c5faf` 上 `git apply --cached --check` 退出 0，应用后 tree
   `84f67dcf012fe3c8c84e2806df63b6fec91b0a38`（与基线 worktree 逐文件一致，见 VERIFY.txt）。
5. **验证及溯源：** 本次禁令前已运行现成 CPU stock/OFF/ON 三臂，结果见 §6。
   库时间戳晚于所列源码、库符号及运行时计数仅是辅助证据；没有可复现构建溯源，
   **无法证明现成二进制与冻结源码精确对应**，不能仅凭时间戳断言包含全部修复。
   新禁令生效后不再执行任何编译、测试或语法检查。

## 2. 两条谱系与差异

两条谱系同源于此前的 prefab-share / ab-compose 工作，但代码组织、验证状态与缺陷状态不同。

| 维度 | **080352（选定基线）** | 085000（仅对照） |
|---|---|---|
| 目录 | `prefab-share-live-r8-080352` | `prefab-share-live-r8-085000` |
| 源码树 | `wt/`，HEAD `f1c5faf`（干净 stock 上重建） | `wt/`，接入基线为 stock + `prefab-share-nodes-r8` 累计补丁（`git write-tree`=`ca001036…`） |
| 补丁 | `patch.diff` 173,262 B / 16 文件 / +3311 −38；SHA `ddf7e97a…0aee`；应用树 `84f67dc` | `patch.diff` 185,952 B / 19 文件 / +3453 −38；SHA `4c9957ea…5111`；冻结树 `bb78f46f…7616` |
| 核心结构 | `llama-prefab-hotpath.h`（ggml-free 热路径 seam）+ `llama-prefab-mtp.cpp`（A×B compose）+ `llama-prefab.cpp` + `llama-prefab-splice.h` | `llama-prefab-share.h`（共享决策核心）+ `llama-prefab-share.h` 命名体系；另改 `include/llama.h`、`common/sampling.cpp`、`tools/server/server-context.cpp` 做 perf 可观测 |
| qwen4exp | `graph_mtp` 拆为 `build_mtp_prologue`（A）与 `build_mtp_block`（B），compose 只重发 B | 同类拆分，但 no-clone/share 路线不同 |
| 构建/运行证据 | **真实 CPU-only 构建成功**；`harness/live_multi_request` 链接真实 libllama，`llama_decode` 实走 `graph_build → prefab_process_ubatch → llama_prefab_mtp_compose`；五臂逐字节对照（55/264 decodes，11,264 B / 87,552 B 全部相同） | **无任何成功构建/目标文件/二进制**；只有模拟 harness；REPORT 明示所有结论“未验证” |
| 点名缺陷 1：IMROPE pos2d | **已修复**（两处 MTP 入口） | **仍存在**：patch 行 539、636 `!ubatch.is_pos_2d()` |
| 点名缺陷 2：MTP KV layer 0 | **已修复**：`first_layer_id()` | **仍存在**：patch 行 2961 `kv->get_k_storage(0)`；无 `first_layer_id` |
| 结论 | **选定为唯一基线** | 不作基线；不移植其 share/perf/server 等无关补丁 |

085000 的 REPORT（§“与前序的关系”）自述只读查看过 080352 的 `wt` 并沿用了其计数口径，
未修改该目录。因此两条谱系不是“同一补丁的两个副本”，不能混用。

## 3. 点名缺陷：根因、修复点与正确性

### 3.1 IMROPE pos2d 守卫（qwen4exp MTP 永久 fallback）

**缺陷（基线已修，属基线已有修复）**

- 位置：`src/llama-context.cpp` 的 `llama_context::prefab_process_ubatch`
  - decouple 路径：worktree 行 6004（patch 行 483）
  - whole-graph MTP 路径：worktree 行 6104（patch 行 583）
- 原代码：`!ubatch.is_pos_2d()` 作为 MTP lane 的运行时合同。
  qwen4exp 是 IMROPE，`llama_hparams::n_pos_per_embd()` 对 `MROPE/IMROPE` 返回 4；
  当 `ubatch.n_pos >= 3` 时 `llama_ubatch::is_pos_2d()` 为真（`src/llama-batch.h:25`），
  于是 MTP lane 对**每一个** draft ubatch 都判 `ok/ok_m=0`，全部回退 stock。
- 证据（080352 `VERIFY.txt`，基线记录，非本工人重跑）：
  `DIAG decline-B ok_m=0 kvc=1 token=1 pos2d=1 views=0 kvstream=0 loras=0 samplers=0 n_tokens=4 slots=3 envelopes=3 is_m=-1`；
  修复前 ON 臂 55/55 ubatch 全部 `n_fallback_runtime`。
- 修复：两处改为 `!ubatch.is_pos_2d() || st.mtp_mode`（decouple 路径 `pos_ok`、MTP 路径 `pos_ok_m`）。

**为什么正确**

- MTP 头使用 stock `graph_mtp` 自己的 `build_inp_pos()`（`src/models/qwen4exp.cpp:765` 的
  `build_mtp_block` 内）；`n_pos_per_embd()==4` 时 `llm_graph_input_pos::set_input`
  对带 token 的文本 draft 把 1-D `pos` 展开为 4 段（`src/llama-graph.cpp:161-180`），
  与 stock 路径完全一致。
- KV 写入路径本来就支持 2-D：`llama_kv_cache::apply_ubatch` 在 `ubatch.is_pos_2d()` 时写
  `llama_kv_cell_ext{x,y}`（`src/llama-kv-cache.cpp:1623`）；mask writer 也有 `is_2d` 模板分支
  （`src/llama-kv-cache.cpp:2623`）。
- **factor 路径仍保留 1-D 限制**（worktree 行 6244）；它使用 `llm_graph_input_prefab_pos`
  只写 `n_env` 个 1-D 位置，因此该限制是必需的，不是本次修复对象。
- `st.mtp_mode` 在入口处只允许 MTP 上下文；两条分支中非 `DECODER_MTP` 的 gtype 均提前回退，
  故此放行不会让 2-D pos 进入 factor 图。

### 3.2 MTP KV `first_layer_id`（过滤型 KV 取 global layer 0）

**缺陷（基线已修，属基线已有修复）**

- 位置：`src/llama-prefab.cpp::prefab_make_key`（worktree 行 735；patch 行 2797）
- 原代码：`kv->get_k_storage(0)` / `kv->get_v_storage(0)`。
- 根因：MTP 上下文拥有的是**过滤型** KV cache，只映射 nextn 层（`il >= n_layer`），
  `map_layer_ids` 中**没有 global layer 0**；`get_k_storage()` 走 `map_layer_ids.at(il)`
  （`src/llama-kv-cache.cpp:1885`），于是 `unordered_map::at` 抛 `std::out_of_range`。
- 证据（080352 `VERIFY.txt`，gdb 回溯）：
  `#8 __throw_out_of_range → #11 llama_kv_cache::get_k_storage(int) → #12 prefab_make_key →
  #13 llama_prefab_key_for → #14 llama_context::decode`；`what(): unordered_map::at`。
- 修复：新增访问器
  `int32_t llama_kv_cache::first_layer_id() const`
  （声明 `src/llama-kv-cache.h:199`；定义 `src/llama-kv-cache.cpp:1881`），
  `prefab_make_key` 改为在该层上取 storage 类型（worktree 行 750-752）。

**正确性边界**

- 访问器实现：`return map_layer_ids.empty() ? 0 : map_layer_ids.begin()->first;`
  `map_layer_ids` 是 `std::unordered_map<int32_t,int32_t>`（`src/llama-kv-cache.h:432`），
  故 `begin()` 给出的是**任一实际映射层**，不是“最小层号”；名字语义应理解为
  “a layer id present in this cache”。
- 该用法只取 **K/V storage type**，而一个 `llama_kv_cache` 的 `type_k`/`type_v` 是 per-cache
  单一值（构造函数只有一对 `ggml_type type_k, type_v`，见 `src/llama-kv-cache.h:27`，
  访问器 `type_k()/type_v()` 行 519-520），所以任取一个映射层得到的类型与 cache 全层一致，
  topology key 语义不变。
- 空映射时返回 0 只是兜底；本基线两条 lane 的 cache 都非空：
  MTP 上下文固定映射 nextn 层（`n_layer_nextn == 1` 的静态断言，`src/models/qwen4exp.cpp:671`），
  factor lane 映射整个 trunk。
- 修复只影响拓扑 key 的 `type_k/type_v` 两项；`kv_size/v_trans/causal/flash_attn` 等含义不变。

### 3.3 同轮第三处基线修复（为完整性列出，非本次新增）

MTP compose 的 KV 输入张量此前从未创建：`llm_graph_input_prefab_attn_kv` 的
`self_k_idxs/self_v_idxs/self_kq_mask/self_kq_mask_cnv/self_k_rot/self_v_rot` 全为 null。
基线在 `src/llama-prefab-mtp.cpp`（worktree 行 371-390；patch 行 1877 起）按 stock
`build_attn_inp_kv_impl` 与 factor 路径同一构造补齐，并在 `build_mtp_block` 前
`g.res->add_input(...)` 注册，使 `set_input`/`can_reuse` 每 ubatch 生效。
此项同样是**080352 基线已有**，不是本次新增。

## 4. 本次新增修复：无

**判定：不需要新增代码。** 理由：

1. 两个点名缺陷在 080352 基线 patch 中均已存在真实修复（§3），且逐点复核未发现残留缺陷：
   - pos2d：MTP 的两个入口都已放行；factor 路径的 1-D 限制按设计保留。
   - first_layer_id：唯一命中点 `prefab_make_key` 已改；全树无其它 `get_k_storage(0)` /
     `get_v_storage(0)` 的 prefab 用法（`grep` 仅剩 dsv4 与已修复的 prefab）。
2. 085000 中与这两个缺陷相关的代码只证明其自身仍未修复（§2 表、§4 后列），
   不存在“085000 有而 080352 缺”的必要修复。
3. 085000 的净增（`llama-prefab-share.h`、`include/llama.h` perf 字段、
   `common/sampling.cpp`、`tools/server/server-context.cpp`）属于可观测性/结构重排，
   与本任务的两个缺陷无因果关系；移植即“无关补丁”，故不做。

**为免误读，明确区分：**

| 项 | 归属 | 本次是否新增 |
|---|---|---|
| IMROPE pos2d 放行（`pos_ok` / `pos_ok_m`） | 080352 基线已有 | 否 |
| `llama_kv_cache::first_layer_id()` + `prefab_make_key` 使用 | 080352 基线已有 | 否 |
| MTP compose 补 KV 输入张量并注册 | 080352 基线已有 | 否 |
| 其它任何代码改动 | — | **0 行** |

**禁止项核对（最终 patch）：** 新增行中
`GGML_TP5_RSZ_RT`=0、`HIP`=0、`CUDA`=0、`GPU`=0（逐词大小写敏感计数，见 VERIFY.txt）；
未引入 GPU/HIP 路径；未修改任何名为 `selected` 的开关/变量；
补丁未触碰 `include/llama.h`、`common/sampling.cpp`、`tools/server/server-context.cpp`。

## 5. 冻结交付与身份

| 项 | 值 |
|---|---|
| 最终 patch | 本目录 `patch.diff`（= `prefab-share-live-r8-080352/patch.diff` 原件） |
| 大小 / 文件数 | 173,262 字节 / 16 文件（+3311 −38） |
| SHA-256 | `ddf7e97a9f26f52582f0991a8b78327e26d9e95b199dc98291ac2e418aef0aee` |
| stock 基线 | `f1c5fafc0449b22969ddabfcc9c361940d94d033` |
| 隔离 apply-check | `git apply --cached --check` 退出 0；应用后 `git write-tree` = `84f67dcf012fe3c8c84e2806df63b6fec91b0a38` |
| 基线 worktree 树 | 临时 index `read-tree HEAD` + `add -A` 得 `84f67dcf012fe3c8c84e2806df63b6fec91b0a38`（与 patch 应用态**一致**） |
| 精确源码目录 | `/home/kunweiz/work/prefab-share-live-r8-080352/wt` |
| 现成真实 CPU 驱动 | `/home/kunweiz/work/prefab-share-live-r8-080352/harness/live_multi_request`（08:45；动态链接） |
| live 库 | `/home/kunweiz/work/prefab-share-live-r8-080352/build-live/bin/libllama.so.0.0.2`（08:56；晚于最后源码 08:55） |
| stock 对照库/驱动 | `build-stock/bin/libllama.so.0.0.2`、`harness/stock_multi_request` |
| mock 模型 | `qwen4exp-tiny-mtp.gguf`，9,856 B，sha256 `77aab02e…1c67`（`verify-mock-model.py` PASS） |

该 patch 是在干净 stock 上应用（不是从 085000 的 `ca001036` 累计树导出），
因此与 085000 的补丁不可互换、不可叠加。

## 6. 本次三臂实测与未来验证依赖

新禁令生效前使用现成 CPU 二进制，单一 MTP context 串行处理 8 请求，完成 55 次真实 `llama_decode`、88 输出行。三臂均退出 0；各 `stream.bin` 的 SHA-256 均为 `fc55bbbbf2b8a9aca93d438b286fbe754b04f80320343aef89f228e95bfdb4e9`。这是哈希一致，未执行逐字节 `cmp`。

| 指标 | stock | OFF | ON |
|---|---:|---:|---:|
| prefab / stock ubatch | 无计数 | 0 / 55 | 55 / 0 |
| stock builds / reuses | 无计数 | 10 / 45 | 0 / 0 |
| envelope allocs / reuses | 无计数 | 0 / 0 | 5 / 50 |
| 构建节点数 | 无计数 | 1320 | 714 |
| compose hits / A reuses / clones | 无计数 | 0 / 0 / 0 | 5 / 3 / 0 |
| runtime fallback / MTP declines | 无计数 | 0 / 0 | 0 / 0 |

三臂的请求 2 和 5 分别成功删除后半序列，`pos_max` 变为 15 和 13；每请求清理 memory。没有覆盖并发多序列、release/reserve 或长窗口。

实际执行命令及输出：`/home/kunweiz/work/lead-prefab-fix-20261030/cpu-080352/command.txt`；每臂日志与输出：`/home/kunweiz/work/lead-prefab-fix-20261030/cpu-080352/{stock,off,on}/{events.log,stdout.log,stderr.log,stream.bin}`。

未来本机验证依赖：源码 `/home/kunweiz/work/prefab-share-live-r8-080352/wt`，驱动源码和现成驱动 `harness/live_multi_request.cpp`、`harness/{stock,live}_multi_request`，stock/live 库 `build-{stock,live}/bin`，模型 `qwen4exp-tiny-mtp.gguf`，后几项均相对于 `/home/kunweiz/work/prefab-share-live-r8-080352`。此前实际三臂命令在上述 `command.txt`；历史五臂命令及依赖在该目录 `RUNBOOK.md`、`run-arms.sh`，后者会覆盖其 `arm-*` 目录。未来如获明确许可才可执行；新禁令下不得运行、构建或做语法检查。若未来许可编译，必须使用 `/home/kunweiz/.local/bin/build-slot`，遇 75/124 停止且不重试。

## 7. 边界与未验证项

- 本次实际三臂 decode 在新禁令之前执行，见 §6；历史五臂数值是 080352 基线记录，不是本次五臂复验。新禁令后没有运行编译、测试或语法检查。
- 基线证据的边界沿用其 REPORT：单一架构（qwen4exp）与极小 mock 权重（D=8、2 层、vocab=6）；
  单序列 `n_ctx=1024`；未覆盖跨 1024 窗口重建、adapter/cvec、release/reserve、
  多序列交错、真实模型数值/性能；CPU-only，无 GPU/HIP。
- `harness/cpu_multi_request.cpp` 仍是逻辑自检（文件头标注非验收），不得作为验收依据。
- 本次未修改 `/home/kunweiz/work/prefab-share-live-r8-080352/wt`、
  `/home/kunweiz/work/prefab-share-live-r8-085000/wt` 或
  `/home/kunweiz/src/atomic-llama-cpp-turboquant` 的任何工作树文件与 index
  （apply-check 全部使用临时 `GIT_INDEX_FILE`，用后删除）。
