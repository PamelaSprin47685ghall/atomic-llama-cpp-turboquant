# REPORT — lead-fa-boundary-20261030（共同基线核对 + 生产路径 CPU 边界验证设计）

日期：2026-10-30。工作区：`/home/kunweiz/work/lead-fa-boundary-20261030/`。
角色：本刀只做溯源、基线、验证代码与静态审查；**不修改生产代码**（生产改动由 `production-audit` 落盘）。
验收状态：**PARTIAL**（本会话纪律禁止任何编译/测试执行，见 §7）。

## 1. 交付物

| 件 | 路径 | 说明 |
|---|---|---|
| 最终累计补丁 | `./patch.diff` | 相对 stock `f1c5faf`，23 文件 +2563/−130，SHA-256 `7cff03f9f48173ba48c8c1273e4d4d1d57752054e6606a6083f19b7590c2d769` |
| 共享源码树 | `./source/` | stock `f1c5faf` clone + 全量工作树（基线 + production-audit 增量），未提交未推送 |
| CPU 验证代码 | `./verify/` | 4 个 harness + 1 个生产 TU 包装 + `run-all.sh`（**均未执行**） |
| 验证记录 | `./VERIFY.txt` | 本机实际执行的静态命令与逐字输出；禁令合规记录 |
| 真机手册 | `./RUNBOOK.md` | 可执行验证命令（未运行）与真机验收步骤 |
| 本报告 | `./REPORT.md` | 结论、证据分级、边界 |

## 2. 基线与溯源（全部本机实际执行，见 VERIFY.txt §1-§3）

- **stock**：`/home/kunweiz/src/atomic-llama-cpp-turboquant`，HEAD `f1c5faf`（`f1c5fafc0449b22969ddabfcc9c361940d94d033`），除未跟踪 `wt/` 外干净。
- **共同基线**：`6e8fb4bb4437e5d1fe53b488e79beea92f6c17d5deb34651a5cf257357943b22`（20 文件 +2492/−110），源提交 `6d1d582`（`nkv-fa-true-len-prove-r8-061203/source`）。
- **084457 = 061203 累计补丁**：两份 `patch.diff` SHA-256 逐位相同；本工作区 `source/` 从干净 stock 应用该补丁后重新生成的 diff 与之 `cmp` 字节一致；`diff -r --exclude=.git` 显示重现树与 `061203/source`、`084457/source` **完全无差异**（两棵树均 0 差异行）。
- **并行分叉不可直接叠加**（实测）：
  - 084451 `233ea017…`、082510 `939640fc…`、081601 `beea7310…`；三者施于干净 stock 均退出 0（各自是 stock-cumulative）。
  - 各自树相对基线树只差本刀增量文件：084451 6 文件、082510 2 文件、081601 3 文件（`diff -r -q` 实测）。
  - 六种两两叠加 `git apply --check` **全部 rc=1**，首个失败点 `ggml/include/ggml.h:2478`、`ggml/src/ggml-cpu/ops.cpp:8548`（基线 hunk 被重复应用）。结论：四份补丁共享同一基线，各自带不同增量，只能择一，不可叠加。
- **最终补丁（含 production-audit 增量）**：在干净 `f1c5faf` 上 `git apply --check` rc=0 → `git apply` rc=0 → 新文件 `git add -N` 后 `git diff` 与 `patch.diff` `cmp` 字节一致（同一 SHA）→ 工作树 `git apply -R --check` rc=0。
- 增量文件恰为 5 个：`src/llama-graph.h`、`src/llama-graph.cpp`、`src/models/qwen35.cpp`、`src/models/qwen4exp.cpp`、`src/models/deepseek4.cpp`。

## 3. 生产侧增量（production-audit，EDITSFINAL；本刀静态复核）

其报告的三处基线缺陷：

- **D1（qwen35 MTP chain）**：基线把 KQ mask 收窄到真长，而链式 MTP 图用 `build_attn_mha` 配全宽 `get_k/get_v` 且 op param 为 -1 → kernel 界 = `K->ne[1]`（stock 宽）> mask 宽 ⇒ mask 越界读。
- **D2（qwen4exp trunk/QSA）**：trunk mask 被收窄，但 `build_layer_attn`/`build_attn_qsa` 用全宽 K/V 与 cell 索引 `top_k`，mask 宽度还是语义键与 `set_rows` 目标 ⇒ 越界写 + 偏置静默改变。
- **D3（op param 与钳制不一致）**：mask/get_k/get_v 在 `(0, n_kv]` 之外回退 stock 宽度，而 `build_attn_mha` 曾发布原始请求值 ⇒ 越界请求变成 launcher ABORT 而非同样的回退。

修复形态（本刀逐 hunk 复核，行号为当前树）：
1. `llm_graph_input_attn_kv::fa_true_len`（默认 **false**，`llama-graph.h:355`）；`build_attn_inp_kv(bool=true)`、`build_inp_mem_hybrid(bool=true)`、`build_attn_inp_kv_impl(..., bool)`；三处收窄（mask、`self_nkv_req`、`get_k/get_v`）统一要求 `fa_true_len`，一图要么三者同收窄、要么全 stock。
2. `build_attn_mha`：slot 5 发布值钳制 `(0, k->ne[2]]` 否则 -1（`llama-graph.cpp:2845-2852` 一带），并在 FA 分支加 `GGML_ASSERT(kq_mask == nullptr || kq_mask->ne[0] == n_kv_built)`。
3. qwen35 `graph_mtp`：`build_attn_inp_kv(!mtp_chain)`；链式分支保持 stock 宽度（`mask_b` 仅在此分支），独立 MTP 路径（`build_attn`）保留真长。
4. qwen4exp trunk：`build_inp_mem_hybrid(mctx_hyb->get_idx() == nullptr)`（dense-only 才收窄）。
5. `src/models/deepseek4.cpp`：3 处 `build_attn_mha` 调用的 `nkv_eff` 形参补 `nullptr`。**属基线的签名迁移漏改**（基线改了签名未改该 TU），非新语义。

**签名迁移的独立证据（production-audit 提供，本刀复核可见）**：基线把 `build_attn_mha` 的第 6 形参改成必填
`ggml_tensor * nkv_eff`（`src/llama-graph.h`，无默认值），而 `deepseek4.cpp` 三处调用（791/846/882）与
`qwen35.cpp:646`、`qwen4exp.cpp:1201` 在冻结树里仍是 10 参 ⇒ 冻结树这两个 TU 会报 “too few arguments”。
修法是在 `kq_mask` 与 `sinks` 之间插入 `nullptr`；这些站点不拥有 RTLEN 输入，slot 5 发布 -1，语义不变。
当前实参计数：`llama-graph.cpp` 3027/3453/3654/3752/3811=11，3312=12（`nkv_eff=self_nkv_eff`、`runtime=self_nkv_req`），
3564=11；`deepseek4` 791/846/882、`qwen35:646`、`qwen4exp:1201` 均 11。

本刀独立静态核对（读代码，不运行）：

- 全树只有 `llama-graph.cpp:3091` 一处把 `n_kv_read` 传给 mask builder；其余 mask 构造（`3390` no-cache、`3837/3848` MLA/LID、`3871` MSA、`3897/3907/3928/3937` ISWA、`4132/4140` hybrid ISWA）**全部 stock 宽度** ⇒ 任何自定义 mask 消费者都拿不到收窄 mask。
- `build_attn` 的收窄条件为 `cparams.flash_attn && readers.empty() && inp->fa_true_len && 门`，与 mask builder 同一组条件；`nkv_read` 与 mask 用同一 `self_nkv_req`。
- 自定义 `build_attn_mha` 消费者（deepseek4 3 处、qwen35 链式、qwen4exp top_k）均传默认 `nkv_runtime = -1`，且其 mask 来自不收窄的构造器或 `fa_true_len=false` 的图 ⇒ 未发现残留的“全宽 K + 窄 mask”组合。
- `build_attn_inp_kv_msa`（minimax-m3）、`build_inp_mem_hybrid_iswa`（inkling 等）仍走 stock 宽度 ⇒ MSA/ISWA 子路径不受本 lane 影响（同时意味着这些路径**没有**真长收益）。

## 4. CPU 边界验证设计（覆盖矩阵）

目标不是“复刻表达式”，而是**执行生产 TU 的真实函数/模板**；4 个 harness 全部未执行（§7）。

| harness | 执行的生产代码 | 覆盖 |
|---|---|---|
| `verify/host_mask_boundary.cpp` | `#include "llama-kv-cache.cpp"` → `set_input_kq_mask_impl<T, causal, ...>` | 宽度 **1/255/256/257/511/512**；F16 与 F32；causal 与 non-causal；单 stream 2 token 与 **2 stream/3 sequence 多列**（含空洞、跨序列、同序列复排行）；每行 block 两端 canary；`WITH_NKV_TRUE_LEN` 下用 `seq_used_max_p1` 的真实宽度喂 fill，断言该宽度覆盖该序列每个活 cell |
| `verify/host_frontier_boundary.cpp` | `llama_kv_cells::used_max_p1`、`seq_used_max_p1`（r8 新代码） | 空 cells = 0；最高 cell 恰在 width-1 时 frontier = **1/255/256/257/511/512**；多序列/共享 cell/移除；缺席序列 = 0；4 种容量下确定性伪随机 **brute-force 交叉验证** |
| `verify/host_clamp_boundary.cpp` | `#include "llama-graph.cpp"` → 静态 `build_attn_inp_kq_mask`（+ 链接 `kv_context_tu.cpp` 取 `get_n_kv`） | **非法长度** `-2/-1/0/513/INT32_MAX/INT32_MIN` 一律回退 stock 宽；边界值 1/255/256/257/511/512 精确收窄；`kv_unified` 两态（1 stream/4 rows、2 stream/2 rows）；mask dtype（FA→F16、非 FA→F32）；name 与 4D 形状 |
| `verify/host_cpu_fa_off.cpp` | 公开 ggml API 的 CPU 后端跑 `GGML_OP_FLASH_ATTN_EXT`（真实 `ggml-cpu/ops.cpp`） | **OFF≡stock**：同一份源码分别对 stock 树与基线树编译，`A` 行位模式 dump 必须逐字节相同；slot 5 写入 -1/300 对 CPU 无效（应为 A）；`src[5]`（RTLEN lane）取 1/255/256/257/511 时与独立标量参考 1e-4 对齐，取 n_kv 时与 A 相同 |

“reader”覆盖方式（本 lane 在 reader 分支的语义是**保持 stock 宽度**，不是收窄）：

- 静态（本刀）：`build_attn` 在 `readers` 非空时走 `build_attn_reader`，该函数只构造 `mul_mat / soft_max / set_rows`，**不产生 FLASH_ATTN_EXT 节点**，因此既不写 op param 也不受 `nkv_read` 影响；`nkv_read` 与 `build_attn_mha` 的 `nkv_runtime` 均在 `readers.empty()` 条件下才非 -1。
- 可执行（未运行）：`verify/host_clamp_boundary.cpp` 的同源钳制覆盖到 mask 侧；reader 侧无收窄入口可执行，故不列为已验。

## 5. OFF ≡ stock（本刀不声称已证明）

**证据分级（明确区分）：**

- **A 级（已做，静态）**：
  1. 所有 9 个门（`GGML_TP5_FA_NKV_PARAM` / `FA_CHECKED_TAIL` / `FA_RTLEN` / `FA_TILE_FAST_FULL` / `FA_RUNTIME_NKV` / `UNION_BINDER` / `NKV_TRUEWIDTH` / `NKV_COV` / warn-only `LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME`）均为 `env != nullptr && strcmp(env,"1")==0` 的一次性静态求值；未设/任何其他值 = 该 lane stock。
  2. 门关时：mask 调用传 `-1`（`llama-graph.cpp:3091` 条件短路）、`self_nkv_req` 不计算、`nkv_read = -1`、slot 5 不写（`llama-graph.cpp:2849` 门内才调用 setter；`ggml_new_tensor` 的 `op_params` 零初始化）、`src[5]` 为空。
  3. 新增 assert 在 stock 调用点成立：所有调用方传入同宽 mask/K（本刀逐一核对 plain/no-cache/cross/MLA/ISWA/DSA/DSV4/QSA 调用点；peer 亦独立核对）。
  4. `GGML_TP5_RSZ_RT` 全树 0 处。
- **B 级（已写代码，未运行）**：`verify/host_cpu_fa_off.cpp` 的跨树位级对照、`host_mask_boundary` 的跨树 stdout 对照。
- **未证明**：真实 logits/位级端到端、GPU/HIP 路径、图复用与 multi-ubatch 常驻、binder 计数器输出。
**因此本报告不写“OFF≡stock 已证明”，只写“源码级论证 + 未执行的 CPU 位级对照设计”，验收 = PARTIAL。**

## 6. 边界与残余风险（如实）

1. 设备侧（CUDA/HIP）hunk 未做语法/设备验证（纪律禁止 GPU）；`fattn-*.cuh` 的声明契约仅源码审查。
2. `verify/host_clamp_boundary.cpp` 用 `#define private public` 测试访问垫片（不改生产文件、不改布局）；这是**测试专用**手段，若评审不接受需改为生产侧 friend/访问器（需改生产代码，非本刀权限）。
3. CPU FA harness 需要 12 个 ggml/CPU TU；若 build-slot 分片编译仍超时，只能停在部分 harness（RUNBOOK 已定“一 TU 一槽、75/124 即停”）。
4. RTLEN（`src[5]`）CPU 侧读取用 `min(value, k->ne[1])`，**无下界保护**：`<= 0` 会产生零长度 KV 区间（注意力退化），`> ne[1]` 被钳制。契约要求“永不超过建图宽度”，但代码不拒绝 0/负值；本刀只记录，不修改（非本刀生产代码）。
5. **mask 行 stride 一致性（未在设备验证）**：vec launcher 传的是 mask 自身的 `nb[1]/nb[2]/nb[3]`（`fattn-vec.cuh:890-891`），内核按 `nb31` 索引（`fattn-vec.cuh:169`），**不是**硬编码 `K->ne[1]`。因此风险不是“硬编码不匹配”，而是流式/暂存路径必须保证交给内核的 mask 张量与切片数据的行 stride 一致：若切片压缩了行而没有同步更新 `nb[1]`，就会错索引。该一致性无设备证据。另：`fattn.cu` 的 staged dst 整体拷贝 `*dst`，只覆盖 `src[5]`（chunk 界）与 `op_params[5]`（暂存 chunk 界），不改写 mask 张量。
6. `GGML_TP5_FA_MASK_KVLEN` 全树 0 处（与 production-audit 一致）。
7. 081601 分叉的“binder reader 拒绝去 cov 门控”修复**未叠加**；本基线仍为 cov 门控（残余 unsoundness，见 081601 REPORT）。
8. **GPU 侧（production-audit 记录，本刀未能验证）**：checked-tail 的 tile/vec 分支在设备上无证据；有界时 MMA 被改道 TILE
   （融合 turbo + 流式 MMA prefill 被跳过；MMA→TILE 改道 `fattn.cu:3493-3496`）；流式/暂存路径的 mask 切片行 stride
   一致性（内核读 mask 自身 `nb[1]`，非硬编码）未在设备验证；head_dim 192/320/512/576/640 且收窄后的 K 非 256 对齐时
   选择器 NONE → `GGML_ABORT`（`fattn.cu:3499`，`ggml_cuda_flash_attn_ext_supported == (selector != NONE)` 见 `3513`）；
   `ncols2 > 1` 要求 `K->ne[1] % 256 == 0`，故非对齐收窄宽度会落回 `ncols2 = 1`（均为静态结论）。
9. **门开的代价**：`GGML_TP5_FA_NKV_PARAM=1` 同时关闭图复用与 union binder（`llama-context.cpp:3443/3504/3530`），
   每次 eval 全量重建（slot 5 是建图期值），解码会变慢——设计如此，不应被当作性能回归。

## 7. 禁令合规记录（本会话，逐条）

- 本刀**从未**运行 `g++/gcc/clang`、`-fsyntax-only`、`build-slot`、任何 harness/可执行、`shell -n`、mock/fixture。
- 禁令下达（“立即停止任何正在运行或计划的编译和测试”）之后，本刀只执行：`grep`/`sed` 读取、`git status/diff/apply --check/apply -R --check/apply/worktree/add -N`、`diff -r -q`、`cmp`、`sha256sum`、`awk` 统计，以及文件读写。
- `git apply --check` 属补丁完整性静态校验，不编译、不执行被测代码；已在此明确披露。
- 无任何后台编译任务：本会话启动过的后台项均为只读 `grep`（已完成）。
- 因此 §4 全部产物为**已写未跑**；任何“已通过/已验证”的表述都不适用于 CPU harness。

## 8. 未决事项

- 获得授权后按 `RUNBOOK.md` 执行 CPU harness（build-slot 纪律），并把结果回填 `VERIFY.txt`；通过后才可把对应行从 PARTIAL 升级。
- GPU/HIP 需另行授权；本刀不建议在无设备证据时合并。
