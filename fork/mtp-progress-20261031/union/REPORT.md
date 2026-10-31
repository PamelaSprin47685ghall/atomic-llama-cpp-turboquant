# REPORT — union binder × tp5-relay fail-closed（PARTIAL / NO-PROMOTE）

## 状态

- **PARTIAL。最终源码未编译、完整 `libllama` 未链接、任何 harness/比较未运行；所有运行级结论均未验证。** 新纪律（约 10:32）后未再编译/测试。中断前 10:24 只对当时的 `llama-context.cpp` 做过一次单对象编译（exit 0）；随后源码又做了清理，该 `.o` 已过期且未重编。已立即停止正在运行的构建（`bg_26` 取消，无退出码）。本文除该单对象编译与静态检查外，全部为**静态源码审查**。
- 未触碰 GPU/HIP、`RSZ_RT`、`selected`。未提交、未推送。
- 不声称 relay hit、bitwise 一致、性能或设备安全。真实 tp5-relay 需要 5 张同 registry HIP GPU，本机无此条件。

- 新纪律下**无需再执行任何测试**；本包交付以补丁、静态审查和未来命令为限。

## 独立树与基线

- 独立树：`/home/kunweiz/work/lead-union-relay-20261030/wt`
- 基线：`b766f29549e46abb2ba65adfbae69ec035a85fe0`
  - 其父为干净 stock `f1c5fafc0449b22969ddabfcc9c361940d94d033`。
  - `b766f29` 本身是前序 r5 nout + r7 nkv 的累计基线；本包**只加自己的增量**，没有叠加 r8 工作树未提交的 `llama-kv-cache.*` exact-span 改动，也没有叠加 maxfp 或其他并行补丁。
- stock 对照 worktree：`/home/kunweiz/work/lead-union-relay-20261030/stock` = `f1c5faf`（`git worktree`，只读对照，尚未构建）。
- 真实补丁：`/home/kunweiz/work/lead-union-relay-20261030/patch.diff`
  - `src/llama-context.cpp`，+63/−36，sha256 `7fcf73674fd7efa6b058b7aad426cc97a72b27975276eda093a52f44b49d0980`
  - `git -C wt apply --check --reverse patch.diff` exit 0（确认补丁与工作树增量一致）。
  - 只改 1 个生产文件；没有新增依赖、公开 ABI 或数据成员。

## 关键现状（r8 调查结论）

1. `src/llama-context.cpp:206-229` 对 `params.tp5_relay` 强制：tensor split、单 Meta device、`GGML_RELAY_RANKS`=5、全 GPU layer/output offload、`offload_input`、1..4 行等。
2. `ggml/src/ggml-backend-meta.cpp:1864-1889` 对 `"tp5-relay"` 强制 5 个设备全部来自**同一个名为 `ROCm` 的 registry**、`GGML_BACKEND_DEVICE_TYPE_GPU`、且该 registry 导出 `ggml_backend_relay_get_interface`。
3. 该 provider 只由 `ggml/src/ggml-cuda/ggml-cuda.cu:7802-7803` / `relay.hip:1815` 导出。CPU 后端不导出该符号，registry 名也不是 `ROCm`。
4. 因此：**生产 tp5-relay context 在纯 CPU 上不可创建**；不能把非 relay CPU binder hit 称作 relay hit。
5. r8 原实现中，两个 rebind lane 在 `backend_tp5_relay != nullptr` 时拒绝（原 :2795、:2955），但：
   - 绑定注册（:466-508）不检查 relay；
   - n_outputs 的 record 在 relay 下仍会执行（原 :3497-3541）；nkv 的 record 已被 :3480 排除；
   - 于是 relay 下存在“先 record、再 rebind 必拒”的无效工作，且拒绝计数只在 rebind 被调用时才出现。

## 本轮改动（fail-closed，三层）

### 1) 构造期整体拒绝（主路径，可观测）

`llama_context` 构造函数中，`relay_binder_refused = params.tp5_relay && (GGML_TP5_UNION_BINDER=="1" || GGML_TP5_NKV_TRUEWIDTH=="1")`：

- 不创建 binder state slot，不注册任何 lane；
- 打印一次 `UNION_BINDER_REFUSED tp5-relay context: ...` warning；
- relay context 的 record/rebind 路径因此完全不可达。

注意：此分支在 `:206-229` 的 relay 前置校验**之后**才执行。CPU 上以 `params.tp5_relay=true` 创建 context 会先在前置校验抛错，所以本机无法观测到该 warning。**正常 relay 路径不会产生 `NKV_BINDER_STATS`，也不会产生 `refused_relay` 计数**；报告和验收不得宣称正常路径有拒绝计数。

### 2) record 硬门

- `process_ubatch` 的 `env_out` 增加 `backend_tp5_relay == nullptr`；relay context 不再对 n_outputs 取 pre-allocation record。
- `binder_record_snapshot()` 增加第二层 backstop：`backend_tp5_relay != nullptr` 时 `++c_refused_relay` 并直接返回，不读取/记录 graph。该函数只对 relay 指针做 null 判断，不解引用。
- nkv record 原本已由 `backend_tp5_relay == nullptr` 门控。

### 3) rebind 拒绝提前到读取任何状态之前

`union_rebind_n_kv` 与 `union_rebind_outputs` 中，relay/pipeline 拒绝从“读取 record 之后”提前到 `++c_attempts` 之后、`res`/`set`/record 检查之前：

- 先计数 `c_refused_relay` / `c_refused_pipeline`，再 `++c_fallbacks`、返回 false；
- nkv 的 `*set.tensors` 别名移到拒绝之后，避免 relay 拒绝路径解引用空/无效 set；
- 这两个计数只在 lane 仍被注册时才可能增加，即**仅用于 backstop 或受控注入测试**，不是正常 relay 路径的观测。

## 生命周期 / 线程 / shape 依赖（静态审查）

- `backend_tp5_relay` 仅在 `llama-context.cpp` 初始化后端时赋值一次（原 :608），且位于 `if (params.tp5_relay)` 内；grep 全树无第二处赋值。该成员是**非拥有指针**，所有其他使用点均为 `!= nullptr` / `== nullptr` 判断，无解引用。
- 绑定注册发生在同一构造函数的更早位置（原 :466），此时成员尚未赋值，因此主门必须用 `params.tp5_relay`；两者由上述唯一赋值点绑定，语义等价。
- binder state 是每个 `llama_context` 自有的 `unique_ptr<llama_nkv_ctx_state>`，无进程全局表、无跨 context 共享；decode/process_ubatch 在 context 自身调用路径内使用，本轮未新增跨线程状态。
- relay 侧契约：Meta backend 在 `needs_rebuild`（`ggml-backend-meta.cpp:2226`，由 cgraph uid 决定）时才重新 `prepare` relay terminals（:3787），并依赖 Graph/layout/binding 的稳定元数据；llama 侧还会在 :3554-3579 校验“单 split / 所有 compute 节点在 Meta relay 上”。
- binder 会在 `sched_alloc_graph` 之后原地修改 live graph 的 `ne`/`nb`/`op_params`/`view_offs` 并重跑 scheduler 分配。该行为与 relay 的“metadata 在 prepare 前不可变、submit 前重新验证 placement”契约冲突，且没有 relay-aware 的重新 `prepare`/原子回滚证据。
- **结论：选择 fail-closed，不做 relay-aware in-place binding。** 这是可证明的最小安全行为；relay-aware 绑定需要 5×HIP 真机验证，超出本机能力。

## 产物与未执行项

- 补丁：`patch.diff`（仅生产代码）。
- 验证驱动（未编译、未运行）：
  - `harness/resident_prod.cpp`：公开 llama API 的 CPU 真实路径，4 请求，每请求 768 token → 删除 `[256,inf)` → decode 64 → decode 448（跨 256 桶边界，n_outputs 768/64/448 同步变化），输出完整 logits 记录；用于**未来验证** OFF/ON 的 binder 真实命中与 bitwise 对照。当前未运行，不能写成已发生 hit。
  - `harness/relay_branch_test.cpp`：受控注入。用真实、已初始化的 CPU backend 指针填入私有的 `backend_tp5_relay`（所有生产检查只做 null 判断，不解引用），强制一次 graph-reuse 失败，断言两条 rebind lane 与 record 均被拒绝；**这是生产分支测试，不是 relay 运行**。
  - `compare_logits.py`：逐记录、逐 float32 bit 比较。
- 构建脚本：`build-segmented.sh [SRC] [BUILD]`（每次 build-slot 最多 3 个对象；75/124 立即停止，不重试）；`build-stock-segmented.sh` 是干净 stock `f1c5faf` 的完整 configure + 分段构建 + 链接脚本。
- 已停止的构建：`bg_26`（`./build-segmented.sh`）在用户新纪律下达后被取消；停止前已有 112 个 `.o` 和 ggml 共享库，**未链接 libllama**。详见 `VERIFY.txt`。
- **本包没有运行过任何 mock**；`harness/relay_branch_test.cpp` 的受控注入也未编译/运行。纪律后唯一执行的 shell 检查是 `bash -n`（10:37:17，仅 shell 语法，不是 C++ 编译/测试）。

## 明确阻碍

1. 本机无 GPU/HIP，且生产 relay 强制 5×ROCm GPU + HIP provider；**无法在本机产生真实 relay context、真实 relay hit 或 relay bitwise 证据**。
2. 新纪律禁止一切编译/测试；因此 CPU 真实 binder hit、OFF vs stock bitwise、注入拒绝测试均只有**可执行命令与静态断言**，没有运行结果。
3. relay-aware in-place binding 的可行性需要真机验证 Meta `prepare`/uid/placement 的完整生命周期；当前没有证据，故不实现。
