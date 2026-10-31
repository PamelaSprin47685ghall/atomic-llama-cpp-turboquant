# RUNBOOK — union binder × tp5-relay fail-closed

状态：**PARTIAL。最终源码未编译、完整 `libllama` 未链接、harness/比较未运行。** 中断前（10:24）只编译过一个随后过期的单对象；新纪律（约 10:32）后未再编译/测试。本 runbook 区分“已执行”与“未来在获准环境执行”；所有 hit/bitwise 均为未来验收预期。

## 1. 基线与补丁身份

- 独立树：`/home/kunweiz/work/lead-union-relay-20261030/wt`
- HEAD：`b766f29549e46abb2ba65adfbae69ec035a85fe0`
  - 父 commit：干净 stock `f1c5fafc0449b22969ddabfcc9c361940d94d033`
  - `b766f29` 自带 r5 nout + r7 nkv 累计改动；本包不叠加任何其他并行补丁。
- 本包增量：`patch.diff`，仅 `src/llama-context.cpp`，+63/−36，
  sha256 `7fcf73674fd7efa6b058b7aad426cc97a72b27975276eda093a52f44b49d0980`
- 干净 stock 对照 worktree：`/home/kunweiz/work/lead-union-relay-20261030/stock`
  = `f1c5faf...`。只读，未修改。
- 未纳入：union-binder-true-width-r8 工作树里未提交的
  `src/llama-kv-cache.{cpp,h}` exact-span 增量（与本问题无关，避免叠加）。

补丁应用检查（静态，已于 2026-10-30 执行）：

```
git -C wt apply --check --reverse "$PWD/patch.diff"   # exit 0
git -C wt diff --check                                # exit 0
```

**不要**把补丁应用到 r8 工作树或任何已有累计改动的树；只应用到干净基线
`b766f29`（或本独立树）。

## 2. 代码改动摘要

三层 fail-closed：

1. **构造期拒绝**：`params.tp5_relay` 为真且任一 binder 门为 "1" 时，
   不创建 state、不注册 lane，打印一次
   `UNION_BINDER_REFUSED tp5-relay context: ...`。
2. **record 拒绝**：`env_out` 要求 `backend_tp5_relay == nullptr`；
   `binder_record_snapshot()` 在 relay 非空时计数并直接返回，不读 graph。
3. **rebind 拒绝**：两条 lane 在 `++c_attempts` 后、读取 `res`/`set`/record
   之前检查 relay/pipeline；relay 非空时 `++c_refused_relay` 并 fallback。
   nkv lane 不再在拒绝前解引用 `*set.tensors`。

观测语义（必须如实报告）：

- **正常 relay context**：只有构造期 `UNION_BINDER_REFUSED` warning；
  **没有** `NKV_BINDER_STATS`，**没有** `refused_relay` 计数。
- **CPU 非 relay + 两门 ON（未来运行）**：若真实 binder 命中，析构会打印
  `NKV_BINDER_STATS attempts=... applied=... record_builds=...`。
  `applied` 才是 hit；`record_builds` 是 fresh build 次数，不是 hit。
  当前未运行，不能写成已经命中。
- **受控注入 `backend_tp5_relay`**：可观察 rebind/record backstop 的
  `refused_relay`，但这是生产分支测试，不是 relay 运行。

## 3. 依赖

编译/运行环境：

- `cmake`、GNU make、`g++` 16.2.1、`ccache`
- `/home/kunweiz/.local/bin/build-slot`（180 s 排队 / 180 s 编译；75/124 停止，不重试）
- 可选 OpenSSL（本配置自动找到；不影响 binder 代码）
- 无新增库依赖；改动只使用已有成员和计数器，公开 `llama.h` ABI 不变。

测试输入：

- `tiny-fake.gguf`，sha256 `d0d12f836a87daaa56776ea7cf4758341b83d23d0d7889c4a61aa2afe5116c45`
  （复制自 `/home/kunweiz/work/nkv-fa-true-len-prove-r8-065620/tiny-fake.gguf`；
  该文件由 `gen-tiny.py` 生成，不是本树产物）。
- `harness/resident_prod.cpp`、`harness/relay_branch_test.cpp`、
  `compare_logits.py`。三者均未编译/运行。

## 4. 构建（未来、获准后）

### 4.1 patched

```
cd /home/kunweiz/work/lead-union-relay-20261030
/home/kunweiz/.local/bin/build-slot cmake -S wt -B build -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS_DEBUG='-O1 -g0' \
  -DCMAKE_CXX_FLAGS_DEBUG='-O1 -g0' \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF \
  -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_SSE42=ON -DBUILD_SHARED_LIBS=ON
./build-segmented.sh
```

- 本机是 AMD Athlon X4 760K；flags 无 BMI2，必须关闭 `GGML_BMI2`，否则运行可能 SIGILL。
- `build-segmented.sh` 每批 3 个对象；只编译缺失或比源文件旧的对象；
  遇 75/124 立即停止，不重试。
- 当前 `build/` 有 112 个旧 `.o` 和 ggml 库，但**没有 libllama**；
  `llama-context.cpp.o` 产生于最后一次源码清理之前，重新构建时必须由 make 重新编译。

### 4.2 stock 对照（完整脚本）

```
cd /home/kunweiz/work/lead-union-relay-20261030
./build-stock-segmented.sh
```

该脚本完整执行 4.1 同参数的 stock `f1c5faf` configure，然后调用
`./build-segmented.sh "$PWD/stock" "$PWD/build-stock"` 完成分段对象与链接。
也可手工执行：

```
/home/kunweiz/.local/bin/build-slot cmake -S "$PWD/stock" -B "$PWD/build-stock" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS_DEBUG='-O1 -g0' \
  -DCMAKE_CXX_FLAGS_DEBUG='-O1 -g0' \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF \
  -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_SSE42=ON -DBUILD_SHARED_LIBS=ON
./build-segmented.sh "$PWD/stock" "$PWD/build-stock"
```

不要把 patched 的 `.o`/`.so` 与 stock 混用。stock 构建尚未执行（纪律禁止）。

## 5. 验证矩阵（未来、获准后）

| 臂 | 树 | 环境 | 预期 |
|---|---|---|---|
| stock | `stock` | 两门未设置 | 基线 logits |
| OFF | patched | 两门未设置 | 与 stock 逐 bit 一致 |
| ON | patched | `GGML_TP5_NKV_TRUEWIDTH=1 GGML_TP5_UNION_BINDER=1` | 预期：真实 nkv 绑定 hit（运行后按 `applied>0` 判定）；logits 预期与 stock 逐 bit 一致（若 binder 正确） |
| 注入 | patched | 两门 ON + harness 内注入 | 两条 rebind lane 拒绝，`refused_relay>=2`，`applied=0`，无新 record |
| 真实 relay | 5×HIP 环境 | `tp5_relay` + 门 ON | `UNION_BINDER_REFUSED` warning；无 stats；无 GPU 不得执行 |

运行细节：

- `resident_prod`：4 个请求，每请求 prime 768 token（单 batch）→ 删除
  `[256,inf)` → decode 64 → decode 448；`get_n_kv` 的 256 桶因此从 768→256→512→768
  移动，n_outputs 也从 768→64→448 变化。每个 batch 的 `batch.logits[i]` 全置 1。
  输出 `(n_tokens, vocab, float32[])` 记录。
- OFF/ON 必须使用同一模型、同一 token 序列、同一线程数；分别运行进程。
- `compare_logits.py` 逐记录、逐 float32 bit 比较；任何缺失/长度不符/首个
  差异必须原样记录，不能只看摘要。
- ON 的 `NKV_BINDER_STATS` 必须来自同一 context 正常析构；`applied` 为真实
  hit。不得把 `record_builds`、graph rebuild 次数或 mock 调用计为 hit。
- `relay_branch_test` 只用于拒绝分支：它把真实 CPU backend 指针注入私有
  `backend_tp5_relay`；生产代码对该成员只做 null 判断，不解引用。结果只能
  称“受控注入的生产分支测试”，不能称 relay hit/bitwise。

## 6. 已知阻碍

- 无 5×HIP，真实 relay context 无法创建；本机不能产生 relay 证据。
- 新纪律禁止编译/测试；当前所有运行级项均为未执行。
- relay-aware in-place binding 未实现：relay 要求 graph metadata 在
  `prepare` 前稳定并在 rebuild 后重新验证 placement；binder 原地改 shape 的
  契约不兼容，且无真机证据。当前选择 fail-closed。
