# RUNBOOK-12022: NOUT floor1 真实长驻驱动验收

## 执行边界（12022：仅静态读写）

* **驱动源码已交付**：`driver/nout_floor1_driver.cpp`（本仓库，见下文构建）。
* **12022 禁止一切编译、测试、语法检查及 GPU**：本文所有构建/运行命令只在 **13022**（持本机串行锁的执行机）执行；13022 同样**禁止 GPU**（驱动为纯 CPU 路径，`GGML_CUDA=OFF` 构建，`llama_supports_gpu_offload()` 预期 `false`，实际值记录在 `context_new` 事件的 `gpu_offload_supported` 字段）。
* **13022 串行锁的具体路径/协议未知 → 缺依赖**：执行者必须先确认锁的获取方式（持锁前不得运行），本 runbook 不猜测。
* 交付物不改动 `selected`；不触碰 instrumentation 工作树；不叠加任何累计补丁。

## 前置依赖（缺失即 BLOCKED）

1. **13022 串行锁的路径/协议**（未知，不得猜测；执行者确认后方可运行任何命令）。
2. **真实 GGUF**：执行者提供的任意 decoder-only 模型；绝对路径替换下文 `$MODEL`。
3. **已构建的 libllama（独立副本，不写入 source tree/build）**：由 pristine floor1 树（提交 `d6d57375e88f4e223ab434eeb742f6a7f1412f83`）**复制出的独立副本**（`tree-patched`）应用最新 instrumentation 补丁后，在 13022 上 CPU-only 构建的产物；**原 tree 不打补丁、不覆盖**。下文以 `$BUILD` 表示该产物所在目录（含 `libllama.so`），`$TREE` 表示 tree-patched（仅用于 `-I$TREE/include`）。
4. g++ / GNU make（13022 上）。

## 构建顺序与前置补丁（13022 上执行；12022 不构建）

1. **instrumentation 补丁先应用且取最新版**（`nout-instrument-v1.patch`；评审仍在进行，`step_outcome` 现为 post-compute 语义——构建 libllama 前必须重新生成/取用最新补丁）。public API：`llama_nout_instrument_context_id(ctx)`、`llama_nout_instrument_set_tag(ctx, tag)`；环境 `LLAMA_NOUT_INSTR=/绝对路径/events.jsonl`。
   最新补丁指纹（instrumentation worker 交付）：sha256 `2c585ada9ea1cbe6e2d8e57836eacefd3aaff758bd69c4da22f4830f6d44adab`，5 文件 / 1287 行，`git apply --check` 在**独立副本**（tree-patched）上通过；应用后必须与该指纹核对。原 tree 不打补丁、不覆盖。
2. 构建驱动（产物与库副本均放独立目录，**绝不写入 source tree/build**）：

```sh
# 原 tree 只读、绝不覆盖：先复制出独立副本，补丁只打在副本上
cp -a /home/kunweiz/work/lead-nout-reuse-20261030/tree \
      /home/kunweiz/work/lead-nout-reuse-20261030/tree-patched
cd /home/kunweiz/work/lead-nout-reuse-20261030/tree-patched
git apply --check /home/kunweiz/work/lead-nout-instrument-20261030/nout-instrument-v1.patch
git apply /home/kunweiz/work/lead-nout-instrument-20261030/nout-instrument-v1.patch

TREE=/home/kunweiz/work/lead-nout-reuse-20261030/tree-patched   # 打补丁后的独立副本（llama.h 含 instrumentation API 声明；纯净树没有，直接 -I 纯净树编译失败）
BUILD=/home/kunweiz/work/lead-nout-driver-20261030/build        # 独立副本目录（13022 上，含 libllama.so）
LIBLLAMA_SRC=/13022上构建好的libllama.so所在目录/libllama.so     # 执行者填真实绝对路径（13022 从 tree-patched 构建）
cp "$LIBLLAMA_SRC" "$BUILD"/
cd /home/kunweiz/work/lead-nout-driver-20261030
g++ -std=c++17 -O2 -Wall -Wextra -I"$TREE"/include driver/nout_floor1_driver.cpp \
    -L"$BUILD" -lllama -o "$BUILD"/nout_floor1_driver
ldd "$BUILD"/nout_floor1_driver   # 记录实际动态库解析结果
```

## 运行（13022 上执行，持串行锁）

```sh
MODEL=/absolute/path/to/real-model.gguf
BUILD=/home/kunweiz/work/lead-nout-driver-20261030/build
export LLAMA_NOUT_INSTR=$PWD/nout_instrument_events.jsonl   # instrumentation 事件文件（绝对路径，运行前设置）
LD_LIBRARY_PATH=$BUILD "$BUILD"/nout_floor1_driver --model "$MODEL" --event-log "$PWD"/driver_events.jsonl > driver_stdout.log 2> driver.log
echo "exit=$?"   # 0 = 全部对照通过；1 = 存在失配；3 = 用法/资源错误
```
## 判定（驱动自动执行，人工复核事件流）

| 检查 | 判定 | 事件 |
|---|---|---|
| 缺行：n_out>0 时每个被标记行（`n_tokens-n_out+j`）必须可读 | 任一缺行 → `api_err` + FAIL；**始终核验，不依赖探针**（关闭 `--probe-empty-rows` 也核验） | `row_missing` |
| OFF/ON 输出行数一致 + 全词表逐行位相等（`--tol` 0 = 位相等） | 行数不等或任一元素不等 → `mismatches++` → FAIL | `compare` |
| 零输出步（n_out=0）不读取任何 logits 行；`--probe-empty-rows` 时追加负探针：row 0 必须不可读 | 探针读到非空 → `row_probe_bad` + FAIL；不开探针时该性质**不核验** | `row_probe` |
| 额外行（n_out>0 时第 `n_out` 行可读） | **当前驱动不探测**：n_out>0 步骤无此检查，`--probe-empty-rows` 只覆盖 n_out=0 场景——"无额外行"在 grid/k 步骤上**未验证**，不得宣称 | — |
| R3 复算 R1 脚本：逐行 argmax + checksum 与 R1 一致 | 不一致 → FAIL | `temporal_check`（仅 R3） |
| 每请求 KV 清理（seq_rm 全范围）两臂一致 | `removed=false` → FAIL；n_kv 计量仅记录（`kv_bucket_cross`），不作为判定 | `kv_cleanup` |

## Sampler 转换阶段（report:17，独立于 grid 的请求 R4）

四类真实 backend sampler 状态转换按锁定步执行于两臂：**add**（`llama_set_sampler` 挂 chainA）、**replace**（chainA→chainB，仅在 add 双臂接受后执行）、**remove**（nullptr，floor1 lane 重新武装）、**remove-again**（no-op：无状态变化、无 reserve——report:17）。每次转换后两臂各 decode 一步（n_out=1）并按 (pid, ctx, step) 与 instrumentation 事件合并。

**CPU-only 阻碍（如实记录，不回避）**：attach 成功要求 `can_offload` = `backend_init && backend_apply && chain_n > 0`（`llama-context.cpp:2585-2586`，纯净树行号；tree-patched 上因补丁偏移为 2623-2624）；chain 的 `backend_init`（`llama_sampler_chain_backend_init`，`llama-sampler.cpp:700-731`）要求每个元素实现 `backend_init` 并返回 true。链不可 offload 时 attach 被拒绝：返回 false、sampler 集保持为空、lane 保持武装、无 reserve（base 行为 `llama-context.cpp:2600-2611`，已核对）。驱动记录每次转换的实际返回值与后续 decode 的对照结果；成功路径（reserve 触发、图身份重建、lane 恢复）只有在 instrumentation 事件流证明 reserve 事件存在时才算验证，不算预先通过。

**未实现（不得宣称完成）**：运行时（base + 当前 instrumentation 补丁）对"sampler 拒绝替换已有链"没有拒绝分支——`set_sampler` 在新链可 offload 时直接覆盖 `sampling.samplers[seq_id]`（`llama-context.cpp:2588-2598`，补丁未改动该函数），不发任何"拒绝"事件。驱动 S2 的 replace 仅记录运行时实际返回值与后续对照；"拒绝"在当前运行时上不会发生，该分支属于运行时侧后续交付，事件流中出现明确拒绝事件之前本项不算覆盖。
## Instrumentation 关联（另一 worker 的对接契约）

* **进程级**：驱动在每次 `llama_decode` 前发 `step_begin`（`arm`、`request`、`step`、`phase`、`n_tokens`、`n_out`、`pos0`、`ctx_id`），每臂每步后发 `step_end`（`n_out_seen`、`n_kv_est`、`bucket_crossed`、`trace_seen`/`trace_reuse`/`trace_miss`/`trace_dec`/`trace_n_out`/`trace_stored`）。`max_abs`/`max_rel`/`bitexact` 属于 `compare` 事件，不在 `step_end`。
* **运行时侧**（instrumentation 工作树负责）：`LLAMA_NOUT_INSTR` JSONL 事件按 (pid, ctx, step) 发，**两臂都发**——补丁的 step 守卫在 `process_ubatch` 入口无条件构造，仅受 env 开关控制，不随 lane 关闭（见 `nout-instrument-v1.patch` 的 `process_ubatch` hunk）。`NOUT1_FLOOR_TRACE` stdout 统计则**仅 ON 臂**（OFF 臂 floor1 lane 解除武装，静默）。驱动侧 `step` 与 ON 臂 `dec=` 计数**期望** 1:1（`step` 从 1 起计数，每个 context 独立），但以事件流顺序为准、不作为假设（异常路径可吞步）。
* **事件流顺序（每步）**：`step_begin(OFF)` → `decode(OFF)` → `step_end(OFF)` → `step_begin(ON)` → `decode(ON)` → `step_end(ON)` → `compare`（R3 另发 `temporal_check`）；sampler 阶段每转换前另发 `sampler_phase`（`arm`、`action`、`accepted`）。JSONL 事件两臂都有、以 `ctx_id`/`arm` 区分；`dec=` stdout 行仅 ON 臂。
* **关联请求**：`request` 单调递增，两臂同 `request` 执行相同 token 序列（LCG seed 固定，`--seed`）；`pos0` 为该请求内绝对位置起点。

## 已知边界（不宣称覆盖）

* **每请求 KV 隔离**：驱动在每请求后 `llama_memory_seq_rm(mem, seq, 0, -1)` 清理该请求全部 KV（两臂一致），下一请求位置从 0 重新计数。请求间共享 prefix 不在本驱动范围（那是 union binder 的职责）。
* **n_kv bucket 不预设保证**：驱动只记录每步的驱动侧 cell 估算与 `bucket_crossed` 事件；不预先断言全程不跨 256 对齐 bucket，也不把 crossing 当作通过或失败——归因必须由 instrumentation 侧的 n_kv 事件核对（两套数字不一致本身就是发现）。
* **sampler**：grid/gen 阶段不挂任何 sampler chain（floor1 前置条件）；logits→token 的确定性转换（贪心 argmax，首个最大者胜）由驱动自行完成，两臂同规则。**采样限制（如实说明）**：token 序列由脚本（固定 LCG token）与 OFF 臂贪心 argmax 决定，R4 挂载的 greedy/dist chain 从不参与 token 选择，其唯一作用是触发并观察 attach/replace/remove 的状态转换与 lane 开关；除贪心 argmax 外本驱动不覆盖任何随机采样路径（温度/top-k/top-p/dist 均不进入判定）。真实的 backend sampler 状态转换由独立的 Sampler 转换阶段覆盖（见上节），两者不相混。
* **本仓库（12022）未编译、未运行本驱动**；所有构建/运行/校验都在 13022（持串行锁，禁 GPU）执行。`bitexact` 预期在 CPU 单线程同构路径成立，跨后端不适用（`--tol` 语义见 `--help`）。
* **13022 串行锁的路径/协议未知**（见前置依赖第 1 条）：在执行者确认之前，本文所有命令都只是记录，**整体处于 BLOCKED 状态**，不是可执行验证；持锁并确认协议之前不得运行任何命令。
