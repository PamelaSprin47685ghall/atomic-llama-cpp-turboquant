# NOUT floor1 本机静态验证设计与执行依赖

## 结论和本次边界

本报告仅基于静态源码；没有执行任何构建、测试、syntax-only、mock、fixture、harness 或 GPU 工作，没有操作 selected、运行 `setup-tree.sh` 或使用 RSZ_RT。

**零行保留、风险未验证**：静态证据不足以认定零行执行必然有漏洞，因此保留原零行 bind；CPU `ggml-cpu.c:1934` 空 tensor 不计算，CUDA `ggml-cuda.cu:3970,6216` 将空节点视为 no-op，但这不足以证明 scheduler、分裂、内存分配及所有后端的端到端安全。零行真实计算、图身份与输出均未运行验证，不能宣称安全或把回退重建算命中。

`fix-d6d57375.patch` 仅针对本地提交 `d6d57375e88f4e223ab434eeb742f6a7f1412f83` 的增量。可从现存 `tree` 用 `git bundle create <bundle路径> HEAD` 导出基线供外部 clone；不能运行会删除该树的 `setup-tree.sh`。该提交的内容来自源仓 `f1c5fafc0449b22969ddabfcc9c361940d94d033` 的归档加一次 `in/wire-cumulative.patch`；自建提交哈希不同，必须确认内容一致，不得叠加其他累计补丁。本次未应用补丁或执行验证。

## 已有证据及限制

* `tree/src/llama-context.cpp:508-531`：`LLAMA_NOUT_FLOOR1_REUSE` 仅精确 `1` 开启，环境在 context 创建时读取；对照必须创建独立 OFF 与 ON context。`GGML_TP5_NOUT_REUSE` envelope 路径不能混入 floor1 单因素比较。
* `tree/src/llama-context.cpp:3088-3109` 与 `:3153-3226`：复用决定、`NOUT1_FLOOR_TRACE` 的 `reuse/miss/n_out/key/stored`；但此 trace 在后续 bind 之前。`:3383-3468` bind/replan 失败会走重建并回滚已增的 `n_reused`。所以 trace 决定本身既非最终 hit，也不是图实例标识；须核对最终 STEP、fallback、性能计数和真实图代次。
* `tree/harness/nout_floor1_reuse_hits.cpp:1-68,403-414,465-518` 是未执行过的合成模型 harness，grid 实际为 `(4,4),(1,0),(1,1),(1,1),(1,0),(3,1),(1,0),(1,1),(1,0)`；没有同形 2/k 转换、长驻多请求、真实图身份。文件自身描述不能当验证结果；本任务禁止运行它。
* `tree/src/llama-context.cpp:216-237` 显示 relay 的五 HIP rank、完整 offload、批次/sequence 等 profile 限制，并禁止 sampler、eval callback 等配置；不能假设通用 CPU driver 或 callback 能原样进入 relay。`:3038-3043` relay 要在图资源改动前同步。详见下文冲突。
* sampler 添加、替换、移除或拒绝后删掉已存在 sampler 均触发 reserve；reserve 同步并销毁旧结果。旧 NKV record 含旧图指针，现于替换前标记无效，避免切换后被复用。无 sampler 的重复移除无状态变化，不触发 reserve。以上为静态推断，挂卸及多请求尚未实测。
* owner 表标志原按 context 生效，`can_reuse` 在 encoder/MTP 图上也可能跳过输出数检查，而相应 build/bind 未启用；现在两个 runtime helper 均按 graph type 拒绝非 DEFAULT/DECODER，与 build/bind 门一致。图类型切换和计数仍需真实验收。

## 现在可执行：只读清单命令

下面命令不改文件、不运行目标程序。绝对路径可直接复制；命令输出仅供记录，不能证明行为。若某路径不存在，保留错误并列为缺依赖，不做 setup、不创建替代物。

```sh
cd /home/kunweiz/work/lead-nout-reuse-20261030
pwd
 git -C tree rev-parse HEAD
 git -C tree status --short
sha256sum in/wire-cumulative.patch /home/kunweiz/work/lead-union-relay-20261030/patch.diff
 git -C tree diff --name-only
```

对于选定的真实 GGUF 和将来授权提供的 instrumented 二进制及依赖库，使用实际绝对路径，只读记录文件身份（不要改 selected）：

```sh
MODEL=/absolute/path/to/real-model.gguf
DRIVER=/absolute/path/to/real-request-driver
LIBLLAMA=/absolute/path/to/libllama.so
stat -- "$MODEL" "$DRIVER" "$LIBLLAMA"
sha256sum -- "$MODEL" "$DRIVER" "$LIBLLAMA"
ldd -- "$DRIVER"
```

占位路径必须由操作者替换为实际存在的产物；`ldd` 仅用于可信的本机二进制。还应记录实际动态库解析结果、CPU 后端和线程数、完整不含敏感信息的环境及源码版本；同一批对照不得混用不同构建或模型。不得用这里的只读命令替代下述行为验收。

## 尚需实现的真实驱动与图身份记录

需另行提供一个使用真实 GGUF、真实 `llama_decode` 和生产 context 的本地驱动；本报告不声称它已存在，也不指定虚构的可运行命令。驱动需在**一个进程内**创建 ON 和 OFF 各一个持久 context，分开维护等价 KV 历史，在多个请求中保持这两个 context 存活，记录 PID、context 实例 ID、request ID、seq ID、step、绝对 position、token、每个 token 的 logits flag；请求交替、结束及下一请求的 sequence/KV 清理语义在两条臂完全一致。设置 context 上限和物理位置，确保测试段在同一个 n_kv bucket，`n_batch/n_ubatch` 足够并保证每次恰一个同形 ubatch，关闭 backend sampler。实际输出 count 必须由 runtime 记录核实，不能仅信请求标记。

每个测量阶段以相同 token 数、sequence 数和其余 graph key 相同的真实 ubatch，使用输出数 **0, 1, 2, k, 0**，其中选取 `k > 2` 且 `k <= n_tokens`，例如 `n_tokens=4, k=3`；0 时显式全部 logits flags 为 0。保证这些 batch 不被切分，记录实际 n_kv、ubatch 结构、graph type、sampler 和上下文参数，否则无法将 miss 唯一归因于输出数。应在多个请求中重复整段，并在同一个 context 内至少覆盖一次紧邻的 `0→1`、`1→0`（补一段 `1,0`）及 `1→2→k→0`；另记录首图冷启动、跨请求状态、请求结束后的缓存清理和意外 bucket 边界，不能把形状变更导致的 miss 算成输出数 miss。

**静态预期，非验收结果**：允许零行 bind；planner 不支持、容量不足、调度重分配失败等仍可能触发 exact-count fallback，不能预判任何一步命中。记录 `bind_zero`、真实图代次、fallback 和最终计算图；比较 `0→1`、`1→0` 以及 `1→2→k→0` 的实际结果，不能把 bind 成功单独当成复用。stock/OFF 零输出路径与所有目标后端的安全性尚未验证；任何额外 key 变动均须记录，不能强行归因。

instrumentation 必须在实际 `process_ubatch` 路径中记录 context 唯一 ID、step、原 graph 指针、当前 graph 指针、每次真实 build 的单调 generation、最终计算图和 fallback/命中；指针地址本身不足以证明身份。关联 request、graph generation、输出数及最终计数，区分 scheduler replan 与 graph rebuild。零输出 fallback 单独计数且不算命中。对于 relay，另记录 Meta 子图、provider prepare/submit、epoch 和完成，不能以 host graph 指针声称 GPU native replay 命中。instrumentation 需在另行授权的工作树实现。

## 输出对照与验收

同一二进制、真实模型、硬件/后端、线程和输入上，分别使用 floor1 OFF（`LLAMA_NOUT_FLOOR1_REUSE=0`，envelope 和其他实验开关关闭）及 ON（`LLAMA_NOUT_FLOOR1_REUSE=1`、`LLAMA_NOUT_FLOOR1_TRACE=1`、envelope 关闭）。同一进程内在创建各 context 前设置其环境，之后避免并发修改进程环境；两臂交错执行相同的请求和 KV 操作。可另以独立 stock 构建交叉确认 OFF，但需先明确 stock 构建来源和可比性，不能把 floor1 OFF 自动冒充已验证独立 stock 二进制。

同步完成后，按 **token index** 通过 `llama_get_logits_ith` 取每个标记输出的完整 vocab 行，对照 token ID、位置、输出行数、数据和后续确定性生成结果；缺行或额外行失败。CPU 固定单线程同构路径可先要求逐字节一致；不同设备、算法或后端不得无依据强制逐字节一致，应先约定数值误差上限并逐元素报告 max abs/relative、NaN/Inf 和差异行，不能事后调宽。0 输出不得读取不存在的 logits 行；确认 API 允许后才检查缓存可见性，不能把旧 buffer 内容误当新输出。对照执行顺序和 KV 清理必须相同。输出结果、逐步日志、完整环境、产物哈希和退出码均需留存；只要驱动或 instrumentation 不存在，该项标记 BLOCKED，不给伪命令、不宣称通过。

## Union binder 与 relay：不能盲叠补丁

用户给定 `/home/kunweiz/work/lead-union-relay-20261030/patch.diff` 的基线是 `b766f29549e46abb2ba65adfbae69ec035a85fe0`；它依赖 union binder record/rebind，不能直接应用于 stock 或 stock+floor1。该补丁 `:5-29` 中 relay 拒绝门仅检查 `GGML_TP5_UNION_BINDER` / `GGML_TP5_NKV_TRUEWIDTH`，不覆盖 floor1；`:32-45` 对 binder record 增加 relay backstop。用户提供的分支冲突是：floor1 和 binder 均定义 `src/llama-nout-state.{h,cpp}` 的 `llama_nkv_ctx_state` 和同名 getter，字段不同，并共同修改 context、graph、KV、CMake。必须先在另行授权的集成分支统一 state 的生命周期、字段、getter 和 record/rebind 所有权，明确 floor1 与 relay 是否互斥并实现覆盖 floor1 的 fail-closed 门，再检查所有调用点及 OFF 语义；不能盲目累计应用补丁，也不能将现有两个拒绝条件误认为 floor1 已被保护。

现有 floor1 bind 会对 live graph 缩窄并重置/重分配 scheduler（`tree/src/llama-context.cpp:3383-3421`），而 relay 补丁明确依赖 shape stability、provider 对图与终端的准备和所有权，这就是契约冲突，不是仅靠门控变量改名可解决。若目标是允许组合，需先设计并证明同步、record 恢复、Meta 子图/terminal 重新准备、native submission/完成及错误回退的完整协议；否则明确拒绝组合，独立验证 CPU floor1 和 relay OFF。relay 验收还需受支持的五张 HIP 设备、匹配 provider/ABI、真实模型与输入、满足 profile 的 driver、完成事件与错误日志；在本次禁止 GPU 的条件下均未运行。

## 交付前依赖清单

1. 经授权可用的真实 GGUF、可追溯的 instrumented runtime 和其匹配库；不覆盖 selected。
2. 实现并交付上述真实长驻多请求驱动、图实例/最终 hit instrumentation；提供真实的 `--help` 和该产物对应的**实际**启动命令。现在不存在可诚实复制的完整验收启动命令。
3. 在授权环境确定 stock/OFF/ON 对照与容差、执行许可和证据保存位置；若涉及 relay，先解决集成冲突、准备五 HIP rank 和 relay provider。
4. 执行者随后才按驱动真实接口运行，保存完整命令、退出码、逐步结构化事件及结果；失败或缺证据不得标记通过。本次未做该步骤。
