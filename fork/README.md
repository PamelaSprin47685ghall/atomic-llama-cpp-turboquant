# 干净的 TurboQuant fork

这里是自有功能的维护入口。A 固定在 AtomicBot-ai 上游 master 的 `cda0f9256665845cd9d6f3a0176880883f653a3d`；自有库 B 和编排程序 C 放在 `fork/`。少数必须由 A 自己完成的内存操作，以及已经有失败用例的正确性修复，以显式小补丁保留。

当前已实现 **RERoT 的逐 token 共享白板**：独立 B 拥有 DAG、reader 可见性和原子发布状态，C 通过公开接口编排一个真实 A context，并复用 TQRelay 的提交与排空合同。它建立在此前的阶段计算工作区、独立物理 KV 容量、Prefill 优先、准入队列、文本输出和校准工具之上。实现范围与运行资格分别记录；A 的实际变动以 [upstream-exceptions.json](upstream-exceptions.json) 登记的完整文件内容、消费者、理由和退出条件为准。

完整清单与取舍见 [AUDIT.md](AUDIT.md)、[MIGRATION.md](MIGRATION.md)，实际运行记录见 [VALIDATION.md](VALIDATION.md)。未迁入的有用能力仍是保留候选；旧实现与证据保存在 `archive/pre-clean-20261018`。

TP5 重写后的细节与原始论文索引见 [REWRITE-CANDIDATES.md](REWRITE-CANDIDATES.md)。该文保留最初文本审计的固定身份，另在 E11 / E23 记录后续 RERoT 重写及剩余条件。未证实工作的旧代码继续只作文本依据；已经属于 A 的能力直接复用。

## A / B / C 的职责

| 参与者 | 完整职责 | 声明依赖 |
|---|---|---|
| A：固定上游与登记的小补丁 | 模型、tokenizer、推理、采样、KV、后端和原版应用 | 原上游依赖 |
| B：TQSampling | 从数值数组返回拥有自身数据的有限 top-k 候选 | C++17 标准库 |
| B：TQCorpus | 请求 token trie、生产 V1 格式、请求种类及时间戳 | C++17 标准库 |
| B：TQScheduling | 独立的 token / 槽位预算，以及保持当前 Prefill 的候选选择；两对象由 C 组合 | C++17 标准库 |
| B：TQText | 增量 UTF-8、跨块 stop 匹配、拥有自身文本和字节游标的结果 | C++17 标准库 |
| B：TQCalibration | 有界读取 WXQCAL03，验证格式、统计人口和数值，逐层返回拥有自身数据的结果 | C++17 标准库 |
| B：TQRelay | TP5 五卡 RELAY 的纯 CPU 合同：通道 bank 生命周期、精确票据、两种 credit、fail-stop 与排空事实、逐 rank 原生提交登记、五 rank 合成 GLU 分片计划 | C++17 标准库 |
| B：TQRerot | DAG 与有界 pen 驻留、拥有自身数据的 reader 视图、逐 token cohort 原子发布、private / source-end 隔离、陈旧票据与溢出检查 | C++17 标准库 |
| C：compose | 用 B 选候选，调用 A 的真实采样链和模型生成 | 已安装 llama、TQSampling |
| C：corpus-compose | 用 B 读写请求，编排 A 的 tokenizer、decode 和序列快照 | 已安装 llama、TQCorpus |
| C：batch-compose | FIFO 准入、真实序列槽位、模型批次、阶段释放及 stop / 流式输出生命周期 | 已安装 llama、TQScheduling、TQText |
| C：logits-compose | 录制真实输入 token 和原始 logits，完整后发布新文件 | 已安装 llama |
| C：imatrix-compose | 明确选择 partial / unseen 策略，用有界磁盘暂存导出 A 可消费的 GGUF imatrix | 已安装 ggml 的 `ggml::ggml-base`、TQCalibration |
| C：rerot-compose | 显式 DAG 的 `assistant-eot` 共享白板；原生模板、真实 reader attention、独立 sampler、阶段释放和终结 | 已安装 llama、TQRerot、TQRelay、TQText |

A 与 B 不包含、调用或注入彼此的服务；各 B 也不依赖兄弟库。C 只消费安装后的公开接口。没有构建时改写源码、链接符号拦截或私有头文件桥接。原版 `llama-server` 的小范围调度修改是明确的 A 例外，尚未拆成完整独立 HTTP 编排层；详见 [UPSTREAM-EXCEPTIONS.md](UPSTREAM-EXCEPTIONS.md)。

## 构建

需要 Git、Python 3.12 或更新版本、CMake、C++17 编译器，以及所选 A 后端需要的工具链。完整交付目前在 Linux 实测；logits 和 imatrix 文件发布需要 POSIX `openat` / `linkat`，imatrix 输出限定为小端系统。转换本身不需要 Python 或 NumPy，Python 用于构建入口和测试。从仓库根执行：

```sh
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
python3 fork/build.py --backend vulkan --tools --build-dir build-clean-vulkan --jobs 4
```

脚本先检查 A，再单独构建和安装 A；随后分别构建、测试和安装七个 B，再分别构建六个 C。产物位于相应构建目录的 `install/`。默认组件测试不需要下载模型；启用的 Vulkan 接缝测试需要真实 Vulkan 设备。真实 RERoT 模型验收需要另外提供兼容的模型。

`--tools` 额外构建 `llama-server`、`llama-bench` 及 A 在 tools 模式下必须安装的 `mtmd`；工具可执行文件位于 `upstream/bin/`。脚本也接受 `cuda`、`hip`，可选后端不等于已经取得运行资格。2026-10-18 的 CPU 独立交付与 CPU / Vulkan 同形状流式模型对照保留各自身份；RERoT 的后续资格单列在 [VALIDATION.md](VALIDATION.md)。`--no-test` 只用于明确跳过测试的构建，不构成验证记录。

TP5 mapped-relay 的平台资格另有独立探针 [relay-probe](relay-probe/README.md)，其无 GPU 的 config-test 与证据校验器 `--self-test` 已进入上述默认流水线；探针可执行文件需要 ROCm HIP 工具链，只在 HIP 存在时构建，默认流水线不安装它（isolation 验收只排除无 HIP 的安装情形；探针自带的 `cmake --install` 在 HIP 存在时装出两个可执行文件、校验器与文档，已做过独立前缀冒烟）。它在每张现有设备上验证双向运行期可见、热复用、精确票据、多工作组完成与 abort、同流逆序依赖识别、双窗口独立前进、CPU 服务循环独立前进、非有限归约门与 host 有界等待，并按操作记录 host 内存系统 RMW 能力位；本机 RX 6800 实测 host fetch_add 产生错误值而 CAS 可用，恰好要求协议只用单写者 load/store 字。同批交付的 `tq-clock-rate` 在每张可见设备上实测 `clock64()` 速率，把探针设备侧自旋预算的 2.3 GHz 定标从一次性手工测量变成可复核的工具（它与探针共用同一份映射 host 内存合同：默认 pinned 本机实测 flags=0 却可被核写入，属平台意外，工具逐卡校验 Mapped 标志）。五卡用例的逐 rank 索引（核与 guard 布局）由单卡钉住用例 `five_rank_indexing` 在每张开发机上持续验证，跨设备可见性仍只在真实五卡机首次运行。五卡用例只在恰好五张设备时运行，运维人员在目标机上重跑同一程序即可取得新报告，归档前用校验器做机器检查。

## 多请求与阶段内存

准备 UTF-8 文本 `prompts.txt`，每行一个原始模型提示。需要聊天模板时，由调用者先形成正确的提示文本。

```sh
build-clean-vulkan/install/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt \
  --ctx-size 2048 --total-kv 8192 --predict 64 \
  --inject-after 2 --gpu-layers 99 --json
```

`--inject-after 2` 让前两个请求先执行两批 Decode，再让后续提示进入待准入队列。B 选择一个 Prefill 并保持它到完成；这期间 C 暂停其他请求生成，然后从原位置继续。JSON Lines 记录真正提交的 token、位置、批次、阶段变化和结果，不能用一次参数设置冒充已经发生的调度。

| 控制 | 实际含义 | 默认与限制 |
|---|---|---|
| `--ctx-size` | 每请求逻辑窗口 | 不因扩大物理 KV 自动扩大逻辑窗口 |
| `--total-kv` | 同一 context 的统一物理 KV token 容量 | 由 A 按 256 对齐；必须覆盖一个逻辑窗口；不等于设备自动容量探测 |
| `--parallel` | 同时驻留的请求数，也是 A 的序列槽位数上限 | 默认取请求数与 batch-size 的较小值；队列可长于槽位数 |
| `--phase-memory` | Prefill→Decode 时同步、丢弃可重建图并释放计算分配器；下一批按实际图重新分配 | C 默认开启；保留模型、KV / recurrent、输出和 sampler 状态 |
| `--no-phase-memory` | 使用相同调度，但保留原计算工作区 | 用于真实输出与分配量对照 |

C 未指定物理容量时，使用逻辑窗口与最多 `--parallel` 个最大请求需求之和中的较大值。每次准入先预留 `prompt_tokens + predict - 1`；token 或槽位不足时按输入顺序等待。完成后先同步并成功移除 A 的序列状态，再释放 B 的预算。请求 ID 保持稳定，A 的槽位带 generation 重用。这个计算不探测显存，也不执行驱逐或容量不足重试。此处的 batch C 明确拒绝纯 recurrent、DeepSeek4、共享 CUDA KV-stream arena 等不兼容配置；新的独立 recurrent 槽位配置由下述 RERoT C 消费，没有据此扩大 batch C 的资格。

释放后的 `llama_memory_compute_bytes()` 是该 context 当前计算分配器拥有的字节数，不是进程峰值显存或设备可用显存。扩大物理 KV 也可能扩大注意力工作区。阶段释放会带来重新分配成本，吞吐收益需要具体负载评估。完整合同和真实对照见 [batch-compose/README.md](batch-compose/README.md)。

### 增量文本与 stop

```sh
build-clean-fork/install/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt --parallel 2 \
  --ctx-size 2048 --predict 64 --stop '<end>' --stop '###' --json --stream
```

`--stream` 要求 `--json`，为每条请求输出带 `request_id`、`offset_bytes` 和 `finished` 的 `text` 事件。每个增量都是合法 UTF-8，连续字节游标允许调用者按请求拼接；拼接结果等于最终 `result.text`。成功请求恰有一个终止文本事件，输出失败会终止程序。CLI 输出不是 HTTP SSE，也不提供网络送达确认或重试恢复。

`--stop` 可重复，匹配区分大小写的非空 UTF-8 字符串。先完成的匹配获选；同一字节位置完成时选最长，相同字符串按配置顺序。B 暂存可能成为 stop 的后缀，匹配后抑制 stop 及其后的文本。`tokens` 保留已采样的非 EOG token，`--ignore-eog` 时也保留 EOG；文本抑制不回删 token 计数。匹配确定后，C 不再提交该请求的 decode，并在消费本批全部输出后释放完成请求的状态和预算。

不完整 UTF-8 可跨块等待；非法字节和最终未完成的每个字节各替换为一个 U+FFFD。停止与输出规则不随 token piece 的切分而改变。只有驻留请求分配匹配状态。完整边界见 [TQText](text/README.md) 和 [batch C](batch-compose/README.md)。这是对有用文本合同的重新实现，没有宣称恢复旧非流式扫描 fastpath 的速度收益，也未迁入多 choice usage、lane parser 或完整 SSE 生命周期。

### 原版 HTTP server

```sh
build-clean-vulkan/upstream/bin/llama-server \
  --model /path/to/model.gguf -ngl 99 \
  --ctx-size 2048 --total-kv 8192 -np 4 \
  --prefill-priority --phase-memory
```

server 的 Prefill 优先默认开启，阶段内存释放默认关闭。`--no-prefill-priority` 恢复上游混合调度；阶段释放要求优先策略开启，并拒绝共享 CUDA phase arena。优先顺序是当前提示保持加槽位顺序，持续新提示可能使 Decode 饥饿；没有偷偷加入 FIFO 或公平性保证。

server 继续使用自己的采样入口；compact sampling 的入口是下面的 `llama-compose`。此前真实 HTTP 测试覆盖三路 SSE、插入 Prefill、完整 token 预算和阶段切换，保留原测试身份与范围；它不证明 batch 文本路径或 RERoT 已通过 HTTP 验收。MTP、语法及随机采样的暂停恢复仍需要专门验证。

## RERoT：逐 token 共享白板

`tq-rerot-compose` 接受显式任务 DAG。每个 worker 是一个完整的原生 assistant 阶段，以模型的 EOG 结束；全部 worker 结束后启动新的 synthesis 阶段。当前 profile 为 `assistant-eot`，模板必须能通过公开 A renderer 形成保持共同前缀的 FRAME。

```sh
build-clean-fork/install/bin/tq-rerot-compose \
  --model /path/to/instruct.gguf \
  --prompt 'What is 17 times 19?' \
  --stage 'multiply=Compute the product. Reply with the integer only.' \
  --stage 'check=Check the product independently. Reply with the integer only.' \
  --stage 'review=Compare the calculations and give the agreed integer.' \
  --after review=multiply --after review=check \
  --synthesis 'Answer the original question with the integer only.' \
  --parallel 2 --ctx-size 8192 --total-kv 8192 \
  --worker-tokens 512 --final-tokens 512 \
  --temperature 0 --seed 0 --json
```

默认在 CPU 执行。`--devices`、`--gpu-layers` 与 `--split-mode none|layer|row` 交给 A 的原生设备放置和调度；接受这些选项不代表该设备组合已经通过验证。DAG 的边限制 child 何时开始，reader 仍能看到所有已提交的 public 工作，包括没有祖先关系的 worker。

本次 RX 6800 单卡的 `none`、`layer` 均完成实际白板生成；`row` 在原生模型加载时因 Vulkan0 不支持 split buffers 被拒绝，不做隐式回退。多设备 PP 仍需独立运行证据，详见 [VALIDATION.md](VALIDATION.md)。

| 时点 | 实际可见性 |
|---|---|
| frontier 开始 | 所有 reader 固定在同一个已提交 peer revision |
| 本轮 forward | 每个运行阶段解码一个当前 BODY token；它可见自己的因果前缀及当前 token，peer 的当前 token 尚不可见 |
| 本轮全部完成 | C 完成真实同步，B 将整个 cohort 的 public token 一次提交 |
| 紧接的下一轮 | 每个 reader 立即看到 peer 刚提交的 token；不等待整个 BODY、worker 或任务结束 |

一次 forward 采样出的后继 token，须在下一次 forward 形成实际 KV 后才能发布和显示。新阶段的 FRAME 可以有多个 token；C 将它们分成每序列每次一个 query 的波次，保留同一 peer 快照，完成全部波次后一次提交 FRAME。运行中的 BODY 始终按 token 发布。

B 给出拥有自身数据的 immutable span，分别标识逻辑 stage / run、writer 的存储位置和 reader 的虚拟位置。A 保持一个 context 和同一物理 KV owner：不需改变相位的 span 复用原生 batched KQ 分数，需要改变相位的 span 从原始 pre-RoPE Q 重建查询。随后把可见分数写回对应的物理 KV 槽，其他槽置为负无穷，对每个 query/head 作一个全局 softmax，再调用原生 batched V 乘法。这样保留原生 KV 布局与物理归约顺序；V 不再按 reader gather。历史 KV 不因 reader 不同而持久复制一份，独立 sampler 与 recurrent / convolution lineage 按 stage 隔离。该 helper 的已测范围见下表，执行结构本身不构成任意形状按位一致或性能收益证明。

若有 `W` 个 worker、最多 `P` 个活跃 pen，C 使用 `W + 2` 个逻辑序列身份和 `P + 1` 个 recurrent 物理槽。额外一槽保存只 Prefill 一次、此后不再推进的共同 base。阶段结束只释放它的 recurrent 状态，public attention KV 保留给后续 reader；新阶段从 base 建立自己的 lineage。容量按整个 episode 保守预检，尚无驱逐、持久恢复或失败重试。

TQRelay 在这里登记的是 **一个真实 A 提交域**，只有 `llama_synchronize` 返回后才记为完成；pen 数或 batch 行数不是 TP rank 数。失败会排空真实工作、终止未发布的 B frontier 并停止 context，不承诺回滚已经写入的设备状态。达到 token 上限而没有原生 source end 是未完成错误，不能伪造成功终结。

| 资格层次 | 当前记录 |
|---|---|
| 独立 B 状态与数学 | Release、UBSan、ASan 各 4/4 CTest 通过；包含四节点 DAG 穷举、事务反例和独立 F64 attention oracle |
| 真实 A reader 数学 | 最新 helper 已通过 16 例 CPU 独立 F64 logits oracle：单层合法 GGUF、零 FFN、Llama NORMAL / Qwen2 NEOX、4 Q / 2 KV heads、RoPE amplitude 1 / 1.375、行重排及同一 frozen cohort 的串行对照。此前 8 例、最大绝对误差 `7.72386e-7` 的回执保留原实现身份。GPU 独立 F64 oracle 仍未通过，其 ordinary 校准先失败 |
| 真实 F32 reader 状态 | LFM2.5 的最终 33 项严格门通过：31 个全 logits 等价组、2 个 peer 因果效应组。等价组最大绝对 logits 差为 CPU 0、Vulkan `0.0000133514404296875`，包含新增 reversed-serial / visible-serial 同形状对照；最终分组和回执见 [VALIDATION.md](VALIDATION.md)。关系与状态门的通过不代替 GPU 独立数学资格 |
| 真实 LFM2.5 完整生成 | 正在验证；不能由小模型 oracle 或历史 batch / corpus 测试推出通过 |
| 多 GPU PP、MMVQ 性能与其他优化组合 | 多 GPU PP 尚未实际运行；没有 RERoT 性能收益结论。LateBind、MTP、整模型 TP5 provider 和 Turbo / 量化 KV reader 路线尚未接入 |

A 的 oracle 未取得 partial rotary / GPTNeoX 资格；B 中的 partial rotary 数学对照不能扩大 A 的模型支持声明。本次 reader profile 只发布 **resident F32 K / F32 V**，C 显式配置 F32 并关闭 FA，A capability 拒绝 F16 或混合 KV 类型的 reader。普通 A 的 F16 解码支持不变。F16 reader 的 CPU 重排 / restore 对照最大误差 `0.11799526`、Vulkan 行重排误差 `0.0003576` 未过门，保留为 E23 的独立精度策略重写项。旧自动 planner、reasoning-end / tool 路由与模板启发式也保持独立候选。完整合同见 [RERoT B](rerot/README.md)、[RERoT C](rerot-compose/README.md)；最终命令、运行身份与资格以 [VALIDATION.md](VALIDATION.md) 为准。

## 精确 compact 采样

```sh
build-clean-fork/install/bin/llama-compose \
  -m /path/to/model.gguf -p 'The capital of France is' \
  -n 64 -c 2048 --temp 0 --json
```

固定采样链为 top-k → top-p → min-p → temperature → dist。仅在 `temperature ≤ 0`、`2 ≤ top-k ≤ 128`、存在唯一有限最大值时启用 compact。并列、正温或不支持的 k 使用完整候选行；NaN、正无穷和没有有限候选明确报错，负无穷 mask 合法。

C 保留 A 的 distribution sampler 和随机状态。`--no-compact` 用于完整行对照；JSON 返回精确 token IDs、文本、停止原因和实际路由计数。程序不提供原版 server 的所有语法、惩罚、模板、并发或投机参数。

## 请求 corpus 与前缀评分

```sh
build-clean-fork/install/bin/llama-corpus build \
  -m /path/to/model.gguf --input prompts.txt --corpus request-corpus

build-clean-fork/install/bin/llama-corpus score \
  -m /path/to/model.gguf --corpus request-corpus -c 2048

build-clean-fork/install/bin/llama-corpus score \
  -m /path/to/model.gguf --corpus request-corpus -c 2048 --reference
```

B 兼容旧生产 `nodes-000000.bin`、`requests-000000.bin` V1 格式：nodes 起始 ID 为 1、requests 为 0；保留请求 kind、64 位 timestamp、空请求、重复、前缀终点和原有顺序。此前已用归档中未修改的生产 writer / reader 完成真实字节往返验证。第一版迁移及旧 `nanbeige-audit.py` 手写的不规范 start ID / kind 数据应重建；不会静默解释成有效生产数据。

`build` 用 A 的 tokenizer，目标目录必须不存在。`score` 使用录制的 token IDs，从第二个 token 开始计算条件负对数似然，保留重复请求的权重。默认保存分叉点的 A 序列状态，每个唯一节点 decode 一次；`--reference` 独立重放每个请求。边评分在进入子分支前完成，因为快照不保证恢复 logits。全部请求都没有可计分边时明确失败。

V1 没有 tokenizer 身份，须使用相同 tokenizer；当前只支持规范单分片。两文件保存要求独占同步访问，未提供并发写入或崩溃事务。前缀评分没有承诺 MoE 激活校准、RERoT 多 reader 共享可见性或批量吞吐优化。

## 校准统计与 imatrix

已有 WXQCAL03 数据可以通过独立 C 转换为 A 的 GGUF imatrix：

```sh
build-clean-fork/install/bin/tq-imatrix-compose \
  --input calibration.bin --output imatrix.gguf \
  --dataset 'selected-calibration-range'
```

B 逐层读取、验证统计和精确文件结尾；C 保留测量得到的平方和与专家计数，让 A 的 loader 完成每专家归一化。计数须为精确 F32 整数且不超过 INT32_MAX；非有限值、负平方和、F32 溢出或非零值在 A 归一化后下溢为零均拒绝。默认要求完整标志、正的声明总量以及 `decoded == total`，并拒绝未观测专家。

只有明确接受范围不完整和估计填补时才使用以下策略：

```sh
build-clean-fork/install/bin/tq-imatrix-compose \
  --input calibration.bin --output imatrix-estimated.gguf \
  --dataset 'explicitly-accepted-selected-range' \
  --allow-partial --unseen global-mean
```

两个选项彼此独立：`--allow-partial` 保留原计数与所有范围原因，包括旧 writer 最后一批超过声明 total；`--unseen global-mean` 用已观测专家的总平方和 / 总计数估计缺失行。输出记录填补的专家 ID，明确这些行不是测量结果。二者都不绕过结构、数值或容量限制。

输出必须是新路径。默认限制为每层 256 MiB、总输入 16 GiB、最终输出 2 GiB；C 用私有 payload spool 暂存，全部验证后无覆盖发布。转换不加载模型或 GPU，也不验证数据与模型 / tokenizer 的身份匹配。保留原输入、模型和语料的摘要以及采集命令；数据集标签不能代替它们。实际 callback 激活采集、trie 复用和检查点写入仍是独立候选。

2026-10-18 的格式互操作使用与旧提交完全相同的 writer 生成生产 fixture，再由新 B/C 导出；实际 `llama-quantize` loader 读取 12 个 tensors，组成 6 个 importance entries、9 chunks。故意缺失模型的后续阶段以 exit 1 结束，证明范围是 loader 消费，不是完整模型量化或质量评测。完整格式、发布限制与原 writer 回执步骤见 [TQCalibration](calibration/README.md)、[imatrix C](imatrix-compose/README.md) 和 [VALIDATION.md](VALIDATION.md)。

## 独立诊断工具

采集器只调用公开 A 接口，用文件中的 token 逐步驱动模型，不让采样改变输入：

```sh
build-clean-fork/install/bin/tq-logits-compose \
  --model /path/to/model.gguf --prompt-file input.txt \
  --output logits.bin --ctx-size 2048
```

默认每个输入 token 后保存一行完整 logits；`--prefill N` 保存前缀最后一行，再记录后续逐 token 行。输出逐行检查有限值、预先检查完整尺寸，完成后无覆盖地发布新文件；存在的文件和 symlink 均拒绝。完整合同、文件身份和失败状态见 [logits-compose/README.md](logits-compose/README.md)。

[tools/compare-logits.py](tools/compare-logits.py) 使用 Python 标准库比较兼容 `TP5LOG2` 的 logits 文件，无私有运行时依赖。它分别报告 bitwise、数值容差与 argmax，一并检查输入 token 身份、尺寸、截断、非有限值和输出覆盖风险。保留格式不代表引入 TP5；参见 [tools/README.md](tools/README.md)。

## 验证与独立交付

```sh
python3 fork/check-upstream.py
python3 fork/tests/test-upstream-boundary.py
python3 fork/tests/isolation.py \
  --output build-isolation --jobs 4 --model /absolute/path/to/model.gguf
```

默认隔离流程从固定 A 提交导出源码，只应用验证通过的登记文件；A 构建时不含 B/C。七个 B 与六个 C 分别从各自源码副本构建，C 只找安装包。移除该流程创建的源码、构建副本和原安装前缀后，在搬迁后的安装包上重新编译消费者并运行程序；不会删除用户仓库。TQRerot 的独立消费者真实执行两 worker 的逐 token 发布，RERoT C 运行已安装程序的 help / CLI 检查。`--model` 另外启用既有真实 batch、streaming 和 logits 测试；独立消费者的 logits 对照限定为最终 greedy token / 最大 logit。

真实共享白板使用独立选项，避免让只适用于旧测试的模型隐式承担聊天模板与 source-end 合同：

```sh
python3 fork/tests/isolation.py \
  --output build-rerot-isolation --jobs 4 \
  --rerot-model /absolute/path/to/instruct.gguf \
  --rerot-ubatch-sizes 128,1 --rerot-phase-modes release,retain
```

该 harness 保存模型 / binary 身份、命令、原始日志与 frontier 检查回执，验证 peer 的已提交前缀、逐 token 增长及实际完成事件。可显式选择模板、原生设备与 split mode；真正的多设备验证另需 `--rerot-require-multi-device` 和实际设备日志。事件合同检查不能替代全词表数值 oracle、设备 kernel 命中或性能对照。未提供 `--rerot-model` 的流程会明确记为跳过真实 RERoT 模型测试。

2026-10-18 的五 B / 五 C 交付记录为完整 CPU archive/isolation 的 62 条命令通过，TQText 与 TQCalibration 的独立测试通过 ASan 和 UBSan，另有原 writer / 实际 imatrix loader 互操作。该批次最终 Vulkan 源码上的同形状流式模型对照与 7 项公开 A 接缝 CTest 通过，其中包含 5 个 scratch 场景、2 个输入别名场景和 16 项 SET_ROWS 字节比较。完整 Vulkan 算子门仍有失败记录。这些历史结果不构成后来增加 TQRelay / RERoT 后的完整重跑；当前构建范围与最终发布门见 [VALIDATION.md](VALIDATION.md)。自有 CI 已配置，尚未推送或在 GitHub 执行。

各模块也可独立构建，例如：

```sh
cmake -S fork/scheduling -B build-scheduling \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build-scheduling
ctest --test-dir build-scheduling --output-on-failure
cmake --install build-scheduling
```

独立 B 的安装目标为 `TQSampling::sampling`、`TQCorpus::corpus`、`TQScheduling::scheduling`、`TQText::text`、`TQCalibration::calibration`、`TQRelay::relay` 和 `TQRerot::rerot`。采样 C 另导出 `TQCompose::compose`；其他五个 C 交付程序。imatrix C 只链接 A 的 `ggml::ggml-base`，配置时仍需要完整 A 安装，因为其 `ggml` package 会查找已启用的 backend。上游公开接口不承诺跨固定版本的二进制 ABI 稳定；本次新增 context 参数也要求安装头、A 库和消费者一起重建。

### 数值对照的必要条件

此前 LFM2.5 的实际对照已复现批形状变化导致的 token 差异；当时的完整 logits 也显示分块 Prefill 与逐 token Prefill 的数值差异。不能把它归为已证明无害的舍入误差，或承诺不同 batch / 后端输出一致。准入与阶段释放以相同形状对照、真实输入行和状态复用为验收条件；跨形状观察保留原身份单独记录。细节见 [RELEASE-NOTES.md](RELEASE-NOTES.md)。

## 更新与回查

`upstream.json` 固定本次刚合并的 master。该仓库默认 HEAD 另指 dev，本次没有改换基线。更新时在新上游上重放本分支的独立提交，核对各 A 小补丁的需要、更新 pin 与审查过的摘要，再运行边界和真实验证。

旧 527 个 fork 提交没有重新作为祖先带入干净分支；原 `master` 工作区和 `archive/pre-clean-20261018` 保留旧实现与实验资料。详细差异范围、代码阅读深度、候选和未验证项均可从 [AUDIT.md](AUDIT.md) 追溯。
