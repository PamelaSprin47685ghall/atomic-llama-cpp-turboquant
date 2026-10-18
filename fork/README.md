# 干净的 TurboQuant fork

这里是自有功能的维护入口。A 固定在 AtomicBot-ai 上游 master 的 `cda0f9256665845cd9d6f3a0176880883f653a3d`；自有库 B 和编排程序 C 放在 `fork/`。少数必须由 A 自己完成的内存操作，以及已经有失败用例的正确性修复，以显式小补丁保留。

在阶段计算工作区、独立物理 KV 容量、Prefill 优先、准入队列和 logits 采集的基础上，本轮补上 **增量 UTF-8 / stop 输出、WXQCAL03 校准格式读取和 imatrix 导出**。这些能力由独立库和公开接口消费者承担。A 的实际变动以 [upstream-exceptions.json](upstream-exceptions.json) 登记的完整文件内容、消费者、理由和退出条件为准。

完整清单与取舍见 [AUDIT.md](AUDIT.md)、[MIGRATION.md](MIGRATION.md)，实际运行记录见 [VALIDATION.md](VALIDATION.md)。未迁入的有用能力仍是保留候选；旧实现与证据保存在 `archive/pre-clean-20261018`。

## A / B / C 的职责

| 参与者 | 完整职责 | 声明依赖 |
|---|---|---|
| A：固定上游与登记的小补丁 | 模型、tokenizer、推理、采样、KV、后端和原版应用 | 原上游依赖 |
| B：TQSampling | 从数值数组返回拥有自身数据的有限 top-k 候选 | C++17 标准库 |
| B：TQCorpus | 请求 token trie、生产 V1 格式、请求种类及时间戳 | C++17 标准库 |
| B：TQScheduling | 独立的 token / 槽位预算，以及保持当前 Prefill 的候选选择；两对象由 C 组合 | C++17 标准库 |
| B：TQText | 增量 UTF-8、跨块 stop 匹配、拥有自身文本和字节游标的结果 | C++17 标准库 |
| B：TQCalibration | 有界读取 WXQCAL03，验证格式、统计人口和数值，逐层返回拥有自身数据的结果 | C++17 标准库 |
| C：compose | 用 B 选候选，调用 A 的真实采样链和模型生成 | 已安装 llama、TQSampling |
| C：corpus-compose | 用 B 读写请求，编排 A 的 tokenizer、decode 和序列快照 | 已安装 llama、TQCorpus |
| C：batch-compose | FIFO 准入、真实序列槽位、模型批次、阶段释放及 stop / 流式输出生命周期 | 已安装 llama、TQScheduling、TQText |
| C：logits-compose | 录制真实输入 token 和原始 logits，完整后发布新文件 | 已安装 llama |
| C：imatrix-compose | 明确选择 partial / unseen 策略，用有界磁盘暂存导出 A 可消费的 GGUF imatrix | 已安装 ggml 的 `ggml::ggml-base`、TQCalibration |

A 与 B 不包含、调用或注入彼此的服务；各 B 也不依赖兄弟库。C 只消费安装后的公开接口。没有构建时改写源码、链接符号拦截或私有头文件桥接。原版 `llama-server` 的小范围调度修改是明确的 A 例外，尚未拆成完整独立 HTTP 编排层；详见 [UPSTREAM-EXCEPTIONS.md](UPSTREAM-EXCEPTIONS.md)。

## 构建

需要 Git、Python 3.12 或更新版本、CMake、C++17 编译器，以及所选 A 后端需要的工具链。完整交付目前在 Linux 实测；logits 和 imatrix 文件发布需要 POSIX `openat` / `linkat`，imatrix 输出限定为小端系统。转换本身不需要 Python 或 NumPy，Python 用于构建入口和测试。从仓库根执行：

```sh
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
python3 fork/build.py --backend vulkan --tools --build-dir build-clean-vulkan --jobs 4
```

脚本先检查 A，再单独构建和安装 A；随后分别构建、测试和安装五个 B，再分别构建五个 C。产物位于相应构建目录的 `install/`。默认组件测试不需要模型；启用的 Vulkan 接缝测试需要真实 Vulkan 设备。

`--tools` 额外构建 `llama-server`、`llama-bench` 及 A 在 tools 模式下必须安装的 `mtmd`；工具可执行文件位于 `upstream/bin/`。脚本也接受 `cuda`、`hip`，可选后端不等于已经取得运行资格。本轮 CPU 独立交付和 CPU / Vulkan 同形状流式模型对照已完成；专项回归与完整后端门槛的范围分别列在 [VALIDATION.md](VALIDATION.md)。`--no-test` 只用于明确跳过测试的构建，不构成验证记录。

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

C 未指定物理容量时，使用逻辑窗口与最多 `--parallel` 个最大请求需求之和中的较大值。每次准入先预留 `prompt_tokens + predict - 1`；token 或槽位不足时按输入顺序等待。完成后先同步并成功移除 A 的序列状态，再释放 B 的预算。请求 ID 保持稳定，A 的槽位带 generation 重用。这个计算不探测显存，也不执行驱逐或容量不足重试。纯 recurrent、DeepSeek4、共享 CUDA KV-stream arena 等不兼容配置明确拒绝；独立 recurrent 容量尚未抽取。

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

server 继续使用自己的采样入口；compact sampling 的入口是下面的 `llama-compose`。此前真实 HTTP 测试覆盖三路 SSE、插入 Prefill、完整 token 预算和阶段切换，保留原测试身份与范围；它不证明本轮新增的 batch 文本路径已经通过 HTTP 验收。MTP、语法及随机采样的暂停恢复仍需要专门验证。

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

本轮已用与旧提交完全相同的 writer 生成生产格式 fixture，再由新 B/C 导出；实际 `llama-quantize` loader 读取 12 个 tensors，组成 6 个 importance entries、9 chunks。故意缺失模型的后续阶段以 exit 1 结束，证明范围是 loader 消费，不是完整模型量化或质量评测。完整格式、发布限制与原 writer 回执步骤见 [TQCalibration](calibration/README.md)、[imatrix C](imatrix-compose/README.md) 和 [VALIDATION.md](VALIDATION.md)。

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

默认隔离流程从固定 A 提交导出源码，只应用验证通过的登记文件；A 构建时不含 B/C。五个 B 与五个 C 分别从各自源码副本构建，C 只找安装包。移除源码和构建副本、搬迁安装前缀后，重新编译消费者并运行已安装程序。`--model` 另外启用真实 batch、streaming 和 logits 测试；独立消费者的 logits 对照限定为最终 greedy token / 最大 logit。

本轮完整 CPU archive/isolation 的 62 条命令通过，TQText 与 TQCalibration 的独立测试也通过 ASan 和 UBSan。新 imatrix 的原 writer 互操作与实际 A loader 消费另外验证。最终 Vulkan 源码上的同形状流式模型对照与 7 项公开 A 接缝 CTest 通过，其中包含 5 个 scratch 场景、2 个输入别名场景和 16 项 SET_ROWS 字节比较。完整 Vulkan 算子门仍有失败记录，不能由这些专项通过推出整个后端全绿。确切命令、源码身份和结果见 [VALIDATION.md](VALIDATION.md)。自有 CI 已配置，尚未推送或在 GitHub 执行。

各模块也可独立构建，例如：

```sh
cmake -S fork/scheduling -B build-scheduling \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build-scheduling
ctest --test-dir build-scheduling --output-on-failure
cmake --install build-scheduling
```

独立 B 的安装目标为 `TQSampling::sampling`、`TQCorpus::corpus`、`TQScheduling::scheduling`、`TQText::text` 和 `TQCalibration::calibration`。采样 C 另导出 `TQCompose::compose`；其他四个 C 交付程序。imatrix C 只链接 A 的 `ggml::ggml-base`，配置时仍需要完整 A 安装，因为其 `ggml` package 会查找已启用的 backend。上游公开接口不承诺跨固定版本的二进制 ABI 稳定，更新 A 时须一起重建消费者。

### 数值对照的必要条件

此前 LFM2.5 的实际对照已复现批形状变化导致的 token 差异；当时的完整 logits 也显示分块 Prefill 与逐 token Prefill 的数值差异。不能把它归为已证明无害的舍入误差，或承诺不同 batch / 后端输出一致。准入与阶段释放以相同形状对照、真实输入行和状态复用为验收条件；跨形状观察保留原身份单独记录。细节见 [RELEASE-NOTES.md](RELEASE-NOTES.md)。

## 更新与回查

`upstream.json` 固定本次刚合并的 master。该仓库默认 HEAD 另指 dev，本次没有改换基线。更新时在新上游上重放本分支的独立提交，核对各 A 小补丁的需要、更新 pin 与审查过的摘要，再运行边界和真实验证。

旧 527 个 fork 提交没有重新作为祖先带入干净分支；原 `master` 工作区和 `archive/pre-clean-20261018` 保留旧实现与实验资料。详细差异范围、代码阅读深度、候选和未验证项均可从 [AUDIT.md](AUDIT.md) 追溯。
