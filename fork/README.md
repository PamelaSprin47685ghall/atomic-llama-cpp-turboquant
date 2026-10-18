# 干净的 TurboQuant fork

这里是自有功能的维护入口。A 固定在 AtomicBot-ai 上游 master 的 `cda0f9256665845cd9d6f3a0176880883f653a3d`；自有库 B 和编排程序 C 放在 `fork/`。少数必须由 A 自己完成的内存操作，以及已经有失败用例的正确性修复，以显式小补丁保留。

在阶段计算工作区、独立物理 KV 容量和 Prefill 优先的基础上，本轮补上 **有界 token / 槽位准入、真实排队与槽位复用、公开 API logits 采集**，并以独立回归修复 Meta 异步偏移。A 的代码修订保持为 **17 个文件，328 行新增、43 行删除**；另有 49 行自有 CI 工作流。全部 18 个仓库层例外都由 [upstream-exceptions.json](upstream-exceptions.json) 固定内容、消费者、理由和退出条件。

完整清单与取舍见 [AUDIT.md](AUDIT.md)、[MIGRATION.md](MIGRATION.md)，实际运行记录见 [VALIDATION.md](VALIDATION.md)。未迁入的有用能力仍是保留候选；旧实现与证据保存在 `archive/pre-clean-20261018`。

## A / B / C 的职责

| 参与者 | 完整职责 | 声明依赖 |
|---|---|---|
| A：固定上游与登记的小补丁 | 模型、tokenizer、推理、采样、KV、后端和原版应用 | 原上游依赖 |
| B：TQSampling | 从数值数组返回拥有自身数据的有限 top-k 候选 | C++17 标准库 |
| B：TQCorpus | 请求 token trie、生产 V1 格式、请求种类及时间戳 | C++17 标准库 |
| B：TQScheduling | 独立的 token / 槽位预算，以及保持当前 Prefill 的候选选择；两对象由 C 组合 | C++17 标准库 |
| C：compose | 用 B 选候选，调用 A 的真实采样链和模型生成 | 已安装 llama、TQSampling |
| C：corpus-compose | 用 B 读写请求，编排 A 的 tokenizer、decode 和序列快照 | 已安装 llama、TQCorpus |
| C：batch-compose | FIFO 准入、请求身份、真实序列槽位、模型批次与阶段释放 | 已安装 llama、TQScheduling |
| C：logits-compose | 录制真实输入 token 和原始 logits，完整后发布新文件 | 已安装 llama |

A 与 B 不包含、调用或注入彼此的服务；各 B 也不依赖兄弟库。C 只消费安装后的公开接口。没有构建时改写源码、链接符号拦截或私有头文件桥接。原版 `llama-server` 的小范围调度修改是明确的 A 例外，尚未拆成完整独立 HTTP 编排层；详见 [UPSTREAM-EXCEPTIONS.md](UPSTREAM-EXCEPTIONS.md)。

## 构建

需要 Git、Python 3.12 或更新版本、CMake、C++17 编译器，以及所选 A 后端需要的工具链。完整交付目前在 Linux 实测；logits 采集器需要 POSIX `openat` / `linkat`。从仓库根执行：

```sh
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
python3 fork/build.py --backend vulkan --tools --build-dir build-clean-vulkan --jobs 4
```

脚本先检查 A，再单独构建和安装 A；随后分别构建、测试和安装三个 B，再分别构建四个 C。产物位于相应构建目录的 `install/`。默认组件测试不需要模型；Vulkan WHT 测试需要真实 Vulkan 设备。

`--tools` 额外构建 `llama-server`、`llama-bench` 及 A 在 tools 模式下必须安装的 `mtmd`；工具可执行文件位于 `upstream/bin/`。脚本接受 `cuda`、`hip`，本次实际资格为 CPU 与 Vulkan。`--no-test` 只用于明确跳过测试的构建，不构成验证记录。

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

### 原版 HTTP server

```sh
build-clean-vulkan/upstream/bin/llama-server \
  --model /path/to/model.gguf -ngl 99 \
  --ctx-size 2048 --total-kv 8192 -np 4 \
  --prefill-priority --phase-memory
```

server 的 Prefill 优先默认开启，阶段内存释放默认关闭。`--no-prefill-priority` 恢复上游混合调度；阶段释放要求优先策略开启，并拒绝共享 CUDA phase arena。优先顺序是当前提示保持加槽位顺序，持续新提示可能使 Decode 饥饿；没有偷偷加入 FIFO 或公平性保证。

server 继续使用自己的采样入口；compact sampling 的入口是下面的 `llama-compose`。本次真实 HTTP 测试覆盖三路 SSE、插入 Prefill、完整 token 预算和阶段切换；MTP、语法及随机采样的暂停恢复仍需要专门验证。

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

B 兼容旧生产 `nodes-000000.bin`、`requests-000000.bin` V1 格式：nodes 起始 ID 为 1、requests 为 0；保留请求 kind、64 位 timestamp、空请求、重复、前缀终点和原有顺序。已用归档中未修改的生产 writer / reader 完成真实字节往返验证。第一版迁移及旧 `nanbeige-audit.py` 手写的不规范 start ID / kind 数据应重建；不会静默解释成有效生产数据。

`build` 用 A 的 tokenizer，目标目录必须不存在。`score` 使用录制的 token IDs，从第二个 token 开始计算条件负对数似然，保留重复请求的权重。默认保存分叉点的 A 序列状态，每个唯一节点 decode 一次；`--reference` 独立重放每个请求。边评分在进入子分支前完成，因为快照不保证恢复 logits。全部请求都没有可计分边时明确失败。

V1 没有 tokenizer 身份，须使用相同 tokenizer；当前只支持规范单分片。两文件保存要求独占同步访问，未提供并发写入或崩溃事务。前缀评分没有承诺 MoE 激活校准、RERoT 多 reader 共享可见性或批量吞吐优化。

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

默认隔离流程从固定 A 提交导出源码，只应用验证通过的登记文件；A 构建时不含 B/C。三个 B 与四个 C 分别从各自源码副本构建，C 只找安装包。移除源码和构建副本、搬迁安装前缀后，重新编译消费者，并运行已安装程序和可选真实模型测试，包括 logits 采集，以及独立消费者对最终 greedy token / 最大 logit 的对照。实际命令和结果见 [VALIDATION.md](VALIDATION.md)。自有 CI 已配置无模型的完整流程，尚未推送或在 GitHub 执行；本地真实模型和 GPU 资格另行记录。

各模块也可独立构建，例如：

```sh
cmake -S fork/scheduling -B build-scheduling \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build-scheduling
ctest --test-dir build-scheduling --output-on-failure
cmake --install build-scheduling
```

独立 B 包为 `TQSampling`、`TQCorpus`、`TQScheduling`，目标分别为 `TQSampling::sampling`、`TQCorpus::corpus`、`TQScheduling::scheduling`。采样 C 另导出 `TQCompose::compose`；其他三个 C 交付程序。上游公开结构体扩展不承诺二进制 ABI 稳定，更新 A 时须一起重建消费者。

### 数值对照的必要条件

LFM2.5 的实际对照在原编排程序中已经复现批形状变化导致的 token 差异；本轮完整 logits 也显示分块 Prefill 与逐 token Prefill 的数值差异。不能把它归为已证明无害的舍入误差，或承诺不同 batch / 后端输出一致。准入与阶段释放以相同形状对照、真实输入行和状态复用为验收条件；跨形状观察单独保留。细节见 [RELEASE-NOTES.md](RELEASE-NOTES.md)。

## 更新与回查

`upstream.json` 固定本次刚合并的 master。该仓库默认 HEAD 另指 dev，本次没有改换基线。更新时在新上游上重放本分支的独立提交，核对各 A 小补丁的需要、更新 pin 与审查过的摘要，再运行边界和真实验证。

旧 527 个 fork 提交没有重新作为祖先带入干净分支；原 `master` 工作区和 `archive/pre-clean-20261018` 保留旧实现与实验资料。详细差异范围、代码阅读深度、候选和未验证项均可从 [AUDIT.md](AUDIT.md) 追溯。
