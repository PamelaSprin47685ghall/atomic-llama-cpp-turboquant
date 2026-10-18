# 干净的 TurboQuant fork

这里是自有功能的维护入口。A 固定在 AtomicBot-ai 上游 master 的 `cda0f9256665845cd9d6f3a0176880883f653a3d`；自有库 B 和编排程序 C 放在 `fork/`。少数必须由 A 自己完成的内存操作，以及已经有失败用例的正确性修复，以显式小补丁保留。

本轮按小行为重新排查，补回了 **Prefill / Decode 阶段计算工作区、独立物理 KV 容量、Prefill 优先调度**，并修复旧生产 corpus 格式兼容、CPU 精度、Vulkan WHT 等问题。A 目前只有 **16 个变更文件，324 行新增、39 行删除**，由 [upstream-exceptions.json](upstream-exceptions.json) 固定每个文件的完整内容、消费者、理由和退出条件。

完整清单与取舍见 [AUDIT.md](AUDIT.md)、[MIGRATION.md](MIGRATION.md)，实际运行记录见 [VALIDATION.md](VALIDATION.md)。未迁入的有用能力仍是保留候选；旧实现与证据保存在 `archive/pre-clean-20261018`。

## A / B / C 的职责

| 参与者 | 完整职责 | 声明依赖 |
|---|---|---|
| A：固定上游与登记的小补丁 | 模型、tokenizer、推理、采样、KV、后端和原版应用 | 原上游依赖 |
| B：TQSampling | 从数值数组返回拥有自身数据的有限 top-k 候选 | C++17 标准库 |
| B：TQCorpus | 请求 token trie、生产 V1 格式、请求种类及时间戳 | C++17 标准库 |
| B：TQScheduling | 保持当前 Prefill，按槽位顺序选择下一个请求；没有待处理 Prefill 时恢复 Decode | C++17 标准库 |
| C：compose | 用 B 选候选，调用 A 的真实采样链和模型生成 | 已安装 llama、TQSampling |
| C：corpus-compose | 用 B 读写请求，编排 A 的 tokenizer、decode 和序列快照 | 已安装 llama、TQCorpus |
| C：batch-compose | 拥有请求及位置，按 B 的选择提交 A 的真实批次，控制阶段释放 | 已安装 llama、TQScheduling |

A 与 B 不包含、调用或注入彼此的服务；各 B 也不依赖兄弟库。C 只消费安装后的公开接口。没有构建时改写源码、链接符号拦截或私有头文件桥接。原版 `llama-server` 的小范围调度修改是明确的 A 例外，尚未拆成完整独立 HTTP 编排层；详见 [UPSTREAM-EXCEPTIONS.md](UPSTREAM-EXCEPTIONS.md)。

## 构建

需要 Git、Python 3.12 或更新版本、CMake、C++17 编译器，以及所选 A 后端需要的工具链。从仓库根执行：

```sh
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
python3 fork/build.py --backend vulkan --tools --build-dir build-clean-vulkan --jobs 4
```

脚本先检查 A，再单独构建和安装 A；随后分别构建、测试和安装三个 B，再分别构建三个 C。产物位于相应构建目录的 `install/`。默认组件测试不需要模型；Vulkan WHT 测试需要真实 Vulkan 设备。

`--tools` 额外构建 `llama-server` 和 `llama-bench`，可执行文件位于 `upstream/bin/`。脚本接受 `cuda`、`hip`，本次实际资格为 CPU 与 Vulkan。`--no-test` 只用于明确跳过测试的构建，不构成验证记录。

## 多请求与阶段内存

准备 UTF-8 文本 `prompts.txt`，每行一个原始模型提示。需要聊天模板时，由调用者先形成正确的提示文本。

```sh
build-clean-vulkan/install/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt \
  --ctx-size 2048 --total-kv 8192 --predict 64 \
  --inject-after 2 --gpu-layers 99 --json
```

`--inject-after 2` 让前两个请求先执行两批 Decode，再接纳后续提示。B 选择一个 Prefill 并保持它到完成；这期间 C 暂停其他请求生成，然后从原位置继续。JSON Lines 记录真正提交的 token、位置、批次、阶段变化和结果，不能用一次参数设置冒充已经发生的调度。

| 控制 | 实际含义 | 默认与限制 |
|---|---|---|
| `--ctx-size` | 每请求逻辑窗口 | 不因扩大物理 KV 自动扩大逻辑窗口 |
| `--total-kv` | 同一 context 的统一物理 KV token 容量 | 由 A 按 256 对齐；必须覆盖一个逻辑窗口；不等于设备自动容量探测 |
| `--phase-memory` | Prefill→Decode 时同步、丢弃可重建图并释放计算分配器；下一批按实际图重新分配 | C 默认开启；保留模型、KV / recurrent、输出和 sampler 状态 |
| `--no-phase-memory` | 使用相同调度，但保留原计算工作区 | 用于真实输出与分配量对照 |

C 未指定物理容量时，使用逻辑窗口与已知请求最大 token 需求总和中的较大值。这个计算不探测显存，也不执行驱逐或容量不足重试。纯 recurrent、DeepSeek4、共享 CUDA KV-stream arena 等不兼容配置明确拒绝；独立 recurrent 容量尚未抽取。

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

[tools/compare-logits.py](tools/compare-logits.py) 使用 Python 标准库比较兼容 `TP5LOG2` 的 logits 文件，无私有运行时依赖。它分别报告 bitwise、数值容差与 argmax，一并检查输入 token 身份、尺寸、截断、非有限值和输出覆盖风险。保留格式不代表引入 TP5；参见 [tools/README.md](tools/README.md)。

## 验证与独立交付

```sh
python3 fork/check-upstream.py
python3 fork/tests/test-upstream-boundary.py
python3 fork/tests/isolation.py \
  --output build-isolation --jobs 4 --model /absolute/path/to/model.gguf
```

默认隔离流程从固定 A 提交导出源码，只应用验证通过的登记文件；A 构建时不含 B/C。三个 B 与三个 C 分别从各自源码副本构建，C 只找安装包。移除源码和构建副本、搬迁安装前缀后，重新编译消费者，并运行已安装程序和可选真实模型测试。最近一次完整流程为 41 个命令全部通过。

各模块也可独立构建，例如：

```sh
cmake -S fork/scheduling -B build-scheduling \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build-scheduling
ctest --test-dir build-scheduling --output-on-failure
cmake --install build-scheduling
```

独立 B 包为 `TQSampling`、`TQCorpus`、`TQScheduling`，目标分别为 `TQSampling::sampling`、`TQCorpus::corpus`、`TQScheduling::scheduling`。采样 C 另导出 `TQCompose::compose`；其他两个 C 交付程序。上游公开结构体扩展不承诺二进制 ABI 稳定，更新 A 时须一起重建消费者。

## 更新与回查

`upstream.json` 固定本次刚合并的 master。该仓库默认 HEAD 另指 dev，本次没有改换基线。更新时在新上游上重放本分支的独立提交，核对各 A 小补丁的需要、更新 pin 与审查过的摘要，再运行边界和真实验证。

旧 527 个 fork 提交没有重新作为祖先带入干净分支；原 `master` 工作区和 `archive/pre-clean-20261018` 保留旧实现与实验资料。详细差异范围、代码阅读深度、候选和未验证项均可从 [AUDIT.md](AUDIT.md) 追溯。
