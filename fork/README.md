# 干净的 TurboQuant fork

本目录是维护入口。仓库其余部分保持所选 upstream master 原样；自有代码全部在 fork/。

当前保留两项独立能力：

- 精确的有限 top-k 候选提取，以及调用真实 llama 模型和采样链的生成程序。
- Wanxiangqi 请求 token trie 的 V1 格式读写，以及调用真实 llama 序列状态接口的前缀复用评分程序。

完整取舍、证据来源和未迁移范围见 [MIGRATION.md](MIGRATION.md)。这不是旧 fork 的全部功能等价替换。TP5、RERoT、FlashPrefill、XKV-SR、direct-hidden 等扩展仍在归档分支；上游自身已有的能力继续由上游提供。

## 结构

| 参与者 | 完整职责 | 依赖 |
|---|---|---|
| A：仓库上游部分 | 模型加载、tokenizer、推理、采样、序列状态 | 原上游依赖 |
| B：sampling | 从数值数组返回拥有自身数据的有限 top-k 候选 | C++17 标准库 |
| B：corpus | 维护、验证和保存请求 token trie | C++17 标准库 |
| C：compose | 调用 B 选择候选，再运行 A 的真实采样链；执行模型生成 | 已安装 llama、TQSampling |
| C：corpus-compose | 调用 B 读取请求，安排 A 的 decode、状态保存和恢复 | 已安装 llama、TQCorpus |

A 不认识 B 或 C。B 不包含 llama、ggml 或其他兄弟库的头文件。C 使用已安装的公开 API；没有构建时改写 A 源码、链接符号拦截或私有头文件桥接。维护规则见 [AGENTS.md](AGENTS.md)。

## 构建

需要 Git、Python 3.12 或更新版本、CMake、C++17 编译器，以及所选上游后端需要的工具链。隔离测试使用 Python 的 tar 数据过滤器。

从仓库根执行：

~~~sh
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
~~~

脚本先单独构建和安装 A，再从其安装包构建 B、C，执行测试并安装。产物在 build-clean-fork/install。没有模型文件也能运行这组组件测试。

选择 Vulkan：

~~~sh
python3 fork/build.py --backend vulkan --build-dir build-clean-vulkan --jobs 4
~~~

脚本也接受 cuda、hip；实际验证范围见 [VALIDATION.md](VALIDATION.md)。用 --tools 额外构建原上游 llama-server 和 llama-bench，产物在构建目录的 upstream/bin；它们保持原上游行为，未自动接入本目录的采样优化。

## 生成

~~~sh
build-clean-fork/install/bin/llama-compose \
  -m /path/to/model.gguf -p 'The capital of France is' \
  -n 64 -c 2048 --temp 0
~~~

-p 接受原始模型输入；需要聊天模板的模型应传入格式正确的提示。默认 CPU、4 线程。Vulkan 等后端可用 -ngl 99 请求卸载层；实际设备选择由 A 决定。

采样顺序固定为 top-k → top-p → min-p → temperature → dist。仅在 temperature ≤ 0、2 ≤ top-k ≤ 128、logits 中存在唯一有限最大值时启用 compact。最大值并列、k 不支持或正温时走完整候选行。NaN、正无穷和没有有限候选的输入明确报错；负无穷屏蔽值允许存在。

C 保留 A 的真实 distribution sampler 和随机状态。它不是只返回 argmax 的替代采样器。--no-compact 可显式选择完整行，对拍输出。--json 返回 token IDs、生成文本、停止原因和实际路由计数；若 token 上限截断 UTF-8 字符，文本使用 U+FFFD，token IDs 保持精确。

这个程序不宣称实现上游 server 的全部参数、语法、惩罚、聊天模板、并发服务或投机解码。需要这些功能时，使用原上游程序或新增拥有相应协作职责的 C。

## 录制和评分 token 请求

准备 UTF-8 文本，每行一个原始提示：

~~~sh
build-clean-fork/install/bin/llama-corpus build \
  -m /path/to/model.gguf --input prompts.txt --corpus request-corpus

build-clean-fork/install/bin/llama-corpus score \
  -m /path/to/model.gguf --corpus request-corpus -c 2048

build-clean-fork/install/bin/llama-corpus score \
  -m /path/to/model.gguf --corpus request-corpus -c 2048 --reference
~~~

build 用 A 的 tokenizer 生成 token IDs，再交给 B。目标目录必须不存在，避免覆盖已录制请求。score 直接读取 token IDs，不重新 tokenize；它计算每个请求从第二个 token 开始的负对数似然，重复请求保留原有权重。

默认评分在分叉点保存 A 提供的 opaque 序列状态，每个唯一节点只 decode 一次。所有后继 token 的边评分在进入子分支前计算，因为 A 的序列快照不保证恢复 logits。--reference 对每个请求清空状态并独立重放，作为正确性对照。报告中的 decoded_tokens 是实际调用 decode 的 token 数，不是推算速度。程序逐 token 运行，未作吞吐批处理优化。

兼容旧 Wanxiangqi 的 nodes-000000.bin、requests-000000.bin V1 两文件格式。V1 不包含 tokenizer 身份，必须使用与录制相同的 tokenizer；词表大小检查只能验证 token 范围。旧 reserved 捕获元数据在读取时忽略，保存时写零；token 路径、节点 ID、请求顺序和重复次数保留。它不等于旧 MoE 激活校准器，也不等于 RERoT 的共享 KV。

## 验证与安装包消费

~~~sh
python3 fork/check-upstream.py
python3 fork/tests/isolation.py --output build-isolation --jobs 4
python3 fork/tests/real-model.py \
  --model /path/to/model.gguf \
  --bin-dir build-clean-fork/install/bin \
  --output build-model-check
~~~

isolation.py 从固定 Git 提交导出没有 B、C 的 A，构建并运行 A 自己的采样测试。随后独立构建每个 B、C，移走源码副本，将安装产物搬到新前缀，再编译和运行实际消费程序。--a-prefix 可复用已验证的 A 安装包；这种运行不会重新证明 A 的独立源码构建。

各模块可以单独发布和消费：

~~~sh
cmake -S fork/sampling -B build-sampling -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build-sampling
ctest --test-dir build-sampling --output-on-failure
cmake --install build-sampling

cmake -S fork/compose -B build-compose -DCMAKE_PREFIX_PATH=/path/to/prefix
cmake --build build-compose
~~~

包名为 TQSampling、TQCorpus、TQCompose；导出目标为 TQSampling::sampling、TQCorpus::corpus、TQCompose::compose。corpus-compose 是程序交付，不伪装成另一个职责库。

## 更新上游

upstream.json 固定 master 提交。上游默认 HEAD 当前另指 dev；本次没有把用户刚合并的 master 换成 dev。

更新时先创建新 worktree，在新上游 master 上按顺序 cherry-pick 本分支的少量提交，更新 pin，然后运行上述验证。若公开 API 改变，调整 C；若不能用公开契约实现，先登记所缺语义和证据，再决定是否接受一个明确的 A 补丁。不要悄悄把私有依赖放进 B。

旧的 527 个 fork 提交没有作为祖先重新带入干净分支。它们由 archive/pre-clean-20261018 保留，原 master 工作区及用户未跟踪文件保持原状。
