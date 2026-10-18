# 验证记录：2026-10-18

本记录区分实际执行、历史资料与未覆盖范围。A 固定为 upstream master cda0f9256665845cd9d6f3a0176880883f653a3d。

## 实际环境与模型

- Linux x86-64，GCC 16.2.1。
- CPU 路径：主构建使用 A 的 native CPU 选项；独立 A 构建显式关闭 GGML_NATIVE。
- 实际模型：LFM2.5-8B-A1B-UD-IQ4_NL.gguf。
- 模型 SHA-256：a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a。
- Vulkan 枚举到一张 AMD Radeon RX 6800，RADV/Mesa 26.2.2。未把它称作旧 TP5 的五卡环境。

日志与原始 JSON 留在当前 worktree 的 build-* 目录，未将构建产物或模型提交到 Git。

## 源码边界

~~~sh
python3 fork/check-upstream.py
git diff --numstat cda0f9256665845cd9d6f3a0176880883f653a3d -- . ':(exclude)fork/**'
~~~

结果：A 的变更路径为 0，第二条命令无输出。检查脚本同时检查未跟踪但未被 Git 忽略的文件，避免只看已提交差异。根 CMake、src、ggml、common、tools 等上游路径均未修改。

## 独立构建和交付

### A 单独成立

isolation.py 从固定 Git 提交导出 A，确认导出目录没有 fork/，然后只构建 A 和 A 自己的 test-sampling。LLAMA_BUILD_COMMON/TESTS 开启，其余应用、GPU 后端关闭。

实际完成：

- A 的 test-sampling 目标构建成功。
- A 的 test-sampling 测试通过，1/1，0.83 秒。
- A 的安装成功。
- 在开始 B 阶段前移除 A 源码副本。

记录：build-isolation-20261018/a-configure.log、a-build.log、a-test.log、a-install.log。开发过程中这一轮完整脚本在尚未加入 B 目录时停止；不能把该次总退出码写成成功。

### B、C 分别成立

随后复用上一步已经验证的 A 安装包：

~~~sh
python3 fork/tests/isolation.py \
  --a-prefix build-isolation-20261018/package-a \
  --output build-isolation-bc-20261018 --jobs 4
~~~

这一次 21 个命令全部退出 0。每个模块只从自己的源码副本构建，C 的声明依赖只通过安装包提供；每个源码副本在该模块安装后被移除。最后将所有安装包移到一个新的 delivered 前缀，并移除原安装前缀。

| 实际交付物 | 验证 |
|---|---|
| TQSampling | 独立构建、行为测试、安装通过；无 A/B 兄弟服务参与 |
| TQCorpus | 独立构建、行为测试、安装通过；无模型或 GPU 参与 |
| TQCompose | 仅安装版 A 和 TQSampling；真实 A 采样对拍及 help 通过 |
| llama-corpus | 仅安装版 A 和 TQCorpus；构建与 help 通过 |
| relocated installed consumer | find_package 查找新前缀，编译并实际调用三个库通过 |
| relocated CLI | 两个已安装程序的 help 通过；另行实际 GGUF 生成验证记录于 relocated-model.json |

记录：build-isolation-bc-20261018/commands.json、result.json 和各阶段日志。result.json 中 a_archive_has_no_b_or_c=false 表示这次复用了 A 包；A 的独立证明来自前一阶段，不是声称这一轮又构建了 A。

这里验证的是源码和交付边界，以及指定公开行为。它不是对上游全部内部依赖、全部模型和并发行为的证明。

## 组件行为

~~~sh
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
~~~

完整构建、安装、CTest 5/5 通过。其中 3 项是行为测试，2 项是 CLI help。

- B sampling：与完整排序比较；覆盖 k 边界、最大值并列、K 边界并列、负无穷 mask、NaN、正无穷、向量尾部和结果所有权；词表大小覆盖到 32,000。
- B sampling 的运行时 AVX2 版和强制 scalar 版还在 scratch 独立用 g++、-O3、-DNDEBUG 和 -Werror 编译并通过同一行为测试。
- B corpus：共享前缀、重复与前缀请求、复制/实例/输入/输出隔离、追加失败、往返、显式小端字节、现有目录保护、坏头、截断、parent/leaf/depth/token 反例。
- C sampling：2,940 次真实上游 token 比较；从贪心切为正温后保持同一随机状态；clone、reset、无效输入不改变随机状态和计数。
- 测试使用显式失败检查；Release 的 NDEBUG 不会移除检查。

记录：build-clean-fork/build.log、build-clean-fork/extension/Testing/Temporary/LastTest.log。

## 真实模型：CPU

~~~sh
python3 fork/tests/real-model.py \
  --model /path/to/LFM2.5-8B-A1B-UD-IQ4_NL.gguf \
  --bin-dir build-clean-fork/install/bin \
  --output build-model-cpu-20261018 --threads 4
~~~

实际命令中的模型路径记录在 commands.json；以上路径是可替换的使用示例。

| 检查 | 结果 |
|---|---|
| greedy，compact 对完整行 | 24/24 token IDs、文本、停止原因相同；实际 compact 24 次 |
| temperature=0.8，固定 seed=42 | 24/24 token IDs、文本、停止原因相同；两条路径都使用完整行 |
| 6 个请求，含重复、共享前缀、长短分支与前缀终点 | 6 个请求全部保留；重复请求按原权重计分 |
| 前缀重放 decode token 数 | 44；等于不含隐式根的实际节点数 |
| 完整请求重放 decode token 数 | 120；等于各请求 token 数之和 |
| 整体 mean NLL 差值 | 1.7763568394002505e-15 |
| 最大逐请求 mean NLL 差值 | 2.0301221021717147e-15 |
| 反转请求顺序后的最大逐请求差值 | 2.0301221021717147e-15 |
| 预先设定误差门 | 1e-5，没有因结果放宽 |
| 非法上下文、重复写入 | 明确失败；写入拒绝前后已有 corpus 文件 SHA-256 相同 |

均值约为 5.21977764667728；它是本测试语料给定首 token 后的条件负对数似然，不是模型质量基准。逐请求比较与反转顺序用于排查状态污染，不能只靠整体均值抵消误差。

记录：build-model-cpu-20261018/result.json、commands.json，以及每次程序调用的 stdout/stderr。

## 真实模型与工具：Vulkan

~~~sh
python3 fork/build.py --backend vulkan --tools --build-dir build-clean-vulkan --jobs 4
python3 fork/tests/real-model.py \
  --model /path/to/LFM2.5-8B-A1B-UD-IQ4_NL.gguf \
  --bin-dir build-clean-vulkan/install/bin \
  --output build-model-vulkan-20261018 --threads 4 --gpu-layers 99
~~~

两条命令成功完成。Vulkan 构建中的组件 CTest 5/5 通过，原上游 llama-server、llama-bench 构建成功，版本/help 检查退出 0。server 报告 b10298-2.0.0、commit cda0f9256。上游在 GCC 16 下仍产生 mtmd 枚举转换、u8path 等编译警告；构建正常退出，没有因此修改 A。

实际日志确认使用 RX 6800 的 Vulkan0，25/25 层卸载到 GPU，模型 buffer 4149.06 MiB，KV 与 recurrent buffer 均位于 Vulkan0。验收脚本还检查七次实际运行日志中存在正数层卸载，避免把请求的 --gpu-layers 参数当成执行证据；该检查已另行对本次原始日志执行，通过值为 25。

| 检查 | 结果 |
|---|---|
| greedy，compact 对完整行 | 24/24 token、文本与停止原因相同；compact 实际执行 24 次 |
| temperature=0.8，固定 seed=42 | 24/24 token、文本与停止原因相同；使用完整行 |
| 前缀与完整重放 | 同样保留 6 个请求，分别实际 decode 44、120 个 token |
| 整体 mean NLL 差值 | 0 |
| 最大逐请求与反转顺序后的差值 | 均为 9.473903143468002e-16 |
| 预先设定误差门 | 1e-5 |
| 非法上下文、重复写入与文件保护 | 通过 |

Vulkan 两种评分路径的 mean NLL 均为 5.15723864954567。所有等价结论都针对同一后端、模型与配置的 compact/reference 或 prefix/reference 对照。

记录：build-clean-fork/vulkan-build.log、build-model-vulkan-20261018/result.json、commands.json 及逐次 stdout/stderr。

## CPU 采样微基准

~~~sh
build-clean-fork/extension/compose/test-tq-compose --benchmark
~~~

实际使用 248,320 个 logits、top-k=40、唯一最大值；每臂预热 10 次、计时 200 次；4 个 AB/BA 交错配对。每次都检查实际 A 选择的 token。

| 配对 | 完整行 µs/row | compact µs/row |
|---|---:|---:|
| 0 | 252.395 | 29.9904 |
| 1 | 309.630 | 29.9273 |
| 2 | 312.292 | 29.9189 |
| 3 | 309.700 | 29.9385 |

这是同一个构建中 C 的 CPU 采样阶段微基准。它支持“保留数值扫描优化有当下收益”，不支持整模型获得同样倍数的加速。没有用单轮模型生成时间比较吞吐，也没有把旧 fork 的历史 +3.66% 改写为本分支实测收益。

## 明确没有证明的事项

- 所有模型族、任意 tokenizer、极长上下文、任意 sampler 组合的等价。
- 多 GPU/五卡 TP5、RERoT 共享 KV、FlashPrefill/XKV 质量与内存收益、DFlash/DSpark 设备隐藏层交接。
- CUDA、HIP、Metal、Windows、macOS 的构建或运行资格。
- 新分支比旧完整 fork 更快，或已保留旧 fork 全部功能。
- Corpus 的崩溃事务、并发写入和旧 reserved metadata 的完整归档。
- 旧上游的全部测试套件；这里只运行了对实际边界有用的指定测试。

有关未迁移功能的理由与重启条件，见 [MIGRATION.md](MIGRATION.md)。
