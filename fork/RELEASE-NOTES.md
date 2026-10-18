# 2026-10-18 发布候选变更说明

## 本批次：增量文本、校准互操作与后端缓存正确性

本批次从 `6de088714f35f84427d9952d713f9d6cdffc668b` 继续整理有价值的小功能；固定上游 A 仍是 `cda0f9256665845cd9d6f3a0176880883f653a3d`。当前维护五个独立 B 库和五个 C 编排包。新增能力由独立的格式、文本对象与真实公开接口消费者承担；Vulkan 私有缓存的修复保留为有明确消费者和回归用例的 A 例外。

**本页记录已取得的开发与运行证据，不代表最终发布门槛全部通过。** scratch 与源数据原地修改两个修复已通过最终专门 GPU 用例，最终 CPU 来源隔离和普通 tools 构建通过。最终全量 GPU 为 25932/25933，仍有一个 FA 失败；较早 TOPK 失败的归因也未闭合。普通 Vulkan 构建入口也已通过，实际身份和统一状态见 [Final release gates](VALIDATION.md#final-release-gates)。A 的文件集合和精确内容以 [manifest](upstream-exceptions.json) 为准。

### 新增与重新划分的职责

| 变化 | 维护合同与使用价值 |
|---|---|
| B `TQText` | 每个对象拥有增量 UTF-8 和 stop 状态，返回拥有自身文本的结果与字节游标；对输入分块保持一致，不依赖 A 或兄弟 B。 |
| C `tq-batch-compose` 的文本输出 | 用真实模型 token piece 驱动 B；支持重复 `--stop`、`--json --stream`，在匹配确定后结束该请求的后续 decode，并按现有序列与预算合同释放状态。 |
| B `TQCalibration` | 有界逐层读取 WXQCAL03，校验几何、统计人口、数值、截断和精确文件结束；各层结果拥有自己的数据。 |
| C `tq-imatrix-compose` | 明确选择完整范围、partial 与 unseen expert 策略，保留原始平方和及可精确表示的计数，用已安装的 A 公共 GGUF/GGML 元数据接口导出 imatrix。 |
| Vulkan scratch 复用 | 共享工作区被大 softmax 或 masked attention 改写时失效转换缓存；复核 no-split 路径最后一次读取与下一次写入的同步责任。该阶段专门用例通过，范围见下表。 |
| 既有 Turbo SET_ROWS | 复核固定 A 已有的 F16 支持，以真实 packed-row 对照验证；不迁入旧 fork 的 F16 拒绝条件，也不把上游已有功能记作新增。 |
| 源数据原地修改修复 | 同一个 source view 的底层值被 SCALE 改写后，A 对重叠写入失效转换缓存。非零偏移 probe 修复前第二次 matmul 64/64 错，重建重装后 matmul 与源存储的完整比较误差为 0；独立于 scratch 覆盖问题验收。 |

五个 B 为 Sampling、Corpus、Scheduling、Text、Calibration，均只依赖 C++17 标准库。五个 C 为 compose、corpus-compose、batch-compose、logits-compose、imatrix-compose。新 imatrix C 仅链接安装后的 `ggml::ggml-base` 和 TQCalibration；A 的 package config 仍会查找启用的 backend，配置时需要完整 A 安装。其他消费者的依赖边界见 [README.md](README.md)。

### 文本输出是用途重建

`TQText` 保留有用的 UTF-8 边界、跨块 stop 与输出生命周期合同，没有宣称复现旧 `server:Q14` 的非流式跳扫描 fastpath 或速度收益。`server:T09` 只提供了跨块终止符机制的参考；完整 thinking/lane parser 没有迁入。

`tokens` 保留已采样的非 EOG token，`--ignore-eog` 时也保留 EOG；stop 抑制文本不回删计数。匹配确定后 C 不再提交该请求的 decode。若最终 UTF-8 修复才完成 stop，承载未完成字节的早先 token 可能已经提交给 A。每个成功请求恰有一个终止文本事件，按字节游标拼接的增量等于最终文本；stdout 失败令运行失败。此合同没有包含 HTTP SSE、网络送达、取消、重试或多 choice usage。

以下为使用示例，不是本批次的实际验证命令回执：

```sh
build-clean-fork/install/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt --parallel 2 \
  --ctx-size 2048 --predict 64 --stop '<end>' --stop '###' --json --stream

build-clean-fork/install/bin/tq-imatrix-compose \
  --input calibration.bin --output imatrix.gguf \
  --dataset 'my-selected-calibration-range'
```

### 校准导出保留测量与估计的区别

默认要求生产者报告 complete、声明的总量为正且 `decoded == total`，并拒绝没有观测值的 expert。旧 writer 的最后一批可超出所选节点上限；`--allow-partial` 会保留原始计数并记录每个实际原因，不能用它跳过结构或数值错误。`--unseen global-mean` 是显式估计，记录填补的 expert ID、所用策略与“填补行不是测量”的元数据，绝不把 count 1 当作真实观测数。

导出保留 raw sum2；A 才按 expert count 归一化。计数必须是可精确表示的 F32 整数且不超过 INT32_MAX；平方和及 A 中实际 F32 除法的溢出、非零下溢都拒绝。一个 parsed layer、有限转换缓冲和私有磁盘 spool 控制内存；完整验证后才无覆盖发布新文件。完整格式、平台、磁盘容量和失败合同见 [imatrix-compose/README.md](imatrix-compose/README.md)。

### 本批次已观察的验证结果

| 检查 | 实际结果与范围 |
|---|---|
| 最终 CPU archive / isolation | `build-isolated-details-qualified-20261018` 的 62 条命令通过；五个 B、五个 C 独立构建、安装，移除原副本和前缀后搬迁，真实 batch/logits/stream 与 imatrix 消费通过。 |
| 两个新增 B | TQText、TQCalibration 的 ASan 与 UBSan 测试通过；不据此补写较早 logits writer 的 ASan 资格。 |
| 旧 writer 与实际 A loader | 原 `calib-dump.cpp/.h` 与旧提交 `2777543b2358c88cd1c2000f7fa342b764a0e213` 完全相同；生产格式 fixture 经新 C 产生 12 tensors，A quantizer loader 读入 6 entries / 9 chunks。随后故意缺失模型以 exit 1 结束，只证明格式互操作与 loader 消费。 |
| 最终 Vulkan 接缝 | 7/7 CTest 通过，含 5 个 scratch 用例、2 个 source-alias 用例及 16 次 SET_ROWS packed 比较；no-split 检查全部 524288 输出及闭式结果。source-alias 的 4-byte-offset probe 修复前 64/64 错，最大绝对差 15.4662，修复后完整比较误差为 0。 |
| 最终 Vulkan 文本消费者 | 10 条命令，其中 6 个真实模型配置各 3 个请求，共 18 个请求通过，脚本用时约 14.5 秒；stream 开关对照保持同配置提交形状。该时间不是吞吐基准。 |
| 已完成的全量 CPU backend 测试 | 27639 / 27639 通过，保留该次运行身份；最终广覆盖复验状态另列。 |
| 最终全量 Vulkan backend 测试 | 25932 / 25933，exit 1，仍有同一 mask+sinks、Q8_0 K/V、head dimension 256 的 FA 失败，报告 ERR 0.009055835 超过 0.0005 门限。最终 TOPK 未复现；较早 +12 full 中的 TOPK 失败保留，归因仍未闭合，没有因此排除回归。 |
| 普通 tools 构建入口 | 实际 CPU 与 Vulkan `build.py --tools` 流程均 exit 0，分别约 67.84 秒和 7.31 秒；这是构建回执，不是推理性能测量。 |
| Vulkan K/V 执行 smoke | 9 个 K/V 组合的 Prefill / Decode 共 18 行结果，exit 0，只有 1 repeat；这只是执行检查，不提供可靠吞吐或收益结论。 |
| 设备范围 | 本批次实际 GPU 后端是 Vulkan；没有 CUDA 设备，CUDA 未取得运行资格。 |

最终专项已对包含两个修复的 A 重新运行，但不能抵消最终全量 GPU 中保留的 FA 失败，也不能撤销较早未定位的 TOPK 失败。最初复杂 V fixture 观察到的 `0.0146484` 闭式偏差保留为历史反例；改用可精确构造的 binary fixture 后仍使用原来的误差门。这个过程没有证明曾实际复现 no-split 缺少同步的竞态。协议、映射与证据边界见 [本批次 findings](audit/details-findings-20261018.md)、[VALIDATION.md](VALIDATION.md) 和 [本批次机器摘要](audit/details-qualification-20261018.json)。

### 对 A 的维护成本

最终已核对 17 个 A 源码文件合计 +357/-43；另有 49 行自有 CI，manifest 共 18 路径、合计 +406/-43。本批次在既有 A 差异上新增 29 行。文件数本身不证明正确性；每项理由、消费者、删除条件和真实回归仍随 manifest 与验证记录维护。

### 仍保留的有用候选

`small:S04` 的格式和 imatrix 导出已由 B/C 重建；`small:S05` 的真实 callback 采集、DFS、checkpoint 及模型契约仍待迁。`server:Q15/Q16/Q17` 的首 chunk 状态、多 choice usage 与最终 SSE 事件，完整 thinking/lane parser、HTTP 取消和恢复，均没有被 CLI 文本输出替代。`backend:G14` 的正确设备 pinned host 分配/释放/查询仍是候选；G09 的正确性工作不能为它取得多 GPU 资格。

已保留的 Prefill/Decode 内存、统一物理 KV、准入、采样、corpus 和 logits 能力继续维护，各自先前测量保留原身份。没有新吞吐、峰值 VRAM、模型量化质量或跨形状一致性结论；逐项清单见 [MIGRATION.md](MIGRATION.md)。

## 同日较早发布候选记录：起点 3746ebf82585

下面保留该批次原文。其“本轮”“当前”、三个 B / 四个 C、47 条命令与 A 差异数字仅指起点 `3746ebf82585f47c88fb48097f6ae45cb972b031` 的较早资格，不能用于声称本页新增代码或最终 A 已完成相同验证。Meta graph-view UID 的源码纠正及 admission API 名称修正已保留。

本轮在 `clean/fork-20261018` 上继续处理有价值的小功能。固定上游A仍为 `cda0f9256665845cd9d6f3a0176880883f653a3d`，本轮起点为 `3746ebf82585f47c88fb48097f6ae45cb972b031`。交付三个独立B库与四个C编排包；新运行时能力都有真实消费者、失败合同、安装验证和可回查证据。本轮没有推送、打tag或发布发行包。

### 新增与调整

| 变化 | 设计与实际结果 |
|---|---|
| token/槽位准入 | `tq::scheduling::admission_budget` 同时限制已知token需求和驻留请求数，拥有自己的账目。domain/generation拒绝外来、陈旧、重复释放；失败不改变账目。 |
| 真实排队与槽位复用 | `tq-batch-compose --parallel` 将稳定请求ID与可复用A序列槽分开，FIFO准入、保持Prefill选择、消费全部输出并同步后删除序列，最后回收B预算。队列可长于驻留槽位。 |
| 有界完整logits采集 | 新 `tq-logits-compose` 用真实token做teacher forcing，记录TP5LOG2完整词表行。只调用安装后的A公开API；默认逐token，支持明确前缀。 |
| 可预测的文件发布 | 预检尺寸、逐行有限值、有限缓冲、失败终止、完成后无覆盖发布。原文件和symlink保留；stdout失败准确报告已完整发布的tape。 |
| Meta async偏移 | 一个A文件+4/-4，将逻辑chunk起点换算为child tensor偏移；原A非零get/set分别失败，修复后48个真实用例通过。 |
| recurrent生命周期 | 复用当前A，新增公开接口的clear、共享、邻居、dirty槽位复用及合法恢复回归；没有复制旧eager-clear实现。 |
| 独立构建与CI | 安装/搬迁验证加入第四个C和admission外部消费者，`--tools`补建A所需mtmd，新增独立CI配置。 |

完整用法在 [README.md](README.md)；各模块的README定义输入、所有权、退出码和平台范围。具体测试、命令与数值在 [VALIDATION.md](VALIDATION.md)，可移交机器摘要在 [audit/release-qualification-20261018.json](audit/release-qualification-20261018.json)。

#### B 只拥有预算，C 执行模型操作

准入需求是已知上界 `prompt_tokens + predict - 1`。B不探测显存，不持有A context，不代替A清理状态，也不因reservation析构就回收尚在使用的模型资源。C只有在真实序列删除成功后才释放reservation。失败的释放不会令容量被重复计入。

当前C是确定输入、有限生成长度的批处理程序。token或槽位不足时等待；无法容纳的单请求提前失败。请求生命周期的不可恢复错误令本次进程失败。本轮没有把这个合同扩写成在线HTTP取消、压力重试、抢占、LRU或未知长度生成的保证。

Prefill选择策略与FIFO准入分开：已经准入的提示由B保持当前Prefill，C在提示期间暂停Decode；队列资格按输入顺序。持续提示流的公平性没有获得证明。阶段内存仍只释放context自己的compute allocator，保留模型、KV/recurrent、输出和sampler；报告的字节不是进程峰值VRAM。

#### logits 文件可以复查，身份需要随运行保存

默认在每个输入token之后保存一行完整logits；该行预测下一个token。`--prefill N`保存前缀最后一行，随后逐token保存。输入始终来自文件，不由模型采样反馈。输出大小在创建context前检查；writer只有16KiB转换缓冲。

TP5LOG2保留既有二进制格式，头中没有模型、tokenizer、构建或设备身份。必须同时保存输入/模型摘要、命令、stdout、后端日志和构建版本。相同token ID和词表大小不能证明这些身份相同。

发布依赖POSIX同目录hard link与调用者控制的目录。完整文件出现前不会暴露部分目标；存在目标不覆盖。完整发布后stdout写入失败会退出1并说明文件已发布，文件保持完整。没有Windows、目录断电持久性或强制终止自动清理的承诺。

### 实际验证

本轮使用LFM2.5-8B-A1B-UD-IQ4_NL，同一个模型SHA-256在摘要中固定。分别运行CPU与单张RX6800 Vulkan；GPU模型日志实际确认25层卸载。

| 检查 | 通过范围 |
|---|---|
| 准入/阶段/复用 | 每后端7个运行、60个成功请求；同配置开关、串行容量对照、同槽位重复波次均通过，两后端共120请求。 |
| Meta传输 | 48例，axes0/1/2/mirror、F32/F16、不均分片、偏移、get/set与guard；真实参与者是两个CPU child backend。 |
| recurrent | 每后端24邻居快照、8共享快照检查，合法恢复继续最大绝对差0；模型范围为LFM2.5 hybrid convolution/KV，`n_rs_seq=0`。 |
| logits完整行 | 每后端15×128000个值；重复tape按位相同，同标量形状的suffix/prefix全行差0。 |
| logits独立C对照 | 相同提示token与提交形状，最终greedy token/最大logit在1e-5内通过；这项独立对照不提供第二份全词表行。 |
| 文件与错误 | 真实文件故障、竞争发布、symlink、缺行/多行、NaN、上下文与文件预算、stdout失败均有检查。 |
| 独立交付 | 47个命令通过：从固定A archive重建，三个B与四个C各自构建/安装，移除源码/构建副本和原前缀，再搬迁、编译消费者与运行真实模型。 |
| 常规入口 | CPU默认、Vulkan `--tools`完整构建退出0；边界12项与比较器20项通过。 |

本地检查通过不等于GitHub已经执行CI。新工作流已配置，尚未推送；没有远端CI或发布状态可宣称。

### 保留的已知数值限制

这次没有隐藏一个不符合最初预期的结果：LFM2.5三路与单路的greedy输出存在差异。该现象已在准入改动前的C中、关闭阶段释放时复现；单路只改变KV容量则输出相同。原始失败记录、旧二进制摘要、结果token和复现脚本均保留。因而本轮对照固定实际提交形状，跨形状观察另列。

完整logits诊断同样观察到chunked与scalar前缀的最终行差异，最大绝对差CPU为 `2.6373519897460938`，Vulkan为 `0.9928340911865234`。这些数值不能被称作已经证明无害的舍入误差；数值原因尚未定位，也没有承诺跨后端或跨批形状结果一致。固定相同形状的重复、行选择和状态复用检查仍保留严格判定。

recurrent测试也有明确范围：没有一般GDN、MTP、PLE或损坏snapshot事务恢复资格。源码审查发现坏restore可能先改变目的状态，不能把合法恢复成功写成所有错误恢复都不改变状态。

### 为什么有些旧细节没有直接搬入

| 旧细节 | 当前判断 |
|---|---|
| M33单cell清零优化 | 旧907–935ms到12.8–16.8ms减少的是旧fork新增的清零。当前A已有整buffer清零与图内延迟初始化；搬入旧路径会增加写与fence，因此保留目的、复用A、增加生命周期回归。 |
| Meta graph-view cache-key | 当前 A 会复用非零图 UID，但 graph view 的 UID 为零。旧 fork 才让 view 继承父图 UID；旧 key 修复与该优化共同设计，不能孤立搬入。 |
| pinned-readback retire | 当前Meta路径没有旧pending transaction；不孤立引入退休队列或pin。需要真实异步消费者、所有权与设备验证后再设计。 |

auto-fit/slack、PP reserve、idle LRU/cache affinity、物理recurrent槽位、取消和随机状态恢复、流式stop/usage、在线corpus、state envelope、imatrix与模型兼容等继续保留逐项候选，不按大模块名称删除。详见 [MIGRATION.md](MIGRATION.md)。

### 对 A 的维护成本

当前A代码差异为17文件、+328/-43；另有49行自有CI，manifest总计18个例外。每个文件登记精确内容、消费者、理由与删除条件。自有代码都在 `fork/`，三个B只依赖C++17标准库，C通过安装包调用公开A/B接口。原版server的小范围私有调度修改仍是明确例外。

上游公开结构体扩展不承诺跨版本ABI；更新A时须重建消费者。可用 [check-upstream.py](check-upstream.py) 检查内容边界，再执行独立交付与相关真实回归。原master和 `archive/pre-clean-20261018` 保留旧实现；没有为了清理分支删除历史功能。

本轮没有测量新吞吐、峰值VRAM或reset加速。CUDA、HIP、Metal、Windows、macOS、多GPU和异构Meta仍没有这次运行资格。714路径的审计记录保持原阅读深度，不因本轮完成的功能而扩大“已审完”的范围。
