# 2026-10-18 发布候选变更说明

本轮在 `clean/fork-20261018` 上继续处理有价值的小功能。固定上游A仍为 `cda0f9256665845cd9d6f3a0176880883f653a3d`，本轮起点为 `3746ebf82585f47c88fb48097f6ae45cb972b031`。交付三个独立B库与四个C编排包；新运行时能力都有真实消费者、失败合同、安装验证和可回查证据。本轮没有推送、打tag或发布发行包。

## 新增与调整

| 变化 | 设计与实际结果 |
|---|---|
| token/槽位准入 | `TQScheduling::admission_budget` 同时限制已知token需求和驻留请求数，拥有自己的账目。domain/generation拒绝外来、陈旧、重复释放；失败不改变账目。 |
| 真实排队与槽位复用 | `tq-batch-compose --parallel` 将稳定请求ID与可复用A序列槽分开，FIFO准入、保持Prefill选择、消费全部输出并同步后删除序列，最后回收B预算。队列可长于驻留槽位。 |
| 有界完整logits采集 | 新 `tq-logits-compose` 用真实token做teacher forcing，记录TP5LOG2完整词表行。只调用安装后的A公开API；默认逐token，支持明确前缀。 |
| 可预测的文件发布 | 预检尺寸、逐行有限值、有限缓冲、失败终止、完成后无覆盖发布。原文件和symlink保留；stdout失败准确报告已完整发布的tape。 |
| Meta async偏移 | 一个A文件+4/-4，将逻辑chunk起点换算为child tensor偏移；原A非零get/set分别失败，修复后48个真实用例通过。 |
| recurrent生命周期 | 复用当前A，新增公开接口的clear、共享、邻居、dirty槽位复用及合法恢复回归；没有复制旧eager-clear实现。 |
| 独立构建与CI | 安装/搬迁验证加入第四个C和admission外部消费者，`--tools`补建A所需mtmd，新增独立CI配置。 |

完整用法在 [README.md](README.md)；各模块的README定义输入、所有权、退出码和平台范围。具体测试、命令与数值在 [VALIDATION.md](VALIDATION.md)，可移交机器摘要在 [audit/release-qualification-20261018.json](audit/release-qualification-20261018.json)。

### B 只拥有预算，C 执行模型操作

准入需求是已知上界 `prompt_tokens + predict - 1`。B不探测显存，不持有A context，不代替A清理状态，也不因reservation析构就回收尚在使用的模型资源。C只有在真实序列删除成功后才释放reservation。失败的释放不会令容量被重复计入。

当前C是确定输入、有限生成长度的批处理程序。token或槽位不足时等待；无法容纳的单请求提前失败。请求生命周期的不可恢复错误令本次进程失败。本轮没有把这个合同扩写成在线HTTP取消、压力重试、抢占、LRU或未知长度生成的保证。

Prefill选择策略与FIFO准入分开：已经准入的提示由B保持当前Prefill，C在提示期间暂停Decode；队列资格按输入顺序。持续提示流的公平性没有获得证明。阶段内存仍只释放context自己的compute allocator，保留模型、KV/recurrent、输出和sampler；报告的字节不是进程峰值VRAM。

### logits 文件可以复查，身份需要随运行保存

默认在每个输入token之后保存一行完整logits；该行预测下一个token。`--prefill N`保存前缀最后一行，随后逐token保存。输入始终来自文件，不由模型采样反馈。输出大小在创建context前检查；writer只有16KiB转换缓冲。

TP5LOG2保留既有二进制格式，头中没有模型、tokenizer、构建或设备身份。必须同时保存输入/模型摘要、命令、stdout、后端日志和构建版本。相同token ID和词表大小不能证明这些身份相同。

发布依赖POSIX同目录hard link与调用者控制的目录。完整文件出现前不会暴露部分目标；存在目标不覆盖。完整发布后stdout写入失败会退出1并说明文件已发布，文件保持完整。没有Windows、目录断电持久性或强制终止自动清理的承诺。

## 实际验证

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

## 保留的已知数值限制

这次没有隐藏一个不符合最初预期的结果：LFM2.5三路与单路的greedy输出存在差异。该现象已在准入改动前的C中、关闭阶段释放时复现；单路只改变KV容量则输出相同。原始失败记录、旧二进制摘要、结果token和复现脚本均保留。因而本轮对照固定实际提交形状，跨形状观察另列。

完整logits诊断同样观察到chunked与scalar前缀的最终行差异，最大绝对差CPU为 `2.6373519897460938`，Vulkan为 `0.9928340911865234`。这些数值不能被称作已经证明无害的舍入误差；数值原因尚未定位，也没有承诺跨后端或跨批形状结果一致。固定相同形状的重复、行选择和状态复用检查仍保留严格判定。

recurrent测试也有明确范围：没有一般GDN、MTP、PLE或损坏snapshot事务恢复资格。源码审查发现坏restore可能先改变目的状态，不能把合法恢复成功写成所有错误恢复都不改变状态。

## 为什么有些旧细节没有直接搬入

| 旧细节 | 当前判断 |
|---|---|
| M33单cell清零优化 | 旧907–935ms到12.8–16.8ms减少的是旧fork新增的清零。当前A已有整buffer清零与图内延迟初始化；搬入旧路径会增加写与fence，因此保留目的、复用A、增加生命周期回归。 |
| Meta graph-view cache-key | 当前A的UID未建立旧cache复用前提；单独搬一个key修复没有当前消费者。未来引入cache应一起闭合生命周期和实际收益。 |
| pinned-readback retire | 当前Meta路径没有旧pending transaction；不孤立引入退休队列或pin。需要真实异步消费者、所有权与设备验证后再设计。 |

auto-fit/slack、PP reserve、idle LRU/cache affinity、物理recurrent槽位、取消和随机状态恢复、流式stop/usage、在线corpus、state envelope、imatrix与模型兼容等继续保留逐项候选，不按大模块名称删除。详见 [MIGRATION.md](MIGRATION.md)。

## 对 A 的维护成本

当前A代码差异为17文件、+328/-43；另有49行自有CI，manifest总计18个例外。每个文件登记精确内容、消费者、理由与删除条件。自有代码都在 `fork/`，三个B只依赖C++17标准库，C通过安装包调用公开A/B接口。原版server的小范围私有调度修改仍是明确例外。

上游公开结构体扩展不承诺跨版本ABI；更新A时须重建消费者。可用 [check-upstream.py](check-upstream.py) 检查内容边界，再执行独立交付与相关真实回归。原master和 `archive/pre-clean-20261018` 保留旧实现；没有为了清理分支删除历史功能。

本轮没有测量新吞吐、峰值VRAM或reset加速。CUDA、HIP、Metal、Windows、macOS、多GPU和异构Meta仍没有这次运行资格。714路径的审计记录保持原阅读深度，不因本轮完成的功能而扩大“已审完”的范围。
