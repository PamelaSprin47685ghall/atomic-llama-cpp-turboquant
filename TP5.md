# TP5：Qwen4EXP 在五张 RX 6800 上的 Vulkan 张量并行实现方案

版本：设计稿 v1，2026-09-12。源码基线：`cbe8014c1947d78a9b7a593fc0f9ec92ac74cafb`，分支 `master`。

本文面向实际实现者。没有看过原对话，也可以按本文完成设计审查、开发拆分和验收。本文新增的文件、类型、接口、命令行参数和测试目标均是**拟实现项**，不是当前仓库已经具备的功能。

本轮交付仅为文档。做过源码检查、开发机只读环境检查及小尺寸 CPU 代数核对；没有编译本仓库，没有运行 Vulkan 测试、加载模型或复测五卡性能。文末列出源码和规范依据。

## 0. 先读这四个结论

**第一版采用 A-F，但不把它写成已证明的最快方案。** HC 的残差、权重和计算在五卡复制；优化 HC 的本地执行；MoE 切专家内部通道；普通/QSA attention 切完整 Q 头；GDN 切完整状态头。每个子层完成本地输出投影和本地分支合并后，只做一次输出求和。单 token decode 首先实现全量 PUSH 群发，FP32 建立正确性基线，FP16 通信单独验收。

**这不是删一个架构检查、加一个归约 shader 就能完成的改动。** 当前 meta backend 可以复用，但它的连续分片描述、图切分边界、Vulkan collective 接口、缓存布局和跨设备同步都需要补齐。尤其不能把重叠的 KV 副本硬塞进要求各片总长等于原长的旧描述。

**当前连接的是开发机，不是附件中的五卡测试机。** 本轮只读检查枚举到 Phoenix 集显，内存约 62 GiB，CPU 信息包含 Ryzen 9 7940H/7940HS，内核为 `7.1.13-2-MANJARO`。附件所述 X99、DDR4、五张 RX 6800、定制内核及 P2P 延迟，均属于另一台目标机的历史报告。不能据此声称本轮已经验证五卡硬件或性能。

**先算逐卡显存，再承诺上下文长度。** 全量复制 HC、复制 KV 和 indexer 会吃掉相当多显存。80 GiB 总容量不是五张卡任意调剂的内存池。128K 上下文可能装不下某个实际 GGUF；文中给出明确的条件算例，而不是预先保证能装下。

### 阅读顺序

先看第 1—5 节，确定证据、范围、显存和架构；第 6—10 节规定模型计算；第 11—17 节规定通信、同步、接线和状态；第 18 节以后给出精度、任务拆分、测试与验收。所有实现任务都以“输出是什么、由谁拥有、消费者何时能读”为边界。

## 1. 证据基线：哪些东西已经存在

### 1.1 三种信息不能混写

本文使用三类依据：

| 标记 | 含义 | 使用方式 |
|---|---|---|
| R | 本轮检查的仓库源码，均以本文 HEAD 为准 | 给出相对路径、符号或行号 |
| H | 用户提供的《Prefill速度估算》历史对话导出 | 保留其决策背景；性能数字不当作本轮实测 |
| V / M | 本轮查询的 Vulkan 规范、官方模型配置 | 用来核对 API 约束和目标尺寸；不替代目标机能力查询或实际 GGUF |

行号只用于定位本文基线。后续修改应按符号查找，并更新实现报告中的提交号。历史对话中“已跑通”“已推送”的说法，必须在当前源码和对应日志中找到落实，才能升级为当前能力。

### 1.2 当前代码的真实状态

| 部位 | 本轮确认的状态 | 工程含义 |
|---|---|---|
| `src/models/qwen4exp.cpp` | 已有 HC、QSA、GDN、MoE、PLE 和最终 HC 图 | 保留原图作为参考，不另写一套模型公式 |
| `src/llama-arch.cpp::llm_arch_supports_sm_tensor` | `QWEN4EXP` 仍在不支持分支，注释要求修架构测试 | 不能先删检查再处理后续断言 |
| `src/llama-model.cpp::llama_meta_device_get_split_state` | 已有 Qwen 家族分段、量化粒度、层间轮换；Qwen4EXP 已进入部分规则 | 不是从零开始，但现有规则不能直接表达本文 TP5 |
| `ggml/src/ggml-backend-meta.cpp` | 有 MIRRORED、PARTIAL、按轴分片及多种算子传播；支持非二次幂卡数的归约回退 | 五张卡不是框架数学禁区 |
| 同上，构造与执行路径 | 已查询 `ggml_backend_comm_init/free/allreduce_tensor` | 优先复用这个后端通信入口 |
| `ggml-vulkan.cpp::ggml_backend_vk_reg_get_proc_address` | 当前只返回 FlashPrefill scratch 查询，没有上述 collective 接口 | Vulkan TP 通信尚未在生产后端接线 |
| `ggml-vulkan.cpp::ggml_vk_buffer_copy` | 跨设备分支经源卡 `sync_staging` 再写目标 | 当前通用路径是 host staging，不能拿 POC 的 P2P 代替它 |
| `pocs/vulkan-p2p/` | 只有 CMake 和两阶段 AllReduce 测试源码 | 没有本轮历史对话后段所述 A-F/A-D 探针或 SYNC_FD 执行器源码 |

主要定位：R1—R7，见文末。架构头文件实际在 `src/llama-arch.h`，不是原对话偶尔写到的 `include/llama-arch.h`。

### 1.3 当前 POC 可以复用什么，不能证明什么

`test-vulkan-p2p-allreduce.cpp` 的算法地址关系是成立的：第一阶段把发送卡的目标片段送进接收卡的发送者槽，本地五路相加，再广播已归约片段。它也确实遍历五卡共 51,200 个 FP32 元素，与 15.0 比较。

但提交路径是两轮“提交五卡 → 主机等待五卡 fence”。没有启用外部 semaphore 扩展，没有 SYNC_FD 导入导出，也没有 HC 折叠。100 次计时重复使用固定输入，打印的 `96 × latency` 是乘法，不是一条含 96 个消费依赖的模型链。不能据此支持几微秒异步归约或某个 token/s。

这份 POC 还需要修正：导入端内存要求没有与 FD 类型位求交；逻辑载荷大小被用于导入分配；找不到内存类型时默认索引 0；固定 queue family 0；共享资源所有权未明确表达；部分返回值未检查；固定 `/tmp` shader 文件和清理不完整。因此保留其寻址与数学验收，不能直接复制成生产通信层。主机等待也不能替代这些外部内存规则。

### 1.4 历史数字的处理

| 历史数字 | 本文如何使用 |
|---|---|
| 23.4 GB/s P2P | 目标机某种测试口径的报告；不是 shader 远读带宽或全网并发带宽 |
| 0.73 μs / copy | 批量拷贝平均成本，不是五卡 AllReduce 延迟 |
| 约 17.5 μs / 阶段 | 特定探针结果，不是所有尺寸、同步方式通用常数 |
| 约 970 μs / 两阶段 AR | 历史主机同步测试报告；当前源码具有相应计时结构，本轮没有运行 |
| A-F 9.08 ms，A-D 10.54 ms | 支持先把 A-F 作为开发候选；探针源文件和完整计时范围未在当前 POC 中找到 |
| 52、60、67、95 token/s | 不进入验收结果，也不作为本方案的性能承诺 |

尤其是 `96 × 320 × 10240 × 2 = 629145600 B`，只对应 96 个 FP16 **单投影**。HC 的 down 和 up 都执行时，这两个大投影的逻辑权重字节数要乘二。没有确认探针范围前，不能把 9.08 ms 命名为“全部 HC 耗时”。[H：3840—3953]

## 2. 首版范围与决策

### 2.1 支持目标

目标为 Linux、Vulkan、五张同类独立 RX 6800、Qwen4EXP 文本主干；权重除 n-gram 表外驻 GPU，KV 与 GDN 状态驻 GPU，n-gram 表驻 CPU RAM。首先优化单请求、单 token decode；prefill 必须正确，并与 decode 使用同一缓存和状态归属。

首个闭环只承诺单序列。多请求并发、MTP、推测解码、视觉编码器、LoRA、自定义控制向量、XKV/RERoT/FlashPrefill 等组合，不能因旧路径有功能就默认 TP5 也支持。未适配的组合在初始化时清晰拒绝；旧非 TP5 路径照常保留。后续逐项补齐，而不是静默丢掉模型参数或状态。

### 2.2 固定的首版决策

| 部分 | 决策 | 不采用的捷径 |
|---|---|---|
| HC | 全复制；先实现未融合参考，再做 F3、F2、F1 | 不把复制工作按五分之一计时 |
| Routed MoE | 每个专家切 128 个中间通道，五卡拥有全部专家的对应切片 | 不用专家数量均分冒充 token 负载均衡 |
| Shared expert | 同样切中间通道；本地乘共享 gate，和 routed 部分合并后归约 | 不增加第二次 FFN 输出归约 |
| Router | 复制，保证相同输入、同一算法、同一 tie-break | 不先把局部 logits 做 softmax 再相加 |
| QSA attention | 完整 Q 头不等长切分；KV 必要副本；桥接卡分成合法 GQA 调用 | 不补假头，不用 5Q/2KV 的错误等比例映射 |
| GDN | 保留完整 V/state 头；显式全局 Q/K 映射 | 不独立切时间段，不改全局 head_id 的语义 |
| Indexer | 首版复制计算和索引缓存 | 不称复制成本为零 |
| 输出通信 | decode 首先测试单阶段全量 PUSH mesh | 不把单卡完整部分和误当成结果的五分之一 |
| 传输精度 | FP32 基线；可选 FP16 传输、FP32 求和 | 不把权重量化类型当作通信 dtype |
| 同步 | 正确主机基线 → SYNC_FD 队列依赖 | 不把本地 barrier 或最后一次 fence 当跨卡就绪协议 |
| 显存 | 加载前逐卡预算，角色可按层固定轮换 | 不按全机平均显存作准入 |

### 2.3 A-D、B、C 保留为实验，不进入默认

A-D 只切 HC down；B 切 down 输出与 up 输出，用两次 AllGather；C 切 down 输出与 up 输入，在 sigmoid 前归约完整 gate；C-fold 将 gate ReduceScatter 后的折叠插入通信中，只汇集 H 维 mixed。它们在实数代数上可成立，但通信、内核形状和复制后处理成本不同。

首版不实施这些分支，不为它们设计进程级动态自动切换。保留方案定义、预算和测试接口即可。A-F 经过本地融合后，A-D 的收益门槛还会变化；历史某个探针较慢不等于它在所有 batch、驱动和模型类型上都不值得做。

## 3. 尺寸、符号和 ggml 内存方向

### 3.1 本文示例使用的目标尺寸

官方配置和附件一致的主要尺寸如下；实际加载仍以 GGUF 为唯一运行依据。[M1；R1]

| 符号 | 含义 | 目标值 |
|---|---|---:|
| P | TP rank 数 | 5 |
| L | 主体层数，不计额外 MTP | 48 |
| H | 隐藏维 | 2560 |
| C | HC 流数 | 4 |
| R | HC 低秩维 | 320 |
| E / K | routed 专家数 / 每 token 选中数 | 512 / 10 |
| F / Fs | routed / shared 中间维 | 640 / 640 |
| Nq / Nkv / da | full attention Q / KV 头数、头维 | 24 / 2 / 256 |
| Nk / Nv / ds | GDN QK / V 头数、头维 | 16 / 48 / 128 |
| Ni / di | indexer Q 头数、头维 | 4 / 128 |
| B | 本次子图处理的 token 数 | decode 为 1；prefill 可更大 |

默认每四层一层 full attention，因此示例有 12 层 full attention、36 层 GDN。但计划生成器必须读取逐层类型，不能固定 `il % 4`。PLE 层、卷积核、RoPE sections、实际 rotary 维度、epsilon、注意力缩放、词表及 indexer budget 都读元数据，不从 H 或头数推测。

**H 不是 Q 投影总宽度。** 本例 `Nq × da = 6144`，含 gate 的 Q 投影输出为 12288；不能套 `head_dim = H / Nq`。RoPE 也不必覆盖全部 256 个头内通道。

### 3.2 矩阵方向约定

数学公式采用列向量：`y = W x`，数学 W 的形状是 `[输出, 输入]`。

ggml 的普通乘法权重采用 `ne[0]=输入维、ne[1]=输出维`；`ggml_mul_mat(W, X)` 得到输出维在 `ne[0]` 的张量。文中所有权重表明确写 ggml 形状，避免把论文中的 row/column parallel 名称直接当作 `ne[0]/ne[1]`。

量化编码一般沿 `ne[0]` 成块。是否能切片必须同时检查逻辑形状、量化块、编码行长度、物理 strides、Vulkan 内核支持和对齐。`128` 是本方案的逻辑宽度，不是通用硬件最优粒度。

### 3.3 关键权重切片表

| 权重 | 原始 ggml shape | 首版每 rank shape / 规则 | 输出语义 |
|---|---|---|---|
| HC down | `[CH,R]` | 完整复制 | 完整低秩值 |
| HC up | `[R,CH]` | 完整复制 | 完整 gate |
| HC inject | `[CH,C]` | 完整复制 | 完整 inject |
| routed gate/up | `[H,F,E]` | `[H,128,E]`，切 axis 1 | 对应的中间通道 |
| routed down | `[F,H,E]` | `[128,H,E]`，切 axis 0 | 全 H 维部分和 |
| fused routed gate_up | 按加载器实际排列，常为 `[H,2F,E]` | gate 与 up 各取对应 128，不能直接取连续 256 后猜含义 | 两个配对片段 |
| shared gate/up | `[H,Fs]` | `[H,128]` | 局部中间通道 |
| shared down | `[Fs,H]` | `[128,H]` | 全 H 维部分和 |
| router | `[H,E]` | 复制 | 完整路由 logits |
| Q+gate | `[H,2daNq]` | 按完整交错头的输出行抽取 | 本卡 Q 和匹配 gate |
| K/V | `[H,daNkv]` | 本卡所需 KV 头完整输出行，允许跨 rank 副本 | 对应完整 KV |
| WO | `[daNq,H]` | `[da·本卡Q头数,H]` | 全 H 维部分和 |
| GDN out | `[dsNv,H]` | `[ds·本卡V头数,H]` | 全 H 维部分和 |
| LM head | `[H,V]` | 沿词表输出行切片 | 不重叠的完整 logits |

凡有 `*_s` 伴随缩放张量、分块比例或 LoRA 参数，必须与主权重联动处理；首版不支持的形态在 plan 校验时拒绝。不能只切主 tensor 后忘记缩放元数据。

## 4. 加载前清单与逐卡显存预算

### 4.1 一个真正不分配 GPU 的检查入口

拟新增 `tools/tp5/tp5-inspect-model.py`，只读 GGUF metadata 和 tensor headers，产出 `tp5-manifest.json`。不要调用模型加载，不创建 Vulkan backend，不触碰全部权重页，不为核对模型身份顺手全量读取几十 GiB 做 hash。模型身份可先记录现有仓库版本、文件大小、mtime、GGUF 元数据摘要和用户已有校验值；发布验收时再提供完整权重校验来源。

**不要把当前 `params.dry_run` 当作 metadata-only。** `src/llama-model.cpp` 的注释明确说明它仍按真实加载分配和写入，只跳过权重文件读取；Vulkan 还可能延迟实际分配。[R2：345—354]

manifest 对每个 tensor 记录：名称、语义枚举、层号、`ne[]`、类型、量化块、编码 row_size、逻辑字节数、实际 shard 字节数、每 rank 的源范围/目标范围、复制原因、重排方式、伴随 scale 和计划消费者。缓存另记头映射、容量、dtype、padding、物理 strides、序列槽数和快照份数。

全局记录 `schema_version`、源码 HEAD、模型元数据摘要、rank 到设备 UUID/PCI BDF 的绑定、角色轮换表、通信精度、状态格式版本和 plan hash。manifest 不是测速日志，也不能写未查询到的显存空闲量。

### 4.2 准入公式

对物理 rank i：

\[
M_i=M_{W,shard,i}+M_{W,replica,i}+M_{KV,i}+M_{idx,i}
  +M_{GDN,i}+M_{PLEstate,i}+M_{activation,i}+M_{scratch,i}
  +M_{comm,i}+M_{runtime,i}+M_{reserve,i}.
\]

必须 `M_i <= 当前可用的设备预算_i`，而不是只满足五卡总和。reserve 由目标机实际运行需求给出，不把设备标称容量全部用满。报告静态预算与运行峰值两列；原始权重映射、重排临时副本、pipeline/descriptor 和驱动开销不能藏在“其他约零”中。

### 4.3 两个必要的容量算例

96 个主体 HC 的两个大投影，假设权重均为 FP16：

\[
96\cdot 2\cdot10240\cdot320\cdot2
=1.171875\ \mathrm{GiB/rank}.
\]

最终 HC 的两个大投影再加 12.5 MiB/rank；norm、inject、其他权重另计。这里的“五卡复制”确实会在五卡各占一份，不是只在某张主卡保留。

所有 routed 专家的三个投影共有：

\[
48\cdot512\cdot3\cdot2560\cdot640
=120795955200\text{ 个权重元素}.
\]

理想无开销 4 bit 且完全均分，也需要 11.25 GiB/rank。假设 gate/up 每 256 元素编码为 136 字节、down 每 32 元素编码为 18 字节，则这部分变成 12.1875 GiB/rank。**这是明确指定编码密度的条件算例，不是本轮发现实际 GGUF 采用了这些类型。** 实际编码字节数必须由 `ggml_row_size` 等类型规则计算。

仅用后一算例加 HC 与固定桥接卡 128K FP16 KV，即 `12.1875 + 1.171875 + 3 = 16.359375 GiB`，已经超过单卡 16 GiB，还没有放其他张量。因此不能预先保证 IQ4 名称的某个文件在 128K 可以运行。

### 4.4 KV 副本与角色轮换

本方案每个 full attention 层的 Q 角色为 `[5,5,5,5,4]`，KV 实例数为 `[1,1,2,1,1]`。12 个 full attention 层、单序列 128K、K/V 均 FP16、头维 256 时：

\[
M_{KV}=N_{layers}\cdot T_{ctx}\cdot2_{K,V}\cdot N_{KV}\cdot256\cdot2_{bytes}.
\]

原来全机约 3 GiB；采用副本后全机约 9 GiB。固定角色会使桥接卡占 3 GiB，其余各 1.5 GiB。

建议为第 a 个 full attention 层使用固定映射：

\[
physical\_rank=(logical\_role+a)\bmod5.
\]

12 层后各物理卡持有的 KV 头实例为 `[14,14,15,15,14]`，相应容量为 `[1.75,1.75,1.875,1.875,1.75] GiB`。总量不变，最高单卡 KV 从 3 GiB 降到 1.875 GiB。这是本文的容量平衡设计；**不能把它记作每个同步阶段的耗时下降**。

角色表加载时固定，并写入 plan 和状态版本。prefill/decode 必须一致，不得每 token 轮换；分片权重、KV、WO 和 global head 映射同步轮换。测试版支持固定角色，用来排查轮换导致的绑定错误。

在前述条件量化例中，使用 1.875 GiB KV 后三项仍有 15.234375 GiB/rank。其余权重、indexer、工作区可能让它继续超限。显存不足的正确处理是降低已声明的上下文/工作区、采用已经验证的其他缓存格式或换权重；不是偷偷把权重退回 CPU 后继续标成纯 GPU。

### 4.5 其他不能漏掉的容量

GDN 基础状态按 FP32、36 层、48 个 128×128 状态矩阵计，全机为 108 MiB/序列，按头分片后分摊；共享状态、快照、hand echo 和 sequence slots 另计。卷积历史不是这个矩阵的一部分。

若 indexer 全量 token key cache 为 128 维 FP16，12 层、128K 的复制缓存约为 **384 MiB/rank**；FP32 则加倍。实际当前缓存类型、长度、padding 和 pooled 工作区应从内存构造器读取；不能仅因 indexer 权重小就忽略缓存。

HC 单个残差 FP32 是 40 KiB/token，B=1024 时是 40 MiB/rank。96 阶段固定 inbox 在 decode 只需约 4.69 MiB/rank，但若机械乘 B=1024 就接近 4.69 GiB/rank，prefill 必须换有界复用方案，见第 17 节。

## 5. 整体架构：复用框架，但扩展真正缺失的契约

### 5.1 分成五层

```text
GGUF metadata / tensor headers
        ↓
llama_tp5_plan：语义、源切片、头映射、状态归属、预算
        ↓
Qwen4EXP 逻辑图 + 明确的子层/HC/副作用边界
        ↓
meta backend：每 rank 子图、局部布局、PARTIAL 合并、collective 计划
        ↓
Vulkan backend：现有矩阵/attention/GDN + 新 collective + HC fusion
        ↓
既有 VkDevice/queue/buffer 生命周期；DMA-BUF mailbox；SYNC_FD
```

模型层不调用 Vulkan，也不操作 FD。通信层不知道“第几层 Qwen 的专家”。执行层收到的是已经校验的 tensor layout、collective 描述、ready 依赖和 consumer recipe。

### 5.2 旧 meta 描述的真实限制

`ggml_backend_meta_split_state` 使用一个 axis、每段每卡长度和重复次数；当前校验要求所有片段总长等于原 tensor 轴长。它可以表达很多连续分片，却不能直接表达“KV0 同时放三卡、KV1 同时放三卡”或任意 gather 后的 GDN 头顺序。[R3；R4]

因此需要一个兼容旧接口的显式布局扩展，而不是去掉 `ne_sum == tensor->ne[axis]` 的断言。去掉断言只会使加载偏移、view 和缓存写入的语义更模糊。

拟新增内部 `ggml_backend_meta_layout_v2`：

```cpp
// 设计草图；不要把此草图当作现有 ABI。
struct ggml_meta_span {
    int32_t source_axis;
    int64_t source_begin;
    int64_t length;
    int64_t destination_begin;
};

enum class ggml_meta_value_kind {
    mirrored,        // 每卡同一份完整值
    disjoint,        // 最终值的不相交片段
    partial_sum,     // 每卡均覆盖完整输出域，但只是部分贡献
    indexed_replica  // 显式源范围/头编号，允许有声明的副本
};

struct ggml_meta_rank_layout {
    ggml_meta_value_kind kind;
    std::vector<ggml_meta_span> spans;
    std::array<int64_t, GGML_MAX_DIMS> local_ne;
    std::vector<int32_t> global_heads;
    uint64_t contribution_domain; // 哪些乘法贡献组成这个部分和
};
```

实际接口应放在内部边界，并给出版本与生命周期。旧 backend 不用新接口时保持原有路径。布局扩展不只是存一份 vector：权重装载、数据上传、tensor view、reshape、cache get/set、FA/GDN lowering、回读重建都要消费同一个描述。

### 5.3 图切分不能过早归约

当前 meta 源状态查询使用 `assume_sync=true`，并据此建立同步子图。不能假定 routed down 和 shared down 的 PARTIAL 会自然在卡内合并后才归约。[R3：约 790—1064、1815 以后]

应引入明确的局部线性区域：相同贡献域的 `PARTIAL + PARTIAL` 仍是 PARTIAL；`PARTIAL × MIRRORED` 的逐元素已知权重仍可保留 PARTIAL；不同中间通道的完整分片不能混成 PARTIAL。遇到非线性、归一化、需要完整输入的 router，或者区域指定的输出边界，才归约。

尤其不能把 `residual + partial` 在五卡各做完后直接求和，那会把残差加五次。HC combine 必须消费已归约的 y，或者由专用融合消费五份 y_i 后只加一次 residual。

对 Qwen4EXP 首版，三个输出边界明确命名：`attn_output`、`linear_attn_out`、合并 routed/shared 后的 `ffn_out`。48 层主体应产生 96 个逻辑输出归约点；数量由实际图生成并断言，而不是计时器里写死 96 掩盖额外归约。

### 5.4 不启动第二套 Vulkan 设备

POC 自己创建五个 VkDevice 适合独立测试；生产路径必须使用现有 backend 的设备、队列、分配器、锁和提交状态。另建 VkDevice 会使权重/缓存句柄不可直接复用，打乱显存预算和同步归属。

`ggml-vulkan.cpp` 内有大量私有类型和 static 函数。拆分新 `.cpp` 时，先提取窄的内部桥接接口，不把整个 `vk_device_struct` 暴露到 llama 模型层。桥接只提供设备能力、tensor 子缓冲定位、生产者提交/完成事件、待消费等待、mailbox 管理和受锁保护的队列提交。

## 6. MoE：专家内 TP 的数学与程序设计

### 6.1 路由不变，切专家内部

给定完整输入 x，使用现有 router 得到选中专家集合 E(x) 和权重 p_e。保留当前 softmax、选中权重归一化、`expert_weights_scale` 及 tie-break 行为，不另写近似 router。

单个专家：

\[
z_e=\operatorname{SiLU}(G_ex)\odot(U_ex),\qquad y_e=D_ez_e.
\]

把 F=640 沿中间通道分为 `I_i=[128i,128(i+1))`：

\[
z_{e,i}=\operatorname{SiLU}(G_e[I_i,:]x)\odot(U_e[I_i,:]x),
\quad y_{e,i}=D_e[:,I_i]z_{e,i}.
\]

于是 `y_e = sum_i y_{e,i}`。每卡中间结果是 128 维，但 down 输出是完整 H 维的部分和，不是 512 维结果片段。

共享专家还必须乘同一个输入产生的标量门：

\[
a_s=\sigma(w_s^Tx),\quad
y_i=\sum_{e\in E(x)}p_e y_{e,i}+a_s y_{shared,i},\quad
y=\sum_i y_i.
\]

共享 gate 的 sigmoid 不能漏掉；它在当前 `build_layer_ffn` 中确实存在。[R1：988—1032]

### 6.2 实现落点

plan 为 `ffn_gate_exps/up_exps/down_exps` 和 `ffn_*_shexp` 生成配对切片；修改 `llama_meta_device_get_split_state` 的架构专用适配，不能只依赖当前不覆盖所有 shared 命名的通用 regex。

继续使用 `build_moe_ffn` 和 `GGML_OP_MUL_MAT_ID` 的设备端专家选择。不允许每个专家一个 CPU 分支、每层将专家 ID 回读再重新构图，也不允许每专家 AllReduce。128 宽是否适合已有 Vulkan grouped kernel，应以真实 token 分布做小基准。

本地 routed 和 shared 输出保持 PARTIAL，先合并，再送一次公共 collective。若 debug 需要查看二者，应写调试副本，不为查看副本提前归约。

### 6.3 量化切片的合法性

gate/up 切的是输出行，整行输入长度仍是 H；down 切的是编码输入轴，起点与长度都必须满足实际块大小。读取同一个 GGUF 的每个 down 类型，不能根据文件名中的 IQ4 推断。

若 down 的量化块为 32，则 128 通道包含四个完整块；若实际要求 256 且不支持分块抽取，则这条 128 切法不能直接启用。首版明确报错，不静默反量化为 FP16、不截半个 block、不补假通道。后续可设计重新编码或不同分片粒度，但要重新做显存与精度验收。

重排按完整编码单元拷贝；同时处理专家轴 `ne[2]`、`nb[2]` 和 fused gate/up 的两个段。每个专家各自的 I_i 相同，不能把 512 个专家 flatten 后切成五段。

### 6.4 验收

先用固定路由检查每个专家的局部和，再接真实 router。构造所有 token 都选同一个专家的热点输入，确认五卡仍共同计算该专家；构造不同 batch 的路由，验证专家槽复用和累加清零。记录真实 dispatch 数、MUL_MAT_ID 形状和 fallback；均分 FLOPs 不等于 kernel 时间降低到五分之一。

## 7. QSA attention：头映射、缓存与输出投影

### 7.1 不补头的角色表

原始映射为 `kv(q)=floor(q/12)`。逻辑角色如下，之后才按第 4 节映射到物理卡：

| 角色 | Q 头半开区间 | 所需 KV | 本地 attention 调用 |
|---:|---|---|---|
| 0 | `[0,5)` | KV0 | 5Q / 1KV |
| 1 | `[5,10)` | KV0 | 5Q / 1KV |
| 2 | `[10,15)` | KV0、KV1 | `[10,12)` 的 2Q/1KV；`[12,15)` 的 3Q/1KV |
| 3 | `[15,20)` | KV1 | 5Q / 1KV |
| 4 | `[20,24)` | KV1 | 4Q / 1KV |

桥接卡的两次调用是为了使用现有等比例 GQA kernel，不是把原模型改成局部 5Q/2KV。后续显式 `q_to_kv[]` 融合 kernel 可以减少调用次数，但先不作为正确性闭环的前置条件。

Q 头最大数量相对理想 4.8 头是 `5/4.8-1 = 4.1667%`。这只是 Q 头数量的最大尾差，不是全模型统一性能系数。

### 7.2 Q 与 gate 的行定位

当前 Q 投影的每个全局头存 `[q_d通道, gate_d通道]`，然后下一个头。全局 q 头对应输出行：

```text
Q rows    = [2*da*q,     2*da*q + da)
gate rows = [2*da*q+da,  2*da*(q+1))
```

取整头时两段一起复制；归一化和 RoPE 只应用到 Q，不应用到 gate。attention 输出乘 `sigmoid(gate)` 后，使用 WO 中对应全局头通道的输入列。若本地调用先分两组再拼接，拼接顺序与 WO 的本地列顺序必须同表生成。

### 7.3 缓存必须与计算一起分片

各卡本地持有需要的 K/V 投影权重并更新副本，避免每步远程搬历史 KV。每层每个本地唯一 KV 头的 cache 写入只能执行一次；桥接角色的两个 attention 调用共享已经写好的 cache，不能各自再推进 cache cell。

所有 rank 使用同一 logical cell/position/sequence ID。局部物理头维可不同，但 token 轴必须一致。修改点包括 `llama-kv-cache` 的分配、get/set view、索引输入和状态读写；只改 QKV 权重不会自动让缓存变成本文布局。

保存计划中的 `(layer, global_kv_head) -> local_head` 映射。跨卡副本在 debug 中比较内容；生产不为此增加逐层回读。若不同 rank 的复制输入发生数值漂移，先解决通信表示和 router/indexer 确定性，不能容忍副本长期各算各的。

### 7.4 QSA 的计算保持当前语义

按当前 `build_qsa_top_k` 保留 indexer 的 query/key 归一化、RoPE、块均值、因果限制、ReLU/头合并及选择规则。索引选择属于非线性离散决策，不能把四个 indexer 头分到四卡后只各取本地 top-k 再拼起来。

本轮源码中的 `build_attn_qsa` 根据选中 cell 构造 mask，再调用 `build_attn_mha`；`build_qsa_top_k` 为绕过 Vulkan TOP_K 的 k 范围使用 `ggml_argsort_top_k`。这条路径应描述为“QSA 功能图”，不能自动称为已经跳过未选 KV 的稀疏 Flash Attention。[R1：525—760]

首版在各卡复制 indexer，保留相同 top-k 和 cache cell 对应关系。耗时和容量都要单列。增量 pooled-key cache、稀疏 kernel、索引广播属于后续优化，不能与 TP5 正确性改动混在同一验收里。

### 7.5 正确性检查

用每个全局 Q/KV 头具有不同数据签名的输入检测错映射；只用相同头值会漏掉这类错误。测试桥接角色、全因果与尾块、不整齐 context 长度、单次及分块 prefill、连续 decode。对照 attention 的 pregate、gate 后输出、WO 部分和、KV 新写行与 indexer 选择。

## 8. GDN：完整状态头与全局取模映射

### 8.1 切片表

逻辑 V/state 头范围先取 `[0,10)、[10,20)、[20,30)、[30,40)、[40,48)`。每个头包含完整的 ds×ds 递归状态、V、z、alpha、beta、norm 输出和对应的 ssm_out 输入通道。

选择偶数头数使 `ds × 头数` 适合 256 元素量化块的一类布局；这仍要以实际 down/out 编码校验。若未来改为 `[10,10,10,9,9]`，先证明量化和内核均支持，不能只按头数尾差决定。

### 8.2 Q/K 的映射是全局头号取模

当前 Qwen4EXP 图使用与 Qwen3.5 相同的重复模式；Vulkan `gated_delta_net.comp` 也明确使用 `head_id % neq1`。[R1：849—986；R8：108、417]

因此本例的全局 V 头 h 使用 Q/K 头 `h % 16`。这不是 `floor(h/3)`。当本地 V 头从全局 10 开始时，本地编号 0 应读取全局 QK10，不能再读 QK0。

首版采用预排列法：将每卡所需 Q/K 及对应卷积通道，按本地 V 头顺序排列为相同数量的本地 Q/K 头，使本地 kernel 的一一匹配仍然正确。或者在降层时显式提供 map；两者选择一种作为默认，不同时隐藏两套规则。

预排列增加 Q/K 投影与卷积历史副本。全机原先 16 个 QK 头可能变为 48 个实例；`Q+K+V` 的通道总数从 10240 增至 18432。它不增加跨卡状态交换，但增加本地权重、卷积和访存成本，必须进入预算和 profile。

### 8.3 递归数学契约

用每个头的状态矩阵 S 表示从 key 维到 value 维的映射，约定 S 为 `[key,value]`。令 q、k 是当前图完成卷积、SiLU、L2 normalization 及原路径要求的缩放后的向量，v 为 value，`b=sigmoid(beta_linear)`，`g=ssm_a*softplus(alpha_linear+dt_bias)`。单序列的等价递归可写为：

\[
\bar S_t=e^{g_t}S_{t-1},\quad
u_t=v_t-\bar S_t^T k_t,\quad
S_t=\bar S_t+b_t k_tu_t^T,\quad
o_t=S_t^Tq_t.
\]

这里不另增加缩放因子：query 缩放由现有 recurrent builder/kernel 的契约决定，并在参考测试中核对。实际 ggml 状态存储的轴方向可能与数学写法转置，必须通过 `ne/nb` 定位，不能直接按公式解释内存。

各头状态独立，所以完整头切片不需要跨 rank 交换 S。但同一头沿 token 时间有递归依赖；把不同时间段随意分给五卡不成立。prefill 的 chunk 算法也必须与同一递归等价，并连续接上前一 ubatch 的状态。

输出阶段使用当前 Qwen4EXP 的 `RMSNorm(o) × sigmoid(z)`，**不是其他架构常见的 SiLU(z)**。之后本地 ssm_out 产生全 H 维部分和，再归约一次。

### 8.4 状态对象与副作用

`cache_r` 卷积历史、`cache_s` 递归状态及 PLE history 不混为一个平面切片。使用 plan 为每类状态建立独立映射。原图有共享状态和 hand echo 路径；首版未适配时应拒绝这些模式，而不是强行走普通状态指针。

GDN 更新具有副作用。collective 出错后，不能重新执行整个子图来“回退”，否则状态可能推进两次。执行器必须知道提交前、已经提交、已完成和状态已发布的区别，失败处理见第 16 节。

验收先比较单头单步，再比较多步和分块 prefill；故意让每个全局头的初始 S 不同，验证切片、序列 copy/reset、保存恢复和角色轮换。只检查最终文本不能定位状态映射错误。

## 9. HC：先固定数学，再做 F1 / F2 / F3

### 9.1 原图的完整公式

对每个 token，残差是 `x[c,d]`，不是一条共同归一化的 10240 维向量。设 gamma 是 GGUF 中已经折叠好的 norm weight：

\[
r_c=\left(\frac1H\sum_{d=0}^{H-1}x_{c,d}^2+\epsilon\right)^{-1/2},
\qquad n_{c,d}=\gamma_{c,d}x_{c,d}r_c.
\]

将 n 按原图流优先顺序展平为 n_flat：

\[
\ell=\operatorname{SiLU}(D n_{flat}/C),\qquad
g=U\ell,\qquad
m_d=\frac1C\sum_c n_{c,d}\sigma(g_{c,d}).
\]

独立的注入分支为：

\[
j=J n_{flat},\qquad a_c=2\sigma(j_c/C).
\]

子层输出 y 完成后：

\[
x'_{c,d}=x_{c,d}+a_c y_d.
\]

这些分别对应 `build_hc_mix`、`build_hc_combine`。转换器已将 gamma 折叠为 `(1+w)`，融合不能再加一。代码使用 `build_lora_mm`，所以也不能在开启 LoRA 时擅自当作纯 W·x。[R1：266—335]

数学上可以按低秩维切 D 的输出和 U 的输入，得到 `g=sum_i U_i·SiLU(D_i n/C)`，但它要求在 sigmoid 前归约。A-F 不采用此通信布局。将输入和输出都除以五而不补交叉项，或把各卡 sigmoid 后的值相加，都不是原模型。

### 9.2 A-F 的数据路径

```text
完整 HC 残差（各 rank 复制）
        ↓
本地 grouped RMSNorm / gamma
        ↓
F2：完整 down → 1/C → SiLU；生成本子层 inject
        ↓
F3：完整 up → sigmoid → 乘 n → 四流平均
        ↓
完整 mixed[H,B]（各 rank 一致）
        ↓
本子层 TP：attention / GDN / MoE
        ↓
每 rank 的完整 partial[H,B]
        ↓
PUSH + 跨设备就绪依赖
        ↓
F1：求和 → 本子层 HC combine → 下一 HC 的 grouped RMSNorm
```

F1、F2、F3 是逻辑融合段，不强求只有三个 dispatch。一个段拆成两次高效 kernel，可能比一个寄存器溢出的“大融合”更快。先保持可对照的未融合路径，再逐段替换。

### 9.3 F3：最先实现的融合点

输入：完整 `lo[R,B]`、`n[H,C,B]`、up 权重。输出：`mixed[H,B]`。不必永久保存 `gate[CH,B]`、`sigmoid[CH,B]` 和 `gated[CH,B]` 三份中间张量。

计算单位按同一个隐藏通道 d 的四条流组织：

```text
for token t:
  for hidden d:
    mixed = 0
    for stream c in 0..C-1:
      gate = dot(U[c*H+d, :], lo[:, t])   // 完成整个 R 维点积
      mixed += n[d,c,t] * sigmoid(gate)
    output[d,t] = mixed / C
```

这是算法伪代码，不是推荐逐标量执行。Vulkan kernel 中由线程组协作读 lo、分担输出 d，并保留 FP32 累加。先试每线程或线程子组处理一组 d 的四个流输出，再比较每组多个 d；对 R=320，向量化、尾部和共享内存容量要单独处理。

权重原始编码行 `row=(c*H+d)`，行偏移用 `ggml_row_size(type,R)`，不是默认 `R*sizeof(half)`。可在加载时把完整行从 `[stream,hidden,rank]` 重排到 `[hidden,stream,rank]`，但必须保留编码单元和类型。不要为了重排改变权重量化。若存在旧布局和新布局双份驻留，把峰值列入第 4 节预算，并在转换后释放不再需要的副本。

首版至少有以下 guard：C/R/H 与 kernel 契约匹配；输入和权重类型在支持表内；无未处理 LoRA/bias/scale；strides 可解释；中间量没有其他观察者；输出 token 集合与原链一致。条件不满足则回原算子链，日志记录原因。

debug 模式允许额外写 gate 供逐项比较；性能模式不写。每次融合必须更新后端的读写依赖和 `fused_ops_write_mask`，不能只跳过图节点而让 allocator 提前复用仍在读的 n 或 lo。

### 9.4 F2：完整 down 的后处理和 inject

输入完整 n_flat，计算 320 个 down 输出，点积完成后先乘 `1/C`，再 SiLU，直接写 lo。初版保留现有 GEMV 的主体，只把后处理加进 epilogue，避免同时重写量化解码和融合。

inject 的 C 个输出也读取 n_flat。可实现共享输入的双投影 dispatch，逻辑输出为 `R+C` 个值；但是两个权重 dtype 不同或算法不合适时，保留独立 inject GEMV。实际可选择保存 raw inject 或已转换的 a，必须在类型/字段名中说明，不能让 combine 再做一次 sigmoid。

建议字段名：

```text
hc_raw_inject_current[C,B]  // 原图接口仍使用 raw 值时
hc_alpha_current[C,B]       // 已是 2*sigmoid(raw/C)
hc_alpha_next[C,B]          // 下一个子层自己的系数
```

当前子层的 alpha 必须到 F1 消费完才可覆盖。把 inject 放在 PUSH 期间计算和把 inject 融进 F2 是两种不同安排；不能同时记两份“重叠收益”。首版先选后者或独立串行分支，重叠放后续实验。

### 9.5 F1：不要一次跨越所有边界

F1 的通常输入是五份 `partial[H,B]`、当前残差 x、当前子层 alpha、**下一 HC** 的 gamma/epsilon；输出是更新残差 x' 和下一 n。先按固定 rank 顺序求 y，再完成 x' 和四个流各自的平方和与归一化。

至少定义四个路径：

| 路径 | 使用场景 |
|---|---|
| `sum_only` | collective 参考实现，或 y 有其他消费者 |
| `sum_combine` | 后面接 PLE、输出筛选或其他不能跨越的操作 |
| `sum_combine_norm` | 普通子层相邻边界，能够直接进入下一 HC |
| `norm_only` | 模型入口或 PLE 已完成之后 |

第一版先实现“公共归约写 y → 本地 combine+norm”。这一步可以复用旧 collective ABI，便于分离同步问题。第二步才实现“接收槽五路求和+combine+norm”，需要第 14 节的 fused consumer 契约。不能既在通信层求过和，又在 F1 将五份输入再求一次。

单 token 候选 kernel：每个流一个 workgroup，组内线程分担 H=2560 项，计算 x' 并累计平方和，组内规约后写 n。所有线程必须参加 barrier，越界线程使用零贡献，不能提前 return。只启动四个 workgroup 可能占用不足；同时保留多组 partial-sum 加第二个 norm dispatch 的版本，按完整 F1 耗时选择。

更新残差仍须落地，因为本子层 combine 后的残差是后续状态链的输入，也可能被图观察者使用。不能为了融合只写 n 丢掉 x'。

### 9.6 四个不能自动跨越的边界

**PLE。** 原图在该层 HC 前先执行 PLE。上一层输出归约后只能先 combine，执行 PLE，再 norm。把 norm 提前会改变输入。

**最后一层 `inp_out_ids`。** 当前图在最后一次 attention/GDN 后，先选择 cur、inject 和 residual 的输出 token，再做 combine/FFN。prefill 的 B 在这里可能从 B_all 变为 B_out。融合必须使用相同 selector，不能访问被裁掉 token 或错把 B_all 写入后续 slots。

**首层和最终 HC。** 首层 residual 由 embedding 复制 C 次，没有前一 y。最后一个 FFN combine 后还有 `hc_head_norm/down/up` 的最终 mix；它没有 inject，不能假装下一次普通子层。

**观察者和副作用。** `t_layer_inp`、`l_last`、embedding 输出、debug callback、imatrix/control vector 等可能要求中间结果存在。融合只消除确实没有独立使用者的值；不支持某个观察模式时拒绝融合，不能让观察 API 静默读错。

## 10. 入口、PLE、最终输出：96 次以外也要实现

### 10.1 Token embedding 与 LM head

推荐 plan 支持沿词表行分片的 token embedding 和 LM head：当前 token 的 embedding 行在一个 owner 上读出，分发完整 H 维到其余 rank；最终 mixed 在各卡完整复制，LM head 在各卡算本地词表区间的 logits。

这是输出轴分片，不需要对 logits 求和。首版收集完整 FP32 logits 到既有采样器，再使用原有全局 softmax/top-p、重复惩罚、grammar 和采样逻辑。不能用“每卡只给局部 top-k”代替完整接口后仍声称所有采样行为未变。

词表不整除时使用显式行范围；此轴不是权重的量化输入轴，但仍检查物理 row_size。tied embedding/output 必须保持权重共享和布局一致，不能多加载一份或错误共享不同 shard。若首个实现暂时复制 token embedding，应在 manifest 明示额外字节数，而非悄悄改变准入预算。

入口广播和 logits 收集都计入整 token 延迟，不能因为不属于 96 个子层归约就略去。不要在文档里固定词表容量的通信字节数，运行时根据实际 V、B_out 和 dtype 生成。

### 10.2 PLE 查表与历史

当前 `build_inp_ple` 生成 I32 rows，从 host 上的 `per_layer_tok_embd` 做 `get_rows`，再把结果展平；查表留在表所在 backend，只有结果跨到设备。[R1：1208—1231]

首版共享一次 CPU hash/查表结果，向五卡提供一致的 PLE embedding。结果宽度为 `ple_head_dim * ple_n_heads`，必须核对等于 PLE 投影期望的输入宽度，不能无条件写死为 H。表本身不复制到 GPU，也不因 TP5 五个子图执行五遍 token history 更新。

PLE GPU 部分首版复制，保留原图：key/value 投影；按流 RMSNorm；key 与 hidden query 的点积除以 sqrt(H)；带符号平方根与 sigmoid；value 按流加 gate；norm 后的膨胀因果卷积；SiLU；最终 `hidden + gated + conv_out`。这些操作不等于一个普通 embedding 相加。[R1：1233—1325]

卷积历史长度为 `(ple_conv_kernel-1)*ple_ngram_size`，逐序列维护。当前加载器只允许一个 PLE 层，且要求其为 recurrent 层；TP5 不绕过这些验证。EOS、序列 reset/copy、chunked prefill 和 token-history 提交必须在统一请求状态中执行一次。

### 10.3 CPU 工作并不只有 n-gram

还存在 tokenizer/输入准备、调度、SYNC_FD 操作、采样及可能的日志。所谓“权重和 KV 全在 GPU”不等于 TP5 的 CPU 时间为零，也不等于 host staging 不可能发生。性能版对非允许节点的 CPU fallback 计数并报告；无意的矩阵/attention 回 CPU 视为未达到首版目标。

## 11. 单阶段 PUSH collective 的准确契约

### 11.1 输入和输出

每 rank 提供 `partial[H,B]`，全部 H 维都是该 rank 的部分贡献。输出为每 rank 都能安全消费的：

\[
y_{d,t}=\sum_{i=0}^{P-1}partial_{i,d,t}.
\]

单阶段 mesh：每卡把自己的**完整** partial 发给其他四卡；每个接收者加自己的 canonical local contribution 后，固定顺序求和。没有第二次 AllGather。若改成发送 S/5，则已换成另一种算法，必须补齐 ReduceScatter 与 AllGather 的依赖及计时。

### 11.2 字节账本

`S = H * B * wire_bytes`。下面只算 payload，不含 padding、FD、提交和协议开销：

| 单 token decode | FP32 | FP16 |
|---|---:|---:|
| 一份完整 partial S | 10240 B = 10 KiB | 5120 B = 5 KiB |
| 向一个对端发送 | S | S |
| 每卡向四个对端发送 | 40 KiB | 20 KiB |
| 每卡接收 | 40 KiB | 20 KiB |
| 五卡总发送 | 200 KiB | 100 KiB |
| 96 个主体事件，每卡发送 | 3.75 MiB | 1.875 MiB |
| 96 个主体事件，五卡总发送 | 18.75 MiB | 9.375 MiB |

每个事件有 20 笔远端发送，96 个事件有 1920 笔；本地 self-copy 或 packing 另计。全网总发送已经把每份传输记一次，不要再加总接收来声称更大的“有效带宽”。

### 11.3 FP16 wire 必须统一本地与远端表示

```text
FP32 partial_i
    → 同一个转换 kernel 生成 canonical_wire_i
    → 本卡求和也从 canonical_wire_i 读取
    → 四份远端 PUSH 都复制 canonical_wire_i
    → 各接收者转 FP32，按 rank 0,1,2,3,4 顺序相加
```

禁止“自己用未舍入 FP32、别人读 FP16 副本”。否则各 rank 得到的 y 不一致，复制的 HC、router、indexer 和 KV 会开始分叉。求和顺序也不能按数据到达顺序决定。

### 11.4 单阶段不代表无同步

一次完整事件的边界是：`生产完成 → copy 可读 → 四路发送完成并 release → 接收端 wait/acquire → 求和/融合 → 消费完成`。只测 vkCmdCopyBuffer 的时间、或只等发送提交返回，都不是这条链的延迟。

PUSH 不要求 shader 高速读取远端显存：发送者 copy 到接收卡本地 mailbox，求和 shader 只读本地内存。远端 shader 任意访问是另一项能力，不是本方案前提。

### 11.5 对照算法

同长 S 的一般 AllReduce，可对照 ring 或直接 ReduceScatter+AllGather，后者每卡发送 `2(P-1)S/P = 1.6S`。mesh 为 `4S`，省交换依赖但多流量。不要把两者的 event 数与物理步骤混写。

对 B=1024 的 FP32 输出，S=10 MiB，mesh 每卡一次发 40 MiB；它不应自动成为 prefill 的最快选择。算法选择由消息大小和目标机完整依赖链测试决定；首版先保留数学/状态不变的替换入口，见第 17 节。

## 12. Mailbox、生命周期与显存所有权

### 12.1 逻辑地址

decode 正确性优先，每 rank 逻辑上有：

```text
inbox[epoch_slot][stage][sender_rank][token][hidden]
stage = 0 .. 2*L-1
```

同一 token 内 96 个 stage 使用不同范围。`GPU i → GPU j` 的目标固定为 `j.inbox[stage][i]`，源是 i 的完整 canonical wire，不再用两阶段 POC 的 `src_chunk=j`。

包括自身槽，FP32、B=1、单 epoch 的逻辑大小为 `96*5*2560*4 = 4.6875 MiB/rank`；FP16 为 2.34375 MiB。物理对齐和元数据另计。

### 12.2 物理对象建议

为每个有向卡对建立独立 mailbox，即 20 个远端接收对象，各对象内部装 96 个 stage 的范围；自己的一份 contribution 可以在非导出的本地 buffer。这样单一写者明确，也避免每个 stage 一个小 allocation 的数量膨胀。

第一次实现不做复杂的多发送者别名复用。每个 mailbox 由接收卡分配 device-local storage，并导出 DMA-BUF，由指定发送卡导入为可写目标。一个物理 allocation 的范围可以细分，但所有权和重用规则必须精确到有效范围；若实现只支持整 buffer 转移，则按整 buffer 生命周期组织，不能假装可独立转移其中尚在使用的片段。

stage stride 至少满足实际 storage-buffer offset alignment、copy offset/size 和所选访问方式要求；用 checked arithmetic 计算 `align_up(S,alignment)`，禁止 int32 字节偏移溢出。非连续 tensor 先通过已计时的 packing 生成规范线性 layout，不能把 `ggml_nelements*element_size` 当作任意 tensor 的连续字节数。

### 12.3 显式状态机

每个共享范围维护：

```text
AVAILABLE_TO_WRITER
  → WRITER_OWNED
  → WRITE_SUBMITTED
  → RELEASED_WITH_READY_EVENT
  → READER_ACQUIRED
  → READER_CONSUMED_AND_RELEASED
  → RECYCLABLE
```

对应数据至少包含 `epoch、stage、sender、receiver、generation、byte_range、writer_ready、reader_done`。相同地址不同 epoch 不是同一份数据。debug 可以加哨兵和 generation 校验，但不能用 shader 自旋检查一个标志来代替 Vulkan 同步。

decode 首版在整 token 的所有消费者结束后才回收 epoch，包括最后一次 y 的消费和共享范围 release。主卡已经得到 logits，不等于其他 rank 已经结束。使用两份 epoch buffer 也不自动合法，复用仍需完成依据。

### 12.4 release/acquire 与外部域

独立物理设备的外部共享不能假定一个本地 VkMemoryBarrier 已解决所有权。`VK_QUEUE_FAMILY_EXTERNAL` 代表的队列要求相同物理设备和驱动版本；`VK_QUEUE_FAMILY_FOREIGN_EXT` 可表达不同设备，但支持更严格，需要实际能力路径确认。[V6]

对本方案的有向 mailbox，初始化时先建立可供发送者取得的外部状态；发送者 acquire → write → release 到适用外部域并 signal；接收者等待对应完成事件后 acquire → read → release；下一 epoch 才重新取得。使用带 buffer、offset、size 和 queue family 的 barrier，不能把不同物理设备的 queue family 数字直接当作同一设备的两个队列编号。

这部分必须按目标驱动支持的外部内存路径写成共享资源协议并测试。能力不满足时禁用 fast path，而不是删 ownership 操作“试着跑”。普通本地 barrier、跨设备 semaphore、所有权转移各管一部分，三者不能互相替代。

## 13. SYNC_FD：提交依赖，而不是逐层等 CPU fence

### 13.1 初始化能力查询

每个目标设备查询：external memory FD、DMA-BUF handle type 对指定 buffer usage 的可导入/导出性；实际 FD 的内存类型；binary SYNC_FD 的可导入/导出性；适用的外部 queue-family ownership；compute/copy 队列；所需 synchronization2 能力。

不要只查扩展名称。不要把 device group 的 peer-memory bits 当作这条独立 VkDevice DMA-BUF 路径的证明。不要默认同型号、同驱动的不同 GPU 有相同 deviceUUID。[V1—V7]

### 13.2 导入内存的检查顺序

```text
创建导入端 VkBuffer，携带相同 handle type 与所需 usage
    → 查询目标 VkMemoryRequirements / dedicated 要求
    → vkGetMemoryFdPropertiesKHR(actual_fd)
    → compatible = buffer.memoryTypeBits & fd.memoryTypeBits
    → 核对导出 allocation 大小、目标要求、绑定范围和 alignment
    → 从交集中选合法 memoryType；无解则返回明确错误
    → 按要求建立 import / dedicated pNext 链
    → allocate / bind；成功后完成 FD 所有权转移
```

记录逻辑 payload 大小和实际 allocation 大小两个字段。不能将 S 直接当作导入 allocation 大小，也不能随便把大小取 max 后认为两个设备就一定兼容。[V7；R6]

### 13.3 一道输出边界的提交顺序

对 stage s，所有 rank 先提交本地计算、wire 转换、四路 PUSH、release 和 source_done 信号。**五个生产者提交完成以后**，主机逐个导出事件并建立消费者等待。它不等待 GPU 完成本 stage。

```text
for each rank i:
    submit(produce_i + push_to_four_peers + release, signal=done[s][i])

for each producer i:
    fd = export_once(done[s][i], SYNC_FD)
    for each consumer j != i:
        import_temporary(dup_or_signaled_minus_one(fd), wait[s][j][i])
    close_original_if_valid(fd)

for each rank j:
    submit(next_consumer_graph,
           waits=wait[s][j][all_remote_senders])
```

这是提交层伪代码，不是完整 Vulkan 实现；每个 submit 仍带准确的本地 memory barriers 和共享范围 acquire/release。自己的 contribution 依赖本卡队列顺序或相应本地 semaphore；四个导入 semaphore 管远端生产者。下一图的第一次相关读取不能早于这些 wait。

### 13.4 API 规则必须落实为代码

SYNC_FD 是 binary semaphore 的 copy-transference、临时导入载荷，不是跨物理卡永久共享的 timeline。导入带 `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`；`fd=-1` 合法，表示已完成事件，不能对它调用 dup。[V2、V4]

导出前必须已有提交的 signal operation，且它依赖的 signal 也已提交；可以是 pending，不必先等待。copy-transference 导出时不能另有 queue 正在等待这个源 semaphore。源 done 对象只用于导出，不同时当作本地其他队列的 wait 对象。[V3]

每个生产者事件导出一次，再为四个接收者复制 FD。不要对同一次 signal 连续导出四遍。每个成功导入的 FD 所有权转给 Vulkan；失败或未转移的 FD 由应用关闭。导入目标 semaphore 不能仍被未完成 queue command 使用。[V4、V5]

预先分配 `done[stage][rank]` 和 `wait[stage][consumer][producer]`，避免每层创建对象；但**每个新 epoch 的事件载荷仍要重新导出/导入**。对象复用必须等上次使用完成。生产使用 checked `dup`/`F_DUPFD_CLOEXEC` 和 RAII，不能泄漏句柄或在导入成功后双重 close。

### 13.5 预录的准确含义

可以预录每个 stage 的稳定 command buffer，复用 pipeline、descriptor 和已导入的显存对象；SYNC_FD 的载荷和 wait/signal submission 仍按本轮更新。

不能预录一个完全没有中间跨卡等待的巨大 command buffer，然后只在头尾等 fence，声称它等价于 96 道正确依赖。SYNC_FD 也不能提前代表任意未来尚未提交的 signal。第一版按 stage 提交，先保正确；降低 host 提交成本是另一个明确的优化任务。

当前 Vulkan 图提交还有节点数/FLOPs 阈值和较弱设备的超时保护。TP5 接线不能删除这些保护来追求“一次 submit”。即使每个逻辑 stage 内分多次本地提交，其最后的 release/signal 仍须覆盖真正的生产者。

### 13.6 主机成本账本

按 96 道边界直接全互连转发，每 token 有 480 次生产者事件导出、1920 次消费者事件导入，另有 FD 复制、关闭、图提交和锁操作。实际 final consumer、入口/PLE/LM head 的事件另外计数。

这些调用可能成为关键路径，必须在 end-to-end 计时范围内。没有 `vkWaitForFences` 不等于没有 CPU 参与，也不能由导入成功推出纳秒唤醒或固定的几十微秒 collective 延迟。

第一版由一个调度线程按 rank 顺序提交，保证 VkQueue 外部主机同步和可追踪性；多个线程并行提交是后续优化。不得在一条线程持有 rank0 队列锁时等待另一线程持有 rank1 队列锁的操作，统一锁顺序或不跨队列持锁。

### 13.7 本地 barriers 的模板

| 依赖 | 源 stage / access | 目标 stage / access |
|---|---|---|
| 本地输出给 copy | COMPUTE_SHADER / SHADER_WRITE | TRANSFER / TRANSFER_READ |
| 本地 copy 给求和 | TRANSFER / TRANSFER_WRITE | COMPUTE_SHADER / SHADER_READ |
| F1 写 n 给 F2 | COMPUTE_SHADER / SHADER_WRITE | COMPUTE_SHADER / SHADER_READ |
| F2 写 lo 给 F3 | COMPUTE_SHADER / SHADER_WRITE | COMPUTE_SHADER / SHADER_READ |
| device 回读给 CPU | TRANSFER / TRANSFER_WRITE | HOST / HOST_READ |

外部 mailbox 的行还要加适用的跨设备 event 和 ownership 操作，不能只套第二行。使用 stage 支持的 access bits；不要把 SHADER_WRITE 放在只有 TRANSFER 的源阶段里。非 coherent host 内存还需按 `nonCoherentAtomSize` 对齐 flush/invalidate；coherent 也不免除执行依赖。[V8]

### 13.8 真正的等待测试

独立 SYNC_FD 测试不能只导入一个已经 signaled 的 FD 后返回成功。让生产者先执行有限工作，再写本轮新数据并 signal；消费者在生产者未完成时就提交 wait 后的校验 kernel。全部数据按 index/epoch 变化，并人工延后一个生产者。

不在导出前等待 producer fence，不用无限自旋制造“延迟”，不依赖一个只能打印 VK_SUCCESS 的探针。测试必须证明消费者读取的是本轮写入值，且同步后继续生成下一轮输入。

## 14. Vulkan collective 与 F1 的具体接线

### 14.1 首先实现现有三个 registry 入口

当前 `ggml/include/ggml-backend.h:233—235` 定义：

```cpp
typedef void * (*ggml_backend_comm_init_t)(ggml_backend_t *, size_t);
typedef void   (*ggml_backend_comm_free_t)(void *);
typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void *, ggml_tensor **);
```

在 Vulkan registry 中接入对应过程名，meta 的构造、执行和析构就可使用该路径。`comm_init` 只接受已经通过能力检查、同一 Vulkan 实现、rank 顺序确定的 backend 集合，创建共享 transport/context；`free` 等待或确认全部在途资源退休后释放。

初版 `allreduce_tensor` 只接受连续、同 shape、支持 dtype、预分配目标的输入，完成 mesh + 固定顺序本地 sum，输出仍按旧接口约定写回各 tensor。对不支持的尺寸，在**没有提交任何副作用前**返回 false 让旧 fallback 处理；性能严格模式应禁止此 fallback，返回可读错误。

异步返回 true 的含义不是“数据已经在 CPU 看来完成”，而是“后续同 backend 的合法消费者必定通过已建立的依赖读到结果”。必须把外部 wait/acquire 安装到消费队列或当前提交链。单纯发出 PUSH 就返回 true 是错误实现。

### 14.2 bool 接口的错误边界

已经向部分 GPU 提交后再返回 false，会让 meta fallback 重新计算/归约已被修改的数据；对 GDN 还可能伴随状态推进。这种失败不能走旧的“unsupported”分支。

建议增加内部 v2 状态：`unsupported_before_submit`、`queued`、`failed_after_submit`，并向上转为 `ggml_status`。旧 bool 入口只用于能保持原契约的操作，错误必须进入 backend error 状态，禁止继续消费；不把运行失败伪装为“不支持”后重放。

### 14.3 完整 F1 需要一个明确的新契约

普通 allreduce 入口只知道输入输出 tensor，不知道下一 HC 的 residual、alpha、gamma、selector。因此不能在这个函数里凭 tensor 名猜下一层，然后偷偷做 combine。

拟新增版本化内部 `ggml_comm_consumer_recipe`，由 meta 在完成每 rank 图 lowering 后生成：

```cpp
enum class ggml_comm_consumer_kind {
    sum_to_tensor,
    qwen4_hc_sum_combine,
    qwen4_hc_sum_combine_norm
};

struct ggml_comm_consumer_recipe {
    uint32_t version;
    ggml_comm_consumer_kind kind;
    uint64_t graph_generation;
    uint64_t stage_id;
    ggml_tensor * residual;
    ggml_tensor * alpha_current;
    ggml_tensor * gamma_next;      // 无 norm 模式时为 nullptr
    ggml_tensor * residual_out;
    ggml_tensor * normalized_out;  // 无 norm 模式时为 nullptr
    ggml_tensor * optional_sum_out;// 其他观察者需要 y 时保留
    int64_t H, C, B;
    float epsilon;
};
```

实际执行函数还应收到每 rank 的 recipe、源 tensor、wire 类型、buffer 范围和 dependencies，不是只收到这一张结构体。其生命周期属于一次已准备的 graph plan，所有 tensor 与 generation 校验一致。模型层仅标注数学边界，不能保留后端异步事件裸指针。

融合可用时，meta 将后续 combine/norm 节点标为由该 recipe 实现，同时保留图输出 tensor 的真实 producer 和 allocator live range。融合不可用时，退化为 `sum_to_tensor` 和原图；不能在普通 consumer 仍等待读 y 时只写了 n。

### 14.4 队列桥接的最小职责

拟提取 `ggml-vulkan-internal.h` 中的窄接口，职责包括：

```text
inspect_device_caps()
resolve_tensor_span(tensor, expected_generation)
flush_producer_and_get_completion()
allocate_export_import_mailbox()
enqueue_peer_push_and_release()
attach_external_waits_and_acquire()
enqueue_sum_or_hc_recipe()
retire_epoch_resources()
```

名称是拟定职责，不是当前 API。所有队列操作遵守原 backend 的 mutex/queue lock；tensor 的 `view_offs` 和 buffer offset 都要加，不能只拿 `tensor->data` 当设备地址。新 transport 不自行销毁原 backend 的 queue、device 或权重 buffer。

### 14.5 缓存、重录和动态输入

graph/command cache key 至少含：model/plan ID、graph generation、rank、H/C/R、B 与 B_out、layout/strides、weight dtype、wire dtype、fusion mask、状态模式、P2P/sync 路径、buffer allocation generation 和 pipeline specialization。

token ID、position、expert IDs、cache indices 优先进入稳定的 GPU 输入 buffer；不把每次值变化当成重新创建 pipeline 的理由。但 buffer 重新分配、selector 形状变化、cache 容量改变、模型/adapter变化必须使相关缓存失效。

reset command pool、复用 descriptor、销毁 imported memory 前，必须确认引用它的所有 submissions 完成。不能依赖“通常下一 token 会更晚”作为生命周期证明。

## 15. 模型图与 meta layout 的改造边界

### 15.1 不改原数学参考图

`src/models/qwen4exp.cpp` 保留现有 builder；仅为 TP5 增加可选的结构化 annotation 或架构专用 lowering 入口。annotation 明确 HC 输入输出、当前 inject、下一 norm、PLE、输出筛选、缓存副作用与子层输出合并点。

这些信息属于 `llm_graph_result` 或等效 graph-owned 对象；不能放进全局静态表以 tensor 地址作为永久 key。图重建后地址复用很常见，旧 annotation 会指向错误层。

### 15.2 layout v2 的完整消费链

| 层次 | 必须落实的行为 |
|---|---|
| 权重加载 | 按每 rank source spans 读取/拷贝；支持声明的副本、量化行和 scale |
| 元 tensor | 保留逻辑原 shape，同时记录本地物理 shape 和映射 |
| VIEW/RESHAPE/PERMUTE | 检查变换是否保留分片语义；不保留时显式 pack/gather，不能仅改 `ne` |
| QSA lowering | 产生角色表规定的合法 GQA 调用；唯一 KV 更新只出现一次 |
| GDN lowering | 按 global head map 重排 Q/K、V、状态、卷积和 out 权重 |
| cache get/set | logical cell 共用，本地 head layout 不同；状态读写按同一计划 |
| collective | 严格区分 PARTIAL 与 DISJOINT，前者求和，后者拼接 |
| 回读/保存 | 重建全局逻辑顺序；副本只序列化约定的一份或带明确冗余标记 |

先用 CPU 小 tensor 对每个变换测试，再接 Vulkan。允许非连续 span 不等于任意 GGML view 自动正确。

### 15.3 架构支持开关最后打开

`llm_arch_supports_sm_tensor(QWEN4EXP)` 不能在实现一开始直接改 true。阶段测试可通过内部 test-only factory 注入 TP5 plan，或者新增专门的实验配置；正式入口只有在实际型号、五卡、layout、状态与 operator 支持全部满足时才放行。

长期应将单一 arch boolean 补充为参数化 capability check，包含设备数、后端、实际 tensor 类型和请求功能。保留旧函数给现有调用者，避免让“QWEN4EXP 已支持”被误解成任意卡数/后端/量化组合都能跑。

## 16. 状态提交、失败和恢复

### 16.1 逻辑状态统一，物理 shard 分散

维护一个请求级逻辑 token/position/sequence 状态，由五个物理 cache/state shard 实现。每个 ubatch 拿到相同的 logical cell 索引与 sequence 操作；各 rank 只执行自己物理布局对应的读写。

保存 plan hash、rank role table、global head IDs 和状态格式版本。支持保存/恢复前必须完成全 rank 的状态提交；恢复时先解析、验证和分配，再统一发布，不能一张卡先改 live state，另一张卡失败后留下混合版本。

### 16.2 首版故障策略

初始化阶段不满足能力/预算：在加载大权重前拒绝，给出具体张量/设备/限制。

提交前不支持某个融合：可回原算子链，不改变状态。

提交后出现 Vulkan/FD/同步错误：标记当前执行和 context 不可继续，停止后续提交，收集已知完成事件与错误；只有已建立、经过测试的 token 边界快照才能用于恢复。没有快照时请求失败，不尝试重跑已含 GDN/PLE history 写入的图。

设备丢失或等待超时：返回诊断，不自动重启驱动、重置其他生产 GPU 或无限重试。资源回收遵守设备失效和在途引用规则，不能为了快速返回直接 free 正在被远端写入的 allocation。

### 16.3 取消与 graceful stop

取消令牌阻止提交新 stage，但已经提交的工作要在安全边界退休。只在全部 shard 对应同一个逻辑 token 边界时，才能继续请求或保存状态。状态检查至少记录最后提交/完成/发布的 `(epoch,ubatch,layer,subtype)`，不要只保留一个含糊的 `token_count`。

## 17. Prefill：相同状态布局，不同工作区与通信算法

### 17.1 正确性先于加速

首版 prefill 使用与 decode 相同的 Q/KV/GDN 角色表、量化切片和状态格式。可以暂时关闭 decode 专用的 HC GEMV fusion，走 B>1 的原算子图和已正确的 collective。不能用旧 layer split 完成 prefill 后，直接把另一种 cache layout 交给 TP5 decode。

必须支持 `prefill → 多步 decode`；单次大 prefill 与多个 ubatch 的结果、KV、GDN 与 PLE history 做对照。GDN 的时间依赖和 PLE 膨胀卷积历史要延续，不能每个 ubatch 清零。

### 17.2 工作区用有界 ring，不复制 96 份大消息

对 B=1024，固定 96 阶段 inbox 会占数 GiB/rank，不能照搬 decode 方案。prefill 采用少量 stage slots，例如 2 或 3 组，每次使用带 epoch/generation 的 reader credit：

```text
producer 获得可写 slot
    → 本地计算/packing
    → PUSH + ready
    → consumer 完整使用该 slot
    → release + reader_done
    → slot 才可归还下一阶段
```

reader_done 是全部相关读取结束的证据，不是发送者完成 copy 的证据。首个正确版本允许阶段性 host wait 来建立复用边界，但这部分计入 prefill 延迟，并明确不是最终异步实现。异步 ring 在专门的多 epoch 覆盖测试通过后替换。

显存峰值按 `n_slots * P * H * B * wire_bytes` 加 send/pack 临时值和实际 padding 计算，而不是按总 context 代替 B。

### 17.3 消息大小驱动算法选择

保留 `mesh_sum`、`reduce_scatter_allgather` 的统一输出契约。auto 决策表按 `(device topology, B, H, wire dtype, required consumer)` 建立，来自完整消息和完整同步的基准，不使用 1 KiB 探针外推。

首版 auto 可以只有两种显式静态选择，尚未标定时采用用户指定或安全参考值；不能运行中无日志更换算法。不同算法导致的求和顺序变化应纳入数值比较。

### 17.4 后续才做的计算通信重叠

有实际独立分支时可以重叠；有直接依赖时需把 GEMM 与通信拆成少量块。先试不分块、两块、四块，记录 GEMM 效率下降和新增 submit/event 成本。通信与计算共享显存带宽或队列资源时，时间线重叠未必降低延迟。

不能对完整输入尚未归约的下一 HC 直接开跑，也不能把多请求吞吐改善称为单请求下一 token 延迟下降。对单流 decode，真正目标是减少关键路径，不是让更多队列看起来忙。

## 18. 精度、确定性与回退

### 18.1 分开四种参考

| 配置 | 用途 |
|---|---|
| 小尺寸 FP64 CPU 公式 | 检查代数与索引，不证明量化或 GPU 执行 |
| 同 GGUF 的原有非 TP5 图 | 作为模型行为参考，记录原路径本身的误差与状态 |
| TP5 + 未融合 HC + FP32 wire | 分离切片、状态、路由和归约顺序的影响 |
| TP5 + A-F + FP32 / FP16 wire | 依次评价融合和通信舍入，不能同时打开后只测文本 |

本轮只做了第一类的小尺寸检查。后三类需要实现完成后在相应设备上运行，本文不标为通过。

### 18.2 FP16 通信的误差来自哪里

发送前舍入改变的是 `sum_i y_i` 到 `sum_i round16(y_i)`。接收端 FP32 累加不能恢复发送前丢失的精度。对无溢出、正常数值范围可用：

\[
|\delta y|\lesssim u_{16}\sum_i|y_i|+\text{FP32 求和误差}.
\]

若各卡部分和很大但强烈相消，最终 y 很小，相对误差可能很大。sigmoid 的导数上界 1/4 只说明某个局部算子的敏感度，不保证经过 HC、router、QSA 离散选择和多层递归后 logits 足够接近。

FP16 wire 测试包含大幅值、相消、subnormal、溢出到 Inf 和 NaN。遇到发送前超范围不能静默 clamp；首版明确拒绝该精度配置或报告执行错误。动态精度回退若以后实现，必须五卡对同一 event 一致选择，且转换/额外控制通信计时。

### 18.3 比较指标与门槛

每个中间张量记录 `max_abs`、带小分母保护的 relative L2、RMS error、有限值计数；router/top-k 另记 ID 差异、概率差异与边界间距；logits 记 top1 margin、top-k 排名与固定 token 序列上的 logprob 差异。

整数小幅值 collective 测试要求精确值，不放宽误差。对于浮点模型，先用同 dtype 未融合与重复运行的差异建立数值基线，再在打开新优化前固定阈值并写入测试配置。不能在看到失败后随意把阈值放大到通过，也不能用一个统一 `1e-3` 覆盖所有张量和上下文长度。

相同 wire、固定 rank 求和顺序、同一实现的复制路径应避免 rank 间漂移；不应要求 TP5 与未切分图全部逐位一致，因为点积及求和顺序已经变化。前者是分布式一致性问题，后者是浮点等价误差，两者不能混用同一个宽松门槛。

### 18.4 Teacher forcing 再自由生成

对照运行输入同一串 token，逐步检查 logits 和状态；不要让两条路径自行采样后分别继续生成，一旦 token 不同就无法定位最初差异。短序列通过后覆盖长 context、chunked prefill、持续 decode 和 sequence reset；最后再测自由生成、采样稳定性及任务质量。

## 19. 配置入口、诊断与可操作性

### 19.1 拟新增配置，不污染旧默认

建议新增版本化的 TP 配置对象，包含：

```text
version / struct_size
enabled_plan = qwen4exp_tp5_af
device_ids / expected_device_uuids
hc_mode = reference | af
wire_type = f32 | f16
sync_mode = host_reference | sync_fd
collective_mode = mesh | rs_ag | calibrated
role_rotation = fixed | layer_rotation
strict_gpu_residency = true
manifest_output
trace_output
prefill_slot_count / workspace_budget
```

布局和模型相关字段在加载前冻结；epoch/event 不放进用户配置。配置属于 model/context，不使用影响所有实例的进程级 static 全局变量。相同进程加载两个模型应各有计划、通信资源和错误状态。

现有 `llama_model_params` 是公开接口，不能无说明地改变旧调用方的二进制布局。建议添加一个携带版本化 TP 配置的显式加载入口，由现有内部加载实现接收可选配置；原入口传空配置，行为不变。CLI 只在用户选择 TP5 时使用新入口。实际 API 命名在实现提交中定稿，变更必须带头文件、默认值和兼容测试。

拟定 CLI：

```text
--tp5 qwen4exp-af
--tp5-hc reference|af
--tp5-wire f32|f16
--tp5-sync host|syncfd
--tp5-collective mesh|rs-ag|calibrated
--tp5-role-rotation fixed|layer
--tp5-manifest <file>
--tp5-trace <file>
```

以上当前均不存在。实现后仍使用仓库既有 device 选择，TP5 要求恰好五个合法 GPU；与 `split-mode layer/row`、非均匀且不受支持的 tensor_split、CPU 权重混放等冲突时明确报错，不静默忽略用户选项。

### 19.2 初始化日志要能定位错误

成功日志至少打印：源码/计划版本；模型元数据摘要；五个设备的 UUID/BDF/rank；各类分片和副本数量；逐卡预算；wire dtype；同步路径；HC 实际 fusion；实际 collective 个数；prefill slot 容量；回退次数。

失败示例：

```text
TP5_E_QUANT_SLICE:
  tensor=blk.7.ffn_down_exps.weight axis=0
  requested_begin=128 requested_length=128 block_size=256
  action=reject_before_weight_upload

TP5_E_EXTERNAL_SYNC_UNSUPPORTED:
  producer_rank=2 consumer_rank=4 handle=SYNC_FD
  required=IMPORTABLE|EXPORTABLE observed=<actual flags>

TP5_E_MEMORY_BUDGET:
  rank=3 static=<bytes> transient_peak=<bytes> reserve=<bytes>
  available_budget=<queried bytes> context=<tokens>

TP5_E_GRAPH_LAYOUT:
  layer=11 op=FLASH_ATTN_EXT
  local_q_heads=5 local_kv_heads=2
  reason=explicit_q_to_kv_lowering_missing
```

字段为格式示例，尖括号处运行时填写真实值，不能把本文预算算例当日志。错误中保留 tensor 名、层、rank、stage、generation 和原因，避免只暴露一个 ggml assertion。

### 19.3 回退策略

HC fusion 不满足 guard：可退未融合同语义 GPU 算子链，并计数。

P2P/sync 不满足：实验参考模式可选择已声明的 host baseline；严格性能模式初始化失败。不能运行到一半退 host staging 却继续给出“Vulkan P2P TP5”标签。

特殊模型功能不支持：在激活该功能时拒绝 TP5 或使用用户明确选定的旧完整路径。不能自动改模型输出、丢 LoRA、取消采样约束或切换缓存格式。

## 20. 文件级落点与提交拆分

以下“改”表示拟修改已有文件，“新”表示拟新增。今天只生成本文，不表示这些文件已被创建。

| 任务 | 文件/符号 | 具体内容 | 单独通过条件 |
|---|---|---|---|
| T01 | 新 `tools/tp5/tp5-inspect-model.py` | 不初始化 backend 的 header/metadata manifest | 无权重读取、无 GPU 分配；字段与 GGUF header 一致 |
| T02 | 新 `src/llama-tp5-plan.h/.cpp` | 不可变 plan、量化粒度、角色轮换、预算、hash、结构化错误 | CPU 小 tensor 的全覆盖/无错位测试 |
| T03 | 改 `src/llama-model.cpp::llama_meta_device_get_split_state` | 架构专用 plan adapter，显式 shared/HC/indexer/embedding 规则 | 无隐式默认造成的错误复制或额外分片 |
| T04 | 改 `ggml/include/ggml-backend.h` 与 meta 内部 | 兼容旧 layout 的 v2 扩展、版本与生命周期 | 旧模型/旧 backend 单元测试不变 |
| T05 | 改 `ggml/src/ggml-backend-meta.cpp` | 显式 span 上传/回读、view lowering、PARTIAL 区域和 reduction 边界 | 单个 MoE 只归约一次；残差不乘五 |
| T06 | 改 `src/models/qwen4exp.cpp`、模型/图声明 | 结构化 HC/PLE/selector/子层 annotations，保留参考图 | 关闭 TP5 时图语义及输出不变 |
| T07 | 改 `src/llama-kv-cache.*`、`llama-memory-hybrid-idx.*` | 本地 KV/indexer layout、全局 cell、读写与序列化 | 唯一 KV 写入；桥接调用共享同一 cell |
| T08 | 改 `src/llama-memory-recurrent.*` 及 GDN lowering | 状态头映射、conv/history、提交版本 | 单步/多步/分块与参考一致 |
| T09 | 新 `ggml/src/ggml-vulkan/ggml-vulkan-p2p.h/.cpp` | 设备能力、DMA-BUF 分配导入、所有权、RAII、mailbox | 有向卡对传输和生命周期测试 |
| T10 | 新 `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp` | mesh/参考 RS+AG、wire 转换、SYNC_FD、epoch | 动态输入五卡完整消费链通过 |
| T11 | 改 `ggml-vulkan.cpp`、新窄内部头 | 三个 registry 接口、生产提交和消费 wait 接线 | meta 真正进入 native collective，无 staging |
| T12 | 新 shaders `tp5_pack.comp`、`tp5_sum.comp` | 规范 wire、固定顺序 FP32 求和 | wire 模拟参考逐元素通过 |
| T13 | 新 shader `qwen4_hc_up_fold.comp` | F3 | 与原 up+sigmoid+multiply+mean 比较 |
| T14 | 新 shader `qwen4_hc_down.comp` | F2 epilogue，optional inject | down/lo/inject/alpha 分别比较 |
| T15 | 新 shader `qwen4_hc_combine_norm.comp` | 本地 F1；随后扩展 sum consumer | combine/四个 norm 统计正确 |
| T16 | 改 Vulkan fusion 与 meta consumer recipe | 联合 collective+F1、live range、write mask | 无双重求和；debug/selector/PLE guard 生效 |
| T17 | 改 `ggml/src/ggml-vulkan/CMakeLists.txt`、`vulkan-shaders-gen.cpp` 等 | 编译并注册 shader 变体，避免运行时 shell 编译 | 干净构建可复现，无固定 `/tmp` 文件竞争 |
| T18 | 改 `common/common.h`、`common/arg.cpp`、加载 API | 新配置解析、冲突检查、manifest/trace 输出 | 默认关闭；旧调用方保持兼容 |
| T19 | 新 CPU/GPU/模型测试及相应 CMake | 第 22 节测试矩阵 | 正确分类 PASS/FAIL/SKIP |
| T20 | 最后改 `src/llama-arch.cpp` capability 入口 | 正式允许已验收的配置 | 架构、后端、形状、状态与全模型门槛齐备 |

`ggml-vulkan.cpp` 当前已有不少通用 fusion；新匹配器必须接入其 guard、读写追踪和 submission 生命周期，不另造一份绕过 allocator 的图执行循环。

POC 父级 CMake 当前在非动态 backend 的分支加入 `vulkan-p2p`，而子目录直接 `find_package(Vulkan REQUIRED)`。新增 CPU-only plan 测试时，要避免把这类硬件依赖强制带进通用构建；必要的 CMake guard 修正作为独立兼容提交，不与模型算法混改。[R9]

## 21. 开发顺序与每一步的退出条件

### P0：冻结证据与容量，不运行大模型

输入：本文 HEAD、实际 GGUF header、目标五卡设备清单。产物：manifest、rank role map、逐卡容量表、支持/拒绝列表。

先确认目标 GGUF 中 MoE down 的实际类型、HC 类型、缓存类型和 model metadata。若缺少模型文件，只能产出模板与条件预算，不能标为模型准入通过。当前开发机不具备五卡测试条件，目标机能力采集列为待执行项。

退出条件：每个 tensor 的去向可解释，所有切片可重建逻辑值，全部预算列出数据来源。不能通过增大 `--ctx-size` 反复试到 OOM 来代替预算。

### P1：CPU layout 和算法参考

完成 T01/T02 与 layout v2 的纯数据部分；用小尺寸 synthetic tensors 覆盖 partition、replica、interleaved Q/gate、global GDN map、量化边界与角色轮换。

退出条件：不存在空洞、未声明重叠、非法量化半块和错误输入/输出轴；五卡结果重建与单份逻辑参考一致。P1 不需要 GPU。

### P2：通信正确性与 SYNC_FD

先复用 POC 地址逻辑构建修正过的主机同步参考，再实现 mesh-SUM 的 2560 元素 FP32 路径；补齐 external memory/ownership；然后只改变同步方式接 SYNC_FD。

退出条件：所有 rank 所有元素正确；延迟生产者、不同提交顺序、多 epoch 槽复用仍正确；完整消费链、FD 泄漏、错误路径均通过。没有五卡硬件时此阶段是 SKIP/未验收，不是通过。

### P3：单卡 HC fusion

按 F3 → F2 → 本地 F1 的顺序，先独立于 collective。测试实际 H/C/R 和真实权重类型；debug 输出保留所有中间量，数值通过后再关闭中间写回。

退出条件：每个融合 guard 有正例和拒绝例；原路径可回退；注册/编译无需运行时 `system(glslc)`；性能报告区分单投影、投影对、完整 HC。

### P4：逐种子层的五卡图

先 MoE，再 GDN，再含桥接角色的 QSA；每种都连上 HC 参考和真实输出 collective。不要一开始用完整 48 层的最终文本调试。

退出条件：每个子层局部结果求和正确；shared gate/专家权重没有遗漏；KV、GDN、PLE 副作用执行次数准确；实际一个输出边界只有一次 collective。

### P5：完整模型与 prefill/decode 状态连续性

接 embedding/PLE、最后层输出 selector、最终 HC、LM head 和采样；完成 prefilling 后连续 decode。采用 FP32 wire、参考 HC 或已单独通过的 fusion，先不混入新的重叠优化。

退出条件：teacher forcing 中间量与 logits/state 通过；多 ubatch 和单次 prefill 对照通过；反复 sequence reset 无残留；无意 CPU fallback 为零；实际峰值满足逐卡预算。

### P6：A-F 的生产接线与提交优化

接 collective consumer recipe，实现完整 F1；加入 command/pipeline/descriptor 复用；量化所有 FD 和 host submit 成本。然后独立开启 FP16 wire，完成第 18 节质量验收。

退出条件：相同路径端到端延迟改善可重复；正确性不回退；没有资源泄漏、跨 epoch 覆盖和隐藏 host staging。未通过的某个 fusion 可以关闭，不要求把所有候选一起设成默认。

### P7：prefill 算法选择和可选重叠

实现有界 stage slots、完整 reader_done、消息大小标定；再比较两/四块计算通信流水、indexer 优化等。

退出条件：完整 prefill 吞吐或单请求 decode 延迟有实际改善，而不是只减少 copy 时间或提高某张卡利用率。质量和状态测试全量复跑。

阶段依赖：`P0 → P1`；通信 P2 与单卡融合 P3 可由不同实现者独立推进；`P1+P2 → P4`；`P3+P4 → P5/P6`；P7 最后。每阶段形成可审阅的小提交，不把性能开关和能力放行提前合到主分支。

## 22. 测试矩阵与具体断言

### 22.1 CPU plan 测试（拟 `tests/test-tp5-plan.cpp`）

覆盖 H/F/heads 不整除、quant block=32/256、大小溢出、零尺寸、未知 dtype、scale 缺失、五卡角色轮换、错误 rank 数、重复设备 ID、旧 plan hash 和不兼容状态版本。

对每个权重 shard 用唯一整数标记原元素，拼回全局值；副本应恰好等于源值。GDN 用 `1000*global_head + element` 标记；Q/gate 用不同区间签名。验证 role rotation 后恢复的逻辑模型完全不变。

对 PARTIAL 规则至少测：两局部分支相加只归约一次；加完整残差不得重复累计；partial 经 sigmoid 必须先归约；partial×完整 scalar 可延后归约；DISJOINT 只能按声明顺序拼接。

### 22.2 Vulkan collective（拟 `test-vulkan-tp5-mesh`）

基础测试 rank i 输入全 i+1，所有元素等于 15，失败返回非零。这仅是第一项。

增加 index/stage/epoch 都变化的可精确表示整数输入，避免相同 chunk 的常数掩盖错位；接收未写区填 NaN/哨兵。每次检查全部 rank 和全部元素，报告前若干个错误后保留总错误数。

连续依赖链可采用有界递推，例如令下一轮输入由上一轮结果缩放到有限范围再加确定性 rank/index 项，CPU 按同一顺序模拟。不能简单连续相加 96 轮使数值爆炸；也不能每轮重用同一静态输入假装检查依赖。

尺寸覆盖 256 B 级、5/10 KiB、20/40 KiB，以及 prefill 的 MiB 级；包含非对齐逻辑长度、padding 和不同 stride 的 packing。每种 dtype 都比较“先模拟 wire 舍入，再 FP32 求和”的参考。

同步覆盖：人为延迟任一 rank 的有限生产工作；改变提交顺序；单次与 96 轮依赖链；多个 token 的槽复用；取消；不支持外部 handle；导出/dup/import 分别失败；过期 generation；不足显存；设备数小于五。

FD/资源测试记录测试前后打开句柄数、Vulkan allocation 数、descriptor/command 数和内存占用趋势。对象预分配不是免测泄漏的理由。性能模式不在每轮回读，但正确性压力测试在适当边界全量检查最终和中间状态。

### 22.3 HC（拟 `test-vulkan-qwen4exp-hc`）

逐项比较：y、x'、四个 sumsq/rstd、n、down 原始输出、lo、raw inject/alpha、gate 和 mixed。覆盖 eps、不同 gamma、零输入、大幅值、相消、B=1 与 B>1、C/R 不支持时回退、非连续输入、debug 中间量观察。

专门测试以下负例：重复加 1 的 gamma；把四条流合成一个 RMS；用 next alpha 更新 current residual；在 sigmoid 前漏求和；跨 PLE 提前 norm；最后层 B_out != B_all；最终 HC 错误读取 inject。

### 22.4 子层与模型（拟 `test-qwen4exp-tp5-layer` / 集成测试）

MoE 检查专家 ID/weight、共享 sigmoid gate、热点专家、专家槽清零和同类型不同量化块。QSA 检查桥接卡两个 GQA 调用、selected indices、mask、Q/gate interleave、RoPE 维度、KV 写入一次。GDN 检查 modulo 映射、完整状态矩阵、conv history、gate 类型和多步状态。

全模型使用固定 token fixtures，测试短 prompt、跨 ubatch、长 context、prefill 后 decode、sequence reset/copy、保存恢复（实现后）、中途取消与恢复策略。融合开关、wire 精度、固定/轮换角色都作为独立矩阵维度，避免一次改变多个变量无法归因。

没有目标 GGUF 的测试不能叫“目标模型验收”；没有五张卡的测试不能叫“TP5 性能通过”；仅数学测试通过不能叫“Vulkan 同步通过”。测试系统明确输出 SKIP 及缺失前提。

## 23. 性能测量：一条路径一笔账

### 23.1 四级基准分别回答问题

| 层级 | 输入与范围 | 不能外推成什么 |
|---|---|---|
| GEMV | 指定矩阵形状、类型、工作集 | 不是完整 HC 或模型耗时 |
| HC | norm、down、inject、up、fold、combine 按约定范围 | 不是 TP5 整层加速 |
| 完整子层 | 五卡局部计算、通信就绪、HC 消费、状态写入 | 不是完整 token，仍缺入口/最终层等 |
| 整模型 | 输入准备到可交付输出，固定采样与上下文 | 用户实际延迟/吞吐依据 |

“96 次 GEMV”要使用 96 组不同权重，或相同容量/访问特性的 synthetic 工作集；反复读一块可能驻缓存的矩阵不代表全模型。区分逻辑权重字节、理论访问字节与硬件计数器得到的 DRAM 流量，未采集计数器时不称“物理读取”。

### 23.2 测量字段

```text
source_commit / plan_hash / model_identity
device_uuid[] / driver / kernel / topology
test_scope / H,C,R,B,context / unique_weight_bytes
weight_dtype / activation_dtype / accumulation_dtype / wire_dtype
warmup_iterations / measured_iterations / dispatch_count
collective_event_count / physical_copy_count / payload_bytes
fd_exports / fd_imports / queue_submits / host_waits
host_e2e_us / device_phase_us[] / p50,p90,p99
per_rank_memory_peak / fallback_count / mismatch_count
```

真实模型测试另记录 prompt tokens、generated tokens、batch/ubatch、已用上下文、cache 类型、冷/热 PLE 状态、sampling 配置、是否包括首 token 和输出收集。

### 23.3 计时规则

主机端使用单调时钟覆盖提交、FD 操作、GPU 执行、必要等待和消费。GPU 时间戳只比较同设备内可解释的区间，按该设备 timestampPeriod/有效位处理；未校准的不同设备 timestamp 不能直接相减。

独立预热，不把预热次数写进测量循环。所有 Vulkan 返回值检查。query 结果在正常完成边界统一读取，不在每个 kernel 后插 fence 来测性能。validation layers 的正确性结果与关闭 layers 的性能结果分开保存。

至少保存多轮原始结果、p50 与尾延迟，避免只展示最好一次。慢卡取实际 critical path；五卡执行时间不能平均后当作 stage 时间。

### 23.4 A-F 的总时间模型

\[
T_{token}=T_{input+PLE}+T_{body\ critical\ path}
+T_{finalHC+LMhead}+T_{sampling}.
\]

主体逐 stage 分为本地 TP、packing、PUSH/ready、F1、F2、F3、调度空隙。F1 已经做接收端求和时，不再在“完整 AR 时间”里加同一次 sum kernel。多队列重叠不能简单把这些分项全相加，应同时保留独立运行和并发运行的时间。

没有完整 trace 时只使用条件模型：

\[
T_5\approx T_{replicated}+\sum_s\max_i T_{sharded,s,i}
+T_{communication,critical}+T_{host,uncovered}+T_{other}.
\]

不使用 `原时间/5`；不把 Q 头 4.17% 尾差乘到所有算子；不把切走的 HC 投影从账上消失。即使总 FLOPs 均分，较小 GEMM 的效率、KV 副本和 indexer 复制仍然影响时间。

### 23.5 A-F 是否值得默认启用

依次比较：同 TP5 通信/布局下的未融合 HC → 仅 F3 → F3+F2 → 加本地 F1 → 完整 joint F1 → 可选 FP16 wire。最后与同目标机上可运行的旧 layer split 做用户可感知延迟比较。

只把有可重复端到端收益、质量无回退的优化加入默认。某个 fusion 更慢时允许关闭；“代码更少”“dispatch 更少”“通信事件更少”都不是独立的上线理由。

## 24. 构建与运行步骤

本节命令是给实现阶段的操作说明，**本轮未执行**。执行前确认不是生产实例目录、目标 GPU 空闲且得到测试许可。不重启现有服务、不启动第二个占满显存的模型、不为测试修改内核白名单或 DMA mask。

### 24.1 先做不触碰 GPU 的工作

在仓库根目录确认基线和工作区，保留用户已有文件：

```bash
git rev-parse HEAD
git status --short
cmake --version
glslc --version
```

T01 实现后才运行：

```bash
python3 tools/tp5/tp5-inspect-model.py \
  --model "$MODEL" --ranks 5 --context "$CONTEXT" \
  --output tp5-manifest.json
```

此脚本的验收要求是 header-only；若实现调用了真正的模型 loader，就不符合 T01。

### 24.2 构建独立目录

```bash
cmake -S . -B build-tp5 -G Ninja \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-tp5 --parallel "$BUILD_JOBS"
```

`BUILD_JOBS` 由开发机可用资源设置，不沿用历史 44 线程。新测试目标加入 CMake 后，先只构建并运行 CPU plan 测试，再选单卡 HC、目标五卡 collective，最后完整模型。既有 `test-vulkan-p2p-allreduce` 是参考 POC，不自动等于新 `test-vulkan-tp5-mesh`。

### 24.3 拟定的测试执行形式

下列参数也属于拟实现测试接口：

```bash
# 小型 CPU 测试，不能要求真实五卡才能执行。
ctest --test-dir build-tp5 --output-on-failure -R '^test-tp5-plan$'

# 五卡目标机：先 reference，同一数据再 syncfd。
build-tp5/bin/test-vulkan-tp5-mesh \
  --devices "$TP5_DEVICES" --elements 2560 --wire f32 \
  --sync host --check-all --vary-input --rounds 96

build-tp5/bin/test-vulkan-tp5-mesh \
  --devices "$TP5_DEVICES" --elements 2560 --wire f32 \
  --sync syncfd --check-all --vary-input --rounds 96
```

`TP5_DEVICES` 必须由目标机实际设备枚举得到，不直接假设 Vulkan0—4 就是五张 RX 6800。第一张可能是集显，枚举顺序也可能变化。

全模型执行先用 `--tp5-hc reference --tp5-wire f32`，再单独打开 af/f16；模型参数、设备参数以实现后的 `--help` 和 manifest 为准。不要在尚未实现开关时复制一条“可直接运行”的虚假命令。

## 25. 完成定义与交付清单

### 25.1 功能完成必须同时满足

1. 实际 GGUF 的所有分片和副本有 manifest，量化块、scales、全局头与状态映射通过验证。
2. 五卡 collective 在完整消费依赖、多 epoch、调度扰动下全量正确；所有权、句柄和资源生命周期经过检查。
3. HC reference、F3、F2、F1 的逐项对照通过；PLE、输出筛选、最终 HC 和观察者边界没有被融合破坏。
4. MoE、QSA、GDN 的单层与全模型 teacher forcing、prefill/decode 连续状态测试通过。
5. 实际峰值不超过逐卡预算，无未声明的 CPU fallback；模型采样和输出接口没有被近似替换。

任何一项缺失，状态写“未完成/未验收”，不要用生成了一段看似通顺的文本替代。

### 25.2 性能完成再增加

保存同机、同模型、同上下文和 sampling 的基线及优化 trace；提交延迟、FD 操作、额外复制、最后 HC 和 LM head 都在账内；多轮结果稳定且质量无回退。报告 prefill token/s、decode ms/token 与 token/s、TTFT 的口径各自独立，不能拿一个代替另一个。

没有固定 token/s 门槛。本方案的性能目标是相对于同条件可运行基线的实测改善；用户最终期望的延迟可在获得目标机真实 trace 后作为独立产品门槛，不从历史带宽直接推导。

### 25.3 每个实现阶段附带的材料

每次交付包含源码提交号、配置/模型身份、manifest、运行命令、原始测试日志、错误统计、性能原始数据和仍不支持的组合。数学参考、Vulkan 正确性测试和整机性能报告分开存放。

不要覆盖本仓库原有 `解决方案.md`；它是另一份审查材料，不属于 TP5 实现产物。实现过程的 commit/push 由用户工作流决定，本文不授权自动提交或推送。

## 26. 后续优化的进入条件

### 26.1 HC 分片方案重新进入比较

只有 A-F 的新 profile 证明复制 HC 投影仍是主要瓶颈，且低秩通信/不同形状 kernel 有可靠数据时，再测 A-D/B/C。比较完整 HC critical path，不比较单一 copy。C 必须在 sigmoid 前求和；C-fold 必须让一个 owner 拿到四条流相同 hidden 区间后才折叠。

### 26.2 更细 Q 任务切分

`Q head × query-token block` 可以在 prefill 降低整数头尾差，但要保存对应 Q/WO 权重副本、完整 KV 可见范围和因果权重。先完成 `[5,5,5,5,4]` 的高效实现；不能为了削掉头数上的几个百分点，提前打碎所有 GEMM。

### 26.3 其他值得测试的方向

共享专家整体复制并与 routed 输出归约重叠；indexer 的增量缓存；真正按选中 KV 访问的 sparse attention；通信聚合/减少 FD 主机成本；HC 权重布局和适配真实 dtype 的 GEMV。每项单独提交、单独测量，不把“有异步队列”当作收益结论。

纯 remote memory flag 自旋、跨设备原子归约、永久 OPAQUE_FD timeline 等，不作为本计划的默认同步捷径。要尝试这些路线，先另写设备/驱动内存模型的完整证明和验证程序，不能用一次固定输入通过替代。

## 27. 实现者逐项检查表

### 开始写 kernel 前

- [ ] 实际 GGUF 的 H/C/R、专家形状、weight/scale 类型已导出。
- [ ] 每个 rank 的显存预算已通过，开发机与目标机报告分开。
- [ ] Q/gate、KV 副本、GDN modulo map 和角色轮换可由 CPU 测试重建。
- [ ] PARTIAL 与 DISJOINT 不混用，shared/routed 在本地合并。
- [ ] 支持范围、拒绝组合、旧路径兼容行为已写入配置。

### 开始测“异步延迟”前

- [ ] 测试有生产、写入就绪、归约、消费和下一轮依赖。
- [ ] 目标 buffer requirements 与 FD properties 已求交。
- [ ] external ownership 与跨设备 wait 都已表达。
- [ ] pending SYNC_FD 真实等待通过，不是只测已完成 FD。
- [ ] FD 所有权、minus-one、object reuse、epoch credit 无遗漏。
- [ ] 所有返回值检查，失败不重放有副作用图。

### 开始发布 token/s 前

- [ ] 不是用 96 次乘法代替真实模型链。
- [ ] 不把单投影 GEMV 时间称为全部 HC。
- [ ] 不把逻辑字节称为已测 DRAM 流量。
- [ ] 不重算 F1 中已经包含的 sum。
- [ ] 包含 PLE、最终 HC、LM head、采样及 host 提交成本。
- [ ] 冷/热状态、B/ubatch/context、dtype 和 fallback 明确。
- [ ] FP32、fusion、FP16 wire 的精度影响分别通过。
- [ ] 保存实际结果；未运行项目标为未验收。

## 28. 源码与外部规范索引

### 28.1 本地源码 R

以下行号对应本文 HEAD；相对路径也可直接用于搜索。

| 编号 | 路径与定位 | 支持的本文结论 |
|---|---|---|
| R1 | `src/models/qwen4exp.cpp`；25 起 hparams；157 起 tensors；266—335 HC；337—440 主图；525—760 QSA；763—847 attention；849—986 GDN；988—1032 FFN；1208—1325 PLE | 模型公式、边界、形状和状态依赖 |
| R2 | `src/llama-model.cpp`；336 架构拒绝；345—354 dry_run；370 起 `llama_meta_device_get_split_state` | 当前准入、加载行为、分片粒度/分段/轮换 |
| R3 | `ggml/src/ggml-backend-meta.cpp`；约 790—1064 状态传播；1640—1705 comm 接线；1815 图执行；2100—2265 fallback/native collective | 可复用框架与显式 layout/边界缺口 |
| R4 | `ggml/include/ggml-backend.h`；233—235 comm typedef；399 起 meta 类型；417 起 split state | 实际 ABI 与描述能力 |
| R5 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`；8981—9005 copy；16726 起读写追踪；18085 起 fusion guard；18624 graph_compute；20873 registry | host staging、现有融合/提交、缺失 comm 入口 |
| R6 | `pocs/vulkan-p2p/test-vulkan-p2p-allreduce.cpp` | 两阶段地址、host fences、全量检查、计时及待修资源问题 |
| R7 | `src/llama-arch.cpp::llm_arch_supports_sm_tensor`，1111 起，QWEN4EXP 约 1142 | 当前架构仍未开放 tensor 模式 |
| R8 | `ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp`，108 和 417 附近 | 全局/本地 head 取模语义 |
| R9 | `CMakeLists.txt` 217—227；`pocs/CMakeLists.txt`；`pocs/vulkan-p2p/CMakeLists.txt` | 现有构建入口和 POC Vulkan 依赖 |

### 28.2 历史附件 H

用户上传的 `ChatGPT-Prefill速度估算-20260912-2010.txt`，导出标题《Prefill速度估算》。关键范围：3893—4960 为最后一次 A-F 实施讨论；3840—3953 涉及 A-F/A-D GEMV 报告和口径修正；3117—3326 涉及 host fence、SYNC_FD 和 FD 生命周期；2571—2864 涉及 POC 寻址与同步审查。

该文件是方案背景和用户提供的历史证据，不是本轮生成的 GPU 测量报告。本文没有将其中后续未出现在当前源码的探针视为已实现。

### 28.3 官方模型与 Vulkan 资料

查询日期为 2026-09-12。链接采用代码形式，避免与本地源码路径混淆；实现时应再对照目标 Vulkan headers/驱动支持的版本。

M1：Qwen 官方模型 `Qwen/Qwen3.8-Flash-Next` 的 `config.json`，本轮检索可见配置提交标识 `34567a4`；并用官方 `Qwen3.8-Flash-Next-FP8` 配置核对主体尺寸。模型配置不是 GGUF 的逐 tensor 类型清单。

`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blame/main/config.json`

`https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json`

V1：外部 semaphore handle 的 UUID 兼容性表；OPAQUE_FD 与 SYNC_FD 不能互换。

`https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalSemaphoreHandleTypeFlagBits.html`

V2：导入句柄的 copy/reference 与临时/永久语义、`fd=-1`。

`https://docs.vulkan.org/refpages/latest/refpages/source/VkImportSemaphoreFdInfoKHR.html`

V3：导出 copy-transference semaphore 的 pending signal、binary 类型、已提交依赖等前置条件。

`https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreGetFdInfoKHR.html`

V4：成功导入后的 FD 所有权以及 semaphore 不得仍在队列使用中。

`https://docs.vulkan.org/refpages/latest/refpages/source/vkImportSemaphoreFdKHR.html`

V5：同步对象、执行/内存依赖和外部载荷规则的完整规范。

`https://docs.vulkan.org/spec/latest/chapters/synchronization.html`

V6：EXTERNAL 与 FOREIGN 队列域的设备/驱动限制。

`https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_queue_family_foreign.html`

V7：实际外部内存 FD 的兼容属性，以及外部 buffer 能力查询。

`https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryFdPropertiesKHR.html`

`https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceExternalBufferProperties.html`

V8：本地/跨队列同步例子与队列能力、硬件映射说明。

`https://docs.vulkan.org/guide/latest/synchronization_examples.html`

`https://docs.vulkan.org/guide/latest/queues.html`

## 附录 A. 可复现的 CPU 代数检查

下面的 Python 只检查小尺寸浮点代数、专家分片、角色轮换与字节公式，不加载模型，不使用 GPU。需要 NumPy。它不是 Vulkan 测试，也没有检查 GGUF 量化或 router/top-k 的实际实现。

```python
import numpy as np

def sigmoid(x):
    # 本测试输入幅度有限；生产 kernel 应使用既有稳定实现。
    return 1.0 / (1.0 + np.exp(-x))

def main():
    rng = np.random.default_rng(20260912)
    H, C, R, P, F, E, B = 40, 4, 10, 5, 20, 7, 3
    assert H % P == F % P == R % P == 0
    x = rng.normal(size=(B, C, H))
    gamma = rng.normal(1.0, 0.1, size=(C, H))
    down = rng.normal(0.0, 0.07, size=(R, C * H))
    up = rng.normal(0.0, 0.07, size=(C * H, R))
    inject_w = rng.normal(0.0, 0.07, size=(C, C * H))

    n = x * gamma / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    flat = n.reshape(B, C * H)
    low_raw = flat @ down.T / C
    low = low_raw * sigmoid(low_raw)
    gate = low @ up.T
    mixed = np.mean(n * sigmoid(gate.reshape(B, C, H)), axis=1)

    # 检查低秩 C 方案的线性恒等式；A-F 生产设计不启用此通信。
    low_width = R // P
    gate_tp = sum(
        low[:, i*low_width:(i+1)*low_width]
        @ up[:, i*low_width:(i+1)*low_width].T
        for i in range(P)
    )
    # F3 直接按同一个 d 的四条流计算，不保留 gated 中间张量。
    mixed_f3 = np.stack([
        np.sum(n[:, :, d] * sigmoid(gate_tp.reshape(B, C, H)[:, :, d]), axis=1) / C
        for d in range(H)
    ], axis=1)

    wg = rng.normal(0.0, 0.1, size=(E, F, H))
    wu = rng.normal(0.0, 0.1, size=(E, F, H))
    wd = rng.normal(0.0, 0.1, size=(E, H, F))
    p = rng.random((B, E))
    p /= p.sum(axis=1, keepdims=True)
    eg = np.einsum('efh,bh->bef', wg, mixed)
    eu = np.einsum('efh,bh->bef', wu, mixed)
    hidden = eg * sigmoid(eg) * eu
    expert_full = np.einsum('bef,ehf,be->bh', hidden, wd, p)
    width = F // P
    expert_tp = sum(
        np.einsum('bef,ehf,be->bh',
                  hidden[:, :, i*width:(i+1)*width],
                  wd[:, :, i*width:(i+1)*width], p)
        for i in range(P)
    )

    alpha = 2.0 * sigmoid(flat @ inject_w.T / C)
    residual_reference = x + alpha[:, :, None] * expert_full[:, None, :]
    residual_tp = x + alpha[:, :, None] * expert_tp[:, None, :]

    errors = {
        'HC lowrank linear identity': np.max(np.abs(gate - gate_tp)),
        'HC F3 mixed': np.max(np.abs(mixed - mixed_f3)),
        'expert TP': np.max(np.abs(expert_full - expert_tp)),
        'HC combine': np.max(np.abs(residual_reference - residual_tp)),
    }
    for name, error in errors.items():
        print(f'{name}: max_abs={error:.3e}')
        assert np.isfinite(error) and error < 1e-12

    roles = [1, 1, 2, 1, 1]
    kv_instances = [0] * P
    for full_layer_ordinal in range(12):
        for role, count in enumerate(roles):
            kv_instances[(role + full_layer_ordinal) % P] += count
    assert kv_instances == [14, 14, 15, 15, 14]
    assert sum(kv_instances) == 72
    print('rotated KV instances:', kv_instances)

    for first, last in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 48)]:
        qk_map = [head % 16 for head in range(first, last)]
        assert len(qk_map) == last - first
        print(f'GDN V[{first}:{last}] -> QK {qk_map}')

    for wire_bytes in (4, 2):
        S = 2560 * wire_bytes
        assert 96 * P * (P - 1) == 1920
        print('wire_bytes=', wire_bytes,
              'S=', S,
              'sent_per_rank_per_event=', (P - 1) * S,
              'sent_all_ranks_96_events=', 96 * P * (P - 1) * S)

if __name__ == '__main__':
    main()
```

本轮在独立 CPU 数值环境核对过上述核心代数：HC gate 的低秩分片最大绝对差约 `2.08e-17`，F3 mixed 差为 0，专家分片最大绝对差约 `1.73e-18`；均为小尺寸 FP64 随机输入结果。不同 NumPy/BLAS 的最后几位可不同。这些结果不能用于宣称 FP16 wire、实际量化模型或 Vulkan 同步已经通过。

## 附录 B. 计划文件的最小结构

以下仅表示字段组织，不是实际目标机 manifest。`null` 表示必须从实际 header/设备查询补齐，不能拿示例替代。

```json
{
  "schema_version": 1,
  "source_commit": "cbe8014c1947d78a9b7a593fc0f9ec92ac74cafb",
  "plan": "qwen4exp_tp5_af",
  "model_identity": null,
  "actual_model_validated": false,
  "ranks": 5,
  "devices": [],
  "hc": {"layout": "mirrored", "compute": "af", "reference_available": true},
  "moe": {"partition": "intra_expert", "channels_per_rank": 128},
  "attention": {
    "q_role_counts": [5, 5, 5, 5, 4],
    "kv_role_counts": [1, 1, 2, 1, 1],
    "rotation": "full_attention_ordinal_mod_5",
    "global_head_maps": []
  },
  "gdn": {
    "state_head_counts": [10, 10, 10, 10, 8],
    "qk_mapping": "global_v_head_mod_global_qk_count"
  },
  "collective": {
    "decode_algorithm": "mesh_sum",
    "wire_type": "f32",
    "accumulation_type": "f32",
    "expected_main_events": 96,
    "actual_main_events": null,
    "sync_capability_verified": false
  },
  "tensor_layouts": [],
  "state_layouts": [],
  "memory_budget_per_rank": [],
  "unsupported_features": [],
  "plan_hash": null
}
```

最终原则：**先使每个值的数学含义、物理位置和就绪条件清楚，再减少搬运与 dispatch；任何性能结论都回到同一条正确的完整执行链。**
