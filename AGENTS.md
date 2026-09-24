## 结论

**应该把主线改成“TP5 原生的多步 MTP 执行程序”，而不是继续给两个独立上下文拼出来的草稿路径调参数。当前约 72 tok/s，不能视为这套硬件上 MTP 的上限。**

但需要区分两件事：**“草稿指定五张卡”已经做过；“草稿真正进入 TP5 的设备端多步执行、状态管理和通信体系”还没有完成。** 前一种实现跑得慢，不能证明后一种没有收益。

我检查了 E5-2699A 上的工作树、《困惑.md》《TP5.md》、关键源码和现有分相日志片段。**本次没有改代码，也没有把历史测试冒充新的性能复跑。** 下面给出的是现场审计后的工程判断；没有可靠证据的计算、通信、信令细分，我不会编数字。

完整工单、源码索引和测试要求已整理成：[下载《TP5 MTP 现场审计与工程路线图》](sandbox:/mnt/data/TP5_MTP_现场审计与工程路线图_2026-09-24.md)。

---

## 一、先把账立正确：问题不只是 514 ms，也不是 target 已经没有优化空间

《TP5.md》中那次 `predicted_ms=2437.917`、25 个 cycle、完整正确输出 171 token 的请求，可以重排成下面这张**互不重叠**的账：

| 项目                       |          整次请求耗时 | 含义                     |
| ------------------------ | --------------: | ---------------------- |
| 草稿                       |  **514.051 ms** | 150 次串行单步，包含 GPU 等待和采样 |
| target 图区间               | **1700.078 ms** | 包含建图、分配绑定、输入设置、提交与等待   |
| target `llama_decode` 图外 |   **90.072 ms** | 图区间之外的准备、状态处理、输出提取等    |
| target 验收后处理             |   **95.056 ms** | 主要是逐行验收采样              |
| draft catch-up           |   **35.891 ms** | 对草稿上下文执行真实的补齐计算        |
| 计时边界残差                   |    **2.769 ms** | 不是固定“系统开销”             |
| **合计**                   | **2437.917 ms** | 与该请求计时一致               |

依据：`TP5.md`“不以‘额外开销’掩盖草稿与 target 账”段，以及对应的 `tp5-mtp-n6-granular-ledger-*` 日志。

这里有两个必须同时成立的判断：

**第一，仅处理 90 ms 和 95 ms 不够。** 即使把它们全部删掉，这个诊断样本也只到约 **75.9 tok/s**。

**第二，1885 ms 的 target 总时间不是 GPU 计算下限。** 它包含首图绑定、首次执行、图外接口、验收采样。不能先把这些装进 target 总账，再用“target 自己已经超过 1710 ms”证明只能大改计算核。

实际需要解决的是：

$$
T_{\text{请求}}=\sum_{\text{cycle}}
(T_{\text{draft}}+T_{\text{verify}}+T_{\text{catchup}}+T_{\text{接口}})
+T_{\text{冷税}}
$$

这次请求平均每轮 **97.52 ms**，实际公开输出平均 **6.84 token/轮**。若仍为 25 轮，要过 100 tok/s，就必须压到**含冷税平均 68.4 ms/轮以内**。

因此，合理方向是联合削减草稿、状态物化、验证接口和首图成本，而不是指望一个小补丁省出全部差额。

---

## 二、514 ms 里到底有哪些计算、通信和信令？

### 1. 已知的是“等待位置”，还不是“等待原因”

现有草稿细账是：

* `llama_decode` enqueue：**67.454 ms**。
* `common_sampler_sample` 外围计时：**445.386 ms**。
* hidden getter：**0.099 ms**。
* 其余：**1.112 ms**。

平均每步约 **3.427 ms**，其中 enqueue 约 **0.450 ms**，等待加采样约 **2.969 ms**。

源码 `common/sampling.cpp::common_sampler_sample` 明确先同步上下文，再开始内部 sampler 计时。因此：

> **445 ms 不能叫“CPU 采样成本”，也不能叫“全是可以删掉的同步水分”。它包含上一段 GPU 尚未完成的时间。**

真正需要消除的是：**为什么每生成一个草稿 token，都必须让高层 CPU 采样器返回一次，才能启动下一步？**

`common/speculative.cpp::common_speculative_impl_draft_mtp::draft` 当前就是：

```text
llama_decode 一步
    → CPU 等待并取得采样结果
    → CPU 构造下一步 batch
    → llama_decode 下一步
```

原生 device-hidden 只替换了 hidden 的交接方式，没有消除这条逐步控制链。

### 2. 草稿确实有实际计算，不能把“一层”理解成近乎免费

`src/models/qwen4exp.cpp::graph_mtp` 中，这一层包含 token embedding、EH 投影、HC 混合、完整注意力、MoE、输出混合与 LM head。

所以，514 ms 中确实有矩阵计算和显存访问；但现在没有同一请求、同一时间线上的证据，能告诉我们其中各占多少。

另外，**这个 MTP 草稿块走普通注意力；大量 GDN 前缀快照的问题主要在 target 验证侧。** 不能把两者的状态成本混为一谈。

### 3. TP5 的 P2 也不等于纯通信

我看了 `ggml-vulkan-collective.cpp::tp5_star_handoff`。当前 RELAY 路径实际包括：

```text
等待五卡 producer ready
    → CPU F32 归约
    → 写回五张 BAR
    → 发布 generation
    → GPU 消费者继续执行
```

现有代码已经有 `rank_ready_us`、`ready_skew_us`、`arm_us` 等字段。

因此，七行链中的 **8.256 ms P2** 不能直接叫“PCIe 传输成本”：里面可能包含慢 rank、CPU relay、发布和恢复等待。**46.257 ms compute 也只是计算命令段，不是纯 ALU 时间。**

正确的账应该区分：

| 分类      | 应测内容                                         | 不能混进去的东西             |
| ------- | -------------------------------------------- | -------------------- |
| 计算与本地访存 | 矩阵、MoE、注意力、状态更新、临时物化                         | 别的 rank 尚未完成造成的等待    |
| 数据传输    | logits/hidden/token、collective payload 的实际搬运 | 等 producer ready 的时间 |
| 控制与信令   | 提交、generation 发布、协议处理、消费者恢复                  | 被依赖的 GPU 计算本身        |

**不要把“CPU 在等 GPU”的同一段时间，同时算一次 GPU 计算、再算一次同步浪费。**

---

# 三、路线图：按激进到普通排序

## 路线 1：最激进，也最值得作为主线——把草稿纳入统一 TP5 执行程序

### 目标不是五卡参数，而是一个完整的执行闭环

我建议首版只覆盖当前明确场景：**单序列、当前 Qwen MTP、RELAY/F32、n=6、语义允许的 greedy**。其他组合继续走已有路径。

将执行层拆成：

```text
Program
    固定算子与通信拓扑、rank 布局、描述符计划、scratch 生命周期

RunInputs
    token、position、KV 位置、active rows、接受长度、generation

RequestState
    target 与 draft 各自的状态、有效前缀、取消与提交边界
```

然后让六步草稿成为一个有限步执行程序，而不是六次高层 `llama_decode → sampler` 往返。

**自回归依赖仍然存在，但不要求每一步都回 CPU 决策。** 投机解码的收益来自减少昂贵目标模型的串行调用，并不意味着草稿的因果依赖可以被取消。([Proceedings of Machine Learning Research][1])

### 第一项关键改造：设备端 token → embedding → 下一步

每一步应形成：

```text
分片 LM head
    → 每 rank 局部最大值与 global token id
    → 全局候选发布
    → 设备端 embedding lookup
    → 下一步 MTP
```

这里最重要的是：

**局部最大值与 token id 一起发布，不能先回读 index，再按 index 发第二次 logit 回读。**

两级 GPU reduction 没问题；**两轮依赖 CPU 结果的设备回读才是应该消掉的结构。**

首版完全可以复用现有 RELAY 线程：CPU 只合并五个小候选记录并发布 token，不再经过高层 sampler、全词表候选构造和逐 token decode 调度。这样仍然存在 CPU relay，但与现在的高层串行路径不是一个成本结构。

必须对拍不等长词表分片、最小全局 id 平局规则、异常值处理和零行输出。之前两趟回读版比 CPU 慢，只说明那个实现没有优势，不说明设备端选择没有优势。

### 第二项关键改造：共享执行资源，但不要偷偷共享模型状态

**不要复制六套 graph scratch。** 六份命令或参数不等于六份大临时缓冲；token、hidden 可以用小环形缓冲，计算临时区按实际生存期复用。

还有一个容易踩的具体坑：

> **不要直接设置 `ctx_other=target` 来偷共享执行池。**

当前 MTP 驱动会据此判断共享记忆模式，可能改变算法分支。应新增独立的执行资源共享契约，保留这个模型自身的 draft KV 所有权。

同样，不能因两个张量都叫 `shared_head` 就共享 target/draft 权重。两个 GGUF 的量化类型、内容和布局必须先逐张量核对。

### 第三项关键改造：有界设备控制，而不是危险的永久 kernel

可以预录有限六步，使用设备端控制缓冲更新 token、位置和有效步数。Vulkan 的间接 dispatch 支持执行时从设备缓冲读取工作组数量，可作为实现手段之一；实际扩展和设备能力仍须在这台机器上确认。([Vulkan Documentation][2])

EOG、取消和失效步骤必须同时约束状态写入与 collective 参与，不能某些 rank 退出、其他 rank 永久等 generation。

**验收门：** 高层逐 token 往返消失，scratch 不按 horizon 倍增，取消可恢复，且整请求优于同版本单卡对照。只证明 hidden 在 GPU 上、只证明五卡能输出正确 token，都不算完成。

---

## 路线 1 的重要子项：catch-up 做成 K/V-only，并接到 target 尾部

这是本次源码审计中值得单独强调的机会。

`common_speculative_impl_draft_mtp::commit()` 确实又执行一次 draft decode。它不是纯复制。原因是：

**候选 token 全部被接受，不代表用“草稿预测 hidden”构建的 draft K/V，等于用“target 真实 hidden”构建的 K/V。**

所以不能直接删除 catch-up。

但对这里的普通注意力 MTP 块，验收补齐需要的是正确 draft K/V。建议新增明确的 **CATCHUP_KV phase**，只完成必要的：

```text
正确 token + 对齐的 target hidden
    → EH / HC 前缀
    → K/V 投影、norm、RoPE
    → draft KV 写入
```

不应为了“沿用完整 decode 接口”，继续执行没有消费者的输出头和其他计算。

这里还需要一次验证：**列出当前零输出图实际执行的 dispatch。** 图构造源码看起来完整，不代表所有节点都会运行；不能在未看实际图前宣布已经找到了多少毫秒。

进一步，在 target 宽 hidden 产生后，K/V 补齐可以进入同一执行计划，与 target 词表头中相互独立的工作安排重叠。重叠收益与新增显存必须一起算。

对齐规则直接以当前 `workspace::commit_row` 为 oracle，逐个验 accepted=0…6；不要凭直觉手写“取第 a 行还是 a+1 行”。

---

## 路线 2：激进——GDN 改为 checkpoint＋精确重算，而不是每个前缀完整 snapshot

**这一项应该比继续尝试 n=10、Q3、挪草稿卡更早。**

`src/models/delta-net-base.cpp::build_recurrent_attn` 明确先让 `gdn_out` 包含多个完整状态，再把它们复制进 recurrent rollback bank。

现有 n10 相比 n6，每卡 target scratch 增加约 **416 MiB**；加载后草稿卡 GTT 也出现约 **19 MB → 978 MB** 的差异。不能把这些直接命名为某块权重迁移，但足以说明**状态保存策略和驻留预算必须一起重做**。依据：`TP5.md` 的 n10 scratch、allocator 与 GTT 记录。

### 推荐的正常路径

```text
保留起点 checkpoint
    → 正常完成候选块验证
    → 保留终点状态与少量恢复日志

全接受：提交终点
中途拒绝：从起点恢复到接受前缀
```

关键是：**拒绝时不一定要重跑整个 target。**

第一版可以保存 GDN 每步使用的必要输入，例如 k、v、g、β 等，用相同 F32 运算序列重新执行状态转移：

$$
S_t = F_{\mathrm{F32}}(S_{t-1};k_t,v_t,g_t,\beta_t)
$$

这样恢复的是 recurrent transition，而不是所有投影、MoE 和输出头。卷积历史另保留较小的前缀记录；普通注意力 KV 按有效前缀提交。

### 不要用平均接受率代替恢复概率

必须分别统计：

**真实模型拒绝、EOG、长度截断、取消。**

最后一轮因为结束而少接受一个候选，不等于发生了需要继续生成的昂贵回滚。若请求结束且状态不再复用，可以让槽位失效；若要复用 prompt/KV，就必须恢复或明确标记失效。

经济账应写成：

$$
\text{收益}
=
\text{少写完整快照及改善驻留的收益}
-
\text{恢复日志成本}
-
\sum_j P(\text{在位置 }j\text{拒绝})\,T_{\text{恢复},j}
$$

### 给工程师的实现阶梯

先做**直接写持久 snapshot bank，去掉重复大临时物化**；再做稀疏 checkpoint；最后做起点/终点＋日志。每一步都能独立验收，不需要一口气重写所有状态管理。

**验收门：** 每个拒绝位置、每层 GDN 状态、卷积历史、KV 有效范围、target hidden 选择、取消后的下一请求都要对拍。不能只看最终计数题正确。

---

## 路线 3：较激进——专门优化七行验证，不重新折腾成熟的单行 target

我同意不另开一条“纯 target 50 tok/s 再挤一点”的主线。但：

> **单行 target 已优化，不意味着七行 MTP 验证程序也已优化。**

当前 n=7 仍落在 `mul_mat_vec_max_cols=18` 的 direct-quant 路径。这里应该研究的是**实际七行形状下的权重复用、量化解码、寄存器压力、状态写入和通信边界**，不是简单把阈值改成 GEMM。

具体要求：

**普通投影**看同一量化权重是否真被多行复用，是否为七行重复解码；**MoE**先统计这七行到底共享哪些专家，不能假设所有 token 都复用同一专家；**HC/GDN**看能否减少中间物化和重复状态读写，而不只是减少 dispatch 数。

已有“dispatch 更少但整请求没更快”的融合试验不原样重做。每个候选必须先写出：它减少了多少真实字节、多少计算或多少关键路径依赖。

前三个 stage 偏慢也值得查，但要在相同算子组合、同 rank、同驻留下比较。不同 stage 的算子不同，不能把 4–5 倍时间差直接命名为 DPM 或迁移。

**验收门：** MTP 七行整周期变快，MTP-off 单行路径不回退；改变浮点归约顺序的核单列数值验证，不能把“输出 1..60 正确”当作位级等价。

---

## 路线 4：中等——首图可以研究 direct/recompute，但必须明确重算哪一层

你提出“不必执着 cache”是有价值的，尤其在**状态恢复层**，我明确支持先做 checkpoint＋recompute。

但首图这里要区分：

| 操作                                   | 判断          |
| ------------------------------------ | ----------- |
| 不保存所有候选状态，拒绝时重算                      | **优先路线**    |
| 首次图直接执行，不为一次性使用付完整 replay-cache 维护成本 | **值得做同构对照** |
| 全局关闭 graph reuse，每轮重新建图和绑定           | **不是建议方向**  |

现有首图约 100 ms 的主因是 **Meta tensor binding**，不是 backend fence。`TP5.md` 的分相记录中，binding 约 99–110 ms，backend sync 只有微秒级。

因此：

**只跳过 Vulkan CB cache，却仍然走原来的 Meta 张量展开和绑定，未必省掉这 100 ms。**

真正该改的是固定执行定义与运行输入的分离：

```text
固定：拓扑、rank-local 布局、绑定计划、合法缓冲范围
变化：token、KV 位置、active rows、scratch 起址、generation
```

首轮 one-shot、稳态 replay 都可以建立在这个契约上；不是缓存旧请求的状态，也不是拿旧 tensor 指针强行复用。

现有 phase 切换还会释放计算缓冲，导致下一阶段重新建立资源。应该给 draft 单步、target 七行、K/V-only catch-up 明确程序身份，而不是只按 token 数猜 phase。

**已有容量模式的 46 个未覆盖 dispatch 不能放宽守卫硬跑。** 要逐类证明 inactive rows 不读取非法输入、不写状态、不破坏通信参与者。

---

## 路线 5：普通，但已经有明确源码靶点——修 90 ms 图外回读

这是本次最具体的一个定位。

我追到了下面这条源码链：

```text
llama_context::output_reserve
    尝试 output device 的 host buffer，否则普通 CPU buffer

Meta::get_host_buffer_type
    各 rank host buffer 类型不同 → 返回 nullptr

Vulkan::get_host_buffer_type
    按 device 分别创建类型

Meta::get_tensor_async
    按 rank 依次调用 get_tensor_2d_async

Vulkan::get_tensor_2d_async
    目标地址不是本 device 的 pinned buffer
    → staging fallback
    → 函数内部立即 synchronize
```

对应文件是 `src/llama-context.cpp`、`ggml-backend-meta.cpp` 和 `ggml-vulkan.cpp` 的上述函数。

**所以，“async”这个名字并不保证五卡回读已经并行排队。** 这是非常具体的退化条件，比“在外层少调用一次 synchronize”更值得优先处理。

但仍须测它的动态命中次数，不能把全部 90.072 ms 都扣在这里。`mctx::apply`、状态提交、输出缓冲处理等也位于相关图外边界。

### 建议的修法

每个 rank 使用自己可识别的、持久的 pinned 输出 slot。先让全部 rank 的 copy 入队，再在真正消费结果的边界等待；之后拼接，或者直接按分片采样。

不要把不同设备的 host buffer type 强行返回同一个指针来蒙混过关；也不要只删除 staging 分支里的同步。那个同步目前保护了 deferred memcpy 和临时缓冲的生存期。

slot 必须有 owner、generation、未完成状态与退休规则。Vulkan 对执行顺序和内存可见性要求显式同步，不能把“地址可映射”当作“CPU 已经可以安全读”。([Vulkan Documentation][3])

**验收门：** 打印每 rank 的 pinned 命中、fallback、字节数、提交和等待次数，证明从逐 rank 阻塞变成全部入队后消费；随后确认时间没有只是从 decode 挪到 sampler。

---

## 路线 6：普通到中等——95 ms 后处理，重点改全词表物化和采样状态复制

这 95.056 ms 已经有足够明确的方向：

| 子项            |            耗时 |
| ------------- | ------------: |
| 验收采样          | **87.058 ms** |
| sampler clone |  **6.363 ms** |
| seq_rm        |      0.099 ms |
| 已计输出循环        |      0.121 ms |
| 其余            |      1.415 ms |

源码 `common/sampling.cpp::set_logits` 会在 CPU 上逐词表构造 `llama_token_data`；`common_sampler_clone` 还会复制整个 `cur` 候选数组。优化重点应该落在这里，而不是 seq_rm。

### 最强方案：七行只回小候选，不回完整 logits

以仓库记录的 248320 词表计算，七行 F32 logits 是 **6,952,960 字节**。

五 rank、七行、每行一对 F32 最大值和 32 位 token id，原始候选载荷只有 **280 字节**，另加对齐、header 和 generation。

注意：**这减少回读和候选物化，不等于不再计算整个 LM head。**

对语义允许的纯 greedy，先得到七行的全局最大 id，再一次确定接受前缀。第一版甚至可以只回读这几个 id，在 CPU 比较前缀；不必为了追求“全部在 GPU”而增加复杂性。

有 grammar、有效 penalty、logit bias、reasoning budget 或其他状态性采样依赖时，不能擅自改成七行独立 argmax，必须保留顺序状态语义或回退。

### 较小方案：CPU 也可以不构造整张候选表

严格纯 greedy 的路径直接扫描 logits/词表分片，避免先构造全词表候选再 top-k/sort。草稿 `p_min<=0` 的情形，也应检查是否仍做了无用途的概率和候选物化。

clone 则新增**验收专用的持久状态 checkpoint**：保存 RNG、grammar、history 等，临时候选 scratch 不复制。不要直接改变通用 clone 的可观察语义；`cur_p` 等指针也必须正确重绑。

**验收门：** 拆出真正 CPU 独占采样时间，确认候选构造、排序和 scratch 复制减少，而不是只把等待藏到别处。

---

## 路线 7：最后才是普通策略优化——自适应 horizon，而不是继续扫 n

在执行和驻留修好之前，我不会继续让工程师依次尝试 n7、n8、n10、n12、挪卡、Q3、Q2。

horizon 应依据**前缀存活分布和边际成本**选择：

> 再多猜一步带来的预计新增公开 token，必须值回新增 draft、target、恢复与内存压力成本。

高接受率不授权无限增加 horizon。n10 已出现显著驻留风险与同形核退速，先解决状态和执行布局，再决定是否加长。

此外，诊断中某个 MoE 算子的 `n=10` 是专家维度，不是 MTP horizon。所有性能表都应标明轴的含义，防止改错门限。依据：`TP5.md` 对 n6/n10 同形量化 GEMV 的记录。

---

## 四、不要再无目标试验：给整条路径一个可检验预算

下面是我建议采用的**工程目标，不是已实现成绩，也不是收益保证**：

| 项目                    |     目标预算 |
| --------------------- | -------: |
| 六步草稿闭环                | ≤10 ms/轮 |
| 七行 target 设备链，含链内通信同步 | ≤48 ms/轮 |
| 链外验证接口、验收和提交          |  ≤3 ms/轮 |
| catch-up 新增的非重叠关键路径   |  ≤1 ms/轮 |
| 整请求额外冷税               |  ≤100 ms |

若仍是 25 轮：

$$
25\times(10+48+3+1)+100=1650\text{ ms}
$$

对应约 **103.6 tok/s**。

这个预算的意义是：**每个工程任务必须说明自己在缩短哪一项，其他项是否恶化。** 不是声称这些数字一定能全部实现，更不是把几次不同进程的最好值拼成最终成绩。

实际实现应继续留裕量；周期数、拒绝分布变化时重新算账。

---

## 五、实施管理：先交证据，再交补丁

展示顺序按激进程度；实际落地可以拆成可独立验收的工单，不需要等一个巨型重写全部完成。

我会要求首先交付三份材料：

1. **514 ms 的同请求关键路径图。** 分清 GPU 运行、主机等待、真实 CPU 采样、传输与逐步控制开销。
2. **90 ms 的回读分支命中账。** 每 rank 的 host buffer、pinned 命中、staging fallback、字节数和等待次数。
3. **GDN 重算的状态与内存账。** 每个拒绝位置的恢复正确性、恢复成本、live allocation 和峰值驻留。

CPU/GPU 时间线需要校准后关联；不同设备原始 timestamp 不能直接拿来对齐。Vulkan 提供 calibrated timestamps 契约，但要检查机器支持并记录偏差。([Vulkan Documentation][4])

之后所有候选都必须经过：完整接受、各位置拒绝、EOG、取消后下一请求、零输出 catch-up、1→7→1 形状切换和非支持采样器回退。状态改动要对拍状态，不能只看最终文本。

正式性能仍坚持：**独占五卡、watchdog active、参考 F32/RELAY、未开 profiler、逐字正确、自然停止、171/171、server `predicted_ms<1710`。** 保留所有预先安排的 A/B 样本，不挑最快一条。

**最终建议很明确：保留成熟的单行 target，停止继续扫草稿参数；以“统一 TP5 多步执行＋精确状态重算”为主线，以“rank-local 回读＋批量验收”为可独立兑现的改进。现有五卡草稿的负结果应成为重做执行结构的证据，而不是给 MTP 潜力封顶的理由。**

[1]: https://proceedings.mlr.press/v202/leviathan23a "https://proceedings.mlr.press/v202/leviathan23a"
[2]: https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDispatchIndirect.html "https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDispatchIndirect.html"
[3]: https://docs.vulkan.org/spec/latest/chapters/synchronization.html "Synchronization and Cache Control :: Vulkan Documentation Project"
[4]: https://docs.vulkan.org/refpages/latest/refpages/source/vkGetCalibratedTimestampsKHR.html "https://docs.vulkan.org/refpages/latest/refpages/source/vkGetCalibratedTimestampsKHR.html"
# TP5 + MTP 现场审计与工程路线图

日期：2026-09-24  
工程：E5-2699A，`/home/kunweiz/atomic-llama-cpp-turboquant`  
审计对象：当前未提交工作树、`困惑.md`、`TP5.md`、关键执行源码、现有分相日志片段。

## 0. 结论与证据边界

主方向应从“给独立草稿模型调参数”转为“把 MTP 做成 TP5 执行程序的组成部分”。保留单卡草稿作为性能对照，不把它当最终架构；也不能把已经存在、但有逐步主机往返的五卡草稿当作融合版的性能上限。

本次没有修改仓库、没有启动新一轮模型性能测试、没有把历史数据说成此次复跑。已检查机器现场：检查时没有 llama-server，五卡 GPU busy 均为 0，eagle-gpu-watchdog 为 active。仓库 master 的 HEAD 为 `4e17c2faf`，但存在大量未提交改动，不能只用 HEAD 标识被审计源码或历史测试二进制。源码定位以下以函数名为准；历史临时计时器已有撤回，不能假定当前二进制能原样输出每个字段。

本报告最重要的新定位，是图外回读的完整源码条件链、MTP catch-up 的 K/V 专用化机会，以及将设备端多步草稿、精确状态重算、批量验证组成同一个路线，而不是重复已有负收益试验。

没有现成的可靠数字可以把 514 ms 精确拆成“纯 GPU 计算多少、传输多少、信令多少”。给出三个百分比会是编造。以下区分已测值、源码确定行为、待验证归因与工程预算。

## 1. 已有实测账：一个请求只能算一次

`TP5.md` 的“不以‘额外开销’掩盖草稿与 target 账”记录了一次开启分相诊断、内容正确、自然停止、171 个输出 token 的请求：`predicted_ms=2437.917`，25 个 MTP cycle。

| 互不重叠的项目 | 整请求 ms | 解释 |
|---|---:|---|
| 草稿 | 514.051 | 150 次串行单步，含等待与采样 |
| target 图区间 | 1700.078 | 含建图、绑定、输入设置、提交与等待 |
| target llama_decode 图外 | 90.072 | 尚未逐项测清，不可全算 CPU 计算 |
| target decode 后处理 | 95.056 | 大头是逐行验收采样 |
| draft catch-up | 35.891 | 对草稿上下文执行真实补齐计算 |
| 计时边界残差 | 2.769 | 不是固定系统开销 |
| 合计 | 2437.917 | 与该请求 server 计时一致 |

嵌套关系为：target 总时间 1885.206 = llama_decode 1790.150 + 后处理 95.056；llama_decode 1790.150 = 图区间 1700.078 + 图外 90.072。禁止把父项与子项重复相加。

草稿内部：enqueue 67.454 ms，`common_sampler_sample` 外围计时 445.386 ms，hidden getter 0.099 ms，其余 1.112 ms。每步平均 3.427 ms，其中 enqueue 0.450 ms、等待加采样 2.969 ms。`common_sampler_sample` 内部先同步，再开始自身的 CPU sampler 计时；外部包围计时和内部 sampler 计时口径不同。

后处理内部：clone 6.363 ms，sample-and-accept 87.058 ms，seq_rm 0.099 ms，已计 token 输出循环 0.121 ms，其余 1.415 ms。每轮后处理约 3.802 ms。不要去优化占不到一毫秒的 seq_rm/输出循环以解释这 95 ms。

图内部：build 2.238 ms，alloc 121.978 ms，submit+wait 1574.206 ms，set_inputs 1.357 ms，剩余约 0.299 ms 为边界。首次图 287.519 ms，后续同形图约 58–59 ms。

已有未开 profiler 的参考结果约 72 tok/s，例如 2370.901/2376.041 ms，见 `/var/tmp/tp5-restored-after-greedy-rejection-unprofiled.json`。上述 2437.917 ms 是归因样本，不能冒充未开 profiler 基准。

### 吞吐预算

该请求平均每 cycle 97.517 ms，公开输出平均 171/25 = 6.84 token/cycle。若周期数仍为 25，100 tok/s 要求含冷启动在内平均小于 68.4 ms/cycle。不能按每轮必出 7 个公开 token 计算，也不能把 accepted+bonus 的内部计数直接等同公开 completion_tokens。

仅把图外 90.072 ms 和后处理 95.056 ms 假设全部删掉，此诊断样本也只有约 75.91 tok/s。另一方面，target 的 1885.206 ms 含冷启动和接口开销，不是不可改变的 GPU 计算下限。两句话都成立。

以下是工程目标，不是预测或新实测：

| 目标项 | 建议预算 |
|---|---:|
| 六步草稿闭环 | ≤10 ms/cycle |
| target 七行设备链，包含该链通信与同步 | ≤48 ms/cycle |
| 设备链之外的验证接口、验收、提交 | ≤3 ms/cycle |
| catch-up 的新增非重叠关键路径 | ≤1 ms/cycle |
| 整请求额外冷税 | ≤100 ms |

若这些互斥口径均实现且仍为 25 轮，总时间为 25×(10+48+3+1)+100=1650 ms，约 103.64 tok/s。实际研发应继续留出裕量；若真实周期数、拒绝分布或边界改变，必须重算预算。

## 2. 正确拆分计算、通信、信令

### 草稿 514 ms

当前单卡草稿没有五卡草稿内部的张量归约；它仍有 GPU 计算、局部显存读写、logits/token/hidden 的主机与设备搬运、命令提交及完成等待。这个 MTP 不是单个微小矩阵：源码包含 token embedding、EH 投影、HC 混合、完整注意力块、MoE 和 LM head。

445 ms 的 sample 外围时间混合了“上一段 GPU 尚未完成”和“CPU 真正在处理候选”。GPU 运行的时间不会因为删掉等待函数而消失。应消除的是每个 token 都回到 CPU 决定下一次 decode 的控制依赖。

### TP5 的通信不等于 P2

源码 `tp5_star_handoff` 会轮询五个 producer ready，记录 rank-ready 时间，执行 CPU F32 归约，写五张 BAR，再发布 generation。P2 区间可能包括等待慢 rank、等待 CPU、传输和消费者恢复；它不是独立纯 PCIe 带宽账。计算命令段本身也包含 dispatch、屏障、访存和调度，不是纯 ALU 时间。

已有独立进程 GPU 诊断中，七行 rank0 compute/P2/total 为 46.257/8.256/54.523 ms；单行为 20.422/5.129/25.564 ms。七行链相对七次单行有摊销，但这些数不能与另一次请求的主机账逐项相减，也不能相加五个 rank 的 compute 当作请求延迟。

### 必须交付的观测格式

所有记录带 request_id、cycle_id、phase、step、rank、stage、rows、generation。CPU 使用互斥范围，GPU 用每 rank 的设备时间线；需要跨时钟关联时检查 calibrated timestamp 支持并记录最大偏差，不直接比较原始 tick。

需要区分以下边界：CPU prepare/enqueue；GPU 首次开始、各计算段、copy、结束；producer ready；最后一个 rank ready；CPU reduce/broadcast；generation publish；消费者开始；主机等待结束；CPU sampler 独占时间；accepted prefix commit。将等待归因到依赖的生产者，不能把全部等待再算一遍信令浪费。

记录 H2D/D2H 字节、每 rank staging fallback 次数、queue submit 次数、fence/timeline 等待次数、scratch 峰值、buffer 身份及显存/GTT。优先复用现有 `tp5_star_times` 的 ready skew、arm、publish 等字段。

记录放预分配内存，请求结束统一输出。先用阶段级采样保留 replay，不给每个 op 强插屏障。已有逐 op logger 会禁用 replay，并曾把整请求扰动到约 24 秒，不能拿它作为性能 A/B。

## 3. 路线一：最激进——统一 TP5 MTP 执行程序

### 3.1 结束两个独立主机驱动器逐 token 往返

现有 `common_speculative_impl_draft_mtp::draft` 每一步都 llama_decode → common_sampler_sample → CPU 构造下个 batch。原生 hidden 交接只换了 hidden 的传输方式，没有去掉这条控制链。原五卡草稿更慢，因此不能简单将设备列表改为五卡就宣布完成 TP5 化。

建议新增专用执行层（模块名为建议，不是已存在 API），形成 Program 与 RunInputs：

- Program：固定算子/通信拓扑、rank 布局、描述符计划、合法 scratch 区间、phase 定义。
- RunInputs：token、position、KV 位置、active rows、accept 长度、取消标志、generation。
- RequestState：target 与 draft 各自的有效状态和提交边界。

先覆盖单序列、当前 Qwen MTP、参考 RELAY/F32、n=6、语义允许的 greedy。其他组合走原路径。不要一开始把所有模型、采样器和并发情况卷入首版。

重要：共享执行池不等于共享 KV。不要直接把 `ctx_other` 指向 target 来偷共享资源；当前 MTP 驱动用它判断共享记忆分支，可能改变模型语义。应给执行 arena/命令资源独立的共享契约，保留本模型独立 draft KV。

### 3.2 TP5 是放置和执行策略，不要求每个微小算子切五份

大权重与 LM head 采用与模型分片契约一致的 TP5；小的归一化、门控和必要镜像可在 rank 内完成，避免为省极少 FLOPs 引入额外 collective。先用实际张量形状与字节数确定切法。

target/draft 量化权重不一定相同。只有逐张量证明类型、内容、布局和所有权一致时才共享权重；不能因名称含 shared_head 就直接 alias 两个 GGUF 的输出矩阵。

### 3.3 六步仍自回归，但不需要六次 CPU 决策

预录有限步程序：LM head → 本地候选归约 → 全局候选发布 → embedding lookup → 下一步 MTP。token id、hidden、positions 和 active 状态留在设备控制缓冲。

局部 argmax 一次产生 `(max_logit, global_token_id)`，不再先读 index、再为该 index 发第二次 logit 回读。五 rank 候选可由现有 RELAY 线程合并并写回各 rank；这仍有 CPU relay，但没有高层 sampler/fence/llama_decode 的逐 token往返。直接跨 GPU 方案必须先证明实际互操作路径，不假设有现成高速 P2P。

最小全局 id 的 tie-break、不等长词表切片、NaN/Inf 策略必须与参考一致。词表归约不是浮点求和归约，要有独立 payload ABI 和有效位。

不能把六个有依赖的草稿 token 改成六个并行 token；改变的是控制位置，不是模型因果关系。

### 3.4 不复制六份大 scratch

只保留必要的 token/hidden 环形缓冲和 per-step 参数，复用峰值 scratch。每步的命令/参数不等于每步一套模型 scratch。资源复用须建立真正的 live range 和退休条件；未完成提交引用的 descriptor、地址与参数不得重写。

GPU 侧可通过间接 dispatch 参数控制后续工作；feature 支持时也可评估 conditional rendering。无论采用哪种方案，EOG、取消和 active=0 都要阻止无效 KV/状态写入，而且 collective 的参与者必须一致，不能让部分 rank 退出而其他 rank 永久等 generation。不要用占满 GPU 工作组的无限自旋充当跨步同步。

### 3.5 catch-up 先做 K/V-only，再谈重叠

`commit()` 确实对 draft 再做 decode。即使候选 token 全接受，draft 中由预测 hidden 构建的 K/V 也不等于由 target 真实 hidden 构建的 K/V，因此不能直接删 catch-up。

本模型 MTP 块是完整注意力，不是 target 的 GDN。验收补齐的目标是正确 draft K/V；应新增明确的 CATCHUP_KV phase，计算必要的 EH/HC 前缀、K/V 投影、norm/RoPE 和缓存写入。先列出当前零输出图实际 dispatch，确认已有图裁剪到了哪里，不能仅看源码有完整 attention 就断言所有算子都执行。

以现有 workspace::commit_row 的 token/position/hidden 对齐为唯一 oracle，检查是否能消除无消费者的 Q/attention/output/MoE/LM-head 工作。不要凭 accepted 数手写 off-by-one 规则。

进一步可以在 target 宽 hidden 产生后，把候选前缀的 draft K/V 补齐挂到同一执行计划，和 target 词表头的独立工作安排重叠；只有经过验收的前缀可发布为有效。重叠会增加同时存活的缓冲需求，不能同时宣称所有 target/draft scratch 都可无条件 alias。

### 3.6 验收

主机逐步高层采样往返从每轮六次变为轮级交互；记录 RELAY 控制仍有多少步。D+C 及总请求必须优于同版本单卡对照；观察 target 是否因资源竞争变慢。单独 hidden 复制正确或单独 argmax kernel 快不算成功。

## 4. 路线二：激进——GDN checkpoint + 精确重算

### 4.1 原问题

`build_recurrent_attn` 让 GDN 产生多份完整 F32 状态，形成 gdn_out 中间张量，再拷贝进 recurrent rollback bank；卷积历史也按候选前缀保存。n10 的目标 scratch 比 n6 每卡约多 416 MiB，加载后草稿卡 GTT 从约 19 MB 增到约 978 MB，存在显著驻留风险。

但 416 MiB 不等于已经证明全由 GDN 构成；GTT 也不等于已经证明迁移的是 Q4 draft 权重。应按 allocation/张量生命周期建立表，不给未知 BO 身份起名字。

### 4.2 推荐策略

正常路径保留起点 checkpoint、工作/终点状态，以及足够精确恢复的逐步输入日志；全接受直接提交终点，真正中途拒绝才从起点恢复到接受前缀。

第一版最稳妥地保存每步 GDN 使用的 k、v、g、b 等必要输入，并按原算子相同 F32 运算序列执行仅更新状态的恢复核。也可研究保存更新向量，但不能未经证明用简化代数替换原舍入顺序。不要默认必须重跑整个 target；有必要中间输入时只重放 recurrent transition。卷积历史单独保留小前缀或恢复日志；普通注意力 KV 通过有效前缀/位置管理提交。

有三种阶梯方案：直接写持久 snapshot bank，先去掉重复大临时物化；每隔几个候选留 checkpoint；起点/终点加日志。选择依据是实测恢复分布与内存峰值，不是穷举 horizon 与 flag。

### 4.3 正确性与经济账

必须分别统计真实 model mismatch、EOG、长度限制、取消。token 接受率不等于 block 完全接受率，也不等于恢复概率。请求已结束且状态不复用时，可让槽位失效而不为无后续消费者的尾块做恢复；若会复用 prompt/KV，则必须恢复或明确失效，不能保留错误 cache。

期望收益 = 全前缀快照写入/物化与驻留代价的减少 − 日志开销 − 各拒绝位置概率×该位置恢复成本。

测试每个拒绝位置、target hidden 选择、所有 GDN 层状态、卷积历史、普通 KV 有效区间、sampler 状态、取消及下一请求。重算路径优先保持原 F32 指令次序并逐位对拍。只验证最终计数题正确远远不够。

### 4.4 验收

给出修改前后的 live allocation 表、峰值 scratch、驻留/GTT、状态恢复分布和未插桩吞吐。若 n10 仍越过实际驻留预算，就不要升级 horizon。不要仅因释放了一块显存就宣布同形 kernel 变慢的根因已证明。

## 5. 路线三：较激进——只优化 MTP 七行 target 执行路径

无 MTP 的成熟单行路径作为保护线，不重开一轮泛化单 token 调优。但单行最优不等于七行验证最优，尤其多行状态快照与不同 MoE 路由均改变执行特征。

针对实际七行形状设计量化小矩阵批处理，而不是只把 GEMV/GEMM 阈值改一下。现有 n=7 处在 `mul_mat_vec_max_cols=18` 的 direct-quant 路径。测量相同权重是否被七行复用、解量化次数、激活布局、寄存器压力与 spill；MoE 统计实际被多个 token 共用的专家，不默认七行都选同一专家。

从投影组、HC、状态读写、通信边界之间减少真正的中间物化和生产者/消费者等待。已有减少 dispatch 却无净收益的融合试验不原样重做。减少 dispatch 数、提高单个核吞吐都不是最终证据。

早期前三 stage 的慢点保留为定向问题：相同算子组合、同 rank、同驻留、同轮对比。不同 stage 有不同算子与输入，不能把 4–5 倍时间差直接归于 DPM；不需要写时钟或改 watchdog 来做路线图。

验收使用完整周期和请求；MTP-off 单行路径不得回退。改变浮点归约次序的核要单列数值变更与对拍结果，不能冒称位级等价。

## 6. 路线四：中等——首图 direct/recompute 与执行定义生命周期

这里要分清三种“缓存”：执行程序/描述符、临时内存绑定、模型的 KV/GDN 状态。

对模型状态，checkpoint + 拒绝时重算就是优先路线，不必执着保存每个前缀。

对首图执行，建议做首轮一次性 direct execution 与构建 replay 缓存的同构对照。首轮没有复用收益，不能要求为了缓存而无条件付所有维护成本；但这不是一个现成 flag 一开就能省 100 ms 的结论。

现有定位显示首图大头是 Meta tensor binding，约 99–110 ms；backend sync 只有微秒。跳过 Vulkan CB cache 而仍调用同一套 Meta 建图/绑定，不会自动消灭 binding。全局关闭 graph reuse 反而可能每轮重付它。建图、绑定、record、首次 GPU 执行必须分开记账。

合理实现方向是将固定图拓扑和 rank-local 绑定计划编译成程序，将实际 KV 位置、active rows、scratch 起址和 generation 变成每次运行的参数。可在合法启动阶段构建执行定义，但要记录真实加载/首请求耗时，不用假 warmup 或把请求工作挪出计时窗口冒充收益。

先维护有明确契约的 draft 单步、target 七行与 KV-only catch-up 程序，避免因 token 数猜 phase 再释放整个 arena。容量模式已有 46 个 dispatch 未证明 active<capacity 合法，必须逐类补齐输入读取、状态写入、依赖边界的覆盖；不放宽守卫硬跑。

Program 复用也不等于复用旧 tensor 地址：取消、阶段转换、scratch 重分配与上下文销毁都要有 generation 和退休证明。

## 7. 路线五：普通但具体——90 ms 图外回读

### 7.1 源码链

1. `llama_context::output_reserve` 尝试 output device 的 host buffer，否则普通 CPU buffer。
2. `ggml_backend_meta_device_get_host_buffer_type` 只在所有 rank 返回同一个 host buffer type 时才提供统一类型；不同则 nullptr。
3. Vulkan 的 host buffer type 按 device 分别创建。
4. Meta 的 `get_tensor_async` 对词表分片按 rank 依次调用 `get_tensor_2d_async`。
5. Vulkan 的 read helper 检查目标 host 地址是否属于该 device 的 pinned buffer；否则 async wrapper 走 staging，并在函数内部立即 `ggml_vk_synchronize(ctx)`。

因此，“名字叫 async”不等于五卡回读被并行排队。对于 Meta 输出缓冲配置，这是一条非常具体的退化路径。它的动态命中次数和时间仍需测量；不把 90.072 ms 全归给它。

图外还要单独记 memory batch prepare、mctx::apply、postcompute_success、compute guard 生命周期及输出提取。边界名称不能替代归因。

### 7.2 修法

为每 rank 建立自身可识别的持久 pinned output ring/slot；该 rank 的 logits 子片先写入自己的连续区域。先 enqueue 全 rank 的 copy，再在真正消费结果的边界统一退休，随后拼接或直接按分片采样。

每个 slot 必须有归属、generation、未完成标志，不能让多个异步 copy 覆盖同一 sync_staging。不要只删除 synchronize；它目前保护 deferred memcpy 与 staging 生存期。映射内存可见性和 CPU 消费同样须满足 Vulkan 同步契约。

测试从 buffer 构造、copy offset/stride、五 rank 不等长片段、部分输出行、并发提交和取消开始。新方案即使“更异步”也必须能证明输出字节正确。

### 7.3 验收

记录 rank、host buffer 类型、pinned 命中、fallback 次数、字节数、提交/等待次数；证明从逐 rank 阻塞变为全部提交后消费。随后测完整请求，检查 copy/排队是否仅把时间移到 sampler 而非真正减少关键路径。

## 8. 路线六：普通到中等——95 ms 后处理

最强方案与统一程序共用：给语义允许的 greedy target 验证做七行局部候选归约和全局前缀验收，而不是回读全量词表。

以仓库记录的 248320 词表举例，七行 F32 logits 是 6,952,960 字节；五 rank×七行×一个 F32 值与一个 32 位 id，原始候选载荷是 280 字节，不含对齐、header、generation。它减少的是回读和候选物化，不能声称不再需要计算 LM head 的全部候选分数。

对纯 greedy、无有效 grammar/penalty/logit bias/reasoning-budget 等依赖的路径，可独立生成每行 argmax，然后只提交首个 mismatch 之前的正确前缀以及参考语义要求的纠正/bonus token。一般状态性采样不得改为七行无条件并行 argmax；保留原 fallback 与 RNG/grammar/penalty 状态顺序。

较小实现不必等待 GPU sampler：在严格等价的纯 greedy 路径上直接扫描 logits 或 rank 分片，不先构造全词表 `llama_token_data` 数组再 top-k/sort。草稿 p_min<=0 时也可研究删除无用途的概率/候选物化；p_min>0 必须保留置信度语义。

`common_sampler_clone` 会复制整个 cur 候选数组。新增验收用 durable-state checkpoint，只保存 RNG、grammar、penalty/history 等必要状态，并在新采样时重建 scratch。不要盲改通用 clone 的可观察语义；也要正确重绑 cur_p，不能让临时 checkpoint 持有旧候选指针。

已有“一批只调用一次 synchronize”的试验没有证明收益，原因不能简化成“同步不重要”：getter 仍有同步、全词表 materialization 仍存在，底层回读也可能已阻塞。应改结构，而非只改一层函数参数。

## 9. 路线七：普通——最后再调 horizon 与策略

在执行与驻留修好以前，不再直接尝试 n7/n8/n10/n12、挪草稿卡、改 Q3/Q2 量化。历史已经说明接受率和更多候选本身不足以保证收益。

动态 horizon 使用实际前缀存活率分布：本轮再多一步的预计新增公开 token，必须值回新增 draft、target、恢复和内存压力成本。连续接受并不授权无限加长；检测到驻留恶化或 target 批宽成本跳变时，应缩短。

同形 MoE kernel 标签中的 n=10 可能是 top-10 专家维度，不是草稿 horizon。所有性能表明确写形状轴含义，避免改错 kernel 门限。

Q2 的命中下降已使总体退速，不把低比特当作必然优化；任何更换草稿权重的实验与执行优化分开记录。

## 10. 实施工单与依赖

展示顺序按激进程度。实际合并不应等一个巨型重写全部完成；先完成可审计计时与精确前缀测试，再分支推进，最终集成为同一执行程序。

| 工单 | 修改入口 | 必交付证据 | 禁止的替代品 |
|---|---|---|---|
| W0：同请求关键路径账 | speculative/server/context/collective | 同 cycle 的 CPU、GPU、ready、发布、回读、采样互斥账 | 不同进程 profiler 相减 |
| W1：rank-local 回读 | meta/Vulkan/context 输出缓冲 | pinned/fallback/bytes/waits，字节对拍，总请求 A/B | 只删 fence |
| W2：greedy 前缀验收 | sampling、Meta lowering、Vulkan kernel | 不等长词表、tie、零行、所有拒绝位置、fallback | 两趟 index/logit D2H |
| W3：状态重算 | delta-net-base、recurrent memory、GDN shader | 每层状态、卷积/KV、拒绝分布、显存峰值、恢复成本 | 丢 snapshot 不补契约 |
| W4：K/V-only catch-up | qwen4exp::graph_mtp、workspace/commit | 零输出 dispatch 清单和每位置 K/V 等价 | 接受即删 catch-up |
| W5：TP5 多步闭环 | 新执行 Program、speculative、Meta/Vulkan collective | 六步无高层逐 token 往返、scratch 寿命、取消代际、整请求收益 | 改五卡设备列表 |
| W6：七行程序与首图 | context、Meta binding、Vulkan replay | 首图 build/bind/record/execute 分离、热图收益 | 假 UID、假 warmup、放开容量门 |
| W7：自适应 horizon | speculative 策略层 | 前缀分布与边际成本决策，总请求分布 | 只看接受率 |

### 固定验收规则

独占五卡，保持 watchdog active，保持参考 F32/RELAY 与模型身份，记录实际二进制和 dirty diff、设备 UUID/BDF、环境、请求参数与采样链。诊断与正式性能两种构建/运行口径分开；不把 per-op logger 请求作为基准。

完整 1..60 必须逐字正确、自然停止、predicted_n 与 completion_tokens 均为 171；正式判定用 server predicted_ms<1710。另保留真实 client wall/首 token 时间作体验指标，但不能替换验收口径。报告所有预先安排的正式样本，不从波动里挑最快一条。

建议固定至少五个交错 A/B 比较块，报告各块结果、median 与离散程度；样本不足时只称 pilot，不称稳定收益。一个小补丁若收益小于噪声，先记录机制改善，不夸大吞吐。

正确性集包括：全接受；各位置真实拒绝；EOG/长度截断；取消后下一请求；1→7→1 与部分行；零 logits catch-up；不等长词表/tie/异常值策略；不同提示和上下文；参考不支持的 sampler 路由回退。缓存/状态恢复改动要比最终文本更深地比较状态。

首批需要看到的不是新旗标，而是三份证据：90 ms 回读路径命中账、514 ms 草稿完整关键路径、GDN 精确重算的内存与拒绝成本账。随后按同一预算把这些能力接进 TP5 MTP 程序。

## 11. 本地证据索引

- `困惑.md`；`TP5.md` 中“草稿与 target 账”“五卡草稿同次周期账”“七行 graph 首次分配”“五 rank 分片 GPU 贪心”“n10 scratch”各段。
- 现场确认存在并读取过首尾片段的原始日志：`/var/tmp/tp5-mtp-n6-granular-ledger-{cycles,draft,target,graphs}.log`。汇总响应与未插桩参考路径见正文；本次没有声称重新完整解析所有历史 JSON 或重新复跑历史测量。
- `common/speculative.cpp::common_speculative_impl_draft_mtp::{draft,commit,process_impl,enable_device_hidden}`。
- `common/sampling.cpp::{common_sampler::set_logits,common_sampler_clone,common_sampler_sample,common_sampler_sample_and_accept_n}`。
- `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph_mtp::graph_mtp`，约 564–755 行。
- `src/models/delta-net-base.cpp::build_recurrent_attn` 的 K/n_written/gdn_out 到 rollback bank 路径，约 933 行以后；`build_conv_state` 的前缀历史保存。
- `src/llama-context.cpp::{process_ubatch,decode_impl,output_reserve}`；`src/llama-model.cpp::dev_output`。
- `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_device_get_host_buffer_type`，约 438–459 行；`ggml_backend_meta_get_tensor_async`，约 3068–3167 行。
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_backend_vk_device_get_host_buffer_type`，约 25862 行；`ggml_vk_buffer_read_2d_async`，约 11038 行；`ggml_backend_vk_get_tensor_2d_async`，约 22460–22510 行。
- `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp::{tp5_star_handoff,tp5_poll_gpu_timing,tp5_relay_submit_epoch_chain}`。

外部语义仅采用原始论文与官方 Vulkan 文档：Leviathan 等的 speculative decoding 论文；vkCmdDispatchIndirect/VkDispatchIndirectCommand；dispatch 的 conditional-rendering 语义；calibrated timestamps；Vulkan synchronization。它们说明算法与 API 契约，不证明本工程已实现上述新路线，也不证明机器实际支持所有可选扩展。
