# 功能迁移与取舍

## 固定身份与原则

A 为 AtomicBot-ai/atomic-llama-cpp-turboquant 的 master，`cda0f9256665845cd9d6f3a0176880883f653a3d`，标记 `b10298-2.0.0`。旧 fork 为 `2777543b2358c88cd1c2000f7fa342b764a0e213`，由 `archive/pre-clean-20261018` 保留。旧分支比该 A 多 527 个提交；这些提交没有重新作为祖先进入干净分支。

完整路径差异为 4,634 项，其中 `artifacts/` 3,908 项、`reports/` 12 项，剩余 **714 个代码、工具、测试、文档和构建文件**。第一轮使用的缩减清单不是全库分母，先前“547 文件”的说法在本记录中撤回。逐文件范围、实际阅读深度和证据见 [AUDIT.md](AUDIT.md) 与 [audit/files.json](audit/files.json)。有记录不等于每行都做过语义证明。

迁移按具体行为作决定。一个大模块暂未迁入，不构成丢弃其小工具、数学函数、协议或内存策略的理由；发现旧实现错误，也不构成功能目的无价值的证明。无内容证据的项目保持待审或保留候选。

## 已迁入并有真实消费者的能力

| 行为 | 当前实现 | 保留的合同与证据 |
|---|---|---|
| 精确 compact top-k | B `TQSampling`；C `compose` | B 拥有数值结果；C 用真实 A 采样链，保留 distribution sampler 的状态。新构建 CPU / Vulkan 的 greedy、正温 token 对照通过。 |
| 请求 token trie | B `TQCorpus`；C `corpus-compose` | 共享前缀、重复权重、请求顺序和前缀终点；真实 prefix / full replay 的逐请求 NLL 对照通过。 |
| 旧生产 corpus 小合同 | 同上，另有归档 writer / reader 互操作测试 | nodes ID 从 1 开始；kind、64 位 timestamp、空请求和缺失空流正确处理；真实旧 writer → 新 B → 旧 reader 字节往返通过。 |
| Prefill 优先 | 独立 B `TQScheduling`；C `batch-compose` | 保持当前提示、按槽位顺序选择，不隐含 FIFO；有待处理提示时暂停 Decode，完成后恢复。真实三请求逐批输入验证通过。 |
| 逻辑窗口与物理 KV 分离 | A 的 `n_ctx_kv` / `llama_n_ctx_kv`；C 配置与使用 | 复用已有统一 KV 分配器，只将物理容量传入 attention KV 构造；不修改逻辑窗口 / RoPE 或 recurrent 容量。真实 logical512 / physical1024 通过。 |
| Prefill / Decode 计算工作区 | A 的同步释放 / 当前字节观察 API；C 显式调用 | 同步、失效可重建图、释放 galloc，保留模型 / KV / recurrent / 输出；下一批按实际形状重建。CPU / Vulkan 两臂输出、位置、token 相同，分配器字节确实归零。 |
| 原版 server 的 Prefill / phase 开关 | 五个 common / server 文件的有限协作补丁 | 保留实际 HTTP 入口；prefill 默认 on、phase memory 默认 off，错误组合加载前拒绝。三路 SSE、插入 Prefill、25 层卸载与 KV1536 实测通过。 |
| 普通 server task 默认槽位 | `server-task.h` 一行 | `id_slot=-1`，普通 task 移动不读取未初始化整数。与原版 server 请求路径共同验证。 |
| CPU 显式 F32 attention | `ggml-cpu/ops.cpp` 小修复 | F16 V 也尊重请求的 F32 累加。真实公开操作用例在原 A 失败，修复后通过。 |
| Vulkan 公开 Turbo WHT | host / shader 两个文件 | 既有 group32 / 64 / 128、正反 scale 与实际 grid；原 A 12 例失败 10 例，修复后 12 例全过。 |
| 独立 logits 比较器 | `fork/tools/compare-logits.py` | 保留 TP5LOG2 文件兼容，stdlib、逐行有界读取、token 身份和完整性校验，数值 / bitwise / argmax 分开；20 个真实 CLI 测试通过。 |
| 源码与安装包边界 | build、manifest guard、isolation | A 只加登记的16文件；每 B 和每 C 单独构建、测试、安装，搬迁后真实消费；完整41步通过。 |

所有当前 A 例外累计 **16 文件，+324 / -39 行**，精确内容与退出条件见 [upstream-exceptions.json](upstream-exceptions.json)。原版 HTTP server 的私有调度循环仍属于 A 例外，不能把它包装成“已经独立的 B”。三个 B 均只依赖 C++17 标准库。

### 内存能力必须分开看

| 子功能 | 当前状态 |
|---|---|
| 同一 context 中统一 KV / sequence 存储 | 上游 A 已有，直接复用 |
| 物理 attention KV token 容量独立于逻辑窗口 | 已补回，支持范围有显式检查 |
| Prefill / Decode 计算工作区释放与重建 | 已补回单 context 生命周期 |
| Prefill 优先任务选择 | 已补回 B、C 和原版 server 入口 |
| Prefill 最大并发 / 初始 reserve 单序列图 | 独立候选，未因已有优先策略就宣称完成 |
| 基于设备真实容量的 auto-fit / slack / 预算 | 保留目的，旧 solver 不照搬 |
| recurrent 物理槽位、COW / 事务预检 | 独立候选，当前物理 KV 参数不改变它 |
| 压力下 idle KV LRU 回收、admission、抢占与恢复 | 独立候选，当前 C 不自动驱逐或重试 |
| target / draft 跨 context 共享 allocator | 待明确共享所有权和同步协议，当前未实现 |

`--total-kv` 当前接受非负整数，非零启用 unified；不接受旧 `auto` 语法。C 根据已知 token 需求选择默认值，不是显存探测。`llama_memory_compute_bytes()` 只报告计算分配器拥有的字节；没有把小测试的工作区缩小写成整机峰值显存下降。

## 有价值、尚未迁入的小能力

下表是实际内容审查后的候选。它们不是“因性能低而删除”；每项需要自己的真实消费者、边界和必要验证。具体代码位置及未读范围见各 [审计分册](AUDIT.md)。

| 组别 | 应保留的独立行为 | 下一步边界 / 验证 |
|---|---|---|
| 容量策略 | bounded fit、预算保留、每设备余量、PP reserve 限制 | B 只算拥有值的预算 / 探测计划；C 拥有真实 A context 并执行探测；不能拿 metadata-only 假分配当峰值测量 |
| 槽位与池 | idle LRU、cache-key 亲和、物理 recurrent 槽、generation / payload 身份 | 先定义分配、释放、共享引用与失败回滚；旧 `llama-xkv-hot` 的槽位元数据责任可独立提取，不等于接管 A 的 tensor |
| recurrent 清零 | 单 cell / 非 grouped 的整 buffer 清零、普通路径 strided 写与 fence | memory 审计 M33 已定位；历史逐 tensor 907–935 ms → 整 buffer 12.8–16.8 ms 是局部重置数据，须在当前 A 上验证单 cell、其他 cell 不受影响及真实取消恢复，再独立迁入 |
| 重试与输出 | 已输出文本游标、预算 / usage 聚合、首块语义、重试恢复 | C 拥有 HTTP 请求与 sampler；必须验证随机状态、取消和部分输出，不能只比较最后 token 总数 |
| 请求 / 文本 | UTF-8 与 stop 扫描 fastpath、cache affinity、按 lane 拼完整行、usage 只计算一次 prompt | 标准字符串 / 值策略适合 B；HTTP 协作归 C，须以实际多 choice 与流式消费者验收 |
| 在线 corpus 录制 | 请求结束时录制真实 token 路径、种类、时间戳，供后续校准与回放 | B 的生产格式已经恢复；在线请求生命周期仍需 C 消费，离线 `llama-corpus build` 不等于原 server 录制钩子 |
| 退出与失败 | signal 自管道、正常线程清理、第二信号、LoRA 失败传播 | C 生命周期职责；错误必须传给真实调用者，不能靠测试环境替代实际队列 |
| grammar / sampler | grammar 替换、clone 后结果所有权、实例级 Turbo 配置、额外 RNG draw 语义 | 使用公开 A sampler 原语；消除对私有 distribution 对象的强转，专测后续随机状态 |
| 模型 / GGUF | BailingMoE3 名称及缺省兼容、缺失 MTP 权重、uint64 PLE hash | 上游实际消费者和 GGUF 元数据校验；每个兼容点独立提交，不整包恢复所有图修改 |
| 校准与 imatrix | WXQCAL03 → A GGUF imatrix、公共回调激活采集、trie 复用 | 原始 sum2 + expert counts 可被 A 消费；补类型、形状、有限值、计数和不完整校准拒绝，再由 C 编排 A |
| 独立证据工具 | logits 采集、GGUF 头读取、离线性能报告、请求 / 构建身份 | 比较器已迁；采集器用公开 A API；统计分母、有限值、完整请求与输出保护各自验收 |
| 工作区计划 | 64 KiB 容量向上取整、阶段 scratch 最大值、间接 dispatch 算术 | 纯值 B；容量取整不能改变 dispatch 几何；实际 buffer 与设备生命周期由 C / A 拥有 |
| 状态封装 | 有界 state envelope、scope / policy / adapter / nonce 校验 | B 可封装字节，C 包装 A 的 opaque state；需明确端序，同进程身份不能冒充持久跨进程模型身份 |
| 数学参考 | 无符号 Hadamard、RoPE 平移、RERoT mask / 可见性、GQA / DDVR 参考 | 可成为独立 B / 对照实现；不声称已改变生产 KV 可见性，也不与 signed Turbo WHT 混为一项 |
| 稀疏 / 合并参考 | mean / energy 选择、online softmax 三元组归并、有限 mask 保留 | 单独证明数学合同和异常回退；局部参考不自动获得稀疏 attention 整模型质量资格 |
| Vulkan 小功能 | FA Y 缓存失效、SET_ROWS 类型能力、device 对应 pinned host allocator | 先构造真实 before / after，分别处理能力声明、缓存身份和共享设备所有权 |
| 后端算子 / 生命周期 | SCALE F16、MROPE / IMROPE 融合、字节 memmove、meta 分片非零偏移、readback pin / 可见性、device staging 清理 | 公开既有算子可小补丁；overlap、stream / fence 和多 context 生命周期需对应证明 |
| CPU / CUDA 小优化 | 既有 Q2_0 VNNI、Turbo 小查询优先 native VEC | 与私有 PQ2_0 / TP5 分开评估；本轮没有 CUDA 或该 CPU kernel 的性能测量 |
| 评测与参考 | DAG 参考、reader 顺序、性能预算、评测 split 与协议 fixtures | 保留测试目的，修正错答案、泄漏分组、NaN 假通过和提前 PASS 后才能作证据 |

## 舍弃的具体实验补丁

仅下列已有收益依据的实验组合在本次选择中明确不迁。数据来自归档 `PERFORMANCE.md`、`MMVQ+MTP.md`，不是本轮重新测量，不能为其他小改动批量背书。

| 实验 | 历史证据 | 决定 |
|---|---|---|
| router row split | 同一 pilot 66.6914 → 61.8741 tok/s，约 -7.22%，未晋级 | 不携带该实验补丁 |
| BF16 router | 8 配对块 / 16 进程，约 +1.14%，原记95% CI [0.79%,1.48%]，低于原2%门槛 | 不为它保留跨模型 / 后端耦合 |
| MMVQ + MTP 联合实验选择器 | 131 配置，最终候选等于原基线；C/R=1.034315，原记95% CI [0.996803,1.073239] 包含1 | 不携带该联合选择器、强制模式和控制复杂度；不否定整个 MMVQ / MTP 家族 |

旧诊断工具不再一概归为“无用并舍弃”。比较器已恢复，离线报告、校准转换和小参考实现各自进入候选表。硬件档案、历史报告与实验输出保留在旧分支，作为来源证据，不进默认构建。

## 不复制的旧错误实现

静态审计指出的合并风险按具体语句和状态边界记录，而不是宣称都已在运行中复现：共享 allocator 初始化随后被普通初始化覆盖；默认 KEEP_CHUNKS 跳过 trim；KV cells 特殊成员遗漏新增身份字段；Turbo K shift 新分支落在已有 `continue` 后；recurrent / KV 失败预检仍有部分先行修改；某些 Qwen4 MTP 图路径重复归一化、RoPE 或 gather。

测试和脚本也有可见失效条件：并发容量脚本实际串行且并发代码不可达、写 PASS 早于最后检查；部分数值比较让 NaN 被 `std::max` 吞掉；新增 CHECK 在最后失败汇总之后执行；小样本 t 临界值错误；评测算术答案错误或 dev / blind 家族交叉。详见审计分册。当前验收采用直接 A 操作、真实输入行、真实设备日志和严格有限值检查。

五个旧 reader 问题另有实际纯函数反例，包括缺键 / null 漏报、非有限参考答案误判、LP 容差伪下界、坏 SSE JSON 未纳入 error，以及多行 PID 只读首行。输入、输出与来源摘要见 [audit/reader-counterexamples-20261018.json](audit/reader-counterexamples-20261018.json)，不代表运行了旧模型或硬件恢复脚本。

这些结论支持“重新实现有用合同”，不支持“所有旧功能都无效”。未跑过的旧路径不写成已复现崩溃或已测性能下降。

## 较大能力的状态

| 能力 | 当前判断 | 重新进入的条件 |
|---|---|---|
| TP5 / LateBind / relay / device loop | 历史有59.66 →65.17 tok/s的实质收益，不能称无用；整条私有 runtime 暂留归档 | 提取纯计划 / 数学 / 元数据职责，明确 A 必需的最小执行与生命周期接口，再用真实目标多卡负载验证 |
| RERoT | 普通多请求、corpus 前缀重放或独立 parser 不具备其多 reader 共享 KV / 可见性语义 | 明确同 context、同 owner、动态可见性、recurrent / RNG 隔离协议，验证真实 reader 行为 |
| FlashPrefill / TriAttention | 独立数学、格式和状态封装有价值；历史 online value accumulation 的局部 prefill 约 +31%，但 greedy 输出变化、部分检索失败，整体验收未闭合 | 固定模型 / 上下文 / 质量门，公开消费者和最小接缝，避免把局部 kernel 测试当整模型证明 |
| XKV-SR | 槽位元数据、codec、预算和验证可拆；完整量化存储 / 图 / 事务运行时暂未迁 | payload、实际驻留、质量、吞吐与共享状态失败回滚同时验收 |
| DFlash / DSpark direct-hidden | 旧记录有内部交接与 probe；15–40%是目标，缺乏本次可用的独立整模型配对证明 | 真正 target hidden → draft 设备消费、数值与生命周期验证，再测相对合格基线的收益 |

精确文档差异还确认，speculative aliases、chained / adaptive MTP、默认 draft 深度和 draft CPU / KV 参数已经属于 A；本 fork 在对应文档新增的是 direct-hidden 段。已有能力直接复用，不再冒充自创 B 重写。

上游已有模型、量化、后端、server、bench 和 speculative 继续由 A 提供。当前 clean 分支不宣称与旧完整 fork 功能等价，但有价值的小能力都保留独立入口或可追溯的迁移候选，不按大模块名称抹去。

## 回查

```sh
git show archive/pre-clean-20261018:PERFORMANCE.md
git show archive/pre-clean-20261018:wanxiangqi/server/server-corpus-dump.cpp
git diff cda0f9256665845cd9d6f3a0176880883f653a3d \
  archive/pre-clean-20261018 -- src/llama-kv-cache.cpp
```

原 master、旧实现、用户未跟踪实验资料和模型均保留。当前 clean 分支的每一项迁入、候选或拒绝都能回到固定原提交与实际测试范围；没有删除旧分支来掩盖尚未迁移的工作。
