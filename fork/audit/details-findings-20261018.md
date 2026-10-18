# 本批次小功能补审：2026-10-18

本记录追加实现、公开接口消费与真实运行后的判断，不改写此前静态分册或逐路径 ledger。批次起点为 `6de088714f35f84427d9952d713f9d6cdffc668b`，上游 A 为 `cda0f9256665845cd9d6f3a0176880883f653a3d`，旧 fork 为 `2777543b2358c88cd1c2000f7fa342b764a0e213`。旧 fork 的路径、行号和 hunk ID 均沿用原分册定义，不是新 clean 源码行号。

当前五个 B 与五个 C 的完整职责见 [README](../README.md)。最终来源文件与 manifest 摘要、实际归档和 GPU 运行目录，以及全量结果和保留的失败统一记录于 [VALIDATION 的 Final release gates](../VALIDATION.md#final-release-gates)。本页没有扩大旧总账 714 路径、546 个完整改动内容与 168 个选定范围的阅读深度，也没有修改原 `runtime_verified: false` 的历史记录。

## 从旧 finding 到当前职责

ID 按分册限定命名空间。此处 `small:S04/S05` 是校准项目，不是同名 `server:S04/S05` 的 Prefill reserve / 输出容量。

| 原 finding 与路径 | 原迁移判断 | 本批次结果与精确边界 |
|---|---|---|
| `server:Q14`，`tools/server/server-context.cpp:7487-7491` | 无 stop、非 stream 时跳过逐 token UTF-8/stop 扫描；待最后边界与性能验证 | 用独立 TQText 与真实 batch C 重建增量文本、合法 UTF-8 和 stop 合同。没有机械恢复该 fastpath，也没有扫描速度收益声明。 |
| `server:T09`，`tools/server/server-rerot-thinking.cpp:39-47,73-149` | 跨块 terminator 回看可独立，完整 parser 还包括 EOF、嵌套和条目规则 | 仅重建跨块终止符这一机制。没有迁入 thinking/lane 语言、完整 parser 或 grammar 一致性保证。 |
| `server:Q15/Q16/Q17`，server task/context 的首 chunk、usage 和最终 SSE | 逐项保留真实输出/聚合语义 | 仍待迁。新字节游标不是旧 `stream_started` 或多 choice usage 的替代，也不是完整 HTTP SSE 生命周期。 |
| `small:S04`，`wanxiangqi/probe/calib-to-imatrix.py`、`wxqcal.py`、`wanxiangqi/lib/calib-dump.h/.cpp` | 独立 parser + imatrix C 候选，要求完整性、数值与范围策略 | TQCalibration B + imatrix C 已实现，旧生产 writer 与实际 A loader 互操作通过。原 NumPy/Python 方案收缩为 C++17 B 与公共 `ggml::ggml-base` 消费者，转换无需 NumPy/Python 运行时。 |
| `small:S05`，`wanxiangqi/calib/trie-calib.cpp` | 用公开 eval callback、MUL_MAT_ID 统计与 seq snapshots 做真实校准和 DFS；需要模型对照 | 格式基础由 S04 闭合，collector、checkpoint、DFS、样本人口与模型契约仍待迁。不能用旧 writer 人工 fixture 冒充实际模型采集。 |
| `backend:G09`，`ggml/src/ggml-vulkan/ggml-vulkan.cpp#h183` | mask bitmap 覆盖 activation scratch 后失效三个 key 字段，高价值正确性候选，尚无合法图 before/after | 实际复现 scratch 覆盖后的错误，并扩展到相同资源被大 softmax 覆盖及 no-split 读取的同步责任。最终公开 API 专项通过，细分证据见下文。 |
| `backend:G44`，同文件 `#h290/#h292` | 基于“只有 F32 shader”的前提，候选要求 Turbo SET_ROWS 拒绝 F16 | 固定 A 实际已有 F32/F16 变体，因此保留 A 支持，拒绝迁入旧拒绝条件；实际 16 次 packed 比较通过，不附带迁入相邻 G19 日志。 |
| `backend:G14`，同文件旧 fork 行 23622-23675、27458-27512 | pinned host alloc/free/alignment/max-size 应由实际设备拥有，独立多 GPU 候选 | 未迁。必须把分配、释放和查询放在同一个 device owner 合同里；其他 Vulkan 用例不能提供此项资格。 |

`G09#h183` 对应旧 ledger 的 zero-based hunk 182，`G44#h290/#h292` 对应 289/291。这里的映射用于回查，不是将包含其他行为的整 hunk 标为已移植。原判断的完整上下文保留在 [server 分册](audit-server-findings.md)、[small 分册](audit-small-findings.md) 与 [backend 分册](audit-backend-findings.md)。

## 文本输出：一个完整 B，一个实际 C 消费者

TQText 每个实例拥有 UTF-8 不完整尾部、stop 状态和输出字节游标，返回 owned 值；各实例不共享可变状态，也不需要 A 或 I/O。跨 piece 的匹配、非法 UTF-8 逐字节 U+FFFD、最终修复、最早完成优先和同位置最长匹配都有明确合同。有效 stop 字符串及数量、单串和总字节数均有边界。

C 才拥有真实 tokens、seq ID、sampler、decode、stdout 和完成顺序。`--stop` 是重复的 literal UTF-8 参数；`--json --stream` 给出可按 `request_id` 和连续 `offset_bytes` 拼接的增量。成功请求恰有一个终止事件，拼接结果等于最终文本。只有驻留请求分配匹配状态，完成后先消费本 batch 全部输出并同步，再释放真实序列，最后归还 B 预算。

文本抑制不删除采样 token 计数。默认保留已采样非 EOG token，`--ignore-eog` 时也保留 EOG；匹配确定后才停止后续 decode。最终 UTF-8 修复可能在 EOG 时完成 stop，此前携带不完整字节的 token 可能已经 decode，不能把“匹配后不再提交”写成“所有被 stop 抑制的 token 都未执行”。

最终 Vulkan 运行包含 10 条命令，其中六种模型配置各三个请求。serial/stop/parallel 的 buffered 与 stream 对照固定相同提交形状，18 个请求通过；参数失败命令不是模型运行。脚本约 14.5 秒的时间只描述执行记录。纯 B 的 sanitizer 与边界测试、真实 C 的模型输出各自提供不同证据，不提供 HTTP transport、重试、取消、lane 或多 choice usage 资格。

## 校准导出：保持测量，显式承认估计

原 S04 的可用目的，是把 WXQCAL03 的 expert input / hidden 平方和与样本计数交给 A 的 imatrix 消费路径。B 只读格式；C 选择完整性与 unseen expert 策略，使用已安装 GGUF/GGML 元数据接口编排输出。公开 `ggml::ggml-base` target 与 `GGML_INCLUDE_DIR` 来自固定 A 的安装 config；C 不链接 llama，也不创建 inference context 或 GPU。config 本身仍发现启用的 backend，因此保持完整 A 安装。

B 在分配前界定几何和字节量，一次返回一个 owned layer。covariance 以有限块读取并验证后丢弃，不能通过 seek 越过 EOF 冒充有效数据。未知 flags、padding、重复层、坏计数/数值、截断和尾部额外数据都明确失败。routing 与 expert activation 是不同统计人口，不要求它们机械相等。

C 每层输出 gate/up/down 各自的 `.in_sum2` 与 `.counts`，六个 F32 tensors；顺序为 expert-major。保留 measured raw square sums，由 A 的实际 F32 除法归一化。count 必须可精确表示且不超过 INT32_MAX，既防 F32 舍入也兼顾 A `std::lround` 的 long 范围；原始与归一化后的非零下溢或溢出都拒绝。

默认 selected range 要求 complete、total 正数且 decoded==total，默认拒绝没有观测值的 expert。旧 complete 只表示遍历未报错，旧最后 batch 可 overshoot 所选上限。显式 `--allow-partial` 保留头部所有计数及每个原因，包括超出与未达到；不会截断统计伪装修复范围。显式 `--unseen global-mean` 记录被填补的 input/hidden expert ID、策略和非测量事实，count 1 只是编码估计，不是假造样本。

输出用一层内存、16 KiB 转换缓冲与同目录 private spool；metadata 有独立边界。spool 与 staging 共存约需两倍最终 payload 磁盘空间。完整验证和 sync/close 后以无覆盖 hard link 发布，已有目标和 symlink 保留；caught failure 清理自有临时文件。平台范围、stdout 发布后失败与调用者控制目录假设见 [imatrix C](../imatrix-compose/README.md)。

### 真实格式互操作，不扩写为量化质量

原 `calib-dump.cpp/.h` 与旧提交逐字节相同。fixture driver 调用实际 `wxq::write_calib`，写出全部三个统计 flags、两层以及 decoded/total=9 的生产格式；没有使用仅由新测试 pack 出来的字节作为唯一兼容证据。新 C 产生的 12 tensors 经独立 reader 完整检查，再由实际 A `llama-quantize` 的 loader 读为 6 importance entries / 9 chunks。

本次未找到历史真实模型校准文件。fixture 是格式样本；后续模型路径故意不存在，量化程序在模型阶段退出 1。它证明真实 A 的格式消费，不能证明完成了模型量化、模型身份正确或量化质量收益。旧源文件摘要、producer/converter binary 摘要、命令和输入/输出摘要要随实际 receipt 保存；不能借用较早 corpus writer 的摘要。旧 writer 生产过程的坏 flags/vector 状态等问题也没有借此次 writer 互操作被声明修复。

S05 的 eval callback 数据选择、DFS 顺序、共享前缀统计人口、checkpoint 与模型支持仍需真正采集器和实际模型对照，继续保留候选。

## Vulkan G09：缓存失效与同步责任分别验收

公开 GGML/Vulkan 图直接执行 GPU，不包含 scheduler CPU fallback 或私有 backend bridge。两个 matmul 使用同一不连续 F32 source view、两个内容相同的独立 F16 weights，中间穿插实际 softmax 或 attention；图顺序和全部输出都会检查。

| 用例 | 原 scratch 修复前 | scratch 与 source-write 最终修复后 |
|---|---|---|
| 小 softmax，16384 columns | 负控制通过 | 通过 |
| 无 mask FA，width 64 / Q32 / KV4096 / 一头 | 负控制通过 | 通过 |
| 大 softmax，32768 columns | 第二个 matmul 32/64 错，最大绝对差约 3380.01 | 通过 |
| masked one-head FA | 第二个 matmul 32/64 错，最大绝对差约 `2.70e13` | 通过 |
| masked no-split target，256 query heads / 一 KV head | 初版复杂 V 的独立控制存在下述数值限制；后续 matmul 仍显示已有缓存问题 | binary-scaled fixture 的全部 524288 attention 输出、独立同 GPU 中间操作及前后 matmul 检查通过 |

bitmap 与 large-softmax partials 覆盖 `prealloc_y` 时必须清除 pipeline/source/decode-layout 身份。此前只有 bitmap 的 G09 候选，因此本轮按实际共享资源责任补齐 softmax，而不是原样挪一个旧 if。

同时，private scratch 不在公开 tensor overlap tracker 的资源范围内。split-K FA 已有最后 bitmap 读取后的 reduction barrier；no-split 需要标记 Y 待同步，让下一次覆盖 Y 的 matmul 使用原有 pre-copy barrier。缓存失效解决“值是否仍有效”，同步解决“何时可以重用”，不能用其中一个替另一个。

初版 no-split 的复杂 fractional V 在同 GPU 独立对照中全部 524288 值一致，但 475136 个值不满足闭式门限，最大差 0.0146484。这是独立控制的数值限制，不是被复现的同步竞态；该失败保留。最终 fixture 改用 binary scales 0.25/0.5/1、Q/K 零值和明确 visible bands，预期均值可精确构造。attention absolute tolerance 仍为 `2e-4`，没有放宽误差门，也没有仅抽查若干输出。

最终五个 scratch case 全部通过。no-split 是固定设备和源码调度分析对应的目标条件；公开 API 没有读取私有 FA tuning 状态，不能假装实际探测了该字段。数值通过也不证明所有同步竞态、设备或图组合，未运行的 Vulkan synchronization validation layer 不记为通过。完整形状、命令和责任见 [scratch 协议](../tests/a-seams/vulkan-scratch-reuse-protocol.md)。

## G44：纠正原静态前提，保留 A 已有支持

原 backend 分册将 F16 拒绝视为 capability 修复，前提是只生成 F32 shader。重新读取固定 A 的 host registration、shader generator 和 runtime selection 后，确认 F32/F16 source 到 Turbo2/3/4、I32/I64 index 的变体实际已经存在。这不是换了 A 基线，也不需要新增 A capability 拒绝。

真实测试覆盖 Q8_0、Turbo2/3/4 目的类型与 I32/I64 index 的八个配置，每个执行 F32 和 F16 scatter，共 16 个 packed-byte 比较。使用 width 256、目的行 5/1/3 和其余四行 sentinel；先在同 GPU 建立 F32 identity rows，再验证 scatter 的全目的内容。全部通过。

这证明指定布局中既有 source type 和 row addressing 行为；同 GPU packed reference 不独立建立量化误差，也不涵盖任意 strides、其他 GPU 或不在公共合同内的 duplicate destination index。G14 device pinned owner 尚未迁入。

## 新发现：原地修改 source 仍可留下旧转换

这是本批次后续建立的新 finding，不伪造为某个旧 ledger ID。即使 scratch 覆盖修复已存在，source tensor 指针不变时，底层内容原地写入也可能让转换缓存错误命中。

初始零偏移 graph 为 matmul -> SCALE inplace -> matmul，两次 matmul 读同一个原始 view 指针。disjoint SCALE 负控制通过；inplace 中 SCALE 和 source storage 的全部 514 值正确加倍，第二个 matmul 的 64/64 值错误，最大绝对差约 16.0306，记录的一个值为 expected 31.8711 / actual 15.9355。SCALE 实际执行且写入正确，因而不能把第二个结果解释成 SCALE 被跳过。

冻结版本为 257-by-2 F32 storage 上 256-by-2 view，row stride 257 floats，**4-byte offset**。独立同 GPU reference 保留相同布局，以原值和预先加倍的值分别计算。非零偏移版本修复前第二个 matmul 64/64 错、最大绝对差 15.4662；最终重建重装后 control/inplace 均通过，每个 matmul 的全部 64 个输出，以及 SCALE/源存储各 514 个值的相应比较误差为 0。两个不同布局的数字分开记录。

A 对记录的 write 与 cached source storage 范围作 overlap 检查，包括 exact alias、每个 fused-write 输出；任何重叠失效三个 identity 字段。普通 hazard barrier 只排序原 storage 写入，不刷新 converted copy。找到一个同步依赖后也不能提前结束对后续 fused writes 的检查。这个责任由 private cache owner 完成，不需要另建一个 B 回调或复制 backend。

最终 Vulkan 接缝 7/7 CTest 通过，包含五个 scratch case、两个 source-alias case及 SET_ROWS 的16次比较。disjoint control 证明数学结果保持，不测 cache-hit 次数或收益。没有推导 recurrent clear 或某生产模型实际触发该 graph，也没有覆盖所有 fusion、stride、graph capture、执行中 host write 或其他后端。协议见 [activation source mutation](../tests/a-seams/vulkan-activation-cache-protocol.md)。

## 发布资格仍按实际阶段保留

最终 CPU archive/isolation 62/62 条命令已通过；source/build 副本与原前缀移除后，五个 B、五个 C 的搬迁消费以及实际 batch/logits/stream/imatrix 检查通过。TQText 与 TQCalibration 的 ASan+UBSan 独立用例通过；它们不补齐其他模块以前没有取得的 sanitizer 资格。

较早全量 CPU 是 27639/27639。+12 阶段全量 GPU 是 25931/25933：FA mask+sinks、Q8_0 K/V、head dimension 256 个案在 baseline 与 +12 单项均失败；TOPK 的全量失败在两边单项都未复现。最终 GPU 重跑为 25932/25933、exit 1，只剩同一 FA，报告 ERR 0.009055835 > 0.0005；最终 TOPK 没有复现，旧失败仍未定位顺序或状态影响，不能撤销记录或据此排除回归。

普通 CPU / Vulkan `build.py --tools` 均已 exit 0；另有 9 个 K/V 组合、pp/tg 共 18 行的单 repeat benchmark 执行 smoke 退出 0。后者只证明这批组合单次可执行，不能当作可靠的吞吐比较。最终 A 统计已核对为 17 个源码文件 +357/-43，连同 CI 共 18 路径 +406/-43。实际源码/manifest 身份、构建与测试状态见 [统一门槛表](../VALIDATION.md#final-release-gates)，可移交索引见 [details-qualification-20261018.json](details-qualification-20261018.json)。全量 GPU 未全部通过，不宣称所有发布门槛通过。

没有 CUDA 设备，没有本批次 CUDA、其他 GPU、模型量化质量、吞吐或峰值显存结论。此前的 HTTP SSE、采样、corpus、logits 和模型数值分别保留此前身份；714 路径的旧阅读记录与全部剩余候选也按原范围保留。
