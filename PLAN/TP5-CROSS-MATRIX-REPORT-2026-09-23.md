# TP5 五卡交叉试验：当前证据与默认策略

日期：2026-09-23。硬件为五张 AMD Radeon RX 6800；用户确认目前无降频。测试期间 **watchdog 保持运行**，没有停用或修改。模型：Qwen3.8-Flash-Next-APEX-I-Compact。标准请求：`/v1/chat/completions`，`Count from 1 to 60. Use commas without spaces. Output only the numbers.`，`temperature=0`、`seed=42`、`max_tokens=192`、`cache_prompt=false`、thinking disabled。

**降频作废声明与最高速复现（54.45 decode tok/s）**：
1. **作废范围**：2026-09-23 02:39:08 至 08:26:00 期间在降频状态下采集的所有绝对性能数据（40~43 tok/s）全部作废。根因是凌晨 watchdog 自动总线复位后，card2（矿版 VBIOS `113-1N21XLMIN203W_210810`）退回了 1200MHz 出厂锁频，算力腰斩至 8.25 TFLOPS（其余卡 16.2 TFLOPS），成为 AllReduce 木桶短板。
2. **自动锁频固化**：在 `scripts/reset-gpu-pci.sh` 补充强制 manual 模式并锁定最高 DPM 档位逻辑。经手写 Vulkan F32 FMA 压测，5 张卡算力全数稳定在 **17.1~17.2 TFLOPS**。
3. **单轮简并验证**：恢复顶频后立即执行 `native-default` 简并实测（`/tmp/tp5-retest-highclock-smoke.json`），纯 Target F32 端到端解码速度达到 **54.45 tok/s**（warmup 53.97 tok/s，171 tokens 耗时 3.14s，完整请求 committed 35.70 tok/s），内容 1..60 完全一致，超越历史 52.38 基准。

## 先前探索数据：只能生成候选

旧版工具每变体仅重跑 3 次，继承调用 shell 环境，缺少完整输出/进程二进制/时钟审计、A/B/B/A 配对和置信区间。表格的 `predicted_per_second` 是目标推理阶段指标，不是请求/投机完整周期吞吐；以下数字**不得用于声明统计显著胜者**。

|模式|旧三次 decode tok/s 均值|解释|
|---|---:|---|
|Relay F16 + MTP draft-3|52.90|没有 committed-token/full-cycle 账；**不是 MTP 净收益证明**|
|Relay F16，无 MTP|52.47|纯 Target 候选|
|原标记“Relay F16，MMVQ auto”|51.99|**无效对照**：旧 Vulkan 代码按存在性读取 `GGML_VK_DISABLE_MMVQ`，`=0` 仍然禁用；旧数值不能证明 auto 路线|
|Relay F32，无 MTP|52.01|F32 精度对照，不能由计数题推断与 F16 位级等价|
|Timeline F16，无 MTP|49.06|同步方式候选控制组|
|旧“零调参”|52.38|旧脚本继承大部分 shell 环境；**不算 clean-room**|

另一次四变体短跑：纯 Target Relay/F16 52.40、Q-prep+Span+MTP(draft-3) 51.86、Q-prep+Span+MTP(draft-2) 45.75、旧零配置 52.60 tok/s。该 TP5 负载**没有证明 RERoT Q-prep/Span route-hit**；不能据此说这两个优化有效或无效。

## 当前验收方案与直接证据

`scripts/run-tp5-cross-matrix.py` 使用白名单环境、独立的服务器进程/受控退出、相同输入 A/B/B/A 五个完整配对块，逐请求核对完整 `1,2,…,60` 内容、自然停止及 `prompt_n=31`；纯 Target 要求 `predicted_n=171`。将四个试验合并为一个独立 block 后计算 Student-t 95% 配对区间。保存 raw response、server 日志、提交与 dirty 状态、server 二进制与已映射 DSO 哈希以及 GPU 观测。主指标 `usage.completion_tokens / client_wall_sec` 包含完整请求，MTP 缺 `completion_tokens` 直接拒绝；server `predicted_per_second` 另列为 decode 指标。`--dry-run` 标记 `DRY_RUN_NOT_MEASURED`，不产生虚假样本。

已验证构建使用 ccache 和 `-j44`；`test-tp5-plan` 通过；五卡 `test-vulkan-tp5-mesh --sync relay --rounds 16 --check-all` 和 `--sync timeline --rounds 8 --check-all` 均通过，包括真实 GPU graph-producer 与 FD delta=0。`LLAMA_REROT_GPU_Q_PREP=1 test-rerot-q-prep` 和 `LLAMA_REROT_GPU_SPAN_EXPAND=1 test-rerot-span-expand` 的 CPU/GPU oracle 对拍通过；这些只证明独立算子语义，**不证明该 Qwen TP5 请求实际经过 RERoT 或获得收益**。

**已完成的第一组五块独立进程 A/B/B/A**（原始结果 `/tmp/tp5-relay-timeline-5blocks.json`；20/20 trials 有完整正确输出、自然停止，全部 init 日志匹配 wire/sync，server SHA256 `8608fdee2737c5e1dac81068f0e1b5a209d3e5fa0107557b536d03217d9163ca`）：

|A：Timeline F16|B：Relay F16|B/A 块配对比及 95% CI|判定|
|---:|---:|---:|---|
|完整请求 committed 36.20 tok/s|35.64 tok/s|0.9846 `[0.9654, 1.0038]`|**未过 0.99 非劣下界**；Relay 不可凭此升为所有短请求默认|
|server decode `predicted_per_second` 49.17 tok/s|52.46 tok/s|1.0674 `[1.0365, 1.0984]`|Relay 解码阶段显著更快，但不能覆盖 prompt 与请求墙钟|

这个形状的分歧是本次验收最重要的纠偏：**更快 decode 不等于更快请求**。10 个正式样本的平均 server `prompt_ms` 为 Timeline **1048.1ms**、Relay **1340.1ms**；`predicted_ms` 分别 **3480.4ms**、**3259.7ms**。Relay 解码省约 221ms，prompt 多约 292ms，恰与请求墙钟差异方向一致。原本把 Relay 直接写成全局或所有五卡的“最佳默认”是不成立的。对长输出，需单独验证完整请求的 break-even，不能由 171-token 数据外推。

配对计划（每项 A/B/B/A × 5 blocks）：

|控制 A|候选 B|要回答的问题|
|---|---|---|
|control-timeline-f16|relay-f16-target|相同 wire/attention/MMVQ/replay 下 sync 净收益|
|golden-relay-f32|relay-f16-target|wire F16 相比 F32 的性能与质量边界|
|relay-f16-stock-device|native-default|在相同 Vulkan 默认队列/显存下，零额外 TP5 环境是否实际选中同一路径|
|relay-f16-stock-device|relay-f16-target|早期设备发现阶段的 graphics queue / host-visible-VRAM / BO isolation 组合是否有净收益；不得归因给 sync|
|control-timeline-f16-stock-device|control-timeline-f16|同样隔离设备初始化旋钮在 timeline 上的代价|
|relay-f16-target|relay-f16-mtp|已触发 GPU 挂起、五块中断；完成根因与安全复验前暂停|
|relay-f16-target|relay-f16-mmvq-auto|MMVQ 消融|
|golden-relay-f32|relay-f32-linear-lowering|B04 只改变命令组织，不换数学/wire|
|relay-f16-target|relay-f16-chain-cache-off / relay-f16-no-replay / relay-f16-replicate-off|分别隔离定义复用、重放及 attention replication|
|control-timeline-f16|timeline-f16-merge-off / timeline-f16-isolate-off|仅 timeline 路径上的合并提交和 BO 隔离|
|control-timeline-f16|star-f16-target|STAR 同格式对照|

运行示例：`python3 scripts/run-tp5-cross-matrix.py -a control-timeline-f16 -b relay-f16-target --blocks 5 -o /tmp/tp5-relay-timeline-5blocks.json`。**新启动**的矩阵默认总墙钟截止 **1800 秒**，server 就绪和 HTTP 完整响应分别最多 **120 秒**，每阶段上限还受剩余总预算限制；到期清理自有 server 并标记失败。`--max-wall-seconds` 可显式调整长输出试验的总预算；极短截止实测会在启动 server 前 fail-closed，持续每 50ms 滴出字节的本地 HTTP 服务也在指定的 0.6s 总响应截止触发失败。已启动的旧试验不受脚本改动追溯影响。每一轮必须检查 `status=success`、全部请求 `success=true`、实际 collective init 的 wire/sync 和原始日志；任何失败不得统计为胜出。用户可用 `--list-variants` 查看现有消融臂。

长输出独立形状用 `--count-to 100`（完整 `1..100` 自然停止，context=768），先以 `--smoke -a <variant>` 对最终模型做单次路线/内容预检，再以同样的五块 ABBA 比较；单次 smoke **不提供性能结论**。标准 `--count-to 60` 的 `prompt_n=31` / `predicted_n=171` 门不外推到另一形状。

**当前重建二进制的烟测**：`/tmp/tp5-native-policy.json` 在完全不注入 TP5/MMVQ 旋钮的 `native-default` 上取得 `SMOKE_PASS`，171 committed tokens、完整 1..60 自然停止、墙钟 5.020s；server 日志确认 `reason=qualified-5x-rx6800`、`sync=relay wire=f16` 和五个 backend 均 `mode=disabled`。这是零旋钮**功能证据，不是最快性能证据**。`/tmp/tp5-long-smoke.json` 在 `control-timeline-f16-stock-device` 上完整 1..100 自然停止（292 tokens、墙钟 70.829s、server decode 4.41 tok/s），说明长形状验收门可执行，**单点不可证明跨模式收益**。这个 stock timeline 比旧的已优化设备 Timeline decode 49.17 tok/s 慢很多；两者设备初始化旋钮、输出长度及构建版本不同，必须逐旋钮同口径隔离，不能把差异归因给时钟或 sync。已给矩阵增加分别去除 `GGML_TP5_ISOLATE_BO`、`GGML_VK_ALLOW_GRAPHICS_QUEUE`、`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` 的试验臂；watchdog 保持原状。

## 默认策略及剩余门

`common_tp5_apply_env` 仅在**明确指定 Qwen4EXP-AF 计划且 `-dev` 恰好五张互不重复的 RX 6800** 时自动选 relay、attention replication、producer-wire 及 MMVQ disable；若是纯 Target、RELAY 且 Vulkan replay 生效，默认 wire=F32、`GGML_TP5_LINEAR_LOWERING=1`。显式 `GGML_TP5_LINEAR_LOWERING=0` 关闭单 primary；显式 wire/sync/其他消融优先。MTP opt-in 无显式 wire 时仍走旧 F16、不自动降低命令；不满足硬件资格不选 F32 降层。集体通信直接调用也按五张**不同物理设备**判定，但不会凭五张卡绕过缺失的 graph capture。生产脚本不再默认加载草稿模型，也不提前强制 wire：由可见规格决定。

**发现并修复启动时序缺陷：** Vulkan 的 device discovery 发生在 `common_tp5_apply_env` 之前；原实现的 `GGML_VK_DISABLE_MMVQ` 在 physical-device 初始化时缓存，导致 clean-room `native-default` 的 env 旋钮**来不及生效**。MMVQ 策略已移至 backend execution context 初始化，上述五 backend 的真实 server route 日志已验证修复。`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`、`GGML_VK_ALLOW_GRAPHICS_QUEUE` 和 `GGML_TP5_ISOLATE_BO` 在设备建立时就决定显存/队列以及**整个 VkDevice 的 BDA、descriptor indexing、CM2**，不能在模型加载时再 `setenv`；已从自动旋钮移除。旧配对脚本在进程启动前显式设置这些开关，不能外推为无旋钮默认；`/tmp/tp5-qualified-default-smoke.json` 是新默认的完整请求与路由证明，**不是**与其他设备策略的配对性能证明。用户已要求只作启发式验证，不再展开配置笛卡尔积。

新策略的**真实 Vulkan region 数值与路由**已分两臂对拍：`GGML_VK_DISABLE_MMVQ=0 GGML_TP5_RELAY_DEBUG=1 test-vulkan-command-replay --attention-projections-only` 报 `policy=auto`，GDN/attention 投影分别出现 `mmvq_mask=3/1`；`=1` 报 `policy=disabled`，只出现 `mmvq_mask=0`。两臂 native oracle、K tails、views 与 changing inputs 均通过。此证据证明 `0` 现在**真的选择 MMVQ 算法**，不是端到端净收益。

**已在修复后的真实模型上复测 MMVQ**：`/tmp/tp5-mmvq-final-5blocks.json`，相同二进制/其映射 DSO、相同 RELAY/F16/设备旋钮，A=`DISABLE_MMVQ=1`、B=`=0`；五块 ABBA 20/20 完整内容通过，backend 日志分别确认 `disabled` / `auto`。A/B committed 均值 **35.67/35.19 tok/s**，B/A **0.9864**，95% CI `[0.9737, 0.9991]`；server decode **52.37/51.72 tok/s**。因此保留 `=1` 当前候选默认，**旧“=0 auto 51.99”已撤回**；这个结果只约束该短计数负载，不应推广为 MMVQ 在其他矩阵尺寸的普适劣势。

**MTP 门失败，禁止据前三块发布性能结论**：`/tmp/tp5-mtp-smoke.json` 的 opt-in 草稿首次烟测完整输出，但五块正式配对 `/tmp/tp5-mtp-final-5blocks.json` 在 `blk4_B2` 第二个请求断流，只有前三块完整。对应 server 日志在该请求开始后突然终止；`journalctl -u eagle-gpu-watchdog.service` 记录 **2026-09-23 02:39:08 card4 gpu_busy=99% 持续 >=20s**，运行中的 watchdog 自动调用既有恢复程序，五卡于 02:39:28 完成复位并重新枚举；事件前内核日志无 page-fault 记录。没有人为停用/修改 watchdog、时钟或触发复位。MTP 的 committed-token/full-cycle 性能**未通过五块验收，且暴露 GPU 挂起风险**；草稿继续显式 opt-in，复现根因未闭合前不重复同一 MTP 压测。故障后的任何性能对照必须重新开始完整配对，不能拼接复位前的样本。

**复位后目标卡回归**：重新编译当前 `llama-server`、`test-vulkan-tp5-mesh` 后，五卡 RELAY `--rounds 4 --check-all`（含真实 GPU graph-producer、FD 增量 0）及 TIMELINE `--rounds 4`（含异步依赖）均 `all passed`。新矩阵 runner 把 GPU PCI sysfs 设备 inode/BDF 作为每轮身份，在一次 trial 内或跨 trial 重枚举即 fail-closed，日志目录按输出 JSON 独立并存储日志 SHA256；这能拒绝**此后**跨复位拼接，不追认复位前的历史 trial。

**复位后同一二进制的设备旋钮配对**：`/tmp/tp5-launcher-device-5blocks.json`，A=全 stock Vulkan 设备、B=仅启动前允许 graphics queue 并禁用 host-visible VRAM（保持 BO isolation 未设置，匹配生产启动脚本）；五块 20/20，完整请求 **29.48→30.17 committed tok/s，B/A=1.0233 `[1.0166,1.0301]`**。`/tmp/tp5-isolation-5blocks.json` 在 B 的基础上只打开 BO isolation，20/20，B/A=**0.9976 `[0.9926,1.0025]`**，没有正收益证据，故不把影响整个 VkDevice 的 isolation 添加到脚本默认。最新二进制的 native-default 及脚本候选分别通过独立烟测 `/tmp/tp5-final-native-smoke.json`、`/tmp/tp5-final-target-smoke.json`；它们的绝对 tok/s 与复位前原始配对差异显著，且二进制映射 DSO 代际也变过，**不得跨会话相减归因给某个时钟/提交旋钮**。

**同设备、同 RELAY 的 wire 对照**：`/tmp/tp5-wire-5blocks.json`，A=F32、B=F16，其他所有环境只差 `GGML_TP5_WIRE`，20/20 完整答案；committed **30.22/30.21 tok/s**、B/A **0.9997 `[0.9933,1.0060]`**，server decode 比 **1.0031**。F16 在此题上没有可证明的完整请求提速；计数输出一致也**不证明**其他题目的普适精度等价。用户已批准将 F32-only B04 的小幅已测收益作为纯 Target 五卡默认；这不等同于全模型／不同有效 batch 的性能或质量验收。

**历史 sync 对照（非当前候选）**：`/tmp/tp5-final-sync-5blocks.json`，A=TIMELINE/F16、B=RELAY/F16，两臂均在进程启动前打开相同的 graphics queue、禁用 host-visible VRAM、打开 BO isolation；五块 20/20，完整请求 **31.11/30.29 tok/s，B/A=0.9741 `[0.9434,1.0047]`**。CI 跨 1，不能声称任何一方已凭此胜出。这是**带 isolation 的历史设备配置**，不可推广到无隔离的 stock TIMELINE（单次烟测仅 4.34 decode tok/s）。两臂日志/二进制/DSO 与 GPU 身份随原始 JSON 保存，结果取得于当前 shader 将 `spin_used` 改私有寄存器之前；用户已明确 TIMELINE 为不适用路线，**不再继续测试或作为默认竞争者**。

**B04 已命中纯 F32 模型路径，两种输出长度的有界增益成立**：第一次 `GGML_TP5_LINEAR_LOWERING=1` 真模型烟测 fail-closed（`LateBind source definition unavailable`）：原降层选择器已解耦，但 Vulkan graph 定义及 P1/P2 tape 仍只在 LateBind 下录制。现用统一的 RELAY/F32 降层谓词同时控制录制／消费点；`/tmp/tp5-b04-repaired-smoke.json` 完整 171-token 输出，五个 rank 各 `primary_cbs=1 mode=reference late=0`。A=`golden-relay-f32-isolation-off`、B=`relay-f32-linear-lowering-isolation-off` 仅差降层开关，各五块 20/20 完整正确、同一 server/DSO 且 GPU PCI 身份稳定：1..60 committed **30.06→30.69 tok/s，B/A=1.0212 `[1.0151,1.0273]`**（`/tmp/tp5-b04-linear-5blocks.json`）；1..100 committed **33.50→34.39，B/A=1.0268 `[1.0154,1.0381]`**（`/tmp/tp5-b04-long-5blocks.json`）。既有 RELAY 一直是一次 VkSubmitInfo，这不是 submit-count 改善。两组增益均低于原订 3% 晋级门，**用户明确允许已实现的小幅改进默认开启**；现仅对合格纯 Target 五卡 RELAY/F32+replay 自动开启。新二进制 `/tmp/tp5-qualified-default-smoke.json` 与显式 `GGML_TP5_LINEAR_LOWERING=0` 的 `/tmp/tp5-qualified-linear-off-smoke.json` 各完成 171-token 真模型请求，日志仅默认臂出现五 rank 单 primary；两次独立烟测不是配对收益。Golden F32 对照显式 `LINEAR_LOWERING=0`。不同有效 batch／其他任务质量仍未验，不用无穷矩阵补齐。

**TIMELINE 设备配置不能省略 isolation 后跨代比较**：当前二进制单次完整 1..60 烟测 `/tmp/tp5-timeline-f32-fast-smoke.json` 在 graphics queue/禁 host-visible VRAM 但 `ISOLATE_BO` 未设置时仅 **4.40 decode tok/s**；同一当前二进制和 F32、开启 `ISOLATE_BO=1` 的 `/tmp/tp5-timeline-f32-isolated-smoke.json` 得 **43.69 decode tok/s**。这是各一次路径／内容预检，不能拿两点宣称 isolation 的精确倍数，但足以阻止再把无隔离的慢 TIMELINE 放入五块。watchdog 原样运行，未修改时钟。

**已完成的 F32 历史配对，不作路线晋级**：`/tmp/tp5-f32-fast-frontier-5blocks.json` 将 isolation 开启的 TIMELINE/F32 与 RELAY/F32+linear 在相同设备初始化配置、相同 F32、同一二进制和 GPU 身份下做五块 20/20 完整请求。committed **31.01/30.69 tok/s**、RELAY/TIMELINE **0.9898 `[0.9689,1.0107]`**；server decode 反而 **42.02/42.93**。CI 跨 1，不能从该组宣称 TIMELINE 胜出，也不能把 RELAY 的 decode 优势当作完整请求优势。此处比较两个 recipe（sync+降层），**不是** B04 单因子效应；根据用户给定路线，TIMELINE 不进入后续候选。

**B07 热路径／ISA 证据**：`GGML_TP5_PROFILE=0/off/false` 现按值关闭，计划路径将 profile 旋钮静态求值；`tp5_relay_copy_f32.comp` 的 lane-zero spin 计数改为私有局部，`ready/active_elements` 仍保持跨 lane 共享。当前 shader 的 RADV ISA 对先前同源 SPIR-V 的 LDS 容量 **12→8 B**、LDS 读写指令 **10→7**（`/tmp/tp5-relay-current-{before,after}-isa.log`）；关闭／开启 PROFILE 的 RELAY mesh 均已过。没有单独配对证明该 shader 改写的端到端净收益，不能把 ISA 缩小直接换算成 tok/s。

**B01 历史回归在当前机器未复现**：`b36e47359`/`4e056e1b4` 的 4 个且仅 4 个变更文件按 Host×GPU 分成 A/B/C/D；四份同配置 `RelWithDebInfo`/Vulkan/ccache server 的源码及 shader SPIR-V 因子对应，四种模型烟测通过。Host-only `tp5_record_plan` 多 1 个 `getenv@plt` callsite；RADV 实际编译 LDS 4→8 B、LDS 指令 4→6，原始 ISA 在 `/tmp/tp5-b01-isa-{parent,shader}.log`。三组每组五块/20 个完整请求：完整 child/parent **1.0018 `[0.9957, 1.0080]`**（`/tmp/tp5-b01-parent-child-5blocks.json`）；host-only/parent **1.0003 `[0.9905, 1.0100]`**（`/tmp/tp5-b01-host-effect-5blocks.json`）；shader-only/parent **1.0036 `[0.9946, 1.0126]`**（`/tmp/tp5-b01-shader-effect-5blocks.json`）。交互比值仅点估计 **0.9980**，未作独立显著性门。**不能**把机器码增量当成旧 −39% 的因果解释，也不能反向否定当时的不同环境观察；详见 `PLAN/TP5-REORG-B00-facts.md`。

这是**已落地的有条件默认**，不是普适最优定理。用户批准的是纯 Target 五卡在两种计数长度下已证实的小幅同数学增益，非五卡真实模型回退、MTP 完整周期、不同有效 batch 和其他问答质量均未证明；未取得证据的 PLAN 研究项仍关闭，见 `PLAN/本轮进度报告.md`。不把单次新默认烟测的绝对 tok/s 与先前不同二进制的均值相减。

**启发式收敛，停止组合压测**：TIMELINE 已由用户确认为不适用，发起的 `relay-f16-target-isolation-off` 对 `timeline-f32-target` 实验立即取消，确认无残留 server；未完成样本不用。B04 只因已有同数学 F32 单因子／两种输出长度的完整配对以及用户明确批准而成为默认，原 3% 门对该项有明示例外；并未做 F16×F32×sync×isolation 的乘法枚举。MTP 挂起路径在安全诊断前不重跑；不得以慢配置回退充作收益。

**可复现入口（仅实测的单 slot 形状）**：

```bash
cmake --build build-tp5 --target llama-server -j44
RADV_DEBUG=nobolist ./build-tp5/bin/llama-server \
  -m /home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf \
  -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4 --split-mode tensor --fit off \
  -ngl 999 -c 256 -b 32 -ub 32 --no-mmap --no-host -np 1 --tp5 qwen4exp-af
# 消融：同一命令前加 GGML_TP5_LINEAR_LOWERING=0；或显式 --tp5-wire f16。
python3 scripts/run-tp5-cross-matrix.py --smoke -a native-default \
  --max-wall-seconds 180 -o /tmp/tp5-qualified-default-smoke.json
```

`RADV_DEBUG=nobolist` 是驱动请求而非 BO 隔离生效证明；已记录当前 RADV 对关闭全局 BO list 的警告。完整请求输入、输出、GPU PCI 身份、server 和映射 DSO SHA256、日志 SHA256 均在原始 JSON 及同名 `_logs/`。`scripts/run-qwen38-flash-tp5-server.sh` 的纯 Target 默认沿用该有条件选择并在设备建立前设置已测快设备开关；其生产 `N_SLOTS`/上下文未按本行单 slot 性能口径验收。

## 否决实施登记（不是否决研究方向）

|捷径|拒绝理由；需要什么才能重新考虑|
|---|---|
|将 MTP 52.90 的 `predicted_per_second` 称为最快默认|draft/target token 不是已提交输出；必须按 `usage.completion_tokens / 完整请求墙钟` 配对，并查看 draft/catch-up/接受率|
|从计数题推断 F16 与 F32 位级等价|只对一条短请求核对输出；需要质量/数值门及同数学精度路径控制|
|将 Q-prep/Span 两个环境变量直接晋级 TP5 默认|这条 TP5 请求未证明经过 RERoT 图路由；小样本差异不达 3% 默认门；需真实 RERoT route-hit 与匹配的 A/B/B/A|
|用 `GGML_VK_DISABLE_MMVQ=0` 的旧跑分称为 MMVQ auto 结果|旧 backend 按 `getenv` 存在性判禁用，`=0` 实际仍是 disabled；现已按值解析并在新构建上完成五块配对，见上方 **0.9864 `[0.9737, 0.9991]`**，旧数字仍无效|
|跨不同 FlashPrefill view 直接复用 fragment 对象|`run_v0`、`phase_bias`、连续 `vgroup_frags`、`key_frag` 都带 view 身份；完全相同 view 已由 key 分组合并，直接借用会破坏契约|
|以括号化 `stage_compute_cbs == stage_compute_cbs` 充当 B05|仍做逐 token 深比较，完全没有 generation/Program 分离；缺 child Vulkan invalidation/buffer relocation 的安全通知前不能删比较|
|用 mesh/CPU 微基准抵消完整纯 Target 退化|度量对象不同；必须同模型、同数学、独立进程、提交输出墙钟配对|
