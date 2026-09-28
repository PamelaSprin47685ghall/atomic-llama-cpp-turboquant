# MMVQ + MTP：组合效应验证与实施手册

编写日期：2026-09-27。对象：五张 RX 6800、RADV、Qwen4EXP TP5、当前自然文本负载。

状态：**PLAN / NOT RUN**。本次只写规划，没有运行组合实验，没有改生产代码、默认配置或 harness，没有 commit/push。

**要检验的是：MMVQ 在单 token 上不划算，但在 MTP 的多行 target 验证上可能划算，从而让原本收益很小的短 horizon MTP 变得有用。现有证据没有排除这个可能，也没有证明组合会有大收益。**
**【2026-09-28 晚更新：已按本手册规则执行完毕并关闭。四格 + h 扫描判决见文末 §21；
命中已证实（融合 attention 投影 gdn mask=3/qsa mask=1），但无可检定收益；
多行验证下更亏（h=2 慢 13.6%）。最优仍是 aggressive h=1 + disabled（现默认）。**

本文件是以后明确恢复实验时的操作手册，不是现在重启 MTP 或启动 GPU 的授权。先做 h=1 的四格对照；再根据真实 kernel 命中决定是否看 h=2/3。不要从 n10、全局 force 或修改量化权重起步。

---

## 0. 开工前先读这页

### 0.1 当前有效目标

- 唯一主验收负载是自然文本，不再拿计数题的高接受率代表真实收益。
- 现有固定自然问题为：`请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。`
- 现有请求为 temperature=0、seed=42、关闭 thinking、cache_prompt=false、自然停止。模板后 prompt_n=31；不能凭字符数推断 token 数。
- `1..60` 和原来的 1710 ms 门只保留为历史与回归线，不再作为本次产品晋级目标。
- 单个自然问题通过，只能声明在该问题上有效；要称自然文本普遍有效，必须另过预先冻结的多提示集合。
- 不将减重回答的内容作为本工程的医学结论；这里仅把该文本当作固定测试输入。

### 0.2 七条红线

1. **实际命中比开关名重要。** `mode=auto` 不证明真实 target verify 用了 MMVQ；测试强制路径通过也不证明 auto 会选它。
2. **不削弱对照。** 组合最终必须赢过当时合格的最好 no-MTP 配置，不只赢过一个被关闭融合、改慢布局的对照。
3. **不混数学。** MMVQ 的 activation Q8_1 会改变数值；MTP 不能纠正 target 自身因这项量化造成的误差。
4. **不混因素。** MMVQ 全局开关同时影响 target、draft、prefill 和 catch-up；MTP 开关还会连带改变布局与 sampler 路由。
5. **不从日志名字猜时间。** host wait、GPU compute、ready、publish 有重叠；链累计不是单 site，stage 数不是 dispatch 数。
6. **不破坏现场。** 不 reset/clean/stash 用户 dirty 树，不改 watchdog、时钟、功耗、驱动、PCI 拓扑，不广域 kill。
7. **失败保留，阶段止损。** 错误输出、未命中、无收益和样本排除都要留档；不反复抽样直到撞到漂亮数字。

### 0.3 文件优先关系

用户最新指令优先。当前对应源码、实际运行身份和原始证据优先于历史总结。

`AGENTS.md` 仍有 83→100 tok/s、计数 canonical 和先前任务排序等旧内容。本专题以本文件的自然文本目标和分阶段实验规则为准；这不是授权删除 AGENTS、重启已停任务或覆盖用户其他改动。

`交接.md` 的 MTP 退役是此前工作范围裁决。本文件记录新假设，**写下计划不等于已恢复实施**。以后收到实施授权才执行第 15 节命令。

---

## 1. 现有证据到底支持什么

证据编号的完整路径在第 20 节。路径行号会变化，优先按函数名、日志字段和实验目录定位。

| 编号 | 已核实事实 | 可以推断 | 不能推断 |
|---|---|---|---|
| E1 | 近期目录中 18 份带 A/B 配置的结果文件，共 36 臂；35 臂声明 MMVQ disabled，另 1 臂按 harness 默认检查为 disabled，原始日志亦确认 | 近期主实验集中在 MMVQ 关闭条件下 | 整个仓库从未测试过 MMVQ |
| E2 | G8 n1 与 no-MTP 两臂，target/draft 日志都是 disabled；n1 约 3263 ms、no-MTP 约 3361 ms，只有一块 pilot | 当时配置下 MTP 收益较小 | MMVQ 开启后的 MTP 也无收益；2.9% 是稳定普遍收益 |
| E3 | G7 五块 ABBA 的 1.646× 是 n1 对 n6；两边 MMVQ 都关闭 | 短 horizon 在该自然问题上明显优于 n6 | MTP 比 no-MTP 快 1.646× |
| E4 | 早期确有 MTP+MMVQ-auto 烟测，约 35.40 committed tok/s；单次、非配对、旧代码、计数负载 | 当时组合未显出足够的晋级收益 | 当前自然文本、当前内核上的交互作用已被否定 |
| E5 | Q5_K MMVQ 解包上移实验，两臂最终 ISA 相同；相关客户端参考还开着 DISABLE_MMVQ=1 | 那项 GLSL 改写没有改变真实执行 | MMVQ 本身或 MMVQ+MTP 家族无用 |
| E6 | selector 对受支持类型在 n>1 时倾向 MMVQ；但 AMD Q6_K、small-Q8 降层和类型覆盖另有门 | 多行验证确实可能改变选择和成本 | 所有多行矩阵都会自动加速 |
| E7 | G8 no-MTP 日志是 48-stage 链，n1 是 96-stage 链；默认 attention 复制策略不同 | 开 MTP 不仅增加一个草稿，还改变执行结构 | 一行成本乘二即可得到两行验证成本 |

这些都是既有证据，不是本次复跑。早期 `/var/tmp` 原始文件不在当前可访问范围时，仅将 `TP5.md` 的记录列作二级证据，不能假称重新解析过原始响应。

### 1.1 当前源码身份边界

写作时主仓为 `master@4a30c7e9b`，已有 TP5.md、harness、artifacts 和交接的未提交内容。此前实验主体另在隔离 worktree 中，不能把主仓 HEAD 当作那份二进制的完整源码身份。

`16e28b8ac` 是历史融合检查点，不自动包含后来 G4b 和自然文本 harness。以后先冻结本次实际选择的源码快照、补丁、构建和模型，再取当前 no-MTP/n1 对照；**不以重现计数题 83 tok/s 代替自然文本基线资格**。

### 1.2 对假设的当前判断

值得先补一个小的交互实验；更看好 **h=1/2/3、两到四行 target verify 的选择性 MMVQ**，不押注 n10 和全局强制模式。

这只是实验排序。不得在报告里提前写“组合必有大用”“auto 打开就会提速”，也不得写“两个单独无用，所以组合必然无用”。

---

## 2. 术语和形状：先把轴写对

### 2.1 本文的 MMVQ 指什么

当前 Vulkan 相关路径先将合格 F32 activation 转成 Q8_1，再选相应的量化收缩 pipeline。它不是更换 GGUF 权重量化，也不等于“只要矩阵有多行就是 MMVQ”。

关闭 MMVQ 仍有原生浮点多列 GEMV、权重解量化和跨列复用。权重本来就是量化格式，KV/其他专用内核也有各自精度；`MMVQ disabled` 不等于全模型每一步都是纯 F32。

本文只讨论这个仓库的 Vulkan 实现，不把其他 backend 的 MMVQ/MMQ 名称、性能或支持表搬过来。

### 2.2 统一记号

| 记号 | 含义 | 容易混淆的对象 |
|---|---|---|
| h | 一轮最多提案的草稿 token 数，即 horizon | MoE 专家数、矩阵的 N 维 |
| rows | 一次实际 verify 的有效 token 行数；通常不超过 h+1 | 配置容量、输出行 tile、工作组数 |
| M | 当前权重输出维 | token 数 |
| N | 当前收缩实际并行处理的 activation 列数 | horizon 本身 |
| K | 内积长度 | GDN snapshot 数 |
| mat_vec_cols | selector/核实际看到的列数 | 原图逻辑 token 行数 |
| outer_batches | 外层批维，可能是 token/lane/其他 batch | 同一个多列工作组共享权重 |
| L | 完整请求生成时间/公开 completion token，或明确标记的固定轨迹成本 | 内部 accepted、drafted、bonus 计数 |

必须同时记录原图 shape 与 lowering 后 shape。`h=1` 通常对应两行 verify，但 EOG、预算、拒绝和末轮可能改变实际行数；按执行日志算，不按 h+1 填表。

### 2.3 各 phase 独立记账

`TARGET_PREFILL`、`TARGET_SINGLE`、`TARGET_VERIFY`、`DRAFT_STEP`、`DRAFT_PREFILL`、`CATCHUP_KV` 分开记录。

全局 auto 可能只让 prefill 或 draft 命中。那不算“target 多行验证已用 MMVQ”。同一张物理卡上的 target backend 与 draft backend 要有不同 context/role 身份，不能只打印 Vulkan0。

---

## 3. 两个问题、两套实验，不要混为一个结论

### 3.1 P 系列：产品组合有没有实际价值

固定 h=1，四格记为：

| 格子 | MTP 配置 | MMVQ 策略 | 要回答什么 |
|---|---|---|---|
| A0 | 合格 no-MTP 产品配置 | disabled | 当前基础对照 |
| A1 | 与 A0 完全相同 | auto | 单独开启 MMVQ 的价值 |
| B0 | 合格 h=1 MTP 产品配置 | disabled | 当前短 horizon MTP 对照 |
| B1 | 与 B0 完全相同 | auto | 组合价值 |

A0/A1 内只改变 MMVQ；B0/B1 内也只改变 MMVQ。P 系列允许保留 A、B 各自已经合格的布局和 sampler，但必须列出差异。因此它测的是**产品配置的交互作用**，不是纯粹的单一内核因果效应。

B1 必须与 `min(A0,A1)` 所代表的较好 no-MTP 配置比较，而不只与 B0 比较。延迟的 min 只能用于选待验证对照，不能事后挑最快单个样本；选定配置后另做正式配对。

### 3.2 M 系列：收益来自多行 MMVQ，还是其他变化

在相同固定 token 轨迹、相同 target verify 图/状态上比较 off/auto。额外核验：

1. target 的实际收缩是否改变，量化输入是否复用，GPU 完整 verify 是否更短。
2. draft、catch-up、sampler 和接受长度是否改变。
3. attention 复制、headmap、融合覆盖、stage 数和内存驻留是否改变。
4. auto 是否只改善 prefill，而 warm generation 没变。

需要只开 target、只开 draft 时，必须新增独立 backend-role 策略并测试；**当前进程级环境变量不能假装已支持这种隔离**。不要在两个 context 初始化之间临时改全局 env：静态缓存、设备复用和线程并发会使行为难以证明。

M 系列不得为了同形对照而削弱最终产品 A1。固定轨迹实验是归因工具，不替代自由生成和最强产品对照。

### 3.3 什么才算支持“单独小、组合大”

用同一预先指定指标、同一类请求得到四个格子的成本：

```text
单独 MMVQ 节省：  S_A = L_A0 - L_A1
MTP 中 MMVQ 节省：S_B = L_B0 - L_B1
加性交互：         I = S_B - S_A
组合实际收益：     G = 1 - L_B1 / L_best_no_mtp
```

I>0 表示按该成本尺度，MMVQ 在 MTP 中省得更多；但 I>0 不保证 B1 是最优配置。还必须 G>0、质量/正确性通过。

相对尺度另报告：`J=log(L_B0/L_B1)-log(L_A0/L_A1)`。I 和 J 可给出不同强弱，预先指定哪一个是主要交互指标，不事后换尺度找显著。

“未检出单独收益”不等于“证明单独无用”；要声称等效，须预先给等效区间并做相应统计。正交非劣成员也不能被宣传成独立提速。

---

## 4. 源码路由地图与最容易踩的开关坑

### 4.1 disable 的优先级高于 force

`common_tp5_apply_env()` 在合格 TP5/RELAY 情况下会给未设置的 `GGML_VK_DISABLE_MMVQ` 默认写入 1。后端选择顺序是 disable → force → auto。

| 目的 | DISABLE_MMVQ | FORCE_MMVQ | 预期启动策略 |
|---|---:|---:|---|
| off 对照 | 1 | 0 | disabled |
| auto 候选 | 0 | 0 | auto |
| 仅定向 force 诊断 | 0 | 1 | forced；仍受类型/资源支持限制 |
| 错误组合 | 1 或被默认写成 1 | 1 | 仍可能 disabled，不是 force 实验 |

这些值规则只用于已确认按数值解析的两个开关。仓库还有按 `getenv()!=nullptr` 判断的变量；**不要普遍采用“所有诊断变量设 0 就等于关闭”**。白名单外变量应不传入。

### 4.2 auto 的重要覆盖边界

| 对象 | 当前可访问源码的行为 | 实验要求 |
|---|---|---|
| Q5_K 等受支持类型、多列 F32 contiguous 输入 | 满足设备/对齐条件后，n>1 倾向 MMVQ | 记录实际量化和收缩，不仅看策略行 |
| Q6_K、AMD | auto 在多列判断之前有排除；force 可越过偏好，不越过基本合法性 | 先做单 shape 诊断，不全模型强开 |
| Q8_0、RDNA2 单列 | auto 通常不选择 MMVQ | 检查逻辑多行是否被降成 selector 的单列 |
| IQ4_NL、IQ4_XS、PQ2_0、部分 IQ2/IQ3 等 | 不在当前通用 Q8_1 MMVQ 支持表 | 不把相关 MoE 或权重时间计入可加速覆盖面 |
| F32/F16/BF16 收缩、其他专用路径 | 有各自内核；不因 env=auto 就变成 MMVQ | 单独计为 unchanged 或明确的专用路径 |
| 非连续/对齐不满足/无合法 pipeline | 可能回退 | 保留回退并输出原因，不能删守卫硬跑 |

类型支持表以后可能变化；每次以实际冻结源码的 `ggml_vk_should_use_mmvq()` 和 `ggml_vk_get_dequantize_mul_mat_vec()` 为准。[E6]

### 4.3 small-Q8 是重点，不是可随意删除的障碍

当前 small-Q8 路径可把逻辑多列改为 `skinny_as_batch=true` 的外层 batch。该分支不参加普通 outer-batch 展平成多列的路径，selector 的 `mat_vec_cols` 可能是 1。

因此，h=1 的逻辑两行 HC Q8 收缩也可能没命中 auto。这是核验点，不是已经证明 selector 有 bug。原来的 small-Q8 路径有自身资格，不能直接关闭它来制造 MMVQ 命中。

定向调查顺序：实际 shape → 原生路径耗时 → auto 命中/回退原因 → 合格的 force 单核对照 → 保留布局的局部策略候选。只改 shape 或只改选择器，一次一项。

### 4.4 融合不应被低层选择静默拆掉

现有规则是 region 先按自己的数学、形状和别名条件取得节点，再在 region 内选择收缩实现。MMVQ 不应以局部 heuristic 否决整个 HC/MoE/Attention/GDN region。[E9]

但是多行形状本身可能不满足当前融合条件。要区分：

- off/auto 相同 shape，auto 意外拆融合：候选回退，先修合法的组合边界。
- 一行融合、两行不融合：记录为形状/覆盖差异，不冒称 MMVQ 必然导致。
- 多行核已经逐列共享权重：不再把“第一次实现权重复用”写作新收益。

### 4.5 MMVQ 与 ARGMAX 是两件事

hierarchical ARGMAX、G4b host verify argmax、GPU candidate 不是 MMVQ。前者处理采样/选择，后者改变矩阵收缩。日志中的 `argmax enabled` 不能证明 MMVQ 命中。

G4b 属于后来实验源码；harness 存在 varargmax 名称不证明当前 binary 实现了它。若 B0/B1 使用 G4b，必须固定同一实现、同一资格条件和同一实际采样链；不拿它相对 A 的额外收益冒充 MMVQ 的独立收益。

---

## 5. 冻结配置表：每个格子都必须填写

先填 resolved 值，不只记录命令行输入。空白表示未完成，不表示默认相同。

| 项目 | A0 | A1 | B0 | B1 | 约束 |
|---|---|---|---|---|---|
| server/实际 DSO/shader 身份 | 待填 | 待填 | 待填 | 待填 | 第一轮四格同一 binary/DSO；禁止旧库混用 |
| target GGUF 全部分片 | 待填 | 同 A0 | 同 A0 | 同 A0 | 类型、内容、模型配置一致 |
| MTP GGUF/放置 | 无 | 无 | 待填 | 同 B0 | 单卡 draft；不同时改草稿量化 |
| sync/wire | RELAY/F32 | 同 A0 | 同 A0 | 同 A0 | 不借 F16 wire 变体做 MMVQ 臂 |
| linear lowering/replay | 待填 | 同 A0 | 待填 | 同 B0 | 保留各自已验证实现 |
| MMVQ effective policy | disabled | auto | disabled | auto | 后端逐角色核验 |
| horizon/实际 rows 分布 | 无 | 无 | h=1 | h=1 | 实际 rows 单独记录 |
| attention replicate/headmap | 待填 | 同 A0 | 待填 | 同 B0 | A/B 间允许产品差异但必须公开 |
| target stage 数/融合覆盖 | 待填 | 待填 | 待填 | 待填 | 不预填固定 48/96，按新日志核验 |
| small-Q8/producer-wire | 待填 | 同 A0 | 待填 | 同 B0 | 不随 auto 顺带改 |
| context/batch/ubatch/slots | 待填 | 同 A0 | 待填 | 同 B0 | 产品比较使用冻结相同请求容量 |
| KV 类型、rollback 配置 | 待填 | 同 A0 | 待填 | 同 B0 | 不能把状态压缩混进候选 |
| sampler 实际链/seed/约束 | 待填 | 同 A0 | 待填 | 同 B0 | temp=0 不自动等于纯无状态 greedy |
| draft p_min/backend sampling | 无 | 无 | 待填 | 同 B0 | 不将阈值变化混进接受率 |
| LateBind/ZSNAP/RHO/诊断变量 | 关闭或不存在 | 同 A0 | 同 A0 | 同 A0 | LB 系列不进入本实验 |
| GPU UUID/BDF、驱动、运行负载 | 待填 | 待填 | 待填 | 待填 | 设备变化/复位按预定规则废弃整个块 |
| VRAM/GTT/host pinned 峰值 | 待填 | 待填 | 待填 | 待填 | 量化 scratch 不得触发驻留悬崖 |

表中的 ZSNAP 指 `GGML_TP5_LATEBIND_ZSNAP`。其余缩写也应展开为源码中的实际变量名后核验。

若 A/B 的 sampler 或 attention 布局不同，最终结论写“产品组合”，不写“只由 MTP 算法导致”。需要严格机制结论时执行 M 系列。

---

## 6. 实施工单与依赖

| 工单 | 要交付什么 | 不通过时怎么办 |
|---|---|---|
| Q0 基座与来源 | frozen manifest、当前 A0/B0、A/A 漂移检查 | 停止性能排序，先查身份/环境/默认回退 |
| Q1 四格与 harness | 合法 F32 variants、严格路由检查、mock/dry-run | 未命中/混因素不准进性能门 |
| Q2 数值与状态 | 单核、固定轨迹、所有受影响状态、采样/取消回退 | 留失败证据，禁止用变体集掩盖错误 |
| Q3 h1 产品 pilot | A0/A1、B0/B1、组合对最好 no-MTP 的完整结果 | 明显负收益停止当前配置；缺命中转 Q4 |
| Q4 命中与形状归因 | 每 phase/type/shape 路由和完整 verify 成本 | 若覆盖面小到无意义，关闭该候选 |
| Q5 定向策略 | target/draft 隔离或局部形状 selector，单因素补丁 | 不与新量化/融合/草稿模型一起提交 |
| Q6 h2/h3 筛选 | 仅必要的三、四行验证 | 接受收益不足或内存恶化时停止加长 |
| Q7 正式资格 | 独立 block、产品胜出、交互证据和多提示质量 | 只通过 pilot 就留 pilot，不晋级 |
| Q8 整合与回退 | 组合重验、默认范围、退出机制、最终 decision | 没有净收益不保留无谓开关和新维护负担 |

推荐顺序：Q0→Q1→Q2 最小门→Q3；再依据结果选择 Q4/Q5/Q6；候选定型后补齐 Q2 全门并完成 Q7/Q8。CPU 分析可以并行，GPU 实验只能串行。

每个任务以完整交付为阶段结束，不以“代码已写”“单核更快”“关了几个等待函数”算完成。

---

## 7. Q0：先站稳对照，防止沿用 LB 的漂移问题

### 7.1 源码和构建身份

保存：主仓与实验 worktree 的 HEAD、branch、dirty patch、未跟踪源码、submodule 状态（存在时）、CMake 配置、编译器、shader compiler、server 哈希、实际加载 DSO 哈希。

server 同名路径可以被重编；共享库也可以被其他 worktree 替换。`--version`、HEAD 和二进制文件名都不足以证明身份。实际进程映射里没有出现的库，不用“猜测可能加载”替代取证。

若未来使用含 G4b 的 dirty 实验树，必须有可重建快照及单因素列表。不要把未审核的 LB1/LB2 plumbing 自动带进 MMVQ 四格，更不要用较慢的 feature-off 回归版当作新基线。

### 7.2 A/A 先验

选同一 binary、同一 variant，在独立服务进程交错重复，区分：进程首请求、同进程后续请求、请求内首次 shape、稳态执行。

建议预注册漂移警戒线：相同控制配置块间成本漂移超过 3%，或超过准备判定的候选收益一半时，暂停小收益判定。该数是实验建议，不是已证明的硬件噪声上界；可在见候选结果之前根据 A/A 固定更合适的阈值。

排查优先顺序：binary/DSO/生成 shader → inherited env/默认解析 → 同时运行的服务/编译/哈希 → replay/形状/融合 → 显存驻留 → 只读观察的温度/频率/CPU调度。不能看到漂移就先改 GPU 时钟。

“同 session”不等于交错配对；AAA 后 BBB 的顺序实验，只是 pilot。只有一个独立 block，也不因为每进程跑了十次就变成十个独立 block。

### 7.3 机器前置门

确认五张目标 GPU 身份、watchdog active、没有其他推理任务；不接管或终止别人的进程。模型哈希、文件拷贝和构建全部在计时前结束。

出现 GPU fault、device lost、复位、BDF/UUID 变化、请求挂死或不明进程残留，立即停止本候选后续任务；按既有安全恢复流程由有权限的操作者处理，不在 benchmark 中自动复位追求连续样本。

若工具不允许访问某个目录，记录缺失证据并停止相关核验；不从 shell 或符号链接绕过权限边界。

---

## 8. Q1：最小 harness 改动说明

### 8.1 已有能力与缺少的能力

当前脚本已有 `--variant-a/--variant-b`、`--bin/--bin-b`、`--workload weightloss`、`--smoke`、`--pilot`、`--dry-run`、blocks/repeats 和进程所有权管理。[E8]

**没有本专题命名的四格 variants，也没有原生四格交互统计 runner。** 下文给出以后添加的代码样例；本次没有修改脚本。不要直接运行尚不存在的 variant 名称。

现有 `relay-f16-mmvq-auto` 改了 wire、其他默认和 linear 条件，不能拿它当 A1/B1。复用其名字会把多个因素混起来。

### 8.2 建议新增的四格注册函数（将来实施）

将函数放在脚本模块级；在 `get_available_variants()` 所有基础 variants 定义完成、return 之前调用 `register_mmvq_mtp_variants(variants)`。先只注册 h=1；确有必要时传入 `(1, 2, 3)`。

该样例按产品配置保留 B 的 G4b，A 使用当前 native-default。它不负责证明 G4b 数值资格，也不实现 target/draft 分离；这些由 Q0/Q2/Q5 负责。

```python
def register_mmvq_mtp_variants(variants: dict, horizons=(1,)) -> None:
    """未来添加：产品四格，仅比较显式 off/auto，不强开 MMVQ。"""
    from copy import deepcopy

    def set_arg(argv, key, value):
        argv = list(argv)
        positions = [i for i, item in enumerate(argv) if item == key]
        if len(positions) > 1:
            raise ValueError(f"duplicate CLI option: {key}")
        if positions:
            i = positions[0]
            if i + 1 >= len(argv):
                raise ValueError(f"missing CLI value: {key}")
            argv[i + 1] = str(value)
        else:
            argv.extend([key, str(value)])
        return argv

    requested = tuple(horizons)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("horizons must be non-empty and unique")
    if any(h not in (1, 2, 3) for h in requested):
        raise ValueError("initial matrix only permits horizons 1, 2, 3")

    pending = {}

    def add(name, base_name, enabled, mtp, horizon=None):
        if name in variants or name in pending:
            raise ValueError(f"variant already exists: {name}")
        if base_name not in variants:
            raise KeyError(f"missing audited base variant: {base_name}")
        item = deepcopy(variants[base_name])
        env = item.setdefault("env", {})
        # 基础配置只允许保留已审阅的声明；不要继承未知实验 plumbing。
        allowed = {"RADV_DEBUG", "GGML_TP5_SPEC_VERIFY_ARGMAX"}
        unexpected = set(env) - allowed
        if unexpected:
            raise ValueError(f"audit new base env keys first: {sorted(unexpected)}")
        env.update({
            "GGML_TP5_SYNC": "relay",
            "GGML_TP5_WIRE": "f32",
            "GGML_TP5_LINEAR_LOWERING": "1",
            "GGML_VK_CMD_REPLAY": "1",
            "GGML_VK_DISABLE_MMVQ": "0" if enabled else "1",
            "GGML_VK_FORCE_MMVQ": "0",
        })
        argv = item["cli_args"]
        argv = set_arg(argv, "--spec-type", "draft-mtp" if mtp else "none")
        argv = set_arg(argv, "--tp5-sync", "relay")
        argv = set_arg(argv, "--tp5-wire", "f32")
        argv = set_arg(argv, "--tp5-latebind", "off")
        if mtp:
            argv = set_arg(argv, "--spec-draft-n-max", horizon)
        elif "-md" in argv or "--spec-draft-n-max" in argv:
            raise ValueError("no-MTP base unexpectedly carries draft options")
        item.update({
            "name": name,
            "description": f"QMP product matrix: mtp={mtp}, h={horizon}, mmvq={'auto' if enabled else 'off'}",
            "cli_args": argv,
            "expected_sync": "relay",
            "expected_wire": "f32",
            "expected_mmvq": "auto" if enabled else "disabled",
            "is_mtp": mtp,
            "requires_linear": True,
        })
        pending[name] = item

    add("qmp-a0-nomtp-off", "native-default", False, False)
    add("qmp-a1-nomtp-auto", "native-default", True, False)
    for h in requested:
        base = f"native-mtp-fused-n{h}-varargmax"
        add(f"qmp-b0-h{h}-off", base, False, True, h)
        add(f"qmp-b1-h{h}-auto", base, True, True, h)
    variants.update(pending)
```

不要用 `cli_args[:-1]` 猜 horizon 在最后一项，也不要浅复制 env 导致改 B1 时污染 B0。将来基础配置多出新变量，上述函数故意报错，要求先审阅再扩展白名单。

代码没有显式设置 ZSNAP/RHO 等未知解析方式的开关；通过干净环境不继承它们，并以 `--tp5-latebind off` 保持 LateBind 关闭。若底层新代码在关闭情况下仍改行为，属于 Q0 feature-off 回归，不是本专题的“正常默认”。

### 8.3 必须补的 harness 验证

给四格配置做 CPU-only 断言：

| 检查 | 期望 |
|---|---|
| off/auto 对内的 env/CLI 差分 | 只差 MMVQ 策略与描述/expected 字段 |
| A0/A1 没有 MTP 模型、草稿 horizon | 是；不是 `n_max=0` 冒充 no-MTP |
| B0/B1 的 h、draft 模型、p_min、sampler、设备 | 完全相同 |
| 四格 wire/sync/linear | F32/RELAY/开启 |
| 四格 force | 都为 0，不被默认 disable+force 抵消 |
| 基础 variant 的嵌套 dict/list | 注册前后未被污染 |
| 未知名称/错误路由/缺策略日志 | fail-closed，不跳过校验 |
| dry-run | NOT_MEASURED；不启动模型，不生成虚假成功 |

当前 policy 检查偏向检查“见到过合法策略行”。专题正式门还需确认**全部五 target rank，以及实际 draft context**都符合预期；缺一个 role/rank 不能由其他行替代。

M 系列如果新增 per-role 策略，旧单一 `expected_mmvq` 不足以表达，必须增加 role-aware 检查，不能把校验关掉。

---

## 9. Q2：正确性与数值门，分三层执行

### 9.1 第一层：现有小测试先跑

先检查当前构建确实注册了 test-sampling、test-arg-parser、test-mtp-workspace、test-tp5-plan；再逐个跑所需 GPU 定向测试。

已确认存在的 replay 测试入口：`--skinny-only`、`--attention-mmvq-only`、`--attention-projections-only`、`--attention-region-only`、`--moe-only`、`--moe-output-only`、`--hc-fold-only`、`--hc-combine-only`、`--gdn-only`、`--gdn-cache-only`、`--rows-only`、`--transfer-only`、`--argmax-only`。[E10]

**重要：`--skinny-only` 与 `--attention-mmvq-only` 在 main 中主动清除 DISABLE 并设置 FORCE=1。** 它们是数值/保活测试，不是 auto 策略或性能对照。`--attention-mmvq-only` 还会检查相关 region-MMVQ 能力；return 77 是跳过，不能写通过。

测试可用不等于新 shape 已覆盖。实际两、三、四行、每 rank 不等长分片及本次真实量化类型，需要补成对应 fixtures。不要仅用四行 Q5 合成算子代替整个 MTP 验证图。

### 9.2 第二层：原始输入/布局/重放测试矩阵

| 维度 | 必测内容 |
|---|---|
| token rows | 1、2、3、4；7 只作已有回归；partial/tail/零 logits catch-up |
| 形状 | 实际 M/N/K、输出分片、非整齐尾块、outer-batch 与普通列表示 |
| 类型 | 实際命中的 Q5_K/Q8_0 等；Q6_K 定向诊断；不支持类型正确 fallback |
| 输入布局 | contiguous、合法 view offset、原路径支持的 strided fallback；guard bytes |
| 转换复用 | 连续兄弟投影共享同一输入；相同 tensor 地址但内容更新；不同输入连续调用 |
| cache 身份 | 禁止临时栈 view 指针成为跨调用 identity；scratch 扩容后不能用旧转换 |
| 重放 | cold → replay → 改 token/position → replay；1→4→1；取消/异常后的下一请求 |
| 异常数值 | 0、极小/极大有限值、饱和量化块、NaN/Inf 按原接口规则拒绝或回退 |
| 分片 | 不等长/空词表 shard、最低 global-id tie、headmap、expert id/权重 |

历史 skinny 输入转换缓存 bug 的回归必须保持；auto/force 重新触发它时优先查数据寿命，不通过放宽误差阈值掩盖错误。[E11]

### 9.3 第三层：固定自然文本轨迹与完整状态

保存一个通过审阅的自然文本 token-id 序列，在完全相同的 prefix、position、context 和 KV 状态下执行不同 shape/策略。对比原始 logits、top1/top2 gap、归一化误差，以及受影响的 target/draft hidden、KV、GDN/卷积历史。

区分两个 oracle：

- **实现 oracle**：同一种量化/算术定义下，用 CPU 或独立实现验证收缩及状态更新；这应能发现错误索引、旧 activation、越界与同步缺陷。
- **数值资格 oracle**：MMVQ 相对原 target 的误差和质量变化；允许的误差界必须预先确定，并经过完整模型轨迹验证，不能因为想晋级才放宽。

auto 可能在一行与多行选择不同算术。因此不能天真要求“auto 单行”与“auto 多行”逐位相等，也不能把所有差异都自动归为正常量化。先确认实际选中的核，再用对应数学参考区分数值差异与实现错误。

若无法建立同算术的单行参考，记录缺口，补独立量化 oracle；不偷偷修改参考 logits 或接受结果。对有明显 top1 margin 的翻转必须追查；近 tie 也要给出误差证据，不只用 near-tie 一词解释。

### 9.4 MTP/采样/服务行为矩阵

对 h=1 覆盖 accepted=0、1；扩到 h=2/3 时覆盖 0…h。强制拒绝测试与真实模型拒绝都需要。

检查 anchor/accepted/correction/bonus 的位置、公开输出与内部计数映射；EOG、长度限制、零输出 catch-up、stream cancel、取消后短/长请求、prompt-cache 复用、保存/恢复（已有支持范围内）。不能自行假定 accepted=a 就消费 a 行。

采样覆盖原 canonical chain、grammar、JSON、有效 penalty、logit bias、reasoning budget、随机采样及 checkpoint/rollback。MTP/G4b/后端采样不支持的组合应在原有边界回退；不能为命中新路径换掉语义。

MMVQ 改变 target 算术，不因经过 speculative verify 就获得“与原模型完全相同”的保证。输出文本改变时先保留全文和原始 token/logits，再判定数值和任务质量；不能自动添加到 `weightloss-variants.json` 后宣布正确。

### 9.5 正确性退出门

任何 stale-input、非法状态、错误 cache 范围、GPU fault、无法解释的非近 tie 大偏差：停止本候选性能测试。结构测试通过但模型失败时，以模型失败为准；测试失败不作为慢样本从统计里删除后继续宣称成功。

---

## 10. Q3/Q4：从四格 pilot 到真实机制

### 10.1 先跑什么

先跑 A0/A1 的一块交错 pilot，再跑 B0/B1 的一块；每进程首请求单列，后续重复数固定。下一组反过来安排 B 配对再 A 配对，减少长期顺序漂移。先过 smoke/正确性，不要第一次命中 auto 就直接跑大规模正式矩阵。

这两次独立 pilot 只筛选，不构成严格四格交互统计。若候选明显回退且路由已确认，可停止该具体配置，不必为了“正式关闭”浪费五块 ABBA。

### 10.2 每次输出必须包括

完整 request/response、completion token 数、自然停止原因、server predicted_ms、prompt_ms、client wall、首内容时间（能可靠测时）、drafted/accepted/cycles、有效 verify rows 直方图、当前配置身份和所有失败。

原 harness 保存的内容不足时先补输出，不再只存 `{ms,tok,stop}`。原始响应应保留，但可能含用户私有提示时做本地权限控制；日志和 manifest 不采集无关密钥。

### 10.3 最小命中清单

先用定义期或低扰动计数，避免每 op 同步、逐行打印影响 replay。现有 region policy 和 `ggml_backend_vk_get_region_mmvq_stats` 可复用，但不能假定它覆盖所有 standalone 或每个 replay 的真实运行次数。[E12]

建议新增的诊断字段如下，**字段建议不代表当前已实现这些 API/日志**：

```text
run_id, request_id, phase, context_role, backend_id, rank, graph_generation
op, weight_name/type, original_ne/nb, lowered_M/N/K, mat_vec_cols, outer_batches
skinny_as_batch, region_family, effective_policy, selected_kernel, fallback_reason
activation_quant_type, quantized_input_owner/generation, conversion_reuse
definition_count, actual_execution_count, full_phase_time, logical_bytes
```

定义捕获一次而 replay 一百次，不能把定义计数当执行计数。没有硬件 counter 时，“逻辑字节”不命名成物理 DRAM 流量。

输出 `coverage.csv`，按 target verify 的真实关键路径贡献排序。不是按操作次数最多、张量最大或名字看起来最贵排序。

### 10.4 CPU/GPU 账的口径

CPU 记录互斥阶段：准备、提交、等待、独占采样、accept/commit、catch-up；GPU 在各自时钟域记录量化、收缩、后续融合、copy 和完整 phase。若跨设备时间未校准，只使用局部时长与因果关系，不相加五卡 GPU 时间当请求延迟。

比较完整 verify 时间是否缩短。量化 kernel 更快但转换 scratch/同步/融合回退更慢，最终仍是负收益。不得把 wait 从 decode 搬到 sampler 后算作省时。

正式性能不开 profiler；诊断必须另有同版本未插桩对照，说明扰动。不以 G1/G8 旧分项或 LB 的不同版本计时相减来解释新结果。

### 10.5 一眼判断表

| 观察 | 当前解释 | 下一步 |
|---|---|---|
| auto 策略行正确，但 verify 无 MMVQ 命中 | 路由/覆盖问题，不是组合性能已测 | 查类型、shape、small-Q8、region guard |
| 只在 prefill 命中 | 可能改善首字，不是多行验证收益 | 首字单列；调查 verify 是否值得支持 |
| target verify 快，draft 慢 | 全局开关混合了两种作用 | Q5 做 role 隔离，不直接宣布组合无用 |
| 单核快，整个 verify 不变 | 覆盖面小或其他成本抵消 | 停止扩大该核；查完整链，不堆 flags |
| verify 快，但每轮提交变少 | 数值/提案分布或采样变化 | 先过质量和接受分布门，再算经济账 |
| B1 快于 B0，但仍慢于 A1 | MMVQ 有用，MTP 未带来产品优势 | 不晋级组合 |
| A1/B1 都相近比例变快 | MMVQ 普遍有效，交互未必强 | 保留产品收益，不宣称“只有组合才有效” |
| B1 明显优于最好 no-MTP，且 I/J 支持交互 | 假设获得支持 | Q7 正式与多提示验证 |
| off 控制自身漂移 | 对照未站稳 | 回 Q0，不做候选排名 |

---

## 11. Q5：什么时候才值得改 selector 或隔离 target/draft

### 11.1 只在明确缺口上修改

允许立项的前提：已有真实 shape、当前成本、auto 回退原因、受支持单核的数值与时间对照，且完整阶段中有足够覆盖面。

优先候选是目标验证中的 Q5_K、逻辑多行但被按单列处理的 Q8 瘦矩阵，以及经独立证据支持的 Q6_K 形状。不是给所有 AMD、所有 K、所有模型粗暴改默认。

既有类型不支持（例如某些 IQ 专家格式）时，force 不会变出新 pipeline。实现新类型属于新的 kernel 项目，必须另列投入、正确性和收益预算，不能藏在“四格开关测试”里。

### 11.2 per-role 策略最小合同

新增策略应属于 context/backend 的不可变配置，例如概念上的 `role={target,draft}` 与 `policy={off,auto,scoped}`。这里只是设计名称，不是已有 CLI/API。

同一物理 device 可被多个 backend/context 使用；pipeline 缓存可共享，但选择策略、activation 转换 scratch 的身份和生命周期不能串用。策略进入程序/cache key，禁止 replay 继续使用旧策略编译的执行定义。

四个内部消融格子是 target off/draft off、target auto/draft off、target off/draft auto、target auto/draft auto。先在固定 h、同请求和相同 sampler 上比较，再决定是否需要 phase-aware 策略。

### 11.3 phase-aware 是后续任务，不是伪造 shape

若需要只对 TARGET_VERIFY 启用，调用方必须明确标识 phase；不能只用 `n_tokens>1` 猜测，因为 prefill 和 catch-up 也可能多行。不借此次机会放开 target-capacity 的 GDN/inactive-row 守卫。

small-Q8 候选必须保留真实 strides、输出顺序、量化缓冲身份和 tail 规则。把 `mat_vec_cols` 数字改大但不改变实际 layout/dispatch，是错误实现，不是策略优化。

### 11.4 force 的用途和止损

force 只用于“该硬件/类型/shape 有没有更好的受支持收缩”的定向诊断。若 Q6_K auto 排除有性能理由，force 反而可能慢；这是待测，不是 selector 注释已经证明的结论。

不在正式四格里让 B 使用全局 force、A 使用 auto，然后把全部差额归因 MTP。force 带来新数值/性能结果时单列候选，先回到同条件对照。

---

## 12. Q6：何时扩大到 h=2/3，何时停止

G7 历史自然问题的 mean_len：h1≈1.77、h3≈2.45、h6≈2.68、h10≈2.63。该趋势只适用于那些实验轨迹，不当作所有提示的固定分布。[E3]

MMVQ 便宜化的是验证计算，不是让草稿更准确。优先逐级：h1 两行 → 有明确形状/边际证据才测 h2 三行 → 再决定 h3 四行。固定 h 下先完成 off/auto 对照，再调 h；不把两个变化混成一条收益。

对 h 的成本写成：

```text
C_h = draft_h + verify_h + catchup_h + accept/接口_h + 必要恢复_h
L_h ≈ E[C_h] / E[公开提交 token_h]
```

h 增大值得的条件是新的成本/公开 token 更低，不是 acceptance 百分比或 GPU 吞吐单独更漂亮。公开 token 要按服务器实际发出的 completion 算，不能直接用内部 mean_len 替代尾轮、EOG 和预算截断。

每个 h 同时记录：候选位置存活概率、真实拒绝位置、draft 成本、verify rows 分布、accepted 数、首轮成本、恢复/取消、VRAM/GTT 峰值。MMVQ 引起新的输出轨迹时重取分布，不套旧 G7。

停止加长的情况：mean_len 不再增长、新增行很少被接受、显存驻留恶化、target 被换成更慢路径、量化误差/状态质量不合格。没有这些证据之前也不扫 n6/n8/n10/n12；本计划不是旧 horizon 扫描的换名重做。

---

## 13. Q7：性能统计和“大收益”的判定

### 13.1 指标分层

主产品指标：同一预定自然提示集上的生成延迟与每公开 token 成本。同步报告完整请求 wall/首内容时间、输出长度、停止情况及质量。不能通过提前 EOS、漏掉要点或更短回答换取“提速”。

相同 token 轨迹的单步/多行成本只用于归因。自由生成长度变化时，既报告总 predicted_ms，也报告逐请求 `predicted_ms/completion_tokens`；不要用平均 tok/s 的倒数代替平均成本。

冷启动、进程首请求、warm 请求分开。允许把预先定义的首请求单列，但必须保留它及任何失败；不事后删掉慢的首个“正式”样本。

### 13.2 用现有 ABBA 先证明产品胜出

先通过 pilot 和质量门，选定 B1 与合格最好 no-MTP 配置，再用当前 harness 跑至少五个完整 ABBA block；每块四个独立服务进程，每进程预先固定重复数。

block 是独立配对单元，进程内重复只降低测量噪声。报告全部块、mean/median、离散度、配对差和比值区间；不把 40 次请求说成 40 个独立 block。

若仅能通过非劣，应因明确维护/正确性价值才考虑保留；本专题以性能为目的，没有收益的附加选择器通常不值得晋级。

### 13.3 真正的四格交互统计需要补 runner

当前二臂 harness 不能直接证明四格交互。以后需要一个有进程所有权和 bounded teardown 的编排层，复用现有 trial 逻辑；不要发明当前不存在的 `--four-cells` 参数。

建议每个宏块为四格的镜像序列，例如 `A0 A1 B0 B1 B1 B0 A1 A0`；下一块轮换/按预先冻结的种子随机化前半排列，再镜像后半。所有进程串行，全部身份/失败规则相同。该序列是设计建议，不是本次已运行的实验。

至少五个完整宏块，每块内先对同格取预定聚合，再计算 I/J/G；置信区间以宏块为单位。两个分离日期分别做的 A0/A1 与 B0/B1，即使各自 ABBA 合格，也不自然消除跨日期漂移。

统计方法在见结果前固定，例如独立块差值的 t 区间并报告原始块值。小样本区间的适用假设写清；不能用大量相关请求 bootstrap 制造窄区间。

### 13.4 建议的预注册判据

- 晋级至少要求质量/状态通过、真实命中、整体生成成本优于合格最好 no-MTP 对照，且预注册配对统计支持收益。
- 建议先把“有工程意义”设为至少 5% 的生成成本下降，把“大收益”设为至少 10%；这些是可在实验前修改的计划门槛，不是用户已承诺的新验收标准。
- 单独 A0/A1 未显著改善，不足以证明“MMVQ 单独无用”。要用该表述，须预注册等效边界并实际检验；否则写“未检出”。
- 要称组合协同，I 或 J 的预定主要检验须支持正交互；仅 B1 最快，可能是 MMVQ 的一般收益，不一定有额外交互。
- 重复测量因机器故障等排除时，按事先规则废弃整个受污染块，保存原因和原始数据，不只删较慢一臂。
- 不因第一次没显著就无限追加样本。最大块数与追加条件在开始前固定。

### 13.5 多提示与泛化

先固定现有自然问题完成机制闭环，再冻结一个不依赖候选输出挑选的小集合：中文说明、代码、算术/推理、翻译或摘要等不同接受分布。提示、seed、预算、质量规则在评测前锁定。

不同提示分别报告收益、接受分布和回退；不能用一个高接受率提示的加速掩盖另一类明显回退。grammar/JSON 等不支持优化的请求，正确回退也是必交付结果。

---

## 14. 经济账：先估能省多少，不把整段 verify 当可回收成本

G8 旧诊断给出 h1 稳态周期约 29.76 ms，其中 verify 26.02、draft 2.94、catch-up 0.80。它提示 verify 值得研究，但不是新四格的预算，也不采纳其有争议的固定 45% 锁步拆分。[E13]

仅作条件计算：若整个 verify 真缩短 20%，其他项和公开提交长度不变，则周期约少 17.5%。**没有证据说 MMVQ 能让整个 verify 缩短 20%。**

更有用的模型是：

```text
ΔC = 可命中收缩的关键路径减少
     - 新 activation 量化/转换成本
     - 新同步/融合回退/驻留成本
     - draft 与 catch-up 的净恶化

ΔL 还取决于新的实际公开提交长度和首轮成本。
```

若可加速部分只占完整请求 10%，即使该部分无限快，理论上最多减少该假设下的 10% 请求时间。不要把不支持的 IQ4_NL 专家、CPU RELAY 等待和未命中的 Q8 全部填进“MMVQ 可省项”。

量化 activation 随行数增长，不是 h 行只量化一行。真正可摊销的是一些固定派发、共享输入的重复转换和有证据的收缩/访存成本。MoE 权重复用须看实际专家交集，不能默认所有 token 选同一专家。

---

## 15. 后续操作模板：按顺序执行，前置门不满足就停止

**以下全部是未来实施命令。本次没有执行。** 变量指向操作者有权限访问的真实目录；路径不存在、权限不足或来源不明时停下，不擅自改成其他实验树。所有输出目录必须新建。

### 15.1 建立独立来源与证据目录

下列 SOURCE_REF 必须由 Q0 选定；若所需 G4b/自然文本代码还未提交，先制作完整可审阅快照，不能假装某个历史 commit 已包含它。此处只示范隔离方式，不自动合入任何 dirty 文件。

```bash
set -euo pipefail
REPO="${REPO:-$HOME/atomic-llama-cpp-turboquant}"
: "${SOURCE_REF:?填入本次已审阅的完整源码引用，不默认取 master}"
: "${EXPERIMENT_WT:?填入允许访问且尚不存在的隔离 worktree 路径}"
: "${RUN_ROOT:?填入允许访问且尚不存在的新证据目录}"
test ! -e "$EXPERIMENT_WT"
test ! -e "$RUN_ROOT"
git -C "$REPO" status --short
git -C "$REPO" show --no-patch --format=fuller "$SOURCE_REF"
git -C "$REPO" worktree add --detach "$EXPERIMENT_WT" "$SOURCE_REF"
mkdir -p "$RUN_ROOT"
git -C "$EXPERIMENT_WT" rev-parse HEAD > "$RUN_ROOT/source-head.txt"
git -C "$EXPERIMENT_WT" status --short > "$RUN_ROOT/source-status.txt"
git -C "$EXPERIMENT_WT" diff --binary HEAD > "$RUN_ROOT/source-dirty.patch"
```

source-dirty.patch 不包含未跟踪源码。另保存 `git ls-files --others --exclude-standard` 清单及本实验实际依赖的文件；避免无差别打包私密文件或巨量旧 artifacts。

### 15.2 构建与小测试

先恢复 Q0 冻结的编译器/CMake/shader 配置；下面的 RelWithDebInfo 只是示例，不声称天然等同某次历史构建。四格共用同一份构建，构建期间不跑性能。

```bash
set -euo pipefail
: "${EXPERIMENT_WT:?}"
: "${RUN_ROOT:?}"
: "${BUILD_JOBS:=4}"
BUILD_DIR="$EXPERIMENT_WT/build-qmp"
test ! -e "$BUILD_DIR"
cmake -S "$EXPERIMENT_WT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON \
  2>&1 | tee "$RUN_ROOT/configure.log"
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" \
  2>&1 | tee "$RUN_ROOT/build.log"
ctest --test-dir "$BUILD_DIR" -N > "$RUN_ROOT/tests-registered.txt"
for NAME in test-sampling test-arg-parser test-mtp-workspace test-tp5-plan; do
  ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error \
    -R "^${NAME}$" 2>&1 | tee "$RUN_ROOT/${NAME}.log"
done
BIN="$BUILD_DIR/bin/llama-server"
test -x "$BIN"
sha256sum "$BIN" > "$RUN_ROOT/server.sha256"
```

`ctest -N` 不是通过。分别跑四个表达式，避免只注册其中一个却把整组称为 4/4。共享 Vulkan 改动还需 Q2 相关测试；历史 4/4 CPU 通过不证明新增 MMVQ 数值/状态正确。

GPU 定向测试一项一项运行，不把多个 `--*-only` 选项放进同一调用；当前 main 只接受一个选项。以下只是最小入口，先确保五卡没有其他负载：

```bash
set -euo pipefail
: "${BUILD_DIR:?}"
: "${RUN_ROOT:?}"
TEST="$BUILD_DIR/bin/test-vulkan-command-replay"
test -x "$TEST"
for CASE in skinny attention-mmvq argmax transfer; do
  "$TEST" "--${CASE}-only" 2>&1 | tee "$RUN_ROOT/replay-${CASE}.log"
done
```

非零退出即停止，包括 SKIP=77；操作者检查原因并标记 SKIP，不将其改为成功。skinny/attention-mmvq 的内部 force 规则见 §9.1。

### 15.3 注册 variants、mock 和 dry-run

在隔离实验树按 §8 添加注册函数及调用，补齐 CPU-only 配置断言。必须把新 harness 的 diff/hash 一起入档。此时仍不运行模型。

```bash
set -euo pipefail
: "${EXPERIMENT_WT:?}"
: "${BIN:?}"
: "${RUN_ROOT:?}"
: "${TARGET_MODEL:?填同一已验收 target GGUF 的首分片}"
: "${MTP_MODEL:?填同一已验收 draft GGUF}"
HARNESS="$EXPERIMENT_WT/scripts/run-tp5-cross-matrix.py"
python3 "$HARNESS" --list-variants > "$RUN_ROOT/variants.txt"
grep -F 'qmp-a0-nomtp-off' "$RUN_ROOT/variants.txt"
grep -F 'qmp-b1-h1-auto' "$RUN_ROOT/variants.txt"
test ! -e "$RUN_ROOT/dry-h1.json"
python3 "$HARNESS" \
  --variant-a qmp-b0-h1-off --variant-b qmp-b1-h1-auto \
  --bin "$BIN" --model "$TARGET_MODEL" --mtp-model "$MTP_MODEL" \
  --workload weightloss --dry-run --blocks 1 --repeats 2 \
  --output "$RUN_ROOT/dry-h1.json"
```

dry-run 只证明参数编排，不证明模型存在完整分片、设备空闲、数值正确或实际命中。检查输出明确为 NOT_MEASURED。不要通过外层 `export GGML_VK_DISABLE_MMVQ=0` 期待绕过 harness 白名单。

### 15.4 pilot 命令

仅在 Q0、Q1、Q2 最小门完成且收到 GPU 执行授权后使用。同一个 shell 会话保留上述变量，先验证实际绑定的 DSO。每次新实验根目录，不覆盖旧 JSON。

```bash
set -euo pipefail
: "${HARNESS:?}"
: "${BIN:?}"
: "${TARGET_MODEL:?}"
: "${MTP_MODEL:?}"
: "${RUN_ROOT:?}"
run_pair() {
  local tag="$1" a="$2" b="$3" blocks="$4"
  shift 4
  test ! -e "$RUN_ROOT/$tag.json"
  python3 "$HARNESS" \
    --variant-a "$a" --variant-b "$b" \
    --bin "$BIN" --model "$TARGET_MODEL" --mtp-model "$MTP_MODEL" \
    --workload weightloss --blocks "$blocks" --repeats 2 \
    --host 127.0.0.1 --port 8097 --max-wall-seconds 2400 \
    --output "$RUN_ROOT/$tag.json" "$@"
}
run_pair pilot-nomtp qmp-a0-nomtp-off qmp-a1-nomtp-auto 1 --pilot
run_pair pilot-h1 qmp-b0-h1-off qmp-b1-h1-auto 1 --pilot
```

两对跑完即审阅，不自动继续更长 horizon。port 如已占用，不连接他人 server 冒充自己的进程；复用 harness 的 owned-process/preflight 规则。

正式性能前确保所有 profiler/trace/临时 debug 不存在，原始日志级别相同。不开 timing=60 等采样再把结果叫无插桩。

### 15.5 定型后的正式产品对照

下面的变量由 pilot、质量与独立 control 核验共同确定，不能直接选最快单样本。若 A1 更好，必须以 A1 为对照；B1-h1 不赢就不要硬填它。

```bash
set -euo pipefail
: "${BEST_NO_MTP_VARIANT:?填已冻结的合格 no-MTP 对照配置}"
: "${QUALIFIED_COMBO_VARIANT:?填已过正确性和 pilot 的组合配置}"
run_pair formal-product "$BEST_NO_MTP_VARIANT" "$QUALIFIED_COMBO_VARIANT" 5
```

这只完成二臂产品资格。要宣称四格交互，另实现 §13.3 的宏块编排与分析；没有实现就明确写“未完成交互统计”。

---

## 16. 证据文件格式与归档要求

建议目录：`artifacts/mmvq-mtp/<日期>/<工单>/<run-id>/`。所有正式资格证据应能持久访问；临时 build 可以在 scratch，但不能只留下一个消失的二进制 SHA 前缀。

```text
manifest.json                 # source、binary/DSO、模型、设备、driver、resolved 配置
preregistration.md            # 假设、格子、指标、样本、排除/停止/追加规则
candidate.patch               # 单因素代码或 harness diff
build/                        # 编译/生成 shader 配置及日志
tests/                        # 实现 oracle、数值、状态、fallback、服务回归
requests.jsonl                # 完整请求和 seed/采样链
responses.jsonl               # 完整响应、token、停止原因、时间
trials.json                   # 全部独立进程和进程内请求，不删首请求
coverage.csv                  # role/phase/shape/type/kernel 命中及回退
diagnostics/                  # 有插桩时间线、驻留、转换次数，和正式性能分开
formal/                       # ABBA/四格宏块原始样本与可重算分析
exclusions.jsonl              # 故障、未命中、污染块、排除原因
decision.md                   # 结论、边界、保留/撤回/下一门
```

manifest 至少有：`run_id`、时间与时区、P/M 系列、格子名称、h、target/draft policy、原始/生效配置、源码 commit+dirty 身份、binary/实际 DSO、模型分片、GPU UUID/BDF、build options、workload/quality-set 身份、是否 profiler、是否发生 reset。

不能把路径字符串当作内容身份；不能让 harness 的 Git provenance 只记录脚本所在树，却不记录候选 binary 的来源。`--bin-b` 使用两份 binary 时两边全部记录。

每份 decision 回答：假设是什么；实际命中哪里；相比旧失败机制有何不同；节省哪一项；量化/驻留/采样/接受长度付了什么代价；正确性通过到哪一层；全部结果如何；为何保留或停止。

---

## 17. 常见失败的处理，不允许“修到看起来通过”

| 失败现象 | 首查 | 禁止做法 |
|---|---|---|
| force 请求但仍 disabled | 默认 DISABLE 优先级、harness env | 不核对实际策略就测吞吐 |
| auto 已打印但没有目标核 | type/shape/contiguity/skinny/region | 关闭日志守卫宣布命中 |
| GPU 测试返回 77 | feature、device、测试覆盖 | 将 skip 改为 pass |
| 新词/长度变化 | 原始 logits、token 轨迹、状态、量化误差 | 自动加到允许变体集 |
| 第二投影或 replay 结果异常 | activation 转换缓存身份和 scratch 生命周期 | 容差放大、跳过对应行 |
| B 快而质量明显差 | EOS、输出完整性、target 精度、采样器 | 以 ms/token 一项晋级 |
| 单核显著快但端到端负 | 覆盖面、额外量化、融合丢失、等待搬家 | 堆另一个开关掩盖第一项失败 |
| draft 接受率改变 | draft/target 两边策略、sampler、候选阈值 | 把提升都归因 GPU 算力 |
| GTT/VRAM 突变 | 每 role scratch、同时存活分配、权重/状态布局 | 猜某个权重迁移后据此大改 |
| baseline 跨轮漂移 | binary/DSO/env/负载/图路径 | 用历史最快控制值拼新表 |
| GPU/服务崩溃 | 保留事故、停止候选、安全排空 | 自动 reset 后接着挑有效快样本 |

看到假阴性风险也不授权无限重开。每次重启必须指出新的可验证机制或缺失覆盖，并有成本预算。没有新机制就保存负结果，停止当前候选。

---

## 18. 最终可能得到的结论，提前写明白

| 结果 | 应写结论 | 后续 |
|---|---|---|
| A1≈A0，B1明显更好且胜过最好 no-MTP | 在已覆盖任务与配置下，支持 MMVQ×MTP 的正交互 | 正式质量/多提示门后考虑受限默认 |
| A1、B1都改善，交互不强 | MMVQ 普遍有价值，组合不一定有额外协同 | 发布真实产品收益，不强贴组合故事 |
| B1快于B0但不如A1 | MTP内核有改善，组合仍不值得默认使用 | 保留必要机制或撤回维护负担 |
| global B1无收益，target-only有收益 | 全局开关掩盖了 target/draft 差异 | 独立策略完成兼容/状态门后重验 |
| auto无命中，局部合法策略有收益 | 问题在覆盖/选择，不是两个开关本身 | 只晋级已测 shape，其他回退 |
| 只在计数或高接受率提示有收益 | 适用范围有限 | 不推广到全部自然文本 |
| 仅质量退化后更快 | 是精度/质量权衡，不是无损性能胜利 | 另需用户明确批准，不混入原默认 |
| 所有合格候选都不赢 | 本次形状/模型/驱动范围内未发现有用组合 | 关闭具体配置，不写所有未来体系架构不可能 |

这次任务的成功不必是“证明猜想正确”。能用可信对照说明组合何时值得、何时不值得，同样是完整交付。

---

## 19. 最后检查单

### 进入 GPU pilot 前

- [ ] 已收到后续实施授权；本次规划本身不算授权。
- [ ] 源码、harness、模型、binary/DSO/shader 身份冻结，未跟踪依赖也已保存。
- [ ] A/A 对照无未解释的大漂移；最好 no-MTP 未被削弱。
- [ ] 四格配置和 env 白名单正确；F32/RELAY、LateBind off；force 不混入 auto。
- [ ] G4b、draft sampling、attention/headmap 差异已列明。
- [ ] 小测试没有把 0 tests/skip 算通过；新 rows/type 有覆盖。
- [ ] 自然文本正确性、状态、失败回退的最小门通过。
- [ ] 五卡独占、watchdog active，无并行构建/哈希/推理。

### 晋级前

- [ ] actual target verify MMVQ 命中有证据，而不只是启动 policy。
- [ ] 完整请求收益不是来自漏输出、提前 EOS 或换 sampler。
- [ ] B1 胜过已冻结的最好 no-MTP 配置，而不只胜 B0。
- [ ] 正式无插桩配对有完整 block、全样本、误差/排除规则。
- [ ] 若声称协同，已有四格宏块交互统计；否则仅称产品收益。
- [ ] h2/h3、target-only、selector 等改动各自单因素资格完整。
- [ ] 数值、KV/GDN、接受/拒绝、取消、重放、grammar/JSON/penalty 回退覆盖完整。
- [ ] 新策略只影响已验证范围；off 和其他模型行为受保护。
状态：**EXECUTED / CLOSED（2026-09-28 晚，run `/var/tmp/tp5-mmvq/`，判决见文末 §21）**。
四格 + h 扫描已按本手册规则执行完毕：MMVQ 命中已证实（融合 attention 投影 gdn mask=3 / qsa mask=1），
但无司检定收益；最优配置仍为 aggressive h=1 + MMVQ disabled（TP5 recipe 现默认）。
- [ ] 整合后的实际提交/补丁重新验证；未获授权不 commit/push。

---

## 20. 本地证据与源码索引

以下均相对仓库根目录。文中的历史数值按原记录注明 pilot/正式/诊断；本文件没有新增性能数字。

- **E1：近期关闭 MMVQ 的实验面。** `artifacts/tp5-next-20260926/` 下带 `variant_a/variant_b` 的结果 JSON；写作前分析计得 18 份、36 臂。以后目录增大应重新统计，不能把这个计数当永久事实。
- **E2：n1 与 no-MTP。** `artifacts/tp5-next-20260926/g8-natural-ledger/run-20260927T0400/pilot-n1-vs-nomtp.json`，以及其 `pilot-n1-vs-nomtp_logs/server_native-default_blk1_B1.log`、`server_native-mtp-fused-n1-varargmax_blk1_A1.log`。
- **E3：自然文本 horizon 与正式 n6/n1。** `artifacts/tp5-next-20260926/g7-realistic-workload/run-20260927T0330/decision.md`、`abba5-n6-vs-n1.json`、同目录的各 pilot；题目/变体集在上一层。
- **E4：历史 MTP+MMVQ-auto。** `TP5.md` 的“两项未晋级的单因素探针”；引用旧 `/var/tmp/tp5-mtp-mmvq-auto-smoke.json`。原件未访问时仅作文档记录，不冒充本次原始重算。
- **E5：解包上移的假阴性边界。** `TP5.md` 的“Q5_K MMVQ 多列解包上移也已退选”；记录 ISA 相同以及客户端 DISABLE_MMVQ=1。
- **E6：选择与支持。** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`：`ggml_vk_should_use_mmvq`、`ggml_vk_get_dequantize_mul_mat_vec`、`ggml_vk_mul_mat_vec_q_f16`、`ggml_vk_mul_mat_id_q_f16`、small_q8_columns 分支。
- **E7：默认与隐藏因素。** `common/common.cpp::common_tp5_apply_env`；E2 的 48/96-stage 日志。写作前定位约 496–518 行。
- **E8：harness。** `scripts/run-tp5-cross-matrix.py`：`get_available_variants`、`ENV_WHITELIST_KEYS`、effective route 校验、owned-process trial、ABBA 和参数解析。
- **E9：融合正交规则。** `TP5.md` 的“正交优化规则：MMVQ、Qwen region fusion、producer transport、LateBind”；`ggml_vk_can_fuse_attention_projections`、`ggml_vk_can_fuse_attention_segment` 等实际 guard。
- **E10：测试强制规则。** `tests/test-vulkan-command-replay.cpp::main`、`test_attention_projections_replay`、`test_skinny_distinct_inputs`；写作前 main 约 4068–4203 行。
- **E11：历史输入缓存修复。** `TP5.md` 的“RERoT 报告中的共享 Vulkan 缺陷”，稳定原始 tensor identity 的修复；`ggml_vk_mul_mat_vec_q_f16` 的 `src1_identity`。
- **E12：已有 region 观测。** `ggml-vulkan.cpp` 的 `[vulkan-region-policy]`、`region_q8_used_masks`、`ggml_backend_vk_get_region_mmvq_stats`；测试里有使用例。
- **E13：旧分相参考。** `artifacts/tp5-next-20260926/g8-natural-ledger/run-20260927T0400/decision.md`、`ledger.json`、`server-ledger.log`。只用其明确 phase 测量，不沿用已指出口径混乱的固定延迟外推。
- **E14：最新交接与漂移教训。** 根目录 `交接.md`；`artifacts/tp5-next-20260926/lb2-rho/run-20260927T0730/`。本文件不继续 LB 试验，仅保留身份、漂移、错误归因和数据保留教训。

**首个后续交付：冻结当前 no-MTP/n1 对照，补上真正生效的 F32 四格 variants，给出两行 verify 的实际 MMVQ 命中表。先证明“测到了这个组合”，再判断它是否有用。**

---

## 21. 【终局判决 2026-09-28 晚】四格 + h 扫描执行完毕：假设否决

执行记录：`/var/tmp/tp5-mmvq/`（p0/p1 = h1 四格镜像，h2/h3 = horizon 扫描，
dbg1 = 命中证明，decision.md = 完整账）。同二进制（含 KEEP_CHUNKS 修复）、
GPU 锁、`env -i` 纪律、每臂 warm-up + 3 请求、账本断言。

### 21.1 命中证明（红线 1：先证"测到了"）

`GGML_TP5_RELAY_DEBUG=1`（dbg1.log，五 rank 一致）：

```
family=gdn-projections fused=1 mmvq_mask=3 shared_q8_1=1
family=qsa-projections  fused=1 mmvq_mask=1 shared_q8_1=1
```

融合 attention 投影（Q5_K/Q6_K）的 MMVQ **真实命中**；auto 臂账本
`dispatches=3024 vs 2879`（+145 = activation Q8_1 量化 kernel）。
注意 `[vulkan-region-policy]` 打印被 RELAY_DEBUG 门控——
**日志里 region 行数为 0 不代表未命中**（p0/p1 的 region=0 即此坑）。

### 21.2 四格 + h 扫描（ms/cyc 中位，配对镜像比值）

| 格 | h=1 | h=2 | h=3 |
|---|---|---|---|
| A0（no-MTP, off） | 29.5/29.1 | — | — |
| A1（no-MTP, auto） | **1.067×慢** | — | — |
| B0（aggr, off） | 28.2/28.5（62.5 tok/s） | 38.6（57.6） | 49.1（53.6） |
| B1（aggr, auto） | 0.990×（不可分） | **1.136×慢** | 输出不稳定，不可判 |

- **I = S_B − S_A ≈ +0.29 ms/cyc 名义正交互**，但 B1 不快于 B0：交互只是
  "MMVQ 在 MTP 下不那么亏"，非收益。
- h 增大单调变差（62.5 → 57.6 → 53.6）：短 horizon 判决再次成立。
- h=3 格 B0/B1 输出 md5 自身漂移（接受率 0.43-0.55 摆动）——该格不可判，
  按红线 7 留档不重抽。
- 红线 3 确认：A1/B1 输出 md5 均变（activation 量化改变 logits），文本仍连贯。

### 21.3 判决（对应 §18 第 8 行）

**在本机（RX 6800/RADV）、本模型（Qwen4EXP，投影 Q5_K/Q6_K）、当前
LateBind 刀锋（aggressive h=1）上，MMVQ 无可检定收益。**
TP5 recipe 默认 `GGML_VK_DISABLE_MMVQ=1` 是正确默认，保持不变。

机制归因：+145 dispatch 的量化成本 + Q8_1×Q5_K 整数 dot 在 RDNA2 上不快于
F32 GEMV；多行（h≥2）下量化成本随行数增长，与 §14 经济账一致。

### 21.4 边界（不外推）

未测且不因此判死的：Q8_0 主导权重的模型、Intel/NVIDIA、h≥3 稳定重测、
非 LateBind 布局、per-role 隔离（Q5 的独立策略）。IQ 专家类型本就不在
MMVQ 支持表内。h=1 的 0.990 在 ±3-5% 会话漂移带内，正确表述是"未检出"。
