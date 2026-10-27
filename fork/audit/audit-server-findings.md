# 旧 fork 服务端与 common 小行为静态审计记录

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

行为审计输入为 `audit-server.diff`、`audit-common.diff` 与 `audit-old-numstat.txt`；最终逐文件总账以完整的 `audit-inventory.json` 为准（4634个差异文件，此代理范围45个，其余4589个明确标为范围外仅总账）。位置均是 diff 的 **b 侧，即旧 fork 合并后源代码行号**，不是本次干净分支行号。只读静态审计；下文提到旧 fork 测试表示已读测试用例或确认文件存在，并不表示本轮执行过。大文件的未读区间不推定正确，也不把代码注释中的数字当作本轮测量。覆盖状态另见 `audit-server-coverage.json`。

## 必须避免的错误取舍

1. **Prefill 优先调度、Prefill 图的单序列预留、Prefill/Decode 计算工作区释放、统一物理 KV 容量，是四个独立行为。** 保留其中一个不能声称另外三个已经恢复。原 Prefill selector 不依赖 FlashPrefill，删除稀疏注意力实验时不能一并删掉它。
2. **普通请求槽 LRU 选择已经存在于 a 侧。** 自创的是压力下 idle KV 的 LRU 回收、请求 cache key 亲和、动态物理池的保留与抢占路径等增量，不能把整个旧函数移动当作自创。
3. **重试能完成请求不等于随机采样完全等价。** 当前恢复了文本、计数、时间与 prompt，却重新初始化 common sampler；已有并发测试主要检查预算和 SSE 数量，不证明随机状态一致。
4. **声明放进 `include/llama.h` 不代表上游已有此 API。** KV/recurrent usage、XKV admission、RERoT view、`*_no_sync` 等均是本 diff 新增；C 不能调用它们后声称完全使用原版 A。
5. **独立 B 的 parser 不自动保留 RERoT shared-KV 推理。** 解析、治理、数值判断可以移植，运行语义仍需要 C 实际消费和 A 可表达的状态操作。

## 建议本轮优先保留的独立行为

| 优先项 | 独立 B | C 的实际消费 | A 的最小边界 |
|---|---|---|---|
| 原始 sticky Prefill 优先 | 输入按原 slot 顺序的值快照，返回 Prefill slot id 或 Decode；B 自己只持当前 Prefill id | C 持 token/position/seq/sampler，实际构造并执行被选批次；注入新 prompt 验证 Decode 被暂停、结束后恢复 | `llama_decode`、公共 sampler、`llama_synchronize`。不需要 B 进入 A |
| Prefill 后工作区释放 | 不在 selector 内伪造显存功能 | C 在真实 Prefill→Decode 边界调用释放 seam 并检查返回值；关闭开关做实测对照 | 需要显式、有限的 A 兼容 seam，由 A 自己同步与释放内部计算 buffer |
| 物理 KV 独立容量 | 可独立做容量策略，但必须以 A 的真实结果为准 | C 配置逻辑上下文与物理 KV，并报告实际 getter 值 | `n_ctx_kv` 与 getter 是有限 A 兼容补丁；不能仅把 `n_ctx` 改小然后宣称保留逻辑窗口 |
| cache key 亲和 | 字符串、slot id、LCP 长度的值策略 | C 在请求分配、释放、清 cache 时维护映射，实际复用 A seq state | 已有公共 sequence/state API 足够做保守重放；物理使用量优化需额外观测 seam |
| 多 completion 的 usage 聚合 | 纯计数聚合 | C 输出真实所有 choices，仅按 prompt 计一次输入 token | HTTP server 编排层行为，无须修改模型 A |
| UTF-8/stop 扫描 fastpath | 纯文本累计策略 | 无 stop、非流式请求跳过逐 token 扫描，最后仍验证输出边界 | 无 A 内部依赖；应独立测量，不能继承 top-k 的收益数字 |
| 安全信号退出 | POSIX 事件转发可单独实现 | C 正常线程接收事件并依次停止队列、HTTP、模型；第二信号立即退出 | 不应向 A 注入回调；生命周期由 C 管理 |
| grammar 替换及 sampler clone 修复 | B 不持 A sampler | C 自己拥有 A sampler，重建 grammar 并切换；clone 后修复候选数据所有权 | 使用公共 sampler 原语；避免 common 私有结构跨边界 |

### Prefill selector 的精确旧语义

`server-context.cpp:9038–9058`：如果上次选中的 slot 仍是 `STARTED` 或 `PROCESSING_PROMPT`，继续该 slot；否则从 `slots` 首项开始找 Prefill；找不到才 Decode。它不按到达时间排 FIFO，不实行 round-robin，也没有 Decode 的等待上限。因此新请求持续到达时 Decode 可能饥饿。要保留原行为，应明确写入 SPEC，不静默改成另一种公平策略。

最少真实 C 接入点：读取自有请求阶段 → B 选择 → 阶段改变时同步 A → 只为所选 prompt 构造 Prefill batch，或为所有可 Decode 请求构造一个 Decode batch → A 成功后更新各自 position/输出与阶段。仅启动上游 server 并传 flags 不能验证 B 被消费。stock server 可以独立保留约 65–90 行的小调度兼容补丁（枚举/字段、selector、相位更新、四个主循环 gate）；这是当前原编排程序的显式耦合例外，不应称作 B/C 分离完成。`n_parallel_pp=1`、释放 compute 和物理 KV 应分开提交。

## 逐行为清单

结论：**保留**表示有具体用户行为且可独立重建；**拆分**表示基础行为与大功能专属部分分别处理；**暂存**表示本轮未迁移或验证不足，不等于无价值；**不原样保留**表示看到具体风险或仅试验辅助。

### S — 调度、批次、初始化

| ID | 旧源文件、函数、行号 | 行为与证据 | 大功能专属？ | 边界、取舍、风险 |
|---|---|---|---|---|
| S01 | `server-context.cpp:320–323,1663–1664,9038–9058` `select_inference_mode` | 黏住未完成的 Prefill；完成后按 slot 顺序选下一个；无 Prefill 才 Decode | 否 | 保留为独立 B。不能宣称 FIFO/公平，必须测试高序号 sticky 与低序号新请求 |
| S02 | 同文件 `update_inference_mode:9060–9074` | phase 变化时同步 target 和独立 draft context | 否 | 保留在 C，B 不接受 context 或 callback；此函数没有释放 compute，不要把同步当释放 |
| S03 | 同文件 `update_slots:9114,9268–9270,9389–9391,9494–9496,9581–9583` | Prefill 时跳过 decode context shift、generation、spec draft，仅填所选 prompt | 否 | 保留真实 batch 行为。只迁移 selector 而不接这些消费者无效 |
| S04 | `server.cpp:211–219`; `server-context.cpp:5524–5527` | `n_parallel_pp=1`；即使用户固定 `-np` 也只预留一个 Prefill 图序列 | 否 | 保留有限 A/C 配置 seam。注释引用 a0c798b51，五卡历史 reserve 1591/1044/1045 MiB 对比 14 MiB、并发 Prefill device loss；属于历史注释，尚未本轮复现 |
| S05 | `server-context.cpp:112–134` `server_n_outputs_max` | 普通生成按 slots×(1+draftmax) 且限 n_batch；embedding 至少 n_batch；RERoT 按 pen capacity | 部分 | 拆分。普通输出 buffer 容量独立有用；pens arena 不应传播进所有运行模式 |
| S06 | `server.cpp:203–209`; `server-context.cpp:5924–5929` | 动态 KV 时 auto parallel 使用最大逻辑 seq 占位，初始化后按实际 recurrent capacity 收紧 slots | 否，依赖动态池 | 暂存完整 auto-fit 路径；保留明确数字容量路径更小。未读 fit 数学，不给自动容量精确性背书 |
| S07 | `server-context.cpp:5564–5567` | 外置 draft model 与动态 `n_ctx_kv` 明确不兼容，启动时拒绝 | 否 | 必须保留限制或在新实现中证明兼容；不能静默忽略 draft 的容量 |
| S08 | `server-context.cpp:5607,5618–5625` | 自动 fit 加 mmproj reserve，检查可加范围 | 多模态+auto-fit | 保留在未来 fit C；不能与纯文本默认绑死 |
| S09 | `server-context.cpp:5652,5931–5941,6061–6068` | MTP recurrent bootstrap、RERoT people/pens 与 batch arena capacity 分开 | 是/部分 | 拆分业务请求数、执行槽数、物理 recurrence；独立 B 不导出 llama seq 类型 |
| S10 | `server-context.cpp:5765–5770,5970–5974` | 模型加载异常返回 failure；用户要求 spec 却不可用时拒绝启动、清理 target | 否 | 保留可见失败语义；避免悄然降级改变用户配置 |
| S11 | `server-context.cpp:9570` | prompt batch 上限结合实际 `kv_batch_limit` | 动态容量基础 | 保留实际容量决定批次；不能把超容量交给 A 后把 retry 当成功 |

### K — KV、闲置 cache、容量压力

| ID | 旧源文件、函数、行号 | 行为与证据 | 大功能专属？ | 边界、取舍、风险 |
|---|---|---|---|---|
| K01 | `server-context.cpp:6089,6158–6165,8235–8245` | 动态池 RAM cache 不再另用逻辑 token cap；空闲 slot KV 保留到压力出现，普通统一池仍可 eager save/clear | 否 | 保留基础策略。RAM cache byte budget 仍应生效；逻辑 token cap=0 不等于无限 RAM |
| K02 | 同文件 `try_clear_idle_slots:6542–6577` | 从“第一个 idle”变为最久未用的非空 idle；有 RAM cache 则先保存；跳过正在处理与 RERoT owner | 部分 | 保留 LRU 选择 B + C save/clear；active episode exclusion 随 RERoT 暂存。真正 A rm 失败需处理 |
| K03 | 同文件 `find_preemption_victim:6579–6624` | 只抢占 generating，排除 protected prefill 与 RERoT；优先物理 target+draft KV 多的请求；recurrent pressure 要选实际占 recurrence 者 | 部分 | B 可返回 victim id，C 执行；usage API 是 fork 新增，不能用假“tokens=物理 cells”补齐 |
| K04 | 同文件 `estimate_next_kv_cells:6705–6755` | Decode 按兼容批次/forced chunk/draft 上限估计；Prefill 从 cache LCP 算剩余且 cap n_batch | 否/有扩展 | 可独立 B 但 C 必须传实际 planned rows；unknown model 估计应保守 |
| K05 | 同文件 `estimate_next_recurrent_cells:6757–6784` | 单独计算 target/draft 新 sequence 的 recurrent slot 需求 | recurrence | 保留独立维度，不能只检查 attention KV 就宣称 admission 安全 |
| K06 | 同文件 `ensure_next_kv_capacity:6786–6802,6852–6875` | 仅统一+显式物理池启用；查询 KV 与 recurrence；全不支持则回 legacy | 否 | 拆分基础 admission；unsupported 必须报告回退范围而非给“池已管理”假象 |
| K07 | 同文件 `6816–6889,6910–6970` | XKV snapshot target/draft 取最紧 safe bound、new-seq gate；先 maintenance fill；错误退出，重试 bounded/no-progress 退出 | XKV | 暂存；算法原则可复用但真实 physical hot/store/workspace 数据必须来自 A，不能复制整个 XKV 接口进 B |
| K08 | 同文件 `7030–7050,7107–7189` | Tri floor 至少128；reclaim 请求是缺口而非整批；target 到 floor 不应屏蔽 draft reclaim；recurrent pressure 分开 | Tri | 暂存 Tri，单独记录这些正确性细节；callback helper 改成 C 顺序执行目标与 draft，不称独立库 |
| K09 | 同文件 `7217–7236` | 压力先 idle 清除，再抢占，protected Prefill 最后可缩小 batch | 否 | 保留基础 lossless policy；必须避免零 batch 无进展 |
| K10 | 同文件 `7246–7292` | XKV 明确 safe bound0 失败；无 XKV 的零空间分支可返回 true | 部分 | 风险：需做 no-progress 测试，diff 不足以证明一定死循环，不能直接报确定 bug |
| K11 | `server-triattention.h:8–20` | state load 成功且真加载时才重新判断 compressed；target/draft 回收独立调用 | Tri | first function 可变纯值 B；second 当前用 callbacks 且返回 A 类型，不能原样称独立 |
| K12 | `include/llama.h:1457–1529,1687–1699` | 新增真实使用量、admission、maintenance API | 基础+XKV | 记 A seam 清单，默认上游无此能力；不能从头文件位置认定 upstream 公共 API |

### R — 抢占重试、生命周期、取消

| ID | 旧源文件、函数、行号 | 行为与证据 | 大功能专属？ | 边界、取舍、风险 |
|---|---|---|---|---|
| R01 | `server-context.cpp:6626–6653` `make_retry_task` | 同 request id/index、cache key、params；由 committed prompt+pending anchor/forced chunk 或 last sampled 重建输入；n_keep=-1 固定为原 prompt | 否 | 可保留 C；原 request 与 resumed prompt 必须分开记账 |
| R02 | 同文件 `preempt_slot:6655–6703` | 保存输出 text/tokens/probs、发送字节 cursor、计数、时序；先 defer 再 `release(false)` 防止中途调度 | 否 | 有真实用户价值，暂不原样移植所有状态对象。保存/恢复失败必须终止或安全从 committed input 重放 |
| R03 | 同文件 `release:1094–1115` | `schedule_deferred` 控制 release callback | 否 | C 队列拥有调度顺序；不能以 A 调 B callback 实现“分离” |
| R04 | 同文件 `7385,7419–7447,10147`; retry_state 1638附近 | fresh sampler 后恢复计数；没有完整 sampler RNG/grammar/repetition 状态 checkpoint | 否 | 不原样承诺随机等价。只允许带明示行为范围的 resume，或深 clone sampler 并证明 token/RNG 连续性 |
| R05 | 同文件 `786–813,1008–1035,10605–10639` | forced token/repetition 状态 reset 与 resume 恢复不是同一集合 | repetition+retry | 可能重置 injection 次数上限；需单独测试，不可被“预算恢复”覆盖 |
| R06 | 同文件 `9690,9911–9923,10651` | retry 即使 cache_prompt=false 仍复用自身前缀；不重复 prompt begin/progress；原 cached count 只赋一次 | 否 | 保留 C 事件语义。自请求重放与跨请求 cache 策略必须分开 |
| R07 | 同文件 `7733–7734,7792–7798,1118–1132,1212–1229,10711–10719` | 最终 prompt 计数不把已有 completion 算输入；累计多段 Prefill time；避免零分母与重复设置首次 generation 时间 | 否 | 保留纯值统计；time 是实际累计，不能按单次 resume 覆盖 |
| R08 | 同文件 `1305–1316` | n-completions 子槽复制 prompt timing/truncated 与 speculative state | 部分 | 普通统计保留；独立 draft checkpoint 完整度另验 |
| R09 | 同文件 `8255–8288,8991–8996` | 普通取消清 retry map；RERoT取消全 episode；abort_all 也清 retry | 部分 | 普通取消必保留，episode 原子取消随该功能暂存 |
| R10 | 同文件 `8953–8965` 等异常处理 | 运行异常向请求返回 error 并 release，而非只写日志 | 否 | 保留异常闭环，确保只发一次 final/error 且清物理状态 |
| R11 | `test_completion.py:406–479`; `test_chat_completion.py:568–591` | 4请求×70token、RAM64/0、SSE一次结束、dynamic slots>4、n=2 并发 | 否 | 已读用例；本轮未执行。没有断言抢占确实发生，也没有 RNG/token 等价断言 |

### C — 请求亲和、cache identity、prompt 状态

| ID | 旧源文件、函数、行号 | 行为与证据 | 大功能专属？ | 边界、取舍、风险 |
|---|---|---|---|---|
| C01 | `server-context.cpp:6348–6376` `get_slot_by_cache_key` | stale id 清除、空 prompt 清绑定、busy 拒绝而保留绑定 | 否 | 保留小 B 数据策略和 C 生命周期；不能被复用 pointer 做 key |
| C02 | 同文件 `6378–6397` | key 命中仍要求 LCP 比例和绝对 min-prefix，空输入拒绝 | 否 | 保留。key 只是亲和建议，不是跳过真实 token 比较的信任凭证 |
| C03 | 同文件 `6399–6414,7468,6575,6699` | slot 绑定新 key 前清旧映射；idle 清除与 preempt 清绑定 | 否 | 保留防 stale affinity；一处遗漏便可能绑定错误上下文 |
| C04 | 同文件 `6417,8190–8197` | `allow_prompt_similarity` 参数与 cache-key miss 路径新增 | 否 | 意图是控制跨 key 相似度 fallback，但 diff gap 未确认参数在函数体内实际消费；需要原完整函数，暂不宣称行为完成 |
| C05 | 同文件 `6316–6326` | 对外 slot 只允许 people 范围，隐藏 pens | RERoT | 暂存；显式 id 路径是否也用同校验尚待完整函数，记录风险 |
| C06 | 同文件 `666–725` | RAM prompt cache 额外保存 speculative opaque state，恢复成功才装载；clear 一并清 spec/Tri 标记 | spec+Tri | 拆分，缓存 envelope 应按 model/policy identity 验证；只保存 target KV 不足以继续 MTP |
| C07 | 同文件 `49–59` | recurrent/hybrid checkpoint 使用 partial + ON_DEVICE | recurrence | 注释称 backend clone 支持已验证，但这里仅 model type 判断；不可当通用硬件能力判定 |
| C08 | `server-context.h:197–211`; `server-context.cpp:1733–1746` | cache key unknown 拒绝；legacy 仅 OFF；live policy/context/adapter identity 变更清 RAM cache | FlashPrefill | 具体 key 内容随实验暂存，但“状态与策略指纹绑定”是可独立 C 约束 |
| C09 | `server-context.cpp:8745–8751,8807–8814,8833–8838` | active RERoT slot 禁手工保存/恢复；FlashPrefill state envelope 拒绝需重 Prefill | RERoT/FP | 保留功能边界说明，不能用普通 public seq save 伪装 shared-KV 序列化 |

### Q — 采样、请求参数、输出协议

| ID | 旧源文件、函数、行号 | 行为与证据 | 大功能专属？ | 边界、取舍、风险 |
|---|---|---|---|---|
| Q01 | `server-common.cpp:899–914,1108–1120`; `server-schema.cpp:323–333` | strict `reasoning_effort` string/null；none/low/medium/high/xhigh/max，none 关 thinking | 否 | 保留 C 输入语义；不接受错类型后静默默认 |
| Q02 | `server-common.cpp:1158–1197` | explicit budget > thinking alias > server default；否则 effort 按 output cap 的20/50/80/95/100%，下界1024上限min128000,output−1 | 否 | 可拆纯数值 B。cap=1 时 upper 被抬为1，不能保证“留1个输出token”；需明确小预算语义 |
| Q03 | 同文件 `1138`; `server-schema.cpp:274–275` | json schema 标为 OUTPUT_FORMAT，不与工具 grammar 混同；影响预填与 grammar恢复 | 否/供RERoT | 保留语义 distinction；旧 grammar helper 大量是移动，不能重复算自创 |
| Q04 | `common/sampling.cpp:328–332` | `penalty_last_n=-1` 用实际 context length；无传入 n_ctx 则模型训练 context | 否 | 保留正确窗口但 C 必须传真正请求/模型上下文，不无意分配巨大历史 |
| Q05 | 同文件 `clone:581–619` | clone 后 `cur_p.data` 指回自己的 candidate vector；checkpoint clone 可省候选但保留 grammar/reasoning/chain/prev | 否 | 具体 dangling/alias 修复有价值。新 C 自己拥有 sampler 与 candidates，不能依赖 common 私有布局 |
| Q06 | 同文件 `621–627` | grammar sampler 追加为链 | 当前mindmap | C 可以公共 chain 组合自己的 A sampler；B 只给 allowed-token 决策，不能接收 llama_sampler* |
| Q07 | 同文件 `682–721` | optional synchronize；backend 已 sampled token 时直接返回，避免 CPU候选 materialize | GPU spec优化 | 暂存此 fastpath；`*_no_sync` 是 fork新增，异步正确性由A保证，不能手工绕过同步 |
| Q08 | 同文件 `726–735,934–952` | compact top-k 前缀，真实 chain 消费保留后续 RNG；拒绝活跃前置过滤器 | 否 | 已选保留独立数值 B + C；历史+3.66%只限定旧TP5场景，不归因给全采样改动 |
| Q09 | 同文件 `21–34,746–766,954–1068` | legacy verify-argmax eligibility+手工dist draw；x86 fast function 有avx2/fma target | TP5验证 | 不原样保留：CPU能力/NaN第一项/手工RNG这些需单独证明；并非 compact top-k 的必要依赖 |
| Q10 | 同文件 `857–915` `common_sampler_replace_grammar` | 先构造成功再换；仅 lazy grammar 回放 committed native end；非lazy不喂旧 `</think>` | 当前RERoT边界 | 通用 grammar切换可保留在 C；单独测试JSON/无budget/未匹配end，不能只测试普通文本 |
| Q11 | 同文件 `921–926` | 以值返回 token history | 否 | 可保留 owned result；不要导出 prev vector 的可变 alias |
| Q12 | `common/sampling.cpp:36–55` | GGML_DECODE_COST gate 记录 sync/select/all microseconds | 否 | 可保留 C instrumentation，但分母和未测量值应明确；不当成功收益证据 |
| Q13 | `common/common.cpp:2436–2454` `common_token_to_piece` | static全局262144项缓存仅按 token id，特殊解析模式条件下复用 | 否 | **不原样保留**：词表不同但token id相同会串文本；并发写同string数据竞争。改为 C per-vocab不可变/单所有者缓存或直接A函数 |
| Q14 | `server-context.cpp:7487–7491` | 非stream且无antiprompt跳过逐tokenUTF8/stop扫描，直接累计text | 否 | 独立保留候选，最后UTF8边界与异常截断需验；尚无独立性能数字 |
| Q15 | `server-task.cpp:1574–1575,1605,1709,1923` | `stream_started` 控制首chunk，不依赖n_decoded==1；支持特殊事件前后 | 否/供RERoT | 保留输出生命周期独立bool，普通n-completion/retry也受益 |
| Q16 | `server-context.cpp:11073–11135,11213–11216,11269–11278` | 多input×多completion，completion全求和，prompt/cache每输入只计一次；聚合所有choices | 否 | 必须保留真实API语义，pure aggregation可B，HTTP事件由C |
| Q17 | 同文件 `11308–11326,11412–11419`;上一聚合函数 | SSE仅最后final携usage；抑制null伪chunk；若抑制usagechunk，把timings移到前一真实chunk | 否 | 可独立保留；测试标准终止一次及最后usage，不让RERoT删减顺带回退 |
| Q18 | `server-common.cpp:1044–1056` | 对确切已知chronicle回填canonical reasoning，未知/篡改输入原样保留 | RERoT | parser前的具体兼容行为可纯映射B；运行时chronicle由C所有，不能信任任意client引用 |
| Q19 | `server-task.cpp:647–718` | 真实explicitreason/content重建think边界，native PEG优先，content parser再兜底raw，tool id稳定 | RERoT | 暂存其特殊路径，不称通用tool parser升级；原生普通parser大部分来自上游 |
| Q20 | `server-common.cpp:1130–1135`; `server-schema.cpp:300–313` | 保存原始messages/tools与固定now供prefix重建 | RERoT | C owned input snapshot有价值；不把可变JSON别名跨库传递 |
| Q21 | `server-common.cpp:1674–1787` | 请求级RERoT开关/trace/frontier严格类型，全局gate；抽最后user的text parts | RERoT | 暂存；后读frontier可重新enable先前rerot:false，是参数优先级事实，不能无条件文档为false绝对关闭 |

### T — 重复抑制、纯文本 parser、工具

| ID | 旧源文件、函数、行号 | 行为与证据 | 大功能专属？ | 边界、取舍、风险 |
|---|---|---|---|---|
| T01 | `server-context.cpp:62–110`; `server-context.h:438–459` | confirmed block detokenize给重复guard，触发后reset，输出固定progress文本且有限次数 | 否 | 重复检测B纯bytes输入→owned结果；tokenization/forceddecode由C。当前factoryA调用B是未分离耦合 |
| T02 | `server.cpp:234–250` | `LLAMA_TOKEN_HACK_PROGRESS_AT>=0`仅开关，不使用其数字作threshold；MAX>=1；TEXT可配 | 否 | 配置名会误导。新接口直接说明enable/max/text，不承诺旧数字阈值 |
| T03 | `server-context.cpp:1008–1035,10605–10639,10744,10925` | forced chunk真正排入后续KVdecode、暂停spec、接受sampler并增加decoded与响应输出 | 否/兼容spec | 独立C需要真实消费；可能先发送forcedtext后下一decode失败，要测试失败边界，不能只输出字符串伪称注入 |
| T04 | `server-rerot-mindmap.cpp:20–194` | 固定Unicode标量范围、UTF8严检；label不允许双空格/markup/emoji等 | mindmap | 可提独立B parser；这套词法是产品协议，不是通用Mermaid接受器 |
| T05 | 同文件 `206–245,316–508` | canonicalDFS序列化、单root、depth不得跳层、无CR/tab、完整闭fence、complete/incomplete/invalid与byte offset | mindmap | 保留有实用的严格解析。FNV64 hash只能辅助比较；“相等 iff tree相等”注释不成立，存在碰撞 |
| T06 | 同文件 `247–314,479–483`; header `19–21,30–35` | suffix grammar与跨token缩进状态；用局部copy试探，成功才commit | mindmap | 独立B state machine。**已见具体契约反例**：最后一个未换行label为`bad  label`时返回incomplete，内部双空格无法由后缀修复，违背“存在合法后缀”保证；应区分可延展prefix与已非法prefix |
| T07 | 同文件 `512–610` | per-vocab immutable tokenpieces，sampler accept/apply/reset/clone真实约束A token | mindmap | 拆分。C拿A词表造值pieces并执行约束；文件下半含A接口，整文件不能直接当B |
| T08 | `server-rerot-thinking.cpp:7–36` | 受限`ul/li`grammar去自由prose、保留路径字符；注释记录1670tokens不关闭li的历史失败 | thinking probe | B可产grammar字符串，C调用A；历史注释不是质量通用证明，不能强加给所有任务 |
| T09 | 同文件 `39–47,73–149` | stop扫描回看terminator−1防token切分；缺终止且EOF报错，非空items、拒嵌套markup | thinking probe | 可独立B。parser比grammar宽松，丢弃首item前head、允许更多whitespace，不要宣称二者完全同一语言 |
| T10 | 同文件 `152–163` | list序列化 | thinking probe | 只对已验证item安全；不转义任意输入，作为HTML输出前另验 |
| T11 | `server-rerot-todo.cpp:7–41,53–119` | 新行不是`-`即stop；partial line到EOF才提交；可CRLF；空项错误；有terminator或EOF才complete | todo probe | 纯std B可独立保留，文档不是完整Markdown parser |
| T12 | 同文件 `121–129` | todo序列化 | todo | 仅受控item；任意item含newline会注入新条目，不能作通用escaping函数 |
| T13 | `server-rerot-audit.h:17–162` | env opt-in、小tensor抽样、bounded block/attention dump、激活RMS/cos/nonfinite | RERoT研究 | 暂存diagnostics，含ggml私有头/graph tensor名字，不独立；文件IO失败有返回但敏感prompt激活不可默认落盘 |

### X — 信号、清理、小配置与合并风险

| ID | 旧源文件、函数、行号 | 行为与证据 | 大功能专属？ | 边界、取舍、风险 |
|---|---|---|---|---|
| X01 | `server-signal.h:17–65,98–112` | self-pipe转信号到正常线程；EINTR、CLOEXEC；handler只写1byte并恢复errno，第二signal `_Exit(1)` | 否 | 保留C POSIX生命周期；class只允许一个活动instance，handler引用必须长于worker |
| X02 | 同文件 `68–95`; `server.cpp:559–569,591,609` | stop恢复旧handlers、发0、join、close；普通退出也能停worker | 否 | 保留；异常partial-install也回滚，不能只测收到signal分支 |
| X03 | `server.cpp:514–524,541,618–621` | stopGC/http/queue、joinHTTP、destroyctx、stopMCP、backendfree；失败加载也cleanup | 否 | 保留C资源拥有顺序。也要确认monitor不再访问先destroyctx（上下文gap需结合完整源文件） |
| X04 | `server-queue.h:16–17`; `server-queue.cpp`删除旧start_loop中的赋true | running初值true，start_loop不重新true，使提前terminate不被撤销 | 否 | 保留语义。这里是普通bool，不得误称atomic/data-race修复 |
| X05 | `common/common.cpp:2177–2189`; `server-context.cpp:9156–9169` | LoRA apply返回bool；失败拒绝batch，防使用旧adapter却回答成功 | 否/部分FP | 保留可见失败；FlashPrefill/RERoT的禁用限制单列 |
| X06 | `common/common.cpp:2235–2251` | cparams适配 n_ctx_kv、n_seq_max、outputs、recurrence | 部分 | **合并风险**：2247无条件重新赋n_seq_max=params.n_parallel，覆盖前面的RERoT LLAMA_MAX_SEQ选择；其它字段也有重复赋值。不能机械cherry-pick |
| X07 | `common/common.cpp:406–569` | TP5仅精确5张不同RX6800+指定模型qualified；显式CLI/env优先，尊重字符串`0`；manifest文件存在才启用；日志effective设置 | TP5 | 暂存runtime，保留配置优先级原则；processwide setenv不适合一个进程多个模型context |
| X08 | `common/arg.cpp:373–434` | u32/u64仅非空十进制数字、拒绝sign/whitespace/trailing/overflow；fraction用stod、必须全消耗且finite并在[0,1] | 否/目前实验使用 | 可保留实际消费者需要的解析。fraction沿用stod接受前导空格/正号等，不是“仅十进制字符”的契约；新C只实现自己真实消费的少量strict整数 |
| X09 | `common/arg.cpp:1778–1800` | --kv-size/--total-kv数字或auto；数字>0强制unified | 否 | 本轮先保留明确数字容量，auto需fit验收；不要接受flag但忽略 |
| X10 | `common/common.cpp:616–625`;旧server-common随机helper删除 | random_string移到common，random_device+uniform | 否 | 不算核心自创收益，可用C自己的ID生成，不携带整个common依赖 |
| X11 | `tools/server/CMakeLists.txt:16–19,38` | server并入RERoT源且添加`${CMAKE_SOURCE_DIR}/src`私有include | RERoT | 不原样保留作独立B构建；这是原server与A私有内部耦合的明确证据 |

### F — FlashPrefill 的小正确性边界（不把它们误归通用 phase memory）

| ID | 旧源文件、函数、行号 | 行为与证据 | 取舍、风险 |
|---|---|---|---|
| F01 | `server-context.h:51–59`; `server-context.cpp:9896–9908` | LCP/cache/checkpoint/last-logit调整后冻结真正Prefill边界；约束0<=past<total | 暂存具体FP metadata，可作为C批次角色设计依据；batch>1不是Prefill判定 |
| F02 | `server-context.h:97–156` | embedding>rerank>multimodal>spec replay>MTP>RERoTforced/frontier>Prefill>Decode；仅Prefill/teacherforced可sparse | 暂存真实sparse消费者未移植的角色，不出只有flag的空模块 |
| F03 | `server-context.cpp:10214–10256` | row metadata与实际slice数量/version结构匹配，不把旧整批metadata套子slice | 小正确性必须记录，未来port需失败测试 |
| F04 | `server-context.h:220–325`; `server-context.cpp:10373–10384` | 成功decode才计rows/plan、读取GPU已测counter、失败重试不重复计数 | 保留指标原则；未测GPUtime不得宣称零成本 |
| F05 | `server/README.md:2038–2162` | off/auto/required、实验矩阵和quality gate，KV policy cache isolation及实际counter含义 | 只是文档与测试协议，无本轮整体质量/吞吐验收。FP暂存不能写成“证明无收益” |

### 后续精读补充

| ID | 旧源文件、函数、行号 | 单独行为与判断 | A/B/C 边界与风险 |
|---|---|---|---|
| X12 | `server-task.h:920–927` `slot_action` | `id_slot = -1` 默认初始化。普通completion的move也会移动该member，修复原未初始化int；与RERoT无关 | 建议保留一行普通server修复。类型的其它默认字段是否初始化要另外检查，不把这一行说成全部任务安全 |
| C10 | `server-task.h:1026–1037` child task clone | child清cache_key且清episode id，防独立completion继承同亲和和episode；child seed偏移是未改上游行 | cache亲和清理可保留C；episode部分随RERoT。不要把上游原有seed行为算自创 |
| T14 | `server-rerot.cpp:177–309,1113–1135` JSON duplicate scan | 试图拒绝重复成员，但221用raw JSON字面量作set key。`strategy`与`\u0073trategy`漏检，parse先保留最后成员后会按simple通过；所有syntax异常又一律incomplete | 不原样保留parser；B必须按解码后的键检测，且分别定义截断与不可修复语法错误 |
| T15 | 同文件 `1138–1280` routing decision | 根仅strategy/payload；simple只许空payload；DAG严格questions/id/intent、禁止空白与reserved0、重复id/edge、自环、未知端点；Kahn检查环 | 是有价值的独立值治理B，当前调用者RERoT；C未来必须实际消费才称恢复运行功能。无节点/深度/字节预算，不宣称抗无限输入 |
| K13 | `server-task.h:1342–1384` `xkv_maintain_max_attempts/should_maintain` | 根据缺口ceil(deficit/segment)+4域确定重试上限；unknown按capacity或域数保守；shadow始终maintenance直到bound，普通fill-first | 纯策略可B，真实snapshot/maintenance由C消费。属XKV当前消费者，暂存，不要为了小函数把整体XKV带回 |
| K14 | `server-task.h:260–398,775–838` XKV metrics | requested与observed/effective配置分开；uint64 seed用精确字符串；未测比率/时间/counter/null或NaN，不假填0；runtime未armed与OFF分开 | 观测原则可独立C保留，具体上百XKV字段随runtime暂存；当前只精读部分字段与serializer段，不宣称整个metrics正确 |
| X13 | `common/CMakeLists.txt:101–103` | 原common直接编译wanxiangqi的arg-options/params，独立参数功能与A耦合 | 不原样带进独立B；C拥有自己的输入解析并显式消费真实参数。只换include路径不能分离 |
| R12 | `tools/server/tests/unit/test_basic.py:19–27` | 启动smoke从只GET health补上真实8token completion、ignore_eos、limit结束 | 有必要的最小功能gate，迁移server时保留；本轮只读未执行 |

## 证据范围与后续覆盖

- 此 checkpoint 对服务端主循环、动态容量、retry、请求/输出与若干纯parser做逐行为审读。未完整审核 `server-rerot.cpp` 6716行与其大header、`common/fit.cpp` 2684行新增、`common/speculative.cpp` 全实现；coverage明记部分或未读。
- `audit-old-numstat.txt` 中存在大量 RERoT/XKV/容量/信号等测试，只确认文件存在时不授予“通过”状态。`test-server-signal.cpp`、`test-server-task.cpp` 等应由后续源码/执行确认，不能由文件名推断全部覆盖。
- 核心 phase/physicalKV 实现、GPU kernels、模型新增架构由另外的审计负责。这里把明确使用的新增seam逐项标出，避免边界丢失。
- keep/drop/defer 的性能依据仍按父任务已读取文档：compact top-k历史+3.66%；TP5 LateBind约9.24%有价值暂存；BF16router+1.14%低于2%gate与row-split−7.22%不带；MMVQ+MTP确认CI含1，不携无显著净收益的联合实验；FP/XKV整体gate未闭合暂存。这些数字不能为上面的所有小改动批量背书。

## 整合后的交付状态

TQScheduling 和 batch-compose 已进入 clean 分支，并完成真实 A/B 安装包构建、搬迁消费者与 CPU/Vulkan 三请求逐批对照。原版 server 的有限调度/phase 接缝已通过真实三路 SSE 验证；完整 HTTP 编排仍是 A 例外。当前测试脚本与源代码在 `fork/scheduling`、`fork/batch-compose` 和 `fork/tests`；先前候选 patch 文件不是当前交付入口。具体范围见上面的最终验证链接。
