# TP5 重写后的剩余候选：按证据决定重写

审阅日期：2026-10-19（Asia/Shanghai）。本文是需求与重写依据的文本记录。
**本文全部候选均为 TEXT_ONLY / NOT_ADMITTED；没有迁移旧实现，也没有取得新的模型或设备运行资格。**

## 1. 审阅身份与使用方法

| 对象 | 固定身份 |
|---|---|
| 当前 clean 分支 | `clean/fork-20261018` |
| 本次审阅起点 | `7ccaa898451270e395ab1cf150b2c62e9fb1f2a5` |
| A：固定上游 | `cda0f9256665845cd9d6f3a0176880883f653a3d` |
| 旧 fork / 归档来源 | `2777543b2358c88cd1c2000f7fa342b764a0e213`，`archive/pre-clean-20261018` |
| 逐文件总账 | [audit/files.json](audit/files.json)，714 个非 artifacts/reports 路径 |
| 总账 SHA-256 | `1356aa167c1ebf247e55ab2b8dab9b14aba8c8514367c27bf18c4d549df26368` |
| 起点的 A 例外清单 SHA-256 | `79c426aee09fa4395bb7b1685d8dc083a75238c6bca62a31ebbf64b8bc96e9be` |

本次把当前 README、迁移表、relay/probe 交付说明和新增路径，与已有逐文件总账、差异及重点原函数交叉核对，并阅读下列原论文。总账原有的 546 项完整所供差异/文件阅读、168 项选定范围阅读是历史阅读深度，**不是本次逐行重审或 714 项运行通过**。归档性能、注释、fixture、脚本返回 PASS、可编译、论文结果均不自动证明旧实现可迁。未见充分证据不等于已证明无用或必然错误。

每个条目保留四件事：具体消费者；固定归档中的入口；已经知道的证据缺口；重新实现后应满足的最低条件。旧 ID 如 `small:S05`、`server:K01`、`G14` 用于回查对应[审计分册](AUDIT.md)，不是新测试编号。下文列出的是未来验收条件，本次未执行这些实验。

**用户已经重新实现 TP5。** 当前路线以 [TQRelay](relay/README.md)、[relay-probe](relay-probe/README.md) 和 [VALIDATION.md](VALIDATION.md) 为准；不把旧 TP5 runtime 再列为迁移任务。平台探针已经落地，旧迁移表“探针尚未落地”的描述应更新。单设备 probe、五 rank 索引 fixture、真正五设备探针和整模型五卡运行各有自己的证据范围，互不替代；本次不重新审定用户这条路线的完整运行资格。

### 已交付能力的去重

| 已有能力 | 此次只保留的相邻缺口 |
|---|---|
| Prefill 优先、token/驻留槽位 admission、真实 batch C | 压力观测、idle 回收、真正抢占/恢复、并发上限与 reserve 形状 |
| 逻辑窗口与 attention 物理 KV 分离、已有统一分配器 | recurrent 物理容量/COW、碎片压紧、真实共享占用与容量测量 |
| 每 context 的 Prefill/Decode compute workspace 释放与字节观察 | 跨 context 的共享所有权、device staging 的最终回收；不重复包装已交付 API |
| TQCorpus V1、离线构建与 prefix replay | 在线事件的身份/时机、崩溃或重启后的完整性 |
| TQCalibration reader、imatrix 导出 | 真正模型 callback 采集、明确统计人口、checkpoint 和原始样本 |
| 全词表 logits 采集与严格比较器 | 身份对齐后的概率指标、任务质量与统计报告 |
| TQText、batch C 的 UTF-8/stop/流式输出 | 多 choice HTTP/SSE、取消、重试和一次终结语义 |
| 已验证的 CPU F32 attention、Vulkan WHT/scratch/输入别名、Meta 非零 offset | 本文另列的设备所有权、普通异步路径和有具体消费者的小项 |
| 当前 A 的 recurrent clear/reuse 与已有 speculative 选项 | recurrent 独立容量与 COW；direct-hidden 实际交接及正确测量 |

### 建议次序

1. **先补可复用基础：** E01–E08 的请求与采集证据链，以及 E09–E16 的容量、所有权和失败行为。它们可服务 TP5、普通推理和后续研究。
2. **再做有目标负载的论文重写：** Prefill 瓶颈看 P01；长 Decode 的 KV 压力看 P02；跨层冗余看 P03。先分别建立基线，组合方法另验。
3. **有具体消费者再做：** E17–E23 的抢占、隐藏行交接与多 reader，P07–P09 的新编码/训练实验，以及第 5 节的小 kernel。
4. 新代码只有在目标消费者、当前 A 的公开接口、反例与端到端价值明确后，才进入单独提交。B 拥有独立数值/计划/状态，C 调用 A+B；A 内部资源问题由最小、登记清楚的 A 修复承担。把私有结构移到 `fork/` 或增加 callback 不会自动消除耦合。

这里的先后是维护成本、证据依赖与可用性的工程判断，不是未经测量的性能排名。

## 2. 请求、校准与质量基础

### E01 — HTTP 多请求、多 choice、取消与一次终结

**消费者与价值：** 真实 server 客户端需要可靠的流开始、输出、usage、错误和结束事件。当前 batch C 的文本合同不覆盖整个 HTTP 生命周期。

**归档入口：** `server:Q15/Q16/Q17/R06–R10`；`tools/server/server-task.h` 的 `task_result_state::stream_started`，`server-task.cpp` 的 first_chunk/update/序列化；`tools/server/server-context.cpp` 的 `server_accumulate_oai_usage/server_aggregate_stream_usage`、响应 reader 与终结分支。

**已知限制：** 旧改动避免 headers-only begin 产生 JSON null 伪 chunk，并处理重复开始与多 choice 重复计 prompt；空 output 只发 headers 是允许的。这些分支的存在不足以证明取消、背压和重试组合工作。不能把自然 stop、错误、断开连接统一标成成功完成。

**重写边界与最低条件：** C 拥有每请求/choice 的事件状态和已发送游标，复用 A 推理及已交付文本能力。用真实 HTTP 覆盖 2 个输入 × 2 个 choice，首个 progress/begin、空文本、special token、断开、取消、重试、背压；确认 prompt usage 的分母、每 choice 终止原因、最多一次终结及正确资源回收。来源是工程需求，无需冒认算法论文。

### E02 — Cache affinity、身份与陈旧映射

**消费者与价值：** 多轮请求可尽量复用正确的 prefix，减少无效 Prefill；key 只能是路由提示。

**归档入口：** `server:C01–C04/C08/C10`；`tools/server/server-context.cpp::{get_slot_by_cache_key,cache_key_slot_has_enough_similarity,clear_cache_keys_for_slot,bind_cache_key_to_slot}`，旧文件约 6348–6414 行及任务绑定/失效分支。

**已知限制：** 旧代码仍以实际 token LCP 判断相似性，空/失效槽要解绑，busy 槽暂不能被选中；缺少当前模型/tokenizer/template/adapter/RoPE 身份与真实命中路径的完整证明。相同字符串 key 不能授权复用不一致状态。

**重写边界与最低条件：** B 可拥有不含 A 指针的 affinity 元数据；C 负责真实序列及全部执行身份。同 key 不同 tokens、模型/adapter 切换、busy、清空、重绑、child completion、restore 失败均要覆盖；另测实际 cached token 数和 TTFT。SGLang 的 radix cache/路由是参考 [R01]，不是已查明的旧 slot-key 算法出处。

### E03 — 有压力才回收的 idle KV LRU

**消费者与价值：** 闲置缓存可提高后续命中；压力出现时又应让活跃请求取得实际空间。

**归档入口：** `server:K01/K02/K09`；`tools/server/server-context.cpp::try_clear_idle_slots`（约 6542–6577 行）及调用处。

**已知限制：** 旧行为包括优先清最久闲置槽、可选 RAM 保存、清理后解绑；未取得当前真实 reclaimed capacity、共享 prefix、save/remove 失败的证据。删除映射或减少账本数字不能证明 A 已释放空间。

**重写边界与最低条件：** C 读取 A 压力、选择 idle 受害者、完成真实保存/移除后才更新预算。测试无压力保留、LRU 顺序、active 不动、共享物理占用、保存失败、删除失败、再次命中/回放。SGLang 的 LRU/active reference 可借鉴 [R01]；slot LRU 与 radix leaf eviction 是不同设计。

### E04 — 在线输入 corpus、事件身份与重启完整性

**消费者与价值：** 从实际服务流量获得可追溯的校准/回放输入。

**归档入口：** `small:S01` 在线部分；`wanxiangqi/server/server-corpus-dump.cpp::{append,load_existing,write_request_record}`；`tools/server/server-context.cpp` 约 11231–11240 行；`wanxiangqi/docs/request-corpus-dump.md`。

**已确认的重要纠正：** 旧钩子先 `dump_request_tokens(task.tokens, ...)`，后 `tasks.push_back` / `rd.post_tasks`。记录的是 **入队前、模板与 tokenization 后的输入**，不是请求成功结束后的输出。它不能证明请求被接受、重试只记一次或生成成功；多模态占位被略去的 text-only 序列也不能重建完整多模态请求。旧 V1 格式互操作不覆盖这些生命周期。

**重写边界与最低条件：** 先选明记录事件是 received、admitted 还是 completed，再由 C 维护稳定 request/event ID 与 disposition。保留 V1 时，可另加有版本的来源 manifest，记录模型、tokenizer、template、adapter 和完整性。验证空请求、重复、拒绝、取消、重试、各 request kind、写失败、分片与重启；`flush` 不等于 durable，若只承诺同步独占写入，应明确该范围。属于工程需求。

### E05 — 真实 MoE 统计、稳定抽样与人口定义

**消费者与价值：** 为量化、剪枝和专家分析提供真正的路由次数、输入二阶矩及 hidden 能量。

**归档入口：** `small:S05`；`wanxiangqi/calib/trie-calib.cpp` 的 `collector::{kind_of,want,observe_gate,observe_down}`、`stage_prefix`、`cb_eval`；`wanxiangqi/lib/linalg.cpp::accum_cov`。

**已知限制：** 旧 callback 依赖权重命名、F32/I32、shape 和 gate/down 顺序；不满足时有静默跳过。`act_stride/route_stride` 使用外层 `llama_decode` 次数，而每 callback 又截取前若干 token；抽样启用时，改变 batch/ubatch 或 DFS 边界可能改变样本人口。按唯一 trie node 与按请求重复权重统计也不同。`accum_cov` 实际积累未中心化 `sum(x x^T)`；不能因为名称含 covariance 就机械改成中心化统计。

**重写边界与最低条件：** B 拥有统计、采样计划与结果；C 验证当前真实 A callback 并复制所需数值，B 不持 ggml 指针或接受 A 服务。固定真实 MoE/token tape，以少量原值和独立 F64 计算核对 route/sum/sum2；覆盖稀有/未见专家、重复 expert、stride、分块变化和不支持模型。明确路由与 activation 的各自分母。GPTQ/OBC [R08]、[R09] 可解释二阶重建目标，旧 collector 不是它们的完整复现。

### E06 — DFS 复用、partial dump 与真正可恢复 checkpoint

**消费者与价值：** 长时间校准应保留诚实的完成范围，并在支持时避免完全重跑。

**归档入口：** `small:S05`；`wanxiangqi/calib/trie-calib.cpp::dfs_driver::{visit,decode_span,maybe_dump}` 与文件级 `write_dump`；`wanxiangqi/lib/calib-dump.cpp::write_calib`。

**已知限制：** 保存统计不等于能恢复遍历。旧 `max_nodes` 可在最后一批 overshoot，`complete=ok` 只表明已执行部分没有报错；DFS 的 unique-node 人口不等于普通请求重放。现有 WXQCAL03 reader 不能关闭这些缺口。

**重写边界与最低条件：** C 拥有遍历位置、pending batch、抽样状态、统计与受 A 支持的模型 snapshots；先交付可靠 partial，再谈 resume。在分叉/短叶、窗口重置、非整批限额、save/restore/写失败处验证；恢复后无重复/漏计，中止范围明确且不能伪装整 corpus 成功，失败不覆盖已有完整文件。设备丢失后的可恢复性要单独实测。属于恢复工程需求。

### E07 — 原始 activation 与可追溯样本封装

**消费者与价值：** 新 student/量化方法需要原始小样本；汇总矩不能替代层输出误差重算。

**归档入口：** `small:S25`；`wanxiangqi/probe/wxq-manifold.cpp::{collector::fetch,collector::observe,write_dump}`，旧格式 `WXQMAN01`。

**已知限制：** 旧读取假定 contiguous，down 只取 expert slot 0；缺 request/node/token/模型身份和 complete。decode 失败可只退出内层循环，最终仍返回成功。

**重写边界与最低条件：** C 输出 owned 样本及 request、node、token、position、layer、expert ID、slot、dtype、来源指纹和 reset 策略。对实际 CPU/设备 callback 与 strided/F16/F32 数据核对 ID/值，严格限制 bytes/records，验证中止、写失败与退出状态。没有可据此认定的特定蒸馏论文实现。

### E08 — 严格对齐的分布差异与尾部风险

**消费者与价值：** 评价新量化/压缩对模型预测分布的影响，复用现有全词表 tape。

**归档入口：** `small:S27`；`wanxiangqi/probe/wxq-kl.cpp::{score,ref_rec}`、DFS 与 `WXQKL001` reader。

**源码与代数反例（本次未执行旧工具）：** 旧指标是 `sum(i in S, p_i log(p_i/q_i)) / p(S)`。它既不是全词表 KL，也不是双方条件化后的 KL。例如 `p=(0.6,0.4)`、`q=(0.9,0.1)`、`S={0}`，该值为 `log(2/3)<0`，而单元素条件 KL 为 0。旧 node mismatch、早 EOF 及 pending batch 未 flush 就 reset 也使结果不可直接作质量门。

**重写边界与最低条件：** 对模型/corpus/tokenizer/node/position 严格对齐，计算完整 KL，或清楚命名的 top-k 加聚合 tail 的粗粒化 KL；条件 KL 另报。B 可计算纯数值，C/CLI 负责文件身份。解析两类例、相同分布、零/极小概率、完整 support、NaN/Inf、缺行、早 EOF 和 token mismatch 必须有定义。quantile/CVaR 写明人口和分母。这里的纠错来自定义与直接代数；不需要引入旧评分运行时，也不替代任务质量。

## 3. 容量、后端生命周期与进阶服务

### E09 — 分单位的物理占用与压力快照

**消费者与价值：** admission、idle 回收与容量规划需要知道实际可用资源。已有 token/slot reservation 不应重复重写。

**归档入口：** `M08/M09/M21、RMC55`；`src/llama-memory.h`、`llama-memory-hybrid.h` 的容量/压力接口，`llama-kv-cache.cpp` 与 `llama-kv-cells.h` 的计数。

**已知限制：** 旧状态把 attention cells、逻辑序列、recurrent slots、bytes 等并列；共享 prefix 的物理 cell 应只计一次，每序列引用可以分别计数。无类型总数或只有模型估算，不能代替实际可分配空间。

**重写边界与最低条件：** A 提供其拥有资源的有版本 POD 快照；C 合成决策；B 只消费声明单位的数据。用共享 prefix、空槽/洞、recurrent 混合、snapshot pins 与失败释放验证计数；分清 committed/live、reserved、peak、driver estimate，不把 XKV 整套状态搬入观察接口。工程需求。

### E10 — 可靠 auto-fit、Prefill 并发与初始 reserve

**消费者与价值：** 保留目标逻辑窗口，在真实设备上确定 attention/recurrent 容量、PP/TG 工作区及并发。

**归档入口：** `M01/M02、R06–R10`；`common/fit.cpp` 的 probe、求解及 recurrent 估算，`common/common.cpp` 初始化；Prefill 并发/reserve 的小项另见第 5 节补充。

**已知限制：** 旧 solver 可能接纳失败或不完整 probe，存在 decode/fence 缺失、不同阶段预算定义混用、全局缓存和 resident 清除时机问题。已交付 phase release 不证明 auto-fit 已可靠；首次图按单序列 reserve 也不等于以后并发请求可以共用同一峰值。

**重写边界与最低条件：** B 只从 owned 测量记录求下一个有界 probe 和接受/拒绝；C 建真实 A context、完成 decode/同步后测量。验证非单调/空/失败测量、无可行解、算术边界、placement；最终接受配置必须完整执行目标 PP 与 TG。分别报告元数据估算、A allocator bytes、设备估计与实测峰值；Prefill 优先、并发上限、首次 reserve 形状分别说明。无须借用不相干的论文给旧 solver 背书。

### E11 — 独立 recurrent 物理槽与 COW

**消费者与价值：** 排队请求的逻辑 ID 空间可以大于真正驻留的 recurrent 状态数，共享后继续写入时保持隔离。

**归档入口：** `M06/M07`；`src/llama-memory-recurrent.cpp` 的构造、prepare/update 和 COW，`n_seq_recurrent`。

**已知限制：** 这是独立于 attention `n_ctx_kv` 的物理资源模型；之前 clear/reuse 与 snapshot 回归并未证明可改变其容量或安全处理所有 COW。也不能借 RERoT “brain” 容量混用普通 recurrent 槽位。

**重写边界与最低条件：** A 拥有实际 buffer/尾状态/COW；C 负责驻留和排队；B 可持容量与计划。用真实 hybrid 模型验证逻辑 ID 超过物理槽、共享 parent 后分别写、正好容量与超出一槽、prepare 失败后状态、恢复后 continuation。当前 A 无最小公开能力时再提出有消费者的 A 接缝。

### E12 — Pinned host buffer 的设备身份

**消费者与价值：** 普通 Vulkan 多设备上传/读回也需要正确 owner，不依赖旧 TP5。

**归档入口：** `G14`；`ggml/src/ggml-vulkan/ggml-vulkan.cpp` 的 host alloc/free/getter/alignment/max-size 和 `ggml_backend_vk_set_tensor_2d_async`。

**已确认的细节：** 旧 allocator 已加入 device 归属，但相邻 2D async 消费者仍有只接受全局 device-0 host buft 的检查。只搬 alloc/free 几行无法闭合消费者身份。

**重写边界与最低条件：** 先在当前 A 的普通公开路径取得反例，再修其资源 owner；不为 A buffer 创建空壳 B。使用两个真实设备，覆盖各自 alloc/free、offset/stride、失败 fallback、多实例与错误 device。Vulkan 资源销毁的 device/完成要求见官方规范 [R12]；新 HIP probe 的 mapped-memory 资格不能代替普通 Vulkan 路径。

### E13 — 异步复制两端保活与可见性

**消费者与价值：** 普通 readback、allocator 增长/复用、异步 teardown。

**归档入口：** `G11`；`ggml-vulkan.cpp` 的 `vk_context_struct::copy_buffers`、`ggml_vk_buffer_write_2d_async`、`ggml_vk_buffer_read_2d_async`。

**已知限制：** 旧代码保留两个 buffer 并插入 host 可见性屏障，但包装器与旧 TP5 交织；历史 GPU fault/注释不能证明当前修复完整。device 执行结束、host 可见和 caller 允许的生命周期是不同条件。

**重写边界与最低条件：** A 保活真实提交引用，C 遵守公开异步接口。通过合法的公开调用构造延迟完成、allocator growth/reset、连续两次 readback、fallback、teardown；验证数据和回收时机，不能靠故意违反 caller lifetime 来“复现”。官方同步例 [R14] 是规范参考，不是旧补丁运行证据。

### E14 — Device 共享 staging 的最后 owner 回收

**消费者与价值：** 多 context、模型卸载重载、auto-fit 反复创建 probe。

**归档入口：** `G18`；`ggml_backend_vk_free` 及 device `sync_staging`。

**已知限制：** 旧 backend 析构直接清理 device 共享 staging，不能据“8 次重载 OOM”注释推断其他 context 已停止使用。它与单 context compute workspace release 不是同一所有权。

**重写边界与最低条件：** A 维护 device 级 owner 和在途任务，最后 owner 且全部引用完成后才回收 [R12]。两个 context 共用 device，先释放一个，另一个继续工作；最后释放后实际 owned bytes 回收，再重复加载。当前 A 是否仍有这个缺口要先复现。

### E15 — 无损 KV 物理压紧与字节移动

**消费者与价值：** 清理洞和碎片而保留全部存活 tokens；与 Tri 淘汰、xKV 有损近似分开。

**归档入口：** `M23/M18、RMC05`；`src/llama-kv-cache.cpp` 的压紧、`llama-kv-cells.h` 重排、`ggml_backend_tensor_memmove_regions`。

**已知限制：** 旧计划依赖目标不晚于来源、K/V stride 和布局；有量化转置 V 未支持的路径。部分复制成功后的失败不能自动称为 rollback。

**重写边界与最低条件：** B 可从 owned cell 描述返回检查完毕的移动计划；A 执行实际 buffer 操作并明确提交/失败状态；C 决策。验证全部 packed bytes 与 guard、overlap、多 region、共享 cell、稀疏位置、zero rows、算术边界及中途失败；不支持布局显式拒绝或走完整参考。只有消费者碎片率/延迟值得时才增加接缝。

### E16 — Generation、快照身份与失败导入

**消费者与价值：** cache reuse、compaction、恢复与后续压缩都不能使用过期身份或半导入状态。

**归档入口：** `M31/M38、R12/R13/R19/R20、RMC54`；`llama-kv-cells.h` 的 copy/move；`llama-kv-cache.cpp` 的 seq keep、shift、state import；recurrent 预检。

**已知限制：** 旧特殊成员遗漏新 payload 字段；counter 可 wrap；某些失败预检之前已经修改状态。名为 transaction 的结构不提供原子性证明。

**重写边界与最低条件：** A 先验证完整输入、geometry、domain、模型/generation 和申请结果，再按清楚的提交协议变更；失败作用范围明确。复制/移动身份、计数饱和、陈旧 snapshot、错模型、截断输入和逐阶段失败均要覆盖。未实现 rollback 时不能承诺失败完全无副作用；不为此重建通用回调事务框架。

### E17 — 真正的抢占、重试与采样状态恢复

**消费者与价值：** 有明确压力/公平性目标时，让长请求暂让资源且保持客户端语义。

**归档入口：** `server:K03–K06/R01–R10`；`tools/server/server-context.cpp::{make_retry_task,preempt_slot}`（约 6579–6703 行）及延迟 release。

**已知限制：** 旧实现保留 prompt/output 游标、时间和计数，却新建 sampler；计数恢复不能恢复 RNG、grammar、repetition 等全部状态。旧测试未必真的触发抢占。

**重写边界与最低条件：** C 拥有 A 状态、sampler、请求身份、已输出游标与恢复协议，B 只做受害者/计划。强制真实抢占，覆盖 greedy、状态化随机采样、grammar/history、snapshot 失败、预算与一次输出；失败重试有界。PagedAttention 的 preemption/swap/recompute 是设计参考 [R02]，不能替代本 server 的 sampler/HTTP 合同，也不意味着必须迁入整套 paged runtime。

### E18 — Grammar 变更与 sampler clone 的所有权

**消费者与价值：** 请求分支、grammar 重新配置和取消后的独立采样。

**归档入口：** `server:Q05/Q10/Q11`；`common/sampling.cpp` 的 clone、`cur_p.data` 重绑定、grammar 替换及 lazy replay。

**已知限制：** clone 的指针需要指向自己的候选数组；新 grammar 应构造成功后再替换；部分 committed piece/end 需正确重放。先检查当前 A 是否已具备所需行为，不能从旧差异推断仍缺。

**重写边界与最低条件：** sampler 是 A 的对象，C 通过公开接口使用；源 sampler 销毁后 clone 继续、替换失败不破坏旧状态、跨 piece 的 grammar end、后续 RNG 行为均验证。该需求不因已有 TQSampling 数值 top-k 自动完成。工程需求。

### E19 — Speculative 周期测量与真实提交数

**消费者与价值：** 判断新的 draft/hidden 方案是否值得维护，找到浪费的 draft 或 catch-up。

**归档入口：** `FC002`；`common/speculative.h` / `common/speculative.cpp` 的周期统计、settle guard 与 `final_tokens=accepted+1`。

**已知限制：** guard 可能把失败周期也登记，缺 success/commit 标志时 `accepted+1` 不等于已提交输出。阶段时间相加不一定等于用户 wall time；平均各周期比率也不等于总接受率。

**重写边界与最低条件：** C 区分 attempted/successful/failed、drafted/accepted/committed，记录相同 clock 的端到端时间与阶段。失败 catch-up 不计输出收益；可从原始事件重算总量。Speculative Decoding [R03] 的目标是维持目标分布，不能把“相同 seed 必须逐 token 相同”强行当所有随机实现的论文承诺；具体 RNG 可复现性另定义。

### E20 — Accepted hidden row 计划与真正 draft 交接

**消费者与价值：** 当新的 draft 算法实际需要 target hidden 时，避免错误行或无收益的主机往返。

**归档入口：** `FC003/FC004、small:S19`；`common/speculative.cpp` 的 RESULT/CARRY/SEED 行计划；`include/llama-predefined-hidden.h`。

**已知限制：** 旧 MTP 的 `set_state` 可在长度验证前 reset。交接依赖旧 fork 的 feature/predefined 接口，其中 RESULT/CARRY/SEED 的 MTP 路径还要求旧 TP5/Qwen4EXP；DFlash `enable_direct_hidden` 则有同一非 CPU 设备的独立路线，没有该 TP5 条件。存在 host API/probe 不证明 draft 消费了正确 target hidden，也不证明净收益。

**重写边界与最低条件：** B 可拥有 row/generation/shape 计划；C 持真实 A 对象，必要 A seam 另提。短输入、stale、overlap、越界应在修改前失败；用真实 target/draft hidden 与 host 参考核对 accepted/carry/seed 行、寿命、取消/reset，再测完整净收益。DFlash [R04] 提供 hidden-conditioned block diffusion 方法依据；另已核对 DSpark 原论文 [R19] 的 semi-autoregressive draft 与 confidence-scheduled verification；它与 DFlash 是不同方法。论文都不证明旧 direct-hidden API 工作；先复用 A 已支持的 draft 方法，待重写的是有明确缺口的交接与测量。

### E21 — Repetition 诊断而非隐式改写输出

**消费者与价值：** 找出循环生成、观察退化或供用户显式开启中止策略。

**归档入口：** `small:S06、server:T01–T03`；`wanxiangqi/common/repetition-guard.cpp`；`tools/server/server-context.cpp` 的强制注入文本分支。

**已知限制：** 旧指数遗忘/distinct-token 分数和阈值是项目启发式；将各文本 chunk 独立用 o200k tokenization 会使结果依赖切块。旧 95% 代码样本阈值不能代表所有任务，强制插入进度文本也会改变输出。

**重写边界与最低条件：** B 按真实生成 token ID 维护有界诊断，C 选择可见告警或明确 opt-in 中止策略。验证切块不变、reset、计数边界，分别衡量代码/自然语言/重复格式任务的误报。没有已识别的算法原论文；先保留诊断价值。

### E22 — Signal、退出与失败传播

**消费者与价值：** 正常停止服务、启动失败和运行期错误时，调用者能得到可靠结果。

**归档入口：** `server:X01–X05/X12/R10`；`tools/server/server-signal.h`、`server.cpp`、`server-queue.h`；`common/common.cpp` 的 LoRA 失败传播。

**已知限制：** 旧 self-pipe、普通线程清理与第二信号机制有价值，但普通 `bool running` 不自动解决线程竞态；错误不能只打日志。普通 task 的 `id_slot=-1` 已交付，去重。

**重写边界与最低条件：** 先核对当前 A，再由实际应用 C/有限 A 修复承接，不另造通用生命周期 B。验证首次 signal、第二次 signal、初始化失败、已有活跃请求、取消与 adapter apply 拒绝；信号处理器只执行允许的最小操作，正常清理走正常线程。属工程/平台合同。

### E23 — RERoT 的协议小件与多 reader 研究分别留档

**消费者与价值：** 多路任务输出、人可读调试、结构化计划和真正的共享上下文推理有不同消费者，不能混为一个已工作“大功能”。

| 可保留目的 | 归档入口及限制 | 重新实现后的最低条件 |
|---|---|---|
| 按 lane 拼完整行、隔离 pending 文本 | `FC009`；`tools/server/server-rerot.h` 的 lane mux；换行/terminal residual 需按 lane 归属 | B 拥有各 lane 状态；任意 chunk partition、交错 lane、终结残留、UTF-8 与有界缓存；一个 lane 不吞另一个的残片 |
| 增量 JSON、稳定 node ID、DAG 检查 | `server:T14/T15`；`server-rerot.cpp`；旧 raw key 检测可漏掉 `strategy` 与 `\u0073trategy` 的等价写法，语法错误也不能全报“未完成” | B 明确 complete/extendable/invalid 三态，decode 后判 key，检查循环、深度/节点/字节限额，C 消费 owned 计划 |
| Thinking/mindmap/todo 文本协议 | `server:T04–T12`；旧 HTML/PAC-DFS 后续生产路径可 hard abort | 先选一个真实消费者与稳定输入格式；完整分块、失败和显示语义通过后才考虑功能，不恢复所有旧协议 |
| Chronicle → canonical 的精确历史映射 | `FC012`；`tools/server/server-rerot.h::{server_rerot_register_chronicle_mapping,server_rerot_resolve_canonical_reasoning,server_rerot_clear_chronicle_registry}`；未知/篡改/编辑返回 nullopt，`server-common.cpp` 调用侧保留原文 | 若有真实消费者，验证精确已知映射、未知/编辑输入保持原文、注册/清理生命周期；通用可编辑协议不是这段旧代码已提供的功能 |
| 共享 KV 的 reader 可见性 | `RMC43、FC010`；`src/llama-rerot.*`、`RERoT.md` | 先用小型精确 oracle 验证同 context/owner、PUBLIC 只看已提交 peer frontier、private 隔离、GQA/partial RoPE、单一全局 softmax、recurrent/RNG 独立；再做真实模型与性能 |

本次没有核实到可作为这整套 RERoT 算法来源的公开原论文；记录为项目研究设计。RoFormer [R05] 仅能给其中 RoPE 数学提供参考。普通 batch、prefix replay 或 parser 测试不能授予动态共享 reader 的运行资格。

## 4. 值得按原论文或明确数学合同重写的方向

### P01 — FlashPrefill V2 的选块与 mean correction

**消费者与价值：** 实际长 prompt 的 Prefill C；目标是降低 TTFT 和 attention 成本。复用现有阶段边界，不重复实现 phase memory。

**归档入口：** `RMC21/RMC22/RMC24/RMC26/RMC27、FC006、server:F01–F05`；`src/llama-flashprefill.cpp`；`ggml/src/ggml-flashprefill.cpp::{ggml_flashprefill_ref_means,ggml_flashprefill_ref_select}`；Vulkan `flashprefill_{pool,select,attn,merge}.comp`；`PREFILL.md`。

**原论文与版本：** FlashPrefill v1 [R06] 和 FlashPrefill V2 [R07] 必须分开。旧 `PREFILL.md` 明确固定 `qhfan/FlashPrefillv2@75b58f2ecdba1c269a87dd34d8f1ae57bef50c57`，并引用其 `test_mean_correction.py` 及 V2 论文。因此旧设计确实意图移植 V2；不是仅凭名称猜来源。本次直接核对了本地引用和论文，未成功读取该固定作者 GitHub 快照，不能声称逐行比较了作者实现。

V2 第 3.2 节的 mean surrogate 与 exact blocks 共用归一化分母，按实际块 token 数加权。`alpha` 是相对最大能量的阈值，不是固定稀疏比例；proxy 是近似，不能宣称 dense 等价。v1 的基础方法与 V2 的 correction 要分别做消融，名称也不能与 FlexPrefill 混用。

**已知限制：** 旧 policy 有 partial fragment 强制 exact、容量预检等可保留合同；完整路径仍依赖新增 GGML op、wire、owner stamp 和 barrier。错误分支的输出处理不一致，pool 有写 0 而注释称 NaN 的分支。flags/fixture/脚本不证明真实模型消费了 sparse 输出；CUDA/Hopper 优化也不证明 AMD/Vulkan 净收益。

**重写边界与最低条件：** B 拥有 policy、checked layout、选择参考和 exact/proxy 分区；C 提供真实阶段、logical prompt 边界与 A 能力，必要 kernel seam 单独登记。一个 exact token 加两枚相同 token 的 proxy block 应验证 count=2；partial causal block 不得把不可用 pooled 值带入阈值；sink/window/tail 重叠只能计一次；小 ubatch 不能把 logical tail 错当每次调用尾部。真实运行记录实际 selected/proxy rows，再比较 TTFT、质量、峰值与完成率。decode、draft/verify、embedding 的支持范围分别定义。

**相邻状态与计数：** `RMC23/RMC24`，`src/llama-flashprefill-state.cpp` 与 metrics：scope、process nonce、context serial、adapter generation、policy 身份以及实际执行计数有用。旧 envelope 的 host-endian memcpy 与 LE 注释不一致；同 context 的临时身份也不是持久跨进程模型身份。等真实 sparse C 消费时，由 C 持执行身份，按明确字节格式重写保存/恢复与计数。

### P02 — TriAttention 校准、评分与压力策略

**消费者与价值：** 长 Decode 的 KV 超预算时，保留较有价值的 KV，验证可服务长度与质量的取舍。

**归档入口：** `RMC45、small:S20、server:K08/K11`；`src/llama-triattention.cpp` 的 calibration loader、`triattention_precompute_head_derived`、`triattention_invert_rope`、`triattention_score_keys` 和 pressure/dedup helpers；`ggml/src/ggml-cuda/triattention-score.cu`。

**原论文：** [R15] 的正确题名为 *TriAttention: Efficient Long Reasoning with Trigonometric KV Compression*。第 4 节区分 pre-RoPE Q 统计、trig/norm 评分、未来 offsets 平均、每 query head 的 z-score 后 max 聚合以及窗口触发。旧首注释题名和部分式号不准确，应按原文重建。旧 3/32 ratio、tail guard、recurrent-only pressure、RERoT dedup 是项目工程策略，不能全部归因论文。

**已知限制：** 旧“pure oracle”仍暴露 ggml/backend 状态；空 offsets 可导致非法读取/除法；位置转 float 的范围未定义；tail guard 可大于逻辑 token 数；同物理 cell 的冲突位置没有完整身份校验。MoE 的 WXQCAL03 统计不能代替此处 pre-RoPE Q 统计，普通 layer 名称也不能替代 logical/physical layer 映射。

**重写边界与最低条件：** B 只持完整验证的 calibration、数值、position 和评分结果；C 负责模型/tokenizer/adapter/RoPE/层身份与真实剪除，A 保持共享引用和失败行为。单频解析例、零均值/恒定 scores、空 offsets、NaN、GQA 尺度相差很大、half/interleaved/partial RoPE、大位置均验。重复 physical cell 去重且拒绝冲突身份；recurrent-only 压力不应无谓删 attention KV。最后必须真正触发 pruning，量实际保留/峰值，在多个长推理任务上测质量、完成率和 wall time。

### P03 — xKV 跨层低秩与选择重建

**消费者与价值：** 在目标模型确有跨层冗余时压缩长 Prefill 的 KV；潜力高，实现与验证成本也高。

**归档入口：** `src/llama-xkv-factor.*` 的 `factorize_matrix/factorize_kv/evaluate_quantized_shadow`；`llama-xkv-canonical.*`、`llama-xkv-landmark.*`、`llama-xkv-reader.*`、`llama-xkv-seal.*`；Vulkan xkv shaders；`XKV-SR.md`、`docs/xkv/compression_gate.json`。

**现有证据不能放行：** 上述旧 gate 实际为 `compression_goal_met=false`、`evidence_provided=false`，**PASS=0、FAIL=0、NOT_EVALUATED=16**。这是未评估，不是 16 项失败，也不是成功。静态审计另记 reconstruct layer-slot 上界、64-bit position low-word 丢失、非一致 barrier 分支、status 覆盖/未消费、失败 refresh 先改 stamp 等缺口。本次未运行旧模型来复现这些路径。

**原论文：** 按 xKV v2 [R16] 的跨层因子、head-specific selective reconstruction、landmark 选择与成本定义重写。其量化结合试验使用 RTN；第 7 节只压缩初始 context，生成 tokens 保持未压缩。旧四路 Turbo 因子、量化 landmarks、滚动 seal 与 RERoT 组合均是额外假说，不属于论文已经证明的完整方法。

**重写边界与次序：** 独立 F64 小矩阵 → 真实 KV 的 full-rank/no-quant → 同预算单层/跨层 → dense reconstruct → SR=all 等价 → rank/选择误差分开 → 因子量化 → 设备执行。B 接 owned 矩阵/格式/计划；C 从真实 A 状态取样并调用，A 必需的生命周期/存储 seam 逐项证明必要。

**最低条件：** 同 input/factor/seed/codec 的 CPU/设备对照；量化后按最终乘积误差验收；计算 factors、热区、landmarks、索引、workspace、snapshot pins、host 副本和峰值，确认 full dense backing 真正释放。与当前实际 Turbo KV 作同 survivor/source、同质量和资源预算对照，不能重复计算已有 16→4 bit 收益。长上下文、长生成、任务质量、完成率与端到端时间全部留证；paper speedup 和名义 MiB 不算本仓库收益。

### P04 — Codec 的格式身份、数值域与按行有界读取

**消费者与价值：** 新压缩矩阵确需跨阶段/文件传递时，保留 owned config、logical/padded shape、orientation、role、canonical/rotated domain 与版本。

**归档入口：** `FBE06/FBE07、RMC47`；`src/llama-xkv-codec.h` / `src/llama-xkv-codec.cpp` 的 `codec_desc`、`encode_matrix/decode_rows`、descriptor parser 与 factor compatibility；`ggml/src/ggml-turbo-quant.c`。

**已知限制：** 旧 descriptor parser 有长度正确但 data=null 的入口；新显式 group row API 不依赖共享 group，原兼容 row wrapper 仍依赖全局配置，不能据前者宣称全库可重入。私有 ggml type、byte-span 与格式兼容仍需独立定义。mixed Turbo/non-Turbo 不得无条件直接 rotated GEMM。

**来源与重写边界：** 格式、版本、长度和所有权是工程合同。TurboQuant [R10] 可作旋转量化数学参考，但论文不规定本 fork 的 ABI，也不保证 xKV 因子量化质量。A 已有 TurboQuant，不能创建 B 包装其私有类型冒充独立实现。

**最低条件：** 有实际消费者后，定义自己的稳定 wire bytes/golden files、严格枚举/端序/长度/溢出检查、支持 group、零/尾 padding、部分坏行、精确 scratch 边界、两实例与 input/result 所有权；由最终消费者验证兼容域，避免只做无业务意义的 round-trip。

### P05 — KV 相位平移、RoPE 与存储旋转域

**消费者与价值：** context shift、缓存重定位或稀疏位置操作需要正确的 key 表示。

**归档入口：** `RMC07、R14`；`src/llama-kv-transform.h` 的 Hadamard/shift/decode helper；`llama-kv-cache.cpp` 的 Turbo shift。

**已知限制：** 旧新 Turbo 分支位于既有 Turbo `continue` 后，静态不可达；有 host helper 不证明生产路径使用。unsigned normalized Hadamard 与 signed Turbo WHT 不等价，已有 WHT 修复不能给所有旋转域背书。

**原理与最低条件：** RoFormer [R05] 提供 RoPE 相对相位依据，存储量化/reader 集成仍须自证。明确 decode、移除存储旋转、相位变换、重新存储的顺序；验证 pairing、partial rotary、未旋转尾维、负/重复 delta、大位置、padding 和允许误差，不重复应用 YaRN 等已有幅度项。C/A 负责真实缓存，B 只提供独立数值参考；先查当前 A 的必要性。

### P06 — GGUF expert slab 与诚实的谱/失真分析

**消费者与价值：** 不启动完整模型即可检查某专家矩阵，筛选值得真正编码的方向。

**归档入口：** `wanxiangqi/lib/gguf-slab.cpp::{open,find,slab}`；`wanxiangqi/calib/calib-probe.cpp`；`wanxiangqi/lib/linalg.cpp`。

**已知限制：** expert/offset/size/reopen 与 exposed pointer 生命周期不完整；固定 sweep 求解没有收敛状态。旧 `delta(b)=1.9*2^(-2b)` 及 tail/失真预测是启发式，不能凭 LOW-RANK/IQ 标签推断可部署收益；已量化源权重不能冒充原始 FP16 谱。

**边界与最低条件：** GGUF/ggml 适配留 C，B 接 owned 数值和几何。小矩阵精确 F64、秩亏/病态/零矩阵、未收敛、实际 dequantized slab、分片/坏 expert/尾部均校验。OBC/GPTQ [R08]、[R09] 是重建目标参考，随机低秩可参考 Halko 等 [R11]；这些论文不为旧启发式提供收益保证。真实结论必须实际编码、计 bytes 和重建误差。

### P07 — QMoE 的完整编码/解码研究

**消费者与价值：** 目标现代 top-k MoE 确有可利用冗余时，探索真实 sub-bit 格式。

**归档入口：** `wanxiangqi/calib/qmoe-probe.cpp::{ternary_row,group_entropy}` 与 gate/up/down 对角加权误差。

**已知限制：** 旧工具只有三值 dead-zone、H1/H4/H8 熵与对角误差，没有完整 GPTQ 反馈、字典码流、随机访问或 device decoder。熵不是实际 bpw；已量化源权重也不能未经验证地作为原精度符号熵或码率的保守界。

**原论文与最低条件：** QMoE [R17] 是明确来源，论文包含专用压缩格式与 decoder。用明确源 dtype、固定真实模型及代表性校准人口做 RTN/论文方法消融，量出 dictionaries/scales/indices/padding 的全部字节；参考解码逐元素对照，再验证真实 expert matmul、held-out 质量与端到端收益。论文 SwitchTransformer 的结果不能直接外推到本项目目标模型。只有收益成立再承担新格式与 A seam 的维护成本。

### P08 — Shared basis、nonlinear student 与 GGUF ablation

**消费者与价值：** 低优先研究问题；在原始样本和可靠质量指标具备后才有继续价值。

**归档入口：** `wanxiangqi/probe/make-transforms.py`、`fit-test.py`、`wxq-bake.cpp::fitted::load`。

**已知限制：** 旧脚本缺 import，fitted geometry/完整性未严验；“proj512 失败”或 intrinsic-dimension 注释不足以构成固定实验证据。窄网络零 padding 回原 shape 不会自动减少文件或原 dense kernel 计算。

**来源与最低条件：** 所查文件未找到可核实的该特定 nonlinear student 原论文，记录为自有研究假说，不套 AWQ/GPTQ/xKV 名义。先有 identity/requantization control、按请求/来源隔离的 train/held-out、真实 sample ID、层输出与整模型质量；新 shape/codec 必须被实际 kernel 消费，真正存储/计算下降才称压缩。输出先完整验证再发布。

### P09 — 私有 PQ2_0 格式

**消费者与价值：** 低优先，只保留明确实验问题。

**归档入口：** `ggml/src/ggml-quants.c::{quantize_row_pq2_0_ref,dequantize_row_pq2_0,quantize_pq2_0}`，相关 backend、quantize 与 GGUF type remap。

**已知限制与来源：** 旧代码明确称其为 Prism-private、group-128 的 Q2_0 变体。它不是 QMoE，也不能凭 PQ 名称认定为 PolarQuant；本次未找到可确认的特定原论文。不能为 CLI 名称搬回私有 type ID、RPC 编号和 heuristic remap。

**重新考虑条件：** 在目标模型相同质量与资源预算下，实际 packed bytes、scalar/ISA/backend 对照、整模型质量和 decode 吞吐证明优于当前 A 的可用格式；否则维持文本。普通 Q2_0 的 x86 优化另列，避免绑在一起。

### P10 — Online softmax M/L/O 的独立数值参考

**消费者与价值：** split/fragment attention、exact/proxy 或 hot/cold 合并需要正确全局分母；首先是数学参考，不是独立推理加速承诺。

**归档入口：** `RMC25/RMC50`；`ggml/src/ggml-flashprefill.cpp` 的 `ggml_flashprefill_mlo_init/add/add_sink/merge/finalize`；Vulkan `flashprefill_merge.comp`。

**已知限制：** 旧 add/merge 在全部验证前已改部分 numerator，只检查 double 中间有限后转 float；空侧 merge 未完整验证另一侧状态。源码推导的待执行反例：两项同 logit、`v=FLT_MAX`，最终均值合法且仍为 `FLT_MAX`，但未归一化 float numerator 可溢出成 Inf，而 add 返回成功。本次没有执行该反例，不能写成实测失败。

**数学依据与最低条件：** FlashAttention [R18] 附录 B.3 的分块重标定可作参考；必须重新定义本库 O 是 numerator 还是归一化输出。B 拥有完整 accumulator，声明范围/累积类型/别名；失败不能留下仍可当成功消费的半状态。one-shot/不同 chunk/tree merge、empty identity、全 mask、sink 恰好一次、大动态范围、晚分量非法、无效状态遇空 merge 均验证。浮点合并顺序可有声明的容差；V2 proxy 的近似不能被“exact softmax”名称掩盖。

## 5. 不能被大功能名称掩盖的小项

以下条目同样只有文本资格。先确认当前 A/用户新实现仍有缺口和真实消费者，再决定是否重写。能由已有公开算子提供的行为直接复用；backend 内部问题不应通过装饰器掩盖所有权。

本节 `VK` 指归档 `ggml/src/ggml-vulkan/ggml-vulkan.cpp`，`KV` 指 `src/llama-kv-cache.cpp`；表中旧 ID 可在 [audit/files.json](audit/files.json) 找到原分册与详细范围。

### 5.1 后端与内存细节

| 本文定位 / 旧 ID | 有用目的与归档入口 | 证据缺口与未来最低验收 |
|---|---|---|
| K01 / G06 | Vulkan F16 SCALE；VK 的算子支持与 dispatch | 用当前公开图与 F32/F16 参考证明支持范围，确认 graph 实际进入该分支；不凭 supports_op 返回 true 晋级 |
| K02 / G07 | Norm/mul/MROPE/IMROPE/SET_ROWS 融合；VK fusion 识别及 RoPE shader | partial dim、alias、转置/stride 与 unfused 参考，真实模型路径及净收益；不整包搬旧模型图 |
| K03 / G10 | 同步 2D write/memset 与 compute queue 的顺序；VK | 定义公开同步语义，合法 caller 顺序下作 upload/readback 对照；不把未授权并发操作的失败当现有接口错误 |
| K04 / G12 | scratch 增长后携带实际记录 context，正确处理 retired context；VK | 强制真实 growth 和延迟完成，在公开路径验证所记录 stream/context 身份；已有 scratch-content 修复不覆盖所有生命周期 |
| K05 / G13 | root buffer / descriptor 对齐至少满足实际向量引用；VK | 真实低对齐设备、4-byte view offset、vec4/descriptor 使用分别验；root alignment 不能代替内部每块对齐 |
| K06 / G15、M24 | 零行 no-op、受检 byte span、无效 seq_div divisor；VK memmove 与 KV seq_div | 零/负/溢出/错 span 在修改前拒绝或按定义 no-op，guard 与原状态保持；避免只测 sizing helper |
| K07 / G16 | 用有界 fence wait 控制 host busy polling；VK | 测真实 CPU 占用、尾延迟与退出时限，确认没有隐藏 drain 延迟；不直接照搬 wait 常量 |
| K08 / G17 | free-memory 减法饱和、不下溢；VK budget 查询 | usage 超 budget、缺扩展、设备差异；driver budget 是动态进程用量指引，不能替换成 physical heap 冒充可分配预算 [R13] |
| K09 / G20 | 小列 MMVQ、staged 与 Q8/Q6 路由；VK 与对应 MMVQ shaders | 记录实际 route、shape/dtype 数值、整段耗时与切换边界；旧 8→18 等阈值不自动有效 |
| K10 / G21 | MoE 单专家/grouped singleton 路由；VK MoE dispatch | expert identity、ABI、shape/overflow、scatter/reorder 全成本；单 kernel 变快不代表请求变快 |
| K11 / G22、RMC28 | 所有 finite additive mask 的稳定 compact；`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.comp` 与 `.glsl` | 保留原 mask bits/index/order，只排除 -Inf；NaN/+Inf/capacity 溢出必须真实消费 dense fallback。例：两项 QK=0、mask=-65504、V=(1,3)，输出应为 2，不能把 finite 负值都删掉。再测高/低密度净收益 |
| K12 / G23、RMC30 | FA 输出 gate、pair-L2 融合；VK 与 attention shader | alias、输出 lifetime、unfused 数值及实际消费；不复用别的 fusion 测试背书 |
| K13 / G31 | CUDA 少量 query 的 Turbo native VEC 路径，减少 F16 临时转换；`ggml/src/ggml-cuda/` attention route | 需要真正 CUDA 设备，覆盖 query 边界、kernel 命中、分配峰值、数值和耗时；当前没有本项 CUDA 资格 |
| K14 / G32、FBE05 | 已有 Q2_0 × Q8_0 的 x86 VNNI；`ggml/src/ggml-cpu/arch/x86/` | ISA 检测/链接分派/fallback、scalar 参考与目标 shape 速度；与私有 PQ2_0 严格分开 |
| K15 / FBE01 | CPU per-thread workspace padding；`ggml/src/ggml-cpu/ggml-cpu.cpp` 与 `ops.cpp` 的 SET_ROWS/SOFTMAX/ROPE | awkward width、多线程、实际 consumer stride、ASan；正确字节公式不证明每线程消费者用了相同布局 |
| K16 / FBE16 | Meta CPU host-view 避免错误绕过 copy/virtual buffer；`ggml/src/ggml-backend-meta.cpp` | 当前公开普通消费者、copy ops、host alias、virtual buffer；此前非零 offset 测试没有覆盖此独立路径 |
| K17 / M22 | 按实例 adaptive V 类型；KV/配置 resolver | 旧 per-instance 路径可能只接了日志，旧 global 仍在；验证两个实例真正的 buffer type、kernel 与质量，不能只看配置打印 |
| K18 / M34 | raw indexer K 不作 RoPE shift、Turbo 请求时 indexer 用 F16；`src/llama-memory-hybrid-idx.cpp` | 必须真实支持的 hybrid-indexer 模型与 shift/状态操作；普通 dense 模型无覆盖能力。M29 的原 QSA ext 恢复 A 已有，不计为新修复 |
| K19 / M35–M37 | pooled indexer K、dirty watermark 与唯一占用；indexer KV 与 graph | 所有相关序列失效、cross-seq owner、restore/reuse、shared bytes 去重；不能让 headroom 仅按子缓存 getter 合计 |
| K20 / RMC08 | 按行 pread、去重和有界 worker；`src/llama-lazy-reader.h` 及 loader 接线 | 先有独立于旧 TP5 的真实消费者；0 workers、受检 offset、EINTR/短读、worker 异常与完成集合；测试手工注册不证明生产已接通 |

### 5.2 模型、格式与调度接线补充

这十项是小型需求或前文条目的具体消费者，不代表十个已经证实遗漏的模型功能。

| 本文定位 / 旧 ID | 归档路径与有用需求 | 重新实现/复用边界与最低条件 |
|---|---|---|
| D01 / small:S02 | `src/models/bailingmoe3.cpp::load_arch_tensors`：`ssm_f_a/ssm_g_a` 与旧 `ssm_f/ssm_g` 名称兼容 | 优先 C 离线 GGUF 名称规范化，必要时窄 A 兼容；新旧命名、同时存在冲突、shape 错、全缺失和真实 checkpoint 输出均验 |
| D02 / small:S02/S13 | BailingMoE3 与 `src/models/qwen4exp.cpp::load_arch_hparams`：辨认剥离 NextN 后仍完整的 target trunk | 不能按缺一张 `nextn.eh_proj.weight` 就静默修坏模型；验证已知导出版本、完整 trunk 和真实尾部数量。`TENSOR_SKIP` 不证明 MTP 工作 |
| D03 / small:S13/S21 | `qwen4exp.cpp::load_arch_hparams`：PLE u64 offsets/vocab 进入 I32 gather 前受检窄化 | 零 vocab、单项/和超过 INT32_MAX、长度不匹配均拒绝；multiplier 保持 u64。它是范围验证，不是新 hash 算法 |
| D04 / small:S08 | `src/llama-model-loader.cpp`：大 PLE 表可选 CPU placement，用户 override 优先 | 优先 C 使用已公开 tensor buffer override/profile；证明实际 tensor 匹配、输出不变并分别测 RAM/VRAM/延迟；与未接通 lazy reader 分开 |
| D05 / M28/M29 | `src/llama-kv-cells.h`、KV、`qwen4exp.cpp::llm_graph_input_ple::set_input`：PLE 前驱 token 与扩展状态身份 | 稀疏位置、多序列、不同 batch、EOS/图像、copy/remove/restore 须定义，先评估显式 token history。原 QSA x/y 恢复 A 已有；旧 PLE 扩展在 `has_cell_ext=true`、`n_pos_per_embd<=1` 时存在写未 resize 的 exts 风险，属静态待验证反例，不迁入 |
| D06 / small:S13/S21 | `qwen4exp.cpp::{graph::build_qsa_top_k,llm_graph_input_qsa::set_input,can_reuse}`：全覆盖退化与大 top-k fallback | 全覆盖仍写 raw indexer K、保持相同 sparse attention/mask；覆盖→部分选择、尾块、restore watermark、graph reuse。旧 `argsort_top_k` 是无条件替换；旧“Vulkan k≤1024”注释不是当前阈值证据 |
| D07 / RMC31 | `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_turbo_cm2.glsl`：Turbo3 50-byte 块的 2-byte 引用对齐和 stored-WHT 坐标 | 核对当前 block ABI，奇数 block/view offset、CM2 实际 route 与数值；不能为了几行 alignment 搬整个未验证 kernel |
| D08 / small:S20 | `src/models/nanbeige.cpp`、`wanxiangqi/calib/triattention-config.h`：逻辑执行层与共享物理 block 校准映射 | 归入 P02 输入合同：旧 44 logical/22 physical 的目标不能按物理权重去重统计；真实几何、每次逻辑执行、full-attention 层覆盖、pre-RoPE 观察点独立验 |
| D09 / G39 | `src/llama-kv-cache-dsv4.cpp`：HCA/LID 与 CSA 接收一致的显式 stream 参数 | 参数转发遗漏的候选；不是新 phase arena/统一池。分别检查各实际子缓存的配置、live/reserved/peak；DSA/MSA 只转发 base getter 不能代表 indexer 总字节 |
| D10 / server:S04/S05、M01/R05 | `tools/server/server.cpp`、`server-context.cpp`、`src/llama-context.cpp`：初始 `n_parallel_pp=1` 和容量参数的真实接线 | 归入 E10：优先、并发上限、reserve 形状分开；`server_n_outputs_max` 的普通生成/embedding 角色公式 A 原有，直接复用。只检查实际参数传播与 R05 的后续赋值覆盖风险；不把新增 RERoT pens 容量当普通功能重写 |

### 5.3 证据工具本身的剩余需求

| 定位 | 目的与归档来源 | 未来需要的实际结果 |
|---|---|---|
| V01 | run 与 source/model/calibration/loaded-library/request 身份绑定；旧 `docs/xkv/`、质量 gate、[脚本审计](audit/audit-script-content-findings.md) | 可追溯的原始输入与产物，不接受“非空字符串 + 自报 real”作为硬件证明。PASS、FAIL、NOT_EVALUATED、NOT_EXERCISED 保持不同 |
| V02 | 同工作量、完成率与正确配对统计；旧 capacity/paired/http benchmark，见同分册 | 真并发且所有请求结算后再判定；wall clock 与分母一致；小样本区间/异常值策略正确，报告质量与完成率，不能只挑快样本 |
| V03 | 唯一资源的 live/reserved/peak 与 padding/host 副本核算；旧 memory/XKV/fragment 报告 | 按真实 allocation identity 去重；相同物理前缀或 pins 不重复记收益，实际释放与名义容量分开 |
| V04 | 严格数值与评测集完整性；[测试审计](audit/audit-test-content-findings.md) 和 [reader 反例](audit/reader-counterexamples-20261018.json) | NaN、1e309、null、missing、错答案、缺行不漏报；train/dev/blind 按来源/题目族隔离。mock HTTP 和纯 parser 测试不冒充真实模型/设备资格 |

新 TP5 probe 已有自己的严格 evidence validator。本表保留其他候选需要的通用事实，不要求再复制一套 TP5 验证器。任何小工具重写都应修正已知反例后验证真实 CLI 输入，不能机械搬旧脚本。

## 6. 降级、纠错与不再重复立项

| 项目 | 本次判断与理由 |
|---|---|
| 旧 `small:S03` “新增 uint64 saver / PLE hash” | **纠正旧审计结论。** `src/llama-model-saver.cpp` 的旧侧已支持 `uint64_t → GGUF_TYPE_UINT64` 且已有 `vector<uint64_t>` 实例化；fork 又插入同样分支。PLE multiplier 数组及 hash 中间值也原有 u64。不要迁入重复代码；真正窄化检查见 D03 |
| `model_compatibility=7` | 是总账的 7 个测试路径分类，不是 7 个模型新功能。BPE 加 include、resolution 加构造、下载错误改 skip 不能计为模型兼容收益；fixture 也不是模型运行证据 |
| 旧 M29 “QSA indexer mirror ext 恢复” | 对照 old-side，saved ext 在 apply_ubatch 后回填已属于 A；fork 新增的是 PLE 适用范围。恢复循环不能当新修复重复迁入，相关状态格式与条件一致的容量要求归 D05 |
| target/draft 跨 context 共享 allocator | 保留节省重复 workspace 的目的；旧初始化被普通初始化覆盖、KEEP_CHUNKS 不 trim、缺 fence、失败 grow 留旧 generation。只有新 TP5/普通双 context 的实测重复成本值得时才重写真正 owner/同步协议 |
| Meta graph-view cache-key（FBE15） | 当前 A 已有非零图 UID cache，但 `ggml_graph_view()` 的 view UID 为 0，view 因而重建；旧 fork 的“view 继承父 UID”前提不在。若以后改变 view 复用，父图/切片身份、生命周期与收益一起验；见 [Meta 接缝说明](tests/a-seams/meta-transfers-protocol.md) |
| Meta pinned-readback retire（FBE12） | 当前 Meta 路径没有旧 pending transaction。不能孤立迁入 retire 修补；以后如建立新读回路径，保活、完成、可见性和实际收益一起验 |
| Turbo SET_ROWS 拒绝 F16 source | 固定 A 已有相应 shader 支持，旧拒绝前提不成立；现有相关回归保留原资格，不重复立项 |
| 旧 eager recurrent clear、既有 speculative aliases/chained/adaptive MTP、CUDA phase arena | A 已有对应能力，或已有 clean 交付；相邻未证实行为按 E11/E18–E20/K18 等另列 |
| arbitrary allocation override、第二轮量化清理、日志与未测常量 | 无明确当前消费者/净收益，不为它们引入跨层维护负担；保留归档可回查 |
| router row split、BF16 router、MMVQ+MTP 联合选择器 | 延续 [MIGRATION.md](MIGRATION.md) 的具体历史取舍，不把旧测量扩大到所有 router/MMVQ/MTP 方法 |
| 旧硬件 recovery/reset、sudoers/安装脚本 | 不是推理功能或论文复现依赖，不迁、不运行；需要时按独立运维任务处理 |
| 旧实验报告/二进制工件 | 未重新检验全部 `artifacts/`、`reports/`、`wanxiangqi/qwen35-prune/` 输出；不因存在文件就补成功结论，也不删归档来隐去范围 |

旧分册保留历史审阅的原貌；上述纠正以固定 diff 与完整源文件为依据，供以后回查。它们不改变源分母，也不把“读到了一个差异”变成一项必须重写的功能。

## 7. 原始来源索引

论文用于确定数学/算法及其原实验范围；它们不证明本仓库旧实现工作。工程接口以当前固定 A 和实际平台规范为准。下表记录本次使用的具体版本或正式出版入口，查阅日为 2026-10-19。GPTQ、OBC、QMoE 和随机矩阵分解条目在本次主要核对书目与方法范围；未来逐式重写仍须阅读全文、固定参考实现及差异。其他正文引用的具体章节已按所链接版本阅读。

| 来源 | 论文/规范与固定入口 | 在本文中的作用 |
|---|---|---|
| [R01] | *SGLang: Efficient Execution of Structured Language Model Programs*，arXiv:2312.07104v2 | radix cache、LRU、active reference 与路由参考；不是旧 slot affinity 的出处证明 |
| [R02] | *Efficient Memory Management for Large Language Model Serving with PagedAttention*，arXiv:2309.06180v1 | block ownership、COW、preemption/swap/recompute 参考；不提供本 server 的 sampler/SSE 恢复合同 |
| [R03] | *Fast Inference from Transformers via Speculative Decoding*，PMLR 202（2023） | 分布保持与拒绝采样基础；不承诺任意实现的同 seed 轨迹相同 |
| [R04] | *DFlash: Block Diffusion for Flash Speculative Decoding*，arXiv:2602.06036v1 | hidden-conditioned block draft 方法；不为设备交接 ABI 背书 |
| [R05] | *RoFormer: Enhanced Transformer with Rotary Position Embedding*，arXiv:2104.09864v5 | RoPE 数学；不是 RERoT、Turbo storage 或所有 rope extension 的规范 |
| [R06] | *FlashPrefill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast Long-Context Prefilling*，arXiv:2603.06199v1 | v1 的模式/选块/阈值方法 |
| [R07] | *FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving*，arXiv:2608.19758v1 | V2 mean correction 与执行边界；硬件移植另证 |
| [R08] | *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*，arXiv:2210.17323v1 | 近似二阶权重量化原始方法；不是旧 collector 的完整复现声明 |
| [R09] | *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning*，arXiv:2208.11580v1 | 校准驱动的二阶重建/剪枝参考 |
| [R10] | *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*，arXiv:2504.19874v1 | 旋转量化、重建 MSE 与内积目标的区别；不定义私有 wire format |
| [R11] | *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions*，arXiv:0909.4061v2 | 随机低秩算法参考；不是旧谱启发式的收益证明 |
| [R12] | Vulkan `vkDestroyBuffer` 官方规范 | owning device 与在途引用完成后销毁 |
| [R13] | Vulkan `VkPhysicalDeviceMemoryBudgetPropertiesEXT` 官方规范 | budget/usage 的动态估计含义 |
| [R14] | Vulkan 官方 *Synchronization Examples* | device/host 顺序与可见性参考 |
| [R15] | *TriAttention: Efficient Long Reasoning with Trigonometric KV Compression*，arXiv:2604.04921v1；另核对 [PMLR 正式页](https://proceedings.mlr.press/v306/mao26e.html) | pre-RoPE 校准与评分、GQA 归一化、窗口剪除；私有压力策略另列 |
| [R16] | *xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction*，arXiv:2503.18893v2；另核对 [PMLR 正式页](https://proceedings.mlr.press/v306/chang26d.html) | 跨层低秩/SR、成本、RTN 结合和初始 context 限制 |
| [R17] | *QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models*，arXiv:2310.16795v1 | 完整格式与 decoder 依据；旧熵 probe 不能代替 |
| [R18] | *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，arXiv:2205.14135v2 | 分块 online softmax 数学参考；独立 accumulator 合同仍需定义 |
| [R19] | *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*，arXiv:2607.05147v1 | draft 架构与 confidence 调度原论文；不与 DFlash 混名 |

[R01]: https://arxiv.org/html/2312.07104v2
[R02]: https://arxiv.org/html/2309.06180v1
[R03]: https://proceedings.mlr.press/v202/leviathan23a.html
[R04]: https://arxiv.org/html/2602.06036v1
[R05]: https://arxiv.org/html/2104.09864v5
[R06]: https://arxiv.org/html/2603.06199v1
[R07]: https://arxiv.org/html/2608.19758v1
[R08]: https://arxiv.org/abs/2210.17323v1
[R09]: https://arxiv.org/abs/2208.11580v1
[R10]: https://arxiv.org/html/2504.19874v1
[R11]: https://arxiv.org/abs/0909.4061v2
[R12]: https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyBuffer.html
[R13]: https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html
[R14]: https://docs.vulkan.org/guide/latest/synchronization_examples.html
[R15]: https://arxiv.org/html/2604.04921v1
[R16]: https://arxiv.org/html/2503.18893v2
[R17]: https://arxiv.org/abs/2310.16795v1
[R18]: https://arxiv.org/html/2205.14135v2
[R19]: https://arxiv.org/html/2607.05147v1

## 8. 回查与未来提交

可从固定旧提交只读查看来源，无需把旧分支合并回来：

~~~sh
git show 2777543b2358c88cd1c2000f7fa342b764a0e213:wanxiangqi/calib/trie-calib.cpp
git show 2777543b2358c88cd1c2000f7fa342b764a0e213:PREFILL.md
git show 2777543b2358c88cd1c2000f7fa342b764a0e213:docs/xkv/compression_gate.json
git diff --no-renames cda0f9256665845cd9d6f3a0176880883f653a3d \
  2777543b2358c88cd1c2000f7fa342b764a0e213 -- src/llama-model-saver.cpp
~~~

以后选择一个条目时，先记录目标用户负载、当前 A 是否已具备该行为、所用论文版本/自研假说、具体输入域和未支持范围。参考数值、实际消费者和可证伪的验收闭合后，再提交独立 B/C 或最小 A 例外；A 例外继续记录内容摘要、消费者与去除条件。不以源码行数、私有运行时恢复程度或候选数量作为完成指标。

本次交付为文档审阅与纠错。没有新模型/GPU/性能运行结果，没有改写 [VALIDATION.md](VALIDATION.md) 的历史资格，也没有把尚未执行的验收要求转成 PASS。
