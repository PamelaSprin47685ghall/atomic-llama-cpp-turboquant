# 功能迁移与取舍

## 固定身份

- A：AtomicBot-ai/atomic-llama-cpp-turboquant，master，cda0f9256665845cd9d6f3a0176880883f653a3d，发布标记 b10298-2.0.0。
- 旧 fork：2777543b2358c88cd1c2000f7fa342b764a0e213，保存在 archive/pre-clean-20261018。
- 本次检查日期：2026-10-18。远端 master 与所固定提交一致；远端默认 HEAD 指向另一条 dev。
- 旧 fork 相对该上游多 527 个提交。核心代码、工具和测试范围有 547 个变更文件、291,504 行新增、5,093 行删除；不包含大量实验产物。

这些数字说明审阅规模，不决定功能价值。下面的旧性能数据是历史文档记载，没有重新运行当时的五卡实验，不能直接用于宣称新分支速度。

## 已迁移

| 功能 | 来源 | 本次保留的行为与边界 |
|---|---|---|
| compact top-k | 旧 common/sampling-unique-argmax.h；PERFORMANCE.md；artifacts/perf-gap/run-20261009T0145/REPORT.md | 数值选择进入独立 TQSampling；C 运行 A 的原采样链和真实模型，保留随机状态。原报告 64.77 → 67.15 tok/s，4 个配对块约 +3.66%，95% CI [2.64%, 4.69%]，仅对应当时特定配置。 |
| Wanxiangqi 请求 token trie | 旧 wanxiangqi/lib/trie.h、trie.cpp；wanxiangqi/calib/trie-calib.cpp | 独立 TQCorpus 负责格式、去重、请求权重和不变量；C 使用真实 llama tokenizer、decode 与序列快照实现前缀评分。保留 V1 两文件格式，增加显式小端解码和输入验证。 |

原 trie 的公开可变节点、叶数组改为拥有数据的返回值，内部状态由 B 唯一修改。未把原 FetchContent tokenizer、服务端钩子、MoE activation 采集和诊断程序整包迁入。

## 因收益不足而舍弃本次补丁

| 项目 | 已有证据 | 决定 |
|---|---|---|
| router row split | PERFORMANCE.md 记录同一 pilot 66.6914 → 61.8741 tok/s，约 -7.22%，未晋级 | 不迁移该实验补丁。 |
| BF16 router | PERFORMANCE.md：8 配对块、16 进程，约 +1.14%，95% CI [0.79%, 1.48%]，低于原 2% 晋级门槛 | 不为此保留跨模型与后端的修改。 |
| MMVQ + MTP 的旧联合实验组合 | MMVQ+MTP.md 2026-10-07 节：131 个配置，最终候选等于原基线；4 配对块 C/R=1.034315，95% CI [0.996803, 1.073239]，包含 1 | 不迁移实验选择器、强制模式和对应控制复杂度。这个结论不证明整个 MMVQ 或 MTP 家族无用；上游已有实现继续保留。 |
| 旧诊断、硬件修复建议、实验输出与失败路径开关 | artifacts、临时 probe、TP5 专项脚本等 | 留在归档，按需要读取证据，不参与默认构建，也不把旧性能报告作为当前资格。 |

## 有价值或证据未闭合，暂未迁移

| 项目 | 为什么没有直接移植 | 重新进入干净分支的条件 |
|---|---|---|
| TP5 / LateBind / relay、device loop、tensor-parallel 私有扩展 | 历史另一次会话有 59.66 → 65.17 tok/s 的明显收益。实现横跨 ggml-vulkan、模型图、buffer 身份、epoch、rank、fence 和生命周期。不能因耦合成本就称其无用。 | 先复核合格真实负载的配对收益；提取能独立成立的数值/计划责任。列出 A 必须提供的最小公开语义；必要时将小范围 A 接缝作为单独提交，不能暗中搬用私有源码或符号。 |
| RERoT | RERoT.md 要求同一个 llama_context、同一 KV owner、动态逐 reader 可见性、共享 PUBLIC KV 和 lane-local recurrent/RNG。普通多请求或序列快照重放不具备这些语义。 | 上游提供足够的公开可见性/调度协议，或接受一个具有独立测试的最小 A 接缝；真实多 reader 行为与隔离探针同时验收。 |
| FlashPrefill / TriAttention | PREFILL.md 有设备和局部正确性记录；整模型质量与吞吐资格仍未闭合。修改深入 graph、cache 和 kernel。 | 用固定真实模型、上下文和质量门证明收益，再从实际职责寻找可拆边界。局部设备测试不能代替整模型验收。 |
| XKV-SR | XKV-SR.md 的局部库/CPU/GPU 检查不能证明大模型质量、有效 KV 内存和端到端速度；部分目标硬件证据缺失。 | 真实 decode 的 payload、驻留内存、质量与吞吐同时通过；明确状态所有者和 A 的公开合作协议。 |
| DFlash / DSpark direct-hidden 交接 | e88601100、baaa9b16d、b8426460d 等提交有内部隐藏层、设备交接及 probe。旧任务中的 15–40% 是目标，未找到足以为本次迁移背书的独立配对模型结果。 | 验证真实 target-hidden → draft 消费、数值与生命周期，再量化相对合格无该优化路径的收益。不得把最终 embedding 或重复推理当成同一功能。 |
| 旧模型专用和 speculative/server 协作修改 | 部分修改与以上特性、fork 自有参数或旧合并修复相互依赖，整包 cherry-pick 会重建原冲突面。 | 对每个要保留的用户行为重新给出一个可复现用例和最小差异，再决定由 C、独立 B 或明确的 A 接缝实现。 |

这几项是“留在归档、未迁移”，不是已经完成的解耦。尤其 TP5 和 RERoT 不能通过给普通生成程序换名字来替代。

## 仍由上游直接提供

上游 cda0f9256 已有的模型（包括 qwen4exp、bailingmoe3）、量化格式、后端、server、bench 和 speculative 能力保留原样。原 fork 最新的 HIP llama-bench 修复涉及自有 TriAttention 守卫和重复自有字段，不能把它当作无条件适用于干净上游的通用修复。

构建本目录不会给原上游 server 自动注入 compact sampling；使用该优化的实际入口是 llama-compose，或将 TQCompose 安装包用于另外一个明确的编排程序。

## 查阅旧证据

在任意共享此仓库 Git 对象的 worktree 中执行：

~~~sh
git show archive/pre-clean-20261018:PERFORMANCE.md
git show archive/pre-clean-20261018:RERoT.md
git show archive/pre-clean-20261018:MMVQ+MTP.md
git show archive/pre-clean-20261018:wanxiangqi/lib/trie.cpp
~~~

旧分支与原始文件保留，使这些选择可复核、可撤回；本次没有删除用户模型、实验资料或旧工作区。
