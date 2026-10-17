# q2-contract-20261012 保护集（guard-set）

## 身份

- 所属契约：`../contract/`（contract_version v2，主问题「如何减肥」）。
- 起草日期：2026-10-12（与 v2 契约同日）；**正式冻结时刻以 DevOps 完成
  SHA256 登记为准**。`manifest.json` 中 hash 字段现为 `PENDING_DEVOPS_FREEZE`，
  未运行不声称身份。
- 题数：10 题，每题一个 `guard-NN.txt`（UTF-8 纯文本，单题单文件），
  配一份 `guard-NN-quality.md` 人工评审判据。
- 本目录另含 `neutral-system.txt`：保护题运行时使用的中性 system 提示，
  不含任何题目相关内容与预期答案。

## 用途

本保护集服务于收束阶段（AGENTS.md §14 V8 / §18 F8）：在候选晋级前，
用**未参与选型的独立问题**验证模型通用能力是否保留。

规矩：

1. 正式揭示前，本目录全部内容（题目、判据、中性提示）**不作为任何优化
   输入**：不进入参数搜索、不用于提示工程、不喂给任何代理的调参或选型
   流程。揭示前的访问仅限冻结登记（DevOps 哈希计算）所需。
2. 揭示时机：V8 独立确认轮。评审者须为未参与候选选型与调参的人员，
   逐题按 `guard-NN-quality.md` 人工评审，给出 PASS / FAIL 并注明依据；
   回答原文原样留存，不因失败删除。
3. 保护题**不适用**主契约 160–240 completion token 长度门；每题的长度
   期望在其质量判据中按题型描述。运行时须将 runner 的
   `--length-min/--length-max` 设为宽区间（或不据其判定资格），避免把
   主契约专用长度门误加在短回答题上。
4. 修改任何题目或判据文本意味着新的保护集版本，须另立目录登记，
   不得原地覆盖。
5. 题目与判据均不含预期答案原文（AGENTS.md 红线 6）：需要数值或结论
   时，评审者依据题目条件自行推导核对。保护集验证的是能力保留，
   不是答案复现。

## 类别覆盖矩阵

| 题号 | 文件 | 类别 | 语言 | 期望长度 |
|---|---|---|---|---|
| 01 | guard-01.txt | zh-qa（自然现象解释） | zh | medium |
| 02 | guard-02.txt | zh-qa（历史文化） | zh | medium |
| 03 | guard-03.txt | en-qa（自然过程解释） | en | medium |
| 04 | guard-04.txt | en-qa（短问答） | en | short |
| 05 | guard-05.txt | short-answer（事实短答） | zh | short |
| 06 | guard-06.txt | code（编写函数） | zh | long |
| 07 | guard-07.txt | algorithm-explanation（算法解释） | zh | medium |
| 08 | guard-08.txt | arithmetic（多步应用题） | zh | short |
| 09 | guard-09.txt | logic（真话/假话推理） | zh | medium |
| 10 | guard-10.txt | context-comprehension（给定上下文推断） | zh | short |

对应 AGENTS.md §2 保护集类别要求：中文/英文（01–05）、不同长度
（04/05 为短，03/06/07 为长）、算术逻辑（08/09）、代码（06）、
上下文理解（10）。**跨请求状态与异常恢复类未覆盖**，原因与
future-work 规格见下文。

## 与 runner 现有 guard 的关系

- `scripts/tp5-gap-bench.py` 内置一道「哈希表与数组」guard，仅基础
  回归，不足以证明通用能力（AGENTS.md §4.5 明说）。
- 本保护集**不修改 runner**。逐题运行使用现有 `--prompt-file` 机制
  跑单请求：`--prompt-file guard-NN.txt`、
  `--system-prompt-file guard-set/neutral-system.txt`，采样参数与主
  契约一致（temperature=0、seed=42、enable_thinking=false、
  cache_prompt=false）。每题一次独立请求，输出按 AGENTS.md §5 最低
  字段留存。
- 主契约 `contract/system.txt` 是减重专属引导，**不得**用于保护题；
  保护题统一使用本目录 `neutral-system.txt`，保证题目间中性、公平。
- 保护题回答的质量判定是人工评审；`result.json` 的机械字段只保留
  原始记录，其 status 不代表质量结论。

## 未覆盖类别与 future-work 规格（只写规格，不实现）

跨请求状态与异常恢复类验证，runner 尚无序列化机制（无多请求上下文
编排、无故障注入入口），本保护集不含此类。future-work 规格：

1. **跨请求状态序列**：在同一服务进程内按预定义顺序发送多个请求
   （含相同 prompt 的重复请求与不同 prompt 的交替请求），在请求边界
   断言状态不串扰：相同采样参数下相同 prompt 的两次回答逐 token 一致
   （cache_prompt=false 契约下）；不同 prompt 的回答不出现彼此内容
   泄漏；KV/recurrent/carry/RNG 状态无跨请求残留。
2. **异常恢复序列**：在请求中途发送 SIGTERM 或断开连接，随后重启
   服务；断言重启后首个请求输出与无中断基线一致，且无状态漂移。
   退出信号路径按 AGENTS.md §16.2 处置：SIGTERM 后强杀记失败，
   不得写「正常退出」。
3. **实现前提**：runner 需增加请求序列编排与故障注入入口；该工作
   属于 runner/DevOps 职责，不属于本保护集文件。实现后应作为新的
   保护集版本另行冻结登记。

## 冻结与登记

- `manifest.json` 登记每题相对路径、类别标签与字符数（口径：UTF-16
  code units，本目录内容均为 BMP 字符、与码点数一致；含文件末尾换行）。
  char_count 为起草时统计，DevOps 冻结时应以实际文件复核。
- SHA256 由 DevOps 在正式冻结时计算并登记（回填 manifest 或另立冻结
  清单均可，但须与 `contract/contract-id.json` 同一登记纪律）；
  起草阶段一律不预写哈希值。

## 相关入口

- 所属契约：`../contract/`（README、contract-id.json、question.txt、system.txt）
- 规程：`AGENTS.md` §2（任务 Ω 与保护集类别）、§4.5（保护集冻结）、
  §14 V8（独立确认）、§18 F8（保护集配对与长测）
- runner：`scripts/tp5-gap-bench.py`（--prompt-file / --system-prompt-file /
  --length-min / --length-max）
