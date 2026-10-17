# q2-contract-20261012 契约（候选稿）

## 这是什么

本目录是 Q2_0 自然问句契约的**新版本候选**。`question.txt` 与旧轮
`q2-baseline-20261010T0401/question.txt` 逐字节一致（`如何减肥` + 换行）；
`system.txt` 是新版本系统提示。本目录文本为候选稿，最终裁定权在 Manager。

## 与旧版的差异及动机

- 旧版引导「简洁回答，约200个token，不展开长篇解释」，简洁要求与长度目标
  互相冲突，2026-10-10 实测产生 140 个 completion tokens，低于预注册
  160–240 长度门，该轮仅 DIAGNOSTIC_ONLY。
- 新版改为结构化长度引导：覆盖四个方面、每方面两到三条具体要点、
  篇幅以约两百个模型 token 为目标，使自然 EOS 输出预期落在 160–240 区间。
- 新版只做长度与结构引导，不预存任何具体答案文本或关键词清单
  （AGENTS.md 红线 6）。

## 规矩

1. **修改提示意味着开始新的完整 A/B 轮次**：所有臂必须用同一新版本重测，
   禁止挑某个方案生成更短/更长的特例。
2. **旧 140-token 轮次原样保留，不放宽**：`q2-baseline-20261010T0401/`
   及其 DIAGNOSTIC_ONLY 结论不得回头改成合格。
3. 若新版本本身仍达不到 160–240 token 长度门，标记该契约版本失败，
   不得在优化器里选择能改变 token 数的数值路径来规避长度资格。
4. 若 Manager 修改本目录文本，须以新契约版本号另立目录，不得覆盖本目录。
5. 正式测量前由 DevOps 计算 `question.txt`、`system.txt` 的 SHA256 并冻结
   登记；本文件不预写哈希值，未运行不声称身份。

## 长度门的机械执行

`scripts/tp5-gap-bench.py` 自本契约起支持 `--length-min/--length-max`
（默认 160/240）：仅 measured 相位请求的公开 completion tokens 参与判定；
不在区间内时 `result.json` 顶层 status 保持 `MEASURED_NOT_QUALIFIED` 并附
`length_gate` 审计对象（区间、每请求 n、是否通过）。warmup/guard 请求不计入
判定，但其 n 仍记录。数据一律保留不删除。

MTP 周期账字段（`accepted_prefix_distribution`、`cycle_D_V_S_H_K`）在
horizon>0 时尽力从保留的 `[tp5-mtp-cycle]` 日志行与 timings 计数器提取，
无证据时写 `NOT_COLLECTED`，不编造数值；horizon=0 写 `NOT_APPLICABLE`。
rank/stage 账（`rank_stage_ledger`）当前无采集器，固定写 `NOT_COLLECTED`，
未来采集入口为 `GGML_TP5_PROFILE=1` server.log 的 rank/stage 时间线解析。

## 相关入口

- 旧轮证据：`artifacts/perf-gap/q2-baseline-20261010T0401/`（含 DECISION.md）
- 规程：`AGENTS.md` §2（任务 Ω）、§4.5（契约冻结）、§5（runner 与最低字段）、
  §7.2（周期账）、§7.3（rank/stage 账）
