# 实验证据与清理规则

当前性能入口是 [PERFORMANCE.md](../PERFORMANCE.md)，不是某个旧 run 的 decision。

## 保留什么

当前源码资格所需的原始输出、身份、失败反例和最小复现保留。历史实验保留紧凑的 decision、JSON/TSV 统计与有辨识力的失败原因；不同日期的结论不能混成同一条最终成绩。

`mmvq-mtp/run-20260928T1700/takeover-qualified-search-20261007.proof.json` 及对应包仍有近期来源价值。本次只读取了索引，没有重新解包验证；不要把索引中的 verified 字段当成本次验证。10-08 工作区外的原始确认记录本次未能读取，保留这一缺口。

`tp5-next-20260926/g7-realistic-workload/weightloss-variants.json` 仍由 `scripts/run-tp5-cross-matrix.py` 读取，不能作为旧垃圾删掉。RERoT、readback 等其他特性的证据也不能只因不在本轮主线就删除。

## 清理什么

2026-10-08 整理清理已经被后续工作取代的逐 stage 大日志、旧构建日志、硬编码历史 binary 的一次性实验 runner 和重复交接手册。历史 decision 中指向这些日志的路径仅保留为当时来源记录，不保证文件仍存在，不再从这些路径执行旧程序。

删除的 ignored/untracked 日志不在 Git 历史中，不能声称可通过 Git 恢复。紧凑决策和当前原始证据优先保留；不为了腾空间修改既有结论或擦掉失败原因。具体本轮操作与未清理范围见 [CLEANUP-20261008.md](CLEANUP-20261008.md)。

## 新实验如何落盘

一轮只建一个有日期的目录。保存 manifest、原始请求、机器可读汇总、短 decision；注明控制/候选、质量门、计时范围、warmup、独立块、源码和实际 DSO 身份。不要把各阶段同一份源树和动态库反复压包。

decision 应写：观察、证据级别、适用范围、失败/排除原因、是否独立确认、下一项可证伪动作。过期的运行脚本移入受维护 scripts/ 并去掉个人路径、补测试，或者删除；不要让 archive 成为另一套生产工具目录。
