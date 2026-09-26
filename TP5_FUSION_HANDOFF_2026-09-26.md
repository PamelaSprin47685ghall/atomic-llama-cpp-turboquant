# TP5 MTP 融合版：性能与状态交接

日期：2026-09-26

## 0. 结论与当前停点

目标仍是五张 RX 6800 的 TP5 MTP 请求达到 `predicted_ms < 1710`（171 token，>100 Decode tok/s）。**尚未达到**。

当前已验证、可复现的融合配置在 83 tok/s 左右；它不是 100 tok/s，也不应被表述为达标版本。

用户要求停止编译、运行和继续实现，以供外部性能审阅。本文件冻结当前事实、证据、分支和未解决问题。

## 1. 正确的性能基座

此前错误是把新二进制的 78–79 tok/s 单次 smoke 当作“最佳”。这是错误口径。

历史已验证 canonical 基座：

- 旧 binary SHA：`19034e...`
- 源码基：`b4e87a01` 的 dirty 工作树
- 配置：`native-mtp-kv-only`
- exact `1..60`，prompt `31`，`completion_tokens=predicted_n=171`，自然 `stop`
- `predicted_ms=2065.730`
- **82.779 Decode tok/s**
- 证据：`/var/tmp/tp5-mtp-100-20260925/post-withdrawal-smoke.json`

另一条同旧 binary 的无 profile 实测：

- **83.184 Decode tok/s**
- 证据：`/var/tmp/tp5-mtp-100-20260924/current-profile-smoke.json`

旧 binary 本体未保留；已用可保留的 K/V integration patch 从 `b4e87a01` 重建源状态。

## 2. 已验证融合源与正式 ABBA

重建 worktree：`/var/tmp/tp5-fusion-82-wt`

重建方法：

1. 从 `b4e87a01` 创建 detached worktree。
2. 应用 `/var/tmp/tp5-mtp-100-20260924/kv-reference-fixed.patch`。
3. 恢复 hierarchical ARGMAX 两阶段 shader；修正了 source clean build 中 stage1/stage2 shader 文件与 pipeline descriptor 数不对应的问题。
4. 只叠加 durable sampler checkpoint；不叠加 strict GPU candidate、host-hidden chain、device feedback/direct loop、direct GDN snapshot、CPU compact sampler 或其他已拒绝路线。

融合成员：

- TP5 RELAY/F32 + linear lowering
- W1 rank-local pinned readback
- 单卡 MTP draft、n=6
- hierarchical ARGMAX
- small-Q8 columns
- non-replicated attention
- QSA/GDN headmap
- K/V-only catch-up
- durable sampler checkpoint：复制 grammar、reasoning budget、sampler/RNG、accepted-token history；不复制 decode-local `cur` 候选数组；speculative rollback 使用 checkpoint clone。

五块独立进程 ABBA（A1 B1 B2 A2，20 个进程）：

| 项目 | 基座 A | 融合 B |
|---|---:|---:|
| Decode mean | 83.45 tok/s | 83.41 tok/s |
| stdev | 0.30 | 0.44 |
| B - A | \-0.04 tok/s | |
| 95% CI | [\-0.68, +0.60] tok/s | |
| B / A | 0.9995 | |
| 95% ratio CI | [0.9919, 1.0072] | |
| non-inferiority lower bound | 0.9919 >= 0.99 | |

结论：checkpoint 在完整请求上没有可测的独立正收益，但满足预先定义的非劣性门，因此保留为状态安全、无可测回退的融合成员；不得称其单独提升吞吐。

完整证据：`/var/tmp/tp5-mtp-100-20260925/fusion-checkpoint-abba5.json`

单次融合 smoke：`2058.120 ms / 83.09 tok/s`，`/var/tmp/tp5-mtp-100-20260925/fusion-base-checkpoint-smoke.json`。

## 3. 无调优参数默认路径

融合 worktree 已将合格 MTP 配置变成默认：

- 无需 `--spec-draft-device Vulkan0`
- 无需 `--spec-draft-n-max 6`
- 无需 TP5/MTP 调优环境变量
- 仍必须提供模型/模式身份：`--tp5 qwen4exp-af -md <mtp.gguf> --spec-type draft-mtp`

默认启用或自动选择：RELAY/F32、linear lowering、单卡 draft、n=6、non-replicated attention、QSA/GDN headmap、K/V-only、small-Q8、graphics queue、device-local VRAM、MMVQ/producer-wire disable、10M spin。

显式退选包括：

- `GGML_MTP_KV_ONLY=0`
- `GGML_TP5_REPLICATE_ATTN=1`
- `GGML_TP5_QSA_HEADMAP=0`
- `GGML_TP5_GDN_HEADMAP=0`
- `GGML_VK_SMALL_Q8_COLUMNS=0`
- `GGML_TP5_LINEAR_LOWERING=0`
- `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=0`
- `GGML_VK_ALLOW_GRAPHICS_QUEUE=0`
- `GGML_TP5_MTP_DRAFT_SINGLE_GPU=0`（恢复 target-rank 草稿布局）
- `--spec-draft-n-max N`（显式 horizon 覆盖）

默认路径 canonical smoke：exact `1..60`、prompt31、171/171、自然 stop，`2076.556 ms / 82.35 Decode tok/s`。

证据：`/var/tmp/tp5-mtp-100-20260925/fusion-default-smoke.json`

此 smoke 只证明默认解析和功能；不要把单次 82.35 与 ABBA 均值作性能优劣判断。

## 4. 当前分支、提交与远端状态

已创建且已测试的融合提交：

- 分支：`work/mtp-fused-83`
- commit：`16e28b8ac tp5: fuse validated MTP runtime profile`
- tag：`tp5-fused-83-20260926`
- worktree：`/var/tmp/tp5-fusion-82-wt`

远端 `origin/master` 已在基座之后推进：

- `origin/master`: `9e4926382 RERoT: parallel-aspect thinking wire, multi-lane decode fixes, Vulkan matvec reuse`

已建立本地整合 worktree：

- 分支：`work/mtp-fused-master`
- worktree：`/var/tmp/tp5-fused-master-wt`
- 内容：`origin/master` 与 `work/mtp-fused-83` 无冲突 merge
- **未验证、未 push**。

用户要求停止后，`build-integration` 编译任务已取消。不得把 `work/mtp-fused-master` 推送到 master，除非重新完成整合构建、精确 canonical 和必要的性能/回归验证。

当前原工作目录仍是 `work/mtp-100`，含较晚的 candidate/ledger/default 实验；它曾产生 78–79 tok/s 的重编译 smoke，不是融合性能基座。

## 5. 已通过的融合 worktree 验证

在 `/var/tmp/tp5-fusion-82-wt/build-fusion`：

- `test-sampling`
- `test-arg-parser`
- `test-mtp-workspace`
- `test-tp5-plan`
- `test-vulkan-command-replay --argmax-only`
- exact canonical fusion smoke
- 五块 ABBA
- parameter-free exact canonical smoke

ARGMAX clean shader build 也实际通过；此前当前源码的 stage 文件名/descriptor mapping 问题会令全新 shader 生成失败，融合提交中已修正。

## 6. 已拒绝或不应直接重启的路线

| 路线 | 结论/证据 |
|---|---|
| strict GPU greedy top-1 candidate | audit-free pilot 比 feature-off 慢 22.425 ms；没有全路径收益前提。 |
| host-hidden unrolled MTP chain | 已修复正确性，仍为 3488.112 ms / 49.02 tok/s；普通 n=2 为 3019.149 ms / 56.64 tok/s。 |
| CPU stateless greedy / stateful top-k compact | canonical 正确但 pilot 回退。 |
| device token feedback / external token fence / staged relay draft | 已归档或退选；未显示净收益。 |
| direct GDN snapshot injection | 状态语义不安全，退选。 |
| GDN target-capacity | fail-closed；inactive row、KV、rollback 合约未证明。 |
| n7/n8/n10/n12、rank split、草稿五卡、Q2/Q3 扫描 | 不作为主线；已有负结果或驻留风险。 |
| device embedding placement | 有孤立高值 smoke，但没有独立正向资格，不能纳入融合。 |

## 7. 已知性能账与主要瓶颈

历史 171-token 分相诊断（有 profiler，不能当正式吞吐）：

- 总 2437.917 ms，25 MTP cycles。
- draft 514.051 ms；其中逐 token host sampler 外围 445.386 ms，GPU enqueue 67.454 ms。
- target graph 区间 1700.078 ms。
- target graph 外 90.072 ms。
- decode 后验收/采样 95.056 ms；其中 sample-and-accept 87.058 ms、clone 6.363 ms。
- catch-up 35.891 ms。

关键事实：445 ms 不是纯 CPU sampler；它含 GPU 等待。真正问题是每个草稿 token 都经 `llama_decode -> CPU sampler -> 下一 batch` 的控制依赖。

TP5 relay ledger：96 fallback relay stages；producer-ready cumulative 46.403 ms，CPU relay work 3.225 ms，prefix transfer 0.106 ms，最大 last-rank-ready 2.065 ms。结论是 rank-local producer readiness 为主，不是 CPU relay bookkeeping。

为 100 tok/s，仍需要主线结构改造而非更多 flag：

1. 将六步草稿纳入 TP5 execution program，消除六次高层 CPU sampler/decode 往返。
2. 设备端 local candidate/global candidate 发布，保留 deterministic tie/NaN/词表分片语义；不要实现六 token 并行因果错误。
3. target 七行验证减少全词表 D2H 与候选物化；仅对严格语义允许的 sampler 路径。
4. GDN rollback 从全前缀 snapshot 走 checkpoint + 精确重算；逐拒绝位置、卷积历史、KV 有效范围逐项对拍。
5. K/V-only catch-up 进一步列出实际 dispatch，确认可去除无消费者工作；不许因 accepted 就直接删 catch-up。
6. rank-local output slots：全部 rank copy 入队后统一消费，不能只删 synchronize。

## 8. 关键源码与证据索引

融合修改：

- `common/sampling.{cpp,h}`
- `tools/server/server-context.cpp`
- `common/common.{cpp,h}`、`common/arg.cpp`
- `src/llama-context.cpp`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/{argmax_stage1,argmax_stage2}.comp`
- `scripts/run-tp5-cross-matrix.py`
- `TP5.md`

历史主审计与路线图：`AGENTS.md`、`TP5.md`。

特别重要的 artifacts：

- `/var/tmp/tp5-mtp-100-20260925/post-withdrawal-smoke.json`
- `/var/tmp/tp5-mtp-100-20260924/current-profile-smoke.json`
- `/var/tmp/tp5-mtp-100-20260925/fusion-base-checkpoint-smoke.json`
- `/var/tmp/tp5-mtp-100-20260925/fusion-checkpoint-abba5.json`
- `/var/tmp/tp5-mtp-100-20260925/fusion-default-smoke.json`
- `/var/tmp/tp5-mtp-100-20260924/kv-full-behavior-state-comparison.json`
- `/var/tmp/tp5-mtp-100-20260924/kv-canonical-state-comparison.json`
- `artifacts/tp5-mtp-readback-20260924/`
- `/var/tmp/tp5-mtp-100-20260924/kv-reference-fixed.patch`

## 9. 审阅时必须保持的验收规则

正式性能：独占五卡、watchdog active、参考 F32/RELAY、不开 profiler、模型和 workload 不变、逐字正确、自然 stop、171/171、server `predicted_ms < 1710`。报告全部预先安排样本；不从波动中挑最快值。

状态改动必须覆盖：全接受、每个拒绝位置、EOG、长度截断、取消后下一请求、1->7->1、零输出 catch-up、不等长词表、tie、NaN/Inf、grammar、penalty、JSON、reasoning budget 与非 MTP fallback。
