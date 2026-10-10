# G4/G4a：verify 窗口 host 间隙归因 + verify-argmax 后端采样——奖金实测、架构墙关闭（2026-09-26，run-20260926T2310）

## 归因（G4，已完成）

G2 遗留的 ~9.1 ms/cycle host 间隙（target_us 74.5 − 链 GPU span 65.4，19 cycles ≈ 185 ms/请求）用新增诊断子计时器
（`GGML_TP5_VERIFY_SUBTIMERS`，仅插桩、不作吞吐声明）拆解，真机 4 请求全部 171/171 精确，稳态 71 cycle：

|成分|每 cycle|每请求|来源|
|---|---:|---:|---|
|llama_decode（含链 span 64.6 + 提交侧 ~2.5 ms）|67.1 ms|—|链内已归因项|
|**common_sampler_sample（9 行 CPU 全词表采样）**|**4.36 ms**|**~83 ms**|`set_logits` 拷贝整行 248320 vocab + CPU 链 argmax，484 µs/行|
|9×993 KB F32 logits 回读|~0.75 ms|~14 ms|`needs_raw_logits` 对无后端采样序列强制|

**根因**：speculative slot 结构性禁用后端采样——server-context.cpp `backend_sampling &= !(slot.can_speculate())`
（多输出/序列限制，llama-context.cpp:3927）→ 每行 verify 采样走 CPU 全词表物化。
**~97 ms/请求（5%）是 G1/G2 从未拆出的新实测项**，满足 V1 重启门"新账证明存在可省的关键路径"
（AGENTS §2.2）；且消除的正是"全量物化"，不新增同步。

## G4a 候选（已实现、机制命中、被架构墙挡住）

设计：opt-in `GGML_TP5_SPEC_VERIFY_ARGMAX`——资格检查
`common_sampler_is_greedy_argmax_eligible`（逐元素证明链输出 == 逐行 argmax：temp≤0、
`?*` 空占位、top-k/top-p/min-p 任意值保 rank-1、typical/sigma/dry/xtc/penalties 仅惰性默认）
通过时，后端对 `t_logits` 做多行 `ggml_argmax`，跳过全词表回读与 CPU 采样；
RNG 契约用 `llama_sampler_dist_draw`（新 API）在每次 `common_sampler_sample` 后端快路按行推进
mt19937，保持与 CPU 路径逐位一致的流（rollback/checkpoint 语义同源）。

实现（候选树 `/var/tmp/tp5-candidate-k1a`，未提交）：llama.h、llama-context.{h,cpp}
（verify_argmax 成员 + set API + 多行检查跳过 + needs_raw_logits 扩展 + t_sampled_rows 回读）、
llama-cparams.h、llama-graph.{h,cpp}（build_sampling 多行 argmax 分支）、llama-sampler.cpp
（dist_draw API）、common/sampling.{h,cpp}（资格检查 + RNG 对齐）、server-context.cpp（启用门）。
调试中修复三个资格检查缺陷：`?penalties` 空占位名（disabled 采样器以 `?` 前缀空采样器入链）、
`top-k` 连字符、`temp-ext` 静态分支（delta≤0 时等价 temp）。机制确认命中：
`[verify-argmax-elig] eligible=1` + slot 日志。

**撞墙（真机首请求断言）**：`ggml_argmax` 作用在词表维五卡切分的 `t_logits` 上，
meta 切分规划器 `handle_per_row` 断言 axis-0 切分不支持（ggml-backend-meta.cpp:882
`GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`）。
跨 rank argmax 归并（stage1 每 rank 局部 + stage2 全局合并）在本代码库**不存在**：
现有 hierarchical ARGMAX（argmax_stage1/stage2 shader）是单卡 draft 路径用的；
五卡 verify 的全局归并正是现在由 host 全词表回读完成的——即我们要消除的那条路径本身。

## 裁决

- **G4a 关闭**："已实测奖金、被 meta 规划器架构墙挡住"。崩溃路径已编译期禁用
  （`static const bool spec_verify_argmax = false;` + 注释记录原因与奖金数），
  env 置 1 时默认路径 smoke 精确 171/171、acceptance 1.000、无回归。
- 保留物：`GGML_TP5_VERIFY_SUBTIMERS` 诊断子计时器（env 门控，默认零开销）；
  资格检查与 RNG 对齐 API 作为 G4b 的已验证构件。
- **G4b（未立项，重启条件）**：跨 rank argmax 集体归并——stage1 每 rank 对词表分片出
  `(max_logit, global_idx)` 对（8 字节），经既有 RELAY star 或专用 stage2 shader 合并出全局
  argmax。这是新架构候选，须按手册 §3 单独过门（数值资格：tie/NaN/Inf/空分片按 reference；
  与 host 归并逐位对拍；正确性矩阵 §11.1）。若成，兑现 ~97 ms/请求且顺带消除
  `needs_raw_logits` 回读（~14 ms），是当前唯一实测支撑的大额可收项。
- 缺口现状不变：1940.8 ms vs 1710 门，231 ms。

## 证据

- 归因：`g4-verify-gap/run-20260926T2300/`（diag-run.py、server-diag.log 含 84 条
  `[tp5-verify-sub]`、diag.json 4 请求精确）
- 候选：`g4a-verify-argmax/run-20260926T2310/`（g4a-pilot.py、debug-elig.py、
  server-debug.log 机制命中与 eligible=1、server-B.log 断言崩溃、
  pilot-stale-helperbug.json / pilot2-topk-bug.json 两次资格 bug 的无效 pilot、
  server-smoke.log 默认路径回归通过）
- 源码：候选树 diff（六文件 + 注释），未 commit/push
