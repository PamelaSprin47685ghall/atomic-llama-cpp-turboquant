# Q2_0 契约 v2：长度门修复 + cycle 账 + U 侧收敛轮

预注册：2026-10-10（GPU 运行前写入）。目录：`artifacts/perf-gap/frontier-20261010T000000Z/`。

## 目标

上一轮（q2-baseline-20260100T0401）参考臂输出 140 token，低于预注册 160–240 区间，只有 DIAGNOSTIC 资格。本轮：

1. 用新 system 提示（约 250 token 目标）建立契约 v2，重跑原 Q2_0 h0 参考臂，取得长度合格的 U 侧样本。
2. 同一冻结二进制跑 `--profile cycle` 诊断臂（独立目录，不进正式排名），取得 cycle 账证据。
3. 若长度门通过，按 cycle 账选择最大可因果解释的暴露成本项，实施单因素优化并复测。

## 固定项

- 源模型：原 Q2_0 两分片（SHA256 已于本轮重新核实：69820c02…、316b46f3…，与 AGENTS.md 记录一致）。draft 文件存在但不启用（h0）。
- 程序：`../q2-baseline-20261010T0401/measured-bin/llama-server`（冻结快照，binary-identity.json 哈希已核对）。
- 请求：`contract/question.txt`（如何减肥）+ `contract/system.txt`（新版本）。temperature=0、seed=42、thinking=false、cache_prompt=false、max_tokens=400（仅防失控）。
- 长度门（预注册）：160–240 completion tokens、finish_reason=stop。区间外保留原样、不晋级。
- CPU 22 线程、ctx1024、batch/ubatch32、-np 1、tp5 placement、显式环境（REPLICATE_ATTN=0、QSA_HEADMAP=1、GDN_HEADMAP=1、SMALL_Q8_COLUMNS=1、GREEDY_HOST=1、PARALLEL_SUBMIT=1、Q_WORKGROUP=256、NORM_WORKGROUP=256、NORM_PREFETCH=0、PACK_SINGLE_WG=0、SHARD_ROUTER=0）。
- 每进程：1 冷 warmup + 2 measured + 1 guard（哈希表问题）。warmup/guard 不进排名。
- GPU 串行：runner 内部取 /var/tmp/tp5-gpu.lock，外层不加锁。不并行编译。

## 臂

- `A2-q2-h0`：契约 v2 参考臂，profile none，warmup 1、repeats 3。
- `D2-q2-h0-cycle`：同二进制同契约，`--profile cycle`，warmup 1、repeats 2。DIAGNOSTIC_NOT_RANKED，仅取 cycle/阶段账。
- 若 A2 长度合格且 cycle 账显示单一主导暴露项：另立工单，单因素改动，重跑同契约 pilot。

## 判据

- A2 全部 measured 请求 160–240 token、stop、正文切题流畅 → U 侧样本合格，S_lo 更新。
- 任何机械失败（timeout、GPU fault、HTTP 500、非零退出）→ 保留失败，该臂不晋级。
- 修改提示后所有臂必须用同一新契约；不挑选生成更短/更长的数值路径。

## 理论侧（已完成，同目录）

`necessity-certificate-v4.json` + `necessity-status.json`：v3 的 476 tok/s 单一数字不满足 MTP 摊销审计；v4 拆分为 L_global（R4 绑定，5667 tok/s）与 L_nomtp（R3 绑定，730 tok/s，仅无 MTP 类）。串行链 R5 调查结论：完全放松集合下无可证延迟耦合项，收敛杠杆在 U 侧。测试 `tests/test-tp5-necessity-certificate.py` 14 项通过。
