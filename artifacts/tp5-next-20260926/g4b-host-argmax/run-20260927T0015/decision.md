# G4b：verify-argmax host 采样——正式 ABBA 晋级（2026-09-27，run-20260927T0015）

## 改了什么

单因素候选（相对 n8 融合基座）：`GGML_TP5_SPEC_VERIFY_ARGMAX=1` 时，对通过资格检查的贪心
speculative verify slot，`common_sampler_sample` 跳过每行 248320 词表的 `cur[]` 物化 + CPU 链
遍历，直接对已回读的 logits 行做裸 argmax 返回。

- **资格检查** `common_sampler_is_greedy_argmax_eligible`（G4a 遗产，已修三处缺陷）：
  逐元素证明链输出 == 逐行 argmax——temp≤0（含 temp-ext 静态分支）、`?` 空占位采样器、
  top-k/top-p/min-p 任意参数（rank-1 恒存活）、typical/sigma 仅惰性默认、
  penalties/dry/xtc 仅禁用默认；grammar/budget/logit-bias/mirostat/未知一律拒绝。
- **比较语义逐位相同**：裸扫描用 strict `>`（首最大胜，NaN 比较恒 false 不可能胜）——与
  `llama_sampler_temp_impl` 的塌缩扫描完全一致；资格证明保证无 masker 动 rank-1。
- **RNG 流逐位一致**：`llama_sampler_dist_draw`（新 API）每次采样调用推进一次
  mt19937（`std::uniform_real_distribution<double>`），与 CPU dist 路径同序同次；
  rollback/checkpoint 恢复同一克隆态（clone_impl 已传播标志）。
- **门**：`n_probs==0`（否则 populate_token_probs 会读未填充的候选数组）+
  `can_speculate()` + 非 backend_sampling + env opt-in。
- **G4a 清理**：多行 `ggml_argmax` 图路径、`t_sampled_rows` 回读、`needs_raw_logits`
  三参变体、ctx 级 `llama_set_verify_argmax` 机制全部移除（G4a 撞 meta 规划器
  axis-0 断言的崩溃源）；保留资格检查 + dist_draw + 采样器级标志。

## 正式 5-block ABBA（harness `run-tp5-cross-matrix.py`，A=n8 vs B=n8-varargmax）

- 20 trials × 2 repeats = 40 请求，**全部 exact 171/171、finish=stop、predicted_n=171**。
- 同一二进制 SHA b92fda129…（A/B 仅 env 差异），5 block 全部完成。

|指标|A (n8)|B (n8+varargmax)|B−A|95% CI|裁决|
|---|---:|---:|---:|---|---|
|committed tok/s（配对块均值）|52.62|53.49|+0.87|[0.53, 1.20]|**CI 全正，显著正收益**|
|decode tok/s|88.13|90.39|+2.26|[1.58, 2.95]|**CI 全正**|
|B/A ratio (committed)|—|—|1.0165|[1.0101, 1.0228]|非劣 0.99 门通过且为显著收益|
|steady predicted_ms|1940.3|1886.4|−53.8 ms|—|与 pilot −60.2 一致|

B 最快块 1868.0 ms（**91.6 tok/s**），A 最快 1907.2。B 全部 10 个 steady 样本
(1868–1906) 低于 A 全部样本 (1907–1978)——**分布完全分离，零重叠**。

## 裁决

- **晋级**：G4b 通过正式资格（完整行为 + 机制命中 + 整请求显著改善 + 预注册配对统计）。
  −53.8 ms/请求（−2.8%），91.6 tok/s 峰值。
- 与 pilot (−60.2) 一致；差异在会话噪声内。
- **不宣称 100 tok/s**：1940.8→1886.4，距 1710 门还差 ~176 ms。
- 剩余缺口成分：relay 等待（LateBind 拒）、draft 401 ms（D1 互斥）、prompt ~230 ms
  （workload）、G4b 后残余 host 采样 ~0.5 ms/cycle（裸 argmax 扫描本身）+
  logits 回读 ~14 ms/请求（G4c 集体归并才可消除）。
- **G4c（未立项）**：跨 rank argmax 集体归并（stage1 8 字节对 + star 合并）可再收
  ~14 ms 回读 + 消除裸扫描；重启门见 G4a decision.md。

## 证据

- `abba-g4b.json`（5 block 完整、stats 双指标、40 请求 exact、DSO SHA、GPU snapshot）
- `pilot.json`（−60.2 ms pilot）、`server-debug.log`（机制命中 eligible=1）
- `abba-g4b_logs/`（20 份 server 日志）
- 源码：候选树 `/var/tmp/tp5-candidate-k1a`（common/sampling.{h,cpp}、
  llama-sampler.cpp、server-context.cpp；G4a 设备路径已移除），未 commit/push
