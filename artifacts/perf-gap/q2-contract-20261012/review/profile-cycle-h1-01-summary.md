# GAP 4：h1 profile 诊断轮周期账摘要（profile-cycle-h1-01）

诊断轮，不进任何正式排名（status=DIAGNOSTIC_NOT_RANKED）。冻结 measured-bin、h1 配置、--profile cycle、warmup 1 + repeats 2。

## 周期账（§7.2 口径）

- 周期数 184（2 请求 × 160 token；K 合计 318 = 2×160 − 2 首 token 来自 prefill）
- 总周期 ms = 6332.0：**V（verify）= 5801.2（91.6%）、D（draft）= 445.3（7.0%）、S（rollback/catch-up）= 85.5（1.4%）、H（其他 host）= 0.0**
- 每公开 token 成本 = **19.912 ms/token**；MTP 盈利门：h1 19.912 vs h0 ~20.41 → **盈利（约 2.4%）**
- K 分布：K=1 × 50、K=2 × 134 → 接受率 72.8%（与正式轮 67/92=0.728 一致）
- V_ms：p50 25.9、p95 28.8、min 24.2、max 285.8；D_ms p50 2.41；S_ms p50 0.48
- **异常周期 4 个**（V_ms > 50ms，~10 倍于 p50）：cycle[0] 272.2、cycle[89] 264.8、cycle[92] 265.4/285.8——每请求首周期（图缓存冷启动）+ 中段两个（疑似图重编译/缓存失效），单周期影响 ~0.5% 总时长
- accepted_prefix 新增字段：K_per_cycle 逐周期 0/1 序列 + histogram {1:134, 0:50}——比 timings 回退的 per-request 汇总更细

## rank/stage 账（§7.3）覆盖度

| 字段 | 值 |
|---|---|
| rank_stage_ledger | NOT_COLLECTED |
| actual_pipeline_hits | NOT_COLLECTED |
| active_rows | NOT_COLLECTED |
| capacity_rows | NOT_COLLECTED |
| bank_epoch | NOT_COLLECTED |

rank/stage 账采集器仍不存在（契约 README 已声明：未来入口为 GGML_TP5_PROFILE=1 server.log 的 rank/stage 时间线解析）——本诊断轮确认该缺口为真实 NOT_RUN，非数据丢失。

## 其他

- 诊断口径 pooled_tps = 50.387（插桩扰动下仍接近正式轮 51.67；不进排名）
- 2×160 token 自然 stop、零 fallback、退出 0
