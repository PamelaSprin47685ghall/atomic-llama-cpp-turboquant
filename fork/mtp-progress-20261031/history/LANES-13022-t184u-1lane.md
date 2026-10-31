# 13022 t184u — 只留 1 路（更新 2026-10-30 02:42 JST）

Ashley：额度吃紧，只能 1 路。1/1 = **柱2 default-ON surface HIP** 在烧。

## 实核 1/1
| # | 车道 | 状态 |
|---|---|---|
| 1 | mtp-lb-default-on-hip-20261030（柱2 默认开 surface 对照） | **持锁**（双锁 0s 拿到，17:40:50Z = 02:40:50 JST），gpu.sh pid 2542249（kunweiz，setsid，自成 session），首腿 R_UNSET 在跑（run.py 2542264 / timeout 2542269 / compose 2542270）；fd8=/tmp/ple-candidate-gpu.lock fd9=$E/ple-candidate-gpu.lock 已核 |
| — | mtp-lb-promote-lctx-20261030 | 已完成 02:35 JST：PROMOTE-eligible — SWITCH PROVEN，BITWISE ALL，3 对 decode 均 ON 更快（均值 −28%，pair1 OFF 90ms/tok 离群；pair2/3 −3%/−15%）|
| — | mtp-lb-off0-on1-20261030 | 已完成 00:53 JST：SWITCH PROVEN，bitwise，ON ≈ −2.8% |
| — | mtp-lb-reuse-accept-20261029 | NO-PROMOTE（OFF 不是真 stock）；同配置禁止重烧 |
| — | mtp-union-skel-hip / mtp-serialcut-hip / mtp-fa-rtlen-hip | 已停；禁止再挂 |

路径：/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-lb-default-on-hip-20261030

## 环境矩阵（沿用 promote-lctx：sc bin + libs-v3f，prompts-long.txt，predict=128，windows=2，K=9，deadline 2400 / timeout 2700）
BASE = TP5_MTP_LATEBIND_DRAFT1=1 TP5_MTP_LB_COUNTERS=1 GGML_TP5_LB_SELFCHECK_REUSE=1 REUSE_ACCEPT=UNSET GGML_TP5_LATEBIND=UNSET GGML_TP5_RELAY_FRAMES=1 GGML_TP5_RELAY_COUNTERS=1
- R_UNSET = BASE（LATEBIND、RA 都不设 = 拟默认）——command.json 已确认 LATEBIND=None RA=None RSZ_RT=None
- R_ON    = BASE + GGML_TP5_LATEBIND=1 + TP5_MTP_LATEBIND_REUSE_ACCEPT=1
- R_OFF   = BASE + GGML_TP5_LATEBIND=0
- 顺序：R_UNSET → R_ON → R_OFF（1 组三腿；R_UNSET rc!=0 就提前停）。预计每腿 9–17 min，共约 30–45 min
- run.py 唯一改动：额外 pop GGML_TP5_RSZ_RT

## 门槛（patch/result.py 写进 RESULT.txt，末行 VERDICT）
- GATE1 R_UNSET LBSUM == R_ON 且 bind_accept/reuse_kept > 0（不能像 OFF 那样 bind_accept=0）
- GATE2 logits R_UNSET == R_ON 逐位一致
- GATE3 R_OFF 与 ON 仍 SWITCH 不同（LBSUM OFF≠ON，OFF bind_accept=0）
- GATE4 所有腿都没有 GGML_TP5_RSZ_RT
- ms/tok 只作信息，不作门槛

锁：$E/ple-candidate-gpu.lock (fd9) + /tmp/ple-candidate-gpu.lock (fd8)，均由 pid 2542249 持有。
注：~/work/ple-candidate-gpu.lock 是空文件（02:38 JST 探测时误建，不是真锁，无人用）。

规则：烧完本刀再换下一刀；禁止第2路。

## 2026-10-30 ~03:16 JST — default-on HIP DONE
- mtp-lb-default-on-hip-20261030: GATE1-4 PASS, VERDICT DEFAULT-ON SURFACE PROVEN
- 13022 1/1 FREE
- loot: /workspace/loot/mtp-lb-default-on-hip-20261030/RESULT.txt

## 2026-10-30 03:26 JST — 1/1 = mtp-dw-serialcut4-abon-20261030 BURNING（柱3 SERIALCUT4 OFF/ON 整解码 A/B）
路径：/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-dw-serialcut4-abon-20261030
- 新 lane（不是 fallback）。BUILD_OK 03:2x JST：build.py 沿用 serialcut-hip 流程，base = lb-selfcheck runtime.sc.cpp (e8294780) + loot dualwin-serial-seg-hipready-r8-010156 patch.diff 里 overlay runtime.cpp 那段（31 hunk 全部应用，#24 fuzz2），外加 dw-thru.h / dw-async-ts.h。compile 8.7s + link 0.7s。
  tq-tp5-compose-scut4 sha256 2bca566f…；bin 内有 GGML_TP5_DW_SERIALCUT4 和 "TP5_DW_SERIALCUT4 configured=1" 两个字符串。未编 mmvq.cu / fork / relay.hip（libs 沿用 v3f）
- 环境：scut4 bin + libs-v3f，prompts-long.txt，predict=128，windows=2，K=9，deadline 2400 / timeout 2700
  BASE = hdr.inc（LATEBIND=UNSET，RA=UNSET，RELAY_FRAMES=1，SELFCHECK_REUSE=1）
  ISO = TP5_MTP_SUBMIT_ONLY_GATE=0 TP5_MTP_ASYNC_PROTOCOL=1 TP5_MTP_GATE_METRICS=1
  OFF = BASE+ISO+GGML_TP5_DW_SERIALCUT4=0；ON = BASE+ISO+GGML_TP5_DW_SERIALCUT4=1
  run.py 会 pop 掉 RSZ_RT、DW_SERIALCUT(v1)、SERIALCUT_TRIM、SERIALCUT4（只有 extra 里给的才生效）
- 腿：OFF_1 ON_1 OFF_2 ON_2 OFF_3 ON_3（ON_1 rc!=0 就提前停），每腿约 9–17 min，共约 1–1.7 h
- 门槛（patch/result.py → RESULT.txt）：GATE1 全部 rc0；GATE2 logits OFF_i==ON_i 逐位一致；GATE3 ON ms/tok 比 OFF 好且超出 OFF/OFF 噪声带（否则判 NOISE 或 REGRESS）；GATE4 没有 RSZ_RT、SERIALCUT4 banner 与腿对得上、LATEBIND 未设；最后一行 VERDICT
- 03:26:27 JST 开烧：gpu.sh pid 2913405，leg OFF_1 compose pid 2913427；两把锁都 HELD
- 旧 mtp-serialcut-hip-20261029 原样不动（CANCEL 仍在）
规则：禁止第2路。

## 2026-10-30 ~04:25 JST — SERIALCUT4 DONE
- mtp-dw-serialcut4-abon-20261030: GATE1/2/4 PASS, GATE3 NOISE → VERDICT SERIALCUT4 SAFE, SPEED = NOISE。不要再烧同一个 A/B。

## 2026-10-30 04:44 JST — 1/1 = mtp-dw-hipspans-abon-20261030 BURNING（柱3 HIP_SPANS 设备重叠证明 OFF/ON A/B）
路径：/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-dw-hipspans-abon-20261030
- BUILD_OK（patch/build.py）：
  - bin hsp = serialcut4 lane src/runtime.scut4.cpp + thru-prove-hipready-r8-014429 里 3 处最小 hip_spans_req hunk（gate 常量 / 启动 banner `TP5_DW_HIP_SPANS enabled=1` / MAP 条件改为 meter||spans）。compile 8.6s。sha256 89a79a1a…
  - libs-hsp = libs-v3f，只换 libggml-hip：relay.v3f.hip + thru-prove relay.hip diff（trim 门 + OVERLAP_METER + HIP_SPANS producer；hunk7 destroy meter_flush_dump 手工插入；hunk2/3 fuzz2）。hipcc（clang++ gfx1030）在 13022 首次编过 relay.hip span 生产端，5.8s。hip sha256 3c65d3dd…
  - lane 偏差：span ring 4096→32768/rank（serialcut4 每 provider replay 约 7.8k 次，4096 会 dropped>0，joiner 会拒判）
  - 保真：重放同一 argv 编未改的 v3f relay，.text/.rodata/.data 与 libs-v3f 逐字节一致，只有 .hip_fatbin 差 266B（不是逐位复现，仅作参考）
  - joiner join/dw-hip-overlap-join 已编；自测 13/16 PASS，3 个 FAIL 都是缺外部 fixture /tmp/hip-fix1.log（patch 里没带）
- 环境：hsp bin + libs-hsp，prompts-long.txt，predict=128，windows=2，K=9，deadline 2400 / timeout 2700；BASE=hdr.inc（LATEBIND UNSET）；ISO = SUBMIT_ONLY_GATE=0 ASYNC_PROTOCOL=1 GATE_METRICS=1；SERIALCUT4 / OVERLAP_METER / RSZ_RT 由 run.py pop 掉
  OFF = GGML_TP5_DW_HIP_SPANS=0；ON = GGML_TP5_DW_HIP_SPANS=1
- 腿：OFF_1 ON_1 OFF_2 ON_2（ON_1 rc!=0 就提前停），预计结束约 05:25–05:55 JST
- 门槛（patch/result.py → RESULT.txt）：GATE1 全部 rc0；GATE2 logits OFF_i==ON_i 逐位一致；GATE3 ON：joiner rc0 + pair_scope=same_pci_shared_epoch + 至少一对 overlap_ns>0 + dropped=0 errors=0，OFF：没有 span，没有 pair 声明；GATE4 没有 RSZ_RT、HIP_SPANS banner 与腿对得上、LATEBIND/SERIALCUT4 未设；最后一行 VERDICT
- 04:44:29 JST 开烧：gpu.sh pid 3549175（fd8 /tmp lock + fd9 $E lock，两把都 HELD），leg OFF_1 compose pid 3549197
规则：禁止第2路。

## 2026-10-30 ~05:24 JST — HIP_SPANS DONE
- mtp-dw-hipspans-abon-20261030: VERDICT HIP_SPANS SAFE, DEVICE OVERLAP NOT PROVEN
- GATE1/2/4 PASS; GATE3 FAIL joiner parse `unknown key: enabled` (ON span_lines=75 but joinrc=1)
- loot: /workspace/loot/mtp-dw-hipspans-abon-20261030/RESULT.txt
- t188u: NO next flock 1/1; next = 12022 joiner fix then 13022 omp+flock re-prove
- 13022 1/1 FREE after burn

## 2026-10-30 06:05 JST — 1/1 = mtp-dw-hipspans-direct-abon-20261030 BURNING（柱3 GATE3 复验：直计量 HIP_SPANS OFF/ON）
路径：/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-dw-hipspans-direct-abon-20261030
- 根因：旧 lane 只有 RELAY_FRAMES=1、没有 KEY_V3 → relay 走 v2 capture/replay；meter_submit_begin 遇到 capture 就 ++skipped_mode（每 provider 7 次），约 7.8k 次 replay 完全没计量 → 修好的 joiner 拒判 skipped_mode=7
- 修法（只改环境，不重编）：ISO 加 GGML_TP5_KEY_V3=1 → ptrsafe_key → 强制 direct → capture=replay=0、skipped_mode=0。同一个 hsp bin（89a79a1a）+ libs-hsp（3c65d3dd）；joiner 换成修好的版本（39549c63，27/27）
- OFF = BASE+ISO(KEY_V3=1)+HIP_SPANS=0；ON = …+HIP_SPANS=1；腿 OFF_1 ON_1 OFF_2 ON_2；prompts-long，predict=128，windows=2，K=9
- 门槛：GATE3 新增「全部 aggregate skipped_mode=0 + relay capture=replay=0」；banner 不再算 bad_hdr；GATE4 要求 KEY_V3=1+FRAMES=1
- 06:05:48 JST 开烧：gpu.sh pid 19790（两把锁都 HELD），OFF_1 compose pid 19812；预计 06:50–07:15 JST 完成
- READY：/workspace/loot/mtp-dw-hipspans-direct-prep/READY.md
规则：禁止第2路。

## 2026-10-30 06:35 JST — 交接落盘（t192u/t193u）
- 完整交接（数学愿景→四柱→现状→规矩→下一步）：`/home/kunweiz/dispatch/HANDOFF-MATH-VISION-20261030.md`（副本：`$E/HANDOFF-MATH-VISION-20261030.md`）
- 附带源件：`/home/kunweiz/dispatch/HANDOFF-20261030-sources/`（waste ledger、ACCEPTANCE、READY、三刀 RESULT、rejoin SUMMARY、frontier/decision json、box 判定摘录）
- 1/1 仍为 mtp-dw-hipspans-direct-abon-20261030（gpu.sh 19790；OFF_1 rc0 947s；ON_1 自 06:21:38 JST 烧中；RESULT 未出，预计 ~07:10–07:20 JST）
规则：禁止第2路；本刀自然结束后勿自动续挂下一刀 HIP（t188u）。

## 11. 直计量 RESULT（2026-10-30 ~07:18 JST）

- **VERDICT：HIP_SPANS SAFE，DEVICE OVERLAP NOT PROVEN**
- 四腿 rc0；GATE1/2/4 PASS；logits OFF≡ON 逐位。
- `KEY_V3=1` 生效：skipped_mode=0，capture=replay=0，ON spans≈77995。
- **GATE3 FAIL**：joiner 报 `pci 0000:0f:00.0 maps to multiple windows`（map_lines=2，pairs_overlap=0）——根因已非旧的 skipped_mode=7。
- gpu.sh 19790 已自然结束，锁已释放；按 t188u **不自动续 flock 重烧**。
- 证据：`/workspace/loot/mtp-dw-hipspans-direct-abon-20261030/RESULT.txt`（本地）；13022 同 lane 及 `~/dispatch/HANDOFF-20261030-sources/RESULT-mtp-dw-hipspans-direct-abon-20261030.txt`。
- **下一刀**：修 joiner/MAP 的 PCI→window 唯一映射（先在 omp CPU 上做，不必立刻再烧 GPU）。

## 2026-10-30 — provider 身份链微诊断与中断恢复

- 私有车道 `/home/kunweiz/work/provider-private-build-20261030`：getter CPU fixture 20/20、joiner 62/62 和 audit 15/15；私有 base 与 runtime 的未修改对照重链分别逐位复现历史产物，HIP DSO 不变。
- 8×2 微诊断 OFF/ON rc0、logits bitwise；ON 的 provider MAP 与 span 归属一致，五卡出现同 PCI 共享时钟下窗口 enqueue→completion 区间交集。此结果不是 kernel 并行或吞吐证明。
- predict128/K9 四腿在主机重启中断：OFF_1 和 ON_1 rc0、logits bitwise；ON_1 joiner `mode=provider`、`same_pci_shared_epoch`，五卡有正区间交集；OFF_2 不完整，无四腿 RESULT。
- 恢复后第二对在 `/home/kunweiz/work/provider-recovery-20261030` 的 OFF_2 初始化 rc1：HIP 仅枚举四张卡，要求五张；ON_2 未启动。`rocm-smi` 和 `lspci` 均仅显示四张 RX 6800，根因未知。不得以四卡替代 TP5；五卡恢复前停止 GPU 重判。

## provider 第二对重试 — 未完成，暂停同配置

- 用户恢复五卡后，在 `/home/kunweiz/work/provider-recovery2-20261030` 持双锁补第二对；OFF_2 rc0，logits 与 OFF_1 逐位相同。ON_2 未完成，末行 `TP5_RETAINED_IN_FLIGHT`，rc 文件为空。
- 上次正常启动的内核日志在 ON_2 期间报告 `0000:03:00.0 device lost from bus`、SDMA reset 失败；随后发生重启。当前五卡可见不证明负载稳定，硬件或配置根因未定。
- 四腿 GATE3 仍未完成；首对仅证明 enqueue→completion 流包络的跨窗交集。停止重烧同配置，先排查设备链路和供电；保留全部中断与故障日志。
