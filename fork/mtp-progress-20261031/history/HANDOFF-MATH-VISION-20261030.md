# HANDOFF — Qwen3.8-Flash-Next MTP / TP5 双请求：数学愿景 → 现状 全交接
写于 2026-10-30 06:30 JST（Asia/Tokyo）。作者：Grok Bot（E5 lane 执行器），应 Ashley t192u + t193u。
目的：Ashley 接下来几天自己干活、额度见底时，**只读这一份**就能接上。所有路径都是真实路径；凡是没有实测的都标了 `[UNPROVEN]`。

---

## 0. 本文用途与读法

- 这是**单一入口**。先读 §1（为什么这么干）→ §5（四柱现状）→ §7（现在正在烧的那一刀）→ §8（接下来做什么）→ §10（红线）。§3/§4/§9 是查证用的。
- 两台机器：
  - **12022** = 只写代码/静态检查/CPU mock（禁止 GPU/HIP 烧机）。
  - **13022** = 唯一 GPU 实烧机（5×RX6800 gfx1030）。
- 简写：
  - `E` = `/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024`（13022 上 HIP 车道根；**不是** `~/work`）
  - `W` = `/home/kunweiz/work/<run>`（omp 工作目录，12022/13022 都有）
  - `D` = `/home/kunweiz/dispatch/`（omp 派单账本：INDEX.md + 每单 INTENT/PROMPT/LOG）
  - box loot = Grok Bot 盒子上的 `/workspace/loot/...` 与 `/workspace/mtp-war-loot/...`（她看不到盒子时，关键件已拷到 13022 `D/HANDOFF-20261030-sources/`，见下）
- 本文件两份副本（内容相同）：
  - `/home/kunweiz/dispatch/HANDOFF-MATH-VISION-20261030.md`
  - `$E/HANDOFF-MATH-VISION-20261030.md`
- 附带源件（13022）：`/home/kunweiz/dispatch/HANDOFF-20261030-sources/`
  - `mtp-n3-waste-ledger.md`（N3 浪费账本 + 双窗理论 vs 实现，**最权威**）
  - `ACCEPTANCE.md`（soft gate 规程 G1–G4、Rule A–G）
  - `READY-direct-meter.md`（当前直计量刀）
  - `RESULT-*.txt`（10-30 三刀结果）、`REJOIN-SUMMARY.md`
  - `LANES-13022-t184u-1lane.md`（13022 1/1 车道流水）
  - `acceptance-frontier-*.json`、`decision.json`、`critical-path-decision.json`、`delivery.json`、`ready-snapshot-handoff.json`、`hc-mtp-handoff.json`
  - `box-verdicts.txt`（盒子上 ~500 个 loot 目录的判定行摘录，查历史用）

目录树指针（13022）：
```
$E/
├── acceptance-frontier-20261025.json (+ -followup / -postdown / -expanded)   ← 目标定义与 qualified 基线
├── mtp-hc-row-tail-candidate-20261025/        ← 当前 qualified 行尾配置（56.8/58.9 tok/s）
├── mtp-blueprint-20261026/                    ← 蓝图 + 关键路径/归因 + M1/M2 微基准
├── mtp-n3-dispatch-floor-m1-20261027/         ← M1 mock
├── mtp-hc-n3-trio-m3-20261027/                ← M3 HC trio
├── mtp-cold-prewarm-m4-20261027/              ← M4 prewarm（FAIL）
├── mtp-async-protocol-candidate-20261027/     ← 双窗协议地毯扫描修复（built, 未 promote）
├── mtp-dupcompute-candidate-20261027/         ← DC8
├── mtp-{dc6,dc8,m3,gdn,akv,8lru,nkv,u5,latebind,lb,dw*,dualwin,union,prefab...}-2026102x/   ← 各车道
├── mtp-lb-default-on-hip-20261030/            ← 柱2 DEFAULT-ON SURFACE PROVEN
├── mtp-dw-serialcut4-abon-20261030/           ← 柱3 SERIALCUT4 SAFE, SPEED=NOISE
├── mtp-dw-hipspans-abon-20261030/             ← 柱3 HIP_SPANS SAFE, OVERLAP NOT PROVEN
└── mtp-dw-hipspans-direct-abon-20261030/      ← ★ 正在烧（直计量 KEY_V3=1）
/home/kunweiz/dispatch/                        ← omp 派单账本 + 本交接
/home/kunweiz/work/<run>/                      ← omp 工作目录
```

---

## 1. 总目标与数学愿景（从头讲）

### 1.1 目标机与合同（不可改的契约）
- 模型：**Qwen3.8-Flash-Next**（Swift GSQ-RCO Q2_0 GGUF，`/opt/strata/data/models/swift-q2_0/...`），自带 NextN/MTP 头（转换产物 `$E/mtp-qwen4exp-nextn-self-contained-q2_0.gguf`）。
- 拓扑：**TP5**，5×AMD RX6800（gfx1030，16 GB），每卡一 rank；rank 间走 CPU **relay**（F32 线协议）。
- 负载：**两个独立请求**（两个私有 target context，各自 GDN/KV 状态），每个 **KV 262144**，**FA on**，**F32 relay**，各生成 **512 token**，统计 **all 511 intervals**（不剪冷启动、不挑段）。
- 选中安装（**只读，永不改**）：`build-tp5-dual50-20261022/q2-unpack-joint-l2-a-install`、`single-sync-compose-install`；qualified 运行配置 = row-tail（`mtp-hc-row-tail-candidate-20261025`）。保护文件 sha 列在 `acceptance-frontier-followup-20261025.json.preservation`。

### 1.2 目标的演化
1. **10-24**：最初目标 = 每请求 ≥50 tok/s（`hc-mtp-handoff.json`：TARGET_NOT_MET，hc_mean_forward 只有 ~31 tok/s）。
2. **10-25**：ready-snapshot 合并后 50.1/51.7 tok/s 达标（`ready-snapshot-handoff.json`），row-tail 进一步到 **56.77 / 58.93 tok/s**（qualified）。
3. **10-25 起目标改写**（acceptance-frontier）：不是 ">50"，而是 **maximize acceptance-derived actual speedup**——向"接受率决定的理论包络"逼近：
   - 同配置 plain（无 MTP）= 48.5 / 50.3 tok/s
   - 条件包络（plain × 接受率推导）≈ **118 / 122 tok/s**
   - 实现率 realization ≈ **0.478 / 0.475**（只拿到包络的 ~47%）
   - Ashley 口径："~100 tok/s 天花板在不碰 acceptance 的前提下可达"（waste ledger §2 的累计列支持 ~90–98，再砍 D1 可过 100）。

### 1.3 Depth-2 proposal → N=3 verify
- MTP draft 头每轮提 2 个草稿 token（depth 2），target 一次 verify **3 行**（当前位 + 2 草稿）= **N3 图**；接受后提交 1–3 个 token，平均 **2.34 tok/verify**。
- 代价核心：N1（单行）rank0 关键路径 18.71 ms，N3 是 **30.50 ms**（+11.8 ms）。多出来的不是"3 倍真算"，而是**N3 丢掉了所有 t1-only 融合**（见 §3）。
- 所以全部工作的骨架就是：**把 N3 的 per-verify 代价压回接近 N1，同时让两个请求（两窗）互相藏通信**。

### 1.4 双窗数学（柱3 的理论根）
- 每窗一个周期 = 计算 C（verify+draft 在 GPU 上）+ 传输/提交 T（relay fold/commit/submit）。
- **定理**：若 T ≤ C，两窗 **相位错开半周期**（A 在 Compute 时 B 在 Transfer）即可把 T **完全藏掉**，吞吐 → 2 / C。
- 实测（waste ledger "Dual-window theory vs implementation"）：C≈36 ms，T_commit≈8.7 ms → T≤C 成立；但 **uncovered commit ≈7.1 ms/cycle**，commit∩peer_wait 覆盖仅 0.20–0.31，commit∩commit 中位 0.73（**同相**），wait∩wait p90=0.95（锁步）。→ 实现在跟理论对着干。
- 实现要求（Ashley 定的数学红线）：
  - **async send 立即返回** + 对端 **busy-wait**（GPU 端是 device wave 忙等，CPU 端是 pause spin）。
  - **禁止**：doorbell（cv wait/sleep）、scheduler 让出、**同相 meet()** 栅栏。违背者即"邪修"——**有权不测直接改**。
  - GPU→CPU→GPU（CPU fold relay）从 GPU 看只是一次 wait、只是传输延迟，**不算邪修**；邪修是 sleep / doorbell / 同相 meet。
  - 去掉栅栏只能回收 peer-wait（E[max]−E[own]）与跨窗串行化；窗口**自己的** C→T 链（verify→draft→next submit）是真依赖，双窗藏不掉它（要靠 bucket E 流水）。
- 历史警告：10-25 `mtp-ready-pair-candidate` A/B 里 READY_PAIR=0 反而慢（35 vs 41 tok/s）——不是因为 meet 好，而是 meet 掩盖了 host 串行化（整 decode 锁、共享 courier、阻塞 wait）。所以**先修 3–5（锁/courier/spin）再拆栅栏**，否则必回退。

### 1.5 去重复算（柱4 的数学根）
- Ashley："**为缩短单路径而保留的 duplicate compute 是邪修**。" 正确做法：每个结果只算一次、付通信、用双窗把 T 藏掉。
- DC 清单（waste ledger "duplicate-compute 邪修"）：
  - DC1 `TP5_HC_REPLICATED_COMPUTE=1` 5 rank 全算 97 对 HC（~515 MiB/rank/verify 浪费 ≈1.2 ms）；静态 owner 版 **bitwise 相同** → 纯延迟换算力，邪修。
  - DC5 router 5 rank 全算（192 MiB ≈0.45 ms）→ 按 expert 轴切，bitwise。
  - DC6 shared expert 镜像（81 MiB ≈0.2 ms）→ `GGML_TP5_SHARED_EXPERT_SHARD=1`，非 bitwise，走 KL 门。
  - DC7 12 层 full attention 因 n_head_kv=2<5 全镜像（268 MiB ≈0.6 ms + 5×KV 内存）→ 整层 owner 轮转。
  - DC8 N3 下 gate/up 各自重量化同一 src1（48 节点）→ Q8 cache 复用。
  - 合计 ≈1.06 GiB/rank/verify ≈2.3–2.6 ms/rank/verify `[UNPROVEN 估算]`。
- 顺序：**先删重复，再用双窗 overlap 藏通信**。DC1 不许在旧 READY_PAIR binary 上 A/B（会被 host 串行化冤枉）。

### 1.6 数值哲学（soft gate）
- **允许**：数值顺序/舍入差（KL/top1 软门）、多行 kernel 一次读权重、累加重结合、TP 部分和顺序。
- **不允许**：错误实现、粗错（NaN/Inf、形状错、silent no-op 融合、坏 reduction、竞态、top1 崩塌）、**故意降精度买速度**（postdown Q8 截断、top10→top8、HC Q8 捷径）。
- 历史上更慢的近似（MMVQ rows2、MoE top8、HC Q8 捷径、GU-tile、hc-combine-order）一律不复活。

### 1.7 Union 多 shape 图 + max-footprint
- 问题根：N 变化（N1/N2/N3/N4、n_outputs 变化、n_kv 增长）会让 llama `gf_res_prev`（单项）、Meta plan（单 uid，~26 ms）、relay key（10–16 ms）全部失效 → 每次形状切换 ~80 ms host 重建（M4 结论）。
- 愿景：**一张 union 图覆盖所有 shape**，buffer 按 **max footprint** 分配；运行时参数（真 n_kv、真 n_outputs、FA 真 kv_len）走 device param，不重建。
- 双窗下 union buffer 必须 **per-window** 或加 mutex（否则两窗踩同一 scratch）。

### 1.8 LateBind（跨窗/跨步依赖晚绑定）
- 思想：draft 第 1 步可在 verify 结果最终确定前**预取/预绑**，verify 结束只做 accept 检查；命中（bind_accept）就复用，不命中丢弃。
- Ashley 约束："跨窗依赖可以用，但**先写理论边界再实现，勿滥用**。"
- LateBind 不改数值：所有 lb 车道 logits 在 UNSET/ON/OFF 之间 **bitwise 相同**。

### 1.9 CPU 忙等预算
- 忙等可以，但**不要占满所有核**（ngram 等也要用 CPU）。
- 实现（async-protocol candidate）：`TP5_SPIN_CORES` 每窗钉一个物理核（auto：win0=CPU21 worker/43 courier，win1=20/42），其他线程排除这 4 个 CPU。worker 只在忙等时迁到专用核。

---

## 2. Soft gate / 验收框架（源：`ACCEPTANCE.md`）

### 2.1 Rule A–G
| Rule | 内容 |
|---|---|
| A | 小数值/顺序/舍入差 **接受**（KL/top1 软门） |
| B | bug、错误实现、粗错（NaN/Inf、形状、silent no-op、坏 reduction、竞态、top1 崩塌）**fail-closed 拒绝** |
| C | 故意降精度买速度 **拒绝**（postdown Q8、top8） |
| D | 自然 tradeoff（多行 kernel 一次读权重、重结合、TP 部分和顺序）在软门内接受 |
| E | 历史上更慢的近似 **拒绝**（MMVQ rows2、MoE top8、HC Q8 捷径、GU-tile、hc-combine-order） |
| F | **micro/mock 先行**；在绝对把握前**不跑 formal 512×2** |
| G | selected 安装不可动；12022 只写码；`nrows==1` 融合例外**永不放宽**去掩盖别名竞态 |

### 2.2 G1–G4（顺序、fail-closed）
1. **G1 Static**：patch 干净 apply、无 EACCES、无合成 diff、别名/跨 warp 写审计。
2. **G2 Build**：对照 relink 能逐位复现基线（例 hip c4a1634f）；host-only 改动要求 device ISA 不变；selected 不动。
3. **G3 Micro/Mock**（需 `flock -x ple-candidate-gpu.lock`）：真实 body 提速而非只是节点数下降；形状切换 interval0 <50 ms；抓 silent no-op。
4. **G4 Numerics**：先查 Rule B/C/E；再按类：
   - `exact`：对 N1 逐行参考 0 bit 差
   - `numeric-order-only`：mean KL <1e-6，max row KL <1e-5，top1 ≥99.9%，flip 只允许 margin ≤0.10
   - `behavior-changing`（shape 统一等）：mean KL 1e-4..1e-3，max ≤0.021，pad/trim 单独 bitwise 证明
- **Built ≠ PASS**；**PARTIAL ≠ SHIP**；未测数字标 `[UNPROVEN]`。

### 2.3 车道级通用规矩（10-28 以后所有 13022 刀都这么做）
- 每刀一个新 lane 目录 `$E/mtp-<name>-<date>/`，结构：`patch/{build.py,gpu.sh,run.py,result.py,start.sh,common.sh,hdr.inc,prompts-long.txt}`、`build/`、`libs-*/`、`runs/<LEG>/`、`gpu.log`、`RESULT.txt`（末行 `VERDICT ...`）。
- **OFF ≡ stock**：凡宣称"默认关安全"，OFF 腿必须与 stock 逐位相同（`mtp-lb-reuse-accept-20261029` 就因 OFF 不是真 stock 判 NO-PROMOTE）。
- 门变量一律 **exact "1"** 才开（unset/""/0/true/01/" 1"/2 全 OFF）。
- **不走 RSZ_RT 晋升路径**：run.py 一律 pop `GGML_TP5_RSZ_RT`，GATE4 检查无 RSZ_RT。
- 腿型：OFF_i/ON_i 交替，ON_1 rc≠0 立即停；`prompts-long.txt`、predict=128、windows=2、K=9、deadline 2400 s / timeout 2700 s。
- ms/tok 在这些车道只作信息（长 prompt + predict128 负载，与 formal 512×2 的 56 tok/s **不可直接比**）。

---

## 3. N3 浪费账本与砍法排序（源：`mtp-n3-waste-ledger.md`）

### 3.1 N3−N1 真实拆分（rank0）
- 30.50 − 18.71 = **11.8 ms**：gap/compute +9.08（ffn_moe +7.03，linear_attn +2.09）、relay wait +2.54、relay consume +0.46。
- 早期把 "MoE +7.0 / linear_attn +2.1 / device gap ~9 ms" 当三个桶是重复计数，已更正。

### 3.2 冒烟枪：N3 图大 54%
- 原生捕获图（`mtp-blueprint-20261026/mtp-critical-attribution-20261027/native32x2/graph-0x63862cf3c100-{1,2}.csv`）：**N1 = 2476 节点，N3 = 3825 节点（+1349）**。
- 丢失的 t1 融合：
  - **HC trio**（`hc_down_silu_t1_q8` / `hc_owner_tail4_t1_q8` / `hc_combine4_t1`）→ 约 **+700 节点**（最大户）
  - **router `topk_moe_cuda<512>`** → soft_max/argsort/get_rows/sum/clamp/div… 约 **+290**
  - 独立 `quantize_q8_1` +291
  - GDN softplus/l2/silu/concat 约 +170
  - `q2_down_weighted_sum_k128_t1` → 非融合
  - `mul_mat_vec_bf16_load4` → 通用 `mul_mat_vec_f<bf16>`
- 为什么关掉：`ggml-cuda.cu:4467` topk_moe 别名例外只在 nrows==1；N3 下分配器把归一化输出放到 row0 probs 内 128 B → overlap veto；`mmvf.cu` 的 `ne11==1` 守卫；`mmvq-q8-f32.hip` 硬编码 t1。
- 每节点派发 2.4–3.2 µs（M1 实测 ≈2.7 µs）→ +1349 节点 ≈ **3.3–4.3 ms 纯派发**。

### 3.3 桶（per verify，formal row-tail 41.1 ms/cycle → 56.8 tok/s）
| 桶 | 内容 | ms/verify | 单砍 → tok/s | 累计 |
|---|---|---:|---:|---:|
| F | 冷图 build/alloc（每 (ctx,N) 130–230 ms） | 5.6 | 65.9 | 65.9 |
| E | draft+commit+submit 串在 verify 后 | ~5.0 | 64.8 | 76.7 |
| A | +1349 节点派发 | 3.3–4.3 | 61.9–63.6 | 86–89 |
| B | N3 rank 不均/relay wait | 1.3–2.5 | 58.8–60.6 | 90–98 |
| D1 | 非融合 body（中间往返、generic bf16、独立 quantize） | ~2–4 `[未测]` | — | ~100–115 |
| D2 | 真 3 行工作（**必要，保留**） | ~1.5–3 | — | — |
| C | 3×F32 行 relay consume（F32 合同必要） | 0.46 | — | — |
- **F+E+A+B 全砍 ≈ 90–98 tok/s，再砍部分 D1 过 100，不碰 acceptance。**
- 另：instrumented 运行中 ~30% submit≈34 ms = 一窗等另一窗整图 → 未计入的"窗间干扰"桶（双窗问题的直接证据）。

### 3.4 排序后的砍法
1. **F** prewarm 每种 shape 并钉进缓存（→ M4 证明 prewarm 不够，须 union/多项缓存，见柱1）。
2. **A+D1** N=3 版 t1 融合：(a) HC trio (b) topk_moe N3 独立 scratch (c) quantize 融入消费者 (d) q2_down t3 (e) bf16_load4 ne11≤3 (f) GDN。
3. **E** draft step1 与 commit 重叠；下一 N3 submit 叠 4.4 ms 尾巴（≤+10 tok/s）。
4. **B** N3 rank 重平衡；查 split-draft 设备是否偷对窗带宽（DC4，M5 mock 未做）。dev2 自身派发地板最慢（11.26 vs 10.42 ms），单独查 PCIe/IRQ。

### 3.5 已有微基准结论
- **M1**（`mtp-n3-dispatch-floor-m1-20261027/`）PASS：空 kernel 复放 N3−N1 = 3.9–4.2 ms；router 恢复 −0.69，HC trio −0.87，两者 −1.56；两窗同卡几乎完美重叠（union 10.84 vs 10.45）；**M1b**：draft 图在对窗 verify 下只慢 <1% → **硬件能把 T 藏在 C 下**。
- **M2**（`mtp-blueprint-20261026/router-m2-topk-scratch-20261027/`）PASS：融合 N3 router 6.71 µs vs 链 26.78 µs（0.25×），bitwise = N1 逐行（现链 42% 权重差 1–3 ulp）；每图省 ~0.96–1.06 ms。修在**分配器**（独立 scratch），**不许**用 `ggml_set_output`，不许放宽 nrows==1。
- **M3**（`mtp-hc-n3-trio-m3-20261027/`）micro PASS：96 site −0.65 ms（exact）/ −1.6 ms（RB=4）；短模型诊断软门 PASS，节点 3825→~3440；**tail 部分在模型里不 fire**（`up` 是 OUTPUT 的别名守卫，正确的 fallback），需分配器侧 storage。
- **M4**（`mtp-cold-prewarm-m4-20261027/`）**FAIL**：logits bitwise、去掉 137 ms 尖峰，但 build_us 仍出现、interval0 117–200 ms；根因三层单项缓存 → 走 shape 统一（M4a）或多项缓存（柱1 union）。
- 双窗协议地毯扫描（`mtp-async-protocol-candidate-20261027/`，binary 331a0ae9，libggml-hip 202a55b5，**built, NOT run/promoted**）修了：3 处同相 meet、meet 内 doorbell、整 decode std::mutex（→ FIFO ticket spin + SUBMIT_ONLY_GATE）、共享 courier（→ per-window）、HIP ScheduleAuto（→ ScheduleSpin）、yield、relay device wait 的 `s_sleep(2)`、CPU 核预算。

---

## 4. 历史战役时间线（按主题）

> 判定词：PASS=门过；NO-PROMOTE=正确但无收益/不晋升；PARTIAL=未完成验收；STOP=放弃分支；FAIL=门不过。路径前缀 `$E/` 省略时都是 13022 车道；`box:` 是盒子 loot。

### 4.1 HC / row-tail / mean-rows / combine-order / M3 / DC6–DC8 / Q8CACHE
- 10-24 `hc-latejoin-formal-20261024`：hc_mean_forward ON 31.2/30.9 tok/s → TARGET_NOT_MET。`hc-mean-rows-candidate-20261024`：N2 16 行 bitwise。
- 10-25 `mtp-hc-replicated-compute-candidate-20261025`：MIRRORED HC → 56.4/57.9 tok/s（后被认定为 DC1 邪修，但 OFF 臂有 56–59 s TTFT 冷暂停，须先修冷路径再 A/B）。
- 10-25 `mtp-hc-row-tail-candidate-20261025`：**qualified 基线** ON 56.77/58.93（OFF 43.0/55.1，max gap ON 530/328 ms）。
- `mtp-hc-row-epilogue-candidate-20261025`、`mtp-hc-row-layout-control-20261025`：row-epilogue 对象集（hip c4a1634f 来源）。
- `mtp-blueprint-20261026/hc-combine-order`：**STOP**（图序数值失败，bad_bits 2.48M，max_abs 0.49）→ 永禁。
- M3 trio（见 §3.5）；13022 `m3-hc-n3-trio`：HIP 正确性门 PASS，default-off 真 no-op，**不 PROMOTE**（模型内稳态无可见收益）。`mtp-m3-tail-alloc-20261027`/`mtp-m3-tail-alias-20261028`：tail 别名推翻、软门 PASS、稳态无收益 → NO-PROMOTE。`m3-fused-tail-slow-fix-12022` NO-PROMOTE。`m3-tail-noop-reject`：多行 tail body 18.8 µs 比通用链慢 → 作为 silent no-op 反例。
- `mtp-hc-meta-n23-20261028`（Meta HC owner-plan N=1..3）：HARVEST-FAIL（harness bug，非 kernel 结论）；owner-scope fix 修掉原因 4，撞到原因 5 → 仍 NO-PROMOTE。hc-meta-B3 PASS（latejoin-fix 成立），hc-meta-fb4 OFF 回归 PASS。
- DC6 `mtp-dc6-meta-t-20261027`：32x2 软门 PASS（未 promote）；`dc6-dc8-13022-lane`：`SHARED_EXPERT_SHARD=1` 被 loader fail-closed 拒（BLOCKED，非数值）。
- DC8 `mtp-dupcompute-candidate-20261027`（hip 336ef572）→ `mtp-dc8-ids-gate-20261027` exact PASS；`dc8-fold-m2m3` 折入 M2+M3+KEEP PASS（exact）；`mtp-dc8-stderr-hit-20261028` PASS（命中计数证据）；`mtp-dc8-q8cache-20261028` **PARTIAL/NO-PROMOTE**（exact 但无稳态收益）。DC8b（2 槽 cache）未做。

### 4.2 MoE / router / topk_moe / down-n3 / GU-tile / expert-overlap
- `mtp-q2-moe-gu-tile-candidate-20261025`：bitwise 但 8.42 vs 8.12 µs → **STOP_NO_LOCAL_GAIN**。
- `mtp-moe-down-n3-fusion-candidate-20261025`：短参考相等但 0 hit；直写不安全 → **UNSAFE_DIRECT_WRITE_NOT_PROMOTED**。
- `mtp-moe-down-n3-staged-candidate-20261025`：数值+工作量发散（rel L2 1.48）→ **STOP**。
- `mtp-moe-postdown-q8-fusion-candidate-20261025`：数值 PASS，formal 39.98/52.97 vs off 56.1/57.2 + 50 s TTFT → **保持 OFF**（且属 Rule C 类风险）。
- `mtp-moe-group-q2-n3-candidate-20261025` / `mtp-moe-group-gu-q2-n3-candidate-20261025`：**STOP_NO_LOCAL_GAIN**。`mtp-moe-cache-effective-reuse-probe-20261025`：0 cache hit。`mtp-expert-reuse-probe-20261025`：30 槽平均重复率 0.30（仅观测）。
- router 系列（`mtp-blueprint-20261026/`）：router-exact = alias veto fallback；router-snapshot STOP（无本地增益）；router-staged exact 被取代；router-two-stage formal 0.926/0.973 & 1.03/1.006 → 噪声。→ 真正解是 **M2 分配器 scratch**（`mtp-router-scratch-m2-20261027`、`mtp-router-m2-alloc-scratch-20261027`、`mtp-router-m2-review-20261027`）。
- `mtp-moe-n2-fusion-20261028`：PARTIAL；指令"MoE N2 fusion NO-PROMOTE（不如已有 M2）"。
- `mtp-target-nseq2-feasibility-20261025`：一 context 两 seq 38.7 vs 两 context 47.9/49.8 → **STOP_NOT_MTP_PROMOTABLE**。

### 4.3 Dense / shared-load / token-wave
- `mtp-blueprint-20261026/dense-shared-load`：IQ4 bitwise 失败 + Q3_K 更慢 → **STOP**。`dense-token-wave`：exact 但慢 → **STOP**。`dense-row-pack`：预算不足 STOP。`critical-path-decision.json`："停止投机性 dense 调优"。

### 4.4 GDN / fuse / HIP / nosilu / FMA
- `mtp-gdn-strict-step-candidate-20261024`：HYPOTHESIS_NOT_SUPPORTED。
- `mtp-gdn-fuse-n3-20261027` → 13022 gdn-fuse-gate：PASS（bitwise，某配置）；但 FUSE_N3/UNSTRADDLE 在 L3 有粗错漂移：`mtp-gdn-hip-20261028`、`mtp-gdn-hipview-20261028`（NO-PROMOTE）、`mtp-gdn-nanfix-20261028`（fail-closed 守卫 PASS，L3 粗错未变）、`mtp-gdn-nosilu-20261028`（排除"融合 silu 数值"嫌疑）、`mtp-gdn-src2-20261028`（排除 Meta split-state 假设）、`mtp-gdn-fma-20261028`（**FAIL：FMA_ALIGN 无效旗**）、`mtp-gdn-helper-20261028`（NO-PROMOTE）。→ **GDN N3 融合仍有未定位粗错，全部 NO-PROMOTE**。

### 4.5 AKV（actual kv）/ trim / hybrid
- `mtp-actual-kv-hip-20261027`：HIP `FLASH_ATTN_EXT actual_kv` 边界 kernel 微基准 PASS（bitwise，非端到端）。
- `mtp-akv-envelope-20261027`：8×2 诊断 PASS。`mtp-hybrid-akv-tp5-20261027`：数值 bitwise，但 **akv32x2 端到端慢约 14 倍**，归因未定。
- `mtp-akv-trim-20261028` / `mtp-akv-trimcnt-20261028`：正确性 PASS + firing 证据 PASS，**NO-PROMOTE（无提速）**。

### 4.6 Relay / 8lru / KEY_V3 / frames / ptrsafe
- M4 时 relay frame cache 2→8 LRU（hip 706967b5）。
- `mtp-8lru-frames-regate-20261028`：PASS（数值+机制+稳态服务 delta），旗 `GGML_TP5_RELAY_FRAMES`。`mtp-8lru-multishape-20261028`：CONDITIONAL（机制有效，不能按现状 PROMOTE）。`mtp-8lru-sameplan/svcfix/svcport-20261028`：OFF 硬门/正确性 PASS。
- `mtp-8lru-v25-20261029`：**OFF 硬门 PASS；V2 PASS（可 promote）；V3 FAIL（SIGSEGV）**。
- `mtp-8lru-v3fix-20261029`：V3 修复；S_V3_1/S_V3_2/V3 rc0、`capture=0 replay=0` → 这正是今天直计量刀 KEY_V3=1 的先例。V3 = pointer-free/ptrsafe key → 强制 direct 提交（不 capture/replay）。
- 12022 侧 eightlru-* 系列多为 PARTIAL（epoch/fail-closed write-points 未穷尽）。

### 4.7 Prefab / factor-compose / max-footprint / dupcut / U5 admission
- `mtp-factor-compose-20261027`：13022 PASS（CPU/ASan+UBSan/HIP 三套构建 0 failed，只证正确性与结构）。
- `mtp-union-shape-cache-20261027`：U1 meta-containers / U2 sched shells / U3 llama LRU / U4 relay 均 micro PASS（CPU 或 HIP 编译），**NOT in TP5**；`union-u4-32-13022`：32x2 四用例 logits sha 全等（bitwise PASS）。
- `mtp-maxfp-share-20261029`：门全过，但本帧收益 ≈0 → 正确性可 land，内存收益 NO-PROMOTE。max-footprint per-window 版本（union-maxfp-dualwin-*）PARTIAL。
- `mtp-dupcut-20261029` / `mtp-dupcut2-20261029`：BUILD_OK，PARTIAL（命中 >0 未在真实 32×2 证明）。
- U5（admission/多 slot 图缓存）：`mtp-u5-admit-20261028` 正确性 PASS（576/576 bitwise）；`mtp-u5-draft-pw-20261028` PASS（req1 draft 冷账挪出 TTFT，不宣称提速）；`mtp-u5-multireq-20261028` PASS（跨请求命中）；`mtp-u5-rle3-20261028` **PASS 可 promote**（objnew 溢出修好，OFF≡stock）；`mtp-u5-drophr-20261029` PASS；`mtp-u5-integrate-20261027` PARTIAL。
- 冷路径：`mtp-n1n3-rebuild-20261028` PASS（重建消除，不宣称提速）；n1n3-entry-prewarm-once **PASS/PROMOTE 为默认关 opt-in**；`mtp-pwcold-20261028` bitwise PASS；`mtp-pwdraft-ttft-20261028` **FAIL**；`mtp-once2-svc-k4-20261028` PASS（无稳态提速，ENTRY 默认关）；combo-cold-u5-draft-once2 PASS（稳态零提速）。
- **prefab-share-live**（12022 r8 多轮）：全部 **PARTIAL，未 SHIP**——mock 下 ON/OFF 构图次数没有真实下降、OFF≡stock 未实测。

### 4.8 LateBind（sites + default-on surface）
- 站点：site2 `mtp-latebind-site2-20261028` 正确性 PASS（bitwise），default-off 可 land，无提速；zoh-dump PASS（残差吻合）；ldl `mtp-latebind-ldl-20261028` **PASS 可 promote（knob 默认关）**；lbr `mtp-latebind-lbr-20261028` 正确性 PASS 但 bind_accept=0（复用从未发生）→ NO-PROMOTE；pforder/pforder2、don/donfix、open 系列。
- `mtp-latebind-open-20261029`（open-smoke）：PASS，建议 REUSE_ACCEPT=1 默认开。
- `mtp-latebind-long-20261029`：**PASS，REUSE_ACCEPT predict128 稳态 decode −12.4%**。
- `mtp-lb-reuse-accept-20261029`：**NO-PROMOTE（OFF 不是真 stock）**，同配置禁止重烧。
- `mtp-lb-selfcheck-20261029`、`mtp-lb-harden-20261029`、`mtp-lbd-longctx-20261029`、`mtp-lbv2-mismatch-20261029`、`mtp-v2lbd-stack-20261029`：加固/自检/长上下文/不匹配处理。
- `mtp-lb-off0-on1-20261030`（00:53 JST）：SWITCH PROVEN，bitwise，ON ≈ −2.8%。
- `mtp-lb-promote-lctx-20261030`（02:35 JST）：PROMOTE-eligible，SWITCH PROVEN，BITWISE ALL，3 对 decode ON 均更快（均值 −28%，pair1 OFF 90 ms/tok 离群；pair2/3 −3%/−15%）。
- `mtp-lb-default-on-hip-20261030`（~03:16 JST）：**VERDICT DEFAULT-ON SURFACE PROVEN（4 门全过）**
  - GATE1 R_UNSET LBSUM == R_ON（bind_accept=752, reuse_kept=752）
  - GATE2 logits UNSET==ON bitwise（且 UNSET==OFF、ON==OFF 也 bitwise）
  - GATE3 OFF：bind=287 bind_accept=0 reuse_kept=0 discard_mismatch=752 → 开关确实不同
  - GATE4 无 RSZ_RT
  - 信息：UNSET 28.77 ms/tok vs ON 32.05（−10.2%）vs OFF 35.61（−19.2%）——单次、非门。

### 4.9 Dual-window（ready-pair / serial-cut / courier / thru-ab / device-wait / HIP_SPANS / async-protocol）
- 10-25 ready-pair/phase-pair/post-submit-unlock/ready-split：READY_PAIR=1 是 qualified 配置；unlock=1 不比 0 好（~50 vs ~51），TTFT 更差；split-draft 无增益。
- 10-27 `mtp-async-protocol-candidate-20261027`（见 §3.5）+ `mtp-async-proto-8x2-20261027`、`mtp-async-prewarm-draft-8x2-20261027`：DIAG_PASS（8x2 rc0、bitwise），不晋升。`mtp-readypair-cold-dig-20261027`：DIAG_ONLY_PASS。
- 10-28 `mtp-dualwin-courier-20261028`：**NO-PROMOTE**（正确但不提速，且"default-off" 移植改了生产默认）。`mtp-dwcombo-ov-20261028`/`-ov2`：稳定性/正确性 PASS（shutdown 修复可合入），combo overlay NO-PROMOTE（稳态零收益 + 冷启 enqueue hold 回退）。
- 10-29 `mtp-dwasync-rp-20261029`：**NO-PROMOTE**（bitwise，但 READY_PAIR=1 稳态反慢 ~1%）。`mtp-dwasync-pw-20261029`、`mtp-dwsvc-*-20261029`（svc/iso/thru/thru2）：长驻服务吞吐证明车道，多为 PARTIAL。`mtp-serialcut-hip-20261029`：已停、CANCEL，禁止再挂。
- 12022 侧（box loot）：dualwin-courier-*、dualwin-device-wait-live-r8-*（十余轮，均 PARTIAL：设备 wait 替代 meet 的核心替代未交付）、thru-ab-live-r8-*（CPU synthetic A 584 vs B 1121 tok/s，overlap 0.78；**仅合成，非 GPU**）、serial-cut-*、dw-courier-hit-r8-*（计数器落地，PARTIAL）。
- 10-30 13022 三刀：
  - `mtp-dw-serialcut4-abon-20261030`：**SERIALCUT4 SAFE, SPEED = NOISE**（GATE1/2/4 PASS；OFF ms/tok 29.79/38.17/36.46 噪声带 8.38；ON 32.79/41.26/36.02；+5.41% 落噪声带）。**不要再烧同一 A/B**。
  - `mtp-dw-hipspans-abon-20261030`：**HIP_SPANS SAFE, DEVICE OVERLAP NOT PROVEN**（GATE3：joiner `parse error ... unknown key: enabled`）。
  - joiner 修复（box: `hipspans-joiner-enabled-fix-r8-052500` → `hipspans-joiner-rejoin-r8-054800`）：fixed joiner sha **39549c63**，fixture **27/27**；重新 join 旧 ON 日志 → **GATE3_REJOIN FAIL skipped_mode=7**（旧 lane 无 KEY_V3 → relay 走 capture/replay，replay 根本不计量）。
  - **`mtp-dw-hipspans-direct-abon-20261030`：正在烧**（§7）。

---

## 5. 愿景四柱（近两周作战框架）与现状

| 柱 | 内容 | 现状 |
|---|---|---|
| 1 | 真变长：union 图 / n_kv / n_outputs / FA true-len（消桶 F + 让 shape 不再触发重建） | **进行中**（12022 `nkv-fa-*`、`union-*`、`nout-*` 大量 PARTIAL） |
| 2 | LateBind | **DEFAULT-ON SURFACE PROVEN** |
| 3 | 双窗异步真重叠 | SAFE 但 overlap 未证；直计量刀正在烧 |
| 4 | prefab / dedup / 8lru / maxfp（复用与去重） | 大量 PARTIAL；8lru V2 可 promote；prefab 未 SHIP |

### 柱1 真变长 / union / n_kv / FA true-len
- 13022 已证：nkv-abi/abi2（OFF≡B8 bitwise、exact-1 门 OK）、nkv-abi4（稳定/正确 PASS，coverage NO-PROMOTE：fallbacks≠0）、nkv-qsa-rebind/r8r9-binds（8×2 PASS，NKV_FAIL=0）、union-u4-32（bitwise）。
- `mtp-nkv-abi3-20261028` NO-PROMOTE（NKV_BINDER_STATS 不出现）。`mtp-fa-rtlen-hip-20261029`、`mtp-union-skel-hip-20261029`：**已停，禁止再挂**。
- 12022 正在推进：`nkv-fa-true-len-prove-r8-*`、`nkv-fa-tail-*`、`fa-host-true-len-hipready-*`、`union-binder-*`、`nout-*` —— 全部 PARTIAL（CPU harness/语义模型 OK，GPU/HIP 未验）。
- ⚠️ **警告**：`GGML_TP5_FA_MASK_KVLEN=1` 在 FA kernel 修好之前**不得在 GPU/HIP 上开**——`ncols2>1` 路径有越界（OOB）风险。

### 柱2 LateBind
- `mtp-lb-default-on-hip-20261030`：四门全过；**UNSET ≡ ON**（LBSUM + logits bitwise），OFF bind_accept=0。
- 结论：**可以认真推树内默认 UNSET≡ON**（即不设 `GGML_TP5_LATEBIND` 时等价于 `GGML_TP5_LATEBIND=1 + TP5_MTP_LATEBIND_REUSE_ACCEPT=1`）。合入时保持：OFF(=0) 仍可回退、OFF≡stock、无 RSZ_RT。
- 未做：真正的跨请求 DEFER（latebind-defer-*、multireq-* 均 PARTIAL）。

### 柱3 双窗异步真重叠
1. SERIALCUT4 → **SAFE, SPEED=NOISE**（+5.41% 在噪声带内）。
2. HIP_SPANS → **SAFE, DEVICE OVERLAP NOT PROVEN**（joiner `unknown key: enabled`）。
3. joiner enabled-fix → fixture 27/27 解析通。
4. rejoin 旧日志 → **GATE3_REJOIN FAIL skipped_mode=7**（FRAMES=1 无 KEY_V3 → capture/replay；~7.8k replay/provider 不计量）。
5. **当前直计量** `mtp-dw-hipspans-direct-abon-20261030`：ISO 加 `GGML_TP5_KEY_V3=1` 强制 direct；gpu.sh pid 19790；OFF_1 rc0 用时 947 s（06:05:48→06:21:38 JST）；ON_1 自 06:21:38 JST 开烧；写本文时（06:30 JST）RESULT 未出。ETA 原估 06:50–07:15 JST；按 OFF_1 947 s/腿推算 4 腿约 **07:10–07:20 JST** 出 RESULT。
- 理论缺口仍在：即便 overlap 证到 >0，也只是"设备上两窗确有重叠"，不等于吞吐收益；吞吐要靠 async-protocol（拆 meet + spin + per-window courier）+ bucket E 流水。

### 柱4 prefab / dedup
- 12022 大量 PARTIAL：prefab-ab-*、prefab-share-live-r8-*（十几轮）、dupcut-*、router-ncache-*、dwcombo-pool-lifetime 等。
- **prefab-share-live 未 SHIP**：mock ON/OFF 下构图次数没有下降，OFF≡stock 未实测。
- 13022 已证可 land 的：8lru V2、maxfp-share 正确性、u5-rle3、DC8 exact（无收益）、factor-compose 正确性。

---

## 6. 机器 / 账号 / 调度规矩（自续必读）

### 6.1 12022
- 用户 `kunweiz`，`ssh kw12022`。
- 派活：`omp-remote`（默认就是 12022），lazy provider 不限量；保持 **20+ 个 vibe** 并行（额度紧时见 §8.7，少派）。
- 编译：**只用** `/home/kunweiz/.local/bin/build-slot`（**hermes build-slot 已废**，误用过一次导致无效验证）。build-slot 超时 124/75 时按规矩停，不要循环重试。
- 职责：写代码、静态检查（apply --check / -fsyntax-only）、CPU mock。**不跑 GPU/HIP**。
- t191u：与 GPU 无关的代码工作仍在 12022 做。

### 6.2 13022
- `ssh kw13022`（本地 ssh_config 会打印一行 `Unsupported option "gssapiauthentication"`，无害）。
- omp：`omp-remote --host 13022`，lazy，**不占内置 1/1 名额**。
- **GPU 烧机必须双锁**（t189u）：`/tmp/ple-candidate-gpu.lock`（fd8）+ `$E/ple-candidate-gpu.lock`（fd9），由 lane 的 `patch/common.sh lock_gpu` 统一拿。`~/work/ple-candidate-gpu.lock` 是误建空文件，不是锁。
- t184u：HIP 同时**只许 1/1 路**。
- t188u：一刀自然结束后**不要自动续挂下一个 flock HIP**；下一步改走 omp lazy（先在 12022 修/准备，再 13022 omp+flock）。
- t190u：GPU 实验优先 **13022 omp + flock**。
- 路径：HIP 车道根 = `$E`（不是 `~/work`）；omp 工作 = `/home/kunweiz/work/<run>`；派单账本 = `/home/kunweiz/dispatch/`（INDEX.md，每单 INTENT.md/PROMPT.md/LOG.md；omp-task 运行目录 `~/.omp-task/runs/<run>/final.md`）。
- start.sh ssh 挂住：`start.sh` 末尾 `... &` 会占住 ssh stdout，客户端在打印 started 后挂起——**直接 kill 本地 ssh client**；gpu.sh 是 setsid/nohup，会活着（已验证）。
- 查锁/查进程：
```
ssh kw13022 'ps -eo pid,etime,cmd | grep -E "gpu.sh|tq-tp5-compose" | grep -v grep; ls -l /proc/$(pgrep -f "patch/gpu.sh" | head -1)/fd 2>/dev/null | grep lock'
```

### 6.3 Providers
- `kw92.cyou/lazy/{default,task}`：默认用这个。
- ctyun：额度将尽，**勿默认**。

---

## 7. 当前直计量刀（打开就能跟）

### 7.1 是什么
- Lane：`$E/mtp-dw-hipspans-direct-abon-20261030`（README-LANE.md 有全说明）。
- 为什么：旧 HIP_SPANS lane 只有 `RELAY_FRAMES=1`、没 `KEY_V3` → relay v2 watch-gated LRU → `mode = exec ? replay : (warmed ? capture : direct)`；`meter_submit_begin()` 对 capture `++meter_skipped_mode`（每 provider 7 次），~7.8k replay 在计量前就 return → 修好的 joiner fail-closed 拒判（skipped_mode=7）。ASYNC_PROTOCOL 不是原因。
- 修法（只改环境、不重编）：ISO 加 **`GGML_TP5_KEY_V3=1`**（保留 FRAMES=1）→ `relay_ptrsafe_key()` 为真 → 每次提交强制 direct → capture=replay=0、skipped_mode=0、全部计量（span ring 32768/rank ≫ ~7.8k）。先例：`mtp-8lru-v3fix-20261029` S_V3_1/S_V3_2/V3。
- 二进制：`build/tq-tp5-compose-hsp` sha 89a79a1a…；`libs-hsp` hip sha 3c65d3dd…（从旧 lane 原样拷贝）；`join/dw-hip-overlap-join` = **fixed joiner sha 39549c63**（27/27；已验证它对旧 ON_1 正确拒判 skipped_mode=7）。
- 环境：
  - BASE（hdr.inc）= `TP5_MTP_LATEBIND_DRAFT1=1 TP5_MTP_LB_COUNTERS=1 GGML_TP5_LB_SELFCHECK_REUSE=1`，REUSE_ACCEPT=UNSET，LATEBIND=UNSET，`GGML_TP5_RELAY_FRAMES=1 GGML_TP5_RELAY_COUNTERS=1`
  - ISO = `TP5_MTP_SUBMIT_ONLY_GATE=0 TP5_MTP_ASYNC_PROTOCOL=1 TP5_MTP_GATE_METRICS=1 GGML_TP5_KEY_V3=1`
  - OFF = BASE+ISO+`GGML_TP5_DW_HIP_SPANS=0`；ON = BASE+ISO+`GGML_TP5_DW_HIP_SPANS=1`（HIP_SPANS 单独即可 arm meter；OVERLAP_METER/SERIALCUT4/RSZ_RT 由 run.py pop）
  - 腿 OFF_1 ON_1 OFF_2 ON_2（ON_1 rc≠0 提前停）；prompts-long，predict=128，windows=2，K=9，deadline 2400 / timeout 2700
- 这不是旧配置重烧（t188u 允许）：KEY_V3=1 是新配置。

### 7.2 门（patch/result.py → RESULT.txt）
- **GATE1** 4 腿 rc0
- **GATE2** logits OFF_i == ON_i bitwise
- **GATE3** ON：所有 `TP5_DW_HIP_OVERLAP` aggregate `skipped_mode=0`，且 `TP5_RELAY_COUNTERS capture=replay=0`，且 fixed joiner rc0，且 `pair_scope=same_pci_shared_epoch`，且 ≥1 对 `overlap_ns>0`，且 span 头 `dropped=0 errors=0`；OFF：无 span/头/banner/pair
- **GATE4** 每腿 KEY_V3=1 + FRAMES=1，无 RSZ_RT，HIP_SPANS banner 与腿匹配，LATEBIND/SERIALCUT4 未设
- （result.py 已修：`TP5_DW_HIP_SPANS enabled=1` banner 不再算坏头——旧 lane 单这一条就会挂 GATE3）
- **通过的 VERDICT 行**：`HIP_SPANS SAFE + DEVICE OVERLAP PROVEN`

### 7.3 怎么看
```
ssh kw13022 'L=/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-dw-hipspans-direct-abon-20261030; tail -20 $L/gpu.log; ls $L/runs; cat $L/RESULT.txt 2>/dev/null | tail -30; ps -p 19790 -o pid,etime,cmd'
```
- gpu.log 时间是 **UTC**（例：`21:05:48` = 06:05:48 JST，+9h）。
- 失败时看：`$L/runs/ON_*/stderr.log` 里 `TP5_DW_HIP_OVERLAP` aggregate 的 `skipped_mode / dropped / errors`，以及 `TP5_RELAY_COUNTERS capture= replay=`；joiner 输出在 result.py 的 `DEV ...` 行（`join_err=[...]`）。

### 7.4 判读
- **PROVEN**：记证（把 RESULT.txt 拷进 `D/HANDOFF-20261030-sources/` 或 box loot），在 LANES 追加一节；**意义**：设备上两窗确实同时在跑（overlap_ns>0），柱3 的"可重叠"有了硬证据；下一步才是吞吐。
- **FAIL**：按顺序看 skipped_mode（≠0 → 仍有非 direct 提交，查 KEY_V3 是否真生效）、dropped（>0 → ring 不够）、errors（>0 → anchor 排序出负 elapsed，需换 anchor 方案）、pair_scope（不是 same_pci_shared_epoch → 时钟域不可比）。
- 风险：direct-only 可能改变 ms/tok（仅信息，不进门）。

### 7.5 禁令（这一刀）
- **不要**再烧没有 KEY_V3 的旧 HIP_SPANS 配置。
- **不要**在 19790 持锁时开第二路 flock。
- 自然结束后**不要**自动续挂下一刀 HIP（t188u）。

---

## 8. 接下来几天建议动作（按优先级）

1. **收直计量 RESULT → 判 GATE3**（预计 ~07:10–07:20 JST 10-30）。PROVEN → 记证；FAIL → 按 §7.4 顺序读 skipped_mode/errors/dropped，修在 12022（joiner/meter/anchor），再 13022 omp+flock 重证。
2. **1/1 占用时不开第二路 GPU**。先 `ps`/查锁（§6.2）再动。
3. **柱2**：LateBind 树内默认 **UNSET≡ON** 已 PROVEN——可以考虑合入（谨慎）：
   - 合入点：让 `GGML_TP5_LATEBIND` unset 时走 ON 路径（+REUSE_ACCEPT=1 语义），`=0` 显式关；保持 OFF≡stock；不引入 RSZ_RT。
   - 合入后在 13022 跑一刀 UNSET vs 显式 ON vs OFF 的回归（模板：`mtp-lb-default-on-hip-20261030/patch/`）。
4. **柱1**：修 FA true-len（ncols2>1 越界）之前，**`GGML_TP5_FA_MASK_KVLEN=1` 一律不上 GPU**。12022 推 `nkv-fa-true-len-prove-r8-*` 的 kernel 修复 → 先 CPU harness，再 13022 micro。
5. **柱3**：若 overlap 仍未证：查 meter/anchor（不是再烧）。不要 flock 重烧同旧配置。overlap 证到后，真正的吞吐刀是：`mtp-async-protocol-candidate-20261027`（331a0ae9 / hip 202a55b5）+ DC1 静态 owner（先修 DC1-pre 冷路径）——**built, never run**，按 micro→8x2→32x2 推。
6. **柱4**：prefab 只有**真 A/B 下构图次数下降 + OFF≡stock + bitwise** 才 SHIP。否则保持 PARTIAL。
7. **额度紧**：少派 vibe；优先用 13022 现成 lane 脚本克隆（`cp -r` 一个已完成 lane 的 `patch/`，改 hdr.inc/ISO/result.py 门），不要每刀从零让 vibe 写 harness。
8. 账本里仍未落地、值得做的 kernel 侧（按 M1 预估收益）：M2 分配器 scratch 正式集成（−0.96 ms/图）、M3 tail 分配器修（−2 节点/site）、quantize 融入消费者（+291 节点）、bf16_load4 ne11≤3、q2_down t3。

---

## 9. 证据索引表

| 主题 | 路径（13022 `$E/` 下，或 box:） | VERDICT / 状态 |
|---|---|---|
| 目标定义 / 包络 | `acceptance-frontier-20261025.json` | IN_PROGRESS；包络 118/122，实现率 0.478 |
| qualified 基线 | `mtp-hc-row-tail-candidate-20261025/` | ON 56.77/58.93 tok/s |
| ready-snapshot | `mtp-ready-snapshot-consolidated-20261025/` | 50.1/51.7 达 >50 |
| 蓝图决策 | `mtp-blueprint-20261026/decision.json` | NO_NEW_QUALIFIED_SPEEDUP_KEEP_ROW_TAIL |
| delivery | box:`HANDOFF-MTP-20261030/sources/delivery.json` | EVIDENCE_COMPLETE_NO_NEW_SPEEDUP |
| 浪费账本 | `D/HANDOFF-20261030-sources/mtp-n3-waste-ledger.md` | 权威 |
| 验收规程 | `D/HANDOFF-20261030-sources/ACCEPTANCE.md` | 规程 |
| M1 派发地板 | `mtp-n3-dispatch-floor-m1-20261027/` | PASS（N3−N1 3.9–4.2 ms） |
| M2 router scratch | `mtp-blueprint-20261026/router-m2-topk-scratch-20261027/` | PASS 0.25×，bitwise=N1 |
| M3 HC trio | `mtp-hc-n3-trio-m3-20261027/` | micro PASS；模型 NO-PROMOTE；tail 不 fire |
| M4 prewarm | `mtp-cold-prewarm-m4-20261027/` | FAIL |
| async-protocol | `mtp-async-protocol-candidate-20261027/` | BUILT NOT RUN |
| DC8 | `mtp-dupcompute-candidate-20261027/`、`mtp-dc8-*` | exact PASS；q8cache NO-PROMOTE |
| DC6 | `mtp-dc6-meta-t-20261027/` | 32x2 软门 PASS；SHARD 旗被 loader 拒 |
| hc-combine-order | `mtp-blueprint-20261026/hc-combine-order/` | STOP（数值失败） |
| GU-tile | `mtp-q2-moe-gu-tile-candidate-20261025/` | STOP_NO_LOCAL_GAIN |
| down-n3 | `mtp-moe-down-n3-{fusion,staged}-candidate-20261025/` | UNSAFE / STOP |
| postdown Q8 | `mtp-moe-postdown-q8-fusion-candidate-20261025/` | 保持 OFF |
| nseq2 | `mtp-target-nseq2-feasibility-20261025/` | STOP_NOT_MTP_PROMOTABLE |
| dense | `mtp-blueprint-20261026/dense-*` | STOP |
| GDN FUSE_N3 | `mtp-gdn-{hip,hipview,nanfix,nosilu,src2,fma,helper}-20261028/` | NO-PROMOTE；fma FAIL |
| AKV | `mtp-akv-trimcnt-20261028/`、`mtp-hybrid-akv-tp5-20261027/` | NO-PROMOTE；hybrid 慢 14× |
| 8lru | `mtp-8lru-v25-20261029/`、`mtp-8lru-v3fix-20261029/` | V2 PASS 可 promote；V3 修复 rc0 |
| U5 | `mtp-u5-rle3-20261028/` | PASS 可 promote |
| maxfp | `mtp-maxfp-share-20261029/` | 正确性可 land，收益 NO-PROMOTE |
| LateBind long | `mtp-latebind-long-20261029/` | PASS −12.4% decode |
| LateBind reuse-accept | `mtp-lb-reuse-accept-20261029/` | NO-PROMOTE（OFF 非 stock），禁重烧 |
| LateBind promote-lctx | `mtp-lb-promote-lctx-20261030/` | PROMOTE-eligible |
| **LateBind default-on** | `mtp-lb-default-on-hip-20261030/RESULT.txt` | **DEFAULT-ON SURFACE PROVEN** |
| courier | `mtp-dualwin-courier-20261028/` | NO-PROMOTE |
| dwasync-rp | `mtp-dwasync-rp-20261029/` | NO-PROMOTE（−1%） |
| SERIALCUT4 | `mtp-dw-serialcut4-abon-20261030/RESULT.txt` | SAFE, SPEED=NOISE |
| HIP_SPANS | `mtp-dw-hipspans-abon-20261030/RESULT.txt` | SAFE, OVERLAP NOT PROVEN |
| joiner rejoin | box:`/workspace/loot/hipspans-joiner-rejoin-r8-054800/` | GATE3_REJOIN FAIL skipped_mode=7 |
| **直计量** | `mtp-dw-hipspans-direct-abon-20261030/` | **ON_1 烧中，RESULT 未出** |
| prefab-share-live | box:`/workspace/loot/prefab-share-live-r8-*` | PARTIAL，未 SHIP |
| nkv-fa true-len | box:`/workspace/loot/nkv-fa-true-len-prove-r8-*` | PARTIAL |
| 盒子判定摘录 | `D/HANDOFF-20261030-sources/box-verdicts.txt` | ~500 行 |

---

## 10. 明确禁止

- **RSZ_RT 晋升路径**（`GGML_TP5_RSZ_RT`）——任何腿都不许有。
- **故意降精度买速度**（postdown Q8、top10→top8、HC Q8 捷径）；以及历史更慢近似复活（MMVQ rows2、GU-tile、hc-combine-order）。
- **hermes build-slot**（只用 `/home/kunweiz/.local/bin/build-slot`）。
- **1/1 占用时开第二路 flock HIP**；以及一刀结束后自动续挂下一刀 HIP。
- **把 PARTIAL 当 SHIP**；把 Built 当 PASS；用合成 diff 冒充补丁。
- **`GGML_TP5_FA_MASK_KVLEN=1` 在 FA kernel 修好前上 GPU/HIP**（ncols2>1 OOB）。
- **违背双窗数学证明的实现**：同相 meet()、doorbell（cv wait / sleep）、scheduler 让出、整 decode 大锁——见到即可不测直接改。
- 放宽 `nrows==1` 融合例外去掩盖别名竞态；用 `ggml_set_output` 做 router scratch。
- 修改 selected 安装（`q2-unpack-joint-l2-a-install`、`single-sync-compose-install`）。
- 在 12022 跑 GPU/HIP；在 micro/mock 没过前跑 formal 512×2。
- 重烧已判过的同配置 A/B（SERIALCUT4、旧 HIP_SPANS 无 KEY_V3、lb-reuse-accept、union-skel-hip、serialcut-hip、fa-rtlen-hip）。
- 跨窗依赖（LateBind 等）没写理论边界就实现；忙等占满所有 CPU 核。

---

## 附：一页速查

```
# 现在烧的刀
ssh kw13022 'L=/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-dw-hipspans-direct-abon-20261030; tail -20 $L/gpu.log; ls $L/runs; cat $L/RESULT.txt 2>/dev/null | tail -30'
# 谁在持 GPU
ssh kw13022 'ps -eo pid,etime,cmd | grep -E "gpu.sh|tq-tp5-compose" | grep -v grep'
# 派单账本
ssh kw13022 'cat /home/kunweiz/dispatch/INDEX.md'
# 新开一刀（只在 1/1 空闲时）：克隆已完成 lane 的 patch/，改 hdr.inc + ISO + result.py 门，然后
ssh kw13022 'bash $E/<new-lane>/patch/start.sh'   # 打印 started 后 kill 本地 ssh client
```

## 11. 直计量 RESULT（2026-10-30 ~07:18 JST）

- **VERDICT：HIP_SPANS SAFE，DEVICE OVERLAP NOT PROVEN**
- 四腿 rc0；GATE1/2/4 PASS；logits OFF≡ON 逐位。
- `KEY_V3=1` 生效：skipped_mode=0，capture=replay=0，ON spans≈77995。
- **GATE3 FAIL**：joiner 报 `pci 0000:0f:00.0 maps to multiple windows`（map_lines=2，pairs_overlap=0）——根因已非旧的 skipped_mode=7。
- gpu.sh 19790 已自然结束，锁已释放；按 t188u **不自动续 flock 重烧**。
- 证据：`/workspace/loot/mtp-dw-hipspans-direct-abon-20261030/RESULT.txt`（本地）；13022 同 lane 及 `~/dispatch/HANDOFF-20261030-sources/RESULT-mtp-dw-hipspans-direct-abon-20261030.txt`。
- **下一刀**：修 joiner/MAP 的 PCI→window 唯一映射（先在 omp CPU 上做，不必立刻再烧 GPU）。

## 12. hipspans joiner PCI→window 修复（r8-081800，2026-10-30）
- 结论：PARTIAL。原四腿日志上 GATE3 仍 FAIL。
- joiner 改为用 MAP `provider=` 做窗口归属（binary f5592cc6；44cf8ee4 已拒）；fixture 51/51。
- ON joinrc=0，pairs_overlap=0，scope=cross_provider_unattributed。
- 下一步：需带 `provider=` MAP 的 HIP 实烧；不自动续 flock。
- loot：/workspace/loot/hipspans-joiner-pcimap-r8-081800/（REPORT.md / REJOIN_RESULT.txt / patch.diff / rejoin/）
