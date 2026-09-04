# TriAttention：当前基线与剩余交付工作

本文替代旧的设计对话导出。TriAttention 已完成基础移植、校准和受控压力验证；后续工作必须以当前实现为基线，不得重新实现已完成阶段或恢复旧的 duplicate cell tracking。

## 固定产品契约

TriAttention 开关语义不得改变：

```text
TriAttention OFF
    完全保持 atomic 原有 unified KV / recurrent / RAM swap / preemption 行为

TriAttention ON
    1. fill-first：有空闲 physical KV 时保持 dense，不提前压缩
    2. 首次 KV 压力：所有 resident eligible sequences 主动 drain 到 3/32
    3. 已压缩 sequence sticky maintenance：增长超过 target + 128 时重新压到 target
    4. Tri floor 耗尽后，atomic 原有 idle demotion / active preemption 才接手
    5. recurrent-only pressure 不得触发 TriAttention KV reclaim
```

固定参数：

```text
target residency = 3/32 = 9.375%
virtual factor   = 32/3 ≈ 10.67x
recent window    = 128
future offsets   = mean
head aggregation = normalized max/union
local clustering = max pool, kernel 5
protect prefill  = false
```

`--triattention-ratio` 只用于显式实验和 A/B；生产基线仍固定为 `3/32`。任何非默认 ratio 都必须由启动参数明确指定并记录，runtime 不得根据质量、压力或模型输出自行修改。

每个 sequence 的目标：

```text
target(L) = max(128, ceil(3L/32))
```

不得在 runtime 因质量担忧静默提高 residency；质量失败应阻止发布或修正实现/校准。

## 已完成，禁止重复建设

### Scorer 与校准格式

- `src/llama-triattention.{h,cpp}` 已实现纯 scorer。
- calibration format 当前为 version 2；version 1 loader 保持兼容。
- 已支持 Ornith/Qwen hybrid 的 partial IMRoPE；注意 IMRoPE 的 position sections
  是 interleaved，但 ggml 的向量旋转配对仍是 NeoX/front-back half：
  - `head_dim = 256`
  - `rotary_dim = 64`
  - `freq_count = 32`
  - `rope_theta = 10,000,000`
- `llama_kv_cells` 是唯一 physical cell metadata owner；不得恢复旧版 `cell_positions[]`、global absolute position 或独立 cell lifecycle callbacks。
- `llama_memory_i::reclaim_kv()` 是通用 lossy reclaim 接口。
- iSWA/hybrid wrapper 已把 reclaim 转发到 attention/base KV，recurrent memory 不参与 TriAttention。
- sparse-position capability 已暴露，调用方不得假设 `[pos_min,pos_max]` 全连续。
- `score_combined()` 必须使用每个 calibration entry 的 exact sampled head stats；不得再按 KV head 退化成“取第一个 sampled head”。
- Vulkan/CPU fallback 的 runtime scoring 已按 sampled layer 批量 readback K，并让同一 KV head 的多个 Q heads 复用一次 dequant + inverse-RoPE；不得恢复 per-cell/per-head 同步 D2H。

### Trie calibration collector

`wanxiangqi/calib/trie-triattention-calib.cpp` 已完成：

- 从 `wanxiangqi/calibration/` 加载 trie；
- DFS 每个 trie node/edge 只 decode 一次；
- selected pre-RoPE Q 作为 graph outputs；
- Vulkan 对全部 sampled layers 排队 async D2H；
- 每个 decoded batch 只做一次 `llama_synchronize()`；
- 无 per-layer graph cut 或 per-layer synchronize。

2026-09 pairing/scaled-RoPE correctness 修复后，旧校准
`7fbfcdcfc7903e11efba96d7c13bea0ae9d60ff81c75475d54ee613bae2b3cc7`
已删除，**不再可信且不得部署**。原因：该 v2 文件的 `rope_style=1`，由旧
collector 把 IMRoPE 的 section interleaving 错当成 even/odd vector pairing
生成。当前 runtime 会 fail-closed 拒绝该文件；旧文件中的 `q_abs_mean` 已按
错误 pair 聚合，不能从现有 aggregate 无损重排修复。

修正后重新采集并部署的可信校准：

```text
source artifact: /tmp/ornith-1.5-35b.triattention
deployed path:   /opt/llama/data/ornith-1.5-35b.triattention
SHA-256:         e95dae507d1f4a64e29be160c5281f8a4308a3332dc9c9176e1a3a0af32e50e2
file size:       83,288 bytes
format:          version 2, rope_style=0, rotary_dim=64, freq_count=32
model:           Ornith-1.5-35B-Uncensored-YMQ-S-MTP
```

源文件与部署文件已逐字节 checksum 一致。文件长度与 160 个 sampled-head
entry 的二进制布局精确一致；10 个 full-attention layer 各包含 16 个唯一 Q
head entry。

采集结果：

```text
unique trie positions = 270,952
context resets        = 0
sampled layer/heads   = 160 = 10 full-attention layers × 16 Q heads
samples per entry     = 270,952
total Q samples       = 43,352,320
peak host snapshots   = 2.18 GiB
full-attn layers      = 3,7,11,15,19,23,27,31,35,39
```

同名 `/tmp/ornith-1.5-35b.triattention` 曾存在早期 tensor-pointer 生命周期
问题产物；该文件已删除并由上述 checksum 的新采集结果替换。不得仅凭路径
判断可信度，必须核对 SHA-256。

### Runtime pressure 与 state 行为

`tools/server/server-context.cpp` 已完成并实测：

- 首次 pressure drain；
- per-slot sticky compressed state；
- unrelated/new prompt 清除 sparse gaps 后恢复 fill-first；
- pressure hints 包含 resident idle slots，而非只看 active slot；
- logical length 使用 `slot.prompt.n_tokens()`，不得再叠加 `n_decoded`；
- recurrent-only pressure 跳过 Tri reclaim；
- floor exhausted 时先走 atomic fallback；
- protected prefill slot 无法抢占时，以真实 available KV 限制下一批大小，避免 decode 内部半批成功后耗尽 KV；
- prompt-cache 和手工 slot restore 可重建 sticky compressed 标记；
- exact saved frontier 可直接继续；需要回退到旧 frontier 时，hybrid recurrent state 会安全地重新 prefill。

当前日志格式可区分：

```text
TriAttention drain
TriAttention maintenance
TriAttention floor exhausted
KV floor exhausted during prefill; limiting next batch
active slot preemption
```

### 可观测性与 backend-native compaction

- `/metrics` 已暴露 `tri_drain_total`、`tri_maintenance_total`、`tri_cells_before/after/freed`、`tri_references_removed`、`tri_target_references`、`tri_hard_keep`、`tri_shared_keep`、`tri_score_seconds`、`tri_pack_seconds`、`tri_floor_exhausted_total`。
- `tri_atomic_fallback_total{reason="kv|recurrent"}` 已按资源原因区分，Tri OFF 不计入 Tri fallback 指标。
- `ggml_backend_tensor_memmove_regions()` 已提供 ordered memmove-style backend primitive；Vulkan 用 device-local reusable scratch 批量执行 move ranges，一组 region 只提交/等待一次。
- `llama_kv_cache::compact()` 已改为 backend-native K/V pack；全部 backing buffer preflight 成功后才执行 data moves，全部 data move 成功后才提交 metadata。
- 不支持 native memmove/layout 的 backend 会明确失败，不再静默走 CPU-mediated 完整 K/V 搬移。
- Vulkan memmove scratch 已纳入 auto-fit reserve。

### 已通过的受控验证

以下真实 Ornith runtime/smoke 记录使用的是上述旧 calibration，因此只能视为
历史工程验证，不能作为 pairing/scaled-RoPE correctness 修复后的 release gate；
必须在新 calibration 生成后重跑。

1. Scorer：`build/bin/test-triattention-score`，0 failures。
2. Cell metadata：`build/bin/test-kv-cells`，exit 0。
3. 单槽真实模型、Vulkan、Turbo4/Turbo2：
   - `physical KV = 2048`
   - 3,519-token prompt
   - 首次 `2048 -> 192`，释放 1,856 cells
   - sticky maintenance 最终到 `ceil(3519*3/32) = 330`
   - 输出 `STICKY_OK`
4. 极端 fallback：
   - `physical KV = 512`、3 slots、3 个并发 2,420-token prompts
   - 日志确认 `Tri drain -> floor exhausted -> batch limit/preemption`
   - 修复后全部 HTTP 200，无 `Context size exceeded`
5. Pressure 下 streaming：
   - assembled content 为 `STREAM_OK`
   - 一个 `finish=stop`
   - 一个 `[DONE]`
   - 无重复或中断
6. Sparse state：
   - save/erase/restore 1,956 tokens
   - frontier continuation `cache_n=1956`、`prompt_n=8`
   - 继续请求耗时约 0.216 s，而非重新 prefill
7. 生产 smoke：
   - `/health` 返回 `{"status":"ok"}`
   - 回复 `PROD_TRI_OK`
   - MTP smoke 接受 15/16 draft tokens
8. Backend-native pack primitive：
   - `build-vulkan-localhost/bin/test-backend-memmove`
   - AMD Radeon RX 6800 / RADV Vulkan
   - overlap + strided regions 均通过，输出 `PASS: Vulkan backend memmove regions`
9. 当前 delivery-candidate、真实 Ornith、Vulkan、Turbo4/Turbo2、`physical KV=2048`：
   - 5,059-token prompt，HTTP 200
   - initial drain：`2048 -> 192`，`score_ms=55.016`，`pack_ms=4.598`
   - 共 1 次 drain + 7 次 sticky maintenance
   - 全请求累计 `tri_score_seconds=0.190775`、`tri_pack_seconds=0.061720`
   - prompt throughput 约 `901.30 tok/s`
   - `tri_atomic_fallback_total{reason="kv"}=0`、`reason="recurrent"=0`
10. 当前 delivery-candidate Tri OFF 轻量隔离 smoke：
   - 同一 Ornith 模型、Vulkan、Turbo4/Turbo2，HTTP 200
   - 启动日志不创建 Tri scorer
   - 全部 `tri_*` counter/gauge 为 0
11. 当前 delivery-candidate MTP + pressure smoke：
   - 3,401-token prompt，`physical KV=2048`
   - 1 次 drain + 3 次 maintenance
   - 累计 `tri_score_seconds=0.105506`、`tri_pack_seconds=0.023833`
   - 正常 `finish=stop`，答案 `4`
   - MTP draft 接受 `16/20`，acceptance = `0.80`
   - 无 KV/recurrent atomic fallback

第一轮 512-cell fallback 测试曾复现 HTTP 500；根因是 floor 后仍构造大于 available KV 的 prefill batch。该问题已修复并用同一场景复测；失败版本不得部署。

## 当前生产基线

最后验证配置：

```text
model:       /opt/llama/data/Ornith-1.5-35B-Uncensored-YMQ-S-MTP.gguf
service:     /etc/systemd/system/llama-server.service
context:     262,144 per slot
KV:          --total-kv auto
backend:     Vulkan full offload
K/V types:   turbo4 / turbo2
TriAttention enabled with deployed calibration
MTP enabled, draft max 2
```

最近一次 auto-fit 得到：

```text
physical unified KV = 69,376 cells
recurrent capacity  = 6 slots
server slots         = 6
```

Auto-fit 会随启动时可用显存变化；`69,376` 不是固定断言。启动期间 `KV size ... does not fit` 是 probe 候选失败，只有最终 `automatic unified KV capacity`、`model loaded` 和 health 才决定启动成功。

当前已部署并验证的 server binary：

```text
SHA-256 = 9fcd58b10b9b26c80482f1087b1e53a46933818ed81098aca06b664edc7e7694
```

注意：以上 server hash 仅作为该次验证记录。重新构建后必须以新 build/deploy checksum 一致为准，不得把此值当成永久常量。

### 2026-09-02 delivery candidate 部署状态

> 历史状态：下述 calibration 使用旧 `rope_style=1` pairing，已被 2026-09
> correctness 修复判定为不兼容。不要用当前源码 + 该 calibration 启动
> TriAttention；先重新采集并重跑 release gates。

当前 build 与生产部署：

```text
build binary: build-vulkan-localhost/bin/llama-server
SHA-256:      9fcd58b10b9b26c80482f1087b1e53a46933818ed81098aca06b664edc7e7694
calibration:  /opt/llama/data/ornith-1.5-35b.triattention
calib SHA-256: 7fbfcdcfc7903e11efba96d7c13bea0ae9d60ff81c75475d54ee613bae2b3cc7
```

生产路径 `/opt/llama/bin/llama-server` 已部署同一 binary：

```text
SHA-256:          9fcd58b10b9b26c80482f1087b1e53a46933818ed81098aca06b664edc7e7694
system fingerprint: b10724-f0b7765e7
```

`llama-server.service` 当前为 active；`/health` 返回 `{"status":"ok"}`，真实推理返回 `DEPLOY_10724_OK`。build、deployed binary 及所需 runtime libraries 已逐项校验 checksum 一致。

用户明确要求优先轻量验证、尽快交工，因此本次收尾没有重新运行 AIME/MATH 全量 A/B、production auto-fit/full-slot 大压力、shared-prefix 三分支、完整 RAM/checkpoint/context-shift 矩阵或多小时 soak。已运行且通过的轻量交付门为：

- focused `llama-server` build；
- `test-triattention-score`：0 failures；
- `test-kv-cells`：exit 0；
- `test-backend-memmove`：Vulkan PASS；
- `git diff --check`；
- 真实 Ornith 2048-cell pressure smoke；
- 真实 Ornith Tri OFF 隔离 smoke；
- 真实 Ornith MTP + pressure smoke。

因此当前状态是：**代码与轻量真实模型验证达到 delivery-candidate，可提交/推送；下面列出的重型门仍是正式 production release 验收项。**

### 2026-09-04 RERoT 整改部署状态

RERoT 整改提交与生产部署：

```text
branch commit:   2ab7223c6
build binary:    build-vulkan/bin/llama-server
deployed binary: /opt/llama/bin/llama-server
binary SHA-256:  ac626f7726386977d28e6e9316c7444e16298d45b3854210ddb1ff9a6903c740
calibration:     /opt/llama/data/ornith-1.5-35b.triattention
calib SHA-256:   e95dae507d1f4a64e29be160c5281f8a4308a3332dc9c9176e1a3a0af32e50e2
```

binary、`libllama-server-impl`、llama/common/mtmd 及全部 ggml CPU/Vulkan
共享库已逐项核对 build/deploy SHA-256 一致；生产进程的 `/proc/<pid>/maps`
确认实际从 `/opt/llama/bin` 加载这些新库，不再误用 `/opt/llama/lib` 的旧副本。

生产 unit 已：

- 删除违规实验参数 `--triattention-ratio 0.5`，恢复固定 `3/32`；
- 启用 `--rerot --rerot-frontier strong`；
- 将 `/opt/llama/bin` 放在 `LD_LIBRARY_PATH` 首位。

本次启动 auto-fit 日志包含 `automatic unified KV capacity = 34816 tokens`；
hybrid recurrent 最终为 3 个 physical slots，server 将 `-np 6` 安全收敛为
3 个 slots，每槽 context 262,144。日志确认 `RERoT runtime armed
(frontier=strong)`、`model loaded`；`/health` 返回 `{"status":"ok"}`。

生产真实请求 `"如何减肥?"` 返回 HTTP 200、`finish_reason=stop`，包含根级
`<ol>`、5 个公开章节、10,161 字符 reasoning 与 1,635 字符最终答案；
共 3,747 completion tokens，wall time 约 96.97 s。MTP speculative 当前未在
该 unit 启用；这仍是指南 Phase 1 的 correctness-first 部署，不得据此宣称
附录 A 的 MTP/RAM/context-shift/full-slot/soak 等 production release blocker
已经完成。

## 剩余发布阻断项

以下工作未完成。按顺序处理；不得以短 health request 替代。

### P0：Tri OFF 零回归

使用同一模型、prompt、seed 和采样参数比较旧基线与新 binary 的 Tri OFF：

- greedy 输出一致；
- unified KV、recurrent fitting、RAM prompt cache、preemption 行为不变；
- 不创建 scorer；
- 不分配 Tri scratch；
- 不启用 sparse semantics；
- 无显著吞吐或显存回退。

必须增加可重复的自动化回归，而不是只做人工 smoke。

### P0：Ornith 质量 A/B

固定 `3/32`，比较 FullKV 与 TriAttention：

- AIME24/25；
- MATH-500；
- 长上下文 retrieval/needle；
- 多轮聊天；
- 代码仓库问答；
- 真实生产 prompt 样本。

要求：

- 同一模型、模板、seed、采样和输出预算；
- 保存每题输出和评分，不只报平均值；
- 质量断崖阻止发布；
- 不允许通过提高 residency 隐藏实现问题。

当前 calibration correlation 只能证明采集一致性，不能替代任务质量评测。

### P0：真实 production auto-fit KV、全 slot 压力

在 shadow instance 或维护窗口运行与生产完全相同的模型、MTP、Turbo K/V 和 Vulkan 配置，使总 resident KV 明确超过最终 auto-fit capacity。

建议至少：

```text
6 concurrent slots
每槽约 12K+ logical tokens
总 resident history > 最终 auto-fit capacity（最近一次为 69,376）
```

验收：

- 第一次压力先 Tri drain；
- 每个 compressed seq 保持 `max(128,ceil(3L/32))` 加合法 shared/hard guards；
- floor 前无 idle demotion/active preemption；
- floor 后 atomic fallback 能完成所有请求；
- MTP checkpoint/rollback 正常；
- streaming 无重复 token；
- 无 5xx、OOM、死锁或服务重启。

### P0：shared-prefix / multi-sequence union

当前真实压力测试的 `shared_keep=0`，未覆盖 shared physical cells。

构造至少三个 sequence：

```text
共同 8K prefix
├─ branch A
├─ branch B
└─ branch C
```

验收：

- `shared_keep > 0`；
- 一个 sequence 的淘汰不得删除另一个 sequence 的 keep reference；
- physical keep-set 等于所有 per-seq keep-set 的并集；
- `references_removed` 与 `physical_freed` 分开核算；
- 三个 sequence 输出无交叉污染；
- compaction 后 shared refs、positions、K/V bytes 不变。

### P0：runtime scoring/compaction 性能（核心路径已完成，待 production-scale 大 drain 门）

旧实现的受控结果：

```text
3,519-token、单次 drain 的旧行为：约 21.8 s
正确 sticky maintenance：             约 40.3 s
```

当前 delivery-candidate 已完成：

- Vulkan/backend-native stable pack；
- 批量执行 move ranges；
- 不把完整 K/V 搬到 CPU；
- 不做 per-layer/per-move synchronize；
- K/V 全部成功后再提交 metadata；
- 使用实际 tensor `type/ne/nb`，支持 Turbo padding 和 V layout；
- pack 后 `used_max_p1 == used`；
- scoring/pack scratch 纳入 auto-fit reserve；
- runtime `score_ms` / `pack_ms` 已进入日志和 metrics；
- sampled-head exact aggregation bug 已修复并增加回归测试；
- Vulkan fallback scoring 已从 per-cell/per-head D2H 改为 per-layer bulk snapshot + KV-head reuse。

当前 2,048-cell 真实 Ornith 压力下，8 次 reclaim 的累计 scoring/pack 约为 `0.191 s / 0.062 s`；MTP 场景 4 次 reclaim 为 `0.106 s / 0.024 s`。剩余性能发布门只有：

- 首次真实 production-scale drain 不 OOM；
- 在最终 production auto-fit 配置下记录首次大 drain 的 score/pack wall time，并确认没有显著吞吐断崖。

Vulkan 是当前发布目标。其他 backend 未支持时必须明确失败，不得静默走极慢或错误路径。

### P1：recurrent-only pressure 实测

代码已增加 KV/recurrent pressure gate，但缺少强制 recurrent shortage 的运行测试。

验收：

- KV capacity 足够、recurrent capacity 不足；
- 日志中没有 Tri drain/maintenance；
- 直接进入原 recurrent victim handling；
- 请求最终完成。

### P1：sparse state 与 RAM swap 完整矩阵

Exact-frontier 手工 save/restore 已通过，仍需覆盖：

- server idle slot 自动 RAM demotion；
- RAM restore 到不同 physical indices；
- MTP speculative checkpoint；
- partial rollback；
- context shift；
- prompt LCP/cache-key reuse；
- 多次 save/restore 后继续 sticky maintenance。

增加持久 metadata 或等价校验：

```text
layout = sparse
policy = triattention
target = 3/32
recent_window = 128
calibration fingerprint
```

不匹配的 sparse state 必须拒绝，不得当成 FullKV state 加载。

### P1：长稳测试

至少运行：

```text
100 次 drain/maintenance
20 次 save/erase/restore
10 次 floor-exhaustion/preemption
全部 fitted slots 持续多小时
```

验收：

- 无 Vulkan validation error；
- 无 CPU/GPU memory leak；
- warmup 后 VRAM 不持续增长；
- 无 stale tensor/readback；
- 无服务重启；
- 每条 stream 恰好一个终止事件；
- 每个请求的 predict budget 在 retry 后不丢失。

## 必须保持的不变量

### KV metadata

```text
llama_kv_cells 是唯一 metadata owner
empty cell 没有 seq refs
non-empty cell 至少有一个 seq ref
seq_get_used(seq) 等于实际引用数
get_kv_used() 统计 physical cells，不统计 references
```

### Tri target

```text
resident_refs(seq) <= max(128,ceil(3*logical_tokens/32))
```

允许超过 reference target 的原因只能明确归因于 hard/in-flight guards。Shared physical cells影响 physical freed 数，不能伪装成 reference target。

### Compaction

```text
retained cell 的 position/ext/seq refs/K bytes/V bytes 前后不变
occupied physical indices = [0,used)
used_max_p1 == used
metadata 只在所有 data moves 成功后提交
```

### Fallback 顺序

```text
KV pressure:
    full Tri drain
    -> refresh usage
    -> sticky floor confirmed exhausted
    -> idle demotion / active preemption

recurrent-only pressure:
    不调用 Tri
    -> 原 recurrent fallback
```

### Fill-first 与 sticky

```text
新 sequence：首次 physical pressure 前保持 dense
已压缩 sequence：resident > target + 128 时 maintenance
unrelated prompt 清除 sparse state 后重新 fill-first
restored exact frontier 继续保持 sticky
```

## 快速回归配方

### 首次 drain + sticky

```text
-c 8192 --total-kv 2048 -np 1 -b 512 -ub 256
```

使用约 3,519-token prompt。预期：

```text
2048 -> 192 initial drain
后续 maintenance 最终 target = 330
HTTP 200
```

### Floor exhausted + fallback

```text
-c 4096 --total-kv 512 -np 3 -b 384 -ub 128
```

并发三个约 2,420-token prompts。预期：

```text
Tri drain
-> floor exhausted
-> prefill batch limit 和/或 preemption
-> 所有请求 HTTP 200
```

### Streaming

在 pressure 配置下组装全部 SSE delta。要求：

```text
finish_reason = stop
[DONE] count = 1
无重复 content
```

### Sparse restore

```text
drain -> slot save -> erase -> restore -> append from exact saved frontier
```

要求：恢复后的 `cache_n` 等于 saved frontier，`prompt_n` 只包含新增 suffix。

## 最终 Definition of Done

只有以下全部成立才可宣布完整交付：

1. Tri OFF 自动化零回归通过。
2. Ornith FullKV vs 3/32 质量 A/B 无断崖。
3. 真实 production auto-fit/full-slot/MTP/Vulkan 压力通过。
4. shared-prefix union 测试通过。
5. recurrent-only pressure 不调用 Tri。
6. runtime scoring/pack 达到明确性能门且首次大 drain 不 OOM。
7. RAM swap、checkpoint、rollback、context shift 矩阵通过。
8. metrics 可观测，fallback 原因可区分。
9. 长稳测试无泄漏、死锁、5xx 或 stream 重复。
10. 最终 build、部署 binary、校准文件 checksum 一致；生产 health 和真实推理通过。
