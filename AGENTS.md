现在这条线终于可以按“**收口工程**”来规划，而不是继续根因考古。

当前基线 `7509335d1` 我会定义为：

> **RERoT Core Correctness Candidate**：核心数学、Vulkan MoE、Turbo/F16 数值轨迹、child recurrence、final fence、response surface 都已经有很强的正确性证据；但离“atomic-llama-cpp-turboquant production-compatible RERoT”还差完整兼容矩阵、资源压力、长稳和最终性能门。

整张蓝图可以压成一条主线：

```text
锁死当前正确性基线
        ↓
真实长任务闭合
        ↓
长轨迹数值确定性
        ↓
Tri/Turbo/MTP/RAM/context-shift 全组合
        ↓
多请求/共享前缀/抢占/流式/tool 等 server 语义
        ↓
生产级资源压力 + auto-fit
        ↓
质量总验收
        ↓
性能优化到目标
        ↓
长稳 soak
        ↓
artifact sealing + shadow/canary + production
```

下面一段一段走到底。

## Phase 0：把 `7509335d1` 封成不可随便破坏的 Golden Baseline

这一阶段**不追求新功能**，只做一件事：以后任何优化都必须能证明没有破坏今天已经修好的东西。

需要把现在这些证据全部转成稳定的 release gate：

| 门                             | 当前状态                  | 最终应固定成                                    |
| ----------------------------- | --------------------- | ----------------------------------------- |
| Vulkan MoE 18/19              | 已有 hermetic test      | 永久 CTest                                  |
| native-vs-native determinism  | 已证明                   | 18/19/20、多次重复、不同前序 shape                  |
| RERoT vs native fixed tape    | Turbo/F16 128 step 已过 | 128/512/更长固定 tape                         |
| mixed ubatch child recurrence | CPU/Vulkan 已过         | 永久 invariant                              |
| response reasoning/content    | 已清洁                   | OpenAI/Responses/Anthropic 单测             |
| random-ID close               | 已工作                   | exact-owner-only regression               |
| final acquire fence           | 已工作                   | exactly-once regression                   |
| PRIVATE transport             | 已隔离                   | public response 不得出现 forced PRIVATE bytes |

从这一刻以后，任何性能 patch 如果让其中一项红，**直接判 patch 错，而不是调 tolerance 或改 prompt。**

特别是 MoE：

```text
18 → 19 → 20 → 18 → 19
```

现在已经是“红线测试”。

---

# Phase 1：先拿下真正的长任务闭合

当前唯一明显还没完成的真实语义门，是“大洲国家”这种长、多 child 请求。

之前 8192 context 的失败现在不能当算法 FAIL：

```text
8 children
持续并发 111 s
→ episode token budget / context hard limit
```

下一次应该直接用接近 production 的容量，例如：

```text
context >= 32768
最好直接 131072 / 262144
--total-kv auto
Turbo4/Turbo2
STRONG
默认 child direct-worker
```

这次验收不看“几秒结束”，而看完整生命周期：

```text
root planner
→ 8 child admitted
→ 每个 child 自然 exact random-ID close
→ 8/8 retired
→ exactly one final survivor
→ final fence prepare
→ close-token replay exactly once
→ serial final answer
→ HTTP 200
→ finish_reason=stop
```

最终正文要求：

* 七大洲/地区组织合理；
* 没有串洲；
* 没有章节死循环；
* 没有 child 接管兄弟任务；
* 没有 prompt echo；
* 没有 `</think>` 等内部协议；
* 没有随机 ID 暴露；
* usage 与实际 token accounting 一致。

如果这一步过了，可以第一次说：

> **RERoT 核心真实语义闭环通过。**

如果不过，也不要碰 random-ID 规则。直接分类是哪一种失败：

```text
A. context/resource 不够
B. 某 child 不 close
C. child semantic drift
D. final fence
E. serial tail
F. API assembly
```

现在这些层已经能分开查，不需要再猜。

---

# Phase 2：把“数值正确”从 128 step 扩成真正的长轨迹保证

虽然当前已经非常强，但 production 前我仍会补这一道。

固定一份 native teacher tape，至少覆盖：

```text
short prompt
19/20-token MoE threshold prompt
9.11
中文长 prompt
代码 prompt
```

然后对：

```text
Turbo4/Turbo2
F16 control
rollback 0
rollback >0
不同 ubatch composition
不同 physical row ordering
```

跑长轨迹。

核心指标分三级。

第一层必须是：

```text
native vs native
mismatch = 0
```

这属于确定性，不接受 tolerance。

第二层：

```text
RERoT single-lane vs native
argmax decision mismatch = 0
```

这是最重要的语义数值门。

第三层才看：

```text
logits rel L2
layer activation error
```

可以有浮点差，但不能随长度无界增长，也不能越过 sampled-decision boundary。

如果未来 Turbo 在例如 step 700 才产生 first top divergence，就停在那个 token，对 layer 做二分，而不是继续跑自然语言。

做到这一点后，过去那种：

```text
“输出坏了，是模型语义还是 kernel 数值？”
```

基本就可以从项目里消失。

---

# Phase 3：重新认证 TriAttention + Turbo + unified KV

注意这里说的是**重新认证**，很多实现历史上已经有，不是全部重写。

最近底层经历过：

* MoE routing reset；
* attention shader precision；
* F32 recurrent hand；
* recurrent row semantics；
* response handling。

因此以前的兼容测试不能全部自动继承。

需要重新走四个主组合：

```text
RERoT + FullKV
RERoT + Turbo
RERoT + Tri
RERoT + Tri + Turbo
```

Tri 重点不只是“能回答”：

```text
initial drain
sticky maintenance
3/32 target
backend-native compaction
shared physical cell union
sparse position reader
```

都必须正确。

另外必须单独验证 fallback 顺序：

```text
KV pressure:
Tri reclaim
→ refresh
→ floor exhausted
→ atomic demotion/preemption

recurrent-only pressure:
禁止调用 Tri
→ recurrent fallback
```

不能因为最近 RERoT 改动重新把资源类型混掉。

---

# Phase 4：MTP / speculative decoding

这是完整 production 兼容的第一块大状态组合。

原则非常清楚：

```text
MTP draft
必须绑定：
topology epoch
publish epoch
layout epoch
reader view
```

只要 peer publish、fork、context shift 等让 view 改变：

```text
旧 draft invalid
→ reject / rollback
→ 重新 draft
```

不得让 speculative token 穿越未提交 frontier。

要逐项验：

```text
RERoT + MTP no peer update
RERoT + MTP peer update
RERoT + MTP fork barrier
RERoT + MTP rollback
RERoT + MTP Tri pressure
RERoT + MTP final fence
```

最终还应该有真实 Ornith acceptance-rate 数据。

但 acceptance rate 再低也只是性能问题；**首先保证 target 结果与 MTP OFF 一致。**

---

# Phase 5：RAM / checkpoint / rollback / prompt cache / context shift

这是我认为整个 production 兼容里最容易藏状态 bug 的一层。

现在 state 不只是 token tape，而是：

```text
tree topology
document runs
PUBLIC/PRIVATE/PENDING
random child IDs
reader views
frontier epochs
brain
F32 hand
conv / R0-R2
sampler RNG
MTP checkpoint
final-fence state
```

RAM save/restore 必须保存的是**episode 语义**，不能只恢复 server slot。

最低门：

```text
fork
→ children running
→ demote
→ RAM save
→ physical slots 被其他请求占用
→ restore 到不同 physical indices
→ 继续生成
```

结果必须与 uninterrupted reference 相同。

checkpoint / partial rollback 则要特别覆盖：

```text
snapshot plane
child native recurrent
PUBLIC brain
PRIVATE root brain
F32 hand
conv
```

context shift 必须理解成：

```text
logical history deletion
```

而不是 Tri eviction。

shift 后要同步更新：

```text
document run coordinates
reader PAC-DFS view
KV positions
epochs
MTP invalidation
```

这部分通过后，RERoT 才真正能跑长会话，而不是只能一次性回答。

---

# Phase 6：Server 外层语义矩阵

核心算法正确之后，再证明它能和整个 server 共存。

正式矩阵至少包括：

| 能力                | 必须证明                        |
| ----------------- | --------------------------- |
| multi-person      | 多独立请求不串状态                   |
| B > P             | queued child + pen fairness |
| `n_cmpl > 1`      | 外层 completion 不互相污染         |
| shared-prefix     | prefix refs 与内层 RERoT 正确组合  |
| idle demotion     | 可恢复                         |
| active preemption | victim 后继续                  |
| cancellation      | 无 orphan                    |
| retry             | 不重复 token                   |
| streaming         | 无重复 delta，恰好一个终止事件          |
| tool calling      | final fence 后恢复用户 grammar   |
| JSON schema       | 同上                          |
| LoRA/aLoRA        | lineage/adapter 不串          |
| multimodal        | shared prelude 后安全 fork     |
| embedding/rerank  | 全局开 RERoT 时完全绕过             |
| graph reuse       | input lifetime 不 stale      |
| pipeline parallel | span/frontier buffer 生命周期正确 |

这里最重要的一条不变量还是：

> **scheduler 的行为不能改变数学。**

换 slot、排队、preempt、不同 batch packing，只能影响**什么时候运行**，不能影响**算什么**。

---

# Phase 7：生产级压力 / auto-fit

到这里才开始真正“折磨服务器”。

规划指南里最有价值的终极压力形状其实已经写好了：

```text
RERoT
+ 6 outer slots
+ recursive forks
+ queued children
+ Tri pressure
+ Turbo4/Turbo2
+ MTP
+ streaming
+ 至少一次 RAM demotion/restore 或 active preemption
```

一次测试里同时让这些机制发生。

验收必须全部满足：

```text
0 5xx
0 OOM
0 deadlock
0 Vulkan validation error
0 orphan seq ref
0 orphan recurrent cell
0 duplicate SSE
每请求 exactly one terminal event
hard abort / natural final 可明确区分
```

然后单独压 shared-prefix：

```text
8K common prefix
├ A
├ B
└ C
```

保证 physical cell union 正确。

最后核 auto-fit。

当前 auto-fit 不只是：

```text
KV
```

而是必须同时为：

```text
B people
P pens
K physical KV
RERoT hand/brain
DDVR scratch
Tri scoring/pack scratch
MTP scratch
Vulkan temporary buffers
```

预留空间。

如果 auto-fit 算出来能装，真实运行就不能再 OOM。

---

# Phase 8：质量总验收

这时不要只测 3 个 demo。

我会冻结一组多层级质量集。

确定性微题：

```text
9.11 vs 9.9
整数运算
简单逻辑
短事实
```

必须 100% 正确。

代码：

```text
Python
C++
算法题
小型 repo QA
```

至少做：

```text
compile / syntax
unit tests
结果正确
```

数学正式 benchmark：

```text
MATH-500
AIME24/25
```

长上下文：

```text
needle/retrieval
长文总结
大洲国家
多章节问答
```

真实生产 prompt 样本也要进入。

每题都保存：

```text
prompt
seed
config
answer
reasoning
usage
artifact hashes
```

不要只报一个平均分。

这里比较的重点不是“RERoT 必须逐 token 等于 serial”，那不现实；而是：

> **RERoT 不能因为并行架构造成任务质量断崖。**

阈值要在跑 benchmark **之前**写死，不能看到分数后再改门槛。

---

# Phase 9：性能阶段正式开始

只有前面全部绿，我才允许大规模性能改动。

正式 production 配置固定：

```text
Ornith 1.5 35B
Vulkan
Turbo4/Turbo2
Tri 3/32
MTP ON
RERoT STRONG
production auto-fit
```

然后测：

```text
serial tok/s
RERoT aggregate tok/s
parallel model tok/s
prefill tok/s
p50 latency
p95 latency
VRAM peak
frontier barrier cost
attention cost
recurrent cost
MoE cost
MTP acceptance
```

之前的目标 **>= 500 tok/s** 应该明确成：

> **标准化多-Lane workload 下 aggregate model throughput >= 500 tok/s，且不能通过少生成、改 prompt、降低质量或取消机制获得。**

性能优化顺序我会遵循“收益/风险比”。

先动：

```text
不必要 synchronize
graph rebuild
metadata upload
small dispatch
重复 gather/scatter
```

再动：

```text
multi-reader DDVR 的 shared physical KV scan
一次 K/V load 服务多 Q reader
Turbo dequant reuse
frontier batch packing
```

再看：

```text
MTP acceptance / draft scheduling
MoE route reuse
kernel fusion
```

任何 MoE route cache 优化现在都必须过刚新增的 18/19 hermetic test。

性能阶段最重要的纪律：

```text
每一个 perf patch
→ fixed-tape numerical gate
→ RERoT CTest
→ semantic microset
→ 再看 benchmark
```

不能攒 20 个优化最后才发现语义坏了。

---

# Phase 10：长稳 Soak

达到性能目标之后还不能 deploy。

至少需要一轮真正的 soak：

```text
连续 full-slot 请求
多次 fork / close
Tri drain / maintenance
MTP accept / rollback
RAM save / restore
preemption
context shift
shared prefix
cancel / retry
stream
```

重点监控：

```text
CPU RSS
VRAM used
allocated buffers
recurrent used rows
KV refs
people/pens
orphan count
Vulkan errors
```

warmup 后内存必须进入平台，不得持续上涨。

最终要证明：

```text
无 leak
无 stale tensor
无死锁
无偶发 nondeterminism
无服务重启
```

---

# Phase 11：Release Candidate Artifact Sealing

到这一步开始禁止“我本地跑的是 A，部署时变成 B”。

每个 RC 保存：

```text
Git HEAD
git status
compiler
CMake config
llama-server --version

SHA-256:
llama-server
libllama-server-impl
libllama
libggml
libggml-vulkan
libggml-cpu
libllama-common
libmtmd

model SHA/fingerprint
Tri calibration SHA
systemd unit
完整启动参数
```

然后在**同一 artifact**上跑 release gates。

如果 rebuild：

> 旧验收立即失效。

这正是 build 10794 那次给项目留下的最大教训。

---

# Phase 12：Shadow → Canary → Production

最后发布不要直接覆盖 production。

先 shadow：

```text
同 production model/config
不接真实用户流量
全 release smoke
```

再 canary：

```text
少量真实请求
重点观察：
5xx
OOM
finish
latency
VRAM
RERoT metrics
Tri metrics
MTP acceptance
```

确认没有异常后再全量。

production 启动后重新核：

```text
/proc/PID/maps
```

确保实际加载的是刚刚验收的 `.so`，而不是某个 `/opt/llama/lib` 旧副本。

最后跑：

```text
health
9.11
代码题
长题
stream
tool call
并发压力
```

并保留上一 RC 的完整 rollback artifact。

---

# 最终什么才叫“完整交付”

规划指南现在给的是 **30 个 production DoD 硬门**。我把它浓缩成六句话：

```text
数学正确。
同一个 logical computation 不因 backend/batch/scheduler 改方程。

语义正确。
长短任务都能自然 fork/close/fence/finalize。

生态兼容。
Tri/Turbo/MTP/RAM/context-shift/preemption/shared-prefix/tool/stream 全工作。

资源正确。
auto-fit、压力、fallback、恢复都不 OOM、不丢状态。

性能达标。
标准 production workload aggregate >= 500 tok/s，质量不退。

工程可发布。
长稳通过，artifact checksum 锁死，shadow/canary/production 同一套二进制。
```

达到这里，才可以把标签从：

```text
RERoT research/core correctness candidate
```

正式改成：

```text
atomic-llama-cpp-turboquant
production-compatible RERoT
```

而从今天 `7509335d1` 的位置看，**最危险的“数学到底对不对 / Vulkan 为什么会生成垃圾”阶段基本已经过去了。下一段最关键的是 Phase 1 的生产上下文长题闭合；它一旦通过，就应该立即转入兼容矩阵，而不是继续打磨 prompt。**
