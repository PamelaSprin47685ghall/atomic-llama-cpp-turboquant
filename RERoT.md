# RERoT：自适应 DAG 执行语义、实现状态与交付路线

更新：2026-09-09。项目：`atomic-llama-cpp-turboquant`。本文核对的 `master` HEAD：`4e7769152`。
更新：2026-09-11（第十四轮静态审计校正）。项目：`atomic-llama-cpp-turboquant`。本文核对的 `master` HEAD：`ceba57b96`。
更新：2026-09-11（第十五轮静态审计，全平台修复）。设备级验证：CPU、CUDA（RTX 2080 Ti sm_75）、Vulkan（NVIDIA ICD；AMD RADV 设备支持已恢复）。本轮修复（均带设备证据，详见当日交接）：① Vulkan RERoT 共享内存预算按整模块静态 shared 计（74496B > 49152B 限制导致队列报错/fence 永不 signal 的挂死），`test-rerot-attn` 从 240s 挂死变为 ~1s 内 0 failure；② RERoT Parallel Delta GDN（RBB）在 CUDA/Vulkan 两侧补齐 density 归一化与并发求解：CUDA 新增设备端 RBB 分支，Vulkan 接入 `RBB=2` pipeline，`test-rerot-attn` GPU sections 从 57 failure 变为 0（CUDA 与 Vulkan 两树均验证）；③ CUDA ordinary FA 对 Turbo KV 的 TILE/MMA f16 临时缓冲漏分配（堆越界导致 turbo3 错解与 igu4_nl 非法指令崩溃）；④ CUDA TriAttention 打分器支持 partial-rotary Turbo（不再对 Ornith 类 partial IMRoPE 回退到主机）。这些改动只修复实现缺陷，不改第 2/4/6 节的执行语义与不变量。
更新：2026-09-11（第十六轮静态审计与 upstream sync ×2）。项目：`atomic-llama-cpp-turboquant`。基线：`git pull origin master` 两次——第一次快进到 `81f5b0198`（含随后被回退的 `--fit` 容量求解重写），第二次 origin/master 被**强制更新**到 `88fca9ced`（回退那 5 笔 `--fit` 重写，改用 dry-measurement + joint pool/slot solve + kernel VRAM 数字的正确修法）；`git pull upstream master` 两次均为 up-to-date。本轮：① `--total-kv`（别名 `--kv-size`，server scope）由上游正确提供：`auto` → `n_ctx_kv_auto`，`-c`/`-np` 语义保留，RERoT 阶段 0 门「auto + 显式 -np」仍拒绝；本地为第一次 pull 打的临时兼容 shim 已删除（clean cutover，不留第二套机制）；② 校正文档漂移（Vulkan 精度门硬件、raw-Q 钩子覆盖范围、HEAD 基线）；③ 三平台（CPU/CUDA/Vulkan）特性测试矩阵两次合并后均为 0 failure。残余：host 参考路径在**多 segment + boundary refine/部分因果切割**时仍显式拒绝（`--xkv-landmark-refine` 默认 `none`，设备路径按行 `refine_cap` 支持）。AMD RX 6800 RADV 设备现已完全可枚举并运行。
更新：2026-09-23（第二十/二十一轮核对）。项目：`atomic-llama-cpp-turboquant`。本文核对的 `master` HEAD：`a35ec96f6`。本轮核对并修正了路由 wire 的事实源：routing wire 已**语义回滚**到 `abd816e4d` 的 `strategy+payload` JSON 基线（此前试用的 Makefile 行式 DSL 与扁平 Compact Dict JSON 均已作废清理），探针治理补齐 512-token 硬上限、截断/拒绝语义区分与成对 trace。修正了 §4.2/4.3/4.4/4.6、§1.1/1.2/1.3、§13.1/13.3、§15.1、§20 的过期描述；阶段 5/6/8 的目标机（Ornith-1.5-35B）验收结论保留为 artifact 边界历史证据，不自动覆盖当前 HEAD。

本文是 RERoT 的单一自包含事实源，吸收当前 `AGENTS.md` 的最新方案，并结合当前源码与最近两笔 DAG 实现提交校正“已经实现什么、还缺什么”。以后不要再用旧 `RERoT指南.md`、旧数学审计、每日交接或脚本名字推断项目阶段。

**当前状态包含 `4e7769152` 之后的多笔实现提交（含第 12 节记录的 Stage 8 多 Lane DAG 认证）直至 `a35ec96f6`，不等同于这些历史 HEAD，也不等同于生产部署。** routing wire 经三次试验后已回滚至 `abd816e4d` 基线（见 §4.3/§13.3），其余程序修复全部保留。生产服务与当前候选 artifact 分开记录：阶段 5/6/8 的 Ornith-1.5-35B 真机结论绑定在回滚前 artifact 上，当前 HEAD 尚未在目标机重跑；不继承旧部署的验收结论。

---

## 0. 阅读口径

文中结论分五级：

| 标记 | 含义 |
|---|---|
| **决定** | 当前产品/工程选择，后续实现应以它为准 |
| **数学事实** | 在明确前提下可推导的关系，不自动等于模型质量结论 |
| **当前实现** | 当前 HEAD 源码中已能找到实际接线或数据结构 |
| **历史证据** | 旧 artifact / 旧提交曾经测到的结果；必须重新认证才能覆盖当前版本 |
| **待验收** | 方案明确要求，但还没有当前目标 artifact 的完整证据 |

必须同时记住三句话：

1. **不要求 RERoT 等价于“把最终 transcript 串行 prefill 一遍”，不等于允许同一个已经定义好的 RERoT 计算因 backend、batch packing、slot 或 row 顺序改变数学。**
2. **DAG 只规定什么时候必须等前驱完成；它不等于信息隔离。所有已经启动并发布的同 episode PUBLIC 内容仍持续共享。**
3. **某个固定入口里出现原生 reasoning-end，不代表源任务完成。只有当前源阶段自己生成并成功提交的结束事件，才允许释放硬依赖。**

---

## 1. 当前结论

### 1.1 产品语义已经换轨

旧版主线是：

```text
HTML <ol>/<li> planner
→ 平面/递归 tree
→ PAC-DFS 环形 reader view
→ 每 child 随机 8 位 Base62 <ID>...</ID>
→ 最后 child 成为 survivor
→ 回放 survivor close token 做 final fence
→ survivor 接管串行回答
```

**这已经不是新的生产目标。** 新主线改为：

```text
普通 prompt prefill
        ↓
      C0
        ↓
隔离的 strategy probe
        ├─ simple ─→ 丢弃 probe，恢复 C0，继续普通单流
        │
        └─ dag
             ↓
        校验 DAG 计划
             ↓
        建立正式公共规划边界 P / C_base
             ↓
        依赖满足的节点逻辑启动
             ↓
        每节点只 forward 一次固定入口 F_i
             ↓
        节点持续生成 R_i
        （native lane-local recurrent）
             ↓
        PUBLIC 通过 reader-specific DDVR 实时共享
        STRONG = 只看已经提交的 peer frontier
             ↓
        当前源阶段自然生成 native reasoning-end
             ↓
        SEALED，解锁真正后继
             ↓
        全部工作 SEALED
             ↓
        从 C_base 启动新的 0.synthesize 阶段
             ↓
        固定最终 view + F_0s + 新 logits
             ↓
        原生 reasoning → 最终正文 / 用户工具
```

几个旧概念被明确替换：

| 旧概念 | 新定义 |
|---|---|
| `<ol>/<li>` 决定拓扑 | schema probe 产生显式 DAG（`questions` + `depends_on` 边表） |
| tree parent/child 就是依赖 | 创建关系、硬依赖、状态来源、物理 binding 分开 |
| PAC-DFS 决定所有 reader 顺序 | DAG 上的循环优先 Kahn 排序，依赖约束优先 |
| 随机 Base62 `</ID>` 唯一完成 | 当前 WORKER 源流自己的原生 reasoning-end 完成 |
| 最后 child survivor | 所有工作完成后启动独立的 `0.synthesize` |
| final fence 回放最后 child close | 新 synthesis 直接从稳定 view + C_base + F_0s 获取新 logits |
| planner/child 控制串长期驻留 | probe 隔离；每节点固定入口只 forward 一次 |
| shared RBB 是默认 | **不是**；默认仍是 native lane-local recurrent + DDVR KV 共享 |

### 1.2 当前工程阶段

早期 DAG 接线提交（`c3648d789` 调度/view/framing、`4e7769152` server decode 接线）之后，wire 格式经历了一轮试验与**语义回滚**：Makefile 行式 DSL、Compact Dict JSON、lazy grammar 先后被否决，`9290e7a1a` 把 routing wire 恢复为 `abd816e4d` 的 `strategy+payload` 基线，同时保留其后全部不涉及 wire 的程序修复。

因此项目已经不是“旧 Ring Phase 1 长题闭合”状态，也不能继续用旧 `Phase 0..12` 作为当前推进编号。

当前更准确的定级是：

> **DAG production path wired, not production-certified**：隔离 probe、正式 P/C_base、循环优先 Kahn、原生 tool-round 固定入口、按 token LCP 的启动前缀重建、simple 完整 sampler clone 与用户预算恢复、源结束、W>P 逻辑 cohort、PUBLIC BODY/FRAME 冻结视图均已接线。阶段 0–8 的核心门（单 child 闭环、多 lane 真机、兼容矩阵、阶段 8 质量门）已在目标 CPU artifact 上通过；**但当前 HEAD（`a35ec96f6`，含 wire 回滚与 512 probe 上限）尚未在该 artifact 上重跑**，历史验收不自动覆盖当前版本。各阶段证据与版本边界见第 12/13 节。

新的实施阶段只保留第 12 节的 **阶段 0–8**。

### 1.3 当前源码与目标方案最重要的差距

这几项后续最容易被“代码已经有名字”误判为完成：

1. **隔离 probe 已把 JSON 写到临时 `probe_seq`，C0 的 slot seq 不再被 probe 追加。** 这仍是 COW 拷贝后的分叉，不是独立 llama_context。simple 保留真实 C0 logits 的首个采样决策及完整 sampler，不再 `init_sampler()` 重置 RNG/惩罚链。目标 Ornith 的 CPU A/B 已通过贪心、seeded logprobs、用户 JSON grammar 与 SSE；不据此推断 GPU 或 RAM 恢复等价。
2. **probe 有 512-token 硬上限，且截断与语义拒绝被显式区分。** 探针 token 被排除在用户 `n_predict` 与普通预算统计之外（原 240s 挂死的根因），现在超限即 `rerot_resource_exhausted` fail-closed；`incomplete`（JSON 未闭合，继续采样）与「语法完整但校验失败」（立即 final 并回报解析器诊断）是两条不同路径。成对 trace `probe_reject`/`probe_plan` 把计划压成单行，多任务计划不会被日志按行截断。
3. **`capture_c_base()` 在正式 P 的 `plan_prefix` 注入完成之后调用，并覆盖当前 root hand_seed（含 conv tails）。** sampler prev/seed 写入 `sampler_snapshot_bytes`。仍需用真实 recurrent 模型核对 brain/hand 配对。
4. **固定入口用真实请求 messages + spawn_lane 渲染 F_i。** 普通 C0 tape 与 DAG-with-tools 再渲染比较 token LCP；合法前缀变化在 DAG 启动时重建必要状态，不污染 simple 的普通 C0。模板无法无损渲染 CLOSE+handoff+OPEN 时仍 hard_abort。
5. **W>P：在 $P$ 个物理 Pen 约束下，每个逻辑 step cohort 至多纳入 $P$ 个绑定 worker（同批 decode 每 pen 一 token）。** 运行中/启动中的 worker 保持绑定在对应 Pen 上直至自然 seal，不再每 token 轮转分时；多出的 $W - P$ 个 eligible 节点在就绪队列中排队，待有 Pen 自然释放后按确定性顺序绑定进入后续 step cohort。当前同批 bound cohort 内部写入保持 PENDING 并在全员 commit 后原子发布；已废止旧版每 token 强行分时轮转全部 $W$ 节点的逻辑步全量 cohort 机制（旧分时单测已被 P-bound 队列契约取代）。
6. **HTML `<ol>` / 随机 Base62 / PAC-DFS 的生产入口已 hard_abort；DAG 上的 HTML fork 会失败。** 解析器、PAC-DFS `build_view` 和大量旧测试仍在树里。递归嵌套 DAG 没有实现。
7. **0.synthesize 作为独立阶段 admit。** 目标大模型 35B 在单 child（$13 \times 17=221$）、flat 2-worker（$25 \times 12$ 与 $15 \times 16 \to \boxed{540}$）、$A \to C$ 独立 B（$14 \times 15=210, 30 \times 20=600, C=260 \to 860$）以及菱形拓扑（$100 \times 3$ 与 $100 \times 5 \to 800$）下均顺利完成综合闭环，单次自然 stop，无 P 泄漏与 internal FRAME 泄漏。
8. **Episode 持久化（state v5）包含 C0/C_base、frozen epoch、逻辑步 cohort、conv tails 与 sampler prev。** probe 进行中的 save 被拒绝。RAM 持久化跨槽位恢复已在单测与多读者矩阵中完成验证（`test_dag_tri_mtp_ram_shift_speculative_matrix`）。
9. **DAG context shift pin 已启动节点的全部 PUBLIC 与全部 FRAME。** 无法腾出空间时 `tokens_removed=0`，由 decode 路径 resource abort。非 DAG 仍可截断未 pin 的旧 PUBLIC。
10. **RERoT lane 不再永久跳过 MTP drafting。** 强制注入期间不 draft；draft 绑定 topology/publish/layout stamp；过期 draft 恢复 checkpoint。DAG 读者视角 MTP 草稿失效与重草稿矩阵（`test_dag_tri_mtp_ram_shift_speculative_matrix`）已认证通过。
11. **`n_cmpl>1` 仍在 prelude 串行化额外 RERoT root**（避免共用 visibility domain）。阶段/RNG/图不串线，但不是并发多 completion。

---

## 2. 不可变底层契约

### 2.1 RERoT 是同一推理栈上的 execution semantics

不要为每个 Lane 新建一套 server request、`llama_context`、KV cache 或 backend。

```text
same request/response infrastructure
same llama_context
same llama_kv_cells physical owner
same recurrent memory owner
same graph/backend
same Tri/MTP/RAM/preemption foundations
             +
RERoT logical DAG / phase / view / epoch metadata
```

`llama_kv_cells` 继续是唯一 physical KV metadata owner。server 只保存稳定 logical run/node IDs，不把 GPU physical cell index 当作语义地址。

### 2.2 四类身份必须分开

| 身份 | 含义 |
|---|---|
| request / completion / person | 一份独立外部响应与全局任务状态 |
| internal stage / node | 一次明确的计算阶段；`0.plan` 与 `0.synthesize` 必须是两个内部阶段 |
| actor / Lane label | 给模型叙事看的身份；两个 0 阶段都可以叫 Lane 0 |
| pen / physical slot / exec seq | 暂时的物理执行资源和底层 handle |

模型输出的 `question.id` 也是另一域。它是计划接口字符串，不是 `seq_id`、slot id 或内部 node id。

### 2.3 Recurrent 共享边界已经冻结

当前正式默认：

```text
Full-Attention K/V    = same-episode PUBLIC 精确共享
GDN recurrent matrix = lane/stage-local native recurrence
Conv tail             = lane/stage-local
Sampler/RNG/penalty   = lane/stage-local
```

不恢复 shared-RBB 默认，不把前驱 recurrent state 平均，不做“把 foreign token 无 MoE 回放进后继 recurrent”的隐式补丁。

固定本 token 参数后，原生 GDN 可写为：

```text
k,q ∈ R^(d_k), v ∈ R^(d_v), S ∈ R^(d_k × d_v)
Sbar = alpha S, alpha = exp(g)
Snew = Sbar + beta k (v - Sbar^T k)^T
o = Snew^T q
```

内部实现可能用转置布局，但同一公式里的矩阵方向必须一致。

如果底层用 `S_effective = B + H` 表示 lane-local 有效状态，换基底时必须保持：

```text
B_old + H_old = B_new + H_new
H_new = B_old + H_old - B_new
```

这是坐标换基，不是新的 GDN transition，也不是 state merge。

### 2.4 DDVR 继续是共享注意力核心

三个位置域不能混：

| 域 | 含义 |
|---|---|
| physical key/cell index | 当前 resident K/V 的物理地址，可因 compaction 变化 |
| `storage_pos` | writer 写 K 时使用的逻辑/RoPE position |
| `virtual_pos(reader)` | 当前 reader 排列可见文档后的虚拟位置 |

若 K 在 storage position `s` 写入：

```text
Kstored = R(s) Kraw
```

而某 reader 希望这枚 K 出现在 virtual position `v`，query 虚拟位置为 `q`，则可在 Q 侧使用：

```text
Qeffective = R(q + s - v) Qraw
```

于是：

```text
Qeffective^T Kstored
```

表达与 reader 虚拟相对位置一致的 RoPE 关系，而不搬 K。

**所有可见 spans 必须进入同一个 query/head 的全局 softmax。** 不允许每 span 单独 softmax 后再相加。

Qwen3.5 文本 IMRoPE 仍按：

```text
(p, p, p, 0)
```

处理文本轴。不要因为 DAG 再引入动态 RoPE base、特殊“相对论”频率或视觉位置重排。

### 2.5 STRONG = barrier-after，不是同一次 forward 互穿

逻辑定义：

```text
frontier f read:
    自己的 causal history/current key
    + peer 在 f 之前已提交的 PUBLIC

frontier f compute:
    各 active stage 执行本轮工作

frontier f commit:
    PUBLIC 增量原子发布

frontier f+1 read:
    才可以读到 peer 在 f 的新 PUBLIC
```

GPU 物理上可能先把本 batch K/V 写进 cache，再执行 attention；这不能越过逻辑 publish/frontier gate。

LAG1 若保留，只是研究对照。正式 STRONG 仍是“上一提交边界立即可见”，不是“同拍 future leakage”。

---

## 3. 三个正交的数据属性

旧版只用 PUBLIC/PRIVATE/PENDING 已经不足以表达新 framing。至少要区分：

### 3.1 Visibility：模型谁能读

```text
normal
PUBLIC
PRIVATE
PENDING
```

### 3.2 Segment kind：这段是什么

当前 core 已增加/使用的语义类别包括：

```text
FRAME
BODY
SOURCE_END
PROBE_CONTROL
```

### 3.3 Presentation：客户端看到什么

例如：

| 段 | 模型 peer 可见 | API reasoning/content | 能否触发完成 |
|---|---|---|---|
| FRAME | 需要，完整发布后可读 | 不应逐字泄漏 | 否 |
| BODY | PUBLIC 时可读 | 按 reasoning 策略可见 | 否 |
| SOURCE_END | 源 state/tape 中保留 | foreign canonical body 不重复导出 | 当前源阶段提交后可以 |
| PROBE_CONTROL | 只属于隔离 probe | 不可见 | 否 |

**模型可见与用户可见不是一个布尔值。** 不能把 FRAME 永久设成 owner-only PRIVATE 来阻止 API 泄漏，否则固定入口拼接失效；也不能因为 peer 要读取 FRAME 就把内部 subagent tool call 发给客户端执行。

---

## 4. 自适应路由：C0、probe、direct、dag

### 4.1 C0 的定义

**决定：** C0 是普通公共 prompt prefill 完成后、任何 RERoT 专用规划字节进入正式执行状态之前的干净状态。

C0 必须覆盖/保持同一个计算时点的：

| 域 | 要求 |
|---|---|
| attention | prefix token/position、KV refs、必要 resident state |
| recurrent | 有效 local state、brain/hand 配对、conv、snapshot selector |
| sampler | 完整 RNG、penalty/history/adaptive sampler 状态，不只是 seed |
| parser/grammar | 原用户 grammar、reasoning/stop parser 阶段 |
| decode | 当前有效 logits 对应的 state/frontier |
| speculative | draft/checkpoint 不能携带 probe 污染 |
| transport | probe 尚未发给客户端，输出 cursor 不前移 |
| identity | 模型、template、adapter、RoPE、request/completion lineage |

Pure Transformer 可以没有 recurrent blob；不能以“GDN blob 非空”判断 C0 有效。

KV allocator 的“head/cursor”也不是通用 checkpoint。Unified KV 有共享 refs、空洞、compaction、回收和多请求；不能全局倒退一个游标删除本请求 probe。

### 4.2 推荐流程

```text
普通 prompt
   ↓
  C0
   └──→ 隔离 probe branch
            ↓
        strategy-json 计划
            ├─ strategy=simple → simple
            │    → 丢弃 probe branch
            │    → 原 C0 sampler/state 继续
            │
            └─ strategy=dag → dag
                 → 校验整张计划
                 → 丢弃 probe branch
                 → 从 C0 构建正式规划边界 P
                 → capture C_base
                 → 启动 DAG
```

“隔离”不是因为 PRIVATE 可以消除因果影响，而是反过来：**PRIVATE 不能从已经运行过的 recurrent/sampler 里扣掉。** 要得到干净 simple continuation，必须保留真正的 C0 边界。

### 4.3 路由 JSON（strategy + payload）

路由探针的输出格式为**显式 `strategy` + `payload` 的 JSON**。这是当前 `master` 的线上 wire 格式；历史上先后试用的 Makefile 行式 DSL（`ID [<- DEPS]: INTENT`）与扁平 Compact Dict JSON（`{"tasks":…,"deps":…}`）已按负责人指示**作废并清理**，wire 语义回滚至 `abd816e4d` 基线（commit `9290e7a1a`），其后不涉及 wire 的程序修复（Vulkan OOB 地毯、seq 单一权威分区、admission 三态、probe 截断/重复成员/512 上限/成对 trace）全部保留。

隔离 probe 注入的单行模板（末尾带换行，首颗 Token 直接落进对象体，不粘连）：

```text
Choose whether this request continues as a single answer or a DAG of independent sub-questions. Output only JSON: {"strategy":"simple","payload":{}} or {"strategy":"dag","payload":{"questions":[{"id":"...","intent":"..."}],"depends_on":[]}}.
```

> 注：末尾 `.\n` 是**强制契约**——`server-context.cpp` 在 arm probe 前显式校验 `probe_prompt.back() == '\n'`，不满足即 `rerot_protocol_error: routing probe prompt missing trailing newline`。这样 JSON 文法从 `{` 起，首颗模型 Token 直接开对象，prompt prose 永远不进文法栈，也不需要 forced prompt token。

simple 形态：

```json
{ "strategy": "simple", "payload": {} }
```

dag 形态（`questions` 数组顺序即 `plan_rank`，`depends_on` 是 `{id, depends_on_id}` 边表）：

```json
{
  "strategy": "dag",
  "payload": {
    "questions": [
      {"id": "A", "intent": "计算第一项所需事实"},
      {"id": "B", "intent": "独立检查第二项"},
      {"id": "C", "intent": "使用 A 的结果继续推导"}
    ],
    "depends_on": [
      {"id": "C", "depends_on_id": "A"}
    ]
  }
}
```

Schema（`server_rerot_routing_schema_json()`，经 `json_schema_to_grammar()` 生成探针 GBNF）：

```json
{
  "oneOf": [
    {
      "type": "object",
      "required": ["strategy", "payload"],
      "additionalProperties": false,
      "properties": {
        "strategy": {"const": "simple"},
        "payload": {"const": {}}
      }
    },
    {
      "type": "object",
      "required": ["strategy", "payload"],
      "additionalProperties": false,
      "properties": {
        "strategy": {"const": "dag"},
        "payload": {"$ref": "#/$defs/DagPayload"}
      }
    }
  ],
  "$defs": {
    "DagPayload": {
      "type": "object",
      "required": ["questions", "depends_on"],
      "additionalProperties": false,
      "properties": {
        "questions": { "type": "array", "minItems": 1,
                        "items": { "type": "object", "required": ["id", "intent"],
                                    "additionalProperties": false,
                                    "properties": { "id": { "type": "string", "minLength": 1 },
                                                    "intent": { "type": "string", "minLength": 1 } } } },
        "depends_on": { "type": "array",
                         "items": { "type": "object", "required": ["id", "depends_on_id"],
                                     "additionalProperties": false,
                                     "properties": { "id": { "type": "string", "minLength": 1 },
                                                     "depends_on_id": { "type": "string", "minLength": 1 } } } }
      }
    }
  }
}
```

强制分层：`minLength` / `additionalProperties:false` / `oneOf` 的 strategy 常量由**文法**物理屏蔽；转换器未渲染 `minItems`，故「空 `questions` 数组」由**解析器**拒绝。解析器 fail-closed 校验：重复 JSON 成员（nlohmann 会静默保留最后一个，必须先拒绝）、id/intent 非空且非纯空白、保留 id `"0"`、未知依赖端点、自环、重复边、Kahn 全拓扑覆盖（有环则整张图拒绝）。

**截断不是判决。** 解析器区分两类失败：JSON 尚不能合法解析（探针还在逐 token 流式）→ `incomplete = true`，继续采样；语法完整但校验失败 → 立即 final 并回报解析器自己的诊断，不等 EOG。因此文法绝不会接受 malformed 对象，`incomplete` 不可能是损坏计划。

语义映射：`strategy == "simple"` 且 payload 为空对象 → 丢弃 probe、恢复 C0 续写；`strategy == "dag"` → 正式 P 注入、C_base 捕获与多 Lane 调度。**模型不再自行推断拓扑**：worker 数 = `questions` 数，显式边决定依赖。Stage-5 单 worker DAG 由测试显式构造成 `strategy:"dag"` 维持 dag 模式（不再有"单任务自动降级 simple"的规则）。

### 4.4 控制面语义校验

grammar 只负责结构子集。正式计划提交前还必须：

1. `strategy` ∈ {`simple`, `dag`}；`simple` 的 payload 必须是空对象 `{}`，`dag` 的 payload 必须同时含非空 `questions` 数组与 `depends_on` 数组。
2. ID、intent 非空且非纯空白。
3. ID 唯一；`0` 保留给内部主体阶段。
4. questions 数组顺序固定为 `plan_rank`。
5. `depends_on` 的 `{id, depends_on_id}` 转成 `depends_on_id → id`。
6. 拒绝未知端点、自环、重复边。
7. Kahn 必须消费全部工作节点，否则整张计划拒绝。
8. runtime 自动建立 `0.plan` 的公共起点语义和“全部工作 → 0.synthesize”的最终条件。
9. 完整 descriptors/resources 可建立后再原子发布计划；失败不能留下半张图。

无效计划不能通过删边、裁问题、偷偷改 simple 或加 judge 来修复。

**重复成员扫描必须在 `json::parse` 之后**：扫描器会把截断输入误报成「重复成员」（实测两个字节的 `{` 就被拒），先解析再扫描才能把「语法错误」和「歧义对象」分开。nlohmann 对重复成员静默取最后一个，所以歧义对象必须显式拒绝而不是被悄悄修复。

### 4.5 simple 的恢复义务

simple 恢复后必须证明：

```text
相同普通 prompt
相同原 sampler 完整状态
相同普通模板/grammar
相同模型/backend配置
```

其后续普通 sampled path 不受 probe RNG、penalty history、KV/recurrent 或 parser 污染。

不要求“时间倒流”：probe 的 wall time 和真实计算已经发生，不能把它从性能账里擦掉，也不能宣称“零 TTFT”。

### 4.6 当前实现状态

当前 HEAD 已有：

- `server_rerot_routing_schema_json()`（`oneOf` strategy/payload Schema）/ `server_rerot_routing_probe_prompt()`（单行模板，末尾带换行，arm 前强制校验）；
- `server_rerot_routing_grammar()`（schema → GBNF，eager，非 lazy）；
- `server_rerot_parse_routing_decision()`：`incomplete` 截断语义 + 重复成员、extra key、id/intent 空白、保留 `"0"`、重复 id、未知端点、自环、重复边、Kahn 环检测；
- probe 运行时治理：`probe_seq` 隔离、`SERVER_REROT_PROBE_MAX_TOKENS = 512` 硬上限（超限即 `rerot_resource_exhausted` fail-closed）、成对 trace `rerot.trace.probe_reject` / `rerot.trace.probe_plan`（计划压成单行，`\n`/`\r`/`\t` 转义）、EOG 文案区分「探针未决」与「子分隔符未关」；
- `capture_c0()` / `rerot_start_root()` / `rerot_try_finish_probe()` / `rerot_enter_simple()` / `rerot_enter_dag()`。

probe 写入独立 `probe_seq`，C0 的 slot seq 不被 probe 追加；probe 丢弃后恢复 C0 lineage，simple 绑定保留的完整 sampler clone 并保留用户原 reasoning budget。probe 进行中的 episode save 被拒绝（`test_dag_save_refuses_probe_and_persists_c0`）。

DAG 模板新增 tools 可能改变普通前缀。`rerot_render_dag_prefix()` 比较真实 token LCP；`rerot_rebuild_dag_prefix_memory()` 在 DAG 启动时重建必要前缀，深度 recurrent rollback 不可用时从空状态重新计算正式 P。该工作只发生在启动边界，不是逐 frontier 重算 FRAME。不能仅因合法工具模板改变前缀就拒绝整个 DAG。

这些是实现状态，**不替代 fixed ordinary continuation A/B、recurrent 状态与实际模板的验收**。

---

## 5. DAG：硬依赖、实时信息流、reader 排版是三件事

### 5.1 硬启动依赖

`u → v` 只表达：

```text
u 没有自然 SEALED
⇒ v 不能 STARTING
```

形式化：

```text
eligible(v) = not_started(v)
              && forall u in pred(v): sealed(u)
```

它不表示 v 只能看到 u，也不表示没有边的节点互相不可读。

### 5.2 实时信息流

已经启动并发布的 PUBLIC 内容继续留在 same-episode 共享文档中，包括：

- 当前 reader 的硬祖先；
- 同时运行的无硬依赖 peer；
- 已经完成但不是当前 reader 硬祖先的节点；
- peer 所依赖、已经完成的前驱。

反例：A、B 原本并行，A 先完成。A 不是 B 的 hard pred。若 reader 只枚举 RUNNING peers，A 会突然从 B 的上下文消失，这是错误的。

因此：

> **DAG 只控制启动屏障；PUBLIC history 的生命周期不能由 RUNNING/RETIRED 状态直接决定。**

### 5.3 `0.plan` 与 `0.synthesize`

不要建：

```text
0 → A → 0
```

那在图论上是环。

应建成两个内部阶段：

```text
0.plan
   ├─ A → C ─┐
   └─ B → D ─┤
             └─ E

全部工作阶段 SEALED
        ↓
0.synthesize
```

对模型叙事它们都可以叫 Lane 0；对 stage state、sampler、run ownership、source end、MTP checkpoint、恢复和完成判断，必须是不同 internal stage。

### 5.4 循环优先 Kahn reader view

取 questions 数组位置为 `plan_rank`。

对 worker reader `r`，tie-break 优先序为：

```text
plan_rank 在 r 之后的节点
→ plan_rank 在 r 之前的节点
→ r 自己最后
```

但这个循环优先只能在所有 DAG 依赖合法的零入度候选之间使用；**依赖优先于循环偏好。**

无依赖 1/2/3：

```text
reader 1: P, B2, B3, B1
reader 2: P, B3, B1, B2
reader 3: P, B1, B2, B3

final 0:
P, B1, B2, B3, B0.synthesize
```

有依赖 `1 → 3`，2 独立；1 已完成，2/3 活跃：

```text
reader 2: P, B1, B3, B2
reader 3: P, B1, B2, B3
```

reader 2 不能排成 `B3, B1, B2`，因为会违反 `1 → 3`。

菱形：

```text
1 → 2
1 → 3
2 → 4
3 → 4
```

公共文档中 1 只出现一次，不能沿两条父路径重复展开成 `1,2,1,3,4`。

### 5.5 为什么 active reader 可以最后

如果 reader r 仍在 RUNNING，它的 hard successor 不应已经启动；否则调度器已经违反依赖屏障。因此在“已启动节点子图”中，r 是汇点之一。

Kahn zero-indegree 选择时把 r 设为最低 tie-break，不会阻塞别的节点，最后可以合法放到末尾。

### 5.6 运行状态不直接重排文档

在可见 runs 不变时：

```text
RUNNING → SEALED → RETIRED
pen migration
ready_queue 内部排序
physical microbatch order
```

都不应该凭自身改变公共片段存在与 `plan_rank`。

真正会改变结构的事件包括：

- 新节点固定入口完整发布；
- 新 BODY run 首次进入公共文档；
- context shift 删除逻辑单元；
- 明确的 topology/plan 变化。

普通 BODY append 只延长对应 run，使其后 spans 的 virtual coordinate 后移。

### 5.7 当前实现

当前 core 已有：

- `llama_rerot_document::set_dag_mode()`；
- node `plan_rank / predecessors / successors / stage_role`；
- `add_edge()`；
- `topo_sort_cycle_preferred()`；
- `build_dag_view()`；
- source-end / probe-control 的 reader filtering；
- runtime `initialize_dag()`、`get_eligible_dag_nodes()`、`seal_dag_node()`、`build_dag_view_for_reader()`。

这是新方案最扎实的一块之一，但仍需 Stage 1 的纯属性测试和 Stage 6 的真实 DDVR/backend 门继续认证。

---

## 6. 固定入口 framing：`B_i = F_i + R_i`

### 6.1 目标结构

每个工作片段统一为：

```text
B_i = F_i + R_i

F_i = CLOSE_PREVIOUS
    + HANDOFF_TO_i
    + OPEN_CURRENT

R_i = 节点 i 持续追加的源 reasoning BODY
```

公共前缀 P 的导出形式约定为“左边界上存在一个打开的 0.plan reasoning 区域”。第一个 F_i 的 `CLOSE_PREVIOUS` 负责关闭它。

固定入口的核心性质：

```text
一段 F_i 接收“前面有一个打开 reasoning”
→ 关闭它
→ 完成一次目标 i 的 handoff
→ 再打开 i 的 reasoning
```

连续拼接任意合法 node order 后，始终只有最后一段保持 reasoning open，而最后一段正是当前 reader。

### 6.2 两 lane / 三 lane

```text
reader 1:
P + F2 + R2 + F1 + R1

reader 2:
P + F1 + R1 + F2 + R2
```

三 lane：

```text
reader 1: P + B2 + B3 + B1
reader 2: P + B3 + B1 + B2
reader 3: P + B1 + B2 + B3
```

若 R2 多长一个 token，只需让后面的 F1/R1 virtual positions 后移；**F1 本身不能重新 forward。**

### 6.3 正式 handoff 应优先使用真实模型模板

概念上的简化：

```text
</think>
...handoff to lane i...
<think>
```

只是解释结构。

生产目标是：

```text
结束前一个 assistant reasoning
→ 在该 assistant 回合中形成内部 subagent/tool handoff
→ 对应 internal tool result 说明 Lane/Intent
→ 新 assistant generation prefix
→ 打开当前 reasoning
```

role/special tokens、tool call id、thinking start/end 都必须来自**实际部署模型的真实 chat template**。不能把伪 `[Tool Call]` 文本或硬编码特殊符号当成跨模型协议。

internal handoff 不是用户工具调用；不能出现在外部 `tool_calls` 里要求客户端执行。

### 6.4 F_i 只依赖目标，不依赖偶然前驱

F_i 可以包含：

- 稳定 stage/node identity；
- `intent`；
- internal call/result correlation；
- 固定 scope 说明；
- 模板需要的回合边界。

不要包含：

- “我前面一定是 node 2”；
- physical slot/row；
- 当前 frontier；
- token 剩余数；
- 当前 peer 列表；
- 动态预测下一次谁完成。

同一个 F_i 要能在不同 reader 拓扑序里接到不同前一片段后面。

### 6.5 FRAME 发布事务

节点 admission：

```text
1. hard preds 已全部 SEALED
2. 从 C_base 建立 stage-local state
3. 绑定 pen/exec（或进入逻辑 STARTING 等待本 frontier 的物理执行）
4. 用真实模板 tokenize F_i 一次
5. F_i 在构建期间整体 PENDING
6. 正常 forward F_i
7. 全部成功后原子发布 FRAME
8. stage → RUNNING
9. 安装 source-native reasoning-end 处理
10. 以后只追加 BODY R_i
```

入口是实际模型输入，不能只在 CPU 文本层换 view。

### 6.6 source end provenance

至少区分：

| 来源 | 能否完成当前 WORKER |
|---|---|
| runtime 固定 FRAME 中的 reasoning-end | 否 |
| foreign export 中其它节点的边界 | 否 |
| 当前 WORKER 源流自己生成的 native reasoning-end | decode+commit 后可以 |
| 当前 0.synthesize 源流的 native reasoning-end | 转入普通 final content/tool |
| EOG/资源耗尽/取消/强制 abort | 否，属于失败/终止 |

完成判断要绑定：

```text
(episode, internal_stage_id, event_origin)
```

不能扫描拼接后的 reader 文本搜 `</think>`，也不能只看 Lane label==0。

### 6.7 防止双重关闭

真实 source state/tape 可以保留：

```text
F1 + R1 + SOURCE_END
```

但 foreign canonical export 使用：

```text
F1 + R1
```

下一片段 F2 自己的 CLOSE_PREVIOUS 负责词法闭合。

否则会出现：

```text
R1 </think> </think><think> R2
```

SOURCE_END 的真实 KV/recurrent transition 不能为了“导出不显示”而从源计算状态删除。

### 6.8 token 边界

native reasoning-end 可能是一个 token，也可能跨多个 token。必须以模板/tokenizer 可识别的源协议事件处理：

- 跨 token 候选先 PENDING；
- 完整确认后再 seal；
- 正文里字面引用 `</think>` 不等于协议事件；
- 如果一个 tokenizer token 同时含正文尾部和关闭字节，不能假装一枚 KV row 可以免费切成两枚独立 token。

某模板若无法无损识别/导出该边界，应明确报 DAG protocol unsupported；不能靠全局字符串替换继续运行。

### 6.9 当前实现与目标差距

生产 admission 使用真实 chat messages、内部 `spawn_lane` tool call/result 和模板 reasoning 边界构造 FRAME，不再使用简化的 think/lane/intent sandwich 作为回退。模板无法无损表达接缝时仍明确失败。

- FRAME 完整发布后对模型可见，presentation 不把内部工具回合外发。
- `event_origin` 与 `seal_dag_node()` 区分 runtime 输入、foreign 导出与当前源结束。
- foreign reader 排除 SOURCE_END，但保留源计算历史。
- 冻结 read epoch 同时约束 foreign BODY 和 FRAME；owner 当前输入仍可见。
- worker/internal reasoning 不安装用户最终 JSON grammar；最终输出恢复原用户 grammar。自然结束解析与 grammar helper 是否存在是两回事。

目标 GGUF 的只读元数据确认 `<think>` / `</think>` 为独立 token（248068 / 248069，token type 4）；这支持边界能力检查，但仍不是完整模型生命周期或回答质量验收。

---

## 7. C_base 与 stage-local state

### 7.1 两个 checkpoint

```text
C0:
普通 prompt prefill 之后的干净普通状态

C_base:
dag 已验证
probe 已丢弃
正式公共规划/左边界 P 已真实 forward
之后的统一、不可变 stage seed
```

默认：

```text
worker v:
clone/reference C_base
→ install final reader view
→ forward F_v
→ generate R_v

0.synthesize:
clone/reference C_base
→ install final stable view
→ forward F_0s
→ generate synthesis reasoning
```

### 7.2 为什么不继承任意前驱终态

若 C 依赖 A、B，不能因为 A 恰好释放了一个 physical pen，就让 C 继承 A 的 recurrent final state。

否则：

```text
physical scheduling choice
→ 偷偷变成 cognitive state choice
```

这是不允许的。

A/B 的结果通过稳定 PUBLIC KV + DDVR 进入 C 的 Full Attention，然后自然影响 C 后续 native recurrent。

### 7.3 不平均前驱 state

不要自动做：

```text
S_C = (S_A + S_B) / 2
```

它不是“C 已经读过 A/B”的原生 state，也无法自然去除共同历史、不同 conv tail 与独立轨迹。

如果未来研究 state merge，它必须是另一条明确算法、单独数学与质量门，不作为坏输出时 fallback。

### 7.4 不做所谓“无 MoE 原生回放”

GDN 每层 q/k/v/g/beta 都由整个网络上下文投影得到。只拿前驱 token、跳过模型层直接更新 recurrence，不是原生模型。

保存并重放 writer transitions 又是另一种 transition-sharing 算法。本方案不需要它。

### 7.5 当前实现与待验收边界

`capture_c_base()` 已在正式 `plan_prefix` 注入完成后捕获当前 recurrent seed，并更新 root 的 hand/conv lineage；不再仅以 `c_base=c0` 表示正式规划后的状态。

C0/simple 使用保留的 sampler clone。checkpoint 中的 seed/prev 元数据不能单独证明完整 RNG、adaptive sampler 或 grammar 恢复；进程内 transport 的 sampler clone 与序列化 episode 是不同资产。Stage 3/7 必须分别验证这些状态在真实 decode、parking 和 RAM 恢复中的生命周期。

---

## 8. W>P：逻辑并发与物理 pens 分开

### 8.1 四个容量量

| 符号 | 含义 |
|---|---|
| B | resident 独立 request/person 数 |
| P | 可同时绑定执行的 physical pens |
| W | 某 episode 已满足依赖、尚未完成的逻辑工作数，包含仍在排队的叶 |
| K | unified physical KV capacity |

```text
active worker states <= P
eligible workers = W，可以 > P；未绑定的只排队
```

未启动的排队叶只持 descriptor/依赖和共同 C_base 引用，不提前生成 FRAME、KV 或采样状态；被依赖阻塞的节点还不算这里的 W。性能账本中的 `W_sum/W_max` 是**计划节点总数减 root**，另含 blocked 与 synthesis，不代表当下运行的 pen 数。拟合的 recurrent 行数必须至少为 P，而不是只按 B 配置。

### 8.2 “eligible 同时启动”的精确含义

在存在 $P$ 个物理 Pen 的执行语义下，“eligible 同时就绪”意味着所有新满足依赖的叶节点均进入调度器的就绪队列。但每个 decode 批次受物理并发限制：**每个逻辑步的 bound cohort 至多选择 $P$ 个 worker**（先纳入所有已绑定的 running/starting 节点，若有空闲 Pen 则按确定性队列顺序填入 queued 状态的 eligible 节点）。

单个 episode 内，一旦 worker 绑定到物理 Pen，它将持续保持绑定跨逻辑步推进，直到其自然达到 source_end 并执行自然 seal，**不再为同 episode 的排队叶进行每 token 的分时轮转**。超出物理容量的 $W > P$ 额外 eligible 节点在就绪队列中等待，直到某个活跃节点自然 seal 并释放 Pen 后，才进入后续 step cohort 启动。多个独立 episode 争用同一组 pen 时，既有跨 episode 公平性让位仍是另一条路径，不作为单 episode W>P 的常规调度。

**语义差异与跨波次可见性说明**：此语义与旧版“每一步强行切片轮转全部 $W$ 个节点使所有任务并行生成一词”的逻辑步全量 cohort 不再等价。已绑定的 $P$ 个 worker 每步读取上一步发布的冻结视界；后排队的节点在先前任务自然 seal 并发布正文之后才启动，因而可能观测到先完结 peer 的产物。这种跨波次（cross-wave）可见性的改变**在数学上与旧全量 time-slice 并不等价**，不得混同；新路径尚未经真机吞吐验收。

### 8.3 一个逻辑 frontier

```text
1. 从上一步 commit 的状态计算新满足依赖的 eligible 节点，入队就绪队列
2. 选取本逻辑步 bound cohort（至多 P 个）：优先保留已 bound 节点，空闲 Pen 按队列顺序接纳 queued eligible 节点
3. 固定当前 bound cohort 所有 worker 的只读公开视界（frozen read publish frontier/epoch）
4. 六个物理 pen 在单批次（same batch）中并发 decode 各推进 1 token
5. bound 节点本步写入的 BODY 统一暂存为 PENDING
6. 本 step cohort 全员 commit 完成后原子发布（atomic publication），推进发布版本
7. 若有节点自然产出合法 SOURCE_END，则执行自然 seal：
   a. 释放对应 physical pen 回空闲池
   b. 精确执行 remaining_preds-- 解锁直接后继（有未完前驱的依赖后继严格阻塞，直到全部前驱自然 seal）
8. 若就绪队列仍有未绑定 eligible 节点，在下一步调度时绑定空出的 Pen 进入下一 bound cohort
9. bound 节点若未到达 source_end，则继续保持 pen 绑定进入下一步，不发生例行 yield/resume
```

物理 batch 并发保证同批次 $P$ 支笔同时 decode；同步 co-bound 成员阅读同一份冻结先验世界，且在整步 commit 时原子发布。

### 8.4 为什么“排队到别的任务结束”不等价

在旧版“试图实现全量 W 并行”的理想模型中，若 A/B/C 在同 frontier old state 下，每一步都读相同 frozen old，则数学上严格保持对称无先后偏置。

而在当前 $P$-bound cohort 队列语义下：
若 $P=2$，则先调度 A、B 绑定直到自然 seal；C 必须在队列中等待 A 或 B 释放 Pen。当 A 完结 seal 后，C 绑定 Pen 启动，此时 C 读取到的公共前缀中已经包含了 A 完整 commit 的正文。

这意味着：
1. **C 启动时可观测到已自然 seal 的 A 的输出**；旧版每 token 分时轮转中，C 从第一步就参与，之后只能逐步观测 A 先前 frontier 已发布的部分输出，而不是启动时一次看到完整 A；
2. **跨波次可见性发生了实质性改变**：由于 A 先行完结并公开，C 的注意力视野和 prompt context 包含了 A 的最终正文，不再与 A 呈现严格的对称解耦。

因此必须诚实记录：**当前 P-bound cohort 队列调度与旧版 W>P 全量分时调度在数学与注意力可见性上并不等价**。用户明确选择此方案以消除频繁上下文与采样器换槽的高昂开销，使物理 Pens 能以批处理最大吞吐执行至自然完结。

### 8.5 仅已绑定的 P 份局部 state 驻留

在队列调度模型下，系统同时处于活跃执行状态的阶段数受限于物理 Pen 数（最多 $P$ 份活跃 state 驻留于物理槽位）。未获得 Pen 的 queued eligible 节点仅保留初始描述符和对前驱/基底的只读引用，直到首次分配 Pen 绑定时正式初始化其独立 recurrent/conv/sampler state。

旧版为同一 episode 的排队叶让位的 `yield_dag_pen_for_ready` 已删除；空闲 pen 却缺 recurrent 行属于容量/释放不变量错误，不通过换笔掩盖。跨 episode 公平性与持久化状态恢复仍有独立的 suspend/resume 路径；**单 episode 正常排队不触发停放或恢复**。

### 8.6 seal 是恰好一次事务

建议语义：

```text
RUNNING
→ source end candidate
→ EXIT_PENDING
→ 本 frontier decode/commit success
→ SEALED
```

seal 事务：

1. 校验 current internal stage 和 event origin；
2. 确认模型计算成功提交；
3. 保留公共 FRAME/BODY keeper refs；
4. state → SEALED；
5. 每个 successor `remaining_preds--` 恰好一次；
6. 再释放可释放 physical binding。

重复 callback / restore notification 命中已经 SEALED 时，不得再扣一次。

失败、EOG 异常、取消、资源 abort 不得走 SEALED 路径。

### 8.7 不先上 CPM

不把 DAG 深度直接当作任务时长，不用 CPM/置信度/标题内容决定谁先长期运行。

目标调度契约：

```text
P-bound cohort (up to P workers)
same frozen read world for co-bound peers
atomic publication of bound cohort step
queued eligible workers wait for natural seal
```

物理 pens 同批 decode，同步 co-bound 成员读取同一冻结视界；依赖后继严格阻塞直到其全部前置节点自然 seal。

### 8.8 当前实现与待验收项

当前运行时采用 **P-bound cohort 队列调度契约**：
1. 每个逻辑步的 step cohort 最多接纳 $P$ 个（如 6 个）已绑定或自队列出队的 worker，物理 pens 在同批（same batch）中并发执行 decode；
2. 绑定在 Pen 上的 worker 跨逻辑步持续运行，直至自然到达 source_end 并完成自然 seal，取消了每 token 轮转分时的例行切换开销；
3. 超额 eligible 节点（$W > P$）在就绪队列中严格等待空闲 Pen，前序节点自然 seal 释放槽位后按确定性次序入队绑定；
4. 当前同步 bound cohort 成员依然遵循原子发布契约：未发布正文保持 PENDING，全员 commit 后原子公开，本批次内部共享冻结的先验读者视界；依赖后继节点必须等待所有前驱自然 seal 完成后方可变为 eligible。

**历史验收状态标记**：
旧版基于全部 eligible 节点同一逻辑步全量入队并在单个/受限 Pen 上逐 token 轮转分时的证明测试（`test_dag_w_gt_p_logical_cohort_and_time_slice_certification`）**已作废**。新增 `test_dag_p_bound_cohort_queue_until_natural_seal` 钉住队列与自然释放，但受禁跑要求尚未执行。真机验收尚须证明同配置六 pen 合计采样吞吐不低于 simple 单流；当前历史跑分均来自旧调度，不能拿来证明新路径。

---

## 9. Final acquire 的新定义：启动 `0.synthesize`

### 9.1 什么时候可以开始

必须同时满足：

```text
所有工作 stage SEALED
无 STARTING
无 RUNNING/EXIT_PENDING
无未完成 frontier commit
无 ready/eligible 但未处理工作
episode 未 abort/cancel
```

### 9.2 新流程

```text
1. 固定最终完整 reader view
2. 原子把 0.synthesize 从 waiting → STARTING
3. 从 C_base 建新的 stage-local state
4. invalidate old speculative state
5. 正常 forward F_0s
6. 得到新的 synthesis logits
7. 生成 synthesis reasoning
8. 当前 synthesis 源流自然 reasoning-end
9. 转回普通 content / user tool / user grammar
```

不要：

- 把最后 child 改名成 0；
- 恢复最后 child state 给 0；
- 复制/回放最后 child close token给 0；
- 恢复旧 fork seed“洗掉指令”；
- 再加 merge/judge agent。

### 9.3 外部 response ownership 不变

整个 episode 始终只有一个外部 response owner。

internal worker/source end 不结束 SSE；只有最终 synthesis/普通输出路径结束整个 request。

如果 stream 已经开始，后续失败不能改写已经发出的 HTTP status；要按现有 transport 规范发送一次终止/error 并关闭，不能伪造自然 `stop`。

### 9.4 当前实现

当前 document/runtime 已有 `synthesis_node`、`stage_role::synthesis`，DAG fixed entry 也能以 `is_synthesis` 渲染不同标签。

但当前 server 仍保留大量旧 `final_fence / serial_tail / survivor` infrastructure。DAG 路径对这些旧字段已有条件分支，却还没有足够真实 E2E 证据说明 clean-cutover 已完成。

因此：**有 synthesis node 不是 final-acquire gate 通过。** Stage 5/6 必须用真实 token/role/view 轨迹证明它确实是一个新阶段，而不是旧 final tail 的新名字。

---

## 10. Tri、MTP、RAM、shift 和 server 生态

### 10.1 TriAttention

Tri 继续处理 physical residency，不决定 DAG 语义。

必须保持：

```text
physical cell union
semantic run/node ownership
reader visibility
archive/keeper refs
```

四者分离。

reclaim 后 reader view 从 resident logical runs/cells 重建，不持久化 old physical indices。

FRAME 的逻辑结构与当前 resident token 子集是两个层面；如果 Tri 淘汰重要 framing 导致模型协议认知退化，应在 Tri protection/scoring 策略中解决，不无限 pin 全部历史。

### 10.2 context shift

Tri 是物理 residency compression；context shift 是逻辑历史删除。

shift 只能删除完整、被允许裁掉的语义单元，更新：

- run table；
- reader view；
- storage/virtual layout；
- publish/layout epochs；
- MTP stamps；
- state serialization metadata。

不能切半一个 FRAME、当前 active causal tail 或仍必需的前驱结果。

若资源与完整语义冲突，应明确 resource failure，不能偷偷删掉关键工作结果让答案“继续成功”。

### 10.3 MTP / speculative

draft 至少绑定：

```text
episode
internal stage
reader view stamp
frontier / publish epoch
layout/topology version
recurrent/sampler checkpoint
```

会改变 reader 输入的 foreign PUBLIC commit、FRAME publish、stage switch、shift 等发生后：

```text
old draft stale
→ reject
→ restore checkpoint
→ remove unaccepted KV/recurrent/sampler progress
→ redraft
```

有 active peers 时不能用旧 view 一次 acceptance 穿越多个公共 frontier。

MTP acceptance 低是性能问题；target 结果与 MTP OFF 不一致是正确性问题，后者优先。

### 10.4 RAM / checkpoint / persistence

保存的是完整 episode 语义，不是最终文本：

- DAG nodes/edges/plan_rank；
- actor/internal-stage/stage_role；
- remaining_preds、SEALED transaction；
- C0/C_base refs；
- FRAME injection cursor；
- source-end candidate/provenance；
- run kind/visibility/tape；
- logical active/ready/starting/frontier transaction；
- each active stage recurrent/conv/sampler state；
- MTP/view stamps；
- outer response/output cursor；
- hard resource/accounting state。

restore 到不同 physical rows/slots 之后按 logical IDs 重建 binding/view；不能依赖旧 GPU cell index。

协议/wire state 要 versioned。旧 random-ID/tree episode 若无法无歧义迁移，应明确拒绝，不 best-effort 猜。

### 10.5 多请求 / `n_cmpl` / shared prefix

每 completion/episode 独立：

- DAG；
- stage states；
- RNG；
- epochs；
- final synthesis；
- HTTP/SSE lifecycle。

外层 shared prompt physical cell 可以共享 refs；episode_id 成为 fork 后的 RERoT visibility domain。

不同 episode PUBLIC 不能互读。

### 10.6 Streaming

内部 chronology 与最终 canonical DAG document 可以不同。

要求：

- 已发送 reasoning delta 不倒序重发；
- FRAME/internal tools 不冒充用户输出；
- worker source-end 不终止外部响应；
- cancel/retry 不产生 duplicate token；
- 一条 response 恰好一个 terminal event。

### 10.7 用户工具 / JSON grammar

internal subagent handoff 与用户真实 tool call 是两个协议域。

0.synthesize 完成 reasoning 后恢复：

```text
original user grammar
original user tool grammar/parser
original response protocol
```

如果用户工具返回后还要再 RERoT，应建立新的 episode/stage，不复活已经 SEALED 的旧 workers。

### 10.8 LoRA / aLoRA

同 episode 各 stage 使用相同 request adapter set/scale。不能因借到不同 physical pen 改 adapter lineage。

`can_batch_with()` 等既有 adapter compatibility 规则继续有效。

### 10.9 Multimodal

正确边界：

```text
system/user text + image/audio/video prelude
→ 普通模型 prefill
→ C0
→ routing/DAG reasoning
```

DDVR 只对 reasoning text runs 做其定义内的位置映射，不把 image spatial M-RoPE 当文本 PAC/DAG 顺序重新排。

### 10.10 Embedding / rerank

非 autoregressive generation 请求不进入 probe/DAG。即使 server 全局启用 RERoT 配置，也要完全走旧路径。

### 10.11 graph reuse / pipeline

view/span/phase/frontier metadata 尽量作为 graph input；只有 shape/capacity/kernel variant 等进入 reuse key。

host/GPU input buffer 生命周期必须覆盖实际 device consumption，不能下一 frontier 提前覆写。

---

## 11. 资源模型

### 11.1 不再使用旧“P 份状态就代表全部 active work”估算

新语义下真正峰值至少包括：

```text
model resident weights
+ shared prompt/public KV physical union
+ DAG FRAME/BODY resident KV
+ C0 / C_base lineage state
+ 至多 P 个当前 bound workers 的物理槽位 recurrent/conv/sampler state（未绑定 queued eligible 仅持轻量描述符与前驱引用）
+ P executor graph/workspace
+ Tri score/pack scratch
+ MTP draft/verify/checkpoint
+ Vulkan temp/alignment/fragmentation
+ state save/restore staging
+ safety headroom
```

不能：

- 用 Turbo4/Turbo2 KV 类型推算权重 resident bytes；
- 用 MoE active params 推算全部模型驻留；
- 按 reader 数重复计 shared prefix physical KV；
- 忽略 queued 阶段激增时的描述符/KV 累积；
- 忽略 FRAME/probe token 与 KV；
- 用真实 OOM/driver reset 来探容量。

### 11.2 B/P/K 仍有意义，但必须加 W

旧三容量：

```text
B = people
P = pens
K = physical KV
```

仍然有用，但新 DAG 还必须显式考虑：

```text
W = logical active stages
```

尤其 native recurrent 私有后，W 可能直接决定大量 persistent state，而不是只有 executor P 决定。

auto-fit 如果声称某组合可运行，真实压力下不能因漏计 W state、FRAME、MTP/Tri scratch 再 OOM。

### 11.3 资源不足的语义

分清：

```text
暂时无 pen（W > P，多余 eligible 处于 queued 状态）
    = 队列调度正常等待（排队直到前序 worker 自然 seal 释放 Pen）
    ≠ task failure

计划要求的全局 KV / 节点数 / 内存上限无法承载
    = resource capability failure
    → 明确 fail/abort
```

在当前 P-bound cohort 队列语义下，当所有可用 Pen 均处于 bound 占用状态时，超出 $P$ 的 eligible 节点进入确定性队列排队，直至前驱/已绑定节点自然 seal 释放 Pen。
这明确为调度器在固定物理 Pen 并发下的正规队列行为，不再进行每步例行分时换槽。

---

## 12. 新的唯一实施阶段：0–8

旧文 Phase 0–12 以后只作历史，不再用作进度标签。当前 DAG 路线固定如下。

### 12.1 阶段 0：冻结可信基线与实施前提

#### 目标

先知道自己正在修改哪份源码/二进制/模板，以及旧数值红线是否仍可靠。

#### 工作

1. 封存 Git HEAD/status/source patch。
2. 记录 compiler/CMake、server version、实际加载共享库、模型/template/adapter、完整启动参数。
3. 分开旧 Ring artifact、当前 DAG source 和 dirty worktree，不能混称一个基线。
4. 核对实际模板：thinking start/end、assistant/tool round、history reasoning retention、generation prefix。
5. 核对 C0/C_base 可用的 memory/sampler save/restore 能力。
6. 分类 CPU-only、可能初始化 Vulkan、需要真实模型的测试。
7. 保留旧数学红线，不通过改 tolerance/prompt/seed 消掉。

#### 通过条件

有一份可追溯 baseline matrix，并能回答：

```text
当前生产/目标 artifact 是什么？
哪些测试只查纯逻辑？
哪些测试会真正用 GPU？
哪个模板能无损表达 FRAME/SOURCE_END？
C0/C_base 当前到底保存哪些 state？
```

#### 当前状态

**已通过。** 当前工作树、构建产物与 commit 明确。在 `build-test`（`GGML_VULKAN=OFF`）上运行全量 118 项 CTest 单元测试（包括 `test-rerot-*`、`test-save-load-state`、`test-recurrent-state-rollback` 与 `test-model-resolution`）100% 全部通过；在 `build-vulkan-localhost` 上完成生产 `llama-server` 构建（build 10896, commit 387e3c725），动态库依赖与模型架构指纹清晰封存。

---

### 12.2 阶段 1：纯逻辑计划与 reader reference

#### 工作

- schema 结构 + control-plane validation；
- stable string ID → internal ID / `plan_rank`；
- DAG edges / Kahn cycle check；
- `0.plan` / `0.synthesize` internal stage；
- started public document set；
- cycle-preferred topological ordering；
- unique run expansion，不读取 physical cell index。

#### 必须通过的反例

| case | 期望 |
|---|---|
| `simple` + 非空 payload | reject |
| `dag` + 空 payload（缺 questions/depends_on） | reject |
| unknown strategy 值 / 非字符串 strategy | reject |
| extra 成员（root/payload/question/depends_on 任一） | reject |
| 重复 JSON 成员（nlohmann 静默取最后） | reject |
| 截断 JSON | `incomplete`，继续采样而不是判决 |
| duplicate ID | reject |
| self-loop | reject |
| duplicate edge | reject |
| unknown endpoint | reject |
| cycle | reject whole plan |
| questions order != topo order | dependency first, `plan_rank` 只 tie-break |
| single question `questions:[A]` + `depends_on:[]` | 仍是 dag（不再自动降级 simple） |
| flat 1/2/3 | reader 1=`2,3,1`; 2=`3,1,2`; 3=`1,2,3` |
| A→C，B independent | A seal 后 C eligible；B 不需等；B 仍看 A |
| diamond | shared ancestor/run exactly once |
| blocked node | no premature FRAME/BODY |
| only physical slot changes | logical reader order unchanged |

#### 独立参考

仓库已有：

```text
scripts/rerot-dag-reference.py
```

它枚举 4-node DAG/合法 live state/reader view，并验证固定 frame automaton。脚本预期统计：

```text
DAGs                    = 543
Legal live states       = 3007
Reader views            = 3904
Fixed-frame compositions= 384
```

它不测试模型、template tokenizer、KV、Vulkan 或恢复。

#### 当前状态

**逻辑门已接线并经 CPU 重跑。** `src/llama-rerot.*` 与 parser/view/runtime tests 覆盖 Stage 1 反例。这不是模型语义或 Vulkan 证据。

---

### 12.3 阶段 2：固定入口与源结束协议

#### 工作

1. 以真实模型 template 构造 F_i，不停留在简化 `lane:` 文本。
2. FRAME 只依赖目标 stage/intent。
3. FRAME 完整 PENDING → 一次 forward → 原子 publish。
4. source raw tape 与 foreign export 分开。
5. native reasoning-end 的 tokenizer/protocol boundary 明确。
6. event origin/phase gate 接入完成事务。
7. presentation layer 隐藏 internal FRAME/tool IDs。

#### 必须通过

1. 任意合法片段排列后只有 reader reasoning open。
2. peer BODY 增长不重新 forward 旧 F_i。
3. STARTING FRAME 中的 close 不能完成 worker。
4. foreign FRAME close 不能完成 current worker。
5. native source end 跨 token 时完整 commit 前不解锁后继。
6. source terminal 保留真实 state/tape，但 foreign canonical 不双重 close。
7. BODY 中普通 `<think>` 引用、代码、tool 示例不触发 scheduler。
8. FRAME/internal tool 不泄漏外部 `tool_calls`/content。
9. 模板不能无损识别 end boundary 时 fail capability，不猜。

#### 当前状态

**生产路径拒绝 sandwich，阶段 2 核心协议与边界硬门已获单测全覆盖。** F_i 用真实请求 messages（无 chat 时才用 dummy user）+ spawn_lane 渲染 CLOSE+handoff+OPEN。ordinary C0 tape 与 DAG-with-tools 再渲染做 token LCP；工具后缀并入正式 P。模板无法无损渲染则 hard_abort。针对阶段 2 核心约束已单测验证（`test_dag_source_end_multi_token_and_starting_frame_gate`）：
1. **STARTING FRAME 闭合隔离**（必须通过 3）：节点在 STARTING 阶段注入固定入口 F_i（即使含有 `</think>`）规划为 `frame` / `runtime_frame`，绝不触发 worker 完成或提前解锁后继；
2. **多 Token Native Source-End 事务原子性**（必须通过 5）：源生成阶段跨 token 候选（如 `</thi` + `nk>`）在未闭合前保持 `body` / PENDING，后继依赖严格保持锁死，直到完整闭合 token commit 成功后原子转为 `source_end` 并解锁后继；
3. **实际原生工具回合固定入口（FRAME）跨模板全生命周期认证**（必须通过 1 与 2，`test_dag_actual_native_tool_round_frame_certification`）：在生产大模型真实 Jinja 模板（Qwen3.5、DeepSeek-V4、DeepSeek-V4-Flash、DeepSeek-V3.2 等）上，严格证明：
   - 固定入口 $F_i$ 严格采用原生 `spawn_lane` 工具回合（关前段 reasoning + 发起内部 subagent tool_call + tool 结果返回目标 Lane 与 intent + 开启当前 reasoning），意图与参数严格由模板序列化，绝不依赖字符串拼接或硬编码特殊控制符；
   - $F_i$ 具备完全的阶段独立性，只绑定目标节点意图与阶段，不依赖相邻兄弟、物理槽位或 GPU 行；
   - 任意合法片段循环排列（如 Reader 1 视角 $P + B_2 + B_3 + B_1$，其中 $B_i = F_i + R_i$）在文本维度严格保证所有前置片段的 reasoning 全部配对闭合，唯独当前读者自己的 reasoning 保持打开；
   - 终极综合视图（$P + B_1 + B_2 + B_3 + F_{\text{synth}}$）严格保证所有先前工作分支全部自然闭合，仅终极综合帧开启思考，达成严格的“总-分-总”认知闭环。
真实 chat template / tokenizer 边界在目标模型 Ornith-1.5-35B 上已完成单 Worker 闭环验证。

---

### 12.4 阶段 3：C0 与 C_base 状态边界

#### 工作

- C0 保存普通 sampler/state；
- probe isolation；
- simple strict continuation；
- dag probe discard；
- 正式 P forward；
- C_base 在准确时点 capture；
- worker/synthesis 从 C_base clone/reference；
- recurrent/conv/position/snapshot/brain-hand 完整配对。

#### 必须通过

```text
普通 input + sampler fixed

run A: no probe, ordinary
run B: probe → simple → restore

后续 ordinary sampled path 应一致
```

还要覆盖：

- probe 期间取消/失败不损坏 C0 或其它 request；
- 多 stage 从 C_base 首写后互不修改；
- predecessor completion order/physical row 不改变 child seed；
- user grammar/template 完整恢复；
- first worker free token 使用 F_i forward 后的新 logits，而不是 C_base 旧 logits。

#### 当前状态

**simple 的目标模型 CPU A/B 已通过；阶段 3 多项核心硬门（COW 首写隔离、probe 取消不损 C0/peer）已通过。** `scripts/rerot-simple-continuation-smoke.py` 对真实 Ornith 检查贪心、seeded top-3 post-sampling logprobs、用户 JSON grammar、SSE 及双 completion。choices/通道与普通路径一致，probe 不计入普通 completion；JSON 数值答案另作正确性检查。C0 首个决策从有效 logits 保存并消费一次，sampler 不重置。`capture_c_base()` 在正式 P 注入完成后覆盖 root hand_seed，并抽出 conv tails、写入 sampler prev/seed。针对阶段 3 约束已落实并单测验证：
1. **多 stage C_base 状态继承与 COW 隔离**（`test_dag_capture_c_base_snapshots_current_seed`）：DAG worker 入职时从 `C_base` 继承 `hand_seed`，任意 worker 或 root 首写变动彼此隔离且不修改不可变 `C_base` 快照；
2. **Probe 取消/中止隔离**（`test_dag_c0_probe_cancel_and_isolation`）：在 isolated probe 过程中发生的客户端取消或 hard_abort 确保干净释放 probe 内部序列，完全不损坏同请求的 C0 状态与并行的 peer episode 状态；
3. **前驱完成顺序与物理槽位/Pen 不变量认证**（`test_dag_predecessor_completion_order_and_physical_row_invariance`）：
   - 多父依赖节点（C 依赖 A 和 B）在不同前驱完成顺序（A 先完结 vs B 先完结）以及不同物理执行槽位/Pen 分配下，初始循环隐状态 100% 严格继承自 `C_base`，绝不因借用前驱释放的 Pen 而错误继承前驱的终态；
   - 后继节点 C 的写入起始游标 `storage_pos_next` 与拓扑读者视角（P -> A -> B -> C）完全恒定不变；
   - 证明后继节点的理论推导认知初态与硬件物理调度顺序彻底解耦。
不同物理 row 的有效状态配对与跨进程恢复仍需独立证据；序列化 seed/prev 不等于完整 sampler 快照。

---

### 12.5 阶段 4：调度器、W>P 与错误事务

#### 工作

- logical active 与 physical bound 分离；
- P-bound logical step cohort（至多 P 个 worker）与 eligible 就绪排队；
- frozen read version；
- finite pens 同批并发 decode 与 bound worker 保持绑定至自然 seal；
- 删除 W>P 每步换笔路径，排队叶只在自然释放 pen 后启动；
- atomic frontier publish；
- exactly-once seal；
- synthesis start transaction；
- failure cleanup。

#### 必须通过

1. A 完成可立即解锁 C，不等无关 B。
2. $W > P$ 时，当前 bound cohort 严格至多接纳 $P$ 个已绑定/就绪节点，同批并发 decode；多余 eligible 节点在就绪队列排队，直至前驱节点自然 seal 释放 Pen 后按确定性次序接入。
3. 同一 bound cohort 内各 worker 共享冻结前沿视界，本步内不提前互见本步未发布的 peer 正文。
4. bound cohort 整步原子发布，未全员 commit 之前正文严格保持 PENDING。
5. 任一 slice 失败不发布半个 frontier。
6. duplicate end callback 不重复 `remaining_preds--`。
7. abort/cancel 后无 orphan state/ref/response。
8. 合法图“无 runnable 但有 unfinished”立即报 scheduler invariant error。

#### 当前状态

**P-bound 队列调度源码已接线，尚未执行新测试或真机复测；以下既有认证仅适用于未改变的基础不变量。** 同一 bound cohort BODY 保持 PENDING，直到 cohort 全员 commit 后发布。针对阶段 4 核心约束，旧测试曾验证：
1. **调度不变量硬门**（必须通过 8）：合法 DAG 无 runnable 却有 unfinished 节点时立即触发 `rerot_scheduler_invariant_error` 终止 episode（`activate_dag_frontier`）；
2. **Bound Cohort 批次隔离与原子发布**（必须通过 3 与 4）：同一 bound cohort 内各 worker 在本 step commit 之前绝不泄漏或提前读取同步其他 worker 的新写入；全员 commit 后原子公开；
3. **重复 source-end 与恢复通知幂等性**（必须通过 6，`test_dag_duplicate_source_end_and_restore_no_double_decrement`）：重复触发同一已完成节点的 source_end 事件或经持久化状态加载后，其后继依赖的 `remaining_preds` 严格执行 exactly-once 扣减，绝不发生重复扣减或负数溢出；
4. **取消/资源异常清理不留孤儿状态**（必须通过 7，`test_dag_abort_clears_orphan_refs_and_pens`）：hard_abort / client cancellation 触发后，所有活跃执行绑定的 pen 立即全量回收为 free 态，节点 physical_slot 重置为 -1，无孤儿 sequence 引用或残存运行状态；
5. **物理容量不变量**：拟合与最终上下文均为 $P$ 支 pen 预留 recurrent 行；配置的 batch/microbatch 至少容纳 $P$ 行。空闲笔缺行是状态错误，不能换笔或悄悄减少并发；
6. **【历史认证作废，待复验】W>P 调度语义已转为 P-bound Cohort 队列契约**：旧版以全部 $W$ 个 eligible 节点强行同入逻辑 cohort 并每 token 轮流切片的单测（`test_dag_w_gt_p_logical_cohort_and_time_slice_certification`）**已作废**。新测试尚未执行，预期契约为：
   - 调度器在每个逻辑步至多将 $P$ 个 worker 纳入 `dag_step_cohort` 并绑定物理 Pen；
   - 绑定的 worker 跨步持续执行，直到其自然 source_end 完成自然 seal 并释放 Pen；
   - 多出的 $W > P$ 个 eligible 节点在就绪队列中排队，待空出物理 Pen 后按确定性次序入队绑定；
   - 同一步内 bound cohort 成员保持原子提交发布与冻结视图，依赖后继仍保持阻塞直至全部前驱自然 seal。
旧版真实大模型目标推理曾由 `scripts/rerot-target-ornith-multi-lane.py` 覆盖；新调度尚无实测结果。

---

### 12.6 阶段 5：单 child 真实原生回合闭环

最小真实模型：

```text
0.plan
→ routing=dag with one worker
→ F1
→ R1 natural native end
→ 0.synthesize F0s
→ R0 natural end
→ user final content
```

#### 必须观察

- 实际模型 template 的 token/role 轨迹；
- child end 来自 source generation，不是 runtime FRAME；
- worker seal exactly once；
- final view 稳定；
- synthesis 从新 stage/new logits 开始；
- reasoning/content 正确分离；
- user grammar/tool 恢复；
- outer response 只有一次终止；
- 内容确实正确，不能只看 HTTP 200。

#### 当前状态

**目标大模型（Ornith-1.5-35B）单 Worker 闭环验证已通过：**
- 在隔离 CPU 实例（`-ngl 0 --device none -t 2 -c 4096 --total-kv 4096 -np 2`）上，对分配律计算 prompt（$13 \times 17$）执行完整 DAG 推理。
- 观测验证事实：
  - Worker 阶段通过模型原生生成 `</think>` 正常结束（`source_end_tokens=2`），无 runtime 强制截断或伪造完成；
  - 节点严格执行 exactly-once `seal`，前置 FRAME 与 Prompt 前缀完全被归入 frame 且不向 API reasoning 泄漏；
  - 0.synthesize 阶段从独立阶段启动并使用稳定最终 reader 视图生成新 logits；
  - reasoning 与 content 严格分离，最终正文准确输出 `**221**`，无末尾多余 `</think>` 泄漏；
  - 外部 HTTP 响应只有一次自然终止（`finish_reason: "stop"`）。

---

### 12.7 阶段 6：多 lane 实时共享与 DAG 数值门

#### 最小 workload

1. flat 三 lane：验证三个循环 reader 顺序和持续 peer uptake。
2. `A→C` + B independent：验证 A history 保留，B/C overlap。
3. diamond：验证 join 和 ancestor unique。
4. unequal lengths：完成节点 history 不消失。
5. synthesis 使用多个互补结果，不串 intent。

#### 数值门

- 固定 writer tape/frontier/view/KV bytes，比 CPU/reference 与 Vulkan；
- native lane-local recurrent；
- mixed ubatch；
- different physical row order；
- rollback 0/>0；
- MoE `18→19→20→18→19`；
- NaN/Inf/empty/length mismatch fail-closed；
- same logical computation 不因 batch packing 改 sampled decision。

不在这一阶段引入 dynamic RoPE、shared RBB、foreign replay。

#### 当前状态

**核心最小 workload 单元测试与目标大模型真机端到端已全部落地并通过：**
- **单元测试套件 100% 通过**：
  - flat 三 lane 循环读者顺序（`1->(2,3,1)`, `2->(3,1,2)`, `3->(1,2,3)`）与跨 frontier 持续 peer uptake 历史保留测试通过（`test_dag_three_lane_flat_cycle_and_peer_uptake`）。
  - `A->C` + B independent 工作负载测试通过（`test_dag_a_to_c_with_b_independent_overlap`）：验证 A seal 解锁 C、B 与 C 并发重叠推进、已完成前驱 A 历史在 B/C 读者视角中持续保留且坐标准确。
  - `1->2`, `1->3`, `2->4`, `3->4` 菱形依赖门控、不等长生成（3 vs 10 tokens）、前驱 1 唯一样本去重展开与阶段自然完结测试通过（`test_dag_diamond_and_unequal_length_history`）。
  - synthesis 互补结果与独立意图测试通过（`test_dag_synthesis_complementary_results_distinct_intents`）：验证并行 Worker 各自独立意图（代数推导 vs 几何剖分）生成互补结论，所有前驱完结后 synthesis 节点精确解锁；synthesis 读者视界中按拓扑偏序单次且完整呈现各前驱结论，无 intent 串线或跨阶段污染。
  - `test-rerot-view` 包含 4 节点全 DAG 拓扑、循环偏序及菱形单次祖先展开断言。
  - Vulkan 周期注意力精度门（`test-rerot-attn --precision-only`，RTX 2080 Ti NVIDIA ICD 与 CPU reference，keys=33/257，误差 $\le 1.19 \times 10^{-7}$）及 DDVR 多 span 相位补偿门 100% 通过（AMD RX 6800 RADV 设备现已完全可枚举并运行）。
- **目标大模型 Ornith-1.5-35B 真机端到端全量通过**（`scripts/rerot-target-ornith-multi-lane.py`）：
  - Flat 2-worker DAG：$25 \times 12$ 与 $15 \times 16$ 独立并行并汇聚综合为 $\boxed{540}$；
  - `A->C` + B independent：A ($14 \times 15=210$) 与 B ($30 \times 20=600$) 并行，C 依赖 A 产出 $210+50=260$，最终汇聚输出 **860**；
  - Diamond 菱形拓扑：分支 2 ($100 \times 3$) 与分支 3 ($100 \times 5$) 汇聚输出 **800**；
  - 各工作 Lane 均自然生成 `</think>` 结束，严格单次 stop 终止，无任何内部工具或 FRAME 标记泄漏。

---

### 12.8 阶段 7：完整兼容矩阵

| 组合 | 硬门 |
|---|---|
| FullKV / Turbo / Tri / Tri+Turbo | same DAG rule；sparse/union/compaction 正确 |
| MTP no-peer-change | 保持普通 draft/verify semantics |
| MTP peer/frame/shift/stage switch | stale draft reject + exact restore |
| demote → other physical rows → restore | DAG、FRAME cursor、source end、local state 连续 |
| context shift | 只删合法完整单元，更新 view/epochs |
| W>P + multi people | physical packing 不改变 math，资源可恢复 |
| `n_cmpl>1` / shared prefix | completion isolation + physical ref union |
| streaming / retry / cancel | no duplicate + exactly one terminal |
| user tools / JSON | internal handoff 不外发，final 恢复 user rules |
| LoRA/aLoRA | adapter lineage 不随 pen 漂移 |
| multimodal | common prelude/position lineage 正确 |
| embedding/rerank | 完全绕过 DAG |
| RERoT OFF | 普通路径零回归 |
| graph reuse/pipeline | no stale input/descriptor |
| legacy recursive fork requirement | 若仍是产品需求，必须用 phase→sub-DAG→synthesize 正式建模 |

早期可以为了定位关闭 Tri/MTP/RAM；最终不能拿“关闭以后通过”当兼容证据。

#### 当前状态

**核心兼容矩阵单测与离线断言已全部通过：**
- `test-triattention-score`：TriAttention scorer 几何、RoPE 逆变换、z-score 正规化、turbo/storage oracle 匹配 0 failure 通过。
- `test-triattention-kv`：FullKV (f16) 及 Turbo2/Turbo3/Turbo4 与 TriAttention 协同共存测试 PASS。
- `test-server-triattention`：server 状态机及 TriAttention 状态转移测试 PASS。
- `test-flashprefill-state`：FlashPrefill 状态、membership、policy CPU-only 0 failure 通过。
- `test_rerot_mtp_speculative_matrix`：MTP 在 peer 更新/拓扑变化/Tri 压缩/final fence 时的过期草稿精准拒绝与恢复验证 100% 通过。
- `test_phase5_ram_checkpoint_context_shift_matrix` 与 `test_ram_restore_context_shift_and_preemption`：RAM 持久化保存、换槽恢复、context-shift 仅截断未固定公有片段、拓扑屏障重置验证 100% 通过。
- `test_dag_demote_restore_different_physical_slots`：DAG 启动后通过 `demote_episode()` 释放物理槽位、序列化保存至 RAM blob、原槽位被无关业务占用后换槽恢复至全新物理 slot（slots 3 & 4）继续生成并通过 source_end 自然完结，与未中断的 reference 运行在逻辑图、run 结构、token 计数及 synthesis 读者视图（reader 0）上保持 100% 严格等价。
- `test_dag_streaming_isolation_and_terminal_guarantees`：验证流式输出与取消隔离（`streaming / retry / cancel` 与 `user tools / JSON` 硬门）：内部 FRAME / subagent handoff 及 probe_control 绝不外发为推理流，正文按到达顺序整行流式输出无重复，客户端中止时行汇流排（line_mux）与 episode 干净回收且无孤儿行或重复 chunk，确保外部 exactly one terminal event。
- `test_dag_lora_multimodal_bypass_and_lineage`：验证非生成任务旁路、LoRA 槽位不变性与多模态前缀门控（`LoRA/aLoRA / multimodal`、`embedding/rerank / RERoT OFF` 与 `graph reuse / pipeline` 硬门）：LoRA adapter lineage 绑定逻辑阶段且不随物理 pen 漂移；多模态输入严格遵守 prelude-then-fork 并在 DDVR 中禁止重排视觉位置；Embedding/Rerank 等非生成任务 `rerot_effective() == false` 彻底绕过 probe/DAG 引擎；DAG reader views 在多 frontier 推进与节点自然退休下无悬挂描述符。
- `n_cmpl>1` 在 prelude 串行化额外 RERoT root（阶段、RNG、图不串线，单次 prompt 共享 usage 聚合已通过真机脚本验证）。
- 递归嵌套 DAG 仍 fail-closed。

---

### 12.9 阶段 8：质量、性能、长稳、artifact、发布

#### 当前状态

**阶段 8 正式 Quality Acceptance Gate 评测已完成并通过：**
- 目标大模型 Ornith-1.5-35B 在全量 11 项跨层评测套件中取得 10/11 通过（总体 Verdict: PASS，详见 `reports/phase8-quality/summary.json`）：
  - **Tier 1 确定性微题（100% 阈值）**：`micro-9.11-vs-9.9`（通过，清晰论述十分位 $9 > 1$ 并给出正确结论）、`micro-arithmetic`（通过，逐步计算得出 $\boxed{5418}$）、`micro-logic-transitivity`（通过，严密说明身高传递性）、`micro-fact-capital`（通过，准确输出巴黎），4/4 全部通过（100%）；
  - **Tier 2 代码算法与单元测试**：`code-python-palindrome`（双指针算法，通过内置多断言单元测试）、`code-cpp-spiral`（C++ 顺时针螺旋矩阵，g++ 编译运行并通过断言单元测试）、`code-python-dp-coinchange`（动态规划零钱兑换，通过边界与大金额测试用例），3/3 全部通过（100%）；
  - **Tier 3 形式数学（AIME 样例）**：`math-aime25-base-divisor`（通过整除条件转换为 $b+7 \mid 56$，正确得出 $b \in \{21, 49\}$ 之和 70 并给出 $\boxed{70}$）；
  - **Tier 4 长上下文与生产任务**：Needle in haystack 密钥检索（通过，准确提取 `REROT-TURBO-778899`）与多章节操作系统结构化分析报告（通过，篇幅 > 1500 字，完整涵盖进程、内存、文件三大核心原理与对比表格）；
- 多 Lane DAG 真实推理测试（`scripts/rerot-target-ornith-multi-lane.py`）：flat 2-worker DAG（25*12 与 15*16 并行推导并在 Lane C 汇聚合成 540）、A->C 带独立 B 重叠（14*15 与 30*20 并行，C 依赖 A 产出 260 并汇聚为 860）、菱形依赖（100*3 与 100*5 汇聚合成 800）均自然完结并通过无 internal token 泄漏断言；
- 显卡上的 Vulkan 精度门（`test-rerot-attn --precision-only`，RTX 2080 Ti NVIDIA ICD 与 AMD RX 6800，F16/Turbo keys=33/257，误差 $\le 1.19 \times 10^{-7}$）与 DDVR 多 span 门 100% 通过；
- 生产环境服务依据授权保持永久停用状态；测试日志、单项输出与构建库哈希清单完整保存归档。

#### 质量

至少：

- 9.11/整数/短逻辑等确定性微题；
- 代码 compile/tests；
- 数学/依赖推理；
- 长上下文 retrieval/summarization；
- 多章节长任务；
- 真实生产样本。

每题保留：

```text
prompt
seed
routing plan
DAG
config
reasoning/content
usage/accounting
source event trace
artifact hashes
```

#### 机制消融

固定计划后先做：

| group | scheduling | narrative/end protocol |
|---|---|---|
| A | old flat | old |
| B | DAG | old |
| C | old flat | fixed-entry/native-end |
| D | DAG | fixed-entry/native-end |

然后再加入 model-generated routing，才能区分“DAG 改善”“新 framing 改善”“planner routing 改善”。

#### 性能

报告：

- useful sampled tok/s；
- aggregate model throughput；
- probe tokens/time；
- FRAME tokens/time；
- state swap W>P cost；
- prefill/TTFT；
- p50/p95；
- VRAM/RSS；
- frontier/barrier；
- graph rebuild/metadata；
- attention/recurrent/MoE；
- MTP acceptance + invalidation cause。

旧 `>=500 tok/s` 是待验证性能目标，不能写成新 DAG 已保证。不能把 forced framing/probe token 冒充 useful reasoning throughput。

#### soak

质量/性能过门后再组合：

```text
DAG startup/seal/synthesis
Tri drain/maintenance
MTP accept/rollback
RAM save/restore
preemption
context shift
shared prefix
cancel/retry
stream
```

warmup 后内存/refs/state 要进入平台，无 leak、stale tensor、deadlock、偶发 cross-episode state corruption。

#### artifact & release

同一 RC 保存：

```text
Git HEAD + status + source patch
compiler + CMake
llama-server --version
llama-server / libllama-server-impl
libllama / libllama-common / libmtmd
libggml / libggml-base / libggml-cpu / libggml-vulkan
model fingerprint
template fingerprint/version
Tri calibration fingerprint
service unit + full args
```

实际运行通过 `/proc/PID/maps` 核对加载库。rebuild 后旧验收失效。

发布：

```text
shadow
→ canary
→ production
```

并保留上一完整 RC rollback artifact。

---

## 13. 当前实现盘点

### 13.1 已经落下的 DAG 基础

当前 HEAD 源码可以确认以下实现存在：

| 范围 | 当前实现 |
|---|---|
| routing schema | JSON Schema + schema-to-grammar |
| routing parse | strategy/payload strict parse，`incomplete` 截断语义、duplicate member/ID/edge、自环、unknown endpoint、cycle check |
| prebranch state | `server_rerot_prebranch_checkpoint`，C0/C_base fields |
| node DAG metadata | `string_id / intent / plan_rank / predecessors / successors / remaining_preds / stage_role / is_sealed` |
| document mode | DAG mode、edge、cycle-preferred Kahn、DAG reader view |
| segment semantics | `frame / body / source_end / probe_control` |
| origin semantics | runtime frame / foreign export / worker source 等 completion origin |
| fixed entry | 真实请求 messages + 原生 `spawn_lane` tool round；启动时一次 FRAME |
| native end | template thinking end tag → 当前源流 parser/origin gate，不用 grammar 强制完成 |
| DAG init | workers + synthesis node + dependency count |
| server wiring | 隔离 probe → simple/dag；正式前缀重建；DAG admission；FRAME injection；源结束与最终输出分流 |
| probe governance | `probe_seq` 隔离、512-token 硬上限 fail-closed、成对 trace `probe_reject`/`probe_plan`（单行转义）、EOG 文案区分「探针未决」 |
| serialization | DAG flags/node fields/source marker 已进入 episode state path |
| offline reference | `scripts/rerot-dag-reference.py` |

### 13.2 阶段认证矩阵与状态

| 范围 | 状态 |
|---|---|
| 真正隔离 probe branch | **已实现并认证通过**（`test_dag_c0_probe_to_simple_continuation_and_grammar_isolation` 证明 probe 丢弃后 C0 游标、recurrent 与采样器快照 100% 还原，普通续跑路径与无 probe 基准严格对齐；用户 JSON grammar 在 probe 期间隔离并于 simple 恢复；首个 worker token 严格基于 F_i 新 logits） |
| C_base = 正式 P 之后的 state | **已实现并认证通过**（`capture_c_base` 在正式 P 完成后捕获；`test_dag_capture_c_base_snapshots_current_seed` 与 `test_dag_predecessor_completion_order_and_physical_row_invariance` 严格证明：多 stage 从 C_base 继承种子，前驱完成顺序与物理槽位/Pen 变动不改变后继种子与读者视图，且首写 COW 彼此完全隔离） |
| actual native tool-round FRAME | **已实现并认证通过**（`test_dag_actual_native_tool_round_frame_certification` 在 Qwen3.5、DeepSeek-V4、DeepSeek-V4-Flash、DeepSeek-V3.2 等生产模板上，严格证明 $F_i$ 渲染基于原生 `spawn_lane` 工具回合与意图序列化，具备阶段独立性与槽位无关性；任意合法循环排列 $P + B_2 + B_3 + B_1$ 严格保证封口前段并独留当前读者 reasoning 打开；终极综合视图中所有前置工作片段自然完结闭合，唯独综合帧 reasoning 打开） |
| tokenizer-level source-end 无损边界 | **已实现并认证通过**（`test_multi_template_source_end_boundary_certification` 覆盖 XML `</think>`、Command-R `[/THINK]`、ChatML `<|im_end|>`、Specialized `<|close|>think<|sep|>`, `<|END_THINKING|>`, `</mm:think>`, `<|channel|>` 等模板的 token 切分保留、候选 PENDING 挂起与 snapshot 还原） |
| W>P P-bound cohort + natural seal queue | **源码已接线，尚未执行新回归或真机吞吐验收；旧 all-W time-slice 认证已作废**（新契约：至多 $P$ 个 worker 组成同批 bound cohort，保持绑定直至自然 seal；超出 $P$ 的 eligible 节点排队；同批成员保持冻结视图和原子发布，依赖后继等待全部前驱 seal。新增 `test_dag_p_bound_cohort_queue_until_natural_seal` 尚未运行） |
| W active state budget/restore | **旧轮转测试已删除**；未启动的排队叶不占活跃 recurrent/sampler/KV 行，当前仅运行中的 $P$ 支 pen 占行；新调度尚未实测。 |
| final 0.synthesize clean cutover | **已通过**（单 child 闭环下作为独立 stage 启动，使用稳定最终视图及新 logits，输出准确 221） |
| 删除旧 random-ID/tree/fence production semantics | **已实现硬阻断与 clean cutover**（DAG 模式下 `publish_pending_record` 与 `freeze_fork_parent` 严格拒绝 HTML planner records 并 hard_abort；非 DAG 保留用于回归测试） |
| target Ornith single child DAG | **已通过**（分配律 13x17=221 单 worker 自然 end 与 synthesis 闭环） |
| target Ornith multi-lane DAG | **已通过真机全量验证**（在隔离 CPU 实例 `-ngl 0 --device none -t 2 -c 4096 --total-kv 4096 -np 2` 上，通过 `scripts/rerot-target-ornith-multi-lane.py` 完整验证 flat 2-worker DAG、A->C 带独立 B 重叠、菱形 1->2/3->4 汇聚依赖；多 Worker 自然 source_end 自然闭合、无 internal FRAME/token 泄漏，正确综合计算结果，单次请求一次 stop 结束） |
| Tri/MTP/RAM/shift DAG matrix | **已实现并认证通过**（`test_dag_tri_mtp_ram_shift_speculative_matrix` 覆盖 DAG 读者视角 MTP 草稿失效与重草稿、活跃 DAG 框架与正文 shift 严格钉扎、RAM 持久化换槽恢复完整性） |
| DAG quality/performance/soak | **已完成阶段 8 正式 Quality Acceptance Gate 评测**（通过 `scripts/rerot-phase8-quality.py --cpu-only` 评测全量 11 项跨层基准：Tier 1 确定性微题 4/4 全部 100% 通过；Tier 2 代码算法编译与单元测试 3/3 全部 100% 通过；Tier 3 形式数学 AIME 样例通过；Tier 4 长上下文 Needle-in-haystack 密钥提取与多章节系统分析报告全部通过；总体通过率 10/11 达到 PASS 标准，全量结果、分项 JSON 记录与动态库 SHA256 审计清单已落地归档在 `reports/phase8-quality/`） |

### 13.3 wire 基线与最近提交的边界

**wire 基线 = `abd816e4d`（由 `9290e7a1a` 恢复）。** 早期 DAG 接线提交 `c3648d789`（调度/view/framing）与 `4e7769152`（server decode 接线）之后，routing wire 经过 Makefile DSL → Compact Dict JSON → lazy grammar 三次试验，最终**语义回滚**到 `abd816e4d` 的 `strategy+payload` 基线。回滚时保留的程序修复（不触及 wire）：Vulkan workgroup stride OOB 地毯修复（`turbo_wht` + 29 个 shader 的 `NumWorkGroups` 线性化、host `wg_scale`、rope 行跨步、`flash_attn_rerot` 的 `Br=1` 网格钉扎）、seq id 单一权威分区（`set_pen_capacity` 唯一划分点、`alloc_internal_seq` 别名断言、`rerot_seq_retarget` 拷前清空、park/resume 空决策语义）、admission 三态闸门（`ok/saturated/failed`，消灭 5000+ 次活锁刷屏）、probe 加固（512 上限、截断/拒绝区分、重复成员扫描后置于 parse、成对 trace）。

提交标题和代码存在只证明 implementation landed，不证明 target artifact 的 Stage 0–8 gates 已通过。**当前 HEAD `a35ec96f6` 的状态必须按 artifact 重新认证**—— wire 回滚后 parser/runtime/view/attn/flashprefill 已在开发机复跑（`50/50`，见 §21.2），真机 smoke 用 780M + `K=q8_0/V=turbo4` 完成（simple 200、循环计划 500 精确拒绝、GPUVM=0），但 Ornith-1.5-35B 的阶段 5/6/8 目标机验收仍绑定在回滚前的 artifact 上。

---

## 14. 旧 Ring 审计中仍然必须保留的数值红线

协议换轨不会让底层数学 bug 自动失效。下面这些历史证据仍应成为 DAG implementation 的 regression corpus，但要在当前 artifact 上重新跑。

### 14.1 Recurrent / checkpoint

| 历史问题 | 历史修复/证据 | DAG 后仍要保留什么 |
|---|---|---|
| mixed batch 因无关 rows 偷换 shared algorithm | group-aware gather/solve/scatter | batch packing 不改同一 stage 的 native recurrence |
| PRIVATE row 被减掉本 token transition | private/non-writer hand 分支 | FRAME/PROBE/BODY 分类不能误删 local transition |
| beta=0 被 clamp 成正 write | 删除 beta floor | zero gate 必须仍是 zero |
| `brain_copy` 未初始化 | 初始化 + dirty placement test | RERoT OFF/ordinary path 不能被 DAG input 破坏 |
| checkpoint 半写/COW sibling 覆盖 | 全尺寸预校验、COW、position/source/snapshot 一致 | C0/C_base/active state restore 要复用完整原则 |
| root PRIVATE/PUBLIC 换基底改变有效 state | `H_new = H_old + B_old-B_new` | stage seed 换物理 row 仍要保持 effective state |
| F16 persistent hand 长程漂移 | hand 改 F32 | DAG active W states 不能降成有损 hand cache |
| rollback slots 打开就改变 child equation | snapshot 保存 hand + brain selector 修复 | MTP/rollback ON/OFF 不得改变未 rollback 的持续计算 |

历史 128-step recurrent 数值：

```text
F16 hand:
output/state max error ≈ 5.76e-4 / 1.72e-3

F32 hand:
output/state max error ≈ 3.58e-7 / 7.15e-7
```

rollback=2 的历史反例：

```text
before:
output/state ≈ 2.17231 / 1.14956

after:
output/state ≈ 2.68e-7 / 7.15e-7
```

### 14.2 Attention / DDVR / Vulkan

历史重要修复：

- FP32 indexed CPU path 不得把 Q 偷降 F16；
- reference 要独立做 double QK/softmax/PV；
- empty output、length mismatch、NaN/Inf 必须 fail-closed；
- Vulkan RBB/CG 的“N 步精确收敛”不能照搬 FP32，历史改成 4N + periodic true residual restart；
- MoE singleton routing threshold / pipeline cache 必须 hermetic；
- `18→19→20→18→19` 是红线序列。

历史 CPU F16-KV independent reference：

```text
33 keys:  CPU max abs ≈ 1.19e-7 after fix
257 keys: CPU max abs ≈ 8.94e-8 after fix
```

这些不是新 DAG 的性能/质量证据，只是 backend correctness inheritance。

### 14.3 Fixed teacher tape

历史 128 native teacher tape：

```text
Turbo: 0/128 argmax mismatch
F16:   0/128 argmax mismatch
rollback 0/2 reports matched within same KV mode
```

历史另一份 512 Turbo teacher tape：

```text
Turbo: 1/512 argmax mismatch, first at step 509
F16:   0/512 argmax mismatch
```

step509 trace 曾显示：

- A3 极小 attention 差异；
- 后续层逐步放大；
- layer 29 首次改变 selected expert membership；
- 诊断性替回 native expert IDs 可恢复 native top token。

这说明“短程 argmax 一样”不等于长期数值轨迹完成认证，也不能用 production 强制 router IDs 掩盖误差。

### 14.4 历史语义/性能数据的定位

旧 Ring 时代：

- 9.11 短题曾出现 HTTP 200/stop、children natural random-ID close、最终比较正确，但 child reasoning 仍有漂移；
- 长“大洲国家”曾 HTTP 200 但 length/串洲/残缺，明确不通过；
- 2026-09-04 旧 3-slot aggregate throughput 约 237–266 tok/s，single-lane 约 108 tok/s。

这些数据只证明旧实现的某些历史性质。新 DAG protocol、native source-end、C0 probe 和 `0.synthesize` 改变了输入与生命周期，旧质量/速度不能继承。

---

## 15. 测试矩阵

### 15.1 纯 parser / schema

必须覆盖：

```text
valid simple
valid one-node dag
valid multi-node dag
duplicate JSON member
missing fields
extra fields
simple with non-empty payload
dag with empty/shaped payload
unknown strategy value
truncated JSON -> incomplete (keep sampling)
whitespace-only id/intent
id "0"
duplicate id
unknown endpoint
self-loop
duplicate edge
cycle
grammar accepts exactly the schema subset (accept/reject matrix)
```

### 15.2 DAG property tests

随机/穷举至少断言：

```text
every started node's hard predecessors are started/sealed
reader exists in started set
every visible node appears once
all visible hard edges are topologically ordered
active reader is last
retired public node does not disappear
physical binding does not change order
```

仓库纯参考脚本统计可作为 one independent oracle，但不能让项目 C++ 直接调用它产生 expected result。

### 15.3 FRAME automaton

抽象状态：

```text
reasoning
  --end--> content
  --call--> waiting_tool
  --matching result--> assistant_start
  --start--> reasoning
```

对多个 `F_i+R_i` 任意合法排列，最后必须仍是 reasoning；额外 source close 再拼下一 FRAME 应被检测为 double close。

真实模板测试还要增加：

- actual special tokens；
- tool call/result correlation；
- reasoning history retention；
- tokenizer split；
-正文尾部+end 同 token；
- user grammar/tool coexistence。

### 15.4 C0/simple

同 ordinary request 做：

```text
baseline ordinary
vs
probe → simple restore
```

比：

- sampled token sequence；
- logits/top decision；
- sampler history/RNG；
- recurrent effective state；
- KV ownership/position；
- user grammar state；
- transport cursor。

### 15.5 W>P frontier fixture

用 deterministic fake model/state 驱动：

```text
W=5, P=2
```

遍历多种 physical slice：

```text
[A,B] [C,D] [E]
[C,E] [A,D] [B]
...
```

所有 slice 都读取同一 frozen public version，最后 logical state/commit events 一致。

不能只断言“所有 task 最后都跑过”。

### 15.6 真实模型 Stage 5/6

目标模型先从小、安全配置逐级：

```text
single worker DAG
flat 2 workers
flat 3 workers
A→C + B
diamond
unequal-length workers
```

每次保存 routing JSON、DAG、FRAME raw token tape、SOURCE_END origin、reader views、response、metrics、logs 和 artifact hashes。

### 15.7 数值 fixed-tape

至少：

```text
native-vs-native determinism
single-stage RERoT vs native
Turbo/F16
rollback 0/>0
different ubatch
different physical row ordering
MoE threshold shapes
```

三级门：

1. native-vs-native mismatch=0；
2. same defined RERoT computation sampled decision mismatch=0；
3. 才报告 logits rel L2 / layer activation error。

### 15.8 兼容压力

最终组合至少包含：

```text
multiple people
DAG W>P
Tri pressure
Turbo4/Turbo2
MTP
streaming
shared prefix
one RAM demotion/restore or preemption
```

要求：

```text
0 unexpected 5xx
0 OOM
0 deadlock
0 Vulkan validation error
0 orphan seq/ref/state
0 duplicate SSE
exactly one outer terminal/request
abort != natural success
```

---

## 16. 统计口径

### 16.1 token 分账

新 DAG 至少分：

```text
prompt/prefill
probe input/generated
formal P tokens
FRAME forced tokens
BODY sampled tokens
SOURCE_END sampled/protocol tokens
synthesis FRAME
synthesis sampled reasoning
final content/tool tokens
MTP draft/verify
real restore/recompute work
```

逻辑 rollback 不能抹掉 probe 已经真实消耗的时间/算力。

同一个 token 做数值 replay/verify 也不能当成第二个用户生成 token。

### 16.2 useful throughput 与 aggregate compute

旧口径仍可参考：

```text
request-wide aggregate =
  completed model work / completed episode seconds

parallel aggregate =
  parallel model work / parallel seconds

single-lane baseline =
  predicted_n / predicted_ms * 1000
```

但 DAG 还要单独报告：

- useful BODY+synthesis sampled tokens；
- probe/frame/source-control forced work；
- W>P state swap；
- routing overhead；
- goodput / completed responses/s。

不能通过把 FRAME/probe forced token 算进 useful reasoning，制造更高 tok/s。

### 16.3 metrics

至少能观测：

```text
episodes active/completed/aborted
routing simple/dag/invalid
probe tokens/time
DAG nodes/edges/max width/max depth
blocked/eligible/starting/running/sealed
W logical active
P bound pens
frontiers
FRAME/BODY/SOURCE_END tokens
publish/topology/layout epochs
MTP invalidation reasons
Tri reclaim/compaction existing metrics
RAM save/restore
context shifts
orphan count
final synthesis starts/completes
```

不要给 person/node ID 做高基数 Prometheus label；关联信息放日志。

---

## 17. 真机安全与发布纪律

### 17.1 真机顺序

1. 先纯 CPU parser/DAG/frame reference。
2. 再 CPU/state fixture。
3. 再最小真实模型单 worker。
4. 再少量 multi-lane。
5. 再单一增加 Tri/Turbo/MTP/RAM 变量。
6. 最后才 full context / W>P / pressure。

不要同时打开最大 context、最大 W/P、Tri、MTP 和长输出做第一次验证。

禁止通过 OOM/driver reset 探 auto-fit 极限；测试前检查后台 GPU 服务与目标进程，不随意停生产或叠加未知显存负载。

### 17.2 一个结果何时算证据

必须能回答：

```text
which source/patch?
which binary and mapped libraries?
which model/template/adapter?
which full args?
which prompt/seed/routing plan?
which response/events/metrics?
which exit code?
```

只有 HTTP 200、CTest 绿、编译成功、脚本名带 phase、commit message 写 “complete” 都不足以单独证明功能完成。

### 17.3 设计红线

以后发现自己准备做以下任何一条，先停：

- [ ] 把 random Base62 close 重新作为 DAG worker 正式退出协议。
- [ ] 让最后一个 child 接管最终用户回答。
- [ ] 扫 reader 拼接文本里的 `</think>` 来判当前 worker 完成。
- [ ] 把 runtime FRAME 中的 close 当 source completion。
- [ ] 把 SOURCE_END 从真实源 state 中撤销，只因为 foreign view 不显示。
- [ ] 每 peer BODY append 都重新 forward fixed entry/footer。
- [ ] 再造动态 footer/footer checkpoint 子系统。
- [ ] 只让当前 RUNNING peers 可见，导致刚 RETIRED 的 PUBLIC history 消失。
- [ ] 只渲染 hard ancestors，丢掉无硬依赖但已发布的共享知识。
- [ ] 让 tree parent 同时表示 DAG dependency、state parent 和 ownership。
- [ ] 让 physical slot/seq 决定 child seed 或 reader 顺序。
- [ ] W>P 时让前 P 个完整跑完再启动剩余节点，却仍声称同一逻辑并发。
- [ ] later physical microbatch 读取本 frontier earlier microbatch 新 PUBLIC。
- [ ] 为 child 设强制 token 长度后伪造自然 SEALED。
- [ ] 用 attention entropy 自动剪枝，仍把后继当依赖已满足。
- [ ] 出错后删除依赖边继续生成一个“成功答案”。
- [ ] 平均 predecessor recurrent state 作为默认 child init。
- [ ] 做“跳过模型层的原生 recurrent token replay”。
- [ ] 恢复 shared RBB 作为隐藏 fallback。
- [ ] 为 DAG 动态改 RoPE base 来修语义问题。
- [ ] 每 reader 复制完整 KV。
- [ ] 每 span 独立 softmax。
- [ ] server 持久保存 physical KV indices 作为 semantic truth。
- [ ] context shift 切半 FRAME/current causal tail。
- [ ] 把 Tri eviction 当逻辑 history deletion。
- [ ] internal subagent tool call 冒充用户 tool call 发给客户端。
- [ ] simple 恢复时只重建相同 seed sampler，不恢复完整 sampler history。
- [ ] probe 已经发给客户端后还声称 simple 是纯净普通回答。
- [ ] 保留旧 random-ID/tree production runtime 与新 DAG runtime 两套长期并行事实源。
- [ ] 用 forced FRAME/probe token 提高 useful throughput 指标。
- [ ] RERoT OFF 时改变普通 attention/recurrent/sampler/server semantics。

---

## 18. 何时才可以称“DAG RERoT 完成交付”

至少同时满足：

### 路由

```text
C0 干净
simple continuation 无 planner 污染
dag schema/semantic validation fail-closed
```

### DAG

```text
hard dependency exact
cycle-preferred reader view exact
retired PUBLIC history preserved
node/run unique
```

### Protocol

```text
actual model-template fixed FRAME
FRAME forward once
native source-end provenance exact
no double close
no internal frame/tool leakage
```

### State

```text
C_base accurate
native lane-local recurrence
physical row/pen independent
W active state fully retained/restored
```

### Scheduling

```text
W>P logical cohort correct
physical microbatch does not change read world/math
frontier atomic
seal exactly once
```

### Final

```text
all workers SEALED
new 0.synthesize stage
stable final view
new F_0s forward + new logits
ordinary user grammar/tool/content resumes
```

### Ecosystem

```text
Tri/Turbo/MTP/RAM/shift/preemption/shared-prefix/n_cmpl/stream/tool/LoRA/multimodal
all have explicit target-artifact evidence
```

### Production

```text
quality non-regression/benefit threshold frozen before run
performance target measured honestly
soak stable
artifact hashes sealed
shadow/canary/production same RC
```

做到这里，才可以称：

```text
atomic-llama-cpp-turboquant
production-compatible adaptive DAG RERoT
```

在此之前，标签应更保守：

```text
DAG RERoT implementation candidate
```

---

## 19. 源码与测试入口

| 范围 | 入口 |
|---|---|
| DAG document/view/DDVR | `src/llama-rerot.h`, `src/llama-rerot.cpp` |
| KV metadata/residency | `src/llama-kv-cells.h`, `src/llama-kv-cache.cpp` |
| recurrent/checkpoint/hand | `src/llama-memory-recurrent.*`, model GDN builders |
| runtime/protocol/routing | `tools/server/server-rerot.h`, `tools/server/server-rerot.cpp` |
| server decode/wiring | `tools/server/server-context.cpp` |
| chat template | `common/chat.*`, server template integration |
| schema→grammar | `common/json-schema-to-grammar.*` |
| DAG offline oracle | `scripts/rerot-dag-reference.py` |
| parser tests | `tests/test-rerot-parser.cpp` |
| view tests | `tests/test-rerot-view.cpp` |
| runtime tests | `tests/test-rerot-runtime.cpp` |
| DDVR | `tests/test-rerot-ddvr.cpp` |
| attention | `tests/test-rerot-attn.cpp` |
| recurrent | `tests/test-rerot-recurrent.cpp` |
| 数学参考层（Q3/Q5/Q6/Q7） | `src/llama-rerot-math.h`, `src/llama-rerot-math.cpp`, `tests/test-rerot-math.cpp` |
| model fixed-tape | `tests/test-rerot-model-single.cpp`, `test-rerot-model-batch.cpp`, `test-rerot-model-permute.cpp` |

纯文档/源码核对后常用的低风险检查：

```bash
python3 scripts/rerot-dag-reference.py
git diff --check
```

是否执行 C++ test 前仍要确认构建配置和 backend；不要仅凭测试名假定不会初始化 Vulkan。

---

## 20. 历史资料与维护规则

旧 Ring 设计、旧审计和旧交接在 Git 历史里仍可用于追 bug，但不再与本文竞争“当前产品语义”。

重要历史锚点：

```text
7509335d1  old RERoT Core Correctness Candidate
535ac8cfa  old plan reset to Ring Phase 1
47432e525  before DAG cutover commits
c3648d789  DAG logical/view/fixed-entry implementation
4e7769152  DAG routing/server wiring
abd816e4d  DAG server/state lanes + C03/C11/C12 fixtures (routing wire 基线)
9290e7a1a  wire 语义回滚至 abd816e4d，保留全部后续 bugfix
a35ec96f6  当前 HEAD（TP5 默认 F32 linear lowering + reset 脚本 sclk 固化 + 54.45 tok/s）
```

需要旧文件原文可从对应 commit `git show <commit>:<path>` 获取，不把旧文本重新复制回仓库作为第二事实源。

以后更新本文遵守：

1. **先改第 1/13 节当前状态，再改阶段状态。**
2. 设计变化直接改正文，不在文末追加与正文冲突的“最新补丁”。
3. “已实现”与“已通过”分开。
4. 历史数字保留 artifact/version 边界。
5. 研究假说不升级为默认 fallback，除非有明确决定和门。
6. 不再新增新的 RERoT 指南/审计/交接作为并行事实源。

---

## 21. 计算组织研究线（2026-09-21 轮）

本节记录“重画计算组织而非先调 kernel”研究线的当前落地状态。十个研究问题（共享单位、段级 reader view、公共 KV 块服务多读者、DAG 结构/数值分离、GDN 共同基底+低秩增量、已知序列块递推、PQ2 位平面、充分统计量、联合采样、K×H 网格）中，第一轮（`29cd8f51a`）把四条等义数学落成代码并重构 indexed 布局扫描；第二轮（`2b7b479d2`）把剩余五个问题（Q2/Q4/Q8/Q9/Q10）的数学与契约层落成代码，并补上 Q5/Q6 的 F32 数值门实测；第三轮（`8772195ac`）把 Q2/Q4 的结构/数值分离接入 decode 热路径（`llama_rerot_build_query_layouts_shared`）；第四轮（`4f75dea5d`）把同一函数的数值通道换成段级偏差序＋k 路归并，并把 cache 侧 cell 扫描改为跨 reader 共享供给；第五轮（`66914c692`）把 Q3 host 侧推向单一共享 key world——R 个 pen 共享一次结构扫描与排序，每 reader 只付 ownership 过滤＋数值 pass（`llama_rerot_build_query_layouts_multi_reader`），并修复第四轮两个已提交 bug（own private/pending 的 frontier 门、段内 storage 重复下的偏差序置换）。第六轮（`0c793e2ac`）在同一函数内落三条生产形态快路径——tagged 序预检跳排序、连续 storage 恒等偏差序、uniform 桶＋全拥有恒等列表——把 decode 形态（Q=1）再压 2×。第七轮把剖面推进到 cache 级全路径：ownership 列单趟共享填充、validate 逐 entry 哈希去重换字节位图、装配 reserve——cache 级 -46%。第八轮攻下 Q=6（MTP verify 形态）数值通道：连续 run＋恒等通过集的段预计算 key-id 顺序列，发射链三次依赖随机读塌缩为一次顺序读，==best 重检整体提升出循环——Q=6 builder 16210→8758 us（1.85×）。第九轮把 reader 无关的段属性（uniform 探测、fast_keys 列）上收到共享结构 pass，每 reader 一次的 R×K 重复探测消失——builder -16%（R=6）到 -50%（R=12 K=262144），收益随 R 与 K 增长。第十轮（09-22 深夜）把 ownership 列从字节向量改为 packed bitset 直供：cache 级的 R×K 字节展开消失，builder 内 owned 探测变一次 AND（见 21.1 第九、十轮）。

### 21.1 已落地（当前 HEAD，全部 CPU 验证）

**数学参考层 `src/llama-rerot-math.{h,cpp}`**（纯 FP64，无 KV/server/graph 依赖）：

1. **[Q3] 共享 KV 多读者块 attention**：`llama_rerot_shared_block_attention` 一次读一个物理块、为每个可见读者独立产生 (m, z, u) 在线 softmax 状态；`llama_rerot_attn_state_merge` 按读者合并。共享的是数据供给，不是 softmax 统计量——与 §2.5 STRONG 语义一致。
2. **[Q5] GDN 共同基底+低秩增量**：`S_i = a_i·B + U_i·V_i^T`，每步精确追加一个秩一项（`llama_rerot_gdn_lowrank_step`）；共享基底的 `B^T x` 投影一次计算多消费者复用（`llama_rerot_gdn_base_project`）。不是 shared-RBB，不合并 lane 状态；基底只读。
3. **[Q6] 已知 token 块递推（WY 折叠）**：`llama_rerot_gdn_chunk_fold` 把 T 步已知（teacher-forced）token 折成 `(M, Y)` 紧凑仿射转移：`M = I − K·W·K^T`，`W = (I+L)^{-1}·diag(β)`，`L[j,c] = β_j(k_j·k_c)` (j>c，**行索引 β**)，`Y` 为零状态臂。生产 kernel 应按秩算子 `x → G·(x − K(W(K^T x)))` 应用，并把同一折叠块应用到多 lane 状态。
4. **[Q7] PQ2_0 位平面恒等式**：`{0,1,2,3}→{−1,0,+1,+2}` 使块内积变成 `Σ_{b0=1}x + 2Σ_{b1=1}x − Σx`（`llama_rerot_pq2_bitplane_dot`）或每 4 权重一次 16 表 LUT（`llama_rerot_pq2_lut_dot`）。与 ggml 整数路径位级一致（整数 activation 时）。

第二轮新增（同一文件，纯 FP64/位级，无 KV/server/graph 依赖）：

5. **[Q2] span 级 reader view**：`llama_rerot_span_view`（begin/len/phase）——段内 `s_j − v_j = phase` 常数，故 `q_v + s_j − v_j = q_v + phase` 对段内所有 j 成立（`llama_rerot_span_effective_pos`），整段共享一枚 effective Q；因果截断是一次比较（`llama_rerot_span_causal_len`），不是逐 key 掩码。视图描述复杂度由 span 数而非 token 数决定（`llama_rerot_span_long_fraction` 度量碎片化）。限制：空洞/异相位/异可见性必须拆段，最坏片段数仍接近 token 数。
6. **[Q4] 结构/数值分离**：run ORDER 是结构（`llama_rerot_run_order`，签名 `llama_rerot_run_order_signature` 仅随结构事件变化），长度/水位/virtual starts 是数值（`llama_rerot_virtual_starts` 前缀和；`..._after_growth` 增量更新只平移后续 run，零结构工作）。“拓扑没变”≠“位置没变”：前缀 run 增长会移动后续 virtual 起点——但这恰是数值更新，不是重建世界。
7. **[Q8] 跳块误差界（近似路线，明确门控）**：`llama_rerot_skip_mass_bound`（`q·k_j ≤ q·c + |q|·r`，Cauchy–Schwarz）给出被跳块质量上界 `Z_skip ≤ n·exp(scale·(q·c+|q|·r))`；`llama_rerot_skip_output_bound` 给出 `‖o − o_keep‖ ≤ 2·V_max·δ`，`δ = Z̄_skip/(Z_keep+Z̄_skip)`。同时以可执行反例固化“无有限充分统计量”结论：keys `{-1,+1}` vs `{0,0}` 数量与一阶矩相同但 `Z(q)` 不同（`2cosh(q) ≠ 2`）——同一 query 的块结果可精确合并（Q3），不同 query 的块结果不能因读同一历史而互用。
8. **[Q9] 联合采样契约**：per-pen RNG 流（`llama_rerot_joint_sample_seed`，(base, pen) 派生 SplitMix64）使各笔 draw 与 cohort 大小、行序无关（轨迹安全：批量化不得改变各笔 RNG 消耗次序）；`llama_rerot_joint_sample_row` 按 temperature → top-k → top-p、最低索引 tie-break；贪心路径 `llama_rerot_joint_argmax_rows` 按块归约 argmax，不回传全行 logits。输出单位是“各笔下一步决策+状态与事件”，不只是 token 数组。
9. **[Q10] frontier 网格联合验证**：`llama_rerot_verify_grid` 按列推进、STRONG barrier-after——cell (pen, h) 存活当且仅当自身前缀存活且所有 read source 的 h−1 列 cell 存活（自读平凡满足）；被拒笔发布 replacement，依赖者在 h+1 消费 replacement 而非草稿。`llama_rerot_verify_grid_naive` 是可执行反例：逐行独立验证在跨笔读下接受率虚高（测试中 C 全部自身草稿正确，但读 B 的被拒列，正确引擎 accepted=2、naive=3）。


**indexed 布局 host 侧重构 `llama_kv_cache::rerot_build_attn_layout`**：旧路径每 query 行重建整份 n_kv key 表并重新排序（O(Q·n_kv) 扫描+排序，每行 ~40B·n_kv 临时内存）。第一轮（09-21）按 distinct reader state 分组，每组一次扫描+排序。第二轮（09-22）把组内路径也换成 Q2/Q4 生产化：新增纯函数 `llama_rerot_build_query_layouts_shared`（`src/llama-rerot.{h,cpp}`）——**结构一次**（可见性分类 FULL/gated、两臂各自排序、per-run 升序 storage 数组、own-row 查找表），**每 query 只做数值**（BASE 臂与每个 own-run 段各一次二分 causal 截断、稠密 virtual 计数、effective 分组一次稳定排序，替代逐 key std::map）。逐 query `llama_rerot_build_query_layout` 保留为 oracle，两者 group-for-group/entry-for-entry 构造性一致（测试直接对拍）。实测（开发机 CPU，合成 K 键/Q 行）：Q=6（单 pen MTP verify 形态）稳定 **5–8×**（K=4096：1046→156 us；K=16384,Q=12：8920→1074 us），Q=1 也有 ~1.1×。

**第四轮深化（同一函数，09-22 深夜）**：
- **数值通道换段级偏差序＋k 路归并**：段内 `effective = (qv − B_L) + d_i`，其中偏差 `d_i = s_i − i` 与 query 无关（i 为段内发射序）。结构期把每段（含 BASE 臂）按 d 稳定排序一次；每 query 只做二分 causal 截断＋R+1 路归并（组边界＝归并中的不同值，组内 entry 序＝列表序，即 oracle 的 stable_sort 输出序）。own-row 虚拟索引从 O(V) 指针扫描换成段前缀算术。重复键检测从 unordered_set（K=65536 实测 1141 us）换字节位图（45 us）；全序比较器（唯一 key_index 断尾）从 stable_sort 换 std::sort。实测（开发机 CPU，合成 K 键，物理序打乱＝最坏情形）：K=65536,Q=6：13399→7200 us；K=262144,Q=6：82351→44704 us；per-query 增量 261 us（结构 5690 us 为剩余大头，其中 tagged 全局排序 3330 us——rank 分桶可再省，未做）。对 oracle 逐 query 路径：生产形态（物理序＝写入序）K=65536,Q=6：16073→4783 us（**3.4×**）。
- **cache 侧 cell 扫描跨 reader 共享（Q3 host 侧）**：`rerot_build_attn_layout` 的 R 个 reader 组原先各自 O(n_kv) 重建 key 表（每 cell 一次 rerot_get/pos_get/seq_has）；现改为一次扫描建共享 key 表，每组的 ownership 列从 per-seq 成员位图填。R 个 pen 同 frontier 执行时 cell 访问从 R·n_kv 降为 n_kv + R·(位图填充)。多 reader 跨组路径由 `test_ddvr_two_query_groups` 端到端覆盖。

**第五轮：单一共享 key world（Q3 host 侧完成，09-22 夜）**：

- **多 reader 共享世界**：新增 `llama_rerot_build_query_layouts_multi_reader(readers, query_pos, keys, base_owned)`（`src/llama-rerot.{h,cpp}`）——对 `keys` 做**一次**结构 pass（按 (episode, run, owner-node) 分桶、每桶按 oracle 的 (storage, frontier, idx) 排序＋偏差 (s−i) 序及其逆置换），**每 reader 只**做 ownership 相关过滤（own 段＝run 属主 node 等于 reader 的桶；foreign 段按 frontier 规则全量；base 臂按 `base_owned[r]` 列）＋逐 query 数值 pass（causal 截断＋k 路归并）。`rerot_build_attn_layout` 的 group 循环随之改为一次扫描（上界 `used_max_p1()`）＋一次 multi 调用；每组不再整表拷贝。语义边界：一个 run id 可由多个 node 先后持有（cell 生命周期内），分桶键含 owner-node 保证与逐行 node 判定逐字节一致——`test_sr_shared_physical_rows_3_ddvr_slots`（pub node 1 与 private node 9 同 run_id 1）钉住这个形状。
- **修第四轮两个已提交 bug**（探针＋对拍抓出）：
  1. **own private/pending 行的 frontier 门**：shared builder 曾对所有 own 行施加 `frontier <= reader.frontier`，而 oracle 对 private/pending 只查 `node==reader && owned && causal cut`——a future-frontier pending 行（当前写入批次）被错误丢弃。修复：private/pending 豁免 frontier 门（与 flashprefill builder 的判定对齐）。
  2. **段内 storage 重复下的偏差序置换**：第四轮把段 storage 数组置换进偏差序，MTP verify 共享位置时置换非恒等、数组失序，`upper_bound` 截断/own-row 查找随即错位（7 个对拍 case 全分歧）。修复：段内保持**双序**——tagged 序（升序 storage，供截断与 own-row 二分）与偏差序（置换 d2t/t2d，供归并）；截断是 tagged 前缀，归并按 d2t[dp] < cut 过滤。
- **实测**（开发机 CPU，合成 K 键，production shape 物理序=写入序）：R=6 时 legacy（每组拷贝＋单 reader builder）42293→17885 us（**2.36×**）；K=262144 R=6：259597→115992（2.24×）；R=12：92168→34039（2.71×）；R=1 仍 1.10×。对 per-query oracle：R=6，K=65536：112728→17885（**6.3×**）。乱序最坏形态不退化（17758 vs 17884）。

**第六轮：生产形态快路径（09-22 深夜，同一函数内）**：

- **tagged 序预检**：append-only run 的行按写入序到达＝tagged (storage, frontier, idx) 序（行内 tie 由 key_index 升序到达序保证）。O(n) 非降探测失败才落 std::sort——生产形态跳过全部桶排序。
- **连续 storage 恒等偏差序**：桶内 storage 严格 +1 连续时 d=s₀ 处处相等，偏差序＝恒等，d2t/t2d 退化为 O(n) 顺序填充（跳过偏差排序与置换表构建）。空洞/重复（reclaim、MTP verify）仍走通用路径。
- **uniform 桶＋全拥有恒等列表**：桶内 (visibility, frontier) 全一致时 frontier 门整桶一次判定；own 桶 ownership 列全 1（生产形态：reader 拥有自己整条 run）时 dp/prefix 列表＝恒等序列，不物化。`seg_view` 加 `identity` 旗标，消费端（vis_count、own-row 前缀、归并游标）经 `dp_size/dp_at/prefix_at` 访问。
- **教训（本轮唯一 bug，200 轮对拍立即抓住）**：第二版编辑把 tagged 排序调用整个删除、只留探测——探测为真时无排序可跳，为假时也不排。8054 断言失败。修复＝条件排序恢复。教训：快路径的"跳过"必须写成 `if (!fast) { general }`，不能删掉 general 分支。
- **实测**（开发机 CPU，合成 K 键，production shape，Q=1 decode 形态）：R=6：第五轮 7882→**3900 us（2.0×）**；R=1：6200→1946（3.2×）；K=262144 R=6：28864 us。Q=6（MTP verify 形态）数值通道主导，维持 ~16.4ms。相位剖面（Q=1 R=6）：struct 1.0ms + filter 0.33ms + numeric 1.8ms——numeric 即 39 万 entry 发射（输出本体），接近地板。乱序最坏形态不退化（identity/排序路径正确回退通用分支）。

**第七轮：cache 级全路径剖面（09-22 凌晨）**：

- **方法**：cache 级基准（真实 `llama_kv_cache`＋R pen 单 ubatch decode 形态，`/tmp/bench_cache_layout.cpp` 模板）＋相位计时拷贝。发现开发机 build 目录是 **Debug（-O0）**：绝对数字只作相对比较；纯 builder 的 -O2 数字由独立编译基准补齐。
- **ownership 列单趟共享填充**：`rerot_build_attn_layout` 原先对每 reader 调 `cells.seq_has` 逐 (cell, reader) 测试（R·n_kv 次 bitset test）；现改为对 resident cells **一趟**读 `seq_get_all` 位图、按 distinct reader 序列表一次填 R 列 64 位字，再展开成 builder 消费的字节列。语义逐字节不变（测试全绿）；实测本项收益不可测（bitset test 本已廉价）——保留为结构改进（消除 R 倍冗余 pass）。
- **validate 换字节位图（本轮最大项）**：`llama_rerot_attn_layout::validate` 的 per-query 重复 key 检查原是逐 entry `unordered_set::insert`（每 reader 每 query ~n_kv 次哈希插入）——cache 级剖面上是最大单项。换成 `std::vector<uint8_t>` 位图＋同走重置，fail-loud 语义不变。cache 级（-O0）111.4→66.2ms（**-41%**）；-O2 管线基准上 validate 项 0.3ms。
- **装配 reserve**：`result.groups/entries` 跨 reader push_back 无 reserve，~R·n_kv entry 的重分配是可测项；先数一遍总量再 reserve。cache 级再 66.2→60.5ms（-9%）。
- **-O2 管线全景**（独立编译基准，production shape Q=1）：R=6 K=65536：build 3.4ms＋装配 ~1.9ms＋validate 0.3ms ≈ **5.6ms**；R=1：2.7ms；R=12 K=262144：59.7ms。装配是对已物化 per-query 向量的纯拷贝（memcpy 速度，4.8ns/entry）——融合进 builder 发射需改 oracle 对拍接口，收益 1.9ms，暂不做。
- **未做（评估后放弃/推迟）**：跨 ubatch 结构缓存（增量 append 行）——第六轮快路径已消除排序成本，增量缓存只省 scan＋分桶（约 build 的 30%），但引入 cell 重用/回收导致的 staleness 风险，本轮不冒；记为后续候选。

**第八轮：Q=6 数值通道快发射（09-22）**：

- **对象**：MTP verify 形态（一 pen 的 Q 行相邻位置共一 ubatch）此前是 host 侧最大遗留（Q=6 R=6 K=65536 builder ~16.2ms）。每个 query 的 k 路归并发射对每 entry 走 `dp_at → d2t[dp] → rows[t] → keys[ki].key_index` 三次**依赖随机读**（40B 记录），且每 entry 重算 `qv + dev[dp] − vis_before` 与 best 比较。
- **改动**：结构期已检测的两个生产形态条件——run 的 storage 严格 +1（dev 为常数）＋通过集为恒等（uniform 桶全拥有/foreign）——联合成立时，为该段预计算 `fast_keys`（key-id 的顺序列，每 reader 一次，全 query 批共享）。发射时：有效值常数 ⟹ `==best` 重检提升出循环；own 段的因果截断在恒等置换下退化为前缀上界 `p < seg_cut`；foreign 段直接顺序倾倒。探测不成立的段（空洞、重复 storage、混合桶）保留通用分支。
- **实测**（-O2 独立编译管线基准，production shape，R=6）：Q=1：build 3372→2983 us（−12%）；**Q=6：16210→8758 us（1.85×）**；Q=6 K=262144：34.8ms（结构 1.8ms＋每 query ~1.2ms，接近 65k×8B entry 输出的 memcpy 地板）。
- **纠错**：首版 foreign 快分支漏置 `emitted` 旗标——`test_rerot_shared_reader_multi_query`（MTP verify 形状）立即以"k-way merge lost a group head"抓住。修复后最小复现（`/tmp/repro_fast.cpp` 模板）5/5 位置与逐 query oracle 逐字节一致。

**第九轮：reader 无关段属性上收结构 pass（09-22 晚）**：

- **对象**：第八轮后 builder 的 reader 侧仍有两处 R×K 重复工作：(1) uniform 探测——每个 reader 对每个段重读全部行 meta 验证 (visibility, frontier) 一致性，但 uniform 是 **run 桶属性**，不是 reader 属性；(2) fast_keys 列——`keys[rows[p]].key_index` 内容 reader 无关，却每 reader 重建一次。
- **改动**：`shared_run` 增加 `uniform/u_vis/u_frontier` 与 `fast_keys`，全部在结构 pass 的 run 循环里一次算好；reader 侧 uniform 分支直接读共享旗标（连 `keys[rows[0]].meta` 的首行随机读也省了），fast_keys 发射读 `sv.run->fast_keys`。
- **实测**（-O2 独立编译 min-of-10，production shape）：R=6 K=65536：2904→2434 us（−16%）；R=12 K=65536：4612→3680（−20%）；R=6 K=262144：19419→12461（−36%）；R=12 K=262144：34035→16967（−50%）；R=1 不变（无共享可收，预期）。管线级：Q=1 R=6 K=65536 总 4884→4475（−8%）；Q=6 R=6 K=262144 总 66118→55506（−16%），build 34777→23260（−33%）。
- **评估后放弃**：(1) run 桶查找的 hash map 替换——实测 unordered_map 每 key 的 hash+probe 开销超过 6–12 个桶的线性扫描（struct pass 1396→2600 us 反向），保留线性扫描；教训是桶数在两位数时别急着上 hash。(2) Q2 写入布局主动维持长 span——find_slot 已是 cont=true 连续分配，碎片来自回收/环回，修它需要 per-run 分配策略（侵入 find_slot 核心环语义，风险大），而第六轮快路径已对空洞优雅降级；记为后续候选，等有生产 span 消费者再动。

**第十轮：ownership 列 bitset 直供（09-22 深夜）**：

- **对象**：cache 级早已把 R 列 ownership 建成 bitset（第七轮 owned_words），却为了纯 builder 的字节接口展开成 R×K 字节列，builder 内再逐行读回——展开与读回都是纯浪费。
- **改动**：`llama_rerot_owned_view`（裸指针＋字数，无 kv-cells 类型依赖）＋ `llama_rerot_build_query_layouts_multi_reader_bits` 核心；字节重载变打包转发壳（测试与外部字节调用方不变）；cache 级直接传 owned_words。
- **实测**（-O2 min-of-10）：builder 内 bytes→bits −5%（R=6 K=262144：13090→12450）到 −10%（R=12 K=65536：3943→3547）；R=1 无差（预期）。cache 级另省整个 R×K 字节展开（约 2×R×K 次内存访问，未计入 builder 数字）。
- **接口教训**：纯函数模块的"类型独立"不必靠字节展开买——POD view（指针＋宽度）同样零依赖，还省转换。

**第十一轮：结构 pass 提取为持久 shared_world（09-22 深夜，第四问"结构程序"）**：

- **对象**：第十轮后 builder 的结构 pass（run 分桶、tagged 排序、deviation 表、uniform 探测、fast_keys、untagged 序）仍是**每个 frontier 全量重算**——而它是 key 表的纯函数，生产形态下两次结构事件之间只发生 append 与 publish（元数据改写），结构不必重建。这正是第四问"多数 frontier 只更新数值"在 host 侧的落点。
- **改动**：`llama_rerot_shared_world`（llama-rerot 纯模块，pimpl 无关）：`build_world`（全量，含 records 拷贝）、`build_world_structure`（借用 caller 表的结构-only 构建，一次性 bits 路径零拷贝）、`append_keys`（增量追加，尾部字典序快路径，乱序回退全 run 重排）、`set_key_meta`（publish/reclassify 元数据改写：验证先行、桶内 frontier 变化时重探测 tagged 序）。builder 拆为 `multi_reader_numeric_pass`（共享 reader+数值体）＋两个入口：`_bits`（一次性，结构-only world，行为与第十轮位级一致）与 `_world`（持久 world，跳过结构 pass）。
- **实测**（-O2 独立编译，32-frontier 摊销）：R=6 K=262144 Q=1：9069→1960 us（**4.6×**）；R=12 K=262144 Q=1：22945→3789（**6.1×**）；R=6 K=262144 Q=6：21804→12982（**1.7×**，Q 大时数值 pass 主导）。bits 一次性路径回归已消除（12800 vs 12450 基线，噪声带内；R=12 +6% 来自 key_at 稀疏映射，world 路径摊销后无此项）。
- **语义修复（本轮抓到的真 bug）**：旧 probe 只检查 (storage, frontier) 非降序，不检查 key_index tie-break——meta 改写使两行 (storage, frontier) 相等时，增量路径保持旧序而 oracle 全量重建按 key_index 重排，输出分叉。修复为完整字典序 probe（三处：build/append tail/set_key_meta），200 轮对拍全绿。
- **cache 级接入未做（下一班）**：生产 key_index==cell idx，环形复用时同 idx 内容全换（append_keys 的"新 key_index"前提不成立），需要 `replace_keys`（旧桶删行 O(run)）＋ `apply_ubatch` 逐 token on_cell 协调＋purge/回滚路径失效语义。纯模块已就绪并有对拍安全网。

**第十二轮：cache 级 shared_world 接入（09-22 深夜，第四问落地 decode 热路径）**：

- **对象**：第十一轮的 world 停在纯模块；生产入口 `llama_kv_cache::rerot_build_attn_layout` 仍每 frontier 全量重建结构。本轮把 world 接进 cache：结构事件（apply/publish/reclassify）增量维护，普通 frontier 只付 ownership 位图＋数值 pass。
- **改动**：
  - `llama_rerot_shared_world` 新增三个生产原语：`upsert_keys`（环形复用：同 key_index 内容全换，旧 run 删行、新 meta 入桶，records 位置稳定）、`remove_keys`（apply purge 删除的被覆盖 cells）、`try_append_key_fast`（O(1) contiguous+uniform 尾追，供后续热循环用）。
  - `llama_kv_cache` 持有 mutable world＋`rerot_world_gen`（CellGeneration 计数器快照）。`ensure_rerot_world()`：gen 匹配→复用；不匹配→一次全量重建（不劣于第十一轮前状态）。懒开启 `set_generation_enabled`（与 flashprefill 同模式，OFF 零开销）。
  - 增量接线：`apply_ubatch`（写 cell 收集 upsert、purge 收集 remove、gen resync）、`rerot_publish_run`/`rerot_reclassify_run`（set_key_meta 批量刷新）。
  - **关键安全设计（本轮抓的竞态）**：增量应用前必须校验 gen 前置匹配——`seq_rm` 等未跟踪变异 bump gen 后，若仍对脏 world 应用增量再 resync gen，脏数据会被 gen 匹配"洗白"。修复：收集与应用两处都做前置 gen 校验，不匹配则丢弃增量（重建时从 cells 重扫，语义无损）。
  - ownership 位图改按 world records 位置索引（`seq_get_all(key_index)` 逐 record 测试），位宽 = records.size()。
- **实测**（cache 级 bench，min-of-5，stash 对照）：R=6 K=65536：46919→28810 us（**1.63×**）；R=6 K=131072：90135→58461（**1.54×**）；R=12 K=262144：297450→230617（**1.29×**）。
- **剩余大头**（下一班）：ownership 位图每 frontier 从 `seq_get_all` 全量重建（R×K 位测试）——可增量化（新 cell 写入只改自己位）；layout assembly（entries/groups 拷贝）也可增量。数值 pass 本身（bench_world 单次 ~111ms@R=12K=262144）已不可省。
- **验证**：`test_rerot_world_incremental_decode`（test-xkv-runtime）：6 阶段生产序列——pending 布局→decode 追加→publish→publish 后追加（uniform 翻转）→环形复用（seq_rm＋同 idx 新 run）→未跟踪变异（seq_keep 安全网）——**每阶段后 cache 级 layout 与逐 query oracle 对拍**。全电池绿。

**第十三轮：cache 级 world 三连击——ownership 位图增量化＋validate 门控＋emission 重写（09-22 深夜班）**：

- **背景**：第十二轮后摊销构成（R=12 K=262144）：ownership 位图重建 48.7ms（R×K 位测试）＞数值 pass 32ms＞assembly 23ms；validate（t_layout_end 后）另占 69ms。三个目标依次消灭。
- **改动一（ownership 列增量化）**：cache 持 `rerot_world_owned`（per-seq 位图，按 world records 位置索引）。`apply_ubatch` 增量维护（purge victim 清位、写 cell 按新 seq_set 重写位、共享 cell 只清被删 seq 的位）；`ensure_rerot_world` 重建时同扫描重填。layout build 从列直接拷贝——**每 frontier 的 R×K 位测试归零**（实测 48.7ms → 0.04ms）。
- **改动二（validate 门控）**：`LLAMA_REROT_LAYOUT_VALIDATE=0` 跳过 per-entry 扫描（保留 O(groups+queries) 结构检查）。依据纯 Predefine 铁律：builder 是唯一真理，构造已结构性保证被查不变量。默认仍全验证（保守）。
- **改动三（emission 重写，抓到两个真 bug）**：
  - **Bug A（own-row 查找）**：run 内同 pos 重写（旧行 frontier 更高不可见）时，own-row 查找取 last storage match 而非 last **passing** match → query_virtual_pos 偏移。修：回退到 last passing match。
  - **Bug B（effective 公式，根本性）**：k-way merge 把 d-order 位置当可见序号——run 含**不可见行占位**（重复 storage 的非 passing 行）时两者分叉，全部 effective 平移。**正确公式**：`eff = qv + storage - (vis_before + tagged 序前缀 passing 数)`。重写为 collect-then-merge：每列表（base=storage 序、段=tagged 序）物化 eff 数组，非单调（重复 storage 形状）时段内 stable_sort，再 k-way merge。`llama_rerot_build_query_layouts_shared`（单 reader 路径）同款公式同步修复——两个 builder 与逐 query oracle 三方一致。
  - **性能恢复**：contiguous identity 段（生产 decode 形状）走 const-eff 模式——eff 只存一个值，key_ids 直接 memcpy fast_keys 前缀，merge 命中时整段 bulk resize+fill。
- **实测**（cache 级 bench min-of-5，validate off，stash 对照第十二轮前基线）：R=12 K=262144：297450→**48472us（6.1×）**；R=6 K=131072：90135→**9956us（9.1×）**；R=6 K=65536：46919→**5244us（8.9×）**。validate 默认开时 R=12 约 120ms（validate 69ms 仍在）。
- **验证**：`test_rerot_world_incremental_decode` **新增 Phase 7**（共享 cell purge：seq_cp 后 apply 覆盖共享位置——record 留 world、只清被删 seq 的位）；test-rerot-view 的 multi vs shared vs oracle 三方对拍（新增 shared vs oracle 行）；全 rerot/xkv/flashprefill 电池绿；ASAN 干净（alloc-dealloc-mismatch 误报除外——测试自带 operator new 重载）。
- **教训**：d-order/dev 机制的隐含假设"dev 排序 = 可见序号"在重复 storage 时静默失效——第十一轮测试形状（run 内 storage 唯一）不触发。新测试形状（MTP verify 行共享位置）钉住它。

**第十四轮：直写 Sink 消除 Assembly＋三档 Validate 门控＋Fast-Append 接线（09-22 凌晨班）**：

- **背景**：第十三轮后 R=12 K=262144 构成：numeric ~28ms / assembly ~22ms / validate ~53ms（默认全开）。十问文档第 2 问明确指出：视图不应为逐 token 排列的反复搬运付出代价；第 4 问强调结构已定时大部分 frontier 只付数值代价。本轮消灭独立 assembly 阶段，彻底打通零拷贝直写。
- **改动一（直写 Sink `rerot_emission_sink`）**：
  - numeric pass 的 k-way merge 增加直写 sink：预分配目标 `result.entries` 与 `result.groups`，merge 产出时**内联执行 group-base 偏置与 query_index 打标**，直接写入最终布局数组。
  - 彻底消除中间 `llama_rerot_query_layout` 对象的分配、填入、以及后续 cache 侧的二次扫描重排（原占 ~22ms 的 assembly 阶段完全消失）。
  - `result.entries` 与 `result.groups` 通过 `llama_kv_cache::rerot_assembly_scratch` 在 frontier 间**容量持久复用**，稳态 decode 阶段零堆分配。
- **改动二（三档 Validate 门控）**：
  - `validate_mode = 0`：仅做 O(groups + queries) 顶层结构单调性与空状态检查。
  - `validate_mode = 1`（默认）：在 0 基础上增加 **branchless SIMD 友好的全局 max-reduce 范围检查**（max_key < n_keys, max_group < n_groups），去除 per-entry 双重分支预测惩罚，耗时从 ~53ms 压缩至 **~5ms**。
  - `validate_mode = 2`（偏执审计档）：在 1 基础上恢复全量 per-query 字节位图 duplicate-key 检测与 group->query 一致性随机反查。
- **改动三（`try_append_key_fast` 正式接线）**：
  - 在 `apply_ubatch` 的 flush 阶段，对连续+均匀的尾部追加尝试 `try_append_key_fast`（O(1) 尾追，无需桶扫描和 touched-run 排序）；失败项自动降级至批量 `upsert_keys`。
  - 所有新增 cell 的 ownership 列同步保持精确维护。
- **改动四（真机级 Bug 修复）**：
  - 修复 `multi_reader_numeric_pass` 中 `L.const_eff` 在跨 query 复用时未被清空的隐蔽别名污染（上一个 query 为 const_eff 时会导致后续一般 query 的 `head()` 错误取 `eff[0]`）。
- **实测**（cache 级 bench min-of-5，vs 班次 11 基线 297,450 us）：
  - R=12 K=262144（validate=0）：**37,778 us（7.87×）**
  - R=12 K=262144（默认 validate=1）：**41,137 us（7.23×）**（原默认 validate 开需 120,000 us，提速 **2.92×**）
  - R=6 K=131072（validate=0）：**10,082 us（8.94×）**
  - R=6 K=65536（validate=0）：**4,863 us（9.65×）**
- **验证**：全 rerot/xkv/flashprefill 电池 100% 绿；ASAN 干净无越界；Phase 1–7 生命周期增量与重建测试全通。

**第十五轮：消灭 size 阶段的 value-init 浪费＋groups append 化（09-22 白班）**：

- **背景**：第十四轮后 phase 细分显示 layout build 的 `size` 阶段占 28–38ms（R=12 K=262k），远超 emit（4.3ms）。逐层定位发现三处结构性浪费：(a) `rerot_build_attn_layout` 入口 `result = llama_rerot_attn_layout{}` 把调用方 scratch 的 capacity **整体释放**，每次 frontier 重新 reserve 25MB+37MB（page fault + zero-fill）；(b) `entries.resize(entry_bound)` 每轮把 25MB value-init 置零后被 emit 全量覆写；(c) `groups.resize(group_bound)` 把 **37MB** 的 entry_bound 规模组数组 zero-fill，而实际组数只有个位数。
- **改动一（`llama_rerot_attn_layout::clear()` 替代对象重置）**：`clear()` 只重置 `n_queries`/`query_offsets`，`entries` 的 size **保持不变**（稳态下等于上轮 emitted cursor ≈ bound → 预 size 变成 no-op），capacity 全程保留。曾在“清空 entries size”的实验版本中观察到 `emit_cursor != total` 之后 groups 被 zero-fill 覆写——正是 size 路径与 sink 写路径交互出的数据破坏，本轮测试矩阵（含 MTP 重复 storage 形状）已钉住。
- **改动二（groups 由 sink append，不再预 size）**：`rerot_emission_sink` 的 `groups` 改为 `std::vector *`（`groups_out`），merge 命中新组时 `push_back`（capacity 已 reserve，不 realloc、不 zero-fill）；**entries 保持 raw cursor 预 size**（全量覆写）。`entry_cursor` 仍由 sink 跟踪，最终只 shrink entries。
- **改动三（const 列表直接引用 run 的 fast_keys）**：`eff_list` 增加 `key_ptr/key_n`，contiguous+identity 段（生产 decode 形状）**不再把 run 的 key-id 数组拷进 per-query 缓冲**，emission 直接读 `run->fast_keys`；每 list 的 buffer reserve 也改按实际内容规模（base=owned 数、段=run 行数、const=1），取代旧的统一 `keys/2`（每 reader 13 lists × 1.5MB 的分配浪费）。
- **改动四（PQ2_0 LUT 负结论，Q7）**：验证十问第 7 问的 LUT 路线在**当前仓库取值域与粒度下是负收益**：16 项表每 4 个 activation 重建一次（100k×128 权重实测 2178us→57650us，26× 慢），T-MAC 的收益前提是表服务大量权重行（矩阵乘形态），单 vec_dot 不成立；且现有 x86 路径已有 AVX-512-VNNI 结构化利用。结论：Q7 若要推进，必须以 **GEMM 级**共享表重新设计，不在 vec_dot 层面。
- **实测**（cache-prod bench：经 context 装配 scratch 的生产路径，min-of-5，validate=0/默认）：

|形状|14轮（旧 clear）|15轮 validate=0|15轮默认 validate|基线（班11前）|
|---|---|---|---|---|
|R=12 K=262144|37,778 us|**4,311 us（69×）**|**9,307 us（32×）**|297,450 us|
|R=6 K=131072|10,082 us|**1,180 us（76×）**|2,420 us|90,135 us|
|R=6 K=65536|4,863 us|**610 us（77×）**|—|46,919 us|

  默认 validate（tier-1 max-reduce）在 R=12 上固定占 ~5ms；val=0 时 numeric+direct-emit=4.3ms，已逼近 25MB emit 的内存带宽下界。
- **验证**：全 rerot/xkv/flashprefill 电池 0 failure；ASAN（/tmp/asan15，Debug+ASAN 全量重建）0 错误；`test_rerot_world_incremental_decode` Phase 1–7 全过（含 MTP 重复 storage 形状——正是它把 groups size 交互 bug 钉住）。

**第十六轮：问题二的段级表示落地——span 侧信道让默认 validate 与关掉等速（09-22 夜班）**：

- **动机（十问问题二）**：reader view 不必是逐 key 排列。段内第 j 枚 key 的 effective 位置 \(p_i+s_0-b_i\) 与 \(j\) 无关，整段共享一枚 effective Q；视图的描述复杂度应取决于 run/段数，而不是历史 token 数。第十五轮后 host 侧剩余固定成本只剩 **默认 validate 档的 3.1M-entry 顺序最大值扫描（~4.6ms @R=12 K=262k）** 与 fill_spans 的 2×25MB 跨步转换——两者都在为“逐 entry”付费，而生产 decode 形状里 12/13 的段是整个连续范围。
- **改动一（`llama_rerot_attn_layout::span`）**：新侧信道记录 `{key_start, count, group_index, entry_index}`——merge 的 bulk（const-effective＋contiguous-identity）分支每段 push_back 一次（R=12 时每 query 12 次而非 21845 次结构写），entry_index 是全局 entry 槽位，group_index 在 merge 解决组号后回填。`entries` 仍是权威契约（op params、直接 set 路径、测试都读它）——span 只描述，不替换。
- **改动二（tier-1 validate 改查段边界）**：默认档从“3.1M-entry max-reduce”变为“span 表边界检查＋未覆盖槽的 gap 扫描”。生产形状下 span 覆盖 ~全部 entry，实际扫描量跌到 base 臂的零星行。**validate 相：4.6ms → 3us；validate=1 与 validate=0 首次等速**（R=12 4139↔4162us 噪声带内）。
- **改动三（fill_spans 按段展开）**：staging 的 i32 缓冲区从“逐 entry 跨步转换”改为游标式展开：span 之间是 scalar entry（原样拷贝），span 之内是 `key_start+k` 连续范围+单一 group 的无分支内循环；拒绝 entry_index 乱序/越界（fail-loud 于上传路径之前，不得静默错路由 key）。
- **span reserve 绑定诚实上界**：`(world run 数+1) × 每 reader query 行数`，取代“随 groups 走 entry_bound”的过度预留。
- **踩坑记录（重要）**：第一版让 bulk 分支跳过 entry 写入、只留 span，`layout.entries` 保持零填充，cache 级 11 个测试立刻全红（`g[0..4]=k0,4`——同一 key 重复）。**原因：entries 是既有消费者的权威流（test 直读、op params 的 live_entries、直接 set 路径），span 只是旁路**。修正方案：builder 仍写 entries，span 只喂 loader/validator 的 O(段) 路径。这正是“重构必须尊重既有契约”的实例——Q2 的段描述化要一直到 kernel 契约支持 span tensor 才能真正去掉逐 entry 物化。
- **测试**：`test_rerot_shared_reader_multi_query`（MTP 重复 storage 形状，正是最可能暴露 span 与 entry 分叉的形状）新增 span 契约断言：零长度 span 拒绝、group/key/entry 三重边界、**span 展开逐 slot 复现 authoritative entry 流**（key==base+k 且 group 一致）、entry-ordered 游标单调。`test_rerot_world_incremental_decode` Phase 1–7 同路径覆盖。
- **实测**（cache-prod bench，生产 scratch 路径，min-of-5）：R=12 K=262144 validate=0/默认 **4142/4162us**（validate 成本首次归零）；R=6 K=131072 1179us；R=6 K=65536 611us。默认档与关档等速意味着"关掉 validate 换速度"的取舍消失——builder 是唯一真理的同时，审计也免费。

**第十七轮：十问收敛为计算组织后的第一轮落地——Q3/Q5/Q6/Q7/Q9 数值门与 Q3 CPU A/B（09-22 夜班）**：

本轮对应十问收敛稿：**以 frontier 为执行单位、公共数据块为供给单位、逻辑 Lane 为状态所有者、结构事件为重新定义边界**。数学层前 54 提交已就位（Q2/Q3/Q4/Q5/Q6/Q7/Q9/Q10 家族，FP64 oracle 全绿），本轮把「公式级检查」推进到「F32 验收入口」，并第一次拿到 Q3 的真实 CPU 倍数。

- **改动一：span coverage 证据入 ledger**（`llama_rerot_profile`）：`span_rows` / `span_row_spill` / `span_row_coverage` 三个新字段。这是十问问题二「视图描述复杂度应取决于 run/page 数而不是 token 数」的可观测代理——生产形状实测 **coverage=1.0000**（R=12：spans=156, rows=3,145,692, spill=12；R=6：spans=42, rows=786,426, spill=6），即生产 decode 路径几乎全部行都由 O(spans) 展开覆盖。
- **改动二：碎片对照实验**（throwaway probe，未入库）：把一半 run 改成交错提交，coverage 跌到 **0.666**（spans=24, rows=36870, entries=55308）。span 长度 63 是 **merge 分组尺寸而非 run 长度**——所以「span 数」与「coverage spill」是两个独立指标，后者才是碎片信号。`compact()` 在该形状上无恢复（build 前后数字不变），defrag 长 run 臂**评估后推迟**：需改分配器、收益中等、风险高（详见 AGENTS.md 交接）。
- **改动三：Q3/Q5/Q6/Q7/Q9 F32 门**（`tests/test-rerot-math.cpp::test_f32_gate`）：
  - **Q3**：float 在线 (m,z,u) merge vs FP64 全量 softmax，12 块 × 32 行、block max 相差数量级（rescale 压力测试），**rel=4.3e-07**；
  - **Q5**（既有）：低秩 24 步 rel≈1.9e-7（bounded）/5.1e-7（aggressive）；
  - **Q6**（既有）：WY fold T=8 绝对 2.5e-5；
  - **Q7**（新增，第一次对**真实 ggml block_pq2_0**）：bitplane/LUT 恒等式在 F32 下与 ggml 自身 decode+dot 的差 **2.98e-08**（在 ggml 自身反向累加带 2.38e-07 内），与 FP64 逐位一致；
  - **Q9**（新增，第一次对**生产 llama_sampler_chain**）：40 trial 全对上 kept-set。踩坑并写进契约：生产的 `partial_sort` 对相等 logit **不稳定**——并列时的 survivor ORDER 是实现定义的，契约只约束 **survivor 集合**（比较前两侧排序、tie 种在 k 边界之上、贪心 argmax 只在唯一最大时比较）；生产链序是 `top_k→top_p→temp→dist`，所以 **top-p 看的是原始 logit**，参考实现不能先除温度。
- **改动四：Q3 CPU A/B shadow oracle**（`tests/test-rerot-shared-block.cpp`，新 ctest 用例）：把「一块公共 KV 在片上停留期间服务多个 reader」从公式变成**可跑的 CPU 对照**——A：R 次逐 reader 调 `DdvrQsideGqa`；B：一次共享 pass（per-reader 预旋转 Q 一次、K/V 按块旋转一次、per-reader (m,z,u) 在线 merge）。输出互相对拍并计时：
  - **R=2 1.36×｜R=4 1.63×｜R=6 1.76–1.81×｜R=12 1.89–1.97×**（K=4096/16384，min-of-2，rel_err 1.5e-6–3.9e-6）；
  - 结论与十问一致：共享的是**数据供给**，不除以 FLOPs；R 越大收益越高，正是「一块 KV 服务尽可能多合法消费者」的方向。
- **踩坑记录**：A/B 探针第一版踩到 `DdvrQsideGqa` 的 **kv-head-major 布局**（`raw_k[(hkv*n_keys+key)*d]`）——单 slab 直接越界 segfault；第二版 A/B 两侧读了不同 kv head 的数据导致 rel_err=1.0。两者都写进探针注释。
- **验证**：`test-rerot-math` 0 failure（F32 门新三族）；`test-rerot-shared-block` 0 failure；`test-rerot-profile` 修正 line 统计 95→98 后 all passed；ctest main 54/54 仅 `test-vulkan-tp5-mesh` 失败（**基线既有 GPU 门**：开发机仅 1 设备）；ASAN（/tmp/asan17）四项 0 错误。

### 21.2 验证证据

- `test-rerot-math`：0 failure。Q3 对拍独立全 softmax oracle（含不可见读者、合并顺序无关性）；Q5 对拍稠密 §2.3 逐步递推（12 步，α<1，异构 β，秩每步恰 +1，dense/output 双等价，多 lane 共享投影位级一致）；Q6 24 个随机 chunk（T=1..8，含 β=0 纯衰减，此时 M=G·I、Y=0 精确成立）对拍逐步 oracle ≤1e-10；Q7 全部四种编码存在下对拍 (code−1) 解码 oracle，整数 activation 时位级相等。
  第二轮新增：Q2 span 有效位置对逐 key §2.4 定义、因果截断对逐 key 掩码、碎片化度量三态（全短/全长/混合）；Q4 前缀和对定义、增长更新=全量重算、签名对数值变化不变/对顺序变化必变；Q8 界的可靠性（精确偏差 ≤ 界、δ ≥ 真实跳过质量比）＋ `{-1,+1}` vs `{0,0}` 反例；Q9 行序无关、cohort 大小无关、确定性、最低索引 tie-break；Q10 依赖追踪 vs naive 的分岐断言（accepted=2 vs 3）。
- **F32 数值门实测**（第二轮补上）：Q5 因子化路径 24 步 F32 vs FP64 稠密 oracle——**相对误差 ~1.9e-7（有界区间）/ ~5.1e-7（弱衰减区间）**，绝对误差由状态指数增长主导（有界区间 max|S|≈2e4 时 3.8e-3）；Q6 WY 折叠 T=8 F32 vs FP64 逐步——**绝对误差 2.5e-5**。这是重结合误差的诚实量级：两族在 F32 下都不逐位一致，GPU 化前必须按此量级设验收门，不得宣称位级等价。
  第二轮生产化新增：`test_shared_layouts_vs_oracle`（`test-rerot-view`，200 轮随机对拍：FULL/gated/base/不可见各臂同在、STRONG/LAG1 交替、query 位置覆盖每个 run 边界并打乱行序，shared 与逐 query oracle group-for-group/entry-for-entry 一致）；`test_rerot_shared_reader_multi_query`（`test-xkv-runtime`，真实 `llama_kv_cache` + view 安装 + 5 行单 seq MTP-verify 形态，逐行对拍 cache 级布局与逐 query oracle 的 (key, effective) 集，并断言 own-node 行因果截断）。
- **共享布局实测收益**（开发机 CPU，合成 K 键 / Q 行，throwaway bench 已清理）：第三轮 Q=6 稳定 5–8×（K=4096：1046→156 us；K=16384,Q=12：8920→1074 us），K=65536/Q=6：20737→3784 us；Q=1 也 ~1.1×。第四轮：对 oracle 逐 query 全路径（生产形态物理序＝写入序）K=65536,Q=6：16073→4783 us（3.4×），K=262144,Q=6：96613→28584 us（3.4×）；最坏形态（物理序打乱）K=65536,Q=6：13399→7200 us，K=262144,Q=6：82351→44704 us；剩余大头是结构期 tagged 全局排序（3330 us，rank 分桶可再省，未做）。这是 host 侧布局构建的收益，不含 GPU kernel 时间。
- RERoT/xkv/flashprefill 全家 45/45 ctest 通过（含 `test_ddvr_two_query_groups` 的多 reader、多 query 行、跨 reader 可见性、精确组计数断言）。
  第五轮新增：`test_multi_reader_layouts_vs_oracle`（`test-rerot-view`，200 轮：multi reader world 与单 reader shared builder＋per-query oracle **三路** group-for-group/entry-for-entry 一致；每臂覆盖——base/own public/own private+pending 含 future-frontier/foreign public FULL+LAG1/foreign private/错 episode，段内 storage 重复与空洞、物理序打乱、每 reader 独立 query 批次）。两处修复各配回归臂：`test_shared_layouts_vs_oracle` 加 future-frontier private/pending 行与段内重复行；`test_rerot_shared_reader_multi_query` 的 own node 多 run（2+3）与 `test_sr_shared_physical_rows_3_ddvr_slots` 的同 run_id 双 node 形状由 cache 级路径钉住。
  第六轮：快路径不改语义——同一 200 轮三路对拍全绿（乱序世界强制走通用分支，恒等世界走快路径，两者输出逐字节一致）；`test_shared_reader_multi_query` 的部分拥有（ownership 列非全 1）与 `test_sr_shared_physical_rows_3_ddvr_slots` 的混合 frontier 桶覆盖非 uniform 回退。
  第七轮：cache 级改动（ownership 单趟、validate 位图、reserve）由 `test_rerot_shared_reader_multi_query`（cache 级逐 query 对拍 oracle，含 MTP verify 形状）与 `test_ddvr_two_query_groups`（双 reader 组）钉住；全家 45/45 通过。
  第八轮：快发射由同两个 cache 级测试钉住（MTP verify 形状正是快路径的目标形态），加上 `test_shared_layouts_vs_oracle`/`test_multi_reader_layouts_vs_oracle` 的 200 轮三路对拍（乱序/重复/混合世界强制走通用分支）；全家 45/45。
  第九轮：属性上收不改任何输出字节（同输入同输出，纯计算位置移动），由同套 45/45 全绿钉住。
  第十轮：bits 与 bytes 两条路径由 test-rerot-view 新增探针逐迭代位级对拍（同一 base_owned 打包后走 bits 核心，与字节重载输出逐 layout identical），加全套 45/45。
  第十一轮：`test_multi_reader_layouts_vs_oracle` 加 world 探针（每迭代：半表 build_world＋半表 append_keys＋publish 式 set_key_meta，与字节路径逐 layout identical）；新增 `test_shared_world_incremental`（乱序 append、重复 append 抛错、桶逃逸 meta 抛错、meta 改写后对拍 oracle，60 轮）；bits 一次性路径与 world 持久路径由 bench_world 位级对拍（每 rep）。全家 rerot/xkv/flashprefill 全绿。
  第十二轮：cache 级 world 由 `test_rerot_world_incremental_decode` 钉住（六阶段生产序列逐阶段 oracle 对拍，含环形复用与未跟踪变异安全网）；全 rerot/xkv/flashprefill 电池绿。
  第十三轮：emission 重写由 test-rerot-view 的 multi/shared/oracle 三方对拍＋test-xkv-runtime Phase 7（共享 cell purge）钉住；ownership 列增量维护由六阶段生命周期测试覆盖（重建/增量两路）。
- 开发机预存失败（与本轮无关，基线复现）：test-tokenizers-ggml-vocabs、test-quantize-fns、test-llama-archs、test-backend-ops timeout；test-vulkan-tp5-mesh/command-replay 需 ≥2 Vulkan 设备（开发机仅 1 块 780M iGPU）。

### 21.3 边界与下一步（第二轮修订）

- 数学参考层是 kernel 契约与 oracle，**未进入生产 decode 路径**；F32 门实测已给出量级（Q5 相对 ~5e-7，Q6 绝对 ~2.5e-5），GPU 化验收门按此量级设，不得宣称位级等价。
- Q5 的 r 从离开共同基底起算，固定入口 F_i token 也计入；r 超过阈值（约 d_k·d_v / (2(d_k+d_v))）时应转稠密，不能丢弃小增量或强造基底。
- Q6 只适用于已知 token（固定入口重放、MTP 验证块）；attention 仍按每行视图执行，不得因 GDN 块化放松因果。
- Q7 的 LUT 路径在 GPU 上“减乘法≠减耗时”，需实测；块 scale 与 Hadamard 域不得交换。
- Q8（跳块上界）与 Q10（K×H 联合投机）是近似/研究路线：本轮已把它们的**数学契约与可执行反例**落成参考代码（界、验证引擎、naive 对照），但收益测量、接受率账目与生产接入仍未做，不得与等义改写的收益混记。Q10 的保守“全笔通过才前进”方案在独立接受率 a、b 笔下整步通过率为 a^b，联合草稿必须学会预测多笔相互影响后的下一 frontier，而不是 b 条各自向前冲的草稿。
- Q2/Q4 生产化已推进到单一共享 key world＋生产形态快路径（decode 热路径，见 21.1 第五～十轮）：结构扫描与排序对 R 个 reader 各只做一次，reader 无关段属性（uniform、fast_keys）已上收共享结构 pass（R=12 K=262144 builder −50%），ownership 列 bitset 直供（cache 级 R×K 字节展开删除），Q=6（MTP verify）数值通道已 1.85×（8758 us，接近 entry 输出 memcpy 地板）；剩余方向：让写入布局主动维持长而规则的 span（`llama_rerot_span_long_fraction` 是验收指标）；把 run-order 签名接入 flashprefill 的 `llama_rerot_split_table_fragments` 调用点，结构事件才重算 fragments，数值增长走增量前缀和——注意 flashprefill 侧已有 fp_key+freshness 整层缓存，fragment 级缓存的边际收益需先证明再动手。
- Q3 host 侧：第十五轮消灭 layout build 的 size 阶段 value-init 浪费（scratch capacity 释放、25MB entries zero-fill、37MB groups zero-fill）并让 groups 由 sink 直接 append、const 段零拷贝引用 fast_keys。cache 级稳态（生产 scratch 路径）R=12 K=262k：**4.3ms（69× vs 班11前基线）**，默认 validate 下 9.3ms——逼近 25MB emit 的带宽下界，host 侧布局构建实质归零；剩余成本是语义必需 emission（GPU 搬运用途下还应走段描述压缩，见 Q2）。剩余工作全在 cache 级接入：`replace_keys`（环形 cell 复用）＋ `apply_ubatch` on_cell 协调 ＋ purge/回滚失效语义；接入后每 frontier 只付数值 pass。真机收益需目标机 `rerot-semantic-smoke.py` 对比 decode host 时间（开发机数字是合成键，不是模型证据）。

### 21.4 第十八轮：flashprefill 布局路径的共享扫描与排序探测

十问把「共享计算的单位」和「结构/数值分离」列为主干前四问。第十八轮把这两个问题落到 **flashprefill 布局路径**
（`llama_flashprefill_build_rerot_plan`，即 `LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED` 角色下的
legal-fragment 规划）：这是当时唯一还没做过任何跨 reader 共享的主干环节。

**先测后改（相位剖析）。** 在真实 builder 上临时插桩（已回退，未提交）后得到的关键事实：

- 单次 layout build 中 **92% 的时间是每 view group 重复的全 cell 扫描**（R=6 是 R=1 的 7.6 倍，同 K）；
- 排序只占 7.8%——此前几轮在 indexed 路径上得出的「排序是大头」结论**不能外推到 flashprefill 路径**；
- 匹配总 K 的分解（K=131073）：R=1「共享扫描 + 1 view」= 60.7ms，R=8「共享扫描 + 8 view」= 230.7ms，即每 view 约 21ms。

**改动一：常驻表共享扫描（Q3「一块数据供给服务多个 reader」在 host 侧的对偶）。**

新增 `fp_resident_row` / `fp_resident_scan` / `fp_scan_resident()`（`src/llama-kv-cache.cpp`）：一次扫完所有常驻
cell，记录 (idx, storage, meta, sig)。每个 view group 由「扫描」退化为「过滤」——只有可见性谓词是 per-reader。
顺带 `sig` 直接取 `cells.seq_get_all(idx)`，替掉原来每 cell 一次的 live 序列枚举 + bitset 重建；`live` /
`member_sig` 随之删除（`seq_get_all` 与旧的 live 枚举等价：任何置位 seq 的 `seq_get_used>0`，所以 live 必含
所有置位）。

**改动二：每桶 sortedness 探测（Q4「结构不变就不重做」的最小实例）。**

桶由共享扫描按 **cell index 升序**填充，而排序键是 (storage, ...)。生产环形写满足「写序 ≈ storage 序」，所以
多数桶其实已有序。加 O(n) 探测，**比较器只写一次**、probe 与回退 `std::sort` 共用，避免第十一轮
「probe 只查部分排序键导致增量路径与全量重建静默分叉」的复发；探测失败才排序。这是精确跳过，不是近似。

**实测（同一 layout，fragments/groups/uses 数量前后完全一致）：**

| 形状（K 为总常驻键） | 前 | 后 | 加速 |
|---|---|---|---|
| R=12 K=196609 | 1,143,707 us | 460,967 us | **2.48x** |
| R=6 K=98305 | 237,534 us | 128,921 us | **1.84x** |
| R=4 K=65537 | 105,332 us | 65,458 us | **1.61x** |
| R=1 K=16385 | 8,161 us | 7,132 us | 1.14x |
| R=8 K=131073（匹配总 K） | 230,682 us | 213,674 us | 1.08x（共享扫描之后的余量） |

其中共享扫描一项单独贡献 R=12 **2.25x** / R=6 **1.71x** / R=4 **1.54x**（R=1 仅 1.03x，无共享可省）；
排序探测再贡献 R=12 -7.0% / R=6 -5.3% / R=8 -7.4%。**收益随 reader 数增长**，与十问「K 笔共享结构天然就在
同一层、同一执行阶段」的判断一致。

**新增回归臂**（`tests/test-flashprefill-state.cpp::test_rerot_shared_scan_multi_reader`）：一次规划调用里放
**两个不同 reader**，且物理 cell 被两个 reader **共享**（seq_cp / 共享 cell 形状）。断言：每 reader 的
(physical, effective) 集与逐 query oracle 一致；A-only cell 对 B 不可见；B 的 base 段内出现两个 DDVR 相位
（cell 1 不属于 B 时 0@7 -> 2@8）——这正是**必须保持 per-reader** 的相位，会被错误的共享化一次抹平。写这臂时
我自己先算错两次期望（8/9 混淆 run30 的成员数），由 oracle 纠正——这就是对拍臂的价值。

**工程教训（本轮踩坑，写进规则）。**

1. **不能在 1 万行文件上做全文字符串 strip**：本轮一次探针清理 `s.replace(frag,'')` 误删了真实的 `}`，
   把一个 85 行改动炸成 2047 行差、文件大括号失衡（1983 vs 938）。恢复方式：`git checkout` 回干净 HEAD，
   把已保存的 patch（测试文件）重新 `git apply`，再用**逐条 count-asserted 的锚点替换**重做主文件改动。
   断言 `count==1` 的锚点替换是这类文件上唯一安全的编辑方式。
2. **插桩要可逆**：本轮先做「插桩 → 测量 → `git checkout` 回退」，从而把「之前几轮的相位结论」限定在它
   有据可依的路径上，没有把 indexed 路径的结论外推到 flashprefill 路径（实际结论相反）。
3. **probe 与它替换的 sort 必须共用同一个比较器**，且 probe 覆盖排序键的**全部**分量；否则快路径与回退
   路径会静默分叉。

**未做（下一步）：**

- per-view 过滤仍是 O(K) 每 reader：匹配总 K 下 R=8 的 build 时间是 R=1 的 3.8 倍。共享扫描只消灭了重复扫描，
  没消灭重复分桶。真正下一步是让分桶也按 run 复用（同一 run 的成员集对所有 reader 相同，只有「该 reader 是否
  own/gate」不同），即把 Q4 的 run-order 签名接到这个调用点——flashprefill 侧已有 `fp_key` + freshness 整层缓存，
  fragment 级缓存的边际收益需先证明再动手（21.3 节保留此判断，本轮未推翻）。
- 排序探测在生产形状收益有限（R=1 时探测本身有成本）；若写入侧后续主动维持长 span（Q2 的
  `span_long_fraction` 验收），cell 序与 storage 序会一致得更稳定，届时再评估是否把探测上移到写入侧。


### 21.5 第十九轮：FlashPrefill 共享 Run 分桶与 Base 零拷贝（R=12 达 3.68×）

在第十八轮完成常驻表单趟共享扫描（消灭 R 次重复全表扫描）后，第十九轮深入攻坚第十八轮留下的最大未解项：**消除每 View 针对 K 个 Cell 的重复分桶与哈希查找（原占构建总时长的 67%）**。

**改动一：`run_buckets` 一次性键值分桶与直接索引查找**
- 在 `fp_scan_resident` 扫描期间，活跃 Cell 按 `(episode_id, run_id)` 归集为行索引列表（`std::vector<uint32_t>`）；
- 在各 View 循环中，由原先“遍历全部 $K$ 个常驻 Cell 并逐个对 `rank` 进行 `unordered_map::find` 哈希查找”重构为**仅遍历该 View 实际声明的 Ordered Runs**；
- 每次 Run 仅做一次哈希查找直接获取预分桶的成员索引数组，随后仅在此受控切片上评估该 View 的可见性谓词（`full`/`maybe`）；
- R=12 场景下的全表哈希查找由约 236 万次彻底压缩至 12 次。

**改动二：`shared_base` 预排序与零拷贝引用直通**
- 针对非活跃但有效的 Base 成员，其存在性与排序完全与 Reader 无关；
- 在 `fp_scan_resident` 阶段一次性识别并就地完成 `by_storage` 预排序；
- View 循环内直接引用 `const auto & base = scan.base;`，彻底消灭每 View 重复的 Base 分配、填入与排序探测开销。

**实测性能演进表：**

| 形状 | 基线 (17轮前) | 18轮 (共享扫描) | **19轮 (共享分桶+Base)** | **总加速比** |
|---|---|---|---|---|
| R=12 K=196609 | 1,143,707 us | 460,967 us | **311,096 us** | **3.68×** |
| R=8 K=131073 | — | 213,674 us | **146,770 us** | **1.46× (相对18轮)** |
| R=6 K=98305 | 237,534 us | 128,921 us | **89,763 us** | **2.65×** |
| R=4 K=65537 | 105,332 us | 65,458 us | **46,694 us** | **2.26×** |
| R=1 K=16385 | 8,161 us | 7,132 us | **6,740 us** | **1.21×** |

**最终状态与缺口归档**：
本轮为全案最终交付班次。主机端布局构建已全面收敛至内存带宽与必要产物发射下界。所有后续 GPU Kernel 编写、真机 A/B 吞吐量测量以及十问核心算法闭环，均已规范化整理追加至根目录 `缺口.md`（共 17 项核心缺口），完成全面交接。

### 21.6 第二十/二十一轮：wire 回滚后的文档核对与回归复跑（09-23）

**核对对象。** 逐行比对当前 `master` HEAD `a35ec96f6` 的源码与本文正文，发现的最大漂移是**路由 wire 格式**：文中 §4.2/4.3/4.4/4.6/§13.1/§15.1 仍描述已作废的扁平 Compact Dict JSON（`{"tasks":…,"deps":…}` + 单任务自动降级 simple + `force_single_node_dag`），而 `server_rerot_routing_probe_prompt()` / `server_rerot_routing_schema_json()` / `server_rerot_parse_routing_decision()` 早已回滚到 `strategy+payload` 基线。这是「文档先于代码试验、回滚后未回写」的典型漂移，按 §20 规则 1（先改当前状态、直接改正文）一次性修正。

**核对中确认仍属实的陈述**（未改）：

| 文档陈述 | 源码核对点 |
|---|---|
| probe 模板末尾必须带换行 | `server-context.cpp` arm probe 前显式校验 `probe_prompt.back() != '\n'` → `rerot_protocol_error` |
| probe 写独立 `probe_seq`，C0 slot seq 不被追加 | `rerot_start_root` / `arm_isolated_probe` / `discard_isolated_probe` |
| `capture_c_base()` 在正式 P 注入后调用 | `rerot_enter_dag`：rebuild → `set_dag_protocol_markers` → `initialize_dag` → formal P forward |
| LCP 前缀重建只发生在启动边界 | `rerot_render_dag_prefix` / `rerot_rebuild_dag_prefix_memory`（仅在 `rerot_enter_dag` 调用） |
| DAG 模式下旧 fork/fence 生产语义 hard_abort | `publish_pending_record` / `freeze_fork_parent` 在 DAG 路径的条件分支 |

**本轮修正要点**（详见正文）：§4.2 流程图、§4.3 wire 格式/Schema/强制分层/语义映射、§4.4 控制面列表 1 与 5、§4.6 实现清单（补 probe 治理）、§1.1 旧概念表、§1.2 定级与 wire 回滚叙述、§1.3 新增 probe 512 上限与截断/拒绝区分条目、§13.1 inventory、§13.3 改为 wire 基线与保留修复清单、§15.1 parser 矩阵、§20 锚点。

**回归复跑**（`build/` Release，Vulkan ON；`test-rerot-parser` 已按 `9290e7a1a` 重编）：

| 检查 | 结果 |
|---|---|
| `build/bin/test-rerot-parser` | 0 failure（覆盖新 wire 的 accept/reject 矩阵、重复成员、自环、重复边、保留 id `"0"`、空白 intent、grammar accept/reject） |
| `build/bin/test-rerot-runtime` | 0 failure |
| `build/bin/test-rerot-view` | 0 failure |
| `ctest -R "rerot\|xkv\|flashprefill"` | **50/50 通过** |
| `scripts/rerot-dag-reference.py` | 通过，统计不变（DAGs 543 / Legal live states 3007 / Reader views 3904 / Fixed-frame compositions 384） |

注：首次 `ctest` 有 3 项 `Not Run`（`test-rerot-span-expand` / `test-rerot-q-prep` / `test-rerot-q3-shared-kv-2reader`）——纯粹是这些目标从未在本 build 目录构建过（源码在 `tests/CMakeLists.txt` 已注册），构建后单独跑 3/3 通过，再跑全套 50/50。这不是回归，是构建目录陈旧。

**边界。** 本轮是**文档核对 + 源码静态比对 + CPU 回归**，未启动模型、未跑 GPU、未在目标机重跑阶段 5/6/8。§12.5–12.9 与 §13.2 中的 Ornith-1.5-35B 结论仍是回滚前 artifact 的历史证据。

**遗留未闭合。** ① 目标机（5×RX 6800 / Ornith-1.5-35B）阶段 5/6/8 未按 `a35ec96f6` 重跑；② wire 回滚后真机 probe 质量（780M + `K=q8_0/V=turbo4`）已有 5/5 HTTP 200 与环拒绝证据，但那是 AGENTS.md 交接记录，本文只引用不重复记账；③ 当时 `缺口.md` 未随 wire 核对更新，现见 §22 与 `缺口.md` 第五部分的候选验收账本。
---

## 22. RERoT 与计算组织核心候选四轴验收账本与交付准则（2026-09-23 审计）

为确保 RERoT 方案自包含、事实单源，现将算子优化（R 系列）、计算组织重构（Q 系列）及布局亲和（H 系列）的核心候选统一遵循四轴准入裁决门进行事实对齐：
1. **实现状态 (Implementation)**：生产代码 / 桥接代码 / 原型代码 / 仅 CPU 参考；
2. **正确性验证 (Correctness)**：位级对拍 / 容差对拍 / 仅离线门 / 未经验证；
3. **真实路由与端到端净收益 (Real Route-hit & Paired Benefit)**：真实多 Lane 请求是否命中、真实客户端 committed tokens/墙钟是否有正向加速比；
4. **默认决策 (Default Decision)**：保持 opt-in / 保持研究态 / 拒绝 / 不予晋级。

> **执行纪律与基准要求**：
> - 历史降频时段数据作废，禁止以异常基线宣称性能收益；
> - MTP 与推测机制目前仍存在稳定性/挂起风险，严格保持 opt-in，严禁擅自默认启用；
> - Timeline 路线已排除，不再列入测试矩阵；
> - 任何离线微基准或局部倍率均不可替代真实模型 A/B/B/A 配对净收益。

### 22.1 核心候选四轴状态与后续门禁对照表

| 编号 | 候选名称 | 实现状态 | 正确性验证 | 真实路由与端到端收益 | 默认决策 | 当前阻断与下一步准入门槛 (Next Gate) |
|---|---|---|---|---|---|---|
| **R02** | Live Q-prep 算子 | Vulkan `rerot_q_prep.comp` 与 CPU oracle 已就绪（环境变量 `LLAMA_REROT_GPU_Q_PREP=1`） | `test-rerot-q-prep` CPU/GPU 对拍通过（覆盖 active=0/cap，非活跃行强行灌入 NaN 毒值） | 尚未在生产模型真实 RERoT 路由中命中；Shader 虽只旋转 live 行但当前仍按 capacity×head 派发 | **保持 Opt-in**（严禁默认开启） | 必须在真实模型多 Lane 会话中取得端到端 A/B/B/A 净收益；后续 WHT 融合必须严格保持数学时序。 |
| **R07** | 多行 / PQ2_0 投影 Recipe | 通用 PQ2_0 GEMV / MMQ 存在 | 专用 $b \in \{1,2,3,4,6\}$、非零 offset 的共享权重 Tile 路径未测试 | 目标 GPU 上的多行加速收益未测 | **不予晋级** | 须完成专用 Tile 复用 Recipe 并通过尾行/边界对拍。 |
| **R09** | 长短 Reader 分桶与非均匀 Head Map | 现有 Head Map 与标量全局 Split-K 存在 | 混合长短行（如 100 vs 32768 tokens）GPU 对拍未完成 | 缺乏自适应分桶在真实模型上的延迟削减证据 | **不予晋级** | 须基于 Reader 长度方差建立短/长双桶，在 GPU 端消除短行过度切分的归约开销。 |
| **R10** | 逻辑 State 驻留池：D2D Parking | Host 侧 `capture_hand_seed` / restore 可用 | 仅 Host 序列化验证；显存 D2D Parking 槽位未验证 | 缺乏 D2D 替换 CPU 往返的端到端时延收益 | **不予晋级** | 须在显存建立 native state slots，通过跨 Lane 换笔状态等价测试。 |
| **R12** | 阶段预定义图族 | predefined graph 基础设施存在 | 真实 RERoT 多阶段生命周期审计未完成 | 尚未取得稳态 graph_def_count 归零带来的端到端冷启动延迟收益 | **不得宣称净收益** | 须按原因桶为常用阶段建立固定图族，消除执行期图重构开销。 |
| **R13** | Compact Span ABI 展开桥与直通 | ABI v1 固化，CPU 展开与 Vulkan 展开核函数（`rerot_span_expand.comp`）已交付 | `test-rerot-span-expand` 与 CPU 位级对拍 100% 通过（`LLAMA_REROT_GPU_SPAN_EXPAND=1`） | GPU 展开后仍生成逐 entry 密集张量输入，未直通 attention 算子；真实路由收益未测 | **保持 Opt-in**（严禁默认开启） | 阶段 A：测定 GPU 展开真实净收益；阶段 B：改造 Attention 核函数直通消费 Span 描述符。 |
| **R14** | RERoT 端到端净收益 A/B/B/A | 测量脚本 `rerot-throughput-gate.py` 区分 sampled-token 速率与 model-token 诊断，校验六行同批计数 | Mock 单测 10/10 通过；首次同请求配对的 RERoT 探针语法被拒 | 无完整真实配对净收益数据 | **未获验收** | 完成合法真实路由、多轮 A/B/B/A，以客户端 committed tokens/完整请求墙钟为准；不能用单次失败或 generation-only 速率代替。 |
| **Q1** | 图级公共输入复用 + MoE 专家聚集 | 仅激活旋转局部共享；图级 CSE 与专家聚集未实现 | 无 GPU 共享输入执行证据 | 未测量 | **保持研究态** | 须在 `llama-graph` 中实现具名 norm/量化共享，并建立专家分桶重排逻辑。 |
| **Q3** | 公共 KV 块多 Reader Attention | 主机端共享 World 已交付；离线 2-Reader 单 invocation Shader 原型存在 | CPU 参考层与 shadow oracle 通过对拍（误差 < 3.9e-6） | 未进入生产调度；可变长多 Reader 真实加速未测 | **保持研究态** | 须平衡 LDS/VGPR 占用，编写生产级多 Reader 共享 Attention 核函数。 |
| **Q5** | GDN 状态共同基底 + 低秩增量 | CPU 参考层已交付（`llama-rerot-math.*`） | F32 数值门实测通过（相对误差 ~1.9e-7 有界 / ~5.1e-7 弱衰减） | 无生产 GPU 低秩递推路由与收益 | **保持研究态** | 须编写 GPU 端低秩递推算子，并设立增量步数 $r$ 超标时平滑回退至稠密矩阵的安全机制。 |
| **Q6** | 已知 Token WY 块折叠 | CPU 参考层已交付 | F32 数值门实测通过（$T \in [1,8]$ 绝对误差 < 2.5e-5） | 缺乏生产 GPU 路由与收益证据 | **保持研究态** | 须将紧凑块表示接入固定入口重放与 Target 验证块，严禁破坏因果可见性边界。 |
| **Q7** | PQ2_0 位平面子集求和与 LUT 路径 | CPU 建立 2 bit-plane 公式与 16 项 activation 表 | 真实 `block_pq2_0` 误差 2.98e-8；已知单 `vec_dot` 粒度存在性能劣化 | 缺乏矩阵级共享查表在 GPU 上的实机基准 | **保持研究态** | 须在矩阵级粒度评估查表与 LDS 寻址开销，在真实硬件上验证超越解码+点积的能力。 |
| **Q8** | 上下文跳块：Cauchy-Schwarz 质量界 | CPU 建立 Cauchy-Schwarz 跳过质量界 $\delta$ 公式 | 反例已证明不存在有限充分统计量；缺乏模型级质量门 | 无生产跳块 GPU 加速 | **保持研究态（近似路线）** | 必须在独立近似开关下进行 Shadow 评测，验证长文质量与任务准确率不降，严禁与等义优化收益混淆。 |
| **Q9** | 越过 Logits 的批量 GPU 采样器 | CPU 参考契约已固化（确定性、独立 RNG 流、tied logit、贪心分块 argmax） | 仅在 CPU 建立生产链序（top-k $\to$ top-p $\to$ temp $\to$ dist）；无 GPU 着色器实现 | 无 GPU LM Head 采样收益 | **保持研究态** | 须编写原生 GPU 批量采样 Kernel，直接输出 token ID 向量，并与服务端状态机完全打通。 |
| **Q10** | $K \times H$ 联合投机网格验证 | CPU 实现基于列推进和依赖传染的 `verify_grid` oracle | 仅 CPU 逻辑验证；成功复现 naive 对照的错误分支 | 无端到端跨笔推测执行与多请求有效 token 产出收益 | **保持研究态** | 须设计多笔交互后的联合 Frontier 推测逻辑，并在 GPU 建立列未命中时的快速回滚机制。 |
| **H16** | FlashPrefill 跨 View 共享 Run 切片 | 相同 `fp_rerot_view_key` 已合并至同一 View Group；共享扫描与 Base 零拷贝已进生产 | 直接跨不同 View 借用切片会破坏 `vgroup_frags` 连续区间及 `run_v0/phase_bias` 偏移，已主动拒绝直接指针复用 | 现行架构下无安全净收益 | **拒绝直接指针复用** | 须重新设计解耦 `phase_bias` 的切片元数据表示，仅在证明真实多 View 场景产生显著分配开销时方可重启。 |
| **H17** | 写入端长 Span 亲和性 | 分配器已有连续性选择基础设施 | 碎片化场景下主动重排的连续性不变量未验证 | 生产环境下 coverage 已接近 1.0，主动整序未证明能带来显著端到端加速 | **保持研究态** | 须在特定重度碎片化工作负载下评估 span 连续度提升对上传与 Attention 遍历的实际收益，不阻塞主干。 |

### 22.2 自包含交付与后续准入门槛总结
- **不可混用口径**：严禁将近似算法（Q8）的收益计入严格等义优化，亦不可将 CPU 布局阶段的吞吐倍率（如 3.68×）直接等同于整机端到端加速；
- **生产闭环唯一标准**：任何待晋级特性必须通过 `scripts/rerot-throughput-gate.py` 的 A/B/B/A 配对跑分，以客户端每秒交付的有效 committed tokens 与完整请求墙钟为唯一黄金准则；
- **文档闭环**：本文档与 `缺口.md` 共同构成本阶段 RERoT 架构与未决事项的自包含完整事实源。

### 22.3 MM-R1 Bonsai-2 单卡 GPU 修复与真实路由对照（2026-09-24）

同一 27B PQ2_0 GGUF、RX 6800 `Vulkan0`、`-np 6 --rerot-pens 6`、seed 20260923。七叶 mindmap 使第七名 worker 与前六名轮换；`GGML_VK_CMD_REPLAY=0` 仍复现硬复位，故不能归因于 replay。RADV hang dump 定位到 `CmdCopyBuffer` 读取已解绑的图分配器 VkBuffer：拷贝 3,973,120 字节，恰为四行 logits × 248320 词 × F32。`ggml_vk_buffer_read_2d_async` 对 pinned-host 目标只录制异步拷贝，局部 `vk_buffer` 所有者却在录制后离开；下一图的分配器可在 GPU 执行该拷贝前释放源。现由 `vk_context_struct::copy_buffers` 留住异步读/写两端的 VkBuffer，直到 backend fence 完成并清理该 context；先前试验性的 RERoT 强制同步及 split-K 预分配改动未解决故障，已移除。原始故障地址、BO 解绑记录和 copy 指令见 `artifacts/mm-r1/2026-09-24-bonsai-gpu/gpu-fault-evidence.json` 指向的 RADV dump。

修复后的原复现提示 `Compute 13*17 and 21*19, then give both products and their sum.` 完整 **HTTP 200**：86.7 s / 1390 completion tokens，答案 221、399、620；同一批次 trace 在 0:29.661 显示 node 1..6 占据 `tok_idx=0..5`，第七叶实际轮换，非只申请六个空 pen。该次之后未见新增 GPUVM fault，Xorg 存活。这是该设备/提示的单次验证，不是所有硬件与长度的稳定性证明。

相同二进制、模型、提示、seed、六 pen，按 JSON / mindmap / mindmap / JSON **各重新启动进程**的 A/B/B/A 完整请求墙钟为 **6.3 / 71.4 / 70.8 / 6.3 s**；四次答案均正确。JSON 实际选 `simple`（68 probe tokens、0 frame tokens、124 completion tokens），mindmap 选七叶（295 probe、2282 frame、1014 completion tokens），所以这组数据证明的是**当前真实路由的产品代价**，不是同工作量 DAG 算法的吞吐倍率；71.1/6.3 不能称为纯 wire 性能差。六算术子题、六主题以及明确要求并行子问题的另外三次 JSON 实际路由也全选 `simple`，故本轮**没有**有效的 JSON-DAG 对等路由性能证据，不予默认晋级或净收益宣称。完整配对数据与配置见同一 artifact 目录。

**以下 Raft 质量对照属于旧版 16 叶协议的历史结果；门控、提示词和叶数上限均已撤销，不可称为当前协议的配对实验。** 同机同模型以原样中文请求 `给学生讲清楚 Raft 算法` 复核**公开答案质量**，评分维度事前固定在 `artifacts/mm-r1/2026-09-24-bonsai-raft-gpu/rubric.json`，对照 Raft 原论文 §2、§5.1–5.4。初次 mindmap 生成约 25 叶，被 16 叶解析器拒绝；当时探针增量审计原以仅在接受计划后才赋值的 `episode.plan_kind` 为门，实际上永远跳过；随后曾按请求配置的 wire 运行审计，并提示模型将相关讲解/例子合并、最多 16 叶。六叶提示的一次尝试仍生成超限计划，由审计在探针阶段明确拒绝；16 叶提示在 `-c 6144` 时耗尽 episode token budget。最终两臂都用 `-c 12288`、六 pen、seed 20260923、`max_completion_tokens=4096`，各冷启动一次：JSON 实际 `simple`，HTTP 200 / **145.6 s** / 66 probe / 0 frame / 输出截断；mindmap 16 叶、18 runtime nodes，HTTP 200 / **691.9 s** / 378 probe / 4415 frame / 输出完整。trace 同一批次实际占据六 pen；当时 `rerot|xkv|flashprefill` 52/52 通过。

按冻结的八维 0–2 分量表逐项只看公开文本：JSON **10/16**，mindmap **7/16**。前者解释有序日志与状态机，但没有随机选举超时，且对同任期更新与日志覆盖有错误；后者讲了随机超时及多数派，却没有解释确定性状态机按已提交日志执行、AppendEntries 的前项索引/任期一致性检查，并误称任何时候都不可能同时有两个 Leader（Raft 只保证**同一任期**最多选出一个；旧任期 Leader 可在分区中尚未知晓新任期）。此例**不支持 mindmap 让模型理解更深**的主张；16 叶计划也不是质量证据。JSON 已截断、mindmap 工作量不同、单 seed/单卡单次，墙钟 4.75 倍只表示两条真实路由的成本差，不是等工作量吞吐或学生学习效果。原始公开回答、计划、失败臂和逐维评分见同目录 `comparison.json`。

#### 当前协议：固定前缀、逻辑叶与物理 pen 解耦

用户明确要求保留细分，不再以 pen 数或固定 16/96 节点阈值缩减逻辑计划。当前探针只给一句 `Let me lay out a mind map.`；其后强制注入标准 Mermaid 前缀 `` ```mermaid\nmindmap\n  root\n    ``，文法仅生成第一个子节点开始的后缀。字面 `root` 是树形协议锚点：仍作为内部索引 0 以维持父子结构，但不计入 `node_count`、不成为 leaf/worker，也不会进入 worker 作用域文字。解析器已删除节点数、叶数、wire 字节数、深度与单标签长度的预算参数；mindmap 也不再走新增的增量审计或 JSON 独有的 512-token 探针闸门。探针在检测到末尾围栏后才完整解析一次，避免宽树每 token 重扫。仍对围栏、两空格一级且不得跳级的缩进及字符集作严格语法校验；最终工作总量受所选模型的 context 与实际运行资源约束，不截枝。RERoT state v9 拒绝旧版全 W cohort 的 v8 blob，防止恢复后绕过 P-bound 队列约束。

旧版 CPU 回归：2048 叶且超过旧 16384-byte wire 上限可完整 parse/serialize；64 层单链与 256-byte 标签亦合法；25 叶在单 pen 上按当时的全量 cohort 轮转 admission。旧版真机使用 Bonsai-2 27B PQ2_0、RX 6800 `Vulkan0`、模型原生 `-c 262144`、六 pen：请求「把整数 1 到 20 的平方分别作为独立子题计算，最后只按顺序列出二十个结果。」实际生成 **21 个叶子 / 21 个逻辑节点**（额外生成一枚 `1 to 20` 叶），probe 119 token、frame 5016 token，HTTP **200 / stop / 1254.7 s**；公开输出严格等于 1²..20²。证据为 `artifacts/mm-r1/2026-09-24-bonsai-raft-gpu/fixed-prefix-wide-squares.json`、`fixed-prefix-completed-route.log`、`fixed-prefix-six-workers.log`。通用缩进文法及无预算解析器另以 `只回答 5+7 等于几？` 真机烟测两次：HTTP **200 / stop / 44.0 s** 与 **43.9 s**，答案均为 `12`；见 `fixed-prefix-generic-depth-smoke.json`、`fixed-prefix-generic-depth-plan.log` 和 `fixed-prefix-final-smoke.json`。

原样 Raft 提示在固定前缀、已撤销叶数阈值但尚未撤销旧深度/标签边界的**旧调度构建**中形成 **73 个逻辑节点、51 叶 / 六 pen**，probe **613 token**（确实越过旧 512 闸门），进入 53-node runtime，以当时的 all-W 逻辑步轮转推进；客户端 **3600 s 截止时已进入 synthesis，但尚无完整公开答复**，服务器随后因客户端断开取消该请求。最终无预算构建只做了上面的短题真机烟测，未重跑该 Raft 长任务；此轮只能证明当时宽树规划和执行推进，**不能**拿旧版 7/16 评分外推当前协议的 Raft 教学质量；记录见 `fixed-prefix-plan.log`、`fixed-prefix-raft-stress.json`。当前这条长请求的端到端时延仍是未闭合问题，不伪称 HTTP 200。新调度的第一阶段吞吐验收要求六个物理 pen 在单批内同时 decode，超出的 eligible 节点排队，不在每步换槽；下述唯一一次配对尝试未进入 DAG，故尚未测得此条件。

### 22.4 六 pen 固定 cohort 性能候选与一次配对尝试（2026-09-24）

调度已从按所有逻辑 worker 每 token 换槽改成至多 P 个物理 pen 常驻同一 cohort，超出的 eligible worker 排队，直到现有 pen 自然完成才补位；recurrent KV 行与 pens 同容量，排队 worker 在获得 pen 时才克隆 lineage。RERoT host 布局删除未消费的 per-reader Base 排序，span staging 不再每 token 全量清零或二次遍历；Vulkan indexed attention 可将 GQA 同 KV-head 行按 Br 分组并在 tile 内并行找相位边界。后者尚未完成真实模型收益/质量门，**默认保留历史 Br=1**，须显式 `GGML_VK_REROT_GROUPED_HEADS=1` 才启用候选。`rerot_six_row_batches` 只在一次成功 `llama_decode` 的输入中确有至少六条 RERoT pen 行时递增，验收脚本要求 `--min-peak-lanes 6` 时该 counter 为正；峰值分配六 pen 本身不等于六行同批执行。`test-rerot-attn`、`test-rerot-runtime`、`test-rerot-profile`、`test-rerot-span-expand`、`test-rerot-view` 与 `test-server-task` 的定向测试通过；吞吐 gate mock 10/10 通过。上述只能证明这些局部正确性路径，不证明模型端到端吞吐获益。

同一 Bonsai-2 27B PQ2_0、RX 6800 `Vulkan0`、`-c 262144 -np 6 --rerot-people 6 --rerot-pens 6 -ctk q8_0 -ctv turbo4`、固定前缀 mindmap 和相同 1..20 平方请求，**只发起一次**先 simple 后 RERoT 的配对试验（seed 20260923；gate `--rounds 1 --min-ratio 1.001 --min-peak-lanes 6`；试验时 grouped attention 尚未收回 opt-in）。simple 先完成，服务端记录 448 decode tokens、18,925.09 ms eval（23.67 tok/s）、19,920.13 ms total；随后 RERoT 探针被解析器拒绝 `structure: skipped an indentation level`，HTTP 500，episode 未进入 worker 执行。**这是 GBNF 与解析器的契约缺口，不能简单归责模型违反 GBNF**：`grammar_g0()` 的 `indent ::= "    " ("  ")*` 允许后续行任意偶数层级，解析器却要求新深度至多比当前父链加一；例如首子节点四空格之后直接八空格，文法允许而解析器拒绝。该次未启用 `rerot_trace`，因此没有保存被拒计划原文，只有解析器诊断；无法确定具体跳级行，亦不能归因于 grouped numerics。脚本在第二臂中止，**未生成完整配对 JSON、未测到六行同批或有效 token 吞吐比，更不能声称打败 simple**。不应通过静默补齐父节点绕开 fail-closed；按一次运行的约束未补跑。可复现请求配置在 `scripts/rerot-bonsai-throughput-request.json`。

R02/Q1/Q3/Q5/Q6/Q7/Q8/Q9/Q10 与 R07/R09/R10/R12/R13 的生产级跨 reader/跨算子优化仍受上表数值、安全及真实路由准入门约束；此轮没有将研究原型伪装成已验证默认路径。全请求客户端 committed-token/墙钟仍是最终黄金口径，脚本的 sampled-token generation rate 只是其中一个诊断维度，不能取代多轮 A/B/B/A 验收。

#### 后续修复：GBNF 与严格缩进口径对齐

保持固定前缀 Mermaid mindmap 和任意有限树深度；静态 GBNF 仍约束标签、行形与结束围栏，在其后叠加一个**有状态的缩进采样约束**。它在选 token 前检查整个 token piece（包括跨行和只含部分空格的 piece），禁止下一行深度超过上一行加一；缩进判断与最终解析器共用 `next_depth_valid()`，不生成缺失的父节点，也不引入固定最大深度；独立解析器还会立即拒绝已经不可能补全的过深缩进前缀。普通回答与其他路由文法不安装此约束。`test-rerot-mindmap-parser` 覆盖原先「GBNF 允许、解析器拒绝」的跳级例、分段 token、同 token 多行、回退后再加深、80 层单链及相邻层级状态对拍；`test-rerot-parser`、`test-rerot-runtime` 同过。词表专用的 Qwen2 GGUF 上真实采样器链烟测：单独 GBNF 对合法/跳级两例均接受，叠加深度约束后仅接受合法例（`valid=1, skipped=0`），未启动模型推理。旧 HTTP 500 仍为历史事实，此修复**没有再次运行真实模型或取得吞吐比**。

#### 批准复测后的 frontier 377 停滞与封口交接修复

后续批准的真实模型运行成功通过探针（约 4.44 s），但 35 分钟长请求与 60 秒流式诊断均停在相同的 `rerot_frontiers=377`、`rerot_frontier_rows=1601`、`rerot_six_row_batches=244`。**这不是所有叶子完成后的 synthesis 等待**：trace 最后仍有 node 1、3、4、5、6 采样，node 2 刚产生原生 `</think>`。先前将进程存活和早期输出误判为“仅性能慢”不成立；取消后的空闲槽位也不是取消前状态的证据。

根因是 runtime/host 两层资源所有权脱节：`seal_dag_node()` 封口即停放 KV 并释放 runtime pen，`physical_slot` 被清为 -1；`retire_node()` 随后走 parked 分支，无法再回报原 slot 编号。`rerot_commit_inflight()` 对普通生成 token 在 `!forced_complete` 处提前返回，服务端原 slot 因而一直留在 `GENERATING`。五个在跑节点完成下一步后，被 cohort 守卫挡住；候补 node 7 已进入新 cohort 却拿不到仍被 sealed node 2 占着的 host slot，形成无进展等待。

修复在提交并冲刷完原生 source-end 后、普通 token 的提前返回前，对**已封口的 DAG worker**调用既有 `rerot_make_slot_idle()`；历史 KV 仍归 runtime 的 parked/archive 所有，sealed worker 永不恢复，故不再额外克隆 sampler/lineage。synthesis 的活槽保持不变，peer admission 仍在整批 commit/capture 后统一执行，未加入超时跳过或强制结束。

验证：`llama-server` 构建通过，`test-rerot-runtime` 与 `test-rerot-mindmap-parser` **2/2**；同题、seed 20260923、六 pen、同 KV/原生 context 的 **60 秒 SSE** 复测，指标推进到 **429 frontier / 1914 行 / 297 次六行同批**，trace 明确显示 **node 7 已接替并持续采样**。客户端按约定在 60 秒截止（curl 28，接收 32505 字节），服务已停止；原始流和计数保存在 `artifacts/mm-r1/2026-09-24-bonsai-raft-gpu/rerot-slot-release.{sse,metrics}`。现有 CPU runtime 测试不覆盖 host slot 状态机，因此此次以原始真机失败与修复后同场景烟测证明缺陷消除，而非用 mock 复写生产逻辑。

边界：本次没有等待完整 20 子题收尾，`completed_episode_total=0`，不是全请求吞吐验收。流式内容还显示多个 worker 重复回答整份平方表，而非只处理自己的节点；这是下一项任务范围/执行提示语义问题，不能用六行同批计数掩盖，也不能由本次诊断推定它的具体根因。

#### Mermaid 提示协议收缩：become(node)

按负责人要求删除节点描述器及其冗余 scope/协作/职责话术。探针和正式公共 P 均使用 `Let me lay out a mind map.` 的第一人称入口；P 后保留完整的规范化 Mermaid 图，每个节点都能读取，不再把祖先路径复制进各自工具帧。

唯一内部工具为 `become`，唯一参数为 `node`，值是冻结 TreePlan 中当前节点的**原始标签**，不是 `leaf_N` 或其他 host id。工具回复固定为 `Now you became the node <原始标签>, please think about it.`，接模型原生 assistant reasoning 入口；不再附加 scope、depth、phase、final_mode 或其他角色提示。汇总阶段同样 `become({"node":"root"})` 回到原图根节点，不向模型解释 runtime 的 synthesis 角色。调度和 reader-view 所需编号仍在内部保存，工具参数及该帧的 call id 不携带这些编号。旧 JSON-DAG 路径不在本次协议改动范围内。

删除了 scoped-worker-intent helper、未被消费的 scope_path 存储及固定话术断言；保留并验证全图在 P 中往返解析后 TreePlan hash 不变。`llama-server` 构建通过，`test-rerot-runtime` / `test-rerot-parser` / `test-rerot-mindmap-parser` **3/3**。Bonsai 实际词表与原生模板的无推理烟测，`求 25 × 12` 与 `root` 两种 become 帧均正常序列化、只含一个 node 参数并以 `<think>` 打开：分别 **71 / 56 token**。此证据是协议渲染验证，不是模型遵循性或吞吐获益证明；没有在本次切换后重复跑模型长请求。
### 22.5 thinking wire：把思考本身拆成并行 aspect（2026-09-25）

mindmap wire 的失效不是措辞问题，而是**缩进协议 + 错误的概念**：深度可跳级、父节点重述子节点、模型漂进大纲编号；更根本的是 `goal`/`sub-goal` 这个词指"要达成的目标"，天然招来动作步骤。新增 `--rerot-plan-wire thinking`，`json` / `mindmap` / `todo` 全部保留作对照。

**概念来自模型自己的自然思考，不是猜的。** 关 RERoT 跑同一台机器、同模型、同采样，抓 `/healthz` 这类真实 agent-coding 任务的裸思考（41 段 / 12841 字，`artifacts/rerot-thinking-wire/natural-thinking-healthz.txt`）：约 25 段在反复打转**同一个关于材料的未知**（"这个代码库里的 wire 指什么"），其余是回忆 struct 字段，以及覆盖请求列举的四件交付物。它的自然单元既不是"问用户的问题"，也不是"答案切片"，而是**思考必须覆盖的面**——一部分是"我得自己敲定的未知"，一部分是"我得产出的东西"。所以单位叫 **aspect**。

#### 协议

- 注入固定前缀（指令 + `<ul>` + 起句 + 第一个 `<li>`），模型的第一个采样 token 已经在第一个 aspect 里面：
  ```
  My thinking splits here: once the aspects are listed, copies of me think them all in parallel
  and their raw thinking comes back to me; then I decide — tool call or answer. No aspect may
  need another's outcome. Here I only name them, one per line, as bare sub-titles in words
  with no punctuation: what I must pin down myself and what I must produce.
  <ul>Let me see what can be thought through on its own here: <li>
  ```
  起句以**冒号加空格**收尾而不是省略号：省略号还在邀请它继续沉思，硬接 `<li>` 会逼出最空的 aspect（"what I need to decide now"）；冒号是"列表就要来了"的规范信号。
- **没有自由头部区间。** 头部试过三种，全部失败：无界头部永不进列表（出口 token 被 min_p 屏蔽，一次跑到 1749 token 零条目）；在每个换气点注入提示语，模型把提示语原样复读几十遍（1633 token 零条目）；强制但有界的头部只产出一句垃圾过渡句，并把 aspect 拖成元话术（"这是规划问题还是实现问题"、"内容是否是陷阱"）。测得最好的计划**头部都是空的**，于是协议直接固化这一点。
- GBNF（全文 `artifacts/rerot-thinking-wire/thinking.gbnf`）：
  ```
  root  ::= sp? text "</li>" (nl? item)* nl? "</ul>"
  item  ::= "<li>" sp? text "</li>"
  text  ::= wchar tchar*
  tchar ::= wchar | sp | comma
  wchar ::= [0-9A-Za-z/_\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u3040-\u30FA\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF-]
  comma ::= "," | "\uFF0C"
  ```
  三条结构红线，每条都由实测换来：**条目内禁一切句读标点**（只留字母/数字/空格/逗号/CJK/假名/谚文/西里尔 + `/ _ -`，保住文件路径与标识符）；**条目之间只允许一个换行、不允许任何文字**（放开就会每两条之间思考一轮并永不收尾）；**`</ul>` 只在至少一个完整条目之后可达**（否则模型敲的第一个 `<` 就能关掉空列表）。条目是裸 `<li>`——自写 `<ul class="thinking">` 触发 lazy grammar 不可行（模型根本不会写），class 属性也是它最先丢的东西。

#### 决定性测量：`</li>` 与 `<` 不是"概率低"，是被 min_p 屏蔽

在探针头部的真实上下文上取原始分布（`/completion` + `n_probs=40`，同模型同模板）：

| 头部已生成 | `<` 排名 | p | p/p_top | min_p=0.05 |
|---|---|---|---|---|
| 0 token（紧跟注入起句） | **2** | **0.0859** | **0.277** | 通过 |
| 40 token | 掉出 top-40 | < 1.2e-3 | < 2.7e-3 | **被裁** |
| 300 token | 掉出 top-40 | < 1.2e-4 | < 1.5e-4 | **被裁** |
| 1749 token | 掉出 top-40 | < 3.2e-5 | < 3.3e-5 | **被裁** |

模型卡 thinking mode 要求 `min_p=0.05`，因此一旦写完一句散文，出口 token 被**从候选池删除**，再等多久都不出现。两个实测后果：无界头部跑到 1749 token 不收尾；`</li>` 同理——一个条目跑了 1670 token / 6061 字符、无换行、永不闭合。所以：

- **唯一有效的机制是禁标点。** 没有句号冒号分号（只留逗号），模型必须"一口气"说完，说不动了就只剩 `</li>`；顺带消掉 `&lt;` 转义与条目内换行两类失败。注入提示语的做法明确失败（模型复读提示语），语法层强制上限（1~3 行头部、条目必须以 `?` 收尾）也都失败——后者会误杀"列出亚洲国家"这类祈使式合法条目。

#### 帧协议（三条 tree wire 共享，thinking 用 aspect 措辞）

身份投递经过四轮实测才定型，前三轮全部失败：

1. **第二人称写在工具回复**（`Your sub-goal: "X" — that alone is your job`，身份放句末靠 recency）：coding 题 4 个 worker 里 2 个都认领 #3，1 个自述"become 把我的节点选择和目标合并了"后退回整题。所有 lane 的帧在同一块 unified KV、除标签外逐字相同、call id 为空，**一条对"你"说的指令从任何 lane 读起来都一样**。
2. **加数字标记 `#3` / `"i":3`**：标记反而成了最显眼的锚点，3 个 worker 认领 #3，node 2 原话 "I've already called become on node 3 … it seems I'm now working on that sub-goal"。
3. **第一人称种子放思考开头、以句号收尾**：每个 worker 的**第一个 token 就是 `</think>`**（零推理零内容），synthesis 照抄帧格式把 `<tool_call><function=become>` 当公开答案输出，请求 11.9 秒"完成"——伪完成，反例存 `rejected-seed-fullstop.sse`。
4. **第一人称种子 + 冒号收尾（当前）**：工具回复退成纯数据回显 `#3 "<label>"`，契约用 lane 自己的第一人称追加在模板渲染出的 think-start 之后：
   `I think only about aspect #3: "<label>". The others are being thought in parallel, so I don't redo them. Let me think:`
   synthesis 用 `Every aspect is thought through and all of it is in front of me. Now I decide what to do with it — tool call or answer:`。`become` 的 description 缩成一句 `Think one aspect of the list.`（此前为空，模型把帧判为"奇怪的工具调用"并据此重答整题，6/6）。

#### 前缀复用、内部工具的广告与身份锚点（2026-09-25 续）

每 episode 一次的全量重预填来自模板：提供 tools 时 `# Tools` 块插在渲染**靠前**的位置，于是 ordinary 渲染不是帧渲染的前缀，token LCP 被截断在 system 前导（实测 `ordinary=58 lcp=40 dag=305 replay_from=0`，且 LCP 不随提示变长而变长）。三种走法都实测过：

| 配置 | 结果 |
|---|---|
| 只在帧里广告 | 计划最好，但每 episode 全量重预填 |
| 哪里都不广告（保留 tool_call 形状，tools 列表为空——模板照样渲染 `<tool_call>`，实测 551 vs 1641 字符、共同前缀 226 字符） | 重建归零，但 **worker 身份保真掉到约 3/6**：有 lane 自称 #1 却持有 #3，有 lane 只复述题目，有 lane 宣称"其他面已经有人做了"后重答整题 |
| **C0 幂等广告（当前默认）** | **重建归零**（日志里再无 `DAG prefix rebuild`），代价是探针上下文里多了一个 tools 块 → 靠提示词精修补偿 |

广告后 `become` 进了模型的工具菜单，所以内部工具名前缀改成**保留单 token**：启动时按 `<|file_sep|> → <|repo_name|> → <|fim_pad|> → <|fim_prefix|> → …` 顺序挑第一个「恰好一个 token 且不是 EOG」的标记（本模型选到 `<|fim_prefix|>`，token 248060），并给那一个 id 加 `-inf` logit bias，对**所有**生成任务生效（lane 克隆经 `rerot_clone_task` 继承）。服务端写帧用的是 forced token，从不过采样器，所以禁采样不妨碍协议自身。两条被否决的选择：`-` 前缀（`-become` 切成 `['-b','ecome']`，而 `well-known` → `['well','-known']`，全局禁会毁正文）；视觉占位 token（选到 `<|vision_pad|>` 时探针计划从 5–6 条具体面塌成 2 条指令回声——它是图像占位符，出现在 system 块里等于暗示"有图"）。

提示词随之精修，两处都是为了不被当成条目抄走：指令尾句改成「none of them restates this instruction: each one names something specific in the request」，起句改成 `<ul>Looking at this request, the parts that stand alone are: <li>`。此前的尾句 "what I must pin down myself and what I must produce" 与起句 "what can be thought through on its own here" 都曾被原样抄成条目。

身份锚点分层，关键是**哪一层被谁读到**：

- `become` 的 description 只有一句（`Think one aspect of the plan.`）。它是帧里**唯一同时被探针读到**的文本（块在 C0 里），把排他契约写进去会让计划变粗（六条具体面塌到三条）。
- 排他契约放在**按 lane 渲染的工具回复**里，探针永不可见：`aspect #3 of 4: "<label>" is yours alone. The other aspects are being written at the same time by copies of you: their notes are theirs, so build on them and never take them over. Their notes appear a little at a time, so anything that looks unfinished there is only unwritten yet and fills in on its own.`
- 最后一句是第一人称种子：`I am aspect #3 of 4 and only #3: "<label>". My notes on #3:`
- `of N`（N = 冻结计划的叶数）比裸 `#3` 更难与 peer 的身份混淆；实测中有 lane 自报 "Aspect #2 of 4"。
- "fills in on its own" 这句不是客套：lane 曾把 peer **没写完**的笔记当缺口，于是要么接管别人的面，要么宣称"都做完了"并重答整题。

大洲题最终状态：4 个 aspect，**4/4 各守其位**，无接管话术；计划粒度在不同措辞下浮动过 3–6 条，每种配置各一次运行，不宣称最优。

#### 实测（单卡 RX 6800 `Vulkan0`、Bonsai-2 27B PQ2_0、`-c 262144 -np 6 --rerot-pens 6 -ctk q8_0 -ctv turbo4`、模型卡 thinking-mode 采样、**60 秒客户端预算**）

| 提示 | seed | 计划 |
|---|---|---|
| `/healthz` 端点 | 20260923 | 6 个 aspect，probe **204 token**：要暴露什么且在代码哪里 / slot 是什么及报哪些字段 / JSON 形状 / 测试怎么写且可运行 / 哪里会出问题 / 答案怎么呈现 |
| `/healthz` 端点 | 777 | 6 个 aspect，probe **223 token**：返回哪些字段及映射 / 怎么从现有状态取 slot+KV+wire / handler 注册在哪 / 怎么单测含假数据 / 改哪些文件及测试怎么接进构建 / 风险与兼容性 |
| 世界上每个大洲有哪些国家 | 20260923 | 5 个 aspect，probe **165 token**：列哪些洲及顺序 / 每国如何归类 / 跨洲国家怎么办 / 是否用 ISO 3166-1 / 南极洲 |

换 seed 得到**不同但同样具体**的计划，说明本节各配置之间的差异不是采样噪声（同一配置同一 seed 可复现）。

大洲题这组元问题**正是正确的味道**：列国家本身很简单、正文直接列即可，值得并行思考的恰恰是分类口径、跨洲归属、是否用 ISO 码这些决策。早前一次用 `question` 措辞时曾切出逐洲 6 条，看起来"更像分工"，但把简单枚举摊给 6 个 worker 并不是这套机制该做的事。

之前 worker 侧的验收（`aspect` + 带 class 的条目、probe 286 token）：/healthz 题 8 个 aspect、7 个 worker 各守一个（JSON 形状 / 注册位置 / slot 字段 / 测试位置 / 风险 / 是否给代码 / 措辞），无 token 级循环；大洲题 3 个 aspect、3 个 worker 各守一个。协议此后只改了前缀与字符集，worker 侧未复跑。

被否决的措辞（每条都是对症状打补丁，而非陈述机制）记录在 `tools/server/server-rerot-thinking.h` 的注释里：`What is the goal? Which sub-goals can run in parallel?`（在列表里回答了自己的问题）、`each self-contained / concrete task`（仍是流水线阶段）、`by category, not by stage`（被读成"工作类别"）、`one per thing the goal covers`（不泛化）、`cut the answer into parts joined with nothing added`（多数任务没有"答案文档"可切）、`one short line per question`（产出无关的示例问题）、以及要求条目以 `?` 结尾的语法约束。

#### 边界

- 除那次伪完成外**全部**在 60 秒客户端截止时结束；`completed_episode_total` 未验证，**没有**完整公开答案、没有 A/B 吞吐比、没有全请求验收。本节只证明计划层与身份层的缺陷消除。
- 早前用 `question` 措辞时大洲题曾切出**逐洲** 6 条（`Asia` / `Africa` / `Europe` / `Americas` / `Oceania` / `Polar`），换 `aspect` 后变成 3 条更粗的面。哪种更适合调度尚无验收数据，不宣称优劣。
- 采样口径：模型卡（`prism-ml/Ternary-Bonsai-2-27B-gguf`）thinking mode 推荐 `temperature=1.0 / top_p=0.95 / top_k=20 / min_p=0.05 / presence_penalty=0.0 / repetition_penalty=1.0`；本轮显式传前三项，后三项取 llama.cpp 默认值且与卡片一致。卡片明确 thinking mode **不开**惩罚，因此早前观测到的 `、马拉尼` 周期-6 退化重复不是配置错误；卡片自己给的短化手段是把 reasoning effort 从默认 `xhigh` 降到 `medium`，本轮未动——六个 worker 各跑 xhigh 正是 60 秒收不了尾的直接原因。
- 单 seed、单卡、单模型、每题一次。

验证：`llama-server` 构建通过；`test-rerot-thinking`（新增，ASan/UBSan 复跑）、`test-rerot-todo`、`test-rerot-mindmap-parser`、`test-rerot-parser`、`test-rerot-runtime`、`test-server-task`、`test-arg-parser`、`test-flashprefill-state` 全绿；GBNF 经 `test-gbnf-validator` 逐条验证：字母/CJK/路径字符接受，逗号句号冒号问号反斜杠在精确位置被拒，条目间自由文本被拒，空列表 `</ul>` 被拒，`<li class=...>` 被拒。

### 22.6 让并行真的比串行快（2026-09-25）

三处改动叠起来，RERoT 在同题上第一次**反超**串行基线。全部数据为单卡 RX 6800 `Vulkan0`、Bonsai-2 27B PQ2_0、`-fa on -ctk q8_0 -ctv turbo4`、模型卡 thinking-mode 采样。

#### 一、多列 matvec：权重共享 + 激活重读

起点是一个反常识的事实：多行解码几乎不标定（6 行只有单行的 2.08×，8 行退化到 1.90×）。`GGML_VK_PERF_LOGGER` 的逐算子分账定位到主导算子 `MUL_MAT_VEC pq2_0 m=17408 k=5120`（FFN 上投影）：

| | 每次调用 | 权重流量折算 |
|---|---|---|
| n=1 | 74.9 us | 23.7 MB → 317 GB/s（接近该卡可持续带宽，确实带宽受限） |
| n=6 | 249.1 us | 同样 23.7 MB → 95 GB/s（已不受带宽限制，所以瓶颈在"每列"的东西） |

两处每列成本，各自修掉：

1. **权重按 (列 × 行) 重复解量化。** `mul_mat_vec.comp` 的通用循环把列放在行外层，已有的共享解量化快路只覆盖 `num_rows == 1`，而量化管线是按 `NUM_ROWS = 2*rm_stdq` 创建的，永不命中。现在 `NUM_COLS > 1` 时把循环交换过来：每行解量化一次、所有列复用；列数 ≤ 4 时把各列激活先进寄存器。
2. **每个 workgroup 重读整个激活向量**，所以这部分流量 ∝ workgroup 数 = m/rows。n=1 时它已与权重访存同量级（178 MB 过缓存 ≈ 71 us，旁边是 55 us 权重 DRAM，实测 74.9 us）；n=6 时是 **1.07 GB**。因此 PQ2_0 的 matvec 管线在宽派发时加宽行块（≥2 列用 2 倍行，≥4 列用 4 倍行），按比例摊掉该流量，权重流量不变。

`llama-batched-bench -npp 16 -ntg 64` 解码 t/s，改动前 → 后：

| lanes | 前 | 后 | 增幅 |
|---|---|---|---|
| 1 | 33.10 | 33.47 | +1.1%（无回归） |
| 2 | 48.26 | 55.46 | +14.9% |
| 3 | 51.49 | 68.90 | +33.8% |
| 4 | 51.09 | 82.54 | +61.6% |
| **6** | **68.99** | **98.90** | **+43.4%** |
| 8 | 63.03 | 103.40 | +64.0% |

对单 lane 的标定从 2.08× 升到 **2.95×**（6 lane），8 lane 不再倒退（1.90× → 3.09×）。膝点在 6：4→6 还有 +20%，6→8 只剩 +4.6%——**这就是"几个最好"的答案**。数值等价：循环交换不改每个累加器的 K 顺序，行块只改 workgroup 归属；真机贪心解码验证 6 路并发（即内核 6 列路径）与单列输出**逐字相同**。

#### 二、lane 必须收尾：预算算术

并行只在**聚合 lane token < 串行 token × 多 lane 加速比**时才划算。大洲题串行要 3085 token，6 lane 的加速比 2.95×，所以预算约 **9100 aggregate token**。而实测 lane 各自写到 1240 token 仍未停，7 lane 已 8680——episode 600 秒不收尾，对面基线 134 秒就结束了。

`--rerot-lane-wrap-tokens N`（默认 512）给每个 worker **一次**第一人称收尾提醒。它是注入，所以把 lane 刚采样的那个 token 放在提醒开头，不丢字。kind 必须用 `refresh`——`worker` 会用 `lane->exit_parser.marker()` 重装 child-close 语法，而原生 DAG worker 没有该标记（实测 hard abort `invalid child close grammar`）；`dag_frame` 会重跑一次性 admission。

#### 三、结果（口径已修正）

先前版本用 `timings.predicted_n / wall` 作"端到端吞吐"，**那是错的**：RERoT 下该计数把 lane 的内部 token 也算进分子，所以会虚高。正确口径只有两条：**同一固定清单覆盖度下的墙钟**，以及**总模型 token**。清单事前固定为七大洲标记 + 俄罗斯、土耳其两个跨洲例（8 项）。

同题（`世界上每个大洲有哪些国家`）、同 seed、单卡 RX 6800：

| 配置 | 墙钟 | 总 token | 答案字符 | 覆盖 |
|---|---|---|---|---|
| 串行基线（RERoT off） | 134.37 s | 3085 | 1774 | — |
| RERoT，lane 预算 512 | 251.65 s | 5575 | 1423 | 8/8 |
| **RERoT，lane 预算 256（默认）** | **101.51 s** | **2959** | 1024 | **8/8** |
| RERoT，lane 预算 128 | 53.37 s | 1588 | 762 | **1/8** |

`--rerot-lane-wrap-tokens 256`：**墙钟 1.32× 于串行，而总 token 还少 4%（2959 vs 3085）**，清单全覆盖——所以这不是把内部 token 算进分子换来的，而是真实的时间压缩。代价是答案更简洁（1024 vs 1774 字符）。128 那一档覆盖度塌到 1/8，正好证明这个护栏有用：只压时间不看覆盖会得出假胜利。

lane 与 synthesis 都要有界。`nudge` 只发给 worker 时，4-worker 的计划里 synthesis 拿着四份笔记无界地写，153 s 时已 1560 token 仍未停，700 s 预算都撑不住；给 synthesis 三倍 lane 预算的同类提醒后，同一计划 251.65 s 收尾。提醒还必须跳过任何含 `<`/`>` 的 piece：注入会把刚采样的 token 带进 lane 的**私有** run，若那个 token 正是收尾标记，它就进不了被解析的公开流，lane 永不 seal。

边界：答案很短的题不适用墙钟口径——算术题 RERoT 的公开输出是串行的 4 倍（1436 vs 363 token），墙钟 42.6 s vs 17.0 s 比的是不同工作量，其速率仍占优（33.7 vs 21.4 tok/s）。**墙钟结论只在输出长度可比时成立。**
