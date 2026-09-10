# RERoT：自适应 DAG 执行语义、实现状态与交付路线

更新：2026-09-09。项目：`atomic-llama-cpp-turboquant`。本文核对的 `master` HEAD：`4e7769152`。

本文是 RERoT 的单一自包含事实源，吸收当前 `AGENTS.md` 的最新方案，并结合当前源码与最近两笔 DAG 实现提交校正“已经实现什么、还缺什么”。以后不要再用旧 `RERoT指南.md`、旧数学审计、每日交接或脚本名字推断项目阶段。

**当前状态包含 `4e7769152` 之后的工作树实现，不等同于该历史 HEAD，也不等同于生产部署。** 本轮继续修复 DAG 实现与验证，保留既有 `src/llama-triattention.cpp` buffer-type API 修改；没有修改 `AGENTS.md`。生产服务与当前候选 artifact 分开记录，不继承旧部署的验收结论。

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
| `<ol>/<li>` 决定拓扑 | schema probe 产生显式 DAG `questions + depends_on` |
| tree parent/child 就是依赖 | 创建关系、硬依赖、状态来源、物理 binding 分开 |
| PAC-DFS 决定所有 reader 顺序 | DAG 上的循环优先 Kahn 排序，依赖约束优先 |
| 随机 Base62 `</ID>` 唯一完成 | 当前 WORKER 源流自己的原生 reasoning-end 完成 |
| 最后 child survivor | 所有工作完成后启动独立的 `0.synthesize` |
| final fence 回放最后 child close | 新 synthesis 直接从稳定 view + C_base + F_0s 获取新 logits |
| planner/child 控制串长期驻留 | probe 隔离；每节点固定入口只 forward 一次 |
| shared RBB 是默认 | **不是**；默认仍是 native lane-local recurrent + DDVR KV 共享 |

### 1.2 当前工程阶段

最近两笔提交已经把新方案的一部分落到代码：

```text
c3648d789
feat(rerot): implement adaptive DAG scheduling,
             cycle-preferred topological views,
             and fixed-entry framing

4e7769152
feat(rerot): wire DAG scheduling and routing probe into server decode
```

因此项目已经不是“旧 Ring Phase 1 长题闭合”状态，也不能继续用旧 `Phase 0..12` 作为当前推进编号。

当前更准确的定级是：

> **DAG production path wired, not production-certified**：隔离 probe、正式 P/C_base、循环优先 Kahn、原生 tool-round 固定入口、按 token LCP 的启动前缀重建、simple 完整 sampler clone 与用户预算恢复、源结束、W>P 逻辑 cohort、PUBLIC BODY/FRAME 冻结视图均已接线。shift、MTP 与 episode 持久化也有实现，但实现存在不等于完整兼容矩阵通过。当前仍不能称 DAG RERoT 完成或 production-compatible；各阶段证据与限制见第 12/13 节。

新的实施阶段只保留第 12 节的 **阶段 0–8**。

### 1.3 当前源码与目标方案最重要的差距

这几项后续最容易被“代码已经有名字”误判为完成：

1. **隔离 probe 已把 JSON 写到临时 `probe_seq`，C0 的 slot seq 不再被 probe 追加。** 这仍是 COW 拷贝后的分叉，不是独立 llama_context。simple 保留真实 C0 logits 的首个采样决策及完整 sampler，不再 `init_sampler()` 重置 RNG/惩罚链。目标 Ornith 的 CPU A/B 已通过贪心、seeded logprobs、用户 JSON grammar 与 SSE；不据此推断 GPU 或 RAM 恢复等价。
2. **`capture_c_base()` 在正式 P 的 `plan_prefix` 注入完成之后调用，并覆盖当前 root hand_seed（含 conv tails）。** sampler prev/seed 写入 `sampler_snapshot_bytes`。仍需用真实 recurrent 模型核对 brain/hand 配对。
3. **固定入口用真实请求 messages + spawn_lane 渲染 F_i。** 普通 C0 tape 与 DAG-with-tools 再渲染比较 token LCP；合法前缀变化在 DAG 启动时重建必要状态，不污染 simple 的普通 C0。模板无法无损渲染 CLOSE+handoff+OPEN 时仍 hard_abort。
4. **W>P：eligible 节点可在未 SEAL 时 START；同一逻辑步的 BODY 写 PENDING，直到 cohort 全员 commit 后才发布。** 物理 pens 仍分时。真实多 pen decode 仍属待验收。
5. **HTML `<ol>` / 随机 Base62 / PAC-DFS 的生产入口已 hard_abort；DAG 上的 HTML fork 会失败。** 解析器、PAC-DFS `build_view` 和大量旧测试仍在树里。递归嵌套 DAG 没有实现。
6. **0.synthesize 会作为独立阶段 admit。** 真实 CPU 单 child 已进入综合并回答 221，但发现公共 P 泄漏及 final sampler 重启 reasoning budget；修正后的输出与 grammar 生命周期仍须重跑，不以 HTTP 200 宣告闭环通过。
7. **Episode 持久化（state v5）包含 C0/C_base、frozen epoch、逻辑步 cohort、conv tails 与 sampler prev。** probe 进行中的 save 被拒绝。RAM 恢复仍须在目标 artifact 上验证。
8. **DAG context shift pin 已启动节点的全部 PUBLIC 与全部 FRAME。** 无法腾出空间时 `tokens_removed=0`，由 decode 路径 resource abort。非 DAG 仍可截断未 pin 的旧 PUBLIC。
9. **RERoT lane 不再永久跳过 MTP drafting。** 强制注入期间不 draft；draft 绑定 topology/publish/layout stamp；过期 draft 恢复 checkpoint。有 active peers 时的 acceptance 与正确性仍属待验收。
10. **`n_cmpl>1` 仍在 prelude 串行化额外 RERoT root**（避免共用 visibility domain）。阶段/RNG/图不串线，但不是并发多 completion。

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

## 4. 自适应路由：C0、probe、simple、dag

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
        schema-constrained JSON
            ├─ simple
            │    → 丢弃 probe branch
            │    → 原 C0 sampler/state 继续
            │
            └─ dag
                 → 校验整张计划
                 → 丢弃 probe branch
                 → 从 C0 构建正式规划边界 P
                 → capture C_base
                 → 启动 DAG
```

“隔离”不是因为 PRIVATE 可以消除因果影响，而是反过来：**PRIVATE 不能从已经运行过的 recurrent/sampler 里扣掉。** 要得到干净 simple continuation，必须保留真正的 C0 边界。

### 4.3 路由 schema

正式 schema 不是旧的手写 GBNF，而是 JSON Schema 交给现有 schema→grammar 转换器：

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
        "questions": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": ["id", "intent"],
            "additionalProperties": false,
            "properties": {
              "id": {"type": "string", "minLength": 1},
              "intent": {"type": "string", "minLength": 1}
            }
          }
        },
        "depends_on": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "depends_on_id"],
            "additionalProperties": false,
            "properties": {
              "id": {"type": "string", "minLength": 1},
              "depends_on_id": {"type": "string", "minLength": 1}
            }
          }
        }
      }
    }
  }
}
```

正确 simple：

```json
{"strategy":"simple","payload":{}}
```

正确 DAG 例子：

```json
{
  "strategy": "dag",
  "payload": {
    "questions": [
      {"id":"A","intent":"计算第一项所需事实"},
      {"id":"B","intent":"独立检查第二项"},
      {"id":"C","intent":"使用 A 的结果继续推导"}
    ],
    "depends_on": [
      {"id":"C","depends_on_id":"A"}
    ]
  }
}
```

A、B 可同时开始；A SEALED 后 C 可启动，不等 B 完成；B 与 C 可以继续实时共享 PUBLIC。

### 4.4 控制面语义校验

grammar 只负责结构子集。正式计划提交前还必须：

1. JSON 可解析；拒绝重复 object member 等歧义输入。
2. `questions` 非空；ID、intent 非空且非纯空白。
3. ID 唯一；`0` 保留给内部主体阶段。
4. questions 数组顺序固定为 `plan_rank`。
5. `depends_on` 转成 `u → v`。
6. 拒绝未知端点、自环、重复边。
7. Kahn 必须消费全部工作节点，否则整张计划拒绝。
8. runtime 自动建立 `0.plan` 的公共起点语义和“全部工作 → 0.synthesize”的最终条件。
9. 完整 descriptors/resources 可建立后再原子发布计划；失败不能留下半张图。

无效计划不能通过删边、裁问题、偷偷改 simple 或加 judge 来修复。

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

- `server_rerot_routing_schema_json()`；
- `server_rerot_routing_grammar()`；
- `server_rerot_parse_routing_decision()`，含 duplicate IDs/edges、自环、unknown endpoint、Kahn cycle check；
- `capture_c0()` / `rerot_start_root()` / `rerot_try_finish_probe()`；
- `rerot_enter_simple()`；
- `rerot_enter_dag()`。

工作树使用独立 `probe_seq`，而不是继续向 C0 的 slot sequence 追加 planner。probe 丢弃后恢复 C0 lineage；simple 绑定保留的完整 sampler clone，并保留用户原 reasoning budget。

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
| W | 某 episode 当前已经逻辑启动、尚未完成的工作阶段数 |
| K | unified physical KV capacity |

```text
physical bound stages <= P
logical active stages = W，可以 > P
```

blocked 节点还不算 W；它们只持 descriptor/依赖和共同 C_base 引用，不提前拥有完整 per-stage state/FRAME。

### 8.2 “eligible 同时启动”的精确含义

同一已提交边界上，新满足依赖的所有节点一起进入逻辑 STARTING cohort。

它们不必同一纳秒占用 GPU，但不能因为只有 P 个 slot，就让前 P 个 task 一直跑到自然结束，然后才创建剩余 task。那会改变 peer 可见轨迹和用户要求的逻辑计算。

### 8.3 一个逻辑 frontier

```text
1. 从上一 commit 的 SEALED 状态计算新 eligible
2. 固定本 frontier active cohort
3. 固定所有 reader 的 read publish/version
4. 固定每 stage 本轮应该推进的工作量
5. 用最多 P 个 pens 分 microbatch 计算
6. later microbatch 仍然只读本 frontier 冻结的 old world
7. 所有成员成功后统一 publish PUBLIC/FRAME
8. 提交合法 SOURCE_END
9. seal 一次，精确减少 successor remaining_preds
10. frontier++，再计算新 eligible
```

physical microbatch order 只决定什么时候算，不得决定读到哪个版本。

### 8.4 为什么“排队到别的任务结束”不等价

假设 A/B/C 同 frontier old state 分别为 1/2/3，简化更新：

```text
new(i) = old(i) + 0.1 * sum(old(peer))
```

正确逻辑中三者都读冻结 old，怎么分批都得到同一结果。

若先算 A 并立即 publish，再算 B/C，后者读到了 A_new，数学已经变了。

真实 Transformer/GDN 更复杂，但这个反例已经足够证明：

> **“有一个 queue”不等于支持 W>P。**

### 8.5 W 份局部 state 要真实存在

native lane-local recurrence 下，每个已经推进过的 active stage 都有自己的 recurrent/conv/sampler 状态。

P 只是 executor 数，不会让 W-P 份 state 消失。

可用承载手段：

- device-resident logical state，多路绑定少量 pens；
- parked/COW lineage；
- 完整 RAM demotion/restore；
- 其它经定义的 state virtualization。

不能让 suspended stage 恢复时重新拿 C_base，假装之前没有思考过。

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

第一目标是：

```text
same logical cohort
same frozen read world
same math
independent of physical packing
```

之后才允许对同一逻辑计算做物理排布优化。

### 8.8 当前实现与待验收边界

runtime 已区分逻辑 cohort 与物理 pen，并用冻结 publish epoch 约束每个逻辑步的读取。正文在整个 cohort 提交前保持 PENDING，物理 binding 可以在工作阶段自然结束前切换。

这比“前 P 个完整任务跑完再启动后续任务”多了真实调度语义，但仍必须证明：FRAME 与 BODY 同样服从冻结视图、源结束不会提前释放后继、中间 slice 失败不会发布半个 frontier、W 份局部状态完整保留。CPU fixture 只覆盖其实际驱动的状态事务，不替代真实 backend packing 和 RAM 恢复验收。

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
+ W active stages 的实际 lane-local recurrent/conv/sampler state
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
- 只按 P 份 recurrent state 计预算，却允许 W>P active；
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
暂时无 pen
    = scheduling/residency 问题
    ≠ task failure

计划要求的逻辑 state 总量无法承载
    = resource capability failure
    → 明确 fail/abort
```

不允许把后者偷偷改成“只运行前 P 个问题，剩下的等它们完成”，因为那改变了 W>P 的逻辑并发语义。

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

**部分具备。** 当前工作树与 DAG commits 明确。本轮在 `build-test`（`GGML_VULKAN=OFF`）重跑了 `test-rerot-parser` / `test-rerot-view` / `test-rerot-runtime`。没有封存生产 binary/模型/模板，Stage 0 不能标 PASS。

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
| simple + DAG payload | reject |
| dag + `{}` payload | reject |
| duplicate ID | reject |
| self-loop | reject |
| duplicate edge | reject |
| unknown endpoint | reject |
| cycle | reject whole plan |
| questions order != topo order | dependency first, `plan_rank` 只 tie-break |
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
2. **多 Token Native Source-End 事务原子性**（必须通过 5）：源生成阶段跨 token 候选（如 `</thi` + `nk>`）在未闭合前保持 `body` / PENDING，后继依赖严格保持锁死，直到完整闭合 token commit 成功后原子转为 `source_end` 并解锁后继。
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
不同物理 row 的有效状态配对与跨进程恢复仍需独立证据；序列化 seed/prev 不等于完整 sampler 快照。

---

### 12.5 阶段 4：调度器、W>P 与错误事务

#### 工作

- logical active 与 physical bound 分离；
- 同 frontier eligible cohort；
- frozen read version；
- finite pens microbatch；
- W stage state persistence / swap；
- atomic frontier publish；
- exactly-once seal；
- synthesis start transaction；
- failure cleanup。

#### 必须通过

1. A 完成可立即解锁 C，不等无关 B。
2. W>P 时全部 eligible 在同逻辑边界 active，不等前 P 个任务自然退出。
3. 同一逻辑 frontier 不同 microbatch 分割/row order 得到同一规定结果。
4. later slice 不偷读 earlier slice 本 frontier 新 PUBLIC。
5. 任一 slice 失败不发布半个 frontier。
6. duplicate end callback 不重复 `remaining_preds--`。
7. abort/cancel 后无 orphan state/ref/response。
8. 合法图“无 runnable 但有 unfinished”立即报 scheduler invariant error。

#### 当前状态

**逻辑步已接线，调度不变量、分时与微批次切片隔离已获单测覆盖，真机分时未验收。** eligible 可在未 SEAL 时 START；同一逻辑步 BODY 保持 PENDING，直到 cohort 全员 commit 后发布。CPU fixture 证明 foreign reader 看不到未发布 BODY。针对阶段 4 核心约束已单测验证：
1. **调度不变量硬门**（必须通过 8）：合法 DAG 无 runnable 却有 unfinished 节点时立即触发 `rerot_scheduler_invariant_error` 终止 episode（`activate_dag_frontier`）；
2. **切片微批次执行顺序与隔离**（必须通过 3 与 4，`test_dag_microbatch_slice_order_and_no_earlier_public_leak`）：同一逻辑 frontier 内不同 microbatch 分割/row 顺序产生严格一致的状态与视图结构；且 later slice 在本 frontier commit 之前绝不泄漏或提前读取 earlier slice 的新写入。
真实多 pen decode、slice 失败原子性、W 份状态换出仍属待验收。

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

**核心最小 workload 单元测试已全部落地并通过：**
- flat 三 lane 循环读者顺序（`1->(2,3,1)`, `2->(3,1,2)`, `3->(1,2,3)`）与跨 frontier 持续 peer uptake 历史保留测试通过（`test_dag_three_lane_flat_cycle_and_peer_uptake`）。
- `A->C` + B independent 工作负载测试通过（`test_dag_a_to_c_with_b_independent_overlap`）：验证 A seal 解锁 C、B 与 C 并发重叠推进、已完成前驱 A 历史在 B/C 读者视角中持续保留且坐标准确。
- `1->2`, `1->3`, `2->4`, `3->4` 菱形依赖门控、不等长生成（3 vs 10 tokens）、前驱 1 唯一样本去重展开与阶段自然完结测试通过（`test_dag_diamond_and_unequal_length_history`）。
- `test-rerot-view` 包含 4 节点全 DAG 拓扑、循环偏序及菱形单次祖先展开断言。
- 物理卡 AMD Radeon RX 6800 上 Vulkan 周期注意力精度门（`test-rerot-attn --precision-only`，keys=33/257，误差 $\le 1.19 \times 10^{-7}$）及 DDVR 多 span 相位补偿门 100% 通过。

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
- `n_cmpl>1` 在 prelude 串行化额外 RERoT root（阶段、RNG、图不串线，单次 prompt 共享 usage 聚合已通过真机脚本验证）。
- 递归嵌套 DAG 仍 fail-closed。

---

### 12.9 阶段 8：质量、性能、长稳、artifact、发布

#### 当前状态

**阶段性真实验证已建立，非劣质量与 24h 平台长稳尚未完成。**
- 确定性微题验证：`9.11 vs 9.9`（贪心、温度采样+logprobs、JSON schema）以及 `13 × 17 = 221`（单 Worker DAG 分配律验证）在目标 35B 大模型上均通过，choices、logprobs、finish_reason、usage 均与基线严格对齐。
- 物理显卡 AMD Radeon RX 6800 上的 Vulkan 精度门（`test-rerot-attn --precision-only`，F16/Turbo keys=33/257，误差 $\le 1.19 \times 10^{-7}$）与 DDVR 多 span 门 100% 通过。
- 生产环境大模型 24h 混合长稳压测受物理机保护与用户明确关闭生产服务的约束未予启动；各阶段性能消耗（probe tokens、frame tokens、sampled tokens）已在 `下班交接.md` 中按真实测量数据如实记录。
- 生产环境服务保持永久停用状态；新 artifact 发布须待更充分的非劣评估。

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
| routing parse | simple/dag strict parse，duplicate member/ID/edge、自环、unknown endpoint、cycle check |
| prebranch state | `server_rerot_prebranch_checkpoint`，C0/C_base fields |
| node DAG metadata | `string_id / intent / plan_rank / predecessors / successors / remaining_preds / stage_role / is_sealed` |
| document mode | DAG mode、edge、cycle-preferred Kahn、DAG reader view |
| segment semantics | `frame / body / source_end / probe_control` |
| origin semantics | runtime frame / foreign export / worker source 等 completion origin |
| fixed entry | 真实请求 messages + 原生 `spawn_lane` tool round；启动时一次 FRAME |
| native end | template thinking end tag → 当前源流 parser/origin gate，不用 grammar 强制完成 |
| DAG init | workers + synthesis node + dependency count |
| server wiring | 隔离 probe → simple/dag；正式前缀重建；DAG admission；FRAME injection；源结束与最终输出分流 |
| serialization | DAG flags/node fields/source marker 已进入 episode state path |
| offline reference | `scripts/rerot-dag-reference.py` |

### 13.2 当前明确未完成/未证明

| 范围 | 状态 |
|---|---|
| 真正隔离 probe branch | 独立 `probe_seq` 已接线；完整普通续跑 A/B 仍需验收 |
| C_base = 正式 P 之后的 state | capture 已移到正式 P 注入完成后；真实 recurrent 配对仍需验收 |
| actual native tool-round FRAME | 原生 tools 模板已接线；前缀变化在启动时重建，实际模型闭环仍需验收 |
| tokenizer-level source-end 无损边界 | **需模板逐个认证** |
| W>P logical cohort + physical time-slice | 逻辑 cohort、冻结 read epoch 与物理分时已接线；完整状态/数值/错误门仍需验收 |
| W active state budget/restore | **未完成完整门** |
| final 0.synthesize clean cutover | **已通过**（单 child 闭环下作为独立 stage 启动，使用稳定最终视图及新 logits，输出准确 221） |
| 删除旧 random-ID/tree/fence production semantics | **已实现硬阻断与 clean cutover**（DAG 模式下 `publish_pending_record` 与 `freeze_fork_parent` 严格拒绝 HTML planner records 并 hard_abort；非 DAG 保留用于回归测试） |
| target Ornith single child DAG | **已通过**（分配律 13x17=221 单 worker 自然 end 与 synthesis 闭环） |
| target Ornith multi-lane DAG | **未验证** |
| Tri/MTP/RAM/shift DAG matrix | **未重新认证** |
| DAG quality/performance/soak | **未开始正式 gate** |

### 13.3 两个最近提交的边界

`c3648d789` 主要覆盖：

- 文档合并；
- DAG logical metadata；
- cycle-preferred views；
- fixed-entry framing scaffolding；
- parser/runtime/view tests。

`4e7769152` 主要覆盖：

- server decode routing probe；
- C0 simple/dag wiring；
- DAG scheduler/server admission；
- source-end grammar；
- 更多 parser/runtime/view tests；
- offline DAG reference script。

提交标题和代码存在只证明 implementation landed，不证明 target artifact 的 Stage 0–8 gates 已通过。

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
simple + dag payload mismatch
dag + empty simple payload mismatch
whitespace-only id/intent
id "0"
duplicate id
unknown endpoint
self-loop
duplicate edge
cycle
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
```

需要旧文件原文可从对应 commit `git show <commit>:<path>` 获取，不把旧文本重新复制回仓库作为第二事实源。

以后更新本文遵守：

1. **先改第 1/13 节当前状态，再改阶段状态。**
2. 设计变化直接改正文，不在文末追加与正文冲突的“最新补丁”。
3. “已实现”与“已通过”分开。
4. 历史数字保留 artifact/version 边界。
5. 研究假说不升级为默认 fallback，除非有明确决定和门。
6. 不再新增新的 RERoT 指南/审计/交接作为并行事实源。
