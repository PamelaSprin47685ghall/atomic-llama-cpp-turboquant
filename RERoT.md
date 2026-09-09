# RERoT：设计、实现与当前状态

更新：2026-09-09。项目：`atomic-llama-cpp-turboquant`。本次整理核对的 `master` HEAD：`47432e525`。

本文合并原实施指南、数学与语义审计、交接记录，以及 `AGENTS.md` 中的路线图。以后只在这里维护 RERoT 的设计和状态。**本次是文档整理与源码核对，没有重新构建、运行模型、部署或认证当前二进制。** 下文严格区分当前实现、历史测试结果和待验收目标。

## 1. 先看结论

**当前阶段是 Phase 1：核心真实长任务闭环验证。尚未通过，不能称为 production-compatible RERoT，也不能按 Phase 8 / Phase 9 已完成继续推进。**

核心并非从零开始：DDVR、树与队列、随机 ID 退出、child 原生递推、完整 final fence，以及多项 CPU/Vulkan 数值回归已有实现和局部证据。但局部测试通过，不等于长任务能完整、正确、自然结束；此前的兼容与性能成绩也不能自动继承到当前代码。

### 1.1 当前口径

| 项目 | 当前结论 |
|---|---|
| 主要对象 | Ornith / Qwen3.5 hybrid；CPU 作数值参考，Vulkan 是首要认证后端，Turbo4/Turbo2 是目标 KV 配置 |
| 跨 Lane 共享 | **DDVR 共享已提交的 PUBLIC K/V；child 的 GDN recurrence 和 conv 保持私有、原生递推** |
| Shared RBB | 仅显式科研模式，不是默认算法，也不是坏输出时自动切换的备用路径 |
| STRONG | **barrier-after**：peer 本 frontier 的 token 不得同拍可见；下一 frontier 才能读取 |
| child 完成 | 模型自然生成自身精确的随机 `</ID>`；换行、EOG、超时、长度或重复次数都不能替代它 |
| final fence | 已有闭合前 checkpoint、原 token 回放、逐行确认与幂等处理；仍须随完整长任务验收 |
| 当前实验基线 | TriAttention OFF、MTP OFF、RAM checkpoint/demotion OFF；隔离 context shift 等状态迁移 |
| 下一项工作 | 原题“世界上每个大洲有哪些国家”的多 child 长任务闭环，见第 7 节 |
| 发布资格 | 未取得：完整兼容矩阵、资源压力、质量、性能、长稳与同一 artifact 验收均未收齐 |

`7509335d1` 是历史上的 **Core Correctness Candidate**，不是当前 HEAD，也不是生产认证标签。此后合并过 FlashPrefill、XKV-SR 等改动；当前 HEAD 又包含 Vulkan MoE 类型约束和不支持 pipeline 时的回退修复。旧提交的通过记录不能替代这些改动后的重新验证。

### 1.2 这次统一掉的冲突

| 旧文中的说法 | 以后采用的说法 |
|---|---|
| “global recurrent 共享是正式默认，lane-local 只是 Scheme 0” | child 默认已回到 lane-local native recurrence；shared RBB 降为研究模式 |
| “联合 forward 内 peer 当前 K/V 可互读” | 物理上先写 KV 不代表逻辑提交；STRONG 必须隔开 read / write / barrier |
| “child 首个换行后强制 close” | 已撤销；正文可多行、多段、含代码块，精确 owner delimiter 才结束 |
| “正文出现 `<ol>` 就递归 fork” | 只有显式 planner phase 中的完整规划列表有拓扑含义；worker 的普通列表只是正文 |
| “恢复 fork seed 可以清除私有指令” | 错误；会丢掉后续局部状态。serial tail 保留当前状态，fence 只恢复对应闭合前 checkpoint |
| “fence 只是换 view / synchronize，尚未回放” | 属于早期记录；后续已实现正常 batch 路径中的完整回放，见第 6 节 |
| “hand 是 FP16，旧 SEE2 可继续用” | persistent hand 已改 F32；SEE3 明确拒绝旧 SEE2，不隐式有损转换 |
| “已到 Phase 8 / Phase 9” | 2026-09-09 已纠偏回 Phase 1；脚本名、提交标题或旧阶段编号不是验收结果 |
| “HTTP 200、CTest 全绿、关键词评分 PASS 就通过” | 传输、局部数学、完整任务质量是三种证据，不能互相代替 |

## 2. RERoT 到底做什么

RERoT 是 Recursive Elastic Ring-of-Thought：在一次请求的思考过程中，把可并行推进的任务拆成多个逻辑 Lane。它不是多开几个独立 completion，再找一个模型汇总；Lane 在运行时共享公开推导，最后由自然退出的最后一条 Lane 接着完成串行回答。

DDVR 是 Dynamic Dense Virtual RoPE：同一份物理 K/V，对不同 reader 呈现不同的连续虚拟文档与位置，不为每个 reader 复制整份缓存。

```text
一个外部请求 / 一个 episode / 一个 response owner
  → PRIVATE planner 提示
  → PUBLIC 分析；规划列表暂存为 PENDING
  → 完整 <ol><li>...</li>...</ol> 原子公开
      ├─ N=1：当前 Lane 继续，不 fork，也不再递归规划
      └─ N>1：parent freeze；创建全部 child descriptor
                 → 有 pen 的 admission，其余 FIFO 排队
                 → 各 child 推导；通过 DDVR 读取已提交 PUBLIC KV
                 → 显式 planner phase 可以再次 fork
                 → 各 child 自然输出自己的 </ID> 并退场
  → 无 queued / starting / suspended work，确定唯一 final survivor
  → 在稳定共享视图上完成 exactly-once final acquire fence
  → 从 survivor 继续串行思考
  → 原有 chat 路径自然进入正文或 tool call，结束同一条响应
```

没有 merge agent，不回到冻结的 main，不按答案质量挑 survivor。并行历史产生的是 writer-contextual KV 和 recurrent state；它不等价于把最终文档重新串行 prefill 一遍。

### 2.1 四种身份不能混在一起

| 身份 | 含义 |
|---|---|
| request / person / episode | 一份独立任务状态与外部响应；普通串行请求也占一个 person |
| logical node / Lane | 树上的任务；可以排队、运行、递归 fork、暂停或退休 |
| pen / physical execution slot | 执行 Lane 的有限物理资源，可换绑不同 node |
| `llama_seq_id` | 底层执行、保活或状态引用 handle，不是逻辑任务身份 |

`slot.id`、`node_id`、`exec_seq` 不能因早期实现恰好相等就互相替代。外层 `n_cmpl` 的不同 completion 各有独立 episode；内层 RERoT child 不能另建外部 completion 或单独终止 SSE。

### 2.2 复用现有推理栈

`llama_kv_cells` 是唯一 physical KV metadata owner。RERoT 扩展其 visibility、episode/node/run、frontier 等元数据，不另建 server 私有的 physical-cell registry。Tri compaction、清理、restore 都必须保持这些元数据与 cell 一致。

server 管协议、树、队列、admission、sampler、response 和 frontier；core 管 view、KV 解析、位置、memory 和 graph。不要让 `server-context.cpp` 自行遍历物理 KV，也不要为每条 Lane 建一个 `llama_context`。

### 2.3 当前逻辑状态模型

接班时至少要能区分下面几层状态。字段名以当前源码为准，文档只保留语义，不要求未来 struct 排列永远不变。

```text
server_rerot_episode
  id / root_task_id / response_task_id
  frontier
  publish_epoch / topology_epoch / layout_epoch
  document
  nodes[]
  ready_queue / running / starting / suspended
  archive_seq
  hard limits + token counters
  topology_barrier_pending
  finalizing / hard_aborted
  fence_refreshed / serial_tail / serial_node

server_rerot_node_runtime
  logical node id
  pen_id / physical_slot compatibility alias
  exec_seq / parked_seq compatibility state
  storage_pos_next
  current public/private/pending run
  planner parser / exact-close parser
  enqueue_frontier
  sampler_blob / mtp_blob
  shared fork hand_seed
  final-fence checkpoint
  last installed view_stamp

server_pen
  pen id
  person / episode / node binding
  exec_seq
  free / allocated / running / suspended
```

逻辑 document 在 core 层保存：

```text
llama_rerot_node
  parent / children / depth / title / state / runs

llama_rerot_run
  owner
  visibility
  storage_pos0 / token_count
  publish_epoch

llama_kv_rerot_meta   # 跟 physical cell 一起移动
  episode_id
  node_id
  run_id
  publish_epoch
  frontier
  visibility

llama_rerot_reader_state
  episode / reader / query_run
  frontier
  topology/publish/layout epoch
  frontier mode
  ordered run ids
```

node 状态机当前至少包含：

```text
planning
terminal_running
forked
queued
starting
running
ready_suspended
retired
```

这些状态的关键边界是：**logical node 可以长期存在而没有 physical pen；reader view 只带稳定 run id，不带长期 physical KV index；pen 和 exec_seq 都是可回收 execution handle。**

### 2.4 三个 epoch 各自表示什么

不要把所有“发生变化”都塞进一个 generation counter。

```text
publish_epoch
  PUBLIC 文档/共享内存发生可见提交

topology_epoch
  tree、fork、heading、admission 等改变 reader 的结构关系

layout_epoch
  context shift、restore/重建、resident layout 等改变有效地址/布局
```

MTP、graph/input refresh、state restore 都要明确自己依赖哪几个 epoch。一个 peer 只追加普通 PUBLIC token，通常推进 publish/frontier；fork/heading 可能同时推进 topology；逻辑 history deletion 必须推进 layout。不能为了省事让所有事件全局 invalidation，也不能漏掉真正改变 reader view 的事件。

## 3. 协议：公开文档和私有控制分开

### 3.1 三种 visibility

| 类型 | owner 可读 | peer 可读 | 客户端可见 |
|---|---|---|---|
| PUBLIC | 是 | 同 episode、reader view 包含且满足提交边界时可读 | 已提交内容可见 |
| PRIVATE | 按自身执行 lineage 可读 | 不可直接 lexical 读取 | 不输出 runtime 强制控制字节 |
| PENDING | 是 | 完整记录提交前不可读 | 提交前不可见 |

普通非 RERoT cell 仍走原有 sequence membership。PUBLIC 也不是“同 episode 一律可见”，仍由 reader view、位置和 frontier 判定；不同 episode 的 PUBLIC 绝不能串读。

**PRIVATE 只是词面隔离，不是安全隔离，更不是状态中可逆删除的指令。** 私有提示影响 owner 的 hidden/recurrent，随后生成的 PUBLIC 内容可以带着这种因果影响。模型自己采样出内部标签，与 runtime 强制 PRIVATE 字节直接泄漏，也必须分开查。

RERoT 不解析、注入、删除、替换或强制闭合原生 `<think></think>`，不把它们当调度信号。原有 chat template / reasoning parser 继续负责思考与正文的分离。

### 3.2 Planner 与 worker

planner 提示采用第一人称、简明内省，要求列出工作量大致相当、可独立推进的平面 `<li>` 标题，不提前展开答案，也不告诉模型硬件有几支 pen。2026-09-09 当前实现文本是：

```text
我先把回答中可以同时展开、工作量大致相当的并列对象或章节列成一个平面的 HTML 有序列表：以 <ol> 开头，每项只写一个简短 <li> 标题，不提前展开内容，以 </ol> 结尾。一个 <li> 只对应一个可独立展开的对象或子主题；同一项里仍有互不依赖的部分时，我继续把它们分成并列项。只有确实无法并行拆解时才写一个 <li>。输出完 </ol> 后我再展开分析。
```

这是协议的一部分，不是普通 system prompt 调参项。若源码以后修改，应同步更新本文并重新做 Phase 1 语义验收；不能让文档和 binary 各自维护不同“冻结 prompt”。

字节 parser 必须处理 tokenizer 任意拆分。疑似 `<ol>` 的片段先 hold；真正规划列表从开始到完整 `</ol>` 均为 PENDING，随后原子公开；候选失败则按原字节顺序释放。不得让 sibling 看到半个列表。

N=1 时不新建 child。root 解除 planner grammar 后用 PRIVATE continuation 接着思考；child 从显式 planner 返回 worker 时保留/恢复自身 exact-close 约束。该 node 后续普通列表不再触发规划。

N>1 时 parent 停止运行，为所有 direct `<li>` 建 descriptor。admission 后把标题机械转成 heading：第一级 `<h1>`，随后逐级增加，显示最多到 `<h6>`；真实深度保留在 metadata 中。heading 完整后才原子公开。

当前 child 默认直接做 worker，并有绑定自身标题和 close marker 的 PRIVATE task contract；不是旧文所写的“完全没有 worker control”。递归规划必须显式进入 planner phase；普通 worker 正文中的 `<ol>/<li>`、标题和代码块没有调度含义。实现入口是 `server_rerot_child_contract()`、`server_rerot_child_planner_prompt()`、`server_rerot_child_worker_prompt()` 与 runtime 的 planner 状态转换。

当前 contract 的语义固定为：

```text
我只做自己的唯一子任务 <title>；
sibling PUBLIC 内容只作参考，不接管兄弟章节，也不重答整个用户问题；
普通列表/标题/代码块只是正文，不触发 scheduler；
完成自己的任务后直接输出自身精确 </ID>，随后停止该 child。
```

child 若需要继续递归拆分，runtime 显式进入 PRIVATE planner phase，再使用只针对当前 title 的 planner prompt；不能靠 worker 正文偶然出现 `<ol>` 自动 fork。

### 3.3 随机 ID 是唯一自然退出协议

每个 logical child 创建时分配随机 8 位 base62 ID，字母表为 `0-9A-Za-z`，大小写敏感。PRIVATE opener 为 `<ID>`，close 为 `</ID>`。ID 随 node 保存，不随 slot 改变。

协议随机源独立于模型 sampler。排队、换 pen、RAM 恢复、checkpoint 和 retry 恢复同一 logical child 时不能重生成 ID。按现有契约假设随机 ID 不碰撞，不添加查重集合、上下文扫描或内容 escape；首字符可以是数字，这不是 XML tag-name 协议。

只匹配**当前 owner 的完整精确字节序列**。错误 ID、大小写不符、普通 HTML、固定 `<blockquote>` 都不能触发退出。不假设 delimiter 是一个 token；不完整候选保持 PENDING，失败则原样释放，成功只消费 delimiter，不能吞掉相邻正文。

child 正文须先有非空白内容，随后可任意多行、多段、含代码块。`server_rerot_child_grammar()` 没有“一行完成”规则。EOG 在 close 之前是协议错误，不允许据此由 runtime 偷补 `</ID>`。不设置 per-child/per-depth token budget，不用重复截断或题目专用 stop 冒充完成。

### 3.4 Response 与采样

episode root request 始终拥有外部响应。PUBLIC reasoning 使用现有 reasoning/thinking 字段，不能混入最终 `content`；PRIVATE 和未提交 PENDING 不进客户端 delta。

流式时每个 Lane 缓冲不完整行，按完整行实际提交/完成顺序输出，同 frontier 以稳定 commit 顺序裁决。退场或 fork 可补发最后非空残行。非流式 reasoning 按最终 PAC-DFS 文档渲染；两者顺序不必相同。serial tail 不重发已流出的 reasoning；一条响应恰好一个起始事件和一个终止事件。

每个 live Lane 有独立 sampler、grammar 和 RNG；pen 换绑须正确初始化/恢复 sampler。sibling 内容通过 attention 影响 logits，不机械加入本 Lane 的 repetition penalty history。协议 ID 的随机数不能消耗模型采样随机流。

`reasoning_effort` 已进入本地请求链路，支持 `none/low/medium/high/xhigh/max`，Responses 的 `reasoning.effort` 转到同一路径。它不是 RERoT child 的结束协议。历史审计中的预算比例曾多次修订；普通非 RERoT 的 budget-only fallback 最后记录为 `20%/50%/80%/95%/100%`，受显式预算和最终输出余量约束，不把它解释成第三方模型的原生行为。具体请求语义查 [server README](tools/server/README.md) 和 `test-chat`，不要把这组比例复制成 RERoT 调度规则。

## 4. Attention：PAC-DFS、DDVR 与 frontier

### 4.1 PAC-DFS 给每个 reader 排一份文档

对某节点的 children `c0 ... ck`，若 reader 位于 `ci` 子树，渲染顺序为：

```text
本节点 PUBLIC prefix
→ c(i+1) ... ck
→ c0 ... c(i-1)
→ ci 递归放最后
```

不在 reader 路径上的子树按原始 child 顺序稳定 DFS。这样 parent heading 在 child 前，每个 PUBLIC token 恰好出现一次，reader 自身路径在每层最后，off-path 子树不随内部 writer 活跃情况洗牌。

core 从稳定 run id 解析当前 resident cells，再分配连续、唯一的 virtual positions。Tri eviction 后可以有 storage/physical 空洞，但虚拟文档仍是 dense；不能假定一个逻辑 run 对应连续物理区间。server 不保存长期有效的 GPU cell index。

### 4.2 DDVR 的数学

令一段 K 在 storage position `s+m` 写入，而 reader 将它排到 virtual position `v+m`；query 的虚拟位置是 `q`。在固定、已验证的 RoPE 参数下：

```text
缓存：K_m = R(s+m) K̄_m
目标：<R(q) Q̄, R(v+m) K̄_m>
等价：<R(q+s-v) Q̄, R(s+m) K̄_m>
```

因此 K 不动；针对不同 source span 调整 Q 的相位。正确性基准是显式 rephase K 后的 dense attention，与 Q-side DDVR 对照，而不是两份实现互相抄预期。

**所有可见 span 共用一个 softmax。** 不能分别算 `softmax(QK_A)V_A`、`softmax(QK_B)V_B` 再相加。fused kernel 用跨 span 的 online-softmax `(m,l,O)` 累积；PUBLIC 与 PRIVATE 分段扫描也必须按相同归一化合并。

本项目 Qwen3.5 文本 IMRoPE 输入是 `(p,p,p,0)`。文本位移 `Δ` 必须成为 `(p+Δ,p+Δ,p+Δ,0)`，不能只改第一轴或改第四轴。多模态 spatial positions 不能照此处理。

Turbo 路径还必须保持变换顺序：Q projection/norm → DDVR/IMRoPE → Turbo WHT → K dot；K 保持 writer storage phase 的缓存表示。以实际 tensor type/shape/stride/padding 为准，不能把 storage head dimension 当成逻辑 head dimension。F16 是数值对照，不是生产中静默替换 Turbo 的理由。

实现与基准入口：[llama-rerot.cpp](src/llama-rerot.cpp)、[test-rerot-ddvr.cpp](tests/test-rerot-ddvr.cpp)、[test-rerot-attn.cpp](tests/test-rerot-attn.cpp)、[llama-graph.cpp](src/llama-graph.cpp) 和 CPU/Vulkan attention 实现。

### 4.3 STRONG 不是“同拍互读”

```text
read：自己的 causal history / current K/V + barrier 前已提交的 peer PUBLIC
write：各 Lane 写本 frontier 的 K/V，执行自身 recurrent transition
barrier：该 frontier 完成并提交
next read：下一 frontier 才可读取这些 peer writes
```

当前 `llama_rerot_cell_visible_public_full()` 以同 episode、PUBLIC 且 `meta.frontier < reader.frontier` 为完整 peer 可见条件；自己的 current K/V 走 owner-gated causal 路径。Vulkan/FlashPrefill 不能因先 copy KV 再算 attention 而绕开它。LAG1 再额外延迟一个已提交 frontier，只作显式对照。

sampled/PUBLIC/PENDING 推进按 Lane frontier 组织；连续 PRIVATE control 可用有界 causal microbatch（既有设计上限 32 token），但不得部分提交逻辑事务。物理组批可以改变执行时间，不能把同一个逻辑读写集合换成另一条方程。

### 4.4 Topology 与 graph input

列表公开、heading 公开、fork/admission、大规模 view 变化都可能使旧 next-token logits 失效。refresh 必须读取最新 view，又不重复推进 KV/recurrent、计数或 stream。**单纯 synchronize 不证明完成了 logits refresh。** 一般 topology refresh 的覆盖范围要继续测试，不能用 final-fence 回放测试代替所有 barrier。

view/visibility/phase offsets 尽量做 graph input，shape/capacity/kernel variant 才进入 reuse key。异步执行和 pipeline parallel 下，旧 graph 还在读的 host/GPU input 不能提前被下一 frontier 覆写。

多 reader 共享 K/V tile load 与 Turbo dequant 是可追求的优化，各自仍保留独立 Q、phase、attention output 和 `(m,l,O)`。reader tile 宽度不是每人 Lane 上限；“一次加载服务多 reader”也不能仅凭架构图宣布当前所有路径都已优化完成。

## 5. Recurrent：默认原生递推，研究共享另列

### 5.1 当前默认及状态表示

目标 hybrid 的主干是周期性 `3×GDN + 1×Full Attention`；该 Ornith 配置的 Full Attention 位于 `3,7,...,39`。当前 child 的所有 GDN 层保持本 Lane 的原生 recurrence，conv 始终私有。A3/A7 等层读到的 peer PUBLIC 信息会自然进入后续本地递推，不再直接合并 child 的 recurrent matrix。

固定本 token 的 `k,v,g,beta` 后，原生 GDN 状态更新是仿射算子：

```text
T_i(S) = A_i S + C_i
A_i = exp(g_i) (I - beta_i k_i k_iᵀ)
C_i = beta_i k_i v_iᵀ
```

transition composition 结合但一般不交换，state 本身不是可随 PAC-DFS 重排的文档。没有 token 就没有 transition，不能为 position gap 另加未训练的 decay。

当前 grouped memory 仍可能用 `B + H_i` 表示有效状态：`B` 是存储基底，`H_i` 是 Lane 的 hand overlay。**保留 brain/hand arena 不等于 child 仍在共享更新 brain。** 默认 child 不提交 shared brain，其 hand 必须保存完整本地 transition：

```text
有效输入：S_i = B + H_i
默认 child：H_i' = T_i(B + H_i) - B
有效输出状态：B + H_i' = T_i(B + H_i)
```

root 的 PRIVATE shadow 与 PUBLIC 基底切换只是换坐标，不是重新递推：

```text
H_new = H_old + B_old - B_new
B_old + H_old = B_new + H_new
```

不能通过 batch 里“恰有两组”等 shape 猜 ordinary root 并忽略 hand。persistent hand 已为 **F32**；不能把它当作可随意有损量化的 KV cache。

实现入口：[delta-net-base.cpp](src/models/delta-net-base.cpp) 的 `rerot_shared_rbb_enabled()`、[llama-memory-recurrent.cpp](src/llama-memory-recurrent.cpp)、模型 graph builder 与 [test-rerot-recurrent.cpp](tests/test-rerot-recurrent.cpp)。

### 5.2 为什么 shared RBB 不再默认启用

DDVR 已把 peer 文本写进各 Lane 的 hidden；后续 `q/k/v/g/beta` 是这些 hidden 的非线性投影。再把这些 transition 合进同一个 global brain，会增加第二条反馈通道。仅从当前投影量，不能唯一拆出“本 Lane 新信息”和“刚读过的 peer 信息”。

这不是说共享 recurrent 永远不可研究，而是现有方案缺少可验证的去重/innovation 定义。置换对称、单次 decay、数值稳定都不能代替未重训模型的任务质量证据。因此只有显式 `LLAMA_REROT_RBB_ABLATION=shared-rbb` 才启用共享研究路径；`raw-redundant` 保留重复证据反例。默认不是根据输出好坏动态 fallback。

未来要把 shared RBB 升为默认，须先给出可验证的 innovation operator 或明确的 transition virtualization 契约，再过数学、语义和质量门。foreign-token state-only replay 的全网成本接近随 Lane 数平方增长，不作为当前默认方案。

### 5.3 保留的研究数学与已纠正边界

以下是审计所用的 block-Delta 基础方程，**不是默认 child 的更新式，也不是后续 evidence-normalization 研究实现的完整替代定义**。令 `N` 只计同一 brain、本 frontier 的 PUBLIC writers，`K`/`V` 按 writer 堆叠：

```text
N=0：B' = B
N=1：Bbar = exp(g_1) B
     B' = Bbar + beta_1 k_1 (v_1 - Bbarᵀ k_1)ᵀ
N>1：Bbar = exp(mean(g_i)) B
     C = diag(sqrt(beta_i))
     M = C K Kᵀ C + diag(1-beta_i) + eps C²
     M z = C (V - K Bbar)
     B' = Bbar + Kᵀ C z，eps = 1e-4
```

`N=1` 必须显式走原生分支。直接把带 `eps` 的 block 公式套到单笔，其系数是 `beta/(beta*||k||² + 1-beta + eps*beta)`，并不普遍等于 beta。

`beta` 定义域包括 `[0,1]` 端点。零 write gate 必须严格不写，不能 clamp 到 `1e-6`，但该 PUBLIC token 的 decay 仍计入公开时钟。对称缩放避免除零；合法参数下的正定系统用独立参考验证，非有限求解不得静默换成零。

研究 shared writer 的基本 hand 项是 `T_i(B+H_i)-T_i(B)=A_iH_i`；不提交 brain 的 PRIVATE/PENDING row 则是 `T_i(B+H_i)-B`。二者不能混用。当前 token 的 readout 走自身 native transition，不能先合并本步 peer writes 再立刻读 `B'`；shared commit 只供下一 frontier 使用。

未归一化的重复 writer 有确定的宽度放大。相同单位 key/value/beta 的有效写门为：

```text
beta_effective(N) = N*beta / (1 + (N-1)*beta + eps*beta)
```

`beta=0.3,N=8` 时约为 `0.774`，不是 `0.3`。后续研究路径加入 evidence-density normalization 和零均值 self-echo；这些只修正特定动力学问题，不证明两条共享通道叠加已有效。

CPU 使用独立 double 参考；Vulkan CG 的“N 步精确收敛”不能照搬到 FP32。真实捕获张量曾暴露该错误，后改有界 4N 迭代、每 N 步重算真实残差并重启。通过有限 captured blocks 不是对任意病态输入的误差上界证明。

## 6. 生命周期、内存与恢复

### 6.1 Fork、排队与 archive

parent fork 或 Lane retire 时，PUBLIC KV 需要通过 attention-only archive/keeper 引用保活，再释放 exec 引用；PRIVATE/PENDING 不作为公开 archive。archive 只解决生命周期，不定义 reader visibility。

queued child 只有逻辑 descriptor、标题、ID、parser/sampler seed，以及必要的共享 fork-state seed 引用。**排队不能按 child 数提前分配一整份 GPU KV/recurrent 执行状态。** 同一 fork 的多个 sibling 可以共享不可变 seed，admission 时才 materialize 到 pen；旧 parked recurrent COW 机制仍是实现/对照的一部分，不应按“共享脑默认”旧设计误删当前原生递推所需 lineage。

默认 native child 继承相应 fork 局部状态，同时 admission 读取最新已提交 PUBLIC KV。不得照旧文写成“只接当前共享脑、不需要 child recurrence”。child 数没有 planner-visible 上限；超过物理 pen 数只排队，不截断 `<li>`。队列采用 enqueue frontier / tree-path 的稳定 FIFO，不按标题、置信度或答案好坏选任务。

### 6.2 Frontier 是原子提交单位

执行前核对整组所需 KV、recurrent、metadata、预算与 backend 资源。不能只让 A 成功提交而 B 因资源不足未执行，却继续当作同一完整 frontier。PRIVATE causal microbatch 的物理推进和逻辑 run 扩展也必须一致。

只维护 episode/global 硬资源门，不给 child/depth 分预算，也不为了“保住答案”预留 final token。pen 暂时不足是排队，不是资源失败；不可恢复的硬资源耗尽则 HARD_ABORT 整个受影响 episode：停止 writers、清理 queue/refs/drafts，不选 survivor，不做 final fence，不补答案或伪造自然 `stop`。

对外用现有错误/终止通道明确区分 hard abort、cancel、timeout、length 和自然完成；不依赖一个文档自造的 `finish_reason` 值。每条响应仍须正常收尾且只终止一次。

### 6.3 Final survivor 与 acquire fence

queue、starting、suspended 和尚未 admission 的工作都必须清空。最后一条 Lane 自然输出自身 `</ID>`，才有资格进入 final。多个 Lane 同 frontier 退出时按稳定 tree-path 顺序裁决，不引入 judge。

完整 fence 不只是安装 stable view，步骤是：

```text
提交其他 Lane 最终 PUBLIC writes，确定 survivor
→ 安装稳定 PAC-DFS reader view
→ 恢复首个闭合候选 token 之前的完整局部 checkpoint
  （不回退当前 PUBLIC brain，不覆盖其他 Lane）
→ 移除 survivor 原 PRIVATE suffix KV
→ 用原 token ID / storage position / run ID 逐行 PRIVATE 回放
→ 每行经正常 batch/decode，确认全部成功
→ 才允许 complete_serial_tail、恢复用户 grammar 和串行继续
```

不能重新 tokenize delimiter，也不能直接再走一次已提交的 recurrent transition。不得在 `post_decode()` 递归调用 `llama_decode()` 覆盖其他 slot 尚待采样的 logits。prepared 状态与 cursor 必须幂等；重复 poll/prepare 不能重置回放。

回放不再次扩展 logical run、不重写 prompt tape、不重复计 model/sample/visibility tokens 或发 stream。物理计算开销仍要反映在耗时中。非 survivor 及时释放 checkpoint；final 回放完成后也释放其局部快照。

serial tail 保留 survivor 的最新 conv、local recurrent、hand 和共享 view，不恢复 fork 旧 seed 来“洗掉指令”。未 fork 的 N=1 root 同样保留当前状态。可以继续使用 DDVR attention，不要求立即搬动并重相位全部 K。

实现入口：[server-rerot.h](tools/server/server-rerot.h) 的 fence checkpoint、[server-rerot.cpp](tools/server/server-rerot.cpp) 的 serial transition，以及 [server-context.cpp](tools/server/server-context.cpp) 的正常 decode 回放接线。

### 6.4 Checkpoint / RAM / context shift 的边界

checkpoint 要覆盖同一个完整局部时点：conv、R0-R2/local S、F32 hand、position、source row、rollback selector。restore 应先校验所有尺寸和格式，再取得独占 cell 后整体写入；坏输入不能半写状态或分配一半。共享 sibling 不能被覆盖。

选择旧 snapshot 后接到当前 PUBLIC 基底时，需要 rebasing：`H_seed = B_selected + H_selected - B_public_now`。默认 native child 的 hand snapshot index 不能误用来选择它从未写过的 root 历史 brain 槽。启用 rollback slots、即使没真正 rollback，也不能改变后续递推方程。

SEE3 是 F32 hand seed 格式；episode blob 另有 `LLAMA_REROT_STATE_VERSION`。旧审计记录的 episode `2→3` 是一次历史升级，不应把“3”当成永久当前版本。版本、model/RoPE/Tri fingerprint 和实际 loader 接受范围须一并核对。

完整 RAM save/restore 的对象是 episode，而不是一条 token tape 或一个 server slot。必须保存树、runs、visibility、ID/parser、queues、epochs、memory lineage、sampler RNG、MTP checkpoint、预算与 final-fence 状态；恢复到不同物理 indices 后重建 view。局部 blob roundtrip 通过不代表真实 demotion/resume 已验收。

context shift 是**逻辑历史删除**，Tri eviction 是**物理驻留压缩**。shift 必须同步裁剪 runs/KV/view、更新 epochs、失效相关 drafts；不能用普通 `seq_add()` 随意修 active DDVR 的虚拟地址，也不逆向重写 recurrent 来假装删除过去的语义影响。

## 7. 当前唯一主线：把 Phase 1 长任务做完

### 7.1 先锁配置，再运行

当前目标不是再调 prompt 或追 tok/s，而是让原题完整经历 fork → 自然 close → final fence → serial answer。最小基线为 STRONG、默认 child native recurrence、Turbo4/Turbo2，关闭 TriAttention、MTP、RAM checkpoint/demotion，并隔离 context shift 等干扰。

context 和 episode output/work budget 是两回事，必须分别记录。历史 8192 context / budget 的资源失败不能直接判算法错误；同样也不能据此盲开 131K/262K 窗口。先确认设备、实际 B/P/K 和显存余量，再选择能承载长任务的容量；32K 可作为候选起点，不能当作已证明足够的固定值。容量或预算改变后是新配置，不和旧结果冒充严格 A/B。

**真机测试须谨慎：不能把桌面 GPU 或生产机器拖死。** 不自动重启服务、停止不属于本次测试的进程，或与已有模型服务叠加显存。先检查服务状态、端口、环境变量与完整启动日志。仅仅“不写 `--mtp` / `--triattention`”不能代替核实实际生效配置。

### 7.2 两个现有入口，各有用途

优先用保存完整证据的 [rerot-semantic-smoke.py](scripts/rerot-semantic-smoke.py)。它会**启动本地开发 server 并在结束后清理自己的子进程**，不是只读命令；应在完成上面的资源检查后显式执行：

```bash
# 在仓库根目录；MODEL 指向本地目标 GGUF。
# 输出目录须尚不存在；再次运行时换一个目录名。
: "${MODEL:?请先设置本地目标模型路径}"
python3 scripts/rerot-semantic-smoke.py \
  --model "$MODEL" \
  --context 32768 \
  --total-kv auto \
  --frontier strong \
  --port 18081 \
  --output reports/phase1-semantic-01
```

该脚本的原题请求默认 temperature=0、seed=424242、max_tokens=8192；这些默认值并不保证长任务预算充足。它保存 request、原始 response、metrics、日志、trie、源码 patch 与 binary/library hashes，只检查传输/完成条件，不替国家清单评“质量通过”。`--audit` 会增加同步和读回，不计入性能成绩。RERoT full-auto 下不要再加 `--parallel`；显式 manual KV/parallel 仅作单独标记的开发对照。

已有服务上的 [rerot-continents-benchmark.py](scripts/rerot-continents-benchmark.py) 可采集报告；它**不负责配置或启动服务**：

```bash
python3 scripts/rerot-continents-benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --output reports/phase1-continents.json
```

`--output` 是 JSON **文件**，不是旧交接所写的目录。当前脚本还会加 low-effort system 提示和 `reasoning_effort=low`，并非上面 smoke 的相同请求。报告必须保留实际 payload，不能混报。

**该 continents 评分器尚不能作为 Phase 1 质量裁判。** 它硬性要求八个 `<li>`，把不同地区别名合进第八项，并在全文而非各章节内部找国家关键词；因此不能可靠识别串洲，也可能误判合理拆分。修评分器与修模型应分开，不以改答案、强塞“第八洲”或放宽阈值制造 PASS。本次只整理文档，没有修改这两个脚本。

### 7.3 完整验收条件

生命周期要有日志/trie 证据，而不只看最后一行响应：

```text
root planner
→ 全部 child 建立并按资源 admission
→ 每个 child 自然生成自身 exact random-ID close
→ 全部子任务退休，无剩余 queue / starting / suspended / orphan
→ exactly one final survivor
→ final fence prepare 与原 close-token replay exactly once
→ serial final answer
→ HTTP 200，finish_reason=stop
```

既有八 child 压力用例要证明 8/8 完成；这只是该用例的覆盖目标，不是协议 child 上限或通用地理分类定义。对自然规划结果，要检查任务拆分是否合理、每项是否真正完成，而不是只数 `<li>`。

质量还要逐项检查：章节归属正确、无串洲/接管兄弟任务、无死循环或反复标题、无 prompt echo、无 PRIVATE/random ID 暴露、无内部协议混入公开正文；最终答案完整且确实回答原题。usage、sampled tokens、forced tokens、visibility counters 和实际提交 tape 必须核对，不将 fence 回放重复算成新 token。

失败按层分类保存，不改停止协议遮住问题：资源/context/budget；child 不 close；child 语义偏航；final fence；serial tail；API/stream assembly。timeout 是未完成观察，不等于自然 length/stop，也不能单凭短窗口超时证明模型永不结束。

通过这一完整门后才能宣告“Phase 1 核心真实语义闭环通过”，再转入长轨迹与兼容矩阵。

## 8. 已有证据：哪些修了，哪些仍不能下结论

下表是原文记录的历史证据摘要，**不是本次重跑结果，也不是当前 HEAD 的完整回归报告**。历史 FP16 hand 数值属于当时实现，不能当作当前 F32 hand 的新测量。

### 8.1 局部数学、memory 与 backend

| 发现 | 修复/证据 | 边界 |
|---|---|---|
| mixed batch 随无关 rows 改用 candidate mean；PRIVATE transition 被抵消 | 按 brain group gather/solve/scatter，区分两类 hand；CPU/Vulkan 独立 oracle 覆盖 mixed visibility、两个 brain、倒序 rows | 无 snapshot 单步通过不能推广到全部 multi-token/MTP |
| beta 下限偷偷打开零 write gate | 对称缩放、单笔 native 分支；GPU beta=0/1e-8 的 brain 最大误差由约 `1.02e-3` 降到 `1.86e-9` | 不宣称任意非法输入已有服务层优雅处理 |
| 未初始化 `brain_copy` 影响非 grouped 路径 | 指针/标量初始化；脏存储 placement-new 回归 | 曾出现 CTest 8/8 而直接运行崩溃，故必须保留 direct test |
| checkpoint 半写、COW sibling 覆盖、位置/source/snapshot 错配 | 先加失败反例，再整体校验与写入；包含局部 capture/apply/rebase | 不等于整个 server RAM 恢复通过 |
| final fence 重复 prepare | 原 token tape 与闭合前 checkpoint 经正常 batch 回放，prepared/cursor 幂等；runtime 测 exactly-once | 早期真实请求仍曾 HTTP 500，后修；传输修复不等于答案正确 |
| root PRIVATE/PUBLIC 换基底丢有效状态 | 验证 `4+2 = 1+5`，同 tag 重装不重复换算 | 不是清除私有指令 |
| FP32 CG 只跑 N 步不收敛 | captured 四 writer block 的 brain 误差 `0.00918055 → 4.76837e-7`，output `6.5567e-5 → 7.45058e-9` | 4N 与残差重启仅对已测输入给证据 |
| F16 hand 长轨迹漂移 | 128 步 output/state 误差约 `5.76e-4/1.72e-3 → 3.58e-7/7.15e-7`，改 persistent F32 | 不代表整模型 greedy 恒等 |
| 启用 rollback slots 就丢 child transition | rollback=2 的 128 步 output/state 误差 `2.17231/1.14956 → 2.68e-7/7.15e-7`；另测三快照、suffix discard、resume | MTP 完整矩阵仍须另验 |
| 比较器把空输出/NaN 当零误差；CPU F32 indexed Q 偷降 F16 | 比较器 fail-closed；独立 double QK/softmax/PV；33/257 keys 的 CPU F16-KV 误差降至 `1.19e-7/8.94e-8` | CPU 修复不能冒称 Vulkan 长轨迹根因修复 |
| Vulkan MoE threshold / shape 状态污染 | `7509335d1` 记录 hermetic 18/19 门；后续加入 supported B types 与 unsupported pipeline 回退 | 当前需保留 `18→19→20→18→19` 再认证，不能只看提交标题 |

### 8.2 整模型与真实语义的历史结果

| 输入/配置 | 已观测结果 | 应如何解读 |
|---|---|---|
| 修复 rollback 后的旧 128 teacher tape | Turbo/F16 均 0/128 argmax mismatch；同 KV 下 rollback 0/2 报告一致；最大 logits rel L2 约 `0.07424/0.03486` | 本反例的 rollback 开关不再改递推，仍有形状/精度差 |
| 另一份 512 native Turbo tape | Turbo 1/512 mismatch，首次零起算 step=509；F16 0/512，最大 rel L2 仍到 `0.132615` | 不同 tape 的 step33/509 不能比较成“分叉被推迟”；需同一冻结输入复验 |
| step509 诊断 counterfactual | A3 微差逐层放大；layer29 expert membership 变化；仅诊断替回 native IDs 可恢复 native top token | 定位了能改变决策的位置，不证明 router 本身算错，更不能作生产修复 |
| 默认模型多路固定不同 token | F16/Turbo 短程各路 raw top 与本路参考相同，但 Turbo rel L2 明显更大 | 短程 argmax 相同不等于数值/长期 greedy 等价 |
| 两 Lane 物理行反转 | Turbo 12 步最大 rel L2 约 `6.3e-7`，top token 不变 | 覆盖该排列反例，不覆盖全部 scheduler/restore |
| F16、context8192 的 reader attention audit | observation128/192/256 的 pair cosine mean 约 `0.04088/0.03542/0.03469`，top-key overlap 为 0 | 该观测中 reader attention 确实不同；不是所有运行的证明，不能冒用作 Turbo 解码结果 |
| 早期长题 fence 重入修正后 | HTTP 200 但 `finish_reason=length`，仍串洲、HTML 残缺，completion 8426 与请求预算 8192 不一致 | 明确不通过；不能拿“fence 已工作”覆盖语义与计数失败 |
| 2026-09-08 的 9.11 短题 | HTTP 200/stop，三个 child 自然 close，最终比较正确；公开 child 文本仍有数字漂移和字面内部标签 | 任务完成改善，不是完整语义通过；采样偏航与 PRIVATE transport 泄漏分开 |
| `7509335d1` 候选总结 | 记录了后续 Turbo/F16 fixed-tape、native determinism、response surface 等局部门 | 比上面的失败记录更新，但未给当前合并后版本的全部长轨迹重跑证据 |
| 2026-09-09 交接纠偏 | 锁回 Phase 1 最小配置；同拍物理可见性、提示措辞、sampler 初始化、grouped state 映射已有修复记录 | “长任务尚待完成”仍是当前结论 |

旧 step509 失败是必须保留的回归入口，不能因后续短 tape 通过就删掉；也不能不重跑就断言它在当前 HEAD 必然仍复现。当前应在新 artifact 上冻结 tape、核实对应源码与库，再更新这张表。

可在仓库内查找的历史证据定位：`build-vulkan-localhost/rerot-evidence/20260908-AuxjZd/`，包括 `final-build.log`、`final-ctest.log`、`model512-inherited/teacher.tape`、`final512-{turbo,f16}-rollback*/`。旧 `/tmp/rerot-*` 是当次运行证据，不保证今天仍存在；完整目录名、原始日志清单和逐轮数据可按第 12 节从旧审计取回。

### 8.3 旧速度数据只作历史

2026-09-04 的旧三 slot 配置记录：single-Lane 约 `108.429 tok/s`；两次 request-wide aggregate `237.061/245.028 tok/s`，parallel aggregate `252.476/266.110 tok/s`。当时使用 Tri、Turbo 和旧实现，其部署记录对应 `a0ad22005`。

这些数据说明当时的总吞吐统计已修正，不证明当前版本、默认 recurrence、长任务质量或最终生产配置通过；更没有达到或认证 `>=500 tok/s`。旧记录里的 service active/inactive、代理修复和动态库路径都是当时快照，不是今天的服务状态。

## 9. 完整交付还要补什么

### 9.1 三容量契约保留，但不能沿用旧共享脑成本

```text
B = device-resident people / 独立请求状态容量
P = device-resident pens / Lane 执行状态容量
K = unified physical KV cells

resident_people <= B
sum(person.allocated_pens) == allocated_pens <= P
physical_KV_used <= K
```

一个 person 空闲时可使用全部 P；没有每人三笔或每人固定上限。queued child 不占 physical pen，不按人数/笔数静态切 KV。`n_batch/n_ubatch/reader_tile` 是执行形状，不是第四种逻辑容量。

生产目标继续由 `--total-kv auto` 联合选择 B/P/K，不增加生产 pen 配额参数。显式 `-np` 与 RERoT full-auto 冲突应 fail-fast；RERoT OFF 的 `-np` 语义不变。动态 handle/output/graph 容量须由布局派生，不能让固定 `LLAMA_MAX_SEQ` 偷变成独立产品上限。

**旧文的 `B×27层共享S + P×(R0-R2+conv)` 显存估算已不足以描述默认 native child。** 当前需从实际 grouped brain、完整 F32 hand、local S、conv、snapshot 布局 probe 字节；名字叫 hand 不代表仍只是一个小缓存。

```text
M_total = M_model
        + M_actual_brain_layout(B)
        + M_actual_hand_local_conv_layout(P)
        + M_KV(K)
        + M_graph_DDVR_Tri_MTP_scratch(B,P,K)
        + allocator/headroom/state-staging
```

所有成本由真实 tensor/allocator/probe 获取，并逐 device 检查；不能把多 GPU 显存简单求和，或硬编码某张卡/context 的 token 容量。auto-fit 通过而真实同配置 OOM，仍是容量门失败。

旧三容量规划给出的无在线 benchmark 先验是 pen 候选 `6,5,7,4,3,2,1`，不是算力自适应结论；people 目标为 `min(P,ceil(P/2),B_kv)`，`B_kv≈round(2K/(rho*C))`，`rho` 为 Tri ratio（OFF 为 1），`C` 为实际 context。这些是规划/代码版本级选择策略，须按当前 F32/native-child 成本重新认证，不能把历史“预测 B=3/P=6”写成机器今天的结果。

压力必须分开：人数满则 root 留 admission queue；笔满则 child 排队；KV 满才走 Tri drain/maintenance → refresh → floor exhausted → episode-level demotion/preemption。recurrent-only/brain/hand 压力不调用 Tri。最终公平性还须覆盖 B>P 与后来请求到达；pen yield 只能在 frontier 边界完整保存/恢复局部 state、sampler/MTP 并 refresh，不是杀掉 child。

### 9.2 兼容矩阵：有实现不等于已重新认证

以下均为最终交付要求。当前 Phase 1 为定位故障而关闭某项，不代表最终允许永久缺失。

| 范围 | 必须证明 |
|---|---|
| RERoT OFF | mask、RoPE、KV、recurrent、sampler、batch、Tri/MTP、server 生命周期及既有 backend 零回归，不额外保留无用 GPU scratch |
| FullKV / Turbo / Vulkan DDVR | 四种基础组合 RERoT+FullKV、+Turbo、+Tri、+Tri+Turbo；CPU 独立参考、GQA/IMRoPE/padding/softmax 与精度门 |
| TriAttention | fill-first、首次 drain、3/32、sticky maintenance、floor/fallback、sparse reader、backend-native compaction；archive refs 不重复抬高 target |
| Shared physical union | 外层共享 prompt × n_cmpl × 内层 ancestry/archive/exec；删除一个 ref 不提前 free cell，references_removed 与 physical_freed 分开 |
| MTP / speculative | draft 绑定 person/pen/node、frontier、topology/publish/layout epoch 和 reader view；peer publish/fork/shift 使受影响 draft 失效并正确 rollback |
| Snapshot / RAM / prompt cache | 完整 episode 保存；restore 到不同 physical rows 后与 uninterrupted reference 对照；cache key 不只取最终可读 transcript |
| Context shift / cache reuse | 逻辑 run/view/KV/epoch 同步更新，active DDVR 不误走普通线性 shift；与 Tri eviction 分离 |
| Demotion / preemption / yield | episode victim 不留 orphan；局部 pen yield 精确恢复；recurrent-only pressure 不错误触发 Tri |
| 多请求 / n_cmpl / B>P | 不同 episode 不串 KV、brain、epochs、RNG、final 或响应；排队公平，不按答案内容调度 |
| Streaming / cancel / retry | 无重复 delta，恰好一个 terminal；取消清理 queue/draft/refs，保留其他请求仍用的 shared prefix；旧 generation 回调不污染 retry |
| Tool / JSON / grammar | planner/child grammar 与用户 grammar 分离；final fence 后恢复原 parser、sampler 与工具能力；工具结果回来开启新 episode |
| LoRA / aLoRA | 同 episode adapter set/scale 与 invocation lineage 一致；原 `can_batch_with()` 约束不能绕过 |
| Multimodal | 共同 multimodal prelude 后 fork；只重映射 reasoning text，不重排视觉/音频 spatial positions，不切坏 image chunks |
| Embedding / rerank | 即使全局开 RERoT，也完全绕过 RERoT runtime，保持原路径 |
| Graph / pipeline / backend sampling | input lifetime、reuse key、各 reader 的 logits/sampler row 正确；不为了并行共享而强行 CPU 全量读回 |
| CPU / Vulkan / 其他 backend | CPU 保留 reference，Vulkan 为首要 release gate；其他 backend 明确认证范围，不静默退到语义错误的 stock attention |
| State / metrics | 版本/fingerprint 不匹配明确处理；保留已有 Tri metrics 语义，RERoT 计数与 allocator 一致 |

MTP 要先证明 target 语义正确，再看 acceptance rate；近期“active RERoT 暂停 draft mirroring”的隔离修复不是 MTP 全组合通过。至少覆盖 no-peer-update、peer invalidation、fork、rollback、Tri pressure、final fence 六类。

最小真实 RAM 组合：fork → children running → demote/save → 物理 slots 被其他请求占用 → restore 到不同 indices → 继续；核对 brain/F32 hand/conv、RNG、view、fence 与不中断参考。只看 shape、保存成功或 blob roundtrip 不够。

### 9.3 压力、质量、性能和长稳是独立门

最终压力形状至少组合：6 个外层请求槽位、递归 fork、queued child、Tri pressure、Turbo4/Turbo2、MTP、streaming，以及一次 RAM demotion/restore 或 active preemption。另测 8K common prefix 的多请求 physical union。

要求无 5xx/OOM/deadlock/Vulkan validation error、无 orphan seq/recurrent cell、无重复 SSE，每请求一个终止事件；hard abort 与自然 final 明确区分。auto-fit 的 B/P/K 与真实 allocator 对得上。

质量集包括确定性微题（9.11、整数运算、逻辑）、有 compile/unit tests 的代码题、数学集、长上下文检索/总结/多章节任务，以及真实 prompt 样本。MATH-500、AIME24/25 是原规划中的候选集，不是已完成成绩。阈值在运行前冻结，每题保留输入、seed、配置、answer/reasoning、usage 和 artifact hashes。多 Lane 输出不必逐 token 等于 serial，但不能因并行架构出现质量断崖。

正式性能目标是**标准化多 Lane workload 的 aggregate model throughput >=500 tok/s，且质量不退**。不能靠缩短答案、改题、减预算、删机制或统计 forced token 的方式偷换目标。除 tok/s，还看 quality-qualified goodput、completed responses/s、p50/p95、显存、队列等待、frontier/attention/recurrent/MoE 成本及 MTP acceptance。

最后做长稳：连续请求、fork/close、Tri、MTP、RAM、preemption、shift、shared prefix、cancel/retry、stream 都实际发生。warmup 后 CPU RSS、VRAM、buffer、KV refs、recurrent rows 应进入平台；无 leak、stale tensor、偶发 nondeterminism、死锁或服务重启。

### 9.4 Production Definition of Done：30 个硬门

下面 30 项来自原完整兼容契约，保留为正式 release blocker。前面的阶段可以为了定位问题暂时关闭某个机制，但**最终交付不能靠永久关闭已有能力来通过**。

1. RERoT OFF 全机制零回归。
2. RERoT ON + unified KV 正常。
3. RERoT ON + hybrid recurrent 正常；当前默认 child 为 native lane-local recurrence，研究 RBB 不能混报。
4. RERoT ON + TurboQuant K/V 正常。
5. RERoT ON + Vulkan DDVR/FlashAttention 正常，不能长期依赖语义不同的 fallback。
6. RERoT ON + TriAttention 3/32 正常。
7. backend-native compaction 后 RERoT cell metadata、refs、reader view 正常。
8. RERoT ON + MTP 正常；view/epoch invalidation、rollback 正确。
9. prompt cache / RAM save/restore 保存完整 episode 语义。
10. checkpoint / partial rollback 保存并恢复正确局部状态。
11. context shift 更新逻辑历史、view、KV、epochs，并与 Tri eviction 分离。
12. idle demotion / active preemption 不留 orphan episode/Lane/state。
13. recurrent-only / brain / hand pressure 不错误调用 Tri。
14. outer `n_cmpl` / shared-prefix 与 inner RERoT 正确嵌套。
15. streaming 无重复 reasoning/content delta，且每请求恰好一个 terminal event。
16. cancellation / retry 清理旧 generation，不让旧 callback 污染新请求。
17. LoRA / aLoRA adapter 与 invocation lineage 不串。
18. multimodal prompt 可以在共同 prelude 后 fork，multimodal position 不被 DDVR 错改。
19. tool calling / JSON schema / user grammar 在 final fence 后恢复 stock 能力。
20. embedding / rerank 等非生成请求在全局开启 RERoT 时仍完全绕过。
21. graph reuse / pipeline parallel 不出现 stale input / buffer lifetime bug。
22. auto-fit 计入 RERoT、DDVR、Tri、MTP、recurrent、backend scratch 和 headroom。
23. production-scale full-slot 压力通过。
24. shared physical cell union 压力通过，一个 ref 删除不能提前 free 其他 owner 仍需的 cell。
25. RAM/checkpoint/context-shift/MTP/Tri 组合矩阵通过。
26. 长稳无 CPU/GPU leak、deadlock、stale tensor、state corruption 或偶发 nondeterminism。
27. Vulkan 当前生产配置性能没有不可接受断崖，并达到第 10.10 节固定的性能门。
28. 所有新增 state format 有 version/fingerprint，错误版本明确拒绝或按明确兼容规则处理。
29. 现有 Tri metrics 语义不变，新增 RERoT metrics 与真实 allocator/token accounting 对得上。
30. 最终验收、shadow、canary、production 使用同一源码/二进制/运行库/模型/校准 artifact；hash 和 `/proc/PID/maps` 一致。

只完成树、DDVR、scheduler、fence 和若干数学回归，可以称 **RERoT research/core correctness candidate**；只有以上 30 项全过，才可称 **production-compatible RERoT**。

## 10. 只保留这一套推进顺序

以下编号沿用原 `AGENTS.md` 的**收口路线**，以后项目进度只用这一套。旧指南中的 Stage 0–10 是早期“从零实现组件”的施工顺序；附录 A/B 的 Phase 是兼容/容量子计划；脚本 `phase8` 只是文件名。它们都不能再拿来表示整个项目已经走到哪个阶段。

总主线是：

```text
Phase 0  锁死局部正确性基线
    ↓
Phase 1  真实长任务完整闭环                     ← 当前在这里
    ↓
Phase 2  长轨迹数值确定性
    ↓
Phase 3  Tri / Turbo / unified KV 重新认证
    ↓
Phase 4  MTP / speculative 组合
    ↓
Phase 5  RAM / checkpoint / rollback / cache / shift
    ↓
Phase 6  Server 外层语义矩阵
    ↓
Phase 7  生产级资源压力 + B/P/K auto-fit
    ↓
Phase 8  质量总验收
    ↓
Phase 9  性能优化与正式性能门
    ↓
Phase 10 长稳 soak
    ↓
Phase 11 RC artifact sealing
    ↓
Phase 12 shadow → canary → production
```

后一个阶段可以提前写代码或做诊断，但**不能把后面某个局部结果当成前一阶段已经通过**。例如 MTP 代码可以已经存在，Phase 4 仍然可以是“未认证”；质量脚本可以叫 phase8，项目仍然处于 Phase 1。

### 10.1 Phase 0：把局部正确性封成 Golden Baseline

**目的**：以后所有功能改动、merge 和性能优化都有一组不能随便破坏的红线。Phase 0 不是开发新能力，而是把已修过的确定错误变成稳定 regression。

必须固定的门至少有：

| 门 | 目标 |
|---|---|
| Vulkan MoE threshold | 覆盖 `18 → 19 → 20 → 18 → 19`，不同前序 shape 不能污染后续 routing/pipeline state |
| native-vs-native | 同一输入、同一配置、多次运行 sampled decision mismatch = 0 |
| fixed teacher tape | 128/512/更长；Turbo/F16；rollback 0/>0；同一 tape 可重复定位 first divergence |
| mixed child recurrence | 加 PRIVATE/PENDING、另一 episode、换 row order、换 ubatch packing 不能偷换递推方程 |
| DDVR / attention | independent double/materialized reference；global softmax；IMRoPE；NaN/空输出 fail-closed |
| random-ID close | 只认 exact owner delimiter；跨 tokenizer token；错误 ID 不结束；restore 后 ID 不变 |
| final fence | checkpoint、原 token replay、幂等 prepare、exactly-once 计数/stream |
| response surface | OpenAI/Responses/Anthropic reasoning/content/finish 不混，forced PRIVATE bytes 不外泄 |
| state format | SEE3/F32 hand 和 episode state version 失败方式明确，坏输入不半写 |

**当前状态**：历史上这些项目有相当强的局部证据，`7509335d1` 曾因此被称为 Core Correctness Candidate；但当前 HEAD 已经过多轮 merge 和 backend 改动，所以 release 时仍要在当前 artifact 上重跑。旧“9/9 PASS”只说明当时那个测试集合。

**通过标准**：这些门都成为可重复自动化；任何后续 patch 让其变红，先判 patch/regression 错，不先调 tolerance、换 prompt、减预算或改 seed。

### 10.2 Phase 1：真实长任务闭环

**目的**：证明 RERoT 不只是局部算子和短 smoke 正确，而是能让一个真实、多 child、长生命周期请求完整自然结束。当前唯一主线就是第 7 节“大洲国家”任务。

基线先隔离干扰项：

```text
RERoT ON
STRONG barrier-after
default child native recurrence
Turbo4/Turbo2
Tri OFF
MTP OFF
RAM demotion/checkpoint OFF
context shift OFF / 隔离
temperature 0
fixed seed
```

容量必须能容纳任务；context、KV capacity、episode token/work budget 分别记录，不能把一个耗尽冒充另一个。历史 8192 容量失败只能说明当时资源/预算不足，不自动证明算法错；同样也不能把盲目加到 131K/262K 当成修复。

完整生命周期必须观测到：

```text
root planner
→ 完整规划记录原子公开
→ 所有 child descriptor 建立
→ 能跑的 admission，其余 FIFO
→ 每个 child 自然生成自己的 exact random-ID close
→ 所有 child retired，无 queue/starting/suspended/orphan
→ exactly one final survivor
→ stable reader view
→ final fence prepare
→ original close-token replay exactly once
→ serial tail
→ final answer
→ HTTP 200
→ finish_reason=stop
```

质量同时要求：任务拆分合理；各章节归属正确；无 child 接管 sibling；无循环标题、prompt echo、随机 ID / forced PRIVATE /内部控制协议泄漏；最终答案完整。`usage`、sampled/forced token、PUBLIC/PRIVATE/PENDING counters 和 replay 不重复计数。

失败分六类，不通过改 delimiter 或 stop 规则遮住：

```text
A. context / KV / episode budget / backend resource
B. child 不自然 close
C. child semantic drift / sibling takeover
D. final fence / stable view / replay
E. serial tail / reasoning-to-answer transition
F. API / streaming / response assembly
```

**当前状态**：未通过。只有这一门通过后，才能第一次说“核心真实语义闭环通过”。

### 10.3 Phase 2：长轨迹数值确定性

**目的**：把“128 步或某个短例子没问题”升级成真正能支撑生产排障的长轨迹保证。它解决的问题是：以后自然语言答案坏了，能先排除“backend/batch/rollback 偷偷改数学”。

冻结一份或一组 **native teacher tape**，至少覆盖：

```text
短 prompt
MoE 18/19/20 threshold prompt
9.11 数值题
中文长 prompt
代码 prompt
长到能覆盖历史 step509 一类晚分叉的输入
```

每份 tape 对下列组合运行：

```text
Turbo4/Turbo2
F16/F16 numerical control
rollback slots = 0
rollback slots > 0
不同 ubatch composition
不同 physical row ordering
不同前序 shape / graph reuse history
必要时不同 reader count / person grouping
```

指标分三层，不能混成一个 tolerance：

1. **native vs native determinism**：sampled/argmax mismatch 必须为 0。这是确定性，不接受“误差很小”。
2. **RERoT single-lane vs native**：在语义应等价的单 Lane 路径，sampled-decision/argmax mismatch 必须为 0。
3. **连续数值误差**：再报告 logits relative L2、逐层 activation error、KV/code 差异、router membership 等。误差允许非零，但不能无限漂移，也不能跨过 sampled-decision boundary。

如果在 step `t` 第一次 top token 分叉：

```text
固定 teacher tape 到 t
→ 复现无插桩分叉
→ 打开逐层 trace
→ 找第一处可测差异
→ 对 attention / recurrent / MoE / quantization 分层二分
→ 必要时做只读 counterfactual
```

counterfactual 只能定位因果点，不能把 native expert ID、native logits 等硬塞回生产路径当“修复”。

**已有历史**：128 tape、512 tape、step509、F16/Turbo、rollback 0/2 都已有重要证据和反例，见第 8 节。**当前任务是重新认证，不是从头发明测试。**

**通过标准**：长 tape 上 deterministic/single-lane decision 门稳定，误差曲线和 first-divergence 调试路径可重复；当前 artifact 的结果和源码/库 hashes 锁定。

### 10.4 Phase 3：TriAttention + Turbo + unified KV 重新认证

**目的**：证明 RERoT 核心在真实生产 KV 栈上仍正确。这里叫“重新认证”而不是“实现”，因为大部分代码历史上已经存在，但近期 DDVR、MoE、F32 hand、FlashPrefill、XKV-SR、response 等都发生过变化。

先跑四个基础组合：

```text
RERoT + FullKV
RERoT + Turbo
RERoT + Tri
RERoT + Tri + Turbo
```

每个组合至少核：

- reader visible cell 集与顺序正确；
- DDVR virtual position dense、unique；
- Turbo4/Turbo2 不静默退 F16；
- IMRoPE/GQA/padding/global softmax 正确；
- public/private/pending visibility 不因 compaction/packing 漂移；
- archive/exec/shared-prefix refs 的 physical union 正确。

Tri 不是“能 reclaim 就过”，必须完整保持原契约：

```text
fill-first
→ first pressure drain
→ 3/32 target
→ sticky maintenance
→ floor exhausted
→ atomic demotion/preemption
```

Tri reclaim 后，DDVR view 从当前 resident cells 重新构造；逻辑文档和物理 residency 分开。archive ref 只是 keeper，不得让 scorer 误以为相同 semantic token 有多份价值。

压力 fallback 必须严格区分：

```text
KV pressure:
  Tri drain/maintenance
  → refresh physical usage
  → floor exhausted
  → episode-level demotion / preemption

recurrent / brain / hand pressure:
  禁止调用 Tri
  → recurrent/pen/person 对应处理
```

shared-prefix 另做至少一组：共同 8K prefix → 多 completion / episode → inner RERoT fork，逐 ref 删除并核 `references_removed` 与 `physical_freed`。

**通过标准**：四个基础组合、Tri sparse/compaction、physical union、fallback 顺序都自动化；FullKV 和 Tri 的差异只来自允许的 lossy residency，不来自 metadata/visibility bug。

### 10.5 Phase 4：MTP / speculative decoding

**目的**：让 speculative decoding 在共享内存会变化的 RERoT 中有明确一致性语义，而不是简单永久关闭。

每个 draft/checkpoint 至少绑定：

```text
person / episode
pen / node
frontier
topology_epoch
publish_epoch
layout_epoch
reader view stamp
```

只要 peer publish、fork、heading publish、context shift、Tri/compaction 导致该 reader 的有效 memory view 改变：

```text
old draft stale
→ reject remaining draft
→ restore target checkpoint
→ 清掉 uncommitted draft KV/recurrent state
→ 从最新 stable view 重新 draft
```

不得让 draft 穿过尚未提交的 frontier，也不能因联合 batch 的物理执行顺序看到同拍 peer write。

至少六类实测：

```text
RERoT + MTP, no peer update
RERoT + MTP, peer update invalidates draft
RERoT + MTP, fork/topology barrier
RERoT + MTP, rollback
RERoT + MTP, Tri pressure/compaction
RERoT + MTP, final fence
```

**正确性门优先于 acceptance rate**：相同 target 配置下，MTP ON 的最终 target 结果必须与 MTP OFF 一致。acceptance 低只是性能问题；结果变了是 correctness fail。

近期“active RERoT 暂停 draft mirroring”只是隔离修复，不能冒充这一阶段完成。

### 10.6 Phase 5：RAM / checkpoint / rollback / prompt cache / context shift

**目的**：证明 RERoT 不只会“一口气生成完”，而是完整状态可以保存、恢复、回滚和迁移。

episode state 远不止 token tape，至少包含：

```text
tree topology / node states / child order
document runs
PUBLIC / PRIVATE / PENDING metadata
random child IDs + parser partial state
ready/running/starting/suspended queues
reader views / topology,publish,layout epochs
KV refs / archive ownership
native child recurrent state
root/private basis state
F32 hand / conv / local S
sampler RNG / grammar state
MTP checkpoint + view stamp
budget/resource counters
final-fence checkpoint/prepared/cursor
```

最低真实 RAM 门：

```text
fork
→ 多 children running
→ demote / save
→ 原 physical rows/slots 被其他请求复用
→ restore 到不同 physical indices
→ rebuild view
→ 继续生成
→ 与 uninterrupted reference 对照
```

只验证“blob 能 roundtrip”不够。checkpoint / partial rollback 必须覆盖 snapshot plane、source row、position、F32 hand、conv/local S、root/private/public basis，以及 suffix discard 后继续推进。

prompt cache 的 key 不能只看最终可读 transcript；同样文字可能对应不同 writer-contextual KV、tree、epochs 和 recurrent lineage。

context shift 定义为：

```text
logical semantic history deletion
```

不是 Tri eviction。shift 后必须同步更新 run coordinates、KV refs/positions、PAC-DFS view、layout/publish epoch，并使相关 MTP draft stale。active DDVR run 不能用普通线性 `seq_add()` 假装改 virtual position。

**通过标准**：保存/恢复/rollback/shift 后的 sampled decisions、state、view 与 reference 一致；版本/fingerprint 不匹配明确失败；不发生半写、orphan 或旧 physical index 依赖。

### 10.7 Phase 6：Server 外层语义矩阵

**目的**：证明 scheduler、HTTP/SSE、outer completion 和已有 server 功能不会改变 RERoT 的数学与生命周期。

正式矩阵至少包括：

| 能力 | 必须证明 |
|---|---|
| multi-person | 多独立 episode 的 KV、recurrent、epochs、RNG、response、final 互不串 |
| `B > P` | root/person admission 与 child pen queue 分开；aging/fairness，无长期饥饿 |
| `n_cmpl > 1` | 外层 completion 各自一 episode，共享 prefix 但 fork 后 visibility domain 分离 |
| shared-prefix | 共同 prefix refs × outer completion × inner ancestry/archive/exec 的 union 正确 |
| idle demotion | 整个 episode 可保存/恢复 |
| active preemption | victim 后精确继续或明确终止，不留下 sibling/orphan |
| pen yield | 只能 frontier boundary，完整保存 local state、sampler/MTP，再从最新 PUBLIC view refresh |
| cancellation | 停止所有 writers、queue、draft，清理本 episode refs，不删其他请求仍用的 prefix |
| retry | 新 generation/id，旧异步 callback 不能提交到新请求 |
| streaming | PUBLIC reasoning 无重复；PRIVATE/PENDING 不漏；恰好一个 terminal event |
| tool calling | final fence 后恢复 stock tool grammar/parser；工具结果回来开新 episode |
| JSON/GBNF | planner/child grammar 不污染用户 grammar，serial tail 恢复原 sampler |
| LoRA/aLoRA | adapter set/scale 与 invocation lineage 一致，不能绕过 `can_batch_with()` |
| multimodal | shared multimodal prelude 后 fork；视觉/音频 position 不进 text DDVR 重排 |
| embedding/rerank | 即使全局开 RERoT 也不创建 episode/runtime，完全走 stock path |
| graph reuse | capacity/shape 进 reuse key，具体 ids/view 数据走 input；无 stale host/GPU buffer |
| pipeline parallel | span/frontier input 生命周期与同步正确 |
| backend sampling | 每 Lane 独立 sampler row，不为方便强制全部 logits 回 CPU |

总不变量：

> scheduler 只能改变“什么时候运行”，不能改变“算什么”。

换 pen、排队、抢占、batch packing、不同外层请求组合，都不能让同一个 logical computation 静默切换 attention mask、recurrent update、RNG lineage 或 grammar。

**通过标准**：上述矩阵有自动化和至少一组真实 Ornith/Vulkan E2E；每请求 response ownership 清楚，无 5xx/orphan/重复终止。

### 10.8 Phase 7：生产级压力与 B/P/K auto-fit

**目的**：在所有主要机制同时活跃时证明资源模型和 fallback 真能成立，不只是小规模 smoke。

核心容量只有三种：

```text
B = resident people / independent request-state capacity
P = resident pens / Lane execution-state capacity
K = unified physical KV cells
```

运行不变量：

```text
resident_people <= B
allocated_pens <= P
sum(person.allocated_pens) == allocated_pens
physical_KV_used <= K
```

没有 per-person pen cap。一个 person 在没有竞争时可拿全部 P；child > P 只排队，不丢 descriptor。人数满和笔满都不触发 Tri，只有 KV pressure 触发 Tri。

终极组合压力至少包含：

```text
RERoT
+ 6 outer requests/slots
+ recursive forks
+ queued children
+ Tri pressure
+ Turbo4/Turbo2
+ MTP
+ streaming
+ 至少一次 RAM demotion/restore 或 active preemption
```

验收必须同时满足：

```text
0 5xx
0 OOM
0 deadlock
0 Vulkan validation error
0 orphan seq ref
0 orphan recurrent/hand/brain cell
0 duplicate SSE
每请求 exactly one terminal event
hard abort / natural final 可区分
Tri fallback 顺序正确
MTP rollback 正确
```

另做 shared-prefix union 压力，例如 8K common prefix、多 completion、多 inner child；逐步取消/retire/demote，保证 physical cell 不提前释放。

auto-fit 不是只算 KV：

```text
model tensors
B/person brain/state
P/pen native local recurrent + F32 hand + conv + sampler/output rows
K physical KV
DDVR span/query metadata
Tri score/pack scratch
MTP target/draft scratch
Vulkan/graph temporary buffers
state-save staging
allocator granularity + headroom
```

`--total-kv auto` 的目标是联合求 B/P/K；真实 bytes 由 allocation probe 和 tensor layout 获取，不写死某张显卡、某模型 context 或“每 pen 几 MiB”。如果 auto-fit 判定可装，真实同配置仍 OOM，就是 Phase 7 fail。

旧 pen sweet-band `6,5,7,4,3,2,1` 只是无在线 benchmark 的历史先验；改变 GPU/backend 后须通过容量/性能矩阵重新认证，不能把它说成硬件自适应最优解。

### 10.9 Phase 8：质量总验收

**目的**：证明并行 execution semantics 没有带来任务质量断崖。不能只靠 9.11、一个长题或几个关键词 smoke。

冻结多层质量集：

```text
确定性微题：
  9.11 vs 9.9
  整数运算
  简单逻辑
  短事实

代码：
  Python
  C++
  算法题
  小型 repo QA
  至少 compile/syntax + unit tests + 结果正确

数学：
  MATH-500
  AIME24/25
  或同级冻结集

长上下文：
  needle/retrieval
  长文总结
  大洲国家
  多章节问答

生产：
  真实 prompt 样本
```

每题都保存：prompt、seed、sampling/config、RERoT/Tri/MTP/Turbo 参数、answer、reasoning、usage、metrics、artifact hashes。不要只报一个平均分。

RERoT 多 Lane 不要求逐 token 等于 serial，因为并行共享本来就是新 execution semantics；但必须与冻结 baseline 比较，不能出现系统性章节错位、事实/数学退化、代码通过率暴跌或长任务无法闭合。

**阈值在跑 benchmark 之前写死。** 看见分数后再放宽阈值、换评分器、换 prompt 或减少任务难度，不算验收。

`rerot-phase8-quality.py` 只是工具入口；文件名不代表这一阶段已经完成。

### 10.10 Phase 9：性能阶段

**前置条件**：Phase 1–8 的 correctness/compatibility/quality 门全部绿。此前可以做小的明显低风险性能修复，但不能把正式性能目标放在错误语义上优化。

标准 production 配置固定后再测，例如：

```text
Ornith 1.5 35B
Vulkan
Turbo4/Turbo2
Tri 3/32
MTP ON
RERoT STRONG
production B/P/K auto-fit
```

至少记录：

```text
serial tok/s
RERoT request-wide aggregate tok/s
parallel model tok/s
completed responses/s
quality-qualified goodput
prefill tok/s
p50 / p95 latency
VRAM peak
queue wait / pen utilization
frontier barrier cost
DDVR attention cost / HBM bytes
recurrent cost
MoE cost
MTP acceptance
graph rebuild / metadata upload / small dispatch cost
```

正式性能门：

> **标准化多-Lane workload 下 aggregate model throughput >= 500 tok/s，并且质量门不退。**

不能通过少生成、改 prompt、降低 reasoning、取消 Tri/MTP、缩 context、减少任务数、把 forced control work 从统计里消失等办法达到。

优化顺序按风险/收益：

```text
第一层：
  不必要 synchronize
  graph rebuild
  metadata upload
  small dispatch
  重复 gather/scatter

第二层：
  multi-reader DDVR shared physical KV scan
  K/V tile load once, serve multiple Q readers
  Turbo dequant reuse
  frontier batch packing / weight batching

第三层：
  MTP acceptance / draft scheduling
  MoE route reuse
  kernel fusion
```

每个 perf patch 都必须：

```text
fixed-tape numerical gate
→ RERoT focused CTest/direct tests
→ semantic microset
→ 再看 benchmark
```

MoE route/pipeline cache 优化尤其必须过 18/19/20 threshold 红线。不能积 20 个优化后才发现某处开始改语义。

### 10.11 Phase 10：长稳 Soak

**目的**：证明达到质量和性能目标后，服务器在长时间真实状态切换中仍稳定。

soak 中要持续发生，而不是只“开着不动”：

```text
full-slot / full-person 连续请求
fork / close / recursive fork
queued child admission
Tri drain / maintenance / floor events
MTP accept / invalidate / rollback
RAM save / restore
preemption / pen yield
context shift
shared prefix
cancel / retry
streaming
tool call round-trip
```

持续监控：

```text
CPU RSS
VRAM used
backend allocated buffers
KV physical cells / refs
brain / hand / local recurrent used rows
people / pens resident/allocated/suspended
queue depth
MTP draft state
orphan counters
Vulkan errors
```

warmup 后内存和资源计数必须进入平台，不能随请求数单调上涨。episode 完成/取消后相应 refs、pens、state 必须归零或回到稳定基线。

**通过标准**：无 leak、stale tensor、死锁、偶发 nondeterminism、state corruption、服务重启或长期 starvation。

### 10.12 Phase 11：Release Candidate Artifact Sealing

**目的**：消除“验收的是 A，部署的是 B”的工程假通过。

每个 RC 至少封存：

```text
Git HEAD
git status / source.patch
compiler
CMake cache / build config
llama-server --version

SHA-256:
  llama-server
  libllama-server-impl
  libllama
  libllama-common
  libggml
  libggml-base
  libggml-vulkan
  libggml-cpu
  libmtmd

model SHA / fingerprint
Tri calibration SHA / fingerprint
systemd unit / service config
完整启动参数
关键环境变量
```

运行时用 `/proc/PID/maps` 记录真正 mmap 的 `.so` 路径和 hash，避免链接到了 `/opt/...` 的旧副本。release gates 必须跑在**同一 artifact**上。

只要 rebuild、换 library、改校准、改模型或 dirty patch：

> 旧验收对新 artifact 失效，相关门必须重跑。

build success 不是 deploy permission；Git HEAD 相同也不代表 dirty worktree、runtime library 和 model artifact 相同。

### 10.13 Phase 12：Shadow → Canary → Production

**目的**：最终发布仍分层，不直接覆盖生产。

Shadow：

```text
同 production model/config/artifact
不接真实用户流量
跑完整 release smoke 和组合检查
```

Canary：

```text
少量真实请求
重点观察：
  5xx
  OOM
  finish / terminal event
  p50/p95 latency
  VRAM / RSS
  RERoT people/pens/queue/fence metrics
  Tri drain/floor/fallback metrics
  MTP acceptance/invalidation
```

Production 放量后重新核 `/proc/PID/maps` 和 artifact hashes，再跑：

```text
health
9.11 / deterministic microset
代码题
长题
stream
tool call
并发/资源压力
```

保留上一 RC 的完整 rollback artifact、service config 和 hashes。rollback 不应依赖重新编译或临时寻找旧库。

### 10.14 阶段状态总表

| 阶段 | 当前状态 |
|---|---|
| 0 Golden Baseline | 有历史强证据；当前 HEAD/artifact 仍需完整复核 |
| **1 真实长任务闭环** | **当前主线，未通过** |
| 2 长轨迹数值 | 有 128/512/step509 历史证据与反例；待当前 artifact 重认证 |
| 3 Tri/Turbo/unified KV | 实现历史丰富；待四组合和压力重新认证 |
| 4 MTP | 有隔离/状态接线；完整 epoch-aware 组合未认证 |
| 5 RAM/checkpoint/shift/cache | 局部 checkpoint/rollback 修复已有；真实 episode 恢复矩阵未收齐 |
| 6 Server 外层矩阵 | 若干机制已有代码；完整 multi-person/B>P/n_cmpl/stream/tool 等矩阵未收齐 |
| 7 资源/auto-fit | B/P/K 设计与工具存在；production full-pressure 未验收 |
| 8 质量 | 工具存在；阶段未通过 |
| 9 性能 | 有旧 237–266 tok/s 历史数据；未达到/认证最终 >=500 门 |
| 10 Soak | 未完成正式长稳 |
| 11 Artifact sealing | 有历史证据采集方法；未形成当前 RC |
| 12 Release | 未获发布许可 |

阶段推进纪律只有两条：第一，**前一阶段没过，不准用后一阶段的局部成绩替它盖章**；第二，出现 regression 先缩到固定 tape / 固定 state / 固定 artifact，而不是回去调 prompt 或 stop 条件。

## 11. 测试、工具和统计口径

### 11.1 源码导航

| 范围 | 入口 |
|---|---|
| 树/view/DDVR/visibility | [src/llama-rerot.h](src/llama-rerot.h)、[src/llama-rerot.cpp](src/llama-rerot.cpp) |
| KV metadata / compaction / reclaim | [src/llama-kv-cells.h](src/llama-kv-cells.h)、[src/llama-kv-cache.cpp](src/llama-kv-cache.cpp) |
| recurrent / snapshot / hand seed | [src/llama-memory-recurrent.cpp](src/llama-memory-recurrent.cpp)、[src/models/delta-net-base.cpp](src/models/delta-net-base.cpp) |
| context / graph / state | [src/llama-context.cpp](src/llama-context.cpp)、[src/llama-graph.cpp](src/llama-graph.cpp) |
| parser / queue / protocol / fence | [tools/server/server-rerot.h](tools/server/server-rerot.h)、[tools/server/server-rerot.cpp](tools/server/server-rerot.cpp) |
| server 接线、sampler、pressure、response | [tools/server/server-context.cpp](tools/server/server-context.cpp) |
| 容量与参数 | [common/fit.cpp](common/fit.cpp)、[common/common.cpp](common/common.cpp)、[common/arg.cpp](common/arg.cpp) |

### 11.2 分层验证，不把诊断程序当质量门

无模型测试覆盖 parser、PAC-DFS、DDVR、recurrent、runtime、KV cells 与 Tri score。以下是原审计使用的 focused 构建/运行方式，**不是本次已经执行的结果**；其中 direct attention/recurrent 可能使用 Vulkan，仍须遵守真机安全约束：

```bash
cmake --build build-vulkan-localhost --target \
  llama-server test-rerot-parser test-rerot-view test-rerot-ddvr \
  test-rerot-recurrent test-rerot-attn test-rerot-runtime \
  test-kv-cells test-triattention-score -j6
ctest --test-dir build-vulkan-localhost \
  -R 'rerot|kv-cells|triattention-score' --output-on-failure
build-vulkan-localhost/bin/test-rerot-recurrent
build-vulkan-localhost/bin/test-rerot-attn
git diff --check
```

CTest 的 8/8、9/9 是历史测试集快照，不是永久应有项数。显式设备门在没有 GPU 时不能 skip-success；必须保留直接运行的 device、数值和 exit code。比较器应拒绝空输出、长度不同和 NaN/Inf。

整模型诊断入口是 [test-rerot-model-single.cpp](tests/test-rerot-model-single.cpp)、[test-rerot-model-batch.cpp](tests/test-rerot-model-batch.cpp)、[test-rerot-model-permute.cpp](tests/test-rerot-model-permute.cpp)。需要本地模型，运行前读各自 CLI；不能把 `model-batch` 调用成功/有限值的 exit 0 当作所有误差已过门。strict fixed-tape 检查与显式 `--report-only` 必须分开。

数值分三层：同一配置的 native-vs-native determinism 要求 mismatch=0；RERoT single-lane vs native 检查 sampled-decision/argmax mismatch=0；再报告 logits rel L2 与 activation error，不能只因误差“小”就忽略跨过采样决策边界。

| 脚本 | 用途和限制 |
|---|---|
| [rerot-semantic-smoke.py](scripts/rerot-semantic-smoke.py) | 真实模型生命周期与完整证据；不自动评答案正确性 |
| [rerot-continents-benchmark.py](scripts/rerot-continents-benchmark.py) | 已有服务采集/离线粗查；八项与全文关键词评分局限见第 7 节 |
| [rerot-audit-replay.py](scripts/rerot-audit-replay.py) | 同一捕获 RBB 输入独立重放，避免比较已分叉自回归轨迹 |
| [rerot-attention-audit.py](scripts/rerot-attention-audit.py) | reader attention 分布审计；按实际支持的 tensor/KV 格式解释，不冒解 Turbo 为 F16 |
| [rerot-throughput-gate.py](scripts/rerot-throughput-gate.py) | 同 prompt/seed/采样配置下统计吞吐与计数门，不代替质量验收 |
| [rerot-capacity-matrix.py](scripts/rerot-capacity-matrix.py) | people/pens/容量分配形状测试入口；当前配置与覆盖范围须逐次记录 |
| [rerot-phase8-quality.py](scripts/rerot-phase8-quality.py) | 质量集工具；名称中的 Phase 8 不代表当前项目阶段 |

### 11.3 Token 与吞吐不能混账

```text
request-wide aggregate =
  Δrerot_completed_model_tokens / Δrerot_completed_episode_seconds

parallel aggregate =
  Δrerot_parallel_model_tokens / Δrerot_parallel_seconds

single-Lane baseline =
  timings.predicted_n / timings.predicted_ms * 1000
```

同一已完成 episode 的 model tokens 要与 PUBLIC + PRIVATE + PENDING committed counters 核对；failed/cancelled episode 不混入 completed 的分子分母。visibility 计数与 sampled/forced 是不同维度，不把所有 PENDING 简单视为“非采样 token”。API usage 与 forced runtime work 要分别报告；不能把 sampled-only `predicted_n` 称为 RERoT 总吞吐。

计数全局共享时，要隔离测试流量或精确归属请求，避免别人的 metrics 增量混进报告。审计读回、fixed-tape、强制 counterfactual、研究 ablation 与正式速度测试分开。

观测至少覆盖 episode 完成/abort、nodes/queue、people/pens、frontiers、topology refresh、visibility tokens、final fences、MTP invalidations、context shifts、brain/hand/scratch bytes、KV refs 和原有 Tri metrics。person/node id 放关联日志，不做高基数 metric labels；默认日志不输出完整 PRIVATE 推理。

### 11.4 必须长期保留的属性测试矩阵

这组不是一次性调试清单，而是设计不变量的 executable form。具体 case 可以增加，下面的语义不能删掉。

**PAC-DFS / document**

```text
单层 2/3/4 children
3 层以上 nested tree
reader 位于不同 leaf
off-path writer 追加内容时旧 subtree 顺序不洗牌
每个可见 PUBLIC token exactly once
virtual positions 连续且无重复
retired / queued / suspended node 不破坏文档顺序
```

**DDVR / attention**

```text
equal / unequal span lengths
storage position overlap
private gaps
sparse physical indices
compaction before / after
Qwen3.5 text IMRoPE (p,p,p,0)
GQA
head-dim padding / stride
F16 / Turbo4+Turbo2
CPU independent reference vs Vulkan
one global softmax across all visible spans
STRONG barrier-after / LAG1 timing
```

**recurrent / state**

```text
default native child N=1 等价原生
rollback slots 0 / >0 不改变未 rollback 的方程
三 snapshot → discard suffix → resume
PRIVATE / PENDING 不误提交 shared research brain
root PRIVATE/PUBLIC basis switch 保持 B+H
不同 episode / mixed visibility / row permutation 不串
F32 hand capture/apply / SEE3 invalid input fail-closed
shared-rbb 仅显式 ablation，不能在默认 path 偷启用
```

**visibility / publication**

用固定 secret 构造：

```text
PUBLIC:  R7K2
PRIVATE: P9M4
PENDING: X3Q8
```

foreign reader 应在正确 frontier 后读到 PUBLIC；不能 lexical 读取 PRIVATE；PENDING 在完整提交前完全不可见，提交后作为完整 record 可见。错误 episode id 永远不可见。

**queue / scheduler**

```text
2 pens，parent 产生 8 children
8 descriptor 全创建
只有资源允许的 child admission
其余保持 FIFO
后 admission child 看得到等待期间新提交的 PUBLIC memory
child > P 不丢任务
B > P 有 aging / fairness
换 pen / row order / batch packing 不改变 logical result
```

**close / final fence**

```text
delimiter 跨 tokenizer token
错误 ID / sibling ID / 大小写不符不能 close
正文可多行、多段、代码块、普通 HTML
restore 后 ID 和 parser partial state 不变
queue 非空不能 final
simultaneous exits deterministic
倒数第二 Lane 最后写 SECRET42
final survivor 的 replay 必须在 stable view 中看到 SECRET42
fence prepare/replay/metrics/stream exactly once
```

**server / transport**

```text
OpenAI Chat Completions
Responses API
Anthropic-compatible surface（若当前 server 支持）
stream / non-stream
cancel / retry
n_cmpl
shared-prefix
tool / JSON grammar
hard abort
```

对所有 surface，PRIVATE/random ID 不出现在公开响应；reasoning/content 不串；自然 stop、length、cancel、hard abort 不互相伪装；每个 response 恰好一个终止事件。

**资源 / compatibility**

```text
FullKV / Turbo / Tri / Tri+Turbo
MTP no-update / invalidation / rollback / final fence
RAM restore to different physical rows
context shift
preemption / pen yield
shared physical union
auto-fit selected B/P/K vs allocator peak
```

如果某个功能只能靠删除这些 case 才“通过”，应视为设计 regression，而不是测试太严格。

## 12. 发布、证据与文档维护

一个通过结果必须能回答：**哪份源码、哪个模型、什么配置、实际加载哪组库、跑了什么输入、输出和退出状态是什么？**

每轮保存 Git HEAD/status/source.patch、compiler/CMake、server version、完整启动参数、请求/seed/预算、response/reasoning/usage、metrics、日志/trie；记录 launcher、server impl、libllama、libggml/base/Vulkan/CPU、common、mtmd，以及模型和 Tri calibration 的 SHA/fingerprint。运行前后 hashes 一致，并通过 `/proc/PID/maps` 核实实际加载的 `.so`，不能只 hash 构建目录里的同名文件。

rebuild 或换库后，旧 artifact 验收失效，相关门必须在新 artifact 上重跑。build 通过不是部署许可；旧 build number 或同一个 Git HEAD 也不代表 dirty worktree 与加载库完全一致。

最终先 shadow，再少量 canary，确认 5xx/OOM/finish/latency/VRAM/RERoT/Tri/MTP 指标后放量；同一 artifact 完成 health、短题、代码、长题、stream、tool、并发验证，保留上一完整 RC 回滚包。本文没有宣称今天的服务 active 或 inactive。

### 12.1 原始记录去哪里找

本次合并前的四份原文都保存在 Git 提交 `47432e525`。需要完整旧公式推演、逐轮实验表、临时证据目录、原始部署 hashes 或已淘汰设计时，直接查看历史，而不再维护第二份“当前状态”：

```bash
git show '47432e525:AGENTS.md'
git show '47432e525:RERoT指南.md'
git show '47432e525:RERoT数学与语义审计.md'
git show '47432e525:下班交接.md'
```

关键定位：`7509335d1` 是历史 core candidate；`32e36c3a3` 记录物理 barrier 与第一人称提示修复；`3355b7c23` 是 active RERoT 的 draft-mirroring 隔离；`0c7fef45c` 移除 Phase 1 评测中的 Tri/MTP 干扰；`535ac8cfa` 明确把进度退回 Phase 1；`0374d3eb0` / `47432e525` 是当前 HEAD 附近的 Vulkan MoE 约束/回退。提交存在只能证明改动记录存在，测试通过仍要看对应 artifact 的结果。

### 12.2 以后怎么更新

新增证据时先改第 1 节的状态，再补第 8 节的输入、配置、artifact、结果和边界；阶段真正通过后才改第 10 节。默认算法或协议改变时直接改对应正文，不在文末再追加一条与正文相反的“最新规范”。

未验证的实现写“已实现、待验收”；历史通过写清版本；失败保留最小复现；研究开关不混进默认配置。`AGENTS.md` 只留安全要求和本文入口，不再存路线图；不再新增每日交接作为竞争事实源。

## 13. 设计红线：这些做法不要再引入

这部分保留原指南中最有价值的“反模式”清单。发现实现准备走到其中任何一条，应先证明为什么旧的不变量已经不适用，而不是静默改变架构。

- 不给每个 Lane 预留固定 KV 长度，也不按 person/pen 静态切统一 KV 池。
- 不让两个可见 token 占同一个 reader virtual position。
- 不为 DDVR 每 frontier 搬整份 K cache，也不为每个 reader 复制完整 KV。
- 不做 per-span 独立 softmax 再相加；一个 reader/head 必须是一个全局 softmax。
- 不用普通 `seq_has(reader)` 代替 RERoT PUBLIC visibility。
- 不在 server 维护第二份长期 physical-cell-index registry；physical metadata 只能跟 `llama_kv_cells` 或稳定 logical id 走。
- 不把 `server_slot.id == node_id == exec_seq == pen_id` 当语义，只能作为过渡实现偶然相等。
- 不把 queued child 当完整运行 Lane 提前复制 GPU KV / full recurrent state。
- 不设置隐藏 per-person pen cap；单 person 空闲时可以使用全部全局 P。
- 不把 `n_batch`、`n_ubatch`、reader tile 或 shader specialization 当逻辑 Lane 上限。
- 不因 child 太多静默截断 `<li>`；资源不足要排队或明确 HARD_ABORT。
- 不给 child/depth 增加“为了防循环”的私有 token budget 来冒充自然 completion。
- 不用 EOG、换行、重复次数、timeout 或题目专用 stop 替代 exact owner `</ID>`。
- 不把普通 worker 正文里的 `<ol>` 当隐式 fork；拓扑变化只能来自显式 planner phase。
- 不解析、补写或劫持模型原生 `<think></think>` 作为 RERoT scheduler 协议。
- 不把 PRIVATE 宣传成安全隔离，也不假设恢复 fork seed 能逆掉私有指令影响。
- 不恢复旧 snapshot 覆盖当前 conv/local recurrent/hand，只为“清理”控制 prompt。
- 不让同一 frontier 的 peer PUBLIC 在联合 forward 内同拍互读；STRONG 是 barrier-after。
- 不因 batch shape、是否有 PRIVATE row、row order 或 rollback slots 改变 recurrent 算法。
- 不把 beta=0 clamp 成小正数；关闭的 write gate 必须真关闭。
- 不以 NaN/空输出/长度不符的比较器结果当零误差通过。
- 不把 shared RBB 研究模式作为默认动态 fallback；算法切换必须显式、可复现、可审计。
- 不为 foreign token 默认做 O(N²) 全网 replay 作为“正确性修复”。
- 不把 Tri eviction 和 context shift 混成一个动作：前者是物理 residency 压缩，后者是逻辑 history deletion。
- 不在 recurrent-only / brain / hand pressure 下调用 Tri。
- 不让 MTP draft 穿越已变化的 reader view / publish/topology/layout epoch。
- 不只保存一个 server slot 就宣称 RAM restore 支持 RERoT；保存单位必须覆盖完整 episode 语义。
- 不只 preempt 一个 Lane 后让 sibling 继续假装 episode 完整，除非显式实现并验收 subtree/pen yield 语义。
- 不把 sibling PUBLIC token 机械塞进本 Lane repetition penalty history；共享影响应主要通过模型 memory/logits。
- 不为了 RERoT 强行 batch 原本 adapter/backend 不兼容的请求。
- 不把多模态 spatial position 当 text DDVR position 重排。
- 不让 embedding/rerank 因全局 RERoT flag 进入 generation runtime。
- 不在 graph reuse 时让 span/view/frontier input 指向已被下一轮覆写的 buffer。
- 不以 capability fallback 的“能跑”代替生产 backend 的正确 DDVR/Turbo 实现。
- 不在 CPU 数学/reference 还没过时先靠 Vulkan shader 调参猜正确性。
- 不再加入 merge/judge agent 选择“最好答案”；final survivor 由生命周期决定，不由质量打分决定。
- 不让模型看到硬件 slot/Lane 数、tree path 等调度角色提示，除非明确做研究 ablation。
- 不把 HTTP 200、CTest summary、keyword score、throughput 单独当成“RERoT 通过”。
- 不在 performance regression 时先改 prompt、seed、预算、质量阈值或统计口径。
- 不在 rebuild 后沿用旧 artifact 的验收结果。

最后压成一句话：

> **同一个 logical computation 不能因为 backend、batch packing、slot/pen 分配、压力处理或恢复路径不同而静默换方程；同一个外部请求也不能因为 RERoT 的内部并行而失去原 server 的生命周期与兼容语义。**
