# K1d 机制说明：PLE 路由门 + GDN 门控小算子链融合

日期：2026-09-26。工单：K1d（AGENTS §5.3 "剩余需要新融合模式 + 浮点次序资格"）。
证据基础：K1b `segments.tsv.gz`（97 phase 全量 dispatch dump，it=-1 行即实际 dispatch）。

## 候选链清单（真实 verify 图逐节点确认）

### 链 A：PLE 路由门（`build_ple`，每 PLE 层 1 组，8 dispatch，全部 ne≤2560）

```
idx13 MUL      node_200 (2560)   key*query 逐元素
idx14 SUM_ROWS node_201 (1)
idx15 SCALE    node_202 (1)      ×1/sqrt(n_embd)
idx16 SGN      node_203 (1)
idx17 ABS      node_204 (1)
idx18 CLAMP    (view)  (1)        clamp(|s|,1e-6,∞)
idx19 SQRT     node_206 (1)
idx20 MUL      node_207 (1)      sgn*mag
idx21 SIGMOID  ple_gate (1)
```
后接 idx22 MUL(ple_gated_value, 2560)（gate 广播乘 value，**不在本链**——2560
与 1 不同形，binary 广播语义单独保留）。

### 链 B：GDN alpha/beta 门控（`build_layer_attn_linear`，每 GDN 层 2 组）

alpha 侧（ne=10）：
```
ADD      node_351 (10)   alpha+dt_bias
SOFTPLUS a_softplus(10)
MUL      gate     (10)   softplus × ssm_a
```
beta 侧（ne=1）：
```
SIGMOID  beta_sigmoid (1)
```
两组无数据依赖，间隔 3 个节点（中间是 gate RESHAPE 与 beta RESHAPE）。

### 占比核实（修正上一轮"44 dispatch/层"的粗估）

逐 stage it=-1 行统计后，**tiny dispatch（ne≤320，除 view/reshape 外）实况**：

- verify7 stage（437 行）：27 个 tiny dispatch/层段；PLE 路由门 8 + GDN 门控 4 = **12/27**
- draft/FFN stage（181 行）：19 个；其中 MoE 归一化 CLAMP/DIV/SUM_ROWS 各 1，hc 侧 SCALE+SIGMOID+SCALE 3 个
- PLE+attn stage（239 行）：19 个 tiny；同上 12 个属本工单
- 97 phase 全 dump：SUM_ROWS 49、SIGMOID 217、SCALE 290、SOFTPLUS 36、SGN/ABS/SQRT 各 49

97 stage 里 PLE/GDN 段共 ~84 stage（46+34+几类边界），链 A+链 B 每段 12
dispatch → 全链 ~1000 tiny dispatch/请求 → 一次 CB 里的 launch 约省
一半（链 A 8→1，链 B 4→2）。

## 为什么此前没做：两个结构性障碍及本方案的规避

1. `ggml_can_fuse` 要求同形邻接链——SUM_ROWS 是归约，SCALE/SGN/ABS/
   CLAMP/SQRT 是 1 元素数组，MUL(node_207) 是标量×标量。**逐个 1 元素
   dispatch 才是浪费本身**：每次 ~5 µs 前端 launch + ~2 µs kernel。
2. GDN 门控链中间夹 RESHAPE（ne=1 的 4D view）且两链间隔 3 节点。

规避：不尝试通用 elementwise-subgraph 融合，而是仿 SSM_CONV_SILU / K1c
先例写**专用结构匹配**：模式 + 边检查 + 形状守卫 + 复用手头算子公式。

## 浮点资格（§5.3 要求单列）

- 链 A 在融合 kernel 内保持**完全相同的公式序列**：树形和（与
  sum_rows.comp 同 BLOCK_SIZE/同树）、×scale、sgn、abs、clamp、sqrt、
  mul、`1/(1+exp(-x))`（与 unary.comp op_sigmoid 同式）→ 逐位一致（同
  GPU 上 fused vs unfused 对拍为 oracle，K1c 先例）。
- 链 B 同理：`(x>20)?x:log(1+exp(x))` 与 `1/(1+exp(-x))` 逐字复制
  unary.comp 公式；ADD/MUL 是基本运算。
- **不搅动归约顺序**：SUM_ROWS 的 2560 列树和保持原 BLOCK_SIZE 分组，
  只是结果在同 kernel 内直接消费，不再落地显存。

## 实现设计

### 新 shader `ple_router_gate.comp`

- 输入：A=key*query 预乘结果（idx13 的 MUL 保留原 dispatch，因其
  2560 元素规模是真实工作）→ **不**：再核。MUL(2560)+SUM_ROWS 可并入：
  kernel 一个 WG 处理一行：先乘后树和，公式同原始两 op（乘是逐元素、和
  是树形，两 op 原本就分离，融合不改任何顺序）。
- 输出：gate 标量（写 idx21 SIGMOID 的 dst）。
- push constant：n_cols、行数、eps/scale、misfit 偏移——沿用
  vk_op_sum_rows_push_constants 布局加 param1/param2。

### 新 shader `gdn_gate.comp`

- 同一 dispatch 处理 alpha 链（ADD+SOFTPLUS+MUL，ne=10 逐元素）与
  beta SIGMOID（ne=1）：行号 < n_alpha 走 alpha 分支，否则 beta 分支。
  公式同 unary.comp 逐字。

### 守卫

- 链 A：8 节点模式 `{MUL, SUM_ROWS, SCALE, SGN, ABS, CLAMP, SQRT, MUL,
  SIGMOID}` span=8；边检查（每节点 src 指向前一节点）；src0/src1 同形
  F32 连续；CLAMP 参数 min=1e-6；SUM_ROWS dst ne=1；SCALE 参数有限。
- 链 B：span=3 alpha（`{ADD, UNARY(SOFTPLUS), MUL}`）+ 独立匹配 beta
  SIGMOID（跨度内作为第 4 输出，或单独 span=0 独立守卫——首选独立匹配，
  减少节点序依赖）。同形 F32、ssm_a/ssm_dt 是外部常量。
- 退选：`GGML_VK_DISABLE_PLE_GATE`（动态 getenv）、
  `GGML_VK_DISABLE_GDN_GATE`。诊断：`GGML_VK_LOG_PLE_GATE`。
- 写掩码：融合只写链尾 dst（SIGMOID/gate 的最终 dst），中间 dst 全部
  elide（单用途守卫 `ggml_node_has_n_uses(...,1)`）。

### 执行器接入点

- 匹配分支插入 fusion chain（l2_pair 之后、attention_segment 之前）。
- dispatch case：`GGML_OP_SUM_ROWS`（链 A，取 idx14）与
  `GGML_OP_ADD`/`GGML_OP_UNARY` 不可行——改在 `GGML_OP_MUL`(idx13) 处
  拦截整链，或按 K1c 方式在链首 op 的 case 里先查 `ctx->fused_*` 标志。
- num_additional_fused_ops = 8（链 A）/ 2（链 B alpha）。

## 预期收益上限与诚实边界

- 12 tiny dispatch/段 → 3（链 A 1 + 链 B 2），每 PLE/GDN 段省 9 个
  launch ≈ 45 µs 前端 + kernel 固有延迟；84 段/请求 ≈ **~4 ms/请求上限**，
  与 K1c 同级（不可在 ±15 ms 噪声内单测分辨，ABBA 判非劣/正收益）。
- 若 ABBA 仍非劣：保留价值=dispatch/caused 维护账同 K1c；不宣传提速。
- 风险：SUM_ROWS 融合改变 warp 调度导致树和数值不同？——不会：乘与和
  的浮点序列逐位保留（乘法原在独立 kernel，无跨线程重组）；BLOCK_SIZE
  与 grid 结构照搬。
- 链 B beta SIGMOID ne=1 单 dispatch 与 alpha 分开更稳妥；若匹配复杂度
  超预算，先只做链 A（最大单段，8→1）。

## 验证计划

1. `test-vulkan-command-replay` 新增 `test_ple_router_gate_fusion` +
   `test_gdn_gate_fusion`：构造 8 节点/3 节点真实形状图，fused vs
  Disabled（env 退选）逐字节对拍；全 `--gdn-only` 回归（GDN 状态输出
   位级不变是既有保护线）。
2. 真实 5 卡 smoke：12/12 精确 171/171 自然停止 +
   `GGML_VK_LOG_PLE_GATE` 命中计数。
3. 5-block ABBA（A=env 全退选, B=默认），同 K1c 模板；预注册非劣保
   留、CI 分离为正。

## 决定

先做链 A（最大、形状最简单、公式全同）；链 B 视链 A 的验证成本决定。
不做 MoE CLAMP/DIV 与 hc 侧 SCALE-SIGMOID-SCALE（0.75 元素 dispatch，
转发 READ 语义改写风险大于收益）。
