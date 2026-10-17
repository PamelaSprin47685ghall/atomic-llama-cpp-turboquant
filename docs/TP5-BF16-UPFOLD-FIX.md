# TP5 BF16 up_fold 变体两处索引级缺陷修复（工单 A 数值验证前置）

更新：2026-10-12。状态：**修复已写入工作树，尚未编译、尚未运行**。编译、SPIRV 重生成与全部对拍由 DevOps 后续执行；本文只登记缺陷机理、修复内容、退化论证与验证方案。缺陷发现来自 bf16-shader-audit 只读审计（见其工作记录「发现 1 / 发现 2」），本修复是该审计的后续工单。

## 0. GAP1 最终轮（down-wire-fixer）：down 输出全零的根因排查——四嫌疑域静态排除

更新：2026-10-12（sitedump 之后）。sitedump 对拍（`identity/quarantine-sitedump/parity-analysis.json` + `arms/sitedump-fused/` 9 个 bin）给出决定性直接证据：fused 臂冷路径三个 site 一致呈现 normalized 全非零（n=10240）+ lo 全零（n=320）+ mixed 全零（n=2560）。受托判读为「down dispatch 没有把结果写进 lo（D binding 错接 / grid 为 0 / dispatch 未提交 / 写入被覆盖）」。本轮对这四个嫌疑域做了逐环静态排查，**全部排除**：

| 嫌疑域 | 排除证据 |
|---|---|
| D binding 错接 | down/up dispatch 共用 `ggml_vk_segment_project` 的唯一 descriptor 模板 `{weights, b_sub, out, extra-or-out, out}`（ggml-vulkan.cpp 16756-16758）；binding 2 恒为 output subbuffer，模板无权重类型分支（函数体内仅有的两处 `weights->type ==` 都是 Q5_K 门）。3c 的 Q8 twin 在同一二进制上正常，而组装路径家族无关——bf16 无独立输出组装可错接。hc_quant_consumer 链亦排除：sitedump fused 臂 env 无 `GGML_VK_HC_DOT`（manifest.json），q8dot/inject_norm_q8 均未创建，`hc_quant_producer/consumer` 均为 false，norm 用 8-binding `pipeline_hc_inject_norm`（normalized dump 正确佐证），down 的 B binding = normalized F32。 |
| grid 产出 0/错维 | down 非 fold：`dispatch_rows = n_rows = 1`，`grid = {m=320, 1, 1}`；与 Q8 版逐参数同构（6671-6673 vs 6677-6679：同 5 binding、同 `vk_mat_vec_push_constants`、同 wg_denoms {1,1,1}、同 spec {subgroup_size,1,1}）。`ggml_vk_dispatch_pipeline` 的 `wg = CEIL_DIV(elements, wg_denoms)` 无零产出路径。 |
| dispatch 未提交 | `ggml_vk_segment_project` 在 BF16 路径无 early return（唯一 return 在 Q5_K output-tail 分支内，BF16 权重永不触发）；`ggml_vk_dispatch_pipeline` 无条件 pushConstants/bindPipeline/bindDescriptorSets/dispatch（仅 GGML_ASSERT，失败即 abort 而非静默）。dump 2/3 的 vkCmdCopyBuffer 与 down/up dispatch 记录在同一 command buffer 且 memcpy 落地（返回了数据）——CB 提交链成立。shader 侧写入 guard `tp5_keep_local_output` 非 TP5_RELAY_OUTPUT 恒 true（mul_mat_vec_iface.glsl 225-235），data_d store 无法被 fusion flag 跳过。 |
| 写入被覆盖 | lo dump 点紧邻 down barrier 之后（17081→17088），中间仅有 recording-gated 的 `hc_down_end` tape 注解（非录制路径不执行）；无任何清零 lo 的操作。 |

**排查产出的两个新事实（比排除本身更重要）**：

1. **「mixed 全零是 fold(up(silu(0)))=0 的下游后果」这一判读在 fold epilogue 的实际公式下不成立。** `reduce_result` 的 HC_UP_FOLD 分支是 `product = data_fuse0[...] × sigmoid(temp[0][n])`；lo=0 ⟹ temp=0 ⟹ sigmoid(0)=**0.5** ⟹ `mixed = 0.125 × Σ_n normalized[n*W+col]`，normalized 非零则 mixed 必然非零。**全零的 mixed 要求 up fold 自身的 A（W_up）读零、fuse0（binding 3 = normalized）读零、或 D 写未落地——「down 没写出」单独解释不了 mixed。任何合格根因必须同时覆盖两个收缩（或 dump 读回本身）。** 该数学事实已钉入 `tests/test-tp5-upfold-indexing.py::DownWiringAuditTests::test_fold_of_zero_contraction_is_not_zero`。
2. **dump 路径无 rank 区分**：TP5 五 rank 同进程，全部写同一 `/tmp/tp5-hc-dump-site%u-%s.bin`（最后写者获胜）。sitedump 的「全零」判读描述的是最后一个写文件的 rank，不是五 rank 共识——若 BF16 HC 权重仅在部分 rank 的 buffer 上有效（TP5 tensor-split 的权重分发/上传链），覆盖写会把正确 rank 的非零 lo 覆盖为零。此限制已钉入 `DownWiringAuditTests::test_dump_paths_have_no_rank_disambiguation`。

**剩余假设（静态不可判，交 DevOps 一跑裁定）**：四个嫌疑域排除后，唯一未被静态排除的机理是**运行时数据侧**——W_down/W_up BF16 权重在 dispatch 时刻不在 A binding 指向的 subbuffer 里（TP5 `--split-mode tensor` 下 BF16 HC 权重的镜像/上传链），A 读零页。最小判别实验（一次 DevOps 运行）：(a) dump 前把 A binding 的 subbuffer 也 dump 出来（W_down 头 64 floats 非零与否直接裁定）；或 (b) 单 rank 配置（`-dev Vulkan0` 单卡）跑同一 dump——若单卡上 lo 非零，即覆盖写 + 权重分片镜像缺陷实锤。本轮**无新源码修复落盘**（静态未定位到缺陷行，最小修复原则下不动代码；与第五轮 unit-semantics-audit 的处理一致），交付为上述排除表 + 数学事实 + 测试钉子。

关联文档：`docs/TP5-BF16-LATEBIND-VERIFICATION.md`（工单 A 的 A4/A5 对拍方案，其 §4.2 单 site 中间量对拍被本缺陷阻塞，本修复解除该阻塞）。

## 1. 缺陷 1：`iter_aligned_nonquant` 缺少 HC_UP_FOLD 权重行折叠

### 1.1 机理

HC up 收缩（`ggml_vk_hc_segment` → `ggml_vk_segment_project`，up 调用传 `output=mixed`、`extra=normalized`）的 shader 侧几何（`mul_mat_vec.comp` + `mul_mat_vec_base.glsl`）：

- `gl_WorkGroupID.x = j`（输出宽度坐标，`j < stride_d = m = W`）；`gl_WorkGroupID.y = t`（token 行，`t < T = mixed->ne[1]`）。
- `mat_vec_first_row()`（base.glsl 216-218 行）：`first_row = t * stride_d + j`——**输出行索引被编码为 token 行 × 宽度 + 列**。
- W_up 权重行主序 `[4W, k]`（4 条 HC stream × 宽度，`k = lo_rank = p.ncols`）。`reduce_result` 的 HC_UP_FOLD 分支（base.glsl 318-334 行）把 `temp[0][n]`（n=0..NUM_ROWS-1=3）按 `sigmoid → × data_fuse0[t*4W + n*W + j] → 累加 → ×0.25` 折叠——即 4 条 stream 的 gate 折叠。
- 因此第 n 条 stream 的 gate 所需权重行是 **`j + n*stride_d`**（W_up 行空间的第 n 个 W 段的第 j 行）。

`iter()`（非对齐路径，mul_mat_vec.comp 319-322 行等三处）自始就有正确的折叠：

```glsl
#ifdef HC_UP_FOLD
    const uint weight_row = (first_row % p.stride_d) + n * p.stride_d;
    const uint ib = (weight_row * p.ncols + col)/QUANT_K;
    ibi += p.ncols * p.stride_d;
#else
    ...线性 ibi 索引...
#endif
```

而 `iter_aligned_nonquant`（374-407 行）的 n 循环只有线性索引 `ib = (ibi + col)/QUANT_K`、`ibi += p.ncols`——即权重行 `first_row + n = t*stride_d + j + n`。

**缺陷后果**（确定性索引级，非精度级）：

- 对每个 (t, j)：错位行与正确行之差为 `t*stride_d + n - n*stride_d`。t=0 且 n≥1 时已经错位（差 `n*(1-stride_d)` ≠ 0）；t≥1 时全部 4 行错位。审计记录表述为「token_row≥1 即错位」，实际从 t=0 的 n≥1 就开始——审计表述偏保守，乱码从第一个 token 就出现。
- `col + n`（输出坐标 j 加上 stream 号 n）被当作行内偏移叠加在 `t*stride_d` 上；当 `j + n ≥ stride_d` 时（j 靠近宽度上界或 stride_d 不被 4 整除的几何）行索引越过 `4*stride_d - 1`，读出 W_up 张量之外。
- 4 条 stream 的 gate 值全错 → `reduce_result` 的 sigmoid/fuse0/0.25 折叠输出全错 → mixed 每层每 token 全错 → **全文乱码**。这正是 off 臂（LateBind 关闭）乱码的头号根因：`hc_bf16_weights` 选择逻辑（ggml-vulkan.cpp 16812-16822 行）只看 down 权重类型、不看任何 env 开关，原 Q2_0 的 BF16 HC 权重走 fused HC segment 时 up_pipeline 无条件选中 BF16 变体。

**为什么 Q8_0 原生版从未踩中**：`qwen4_hc_project_fold` 的宏集含 `DATA_A_Q8_0`，`K_PER_ITER=8`（mul_mat_vec.comp 145-149 行），`is_aligned_nonquant`（477-480 行）要求 `K_PER_ITER == 4` 恒不满足——Q8_0 版永远走 `iter()` 的 staged 路径（那里有折叠）。BF16 版 `DATA_A_BF16` → `K_PER_ITER=4`，且 HC up 实际参数（ncols=320 等，均 4 的倍数）满足对齐条件 → 走入从未为 HC_UP_FOLD 设计的通用对齐路径。`qwen4_hc_project_fold_bf16` 是唯一同时满足「函数被编译」（外层 `#if defined(DATA_A_F32)||F16||BF16`）与「宏 HC_UP_FOLD 被定义」的变体（vulkan-shaders-gen.cpp 1132-1146 行）。

### 1.2 修复

在 `iter_aligned_nonquant` 通用循环（非 F32 特化分支）的 n 循环中加入与 `iter()` 完全相同的折叠分支（逐 token 同构：`weight_row = (first_row % p.stride_d) + n * p.stride_d`、`ib = (weight_row*p.ncols + col)/QUANT_K`、`ibi += p.ncols * p.stride_d`），`#else` 分支保持原线性索引。修改处附注释说明缺陷与机理。

对齐性质保持：`is_aligned_nonquant` 已保证 `ncols % 4 == 0`、`col % 4 == 0`（col = i*BLOCK_SIZE + K_PER_ITER*tid，K_PER_ITER=4 且 BLOCK_SIZE%4==0），折叠后 `ib = weight_row*ncols + col`（QUANT_K=1）仍为 4 的倍数——`dequantize4_2aligned` 的 packed32 读（`data_a_packed32[(a_offset+ib)/2]`，a_offset=0）所需的偶数索引前提不变。

## 2. HC_UP_FOLD=0 退化论证（零行为变化）

修复采用编译期宏分支 `#ifdef HC_UP_FOLD`，逐变体核查（vulkan-shaders-gen.cpp 全部 HC_UP_FOLD 注册：1114、1134、1186-1187 行）：

| 变体 | 宏集 | `iter_aligned_nonquant` 是否编译 | 是否可能执行到折叠分支 | 行为变化 |
|---|---|---|---|---|
| 全部普通 mul_mat_vec 变体（F32/F16/BF16/Q 各型，不含 HC 宏） | 无 HC_UP_FOLD | 编译（F32/F16/BF16 型） | 否——`#else` 分支编译，代码与修复前逐 token 相同 | **零** |
| `qwen4_hc_project_fold` | DATA_A_Q8_0 + HC_UP_FOLD | **不编译**（外层 `#if defined(DATA_A_F32)||F16||BF16` 为假） | 否 | **零** |
| `qwen4_hc_project_fold_r320` | DATA_A_Q8_0 + HC_UP_FOLD + HC_UP_R320 | 不编译（同上）；且 main() 提前路由到 `hc_up_r320_compute` → `iter()`（自带折叠） | 否 | **零** |
| `qwen4_hc_down_silu_bf16` | DATA_A_BF16 + HC_DOWN_SILU（无 HC_UP_FOLD） | 编译 | 否——走 `#else` 线性分支，与现状相同（down 的正确语义本就是线性行，见审计「发现 3」） | **零** |
| `qwen4_hc_project_fold_bf16` | DATA_A_BF16 + HC_UP_FOLD | 编译 | **是——折叠分支生效** | 修复目标 |

结论：折叠公式仅在「宏 HC_UP_FOLD 已定义」且「`iter_aligned_nonquant` 被编译」的交集上生效，该交集当前只有一个变体 `qwen4_hc_project_fold_bf16`。宏未定义时 `#else` 分支与修复前代码逐字节等价（线性 `ib`/`ibi` 递增完全未动）——**HC_UP_FOLD=0 时退化为现状**。

## 3. 缺陷 2：`dispatch_rows` 缺少 `pipeline_hc_project_fold_bf16` 分支

### 3.1 机理与同构核对

`ggml_vk_segment_project`（ggml-vulkan.cpp，dispatch_rows 三元）只列 `pipeline_hc_project_fold` 与 `pipeline_hc_project_fold_r320`，BF16 变体落入 `n_rows`——grid-y 缩小 4 倍。

两变体创建参数完全同构（ggml-vulkan.cpp 6669-6671 vs 6675-6677）：

| 项 | `pipeline_hc_project_fold` (Q8_0) | `pipeline_hc_project_fold_bf16` |
|---|---|---|
| shader 源 | mul_mat_vec.comp（HC_UP_FOLD=1） | mul_mat_vec.comp（HC_UP_FOLD=1） |
| bindings | 5，`vk_mat_vec_push_constants` | 5，`vk_mat_vec_push_constants` |
| wg_denoms | {1,1,1} | {1,1,1} |
| spec constants | {subgroup_size, 4, 1}（BLOCK_SIZE, **NUM_ROWS=4**, NUM_COLS=1） | {subgroup_size, 4, 1} |
| 唯一差异 | DATA_A_Q8_0（K_PER_ITER=8） | DATA_A_BF16（K_PER_ITER=4） |

`ggml_vk_dispatch_pipeline` 的 elements 语义（10639-10642 行）：`wg = CEIL_DIV(elements, wg_denoms)`，wg_denoms={1,1,1} 时 workgroup 网格即 `{m, dispatch_rows, 1}`；grid-y 即 `gl_WorkGroupID.y` = token_row。源码未显示 BF16 变体的 workgroup 形状与 Q8 变体有任何不同——**rows 语义应与 Q8 变体一致（4 \* n_rows）**。

### 3.2 修复

三元判断加入 `contraction == ctx->device->pipeline_hc_project_fold_bf16 ||`，并附注释。

### 3.3 如实记录：4\*n_rows 的静态张力（未定案，交对拍裁定）

静态推导（本工单与 bf16-shader-audit 一致）：`token_row < T` 的 workgroup 已把 mixed 的全部 `T*W` 个元素恰好各写一次（`data_d[first_row] = t*W + j`，reduce_result 直接以 first_row 寻址）；`y ∈ [T, 4T)` 的 workgroup 读越界 A/B/fuse0、写越界 D，其存活依赖 Vulkan robust buffer access 语义（越界写被驱动丢弃）——即 **4\*n_rows 疑似冗余 over-dispatch**。但 Q8_0 版在 HC-Q8 生产轮（2026-10-08）以 4\*n_rows 跑出正确输出，审计明确记录此张力「未能定案，建议由单 site 对拍数据裁定」。

本修复按任务指令与 Q8 变体对齐（4\*n_rows），理由：(a) 两变体同构，dispatch 几何一致使单 site 对拍时数值差异只能归因于 A 解码（Q8_0 block scale vs BF16 位展开），不再混入 dispatch 分叉——这正是恢复工单 A 数值验证所需的控制变量；(b) 若 4\*n_rows 实为 over-dispatch，代价是 3/4 冗余 workgroup（性能损失，正确性由 robustness 吸收），与 Q8 生产行为一致；若静态推导有误，则不加分叉会漏算。对拍数据（§5）可同时裁定该问题。

## 4. 修改清单（工作树）

1. `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp`——`iter_aligned_nonquant` 通用循环 n 循环加 `#ifdef HC_UP_FOLD` 折叠分支（与 `iter()` 三处折叠逐 token 同构）。F32 特化分支（`NUM_COLS==4/15 && NUM_ROWS==1`）**未动**：当前无 F32+HC_UP_FOLD 变体注册，见 §7 疑点。
2. `ggml/src/ggml-vulkan/ggml-vulkan.cpp`——`ggml_vk_segment_project` 的 `dispatch_rows` 三元加 `pipeline_hc_project_fold_bf16`。
3. `docs/TP5-BF16-UPFOLD-FIX.md`——本文。
4. `tests/test-tp5-upfold-indexing.py`——纯 Python 索引镜像回归测试（§6）。

未动：`ggml-vulkan-collective.cpp`（DevOps 在隔离树上做实验 2b，避免混淆）；工单 A 既有改动（capture gate 放宽、family guards 等 dirty 内容）；`vulkan-shaders-gen.cpp`（宏集无需变化——修复在 shader 源内）。

## 5. 验证方案（DevOps 执行，先对拍再性能）

前置：SPIRV 重生成（`qwen4_hc_project_fold_bf16` 变体重编译）；`python3 tests/test-tp5-upfold-indexing.py`（纯 CPU，无 GPU 需求）。

**单 site teacher-forced 固定 token 回放**（沿用 `docs/TP5-BF16-LATEBIND-VERIFICATION.md` §4.2 的框架，本修复解除其阻塞）：

1. 选一个 HC site，冻结该层的输入 token 序列（teacher-forced 回放，非自然生成）。
2. 逐层 dump 并比较中间量：`combined`（residual 合并后）、`normalized`（xn）、`lo`（down 收缩 + SiLU 输出）、`mixed`（up fold 输出）。修复前 `mixed` 全错（乱码源）；修复后应与参考实现（Q8_0 变体路径或 CPU 参考）在 BF16 解码精度内一致——注意 BF16 权重展开是精确 F32 值，与 Q8_0 路径的数值差异只应来自 Q8_0 量化本身，若与 CPU F32 参考比较则应在浮点舍入级。
3. `down_silu_bf16` 同轮顺带确认（审计静态判定其正确，但从未运行验证）：`lo` 中间量对拍。
4. dispatch 几何确认：记录本轮 `dispatch_rows` 实际值（应为 4\*n_rows）与 y∈[T,4T) workgroup 的实际行为（越界写是否被 robustness 吸收、有无校验层告警）——同时裁定 §3.3 张力。
5. NO_HIT 判定（既有文档 §5）不受本修复影响，照常执行。
6. 数值资格通过后才进入无插桩完整请求 pilot（既有文档 §4.3）；性能排名在最后。任何机械失败（GPU fault、校验层越界告警、HTTP 500）保留原始记录，不放宽门禁。

失败出口：若修复后 mixed 仍错 → 回到 §1.1 重审（下一个嫌疑：binding 布局/push constants 传参，审计已静态核对吻合）；若对拍通过但完整请求乱码 → 查跨层状态（KV/recurrent/carry）与其它 site 的变体选择。

## 6. 回归测试（tests/test-tp5-upfold-indexing.py）

纯 Python 镜像 `iter()` 与 `iter_aligned_nonquant` 的索引算术（两者是 first_row/n/ncols/stride_d/col/QUANT_K 的纯整函数），断言清单：

- a) **退化**：HC_UP_FOLD=0 时两镜像产生相同 (weight_row, ib) 序列，且等于线性闭式——折叠公式退化为线性索引，非 fold 变体零行为变化。
- b) **折叠一致**：HC_UP_FOLD=1 时 `iter()` 镜像与修复后 `iter_aligned_nonquant()` 镜像产生相同 weight_row 序列（两份独立转录，非同一函数复用）。
- c) **红绿回归**：修复前镜像（恒线性）在 t≥0、n≥1 即与折叠序列分歧（含 t=0——审计表述的保守修正）；给出具体越界反例（W=10、j=8、t=3、n=3 读行 41 > 4W-1=39）；折叠行恒 < 4\*stride_d。
- d) **语义钉住**：折叠 weight_row 集合恰为 `{j + n*stride_d}`——reduce_result 语义（fuse0 索引 `t*4W + n*W + j`、per-stream sigmoid、0.25 折叠）所要求的 W_up 行。
- e) **对齐不变量**：折叠后 ib 保持偶数（packed32 读前提）。
- f) **源码哨兵**：括号配对提取 `iter_aligned_nonquant` 函数体断言含折叠；`dispatch_rows` 子句断言含 `pipeline_hc_project_fold_bf16` 与 `4 * n_rows`——任一修复被 revert 即红。
- g) **dispatch 几何钉住**：y∈[0,T) 对 mixed 输出恰好一次完全覆盖；y≥T 的写地址恒越界（把 §3.3 的静态推导变成可执行记忆，不判定 4\*n_rows 对错）。

**可表达性边界（如实）**：GLSL 执行语义不可在纯 Python 层断言——subgroupAdd 的归约顺序、`dequantize4_2aligned` 的 BF16 位展开数值、越界 SSBO 访问的 robustness 行为。这些归 §5 的 DevOps teacher-forced 对拍。测试断言的是索引算术与不变量，不是端到端数值。

## 7. 边界、疑点与未决事项

- **F32 特化分支疑点（未修，超出本次范围）**：`iter_aligned_nonquant` 的 F32 特化分支（`NUM_COLS∈{4,15} && NUM_ROWS==1`，mul_mat_vec.comp 376-388 行）无折叠。当前无 F32+HC_UP_FOLD 变体注册，不触发；若未来注册此类变体，需同样补折叠。已记为边界条件，不在本次最小修复内。
- **4\*n_rows 张力**（§3.3）：静态推导与 Q8 生产记录的分歧未定案，交 §5.4 对拍裁定。本修复选择与 Q8 对齐。
- **down_silu_bf16 未运行验证**：审计静态判定正确（线性行 + SiLU 归约路径与 Q8_0 共享），但从未真机执行；§5.3 顺带对拍。
- **late Q 变体**（`tp5_hc_latebind.comp` 的 `TP5_LATE_Q_BF16`，collective.cpp 侧）：与本次修复的 fused 变体是不同资产，未审未改。
- 本修复不含性能主张：up_fold BF16 变体修复后是否比 Q8 转换路径快，属于工单 A 后续的性能轮，须先过 §5 数值资格。


## GAP1 最终根因（R2 判别实验后更新）：mmap 加载路径对 meta buffer 绕过 MIRRORED fan-out

**判别证据**（identity/quarantine-sitedump/r2-verdict.json + r2-parity-analysis.json，50 个 dump）：
全 10 个 (site,rank) 组合一致呈现 W_down/W_up 的 A binding subbuffer **全零**（64/64 u16 = 0），
而 normalized 全非零（真实激活值）。subbuffer 结构存在：r0 的 W_down {buf, off=1049041280,
size=6553600}，W_up 同 buffer、off 恰好相邻（相差 6553600）；逐 rank 句柄不同。size=6553600 =
320×10240×2，正是 BF16 W_down 全量字节数——**分配正确、字节从未写入**。SPIR-V 执行层无罪
（kernel 忠实执行零×激活）；Q8_0 权重的同链路正常（3c HC-Q8 输出正常）。

**机理（完整归因链）**：

1. TP5 下 HC 段权重（hc_attn_down/up 等 BF16 张量）在 plan 层判为 MIRRORED
   （src/llama-tp5-plan.cpp:397-400，类型无关）；meta buffer 的真实存储是 5 个 rank 的
   VkBuffer，字节必须经 ggml_backend_meta_buffer_set_tensor 的 MIRRORED 分支
   （ggml-backend-meta.cpp:2330-2338）逐 rank fan-out 写入。
2. llama_model_loader::load_all_data 的 mmap 路径（src/llama-model-loader.cpp:1670-1671）在
   cur->data == nullptr 时走 ggml_backend_tensor_alloc(buf_mmap, cur, data)。
   ggml-backend.cpp:2388-2400 的实现只做 tensor->buffer = buffer; tensor->data = addr
   （addr = host mmap 指针），**从不调用 set_tensor**。对 CPU buffer 这是 mmap 直读的正确
   语义；对 meta buffer 它把「绑定」误当「上传」，fan-out 被整体绕过。
3. 结果：rank VkBuffer 里 tallocr 分配的 W_down/W_up 区域（dump 看到的合法相邻 offset）
   从未收到任何字节；dispatch 的 A binding 读到零页 → lo = W_down·normalized = 0 →
   mixed = fold(up(silu(0)))·W_up = 0。这同时解释 V-A 两路径逐字相同（权重零与解码路径
   无关）。
4. Q8_0 对照臂为何正常：其权重张量进入该分支时 data != nullptr（走了预分配路径），落入
   else 分支的 ggml_backend_tensor_set(cur, data, 0, n_size) → MIRRORED fan-out 正常执行。
   BF16 与 Q8_0 的分叉不在类型分派本身，而在两条臂的 tensor 预分配状态差异；本修复对
   两种状态都强制走 fan-out，不再依赖该差异。

**修复**：src/llama-model-loader.cpp mmap 分支，tensor_alloc 之后若 buf_mmap 是 meta buffer，
显式调用 ggml_backend_tensor_set(cur, data, 0, n_size) 补齐 fan-out。CPU/非 meta buffer
路径零行为变化（is_meta 判定为假时新增块不执行）。

**回归测试**：tests/test-tp5-upfold-indexing.py::UploadChainAuditTests——
(a) test_mmap_meta_upload_fans_out_after_tensor_alloc：修复前红（meta 分支无 set）；
(b) test_bf16_weight_shard_byte_width_matches_dump：BF16 字节宽度换算与 dump 的 6553600 对账；
(c) test_mirrored_meta_set_tensor_fans_out_to_all_ranks：钉住 fan-out 循环不被类型/数量门控。

**DevOps 验证建议**：重编后跑 sitedump-fused 同配置——预期 W dump 非零 → lo 非零 →
mixed 非零；随后 off 臂三重判定（对拍通过 + 正文不乱码 + dump 复查 lo 非零）恢复 baseline。
若 W 仍全零，则剩余假设收窄为「本尊 data 绑定状态与克隆链 data 计算的交互」，
需在加载期打印 W 的 {buffer, data} 判别。

**未决事项**：Q8_0 臂「data != nullptr」的精确成因（预分配时序）为静态推断，
未经运行时日志直接确认；修复不依赖该解释成立。


## 根因更新（r7 更正：已证伪）：GAP-1 容器路由分叉（r5 loadtrace 判别）

> **r7 更正**：本节的「容器路由分叉」根因**已被证伪**。r6 实测守卫全
> 流程零触发（GGML_ABORT 从未走到）；r7 同一 run 内逐 rank 判别证明
> 加载期与 dispatch 期张量身份完全匹配——r5 比较的两个 buffer 地址
> （0x570564428160 与 0x5705644297d0）来自**不同 run**（ASLR/分配布局
> 不同），系跨 run 比较伪影。两道守卫保留为 fail-closed 纵深防御，
> meta.cpp 注释与 tests docstring 已改写为真实状态；下文 r5 机理叙述按
> 历史原样保留作证据索引，**不作为有效根因**（乱码根因已收窄到写入
> 可见性层，见 DECISION.md §14.3–14.5 与 identity/quarantine-dispatchtrace/
> verdict.md）。

### 运行时证据（r5，2688 行 loadtrace + 50 dump；跨 run 比较，解读已被 r7 修正）

- **加载链正常**：384 次 MIRRORED set_tensor 全覆盖（48 层 × 8 HC 张量，
  完整 6553600 字节），5 rank simple tensor 全有效（rank0
  simple{buf=0x570564428160, data=0x6c8f000}），零 SIMPLE_TENSOR_NULL。
- **dispatch 侧分叉**：W_down A 绑定 subbuffer{buf=0x5705644297d0,
  off=1049041280, size=6553600}——不在加载期 5 rank 句柄集合中；
  5 rank × 0/64 全零，与「写入目标 ≠ 读取目标」精确自洽。
- **算术定位**（vk_tensor_offset = data - 0x1000，ggml-vulkan.cpp:3097）：
  dispatch 绑定反推张量 data=0x3E801000，即**第二个 meta buffer 上
  tallocr/gallocr 的 0x3E800000 偏移分配**——一个 W_down 尺寸、与 W_up
  相邻、**从未被加载写入**的新分配；加载写入的是第一个 meta buffer 的
  stc_static（data=0x6c8f000）。

### 根因假说（r5 提出；r7 证伪为跨 run 比较伪影，未真实发生）

r5 当时的推断：`ggml_backend_meta_alloc_ctx_tensors_from_buft`
（ggml-backend-meta.cpp）的重绑循环对 ctx 中每个张量无条件执行
`t->buffer = meta_buf`；若同一权重 ctx 在图构建/调度侧被**第二次**送入
meta 分配（ggml-alloc.c:1652 对 meta buft 的特判转发），权重张量的
buffer 归属被重绑到**第二个 meta buffer**：
其 stc_static 里生成了全新 per-rank simple tensor（tallocr 分配
0x3E800000 处），但加载期的 MIRRORED fan-out 写入发生在**第一次**分配的
buffer 上——写入与读取自此分叉到两个互不相交的分配。图分裂
（meta.cpp:2030 的 `t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(...)`）
按 `tensor->src[i]->buffer` 解析 simple tensor，于是 dispatch 拿到第二套
从未写入的分配（全零）。

Q8_0/量化 HC 路径（3c 正常）不受影响：其权重解析始终命中第一次分配的
stc_static——量化路径的图/调度链从未触发这次第二次重绑分叉。

### 守卫（r6 证明非主修复；保留为 fail-closed 纵深防御）

1. **守卫一**：重绑循环中，若张量已在**另一个 meta buffer 的
   stc_static** 完成加载期注册（`t->buffer` 是 meta 且其 stc_static 已含
   该张量），保持原归属（`continue`）。r6 全流程零触发、r7 同 run 身份
   匹配证明「第二次分配」在当前路径上不存在；保留以钉住「已注册权重
   所有权不被重绑」的不变量，并对齐标准 alloc「跳过已分配张量」语义。
2. **守卫二（fail-closed 纵深）**：`ggml_backend_meta_buffer_simple_tensor`
   的 lazy init 路径上，权重型张量（op==NONE、非 view）若不在本 buffer 的
   stc_static 注册，直接 GGML_ABORT 并打印张量名。r6 证明权重始终命中
   stc_static、从未落入此路径；保留以把未来可能的路由回归从「静默全零
   乱码」变成当场暴露的路由错误。

### 退化论证

守卫条件极窄：仅当张量**已注册于另一 meta buffer 的 stc_static**（即加载
期完成的权重）才触发。compute 张量、external view（容器 FIXME 所述场景）
从不在别的 buffer 的 stc_static 注册，守卫零触发——Q8_0/Q2_0/所有量化路径
行为不变（3c 证据：量化 HC 路径本就正常）。

### 验证建议与执行结果

1. `python3 tests/test-tp5-upfold-indexing.py`：守卫存在性断言保持绿
   （断言钉住守卫代码文本与不变量；docstring 已改写为「历史防御性
   守卫」，不再声称「修复前镜像红」的分叉真实发生）。
2. loadtrace 复查（r6/r7 已执行）：同 run 内 dispatch 侧绑定与加载期
   rank simple tensor 身份**完全一致**（r7 verdict 逐 rank 对照表）——
   r5 时代「绑定不匹配」系跨 run 地址比较伪影。
3. W dump 复查（r6/r7 已执行）：仍全零——归因已收窄到写入可见性层
   （见 DECISION.md 第 14/15 节），与本节证伪的路由分叉无关。
4. off 臂三重判定：连续多轮被前置 dump 复查阻断，如实记录于
   DECISION.md。
