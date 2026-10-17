# H1 裁决：ggml_vk_segment_project 函数体静态审查（GAP1 第十六轮）

更新：2026-10-13。状态：纯静态审查，未改源码、未编译、未运行。隔离树
`/var/tmp/tp5-verify-rebuild-src/ggml/src/ggml-vulkan/ggml-vulkan.cpp`。

## 结论

**H1 静态排除。** `ggml_vk_segment_project`（16938-17020）的 push constant 值
组装、descriptor 绑定、dispatch workgroup 计算与两个 HC BF16 pipeline 的
shader 期望（mul_mat_vec.comp + mul_mat_vec_base.glsl，r15c 已审）逐项匹配，
无 k/m 填反、无误用 ne 索引、无绑定错接、无 dispatch 维度错位。连续三轮
的静态缺口（r15a 提出、r15c 复列、本轮闭合）至此清偿。

## 证据链

### 1. push constant 字段映射

`vk_mat_vec_push_constants` 结构（ggml-vulkan.cpp:1491-1505）13 字段顺序：
ncols, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b,
batch_stride_d, fusion_flags, base_work_group_y, ne02, ne12, broadcast2,
broadcast3。

host 填充（16975-16977）：`pc{ k, k, k, m, nelements(W), k, m,
output_flags, 0, 1, n_rows, n_rows, 1 }`，其中 k = weights->ne[0]、
m = output->ne[0]、n_rows = output->ne[1]。

| 字段 | host 值 | down（W_down[hc_lr_slice,hc_dim] → lo[hc_lr_slice,T]） | up fold（W_up[4W,hc_lr] → mixed[W,T]） | 裁决 |
|---|---|---|---|---|
| ncols | k | hc_dim（K 维） | hc_lr（K 维） | ✓ |
| stride_a | k | hc_dim | hc_lr | ✓ |
| stride_b | k | hc_dim | hc_lr | ✓ |
| stride_d | m | hc_lr_slice | width（fold 行折叠基数） | ✓ |
| batch_stride_a | nelements(W) | hc_lr_slice×hc_dim | 4W×hc_lr | ✓（且 batch_idx_a 恒 0，见下） |
| batch_stride_b | k | hc_dim | hc_lr | ✓ B token stride |
| batch_stride_d | m | hc_lr_slice | width | ✓ D token stride |
| fusion_flags | output_flags | 0/ELIDE_LOCAL | 0/ELIDE_LOCAL | ✓ |
| base_work_group_y | 0 | 0 | 0 | ✓ |
| ne02 / ne12 / broadcast2 / broadcast3 | 1 / n_rows / n_rows / 1 | 同左 | 同左 | ✓ |

**batch_idx_a 推导**（get_offsets）：batch_idx = wg.y + 0 ∈ [0, n_rows)；
i13 = batch_idx/ne12 = 0；i12 = batch_idx；i02 = i12/broadcast2 = 0（batch_idx
< n_rows = broadcast2）；i03 = 0；batch_idx_a = 0×ne02 + 0 = 0。**权重 batch
偏移恒 0——正确**（权重不随 token 变，a_offset = 0，QUANT_K=1 无缩放）。

**k/m 填反专项**：k 恒取 weights->ne[0]（收缩维），m 恒取 output->ne[0]
（输出维）。down：k=hc_dim、m=hc_lr_slice ✓；up：k=hc_lr、m=width ✓。任务
点名的「k/m 填反或错误 ne 索引」不存在。

### 2. descriptor 绑定（17001-17003）

descriptors{A=weights_sub, B=b_sub, D=out, fuse0=extra-or-out, out}：

| binding | down 调用（17465） | up 调用（17509） | 裁决 |
|---|---|---|---|
| 0 A | W_down | W_up | ✓ |
| 1 B | normalized（input；q8dot 时 b_override=Q8 companion） | lo（input） | ✓ |
| 2 D | lo（output） | mixed（output） | ✓ |
| 3 fuse0 | out 占位（extra=nullptr，down 不消费 fuse0） | normalized（extra） | ✓ |
| 4 | out 占位 | out 占位 | ✓ |

subbuffer 的 offset/size 正确性有运行时直接证据：R2 判别实验
（docs/TP5-BF16-UPFOLD-FIX.md「GAP1 最终根因」节）dump 出 W_down
{off=1049041280, size=6553600=320×10240×2}——BF16 全量字节宽度精确匹配
张量形状，W_up 相邻 offset。绑定地址解析无误。

### 3. dispatch workgroup（17011-17018 + 6668-6679）

pipeline 创建参数实锤：
- `pipeline_hc_down_silu_bf16`（6674-6676）：spec {subgroup_size, **1**, 1}
  ——NUM_ROWS=1，wg_denoms {1,1,1}。
- `pipeline_hc_project_fold_bf16`（6677-6679）：spec {subgroup_size, **4**,
  1}——NUM_ROWS=4，wg_denoms {1,1,1}。与 Q8_0 twin（6671-6673）完全同构。

dispatch：grid = {m, dispatch_rows, 1}；fold 三变体（含 bf16，修复后）
dispatch_rows = 4×n_rows，其余 = n_rows。

- **down**：grid = {hc_lr_slice, T, 1}。shader first_row = NUM_ROWS×(wg.x +
  numX×wg.z) = wg.x ∈ [0, hc_lr_slice)——恰好覆盖 lo 全部 m×T 元素，每元素
  恰一次；batch_idx = wg.y ∈ [0, T)。✓
- **up fold**：grid = {width, 4T, 1}。HC_UP_FOLD 编码 first_row =
  token_row×stride_d + wg.x（不用 NUM_ROWS 因子）；y∈[0,T) 恰好覆盖 mixed 全
  部 W×T 元素；y∈[T,4T) 为 over-dispatch（见下）。✓
- **y 维语义区分**（任务点名项）：down 的 wg.y=batch_idx（token）与 fold 的
  wg.y=token_row——**两者同构（都是 token 索引）**，host 用同一 n_rows 填
  grid.y 语义匹配。两 pipeline 的差异不在 y 维语义，而在 fold 的 4× 因子
  （dispatch 约定，与 Q8 twin 对齐）与 x 维编码（线性行 vs t×W+j）。

### 4. 4×n_rows 张力（非缺陷，已登记未决）

y∈[T,4T) 的 workgroup：A 索引恒界内（weight_row = j + n×W < 4W）；B/fuse0/D
越界（token_row ≥ T）。其无害性依赖 robust buffer access（越界读零、越界写
丢弃）——temp=0、gate=0.5、fuse0 读零 → result=0 → 越界写被丢弃。Q8_0 版以
同 dispatch 在 HC-Q8 生产轮跑出正确输出（间接证人）；docs/TP5-BF16-UPFOLD-
FIX.md §3.3 已登记此张力并交单 site 对拍裁定（§5.4）。**注意：修复前 BF16
用 n_rows 是静态恰好覆盖、无越界的 dispatch；修复后 4T 引入 robustness 依赖
——若对拍后确认 4T 冗余，收窄为 n_rows 是零风险性能优化，不是正确性修复。**

### 5. 形态兼容性

H1 若实锤（k/m 填反、batch_stride 错、绑定错接），形态为确定性索引错乱
——与「量级合理值错+确定性」兼容。但逐项核对全部匹配，错位在源码层不存
在。「前 2 token 对」的形态兼容性由 H1 之外的机理（上游状态累积、权重值
错）解释更优——与 r15c 的 H2 分析一致。

## 剩余嫌疑重排（H1 出局后）

1. **V（写入可见性层）**：r6/r7 复查 W 仍全零后 DECISION.md §14.3-14.5
   收窄的官方主归因——权重字节在 rank VkBuffer 的落地/可见性。**时间线矛
   盾需澄清**：r22 判读 lo/mixed「量级合理」（非零），而 r6/r7 W 全零
   （lo 必零）——r7→r22 之间 W 应已从零变非零（某修复生效），但值仍错。
   「值错」本身仍是间接推断（CPU 参照对拍被 GPU 阻塞未跑成，r15c 已指出）。
2. **H2（上游 normalized 正确性）**：r15c 提出——状态累积型错误天然解释
   「前 2 token 对、第 3 token 乱」；r15a 方案的盲点（normalized 未验证即
   用作输入锚点）仍待对拍前置检查修正。
3. **H3（驱动/编译器 aligned 路径行为）**：r15c 的两路径逐位等价论证下，
   FORCE_ITER（GGML_VK_HC_BF16_FORCE_ITER=1，16951-16971 已实装 0x200 位）
   A/B 出差异即指向驱动/编译器。
4. **latebind 变体**（TP5_LATE_Q_BF16）：未审，但 dump 锚点来自 fused 段
   插桩，非 latebind 路径。
5. **4T robustness**（弱）：Q8 同 dispatch 正确为间接证人。

## 运行时判据（GPU 释放后，交 DevOps）

优先序承接 r15c 并按本轮更新：
1. normalized 前置验证（residual dump + GGUF norm 权重复算）——先钉死输入
   锚点，防 down kernel 冤案。
2. W_down/W_up A binding 头 64 u16 复查（R2 方案）——确认当前二进制上权重
   非零且值正确（对 GGUF 字节）。若 W 值错（非零但错），V 升级为实锤且指
   向上传链的值级缺陷。
3. lo/mixed CPU 参照对拍（r15a 方案 + 盲点修正）。
4. FORCE_ITER A/B——仅当对拍显示 kernel 数值错时区分 aligned/共享路径。
5. 若 lo/mixed 全对：嫌疑移出 HC 段，对拍窗口前移（residual/rms）或后移
   （mixed 消费者/attention 输入）。

## 关联

- docs/TP5-BF16-UPFOLD-FIX.md：upfold 两修复 + GAP1 根因时间线（R2 全零
  → mmap fan-out 修复 → r6/r7 仍全零 → 写入可见性层）。
- artifacts/perf-gap/hc-bf16-s2-verdict-r15c-20261013.md：S2 排除 + H2
  盲点分析 + FORCE_ITER 判据。
- 本轮未改任何源码；隔离树原样。
