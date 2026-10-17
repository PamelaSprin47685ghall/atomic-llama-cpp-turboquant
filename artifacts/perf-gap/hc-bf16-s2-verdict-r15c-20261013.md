# GAP1 r15c：S2 嫌疑静态深查裁决（2026-10-13）

范围：纯静态审查，隔离树 `/var/tmp/tp5-verify-rebuild-src/ggml/src/ggml-vulkan/`。不改源码、不编译、不执行。
承接：r15a（`hc-bf16-audit-r15a-20261013.md` 的嫌疑清单与 CPU 参照方案）、r23（S1 静态排除：QUANT_K=1 实锤、push constants 元素单位一致）。
背景形态：乱码分歧在第 3 token（前 2 token 一致）；dump 锚点跨轮逐字节稳定（r22=r16）；「量级合理但值错」。

## 裁决：S2 静态排除（排除的具体机理：调用侧与实现参数语义不匹配 / packed32-iqs 组合错位 / A-B 列偏移分歧）

## 一、证据链（全部为本轮逐行读出的源码事实）

### E1｜iqs 在调用侧是硬编码 0，B 读取与 A 传参同一变量
`mul_mat_vec.comp:403-406`（iter_aligned_nonquant 的 j 循环头部）：
```glsl
const uint col = i*BLOCK_SIZE + K_PER_ITER*tid;
const uint iqs = 0;            // quant index
const uint iybs = col;         // y block start index
```
- iqs **不是**从 col 推导（不是 col%QUANT_K/QUANT_R），是常量 0。
- line 408 的 B 读取 `data_b_v4[(j*p.batch_stride_b + b_offset + iybs + iqs)/4]` 与 line 430 的 A 传参 `dequantize4_2aligned(ib, iqs, a_offset)` 引用**同一个 iqs 变量（同作用域、同值 0）**。
- **「B 用列相关值而 A 用 0」的分歧不存在**。B 索引 = (j*batch_stride_b + b_offset + col)/4，B 元素 col..col+3。

### E2｜dequantize4_2aligned 的 BF16 实现体内 iqs 完全未使用
`dequant_funcs.glsl:54-61`：位展开只用 `(a_offset + ib)`，iqs 仅存在于签名（为与量化类型共享接口保留）。BF16 的 QUANT_K=1，无块内偏移概念，iqs 语义上就该是 0——调用侧传 0 与实现的忽略**互相印证、无错位源**。

### E3｜QUANT_K=1 / QUANT_R=1 / A_TYPE_PACKED32=uint32_t 三重实锤
- `types.glsl:43-45`：`#if defined(DATA_A_BF16)` → `#define QUANT_K 1`、`#define QUANT_R 1`。
- `types.glsl:54`：`#define A_TYPE_PACKED32 uint32_t`（真 u32 数组；对比 F16 的 f16vec2，line 40）。
- `vulkan-shaders-gen.cpp:1127-1146`：`qwen4_hc_down_silu_bf16` / `qwen4_hc_project_fold_bf16` 宏字典 = {DATA_A_BF16=1, B_TYPE=float, B_TYPEV2=vec2, B_TYPEV4=vec4, D_TYPE=float, USE_SUBGROUP_ADD_NO_SHMEM=1, HC_DOWN_SILU/HC_UP_FOLD=1}；注释明写「QUANT_K=1/QUANT_R=1 via DATA_A_BF16; get_dm returns (0,0)」。
- `mul_mat_vec_iface.glsl:9-26`：data_a（A_TYPE）与 data_a_packed32（A_TYPE_PACKED32）、data_b 与 data_b_v4（B_TYPEV4=vec4）是同一 binding 0/1 buffer 的多视图。小端下 packed32[i] 低 16 位 = data_a[2i]、高 16 位 = data_a[2i+1]。

### E4｜A/B 列配对一致（两分支）
- **HC_DOWN_SILU（非 HC_UP_FOLD）**：`ib = (ibi+col)/QUANT_K`（line 426，ibi 步进 p.ncols）→ QUANT_K=1 下 ib = (first_row+n)*ncols + col。A 读权重第 (first_row+n) 行的 col..col+3 列；B 读激活 col..col+3。**同一 col，dot(v,b) 配对正确**。
- **HC_UP_FOLD**：weight_row = (first_row%stride_d) + n*stride_d（line 422，upfold 修复后），ib = weight_row*ncols + col。A 的 K 维列索引与 B 的索引同为 col。配对正确。
- B 读取的 vec4 整除：is_aligned_nonquant 门槛（`mul_mat_vec.comp:511-515`）要求 batch_stride_b%4==0、b_offset%4==0、ncols%4==0；col = K_PER_ITER*(i*BLOCK_SIZE+tid) 是 4 的倍数 → (j*stride_b+b_offset+col) 被 4 整除精确，data_b_v4[k] 恰覆盖 col..col+3，无跨 vec4 错位。

### E5｜packed32 索引无奇偶截断
`(a_offset+ib)/2` 要求偶数：col 是 4 的倍数；ncols = hc_dim = 4*n_embd（偶）；row*ncols 偶；a_offset = batch_idx_a*(batch_stride_a/QUANT_K) = batch_idx_a*batch_stride_a（元素单位，r23 已裁决），batch_stride_a = 权重 batch 元素数（行数×ncols，偶）。**a_offset+ib 恒偶，无整除截断错位**。

### E6｜两路径逐位等价（FORCE_ITER 判据的理论基础）
- bf16_to_fp32（`types.glsl:1966-1969`）= `uintBitsToFloat(u << 16)`。
- iter() 的 dequantize4：bf16_to_fp32(data_a[a_offset+ib+k]) = uintBitsToFloat(u16 << 16)。
- aligned 的 dequantize4_2aligned：元素 2i 位 = (packed32[i] & 0xffff) << 16 = data_a[2i]<<16；元素 2i+1 位 = packed32[i] & 0xffff0000 = data_a[2i+1]<<16（高 16 位直用 ≡ u16 左移后占据高位）。
- **同一 buffer、同一位展开、同一元素序——两解码路径逐位等价**。
- lane→K 切片映射两路径同公式（col = K_PER_ITER*(i*BLOCK_SIZE+tid)，调用侧都传 i*K_PER_ITER）；HC_UP_FOLD/HC_DOWN_SILU 行映射两路径代码一致（line 243-250/272-279/331-340 vs 412-428）；dot 累加同序（temp[j][n] += dot(v,b)）；reduce 同函数。
- 尾段：num_iters per-lane 调整（line 527-530）保证所有执行的 col < ncols；ncols%4==0（aligned 门槛）保证 col..col+3 界内。iter() 的 lastiter/OOB 机制仅服务 ncols%4!=0 的形状，HC 形状（hc_dim、320 均为 4 的倍数）不触发。

### E7｜get_offsets 与 first_row（共享路径，两变体同走）
`mul_mat_vec_base.glsl:97-169`：a_offset = batch_idx_a*(batch_stride_a/QUANT_K)（QUANT_K=1 → 元素单位直通）；b_offset/d_offset = batch_idx*batch_stride_b/d。`mat_vec_first_row`（line 215-230）：HC_UP_FOLD 编码 token_row*stride_d+col；HC_DOWN_SILU 纯行 NUM_ROWS*(wgID.x+...)。

## 二、S2 排除的边界（诚实声明）
静态排除的是**源码语义层的错位机理**（参数不匹配、iqs 分歧、packed32 截断、A/B 列错配）。静态分析不能裁决的：(a) 驱动/编译器对 aligned 路径实际生成的机器行为——理论逐位等价 ≠ 实测等价，且 HC BF16 是全树仅有的 mul_mat_vec+DATA_A_BF16 实例（无独立执行证人，mul_mat_vec.comp:25-27 注释自认）；(b) host 填入 push constant 的**具体数值**与 descriptor 绑定（r23 排除的是单位契约，不是值的正确性）。

## 三、剩余嫌疑重排（S2 排除后）

1. **H1（最高）｜host 侧 ggml_vk_segment_project 的参数值组装与 descriptor 绑定**：stride_d/batch_stride_b/base_work_group_y/ne02/ne12/broadcast2/broadcast3 的具体值、A/B/D/fuse0 绑定的实际 buffer 与 offset。**静态缺口跨轮遗留（r15a 列为第一优先，本轮预算仍未读到该函数体）——这是唯一未被静态审过的 host 环节。**
2. **H2（高）｜normalized 输入锚点本身的正确性**（上游 residual/rms_norm 的状态累积型错误）：形态兼容性分析——确定性 kernel 错位（S1/S2 类）从 token 0 就错，「前 2 token 对」需要错误恰好不改 argmax 的巧合；**状态逐 token 漂移的上游错误天然解释「前面对、第 3 个乱」**。且 r15a 方案存在盲点：normalized 只被当输入锚点，从未验证自身正确。
3. **H3（中）｜驱动/编译器对 aligned 路径的实际行为**：vec4 B 读取 + packed32 位操作在 RADV 上的生成代码。FORCE_ITER（0x200）一发 A/B 裁决。
4. **H4（中）｜up 段 host 绑定（data_fuse0 布局、up 权重 rank 切片）**：与 H1 同属 host 侧，可合并审查。

## 四、运行时判据优先级（GPU 释放后，交 DevOps）

1. **CPU 参照对拍（r15a 方案 + 本轮盲点修正）**：
   - **新增前置检查**：先验证 normalized 自身——用前一层的 residual dump + GGUF hc_attn_norm 权重在 CPU 复算 normalized，对拍 dump normalized。**若 normalized 已错，直接定罪上游（H2），HC 段无罪释放**；若 normalized 对，再按 r15a 流程对拍 lo/mixed。
   - 判读规则不变（相对误差 <3e-3 → 正确；>5e-2 或结构性模式 → 定位），但 lo 对拍失败时的归因顺序改为：先查 normalized（本轮修正），再查 host 参数（H1），最后才是 kernel（此时 FORCE_ITER 区分 H3）。
2. **FORCE_ITER A/B（0x200 push bit）**：仅当对拍显示 kernel 数值错时执行。静态等价论证（E6）成立下，若 FORCE_ITER 翻转结果 → H3 升级实锤（驱动/编译器）；若不翻转 → 共享路径（行映射/reduce/host 值，H1）。
3. **若 lo/mixed 全对**：嫌疑整体移出 HC 段——上游（H2，往前一层 residual/rms 对拍）或下游（mixed 的消费者/attention 输入），扩大对拍窗口。

## 五、本轮未决
- ggml_vk_segment_project 函数体未读（预算取舍，H1 的实证缺口，连续第三轮遗留——下一轮静态工作若还有，这是唯一必读项）。
- tp5_hc_latebind.comp 的 TP5_LATE_Q_BF16 分支仍未审（若乱码臂走 latebind capture 而非 fused segment，嫌疑面不同；dump 锚点来自 fused 段插桩，本轮聚焦未跑偏）。
- dump 文件实际长度/层名/n_rows 待执行时从文件反推（r15a 已设计自洽反推规则）。
