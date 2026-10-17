# GAP1 r17：两 BF16 kernel 共享路径静态深查 + 判读边界 (a) 清偿（2026-10-13）

范围：纯静态审查，隔离树 `/var/tmp/tp5-verify-rebuild-src/`，不改源码、不编译、不跑。
输入事实：r24 数值级实锤——两个 BF16 kernel 双段同错（lo[0] dump=+0.132684 vs CPU ref=-0.014946 符号反；mixed 注入正确 lo_ref 仍 worst 4.80）。S1/S2/H1 已静态排除。

## 一、判读边界 (a) 裁决：公式一致，r24 判读无需重估

核对对象：`src/models/qwen4exp.cpp::build_hc_mix/build_hc_combine`（363-431 行，主仓与隔离树同源）。
r24 CPU 参照公式逐项对照：

| 项 | CPU 实现（qwen4exp.cpp） | r24 参照 | 裁决 |
|---|---|---|---|
| lo | 382-383：`silu(scale(mm(w_down, xn), 1/hc))` → SiLU(0.25·W_down@xn)，**scale 在 SiLU 之前** | SiLU(0.25·W_down@norm) | 一致 |
| gate | 384：`sigmoid(mm(w_up, lo))`——**无额外缩放** | sigmoid(W_up@lo_ref) | 一致 |
| mixed | 387-400：`gated = xn ⊙ gate`，按 stream 切片（reshape_3d [n_embd, hc, nt]，view_2d stride=row_size*hc）求和后 `scale(1/hc)` → 0.25·Σ_b(xn_b⊙gate_b) | 0.25·Σ_b(norm_b⊙sigmoid(W_up@lo_ref)) | 一致 |
| normalized | 377-379：rms_norm 对单 stream 的 ne[0]=n_embd 维 reduce，reshape_2d [hc_dim, nt] 行内 stream 连续，乘已折叠 (1+w) 的 [hc_dim] gamma | norm（gamma 折叠） | 一致 |
| 因子 | 1.0f/hc = 0.25（非 1-hc）；build_hc_combine 的 2·sigmoid(inject/hc) 是 combine 段，不进 lo/mixed | 0.25 | 一致 |

W 形状：W_down ne=(10240,320)（ne[0]=hc_dim=K，ne[1]=hc_lr=输出）；W_up ne=(320,10240)（ne[0]=hc_lr=K，ne[1]=hc_dim=输出）。ggml mul_mat 行优先（data[row*ne[0]+col]）与 shader `ib = row*ncols + col` 一致。**无偏差。r24 的 lo/mixed 数值判读成立。**

## 二、共享层深查：静态全部排除

两 BF16 kernel = `qwen4_hc_down_silu_bf16`（HC_DOWN_SILU）与 `qwen4_hc_project_fold_bf16`（HC_UP_FOLD），同源 `mul_mat_vec.comp`。

**决定性结构事实（vulkan-shaders-gen.cpp:1112-1145）**：Q8_0 与 BF16 两变体的编译字典中 `B_TYPE="float"`、`B_TYPEV4="vec4"`、`D_TYPE="float"`、`USE_SUBGROUP_ADD_NO_SHMEM=1` **完全相同**；差异仅 `DATA_A_Q8_0` vs `DATA_A_BF16`（→ QUANT_K=32/8-per-iter 的 iter() vs QUANT_K=1/4-per-iter 的 iter_aligned_nonquant + dequantize4_2aligned）。生产正确的 Q8 变体为 B 读取骨架（get_offsets、batch_idx_a=0 推导、reduce_result 公共部分、D 写回、F32 B/D 类型匹配）提供运行时证人。

逐项核对（重点方向 a-d 全覆盖）：

| # | 嫌疑 | 核对结果 | 证据 |
|---|---|---|---|
| 1 | (a) B 的 vec4 视图 vs buffer 布局 | 排除 | B 是 F32（Q8 同声明生产正确）。索引 `(j*batch_stride_b + b_offset + col)/4`：col = 4·(i·BLOCK_SIZE+tid) 恒 4 倍数；down batch_stride_b=10240、up=320，均 4 倍数；b_offset=batch_idx·batch_stride_b 4 倍数 → vec4 组索引整除、无错位。normalized 行内 stream 连续布局与 fuse0 索引 token_row·4W+n·W+j 一致 |
| 2 | (b) A packed32 视图 vs subbuffer 对齐 | 排除 | W_down subbuffer offset=1049041280：mod 4/8/16 全零；W_up 顺延 +6553600 同样对齐。ib = weight_row·ncols + col：col 4 倍数、ncols（10240/320）4 倍数 → ib 恒 4 倍数 → `(ib)/2` 无奇偶错位，u32 读字节 [2ib, 2ib+3] = 元素 {ib, ib+1} 正确 |
| 3 | BF16 位展开（dequantize4_2aligned, dequant_funcs.glsl:54-61） | 排除 | 小端 u32 低/高 16 位 = 元素 ib/ib+1；`(a & 0xffff) << 16` = BF16→F32 位模式补零，`a & 0xffff0000` 直接成 F32 高位——逐位正确（S2 排除维持，且本轮补上了 S2 假设的对齐证据） |
| 4 | (c) subgroup 归约 vs wave64 | 排除 | spec {subgroup_size, N, 1}，local_size_x_id=0 → BLOCK_SIZE=subgroup_size。RX 6800 wave64：BLOCK_SIZE=64，1 workgroup=1 wave=1 subgroup，`subgroupAdd` 跨全 64 lane 正确；BLOCK_SIZE=32 情形下尾段逻辑同样闭合（见 #5）。无 gl_NumSubgroups>1 的丢失路径（USE_SUBGROUP_ADD_NO_SHMEM 分支不依赖 NumSubgroups=1 假设；HC_DOWN_WG 多 subgroup 变体有独立 shmem fold，且非本缺陷形态） |
| 5 | 迭代结构/尾段 | 排除 | down：ncols=10240=40·256 整除，全迭代全 lane col∈[0,10240) 无越界无遗漏。up：ncols=320，num_iters=320/256=1，尾检查 `256+4·tid<320`→tid<16 的 lane num_iters=2，其迭代 1 col∈[256,316]，vec4 读 B[316..319] 界内；tid≥16 不跑迭代 1——per-lane num_iters 是标准设计，覆盖恰 [0,320)。up fold 的 4T over-dispatch（y∈[T,4T)）由 robustness 吸收，Q8 twin 同 dispatch 生产正确（既有裁决 r16 维持） |
| 6 | (d) D_TYPE 写回 | 排除 | D_TYPE="float"，data_d[first_row]=result 直接 F32 存；lo/mixed 张量 F32（Q8 同声明生产正确）；HC_DOWN_SILU 的 `scaled/(1+exp(-scaled))` = SiLU(0.25·temp) 与 CPU 公式一致；HC_UP_FOLD 的 0.25 折叠、sigmoid、fuse0 索引与 CPU mixed 公式一致 |
| 7 | 类型误声明（B/D） | 排除 | 见上：两变体 B/D 声明相同，Q8 运行时证人成立 |

## 三、机理定位：静态无缺陷 → 错误在静态不可达层

S1/S2/H1 + 本轮共享层全部静态排除后，**kernel 源码语义层已无可指认缺陷**。r24 形态（双段同错、确定性、量级合理值错、符号反）与剩余运行时嫌疑的兼容性：

| 嫌疑 | 机理 | 与 r24 形态兼容性 | 既有证据 |
|---|---|---|---|
| **V（A 权重字节正确性/写入可见性）** | W_down/W_up 字节错（错值/错位，非零）→ 两 kernel 各自用错权重 → 双段同错、确定性、量级合理、符号可反 | **完美兼容**——无需 kernel 缺陷即可解释全部观察 | R2 实锤 W 全零（mmap 绕过 fan-out）→ 修复 → r6/r7 复查仍全零 → 归因「写入可见性层」（DECISION.md §14.3-14.5）。若 r7→r24 间 W 从零变错值，形态吻合 |
| **H3（驱动/编译器对 aligned 路径的实际行为）** | vec4 B load + packed32 位展开组合的编译产物行为异常 | 兼容（确定性、双段同错——两 kernel 都走 aligned 路径） | mul_mat_vec.comp:20-31 自认：「两 HC BF16 变体是树中仅有的 mul_mat_vec+DATA_A_BF16 实例，aligned 路径从未有独立执行证人」。FORCE_ITER=0x200 A/B 一发裁决（DevOps 并行） |
| **C（上游 normalized 字节）** | lo 的 B=normalized、mixed 的 fuse0=normalized 若来自 GPU 错误上游 → 双段同错 | 兼容，**取决于 r24 对拍是否注入了正确 normalized**——若 lo_ref/mixed 对拍用 CPU 自算 normalized 而非注入，则「kernel 双段错」的定位本身含上游混入 | 未核实 r24 的注入集；H2 状态累积假说（r15c）天然解释「前 2 token 对、第 3 token 起」 |

## 四、修复指向（依定位结果，下一轮实施）

1. **判别顺序（交 DevOps）**：① W 字节 dump 复验（R2 头 64 u16 方案，当前二进制）——若 W 字节错 → V 层实锤，修复在 upload/fan-out 链（upload-chain-fixer 领域），kernel 无需改；② FORCE_ITER=0x200 A/B——若 iter() 对、aligned 错 → 驱动/编译器层实锤，最小修复 = `is_aligned_nonquant` 排除 `DATA_A_BF16`（一行条件）或改 dequantize4_2aligned 的实际编译行为；③ normalized 字节 dump——裁决 C 是否混入。
2. **若三判别全阴**（W 字节正确、两 decode 路径都错、normalized 正确）：错误在更深运行时层（descriptor 实际绑定值/内存域可见性），静态审查已到边界，需 DevOps 的 layer-by-layer dump。
3. r24 对拍的注入集（lo_ref 是否同时注入了 normalized）需要委派人澄清——它决定 C 嫌疑的存废。

## 五、边界与未决

- 本轮纯静态：源码语义层的排除不能证明运行时正确——「静态正确 + 运行时错」正是 V/H3/C 三嫌疑的定义域。
- 判读边界 已清偿：公式一致，r24 判读成立。
- 未核实：r24 对拍的注入集；W 当前字节状态；FORCE_ITER A/B 结果（DevOps 并行中）。
