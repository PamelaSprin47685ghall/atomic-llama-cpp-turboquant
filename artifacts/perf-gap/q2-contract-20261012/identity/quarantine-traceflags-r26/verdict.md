# r26 插打印运行时裁决 verdict（identity/quarantine-traceflags-r26/）

bin：416922f2（插打印版）。哨兵：52 项中 4 失败——全部为历史哨兵与插桩演进的文本失配（r4 删除 mmap 分支/r4 loadtrace 插桩/r16 dump 改造的旧断言过期），**本轮 ForceIterTraceSentinelTests 三项全绿**（不在失败列表）——历史失配交 Engineer 同步。

## 两跑打印行（决定性数据）

跑 1（TRACE_FLAGS=1，默认路径）32565 行：
`dispatch_is_down_bf16=1 dispatch_is_fold_bf16=1`（交替——fused 段两算子都走 BF16 变体）`dispatch_is_*_q8=0 q8dot=0 force_iter_env=0 force_iter_match=0 fusion_flags=0x00000000`——grid/k/m/n_rows 合理（down: grid(320,1,1) k=10240 m=320；fold: grid(2560,4,1) k=320 m=2560——upfold 4·n_rows 形态在）。

跑 2（TRACE_FLAGS=1 + FORCE_ITER=1）：
**`force_iter_env=1 force_iter_match=1 fusion_flags=0x00000200`**（全部 32070 行 dispatch 注入 0x200 位）——dispatch 布尔与跑 1 相同。

shader 侧判定确认（mul_mat_vec.comp:511-515）：`is_aligned_nonquant` 是**运行时 const bool**（读 `p.fusion_flags & 0x200`）——0x200 注入后 is_aligned_nonquant=false → **iter() 标量路径真实执行**。

## 裁决：开关链双侧全通 + 归因重估

**开关链全通实锤**（host: env→pipeline 匹配→0x200 注入，全部 dispatch；shader: 运行时判定切换路径）。r25 的「开关无效」机理推断**被本轮直接证据推翻**——路径确实切换了。

**归因重估（charge 判读指南的「开关链全通」分支）**：FORCE_ITER 下 iter() 标量路径运行，但正文与 dump（r25 B2 的 lo/mixed 与默认路径逐字节一致）**不变——iter() 与 iter_aligned_nonquant 两条几乎独立的数值实现（标量 load_b+dequantize1/4 vs vec4 data_b_v4+dequantize4_2aligned）产出逐字节相同的错值**。

**推论**：
1. **H3（aligned 路径编译产物缺陷）排除**——标量路径同样错，不是 aligned 编译产物的问题。
2. 错误在**两路径共享的层**。行映射/reduce/B 语义被 Q8 证人排除（Q8 变体正常且共享这些）——剩余共享层是 **BF16 特有的 A 寻址契约**：QUANT_K=1 下的 `ib=(weight_row·ncols+col)`（元素索引）→ data_a 的实际寻址换算（packed32 u32 索引 `(a_offset+ib)/2` 与元素内偏移——dequantize1/dequantize4/dequantize4_2aligned 三函数共享的 A 寻址层）。静态审计的 §1 排除项覆盖的是**位运算**（位展开正确），**寻址/索引契约层未被覆盖**——与「静态正确+运行时错」的矛盾在此消解。
3. 与 r24 CPU 对拍自洽：W 字节正确（GGUF 逐字节）+ A 寻址错 → kernel 读错 A 位置 → lo/mixed 值错但量级合理（读的是别处的真实权重）。

**修复指向（交 Engineer）**：深查 base glsl 的 dequantize1/dequantize4/dequantize4_2aligned 的 **BF16 A 寻址**（ib/iqs/a_offset → data_a 偏移的换算——QUANT_K=1 语义下 packed32 对齐、a_offset 单位、元素内偏移的契约）——三函数共享的寻址层是唯一能解释「两条解码路径同错同值」的层。修复后用 r24 的 CPU 对拍脚本作回归门。

## 附

跑 1/跑 2 的乱码正文均为「健康做，，，，，，且，，，在rata…」（与 r24/r25 一致——形态跨 5 轮稳定）。打印行 3.2 万/轮（限流未设——日志膨胀但可接受，诊断轮）。
