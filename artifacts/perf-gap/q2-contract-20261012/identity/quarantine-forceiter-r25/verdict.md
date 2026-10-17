# r25 FORCE_ITER A/B 判别 verdict（identity/quarantine-forceiter-r25/）

bin：9a69607d（树状态不变）。三轮 GPU：A 臂（默认）、B 臂（FORCE_ITER=1）、B2 臂（FORCE_ITER=1 + DUMP_SITES=1）。

## 三轮结果

| 臂 | env | 32 token 正文 | dump lo/mixed/normalized |
|---|---|---|---|
| A（默认 aligned） | FORCE_ITER 未设 | 乱码「健康做，，，，，，且，，，在rata…」（与 r24 逐字节一致） | —（未开 dump） |
| B（判别） | FORCE_ITER=1（resolved env 确认传递） | **与 A 逐字节一致（乱码）** | — |
| B2（判别+dump） | FORCE_ITER=1 + DUMP_SITES=1 | 同乱码 | **lo/mixed/normalized 与 r24（默认路径）逐字节一致** |

## 裁决：实验无效——FORCE_ITER 的运行时效果未达 kernel

**机理推断（决定性）**：iter() 与 iter_aligned_nonquant 是两条几乎完全独立的数值实现（B 读取：标量 load_b vs vec4 data_b_v4；A 解码：dequantize1/dequantize4 vs dequantize4_2aligned 位展开）——浮点归约顺序不同的两条路径**不可能产出逐字节相同的输出**（即使都错，错的值也会有 1e-7 量级以上的实现差异）。B2 的 lo 与 r24 逐字节一致 ⇒ **kernel 实际执行的仍是默认 aligned 路径——FORCE_ITER 开关没有改变 dispatch**。

**H3 裁决无法给出**（本轮实验设计的前提——开关能切换路径——不成立）。

**开关链断裂点候选（静态正确但运行时不活跃——与 r13 轮 capture gate「静态在但 off 臂不活跃」同型教训）**：
1. host 16951-16974 的 output_flags 组装：`force_iter_bf16 = hc_bf16_force_iter && (pipeline == pipeline_hc_down_silu_bf16 || pipeline == pipeline_hc_project_fold_bf16)`——pipeline 指针匹配条件（若 fused 段 dispatch 实际选了别的 pipeline 指针，位不注入）。
2. shader 侧 0x200 的运行时分支位置/写法（mul_mat_vec.comp 的 FORCE_ITER 检查是否真的在 iter/iter_aligned 分叉点、是否被编译器以非预期方式处理）。
3. push constant 的 fusion_flags 字段在 host→shader 的实际传递（pc 初始化第 8 字段 output_flags——绑定/布局偏移）。

**已确认**：env 传递到进程（result.json resolved env：GGML_VK_HC_BF16_FORCE_ITER="1"）、static 初始化读 env 的代码在树（16965）。

## 下一步（交 Manager）

1. **插打印定位断裂点**（插桩级）：host 侧在 segment_project 的 dispatch 处打印 pc.fusion_flags 实际值（FORCE_ITER 开/关各跑一次对照）——确认 0x200 是否注入；若注入则查 shader 侧分支。
2. 开关链修通后重做本轮 A/B 判别（H3 裁决仍待决）。
3. **判读边界**：本轮「两路径逐字节同值 ⇒ 开关无效」是机理推断而非直接观察（无 kernel 执行路径的直接证据）——插打印将提供直接证据。

## 附：A 臂基线与 r24 一致（乱码形态跨轮稳定）；B/B2 的乱码与 A 一致本身不构成「iter() 也错」的证据（开关无效下 B=A 是必然）。
