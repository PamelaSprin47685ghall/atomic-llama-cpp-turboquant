# TP5 HC 单 site 中间值 dump（GAP 1 第 2 步：site-dumper）

更新：2026-10-12。状态：**插桩源码已落盘工作树，未编译、未运行**。本文档是
对拍操作手册，不是执行记录。

## 1. 目的

定位 fused HC 段 BF16 乱码的根因环节。五轮静态审计已排除索引级缺陷
（「静态无错 ≠ 运行正确」），V-A/V-B 归因现状把嫌疑收口到 fused 段对 BF16
HC 权重的处理链。本工具在**第一个 HC site** 的三个中间张量上取运行时快照，
在两个二进制之间对拍：偏差**首次出现**的环节即根因所在。

## 2. 机制

### 2.1 开关

```
GGML_VK_HC_DUMP_SITES=N     # N ≥ 1：按 graph 顺序 dump 前 N 个 HC site
                             # 未设或 0：零行为变化（无屏障、无拷贝、无提交、无 IO）
```

与既有 opt-in 诊断开关（`GGML_VK_HC_BF16_FORCE_ITER` 等）同模式：函数内
static env 解析，默认 0 时所有 dump 调用点全部短路。

### 2.2 dump 点（`ggml_vk_hc_segment`，ggml/src/ggml-vulkan/ggml-vulkan.cpp）

| # | 张量 | 语义 | 插入位置（时序保证） |
|---|---|---|---|
| 1 | `normalized` | HC norm 输出（F32，4×width 流堆叠） | norm dispatch + segment_barrier + tape `hc_norm_end` 注记**之后**、down contraction 读取它之前 |
| 2 | `lo` | down/SiLU 输出（F32） | down dispatch + barrier + tape `hc_down_end` 注记之后、up/fold contraction 读取它之前 |
| 3 | `mixed` | up/fold 输出（F32） | up dispatch + tape `hc_up_end` 注记之后、函数返回前（任何下一区域消费者录制之前） |

三个张量在各自的 dump 点与下一次改写之间都只有只读消费者，copy 观察到的
就是产生该张量的 dispatch 的输出本身。

### 2.3 读回路径与时序依据

读回复用仓库既有的 staging 读回模式（`ggml_backend_vk_get_tensor_2d_async`
的 fallback 路径同构），但为满足「dispatch 完成后读」的要求采用
**mid-graph synchronize**：

1. `ggml_vk_sync_buffers(nullptr, subctx)`：录制 shader-write →
   transfer-read 执行屏障（`nullptr` ctx 与既有读回路径一致，不清
   graph 级 unsynced-node 跟踪）；
2. `vkCmdCopyBuffer(tensor dev → ctx->sync_staging)` 录制进**与产生该
   张量的 dispatch 相同的 command buffer** 且位于其后——同队列提交顺序
   + 上述屏障保证 dispatch 的写入先于 copy 完成；
3. `deferred_memcpy(host, staging->ptr, bytes, &subctx->out_memcpys)` 登记
   host 侧落地（staging 是 HOST_VISIBLE|HOST_COHERENT）；
4. `ggml_vk_synchronize(ctx)`：结束录制 → `ggml_vk_submit`（seqs 提交后
   清空）→ 空提交挂 `ctx->fence` → `ggml_vk_wait_for_fence` →
   out_memcpys 落地。fence 等待在 dump 代码内完成，读回正确性不依赖
   任何外层消费时机的假设；
5. `ggml_vk_ctx_begin(ctx->device, subctx)` 重开 command buffer，graph
   其余 dispatch 继续正常录制。

mid-graph synchronize 是本文件的既有模式（`ggml_vk_submit` 内
`replay_pending_bufs` 去重注释即为它而写；seqs 在提交时清空，graph 结束
时的最终提交不会重复提交该前缀）。

### 2.4 输出

每个 dump 产出两份：

- **二进制**：`/tmp/tp5-hc-dump-site{K}-{tensor}.bin`，K 为 site 序号
  （graph 顺序，1 起），tensor ∈ {normalized, lo, mixed}。内容为该张量
  完整 F32 载荷（行主序，与 `ggml_nbytes(tensor)` 字节数一致），供逐值对拍。
- **stderr 文本**：`[hc-dump] site K tensor T layer <norm->name> ne=[…]
  floats N -> <path> first64: v0 v1 … v63`，前 64 个 F32 值供人读。

### 2.5 跳过条件（如实）

- `ctx->replay_recording` 或 `ctx->tp5_recording_program` 活跃时跳过
  （打一条 stderr 提示）：tape 的 `hc_norm_end`/`hc_down_end`/`hc_up_end`
  是单一命令流的字节偏移注记，录制模式下中途切分 CB 没有执行先例。
  **对拍运行必须走非录制冷执行路径**（不启用 command replay）。
- 非 F32 / 空 / 非整浮点字节的张量跳过（对拍契约只覆盖三个 F32 中间量）。

## 3. 两个二进制的构建配置

| 臂 | 树 | 说明 |
|---|---|---|
| fused 版 | 当前 verify-rebuild 工作树现状 | 含 upfoldFix（aligned 折叠分支 + dispatch_rows BF16 项）与 FORCE_ITER 开关的 `ggml-vulkan.cpp`；本插桩对两臂同样生效 |
| 入口还原版 | 同树 + 3a 的 14 行入口还原 diff | 3a 已证其输出与 baseline 逐字一致，作为正确参照 |

两臂**同一份源码树**分别构建、各自冻结二进制与 DSO 哈希
（AGENTS.md §4.3 流程）；唯一差异是 14 行入口还原 diff + 各自的构建目录。
运行时除 `GGML_VK_HC_DUMP_SITES=1` 外其余 env/CLI 完全一致，且不启用
command replay（见 §2.5）。

## 4. 对拍判定表

对拍方式：同一固定 token 回放（teacher-forced），逐文件
`cmp`（或逐 F32 比较，浮点容差 0 —— 两臂走不同解码路径时允许
ULP 级差异，先看量级再定容差；**乱码级差异不需要容差讨论**）。

| 首个偏差张量 | 根因环节 | 对应源码位置 |
|---|---|---|
| `normalized` 偏 | norm/SUM 链（HC segment norm 或 inject-norm 融合核） | `vulkan-shaders/tp5_hc_segment_norm.comp`、`tp5_hc_inject_norm.comp` 及其 BF16 变体宏集 |
| `normalized` 对、`lo` 偏 | down 端：`hc_down_silu` 的 BF16 权重解码/收缩 | `vulkan-shaders/qwen4_hc_down_silu_bf16`（mul_mat_vec.comp DATA_A_BF16 路径）、dequant_funcs.glsl BF16 分支 |
| `lo` 对、`mixed` 偏 | up/fold 端：`hc_project_fold` 的 BF16 折叠/几何/epilogue | `vulkan-shaders/qwen4_hc_project_fold_bf16`、mul_mat_vec.comp 的 HC_UP_FOLD arm 与 reduce_result fold epilogue |
| 三个全对但最终输出乱 | 缺陷在 fused HC 段之外（本工具的排除证据） | 回到 V-B 第 1 步开关二分的判读 |

注意：入口还原版的 `lo`/`mixed` 与 fused 版走**不同 kernel 路径**
（iter() vs iter_aligned_nonquant()），逐位一致不是预期——预期是
「数量级一致的浮点值」vs「乱码级偏离」（NaN、1e±30、错位模式）。
判定以量级和模式为主，ULP 级差异记录但不判红。

## 5. 预期输出形态

```
[hc-dump] site 1 tensor normalized layer <name> ne=[5120,1,1,1] floats 5120 -> /tmp/tp5-hc-dump-site1-normalized.bin first64: 0.123 -0.045 …
[hc-dump] site 1 tensor lo layer <name> ne=[512,1,1,1] floats 512 -> /tmp/tp5-hc-dump-site1-lo.bin first64: 0.301 0.117 …
[hc-dump] site 1 tensor mixed layer <name> ne=[1280,1,1,1] floats 1280 -> /tmp/tp5-hc-dump-site1-mixed.bin first64: …
```

（ne 数值为示意；以实际 Q2_0 target 的 HC site 形状为准。）

## 6. 退化论证（env 未设 = 零行为变化）

- env 解析为 `static`，未设时 `hc_dump_sites_max == 0`；
- `hc_dump_site_count` 只在 `hc_dump_sites_max > 0` 时递增（ctx 字段保持 0）；
- 三个 dump 调用点全部形如
  `if (hc_dump_requested && ctx->hc_dump_site_count <= hc_dump_sites_max)`
  ——`hc_dump_requested` 已含 `hc_dump_sites_max > 0`，未设时短路，
  无屏障、无 staging 拷贝、无 mid-graph 提交、无文件 IO、无 stderr 输出；
- 计算路径零改动：插桩只追加 dump 调用，不触碰任何 dispatch、
  barrier、push constant、descriptor 组装；
- `ggml-vulkan-collective.cpp` 未动。

可执行断言：`tests/test-tp5-upfold-indexing.py::HcSiteDumpSentinelTests`
（源码哨兵：env 门控存在且默认 0、三个调用点同守卫、读回时序
sync_buffers→copyBuffer→synchronize→ctx_begin 有序、dump 点跟随
各自 dispatch、文档存在）。

## 7. 已知限制（如实）

- dump 在 replay/program 录制模式下跳过（见 §2.5）；对拍须用冷执行路径。
  这意味着 dump 无法观察 warm replay 帧的中间值——对拍设计（冷路径
  teacher-forced）不受影响。
- mid-graph synchronize 会把一次 graph 录制切成多段提交：**开关开启时
  的执行时间不可用于任何性能口径**（AGENTS.md §3：插桩成绩只进诊断账）。
- `sync_staging` 与既有读回路径共享；dump 张量较大时（normalized
  4×width×rows×4B）staging 会按需增长，属一次性分配。
- 本插桩未编译、未运行；SPIRV/二进制层面的验证归 DevOps
  （AGENTS.md：Engineer 不执行命令）。
