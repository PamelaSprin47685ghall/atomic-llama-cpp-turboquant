# r16 可信读数对拍轮 verdict

bin：fused=6c9ee061…（r16 dump 修复版）、ref=e3ebb7a0…（+入口还原）；树已恢复 fused 状态（恢复后哈希回到 6c9ee061，diff 验证通过备份回拷）。
CPU：test-tp5-upfold-indexing 34 项 OK（含 DumpReadbackTests）、test-tp5-gap-tools 46 项 OK。

## fused 臂（arms/sitedump-fused-r16/）——首次可信读数（核心成果）

修复版 dump（同步读）在崩溃前完成 site 1/2 采集，SYNCCMP 交叉核对 20 行全部 sync_nonzero=1 且与 bisect 数据逐字节一致（raw16 15734=0x3D76/48193=0xBC41 ↔ SYNCCMP first2=3ba53d76 bcaabc41）：

| 张量 | 真值（site1 全 5 rank 一致） | 判读 |
|---|---|---|
| normalized | 非零（0.232/-0.422/1.197/1.081…合理 RMS 量级） | 与旧读数一致（旧 normalized 非零是真的） |
| **lo** | **非零（0.133/-0.163/0.442/0.437…合理收缩量级）** | **r2-r15 的「lo 全零」是 dump 伪影——down dispatch 实际在写** |
| **mixed** | **非零（0.204/-0.324/-0.476…合理）** | **「mixed 全零」同样是伪影——up/gate 段在写** |
| **W down/up** | **非零（BF16 权重位模式，与 SYNCCMP/bisect/r12 fence/r8 readback 全部一致）** | **W 从未为零——GAP 1「权重缺失」叙事终局证伪** |

**结论：GAP 1 的完整因果链（W 全零→lo/mixed 零→kernel 不写→乱码）全部是 dump deferred 伪影。设备上 W/lo/mixed/normalized 全部非零、量级合理。**

## 两臂崩溃（新执行故障，原始失败保留）

fused 臂与参照臂均在预热阶段崩溃于同一点：
`cmd_child_to_router:error:ggml-backend.cpp:349: GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds") failed`
- fused 臂：崩溃在 dump site 1/2 采集之后（「init: Prewarming TP5 decode/verify graph (1 tokens)...」后的 router 路径）——请求从未发出（result 全 NOT_COLLECTED、teardown -6）。
- 参照臂（入口还原版）：崩溃更早，dumps=0（入口还原后 HC 不走 ggml_vk_hc_segment，dump 不触发），伴随 [meta-chain-miss] stage=0 nodes=165 与 [CRITICAL-FALLBACK] Graph fallback to 480 subgraphs——与 3a 轮（upfold 树、无 r13-r16 插桩、正常运行）行为不同。
- **崩溃机理候选（高嫌疑）**：dump 修复移除了 `ggml_vk_ensure_sync_staging_buffer` 调用——旧版 dump（r13-r15）是 sync_staging 初始化的一个触发点；r16 新版不调用 ensure，预热/router 路径的其他用户在 staging 未初始化状态下使用（copy 到空 buffer → 越界断言）。r13-r15 不崩（dump 活跃时 ensure 被调用）、r10 无插桩树不崩（乱码但不崩）——与该假设自洽。
- **对拍未完成**：参照臂崩溃，数值分歧定位无法进行。按「失败不重放」保留原始失败。

## 乱码根因的重新指向（r16 数据下）

- fused 段数值（dump 触发的 mid-graph 分段提交下）非零且量级合理；r10（无插桩、大批量提交）乱码。
- **新候选：乱码与提交粒度相关**——dump 旧版/新版的 mid-graph synchronize（前缀提交+重开）存在时执行正确、无时乱——指向 **fused 段 dispatch 序列缺少 barrier/依赖声明，单大批量提交下 GPU 乱序执行**。这是同步类缺陷候选（非纯数值缺陷）。
- 验证路径（交 Manager 裁定）：在无 dump 插桩的树上加「每 HC 段后 mid-graph synchronize」的最小改动跑 off 臂——乱码消失则坐实。

## 遗留

1. dump 修复的 ensure 副作用缺陷（staging 初始化触发点丢失）——dump-fixer 补 ensure 调用或恢复初始化保证后重跑对拍。
2. 参照臂在 rebuild 树上崩溃（与 3a 轮行为不同）——插桩组合交互，未定位；对拍需在修好 ensure 后重试。
3. 本轮不实施产品修复：数值分歧定位未完成（对拍臂崩溃）；乱码根因候选（提交粒度/barrier）需独立实验验证。
