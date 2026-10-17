# r14/r15 裁决（GAP 1 覆盖者定位轮 + 伪影判别轮）

运行：arms/sitedump-fused-r14/（r14：三修正插桩）与 arms/sitedump-fused-r15/（+SYNCCMP 判别插桩）。
bin：r14=3261fc45…、r15=ba34cba1…（identity/quarantine-watchtrace2/bin-identity.txt）。
两轮 teardown 均 SIGABRT（-6）——插桩树已知形态；dump 采集不受阻碍。

## r14 结果（AMBIG2 形态）

- 三探针 bisect（P1=output_hc 40960 被清区、P2=blk.47.hc_ffn 61012480 被清区、P3=blk.0.hc_attn 100597760 幸存对照）：600 次批次边界全部 p1=1 p2=1 p3=1，数据恒定，无 1→0 翻转、无 MISS、无 WARN。
- W2DA 家族（self 写跳过 + 图启动时间门修正后）：零条——**加载期 fan-out 全被 self-skip，T2 后零 host 写入——W2DA 盲区终局闭合，该家族排除**。
- DISPATCH-HIT：200 条（通过 64MiB 过滤）——全部 rms_norm_mul_rope f32/f16 的 ~5.9MB binding（binding_off≈1046708736）与 blk.15 权重区间（q_proj/k_proj/ffn_gate_inp）的真实相交；8 条 OVERSIZED-SKIP（512MB 噪音被过滤）。**该 binding 区间不覆盖 P1/P2/P3 探针区间——与 bisect 无翻转自洽**。
- MEMSET/SUBBUFFER/BC 家族：零条。cache-invalidate-all：0 次——候选 (2) 维持排除。
- 基线形态复现：dump weightsA 0/64 全 rank、lo/mixed 0、normalized 10240/10240。

## r15 判别（SYNCCMP——同一函数内同步读 vs deferred 读对照）

在 weights dump 函数内、同一 (src.buffer, src.offset) 上追加 bisect 同款同步读（ggml_vk_buffer_read：submit+fence+立即 memcpy）：

- **SYNCCMP 全部 20 行（2 site × 5 rank × down/up）sync_nonzero=1**：
  - site1 down（blk.47.hc_ffn）first2=3ba53d76 bcaabc41——与 bisect P2 的 f2 **逐字节一致**
  - site2 down（output_hc）first2=39b63b42 3d003c3b——与 bisect P1 的 f1 **逐字节一致**
  - up 两 site 亦非零（bd173d75…/bca5bba9…）
- 同一函数、同一位置、同一轮运行：**deferred 路径（原 dump）读零（raw16 全 0）**。

## 终局裁决：GAP 1 的「W 全零」是 dump 插桩伪影，不是设备状态

**W 在设备上从未为零**——四重独立证据全部非零且数据恒定：r8 readback（1920 对一致）、r12 fence（写入时刻）、r13/r14 bisect（600 次批次边界）、r15 SYNCCMP（dump 时刻同位置同步读）。伪影机理：weights 版 dump 的 `deferred_memcpy` 挂 `subctx->out_memcpys`，而 `ggml_vk_synchronize` 只冲刷 `in_memcpys`（实现行 11566 前的循环只迭代 in_memcpys）；out_memcpys 的落地依赖 submit 后 fence 路径，dump 随即 `ggml_vk_ctx_begin` 重开录制——host vector 保持零初始化。

**连锁重估（r2-r14 的全部「覆盖/fork」叙事）**：
- 覆盖者不存在（bisect 无翻转 + W2DA T2 后零 + invalidate 0 + memset/subbuffer/BC 零 + DISPATCH-HIT 不覆盖探针区间）。
- fork 不存在（r13 四元组对照 + r15 同位置同步读非零）。
- r8「384 张量全局清零」、r9-r13「T1/T2 非零→T3 全零覆盖窗口 T2→T3」、r12「写后覆盖」裁决——**全部是 dump deferred 伪影的连锁误读**（T1/T2 的 readback 走 tensor_get 同步路径（真实非零），T3 dump 走 deferred 路径（伪影零）——「窗口」不存在）。
- **lo/mixed 的 dump 读数同样不可信**（tensor 版 dump 用同一 deferred 机制；normalized 非零的机理存疑——sync_staging 残留候选，未验证）。
- r13 verdict 的「选择性覆盖」发现（blk.0 幸存 vs output/blk.47 被清）——**伪影**：三个位置设备上全部非零。

## 乱码根因的重新定向

GAP 1（权重上传链/覆盖/fork）作为乱码根因的假说链**全部证伪**。乱码（BF16 变体路径 HTTP 500）的真实根因仍在 **BF16 kernel 数值缺陷**（未定位）——r10 全新构建（upfold 修复后）仍乱。下一轮方向：**用同步读（SYNCCMP 同款）重做单 site 中间值对拍**（fused vs 参照版），拿到可信的 normalized/lo/mixed 数值后再定位真正的分歧环节。

## 遗留

- dump 插桩修复（deferred→同步读）未实施——属诊断工具修正，留给下一轮对拍 charge 一并做（一行改动：weights/tensor 两版 dump 的读取改用 ggml_vk_buffer_read）。
- 本轮不实施产品修复：无覆盖者可修，乱码根因未定位。
