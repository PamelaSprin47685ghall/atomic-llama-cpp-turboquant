# r17 验证轮 verdict（identity/quarantine-r17/）

bin：fused/segmentsync 共用 9b8890afd16a9e17…（r17 树 = r16 树 + ensure 补回 + segment-sync 开关；注释措辞修正不改变二进制）。
CPU：upfold 39 项 OK（修两处 r17 注释里的 `ctx->sync_staging` 字面量触发文本级判别——行为零变化）、gap-tools 46 项 OK。

## ensure 修复确证实验（arms/sitedump-fused-r17/）——假设证伪

r16 fused 同配置重跑：**崩溃仍在**——`ggml-backend.cpp:349 GGML_ASSERT(offset + size <= ggml_nbytes(tensor)) "tensor write out of bounds"`（行 6779），崩溃链与 r16 完全一致：[meta-chain-miss] stage=0 nodes=165 → [CRITICAL-FALLBACK] Graph fallback to 480 subgraphs ×7 → router 路径越界断言。请求从未发出（result 全 NOT_COLLECTED）。
dump 读数可信性保持：50 个 dump、SYNCCMP 20 行全部非零且与 bisect 一致、lo 读数与 r16 逐字节一致（0.132684/-0.163211/0.441517…）。
**结论：ensure 副作用假设证伪**（ensure 补回未消除崩溃）。崩溃的更强嫌疑：dump 新版同步读在录制中间破坏命令缓冲状态导致 meta 链校验失败——但见下条，崩溃与 dump 调用本身无关。

## barrier 判别实验（arms/segmentsync-off-r17/）——实验无效 + 假设证据基础崩塌

off 臂默认配置（无 dump/watch/bisect，仅 GGML_VK_HC_SEGMENT_SYNC=1）：**同样崩溃于同一断言**（GGML_ASSERT 1 次），且 **[hc-seg-sync] 零触发**——开关在 replay 模式（默认开，本轮未设 CMD_REPLAY=0）下按设计跳过。本轮实际测的是「r17 树 + 默认 replay + 无任何诊断开关」——**崩溃在无 dump 无 sync 的路径上也发生**——崩溃与 dump 调用、segment-sync 触发均无关，与 r16/r17 树状态本身相关。最强嫌疑：r16 轮在同一 build 目录的「入口还原→恢复」交替编译留下的 SPIRV/shader 新鲜度伪影（与 r10 轮曾怀疑的 shaders-gen 增量重生成问题同类）——需全新构建复验。

**同时修正 r16 verdict 的推理错误**：r13-r15（dump 旧版 deferred，含完整 mid-graph synchronize 三元配对）**乱码 HTTP 500 仍在**——「mid-graph sync 存在时执行正确」的观察从未成立（r16 的「无 500」是请求根本未发出的误读，已在 r16 verdict 修正）。因此 **barrier/提交粒度假设已被 r13-r15 的既有数据证伪**：有 mid-graph sync（r13-r15）与无（r10）都乱。无需第三次运行判别。

## 本轮两个问题的最终答案

1. ensure 副作用假设：**证伪**（补回 ensure 崩溃不变）。
2. barrier/提交粒度假设：**证伪**（r13-r15 的 dump 旧版自带 mid-graph synchronize 且乱码仍在；r16「无乱码」系误读）。

## 崩溃线（新故障，独立于乱码线）

r16/r17 树四次运行（fused/ref/segmentsync）全部崩溃于同一链：meta-chain-miss → 480 子图 fallback ×7 → router tensor write 越界。r13-r15 树六次运行零崩溃。分界 = r16 dump 修复的树状态 + r16 轮交替编译。**下一步（交 Manager）**：全新构建（删 build 目录全量重编，消除 SPIRV 新鲜度）后复验——若全新构建不崩，崩溃=交替编译伪影；若仍崩，崩溃=r16 dump 修复代码的静态存在引发的路径变化（需 diff 审查）。

## 乱码根因现状

- GAP 1（权重缺失/覆盖/fork）：已终局证伪（r15/r16 真值）。
- barrier/提交粒度：本轮证伪。
- 数值缺陷（BF16 变体 kernel）：**回到唯一存留候选**——对拍路线仍需先解决参照臂在 rebuild 树上的崩溃（同崩溃线）。
- 诊断工具现状：dump 新版读数可信但会破坏图执行（崩溃线）；对拍需要「读数不破坏执行」的 dump 设计（如请求完成后离线读，或 dump 移到非录制路径）。
