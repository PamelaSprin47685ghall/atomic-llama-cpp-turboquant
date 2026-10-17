# r19 崩溃起点定位 verdict（identity/quarantine-crash-origin-r19/）——纯读判别

## 判读表（全部 sitedump 臂 + 基准，server.log 断言/时序/末行）

| 臂 | assert(ggml-backend.cpp:349) | moe_route | 末行形态 |
|---|---|---|---|
| latebind-parity-off-rebuild（基准，registry 引入前） | **0** | 0 | 正常 cleaning up before exit |
| sitedump-fused（首轮） | 0 | 5 | 正常退出 |
| sitedump-ref | 0 | 5 | 正常退出 |
| r2-r9（九轮） | **全部 0** | 5 | 全部正常 cleaning up before exit |
| **r10** | **1（首次）** | 5 | 断言为末行 |
| r11 | 1 | 5 | 断言为末行 |
| r12 | 1 | 5 | 断言为末行 |
| r13/r14/r15 | 1 | 5 | 断言为末行 |
| r16/r17/r18b | 1 | 5 | 断言为末行 |

r10 崩溃上下文与 r16+ 逐行同构：[meta-chain-miss]（r10 nodes=76，r16+ nodes=165——树状态差异，断言同点）→ [CRITICAL-FALLBACK] 480 subgraphs cnt=7 → router 路径 `ggml-backend.cpp:349 GGML_ASSERT(offset + size <= ggml_nbytes(tensor)) "tensor write out of bounds"`。

## 裁决

1. **崩溃引入点 = r10，精确分界**：r9 树不崩（正常退出）→ r10 树崩。r18 的「r16 引入」判断被本轮推翻——崩溃从 r10 起持续存在（r10-r18 全部九轮）。
2. **registry 嫌疑强化到最强**：r10 = forkfix 轮 = s_gap1_weight_simple_registry（meta.cpp，fallback 路径的 ggml_backend_meta_buffer_simple_tensor 优先返回注册指针）的引入轮——**崩溃引入与 registry 引入精确同步**。基准臂（registry 引入前的树）请求能完成（乱码 500 但进程活着正常退出）——「registry 引入前请求能完成」的对照成立。
3. **修正此前误判**：r13-r15 的「进程活着出乱码」判断错误——它们的 server.log 同样以断言结尾（乱码 HTTP 500 与崩溃共存：请求发出得到乱码 500，进程随后崩溃）。r10-r15（dump 旧版）= 500+崩溃；r16+（dump 新版）= 崩溃且请求未发出（崩溃时点提前）。dump 新旧只影响崩溃时点与 500 有无，不影响崩溃本身。
4. **r18 哈希一致性的重新解读**：增量=全新（无编译伪影）的结论维持——崩溃是 r10 树起的二进制确定性行为，与构建方式无关。

## 下一步指向（判别实验 2：registry 门控验证）

在隔离树上给 s_gap1_weight_simple_registry 加 env 门控（或最小注释禁用）→ 重编 → 跑同配置 sitedump 臂：崩溃消失则 registry 因果坐实（fallback 路径的 simple_tensor 注册指针返回导致 router 越界写）；崩溃仍在则 r10 树的其他改动（r10 轮的 forkfix 修复本体）继续排查。r10 轮的 diff 存档在 identity/quarantine-forkfix/（若存在）——门控实验前先 diff 复核 r9→r10 的全部树改动，确保嫌疑清单完整。
