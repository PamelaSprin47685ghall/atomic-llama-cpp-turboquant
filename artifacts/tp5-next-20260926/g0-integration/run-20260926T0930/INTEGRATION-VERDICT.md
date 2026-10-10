# 整合版（9b3c33881 vs 16e28b8ac control）资格裁决

日期：2026-09-26。run-id：20260926T0930。

## 构建与行为门

- 整合 worktree `/var/tmp/tp5-integ-9b3c33` @ `9b3c33881` + 1 行 `<cstddef>` 修复
  （与基座相同的预存干净构建缺陷；A/B 两侧同构）。
- 干净构建通过（0 errors，server `bcbdff1633b0dba8…`，libggml-vulkan `800d85235bea4e21…`）。
- CPU 套件 14/14 + GPU 定向全过（skinny/argmax/readback `--vulkan`/mesh/GDN multistep）。
- canonical smoke：**83.822 tok/s / 2040.030 ms**，全文精确、171/171、自然 stop、
  路由验证通过（RELAY/F32、linear lowering、MMVQ disabled）。

## 正式五块 ABBA（A=control 16e28b8ac `7bc567b4…`，B=整合 `bcbdff16…`，20 trial）

| 项 | A control | B 整合 |
|---|---:|---:|
| per-block | 83.69/82.88/83.85/83.24/82.67 | 83.11/82.56/82.80/83.52/82.71 |
| mean | **83.267**（sd 0.504） | **82.940**（sd 0.382） |
| B−A | **−0.327 tok/s**，95% CI [−0.973, +0.319]（跨零） | |
| ratio | **0.9961**，95% CI [**0.9884**, 1.0038] | |
| 非劣性下界 ≥0.99 | — | **未达标（0.9884 < 0.99）** |
| client wall ratio | 0.9943（同向） | |

20 trial 全部逐字精确、prompt31、171/171、自然 stop；GPU 身份前后一致。

## 裁决

**整合版资格不通过。** 按手册 §4.1 第 5 步与 §10.3：

1. 非劣性门未过（下界 0.9884 < 0.99），且点估计为负（−0.327 tok/s，−0.39%）。
   虽然差异在噪声内（CI 跨零），但既定的 0.99 非劣界是晋级硬门——不放宽。
2. `9b3c33881`（含 `9e4926382` 的 RERoT parallel-aspect thinking wire、multi-lane
   decode fixes、Vulkan matvec reuse + `5742c9c59` merge）**不得晋升为基座**，
   不得 push 到 master 作为已验证状态。
3. 失败隔离方向（§4.1 第 5 步）：整合 delta 含共享 shader（`mul_mat_vec.comp` 多行
   复用——K1a 已单独验证为非劣 0.9932 下界通过）与 server/RERoT 改动
   （`server-context.cpp` +562 行、`server-rerot*.cpp` 多文件、`common/arg.cpp` 等）。
   K1a 单因素通过而整合版整体非劣失败，指向 **非-shader 的 server/RERoT 部分**为
   回退来源；下一轮若要重启整合资格，应按 §4.1 从 `16e28b8ac` 起逐块叠加 server
   改动做单因素定位，而不是重测整包。
4. 基座维持 `16e28b8ac`（tag `tp5-fused-83-20260926`）。本裁决更新战报 §4 的
   "整合版资格待验证"为"**已测，不通过**"。

## 证据

- `integ-fused-default-smoke.json`（行为门）
- `integ-abba5.json`（5 blocks/20 trials，status=success，含全部 DSO 哈希）
- `integ-abba5_logs/`（20 个 server log）
- 构建日志 `integ-build.log`；worktree diff 仅 1 行 cstddef（与 control 同构）
