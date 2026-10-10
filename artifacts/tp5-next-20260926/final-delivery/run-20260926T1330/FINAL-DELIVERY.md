# 最终交付：完美版（TP5 fused 83 + RERoT 可用性）

日期：2026-09-26。run-id：20260926T1330。按用户指令收尾：
**"最终没有办法的时候，交付一个完美版然后收工"**——356 ms 缺口经全部候选
核验后无最小可验收补丁可兑现（见 K1B1-PREAISE-VERDICT），交付完美版。

## 完美版定义

基座 `16e28b8ac63c9fd5a54611070c6405f064f69a21`（tag `tp5-fused-83-20260926`，
83 tok/s 已验收融合配置）+ 三项已验收、零回退的增量：

| # | 增量 | 类别 | 验收证据 |
|---|---|---|---|
| 1 | `ggml-vulkan-tp5-rows.h` +`<cstddef>` | 干净构建修复（基座预存缺陷，`test-tp5-row-program` 在基座从未编译过） | 干净构建 0 error，test-tp5-row-program 通过 |
| 2 | `mul_mat_vec.comp` 多行权重复用（K1a，93+/28−） | 非劣 shader 成员（消除 7 列路径 6/7 重复权重读的代码债） | 五块 ABBA 非劣下界 0.9932≥0.99、20 trial 全对；全部数值对拍过 |
| 3 | RERoT 可用性修复（`speculative.cpp` 内部序列跳过 + `server-context.cpp` rerot_internal 不 drafting）+ D1 设备 token 机制（opt-in、fail-closed、默认惰性） | 功能修复 + 已归档机制 | mindmap/todo wire 完整产出、自然 stop、零断言；RERoT-off 保护线 171/171 |

**不含**：整合版 server/RERoT 大改动（非劣失败 0.9884<0.99）、任何未过 ABBA 的性能候选。

## 最终验证（本 run-id，干净重建 `/var/tmp/tp5-final-build`）

- **干净构建**：0 errors；server SHA256 `b92fda12915ab595…`，
  libggml-vulkan `fcbaf29d7406fb25…`
- **回归 29/29**：CPU 套件（sampling/arg-parser/mtp-workspace/tp5-plan/target-capacity/
  predefined-capacity/tp5-row-program/recurrent-rollback/meta-reduce-boundary）+
  RERoT 15 项 + grammar-indent + vulkan-command-replay + mesh + GDN multistep
- **GPU 定向**：skinny / argmax / readback `--vulkan` 五卡 / tp5-mesh / GDN multistep /
  rows replay 全过
- **canonical 门（RERoT-off）**：`1..60` warmup+1 均 **171/171 逐字精确、自然 stop、
  draft_n=150/accepted=149**；warm 2062.5 ms（与基座 ~83 tok/s 口径一致）
- **RERoT 门（RERoT-on）**：`--rerot --rerot-pens 4 --rerot-people 2` 下 mindmap wire
  **937 completion、自然 stop、完整 Mermaid mindmap**，零断言（按用户指令，
  开/关性能不做比较）

## 使用方式

```bash
# 默认（83 tok/s 基座行为，RERoT-off）
llama-server -m <target.gguf> -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4 \
  --split-mode tensor --fit off -ngl 999 -c 4096 -b 32 -ub 32 --no-mmap --no-host \
  --tp5 qwen4exp-af -md <mtp.gguf> --spec-type draft-mtp

# RERoT（可用；性能与不开不比较）
  … 同上 … --rerot --rerot-pens 4 --rerot-people 2
```

源码：candidate worktree `/var/tmp/tp5-candidate-k1a`（16e28b8ac + 17 文件
394+/35−，完整 diff 见 `final-version.diff`）。**未 commit/push**（手册 §10.3：
未经用户明确要求不提交）。

## 356 ms 缺口的最终状态（诚实账）

- 已排除：K1a matvec（非劣零增速）、E1（无账）、S1（低于噪声）、D1（架构互斥：
  Vulkan 同设备 copy 限制 × tensor-split 拒绝 backend sampling）、整合版（非劣失败）
- 剩余可归因：PLE 层 0–3 快照 CPY（S1 域，须过 shadow 门）+ 44 小 dispatch/层
  （算子融合新工单，上限 ~25 ms/请求）——均超出"最小可验收补丁"范畴
- **100 tok/s 未达成**（2062 ms warm vs 1710 ms 门）；完美版 = 83 tok/s 基座 +
  RERoT 可用，不含任何未验收的性能声明

## 证据

- 本目录：`final-version.diff`、`server-final-canonical.log`、`server-final-rerot.log`
- 上游归档：`artifacts/tp5-next-20260926/`（g0-baseline / g1-ledgers / k1a-matvec /
  k1b-ple-layers / d1-device-loop / g0-integration / rerot-usability，各含
  decision.md + 原始数据）
