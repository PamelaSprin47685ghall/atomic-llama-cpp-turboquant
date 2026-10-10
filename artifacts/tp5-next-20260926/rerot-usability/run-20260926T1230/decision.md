# RERoT 可用性修复 decision

日期：2026-09-26。run-id：20260926T1230。目标（用户指令）：**使 RERoT 可用；开启与不开启 RERoT 的性能不做比较。**

## 症状

TP5 栈（五卡 + MTP + `--rerot --rerot-pens 4 --rerot-people 2`）下任何 RERoT 请求
（mindmap/todo wire）在 episode start 后立即崩溃：

1. `common/speculative.cpp:1626` `GGML_ASSERT(seq_id >= 0 && seq_id < n_seq)` —
   RERoT 内部序列（probe/park/archive，id ≥ arena_base=slots.size()）进入 MTP 驱动
   `process_impl` 时越界断言。
2. 修复 1 后：`tools/server/server-context.cpp:10302`
   `GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1)` — RERoT lane 槽位被 MTP
   驱动 drafting，但其 token 行由 `rerot_append_planned_token` 构建（teacher-forced），
   永不进入 `handle_last_sampled_token`，验证循环看到非空 `spec_draft` 却没有
   `spec_i_batch` 验证行。

## 修复（两处，均在 candidate worktree，默认路径零变化）

1. **`common/speculative.cpp` process_impl**：`seq_id >= n_seq` 的行（RERoT 内部
   序列）跳过而非断言——与文档契约一致（"never drafts for rerot_internal slots"），
   这些行由 episode runtime 规划，MTP 驱动只须不为其 staging。
2. **`tools/server/server-context.cpp` draft 收集循环**：`slot.rerot_internal` 槽位
   直接 return（不进入 MTP drafting）——它们的 token 是 RERoT 规划器 teacher-forced
   产出（probe 注入、DAG frame、serial tail 均经 `rerot_plan_next_token` →
   `rerot_append_planned_token`），投机 drafting 语义不适用；普通（非 RERoT）请求
   保持完整投机路径。

## 验证（全部当前证据）

- **RERoT ON 功能**（TP5+MTP+五卡，`--rerot --rerot-pens 4 --rerot-people 2`）：
  - mindmap wire：937 completion、自然 stop、完整 Mermaid mindmap（浅草/涩谷/台场
    三日游），`has_mermaid=true`，零断言/崩溃
  - todo wire：1474 completion、自然 stop、`- [ ]` checkbox 列表完整，零断言
  - 两个 episode（episode 1、2）正常启动/结束
- **RERoT OFF 保护线**（同二进制，无 `--rerot`）：canonical `1..60` warmup+1 均
  171/171 逐字精确、自然 stop、draft_n=150/accepted=149 投机正常
- **回归**：RERoT 15 项 + sampling/mtp-workspace/tp5-plan/vulkan-command-replay/
  grammar-indent 共 20/20 通过；GPU 定向 skinny/argmax/readback `--vulkan`/mesh/
  GDN multistep 全过

## 性能口径（按用户指令，不做开/关比较）

RERoT 请求的 tok/s（mindmap ~48 t/s slot 计）**不进入**任何性能比较；本修复只声明
**功能可用性**。RERoT-off canonical 仍以既有基座口径验收。

## 证据

- `rerot-fix.diff`（两处修复的完整 diff）
- `server-v4.log`（修复后 RERoT ON：两 wire 成功、零 error）
- `server-v4-rerotoff.log`（保护线 canonical）
- `server-v3-debug.log`/`server-rerot-mtp*.log`（崩溃复现与根因链）
