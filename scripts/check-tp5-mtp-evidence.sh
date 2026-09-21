#!/usr/bin/env bash
# ==============================================================================
# check-tp5-mtp-evidence.sh
#
# 自动化检查 MTP 单最大定义闭环与协议修正的受控运行证据面。
# 严格对照 (a) 到 (f) 六大维度进行自动化解析与判定，并覆盖提案
# docs/TP5-MTP-VERIFICATION-PROPOSAL.md 第四节的全部可脚本判定项。
#
# 调用方式（无需执行位，二选一）：
#   bash scripts/check-tp5-mtp-evidence.sh <run_log_path>
#   # 或 DevOps 部署时先 chmod +x，再 ./scripts/check-tp5-mtp-evidence.sh <log>
#
# 判定总表（与 docs/TP5-MTP-EVIDENCE.md 同步）：
#   A  MTP 图定义复用（type=3）        —— 脚本硬门禁
#   E  Hidden generation 零错配        —— 脚本硬门禁
#   F  Cycle 时间恒等式/Token 守恒      —— 脚本硬门禁（含缺失 cycle 即 FAIL）
#   M  SUBMIT_EPOCH_CHAIN hits/FAILED  —— 脚本硬门禁
#   N  numerical-mode 存在性与模式      —— 脚本硬门禁（存在性；模式值人工复核）
#   L  sidecar/spin 超时与失败签名      —— 脚本硬门禁（失败签名零容忍；数值分布仅报告）
#   D  DeviceLost/GPUVM/fatal 签名      —— 脚本硬门禁（零容忍）
#   P  phase 翻转/release 直接计数      —— 无日志依据，脚本不判，人工观察项（需新增日志，见文档）
#   S  status[2] 逐 rank 逐 bank 为零   —— 无逐点心跳日志，脚本仅查失败签名，人工观察项
# ==============================================================================

set -euo pipefail

LOG_FILE="${1:-}"

if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
    echo "Usage: bash scripts/check-tp5-mtp-evidence.sh <path_to_run_log>"
    exit 2
fi

echo "======================================================================"
echo "          TP5 MTP 受控运行证据面分析与判定"
echo "======================================================================"
echo "目标日志: $LOG_FILE"
echo

FAILURES=0

# ------------------------------------------------------------------------------
# (A) MTP 图定义复用（type=3 = LLM_GRAPH_TYPE_DECODER_MTP）
# 注意：type=2 是 DECODER（target 主干），不是 MTP。MTP 门禁只看 type=3。
# ------------------------------------------------------------------------------
echo "[检查项 A] 图定义重建与复用率 (type=3: MTP 解码图)"

MTP_REBUILDS=$(grep -c '\[tp5-mtp-graph\] type=3 reuse=0' "$LOG_FILE" || true)
MTP_REUSES=$(grep -c '\[tp5-mtp-graph\] type=3 reuse=1' "$LOG_FILE" || true)
MTP_UIDS=$(grep '\[tp5-mtp-graph\] type=3' "$LOG_FILE" | sed -n 's/.*definition_uid=\(0x[0-9a-fA-F]*\).*/\1/p' | sort -u | tr '\n' ' ' || true)
MTP_UID_COUNT=$(grep '\[tp5-mtp-graph\] type=3' "$LOG_FILE" | sed -n 's/.*definition_uid=\(0x[0-9a-fA-F]*\).*/\1/p' | sort -u | grep -c . || true)
TGT_REBUILDS=$(grep -c '\[tp5-mtp-graph\] type=0 reuse=0' "$LOG_FILE" || true)
TGT_REUSES=$(grep -c '\[tp5-mtp-graph\] type=0 reuse=1' "$LOG_FILE" || true)

echo "  - MTP 图重建次数 (type=3 reuse=0): $MTP_REBUILDS"
echo "  - MTP 图复用次数 (type=3 reuse=1): $MTP_REUSES"
echo "  - 观察到的 MTP definition_uid: ${MTP_UIDS:-<none>} (distinct=${MTP_UID_COUNT})"
echo "  - 参考：Target 主干 type=0 重建/复用: ${TGT_REBUILDS}/${TGT_REUSES}（prefill→decode 期望一次重建，不进门禁）"

if [[ "$MTP_REBUILDS" -eq 0 && "$MTP_REUSES" -eq 0 ]]; then
    echo "  -> [FAIL] 未捕获到任何 type=3 MTP 图执行记录（PROFILE 开关下必须存在）。"
    FAILURES=$((FAILURES + 1))
elif [[ "$MTP_REBUILDS" -gt 1 ]]; then
    echo "  -> [FAIL] 发现 MTP 图多次重建 ($MTP_REBUILDS 次)，违反单最大定义闭环！"
    FAILURES=$((FAILURES + 1))
elif [[ "$MTP_REUSES" -eq 0 ]]; then
    echo "  -> [FAIL] MTP 图零复用（只有构建、无复用命中），复用管线未生效。"
    FAILURES=$((FAILURES + 1))
elif [[ "$MTP_UID_COUNT" -ne 1 ]]; then
    echo "  -> [FAIL] MTP definition_uid 去重后数量为 $MTP_UID_COUNT（期望恰好 1 个恒定 UID）。"
    FAILURES=$((FAILURES + 1))
else
    echo "  -> [PASS] MTP 图实现单最大定义闭环，无反复重建与命令重录。"
fi
echo

# ------------------------------------------------------------------------------
# (E) Hidden RESULT / CARRY / SEED generation 匹配
# ------------------------------------------------------------------------------
echo "[检查项 E] Hidden Generation 匹配校验"

GEN_MISMATCH_COUNT=$(grep -c '\[tp5-mtp-hidden\].*mismatch' "$LOG_FILE" || true)

if [[ "$GEN_MISMATCH_COUNT" -eq 0 ]]; then
    echo "  - Generation 不匹配事件: 0"
    echo "  -> [PASS] 设备 Hidden 在 RESULT/CARRY/SEED 跨步交接中代数严格吻合。"
else
    echo "  - Generation 不匹配事件: $GEN_MISMATCH_COUNT"
    grep '\[tp5-mtp-hidden\].*mismatch' "$LOG_FILE" | head -n 5 || true
    echo "  -> [FAIL] 发现 $GEN_MISMATCH_COUNT 次 Generation 错配，存在读过期数据风险！"
    FAILURES=$((FAILURES + 1))
fi
echo

# ------------------------------------------------------------------------------
# (F) Cycle 账本：时间恒等式 / Token 守恒 / target_us>0 / dev_hidden==1
# 缺失 cycle 行在 PROFILE=1 受控运行下直接 FAIL（不再 WARN 放行）。
# ------------------------------------------------------------------------------
echo "[检查项 F] 投机周期账本恒等式与守恒校验"

# 注：不用行首锚点（server --log-file 可能加时间戳前缀）；用 ' cycle=' 后缀排除 -summary 行。
CYCLE_COUNT=$(grep -c '\[tp5-mtp-cycle\] cycle=' "$LOG_FILE" || true)

if [[ "$CYCLE_COUNT" -eq 0 ]]; then
    echo "  -> [FAIL] 未找到任何 [tp5-mtp-cycle] 记录（GGML_TP5_MTP_PROFILE=1 受控运行必须输出账本）。"
    FAILURES=$((FAILURES + 1))
else
    echo "  - 记录到的投机周期数: $CYCLE_COUNT"
    F_BAD_IDENTITY=0
    F_BAD_TOKEN=0
    F_BAD_TARGET=0
    F_BAD_HIDDEN=0
    F_BAD_ACCEPT=0
    F_BAD_SEQ=0
    ZERO_ACCEPT_COUNT=0
    EXPECTED_CYCLE=1
    while IFS= read -r line; do
        cycle=$(echo "$line" | sed -n 's/.*cycle=\([0-9][0-9]*\).*/\1/p')
        draft_us=$(echo "$line" | sed -n 's/.*draft_us=\([0-9][0-9]*\).*/\1/p')
        target_us=$(echo "$line" | sed -n 's/.*target_us=\([0-9][0-9]*\).*/\1/p')
        catchup_us=$(echo "$line" | sed -n 's/.*catchup_us=\([0-9][0-9]*\).*/\1/p')
        handoff_us=$(echo "$line" | sed -n 's/.*handoff_us=\([0-9][0-9]*\).*/\1/p')
        total_us=$(echo "$line" | sed -n 's/.*total_us=\([0-9][0-9]*\).*/\1/p')
        draft_tk=$(echo "$line" | sed -n 's/.*draft_tokens=\([0-9][0-9]*\).*/\1/p')
        accepted_tk=$(echo "$line" | sed -n 's/.*accepted_tokens=\([0-9][0-9]*\).*/\1/p')
        final_tk=$(echo "$line" | sed -n 's/.*final_tokens=\([0-9][0-9]*\).*/\1/p')
        dev_hidden=$(echo "$line" | sed -n 's/.*dev_hidden=\([0-9][0-9]*\).*/\1/p')
        : "${cycle:=0}" "${draft_us:=0}" "${target_us:=0}" "${catchup_us:=0}" "${handoff_us:=0}" "${total_us:=0}"
        : "${draft_tk:=0}" "${accepted_tk:=0}" "${final_tk:=0}" "${dev_hidden:=0}"
        if [[ "$cycle" -ne "$EXPECTED_CYCLE" ]]; then
            F_BAD_SEQ=$((F_BAD_SEQ + 1))
            echo "  ! 周期序号不连续 (期望 cycle=$EXPECTED_CYCLE, 实际 cycle=$cycle): $line"
        fi
        EXPECTED_CYCLE=$((cycle + 1))
        if [[ "$accepted_tk" -eq 0 ]]; then
            ZERO_ACCEPT_COUNT=$((ZERO_ACCEPT_COUNT + 1))
        fi
        if [[ $((draft_us + target_us + catchup_us + handoff_us)) -ne "$total_us" ]]; then
            F_BAD_IDENTITY=$((F_BAD_IDENTITY + 1))
            echo "  ! 时间恒等式背离: $line"
        fi
        if [[ $((accepted_tk + 1)) -ne "$final_tk" ]]; then
            F_BAD_TOKEN=$((F_BAD_TOKEN + 1))
            echo "  ! Token 守恒背离 (final != accepted+1): $line"
        fi
        if [[ "$target_us" -eq 0 ]]; then
            F_BAD_TARGET=$((F_BAD_TARGET + 1))
            echo "  ! target_us 为零（server-context 真实接入未生效）: $line"
        fi
        if [[ "$dev_hidden" -ne 1 ]]; then
            F_BAD_HIDDEN=$((F_BAD_HIDDEN + 1))
            echo "  ! dev_hidden != 1（存在 CPU 倒腾）: $line"
        fi
        if [[ "$accepted_tk" -gt "$draft_tk" ]]; then
            F_BAD_ACCEPT=$((F_BAD_ACCEPT + 1))
            echo "  ! accepted_tokens > draft_tokens: $line"
        fi
    done < <(grep '\[tp5-mtp-cycle\] cycle=' "$LOG_FILE" || true)

    echo "  - 零接受周期数 (accepted_tokens=0): $ZERO_ACCEPT_COUNT"
    if [[ "$ZERO_ACCEPT_COUNT" -eq 0 ]]; then
        echo "  ! 未覆盖零接受周期（受控验证须包含至少 1 次零接受周期账本以证实无跳过结算）"
        F_BAD_ZERO_COVERAGE=1
    else
        F_BAD_ZERO_COVERAGE=0
    fi

    SUMMARY_LINE=$(grep '\[tp5-mtp-cycle-summary\]' "$LOG_FILE" | tail -n 1 || true)
    if [[ -n "$SUMMARY_LINE" ]]; then
        echo "  - 账本统计汇总: $SUMMARY_LINE"
    fi
    F_TOTAL=$((F_BAD_IDENTITY + F_BAD_TOKEN + F_BAD_TARGET + F_BAD_HIDDEN + F_BAD_ACCEPT + F_BAD_SEQ + F_BAD_ZERO_COVERAGE))
    if [[ "$F_TOTAL" -eq 0 ]]; then
        echo "  -> [PASS] 全部 $CYCLE_COUNT 个周期满足时间恒等式、Token 守恒、序号连续单调递增、覆盖零接受（$ZERO_ACCEPT_COUNT 次）、target_us>0、dev_hidden==1。"
    else
        echo "  -> [FAIL] 账本违例：恒等式背离=${F_BAD_IDENTITY} 守恒背离=${F_BAD_TOKEN} target零值=${F_BAD_TARGET} 非直传=${F_BAD_HIDDEN} 超收=${F_BAD_ACCEPT} 序号不连续=${F_BAD_SEQ} 零接受缺失=${F_BAD_ZERO_COVERAGE}。"
        FAILURES=$((FAILURES + 1))
    fi
fi
echo

# ------------------------------------------------------------------------------
# (M) SUBMIT_EPOCH_CHAIN hits 连续性 / FAILED 零容忍
# ------------------------------------------------------------------------------
echo "[检查项 M] SUBMIT_EPOCH_CHAIN 命中连续性"

CHAIN_HITS=$(grep -c '\[tp5-meta\] SUBMIT_EPOCH_CHAIN' "$LOG_FILE" || true)
CHAIN_FAILED=$(grep -c '\[tp5-meta\] SUBMIT_EPOCH_CHAIN FAILED' "$LOG_FILE" || true)
CHAIN_LAST_HIT=$(grep '\[tp5-meta\] SUBMIT_EPOCH_CHAIN' "$LOG_FILE" | tail -n 1 || true)

echo "  - CHAIN 相关行数: $CHAIN_HITS（含 PREDEFINED TRUTH / SUCCESS）"
echo "  - CHAIN FAILED 行数: $CHAIN_FAILED"
if [[ -n "$CHAIN_LAST_HIT" ]]; then
    echo "  - 末条命中样本: $CHAIN_LAST_HIT"
fi
if [[ "$CHAIN_FAILED" -gt 0 ]]; then
    echo "  -> [FAIL] 出现 SUBMIT_EPOCH_CHAIN FAILED（拒绝 fallback 后的原生提交失败）！"
    FAILURES=$((FAILURES + 1))
elif [[ "$CHAIN_HITS" -eq 0 ]]; then
    echo "  -> [FAIL] 未捕获任何 SUBMIT_EPOCH_CHAIN 命中行（Vulkan TP5 受控运行必须提交命令链）。"
    FAILURES=$((FAILURES + 1))
else
    echo "  -> [PASS] 命令链命中连续，零 FAILED 回退。"
fi
echo

# ------------------------------------------------------------------------------
# (N) numerical-mode 定义期打印存在性
# ------------------------------------------------------------------------------
echo "[检查项 N] 数值模式定义期打印"

NUM_LINES=$(grep -c '\[tp5-numerical-mode\]' "$LOG_FILE" || true)
if [[ "$NUM_LINES" -eq 0 ]]; then
    echo "  -> [FAIL] 未找到 [tp5-numerical-mode] 定义期打印（TP5 collective 初始化必须单次判定模式）。"
    FAILURES=$((FAILURES + 1))
else
    echo "  - 打印次数: $NUM_LINES"
    grep '\[tp5-numerical-mode\]' "$LOG_FILE" | head -n 3 || true
    NUM_MODE=$(grep '\[tp5-numerical-mode\]' "$LOG_FILE" | head -n 1 | sed -n 's/.*mode=\([^ ]*\).*/\1/p' || true)
    echo "  - 首个模式: ${NUM_MODE:-<unknown>}（本提案 timeline+f16 配置期望 mode=reference；若为其他值请人工复核配置一致性）"
    if [[ "$NUM_MODE" != "reference" ]]; then
        echo "  -> [FAIL] 数值模式非 reference，与本提案 timeline+f16 无 LateBind 配置不符。"
        FAILURES=$((FAILURES + 1))
    else
        echo "  -> [PASS] 数值模式打印存在且与配置一致。"
    fi
fi
echo

# ------------------------------------------------------------------------------
# (L) sidecar/spin 超时与失败签名（零容忍）；数值分布仅报告
# ------------------------------------------------------------------------------
echo "[检查项 L] Sidecar/自旋失败签名扫描"

L_FAIL=0
for pat in 'RELAY LateBind sidecar timeout' 'RELAY LateBind sidecar exceeds workspace' 'LateBind sidecar is not 128-bit copy aligned' 'RELAY pre-armed bank is not idle' 'RELAY mailbox status not idle' 'RELAY bank generation differs'; do
    n=$(grep -c "$pat" "$LOG_FILE" || true)
    if [[ "$n" -gt 0 ]]; then
        echo "  ! 命中失败签名 [$pat] x$n"
        grep "$pat" "$LOG_FILE" | head -n 3 || true
        L_FAIL=$((L_FAIL + n))
    fi
done
LB_STAGE_COUNT=$(grep -c '\[tp5-latebind-stage\]' "$LOG_FILE" || true)
LB_PROF_COUNT=$(grep -c '\[tp5-latebind-profile\]' "$LOG_FILE" || true)
echo "  - [tp5-latebind-stage] 行数: $LB_STAGE_COUNT（本提案 timeline 配置下期望为 0 行，有行则逐行人工复核）"
echo "  - [tp5-latebind-profile] 行数: $LB_PROF_COUNT（同上）"
if [[ "$LB_STAGE_COUNT" -gt 0 ]]; then
    echo "  - 首条 stage 样本: $(grep '\[tp5-latebind-stage\]' "$LOG_FILE" | head -n 1 || true)"
fi
if [[ "$L_FAIL" -gt 0 ]]; then
    echo "  -> [FAIL] 共 $L_FAIL 条 sidecar/自旋/邮箱失败签名！"
    FAILURES=$((FAILURES + 1))
else
    echo "  -> [PASS] 无 sidecar 超时、无邮箱非空闲、无 bank 世代分歧签名。"
fi
echo

# ------------------------------------------------------------------------------
# (D) DeviceLost / GPUVM / fatal 签名（零容忍）
# ------------------------------------------------------------------------------
echo "[检查项 D] 设备与致命错误签名扫描"

D_FAIL=0
for pat in 'ErrorDeviceLost' 'VK_ERROR_DEVICE_LOST' 'GPUVM fault' 'dma_fence_wait_timeout' 'VM page fault' 'graph definition ID space exhausted' 'failed to allocate graph' 'failed to initialize graph'; do
    n=$(grep -c "$pat" "$LOG_FILE" || true)
    if [[ "$n" -gt 0 ]]; then
        echo "  ! 命中致命签名 [$pat] x$n"
        grep "$pat" "$LOG_FILE" | head -n 3 || true
        D_FAIL=$((D_FAIL + n))
    fi
done
if [[ "$D_FAIL" -gt 0 ]]; then
    echo "  -> [FAIL] 共 $D_FAIL 条设备/致命签名！"
    FAILURES=$((FAILURES + 1))
else
    echo "  -> [PASS] 无 DeviceLost、无 GPUVM fault、无图分配致命错误。"
fi
echo

# ------------------------------------------------------------------------------
# 总体判定
# ------------------------------------------------------------------------------
echo "======================================================================"
echo "人工观察项（脚本不判，需真机值守确认，详见 docs/TP5-MTP-EVIDENCE.md）："
echo "  P1 phase 翻转/release 直接计数 —— 需新增 [tp5-mtp-phase] 日志（已报需求，未写代码）"
echo "  P2 status[2] 逐 rank 逐 bank 为零 —— 需新增心跳日志（已报需求，未写代码）"
echo "  P3 handoff/catchup 均值预算 —— 参考阈值，dmesg fence/GPUVM 复核"
echo "======================================================================"
if [[ "$FAILURES" -eq 0 ]]; then
    echo "结论: [PASS] 全部脚本可判定项符合单最大定义与协议阈值！"
    exit 0
else
    echo "结论: [FAIL] 共 $FAILURES 个检查项未达标，详见上方 [FAIL] 输出！"
    exit 1
fi
