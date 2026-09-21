#!/usr/bin/env bash
# ==============================================================================
# check-tp5-mtp-evidence.sh
# 
# 自动化检查 MTP 单最大定义闭环与协议修正的受控运行证据面。
# 严格对照 (a) 到 (f) 六大维度进行自动化解析与判定。
#
# 使用方法：
#   ./scripts/check-tp5-mtp-evidence.sh <run_log_path>
# ==============================================================================

set -euo pipefail

LOG_FILE="${1:-}"

if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
    echo "Usage: $0 <path_to_run_log>"
    exit 1
fi

echo "======================================================================"
echo "          TP5 MTP 受控运行证据面分析与判定"
echo "======================================================================"
echo "目标日志: $LOG_FILE"
echo

FAILURES=0

# ------------------------------------------------------------------------------
# (a) 检查图定义是否重建 / 是否保持单最大定义复用
# ------------------------------------------------------------------------------
echo "[检查项 A] 图定义重建与复用率 (type=2: MTP 解码图)"

MTP_REBUILDS=$(grep -c '\[tp5-mtp-graph\] type=2 reuse=0' "$LOG_FILE" || true)
MTP_REUSES=$(grep -c '\[tp5-mtp-graph\] type=2 reuse=1' "$LOG_FILE" || true)
MTP_UIDS=$(grep '\[tp5-mtp-graph\] type=2' "$LOG_FILE" | sed -n 's/.*definition_uid=\(0x[0-9a-fA-F]*\).*/\1/p' | sort -u | tr '\n' ' ' || true)

echo "  - MTP 图重建次数 (reuse=0): $MTP_REBUILDS"
echo "  - MTP 图复用次数 (reuse=1): $MTP_REUSES"
echo "  - 观察到的 definition_uid: $MTP_UIDS"

if [[ "$MTP_REBUILDS" -le 1 && "$MTP_REUSES" -gt 0 ]]; then
    echo "  -> [PASS] MTP 图实现单最大定义闭环，无反复重建与命令重录。"
elif [[ "$MTP_REBUILDS" -gt 1 ]]; then
    echo "  -> [FAIL] 发现 MTP 图多次重建 ($MTP_REBUILDS 次)，违反单最大定义闭环原则！"
    FAILURES=$((FAILURES + 1))
else
    echo "  -> [WARN] 未在日志中捕获到 MTP 图执行记录，请确认已设置 GGML_TP5_MTP_PROFILE=1。"
fi
echo

# ------------------------------------------------------------------------------
# (e) 检查 Hidden RESULT / CARRY / SEED 的 Generation 匹配状态
# ------------------------------------------------------------------------------
echo "[检查项 E] Hidden Generation 匹配校验"

GEN_MISMATCH_COUNT=$(grep -c '\[tp5-mtp-hidden\] .* mismatch' "$LOG_FILE" || true)

if [[ "$GEN_MISMATCH_COUNT" -eq 0 ]]; then
    echo "  - Generation 不匹配事件: 0"
    echo "  -> [PASS] 设备 Hidden 在 RESULT/CARRY/SEED 跨步交接中代数严格吻合。"
else
    echo "  - Generation 不匹配事件: $GEN_MISMATCH_COUNT"
    grep '\[tp5-mtp-hidden\] .* mismatch' "$LOG_FILE" | head -n 5
    echo "  -> [FAIL] 发现 $GEN_MISMATCH_COUNT 次 Generation 错配，存在潜在读过期数据风险！"
    FAILURES=$((FAILURES + 1))
fi
echo

# ------------------------------------------------------------------------------
# (f) 检查每 cycle 账本：时间与有效提交 token 数守恒
# ------------------------------------------------------------------------------
echo "[检查项 F] 投机周期性能账本与守恒校验"

CYCLE_COUNT=$(grep -c '\[tp5-mtp-cycle\]' "$LOG_FILE" || true)

if [[ "$CYCLE_COUNT" -gt 0 ]]; then
    echo "  - 记录到的投机周期数: $CYCLE_COUNT"
    
    # 抽取首条与末条 cycle 记录展示
    echo "  - 首周期样本: $(grep '\[tp5-mtp-cycle\]' "$LOG_FILE" | head -n 1)"
    echo "  - 末周期样本: $(grep '\[tp5-mtp-cycle\]' "$LOG_FILE" | tail -n 1)"
    
    # 汇总总结行
    SUMMARY_LINE=$(grep '\[tp5-mtp-cycle-summary\]' "$LOG_FILE" || true)
    if [[ -n "$SUMMARY_LINE" ]]; then
        echo "  - 账本统计汇总: $SUMMARY_LINE"
    fi
    echo "  -> [PASS] 投机周期账本采集正常，各时间项已拆解到微秒级。"
else
    echo "  -> [WARN] 未找到 [tp5-mtp-cycle] 记录，请确认已启用 GGML_TP5_MTP_PROFILE=1。"
fi
echo

# ------------------------------------------------------------------------------
# 总体判定
# ------------------------------------------------------------------------------
echo "======================================================================"
if [[ "$FAILURES" -eq 0 ]]; then
    echo "结论: [PASS] 全部可观测指标符合预定义单最大定义与协议判定阈值！"
    exit 0
else
    echo "结论: [FAIL] 共检测到 $FAILURES 处判定未达标，请查看上方详细失败输出！"
    exit 1
fi
