#!/usr/bin/env bash
# scripts/rerot-perf-gate.sh
#
# 描述:
#   RERoT 性能/基准运行前置门禁脚本。
#   依据 AGENTS.md 真机安全门与 RERoT 规范 §17、§17.2，在启动任何性能或压力测试前：
#   1. 调用 scripts/gpu-idle-audit.sh 执行只读空闲判定；若非空闲则拒绝运行；
#   2. 空闲时将运行前证据（idle 快照、dmesg 基线、进程清单、时间戳）落盘到 logs/gpu-gate/<timestamp>/；
#   3. 执行传入的目标测试命令，严格捕获其退出码；
#   4. 命令结束后采集 teardown 证据（退出码、idle 复检、dmesg diff、pstore/journal 只读检查）写入同一目录；
#   5. 脚本全程杜绝 vkDeviceWaitIdle、设备 reset、modprobe、关闭 watchdog 等危险动作。
#
# 用法:
#   ./scripts/rerot-perf-gate.sh <command> [args...]
#
# 示例:
#   ./scripts/rerot-perf-gate.sh python3 scripts/rerot-target-ornith-multi-lane.py
#
# 退出码:
#   0   - 前置门通过，测试命令执行成功 (exit 0)，teardown 采集完成
#   非0 - 前置审计失败，或被执行命令本身失败，返回对应退出码

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
IDLE_AUDIT_SCRIPT="$SCRIPT_DIR/gpu-idle-audit.sh"

say() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

count_lines() {
    local text="$1"
    if [ -z "$text" ]; then
        echo 0
    else
        printf '%s\n' "$text" | wc -l
    fi
}

count_file_lines() {
    local file="$1"
    if [ ! -f "$file" ] || [ ! -s "$file" ]; then
        echo 0
    else
        wc -l < "$file" | tr -d ' '
    fi
}

if [ "$#" -lt 1 ]; then
    echo "用法: $0 <command> [args...]"
    echo "示例: $0 python3 scripts/rerot-target-ornith-multi-lane.py"
    exit 2
fi

if [ ! -f "$IDLE_AUDIT_SCRIPT" ]; then
    say "错误: 未找到前置空闲审计脚本: $IDLE_AUDIT_SCRIPT"
    exit 2
fi

say "=================================================="
say "           RERoT 性能运行前置门检查               "
say "=================================================="
say "待执行命令: $*"
echo ""

# -----------------------------------------------------------------------------
# 阶段 1: 运行前置只读空闲审计
# -----------------------------------------------------------------------------
say ">>> 阶段 1: 触发 GPU 只读空闲审计..."
AUDIT_EXIT=0
set +e
bash "$IDLE_AUDIT_SCRIPT"
AUDIT_EXIT=$?
set -euo pipefail

if [ "$AUDIT_EXIT" -ne 0 ]; then
    say "=================================================="
    if [ "$AUDIT_EXIT" -eq 3 ]; then
        say "[GATE REJECTED] GPU 空闲审计存在无法核实项 (退出码: 3，如内核日志不可读)！"
        say "安全门原则：无法核实项不得假定干净，拒绝启动性能测试。"
    else
        say "[GATE REJECTED] GPU 空闲审计未通过 (退出码: $AUDIT_EXIT)，拒绝启动性能测试！"
        say "保护机制生效：防止在非空闲或潜在挂起状态下叠加 GPU 负载。"
    fi
    say "=================================================="
    exit "$AUDIT_EXIT"
fi

say ">>> 阶段 1 通过: GPU 状态确认完全空闲且各项核实无误。"
echo ""

# -----------------------------------------------------------------------------
# 阶段 2: 采集运行前证据（落盘至 logs/gpu-gate/<timestamp>/）
# -----------------------------------------------------------------------------
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
GATE_LOG_DIR="$REPO_ROOT/logs/gpu-gate/$TIMESTAMP"
mkdir -p "$GATE_LOG_DIR"

say ">>> 阶段 2: 采集运行前证据至 $GATE_LOG_DIR"

# 2.1 元数据与时间戳
cat <<EOF > "$GATE_LOG_DIR/pre-run-metadata.json"
{
  "timestamp": "$TIMESTAMP",
  "command": $(printf '%s\n' "$*" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read().strip()))' 2>/dev/null || echo "\"$*\""),
  "user": "$USER",
  "host": "$(hostname)",
  "kernel": "$(uname -r)"
}
EOF

# 2.2 运行前 idle 快照
set +e
bash "$IDLE_AUDIT_SCRIPT" > "$GATE_LOG_DIR/pre-run-idle-audit.txt" 2>&1
set -euo pipefail

# 2.3 运行前 dmesg 基线（安全捕获权限与错误，修正行数）
DMESG_PRE_LINES=0
if command -v dmesg >/dev/null 2>&1; then
    set +e
    DMESG_PRE_OUT=$(dmesg 2>&1)
    DMESG_PRE_STATUS=$?
    set -euo pipefail

    if [ "$DMESG_PRE_STATUS" -eq 0 ]; then
        printf '%s\n' "$DMESG_PRE_OUT" > "$GATE_LOG_DIR/pre-run-dmesg.log"
        DMESG_PRE_LINES=$(count_file_lines "$GATE_LOG_DIR/pre-run-dmesg.log")
    else
        echo "[UNKNOWN] dmesg 执行失败 (退出码 $DMESG_PRE_STATUS): $DMESG_PRE_OUT" > "$GATE_LOG_DIR/pre-run-dmesg.log"
        say "  [UNKNOWN] 运行前 dmesg 权限被拒或读取失败: $DMESG_PRE_OUT"
    fi
fi

# 2.4 运行前进程清单
ps -ef > "$GATE_LOG_DIR/pre-run-process-list.txt" 2>&1 || true

say "运行前证据已固化落盘。"
echo ""

# -----------------------------------------------------------------------------
# 阶段 3: 执行受测命令并捕获退出码
# -----------------------------------------------------------------------------
say ">>> 阶段 3: 启动目标受测命令..."
say "--------------------------------------------------"

CMD_EXIT_CODE=0
set +e
"$@" 2>&1 | tee "$GATE_LOG_DIR/command-stdout-stderr.log"
CMD_EXIT_CODE=${PIPESTATUS[0]}
set -euo pipefail

echo ""
say "--------------------------------------------------"
say "受测命令已结束，退出码: $CMD_EXIT_CODE"
echo ""

# -----------------------------------------------------------------------------
# 阶段 4: 采集 Teardown 证据（安全排空审计）
# -----------------------------------------------------------------------------
say ">>> 阶段 4: 采集 Teardown 证据..."

# 4.1 退出码落盘
cat <<EOF > "$GATE_LOG_DIR/teardown-result.json"
{
  "command_exit_code": $CMD_EXIT_CODE,
  "teardown_timestamp": "$(date '+%Y%m%d_%H%M%S')"
}
EOF

# 4.2 运行后 idle 复检
set +e
bash "$IDLE_AUDIT_SCRIPT" > "$GATE_LOG_DIR/post-run-idle-audit.txt" 2>&1
POST_IDLE_EXIT=$?
set -euo pipefail

# 4.3 运行后 dmesg 差异比对
if command -v dmesg >/dev/null 2>&1; then
    set +e
    DMESG_POST_OUT=$(dmesg 2>&1)
    DMESG_POST_STATUS=$?
    set -euo pipefail

    if [ "$DMESG_POST_STATUS" -eq 0 ]; then
        printf '%s\n' "$DMESG_POST_OUT" > "$GATE_LOG_DIR/post-run-dmesg.log"
        DMESG_POST_LINES=$(count_file_lines "$GATE_LOG_DIR/post-run-dmesg.log")
        if [ "$DMESG_POST_LINES" -ge "$DMESG_PRE_LINES" ]; then
            LINES_DIFF=$(( DMESG_POST_LINES - DMESG_PRE_LINES ))
            if [ "$LINES_DIFF" -gt 0 ]; then
                tail -n "$LINES_DIFF" "$GATE_LOG_DIR/post-run-dmesg.log" > "$GATE_LOG_DIR/dmesg-diff.log" 2>/dev/null || true
            else
                : > "$GATE_LOG_DIR/dmesg-diff.log"
            fi
        fi
    else
        echo "[UNKNOWN] dmesg 执行失败 (退出码 $DMESG_POST_STATUS): $DMESG_POST_OUT" > "$GATE_LOG_DIR/post-run-dmesg.log"
        say "  [UNKNOWN] 运行后 dmesg 权限被拒或读取失败: $DMESG_POST_OUT"
    fi
fi

# 4.4 只读检查 /sys/fs/pstore 与 journalctl 异常（只读安全检查，杜绝修改，权限拒绝时报告 UNKNOWN）
mkdir -p "$GATE_LOG_DIR/system-audit"
if [ -d "/sys/fs/pstore" ]; then
    set +e
    PSTORE_LS_OUT=$(ls -la /sys/fs/pstore 2>&1)
    PSTORE_LS_STATUS=$?
    set -euo pipefail

    if [ "$PSTORE_LS_STATUS" -eq 0 ]; then
        printf '%s\n' "$PSTORE_LS_OUT" > "$GATE_LOG_DIR/system-audit/pstore-listing.txt"
        find /sys/fs/pstore -type f -exec cat {} + > "$GATE_LOG_DIR/system-audit/pstore-content.log" 2>/dev/null || true
    else
        say "  [UNKNOWN] /sys/fs/pstore 权限拒绝或读取失败: $PSTORE_LS_OUT"
        echo "[UNKNOWN] /sys/fs/pstore 权限拒绝或读取失败 (退出码 $PSTORE_LS_STATUS): $PSTORE_LS_OUT" > "$GATE_LOG_DIR/system-audit/pstore-listing.txt"
    fi
else
    say "  [INFO] /sys/fs/pstore 目录不存在（当前内核未挂载 pstore）"
    echo "pstore not mounted" > "$GATE_LOG_DIR/system-audit/pstore-listing.txt"
fi

if command -v journalctl >/dev/null 2>&1; then
    set +e
    JOURNAL_OUT=$(journalctl -k --since "5 minutes ago" --no-pager 2>&1)
    JOURNAL_STATUS=$?
    set -euo pipefail

    if [ "$JOURNAL_STATUS" -eq 0 ]; then
        printf '%s\n' "$JOURNAL_OUT" > "$GATE_LOG_DIR/system-audit/journal-kernel-recent.log"
    else
        say "  [UNKNOWN] journalctl 权限被拒或读取失败: $JOURNAL_OUT"
        echo "[UNKNOWN] journalctl 权限被拒 (退出码 $JOURNAL_STATUS): $JOURNAL_OUT" > "$GATE_LOG_DIR/system-audit/journal-kernel-recent.log"
    fi
fi

# 4.5 运行后残留进程检查
ps -ef > "$GATE_LOG_DIR/post-run-process-list.txt" 2>&1 || true

say "Teardown 证据采集完成，记录保存至: $GATE_LOG_DIR"
say "=================================================="

if [ "$CMD_EXIT_CODE" -ne 0 ]; then
    say "[GATE VERDICT] 受测命令执行失败 (退出码: $CMD_EXIT_CODE)"
    exit "$CMD_EXIT_CODE"
fi

if [ "$POST_IDLE_EXIT" -ne 0 ]; then
    if [ "$POST_IDLE_EXIT" -eq 3 ]; then
        say "[GATE VERDICT] 受测命令执行完毕，但 Teardown 审计存在无法核实项 (退出码: 3，如内核日志不可读)！"
        exit 3
    else
        say "[GATE VERDICT] 受测命令退出后，Teardown 空闲复检未通过（可能存在显存泄漏或 GPU 繁忙挂起）！"
        exit 1
    fi
fi

say "[GATE VERDICT] 测试命令执行成功且 Teardown 审计全绿通过！"
exit 0
