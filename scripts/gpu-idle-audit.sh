#!/usr/bin/env bash
# scripts/gpu-idle-audit.sh
#
# 描述:
#   真机 GPU 只读空闲判定与健康审计脚本。
#   依据 AGENTS.md 真机安全门与 RERoT 规范 §17，对指定 GPU（默认 card1..card5）
#   执行单次只读状态采样，绝不引入自旋、等待或设备修改。
#
# 用法:
#   ./scripts/gpu-idle-audit.sh
#   CARDS="card1 card2" ./scripts/gpu-idle-audit.sh
#   REROT_GPU_VRAM_MAX_MB=100 ./scripts/gpu-idle-audit.sh
#
# 环境变量:
#   CARDS                   - 目标卡列表，空格分隔；未指定时自动探测 /sys/class/drm/card*/device/gpu_busy_percent
#   REROT_GPU_VRAM_MAX_MB  - 单卡显存使用上限阈值（MB），默认 50
#   GPU_AUDIT_STATE_FILE    - 仓库外记录上次 dmesg 时间戳与行数的 state 文件，默认 "/tmp/gpu-idle-audit-dmesg.state"
#
# 退出码:
#   0 - 全部检查项通过（GPU 完全空闲且健康）
#   1 - 存在任一不满足项（繁忙、显存超标、残留进程、新增 amdgpu 错误、未探测到卡）
#   2 - 环境或必要工具缺失
#   3 - 存在无法核实的检查项（如 dmesg 权限被拒，不得假定干净）

set -euo pipefail

# 自动探测实际存在的 DRM 卡（以 sysfs gpu_busy_percent 为准），保留环境变量覆盖能力
if [ -z "${CARDS:-}" ]; then
    DETECTED_CARDS=()
    for busy_file in /sys/class/drm/card*/device/gpu_busy_percent; do
        if [ -f "$busy_file" ]; then
            card_dir=$(dirname "$(dirname "$busy_file")")
            card_name=$(basename "$card_dir")
            DETECTED_CARDS+=("$card_name")
        fi
    done
    if [ ${#DETECTED_CARDS[@]} -gt 0 ]; then
        CARDS="${DETECTED_CARDS[*]}"
    else
        CARDS=""
    fi
fi

REROT_GPU_VRAM_MAX_MB=${REROT_GPU_VRAM_MAX_MB:-50}
GPU_AUDIT_STATE_FILE=${GPU_AUDIT_STATE_FILE:-"/tmp/gpu-idle-audit-dmesg.state"}

say() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

count_lines() {
    local text="$1"
    if [ -z "$text" ]; then
        echo 0
    else
        printf '%s\n' "$text" | wc -l
    fi
}

FAILURES=()
UNKNOWNS=()

say "=================================================="
say "           GPU 只读空闲判定审计开始                "
say "=================================================="
say "目标卡列表: ${CARDS:-[未探测到任何 DRM 卡]}"
say "显存上限阈值: ${REROT_GPU_VRAM_MAX_MB} MB"
say "增量状态文件: $GPU_AUDIT_STATE_FILE"
echo ""

# -----------------------------------------------------------------------------
# 检查项 0: DRM 设备存在性检查
# -----------------------------------------------------------------------------
if [ -z "$CARDS" ]; then
    say "--- [检查项 0] DRM 卡存在性检查 ---"
    say "  [FAIL] 系统中未探测到任何存在的 DRM 卡 (/sys/class/drm/card*/device/gpu_busy_percent 均不存在)"
    FAILURES+=("未探测到任何可用 DRM 卡")
fi

# -----------------------------------------------------------------------------
# 检查项 1 & 2: gpu_busy_percent = 0% 且 显存占用低于阈值
# -----------------------------------------------------------------------------
say "--- [检查项 1 & 2] GPU 负载与显存只读状态采集 ---"

for card in $CARDS; do
    busy_path="/sys/class/drm/$card/device/gpu_busy_percent"
    vram_used_path="/sys/class/drm/$card/device/mem_info_vram_used"
    vram_total_path="/sys/class/drm/$card/device/mem_info_vram_total"

    if [ ! -f "$busy_path" ]; then
        say "  [FAIL] $card: 节点不存在 $busy_path"
        FAILURES+=("$card: sysfs gpu_busy_percent 节点缺失")
        continue
    fi

    busy_val=$(cat "$busy_path" 2>/dev/null || echo "UNKNOWN")
    if [ "$busy_val" != "0" ]; then
        say "  [FAIL] $card: gpu_busy_percent = ${busy_val}% (期望 0%)"
        FAILURES+=("$card: GPU 处于非空闲繁忙状态 (busy=${busy_val}%)")
    else
        say "  [PASS] $card: gpu_busy_percent = 0%"
    fi

    if [ -f "$vram_used_path" ]; then
        vram_used_bytes=$(cat "$vram_used_path" 2>/dev/null || echo "0")
        vram_used_mb=$(( vram_used_bytes / 1048576 ))
        vram_used_fract=$(( (vram_used_bytes % 1048576) * 100 / 1048576 ))

        if [ "$vram_used_mb" -ge "$REROT_GPU_VRAM_MAX_MB" ]; then
            say "  [FAIL] $card: 显存使用 ${vram_used_mb}.${vram_used_fract} MB >= 阈值 ${REROT_GPU_VRAM_MAX_MB} MB"
            FAILURES+=("$card: 显存超出阈值 (${vram_used_mb}.${vram_used_fract} MB >= ${REROT_GPU_VRAM_MAX_MB} MB)")
        else
            say "  [PASS] $card: 显存使用 ${vram_used_mb}.${vram_used_fract} MB < 阈值 ${REROT_GPU_VRAM_MAX_MB} MB"
        fi
    else
        say "  [WARN] $card: 无法读取 mem_info_vram_used 节点"
    fi
done
echo ""

# -----------------------------------------------------------------------------
# 检查项 3: 无残留 llama-server / llama-cli / mesh 测试相关进程
# -----------------------------------------------------------------------------
say "--- [检查项 3] 残留模型及测试进程审计 ---"

TARGET_PROCS_REGEX='llama-server|llama-cli|test-vulkan-tp5-mesh|test-rerot|test-backend-ops|rerot-target-ornith'
ACTIVE_PROCS=$(pgrep -a -f "$TARGET_PROCS_REGEX" 2>/dev/null || true)

# 过滤掉当前正在执行的审计脚本进程自身
FILTERED_PROCS=""
if [ -n "$ACTIVE_PROCS" ]; then
    while IFS= read -r line; do
        pid=$(echo "$line" | awk '{print $1}')
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
            FILTERED_PROCS+="$line"$'\n'
        fi
    done <<< "$ACTIVE_PROCS"
fi

if [ -n "$FILTERED_PROCS" ]; then
    say "  [FAIL] 检测到残留活跃的目标进程:"
    echo "$FILTERED_PROCS" | sed 's/^/         /'
    FAILURES+=("存在残留的模型/测试进程占用资源")
else
    say "  [PASS] 未检测到残留的 llama-server/llama-cli/mesh/rerot 测试进程"
fi
echo ""

# -----------------------------------------------------------------------------
# 检查项 4: 相对上次基线无新增 amdgpu 错误（增量比对，不得假定干净）
# -----------------------------------------------------------------------------
say "--- [检查项 4] 内核 dmesg 增量 amdgpu 错误检查 ---"

AMDGPU_ERROR_REGEX='(amdgpu.*(ring.*timeout|GPU reset|GPUVM fault|ERROR|dma_fence_wait_timeout|page fault|fence.*timeout)|\[gfxhub\].*fault|ErrorDeviceLost)'

CURRENT_DMESG_COUNT=0
CURRENT_LAST_TIMESTAMP=""
NEW_ERRORS=""

if command -v dmesg >/dev/null 2>&1; then
    DMESG_OUTPUT=""
    DMESG_STATUS=0
    set +e
    DMESG_OUTPUT=$(dmesg 2>&1)
    DMESG_STATUS=$?
    set -euo pipefail

    if [ "$DMESG_STATUS" -ne 0 ]; then
        # 缺陷 (1) 修复：dmesg 权限被拒或读取失败，如实报告 UNKNOWN，绝不误报 PASS
        say "  [UNKNOWN] 内核日志不可读: dmesg 执行失败 (退出码: $DMESG_STATUS)"
        say "            诊断信息: $DMESG_OUTPUT"
        UNKNOWNS+=("内核日志不可读 (dmesg 权限被拒或受限: $DMESG_OUTPUT)")
    else
        FULL_DMESG="$DMESG_OUTPUT"
        CURRENT_DMESG_COUNT=$(count_lines "$FULL_DMESG")
        CURRENT_LAST_TIMESTAMP=$(echo "$FULL_DMESG" | tail -n 1 | awk '{print $1}' | tr -d '[]' || echo "")

        PREV_COUNT=0
        PREV_TIMESTAMP=""
        if [ -f "$GPU_AUDIT_STATE_FILE" ]; then
            read -r PREV_COUNT PREV_TIMESTAMP < "$GPU_AUDIT_STATE_FILE" 2>/dev/null || true
        fi

        # 提取增量行
        if [ "$PREV_COUNT" -gt 0 ] && [ "$CURRENT_DMESG_COUNT" -ge "$PREV_COUNT" ]; then
            LINES_TO_CHECK=$(( CURRENT_DMESG_COUNT - PREV_COUNT ))
            if [ "$LINES_TO_CHECK" -gt 0 ]; then
                INCREMENTAL_DMESG=$(echo "$FULL_DMESG" | tail -n "$LINES_TO_CHECK")
                NEW_ERRORS=$(echo "$INCREMENTAL_DMESG" | grep -Ei "$AMDGPU_ERROR_REGEX" || true)
            fi
        else
            # 首次检查或 dmesg buffer 已回绕/重启：全量检查
            say "  [INFO] 未找到有效前序基线或 dmesg 发生轮转，执行全量基线比对"
            NEW_ERRORS=$(echo "$FULL_DMESG" | grep -Ei "$AMDGPU_ERROR_REGEX" || true)
        fi

        # 更新本次 state 文件
        mkdir -p "$(dirname "$GPU_AUDIT_STATE_FILE")"
        echo "$CURRENT_DMESG_COUNT $CURRENT_LAST_TIMESTAMP" > "$GPU_AUDIT_STATE_FILE"

        if [ -n "$NEW_ERRORS" ]; then
            say "  [FAIL] 相对上次检查发现新增 amdgpu 内核级错误/警告:"
            echo "$NEW_ERRORS" | sed 's/^/         /'
            FAILURES+=("内核 dmesg 检测到新增 amdgpu 异常/fault 日志")
        else
            say "  [PASS] 相对上次基线无新增 amdgpu 错误日志 (当前有效行数: $CURRENT_DMESG_COUNT)"
        fi
    fi
else
    say "  [UNKNOWN] dmesg 命令未在 PATH 中找到，内核日志不可读"
    UNKNOWNS+=("系统缺少 dmesg 命令，内核日志不可读")
fi
echo ""

# -----------------------------------------------------------------------------
# 总结与判定
# -----------------------------------------------------------------------------
say "=================================================="
if [ ${#FAILURES[@]} -eq 0 ] && [ ${#UNKNOWNS[@]} -eq 0 ]; then
    say "  >>> 审计结果: 100% PASS (所有 GPU 空闲且所有检查项完全通过) <<<"
    say "=================================================="
    exit 0
elif [ ${#FAILURES[@]} -gt 0 ]; then
    say "  >>> 审计结果: FAIL (发现 ${#FAILURES[@]} 处阻断项) <<<"
    for err in "${FAILURES[@]}"; do
        say "  - 阻断原因: $err"
    done
    if [ ${#UNKNOWNS[@]} -gt 0 ]; then
        say "  附注（同时存在无法核实项）:"
        for unk in "${UNKNOWNS[@]}"; do
            say "  * 无法核实: $unk"
        done
    fi
    say "=================================================="
    say "处理建议 (依据 AGENTS.md 安全指引):"
    say "  1. 检查是否存在未退出的后台进程并由其所属者正常终止；"
    say "  2. 严禁无保护/无界自旋或依赖 vkDeviceWaitIdle；"
    say "  3. 严禁随意执行硬件总线 reset 或盲目重试；"
    say "  4. 绝不关闭内核 Watchdog，保留现场日志并返回 Manager 排查。"
    say "=================================================="
    exit 1
else
    # FAILURES 为空，但 UNKNOWNS 不为空（例如 dmesg 权限被拒）：不得给出整体 PASS
    say "  >>> 审计结果: INCONCLUSIVE / UNKNOWN (存在 ${#UNKNOWNS[@]} 处无法核实项，不得假定通过) <<<"
    for unk in "${UNKNOWNS[@]}"; do
        say "  - 无法核实项: $unk"
    done
    say "  结论说明: 设备硬件 sysfs 空闲状态正常，但因权限或环境限制，关键项（如内核日志）不可读；按安全门要求拒绝标记全量 PASS。"
    say "=================================================="
    exit 3
fi
