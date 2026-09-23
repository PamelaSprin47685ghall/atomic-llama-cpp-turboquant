#!/usr/bin/env bash
# scripts/reset-gpu-pci.sh
# Level 2: PCIe 总线级硬件复位（Secondary Bus Reset / FLR）
# 使用场景：GPU 发生硬件级 Page Fault / 环挂起，且 Level 1 用户态清理后 gpu_busy 仍锁死 99% 时。
# 必须以 root 权限（或 sudo）执行。

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "错误: 本脚本需要 root / sudo 权限来写入 PCIe sysfs 节点。"
    echo "用法: sudo $0 [card_id|bdf|all]"
    exit 1
fi

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# 目标 5 张卡物理 BDF 拓扑
declare -A CARD_BDF=(
    ["card1"]="0000:0f:00.0"
    ["card2"]="0000:09:00.0"
    ["card3"]="0000:0c:00.0"
    ["card4"]="0000:06:00.0"
    ["card5"]="0000:03:00.0"
)

TARGET=${1:-"all"}

lock_manual_clocks() {
    say "=== 设置全部卡为 manual 模式并锁定最高 DPM 档位 ==="
    for c in card1 card2 card3 card4 card5; do
        sclk_file="/sys/class/drm/$c/device/pp_dpm_sclk"
        perf_level="/sys/class/drm/$c/device/power_dpm_force_performance_level"
        [ -r "$sclk_file" ] && [ -w "$perf_level" ] || {
            say "  [$c] 缺少可访问的 DPM sysfs 节点，无法确认时钟策略"
            return 1
        }
        last_idx=$(grep -oE '^[0-9]+' "$sclk_file" | tail -1)
        [ -n "$last_idx" ] || { say "  [$c] 未找到有效 DPM 档位"; return 1; }
        echo manual > "$perf_level"
        echo "$last_idx" > "$sclk_file"
        [ "$(cat "$perf_level")" = manual ] || { say "  [$c] manual 模式未生效"; return 1; }
        say "  [$c] manual + DPM $last_idx 已写入；静态频率表不等于负载下的实际算力"
    done
}

if [ "$TARGET" = "check-clocks-only" ]; then
    lock_manual_clocks
    exit 0
fi

reset_bdf() {
    local bdf=$1
    local card_name=$2
    say "===> 正在对 $card_name ($bdf) 执行 PCIe 总线复位..."

    # 1. 如果支持 reset_method，确认是否支持 bus reset
    if [ -f "/sys/bus/pci/devices/$bdf/reset_method" ]; then
        say "  支持的复位方法: $(cat /sys/bus/pci/devices/$bdf/reset_method)"
    fi

    # 2. 写入 PCIe reset 节点
    if [ -w "/sys/bus/pci/devices/$bdf/reset" ]; then
        say "  向 /sys/bus/pci/devices/$bdf/reset 发送 1..."
        echo 1 > "/sys/bus/pci/devices/$bdf/reset" || {
            say "  标准 reset 节点写入失败，尝试 PCIe 摘除与重新扫描 (remove & rescan)..."
            echo 1 > "/sys/bus/pci/devices/$bdf/remove"
            sleep 1
            echo 1 > /sys/bus/pci/rescan
        }
    else
        say "  未找到 reset 节点，尝试 PCIe 摘除与重新扫描 (remove & rescan)..."
        echo 1 > "/sys/bus/pci/devices/$bdf/remove"
        sleep 1
        echo 1 > /sys/bus/pci/rescan
    fi

    say "  $card_name ($bdf) 复位完成。"
}

# 预先清理占用进程
say "先强制终止所有相关 DRM 进程..."
pkill -9 -f 'llama-server|test-|ctest' 2>/dev/null || true
sleep 1

if [ "$TARGET" = "all" ]; then
    for c in card1 card2 card3 card4 card5; do
        bdf="${CARD_BDF[$c]}"
        # 只复位 busy > 0 或指定 all
        busy=$(cat "/sys/class/drm/$c/device/gpu_busy_percent" 2>/dev/null || echo "0")
        if [ "$busy" -gt 0 ] || [ "${FORCE:-0}" = "1" ]; then
            say "$c 处于繁忙/挂起状态 (busy=$busy%)，触发复位..."
            reset_bdf "$bdf" "$c"
        else
            say "$c 状态正常 (busy=0%)，跳过。"
        fi
    done
else
    # 指定单卡或指定 BDF
    if [[ "$TARGET" =~ ^card[1-5]$ ]]; then
        reset_bdf "${CARD_BDF[$TARGET]}" "$TARGET"
    elif [[ "$TARGET" =~ ^[0-9a-fA-F]{4}: ]]; then
        reset_bdf "$TARGET" "custom"
    else
        echo "未知目标: $TARGET. 请传入 card1..card5 或 BDF (如 0000:03:00.0) 或 all"
        exit 2
    fi
fi

say "等待驱动重新稳定..."
sleep 2

say "当前 GPU 状态:"
python3 -c "
import glob

for c in sorted(glob.glob('/sys/class/drm/card[1-5]/device/gpu_busy_percent')):
    card = c.split('/')[4]
    with open(c) as f:
        busy = f.read().strip()
    with open(f'/sys/class/drm/{card}/device/mem_info_vram_used') as f:
        vram = int(f.read().strip()) / (1024*1024)
    print(f'  {card}: busy={busy}%, vram={vram:.2f} MB')
"

# 与无需总线复位的路径共用同一套时钟策略；失败时不得打印成功。
lock_manual_clocks

say "硬件复位全流程结束。"
