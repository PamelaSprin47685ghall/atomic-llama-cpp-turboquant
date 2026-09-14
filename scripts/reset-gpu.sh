#!/usr/bin/env bash
# scripts/reset-gpu.sh
# 一键 GPU 智能排空与 PCIe 硬件复位脚本
# 无需手动输入密码，自动按阶梯恢复 GPU：
#   Level 1: 扫描并 SIGKILL 卡死的 DRM 用户进程（快速排空）
#   Level 2: 对仍处于异常挂起（busy > 0% 或显存泄漏）的卡执行 PCIe 总线级硬件复位
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "=================================================="
echo "      AMD 5x RX 6800 一键 GPU 快速复位程序       "
echo "=================================================="

# 1. 运行 Level 1 用户态快速排空
"$REPO/scripts/reset-gpu-user.sh"

# 2. 检查是否有卡片依然挂住 (busy > 0% 或显存异常占用)
NEED_HARDWARE_RESET=0
for c in card1 card2 card3 card4 card5; do
    busy=$(cat "/sys/class/drm/$c/device/gpu_busy_percent" 2>/dev/null || echo "0")
    if [ "$busy" -gt 0 ]; then
        echo "--> 检测到 $c 仍处于繁忙/挂起状态 (busy=$busy%)！需要触发 PCIe 硬件复位。"
        NEED_HARDWARE_RESET=1
    fi
done

# 如果参数显式传入 -f 或 --force，强制执行硬件复位
if [ "${1:-}" = "-f" ] || [ "${1:-}" = "--force" ]; then
    echo "--> 用户指定强制硬件复位模式 (--force)。"
    NEED_HARDWARE_RESET=1
fi

if [ "$NEED_HARDWARE_RESET" -eq 1 ]; then
    echo "正在免密以 root 身份调用 PCIe 总线级硬件复位..."
    sudo -n "$REPO/scripts/reset-gpu-pci.sh" all
    echo "PCIe 硬件总线复位完成！"
else
    echo "所有卡片状态已完全恢复至空闲基线 (busy=0%)，无需触发 PCIe 硬件复位。"
fi

echo "=================================================="
echo "              GPU 状态检查确认完成                "
echo "=================================================="
