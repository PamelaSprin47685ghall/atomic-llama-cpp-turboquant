#!/usr/bin/env bash
# scripts/reset-gpu-user.sh
# Level 1: 免 root 用户态 GPU 快速排空与泄漏进程清理
set -euo pipefail

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

say "Step 1: 扫描并强制终止占用 DRM / Vulkan / KFD 的用户态进程..."

# 找出所有打开 /dev/dri 或 /dev/kfd 的当前用户进程
PIDS=$(fuser /dev/dri/card* /dev/dri/renderD* /dev/kfd 2>/dev/null || true)

# 找出所有包含 llama / test / ctest 的相关运行中进程
EXTRA_PIDS=$(pgrep -u "$USER" -f 'llama-server|test-|ctest|ggml' || true)

ALL_PIDS=$(echo "$PIDS $EXTRA_PIDS" | tr ' ' '\n' | grep -v "^$$$" | sort -u | grep -v '^$' || true)

if [ -n "$ALL_PIDS" ]; then
    say "发现卡死/活跃 GPU 进程: $ALL_PIDS"
    kill -15 $ALL_PIDS 2>/dev/null || true
    sleep 0.5
    kill -9 $ALL_PIDS 2>/dev/null || true
    say "已发送 SIGKILL。"
else
    say "未检测到残留 GPU 用户进程。"
fi

say "Step 2: 审计五张 RX 6800 状态..."
python3 -c "
import glob

for c in sorted(glob.glob('/sys/class/drm/card[1-5]/device/gpu_busy_percent')):
    card = c.split('/')[4]
    with open(c) as f:
        busy = f.read().strip()
    with open(f'/sys/class/drm/{card}/device/mem_info_vram_used') as f:
        vram = int(f.read().strip()) / (1024*1024)
    status = 'OK' if busy == '0' and vram < 50 else 'STILL_BUSY'
    print(f'  {card}: busy={busy}%, vram={vram:.2f} MB [{status}]')
"

say "Level 1 清理完成。如果仍有卡显示 STILL_BUSY (如 99%)，请使用 sudo 执行 scripts/reset-gpu-pci.sh 进行 PCIe 总线复位。"
