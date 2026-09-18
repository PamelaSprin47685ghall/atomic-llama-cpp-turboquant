#!/usr/bin/env bash
# scripts/reset-gpu-hard.sh
# 终极内核级 GPU 死锁自愈程序：
# 1. 强制杀灭任何持有 DRM/Vulkan/KFD 的僵尸与存活进程（解除内存持有）
# 2. 调用 amdgpu 内核原生恢复接口 (amdgpu_gpu_recover)
# 3. 如果仍被锁死，通过 PCI sysfs 执行干净的驱动解绑与重新扫描 (PCI unbind -> rescan -> bind)
# 4. 彻底解决以往 reset 脚本因无法写入导致被 watchdog 咬死的缺陷。

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "错误: 本脚本需要 root 权限。"
    exit 1
fi

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

say "Step 1: 强力清空所有占用 DRM/Vulkan 的进程与僵尸进程..."
pkill -9 -f 'llama-server|llama-cli|test-vulkan|microbench|test-' 2>/dev/null || true
sleep 1

# BDF 物理列表
BDFS=("0000:03:00.0" "0000:06:00.0" "0000:09:00.0" "0000:0c:00.0" "0000:0f:00.0")

say "Step 2: 探测卡死状态并尝试内核原生 GPU Recover..."
for bdf in "${BDFS[@]}"; do
    rec_node="/sys/kernel/debug/dri/$bdf/amdgpu_gpu_recover"
    if [ -f "$rec_node" ]; then
        say "  向 $bdf 触发 amdgpu_gpu_recover..."
        cat "$rec_node" >/dev/null 2>&1 || true
    fi
done
sleep 1

say "Step 3: 检查卡死状态，对仍繁忙的卡执行 PCI 驱动级重绑定与重置..."
need_rescan=0
for bdf in "${BDFS[@]}"; do
    dev_dir="/sys/bus/pci/devices/$bdf"
    if [ ! -d "$dev_dir" ]; then
        say "  $bdf 丢失，标记需要总线重新扫描！"
        need_rescan=1
        continue
    fi

    # 检查是否有显存或队列挂起
    if [ -e "$dev_dir/drm" ]; then
        card_dir=$(ls "$dev_dir/drm" 2>/dev/null | grep -E '^card[0-9]+$' | head -n 1 || true)
        if [ -n "$card_dir" ]; then
            busy=$(cat "/sys/class/drm/$card_dir/device/gpu_busy_percent" 2>/dev/null || echo "0")
            if [ "$busy" -gt 0 ]; then
                say "  $bdf ($card_dir) 处于锁死状态 (busy=$busy%)，执行强制重置..."
                if [ -e "$dev_dir/driver" ]; then
                    echo "$bdf" > /sys/bus/pci/drivers/amdgpu/unbind 2>/dev/null || true
                fi
                echo 1 > "$dev_dir/remove" 2>/dev/null || true
                need_rescan=1
            fi
        fi
    fi
done

if [ "$need_rescan" -eq 1 ]; then
    say "  执行 PCIe 总线全局重新扫描..."
    echo 1 > /sys/bus/pci/rescan
    sleep 2
fi

say "Step 4: 最终审计五张卡状态..."
all_ok=1
for i in {1..5}; do
    dev="/sys/class/drm/card${i}/device"
    if [ -d "$dev" ]; then
        busy=$(cat "$dev/gpu_busy_percent" 2>/dev/null || echo "N/A")
        vram=$(cat "$dev/mem_info_vram_used" 2>/dev/null || echo "0")
        vram_mb=$(( vram / 1024 / 1024 ))
        say "  card$i: busy=$busy%, vram=${vram_mb} MB"
        if [ "$busy" != "0" ]; then
            all_ok=0
        fi
    else
        say "  card$i: [错误] 未找到设备节点！"
        all_ok=0
    fi
done

if [ "$all_ok" -eq 1 ]; then
    say "🎉 五张卡全部成功复位至干净空闲状态 (busy=0%)！"
else
    say "⚠️ 部分卡未完全就绪，请复查硬件状态。"
fi
