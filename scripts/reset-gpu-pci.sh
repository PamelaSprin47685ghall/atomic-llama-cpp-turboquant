#!/usr/bin/env bash
# Manual GPU recovery entrypoint. The root-owned installed helper owns every reset.
# `check-clocks-only` is intentionally non-resetting: it restores the policy after
# a reboot or a previously completed GPU recovery without touching PCI topology.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "GPU recovery requires root; use sudo." >&2
    exit 1
fi

if [ "$#" -ne 1 ]; then
    echo "Usage: sudo $0 <card1..card5|all|check-clocks-only|install-watchdog>" >&2
    exit 2
fi

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

lock_manual_clocks() {
    local cards=(card1 card2 card3 card4 card5)
    local bdfs=(0000:0f:00.0 0000:09:00.0 0000:0c:00.0 0000:06:00.0 0000:03:00.0)
    local i perf sclk line highest readback
    for i in "${!cards[@]}"; do
        perf="/sys/bus/pci/devices/${bdfs[$i]}/power_dpm_force_performance_level"
        sclk="/sys/bus/pci/devices/${bdfs[$i]}/pp_dpm_sclk"
        if [ ! -w "$perf" ] || [ ! -r "$sclk" ] || [ ! -w "$sclk" ]; then
            say "${cards[$i]} (${bdfs[$i]}): DPM nodes unavailable; cannot verify clock policy" >&2
            return 1
        fi
        highest=""
        while IFS= read -r line; do
            if [[ "$line" =~ ^([0-9]+): ]]; then
                highest="${BASH_REMATCH[1]}"
            fi
        done < "$sclk"
        if [ -z "$highest" ]; then
            say "${cards[$i]} (${bdfs[$i]}): no DPM index in $sclk" >&2
            return 1
        fi
        printf 'manual\n' > "$perf"
        printf '%s\n' "$highest" > "$sclk"
        readback=$(< "$perf")
        if [ "$readback" != manual ]; then
            say "${cards[$i]} (${bdfs[$i]}): manual mode did not persist" >&2
            return 1
        fi
        say "${cards[$i]} (${bdfs[$i]}): manual + highest DPM index $highest requested; actual loaded frequency requires a workload check"
    done
}

repo=$(readlink -f "$(dirname "$0")/..")
if [ "$1" = install-watchdog ]; then
    exec "$repo/scripts/install-gpu-recovery.sh"
fi
if [ "$1" = check-clocks-only ]; then
    lock_manual_clocks
    exit 0
fi

case "$1" in
    card1|card2|card3|card4|card5|all) ;;
    *) echo "Invalid recovery target: $1" >&2; exit 2 ;;
esac

source_helper="$repo/scripts/gpu-hard-unlock.sh"
installed_helper=/usr/local/bin/gpu-hard-unlock.sh
if [ ! -x "$installed_helper" ] || ! cmp -s "$source_helper" "$installed_helper"; then
    echo "Refusing recovery: installed root-owned helper differs from reviewed repository source." >&2
    exit 1
fi
exec "$installed_helper" "$1"
