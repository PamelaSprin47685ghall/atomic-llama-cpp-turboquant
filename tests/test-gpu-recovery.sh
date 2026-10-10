#!/usr/bin/env bash
# No PCI writes, device kills, or model execution: all watchdog events are synthetic.
set -euo pipefail

repo=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
helper="$repo/scripts/gpu-hard-unlock.sh"
watchdog="$repo/scripts/eagle-gpu-watchdog.sh"

if bash "$helper" >/dev/null 2>&1; then
    echo 'Implicit all-card recovery was accepted' >&2
    exit 1
fi
if bash "$helper" --dry-run card99 >/dev/null 2>&1; then
    echo 'Invalid GPU target was accepted' >&2
    exit 1
fi

if bash "$watchdog" --once >/dev/null 2>&1; then
    echo 'A destructive one-shot scan of historical kernel errors was accepted' >&2
    exit 1
fi

(
    source "$watchdog"
    DRY_RUN=1
    LAST_KERNEL_TIMESTAMP=100.000000
    declare -a recovered=()
    declare -a clocks_restored=()
    fake_logs=''
    fetch_kernel_logs() { printf '%s\n' "$fake_logs"; }
    recover_card() { recovered+=("$1"); }
    restore_clock_card() { clocks_restored+=("$1"); }

    # Five simultaneously busy cards, even across four polls, prove no fault.
    for _ in 1 2 3 4; do
        for c in card1 card2 card3 card4 card5; do record_busy_sample "$c" 99; done
    done
    [ "${#recovered[@]}" -eq 0 ] || { echo 'High utilization triggered reset' >&2; exit 1; }
    for c in card1 card2 card3 card4 card5; do
        [ "${CARD_BUSY_CONSECUTIVE[$c]}" -eq 4 ] || { echo "$c counter not per-card" >&2; exit 1; }
    done
    record_busy_sample card2 0
    [ "${CARD_BUSY_CONSECUTIVE[card2]}" -eq 0 ] || { echo 'Idle card did not clear its counter' >&2; exit 1; }

    # A driver rebind may drop the manual policy without emitting a ring timeout.
    # Restore only the affected card and do not turn a policy drift into PCI recovery.
    fake_policy=auto
    read_clock_policy() {
        if [ "$1" = card2 ]; then printf '%s\n' "$fake_policy"; else printf 'manual\n'; fi
    }
    restore_clock_card() {
        clocks_restored+=("$1")
        [ "$1" != card2 ] || fake_policy=manual
    }
    run_watchdog_step
    [ "${clocks_restored[*]}" = card2 ] && [ "${#recovered[@]}" -eq 0 ] || {
        echo 'Lost manual policy did not trigger card-only clock restoration' >&2; exit 1;
    }
    run_watchdog_step
    [ "${clocks_restored[*]}" = card2 ] || { echo 'Stable manual policy retriggered restoration' >&2; exit 1; }
    clocks_restored=()

    fake_logs=$'[ 101.000001] amdgpu 0000:06:00.0: ring comp_1.0.0 timeout\n[ 101.000002] amdgpu 0000:03:00.0: GPU reset begin'
    run_watchdog_step
    [ "${recovered[*]}" = '' ] && [ -n "${CARD_PENDING_SINCE[card4]:-}" ] || {
        echo 'Ring timeout must wait for driver recovery, not immediately hot-unplug' >&2; exit 1;
    }
    run_watchdog_step
    [ "${#recovered[@]}" -eq 0 ] || { echo 'Same kernel error retriggered recovery' >&2; exit 1; }

    fake_logs=$'[ 102.000001] amdgpu: ring comp_1.0.0 timeout'
    run_watchdog_step
    [ "${#recovered[@]}" -eq 0 ] || { echo 'Unattributed kernel error triggered reset' >&2; exit 1; }
    [ "$LAST_KERNEL_TIMESTAMP" = 102.000001 ] || { echo 'Ambiguous error did not advance cursor' >&2; exit 1; }
    run_watchdog_step
    [ "${#recovered[@]}" -eq 0 ] || { echo 'Ambiguous error retriggered reset' >&2; exit 1; }

    fake_logs=$'[ 103.000001] amdgpu 0000:06:00.0: GPU reset(1) succeeded!'
    run_watchdog_step
    [ "${#recovered[@]}" -eq 0 ] && [ "${clocks_restored[*]}" = card4 ] &&
        [ -z "${CARD_PENDING_SINCE[card4]:-}" ] || {
        echo 'Kernel recovery must cancel PCI reset and restore only clocks' >&2; exit 1;
    }

    fake_logs=$'[ 104.000001] amdgpu 0000:09:00.0: ring gfx_0.0.0 timeout'
    run_watchdog_step
    [ -n "${CARD_PENDING_SINCE[card2]:-}" ] || { echo 'New card2 error was lost' >&2; exit 1; }
    CARD_PENDING_SINCE[card2]=$(( $(date +%s) - AUTO_RESET_GRACE - 1 ))
    fake_logs=''
    run_watchdog_step
    [ "${recovered[*]}" = card2 ] || { echo 'Unresolved card2 error did not escalate singly after grace' >&2; exit 1; }

    fake_logs=$'[ 105.000001] amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout\n[ 105.000002] amdgpu 0000:03:00.0: Ring gfx_0.0.0 reset failed\n[ 105.000003] amdgpu 0000:03:00.0: GPU reset succeeded, trying to resume'
    run_watchdog_step
    [ -n "${CARD_PENDING_SINCE[card5]:-}" ] && [ "${clocks_restored[*]}" = card4 ] || {
        echo 'Intermediate reset progress was mistaken for final driver recovery' >&2; exit 1;
    }
    fake_logs=$'[ 105.000004] amdgpu 0000:03:00.0: GPU reset(1) succeeded!'
    run_watchdog_step
    [ "${recovered[*]}" = card2 ] && [ "${clocks_restored[*]}" = 'card4 card5' ] || {
        echo 'Mode1 kernel success after ring failure caused a redundant hot-unplug' >&2; exit 1;
    }

    fake_logs=$'[ 106.000001] amdgpu 0000:0f:00.0: GPU reset failed'
    run_watchdog_step
    CARD_PENDING_SINCE[card1]=$(( $(date +%s) - FAILED_RESET_GRACE - 1 ))
    fake_logs=''
    run_watchdog_step
    [ "${recovered[*]}" = 'card2 card1' ] || { echo 'Confirmed unresolved reset failure not escalated' >&2; exit 1; }

    fake_logs=$'[ 107.000001] amdgpu 0000:0c:00.0: device lost from bus!\n[ 107.000002] amdgpu 0000:0c:00.0: SMU: bus error for message: TransferTableSmu2Dram(18) response:0xFFFFFFFF'
    run_watchdog_step
    [ "${CARD_PENDING_KIND[card3]:-}" = bus_lost ] && [ -z "${CARD_CLOCK_SINCE[card3]:-}" ] &&
        [ "${clocks_restored[*]}" = 'card4 card5' ] || {
        echo 'Lost PCI bus was treated as a DPM-only event' >&2; exit 1;
    }
    CARD_PENDING_SINCE[card3]=$(( $(date +%s) - FAILED_RESET_GRACE - 1 ))
    fake_logs=''
    run_watchdog_step
    [ "${recovered[*]}" = 'card2 card1 card3' ] || {
        echo 'Unrecovered bus loss did not escalate to single-card reset before IPMI timeout' >&2; exit 1;
    }

    fake_logs=$'[ 108.000001] amdgpu 0000:0c:00.0: device lost from bus!\n[ 108.000002] amdgpu 0000:0c:00.0: GPU reset begin'
    run_watchdog_step
    CARD_PENDING_SINCE[card3]=$(( $(date +%s) - AUTO_RESET_GRACE - 10 ))
    fake_logs=''
    run_watchdog_step
    [ "${recovered[*]}" = 'card2 card1 card3' ] || {
        echo 'Eagle raced a late-starting kernel reset' >&2; exit 1;
    }
    fake_logs=$'[ 109.000001] amdgpu 0000:0c:00.0: GPU Recovery Failed: -19'
    run_watchdog_step
    [ -z "${CARD_DRIVER_RESET_SINCE[card3]:-}" ] &&
        [ "${recovered[*]}" = 'card2 card1 card3 card3' ] || {
        echo 'Failed kernel reset was not escalated before IPMI timeout' >&2; exit 1;
    }
)

(
    source "$watchdog"
    DRY_RUN=1
    LAST_KERNEL_TIMESTAMP=200.000000
    fake_logs=$'[ 201.000001] amdgpu 0000:0c:00.0: device lost from bus!'
    fetch_kernel_logs() { printf '%s\n' "$fake_logs"; }
    scan_gpu_busy_advisory() { :; }
    read_clock_policy() { printf 'manual\n'; }
    declare -a rebooted=()
    recover_card() { CARD_HELPER_ATTEMPTED[$1]=1; [ "$1" != card3 ]; }
    card_operational() { return 1; }
    request_early_reboot() { rebooted+=("$1"); }
    run_watchdog_step
    CARD_PENDING_SINCE[card3]=$(( $(date +%s) - FAILED_RESET_GRACE - 1 ))
    fake_logs=''
    if run_watchdog_step; then
        echo 'Failed target recovery was reported successful' >&2; exit 1
    fi
    [ "${rebooted[*]}" = card3 ] || { echo 'Unrecoverable bus loss did not request early reboot' >&2; exit 1; }
    card_operational() { return 0; }
    fake_logs=$'[ 202.000001] amdgpu 0000:0c:00.0: device lost from bus!'
    run_watchdog_step
    CARD_PENDING_SINCE[card3]=$(( $(date +%s) - FAILED_RESET_GRACE - 1 ))
    fake_logs=''
    if run_watchdog_step; then
        echo 'Failed target recovery was reported successful' >&2; exit 1
    fi
    [ "${rebooted[*]}" = card3 ] || { echo 'Recovered card caused an unnecessary host reboot' >&2; exit 1; }

    recover_card() { CARD_HELPER_ATTEMPTED[$1]=0; return 1; }
    card_operational() { return 1; }
    fake_logs=$'[ 203.000001] amdgpu 0000:06:00.0: device lost from bus!'
    run_watchdog_step
    CARD_PENDING_SINCE[card4]=$(( $(date +%s) - FAILED_RESET_GRACE - 1 ))
    fake_logs=''
    if run_watchdog_step; then
        echo 'Missing recovery helper was reported successful' >&2; exit 1
    fi
    [ "${rebooted[*]}" = card3 ] || { echo 'Skipped card reset triggered a host reboot' >&2; exit 1; }
)

(
    source "$watchdog"
    fake_card=$(mktemp -d)
    trap 'rm -rf "$fake_card"' EXIT
    mkdir "$fake_card/drm"
    printf '2: 1200Mhz *\n' > "$fake_card/pp_dpm_sclk"
    printf '99\n' > "$fake_card/gpu_busy_percent"
    card_operational_at_path "$fake_card" || {
        echo 'Healthy GPU at 99 percent utilization was marked dead' >&2; exit 1;
    }
    rm "$fake_card/pp_dpm_sclk"
    if card_operational_at_path "$fake_card"; then
        echo 'GPU with missing DPM was marked recovered' >&2; exit 1
    fi
)

(
    source "$watchdog"
    DRY_RUN=0
    guard_dir=$(mktemp -d)
    trap 'rm -rf "$guard_dir"' EXIT
    EARLY_REBOOT_FILE="$guard_dir/last-early-reboot"
    declare -a reboot_commands=()
    systemctl() { reboot_commands+=("$*"); }
    request_early_reboot card3
    [ "${reboot_commands[*]}" = 'reboot --no-wall' ] || {
        echo 'Early reboot did not request orderly shutdown' >&2; exit 1;
    }
    if request_early_reboot card3; then
        echo 'Durable reboot guard allowed a second reboot within 24 hours' >&2; exit 1
    fi
    [ "${#reboot_commands[@]}" -eq 1 ] || { echo 'A reboot loop was requested' >&2; exit 1; }
)

(
    source "$watchdog"
    ONCE=0
    fetch_kernel_logs() { return 1; }
    result=0
    watchdog_main || result=$?
    [ "$result" -eq 2 ] || { echo 'Kernel log failure replayed stale incidents at startup' >&2; exit 1; }
    fetch_kernel_logs() { printf '%s\n' 'amdgpu: device lost from bus!'; }
    result=0
    watchdog_main || result=$?
    [ "$result" -eq 2 ] || { echo 'Missing kernel timestamp replayed stale incidents at startup' >&2; exit 1; }
)

echo 'GPU recovery fail-closed dispatch checks passed'
