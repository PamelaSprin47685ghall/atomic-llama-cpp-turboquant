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
)

echo 'GPU recovery fail-closed dispatch checks passed'
