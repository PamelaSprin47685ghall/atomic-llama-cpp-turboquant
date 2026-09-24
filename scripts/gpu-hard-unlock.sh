#!/usr/bin/env bash
# scripts/gpu-hard-unlock.sh
#
# Target-specific fail-closed GPU recovery script.
#
# CLI Syntax:
#   gpu-hard-unlock.sh [--dry-run|--clocks-only] <card1..card5|all>
#
# Semantics & Guarantees:
# 1. Targets:
#    Explicit 'card1'..'card5' or explicit 'all'. Implicit 'all' is strictly forbidden.
#    With explicit 'all', all targets are preflighted and validated before any mutation.
# 2. Privileges & Concurrency:
#    --dry-run performs inspection only without writes or kills and can run unprivileged.
#    Mutating runs require root privilege and acquire an exclusive flock on /run/lock/gpu-hard-unlock.lock.
# 3. Topology & Target Quiescence:
#    Hardware mapping:
#      card1: GPU 0000:0f:00.0 (Audio 0000:0f:00.1), Bridge 0000:0e:00.0
#      card2: GPU 0000:09:00.0 (Audio 0000:09:00.1), Bridge 0000:08:00.0
#      card3: GPU 0000:0c:00.0 (Audio 0000:0c:00.1), Bridge 0000:0b:00.0
#      card4: GPU 0000:06:00.0 (Audio 0000:06:00.1), Bridge 0000:05:00.0
#      card5: GPU 0000:03:00.0 (Audio 0000:03:00.1), Bridge 0000:02:00.0
#    Identifies target DRM primary and render nodes and audio device.
#    Never kills unrelated system processes or display managers.
#    Targeted compute processes (matching llama-server, llama-cli, test-*, etc.) are sent SIGTERM,
#    followed by a bounded wait for processes to exit and DRM/audio file handles to drain.
#    SIGKILL is used cautiously only if targeted processes remain.
#    If unknown, non-compute, or display clients hold target handles, or handles fail to drain,
#    the script aborts fail-closed BEFORE any PCI mutation.
# 4. Recovery Sequence:
#    Step A: Try native PCI sysfs device reset (/sys/bus/pci/devices/<gpu>/reset) first.
#    Step B: If native reset fails or does not restore an idle DRM device, and clients are quiesced,
#            execute targeted GPU + audio PCI remove, then secondary bus reset (SBR) on the
#            upstream bridge controller via BRIDGE_CONTROL (0x3e.w |= 0x40).
#            Reset bit is guaranteed to be released (cleared) within 200ms (trapped for safety).
#            Then trigger PCI rescan and verify BDF + DRM nodes reappear.
# 5. Clock & Performance Restoration:
#    After ANY reset (native or SBR), force manual performance level ('manual' in
#    power_dpm_force_performance_level) and set the highest available pp_dpm_sclk index.
#    Policy readback is verified; actual compute needs a separate loaded ALU probe.
#    If write or readback fails, recovery fails with non-zero exit.
#
# Exit codes:
#   0: Device restored and idle, DPM policy applied (ALU throughput unverified)
#   1: General failure or incomplete recovery (fail-closed)
#   2: Invalid CLI argument or syntax
#   3: Privilege error (non-root in mutating mode)
#   4: Lock acquisition failure (concurrency conflict)
#   5: Target quiesce failure (foreign/unknown process or un-drainable handle)
#   6: Hardware / PCI reset or verification failure
#   7: DPM clock restoration failure

set -euo pipefail

LOCK_FILE="/run/lock/gpu-hard-unlock.lock"
DRY_RUN=0
CLOCKS_ONLY=0
TARGET=""
LOCK_FD=200

# Global SBR state tracking for fail-safe trap handler (survives function exit)
SBR_TRAP_ACTIVE=0
SBR_BRIDGE_BDF=""
SBR_CLEAR_VAL=""
SBR_CARD_ID=""

log() {
    printf '[%s] [gpu-hard-unlock] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

err() {
    printf '[%s] [gpu-hard-unlock] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

# ---------------------------------------------------------
# Static Hardware Mapping
# ---------------------------------------------------------
declare -A CARD_GPU=(
    ["card1"]="0000:0f:00.0"
    ["card2"]="0000:09:00.0"
    ["card3"]="0000:0c:00.0"
    ["card4"]="0000:06:00.0"
    ["card5"]="0000:03:00.0"
)

declare -A CARD_AUDIO=(
    ["card1"]="0000:0f:00.1"
    ["card2"]="0000:09:00.1"
    ["card3"]="0000:0c:00.1"
    ["card4"]="0000:06:00.1"
    ["card5"]="0000:03:00.1"
)

declare -A CARD_BRIDGE=(
    ["card1"]="0000:0e:00.0"
    ["card2"]="0000:08:00.0"
    ["card3"]="0000:0b:00.0"
    ["card4"]="0000:05:00.0"
    ["card5"]="0000:02:00.0"
)

# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [--dry-run|--clocks-only] <card1..card5|all>

Options:
  --dry-run    Inspect target status, topology, and client processes without mutation.
               Can be executed without root privileges.
  --clocks-only  Restore manual/highest DPM after the kernel has completed its own reset.
                 Never kill clients or write PCI configuration; requires one explicit card.
  cardN        Target card identifier (card1, card2, card3, card4, card5).
  all          Explicitly target all 5 cards. Implicit 'all' is prohibited.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --clocks-only)
            CLOCKS_ONLY=1
            shift
            ;;
        card1|card2|card3|card4|card5|all)
            if [ -n "$TARGET" ]; then
                err "Multiple targets specified ('$TARGET' and '$1')."
                usage
                exit 2
            fi
            TARGET="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown argument: '$1'"
            usage
            exit 2
            ;;
    esac
done

if [ -z "$TARGET" ]; then
    err "No target specified. Implicit 'all' is strictly refused. Specify card1..card5 or explicit 'all'."
    usage
    exit 2
fi
if [ "$CLOCKS_ONLY" -eq 1 ] && { [ "$DRY_RUN" -eq 1 ] || [ "$TARGET" = all ]; }; then
    err "--clocks-only requires one card and cannot be combined with --dry-run."
    exit 2
fi

# ---------------------------------------------------------
# Target List Resolution
# ---------------------------------------------------------
TARGET_CARDS=()
if [ "$TARGET" = "all" ]; then
    TARGET_CARDS=("card1" "card2" "card3" "card4" "card5")
else
    TARGET_CARDS=("$TARGET")
fi

# ---------------------------------------------------------
# Privilege & Lock Handling
# ---------------------------------------------------------
if [ "$DRY_RUN" -eq 0 ]; then
    if [ "$(id -u)" -ne 0 ]; then
        err "Root privilege required for mutating hardware recovery. Run with sudo or use --dry-run."
        exit 3
    fi

    mkdir -p "$(dirname "$LOCK_FILE")" 2>/dev/null || true
    exec 200>"$LOCK_FILE"
    if ! flock -n 200; then
        err "Another instance of gpu-hard-unlock is currently running (lock $LOCK_FILE held). Aborting."
        exit 4
    fi
fi

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

# Find DRM device nodes for a given GPU BDF
# e.g., /dev/dri/cardX, /dev/dri/renderDxxx
get_target_drm_nodes() {
    local gpu_bdf="$1"
    local nodes=()
    local drm_dir="/sys/bus/pci/devices/$gpu_bdf/drm"

    if [ -d "$drm_dir" ]; then
        for entry in "$drm_dir"/*; do
            [ -e "$entry" ] || continue
            local name
            name=$(basename "$entry")
            if [[ "$name" =~ ^(card[0-9]+|renderD[0-9]+|controlD[0-9]+)$ ]]; then
                if [ -e "/dev/dri/$name" ]; then
                    nodes+=("/dev/dri/$name")
                fi
            fi
        done
    fi
    echo "${nodes[@]}"
}

# Find audio device nodes (ALSA) for an audio BDF
get_target_audio_nodes() {
    local audio_bdf="$1"
    local nodes=()

    # Derive from /sys/class/sound symlinks matching target audio BDF
    if [ -d "/sys/class/sound" ]; then
        for entry in /sys/class/sound/*; do
            [ -e "$entry" ] || continue
            local real
            real=$(readlink -f "$entry" 2>/dev/null || true)
            if [[ "$real" == *"$audio_bdf"* ]]; then
                local name
                name=$(basename "$entry")
                # Skip raw card directories (/dev/snd/cardX does not exist as a character node)
                # Only collect real character devices (/dev/snd/controlCX, /dev/snd/pcm*, /dev/snd/hw*)
                if [ -c "/dev/snd/$name" ]; then
                    nodes+=("/dev/snd/$name")
                fi
            fi
        done
    fi
    echo "${nodes[@]}"
}

# Return PIDs holding files on given device paths
get_pids_for_nodes() {
    local nodes=("$@")
    if [ ${#nodes[@]} -eq 0 ]; then
        return 0
    fi
    # fuser returns 0 if matches found, 1 if no matches found.
    # Any return code > 1 indicates execution failure/error.
    # Furthermore, capture stderr to ensure no permission/access errors are hidden behind rc 1.
    local fuser_err_tmp
    fuser_err_tmp=$(mktemp)
    local fuser_out fuser_rc=0
    fuser_out=$(fuser "${nodes[@]}" 2>"$fuser_err_tmp") || fuser_rc=$?
    local fuser_err
    fuser_err=$(cat "$fuser_err_tmp" 2>/dev/null || true)
    rm -f "$fuser_err_tmp"

    if [ -n "$fuser_err" ] && echo "$fuser_err" | grep -qiE "permission denied|cannot access|no such file"; then
        err "fuser reported access/permission errors on nodes (${nodes[*]}): $fuser_err"
        return 5
    fi

    if [ "$fuser_rc" -eq 1 ]; then
        # No matches found: clean exit, no PIDs
        return 0
    elif [ "$fuser_rc" -gt 1 ]; then
        err "fuser failed with exit code $fuser_rc on nodes: ${nodes[*]}"
        return "$fuser_rc"
    fi
    echo "$fuser_out" | tr -s ' ' '\n' | grep -v '^$' | sort -u || true
}

# Verify whether a PID is still alive and has open file handles on target nodes
pid_has_node_handles() {
    local pid="$1"
    shift
    local nodes=("$@")
    [ -d "/proc/$pid" ] || return 1
    [ -d "/proc/$pid/fd" ] || return 1

    local fd
    for fd in /proc/"$pid"/fd/*; do
        [ -e "$fd" ] 2>/dev/null || continue
        local target
        target=$(readlink -f "$fd" 2>/dev/null || true)
        for n in "${nodes[@]}"; do
            if [ "$target" = "$n" ]; then
                return 0
            fi
        done
    done
    return 1
}

# Check if a process is a known safe-to-terminate compute client
is_known_compute_process() {
    local pid="$1"
    if [ ! -d "/proc/$pid" ]; then
        return 0
    fi

    # Strict check: inspect /proc/$pid/exe realpath / basename ONLY.
    # Never inspect arbitrary cmdline or comm substrings which can match display managers or wrapper scripts.
    local exe_path exe_base
    exe_path=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    if [ -z "$exe_path" ]; then
        # Cannot determine executable; fail closed
        return 1
    fi
    exe_base=$(basename "$exe_path")

    case "$exe_base" in
        llama-server|llama-cli|llama-bench|test-vulkan|test-llama|microbench|benchmark)
            return 0
            ;;
    esac
    return 1
}

# ---------------------------------------------------------
# Preflight Validation
# ---------------------------------------------------------
log "Starting recovery preflight for targets: ${TARGET_CARDS[*]} (dry_run=$DRY_RUN)..."

for c in "${TARGET_CARDS[@]}"; do
    gpu_bdf="${CARD_GPU[$c]}"
    audio_bdf="${CARD_AUDIO[$c]}"
    bridge_bdf="${CARD_BRIDGE[$c]}"

    # Verify bridge exists in sysfs
    if [ ! -d "/sys/bus/pci/devices/$bridge_bdf" ]; then
        err "Target $c upstream PCIe bridge $bridge_bdf does not exist in sysfs."
        exit 6
    fi

    # Validate bridge is the actual parent of the GPU device in sysfs
    if [ -d "/sys/bus/pci/devices/$gpu_bdf" ]; then
        actual_parent=$(basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$gpu_bdf")")")
        if [ "$actual_parent" != "$bridge_bdf" ]; then
            err "Topology mismatch for $c: expected upstream bridge $bridge_bdf, but actual sysfs parent is $actual_parent."
            exit 6
        fi
    fi

    # Verify bridge is indeed a PCI bridge (header type 0x01) or has secondary bus register
    if ! setpci -s "$bridge_bdf" 0x3e.w >/dev/null 2>&1; then
        err "Cannot read BRIDGE_CONTROL (0x3e.w) on upstream bridge $bridge_bdf for $c."
        exit 6
    fi
done

# If targeting 'all', preflight clients and topology across ALL cards before any mutation begins!
if [ "$TARGET" = "all" ]; then
    log "Preflighting client processes across all targets before starting mutation..."
    for c in "${TARGET_CARDS[@]}"; do
        gpu_bdf="${CARD_GPU[$c]}"
        audio_bdf="${CARD_AUDIO[$c]}"
        read -r -a drm_nodes <<< "$(get_target_drm_nodes "$gpu_bdf")"
        read -r -a audio_nodes <<< "$(get_target_audio_nodes "$audio_bdf")"
        all_nodes=("${drm_nodes[@]}" "${audio_nodes[@]}")
        if [ ${#all_nodes[@]} -gt 0 ]; then
            pids_out=""
            if ! pids_out=$(get_pids_for_nodes "${all_nodes[@]}"); then
                err "Preflight: failed to query processes holding nodes for $c."
                exit 5
            fi
            if [ -n "$pids_out" ]; then
                read -r -a pids <<< "$pids_out"
                for p in "${pids[@]}"; do
                    [ -d "/proc/$p" ] || continue
                    if ! is_known_compute_process "$p"; then
                        p_info=$(ps -p "$p" -o pid,user,comm,args --no-headers 2>/dev/null || echo "PID $p")
                        err "Preflight: target $c held by foreign/non-compute process: $p_info"
                        err "Refusing recovery of all targets; aborting before any mutation."
                        exit 5
                    fi
                done
            fi
        fi
    done
    log "Preflight check passed: all targets clear of foreign processes."
fi
# ---------------------------------------------------------
# Process Inspection & Draining (Target Quiescence)
# ---------------------------------------------------------
quiesce_target() {
    local card_id="$1"
    local gpu_bdf="${CARD_GPU[$card_id]}"
    local audio_bdf="${CARD_AUDIO[$card_id]}"

    read -r -a drm_nodes <<< "$(get_target_drm_nodes "$gpu_bdf")"
    read -r -a audio_nodes <<< "$(get_target_audio_nodes "$audio_bdf")"
    local all_nodes=("${drm_nodes[@]}" "${audio_nodes[@]}")

    if [ ${#all_nodes[@]} -eq 0 ]; then
        log "  [$card_id] No active device nodes in /dev/dri or /dev/snd for $gpu_bdf."
        return 0
    fi

    log "  [$card_id] Device nodes to quiesce: ${all_nodes[*]}"

    local pids_out pids=()
    # Fail closed if get_pids_for_nodes fails (non-zero or error)
    if ! pids_out=$(get_pids_for_nodes "${all_nodes[@]}"); then
        err "[$card_id] Failed to inspect processes holding nodes via fuser; failing closed."
        return 5
    fi
    if [ -n "$pids_out" ]; then
        read -r -a pids <<< "$pids_out"
    fi

    if [ ${#pids[@]} -eq 0 ]; then
        log "  [$card_id] No processes currently holding target device handles."
        return 0
    fi

    log "  [$card_id] Found processes holding target nodes: ${pids[*]}"

    # Verify every process: must be known compute, never display manager or foreign system daemon
    for p in "${pids[@]}"; do
        [ -d "/proc/$p" ] || continue
        if ! is_known_compute_process "$p"; then
            local p_info
            p_info=$(ps -p "$p" -o pid,user,comm,args --no-headers 2>/dev/null || echo "PID $p")
            err "Target $card_id held by non-compute or foreign process: $p_info"
            err "Refusing unsafe termination; aborting recovery fail-closed."
            return 5
        fi
    done

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] Would terminate compute processes: ${pids[*]}"
        return 0
    fi

    # Send SIGTERM first to allow graceful cleanup
    log "  [$card_id] Verifying and sending SIGTERM to compute processes: ${pids[*]}"
    for p in "${pids[@]}"; do
        # Recheck executable immediately before signalling to avoid PID reuse race
        if is_known_compute_process "$p"; then
            kill -15 "$p" 2>/dev/null || true
        else
            err "[$card_id] PID $p changed executable or vanished before SIGTERM; aborting fail-closed."
            return 5
        fi
    done

    # Bounded wait for processes and handles to drain (up to 5 seconds)
    local waited=0
    while [ $waited -lt 10 ]; do
        local remaining_pids=()
        local pids_still_holding=()
        local rem_out
        if ! rem_out=$(get_pids_for_nodes "${all_nodes[@]}"); then
            err "[$card_id] Error checking remaining processes via fuser."
            return 5
        fi
        if [ -n "$rem_out" ]; then
            read -r -a remaining_pids <<< "$rem_out"
        fi

        # Double check original PIDs to ensure drm_release / exit_files completely finished
        for orig_p in "${pids[@]}"; do
            if [ -d "/proc/$orig_p" ] || pid_has_node_handles "$orig_p" "${all_nodes[@]}"; then
                pids_still_holding+=("$orig_p")
            fi
        done

        if [ ${#remaining_pids[@]} -eq 0 ] && [ ${#pids_still_holding[@]} -eq 0 ]; then
            log "  [$card_id] All client processes fully exited and device handles cleanly drained."
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done

    # If still remaining, cautiously send SIGKILL only to known compute processes
    local rem_out
    local remaining_pids=()
    if ! rem_out=$(get_pids_for_nodes "${all_nodes[@]}"); then
        err "[$card_id] Error re-checking remaining processes via fuser."
        return 5
    fi
    if [ -n "$rem_out" ]; then
        read -r -a remaining_pids <<< "$rem_out"
    fi
    if [ ${#remaining_pids[@]} -gt 0 ]; then
        log "  [$card_id] Sending SIGKILL to remaining stubborn compute processes: ${remaining_pids[*]}"
        for p in "${remaining_pids[@]}"; do
            if is_known_compute_process "$p"; then
                kill -9 "$p" 2>/dev/null || true
            else
                err "New unexpected process appeared holding node during shutdown: PID $p"
                return 5
            fi
        done

        # Bounded wait after SIGKILL (up to 3 seconds)
        waited=0
        while [ $waited -lt 6 ]; do
            local rem_out2
            local remaining_pids=()
            local pids_still_holding=()
            if ! rem_out2=$(get_pids_for_nodes "${all_nodes[@]}"); then
                err "[$card_id] Error checking remaining processes after SIGKILL."
                return 5
            fi
            if [ -n "$rem_out2" ]; then
                read -r -a remaining_pids <<< "$rem_out2"
            fi
            for orig_p in "${pids[@]}"; do
                if [ -d "/proc/$orig_p" ] || pid_has_node_handles "$orig_p" "${all_nodes[@]}"; then
                    pids_still_holding+=("$orig_p")
                fi
            done
            if [ ${#remaining_pids[@]} -eq 0 ] && [ ${#pids_still_holding[@]} -eq 0 ]; then
                log "  [$card_id] Handles cleanly drained after SIGKILL."
                return 0
            fi
            sleep 0.5
            waited=$((waited + 1))
        done
    fi

    # If handles still remain open, abort before any PCI write to prevent amdgpu_ttm_fini Oops
    local final_pids=()
    local final_out
    if ! final_out=$(get_pids_for_nodes "${all_nodes[@]}"); then
        err "[$card_id] Error performing final fuser check."
        return 5
    fi
    if [ -n "$final_out" ]; then
        read -r -a final_pids <<< "$final_out"
    fi
    if [ ${#final_pids[@]} -gt 0 ]; then
        err "[$card_id] Device handles could not be drained (PIDs: ${final_pids[*]})."
        err "Aborting before PCI mutation to prevent kernel crash."
        return 5
    fi

    # Also verify none of the original target PIDs are still in /proc or hold handles
    for orig_p in "${pids[@]}"; do
        if [ -d "/proc/$orig_p" ]; then
            # Check if zombie with closed fds
            local stat state
            state=$(awk '{print $3}' "/proc/$orig_p/stat" 2>/dev/null || echo "")
            if [ "$state" = "Z" ]; then
                if pid_has_node_handles "$orig_p" "${all_nodes[@]}"; then
                    err "[$card_id] Zombie PID $orig_p still retains open node file handles."
                    return 5
                fi
            else
                err "[$card_id] Process PID $orig_p still exists in state '$state'; exit incomplete."
                return 5
            fi
        fi
    done

    return 0
}

# ---------------------------------------------------------
# DPM Clock and Manual Performance Level Restoration
# ---------------------------------------------------------
restore_target_dpm() {
    local card_id="$1"
    local gpu_bdf="${CARD_GPU[$card_id]}"

    # Locate card directory in sysfs, e.g. /sys/bus/pci/devices/<bdf>/drm/cardX
    local card_sysfs=""
    if [ -d "/sys/bus/pci/devices/$gpu_bdf/drm" ]; then
        for cdir in "/sys/bus/pci/devices/$gpu_bdf/drm"/card[0-9]*; do
            if [ -d "$cdir" ] && [ ! -L "$cdir" ]; then
                card_sysfs="$cdir"
                break
            fi
        done
    fi

    # Fallback to direct device nodes
    local perf_level=""
    local sclk_file=""

    if [ -n "$card_sysfs" ] && [ -e "$card_sysfs/device/power_dpm_force_performance_level" ]; then
        perf_level="$card_sysfs/device/power_dpm_force_performance_level"
        sclk_file="$card_sysfs/device/pp_dpm_sclk"
    elif [ -e "/sys/bus/pci/devices/$gpu_bdf/power_dpm_force_performance_level" ]; then
        perf_level="/sys/bus/pci/devices/$gpu_bdf/power_dpm_force_performance_level"
        sclk_file="/sys/bus/pci/devices/$gpu_bdf/pp_dpm_sclk"
    fi

    if [ -z "$perf_level" ] || [ ! -f "$sclk_file" ]; then
        err "[$card_id] DPM sysfs files not accessible on $gpu_bdf (perf_level='$perf_level', sclk='$sclk_file')."
        return 7
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        local curr_perf curr_sclk highest_idx
        curr_perf=$(cat "$perf_level" 2>/dev/null || echo "unknown")
        highest_idx=$(grep -oE '^[0-9]+' "$sclk_file" | tail -1 || echo "")
        log "  [DRY-RUN] [$card_id] Current perf_level=$curr_perf. Highest DPM sclk index=$highest_idx."
        return 0
    fi

    local highest_idx
    highest_idx=$(grep -oE '^[0-9]+' "$sclk_file" | tail -1 || true)
    if [ -z "$highest_idx" ]; then
        err "[$card_id] No valid DPM frequency index parsed from $sclk_file."
        return 7
    fi

    log "  [$card_id] Setting power_dpm_force_performance_level to 'manual'..."
    if ! echo "manual" > "$perf_level"; then
        err "[$card_id] Failed to write 'manual' to $perf_level."
        return 7
    fi

    local readback_perf
    readback_perf=$(cat "$perf_level" 2>/dev/null || true)
    if [ "$readback_perf" != "manual" ]; then
        err "[$card_id] power_dpm_force_performance_level verification failed (got '$readback_perf', expected 'manual')."
        return 7
    fi

    log "  [$card_id] Setting highest pp_dpm_sclk index '$highest_idx'..."
    if ! echo "$highest_idx" > "$sclk_file"; then
        err "[$card_id] Failed to write index '$highest_idx' to $sclk_file."
        return 7
    fi

    # Readback verifies policy, not effective loaded clocks or ALU throughput.
    # Note: on Navi 21 with specific VBIOS / driver states, pp_dpm_sclk table may not display an active
    # asterisk indicator on the highest index while idle; we verify successful write, manual mode persistence,
    # and log the current DPM table state.
    readback_perf=$(cat "$perf_level" 2>/dev/null || true)
    if [ "$readback_perf" != "manual" ]; then
        err "[$card_id] power_dpm_force_performance_level reverted from 'manual' after sclk write."
        return 7
    fi

    local readback_sclk
    readback_sclk=$(cat "$sclk_file" 2>/dev/null || true)
    log "  [$card_id] Manual DPM policy applied: index ($highest_idx) written. Loaded ALU throughput unverified. Table:\n$readback_sclk"
    return 0
}

# ---------------------------------------------------------
# Secondary Bus Reset (SBR) with guaranteed reset release
# ---------------------------------------------------------
do_bridge_sbr() {
    local card_id="$1"
    local bridge_bdf="${CARD_BRIDGE[$card_id]}"

    log "  [$card_id] Performing Secondary Bus Reset via upstream bridge $bridge_bdf..."

    local ctrl
    if ! ctrl=$(setpci -s "$bridge_bdf" 0x3e.w 2>/dev/null) || [ -z "$ctrl" ]; then
        err "[$card_id] Unable to read BRIDGE_CONTROL (0x3e.w) from $bridge_bdf."
        return 6
    fi

    local reset_val clear_val
    reset_val=$(printf "%04x" "$(( 0x$ctrl | 0x0040 ))")
    # clear_val explicitly preserves all original bits while clearing bit 6 (0x0040)
    clear_val=$(printf "%04x" "$(( 0x$ctrl & ~0x0040 ))")

    # Configure global trap state so EXIT and signals at any point will deassert even after function return
    SBR_TRAP_ACTIVE=0
    SBR_BRIDGE_BDF="$bridge_bdf"
    SBR_CLEAR_VAL="$clear_val"
    SBR_CARD_ID="$card_id"

    trap '
        if [ "${SBR_TRAP_ACTIVE:-0}" -eq 1 ] && [ -n "${SBR_BRIDGE_BDF:-}" ] && [ -n "${SBR_CLEAR_VAL:-}" ]; then
            setpci -s "$SBR_BRIDGE_BDF" 0x3e.w="$SBR_CLEAR_VAL" 2>/dev/null || true
            log "[${SBR_CARD_ID:-unknown}] Emergency EXIT trap: reset-bit release executed on $SBR_BRIDGE_BDF."
        fi
    ' EXIT
    trap '
        if [ "${SBR_TRAP_ACTIVE:-0}" -eq 1 ] && [ -n "${SBR_BRIDGE_BDF:-}" ] && [ -n "${SBR_CLEAR_VAL:-}" ]; then
            setpci -s "$SBR_BRIDGE_BDF" 0x3e.w="$SBR_CLEAR_VAL" 2>/dev/null || true
            log "[${SBR_CARD_ID:-unknown}] Emergency INT trap: reset-bit release executed on $SBR_BRIDGE_BDF."
        fi
        trap - INT TERM EXIT
        exit 130
    ' INT
    trap '
        if [ "${SBR_TRAP_ACTIVE:-0}" -eq 1 ] && [ -n "${SBR_BRIDGE_BDF:-}" ] && [ -n "${SBR_CLEAR_VAL:-}" ]; then
            setpci -s "$SBR_BRIDGE_BDF" 0x3e.w="$SBR_CLEAR_VAL" 2>/dev/null || true
            log "[${SBR_CARD_ID:-unknown}] Emergency TERM trap: reset-bit release executed on $SBR_BRIDGE_BDF."
        fi
        trap - INT TERM EXIT
        exit 143
    ' TERM

    log "  [$card_id] Asserting Secondary Bus Reset (0x3e.w = $reset_val)..."
    if ! setpci -s "$bridge_bdf" 0x3e.w="$reset_val"; then
        err "[$card_id] Failed to write reset assertion ($reset_val) to bridge $bridge_bdf."
        trap - INT TERM EXIT
        return 6
    fi
    SBR_TRAP_ACTIVE=1

    # Hold reset for 200ms per PCIe spec
    sleep 0.2

    log "  [$card_id] Deasserting Secondary Bus Reset (0x3e.w = $clear_val)..."
    if ! setpci -s "$bridge_bdf" 0x3e.w="$clear_val"; then
        err "[$card_id] CRITICAL: Failed to write deassert value ($clear_val) to bridge $bridge_bdf!"
        # Do NOT unhook trap here! Leave trap active so script exit will attempt deassert again
        return 6
    fi

    # Explicitly verify readback to guarantee reset bit is cleared
    local verify_ctrl
    if ! verify_ctrl=$(setpci -s "$bridge_bdf" 0x3e.w 2>/dev/null) || [ -z "$verify_ctrl" ]; then
        err "[$card_id] CRITICAL: Unable to readback BRIDGE_CONTROL from $bridge_bdf after deassert."
        # Do NOT unhook trap here!
        return 6
    fi
    if [ "$(( 0x$verify_ctrl & 0x0040 ))" -ne 0 ]; then
        err "[$card_id] CRITICAL: Bridge $bridge_bdf still has SBR bit 6 asserted (0x$verify_ctrl)!"
        # Do NOT unhook trap here!
        return 6
    fi

    # Successfully deasserted and verified; safely disable SBR trap
    SBR_TRAP_ACTIVE=0
    trap - INT TERM EXIT

    # Allow link training and electrical settling
    sleep 1.5

    return 0
}

# ---------------------------------------------------------
# Recovery Execution per Card
# ---------------------------------------------------------
recover_target() {
    local card_id="$1"
    local gpu_bdf="${CARD_GPU[$card_id]}"
    local audio_bdf="${CARD_AUDIO[$card_id]}"
    local bridge_bdf="${CARD_BRIDGE[$card_id]}"

    log "========================================================="
    log "Beginning recovery sequence for target: $card_id ($gpu_bdf)"
    log "========================================================="

    # Step 1: Quiesce target processes & drain handles
    if ! quiesce_target "$card_id"; then
        return 5
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] [$card_id] Target quiescence simulation completed."
        if ! restore_target_dpm "$card_id"; then
            return 7
        fi
        return 0
    fi

    local reset_success=0

    # Step 2: Try native PCI sysfs reset first if device exists
    if [ -d "/sys/bus/pci/devices/$gpu_bdf" ] && [ -w "/sys/bus/pci/devices/$gpu_bdf/reset" ]; then
        log "  [$card_id] Attempting native PCI sysfs device reset on $gpu_bdf..."
        local native_write_ok=0
        echo 1 > "/sys/bus/pci/devices/$gpu_bdf/reset" 2>/dev/null && native_write_ok=1 || native_write_ok=0
        if [ "$native_write_ok" -eq 1 ]; then
            log "  [$card_id] Native PCI sysfs reset write accepted. Checking recovery..."
            sleep 1.5
            local attempt busy_now drm_ready entry
            for ((attempt = 0; attempt < 10; attempt++)); do
                drm_ready=0
                for entry in "/sys/bus/pci/devices/$gpu_bdf/drm"/card[0-9]*; do
                    if [ -d "$entry" ] && [ ! -L "$entry" ]; then
                        drm_ready=1
                        break
                    fi
                done
                busy_now=""
                if [ -r "/sys/bus/pci/devices/$gpu_bdf/gpu_busy_percent" ]; then
                    busy_now=$(< "/sys/bus/pci/devices/$gpu_bdf/gpu_busy_percent")
                fi
                if [ "$drm_ready" -eq 1 ] && [ "$busy_now" = 0 ]; then
                    reset_success=1
                    log "  [$card_id] Native reset restored DRM and idle GPU."
                    break
                fi
                sleep 1
            done
            if [ "$reset_success" -eq 0 ]; then
                log "  [$card_id] Native reset did not restore idle DRM within 10 checks (busy=$busy_now); falling back to SBR."
            fi
        else
            log "  [$card_id] Native sysfs reset failed or was rejected by kernel; falling back to SBR."
        fi
    else
        log "  [$card_id] Native PCI reset node not available or not writable on $gpu_bdf; using SBR path."
    fi

    # Step 3: Targeted remove + Bridge SBR + Rescan if native reset did not suffice
    if [ "$reset_success" -eq 0 ]; then
        log "  [$card_id] Executing targeted PCI remove and Bridge SBR..."

        # Remove audio function first
        if [ -d "/sys/bus/pci/devices/$audio_bdf" ]; then
            log "  [$card_id] Removing audio function $audio_bdf..."
            if ! (echo 1 > "/sys/bus/pci/devices/$audio_bdf/remove" 2>/dev/null); then
                err "[$card_id] Failed to write remove to /sys/bus/pci/devices/$audio_bdf/remove"
                return 6
            fi
        fi

        # Remove GPU function
        if [ -d "/sys/bus/pci/devices/$gpu_bdf" ]; then
            log "  [$card_id] Removing GPU function $gpu_bdf..."
            if ! (echo 1 > "/sys/bus/pci/devices/$gpu_bdf/remove" 2>/dev/null); then
                err "[$card_id] Failed to write remove to /sys/bus/pci/devices/$gpu_bdf/remove"
                return 6
            fi
        fi

        sleep 0.5

        # Execute bridge SBR
        if ! do_bridge_sbr "$card_id"; then
            err "[$card_id] SBR failed on bridge $bridge_bdf."
            return 6
        fi

        # Rescan PCI bus to discover restored hardware
        log "  [$card_id] Rescanning PCI bus..."
        if ! (echo 1 > /sys/bus/pci/rescan 2>/dev/null); then
            err "[$card_id] Failed to write 1 to /sys/bus/pci/rescan"
            return 6
        fi

        # Wait for driver to bind and sysfs/DRM structures to re-populate
        sleep 2
    fi

    # Step 4: Verify GPU presence and DRM nodes
    log "  [$card_id] Verifying post-reset state..."
    if [ ! -d "/sys/bus/pci/devices/$gpu_bdf" ]; then
        err "[$card_id] GPU BDF $gpu_bdf did not reappear in PCI sysfs."
        return 6
    fi

    local drm_found=0
    local card_node=""
    if [ -d "/sys/bus/pci/devices/$gpu_bdf/drm" ]; then
        for entry in "/sys/bus/pci/devices/$gpu_bdf/drm"/card[0-9]*; do
            if [ -d "$entry" ] && [ ! -L "$entry" ]; then
                card_node=$(basename "$entry")
                drm_found=1
                break
            fi
        done
    fi

    if [ "$drm_found" -eq 0 ]; then
        err "[$card_id] GPU $gpu_bdf present in PCI but DRM card node failed to initialize."
        return 6
    fi

    # Check GPU busy percent: must be 0 after reset
    local busy_percent
    busy_percent=$(cat "/sys/bus/pci/devices/$gpu_bdf/drm/$card_node/device/gpu_busy_percent" 2>/dev/null || cat "/sys/bus/pci/devices/$gpu_bdf/gpu_busy_percent" 2>/dev/null || echo "unknown")
    log "  [$card_id] Verified DRM node $card_node present (gpu_busy_percent=$busy_percent)."

    if [ "$busy_percent" != "0" ]; then
        err "[$card_id] GPU busy percent is '$busy_percent' (expected '0' after recovery)."
        return 6
    fi

    # Step 5: DPM Restoration & Verification
    if ! restore_target_dpm "$card_id"; then
        err "[$card_id] Failed to restore and verify DPM clocks."
        return 7
    fi

    log "[$card_id] PCI/DRM recovery and DPM policy completed; loaded ALU throughput unverified."
    return 0
}

# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
if [ "$CLOCKS_ONLY" -eq 1 ]; then
    card_id="${TARGET_CARDS[0]}"
    gpu_bdf="${CARD_GPU[$card_id]}"
    if [ ! -d "/sys/bus/pci/devices/$gpu_bdf/drm" ]; then
        err "[$card_id] Kernel reset has not restored the DRM device; refusing clock writes."
        exit 7
    fi
    restore_target_dpm "$card_id" || exit 7
    log "[$card_id] Manual DPM policy applied without PCI mutation; loaded ALU throughput unverified."
    exit 0
fi

for c in "${TARGET_CARDS[@]}"; do
    rc=0
    recover_target "$c" || rc=$?
    if [ "$rc" -ne 0 ]; then
        err "Recovery failed for target $c (exit code $rc)."
        exit "$rc"
    fi
done

log "All specified targets (${TARGET_CARDS[*]}) processed successfully."
exit 0
