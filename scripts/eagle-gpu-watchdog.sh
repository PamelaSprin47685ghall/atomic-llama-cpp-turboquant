#!/usr/bin/env bash
# Eagle GPU Watchdog: Intelligent GPU Hang Detection & Single-Card Recovery Dispatcher
#
# Design principles:
# 1. Never automatically reset all five cards or invoke 'gpu-hard-unlock.sh all'.
# 2. Never treat high GPU utilization (gpu_busy_percent >= 99%) alone as a proven hang.
#    Issue per-card advisory warnings only.
# 3. Track new BDF-attributable timeouts, reset failures and reset completion via
#    monotonic kernel log timestamps; never race the driver's own reset.
# 4. Never match normal progress lines like 'GPU reset begin' as failure triggers.
# 5. Coalesce evidence per affected card, durably record diagnostic incident logs
#    before dispatching recovery.
# 6. Invoke fixed '/usr/local/bin/gpu-hard-unlock.sh cardN' with bounded execution
#    and cooldown; treat nonzero exit as recovery failure.
# 7. Provide safe unprivileged '--once --dry-run' mode and sourcing mode for testing.

set -euo pipefail

# Mapping: card identifier -> GPU PCI BDF
declare -A CARD_TO_BDF=(
    ["card1"]="0000:0f:00.0"
    ["card2"]="0000:09:00.0"
    ["card3"]="0000:0c:00.0"
    ["card4"]="0000:06:00.0"
    ["card5"]="0000:03:00.0"
)

# Mapping: GPU PCI BDF (and short format) -> card identifier
declare -A BDF_TO_CARD=(
    ["0000:0f:00.0"]="card1"
    ["0f:00.0"]="card1"
    ["0000:09:00.0"]="card2"
    ["09:00.0"]="card2"
    ["0000:0c:00.0"]="card3"
    ["0c:00.0"]="card3"
    ["0000:06:00.0"]="card4"
    ["06:00.0"]="card4"
    ["0000:03:00.0"]="card5"
    ["03:00.0"]="card5"
)

RESET_SCRIPT="/usr/local/bin/gpu-hard-unlock.sh"
INCIDENT_DIR="/var/log/gpu-incidents"
LOG_TAG="eagle-gpu-watchdog"
POLL_INTERVAL=5
COOLDOWN_SECONDS=60
HELPER_TIMEOUT=120
AUTO_RESET_GRACE=90
FAILED_RESET_GRACE=30
CLOCK_RETRY_WINDOW=60

# CLI flags
ONCE=0
DRY_RUN=0

log() {
    local ts
    ts=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$ts] $*" >&2
}

# Durable incident recording
save_incident_evidence() {
    local card="$1"
    local bdf="$2"
    local evidence="$3"
    local incident_time
    incident_time=$(date "+%Y%m%d_%H%M%S_%N")
    local incident_file
    if [ "$DRY_RUN" -eq 1 ]; then
        incident_file="$INCIDENT_DIR/incident_${card}_${incident_time}_dryrun.log"
        log "[DRY-RUN] Would save durable incident evidence for $card ($bdf) to $incident_file"
        return 0
    fi

    if ! mkdir -p "$INCIDENT_DIR" 2>/dev/null; then
        log "ERROR: Cannot create incident directory $INCIDENT_DIR. Aborting recovery to avoid flying blind."
        log "Evidence was: $evidence"
        return 1
    fi

    incident_file=$(mktemp "$INCIDENT_DIR/incident_${card}_${incident_time}_XXXXXX.log" 2>/dev/null || echo "")
    if [ -z "$incident_file" ]; then
        incident_file="$INCIDENT_DIR/incident_${card}_${incident_time}_$$.log"
    fi

    if ! {
        echo "Incident Timestamp: $(date --iso-8601=seconds 2>/dev/null || date)"
        echo "Target Card: $card"
        echo "Target BDF: $bdf"
        echo "--- Trigger Evidence ---"
        echo "$evidence"
        echo "--- System & PCI State ---"
        if [ -d "/sys/bus/pci/devices/$bdf" ]; then
            echo "gpu_busy_percent: $(cat "/sys/bus/pci/devices/$bdf/gpu_busy_percent" 2>/dev/null || echo 'N/A')"
            echo "mem_info_vram_used: $(cat "/sys/bus/pci/devices/$bdf/mem_info_vram_used" 2>/dev/null || echo 'N/A')"
        fi
    } > "$incident_file" 2>/dev/null; then
        log "ERROR: Failed to write incident file $incident_file. Aborting recovery to ensure durable diagnostics."
        return 1
    fi
    return 0
}

# Fetch kernel messages with monotonic timestamp
# Format expected: "[ <seconds>.<microseconds> ] message" or "[<seconds>.<microseconds>] message"
fetch_kernel_logs() {
    if command -v dmesg >/dev/null 2>&1; then
        # Try dmesg with monotonic timestamps
        local out
        if out=$(dmesg --time-format monotonic 2>/dev/null) && [ -n "$out" ]; then
            echo "$out"
            return 0
        elif out=$(dmesg 2>/dev/null) && [ -n "$out" ]; then
            echo "$out"
            return 0
        fi
    fi

    # Fallback to journalctl kernel messages with monotonic format
    if command -v journalctl >/dev/null 2>&1; then
        local jout
        if jout=$(journalctl -k -b -o short-monotonic --no-pager 2>/dev/null) && [ -n "$jout" ]; then
            echo "$jout"
            return 0
        fi
    fi

    return 1
}

# Parse monotonic timestamp from line (e.g. "[ 123.456789]" -> 123.456789)
extract_monotonic_timestamp() {
    local line="$1"
    if [[ "$line" =~ \[[[:space:]]*([0-9]+\.[0-9]+)\] ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Check if float a > b
time_greater() {
    local a="$1"
    local b="$2"
    local a_int="${a%%.*}"
    local b_int="${b%%.*}"
    a_int="${a_int:-0}"
    b_int="${b_int:-0}"
    if [ "$a_int" -gt "$b_int" ]; then
        return 0
    elif [ "$a_int" -lt "$b_int" ]; then
        return 1
    fi
    local a_frac="${a#*.}"
    local b_frac="${b#*.}"
    [ "$a" = "$a_frac" ] && a_frac="0"
    [ "$b" = "$b_frac" ] && b_frac="0"
    # Pad fractions to equal length (6 digits for microsecond precision)
    printf -v a_frac "%-6.6s" "${a_frac}000000"
    printf -v b_frac "%-6.6s" "${b_frac}000000"
    a_frac="${a_frac// /0}"
    b_frac="${b_frac// /0}"
    [ "$a_frac" -gt "$b_frac" ]
}

# Parse kernel logs for new amdgpu hardware errors strictly attributed to specific BDF
# Arguments:
#   $1: last_seen_timestamp (monotonic float)
# Outputs:
#   Stream of lines: "<card>|<bdf>|<timestamp>|<kind>|<raw_line>"
# Returns:
#   Updated highest timestamp seen via stdout / variable
scan_kernel_errors() {
    local last_ts="$1"
    local raw_logs
    if ! raw_logs=$(fetch_kernel_logs); then
        log "ERROR: Kernel diagnostics unavailable (neither dmesg nor journalctl accessible)."
        return 2
    fi

    # Single-pass pipeline filter across the entire raw_logs to avoid forking grep per line
    local filtered_logs
    # "GPU reset succeeded, trying to resume" is intermediate, not a usable
    # device. Only the final "GPU reset(N) succeeded!" closes an incident.
    filtered_logs=$(printf '%s\n' "$raw_logs" | grep -iE "amdgpu.*(ring.*timeout|\[drm:amdgpu_job_timedout\]|reset failed|GPU reset\([0-9]+\) succeeded|Failed to initialize parser)" || true)
    [ -z "$filtered_logs" ] && return 0

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        # Check timestamp
        local line_ts
        line_ts=$(extract_monotonic_timestamp "$line")
        if [ -z "$line_ts" ]; then
            log "ALERT (MISSING_TS): amdgpu hardware error without monotonic timestamp cursor, skipping automated recovery: $line"
            continue
        fi
        if [ -n "$last_ts" ]; then
            if ! time_greater "$line_ts" "$last_ts"; then
                # Already processed or older than watermark
                continue
            fi
        fi

        # Extract BDF if present
        local matched_bdf=""
        if [[ "$line" =~ ([0-9a-fA-F]{2,4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]) ]]; then
            matched_bdf="${BASH_REMATCH[1]}"
        fi

        if [ -z "$matched_bdf" ]; then
            log "ALERT (AMBIGUOUS): amdgpu hardware error without BDF attribution detected; requiring manual investigation, skipping reset: $line"
            echo "CURSOR|none|$line_ts|ignored|$line"
            continue
        fi

        # Normalize BDF if missing domain prefix (e.g. 0f:00.0 -> 0000:0f:00.0)
        local norm_bdf="$matched_bdf"
        if [ -n "$norm_bdf" ] && [ "${#norm_bdf}" -le 7 ]; then
            norm_bdf="0000:$norm_bdf"
        fi

        local matched_card="${BDF_TO_CARD[$norm_bdf]:-}"
        if [ -z "$matched_card" ]; then
            matched_card="${BDF_TO_CARD[$matched_bdf]:-}"
        fi

        if [ -n "$matched_card" ]; then
            local kind=timeout
            if [[ "$line" =~ [Rr]eset[[:space:]]+failed ]]; then
                kind=failure
            elif [[ "$line" =~ GPU[[:space:]]reset\([0-9]+\)[[:space:]]+succeeded ]]; then
                kind=success
            fi
            echo "$matched_card|$norm_bdf|$line_ts|$kind|$line"
        else
            log "Notice: amdgpu error detected on unmapped or non-target device ($matched_bdf), skipping reset: $line"
            echo "CURSOR|none|$line_ts|ignored|$line"
        fi
    done <<< "$filtered_logs"
}

# Record a busy sample for a card and trigger advisory warnings without reset
# No sysfs dependency in this function; safe for synthetic testing
record_busy_sample() {
    local card="$1"
    local busy="$2"
    local bdf="${CARD_TO_BDF[$card]:-unknown}"

    if [ "$busy" -ge 99 ]; then
        CARD_BUSY_CONSECUTIVE[$card]=$(( ${CARD_BUSY_CONSECUTIVE[$card]:-0} + 1 ))
        local count=${CARD_BUSY_CONSECUTIVE[$card]}
        # Emit advisory warning every 4 samples (approx 20s) without triggering reset
        if [ "$count" -ge 4 ] && [ $(( count % 4 )) -eq 0 ]; then
            log "ADVISORY: $card ($bdf) utilization high at $busy% for consecutive samples ($count). Legitimate compute load is not a reset trigger."
        fi
    else
        CARD_BUSY_CONSECUTIVE[$card]=0
    fi
}

# Scan GPU busy percentages for advisory reporting only
scan_gpu_busy_advisory() {
    for c in card1 card2 card3 card4 card5; do
        local bdf="${CARD_TO_BDF[$c]}"
        local busy_file="/sys/bus/pci/devices/$bdf/gpu_busy_percent"
        if [ -f "$busy_file" ]; then
            local busy
            busy=$(cat "$busy_file" 2>/dev/null || echo 0)
            record_busy_sample "$c" "$busy"
        else
            CARD_BUSY_CONSECUTIVE[$c]=0
        fi
    done
}

# Dispatch single card recovery
recover_card() {
    local card="$1"
    local bdf="$2"
    local evidence="$3"

    log "ALERT: Confirmed ring timeout / reset failure on $card ($bdf)."
    if ! save_incident_evidence "$card" "$bdf" "$evidence"; then
        log "ERROR: Could not durably save incident evidence for $card ($bdf). Skipping helper execution to avoid untracked mutation."
        return 1
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "[DRY-RUN] Action: Would execute $RESET_SCRIPT $card (coalesced trigger for $bdf)"
        return 0
    fi

    if [ ! -x "$RESET_SCRIPT" ]; then
        log "ERROR: Recovery script $RESET_SCRIPT is not executable. Aborting recovery attempt."
        return 1
    fi

    log "Executing recovery: $RESET_SCRIPT $card (bounded timeout ${HELPER_TIMEOUT}s)"
    local rc=0
    # Execute single card recovery with bounded execution, never all
    if ! command -v timeout >/dev/null 2>&1; then
        log "ERROR: 'timeout' binary unavailable. Refusing unbounded recovery execution; failing closed."
        return 127
    fi
    timeout "$HELPER_TIMEOUT" "$RESET_SCRIPT" "$card" || rc=$?

    if [ "$rc" -ne 0 ]; then
        log "ERROR: Recovery script $RESET_SCRIPT $card failed with exit code $rc. Retaining durable diagnostics."
        return "$rc"
    fi

    log "Recovery of $card completed successfully."
    return 0
}

# A successful kernel reset needs clocks restored, never a second PCI reset.
restore_clock_card() {
    local card="$1"
    if [ "$DRY_RUN" -eq 1 ]; then
        log "[DRY-RUN] Would restore manual/highest DPM on $card without PCI mutation"
        return 0
    fi
    if [ ! -x "$RESET_SCRIPT" ]; then
        log "ERROR: Recovery helper missing; cannot restore clocks on $card"
        return 1
    fi
    timeout 30 "$RESET_SCRIPT" --clocks-only "$card"
}

# Driver rebinds can revert the DPM policy without a BDF-attributable timeout.
# This is a policy signal, not proof of the GPU's effective compute throughput.
read_clock_policy() {
    local file="/sys/bus/pci/devices/${CARD_TO_BDF[$1]}/power_dpm_force_performance_level"
    local policy
    [ -r "$file" ] && IFS= read -r policy < "$file" || return 1
    printf '%s\n' "$policy"
}

# Single iteration of watchdog logic
# Returns 0 on success, >0 on failure/unrecoverable diagnostic error
run_watchdog_step() {
    scan_gpu_busy_advisory

    local scan_output
    if ! scan_output=$(scan_kernel_errors "$LAST_KERNEL_TIMESTAMP"); then
        return 2
    fi

    local max_ts="$LAST_KERNEL_TIMESTAMP"
    local now
    now=$(date +%s)
    local step_status=0

    if [ -n "$scan_output" ]; then
        while IFS='|' read -r card bdf ts kind raw_line; do
            [ -z "$card" ] && continue

            # Always advance watermark for any parseable monotonic timestamp (including CURSOR)
            if [ -n "$ts" ]; then
                if [ -z "$max_ts" ] || time_greater "$ts" "$max_ts"; then
                    max_ts="$ts"
                fi
            fi

            # If this is a cursor-advancing marker, do not queue recovery
            [ "$card" = "CURSOR" ] && continue

            # Strictly validate that card is one of card1..card5
            case "$card" in
                card1|card2|card3|card4|card5)
                    ;;
                *)
                    log "ERROR: Invalid card string '$card' from scanner output; ignoring."
                    continue
                    ;;
            esac
            if [ "$kind" = success ]; then
                unset "CARD_PENDING_SINCE[$card]" "CARD_PENDING_KIND[$card]" "CARD_PENDING_EVIDENCE[$card]"
                CARD_CLOCK_SINCE[$card]=$now
                log "Kernel reset succeeded on $card ($bdf); no PCI reset, restoring only clocks."
                continue
            fi
            if [ -z "${CARD_PENDING_SINCE[$card]:-}" ]; then
                CARD_PENDING_SINCE[$card]=$now
                CARD_PENDING_EVIDENCE[$card]="$raw_line"$'\n'
                if ! save_incident_evidence "$card" "$bdf" "${CARD_PENDING_EVIDENCE[$card]}"; then
                    step_status=1
                fi
                log "Kernel $kind on $card ($bdf); waiting for driver recovery before escalation."
            else
                CARD_PENDING_EVIDENCE[$card]+="$raw_line"$'\n'
            fi
            CARD_PENDING_KIND[$card]=$kind
            unset "CARD_CLOCK_SINCE[$card]"
        done <<< "$scan_output"
    fi

    if [ -n "$max_ts" ]; then
        LAST_KERNEL_TIMESTAMP="$max_ts"
    fi

    for card in card1 card2 card3 card4 card5; do
        [ -n "${CARD_PENDING_SINCE[$card]:-}" ] && continue
        [ -n "${CARD_CLOCK_SINCE[$card]:-}" ] && continue
        local policy
        if policy=$(read_clock_policy "$card") && [ "$policy" = auto ]; then
            CARD_CLOCK_SINCE[$card]=$now
            log "Clock policy reverted to auto on $card; restoring manual DPM without PCI reset."
        fi
    done

    for card in "${!CARD_CLOCK_SINCE[@]}"; do
        if restore_clock_card "$card"; then
            log "Manual DPM policy reapplied on $card; effective ALU throughput not verified."
            unset "CARD_CLOCK_SINCE[$card]"
        elif [ $(( now - CARD_CLOCK_SINCE[$card] )) -ge "$CLOCK_RETRY_WINDOW" ]; then
            log "ERROR: Failed to restore clocks on $card after ${CLOCK_RETRY_WINDOW}s; manual intervention required."
            unset "CARD_CLOCK_SINCE[$card]"
            step_status=1
        fi
    done

    # Kernel mode1 reset can follow even a ring-reset failure. Only escalate
    # after a bounded grace without a subsequent success event.
    for card in "${!CARD_PENDING_SINCE[@]}"; do
        local grace=$AUTO_RESET_GRACE
        [ "${CARD_PENDING_KIND[$card]}" = failure ] && grace=$FAILED_RESET_GRACE
        [ $(( now - CARD_PENDING_SINCE[$card] )) -lt "$grace" ] && continue
        local bdf="${CARD_TO_BDF[$card]}"
        local evidence="${CARD_PENDING_EVIDENCE[$card]}"
        local last_rec="${CARD_LAST_RECOVERY[$card]:-0}"
        local elapsed=$(( now - last_rec ))

        if [ "$elapsed" -lt "$COOLDOWN_SECONDS" ] && [ "$DRY_RUN" -eq 0 ]; then
            log "Notice: $card ($bdf) error occurred during cooldown period ($elapsed < ${COOLDOWN_SECONDS}s). Skipping redundant recovery."
            continue
        fi

        CARD_LAST_RECOVERY["$card"]=$now
        if ! recover_card "$card" "$bdf" "$evidence"; then
            step_status=1
        fi
        unset "CARD_PENDING_SINCE[$card]" "CARD_PENDING_KIND[$card]" "CARD_PENDING_EVIDENCE[$card]"
    done

    return "$step_status"
}

# Initialize state structures
declare -A CARD_BUSY_CONSECUTIVE=(
    ["card1"]=0
    ["card2"]=0
    ["card3"]=0
    ["card4"]=0
    ["card5"]=0
)

declare -A CARD_LAST_RECOVERY=(
    ["card1"]=0
    ["card2"]=0
    ["card3"]=0
    ["card4"]=0
    ["card5"]=0
)
declare -A CARD_PENDING_SINCE=()
declare -A CARD_PENDING_KIND=()
declare -A CARD_PENDING_EVIDENCE=()
declare -A CARD_CLOCK_SINCE=()

LAST_KERNEL_TIMESTAMP=""

# Main loop and execution entrypoint
watchdog_main() {
    # Initialize watermark with latest current timestamp so past history does not retrigger
    local init_logs
    if [ "$ONCE" -eq 0 ] && init_logs=$(fetch_kernel_logs 2>/dev/null); then
        local last_line
        last_line=$(echo "$init_logs" | tail -n 1)
        LAST_KERNEL_TIMESTAMP=$(extract_monotonic_timestamp "$last_line")
    fi

    log "Eagle GPU Watchdog initialized. Baseline timestamp: [${LAST_KERNEL_TIMESTAMP:-0}]."

    if [ "$ONCE" -eq 1 ]; then
        run_watchdog_step
        return $?
    fi

    while true; do
        run_watchdog_step || {
            local step_rc=$?
            if [ "$step_rc" -eq 2 ]; then
                log "CRITICAL: Kernel diagnostics inaccessible. Exiting to allow supervisor alert."
                exit 2
            fi
        }
        sleep "$POLL_INTERVAL"
    done
}

# Parse CLI options
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --once)
                ONCE=1
                shift
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--once --dry-run] [--dry-run]"
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Usage: $0 [--once --dry-run] [--dry-run]" >&2
                exit 1
                ;;
        esac
    done

    if [ "$ONCE" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
        echo "ERROR: --once without --dry-run is forbidden to prevent accidental execution against historical kernel logs." >&2
                exit 1
    fi
}

# Only execute if script is invoked directly, allowing sourcing as a library for testing
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    watchdog_main
fi
