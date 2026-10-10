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
EARLY_REBOOT_FILE="/var/lib/eagle-gpu-watchdog/last-early-reboot"
LOG_TAG="eagle-gpu-watchdog"
POLL_INTERVAL=5
COOLDOWN_SECONDS=60
HELPER_TIMEOUT=120
AUTO_RESET_GRACE=90
FAILED_RESET_GRACE=30
EARLY_REBOOT_COOLDOWN=86400
CLOCK_RETRY_WINDOW=60

# Clock policy reactions are driven by GPU log events only, and every write is rate limited:
# a burst of driver messages must not turn into a burst of sysfs writes, and a card that is
# already in the pinned state must not be written to at all (see clock_pin_is_correct).
CLOCK_PIN_COOLDOWN="${CLOCK_PIN_COOLDOWN:-60}"

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
    # Keep unattributed error timestamps in the cursor without treating normal
    # topology/initialization messages as hardware incidents. Never dispatch
    # recovery or clock writes unless the line maps to exactly one known BDF.
    filtered_logs=$(printf '%s\n' "$raw_logs" | grep -iE 'amdgpu' | grep -iE '([0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]|timeout|fail|error|fault|hang|lost from bus' || true)
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
            local kind=gpu_log
            if [[ "$line" == *"device lost from bus!"* ]]; then
                kind=bus_lost
            elif [[ "$line" =~ [Rr]eset[[:space:]]+failed|GPU[[:space:]]Recovery[[:space:]]Failed|GPU[[:space:]]reset[[:space:]]end[[:space:]]with[[:space:]]ret[[:space:]]=[[:space:]]*-[0-9]+ ]]; then
                kind=failure
            elif [[ "$line" =~ GPU[[:space:]]reset\([0-9]+\)[[:space:]]+succeeded ]]; then
                kind=success
            elif [[ "$line" =~ ring.*timeout|[Rr]ing.*[Tt]imed[[:space:]]out|amdgpu_job_timedout|[Ww]edged|GPU[[:space:]]reset[[:space:]]begin|Failed[[:space:]]to[[:space:]]initialize[[:space:]]parser ]]; then
                kind=timeout
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

    CARD_HELPER_ATTEMPTED[$card]=0
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
    CARD_HELPER_ATTEMPTED[$card]=1
    timeout "$HELPER_TIMEOUT" "$RESET_SCRIPT" "$card" || rc=$?

    if [ "$rc" -ne 0 ]; then
        log "ERROR: Recovery script $RESET_SCRIPT $card failed with exit code $rc. Retaining durable diagnostics."
        return "$rc"
    fi

    log "Recovery of $card completed successfully."
    return 0
}

# A failing helper may race a late kernel recovery. Readability, not idle utilization,
# shows whether a card is back; legitimate work on a recovered card must not cause a reboot.
card_operational_at_path() {
    local base="$1"
    local clock_line busy
    [ -d "$base/drm" ] && [ -r "$base/pp_dpm_sclk" ] && [ -r "$base/gpu_busy_percent" ] || return 1
    IFS= read -r clock_line < "$base/pp_dpm_sclk" && [ -n "$clock_line" ] || return 1
    IFS= read -r busy < "$base/gpu_busy_percent" && [[ "$busy" =~ ^[0-9]+$ ]] && [ "$busy" -le 100 ]
}

card_operational() {
    card_operational_at_path "/sys/bus/pci/devices/${CARD_TO_BDF[$1]}"
}

# An unrecoverable, BDF-attributed bus loss warrants one orderly reboot before the IPMI
# watchdog's ten-minute limit. Keep the hardware watchdog active as the last resort if
# the kernel or shutdown hangs. A durable 24-hour guard prevents a persistent PCI fault
# from turning into a reboot loop across boots.
request_early_reboot() {
    local card="$1" now previous marker tmp
    if [ "$DRY_RUN" -eq 1 ]; then
        log "[DRY-RUN] Would request early orderly reboot after unrecovered bus loss on $card"
        return 0
    fi
    now=$(date +%s)
    if [ -e "$EARLY_REBOOT_FILE" ]; then
        IFS=' ' read -r previous _ < "$EARLY_REBOOT_FILE" || true
        if ! [[ "$previous" =~ ^[0-9]+$ ]]; then
            log "ERROR: Invalid early-reboot guard; refusing automatic reboot: $EARLY_REBOOT_FILE"
            return 1
        fi
        if [ "$now" -ge "$previous" ] && [ $(( now - previous )) -lt "$EARLY_REBOOT_COOLDOWN" ]; then
            log "ERROR: Early reboot already requested within 24 hours; refusing a reboot loop for $card."
            return 1
        fi
    fi
    marker=$(dirname "$EARLY_REBOOT_FILE")
    if ! install -d -m 0700 "$marker" || ! tmp=$(mktemp "${EARLY_REBOOT_FILE}.XXXXXX"); then
        log "ERROR: Cannot save early-reboot guard; leaving IPMI watchdog as fallback."
        return 1
    fi
    if ! printf '%s %s\n' "$now" "$card" > "$tmp" || ! sync -f "$tmp" || ! mv -f "$tmp" "$EARLY_REBOOT_FILE" || ! sync -f "$EARLY_REBOOT_FILE"; then
        rm -f "$tmp"
        log "ERROR: Cannot persist early-reboot guard; leaving IPMI watchdog as fallback."
        return 1
    fi
    log "Unrecovered bus loss on $card: requesting orderly reboot; IPMI hardware watchdog remains enabled."
    if ! systemctl reboot --no-wall; then
        log "ERROR: Orderly reboot request failed; IPMI hardware watchdog remains the fallback."
        return 1
    fi
}

# A successful kernel reset needs clocks restored, never a second PCI reset.
# True when the card already is in the state a pin would produce: 'manual' plus the highest
# pp_dpm_sclk index marked selected.  Nothing is written when this returns true.
clock_pin_is_correct() {
    local base="/sys/bus/pci/devices/${CARD_TO_BDF[$1]}"
    local perf top sel
    [ -r "$base/power_dpm_force_performance_level" ] || return 1
    IFS= read -r perf < "$base/power_dpm_force_performance_level"
    [ "$perf" = manual ] || return 1
    [ -r "$base/pp_dpm_sclk" ] || return 1
    top=$(grep -oE '^[0-9]+' "$base/pp_dpm_sclk" | tail -1)
    [ -n "$top" ] || return 1
    sel=$(sed -n 's/^\([0-9]\+\):.*\*.*$/\1/p' "$base/pp_dpm_sclk" | tail -1)
    [ "$sel" = "$top" ]
}

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
    local now last
    now=$(date +%s)
    last=${CARD_CLOCK_LAST[$card]:-0}
    if [ "$last" -gt 0 ] && [ $(( now - last )) -lt "$CLOCK_PIN_COOLDOWN" ]; then
        log "Cooldown: $card was pinned $(( now - last ))s ago (< ${CLOCK_PIN_COOLDOWN}s); skipping write."
        return 0
    fi
    if clock_pin_is_correct "$card"; then
        log "Clock pin already correct on $card (manual + highest index); no write needed."
        CARD_CLOCK_LAST[$card]=$now
        return 0
    fi
    CARD_CLOCK_LAST[$card]=$now
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
            if [ "$kind" = gpu_log ]; then
                # A device with a pending failure cannot safely serve DPM sysfs reads.
                [ -n "${CARD_PENDING_SINCE[$card]:-}" ] && continue
                CARD_CLOCK_SINCE[$card]=$now
                log "GPU log event on $card ($bdf); pinning manual/highest DPM (no reset): $raw_line"
                continue
            fi
            if [ "$kind" = success ]; then
                unset "CARD_PENDING_SINCE[$card]" "CARD_PENDING_KIND[$card]" "CARD_PENDING_EVIDENCE[$card]" "CARD_DRIVER_RESET_SINCE[$card]"
                CARD_CLOCK_SINCE[$card]=$now
                log "Kernel reset succeeded on $card ($bdf); no PCI reset, restoring only clocks."
                continue
            fi
            if [ "$kind" = timeout ] && [[ "$raw_line" == *"GPU reset begin"* ]]; then
                CARD_DRIVER_RESET_SINCE[$card]=$now
            elif [ "$kind" = failure ]; then
                unset "CARD_DRIVER_RESET_SINCE[$card]"
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
            # Do not downgrade bus loss to a later generic timeout or a reset-progress line.
            if [ "$kind" = bus_lost ] || [ -z "${CARD_PENDING_KIND[$card]:-}" ]; then
                CARD_PENDING_KIND[$card]=$kind
            elif [ "$kind" = failure ] && [ "${CARD_PENDING_KIND[$card]}" != bus_lost ]; then
                CARD_PENDING_KIND[$card]=$kind
            fi
            unset "CARD_CLOCK_SINCE[$card]"
        done <<< "$scan_output"
    fi

    if [ -n "$max_ts" ]; then
        LAST_KERNEL_TIMESTAMP="$max_ts"
    fi

    for card in card1 card2 card3 card4 card5; do
        [ -n "${CARD_PENDING_SINCE[$card]:-}" ] && continue
        [ -n "${CARD_CLOCK_SINCE[$card]:-}" ] && continue
        local policy policy_ok=1
        # Healthy is 'high' (the vendor default) or 'manual' (pinned by the restore helper).  Both
        # measure the same on this host (40.15-40.71 vs 40.02-40.23 tok/s on a paired MTP-off arm),
        # so neither is treated as a fault.  The fault is the SMU governor dropping to a low-power
        # policy: 'auto', 'low' or 'battery'.  Treating non-'high' as reverted made this watchdog
        # fight its own restore helper (which pins 'manual'), and per the 2026-09-28 records the
        # overdrive writes that came with that fight are the suspected trigger for card 0000:09:00.0
        # collapsing to half ALU rate.  Restoring only ever pins a level now - never OD, never PCI.
        if policy=$(read_clock_policy "$card"); then
            case "$policy" in
                high|manual) policy_ok=1 ;;
                *) policy_ok=0 ;;
            esac
        else
            continue
        fi
        if [ "$policy_ok" -eq 0 ]; then
            CARD_CLOCK_SINCE[$card]=$now
            log "Clock policy is '$policy' on $card (expected 'high' or 'manual'); restoring the DPM level without PCI reset and without overdrive writes."
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
        if [ "${CARD_PENDING_KIND[$card]}" = failure ] || [ "${CARD_PENDING_KIND[$card]}" = bus_lost ]; then
            grace=$FAILED_RESET_GRACE
        fi
        # Measure driver grace from GPU reset begin, not the older first timeout:
        # a reset that starts late must not be raced by our PCI helper.
        if [ -n "${CARD_DRIVER_RESET_SINCE[$card]:-}" ]; then
            [ $(( now - CARD_DRIVER_RESET_SINCE[$card] )) -lt "$AUTO_RESET_GRACE" ] && continue
        fi
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
            if [ "${CARD_PENDING_KIND[$card]}" = bus_lost ]; then
                if [ "${CARD_HELPER_ATTEMPTED[$card]:-0}" != 1 ]; then
                    log "ERROR: No card-specific recovery attempt on $card; refusing host reboot."
                elif card_operational "$card"; then
                    log "Card $card is readable again after failed helper; suppressing host reboot."
                elif save_incident_evidence "$card" "$bdf" "Early reboot after failed single-card recovery: $evidence"; then
                    request_early_reboot "$card" || step_status=1
                else
                    log "ERROR: Unrecoverable $card, but incident evidence could not be saved; leaving IPMI watchdog as fallback."
                fi
            fi
        fi
        unset "CARD_PENDING_SINCE[$card]" "CARD_PENDING_KIND[$card]" "CARD_PENDING_EVIDENCE[$card]" "CARD_DRIVER_RESET_SINCE[$card]"
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
declare -A CARD_HELPER_ATTEMPTED=()
declare -A CARD_PENDING_SINCE=()
declare -A CARD_PENDING_KIND=()
declare -A CARD_PENDING_EVIDENCE=()
declare -A CARD_DRIVER_RESET_SINCE=()
declare -A CARD_CLOCK_SINCE=()
declare -A CARD_CLOCK_LAST=()

LAST_KERNEL_TIMESTAMP=""

# Main loop and execution entrypoint
watchdog_main() {
    # Initialize watermark with latest current timestamp so past history does not retrigger
    local init_logs
    if [ "$ONCE" -eq 0 ]; then
        if ! init_logs=$(fetch_kernel_logs 2>/dev/null); then
            log "CRITICAL: Cannot watermark kernel history; refusing to act on stale GPU faults."
            return 2
        fi
        local last_line="${init_logs##*$'\n'}"
        LAST_KERNEL_TIMESTAMP=$(extract_monotonic_timestamp "$last_line")
        if [ -z "$LAST_KERNEL_TIMESTAMP" ]; then
            log "CRITICAL: No monotonic kernel timestamp; refusing to act on stale GPU faults."
            return 2
        fi
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
