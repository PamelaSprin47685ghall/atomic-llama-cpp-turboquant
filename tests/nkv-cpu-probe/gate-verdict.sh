# shellcheck shell=bash
#
# gate-verdict.sh - the shell half of the tri-state gate contract.
#
# Source this file. Do not execute it. It sets no shell options.
#
# One encoding, shared with enum llama_gate_verdict (include/llama.h):
#
#   GATE_FAIL     = 0   the check ran and the observation did not hold.
#   GATE_PASS     = 1   the check ran and the observation held.
#   GATE_UNPROVEN = 2   the check did not run, or compared nothing.
#
# Precedence is sticky: FAIL > UNPROVEN > PASS.
#   - gate_fail sets FAIL. Nothing lowers it.
#   - gate_unproven sets UNPROVEN unless the verdict is already FAIL.
#   - gate_pass promotes to PASS only from UNPROVEN. It never overrides FAIL.
#
# A missing line is UNPROVEN, not FAIL: nothing was compared, so the gate
# cannot say the observation is wrong. A line that parses but violates an
# equality is FAIL.
#
# gate_pass promotes only a CLEAN UNPROVEN: no failure and no unproven
# observation has been recorded. The guard is built in, so a block that calls
# gate_unproven and then gate_pass cannot erase the UNPROVEN. A call-site
# guard such as `if [[ -z $gate_reason ]]; then gate_pass; fi` is redundant,
# not wrong.
#
# The exit status of gate_finish is the verdict: 0 PASS, 1 FAIL, 2 UNPROVEN.

GATE_FAIL=0
GATE_PASS=1
GATE_UNPROVEN=2

gate_verdict_name() { # v
    case $1 in
        0) printf 'fail' ;;
        1) printf 'pass' ;;
        2) printf 'unproven' ;;
        *) printf 'unknown' ;;
    esac
}

gate_init() {
    gate_verdict=$GATE_UNPROVEN
    gate_reason=
}

gate_fail() { # slug what observed
    local slug=$1 what=$2 observed=$3
    printf 'FAIL: %s (%s)\n' "$what" "$observed" >&2
    gate_verdict=$GATE_FAIL
    if [[ -z $gate_reason ]]; then
        gate_reason=$slug
    fi
}

gate_unproven() { # slug what observed
    local slug=$1 what=$2 observed=$3
    printf 'UNPROVEN: %s (%s)\n' "$what" "$observed" >&2
    if [[ $gate_verdict -ne $GATE_FAIL ]]; then
        gate_verdict=$GATE_UNPROVEN
    fi
    if [[ -z $gate_reason ]]; then
        gate_reason=$slug
    fi
}

gate_pass() {
    # gate_reason is set by the first fail or unproven. A non-empty reason
    # means something was not proven, so the promotion is refused.
    if [[ $gate_verdict -eq $GATE_UNPROVEN && -z $gate_reason ]]; then
        gate_verdict=$GATE_PASS
    fi
}

gate_require_line() { # file ERE slug what
    local file=$1 re=$2 slug=$3 what=$4
    if ! grep -qE -- "$re" "$file" 2>/dev/null; then
        gate_unproven "$slug" "$what" "no line matching /$re/ in $file"
        return 1
    fi
    return 0
}

gate_require_absent() { # file ERE slug what
    local file=$1 re=$2 slug=$3 what=$4
    if grep -qE -- "$re" "$file" 2>/dev/null; then
        gate_fail "$slug" "$what" "line matching /$re/ present in $file"
        return 1
    fi
    return 0
}

gate_finish() { # name [key=value ...]
    local name=$1
    shift
    # The exit status is not the enum value: PASS is 0, FAIL is 1, UNPROVEN
    # is 2. The enum keeps FAIL=0 and PASS=1 so the ok= field in the summary
    # matches the C++ verdict numbers.
    case $gate_verdict in
        "$GATE_PASS")
            printf '%s ok=1' "$name"
            if [[ $# -gt 0 ]]; then
                printf ' %s' "$@"
            fi
            printf '\n'
            return 0
            ;;
        "$GATE_FAIL")
            printf '%s ok=0 reason=%s\n' "$name" "${gate_reason:-unspecified}"
            return 1
            ;;
        *)
            printf '%s ok=2 reason=unproven:%s\n' "$name" "${gate_reason:-unspecified}"
            return 2
            ;;
    esac
}
