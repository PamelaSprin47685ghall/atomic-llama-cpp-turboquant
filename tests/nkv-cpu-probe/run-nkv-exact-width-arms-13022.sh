#!/usr/bin/env bash
# Run test-nkv-exact-width once per binder arm, then once per negative arm,
# and write a JSON report.
#
# The arm belongs to this script, not to the test. Each arm is one process and
# the gate is injected with `env`; the test never sets, unsets or overrides it.
# The test prints the arm it was given as
#   NKV_ARM binder=<unset|0|1> ntok=<...> reuse_disable=<...> negative=<...>
# and runs the case set that arm names, so the three arms are provably
# distinct runs rather than three copies of one path:
#
#   unset : GGML_TP5_UNION_BINDER unset (the qualified CPU default)
#           -> cases single, multi_seq, multi_stream, widths
#   off   : GGML_TP5_UNION_BINDER=0 (the stock control)
#           -> case off
#   on    : GGML_TP5_UNION_BINDER=1 (explicit opt-in)
#           -> cases single, multi_seq, multi_stream, widths
#
# Negative arms. Each drops one observation from the test's eval callback and
# MUST record that it is missing: the named "FAIL: missing tensor at step=0"
# line with that root's flags (or the named "FAIL: missing pad tail" line for
# negative-pad-tail), then NKV_NEGATIVE ok=1 omitted=<root>, then exit 0. A
# negative arm that passes without that record is a failure of the gate, not a
# pass, and so is a nonzero exit: a run that died at model load proves nothing
# about the observation.
#
#   negative-mask     : --negative omit-mask,     binder unset
#   negative-k        : --negative omit-k,        binder unset
#   negative-v        : --negative omit-v,        binder unset
#   negative-pad-tail : --negative omit-pad-tail, binder unset
#
# Optional: ARMS="unset off on negative-mask negative-k negative-v
# negative-pad-tail" (default). An arm left out of ARMS is reported NOT_RUN,
# never PASS.
#
# Every arm also runs with HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES empty
# (CPU only) and with GGML_TP5_RSZ_RT unset, so the forbidden promotion path
# can never appear. LLAMA_EXPERIMENTAL_VARLEN_NTOK and
# LLAMA_GRAPH_REUSE_DISABLE are unset in every arm: this is the non-token
# lane, and the reuse key must be the default one.
#
# Usage:
#   BIN=/path/to/test-nkv-exact-width \
#   MODEL=/path/to/tiny-dense-llama.gguf \
#   RUN_DIR=/path/to/new-dir \
#       bash tests/nkv-cpu-probe/run-nkv-exact-width-arms-13022.sh
#
# Optional: ARMS="unset off on negative-mask negative-k negative-v" (default).
# An arm left out of ARMS is reported NOT_RUN, never PASS.
#
# RUN_DIR must not exist. The script creates it and writes:
#   <arm>.log        stdout+stderr of that arm's process
#   <arm>.rc         that arm's exit code
#   report.json      nkv-exact-width-report.py output over the positive logs
#   negative.txt     one "<arm> PASS|FAIL <reason>" line per negative arm
#   identity.txt     revision, model sha256, host, date, nproc, ccache
#
# Exit status:
#   0  every arm ran, every positive rc is 0 and OK in the report, and every
#      negative arm failed for its own named reason
#   1  a real failure
#   3  NOT_RUN: BIN or MODEL missing, or an arm was skipped
#
# Nothing here fabricates a result. A missing input writes a report whose
# every arm is NOT_RUN with a reason, prints OVERALL NOT_RUN and exits 3.
set -uo pipefail

D=$(dirname "$(realpath "$0")")
REPORT=$D/nkv-exact-width-report.py

BIN=${BIN:-}
MODEL=${MODEL:-}
RUN_DIR=${RUN_DIR:-}
ARMS=${ARMS:-unset off on negative-mask negative-k negative-v negative-pad-tail}

# The case list and step counts are read from tests/test-nkv-exact-width.cpp:
# single 300, multi_seq 9 (8 decodes + 1 after seq_rm), multi_stream 4,
# widths 257, off 300. A different count is a failure, not a new baseline.
# The expected list depends on the arm, because the arm selects the case set.
EXPECT_CASES_POSITIVE="single,multi_seq,multi_stream,widths"
EXPECT_STEPS_POSITIVE="300,9,4,257"
EXPECT_CASES_OFF="off"
EXPECT_STEPS_OFF="300"

usage() {
    echo "usage: BIN=<test-nkv-exact-width> MODEL=<gguf> RUN_DIR=<new dir> $0" >&2
}

is_negative() { # arm
    case $1 in
        negative-mask|negative-k|negative-v|negative-pad-tail) return 0 ;;
        *) return 1 ;;
    esac
}

negative_target() { # arm
    case $1 in
        negative-mask)     echo omit-mask ;;
        negative-k)        echo omit-k ;;
        negative-v)        echo omit-v ;;
        negative-pad-tail) echo omit-pad-tail ;;
        *) return 1 ;;
    esac
}

# The root name the test prints in its named failure line for a target.
negative_root() { # arm
    case $1 in
        negative-mask)     echo attn_inp_kq_mask ;;
        negative-k)        echo union_nkv_k ;;
        negative-v)        echo union_nkv_v ;;
        negative-pad-tail) echo reduce_pad_tail ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------- NOT_RUN
# Write a report in which every arm is NOT_RUN for the given reason. Used
# only when the run cannot start at all.
not_run_report() { # reason out
    local reason=$1 out=$2 arm
    {
        printf '{\n  "arms": {\n'
        local first=1
        for arm in $ARMS; do
            [ $first -eq 1 ] || printf ',\n'
            first=0
            printf '    "%s": {"arm": "%s", "status": "NOT_RUN", "ran": false, "reason": "%s"}' \
                "$arm" "$arm" "$reason"
        done
        printf '\n  },\n  "summary": {\n'
        first=1
        for arm in $ARMS; do
            [ $first -eq 1 ] || printf ',\n'
            first=0
            printf '    "%s": "NOT_RUN"' "$arm"
        done
        printf '\n  },\n  "overall": {"ran": [], "not_run": ['
        first=1
        for arm in $ARMS; do
            [ $first -eq 1 ] || printf ', '
            first=0
            printf '"%s"' "$arm"
        done
        printf '], "ok": false}\n}\n'
    } > "$out"
}

if [ -z "$BIN" ] || [ -z "$MODEL" ] || [ -z "$RUN_DIR" ]; then
    usage
    echo "OVERALL NOT_RUN: BIN, MODEL and RUN_DIR are required"
    exit 3
fi

if [ -e "$RUN_DIR" ]; then
    echo "FAIL: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi

mkdir -- "$RUN_DIR" || { echo "FAIL: cannot create RUN_DIR" >&2; exit 2; }
run_dir=$(cd -- "$RUN_DIR" && pwd -P)

# ---------------------------------------------------------------- inputs
# A missing binary or model is NOT_RUN, not a failure of the code under
# test. Report it honestly and stop.
missing=""
[ -x "$BIN" ]         || missing="$missing BIN($BIN)"
[ -f "$MODEL" ] && [ -s "$MODEL" ] || missing="$missing MODEL($MODEL)"
if [ -n "$missing" ]; then
    not_run_report "input missing:$missing" "$run_dir/report.json"
    echo "OVERALL NOT_RUN: input missing:$missing"
    exit 3
fi

model=$(realpath -- "$MODEL")
bin=$(realpath -- "$BIN")

# ---------------------------------------------------------------- identity
{
    printf 'bin=%s\n' "$bin"
    printf 'bin_sha256='; sha256sum "$bin" | cut -d' ' -f1
    printf 'model_path=%s\n' "$model"
    printf 'model_sha256='; sha256sum "$model" | cut -d' ' -f1
    printf 'revision='; git -C "$D/../.." rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
    printf 'branch=';   git -C "$D/../.." rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unavailable\n'
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
    printf 'arms=%s\n' "$ARMS"
} > "$run_dir/identity.txt"

# ---------------------------------------------------------------- arms
# One real process per arm, each with a stripped environment. `env` does the
# stripping; there is no eval and no shell re-parsing of the arguments. The
# test reads the gate once at start and prints it back as NKV_ARM; check_arm
# below refuses a log whose NKV_ARM line does not name the arm that ran.
run_arm() { # arm
    local arm=$1 rc log=$run_dir/$1.log target
    case "$arm" in
        unset)
            env -u GGML_TP5_UNION_BINDER -u LLAMA_EXPERIMENTAL_VARLEN_NTOK \
                -u LLAMA_GRAPH_REUSE_DISABLE -u GGML_TP5_RSZ_RT \
                HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
                "$bin" "$model" > "$log" 2>&1
            rc=$? ;;
        off)
            env -u GGML_TP5_UNION_BINDER -u LLAMA_EXPERIMENTAL_VARLEN_NTOK \
                -u LLAMA_GRAPH_REUSE_DISABLE -u GGML_TP5_RSZ_RT \
                HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
                GGML_TP5_UNION_BINDER=0 \
                "$bin" "$model" > "$log" 2>&1
            rc=$? ;;
        on)
            env -u GGML_TP5_UNION_BINDER -u LLAMA_EXPERIMENTAL_VARLEN_NTOK \
                -u LLAMA_GRAPH_REUSE_DISABLE -u GGML_TP5_RSZ_RT \
                HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
                GGML_TP5_UNION_BINDER=1 \
                "$bin" "$model" > "$log" 2>&1
            rc=$? ;;
        negative-mask|negative-k|negative-v|negative-pad-tail)
            target=$(negative_target "$arm") || return 2
            env -u GGML_TP5_UNION_BINDER -u LLAMA_EXPERIMENTAL_VARLEN_NTOK \
                -u LLAMA_GRAPH_REUSE_DISABLE -u GGML_TP5_RSZ_RT \
                HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
                "$bin" "$model" --negative "$target" > "$log" 2>&1
            rc=$? ;;
        *)
            echo "FAIL: unknown arm '$arm'" >&2
            return 2 ;;
    esac
    printf '%d\n' "$rc" > "$run_dir/$arm.rc"
    echo "RUN arm=$arm rc=$rc log=$log"
    return 0
}

# The arm a log actually took. The test prints exactly one NKV_ARM line; a
# missing or duplicated line is a failure, because then nothing in the log
# proves which path ran.
check_arm() { # arm log  -> 0 when the log's NKV_ARM line names this arm
    local arm=$1 log=$2 want binder n
    case "$arm" in
        unset)         want=unset ;;
        off)           want=0 ;;
        on)            want=1 ;;
        negative-*)    want=unset ;;
        *)             return 1 ;;
    esac
    n=$(grep -c '^NKV_ARM ' "$log" 2>/dev/null || true)
    if [ "${n:-0}" -ne 1 ]; then
        echo "ARM $arm: found ${n:-0} NKV_ARM lines, want 1" >&2
        return 1
    fi
    binder=$(sed -n 's/^NKV_ARM binder=\([^ ]*\) .*/\1/p' "$log")
    if [ "$binder" != "$want" ]; then
        echo "ARM $arm: log says binder=$binder, want $want" >&2
        return 1
    fi
    return 0
}

for arm in $ARMS; do
    run_arm "$arm" || exit 2
done

# ---------------------------------------------------------------- negative
# A negative arm passes only when the process exited 0 AND the log carries the
# named record for the root it omitted: the "FAIL: missing tensor at step=0"
# line with that root's flags, and the NKV_NEGATIVE ok=1 line. The exit code
# alone is not proof: a model-load failure, a context failure or a wrong arm
# also exits nonzero, and none of them shows that the gate can tell a missing
# observation from a present one. The test itself refuses to print
# NKV_NEGATIVE unless it recorded the missing observation, so the line is the
# evidence and the exit code only says the run finished.
negative_rc=0
: > "$run_dir/negative.txt"
for arm in $ARMS; do
    is_negative "$arm" || continue
    root=$(negative_root "$arm") || { echo "FAIL: unknown negative arm $arm" >&2; exit 2; }
    # the named failure line for this root. The three n_kv roots print a
    # mask/k/v flag tuple; the pad tail prints its own line.
    case $root in
        attn_inp_kq_mask) want_line="^FAIL: missing tensor at step=0 mask=1 k=0 v=0 " ;;
        union_nkv_k)      want_line="^FAIL: missing tensor at step=0 mask=0 k=1 v=0 " ;;
        union_nkv_v)      want_line="^FAIL: missing tensor at step=0 mask=0 k=0 v=1 " ;;
        reduce_pad_tail)  want_line="^FAIL: missing pad tail at step=0 (negative: omitted reduce_pad_tail " ;;
        *)                want_line="" ;;
    esac
    rc=$(cat "$run_dir/$arm.rc")
    log=$run_dir/$arm.log
    reason=""
    if [ "$rc" -ne 0 ]; then
        reason="exit $rc: the negative run did not report a detected omission"
    elif [ -z "$want_line" ]; then
        reason="unknown root $root"
    elif ! grep -q -- "$want_line" "$log"; then
        reason="no '${want_line}' line"
    elif ! grep -q "^NKV_NEGATIVE ok=1 omitted=${root}$" "$log"; then
        reason="no NKV_NEGATIVE ok=1 omitted=${root} line"
    elif grep -q "negative run omitting ${root} passed" "$log"; then
        reason="the omission went undetected"
    elif ! check_arm "$arm" "$log"; then
        reason="NKV_ARM line does not name the arm"
    fi
    if [ -z "$reason" ]; then
        printf '%s PASS omitted=%s rc=%s\n' "$arm" "$root" "$rc" >> "$run_dir/negative.txt"
    else
        printf '%s FAIL omitted=%s rc=%s reason=%s\n' "$arm" "$root" "$rc" "$reason" \
            >> "$run_dir/negative.txt"
        negative_rc=1
    fi
done
cat "$run_dir/negative.txt"

# ---------------------------------------------------------------- report
# The report covers the positive arms only: a negative log is expected to
# carry a FAIL: line and a nonzero rc, so feeding it to the report would
# report a correct negative run as a broken arm.
#
# Each positive arm is reported on its own, because the arm decides the case
# list: `off` runs the stock control, every other positive arm runs the four
# binder cases. One report per arm keeps the expected list exact.
positive_arms=0
for arm in $ARMS; do
    is_negative "$arm" && continue
    positive_arms=$((positive_arms + 1))
    if ! check_arm "$arm" "$run_dir/$arm.log"; then
        echo "ARM $arm: the log does not prove the arm it ran" >&2
        exit 1
    fi
done

if [ "$positive_arms" -eq 0 ]; then
    echo "OVERALL NOT_RUN: no positive arm in ARMS"
    exit 3
fi

report_rc=0
for arm in $ARMS; do
    is_negative "$arm" && continue
    if [ "$arm" = off ]; then
        cases=$EXPECT_CASES_OFF steps=$EXPECT_STEPS_OFF
    else
        cases=$EXPECT_CASES_POSITIVE steps=$EXPECT_STEPS_POSITIVE
    fi
    python3 "$REPORT" --out "$run_dir/report-$arm.json" \
        --expect-cases "$cases" --expect-steps "$steps" \
        --arm "$arm=$run_dir/$arm.log" \
        --arm-rc "$arm=$(cat "$run_dir/$arm.rc")"
    case $? in
        0) ;;
        3) [ $report_rc -eq 1 ] || report_rc=3 ;;
        *) report_rc=1 ;;
    esac
done

# An arm left out of ARMS never reaches the report at all, so check for it
# explicitly and fail as NOT_RUN rather than pass silently.
overall=0
for arm in unset off on negative-mask negative-k negative-v negative-pad-tail; do
    case " $ARMS " in
        *" $arm "*) ;;
        *)
            echo "ARM $arm NOT_RUN: not in ARMS"
            [ $overall -eq 3 ] || overall=3
            ;;
    esac
done

case "$report_rc" in
    0) ;;
    3) [ $overall -eq 1 ] || overall=3 ;;
    *) overall=1 ;;
esac
[ "$negative_rc" -eq 0 ] || overall=1

# Exit 0 only when every positive arm ran with rc 0.
if [ $overall -eq 0 ]; then
    for arm in $ARMS; do
        is_negative "$arm" && continue
        if [ "$(cat "$run_dir/$arm.rc")" != "0" ]; then
            overall=1
        fi
    done
fi

case "$overall" in
    0) echo "OVERALL PASS" ;;
    3) echo "OVERALL NOT_RUN" ;;
    *) echo "OVERALL FAIL" ;;
esac
exit "$overall"
