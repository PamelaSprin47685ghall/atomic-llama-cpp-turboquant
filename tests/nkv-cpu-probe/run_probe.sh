#!/usr/bin/env bash
# Run the CPU probe with unset, explicit off and on arms.
# CPU only: no device is opened; HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES are
# cleared for every run.
#
# Usage: run_probe.sh [all|nkv|cap|seq]
#   nkv : n_ctx=512, n_ctx_kv=0, unified, 512 x 1 token
#   cap : n_ctx=300, n_ctx_kv=300, unified, 300 x 1 token
#   seq : n_ctx=1001, n_seq_max=2, non-unified; tests the stream guard
#   Arms: unset, off (=0), on (=1)
#
# Raw evidence is never deleted:
#   out/<arm>-<mode>.bin            logits records
#   logs/<arm>-<mode>.stderr.log    full probe stderr (STEP/RUN_PERF/stats)
#   out/PROBE-SUMMARY.txt           run + compare + check transcript
#   out/PROBE-TABLE.tsv             one row per arm/mode
#   out/stats-<mode>.diff           unset vs explicit-on NKV_BINDER_STATS diff
set -uo pipefail

WT=${WT:?set WT to the work directory}
D=$(dirname "$(realpath "$0")")
RUN_ROOT=${RUN_ROOT:-$WT/probe}
PROBE=${PROBE:-$RUN_ROOT/padmax_cpu_probe}
LIB=${LIB:-$WT/build/bin}
MODEL=${MODEL:?set MODEL to a CPU-compatible GGUF file}
MODE=${1:-all}

export HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
ulimit -c 0

if [ ! -x "$PROBE" ]; then
    echo "missing probe: $PROBE (build it first; see build_probe.sh)" >&2
    exit 2
fi
if [ ! -e "$LIB/libllama.so" ]; then
    echo "missing library: $LIB/libllama.so (set LIB=...)" >&2
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "missing model: $MODEL (set MODEL=...)" >&2
    exit 2
fi

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT=$RUN_ROOT/out/$RUN_ID
LOGS=$RUN_ROOT/logs/$RUN_ID
mkdir -p "$OUT" "$LOGS"
SUMMARY=$OUT/PROBE-SUMMARY.txt
TABLE=$OUT/PROBE-TABLE.tsv
: > "$SUMMARY"
: > "$TABLE"
{
    echo "# $(date -Is) padmax CPU probe"
    echo "run_id=$RUN_ID"
    echo "out=$OUT"
    echo "logs=$LOGS"
    echo "worktree=$WT"
    echo "head=$(git -C "$WT/src" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty=$(git -C "$WT/src" status --porcelain 2>/dev/null | wc -l)"
    echo "model=$MODEL"
    echo "model_sha256=$(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "lib=$LIB/libllama.so"
    echo "lib_sha256=$(sha256sum "$LIB/libllama.so" | cut -d' ' -f1)"
    echo "probe=$PROBE"
    echo "probe_sha256=$(sha256sum "$PROBE" | cut -d' ' -f1)"
    printf 'env_hip_visible_devices=[%s]\n' "$HIP_VISIBLE_DEVICES"
    printf 'env_rocr_visible_devices=[%s]\n' "$ROCR_VISIBLE_DEVICES"
} | tee -a "$SUMMARY"

run_arm() {
    local arm=$1 mode=$2
    local bin=$OUT/${arm}-${mode}.bin
    local err=$LOGS/${arm}-${mode}.stderr.log
    local rc
    case "$arm" in
        unset)
            env -u GGML_TP5_UNION_BINDER -u GGML_TP5_RSZ_RT \
                "$PROBE" "$MODEL" "$bin" "$mode" >"$err" 2>&1
            rc=$? ;;
        off)
            env -u GGML_TP5_RSZ_RT \
                GGML_TP5_UNION_BINDER=0 "$PROBE" "$MODEL" "$bin" "$mode" >"$err" 2>&1
            rc=$? ;;
        on)
            env -u GGML_TP5_RSZ_RT \
                GGML_TP5_UNION_BINDER=1 \
                "$PROBE" "$MODEL" "$bin" "$mode" >"$err" 2>&1
            rc=$? ;;
        *) echo "unknown arm $arm" >&2; return 2 ;;
    esac
    echo "RUN arm=$arm mode=$mode rc=$rc bin=$bin stderr=$err" | tee -a "$SUMMARY"
    grep -E '^(MODE_SPEC|CTX_API|RUN_PERF|RUN_END|HARNESS_END)' "$err" \
        | sed "s|^|  [$arm/$mode] |" | tee -a "$SUMMARY"
    grep -E '^(STEP=0 |STEP=255 |STEP=256 |STEP=257 |STEP=299 |STEP=500 |STEP=511 |STEP=512 |STEP=999 |STEP=1000 )' "$err" \
        | sed "s|^|  [$arm/$mode] |" >> "$SUMMARY"
    grep -E 'NKV_BINDER_STATS' "$err" \
        | sed "s|^|  [$arm/$mode] |" | tee -a "$SUMMARY"
    python3 "$D/check_probe.py" "$err" "$mode" "$arm" \
        | sed "s|^|  [$arm/$mode] |" | tee -a "$SUMMARY"
    local check_rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ "$check_rc" -ne 0 ]; then
        return 1
    fi
    return 0
}

compare() {
    local mode=$1 left=$2 right=$3
    local out
    out=$(python3 "$D/compare_records.py" \
        "$OUT/${left}-${mode}.bin" "$OUT/${right}-${mode}.bin" 2>&1)
    local rc=$?
    echo "COMPARE mode=$mode ${left}_vs_${right} rc=$rc ${out}" | tee -a "$SUMMARY"
    return "$rc"
}

stats_line() {
    sed -n 's/.*\(NKV_BINDER_STATS.*\)/\1/p' "$1"
}

overall=0
modes=()
case "$MODE" in
    all) modes=(nkv cap seq) ;;
    nkv|cap|seq) modes=("$MODE") ;;
    *) echo "usage: $0 [all|nkv|cap|seq]" >&2; exit 64 ;;
esac

for mode in "${modes[@]}"; do
    for arm in unset off on; do
        run_arm "$arm" "$mode" || overall=1
    done
    if [ "$mode" = seq ]; then
        compare "$mode" unset off || overall=1
    else
        for left in unset off; do
            case "$left" in
                unset) rights="off on" ;;
                off) rights="on" ;;
            esac
            for right in $rights; do compare "$mode" "$left" "$right" || overall=1; done
        done
        # Both captures must be non-empty before the diff. Two empty captures
        # compare equal, which would read as "identical" over nothing.
        stats_unset=$(stats_line "$LOGS/unset-$mode.stderr.log")
        stats_on=$(stats_line "$LOGS/on-$mode.stderr.log")
        if [ -z "$stats_unset" ] || [ -z "$stats_on" ]; then
            echo "STATS mode=$mode unset_vs_on UNPROVEN (no NKV_BINDER_STATS line in one arm)" | tee -a "$SUMMARY"
            if [ "$overall" -eq 0 ]; then
                overall=2
            fi
        elif diff -u <(printf '%s\n' "$stats_unset") \
                    <(printf '%s\n' "$stats_on") \
                    > "$OUT/stats-unset-$mode.diff" 2>&1; then
            echo "STATS mode=$mode unset_vs_on PASS identical" | tee -a "$SUMMARY"
        else
            echo "STATS mode=$mode unset_vs_on FAIL" | tee -a "$SUMMARY"
            overall=1
        fi
    fi
done

python3 "$D/summarize.py" "$LOGS" "$TABLE" >> "$SUMMARY" 2>&1
sum_rc=$?
if [ "$sum_rc" -eq 2 ]; then
    if [ "$overall" -eq 0 ]; then
        overall=2
    fi
elif [ "$sum_rc" -ne 0 ]; then
    overall=1
fi

echo "OVERALL rc=$overall" | tee -a "$SUMMARY"
exit "$overall"
