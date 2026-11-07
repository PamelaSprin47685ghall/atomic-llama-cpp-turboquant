#!/usr/bin/env bash
# Run the fixed gate by default; variable is a separate token-width diagnostic.
set -uo pipefail
PROBE=${PROBE:?set PROBE to varlen_output_probe}
MODEL=${MODEL:?set MODEL to a tiny CPU LLAMA GGUF}
LIB=${LIB:?set LIB to the CPU-only shared-library directory}
OUT=${OUT:?set OUT to a new output directory}
export HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -c 0
mkdir -p "$OUT"
status=0
case ${1:-fixed} in
    fixed) modes=(fixed) ;;
    variable) modes=(variable) ;;
    all) modes=(fixed variable) ;;
    *) echo "usage: $0 [fixed|variable|all]" >&2; exit 2 ;;
esac
for mode in "${modes[@]}"; do
    "$PROBE" "$MODEL" "$mode" "$OUT/$mode-on.bin" "$OUT/$mode-off.bin" >"$OUT/$mode.log" 2>&1
    rc=$?
    if (( rc != 0 )); then
        # The probe ran and failed. That is a FAIL, not an unproven comparison.
        printf 'MODE=%s PROBE_RC=%d RECORD_COMPARE_RC=-1 LOG=%s\n' "$mode" "$rc" "$OUT/$mode.log"
        status=1
        continue
    fi
    # Two empty record files compare equal. Require a record in both before
    # cmp, so a probe that wrote nothing cannot read as agreement.
    if [ ! -s "$OUT/$mode-on.bin" ] || [ ! -s "$OUT/$mode-off.bin" ]; then
        printf 'MODE=%s PROBE_RC=0 RECORD_COMPARE_RC=2 UNPROVEN=no-record-file LOG=%s\n' \
            "$mode" "$OUT/$mode.log"
        if (( status == 0 )); then status=2; fi
        continue
    fi
    if cmp -s "$OUT/$mode-on.bin" "$OUT/$mode-off.bin"; then cmp_rc=0; else cmp_rc=$?; fi
    printf 'MODE=%s PROBE_RC=%d RECORD_COMPARE_RC=%d LOG=%s\n' "$mode" "$rc" "$cmp_rc" "$OUT/$mode.log"
    if (( cmp_rc != 0 )); then status=1; fi
done
printf 'RUN_RC=%d\n' "$status"
exit "$status"
