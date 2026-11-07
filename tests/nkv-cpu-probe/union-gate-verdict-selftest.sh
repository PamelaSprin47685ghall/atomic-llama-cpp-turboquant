#!/usr/bin/env bash
# union-gate-verdict-selftest.sh - falsifiability proof for the shell half of
# the tri-state gate contract.
#
# Model-free. No GGUF, no build, no ctest. It sources gate-verdict.sh and
# union-mainline-gate-r24o.sh, writes synthetic logs into a mktemp -d, drives
# the REAL check_four_matrix / check_bse / check_ntok / check_kernel_sel, and
# asserts the verdict, the exit status of gate_finish, and the summary shape.
#
# Cases:
#   1. green log set                       -> PASS     (0)
#   2. FOUR_MATRIX line deleted            -> UNPROVEN (2)
#   3. both arms rebuilt once              -> FAIL     (1)
#   4. bitwise=0/0 over a zero-output set  -> UNPROVEN (2)
#   5. FOUR_MASK_UNPROVEN ok=0             -> FAIL     (1)
#   6. a ^FAIL: line in a log              -> FAIL     (1)
#   7. mask_rows=0/0                       -> UNPROVEN (2)
#   8. build_n_outputs=0                   -> FAIL     (1)
#
# Cases 2, 4 and 7 are the empty-set shapes the audit found: a missing line
# read as a pass and an empty comparison read as agreement. Case 8 is the
# zero-output bind claim: a step that rebuilt at width 0 reports a build width
# of 0, and that is a FAIL, not an UNPROVEN. Case 5 is the opposite error: an
# arm that reported failure must stay FAIL, not become UNPROVEN.
#
# Exit status: 0 every case matched; 1 a case did not match; 2 the selftest
# could not run.
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=tests/nkv-cpu-probe/gate-verdict.sh
. "$here/gate-verdict.sh"
# shellcheck source=tests/nkv-cpu-probe/union-mainline-gate-r24o.sh
. "$here/union-mainline-gate-r24o.sh"

tmp=$(mktemp -d) || {
    echo "UNPROVEN: cannot create a temporary directory" >&2
    exit 2
}
trap 'rm -rf -- "$tmp"' EXIT

green=$tmp/green
mkdir -- "$green"

write_green() { # dir
    local dir=$1
    cat > "$dir/matrix.log" <<'EOF'
FOUR_MATRIX ok=1 cells=124 hash=0123456789abcdef rebuilds=0 stock_rebuilds=124 bitwise=123/123 nout_zero=1 mask_cells=124 mask_rows=124/124 mask_expect=124 mask_cells_all=124
FOUR_AXES n_kv=300 n_outputs=2 rs_z=0 n_tokens=2 reason=dense_model
FOUR_MASK_NEG ok=1
FOUR_MASK_ROW_NEG ok=1
FOUR_MASK_UNPROVEN ok=1
FOUR_END ok=1
EOF
    cat > "$dir/bse.log" <<'EOF'
BSE_KERNEL_SEL ok=1 reason=kernel_sel_len_indep
BSE_STOCK_ARM ok=1 binder_stats=0
BSE_SINGLE ok=1 rows=300/300 steps=300 on_decode_builds=1 off_decode_builds=300 rebuild_fallbacks=0 pad_fallbacks=0
BSE_MULTI ok=1 rows=260/260 steps=65 on_decode_builds=1 off_decode_builds=65 rebuild_fallbacks=0 pad_fallbacks=0
BSE_BITWISE ok=1 rows=560/560
BSE_NEG ok=1 differ=5
BSE_END ok=1
EOF
    cat > "$dir/ntok.log" <<'EOF'
NOUT_ZERO_BIND ok=1 bound_n_outputs=0 build_n_outputs=2 refused_nout=0 patch_binds=3 builds=2
PASS: checked_rows=500 expected=500
EOF
    cat > "$dir/kernel_sel.log" <<'EOF'
KERNEL_SEL_END ok=1 stock=12/12 stride=12/12 old=12/12
EOF
}

# Populate the fixture once. Every case copies this directory; without this
# call every case would run against an empty directory and the selftest would
# report the gate's missing-line UNPROVEN instead of exercising a check.
write_green "$green"

fresh_case() { # name
    rm -rf -- "$tmp/case"
    mkdir -- "$tmp/case"
    cp -- "$green"/* "$tmp/case"/
}

cases=0
failures=0

check_case() { # name want_name want_rc want_summary_re
    local name=$1 want_name=$2 want_rc=$3 want_re=$4
    local out rc got_name ok=1
    # The check functions print FAIL/UNPROVEN diagnostics to stderr; keep the
    # per-case line readable.
    check_all "$tmp/case" 2>/dev/null
    out=$(gate_finish UNION_MAINLINE_GATE 2>/dev/null)
    rc=$?
    got_name=$(gate_verdict_name "$gate_verdict")
    cases=$((cases + 1))
    if [[ $got_name != "$want_name" ]]; then ok=0; fi
    if [[ $rc -ne $want_rc ]]; then ok=0; fi
    if [[ ! $out =~ $want_re ]]; then ok=0; fi
    if [[ $ok -ne 1 ]]; then failures=$((failures + 1)); fi
    printf 'CASE %s verdict=%s rc=%d want=%s/%d ok=%d\n' \
        "$name" "$got_name" "$rc" "$want_name" "$want_rc" "$ok"
    if [[ $ok -ne 1 ]]; then
        printf '  summary: %s\n' "$out" >&2
    fi
}

# 1. the complete green set.
fresh_case
check_case green pass 0 '^UNION_MAINLINE_GATE ok=1$'

# 2. the FOUR_MATRIX line deleted. Nothing was compared, so the gate cannot
#    say the observation is wrong: UNPROVEN, not PASS and not FAIL.
fresh_case
grep -v '^FOUR_MATRIX ' "$tmp/case/matrix.log" > "$tmp/case/matrix.tmp"
mv -- "$tmp/case/matrix.tmp" "$tmp/case/matrix.log"
check_case no_matrix_line unproven 2 '^UNION_MAINLINE_GATE ok=2 reason=unproven:matrix_line$'

# 3. both arms rebuilt once: the stock arm rebuilt fewer times than cells.
fresh_case
sed -i 's/stock_rebuilds=124/stock_rebuilds=1/' "$tmp/case/matrix.log"
check_case both_arms_once fail 1 '^UNION_MAINLINE_GATE ok=0 reason=matrix_stock_rebuilds$'

# 4. the vacuous-set hole: one cell, zero outputs, bitwise=0/0. The equality
#    bitwise == cells - nout_zero holds over nothing. Nothing was compared, so
#    the floor makes it UNPROVEN, not FAIL: the gate cannot say the observation
#    is wrong.
fresh_case
sed -i 's/^FOUR_MATRIX .*$/FOUR_MATRIX ok=1 cells=1 hash=0123456789abcdef rebuilds=0 stock_rebuilds=1 bitwise=0\/0 nout_zero=1 mask_cells=1 mask_rows=1\/1 mask_expect=1 mask_cells_all=1/' \
    "$tmp/case/matrix.log"
check_case vacuous_bitwise unproven 2 '^UNION_MAINLINE_GATE ok=2 reason=unproven:matrix_bitwise_empty$'

# 5. the tri-state arm reported ok=0. An explicit failure is a FAIL, not the
#    UNPROVEN a missing line gives.
fresh_case
sed -i 's/^FOUR_MASK_UNPROVEN ok=1$/FOUR_MASK_UNPROVEN ok=0/' "$tmp/case/matrix.log"
check_case unproven_arm_failed fail 1 '^UNION_MAINLINE_GATE ok=0 reason=matrix_mask_unproven$'

# 6. a FAIL: line anywhere in a log is a contradiction with a green summary.
fresh_case
printf 'FAIL: synthetic\n' >> "$tmp/case/matrix.log"
check_case fail_line fail 1 '^UNION_MAINLINE_GATE ok=0 reason=matrix_fail_line$'

# 7. the mask census compared no row: mask_rows=0/0. The zero denominator is
#    the stronger statement and is checked before the per-cell floor, so the
#    first reason is matrix_mask_rows_zero. UNPROVEN, not FAIL: nothing was
#    compared.
fresh_case
sed -i 's/^FOUR_MATRIX .*$/FOUR_MATRIX ok=1 cells=124 hash=0123456789abcdef rebuilds=0 stock_rebuilds=124 bitwise=123\/123 nout_zero=1 mask_cells=124 mask_rows=0\/0 mask_expect=0 mask_cells_all=0/' \
    "$tmp/case/matrix.log"
check_case mask_rows_zero unproven 2 '^UNION_MAINLINE_GATE ok=2 reason=unproven:matrix_mask_rows_zero$'

# 8. the zero-output step rebuilt: build_n_outputs=0. The bind claim is
#    violated, so this is a FAIL. The PASS: line and the zero bound width
#    stay, so the only gap is the build width.
fresh_case
sed -i 's/^NOUT_ZERO_BIND .*$/NOUT_ZERO_BIND ok=1 bound_n_outputs=0 build_n_outputs=0 refused_nout=0 patch_binds=3 builds=3/' \
    "$tmp/case/ntok.log"
check_case ntok_build_width fail 1 '^UNION_MAINLINE_GATE ok=0 reason=ntok_build_width$'

if [[ $failures -eq 0 ]]; then
    printf 'UNION_GATE_VERDICT_SELFTEST ok=1 cases=%d\n' "$cases"
    exit 0
fi
printf 'UNION_GATE_VERDICT_SELFTEST ok=0 cases=%d failures=%d\n' "$cases" "$failures"
exit 1
