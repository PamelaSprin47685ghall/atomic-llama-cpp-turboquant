#!/usr/bin/env bash
# One ctest gate for the union mainline: four binaries, one verdict.
#
# Claims, and the line that carries each one:
#
#   1. four-parameter matrix   FOUR_MATRIX (cells, both rebuild counts, both
#                              bitwise counts, the mask census) plus FOUR_AXES
#                              (the n_kv / n_outputs / rs_z / n_tokens census)
#                              and FOUR_END.
#   2. mask                    the matrix's mask census is internally
#                              consistent, and all three negative arms fired:
#                              FOUR_MASK_NEG, FOUR_MASK_ROW_NEG and
#                              FOUR_MASK_UNPROVEN.
#   3. exact == stock          BSE_BITWISE, BSE_STOCK_ARM, BSE_NEG, BSE_END.
#   4. on_decode_builds == 1   BSE_SINGLE and BSE_MULTI, each with
#                              off_decode_builds >= steps and != 1 and both
#                              fallback counters at 0.
#
# Claim 3 has a precondition: BSE_KERNEL_SEL ok=1
# reason=kernel_sel_len_indep, and the model-free KERNEL_SEL_END with
# stock=a/r, stride=d/r, old=e/r where a == r > 0, d >= 1 and e >= 1.
#
# The binaries are run DIRECTLY, never through `ctest -V`. ctest prefixes every
# line of a test's own output with "<index>: ", and every grep below is
# anchored at the start of the line, so a ctest-mediated log cannot match by
# construction. Running the binaries here also fixes the environment: each one
# starts with the union/varlen switches unset and no GPU visible.
#
# Every check is fail-closed, and the verdict is tri-state. The encoding is
# shared with enum llama_gate_verdict (include/llama.h) and with
# gate-verdict.sh, which this file sources:
#
#   PASS (1)      every required line is present and every equality holds.
#   FAIL (0)      a line parsed and an equality did not hold, or a binary
#                 exited non-zero. The observation is wrong.
#   UNPROVEN (2)  a required line is missing or unparseable, or a binary,
#                 model or argument is missing. Nothing was compared, so the
#                 gate cannot say the observation is wrong.
#
# A missing line is UNPROVEN, never a skip and never a FAIL. A non-zero exit
# from a binary is a FAIL naming that binary and its log. There is no
# `|| true` and no `set +e` around a check. The numeric equalities are
# compared in shell from sed captures, because bash ERE has no backreferences.
#
# Usage:
#   union-mainline-gate-r24o.sh <four-param-matrix> <binder-stock-equiv> \
#                               <ntok-onegraph> <kernel-sel-len-indep> \
#                               <model.gguf> [rounds]
#
# rounds defaults to 2 when the sixth argument is absent or empty. The CMake
# registration passes ${LLAMA_NTOK_TEST_ROUNDS}, whose cache default is 50
# (tests/CMakeLists.txt), so through ctest the value is 50, not 2.
#
# Exit status, the same three values:
#   0  PASS: every claim passed; exactly one UNION_MAINLINE_GATE ok=1 line
#   1  FAIL: a claim failed; UNION_MAINLINE_GATE ok=0 reason=<slug>
#   2  UNPROVEN: a required line was missing or unparseable, or a binary,
#      model or argument was missing; UNION_MAINLINE_GATE ok=2
#      reason=unproven:<slug>. Nothing was compared, or the run could not
#      start. A caller must not read 2 as a failure of the code under test.
#
# Logs are written into a mktemp -d directory. It is removed on success and
# kept, with its path printed, on any other status.
set -uo pipefail

# The tri-state vocabulary. Resolve it from this file, so the selftest can
# source this script from any directory.
_gate_verdict_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=tests/nkv-cpu-probe/gate-verdict.sh
. "$_gate_verdict_dir/gate-verdict.sh"
unset _gate_verdict_dir

gate_init
logdir=
g_cells=
g_bit_a=
g_bit_b=
g_mcells=
g_mok=
g_mchk=
g_nout_zero=
g_bse_a=
g_bse_b=
g_s_off=
g_m_off=
g_nout_refused=
g_ks_a=
g_ks_r=

# ------------------------------------------------------------------- verdict
# Thin adapters over gate-verdict.sh. They keep the "union-mainline-gate: "
# prefix the existing messages carry. The verdict and the first slug live in
# the library's gate_verdict and gate_reason.
fail() { # slug what observed
    gate_fail "$1" "union-mainline-gate: $2" "$3"
}

unproven() { # slug what observed
    gate_unproven "$1" "union-mainline-gate: $2" "$3"
}

# ------------------------------------------------------------------- parsing
# A capture that does not match leaves the whole line in $value; the digit
# test then rejects it, so an absent field reads as empty instead of as a
# number, and the caller's emptiness check catches it.

num_field() { # line key -> digits, or empty
    local value
    value=$(printf '%s\n' "$1" | sed -E 's/^.* '"$2"'=([0-9]+)( .*)?$/\1/')
    if [[ $value =~ ^[0-9]+$ ]]; then
        printf '%s' "$value"
    fi
}

pair_field() { # line key -> "a b", or empty
    local value
    value=$(printf '%s\n' "$1" | sed -E 's/^.* '"$2"'=([0-9]+)\/([0-9]+)( .*)?$/\1 \2/')
    if [[ $value =~ ^[0-9]+[[:space:]][0-9]+$ ]]; then
        printf '%s' "$value"
    fi
}

last_line() { # file regex -> last matching line, or empty
    grep -E "$2" "$1" | sed -n '$p'
}

# --------------------------------------------------------------- claim 1 + 2
check_four_matrix() { # log
    local log=$1 line pair axes
    local cells stock rebuilds bit_a bit_b nout_zero
    local mcells mok mchk mexp mall
    local a_nout a_rsz a_ntok
    local re='^FOUR_MATRIX ok=1 cells=[1-9][0-9]* hash=[0-9a-f]{16} rebuilds=[0-9]+ stock_rebuilds=[0-9]+ bitwise=[0-9]+/[0-9]+ nout_zero=[0-9]+ mask_cells=[0-9]+ mask_rows=[0-9]+/[0-9]+ mask_expect=[0-9]+ mask_cells_all=[0-9]+$'

    gate_init

    line=$(last_line "$log" "$re")
    if [[ -z $line ]]; then
        unproven matrix_line "no FOUR_MATRIX ok=1 line in the pinned field order" "$log"
        return
    fi

    cells=$(num_field "$line" cells)
    stock=$(num_field "$line" stock_rebuilds)
    rebuilds=$(num_field "$line" rebuilds)
    pair=$(pair_field "$line" bitwise)
    bit_a=${pair% *}
    bit_b=${pair#* }
    nout_zero=$(num_field "$line" nout_zero)
    mcells=$(num_field "$line" mask_cells)
    pair=$(pair_field "$line" mask_rows)
    mok=${pair% *}
    mchk=${pair#* }
    mexp=$(num_field "$line" mask_expect)
    mall=$(num_field "$line" mask_cells_all)

    if [[ -z $cells || -z $stock || -z $rebuilds || -z $bit_a || -z $bit_b ||
          -z $nout_zero || -z $mcells || -z $mok || -z $mchk || -z $mexp || -z $mall ]]; then
        unproven matrix_fields "a FOUR_MATRIX field did not parse" "$line"
        return
    fi

    if [[ $rebuilds -ne 0 ]]; then
        fail matrix_rebuilds "the binder arm rebuilt the graph" "rebuilds=$rebuilds"
    fi
    if [[ $stock -lt $cells ]]; then
        fail matrix_stock_rebuilds "the stock arm rebuilt fewer times than cells" \
             "stock_rebuilds=$stock cells=$cells"
    fi
    if [[ $stock -eq $rebuilds ]]; then
        fail matrix_arms_same "both arms rebuilt the same number of times" \
             "rebuilds=$rebuilds stock_rebuilds=$stock"
    fi
    if [[ $bit_a -ne $bit_b ]]; then
        fail matrix_bitwise "the two arms did not agree on every compared cell" \
             "bitwise=$bit_a/$bit_b"
    fi
    if [[ $bit_b -ne $(( cells - nout_zero )) ]]; then
        fail matrix_bitwise_denominator "the bitwise denominator is not cells - nout_zero" \
             "bitwise=$bit_a/$bit_b cells=$cells nout_zero=$nout_zero"
    fi
    # The denominator is cells - nout_zero, so a run whose every cell is
    # zero-output compares 0/0 and passes the equality over nothing. The floor
    # makes that UNPROVEN: no cell was compared, so the gate cannot say the
    # observation is wrong.
    if [[ $bit_b -lt 1 ]]; then
        unproven matrix_bitwise_empty "the bitwise comparison compared no cell" \
             "bitwise=$bit_a/$bit_b cells=$cells nout_zero=$nout_zero"
    fi
    if [[ $nout_zero -lt 1 ]]; then
        unproven matrix_nout_zero "no zero-output cell ran" "nout_zero=$nout_zero"
    fi
    if [[ $mcells -ne $cells ]]; then
        unproven matrix_mask_cells "the mask read-back did not run on every cell" \
             "mask_cells=$mcells cells=$cells"
    fi
    # mask_rows=a/b is (rows that agreed)/(rows compared), so "a == b" is the
    # same equality as "mask_ok == mask_checked" under the two names the test
    # prints. One comparison, not two.
    if [[ $mok -ne $mchk ]]; then
        fail matrix_mask_rows "the mask read-back disagreed on some rows" \
             "mask_rows=$mok/$mchk"
    fi
    # "Compared no row" is the stronger statement and comes first: with a 0/0
    # census the floor below would fire too and hide the real reason. It is
    # UNPROVEN: nothing was compared.
    if [[ $mchk -lt 1 ]]; then
        unproven matrix_mask_rows_zero "the mask read-back compared no row" \
             "mask_rows_checked=$mchk"
    fi
    if [[ $mchk -ne $mexp ]]; then
        unproven matrix_mask_expect "the mask read-back compared a different number of rows than the batches own" \
             "mask_rows_checked=$mchk mask_expect=$mexp"
    fi
    if [[ $mchk -lt $mcells ]]; then
        unproven matrix_mask_floor "fewer mask rows compared than cells" \
             "mask_rows_checked=$mchk mask_cells=$mcells"
    fi
    if [[ $mall -ne $mcells ]]; then
        unproven matrix_mask_cells_all "not every cell compared every mask row" \
             "mask_cells_all=$mall mask_cells=$mcells"
    fi

    # The census is a count of distinct values, not a flag: n_outputs and
    # n_tokens must have moved, and rs_z is 0 on a dense model (inert, with a
    # reason) or >= 2 on a recurrent one. rs_z=1 is neither.
    axes=$(last_line "$log" '^FOUR_AXES n_kv=[1-9][0-9]* n_outputs=[1-9][0-9]* rs_z=[0-9]+ n_tokens=[1-9][0-9]* reason=[a-z_]+$')
    if [[ -z $axes ]]; then
        unproven matrix_axes "no FOUR_AXES line with the axis census" "$log"
    else
        a_nout=$(num_field "$axes" n_outputs)
        a_rsz=$(num_field "$axes" rs_z)
        a_ntok=$(num_field "$axes" n_tokens)
        if [[ -z $a_nout || -z $a_rsz || -z $a_ntok ]]; then
            unproven matrix_axes_fields "a FOUR_AXES field did not parse" "$axes"
        else
            if [[ $a_nout -lt 2 ]]; then
                unproven matrix_axes_nout "the n_outputs axis never moved" "n_outputs=$a_nout"
            fi
            if [[ $a_ntok -lt 2 ]]; then
                unproven matrix_axes_ntok "the n_tokens axis never moved" "n_tokens=$a_ntok"
            fi
            if [[ $a_rsz -eq 1 ]]; then
                unproven matrix_axes_rsz "rs_z=1 is not a census (0 = inert, >= 2 = moved)" \
                     "rs_z=$a_rsz"
            fi
        fi
    fi

    # Three arms, three evidence lines. FOUR_MASK_NEG is the conjunction of
    # the fill's width guard and the read-back's corrupted-row arm, so it
    # stays ok=1 when one arm is deleted. FOUR_MASK_ROW_NEG and
    # FOUR_MASK_UNPROVEN are printed only by their own arms, and they are what
    # goes missing. A missing line is UNPROVEN: the arm did not run.
    if ! grep -qE '^FOUR_MASK_NEG ok=1$' "$log"; then
        unproven matrix_mask_neg "no FOUR_MASK_NEG ok=1" "$log"
    fi
    if grep -qE '^FOUR_MASK_UNPROVEN ok=0$' "$log"; then
        fail matrix_mask_unproven "the tri-state arm reported ok=0" \
             "$(last_line "$log" '^FOUR_MASK_UNPROVEN ok=0$')"
    fi
    if ! grep -qE '^FOUR_MASK_ROW_NEG ok=1$' "$log"; then
        unproven matrix_mask_row_neg "no FOUR_MASK_ROW_NEG ok=1" "$log"
    fi
    if ! grep -qE '^FOUR_MASK_UNPROVEN ok=1$' "$log"; then
        unproven matrix_mask_unproven "no FOUR_MASK_UNPROVEN ok=1" "$log"
    fi
    if ! grep -qE '^FOUR_END ok=1$' "$log"; then
        unproven matrix_end "no FOUR_END ok=1" "$log"
    fi
    if grep -qE '^FAIL:' "$log"; then
        fail matrix_fail_line "a FAIL line is present" "$(last_line "$log" '^FAIL:')"
    fi

    g_cells=$cells
    g_bit_a=$bit_a
    g_bit_b=$bit_b
    g_mcells=$mcells
    g_mok=$mok
    g_mchk=$mchk
    g_nout_zero=$nout_zero

    # Promote only when this block recorded nothing. A missing line or a
    # failed equality leaves gate_reason set.
    if [[ -z $gate_reason ]]; then
        gate_pass
    fi
}

# ------------------------------------------------------------------- claim 3
# The precondition is read first: when kernel selection still depends on the
# runtime length, the equality below is not a claim the exact lane may make.
check_bse() { # log
    local log=$1 line pair
    local s_a s_b s_steps s_on s_off s_rf s_pf
    local m_a m_b m_steps m_on m_off m_rf m_pf
    local b_a b_b neg

    gate_init

    # An explicit ok=0 is a FAIL: the probe ran and answered "coupled". A
    # missing line is UNPROVEN: the probe did not answer.
    if grep -qE '^BSE_KERNEL_SEL ok=0 reason=kernel_sel_coupled$' "$log"; then
        fail bse_kernel_sel_coupled \
             "kernel selection still depends on the runtime length; the exact lane cannot be claimed" \
             "$log"
    fi
    if ! grep -qE '^BSE_KERNEL_SEL ok=1 reason=kernel_sel_len_indep$' "$log"; then
        unproven bse_kernel_sel "no BSE_KERNEL_SEL ok=1 reason=kernel_sel_len_indep" "$log"
    fi
    if ! grep -qE '^BSE_STOCK_ARM ok=1 binder_stats=0$' "$log"; then
        unproven bse_stock_arm "no BSE_STOCK_ARM ok=1 binder_stats=0" "$log"
    fi

    line=$(last_line "$log" '^BSE_SINGLE ok=1 rows=[0-9]+/[0-9]+ steps=[0-9]+ on_decode_builds=[0-9]+ off_decode_builds=[0-9]+ rebuild_fallbacks=[0-9]+ pad_fallbacks=[0-9]+$')
    if [[ -z $line ]]; then
        unproven bse_single "no BSE_SINGLE ok=1 line in the pinned field order" "$log"
    else
        pair=$(pair_field "$line" rows)
        s_a=${pair% *}
        s_b=${pair#* }
        s_steps=$(num_field "$line" steps)
        s_on=$(num_field "$line" on_decode_builds)
        s_off=$(num_field "$line" off_decode_builds)
        s_rf=$(num_field "$line" rebuild_fallbacks)
        s_pf=$(num_field "$line" pad_fallbacks)
        if [[ -z $s_a || -z $s_b || -z $s_steps || -z $s_on || -z $s_off || -z $s_rf || -z $s_pf ]]; then
            unproven bse_single_fields "a BSE_SINGLE field did not parse" "$line"
        else
            if [[ $s_a -ne $s_b ]]; then
                fail bse_single_rows "BSE_SINGLE rows differ" "rows=$s_a/$s_b"
            fi
            if [[ $s_b -ne 300 || $s_steps -ne 300 ]]; then
                fail bse_single_size "BSE_SINGLE is not the 300-row 300-step arm" \
                     "rows=$s_b steps=$s_steps"
            fi
            if [[ $s_on -ne 1 ]]; then
                fail bse_single_on "the binder arm did not build exactly one graph" \
                     "on_decode_builds=$s_on"
            fi
            if [[ $s_off -lt $s_steps ]]; then
                fail bse_single_off "the stock arm built fewer graphs than steps" \
                     "off_decode_builds=$s_off steps=$s_steps"
            fi
            if [[ $s_off -eq 1 ]]; then
                fail bse_single_off_one "the stock arm built one graph; both arms took the same path" \
                     "off_decode_builds=$s_off"
            fi
            if [[ $s_rf -ne 0 ]]; then
                fail bse_single_rebuild_fb "rebuild fallbacks moved on the single arm" \
                     "rebuild_fallbacks=$s_rf"
            fi
            if [[ $s_pf -ne 0 ]]; then
                fail bse_single_pad_fb "pad fallbacks moved on the single arm" \
                     "pad_fallbacks=$s_pf"
            fi
        fi
    fi

    line=$(last_line "$log" '^BSE_MULTI ok=1 rows=[0-9]+/[0-9]+ steps=[0-9]+ on_decode_builds=[0-9]+ off_decode_builds=[0-9]+ rebuild_fallbacks=[0-9]+ pad_fallbacks=[0-9]+$')
    if [[ -z $line ]]; then
        unproven bse_multi "no BSE_MULTI ok=1 line in the pinned field order" "$log"
    else
        pair=$(pair_field "$line" rows)
        m_a=${pair% *}
        m_b=${pair#* }
        m_steps=$(num_field "$line" steps)
        m_on=$(num_field "$line" on_decode_builds)
        m_off=$(num_field "$line" off_decode_builds)
        m_rf=$(num_field "$line" rebuild_fallbacks)
        m_pf=$(num_field "$line" pad_fallbacks)
        if [[ -z $m_a || -z $m_b || -z $m_steps || -z $m_on || -z $m_off || -z $m_rf || -z $m_pf ]]; then
            unproven bse_multi_fields "a BSE_MULTI field did not parse" "$line"
        else
            if [[ $m_a -ne $m_b ]]; then
                fail bse_multi_rows "BSE_MULTI rows differ" "rows=$m_a/$m_b"
            fi
            if [[ $m_b -ne 260 || $m_steps -ne 65 ]]; then
                fail bse_multi_size "BSE_MULTI is not the 260-row 65-step arm" \
                     "rows=$m_b steps=$m_steps"
            fi
            if [[ $m_on -ne 1 ]]; then
                fail bse_multi_on "the binder arm did not build exactly one graph" \
                     "on_decode_builds=$m_on"
            fi
            if [[ $m_off -lt $m_steps ]]; then
                fail bse_multi_off "the stock arm built fewer graphs than steps" \
                     "off_decode_builds=$m_off steps=$m_steps"
            fi
            if [[ $m_off -eq 1 ]]; then
                fail bse_multi_off_one "the stock arm built one graph; both arms took the same path" \
                     "off_decode_builds=$m_off"
            fi
            if [[ $m_rf -ne 0 ]]; then
                fail bse_multi_rebuild_fb "rebuild fallbacks moved on the multi arm" \
                     "rebuild_fallbacks=$m_rf"
            fi
            if [[ $m_pf -ne 0 ]]; then
                fail bse_multi_pad_fb "pad fallbacks moved on the multi arm" \
                     "pad_fallbacks=$m_pf"
            fi
        fi
    fi

    # The aggregate: every row of both arms, compared. b == single + multi is
    # what makes it 560/560 once the arms are pinned at 300 and 260.
    line=$(last_line "$log" '^BSE_BITWISE ok=1 rows=[0-9]+/[0-9]+$')
    if [[ -z $line ]]; then
        unproven bse_bitwise "no BSE_BITWISE ok=1 rows=a/b line" "$log"
    else
        pair=$(pair_field "$line" rows)
        b_a=${pair% *}
        b_b=${pair#* }
        if [[ -z $b_a || -z $b_b ]]; then
            unproven bse_bitwise_fields "a BSE_BITWISE field did not parse" "$line"
        else
            if [[ $b_a -ne $b_b ]]; then
                fail bse_bitwise_rows "the bitwise counts differ" "rows=$b_a/$b_b"
            fi
            if [[ -z $s_b || -z $m_b ]]; then
                unproven bse_bitwise_sum "an arm's row count did not parse" \
                     "single=${s_b:-?} multi=${m_b:-?}"
            elif [[ $b_b -ne $(( s_b + m_b )) ]]; then
                fail bse_bitwise_sum "BSE_BITWISE rows are not the sum of the two arms" \
                     "rows=$b_b single=$s_b multi=$m_b"
            fi
        fi
    fi

    line=$(last_line "$log" '^BSE_NEG ok=1 differ=[0-9]+$')
    if [[ -z $line ]]; then
        unproven bse_neg "no BSE_NEG ok=1 line" "$log"
    else
        neg=$(num_field "$line" differ)
        if [[ -z $neg ]]; then
            unproven bse_neg_fields "differ did not parse" "$line"
        elif [[ $neg -lt 1 ]]; then
            fail bse_neg_zero "the negative control produced identical logits" "differ=$neg"
        fi
    fi

    if ! grep -qE '^BSE_END ok=1$' "$log"; then
        unproven bse_end "no BSE_END ok=1" "$log"
    fi
    if grep -qE '^FAIL:' "$log"; then
        fail bse_fail_line "a FAIL line is present" "$(last_line "$log" '^FAIL:')"
    fi

    # The summary fields are read only on PASS. Default them so a missing arm
    # cannot abort the script under `set -u` before the summary is printed.
    g_bse_a=${b_a:-}
    g_bse_b=${b_b:-}
    g_s_off=${s_off:-}
    g_m_off=${m_off:-}

    if [[ -z $gate_reason ]]; then
        gate_pass
    fi
}

# ------------------------------------------------------- the n_tokens trajectory
check_ntok() { # log
    local log=$1 line refused built pass x y

    gate_init

    # The zero-output step BINDS: bound_n_outputs=0 is the applied width, and
    # build_n_outputs >= 1 is the anti-rebuild claim (a rebuild at the active
    # zero width would report 0). refused_nout is still read: it must not have
    # moved, and the binary's own assertion compares it with its pre-step value.
    line=$(last_line "$log" '^NOUT_ZERO_BIND ok=1 bound_n_outputs=0 build_n_outputs=[0-9]+ refused_nout=[0-9]+ patch_binds=[0-9]+ builds=[0-9]+$')
    if [[ -z $line ]]; then
        unproven ntok_bind "no NOUT_ZERO_BIND ok=1 line with bound_n_outputs=0" "$log"
    else
        refused=$(num_field "$line" refused_nout)
        built=$(num_field "$line" build_n_outputs)
        if [[ -z $refused || -z $built ]]; then
            unproven ntok_bind_fields "a NOUT_ZERO_BIND field did not parse" "$line"
        elif [[ $built -lt 1 ]]; then
            fail ntok_build_width "build_n_outputs=0: the zero step rebuilt the graph" "$line"
        fi
    fi

    pass=$(last_line "$log" '^PASS: checked_rows=[0-9]+ expected=[0-9]+$')
    if [[ -z $pass ]]; then
        unproven ntok_pass "no PASS: checked_rows=X expected=Y line" "$log"
    else
        x=$(num_field "$pass" checked_rows)
        y=$(num_field "$pass" expected)
        if [[ -z $x || -z $y ]]; then
            unproven ntok_pass_fields "checked_rows or expected did not parse" "$pass"
        elif [[ $x -lt 1 ]]; then
            unproven ntok_rows_zero "checked_rows=0: the run compared no row" \
                 "checked_rows=$x expected=$y"
        elif [[ $x -ne $y ]]; then
            fail ntok_rows "checked_rows != expected" "checked_rows=$x expected=$y"
        fi
    fi

    if grep -qE '^FAIL:' "$log"; then
        fail ntok_fail_line "a FAIL line is present" "$(last_line "$log" '^FAIL:')"
    fi

    g_nout_refused=${refused:-}

    if [[ -z $gate_reason ]]; then
        gate_pass
    fi
}

# ------------------------------------------- the kernel-selection precondition
check_kernel_sel() { # log
    local log=$1 line pair
    local stock_a stock_r stride_d stride_r old_d old_r

    gate_init

    if grep -qE '^KERNEL_SEL_FAIL' "$log"; then
        fail ksel_fail_line "a KERNEL_SEL_FAIL line is present" \
             "$(last_line "$log" '^KERNEL_SEL_FAIL')"
    fi
    if grep -qE '^KERNEL_SEL_STOCK .* differ=1$' "$log"; then
        fail ksel_stock_differ "a STOCK case disagreed" \
             "$(last_line "$log" '^KERNEL_SEL_STOCK .* differ=1$')"
    fi

    line=$(last_line "$log" '^KERNEL_SEL_END ok=1 stock=[0-9]+/[0-9]+ stride=[0-9]+/[0-9]+ old=[0-9]+/[0-9]+$')
    if [[ -z $line ]]; then
        unproven ksel_end "no KERNEL_SEL_END ok=1 line with three d/r arms" "$log"
        return
    fi

    pair=$(pair_field "$line" stock)
    stock_a=${pair% *}
    stock_r=${pair#* }
    pair=$(pair_field "$line" stride)
    stride_d=${pair% *}
    stride_r=${pair#* }
    pair=$(pair_field "$line" old)
    old_d=${pair% *}
    old_r=${pair#* }

    if [[ -z $stock_a || -z $stock_r || -z $stride_d || -z $stride_r || -z $old_d || -z $old_r ]]; then
        unproven ksel_fields "a KERNEL_SEL_END field did not parse" "$line"
        return
    fi

    if [[ $stock_r -eq 0 ]]; then
        unproven ksel_stock_zero "the STOCK arm ran no case" "stock=$stock_a/$stock_r"
    fi
    if [[ $stock_a -ne $stock_r ]]; then
        fail ksel_stock "the complete patch did not agree on every case" \
             "stock=$stock_a/$stock_r"
    fi
    # The controls ran and did not differ: that is a failed observation, not a
    # coverage gap. The zero-denominator case cannot reach this line on an
    # ok=1 KERNEL_SEL_END (the test prints KERNEL_SEL_FAIL instead).
    if [[ $stride_d -lt 1 ]]; then
        fail ksel_stride "the incomplete patch was not detected" "stride=$stride_d/$stride_r"
    fi
    if [[ $old_d -lt 1 ]]; then
        fail ksel_old "no tile-boundary case differed" "old=$old_d/$old_r"
    fi

    g_ks_a=$stock_a
    g_ks_r=$stock_r

    if [[ -z $gate_reason ]]; then
        gate_pass
    fi
}

# --------------------------------------------------------------------- runner
run_bin() { # name log bin args...
    local name=$1 log=$2
    shift 2
    local rc=0
    env -u GGML_TP5_UNION_BINDER \
        -u GGML_TP5_UNION_BINDER_EXACT \
        -u LLAMA_GRAPH_REUSE_DISABLE \
        -u LLAMA_EXPERIMENTAL_VARLEN_NTOK \
        -u GGML_TP5_RSZ_RT \
        -u LLAMA_NKV_MASK_OVERRIDE \
        -u GGML_TP5_UNION_PAD_BUCKET \
        -u GGML_TP5_NKV_FORCE_PAD_FALLBACK \
        HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
        "$@" > "$log" 2>&1 || rc=$?
    if [[ $rc -eq 2 ]]; then
        gate_unproven "${name}_rc" "$name exited 2 (unproven)" "rc=$rc log=$log"
        return 1
    fi
    if [[ $rc -ne 0 ]]; then
        gate_fail "${name}_rc" "$name exited non-zero" "rc=$rc log=$log"
        return 1
    fi
    return 0
}

# ----------------------------------------------------------------- aggregate
# Merge one block's verdict into the aggregate. FAIL wins over UNPROVEN, and
# UNPROVEN wins over PASS. The first slug of the winning verdict is kept.
agg_verdict=$GATE_PASS
agg_reason=

merge() { # verdict reason
    local v=$1 r=$2
    if [[ $v -eq $GATE_FAIL ]]; then
        if [[ $agg_verdict -ne $GATE_FAIL ]]; then
            agg_verdict=$GATE_FAIL
            agg_reason=$r
        fi
    elif [[ $v -eq $GATE_UNPROVEN && $agg_verdict -eq $GATE_PASS ]]; then
        agg_verdict=$GATE_UNPROVEN
        agg_reason=$r
    fi
}

# Run every claim against the logs in `dir` and set the aggregate verdict.
# The selftest calls this with synthetic logs.
check_all() { # dir
    local dir=$1
    agg_verdict=$GATE_PASS
    agg_reason=
    check_four_matrix "$dir/matrix.log"
    merge "$gate_verdict" "$gate_reason"
    check_bse "$dir/bse.log"
    merge "$gate_verdict" "$gate_reason"
    check_ntok "$dir/ntok.log"
    merge "$gate_verdict" "$gate_reason"
    check_kernel_sel "$dir/kernel_sel.log"
    merge "$gate_verdict" "$gate_reason"
    gate_verdict=$agg_verdict
    gate_reason=$agg_reason
}

cleanup() {
    local rc=$?
    if [[ $rc -eq 0 && ${gate_verdict:-$GATE_UNPROVEN} -eq $GATE_PASS ]]; then
        rm -rf -- "$logdir"
    else
        printf 'logs: %s\n' "$logdir" >&2
    fi
}

# ----------------------------------------------------------------------- main
main() {
    if [[ $# -lt 5 || $# -gt 6 ]]; then
        printf 'usage: %s <four-param-matrix> <binder-stock-equiv> <ntok-onegraph> <kernel-sel-len-indep> <model.gguf> [rounds]\n' "$0" >&2
        gate_init
        gate_unproven args "union-mainline-gate: wrong argument count" "got $#, want 5 or 6"
        gate_finish UNION_MAINLINE_GATE
        exit $?
    fi

    local bin_matrix=$1
    local bin_bse=$2
    local bin_ntok=$3
    local bin_ksel=$4
    local model=$5
    local rounds=${6:-2}
    local bin

    if [[ ! $rounds =~ ^[1-9][0-9]*$ ]]; then
        gate_init
        gate_unproven rounds "union-mainline-gate: rounds is not a positive integer" "$rounds"
        gate_finish UNION_MAINLINE_GATE
        exit $?
    fi

    for bin in "$bin_matrix" "$bin_bse" "$bin_ntok" "$bin_ksel"; do
        if [[ ! -f $bin || ! -x $bin ]]; then
            gate_init
            gate_unproven binary "union-mainline-gate: missing or unexecutable binary" "$bin"
            gate_finish UNION_MAINLINE_GATE
            exit $?
        fi
    done
    if [[ ! -f $model || ! -r $model || ! -s $model ]]; then
        gate_init
        gate_unproven model "union-mainline-gate: missing, unreadable or empty model" "$model"
        gate_finish UNION_MAINLINE_GATE
        exit $?
    fi

    logdir=$(mktemp -d) || {
        gate_init
        gate_unproven logdir "union-mainline-gate: cannot create a log directory" "mktemp -d"
        gate_finish UNION_MAINLINE_GATE
        exit $?
    }
    trap cleanup EXIT

    # A non-zero exit is a FAIL, but the checks still run: the log that the
    # failing binary left behind is the evidence, and reading it costs nothing.
    # Capture the run verdict first: each check function resets the shared
    # verdict with gate_init, so a binary failure must be merged back after
    # the claims are read.
    run_bin matrix "$logdir/matrix.log" "$bin_matrix" --matrix-only "$model"
    run_bin bse "$logdir/bse.log" "$bin_bse" "$model"
    run_bin ntok "$logdir/ntok.log" "$bin_ntok" "$model" "$rounds"
    run_bin ksel "$logdir/kernel_sel.log" "$bin_ksel"
    # No binary has failed yet. run_bin sets the shared verdict to FAIL on a
    # non-zero exit; map "no failure recorded" to PASS so the merge after the
    # claims cannot turn a green run UNPROVEN.
    local run_verdict=$GATE_PASS
    local run_reason=
    if [[ $gate_verdict -eq $GATE_FAIL ]]; then
        run_verdict=$GATE_FAIL
        run_reason=$gate_reason
    fi

    check_all "$logdir"
    merge "$run_verdict" "$run_reason"
    gate_verdict=$agg_verdict
    gate_reason=$agg_reason

    gate_finish UNION_MAINLINE_GATE \
        cells="$g_cells" \
        bitwise="$g_bit_a/$g_bit_b" \
        mask_cells="$g_mcells" \
        mask_rows="$g_mok/$g_mchk" \
        bse_rows="$g_bse_a/$g_bse_b" \
        on_decode_builds=1 \
        off_decode_builds="$g_s_off/$g_m_off" \
        nout_zero="$g_nout_zero" \
        nout_refused="$g_nout_refused" \
        kernel_sel="$g_ks_a/$g_ks_r"
}

# Sourcing the file defines the parsers without running the gate; that is how
# the offline parser simulation drives check_four_matrix() and friends.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
