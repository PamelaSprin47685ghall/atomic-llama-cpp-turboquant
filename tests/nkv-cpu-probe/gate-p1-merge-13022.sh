#!/usr/bin/env bash
# Unified pillar-1 CPU gate for the merged union/varlen/strict/nout-tail line.
#
# Run ONLY on the authorized CPU build host (13022). This script configures,
# builds and runs the registered CTest gates. It does not download a model and
# does not touch any existing run directory.
#
#   MODEL=/path/to/tiny-dense-llama.gguf RUN_DIR=/path/to/new-dir \
#       bash tests/nkv-cpu-probe/gate-p1-merge-13022.sh
#
# RUN_DIR must not exist. The script creates it and writes:
#   identity.txt      source revision, model sha256, ccache version, nproc
#   configure.log     cmake configure output
#   build.log         cmake build output
#   gate_a.log        Gate A test output (build counts)
#   gate_b.log        Gate B test output (nout tail bitwise)
#   gate_c.log        Gate C test output (strict, no headroom)
#   gate_d.log        Gate D test output (non-token exact n_kv width)
#   gate_e.log        Gate E test output (kernel selection vs runtime length)
#   GATE_SUMMARY      GATE_A=... GATE_B=... GATE_C=... GATE_D=... GATE_E=... each
#                     PASS|FAIL|UNPROVEN
#
# Exit status, the tri-state contract shared with gate-verdict.sh:
#   0  PASS: all five gates passed (A, B, C, D, E -- the same five keys
#      GATE_SUMMARY carries).
#   1  FAIL: at least one gate failed an equality.
#   2  UNPROVEN: no gate failed, and at least one gate could not prove its
#      claim, or an input was missing. Nothing was compared, or the run could
#      not start.
set -uo pipefail

# The tri-state vocabulary, resolved from this file.
_gate_verdict_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=tests/nkv-cpu-probe/gate-verdict.sh
. "$_gate_verdict_dir/gate-verdict.sh"
unset _gate_verdict_dir

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${MODEL:?set MODEL to a tiny dense CPU LLAMA GGUF}"
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
if [[ -e $RUN_DIR ]]; then
    echo "UNPROVEN: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi
if [[ ! -f $MODEL || ! -s $MODEL ]]; then
    echo "UNPROVEN: MODEL missing or empty: $MODEL" >&2
    exit 2
fi
mkdir -- "$RUN_DIR" || { echo "UNPROVEN: cannot create RUN_DIR" >&2; exit 2; }
model=$(realpath -- "$MODEL")
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build
rounds=${ROUNDS:-50}

# ---------------------------------------------------------------- identity
{
    printf 'source=%s\n' "$root"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'branch=';   git -C "$root" rev-parse --abbrev-ref HEAD
    printf 'model_path=%s\n' "$model"
    printf 'model_sha256='; sha256sum "$model" | cut -d' ' -f1
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'rounds=%s\n' "$rounds"
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
} > "$run_dir/identity.txt"

# ---------------------------------------------------------------- configure
# Same backend-off set as tests/nkv-cpu-probe/run_ntok_onegraph_13022.sh.
# All three model variables point at the same GGUF; each registered test keeps
# its own variable so no test can silently skip.
#
# Configure is not a compile: it runs outside build-slot. Every cmake --build
# goes through build-slot; the ctest runs below do not.
: "${BUILD_SLOT:=/home/kunweiz/.local/bin/build-slot}"
if [[ ! -x $BUILD_SLOT ]]; then
    echo "UNPROVEN: build-slot not executable: $BUILD_SLOT" >&2
    exit 2
fi

# Run one build-slot command, resumably. A 75 (queue cap) or 124 (run cap) is
# the environment refusing, not a result: retry, then report it as UNPROVEN.
# Returns 0 on success, 75/124 when the slot refused every attempt, and 1 on a
# real cmake/compiler error.
slot_run() { # log cmd...
    local log=$1
    shift
    local attempt rc last=0
    for attempt in $(seq 1 20); do
        printf '=== build-slot attempt %d at %s\n' "$attempt" "$(date --iso-8601=seconds)" >> "$log"
        "$BUILD_SLOT" "$@" >> "$log" 2>&1
        rc=$?
        printf 'BUILD_RC=%d attempt=%d\n' "$rc" "$attempt" >> "$log"
        last=$rc
        case $rc in
            0) return 0 ;;
            75) sleep 20 ;;
            124) : ;;
            *) return 1 ;;
        esac
    done
    return "$last"
}

# Configure is not a compile: it runs outside build-slot, whose only compiler
# work here would be cmake's seconds-long try_compile probes. A 75/124/143 is a
# transient environment signal, not a configure verdict: retry, then report it
# as UNPROVEN. Returns 0 on success, 75/124/143 when every attempt was
# refused, and 1 on a real cmake error.
configure_run() { # log args...
    local log=$1
    shift
    local attempt rc last=0
    for attempt in $(seq 1 3); do
        printf '=== configure attempt %d at %s\n' "$attempt" "$(date --iso-8601=seconds)" >> "$log"
        cmake "$@" >> "$log" 2>&1
        rc=$?
        printf 'CONFIGURE_RC=%d attempt=%d\n' "$rc" "$attempt" >> "$log"
        last=$rc
        case $rc in
            0) return 0 ;;
            75|124|143) sleep 5 ;;
            *) return 1 ;;
        esac
    done
    return "$last"
}

configure_run "$run_dir/configure.log" -S "$root" -B "$build_dir" \
        -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
        -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_NTOK_TEST_MODEL="$model" -DLLAMA_NTOK_TEST_ROUNDS="$rounds" \
        -DLLAMA_VARLEN_TEST_MODEL="$model" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cfg_rc=$?
if [[ $cfg_rc -eq 75 || $cfg_rc -eq 124 || $cfg_rc -eq 143 ]]; then
    printf 'GATE_A=unproven GATE_B=unproven GATE_C=unproven GATE_D=unproven GATE_E=unproven\n' > "$run_dir/GATE_SUMMARY"
    echo "UNPROVEN: configure transient rc=$cfg_rc; see $run_dir/configure.log" >&2
    exit 2
fi
if [[ $cfg_rc -ne 0 ]]; then
    printf 'GATE_A=FAIL GATE_B=FAIL GATE_C=FAIL GATE_D=FAIL GATE_E=FAIL\n' > "$run_dir/GATE_SUMMARY"
    echo "FAIL: configure; see $run_dir/configure.log" >&2
    exit 1
fi

# ---------------------------------------------------------------- build
# test-nout-tail-* are add_test() entries over the test-varlen-output-probe
# target, so the target is built explicitly.
slot_run "$run_dir/build.log" cmake --build "$build_dir" -j"$(nproc)" \
        --target test-union-binder-gate test-ntok-onegraph test-ntok-replan \
                 test-ntok-strict test-nkv-exact-width test-varlen-output-probe \
                 test-cpu-attn-pad test-cpu-attn-mask \
                 test-cpu-kernel-sel-len-indep
bld_rc=$?
if [[ $bld_rc -eq 75 || $bld_rc -eq 124 ]]; then
    printf 'GATE_A=unproven GATE_B=unproven GATE_C=unproven GATE_D=unproven GATE_E=unproven\n' > "$run_dir/GATE_SUMMARY"
    if [[ $bld_rc -eq 75 ]]; then
        echo "UNPROVEN: build build_slot_queue; see $run_dir/build.log" >&2
    else
        echo "UNPROVEN: build build_slot_timeout; see $run_dir/build.log" >&2
    fi
    exit 2
fi
if [[ $bld_rc -ne 0 ]]; then
    printf 'GATE_A=FAIL GATE_B=FAIL GATE_C=FAIL GATE_D=FAIL GATE_E=FAIL\n' > "$run_dir/GATE_SUMMARY"
    echo "FAIL: build; see $run_dir/build.log" >&2
    exit 1
fi

# ---------------------------------------------------------------- helpers
# Every test runs with a stripped environment: inherited GGML_TP5_*,
# NKV_CAPTURE_KQV and GGML_TP5_RSZ_RT must not change the arm under test.
# NOUT_TAIL_MODEL is read by the probe itself for the --gate/--tail-gate arms.
run_test() { # name log
    local name=$1 log=$2 rc=0
    local raw="$run_dir/ctest-$name.raw"
    # ctest -V prefixes every line of a test's own output with "<index>: ".
    # The gate's greps are anchored at the start of the line, so they read the
    # stripped log; without this no anchored gate line could ever match. The
    # strip is plumbing: every regex below is unchanged.
    env -i PATH="$PATH" HOME="$HOME" NOUT_TAIL_MODEL="$model" \
        ctest --test-dir "$build_dir" --no-tests=error \
              --output-on-failure -R "^${name}$" -V > "$raw" 2>&1 || rc=$?
    sed -E 's/^[[:space:]]*[0-9]+: //' "$raw" >> "$log"
    rm -f -- "$raw"
    return "$rc"
}

# Every gate starts UNPROVEN. A block promotes itself to PASS only after all
# of its checks ran and held. A FAIL is sticky.
gate_a=$GATE_UNPROVEN
gate_b=$GATE_UNPROVEN
gate_c=$GATE_UNPROVEN
gate_d=$GATE_UNPROVEN
gate_e=$GATE_UNPROVEN
: > "$run_dir/gate_a.log"
: > "$run_dir/gate_b.log"
: > "$run_dir/gate_c.log"
: > "$run_dir/gate_d.log"
: > "$run_dir/gate_e.log"

# ---------------------------------------------------------------- Gate A
# Build count == 1 on the binder ON arm. The counts are the tests' own
# assertions, not invented here:
#   test-ntok-onegraph          on_total - init_on == 1 for every decode step
#                               (tests/test-ntok-onegraph.cpp, step check)
#   test-nout-tail-fixed        on_decode_builds == 1 and off_decode_builds
#   test-nout-tail-instrumented == 200 (steps), from the actual
#                               llama_graph_build_count delta per decode
#                               (tests/nkv-cpu-probe/varlen_output_probe.cpp,
#                               builds_ok), plus the ON NKV_BINDER_STATS line
#                               requiring graph_builds == 1.
# The script re-reads the same lines and fails if they are absent or wrong.
gate_init
for t in test-ntok-onegraph test-nout-tail-fixed test-nout-tail-instrumented; do
    if ! run_test "$t" "$run_dir/gate_a.log"; then
        gate_fail a_rc "Gate A: $t returned non-zero" "$run_dir/gate_a.log"
    fi
done
# The log is the stripped one (see run_test), so the step line starts at the
# line start. The prefix this regex used to require was ctest's, not the test's.
if ! grep -qE '^step=[0-9]+ width=[0-9]+ on_total=[0-9]+ on_decode=1 ' "$run_dir/gate_a.log"; then
    gate_unproven a_step "Gate A: no test-ntok-onegraph step line with on_decode=1" "$run_dir/gate_a.log"
fi
if ! grep -qE 'DECODE_END mode=fixed .*on_decode_builds=1 off_decode_builds=200' "$run_dir/gate_a.log"; then
    gate_unproven a_decode_end "Gate A: no nout tail DECODE_END with on_decode_builds=1 off_decode_builds=200" \
        "$run_dir/gate_a.log"
fi
if ! grep -qE 'NKV_BINDER_STATS .*graph_builds=1 ' "$run_dir/gate_a.log"; then
    gate_unproven a_binder_stats "Gate A: no NKV_BINDER_STATS line with graph_builds=1" "$run_dir/gate_a.log"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_a=$gate_verdict

# ---------------------------------------------------------------- Gate B
# nout tail bitwise contract, enforced by the two registered tail gates:
# every requested logits row and the last batch token's tail row match
# bitwise, llama_get_logits_ith(-1) agrees, outputs are finite, the tail
# changes at least once, and shrink and growth both occur. The script
# re-reads the evidence lines and fails if any is missing.
#
# The CPU attention unit tests run here too. They need no model and are the
# only direct check of the ordered-mask kernels the tail gates exercise:
#   test-cpu-attn-pad          F32/F16 padded-reduction contract
#   test-cpu-attn-f16-kqv      marked F16 value products at several widths
#   test-cpu-attn-mask         F16/F32 masks, NaNs, sinks, signed zero
#   test-cpu-attn-mask-widths  ordered live vs reference-width softmax
#   test-cpu-attn-ordered-kq   marked ordered KQ score path
# test-cpu-attn-mask-kq-dispatch is the unmarked upstream-dispatch
# diagnostic, not a gate; it is not run.
gate_init
for t in test-nout-tail-fixed test-nout-tail-instrumented \
         test-cpu-attn-pad test-cpu-attn-f16-kqv test-cpu-attn-mask \
         test-cpu-attn-mask-widths test-cpu-attn-ordered-kq; do
    if ! run_test "$t" "$run_dir/gate_b.log"; then
        gate_fail b_rc "Gate B: $t returned non-zero" "$run_dir/gate_b.log"
    fi
done
if ! grep -qE 'DECODE_END mode=fixed .*tails=200 growths=[1-9] shrinks=[1-9] changed_tail=1' "$run_dir/gate_b.log"; then
    gate_unproven b_tail "Gate B: missing complete tail evidence (tails=200, growths>0, shrinks>0, changed_tail=1)" \
        "$run_dir/gate_b.log"
fi
if ! grep -qE 'BINDER_EVIDENCE mode=fixed valid=1' "$run_dir/gate_b.log"; then
    gate_unproven b_evidence "Gate B: missing BINDER_EVIDENCE valid=1" "$run_dir/gate_b.log"
fi
if grep -qE 'TAIL_CAPTURE_FAIL|OUTPUT_TAIL_NODE_FAIL|NONFINITE|KQV_ROW_MISMATCH' "$run_dir/gate_b.log"; then
    gate_fail b_fail "Gate B: tail/nonfinite failure line present" \
        "$(grep -E 'TAIL_CAPTURE_FAIL|OUTPUT_TAIL_NODE_FAIL|NONFINITE|KQV_ROW_MISMATCH' "$run_dir/gate_b.log" | tail -n 1)"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_b=$gate_verdict

# ---------------------------------------------------------------- Gate C
# Strict token-width plan sizes with no headroom. test-ntok-strict and
# test-ntok-replan print one PLAN_SIZE line per (step, kind) with the
# allocator request and the width that step requested:
#   PLAN_SIZE step=N kind=kv|token|output active_...=W planned_bytes=P requested_bytes=R
# Any P > R is reserved headroom and fails. Every line must also have P == R
# for the strict model test, whose contract is an exact-width plan.
gate_init
for t in test-ntok-replan test-ntok-strict; do
    if ! run_test "$t" "$run_dir/gate_c.log"; then
        gate_fail c_rc "Gate C: $t returned non-zero" "$run_dir/gate_c.log"
    fi
done
plan_lines=$(grep -c 'PLAN_SIZE ' "$run_dir/gate_c.log" || true)
if [[ ${plan_lines:-0} -lt 2 ]]; then
    gate_unproven c_plan_lines "Gate C: expected PLAN_SIZE lines from both tests" \
        "found ${plan_lines:-0}"
fi
if ! grep -q 'PLAN_SIZE .*kind=kv ' "$run_dir/gate_c.log" ||
   ! grep -q 'PLAN_SIZE .*kind=token ' "$run_dir/gate_c.log" ||
   ! grep -q 'PLAN_SIZE .*kind=output ' "$run_dir/gate_c.log"; then
    gate_unproven c_plan_kinds "Gate C: missing kv, token or output plan-size line" "$run_dir/gate_c.log"
fi
# No headroom: planned_bytes must never exceed requested_bytes. The awk
# separates "no line was read" (UNPROVEN) from "a line violates the bound"
# (FAIL), so a truncated log cannot read as a headroom failure and a real
# headroom failure cannot read as a missing log.
plan_check=$(awk '
    /PLAN_SIZE / {
        planned = -1; requested = -1;
        for (i = 1; i <= NF; ++i) {
            if ($i ~ /^planned_bytes=/)   { split($i, a, "="); planned = a[2] + 0 }
            if ($i ~ /^requested_bytes=/) { split($i, b, "="); requested = b[2] + 0 }
        }
        if (planned < 0 || requested < 0 || planned > requested) { bad = 1 }
        seen++
    }
    END { printf "%d %d\n", seen + 0, bad + 0 }
' "$run_dir/gate_c.log")
plan_seen=${plan_check% *}
plan_bad=${plan_check#* }
if [[ $plan_seen -eq 0 ]]; then
    gate_unproven c_plan_seen "Gate C: no PLAN_SIZE line was read" "$run_dir/gate_c.log"
elif [[ $plan_bad -ne 0 ]]; then
    gate_fail c_headroom "Gate C: a planned size exceeds its requested width (headroom) or is malformed" \
        "$run_dir/gate_c.log"
fi
# The strict model test must show the active widths it was asked for.
if ! grep -q 'PLAN_SIZE .*kind=kv active_kv=[1-9]' "$run_dir/gate_c.log" ||
   ! grep -q 'PLAN_SIZE .*kind=token active_tokens=[1-9]' "$run_dir/gate_c.log" ||
   ! grep -q 'PLAN_SIZE .*kind=output active_outputs=[1-9]' "$run_dir/gate_c.log"; then
    gate_unproven c_active "Gate C: active KV, token or output width not reported" "$run_dir/gate_c.log"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_c=$gate_verdict

# ---------------------------------------------------------------- Gate D
# Non-token exact width. test-nkv-exact-width runs the default CPU path with
# the binder gate unset and the token lane off. It prints, per case:
#   NKV_CASE name=N n_seq=S unified=U steps=T
#   ... case-specific lines ...
#   NKV_CASE_END name=N ok=1 steps=T
# and per step of the `single` case:
#   NKV_EXACT step=N pos=P mask_ne0=M k_ne2=K v_nkv=V
# plus one final line:
#   NKV_EXACT_END ok=1 cases=5 steps=867
#
# The expected case list, per-case step counts and the totals are read from
# the test source (tests/test-nkv-exact-width.cpp): single 300, multi_seq 9
# (8 decodes + 1 after seq_rm), multi_stream 4, widths 257, off 300.
gate_d_cases=(single multi_seq multi_stream widths off)
gate_d_steps=(300 9 4 257 300)
gate_d_n_cases=${#gate_d_cases[@]}
gate_d_total_steps=0
for n in "${gate_d_steps[@]}"; do
    gate_d_total_steps=$((gate_d_total_steps + n))
done

gate_init
if ! run_test test-nkv-exact-width "$run_dir/gate_d.log"; then
    gate_fail d_rc "Gate D: test-nkv-exact-width returned non-zero" "$run_dir/gate_d.log"
fi

# Any FAIL: line is a named failure of the test itself.
if grep -q '^FAIL:' "$run_dir/gate_d.log"; then
    gate_fail d_fail "Gate D: test reported a FAIL: line" \
        "$(grep '^FAIL:' "$run_dir/gate_d.log" | tail -n 1)"
fi

# The case headers and footers must appear once each, in this order. No line
# at all is UNPROVEN; a line that names a different case is FAIL.
mapfile -t got_cases < <(sed -n 's/^NKV_CASE name=\([^ ]*\) .*/\1/p' "$run_dir/gate_d.log")
mapfile -t got_ends  < <(sed -n 's/^NKV_CASE_END name=\([^ ]*\) ok=\([^ ]*\) .*/\1:\2/p' "$run_dir/gate_d.log")
if [[ ${#got_cases[@]} -eq 0 ]]; then
    gate_unproven d_cases "Gate D: no NKV_CASE line" "$run_dir/gate_d.log"
elif [[ ${#got_cases[@]} -ne $gate_d_n_cases ]]; then
    gate_fail d_cases "Gate D: NKV_CASE count differs" \
        "expected $gate_d_n_cases, found ${#got_cases[@]}"
else
    for i in "${!gate_d_cases[@]}"; do
        if [[ ${got_cases[$i]} != "${gate_d_cases[$i]}" ]]; then
            gate_fail d_case_name "Gate D: NKV_CASE #$i names a different case" \
                "'${got_cases[$i]}', want '${gate_d_cases[$i]}'"
        fi
    done
fi
if [[ ${#got_ends[@]} -eq 0 ]]; then
    gate_unproven d_ends "Gate D: no NKV_CASE_END line" "$run_dir/gate_d.log"
elif [[ ${#got_ends[@]} -ne $gate_d_n_cases ]]; then
    gate_fail d_ends "Gate D: NKV_CASE_END count differs" \
        "expected $gate_d_n_cases, found ${#got_ends[@]}"
else
    for i in "${!gate_d_cases[@]}"; do
        if [[ ${got_ends[$i]} != "${gate_d_cases[$i]}:1" ]]; then
            gate_fail d_case_end "Gate D: NKV_CASE_END #$i is not ok=1 for its case" \
                "'${got_ends[$i]}', want '${gate_d_cases[$i]}:1'"
        fi
    done
fi

# Per-case step counts must match the source. A missing footer is UNPROVEN;
# a footer with a different count is FAIL.
for i in "${!gate_d_cases[@]}"; do
    name=${gate_d_cases[$i]}
    want=${gate_d_steps[$i]}
    got=$(sed -n "s/^NKV_CASE_END name=${name} ok=1 steps=\([0-9]*\)$/\1/p" "$run_dir/gate_d.log")
    if [[ -z $got ]]; then
        gate_unproven d_case_steps "Gate D: no NKV_CASE_END ok=1 steps= line for case ${name}" \
            "$run_dir/gate_d.log"
    elif [[ $got != "$want" ]]; then
        gate_fail d_case_steps "Gate D: case ${name} has a different step count" \
            "steps=$got, want $want"
    fi
done

# The final line must carry the exact totals. A missing line is UNPROVEN; a
# line with other totals is FAIL.
if ! grep -q '^NKV_EXACT_END ' "$run_dir/gate_d.log"; then
    gate_unproven d_end "Gate D: no NKV_EXACT_END line" "$run_dir/gate_d.log"
elif ! grep -q "^NKV_EXACT_END ok=1 cases=${gate_d_n_cases} steps=${gate_d_total_steps}$" "$run_dir/gate_d.log"; then
    gate_fail d_end_totals "Gate D: NKV_EXACT_END totals differ" \
        "$(grep -E '^NKV_EXACT_END ' "$run_dir/gate_d.log" | tail -n 1), want cases=${gate_d_n_cases} steps=${gate_d_total_steps}"
fi

# The single case still prints one NKV_EXACT line per step, all at the used span.
exact_lines=$(grep -c '^NKV_EXACT ' "$run_dir/gate_d.log" || true)
if [[ ${exact_lines:-0} -lt 300 ]]; then
    gate_unproven d_exact_lines "Gate D: expected 300 NKV_EXACT lines" "found ${exact_lines:-0}"
fi
# The awk separates "no line was read" (UNPROVEN) from "a line disagrees"
# (FAIL), so a truncated log cannot read as a wrong width.
exact_check=$(awk '
    /^NKV_EXACT / {
        pos = -1; mask = -1; k = -1; v = -1;
        for (i = 1; i <= NF; ++i) {
            if ($i ~ /^pos=/)      { split($i, a, "="); pos  = a[2] + 0 }
            if ($i ~ /^mask_ne0=/) { split($i, b, "="); mask = b[2] + 0 }
            if ($i ~ /^k_ne2=/)    { split($i, c, "="); k    = c[2] + 0 }
            if ($i ~ /^v_nkv=/)    { split($i, d, "="); v    = d[2] + 0 }
        }
        if (pos < 1 || mask != pos || k != pos || v != pos) { bad = 1 }
        seen++
    }
    END { printf "%d %d\n", seen + 0, bad + 0 }
' "$run_dir/gate_d.log")
exact_seen=${exact_check% *}
exact_bad=${exact_check#* }
if [[ $exact_seen -eq 0 ]]; then
    gate_unproven d_exact_seen "Gate D: no NKV_EXACT line was read" "$run_dir/gate_d.log"
elif [[ $exact_bad -ne 0 ]]; then
    gate_fail d_exact_width "Gate D: executed n_kv width is not the used span (mask/K/V disagree with pos)" \
        "$run_dir/gate_d.log"
fi
if ! grep -q '^NKV_EXACT step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1$' "$run_dir/gate_d.log"; then
    gate_unproven d_first_step "Gate D: first step did not execute at width 1" "$run_dir/gate_d.log"
fi
if ! grep -q '^OFF_ARM_REUSE ok=1 ' "$run_dir/gate_d.log"; then
    gate_unproven d_off_reuse "Gate D: stock control did not build, or rebuilt per step" "$run_dir/gate_d.log"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_d=$gate_verdict

# ---------------------------------------------------------------- Gate E
# The binder's in-place narrowing must not change the arithmetic. The test
# states that as a comparison: a graph built at REF_W and then patched down to
# width n (ne/nb rewritten, no flags) must equal a graph built at n. No marking
# is involved - REDUCE_PAD_TAIL / MASKED_TAIL_GEMM are gone from this lane - so
# the arm exercises the patch itself, not a flag.
#
# Two falsifiers keep that arm from being vacuous: an INCOMPLETE patch (ne
# narrowed, nb left at the envelope stride) must be detected, and two unmarked
# widths straddling a tinyBLAS tile boundary must differ somewhere. Either one
# agreeing everywhere fails Gate E by name.
#
# A red Gate E with KERNEL_SEL_STOCK ... differ=1 is a finding about the patch,
# not a licence to relax the assertion. See ANALYSIS-EQUALITY.md in the workdir.
gate_init
if ! run_test test-cpu-kernel-sel-len-indep "$run_dir/gate_e.log"; then
    gate_fail e_rc "Gate E: test-cpu-kernel-sel-len-indep returned non-zero" "$run_dir/gate_e.log"
fi
if ! grep -qE '^KERNEL_SEL_END ok=1 stock=[0-9]+/[0-9]+ stride=[0-9]+/[0-9]+ old=[0-9]+/[0-9]+$' \
        "$run_dir/gate_e.log"; then
    gate_unproven e_end "Gate E: no KERNEL_SEL_END ok=1 line with three d/r arms" "$run_dir/gate_e.log"
else
    e_line=$(grep -E '^KERNEL_SEL_END ' "$run_dir/gate_e.log" | tail -n 1)
    e_sa=$(sed -E 's/.* stock=([0-9]+)\/([0-9]+) .*/\1/' <<<"$e_line")
    e_sr=$(sed -E 's/.* stock=([0-9]+)\/([0-9]+) .*/\2/' <<<"$e_line")
    e_td=$(sed -E 's/.* stride=([0-9]+)\/([0-9]+) .*/\1/' <<<"$e_line")
    e_tr=$(sed -E 's/.* stride=([0-9]+)\/([0-9]+) .*/\2/' <<<"$e_line")
    e_od=$(sed -E 's/.* old=([0-9]+)\/([0-9]+)$/\1/' <<<"$e_line")
    e_or=$(sed -E 's/.* old=([0-9]+)\/([0-9]+)$/\2/' <<<"$e_line")
    # A complete patch must be an identity on EVERY case, not on most.
    if [[ -z $e_sa || -z $e_sr ]]; then
        gate_unproven e_stock_fields "Gate E: a stock d/r field did not parse" "$e_line"
    elif [[ $e_sr -eq 0 || $e_sa != "$e_sr" ]]; then
        gate_fail e_stock "Gate E: a complete patch is not an identity" \
            "stock=${e_sa:-?}/${e_sr:-?}"
    fi
    if [[ -z $e_td || -z $e_tr ]]; then
        gate_unproven e_stride_fields "Gate E: a stride d/r field did not parse" "$e_line"
    elif [[ $e_tr -eq 0 || $e_td -eq 0 ]]; then
        gate_fail e_stride "Gate E: an incomplete patch was not detected" \
            "stride=${e_td:-?}/${e_tr:-?}"
    fi
    if [[ -z $e_od || -z $e_or ]]; then
        gate_unproven e_old_fields "Gate E: an old d/r field did not parse" "$e_line"
    elif [[ $e_or -eq 0 || $e_od -eq 0 ]]; then
        gate_fail e_old "Gate E: the tile-boundary falsifier never differed" \
            "old=${e_od:-?}/${e_or:-?}"
    fi
fi
if grep -qE '^KERNEL_SEL_FAIL ' "$run_dir/gate_e.log"; then
    gate_fail e_fail_line "Gate E: a KERNEL_SEL_FAIL line is present" \
        "$(grep -E '^KERNEL_SEL_FAIL ' "$run_dir/gate_e.log" | tail -n 1)"
fi
if grep -qE '^KERNEL_SEL_STOCK .* differ=1$' "$run_dir/gate_e.log"; then
    gate_fail e_stock_differ "Gate E: the completely patched graph and the graph built at that width disagree" \
        "$(grep -E '^KERNEL_SEL_STOCK .* differ=1$' "$run_dir/gate_e.log" | head -n 1)"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_e=$gate_verdict

# ---------------------------------------------------------------- summary
verdict_upper() { # v
    case $1 in
        "$GATE_FAIL")     printf 'FAIL' ;;
        "$GATE_PASS")     printf 'PASS' ;;
        "$GATE_UNPROVEN") printf 'UNPROVEN' ;;
        *)                printf 'UNKNOWN' ;;
    esac
}

printf 'GATE_A=%s GATE_B=%s GATE_C=%s GATE_D=%s GATE_E=%s\n' \
    "$(verdict_upper "$gate_a")" "$(verdict_upper "$gate_b")" \
    "$(verdict_upper "$gate_c")" "$(verdict_upper "$gate_d")" \
    "$(verdict_upper "$gate_e")" > "$run_dir/GATE_SUMMARY"
cat "$run_dir/GATE_SUMMARY"

# The exit status is the worst of the five: FAIL > UNPROVEN > PASS.
overall=$GATE_PASS
for v in "$gate_a" "$gate_b" "$gate_c" "$gate_d" "$gate_e"; do
    if [[ $v -eq $GATE_FAIL ]]; then
        overall=$GATE_FAIL
        break
    fi
    if [[ $v -eq $GATE_UNPROVEN ]]; then
        overall=$GATE_UNPROVEN
    fi
done
# The exit status is not the enum value: PASS is 0, FAIL is 1, UNPROVEN is 2.
case $overall in
    "$GATE_PASS") exit 0 ;;
    "$GATE_FAIL") exit 1 ;;
    *)            exit 2 ;;
esac
