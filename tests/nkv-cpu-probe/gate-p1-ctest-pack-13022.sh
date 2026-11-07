#!/usr/bin/env bash
# Pillar-1 ctest pack: one ctest name per runtime parameter, each its own gate.
#
# Run ONLY on the authorized CPU build host (13022). Configure and build go
# through /home/kunweiz/.local/bin/build-slot (override with BUILD_SLOT=...);
# every ctest invocation stays OUTSIDE the slot.
#
#   NKV_MODEL=/path/tiny-dense-llama.gguf \
#   RSZ_MODEL=/path/tiny-rsz-mamba.gguf \
#   RSZ_HYBRID_MODEL=/path/tiny-hybrid.gguf \
#   NOUT_TAIL_MODEL=/path/tiny-dense-llama.gguf \
#   RUN_DIR=/path/to/new-dir \
#       bash tests/nkv-cpu-probe/gate-p1-ctest-pack-13022.sh
#
# Each parameter is one `ctest -R '^<name>$'`, so each parameter is its own
# ctest result. The script re-reads the machine-readable line the test prints,
# so a truncated run cannot pass.
#
# Model inputs are optional. A parameter whose model is missing is UNPROVEN,
# never PASS: the script does not register the test (no model means no ctest
# name), and it reports unproven for that field.
#
# Exit status, the tri-state contract shared with gate-verdict.sh:
#   0  PASS: every parameter passed.
#   1  FAIL: at least one parameter failed.
#   2  UNPROVEN: no parameter failed, and at least one could not prove its
#      claim (a missing model, a skipped test, a missing evidence line).
#      Nothing was compared, or the run could not start.
#
# Outputs in RUN_DIR:
#   identity.txt    revision, branch, model paths and sha256, ccache, nproc
#   configure.log   cmake configure output
#   build.log       cmake build output
#   <name>.log      ctest -V output for that parameter
#   GATE_SUMMARY    one P1_<PARAM>=pass|fail|unproven field per parameter
set -uo pipefail

# The tri-state vocabulary, resolved from this file.
_gate_verdict_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=tests/nkv-cpu-probe/gate-verdict.sh
. "$_gate_verdict_dir/gate-verdict.sh"
unset _gate_verdict_dir

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
if [[ -e $RUN_DIR ]]; then
    echo "UNPROVEN: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi
mkdir -- "$RUN_DIR" || { echo "UNPROVEN: cannot create RUN_DIR" >&2; exit 2; }
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build

# Every model input is optional. An empty value means "not supplied".
nkv_model=${NKV_MODEL:-}
rsz_model=${RSZ_MODEL:-}
rsz_hybrid_model=${RSZ_HYBRID_MODEL:-}
nout_tail_model=${NOUT_TAIL_MODEL:-}

for pair in "NKV_MODEL:$nkv_model" "RSZ_MODEL:$rsz_model" \
            "RSZ_HYBRID_MODEL:$rsz_hybrid_model" "NOUT_TAIL_MODEL:$nout_tail_model"; do
    name=${pair%%:*}
    value=${pair#*:}
    if [[ -n $value && ( ! -f $value || ! -s $value ) ]]; then
        echo "FAIL: $name is set but missing or empty: $value" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------- identity
{
    printf 'source=%s\n' "$root"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'branch=';   git -C "$root" rev-parse --abbrev-ref HEAD
    for pair in "NKV_MODEL:$nkv_model" "RSZ_MODEL:$rsz_model" \
                "RSZ_HYBRID_MODEL:$rsz_hybrid_model" "NOUT_TAIL_MODEL:$nout_tail_model"; do
        name=${pair%%:*}
        value=${pair#*:}
        if [[ -n $value ]]; then
            printf '%s=%s\n' "$name" "$(realpath -- "$value")"
            printf '%s_sha256=' "$name"; sha256sum "$value" | cut -d' ' -f1
        else
            printf '%s=\n' "$name"
        fi
    done
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
} > "$run_dir/identity.txt"

# ---------------------------------------------------------------- parameters
# One row per parameter: ctest name | model variable | log file.
# A model parameter whose variable is empty is UNPROVEN and its test is not
# registered by the configure step (the cache variable stays empty).
params=(
    "p1-nkv-topology-hash|"
    "p1-nkv-live-width|nkv_model"
    "p1-rsz-runtime-param|rsz_model"
    "p1-rsz-selection-write|"
    "p1-rsz-nkv-topology|rsz_hybrid_model"
    "p1-nout-tail-fixed|nout_tail_model"
    "p1-nout-tail-instrumented|nout_tail_model"
)

# summary key per parameter, for GATE_SUMMARY
summary_key() {
    case $1 in
        p1-nkv-topology-hash)       printf 'P1_NKV_TOPOLOGY_HASH' ;;
        p1-nkv-live-width)          printf 'P1_NKV_LIVE_WIDTH' ;;
        p1-rsz-runtime-param)       printf 'P1_RSZ_RUNTIME_PARAM' ;;
        p1-rsz-selection-write)     printf 'P1_RSZ_SELECTION_WRITE' ;;
        p1-rsz-nkv-topology)        printf 'P1_RSZ_NKV_TOPOLOGY' ;;
        p1-nout-tail-fixed)         printf 'P1_NOUT_TAIL_FIXED' ;;
        p1-nout-tail-instrumented)  printf 'P1_NOUT_TAIL_INSTRUMENTED' ;;
        *)                          printf 'P1_UNKNOWN' ;;
    esac
}

# ---------------------------------------------------------------- configure
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

summary_all() { # verdict
    {
        for row in "${params[@]}"; do
            printf '%s=%s\n' "$(summary_key "${row%%|*}")" "$1"
        done
    } > "$run_dir/GATE_SUMMARY"
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

cmake_model_args=()
[[ -n $nkv_model ]]        && cmake_model_args+=(-DLLAMA_VARLEN_TEST_MODEL="$(realpath -- "$nkv_model")")
[[ -n $rsz_model ]]        && cmake_model_args+=(-DLLAMA_RSZ_TEST_MODEL="$(realpath -- "$rsz_model")")
[[ -n $rsz_hybrid_model ]] && cmake_model_args+=(-DLLAMA_RSZ_HYBRID_TEST_MODEL="$(realpath -- "$rsz_hybrid_model")")

configure_run "$run_dir/configure.log" -S "$root" -B "$build_dir" \
        -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
        -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
        "${cmake_model_args[@]}" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cfg_rc=$?
if [[ $cfg_rc -eq 75 || $cfg_rc -eq 124 || $cfg_rc -eq 143 ]]; then
    summary_all unproven
    echo "UNPROVEN: configure transient rc=$cfg_rc; see $run_dir/configure.log" >&2
    exit 2
fi
if [[ $cfg_rc -ne 0 ]]; then
    summary_all FAIL
    echo "FAIL: configure; see $run_dir/configure.log" >&2
    exit 1
fi

# ---------------------------------------------------------------- build
build_targets=(test-nkv-runtime-param test-rsz-selection-write test-varlen-output-probe)
[[ -n $rsz_model ]]        && build_targets+=(test-rsz-runtime-param)
[[ -n $rsz_hybrid_model ]] && build_targets+=(test-rsz-nkv-rsz-topology)

slot_run "$run_dir/build.log" cmake --build "$build_dir" -j"$(nproc)" \
        --target "${build_targets[@]}"
bld_rc=$?
if [[ $bld_rc -eq 75 || $bld_rc -eq 124 ]]; then
    summary_all unproven
    if [[ $bld_rc -eq 75 ]]; then
        echo "UNPROVEN: build build_slot_queue; see $run_dir/build.log" >&2
    else
        echo "UNPROVEN: build build_slot_timeout; see $run_dir/build.log" >&2
    fi
    exit 2
fi
if [[ $bld_rc -ne 0 ]]; then
    summary_all FAIL
    echo "FAIL: build; see $run_dir/build.log" >&2
    exit 1
fi

# ---------------------------------------------------------------- helpers
# Every test runs with a stripped environment: inherited GGML_TP5_*,
# NKV_CAPTURE_KQV and GGML_TP5_RSZ_RT must not change the arm under test.
# NOUT_TAIL_MODEL is read by the probe itself for the --gate/--tail-gate arms.
run_test() { # name log
    local name=$1 log=$2
    env -i PATH="$PATH" HOME="$HOME" NOUT_TAIL_MODEL="${nout_tail_model:+$(realpath -- "$nout_tail_model")}" \
        ctest --test-dir "$build_dir" --no-tests=error \
              --output-on-failure -R "^${name}$" -V >> "$log" 2>&1
}

# ctest prints "Test #N: <name> ...***Skipped" for a SKIP_RETURN_CODE exit.
# A skipped test is UNPROVEN, never PASS.
was_skipped() { # log name
    grep -qE "Test #[0-9]+: $2 \.\.\.\*\*\*Skipped" "$1"
}

declare -A result

for row in "${params[@]}"; do
    name=${row%%|*}
    model_var=${row#*|}
    log=$run_dir/$name.log
    : > "$log"

    if [[ -n $model_var ]]; then
        model_value=${!model_var}
        if [[ -z $model_value ]]; then
            result[$name]=UNPROVEN
            printf 'UNPROVEN: %s needs %s, which is unset\n' "$name" "$model_var" >> "$log"
            continue
        fi
    fi

    if ! run_test "$name" "$log"; then
        if was_skipped "$log" "$name"; then
            result[$name]=UNPROVEN
        else
            result[$name]=FAIL
        fi
        continue
    fi
    # A test that exits 0 but was reported as skipped is still UNPROVEN.
    if was_skipped "$log" "$name"; then
        result[$name]=UNPROVEN
        continue
    fi
    result[$name]=PASS
done

# ---------------------------------------------------------------- evidence
# Each parameter's own machine-readable line must be present in its log.
# A PASS without its line is a FAIL: the run was truncated or the format moved.
require_line() { # name regex description
    if [[ ${result[$1]:-FAIL} != PASS ]]; then
        return
    fi
    if ! grep -qE "$2" "$run_dir/$1.log"; then
        echo "$1: missing evidence line ($3)" >&2
        result[$1]=FAIL
    fi
}

# no model: the hash is taken at several widths and does not move
require_line p1-nkv-topology-hash \
    '^NKV_RT_HASH_ARM ok=1 cases=[1-9][0-9]* widths=[1-9][0-9]*$' \
    'NKV_RT_HASH_ARM ok=1'

# model: 300 steps, the hash re-taken over real binds
require_line p1-nkv-live-width \
    '^NKV_RT_LIVE ok=1 steps=[1-9][0-9]* hash=[0-9a-f]{16} min_width=[0-9]+ max_width=[0-9]+ bind_steps=[1-9][0-9]* distinct_bound=([2-9]|[1-9][0-9]+)$' \
    'NKV_RT_LIVE ok=1 with bind_steps>0 distinct_bound>1'

# model: the rs_z lane, three legs bitwise equal
require_line p1-rsz-runtime-param \
    '^RSZ_RT_TEST PASS$' \
    'RSZ_RT_TEST PASS'

# no model: the selection-row write contract
require_line p1-rsz-selection-write \
    '^RSZ_SELECTION_WRITE_TEST PASS$' \
    'RSZ_SELECTION_WRITE_TEST PASS'

# model: the rs_z x n_kv cross on a hybrid fixture
require_line p1-rsz-nkv-topology \
    '^RSZ_NKV_RSZ_TEST PASS$' \
    'RSZ_NKV_RSZ_TEST PASS'

# model: the fixed-token-width tail gate
require_line p1-nout-tail-fixed \
    'DECODE_END mode=fixed .*on_decode_builds=1 off_decode_builds=200' \
    'DECODE_END mode=fixed with on_decode_builds=1 off_decode_builds=200'
require_line p1-nout-tail-fixed \
    'BINDER_EVIDENCE mode=fixed valid=1' \
    'BINDER_EVIDENCE mode=fixed valid=1'

# model: the instrumented tail gate
require_line p1-nout-tail-instrumented \
    'DECODE_END mode=fixed .*on_decode_builds=1 off_decode_builds=200' \
    'DECODE_END mode=fixed with on_decode_builds=1 off_decode_builds=200'
require_line p1-nout-tail-instrumented \
    'BINDER_EVIDENCE mode=fixed valid=1' \
    'BINDER_EVIDENCE mode=fixed valid=1'

# A FAIL line anywhere in a passing log is a contradiction: the test printed a
# failure but exited 0. Treat it as FAIL rather than trusting the exit code.
for row in "${params[@]}"; do
    name=${row%%|*}
    if [[ ${result[$name]:-FAIL} == PASS ]] && grep -qE '^(FAIL:|FAIL )' "$run_dir/$name.log"; then
        echo "$name: printed a FAIL line but exited 0" >&2
        result[$name]=FAIL
    fi
done

# ---------------------------------------------------------------- summary
# The result array carries the three names. The summary prints them in lower
# case, so an arm that could not prove its claim reads as `unproven`, never as
# `fail`.
result_name() { # value
    case $1 in
        PASS) printf 'pass' ;;
        FAIL) printf 'fail' ;;
        *)    printf 'unproven' ;;
    esac
}

: > "$run_dir/GATE_SUMMARY"
for row in "${params[@]}"; do
    name=${row%%|*}
    printf '%s=%s\n' "$(summary_key "$name")" "$(result_name "${result[$name]:-UNPROVEN}")" >> "$run_dir/GATE_SUMMARY"
done
cat "$run_dir/GATE_SUMMARY"

# The exit status is the worst parameter: FAIL > UNPROVEN > PASS.
overall=0
for row in "${params[@]}"; do
    name=${row%%|*}
    case ${result[$name]:-UNPROVEN} in
        FAIL)
            overall=1
            break
            ;;
        PASS)
            ;;
        *)
            if [[ $overall -eq 0 ]]; then
                overall=2
            fi
            ;;
    esac
done
exit "$overall"
