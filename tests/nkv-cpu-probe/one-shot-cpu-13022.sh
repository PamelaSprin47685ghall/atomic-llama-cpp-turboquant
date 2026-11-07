#!/usr/bin/env bash
# One shot: build the CPU lane and run every registration this lane owns.
#
# This is the 13022 counterpart of the code-only work done on 12022. That
# host has no compiler, so nothing there proves the code compiles or that the
# kernels behave; this script is the single command that does.
#
# It never touches GPU or HIP: every backend except CPU is configured OFF, so
# a stray device on the host cannot change the result.
#
# The two build steps run inside build-slot (the host's compile queue) and
# nothing else does. Configure and build are the only two things that compile.
# The ctest run and the summary are ordinary processes.
#
# A red result means one of two things, in this order of likelihood:
#
#   1. The binder never bound below its build envelope for this model/context
#      pair, so the trim never had a narrower width to reduce over. The
#      NKV_RT_TRUELEN line reports live_scale=0 in that case. That is a
#      finding about the pair, not a licence to relax a regex; try a model
#      whose used span falls short of the capacity.
#   2. The trim did not execute. The kernel PASS lines are printed only after
#      every assertion in the arm returned true, so a missing line names the
#      arm that failed, and ctest.log carries the arm's own stderr.
#
#   MODEL=/path/to/tiny-dense-llama.gguf [RUN_DIR=/path/to/new-dir] \
#       bash tests/nkv-cpu-probe/one-shot-cpu-13022.sh
#
# RUN_DIR defaults to build-one-shot-cpu-<timestamp> under the source root and
# must not exist. Outputs: identity.txt, configure.log, build.log, ctest.log,
# SUMMARY.
#
# Exit status: 0 only when ctest passed, all five kernel PASS lines are
# present, and both true-length lines are present. 75 and 124 are build-slot's
# own codes and are passed through unchanged.
set -uo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)

host=$(uname -n)
if [[ $host != *13022* ]]; then
    echo "FAIL: this script builds; run it on 13022 (this host is $host)" >&2
    exit 2
fi

if [[ -z ${MODEL:-} ]]; then
    echo "FAIL: MODEL is unset; set it to a CPU-loadable GGUF" >&2
    exit 2
fi
if [[ ! -f $MODEL || ! -r $MODEL || ! -s $MODEL ]]; then
    echo "FAIL: MODEL is not a non-empty readable file: $MODEL" >&2
    exit 2
fi

if [[ -z ${RUN_DIR:-} ]]; then
    RUN_DIR=$root/build-one-shot-cpu-$(date +%Y%m%d-%H%M%S)
fi
if [[ -e $RUN_DIR ]]; then
    echo "FAIL: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi
mkdir -- "$RUN_DIR" || { echo "FAIL: cannot create RUN_DIR: $RUN_DIR" >&2; exit 2; }

model=$(realpath -- "$MODEL")
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build=$run_dir/build

# The identity block is written first, before anything compiles, so a run that
# dies in configure still says what it was testing. The key set matches
# gate-nkv-runtime-13022.sh so the two files can be diffed line for line.
{
    printf 'source=%s\n' "$root"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'branch=';   git -C "$root" rev-parse --abbrev-ref HEAD
    printf 'model_path=%s\n' "$model"
    printf 'model_sha256='; sha256sum "$model" | cut -d' ' -f1
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
} > "$run_dir/identity.txt"

# Every compile on the build host goes through build-slot. Its two codes are
# reported and passed through, never retried: a queue timeout and a compile
# timeout both mean the build is not viable as written.
: "${BUILD_SLOT:=/home/kunweiz/.local/bin/build-slot}"
if [[ ! -x $BUILD_SLOT ]]; then
    echo "FAIL: build-slot not executable: $BUILD_SLOT" >&2
    exit 2
fi

report_slot_failure() { # code log what
    local rc=$1 log=$2 what=$3
    case $rc in
        75)
            echo "FAIL: build-slot queue is over 3 minutes. Do not re-queue; use static analysis." >&2
            ;;
        124)
            echo "FAIL: a single compile exceeded 3 minutes. Optimise the build, do not retry." >&2
            ;;
        *)
            echo "FAIL: $what exited $rc; see $log" >&2
            ;;
    esac
    exit "$rc"
}

"$BUILD_SLOT" cmake -S "$root" -B "$build" \
    -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
    -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
    -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_VARLEN_TEST_MODEL="$model" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    > "$run_dir/configure.log" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
    report_slot_failure "$rc" "$run_dir/configure.log" "configure"
fi

"$BUILD_SLOT" cmake --build "$build" -j"$(nproc)" \
    --target test-nkv-runtime-param test-cpu-attn-pad \
    > "$run_dir/build.log" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
    report_slot_failure "$rc" "$run_dir/build.log" "build"
fi

# The tests run outside the slot. One ctest invocation over exactly the
# registrations this lane owns, so a failure is attributable to one of them.
# test-nkv-runtime-param and test-triparam-matrix are registered only when
# LLAMA_VARLEN_TEST_MODEL is set, which configure above does.
env -i PATH="$PATH" HOME="$HOME" \
    ctest --test-dir "$build" --no-tests=error \
          --output-on-failure -V \
          -R '^(test-cpu-attn-pad|test-cpu-attn-f16-kqv|test-cpu-attn-true-len|test-cpu-attn-true-len-counters|test-nkv-runtime-param-hash|test-nkv-runtime-param|test-triparam-matrix)$' \
          > "$run_dir/ctest.log" 2>&1
ctest_rc=$?

# The summary re-reads the log rather than trusting the exit code: a
# registration whose PASS_REGULAR_EXPRESSION was satisfied still has to leave
# its line in the log for a reader to find.
flag() { # fixed-string
    if grep -F -q -- "$1" "$run_dir/ctest.log"; then
        printf '1'
    else
        printf '0'
    fi
}
capture() { # regex
    local line
    line=$(grep -m1 -E "$1" "$run_dir/ctest.log")
    if [[ -n $line ]]; then
        printf '%s' "$line"
    else
        printf 'MISSING'
    fi
}

pass_truelen=$(flag 'PASS: TRUE_LEN_REDUCE matches a live-width graph bitwise')
pass_counters=$(flag 'PASS: TRUE_LEN_REDUCE counters follow the runtime length')
pass_f16=$(flag 'PASS: F16 ordered kqv rows match reference width bitwise')
pass_pad=$(flag 'PASS: 608 widths, three reference widths each, guarded active rows')
pass_helper=$(flag 'PASS: padded softmax helper keeps its live lanes for every padded width')
nkv_truelen=$(capture '^NKV_RT_TRUELEN ')
triparam_truelen=$(capture '^TRIPARAM_TRUELEN ')
triparam_matrix=$(capture '^TRIPARAM_MATRIX ')

# This script does not run the gate, so the three gate verdicts are not its
# business. Say so explicitly rather than inventing a value.
gate_hash=NOT_RUN
gate_live=NOT_RUN
gate_matrix=NOT_RUN
if [[ -f $run_dir/GATE_SUMMARY ]]; then
    gate_hash=$(sed -n 's/^.*GATE_HASH=\([^ ]*\).*$/\1/p' "$run_dir/GATE_SUMMARY")
    gate_live=$(sed -n 's/^.*GATE_LIVE=\([^ ]*\).*$/\1/p' "$run_dir/GATE_SUMMARY")
    gate_matrix=$(sed -n 's/^.*GATE_MATRIX=\([^ ]*\).*$/\1/p' "$run_dir/GATE_SUMMARY")
    : "${gate_hash:=NOT_RUN}" "${gate_live:=NOT_RUN}" "${gate_matrix:=NOT_RUN}"
fi

{
    printf 'SUMMARY\n'
    printf 'CTEST_RC=%s\n' "$ctest_rc"
    printf 'PASS_KERNEL_TRUELEN=%s\n' "$pass_truelen"
    printf 'PASS_KERNEL_COUNTERS=%s\n' "$pass_counters"
    printf 'PASS_KERNEL_F16=%s\n' "$pass_f16"
    printf 'PASS_KERNEL_PAD=%s\n' "$pass_pad"
    printf 'PASS_KERNEL_HELPER=%s\n' "$pass_helper"
    printf 'NKV_RT_TRUELEN=%s\n' "$nkv_truelen"
    printf 'TRIPARAM_TRUELEN=%s\n' "$triparam_truelen"
    printf 'TRIPARAM_MATRIX=%s\n' "$triparam_matrix"
    printf 'GATE_HASH=%s\n' "$gate_hash"
    printf 'GATE_LIVE=%s\n' "$gate_live"
    printf 'GATE_MATRIX=%s\n' "$gate_matrix"
} | tee "$run_dir/SUMMARY"

ok=1
if [[ $ctest_rc -ne 0 ]]; then
    echo "FAIL: ctest exited $ctest_rc; see $run_dir/ctest.log" >&2
    ok=0
fi
for pair in "truelen:$pass_truelen" "counters:$pass_counters" "f16:$pass_f16" \
            "pad:$pass_pad" "helper:$pass_helper"; do
    if [[ ${pair#*:} != 1 ]]; then
        echo "FAIL: kernel arm ${pair%%:*} did not print its PASS line" >&2
        ok=0
    fi
done
if [[ $nkv_truelen == MISSING ]]; then
    echo "FAIL: no NKV_RT_TRUELEN line; the live lane did not report its measured work" >&2
    ok=0
fi
if [[ $triparam_truelen == MISSING ]]; then
    echo "FAIL: no TRIPARAM_TRUELEN line; the matrix did not report its trim" >&2
    ok=0
fi

echo "ONE_SHOT_CPU ok=$ok run_dir=$run_dir"
if [[ $ok -eq 1 ]]; then
    exit 0
fi
exit 1
