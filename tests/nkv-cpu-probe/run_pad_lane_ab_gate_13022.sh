#!/usr/bin/env bash
# Run on the authorized CPU host only. RUN_DIR must be a new directory.
#
# Builds a CPU-only tree, then runs the padded-lane A/B gate through ctest, the
# same way run_union_varlen_e2e_13022.sh does. ctest is the primary route
# because the target links llama as a shared library in a build tree that is
# never installed, and ctest sets the runtime library path for the test
# environment; a direct run of the binary would need that path passed by hand.
# The direct-run fallback is kept below behind DIRECT=1.
#
# The driver selects both lanes itself (GGML_TP5_UNION_BINDER_EXACT unset and
# set to 0), so the test environment is stripped down to PATH and HOME.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
if [[ -e $RUN_DIR ]]; then
    echo "FAIL: RUN_DIR already exists: $RUN_DIR" >&2
    echo "      every run must write into a new directory; choose another RUN_DIR." >&2
    exit 1
fi
mkdir -- "$RUN_DIR"
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build
steps=${STEPS:-32}
# the report path is named once here and handed to both routes: to CMake for
# the ctest route, and to the binary for the direct route
json=$run_dir/pad-lane-ab-gate.json
url='https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf?download=true'
expected=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739

if [[ -n ${MODEL:-} ]]; then
    model=$(realpath -- "$MODEL")
    [[ -f "$model" && -s "$model" ]] || { echo 'FAIL: model missing or empty' >&2; exit 1; }
    origin="caller-supplied: $model"
else
    model=$run_dir/stories15M-q4_0.gguf
    echo "Downloading $url"
    curl --fail --location --proto '=https' --tlsv1.2 --output "$model" "$url"
    printf '%s  %s\n' "$expected" "$model" | sha256sum --check --status || {
        echo 'FAIL: downloaded model SHA256 mismatch' >&2; exit 1;
    }
    origin=$url
fi
{
    printf 'source=%s\nmodel_origin=%s\nmodel_path=%s\nsteps=%s\n' \
        "$root" "$origin" "$model" "$steps"
    printf 'model_sha256='; sha256sum "$model"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'worktree_status:\n'; git -C "$root" status --short
} | tee "$run_dir/identity.txt"

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

cfg_rc=0
configure_run "$run_dir/configure.log" -S "$root" -B "$build_dir" \
    -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
    -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
    -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_NTOK_TEST_MODEL="$model" \
    -DLLAMA_PAD_LANE_AB_GATE_STEPS="$steps" \
    -DLLAMA_PAD_LANE_AB_GATE_JSON="$json" -DLLAMA_PAD_LANE_AB_GATE=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache || cfg_rc=$?
if [[ $cfg_rc -eq 75 || $cfg_rc -eq 124 || $cfg_rc -eq 143 ]]; then
    echo "UNPROVEN: configure transient rc=$cfg_rc; see $run_dir/configure.log" >&2
    exit 2
fi
if [[ $cfg_rc -ne 0 ]]; then
    echo "FAIL: configure; see $run_dir/configure.log" >&2
    exit 1
fi

bld_rc=0
slot_run "$run_dir/build.log" cmake --build "$build_dir" --target test-pad-lane-ab-gate -j"$(nproc)" || bld_rc=$?
if [[ $bld_rc -eq 75 || $bld_rc -eq 124 ]]; then
    if [[ $bld_rc -eq 75 ]]; then
        echo "UNPROVEN: build build_slot_queue; see $run_dir/build.log" >&2
    else
        echo "UNPROVEN: build build_slot_timeout; see $run_dir/build.log" >&2
    fi
    exit 2
fi
if [[ $bld_rc -ne 0 ]]; then
    echo "FAIL: build; see $run_dir/build.log" >&2
    exit 1
fi

# Pre-flight: prove the test is registered before anything runs. The listing is
# kept so the run records the command line ctest will use, and it is kept on
# failure too, so a run that cannot confirm the registration still leaves
# evidence behind.
list_file=$run_dir/ctest-list.txt
set +e
ctest --test-dir "$build_dir" -N -R '^test-pad-lane-ab-gate$' >"$list_file" 2>&1
list_rc=$?
set -e
cat "$list_file"
if (( list_rc != 0 )); then
    echo "FAIL: ctest -N exited $list_rc; cannot confirm test-pad-lane-ab-gate is registered" >&2
    exit 1
fi
if ! grep -q 'test-pad-lane-ab-gate' "$list_file"; then
    echo 'FAIL: test-pad-lane-ab-gate is not registered' >&2
    exit 1
fi

log=$run_dir/pad-lane-ab-gate.log
# Remove inherited runtime feature switches. The driver sets the two arms itself.
set +e
if [[ ${DIRECT:-0} == 1 ]]; then
    # fallback: run the binary directly. The build tree is not installed, so the
    # shared llama library lives beside the binary and in the build's lib
    # directories; name them explicitly instead of relying on an inherited
    # LD_LIBRARY_PATH that env -i has just stripped.
    lib_dirs="$build_dir/bin:$build_dir/lib:$build_dir/src:$build_dir/ggml/src"
    env -i PATH="$PATH" HOME="$HOME" LD_LIBRARY_PATH="$lib_dirs" \
        "$build_dir/bin/test-pad-lane-ab-gate" "$model" "$json" "$steps" >"$log" 2>&1
    rc=$?
else
    env -i PATH="$PATH" HOME="$HOME" \
        ctest --test-dir "$build_dir" -R '^test-pad-lane-ab-gate$' --no-tests=error -V \
        >"$log" 2>&1
    rc=$?
fi
set -e
# the driver writes its JSON before it decides the exit code, but a crash
# before that point leaves no file. A missing report is UNPROVEN: the run
# cannot show what it compared, so the driver's rc must not be read as a
# verdict. Exit 2, never the driver's status.
if [[ ! -f $json ]]; then
    printf 'UNPROVEN: the driver wrote no JSON report at %s (driver rc=%d)\n' "$json" "$rc" \
        | tee -a "$log" >&2
    printf 'DRIVER_RC=%d JSON=%s LOG=%s\n' "$rc" "$json" "$log"
    exit 2
fi
tee -a "$log" <"$json"
printf 'DRIVER_RC=%d JSON=%s LOG=%s\n' "$rc" "$json" "$log"
exit "$rc"
