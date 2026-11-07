#!/usr/bin/env bash
# Run on the authorized CPU build host (13022) only. RUN_DIR must be a new
# directory.
#
# Builds and runs tests/test-ntok-cross-matrix.cpp: the four-parameter cross
# matrix (n_kv x n_outputs x rs_z x n_tokens) on one fixed-topology graph.
#
# Configure and build go through build-slot (override with BUILD_SLOT=...).
# The ctest runs stay outside the slot.
#
#   RUN_DIR=/path/to/new-dir bash tests/nkv-cpu-probe/run_ntok_cross_matrix_13022.sh
#   MODEL=/path/to/dense-llama.gguf RUN_DIR=... bash ...   # reuse a local model
#
# Outputs:
#   identity.txt   revision, branch, model sha256, ccache, nproc, host, date
#   configure.log  cmake configure output
#   build.log      cmake build output
#   hash.log       test-ntok-cross-matrix-hash (no model)
#   matrix.log     test-ntok-cross-matrix (model)
#   GATE_SUMMARY   GATE_HASH=... GATE_MATRIX=... each PASS|FAIL
#
# The MATRIX gate requires rebuilds=0, stock_rebuilds>0, bitwise=n/n with n
# equal to the cell count, and NTOK_MASK_NEG ok=1. A red gate is a finding,
# never a licence to relax the regex.
#
# Exit status: 0 only when both gates are PASS.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
mkdir -- "$RUN_DIR"
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build
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
    printf 'source=%s\nmodel_origin=%s\nmodel_path=%s\n' "$root" "$origin" "$model"
    printf 'model_sha256='; sha256sum "$model"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'branch=';   git -C "$root" rev-parse --abbrev-ref HEAD
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
    printf 'worktree_status:\n'; git -C "$root" status --short
} | tee "$run_dir/identity.txt"

# Every compile on the build host goes through build-slot. Configure and
# build are wrapped; the tests below run outside it.
: "${BUILD_SLOT:=/home/kunweiz/.local/bin/build-slot}"
if [[ ! -x $BUILD_SLOT ]]; then
    echo "FAIL: build-slot not executable: $BUILD_SLOT" >&2
    exit 2
fi

gate_hash=PASS
gate_matrix=PASS

if ! "$BUILD_SLOT" cmake -S "$root" -B "$build_dir" \
        -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
        -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_NTOK_TEST_MODEL="$model" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        > "$run_dir/configure.log" 2>&1; then
    printf 'GATE_HASH=FAIL GATE_MATRIX=FAIL\n' > "$run_dir/GATE_SUMMARY"
    echo "FAIL: configure; see $run_dir/configure.log" >&2
    exit 1
fi

if ! "$BUILD_SLOT" cmake --build "$build_dir" -j"$(nproc)" --target test-ntok-cross-matrix \
        > "$run_dir/build.log" 2>&1; then
    printf 'GATE_HASH=FAIL GATE_MATRIX=FAIL\n' > "$run_dir/GATE_SUMMARY"
    echo "FAIL: build; see $run_dir/build.log" >&2
    exit 1
fi

run_test() { # name log
    local name=$1 log=$2
    env -i PATH="$PATH" HOME="$HOME" \
        ctest --test-dir "$build_dir" --no-tests=error \
              --output-on-failure -R "^${name}$" -V >> "$log" 2>&1
}

# ---------------------------------------------------------------- Gate HASH
# No model. The token widths include 1, 3, 2, 4, 7, 5 and 300, so the exclusion
# is exercised below, at and above the 256-cell bucket.
: > "$run_dir/hash.log"
if ! run_test test-ntok-cross-matrix-hash "$run_dir/hash.log"; then
    gate_hash=FAIL
fi
if ! grep -qE '^NTOK_HASH_ARM ok=1 cases=[1-9][0-9]* widths=7$' "$run_dir/hash.log"; then
    echo "Gate HASH: no NTOK_HASH_ARM ok=1 with widths=7" >&2
    gate_hash=FAIL
fi
if ! grep -qE '^NTOK_END ok=1$' "$run_dir/hash.log"; then
    echo "Gate HASH: no NTOK_END ok=1" >&2
    gate_hash=FAIL
fi
if grep -qE '^FAIL:' "$run_dir/hash.log"; then
    echo "Gate HASH: FAIL line present" >&2
    gate_hash=FAIL
fi

# -------------------------------------------------------------- Gate MATRIX
# Model. The test's own assertions are the gate; this script only re-reads the
# machine-readable lines so a silently truncated run cannot pass.
: > "$run_dir/matrix.log"
if ! run_test test-ntok-cross-matrix "$run_dir/matrix.log"; then
    gate_matrix=FAIL
fi
if ! grep -qE '^NTOK_MATRIX_AXES n_kv=1 n_outputs=1 rs_z=(0|1) n_tokens=1 reason=[a-z_]+$' "$run_dir/matrix.log"; then
    echo "Gate MATRIX: no NTOK_MATRIX_AXES line" >&2
    gate_matrix=FAIL
fi
# The mask negative case must have reported a pass: the refusal path is proven
# live, not assumed.
if ! grep -qE '^NTOK_MASK_NEG ok=1$' "$run_dir/matrix.log"; then
    echo "Gate MATRIX: no NTOK_MASK_NEG ok=1 (the mask refusal path did not fire)" >&2
    gate_matrix=FAIL
fi
# The two bitwise counts are compared in shell, not by a regex backreference:
# POSIX ERE has no backreferences, and GNU grep's \N is an extension that a
# non-GNU grep reads as a literal N (or rejects the pattern outright).
matrix_line=$(grep -E '^NTOK_MATRIX ' "$run_dir/matrix.log" | tail -n 1 || true)
if ! grep -qE '^NTOK_MATRIX ok=1 cells=[1-9][0-9]* hash=[0-9a-f]{16} rebuilds=0 stock_rebuilds=[1-9][0-9]* bitwise=[1-9][0-9]*/[1-9][0-9]*$' "$run_dir/matrix.log"; then
    echo "Gate MATRIX: no NTOK_MATRIX ok=1 with rebuilds=0, stock_rebuilds>0 and bitwise=n/n" >&2
    gate_matrix=FAIL
elif [[ $(sed -E 's/^.* bitwise=([0-9]+)\/([0-9]+)$/\1/' <<< "$matrix_line") != \
        $(sed -E 's/^.* bitwise=([0-9]+)\/([0-9]+)$/\2/' <<< "$matrix_line") ]]; then
    echo "Gate MATRIX: bitwise counts differ: $matrix_line" >&2
    gate_matrix=FAIL
fi
if ! grep -qE '^NTOK_END ok=1$' "$run_dir/matrix.log"; then
    echo "Gate MATRIX: no NTOK_END ok=1" >&2
    gate_matrix=FAIL
fi
if grep -qE '^FAIL:' "$run_dir/matrix.log"; then
    echo "Gate MATRIX: FAIL line present" >&2
    gate_matrix=FAIL
fi

printf 'GATE_HASH=%s GATE_MATRIX=%s\n' "$gate_hash" "$gate_matrix" > "$run_dir/GATE_SUMMARY"
cat "$run_dir/GATE_SUMMARY"
if [[ $gate_hash == PASS && $gate_matrix == PASS ]]; then
    exit 0
fi
exit 1
