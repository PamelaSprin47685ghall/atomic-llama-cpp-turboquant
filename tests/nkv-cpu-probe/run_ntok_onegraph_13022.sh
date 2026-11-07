#!/usr/bin/env bash
# Run on the authorized CPU host only. RUN_DIR must be a new directory.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
mkdir -- "$RUN_DIR"
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build
rounds=${ROUNDS:-50}
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
    printf 'source=%s\nmodel_origin=%s\nmodel_path=%s\nrounds=%s\n' "$root" "$origin" "$model" "$rounds"
    printf 'model_sha256='; sha256sum "$model"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'worktree_status:\n'; git -C "$root" status --short
} | tee "$run_dir/identity.txt"

cmake -S "$root" -B "$build_dir" \
    -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
    -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
    -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_NTOK_TEST_MODEL="$model" -DLLAMA_NTOK_TEST_ROUNDS="$rounds" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build "$build_dir" --target test-ntok-onegraph -j"$(nproc)"
# Remove inherited runtime feature switches. The test sets the two arms itself.
env -i PATH="$PATH" HOME="$HOME" \
    ctest --test-dir "$build_dir" -R '^test-ntok-onegraph$' --no-tests=error -V \
    2>&1 | tee "$run_dir/ctest.log"
