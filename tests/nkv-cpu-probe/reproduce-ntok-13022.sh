#!/usr/bin/env bash
set -euo pipefail
# Run only on 13022. Supply a dense CPU-supported LLAMA GGUF.
: "${MODEL:?Set MODEL to the tiny LLAMA GGUF}"
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
build=${BUILD_DIR:-"$root/build-ntok-cpu"}
cmake -S "$root" -B "$build" -DGGML_HIP=OFF -DGGML_CUDA=OFF \
    -DGGML_CCACHE=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_VARLEN_TEST_MODEL="$MODEL"
cmake --build "$build" -- -j$(nproc)
ctest --test-dir "$build" --output-on-failure -R '^(test-ntok-replan|test-ntok-strict|test-varlen-output-(fixed|variable))$'
