#!/usr/bin/env bash
# Run only on the authorized CPU build host (13022), not on 12022.
set -euo pipefail
: "${NOUT_TAIL_MODEL:?Set NOUT_TAIL_MODEL to a local tiny LLAMA GGUF}"
[[ -f "$NOUT_TAIL_MODEL" ]] || { echo "model not found: $NOUT_TAIL_MODEL" >&2; exit 2; }
command -v ccache >/dev/null
src=$(realpath "$(dirname "$0")/../..")
build=${NOUT_TAIL_BUILD_DIR:-"$src/../build-nout-tail-cpu"}
unset GGML_TP5_RSZ_RT NKV_CAPTURE_KQV
cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF \
  -DGGML_LLAMAFILE=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build "$build" --target test-varlen-output-probe test-cpu-attn-pad test-cpu-attn-mask -j$(nproc)
export NOUT_TAIL_MODEL
# Each named test must exist and pass. Neither fixed gate runs with KQV capture.
for test in test-cpu-attn-pad test-cpu-attn-f16-kqv test-cpu-attn-mask \
            test-cpu-attn-mask-widths test-cpu-attn-ordered-kq \
            test-nout-tail-fixed test-nout-tail-instrumented; do
    ctest --test-dir "$build" --no-tests=error -R "^${test}$" --output-on-failure
done
# Optional, separate upstream-dispatch and variable-token diagnostics:
# ctest --test-dir "$build" --no-tests=error -R '^test-cpu-attn-mask-kq-dispatch$' --output-on-failure
# MODEL="$NOUT_TAIL_MODEL" PROBE="$build/bin/test-varlen-output-probe" LIB="$build/bin" \
#   OUT=/path/to/diagnostic-output bash "$src/tests/nkv-cpu-probe/run_varlen_output_probe.sh" variable
