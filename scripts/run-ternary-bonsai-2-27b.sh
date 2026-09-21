#!/usr/bin/env bash
# Run Ternary Bonsai 2 27B GGUF (PQ2_0) on Vulkan GPU with optimized prefill & decode parameters
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve binary (prefer build-test or build)
if [[ -x "${ROOT}/build-test/bin/llama-cli" ]]; then
    DEFAULT_BIN="${ROOT}/build-test/bin/llama-cli"
elif [[ -x "${ROOT}/build/bin/llama-cli" ]]; then
    DEFAULT_BIN="${ROOT}/build/bin/llama-cli"
else
    DEFAULT_BIN="${ROOT}/build-test/bin/llama-cli"
fi

LLAMA_BIN="${LLAMA_BIN:-$DEFAULT_BIN}"
MODEL="${MODEL:-/home/kunweiz/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf}"

# Tuned parameters
# -c: context window (262144)
CTX="${CTX:-262144}"
# -ngl: offload all layers to GPU
NGL="${NGL:-99}"
# -b / -ub: physical and logical batch size tuned for Vulkan MMQ / FWHT prefill saturation
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
# Flash attention
FA="${FA:-on}"
# -kv auto: auto-fit unified KV cache
# If KV is set, pass -kv "$KV" (e.g. KV="auto")
KV="${KV:-}"

# Model sampling recommendations (Thinking mode: temp 1.0, top_p 0.95, top_k 20)
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

if [[ ! -f "$MODEL" ]]; then
    echo "error: model file not found: ${MODEL}" >&2
    exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "error: executable not found: ${LLAMA_BIN}" >&2
    exit 1
fi

echo "=== Running Ternary Bonsai 2 27B (PQ2_0) on Vulkan ===" >&2
echo "Binary : ${LLAMA_BIN}" >&2
echo "Model  : ${MODEL}" >&2
echo "Config : CTX=${CTX}, NGL=${NGL}, BATCH=${BATCH}, UBATCH=${UBATCH}, FA=${FA}" >&2
echo "Sampling: TEMP=${TEMP}, TOP_P=${TOP_P}, TOP_K=${TOP_K}" >&2
echo "=====================================================" >&2

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    -c "$CTX" \
    -ngl "$NGL" \
    -b "$BATCH" \
    -ub "$UBATCH" \
    -fa "$FA" \
    ${KV:+-kv "$KV"} \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    "$@"
