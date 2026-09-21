#!/usr/bin/env bash
# Run Ternary Bonsai 2 27B GGUF (PQ2_0) on Vulkan with llama-server (OpenAI-compatible API)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve binary (prefer build-test or build)
if [[ -x "${ROOT}/build-test/bin/llama-server" ]]; then
    DEFAULT_SERVER="${ROOT}/build-test/bin/llama-server"
elif [[ -x "${ROOT}/build/bin/llama-server" ]]; then
    DEFAULT_SERVER="${ROOT}/build/bin/llama-server"
else
    DEFAULT_SERVER="${ROOT}/build-test/bin/llama-server"
fi

SERVER="${LLAMA_SERVER:-$DEFAULT_SERVER}"
MODEL="${MODEL:-/home/kunweiz/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX="${CTX:-262144}"
NGL="${NGL:-99}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
KV="${KV:-}"
FA="${FA:-on}"
PARALLEL="${PARALLEL:-1}"

if [[ ! -f "$MODEL" ]]; then
    echo "error: model file not found: ${MODEL}" >&2
    exit 1
fi

if [[ ! -x "$SERVER" ]]; then
    echo "error: server binary not found: ${SERVER}" >&2
    exit 1
fi

echo "=== Starting Ternary Bonsai 2 27B Server (Vulkan) ===" >&2
echo "Endpoint: http://${HOST}:${PORT}" >&2
echo "Model   : ${MODEL}" >&2
echo "Config  : CTX=${CTX}, NGL=${NGL}, BATCH=${BATCH}, UBATCH=${UBATCH}, FA=${FA}" >&2
echo "=====================================================" >&2

exec "$SERVER" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -c "$CTX" \
    -ngl "$NGL" \
    -b "$BATCH" \
    -ub "$UBATCH" \
    ${KV:+-kv "$KV"} \
    -fa "$FA" \
    --parallel "$PARALLEL" \
    -np "$PARALLEL" \
    --cont-batching \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    "$@"
