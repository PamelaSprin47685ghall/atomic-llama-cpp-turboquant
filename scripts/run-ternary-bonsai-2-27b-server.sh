#!/usr/bin/env bash
# Run Ternary Bonsai 2 27B GGUF (PQ2_0) on Vulkan with llama-server (OpenAI-compatible API)
#
# Robust production architecture aligned with run-qwen38-flash-tp5-server:
#   * Pre-flight sanity checks (refuse if another server or compile job is running)
#   * GPU safety audit (ensure card is idle with clean VRAM before load)
#   * Direct VRAM allocation flags: GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
#   * Hardware FlashAttention compute shaders (-fa on)
#   * Tuned batching (-b 2048 -ub 512) for saturated FWHT & MMQ throughput
#   * Reasoning mode preservation (--reasoning-preserve)
#   * Full lifecycle supervision (background daemon, graceful SIGTERM cleanup trap, /health polling)
#   * Detailed diagnostic metadata & provenance logging

set -euo pipefail

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve binary
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
LOG="${LOG:-/tmp/bonsai_27b_server_${PORT}.log}"
CTX="${CTX:-262144}"
NGL="${NGL:-99}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
KV="${KV:-}"
FA="${FA:-on}"
PARALLEL="${PARALLEL:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# GPU VRAM pinning optimization
export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM="${GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM:-1}"

# --- Pre-flight sanity checks -------------------------------------------------------------
if pgrep -f 'bin/llama-server' >/dev/null; then
    say "REFUSED: Another llama-server is currently running."
    pgrep -af 'bin/llama-server' | head -5
    exit 2
fi

if pgrep -x ninja >/dev/null || pgrep -f 'cc1plus|/bin/ld ' >/dev/null; then
    say "REFUSED: A build is active. Do not mix heavy compiles with full VRAM loads."
    pgrep -af 'ninja|cc1plus' | head -5
    exit 2
fi

# Ensure target Vulkan GPU is idle and clean before launching
# Allow desktop window compositor / browser baselines (< 2.0 GiB)
MAX_BASELINE_VRAM=$((2 * 1024 * 1024 * 1024))

for _ in $(seq 1 10); do
    busy=0
    for c in /sys/class/drm/card[0-9]; do
        [[ -d "$c" ]] || continue
        if [[ -f "$c/device/mem_info_vram_used" && -f "$c/device/gpu_busy_percent" ]]; then
            u=$(cat "$c/device/mem_info_vram_used" 2>/dev/null || echo 0)
            if [[ "$u" =~ ^[0-9]+$ ]]; then
                if [ "$u" -gt "$MAX_BASELINE_VRAM" ]; then
                    busy=1
                fi
            fi
        fi
    done
    [ $busy -eq 0 ] && break
    sleep 1
done

if [ $busy -ne 0 ]; then
    say "REFUSED: GPUs are not idle with clean VRAM."
    exit 3
fi

if [[ ! -f "$MODEL" ]]; then
    say "error: model file not found: ${MODEL}"
    exit 1
fi

if [[ ! -x "$SERVER" ]]; then
    say "error: server binary not found: ${SERVER}"
    exit 1
fi

cleanup() {
    if [ -n "${PID:-}" ]; then
        say "Gracefully terminating llama-server (pid=$PID)..."
        kill -TERM "$PID" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- Launch Server ------------------------------------------------------------------------
say "=== Starting Ternary Bonsai 2 27B Server (Vulkan) ==="
say "Endpoint: http://${HOST}:${PORT}"
say "Model   : ${MODEL}"
say "Config  : CTX=${CTX}, NGL=${NGL}, BATCH=${BATCH}, UBATCH=${UBATCH}, FA=${FA}, PARALLEL=${PARALLEL}"
say "Log     : ${LOG}"
say "Worktree: $(git -C "$ROOT" rev-parse --short HEAD)"
say "====================================================="

stdbuf -oL -eL "$SERVER" \
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
    --reasoning-preserve \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    $EXTRA_ARGS > "$LOG" 2>&1 &

PID=$!
say "llama-server backgrounded (pid=$PID)"

# --- Health check loop --------------------------------------------------------------------
ready=0
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

for i in $(seq 1 120); do
    sleep 1
    if curl -sf --max-time 2 "http://${HEALTH_HOST}:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        say "Server healthy & listening @ $((i * 2))s!"
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        say "FAIL: Process exited unexpectedly."
        tail -n 35 "$LOG"
        exit 5
    fi
done

if [ $ready -ne 1 ]; then
    say "TIMEOUT waiting for server to be healthy."
    tail -n 35 "$LOG"
    exit 6
fi

say "Serving Ternary Bonsai 2 27B on http://${HEALTH_HOST}:${PORT} (Ctrl+C to stop)"
wait "$PID"
