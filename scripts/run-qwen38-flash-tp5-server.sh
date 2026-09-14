#!/usr/bin/env bash
# Qwen3.8-Flash-Next-APEX-I-Compact on 5x RX 6800 via Vulkan Tensor Parallelism (TP5).
#
# Aligned with the production baseline flags:
#   * -lm none: Direct physical loading (avoid mmap virtual pageout faults & DMAR issues)
#   * --no-host --no-repack: Keep weights directly in device VRAM, skip host staging & repack
#   * -fa on: Hardware FlashAttention compute shaders
#   * -ctk q8_0 -ctv turbo4: TurboQuant asymmetric compression for KV cache (Q8_0 K + Turbo4 V)
#   * -b 2048 -ub 512: Balanced token batching / microbatching
#   * --reasoning-preserve: Native chat template reasoning mode preservation
#   * GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1: Keep all allocations pinned in fast device VRAM
#   * GGML_VK_ALLOW_GRAPHICS_QUEUE=1: Use Graphics Queue for reliable cross-GPU DMA-BUF P2P PUSH
#
# Usage:
#   ./scripts/run-qwen38-flash-tp5-server.sh
#   PORT=8099 ./scripts/run-qwen38-flash-tp5-server.sh

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-/home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf}
BIN=${BIN:-$REPO/build-tp5/bin/llama-server}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8095}
LOG=${LOG:-/tmp/qwen38_tp5_$PORT.log}
N_SLOTS=${N_SLOTS:-11}
N_CTX_SLOT=${N_CTX_SLOT:-262144}
MTP_MODEL=${MTP_MODEL:-/home/kunweiz/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf}
KV_SIZE=${KV_SIZE:-1441792}
EXTRA_ARGS=${EXTRA_ARGS:-}

export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=${GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM:-1}
# CLI --tp5* flags (common_tp5_apply_env) override these when passed on the command line.
export GGML_TP5_WIRE=${GGML_TP5_WIRE:-f16}
export GGML_TP5_SYNC=${GGML_TP5_SYNC:-timeline}
export GGML_TP5_RELAY=${GGML_TP5_RELAY:-off}
export GGML_VK_CMD_REPLAY=${GGML_VK_CMD_REPLAY:-1}
export GGML_VK_ALLOW_GRAPHICS_QUEUE=${GGML_VK_ALLOW_GRAPHICS_QUEUE:-1}
TP5_ARGS=${TP5_ARGS:---tp5 qwen4exp-af --tp5-wire f16 --tp5-sync timeline -md "$MTP_MODEL" --spec-type draft-mtp,ngram-mod --spec-draft-n-max 5}

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }

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

# Ensure all 5 GPUs are completely idle before launching
for _ in $(seq 1 30); do
    busy=0
    for c in card1 card2 card3 card4 card5; do
        u=$(cat /sys/class/drm/$c/device/mem_info_vram_used 2>/dev/null || echo 0)
        [ "$u" -gt 200000000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 1
done

if [ $busy -ne 0 ]; then
    say "REFUSED: VRAM is not clean across all 5 GPUs."
    exit 3
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

# --- Launch TP5 Server --------------------------------------------------------------------
say "Launching TP5 server on $HOST:$PORT (model: $(basename "$MODEL"))..."
stdbuf -oL -eL "$BIN" -m "$MODEL" \
    $TP5_ARGS \
    -sm tensor \
    -ngl 999 \
    -lm none \
    --no-host --no-repack \
    -fa on \
    -ctk q8_0 -ctv turbo4 \
    -b 2048 -ub 512 \
    --reasoning-preserve \
    -c "$N_CTX_SLOT" -np "$N_SLOTS" --kv-size "$KV_SIZE" \
    --host "$HOST" --port "$PORT" \
    $EXTRA_ARGS > "$LOG" 2>&1 &

PID=$!
say "llama-server started (pid=$PID, log=$LOG)"

# --- Health check loop --------------------------------------------------------------------
ready=0
for i in $(seq 1 120); do
    sleep 2
    if curl -sf --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        ready=1
        say "Server healthy & listening @ $((i * 2))s!"
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        say "FAIL: Process exited unexpectedly."
        tail -n 25 "$LOG"
        exit 5
    fi
done

if [ $ready -ne 1 ]; then
    say "TIMEOUT waiting for server to be healthy."
    tail -n 25 "$LOG"
    exit 6
fi

say "Serving TP5 on http://$HOST:$PORT (Ctrl+C to stop)"
wait "$PID"
