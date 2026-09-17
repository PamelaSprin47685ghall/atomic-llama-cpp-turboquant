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
#   ./scripts/run-qwen38-flash-tp5-server.sh --baseline --print-config
#   ./scripts/run-qwen38-flash-tp5-server.sh --baseline
# --baseline is the fixed no-MTP, single-slot, c=256/b=32/ub=32 comparison
# profile. It ignores production TP5_ARGS/EXTRA_ARGS and slot/cache settings,
# logs the resolved command and provenance, and execs the server on stdout.
# MODEL/BIN/HOST/PORT and explicitly supplied debug environment remain visible.

set -euo pipefail

BASELINE=0
PRINT_CONFIG=0
for arg in "$@"; do
    case "$arg" in
        --baseline) BASELINE=1 ;;
        --print-config) PRINT_CONFIG=1 ;;
        *) printf 'Unknown option: %s\nUsage: %s [--baseline [--print-config]]\n' "$arg" "$0" >&2; exit 2 ;;
    esac
done
if [ "$PRINT_CONFIG" -eq 1 ] && [ "$BASELINE" -eq 0 ]; then
    printf '%s\n' '--print-config requires --baseline' >&2
    exit 2
fi

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
export GGML_TP5_MERGE_SUBMIT=${GGML_TP5_MERGE_SUBMIT:-1}
export GGML_TP5_RELAY=${GGML_TP5_RELAY:-off}
export GGML_VK_CMD_REPLAY=${GGML_VK_CMD_REPLAY:-1}
export GGML_VK_ALLOW_GRAPHICS_QUEUE=${GGML_VK_ALLOW_GRAPHICS_QUEUE:-1}
TP5_ARGS=${TP5_ARGS:---tp5 qwen4exp-af --tp5-wire f16 --tp5-sync timeline -md "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0}

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }

if [ "$BASELINE" -eq 1 ]; then
    # Set driver options before any process can initialize Vulkan. An explicit
    # empty RADV_DEBUG is retained for controlled comparisons, not silently fixed.
    export RADV_DEBUG=${RADV_DEBUG-nobolist}
    export GGML_TP5_ISOLATE_BO=${GGML_TP5_ISOLATE_BO:-1}
    export GGML_VK_DISABLE_MMVQ=${GGML_VK_DISABLE_MMVQ:-1}
    export GGML_TP5_CMD_REPLAY=${GGML_TP5_CMD_REPLAY:-1}
    export GGML_TP5_WIRE=f16 GGML_TP5_SYNC=timeline GGML_TP5_RELAY=off
    BIN=$(realpath -e "$BIN")
    BIN_DIR=$(dirname "$BIN")
    export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    baseline_cmd=("$BIN" -m "$MODEL" -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4
        --split-mode tensor --fit off --tp5 qwen4exp-af --tp5-sync timeline --tp5-wire f16
        -ngl 999 -c 256 -b 32 -ub 32 --no-mmap --no-host --spec-type none -np 1
        --host "$HOST" --port "$PORT")
    say 'PROFILE=pure-tp-baseline; production TP5_ARGS/EXTRA_ARGS/slot/cache overrides are not used'
    printf 'command:'; printf ' %q' "${baseline_cmd[@]}"; printf '\n'
    for key in RADV_DEBUG GGML_TP5_ISOLATE_BO GGML_TP5_SYNC GGML_TP5_WIRE GGML_TP5_RELAY \
        GGML_VK_CMD_REPLAY GGML_TP5_CMD_REPLAY GGML_VK_DISABLE_MMVQ \
        GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM GGML_VK_ALLOW_GRAPHICS_QUEUE \
        GGML_VK_DISABLE_HC_SUM GGML_VK_DISABLE_PRODUCER_WIRE GGML_VK_DISABLE_ATTENTION_COMPACT \
        GGML_TP5_CHAIN_CACHE GGML_TP5_SHARE_P1 GGML_TP5_PROFILE GGML_VK_PERF_LOGGER \
        GGML_VK_HC_DOWN_WG GGML_VK_HC_UP_R320 GGML_VK_HC_DOT GGML_VK_MOE_DOWN_K128 \
        GGML_VK_ROUTER_WEIGHTS GGML_VK_ROUTER_TILING GGML_VK_DISABLE_ROUTER_SUBGROUP GGML_VK_ROUTER_DIAG_LAYERS \
        GGML_TP5_GDN_HEADMAP GGML_TP5_QSA_HEADMAP \
        GGML_VK_VISIBLE_DEVICES VK_DRIVER_FILES VK_ICD_FILENAMES LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT; do
        printf 'environment %s=%q\n' "$key" "${!key-<unset>}"
    done
    printf 'worktree_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    if git -C "$REPO" diff --quiet HEAD --; then
        printf 'tracked_worktree_dirty=0\n'
    else
        printf 'tracked_worktree_dirty=1\n'
    fi
    printf 'tracked_diff_sha256='
    git -C "$REPO" diff --no-ext-diff --binary HEAD -- | sha256sum
    # A commit/build banner alone cannot identify dirty-tree builds or changed
    # shared libraries. Record the actual binary and local implementation DSOs.
    say 'Worktree metadata does not certify existing binary build provenance; retain the build log and compare these executable/DSO hashes.'
    for binary in "$BIN" "$BIN_DIR/libllama-server-impl.so" "$BIN_DIR/libllama.so" \
        "$BIN_DIR/libggml-base.so" "$BIN_DIR/libggml-vulkan.so" "$BIN_DIR/libggml-cpu.so"; do
        [ ! -f "$binary" ] || sha256sum "$binary"
    done
    if command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -W -f='installed_package ${binary:Package}=${Version}\n' mesa-vulkan-drivers 2>/dev/null || true
    fi
    say 'RADV_DEBUG is a driver debug request, not proof of the active BO policy. Revalidate after driver upgrades; newer Mesa removed nobolist.'
    if [ "$PRINT_CONFIG" -eq 1 ]; then
        exit 0
    fi
fi

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
        if ! read -r u < "/sys/class/drm/$c/device/mem_info_vram_used" ||
           ! read -r gpu_busy < "/sys/class/drm/$c/device/gpu_busy_percent" ||
           [[ ! "$u" =~ ^[0-9]+$ || ! "$gpu_busy" =~ ^[0-9]+$ ]]; then
            say "REFUSED: Cannot read GPU safety state for $c."
            exit 3
        fi
        if [ "$u" -gt 200000000 ] || [ "$gpu_busy" -ne 0 ]; then
            busy=1
        fi
    done
    [ $busy -eq 0 ] && break
    sleep 1
done

if [ $busy -ne 0 ]; then
    say "REFUSED: GPUs are not idle with clean VRAM across all 5 GPUs."
    exit 3
fi

if [ "$BASELINE" -eq 1 ]; then
    exec "${baseline_cmd[@]}"
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
    -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4 \
    --fit off \
    $TP5_ARGS \
    -sm tensor \
    -ngl 999 \
    -lm none \
    --no-host --no-repack \
    -fa on \
    -ctk q8_0 -ctv turbo4 \
    -b 2048 -ub 512 \
    --reasoning-preserve \
    -c "$N_CTX_SLOT" -np "$N_SLOTS" \
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
