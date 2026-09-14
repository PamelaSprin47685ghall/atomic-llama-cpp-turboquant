#!/usr/bin/env bash
# Emit resolved TP5 configuration JSON (TP5.md §3.1). No GPU required.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-/tmp/tp5-resolved-config.json}

git_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
bin=${BIN:-$REPO/build-tp5/bin/llama-server}

cat >"$OUT" <<EOF
{
  "source_commit": "$git_commit",
  "binary": "$bin",
  "GGML_TP5_WIRE": "${GGML_TP5_WIRE:-f16}",
  "GGML_TP5_SYNC": "${GGML_TP5_SYNC:-timeline}",
  "GGML_TP5_RELAY": "${GGML_TP5_RELAY:-off}",
  "GGML_VK_CMD_REPLAY": "${GGML_VK_CMD_REPLAY:-1}",
  "GGML_TP5_PROFILE": "${GGML_TP5_PROFILE:-}",
  "GGML_VK_ALLOW_GRAPHICS_QUEUE": "${GGML_VK_ALLOW_GRAPHICS_QUEUE:-}",
  "note": "target-machine configuration; production fast path: timeline + f16 wire"
}
EOF

echo "wrote $OUT"
