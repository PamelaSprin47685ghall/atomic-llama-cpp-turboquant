#!/usr/bin/env bash
# Manual GPU recovery. A read-only audit is the default; PCI mutation requires an
# explicit physical card or `all`, and the root helper refuses active DRM clients.
set -euo pipefail

repo=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")
if [ "$#" -eq 0 ]; then
    exec "$repo/scripts/gpu-idle-audit.sh"
fi
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [card1..card5|all|check-clocks-only]" >&2
    exit 2
fi
case "$1" in
    card1|card2|card3|card4|card5|all|check-clocks-only) ;;
    *) echo "Invalid recovery target: $1" >&2; exit 2 ;;
esac

sudo -n "$repo/scripts/reset-gpu-pci.sh" "$1"
