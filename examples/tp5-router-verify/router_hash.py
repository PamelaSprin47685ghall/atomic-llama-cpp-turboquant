#!/usr/bin/env python3
"""Reference for the hash lines printed by examples/tp5-router-verify/*.cpp.

Pure helpers that mirror the C++ hashing, so a printed VERIFY_* sha can be
recomputed offline from a saved logits dump. Not used at run time by the
binaries and not imported by the driver script.
"""

from __future__ import annotations

import struct
from typing import Iterable

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def sha_row(floats: Iterable[float]) -> str:
    """VERIFY_ROW sha: FNV-1a over the little-endian F32 bytes of one logits row."""
    return f"{fnv1a64(struct.pack(f'<{len(list(floats))}f', *floats)):016x}"


def sha_floats(floats: Iterable[float]) -> str:
    vals = list(floats)
    return f"{fnv1a64(struct.pack(f'<{len(vals)}f', *vals)):016x}"


def retained_bytes(n_rows: int) -> int:
    """The retained router projection is [512, rows] F32."""
    return 512 * n_rows * 4


if __name__ == "__main__":
    raise SystemExit("formatting helpers only; nothing to run")
