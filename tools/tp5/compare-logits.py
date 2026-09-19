#!/usr/bin/env python3
"""Compare TP5LOG2 fixed-input tapes, using bounded memory and no dependencies."""
from __future__ import annotations

import argparse
import array
import json
import math
from pathlib import Path
import struct
import sys


def header(stream) -> tuple[tuple[int, ...], bytes]:
    if stream.read(8) != b"TP5LOG2\0":
        raise ValueError("not a TP5LOG2 tape")
    raw = stream.read(16)
    if len(raw) != 16:
        raise ValueError("truncated header")
    dims = struct.unpack("<4I", raw)
    vocab, prefill, steps, count = dims
    if not 0 < vocab <= 2_000_000 or count != prefill + steps or count > 2_000_000:
        raise ValueError("invalid tape dimensions")
    tokens = stream.read(count * 4)
    if len(tokens) != count * 4:
        raise ValueError("truncated input tokens")
    return dims, tokens


def compare(reference: Path, candidate: Path, atol: float = 0, rtol: float = 0) -> dict:
    result = {"reference": str(reference), "candidate": str(candidate), "atol": atol, "rtol": rtol,
              "bitwise": True, "different_values": 0, "outside_tolerance": 0, "argmax_mismatches": 0,
              "max_abs": 0.0, "max_rel": 0.0, "nonfinite": 0, "first_different_row": None}
    diff2 = ref2 = 0.0
    with reference.open("rb") as ref, candidate.open("rb") as cand:
        dims, tokens = header(ref)
        if header(cand) != (dims, tokens):
            raise ValueError("different dimensions or teacher-forced input tokens; not a paired comparison")
        vocab, prefill, steps, _ = dims
        result.update(rows=steps + 1, vocabulary=vocab, prefill=prefill)
        for row in range(steps + 1):
            left, right = ref.read(vocab * 4), cand.read(vocab * 4)
            if len(left) != vocab * 4 or len(right) != vocab * 4:
                raise ValueError(f"truncated logits at row {row}")
            if left != right:
                result["bitwise"] = False
                if result["first_different_row"] is None:
                    result["first_different_row"] = row
            a, b = array.array("f"), array.array("f")
            a.frombytes(left)
            b.frombytes(right)
            if sys.byteorder != "little":
                a.byteswap()
                b.byteswap()
            best_a = best_b = -math.inf
            arg_a = arg_b = -1
            for token, (x, y) in enumerate(zip(a, b)):
                if not math.isfinite(x) or not math.isfinite(y):
                    result["nonfinite"] += 1
                    continue
                if x > best_a:
                    best_a, arg_a = x, token
                if y > best_b:
                    best_b, arg_b = y, token
                delta = abs(x - y)
                ref2 += x * x
                diff2 += delta * delta
                if delta:
                    result["different_values"] += 1
                    result["max_abs"] = max(result["max_abs"], delta)
                    result["max_rel"] = max(result["max_rel"], delta / max(abs(x), 1e-12))
                    if delta > atol + rtol * abs(x):
                        result["outside_tolerance"] += 1
            result["argmax_mismatches"] += arg_a != arg_b
        if ref.read(1) or cand.read(1):
            raise ValueError("trailing data after logits")
    result["normalized_mse"] = diff2 / ref2 if ref2 else (0.0 if not diff2 else None)
    result["pass"] = not result["nonfinite"] and not result["outside_tolerance"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not math.isfinite(t) or t < 0 for t in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    try:
        result = compare(args.reference, args.candidate, args.atol, args.rtol)
    except (OSError, ValueError) as error:
        result = {"pass": False, "error": str(error)}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(text, end="")
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
