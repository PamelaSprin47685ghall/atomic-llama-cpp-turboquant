#!/usr/bin/env python3
"""Compare fixed-input TP5LOG2 full-logits tapes with bounded memory."""

from __future__ import annotations

import argparse
import array
import json
import math
from pathlib import Path
import struct
import sys


MAX_VOCABULARY = 2_000_000
MAX_INPUT_TOKENS = 2_000_000


def validate_tolerances(atol: float, rtol: float) -> None:
    for name, value in (("atol", atol), ("rtol", rtol)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")


def header(stream) -> tuple[tuple[int, ...], bytes]:
    if stream.read(8) != b"TP5LOG2\0":
        raise ValueError("not a TP5LOG2 tape")
    raw = stream.read(16)
    if len(raw) != 16:
        raise ValueError("truncated header")
    dims = struct.unpack("<4I", raw)
    vocab, prefill, steps, count = dims
    if (not 0 < vocab <= MAX_VOCABULARY or prefill == 0
            or count != prefill + steps or count > MAX_INPUT_TOKENS):
        raise ValueError("invalid tape dimensions")
    tokens = stream.read(count * 4)
    if len(tokens) != count * 4:
        raise ValueError("truncated input tokens")
    if any(token < 0 or token >= vocab for (token,) in struct.iter_unpack("<i", tokens)):
        raise ValueError("input token outside the declared vocabulary")
    return dims, tokens


def compare(reference: Path, candidate: Path, atol: float = 0, rtol: float = 0) -> dict:
    """Return numerical agreement; bitwise and argmax agreement are separate metrics."""
    validate_tolerances(atol, rtol)
    if array.array("f").itemsize != 4:
        raise ValueError("TP5LOG2 requires a 32-bit float array implementation")
    reference, candidate = Path(reference), Path(candidate)
    result = {"reference": str(reference), "candidate": str(candidate), "atol": atol, "rtol": rtol,
              "bitwise": True, "different_values": 0, "outside_tolerance": 0,
              "argmax_mismatches": 0, "argmax_unavailable_rows": 0,
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
            row_nonfinite = False
            for token, (x, y) in enumerate(zip(a, b)):
                if not math.isfinite(x) or not math.isfinite(y):
                    result["nonfinite"] += 1
                    row_nonfinite = True
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
            if row_nonfinite:
                result["argmax_unavailable_rows"] += 1
            else:
                result["argmax_mismatches"] += arg_a != arg_b
        if ref.read(1) or cand.read(1):
            raise ValueError("trailing data after logits")
    result["normalized_mse"] = diff2 / ref2 if ref2 else (0.0 if not diff2 else None)
    result["pass"] = not result["nonfinite"] and not result["outside_tolerance"]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--output", type=Path, help="new JSON report; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        validate_tolerances(args.atol, args.rtol)
    except ValueError as error:
        parser.error(str(error))
    try:
        result = compare(args.reference, args.candidate, args.atol, args.rtol)
    except (OSError, ValueError) as error:
        result = {"pass": False, "error": str(error)}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        except OSError as error:
            result = {"pass": False, "error": f"cannot write new report: {error}"}
            text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(text, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
