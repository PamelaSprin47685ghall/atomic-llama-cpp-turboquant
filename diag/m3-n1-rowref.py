"""N1 row-reference check for one captured logits pair (standard library only).

Purpose: prove or disprove "the fused tail's per-row arithmetic equals the N=1
reference arithmetic" on the truncated row set a capture actually contains, per
row, without any token-level threshold being relaxed.  A strict numeric-order
claim then has a mechanism-level witness instead of an averaged KL.

Input: two directories that each hold
    report.json.logits.rows.jsonl   (one JSON object per captured row)
    report.json.logits.f32le        (concatenated row logits, float32 LE)

Expected pairing (see audit §6 "N1 reference"):
    LEFT  = candidate leg whose HC sites are the ones under test
    RIGHT = same model/env, but with the tail fusion unable to fire for the
            row's shape, i.e. the N=1 reference.  Two admissible sources:
              a) the same leg run with --predict 1 so every verified row is a
                 single-row graph (the docstring of the M3 micro harness uses
                 exactly this "row r reproduced with nt=1" convention), or
              b) the per-row N=1 image bundle the micro harness writes
                 (chk-N1-r<k>.bin), one file per row order.
Anything else is not an N=1 reference and the script refuses to call it one.

Report per row:
    dlogit_max      max |left - right| over the vocabulary
    ulp_max         the same difference expressed in float32 ulps of the right
                    value (exponent-step count), so "a few ulp" is measurable
    bitdiff         count of exactly different float32 bit patterns
    top1            whether argmax agrees
Exit code 0 always: this is a measurement, never a gate.

Why ulps: a fused kernel that reorders the same additions is allowed to differ
by a bounded number of ulps; a different reduction or a stale/aliased read is
not.  Reporting the exponent step keeps that distinction visible without
inventing a tolerance.
"""
import array
import json
import math
import os
import struct
import sys


def _load_dir(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "report.json.logits.rows.jsonl"))]
    a = array.array("f")
    a.frombytes(open(os.path.join(d, "report.json.logits.f32le"), "rb").read())
    out = []
    for r in rows:
        o = int(r["offset_bytes"]) // 4
        n = int(r["count"])
        out.append(a[o:o + n])
    return rows, out


def _load_bins(pattern_dir, prefix):
    """chk-N1-r<k>.bin: raw float32 rows in row order, one file per row."""
    files = sorted(f for f in os.listdir(pattern_dir) if f.startswith(prefix) and f.endswith(".bin"))
    out = []
    for f in files:
        a = array.array("f")
        a.frombytes(open(os.path.join(pattern_dir, f), "rb").read())
        out.append(a)
    return out


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _ulp(a, b):
    """Exponent-step distance between two float32 values (0 == identical)."""
    ia, ib = _bits(a), _bits(b)
    if (ia >> 31) != (ib >> 31):
        return 1 << 30  # different signs: not a rounding difference
    return abs(ia - ib)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 0
    left, right = argv[1], argv[2]
    li, lx = _load_dir(left)
    if os.path.isdir(right) and any(f.endswith(".jsonl") for f in os.listdir(right)):
        _, rx = _load_dir(right)
        label = "leg"
    else:
        rx = _load_bins(right, "chk-N1-r")
        label = "micro-N1-bins"
    if len(rx) < len(lx):
        print("reference has %d rows, capture has %d: refusing a partial reference" % (len(rx), len(lx)))
        return 0
    print("candidate=%s reference=%s (%s), rows=%d" % (os.path.basename(left), right, label, len(lx)))
    worst_ulp = 0
    worst_bit = 0
    for i in range(len(lx)):
        a = lx[i]
        b = rx[i]
        if len(a) != len(b):
            print("row %d vocabulary mismatch %d vs %d" % (i, len(a), len(b)))
            return 0
        dmax = 0.0
        umax = 0
        bits = 0
        for x, y in zip(a, b):
            if x == y:
                continue
            bits += 1
            d = abs(x - y)
            if d > dmax:
                dmax = d
            u = _ulp(x, y)
            if u > umax:
                umax = u
        top1 = max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        worst_ulp = max(worst_ulp, umax)
        worst_bit = max(worst_bit, bits)
        print("row%2d pos%3s dlogit_max=%.3e ulp_max=%d bitdiff=%d/%d top1=%s" % (
            i, li[i]["input_position"], dmax, umax, bits, len(a), top1))
    print("summary: worst ulp step=%d, worst bit differences=%d" % (worst_ulp, worst_bit))
    print("reading: ulp_max within the micro baseline's 1-3 ulp class + top1 all true => the fused "
          "path is the same arithmetic as N=1; a larger step, a vocabulary-sized bitdiff, or a "
          "top1 change names a real mechanism difference and must be reported as such.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
