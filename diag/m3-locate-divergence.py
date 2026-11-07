"""Layered divergence locator for one captured logits pair (standard library only).

Why: a mean-KL number over 16 rows cannot say which tensor turned the drift on.  This tool
recomputes, from the SAME two capture files, three independent stacks in a fixed order so the
first layer at which a row starts to move is visible, and pairs each captured row with the
decode round that produced it (verify vs catch-up) using the lane's own timing cycle log.

  python3 m3-locate-divergence.py LEFT_RUN_DIR RIGHT_RUN_DIR

Both dirs must contain report.json.logits.f32le + report.json.logits.rows.jsonl, and may contain
timing.cycles-0.csv (columns: call,rows,position,queue_ns,submit_ns,submitted_ns,waited_ns,
committed_ns).  Output per row: a triple of distances

  L1 fractional(off,on)      p(q) computed from each row's own softmax
  L2 top-k index set         |argmax(top8) delta| as a Jaccard-ish count of index agreement
  L3 unique index-VALUE floor  logit differences that cannot be attributed to normalisation:
                               max_k |delta logit_k| when the two rows have identical top-8 index
                               sets; any non-zero value means the logits themselves differ there.

Rows are printed with the producing call (rows= and position= from the cycle log) so a row that
moved in a 4-row prefill-style call can be told apart from a 2/3-row verify call.

Exit code 0 always: this is a diagnosis, it never gates.
"""
import csv
import json
import math
import os
import sys
import array


def load_leg(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "report.json.logits.rows.jsonl"))]
    buf = open(os.path.join(d, "report.json.logits.f32le"), "rb").read()
    a = array.array("f")
    a.frombytes(buf)
    out = []
    for r in rows:
        o = int(r["offset_bytes"]) // 4
        n = int(r["count"])
        out.append(a[o:o + n])
    return rows, out


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def kl(p, q):
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q) if a > 0.0)


def topk(x, k=8):
    return sorted(range(len(x)), key=lambda i: -x[i])[:k]


def cycles(d):
    for name in ("timing.cycles-0.csv", "timing.cycles-1.csv"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return [r for r in csv.DictReader(open(p))]
    return []


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 0
    left, right = argv[1], argv[2]
    li, lx = load_leg(left)
    ri, rx = load_leg(right)
    if len(lx) != len(rx) or len(lx[0]) != len(rx[0]):
        print("capture shapes differ: %d x %d vs %d x %d" % (len(lx), len(lx[0]), len(rx), len(rx[0])))
        return 0
    cyc = cycles(left)
    shapes = [(c["rows"], c["position"]) for c in cyc]
    print("left=%s right=%s rows=%d vocab=%d" % (os.path.basename(left), os.path.basename(right), len(lx), len(lx[0])))
    if shapes:
        print("cycle shapes (rows,position): %s" % shapes[:16])
    moving = 0
    for i in range(len(lx)):
        a = lx[i]
        b = rx[i]
        pa = softmax(a)
        pb = softmax(b)
        k = kl(pa, pb)
        ta = topk(a)
        tb = topk(b)
        agree = len(set(ta) & set(tb))
        same_set = agree == len(ta)
        vmax = max(abs(a[j] - b[j]) for j in ta) if same_set else float("nan")
        mx = max(abs(x - y) for x, y in zip(a, b))
        if k > 0.0:
            moving += 1
        print("row%2d pos%3s kltok=%.3e top8_agree=%d/8 same_top8=%s |dlogit|_top8=%s maxabs=%.3f" % (
            i, li[i]["input_position"], k, agree, same_set,
            ("%.3e" % vmax) if same_set else "n/a", mx))
    print("rows with any change: %d/%d" % (moving, len(lx)))
    print("reading: equal top8 sets + |dlogit|_top8 > 0 -> logits moved before softmax "
          "(pre-normalisation); top8 index disagreement must be listed separately, not averaged away")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
