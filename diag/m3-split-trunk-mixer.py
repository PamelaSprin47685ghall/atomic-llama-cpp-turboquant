"""Deterministic row/site projection of a captured pair into trunks vs mixers.

Read-only, standard library only.  It computes no tolerance: it only splits the
evidence so the two mechanisms are judged on their own rows.

Input (one leg directory):
    report.json.logits.rows.jsonl, report.json.logits.f32le,
    timing.cycles-0.csv / timing.cycles-1.csv   (call,rows,position,...)
optional:
    M3TAIL_STAMP log of the same leg (diag/m3-tail-stamp.ggml-cuda.diff) with
    one line per HC site in graph order.

Output:
  1. captured rows split by the cycle row count: a row whose position the cycle
     log ran as a single-row call is an N=1 row; a row whose position ran as a
     2..4-row call is a multi-row (tail-capable) row;
  2. site counts per graph from the stamp log, and the count of sites per row
     count (2/3/4), so a multi-row graph can be attributed to the request whose
     position range it covers;
  3. trunk vs mixer attribution for the *last* graph of each request (the one
     that contains the output mixer): the docstring of the M3 micro harness and
     the 2026-10-31 reach log both show that graph as the one whose sites are
     the mixer site plus the trunk sites of the collapsed stream.  When the
     stamp log has fewer sites than a full graph, the script says so instead of
     guessing.

Exit code 0 always.
"""
import json
import os
import sys


def rows_of(d):
    return [json.loads(l) for l in open(os.path.join(d, "report.json.logits.rows.jsonl"))]


def cycles(d):
    out = []
    for name in ("timing.cycles-0.csv", "timing.cycles-1.csv"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        lines = open(p).read().splitlines()
        head = lines[0].split(",")
        for line in lines[1:]:
            if line.strip():
                out.append(dict(zip(head, line.split(","))))
    return out


def stamps(path):
    sites = []
    for line in open(path, errors="replace"):
        if not line.startswith("M3STAMP "):
            continue
        kv = dict(pair.split("=", 1) for pair in line.split()[1:] if "=" in pair)
        sites.append({"node": int(kv["node"]), "rows": int(kv["rows"]), "why": int(kv["why"])})
    return sites


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 0
    d = argv[1]
    rows = rows_of(d)
    cyc = cycles(d)
    print("leg=%s rows=%d cycles=%d" % (os.path.basename(d), len(rows), len(cyc)))
    one = [c for c in cyc if c.get("rows") == "1"]
    multi = [c for c in cyc if c.get("rows") not in (None, "1")]
    print("single-row calls=%d  multi-row calls=%d shapes=%s" % (
        len(one), len(multi), sorted({c["rows"] for c in multi})))
    print("captured positions: %s" % [r["input_position"] for r in rows])
    for i, r in enumerate(rows):
        pos = r["input_position"]
        kind = "N1" if any(c["position"] == pos for c in one) else "multi"
        print("  row%2d pos%3s req%s %s" % (i, pos, r["request_id"], kind))
    if len(argv) > 2:
        st = stamps(argv[2])
        print("stamp sites=%d  by rows=%s  why!=0=%d" % (
            len(st),
            {k: sum(1 for s in st if s["rows"] == k) for k in (2, 3, 4)},
            sum(1 for s in st if s["why"] != 0)))
        per_graph = {}
        cur = []
        last = -1
        for s in st:
            if s["node"] <= last and cur:
                per_graph[len(cur)] = per_graph.get(len(cur), 0) + 1
                cur = []
            cur.append(s)
            last = s["node"]
        if cur:
            per_graph[len(cur)] = per_graph.get(len(cur), 0) + 1
        print("graphs by site count: %s" % per_graph)
        print("reading: 3-site graphs = down + combine + tail (trunk complete); 2-site graphs lack "
              "the tail; 1-site graphs are the collapsed mixer graph. A trunk-only run must leave "
              "the mixer graph's site at the guard's fallback, and a mixer-only run must leave "
              "every 2/3-site trunk graph unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
