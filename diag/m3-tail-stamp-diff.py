"""Compare two M3TAIL_STAMP logs of the same 8x2 harness (off/on of one storage knob).

Read-only, standard library only (numpy is absent on this host). Usage:

  python3 m3-tail-stamp-diff.py OFF.stamp ON.stamp [--rows-jsonl off.jsonl]

What it answers, in order:
 1. site alignment: both logs must list the same (node, rows) sequence; a shift means the two
    legs took different graphs (then nothing else is comparable).
 2. placement delta: per site, which of up / lo / xn / out / weights changed its
    (buffer id, offset).  The first site whose placement changed is reported with its index.
 3. lifetime sanity: lo/out ranges must be disjoint within a site (the guard's condition);
    a site listed as fused but overlapping is a bug in the diag build, not a numeric drift.
 4. row k of the diff corresponds to site k of the graph sequence; with --rows-jsonl the
    script prints the resume position (input_position) nearest each changed site, so the KL
    row that first moves can be tied to the site that moved.

Exit codes: 0 same placement, 1 placement differences found (diagnosis), 2 unusable input.
"""
import sys
import json
import re

FIELDS = ("upb", "upo", "lob", "loo", "xnb", "xno", "outb", "outo", "wb", "wo")


def load(path):
    sites = []
    for line in open(path, errors="replace"):
        if not line.startswith("M3STAMP "):
            continue
        kv = dict(pair.split("=", 1) for pair in line.split()[1:] if "=" in pair)
        sites.append({
            "why": int(kv["why"]),
            "node": int(kv["node"]),
            "rows": int(kv["rows"]),
            **{f: int(kv[f]) for f in FIELDS},
        })
    return sites


def overlaps(a, b, size_a, size_b):
    return a < b + size_b and b < a + size_a


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    off = load(argv[1])
    on = load(argv[2])
    if not off or not on:
        print("no M3STAMP lines: was M3TAIL_STAMP=1 and M3TAIL_STAMP_PATH set?")
        return 2
    if len(off) != len(on):
        print("site count differs: off=%d on=%d" % (len(off), len(on)))
        return 2
    keys_off = [(s["node"], s["rows"]) for s in off]
    keys_on = [(s["node"], s["rows"]) for s in on]
    if keys_off != keys_on:
        first = next(i for i, (a, b) in enumerate(zip(keys_off, keys_on)) if a != b)
        print("graph sequence differs at site %d: off=%s on=%s" % (first, keys_off[first], keys_on[first]))
        return 2

    changed = []
    for i, (a, b) in enumerate(zip(off, on)):
        fields = [f for f in FIELDS if a[f] != b[f]]
        if fields:
            changed.append((i, fields, a, b))
        # fused site must still have disjoint lo/out storage
        if b["why"] == 0 and overlaps(b["loo"], b["outo"], 320 * 4 * b["rows"], 2560 * 4 * b["rows"]):
            print("BUG site %d: why=0 but lo and out overlap (loo=%d outo=%d)" % (i, b["loo"], b["outo"]))
    same = len(off) - len(changed)
    print("sites off=%d on=%d identical placement=%d differing=%d" % (len(off), len(on), same, len(changed)))
    for i, fields, a, b in changed[:20]:
        print("  site %d (node %d rows %d): %s" % (i, b["node"], b["rows"],
              ", ".join("%s %d->%d" % (f, a[f], b[f]) for f in fields)))
    if len(changed) > 20:
        print("  ... %d more" % (len(changed) - 20))
    if "--rows-jsonl" in argv:
        rows = [json.loads(l) for l in open(argv[argv.index("--rows-jsonl") + 1])]
        positions = [r["input_position"] for r in rows]
        print("resume positions captured: %s" % positions)
        for i, _, _, _ in changed[:5]:
            k = min(i * len(rows) // max(1, len(off)), len(rows) - 1)
            print("  site %d -> capture row %d at position %s" % (i, k, positions[k]))
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
