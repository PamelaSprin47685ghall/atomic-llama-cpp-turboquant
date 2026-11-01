#!/usr/bin/env python3
"""N3 verify-logits gate (CPU only, stdlib only).

Compares the two captures of one run directory:

  rows.jsonl      one JSON object per row, written by n3_verify_capture.cpp:
                  side / window / seq / step / row / input_position / input_token /
                  vocabulary / offset_bytes / count / dtype
  logits.f32le    IEEE binary32 little-endian, rows at offset_bytes

  side "n3"  rows of a verify step, decoded as one batch of [current, draft0, draft1]
  side "n1"  the same rows decoded one at a time from a copy of the same state

The gate aligns the two sides by (window, seq, step, row, input_position, input_token) and
REFUSES (exit 2) any pair that is not aligned: a missing row, a duplicated key, a position
mismatch, an offset that does not tile, a count that is not the vocabulary, a non-f32le dtype.
It never repairs a capture.

Classes (ACCEPTANCE.md section 4):
  exact               0 differing bits on every aligned row
  numeric-order-only  mean KL < 1e-6, max row KL < 1e-5, top-1 >= 99.9 %,
                      flips only where the reference margin <= 0.10
  behaviour-changing  mean KL in [1e-4, 1e-3], max row KL <= 0.021

The gate cannot show that a row is the model's token logits; it shows that two captures of the
same quantity agree within a class. Authenticity rests on the producer (a row is written from
`llama_get_logits_ith` right after the step it belongs to), which is a claim this tool checks
only for internal consistency.

Exit: 0 PASS, 1 FAIL, 2 REFUSED (not an admissible pair), 3 UNPROVEN (a file is missing).
"""

from __future__ import annotations

import struct

import argparse
import json
import math
import os
import sys

KEY_FIELDS = ("window", "seq", "step", "row", "input_position", "input_token")
THRESHOLDS = {
    "exact": dict(mean_kl=0.0, max_kl=0.0, top1=1.0, flip_margin=0.0),
    "numeric-order-only": dict(mean_kl=1e-6, max_kl=1e-5, top1=0.999, flip_margin=0.10),
    "behaviour-changing": dict(mean_kl=1e-3, max_kl=0.021, top1=0.0, flip_margin=0.10),
    "behaviour-changing-min": dict(mean_kl=1e-4, max_kl=0.021, top1=0.0, flip_margin=0.10),
}


class Refused(Exception):
    pass


def read_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise Refused(f"{path}:{lineno}: not JSON ({err})")
            for field in KEY_FIELDS + ("side", "vocabulary", "offset_bytes", "count", "dtype"):
                if field not in obj:
                    raise Refused(f"{path}:{lineno}: missing field {field!r}")
            if obj["dtype"] != "f32le":
                raise Refused(f"{path}:{lineno}: dtype {obj['dtype']!r} is not f32le")
            rows.append(obj)
    return rows


def physical_layout(rows):
    """The producer's physical contract: ONE file, rows in emission order, one global counter.

    Both producers (n3_verify_capture.cpp and n3_verify_two_model.cpp) hold a single
    `n_bytes` that every row increments by `vocabulary * 4`, and they fwrite the rows in the
    order they are produced. The file is therefore one interleaved stream in which
    rows.jsonl order == file order, and every record - prefill rows included - covers
    `[offset_bytes, offset_bytes + count*4)` with NO overlap and NO hole:

        offsets must be exactly 0, V*4, 2*V*4, ... over ALL records, in file order,
        and the file must be exactly len(rows) * V * 4 bytes.

    A capture whose offsets are per side, reversed, overlapping or gapped is refused: reading it
    would silently pair a row with another row's bytes.
    """
    if not rows:
        raise Refused("empty capture")
    vocab = None
    expect = 0
    for i, obj in enumerate(rows):
        v = int(obj["vocabulary"])
        if v <= 0:
            raise Refused(f"row {i}: empty vocabulary")
        if vocab is None:
            vocab = v
        elif v != vocab:
            raise Refused(f"row {i}: vocabulary changes from {vocab} to {v}")
        if int(obj["count"]) != vocab:
            raise Refused(f"row {i}: count {obj['count']} is not the vocabulary {vocab}")
        off = int(obj["offset_bytes"])
        if off != expect:
            kind = "overlaps or precedes" if off < expect else "leaves a hole before"
            raise Refused(f"row {i}: offset {off} {kind} the expected {expect} (global tiling)")
        expect += vocab * 4
    return vocab, expect


def side_rows(rows, side: str):
    """Rows of ONE side, keyed by identity. The byte layout is validated globally, not here."""
    out = {}
    for i, obj in enumerate(rows):
        if obj["side"] != side:
            continue
        key = tuple(int(obj[f]) for f in KEY_FIELDS)
        if key in out:
            raise Refused(f"{side}: duplicated key {key}")
        out[key] = obj
    if not out:
        raise Refused(f"no {side!r} rows")
    return out


def read_logits(path: str, vocab: int, n_rows: int, total_bytes: int):
    """Load the whole stream and return {offset_bytes: row}, one entry per record."""
    size = os.path.getsize(path)
    # name the two directions differently: a short file means rows address bytes that were never
    # written, a long file means the stream holds records rows.jsonl does not describe
    if size < total_bytes:
        raise Refused(f"{path}: {size} bytes, rows address {total_bytes} "
                      f"(rows * vocabulary * 4): the file is SHORTER than the last row")
    if size > total_bytes:
        raise Refused(f"{path}: {size} bytes, rows address {total_bytes} "
                      f"(rows * vocabulary * 4): the file holds records no row describes")
    with open(path, "rb") as f:
        blob = f.read()
    values = struct.unpack(f"<{n_rows * vocab}f", blob)
    return {(i * vocab * 4): values[i * vocab:(i + 1) * vocab] for i in range(n_rows)}


def write_record(path, record):
    """Write the verdict record atomically, or fail closed.

    Every exit path that has a record goes through here, so a refusal can never leave a stale
    PASS behind and a half-written file can never be read as a verdict:
      * the record is written to a sibling temp file and renamed over the target (atomic on the
        same filesystem);
      * if the destination already exists it is REPLACED only by a complete new record - the
        rename is the only mutation, so a reader sees either the old file or the new one;
      * a failure to write raises, and the caller turns that into exit 2 with a REFUSED line
        (never a silent success).
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as err:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise Refused(f"cannot write the verdict record {path} ({err})")


def finish(record, details):
    """The ONE place a terminal verdict is printed.

    The whole run - including the record write - is decided before this is called, so the
    process prints exactly one `N3GATE <verdict>` line and it is the verdict of this invocation.
    A consumer that reads the first N3GATE line therefore cannot be misled by a later REFUSED:
    there is never a later one on a path that reached a verdict.
    """
    print(f"N3GATE {record['verdict']} " + " ".join(details))
    return 0 if record["verdict"] == "PASS" else 1


def argmax_margin(row):
    best = second = -math.inf
    best_i = -1
    for i, v in enumerate(row):
        if not math.isfinite(v):
            raise Refused("nonfinite logit")
        if v > best:
            second = best
            best = v
            best_i = i
        elif v > second:
            second = v
    return best_i, best - second


def row_kl(ref, cand):
    """KL(ref || cand) over the full vocabulary, computed in double with a max shift.

    A non-finite logit is refused before it can reach any arithmetic: max() over a row that
    holds NaN is not a reliable extremum, and the shift below would turn NaN into a silent 0
    weight instead of a refusal. The check therefore runs first, on both rows.
    """
    for a in ref:
        if not math.isfinite(a):
            raise Refused("nonfinite logit in the reference row")
    for b in cand:
        if not math.isfinite(b):
            raise Refused("nonfinite logit in the candidate row")
    rmax = max(ref)
    cmax = max(cand)
    zr = zc = 0.0
    cross = 0.0
    for a, b in zip(ref, cand):
        wa = math.exp(a - rmax)
        wb = math.exp(b - cmax)
        zr += wa
        zc += wb
        cross += wa * (a - b)
    return cross / zr + (cmax + math.log(zc)) - (rmax + math.log(zr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory written by n3_verify_capture")
    ap.add_argument("--class", dest="cls", default="numeric-order-only", choices=sorted(THRESHOLDS))
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # Every exit that has a --json path must leave a record there: an UNPROVEN exit used to write
    # nothing, so a verdict file from an earlier run survived and could be read as this run's
    # result. The record is written through write_record(), which is atomic and fails closed.
    def refused(reason):
        """Canonical refusal: exactly ONE line and exit 2.

        The record is written first. If the write fails, the old record may remain, so the result
        is unavailable and the single line says so - the run never prints a refusal and then a
        different verdict, nor the reverse.
        """
        if args.json:
            try:
                write_record(args.json, {"verdict": "REFUSED", "reason": str(reason),
                                         "class": args.cls})
            except Refused as werr:
                print(f"N3GATE REFUSED {werr} (the requested record was not written)")
                return 2
        print(f"N3GATE REFUSED {reason}")
        return 2

    def unproven(reason):
        # same order as refused(): the record first, then the one terminal line
        if args.json:
            try:
                write_record(args.json, {"verdict": "UNPROVEN", "reason": reason,
                                         "class": args.cls})
            except Refused as werr:
                print(f"N3GATE REFUSED {werr} (the requested record was not written)")
                return 2
        print(f"N3GATE UNPROVEN {reason}")
        return 3

    rows_path = os.path.join(args.run_dir, "rows.jsonl")
    bin_path = os.path.join(args.run_dir, "logits.f32le")
    for p in (rows_path, bin_path):
        if not os.path.exists(p):
            return unproven(f"missing {p}")
    report = os.path.join(args.run_dir, "report.json")
    if not os.path.exists(report):
        return unproven(f"missing {report}")

    try:
        rows = read_rows(rows_path)
        # physical contract first: one global stream, no overlap, no hole
        vocab, total_bytes = physical_layout(rows)
        n3 = side_rows(rows, "n3")
        n1 = side_rows(rows, "n1")
        if len(n3) != len(n1):
            raise Refused(f"row count differs between sides: n3={len(n3)} n1={len(n1)}")
        missing = sorted(set(n3) - set(n1))
        extra = sorted(set(n1) - set(n3))
        if missing or extra:
            raise Refused(f"identity mismatch: n3-only={missing[:3]} n1-only={extra[:3]}")
        if len(n3) < args.min_rows:
            raise Refused(f"only {len(n3)} aligned rows, floor is {args.min_rows}")
        # The pair is only a pair if every verify row has a reference row at the SAME identity -
        # which is what the set equality above already enforces. Restate it as a count so a
        # producer that emits no reference rows at all cannot slip through on a shared key set.
        if not n1:
            raise Refused("no reference rows: a capture without an N1 side is not comparable")
        # rows are addressed by the offset the producer wrote, in ONE global stream
        data = read_logits(bin_path, vocab, len(rows), total_bytes)
    except Refused as err:
        return refused(err)

    # the report is the producer's own account of the run: the vocabulary and the row count it
    # claims must be the ones the rows and the binary actually carry
    try:
        with open(report, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        return refused(f"report.json is unreadable ({err})")
    # Both producers write "capture": "n3-verify"; the layout field says which one. A report from
    # another tool, or one with no reference rows, is not a pair and is refused.
    if meta.get("capture") != "n3-verify":
        return refused("report.json is not an n3-verify capture")
    if meta.get("layout") not in (None, "single-model", "two-model"):
        return refused(f"report.json layout {meta.get('layout')!r} is not a known capture layout")
    if int(meta.get("vocabulary", -1)) != vocab:
        return refused(f"report vocabulary {meta.get('vocabulary')} != rows vocabulary {vocab}")
    if int(meta.get("rows", -1)) != len(rows):
        return refused(f"report rows {meta.get('rows')} != rows.jsonl {len(rows)}")

    th = THRESHOLDS[args.cls]
    total_kl = 0.0
    max_kl = 0.0
    top1_hits = 0
    bad_flips = []
    max_abs = 0.0
    differing = 0
    # The comparison reads the logits, and row_kl() refuses a non-finite value. That refusal is
    # an INPUT rejection, not a gate result: it must produce the same canonical REFUSED(2) line
    # and exit as every other refusal, instead of escaping main() as a traceback with no record.
    try:
        for key in sorted(n3):
            a = data[int(n3[key]["offset_bytes"])]
            b = data[int(n1[key]["offset_bytes"])]
            ref_top, ref_margin = argmax_margin(b)   # the N1 side is the reference
            cand_top, _ = argmax_margin(a)
            kl = row_kl(b, a)
            total_kl += kl
            max_kl = max(max_kl, kl)
            if cand_top == ref_top:
                top1_hits += 1
            elif ref_margin > th["flip_margin"]:
                bad_flips.append((key, ref_top, cand_top, ref_margin))
            for x, y in zip(a, b):
                if x != y:
                    differing += 1
                max_abs = max(max_abs, abs(x - y))
    except Refused as err:
        if args.json:
            try:
                write_record(args.json, {"verdict": "REFUSED", "reason": str(err),
                                         "class": args.cls, "rows": len(n3), "vocabulary": vocab})
            except Refused as werr:
                print(f"N3GATE REFUSED {werr} (the requested record was not written)")
                return 2
        print(f"N3GATE REFUSED {err}")
        return 2
    mean_kl = total_kl / len(n3)
    top1 = top1_hits / len(n3)

    ok = True
    if args.cls == "exact":
        ok = differing == 0
    else:
        ok &= mean_kl < th["mean_kl"]
        if args.cls.startswith("behaviour-changing"):
            ok &= mean_kl >= THRESHOLDS["behaviour-changing-min"]["mean_kl"]
        ok &= max_kl < th["max_kl"]
        ok &= top1 >= th["top1"]
        ok &= not bad_flips

    record = {
        "verdict": "PASS" if ok else "FAIL",
        "class": args.cls,
        "rows": len(n3),
        "vocabulary": vocab,
        "differing_values": differing,
        "max_abs": max_abs,
        "mean_kl": mean_kl,
        "max_row_kl": max_kl,
        "top1": top1,
        "flips_outside_margin": len(bad_flips),
        "thresholds": th,
    }
    # The record is written BEFORE anything is printed: if it cannot be written the run has no
    # usable result, and the only line this process emits is a single REFUSED with exit 2. No
    # PASS/FAIL line is ever printed and then contradicted.
    if args.json:
        try:
            write_record(args.json, record)
        except Refused as werr:
            print(f"N3GATE REFUSED {werr} (the requested record was not written)")
            return 2
    rc = finish(record, [f"class={args.cls}", f"rows={len(n3)}", f"vocab={vocab}",
                         f"differing={differing}", f"max_abs={max_abs:.9g}",
                         f"mean_kl={mean_kl:.9g}", f"max_row_kl={max_kl:.9g}",
                         f"top1={top1:.6f}", f"flips_outside_margin={len(bad_flips)}"])
    if bad_flips:
        for key, rt, ct, margin in bad_flips[:5]:
            print(f"  flip {key} ref_top1={rt} cand_top1={ct} ref_margin={margin:.6g}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
