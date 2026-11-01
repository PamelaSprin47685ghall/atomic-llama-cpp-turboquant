#!/usr/bin/env python3
"""Synthetic negative fixtures for the N3 gate. CPU only, no model, no GPU.

Each case builds a tiny well-formed capture in a temp directory, breaks exactly one property,
and requires the gate to REFUSE it (exit 2). The positive control must PASS with --class exact.
A case that does not refuse is a gate defect, so the script exits non-zero.

Nothing here is model output: a PASS on the control means only that the gate accepts two
identical synthetic captures.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "n3_logits_gate.py")
VOCAB = 8


def write_capture(directory, rows, blobs, report=None):
    """Write rows.jsonl + logits.f32le in the PRODUCER's layout.

    Both producers write each side as its own contiguous region starting at offset 0 of that
    side, so `blobs` is the concatenation of the side regions in the order the rows list them,
    and every row's `offset_bytes` indexes its own side's region. The gate checks exactly that
    and addresses rows by offset, so the fixture must be built the same way.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(directory, "logits.f32le"), "wb") as f:
        f.write(b"".join(blobs))
    if report is None:
        report = {"schema_version": 1, "capture": "n3-verify", "vocabulary": VOCAB,
                  "rows": len(rows), "bytes": len(b"".join(blobs))}
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f)


def row(side, window, seq, step, i, pos, tok, offset):
    return {
        "side": side, "window": window, "seq": seq, "step": step, "row": i,
        "input_position": pos, "input_token": tok, "vocabulary": VOCAB,
        "offset_bytes": offset, "count": VOCAB, "dtype": "f32le",
    }


def vec(seed):
    # deterministic, finite, distinct per row
    return struct.pack(f"<{VOCAB}f", *[(seed * 0.5) - 2.0 + 0.1 * i for i in range(VOCAB)])


class stream:
    """Emit records with ONE global offset counter, in the producers' emission order.

    Both producers (n3_verify_capture.cpp, n3_verify_two_model.cpp) hold a single `n_bytes`
    that every record advances by `vocabulary * 4`, and they fwrite in production order. The
    file is therefore one interleaved stream: rows.jsonl order == file order, offsets are
    0, V*4, 2*V*4, ... over ALL records (prefill included).
    """

    def __init__(self):
        self.rows = []
        self.blobs = []

    def add(self, side, window, step, i, pos, tok, payload):
        off = len(self.blobs) * VOCAB * 4
        self.rows.append(row(side, window, 0, step, i, pos, tok, off))
        self.blobs.append(payload)

    def prefill(self, window, pos, tok, payload):
        off = len(self.blobs) * VOCAB * 4
        self.rows.append(row("prefill", window, 0, 0, 0, pos, tok, off))
        self.blobs.append(payload)


def base_capture():
    """The real emission order: per window the prefill row, then per step n1 rows, then n3 rows.

    Prefill is a physical record and an identity record; it is never paired (pairing selects
    side n3/n1), so it does not enter any numeric comparison.
    """
    st = stream()
    for window in (0, 1):
        st.prefill(window, 2, 90 + window, vec(90 + window))
        for step in range(2):
            for side in ("n1", "n3"):
                for i in range(3):
                    st.add(side, window, step, i, 5 + step * 3 + i, 100 + step * 3 + i,
                           vec(step * 3 + i + 10 * window))
    return st.rows, st.blobs


def run_gate(directory, cls="exact", min_rows=1, json_path=None):
    argv = [sys.executable, GATE, directory, "--class", cls, "--min-rows", str(min_rows)]
    if json_path:
        argv += ["--json", json_path]
    return subprocess.run(argv, capture_output=True, text=True)


def main():
    failures = 0

    def case(name, mutate, want_rc, vocab=None, report=None, expect=None):
        """`expect` is a substring the canonical N3GATE line must contain.

        A negative that refuses for the WRONG reason is not a test of the intended branch, so
        every case names the reason it must produce (global tiling, duplicated key, vocabulary,
        report, dtype, EOF, reference set, ...).
        """
        nonlocal failures
        rows, blobs = base_capture()
        rows, blobs = mutate(rows, blobs)
        if vocab is not None:
            # a capture whose declared vocabulary disagrees with the binary size
            rows[0]["vocabulary"] = vocab
            rows[0]["count"] = vocab
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "cap")
            write_capture(d, rows, blobs, report)
            # --json goes to a file that already holds a stale PASS: a refusal must replace it,
            # not leave it behind for a later consumer to read as this run's verdict.
            verdict_path = os.path.join(tmp, "verdict.json")
            with open(verdict_path, "w", encoding="utf-8") as f:
                json.dump({"verdict": "PASS", "stale": True}, f)
            out = run_gate(d, "exact", 1, verdict_path)
            got = out.returncode
            verdict = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "(no output)"
            stale = False
            if want_rc != 0:
                try:
                    with open(verdict_path, "r", encoding="utf-8") as f:
                        stale = json.load(f).get("stale") is True
                except (OSError, json.JSONDecodeError):
                    stale = True   # unreadable is also not a stale PASS
            # exactly one terminal line, and it must be the first: a consumer that reads the
            # first N3GATE line must never see PASS followed by a REFUSED
            terminals = [l for l in out.stdout.splitlines() if l.startswith("N3GATE ")]
            one_terminal = len(terminals) == 1
            first_is_terminal = bool(terminals) and out.stdout.splitlines()[0].startswith("N3GATE ")
            reason_ok = expect is None or (verdict and expect in verdict)
            ok = got == want_rc and not stale and one_terminal and first_is_terminal and reason_ok
            print(f"  {'PASS' if ok else 'FAIL'} {name}: rc={got} want={want_rc}"
                  f"{' STALE-PASS-LEFT' if stale else ''}"
                  f"{'' if one_terminal else f' TERMINAL-LINES={len(terminals)}'}"
                  f"{'' if reason_ok else f' WRONG-REASON want~{expect!r}'} :: {verdict}")
            if not ok:
                failures += 1

    print("N3GATE-FIXTURES")
    case("control-identical-pair", lambda r, b: (r, b), 0)

    def duplicate_key(rows, blobs):
        # Append a copy of the LAST n3 row at the next legal GLOBAL offset, so the global tiling
        # passes and the duplicated identity key is the only test that can fail. An n3 row on
        # purpose: a duplicated prefill row would never reach the key check, because prefill is
        # not a paired side.
        last = max(i for i, r in enumerate(rows) if r["side"] == "n3")
        dup = dict(rows[last])
        dup["offset_bytes"] = len(blobs) * VOCAB * 4
        return rows + [dup], blobs + [blobs[last]]

    def retile(rows, blobs):
        """Re-number offsets over the surviving records so the physical layout stays legal."""
        for i, r in enumerate(rows):
            r["offset_bytes"] = i * VOCAB * 4
        return rows, blobs

    def missing_n1(rows, blobs):
        # The predicate matches one n1 record PER WINDOW (step 1, row 2 in every window), so with
        # the two-window base capture it removes TWO n1 records and their payloads. Offsets are
        # then re-tiled, so the physical layout stays legal.
        # The gate sees n1 rows fewer than n3 rows: its FIRST paired check is the row count, so
        # that is the branch this case must name. (Identity mismatch between sides of EQUAL count
        # is covered by position-mismatch.)
        keep = [i for i, r in enumerate(rows)
                if not (r["side"] == "n1" and r["step"] == 1 and r["row"] == 2)]
        return retile([rows[i] for i in keep], [blobs[i] for i in keep])

    def position_mismatch(rows, blobs):
        for r in rows:
            if r["side"] == "n1" and r["step"] == 0 and r["row"] == 1:
                r["input_position"] += 1
        return rows, blobs

    def wrong_count(rows, blobs):
        rows[0]["count"] = VOCAB + 1
        return rows, blobs

    def wrong_dtype(rows, blobs):
        rows[0]["dtype"] = "f16"
        return rows, blobs

    def overlap(rows, blobs):
        # two records claim the same bytes: the second repeats the first one's offset
        rows[1]["offset_bytes"] = rows[0]["offset_bytes"]
        return rows, blobs

    def hole(rows, blobs):
        # a record leaves a gap: every later offset moves one row further. blobs stays the same
        # length, so the file is then SHORTER than the furthest offset addresses - the refusal
        # names the gap at the record where it appears.
        for r in rows[2:]:
            r["offset_bytes"] += VOCAB * 4
        return rows, blobs

    def beyond_eof(rows, blobs):
        # Keep a perfectly legal contiguous offset run (0, V*4, ...) and make the FILE one record
        # shorter: the last record then addresses bytes that were never written, and only the
        # size check can refuse it. That is the branch this case exists for.
        return rows, blobs[:-1]

    def crossed_values(rows, blobs):
        """PHYSICALLY VALID, SEMANTICALLY WRONG - and built so the difference is unmistakable.

        The base capture gives the n3 and n1 record of one identity the SAME payload, so merely
        exchanging them would be a no-op (and prove nothing). Here the two payloads are made
        DIFFERENT first: the n3 record gets a sharply peaked row, its n1 twin a flat one. Every
        offset stays legal and every key stays unique, so only a gate that reads each side from
        its own offset and pairs by identity can see that the two sides disagree - it must FAIL
        the numeric class. A gate that read rows by list position, or that swapped the sides,
        would still see two identical rows and PASS, which is exactly what this case forbids.
        """
        i_n3 = next(i for i, r in enumerate(rows) if r["side"] == "n3")
        k = (rows[i_n3]["window"], rows[i_n3]["step"], rows[i_n3]["row"],
             rows[i_n3]["input_position"], rows[i_n3]["input_token"])
        i_n1 = next(i for i, r in enumerate(rows) if r["side"] == "n1"
                    and (r["window"], r["step"], r["row"], r["input_position"], r["input_token"]) == k)
        new = list(blobs)
        # a peaked candidate (one logit far above the rest) against a flat reference: the KL is
        # large in the direction row_kl(ref || cand) measures, and the top-1 differs
        new[i_n3] = struct.pack(f"<{VOCAB}f", *([8.0] + [0.0] * (VOCAB - 1)))
        new[i_n1] = struct.pack(f"<{VOCAB}f", *([0.0] * VOCAB))
        return rows, new

    # (a "truncated-binary" case that drops the last row AND its bytes is the same branch as
    # report-rows-disagree below - the report is regenerated from the surviving rows - so it is
    # not kept: report-rows-disagree already covers "report.rows disagrees with a physically
    # valid, count-balanced capture", and beyond_eof covers "file shorter than the rows".)

    case("duplicate-key", duplicate_key, 2, expect="duplicated key")
    case("missing-n1-row", missing_n1, 2, expect="row count differs between sides")
    case("position-mismatch", position_mismatch, 2, expect="identity mismatch")
    case("count-not-vocabulary", wrong_count, 2, expect="is not the vocabulary")
    case("vocabulary-not-the-binary", lambda r, b: (r, b), 2, vocab=VOCAB * 2,
         expect="vocabulary changes from")
    # a physically valid capture with equal n3/n1 counts whose report claims a different row
    # count: only the report check can refuse it
    case("report-rows-disagree", lambda r, b: (r, b), 2,
         report={"capture": "n3-verify", "vocabulary": VOCAB,
                 "rows": len(base_capture()[0]) + 1},
         expect="report rows")
    case("report-not-this-capture", lambda r, b: (r, b), 2,
         report={"capture": "tp5-compose", "vocabulary": VOCAB, "rows": 12},
         expect="not an n3-verify capture")
    case("dtype-not-f32le", wrong_dtype, 2, expect="not f32le")
    case("offset-overlap", overlap, 2, expect="overlaps or precedes")
    case("offset-hole", hole, 2, expect="leaves a hole before")
    case("offset-beyond-eof", beyond_eof, 2, expect="SHORTER than the last row")

    # physically valid, semantically wrong: the two sides of one identity carry sharply different
    # rows. Must FAIL the numeric class (rc 1), not refuse and not pass.
    rows, blobs = crossed_values(*base_capture())
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "cap")
        write_capture(d, rows, blobs)
        out = run_gate(d, "numeric-order-only", 1)
        lines = [l for l in out.stdout.splitlines() if l.startswith("N3GATE ")]
        ok = out.returncode == 1 and bool(lines) and lines[-1].startswith("N3GATE FAIL")
        print(f"  {'PASS' if ok else 'FAIL'} crossed-values-must-fail: rc={out.returncode} "
              f"want=1 :: {lines[-1] if lines else '(no N3GATE line)'}")
        if not ok:
            failures += 1

    # A small perturbation of the candidate must stay inside the numeric-order-only class
    # (mean KL < 1e-6, max row KL < 1e-5, top-1 unchanged), a large one must leave it. The
    # perturbations are chosen against the class's own numbers:
    #   small: +1e-6 on ONE non-top element of a row whose reference softmax weight is ~4.5e-3.
    #          To first order KL(ref||cand) = p_ref(e) * (z_ref - z_cand) ~ 4.5e-9, so the row KL
    #          is ~3.6e-14 (measured: 3.57e-14). That is eleven orders below the 1e-5 max-row
    #          limit, which is the point: it is unambiguously inside the class, not near it.
    #   large: +5.0 -> the top-1 flips with a margin far above the 0.10 flip allowance.
    def flip(rows, blobs, lift, index=None):
        new = list(blobs)
        # one global stream: pick the first n3 record by its index in that stream
        if index is None:
            index = next(i for i, r in enumerate(rows) if r["side"] == "n3")
        vals = list(struct.unpack(f"<{VOCAB}f", blobs[index]))
        vals[0] += lift
        new[index] = struct.pack(f"<{VOCAB}f", *vals)
        return rows, new

    def expect(name, lift, want):
        nonlocal failures
        rows, blobs = flip(*base_capture(), lift)
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "cap")
            write_capture(d, rows, blobs)
            out = run_gate(d, "numeric-order-only", 1)
            lines = [l for l in out.stdout.strip().splitlines() if l.startswith("N3GATE ")]
            ok = out.returncode == want and bool(lines)
            print(f"  {'PASS' if ok else 'FAIL'} {name}: rc={out.returncode} want={want} :: "
                  f"{lines[-1] if lines else '(no N3GATE line)'}")
            if not ok:
                failures += 1

    expect("perturbation-within-numeric-order", 1e-6, 0)
    expect("perturbation-outside-numeric-order", 5.0, 1)

    # A non-finite logit is an INPUT refusal (rc 2, canonical N3GATE REFUSED line), never a
    # traceback: row_kl() is called outside the argument checks, so this is the regression case
    # for that handler.
    def nonfinite(rows, blobs):
        # the NaN must be in a PAIRED row: a prefill record never reaches row_kl()
        i_n3 = next(i for i, r in enumerate(rows) if r["side"] == "n3")
        new = list(blobs)
        vals = list(struct.unpack(f"<{VOCAB}f", blobs[i_n3]))
        vals[0] = float("nan")
        new[i_n3] = struct.pack(f"<{VOCAB}f", *vals)
        return rows, new

    rows, blobs = nonfinite(*base_capture())
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "cap")
        write_capture(d, rows, blobs)
        # same stale-PASS trap: a NaN refusal must replace the record, and must not traceback
        verdict_path = os.path.join(tmp, "verdict.json")
        with open(verdict_path, "w", encoding="utf-8") as f:
            json.dump({"verdict": "PASS", "stale": True}, f)
        out = run_gate(d, "numeric-order-only", 1, verdict_path)
        lines = [l for l in out.stdout.strip().splitlines() if l.startswith("N3GATE ")]
        stale = False
        try:
            with open(verdict_path, "r", encoding="utf-8") as f:
                stale = json.load(f).get("stale") is True
        except (OSError, json.JSONDecodeError):
            stale = True
        ok = (out.returncode == 2 and bool(lines) and lines[-1].startswith("N3GATE REFUSED")
              and not stale and "Traceback" not in out.stderr)
        print(f"  {'PASS' if ok else 'FAIL'} nonfinite-logit-is-a-refusal: rc={out.returncode} "
              f"want=2{' STALE-PASS-LEFT' if stale else ''} :: "
              f"{lines[-1] if lines else '(no N3GATE line)'}")
        if not ok:
            failures += 1

    # ---- write failure: the record cannot be written, so the run has no usable result -------
    # The --json path is a DIRECTORY, so os.replace() cannot put a file there. The gate must then
    # print exactly one REFUSED line and exit 2 - and in particular must NOT print a PASS/FAIL
    # line first and a REFUSED afterwards (that is the defect this fixture exists for).
    rows, blobs = base_capture()
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "cap")
        write_capture(d, rows, blobs)
        unwritable = os.path.join(tmp, "verdict.json")
        os.mkdir(unwritable)          # a directory cannot be replaced by a file
        out = run_gate(d, "exact", 1, unwritable)
        lines = out.stdout.splitlines()
        terminals = [l for l in lines if l.startswith("N3GATE ")]
        ok = (out.returncode == 2
              and len(terminals) == 1
              and terminals[0].startswith("N3GATE REFUSED")
              and not any(l.startswith("N3GATE PASS") or l.startswith("N3GATE FAIL") for l in lines))
        print(f"  {'PASS' if ok else 'FAIL'} unwritable-record-is-one-refusal: rc={out.returncode} "
              f"want=2 terminals={len(terminals)} :: {terminals[0] if terminals else '(none)'}")
        if not ok:
            failures += 1

    # same for a REFUSAL path with an unwritable record: one line, exit 2
    rows, blobs = base_capture()
    rows = [x for x in rows if not (x["side"] == "n1" and x["step"] == 1 and x["row"] == 2)]
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "cap")
        write_capture(d, rows, blobs)
        unwritable = os.path.join(tmp, "verdict.json")
        os.mkdir(unwritable)
        out = run_gate(d, "exact", 1, unwritable)
        terminals = [l for l in out.stdout.splitlines() if l.startswith("N3GATE ")]
        ok = (out.returncode == 2 and len(terminals) == 1
              and terminals[0].startswith("N3GATE REFUSED"))
        print(f"  {'PASS' if ok else 'FAIL'} unwritable-record-on-a-refusal: rc={out.returncode} "
              f"want=2 terminals={len(terminals)} :: {terminals[0] if terminals else '(none)'}")
        if not ok:
            failures += 1

    print("N3GATE-FIXTURES", "OK" if failures == 0 else f"FAILED {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
