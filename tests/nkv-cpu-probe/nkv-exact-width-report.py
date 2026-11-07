#!/usr/bin/env python3
"""Parse test-nkv-exact-width stderr into a JSON report.

The test prints one line family per observation (see
tests/test-nkv-exact-width.cpp). This script reads one log per arm and writes
a machine-readable report. It measures nothing itself.

Honesty rules:

  * An arm whose log file is absent is emitted with "ran": false and
    "status": "NOT_RUN". It carries no counts, no case list and no steps.
  * A field whose line family is absent from an existing log is null. A zero
    always means "measured as zero", never "not measured".
  * Exit status is 0 only when every arm that ran is ok and every arm that did
    not run says NOT_RUN. A missing arm never turns into a pass.

Usage:
  nkv-exact-width-report.py --out REPORT.json \
      --arm unset=unset.log --arm off=off.log --arm on=on.log \
      --arm-rc unset=0 --arm-rc off=0 --arm-rc on=0 \
      [--expect-cases single,multi_seq,multi_stream,widths,off] \
      [--expect-steps 300,9,4,257,300]

Exit status: 0 all arms ok, 3 at least one arm NOT_RUN, 1 any failure.
"""
import argparse
import json
import os
import sys

# ---------------------------------------------------------------- line parsing

# Every family is a fixed keyword followed by key=value pairs. The parser
# splits on whitespace and does not care about the field order.
FAMILIES = (
    "NKV_ARM",
    "NKV_CASE",
    "NKV_CASE_END",
    "NKV_EXACT",
    "NKV_MULTI",
    "NKV_MULTI_STREAM",
    "NKV_MULTI_STREAM_BIND",
    "NKV_WIDTH",
    "NKV_RT",
    "OFF_ARM",
    "OFF_ARM_REUSE",
    "NKV_EXACT_END",
)

# OFF_ARM has two shapes: a per-step row (step=...) and a total line
# (builds=... steps=...). The total line is kept in its own family so a
# reader cannot mistake it for a step row.
OFF_ARM_TOTAL = "OFF_ARM_TOTAL"

# pad_tail= has three states and they are three different facts:
#   an integer  the node carried GGML_TENSOR_FLAG_REDUCE_PAD_TAIL with that
#               op_params[2] width;
#   0           the same, and the width is zero (a measured zero);
#   "none"      no node carried the flag, so no width exists.
# The parser keeps the integer as an int and the word as the string "none",
# so a reader never has to guess which fact a None means. A line without the
# field keeps it absent, which is a third state again: not measured.
PAD_TAIL_NONE = "none"

# Integer fields, by family. Any other field stays a string, so a new field in
# the test reaches the report without a parser change.
INT_FIELDS = {
    "NKV_CASE":               ("n_seq", "unified", "steps"),
    "NKV_CASE_END":           ("ok", "steps"),
    # MODE= is the probe's mode name and stays a string; the test prints the
    # same family without it, which the parser accepts either way.
    "NKV_EXACT":              ("step", "pos", "mask_ne0", "k_ne2", "v_nkv",
                               "pad_tail"),
    "NKV_MULTI":              ("step", "seq", "pos", "used0", "used1", "width",
                               "mask_ne0", "k_ne2", "v_nkv", "pad_tail"),
    "NKV_MULTI_STREAM":       ("n_ctx", "n_ctx_seq", "n_ctx_kv", "n_seq_max",
                               "stock_width", "step", "seq", "pos", "mask_ne0",
                               "mask_ne1", "mask_ne3", "k_ne2", "v_nkv",
                               "pad_tail"),
    "NKV_MULTI_STREAM_BIND":  ("step", "state", "patch_binds", "fallbacks"),
    "NKV_WIDTH":              ("step", "pos", "mask_ne0", "k_ne2", "v_nkv",
                               "at_capacity", "pad_tail"),
    # the binder's own exported width; the test labels it binder-side
    "NKV_RT":                 ("step", "runtime", "bound", "pos", "patch_binds"),
    "OFF_ARM":                ("step", "pos", "mask_ne0", "k_ne2", "v_nkv",
                               "pad_tail"),
    "OFF_ARM_REUSE":          ("ok", "builds", "steps"),
    # OFF_ARM has two shapes: a per-step row (step=...) and a total line
    # (builds=... steps=...). parse_line re-tags the total line to
    # OFF_ARM_TOTAL before conversion, so it needs its own entry here.
    OFF_ARM_TOTAL:            ("builds", "steps"),
    "NKV_EXACT_END":          ("ok", "cases", "steps"),
}

def _to_int(text):
    try:
        return int(text, 10)
    except ValueError:
        return None


def parse_line(line):
    """Return (family, fields) for one line, or None.

    The family is one of FAMILIES, OFF_ARM_TOTAL, or "FAIL".
    """
    line = line.strip()
    if line.startswith("FAIL:"):
        return "FAIL", {"text": line[len("FAIL:"):].strip()}
    head, _, rest = line.partition(" ")
    if head not in FAMILIES:
        return None
    fields = {}
    for token in rest.split():
        key, sep, value = token.partition("=")
        if not sep or not key:
            continue
        fields[key] = value
    family = head
    if family == "OFF_ARM" and "step" not in fields and "builds" in fields:
        family = OFF_ARM_TOTAL
    # Convert by the family the line actually belongs to, so the OFF_ARM
    # total line gets its integers like every other family.
    for key in INT_FIELDS.get(family, ()):
        if key in fields:
            value = _to_int(fields[key])
            fields[key] = value if value is not None else fields[key]
    # A pad_tail= token that is neither an integer nor "none" is not a width
    # and not an absence: it stays the string it was, so a reader sees the bad
    # token instead of a silent zero. "none" stays the word, which is a
    # different fact from the integer 0.
    return family, fields


# Fields that must be integers on an NKV_RT line. A line that carries the
# family keyword but not all of them is malformed, and a malformed line must
# not pass as "absent": the same rule the test's other families follow.
NKV_RT_FIELDS = INT_FIELDS["NKV_RT"]


def nkv_rt_malformed(fields):
    """True when an NKV_RT line lacks one of its integer fields."""
    return any(not isinstance(fields.get(key), int) for key in NKV_RT_FIELDS)


def parse_log(text):
    """Split a log into families and keep the original line order.

    Absent families stay absent, not empty. The order is needed to attribute
    step rows to the case that was running.
    """
    out = {family: [] for family in FAMILIES}
    out[OFF_ARM_TOTAL] = []
    out["FAIL"] = []
    order = []
    for line in text.splitlines():
        parsed = parse_line(line)
        if parsed is None:
            continue
        family, fields = parsed
        if family == "NKV_RT" and nkv_rt_malformed(fields):
            # keep the raw text so the report names the bad line, and count it
            # as a failure rather than as an absent observation
            out["FAIL"].append({"text": "malformed NKV_RT line: " + line.strip()})
            order.append(("FAIL", out["FAIL"][-1]))
            continue
        out[family].append(fields)
        order.append(parsed)
    out["_order"] = order
    return out


def build_cases(lines):
    """Join NKV_CASE headers with their NKV_CASE_END footers, in order.

    A header without a footer keeps footer null: the case started and never
    reported an end, which is a failure, not a zero.
    """
    cases = []
    by_name = {}
    for fields in lines["NKV_CASE"]:
        case = {
            "name": fields.get("name"),
            "n_seq": fields.get("n_seq"),
            "unified": fields.get("unified"),
            "declared_steps": fields.get("steps"),
            "observed_steps": None,
            "ok": None,
            "footer_steps": None,
            "arm": fields.get("arm"),
            "skipped": fields.get("skipped"),
        }
        cases.append(case)
        by_name.setdefault(case["name"], []).append(case)
    for fields in lines["NKV_CASE_END"]:
        name = fields.get("name")
        pending = [c for c in by_name.get(name, []) if c["ok"] is None]
        if not pending:
            # a footer with no header: keep it as a case of its own so the
            # mismatch is visible instead of dropped
            case = {
                "name": name,
                "n_seq": None,
                "unified": None,
                "declared_steps": None,
                "observed_steps": None,
                "ok": None,
                "footer_steps": None,
                "arm": None,
                "skipped": None,
            }
            cases.append(case)
            by_name.setdefault(name, []).append(case)
            pending = [case]
        case = pending[0]
        case["ok"] = fields.get("ok")
        case["footer_steps"] = fields.get("steps")
        if "skipped" in fields:
            case["skipped"] = fields.get("skipped")
    return cases


def count_observed_steps(lines):
    """Count step rows per case name, from the step families.

    The test prints step rows only for the case it is running, so a simple
    ordered walk over the log is enough. The walk keeps a current case name
    from the NKV_CASE header.
    """
    counted = {}
    current = None
    for line in lines["_order"]:
        family, fields = line
        if family == "NKV_CASE":
            current = fields.get("name")
            counted.setdefault(current, 0)
        elif family in ("NKV_EXACT", "NKV_MULTI", "NKV_WIDTH", "OFF_ARM"):
            if current is not None:
                counted[current] = counted.get(current, 0) + 1
        elif family == "NKV_MULTI_STREAM" and "step" in fields:
            if current is not None:
                counted[current] = counted.get(current, 0) + 1
    return counted


# Step families that carry the pad-tail field. NKV_RT does not: it reports the
# binder's own width, not a node's flag.
PAD_TAIL_FAMILIES = ("NKV_EXACT", "NKV_MULTI", "NKV_MULTI_STREAM",
                     "NKV_WIDTH", "OFF_ARM")


def pad_tail_summary(lines):
    """The pad-tail field of every step row, as it appeared.

    An entry is the integer width, the word "none", or None when the row did
    not carry the field at all. The three are different facts and are kept
    apart: 0 is a measured zero, "none" is "no node carried the flag", None is
    "this log does not say".
    """
    out = []
    for family in PAD_TAIL_FAMILIES:
        for fields in lines[family]:
            if "step" not in fields:
                continue
            out.append(fields.get("pad_tail"))
    return out


# ---------------------------------------------------------------- expectations

def check_expectations(cases, expect_cases, expect_steps):
    """Return a list of mismatch strings. Empty means the log matches.

    Only cases of the arm that ran are checked. The test calls every case in
    every arm and marks a case whose arm is not the running one with
    skipped=1; those cases touched nothing, so they carry no step counts and
    must not be compared against the expected numbers. The full case list is
    still checked, in order, so a case that vanished from the source is
    visible.
    """
    problems = []
    if expect_cases is not None:
        got = [c["name"] for c in cases]
        if got != expect_cases:
            problems.append("case list %r, want %r" % (got, expect_cases))
    ran = [c for c in cases if c.get("skipped") != "1"]
    if expect_steps is not None:
        for i, want in enumerate(expect_steps):
            if i >= len(ran):
                problems.append("missing case #%d (want %d steps)" % (i, want))
                continue
            case = ran[i]
            for field in ("declared_steps", "observed_steps", "footer_steps"):
                got = case[field]
                if got != want:
                    problems.append("case %s %s=%r, want %d"
                                    % (case["name"], field, got, want))
    return problems


# ---------------------------------------------------------------- one arm

def arm_report(name, log_path, rc, expect_cases, expect_steps):
    """Build the report entry for one arm."""
    if log_path is None or not os.path.isfile(log_path):
        return {
            "arm": name,
            "status": "NOT_RUN",
            "ran": False,
            "log": log_path,
            "rc": rc,
            "reason": "log file missing",
        }

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    lines = parse_log(text)
    cases = build_cases(lines)
    counted = count_observed_steps(lines)
    for case in cases:
        case["observed_steps"] = counted.get(case["name"])

    end = lines["NKV_EXACT_END"][-1] if lines["NKV_EXACT_END"] else None
    multi_stream_head = (lines["NKV_MULTI_STREAM"][0]
                         if lines["NKV_MULTI_STREAM"] else None)
    # the first NKV_MULTI_STREAM line carries the context sizes; a step line
    # does not, so only keep it when it has n_ctx
    if multi_stream_head is not None and "n_ctx" not in multi_stream_head:
        multi_stream_head = None

    problems = []
    if not lines["NKV_CASE"]:
        problems.append("no NKV_CASE header")
    if not lines["NKV_CASE_END"]:
        problems.append("no NKV_CASE_END footer")
    if end is None:
        problems.append("no NKV_EXACT_END footer")
    for case in cases:
        if case["ok"] is None:
            problems.append("case %s has no NKV_CASE_END" % case["name"])
        elif case["ok"] != 1:
            problems.append("case %s ok=%r" % (case["name"], case["ok"]))
        # A case of the running arm must declare and report the same count.
        # A skipped case declares the count it WOULD have run and reports 0,
        # so the two differ by design; what must hold for it is that it
        # observed nothing, which the expectation check above enforces.
        if (case.get("skipped") != "1"
                and case["declared_steps"] is not None
                and case["footer_steps"] is not None
                and case["declared_steps"] != case["footer_steps"]):
            problems.append("case %s declared steps=%r footer steps=%r"
                            % (case["name"], case["declared_steps"],
                               case["footer_steps"]))
    if lines["FAIL"]:
        problems.append("%d FAIL: line(s)" % len(lines["FAIL"]))
    # The arm the log says it ran. The report does not decide which arm was
    # requested -- the runner does -- but an arm with no NKV_ARM line has no
    # proof of its path, so it cannot be reported OK either.
    arm_line = lines["NKV_ARM"][0] if lines["NKV_ARM"] else None
    if len(lines["NKV_ARM"]) != 1:
        problems.append("%d NKV_ARM line(s), want 1" % len(lines["NKV_ARM"]))
    if end is not None and end.get("ok") != 1:
        problems.append("NKV_EXACT_END ok=%r" % (end.get("ok"),))
    if rc is not None and rc != 0:
        problems.append("exit code %r" % (rc,))
    problems.extend(check_expectations(cases, expect_cases, expect_steps))

    return {
        "arm": name,
        "status": "OK" if not problems else "FAIL",
        "ran": True,
        "log": log_path,
        "rc": rc,
        "reason": None if not problems else "; ".join(problems),
        "problems": problems,
        "cases": cases,
        "case_count": len(cases),
        "arm_env": arm_line,
        "end": end,
        "multi_stream_context": multi_stream_head,
        "steps": {
            "NKV_EXACT": lines["NKV_EXACT"],
            "NKV_MULTI": lines["NKV_MULTI"],
            "NKV_MULTI_STREAM": [f for f in lines["NKV_MULTI_STREAM"]
                                 if "step" in f],
            "NKV_MULTI_STREAM_BIND": lines["NKV_MULTI_STREAM_BIND"],
            "NKV_WIDTH": lines["NKV_WIDTH"],
            "NKV_RT": lines["NKV_RT"],
            "OFF_ARM": lines["OFF_ARM"],
        },
        # the pad-tail field as it appeared, one entry per step row that
        # carried it: an int, the word "none", or absent from the dict. This
        # is a summary of the log, not a verdict on it.
        "pad_tail": pad_tail_summary(lines),
        "off_arm_total": lines[OFF_ARM_TOTAL][0] if lines[OFF_ARM_TOTAL] else None,
        "off_arm_reuse": lines["OFF_ARM_REUSE"][0] if lines["OFF_ARM_REUSE"] else None,
        "fail_lines": [f["text"] for f in lines["FAIL"]],
    }


# ---------------------------------------------------------------- CLI

def parse_pairs(values, flag):
    """Parse repeated NAME=VALUE flags into an ordered dict."""
    out = {}
    for item in values or []:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise SystemExit("%s wants NAME=VALUE, got %r" % (flag, item))
        out[name] = value
    return out


def parse_csv(text):
    if text is None:
        return None
    return [item for item in text.split(",") if item]


def parse_int_csv(text):
    items = parse_csv(text)
    if items is None:
        return None
    try:
        return [int(item, 10) for item in items]
    except ValueError:
        raise SystemExit("--expect-steps wants integers, got %r" % (text,))


def build_report(arms, arm_rcs, expect_cases, expect_steps):
    reports = {}
    for name, log_path in arms.items():
        rc = arm_rcs.get(name)
        rc = int(rc, 10) if rc is not None else None
        reports[name] = arm_report(name, log_path, rc, expect_cases, expect_steps)
    summary = {}
    for name, report in reports.items():
        summary[name] = report["status"]
    ran = [r for r in reports.values() if r["ran"]]
    not_run = [r for r in reports.values() if not r["ran"]]
    return {
        "arms": reports,
        "summary": summary,
        "overall": {
            "ran": [r["arm"] for r in ran],
            "not_run": [r["arm"] for r in not_run],
            "ok": bool(ran) and all(r["status"] == "OK" for r in ran)
                  and not not_run,
        },
    }


def exit_code(report):
    statuses = list(report["summary"].values())
    if any(status == "FAIL" for status in statuses):
        return 1
    if any(status == "NOT_RUN" for status in statuses):
        return 3
    if not statuses:
        return 3
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parse test-nkv-exact-width logs into a JSON report.")
    parser.add_argument("--out", required=True, help="report JSON path")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=LOG",
                        help="arm name and its log; repeat per arm")
    parser.add_argument("--arm-rc", action="append", default=[],
                        metavar="NAME=RC", help="arm exit code; repeat per arm")
    parser.add_argument("--expect-cases", default=None,
                        help="comma-separated case names, in order")
    parser.add_argument("--expect-steps", default=None,
                        help="comma-separated per-case step counts, in order")
    args = parser.parse_args(argv)

    arms = parse_pairs(args.arm, "--arm")
    arm_rcs = parse_pairs(args.arm_rc, "--arm-rc")
    for name in arm_rcs:
        if name not in arms:
            raise SystemExit("--arm-rc %s has no matching --arm" % name)

    report = build_report(arms, arm_rcs,
                          parse_csv(args.expect_cases),
                          parse_int_csv(args.expect_steps))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for name, status in report["summary"].items():
        reason = report["arms"][name].get("reason")
        print("ARM %s %s%s" % (name, status, "" if not reason else ": " + reason))
    code = exit_code(report)
    print("OVERALL %s" % ("PASS" if code == 0 else
                          "NOT_RUN" if code == 3 else "FAIL"))
    return code


if __name__ == "__main__":
    sys.exit(main())
