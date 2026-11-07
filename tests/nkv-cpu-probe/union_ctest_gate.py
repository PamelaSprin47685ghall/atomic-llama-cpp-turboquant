#!/usr/bin/env python3
"""Aggregate the union/onegraph CTest suite into one JSON gate report.

Usage: union_ctest_gate.py CONFIG.json

CONFIG.json is written by run_union_ctest_gate_13022.sh and names the files
this summarizer reads:

    {
      "run_dir": "...", "build_dir": "...",
      "preflight_rc": 0, "preflight_list": "ctest-list.txt",
      "run_rc": 0, "junit": "ctest-junit.xml",
      "argv": ["..."],
      "tests": [{"name": "...", "log": "...",
                 "artifacts": [["e2e-json", "/path/to/report.json"]]}]
    }

The declared test list in the runner is the single source of truth for what
must be registered; this file only reads what the runner observed.

The report is written to RUN_DIR/union-ctest-gate.json. The exit status is 0
only when every declared test is registered, ran, passed, and left every
artifact it is supposed to leave. Anything else is a fail with a `failure`
string naming the first failing condition. A missing registration outranks
every other condition, because a test that never registered was never run.

The summarizer never runs the tests and never changes their verdict. It reads
the pre-flight listing, the JUnit XML ctest wrote for the run, and the
per-test artifacts, and it reports what it found. An artifact is evidence only
for a test ctest recorded as passed: for any other status the file may be stale
from an earlier run, so it is recorded as not checked and never read.

Exit status:
    0  every declared test is registered, ran, passed, and left the artifact it
       owns; the aggregate says verdict "pass"
    1  the gate failed; the aggregate says verdict "fail" and names the first
       failing condition
    2  the summarizer cannot read its own inputs (bad usage, unreadable or
       incomplete config); no aggregate is written
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

SCHEMA = "union-ctest-gate/1"

# Per-test artifact kinds and what each one must contain for the test's own
# gate to count as evidence. The checks are structural only: they prove the
# file was written by the tool that owns it, not that the tool passed.
KINDS = {
    # union_varlen_e2e.cpp writes this before it decides its exit code.
    "e2e-json": {"schema": "union-varlen-e2e/4"},
    # pad_lane_ab_gate.cpp writes this before it decides its exit code.
    # It runs the exact lane and the stock padded control lane
    # (GGML_TP5_UNION_BINDER_EXACT=0) and requires the two to be
    # distinguishable. Older reports are not accepted as evidence.
    "pad-gate-json": {"schema": "pad-lane-ab-gate/1"},
    # varlen_output_probe.cpp writes one little-endian header per step:
    # uint32 step, width, pos, vocab, n_outputs, then the row indices.
    "rows-bin": {"header": 5, "min_bytes": 20},
}

# ctest's JUnit status attribute, mapped onto the report's status vocabulary.
# A test ctest skipped (SKIP_RETURN_CODE) and one it could not start both land
# as "notrun"; both are not_run for this gate, which never treats a skip as a
# pass.
STATUS_OF = {"run": "passed", "fail": "failed", "notrun": "not_run"}

# Conditions that fail the aggregate, in the order they are reported. The
# first one that holds names the failure; the rest are still recorded per test
# so one run shows every problem.
ORDER = ("preflight_failed", "not_registered", "not_run", "failed",
         "missing_artifact", "bad_artifact")


def fail(reason):
    print(f"FAIL: {reason}", file=sys.stderr)
    return 2


def registered_names(path):
    """Test names from a `ctest -N` listing.

    ctest prints one `  Test #N: NAME` line per registered test. The name is
    everything after the first colon, so a test name may contain spaces.
    """
    names = []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("Test #"):
                continue
            _, _, rest = stripped.partition(":")
            name = rest.strip()
            if name:
                names.append(name)
    return names


def junit_cases(path):
    """Test name -> (status, time_seconds) from the JUnit XML ctest wrote."""
    cases = {}
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        name = case.get("name", "")
        status = STATUS_OF.get(case.get("status", ""), "unknown")
        try:
            seconds = float(case.get("time", "0"))
        except ValueError:
            seconds = 0.0
        cases[name] = (status, seconds)
    return cases


def check_artifact(kind, path):
    """Return None when the artifact is present and says the tool passed.

    The check is structural for the record files and a verdict check for the
    JSON reports: a report whose own verdict is not "pass" is not evidence
    that the driver passed, whatever its file looks like.
    """
    if not os.path.isfile(path):
        return f"missing: {path}"
    if os.path.getsize(path) == 0:
        return f"empty: {path}"
    if kind in ("e2e-json", "pad-gate-json"):
        try:
            with open(path, "r", errors="replace") as handle:
                report = json.load(handle)
        except (OSError, ValueError) as exc:
            return f"not JSON ({exc}): {path}"
        if not isinstance(report, dict):
            return f"not a JSON object: {path}"
        want = KINDS[kind]["schema"]
        if report.get("schema") != want:
            return f"schema is {report.get('schema')!r}, expected {want!r}: {path}"
        # the driver writes its report before it decides its exit code, so a
        # failing run leaves a well-formed report with verdict "fail"
        verdict = report.get("verdict")
        exit_code = report.get("exit_code")
        if verdict != "pass" or exit_code != 0:
            return (f"report verdict is {verdict!r} with exit_code {exit_code!r}, "
                    f"expected 'pass' and 0: {path}")
        return None
    if kind == "rows-bin":
        size = os.path.getsize(path)
        if size % 4 != 0 or size < KINDS[kind]["min_bytes"]:
            return f"not a probe record file ({size} bytes): {path}"
        return None
    return f"unknown artifact kind {kind!r}"


def summarize(config):
    entries = []
    for declared in config["tests"]:
        entries.append({
            "name": declared["name"],
            "registered": False,
            "status": "not_run",
            "exit_code": None,
            "wall_time_s": None,
            "log": declared.get("log"),
            # the artifacts this test must leave behind, as (kind, path) pairs
            "declared_artifacts": [tuple(pair) for pair in declared.get("artifacts", [])],
            "artifacts": [],
            "problems": [],
        })

    listing = registered_names(config["preflight_list"])
    cases = {}
    junit_path = config.get("junit")
    if junit_path and os.path.isfile(junit_path):
        try:
            cases = junit_cases(junit_path)
        except ET.ParseError as exc:
            for entry in entries:
                entry["problems"].append(f"junit unreadable ({exc})")

    for entry in entries:
        name = entry["name"]
        entry["registered"] = name in listing
        if not entry["registered"]:
            entry["problems"].append("not_registered")
        if name in cases:
            entry["status"], entry["wall_time_s"] = cases[name]
            entry["exit_code"] = 0 if entry["status"] == "passed" else 1
            if entry["status"] != "passed":
                entry["problems"].append(entry["status"])
        elif entry["registered"]:
            # registered, but ctest's run left no case for it
            entry["problems"].append("not_run")
        # An artifact is evidence only when ctest recorded the test as passed.
        # For any other status the file may be stale from an earlier run, so it
        # is recorded as not checked and never read.
        for kind, path in entry.pop("declared_artifacts"):
            if entry["status"] != "passed":
                entry["artifacts"].append(
                    {"kind": kind, "path": path, "ok": False, "checked": False})
                entry["problems"].append("not_checked")
                continue
            reason = check_artifact(kind, path)
            entry["artifacts"].append(
                {"kind": kind, "path": path, "ok": reason is None, "checked": True})
            if reason is not None:
                entry["problems"].append(
                    "missing_artifact" if reason.startswith("missing") else "bad_artifact")
                entry["problems"].append(reason)

    # The first failing condition, in ORDER, names the failure. Every test is
    # still recorded, so one run shows every problem rather than only the first.
    failure = None
    if config.get("preflight_rc", 0) != 0:
        failure = f"preflight_failed: ctest -N exited {config['preflight_rc']}"
    for condition in ORDER:
        if failure is not None:
            break
        for entry in entries:
            if condition not in entry["problems"]:
                continue
            # The detail is the reason behind THIS condition. A test that is
            # not registered also carries not_checked for its unread artifacts,
            # and that must not be appended to a not_registered failure: the
            # artifact was not checked because the test never ran, not because
            # it is bad.
            detail = ""
            if condition in ("missing_artifact", "bad_artifact"):
                detail = next((p for p in entry["problems"] if p not in ORDER), "")
            failure = f"{condition}: {entry['name']}"
            if detail:
                failure += f" ({detail})"
            break
    if failure is None and config.get("run_rc", 0) != 0:
        failure = f"ctest exited {config['run_rc']} although every declared test passed"

    return {
        "schema": SCHEMA,
        "verdict": "pass" if failure is None else "fail",
        "failure": failure,
        "argv": config.get("argv", []),
        "exit_code": 0 if failure is None else 1,
        "run_dir": config.get("run_dir"),
        "build_dir": config.get("build_dir"),
        "preflight": {
            "rc": config.get("preflight_rc"),
            "listing": config.get("preflight_list"),
            "registered": sorted(listing),
        },
        "ctest": {"rc": config.get("run_rc"), "junit": junit_path},
        "declared": [entry["name"] for entry in entries],
        "tests": entries,
        "not_run": (
            "Nothing outside the declared union/onegraph list: no GPU/HIP/CUDA "
            "backend, no relay, no pipeline parallelism, no recurrent or "
            "non-LLAMA architecture, no production qualification. The "
            "summarizer does not run tests and does not re-judge their "
            "numerical gates; it aggregates what ctest and the per-test "
            "artifacts recorded."
        ),
    }


def main():
    if len(sys.argv) != 2:
        return fail("usage: union_ctest_gate.py CONFIG.json")
    try:
        with open(sys.argv[1], "r") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        return fail(f"cannot read config {sys.argv[1]}: {exc}")
    for key in ("run_dir", "preflight_list", "tests"):
        if key not in config:
            return fail(f"config {sys.argv[1]} has no {key!r}")

    report = summarize(config)
    out_path = os.path.join(config["run_dir"], "union-ctest-gate.json")
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    # The aggregate verdict is one line. Each test carries only its own
    # status, so a passing test cannot read as failed because another failed.
    print(f"GATE {report['verdict'].upper()}")
    for entry in report["tests"]:
        print(f"TEST {entry['name']} registered={int(entry['registered'])} "
              f"status={entry['status']} wall={entry['wall_time_s']}")
    if report["failure"] is not None:
        print(f"FAIL: {report['failure']}", file=sys.stderr)
    print(f"AGGREGATE {out_path}")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
