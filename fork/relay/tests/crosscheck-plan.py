#!/usr/bin/env python3
"""Live cross-check: the C++ plan port against the verified Python reference.

Runs `tp5_local_plan.py` (recorded SHA-256 59129982...59e657, guide 54.15)
on the four fixed fixtures and compares its output with the compiled
`relay-plan-dump` summary of `tq::relay::compile_glu_plan` for the same
specs. Passes only when both implementations agree on decision, rejection
code, totals, per-rank weight bytes, per-rank linear MACs, middle ranges
and output ranges.

Scope: CPU-only fixed comparison set; no GPU, no model qualification.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

TP5_LOCAL_PLAN_SHA256 = (
    "5912998214d411bda905a25f58878bf9550c8e7e7cfa9ad43733a5736b59e657"
)

CASES = (
    ("glu-gather", 0, "feasible", None),
    ("glu-gather-replicated", 0, "feasible", None),
    ("glu-middle-infeasible", 3, "infeasible", "INSUFFICIENT_UNITS"),
    ("glu-middle-aligned", 0, "feasible", None),
)


def fail(message):
    raise RuntimeError(message)


def run_python(planner: Path, fixture: Path, expected_exit: int) -> dict:
    run = subprocess.run(
        [sys.executable, str(planner), "--input", str(fixture)],
        capture_output=True, text=True, check=False,
    )
    if run.returncode != expected_exit or run.stderr:
        fail(f"{fixture.name}: exit={run.returncode} stderr={run.stderr!r}")
    return json.loads(run.stdout)


def compare_case(name: str, python: dict, cpp: dict) -> None:
    if cpp["decision"] != python["decision"]:
        fail(f"{name}: decision {cpp['decision']} != {python['decision']}")
    if python["decision"] != "feasible":
        if cpp.get("error") != python["error"]["code"]:
            fail(f"{name}: error {cpp.get('error')} != {python['error']['code']}")
        return
    if cpp["source_weight_bytes"] != python["totals"]["source_weight_bytes"]:
        fail(f"{name}: source_weight_bytes drifted")
    if cpp["linear_macs"] != python["totals"]["linear_macs"]:
        fail(f"{name}: linear_macs drifted")
    if cpp["endpoint_send_bytes"] != python["totals"]["endpoint_send_bytes"]:
        fail(f"{name}: endpoint_send_bytes drifted")
    if cpp["endpoint_receive_bytes"] != python["totals"]["endpoint_receive_bytes"]:
        fail(f"{name}: endpoint_receive_bytes drifted")
    for rank in range(5):
        if cpp["per_rank_source_weight_bytes"][rank] != python["ranks"][rank]["source_weight_bytes"]:
            fail(f"{name}: rank {rank} source bytes drifted")
        if cpp["per_rank_linear_macs"][rank] != python["ranks"][rank]["linear_macs"]:
            fail(f"{name}: rank {rank} linear MACs drifted")
        if cpp["middle_ranges"][rank] != python["ranks"][rank]["gate_up_output_range"]:
            fail(f"{name}: rank {rank} middle range drifted")
        if cpp["output_ranges"][rank] != python["ranks"][rank]["down_output_range"]:
            fail(f"{name}: rank {rank} output range drifted")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: crosscheck-plan.py <relay-plan-dump> <python-reference-dir>",
              file=sys.stderr)
        return 2
    dump_binary = Path(sys.argv[1])
    reference = Path(sys.argv[2])
    if not dump_binary.is_file():
        print("relay-plan-dump binary not built; skip", file=sys.stderr)
        return 77
    planner = reference / "tp5_local_plan.py"
    if not planner.is_file():
        print("python reference not extracted; skip", file=sys.stderr)
        return 77
    digest = hashlib.sha256(planner.read_bytes()).hexdigest()
    if digest != TP5_LOCAL_PLAN_SHA256:
        fail(f"tp5_local_plan.py sha256 {digest} != recorded {TP5_LOCAL_PLAN_SHA256}")

    self_test = subprocess.run([sys.executable, str(planner), "--self-test"],
                               capture_output=True, text=True, check=False)
    if self_test.returncode != 0 or self_test.stderr:
        fail(f"python self-test failed: {self_test.stderr!r}")

    dump_run = subprocess.run([str(dump_binary)], capture_output=True, text=True, check=False)
    if dump_run.returncode != 0 or dump_run.stderr:
        fail(f"relay-plan-dump failed: {dump_run.stderr!r}")
    cpp_results = {item["name"]: item for item in json.loads(dump_run.stdout)}

    for name, expected_exit, expected_decision, expected_error in CASES:
        fixture = reference / f"{name}.json"
        python = run_python(planner, fixture, expected_exit)
        if python["decision"] != expected_decision:
            fail(f"{name}: python decision {python['decision']} != {expected_decision}")
        if expected_error is not None and python["error"]["code"] != expected_error:
            fail(f"{name}: python error drifted")
        if python.get("gpu_used") is not False:
            fail(f"{name}: python claimed GPU use")
        compare_case(name, python, cpp_results[name])

    print("PASS: C++ plan port matches the verified Python reference on the fixed fixture set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
