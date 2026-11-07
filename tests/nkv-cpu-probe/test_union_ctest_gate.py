#!/usr/bin/env python3
"""Fixture test for union_ctest_gate.py, the aggregate gate summarizer.

Runs on CPU with no model, no compiler and no build tree: it writes synthetic
`ctest -N` listings, synthetic JUnit XML and synthetic per-test artifacts into a
temporary directory, runs the summarizer on them, and asserts the aggregate.

What it proves:
  * every declared test registered, ran and passed -> verdict pass, exit 0;
  * a declared test missing from the listing -> verdict fail, exit 1, the
    failure names that test, and no test is recorded as having run;
  * a passing test whose artifact is absent -> status missing_artifact, verdict
    fail, exit 1;
  * an artifact that exists but is not the tool's own output -> bad_artifact;
  * a test ctest reported as failed -> verdict fail, exit 1;
  * a failing pre-flight outranks a later problem, so the first failure named
    is the missing registration;
  * the aggregate is valid JSON with the documented top-level keys.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SUMMARIZER = pathlib.Path(__file__).with_name("union_ctest_gate.py")

# A varlen_output_probe record file: five uint32 header words per step, then the
# row indices and the logits. The summarizer checks structure only, so one
# well-formed header is enough.
ROWS_BIN = b"\x00\x00\x00\x00" * 5

TOP_LEVEL = ("schema", "verdict", "failure", "argv", "exit_code", "run_dir",
             "build_dir", "preflight", "ctest", "declared", "tests", "not_run")


def listing(names):
    """A `ctest -N` listing in ctest's own shape."""
    lines = ["Test project /tmp/synthetic", ""]
    for index, name in enumerate(names, start=1):
        lines.append(f"  Test #{index}: {name}")
    lines += ["", f"Total Tests: {len(names)}", ""]
    return "\n".join(lines)


def junit(cases):
    """JUnit XML in the shape ctest --output-junit writes.

    cases: name -> (status, seconds), status in run | fail | notrun.
    """
    body = []
    for name, (status, seconds) in cases.items():
        body.append(f'\t<testcase name="{name}" classname="{name}" '
                    f'time="{seconds}" status="{status}">\n\t\t<properties/>\n\t</testcase>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<testsuite name="(empty)" tests="{len(cases)}" failures="0" '
            'disabled="0" skipped="0" hostname="" time="0" '
            'timestamp="2026-01-01T00:00:00">\n' + "\n".join(body) + "\n</testsuite>\n")


E2E_JSON = {
    "schema": "union-varlen-e2e/4",
    "verdict": "pass",
    "failure": None,
    "argv": ["test-union-varlen-e2e", "model.gguf", "out.json", "64"],
    "exit_code": 0,
}

DECLARED = ["test-union-binder-gate", "test-union-varlen-e2e", "test-varlen-output-fixed"]


class GateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.run_dir = self.dir / "run"
        self.run_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text)
        return path

    def artifacts(self):
        return {
            "test-union-varlen-e2e": [["e2e-json", str(self.dir / "e2e.json")]],
            "test-varlen-output-fixed": [["rows-bin", str(self.dir / "fixed-on.bin")]],
        }

    def run_gate(self, registered, cases, artifacts=None, preflight_rc=0, run_rc=0):
        listing_path = self.write("ctest-list.txt", listing(registered))
        junit_path = self.write("ctest-junit.xml", junit(cases))
        declared = []
        for name in DECLARED:
            declared.append({
                "name": name,
                "log": str(self.run_dir / "ctest.log"),
                "artifacts": (self.artifacts() if artifacts is None else artifacts).get(name, []),
            })
        config = {
            "run_dir": str(self.run_dir),
            "build_dir": str(self.dir),
            "preflight_rc": preflight_rc,
            "preflight_list": str(listing_path),
            "run_rc": run_rc,
            "junit": str(junit_path),
            "argv": ["synthetic", *DECLARED],
            "tests": declared,
        }
        config_path = self.write("config.json", json.dumps(config))
        result = subprocess.run(
            [sys.executable, str(SUMMARIZER), str(config_path)],
            capture_output=True, text=True, check=False,
        )
        report_path = self.run_dir / "union-ctest-gate.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        return result, report

    def write_good_artifacts(self):
        (self.dir / "e2e.json").write_text(json.dumps(E2E_JSON))
        (self.dir / "fixed-on.bin").write_bytes(ROWS_BIN)

    def test_all_present_and_passing_passes(self):
        self.write_good_artifacts()
        result, report = self.run_gate(
            DECLARED,
            {name: ("run", 0.25) for name in DECLARED},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["verdict"], "pass")
        self.assertIsNone(report["failure"])
        self.assertEqual(report["exit_code"], 0)
        for key in TOP_LEVEL:
            self.assertIn(key, report)
        self.assertEqual(report["schema"], "union-ctest-gate/1")
        for entry in report["tests"]:
            self.assertTrue(entry["registered"])
            self.assertEqual(entry["status"], "passed")
            self.assertEqual(entry["exit_code"], 0)
            self.assertEqual(entry["wall_time_s"], 0.25)
            self.assertTrue(all(artifact["ok"] for artifact in entry["artifacts"]))

    def test_missing_registration_fails_closed(self):
        self.write_good_artifacts()
        # the listing omits test-union-varlen-e2e; ctest therefore never ran it
        result, report = self.run_gate(
            ["test-union-binder-gate", "test-varlen-output-fixed"],
            {"test-union-binder-gate": ("run", 0.1),
             "test-varlen-output-fixed": ("run", 0.2)},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        self.assertEqual(report["exit_code"], 1)
        self.assertIn("not_registered", report["failure"])
        self.assertIn("test-union-varlen-e2e", report["failure"])
        entry = next(e for e in report["tests"] if e["name"] == "test-union-varlen-e2e")
        self.assertFalse(entry["registered"])
        self.assertEqual(entry["status"], "not_run")
        self.assertIn("not_registered", entry["problems"])
        self.assertNotIn("test-union-varlen-e2e", report["preflight"]["registered"])

    def test_missing_artifact_fails(self):
        # the e2e JSON is never written, although the test passed
        (self.dir / "fixed-on.bin").write_bytes(ROWS_BIN)
        result, report = self.run_gate(
            DECLARED,
            {name: ("run", 0.25) for name in DECLARED},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        self.assertIn("missing_artifact", report["failure"])
        self.assertIn("test-union-varlen-e2e", report["failure"])
        entry = next(e for e in report["tests"] if e["name"] == "test-union-varlen-e2e")
        self.assertEqual(entry["status"], "passed")
        self.assertFalse(entry["artifacts"][0]["ok"])
        self.assertIn("missing_artifact", entry["problems"])

    def test_wrong_artifact_content_fails(self):
        (self.dir / "e2e.json").write_text(json.dumps({"schema": "something/else"}))
        (self.dir / "fixed-on.bin").write_bytes(ROWS_BIN)
        result, report = self.run_gate(
            DECLARED,
            {name: ("run", 0.25) for name in DECLARED},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        self.assertIn("bad_artifact", report["failure"])
        entry = next(e for e in report["tests"] if e["name"] == "test-union-varlen-e2e")
        self.assertIn("bad_artifact", entry["problems"])

    def test_failed_test_fails(self):
        self.write_good_artifacts()
        result, report = self.run_gate(
            DECLARED,
            {"test-union-binder-gate": ("run", 0.1),
             "test-union-varlen-e2e": ("fail", 0.3),
             "test-varlen-output-fixed": ("run", 0.2)},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        self.assertIn("failed", report["failure"])
        self.assertIn("test-union-varlen-e2e", report["failure"])
        entry = next(e for e in report["tests"] if e["name"] == "test-union-varlen-e2e")
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["exit_code"], 1)

    def test_preflight_failure_outranks_later_problems(self):
        self.write_good_artifacts()
        result, report = self.run_gate(
            DECLARED,
            {name: ("run", 0.25) for name in DECLARED},
            preflight_rc=8,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        self.assertTrue(report["failure"].startswith("preflight_failed"), report["failure"])

    def test_registered_but_not_run_fails(self):
        self.write_good_artifacts()
        # registered in the listing, but ctest's JUnit has no case for it
        result, report = self.run_gate(
            DECLARED,
            {"test-union-binder-gate": ("run", 0.1),
             "test-varlen-output-fixed": ("run", 0.2)},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        self.assertIn("not_run", report["failure"])
        entry = next(e for e in report["tests"] if e["name"] == "test-union-varlen-e2e")
        self.assertTrue(entry["registered"])
        self.assertEqual(entry["status"], "not_run")

    def test_failed_test_artifact_is_not_evidence(self):
        # the artifact is well formed, but it is stale: the test itself failed,
        # so the file says nothing about this run and must not be read
        self.write_good_artifacts()
        result, report = self.run_gate(
            DECLARED,
            {"test-union-binder-gate": ("run", 0.1),
             "test-union-varlen-e2e": ("fail", 0.3),
             "test-varlen-output-fixed": ("run", 0.2)},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        # the test's own failure is the first condition, not its artifact
        self.assertTrue(report["failure"].startswith("failed: test-union-varlen-e2e"),
                        report["failure"])
        entry = next(e for e in report["tests"] if e["name"] == "test-union-varlen-e2e")
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(len(entry["artifacts"]), 1)
        artifact = entry["artifacts"][0]
        self.assertFalse(artifact["ok"])
        self.assertFalse(artifact["checked"])
        self.assertIn("not_checked", entry["problems"])
        self.assertNotIn("bad_artifact", entry["problems"])
        self.assertNotIn("missing_artifact", entry["problems"])

    def test_failing_e2e_report_is_a_bad_artifact(self):
        # ctest recorded the test as passed, but the driver's own report says
        # it failed: the report is not evidence that the driver passed
        failing = dict(E2E_JSON, verdict="fail", exit_code=1,
                       failure="logits rows differ between arms")
        (self.dir / "e2e.json").write_text(json.dumps(failing))
        (self.dir / "fixed-on.bin").write_bytes(ROWS_BIN)
        result, report = self.run_gate(
            DECLARED,
            {name: ("run", 0.25) for name in DECLARED},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["verdict"], "fail")
        self.assertIn("bad_artifact", report["failure"])
        self.assertIn("test-union-varlen-e2e", report["failure"])
        entry = next(e for e in report["tests"] if e["name"] == "test-union-varlen-e2e")
        self.assertEqual(entry["status"], "passed")
        artifact = entry["artifacts"][0]
        self.assertTrue(artifact["checked"])
        self.assertFalse(artifact["ok"])
        detail = next(p for p in entry["problems"] if p.startswith("report verdict"))
        self.assertIn("'fail'", detail)
        self.assertIn("1", detail)


if __name__ == "__main__":
    unittest.main()
