#!/usr/bin/env python3
"""Unit tests for nkv-exact-width-report.py.

Pure Python, stdlib only. These tests do not need the test binary or a model;
they feed synthetic logs in the exact shapes tests/test-nkv-exact-width.cpp
prints. Run with:

    python3 -m unittest discover -s tests/nkv-cpu-probe -p 'test_nkv_exact_width_report.py'
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPORT = pathlib.Path(__file__).with_name("nkv-exact-width-report.py")

# A well-formed log for one arm. The step counts are small on purpose: the
# parser must not care about the real 300/9/4/257/300 lengths unless asked.
WELL_FORMED = """\
NKV_ARM binder=unset ntok=unset reuse_disable=unset negative=none
NKV_CASE name=single n_seq=1 unified=1 steps=3 arm=binder skipped=0
NKV_EXACT step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1
NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2
NKV_EXACT step=2 pos=3 mask_ne0=3 k_ne2=3 v_nkv=3
NKV_CASE_END name=single ok=1 steps=3 skipped=0
NKV_CASE name=multi_seq n_seq=2 unified=1 steps=2 arm=binder skipped=0
NKV_MULTI step=0 seq=0 pos=0 used0=1 used1=0 width=1 mask_ne0=1 k_ne2=1 v_nkv=1
NKV_MULTI step=1 seq=1 pos=0 used0=1 used1=1 width=1 mask_ne0=1 k_ne2=1 v_nkv=1
NKV_CASE_END name=multi_seq ok=1 steps=2 skipped=0
NKV_CASE name=multi_stream n_seq=2 unified=0 steps=1 arm=binder skipped=0
NKV_MULTI_STREAM n_ctx=300 n_ctx_seq=256 n_ctx_kv=256 n_seq_max=2 stock_width=256
NKV_MULTI_STREAM step=0 seq=0 pos=0 mask_ne0=256 mask_ne1=1 mask_ne3=2 k_ne2=256 v_nkv=256
NKV_MULTI_STREAM_BIND step=0 state=0 patch_binds=0 fallbacks=0
NKV_CASE_END name=multi_stream ok=1 steps=1 skipped=0
NKV_CASE name=widths n_seq=1 unified=1 steps=2 arm=binder skipped=0
NKV_WIDTH step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1 at_capacity=0
NKV_WIDTH step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2 at_capacity=0
NKV_CASE_END name=widths ok=1 steps=2 skipped=0
NKV_CASE name=off n_seq=1 unified=1 steps=2 arm=off skipped=0
OFF_ARM step=0 pos=1 mask_ne0=256 k_ne2=256 v_nkv=256
OFF_ARM step=1 pos=2 mask_ne0=256 k_ne2=256 v_nkv=256
OFF_ARM builds=1 steps=2
OFF_ARM_REUSE ok=1 builds=1 steps=2
NKV_CASE_END name=off ok=1 steps=2 skipped=0
NKV_EXACT_END ok=1 cases=5 steps=10
"""

CASES = ["single", "multi_seq", "multi_stream", "widths", "off"]
STEPS = [3, 2, 1, 2, 2]


class ReportTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = pathlib.Path(self._dir.name)

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def run_report(self, arms, extra=()):
        """arms: list of (name, log_path_or_None, rc_or_None).

        A None log means "this arm did not run". The arm still reaches the
        CLI: the contract under test is that the writer, not the caller,
        decides NOT_RUN from a missing file, so the helper passes a path
        inside the temp dir that does not exist. Skipping --arm here would
        test nothing: the report would simply have no entry.
        """
        out = self.dir / "report.json"
        argv = [sys.executable, str(REPORT), "--out", str(out)]
        for name, log, rc in arms:
            path = log if log is not None else self.dir / (name + ".absent.log")
            argv += ["--arm", "%s=%s" % (name, path)]
            if rc is not None:
                argv += ["--arm-rc", "%s=%s" % (name, rc)]
        argv += list(extra)
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        return proc, report

    # ------------------------------------------------------------ well formed

    def test_well_formed_log_parses_cases_and_steps(self):
        log = self.write("unset.log", WELL_FORMED)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        arm = report["arms"]["unset"]
        self.assertTrue(arm["ran"])
        self.assertEqual(arm["status"], "OK")
        self.assertEqual([c["name"] for c in arm["cases"]], CASES)
        self.assertEqual([c["declared_steps"] for c in arm["cases"]], STEPS)
        self.assertEqual([c["observed_steps"] for c in arm["cases"]], STEPS)
        self.assertEqual([c["footer_steps"] for c in arm["cases"]], STEPS)
        self.assertEqual([c["ok"] for c in arm["cases"]], [1] * 5)
        self.assertEqual(arm["cases"][0]["n_seq"], 1)
        self.assertEqual(arm["cases"][0]["unified"], 1)
        self.assertEqual(arm["cases"][1]["n_seq"], 2)
        self.assertEqual(arm["cases"][2]["unified"], 0)
        self.assertEqual(len(arm["steps"]["NKV_EXACT"]), 3)
        self.assertEqual(len(arm["steps"]["NKV_MULTI"]), 2)
        self.assertEqual(len(arm["steps"]["NKV_MULTI_STREAM"]), 1)
        self.assertEqual(len(arm["steps"]["NKV_MULTI_STREAM_BIND"]), 1)
        self.assertEqual(len(arm["steps"]["NKV_WIDTH"]), 2)
        self.assertEqual(len(arm["steps"]["OFF_ARM"]), 2)
        self.assertEqual(arm["steps"]["NKV_EXACT"][1],
                         {"step": 1, "pos": 2, "mask_ne0": 2, "k_ne2": 2, "v_nkv": 2})
        self.assertEqual(arm["multi_stream_context"]["stock_width"], 256)
        self.assertEqual(arm["off_arm_total"], {"builds": 1, "steps": 2})
        self.assertEqual(arm["off_arm_reuse"], {"ok": 1, "builds": 1, "steps": 2})
        self.assertEqual(arm["end"], {"ok": 1, "cases": 5, "steps": 10})
        self.assertEqual(arm["arm_env"],
                         {"binder": "unset", "ntok": "unset",
                          "reuse_disable": "unset", "negative": "none"})
        self.assertEqual(arm["fail_lines"], [])
        self.assertEqual(report["summary"], {"unset": "OK"})
        self.assertTrue(report["overall"]["ok"])

    def test_arm_line_is_required(self):
        # A log without the NKV_ARM line has no proof of the path it ran, so
        # it cannot be reported OK even when every case passed.
        text = WELL_FORMED.replace(
            "NKV_ARM binder=unset ntok=unset reuse_disable=unset negative=none\n", "")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        arm = report["arms"]["unset"]
        self.assertEqual(arm["status"], "FAIL")
        self.assertIsNone(arm["arm_env"])
        self.assertIn("NKV_ARM", arm["reason"])

    def test_duplicate_arm_line_is_flagged(self):
        text = WELL_FORMED.replace(
            "NKV_ARM binder=unset ntok=unset reuse_disable=unset negative=none\n",
            "NKV_ARM binder=unset ntok=unset reuse_disable=unset negative=none\n"
            "NKV_ARM binder=on ntok=unset reuse_disable=unset negative=none\n")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("NKV_ARM", report["arms"]["unset"]["reason"])

    def test_expectations_match(self):
        log = self.write("unset.log", WELL_FORMED)
        proc, report = self.run_report(
            [("unset", log, 0)],
            ["--expect-cases", ",".join(CASES), "--expect-steps",
             ",".join(str(s) for s in STEPS)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "OK")

    def test_skipped_case_is_not_compared_against_expected_steps(self):
        # The off arm: single/multi_seq/multi_stream/widths are skipped and
        # carry no step rows, so only `off` is compared against its count.
        text = """\
NKV_ARM binder=0 ntok=unset reuse_disable=unset negative=none
NKV_CASE name=single n_seq=1 unified=1 steps=300 arm=binder skipped=1
NKV_CASE_END name=single ok=1 steps=0 skipped=1
NKV_CASE name=multi_seq n_seq=2 unified=1 steps=9 arm=binder skipped=1
NKV_CASE_END name=multi_seq ok=1 steps=0 skipped=1
NKV_CASE name=multi_stream n_seq=2 unified=0 steps=4 arm=binder skipped=1
NKV_CASE_END name=multi_stream ok=1 steps=0 skipped=1
NKV_CASE name=widths n_seq=1 unified=1 steps=257 arm=binder skipped=1
NKV_CASE_END name=widths ok=1 steps=0 skipped=1
NKV_CASE name=off n_seq=1 unified=1 steps=2 arm=off skipped=0
OFF_ARM step=0 pos=1 mask_ne0=256 k_ne2=256 v_nkv=256
OFF_ARM step=1 pos=2 mask_ne0=256 k_ne2=256 v_nkv=256
OFF_ARM builds=1 steps=2
OFF_ARM_REUSE ok=1 builds=1 steps=2
NKV_CASE_END name=off ok=1 steps=2 skipped=0
NKV_EXACT_END ok=1 cases=1 steps=2
"""
        log = self.write("off.log", text)
        proc, report = self.run_report(
            [("off", log, 0)],
            ["--expect-cases", ",".join(CASES), "--expect-steps", "2"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        arm = report["arms"]["off"]
        self.assertEqual(arm["status"], "OK")
        self.assertEqual([c["skipped"] for c in arm["cases"]],
                         ["1", "1", "1", "1", "0"])

    def test_skipped_case_with_the_wrong_step_count_still_fails(self):
        # a skipped case that printed step rows is a contradiction: it claims
        # not to have run and yet observed steps
        text = """\
NKV_ARM binder=0 ntok=unset reuse_disable=unset negative=none
NKV_CASE name=single n_seq=1 unified=1 steps=300 arm=binder skipped=1
NKV_EXACT step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1
NKV_CASE_END name=single ok=1 steps=300 skipped=1
NKV_EXACT_END ok=1 cases=1 steps=1
"""
        log = self.write("off.log", text)
        proc, report = self.run_report(
            [("off", log, 0)],
            ["--expect-cases", "single", "--expect-steps", "1"])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["off"]["status"], "FAIL")

    # ------------------------------------------------------------ NOT_RUN

    def test_missing_log_is_not_run_without_numbers(self):
        proc, report = self.run_report([("unset", None, None), ("on", None, None)])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertIn("OVERALL NOT_RUN", proc.stdout)
        for name in ("unset", "on"):
            arm = report["arms"][name]
            self.assertFalse(arm["ran"])
            self.assertEqual(arm["status"], "NOT_RUN")
            self.assertEqual(arm["reason"], "log file missing")
            for absent in ("cases", "case_count", "steps", "end",
                           "multi_stream_context", "off_arm_total",
                           "off_arm_reuse", "fail_lines"):
                self.assertNotIn(absent, arm)
            self.assertFalse(report["overall"]["ok"])
        self.assertEqual(report["summary"], {"unset": "NOT_RUN", "on": "NOT_RUN"})

    def test_missing_log_path_that_does_not_exist_is_not_run(self):
        proc, report = self.run_report([("off", self.dir / "absent.log", 0)])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(report["arms"]["off"]["status"], "NOT_RUN")
        self.assertFalse(report["arms"]["off"]["ran"])

    def test_one_arm_not_run_blocks_overall_pass(self):
        log = self.write("unset.log", WELL_FORMED)
        proc, report = self.run_report([("unset", log, 0), ("on", None, None)])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(report["summary"]["unset"], "OK")
        self.assertEqual(report["summary"]["on"], "NOT_RUN")
        self.assertFalse(report["overall"]["ok"])

    # ------------------------------------------------------------ failures

    def test_footer_ok_zero_makes_arm_not_ok(self):
        log = self.write("unset.log", WELL_FORMED.replace(
            "NKV_CASE_END name=widths ok=1 steps=2",
            "NKV_CASE_END name=widths ok=0 steps=2"))
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "FAIL")
        self.assertIn("widths", report["arms"]["unset"]["reason"])

    def test_end_footer_ok_zero_makes_arm_not_ok(self):
        log = self.write("unset.log", WELL_FORMED.replace(
            "NKV_EXACT_END ok=1 cases=5 steps=10",
            "NKV_EXACT_END ok=0 cases=5 steps=10"))
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "FAIL")
        self.assertIn("NKV_EXACT_END", report["arms"]["unset"]["reason"])

    def test_fail_line_makes_arm_not_ok(self):
        text = WELL_FORMED.replace(
            "NKV_EXACT step=2 pos=3 mask_ne0=3 k_ne2=3 v_nkv=3\n",
            "FAIL: padded width at step=2 pos=3 mask=4 k=4 v=4\n")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        arm = report["arms"]["unset"]
        self.assertEqual(arm["status"], "FAIL")
        self.assertEqual(arm["fail_lines"],
                         ["padded width at step=2 pos=3 mask=4 k=4 v=4"])
        self.assertIn("FAIL: line", arm["reason"])

    def test_nonzero_rc_makes_arm_not_ok(self):
        log = self.write("unset.log", WELL_FORMED)
        proc, report = self.run_report([("unset", log, 1)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "FAIL")
        self.assertEqual(report["arms"]["unset"]["rc"], 1)

    def test_unexpected_case_name_is_flagged(self):
        text = WELL_FORMED.replace("NKV_CASE name=widths ", "NKV_CASE name=wide ")
        text = text.replace("NKV_CASE_END name=widths ", "NKV_CASE_END name=wide ")
        log = self.write("unset.log", text)
        proc, report = self.run_report(
            [("unset", log, 0)], ["--expect-cases", ",".join(CASES)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "FAIL")
        self.assertIn("case list", report["arms"]["unset"]["reason"])

    def test_wrong_step_count_is_flagged(self):
        log = self.write("unset.log", WELL_FORMED)
        proc, report = self.run_report(
            [("unset", log, 0)],
            ["--expect-steps", "3,2,1,257,2"])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "FAIL")
        self.assertIn("widths", report["arms"]["unset"]["reason"])

    def test_observed_step_count_mismatch_is_flagged(self):
        # the footer claims 2 steps but only one NKV_WIDTH row was printed
        text = WELL_FORMED.replace(
            "NKV_WIDTH step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2 at_capacity=0\n", "")
        log = self.write("unset.log", text)
        proc, report = self.run_report(
            [("unset", log, 0)], ["--expect-steps", ",".join(str(s) for s in STEPS)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "FAIL")

    def test_header_without_footer_is_flagged(self):
        text = WELL_FORMED.replace("NKV_CASE_END name=widths ok=1 steps=2 skipped=0\n", "")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["status"], "FAIL")
        self.assertIsNone(report["arms"]["unset"]["cases"][3]["ok"])

    # ------------------------------------------------------------ nulls

    def test_absent_optional_families_are_null_not_zero(self):
        text = """\
NKV_ARM binder=unset ntok=unset reuse_disable=unset negative=none
NKV_CASE name=single n_seq=1 unified=1 steps=1
NKV_EXACT step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1
NKV_CASE_END name=single ok=1 steps=1
NKV_EXACT_END ok=1 cases=1 steps=1
"""
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        arm = report["arms"]["unset"]
        self.assertIsNone(arm["multi_stream_context"])
        self.assertIsNone(arm["off_arm_total"])
        self.assertIsNone(arm["off_arm_reuse"])
        self.assertEqual(arm["steps"]["NKV_MULTI"], [])
        self.assertEqual(arm["steps"]["NKV_WIDTH"], [])
        # the single case has no multi_seq row, so its observed count stays a
        # measured 1 from NKV_EXACT, and the absent families stay empty
        self.assertEqual(arm["cases"][0]["observed_steps"], 1)

    def test_absent_case_fields_are_null(self):
        text = """\
NKV_ARM binder=unset ntok=unset reuse_disable=unset negative=none
NKV_CASE name=single n_seq=1 unified=1 steps=1
NKV_EXACT step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1
NKV_CASE_END name=single ok=1 steps=1
NKV_EXACT_END ok=1 cases=1 steps=1
"""
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIsNone(report["arms"]["unset"]["multi_stream_context"])

    def test_arm_rc_without_arm_is_rejected(self):
        out = self.dir / "report.json"
        proc = subprocess.run(
            [sys.executable, str(REPORT), "--out", str(out), "--arm-rc", "on=0"],
            capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no matching --arm", proc.stderr)

    # ------------------------------------------------------------ pad tail

    def test_pad_tail_none_and_zero_are_different_facts(self):
        # "none" is "no node carried the flag"; 0 is "a node carried it and
        # the width is zero". A report that folded one into the other would
        # hide which one the log said.
        text = """\
NKV_ARM binder=unset ntok=unset reuse_disable=unset negative=none
NKV_CASE name=single n_seq=1 unified=1 steps=2
NKV_EXACT step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1 pad_tail=none
NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2 pad_tail=0
NKV_CASE_END name=single ok=1 steps=2
NKV_EXACT_END ok=1 cases=1 steps=2
"""
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = report["arms"]["unset"]["steps"]["NKV_EXACT"]
        self.assertEqual(rows[0]["pad_tail"], "none")
        self.assertEqual(rows[1]["pad_tail"], 0)
        self.assertIsNot(rows[0]["pad_tail"], rows[1]["pad_tail"])
        self.assertEqual(report["arms"]["unset"]["pad_tail"], ["none", 0])

    def test_pad_tail_integer_is_an_integer(self):
        text = WELL_FORMED.replace(
            "NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2",
            "NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2 pad_tail=2")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            report["arms"]["unset"]["steps"]["NKV_EXACT"][1]["pad_tail"], 2)

    def test_line_without_pad_tail_is_still_accepted(self):
        # WELL_FORMED carries no pad_tail token at all: the field is absent,
        # which is not a zero and not a failure of the parser
        log = self.write("unset.log", WELL_FORMED)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("pad_tail",
                         report["arms"]["unset"]["steps"]["NKV_EXACT"][0])
        self.assertEqual(report["arms"]["unset"]["pad_tail"], [None] * 10)

    def test_pad_tail_summary_covers_every_step_family(self):
        text = WELL_FORMED.replace(
            "NKV_WIDTH step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1 at_capacity=0",
            "NKV_WIDTH step=0 pos=1 mask_ne0=1 k_ne2=1 v_nkv=1 at_capacity=0 "
            "pad_tail=none").replace(
            "OFF_ARM step=0 pos=1 mask_ne0=256 k_ne2=256 v_nkv=256",
            "OFF_ARM step=0 pos=1 mask_ne0=256 k_ne2=256 v_nkv=256 pad_tail=none")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = report["arms"]["unset"]["pad_tail"]
        self.assertEqual(summary.count("none"), 2)
        self.assertEqual(len(summary), 10)

    # ------------------------------------------------------------ NKV_RT

    def test_nkv_rt_fields_parse_as_integers(self):
        text = WELL_FORMED.replace(
            "NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2\n",
            "NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2\n"
            "NKV_RT step=1 runtime=2 bound=2 pos=2 patch_binds=2\n")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["steps"]["NKV_RT"],
                         [{"step": 1, "runtime": 2, "bound": 2, "pos": 2,
                           "patch_binds": 2}])

    def test_malformed_nkv_rt_line_is_a_failure_not_an_absence(self):
        # the line names the family but drops bound=, so it is not a step row
        # and must not silently disappear
        text = WELL_FORMED.replace(
            "NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2\n",
            "NKV_EXACT step=1 pos=2 mask_ne0=2 k_ne2=2 v_nkv=2\n"
            "NKV_RT step=1 runtime=2 pos=2 patch_binds=2\n")
        log = self.write("unset.log", text)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        arm = report["arms"]["unset"]
        self.assertEqual(arm["status"], "FAIL")
        self.assertEqual(arm["steps"]["NKV_RT"], [])
        self.assertEqual(len(arm["fail_lines"]), 1)
        self.assertIn("malformed NKV_RT line", arm["fail_lines"][0])

    def test_absent_nkv_rt_family_is_empty_not_null(self):
        log = self.write("unset.log", WELL_FORMED)
        proc, report = self.run_report([("unset", log, 0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["arms"]["unset"]["steps"]["NKV_RT"], [])


if __name__ == "__main__":
    unittest.main()
