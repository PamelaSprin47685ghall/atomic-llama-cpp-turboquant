#!/usr/bin/env python3
"""Exercise the real comparison CLI with actual temporary binary tapes."""

import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "compare-logits.py"
SPEC = importlib.util.spec_from_file_location("compare_logits", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareLogitsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = self.root / "reference.bin"
        self.candidate = self.root / "candidate.bin"
        self.tape(self.reference)
        self.tape(self.candidate)

    def tape(self, path, values=(1, 3, 2, 4), *, vocab=2, prefill=1, steps=1, tokens=(1, 1)):
        path.write_bytes(b"TP5LOG2\0" + struct.pack("<4I", vocab, prefill, steps, len(tokens))
                         + struct.pack("<" + "i" * len(tokens), *tokens)
                         + struct.pack("<" + "f" * len(values), *values))

    def cli(self, *extra, expected=0):
        completed = subprocess.run(
            [sys.executable, str(TOOL), str(self.reference), str(self.candidate), *map(str, extra)],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        return json.loads(completed.stdout)

    def test_identical_tape(self):
        result = self.cli()
        self.assertTrue(result["pass"])
        self.assertTrue(result["bitwise"])
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["different_values"], 0)
        self.assertEqual(result["normalized_mse"], 0)

    def test_prefill_only_tape(self):
        for path in (self.reference, self.candidate):
            self.tape(path, (1, 3), steps=0, tokens=(1,))
        self.assertTrue(self.cli()["pass"])

    def test_full_vocabulary_difference_even_if_argmax_matches(self):
        self.tape(self.candidate, (1.125, 3, 2, 4))
        result = self.cli(expected=1)
        self.assertFalse(result["pass"])
        self.assertEqual(result["argmax_mismatches"], 0)
        self.assertEqual(result["different_values"], 1)
        self.assertEqual(result["outside_tolerance"], 1)
        self.assertEqual(result["max_abs"], 0.125)
        self.assertEqual(result["first_different_row"], 0)

    def test_absolute_tolerance_boundary(self):
        self.tape(self.candidate, (1.125, 3, 2, 4))
        self.assertTrue(self.cli("--atol", "0.125")["pass"])
        self.assertFalse(self.cli("--atol", "0.124", expected=1)["pass"])

    def test_relative_tolerance_boundary(self):
        self.tape(self.candidate, (1, 3, 2.25, 4))
        self.assertTrue(self.cli("--rtol", "0.125")["pass"])
        self.assertFalse(self.cli("--rtol", "0.124", expected=1)["pass"])

    def test_signed_zero_is_numerical_not_bitwise_agreement(self):
        self.tape(self.reference, (0.0, 3, 2, 4))
        self.tape(self.candidate, (-0.0, 3, 2, 4))
        result = self.cli()
        self.assertTrue(result["pass"])
        self.assertFalse(result["bitwise"])
        self.assertEqual(result["different_values"], 0)

    def test_argmax_is_reported_separately_from_tolerance(self):
        self.tape(self.reference, (1.0, 1.125, 2, 4))
        self.tape(self.candidate, (1.25, 1.125, 2, 4))
        result = self.cli("--atol", "0.25")
        self.assertTrue(result["pass"])
        self.assertEqual(result["argmax_mismatches"], 1)

    def test_nonfinite_values_always_fail(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.tape(self.candidate, (value, 3, 2, 4))
                result = self.cli("--atol", "1e30", expected=1)
                self.assertFalse(result["pass"])
                self.assertEqual(result["nonfinite"], 1)
                self.assertEqual(result["argmax_unavailable_rows"], 1)

    def test_identical_nan_is_not_valid_agreement(self):
        for path in (self.reference, self.candidate):
            self.tape(path, (math.nan, 3, 2, 4))
        result = self.cli(expected=1)
        self.assertTrue(result["bitwise"])
        self.assertFalse(result["pass"])

    def test_teacher_forced_token_identity(self):
        self.tape(self.candidate, tokens=(0, 1))
        result = self.cli(expected=1)
        self.assertIn("teacher-forced", result["error"])

    def test_token_bounds(self):
        for tokens in ((-1, 1), (2, 1)):
            with self.subTest(tokens=tokens):
                self.tape(self.candidate, tokens=tokens)
                self.assertIn("outside", self.cli(expected=1)["error"])

    def test_dimension_validation(self):
        cases = (
            {"vocab": 0}, {"vocab": 2_000_001}, {"prefill": 0},
            {"prefill": 2}, {"steps": 2},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.tape(self.candidate, **kwargs)
                self.assertIn("dimensions", self.cli(expected=1)["error"])

    def test_dimension_identity(self):
        self.tape(self.candidate, (1, 2, 3, 4, 5, 6), vocab=3)
        self.assertIn("dimensions", self.cli(expected=1)["error"])

    def test_magic_and_truncation_at_each_section(self):
        original = self.candidate.read_bytes()
        for length in (0, 7, 8, 20, 24, 28, len(original) - 1):
            with self.subTest(length=length):
                self.candidate.write_bytes(original[:length])
                self.assertFalse(self.cli(expected=1)["pass"])
        self.candidate.write_bytes(b"BADMAGIC" + original[8:])
        self.assertIn("TP5LOG2", self.cli(expected=1)["error"])

    def test_trailing_data(self):
        self.candidate.write_bytes(self.candidate.read_bytes() + b"x")
        self.assertIn("trailing", self.cli(expected=1)["error"])

    def test_report_output_and_no_overwrite(self):
        output = self.root / "report.json"
        result = self.cli("--output", output)
        before = output.read_bytes()
        self.assertEqual(json.loads(before), result)
        self.assertFalse(self.cli("--output", output, expected=1)["pass"])
        self.assertEqual(output.read_bytes(), before)

    def test_report_cannot_replace_input(self):
        before = self.reference.read_bytes()
        self.assertFalse(self.cli("--output", self.reference, expected=1)["pass"])
        self.assertEqual(self.reference.read_bytes(), before)

    def test_missing_input_and_unwritable_report_location(self):
        self.candidate.unlink()
        self.assertFalse(self.cli(expected=1)["pass"])
        self.tape(self.candidate)
        output = self.root / "missing-parent" / "report.json"
        self.assertFalse(self.cli("--output", output, expected=1)["pass"])

    def test_cli_rejects_invalid_tolerances(self):
        for flag, value in (("--atol", "nan"), ("--rtol", "inf"), ("--atol", "-1")):
            with self.subTest(flag=flag, value=value):
                completed = subprocess.run(
                    [sys.executable, str(TOOL), str(self.reference), str(self.candidate), flag, value],
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("finite and nonnegative", completed.stderr)

    def test_python_api_checks_tolerances_too(self):
        for value in (math.nan, math.inf, -1, True, "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.compare(self.reference, self.candidate, atol=value)


if __name__ == "__main__":
    unittest.main()
