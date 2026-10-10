#!/usr/bin/env python3
"""CPU-only tests for the standalone cost calculator. Never starts inference."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tp5-perf-model.py"
SPEC = importlib.util.spec_from_file_location("tp5_perf_model", SCRIPT)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class CostTests(unittest.TestCase):
    def test_cycle(self):
        r = MODEL.cycle(18, 3, 25, 0, 1, 1.75, 1)
        self.assertAlmostEqual(r["ms_per_token"], 29 / 1.75)
        self.assertAlmostEqual(r["ideal_h1_acceptance_threshold"], 29 / 18 - 1)
        self.assertTrue(r["net_faster"])

    def test_faster_kernel_can_lose(self):
        before = MODEL.cycle(18, 3, 25, 0, 1, 1.75, 1)
        after = MODEL.cycle(18, 3, 23, 0, 1, 1.55, 1)
        self.assertGreater(after["ms_per_token"], before["ms_per_token"])

    def test_equal_cost_not_strict_win(self):
        self.assertFalse(MODEL.cycle(18, 0, 18, 0, 0, 1, 0)["net_faster"])

    def test_threshold_not_clamped(self):
        r = MODEL.cycle(10, 1, 29, 0, 0, 2, 1)
        self.assertEqual(r["ideal_h1_acceptance_threshold"], 2)

    def test_no_h1_claim_for_deep(self):
        self.assertNotIn("ideal_h1_acceptance_threshold", MODEL.cycle(18, 3, 30, 1, 1, 2, 3))

    def test_invalid_cycle(self):
        for args in [(0, 1, 1, 0, 0, 1), (18, -1, 1, 0, 0, 1),
                     (18, 0, 0, 0, 0, 1), (18, 1, 1, 0, 0, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                MODEL.cycle(*args)

    def test_horizon_bounds(self):
        for horizon in (-1, 0.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                MODEL.cycle(18, 1, 1, 0, 0, 1, horizon)
        with self.assertRaises(ValueError):
            MODEL.cycle(18, 1, 1, 0, 0, 2.01, 1)

    def test_budget_100(self):
        r = MODEL.budget(100, 1.75, 60)
        self.assertEqual(r["cycle_ms_budget"], 17.5)
        self.assertAlmostEqual(r["required_time_reduction_fraction"], 0.4)

    def test_slower_target(self):
        self.assertLess(MODEL.budget(30, 1, 60)["required_time_reduction_fraction"], 0)

    def test_amdahl_eliminate(self):
        self.assertAlmostEqual(MODEL.amdahl(0.05, eliminate=True)["speedup"], 1 / 0.95)

    def test_amdahl_factor(self):
        self.assertAlmostEqual(MODEL.amdahl(0.5, factor=2)["speedup"], 4 / 3)
        self.assertEqual(MODEL.amdahl(0, factor=2)["speedup"], 1)

    def test_amdahl_invalid(self):
        for kw in ({}, {"factor": 2, "eliminate": True}, {"factor": 0}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                MODEL.amdahl(0.5, **kw)
        for fraction in (-0.1, 1.1, 1):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                MODEL.amdahl(fraction, eliminate=True)

    def test_resource_decimal_units(self):
        r = MODEL.resource(104294400, 512)
        self.assertAlmostEqual(r["memory_ms"], 0.2037)
        self.assertEqual(r["compute_ms"], 0)

    def test_resource_max_not_sum(self):
        r = MODEL.resource(1e9, 500, flops=3e10, compute_tflops=10)
        self.assertEqual(r["resource_max_ms"], 3)

    def test_resource_invalid(self):
        for args, kw in [((1, 0), {}), ((1, 1), {"flops": 1}),
                         ((-1, 1), {}), ((1, 1), {"compute_tflops": -2})]:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                MODEL.resource(*args, **kw)

    def test_nonfinite_and_bool(self):
        for value in (math.nan, math.inf, -math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODEL.number("test", value)

    def test_overflow(self):
        with self.assertRaises(ValueError):
            MODEL.budget(1e-300, 1e300)

    def test_cli_json(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "budget", "--target-tps", "100",
                               "--committed", "1.75"], capture_output=True, text=True, timeout=5)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "CALCULATION_NOT_BENCHMARK")

    def test_cli_invalid(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "budget", "--target-tps", "nan",
                               "--committed", "1"], capture_output=True, text=True, timeout=5)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("finite", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
