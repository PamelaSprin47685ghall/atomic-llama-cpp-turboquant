#!/usr/bin/env python3
"""CPU-only tests for evidence handling. Does not initialize a GPU/model."""
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench, logits = load("tp5-bench"), load("compare-logits")


class EvidenceTests(unittest.TestCase):
    def test_weighted_timings(self):
        def sample(tokens, milliseconds, wall, warmup=False, correct=True):
            return {"kind": "count", "warmup": warmup, "correct": correct, "wall_ms": wall,
                    "response": {"timings": {"predicted_n": tokens, "predicted_ms": milliseconds}}}
        summary = bench.summarize([sample(100, 1000, 2000), sample(10, 1000, 2000),
                                   sample(999, 1, 1, warmup=True)])
        self.assertEqual(summary["decode_tokens_per_second"], 55)
        self.assertEqual(summary["request_tokens_per_second"], 27.5)
        self.assertEqual(summary["generated_tokens"], 110)
        self.assertFalse(bench.summarize([sample(1, 1, 1, correct=False)])["all_checks_passed"])

    def tape(self, path, values, token=1):
        path.write_bytes(b"TP5LOG2\0" + struct.pack("<4I", 2, 1, 1, 2) +
                         struct.pack("<2i", token, token) + struct.pack("<4f", *values))

    def test_full_vocabulary_not_only_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = Path(directory) / "ref", Path(directory) / "cand"
            self.tape(ref, [1, 3, 2, 4])
            self.tape(cand, [1, 3, 2, 4])
            self.assertTrue(logits.compare(ref, cand)["bitwise"])
            self.tape(cand, [1.125, 3, 2, 4])
            result = logits.compare(ref, cand)
            self.assertFalse(result["pass"])
            self.assertEqual(result["argmax_mismatches"], 0)
            self.assertEqual(result["different_values"], 1)
            self.assertTrue(logits.compare(ref, cand, atol=0.125)["pass"])
            self.tape(cand, [1, 3, 2, 4], token=2)
            with self.assertRaises(ValueError):
                logits.compare(ref, cand)

    def test_truncation_and_nan_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = Path(directory) / "ref", Path(directory) / "cand"
            self.tape(ref, [1, 3, 2, 4])
            self.tape(cand, [math.nan, 3, 2, 4])
            self.assertFalse(logits.compare(ref, cand)["pass"])
            cand.write_bytes(cand.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                logits.compare(ref, cand)


if __name__ == "__main__":
    unittest.main()
