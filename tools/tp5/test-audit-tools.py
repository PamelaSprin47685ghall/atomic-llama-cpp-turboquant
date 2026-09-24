#!/usr/bin/env python3
"""CPU-only tests for evidence handling. Does not initialize a GPU/model."""
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench, logits = load("tp5-bench"), load("compare-logits")
matrix_spec = importlib.util.spec_from_file_location(
    "tp5_matrix", Path(__file__).resolve().parents[2] / "scripts/run-tp5-cross-matrix.py")
matrix = importlib.util.module_from_spec(matrix_spec)
matrix_spec.loader.exec_module(matrix)


class EvidenceTests(unittest.TestCase):
    def count_result(self, content, *, prompt_n=31, predicted_n=171, finish="stop"):
        body = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                "timings": {"prompt_n": prompt_n, "predicted_n": predicted_n, "predicted_ms": 3400},
                "usage": {"completion_tokens": predicted_n}}
        response = io.BytesIO(json.dumps(body).encode())
        response.status = 200
        with patch.object(matrix.urllib.request, "urlopen", return_value=response):
            return matrix.execute_bench_request(
                "127.0.0.1", 1, SimpleNamespace(is_running=lambda: True), False, 60)

    def test_count_acceptance_requires_exact_bytes(self):
        expected = ",".join(map(str, range(1, 61)))
        self.assertTrue(self.count_result(expected)["success"])
        self.assertFalse(self.count_result(expected + "\n")["success"])

    def test_preview_and_context_exhaustion_are_not_acceptance(self):
        expected = ",".join(map(str, range(1, 61)))
        self.assertFalse(self.count_result(expected[:40])["success"])
        self.assertFalse(self.count_result(expected, prompt_n=71, predicted_n=185, finish="length")["success"])

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
