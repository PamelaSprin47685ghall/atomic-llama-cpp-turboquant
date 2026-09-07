#!/usr/bin/env python3
"""Regression tests for the local probe runner; no GPU, model or network needed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "nanbeige_audit", Path(__file__).resolve().parents[1] / "scripts" / "nanbeige-audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(name="case", ctx=8192, batch=256, ubatch=256, k="turbo4", v="turbo2")
        self.args = SimpleNamespace(corpus=None, prompt_tokens=8, predict=2, repeats=1,
                                    no_chat=False, needle_tokens=0, request_timeout=1, out=self.root)

    def response(self, answer="391", finish="stop", budget=2, speed=25.0):
        chats = iter([answer, "12", "-3, 0, 2, 7, 11"])

        def request(base, key, path, data=None, timeout=1):
            if path == "/tokenize":
                return {"tokens": list(range(16))}
            if path == "/completion":
                return {"tokens_predicted": budget, "timings": {
                    "prompt_per_second": speed, "predicted_per_second": 25.0}}
            if path == "/v1/chat/completions":
                return {"choices": [{"message": {"content": next(chats)}, "finish_reason": finish}]}
            if path == "/health":
                return {"status": "ok"}
            raise AssertionError(path)
        return request

    def run_probe(self, **kwargs):
        result = {"config": self.config}
        with mock.patch.object(audit, "request", side_effect=self.response(**kwargs)), contextlib.redirect_stdout(io.StringIO()):
            audit.probe(self.args, "local", "key", result)
        return result

    def test_correct_answers(self):
        result = self.run_probe()
        self.assertTrue(all(case["passed"] for case in result["quality_checks"]))
        self.assertEqual(len(result["requests"]), 4)

    def test_wrong_answer_and_cutoff_are_failures(self):
        for kwargs in ({"answer": "392"}, {"finish": "length"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RuntimeError, "quality checks failed"):
                self.run_probe(**kwargs)

    def test_incomplete_completion_and_nonfinite_timings(self):
        for kwargs in ({"budget": 1}, {"speed": float("nan")}, {"speed": float("inf")}, {"speed": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.run_probe(**kwargs)

    def test_failed_case_saved_and_nonzero_exit(self):
        model = self.root / "fixture.gguf"
        model.touch()
        configs = self.root / "configs.json"
        configs.write_text(json.dumps([self.config]), encoding="utf-8")

        @contextlib.contextmanager
        def server(args, config, result):
            result["config"] = config
            yield "local", "key"

        with mock.patch.object(audit, "server", server), \
                mock.patch.object(audit, "request", side_effect=self.response(answer="392")), \
                contextlib.redirect_stdout(io.StringIO()):
            code = audit.main(["probe", "--server", str(model), "--model", str(model),
                               "--configs", str(configs), "--out", str(self.root),
                               "--prompt-tokens", "8", "--predict", "2", "--repeats", "1"])
        self.assertEqual(code, 1)
        saved = json.loads((self.root / "case.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertFalse(saved["quality_checks"][0]["passed"])
        self.assertEqual(len(saved["requests"]), 4)

    def test_invalid_matrices(self):
        bad = [[], {}, [self.config, self.config], [dict(self.config, name="../escape")],
               [dict(self.config, ctx=True)], [dict(self.config, kv=0)], [dict(self.config, k="invalid")],
               [dict(self.config, extra="--flag")], [dict(self.config, env={"X": 1})],
               [dict(self.config, extra=["--host=0.0.0.0"])],
               [dict(self.config, extra=["--api_key", "other"])]]
        for matrix in bad:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                audit.validate_configs(matrix)
        audit.validate_configs([self.config, dict(self.config, name="auto", kv="auto")])

    def test_invalid_matrix_never_starts_server(self):
        model = self.root / "fixture.gguf"
        model.touch()
        configs = self.root / "configs.json"
        configs.write_text("[]", encoding="utf-8")
        with mock.patch.object(audit, "server") as server, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                audit.main(["probe", "--server", str(model), "--model", str(model),
                            "--configs", str(configs), "--out", str(self.root)])
        self.assertEqual(exc.exception.code, 2)
        server.assert_not_called()

    def test_calibration_deduplicates_shared_prefix(self):
        self.args.corpus = self.root / "corpus.txt"
        self.args.corpus.write_text("x" * 24000, encoding="utf-8")
        result = {"config": self.config}
        with mock.patch.object(audit, "request", side_effect=[{"tokens": [1, 2, 3]}, {"tokens": [1, 2, 4]}]):
            audit.prepare(self.args, "local", "key", result)
        self.assertEqual(result["calibration"]["nodes"], 4)
        directory = Path(result["calibration"]["trie"])
        nodes = (directory / "nodes-000000.bin").read_bytes()
        requests = (directory / "requests-000000.bin").read_bytes()
        self.assertEqual(nodes[:8], b"CLTNOD01")
        self.assertEqual(list(struct.iter_unpack("<QiI", nodes[24:])),
                         [(0, 1, 0), (1, 2, 0), (2, 3, 0), (2, 4, 0)])
        self.assertEqual([record[:2] for record in struct.iter_unpack("<QIIQ", requests[24:])],
                         [(3, 3), (4, 3)])

    def test_empty_calibration_rejected(self):
        self.args.corpus = self.root / "empty.txt"
        self.args.corpus.write_text("  ", encoding="utf-8")
        with self.assertRaises(ValueError):
            audit.prepare(self.args, "local", "key", {"config": self.config})
        self.args.corpus.write_text("some text", encoding="utf-8")
        with mock.patch.object(audit, "request", return_value={"tokens": []}), self.assertRaises(ValueError):
            audit.prepare(self.args, "local", "key", {"config": self.config})


if __name__ == "__main__":
    unittest.main()
