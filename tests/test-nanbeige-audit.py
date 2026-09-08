#!/usr/bin/env python3
"""Regression tests for the local probe runner; no GPU, model or network needed."""
import contextlib
import importlib.util
import io
import json
import os
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

    def response(self, answer="391", finish="stop", budget=2, speed=25.0,
                 cache_n=0, prompt_n=8, predicted_n=2, truncated=False,
                 prompt_ms=320.0):
        chats = iter([answer, "12", "-3, 0, 2, 7, 11"])

        def request(base, key, path, data=None, timeout=1):
            if path == "/tokenize":
                return {"tokens": list(range(16))}
            if path == "/completion":
                return {"tokens_predicted": budget, "truncated": truncated, "timings": {
                    "cache_n": cache_n, "prompt_n": prompt_n, "predicted_n": predicted_n,
                    "prompt_ms": prompt_ms, "predicted_ms": 80.0,
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
        self.assertEqual(result["timing_summary"]["repeated_requests"]["count"], 0)

    def test_first_request_is_not_mixed_into_repeated_rates(self):
        samples = [{"prompt_per_second": pp, "predicted_per_second": tg}
                   for pp, tg in [(30, 5), (2000, 70), (2200, 80), (2100, 75)]]
        summary = audit.summarize_timings(samples)
        self.assertEqual(summary["first_request"], samples[0])
        self.assertEqual(summary["repeated_requests"]["count"], 3)
        self.assertEqual(summary["repeated_requests"]["prompt_per_second"],
                         {"min": 2000, "median": 2100, "max": 2200})
        self.assertEqual(summary["repeated_requests"]["predicted_per_second"]["median"], 75)
        self.assertEqual(samples[0]["prompt_per_second"], 30)  # input evidence preserved

    def test_single_sample_has_no_invented_warm_rate(self):
        summary = audit.summarize_timings([{"prompt_per_second": 30, "predicted_per_second": 70}])
        self.assertEqual(summary["repeated_requests"],
                         {"count": 0, "prompt_per_second": None, "predicted_per_second": None})

    def test_invalid_timing_summaries(self):
        with self.assertRaises(ValueError):
            audit.summarize_timings([])
        for bad in (None, True, "123", float("nan"), float("inf"), 0, -1):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                audit.summarize_timings([{"prompt_per_second": bad, "predicted_per_second": 70}])

    def test_wrong_answer_and_cutoff_are_failures(self):
        for kwargs in ({"answer": "392"}, {"finish": "length"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RuntimeError, "quality checks failed"):
                self.run_probe(**kwargs)

    def test_incomplete_completion_and_nonfinite_timings(self):
        for kwargs in ({"budget": 1}, {"speed": float("nan")}, {"speed": float("inf")}, {"speed": 0}, {"speed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.run_probe(**kwargs)

    def test_cached_shortened_or_inconsistent_work_is_not_throughput(self):
        for change in ({"cache_n": 1}, {"prompt_n": 7}, {"predicted_n": 1},
                       {"truncated": True}, {"prompt_ms": 0}, {"prompt_ms": float("inf")},
                       {"speed": 25000.0}, {"cache_n": False}, {"prompt_n": 8.0}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.run_probe(**change)

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

    def test_flashprefill_requires_observed_plan_work(self):
        config = dict(self.config, require_flashprefill_plan=True)
        metrics = {"sparse_rows": 0, "dense_packed_rows": 384, "selected_blocks": 12,
                   "corrected_blocks": 0, "visible_tokens": 768, "exact_tokens": 768}

        def result_for(values):
            return {"metrics_text": "".join("llamacpp:flashprefill_" + k + "_total " + str(v) + "\n"
                                            for k, v in values.items())}

        result = result_for(metrics)
        audit.check_runtime_evidence(config, result)  # exact-all is still real plan work
        self.assertEqual(result["flashprefill_counters"], metrics)
        for change in ({"selected_blocks": 0}, {"dense_packed_rows": 0},
                       {"visible_tokens": 0}, {"exact_tokens": 769},
                       {"sparse_rows": -1}, {"visible_tokens": "NaN"},
                       {"selected_blocks": "Inf"}, {"selected_blocks": 0.5}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                audit.check_runtime_evidence(config, result_for(dict(metrics, **change)))
        with self.assertRaises(RuntimeError):
            audit.check_runtime_evidence(config, {})
        with self.assertRaises(RuntimeError):
            audit.check_runtime_evidence(config, {"metrics_text": result_for(metrics)["metrics_text"] * 2})
        with self.assertRaises(ValueError):
            audit.validate_configs([dict(self.config, require_flashprefill_plan="true")])

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

    def test_process_signal_preserved_before_cleanup(self):
        self.args.server = self.root / "server"
        self.args.server.touch()
        self.args.model = self.root / "model"
        self.args.model.touch()
        self.args.startup_timeout = 1
        for crashed in (False, True):
            with self.subTest(crashed=crashed):
                proc = mock.Mock(returncode=-11 if crashed else -15)
                proc.poll.side_effect = [None, -11 if crashed else None]
                result = {}
                with mock.patch.object(audit.socket, "socket") as sock, \
                        mock.patch.object(audit.subprocess, "Popen", return_value=proc), \
                        mock.patch.object(audit.threading, "Thread"), \
                        mock.patch.object(audit, "request", return_value={"status": "ok"}):
                    sock.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
                    with self.assertRaisesRegex(RuntimeError, "HTTP disconnected"):
                        with audit.server(self.args, self.config, result):
                            raise RuntimeError("HTTP disconnected")
                self.assertEqual(result["server_exit_before_cleanup"], -11 if crashed else None)
                self.assertEqual(result["server_returncode"], -11 if crashed else -15)
                self.assertEqual(result["server_terminated_by_runner"], not crashed)
                self.assertEqual(proc.terminate.call_count, 0 if crashed else 1)
                self.assertFalse(result["artifacts_changed"])

    def test_changed_library_fingerprint(self):
        binary = self.root / "server"
        binary.write_bytes(b"server")
        library = self.root / "libfixture.so.1"
        library.write_bytes(b"old")
        (self.root / "libfixture.so").symlink_to(library.name)
        before = audit.artifact_fingerprint(binary)
        self.assertEqual(len(before), 2)
        library.write_bytes(b"new")
        self.assertNotEqual(before, audit.artifact_fingerprint(binary))

    def test_build_preflight_rejects_stale_server_object(self):
        source = self.root / "source"
        build = self.root / "build"
        binary = build / "bin" / "llama-server"
        object_dir = build / "tools" / "server" / "CMakeFiles" / "server-context.dir"
        header = source / "tools" / "server" / "server-task.h"
        object_file = object_dir / "server-queue.cpp.o"
        dep_file = object_dir / "server-queue.cpp.o.d"
        binary.parent.mkdir(parents=True)
        object_dir.mkdir(parents=True)
        header.parent.mkdir(parents=True)
        binary.write_bytes(b"server")
        header.write_text("new layout", encoding="utf-8")
        object_file.write_bytes(b"old object")
        dep_file.write_text(f"server-queue.cpp.o: {header}\n", encoding="utf-8")
        (build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n", encoding="utf-8")
        os.utime(object_file, ns=(1_000_000_000, 1_000_000_000))
        os.utime(header, ns=(2_000_000_000, 2_000_000_000))

        report = audit.inspect_build_artifacts(binary)
        self.assertTrue(report["checked"])
        self.assertEqual(len(report["stale_objects"]), 1)
        self.assertIn("stale server object", report["errors"][0])

        os.utime(object_file, ns=(3_000_000_000, 3_000_000_000))
        self.assertEqual(audit.inspect_build_artifacts(binary)["errors"], [])

    def test_build_preflight_rejects_duplicate_static_archive_members(self):
        source = self.root / "source"
        build = self.root / "build"
        binary = build / "bin" / "llama-server"
        archive = build / "tools" / "server" / "libserver-context.a"
        (build / "tools" / "server" / "CMakeFiles").mkdir(parents=True)
        binary.parent.mkdir(parents=True)
        archive.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"server")
        source.mkdir()
        (build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n", encoding="utf-8")

        def member(name, payload):
            encoded = name.encode("ascii") + b"/"
            header = (encoded.ljust(16) + b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6)
                      + b"100644".ljust(8) + str(len(payload)).encode("ascii").ljust(10) + b"`\n")
            return header + payload + (b"\n" if len(payload) & 1 else b"")

        archive.write_bytes(b"!<arch>\n" + member("same.o", b"a") + member("same.o", b"b"))
        report = audit.inspect_build_artifacts(binary)
        self.assertEqual(report["duplicate_archive_members"][str(archive)], ["same.o"])
        self.assertIn("duplicate static archive members", report["errors"][0])

    def test_orphan_versioned_libraries_do_not_change_runtime_fingerprint(self):
        binary = self.root / "server"
        binary.write_bytes(b"server")
        current = self.root / "libfixture.so.2.0.7"
        current.write_bytes(b"current")
        (self.root / "libfixture.so").symlink_to(current.name)
        (self.root / "libfixture.so.2").symlink_to(current.name)
        orphan = self.root / "libfixture.so.1.0.0"
        orphan.write_bytes(b"old orphan")
        before = audit.artifact_fingerprint(binary)
        self.assertEqual(set(before), {str(binary.resolve()), str(current.resolve())})
        orphan.write_bytes(b"changed orphan")
        self.assertEqual(before, audit.artifact_fingerprint(binary))

    def test_switching_runtime_library_symlink_changes_fingerprint(self):
        binary = self.root / "server"
        binary.write_bytes(b"server")
        first = self.root / "libfixture.so.1.0.0"
        second = self.root / "libfixture.so.2.0.0"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        alias = self.root / "libfixture.so"
        alias.symlink_to(first.name)
        before = audit.artifact_fingerprint(binary)
        alias.unlink()
        alias.symlink_to(second.name)
        after = audit.artifact_fingerprint(binary)
        self.assertNotEqual(before, after)
        self.assertIn(str(second.resolve()), after)
        self.assertNotIn(str(first.resolve()), after)

    def test_model_identity_is_content_based_and_stable(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"weights")
        first = audit.model_fingerprint(model)
        self.assertEqual(first, audit.model_fingerprint(model))
        self.assertEqual(first["size_bytes"], 7)
        copied = self.root / "copy.gguf"
        copied.write_bytes(model.read_bytes())
        self.assertEqual(first["sha256"], audit.model_fingerprint(copied)["sha256"])
        model.write_bytes(b"changed")  # same size must still be detected
        self.assertNotEqual(first["sha256"], audit.model_fingerprint(model)["sha256"])

    def test_model_override_flags_are_rejected(self):
        for flag in ("-m", "--model", "--model-url", "-hf", "--hf-repo", "--huggingface-file"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                audit.validate_configs([dict(self.config, extra=[flag, "other-model"])])

    def test_model_changes_and_late_server_exit_fail_the_run(self):
        binary = self.root / "server"
        binary.touch()
        for evidence in ({"model_changed": True}, {"server_exit_before_cleanup": 0},
                         {"server_exit_before_cleanup": -11}):
            @contextlib.contextmanager
            def changed_server(args, config, result):
                result.update(evidence)
                yield "local", "key"

            with self.subTest(evidence=evidence), mock.patch.object(audit, "server", changed_server), \
                    mock.patch.object(audit, "probe"), contextlib.redirect_stdout(io.StringIO()):
                code = audit.main(["probe", "--server", str(binary), "--model", str(binary),
                                   "--out", str(self.root)])
            self.assertEqual(code, 1)
            saved = json.loads((self.root / "baseline.json").read_text())
            self.assertEqual(saved["status"], "failed")

    def test_changed_artifacts_cannot_pass(self):
        binary = self.root / "server"
        binary.touch()

        @contextlib.contextmanager
        def changed_server(args, config, result):
            result["artifacts_changed"] = True
            yield "local", "key"

        with mock.patch.object(audit, "server", changed_server), \
                mock.patch.object(audit, "probe"), contextlib.redirect_stdout(io.StringIO()):
            code = audit.main(["probe", "--server", str(binary), "--model", str(binary),
                               "--out", str(self.root)])
        self.assertEqual(code, 1)
        saved = json.loads((self.root / "baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertIn("artifacts changed", saved["error"])

    def test_pressure_gate_needs_real_drain(self):
        log = self.root / "server.log"
        config = dict(self.config, require_tri_drain=True, max_tri_score_ms=0)
        result = {"server_log": str(log)}
        for text in ("health ok", "TriAttention maintenance: before=384 after=128 freed=256 score_ms=0 pack_ms=1"):
            log.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "expected a real"):
                audit.check_runtime_evidence(config, result)
        text = "TriAttention drain: before=512 after=128 freed=384 score_ms=0.000 pack_ms=2.3"
        log.write_text(text, encoding="utf-8")
        audit.check_runtime_evidence(config, result)
        self.assertEqual(result["tri_events"][0]["freed"], 384)
        for changed in (text.replace("0.000", "2.3"), text.replace("384", "383"),
                        text.replace("0.000", "nan")):
            log.write_text(changed, encoding="utf-8")
            with self.assertRaises(RuntimeError):
                audit.check_runtime_evidence(config, result)

    def test_pressure_gate_can_require_real_scoring(self):
        log = self.root / "server.log"
        config = dict(self.config, require_tri_scoring=True)
        result = {"server_log": str(log)}
        floor_only = "TriAttention drain: before=512 after=128 freed=384 score_ms=0.000 pack_ms=2.3"
        log.write_text(floor_only, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "scored TriAttention drain"):
            audit.check_runtime_evidence(config, result)
        log.write_text(floor_only.replace("score_ms=0.000", "score_ms=2.300"), encoding="utf-8")
        audit.check_runtime_evidence(config, result)
        self.assertGreater(result["tri_events"][0]["score_ms"], 0)

    def test_pressure_gate_config_validation(self):
        for change in ({"require_tri_drain": "true"}, {"require_tri_scoring": "true"}, {"max_tri_score_ms": -1},
                       {"max_tri_score_ms": True}, {"max_tri_score_ms": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_configs([dict(self.config, **change)])
        audit.validate_configs([dict(self.config, require_tri_drain=True, require_tri_scoring=True)])


if __name__ == "__main__":
    unittest.main()
