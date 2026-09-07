#!/usr/bin/env python3
"""Pure offline checks for probe reports; deliberately no GPU or model required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "nanbeige_report", Path(__file__).resolve().parents[1] / "scripts" / "nanbeige-report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture():
    rows = []
    for i, ms in enumerate((20000.0, 400.0, 500.0)):
        rows.append({"case": "prefill-decode-" + str(i), "request_sha256": "b" * 64,
                     "content": "test answer", "wall_seconds": ms / 1000.0 + 0.2,
                     "tokens_predicted": 8, "truncated": False, "timings": {
                         "cache_n": 0, "prompt_n": 768, "predicted_n": 8,
                         "prompt_ms": ms, "predicted_ms": 100.0,
                         "prompt_per_second": 768000.0 / ms, "predicted_per_second": 80.0}})
    return {"status": "completed", "config": {"name": "fixture", "k": "turbo4", "v": "turbo2"},
            "artifacts_before": {"server": "a" * 64}, "artifacts_after": {"server": "a" * 64},
            "artifacts_changed": False, "server_exit_before_cleanup": None,
            "final_health": {"status": "ok"}, "quality_checks": [],
            "model_before": {"sha256": "c" * 64, "size_bytes": 100},
            "model_after": {"sha256": "c" * 64, "size_bytes": 100}, "model_changed": False,
            "workload": {"version": 1, "request_sha256": "b" * 64, "prompt_tokens": 768,
                         "predict": 8, "repeats": 3}, "requests": rows}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, record):
        path = self.root / name
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_first_request_is_not_blended_into_subsequent(self):
        result = report.summarize(fixture())
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["first_request"]["prompt"]["median_tps"], 38.4)
        steady = result["subsequent_requests"]["prompt"]
        self.assertEqual(steady["median_tps"], 1728.0)
        self.assertAlmostEqual(steady["aggregate_tps"], 1536000.0 / 900)
        self.assertEqual(result["subsequent_count"], 2)
        self.assertEqual(result["quality"], "not_tested")
        self.assertFalse(result["production_approved"])

    def test_one_request_never_implies_steady_state(self):
        record = fixture()
        record["requests"] = record["requests"][:1]
        record["workload"]["repeats"] = 1
        result = report.summarize(record)
        self.assertEqual(result["status"], "measured")
        self.assertIsNone(result["subsequent_requests"])
        self.assertEqual(report.compare(result, result)["status"], "not_comparable")

    def test_failed_fast_run_is_not_ranked(self):
        record = fixture()
        record.update(status="failed", error="late request crashed")
        result = report.summarize(record)
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["first_request"])
        self.assertIsNone(result["subsequent_requests"])

    def test_cache_truncation_count_and_rate_fail_closed(self):
        for update in ({"cache_n": 64}, {"prompt_n": 704}, {"prompt_per_second": 99999.0},
                       {"predicted_n": 7}, {"prompt_n": True}, {"prompt_ms": float("nan")}):
            record = fixture()
            record["requests"][1]["timings"].update(update)
            with self.subTest(update=update):
                self.assertEqual(report.summarize(record)["status"], "invalid")
        record = fixture()
        record["requests"][0]["truncated"] = True
        self.assertEqual(report.summarize(record)["status"], "invalid")

    def test_stale_and_crashed_evidence_is_invalid(self):
        changes = [{"artifacts_changed": True}, {"artifacts_after": {}},
                   {"artifacts_changed": 0}, {"model_changed": True},
                   {"server_exit_before_cleanup": 0}, {"server_exit_before_cleanup": -11},
                   {"final_health": {}}, {"quality_checks": [{"passed": False}]}]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(report.summarize(dict(fixture(), **change))["status"], "invalid")

    def test_missing_duplicate_and_reordered_samples(self):
        for indices in ((0, 2), (0, 1, 1), (1, 0, 2)):
            record = fixture()
            record["requests"] = [record["requests"][i] for i in indices]
            self.assertEqual(report.summarize(record)["status"], "invalid")

    def test_legacy_summary_never_asserts_comparability(self):
        record = fixture()
        for field in ("workload", "model_before", "model_after", "model_changed"):
            del record[field]
        result = report.summarize(record)
        self.assertEqual(result["status"], "measured")
        self.assertGreaterEqual(len(result["notes"]), 2)
        comparison = report.compare(result, result)
        self.assertEqual(comparison["status"], "not_comparable")
        self.assertIsNone(comparison["prompt_speedup"])

    def test_model_or_request_mismatch_disables_comparison(self):
        base = report.summarize(fixture())
        for field in ("model_sha256", "request_sha256"):
            other = dict(base, **{field: "d" * 64})
            self.assertEqual(report.compare(base, other)["status"], "not_comparable")
        record = fixture()
        record["requests"][1]["request_sha256"] = "d" * 64
        self.assertEqual(report.summarize(record)["status"], "invalid")

    def test_speedup_and_optional_exact_output_gate(self):
        base = report.summarize(fixture())
        record = fixture()
        for sample in record["requests"]:
            sample["timings"]["prompt_ms"] *= 0.5
            sample["timings"]["prompt_per_second"] *= 2
        candidate = report.summarize(record)
        result = report.compare(base, candidate, True)
        self.assertEqual(result["status"], "comparable")
        self.assertEqual(result["prompt_speedup"], 2)
        self.assertEqual(result["decode_speedup"], 1)
        self.assertTrue(result["output_agreement"])
        record["requests"][2]["content"] = "different answer"
        candidate = report.summarize(record)
        self.assertEqual(report.compare(base, candidate)["status"], "comparable")
        result = report.compare(base, candidate, True)
        self.assertEqual(result["status"], "not_comparable")
        self.assertIsNone(result["prompt_speedup"])

    def test_missing_output_is_not_equality(self):
        record = fixture()
        del record["requests"][0]["content"]
        candidate = report.summarize(record)
        self.assertIsNone(report.compare(candidate, candidate)["output_agreement"])
        self.assertEqual(report.compare(candidate, candidate, True)["status"], "not_comparable")

    def test_cli_saved_json_and_status(self):
        baseline = self.write("base.json", fixture())
        candidate = self.write("candidate.json", fixture())
        output = self.root / "report.json"
        args = [str(baseline), str(candidate), "--baseline", str(baseline), "--out", str(output)]
        self.assertEqual(report.main(args), 0)
        saved = json.loads(output.read_text())
        self.assertEqual(saved["comparisons"][0]["status"], "comparable")
        self.assertEqual(len(saved["records"][0]["source_sha256"]), 64)
        record = fixture()
        record["quality_checks"] = [{"passed": False}]
        self.write("candidate.json", record)
        self.assertEqual(report.main(args), 1)

    def test_cli_rejects_duplicate_json_keys_and_missing_files(self):
        path = self.root / "duplicate.json"
        path.write_text('{"status":"failed", "status":"completed"}')
        for source in (path, self.root / "absent.json"):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(report.main([str(source)]), 1)
            saved = json.loads(output.getvalue())
            self.assertEqual(saved["records"][0]["status"], "invalid")

    def test_cli_rejects_nonfinite_json_even_in_ancillary_fields(self):
        for literal in ("NaN", "Infinity", "-Infinity", "1e309"):
            path = self.root / "nonfinite.json"
            data = fixture()
            data["startup_seconds"] = "NONFINITE"
            path.write_text(json.dumps(data).replace('"NONFINITE"', literal))
            with self.subTest(literal=literal), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(report.main([str(path)]), 1)
            self.assertEqual(json.loads(output.getvalue())["records"][0]["status"], "invalid")

    def test_unknown_workload_version_is_not_inferred(self):
        for version in (2, True, "1"):
            data = fixture()
            data["workload"]["version"] = version
            self.assertEqual(report.summarize(data)["status"], "invalid")

    def test_cli_cannot_overwrite_input_or_compare_one_run(self):
        path = self.write("probe.json", fixture())
        before = path.read_bytes()
        for extra in (["--out", str(path)], ["--baseline", str(path)], ["--require-output-match"], [str(path)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                report.main([str(path)] + extra)
            self.assertEqual(exc.exception.code, 2)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
