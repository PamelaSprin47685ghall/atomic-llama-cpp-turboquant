#!/usr/bin/env python3
"""Summarize saved probes without mixing process-first and subsequent requests.

This reads JSON only: no model execution, deployment, or network access.
Comparisons require matching model contents and exact tokenized requests.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

_SPEC = importlib.util.spec_from_file_location("nanbeige_audit", Path(__file__).with_name("nanbeige-audit.py"))
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _stats(samples, phase):
    rates = [sample["timings"][phase + "_per_second"] for sample in samples]
    if not rates:
        return None
    tokens = sum(sample["timings"][phase + "_n"] for sample in samples)
    duration = math.fsum(sample["timings"][phase + "_ms"] for sample in samples)
    ordered = sorted(rates)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else ordered[n // 2 - 1] / 2 + ordered[n // 2] / 2
    return {"median_tps": median, "min_tps": min(rates), "max_tps": max(rates),
            "total_tokens": tokens, "total_ms": duration, "aggregate_tps": tokens * 1000.0 / duration}


def summarize(record):
    if not isinstance(record, dict):
        raise ValueError("probe result must be an object")
    errors = []
    notes = []
    checks = record.get("quality_checks", [])
    if not isinstance(checks, list) or any(not isinstance(check, dict) or check.get("passed") is not True
                                           for check in checks):
        errors.append("quality checks failed or are malformed")
        quality = "failed"
    else:
        quality = "smoke_passed" if checks else "not_tested"
    if record.get("status") != "completed":
        errors.append("source run did not complete: " + str(record.get("error", record.get("status"))))
    if record.get("artifacts_changed") is not False:
        errors.append("build stability was not established")
    before, after = record.get("artifacts_before"), record.get("artifacts_after")
    if not isinstance(before, dict) or not before or before != after or not all(_sha(v) for v in before.values()):
        errors.append("missing or inconsistent build fingerprints")
    if "server_exit_before_cleanup" not in record or record["server_exit_before_cleanup"] is not None:
        errors.append("server did not remain alive until runner cleanup")
    if record.get("final_health") != {"status": "ok"}:
        errors.append("missing final healthy response")

    model = record.get("model_before")
    model_verified = (isinstance(model, dict) and _sha(model.get("sha256")) and
                      model == record.get("model_after") and record.get("model_changed") is False)
    if not model_verified:
        if "model_before" in record or record.get("model_changed"):
            errors.append("model changed or its fingerprints are inconsistent")
        else:
            notes.append("legacy record has no model-content fingerprint; comparison disabled")

    workload = record.get("workload")
    request_verified = (isinstance(workload, dict) and type(workload.get("version")) is int and
                        workload["version"] == 1 and _sha(workload.get("request_sha256")))
    if not request_verified:
        if workload is not None:
            errors.append("malformed or unsupported workload identity")
        else:
            notes.append("legacy record has no request identity; counts below are observed, not verified against the request")
    raw = record.get("requests")
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        errors.append("missing or malformed request list")
        raw = []
    samples = [row for row in raw if isinstance(row.get("case"), str) and row["case"].startswith("prefill-decode-")]
    if not samples:
        errors.append("no throughput samples")
    if [row["case"] for row in samples] != ["prefill-decode-" + str(i) for i in range(len(samples))]:
        errors.append("throughput samples are missing, duplicated or out of order")

    counts = None
    if request_verified:
        counts = (workload.get("prompt_tokens"), workload.get("predict"))
        if type(workload.get("repeats")) is not int or workload["repeats"] != len(samples):
            errors.append("actual repeat count differs from the request")
    elif samples and isinstance(samples[0].get("timings"), dict):
        counts = (samples[0]["timings"].get("prompt_n"), samples[0]["timings"].get("predicted_n"))
    if counts is None or any(type(n) is not int or n <= 0 for n in counts):
        errors.append("invalid prompt/generation counts")
    else:
        for row in samples:
            try:
                _AUDIT.check_completion(row, *counts)
                if request_verified and row.get("request_sha256") != workload["request_sha256"]:
                    raise ValueError("request fingerprint differs from the workload")
                wall = row.get("wall_seconds")
                if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
                    raise ValueError("invalid wall-clock duration")
            except (RuntimeError, ValueError, TypeError, OverflowError) as exc:
                errors.append(row["case"] + ": " + str(exc))

    result = {"config": record.get("config"), "status": "invalid" if errors else "measured",
              "errors": errors, "notes": notes, "quality": quality,
              "model_sha256": model.get("sha256") if model_verified else None,
              "request_sha256": workload.get("request_sha256") if request_verified else None,
              "sample_count": len(samples), "subsequent_count": max(0, len(samples) - 1),
              "first_request": None, "subsequent_requests": None,
              "startup_seconds": record.get("startup_seconds"),
              "sampled_device_peak_mib": record.get("peak_gpu_mib"),
              "production_approved": False}
    if errors:
        return result  # Never publish attractive throughput from a failed run.
    first = samples[0]
    result["first_request"] = {"prompt": _stats([first], "prompt"), "predicted": _stats([first], "predicted"),
                               "wall_seconds": first["wall_seconds"]}
    if len(samples) > 1:
        result["subsequent_requests"] = {"prompt": _stats(samples[1:], "prompt"),
                                          "predicted": _stats(samples[1:], "predicted"),
                                          "median_wall_seconds": statistics.median(row["wall_seconds"] for row in samples[1:])}
    else:
        notes.append("one request cannot establish subsequent-request throughput")
    if not all(isinstance(row.get("content"), str) for row in samples):
        result["output_sha256"] = None
        notes.append("generated content missing; output agreement cannot be checked")
    else:
        result["output_sha256"] = [hashlib.sha256(row["content"].encode("utf-8")).hexdigest() for row in samples]
    return result


def compare(baseline, candidate, require_output_match=False):
    reasons = []
    if baseline["status"] != "measured" or candidate["status"] != "measured":
        reasons.append("one or both measurements are invalid")
    for field in ("model_sha256", "request_sha256"):
        if not baseline.get(field) or baseline.get(field) != candidate.get(field):
            reasons.append("missing or different " + field)
    if min(baseline["subsequent_count"], candidate["subsequent_count"]) < 2:
        reasons.append("at least two subsequent samples per run are required")
    outputs = (baseline.get("output_sha256"), candidate.get("output_sha256"))
    # Require every repetition to agree, not only a conveniently matching one.
    output_equal = None if not all(outputs) else len(set(outputs[0] + outputs[1])) == 1
    if require_output_match and output_equal is not True:
        reasons.append("generated outputs do not agree or are unavailable")
    result = {"status": "not_comparable" if reasons else "comparable", "reasons": reasons,
              "output_agreement": output_equal, "prompt_speedup": None, "decode_speedup": None,
              "baseline_config": baseline.get("config"), "candidate_config": candidate.get("config"),
              "production_approved": False}
    if not reasons:
        for phase, field in (("prompt", "prompt_speedup"), ("predicted", "decode_speedup")):
            result[field] = (candidate["subsequent_requests"][phase]["median_tps"] /
                             baseline["subsequent_requests"][phase]["median_tps"])
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _finite_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("nonfinite JSON number: " + token)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path, help="must be one of the input files")
    parser.add_argument("--require-output-match", action="store_true", help="exact-path regression gate, not a quality score")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.records]
    if len(paths) != len(set(paths)):
        parser.error("duplicate input files")
    baseline_path = args.baseline.resolve() if args.baseline else None
    if baseline_path is not None and (baseline_path not in paths or len(paths) < 2):
        parser.error("baseline must be an input file, with at least one candidate")
    if args.require_output_match and baseline_path is None:
        parser.error("--require-output-match requires --baseline")
    if args.out and args.out.resolve() in paths:
        parser.error("output must not overwrite input evidence")
    results = []
    failed = False
    for path in paths:
        try:
            data = path.read_bytes()
            record = json.loads(data, object_pairs_hook=_unique_object,
                                parse_float=_finite_float, parse_constant=_finite_float)
            row = summarize(record)
            row["source_sha256"] = hashlib.sha256(data).hexdigest()
        except (OSError, ValueError, RuntimeError, TypeError, OverflowError) as exc:
            row = {"status": "invalid", "errors": [str(exc)]}
        row["source"] = str(path)
        failed |= row["status"] == "invalid"
        results.append(row)
    report = {"schema_version": 1, "records": results, "comparisons": [],
              "notes": ["Process-first is not proof of a cold driver or OS cache.",
                        "Subsequent-request medians exclude the first request, never errors.",
                        "Ratios are descriptive; hardware, driver, load and intentional config differences still require control.",
                        "Memory is a sampled device-wide peak, not proof of GPU-only execution.",
                        "Smoke answers or matching outputs do not establish production quality."]}
    if baseline_path is not None and not failed:
        baseline = results[paths.index(baseline_path)]
        for path, candidate in zip(paths, results):
            if path == baseline_path:
                continue
            comparison = compare(baseline, candidate, args.require_output_match)
            comparison["candidate_source"] = str(path)
            failed |= comparison["status"] != "comparable"
            report["comparisons"].append(comparison)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        try:
            args.out.write_text(text, encoding="utf-8")
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        print(text, end="")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
