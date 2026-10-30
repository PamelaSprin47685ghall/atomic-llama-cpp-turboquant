#!/usr/bin/env python3
"""Qualify completed public GGML operator cases using native Vulkan dispatch logs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


BEGIN = re.compile(r"^TQ_MMVQ_BEGIN ([A-Za-z0-9_-]+) (\d+)$")
END = re.compile(r"^TQ_MMVQ_END ([A-Za-z0-9_-]+) (\d+) ok$")
DISPATCH = re.compile(r"^ggml_vk_dispatch_pipeline\(([^,\s]+),")
SCALE_PIPELINE = "scale_f32"
TYPES = ("q4_k", "q4_0", "iq4_nl")
COLUMNS = (1, 2, 8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def read_events(lines):
    events = [json.loads(line, parse_constant=reject_constant) for line in lines if line.strip()]
    require(len(events) == 11, "expected config, nine cases, and completion")
    require(all(isinstance(event, dict) for event in events), "events must be objects")
    config, *cases, complete = events
    require(config.get("event") == "config" and config.get("schema") == 1, "invalid config")
    require(complete.get("event") == "complete" and complete.get("status") == "ok", "missing successful completion")
    require(config.get("width") == 2048 and config.get("rows") == 1025, "unqualified fixture dimensions")
    samples = config.get("samples")
    require(type(samples) is int and 1 <= samples <= 101, "invalid warm sample count")
    require(complete.get("cases") == 9 and complete.get("completed_graphs") == 9 * (samples + 1),
            "completion count mismatch")
    found = set()
    by_id = {}
    for case in cases:
        require(case.get("event") == "case", "unexpected event inside cases")
        kind = case.get("type", "").lower()
        columns = case.get("columns")
        require(kind in TYPES and type(columns) is int and columns in COLUMNS, "unexpected weight type or batch")
        require((kind, columns) not in found, "duplicate type/batch case")
        found.add((kind, columns))
        case_id = case.get("id")
        require(isinstance(case_id, str) and case_id.lower() == f"{kind}-b{columns}", "invalid case identity")
        require(case_id not in by_id, "duplicate case ID")
        by_id[case_id] = case
        require(case.get("completed_graphs") == samples + 1, "case graph count mismatch")
        require(case.get("checked_outputs") == 1025 * columns * (samples + 1), "case output count mismatch")
        require(type(case.get("f32_valid")) is bool and type(case.get("q8_valid")) is bool, "invalid reference results")
        require(case["f32_valid"] or case["q8_valid"], "no independent reference passed")
        timings = case.get("warm_us")
        require(isinstance(timings, list) and len(timings) == samples, "warm timing count mismatch")
        for key in ("max_abs_error", "max_f32_bound_ratio", "max_q8_bound_ratio", "max_raw_q8_bound",
                    "first_column_shape_difference", "first_compute_us", "warm_median_us", "warm_p25_us", "warm_p75_us"):
            value = case.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"invalid {key}")
        require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in timings),
                "invalid warm timing")
        require(case["f32_valid"] == (case["max_f32_bound_ratio"] <= 1), "F32 validity/ratio mismatch")
        require(case["q8_valid"] == (case["max_q8_bound_ratio"] <= 1), "Q8 validity/ratio mismatch")
    require(found == {(kind, columns) for kind in TYPES for columns in COLUMNS}, "incomplete case matrix")
    controls = config.get("env")
    require(isinstance(controls, dict), "missing control environment")
    disable = controls.get("GGML_VK_DISABLE_MMVQ") is not None
    force = controls.get("GGML_VK_FORCE_MMVQ") is not None
    policy = "disable" if disable else "force" if force else "auto"
    require(config.get("policy") == policy, "presence-based policy does not match the environment")
    return config, by_id


def read_dispatch(lines, cases, samples):
    active = None
    traces = {}
    outside = 0
    for number, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        begin = BEGIN.fullmatch(line)
        if begin:
            require(active is None, f"nested graph marker at line {number}")
            active = (begin[1], int(begin[2]))
            require(active[0] in cases and 0 <= active[1] <= samples, "unknown graph marker")
            require(active not in traces, "duplicate graph marker")
            traces[active] = []
            continue
        end = END.fullmatch(line)
        if end:
            require(active == (end[1], int(end[2])), f"unmatched successful completion at line {number}")
            active = None
            continue
        require(not line.startswith("TQ_MMVQ_"), f"malformed or failed graph marker at line {number}")
        dispatch = DISPATCH.match(line)
        if dispatch:
            if active is None:
                outside += 1
            else:
                traces[active].append(dispatch[1])
    require(active is None, "last graph did not complete")
    require(set(traces) == {(case_id, iteration) for case_id in cases for iteration in range(samples + 1)},
            "missing completed graph markers")
    return traces, outside


def qualify(event_lines, trace_lines, expectation, requested_policy):
    config, cases = read_events(event_lines)
    require(config["policy"] == requested_policy, "process policy differs from the requested control")
    require(not (config["policy"] == "disable" and expectation == "mmvq"),
            "disabled policy cannot qualify an MMVQ route")
    traces, outside = read_dispatch(trace_lines, cases, config["samples"])
    evidence = []
    for case_id, case in cases.items():
        kind = case["type"].lower()
        use_mmvq = expectation == "mmvq" and kind != "iq4_nl"
        expected = f"mul_mat_vec_{kind}_{'q8_1' if use_mmvq else 'f32'}_f32"
        reference_field = "q8_valid" if use_mmvq else "f32_valid"
        require(case[reference_field], f"{case_id}: actual pipeline's independent reference failed")
        counts = {}
        for iteration in range(config["samples"] + 1):
            names = traces[(case_id, iteration)]
            matrix = [name for name in names if name.startswith(("mul_mat", "dequantize_mul_mat"))]
            require(matrix == [expected], f"{case_id} iteration {iteration}: matrix dispatch {matrix}, expected {expected}")
            quantizers = [name for name in names if "quantize" in name and "q8_1" in name]
            require(len(quantizers) == int(use_mmvq),
                    f"{case_id} iteration {iteration}: activation quantization count {len(quantizers)}")
            scales = [name for name in names if name.startswith("scale")]
            require(scales == [SCALE_PIPELINE],
                    f"{case_id} iteration {iteration}: SCALE dispatch {scales}, expected one {SCALE_PIPELINE}")
            scale_index = names.index(SCALE_PIPELINE)
            matrix_index = names.index(expected)
            if use_mmvq:
                require(scale_index < names.index(quantizers[0]) < matrix_index,
                        f"{case_id} iteration {iteration}: expected producer order SCALE, quantize, matrix")
            else:
                require(scale_index < matrix_index,
                        f"{case_id} iteration {iteration}: expected producer order SCALE, matrix")
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        evidence.append({"id": case_id, "type": kind, "columns": case["columns"],
                         "route": "mmvq_q8_1" if use_mmvq else "native_quantized_weight_f32_activation",
                         "completed_graphs": config["samples"] + 1, "dispatch_counts": counts,
                         "reference": reference_field, "max_abs_error": case["max_abs_error"]})
    return {"status": "PASS", "scope": "public GGML operator fixture; no full-model dispatch or speed claim",
            "policy": config["policy"], "device": config["device"], "description": config["description"],
            "device_id": config["device_id"], "width": config["width"], "rows": config["rows"],
            "cases": evidence, "non_case_dispatch_records": outside,
            "timing": "DEBUG-instrumented graph times are diagnostic; use a separate normal build for timing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--expect", required=True, choices=("mmvq", "f32"))
    parser.add_argument("--policy", required=True, choices=("auto", "force", "disable"))
    args = parser.parse_args()
    try:
        events = args.events.read_bytes()
        trace = args.stderr.read_bytes()
        result = qualify(events.decode("utf-8").splitlines(), trace.decode("utf-8").splitlines(),
                         args.expect, args.policy)
        result["inputs"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(contents).hexdigest()}
                            for path, contents in ((args.events, events), (args.stderr, trace))]
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
