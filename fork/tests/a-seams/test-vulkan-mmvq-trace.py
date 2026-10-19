#!/usr/bin/env python3
"""Test the dispatch evidence gate with synthetic logs, without claiming inference."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("mmvq_trace", Path(__file__).with_name("check-vulkan-mmvq.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(route="mmvq", policy="auto"):
    env = {"GGML_VK_DISABLE_MMVQ": "1" if policy == "disable" else None,
           "GGML_VK_FORCE_MMVQ": "1" if policy == "force" else None}
    events = [{"event": "config", "schema": 1, "width": 2048, "rows": 1025, "samples": 1,
               "device": "Vulkan0", "description": "synthetic", "device_id": None, "policy": policy, "env": env}]
    trace = []
    for kind in MODULE.TYPES:
        for columns in MODULE.COLUMNS:
            case_id = f"{kind}-b{columns}"
            events.append({"event": "case", "id": case_id, "type": kind, "columns": columns,
                           "completed_graphs": 2, "checked_outputs": 1025 * columns * 2,
                           "f32_valid": True, "q8_valid": True, "warm_us": [1], "max_abs_error": 0,
                           "max_f32_bound_ratio": 0, "max_q8_bound_ratio": 0, "max_raw_q8_bound": 1,
                           "first_column_shape_difference": 0, "first_compute_us": 2,
                           "warm_median_us": 1, "warm_p25_us": 1, "warm_p75_us": 1})
            mmvq = route == "mmvq" and kind != "iq4_nl"
            for iteration in range(2):
                trace.append(f"TQ_MMVQ_BEGIN {case_id} {iteration}")
                trace.append("ggml_vk_dispatch_pipeline(scale_f32, {}, (1,1,1))")
                if mmvq:
                    trace.append("ggml_vk_dispatch_pipeline(quantize_q8_1_x4, {}, (1,1,1))")
                trace.append(f"ggml_vk_dispatch_pipeline(mul_mat_vec_{kind}_{'q8_1' if mmvq else 'f32'}_f32, {{}}, (1,1,1))")
                trace.append(f"TQ_MMVQ_END {case_id} {iteration} ok")
    events.append({"event": "complete", "status": "ok", "cases": 9, "completed_graphs": 18})
    return events, trace


def qualify(events, trace, route="mmvq", policy="auto"):
    return MODULE.qualify([json.dumps(event) for event in events], trace, route, policy)


class TraceTests(unittest.TestCase):
    def test_real_dispatch_records_and_completed_outputs_required(self):
        events, trace = fixture()
        receipt = qualify(events, trace)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len(receipt["cases"]), 9)
        self.assertTrue(all(case["route"] == "native_quantized_weight_f32_activation"
                            for case in receipt["cases"] if case["type"] == "iq4_nl"))

    def test_disable_control_requires_f32_route(self):
        events, trace = fixture("f32", "disable")
        self.assertEqual(qualify(events, trace, "f32", "disable")["status"], "PASS")

    def test_pipeline_creation_is_not_dispatch(self):
        events, trace = fixture()
        trace = [line.replace("ggml_vk_dispatch_pipeline(", "ggml_vk_create_pipeline(") for line in trace]
        with self.assertRaisesRegex(ValueError, "matrix dispatch"):
            qualify(events, trace)

    def test_missing_warm_activation_quantization_fails(self):
        events, trace = fixture()
        index = trace.index("TQ_MMVQ_BEGIN q4_k-b1 1")
        trace.pop(index + 2)
        with self.assertRaisesRegex(ValueError, "activation quantization count"):
            qualify(events, trace)

    def test_quantization_must_follow_the_single_activation_write(self):
        events, trace = fixture()
        index = trace.index("TQ_MMVQ_BEGIN q4_k-b1 1")
        reordered = trace.copy()
        reordered[index + 1], reordered[index + 2] = reordered[index + 2], reordered[index + 1]
        with self.assertRaisesRegex(ValueError, "expected producer order"):
            qualify(events, reordered)
        duplicate = trace.copy()
        duplicate.insert(index + 1, duplicate[index + 1])
        with self.assertRaisesRegex(ValueError, "expected one scale_f32"):
            qualify(events, duplicate)
        events, trace = fixture("f32", "disable")
        index = trace.index("TQ_MMVQ_BEGIN q4_k-b1 1")
        trace[index + 1], trace[index + 2] = trace[index + 2], trace[index + 1]
        with self.assertRaisesRegex(ValueError, "expected producer order"):
            qualify(events, trace, "f32", "disable")

    def test_disabled_policy_cannot_be_overridden_by_mmvq_expectation(self):
        events, trace = fixture("mmvq", "disable")
        with self.assertRaisesRegex(ValueError, "disabled policy cannot qualify"):
            qualify(events, trace, "mmvq", "disable")
        events[0]["env"]["GGML_VK_FORCE_MMVQ"] = "1"
        with self.assertRaisesRegex(ValueError, "disabled policy cannot qualify"):
            qualify(events, trace, "mmvq", "disable")

    def test_iq4_nl_cannot_claim_q8_pipeline(self):
        events, trace = fixture()
        trace = [line.replace("mul_mat_vec_iq4_nl_f32_f32", "mul_mat_vec_iq4_nl_q8_1_f32") for line in trace]
        with self.assertRaisesRegex(ValueError, "matrix dispatch"):
            qualify(events, trace)

    def test_truncated_or_duplicate_execution_fails(self):
        events, trace = fixture()
        with self.assertRaisesRegex(ValueError, "did not complete"):
            qualify(events, trace[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate graph marker"):
            qualify(events, trace + trace[:5])
        with self.assertRaisesRegex(ValueError, "expected config"):
            qualify(events[:-1], trace)

    def test_dispatch_selects_the_matching_numerical_gate(self):
        events, trace = fixture()
        events[1]["q8_valid"] = False
        events[1]["max_q8_bound_ratio"] = 2
        with self.assertRaisesRegex(ValueError, "actual pipeline's independent reference failed"):
            qualify(events, trace)

    def test_presence_based_control_and_invalid_numbers(self):
        events, trace = fixture()
        changed = copy.deepcopy(events)
        changed[0]["env"]["GGML_VK_DISABLE_MMVQ"] = "0"
        with self.assertRaisesRegex(ValueError, "presence-based policy"):
            qualify(changed, trace)
        events[1]["warm_us"] = [float("nan")]
        with self.assertRaisesRegex(ValueError, "non-finite JSON"):
            qualify(events, trace)


if __name__ == "__main__":
    unittest.main()
