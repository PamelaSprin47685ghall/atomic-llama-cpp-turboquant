#!/usr/bin/env python3
"""Mock HTTP tests for scripts/rerot-throughput-gate.py (C02 throughput gate).

Required test cases:
1. Test that a round-1 abort (hard_aborts=1 delta) with round-2 clean (hard_aborts=0)
   makes the gate FAIL (per-sample judgment catches the bad round).
2. Test that a missing required counter (e.g. rerot_completed_episode_total absent
   from /metrics) causes evidence-incomplete (exit 3), not silent pass with 0.
3. Test that parallel not-exercised (par_tok=0, par_sec=0) causes CHECK_NOT_EXERCISED
   for parallel_work_exercised_paired and an incomplete verdict.
4. Test that gauge delta of rerot_pens_allocated (before=5, after=0) does NOT satisfy
   multi_lane check (no gauge gating).
5. Test that route gate enabled without rerot_operator_route_hits in /metrics ->
   evidence-incomplete.
6. Test that valid round (all counters present, episode=1, no abort, fence=1,
   visibility OK, parallel exercised, ratio>=min_ratio) passes.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Load the gate script dynamically as a module
GATE_SCRIPT_PATH = Path(__file__).resolve().parent / "rerot-throughput-gate.py"
spec = importlib.util.spec_from_file_location("rerot_throughput_gate", str(GATE_SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {GATE_SCRIPT_PATH}")
gate_mod = importlib.util.module_from_spec(spec)
sys.modules["rerot_throughput_gate"] = gate_mod
spec.loader.exec_module(gate_mod)


class MockHTTPResponse:
    """Mock urllib response context manager."""

    def __init__(self, data: bytes, status: int = 200) -> None:
        self.data = data
        self.status = status

    def read(self) -> bytes:
        return self.data

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> MockHTTPResponse:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def make_metrics_text(metrics: dict[str, float]) -> str:
    lines = []
    for k, v in metrics.items():
        lines.append(f"llamacpp:{k} {v}")
    return "\n".join(lines) + "\n"


class ThroughputGateMockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.request_file = Path(self.temp_dir.name) / "request.json"
        self.output_file = Path(self.temp_dir.name) / "output.json"

        # Valid DAG request payload
        request_payload = {
            "model": "ornith-1.5",
            "messages": [
                {
                    "role": "user",
                    "content": "1. Task A 2. Task B multi-task lane dag",
                }
            ],
            "temperature": 0,
            "stream": False,
            "rerot": True,
            "rerot_trace": False,
        }
        self.request_file.write_text(json.dumps(request_payload), encoding="utf-8")

        # Base valid metrics dict
        self.base_metrics: dict[str, float] = {
            "rerot_completed_episode_total": 0.0,
            "rerot_completed_model_tokens": 0.0,
            "rerot_parallel_model_tokens": 0.0,
            "rerot_completed_episode_seconds": 0.0,
            "rerot_parallel_seconds": 0.0,
            "rerot_public_tokens": 0.0,
            "rerot_private_tokens": 0.0,
            "rerot_pending_tokens": 0.0,
            "rerot_hard_aborts": 0.0,
            "rerot_final_fences": 0.0,
            "rerot_frontier_rows": 0.0,
            "rerot_six_row_batches": 0.0,
            "rerot_parallel_peak_lanes_total": 0.0,
            # Gauges
            "rerot_people_capacity": 10.0,
            "rerot_people_resident": 2.0,
            "rerot_pens_capacity": 10.0,
            "rerot_pens_allocated": 2.0,
            "rerot_pens_running": 2.0,
            "rerot_batch_people": 2.0,
            "rerot_batch_pens": 2.0,
            "rerot_brain_bytes": 1000.0,
            "rerot_hand_bytes": 1000.0,
        }

        # Base serial chat response
        self.serial_chat_response = {
            "choices": [{"message": {"content": "Serial answer"}}],
            "timings": {
                "predicted_n": 100,
                "predicted_ms": 1000.0,  # 100 tok/s
                "prompt_ms": 50.0,
            },
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 100,
                "total_tokens": 120,
            },
        }

        # Base RERoT chat response
        self.rerot_chat_response = {
            "choices": [{"message": {"content": "RERoT answer"}}],
            "timings": {
                "predicted_n": 100,
                "predicted_ms": 500.0,
                "prompt_ms": 40.0,
            },
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 100,
                "total_tokens": 120,
                "rerot": {
                    "probe_tokens": 5,
                    "frame_tokens": 10,
                    "actual_replayed_tokens": 10,
                    "sampled_tokens": 100,
                },
            },
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create_mock_urlopen(
        self,
        metrics_sequence: list[dict[str, float]],
        serial_chat_resp: dict[str, Any] | None = None,
        rerot_chat_resp: dict[str, Any] | None = None,
    ):
        """Creates a mock urlopen returning health, metrics, and chat completions in order."""
        metrics_call_count = 0
        s_resp = serial_chat_resp or self.serial_chat_response
        r_resp = rerot_chat_resp or self.rerot_chat_response

        def mock_urlopen(req: Any, *args: Any, **kwargs: Any) -> MockHTTPResponse:
            nonlocal metrics_call_count
            url = req.full_url if hasattr(req, "full_url") else str(req)

            if url.endswith("/health"):
                return MockHTTPResponse(b"OK", status=200)
            elif url.endswith("/metrics"):
                idx = min(metrics_call_count, len(metrics_sequence) - 1)
                m_data = metrics_sequence[idx]
                metrics_call_count += 1
                return MockHTTPResponse(make_metrics_text(m_data).encode("utf-8"), status=200)
            elif url.endswith("/v1/chat/completions"):
                data = json.loads(req.data.decode("utf-8")) if hasattr(req, "data") and req.data else {}
                if data.get("rerot", False):
                    return MockHTTPResponse(json.dumps(r_resp).encode("utf-8"), status=200)
                else:
                    return MockHTTPResponse(json.dumps(s_resp).encode("utf-8"), status=200)
            elif url.endswith("/props"):
                return MockHTTPResponse(json.dumps({"build_info": "build: 4665"}).encode("utf-8"), status=200)
            raise ValueError(f"Unexpected URL in mock_urlopen: {url}")

        return mock_urlopen

    def run_gate(self, extra_args: list[str]) -> tuple[int, dict[str, Any]]:
        argv = [
            "rerot-throughput-gate.py",
            "--base-url", "http://127.0.0.1:8080",
            "--api-key", "test-key",
            "--request", str(self.request_file),
            "--output", str(self.output_file),
            *extra_args,
        ]
        with patch.object(sys, "argv", argv):
            exit_code = gate_mod.main()
        output_data = {}
        if self.output_file.exists():
            output_data = json.loads(self.output_file.read_text(encoding="utf-8"))
        return exit_code, output_data

    def test_round1_abort_round2_clean_fails_gate(self) -> None:
        """1. Test that a round-1 abort (hard_aborts=1 delta) with round-2 clean (hard_aborts=0)
        makes the gate FAIL (per-sample judgment catches the bad round)."""
        # Round 0: serial first, then rerot:
        #   before rerot: hard_aborts=0
        #   after rerot: hard_aborts=1 (delta=1 -> FAIL)
        # Round 1: rerot first, then serial:
        #   before rerot: hard_aborts=1
        #   after rerot: hard_aborts=1 (delta=0 -> CLEAN)

        m0 = dict(self.base_metrics)

        # After round 0 rerot
        m1 = dict(m0)
        m1["rerot_completed_episode_total"] += 1.0
        m1["rerot_completed_model_tokens"] += 100.0
        m1["rerot_public_tokens"] += 100.0
        m1["rerot_completed_episode_seconds"] += 0.5  # 200 tok/s
        m1["rerot_parallel_model_tokens"] += 80.0
        m1["rerot_parallel_seconds"] += 0.3  # 266 tok/s
        m1["rerot_final_fences"] += 1.0
        m1["rerot_parallel_peak_lanes_total"] += 2.0
        m1["rerot_hard_aborts"] += 1.0  # Abort happened in round 0!

        # Round 1 before rerot (same as m1)
        m2 = dict(m1)

        # After round 1 rerot (clean, no new aborts)
        m3 = dict(m2)
        m3["rerot_completed_episode_total"] += 1.0
        m3["rerot_completed_model_tokens"] += 100.0
        m3["rerot_public_tokens"] += 100.0
        m3["rerot_completed_episode_seconds"] += 0.5
        m3["rerot_parallel_model_tokens"] += 80.0
        m3["rerot_parallel_seconds"] += 0.3
        m3["rerot_final_fences"] += 1.0
        m3["rerot_parallel_peak_lanes_total"] += 2.0
        # hard_aborts not incremented!

        metrics_seq = [m0, m1, m2, m3]
        mock_urlopen = self.create_mock_urlopen(metrics_seq)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate(["--rounds", "2", "--min-ratio", "1.0"])

        self.assertEqual(exit_code, gate_mod.EXIT_FAIL)
        self.assertEqual(output_data["status"], "fail")
        self.assertFalse(output_data["passed"])
        self.assertEqual(output_data["round_checks"][0]["status"], "fail")
        self.assertEqual(output_data["round_checks"][0]["checks"]["no_hard_abort"], gate_mod.CHECK_FAIL)
        self.assertEqual(output_data["round_checks"][1]["checks"]["no_hard_abort"], gate_mod.CHECK_OK)
        self.assertEqual(output_data["verdict_checks"]["per_round_all_ok"], gate_mod.CHECK_FAIL)

    def test_missing_required_counter_causes_evidence_incomplete(self) -> None:
        """2. Test that a missing required counter (e.g. rerot_completed_episode_total absent
        from /metrics) causes evidence-incomplete (exit 3), not silent pass with 0."""
        m0 = dict(self.base_metrics)
        del m0["rerot_completed_episode_total"]

        m1 = dict(m0)
        m1["rerot_completed_model_tokens"] += 100.0
        m1["rerot_public_tokens"] += 100.0
        m1["rerot_completed_episode_seconds"] += 0.5
        m1["rerot_parallel_model_tokens"] += 80.0
        m1["rerot_parallel_seconds"] += 0.3
        m1["rerot_final_fences"] += 1.0
        m1["rerot_parallel_peak_lanes_total"] += 2.0

        metrics_seq = [m0, m1]
        mock_urlopen = self.create_mock_urlopen(metrics_seq)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate(["--rounds", "1", "--min-ratio", "1.0"])

        self.assertEqual(exit_code, gate_mod.EXIT_EVIDENCE_INCOMPLETE)
        self.assertEqual(output_data["status"], "evidence_incomplete")
        self.assertFalse(output_data["passed"])
        self.assertIn("rerot_completed_episode_total", output_data["missing_metrics"])
        self.assertEqual(
            output_data["round_checks"][0]["checks"]["one_completed_episode"],
            gate_mod.CHECK_EVIDENCE_INCOMPLETE,
        )

    def test_parallel_not_exercised_causes_check_not_exercised(self) -> None:
        """Parallel absence must not pass on aggregate throughput alone."""
        m0 = dict(self.base_metrics)

        m1 = dict(m0)
        m1["rerot_completed_episode_total"] += 1.0
        m1["rerot_completed_model_tokens"] += 100.0
        m1["rerot_public_tokens"] += 100.0
        m1["rerot_completed_episode_seconds"] += 0.5  # 200 tok/s
        # parallel not exercised:
        m1["rerot_parallel_model_tokens"] += 0.0
        m1["rerot_parallel_seconds"] += 0.0
        m1["rerot_final_fences"] += 1.0
        m1["rerot_parallel_peak_lanes_total"] += 2.0  # lanes admitted, but no parallel work

        metrics_seq = [m0, m1]
        mock_urlopen = self.create_mock_urlopen(metrics_seq)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate(["--rounds", "1", "--min-ratio", "1.0"])

        # No model-token work on parallel pens is incomplete, even if the
        # aggregate sampled-token rate would pass against serial.
        self.assertEqual(exit_code, gate_mod.EXIT_EVIDENCE_INCOMPLETE)
        self.assertEqual(output_data["status"], "evidence_incomplete")
        self.assertEqual(
            output_data["verdict_checks"]["parallel_work_exercised_paired"],
            gate_mod.CHECK_NOT_EXERCISED,
        )
        self.assertEqual(
            output_data["round_checks"][0]["checks"]["multi_lane_parallel_work"],
            gate_mod.CHECK_NOT_EXERCISED,
        )
        self.assertIsNotNone(output_data["not_exercised"])
        self.assertIn("parallel work not exercised", output_data["not_exercised"])

    def test_gauge_delta_does_not_satisfy_multilane_check(self) -> None:
        """4. Test that gauge delta of rerot_pens_allocated (before=5, after=0) does NOT satisfy
        multi_lane check (no gauge gating)."""
        m0 = dict(self.base_metrics)
        m0["rerot_pens_allocated"] = 5.0

        m1 = dict(m0)
        m1["rerot_pens_allocated"] = 0.0  # Gauge changed!
        m1["rerot_completed_episode_total"] += 1.0
        m1["rerot_completed_model_tokens"] += 100.0
        m1["rerot_public_tokens"] += 100.0
        m1["rerot_completed_episode_seconds"] += 0.5
        # Multi-lane counters are 0:
        m1["rerot_parallel_model_tokens"] += 0.0
        m1["rerot_parallel_seconds"] += 0.0
        m1["rerot_final_fences"] += 1.0

        metrics_seq = [m0, m1]
        mock_urlopen = self.create_mock_urlopen(metrics_seq)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate(["--rounds", "1", "--min-ratio", "1.0"])

        # Despite gauge delta of pens_allocated being -5, multi_lane_parallel_work MUST NOT pass.
        # It should be not_exercised because counters were 0.
        self.assertEqual(
            output_data["round_checks"][0]["checks"]["multi_lane_parallel_work"],
            gate_mod.CHECK_NOT_EXERCISED,
        )
        self.assertEqual(output_data["rerot"]["samples"][0]["gauge_before"]["rerot_pens_allocated"], 5.0)
        self.assertEqual(output_data["rerot"]["samples"][0]["gauge_after"]["rerot_pens_allocated"], 0.0)

    def test_route_gate_enabled_without_metric_causes_evidence_incomplete(self) -> None:
        """5. Test that route gate enabled without rerot_operator_route_hits in /metrics ->
        evidence-incomplete."""
        m0 = dict(self.base_metrics)
        # Ensure rerot_operator_route_hits is NOT in metrics
        m0.pop("rerot_operator_route_hits", None)

        m1 = dict(m0)
        m1["rerot_completed_episode_total"] += 1.0
        m1["rerot_completed_model_tokens"] += 100.0
        m1["rerot_public_tokens"] += 100.0
        m1["rerot_completed_episode_seconds"] += 0.5
        m1["rerot_parallel_model_tokens"] += 80.0
        m1["rerot_parallel_seconds"] += 0.3
        m1["rerot_final_fences"] += 1.0
        m1["rerot_parallel_peak_lanes_total"] += 2.0

        metrics_seq = [m0, m1]
        mock_urlopen = self.create_mock_urlopen(metrics_seq)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate([
                "--rounds", "1",
                "--min-ratio", "1.0",
                "--enable-operator-route-gate",
            ])

        self.assertEqual(exit_code, gate_mod.EXIT_EVIDENCE_INCOMPLETE)
        self.assertEqual(output_data["status"], "evidence_incomplete")
        self.assertFalse(output_data["passed"])
        self.assertEqual(
            output_data["round_checks"][0]["checks"]["operator_route_hit"],
            gate_mod.CHECK_EVIDENCE_INCOMPLETE,
        )
        self.assertIn("rerot_operator_route_hits", output_data["missing_metrics"])

    def test_forced_frames_cannot_fake_useful_throughput(self) -> None:
        m0 = dict(self.base_metrics)
        m1 = dict(m0)
        m1["rerot_completed_episode_total"] += 1
        m1["rerot_completed_model_tokens"] += 300
        m1["rerot_public_tokens"] += 100
        m1["rerot_private_tokens"] += 100
        m1["rerot_pending_tokens"] += 100
        m1["rerot_completed_episode_seconds"] += 0.5
        m1["rerot_parallel_model_tokens"] += 240
        m1["rerot_parallel_seconds"] += 0.3
        m1["rerot_final_fences"] += 1
        m1["rerot_parallel_peak_lanes_total"] += 6

        rerot_resp = {
            **self.rerot_chat_response,
            "timings": {**self.rerot_chat_response["timings"], "predicted_ms": 2000.0},
        }
        mock_urlopen = self.create_mock_urlopen([m0, m1], rerot_chat_resp=rerot_resp)
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate(["--rounds", "1", "--min-ratio", "1.0"])

        self.assertEqual(exit_code, gate_mod.EXIT_FAIL)
        self.assertEqual(output_data["pairs"][0]["aggregate_ratio"], 0.5)
        self.assertEqual(output_data["verdict_checks"]["aggregate_no_slower_than_serial_paired"], gate_mod.CHECK_FAIL)
        self.assertGreater(output_data["rerot"]["samples"][0]["model_tokens_per_second"],
                           output_data["serial"]["tokens_per_second"])

    def test_six_pen_gate_rejects_two_active_lanes(self) -> None:
        m0 = dict(self.base_metrics)
        m1 = dict(m0)
        m1["rerot_completed_episode_total"] += 1
        m1["rerot_completed_model_tokens"] += 200
        m1["rerot_public_tokens"] += 100
        m1["rerot_private_tokens"] += 50
        m1["rerot_pending_tokens"] += 50
        m1["rerot_completed_episode_seconds"] += 1.0
        m1["rerot_parallel_model_tokens"] += 150
        m1["rerot_parallel_seconds"] += 0.6
        m1["rerot_final_fences"] += 1
        m1["rerot_parallel_peak_lanes_total"] += 2

        mock_urlopen = self.create_mock_urlopen([m0, m1])
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate(["--rounds", "1", "--min-peak-lanes", "6"])

        self.assertEqual(exit_code, gate_mod.EXIT_FAIL)
        self.assertEqual(output_data["round_checks"][0]["checks"]["multi_lane_peak_lanes"], gate_mod.CHECK_FAIL)

    def test_six_allocated_pens_without_one_six_row_decode_fails(self) -> None:
        m0 = dict(self.base_metrics)
        m1 = dict(m0)
        m1["rerot_completed_episode_total"] += 1
        m1["rerot_completed_model_tokens"] += 100
        m1["rerot_public_tokens"] += 100
        m1["rerot_completed_episode_seconds"] += 1
        m1["rerot_parallel_model_tokens"] += 80
        m1["rerot_parallel_seconds"] += 0.8
        m1["rerot_final_fences"] += 1
        m1["rerot_parallel_peak_lanes_total"] += 6

        with patch("urllib.request.urlopen", side_effect=self.create_mock_urlopen([m0, m1])):
            exit_code, output_data = self.run_gate(["--min-peak-lanes", "6"])

        self.assertEqual(exit_code, gate_mod.EXIT_FAIL)
        self.assertEqual(output_data["round_checks"][0]["checks"]["multi_lane_peak_lanes"], gate_mod.CHECK_OK)
        self.assertEqual(output_data["round_checks"][0]["checks"]["six_pens_in_one_decode"], gate_mod.CHECK_FAIL)

    def test_valid_round_passes(self) -> None:
        """6. Test that valid round (all counters present, episode=1, no abort, fence=1,
        visibility OK, parallel exercised, ratio>=min_ratio) passes."""
        m0 = dict(self.base_metrics)

        m1 = dict(m0)
        m1["rerot_completed_episode_total"] += 1.0
        m1["rerot_completed_model_tokens"] += 100.0
        m1["rerot_public_tokens"] += 60.0
        m1["rerot_private_tokens"] += 30.0
        m1["rerot_pending_tokens"] += 10.0  # 60 + 30 + 10 = 100 == completed_model_tokens
        m1["rerot_completed_episode_seconds"] += 1.0  # parallel phase fits within episode
        m1["rerot_parallel_model_tokens"] += 80.0
        m1["rerot_parallel_seconds"] += 0.95  # 80 model tokens / 0.95 s < serial's 100 sampled tok/s
        m1["rerot_hard_aborts"] += 0.0
        m1["rerot_final_fences"] += 1.0
        m1["rerot_parallel_peak_lanes_total"] += 2.0

        metrics_seq = [m0, m1]
        mock_urlopen = self.create_mock_urlopen(metrics_seq)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code, output_data = self.run_gate(["--rounds", "1", "--min-ratio", "1.5"])

        self.assertEqual(exit_code, gate_mod.EXIT_PASS)
        self.assertEqual(output_data["status"], "pass")
        self.assertTrue(output_data["passed"])
        self.assertEqual(output_data["round_checks"][0]["status"], "ok")
        for check_name, check_val in output_data["round_checks"][0]["checks"].items():
            if check_name == "operator_route_hit":
                self.assertEqual(check_val, gate_mod.CHECK_NOT_EXERCISED)
            else:
                self.assertEqual(check_val, gate_mod.CHECK_OK)
        self.assertEqual(output_data["verdict_checks"]["per_round_all_ok"], gate_mod.CHECK_OK)
        self.assertEqual(output_data["verdict_checks"]["aggregate_no_slower_than_serial_paired"], gate_mod.CHECK_OK)
        self.assertEqual(output_data["verdict_checks"]["parallel_work_exercised_paired"], gate_mod.CHECK_OK)
        self.assertLess(output_data["pairs"][0]["parallel_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
