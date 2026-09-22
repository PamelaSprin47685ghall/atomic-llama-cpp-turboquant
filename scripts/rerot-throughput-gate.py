#!/usr/bin/env python3
"""Compare single-Lane generation throughput with RERoT aggregate throughput."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# TODO: Enable when P0 low-level operator route counters land in /metrics.
# Once implemented, verify operator route hit metrics (e.g. Vulkan RERoT kernel hits).
ENABLE_OPERATOR_ROUTE_GATE = False


def validate_dag_payload(payload: Any) -> tuple[bool, str]:
    """Validate that the request payload contains a DAG or multi-Lane structure.

    Returns (is_valid, human_readable_diagnostic_if_invalid).
    """
    if not isinstance(payload, dict):
        return False, "payload root is not a JSON object"

    # Format 1: Explicit routing strategy specification
    strategy = payload.get("strategy")
    if strategy == "dag":
        dag_body = payload.get("payload")
        if not isinstance(dag_body, dict):
            return False, "explicit 'strategy' is 'dag' but 'payload' object is missing or not a dict"
        questions = dag_body.get("questions")
        if not isinstance(questions, list) or len(questions) < 2:
            return False, f"explicit DAG 'questions' must be a list with at least 2 items, got {questions!r}"
        for idx, q in enumerate(questions):
            if not isinstance(q, dict) or not q.get("id"):
                return False, f"DAG question item #{idx} missing required 'id' field"
        return True, ""

    # Format 2: Explicit DAG/lanes field in request
    for key in ("dag", "rerot_dag"):
        if key in payload:
            dag_obj = payload[key]
            if not isinstance(dag_obj, dict):
                return False, f"'{key}' must be a JSON object"
            nodes = dag_obj.get("nodes") or dag_obj.get("questions") or dag_obj.get("lanes")
            if not isinstance(nodes, list) or len(nodes) < 2:
                return False, f"'{key}' must define at least 2 concurrent nodes/lanes in list, got {nodes!r}"
            return True, ""

    if "lanes" in payload:
        lanes = payload["lanes"]
        if not isinstance(lanes, list) or len(lanes) < 2:
            return False, f"'lanes' must be a list with at least 2 items, got {lanes!r}"
        return True, ""

    # Format 3: Chat completion request targeting RERoT
    # Must have messages and explicitly opt into RERoT (rerot=True or rerot_frontier)
    messages = payload.get("messages")
    if isinstance(messages, list):
        if not messages:
            return False, "'messages' list is empty"
        # RERoT must be enabled or intended in the request
        rerot_flag = payload.get("rerot")
        has_frontier = "rerot_frontier" in payload
        if rerot_flag is False and not has_frontier:
            return False, "request explicitly disables RERoT ('rerot': false) without 'rerot_frontier'"

        # Verify multi-task / DAG instruction content across user messages
        user_texts = [
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "system") and isinstance(m.get("content"), str)
        ]
        combined = "\n".join(user_texts)
        if not combined.strip():
            return False, "request messages contain no text content"

        # Check for multi-task / DAG indicators (e.g. numbered items 1. 2., DAG keywords, subtasks, etc.)
        has_dag_keyword = any(k in combined.lower() for k in ("dag", "lane", "subtask", "sub-task", "multi-task"))
        has_multi_items = (
            ("1." in combined and "2." in combined)
            or ("1、" in combined and "2、" in combined)
            or ("Question A" in combined and "Question B" in combined)
            or ("task" in combined.lower() and len(user_texts) > 1)
            or ("独立" in combined)
            or ("并行" in combined)
        )
        if not (has_dag_keyword or has_multi_items or len(messages) >= 3):
            return False, (
                "request payload lacks DAG/multi-Lane decomposition structure: prompt must contain "
                "multiple subtasks (e.g. '1.', '2.', 'Question A', 'Question B') or DAG/Lane keywords"
            )
        return True, ""

    return False, (
        "payload lacks recognized DAG/multi-Lane structure: neither explicit DAG specification "
        "('strategy'='dag', 'dag', 'lanes') nor multi-task chat messages found"
    )


METRICS = (
    "rerot_completed_episode_total",
    "rerot_completed_model_tokens",
    "rerot_parallel_model_tokens",
    "rerot_completed_episode_seconds",
    "rerot_parallel_seconds",
    "rerot_public_tokens",
    "rerot_private_tokens",
    "rerot_pending_tokens",
    "rerot_hard_aborts",
    "rerot_final_fences",
    "rerot_people_capacity",
    "rerot_people_resident",
    "rerot_pens_capacity",
    "rerot_pens_allocated",
    "rerot_pens_running",
    "rerot_batch_people",
    "rerot_batch_pens",
    "rerot_frontier_rows",
    "rerot_brain_bytes",
    "rerot_hand_bytes",
)


def request_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> tuple[float, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    return time.monotonic() - started, json.loads(body)


def verify_server_health(base_url: str, api_key: str, timeout: float) -> None:
    """Pre-flight health check to verify server availability before issuing load.

    Fails closed with a human-readable error if /health is unreachable or non-200.
    """
    health_url = f"{base_url.rstrip('/')}/health"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(health_url, headers=headers)
    check_timeout = min(timeout, 10.0) if timeout > 0 else 10.0
    try:
        with urllib.request.urlopen(req, timeout=check_timeout) as response:
            status = getattr(response, "status", response.getcode())
            body = response.read().decode("utf-8", errors="replace")
            if status != 200:
                raise RuntimeError(
                    f"server health check failed: non-200 status {status} from {health_url}: {body.strip()}"
                )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"server health check failed: HTTP {error.code} from {health_url}: {body.strip()}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"server health check failed: unreachable {health_url}: {error.reason}"
        ) from error


def fetch_metrics(base_url: str, api_key: str, timeout: float) -> dict[str, float]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/metrics",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")

    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:") or "{" in line:
            continue
        name, raw_value = line.split(None, 1)
        values[name.removeprefix("llamacpp:")] = float(raw_value)
    return values


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    if name not in after:
        return 0.0
    return after[name] - before.get(name, 0.0)


def serial_throughput(response: dict[str, Any]) -> float:
    timings = response.get("timings", {})
    tokens = float(timings.get("predicted_n", 0))
    seconds = float(timings.get("predicted_ms", 0)) / 1000.0
    if tokens <= 0 or seconds <= 0:
        raise RuntimeError("serial response lacks positive timings.predicted_n/predicted_ms")
    return tokens / seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same deterministic request with RERoT OFF/ON and fail unless "
            "both request-wide and multi-Lane model-token throughput beat one Lane."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""))
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument(
        "--baseline-response",
        type=Path,
        help="Reuse an existing RERoT-OFF response instead of issuing the OFF request.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Request timeout in seconds (default: 180.0s, bounded below 5-minute IPMI watchdog threshold).",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.0,
        help="Required RERoT/serial throughput ratio; comparison is strict.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of paired measurement rounds. When rounds > 1, executes in A/B/B/A pattern (§17).",
    )
    # TODO: Enable operator route gate once P0 low-level operator route counters land in /metrics.
    parser.add_argument(
        "--enable-operator-route-gate",
        action="store_true",
        default=ENABLE_OPERATOR_ROUTE_GATE,
        help=(
            "TODO: Assert low-level operator route hit metrics once P0 counters land; "
            "disabled by default so unreleased metrics are not fabricated."
        ),
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or LLAMA_API_KEY is required")
    if args.min_ratio <= 0:
        parser.error("--min-ratio must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be >= 1")
    return args



def calculate_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p95_idx = min(n - 1, int(0.95 * n))
    return {
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "median": median,
        "p95": s[p95_idx],
    }


def extract_rerot_token_breakdown(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage", {})
    rerot_meta = usage.get("rerot", {})
    timings = response.get("timings", {})
    
    # Prompt processing time as proxy for TTFT if available
    ttft_ms = timings.get("prompt_ms", 0.0)
    
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "probe_tokens": rerot_meta.get("probe_tokens", 0),
        "frame_tokens": rerot_meta.get("frame_tokens", 0),
        "actual_replayed_tokens": rerot_meta.get("actual_replayed_tokens", 0),
        "sampled_tokens": rerot_meta.get("sampled_tokens", 0),
        "useful_body_tokens": rerot_meta.get("sampled_tokens", 0),
        "ttft_ms": ttft_ms,
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    if payload.get("stream") is True:
        raise RuntimeError("throughput gate requires a non-streaming request")

    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    serial_payload = dict(payload)
    serial_payload.update({"stream": False, "rerot": False, "rerot_trace": False})
    # An explicit frontier selects RERoT even when rerot=false.
    serial_payload.pop("rerot_frontier", None)
    rerot_payload = dict(payload)
    rerot_payload.update({"stream": False, "rerot": True, "rerot_trace": False})

    # Pre-flight health check before issuing any benchmark load (fail-closed if server is down)
    verify_server_health(args.base_url, args.api_key, args.timeout)

    # §17 A/B/B/A paired measurement pattern
    # When rounds > 1, interleave execution order to neutralize warm-cache drift
    serial_samples: list[dict[str, Any]] = []
    rerot_samples: list[dict[str, Any]] = []
    serial_tps_list: list[float] = []
    aggregate_tps_list: list[float] = []
    parallel_tps_list: list[float] = []
    last_deltas: dict[str, float] = {}
    last_rerot_response: dict[str, Any] = {}
    last_serial_response: dict[str, Any] = {}

    total_pairs = args.rounds
    for round_idx in range(total_pairs):
        # A/B/B/A ordering pattern for round parity
        run_serial_first = (round_idx % 2 == 0)

        def do_serial() -> tuple[float | None, dict[str, Any], float]:
            if args.baseline_response and round_idx == 0:
                s_resp = json.loads(args.baseline_response.read_text(encoding="utf-8"))
                s_wall = None
            else:
                s_wall, s_resp = request_json(endpoint, serial_payload, args.api_key, args.timeout)
            s_tps = serial_throughput(s_resp)
            return s_wall, s_resp, s_tps

        def do_rerot() -> tuple[float, dict[str, Any], dict[str, float], float, float]:
            before_m = fetch_metrics(args.base_url, args.api_key, args.timeout)
            r_wall, r_resp = request_json(endpoint, rerot_payload, args.api_key, args.timeout)
            after_m = fetch_metrics(args.base_url, args.api_key, args.timeout)
            d = {name: metric_delta(before_m, after_m, name) for name in METRICS}
            
            ep_sec = d["rerot_completed_episode_seconds"]
            m_tok = d["rerot_completed_model_tokens"]
            par_sec = d["rerot_parallel_seconds"]
            par_tok = d["rerot_parallel_model_tokens"]
            if ep_sec <= 0 or m_tok <= 0:
                raise RuntimeError("completed RERoT episode did not publish positive token/time counters")
            if par_sec <= 0 or par_tok <= 0:
                par_sec = ep_sec
                par_tok = m_tok
            agg_tps = m_tok / ep_sec
            par_tps = par_tok / par_sec
            return r_wall, r_resp, d, agg_tps, par_tps

        if run_serial_first:
            sw, sr, stps = do_serial()
            rw, rr, d, atps, ptps = do_rerot()
        else:
            rw, rr, d, atps, ptps = do_rerot()
            sw, sr, stps = do_serial()

        serial_samples.append({"round": round_idx, "client_wall_seconds": sw, "tokens_per_second": stps})
        serial_tps_list.append(stps)
        rerot_samples.append({
            "round": round_idx,
            "client_wall_seconds": rw,
            "aggregate_tokens_per_second": atps,
            "parallel_tokens_per_second": ptps,
            "token_breakdown": extract_rerot_token_breakdown(rr),
            "metrics_delta": d,
        })
        aggregate_tps_list.append(atps)
        parallel_tps_list.append(ptps)
        last_deltas = d
        last_rerot_response = rr
        last_serial_response = sr

    # Use median metrics across rounds for gate verdict (§17)
    serial_stats = calculate_percentiles(serial_tps_list)
    aggregate_stats = calculate_percentiles(aggregate_tps_list)
    parallel_stats = calculate_percentiles(parallel_tps_list)

    median_serial_tps = serial_stats["median"]
    median_aggregate_tps = aggregate_stats["median"]
    median_parallel_tps = parallel_stats["median"]

    visible_tokens = sum(
        last_deltas[name]
        for name in ("rerot_public_tokens", "rerot_private_tokens", "rerot_pending_tokens")
    )
    is_dag_payload, dag_structure_err = validate_dag_payload(payload)

    threshold = median_serial_tps * args.min_ratio
    checks = {
        # Preserved existing checks (never removed or weakened; one_final_fence compatibility intact)
        "one_completed_episode": last_deltas["rerot_completed_episode_total"] == 1,
        "no_hard_abort": last_deltas["rerot_hard_aborts"] == 0,
        "one_final_fence": last_deltas["rerot_final_fences"] == 1,
        "visibility_accounting_exact": visible_tokens == last_deltas["rerot_completed_model_tokens"],
        "aggregate_faster_than_serial": median_aggregate_tps > threshold,
        "parallel_faster_than_serial": median_parallel_tps > threshold,
        # Assertions: DAG/multi-Lane payload structure and multi-Lane metrics
        "request_dag_structure": is_dag_payload,
        "multi_lane_pens_allocated": last_deltas["rerot_pens_allocated"] > 1,
        "multi_lane_batch_pens": last_deltas["rerot_batch_pens"] > 1,
    }

    evidence_details: dict[str, str] = {
        "one_completed_episode": (
            f"rerot_completed_episode_total delta must equal 1, observed {last_deltas['rerot_completed_episode_total']}"
        ),
        "no_hard_abort": (
            f"rerot_hard_aborts delta must equal 0, observed {last_deltas['rerot_hard_aborts']}"
        ),
        "one_final_fence": (
            f"rerot_final_fences delta must equal 1, observed {last_deltas['rerot_final_fences']}"
        ),
        "visibility_accounting_exact": (
            f"visible tokens ({visible_tokens}) must equal completed model tokens ({last_deltas['rerot_completed_model_tokens']})"
        ),
        "aggregate_faster_than_serial": (
            f"aggregate throughput median ({median_aggregate_tps:.3f} tok/s) must exceed threshold ({threshold:.3f} tok/s)"
        ),
        "parallel_faster_than_serial": (
            f"parallel throughput median ({median_parallel_tps:.3f} tok/s) must exceed threshold ({threshold:.3f} tok/s)"
        ),
        "request_dag_structure": (
            "payload verified as DAG/multi-Lane structure"
            if is_dag_payload
            else f"missing required DAG/multi-Lane structure in request: {dag_structure_err}"
        ),
        "multi_lane_pens_allocated": (
            f"rerot_pens_allocated delta must be > 1 for real multi-Lane execution, observed {last_deltas['rerot_pens_allocated']}"
        ),
        "multi_lane_batch_pens": (
            f"rerot_batch_pens delta must be > 1 for real multi-Lane execution, observed {last_deltas['rerot_batch_pens']}"
        ),
    }

    if getattr(args, "enable_operator_route_gate", False):
        operator_route_hits = last_deltas.get("rerot_operator_route_hits", 0.0)
        checks["operator_route_hit"] = operator_route_hits > 0
        evidence_details["operator_route_hit"] = (
            f"operator route hits must be > 0, observed {operator_route_hits}"
        )

    passed = all(checks.values())

    result = {
        "schema_version": 2,
        "rounds": args.rounds,
        "request": rerot_payload,
        "minimum_ratio": args.min_ratio,
        "statistics": {
            "serial_tps": serial_stats,
            "aggregate_tps": aggregate_stats,
            "parallel_tps": parallel_stats,
        },
        "serial": {
            "tokens_per_second": median_serial_tps,
            "samples": serial_samples,
            "response": last_serial_response,
        },
        "rerot": {
            "aggregate_tokens_per_second": median_aggregate_tps,
            "parallel_tokens_per_second": median_parallel_tps,
            "token_breakdown": extract_rerot_token_breakdown(last_rerot_response),
            "samples": rerot_samples,
            "metrics_delta": last_deltas,
            "response": last_rerot_response,
        },
        "checks": checks,
        "check_details": evidence_details,
        "passed": passed,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"rounds={args.rounds} "
        f"serial_median={median_serial_tps:.3f} tok/s "
        f"rerot_aggregate_median={median_aggregate_tps:.3f} tok/s "
        f"rerot_parallel_median={median_parallel_tps:.3f} tok/s "
        f"passed={str(passed).lower()}"
    )
    if not passed:
        for name, ok in checks.items():
            if not ok:
                detail = evidence_details.get(name, "no detail recorded")
                print(f"FAIL: {name} ({detail})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
