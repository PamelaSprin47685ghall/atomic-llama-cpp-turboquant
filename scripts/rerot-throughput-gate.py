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
        raise RuntimeError(f"required metric is missing after request: llamacpp:{name}")
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
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.0,
        help="Required RERoT/serial throughput ratio; comparison is strict.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or LLAMA_API_KEY is required")
    if args.min_ratio <= 0:
        parser.error("--min-ratio must be positive")
    return args


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

    if args.baseline_response:
        serial_wall = None
        serial_response = json.loads(args.baseline_response.read_text(encoding="utf-8"))
    else:
        serial_wall, serial_response = request_json(
            endpoint, serial_payload, args.api_key, args.timeout
        )
    serial_tps = serial_throughput(serial_response)

    before = fetch_metrics(args.base_url, args.api_key, args.timeout)
    rerot_wall, rerot_response = request_json(
        endpoint, rerot_payload, args.api_key, args.timeout
    )
    after = fetch_metrics(args.base_url, args.api_key, args.timeout)
    deltas = {name: metric_delta(before, after, name) for name in METRICS}

    episode_seconds = deltas["rerot_completed_episode_seconds"]
    model_tokens = deltas["rerot_completed_model_tokens"]
    parallel_seconds = deltas["rerot_parallel_seconds"]
    parallel_tokens = deltas["rerot_parallel_model_tokens"]
    if episode_seconds <= 0 or model_tokens <= 0:
        raise RuntimeError("completed RERoT episode did not publish positive token/time counters")
    if parallel_seconds <= 0 or parallel_tokens <= 0:
        raise RuntimeError("RERoT episode did not publish a positive multi-Lane interval")

    aggregate_tps = model_tokens / episode_seconds
    parallel_tps = parallel_tokens / parallel_seconds
    visible_tokens = sum(
        deltas[name]
        for name in ("rerot_public_tokens", "rerot_private_tokens", "rerot_pending_tokens")
    )
    threshold = serial_tps * args.min_ratio
    checks = {
        "one_completed_episode": deltas["rerot_completed_episode_total"] == 1,
        "no_hard_abort": deltas["rerot_hard_aborts"] == 0,
        "one_final_fence": deltas["rerot_final_fences"] == 1,
        "visibility_accounting_exact": visible_tokens == model_tokens,
        "aggregate_faster_than_serial": aggregate_tps > threshold,
        "parallel_faster_than_serial": parallel_tps > threshold,
    }
    passed = all(checks.values())

    result = {
        "schema_version": 1,
        "request": rerot_payload,
        "minimum_ratio": args.min_ratio,
        "serial": {
            "client_wall_seconds": serial_wall,
            "tokens_per_second": serial_tps,
            "response": serial_response,
        },
        "rerot": {
            "client_wall_seconds": rerot_wall,
            "aggregate_tokens_per_second": aggregate_tps,
            "parallel_tokens_per_second": parallel_tps,
            "metrics_delta": deltas,
            "response": rerot_response,
        },
        "checks": checks,
        "passed": passed,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"serial={serial_tps:.3f} tok/s "
        f"rerot_aggregate={aggregate_tps:.3f} tok/s "
        f"rerot_parallel={parallel_tps:.3f} tok/s "
        f"passed={str(passed).lower()}"
    )
    if not passed:
        for name, ok in checks.items():
            if not ok:
                print(f"FAIL: {name}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
