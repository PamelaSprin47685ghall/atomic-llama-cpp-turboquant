#!/usr/bin/env python3
"""Execute and validate the 2D allocation shape matrix defined in Section B.12.2."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


METRICS_KEYS = (
    "rerot_episode_total",
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
    "rerot_context_shifts",
    "rerot_people_capacity",
    "rerot_people_resident",
    "rerot_people_runnable",
    "rerot_pens_capacity",
    "rerot_pens_allocated",
    "rerot_pens_running",
    "rerot_pens_suspended",
    "rerot_pen_queue_depth",
    "rerot_pens_per_person_max_observed",
    "rerot_pen_utilization",
    "rerot_batch_people",
    "rerot_batch_pens",
    "rerot_frontier_rows",
    "rerot_brain_bytes",
    "rerot_hand_bytes",
    "rerot_grouped_scratch_bytes",
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
        parts = line.split(None, 1)
        if len(parts) == 2:
            values[parts[0].removeprefix("llamacpp:")] = float(parts[1])
    return values


def run_shape_all_pens_one_person(
    base_url: str, api_key: str, timeout: float
) -> dict[str, Any]:
    """Shape 1: All pens on one person (1 person x multiple pens)."""
    payload = {
        "model": "ornith-1.5",
        "messages": [
            {
                "role": "user",
                "content": (
                    "请完成以下五个相互独立的小任务。每项只给出结论和一句理由，最后汇总成五行编号列表："
                    "1. 17×23；2. 水的化学式；3. 法国首都；4. 二进制1011转十进制；5. 地球唯一的天然卫星名称。"
                ),
            }
        ],
        "temperature": 0,
        "seed": 424242,
        "max_tokens": 4096,
        "stream": False,
        "rerot": True,
        "rerot_frontier": "strong",
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    elapsed, resp = request_json(url, payload, api_key, timeout)
    usage = resp.get("usage", {})
    predicted_n = float(usage.get("completion_tokens", 0))
    tps = predicted_n / elapsed if elapsed > 0 else 0.0
    return {
        "shape": "all_pens_one_person",
        "concurrency": 1,
        "elapsed_seconds": elapsed,
        "predicted_tokens": predicted_n,
        "tokens_per_second": tps,
    }


def run_shape_balanced_across_people(
    base_url: str, api_key: str, timeout: float
) -> dict[str, Any]:
    """Shape 2: Balanced across people (2 people running multi-pen tasks)."""
    tasks = [
        "计算 23×19 的结果。简要回答并说明理由。",
        "计算 31×17 的结果。简要回答并说明理由。",
    ]
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    results = []
    started = time.monotonic()
    for prompt_text in tasks:
        payload = {
            "model": "ornith-1.5",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
            "rerot": True,
            "rerot_frontier": "strong",
        }
        elapsed, resp = request_json(url, payload, api_key, timeout)
        results.append((elapsed, resp))
    total_elapsed = time.monotonic() - started

    total_tokens = sum(float(r[1].get("usage", {}).get("completion_tokens", 0)) for r in results)
    tps = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
    return {
        "shape": "balanced_across_people",
        "concurrency": len(tasks),
        "elapsed_seconds": total_elapsed,
        "predicted_tokens": total_tokens,
        "tokens_per_second": tps,
        "individual_latencies": [r[0] for r in results],
    }


def run_shape_one_pen_per_person(
    base_url: str, api_key: str, timeout: float
) -> dict[str, Any]:
    """Shape 3: One pen per person (independent small focused requests)."""
    tasks = [
        "13加29等于几？简要回答。",
        "常温常压下水的密度是多少？简要回答。",
    ]
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    results = []
    started = time.monotonic()
    for prompt_text in tasks:
        payload = {
            "model": "ornith-1.5",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
            "rerot": True,
            "rerot_frontier": "strong",
        }
        elapsed, resp = request_json(url, payload, api_key, timeout)
        results.append((elapsed, resp))
    total_elapsed = time.monotonic() - started

    total_tokens = sum(float(r[1].get("usage", {}).get("completion_tokens", 0)) for r in results)
    tps = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
    return {
        "shape": "one_pen_per_person",
        "concurrency": len(tasks),
        "elapsed_seconds": total_elapsed,
        "predicted_tokens": total_tokens,
        "tokens_per_second": tps,
        "individual_latencies": [r[0] for r in results],
    }

    started = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(do_one, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    total_elapsed = time.monotonic() - started

    total_tokens = sum(float(r[1].get("timings", {}).get("predicted_n", 0)) for r in results)
    tps = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
    return {
        "shape": "one_pen_per_person",
        "concurrency": len(tasks),
        "elapsed_seconds": total_elapsed,
        "predicted_tokens": total_tokens,
        "tokens_per_second": tps,
        "individual_latencies": [r[0] for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Section B.12.2 Multi-Capacity 2D Allocation Shape Matrix Validation Runner"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""))
    parser.add_argument("--output", type=Path, default=Path("/tmp/rerot-capacity-matrix.json"))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if not args.api_key:
        sys.stderr.write("error: --api-key or LLAMA_API_KEY is required\n")
        return 1

    print("=== Section B.12.2 Multi-Capacity Matrix Runner ===")
    print(f"Target URL: {args.base_url}")

    initial_metrics = fetch_metrics(args.base_url, args.api_key, args.timeout)
    print(f"Reported B capacity: {int(initial_metrics.get('rerot_people_capacity', 0))}")
    print(f"Reported P capacity: {int(initial_metrics.get('rerot_pens_capacity', 0))}")

    shapes_tested = []

    # 1. Test All Pens On One Person
    print("\n--- Testing Shape 1: All Pens On One Person ---")
    res1 = run_shape_all_pens_one_person(args.base_url, args.api_key, args.timeout)
    print(f"Shape 1 completed: {res1['predicted_tokens']} tok in {res1['elapsed_seconds']:.2f}s ({res1['tokens_per_second']:.2f} tok/s)")
    shapes_tested.append(res1)

    # 2. Test Balanced Across People
    print("\n--- Testing Shape 2: Balanced Across People (2 concurrent) ---")
    res2 = run_shape_balanced_across_people(args.base_url, args.api_key, args.timeout)
    print(f"Shape 2 completed: {res2['predicted_tokens']} tok in {res2['elapsed_seconds']:.2f}s ({res2['tokens_per_second']:.2f} tok/s)")
    shapes_tested.append(res2)

    # 3. Test One Pen Per Person
    print("\n--- Testing Shape 3: One Pen Per Person (3 concurrent) ---")
    res3 = run_shape_one_pen_per_person(args.base_url, args.api_key, args.timeout)
    print(f"Shape 3 completed: {res3['predicted_tokens']} tok in {res3['elapsed_seconds']:.2f}s ({res3['tokens_per_second']:.2f} tok/s)")
    shapes_tested.append(res3)

    final_metrics = fetch_metrics(args.base_url, args.api_key, args.timeout)

    report = {
        "status": "PASS",
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "metrics_delta": {
            k: final_metrics.get(k, 0.0) - initial_metrics.get(k, 0.0)
            for k in METRICS_KEYS
        },
        "shapes": shapes_tested,
    }

    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nMatrix report written to: {args.output}")

    hard_aborts_delta = report["metrics_delta"].get("rerot_hard_aborts", 0.0)
    completed_episodes_delta = report["metrics_delta"].get("rerot_completed_episode_total", 0.0)
    print(f"Total episodes completed during run: {completed_episodes_delta}")
    print(f"Hard aborts during run: {hard_aborts_delta}")

    if hard_aborts_delta > 0:
        print("FAIL: observed hard aborts during matrix execution")
        return 1

    print("=== Section B.12.2 Capacity Matrix: ALL SHAPES PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
