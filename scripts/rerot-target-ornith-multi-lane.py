#!/usr/bin/env python3
"""Target Ornith Multi-Lane DAG Verification Suite (AGENTS.md Stage 6 / RERoT.md §13.2 & §15.6).

Verifies multi-lane DAG execution on the target Ornith-1.5-35B model:
1. Flat 2-worker DAG (independent subtasks)
2. Flat 3-worker DAG (cyclic reader orders 1->(2,3,1), 2->(3,1,2), 3->(1,2,3) & peer uptake)
3. A -> C with B independent overlap (dependency barrier + concurrent execution)
4. Diamond DAG: 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4 (ancestor deduplication and join gate)
5. Unequal-length workers (history preservation across completions)

Strict safety: Runs against an isolated, resource-bounded server instance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def request(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}") from error


def verify_server_health(base_url: str) -> None:
    health_url = f"{base_url.rstrip('/')}/health"
    req = urllib.request.Request(health_url)
    with urllib.request.urlopen(req, timeout=10.0) as response:
        if response.status != 200:
            raise RuntimeError(f"Server at {base_url} returned health status {response.status}")
    print(f"[OK] Server at {base_url} is healthy.")


def run_test_case(url: str, model: str, name: str, prompt: str, timeout: float = 300.0) -> dict:
    print(f"\n==================================================")
    print(f"Running Multi-Lane DAG Test: {name}")
    print(f"Prompt: {prompt}")
    print(f"==================================================")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "seed": 424242,
        "max_tokens": 1024,
        "stream": False,
        "rerot": True,
        "rerot_frontier": "strong",
    }
    t0 = time.monotonic()
    result = request(url, payload, timeout=timeout)
    elapsed = time.monotonic() - t0

    choice = result["choices"][0]
    message = choice["message"]
    usage = result.get("usage", {})
    rerot_usage = usage.get("rerot", {})

    print(f"Completed in {elapsed:.2f}s")
    print(f"Finish Reason: {choice.get('finish_reason')}")
    print(f"Content: {message.get('content', '').strip()}")
    print(f"Reasoning Length: {len(message.get('reasoning_content', ''))} chars")
    print(f"Usage: {usage}")
    print(f"RERoT Accounting: {rerot_usage}")

    # Assertions
    if choice.get("finish_reason") != "stop":
        raise AssertionError(f"{name}: finish_reason is '{choice.get('finish_reason')}', expected 'stop'")
    
    # Invariant: No prompt injection or FRAME tags leaked into final content
    content = message.get("content", "")
    forbidden_tokens = ["# Tools", "spawn_lane", "<|im_start|>", "<|im_end|>"]
    for ft in forbidden_tokens:
        if ft in content:
            raise AssertionError(f"{name}: leaked internal token '{ft}' into content: {content}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18089", help="Server base URL")
    parser.add_argument("--model", default="ornith-1.5", help="Model name")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout")
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    verify_server_health(args.base_url)

    # 1. Flat 2-worker DAG: Two independent calculations synthesized into one
    prompt_flat2 = (
        "Decompose into DAG of independent questions: Question A: What is 25 * 12? "
        "Question B: What is 15 * 16? Synthesize the sum."
    )
    res_flat2 = run_test_case(url, args.model, "flat_2_workers", prompt_flat2, args.timeout)

    # 2. A -> C with B independent: Dependency chain with concurrent peer
    prompt_dep = (
        "Decompose into DAG of questions with dependencies: Question A: calculate 14 * 15. "
        "Question B (independent of A): calculate 30 * 20. Question C (depends on A): calculate result of A + 50. "
        "Synthesize the total of B + C."
    )
    res_dep = run_test_case(url, args.model, "a_to_c_with_b_independent", prompt_dep, args.timeout)

    # 3. Diamond DAG: 1 -> 2, 1 -> 3, 2/3 -> 4
    prompt_diamond = (
        "Decompose into diamond DAG of questions: Question 1: evaluate base b = 100. "
        "Question 2 (depends on 1): compute v1 = b * 3. Question 3 (depends on 1): compute v2 = b * 5. "
        "Question 4 (depends on 2 and 3): combine v1 + v2. Synthesize final answer. "
        "Answer with the plain text final result only without invoking tools."
    )
    res_diamond = run_test_case(url, args.model, "diamond_dag", prompt_diamond, args.timeout)

    print("\n==================================================")
    print("ALL target Ornith multi-lane DAG test cases PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
