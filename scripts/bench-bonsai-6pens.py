#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

def request_json(url, payload, timeout=600.0, api_key=""):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json',
                 **({'Authorization': f'Bearer {api_key}'} if api_key else {})},
        method='POST',
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    dur = time.monotonic() - t0
    return dur, json.loads(body.decode('utf-8'))

def fetch_metrics(base_url, timeout=5.0, api_key=""):
    req = urllib.request.Request(f"{base_url.rstrip('/')}/metrics",
                                 headers={'Authorization': f'Bearer {api_key}'} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode('utf-8')
    except Exception:
        return {}
    vals = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:") or "{" in line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            vals[parts[0].replace("llamacpp:", "")] = float(parts[1])
    return vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18089")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""))
    parser.add_argument("--prompt", default=None,
                        help="custom prompt (default: legacy DAG-of-6 arithmetic question)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model", default="bonsai-2-27b")
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    prompt = args.prompt if args.prompt is not None else (
        "Decompose into DAG of 6 independent sub-questions: "
        "Question 1: compute 1 * 10. "
        "Question 2: compute 2 * 10. "
        "Question 3: compute 3 * 10. "
        "Question 4: compute 4 * 10. "
        "Question 5: compute 5 * 10. "
        "Question 6: compute 6 * 10. "
        "Synthesize the total sum of all 6 values. Output the final numerical sum."
    )

    base_payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "seed": 424242,
        "max_tokens": args.max_tokens,
        "stream": False,
    }

    print("==================================================================")
    print("      Ternary Bonsai 2 27B (PQ2_0) 6-Pen Throughput Benchmark     ")
    print("==================================================================")

    # 1. RERoT OFF Baseline
    print("\n>>> [1/2] Running Baseline: RERoT OFF (Single-stream)...")
    payload_off = dict(base_payload)
    payload_off["rerot"] = False
    payload_off["rerot_trace"] = False

    m_before_off = fetch_metrics(args.base_url, api_key=args.api_key)
    dur_off, resp_off = request_json(endpoint, payload_off, timeout=args.timeout, api_key=args.api_key)
    m_after_off = fetch_metrics(args.base_url, api_key=args.api_key)

    timings_off = resp_off.get("timings", {})
    pred_n_off = timings_off.get("predicted_n", 0)
    pred_ms_off = timings_off.get("predicted_ms", 0.0)
    tps_off = (pred_n_off / (pred_ms_off / 1000.0)) if pred_ms_off > 0 else (pred_n_off / dur_off)

    content_off = resp_off["choices"][0]["message"]["content"].strip()
    print(f"  Wall time        : {dur_off:.3f} s")
    print(f"  Predicted tokens : {pred_n_off}")
    print(f"  Server TPS       : {tps_off:.2f} tok/s")
    print(f"  End-to-end TPS   : {(pred_n_off / dur_off):.2f} tok/s")
    print(f"  Output snippet   : {content_off[:120]}...")

    time.sleep(2.0)

    # 2. RERoT ON (6 Pens)
    print("\n>>> [2/2] Running RERoT ON (P=6 Adaptive Multi-Lane)...")
    payload_on = dict(base_payload)
    payload_on["rerot"] = True
    payload_on["rerot_frontier"] = "strong"
    payload_on["rerot_trace"] = False

    m_before_on = fetch_metrics(args.base_url, api_key=args.api_key)
    dur_on, resp_on = request_json(endpoint, payload_on, timeout=args.timeout, api_key=args.api_key)
    m_after_on = fetch_metrics(args.base_url, api_key=args.api_key)

    timings_on = resp_on.get("timings", {})
    pred_n_on = timings_on.get("predicted_n", 0)
    pred_ms_on = timings_on.get("predicted_ms", 0.0)

    content_on = resp_on["choices"][0]["message"]["content"].strip()

    d_par_tok = m_after_on.get("rerot_parallel_model_tokens", 0.0) - m_before_on.get("rerot_parallel_model_tokens", 0.0)
    d_par_sec = m_after_on.get("rerot_parallel_seconds", 0.0) - m_before_on.get("rerot_parallel_seconds", 0.0)
    d_tot_tok = m_after_on.get("rerot_completed_model_tokens", 0.0) - m_before_on.get("rerot_completed_model_tokens", 0.0)
    d_tot_sec = m_after_on.get("rerot_completed_episode_seconds", 0.0) - m_before_on.get("rerot_completed_episode_seconds", 0.0)

    agg_tps = (d_tot_tok / d_tot_sec) if d_tot_sec > 0 else (pred_n_on / dur_on)
    par_tps = (d_par_tok / d_par_sec) if d_par_sec > 0 else 0.0
    client_tps = pred_n_on / dur_on

    print(f"  Wall time        : {dur_on:.3f} s")
    print(f"  Predicted tokens : {pred_n_on} (Model Total: {int(d_tot_tok)})")
    print(f"  Client Wall TPS  : {client_tps:.2f} tok/s")
    print(f"  RERoT Aggregate  : {agg_tps:.2f} tok/s")
    if par_tps > 0:
        print(f"  RERoT Parallel   : {par_tps:.2f} tok/s")
    print(f"  Output snippet   : {content_on[:120]}...")

    print("\n==================================================================")
    print("                      Benchmark Comparison Summary                ")
    print("==================================================================")
    speedup_wall = dur_off / dur_on if dur_on > 0 else 0.0
    speedup_tps = client_tps / tps_off if tps_off > 0 else 0.0
    print(f"Metric                  | RERoT OFF      | RERoT ON (P=6) | Speedup")
    print(f"------------------------+----------------+----------------+---------")
    print(f"Total Wall Time (s)     | {dur_off:14.3f} | {dur_on:14.3f} | {speedup_wall:6.2f}x")
    print(f"Tokens Generated        | {pred_n_off:14d} | {pred_n_on:14d} | -")
    print(f"Client End-to-End TPS   | {(pred_n_off / dur_off):14.2f} | {client_tps:14.2f} | {speedup_tps:6.2f}x")
    if agg_tps > 0:
        print(f"RERoT Aggregate TPS     |              - | {agg_tps:14.2f} | -")
    if par_tps > 0:
        print(f"RERoT Parallel TPS      |              - | {par_tps:14.2f} | -")
    print("==================================================================")

if __name__ == '__main__':
    main()
