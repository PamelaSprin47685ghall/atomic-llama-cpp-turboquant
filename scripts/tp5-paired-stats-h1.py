#!/usr/bin/env python3
"""Paired-block statistics for the h0-vs-h1 Q2_0 confirmation.

Reads pair-{1..8}-{A-h0,B-h1} result.json files, verifies per-block body
identity between arms (greedy MTP must reproduce greedy tokens), pools
measured tokens/ms per process, and reports per-block tps, the geometric
mean ratio h1/h0 with a 95% CI (log-space, t-distribution), and mechanical
gate checks (natural stop, exit codes, no fallback in measured phase).

No speed/semantic qualification is inferred beyond what the raw evidence
shows; promotion decisions stay with DECISION.md.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'artifacts/perf-gap/frontier-20261010T000000Z'


def load_arm(prefix, block, arm):
    folder = ROOT / f'{prefix}{block}-{arm}'
    result = json.loads((folder / 'result.json').read_text())
    toks = 0
    ms = 0.0
    bodies = []
    for r in result['requests']:
        if r['phase'] != 'measured':
            continue
        raw = json.loads((folder / f"request-{r['idx']}.json").read_text())
        choice = raw['response']['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ValueError(f'pair-{block}-{arm}: non-stop finish in measured request')
        if r['issues']:
            raise ValueError(f'pair-{block}-{arm}: issues in measured request: {r["issues"]}')
        toks += raw['response']['usage']['completion_tokens']
        ms += raw['response']['timings']['predicted_ms']
        bodies.append(choice['message']['content'])
    if toks == 0 or ms <= 0:
        raise ValueError(f'pair-{block}-{arm}: empty measured phase')
    rc = result['shutdown']['returncode']
    forced = result['shutdown']['forced']
    return {'tokens': toks, 'ms': ms, 'tps': toks * 1000.0 / ms,
            'bodies': bodies, 'returncode': rc, 'forced': forced}


def t_quantile_95(df):
    # Two-sided 95% t quantile for small df (df = n-1, n = block count).
    table = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
             9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}
    return table.get(df, 2.0)


def main():
    blocks = []
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'pair-'
    blocks = []
    for b in range(1, n + 1):
        arm_a = sys.argv[3] if len(sys.argv) > 3 else 'A-h0'
        arm_b = sys.argv[4] if len(sys.argv) > 4 else 'B-h1'
        a = load_arm(prefix, b, arm_a)
        c = load_arm(prefix, b, arm_b)
        if a['bodies'] != c['bodies']:
            raise ValueError(f'pair-{b}: h0/h1 measured bodies differ (MTP verification mismatch)')
        for arm in (a, c):
            if arm['returncode'] != 0 or arm['forced']:
                raise ValueError(f"pair-{b}: abnormal teardown rc={arm['returncode']} forced={arm['forced']}")
        blocks.append((b, a, c))
    logs = []
    for b, a, c in blocks:
        ratio = c['tps'] / a['tps']
        logs.append(math.log(ratio))
        print(f"block {b}: h0={a['tps']:.4f} tok/s ({a['tokens']} tok)  "
              f"h1={c['tps']:.4f} tok/s ({c['tokens']} tok)  ratio={ratio:.5f}")
    mean = sum(logs) / len(logs)
    if len(logs) > 1:
        var = sum((x - mean) ** 2 for x in logs) / (len(logs) - 1)
        sd = math.sqrt(var)
        tq = t_quantile_95(len(logs) - 1)
        half = tq * sd / math.sqrt(len(logs))
    else:
        half = float('nan')
    lo = math.exp(mean - half) if half == half else float('nan')
    hi = math.exp(mean + half) if half == half else float('nan')
    print(f"\ngeometric mean ratio h1/h0: {math.exp(mean):.5f}  "
          f"95% CI [{lo:.5f}, {hi:.5f}]  (n={len(blocks)} paired blocks)")
    pooled_h0 = sum(a['tokens'] for _, a, _ in blocks) * 1000.0 / sum(a['ms'] for _, a, _ in blocks)
    pooled_h1 = sum(c['tokens'] for _, _, c in blocks) * 1000.0 / sum(c['ms'] for _, _, c in blocks)
    print(f"pooled: h0={pooled_h0:.4f}  h1={pooled_h1:.4f} tok/s")


if __name__ == '__main__':
    main()
