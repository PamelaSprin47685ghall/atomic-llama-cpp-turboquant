#!/usr/bin/env python3
"""Conditional TP5 weight-streaming roofline, NOT a hardware benchmark.

Uses actual GGUF extents, 10/512 routed experts, HC/router replication,
five-rank QSA/GDN geometry from llama-tp5-plan.cpp, and single-GPU draft.
Kernel launch, dequantization, KV/recurrent operations and communication are
not included. Bytes are logical source bytes at an assumed cold weight boundary.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re


def selected_fraction(t):
    n = t['name']
    if n == 'per_layer_token_embd.weight':
        return 16 / t['shape'][-1]
    if n == 'token_embd.weight':
        return 1 / t['shape'][-1]
    if re.search(r'_exps?\.weight$', n):
        return 10 / t['shape'][-1]
    return 1.0


def rank_fraction(t, kind, rank, replicate):
    if kind == 'cpu':
        return 0.0
    if kind == 'mirror':
        return 1.0
    if kind in ('split_axis1_vocab_rows', 'split_axis0_channels', 'split_axis1_rows'):
        return 0.2
    if replicate:
        return 1.0
    base = re.sub(r'^blk\.\d+\.', '', t['name'])
    v = (10, 10, 10, 10, 8)[rank]
    q = (5, 5, 5, 5, 4)[rank]
    kv = (1, 1, 2, 1, 1)[rank]
    if kind == 'split_axis1_vhead_rows':
        if base in ('attn_qkv.weight', 'ssm_conv1d.weight'):
            return 3 * 128 * v / t['shape'][1]
        return v / 48
    if kind in ('split_axis1_vhead', 'split_axis0_vhead_cols'):
        return v / 48
    if kind == 'split_axis1_head_rows_replicated':
        return kv / 2
    if kind in ('split_axis1_head_rows', 'split_axis0_head_cols'):
        return q / 24
    raise ValueError(f'Unaccounted layout {kind}: {t["name"]}')


def estimate(target, draft, layout, bandwidth_gbs, committed):
    if not math.isfinite(bandwidth_gbs) or bandwidth_gbs <= 0 or not 1 <= committed <= 2:
        raise ValueError('Positive bandwidth and h1 committed in [1,2] required')
    kinds = {t['name']: t['kind'] for t in layout['tensor_layouts']}
    tensors = [t for r in target for t in r['tensors']]
    drafter = [t for r in draft for t in r['tensors']]
    scenarios = []
    for replicate, mtp in ((True, False), (False, False), (False, True)):
        # h1 best/worst expert overlap across the two verification rows.
        for expert_union in ((1, 2) if mtp else (1,)):
            ranks = [0.0] * 5
            stages = defaultdict(lambda: [0.0] * 5)
            for t in tensors:
                kind = kinds.get(t['name'])
                if kind is None:
                    raise ValueError(f'No source-plan layout for {t["name"]}')
                active = t['storage_extent_bytes'] * selected_fraction(t)
                if re.search(r'_exps?\.weight$', t['name']):
                    active *= expert_union
                layer = re.match(r'blk\.(\d+)\.(.*)', t['name'])
                stage = ('tail', 0) if layer is None else (int(layer.group(1)), int('ffn' in layer.group(2)))
                for r in range(5):
                    n = active * rank_fraction(t, kind, r, replicate)
                    ranks[r] += n
                    stages[stage][r] += n
            target_bytes = sum(max(v) for v in stages.values())
            draft_bytes = sum(t['storage_extent_bytes'] * selected_fraction(t) for t in drafter) if mtp else 0
            latency_ms = (target_bytes + draft_bytes) / (bandwidth_gbs * 1e6)
            k = committed if mtp else 1.0
            label = 'MTP1 sharded attention' if mtp else 'no-MTP ' + ('replicated' if replicate else 'sharded') + ' attention'
            scenarios.append({'mode': label,
                'verify_expert_union_multiplier': expert_union, 'target_logical_bytes_by_rank': ranks,
                'sum_stage_max_target_bytes': target_bytes, 'draft_logical_bytes': draft_bytes,
                'ideal_target_ms': target_bytes / (bandwidth_gbs * 1e6),
                'ideal_draft_ms': draft_bytes / (bandwidth_gbs * 1e6), 'ideal_cycle_ms': latency_ms,
                'assumed_committed_per_cycle': k, 'conditional_tokens_per_second': k * 1000 / latency_ms})
    return {'status': 'CONDITIONAL_THEORY_NOT_MEASUREMENT', 'bandwidth_GB_s_per_rank': bandwidth_gbs,
        'assumptions': ['HC, routers and other mirrored weights are read independently by every rank.',
            'QSA heads 5/5/5/5/4, KV 1/1/2/1/1, GDN V heads 10/10/10/10/8; Q/K duplication included.',
            'Each required weight block crosses the assumed weight-streaming boundary once per cycle.',
            'Two target rows reuse non-expert weights; expert union ranges from 10 to 20 distinct experts.',
            'Indexed PLE table is not read in full; its CPU lookup cost is omitted.',
            'Catch-up, KV/recurrent state, activation traffic, instruction throughput and synchronization are omitted.',
            'This is not an absolute architecture-independent speed limit or a prediction that this code reaches it.'],
        'scenarios': scenarios}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--target-shards', type=int, required=True)
    p.add_argument('--draft-last', action='store_true', help='Only final record of --draft file is the drafter')
    p.add_argument('--layout', type=Path, default=Path('tp5-manifest.json'))
    p.add_argument('--bandwidth', type=float, default=512)
    p.add_argument('--committed', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    target = json.loads(a.target.read_text())[:a.target_shards]
    draft = json.loads(a.draft.read_text())
    if a.draft_last:
        draft = draft[-1:]
    result = estimate(target, draft, json.loads(a.layout.read_text()), a.bandwidth, a.committed)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
