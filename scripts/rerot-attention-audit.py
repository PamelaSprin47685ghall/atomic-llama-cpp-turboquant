#!/usr/bin/env python3
"""Measure per-reader attention distributions from an audit dump.

Use F16 KV for this diagnostic so the offline reconstruction has no Turbo WHT
or quantized dot-product confound. The model still runs the production indexed
RERoT attention path; only the observation cache dtype is changed explicitly.
"""
import argparse
import math
from pathlib import Path
import struct


def load_meta(directory: Path):
    result = {}
    for line in (directory / 'tensors.tsv').read_text().splitlines():
        fields = line.split('\t')
        name, typ = fields[:2]
        nums = list(map(int, fields[2:]))
        result[name] = {'type': typ, 'ne': nums[:4], 'nb': nums[4:8], 'bytes': nums[8],
                        'raw': (directory / f'{name}.bin').read_bytes()}
    return result


def scalar(t, x, y=0, z=0, w=0):
    off = x*t['nb'][0] + y*t['nb'][1] + z*t['nb'][2] + w*t['nb'][3]
    typ = t['type']
    if typ == 'f32':
        return struct.unpack_from('<f', t['raw'], off)[0]
    if typ == 'f16':
        return struct.unpack_from('<e', t['raw'], off)[0]
    if typ == 'i32':
        return struct.unpack_from('<i', t['raw'], off)[0]
    raise ValueError(f'unsupported dtype {typ}; rerun audit with --cache f16')


def jsd(p, q):
    keys = set(p) | set(q)
    m = {k: 0.5*(p.get(k, 0.0)+q.get(k, 0.0)) for k in keys}
    def kl(a):
        return sum(v*math.log(v/m[k]) for k, v in a.items() if v > 0 and m[k] > 0)
    return 0.5*(kl(p)+kl(q))


def cosine(p, q):
    keys = set(p) | set(q)
    dot = sum(p.get(k, 0.0)*q.get(k, 0.0) for k in keys)
    pn = math.sqrt(sum(v*v for v in p.values()))
    qn = math.sqrt(sum(v*v for v in q.values()))
    return dot/(pn*qn) if pn and qn else 0.0


def analyse(directory: Path):
    t = load_meta(directory)
    q, k, entries, offsets = t['src0'], t['src1'], t['src3'], t['src4']
    if q['type'] != 'f32' or k['type'] not in ('f16', 'f32'):
        raise ValueError('this direct reconstruction requires F32 Q and F16/F32 K')
    dk, groups, hq, _ = q['ne']
    _, nkeys, hkv, _ = k['ne']
    nqueries = offsets['ne'][0]-1
    if hq % hkv:
        raise ValueError('invalid GQA head mapping')
    raw_params = (directory / 'op-params.bin').read_bytes()
    scale = struct.unpack_from('<f', raw_params, 0)[0]
    softcap = struct.unpack_from('<f', raw_params, 8)[0]
    if softcap:
        inner_scale = scale/softcap
    else:
        inner_scale = scale
    distributions = []
    top_keys = []
    entropies = []
    for query in range(nqueries):
        begin = int(scalar(offsets, query))
        end = int(scalar(offsets, query+1))
        aggregate = {}
        tops = []
        head_ent = []
        for head in range(hq):
            kh = head // (hq//hkv)
            scores = []
            for ie in range(begin, end):
                key_idx = int(scalar(entries, 0, ie))
                group_idx = int(scalar(entries, 1, ie))
                if not (0 <= key_idx < nkeys and 0 <= group_idx < groups):
                    raise ValueError('invalid indexed attention descriptor')
                dot = 0.0
                for d in range(dk):
                    dot += scalar(q, d, group_idx, head) * scalar(k, d, key_idx, kh)
                score = dot*inner_scale
                if softcap:
                    score = softcap*math.tanh(score)
                scores.append((key_idx, score))
            max_score = max(score for _, score in scores)
            weights = [(idx, math.exp(score-max_score)) for idx, score in scores]
            total = sum(weight for _, weight in weights)
            probs = [(idx, weight/total) for idx, weight in weights]
            tops.append(max(probs, key=lambda x:x[1])[0])
            head_ent.append(-sum(p*math.log(max(p, 1e-300)) for _, p in probs))
            for idx, p in probs:
                aggregate[idx] = aggregate.get(idx, 0.0)+p/hq
        distributions.append(aggregate)
        top_keys.append(tops)
        entropies.append(sum(head_ent)/len(head_ent))
    pair_js = []
    pair_cos = []
    top_overlap = []
    for i in range(nqueries):
        for j in range(i+1, nqueries):
            pair_js.append(jsd(distributions[i], distributions[j]))
            pair_cos.append(cosine(distributions[i], distributions[j]))
            top_overlap.append(sum(a == b for a, b in zip(top_keys[i], top_keys[j]))/hq)
    print(f'{directory}: queries={nqueries} groups={groups} keys={nkeys} heads={hq}')
    print('  entropy_mean=', sum(entropies)/len(entropies))
    if pair_js:
        print('  pair_jsd_mean=', sum(pair_js)/len(pair_js), 'min=', min(pair_js), 'max=', max(pair_js))
        print('  pair_cosine_mean=', sum(pair_cos)/len(pair_cos), 'min=', min(pair_cos), 'max=', max(pair_cos))
        print('  top_key_overlap_mean=', sum(top_overlap)/len(top_overlap), 'min=', min(top_overlap), 'max=', max(top_overlap))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='one attention-N directory or its tensor-audit parent')
    args = parser.parse_args()
    paths = [args.path] if (args.path/'tensors.tsv').exists() else sorted(args.path.glob('attention-*'))
    if not paths:
        raise SystemExit('no attention dump found')
    for path in paths:
        analyse(path)


if __name__ == '__main__':
    main()
