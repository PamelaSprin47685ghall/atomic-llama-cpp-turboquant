#!/usr/bin/env python3
"""Summarize raw warm GPU samples; do not force agreement with server latency."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import re
import statistics

def summarize(root):
    spans=defaultdict(list)
    stages=defaultdict(list)
    for name in ['warm-span']+[f'warm-8-{i}' for i in range(8)]:
        directory=root/name
        result=json.loads((directory/'result.json').read_text())
        if result['status']=='FAILED' or result['shutdown']['returncode'] or result['shutdown']['forced']:
            raise ValueError(f'Incomplete diagnostic {name}')
        latest={}
        for line in (directory/'server.log').read_text().splitlines():
            if not line.startswith('[tp5-gpu-sparse'):continue
            f={k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)}
            rank=int(f['rank'])
            if line.startswith('[tp5-gpu-sparse]'):
                latest[rank]=int(f['observed_chain'])
                spans[(name,rank)].append((latest[rank],f['span_us']))
            else:
                stage=int(f.get('stage',96))
                if latest[rank]>400:
                    stages[(stage,rank)].append({k:f.get(k,0.0) for k in ('p2_us','compute_us','p1_us','late_pre_us')})
    reports=[]
    for (name,rank),samples in sorted(spans.items()):
        if [s[0] for s in samples]!=list(range(400,416)):raise ValueError('Incomplete query window')
        reports.append({'run':name,'rank':rank,'first_us':samples[0][1],
                        'steady_median_us':statistics.median(s[1] for s in samples[1:]),
                        'steady_min_us':min(s[1] for s in samples[1:]),'steady_max_us':max(s[1] for s in samples[1:])})
    if set(stages)!={(s,r) for s in range(97) for r in range(5)}:raise ValueError('Missing stage or rank; not filling with zero')
    rows=[];totals=defaultdict(lambda:defaultdict(float))
    for (stage,rank),samples in sorted(stages.items()):
        if len(samples)!=15:raise ValueError('Expected15 steady samples per stage and rank')
        row={'stage':stage,'rank':rank,'samples':15}
        for k in samples[0]:
            row[k]=statistics.median(s[k] for s in samples)
            totals[str(rank)][k]+=row[k]
        rows.append(row)
    return {'kind':'SPARSE_STEADY_GPU_ATTRIBUTION_NOT_RESOURCE_BOUND','span_windows':reports,
            'stage_medians':rows,'component_sums_by_rank':dict(totals),
            'limitations':['Selected timestamp boundaries can perturb local scheduling.',
                'Summed medians across separately measured partitions are not one simultaneously measured path.',
                'Ranks overlap; do not add their times. P2 includes waits and consumers, not just transport.',
                'The first captured replay is reported separately, not silently discarded.',
                'No normalization to force the sum to the server or graph_compute time.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    data=summarize(a.root)
    a.out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='stage_medians'},indent=2))
