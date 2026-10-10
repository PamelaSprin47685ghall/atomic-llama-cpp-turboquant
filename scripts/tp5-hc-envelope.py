#!/usr/bin/env python3
"""Independent HC critical-path service reference, never a fitted whole-model bound.

Only source structure, tensor sizes, hardware-bandwidth assumptions and isolated
shader measurements enter. No model request duration is read. Missing costs
remain explicit. This is not an architecture-independent speed-limit proof.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

def last_json(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    if not rows or not rows[-1].get('all_outputs_checked'):
        raise ValueError(f'Unverified shader result: {path}')
    return rows[-1]

def hc_ms(s, sites=94, hidden_injects=23):
    if not 0<=hidden_injects<=sites:raise ValueError('Invalid independent site count')
    if any(not math.isfinite(x) or x<0 for x in s.values()):raise ValueError('Invalid service interval')
    # ACT completes before norm can overwrite z_p. Only Q and norm are parallel.
    return ((sites-hidden_injects)*s['inject']+
            sites*(s['act']+max(s['q'],s['norm'])+s['lo']+s['up']))/1000

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--weight-reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Do not overwrite a frozen reference')
    shards=json.loads(a.metadata.read_text());tensor={t['name']:t for shard in shards[:6] for t in shard['tensors']}
    names=['blk.0.hc_attn_'+x+'.weight' for x in ('down','up','inject','norm')]
    one_site=sum(tensor[n]['storage_extent_bytes'] for n in names)
    if [tensor[n]['raw_type'] for n in names]!=[8,8,8,0]:p.error('Reference HC storage no longer matches Q8/F32')
    ref=json.loads(a.weight_reference.read_text())
    base=next(x for x in ref['scenarios'] if x['mode']=='no-MTP sharded attention')
    native_bytes=base['sum_stage_max_target_bytes']-94*one_site
    if native_bytes<=0:p.error('Inconsistent removed weight inventory')
    sources={
      'inject':a.root/'late-services/inject-local.log',
      'act':a.root/'late-services/act-local.log',
      'up':a.root/'late-services/up-local.log',
      'q_local':a.root/'late-services/q-local.log',
      'q_system':a.root/'late-services/q-system.log',
      'norm_local':a.root/'norm-validated-wg256-rows1.log',
      'norm_system':a.root/'norm-geometry-wg256-system.log',
      'lo_local':a.root/'lo-local.log',
      'lo_system':a.root/'lo-system.log'}
    values={k:last_json(v)['median_us'] for k,v in sources.items()}
    rows=[]
    for domain in ('local','system'):
        service={k:values[k] for k in ('inject','act','up')}
        service.update({k:values[k+'_'+domain] for k in ('q','norm','lo')})
        h=hc_ms(service)
        for bw in (512,403,390):
            native=native_bytes/(bw*1e6)
            rows.append({'mailbox_fixture':domain,'bandwidth_GB_s':bw,'HC_service_ms':h,
                         'native_weight_only_ms':native,'reference_ms':h+native,
                         'reference_tps':1000/(h+native),'component_median_us':service})
    local=rows[0]['component_median_us']
    q64={**local,'q':last_json(a.root/'qgeometry-b1-wg64-local.log')['median_us']}
    norm512={**local,'norm':last_json(a.root/'norm-validated-wg512-rows1.log')['median_us']}
    result={
      'kind':'INDEPENDENT_HC_EXPANDED_OPTIMISTIC_SERVICE_REFERENCE',
      'frozen_utc':datetime.now(timezone.utc).isoformat(),
      'whole_model_times_used':False,'is_hardware_upper_bound_proof':False,
      'original_weight_reference_tps':base['conditional_tokens_per_second'],
      'old_logical_weight_bytes':base['sum_stage_max_target_bytes'],
      'replaced_HC_sites':94,'assumed_fully_hidden_injects':23,
      'removed_weight_bytes_per_site':one_site,'remaining_native_weight_bytes':native_bytes,
      'equation':'(94-23)*inject +94*(ACT + max(Q,norm) +LO +UP), plus remaining native weight resource time',
      'scenarios':rows,
      'local_optimistic_component_savings_ms':{'Q64':hc_ms(local)-hc_ms(q64),'norm512':hc_ms(local)-hc_ms(norm512)},
      'omitted_costs':['unmatched native instruction/dequantization and small-op service',
                       'KV/recurrent/activation traffic not in isolated HC fixtures',
                       'cross-rank ready/payload/integrity latency and pipeline fill',
                       'CPU sampling/driver/request overhead and first graph allocation',
                       'extra barriers between independently timed shaders'],
      'limitations':['Local/system fixtures are sensitivity cases, not proven brackets for production memory placement.',
                     'Independent medians are attainable component costs, not universal minimum latency constants.',
                     'Q/norm overlap and complete hiding of23 injects are optimistic assumptions, not observed full overlap.',
                     'Changing kernels, fusion, queues or memory policies requires recomputation.',
                     'This reference tightens the model by explicit missing work; it does not establish global optimality or Match.'],
      'sources':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in sources.items()}}
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
