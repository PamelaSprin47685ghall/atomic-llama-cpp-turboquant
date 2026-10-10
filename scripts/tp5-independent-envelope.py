#!/usr/bin/env python3
"""Source-counted primitive budget. NEVER a fit to model-request timing.

Reports the weight-only bound, a production-HC/fused-MoE component reference,
and a more complete native-proxy budget separately. A native proxy can be slower
than a fused region; neither sampled service times nor this sum prove a globally
optimal hardware bound. Omitted control/attention/state costs stay omitted.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,importlib.util,json,math,re,statistics
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def positive(v):
    if isinstance(v,bool) or not math.isfinite(v) or v<=0:raise ValueError('Positive finite calibration required')
    return v
def hc_budget(us,sites=94,hoisted=23):
    if not isinstance(sites,int) or not isinstance(hoisted,int) or not 0<=hoisted<=sites:raise ValueError('Invalid source counts')
    for k in ('inject','act','q','norm','lo','up','pack'):positive(us[k])
    return ((sites-hoisted)*us['inject']+sites*(2*us['pack']+us['act']+max(us['q'],us['norm'])+us['lo']+us['up']))/1000
def replace_bytes(total,removed,bandwidth):
    positive(total);positive(bandwidth)
    if removed<0 or removed>total:raise ValueError('Replacement double counts or exceeds source bytes')
    return (total-removed)/(bandwidth*1e6)

def build(root,metadata,layout):
    spec=importlib.util.spec_from_file_location('roof',Path(__file__).with_name('tp5-roofline.py'))
    roof=importlib.util.module_from_spec(spec);spec.loader.exec_module(roof)
    tensors=[t for record in load(metadata)[:6] for t in record['tensors']]
    kinds={t['name']:t['kind'] for t in load(layout)['tensor_layouts']}
    stages=defaultdict(lambda:[0.]*5);groups=defaultdict(list)
    for t in tensors:
        name=re.sub(r'^blk\.\d+\.','',t['name'])
        b=[t['storage_extent_bytes']*roof.selected_fraction(t)*roof.rank_fraction(t,kinds[t['name']],i,False) for i in range(5)]
        m=re.match(r'blk\.(\d+)\.(.*)',t['name']);stage=('tail',0) if not m else (int(m[1]),int('ffn' in m[2]))
        stages[stage]=[x+y for x,y in zip(stages[stage],b)]
        groups[name].append({'name':t['name'],'bytes_by_rank':b,'type':t['raw_type'],'shape':t['shape']})
    total=sum(max(v) for v in stages.values())
    if total!=1869594000:raise ValueError('Re-audit changed source/layout before using these registered counts')
    hc_removed=94*7047680
    records=[]
    for d in ('domain-warm-control','domain-warm-other'):records+=load(root/d/'execution.json')
    hc={}
    for kind in ('inject','act','q','norm','lo','up'):
        for scheme in ('batch0','batch1'):
            rows=[x for x in records if x['name'].startswith(kind+'-') and x['name'].endswith(scheme)]
            if len(rows)!=5 or any(x['returncode'] for x in rows):raise ValueError('Incomplete physical five-card HC calibration')
            hc[kind,scheme]=[positive(x['result']['median_us']) for x in rows]
    pack=load(root/'pack-isolated-check/execution.json')
    pack_times={}
    for arm in (0,1):
        rows=[x for x in pack if x['name'].startswith('guard-') and x['name'].endswith('single'+str(arm))]
        if len(rows)!=5 or any(x['returncode'] for x in rows):raise ValueError('Incomplete pack guard calibration')
        pack_times[arm]=[x['results'][-1]['median_us'] for x in rows]
    moe=[]
    for d in ('moe-warm-control','moe-warm-other'):moe+=load(root/d/'execution.json')
    family_count={22:0,18:0}
    for t in groups['ffn_gate_exps.weight']:family_count[t['type']]+=1
    if family_count!={22:33,18:15}:raise ValueError('Unexpected MoE precision-family counts')
    moe_names=('ffn_gate_exps.weight','ffn_up_exps.weight','ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_gate_inp_shexp.weight')
    def group_bytes(names):return sum(max(t['bytes_by_rank']) for name in names for t in groups[name])
    moe_bytes=group_bytes(moe_names)
    # Native contractions are explicit proxies when production uses larger
    # fused regions. Do not describe their summed service as a proven bound.
    native_groups={
        'router-f32':('ffn_gate_inp.weight',), 'q6-vocab':('output.weight',),
        'gdn-q5-projection':('attn_qkv.weight',), 'gdn-gate-q5':('attn_gate.weight',),
        'q5-output':('ssm_out.weight','attn_output.weight'), 'qsa-q-q5':('attn_q.weight',),
        'qsa-kv-q6':('attn_k.weight','attn_v.weight'), 'moe-down-iq4':('ffn_down_exps.weight',),
        'shared-down-q8':('ffn_down_shexp.weight',),'indexer-bf16':('indexer.q_proj.weight',)}
    native=[]
    for case,names in native_groups.items():
        path=root/'native-warm-calibration'/(case+'-stream.log')
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]
        x=rows[-1];count=sum(len(groups[n]) for n in names)
        native.append({'case':case,'tensor_names':names,'count':count,'removed_bytes':group_bytes(names),
                       'service_us':positive(x['median_us_per_op']),'min_us':positive(x['min_us_per_op']),
                       'service_ms':count*x['median_us_per_op']/1000,'source':str(path),'sha256':digest(path),
                       'qualification':'Isolated native contraction, host graph/sync amortized; not a proof of fused-region latency'})
    native_bytes=sum(x['removed_bytes'] for x in native);native_ms=sum(x['service_ms'] for x in native)
    scenarios=[]
    for scheme in ('batch0','batch1'):
        for pack_single in (0,1):
            for bw in (512,403,390):
                u={k:statistics.median(hc[k,scheme]) for k in ('inject','act','q','norm','lo','up')}
                u['pack']=statistics.median(pack_times[pack_single])
                h_ms=hc_budget(u);moe_ms=0.;mf=[]
                for family,count in (('moe2',33),('moe3',15)):
                    rows=[x for x in moe if x['family']==family and x['stream'] and x['batch_timestamps']==(scheme=='batch1')]
                    if len(rows)!=5 or any(x['returncode'] for x in rows):raise ValueError('Incomplete fused MoE calibration')
                    us=statistics.median(x['result']['median_us'] for x in rows);moe_ms+=count*us/1000
                    mf.append({'family':family,'layers':count,'median_us':us})
                exact_component_ms=h_ms+moe_ms+replace_bytes(total,hc_removed+moe_bytes,bw)
                proxy_ms=h_ms+moe_ms+native_ms+replace_bytes(total,hc_removed+moe_bytes+native_bytes,bw)
                scenarios.append({'timestamp_scheme':scheme,'pack_single':bool(pack_single),'bandwidth_GB_s':bw,
                    'hc_service_us':u,'hc_service_ms':h_ms,'moe_families':mf,'moe_service_ms':moe_ms,
                    'production_component_reference_ms':exact_component_ms,'production_component_reference_tps':1000/exact_component_ms,
                    'native_proxy_budget_ms':proxy_ms,'native_proxy_budget_tps':1000/proxy_ms,
                    'remaining_proxy_weight_ms':replace_bytes(total,hc_removed+moe_bytes+native_bytes,bw)})
    return {'kind':'INDEPENDENT_SOURCE_COUNTED_COMPONENT_BUDGET_NOT_HARDWARE_OPTIMUM',
        'necessity_status':'UNPROVEN_CURRENT_IMPLEMENTATION_COSTS',
        'eligible_as_hardware_latency_lower_bound':False,
        'certifies_optimality_match':False,
        'theory_correction':'Independent measurement is not necessary cost; see docs/performance-necessity.md. No part of this component sum proves the274-to67 hardware-ceiling argument.',
        'frozen_utc':datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'no_whole_model_time_inputs':True,'original_ideal_bytes':total,'original_ideal_tps_512':512e9/total,
        'source_identity':{'metadata_sha256':digest(metadata),'layout_sha256':digest(layout)},
        'hc_sites':94,'assumed_fully_hidden_injects':23,'hc_removed_weight_bytes':hc_removed,
        'fused_moe_removed_weight_bytes':moe_bytes,'native_proxy_removed_weight_bytes':native_bytes,
        'remaining_proxy_weight_bytes':total-hc_removed-moe_bytes-native_bytes,
        'native_proxies':native,'scenarios':scenarios,
        'limits':[
            'This is a component reference, NOT a rigorous silicon ceiling, and not proof of Match.',
            'All HC and fused MoE inputs are model-free GPU fixture measurements; known source shapes/counts replace prior bytes instead of double counting them.',
            'HC assumes local ACT before max(Q,norm), then LO andUP; 23 injects fully hidden; two pack guards per site. These exposed-service assumptions are not a proof of overlap under the whole workload.',
            'Host generation waits are disabled only in standalone fixtures; production wait/sync checks are unchanged. P1 upload and true CPU Y/Q handoff are omitted.',
            'Native-proxy contractions may not match fused attention/down regions; this row is a budget comparison, not a validated lower bound.',
            'Per-dispatch timestamps themselves can perturb scheduling; batch timing includes extra between-dispatch barriers and Q-reset transfer absent from normal CPU re-arm.',
            'Medians across five devices describe calibration, not a proof of the slowest critical rank under whole-model contention.',
            'Residual attention/GDN/PLE/state kernels, exact cache residency, command scheduling, sampling/readback, and first-request graph allocation are not fully accounted.',
            'No arbitrary relay constant, global residual, measured model graph_compute or inverse speed scaling is fitted into this budget.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','metadata','layout','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Refuse to overwrite a frozen reference')
    try:
        d=build(a.root,a.metadata,a.layout);a.out.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in d.items() if k not in ('native_proxies','scenarios')},indent=2))
        for x in d['scenarios']:print(x['timestamp_scheme'],x['pack_single'],x['bandwidth_GB_s'],x['production_component_reference_tps'],x['native_proxy_budget_tps'])
    except (ValueError,KeyError,OSError) as e:p.error(str(e))
