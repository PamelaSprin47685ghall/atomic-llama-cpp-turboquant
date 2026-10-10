#!/usr/bin/env python3
"""Verify all raw A/B responses and identities for the registered match campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def load(p):
    return json.loads(p.read_text())

def audit(root):
    pairs = [(f'greedy-b{i}-A', f'greedy-b{i}-B') for i in range(1,5)]
    pairs += [(f'heldout-{name}-A', f'heldout-{name}-B') for name in ('index','process','debug')]
    identities = set()
    checked, measured, reports = 0, 0, []
    for an, bn in pairs:
        per_arm = []
        environments = []
        for name, policy in ((an,'0'),(bn,'1')):
            folder = root/name
            r, m = load(folder/'result.json'), load(folder/'manifest.json')
            if r['status']!='MEASURED_NOT_QUALIFIED' or r['profile']!='none' or r['shutdown']['returncode'] or r['shutdown']['forced']:
                raise ValueError(f'Non-normal/unmeasured execution: {name}')
            if any('--rerot' in arg for arg in m['argv']) or m['args']['horizon'] != 0:
                raise ValueError('Wrong RERoT/MTP scope')
            identities.add(json.dumps({'bin':m['bin_sha256'], 'dso':load(folder/'mapped-libraries.json'),
                                       'model':m['model_stat_identity']},sort_keys=True))
            env = dict(m['env'])
            if env.pop('GGML_TP5_GREEDY_HOST') != policy:
                raise ValueError('Wrong arm policy')
            environments.append(env)
            log = (folder/'server.log').read_text(errors='replace')
            hits = log.count('ordinary-greedy host sampling enabled')
            if hits != (len(r['requests']) if policy=='1' else 0):
                raise ValueError(f'Unexpected eligibility coverage: {name}, hits={hits}')
            requests = []
            for q in r['requests']:
                raw = load(folder/f"request-{q['idx']}.json")
                rsp=raw['response']; choice=rsp['choices'][0]; body=choice['message']['content']
                sha=hashlib.sha256(body.encode()).hexdigest()
                n=rsp['usage']['completion_tokens']; ms=rsp['timings']['predicted_ms']
                if (sha!=q['content_sha256'] or n!=q['n'] or n!=rsp['timings']['predicted_n'] or
                        ms!=q['predicted_ms'] or not math.isfinite(ms) or ms<=0 or
                        raw['phase']!=q['phase'] or choice['finish_reason']!='stop' or q['issues']):
                    raise ValueError(f'Raw response mismatch: {name}/{q["idx"]}')
                if not math.isclose(n*1000/ms,q['tps'],rel_tol=1e-12):
                    raise ValueError('Incorrect reported throughput')
                if q['phase']=='measured':
                    if q['fallback_lines']: raise ValueError(f'Measured fallback: {name}')
                    measured+=1
                checked+=1
                requests.append({'idx':q['idx'],'phase':q['phase'],'payload':raw['payload'],
                                 'tokens':n,'body_sha256':sha})
            per_arm.append(requests)
        if environments[0] != environments[1] or per_arm[0]!=per_arm[1]:
            raise ValueError(f'A/B config or raw content differs: {an}/{bn}')
        reports.append({'A':an,'B':bn,'raw_body_and_request_match':True,'requests_per_arm':len(per_arm[0]),
                        'measured_hashes':sorted({q['body_sha256'] for q in per_arm[0] if q['phase']=='measured'})})
    if len(identities)!=1: raise ValueError('Executable/DSO/model identity changed')
    return {'status':'RAW_AUDIT_PASS','raw_requests':checked,'measured_requests':measured,
            'normal_process_exits':2*len(pairs),'measured_fallbacks':0,'identity_count':1,'pairs':reports,
            'limits':'Exact equality to the contemporaneous baseline, not independent comprehensive model-quality certification. Original failed screen is not erased.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Audit output already exists')
    try:
        report=audit(a.root)
        a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(report,indent=2,ensure_ascii=False))
    except (ValueError,KeyError,OSError) as e:p.error(str(e))
