#!/usr/bin/env python3
"""Validate both production norm shaders and compare ready-path service."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','shaders'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--variant',type=int,choices=(1,2),default=1)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    s=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'))
    b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
    cases=[]
    # Full signed-output and withheld-generation contracts in all3 domains.
    for domain in ('local','system','production'):
        for rows in (1,2,4):
            for failed in (0,1):
                cases.append((0,domain,rows,a.variant,failed,0))
    # Same warmed, chain-timed five-card service for both modules.
    for d in range(5):
        for on in (0,a.variant):cases.append((d,'production',1,on,0,200))
    done=[]
    for d,domain,rows,on,failed,warm in cases:
        name=f'd{d}-{domain}-r{rows}-stage{on}-fail{failed}-warm{warm}'
        env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ}
        env.update(RADV_DEBUG='nobolist',TP5_TEST_NORM_STAGE_Y=str(on),
                   TP5_TEST_NORM_WITHHOLD=str(failed),TP5_TEST_WARM_MS=str(warm),TP5_TEST_BATCH_TIMING='1')
        cmd=[str(a.bin),str(a.shaders),'norm',domain,str(d),'256',str(rows)]
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            b.idle_gate()
            print('BEGIN',name,flush=True)
            path=a.root/(name+'.log')
            with path.open('xb') as out:rc=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
            parsed=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
            variants={0:'tp5_hc_resume_norm.spv',1:'tp5_hc_resume_norm_stage_y.spv',2:'tp5_hc_norm_shared.spv'}
            shader=variants[on]
            done.append({'name':name,'device':d,'domain':domain,'rows':rows,'staged':bool(on),'withheld':bool(failed),
                         'warm_ms':warm,'returncode':rc,'argv':cmd,'env':env,'binary_sha256':b.sha(a.bin),
                         'shader_sha256':b.sha(a.shaders/shader),'result':parsed[-1] if parsed else None})
            (a.root/'execution.json').write_text(json.dumps(done,indent=2)+'\n')
            if rc or not parsed:print('STOP_ON_FAILURE',name,rc,flush=True);return rc or 1
    return 0

if __name__=='__main__':raise SystemExit(main())
