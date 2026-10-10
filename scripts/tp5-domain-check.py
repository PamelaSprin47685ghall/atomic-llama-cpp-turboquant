#!/usr/bin/env python3
"""Freeze independent shader timing by actual mailbox domain and timestamp scheme."""
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
    for k in ('root','micro','shaders'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--warm-ms',type=int,default=0)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2,3,4])
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if not 0<=a.warm_ms<=1000 or len(set(a.devices))!=len(a.devices) or any(d<0 or d>4 for d in a.devices):p.error('Bounded warmup and unique GPU0..4 indices required')
    spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    jobs=[]
    for device in a.devices:
        for kind,domain in [('inject','local'),('act','local'),('q','production'),('norm','production'),('lo','production'),('up','local'),('packnoop','local')]:
            for batch in (0,1):
                jobs.append((f'{kind}-{domain}-d{device}-batch{batch}',kind,domain,device,batch))
    completed=[]
    for name,kind,domain,device,batch in jobs:
        env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ}
        env.update(RADV_DEBUG='nobolist',TP5_TEST_BATCH_TIMING=str(batch),TP5_TEST_WARM_MS=str(a.warm_ms))
        cmd=[str(a.micro),str(a.shaders),kind,domain,str(device)]
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            deadline=time.monotonic()+20;streak=0
            while streak<2:
                state=b.gpu_state();good=len(state)==5 and all(int(x['gpu_busy_percent'])==0 and int(x['mem_info_vram_used'])<128*1024*1024 for x in state)
                streak=streak+1 if good else 0
                if time.monotonic()>deadline:raise RuntimeError('Five-card idle gate did not settle')
                time.sleep(.5)
            print('BEGIN',name,flush=True)
            with (a.root/(name+'.log')).open('xb') as log:
                rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            row={'name':name,'argv':cmd,'env':env,'before':state,'returncode':rc,'binary_sha256':b.sha(a.micro)}
            results=[json.loads(l) for l in (a.root/(name+'.log')).read_text().splitlines() if l.startswith('{')]
            if results:row['result']=results[-1]
            completed.append(row);(a.root/'execution.json').write_text(json.dumps(completed,indent=2)+'\n')
            if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
