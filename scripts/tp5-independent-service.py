#!/usr/bin/env python3
"""Independent model-free service calibration. No derived value is a model tok/s."""
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
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--micro',type=Path,required=True)
    p.add_argument('--relay',type=Path,required=True)
    p.add_argument('--relay-elements',type=int,nargs='*',default=[256,2560,4096])
    p.add_argument('--cases',nargs='*',help='Named subset for a separately recorded experiment')
    p.add_argument('--warm-ms',type=int,default=0)
    p.add_argument('--production-queue',action='store_true')
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'))
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ}
    env.update(RADV_DEBUG='nobolist',GGML_VK_DISABLE_MMVQ='1',GGML_VK_CMD_REPLAY='1')
    if not 0<=a.warm_ms<=1000:p.error('Warmup must be0..1000ms')
    env['TP5_SERVICE_WARM_MS']=str(a.warm_ms)
    if a.production_queue:env['GGML_VK_ALLOW_GRAPHICS_QUEUE']='1'
    cases=[('q6-vocab',14,2560,49664,1),('q8-hc-down',8,10240,320,1),('q8-hc-up',8,320,10240,1),
           ('router-f32',0,2560,512,1),('gdn-q5-projection',13,2560,3840,1),('q5-output',13,1280,2560,1),
           ('moe-gate-iq2',22,2560,128,10),('moe-up-iq3',18,2560,128,10),('moe-down-iq4',20,128,2560,10),
           ('gdn-gate-q5',13,2560,1280,1),('qsa-q-q5',13,2560,2560,1),('qsa-kv-q6',14,2560,512,1),
           ('shared-down-q8',8,128,2560,1),('indexer-bf16',30,2560,512,1)]
    if a.cases is not None:
        if not set(a.cases) <= {c[0] for c in cases}:p.error('unknown kernel case')
        cases=[c for c in cases if c[0] in a.cases]
    jobs=[(f'relay-aligned-{n}',[str(a.relay),'--run','--stages','96','--replays','16','--elements',str(n)]) for n in a.relay_elements]
    jobs += [(name+'-'+mode,[str(a.micro),str(t),str(k),str(m),str(b),'256',mode])
             for name,t,k,m,b in cases for mode in ('resident','stream')]
    a.root.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,cmd in jobs:
        path=a.root/(name+'.log')
        if path.exists():raise RuntimeError('Refusing to replace a prior experiment: '+str(path))
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # GPU busy counters can lag a completed process. Wait for the same
            # existing idle gate, never weaken it or perform hardware recovery.
            deadline=time.monotonic()+15
            while True:
                try:
                    before=bench.gpu_state()
                    if len(before)!=5 or any(int(g['gpu_busy_percent'])!=0 or int(g['mem_info_vram_used'])>=128*1024*1024 for g in before):
                        raise RuntimeError('Existing five-card idle/VRAM gate not yet satisfied')
                    # Confirm a second idle observation; busy counters may only
                    # update after the first read following process teardown.
                    time.sleep(.5)
                    again=bench.gpu_state()
                    if len(again)!=5 or any(int(g['gpu_busy_percent'])!=0 or int(g['mem_info_vram_used'])>=128*1024*1024 for g in again):
                        raise RuntimeError('Idle state has not settled')
                    break
                except RuntimeError:
                    if time.monotonic()>=deadline:raise
                    time.sleep(.5)
            print('BEGIN',name,flush=True)
            with path.open('xb') as log:
                start=time.monotonic();rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            row={'name':name,'argv':cmd,'env':env,'before':before,'returncode':rc,'wall_seconds':time.monotonic()-start}
            results.append(row)
            (a.root/'service-execution.json').write_text(json.dumps(results,indent=2)+'\n')
            if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
            print('DONE',name,flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
