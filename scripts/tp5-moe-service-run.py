#!/usr/bin/env python3
"""Run independent production fused MoE fixtures, never whole-model timing."""
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
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--warm-ms',type=int,default=0);a=p.parse_args()
    if not 0<=a.warm_ms<=1000:p.error('Warmup must be0..1000ms; no hardware clock writes')
    if len(set(a.devices))!=len(a.devices) or any(d<0 or d>4 for d in a.devices):p.error('Unique physical GPU indices0..4 required')
    a.root.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    done=[]
    for device in a.devices:
        for kind in ('moe2','moe3'):
            for stream in (0,1):
                for batch in (0,1):
                    name=f'{kind}-d{device}-stream{stream}-batch{batch}'
                    env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ}
                    env.update(RADV_DEBUG='nobolist',TP5_TEST_BATCH_TIMING=str(batch),TP5_TEST_STREAM_WEIGHTS=str(stream),TP5_TEST_WARM_MS=str(a.warm_ms))
                    cmd=[str(a.bin),str(a.shaders),kind,'local',str(device)]
                    with open('/var/tmp/tp5-gpu.lock','a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        deadline=time.monotonic()+20;streak=0
                        while streak<2:
                            state=b.gpu_state();good=len(state)==5 and all(int(x['gpu_busy_percent'])==0 and int(x['mem_info_vram_used'])<128*1024*1024 for x in state)
                            streak=streak+1 if good else 0
                            if time.monotonic()>deadline:raise RuntimeError('Physical five-card idle gate failed')
                            time.sleep(.5)
                        print('BEGIN',name,flush=True)
                        path=a.root/(name+'.log')
                        with path.open('xb') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
                        result=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{') and 'INDEPENDENT_FUSED_MOE_SERVICE' in l]
                        shader='qwen4_moe_projections_'+('iq2s_iq3xxs' if kind=='moe2' else 'iq3xxs_iq3s')+'.spv'
                        item={'name':name,'device':device,'family':kind,'stream':bool(stream),'batch_timestamps':bool(batch),
                              'argv':cmd,'env':env,'returncode':rc,'binary_sha256':b.sha(a.bin),
                              'shader_sha256':b.sha(a.shaders/shader),'result':result[-1] if result else None}
                        done.append(item);(a.root/'execution.json').write_text(json.dumps(done,indent=2)+'\n')
                        if rc or not result:print('STOP_ON_FAILURE',name,rc,flush=True);return rc or 1
    return 0

if __name__=='__main__':raise SystemExit(main())
