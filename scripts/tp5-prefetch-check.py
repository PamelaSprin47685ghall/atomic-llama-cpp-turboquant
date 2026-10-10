#!/usr/bin/env python3
"""Bounded production norm-shader and independent relay calibration checks."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','micro','shaders','relay'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--only',choices=('norm','relay','all'),default='all')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    jobs=[]
    if a.only in ('norm','all'):
        for domain in ('local','system'):
            for rows in (1,2,4):
                for on in (0,1):
                    for withhold in (0,1):
                        jobs.append((f'norm-{domain}-r{rows}-p{on}-fail{withhold}',
                            [str(a.micro),str(a.shaders),'norm',domain,'0','256',str(rows)],
                            {'TP5_TEST_NORM_PREFETCH':str(on),'TP5_TEST_NORM_WITHHOLD':str(withhold)}))
    if a.only in ('relay','all'):
        for n in (256,2560,4096):
            for parallel in (0,1):
                jobs.append((f'relay-n{n}-p{parallel}',[str(a.relay),'--run','--service','--stages','96',
                            '--replays','16','--elements',str(n)],{'GGML_TP5_PARALLEL_SUBMIT':str(parallel)}))
    completed=[]
    for name,cmd,extra in jobs:
        env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ}
        env.update(RADV_DEBUG='nobolist',GGML_VK_DISABLE_MMVQ='1');env.update(extra)
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            deadline=time.monotonic()+20
            streak=0
            while streak<3:
                state=b.gpu_state()
                good=len(state)==5 and all(int(x['gpu_busy_percent'])==0 and int(x['mem_info_vram_used'])<128*1024*1024 for x in state)
                streak=streak+1 if good else 0
                if time.monotonic()>deadline:raise RuntimeError('Unchanged five-card idle gate did not settle')
                time.sleep(.5)
            print('BEGIN',name,flush=True)
            with (a.root/(name+'.log')).open('xb') as log:
                rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            row={'name':name,'argv':cmd,'env':env,'before':state,'returncode':rc,
                 'binary_sha256':b.sha(Path(cmd[0]))}
            if name.startswith('norm'):
                row['shader_sha256']=b.sha(a.shaders/'tp5_hc_resume_norm.spv')
            (a.root/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
            completed.append(row)
            (a.root/(a.only+'-execution.json')).write_text(json.dumps(completed,indent=2)+'\n')
            if rc:
                print('STOP_ON_FAILURE',name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
