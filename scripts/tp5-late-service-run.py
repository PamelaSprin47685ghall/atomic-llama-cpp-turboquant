#!/usr/bin/env python3
"""Run registered isolated production-shader service checks with exclusive GPU use."""
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
    for k in ('root','bin','shaders'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ}
    env['RADV_DEBUG']='nobolist'
    cases=[('inject','local'),('act','local'),('q','local'),('q','system'),('up','local')]
    names={'inject':'tp5_hc_late_inject.spv','act':'tp5_hc_late_act_q8.spv','q':'tp5_hc_late_q_q8dot.spv','up':'tp5_hc_late_up_q8dot.spv'}
    for kernel,domain in cases:
        name=f'{kernel}-{domain}'
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            until=time.monotonic()+15
            while True:
                s=b.gpu_state()
                if len(s)==5 and all(int(x['gpu_busy_percent'])==0 and int(x['mem_info_vram_used'])<128*1024*1024 for x in s):break
                if time.monotonic()>until:raise RuntimeError('Five-card idle gate did not pass')
                time.sleep(.5)
            command=[str(a.bin),str(a.shaders),kernel,domain,'0']
            print('BEGIN',name,flush=True)
            with (a.root/(name+'.log')).open('xb') as log:
                rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            info={'argv':command,'env':env,'before':s,'returncode':rc,'binary_sha256':b.sha(a.bin),
                  'shader_sha256':hashlib.sha256((a.shaders/names[kernel]).read_bytes()).hexdigest()}
            (a.root/(name+'.json')).write_text(json.dumps(info,indent=2)+'\n')
            if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
