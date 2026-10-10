#!/usr/bin/env python3
"""Native pending-queue gates followed by preregistered four process pairs."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','bin','probe','model','draft'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'))
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    env={k:os.environ[k] for k in ('HOME','PATH','LANG') if k in os.environ}
    env.update(RADV_DEBUG='nobolist',GGML_TP5_PARALLEL_SUBMIT='1')
    for name,extra in [('native-all5',[]),('native-partial4',['--partial-submit-ranks','4'])]:
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);before=bench.idle_gate()
            cmd=[str(a.probe),'--run','--stages','3','--replays','4','--elements','256']+extra
            with (a.root/(name+'.log')).open('xb') as output:
                rc=subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT).returncode
            bench.save(a.root/(name+'.json'),{'argv':cmd,'returncode':rc,'before':before,'binary_sha256':bench.sha(a.probe)})
        if rc:return rc
        print('NATIVE_PROOF_PASS',name,flush=True)
    base=[sys.executable,str(Path(__file__).with_name('tp5-gap-bench.py')),
          '--bin',str(a.bin),'--model',str(a.model),'--draft',str(a.draft),
          '--horizon','0','--latebind','aggressive','--mmvq','disabled','--threads','22',
          '--warmup','2','--repeats','3','--env','GGML_TP5_REPLICATE_ATTN=0',
          '--env','GGML_TP5_QSA_HEADMAP=1','--env','GGML_TP5_GDN_HEADMAP=1',
          '--env','GGML_VK_SMALL_Q8_COLUMNS=1','--env','GGML_TP5_GREEDY_HOST=1',
          '--env','GGML_TP5_Q_WORKGROUP=256','--env','GGML_TP5_NORM_WORKGROUP=256']
    done=[]
    for block,order in enumerate((('A','B'),('B','A'),('A','B'),('B','A')),1):
        for arm in order:
            name=f'queue-b{block}-{arm}';print('BEGIN',name,flush=True)
            cmd=base+['--env','GGML_TP5_PARALLEL_SUBMIT='+('0' if arm=='A' else '1'),'--out',str(a.root/name)]
            rc=subprocess.run(cmd).returncode
            done.append({'run':name,'returncode':rc})
            bench.save(a.root/'confirmation-execution.json',done)
            if rc:return rc
    print('FOUR_PAIRS_COMPLETE',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
