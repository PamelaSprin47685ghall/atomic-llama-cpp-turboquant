#!/usr/bin/env python3
"""Registered warm sparse stage census and runtime screening; no implicit retries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','model','draft'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    base=[sys.executable,str(Path(__file__).with_name('tp5-gap-bench.py')),
          '--bin',str(a.bin),'--model',str(a.model),'--draft',str(a.draft),
          '--horizon','0','--latebind','aggressive','--mmvq','disabled',
          '--warmup','2','--repeats','2','--env','GGML_TP5_REPLICATE_ATTN=0',
          '--env','GGML_TP5_QSA_HEADMAP=1','--env','GGML_TP5_GDN_HEADMAP=1',
          '--env','GGML_VK_SMALL_Q8_COLUMNS=1','--env','GGML_TP5_GREEDY_HOST=1',
          '--env','GGML_TP5_QCHECK_VECTOR=0']
    jobs=[(f'warm-8-{i}',['--threads','22','--profile','cycle','--env','GGML_TP5_GPU_TIMING=400',
          '--env','GGML_TP5_GPU_TIMING_SPARSE=8:'+str(i),'--env','GGML_TP5_GPU_TIMING_WINDOW=16']) for i in range(8)]
    jobs += [('runtime-threads8',['--threads','8']),('runtime-threads1',['--threads','1']),
             ('runtime-cache4',['--threads','22','--env','GGML_TP5_DEFINITION_CACHE=4']),
             ('runtime-direct',['--threads','22','--env','GGML_VK_DISABLE_PRODUCER_WIRE=0',
               '--env','GGML_TP5_DIRECT_PRODUCER_HOST=1'])]
    done=[]
    for name,extra in jobs:
        cmd=base+extra+['--out',str(a.root/name)]
        print('BEGIN',name,flush=True)
        rc=subprocess.run(cmd,check=False).returncode
        done.append({'run':name,'returncode':rc,'argv':cmd})
        (a.root/'steady-runtime-execution.json').write_text(json.dumps(done,indent=2)+'\n')
        if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
