#!/usr/bin/env python3
"""Run the registered post-greedy interaction screen, stopping at any failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','bin','model','draft'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    cases=[('aggressive','disabled',0),('q8first','disabled',0),('pack-once','disabled',0),
           ('ap','disabled',0),('l2','disabled',0),('aggressive','auto',0),
           ('aggressive','forced',0),('aggressive','disabled',1)]
    base=[sys.executable,str(Path(__file__).with_name('tp5-gap-bench.py')),
          '--bin',str(a.bin),'--model',str(a.model),'--draft',str(a.draft),
          '--threads','22','--warmup','2','--repeats','2',
          '--env','GGML_TP5_REPLICATE_ATTN=0','--env','GGML_TP5_QSA_HEADMAP=1',
          '--env','GGML_TP5_GDN_HEADMAP=1','--env','GGML_VK_SMALL_Q8_COLUMNS=1',
          '--env','GGML_TP5_GREEDY_HOST=1']
    done=[]
    for late,mmvq,h in cases:
        name=f'interaction-{late}-{mmvq}-h{h}'
        cmd=base+['--latebind',late,'--mmvq',mmvq,'--horizon',str(h),'--out',str(a.root/name)]
        print('BEGIN',name,flush=True)
        rc=subprocess.run(cmd,check=False).returncode
        done.append({'run':name,'returncode':rc,'argv':cmd})
        (a.root/'interaction-execution.json').write_text(json.dumps(done,indent=2)+'\n')
        if rc:
            print('STOP_ON_FAILURE',name,rc,flush=True)
            return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
