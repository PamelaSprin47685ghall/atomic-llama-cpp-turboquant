#!/usr/bin/env python3
"""Run the registered bounded TP5 Q/norm/submission interaction screen."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','bin','model','draft'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    registration=json.loads((a.root/'preregister.json').read_text())
    base=[sys.executable,str(Path(__file__).with_name('tp5-gap-bench.py')),
          '--bin',str(a.bin),'--model',str(a.model),'--draft',str(a.draft),
          '--horizon','0','--latebind','aggressive','--mmvq','disabled','--threads','22',
          '--warmup','2','--repeats','2','--env','GGML_TP5_REPLICATE_ATTN=0',
          '--env','GGML_TP5_QSA_HEADMAP=1','--env','GGML_TP5_GDN_HEADMAP=1',
          '--env','GGML_VK_SMALL_Q8_COLUMNS=1','--env','GGML_TP5_GREEDY_HOST=1']
    done=[]
    for case in registration['screen']:
        dest=a.root/('screen-'+case['name'])
        if dest.exists():raise RuntimeError('Never replace an earlier screen: '+str(dest))
        cmd=base+['--env',f"GGML_TP5_PARALLEL_SUBMIT={case['parallel']}",
                  '--env',f"GGML_TP5_Q_WORKGROUP={case['q']}",
                  '--env',f"GGML_TP5_NORM_WORKGROUP={case['norm']}",'--out',str(dest)]
        print('BEGIN',case['name'],flush=True)
        rc=subprocess.run(cmd,check=False).returncode
        done.append({'case':case,'argv':cmd,'returncode':rc})
        (a.root/'screen-execution.json').write_text(json.dumps(done,indent=2)+'\n')
        if rc:
            print('STOP_ON_FAILURE',case['name'],rc,flush=True)
            return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
