#!/usr/bin/env python3
"""Execute the explicitly registered host/sidecar interaction screen."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','model','draft'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();reg=json.loads((a.root/'host-sidecar-register.json').read_text())
    base=[sys.executable,str(Path(__file__).with_name('tp5-gap-bench.py')),'--bin',str(a.bin),
          '--model',str(a.model),'--draft',str(a.draft),'--horizon','0','--latebind','aggressive',
          '--mmvq','disabled','--ctx','1024','--warmup','2','--repeats','2']
    for e in ('GGML_TP5_REPLICATE_ATTN=0','GGML_TP5_QSA_HEADMAP=1','GGML_TP5_GDN_HEADMAP=1',
              'GGML_VK_SMALL_Q8_COLUMNS=1','GGML_TP5_GREEDY_HOST=1','GGML_TP5_PARALLEL_SUBMIT=1',
              'GGML_TP5_Q_WORKGROUP=256','GGML_TP5_NORM_WORKGROUP=256','GGML_TP5_NORM_STAGE_Y=0',
              'GGML_TP5_NORM_PREFETCH=0','GGML_TP5_PACK_SINGLE_WG=0'):base+=['--env',e]
    done=[]
    for case in reg['screen']:
        dest=a.root/('host-'+case['name']);cmd=base+['--threads',str(case['threads']),'--out',str(dest)]
        for e in case['extra']:cmd+=['--env',e]
        print('BEGIN',dest.name,flush=True);rc=subprocess.run(cmd).returncode
        done.append({'case':case,'argv':cmd,'returncode':rc});(a.root/'host-sidecar-execution.json').write_text(json.dumps(done,indent=2)+'\n')
        if rc:print('STOP_ON_FAILURE',dest.name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
