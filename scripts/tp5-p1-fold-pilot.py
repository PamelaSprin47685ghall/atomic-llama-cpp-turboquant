#!/usr/bin/env python3
"""Current-stack P1 copy/fold pilot with bounded existing runner gates."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','model','draft'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();base=[sys.executable,str(Path(__file__).with_name('tp5-gap-bench.py')),
      '--bin',str(a.bin),'--model',str(a.model),'--draft',str(a.draft),'--horizon','0',
      '--latebind','aggressive','--mmvq','disabled','--threads','22','--warmup','2','--repeats','3']
    for e in ('GGML_TP5_REPLICATE_ATTN=0','GGML_TP5_QSA_HEADMAP=1','GGML_TP5_GDN_HEADMAP=1',
              'GGML_VK_SMALL_Q8_COLUMNS=1','GGML_TP5_GREEDY_HOST=1','GGML_TP5_PARALLEL_SUBMIT=1',
              'GGML_TP5_Q_WORKGROUP=256','GGML_TP5_NORM_WORKGROUP=256','GGML_TP5_NORM_PREFETCH=0',
              'GGML_TP5_PACK_SINGLE_WG=0','GGML_TP5_MEMORY_LEDGER=1'):
        base+=['--env',e]
    done=[]
    for arm,fold in (('A',0),('B',1)):
        name='p1-fold-'+arm;cmd=base+['--env',f'GGML_TP5_RELAY_VEC_COPY_FOLD={fold}','--out',str(a.root/name)]
        print('BEGIN',name,flush=True);rc=subprocess.run(cmd).returncode
        done.append({'run':name,'argv':cmd,'returncode':rc});(a.root/'p1-fold-execution.json').write_text(json.dumps(done,indent=2)+'\n')
        if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
