#!/usr/bin/env python3
"""Four preregistered independent pairs for the isolated pack guard candidate."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','model','draft'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    base=[sys.executable,str(Path(__file__).with_name('tp5-gap-bench.py')),'--bin',str(a.bin),
          '--model',str(a.model),'--draft',str(a.draft),'--horizon','0','--latebind','aggressive',
          '--mmvq','disabled','--threads','22','--warmup','2','--repeats','3']
    for e in ('GGML_TP5_REPLICATE_ATTN=0','GGML_TP5_QSA_HEADMAP=1','GGML_TP5_GDN_HEADMAP=1',
              'GGML_VK_SMALL_Q8_COLUMNS=1','GGML_TP5_GREEDY_HOST=1','GGML_TP5_PARALLEL_SUBMIT=1',
              'GGML_TP5_Q_WORKGROUP=256','GGML_TP5_NORM_WORKGROUP=256','GGML_TP5_NORM_PREFETCH=0'):
        base+=['--env',e]
    done=[]
    for i,order in enumerate((('A','B'),('B','A'),('A','B'),('B','A')),1):
        bodies={}
        for arm in order:
            name=f'pack-b{i}-{arm}';dest=a.root/name
            cmd=base+['--env','GGML_TP5_PACK_SINGLE_WG='+('0' if arm=='A' else '1'),'--out',str(dest)]
            print('BEGIN',name,flush=True);rc=subprocess.run(cmd).returncode
            done.append({'run':name,'returncode':rc,'argv':cmd});(a.root/'pack-confirm-execution.json').write_text(json.dumps(done,indent=2)+'\n')
            if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
            r=json.loads((dest/'result.json').read_text())
            if r['shutdown']['returncode'] or r['shutdown']['forced'] or any(x['fallback_lines'] for x in r['requests'] if x['phase']=='measured'):
                raise RuntimeError('Unqualified exit or measured fallback: '+name)
            bodies[arm]=[]
            for x in r['requests']:
                raw=json.loads((dest/f"request-{x['idx']}.json").read_text())
                rsp=raw['response'];bodies[arm].append((raw['phase'],rsp['usage']['completion_tokens'],rsp['choices'][0]['message']['content']))
        if bodies['A']!=bodies['B']:
            (a.root/'pack-confirm-quality-failure.json').write_text(json.dumps({'pair':i,'reason':'Corresponding raw body/count mismatch','A':bodies['A'],'B':bodies['B']},ensure_ascii=False,indent=2)+'\n')
            print('STOP_ON_RAW_OUTPUT_MISMATCH',i,flush=True);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
