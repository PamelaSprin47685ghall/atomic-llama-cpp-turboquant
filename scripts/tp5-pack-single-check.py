#!/usr/bin/env python3
"""Production pack full-byte correctness and guard-only service checks."""
import argparse,fcntl,importlib.util,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','probe','micro','shaders'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    jobs=[]
    for dev in range(5):
        for blocks in ((1,255,256,257,102400,102403) if dev==0 else (102400,102403)):
            jobs.append((f'bytes-d{dev}-b{blocks}',[str(a.probe),str(a.shaders/'tp5_hc_late_pack_single.spv'),str(blocks),str(dev)],{'TP5_TEST_PACK_SINGLE_WG':'1'}))
        for on in (0,1):
            jobs.append((f'guard-d{dev}-single{on}',[str(a.micro),str(a.shaders),'packnoop','local',str(dev)],{'TP5_TEST_PACK_SINGLE_WG':str(on),'TP5_TEST_WARM_MS':'200','TP5_TEST_BATCH_TIMING':'1'}))
    done=[]
    for name,cmd,extra in jobs:
        env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ};env.update(RADV_DEBUG='nobolist');env.update(extra)
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+20;streak=0
            while streak<2:
                state=b.gpu_state();good=len(state)==5 and all(int(x['gpu_busy_percent'])==0 and int(x['mem_info_vram_used'])<128*1024*1024 for x in state)
                streak=streak+1 if good else 0
                if time.monotonic()>deadline:raise RuntimeError('Unchanged five-card idle gate failed')
                time.sleep(.5)
            print('BEGIN',name,flush=True);path=a.root/(name+'.log')
            with path.open('xb') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            results=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]
            shader='tp5_hc_late_pack_single.spv' if extra.get('TP5_TEST_PACK_SINGLE_WG')=='1' else 'tp5_hc_late_pack.spv'
            item={'name':name,'argv':cmd,'env':env,'returncode':rc,'binary_sha256':b.sha(Path(cmd[0])),'shader_sha256':b.sha(a.shaders/shader),'results':results}
            done.append(item);(a.root/'execution.json').write_text(json.dumps(done,indent=2)+'\n')
            if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
