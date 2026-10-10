#!/usr/bin/env python3
"""Independent F32 Y-handoff and matching ready-consumer service calibration."""
import argparse,fcntl,importlib.util,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','relay','micro','shaders'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    s=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('tp5-gap-bench.py'));b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
    jobs=[]
    for elements in (256,2560,4096):
        for parallel in (0,1):
            jobs.append((f'f32-n{elements}-p{parallel}',[str(a.relay),'--run','--service','--stages','96','--replays','16','--elements',str(elements)],
                         {'TP5_PROBE_SERVICE_F32':'1','GGML_TP5_PARALLEL_SUBMIT':str(parallel)}))
    for d in range(5):
        for batch in (0,1):
            jobs.append((f'consumer-d{d}-batch{batch}',[str(a.micro),str(a.shaders),'relayconsume','production',str(d)],
                         {'TP5_TEST_BATCH_TIMING':str(batch),'TP5_TEST_WARM_MS':'200'}))
    done=[]
    for name,cmd,extra in jobs:
        env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ};env.update(RADV_DEBUG='nobolist',GGML_VK_ALLOW_GRAPHICS_QUEUE='1');env.update(extra)
        with open('/var/tmp/tp5-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+20;streak=0
            while streak<3:
                state=b.gpu_state();good=len(state)==5 and all(int(x['gpu_busy_percent'])==0 and int(x['mem_info_vram_used'])<128*1024*1024 for x in state)
                streak=streak+1 if good else 0
                if time.monotonic()>deadline:raise RuntimeError('Five-card idle gate not satisfied')
                time.sleep(.5)
            print('BEGIN',name,flush=True);path=a.root/(name+'.log')
            with path.open('xb') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            data=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]
            done.append({'name':name,'argv':cmd,'env':env,'before':state,'returncode':rc,'binary_sha256':b.sha(Path(cmd[0])),'result':data[-1] if data else None})
            (a.root/'execution.json').write_text(json.dumps(done,indent=2)+'\n')
            if rc:print('STOP_ON_FAILURE',name,rc,flush=True);return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
