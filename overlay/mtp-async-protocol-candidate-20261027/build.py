from pathlib import Path
import json,subprocess,time,hashlib
from datetime import datetime,timezone
D=Path("/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-async-protocol-candidate-20261027");steps=[]
for step in ["compile-meta","link-base","compile","archive-complete","link"]:
 j=json.loads((D/(step+'.argv.json')).read_text());a=j['argv']
 if '-o' in a:assert a[a.index('-o')+1].startswith(str(D)+'/')
 t=time.monotonic();p=subprocess.run(a,cwd=j['cwd'],capture_output=True)
 (D/(step+'.stdout')).write_bytes(p.stdout);(D/(step+'.stderr')).write_bytes(p.stderr)
 r={'step':step,'returncode':p.returncode,'seconds':time.monotonic()-t};steps.append(r);print(json.dumps(r),flush=True)
 if p.returncode:(D/'build-failure.json').write_text(json.dumps(steps,indent=2));print(p.stderr.decode()[-7000:]);raise SystemExit(p.returncode)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'steps':steps,'binary_sha256':h(D/'install/bin/tq-tp5-compose'),'runtime_sha256':h(D/'runtime.cpp'),'meta_sha256':h(D/'ggml-backend-meta.cpp'),'libraries':{p.name:h(p) for p in (D/'install/lib').iterdir() if p.is_file() and not p.is_symlink()},'incremental_scope':'runtime+Meta only; unchanged exact-Q2 HIP and KV-only llama libraries'}
(D/'build-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
