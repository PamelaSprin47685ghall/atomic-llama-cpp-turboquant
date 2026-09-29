import json, subprocess, os, time, urllib.request, signal, hashlib, statistics as st
BIN="/var/tmp/tp5-final-build/bin/llama-server"
MODEL="/home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf"
MTP_MODEL="/home/kunweiz/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"
PROMPT="请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。"
PORT=8097
N_REQ=4
def cell_cmd(h):
    return [BIN,"-m",MODEL,"-dev","Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4","--split-mode","tensor",
     "--fit","off","-ngl","999","-c","256","-b","32","-ub","32","--no-mmap","--no-host","-np","1",
     "--host","127.0.0.1","--port",str(PORT),"--tp5","qwen4exp-af","--tp5-latebind","off",
     "-md",MTP_MODEL,"--spec-type","draft-mtp","--spec-draft-n-max",str(h)]
ENV_BASE={k:os.environ[k] for k in ("PATH","HOME","LANG") if k in os.environ}
ENV_BASE["RADV_DEBUG"]="nobolist"
ENV_BASE["GGML_VK_DISABLE_MMVQ"]="1"
ENV_BASE["GGML_VK_FORCE_MMVQ"]="0"
ENV_BASE["GGML_TP5_SPEC_VERIFY_ARGMAX"]="1"
def run_cell(tag,h,round_idx):
    env=dict(ENV_BASE)
    log=open(f"server-{tag}-r{round_idx}.log","wb")
    proc=subprocess.Popen(cell_cmd(h),stdout=log,stderr=log,env=env)
    try:
        for _ in range(60):
            time.sleep(5)
            if proc.poll() is not None:
                return {"tag":tag,"round":round_idx,"error":"exited %d"%proc.returncode}
            try:
                if json.loads(urllib.request.urlopen("http://127.0.0.1:%d/health"%PORT,timeout=2).read()).get("status")=="ok": break
            except Exception: pass
        res=[]
        for i in range(N_REQ):
            payload={"messages":[{"role":"user","content":PROMPT}],"temperature":0,"seed":42,
                     "max_tokens":192,"cache_prompt":False,
                     "chat_template_kwargs":{"enable_thinking":False},"stream":False}
            req=urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions"%PORT,
                                       data=json.dumps(payload).encode(),
                                       headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=300) as r: body=json.loads(r.read())
            ch=body["choices"][0]
            res.append({"ms":body["timings"]["predicted_ms"],"tok":body["usage"]["completion_tokens"],
                        "fin":ch["finish_reason"]})
        return {"tag":tag,"round":round_idx,"results":res}
    finally:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(20)
        except subprocess.TimeoutExpired: proc.kill()
        time.sleep(3)
rounds=[]
orders=[["h1","h2"],["h2","h1"]]
for r in (1,2):
    rr=[]
    for tag in orders[r-1]:
        res=run_cell(tag, int(tag[1]), r)
        rr.append(res)
        if "results" in res:
            v=[x["ms"]/x["tok"] for x in res["results"][1:]]
            print("r%d %s: %.3f ms/tok"%(r,tag,st.mean(v)), flush=True)
        else:
            print("r%d %s: ERROR %s"%(r,tag,res.get("error")), flush=True)
    rounds.append(rr)
json.dump({"rounds":rounds},open("climb-h2.json","w"),indent=1)
agg={}
for rnd in rounds:
    for arm in rnd:
        if "results" in arm:
            agg.setdefault(arm["tag"],[]).extend([x["ms"]/x["tok"] for x in arm["results"][1:]])
for t in ("h1","h2"):
    if t in agg:
        print("%s: n=%d mean=%.3f sd=%.3f"%(t,len(agg[t]),st.mean(agg[t]),st.stdev(agg[t])))
if "h1" in agg and "h2" in agg:
    d=st.mean(agg["h2"])-st.mean(agg["h1"])
    print("h2-h1 = %+.3f ms/tok -> %s"%(d,"h2 WINS" if d<-0.15 else ("h1 stays" if d<0.15 else "h2 loses")))
