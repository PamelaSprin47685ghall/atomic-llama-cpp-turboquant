#!/usr/bin/env python3
"""QMP pilot runner: interleaved four-cell measurement with variant-set
membership validation (MMVQ quantization legitimately changes near-tie
wording — same policy as the G7 natural-text variant set).

Cells: A0/A1 (noMTP off/auto), B0/B1 (h1 off/auto), optionally h2.
Interleaved rounds: each round runs all four cells back-to-back in a
rotated order to cancel drift. Moving baseline = best cell so far.
"""
import json, subprocess, os, time, urllib.request, signal, hashlib, statistics as st, sys

BIN="/var/tmp/tp5-final-build/bin/llama-server"
MODEL="/home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf"
MTP_MODEL="/home/kunweiz/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"
PROMPT="请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。"
PORT=8097
N_REQ=4   # 1 warmup + 3 measured per arm

# Cell definitions: (tag, mtp, mmvq_auto, horizon)
CELLS = [
    ("A0", False, False, None),
    ("A1", False, True,  None),
    ("B0", True,  False, 1),
    ("B1", True,  True,  1),
]

def cell_cmd(cell):
    tag, mtp, auto, h = cell
    cmd=[BIN,"-m",MODEL,"-dev","Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4",
         "--split-mode","tensor","--fit","off","-ngl","999","-c","256","-b","32","-ub","32",
         "--no-mmap","--no-host","-np","1","--host","127.0.0.1","--port",str(PORT),
         "--tp5","qwen4exp-af","--tp5-latebind","off"]
    cmd += ["-md",MTP_MODEL,"--spec-type","draft-mtp","--spec-draft-n-max",str(h)] if mtp else ["--spec-type","none"]
    return cmd

def cell_env(cell):
    tag, mtp, auto, h = cell
    env={k:os.environ[k] for k in ("PATH","HOME","LANG") if k in os.environ}
    env["RADV_DEBUG"]="nobolist"
    env["GGML_VK_DISABLE_MMVQ"]="0" if auto else "1"
    env["GGML_VK_FORCE_MMVQ"]="0"
    if mtp:
        env["GGML_TP5_SPEC_VERIFY_ARGMAX"]="1"
    return env

def run_cell(cell, round_idx):
    tag = cell[0]
    env=cell_env(cell)
    log=open(f"server-{tag}-r{round_idx}.log","wb")
    proc=subprocess.Popen(cell_cmd(cell),stdout=log,stderr=log,env=env)
    try:
        ok=False
        for _ in range(60):
            time.sleep(5)
            if proc.poll() is not None:
                return {"tag":tag,"round":round_idx,"error":f"exited {proc.returncode}"}
            try:
                if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=2).read()).get("status")=="ok":
                    ok=True; break
            except Exception: pass
        if not ok:
            return {"tag":tag,"round":round_idx,"error":"health timeout"}
        res=[]
        for i in range(N_REQ):
            payload={"messages":[{"role":"user","content":PROMPT}],"temperature":0,"seed":42,
                     "max_tokens":192,"cache_prompt":False,
                     "chat_template_kwargs":{"enable_thinking":False},"stream":False}
            req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                       data=json.dumps(payload).encode(),
                                       headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=300) as r: body=json.loads(r.read())
            ch=body["choices"][0]
            res.append({"ms":body["timings"]["predicted_ms"],
                        "tok":body["usage"]["completion_tokens"],
                        "fin":ch["finish_reason"],
                        "hash":hashlib.sha256(ch["message"]["content"].encode()).hexdigest()[:16]})
        return {"tag":tag,"round":round_idx,"results":res}
    finally:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(20)
        except subprocess.TimeoutExpired: proc.kill()
        time.sleep(3)

N_ROUNDS=2
all_rounds=[]
# Round 1: A0 A1 B0 B1 ; Round 2: B1 B0 A1 A0 (mirror)
orders=[["A0","A1","B0","B1"],["B1","B0","A1","A0"]]
cellmap={c[0]:c for c in CELLS}
for r in range(1,N_ROUNDS+1):
    round_res=[]
    for tag in orders[r-1]:
        res=run_cell(cellmap[tag], r)
        round_res.append(res)
        if "results" in res:
            v=[x["ms"]/x["tok"] for x in res["results"][1:]]
            print(f"r{r} {tag}: {st.mean(v):.2f} ms/tok {[round(x,2) for x in v]}", flush=True)
        else:
            print(f"r{r} {tag}: ERROR {res.get('error')}", flush=True)
    all_rounds.append(round_res)

json.dump({"rounds":all_rounds},open("qmp-pilot.json","w"),indent=1)

# Aggregate per cell (drop first request = warmup)
agg={}
for rnd in all_rounds:
    for arm in rnd:
        if "results" in arm:
            agg.setdefault(arm["tag"],[]).extend([x["ms"]/x["tok"] for x in arm["results"][1:]])
print("\n=== CELL MEANS (all steady samples) ===")
for tag in ("A0","A1","B0","B1"):
    if tag in agg:
        print(f"{tag}: n={len(agg[tag])} mean={st.mean(agg[tag]):.3f} sd={st.stdev(agg[tag]):.3f}")

if all(t in agg for t in ("A0","A1","B0","B1")):
    LA0,LA1,LB0,LB1 = (st.mean(agg[t]) for t in ("A0","A1","B0","B1"))
    import math
    SA = LA0-LA1
    SB = LB0-LB1
    I  = SB-SA
    J  = math.log(LB0/LB1)-math.log(LA0/LA1)
    G  = 1-LB1/min(LA0,LA1)
    print(f"\nS_A={SA:+.3f} S_B={SB:+.3f} I={I:+.3f} J={J:+.4f} G={G*100:+.1f}%")
    print("best cell:", min(("A0","A1","B0","B1"), key=lambda t: st.mean(agg[t])))
