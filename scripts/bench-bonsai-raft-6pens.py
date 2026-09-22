#!/usr/bin/env python3
"""Bonsai 2 27B × Raft 复杂问题 × P=6 RERoT ON/OFF 速度对比 runner.

自包含证据链（仿 rerot-semantic-smoke.py 的安全生命周期）：
  1. 起 child llama-server（--rerot --rerot-frontier strong --metrics）
  2. 等 /health（模型加载 + Vulkan pipeline 编译可能数分钟）
  3. 跑 scripts/bench-bonsai-6pens.py（Raft DAG-of-6 prompt：OFF 单流 vs ON P=6）
  4. 跑 scripts/rerot-throughput-gate.py（A/B/B/A 配对逐轮检查，median ratio 裁决）
  5. teardown：SIGTERM → 15s → SIGKILL；证据全部落盘 logs/rerot-bench-raft/<ts>/

口径铁律：
  - ON/OFF 是同一 server 上的请求级开关（rerot:true/false），同 build 同模型同 seed；
  - prompt 同时满足 C02 gate 的 DAG/multi-Lane 结构校验（6 个编号子问题 + DAG 关键字）；
  - 不修改任何成功定义；证据目录保留 bench 原始输出与 gate 完整 JSON。
"""
import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = os.environ.get(
    "MODEL", "/home/kunweiz/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")

RAFT_PROMPT = (
    "Decompose into DAG of 6 independent sub-questions about the Raft consensus algorithm:\n"
    "Question 1: Explain leader election in Raft: terms, randomized election timeouts, "
    "and how a candidate becomes leader.\n"
    "Question 2: Explain log replication: how the leader appends entries, the AppendEntries "
    "RPC, and the commit rule with majorities.\n"
    "Question 3: Explain Raft's safety properties: the election restriction, leader "
    "completeness, and state machine safety, and why deleting committed entries is unsafe.\n"
    "Question 4: Explain membership changes with joint consensus when adding or removing servers, "
    "including the two-phase switch and the safety pitfall of naive changes.\n"
    "Question 5: Explain log compaction and snapshots: when snapshots are taken, what they "
    "contain, and how InstallSnapshot brings a slow follower up to date.\n"
    "Question 6: Analyze behavior under network partition: which side can keep committing, "
    "why the minority cannot elect a leader alone, and how consistency is preserved.\n"
    "Synthesize a coherent technical summary of how these six mechanisms interact to keep "
    "a replicated log consistent. Output the final synthesized answer."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_json(url: str, data: dict | None, key: str, timeout: float):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def wait_health(port: int, key: str, deadline_s: float, proc) -> None:
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before ready: rc={proc.returncode}")
        try:
            status, _ = http_json(f"{base}/health", None, key, 3)
            if status == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError("server readiness timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--min-ratio", type=float, default=1.0)
    parser.add_argument("--reuse-server", action="store_true",
                        help="reuse an already-running server on --port instead of spawning one")
    args = parser.parse_args()

    (REPO / "logs").mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="rerot-bench-raft-", dir=REPO / "logs"))
    run.mkdir(parents=True, exist_ok=True)
    log(f"evidence dir: {run}")
    key = "benchraftkey"

    proc = None
    try:
        if not args.reuse_server:
            cmd = [str(REPO / "build/bin/llama-server"), "-m", MODEL,
                   "--host", "127.0.0.1", "--port", str(args.port),
                   "-c", str(args.context), "-ngl", "99", "-b", "2048", "-ub", "512",
                   "-fa", "on", "--rerot", "--rerot-frontier", "strong",
                   "--metrics", "--api-key", key, "-np", "6", "--parallel", "6"]
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = str(REPO / "build/bin") + ":" + env.get("LD_LIBRARY_PATH", "")
            with (run / "server.log").open("wb") as slog:
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=slog,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            log(f"child server pid={proc.pid} (ctx={args.context})")
        wait_health(args.port, key, 600.0 if proc else 10.0, proc)
        log("server healthy")

        base = f"http://127.0.0.1:{args.port}"
        status, models = http_json(f"{base}/v1/models", None, key, 10)
        model_id = (models.get("data") or [{}])[0].get("id", "bonsai-2-27b") if status == 200 else "bonsai-2-27b"
        log(f"model id: {model_id}")
        (run / "models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2))

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": RAFT_PROMPT}],
            "temperature": 0.0,
            "seed": 424242,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        (run / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                  capture_output=True).stdout.strip()
        (run / "provenance.json").write_text(json.dumps(
            {"head": git_head, "model": MODEL, "prompt_chars": len(RAFT_PROMPT),
             "max_tokens": args.max_tokens, "context": args.context, "rounds": args.rounds},
            ensure_ascii=False, indent=2))

        # --- 1) bench-bonsai-6pens.py (OFF vs ON, /metrics 差值) ---
        log(">>> bench-bonsai-6pens.py (Raft prompt)")
        bench_cmd = [sys.executable, str(REPO / "scripts/bench-bonsai-6pens.py"),
                     "--base-url", base, "--api-key", key,
                     "--prompt", RAFT_PROMPT, "--max-tokens", str(args.max_tokens),
                     "--model", model_id, "--timeout", str(args.timeout)]
        with (run / "bench.log").open("wb") as bf:
            rc = subprocess.run(bench_cmd, cwd=REPO, stdout=bf,
                                 stderr=subprocess.STDOUT).returncode
        log(f"bench exit={rc} (log: {run / 'bench.log'})")

        # --- 2) rerot-throughput-gate.py (A/B/B/A 配对逐轮检查) ---
        log(">>> rerot-throughput-gate.py paired A/B/B/A")
        gate_cmd = [sys.executable, str(REPO / "scripts/rerot-throughput-gate.py"),
                    "--base-url", base, "--api-key", key,
                    "--request", str(run / "request.json"),
                    "--output", str(run / "gate-result.json"),
                    "--rounds", str(args.rounds), "--min-ratio", str(args.min_ratio),
                    "--timeout", str(args.timeout)]
        with (run / "gate.log").open("wb") as gf:
            gate_rc = subprocess.run(gate_cmd, cwd=REPO, stdout=gf,
                                     stderr=subprocess.STDOUT).returncode
        log(f"gate exit={gate_rc} (result: {run / 'gate-result.json'})")

        print(f"EVIDENCE_DIR={run}")
        print(f"BENCH_EXIT={rc} GATE_EXIT={gate_rc}")
        return 0 if (rc == 0 and gate_rc == 0) else 2
    finally:
        if proc is not None and proc.poll() is None:
            log("teardown: SIGTERM child server")
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log("teardown: SIGKILL child server")
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log(f"child server exit rc={proc.returncode if proc else 'n/a'}")


if __name__ == "__main__":
    raise SystemExit(main())
