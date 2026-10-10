#!/usr/bin/env python3
"""Single-request, RERoT-off TP5 experiment. Preserve raw output and owned teardown.

No speed/semantic qualification is inferred by this runner. Read unique outputs
and use independent paired process blocks before promoting a configuration.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request

CANONICAL = "请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。"
GUARD = "用三句话解释哈希表与数组的主要区别。"


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def model_stat_identity(path):
    match = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", path.name)
    files = sorted(path.parent.glob(match.group(1) + "-?????-of-?????.gguf")) if match else [path]
    if match and len(files) != int(match.group(2)):
        raise RuntimeError("Incomplete model shard set")
    return {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                     "inode": p.stat().st_ino} for p in files}


def gpu_state():
    cards = []
    for p in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        if "-" in p.parent.name:
            continue
        try:
            if (p / "vendor").read_text().strip() != "0x1002":
                continue
            item = {"card": p.parent.name, "bdf": p.resolve().name}
            for key in ("device", "gpu_busy_percent", "mem_info_vram_used", "mem_info_vram_total",
                        "current_link_speed", "current_link_width", "pp_dpm_sclk", "pp_dpm_mclk"):
                q = p / key
                item[key] = q.read_text().strip() if q.exists() else None
            cards.append(item)
        except OSError as e:
            raise RuntimeError(f"Cannot read GPU state: {p}: {e}") from e
    return cards


def idle_gate():
    for _ in range(20):
        cards = gpu_state()
        if len(cards) != 5 or any(x["device"] != "0x73bf" for x in cards):
            raise RuntimeError(f"Need five physical Navi21 GPUs: {cards}")
        if all(int(x["gpu_busy_percent"]) == 0 and int(x["mem_info_vram_used"]) < 50 * 1024**2 for x in cards):
            return cards
        time.sleep(1)
    raise RuntimeError("GPUs not idle; no reset or recovery is attempted")


def build_gate():
    """Reject observed compilation at request boundaries, not just at startup."""
    for name in ("ninja", "cc1plus", "cc1", "glslc", "clang", "clang++"):
        if subprocess.run(["pgrep", "-x", name], stdout=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError(f"Build process active: {name}")


def api(port, path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def library_identity(pid):
    paths = set()
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        p = Path(fields[-1])
        if p.is_file() and ("libggml" in p.name or "libllama" in p.name or "libvulkan" in p.name):
            paths.add(p)
    return {str(p): sha(p) for p in sorted(paths)}


def request_payload(prompt, system_prompt=None, max_tokens=400):
    """Build a frozen request without changing the historical default workload."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ValueError("System prompt must not be empty when supplied")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return {"messages": messages, "temperature": 0, "seed": 42,
            "max_tokens": max_tokens, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=1, choices=range(0, 17))
    ap.add_argument("--latebind", choices=("off", "exact", "aggressive", "q8first", "pack-once", "ap", "l2", "l2-ap"), default="off")
    ap.add_argument("--mmvq", choices=("disabled", "auto", "forced"), default="disabled")
    ap.add_argument("--threads", type=int, default=22)
    ap.add_argument("--placement", choices=("tp5", "layer"), default="tp5",
                    help="Same five devices: TP5 tensor graph or ordinary layer placement")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--profile", choices=("none", "cycle", "full", "kernels"), default="none")
    ap.add_argument("--env", action="append", default=[], help="Explicit single-factor KEY=VALUE; recorded in manifest")
    ap.add_argument("--skip-guard", action="store_true")
    ap.add_argument("--prompt-file", type=Path, help="Explicit frozen UTF-8 workload, separate from the default calibration prompt")
    ap.add_argument("--system-prompt-file", type=Path,
                    help="Frozen UTF-8 system instruction for the primary workload only; guard stays unchanged")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="Primary request safety cap, not a forced output length; natural stop remains required")
    a = ap.parse_args()
    canonical = a.prompt_file.read_text(encoding="utf-8").strip() if a.prompt_file else CANONICAL
    system_prompt = a.system_prompt_file.read_text(encoding="utf-8").strip() if a.system_prompt_file else None
    try:
        primary_payload = request_payload(canonical, system_prompt, a.max_tokens)
    except ValueError as e:
        ap.error(str(e))
    if min(a.threads, a.repeats) < 1 or a.warmup < 0 or a.ctx < 256:
        ap.error("threads/repeats must be positive, warmup nonnegative")
    a.bin, a.model, a.draft, a.out = (p.resolve() for p in (a.bin, a.model, a.draft, a.out))
    for p in (a.bin, a.model, a.draft):
        if not p.is_file():
            ap.error(f"Missing input: {p}")
    a.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "STARTED", "qualification": "NOT_REVIEWED", "profile": a.profile}
    proc = None
    with open("/var/tmp/tp5-gpu.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result["gpu_before"] = idle_gate()
            build_gate()
            env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
            env.update({"LD_LIBRARY_PATH": str(a.bin.parent), "RADV_DEBUG": "nobolist",
                        "GGML_VK_DISABLE_MMVQ": "1" if a.mmvq == "disabled" else "0",
                        "GGML_VK_FORCE_MMVQ": "1" if a.mmvq == "forced" else "0"})
            lb = a.latebind if a.latebind in ("off", "exact") else "aggressive"
            if a.latebind in ("q8first", "ap", "l2-ap"):
                env["GGML_TP5_LATEBIND_Q8_ORDER"] = "first"
            if a.latebind in ("pack-once", "ap", "l2-ap"):
                env["GGML_TP5_LATEBIND_PACK_ONCE"] = "1"
            if a.latebind in ("l2", "l2-ap"):
                env["GGML_TP5_LATEBIND_L2"] = "td"
            if a.profile in ("cycle", "full", "kernels"):
                env["GGML_TP5_MTP_PROFILE"] = "1"
            if a.profile == "full":
                env.update({"GGML_TP5_PROFILE": "1", "GGML_TP5_GPU_TIMING": "1", "GGML_TP5_PROFILE_RELAY_STAGES": "1"})
            if a.profile == "kernels":
                env["GGML_VK_PERF_LOGGER"] = "1"
            for setting in a.env:
                key, sep, value = setting.partition("=")
                if not sep or not key.startswith("GGML_") or "REROT" in key:
                    raise ValueError(f"Expected non-RERoT GGML_* KEY=VALUE: {setting}")
                env[key] = value
            if env.get("GGML_DECODE_COST", "0") not in ("0", "off"):
                result["profile"] = "host-cost"
            cmd = [str(a.bin), "-m", str(a.model), "-dev", "Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4",
                   "--split-mode", "tensor" if a.placement == "tp5" else "layer", "--fit", "off", "-ngl", "999", "-c", str(a.ctx), "-b", "32", "-ub", "32",
                   "--no-mmap", "--no-host", "-np", "1", "-t", str(a.threads), "--host", "127.0.0.1", "--port", str(a.port),
                   ]
            if a.placement == "tp5":
                cmd += ["--tp5", "qwen4exp-af", "--tp5-latebind", lb]
            elif a.latebind != "off":
                raise ValueError("Layer placement has no TP5 LateBind; specify --latebind off")
            if a.horizon:
                cmd += ["-md", str(a.draft), "--spec-type", "draft-mtp", "--spec-draft-n-max", str(a.horizon), "--spec-draft-p-min", "0"]
            else:
                cmd += ["--spec-type", "none"]
            save(a.out / "manifest.json", {"argv": cmd, "env": env, "bin_sha256": sha(a.bin),
                                          "harness_sha256": sha(Path(__file__)),
                                          "model_stat_identity": model_stat_identity(a.model),
                                          "draft_stat_identity": model_stat_identity(a.draft),
                                          "prompt_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                                          "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest() if system_prompt is not None else None,
                                          "primary_payload": primary_payload,
                                          "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            log_path = a.out / "server.log"
            loading_begin = time.monotonic_ns()
            with log_path.open("wb") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, start_new_session=True)
            result["process_id"] = proc.pid
            deadline = time.monotonic() + 240
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited during loading: {proc.returncode}")
                try:
                    if api(a.port, "/health", timeout=1).get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Server loading deadline exceeded")
                time.sleep(0.25)
            result["loading_ms"] = (time.monotonic_ns() - loading_begin) / 1e6
            build_gate()
            save(a.out / "mapped-libraries.json", library_identity(proc.pid))
            requests = []
            prompts = [("warmup", canonical)] * a.warmup + [("measured", canonical)] * a.repeats
            if not a.skip_guard:
                prompts.append(("guard", GUARD))
            for idx, (phase, prompt) in enumerate(prompts):
                build_gate()
                payload = request_payload(prompt) if phase == "guard" else primary_payload
                begin = log_path.stat().st_size
                t = time.monotonic_ns()
                try:
                    response = api(a.port, "/v1/chat/completions", payload)
                except urllib.error.HTTPError as e:
                    save(a.out / f"request-{idx}-error.json", {
                        "payload": payload, "phase": phase, "http_status": e.code,
                        "body": e.read().decode("utf-8", errors="replace"),
                        "wall_ms": (time.monotonic_ns() - t) / 1e6,
                        "log_begin": begin, "log_end": log_path.stat().st_size})
                    raise
                wall_ms = (time.monotonic_ns() - t) / 1e6
                end = log_path.stat().st_size
                save(a.out / f"request-{idx}.json", {"payload": payload, "response": response, "wall_ms": wall_ms,
                                                  "log_begin": begin, "log_end": end, "phase": phase})
                ch = response["choices"][0]
                text = ch["message"].get("content") or ""
                timings = response["timings"]
                n = response["usage"]["completion_tokens"]
                with log_path.open("rb") as f:
                    f.seek(begin)
                    segment = f.read(end - begin).decode("utf-8", errors="replace")
                (a.out / f"request-{idx}.log").write_text(segment)
                issues = []
                if ch["finish_reason"] != "stop":
                    issues.append("NOT_NATURAL_STOP")
                if len(text.strip()) < 20 or re.search(r"(?:^|\n)\s*[-*]\s*\n\s*[-*]\s*\n", text):
                    issues.append("DEGENERATE_OUTPUT")
                item = {"idx": idx, "phase": phase, "n": n, "predicted_ms": timings["predicted_ms"],
                        "tps": n * 1000 / timings["predicted_ms"], "wall_ms": wall_ms,
                        "finish": ch["finish_reason"], "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "timings": timings, "issues": issues,
                        "fallback_lines": [x for x in segment.splitlines() if "fallback" in x.lower()]}
                requests.append(item)
                save(a.out / "requests.json", requests)
                build_gate()
                print(json.dumps({"out": a.out.name, **{k: item[k] for k in ("idx", "phase", "n", "tps", "issues")}}, ensure_ascii=False), flush=True)
                if issues:
                    raise RuntimeError(f"Mechanical output failure: {issues}; raw output retained")
            measured = [r for r in requests if r["phase"] == "measured"]
            result.update({"status": "MEASURED_NOT_QUALIFIED" if result["profile"] == "none" else "DIAGNOSTIC_NOT_RANKED",
                           "pooled_tps": 1000 * sum(r["n"] for r in measured) / sum(r["predicted_ms"] for r in measured),
                           "median_tps": statistics.median(r["tps"] for r in measured), "requests": requests})
        except Exception as e:
            result.update({"status": "FAILED", "error": repr(e)})
        finally:
            if proc is not None:
                start = time.monotonic()
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                forced = False
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    forced = True
                    proc.kill()
                    rc = proc.wait(timeout=10)
                result["shutdown"] = {"returncode": rc, "forced": forced, "seconds": time.monotonic() - start}
                if forced or rc != 0:
                    result["status"] = "FAILED"
                    result["teardown_error"] = "Non-normal owned teardown"
                    result.setdefault("error", "Non-normal owned teardown")
            result["gpu_after"] = gpu_state()
            save(a.out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("requests", "gpu_before", "gpu_after")}, ensure_ascii=False), flush=True)
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
