#!/usr/bin/env python3
"""Bounded, serial TP5 server acceptance run. No third-party Python packages.

Keeps raw responses, executable/DSO identities, resolved argv/environment,
request wall time, prompt/decode timings, and exact counting correctness.
Prefill probes use explicit token arrays and disable prompt reuse. They are
throughput probes, not semantic correctness tests. Never resets a GPU or kills
another process. A server that does not exit on SIGINT is reported, not hidden.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def gpu_state() -> list[dict]:
    result = []
    for index in range(1, 6):
        device = Path(f"/sys/class/drm/card{index}/device")
        result.append({
            "card": f"card{index}",
            "busy_percent": int((device / "gpu_busy_percent").read_text()),
            "vram_bytes": int((device / "mem_info_vram_used").read_text()),
        })
    return result


def require_idle(port: int) -> list[dict]:
    state = gpu_state()  # unreadable state is an error, never assumed idle
    if any(card["busy_percent"] != 0 or card["vram_bytes"] > 200_000_000 for card in state):
        raise RuntimeError(f"GPUs are not idle: {state}")
    for proc in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = proc.read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name in {"llama-server", "tp5-logits", "ninja", "cc1plus", "glslc", "ld", "ld.lld"}:
            raise RuntimeError(f"Refusing to overlap server/build process {proc.parent.name}: {name}")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use")
    return state


def request(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def timeline_env(binary: Path, mmvq: str) -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "RADV_DEBUG": "nobolist", "GGML_TP5_ISOLATE_BO": "1",
        "GGML_TP5_MERGE_SUBMIT": "1", "GGML_VK_CMD_REPLAY": "1",
        "GGML_TP5_CMD_REPLAY": "1",
        "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1", "GGML_VK_ALLOW_GRAPHICS_QUEUE": "1",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    # Do not silently turn off a backend optimization in the benchmark
    # harness.  "auto" measures the backend policy; explicit on/off remains
    # available for numeric/performance A/B.
    env.pop("GGML_VK_DISABLE_MMVQ", None)
    env.pop("GGML_VK_FORCE_MMVQ", None)
    if mmvq == "off":
        env["GGML_VK_DISABLE_MMVQ"] = "1"
    elif mmvq == "on":
        env["GGML_VK_FORCE_MMVQ"] = "1"
    env.update(GGML_TP5_SYNC="timeline", GGML_TP5_WIRE="f16", GGML_TP5_RELAY="off")
    env["LD_LIBRARY_PATH"] = str(binary.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def provenance(binary: Path, model: Path, hash_model: bool) -> dict:
    files = [binary, *sorted(binary.parent.glob("lib*.so"))]
    identity = {str(path): sha256(path) for path in files if path.is_file()}
    shards = sorted(model.parent.glob(model.name.replace("-00001-of-", "-?????-of-")))
    models = []
    for path in shards or [model]:
        stat = path.stat()
        entry = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_model:
            entry["sha256"] = sha256(path)
        models.append(entry)
    root = next((p for p in binary.parents if (p / ".git").exists()), None)
    result = {"executables_sha256": identity, "model_files": models}
    if root:
        def git(*args: str) -> bytes:
            return subprocess.check_output(["git", "-C", str(root), *args])
        result["source_root"] = str(root)
        result["source_commit"] = git("rev-parse", "HEAD").decode().strip()
        result["source_status"] = git("status", "--porcelain").decode()
        result["tracked_diff_sha256"] = hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest()
    return result


def summarize(samples: list[dict]) -> dict:
    counts = [s for s in samples if s["kind"] == "count" and not s["warmup"]]
    valid = [s for s in counts if s["correct"]]
    timings = [s["response"]["timings"] for s in valid]
    tokens = sum(t["predicted_n"] for t in timings)
    decode_ms = sum(t["predicted_ms"] for t in timings)
    wall_ms = sum(s["wall_ms"] for s in valid)
    return {
        "count_passes": len(valid), "count_samples": len(counts),
        "generated_tokens": tokens,
        "decode_ms": decode_ms,
        "request_wall_ms": wall_ms,
        "decode_tokens_per_second": 1000 * tokens / decode_ms if decode_ms else None,
        "request_tokens_per_second": 1000 * tokens / wall_ms if wall_ms else None,
        "all_checks_passed": bool(counts) and all(s["correct"] for s in samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--prefill", default="31,128,512,1024", help="comma-separated lengths; empty disables")
    parser.add_argument("--concurrent", type=int, default=0, help="additional simultaneous, distinct counting requests")
    parser.add_argument("--post-prefill-check", action="store_true", help="check a fresh decode after the prefill sweep")
    parser.add_argument("--cache-k", default="f16")
    parser.add_argument("--cache-v", default="f16")
    parser.add_argument("--mmvq", choices=("auto", "on", "off"), default="auto",
                        help="Vulkan MMVQ policy; auto preserves the backend cost model")
    parser.add_argument("--port", type=int, default=18095)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--hash-model", action="store_true")
    args = parser.parse_args()
    lengths = [int(n) for n in args.prefill.split(",") if n]
    if min(args.ctx, args.batch, args.ubatch, args.slots, args.repeat) < 1 or args.warmups < 0:
        parser.error("context, batches, slots and repeat must be positive; warmups must be nonnegative")
    if args.ubatch > args.batch or any(n < 1 for n in lengths):
        parser.error("ubatch must not exceed batch; prefill lengths must be positive")
    if args.concurrent < 0 or args.concurrent > args.slots:
        parser.error("concurrent requests must be between zero and the slot count")
    if max([224, *lengths]) + 1 > args.ctx // args.slots:
        parser.error("each slot needs room for the full counting/prefill workload")
    binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output exists; use a new path to preserve previous evidence")
    env = timeline_env(binary, args.mmvq)
    argv = [str(binary), "-m", str(model), "-dev", "Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4",
            "--split-mode", "tensor", "--fit", "off", "--tp5", "qwen4exp-af",
            "--tp5-sync", "timeline", "--tp5-wire", "f16", "-ngl", "999",
            "-c", str(args.ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
            "--no-mmap", "--no-host", "--spec-type", "none", "-np", str(args.slots),
            "-fa", "on", "-ctk", args.cache_k, "-ctv", args.cache_v,
            "--host", "127.0.0.1", "--port", str(args.port)]
    result = {
        "schema": 1, "argv": argv,
        "environment": {k: v for k, v in env.items() if k.startswith(("GGML_", "LLAMA_", "RADV_", "VK_", "LD_"))},
        "samples": [], "kernel_log_checked": False,
    }
    server = None
    base = f"http://127.0.0.1:{args.port}"
    log_path = args.output.with_suffix(".server.log")
    with open("/tmp/tp5-bench.lock", "a") as lock, log_path.open("x") as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result["gpu_before"] = require_idle(args.port)
            result["provenance"] = provenance(binary, model, args.hash_model)
            # Hashing may take time; check again immediately before allocating GPUs.
            require_idle(args.port)
            started = time.monotonic()
            server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {server.returncode}; see {log_path}")
                try:
                    request(base + "/health", None, 2)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() - started > args.startup_timeout:
                        raise TimeoutError("server startup timeout")
                    time.sleep(0.25)
            result["startup_ms"] = (time.monotonic() - started) * 1000

            def measure(kind: str, payload: dict, endpoint: str, warmup: bool = False,
                        length: int = 0, count_start: int = 1) -> dict:
                start = time.monotonic()
                body = request(base + endpoint, payload, args.request_timeout)
                sample = {"kind": kind, "warmup": warmup, "wall_ms": (time.monotonic() - start) * 1000,
                          "request": payload, "response": body}
                sample["gpu_after_request"] = gpu_state()
                if kind.endswith("count"):
                    choice = body["choices"][0]
                    sample["count_start"] = count_start
                    sample["correct"] = (choice["message"]["content"] == ",".join(map(str, range(count_start, count_start + 60)))
                                         and choice["finish_reason"] == "stop")
                else:
                    t = body["timings"]
                    sample["prompt_length"] = length
                    sample["correct"] = t["prompt_n"] == length and t.get("cache_n", 0) == 0 and t["predicted_n"] == 1
                result["samples"].append(sample)
                print(json.dumps({k: v for k, v in sample.items() if k not in {"request", "response"}}), flush=True)
                return sample

            payload = {"messages": [{"role": "user", "content": "Count from 1 to 60. Use commas without spaces. Output only the numbers."}],
                       "temperature": 0, "seed": 42, "max_tokens": 192, "cache_prompt": False,
                       "chat_template_kwargs": {"enable_thinking": False}}
            for i in range(args.warmups + args.repeat):
                measure("count", payload, "/v1/chat/completions", i < args.warmups)
            if lengths:
                tokens = request(base + "/tokenize", {"content": "A small river flows past the quiet town. " * (max(lengths) + 1),
                                                       "add_special": True}, args.request_timeout)["tokens"]
                for length in lengths:
                    measure("prefill", {"prompt": tokens[:length], "n_predict": 1, "temperature": 0,
                                        "seed": 42, "cache_prompt": False}, "/completion", length=length)
            if args.post_prefill_check:
                measure("post_prefill_count", payload, "/v1/chat/completions")
            if args.concurrent:
                def parallel_count(slot: int) -> dict:
                    start = 1 + 100 * slot
                    parallel_payload = {**payload, "id_slot": slot, "max_tokens": 256,
                        "messages": [{"role": "user", "content":
                            f"Count from {start} to {start + 59}. Use commas without spaces. Output only the numbers."}]}
                    return measure("concurrent_count", parallel_payload, "/v1/chat/completions", count_start=start)
                with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
                    list(pool.map(parallel_count, range(args.concurrent)))
            result["summary"] = summarize(result["samples"])
        except (Exception, KeyboardInterrupt) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            if server is not None:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGINT)
                try:
                    result["server_exit"] = server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    result["error"] = f"server did not stop after SIGINT; process group {server.pid} left for diagnosis"
                    result["still_running_pid"] = server.pid
            try:
                result["gpu_after"] = gpu_state()
            except OSError as error:
                result["gpu_after_error"] = str(error)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("summary", {"error": result.get("error")}), indent=2))
    return 0 if not result.get("error") and result.get("server_exit") == 0 and result.get("summary", {}).get("all_checks_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
