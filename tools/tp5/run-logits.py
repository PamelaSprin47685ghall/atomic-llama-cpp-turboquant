#!/usr/bin/env python3
"""Run a bounded, serial full-logits probe and record binary/model provenance."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON manifest; tape/log use the same stem")
    parser.add_argument("--prefill", type=int, default=16)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--cache-k", default="f16")
    parser.add_argument("--cache-v", default="f16")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if min(args.prefill, args.steps, args.batch, args.ubatch) <= 0 or args.ctx < args.prefill + args.steps:
        parser.error("positive lengths and a context covering prefill + steps are required")
    if args.ubatch > args.batch:
        parser.error("ubatch must not exceed batch")
    spec = importlib.util.spec_from_file_location("tp5_bench", Path(__file__).with_name("tp5-bench.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tape, log_path = args.output.with_suffix(".logits.bin"), args.output.with_suffix(".log")
    if any(path.exists() for path in (args.output, tape, log_path)):
        parser.error("an evidence path already exists; choose a new output stem")
    env = bench.timeline_env(binary)
    argv = [str(binary), "--audit-output", str(tape), "--audit-prefill", str(args.prefill),
            "--audit-steps", str(args.steps), "-m", str(model), "-dev", "Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4",
            "--split-mode", "tensor", "--fit", "off", "--tp5", "qwen4exp-af", "--tp5-sync", "timeline",
            "--tp5-wire", "f16", "-ngl", "999", "-c", str(args.ctx), "-b", str(args.batch),
            "-ub", str(args.ubatch), "--no-mmap", "--no-host", "-np", "1", "-fa", "on",
            "-ctk", args.cache_k, "-ctv", args.cache_v]
    result = {"argv": argv, "tape": str(tape), "pass": False, "kernel_log_checked": False,
              "environment": {k: v for k, v in env.items() if k.startswith(("GGML_", "LLAMA_", "RADV_", "VK_", "LD_"))}}
    process = None
    with open("/tmp/tp5-bench.lock", "a") as lock, log_path.open("x") as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result["gpu_before"] = bench.require_idle(18095)
            result["provenance"] = bench.provenance(binary, model, False)
            bench.require_idle(18095)
            start = time.monotonic()
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result["exit_code"] = process.wait(timeout=args.timeout)
            result["wall_ms"] = (time.monotonic() - start) * 1000
            result["pass"] = result["exit_code"] == 0
        except (Exception, KeyboardInterrupt) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    result["exit_code"] = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    result["still_running_pid"] = process.pid
            result["gpu_after"] = bench.gpu_state()
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"provenance", "argv", "environment"}}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
