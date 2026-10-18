#!/usr/bin/env python3
"""Smoke-test a real stock server's phase/KV seams and concurrent SSE completion.

AXIOM: the supplied executable and GGUF are the intended real participants.
LEMMA: client events establish completed HTTP work, not internal batch ordering.
SPEC: two active streams precede a third request; logs prove phase cycles,
physical KV capacity and actual GPU offload. No throughput/peak-memory claim.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def require(value, message):
    if not value:
        raise AssertionError(message)


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def sse(response):
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            payload, data = "\n".join(data), []
            if payload != "[DONE]":
                yield json.loads(payload)
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    require(args.timeout > 0, "--timeout must be positive")
    binary, model, output = args.binary.resolve(), args.model.resolve(), args.output_dir.resolve()
    require(binary.is_file() and model.is_file(), "real server binary and model files are required")
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [str(binary), "--model", str(model), "--phase-memory", "--prefill-priority",
               "--ctx-size", "512", "--total-kv", "1536", "-np", "3", "-b", "128",
               "-ub", "64", "-ngl", "999", "-t", "2", "--host", "127.0.0.1",
               "--port", str(port), "--no-warmup", "--verbose", "--log-colors", "off"]
    invalid = [str(binary), "--model", str(output / "missing-model.gguf"),
               "--phase-memory", "--no-prefill-priority"]
    result = {"status": "running", "command": command, "invalid_command": invalid,
              "limits": ["SSE receipt times do not prove batch pause or absence of generation interleaving.",
                         "This smoke test does not measure throughput or peak GPU memory."]}
    process, pool, bad = None, None, None
    condition = threading.Condition()
    states = [{"chunks": 0, "done": False, "error": None} for _ in range(3)]
    started = time.monotonic()
    url = f"http://127.0.0.1:{port}"

    def stream(slot, prompt, predict):
        request = {"prompt": prompt, "id_slot": slot, "stream": True, "n_predict": predict,
                   "temperature": 0.0, "ignore_eos": True, "return_tokens": True, "cache_prompt": False}
        (output / f"request-{slot}.json").write_text(json.dumps(request, indent=2) + "\n")
        events = []
        try:
            post = Request(url + "/completion", json.dumps(request).encode(), {"Content-Type": "application/json"})
            with urlopen(post, timeout=args.timeout) as response, (output / f"response-{slot}.jsonl").open("w") as log:
                require(response.status == 200, f"slot {slot}: HTTP {response.status}")
                for data in sse(response):
                    log.write(json.dumps({"received_seconds": time.monotonic() - started, "data": data}) + "\n")
                    log.flush()
                    require(isinstance(data, dict) and "error" not in data, f"slot {slot}: error event {data}")
                    events.append(data)
                    with condition:
                        if not data.get("stop") and (data.get("content") or data.get("tokens")):
                            states[slot]["chunks"] += 1
                        condition.notify_all()
            finals = [event for event in events if event.get("stop")]
            require(len(finals) == 1, f"slot {slot}: expected one final event")
            final = finals[0]
            require(final.get("id_slot") == slot, f"slot {slot}: response used a different slot")
            require(final.get("stop_type") == "limit", f"slot {slot}: generation did not finish at its limit")
            require(final.get("timings", {}).get("predicted_n") == predict, f"slot {slot}: generation budget mismatch")
            parts = [event for event in events if not event.get("stop")]
            token_support = bool(parts) and all("tokens" in event for event in parts)
            if token_support:
                require(sum(len(event["tokens"]) for event in parts) == predict, f"slot {slot}: returned token count mismatch")
            return {"id_slot": slot, "predicted_n": predict, "stream_chunks": states[slot]["chunks"],
                    "returned_tokens_supported": token_support, "final": final}
        except Exception as error:
            message = error.read().decode(errors="replace") if isinstance(error, HTTPError) else str(error)
            with condition:
                states[slot]["error"] = message
            raise RuntimeError(f"slot {slot}: {message}") from error
        finally:
            with condition:
                states[slot]["done"] = True
                condition.notify_all()

    try:
        with (output / "invalid-cli.log").open("wb") as log:
            bad = subprocess.Popen(invalid, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = bad.wait(timeout=10)
            finally:
                stop(bad)
        invalid_log = (output / "invalid-cli.log").read_text(errors="replace")
        require(code != 0 and "--phase-memory requires --prefill-priority" in invalid_log,
                "invalid phase combination was not rejected with its specific pre-load diagnostic")
        result["invalid_combo_rejected_before_model_load"] = True
        with (output / "server.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + args.timeout
        while True:
            require(process.poll() is None, "server exited before readiness; see server.log")
            require(time.monotonic() < deadline, "server readiness timed out")
            try:
                with urlopen(url + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (HTTPError, URLError, TimeoutError):
                time.sleep(0.1)
        pool = ThreadPoolExecutor(max_workers=3)
        prompts = ["Continue a numbered list of simple facts about the ocean. Water covers much of Earth. 1.",
                   "Continue a numbered list of simple facts about forests. Trees grow in the soil. 1.",
                   "Summarize these observations in a numbered list. " +
                   "Trees have leaves. Rivers carry water. Mountains have rocky peaks. " * 18]
        futures = [pool.submit(stream, i, prompts[i], 48) for i in range(2)]
        with condition:
            ready = condition.wait_for(lambda: all(s["chunks"] >= 2 for s in states[:2]) or
                                       any(s["done"] or s["error"] for s in states[:2]), timeout=args.timeout)
            require(ready and all(s["chunks"] >= 2 and not s["done"] and not s["error"] for s in states[:2]),
                    f"two active streams were not established before injection: {states}")
            result["injection"] = {"seconds": time.monotonic() - started,
                                   "initial_stream_chunks": [s["chunks"] for s in states[:2]],
                                   "initial_streams_client_complete": [s["done"] for s in states[:2]]}
        futures.append(pool.submit(stream, 2, prompts[2], 8))
        result["requests"] = [future.result(timeout=args.timeout) for future in futures]
        require(process.poll() is None, "server exited during successful requests")
        stop(process)
        server_log = (output / "server.log").read_text(errors="replace")
        phases = re.findall(r"inference phase = (prefill|generation)", server_log)
        cycles = sum(a == "prefill" and b == "generation" for a, b in zip(phases, phases[1:]))
        offloaded = [int(value) for value in re.findall(r"offloaded\s+(\d+)\s*/\s*\d+\s+layers", server_log)]
        capacities = [int(value) for value in re.findall(r"n_ctx_kv\s*=\s*(\d+)", server_log)]
        require(cycles >= 2, f"expected initial and injected prefill/generation cycles, got {phases}")
        require(offloaded and max(offloaded) > 0, "no actual GPU layer offload was logged")
        require(capacities and capacities[-1] == 1536, f"physical KV capacity mismatch: {capacities}")
        result.update(status="passed", phases=phases, completed_phase_cycles=cycles,
                      actual_offloaded_layers=offloaded, physical_kv_capacity=capacities[-1])
    except Exception as error:
        result.update(status="failed", error=str(error), stream_states=states)
        raise
    finally:
        stop(process)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
