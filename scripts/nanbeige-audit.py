#!/usr/bin/env python3
"""Bounded, local-only llama-server probes and model-tokenized calibration.

Uses only the Python standard library. Every launched server is stopped in a
finally block. Results are measurements, not an automatic production approval.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request


def request(base, key, path, data=None, timeout=240, *, as_text=False):
    req = urllib.request.Request(
        base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8") if as_text else json.load(response)


def artifact_fingerprint(binary):
    """Fingerprint local build products, deduplicating shared-library symlinks."""
    binary = binary.resolve()
    paths = {binary}
    for pattern in ("lib*.so*", "*.dylib", "*.dll"):
        paths.update(path.resolve() for path in binary.parent.glob(pattern) if path.is_file())
    result = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path)] = digest.hexdigest()
    return result


def model_fingerprint(path):
    """Fingerprint the requested GGUF, rejecting mutation during the read."""
    path = path.resolve()
    before = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if signature(before) != signature(path.stat()):
        raise RuntimeError("model file changed while fingerprinting")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": before.st_size,
            "stat": list(signature(before))}


@contextlib.contextmanager
def server(args, config, result):
    result["artifacts_before"] = artifact_fingerprint(args.server)
    result["model_before"] = model_fingerprint(args.model)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(16)
    base = "http://127.0.0.1:" + str(port)
    command = [str(args.server.resolve()), "-m", str(args.model.resolve()),
               "-ngl", "999", "-np", "1", "-fa", "on", "--fit", "off",
               "-c", str(config["ctx"]), "--total-kv", str(config.get("kv", config["ctx"])),
               "-b", str(config["batch"]), "-ub", str(config["ubatch"]),
               "-ctk", config["k"], "-ctv", config["v"], "-t", "4", "-tb", "4",
               "--host", "127.0.0.1", "--port", str(port), "--api-key", key,
               "-lv", "4"] + config.get("extra", [])
    if config.get("require_flashprefill_plan", False):
        command.append("--metrics")
    result["command"] = ["<ephemeral-key>" if v == key else v for v in command]
    result["config"] = config
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.server.resolve().parent),
               TURBO_AUTO_ASYMMETRIC="0")
    env.update(config.get("env", {}))
    stop = threading.Event()
    samples = []
    errors = []

    def monitor():
        while not stop.is_set():
            try:
                value = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
                ], text=True, timeout=4)
                samples.append(int(value.splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append(str(exc))
            stop.wait(0.5)

    log_path = args.out / (config["name"] + ".server.log")
    result["server_log"] = str(log_path)
    started = time.monotonic()
    proc = None
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with log_path.open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("server exited during startup: " + str(proc.returncode))
                try:
                    if request(base, key, "/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError, ValueError):
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("server health deadline exceeded")
            result["startup_seconds"] = time.monotonic() - started
            yield base, key
    finally:
        if proc is not None:
            # Preserve the original exit/signal before cleanup. Otherwise a
            # disconnected HTTP request cannot distinguish a crashed server
            # from a live server whose connection timed out.
            result["server_exit_before_cleanup"] = proc.poll()
            result["server_terminated_by_runner"] = result["server_exit_before_cleanup"] is None
            if result["server_terminated_by_runner"]:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            result["server_returncode"] = proc.returncode
        stop.set()
        watcher.join(timeout=6)
        result["peak_gpu_mib"] = max(samples) if samples else None
        result["gpu_samples"] = len(samples)
        result["gpu_monitor_errors"] = errors
        result["elapsed_seconds"] = time.monotonic() - started
        try:
            result["artifacts_after"] = artifact_fingerprint(args.server)
            result["artifacts_changed"] = result["artifacts_before"] != result["artifacts_after"]
        except OSError as exc:
            result["artifacts_changed"] = True
            result["artifact_fingerprint_error"] = str(exc)
        try:
            result["model_after"] = model_fingerprint(args.model)
            result["model_changed"] = result["model_before"] != result["model_after"]
        except (OSError, RuntimeError) as exc:
            result["model_changed"] = True
            result["model_fingerprint_error"] = str(exc)


def prepare(args, base, key, result):
    text = args.corpus.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("calibration corpus is empty")
    # Training text only; evaluation must use a different file/split.
    documents = [text[i:i + 12000] for i in range(0, min(len(text), 96000), 12000)]
    trie = args.out / (result["config"]["name"] + ".calibration-trie")
    trie.mkdir(exist_ok=True)
    nodes = bytearray(struct.pack("<8sIIQ", b"CLTNOD01", 1, 16, 0))
    reqs = bytearray(struct.pack("<8sIIQ", b"CLTREQ01", 1, 24, 0))
    count = 0
    lengths = []
    edges = {}
    for document in documents:
        tokens = request(base, key, "/tokenize", {"content": document, "add_special": True})["tokens"]
        tokens = tokens[:1024]
        if not tokens:
            continue
        parent = 0
        for token in tokens:
            edge = (parent, token)
            if edge not in edges:
                nodes.extend(struct.pack("<QiI", parent, token, 0))
                count += 1
                edges[edge] = count
            parent = edges[edge]
        reqs.extend(struct.pack("<QIIQ", parent, len(tokens), 0, 0))
        lengths.append(len(tokens))
    if not lengths:
        raise ValueError("calibration tokenizer returned no tokens")
    (trie / "nodes-000000.bin").write_bytes(nodes)
    (trie / "requests-000000.bin").write_bytes(reqs)
    result["calibration"] = {"nodes": count, "lengths": lengths, "trie": str(trie),
                             "source": str(args.corpus),
                             "source_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest()}


def check_completion(response, prompt_tokens, predict):
    """A throughput sample must actually execute the complete requested work."""
    if not isinstance(response, dict) or not isinstance(response.get("timings"), dict):
        raise RuntimeError("completion has no timing object")
    timings = response["timings"]
    if type(response.get("tokens_predicted")) is not int or response["tokens_predicted"] != predict:
        raise RuntimeError("completion did not generate the requested token budget")
    if response.get("truncated") is not False:
        raise RuntimeError("completion was truncated or its truncation state is missing")
    for field, expected in (("cache_n", 0), ("prompt_n", prompt_tokens), ("predicted_n", predict)):
        if type(timings.get(field)) is not int or timings[field] != expected:
            raise RuntimeError("completion work mismatch: " + field + " must equal " + str(expected))
    for phase, count in (("prompt", prompt_tokens), ("predicted", predict)):
        ms = timings.get(phase + "_ms")
        rate = timings.get(phase + "_per_second")
        for name, value in ((phase + "_ms", ms), (phase + "_per_second", rate)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise RuntimeError("missing or invalid timing: " + name)
        # Server durations are serialized in milliseconds. Permit rounding,
        # not mismatched token counts or stale, unrelated throughput fields.
        if not math.isclose(rate, 1000.0 * count / ms, rel_tol=1e-4, abs_tol=1e-6):
            raise RuntimeError("inconsistent completion duration/rate: " + phase)


def probe(args, base, key, result):
    result["requests"] = []
    text = args.corpus.read_text(encoding="utf-8") if args.corpus else (
        "A careful test records its inputs and checks its outputs. " * 4096)
    tokens = request(base, key, "/tokenize", {"content": text, "add_special": True})["tokens"]
    if len(tokens) < args.prompt_tokens:
        raise ValueError("corpus is shorter than the requested prompt")
    body = {"prompt": tokens[:args.prompt_tokens], "n_predict": args.predict,
            "temperature": 0, "seed": 1234, "cache_prompt": False,
            "ignore_eos": True, "n_probs": 0}
    if not all(type(token) is int and token >= 0 for token in body["prompt"]):
        raise ValueError("tokenizer returned invalid token IDs")
    # Identical lengths do not imply identical prompts. Record the exact
    # tokenized request identity without duplicating the corpus for each run.
    request_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode("utf-8")).hexdigest()
    result["workload"] = {"version": 1, "request_sha256": request_hash,
                          "prompt_tokens": args.prompt_tokens, "predict": args.predict,
                          "repeats": args.repeats,
                          "parameters": {key: value for key, value in body.items() if key != "prompt"}}
    for repeat in range(args.repeats):
        started = time.monotonic()
        response = request(base, key, "/completion", body, timeout=args.request_timeout)
        response["wall_seconds"] = time.monotonic() - started
        response["case"] = "prefill-decode-" + str(repeat)
        response["request_sha256"] = request_hash
        result["requests"].append(response)
        check_completion(response, args.prompt_tokens, args.predict)
        print(json.dumps({"config": result["config"]["name"], "repeat": repeat,
                          "timings": response.get("timings")}), flush=True)
    questions = [] if args.no_chat else [
        ("arithmetic", "只回答计算结果，不要解释：17乘以23等于多少？", [391]),
        ("extract", "记录：北仓有7箱茶，南仓有12箱茶。只输出南仓的箱数，不要解释。", [12]),
        ("sort", "Sort these integers in ascending order. Output only the list: 7, -3, 11, 0, 2.", [-3, 0, 2, 7, 11]),
    ]
    if args.needle_tokens:
        if len(tokens) < args.needle_tokens:
            raise ValueError("corpus is shorter than the requested retrieval fixture")
        filler = request(base, key, "/detokenize", {"tokens": tokens[:args.needle_tokens]})["content"]
        fact = "档案中的唯一验收编号为593174。"
        questions.append(("retrieval", "请记住下列档案事实。" + fact + "\n下面是无关资料：\n" +
                          filler + "\n请只输出档案的六位验收编号，不要解释。", [593174]))
    result["quality_checks"] = []
    for name, question, expected in questions:
        response = request(base, key, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": question}],
            "temperature": 0, "seed": 1234, "max_tokens": 384,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=args.request_timeout)
        response["case"] = name
        choices = response.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0], dict):
            result["requests"].append(response)
            raise RuntimeError("chat response must contain exactly one choice")
        choice = choices[0]
        content = choice.get("message", {}).get("content", "") or ""
        actual = [int(x) for x in re.findall(r"-?\d+", content)]
        result["quality_checks"].append({"case": name, "expected": expected, "actual": actual,
                                        "finish_reason": choice.get("finish_reason"),
                                        "passed": actual == expected and choice.get("finish_reason") == "stop"})
        result["requests"].append(response)
    # One final request also verifies recovery after the preceding workloads.
    result["final_health"] = request(base, key, "/health", timeout=5)
    if result["config"].get("require_flashprefill_plan", False):
        result["metrics_text"] = request(base, key, "/metrics", timeout=5, as_text=True)
    if result["final_health"].get("status") != "ok":
        raise RuntimeError("server was not healthy after the probes")
    if any(not check["passed"] for check in result["quality_checks"]):
        raise RuntimeError("quality checks failed; see saved per-case results")


def check_flashprefill_evidence(config, result):
    """Only completed plan counters prove that the sparse implementation ran."""
    if not config.get("require_flashprefill_plan", False):
        return
    fields = ("sparse_rows", "dense_packed_rows", "selected_blocks", "corrected_blocks",
              "visible_tokens", "exact_tokens")
    counters = {}
    for field in fields:
        name = "llamacpp:flashprefill_" + field + "_total"
        matches = re.findall(r"^" + re.escape(name) + r"\s+(\S+)\s*$",
                             result.get("metrics_text", ""), re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError("missing or ambiguous FlashPrefill counter: " + name)
        value = float(matches[0])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise RuntimeError("invalid FlashPrefill counter: " + name)
        counters[field] = int(value)
    result["flashprefill_counters"] = counters
    if (counters["sparse_rows"] + counters["dense_packed_rows"] <= 0 or
            counters["selected_blocks"] <= 0 or counters["visible_tokens"] <= 0):
        raise RuntimeError("expected completed FlashPrefill plans, but no plan work was recorded")
    if counters["exact_tokens"] > counters["visible_tokens"]:
        raise RuntimeError("inconsistent FlashPrefill token accounting")


def check_runtime_evidence(config, result):
    """Pressure gates must observe real reclaim, not merely HTTP success."""
    check_flashprefill_evidence(config, result)
    required = config.get("require_tri_drain", False)
    ceiling = config.get("max_tri_score_ms")
    if not required and ceiling is None:
        return
    text = Path(result["server_log"]).read_text(encoding="utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        match = re.search(r"TriAttention (drain|maintenance):", line)
        if not match:
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line[match.end():]))
        event = {"kind": match.group(1)}
        for field in ("before", "after", "freed"):
            event[field] = int(fields[field])
        for field in ("score_ms", "pack_ms"):
            event[field] = float(fields[field])
            if not math.isfinite(event[field]) or event[field] < 0:
                raise RuntimeError("invalid TriAttention timing in server log")
        if event["before"] < event["after"] or event["before"] - event["after"] != event["freed"]:
            raise RuntimeError("inconsistent TriAttention physical-cell accounting")
        events.append(event)
    result["tri_events"] = events
    if required and not any(event["kind"] == "drain" and event["freed"] > 0 for event in events):
        raise RuntimeError("expected a real TriAttention drain, but none freed cells")
    if ceiling is not None and (not events or any(event["score_ms"] > ceiling for event in events)):
        raise RuntimeError("TriAttention scoring exceeded the configured bound or no reclaim ran")


def validate_configs(configs):
    """Reject ambiguous matrices before starting servers or writing results."""
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs must be a nonempty list")
    names = set()
    types = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl",
             "q5_0", "q5_1", "turbo2", "turbo3", "turbo4"}
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError("each config must be an object")
        name = config.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("config names must use ASCII letters, digits, hyphens or underscores")
        if name in names:
            raise ValueError("duplicate config name: " + name)
        names.add(name)
        for field in ("ctx", "batch", "ubatch"):
            if type(config.get(field)) is not int or config[field] <= 0:
                raise ValueError(field + " must be a positive integer")
        kv = config.get("kv", config["ctx"])
        if kv != "auto" and (type(kv) is not int or kv <= 0):
            raise ValueError("kv must be a positive integer or auto")
        for field in ("k", "v"):
            if not isinstance(config.get(field), str) or config[field] not in types:
                raise ValueError("unsupported KV type for " + field)
        extra = config.get("extra", [])
        if not isinstance(extra, list) or not all(isinstance(arg, str) for arg in extra):
            raise ValueError("extra must be a list of argument strings")
        # Keep probes local and authenticated even with caller-supplied flags.
        reserved = {"--host", "--port", "--api-key", "--api-key-file",
                    "-m", "--model", "-mu", "--model-url", "-hf", "--hf-repo",
                    "--huggingface-repo", "--hf-file", "--huggingface-file"}
        if any(arg.split("=", 1)[0].replace("_", "-") in reserved for arg in extra):
            raise ValueError("extra may not override the probe listener, authentication or model source")
        env = config.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError("env must map strings to strings")
        if type(config.get("require_tri_drain", False)) is not bool:
            raise ValueError("require_tri_drain must be boolean")
        if type(config.get("require_flashprefill_plan", False)) is not bool:
            raise ValueError("require_flashprefill_plan must be boolean")
        ceiling = config.get("max_tri_score_ms")
        if ceiling is not None and (type(ceiling) not in (int, float) or not math.isfinite(ceiling) or ceiling < 0):
            raise ValueError("max_tri_score_ms must be finite and nonnegative")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "probe"])
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--no-chat", action="store_true", help="throughput/stress only; no short QA checks")
    parser.add_argument("--needle-tokens", type=int, default=0, help="additional long-context retrieval fixture")
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--request-timeout", type=int, default=240)
    args = parser.parse_args(argv)
    if not args.server.is_file() or not args.model.is_file():
        parser.error("server and model must be existing files")
    if args.mode == "prepare" and args.corpus is None:
        parser.error("prepare requires an explicit training corpus")
    if min(args.prompt_tokens, args.predict, args.repeats, args.startup_timeout, args.request_timeout) <= 0:
        parser.error("token counts, repeat counts and timeouts must be positive")
    if args.needle_tokens < 0:
        parser.error("needle token count cannot be negative")
    try:
        configs = json.loads(args.configs.read_text(encoding="utf-8")) if args.configs else [
            {"name": "baseline", "ctx": 8192, "batch": 256, "ubatch": 256, "k": "turbo4", "v": "turbo2"}]
        validate_configs(configs)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for config in configs:
        result = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            with server(args, config, result) as (base, key):
                (prepare if args.mode == "prepare" else probe)(args, base, key, result)
            if result.get("artifacts_changed"):
                raise RuntimeError("build artifacts changed during the probe; result is not valid")
            if result.get("model_changed"):
                raise RuntimeError("model file changed during the probe; result is not valid")
            if result.get("server_exit_before_cleanup") is not None:
                raise RuntimeError("server exited before runner cleanup: " + str(result["server_exit_before_cleanup"]))
            check_runtime_evidence(config, result)
            result["status"] = "completed"
        except Exception as exc:
            failed = True
            result["status"] = "failed"
            result["error"] = str(exc)
        finally:
            (args.out / (config["name"] + ".json")).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: result.get(k) for k in ["config", "status", "error", "peak_gpu_mib", "elapsed_seconds"]}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
