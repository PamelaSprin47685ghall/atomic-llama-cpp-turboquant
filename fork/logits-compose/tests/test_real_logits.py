#!/usr/bin/env python3
"""Qualify real teacher-forced rows, format compatibility, and output failures."""

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def invalid_number(value):
    raise ValueError(f"non-finite JSON number: {value}")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_tape(path):
    require(path.stat().st_size <= 128 * 1024 * 1024, "test tape exceeds its 128 MiB fixture bound")
    with path.open("rb") as stream:
        header = stream.read(24)
        require(len(header) == 24 and header[:8] == b"TP5LOG2\0", "invalid tape header")
        vocabulary, prefill, steps, total = struct.unpack("<4I", header[8:])
        require(0 < vocabulary <= 2000000 and 0 < prefill <= total <= 256, "invalid fixture dimensions")
        require(total == prefill + steps, "input count and row dimensions disagree")
        require(path.stat().st_size == 24 + 4 * total + 4 * vocabulary * (steps + 1), "wrong tape byte count")
        tokens = list(struct.unpack(f"<{total}i", stream.read(4 * total)))
        require(all(0 <= token < vocabulary for token in tokens), "input token is outside vocabulary")
        rows = []
        for _ in range(steps + 1):
            row = array.array("f")
            require(row.itemsize == 4, "test requires IEEE binary32 arrays")
            row.frombytes(stream.read(4 * vocabulary))
            if sys.byteorder != "little":
                row.byteswap()
            require(len(row) == vocabulary and all(math.isfinite(value) for value in row), "invalid logit row")
            rows.append(row)
        require(not stream.read(1), "unexpected trailing bytes")
    return {"vocabulary": vocabulary, "prefill": prefill, "tokens": tokens, "rows": rows}


def row_error(left, right):
    require(len(left) == len(right), "row dimensions changed")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--batch-binary", type=Path)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir)).resolve()
    prompt = output / "input.txt"
    prompt.write_text("The red boat crossed the lake. A blue bird followed the boat.", encoding="utf-8")
    binary = str(args.binary.resolve())
    common = ["--model", str(args.model.resolve()), "--prompt-file", str(prompt),
              "--ctx-size", "256", "--threads", "2", "--gpu-layers", str(args.gpu_layers)]
    records = []

    def execute(name, command, expected=0, stdout_sink=None):
        completed = subprocess.run(command, stdout=stdout_sink or subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        (output / f"{name}.stdout").write_bytes(completed.stdout or b"")
        (output / f"{name}.stderr").write_bytes(completed.stderr)
        records.append({"name": name, "command": command, "returncode": completed.returncode})
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        require(completed.returncode == expected,
                f"{name}: expected {expected}, got {completed.returncode}: {completed.stderr[-2000:].decode('utf-8', 'replace')}")
        require(not list(output.glob(".tq-logits-*.tmp")), f"{name}: an owned temporary remains")
        return completed

    def collect(name, prefill=1, batch=1):
        path = output / f"{name}.bin"
        completed = execute(name, [binary, *common, "--output", str(path), "--prefill", str(prefill),
                                   "--batch-size", str(batch)])
        metadata = json.loads(completed.stdout, parse_constant=invalid_number)
        tape = read_tape(path)
        require(metadata["input_tokens"] == len(tape["tokens"]) and metadata["rows"] == len(tape["rows"]),
                "summary and tape dimensions differ")
        require(metadata["prefill"] == prefill and metadata["bytes"] == path.stat().st_size, "summary lost tape settings")
        require(metadata["sampled"] is False and metadata["parse_special"] is False, "input policy changed")
        offload = re.search(rb"offloaded (\d+)/(\d+) layers to GPU", completed.stderr)
        observed = int(offload.group(1)) if offload else 0
        require(args.gpu_layers <= 0 or observed > 0, "GPU request did not offload any model layers")
        records[-1].update({"summary": metadata, "gpu_layers_observed": observed, "sha256": digest(path)})
        return path, tape

    full_path, full = collect("all-rows")
    repeat_path, repeat = collect("repeat")
    require(digest(full_path) == digest(repeat_path), "identical collector runs changed tape bytes")
    require(len(full["tokens"]) >= 4, "fixture needs at least four tokens")
    _, suffix = collect("suffix", prefill=3)
    _, prefix = collect("prefix-only", prefill=len(full["tokens"]))
    require(full["tokens"] == suffix["tokens"] == prefix["tokens"], "prefill selection changed teacher-forced inputs")
    suffix_error = max(row_error(a, b) for a, b in zip(full["rows"][2:], suffix["rows"]))
    prefix_error = row_error(full["rows"][-1], prefix["rows"][0])
    require(suffix_error <= 1e-5 and prefix_error <= 1e-5, "same one-token geometry changed selected output rows")
    chunked_path, chunked = collect("chunked-prefix", prefill=len(full["tokens"]), batch=8)
    require(full["tokens"] == chunked["tokens"] and len(chunked["rows"]) == 1, "chunked prefill lost input identity")
    chunked_error = row_error(full["rows"][-1], chunked["rows"][0])

    comparison = output / "comparison.json"
    execute("comparator", [sys.executable, str(args.comparator.resolve()), str(full_path), str(repeat_path),
                           "--atol", "0", "--rtol", "0", "--output", str(comparison)])
    comparator = json.loads(comparison.read_text(encoding="utf-8"), parse_constant=invalid_number)
    preserved = digest(full_path)
    execute("existing-output", [binary, *common, "--output", str(full_path)], expected=1)
    require(digest(full_path) == preserved, "output refusal changed the existing tape")
    for name, extra in (("too-small-context", ["--ctx-size", "2"]),
                        ("file-budget", ["--max-output-bytes", "24"])):
        path = output / f"{name}.bin"
        execute(name, [binary, *common, "--output", str(path), *extra], expected=2)
        require(not path.exists(), f"{name}: failed inference published a tape")

    batch_oracle = "skipped_no_batch_binary"
    if args.batch_binary:
        result = execute("batch-oracle", [str(args.batch_binary.resolve()), "--model", str(args.model.resolve()),
                         "--prompts", str(prompt), "--ctx-size", "256", "--total-kv", "256", "--parallel", "1",
                         "--predict", "1", "--batch-size", "8", "--ubatch-size", "8", "--threads", "2",
                         "--gpu-layers", str(args.gpu_layers), "--ignore-eog", "--no-phase-memory", "--json"])
        events = [json.loads(line, parse_constant=invalid_number) for line in result.stdout.splitlines()]
        rows = [row for event in events if event["event"] == "batch" for row in event["rows"]]
        samples = [sample for event in events if event["event"] == "batch" for sample in event["samples"]]
        require([row["token"] for row in rows] == chunked["tokens"], "independent C consumed different prompt tokens")
        require(len(samples) == 1, "independent C did not sample the final prompt row")
        sample = samples[0]
        logits = chunked["rows"][0]
        require(abs(max(logits) - sample["logit_best"]) <= 1e-5 and
                abs(logits[sample["token"]] - max(logits)) <= 1e-5,
                "collector final row differs from independent C's same-shape greedy row")
        batch_oracle = {"status": "passed", "token": sample["token"], "best_logit": sample["logit_best"]}

    stdout_failure = "skipped_no_dev_full"
    if Path("/dev/full").exists():
        path = output / "stdout-failed.bin"
        with Path("/dev/full").open("wb") as sink:
            result = execute("stdout-failed", [binary, *common, "--output", str(path),
                             "--prefill", str(len(full["tokens"])), "--batch-size", "8"], expected=1, stdout_sink=sink)
        require(b"complete tape published, but stdout summary failed" in result.stderr, "stdout failure lacked a precise error")
        require(digest(path) == digest(chunked_path), "stdout failure removed or changed the complete published tape")
        stdout_failure = "failed_visibly_complete_tape_preserved"

    summary = {"status": "passed", "output_dir": str(output), "input_tokens": len(full["tokens"]),
               "vocabulary": full["vocabulary"], "repeat_bitwise_equal": True,
               "same_geometry_suffix_max_abs": suffix_error, "same_geometry_prefix_max_abs": prefix_error,
               "chunked_vs_scalar_max_abs_diagnostic": chunked_error, "batch_oracle": batch_oracle,
               "comparator": comparator, "stdout_failure": stdout_failure,
               "prompt_sha256": digest(prompt), "binary_sha256": digest(args.binary.resolve()),
               "commands": records, "limits": [
                   "Chunked versus scalar row differences are diagnostics; no cross-shape equality is assumed.",
                   "The tape contains no model or build identity; preserve the run receipts separately.",
                   "This verifies teacher-forced collection, not model quality, sampling, or throughput."]}
    (output / "commands.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "commands"}, allow_nan=False))


if __name__ == "__main__":
    main()
