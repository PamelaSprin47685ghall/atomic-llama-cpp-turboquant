#!/usr/bin/env python3
"""Compare the real composers with upstream-only replay on one supplied model."""

import argparse
import json
import hashlib
import math
import re
from pathlib import Path
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def offloaded_layers(output):
    """Require actual GPU placement in the pinned upstream's execution logs."""
    counts = []
    for name in ("greedy-compact", "greedy-full", "stochastic-compact", "stochastic-full",
                 "corpus-prefix", "corpus-full", "corpus-reversed-prefix"):
        log = (Path(output) / f"{name}.stderr").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"offloaded ([0-9]+)/([0-9]+) layers to GPU", log)
        require(match is not None and int(match[1]) > 0, f"{name}: no actual GPU layer offload")
        counts.append(int(match[1]))
    return min(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for results and logs")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nll-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    require(args.model.is_file(), "model does not exist")
    require(args.gpu_layers >= 0 and args.threads > 0, "invalid device or thread count")
    require(math.isfinite(args.nll_tolerance) and args.nll_tolerance >= 0, "invalid tolerance")
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.model.resolve()
    binaries = args.bin_dir.resolve()
    commands = []

    def run(name, command, expect_success=True):
        command = [str(part) for part in command]
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, capture_output=True, check=False)
        (args.output / f"{name}.stdout").write_bytes(result.stdout)
        (args.output / f"{name}.stderr").write_bytes(result.stderr)
        commands.append({"name": name, "command": command, "returncode": result.returncode})
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if expect_success:
            require(result.returncode == 0, f"{name} failed; see its stderr")
            return json.loads(result.stdout)
        require(result.returncode != 0, f"{name} unexpectedly succeeded")
        return None

    generation = [binaries / "llama-compose", "-m", model, "-p",
                  "The capital of France is", "-c", 256, "-n", 24,
                  "-ngl", args.gpu_layers, "-t", args.threads, "--seed", 42, "--json"]
    samples = {}
    for temperature, label in ((0, "greedy"), (0.8, "stochastic")):
        accelerated = run(f"{label}-compact", generation + ["--temp", temperature])
        reference = run(f"{label}-full", generation + ["--temp", temperature, "--no-compact"])
        for field in ("tokens", "text", "stop", "prompt_tokens"):
            require(accelerated[field] == reference[field], f"{label}: {field} differs")
        require(accelerated["tokens"], f"{label}: no token was generated")
        require(reference["compact_rows"] == 0, "reference used compact rows")
        if temperature == 0:
            require(accelerated["compact_rows"] > 0, "greedy did not exercise B")
        else:
            require(accelerated["compact_rows"] == 0, "stochastic path must use full rows")
        samples[label] = {"tokens_equal": True, "tokens": len(accelerated["tokens"]),
                          "compact_rows": accelerated["compact_rows"],
                          "full_rows": accelerated["full_rows"]}

    prompts = args.output / "prompts.txt"
    prefix = "A test of shared prefix replay. The first three prime numbers are "
    requests = [prefix + "two, three, and five. Continue with seven and eleven.",
                prefix + "two, three, and five.",
                prefix + "2, 3, and 5.",
                prefix + "two, three, and five.",
                "This is a separate request.", prefix]
    prompts.write_text("\n".join(requests) + "\n", encoding="utf-8")
    corpus = args.output / "corpus"
    corpus_cli = [binaries / "llama-corpus"]
    built = run("corpus-build", corpus_cli + ["build", "-m", model, "--input", prompts, "--corpus", corpus])
    score = corpus_cli + ["score", "-m", model, "--corpus", corpus, "-c", 256,
                          "-ngl", args.gpu_layers, "-t", args.threads]
    prefix_score = run("corpus-prefix", score)
    full_score = run("corpus-full", score + ["--reference"])
    for field in ("requests", "nodes", "request_tokens", "scored_positions"):
        require(prefix_score[field] == full_score[field], f"corpus {field} differs")
    require(built["requests"] == len(requests), "request multiplicity was lost")
    require(prefix_score["decoded_tokens"] == built["nodes"], "prefix did not decode each unique node once")
    require(full_score["decoded_tokens"] == built["request_tokens"], "full replay omitted request tokens")
    require(prefix_score["decoded_tokens"] < full_score["decoded_tokens"], "prefix reuse was not exercised")
    difference = abs(prefix_score["mean_nll"] - full_score["mean_nll"])
    require(math.isfinite(difference) and difference <= args.nll_tolerance,
            f"mean NLL differs by {difference}, tolerance {args.nll_tolerance}")
    def compare_requests(left, right):
        require(len(left) == len(right), "per-request scores are missing")
        maximum = 0.0
        for candidate, reference in zip(left, right):
            count = reference["scored_positions"]
            require(candidate["scored_positions"] == count, "per-request position count differs")
            delta = abs(candidate["nll_sum"] - reference["nll_sum"]) / max(count, 1)
            require(math.isfinite(delta) and delta <= args.nll_tolerance,
                    f"per-request mean NLL differs by {delta}")
            maximum = max(maximum, delta)
        return maximum

    per_request_difference = compare_requests(prefix_score["request_scores"], full_score["request_scores"])
    reversed_prompts = args.output / "prompts-reversed.txt"
    reversed_prompts.write_text("\n".join(reversed(requests)) + "\n", encoding="utf-8")
    reversed_corpus = args.output / "corpus-reversed"
    run("corpus-reversed-build", corpus_cli + ["build", "-m", model, "--input", reversed_prompts,
                                             "--corpus", reversed_corpus])
    reversed_score = run("corpus-reversed-prefix", corpus_cli + [
        "score", "-m", model, "--corpus", reversed_corpus, "-c", 256,
        "-ngl", args.gpu_layers, "-t", args.threads])
    order_difference = compare_requests(list(reversed(reversed_score["request_scores"])),
                                        full_score["request_scores"])
    original_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in corpus.iterdir()}
    run("corpus-context-rejected", score + ["-c", 1], expect_success=False)
    run("corpus-overwrite-rejected", corpus_cli + ["build", "-m", model, "--input", prompts, "--corpus", corpus],
        expect_success=False)
    # Rejection must leave the already validated recorded files byte-identical.
    after_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in corpus.iterdir()}
    require(after_files == original_files, "rejected corpus write changed the recorded files")
    gpu_count = offloaded_layers(args.output) if args.gpu_layers else 0
    result = {"model": model.name, "gpu_layers": args.gpu_layers,
              "observed_offloaded_layers": gpu_count, "sampling": samples,
              "corpus": {"requests": built["requests"], "unique_tokens": built["nodes"],
                         "full_tokens": built["request_tokens"],
                         "mean_nll_absolute_difference": difference,
                         "maximum_per_request_mean_nll_difference": per_request_difference,
                         "reversed_order_maximum_mean_nll_difference": order_difference,
                         "mean_nll_tolerance": args.nll_tolerance,
                         "prefix_mean_nll": prefix_score["mean_nll"],
                         "full_mean_nll": full_score["mean_nll"]},
              "failure_checks": ["context limit", "no overwrite", "unchanged corpus after rejection"]}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
