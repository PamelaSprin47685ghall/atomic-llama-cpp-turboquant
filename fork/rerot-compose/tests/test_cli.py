#!/usr/bin/env python3
"""Check public CLI validation without opening a model or device."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def run(binary, arguments):
    result = subprocess.run(
        [str(binary), *arguments], capture_output=True, text=True,
        encoding="utf-8", errors="strict", timeout=15, check=False,
    )
    assert result.returncode >= 0, (arguments, result.returncode, result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    result = run(binary, ["--help"])
    assert result.returncode == 0, result.stderr
    for text in ("--stage", "--after", "--parallel", "--split-mode",
                 "--worker-tokens", "--json", "TP5", "LateBind", "MTP"):
        assert text in result.stdout, ("missing help contract", text)

    with tempfile.TemporaryDirectory(prefix="rerot-cli-") as directory:
        missing_model = str(Path(directory) / "must-not-open.gguf")
        base = ["--model", missing_model, "--prompt", "Check the interface.",
                "--stage", "a=Return a short answer."]
        cases = [
            ([], "are required"),
            (["--model"], "missing value"),
            (["--unknown"], "unknown option"),
            (base + ["--stage", "a=Duplicate."], "duplicate stage"),
            (base + ["--stage", "bad name=Text."], "invalid stage name"),
            (base + ["--stage", "empty="], "requires NAME=VALUE"),
            (base + ["--stage", "blank=   "], "invalid or duplicate node"),
            (base + ["--after", "a=missing"], "unknown stage"),
            (base + ["--after", "a=a"], "invalid or duplicate edge"),
            (base + ["--stage", "b=Text.", "--after", "b=a", "--after", "a=b"], "cycle"),
            (base + ["--stage", "b=Text.", "--after", "b=a", "--after", "b=a"],
             "invalid or duplicate edge"),
            (base + ["--parallel", "0"], "invalid unsigned decimal"),
            (base + ["--parallel", "-1"], "invalid unsigned decimal"),
            (base + ["--parallel", "2x"], "invalid unsigned decimal"),
            (base + ["--ctx-size", "2147483648"], "invalid unsigned decimal"),
            (base + ["--seed", "4294967295"], "invalid unsigned decimal"),
            (base + ["--temperature", "nan"], "finite non-negative"),
            (base + ["--temperature", "inf"], "finite non-negative"),
            (base + ["--temperature", "-0.25"], "finite non-negative"),
            (base + ["--temperature", "0.25x"], "finite non-negative"),
            (base + ["--batch-size", "1", "--ubatch-size", "2"], "exceeds --batch-size"),
            (base + ["--split-mode", "tp5"], "requires none, layer, or row"),
            (base + ["--phase-memory", "--no-phase-memory"], "more than once"),
            (base + ["--parallel", "1", "--parallel", "2"], "more than once"),
            (base + ["--synthesis", ""], "must not be empty"),
        ]
        for flag in ("--tp5", "--latebind", "--mtp"):
            cases.append((base + [flag], "execution provider"))
        for arguments, expected in cases:
            result = run(binary, ["--json", *arguments])
            assert result.returncode == 1, (arguments, result.returncode, result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            assert len(events) == 1, ("validation produced work events", arguments, events)
            event = events[0]
            assert event.get("event") == "error" and event.get("complete") is False, event
            assert expected in event.get("message", ""), (arguments, expected, event)
            # These failures must precede model loading and device selection.
            assert missing_model not in result.stderr, ("opened model during validation", result.stderr)
            assert "load_tensors" not in result.stderr and "load_model" not in result.stderr, result.stderr
    print(f"PASS: help and {len(cases)} invalid-input cases; no model execution was requested")


if __name__ == "__main__":
    main()
