#!/usr/bin/env python3
"""Check real CLI failures without a model or a GPU."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    with tempfile.TemporaryDirectory(prefix="tq-logits-cli-") as temporary:
        root = Path(temporary)
        prompt = root / "prompt.txt"
        output = root / "output.bin"
        prompt.write_text("A small input.\n", encoding="utf-8")
        base = ["--model", str(root / "absent.gguf"), "--prompt-file", str(prompt), "--output", str(output)]

        def run(arguments, expected):
            result = subprocess.run([binary, *arguments], text=True, capture_output=True, timeout=30)
            if result.returncode != expected:
                raise AssertionError((arguments, result.returncode, result.stdout, result.stderr))
            if list(root.glob(".tq-logits-*.tmp")):
                raise AssertionError("temporary output was not removed")
            return result

        run(["--help"], 0)
        run([], 2)
        for arguments in (["--unknown"], ["--model"], ["--ctx-size", "0"], ["--ctx-size", "2000001"],
                          ["--threads", "-1"], ["--threads", "1.5"], ["--threads", "1junk"],
                          ["--threads", "2147483647"],
                          ["--max-output-bytes", "18446744073709551616"], ["--ctx-size", "1", "--prefill", "2"]):
            run([*base, *arguments], 2)
        output.write_bytes(b"keep")
        result = run(base, 1)
        if "already exists" not in result.stderr or output.read_bytes() != b"keep":
            raise AssertionError("existing output was not preserved")
        output.unlink()
        output.symlink_to(root / "absent-target")
        run(base, 1)
        if not output.is_symlink():
            raise AssertionError("output symlink changed")
        output.unlink()
        for contents in (b"", b"\xc0\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x82"):
            prompt.write_bytes(contents)
            run(base, 2)
            if output.exists():
                raise AssertionError("invalid input published a tape")
        prompt.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
        run(base, 2)
        prompt.write_text("A small input.\n", encoding="utf-8")
        run(base, 1)
        if output.exists():
            raise AssertionError("model load failure published a tape")
    print("CLI: all checks passed")


if __name__ == "__main__":
    main()
