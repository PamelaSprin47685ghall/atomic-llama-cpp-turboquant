#!/usr/bin/env python3
"""Exercise HIP tool argument parsing without launching device work."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def check(command, expected, fragment, directory):
    environment = dict(os.environ, HIP_VISIBLE_DEVICES="")
    result = subprocess.run(command, cwd=directory, env=environment,
                            capture_output=True, text=True, timeout=15)
    if result.returncode != expected or fragment not in result.stdout + result.stderr:
        raise AssertionError("%r: expected exit %d and %r; got %d\n%s%s" %
                             (command, expected, fragment, result.returncode,
                              result.stdout, result.stderr))
    if list(Path(directory).iterdir()):
        raise AssertionError("argument parsing created an evidence file")


def main():
    probe, clock = [str(Path(path).resolve()) for path in sys.argv[1:]]
    with tempfile.TemporaryDirectory(prefix="tq-probe-cli-") as directory:
        check([probe, "--help"], 0, "usage:", directory)
        check([clock, "--help"], 0, "usage:", directory)
        check([clock, "--unknown"], 2, "usage:", directory)
        for text in ("", "0", "02000", "+2000", " 2000", "2000 ", "-1",
                     "1x", "4294967296", "18446744073709551616"):
            check([probe, "--rounds", text], 2, "--rounds must", directory)
        for option in ("--output", "--bench"):
            check([probe, option, ""], 2, option + " must name a nonempty file", directory)
            check([probe, option], 2, "unknown argument:", directory)
        check([probe, "--output", "same.json", "--bench", "same.json"],
              2, "must name different files", directory)
        check([probe, "--unknown"], 2, "unknown argument:", directory)
    print("PASS: HIP tools reject invalid CLI input before device access")


if __name__ == "__main__":
    main()
