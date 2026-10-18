#!/usr/bin/env python3
"""Reject tracked or untracked source changes outside fork/."""

import json
from pathlib import Path
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "fork/upstream.json").read_text())
    base = manifest["upstream"]["commit"]
    resolved = git(root, "rev-parse", f"{base}^{{commit}}").decode().strip()
    if resolved != base:
        raise RuntimeError("upstream pin must be a full commit ID")
    subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", base, "HEAD"], check=True)
    tracked = git(root, "diff", "--name-only", "--no-renames", "-z", base, "--").decode().split("\0")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    changed = sorted(set(path for path in tracked + untracked if path))
    outside = [path for path in changed if not path.startswith("fork/")]
    if outside:
        raise RuntimeError("upstream boundary changed:\n" + "\n".join(outside))
    subprocess.run(["git", "-C", str(root), "diff", "--check", base, "--"], check=True)
    print(json.dumps({"upstream": base, "changed_upstream_paths": 0,
                      "extension_paths": len(changed)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"boundary check failed: {error}", file=sys.stderr)
        sys.exit(1)
