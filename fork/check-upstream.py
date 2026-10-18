#!/usr/bin/env python3
"""Verify the upstream pin and the exact, explicitly reviewed A overlays."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(["git", "--no-replace-objects", "-C", str(root), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular_file(root, relative):
    if not isinstance(relative, str):
        raise RuntimeError(f"invalid repository path: {relative!r}")
    path = PurePosixPath(relative)
    if (not isinstance(relative, str) or not relative or path.is_absolute() or
            path.as_posix() != relative or ".." in path.parts or
            path.parts[0] == ".git" or "\\" in relative or
            any(ord(ch) < 32 for ch in relative)):
        raise RuntimeError(f"invalid repository path: {relative!r}")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"symlink is not an allowed overlay or metadata file: {relative}")
    info = current.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"not a regular file: {relative}")
    return current.read_bytes(), "100755" if info.st_mode & 0o111 else "100644"


def verify(root):
    """Return owned, verified overlay bytes. A caller must not re-read live files."""
    root = Path(root).resolve()
    pin_bytes, _ = regular_file(root, "fork/upstream.json")
    pin = json.loads(pin_bytes)
    base = pin["upstream"]["commit"]
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base):
        raise RuntimeError("upstream pin must be a full commit ID")
    resolved = git(root, "rev-parse", f"{base}^{{commit}}").decode().strip()
    if resolved != base:
        raise RuntimeError("upstream pin does not resolve to the recorded commit")
    subprocess.run(["git", "--no-replace-objects", "-C", str(root),
                    "merge-base", "--is-ancestor", base, "HEAD"], check=True)
    if git(root, "ls-files", "--unmerged", "-z"):
        raise RuntimeError("unmerged index entries remain")
    for record in git(root, "ls-files", "-v", "-z").split(b"\0"):
        if record and not record[2:].startswith(b"fork/") and (record[:1].islower() or record[:1] == b"S"):
            raise RuntimeError("A index entries may not use assume-unchanged or skip-worktree")

    manifest_bytes, _ = regular_file(root, "fork/upstream-exceptions.json")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or set(manifest) != {"version", "upstream_commit", "files"}:
        raise RuntimeError("exception manifest must contain exactly version, upstream_commit, files")
    if manifest["version"] != 1 or manifest["upstream_commit"] != base or not isinstance(manifest["files"], list):
        raise RuntimeError("exception manifest version or upstream pin is wrong")
    entries = {}
    required = {"path", "sha256", "mode", "consumers", "reason", "removal_condition"}
    for item in manifest["files"]:
        if not isinstance(item, dict) or set(item) != required:
            raise RuntimeError("each exception must contain path, sha256, mode, consumers, reason, removal_condition")
        path = item["path"]
        if not isinstance(path, str) or path.startswith("fork/") or path in entries:
            raise RuntimeError(f"invalid or duplicate A exception path: {path!r}")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise RuntimeError(f"invalid SHA-256 for {path}")
        if item["mode"] not in ("100644", "100755"):
            raise RuntimeError(f"only regular Git file modes are allowed: {path}")
        if (not isinstance(item["consumers"], list) or not item["consumers"] or
                any(not isinstance(value, str) or not value.strip() for value in item["consumers"])):
            raise RuntimeError(f"named consumers are required: {path}")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ("reason", "removal_condition")):
            raise RuntimeError(f"reason and removal condition are required: {path}")
        # Validate canonical paths, parent symlinks, and file type before copying.
        data, mode = regular_file(root, path)
        if digest(data) != item["sha256"] or mode != item["mode"]:
            raise RuntimeError(f"reviewed content or mode differs: {path}")
        entries[path] = {**item, "data": data}

    tracked = git(root, "diff", "--name-only", "--no-renames", "-z", base, "--").decode().split("\0")
    staged = git(root, "diff", "--cached", "--name-only", "--no-renames", "-z", base, "--").decode().split("\0")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    changed = {path for path in tracked + staged + untracked if path}
    outside = {path for path in changed if not path.startswith("fork/")}
    if outside != set(entries):
        undeclared = sorted(outside - set(entries))
        stale = sorted(set(entries) - outside)
        raise RuntimeError(f"A exception set differs; undeclared={undeclared}, stale={stale}")

    base_tree = {}
    for record in git(root, "ls-tree", "-r", "--full-tree", "-z", base).split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            base_tree[path.decode()] = (mode, kind, oid)
    index = {}
    for record in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode().split()
            index[path.decode()] = (mode, oid, stage)
    for path, entry in entries.items():
        original = base_tree.get(path)
        if original and original[:2] not in (("100644", "blob"), ("100755", "blob")):
            raise RuntimeError(f"an overlay cannot replace a symlink or submodule: {path}")
        if original:
            old_data = git(root, "cat-file", "blob", original[2])
            if old_data == entry["data"] and original[0] == entry["mode"]:
                raise RuntimeError(f"unused exception matches upstream exactly: {path}")
        indexed = index.get(path)
        if indexed:
            mode, oid, _ = indexed
            unchanged_index = original and (mode, oid) == (original[0], original[2])
            reviewed_index = mode == entry["mode"] and digest(git(root, "cat-file", "blob", oid)) == entry["sha256"]
            if not (unchanged_index or reviewed_index):
                raise RuntimeError(f"index has unreviewed content or mode: {path}")
        elif original:
            raise RuntimeError(f"staged deletion of an A file is not an overlay: {path}")

    subprocess.run(["git", "--no-replace-objects", "-C", str(root), "diff", "--check", base, "--"], check=True)
    summary = {"upstream": base, "exception_manifest_sha256": digest(manifest_bytes),
               "changed_upstream_paths": len(entries), "extension_paths": len(changed - outside),
               "reviewed_overlays": [{key: value for key, value in entry.items() if key != "data"}
                                     for entry in entries.values()]}
    return {"base": base, "overlays": list(entries.values()), "summary": summary,
            "manifest_bytes": manifest_bytes, "pin_bytes": pin_bytes}


def apply_verified_overlays(snapshot, destination):
    """Overlay captured bytes onto an archive of snapshot['base'], never HEAD."""
    destination = Path(destination).resolve()
    for entry in snapshot["overlays"]:
        if digest(entry["data"]) != entry["sha256"]:
            raise RuntimeError(f"captured overlay was altered: {entry['path']}")
        target = destination / entry["path"]
        current = destination
        for part in PurePosixPath(entry["path"]).parts:
            current = current / part
            if current.is_symlink():
                raise RuntimeError(f"archive overlay crosses a symlink: {entry['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry["data"])
        target.chmod(0o755 if entry["mode"] == "100755" else 0o644)
        if regular_file(destination, entry["path"]) != (entry["data"], entry["mode"]):
            raise RuntimeError(f"materialized overlay differs: {entry['path']}")


if __name__ == "__main__":
    try:
        result = verify(Path(__file__).resolve().parent.parent)
        print(json.dumps(result["summary"], indent=2))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"boundary check failed: {error}", file=sys.stderr)
        sys.exit(1)
