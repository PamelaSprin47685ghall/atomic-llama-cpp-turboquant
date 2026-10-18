#!/usr/bin/env python3
"""Exercise the exact A exception gate against real temporary Git repositories."""

import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import tarfile
import tempfile
import unittest


GUARD = runpy.run_path(str(Path(__file__).resolve().parents[1] / "check-upstream.py"))


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="boundary-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Boundary Test")
        self.git("config", "user.email", "boundary@example.invalid")
        self.git("config", "core.filemode", "true")
        (self.root / "a.cpp").write_bytes(b"upstream\n")
        self.git("add", "a.cpp")
        self.git("commit", "-qm", "upstream")
        self.base = self.git("rev-parse", "HEAD").decode().strip()
        (self.root / "fork").mkdir()
        (self.root / "fork/upstream.json").write_text(json.dumps({"upstream": {"commit": self.base}}))
        self.entries = []
        self.manifest()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.STDOUT)

    def manifest(self):
        (self.root / "fork/upstream-exceptions.json").write_text(json.dumps(
            {"version": 1, "upstream_commit": self.base, "files": self.entries}))

    def overlay(self, path="a.cpp", data=b"reviewed\n", mode="100644"):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        file.chmod(0o755 if mode == "100755" else 0o644)
        self.entries.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "mode": mode,
                             "consumers": ["fork/test-consumer"], "reason": "Exercise a public seam.",
                             "removal_condition": "Remove when the upstream pin supplies the seam."})
        self.manifest()

    def verify(self):
        return GUARD["verify"](self.root)

    def test_clean(self):
        self.assertEqual(self.verify()["overlays"], [])

    def test_reviewed_worktree_and_committed(self):
        self.overlay()
        self.assertEqual(len(self.verify()["overlays"]), 1)
        self.git("add", "a.cpp")
        self.verify()
        self.git("commit", "-qm", "reviewed seam")
        self.verify()

    def test_new_regular_file(self):
        self.overlay("include/public-seam.h")
        self.verify()
        self.git("add", "include/public-seam.h")
        self.verify()

    def test_undeclared(self):
        (self.root / "a.cpp").write_bytes(b"unreviewed\n")
        with self.assertRaisesRegex(RuntimeError, "undeclared"):
            self.verify()

    def test_changed_digest(self):
        self.overlay()
        (self.root / "a.cpp").write_bytes(b"changed after review\n")
        with self.assertRaisesRegex(RuntimeError, "content or mode"):
            self.verify()

    def test_wrong_mode(self):
        self.overlay()
        (self.root / "a.cpp").chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "content or mode"):
            self.verify()

    def test_stale(self):
        self.overlay(data=b"upstream\n")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.verify()

    def test_index_residue(self):
        (self.root / "a.cpp").write_bytes(b"unreviewed staged\n")
        self.git("add", "a.cpp")
        self.overlay()
        with self.assertRaisesRegex(RuntimeError, "index has unreviewed"):
            self.verify()

    def test_staged_deletion(self):
        self.git("rm", "--cached", "a.cpp")
        self.overlay()
        with self.assertRaisesRegex(RuntimeError, "staged deletion"):
            self.verify()

    def test_symlink(self):
        self.overlay()
        (self.root / "a.cpp").unlink()
        (self.root / "a.cpp").symlink_to("fork/upstream.json")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.verify()

    def test_hidden_index_change(self):
        self.git("update-index", "--assume-unchanged", "a.cpp")
        (self.root / "a.cpp").write_bytes(b"hidden\n")
        with self.assertRaisesRegex(RuntimeError, "assume-unchanged"):
            self.verify()

    def test_overlay_uses_captured_bytes(self):
        self.overlay(mode="100755")
        snapshot = self.verify()
        (self.root / "a.cpp").write_bytes(b"later live mutation\n")
        destination = self.root / "fork/extracted"
        destination.mkdir()
        with tarfile.open(fileobj=io.BytesIO(self.git("archive", self.base))) as archive:
            archive.extractall(destination, filter="data")
        GUARD["apply_verified_overlays"](snapshot, destination)
        self.assertEqual((destination / "a.cpp").read_bytes(), b"reviewed\n")
        self.assertEqual((destination / "a.cpp").stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
