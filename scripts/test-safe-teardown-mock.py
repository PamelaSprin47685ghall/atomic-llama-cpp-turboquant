#!/usr/bin/env python3
"""Assert drain-fail retains resources / no fictitious completion (C03).

Mirrors ggml_backend_vk_tp5_comm_free_safe semantics from
ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp:
  - drain failure => return false and RETAIN resources (do not delete)
  - never fabricate completion of submitted work
  - wait only actually submitted values

No 5-GPU required. Usage: python3 scripts/test-safe-teardown-mock.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "scripts" / "experiment-queue.py"


def _load_queue_module():
    spec = importlib.util.spec_from_file_location("experiment_queue", QUEUE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {QUEUE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_queue"] = mod
    spec.loader.exec_module(mod)
    return mod


eq = _load_queue_module()


class SafeTeardownMockTests(unittest.TestCase):
    """Mirror ggml_backend_vk_tp5_comm_free_safe offline (no GPU)."""

    def _comm(
        self,
        *,
        ranks: int = 2,
        submitted: list[int] | None = None,
        completed: list[int] | None = None,
        drain_should_fail: bool = False,
    ) -> eq.MockTp5Comm:
        return eq.MockTp5Comm(
            ranks=[eq.MockTp5Rank(i) for i in range(ranks)],
            submitted=list(submitted or []),
            completed=list(completed or []),
            drain_should_fail=drain_should_fail,
        )

    def test_drain_fail_retains_resources(self) -> None:
        """Forced drain failure must retain resources and return False."""
        comm = self._comm(
            submitted=[1, 2],
            completed=[1, 2],
            drain_should_fail=True,
        )
        ok = eq.ggml_backend_vk_tp5_comm_free_safe_mock(comm)
        self.assertFalse(ok)
        self.assertTrue(comm.resources_retained)
        self.assertTrue(comm.retained)
        self.assertFalse(comm.deleted)
        self.assertTrue(all(r.resources_alive for r in comm.ranks))
        # Must not invent extra completions on failure.
        self.assertEqual(comm.completed, [1, 2])

    def test_unknown_completion_retains_no_fiction(self) -> None:
        """Submitted but incomplete work => retain; never pad completed."""
        comm = self._comm(submitted=[1, 2], completed=[1])
        before_completed = list(comm.completed)
        ok = eq.ggml_backend_vk_tp5_comm_free_safe_mock(comm)
        self.assertFalse(ok)
        self.assertTrue(comm.resources_retained)
        self.assertFalse(comm.deleted)
        self.assertEqual(
            comm.completed,
            before_completed,
            "teardown must not fabricate missing completions",
        )
        self.assertNotEqual(
            set(comm.submitted),
            set(comm.completed),
            "incomplete submissions must remain visible",
        )

    def test_fabricated_completion_treated_as_drain_fail(self) -> None:
        """Completed values outside submitted set => drain fail + retain."""
        comm = self._comm(submitted=[1], completed=[1, 99])
        ok = eq.ggml_backend_vk_tp5_comm_free_safe_mock(comm)
        self.assertFalse(ok)
        self.assertTrue(comm.resources_retained)
        self.assertFalse(comm.deleted)
        # Do not rewrite/fabricate a sanitized completion set.
        self.assertEqual(comm.completed, [1, 99])

    def test_successful_drain_releases(self) -> None:
        """All submitted values completed => free resources and delete."""
        comm = self._comm(submitted=[1, 2], completed=[1, 2])
        ok = eq.ggml_backend_vk_tp5_comm_free_safe_mock(comm)
        self.assertTrue(ok)
        self.assertTrue(comm.deleted)
        self.assertFalse(comm.resources_retained)
        self.assertTrue(all(not r.resources_alive for r in comm.ranks))

    def test_null_comm_ok(self) -> None:
        self.assertTrue(eq.ggml_backend_vk_tp5_comm_free_safe_mock(None))

    def test_cli_mock_teardown_drain_fail_json(self) -> None:
        """CLI path used by --help examples also retains on drain-fail."""
        import json
        import subprocess

        r = subprocess.run(
            [
                sys.executable,
                str(QUEUE_PATH),
                "mock-teardown",
                "--submitted",
                "1,2",
                "--completed",
                "1",
                "--force-drain-fail",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(r.stdout)
        self.assertFalse(data["drain_ok"])
        self.assertTrue(data["resources_retained"])
        self.assertFalse(data["deleted"])
        self.assertFalse(data["fabricated_completion"])
        self.assertEqual(data["semantics"], "mirror ggml_backend_vk_tp5_comm_free_safe")


if __name__ == "__main__":
    unittest.main()
