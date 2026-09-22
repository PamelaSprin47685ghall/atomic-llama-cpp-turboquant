#!/usr/bin/env python3
"""C03 mock safe-teardown unit tests (no 5 GPUs)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "experiment-queue.py"


class SafeTeardownMockTests(unittest.TestCase):
    def test_mock_teardown_retains_on_drain_fail(self) -> None:
        r = subprocess.run(
            [sys.executable, str(QUEUE), "mock-teardown"],
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

    def test_envelope_inert_not_executable(self) -> None:
        r = subprocess.run(
            [sys.executable, str(QUEUE), "validate-envelope"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(r.stdout)
        self.assertTrue(data["structural_ok"])
        self.assertFalse(data["executable"])


if __name__ == "__main__":
    unittest.main()
