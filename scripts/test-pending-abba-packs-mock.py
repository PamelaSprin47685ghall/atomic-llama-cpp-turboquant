#!/usr/bin/env python3
"""Test pending R14/TM5 ABBA packs in PLAN/experiment-queue/pending-packs/."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_SCRIPT = ROOT / "scripts" / "experiment-queue.py"
PACKS_DIR = ROOT / "PLAN" / "experiment-queue" / "pending-packs"

import sys
sys.path.insert(0, str(ROOT / "scripts"))
import importlib
eq = importlib.import_module("experiment-queue")


class TestPendingAbbaPacksMock(unittest.TestCase):
    def test_packs_exist_and_count(self) -> None:
        packs = sorted(PACKS_DIR.glob("*.json"))
        self.assertGreaterEqual(
            len(packs),
            2,
            f"Expected at least two pending pack JSON files in {PACKS_DIR}, found {len(packs)}",
        )

    def test_pack_schema_and_unapproved_status(self) -> None:
        packs = sorted(PACKS_DIR.glob("*.json"))
        for p in packs:
            with self.subTest(pack=p.name):
                data = json.loads(p.read_text(encoding="utf-8"))
                
                # Check envelope schema validation
                problems = eq.validate_envelope(data, require_approved=False)
                # Approval problems are expected since approval_status != APPROVED
                structural_problems = [
                    prob for prob in problems
                    if not prob.startswith("approval_status")
                    and not prob.startswith("document_kind")
                ]
                self.assertEqual(
                    structural_problems,
                    [],
                    f"Structural schema violations in {p.name}: {structural_problems}",
                )

                # Required contract fields
                self.assertEqual(
                    data.get("approval_status"),
                    "NOT_APPROVED",
                    f"{p.name} approval_status must be NOT_APPROVED",
                )
                self.assertEqual(
                    data.get("document_kind"),
                    "PENDING_EXPERIMENT_PACK",
                    f"{p.name} document_kind must be PENDING_EXPERIMENT_PACK",
                )
                self.assertIsNone(data.get("approved_by"))
                self.assertIsNone(data.get("approved_at"))

                # Completion contract safety invariants
                contract = data.get("completion_contract", {})
                self.assertTrue(contract.get("wait_only_actually_submitted_values"))
                self.assertTrue(contract.get("retain_resources_if_completion_unknown"))
                self.assertFalse(contract.get("timeout_is_cancellation"))

                # Forbidden recovery must prevent fabricated signals
                forbidden = data.get("forbidden_recovery", [])
                self.assertIn("fabricate_completion_signal", forbidden)
                self.assertIn("release_inflight_resources_without_completion_proof", forbidden)

    def test_no_fabricated_winners_in_verdicts(self) -> None:
        packs = sorted(PACKS_DIR.glob("*.json"))
        for p in packs:
            with self.subTest(pack=p.name):
                data = json.loads(p.read_text(encoding="utf-8"))
                exp = data.get("experiment", {})
                verdict = exp.get("verdict", {})

                # Check verdict structure in envelope
                self.assertIn(verdict.get("status"), ("PENDING_5GPU", "NOT_RUN", "PENDING"))
                self.assertEqual(verdict.get("winner"), "NO_WINNER_CLAIMED")
                self.assertFalse(verdict.get("fabricated"))
                self.assertFalse(verdict.get("pass"))

                # Check verdict.json in §11.5 layout directory
                layout_dir = ROOT / exp.get("evidence_layout_dir")
                verdict_file = layout_dir / "verdict.json"
                self.assertTrue(
                    verdict_file.exists(),
                    f"Expected layout verdict file {verdict_file} to exist",
                )
                v_data = json.loads(verdict_file.read_text(encoding="utf-8"))
                self.assertIn(v_data.get("verdict"), ("PENDING_5GPU", "N/A", "PENDING"))
                self.assertEqual(v_data.get("winner"), "NO_WINNER_CLAIMED")
                self.assertFalse(v_data.get("pass"))
                self.assertFalse(v_data.get("fabricated"))

    def test_section_11_5_layout_placeholders(self) -> None:
        packs = sorted(PACKS_DIR.glob("*.json"))
        for p in packs:
            with self.subTest(pack=p.name):
                data = json.loads(p.read_text(encoding="utf-8"))
                exp = data.get("experiment", {})
                layout_dir = ROOT / exp.get("evidence_layout_dir")
                self.assertTrue(layout_dir.is_dir(), f"{layout_dir} is not a directory")

                for req_file in eq.EVIDENCE_LAYOUT_FILES:
                    target_file = layout_dir / req_file
                    self.assertTrue(
                        target_file.exists(),
                        f"Missing §11.5 evidence layout file {req_file} in {layout_dir}",
                    )

                # Ensure completion.json does not claim completed
                comp_file = layout_dir / "completion.json"
                c_data = json.loads(comp_file.read_text(encoding="utf-8"))
                self.assertNotEqual(c_data.get("status"), "COMPLETED")
                self.assertFalse(c_data.get("fabricated"))

    def test_task_board_r14_evidence_paths(self) -> None:
        tb_path = ROOT / "PLAN" / "task-board.json"
        tb_data = json.loads(tb_path.read_text(encoding="utf-8"))
        tasks = {t["id"]: t for t in tb_data.get("tasks", [])}
        self.assertIn("R14", tasks)
        r14 = tasks["R14"]
        self.assertEqual(r14.get("default_enable_status"), "NOT_AUTHORIZED_BY_THIS_PLAN")
        evidence = r14.get("evidence_paths", [])
        self.assertIn("PLAN/experiment-queue/pending-packs/r14-tm5-c06-span-expand-abba.json", evidence)
        self.assertIn("PLAN/experiment-queue/pending-packs/r14-tm5-c08-q-prep-abba.json", evidence)
        self.assertIn("scripts/test-pending-abba-packs-mock.py", evidence)


if __name__ == "__main__":
    unittest.main()
