#!/usr/bin/env python3
"""Drive the shipped baseline locator against the real evidence tree."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "baseline_locator.py"
REAL_ROOT = Path(__file__).resolve().parents[3] / "build-tp5-dual50-20261022" / "mtp-endtoend-20261024"
CANDIDATE = "mtp-hc-row-tail-candidate-20261025"
RELATIVE_FILES = (
    "acceptance-frontier-20261025.json",
    "acceptance-frontier-followup-20261025.json",
    "acceptance-frontier-postdown-20261025.json",
    "acceptance-frontier-expanded-20261025.json",
    "mtp-device-stage-diagnostic-20261025/gpu-wait-n3-vs-n1-analysis.json",
    f"{CANDIDATE}/build-receipt.json",
    f"{CANDIDATE}/numeric-comparison.json",
    f"{CANDIDATE}/on512x2/command.json",
    f"{CANDIDATE}/on512x2/execution.json",
    f"{CANDIDATE}/on512x2/loaded-maps.txt",
    f"{CANDIDATE}/on512x2/metrics.json",
    f"{CANDIDATE}/on512x2/input-identities.json",
    f"{CANDIDATE}/off512x2/command.json",
    f"{CANDIDATE}/off512x2/execution.json",
    f"{CANDIDATE}/off512x2/loaded-maps.txt",
    f"{CANDIDATE}/off512x2/metrics.json",
)


def run_locator(*extra):
    completed = subprocess.run(
        [sys.executable, str(TOOL), *map(str, extra)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed


class BaselineLocatorTests(unittest.TestCase):
    def copy_tree(self, root: Path) -> None:
        for relative in RELATIVE_FILES:
            source = REAL_ROOT / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (root / "mtp-ready-snapshot-consolidated-20261025").mkdir()

    def rewrite(self, path: Path, mutate) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_real_receipts_stay_consistent_and_do_not_land_a_kernel(self):
        completed = run_locator("--evidence-root", REAL_ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        result = json.loads(completed.stdout)
        self.assertEqual(result["decision"], "consistent")
        self.assertEqual(result["gates"]["G0"], "blocked")
        self.assertEqual(result["gates"]["G1"], "blocked")
        self.assertEqual(result["gates"]["G1_recorded_library_bytes"], "match")
        self.assertEqual(result["gates"]["G3"], "SKIP")
        self.assertEqual(result["gates"]["G4"], "SKIP")
        self.assertEqual(result["gates"]["G5"], "SKIP")
        self.assertFalse(result["gates"]["formal_dual_stream_pass"])
        self.assertEqual(result["not_run"], ["gpu", "model", "formal_512x2", "native_drain"])
        self.assertEqual(len(result["libraries"]), 5)
        self.assertEqual(len({row["sha256"] for row in result["libraries"]}), 5)
        self.assertTrue(all(CANDIDATE in row["path"] for row in result["libraries"]))
        ceiling = result["cost_ceiling"]
        self.assertEqual(ceiling["answer"], "no")
        self.assertFalse(ceiling["kernel_or_scheduling_change_landed"])
        self.assertIn("not landed", ceiling["reason"])
        self.assertGreater(ceiling["observed_span_difference_ms"], 0)
        self.assertFalse(result["a_source_edited"])
        self.assertIn(CANDIDATE, result["locator"]["candidate"])
        self.assertNotIn("mtp-moe-down-n3-fusion", result["locator"]["candidate"])

    def test_swapped_library_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "mtp-endtoend-20261024"
            self.copy_tree(root)
            execution = root / CANDIDATE / "on512x2" / "execution.json"
            def swap(value):
                loaded = value["loaded_libraries"]
                old = next(iter(loaded))
                loaded[old.replace(CANDIDATE, "mtp-later-candidate-20261026")] = loaded.pop(old)
            self.rewrite(execution, swap)
            completed = run_locator("--evidence-root", root)
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(result["code"], "WRONG_LIBRARY_INSTALL")
        self.assertFalse(result["gates"]["formal_dual_stream_pass"])

    def test_collapsed_prompts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "mtp-endtoend-20261024"
            self.copy_tree(root)
            command = root / CANDIDATE / "on512x2" / "command.json"
            def collapse(value):
                argv = value["argv"]
                first = argv.index("--prompt")
                second = argv.index("--prompt", first + 1)
                argv[second + 1] = argv[first + 1]
            self.rewrite(command, collapse)
            completed = run_locator("--evidence-root", root)
        result = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(result["code"], "COLLAPSED_REQUEST_IDENTITY")

    def test_missing_drain_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "mtp-endtoend-20261024"
            self.copy_tree(root)
            comparison = root / CANDIDATE / "numeric-comparison.json"
            def drop_drain(value):
                value["work"][0]["native_drained"] = False
            self.rewrite(comparison, drop_drain)
            completed = run_locator("--evidence-root", root)
        result = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(result["code"], "MISSING_DRAIN")
        self.assertNotIn("tok/s", result["detail"])

    def test_missing_root_names_the_path(self):
        missing = Path(tempfile.gettempdir()) / "mtp-baseline-locator-absent" / "mtp-endtoend-20261024"
        completed = run_locator("--evidence-root", missing)
        result = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 3)
        self.assertEqual(result["decision"], "blocked")
        self.assertEqual(result["code"], "MISSING_ROOT")
        self.assertIn(str(missing), result["missing_paths"])

    def test_small_claim_does_not_clear_the_ceiling(self):
        with tempfile.TemporaryDirectory() as folder:
            claim = Path(folder) / "claim.json"
            claim.write_text(json.dumps({
                "removal_ms": 1.5,
                "full_n3_target_path": True,
                "equivalent_overlap_improves_slower_stream": False,
            }), encoding="utf-8")
            completed = run_locator("--evidence-root", REAL_ROOT, "--removal-claim", claim)
        result = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(result["cost_ceiling"]["answer"], "no")
        self.assertFalse(result["cost_ceiling"]["kernel_or_scheduling_change_landed"])

    def test_large_claim_answers_yes_without_landing_code(self):
        with tempfile.TemporaryDirectory() as folder:
            claim = Path(folder) / "claim.json"
            claim.write_text(json.dumps({
                "removal_ms": 9.0,
                "full_n3_target_path": True,
                "equivalent_overlap_improves_slower_stream": False,
            }), encoding="utf-8")
            completed = run_locator("--evidence-root", REAL_ROOT, "--removal-claim", claim)
        result = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        ceiling = result["cost_ceiling"]
        self.assertEqual(ceiling["answer"], "yes")
        self.assertTrue(ceiling["can_remove_about_8_to_12_ms_from_full_n3_target_path"])
        self.assertFalse(ceiling["kernel_or_scheduling_change_landed"])
        self.assertIn("not landed", ceiling["reason"])


if __name__ == "__main__":
    unittest.main()
