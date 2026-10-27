#!/usr/bin/env python3
"""Falsify the trace checker with owned protocol fixtures, without a model."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import runpy
import unittest


driver = runpy.run_path(str(Path(__file__).with_name("real-whiteboard.py")))
validate = driver["validate_trace"]
ContractError = driver["ContractError"]


def trace(schedule, workers, parallel, phase_memory):
    """Encode a declared schedule as events; these are not inference receipts."""
    events = [{"event": "config", "workers": workers, "parallel": parallel,
               "synthesis_stage": workers + 1, "provider": "native-a",
               "submission_domains": 1, "logical_context": 256,
               "physical_kv": 256, "base_tokens": 1,
               "logical_sequences": workers + 2, "phase_memory": phase_memory}]
    full = defaultdict(list)
    public = defaultdict(list)
    texts = defaultdict(str)
    bodies = defaultdict(int)
    previous_frame = False
    for frontier, members in enumerate(schedule, 1):
        has_frame = any(kind == "frame" for _, _, kind in members)
        if phase_memory and previous_frame and not has_frame:
            events.append({"event": "compute_release", "before": 4096, "after": 0})
        previous_frame = has_frame
        readers = []
        additions = {}
        for stage, pen, kind in members:
            count = 2 if kind == "frame" else 1
            first = len(full[stage])
            added = [(stage * 100 + first + offset, 0, first + offset + 1)
                     for offset in range(count)]
            additions[stage] = added
            groups = [(0, [(1, 0, 0)])]
            groups += [(peer, public[peer]) for peer in sorted(public)
                       if peer != stage and public[peer]]
            groups.append((stage, full[stage] + added))
            spans = []
            position = 0
            for owner, keys in groups:
                for run, offset, storage in keys:
                    spans.append({"stage": owner, "run": run, "offset": offset,
                                  "storage": storage, "virtual": position, "count": 1})
                    position += 1
            readers.append({"stage": stage, "pen": pen, "starting": kind == "frame",
                            "keys": position, "query_virtual": position - 1, "spans": spans})
        events.append({"event": "frontier", "frontier": frontier, "read_revision": frontier,
                       "revision": frontier + 1, "rows": sum(map(len, additions.values())),
                       "readers": readers})
        for stage, _, kind in members:
            full[stage].extend(additions[stage])
            if kind != "end":
                public[stage].extend(additions[stage])
            if kind == "frame":
                continue
            ending = kind == "end"
            events.append({"event": "token", "frontier": frontier, "revision": frontier + 1,
                           "stage": stage, "token": 2 if ending else 100 + stage,
                           "source_end": ending})
            events.append({"event": "text", "stage": stage, "offset": len(texts[stage]),
                           "text": "" if ending else "x", "finished": ending})
            if ending:
                events.append({"event": "sealed", "stage": stage})
            else:
                texts[stage] += "x"
                bodies[stage] += 1
    events.append({"event": "result", "complete": True, "revision": len(schedule) + 1,
                   "text": texts[workers + 1], "body_tokens": bodies[workers + 1]})
    return events


def coverage_trace(phase=True, first_sealed=1):
    schedule = [
        [(1, 0, "frame"), (2, 1, "frame")],
        [(1, 0, "body"), (2, 1, "body")],
        [(1, 0, "body"), (2, 1, "body")],
        [(1, 0, "end" if first_sealed in (0, 1) else "body"),
         (2, 1, "end" if first_sealed in (0, 2) else "body")],
    ]
    reused_pen = 1 if first_sealed == 2 else 0
    admission = [(3, reused_pen, "frame")]
    if first_sealed:
        admission.append((3 - first_sealed, 1 - reused_pen, "end"))
    schedule.append(admission)
    schedule.extend([[(3, reused_pen, "body")], [(3, reused_pen, "end")]])
    for stage in (4, 5):
        schedule.extend([[(stage, 0, kind)] for kind in ("frame", "body", "end")])
    return trace(schedule, 4, 2, phase)


class WhiteboardValidator(unittest.TestCase):
    edges = [(1, 4), (2, 4), (3, 4)]

    def check(self, events, phase="release"):
        return validate(events, 4, 2, self.edges, 2, phase,
                        require_completed_nonancestor=True)

    def reject(self, events, message, phase="release"):
        with self.assertRaisesRegex(ContractError, message):
            self.check(events, phase)

    def test_release_and_retain_with_either_or_both_writers_sealed(self):
        for first in (0, 1, 2):
            for phase in ("release", "retain"):
                with self.subTest(first=first, phase=phase):
                    result = self.check(coverage_trace(phase == "release", first), phase)
                    self.assertEqual(len(result["phase_memory"]["releases"]),
                                     4 if phase == "release" else 0)
                    witness = result["completed_nonancestor_visibility"]
                    self.assertEqual(witness["status"], "PASS")
                    self.assertTrue(any(item["reader"] == 3 and item["peer"] in (1, 2)
                                        and item["peer_sealed_at_frontier"] < item["frontier"]
                                        for item in witness["witnesses"]))

    def test_requested_phase_must_match_config(self):
        self.reject(coverage_trace(), "differs from the request", "retain")
        self.reject(coverage_trace(False), "differs from the request", "release")
        events = coverage_trace()
        events[0]["phase_memory"] = 1
        self.reject(events, "phase-memory mode is missing")

    def test_missing_duplicate_or_incomplete_release(self):
        original = coverage_trace()
        index = next(i for i, item in enumerate(original) if item["event"] == "compute_release")
        events = deepcopy(original)
        del events[index]
        self.reject(events, "compute release is missing")
        events = deepcopy(original)
        events.insert(index, deepcopy(events[index]))
        self.reject(events, "duplicate compute releases")
        events = deepcopy(original)
        events[index]["after"] = 1
        self.reject(events, "retained transient allocation")

    def test_release_cannot_move_to_frame_or_result(self):
        events = coverage_trace()
        events.insert(1, {"event": "compute_release", "before": 4096, "after": 0})
        self.reject(events, "outside a FRAME-to-BODY transition")
        events = coverage_trace()
        events.insert(-1, {"event": "compute_release", "before": 4096, "after": 0})
        self.reject(events, "no following frontier")

    def test_retain_cannot_emit_release(self):
        events = coverage_trace(False)
        events.insert(1, {"event": "compute_release", "before": 4096, "after": 0})
        self.reject(events, "retain mode emitted a compute release", "retain")

    def test_ancestors_including_transitive_and_synthesis_are_not_witnesses(self):
        schedule = [[(stage, 0, kind)] for stage in range(1, 5)
                    for kind in ("frame", "body", "end")]
        events = trace(schedule, 3, 1, True)
        edges = [(1, 2), (2, 3)]
        result = validate(events, 3, 1, edges, 0, "release")
        self.assertEqual(result["completed_nonancestor_visibility"]["status"], "NOT_QUALIFIED")
        with self.assertRaisesRegex(ContractError, "no reader observed BODY work from a completed non-ancestor"):
            validate(events, 3, 1, edges, 0, "release", require_completed_nonancestor=True)

    def test_a_missing_completed_peer_key_is_rejected(self):
        events = coverage_trace()
        reader = next(item for item in events if item["event"] == "frontier" and item["frontier"] == 5)["readers"][0]
        del reader["spans"][1]
        for position, span in enumerate(reader["spans"]):
            span["virtual"] = position
        reader["keys"] -= 1
        reader["query_virtual"] -= 1
        self.reject(events, "did not see exactly the prior committed public keys")

    def test_dependency_admission_still_rejects_an_unsealed_parent(self):
        with self.assertRaisesRegex(ContractError, "dependency started before source end"):
            validate(coverage_trace(), 4, 2, [(2, 3), *self.edges], 2, "release")


if __name__ == "__main__":
    unittest.main()
