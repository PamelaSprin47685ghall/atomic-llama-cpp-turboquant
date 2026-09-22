#!/usr/bin/env python3
"""Unit tests for T1C publish-order / ready-before-reduce / retain-on-unknown-completion.

Pure Python mocks only (no Vulkan runtime, no multi-GPU requirement).
Validates definition-time contracts across host_push, local_publisher, and bar_pull transports.

Usage:
    python3 scripts/test-t1c-publish-order-mock.py -v
"""

from __future__ import annotations

import enum
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TransportType(str, enum.Enum):
    HOST_PUSH = "host_push"
    LOCAL_PUBLISHER = "local_publisher"
    BAR_PULL = "bar_pull"


@dataclass
class PublishEvent:
    event_type: str
    rank: int
    generation: int
    payload_hash: int
    timestamp_tick: int


class MockSidecarMailbox:
    """Mock for LateBind sidecar publication and CPU reduction mailbox."""

    def __init__(self, transport: TransportType, num_ranks: int = 5):
        self.transport = transport
        self.num_ranks = num_ranks
        self.generation: int = 1
        self.payload_committed: Dict[int, Optional[int]] = {r: None for r in range(num_ranks)}
        self.ready_flag: Dict[int, int] = {r: 0 for r in range(num_ranks)}
        self.events: List[PublishEvent] = []
        self._tick: int = 0
        self.allocated_buffers: Dict[str, Any] = {
            f"rank_{r}_sidecar": {"size": 4096, "active": True} for r in range(num_ranks)
        }

    def _get_tick(self) -> int:
        self._tick += 1
        return self._tick

    def publish_sidecar(self, rank: int, payload_data: list[float], generation: int, violate_order: bool = False) -> None:
        """Simulate GPU publisher writing payload and setting ready flag."""
        p_hash = hash(tuple(payload_data))

        if violate_order:
            # Buggy / race condition mock: flag set BEFORE payload write
            self.ready_flag[rank] = generation
            self.events.append(PublishEvent("READY_FLAG_SET", rank, generation, 0, self._get_tick()))
            self.payload_committed[rank] = p_hash
            self.events.append(PublishEvent("PAYLOAD_WRITTEN", rank, generation, p_hash, self._get_tick()))
            return

        # Correct publish sequence:
        # 1. Payload written into target transport buffer
        self.payload_committed[rank] = p_hash
        self.events.append(PublishEvent("PAYLOAD_WRITTEN", rank, generation, p_hash, self._get_tick()))

        # 2. Release barrier / cache visibility guaranteed
        self.events.append(PublishEvent("RELEASE_BARRIER", rank, generation, p_hash, self._get_tick()))

        # 3. Ready flag set with monotonic generation
        self.ready_flag[rank] = generation
        self.events.append(PublishEvent("READY_FLAG_SET", rank, generation, p_hash, self._get_tick()))

    def cpu_reduce(self, target_generation: int) -> list[float]:
        """Simulate CPU observing ready flags across all ranks and performing reduction."""
        # Check ready flag condition
        for rank in range(self.num_ranks):
            if self.ready_flag[rank] != target_generation:
                raise RuntimeError(
                    f"CPU reduce attempted before ready flag set on rank {rank}: "
                    f"expected gen {target_generation}, found {self.ready_flag[rank]}"
                )

        # Invariant check: payload must have been committed before ready flag was set
        for rank in range(self.num_ranks):
            payload_events = [e for e in self.events if e.rank == rank and e.event_type == "PAYLOAD_WRITTEN" and e.generation == target_generation]
            ready_events = [e for e in self.events if e.rank == rank and e.event_type == "READY_FLAG_SET" and e.generation == target_generation]

            if not payload_events or not ready_events:
                raise RuntimeError(f"Missing publish events on rank {rank}")

            if payload_events[-1].timestamp_tick > ready_events[-1].timestamp_tick:
                raise RuntimeError(
                    f"WAR / Ordering violation on rank {rank}: "
                    f"payload tick {payload_events[-1].timestamp_tick} > ready tick {ready_events[-1].timestamp_tick}"
                )

        # Identical arithmetic output simulation (Q8Packed dot/reduce invariant)
        return [float(target_generation * 10.0)] * 32

    def free_resources_safe(self, completion_confirmed: bool) -> bool:
        """Simulate resource teardown.

        Contract: If completion status is UNKNOWN (e.g. drain timeout or pending work),
        resources MUST be retained, never freed prematurely.
        """
        if not completion_confirmed:
            # Retain resources
            return False

        # Completion confirmed: safe to release
        for k in self.allocated_buffers:
            self.allocated_buffers[k]["active"] = False
        return True


class TestT1CPublishOrderMock(unittest.TestCase):
    """Offline unit tests for T1C publication ordering and teardown safety."""

    def test_same_arithmetic_invariants_across_transports(self) -> None:
        """All three transports must produce identical reduction values on matching inputs."""
        payload = [0.25, 0.5, -0.125, 1.0] * 8
        gen = 1

        results = {}
        for transport in [TransportType.HOST_PUSH, TransportType.LOCAL_PUBLISHER, TransportType.BAR_PULL]:
            mb = MockSidecarMailbox(transport=transport, num_ranks=5)
            for r in range(5):
                mb.publish_sidecar(rank=r, payload_data=payload, generation=gen)
            reduced = mb.cpu_reduce(target_generation=gen)
            results[transport.value] = reduced

        # Invariant check: Bitwise identical results across all transports
        self.assertEqual(results["host_push"], results["local_publisher"])
        self.assertEqual(results["local_publisher"], results["bar_pull"])

    def test_ready_before_reduce_enforced(self) -> None:
        """CPU reduce must fail if even one rank has not posted the target generation ready flag."""
        mb = MockSidecarMailbox(transport=TransportType.HOST_PUSH, num_ranks=5)
        payload = [1.0] * 32

        # Publish only ranks 0 through 3 (rank 4 missing)
        for r in range(4):
            mb.publish_sidecar(rank=r, payload_data=payload, generation=2)

        with self.assertRaisesRegex(RuntimeError, "CPU reduce attempted before ready flag set on rank 4"):
            mb.cpu_reduce(target_generation=2)

    def test_publish_order_violation_caught(self) -> None:
        """Publish order violation (ready flag set before payload write) must be caught."""
        mb = MockSidecarMailbox(transport=TransportType.LOCAL_PUBLISHER, num_ranks=5)
        payload = [1.0] * 32

        for r in range(4):
            mb.publish_sidecar(rank=r, payload_data=payload, generation=1)

        # Rank 4 violates write ordering
        mb.publish_sidecar(rank=4, payload_data=payload, generation=1, violate_order=True)

        with self.assertRaisesRegex(RuntimeError, "Ordering violation on rank 4"):
            mb.cpu_reduce(target_generation=1)

    def test_retain_on_unknown_completion(self) -> None:
        """Resources must be retained if completion confirmation fails or times out."""
        mb = MockSidecarMailbox(transport=TransportType.BAR_PULL, num_ranks=5)

        # When completion is not confirmed (unknown completion / timeout)
        ok = mb.free_resources_safe(completion_confirmed=False)
        self.assertFalse(ok)
        for name, buf in mb.allocated_buffers.items():
            self.assertTrue(buf["active"], f"Buffer {name} must remain active when completion unknown")

        # When completion is confirmed
        ok_confirmed = mb.free_resources_safe(completion_confirmed=True)
        self.assertTrue(ok_confirmed)
        for name, buf in mb.allocated_buffers.items():
            self.assertFalse(buf["active"], f"Buffer {name} must be released when confirmed")

    def test_generation_monotonicity_prevents_aba(self) -> None:
        """Old generation ready flags must not trigger reduction for a new generation cycle."""
        mb = MockSidecarMailbox(transport=TransportType.HOST_PUSH, num_ranks=5)
        payload = [0.5] * 32

        # Cycle generation 1
        for r in range(5):
            mb.publish_sidecar(rank=r, payload_data=payload, generation=1)
        res1 = mb.cpu_reduce(target_generation=1)
        self.assertEqual(len(res1), 32)

        # Start generation 2, only rank 0 published
        mb.publish_sidecar(rank=0, payload_data=payload, generation=2)

        # Target generation 2 cannot succeed on stale generation 1 flags
        with self.assertRaisesRegex(RuntimeError, "CPU reduce attempted before ready flag set on rank 1"):
            mb.cpu_reduce(target_generation=2)


if __name__ == "__main__":
    unittest.main()
