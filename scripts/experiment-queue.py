#!/usr/bin/env python3
"""Offline experiment queue and executor scheduler (C03).

Validates PLAN/run-envelope.example.json schema fields, dry-runs the
run-id evidence directory layout from 攻坚总计划 §11.5, and serializes
experiment packs under an exclusive lock file. Never fabricates completion
records or GPU results. Does not require 5 GPUs.

Usage overview is in --help (and subcommand --help).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVELOPE = REPO_ROOT / "PLAN" / "run-envelope.example.json"
DEFAULT_QUEUE_DIR = REPO_ROOT / "PLAN" / "experiment-queue"
DEFAULT_LOCK_PATH = DEFAULT_QUEUE_DIR / ".exclusive.lock"

# Top-level and nested fields required by PLAN/run-envelope.example.json.
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "document_kind",
    "approval_status",
    "approved_by",
    "approved_at",
    "expires_at",
    "warning",
    "target",
    "models",
    "build",
    "permitted_tiers",
    "scope",
    "executor",
    "logs",
    "completion_contract",
    "stop_conditions",
    "forbidden_recovery",
    "on_block",
)

REQUIRED_TARGET = (
    "workspace",
    "audited_head",
    "host_identity",
    "device_bdfs",
    "device_uuids",
)

REQUIRED_BUILD = (
    "binary_path",
    "binary_sha256",
    "compiler",
    "flags",
    "driver_identity",
)

REQUIRED_SCOPE = (
    "sync_modes",
    "wire_formats",
    "fixed_spin_max",
    "max_context_tokens",
    "max_active_rows",
    "max_capacity_rows",
    "max_readers",
    "max_logical_workers",
    "physical_pens",
    "max_requests",
    "max_dispatches",
    "memory_budget_bytes_per_rank",
    "headroom_bytes_per_rank",
)

REQUIRED_EXECUTOR = (
    "owner",
    "exclusive_lock_id",
    "bind_address",
    "private_port",
)

REQUIRED_LOGS = (
    "before_cursor",
    "persistent_destination",
)

REQUIRED_COMPLETION_CONTRACT = (
    "implementation_reference",
    "native_wait_deadline",
    "wait_only_actually_submitted_values",
    "timeout_is_cancellation",
    "retain_resources_if_completion_unknown",
)

FORBIDDEN_RECOVERY_MUST_INCLUDE = (
    "fabricate_completion_signal",
    "release_inflight_resources_without_completion_proof",
)

# 攻坚总计划 §11.5 — every experiment evidence pack layout.
EVIDENCE_LAYOUT_FILES = (
    "manifest.json",
    "command.txt",
    "input.json",
    "sample-0001.json",
    "raw-metrics.txt",
    "profiler.json",
    "health-before.json",
    "health-after.json",
    "kernel-log-delta.txt",
    "completion.json",
    "verdict.json",
)

INERT_DOCUMENT_KINDS = frozenset({"INERT_EXAMPLE_NOT_A_RUNNER_CONFIG"})
APPROVED_STATUSES = frozenset({"APPROVED"})


class EnvelopeError(ValueError):
    """Envelope failed schema or policy validation."""


def _missing_keys(obj: dict[str, Any], required: tuple[str, ...], prefix: str) -> list[str]:
    missing: list[str] = []
    for key in required:
        if key not in obj:
            missing.append(f"{prefix}{key}" if prefix else key)
    return missing


def validate_envelope(envelope: dict[str, Any], *, require_approved: bool = False) -> list[str]:
    """Validate schema fields from PLAN/run-envelope.example.json.

    Returns a list of human-readable problems (empty means structurally OK).
    Never promotes an inert/unapproved example into an executable approval.
    """
    problems: list[str] = []
    if not isinstance(envelope, dict):
        return ["envelope must be a JSON object"]

    problems.extend(_missing_keys(envelope, REQUIRED_TOP_LEVEL, ""))

    target = envelope.get("target")
    if isinstance(target, dict):
        problems.extend(_missing_keys(target, REQUIRED_TARGET, "target."))
    elif "target" in envelope:
        problems.append("target must be an object")

    build = envelope.get("build")
    if isinstance(build, dict):
        problems.extend(_missing_keys(build, REQUIRED_BUILD, "build."))
    elif "build" in envelope:
        problems.append("build must be an object")

    scope = envelope.get("scope")
    if isinstance(scope, dict):
        problems.extend(_missing_keys(scope, REQUIRED_SCOPE, "scope."))
    elif "scope" in envelope:
        problems.append("scope must be an object")

    executor = envelope.get("executor")
    if isinstance(executor, dict):
        problems.extend(_missing_keys(executor, REQUIRED_EXECUTOR, "executor."))
    elif "executor" in envelope:
        problems.append("executor must be an object")

    logs = envelope.get("logs")
    if isinstance(logs, dict):
        problems.extend(_missing_keys(logs, REQUIRED_LOGS, "logs."))
    elif "logs" in envelope:
        problems.append("logs must be an object")

    contract = envelope.get("completion_contract")
    if isinstance(contract, dict):
        problems.extend(
            _missing_keys(contract, REQUIRED_COMPLETION_CONTRACT, "completion_contract.")
        )
        if contract.get("wait_only_actually_submitted_values") is not True:
            problems.append(
                "completion_contract.wait_only_actually_submitted_values must be true"
            )
        if contract.get("retain_resources_if_completion_unknown") is not True:
            problems.append(
                "completion_contract.retain_resources_if_completion_unknown must be true"
            )
        if contract.get("timeout_is_cancellation") is True:
            problems.append(
                "completion_contract.timeout_is_cancellation must remain false "
                "(timeout is not cancellation of GPU work)"
            )
    elif "completion_contract" in envelope:
        problems.append("completion_contract must be an object")

    forbidden = envelope.get("forbidden_recovery")
    if isinstance(forbidden, list):
        for item in FORBIDDEN_RECOVERY_MUST_INCLUDE:
            if item not in forbidden:
                problems.append(f"forbidden_recovery missing required entry: {item}")
    elif "forbidden_recovery" in envelope:
        problems.append("forbidden_recovery must be an array")

    if envelope.get("schema_version") != 1:
        problems.append("schema_version must be 1")

    kind = envelope.get("document_kind")
    status = envelope.get("approval_status")
    if kind in INERT_DOCUMENT_KINDS:
        problems.append(
            "document_kind is INERT_EXAMPLE_NOT_A_RUNNER_CONFIG: "
            "valid as a schema example only; not executable"
        )
    if status not in APPROVED_STATUSES:
        problems.append(
            f"approval_status={status!r} is not APPROVED; refuse GPU/mesh/model launch"
        )

    if require_approved and problems:
        # Caller asked for runnable approval; keep all structural/policy problems.
        pass

    return problems


def load_envelope(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise EnvelopeError(f"{path}: expected JSON object")
    return data


def make_run_id(prefix: str = "dry") -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def evidence_layout_paths(run_dir: Path) -> list[Path]:
    return [run_dir / name for name in EVIDENCE_LAYOUT_FILES]


def dry_run_evidence_layout(
    out_root: Path,
    run_id: str | None = None,
    *,
    write_placeholders: bool = True,
) -> dict[str, Any]:
    """Create (or describe) the §11.5 evidence directory for a run-id.

    Dry-run never marks completion as success and never invents metrics.
    Placeholders explicitly state NOT_RUN / PENDING so nothing is fabricated.
    """
    rid = run_id or make_run_id("dry")
    run_dir = out_root / rid
    files = evidence_layout_paths(run_dir)
    created: list[str] = []

    if write_placeholders:
        run_dir.mkdir(parents=True, exist_ok=True)
        placeholders: dict[str, Any] = {
            "manifest.json": {
                "run_id": rid,
                "status": "DRY_RUN_LAYOUT_ONLY",
                "note": "Placeholder from experiment-queue dry-run; not a measured run.",
                "sha": None,
                "dirty": None,
                "build": None,
                "model": None,
                "devices": [],
                "budget": None,
            },
            "command.txt": (
                f"# dry-run only; no command executed\n"
                f"# run_id={rid}\n"
                f"# secrets redacted by policy\n"
            ),
            "input.json": {
                "status": "NOT_PROVIDED",
                "fixture_hash": None,
                "note": "Dry-run layout; attach real fixture before execution.",
            },
            "sample-0001.json": {
                "status": "NOT_RUN",
                "note": "No request executed; sample slot reserved.",
            },
            "raw-metrics.txt": "# NOT_RUN: no metrics collected\n",
            "profiler.json": {
                "status": "NOT_APPLICABLE_FOR_THROUGHPUT_DRY_RUN",
                "note": "Profiler reserved for profile experiments only.",
            },
            "health-before.json": {"status": "NOT_CAPTURED"},
            "health-after.json": {"status": "NOT_CAPTURED"},
            "kernel-log-delta.txt": (
                "# NOT_CAPTURED\n"
                "# cursor range: n/a\n"
                "# permission/missing: dry-run did not touch kernel logs\n"
            ),
            # completion.json: never fabricate submitted/completed equality.
            "completion.json": {
                "status": "INCOMPLETE",
                "fabricated": False,
                "ranks": [],
                "submitted_values": [],
                "completed_values": [],
                "drain_conclusion": "NOT_ATTEMPTED",
                "note": (
                    "Dry-run refuses to invent completion. "
                    "Wait only actually submitted values; "
                    "retain resources if completion unknown."
                ),
            },
            "verdict.json": {
                "verdict": "N/A",
                "reason": "Dry-run layout only; no experiment executed.",
                "pass": False,
                "fabricated": False,
            },
        }

        for name in EVIDENCE_LAYOUT_FILES:
            path = run_dir / name
            payload = placeholders[name]
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            created.append(str(path.relative_to(out_root)))

    return {
        "run_id": rid,
        "run_dir": str(run_dir),
        "layout": [str(p.name) for p in files],
        "created": created,
        "completion_fabricated": False,
        "source": "攻坚总计划 §11.5",
    }


@dataclass
class ExclusiveLock:
    """Process-exclusive lock for serializing experiment packs."""

    path: Path
    fd: int | None = field(default=None, repr=False)

    def acquire(self, *, blocking: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(self.fd, flags)
        except OSError as exc:
            os.close(self.fd)
            self.fd = None
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(
                    f"exclusive lock held by another process: {self.path}"
                ) from exc
            raise
        os.write(self.fd, f"pid={os.getpid()} ts={time.time()}\n".encode("utf-8"))
        os.fsync(self.fd)

    def release(self) -> None:
        if self.fd is None:
            return
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        self.fd = None

    def __enter__(self) -> ExclusiveLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def _pack_state_path(queue_dir: Path) -> Path:
    return queue_dir / "packs.json"


def _load_packs(queue_dir: Path) -> list[dict[str, Any]]:
    path = _pack_state_path(queue_dir)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of packs")
    return data


def _save_packs(queue_dir: Path, packs: list[dict[str, Any]]) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = _pack_state_path(queue_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(packs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def enqueue_pack(
    pack_path: Path,
    *,
    queue_dir: Path = DEFAULT_QUEUE_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    envelope_path: Path | None = None,
) -> dict[str, Any]:
    """Serialize an experiment pack onto the queue under an exclusive lock.

    The pack is recorded as QUEUED / NOT_RUN. Completion is never invented.
    """
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict):
        raise ValueError("pack must be a JSON object")

    if envelope_path is not None:
        envelope = load_envelope(envelope_path)
        problems = validate_envelope(envelope)
        # Inert example is expected for schema checks; still refuse marking complete.
        structural = [
            p
            for p in problems
            if not p.startswith("document_kind")
            and not p.startswith("approval_status")
        ]
        if structural:
            raise EnvelopeError("; ".join(structural))

    with ExclusiveLock(lock_path):
        packs = _load_packs(queue_dir)
        entry = {
            "pack_id": pack.get("pack_id") or make_run_id("pack"),
            "source_path": str(pack_path.resolve()),
            "enqueued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": "QUEUED",
            "completion": {
                "status": "NOT_RUN",
                "fabricated": False,
                "submitted_values": [],
                "completed_values": [],
                "drain_conclusion": "NOT_ATTEMPTED",
            },
            "verdict": {
                "verdict": "N/A",
                "reason": "Pack queued only; offline queue never fabricates completion.",
                "pass": False,
                "fabricated": False,
            },
            "pack": pack,
        }
        packs.append(entry)
        _save_packs(queue_dir, packs)
        return entry


def list_packs(queue_dir: Path = DEFAULT_QUEUE_DIR) -> list[dict[str, Any]]:
    return _load_packs(queue_dir)


# ---------------------------------------------------------------------------
# Pure-Python mirror of ggml_backend_vk_tp5_comm_free_safe for offline mocks.
# Semantics (ggml-vulkan-collective.cpp):
#   - drain failure => return false and RETAIN resources (do not delete)
#   - never fabricate completion of submitted work
#   - wait only actually submitted values
# ---------------------------------------------------------------------------


@dataclass
class MockTp5Rank:
    rank: int
    resources_alive: bool = True


@dataclass
class MockTp5Comm:
    """Minimal communicator state for safe-teardown mock tests."""

    ranks: list[MockTp5Rank] = field(default_factory=list)
    failed: bool = False
    allreduce_calls: int = 0
    submitted: list[int] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    drain_should_fail: bool = False
    deleted: bool = False
    retained: bool = False

    @property
    def resources_retained(self) -> bool:
        return (not self.deleted) and any(r.resources_alive for r in self.ranks)


def tp5_drain_submitted(comm: MockTp5Comm) -> bool:
    """Drain only values that were actually submitted; never invent completions."""
    if comm.drain_should_fail:
        return False
    # Completion may only cover the submitted prefix — never pad completed.
    if len(comm.completed) > len(comm.submitted):
        # Fabricated completion detected; treat as drain failure.
        return False
    if any(c not in comm.submitted for c in comm.completed):
        return False
    # Mark submitted work that is already completed as drained; unknowns stay.
    unknown = [s for s in comm.submitted if s not in comm.completed]
    if unknown:
        # Unknown completion => retain (caller decides); drain not fully OK.
        return False
    return True


def ggml_backend_vk_tp5_comm_free_safe_mock(comm: MockTp5Comm | None) -> bool:
    """Mirror ggml_backend_vk_tp5_comm_free_safe.

    On drain failure: retain resources, return False, do not fabricate completion.
    On success: release rank resources and delete the communicator.
    """
    if comm is None:
        return True
    if comm.deleted:
        return True

    if not tp5_drain_submitted(comm):
        comm.retained = True
        # Do not free ranks; do not invent completed_values.
        return False

    for rank in comm.ranks:
        rank.resources_alive = False
    comm.deleted = True
    comm.retained = False
    return True


def cmd_validate_envelope(args: argparse.Namespace) -> int:
    path = Path(args.envelope)
    envelope = load_envelope(path)
    problems = validate_envelope(envelope)
    # For the inert example, structural OK + policy warnings is the expected outcome.
    structural = [
        p
        for p in problems
        if not p.startswith("document_kind") and not p.startswith("approval_status")
    ]
    report = {
        "path": str(path),
        "structural_ok": not structural,
        "executable": False,
        "problems": problems,
        "note": (
            "Inert/unapproved envelopes validate as schema examples only. "
            "This tool never auto-promotes them to APPROVED."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_approved:
        return 0 if not problems else 2
    return 0 if not structural else 2


def cmd_dry_run_layout(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir)
    result = dry_run_evidence_layout(
        out_root,
        run_id=args.run_id,
        write_placeholders=not args.describe_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    entry = enqueue_pack(
        Path(args.pack),
        queue_dir=Path(args.queue_dir),
        lock_path=Path(args.lock_file),
        envelope_path=Path(args.envelope) if args.envelope else None,
    )
    # Strip bulky pack body for stdout summary.
    summary = {k: v for k, v in entry.items() if k != "pack"}
    summary["pack_keys"] = sorted(entry["pack"].keys())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    packs = list_packs(Path(args.queue_dir))
    slim = []
    for p in packs:
        slim.append(
            {
                "pack_id": p.get("pack_id"),
                "state": p.get("state"),
                "completion": p.get("completion"),
                "verdict": p.get("verdict"),
                "source_path": p.get("source_path"),
                "enqueued_at": p.get("enqueued_at"),
            }
        )
    print(json.dumps({"count": len(slim), "packs": slim}, indent=2, ensure_ascii=False))
    return 0


def cmd_mock_teardown(args: argparse.Namespace) -> int:
    """Demonstrate / exercise the free_safe mock (no GPU)."""
    submitted = [int(x) for x in args.submitted.split(",") if x.strip() != ""]
    completed = [int(x) for x in args.completed.split(",") if x.strip() != ""]
    comm = MockTp5Comm(
        ranks=[MockTp5Rank(i) for i in range(args.ranks)],
        submitted=submitted,
        completed=completed,
        drain_should_fail=args.force_drain_fail,
    )
    ok = ggml_backend_vk_tp5_comm_free_safe_mock(comm)
    report = {
        "drain_ok": ok,
        "resources_retained": comm.resources_retained,
        "deleted": comm.deleted,
        "retained_flag": comm.retained,
        "submitted": list(comm.submitted),
        "completed": list(comm.completed),
        "fabricated_completion": len(comm.completed) > len(comm.submitted)
        or any(c not in comm.submitted for c in comm.completed),
        "semantics": "mirror ggml_backend_vk_tp5_comm_free_safe",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # Exit 0 always for demo; tests assert semantics separately.
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment-queue.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Offline experiment queue / executor scheduler (C03).\n"
            "Validates PLAN/run-envelope.example.json schema fields, dry-runs the\n"
            "§11.5 run-id evidence directory layout, and serializes packs under an\n"
            "exclusive lock. Never fabricates completion. No 5-GPU required."
        ),
        epilog=(
            "examples:\n"
            "  %(prog)s validate-envelope\n"
            "  %(prog)s dry-run-layout --out-dir /tmp/evidence-dry\n"
            "  %(prog)s enqueue --pack my-pack.json\n"
            "  %(prog)s list\n"
            "  %(prog)s mock-teardown --submitted 1,2 --completed 1 --force-drain-fail\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser(
        "validate-envelope",
        help="Validate schema fields of PLAN/run-envelope.example.json (or --envelope).",
    )
    p_val.add_argument(
        "--envelope",
        default=str(DEFAULT_ENVELOPE),
        help=f"Envelope JSON path (default: {DEFAULT_ENVELOPE})",
    )
    p_val.add_argument(
        "--require-approved",
        action="store_true",
        help="Fail unless approval_status is APPROVED (default: allow inert example).",
    )
    p_val.set_defaults(func=cmd_validate_envelope)

    p_dry = sub.add_parser(
        "dry-run-layout",
        help="Dry-run generate §11.5 run-id evidence dir layout (no GPU, no fabricated completion).",
    )
    p_dry.add_argument("--run-id", default=None, help="Optional run-id (default: dry-<utc>-<hex>)")
    p_dry.add_argument(
        "--out-dir",
        default=str(DEFAULT_QUEUE_DIR / "evidence-dry"),
        help="Root directory under which run-id/ is created",
    )
    p_dry.add_argument(
        "--describe-only",
        action="store_true",
        help="Print layout plan without writing files",
    )
    p_dry.set_defaults(func=cmd_dry_run_layout)

    p_enq = sub.add_parser(
        "enqueue",
        help="Append an experiment pack under an exclusive lock (state=QUEUED/NOT_RUN).",
    )
    p_enq.add_argument("--pack", required=True, help="Path to pack JSON")
    p_enq.add_argument(
        "--queue-dir",
        default=str(DEFAULT_QUEUE_DIR),
        help=f"Queue directory (default: {DEFAULT_QUEUE_DIR})",
    )
    p_enq.add_argument(
        "--lock-file",
        default=str(DEFAULT_LOCK_PATH),
        help=f"Exclusive lock file (default: {DEFAULT_LOCK_PATH})",
    )
    p_enq.add_argument(
        "--envelope",
        default=None,
        help="Optional envelope to structurally validate before enqueue",
    )
    p_enq.set_defaults(func=cmd_enqueue)

    p_list = sub.add_parser("list", help="List queued packs (completion never fabricated).")
    p_list.add_argument(
        "--queue-dir",
        default=str(DEFAULT_QUEUE_DIR),
        help=f"Queue directory (default: {DEFAULT_QUEUE_DIR})",
    )
    p_list.set_defaults(func=cmd_list)

    p_mock = sub.add_parser(
        "mock-teardown",
        help="Exercise ggml_backend_vk_tp5_comm_free_safe mock (drain-fail retains resources).",
    )
    p_mock.add_argument("--ranks", type=int, default=2, help="Mock rank count (default: 2)")
    p_mock.add_argument(
        "--submitted",
        default="1,2",
        help="Comma-separated actually submitted values (default: 1,2)",
    )
    p_mock.add_argument(
        "--completed",
        default="1",
        help="Comma-separated known completed values (default: 1; missing => retain)",
    )
    p_mock.add_argument(
        "--force-drain-fail",
        action="store_true",
        help="Force drain failure regardless of submitted/completed sets",
    )
    p_mock.set_defaults(func=cmd_mock_teardown)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (EnvelopeError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
