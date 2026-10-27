#!/usr/bin/env python3
"""Locate the 2026-10-25 MTP baseline receipts and decide the next cost ceiling.

This tool reads existing evidence. It does not load a model, start a GPU
run, relink libraries, or edit a kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


EVIDENCE_DIR_NAME = "mtp-endtoend-20261024"
CANDIDATE_DIR_NAME = "mtp-hc-row-tail-candidate-20261025"
DEVICE_STAGE_RELATIVE = (
    "mtp-device-stage-diagnostic-20261025/gpu-wait-n3-vs-n1-analysis.json"
)
SNAPSHOT_DIR_NAME = "mtp-ready-snapshot-consolidated-20261025"
FRONTIER_NAMES = (
    "acceptance-frontier-20261025.json",
    "acceptance-frontier-followup-20261025.json",
    "acceptance-frontier-postdown-20261025.json",
    "acceptance-frontier-expanded-20261025.json",
)
REMOVAL_FLOOR_MS = 8.0


class LocateError(Exception):
    def __init__(self, decision: str, code: str, detail: str, missing: list[str] | None = None):
        super().__init__(detail)
        self.decision = decision
        self.code = code
        self.detail = detail
        self.missing = list(missing or [])


def repo_evidence_root(start: Path | None = None) -> Path:
    origin = (start or Path(__file__)).resolve()
    for parent in [origin, *origin.parents]:
        candidate = parent / "build-tp5-dual50-20261022" / EVIDENCE_DIR_NAME
        if candidate.exists():
            return candidate
    raise LocateError(
        "blocked",
        "MISSING_ROOT",
        "named evidence root is not visible from the tool path",
        [f"build-tp5-dual50-20261022/{EVIDENCE_DIR_NAME}"],
    )


def load_json(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LocateError("blocked", "UNREADABLE", str(error), [str(path)]) from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise LocateError("rejected", "BAD_JSON", f"{path}: {error}", [str(path)]) from error


def flag_values(argv: list, flag: str) -> list[str]:
    values = []
    index = 0
    while index < len(argv):
        if argv[index] == flag:
            if index + 1 >= len(argv) or not isinstance(argv[index + 1], str):
                raise LocateError("rejected", "BAD_ARGV", f"{flag} has no value", [])
            values.append(argv[index + 1])
            index += 2
            continue
        index += 1
    return values


def require_mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise LocateError("rejected", "BAD_SHAPE", f"{label} is not an object", [])
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def required_paths(root: Path) -> list[Path]:
    candidate = root / CANDIDATE_DIR_NAME
    paths = [root / name for name in FRONTIER_NAMES]
    paths.append(root / DEVICE_STAGE_RELATIVE)
    paths.extend(
        [
            candidate / "build-receipt.json",
            candidate / "numeric-comparison.json",
            candidate / "on512x2" / "command.json",
            candidate / "on512x2" / "execution.json",
            candidate / "on512x2" / "loaded-maps.txt",
            candidate / "on512x2" / "metrics.json",
            candidate / "on512x2" / "input-identities.json",
            candidate / "off512x2" / "command.json",
            candidate / "off512x2" / "execution.json",
            candidate / "off512x2" / "loaded-maps.txt",
            candidate / "off512x2" / "metrics.json",
        ]
    )
    return paths


def missing_required(root: Path) -> list[str]:
    missing = [str(path) for path in required_paths(root) if not path.is_file()]
    snapshot = root / SNAPSHOT_DIR_NAME
    if not snapshot.is_dir():
        missing.append(str(snapshot))
    return missing


def prompts_of(command: dict, label: str) -> list[str]:
    argv = command.get("argv")
    if not isinstance(argv, list):
        raise LocateError("rejected", "BAD_ARGV", f"{label} argv is missing", [])
    prompts = flag_values(argv, "--prompt")
    windows = flag_values(argv, "--windows")
    if len(prompts) != 2 or len(set(prompts)) != 2 or windows != ["2"]:
        raise LocateError(
            "rejected",
            "COLLAPSED_REQUEST_IDENTITY",
            f"{label} does not record two distinct prompt windows",
            [],
        )
    return prompts


def model_of(command: dict, label: str) -> str:
    models = flag_values(command["argv"], "--model")
    if len(models) != 1:
        raise LocateError("rejected", "MODEL_IDENTITY", f"{label} model flag is not singular", [])
    return models[0]


def library_table(execution: dict, label: str) -> dict[str, str]:
    loaded = require_mapping(execution.get("loaded_libraries"), f"{label} loaded_libraries")
    if len(loaded) != 5:
        raise LocateError("rejected", "LIBRARY_COUNT", f"{label} does not record five libraries", [])
    table = {}
    for path, digest in loaded.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise LocateError("rejected", "LIBRARY_RECORD", f"{label} library record is not text", [])
        table[path] = digest
    return table


def path_is_candidate_install(path: str, root: Path) -> bool:
    install = (root / CANDIDATE_DIR_NAME / "install" / "lib").resolve()
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    if CANDIDATE_DIR_NAME not in Path(path).parts:
        return False
    try:
        resolved.relative_to(install)
    except ValueError:
        return False
    return True


def check_library_paths(table: dict[str, str], receipt: dict[str, str], root: Path, label: str) -> None:
    receipt_names = set(receipt)
    seen = set()
    for path, digest in table.items():
        if not path_is_candidate_install(path, root):
            raise LocateError(
                "rejected",
                "WRONG_LIBRARY_INSTALL",
                f"{label} library path is outside {CANDIDATE_DIR_NAME}/install/lib: {path}",
                [],
            )
        name = Path(path).name
        if name not in receipt or receipt[name] != digest:
            raise LocateError(
                "rejected",
                "LIBRARY_MISMATCH",
                f"{label} library {name} does not match the candidate build receipt",
                [],
            )
        seen.add(name)
    if seen != receipt_names:
        raise LocateError("rejected", "LIBRARY_MISMATCH", f"{label} library set differs from the build receipt", [])


def check_drain(comparison: dict) -> None:
    work = comparison.get("work")
    if not isinstance(work, list) or len(work) < 2:
        raise LocateError("rejected", "MISSING_DRAIN", "numeric comparison has no two-request drain record", [])
    identities = []
    for item in work:
        if not isinstance(item, dict) or item.get("native_drained") is not True:
            raise LocateError("rejected", "MISSING_DRAIN", "a work item has no native_drained true record", [])
        identities.append(item.get("id"))
    if len(set(identities)) < 2:
        raise LocateError("rejected", "COLLAPSED_REQUEST_IDENTITY", "drain records do not name two work ids", [])


def check_metrics(metrics: dict, label: str) -> None:
    requests = metrics.get("requests")
    if not isinstance(requests, list) or len(requests) != 2:
        raise LocateError("rejected", "COLLAPSED_REQUEST_IDENTITY", f"{label} metrics do not name two requests", [])
    identities = []
    for item in requests:
        if not isinstance(item, dict):
            raise LocateError("rejected", "COLLAPSED_REQUEST_IDENTITY", f"{label} request row is not an object", [])
        identities.append(item.get("request_id"))
    if len(set(identities)) != 2:
        raise LocateError("rejected", "COLLAPSED_REQUEST_IDENTITY", f"{label} request ids are not distinct", [])


def file_hashes_match(table: dict[str, str]) -> list[dict]:
    rows = []
    for path, expected in table.items():
        file_path = Path(path)
        if not file_path.is_file():
            raise LocateError("blocked", "MISSING_LIBRARY_FILE", "recorded library file is absent", [path])
        actual = sha256_file(file_path)
        if actual != expected:
            raise LocateError(
                "rejected",
                "LIBRARY_BYTES",
                f"on-disk bytes differ from the receipt for {file_path.name}",
                [],
            )
        rows.append({"path": path, "sha256": actual})
    return rows


def maps_mention(maps_path: Path, table: dict[str, str]) -> None:
    text = maps_path.read_text(encoding="utf-8", errors="replace")
    for path in table:
        marker = f"install/lib/{Path(path).name}"
        if marker not in text:
            raise LocateError("rejected", "LOADED_MAPS", f"loaded map omits {marker}", [str(maps_path)])


def cost_ceiling(frontier: dict, claim: dict | None) -> dict:
    stage = require_mapping(frontier.get("gpu_stage_cost"), "gpu_stage_cost")
    for key in ("n1_rank0_span_median_ms", "n3_rank0_span_median_ms"):
        if not isinstance(stage.get(key), (int, float)) or isinstance(stage.get(key), bool):
            raise LocateError("rejected", "STAGE_RECORD", f"gpu_stage_cost.{key} is not a number", [])
    n1 = float(stage["n1_rank0_span_median_ms"])
    n3 = float(stage["n3_rank0_span_median_ms"])
    source = stage.get("source")
    if not isinstance(source, str) or not source.endswith(DEVICE_STAGE_RELATIVE):
        raise LocateError("rejected", "WRONG_BASELINE", "stage source is not the named device diagnostic", [])
    answer = "no"
    reason = (
        "No measured candidate removal is attached. The named diagnostic records an "
        f"N=3 rank0 span of {n3} ms and an N=1 span of {n1} ms. Its scope says the gap "
        "includes graph submit and wait. That difference is not proof that a code change "
        "removes about 8–12 ms from the full N=3 target path, and it does not prove "
        "overlap that improves the slower stream. The kernel and scheduling change was not landed."
    )
    if claim is not None:
        removal = claim.get("removal_ms")
        full_path = claim.get("full_n3_target_path") is True
        overlap = claim.get("equivalent_overlap_improves_slower_stream") is True
        if full_path and isinstance(removal, (int, float)) and not isinstance(removal, bool) and float(removal) >= REMOVAL_FLOOR_MS:
            answer = "yes"
            reason = (
                "The attached claim records a full N=3 target-path removal of at least 8 ms. "
                "This locator still does not edit kernels or scheduling. The change was not landed."
            )
        elif full_path and overlap:
            answer = "yes"
            reason = (
                "The attached claim records equivalent overlap that improves the slower stream. "
                "This locator still does not edit kernels or scheduling. The change was not landed."
            )
        else:
            reason = (
                "The attached claim does not show about 8–12 ms removed from the full N=3 "
                "target path and does not show overlap that improves the slower stream. "
                "The kernel and scheduling change was not landed."
            )
    return {
        "answer": answer,
        "can_remove_about_8_to_12_ms_from_full_n3_target_path": answer == "yes" and claim is not None and claim.get("equivalent_overlap_improves_slower_stream") is not True and claim.get("full_n3_target_path") is True,
        "equivalent_overlap_improves_slower_stream": bool(claim and claim.get("equivalent_overlap_improves_slower_stream") is True and answer == "yes"),
        "kernel_or_scheduling_change_landed": False,
        "reason": reason,
        "observed_n1_rank0_span_median_ms": n1,
        "observed_n3_rank0_span_median_ms": n3,
        "observed_span_difference_ms": n3 - n1,
        "n3_extra_device_gap_approx_ms": stage.get("n3_extra_device_gap_approx_ms"),
        "moe_stage_group_extra_approx_ms": stage.get("moe_stage_group_extra_approx_ms"),
        "scope": stage.get("scope"),
        "number_label": "existing evidence, not a new measurement",
    }


def ratios_from_frontier(frontier: dict) -> dict:
    matched = require_mapping(frontier.get("matched_hc_mirrored_mtp_vs_plain"), "matched pair")
    row = require_mapping(frontier.get("row_tail_formal"), "row tail formal")
    return {
        "end_to_end_versus_no_mtp": {
            "source": "matched_hc_mirrored_mtp_vs_plain",
            "mtp_tok_s": matched.get("mtp_tok_s"),
            "plain_tok_s": matched.get("plain_tok_s"),
            "note": matched.get("note"),
        },
        "absolute_throughput_versus_ideal_envelope": {
            "source": "matched_hc_mirrored_mtp_vs_plain",
            "conditional_envelope": matched.get("conditional_envelope"),
            "realization_fraction": matched.get("realization_fraction"),
        },
        "new_gain_versus_theoretical_new_gain": {
            "source": "derived from matched_hc_mirrored_mtp_vs_plain fields already in the frontier file",
            "label": "预算假设 only if a field above is absent; otherwise derived from that file",
        },
        "row_tail_formal_on_off": {
            "source": "row_tail_formal",
            "on_tok_s": row.get("on_tok_s"),
            "off_tok_s": row.get("off_tok_s"),
            "cold_pause_caveat": row.get("cold_pause_caveat"),
        },
    }


def derive_new_gain(frontier: dict) -> list | None:
    matched = frontier["matched_hc_mirrored_mtp_vs_plain"]
    mtp = matched.get("mtp_tok_s")
    plain = matched.get("plain_tok_s")
    envelope = matched.get("conditional_envelope")
    if not all(isinstance(seq, list) and len(seq) == 2 for seq in (mtp, plain, envelope)):
        return None
    values = []
    for index in range(2):
        denominator = float(envelope[index]) - float(plain[index])
        if denominator == 0:
            return None
        values.append((float(mtp[index]) - float(plain[index])) / denominator)
    return values


def assess(root: Path, claim: dict | None = None) -> dict:
    root = root.resolve()
    if not root.is_dir() or root.name != EVIDENCE_DIR_NAME:
        raise LocateError("blocked", "MISSING_ROOT", "named evidence root is absent", [str(root)])
    missing = missing_required(root)
    if missing:
        raise LocateError("blocked", "MISSING_RECEIPT", "a required baseline receipt is absent", missing)
    candidate = root / CANDIDATE_DIR_NAME
    frontier = load_json(root / "acceptance-frontier-20261025.json")
    row = require_mapping(frontier.get("row_tail_formal"), "row_tail_formal")
    recorded_candidate = row.get("candidate")
    if not isinstance(recorded_candidate, str) or Path(recorded_candidate).name != CANDIDATE_DIR_NAME:
        raise LocateError("rejected", "WRONG_BASELINE", "frontier candidate is not the named row-tail baseline", [])
    device = root / DEVICE_STAGE_RELATIVE
    stage = frontier.get("gpu_stage_cost")
    if not isinstance(stage, dict) or not str(stage.get("source", "")).endswith(DEVICE_STAGE_RELATIVE):
        raise LocateError("rejected", "WRONG_BASELINE", "frontier stage source moved off the named diagnostic", [])
    if not device.is_file():
        raise LocateError("blocked", "MISSING_RECEIPT", "device diagnostic is absent", [str(device)])

    build = load_json(candidate / "build-receipt.json")
    receipt = require_mapping(build.get("libraries"), "build-receipt libraries")
    if len(receipt) != 5:
        raise LocateError("rejected", "LIBRARY_COUNT", "build receipt does not name five libraries", [])
    comparison = load_json(candidate / "numeric-comparison.json")
    check_drain(comparison)
    on_command = load_json(candidate / "on512x2" / "command.json")
    off_command = load_json(candidate / "off512x2" / "command.json")
    on_prompts = prompts_of(on_command, "on512x2")
    off_prompts = prompts_of(off_command, "off512x2")
    if on_prompts != off_prompts:
        raise LocateError("rejected", "COLLAPSED_REQUEST_IDENTITY", "on and off prompt lists differ", [])
    on_model = model_of(on_command, "on512x2")
    off_model = model_of(off_command, "off512x2")
    if on_model != off_model:
        raise LocateError("rejected", "MODEL_IDENTITY", "on and off model paths differ", [])
    identities = load_json(candidate / "on512x2" / "input-identities.json")
    model_paths = [item.get("path") for item in identities.get("model_files", []) if isinstance(item, dict)]
    if on_model not in model_paths:
        raise LocateError("rejected", "MODEL_IDENTITY", "command model path is absent from input identities", [])
    mtp = identities.get("mtp") if isinstance(identities.get("mtp"), dict) else {}
    environment = on_command.get("environment") if isinstance(on_command.get("environment"), dict) else {}
    if environment.get("GGML_TP5_REAL_TRAINED_MTP_GGUF") != mtp.get("path"):
        raise LocateError("rejected", "MODEL_IDENTITY", "MTP file in the command differs from input identities", [])
    on_exec = library_table(load_json(candidate / "on512x2" / "execution.json"), "on512x2")
    off_exec = library_table(load_json(candidate / "off512x2" / "execution.json"), "off512x2")
    identity_libs = identities.get("libraries")
    if not isinstance(identity_libs, dict):
        raise LocateError("rejected", "LIBRARY_RECORD", "input identities omit libraries", [])
    check_library_paths(on_exec, receipt, root, "on512x2")
    check_library_paths(off_exec, receipt, root, "off512x2")
    check_library_paths(identity_libs, receipt, root, "input-identities")
    if on_exec != off_exec or on_exec != identity_libs:
        raise LocateError("rejected", "LIBRARY_MISMATCH", "on, off, and input-identity library tables differ", [])
    comparison_exec = comparison.get("execution", {}).get("loaded_libraries")
    if comparison_exec != on_exec:
        raise LocateError("rejected", "LIBRARY_MISMATCH", "numeric comparison libraries differ from the formal run", [])
    check_metrics(load_json(candidate / "on512x2" / "metrics.json"), "on512x2")
    check_metrics(load_json(candidate / "off512x2" / "metrics.json"), "off512x2")
    maps_mention(candidate / "on512x2" / "loaded-maps.txt", on_exec)
    maps_mention(candidate / "off512x2" / "loaded-maps.txt", off_exec)
    hashed = file_hashes_match(on_exec)
    ceiling = cost_ceiling(frontier, claim)
    ratio_report = ratios_from_frontier(frontier)
    ratio_report["new_gain_versus_theoretical_new_gain"]["per_stream"] = derive_new_gain(frontier)
    return {
        "schema": "mtp-baseline-locator/1",
        "decision": "consistent",
        "requested_mode": "handbook final two-request 512-token qualification on E5",
        "executed_mode": "cpu read of the named 2026-10-25 receipts",
        "not_run": ["gpu", "model", "formal_512x2", "native_drain"],
        "gates": {
            "G0": "blocked",
            "G0_reason": "This process does not hold the E5 experiment lock and does not observe five-GPU occupancy.",
            "G1": "blocked",
            "G1_reason": "Five on-disk libraries match the named execution and build receipts. A new rebuild, full source and header closure, and relink were not run.",
            "G1_recorded_library_bytes": "match",
            "G3": "SKIP",
            "G4": "SKIP",
            "G5": "SKIP",
            "formal_dual_stream_pass": False,
        },
        "locator": {
            "evidence_root": str(root),
            "candidate": str(candidate),
            "command_json": str(candidate / "on512x2" / "command.json"),
            "loaded_maps": str(candidate / "on512x2" / "loaded-maps.txt"),
            "execution_json": str(candidate / "on512x2" / "execution.json"),
            "build_receipt": str(candidate / "build-receipt.json"),
            "numeric_comparison": str(candidate / "numeric-comparison.json"),
            "input_identities": str(candidate / "on512x2" / "input-identities.json"),
            "model": on_model,
            "mtp": mtp.get("path"),
            "frontiers": [str(root / name) for name in FRONTIER_NAMES],
            "device_stage": str(device),
            "snapshot": str(root / SNAPSHOT_DIR_NAME),
            "selected_dual_config": "absent from section 1.3 named receipts; not used as this baseline",
        },
        "libraries": hashed,
        "prompts": on_prompts,
        "ratios": ratio_report,
        "cost_ceiling": ceiling,
        "a_source_edited": False,
    }


def load_claim(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        raise LocateError("blocked", "MISSING_CLAIM", "removal claim file is absent", [str(path)])
    value = load_json(path)
    if not isinstance(value, dict):
        raise LocateError("rejected", "BAD_CLAIM", "removal claim is not an object", [str(path)])
    return value


def report_from_error(error: LocateError) -> dict:
    return {
        "schema": "mtp-baseline-locator/1",
        "decision": error.decision,
        "code": error.code,
        "detail": error.detail,
        "missing_paths": error.missing,
        "requested_mode": "handbook final two-request 512-token qualification on E5",
        "executed_mode": "cpu read of the named 2026-10-25 receipts",
        "not_run": ["gpu", "model", "formal_512x2", "native_drain"],
        "gates": {"G3": "SKIP", "G4": "SKIP", "G5": "SKIP", "formal_dual_stream_pass": False},
        "cost_ceiling": {
            "answer": "no",
            "kernel_or_scheduling_change_landed": False,
            "reason": "The identity check did not reach a cost-justified kernel change. The change was not landed.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--removal-claim", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.evidence_root if args.evidence_root is not None else repo_evidence_root()
        result = assess(root, load_claim(args.removal_claim))
        code = 0
    except LocateError as error:
        result = report_from_error(error)
        code = 3 if error.decision == "blocked" else 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
