#!/usr/bin/env python3
"""
scripts/xkv/collect-runtime-evidence.py
---------------------------------------
Collects runtime resource evidence into the exact compression-gate keys.

Inputs: llama-server /metrics snapshots (Prometheus text or JSON dict) taken
before/after a run, an optional baseline snapshot for peak ratios, an optional
quality/compat sidecar JSON, and the matrix-run manifest.

Derives:
  - Factored byte coverage: ONLY from explicit covered-baseline byte metrics
    (xkv_factored_baseline_same_rows_bytes / xkv_baseline_same_rows_bytes or
    explicit xkv_factored_baseline_byte_coverage metric in [0, 1]).
    NEVER derived from compressed factor bytes (fail-closed).
  - Effective profile: derived SOLELY from observed runtime metrics/info
    (xkv_profile_info{effective="..."}). Never from requested target or manifest.
    Missing observation => "not_evaluated" (fails closed in gate).
  - Source and profile fingerprints: parsed from actual info series
    xkv_source_info{fingerprint="..."} and
    xkv_profile_fingerprint_info{fingerprint="..."}.
  - Strips llamacpp: metric/info namespace uniformly.
  - Propagates synthetic fixture flags from quality sidecar or manifest;
    any synthetic fixture is marked __synthetic_test_fixture__: true so
    the production compression gate rejects it unconditionally.
  - Allocator, workspace, decode-cache, dense mirror, and sync counters.
  - Underivable keys are "not_evaluated" (fail-closed, never fabricated).

Only standard library Python is used. Writes are atomic (os.replace).
Evidence schema: --schema prints EVIDENCE_SCHEMA_JSON ($schema "xkv-evidence/v1").
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

VERSION = 1

# Profile JSON schema version with explicit mode/factorizer/budgets/server
# defaults. Exact supported version is validated; old/unknown fail closed.
PROFILE_SCHEMA_VERSION = 2


def require_profile_schema(profile_data: Dict[str, Any]) -> int:
    """Validates the exact supported profile schema version."""
    v = profile_data.get("version")
    if v != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"profile schema version {v!r} unsupported: this harness requires "
            f"version {PROFILE_SCHEMA_VERSION} (explicit mode/factorizer/budgets/"
            f"server-defaults per XKV-SR audit). Migrate the profile JSON: set "
            f"\"version\": {PROFILE_SCHEMA_VERSION} and pin mode, factorizer, "
            f"store/workspace/decode-cache budgets, and server defaults on every profile.")
    return v

EVIDENCE_SCHEMA_JSON = {
    "$schema": "xkv-evidence/v1",
    "description": (
        "Runtime resource evidence consumed by scripts/xkv/compression-gate.py. "
        "Keys mirror GateEvaluator evidence fields exactly. "
        "Missing/unmeasured keys MUST be the string 'not_evaluated'."
    ),
    "keys": {
        "xkv_requested_profile": "str",
        "xkv_effective_profile": "str | 'not_evaluated'",
        "xkv_effective_storage_profile": "str | 'not_evaluated'",
        "xkv_codec_a_k": "str | 'not_evaluated'",
        "xkv_codec_b_k": "str | 'not_evaluated'",
        "xkv_codec_a_v": "str | 'not_evaluated'",
        "xkv_codec_b_v": "str | 'not_evaluated'",
        "xkv_codec_landmark": "str | 'not_evaluated'",
        "calibration_sha256": "str hex | 'not_evaluated'",
        "xkv_factored_baseline_byte_coverage": "float 0..1 | 'not_evaluated'",
        "has_resident_dense_mirror": "bool | 'not_evaluated'",
        "xkv_net_saved_fraction": "float | 'not_evaluated'",
        "xkv_net_extra_compression_ratio": "float | 'not_evaluated'",
        "xkv_peak_vs_baseline_ratio": "float | 'not_evaluated'",
        "xkv_live_bytes": "int | 'not_evaluated'",
        "xkv_reserved_bytes": "int | 'not_evaluated'",
        "xkv_peak_bytes": "int | 'not_evaluated'",
        "source_fingerprint": "str | 'not_evaluated'",
        "profile_fingerprint": "str | 'not_evaluated'",
        "quality_ppl_degradation_pct": "float | 'not_evaluated'",
        "quality_context_scaling_ratio": "float | 'not_evaluated'",
        "speculative_acceptance_parity": "bool | 'not_evaluated'",
        "triattention_compatible": "bool | 'not_evaluated'",
        "rerot_compatible": "bool | 'not_evaluated'",
        "state_code_stream_exact": "bool | 'not_evaluated'",
    },
}

DIAG_COUNTERS = (
    "xkv_segments_sealed",
    "xkv_sr_selected_rows",
    "xkv_sr_fragments",
    "xkv_landmark_refine_rows",
    "xkv_landmark_refine_cap_hits",
    "xkv_spec_stale_total",
    "xkv_transaction_abort_total",
    "tri_drain_total",
    "tri_maintenance_total",
    "tri_floor_exhausted_total",
)


# ---------------------------------------------------------------- utilities

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_nonneg_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def strip_llamacpp_prefix(name: str) -> str:
    if name.startswith("llamacpp:"):
        return name[len("llamacpp:"):]
    return name


# ---------------------------------------------------------------- metrics parsing

def parse_prometheus_text(text: str) -> Tuple[Dict[str, float], List[str], Dict[str, str]]:
    """Parses Prometheus exposition text into (metrics, notes, info_labels).

    Strips 'llamacpp:' metric namespace uniformly.
    Parses actual info metrics:
      - xkv_source_fingerprint_info{fingerprint="..."} (legacy xkv_source_info accepted)
      - xkv_profile_fingerprint_info{fingerprint="..."}
      - xkv_codec_info{fingerprint="..."}
      - xkv_backend_info{fingerprint="..."}
      - xkv_profile_info{requested="...",effective="..."}
    Drops non-finite values (NaN, Inf) fail-closed.
    """
    metrics: Dict[str, float] = {}
    notes: List[str] = []
    info: Dict[str, str] = {}

    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.rsplit(None, 1)
        if len(parts) != 2:
            notes.append(f"line {lineno}: unparseable, skipped")
            continue

        name_part, val_part = parts
        raw_name = name_part.split("{", 1)[0].strip()
        name = strip_llamacpp_prefix(raw_name)
        if not name:
            notes.append(f"line {lineno}: empty metric name, skipped")
            continue

        # Parse labels if present
        labels: Dict[str, str] = {}
        if "{" in name_part and "}" in name_part:
            label_body = name_part.split("{", 1)[1].rsplit("}", 1)[0]
            for kv in label_body.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')

        # Handle specific info series
        # xkv_source_info carries the {source} label; the fingerprint series
        # is xkv_source_fingerprint_info (legacy fingerprint-labeled
        # xkv_source_info still accepted for back-compat).
        if name in ("xkv_source_fingerprint_info", "xkv_source_info"):
            if "fingerprint" in labels:
                info["source_fingerprint"] = labels["fingerprint"]
        elif name == "xkv_profile_fingerprint_info":
            if "fingerprint" in labels:
                info["profile_fingerprint"] = labels["fingerprint"]
        elif name == "xkv_codec_info":
            if "fingerprint" in labels:
                info["codec_fingerprint"] = labels["fingerprint"]
        elif name == "xkv_backend_info":
            if "fingerprint" in labels:
                info["backend_fingerprint"] = labels["fingerprint"]
        elif name == "xkv_profile_info":
            if "requested" in labels:
                info["xkv_profile_requested"] = labels["requested"]
            if "effective" in labels:
                info["xkv_profile_effective"] = labels["effective"]
        elif name == "tri_calibration_info":
            if "sha256" in labels:
                info["tri_calibration_sha256"] = labels["sha256"]
                info["calibration_sha256"] = labels["sha256"]
            if "fingerprint" in labels:
                info["tri_calibration_fingerprint"] = labels["fingerprint"]
        elif name in ("xkv_build_info", "xkv_fingerprint_info"):
            info.update(labels)

        try:
            v = float(val_part)
        except ValueError:
            notes.append(f"line {lineno}: non-numeric value for {name}, skipped")
            continue
        if not math.isfinite(v):
            notes.append(f"line {lineno}: non-finite value for {name}, dropped (fail-closed)")
            continue

        metrics[name] = metrics.get(name, 0.0) + v

    # extract xkv_compression_goal_met tri-state:
    # evaluated gauge present and 0 -> not_evaluated (null/NaN)
    goal_eval = metrics.get("xkv_compression_goal_evaluated")
    goal_met_raw = metrics.get("xkv_compression_goal_met")
    if goal_eval is not None and goal_eval == 0:
        info["xkv_compression_goal_met"] = "not_evaluated"
    elif goal_met_raw is not None and math.isfinite(goal_met_raw):
        info["xkv_compression_goal_met"] = "true" if goal_met_raw > 0 else "false"
    else:
        info["xkv_compression_goal_met"] = "not_evaluated"

    return metrics, notes, info


def load_snapshot(path: str) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """Loads metrics snapshot; supports Prometheus exposition and JSON dict.

    Strips 'llamacpp:' prefix across both formats.
    """
    if not path:
        return {}, ["no snapshot path supplied"], {}
    if not os.path.exists(path):
        return {}, [f"metrics snapshot missing: {path}"], {}

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            metrics: Dict[str, Any] = {}
            info: Dict[str, str] = {}
            notes: List[str] = []
            for k, v in obj.items():
                nk = strip_llamacpp_prefix(k)
                if isinstance(v, dict):
                    # info-like nested dict in JSON format
                    if nk == "xkv_source_info" and "fingerprint" in v:
                        info["source_fingerprint"] = str(v["fingerprint"])
                    elif nk == "xkv_profile_fingerprint_info" and "fingerprint" in v:
                        info["profile_fingerprint"] = str(v["fingerprint"])
                    elif nk == "xkv_profile_info":
                        if "requested" in v:
                            info["xkv_profile_requested"] = str(v["requested"])
                        if "effective" in v:
                            info["xkv_profile_effective"] = str(v["effective"])
                elif isinstance(v, str):
                    if nk in ("source_fingerprint", "xkv_source_fingerprint"):
                        info["source_fingerprint"] = v
                    elif nk in ("profile_fingerprint", "xkv_profile_fingerprint"):
                        info["profile_fingerprint"] = v
                    elif nk in ("effective_profile", "xkv_effective_profile"):
                        info["xkv_profile_effective"] = v
                    elif nk in ("requested_profile", "xkv_requested_profile"):
                        info["xkv_profile_requested"] = v
                elif is_finite_number(v):
                    metrics[nk] = float(v)
                else:
                    notes.append(f"{k} dropped (non-numeric/non-finite)")
            return metrics, notes, info
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return parse_prometheus_text(raw)


def delta_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    keys = set(before) | set(after)
    out: Dict[str, float] = {}
    for k in sorted(keys):
        b, a = before.get(k), after.get(k)
        if is_finite_number(b) and is_finite_number(a):
            out[k] = float(a) - float(b)
    return out


# ---------------------------------------------------------------- nominal cross-check

def nominal_reference(w_layers: int = 4, d_k: int = 1024, d_v: int = 1024,
                      n_tokens: int = 4096, rk: int = 384, rv: int = 576) -> Dict[str, Any]:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory-model.py")
    if os.path.exists(here):
        try:
            spec = importlib.util.spec_from_file_location("xkv_memory_model", here)
            assert spec and spec.loader
            mm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mm)
            res = mm.calculate_nominal_case(w_layers, d_k, d_v, n_tokens, rk, rv)
            return {"source": "scripts/xkv/memory-model.py", "table": res["nominal_table"]}
        except Exception as e:
            return {"source": "memory-model-error", "error": repr(e)}
    n, W = n_tokens, w_layers
    orig = 2 * n * W * (d_k + d_v)
    a_elems = n * (rk + rv)
    b_elems = rk * W * d_k + rv * W * d_v
    lm_elems = (n // 8) * W * d_k
    mib = 1024 * 1024
    return {"source": "closed-form-fallback",
            "table": [{"total_bytes": b, "total_mib": b / mib} for b in
                      (orig, orig // 32 * 6, 2 * (a_elems + b_elems) + 2 * lm_elems,
                       (a_elems + b_elems) // 2 + 2 * lm_elems,
                       (a_elems + b_elems) // 2 + lm_elems,
                       (a_elems + b_elems) // 2 + lm_elems // 2)]}


# ---------------------------------------------------------------- evidence assembly

def _int_or_ne(v: Any) -> Any:
    if is_nonneg_int(v):
        return v
    if is_finite_number(v) and float(v).is_integer() and v >= 0:
        return int(v)
    return "not_evaluated"


def _float_or_ne(v: Any) -> Any:
    if is_finite_number(v):
        return float(v)
    return "not_evaluated"


def build_evidence(args: argparse.Namespace,
                   metrics: Dict[str, Any],
                   notes: List[str],
                   info: Dict[str, str],
                   manifest: Dict[str, Any],
                   profile_data: Dict[str, Any],
                   quality: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
    reasons: List[str] = []
    require_profile_schema(profile_data)
    target = args.target_profile or manifest.get("profile", "E1")
    spec = profile_data.get("profiles", {}).get(target, {})
    if not spec:
        reasons.append(f"target profile {target!r} not found in profile JSON")

    if quality is None:
        quality = {}

    def m(key: str) -> Any:
        v = metrics.get(key)
        return v if is_finite_number(v) else None

    # -- Synthetic fixture detection & propagation
    is_synthetic = bool(
        args.dry_run
        or args.self_test
        or quality.get("__synthetic_test_fixture__")
        or quality.get("synthetic_test_fixture")
        or manifest.get("__synthetic_test_fixture__")
        or manifest.get("synthetic_test_fixture")
        or "SYNTHETIC" in str(quality.get("note", ""))
        or "SYNTHETIC" in str(manifest.get("note", ""))
    )

    # -- Effective profile: derived SOLELY from observed runtime metrics/info
    # NEVER from requested target or manifest. Server labels are STORAGE names:
    # the matrix label is identified iff observed storage == spec.storage_profile.
    exp_storage = spec.get("storage_profile", "none")
    observed_eff_storage = (
        info.get("xkv_profile_effective")
        or (metrics.get("xkv_effective_profile") if isinstance(metrics.get("xkv_effective_profile"), str) else None)
        or (metrics.get("xkv_profile_effective") if isinstance(metrics.get("xkv_profile_effective"), str) else None)
    )
    observed_req_storage = info.get("xkv_profile_requested")
    if spec.get("xkv_enabled"):
        if observed_eff_storage == exp_storage:
            eff_profile = target
        else:
            eff_profile = "not_evaluated"
            reasons.append(
                f"observed effective storage {observed_eff_storage!r} != plan storage "
                f"{exp_storage!r} for matrix {target!r}")
        if observed_req_storage == exp_storage:
            req_profile = target
        else:
            req_profile = "not_evaluated"
            reasons.append(
                f"observed requested storage {observed_req_storage!r} != plan storage "
                f"{exp_storage!r} for matrix {target!r}")
    else:
        req_profile = target
        if observed_eff_storage in (None, "", "none"):
            eff_profile = target
        else:
            eff_profile = "not_evaluated"
            reasons.append(
                f"foreign effective storage {observed_eff_storage!r} on baseline {target!r}")

    # -- Factored byte coverage: ONLY from explicit covered-baseline byte metric
    # NEVER derived from compressed factor bytes (fail-closed).
    cov_metric = m("xkv_factored_baseline_byte_coverage")
    cov_bytes = (
        m("xkv_factored_baseline_same_rows_bytes")
        or m("xkv_covered_baseline_bytes")
        or m("xkv_factored_baseline_bytes")
        or m("xkv_covered_baseline_same_rows_bytes")
    )
    base_bytes = m("xkv_baseline_same_rows_bytes")

    coverage: Any = "not_evaluated"
    if cov_bytes is not None and base_bytes and base_bytes > 0:
        c = float(cov_bytes) / float(base_bytes)
        if 0.0 <= c <= 1.0:
            coverage = c
        else:
            reasons.append(
                f"derived coverage out of range [0, 1]: {c:.4f} ({cov_bytes} / {base_bytes})"
            )
    elif cov_metric is not None:
        c = float(cov_metric)
        if 0.0 <= c <= 1.0:
            coverage = c
        else:
            reasons.append(f"xkv_factored_baseline_byte_coverage metric out of range [0, 1]: {c:.4f}")
    else:
        reasons.append(
            "missing explicit covered-baseline byte metric "
            "(xkv_factored_baseline_same_rows_bytes or xkv_factored_baseline_byte_coverage); "
            "NEVER derived from compressed factor bytes"
        )

    # -- Factored compressed bytes (for diagnostics / net savings, NOT coverage)
    factor_keys = ("xkv_factor_ak_bytes", "xkv_factor_bk_bytes",
                   "xkv_factor_av_bytes", "xkv_factor_bv_bytes")
    factor_vals = [m(k) for k in factor_keys]
    lm_bytes = m("xkv_landmark_payload_bytes")
    factored_compressed = float(sum(factor_vals)) if all(v is not None for v in factor_vals) else None

    # -- Net savings vs baseline
    aux_keys = ("xkv_landmark_metadata_bytes", "xkv_factor_metadata_bytes",
                "xkv_factor_padding_bytes", "xkv_index_bytes")
    aux = sum(v for k in aux_keys if (v := m(k)) is not None)
    aux_known = all(m(k) is not None for k in aux_keys)
    if factored_compressed is not None and lm_bytes is not None and base_bytes and base_bytes > 0:
        cold_new = factored_compressed + float(lm_bytes) + (aux if aux_known else 0.0)
        saved_fraction = 1.0 - cold_new / float(base_bytes)
        extra_ratio = float(base_bytes) / cold_new if cold_new > 0 else "not_evaluated"
    else:
        cold_new = "not_evaluated"
        saved_fraction, extra_ratio = "not_evaluated", "not_evaluated"
        reasons.append("missing factor/landmark/baseline gauges for net savings")

    # -- Dense mirror detection
    mirror = metrics.get("has_resident_dense_mirror", metrics.get("xkv_resident_dense_mirror"))
    if isinstance(mirror, bool):
        dense_mirror: Any = mirror
    elif is_finite_number(mirror):
        dense_mirror = bool(mirror)
    else:
        count = m("xkv_resident_dense_mirror_count")
        nbytes = m("xkv_resident_dense_mirror_bytes")
        if count is not None or nbytes is not None:
            dense_mirror = bool((count or 0) > 0 or (nbytes or 0) > 0)
        else:
            dense_mirror = "not_evaluated"
            reasons.append("no dense-mirror gauge (has_resident_dense_mirror)")

    # -- Allocator / peak bytes
    live = _int_or_ne(metrics.get("xkv_allocator_live_bytes", metrics.get("xkv_live_bytes")))
    reserved = _int_or_ne(metrics.get("xkv_allocator_reserved_bytes", metrics.get("xkv_reserved_bytes")))
    peak = _int_or_ne(metrics.get("xkv_device_peak_bytes", metrics.get("xkv_peak_bytes")))
    if live == "not_evaluated" or reserved == "not_evaluated" or peak == "not_evaluated":
        reasons.append("missing allocator live/reserved/peak gauges")

    peak_ratio: Any = "not_evaluated"
    base_peak = args.baseline_peak_bytes
    if base_peak is None and args.metrics_baseline:
        bm, _, _ = load_snapshot(args.metrics_baseline)
        bp = bm.get("xkv_device_peak_bytes", bm.get("xkv_peak_bytes"))
        base_peak = bp if is_finite_number(bp) else None
    if peak != "not_evaluated" and is_finite_number(base_peak) and base_peak > 0:
        peak_ratio = float(peak) / float(base_peak)
    else:
        reasons.append("no baseline peak for xkv_peak_vs_baseline_ratio")

    # -- Codecs / storage profile from profile JSON
    def codec(key: str, fallback: str = "not_evaluated") -> Any:
        v = spec.get(key, fallback)
        return v if isinstance(v, str) and v else "not_evaluated"

    # -- Fingerprints & calibration
    calib = manifest.get("calibration_sha256") or info.get("calibration_sha256", "")
    if not calib:
        calib = "not_evaluated"
        reasons.append("missing calibration sha256")

    src_fp = info.get("source_fingerprint") or (
        manifest.get("source_fingerprint") if not (args.dry_run or args.self_test) else None
    )
    if not src_fp:
        src_fp = "not_evaluated"
        reasons.append("missing source fingerprint (xkv_source_info{fingerprint})")

    prof_fp = info.get("profile_fingerprint") or (
        manifest.get("profile_fingerprint") if not (args.dry_run or args.self_test) else None
    )
    if not prof_fp:
        prof_fp = "not_evaluated"
        reasons.append("missing profile fingerprint (xkv_profile_fingerprint_info{fingerprint})")

    # -- Quality / compatibility sidecar
    def qbool(key: str) -> Any:
        v = quality.get(key, "not_evaluated")
        if isinstance(v, bool):
            return v
        if v != "not_evaluated":
            reasons.append(f"quality key {key} malformed (want bool): {v!r}")
        return "not_evaluated"

    evidence = {
        "__synthetic_test_fixture__": is_synthetic,
        "note": ("SYNTHETIC TEST FIXTURE ONLY - NOT REAL EMPIRICAL EVIDENCE"
                 if is_synthetic else
                 "runtime resource evidence from metrics snapshots"),
        "xkv_requested_profile": req_profile,
        "xkv_effective_profile": eff_profile,
        "xkv_effective_storage_profile": (observed_eff_storage
                                            if observed_eff_storage else "not_evaluated"),
        "matrix_profile": target,
        "expected_storage_profile": exp_storage,
        "xkv_codec_a_k": codec("a_k_codec", spec.get("k_codec", "not_evaluated")),
        "xkv_codec_b_k": codec("b_k_codec", spec.get("k_codec", "not_evaluated")),
        "xkv_codec_a_v": codec("a_v_codec", spec.get("v_codec", "not_evaluated")),
        "xkv_codec_b_v": codec("b_v_codec", spec.get("v_codec", "not_evaluated")),
        "xkv_codec_landmark": codec("landmark_codec"),
        "calibration_sha256": calib,
        "xkv_factored_baseline_byte_coverage": coverage,
        "has_resident_dense_mirror": dense_mirror,
        "xkv_net_saved_fraction": _float_or_ne(saved_fraction),
        "xkv_net_extra_compression_ratio": _float_or_ne(extra_ratio),
        "xkv_peak_vs_baseline_ratio": _float_or_ne(peak_ratio),
        "xkv_live_bytes": live,
        "xkv_reserved_bytes": reserved,
        "xkv_peak_bytes": peak,
        "source_fingerprint": src_fp,
        "profile_fingerprint": prof_fp,
        "quality_ppl_degradation_pct": _float_or_ne(
            quality.get("quality_ppl_degradation_pct", "not_evaluated")),
        "quality_context_scaling_ratio": _float_or_ne(
            quality.get("quality_context_scaling_ratio", "not_evaluated")),
        "speculative_acceptance_parity": qbool("speculative_acceptance_parity"),
        "triattention_compatible": qbool("triattention_compatible"),
        "rerot_compatible": qbool("rerot_compatible"),
        "state_code_stream_exact": qbool("state_code_stream_exact"),
        "diagnostics": {
            "factored_compressed_bytes": _float_or_ne(factored_compressed),
            "landmark_bytes": _float_or_ne(lm_bytes),
            "cold_new_bytes": _float_or_ne(cold_new),
            "baseline_same_rows_bytes": _float_or_ne(base_bytes),
            "workspace_peak_bytes": _int_or_ne(metrics.get("xkv_workspace_peak_bytes", "not_evaluated")),
            "decode_tile_cache_bytes": _int_or_ne(metrics.get("xkv_decode_tile_cache_bytes", "not_evaluated")),
            "compression_goal_met": info.get("xkv_compression_goal_met", "not_evaluated"),
            "counters": {k: metrics[k] for k in DIAG_COUNTERS if k in metrics},
            "budget": {
                "factor_budget_bytes": _int_or_ne(metrics.get("xkv_factor_budget_bytes", "not_evaluated")),
                "factor_free_bytes": _int_or_ne(metrics.get("xkv_factor_free_bytes", "not_evaluated")),
                "workspace_budget_bytes": _int_or_ne(metrics.get("xkv_workspace_budget_bytes", "not_evaluated")),
                "workspace_free_bytes": _int_or_ne(metrics.get("xkv_workspace_free_bytes", "not_evaluated")),
            },
            "metrics_notes": notes,
        },
        "provenance": {
            "metrics_after": args.metrics_after,
            "metrics_before": args.metrics_before,
            "metrics_baseline": args.metrics_baseline,
            "manifest": args.manifest,
            "nominal": nominal_reference(),
        },
        "collector_reasons": reasons,
    }
    return evidence, reasons


# ---------------------------------------------------------------- self-test

def _synthetic_metrics(path: str, scale: float = 1.0, include_coverage: bool = True,
                       requested: str = "tq-factors-landmarks",
                       effective: str = "tq-factors-landmarks") -> None:
    lines = [
        "# synthetic metrics fixture with llamacpp: namespace",
        f"llamacpp:xkv_factor_ak_bytes {int(1048576 * scale)}",
        f"llamacpp:xkv_factor_bk_bytes {int(1048576 * scale)}",
        f"llamacpp:xkv_factor_av_bytes {int(1572864 * scale)}",
        f"llamacpp:xkv_factor_bv_bytes {int(1572864 * scale)}",
        f"llamacpp:xkv_landmark_payload_bytes {int(2097152 * scale)}",
        "llamacpp:xkv_landmark_metadata_bytes 512",
        "llamacpp:xkv_factor_metadata_bytes 512",
        "llamacpp:xkv_factor_padding_bytes 0",
        "llamacpp:xkv_index_bytes 16384",
        f"llamacpp:xkv_baseline_same_rows_bytes {int(12582912 * scale)}",
        "llamacpp:xkv_allocator_live_bytes 10485760",
        "llamacpp:xkv_allocator_reserved_bytes 16777216",
        "llamacpp:xkv_device_peak_bytes 20971520",
        "llamacpp:xkv_workspace_peak_bytes 1048576",
        "llamacpp:xkv_decode_tile_cache_bytes 524288",
        "llamacpp:xkv_resident_dense_mirror_count 0",
        "llamacpp:xkv_resident_dense_mirror_bytes 0",
        "llamacpp:xkv_factor_budget_bytes 8388608",
        "llamacpp:xkv_factor_free_bytes 3145728",
        "llamacpp:xkv_workspace_budget_bytes 268435456",
        "llamacpp:xkv_workspace_free_bytes 267386880",
        "llamacpp:tri_drain_total 3",
        'llamacpp:xkv_source_fingerprint_info{fingerprint="0123456789abcdef"} 1',
        'llamacpp:xkv_profile_fingerprint_info{fingerprint="fedcba9876543210"} 1',
        f'llamacpp:xkv_profile_info{{requested="{requested}",effective="{effective}"}} 1',
    ]
    if include_coverage:
        lines.append(f"llamacpp:xkv_factored_baseline_same_rows_bytes {int(10695475 * scale)}")
        lines.append("llamacpp:xkv_factored_baseline_byte_coverage 0.85")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_self_test() -> Dict[str, Any]:
    import shutil
    checks: Dict[str, Any] = {}
    tmp = tempfile.mkdtemp(prefix="collect-selftest-")
    try:
        prof_path = os.path.join(tmp, "profiles.json")
        prof_stub = {"version": 2, "profiles": {"E1": {
            "xkv_enabled": True, "storage_profile": "tq-factors-landmarks",
            "a_k_codec": "turbo4_0", "b_k_codec": "turbo4_0",
            "a_v_codec": "turbo4_0", "b_v_codec": "turbo4_0",
            "landmark_codec": "q8_0"}}}
        with open(prof_path, "w") as f:
            json.dump(prof_stub, f)

        # 1. Namespace stripping & actual fingerprint extraction
        mp = os.path.join(tmp, "metrics.txt")
        _synthetic_metrics(mp)
        metrics, notes, info = load_snapshot(mp)
        assert "xkv_factor_ak_bytes" in metrics, "llamacpp: prefix not stripped"
        assert "llamacpp:xkv_factor_ak_bytes" not in metrics
        assert info.get("source_fingerprint") == "0123456789abcdef", info
        assert info.get("profile_fingerprint") == "fedcba9876543210", info
        assert info.get("xkv_profile_effective") == "tq-factors-landmarks", info
        checks["namespace_and_fingerprints"] = {"status": "PASS"}

        # 2. Coverage derivation: ONLY from explicit covered baseline bytes, NOT compressed factor bytes
        man = {"profile": "E1", "calibration_sha256": "e95dae507d1f4a64e29be160c5281f8a4308a3332dc9c9176e1a3a0af32e50e2"}
        args = argparse.Namespace(
            target_profile="E1", metrics_after=mp, metrics_before="",
            metrics_baseline="", manifest="", quality="",
            baseline_peak_bytes=19922944.0, dry_run=False, self_test=True)
        ev, reasons = build_evidence(args, metrics, notes, info, man, prof_stub)
        # Expected: 10695475 / 12582912 = 0.85
        assert abs(ev["xkv_factored_baseline_byte_coverage"] - 0.85) < 1e-4, ev["xkv_factored_baseline_byte_coverage"]
        assert ev["source_fingerprint"] == "0123456789abcdef"
        assert ev["profile_fingerprint"] == "fedcba9876543210"
        assert ev["xkv_effective_profile"] == "E1"
        assert ev["diagnostics"]["budget"]["factor_budget_bytes"] == 8388608, ev["diagnostics"]["budget"]
        assert ev["diagnostics"]["budget"]["workspace_budget_bytes"] == 268435456
        checks["coverage_from_covered_baseline"] = {"status": "PASS", "coverage": ev["xkv_factored_baseline_byte_coverage"]}

        # 3. Adversarial coverage test: missing explicit covered baseline bytes MUST NOT fall back to factor bytes!
        mp_no_cov = os.path.join(tmp, "metrics_no_cov.txt")
        _synthetic_metrics(mp_no_cov, include_coverage=False)
        m_nc, n_nc, i_nc = load_snapshot(mp_no_cov)
        ev_nc, _ = build_evidence(args, m_nc, n_nc, i_nc, man, prof_stub)
        assert ev_nc["xkv_factored_baseline_byte_coverage"] == "not_evaluated", (
            f"Expected not_evaluated coverage when explicit covered metric missing, got {ev_nc['xkv_factored_baseline_byte_coverage']}"
        )
        checks["coverage_never_derived_from_compressed"] = {"status": "PASS"}

        # 4. Adversarial effective profile test: missing observation => not_evaluated
        mp_no_prof = os.path.join(tmp, "metrics_no_prof.txt")
        with open(mp_no_prof, "w") as f:
            f.write("llamacpp:xkv_live_bytes 100\n")
        m_np, n_np, i_np = load_snapshot(mp_no_prof)
        ev_np, _ = build_evidence(args, m_np, n_np, i_np, man, prof_stub)
        assert ev_np["xkv_effective_profile"] == "not_evaluated", "must be not_evaluated when unobserved"
        checks["effective_profile_unobserved_fails_closed"] = {"status": "PASS"}

        # 5. Adversarial effective profile mismatch test: silent degradation caught
        mp_degrade = os.path.join(tmp, "metrics_degrade.txt")
        _synthetic_metrics(mp_degrade, requested="tq-factors-landmarks",
                            effective="tq-factors")
        m_deg, n_deg, i_deg = load_snapshot(mp_degrade)
        ev_deg, r_deg = build_evidence(args, m_deg, n_deg, i_deg, man, prof_stub)
        assert ev_deg["xkv_effective_profile"] == "not_evaluated", ev_deg["xkv_effective_profile"]
        assert ev_deg["xkv_effective_storage_profile"] == "tq-factors"
        assert any("effective storage" in r for r in r_deg), r_deg
        assert ev_deg["xkv_requested_profile"] == "E1", ev_deg["xkv_requested_profile"]
        checks["effective_profile_degradation_detected"] = {"status": "PASS"}

        # 6. Adversarial synthetic propagation: synthetic quality sidecar forces __synthetic_test_fixture__
        q_sidecar = {"__synthetic_test_fixture__": True, "speculative_acceptance_parity": True}
        ev_syn, _ = build_evidence(args, metrics, notes, info, man, prof_stub, quality=q_sidecar)
        assert ev_syn["__synthetic_test_fixture__"] is True
        checks["synthetic_flag_propagated_from_quality"] = {"status": "PASS"}

        # 7. Quality sidecar with note 'SYNTHETIC' forces __synthetic_test_fixture__
        q_note = {"note": "SYNTHETIC TEST FIXTURE", "speculative_acceptance_parity": True}
        ev_syn2, _ = build_evidence(args, metrics, notes, info, man, prof_stub, quality=q_note)
        assert ev_syn2["__synthetic_test_fixture__"] is True
        checks["synthetic_note_propagated"] = {"status": "PASS"}

        # 8. Deterministic output & atomic aggregation
        ev_det, _ = build_evidence(args, metrics, notes, info, man, prof_stub)
        ev_det2, _ = build_evidence(args, metrics, notes, info, man, prof_stub)
        assert canonical(ev_det) == canonical(ev_det2)
        out = os.path.join(tmp, "evidence.json")
        atomic_write_json(out, ev_det)
        with open(out) as f:
            assert json.load(f)["source_fingerprint"] == "0123456789abcdef"
        checks["deterministic_and_atomic"] = {"status": "PASS"}

        # Schema versioning: exact v2 accepted; old/missing/unknown fail closed
        assert require_profile_schema({"version": 2}) == 2
        for bad in ({}, {"version": 1}, {"version": 3}, {"version": "2"}):
            try:
                build_evidence(args, metrics, notes, info, man, bad)
                raise SystemExit(f"schema {bad.get('version')!r} not rejected")
            except ValueError as e:
                assert "Migrate" in str(e), str(e)
        checks["schema_versioning"] = {"status": "PASS"}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"status": "PASS", "checks": checks}


# ---------------------------------------------------------------- CLI

SCHEMA_HELP = (
    "Evidence schema (--schema prints EVIDENCE_SCHEMA_JSON, $schema 'xkv-evidence/v1'): "
    "effective profile derived SOLELY from observed xkv_profile_info{effective}; "
    "coverage derived ONLY from explicit covered baseline bytes; "
    "synthetic flag propagated from quality sidecars; "
    "underivable keys are 'not_evaluated' (fail-closed)."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect runtime /metrics evidence into compression-gate keys. " + SCHEMA_HELP)
    p.add_argument("--metrics-after", default="",
                   help="Candidate /metrics snapshot (Prometheus text or JSON)")
    p.add_argument("--metrics-before", default="",
                   help="Optional pre-run snapshot (delta diagnostics)")
    p.add_argument("--metrics-baseline", default="",
                   help="Optional baseline-profile snapshot for peak ratio")
    p.add_argument("--baseline-peak-bytes", type=float, default=None,
                   help="Baseline device peak bytes (alternative to --metrics-baseline)")
    p.add_argument("--manifest", default="",
                   help="Matrix-run manifest.json (hashes/fingerprints)")
    p.add_argument("--profile", default="docs/xkv/profiles/ornith-r2.json")
    p.add_argument("--baseline", default="docs/xkv/baseline.json")
    p.add_argument("--target-profile", default="E1")
    p.add_argument("--quality", default="",
                   help="Optional quality/compat sidecar JSON (bool verdicts)")
    p.add_argument("--out", default="",
                   help="Output evidence JSON path (default: stdout only)")
    p.add_argument("--dry-run", action="store_true",
                   help="Emit synthetic fixture evidence; no gate-acceptable output")
    p.add_argument("--self-test", action="store_true",
                   help="Run in-file self-test on synthetic temp fixtures only")
    p.add_argument("--schema", action="store_true",
                   help="Print EVIDENCE_SCHEMA_JSON and exit")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.schema:
        print(json.dumps(EVIDENCE_SCHEMA_JSON, indent=2, sort_keys=True))
        return
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return

    with open(args.profile, "r", encoding="utf-8") as f:
        profile_data = json.load(f)

    manifest: Dict[str, Any] = {}
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    if not args.target_profile and manifest.get("profile"):
        args.target_profile = manifest["profile"]

    quality: Dict[str, Any] = {}
    if args.quality and os.path.exists(args.quality):
        with open(args.quality, "r", encoding="utf-8") as f:
            quality = json.load(f)

    if args.dry_run:
        tmp = tempfile.mkdtemp(prefix="collect-dryrun-")
        try:
            mp = os.path.join(tmp, "metrics.txt")
            _synthetic_metrics(mp, requested="tq-factors-landmarks",
                                effective="tq-factors-landmarks")
            metrics, notes, info = load_snapshot(mp)
            manifest = {"profile": args.target_profile or "E1"}
            ev, _ = build_evidence(args, metrics, notes, info, manifest, profile_data, quality)
            print(json.dumps(ev, indent=2, sort_keys=True))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        return

    metrics, notes, info = load_snapshot(args.metrics_after)
    if args.metrics_before:
        before, bnotes, _ = load_snapshot(args.metrics_before)
        notes = bnotes + notes
        deltas = delta_metrics(before, metrics)
        metrics = dict(metrics)
        metrics["_deltas"] = {k: v for k, v in deltas.items()}

    try:
        evidence, reasons = build_evidence(args, metrics, notes, info, manifest,
                                            profile_data, quality)
    except ValueError as e:
        evidence = {
            "__synthetic_test_fixture__": False,
            "note": "REAL schema-version failure; not_evaluated, never synthetic",
            "xkv_requested_profile": args.target_profile,
            "xkv_effective_profile": "not_evaluated",
            "collector_reasons": [str(e)],
        }
        for k in EVIDENCE_SCHEMA_JSON["keys"]:
            evidence.setdefault(k, "not_evaluated")
        if args.out:
            atomic_write_json(args.out, evidence)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        sys.exit(1)
    if args.out:
        atomic_write_json(args.out, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))

    needed = ("xkv_factored_baseline_byte_coverage", "xkv_net_saved_fraction",
              "xkv_peak_vs_baseline_ratio", "xkv_live_bytes",
              "calibration_sha256", "source_fingerprint", "profile_fingerprint",
              "xkv_effective_profile")
    if any(evidence.get(k, "not_evaluated") == "not_evaluated" for k in needed):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
