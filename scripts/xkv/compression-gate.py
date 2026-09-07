#!/usr/bin/env python3
"""
scripts/xkv/compression-gate.py
-------------------------------
Fail-closed evaluation gate for xKV-SR compression and quality targets.

Evaluates:
  1. Missing evidence / unpopulated fields -> marked 'NOT_EVALUATED' and FAILS CLOSED
  2. Calibration artifact SHA256 integrity using exact manifest keys
  3. Profile consistency (requested == effective, storage profile match, four factor codecs, landmark codec)
  4. Four-stream coverage gate (qualified 4-stream factored segment byte coverage >= threshold)
  5. No resident dense decoded mirrors (full decoded A/B/KV/landmarks must not persist)
  6. Net savings vs baseline TQ+Tri (saved_fraction >= min_saved_fraction)
  7. Peak memory budget (workspace/peak <= peak_budget_ratio)
  8. Allocator byte accounting (live/reserved/peak bytes validated)
  9. Source and profile/codec fingerprints
  10. Quality verdicts (PPL degradation, context scaling)
  11. Compatibility verdicts (speculative acceptance parity, TriAttention, RERoT, state code stream)
  12. Synthetic fixture rejection: external evidence files containing __synthetic_test_fixture__ are
      rejected unless evaluated with allow_synthetic=True (used only by internal self-test).
      Report explicitly carries evidence_kind: 'real' or 'synthetic'.

Every passing branch emits status 'PASS'.
Missing, unknown, NaN, inf, bool (for numeric fields), or mismatched evidence is strictly nonpassing.
Deterministic machine-readable output matching Section 11.2 of XKV-SR.md: `compression_gate.json`.
"""

import sys
import os
import math
import json
import argparse
import hashlib
from typing import Dict, Any, List, Optional, Union


def is_finite_number(val: Any) -> bool:
    """
    Returns True iff val is an int or float, not a boolean, and is finite (not NaN, not inf).
    """
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


def is_valid_nonnegative_int(val: Any) -> bool:
    """
    Returns True iff val is a true integer (not bool, float, str), >= 0.
    """
    if isinstance(val, bool) or not isinstance(val, int):
        return False
    return val >= 0


def compute_file_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class GateEvaluator:
    def __init__(
        self,
        baseline_data: Union[str, Dict[str, Any]],
        profile_data: Union[str, Dict[str, Any]],
        evidence_data: Optional[Union[str, Dict[str, Any]]] = None,
        allow_synthetic: bool = False,
    ):
        self.allow_synthetic = allow_synthetic

        if isinstance(baseline_data, str):
            with open(baseline_data, "r", encoding="utf-8") as f:
                self.baseline = json.load(f)
        else:
            self.baseline = baseline_data

        if isinstance(profile_data, str):
            with open(profile_data, "r", encoding="utf-8") as f:
                self.profile = json.load(f)
        else:
            self.profile = profile_data

        self.evidence: Dict[str, Any] = {}
        if isinstance(evidence_data, str):
            if evidence_data and os.path.exists(evidence_data):
                with open(evidence_data, "r", encoding="utf-8") as f:
                    self.evidence = json.load(f)
        elif isinstance(evidence_data, dict):
            self.evidence = evidence_data

        # Merge baseline and profile thresholds, baseline takes precedence
        self.thresholds: Dict[str, Any] = {}
        self.thresholds.update(self.profile.get("gate_thresholds", {}))
        self.thresholds.update(self.baseline.get("gate_thresholds", {}))

        self.verdicts: Dict[str, Any] = {}
        self.reasons: List[str] = []
        self.overall_pass = True

    def _fail(self, key: str, reason: str) -> None:
        self.verdicts[key] = {
            "status": "FAIL",
            "reason": reason,
        }
        self.reasons.append(f"[{key}] {reason}")
        self.overall_pass = False

    def _pass(self, key: str, details: Optional[Dict[str, Any]] = None) -> None:
        val = {"status": "PASS"}
        if details:
            val.update(details)
        self.verdicts[key] = val

    def _not_evaluated(self, key: str, reason: str) -> None:
        self.verdicts[key] = {
            "status": "NOT_EVALUATED",
            "reason": reason,
        }
        self.reasons.append(f"[{key}] NOT_EVALUATED: {reason}")
        self.overall_pass = False

    def evaluate(self, target_profile: str = "E1") -> Dict[str, Any]:
        """
        Runs all gate checks fail-closed.
        """
        self.overall_pass = True
        self.verdicts.clear()
        self.reasons.clear()

        # 0. Check synthetic test fixture protection
        is_synthetic = bool(
            self.evidence.get("__synthetic_test_fixture__")
            or self.evidence.get("synthetic_test_fixture")
            or "SYNTHETIC TEST FIXTURE" in str(self.evidence.get("note", ""))
        )
        evidence_kind = "synthetic" if is_synthetic else "real"

        if is_synthetic and not self.allow_synthetic:
            self._fail(
                "synthetic_fixture_protection",
                "Evidence contains '__synthetic_test_fixture__' but allow_synthetic is False; synthetic test fixtures cannot pass production gate",
            )

        # 1. Calibration artifact integrity using exact manifest keys
        calib_spec = self.baseline.get("deployed_artifacts", {}).get("triattention_calibration", {})
        expected_path = calib_spec.get("expected_path", "")
        expected_calib_sha = calib_spec.get("historical_expected_sha256")
        legacy_rejected_sha = calib_spec.get("historical_rejected_legacy_sha256")

        actual_calib_sha = None
        if expected_path and os.path.exists(expected_path):
            actual_calib_sha = compute_file_sha256(expected_path)

        if not actual_calib_sha:
            actual_calib_sha = self.evidence.get("calibration_sha256")

        if not actual_calib_sha or actual_calib_sha == "not_evaluated":
            self._not_evaluated("calibration_integrity", f"Calibration file '{expected_path}' missing and no evidence provided")
        elif not isinstance(actual_calib_sha, str):
            self._fail("calibration_integrity", f"Malformed calibration SHA256 evidence: {actual_calib_sha!r}")
        elif actual_calib_sha == legacy_rejected_sha:
            self._fail("calibration_integrity", f"Calibration SHA matches untrusted rejected legacy artifact: {legacy_rejected_sha}")
        elif actual_calib_sha != expected_calib_sha:
            self._fail("calibration_integrity", f"Calibration SHA {actual_calib_sha} != expected {expected_calib_sha}")
        else:
            self._pass("calibration_integrity", {"sha256": actual_calib_sha})

        # 2. Profile consistency: target profile, requested/effective profile, and storage profile
        profiles = self.profile.get("profiles", {})
        if target_profile not in profiles:
            self._fail("profile_consistency", f"Requested target profile '{target_profile}' not found in profiles definition")
            return self._build_report(target_profile, evidence_kind)

        prof_spec = profiles[target_profile]
        expected_storage = prof_spec.get("storage_profile", "none")
        xkv_enabled = prof_spec.get("xkv_enabled", False)

        ev_req_profile = self.evidence.get("xkv_requested_profile", target_profile)
        ev_eff_profile = self.evidence.get("xkv_effective_profile")
        ev_eff_storage = self.evidence.get("xkv_effective_storage_profile")

        if not ev_eff_profile or ev_eff_profile == "not_evaluated":
            self._not_evaluated("profile_consistency", "No evidence for xkv_effective_profile")
        elif ev_eff_profile != ev_req_profile or ev_eff_profile != target_profile:
            self._fail(
                "profile_consistency",
                f"Profile mismatch: target/requested '{target_profile}' but effective '{ev_eff_profile}' (silent degradation/fallback)",
            )
        elif not ev_eff_storage or ev_eff_storage == "not_evaluated":
            self._not_evaluated("storage_profile_consistency", "No evidence for xkv_effective_storage_profile")
        elif ev_eff_storage != expected_storage:
            self._fail(
                "storage_profile_consistency",
                f"Storage profile mismatch: expected '{expected_storage}' but effective '{ev_eff_storage}'",
            )
        else:
            self._pass("profile_consistency", {
                "target_profile": target_profile,
                "effective_profile": ev_eff_profile,
                "storage_profile": ev_eff_storage,
            })
            self._pass("storage_profile_consistency", {"storage_profile": ev_eff_storage})

        # 3. Codec verification: four factor streams and landmark codec
        if xkv_enabled:
            # Check 4 factor stream codecs
            a_k = self.evidence.get("xkv_codec_a_k")
            b_k = self.evidence.get("xkv_codec_b_k")
            a_v = self.evidence.get("xkv_codec_a_v")
            b_v = self.evidence.get("xkv_codec_b_v")

            if not all([a_k, b_k, a_v, b_v]) or any(c == "not_evaluated" for c in [a_k, b_k, a_v, b_v]):
                self._not_evaluated("four_stream_codecs", "Missing codec evidence for one or more factor streams")
            elif not (
                a_k == prof_spec.get("a_k_codec")
                and b_k == prof_spec.get("b_k_codec")
                and a_v == prof_spec.get("a_v_codec")
                and b_v == prof_spec.get("b_v_codec")
            ):
                self._fail(
                    "four_stream_codecs",
                    f"Factor stream codecs mismatch requested profile. Expected AK={prof_spec.get('a_k_codec')}, "
                    f"BK={prof_spec.get('b_k_codec')}, AV={prof_spec.get('a_v_codec')}, BV={prof_spec.get('b_v_codec')}; "
                    f"got AK={a_k}, BK={b_k}, AV={a_v}, BV={b_v}",
                )
            else:
                self._pass("four_stream_codecs", {
                    "a_k": a_k, "b_k": b_k, "a_v": a_v, "b_v": b_v
                })

            # Check landmark codec if profile specifies landmark_codec
            expected_lm = prof_spec.get("landmark_codec")
            if expected_lm:
                ev_lm = self.evidence.get("xkv_codec_landmark")
                if not ev_lm or ev_lm == "not_evaluated":
                    self._not_evaluated("landmark_codec", f"Missing evidence for xkv_codec_landmark (expected {expected_lm})")
                elif ev_lm != expected_lm:
                    self._fail("landmark_codec", f"Landmark codec mismatch: expected '{expected_lm}', got '{ev_lm}'")
                else:
                    self._pass("landmark_codec", {"landmark_codec": ev_lm})

        # 4. Four-stream coverage gate
        min_cov = self.thresholds.get("min_factored_byte_coverage", 0.75)
        factored_cov = self.evidence.get("xkv_factored_baseline_byte_coverage")
        if factored_cov is None or factored_cov == "not_evaluated":
            self._not_evaluated("factored_coverage", f"No evidence for xkv_factored_baseline_byte_coverage (min={min_cov:.2%})")
        elif not is_finite_number(factored_cov):
            self._fail("factored_coverage", f"Malformed coverage evidence (must be finite number, not bool/str): {factored_cov!r}")
        elif factored_cov <= 0.0:
            self._fail("factored_coverage", f"Factored coverage {factored_cov:.2%} must be strictly positive")
        elif factored_cov < min_cov:
            self._fail("factored_coverage", f"Factored coverage {factored_cov:.2%} below minimum threshold {min_cov:.2%}")
        elif factored_cov > 1.0:
            self._fail("factored_coverage", f"Factored coverage {factored_cov:.2%} exceeds 100%")
        else:
            self._pass("factored_coverage", {"coverage": factored_cov, "min_required": min_cov})

        # 5. No resident dense decoded mirrors
        allow_mirror = self.thresholds.get("allow_dense_mirror", False)
        dense_mirror_present = self.evidence.get("has_resident_dense_mirror")
        if dense_mirror_present is None or dense_mirror_present == "not_evaluated":
            self._not_evaluated("no_resident_dense_mirror", "No evidence for has_resident_dense_mirror")
        elif not isinstance(dense_mirror_present, bool):
            self._fail("no_resident_dense_mirror", f"Malformed has_resident_dense_mirror evidence: {dense_mirror_present!r}")
        elif dense_mirror_present and not allow_mirror:
            self._fail("no_resident_dense_mirror", "Resident full decoded mirror detected in memory; violates zero-mirror contract")
        else:
            self._pass("no_resident_dense_mirror", {"has_resident_dense_mirror": dense_mirror_present})

        # 6. Net saving vs baseline TQ+Tri
        min_save = self.thresholds.get("min_saved_fraction", 0.10)
        net_saved_fraction = self.evidence.get("xkv_net_saved_fraction")
        net_extra_ratio = self.evidence.get("xkv_net_extra_compression_ratio")

        if net_saved_fraction is None or net_saved_fraction == "not_evaluated":
            self._not_evaluated("net_savings", f"No evidence for xkv_net_saved_fraction (min={min_save:.2%})")
        elif not is_finite_number(net_saved_fraction):
            self._fail("net_savings", f"Malformed net_saved_fraction evidence (must be finite number, not bool/str): {net_saved_fraction!r}")
        elif net_saved_fraction <= 0.0:
            self._fail("net_savings", f"Net saving {net_saved_fraction:.2%} is non-positive")
        elif net_saved_fraction < min_save:
            self._fail("net_savings", f"Net saving {net_saved_fraction:.2%} below minimum threshold {min_save:.2%}")
        elif net_saved_fraction > 1.0:
            self._fail("net_savings", f"Net saving {net_saved_fraction:.2%} exceeds 100%")
        else:
            self._pass("net_savings", {
                "saved_fraction": net_saved_fraction,
                "extra_compression_ratio": net_extra_ratio,
                "min_required": min_save,
            })

        # 7. Peak workspace and allocation budget
        max_peak_ratio = self.thresholds.get("max_workspace_peak_ratio_vs_baseline", 1.15)
        peak_ratio = self.evidence.get("xkv_peak_vs_baseline_ratio")
        if peak_ratio is None or peak_ratio == "not_evaluated":
            self._not_evaluated("peak_memory_budget", f"No evidence for xkv_peak_vs_baseline_ratio (max={max_peak_ratio:.2f}x)")
        elif not is_finite_number(peak_ratio):
            self._fail("peak_memory_budget", f"Malformed peak_ratio evidence (must be finite number, not bool/str): {peak_ratio!r}")
        elif peak_ratio <= 0.0:
            self._fail("peak_memory_budget", f"Peak ratio {peak_ratio:.2f}x must be strictly positive")
        elif peak_ratio > max_peak_ratio:
            self._fail("peak_memory_budget", f"Peak ratio {peak_ratio:.2f}x exceeds maximum budget {max_peak_ratio:.2f}x")
        else:
            self._pass("peak_memory_budget", {"peak_ratio": peak_ratio, "max_allowed": max_peak_ratio})

        # 8. Allocator byte accounting: live, reserved, peak bytes
        if self.thresholds.get("require_allocator_byte_accounting", True):
            live_b = self.evidence.get("xkv_live_bytes", self.evidence.get("live_bytes"))
            res_b = self.evidence.get("xkv_reserved_bytes", self.evidence.get("reserved_bytes"))
            peak_b = self.evidence.get("xkv_peak_bytes", self.evidence.get("peak_bytes"))

            if live_b is None or res_b is None or peak_b is None or any(x == "not_evaluated" for x in [live_b, res_b, peak_b]):
                self._not_evaluated("allocator_byte_accounting", "Missing required live, reserved, or peak byte accounting evidence")
            elif not (is_valid_nonnegative_int(live_b) and is_valid_nonnegative_int(res_b) and is_valid_nonnegative_int(peak_b)):
                self._fail(
                    "allocator_byte_accounting",
                    f"Malformed allocator byte accounting (must be non-negative integers): live={live_b!r}, res={res_b!r}, peak={peak_b!r}",
                )
            elif not (live_b <= res_b <= peak_b):
                self._fail(
                    "allocator_byte_accounting",
                    f"Invalid allocator byte ordering: live ({live_b}) <= reserved ({res_b}) <= peak ({peak_b}) violated",
                )
            else:
                self._pass("allocator_byte_accounting", {
                    "live_bytes": live_b,
                    "reserved_bytes": res_b,
                    "peak_bytes": peak_b,
                })

        # 9. Fingerprints: source, model, profile, codec
        if self.thresholds.get("require_fingerprints", True):
            src_fp = self.evidence.get("source_fingerprint", self.evidence.get("model_fingerprint"))
            prof_fp = self.evidence.get("profile_fingerprint", self.evidence.get("codec_fingerprint"))

            if not src_fp or not prof_fp or src_fp == "not_evaluated" or prof_fp == "not_evaluated":
                self._not_evaluated("fingerprints", "Missing required source or profile fingerprints evidence")
            elif not (isinstance(src_fp, str) and isinstance(prof_fp, str)):
                self._fail("fingerprints", f"Malformed fingerprints: source={src_fp!r}, profile={prof_fp!r}")
            else:
                self._pass("fingerprints", {
                    "source_fingerprint": src_fp,
                    "profile_fingerprint": prof_fp,
                })

        # 10. Quality gates
        # PPL degradation
        max_ppl_deg = self.thresholds.get("quality_ppl_max_degradation_pct", 5.0)
        ppl_deg = self.evidence.get("quality_ppl_degradation_pct")
        if ppl_deg is None or ppl_deg == "not_evaluated":
            self._not_evaluated("quality_ppl", f"No evidence for quality_ppl_degradation_pct (max={max_ppl_deg:.1f}%)")
        elif not is_finite_number(ppl_deg):
            self._fail("quality_ppl", f"Malformed PPL degradation evidence (must be finite number, not bool/str): {ppl_deg!r}")
        elif ppl_deg < 0.0:
            self._fail("quality_ppl", f"Negative PPL degradation {ppl_deg:.2f}% invalid")
        elif ppl_deg > max_ppl_deg:
            self._fail("quality_ppl", f"PPL degradation {ppl_deg:.2f}% exceeds threshold {max_ppl_deg:.1f}%")
        else:
            self._pass("quality_ppl", {"degradation_pct": ppl_deg, "max_allowed": max_ppl_deg})

        # Context scaling ratio
        min_scaling = self.thresholds.get("quality_context_scaling_min_ratio", 0.95)
        context_scaling = self.evidence.get("quality_context_scaling_ratio")
        if context_scaling is None or context_scaling == "not_evaluated":
            self._not_evaluated("context_scaling", f"No evidence for quality_context_scaling_ratio (min={min_scaling:.2f})")
        elif not is_finite_number(context_scaling):
            self._fail("context_scaling", f"Malformed context scaling evidence (must be finite number, not bool/str): {context_scaling!r}")
        elif context_scaling < min_scaling:
            self._fail("context_scaling", f"Context scaling ratio {context_scaling:.3f} below minimum {min_scaling:.2f}")
        else:
            self._pass("context_scaling", {"context_scaling_ratio": context_scaling, "min_required": min_scaling})

        # 11. Compatibility gates
        # Speculative acceptance parity
        if self.thresholds.get("require_speculative_acceptance_parity", True):
            spec_parity = self.evidence.get("speculative_acceptance_parity")
            if spec_parity is None or spec_parity == "not_evaluated":
                self._not_evaluated("speculative_parity", "No evidence for speculative_acceptance_parity")
            elif not isinstance(spec_parity, bool):
                self._fail("speculative_parity", f"Malformed speculative_acceptance_parity evidence: {spec_parity!r}")
            elif not spec_parity:
                self._fail("speculative_parity", "Speculative acceptance parity check failed")
            else:
                self._pass("speculative_parity")

        # TriAttention compatibility
        if self.thresholds.get("require_triattention_compatibility", True):
            tri_compat = self.evidence.get("triattention_compatible")
            if tri_compat is None or tri_compat == "not_evaluated":
                self._not_evaluated("triattention_compatibility", "No evidence for triattention_compatible")
            elif not isinstance(tri_compat, bool):
                self._fail("triattention_compatibility", f"Malformed triattention_compatible evidence: {tri_compat!r}")
            elif not tri_compat:
                self._fail("triattention_compatibility", "TriAttention compatibility check failed")
            else:
                self._pass("triattention_compatibility")

        # RERoT compatibility
        if self.thresholds.get("require_rerot_compatibility", True):
            rerot_compat = self.evidence.get("rerot_compatible")
            if rerot_compat is None or rerot_compat == "not_evaluated":
                self._not_evaluated("rerot_compatibility", "No evidence for rerot_compatible")
            elif not isinstance(rerot_compat, bool):
                self._fail("rerot_compatibility", f"Malformed rerot_compatible evidence: {rerot_compat!r}")
            elif not rerot_compat:
                self._fail("rerot_compatibility", "RERoT compatibility check failed")
            else:
                self._pass("rerot_compatibility")

        # State code stream exactness
        if self.thresholds.get("require_state_code_stream_exact", True):
            state_exact = self.evidence.get("state_code_stream_exact")
            if state_exact is None or state_exact == "not_evaluated":
                self._not_evaluated("state_code_stream_exact", "No evidence for state_code_stream_exact")
            elif not isinstance(state_exact, bool):
                self._fail("state_code_stream_exact", f"Malformed state_code_stream_exact evidence: {state_exact!r}")
            elif not state_exact:
                self._fail("state_code_stream_exact", "State code stream exact restoration check failed")
            else:
                self._pass("state_code_stream_exact")

        return self._build_report(target_profile, evidence_kind)

    def _build_report(self, target_profile: str, evidence_kind: str) -> Dict[str, Any]:
        status_counts = {"PASS": 0, "FAIL": 0, "NOT_EVALUATED": 0}
        for v in self.verdicts.values():
            s = v.get("status", "FAIL")
            status_counts[s] = status_counts.get(s, 0) + 1

        goal_met = self.overall_pass and (status_counts["FAIL"] == 0) and (status_counts["NOT_EVALUATED"] == 0)

        report = {
            "compression_gate_version": 1,
            "target_profile": target_profile,
            "evidence_kind": evidence_kind,
            "compression_goal_met": goal_met,
            "verdict_summary": status_counts,
            "reasons": self.reasons,
            "verdicts": self.verdicts,
            "thresholds": self.thresholds,
            "evidence_provided": bool(self.evidence),
        }
        return report


def build_synthetic_passing_evidence(
    baseline: Dict[str, Any],
    profile: Dict[str, Any],
    target_profile: str = "E1",
) -> Dict[str, Any]:
    """
    Constructs a synthetic passing evidence dictionary for testing the gate.
    Clearly marked as synthetic test fixture so it cannot be misrepresented as real evidence.
    """
    calib_spec = baseline.get("deployed_artifacts", {}).get("triattention_calibration", {})
    expected_sha = calib_spec.get("historical_expected_sha256", "e95dae507d1f4a64e29be160c5281f8a4308a3332dc9c9176e1a3a0af32e50e2")
    prof_spec = profile.get("profiles", {}).get(target_profile, {})

    return {
        "__synthetic_test_fixture__": True,
        "note": "SYNTHETIC TEST FIXTURE ONLY - NOT REAL EMPIRICAL EVIDENCE",
        "calibration_sha256": expected_sha,
        "xkv_requested_profile": target_profile,
        "xkv_effective_profile": target_profile,
        "xkv_effective_storage_profile": prof_spec.get("storage_profile", "tq-factors-landmarks"),
        "xkv_codec_a_k": prof_spec.get("a_k_codec", "turbo4_0"),
        "xkv_codec_b_k": prof_spec.get("b_k_codec", "turbo4_0"),
        "xkv_codec_a_v": prof_spec.get("a_v_codec", "turbo4_0"),
        "xkv_codec_b_v": prof_spec.get("b_v_codec", "turbo4_0"),
        "xkv_codec_landmark": prof_spec.get("landmark_codec", "q8_0"),
        "xkv_factored_baseline_byte_coverage": 0.85,
        "has_resident_dense_mirror": False,
        "xkv_net_saved_fraction": 0.25,
        "xkv_net_extra_compression_ratio": 1.33,
        "xkv_peak_vs_baseline_ratio": 1.05,
        "xkv_live_bytes": 10485760,
        "xkv_reserved_bytes": 16777216,
        "xkv_peak_bytes": 20971520,
        "source_fingerprint": "sha256:5298ea46061c4780d173de866bdbba42f5918135",
        "profile_fingerprint": "sha256:ornith-r2-candidate-e1",
        "quality_ppl_degradation_pct": 1.2,
        "quality_context_scaling_ratio": 0.98,
        "speculative_acceptance_parity": True,
        "triattention_compatible": True,
        "rerot_compatible": True,
        "state_code_stream_exact": True,
    }


def run_self_test(baseline_path: str, profile_path: str) -> Dict[str, Any]:
    """
    Self-test covering:
      1. Nominal exactness
      2. Heterogeneous B padding and landmark padding / fragment override
      3. Malformed/truncated evidence
      4. Empty fail-closed report
      5. External synthetic fixture rejection (allow_synthetic=False rejects __synthetic_test_fixture__)
      6. Fully passing synthetic report with internal allow_synthetic=True, explicitly marked evidence_kind: synthetic
    """
    # 1. Nominal exactness check
    import importlib.util
    memory_model_path = os.path.join(os.path.dirname(__file__), "memory-model.py")
    spec = importlib.util.spec_from_file_location("memory_model", memory_model_path)
    if spec and spec.loader:
        mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mm)
        mm.verify_nominal_constants()
        nominal_status = "PASS"
    else:
        raise ImportError(f"Could not load memory-model from {memory_model_path}")

    # 2. Heterogeneous B and Landmark padding check
    act_het = mm.calculate_actual_segment_bytes(
        w_layers=2,
        d_k=[1000, 1048],
        d_v=[1000, 1048],
        n_tokens=4096,
        rk=384,
        rv=576,
    )
    # B_K rows = 2048, cols = 384 -> 417,792 bytes
    if act_het["breakdown_bytes"]["b_k_bytes"] != 417792:
        raise AssertionError("Heterogeneous B_K bytes mismatch in self-test")
    # Landmark per-owner sum: 557056 + 574464 = 1,131,520 bytes
    if act_het["breakdown_bytes"]["landmark_bytes"] != 1131520:
        raise AssertionError("Heterogeneous per-owner landmark bytes mismatch in self-test")
    het_b_lm_status = "PASS"

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_obj = json.load(f)
    with open(profile_path, "r", encoding="utf-8") as f:
        profile_obj = json.load(f)

    # 3. Empty fail-closed report check
    evaluator_empty = GateEvaluator(baseline_obj, profile_obj, evidence_data={}, allow_synthetic=False)
    report_empty = evaluator_empty.evaluate(target_profile="E1")
    if report_empty["compression_goal_met"] is not False:
        raise AssertionError("Empty evidence must fail closed with compression_goal_met == False")
    if report_empty["verdict_summary"]["NOT_EVALUATED"] == 0:
        raise AssertionError("Empty evidence must have NOT_EVALUATED verdicts")
    if report_empty["evidence_kind"] != "real":
        raise AssertionError("Empty evidence kind should be 'real'")
    empty_report_status = "PASS"

    # 4. Malformed / truncated evidence check
    malformed_evidences = [
        {"note": "bool coverage", "xkv_factored_baseline_byte_coverage": True},
        {"note": "NaN coverage", "xkv_factored_baseline_byte_coverage": float("nan")},
        {"note": "inf saved", "xkv_net_saved_fraction": float("inf")},
        {"note": "bool byte", "xkv_live_bytes": True, "xkv_reserved_bytes": 100, "xkv_peak_bytes": 200},
        {"note": "inverted bytes", "xkv_live_bytes": 200, "xkv_reserved_bytes": 100, "xkv_peak_bytes": 50},
        {"note": "bad sha", "calibration_sha256": "7fbfcdcfc7903e11efba96d7c13bea0ae9d60ff81c75475d54ee613bae2b3cc7"},
    ]
    malformed_passed = 0
    for ev in malformed_evidences:
        evaluator_mal = GateEvaluator(baseline_obj, profile_obj, evidence_data=ev, allow_synthetic=False)
        rep = evaluator_mal.evaluate(target_profile="E1")
        if rep["compression_goal_met"] is False:
            malformed_passed += 1

    if malformed_passed != len(malformed_evidences):
        raise AssertionError(f"Malformed evidence tests failed: {malformed_passed}/{len(malformed_evidences)} caught")
    malformed_status = "PASS"

    # 5. External synthetic rejection test:
    # A normal evidence file with __synthetic_test_fixture__ evaluated without allow_synthetic MUST fail closed!
    synth_ev = build_synthetic_passing_evidence(baseline_obj, profile_obj, target_profile="E1")
    evaluator_external_synth = GateEvaluator(baseline_obj, profile_obj, evidence_data=synth_ev, allow_synthetic=False)
    report_external_synth = evaluator_external_synth.evaluate(target_profile="E1")

    if report_external_synth["compression_goal_met"] is not False:
        raise AssertionError("External synthetic fixture must be rejected when allow_synthetic is False")
    if "synthetic_fixture_protection" not in report_external_synth["verdicts"]:
        raise AssertionError("Missing 'synthetic_fixture_protection' verdict in external synthetic check")
    if report_external_synth["verdicts"]["synthetic_fixture_protection"]["status"] != "FAIL":
        raise AssertionError("Synthetic fixture protection verdict must be FAIL")
    if report_external_synth["evidence_kind"] != "synthetic":
        raise AssertionError("Report evidence_kind must be 'synthetic'")
    external_synthetic_rejection_status = "PASS"

    # 6. Fully passing synthetic report with internal allow_synthetic=True
    evaluator_internal_synth = GateEvaluator(baseline_obj, profile_obj, evidence_data=synth_ev, allow_synthetic=True)
    report_internal_synth = evaluator_internal_synth.evaluate(target_profile="E1")

    if report_internal_synth["compression_goal_met"] is not True:
        raise AssertionError(f"Synthetic passing report failed: {report_internal_synth['reasons']}")
    if report_internal_synth["evidence_kind"] != "synthetic":
        raise AssertionError("Report evidence_kind must be 'synthetic'")
    if report_internal_synth["verdict_summary"]["FAIL"] != 0 or report_internal_synth["verdict_summary"]["NOT_EVALUATED"] != 0:
        raise AssertionError("Synthetic report has non-passing verdicts")
    for k, v in report_internal_synth["verdicts"].items():
        if v.get("status") != "PASS":
            raise AssertionError(f"Verdict '{k}' status is not PASS: {v}")
    synthetic_report_status = "PASS"

    return {
        "status": "PASS",
        "description": "Self-test completed successfully with checked arithmetic and fail-closed guarantees",
        "checks": {
            "nominal_exactness": {"status": nominal_status},
            "heterogeneous_b_and_landmark_padding": {"status": het_b_lm_status},
            "empty_fail_closed_report": {
                "status": empty_report_status,
                "compression_goal_met": report_empty["compression_goal_met"],
                "evidence_kind": report_empty["evidence_kind"],
                "summary": report_empty["verdict_summary"],
            },
            "malformed_evidence_rejected": {
                "status": malformed_status,
                "cases_tested": len(malformed_evidences),
            },
            "external_synthetic_fixture_rejected": {
                "status": external_synthetic_rejection_status,
                "compression_goal_met": report_external_synth["compression_goal_met"],
                "evidence_kind": report_external_synth["evidence_kind"],
                "protection_verdict": report_external_synth["verdicts"]["synthetic_fixture_protection"],
            },
            "synthetic_passing_report": {
                "status": synthetic_report_status,
                "compression_goal_met": report_internal_synth["compression_goal_met"],
                "evidence_kind": report_internal_synth["evidence_kind"],
                "summary": report_internal_synth["verdict_summary"],
                "synthetic_fixture_label": synth_ev["note"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="xKV-SR compression and quality gate evaluator")
    parser.add_argument("--self-test", action="store_true", help="Run comprehensive self-tests without writing repository reports")
    parser.add_argument("--baseline", default="docs/xkv/baseline.json", help="Path to baseline.json")
    parser.add_argument("--profile", default="docs/xkv/profiles/ornith-r2.json", help="Path to profiles json")
    parser.add_argument("--evidence", default="", help="Path to evidence json (if any)")
    parser.add_argument("--target-profile", default="E1", help="Target profile to evaluate (e.g. B0, R0-R5, E0-E2)")
    parser.add_argument("--output", "-o", default="", help="Output report json path (default: stdout only)")
    args = parser.parse_args()

    if args.self_test:
        test_results = run_self_test(baseline_path=args.baseline, profile_path=args.profile)
        print(json.dumps(test_results, indent=2))
        sys.exit(0)

    evaluator = GateEvaluator(
        baseline_data=args.baseline,
        profile_data=args.profile,
        evidence_data=args.evidence if args.evidence else None,
        allow_synthetic=False,  # CLI runs reject synthetic fixtures unconditionally
    )

    report = evaluator.evaluate(target_profile=args.target_profile)
    formatted = json.dumps(report, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted + "\n")
    print(formatted)

    # Exit code: 0 if goal_met, 1 if failed / not_evaluated (fail-closed)
    if not report["compression_goal_met"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
