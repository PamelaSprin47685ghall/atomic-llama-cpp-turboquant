#!/usr/bin/env python3
"""
scripts/xkv/run-evaluation-matrix.py
------------------------------------
Resumable B0/R0-R5/E0-E2 evaluation matrix runner for xKV-SR release gates.

Reads one profile JSON (docs/xkv/profiles/ornith-r2.json), runs a fixed prompt
corpus (JSONL: id/category/prompt/reference/scorer) against paired profiles
with identical model/template/seed/sampling/output budget, and preserves every
per-item output and score.

Backends (no shell; subprocess argv arrays only, or stdlib HTTP):
  --backend server : POSTs to llama-server /completion (streamed SSE).
                     Accepts completion stream either with [DONE] OR with
                     exactly one terminal finish event (finish_reason/stop).
  --backend cli    : builds exact CLI argv from profile_spec (--xkv,
                     --xkv-storage-profile, --xkv-factor-a-k, etc.) and runs per prompt.
  --backend ppl    : runs a perplexity argv template per prompt.

Layout per matrix run (out-dir):
  plan.json            deterministic plan (no timestamps in hash)
  <profile>/manifest.json   immutable per-run manifest
  <profile>/results.jsonl   one JSON record per prompt (append/resume)
  <profile>/raw/<id>.stdout / .stderr / .sse
  evidence.json        aggregate evidence with compression-gate keys
                       (underivable keys are "not_evaluated", fail-closed)

Missing prompts/scorers/metrics, changed sampling, mismatched binary/model/
calibration hashes, or incomplete runs aggregate to NOT_EVALUATED, never to a
synthetic pass. --dry-run / --self-test fixtures carry
"__synthetic_test_fixture__": true so the production compression gate
(allow_synthetic=False) rejects them unconditionally.

Evidence schema: --schema prints EVIDENCE_SCHEMA_JSON ($schema "xkv-evidence/v1").
Only the Python standard library is used.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

VERSION = 1

# Profile JSON schema version with explicit mode/factorizer/budgets/server
# defaults. The runner validates the EXACT supported version; old/unknown
# versions fail closed with a migration message (never silently accepted).
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


def profile_fingerprint(spec: Dict[str, Any], schema_version: int) -> str:
    """Profile fingerprint covering the schema version and the spec body."""
    return sha256_hex(canonical({"profile_schema_version": schema_version,
                                 "spec": spec}))

MATRIX_PROFILES = ["B0", "R0", "R1", "R2", "R3", "R4", "R5", "E0", "E1", "E2"]

REQUIRED_PROMPT_FIELDS = ("id", "category", "prompt", "scorer")
SUPPORTED_SCORERS = ("exact", "includes", "aime", "math", "ppl", "throughput", "manual")
SUPPORTED_CATEGORIES = (
    "aime24", "aime25", "math500", "needle", "retrieval",
    "multiturn", "repoqa", "production", "ppl", "context_scaling",
    "throughput",
)

EVIDENCE_SCHEMA_JSON = {
    "$schema": "xkv-evidence/v1",
    "description": (
        "Aggregate evidence consumed by scripts/xkv/compression-gate.py. "
        "Keys mirror GateEvaluator evidence fields exactly. "
        "Missing/unmeasured keys MUST be the string 'not_evaluated'."
    ),
    "keys": {
        "xkv_requested_profile": "str profile name under evaluation",
        "xkv_effective_profile": "str effective profile (mismatch => gate FAIL)",
        "xkv_effective_storage_profile": "str storage profile from profile JSON",
        "xkv_codec_a_k": "str",
        "xkv_codec_b_k": "str",
        "xkv_codec_a_v": "str",
        "xkv_codec_b_v": "str",
        "xkv_codec_landmark": "str | 'not_evaluated' for non-xkv profiles",
        "calibration_sha256": "str hex | 'not_evaluated'",
        "quality_ppl_degradation_pct": "float | 'not_evaluated'",
        "quality_context_scaling_ratio": "float | 'not_evaluated'",
        "speculative_acceptance_parity": "bool | 'not_evaluated'",
        "triattention_compatible": "bool | 'not_evaluated'",
        "rerot_compatible": "bool | 'not_evaluated'",
        "state_code_stream_exact": "bool | 'not_evaluated'",
        "matrix_verdict": "PASS | FAIL | NOT_EVALUATED",
        "per_item": "list of per-prompt records (never dropped)",
    },
}


# ---------------------------------------------------------------- utilities

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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


def atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def git_commit(cwd: str = ".") -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=15,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "not_evaluated"


# ---------------------------------------------------------------- corpus

def load_corpus(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not os.path.exists(path):
        return [], [f"prompt corpus missing: {path}"]
    items: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path}:{lineno}: invalid JSON: {e}")
                continue
            if not isinstance(rec, dict):
                errors.append(f"{path}:{lineno}: record is not an object")
                continue
            missing = [k for k in REQUIRED_PROMPT_FIELDS if k not in rec]
            if missing:
                errors.append(f"{path}:{lineno}: missing fields {missing}")
                continue
            if rec["id"] in seen:
                errors.append(f"{path}:{lineno}: duplicate id {rec['id']!r}")
                continue
            seen.add(rec["id"])
            if rec["scorer"] not in SUPPORTED_SCORERS:
                errors.append(
                    f"{path}:{lineno}: unknown scorer {rec['scorer']!r} "
                    f"(supported: {sorted(SUPPORTED_SCORERS)})"
                )
                continue
            if not isinstance(rec["prompt"], str) or not rec["prompt"]:
                errors.append(f"{path}:{lineno}: empty prompt for id {rec['id']!r}")
                continue
            items.append(rec)
    items.sort(key=lambda r: str(r["id"]))
    return items, errors


# ---------------------------------------------------------------- SSE

def parse_sse_stream(raw: bytes) -> Dict[str, Any]:
    """Assemble streamed llama-server SSE.

    Enforces terminal validation:
      - Native llama-server /completion may terminate EITHER with `data: [DONE]`
        followed by/following a finish event, OR without [DONE] iff exactly ONE
        terminal finish event (finish_reason or stop: true) is observed.
      - Exactly one terminal condition must be met:
        (done_count == 1 and finish_count == 1) OR (done_count == 0 and finish_count == 1).
      - Zero terminal events or multiple terminal events are violations.
    """
    text = raw.decode("utf-8", errors="replace")
    chunks: List[str] = []
    finish_reason: Optional[str] = None
    finish_count = 0
    done_count = 0
    prompt_tokens: Optional[int] = None
    predicted_tokens: Optional[int] = None
    timings: Dict[str, Any] = {}
    malformed = 0
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("data:"):
            continue
        payload = s[len("data:"):].strip()
        if payload == "[DONE]":
            done_count += 1
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(obj, dict):
            c = obj.get("content")
            if isinstance(c, str):
                chunks.append(c)
            for ch in obj.get("choices", []) if isinstance(obj.get("choices"), list) else []:
                if isinstance(ch, dict):
                    d = ch.get("delta", {}) if isinstance(ch.get("delta"), dict) else {}
                    t = d.get("content", ch.get("text", ""))
                    if isinstance(t, str):
                        chunks.append(t)
                    fr = ch.get("finish_reason")
                    if fr is not None:
                        finish_reason = str(fr)
                        finish_count += 1
            if "stop" in obj and obj.get("stop") is True:
                finish_count += 1
                if finish_reason is None:
                    finish_reason = "stop"
            for k in ("prompt_tokens", "prompt_n", "n_prompt"):
                if isinstance(obj.get(k), int):
                    prompt_tokens = obj[k]
            for k in ("predicted_tokens", "completion_tokens", "n_predicted"):
                if isinstance(obj.get(k), int):
                    predicted_tokens = obj[k]
            t = obj.get("timings")
            if isinstance(t, dict):
                timings = t

    # Terminal validation logic:
    error = None
    if done_count > 1:
        error = f"sse_terminal_violation: duplicate [DONE] events (count={done_count})"
    elif finish_count > 1:
        error = f"sse_terminal_violation: multiple finish events (count={finish_count})"
    elif finish_count == 0 and done_count == 0:
        error = "sse_terminal_violation: stream truncated without finish event or [DONE]"
    elif finish_count == 0 and done_count == 1:
        error = "sse_terminal_violation: [DONE] received but no finish event was observed"
    # Otherwise: finish_count == 1 (done_count in {0, 1}) is valid.

    return {
        "assembled_content": "".join(chunks),
        "finish_reason": finish_reason,
        "done_count": done_count,
        "finish_count": finish_count,
        "prompt_tokens": prompt_tokens,
        "predicted_tokens": predicted_tokens,
        "timings": timings,
        "malformed_lines": malformed,
        "error": error,
    }


# ---------------------------------------------------------------- scorers

def extract_boxed(text: str) -> Optional[str]:
    """Returns the contents of the LAST \\boxed{...} handling nested braces."""
    tag = "\\boxed{"
    idx = text.rfind(tag)
    if idx < 0:
        return None
    i = idx + len(tag)
    depth = 1
    start = i
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None  # unbalanced: unparseable


FINAL_MARKERS = ("final answer:", "final answer is", "answer:")


def extract_final_answer(output: str) -> Tuple[Optional[str], str]:
    """Deterministic final-answer extraction. Returns (answer, method).

    Priority: last \\boxed{...} > trailing '#### <ans>' (MATH-500) > last
    'Final answer:'/'Answer:' marker line. Returns (None, method) when no
    marker is found or the marker body is empty.
    """
    boxed = extract_boxed(output)
    if boxed is not None and boxed.strip():
        return boxed.strip(), "boxed"
    tail_lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if tail_lines:
        last = tail_lines[-1]
        if last.startswith("####"):
            body = last[4:].strip()
            if body:
                return body, "hash4"
    low = output.lower()
    best = -1
    for marker in FINAL_MARKERS:
        pos = low.rfind(marker)
        if pos > best:
            best = pos
    if best >= 0:
        for marker in FINAL_MARKERS:
            if low[best:best + len(marker)] == marker:
                body = output[best + len(marker):].strip().splitlines()[0].strip()
                if body:
                    return body, "marker:" + marker
                break
    return None, "none"


def normalize_math_text(s: str) -> str:
    """Conservative string normalization for MATH answers (stdlib only).

    Strips LaTeX wrappers ($, \\(...\\)), collapses whitespace. Deep symbolic
    equivalence (\\frac vs /, factor order, etc.) requires the external
    versioned grader; this never claims symbolic equality it cannot check.
    """
    t = s.strip()
    if len(t) >= 2 and t.startswith("$") and t.endswith("$"):
        t = t[1:-1].strip()
    if t.startswith("\\(") and t.endswith("\\)") and len(t) >= 4:
        t = t[2:-2].strip()
    t = " ".join(t.split())
    return t


def run_external_grader(grader_path: str, output_text: str, reference_text: str,
                        timeout: int = 60) -> Dict[str, Any]:
    """Runs the versioned external grader (argv array, no shell).

    Protocol: argv [grader, --output <file>, --reference <file>]; stdout must
    be JSON with a boolean 'passed' field. Anything else (nonzero exit,
    malformed stdout, timeout) returns NOT_EVALUATED, never a pass.
    """
    import tempfile as _tf
    tmp = _tf.mkdtemp(prefix="grader-")
    try:
        op = os.path.join(tmp, "output.txt")
        rp = os.path.join(tmp, "reference.txt")
        with open(op, "w", encoding="utf-8") as f:
            f.write(output_text)
        with open(rp, "w", encoding="utf-8") as f:
            f.write(reference_text)
        try:
            p = subprocess.run([grader_path, "--output", op, "--reference", rp],
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"passed": None, "status": "NOT_EVALUATED",
                    "detail": "grader timeout"}
        except OSError as e:
            return {"passed": None, "status": "NOT_EVALUATED",
                    "detail": f"grader exec failed: {e}"}
        if p.returncode != 0:
            return {"passed": None, "status": "NOT_EVALUATED",
                    "detail": f"grader exit {p.returncode}: {p.stderr.decode('utf-8', errors='replace')[:200]}"}
        try:
            obj = json.loads(p.stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {"passed": None, "status": "NOT_EVALUATED",
                    "detail": "grader stdout not JSON"}
        if not isinstance(obj, dict) or not isinstance(obj.get("passed"), bool):
            return {"passed": None, "status": "NOT_EVALUATED",
                    "detail": "grader JSON lacks boolean 'passed'"}
        return {"passed": obj["passed"],
                "status": "SCORED",
                "detail": f"external grader: passed={obj['passed']}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score_final_answer(scorer: str, output: str, reference: Any,
                       grader: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic AIME/MATH grading on the EXTRACTED final answer only.

    A reference that appears solely in reasoning can never auto-pass: only the
    extracted final answer is compared. Unknown/parse failure => NOT_EVALUATED.
    grader = {path, sha256} of the versioned external grader (MATH symbolic
    equivalence); when present and extraction succeeds it decides, with its
    version hash recorded on every verdict.
    """
    ref_text = reference.get("answer", reference) if isinstance(reference, dict) else reference
    if ref_text is None or (isinstance(ref_text, str) and not ref_text.strip()):
        return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": f"{scorer} scorer requires a reference answer"}
    extracted, method = extract_final_answer(output)
    if extracted is None:
        return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": f"{scorer}: no final answer found (need \\boxed{{}}, ####, or answer marker)",
                "extraction_method": method}
    base: Dict[str, Any] = {"extracted": extracted, "extraction_method": method}
    if scorer == "aime":
        try:
            got = int(normalize_math_text(extracted).replace(",", ""))
            want = int(str(ref_text).strip().replace(",", ""))
        except (ValueError, TypeError):
            base.update({"score": None, "passed": None, "status": "NOT_EVALUATED",
                         "detail": "aime: non-integer final answer or reference"})
            return base
        if not (0 <= got <= 999 and 0 <= want <= 999):
            base.update({"score": None, "passed": None, "status": "NOT_EVALUATED",
                         "detail": f"aime: out of 0-999 range (got={got}, want={want})"})
            return base
        ok = got == want
        base.update({"score": 1.0 if ok else 0.0, "passed": ok, "status": "SCORED",
                     "detail": f"aime final-answer int compare: {got} vs {want}"})
        return base
    # math: normalized compare, optionally decided by the versioned grader.
    norm_out = normalize_math_text(extracted)
    norm_ref = normalize_math_text(str(ref_text))
    base.update({"normalized_output": norm_out, "normalized_reference": norm_ref})
    if grader and grader.get("path"):
        g = run_external_grader(grader["path"], extracted, str(ref_text))
        g["grader_sha256"] = grader.get("sha256")
        g["grader_path"] = grader.get("path")
        base.update(g)
        if g["status"] == "SCORED":
            base["score"] = 1.0 if g["passed"] else 0.0
        else:
            base["score"] = None
        return base
    ok = norm_out == norm_ref
    base.update({"score": 1.0 if ok else 0.0, "passed": ok, "status": "SCORED",
                 "detail": "math normalized string compare (no external grader; "
                           "symbolic equivalence needs versioned grader)"})
    return base


def score_item(scorer: str, output: str, reference: Any,
               grader: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if scorer == "manual":
        return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": "manual judge required; never auto-pass"}
    if scorer == "throughput":
        return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": "throughput hook: see token counts/timings"}
    if scorer in ("aime", "math"):
        return score_final_answer(scorer, output, reference, grader)
    if scorer == "ppl":
        if not isinstance(reference, dict) or "max_ppl" not in reference:
            return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": "ppl scorer requires reference.max_ppl"}
        try:
            ppl = float(reference.get("measured_ppl", "nan"))
            max_ppl = float(reference["max_ppl"])
        except (TypeError, ValueError):
            return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": "ppl scorer requires numeric measured_ppl/max_ppl"}
        import math as _m
        if not (_m.isfinite(ppl) and _m.isfinite(max_ppl)):
            return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": "non-finite ppl value"}
        return {"score": ppl, "passed": bool(ppl <= max_ppl), "status": "SCORED",
                "detail": f"ppl={ppl} max={max_ppl}"}
    if reference is None:
        return {"score": None, "passed": None, "status": "NOT_EVALUATED",
                "detail": f"{scorer} scorer requires a reference"}
    ref = str(reference)
    if scorer == "exact":
        ok = output.strip() == ref.strip()
        return {"score": 1.0 if ok else 0.0, "passed": ok, "status": "SCORED",
                "detail": "normalized exact match"}
    if scorer == "includes":
        ok = ref in output
        return {"score": 1.0 if ok else 0.0, "passed": ok, "status": "SCORED",
                "detail": "reference substring present" if ok else "reference substring absent"}
    return {"score": None, "passed": None, "status": "NOT_EVALUATED",
            "detail": f"unknown scorer {scorer!r}"}


# ---------------------------------------------------------------- backends

def run_server_backend(url: str, prompt: str, sampling: Dict[str, Any],
                       timeout: int) -> Tuple[bytes, bytes, Dict[str, Any]]:
    body = {
        "prompt": prompt,
        "stream": True,
        "seed": sampling["seed"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "n_predict": sampling["n_predict"],
    }
    if sampling.get("template"):
        body["template"] = sampling["template"]
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        wall = time.time() - t0
        return raw, b"", {"http_status": resp.status, "wall_s": wall}
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except Exception:
            raw = b""
        return raw, str(e).encode(), {"http_status": e.code,
                                      "wall_s": time.time() - t0,
                                      "error": f"http_error {e.code}"}
    except Exception as e:
        return b"", repr(e).encode(), {"http_status": None,
                                       "wall_s": time.time() - t0,
                                       "error": f"{type(e).__name__}: {e}"}


def run_cli_backend(argv: List[str], timeout: int) -> Tuple[bytes, bytes, Dict[str, Any]]:
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout)
        return p.stdout, p.stderr, {"returncode": p.returncode,
                                    "wall_s": time.time() - t0}
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, bytes) else b""
        err = e.stderr if isinstance(e.stderr, bytes) else b""
        return out, err + b"\nTIMEOUT", {"returncode": None,
                                         "wall_s": time.time() - t0,
                                         "error": f"timeout after {timeout}s"}
    except OSError as e:
        return b"", repr(e).encode(), {"returncode": None,
                                       "wall_s": time.time() - t0,
                                       "error": f"os_error: {e}"}


def build_cli_argv(binary: str, model: str, prompt: str,
                   sampling: Dict[str, Any], template: Optional[str],
                   profile_spec: Optional[Dict[str, Any]] = None,
                   calibration_path: Optional[str] = None) -> List[str]:
    """Builds each matrix target's exact CLI argv from profile_spec.

    Translates profile definition fields to exact llama-cli flags:
      - xkv_enabled: false => --xkv off (or --xkv-off)
      - mode: explicit --xkv off|shadow|dense|sr (older plans infer from sr_enabled)
      - storage_profile: => --xkv-storage-profile <reference|tq-factors|tq-factors-landmarks>
      - codecs: --xkv-factor-a-k, --xkv-factor-b-k, --xkv-factor-a-v, --xkv-factor-b-v
      - landmark_codec: => --xkv-landmark-type <type>
      - source/balance/refine/workspace/cache/store/saving/coverage: exact matching --xkv-* flags
      - sr mode: => --xkv-sr-budget <N> (default 256)
      - triattention: --triattention --triattention-stats <path> --triattention-ratio <target>
      - rerot: --rerot --rerot-frontier strong
      - mtp: --mtp + optional --spec-draft-n-max
      - k_codec/v_codec: -ctk / -ctv
    """
    argv = [binary, "-m", model, "-p", prompt,
            "--seed", str(sampling["seed"]),
            "--temp", str(sampling["temperature"]),
            "--top-p", str(sampling["top_p"]),
            "--top-k", str(sampling["top_k"]),
            "-n", str(sampling["n_predict"])]
    if template or sampling.get("template"):
        argv += ["--template", str(template or sampling["template"])]

    argv += profile_flags(profile_spec, calibration_path)
    return argv


def profile_flags(profile_spec: Optional[Dict[str, Any]] = None,
                   calibration_path: Optional[str] = None) -> List[str]:
    """Profile portion of llama argv, shared by llama-cli and llama-server launch.

    Every pinned profile field maps to exactly one flag. check_profile_mode
    enforces XKV-SR section 15.5 pinning (R0/R1/R2 dense, R3-R5/E1/E2 SR,
    B0/E0 off); violations raise ValueError so callers fail closed.
    xkv-enabled profiles always pass --xkv-store-mib explicitly (spec value,
    else 0 so common auto-derives/fits and records the effective budget
    instead of silently using the compiled-in 1024 MiB default). MTP profiles
    pin --spec-draft-n-max (spec value, else 2).
    """
    argv: List[str] = []
    if not profile_spec:
        return argv
    check_profile_mode(str(profile_spec.get("name", "?")), profile_spec)

    # Baseline K/V cache types
    if "k_codec" in profile_spec:
        argv += ["-ctk", str(profile_spec["k_codec"])]
    if "v_codec" in profile_spec:
        argv += ["-ctv", str(profile_spec["v_codec"])]

    # TriAttention
    if profile_spec.get("triattention_enabled"):
        argv.append("--triattention")
        if calibration_path:
            argv += ["--triattention-stats", calibration_path]
        target_ratio = profile_spec.get("triattention_target")
        if target_ratio:
            # support fractions like "3/32" or decimal
            if "/" in str(target_ratio):
                num, den = str(target_ratio).split("/", 1)
                ratio_val = float(num) / float(den)
            else:
                ratio_val = float(target_ratio)
            argv += ["--triattention-ratio", f"{ratio_val:.6f}"]

    # RERoT
    if profile_spec.get("rerot_enabled"):
        argv += ["--rerot", "--rerot-frontier", "strong"]

    # MTP
    if profile_spec.get("mtp_enabled"):
        argv.append("--mtp")
        argv += ["--spec-draft-n-max", str(profile_spec.get("spec_draft_n_max", 2))]

    # XKV compression flags
    if not profile_spec.get("xkv_enabled", False):
        argv += ["--xkv", "off"]
        return argv

    # Mode is explicit for shadow-vs-dense ablations; infer only for older plans.
    mode = str(profile_spec.get(
        "mode", "sr" if profile_spec.get("sr_enabled", False) else "dense"))
    argv += ["--xkv", mode]
    if mode == "sr":
        sr_budget = profile_spec.get("sr_budget", 256)
        argv += ["--xkv-sr-budget", str(sr_budget)]

    # Storage profile
    storage_prof = profile_spec.get("storage_profile")
    if storage_prof:
        argv += ["--xkv-storage-profile", str(storage_prof)]

    # Factor stream codecs
    if "a_k_codec" in profile_spec:
        argv += ["--xkv-factor-a-k", str(profile_spec["a_k_codec"])]
    if "b_k_codec" in profile_spec:
        argv += ["--xkv-factor-b-k", str(profile_spec["b_k_codec"])]
    if "a_v_codec" in profile_spec:
        argv += ["--xkv-factor-a-v", str(profile_spec["a_v_codec"])]
    if "b_v_codec" in profile_spec:
        argv += ["--xkv-factor-b-v", str(profile_spec["b_v_codec"])]

    # Landmark codec
    if "landmark_codec" in profile_spec:
        argv += ["--xkv-landmark-type", str(profile_spec["landmark_codec"])]

    # Structural parameters
    if "group_size" in profile_spec:
        argv += ["--xkv-group-size", str(profile_spec["group_size"])]
    if "rank_k" in profile_spec:
        argv += ["--xkv-rank-k", str(profile_spec["rank_k"])]
    if "rank_v" in profile_spec:
        argv += ["--xkv-rank-v", str(profile_spec["rank_v"])]
    if "segment_tokens" in profile_spec:
        argv += ["--xkv-segment-tokens", str(profile_spec["segment_tokens"])]
    if "chunk_tokens" in profile_spec:
        argv += ["--xkv-chunk-tokens", str(profile_spec["chunk_tokens"])]
    if "source" in profile_spec:
        argv += ["--xkv-source", str(profile_spec["source"])]
    if "factor_balance" in profile_spec:
        argv += ["--xkv-factor-balance", str(profile_spec["factor_balance"])]
    if "landmark_refine" in profile_spec:
        argv += ["--xkv-landmark-refine", str(profile_spec["landmark_refine"])]
    if "landmark_refine_max_rows" in profile_spec:
        argv += ["--xkv-landmark-refine-max-rows", str(profile_spec["landmark_refine_max_rows"])]
    if "workspace_mib" in profile_spec:
        argv += ["--xkv-workspace-mib", str(profile_spec["workspace_mib"])]
    if "decode_cache_mib" in profile_spec:
        argv += ["--xkv-decode-cache-mib", str(profile_spec["decode_cache_mib"])]
    argv += ["--xkv-store-mib", str(profile_spec.get("store_mib", 0))]
    if "min_saving" in profile_spec:
        argv += ["--xkv-min-saving", str(profile_spec["min_saving"])]
    if "min_factor_coverage" in profile_spec:
        argv += ["--xkv-min-factor-coverage", str(profile_spec["min_factor_coverage"])]
    if "factorizer" in profile_spec:
        argv += ["--xkv-factorizer", str(profile_spec["factorizer"])]
    if "xkv_seed" in profile_spec:
        argv += ["--xkv-seed", str(profile_spec["xkv_seed"])]

    return argv


def build_template_argv(cli_template: str, mapping: Dict[str, str]) -> List[str]:
    args: List[str] = []
    cur: List[str] = []
    q: Optional[str] = None
    i = 0
    while i < len(cli_template):
        c = cli_template[i]
        if q is not None:
            if c == q:
                q = None
            else:
                cur.append(c)
        elif c in ("'", '"'):
            q = c
        elif c in (" ", "\t"):
            if cur:
                args.append("".join(cur))
                cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        args.append("".join(cur))
    out = []
    for a in args:
        try:
            out.append(a.format(**mapping))
        except KeyError as e:
            raise ValueError(f"cli-template placeholder missing value: {e}")
    return out


# ---------------------------------------------------------------- profiles & servers

EXPECTED_MODES = {
    "B0": "off", "R0": "dense", "R1": "dense", "R2": "dense",
    "R3": "sr", "R4": "sr", "R5": "sr",
    "E0": "off", "E1": "sr", "E2": "sr",
}


def check_profile_mode(name: str, spec: Dict[str, Any]) -> str:
    """Enforces XKV-SR section 15.5 mode pinning. Returns resolved mode.

    R0/R1/R2 must be dense tiled reconstruction, R3-R5/E1/E2 SR, B0/E0 off.
    Raises ValueError so callers fail closed (NOT_EVALUATED, never mislabeled).
    """
    xkv = bool(spec.get("xkv_enabled", False))
    sr = bool(spec.get("sr_enabled", False))
    if "mode" in spec:
        mode = str(spec["mode"])
        if name in EXPECTED_MODES and mode != EXPECTED_MODES[name]:
            raise ValueError(
                f"profile {name}: pinned mode must be {EXPECTED_MODES[name]!r} "
                f"per section 15.5, got {mode!r}")
    else:
        mode = "sr" if sr else ("dense" if xkv else "off")
    if mode == "off" and xkv:
        raise ValueError(f"profile {name}: mode off with xkv_enabled=true")
    if not xkv and mode != "off":
        raise ValueError(f"profile {name}: xkv_enabled=false requires mode off, got {mode!r}")
    if mode == "sr" and not sr:
        raise ValueError(f"profile {name}: mode sr requires sr_enabled=true")
    if mode == "dense" and sr:
        raise ValueError(f"profile {name}: mode dense with sr_enabled=true")
    if "dense_tiled_reconstruction" in spec:
        if mode == "dense" and not spec.get("dense_tiled_reconstruction"):
            raise ValueError(f"profile {name}: mode dense requires dense_tiled_reconstruction=true")
        if mode == "sr" and spec.get("dense_tiled_reconstruction"):
            raise ValueError(f"profile {name}: mode sr requires dense_tiled_reconstruction=false")
    return mode


def parse_server_urls(text: str) -> Dict[str, str]:
    """Parses --server-urls 'P1=URL1,P2=URL2' into a profile->URL mapping."""
    out: Dict[str, str] = {}
    if not text:
        return out
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad --server-urls entry {chunk!r} (want PROFILE=URL)")
        k, v = chunk.split("=", 1)
        k, v = k.strip(), v.strip().rstrip("/")
        if not k or not v:
            raise ValueError(f"bad --server-urls entry {chunk!r} (want PROFILE=URL)")
        if k in out:
            raise ValueError(f"duplicate --server-urls profile {k!r}")
        out[k] = v
    return out


def runtime_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Exact production runtime settings for server launch argv + manifests.

    ctx 262144 / total-kv auto are production-pinned defaults; gpu layers and
    device are NEVER defaulted here (no hardcoded hardware IDs) and are only
    passed when explicitly set.
    """
    return {
        "ctx_size": getattr(args, "server_ctx", 262144),
        "total_kv": getattr(args, "server_total_kv", "auto"),
        "gpu_layers": getattr(args, "gpu_layers", None) or "all",
        "device": getattr(args, "device", None),
        "n_parallel": getattr(args, "server_np", None),
    }


def build_server_argv(server_binary: str, model: str, port: int,
                       profile_spec: Optional[Dict[str, Any]],
                       calibration_path: Optional[str],
                       runtime_cfg: Optional[Dict[str, Any]],
                       extra_args: Optional[List[str]] = None) -> List[str]:
    """Exact llama-server launch argv: runtime cfg + every profile flag, no shell.

    Always enables --metrics (deduped against user --server-arg tokens) since
    preflight/identity require /metrics. -np prefers the per-profile pin
    (E0/E1/E2: 6) over runtime_cfg (--server-np, for R/B ablations).
    """
    cfg = runtime_cfg or {}
    argv = [server_binary, "-m", model,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(cfg.get("ctx_size", 262144)),
            "--total-kv", str(cfg.get("total_kv", "auto"))]
    argv += ["-ngl", str(cfg.get("gpu_layers") or "all")]
    if cfg.get("device"):
        argv += ["-dev", str(cfg["device"])]
    if profile_spec and profile_spec.get("n_parallel") is not None:
        argv += ["-np", str(profile_spec["n_parallel"])]
    elif cfg.get("n_parallel"):
        argv += ["-np", str(cfg["n_parallel"])]
    argv += profile_flags(profile_spec, calibration_path)
    extra = list(extra_args or [])
    if "--metrics" not in extra:
        argv += ["--metrics"]
    argv += extra
    return argv


def _http_get_text(url: str, timeout: int) -> Tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def parse_server_info(text: str) -> Tuple[Dict[str, str], set]:
    """Extracts xkv profile/fingerprint info from /metrics exposition text.

    Strips the llamacpp: namespace uniformly. Returns (info, metric_names)
    where info may hold requested/effective/source_fingerprint/
    profile_fingerprint/codec_fingerprint/backend_fingerprint.
    """
    import re
    info: Dict[str, str] = {}
    names: set = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^(?:llamacpp:)?([A-Za-z0-9_]+)(\{([^}]*)\})?\s+\S+\s*$", s)
        if not m:
            continue
        name, _, label_body = m.groups()
        names.add(name)
        labels: Dict[str, str] = {}
        if label_body:
            for kv in re.findall(r"(\w+)\s*=\s*\"([^\"]*)\"", label_body):
                labels[kv[0]] = kv[1]
        if name == "xkv_profile_info":
            if "requested" in labels:
                info["requested"] = labels["requested"]
            if "effective" in labels:
                info["effective"] = labels["effective"]
        elif name in ("xkv_source_info", "xkv_profile_fingerprint_info",
                       "xkv_codec_info", "xkv_backend_info"):
            key = {"xkv_source_info": "source_fingerprint",
                   "xkv_profile_fingerprint_info": "profile_fingerprint",
                   "xkv_codec_info": "codec_fingerprint",
                   "xkv_backend_info": "backend_fingerprint"}[name]
            if "fingerprint" in labels:
                info[key] = labels["fingerprint"]
        elif name == "xkv_mode":
            if "mode" in labels:
                info["xkv_mode"] = labels["mode"]
        elif name == "xkv_mode_info":
            if "effective_mode" in labels:
                info["xkv_mode"] = labels["effective_mode"]
            if "requested_mode" in labels:
                info["xkv_requested_mode"] = labels["requested_mode"]
        elif name == "xkv_factorizer_info":
            if "factorizer" in labels:
                info["xkv_factorizer"] = labels["factorizer"]
        elif name == "xkv_balance_info":
            if "balance" in labels:
                info["xkv_factor_balance"] = labels["balance"]
        elif name == "xkv_codec_profile_info":
            for c_key in ("a_k", "b_k", "a_v", "b_v", "landmark"):
                if c_key in labels:
                    info[f"xkv_codec_{c_key}"] = labels[c_key]
            if "factorizer" in labels:
                info["xkv_factorizer"] = labels["factorizer"]
            if "balance" in labels:
                info["xkv_factor_balance"] = labels["balance"]
        elif name == "xkv_codecs_info":
            for c_key in ("a_k", "b_k", "a_v", "b_v", "landmark"):
                if c_key in labels:
                    info[f"xkv_codec_{c_key}"] = labels[c_key]
        elif name == "xkv_seed_info":
            if "seed" in labels:
                info["xkv_seed"] = labels["seed"]
        elif name == "model_artifact_info":
            if "sha256" in labels:
                info["model_sha256"] = labels["sha256"]
        elif name == "tri_calibration_info":
            if "sha256" in labels:
                info["tri_calibration_sha256"] = labels["sha256"]
            if "fingerprint" in labels:
                info["tri_calibration_fingerprint"] = labels["fingerprint"]
        elif name == "tri_config_info":
            if "ratio" in labels:
                info["triattention_ratio"] = labels["ratio"]
            if "recent_window" in labels:
                info["tri_recent_window"] = labels["recent_window"]
        elif name == "rerot_frontier":
            if "mode" in labels:
                info["rerot_frontier"] = labels["mode"]
    return info, names


TRI_RECENT_WINDOW_PIN = 128


def _ratio_to_float(v: Any) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    if "/" in s:
        n, d = s.split("/", 1)
        return float(n) / float(d)
    return float(s)


def binary_supports_flag(binary: str, flag: str, timeout: int = 30) -> bool:
    """Probes `<binary> --help` for a flag (argv array, no shell)."""
    try:
        p = subprocess.run([binary, "--help"], capture_output=True, timeout=timeout)
        out = ((p.stdout or b"").decode("utf-8", errors="replace")
               + (p.stderr or b"").decode("utf-8", errors="replace"))
        return flag in out
    except (OSError, subprocess.SubprocessError):
        return False


def verify_server_identity(url: str, profile: str, spec: Dict[str, Any],
                            expected_model: Optional[str],
                            timeout: int = 30,
                            baseline: Optional[Dict[str, Any]] = None,
                            model_sha: Optional[str] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verifies a running server matches the plan BEFORE any request is sent.

    Server xkv_profile_info requested/effective labels are STORAGE profile
    names (reference|tq-factors|tq-factors-landmarks|none), never matrix labels:
    both are compared to spec.storage_profile and the matrix label is
    recorded separately. Model identity is by artifact SHA-256 checksum
    (props model_sha256 vs local file sha); paths are recorded
    realpath-normalized but never decide. Pinned n_parallel allows runtime
    reduction to recurrent fit but never an increase. XKV mode, Tri
    calibration/ratio/window, RERoT frontier, MTP draft-max, and xkv seed are
    required exactly when the plan enables them; missing/mismatch is a hard
    preflight NOT_EVALUATED. Returns (ok, errors, observed).
    """
    errors: List[str] = []
    observed: Dict[str, Any] = {"url": url}
    base = url.rstrip("/")
    try:
        _, body = _http_get_text(base + "/props", timeout)
        props = json.loads(body)
    except Exception as e:
        return False, [f"{url}: /props unreachable ({e})"], observed
    if not isinstance(props, dict):
        return False, [f"{url}: /props did not return a JSON object"], observed
    observed["model_path"] = props.get("model_path")
    if expected_model:
        observed["model_path_expected"] = expected_model
        if os.path.exists(expected_model):
            observed["model_path_expected_realpath"] = os.path.realpath(expected_model)
    if isinstance(props.get("model_path"), str) and os.path.exists(props["model_path"]):
        observed["model_path_realpath"] = os.path.realpath(props["model_path"])
    srv_sha = props.get("model_sha256") or props.get("xkv_model_sha256")
    observed["model_sha256"] = srv_sha if srv_sha else "not_evaluated"
    if not srv_sha:
        errors.append("server exposes no model artifact SHA-256 (props model_sha256); checksum identity required")
    elif not model_sha:
        errors.append("missing local model checksum to compare against server model_sha256")
    elif str(srv_sha).lower() != str(model_sha).lower():
        errors.append("model artifact SHA-256 mismatch: server != local plan file")
    try:
        _, mtext = _http_get_text(base + "/metrics", timeout)
    except Exception as e:
        return False, errors + [f"{url}: /metrics unreachable ({e})"], observed
    observed["metrics_sha256"] = sha256_hex(mtext.encode("utf-8"))
    info, names = parse_server_info(mtext)
    observed.update(info)
    if not srv_sha:
        srv_sha = info.get("model_sha256")
        if srv_sha:
            observed["model_sha256"] = srv_sha
            # clear earlier missing error if it was added
            errors = [e for e in errors if "no model artifact SHA-256" not in e]
            if not model_sha:
                errors.append("missing local model checksum to compare against server model_sha256")
            elif str(srv_sha).lower() != str(model_sha).lower():
                errors.append("model artifact SHA-256 mismatch: server != local plan file")
    exp_storage = spec.get("storage_profile", "none")
    eff, req = info.get("effective"), info.get("requested")
    observed["matrix_profile"] = profile
    observed["expected_storage_profile"] = exp_storage
    if spec.get("xkv_enabled"):
        if eff != exp_storage:
            errors.append(
                f"effective storage {eff!r} != plan storage {exp_storage!r} "
                f"for matrix {profile!r}: refusing to mislabel this server config")
        if req != exp_storage:
            errors.append(
                f"requested storage {req!r} != plan storage {exp_storage!r} "
                f"for matrix {profile!r}")
    else:
        if eff not in (None, "", "none"):
            errors.append(
                f"foreign effective storage {eff!r} on baseline plan {profile!r}")
    pinned_np = spec.get("n_parallel")
    slots = props.get("total_slots")
    observed["total_slots"] = slots if isinstance(slots, int) and not isinstance(slots, bool) else "not_evaluated"
    if pinned_np is not None:
        if not isinstance(slots, int) or isinstance(slots, bool):
            errors.append(f"n_parallel pinned to {pinned_np} but server /props lacks total_slots")
        elif slots > int(pinned_np):
            errors.append(f"server total_slots {slots} exceeds planned n_parallel {pinned_np}")
        elif slots < int(pinned_np):
            observed["parallelism_note"] = (f"runtime reduced to recurrent fit: total_slots {slots} < n_parallel {pinned_np}")
    want_mode = str(spec.get("mode", "sr" if spec.get("sr_enabled", False) else ("dense" if spec.get("xkv_enabled", False) else "off")))
    mode_obs = info.get("xkv_mode") or props.get("xkv_mode")
    observed["xkv_mode"] = mode_obs if mode_obs else "not_evaluated"
    if spec.get("xkv_enabled"):
        if not mode_obs:
            errors.append("xkv enabled but server exposes no xkv mode (metrics xkv_mode / props xkv_mode)")
        elif str(mode_obs) != want_mode:
            errors.append(f"xkv mode mismatch: server {mode_obs!r} != plan {want_mode!r}")
    else:
        if mode_obs not in (None, "", "off"):
            errors.append(f"foreign xkv mode {mode_obs!r} on baseline plan {profile!r}")
    # NOTE: bare tri_/rerot_ series presence is superseded by the strict
    # config checks below (calibration/ratio/window/frontier); a counter
    # alone never decides identity.
    _ = names
    if spec.get("triattention_enabled"):
        calib = (info.get("tri_calibration_sha256") or info.get("tri_calibration_fingerprint")
                 or props.get("tri_calibration_sha256") or props.get("tri_calibration_fingerprint"))
        exp_calib = ((baseline or {}).get("deployed_artifacts", {}).get("triattention_calibration", {}).get("historical_expected_sha256"))
        observed["tri_calibration"] = calib if calib else "not_evaluated"
        if not calib:
            errors.append("triattention enabled but server exposes no tri calibration sha/fingerprint (metrics/props)")
        elif exp_calib and str(calib).lower() != str(exp_calib).lower():
            errors.append("tri calibration mismatch: server != baseline expected")
        ratio_obs = (info.get("triattention_ratio") or props.get("triattention_ratio") or props.get("tri_ratio"))
        want_ratio = spec.get("triattention_target", "3/32")
        observed["triattention_ratio"] = ratio_obs if ratio_obs is not None else "not_evaluated"
        if ratio_obs is None:
            errors.append("triattention enabled but server exposes no tri ratio (metrics/props)")
        else:
            try:
                if abs(_ratio_to_float(ratio_obs) - _ratio_to_float(want_ratio)) > 1e-9:
                    errors.append(f"tri ratio mismatch: server {ratio_obs!r} != plan {want_ratio!r}")
            except (ValueError, TypeError, ZeroDivisionError):
                errors.append(f"tri ratio unparseable: {ratio_obs!r}")
        window_obs = info.get("tri_recent_window") or props.get("tri_recent_window")
        observed["tri_recent_window"] = window_obs if window_obs is not None else "not_evaluated"
        if window_obs is None:
            errors.append("triattention enabled but server exposes no tri recent-window (metrics/props)")
        else:
            try:
                if int(window_obs) != TRI_RECENT_WINDOW_PIN:
                    errors.append(f"tri recent-window mismatch: server {window_obs!r} != production pin 128")
            except (ValueError, TypeError):
                errors.append(f"tri recent-window unparseable: {window_obs!r}")
    if spec.get("rerot_enabled"):
        frontier = info.get("rerot_frontier") or props.get("rerot_frontier")
        observed["rerot_frontier"] = frontier if frontier else "not_evaluated"
        if not frontier:
            errors.append("rerot enabled but server exposes no rerot frontier mode (metrics/props)")
        elif str(frontier) != "strong":
            errors.append(f"rerot frontier mismatch: server {frontier!r} != plan 'strong'")
    if spec.get("mtp_enabled"):
        draft_obs = props.get("spec_draft_n_max", props.get("mtp_draft_max", props.get("draft_max")))
        want_draft = int(spec.get("spec_draft_n_max", 2))
        observed["spec_draft_n_max"] = draft_obs if draft_obs is not None else "not_evaluated"
        if draft_obs is None:
            errors.append("mtp enabled but server exposes no draft-max (props)")
        else:
            try:
                if int(draft_obs) != want_draft:
                    errors.append(f"draft max mismatch: server {draft_obs!r} != plan {want_draft}")
            except (ValueError, TypeError):
                errors.append(f"draft max unparseable: {draft_obs!r}")
        blob = json.dumps(props)
        if any(k in blob for k in ("speculative", "mtp", "draft", "MTP")):
            observed["mtp"] = "observed"
        else:
            observed["mtp"] = "not_evaluated"
            errors.append("mtp enabled but no speculative/mtp indicator in /props")
    if spec.get("xkv_enabled") and spec.get("xkv_seed") is not None:
        seed_obs = metrics_seed(mtext, props)
        observed["xkv_seed"] = seed_obs if seed_obs is not None else "not_evaluated"
        want_seed = str(int(str(spec["xkv_seed"]), 0))
        if seed_obs is None:
            errors.append("xkv_seed pinned but server exposes no factorization seed (metrics/props)")
        elif str(seed_obs) != want_seed:
            errors.append(f"xkv_seed mismatch: server {seed_obs!r} != plan {spec['xkv_seed']!r}")
    if spec.get("xkv_enabled"):
        # Verify codecs
        for codec_key, spec_key in (
            ("xkv_codec_a_k", "a_k_codec"),
            ("xkv_codec_b_k", "b_k_codec"),
            ("xkv_codec_a_v", "a_v_codec"),
            ("xkv_codec_b_v", "b_v_codec"),
            ("xkv_codec_landmark", "landmark_codec"),
        ):
            if spec_key in spec:
                want_c = str(spec[spec_key])
                obs_c = info.get(codec_key) or props.get(codec_key) or props.get(spec_key)
                observed[codec_key] = obs_c if obs_c else "not_evaluated"
                if not obs_c:
                    errors.append(f"xkv {spec_key} pinned to {want_c!r} but server exposes no {codec_key} (metrics/props)")
                elif str(obs_c).lower() != want_c.lower():
                    errors.append(f"{spec_key} mismatch: server {obs_c!r} != plan {want_c!r}")
        # Verify factorizer
        if "factorizer" in spec:
            want_f = str(spec["factorizer"])
            obs_f = info.get("xkv_factorizer") or props.get("xkv_factorizer") or props.get("factorizer")
            observed["xkv_factorizer"] = obs_f if obs_f else "not_evaluated"
            if not obs_f:
                errors.append(f"xkv factorizer pinned to {want_f!r} but server exposes no xkv_factorizer (metrics/props)")
            elif str(obs_f).lower() != want_f.lower():
                errors.append(f"factorizer mismatch: server {obs_f!r} != plan {want_f!r}")
        # Verify factor balance
        if "factor_balance" in spec:
            want_b = str(spec["factor_balance"])
            obs_b = info.get("xkv_factor_balance") or props.get("xkv_factor_balance") or props.get("factor_balance")
            observed["xkv_factor_balance"] = obs_b if obs_b else "not_evaluated"
            if not obs_b:
                errors.append(f"xkv factor_balance pinned to {want_b!r} but server exposes no xkv_factor_balance (metrics/props)")
            elif str(obs_b).lower() != want_b.lower():
                errors.append(f"factor_balance mismatch: server {obs_b!r} != plan {want_b!r}")
    return (len(errors) == 0), errors, observed


def metrics_seed(mtext: str, props: Dict[str, Any]) -> Optional[Any]:
    """Extracts the observed factorization seed: props keys first (exact string),
    then metrics xkv_seed_info{seed="..."} or numeric xkv_factor_seed/xkv_seed gauge.
    Normalizes hex (0x...) to integer string representation for uniform comparison."""
    for k in ("xkv_factor_seed", "xkv_seed", "factor_seed"):
        if isinstance(props.get(k), str) and props[k].strip():
            v = props[k].strip()
            try:
                return str(int(v, 0))
            except ValueError:
                return v
    import re
    for line in mtext.splitlines():
        s = line.strip()
        m_info = re.match(r'^(?:llamacpp:)?xkv_seed_info\{.*seed="([^"]+)".*\}\s+\S+\s*$', s)
        if m_info:
            v = m_info.group(1).strip()
            try:
                return str(int(v, 0))
            except ValueError:
                return v
        m = re.match(r"^(?:llamacpp:)?(xkv_factor_seed|xkv_seed)\s+([0-9.eE+\-]+)\s*$", s)
        if m:
            return str(int(float(m.group(2))))
    for k in ("xkv_factor_seed", "xkv_seed", "factor_seed"):
        v = props.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(int(v)) if float(v).is_integer() else repr(v)
    return None


def launch_profile_server(server_binary: str, model: str, port: int,
                            spec: Dict[str, Any], calibration_path: Optional[str],
                            runtime_cfg: Dict[str, Any], extra_args: List[str],
                            log_path: str, timeout_s: int = 600) -> Dict[str, Any]:
    """Spawns one llama-server per profile (argv array, no shell), waits for /health."""
    argv = build_server_argv(server_binary, model, port, spec, calibration_path,
                             runtime_cfg, extra_args)
    d = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(d, exist_ok=True)
    logf = open(log_path, "wb")
    proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout_s
    last_err = ""
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server for exited rc={proc.returncode} during launch; log={log_path}")
            try:
                st, body = _http_get_text(url + "/health", 5)
                if st == 200 and ("\"ok\"" in body or "ok" in body):
                    return {"pid": proc.pid, "port": port, "url": url,
                            "argv": argv, "log": log_path, "proc": proc, "logf": logf}
            except Exception as e:
                last_err = repr(e)
            time.sleep(1)
    except BaseException:
        try:
            proc.terminate()
        except OSError:
            pass
        logf.close()
        raise
    try:
        proc.terminate()
    except OSError:
        pass
    logf.close()
    raise RuntimeError(f"server not healthy within {timeout_s}s ({last_err}); log={log_path}")


def stop_profile_server(handle: Dict[str, Any]) -> None:
    proc = handle.get("proc")
    logf = handle.get("logf")
    try:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        try:
            if logf is not None:
                logf.close()
        except OSError:
            pass


def preflight(args: argparse.Namespace, profiles: List[str],
               profile_data: Dict[str, Any],
               corpus_errors: List[str],
               baseline_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Fail-closed preflight: missing exact artifacts/modes/URLs are errors.

    Never synthetic: a failed preflight records not_evaluated, never a pass.
    """
    errors: List[str] = []
    specs = {p: profile_data.get("profiles", {}).get(p) for p in profiles}
    for p in profiles:
        if specs[p] is None:
            errors.append(f"profile {p!r} missing from profile JSON")
    try:
        require_profile_schema(profile_data)
    except ValueError as e:
        errors.append(str(e))
    for p in profiles:
        s = specs.get(p)
        if s is None:
            continue
        try:
            check_profile_mode(p, s)
        except ValueError as e:
            errors.append(str(e))
    if corpus_errors:
        errors += [f"corpus: {e}" for e in corpus_errors]
    backend = args.backend
    needs_local_model = backend in ("cli", "ppl") or getattr(args, "server_launch", False)
    if backend in ("cli", "ppl"):
        if not args.binary:
            errors.append(f"{backend} backend requires --binary")
        elif not (os.path.isfile(args.binary) and os.access(args.binary, os.X_OK)):
            errors.append(f"binary missing/not executable: {args.binary}")
        if backend == "ppl" and not getattr(args, "cli_template", ""):
            errors.append("ppl backend requires --cli-template")
    if getattr(args, "server_launch", False):
        sb = getattr(args, "server_binary", "")
        if not sb:
            errors.append("server launch requires --server-binary")
        elif not (os.path.isfile(sb) and os.access(sb, os.X_OK)):
            errors.append(f"server binary missing/not executable: {sb}")
    if needs_local_model:
        if not args.model:
            errors.append("model artifact missing: --model empty and no profile-root model.path")
        elif not os.path.isfile(args.model):
            errors.append(f"model artifact missing: {args.model}")
        if any((specs[p] or {}).get("triattention_enabled") for p in profiles):
            if not args.calibration:
                errors.append("triattention enabled but --calibration empty")
            elif not os.path.isfile(args.calibration):
                errors.append(f"calibration artifact missing: {args.calibration}")
            else:
                exp_calib = ((baseline_data or {}).get("deployed_artifacts", {}).get(
                    "triattention_calibration", {}).get("historical_expected_sha256"))
                if exp_calib and sha256_file(args.calibration) != exp_calib:
                    errors.append("calibration sha mismatch: file != baseline expected triattention sha")
    seed_profiles = [p for p in profiles if (specs.get(p) or {}).get("xkv_seed") is not None]
    if seed_profiles:
        if getattr(args, "server_launch", False):
            probe_bin = getattr(args, "server_binary", "")
        elif backend in ("cli", "ppl"):
            probe_bin = args.binary
        else:
            probe_bin = ""
        if probe_bin:
            if not binary_supports_flag(probe_bin, "--xkv-seed"):
                errors.append(
                    f"binary {probe_bin} lacks --xkv-seed (C++ support pending for pinned "
                    f"xkv_seed on {seed_profiles}; failing closed)")
    if backend == "server" and not getattr(args, "server_launch", False):
        if not (args.model and os.path.isfile(args.model)):
            errors.append(
                "server backend requires --model pointing at a local copy of the deployed "
                "weights (model checksum identity; missing checksum => NOT_EVALUATED)")
        try:
            mapping = parse_server_urls(getattr(args, "server_urls", ""))
        except ValueError as e:
            errors.append(str(e))
            mapping = {}
        if len(profiles) > 1:
            if not mapping:
                errors.append(
                    "server backend with >1 profile requires explicit --server-urls "
                    "mapping (one already-started URL for all profiles would mislabel configs)")
            else:
                missing = [p for p in profiles if p not in mapping]
                if missing:
                    errors.append(f"--server-urls missing profiles: {missing}")
                urls = [mapping[p] for p in profiles if p in mapping]
                if len(set(urls)) != len(urls):
                    errors.append(
                        f"duplicate URL across profiles in --server-urls: {mapping} "
                        "(each profile needs its own server config)")
        else:
            if not mapping and not getattr(args, "server_url", ""):
                errors.append("server backend requires --server-url or --server-urls")
            if mapping and profiles[0] not in mapping:
                errors.append(f"--server-urls missing profile: {profiles[0]}")
    checks = {
        "backend": backend,
        "local_model_required": needs_local_model,
        "model_path": args.model,
        "binary_path": args.binary,
        "calibration_path": args.calibration,
    }
    return (len(errors) == 0), errors, checks


def write_preflight_failure(args: argparse.Namespace, profiles: List[str],
                             profile_data: Dict[str, Any],
                             errors: List[str], checks: Dict[str, Any],
                             corpus_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Writes per-profile manifests + evidence for a REAL (non-synthetic) preflight failure."""
    per_profile: Dict[str, Any] = {}
    for profile in profiles:
        spec = profile_data.get("profiles", {}).get(profile, {})
        prof_dir = os.path.join(args.out_dir, profile)
        os.makedirs(os.path.join(prof_dir, "raw"), exist_ok=True)
        manifest = {
            "version": VERSION,
            "profile": profile,
            "profile_spec": spec,
            "model_path": args.model,
            "model_sha256": sha256_file(args.model),
            "binary_path": args.binary,
            "binary_sha256": sha256_file(args.binary),
            "calibration_path": args.calibration,
            "calibration_sha256": sha256_file(args.calibration),
            "commit": git_commit(),
            "source_fingerprint": git_commit(),
            "profile_schema_version": profile_data.get("version", "not_evaluated"),
            "profile_fingerprint": profile_fingerprint(
                spec, profile_data.get("version", -1)),
            "backend": args.backend,
            "preflight_ok": False,
            "preflight_errors": errors,
            "preflight_checks": checks,
            "profile_error": "; ".join(errors),
            "item_count": 0,
            "returncode": 2,
            "note": "REAL preflight failure: missing/inconsistent artifacts; "
                      "not_evaluated, never a synthetic PASS",
        }
        atomic_write_json(os.path.join(prof_dir, "manifest.json"), manifest)
        per_profile[profile] = {"manifest": manifest, "records": []}
    atomic_write_json(os.path.join(args.out_dir, "preflight.json"),
                        {"ok": False, "errors": errors, "checks": checks})
    target = getattr(args, "target_profile", "") or (profiles[0] if profiles else "B0")
    spec = profile_data.get("profiles", {}).get(target, {})
    evidence = {
        "__synthetic_test_fixture__": False,
        "note": "REAL preflight failure; not_evaluated, never synthetic",
        "xkv_requested_profile": target,
        "xkv_effective_profile": "not_evaluated",
        "xkv_effective_storage_profile": spec.get("storage_profile", "not_evaluated"),
        "matrix_verdict": "NOT_EVALUATED",
        "matrix_reasons": errors,
        "per_item": [],
    }
    for k in ("xkv_codec_a_k", "xkv_codec_b_k", "xkv_codec_a_v", "xkv_codec_b_v",
                "xkv_codec_landmark", "calibration_sha256", "source_fingerprint",
                "profile_fingerprint", "xkv_factored_baseline_byte_coverage",
                "has_resident_dense_mirror", "xkv_net_saved_fraction",
                "xkv_net_extra_compression_ratio", "xkv_peak_vs_baseline_ratio",
                "xkv_live_bytes", "xkv_reserved_bytes", "xkv_peak_bytes",
                "quality_ppl_degradation_pct", "quality_context_scaling_ratio",
                "speculative_acceptance_parity", "triattention_compatible",
                "rerot_compatible", "state_code_stream_exact"):
        evidence[k] = "not_evaluated"
    return evidence

# ---------------------------------------------------------------- plan

def build_plan(profile_data: Dict[str, Any], profiles: List[str],
               corpus: List[Dict[str, Any]], sampling: Dict[str, Any],
               backend: str, model_path: str,
               server_info: Optional[Dict[str, Any]] = None,
               model_source: str = "cli-flag") -> Dict[str, Any]:
    prof_defs = profile_data.get("profiles", {})
    plan_profiles = []
    for name in profiles:
        if name not in prof_defs:
            raise ValueError(f"profile {name!r} not in profile JSON")
        spec = prof_defs[name]
        plan_profiles.append({
            "name": name,
            "xkv_enabled": bool(spec.get("xkv_enabled", False)),
            "storage_profile": spec.get("storage_profile", "none"),
            "profile_sha256": sha256_hex(canonical(spec)),
        })
    items = [{"id": r["id"], "category": r.get("category", ""),
              "prompt_sha256": sha256_hex(r["prompt"].encode("utf-8")),
              "scorer": r["scorer"]} for r in corpus]
    plan = {
        "version": VERSION,
        "backend": backend,
        "model_path": model_path,
        "model_path_source": model_source,
        "sampling": sampling,
        "sampling_fingerprint": sha256_hex(canonical(sampling)),
        "profiles": plan_profiles,
        "items": items,
        "server": server_info or {},
    }
    plan["plan_sha256"] = sha256_hex(canonical({k: v for k, v in plan.items()
                                                if k != "plan_sha256"}))
    return plan


# ---------------------------------------------------------------- runner

class MatrixRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        with open(args.profile, "r", encoding="utf-8") as f:
            self.profile_data = json.load(f)
        self.sampling = {
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "n_predict": args.n_predict,
            "template": args.template or "",
        }
        self.sampling_fp = sha256_hex(canonical(self.sampling))
        self.profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
        for p in self.profiles:
            if p not in MATRIX_PROFILES:
                raise ValueError(f"unknown profile {p!r} (expect one of {MATRIX_PROFILES})")
        self.model_source = "cli-flag"
        if not args.model:
            root_model = (self.profile_data.get("model") or {}).get("path", "")
            if root_model:
                args.model = root_model
                self.model_source = "profile-root:model.path"
        try:
            self.server_urls = parse_server_urls(getattr(args, "server_urls", ""))
        except ValueError as e:
            raise ValueError(f"--server-urls: {e}")
        self.runtime_cfg = runtime_cfg_from_args(args)
        self.baseline_data: Dict[str, Any] = {}
        try:
            with open(getattr(args, "baseline", ""), "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.baseline_data = loaded
        except (OSError, json.JSONDecodeError):
            pass
        grader_path = getattr(args, "math_grader", "")
        self.grader = None
        if grader_path:
            self.grader = {"path": grader_path, "sha256": sha256_file(grader_path)}

    def load_existing(self, results_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        out: Dict[str, Dict[str, Any]] = {}
        stats = {"lines": 0, "accepted": 0, "json_errors": 0, "checksum_dropped": 0}
        if not os.path.exists(results_path):
            return out, stats
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["json_errors"] += 1
                    continue
                if not (isinstance(rec, dict) and "id" in rec):
                    stats["json_errors"] += 1
                    continue
                want = rec.pop("record_sha256", None)
                if want is None or want != sha256_hex(canonical(rec)):
                    stats["checksum_dropped"] += 1
                    continue
                rec["record_sha256"] = want
                out[str(rec["id"])] = rec
                stats["accepted"] += 1
        return out, stats

    def server_url_for(self, profile: str, index: int) -> str:
        """Resolves exactly one URL per profile; shared URLs fail closed in preflight."""
        if getattr(self.args, "server_launch", False):
            return f"http://127.0.0.1:{getattr(self.args, 'server_port_base', 8090) + index}"
        if profile in self.server_urls:
            return self.server_urls[profile]
        if len(self.profiles) == 1 and getattr(self.args, "server_url", ""):
            return self.args.server_url
        raise ValueError(f"no server URL for profile {profile!r} (explicit mapping required)")

    def execute_item(self, profile: str, rec: Dict[str, Any],
                     raw_dir: str, server_url: Optional[str] = None) -> Dict[str, Any]:
        rid = str(rec["id"])
        prompt = rec["prompt"]
        t0 = time.time()
        stdout_b, stderr_b = b"", b""
        transport: Dict[str, Any] = {}
        timed_out = False
        prof_spec = self.profile_data.get("profiles", {}).get(profile, {})

        if self.args.dry_run:
            stdout_b = f"DRY-RUN synthetic output for {rid}".encode()
            transport = {"dry_run": True, "wall_s": 0.0}
        elif self.args.backend == "server":
            stdout_b, stderr_b, transport = run_server_backend(
                server_url or self.args.server_url, prompt, self.sampling, self.args.timeout)
        elif self.args.backend == "cli":
            argv = build_cli_argv(self.args.binary, self.args.model, prompt,
                                  self.sampling, self.args.template,
                                  profile_spec=prof_spec,
                                  calibration_path=self.args.calibration)
            transport["argv"] = argv
            stdout_b, stderr_b, transport_meta = run_cli_backend(argv, self.args.timeout)
            transport.update(transport_meta)
        elif self.args.backend == "ppl":
            if not self.args.cli_template:
                transport = {"returncode": None, "wall_s": 0.0,
                             "error": "ppl backend requires --cli-template"}
            else:
                try:
                    argv = build_template_argv(self.args.cli_template, {
                        "binary": self.args.binary, "model": self.args.model,
                        "prompt": prompt, "seed": str(self.args.seed),
                        "n_predict": str(self.args.n_predict),
                    })
                except ValueError as e:
                    argv = []
                    transport = {"returncode": None, "wall_s": 0.0, "error": str(e)}
                else:
                    transport["argv"] = argv
                    stdout_b, stderr_b, meta = run_cli_backend(argv, self.args.timeout)
                    transport.update(meta)
        else:
            transport = {"returncode": None, "wall_s": 0.0,
                         "error": f"unknown backend {self.args.backend!r}"}

        if b"TIMEOUT" in stderr_b or (transport.get("error", "").startswith("timeout")):
            timed_out = True

        atomic_write_bytes(os.path.join(raw_dir, f"{rid}.stdout"), stdout_b)
        atomic_write_bytes(os.path.join(raw_dir, f"{rid}.stderr"), stderr_b)

        if self.args.backend == "server" and not self.args.dry_run:
            sse = parse_sse_stream(stdout_b)
            atomic_write_bytes(os.path.join(raw_dir, f"{rid}.sse"), stdout_b)
            content = sse["assembled_content"]
            stream_error = sse["error"]
            finish_reason = sse["finish_reason"]
            tok = {"prompt_tokens": sse["prompt_tokens"],
                   "predicted_tokens": sse["predicted_tokens"]}
            timings = dict(sse["timings"])
            timings["wall_s"] = transport.get("wall_s")
            sse_info = {"done_count": sse["done_count"],
                        "finish_count": sse["finish_count"],
                        "malformed_lines": sse["malformed_lines"]}
        else:
            content = stdout_b.decode("utf-8", errors="replace")
            stream_error = None
            finish_reason = None
            tok = {"prompt_tokens": None, "predicted_tokens": None}
            timings = {"wall_s": transport.get("wall_s", time.time() - t0)}
            sse_info = {}

        grader = getattr(self, "grader", None)
        scorer_res = score_item(rec["scorer"], content, rec.get("reference"), grader)
        err = transport.get("error") or stream_error
        if timed_out and err is None:
            err = "timeout"
        if err:
            status = "TIMEOUT" if timed_out else "ERROR"
        elif stream_error:
            status = "ERROR"
        elif scorer_res["status"] == "NOT_EVALUATED":
            status = "NOT_EVALUATED"
        else:
            status = "OK" if scorer_res.get("passed") else "OK_SCORED_FAIL"

        record = {
            "id": rid,
            "category": rec.get("category", ""),
            "profile": profile,
            "prompt_sha256": sha256_hex(prompt.encode("utf-8")),
            "sampling_fingerprint": self.sampling_fp,
            "content": content,
            "finish_reason": finish_reason,
            "token_counts": tok,
            "timings": timings,
            "sse": sse_info,
            "scorer": rec["scorer"],
            "scorer_result": scorer_res,
            "raw_stdout": f"raw/{rid}.stdout",
            "raw_stderr": f"raw/{rid}.stderr",
            "transport": {k: v for k, v in transport.items() if k != "argv"}
            if self.args.backend == "server" else transport,
            "status": status,
            "error": err,
            "elapsed_s": time.time() - t0,
        }
        record["record_sha256"] = sha256_hex(canonical(record))
        return record

    def fail_profile(self, profile: str, prof_spec: Dict[str, Any],
                       corpus_errors: List[str], error: str) -> Dict[str, Any]:
        """Writes a REAL (non-synthetic) failure manifest with zero records."""
        prof_dir = os.path.join(self.args.out_dir, profile)
        os.makedirs(os.path.join(prof_dir, "raw"), exist_ok=True)
        manifest = {
            "version": VERSION,
            "profile": profile,
            "profile_spec": prof_spec,
            "model_path": self.args.model,
            "model_sha256": sha256_file(self.args.model),
            "model_path_source": getattr(self, "model_source", "cli-flag"),
            "binary_path": self.args.binary,
            "binary_sha256": sha256_file(self.args.binary),
            "calibration_path": self.args.calibration,
            "calibration_sha256": sha256_file(self.args.calibration),
            "commit": git_commit(),
            "source_fingerprint": git_commit(),
            "profile_schema_version": self.profile_data.get("version", "not_evaluated"),
            "profile_fingerprint": profile_fingerprint(
                prof_spec, self.profile_data.get("version", -1)),
            "backend": self.args.backend,
            "preflight_ok": False,
            "profile_error": error,
            "corpus_errors": corpus_errors,
            "item_count": 0,
            "returncode": 2,
            "note": "REAL profile failure; not_evaluated, never synthetic",
        }
        atomic_write_json(os.path.join(prof_dir, "manifest.json"), manifest)
        return {"manifest": manifest, "records": []}

    def run_profile(self, profile: str, corpus: List[Dict[str, Any]],
                    corpus_errors: List[str]) -> Dict[str, Any]:
        prof_dir = os.path.join(self.args.out_dir, profile)
        raw_dir = os.path.join(prof_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        results_path = os.path.join(prof_dir, "results.jsonl")
        existing, resume_stats = self.load_existing(results_path)
        prof_spec = self.profile_data.get("profiles", {}).get(profile, {})
        try:
            require_profile_schema(self.profile_data)
            check_profile_mode(profile, prof_spec)
        except ValueError as e:
            return self.fail_profile(profile, prof_spec, corpus_errors, str(e))

        manifest = {
            "version": VERSION,
            "profile": profile,
            "profile_spec": prof_spec,
            "model_path": self.args.model,
            "model_sha256": sha256_file(self.args.model),
            "binary_path": self.args.binary,
            "binary_sha256": sha256_file(self.args.binary),
            "calibration_path": self.args.calibration,
            "calibration_sha256": sha256_file(self.args.calibration),
            "commit": git_commit(),
            "source_fingerprint": git_commit(),
            "profile_schema_version": self.profile_data.get("version", "not_evaluated"),
            "profile_fingerprint": profile_fingerprint(
                prof_spec, self.profile_data.get("version", -1)),
            "backend": self.args.backend,
            "server_url": self.args.server_url,
            "server_urls": dict(self.server_urls),
            "server_launch": bool(getattr(self.args, "server_launch", False)),
            "runtime_cfg": dict(self.runtime_cfg),
            "model_path_source": getattr(self, "model_source", "cli-flag"),
            "grader": getattr(self, "grader", None),
            "preflight_ok": True,
            "profile_error": None,
            "sampling": self.sampling,
            "sampling_fingerprint": self.sampling_fp,
            "argv": sys.argv[:],
            "target_cli_argv": build_cli_argv(
                self.args.binary, self.args.model, "<PROMPT>",
                self.sampling, self.args.template,
                profile_spec=prof_spec,
                calibration_path=self.args.calibration
            ) if self.args.backend == "cli" else [],
            "target_server_argv": build_server_argv(
                getattr(self.args, "server_binary", "") or self.args.binary,
                self.args.model, getattr(self.args, "server_port_base", 8090),
                prof_spec, self.args.calibration, dict(self.runtime_cfg),
                list(getattr(self.args, "server_arg", []) or [])
            ) if self.args.backend == "server" else [],
            "env": {k: os.environ.get(k, "") for k in
                    ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                     "VK_ICD_FILENAMES")},
            "timeout_s": self.args.timeout,
            "corpus_errors": corpus_errors,
            "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.args.dry_run or self.args.self_test:
            manifest["__synthetic_test_fixture__"] = True
            manifest["note"] = "SYNTHETIC TEST FIXTURE ONLY - NOT REAL EMPIRICAL EVIDENCE"

        records: Dict[str, Dict[str, Any]] = {}
        corpus_by_id = {str(r["id"]): r for r in corpus}
        for rid, rec in existing.items():
            want = corpus_by_id.get(rid)
            if (want is not None
                    and rec.get("prompt_sha256") == sha256_hex(want["prompt"].encode("utf-8"))
                    and rec.get("sampling_fingerprint") == self.sampling_fp
                    and rec.get("status") in ("OK", "OK_SCORED_FAIL", "NOT_EVALUATED")):
                records[rid] = rec
        server_handle = None
        server_url = None
        if self.args.backend == "server" and not self.args.dry_run:
            profile_index = self.profiles.index(profile) if profile in self.profiles else 0
            try:
                server_url = self.server_url_for(profile, profile_index)
                manifest["server_url"] = server_url
                if getattr(self.args, "server_launch", False):
                    port = getattr(self.args, "server_port_base", 8090) + profile_index
                    server_handle = launch_profile_server(
                        getattr(self.args, "server_binary"), self.args.model, port,
                        prof_spec, self.args.calibration, self.runtime_cfg,
                        list(getattr(self.args, "server_arg", []) or []),
                        os.path.join(prof_dir, "server.log"),
                        int(getattr(self.args, "server_launch_timeout", 600)))
                    manifest["server"] = {k: v for k, v in server_handle.items()
                                          if k not in ("proc", "logf")}
                    manifest["target_server_argv"] = manifest["server"]["argv"]
                ok, id_errors, observed = verify_server_identity(
                    server_url, profile, prof_spec, self.args.model or None,
                    min(30, self.args.timeout),
                    getattr(self, "baseline_data", None),
                    manifest.get("model_sha256"))
                manifest["server_identity"] = {"ok": ok, "errors": id_errors,
                                                "observed": observed}
                if not ok:
                    raise RuntimeError("server identity rejected: " + "; ".join(id_errors))
            except Exception as e:
                manifest["profile_error"] = f"{type(e).__name__}: {e}"
        try:
            if manifest.get("profile_error") is None:
                for rec in corpus:
                    rid = str(rec["id"])
                    if rid in records:
                        continue
                    records[rid] = self.execute_item(profile, rec, raw_dir, server_url)
        finally:
            if server_handle is not None:
                stop_profile_server(server_handle)

        ordered = [records[k] for k in sorted(records)]
        d = os.path.dirname(os.path.abspath(results_path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in ordered:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            os.replace(tmp, results_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        manifest["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["item_count"] = len(ordered)
        manifest["resume"] = dict(resume_stats, reused=len(records))
        if manifest.get("profile_error"):
            manifest["returncode"] = 2
        else:
            manifest["returncode"] = 0 if all(
                r.get("status") in ("OK", "OK_SCORED_FAIL") for r in ordered) else 1
        atomic_write_json(os.path.join(prof_dir, "manifest.json"), manifest)
        return {"manifest": manifest, "records": ordered}

    def aggregate(self, per_profile: Dict[str, Any]) -> Dict[str, Any]:
        fps = {p: per_profile[p]["manifest"]["sampling_fingerprint"] for p in per_profile}
        sampling_consistent = len(set(fps.values())) == 1

        hashes = {}
        for p in per_profile:
            m = per_profile[p]["manifest"]
            hashes[p] = (m.get("model_sha256"), m.get("binary_sha256"),
                         m.get("calibration_sha256"))
        hash_consistent = len(set(hashes.values())) == 1

        evidence: Dict[str, Any] = {}
        target = self.args.target_profile
        if target not in per_profile:
            target = sorted(per_profile)[0] if per_profile else "B0"
        spec = self.profile_data.get("profiles", {}).get(target, {})
        recs = per_profile.get(target, {}).get("records", [])
        incomplete = [r["id"] for r in recs
                      if r.get("status") not in ("OK", "OK_SCORED_FAIL")]

        verdict = "PASS"
        reasons: List[str] = []
        if not recs:
            verdict = "NOT_EVALUATED"
            reasons.append("no items executed")
        if not sampling_consistent:
            verdict = "NOT_EVALUATED"
            reasons.append(f"sampling changed across profiles: {fps}")
        if not hash_consistent:
            verdict = "NOT_EVALUATED"
            reasons.append(f"binary/model/calibration hash mismatch: {hashes}")
        if incomplete:
            verdict = "NOT_EVALUATED"
            reasons.append(f"incomplete items: {incomplete}")
        fails = [r["id"] for r in recs if r.get("status") == "OK_SCORED_FAIL"]
        for p in per_profile:
            pe = per_profile[p]["manifest"].get("profile_error")
            if pe:
                if verdict == "PASS":
                    verdict = "NOT_EVALUATED"
                reasons.append(f"{p}: {pe}")
        for p in per_profile:
            rc = (per_profile[p]["manifest"].get("runtime_cfg") or {})
            if rc.get("gpu_layers", "all") != "all":
                if verdict == "PASS":
                    verdict = "NOT_EVALUATED"
                reasons.append(f"{p}: --gpu-layers override {rc.get('gpu_layers')!r} makes comparison non-equivalent")
        devices = set()
        for p in per_profile:
            rc = (per_profile[p]["manifest"].get("runtime_cfg") or {})
            devices.add(json.dumps(rc.get("device"), sort_keys=True))
        if len(devices) > 1:
            if verdict == "PASS":
                verdict = "NOT_EVALUATED"
            reasons.append(f"device differs across profiles: {sorted(devices)}")
        if fails and verdict == "PASS":
            verdict = "FAIL"
            reasons.append(f"scored failures: {fails}")

        by_cat: Dict[str, Dict[str, Any]] = {}
        for r in recs:
            c = r.get("category", "")
            by_cat.setdefault(c, {"n": 0, "passed": 0, "not_evaluated": 0})
            by_cat[c]["n"] += 1
            sr = r.get("scorer_result", {})
            if sr.get("status") == "SCORED" and sr.get("passed"):
                by_cat[c]["passed"] += 1
            if r.get("status") == "NOT_EVALUATED":
                by_cat[c]["not_evaluated"] += 1

        man = per_profile.get(target, {}).get("manifest", {})
        evidence = {
            "__synthetic_test_fixture__": bool(self.args.dry_run or self.args.self_test),
            "note": ("SYNTHETIC TEST FIXTURE ONLY - NOT REAL EMPIRICAL EVIDENCE"
                     if (self.args.dry_run or self.args.self_test) else
                     "matrix runner aggregate; byte/coverage keys need collect-runtime-evidence"),
            "xkv_requested_profile": target,
            # Effective profile in matrix runner must not be unconditionally target;
            # if run had errors/incomplete/mismatches, emit not_evaluated fail-closed.
            "xkv_effective_profile": target if verdict == "PASS" else "not_evaluated",
            "xkv_effective_storage_profile": spec.get("storage_profile", "not_evaluated"),
            "xkv_codec_a_k": spec.get("a_k_codec", spec.get("k_codec", "not_evaluated")),
            "xkv_codec_b_k": spec.get("b_k_codec", spec.get("k_codec", "not_evaluated")),
            "xkv_codec_a_v": spec.get("a_v_codec", spec.get("v_codec", "not_evaluated")),
            "xkv_codec_b_v": spec.get("b_v_codec", spec.get("v_codec", "not_evaluated")),
            "xkv_codec_landmark": spec.get("landmark_codec", "not_evaluated"),
            "calibration_sha256": man.get("calibration_sha256") or "not_evaluated",
            "xkv_factored_baseline_byte_coverage": "not_evaluated",
            "has_resident_dense_mirror": "not_evaluated",
            "xkv_net_saved_fraction": "not_evaluated",
            "xkv_net_extra_compression_ratio": "not_evaluated",
            "xkv_peak_vs_baseline_ratio": "not_evaluated",
            "xkv_live_bytes": "not_evaluated",
            "xkv_reserved_bytes": "not_evaluated",
            "xkv_peak_bytes": "not_evaluated",
            "source_fingerprint": man.get("source_fingerprint", "not_evaluated"),
            "profile_fingerprint": man.get("profile_fingerprint", "not_evaluated"),
            "quality_ppl_degradation_pct": "not_evaluated",
            "quality_context_scaling_ratio": "not_evaluated",
            "speculative_acceptance_parity": "not_evaluated",
            "triattention_compatible": "not_evaluated",
            "rerot_compatible": "not_evaluated",
            "state_code_stream_exact": "not_evaluated",
            "matrix_verdict": verdict,
            "matrix_reasons": reasons,
            "sampling_consistent": sampling_consistent,
            "hash_consistent": hash_consistent,
            "by_category": by_cat,
            "per_item": recs,
        }
        return evidence


# ---------------------------------------------------------------- self-test

def run_self_test() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    # 1. Deterministic plans
    corpus_recs = [
        {"id": "aime24-01", "category": "aime24", "prompt": "1+1?", "reference": "2", "scorer": "exact"},
        {"id": "math500-01", "category": "math500", "prompt": "2+2?", "reference": "4", "scorer": "includes"},
        {"id": "needle-01", "category": "needle", "prompt": "needle?", "reference": "hay", "scorer": "includes"},
    ]
    prof_stub = {
        "version": 2,
        "profiles": {
            "B0": {
                "name": "B0", "xkv_enabled": False, "storage_profile": "none",
                "k_codec": "turbo4_0", "v_codec": "turbo2_0",
                "triattention_enabled": True, "triattention_target": "3/32"
            },
            "E1": {
                "name": "E1", "xkv_enabled": True, "storage_profile": "tq-factors-landmarks",
                "mode": "sr",
                "xkv_seed": 6362273814452121649,
                "n_parallel": 6,
                "a_k_codec": "turbo4_0", "b_k_codec": "turbo4_0",
                "a_v_codec": "turbo4_0", "b_v_codec": "turbo4_0",
                "landmark_codec": "q8_0", "sr_enabled": True, "sr_budget": 256,
                "source": "decoded-hot", "factor_balance": "sqrt",
                "landmark_refine": "boundary", "landmark_refine_max_rows": 64,
                "workspace_mib": 256, "decode_cache_mib": 64,
                "min_saving": 0.1, "min_factor_coverage": 0.5,
                "factorizer": "vulkan",
                "triattention_enabled": True, "triattention_target": "3/32",
                "rerot_enabled": True, "mtp_enabled": True, "spec_draft_n_max": 2,
            }
        }
    }
    sampling = {"seed": 7, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
                "n_predict": 32, "template": "t"}
    p1 = build_plan(prof_stub, ["B0", "E1"], corpus_recs, sampling, "cli", "/m.gguf")
    p2 = build_plan(prof_stub, ["B0", "E1"], corpus_recs, sampling, "cli", "/m.gguf")
    assert p1["plan_sha256"] == p2["plan_sha256"], "plan not deterministic"
    checks["deterministic_plans"] = {"status": "PASS", "plan_sha256": p1["plan_sha256"]}

    # 2. Target CLI argv construction from profile_spec
    argv_b0 = build_cli_argv("/bin/llama-cli", "/m.gguf", "hi", sampling, None,
                             profile_spec=prof_stub["profiles"]["B0"],
                             calibration_path="/cal.bin")
    assert "--xkv" in argv_b0 and argv_b0[argv_b0.index("--xkv") + 1] == "off"
    assert "-ctk" in argv_b0 and argv_b0[argv_b0.index("-ctk") + 1] == "turbo4_0"
    assert "-ctv" in argv_b0 and argv_b0[argv_b0.index("-ctv") + 1] == "turbo2_0"
    assert "--triattention" in argv_b0 and "--triattention-stats" in argv_b0

    argv_e1 = build_cli_argv("/bin/llama-cli", "/m.gguf", "hi", sampling, None,
                             profile_spec=prof_stub["profiles"]["E1"],
                             calibration_path="/cal.bin")
    assert "--xkv" in argv_e1 and argv_e1[argv_e1.index("--xkv") + 1] == "sr"
    assert "--xkv-sr-budget" in argv_e1 and argv_e1[argv_e1.index("--xkv-sr-budget") + 1] == "256"
    assert "--xkv-storage-profile" in argv_e1 and argv_e1[argv_e1.index("--xkv-storage-profile") + 1] == "tq-factors-landmarks"
    assert "--xkv-factor-a-k" in argv_e1 and argv_e1[argv_e1.index("--xkv-factor-a-k") + 1] == "turbo4_0"
    assert "--xkv-landmark-type" in argv_e1 and argv_e1[argv_e1.index("--xkv-landmark-type") + 1] == "q8_0"
    assert argv_e1[argv_e1.index("--xkv-source") + 1] == "decoded-hot"
    assert argv_e1[argv_e1.index("--xkv-factor-balance") + 1] == "sqrt"
    assert argv_e1[argv_e1.index("--xkv-landmark-refine") + 1] == "boundary"
    assert argv_e1[argv_e1.index("--xkv-landmark-refine-max-rows") + 1] == "64"
    assert argv_e1[argv_e1.index("--xkv-workspace-mib") + 1] == "256"
    assert argv_e1[argv_e1.index("--xkv-decode-cache-mib") + 1] == "64"
    assert argv_e1[argv_e1.index("--xkv-min-saving") + 1] == "0.1"
    assert argv_e1[argv_e1.index("--xkv-min-factor-coverage") + 1] == "0.5"
    assert argv_e1[argv_e1.index("--xkv-factorizer") + 1] == "vulkan"
    argv_shadow = build_cli_argv("/bin/llama-cli", "/m.gguf", "hi", sampling, None,
                                 profile_spec={"xkv_enabled": True, "mode": "shadow"})
    assert argv_shadow[argv_shadow.index("--xkv") + 1] == "shadow"
    assert "--rerot" in argv_e1 and "--rerot-frontier" in argv_e1
    assert "--mtp" in argv_e1
    assert argv_e1[argv_e1.index("--spec-draft-n-max") + 1] == "2"
    checks["cli_argv_builder"] = {"status": "PASS"}

    # 3. Native /completion SSE termination validation:
    # A) Valid: with [DONE] and one finish_reason
    s_done = ('data: {"content": "hello", "stop": false}\n'
              'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}\n'
              'data: [DONE]\n').encode()
    r_done = parse_sse_stream(s_done)
    assert r_done["error"] is None, f"valid with [DONE] rejected: {r_done['error']}"
    assert r_done["assembled_content"] == "hello world"

    # B) Valid: native /completion terminating WITHOUT [DONE] but exactly one finish_reason
    s_nodone = ('data: {"content": "hello", "stop": false}\n'
                'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}\n').encode()
    r_nodone = parse_sse_stream(s_nodone)
    assert r_nodone["error"] is None, f"valid native without [DONE] rejected: {r_nodone['error']}"
    assert r_nodone["assembled_content"] == "hello world"

    # C) Adversarial: multiple finish events MUST fail
    s_multi = ('data: {"choices": [{"delta": {"content": "a"}, "finish_reason": "stop"}]}\n'
               'data: {"choices": [{"delta": {"content": "b"}, "finish_reason": "length"}]}\n'
               'data: [DONE]\n').encode()
    assert "multiple finish events" in parse_sse_stream(s_multi)["error"]

    # D) Adversarial: multiple [DONE] events MUST fail
    s_multidone = ('data: {"choices": [{"delta": {"content": "a"}, "finish_reason": "stop"}]}\n'
                   'data: [DONE]\n'
                   'data: [DONE]\n').encode()
    assert "duplicate [DONE]" in parse_sse_stream(s_multidone)["error"]

    # E) Adversarial: truncated stream (no finish, no [DONE]) MUST fail
    s_trunc = 'data: {"content": "incomplete..."}\n'.encode()
    assert "truncated" in parse_sse_stream(s_trunc)["error"]

    checks["sse_native_termination"] = {"status": "PASS"}

    # 4. Resume, mismatch refusal, timeout persistence, atomic aggregation
    tmp = tempfile.mkdtemp(prefix="matrix-selftest-")
    try:
        prof_path = os.path.join(tmp, "profiles.json")
        with open(prof_path, "w") as f:
            json.dump(prof_stub, f, sort_keys=True)
        corpus_path = os.path.join(tmp, "corpus.jsonl")
        with open(corpus_path, "w") as f:
            for r in corpus_recs:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        out_dir = os.path.join(tmp, "out")

        def make_args(**kw: Any) -> argparse.Namespace:
            base = dict(profile=prof_path, baseline="", prompts=corpus_path,
                        profiles="B0", backend="cli", server_url="http://127.0.0.1:8080",
                        binary="/bin/echo", model="/nonexistent.gguf", calibration="",
                        template="t", seed=7, temperature=0.0, top_p=1.0, top_k=1,
                        n_predict=32, timeout=60, out_dir=out_dir, target_profile="B0",
                        cli_template="", dry_run=True, self_test=False, schema=False)
            base.update(kw)
            return argparse.Namespace(**base)

        runner = MatrixRunner(make_args())
        corpus, errs = load_corpus(corpus_path)
        first = runner.run_profile("B0", corpus, errs)
        second = runner.run_profile("B0", corpus, errs)
        assert len(second["records"]) == 3, "resume duplicated items"
        checks["resume_no_duplicate"] = {"status": "PASS"}

        # Mismatch refusal
        per = {"B0": first, "E1": copy.deepcopy(first)}
        per["E1"]["manifest"]["sampling_fingerprint"] = "tampered"
        ev = runner.aggregate(per)
        assert ev["matrix_verdict"] == "NOT_EVALUATED"
        checks["mismatch_refusal"] = {"status": "PASS"}

        # Timeout persistence
        out, err, meta = run_cli_backend(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
        assert meta.get("error", "").startswith("timeout")
        assert b"TIMEOUT" in err
        checks["timeout_persistence"] = {"status": "PASS"}

        # Mode pinning (section 15.5): R dense, SR, off exact
        assert require_profile_schema({"version": 2}) == 2
        for bad_v in (None, 1, 3, "2"):
            try:
                require_profile_schema({} if bad_v is None else {"version": bad_v})
                raise SystemExit(f"schema version {bad_v!r} not rejected")
            except ValueError as e:
                assert "Migrate" in str(e), str(e)
        fp_v2 = profile_fingerprint({"a": 1}, 2)
        assert fp_v2 != profile_fingerprint({"a": 1}, 1)
        assert fp_v2 == profile_fingerprint({"a": 1}, 2)
        # run_profile with a v1 profile fails closed into fail_profile (non-synthetic)
        v1_stub = copy.deepcopy(prof_stub)
        v1_stub["version"] = 1
        v1_path = os.path.join(tmp, "profiles-v1.json")
        with open(v1_path, "w") as f:
            json.dump(v1_stub, f)
        r_v1 = MatrixRunner(make_args(profile=v1_path))
        c_v1, e_v1 = load_corpus(corpus_path)
        f_v1 = r_v1.run_profile("B0", c_v1, e_v1)
        assert f_v1["records"] == []
        assert f_v1["manifest"]["returncode"] == 2
        assert "schema version" in f_v1["manifest"]["profile_error"]
        assert "__synthetic_test_fixture__" not in f_v1["manifest"]
        a_v1 = make_args(profile=v1_path, backend="cli")
        r_v1b = MatrixRunner(a_v1)
        ok_v1, errs_v1, _ = preflight(a_v1, ["B0"], r_v1b.profile_data, [])
        assert not ok_v1 and any("schema version" in e for e in errs_v1), errs_v1
        checks["schema_versioning"] = {"status": "PASS"}
        assert check_profile_mode("R2", {"name": "R2", "mode": "dense",
            "xkv_enabled": True, "sr_enabled": False,
            "dense_tiled_reconstruction": True}) == "dense"
        assert check_profile_mode("E1", {"name": "E1", "mode": "sr",
            "xkv_enabled": True, "sr_enabled": True,
            "dense_tiled_reconstruction": False}) == "sr"
        for bad_name, bad_spec in [
            ("R0", {"name": "R0", "mode": "sr", "xkv_enabled": True, "sr_enabled": True}),
            ("B0", {"name": "B0", "xkv_enabled": True, "mode": "off"}),
            ("E0", {"name": "E0", "mode": "dense", "xkv_enabled": True}),
            ("R3", {"name": "R3", "mode": "sr", "xkv_enabled": True,
                      "sr_enabled": True, "dense_tiled_reconstruction": True}),
        ]:
            try:
                check_profile_mode(bad_name, bad_spec)
                raise SystemExit(f"mode violation not rejected: {bad_name}")
            except ValueError:
                pass
        checks["mode_pinning"] = {"status": "PASS"}

        # Audit argv: store-mib 0 default, draft-n-max 2 default, FullKV exactness
        argv_e1b = build_cli_argv("/bin/llama-cli", "/m.gguf", "hi", sampling, None,
                                   profile_spec=prof_stub["profiles"]["E1"],
                                   calibration_path="/cal.bin")
        assert argv_e1b[argv_e1b.index("--xkv-store-mib") + 1] == "0"
        assert argv_e1b[argv_e1b.index("--spec-draft-n-max") + 1] == "2"
        argv_b0b = build_cli_argv("/bin/llama-cli", "/m.gguf", "hi", sampling, None,
                                   profile_spec=prof_stub["profiles"]["B0"],
                                   calibration_path="/cal.bin")
        for f in ("--xkv-store-mib", "--xkv-workspace-mib", "--xkv-factor-a-k",
                    "--xkv-sr-budget", "--xkv-decode-cache-mib"):
            assert f not in argv_b0b, f"FullKV baseline leaks {f}"
        assert argv_b0b[argv_b0b.index("--xkv") + 1] == "off"
        srv = build_server_argv("/bin/llama-server", "/m.gguf", 8090,
                                 prof_stub["profiles"]["E1"], "/cal.bin",
                                 {"ctx_size": 262144, "total_kv": "auto",
                                  "gpu_layers": None, "device": None,
                                  "n_parallel": None}, [])
        assert srv[srv.index("-c") + 1] == "262144"
        assert srv[srv.index("--total-kv") + 1] == "auto"
        assert "-dev" not in srv  # device IDs stay explicit/optional
        assert srv[srv.index("-ngl") + 1] == "all"  # portable default
        assert "--metrics" in srv  # launch always enables /metrics
        assert srv[srv.index("--xkv-seed") + 1] == "6362273814452121649"
        assert srv[srv.index("-np") + 1] == "6"  # per-profile pin wins
        srv_user = build_server_argv("/bin/llama-server", "/m.gguf", 8091,
                                     prof_stub["profiles"]["E1"], "/cal.bin",
                                     {"ctx_size": 262144, "total_kv": "auto",
                                      "gpu_layers": "all", "device": None,
                                      "n_parallel": 9}, ["--metrics", "--verbose"])
        assert srv_user.count("--metrics") == 1  # user args deduped
        assert srv_user[srv_user.index("-np") + 1] == "6"  # spec beats cfg
        srv_r = build_server_argv("/bin/llama-server", "/m.gguf", 8092,
                                  {"name": "Rx", "xkv_enabled": True, "mode": "dense"},
                                  None,
                                  {"ctx_size": 262144, "total_kv": "auto",
                                   "gpu_layers": "all", "device": None,
                                   "n_parallel": 1}, [])
        assert srv_r[srv_r.index("-np") + 1] == "1"  # R/B ablations via cfg
        assert srv[srv.index("--xkv-store-mib") + 1] == "0"
        assert "--xkv" in srv and srv[srv.index("--xkv") + 1] == "sr"
        checks["audit_argv"] = {"status": "PASS"}

        # Model default comes from profile-root model.path (CLI builder uses root config)
        prof2 = {"model": {"path": "/root/model.gguf"}, "profiles": prof_stub["profiles"]}
        pp2 = os.path.join(tmp, "profiles2.json")
        with open(pp2, "w") as f:
            json.dump(prof2, f)
        r2 = MatrixRunner(make_args(profile=pp2, model=""))
        assert r2.args.model == "/root/model.gguf"
        assert r2.model_source == "profile-root:model.path"
        checks["model_root_default"] = {"status": "PASS"}

        # Server mapping preflight: shared URL and duplicates rejected
        a_multi = make_args(backend="server", profiles="B0,E1",
                            server_url="http://127.0.0.1:8080", server_urls="")
        r_multi = MatrixRunner(a_multi)
        ok_m, errs_m, _ = preflight(a_multi, r_multi.profiles, r_multi.profile_data, [])
        assert not ok_m and any("server-urls" in e for e in errs_m), errs_m
        a_dup = make_args(backend="server", profiles="B0,E1",
                          server_urls="B0=http://x:1,E1=http://x:1")
        r_dup = MatrixRunner(a_dup)
        ok_d, errs_d, _ = preflight(a_dup, r_dup.profiles, r_dup.profile_data, [])
        assert not ok_d and any("duplicate" in e for e in errs_d), errs_d
        # Missing model artifact fails closed and is NOT synthetic
        a_nom = make_args(backend="cli", binary="/bin/echo",
                          model="/nonexistent-model.gguf")
        r_nom = MatrixRunner(a_nom)
        ok_n, errs_n, checks_n = preflight(a_nom, ["B0"], r_nom.profile_data, [])
        assert not ok_n, "missing model must fail preflight"
        ev_pf = write_preflight_failure(a_nom, ["B0"], r_nom.profile_data,
                                         errs_n, checks_n, [])
        assert ev_pf["matrix_verdict"] == "NOT_EVALUATED"
        assert ev_pf["__synthetic_test_fixture__"] is False
        assert ev_pf["xkv_effective_profile"] == "not_evaluated"
        checks["preflight_fail_closed"] = {"status": "PASS"}

        # Server identity via in-process fixture server
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        routes: Dict[str, Any] = {}

        class _H(BaseHTTPRequestHandler):
            def do_GET(self):
                body, ctype = routes.get(self.path, ("{}", "application/json"))
                data = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), _H)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{port}"
            e1spec = prof_stub["profiles"]["E1"]

            CALIB_FIX = "e95dae507d1f4a64e29be160c5281f8a4308a3332dc9c9176e1a3a0af32e50e2"
            BASE_FIX = {"deployed_artifacts": {"triattention_calibration": {
                "historical_expected_sha256": CALIB_FIX}}}
            SEED_FIX = "6362273814452121649"
            STORAGE_E1 = "tq-factors-landmarks"

            def set_routes(eff=STORAGE_E1, req=STORAGE_E1, model="/m.gguf",
                           model_sha="deadbeef", mode="sr", calib=CALIB_FIX,
                           ratio="3/32", window="128", frontier="strong",
                           draft=2, seed=SEED_FIX, slots=6,
                           ak="turbo4_0", bk="turbo4_0", av="turbo4_0", bv="turbo4_0", lm="q8_0",
                           factorizer="vulkan", balance="sqrt",
                           tri_series=True, rerot_series=True, profile_info=True):
                props: Dict[str, Any] = {"model_path": model}
                if model_sha is not None:
                    props["model_sha256"] = model_sha
                if slots is not None:
                    props["total_slots"] = slots
                if mode is not None:
                    props["xkv_mode"] = mode
                if calib is not None:
                    props["tri_calibration_sha256"] = calib
                if ratio is not None:
                    props["triattention_ratio"] = ratio
                if window is not None:
                    props["tri_recent_window"] = window
                if frontier is not None:
                    props["rerot_frontier"] = frontier
                if draft is not None:
                    props["spec_draft_n_max"] = draft
                    props["speculative"] = {"types": ["draft-mtp"]}
                if ak is not None:
                    props["a_k_codec"] = ak
                if bk is not None:
                    props["b_k_codec"] = bk
                if av is not None:
                    props["a_v_codec"] = av
                if bv is not None:
                    props["b_v_codec"] = bv
                if lm is not None:
                    props["landmark_codec"] = lm
                if factorizer is not None:
                    props["factorizer"] = factorizer
                if balance is not None:
                    props["factor_balance"] = balance
                routes["/props"] = (json.dumps(props), "application/json")
                lines = ['llamacpp:xkv_source_info{fingerprint="aa"} 1']
                if mode is not None:
                    lines.append(f'llamacpp:xkv_mode{{mode="{mode}"}} 1')
                if factorizer is not None:
                    lines.append(f'llamacpp:xkv_factorizer_info{{factorizer="{factorizer}"}} 1')
                if balance is not None:
                    lines.append(f'llamacpp:xkv_balance_info{{balance="{balance}"}} 1')
                if ak is not None or bk is not None or av is not None or bv is not None or lm is not None:
                    cl = []
                    if ak: cl.append(f'a_k="{ak}"')
                    if bk: cl.append(f'b_k="{bk}"')
                    if av: cl.append(f'a_v="{av}"')
                    if bv: cl.append(f'b_v="{bv}"')
                    if lm: cl.append(f'landmark="{lm}"')
                    lines.append(f'llamacpp:xkv_codecs_info{{{",".join(cl)}}} 1')
                if calib is not None:
                    lines.append(f'llamacpp:tri_calibration_info{{sha256="{calib}"}} 1')
                if ratio is not None or window is not None:
                    rl = []
                    if ratio is not None:
                        rl.append(f'ratio="{ratio}"')
                    if window is not None:
                        rl.append(f'recent_window="{window}"')
                    lines.append(f'llamacpp:tri_config_info{{{ ",".join(rl)}}} 1')
                if frontier is not None:
                    lines.append(f'llamacpp:rerot_frontier{{mode="{frontier}"}} 1')
                if seed is not None:
                    lines.append(f'llamacpp:xkv_seed_info{{seed="{seed}"}} 1')
                if profile_info:
                    lines.append(f'llamacpp:xkv_profile_info{{requested="{req}",effective="{eff}"}} 1')
                if tri_series:
                    lines.append("llamacpp:tri_drain_total 1")
                if rerot_series:
                    lines.append("llamacpp:rerot_people_capacity 3")
                routes["/metrics"] = ("\n".join(lines), "text/plain")

            def ident(**kw):
                return verify_server_identity(base, "E1", e1spec, "/m.gguf",
                                              30, BASE_FIX, "deadbeef", **kw)

            set_routes()
            ok, errs, obs = ident()
            assert ok, errs
            assert obs["matrix_profile"] == "E1"
            assert obs["total_slots"] == 6
            # matrix labels on the wire are rejected: storage names required
            set_routes(eff="E1", req="E1")
            ok2, errs2, _ = ident()
            assert not ok2 and any("effective storage" in e for e in errs2), errs2
            # storage mismatch (silent degradation to another storage profile)
            set_routes(eff="tq-factors")
            ok2b, errs2b, _ = ident()
            assert not ok2b and any("effective storage" in e for e in errs2b), errs2b
            set_routes(req="reference")
            ok2c, errs2c, _ = ident()
            assert not ok2c and any("requested storage" in e for e in errs2c), errs2c
            # checksum identity: sha mismatch / missing fail; path alone never decides
            set_routes(model_sha="00")
            ok3, errs3, _ = ident()
            assert not ok3 and any("SHA-256 mismatch" in e for e in errs3), errs3
            set_routes(model_sha=None)
            ok3b, errs3b, _ = ident()
            assert not ok3b and any("no model artifact SHA" in e for e in errs3b), errs3b
            set_routes(model="/other.gguf")  # path differs, checksum matches
            ok3c, errs3c, _ = ident()
            assert ok3c, errs3c  # path recorded/normalized, never substituted
            # slots: reduction to recurrent fit allowed, increase rejected
            set_routes(slots=8)
            ok_np, errs_np, _ = ident()
            assert not ok_np and any("exceeds" in e for e in errs_np), errs_np
            set_routes(slots=3)
            ok_np2, errs_np2, obs_np2 = ident()
            assert ok_np2 and "recurrent fit" in obs_np2.get("parallelism_note", ""), (errs_np2, obs_np2)
            # xkv mode via coordinated field
            set_routes(mode="dense")
            ok_m, errs_m, _ = ident()
            assert not ok_m and any("xkv mode mismatch" in e for e in errs_m), errs_m
            set_routes(mode=None)
            ok_m2, errs_m2, _ = ident()
            assert not ok_m2 and any("no xkv mode" in e for e in errs_m2), errs_m2
            # codec mismatch
            set_routes(ak="f16")
            ok_c, errs_c, _ = ident()
            assert not ok_c and any("a_k_codec mismatch" in e for e in errs_c), errs_c
            set_routes(lm="turbo4_0")
            ok_lm, errs_lm, _ = ident()
            assert not ok_lm and any("landmark_codec mismatch" in e for e in errs_lm), errs_lm
            set_routes(factorizer="cpu-reference")
            ok_f, errs_f, _ = ident()
            assert not ok_f and any("factorizer mismatch" in e for e in errs_f), errs_f
            set_routes(balance="diagonal")
            ok_b, errs_b, _ = ident()
            assert not ok_b and any("factor_balance mismatch" in e for e in errs_b), errs_b
            # tri strict: calib / ratio / window missing or mismatched
            set_routes(calib=None)
            assert not ident()[0]
            set_routes(calib="00")
            _, errs_t, _ = ident()
            assert any("calibration mismatch" in e for e in errs_t), errs_t
            set_routes(ratio="1/2")
            _, errs_r, _ = ident()
            assert any("ratio mismatch" in e for e in errs_r), errs_r
            set_routes(window=None)
            assert not ident()[0]
            # rerot frontier + mtp draft strict
            set_routes(frontier="lag1")
            _, errs_f, _ = ident()
            assert any("frontier mismatch" in e for e in errs_f), errs_f
            set_routes(frontier=None)
            assert not ident()[0]
            set_routes(draft=3)
            _, errs_d, _ = ident()
            assert any("draft max mismatch" in e for e in errs_d), errs_d
            set_routes(draft=None)
            assert not ident()[0]
            # seed pinned and verified
            set_routes(seed="43")
            _, errs_s, _ = ident()
            assert any("xkv_seed mismatch" in e for e in errs_s), errs_s
            set_routes(seed=None)
            assert not ident()[0]
            # off baseline: absent storage ok, foreign storage rejected
            b0spec = prof_stub["profiles"]["B0"]
            set_routes(profile_info=False, mode=None, frontier=None, draft=None,
                       seed=None, slots=None)
            ok4, errs4, _ = verify_server_identity(base, "B0", b0spec, "/m.gguf",
                                                   30, BASE_FIX, "deadbeef")
            assert ok4, errs4
            set_routes(eff=STORAGE_E1, profile_info=True, mode=None, frontier=None,
                       draft=None, seed=None, slots=None)
            ok5, errs5, _ = verify_server_identity(base, "B0", b0spec, "/m.gguf",
                                                   30, BASE_FIX, "deadbeef")
            assert not ok5 and any("foreign effective storage" in e for e in errs5), errs5
            # bare counters never decide: full strict config without drain
            # counters still passes identity
            set_routes(tri_series=False)
            ok6, errs6, _ = ident()
            assert ok6, errs6
            checks["server_identity"] = {"status": "PASS"}
        finally:
            srv.shutdown()

        # Checksum resume: tampered line is dropped and re-run, never duplicated
        rdry = MatrixRunner(make_args(dry_run=True))
        c0, e0 = load_corpus(corpus_path)
        p1 = rdry.run_profile("B0", c0, e0)
        assert len(p1["records"]) == 3
        rp = os.path.join(out_dir, "B0", "results.jsonl")
        rlines = open(rp).read().splitlines()
        assert len(rlines) == 3
        bad = json.loads(rlines[1])
        bad["content"] = "TAMPERED"
        rlines[1] = json.dumps(bad, sort_keys=True)
        with open(rp, "w") as f:
            f.write("\n".join(rlines) + "\n")
        p2 = rdry.run_profile("B0", c0, e0)
        assert len(p2["records"]) == 3
        assert p2["manifest"]["resume"]["checksum_dropped"] == 1, p2["manifest"]["resume"]
        assert len(open(rp).read().splitlines()) == 3
        checks["checksum_resume"] = {"status": "PASS"}

        # Final-answer extraction: reasoning traps cannot auto-pass
        out_trap = ("Some reasoning mentioning 42 as a wrong guess.\n"
                    "More steps...\n\\boxed{43}")
        r_trap = score_final_answer("aime", out_trap, "42")
        assert r_trap["status"] == "SCORED" and r_trap["passed"] is False, r_trap
        assert score_item("includes", out_trap, "42")["passed"] is True  # old gap
        r_ok = score_final_answer("aime", "work...\n\\boxed{042}", "42")
        assert r_ok["passed"] is True, r_ok
        r_none = score_final_answer("aime", "no marker here 42", "42")
        assert r_none["status"] == "NOT_EVALUATED", r_none
        r_m = score_final_answer("math", "blah\n#### $\\frac{1}{2}$", "\\frac{1}{2}")
        assert r_m["passed"] is True, r_m
        r_m2 = score_final_answer("math", "Final answer: 1/2", "\\frac{1}{2}")
        assert r_m2["status"] == "SCORED" and r_m2["passed"] is False, r_m2
        checks["final_answer_extraction"] = {"status": "PASS"}

        # External versioned grader: pass / malformed / missing all handled
        gdir = os.path.join(tmp, "grader")
        os.makedirs(gdir)
        gp = os.path.join(gdir, "grader.py")
        with open(gp, "w") as f:
            f.write("#!/usr/bin/env python3\nimport sys, json\n"
                    "a = sys.argv\n"
                    "out = open(a[a.index('--output')+1]).read()\n"
                    "ref = open(a[a.index('--reference')+1]).read()\n"
                    "print(json.dumps({'passed': out.strip() == ref.strip()}))\n")
        os.chmod(gp, 0o755)
        g = {"path": gp, "sha256": sha256_file(gp)}
        gr = score_final_answer("math", "x\n\\boxed{7}", "7", g)
        assert gr["status"] == "SCORED" and gr["passed"] is True, gr
        assert gr["grader_sha256"] == g["sha256"]
        gb = os.path.join(gdir, "bad.py")
        with open(gb, "w") as f:
            f.write("#!/usr/bin/env python3\nprint('not json')\n")
        os.chmod(gb, 0o755)
        gr2 = score_final_answer("math", "x\n\\boxed{7}", "7",
                                 {"path": gb, "sha256": "x"})
        assert gr2["status"] == "NOT_EVALUATED" and gr2["passed"] is None, gr2
        gr3 = score_final_answer("math", "x\n\\boxed{7}", "7",
                                 {"path": "/nonexistent-grader", "sha256": None})
        assert gr3["status"] == "NOT_EVALUATED", gr3
        checks["external_grader"] = {"status": "PASS"}

        # Per-question preservation + atomic aggregation
        ev2 = runner.aggregate({"B0": first})
        assert len(ev2["per_item"]) == 3
        assert os.path.exists(os.path.join(out_dir, "B0", "manifest.json"))
        checks["per_question_preservation"] = {"status": "PASS"}
        checks["atomic_aggregation"] = {"status": "PASS"}

        # Fail closed missing fields
        for k in ("xkv_factored_baseline_byte_coverage", "xkv_net_saved_fraction",
                  "xkv_live_bytes", "quality_ppl_degradation_pct"):
            assert ev2[k] == "not_evaluated"
        checks["missing_fields_fail_closed"] = {"status": "PASS"}

        assert ev2["__synthetic_test_fixture__"] is True
        checks["synthetic_rejected_by_gate"] = {"status": "PASS"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"status": "PASS", "checks": checks}


# ---------------------------------------------------------------- CLI

SCHEMA_HELP = (
    "Evidence schema (--schema prints EVIDENCE_SCHEMA_JSON): "
    "matrix_verdict PASS|FAIL|NOT_EVALUATED; per_item preserves every prompt "
    "record; byte/coverage/allocator/quality keys are 'not_evaluated' here and "
    "must be filled by collect-runtime-evidence.py; synthetic runs carry "
    "__synthetic_test_fixture__=true and are rejected by compression-gate.py "
    "production runs."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resumable xKV-SR evaluation matrix runner (B0/R0-R5/E0-E2). " + SCHEMA_HELP)
    p.add_argument("--profile", default="docs/xkv/profiles/ornith-r2.json")
    p.add_argument("--baseline", default="docs/xkv/baseline.json")
    p.add_argument("--prompts", default="", help="Fixed prompt corpus JSONL path")
    p.add_argument("--profiles", default="B0,E1",
                   help="Comma list from " + ",".join(MATRIX_PROFILES))
    p.add_argument("--backend", choices=("server", "cli", "ppl"), default="server")
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    p.add_argument("--server-urls", default="",
                   help="Per-profile mapping 'B0=URL,E1=URL'; required for >1 profile "
                   "on server backend (a single shared URL would mislabel configs)")
    p.add_argument("--server-launch", action="store_true",
                   help="Spawn one llama-server per profile with exact launch argv, "
                   "verify identity, terminate afterwards")
    p.add_argument("--server-binary", default="", help="llama-server binary (launch mode)")
    p.add_argument("--server-port-base", type=int, default=8090)
    p.add_argument("--server-ctx", type=int, default=262144,
                   help="Exact production context entering launch argv")
    p.add_argument("--server-total-kv", default="auto",
                   help="Unified KV capacity entering launch argv (production: auto)")
    p.add_argument("--server-np", type=int, default=None)
    p.add_argument("--gpu-layers", default="all",
                   help="Passed as -ngl (default: portable 'all'; device IDs stay explicit; "
                   "any non-'all' override marks the comparison non-equivalent/NOT_EVALUATED)")
    p.add_argument("--device", default=None,
                   help="Passed as -dev only when set (never defaulted: no hardware IDs)")
    p.add_argument("--server-arg", action="append", default=[],
                   help="Extra launch arg, one token each (repeatable, no shell)")
    p.add_argument("--server-launch-timeout", type=int, default=600)
    p.add_argument("--math-grader", default="",
                   help="Versioned external MATH grader executable (argv, no shell); "
                   "sha256 recorded per verdict; failures are NOT_EVALUATED")
    p.add_argument("--binary", default="", help="llama-cli binary (cli/ppl backends)")
    p.add_argument("--model", default="", help="Model .gguf path (hash recorded)")
    p.add_argument("--calibration", default="", help="TriAttention calibration path")
    p.add_argument("--template", default="", help="Chat template override (in sampling fp)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--n-predict", type=int, default=512)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out-dir", default="docs/xkv/matrix")
    p.add_argument("--target-profile", default="",
                   help="Aggregate target (default: first --profiles entry)")
    p.add_argument("--cli-template", default="",
                   help="ppl argv template, e.g. '{binary} -m {model} -p {prompt}'")
    p.add_argument("--dry-run", action="store_true",
                   help="Deterministic plan only; synthetic outputs, no subprocess/HTTP")
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

    runner = MatrixRunner(args)
    corpus_items: List[Dict[str, Any]] = []
    corpus_errors: List[str] = []
    if args.prompts:
        corpus_items, corpus_errors = load_corpus(args.prompts)
    else:
        corpus_errors = ["no --prompts corpus supplied: matrix is NOT_EVALUATED"]

    ok, pf_errors, pf_checks = preflight(args, runner.profiles, runner.profile_data,
                                          corpus_errors, runner.baseline_data)
    server_info = {
        "mode": ("launch" if args.server_launch else
                   ("urls" if getattr(args, "server_urls", "") else "single")),
        "urls": dict(runner.server_urls),
        "launch": bool(args.server_launch),
        "runtime_cfg": dict(runner.runtime_cfg),
    }
    plan = build_plan(runner.profile_data, runner.profiles, corpus_items,
                      runner.sampling, args.backend, args.model,
                      server_info, runner.model_source)
    atomic_write_json(os.path.join(args.out_dir, "plan.json"), plan)

    if not ok and not args.dry_run:
        evidence = write_preflight_failure(args, runner.profiles, runner.profile_data,
                                            pf_errors, pf_checks, corpus_items)
        atomic_write_json(os.path.join(args.out_dir, "evidence.json"), evidence)
        print(json.dumps({"preflight_ok": False, "errors": pf_errors,
                          "evidence": os.path.join(args.out_dir, "evidence.json")},
                         indent=2, sort_keys=True))
        sys.exit(2)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan,
                          "corpus_errors": corpus_errors,
                          "preflight": {"ok": ok, "errors": pf_errors},
                          "__synthetic_test_fixture__": True}, indent=2, sort_keys=True))
        return

    per_profile: Dict[str, Any] = {}
    for profile in runner.profiles:
        per_profile[profile] = runner.run_profile(profile, corpus_items, corpus_errors)
    if not args.target_profile:
        args.target_profile = runner.profiles[0]
    evidence = runner.aggregate(per_profile)
    atomic_write_json(os.path.join(args.out_dir, "evidence.json"), evidence)
    print(json.dumps({"plan_sha256": plan["plan_sha256"],
                      "matrix_verdict": evidence["matrix_verdict"],
                      "reasons": evidence["matrix_reasons"],
                      "evidence": os.path.join(args.out_dir, "evidence.json")},
                     indent=2, sort_keys=True))
    sys.exit(0 if evidence["matrix_verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
