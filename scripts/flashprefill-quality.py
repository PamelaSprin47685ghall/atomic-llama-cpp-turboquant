#!/usr/bin/env python3
"""Paired quality/performance harness for FlashPrefill V2 (PREFILL.md sections 18-19).

Compares four explicit configurations with RERoT fixed OFF, then optional
paired RERoT/MTP configurations, using one alternating request driver::

    A = FullKV + dense prefill            (reference)
    B = FullKV + FlashPrefill sparse
    C = Tri(3/32) + dense prefill
    D = Tri(3/32) + FlashPrefill sparse

Optional extras (each names a ``parent`` it mirrors, e.g. ``"D+rerot"`` with
``"parent": "D"``) isolate RERoT/MTP effects after the base four are compared.
MTP is only compared between legal identical-sampling stages; the harness never
compares answers across different sampling strategies.

Two evidence modes per configuration (mixed matrices are allowed):

* live: the harness POSTs to an explicit endpoint (chat-completions style,
  same convention as ``scripts/rerot-throughput-gate.py``);
* offline: the harness consumes a harness-defined ``responses_jsonl``
  file with pre-recorded raw responses instead of issuing HTTP requests.

The smallest complete flow is a single alternating loop: for each repetition
round the harness rotates the configuration order, issues (or replays) every
dataset item once per configuration, saves raw evidence, then scores every
response. Repetition 0 is the primary quality sample; all repetitions double
as independent performance samples. Nothing is averaged away: raw requests,
responses, scores, timings and resource series are all preserved, and any
missing cell blocks PASS.

CLI examples::

    # Validate plan + dataset + pair equality without touching any endpoint.
    python3 scripts/flashprefill-quality.py \\
        --plan plan.json --dataset dataset.jsonl \\
        --output-dir out/quality-v1 --dry-run

    # Live paired run (per-config endpoints/auth come from the plan file).
    python3 scripts/flashprefill-quality.py \\
        --plan plan.json --dataset dataset.jsonl \\
        --output-dir out/quality-v1

    # Fully offline scoring of pre-recorded evidence rows.
    # (Each config in plan.json points at its own responses_jsonl file.)
    python3 scripts/flashprefill-quality.py \\
        --plan plan-offline.json --dataset dataset.jsonl \\
        --output-dir out/quality-v1

Plan JSON schema (explicit JSON input; thresholds must be declared BEFORE
execution, otherwise the harness refuses to run)::

    {
      "plan_id": "flashprefill-quality-v1",
      "model": "some-quant-gguf",
      "template": "chatml:{prompt}",
      "seed": 1234,
      "max_tokens": 512,
      "temperature": 0.0,
      "base_url": null,
      "endpoint_path": "/v1/chat/completions",
      "warmup_requests": 1,
      "repetitions": 5,
      "capacity_mode": "fixed-capacity | auto-fit | both",
      "thresholds": {
        "overall_max_drop": 0.01,
        "per_task_max_drop": 0.02,
        "retrieval_max_drop": 0.02
      },
      "provenance": {"repo_commit": "...", "model_hash": "..."},
      "configs": {
        "A": {"label": "FullKV+dense",
              "switches": {"kv": "full", "prefill": "dense"},
              "base_url": "http://127.0.0.1:8081",
              "startup": {"argv": ["./build/bin/llama-server", "-m",
                                   "model.gguf", "<FullKV+dense policy flags>"],
                          "rerot": "off"},
              "cache_dir": "/data/fp-q-A",
              "capacity_group": "fixed",
              "capacity": {"K": 0, "B": 0, "P": 0, "batch": 1},
              "responses_jsonl": null,
              "resources_jsonl": null},
        "B": {..., "startup": {"argv": ["./build/bin/llama-server", "-m",
                                        "model.gguf", "<FullKV+sparse policy flags>"],
                               "rerot": "off"}, ...},
        "C": {..., "startup": {"argv": ["./build/bin/llama-server", "-m",
                                        "model.gguf", "<Tri+dense policy flags>"],
                               "rerot": "off"}, ...},
        "D": {..., "startup": {"argv": ["./build/bin/llama-server", "-m",
                                        "model.gguf", "<Tri+sparse policy flags>"],
                               "rerot": "off"}, ...},
        "D+rerot": {"parent": "D", "label": "D+RERoT",
                    "switches": {"kv": "tri-3/32", "prefill": "sparse",
                                 "rerot": "on"},
                    "base_url": "http://127.0.0.1:8085",
                    "startup": {"argv": ["./build/bin/llama-server", "-m",
                                          "model.gguf", "<Tri+sparse+RERoT policy flags>"],
                                "rerot": "on"},
                    "cache_dir": "/data/fp-q-D-rerot",
                    "capacity_group": "fixed",
                    "capacity": {"K": 0, "B": 0, "P": 0, "batch": 1}}
      }
    }

Switches live in endpoint startup provenance, never in request fields.
FlashPrefill policy is immutable context-lifetime server configuration and
Tri flags are startup flags: there is NO per-request API for them, so the
harness sends BYTE-IDENTICAL request payloads to every configuration (common
base only) and evidences the A/B/C/D difference through the declared
startup commands. ``request_overrides`` and ``allowed_switch_keys`` are
REJECTED outright (exit 1): they would fake the comparison with invented
request fields, and an ``allowed_switch_keys`` default-union would
permissively bless sampling/model/template/budget drift.

Plan rules enforced by the harness:

* Every config MUST declare ``startup``: an object with ``argv`` (required,
  non-empty string list: the literal server startup command, recorded
  verbatim in ``commands.txt``/``manifest.json``/fingerprints for audit),
  optional ``config_file``/``config_sha256``/``notes``, and required
  machine-readable ``rerot`` (``"off"``|``"on"``). Startup ``argv`` MUST NOT
  contain secrets (tokens matching api-key/secret/password are refused so
  keys never enter evidence files).
* ``configs`` MUST contain ``A``, ``B``, ``C``, ``D``. The base four MUST
  declare ``startup.rerot == "off"`` and MUST NOT claim rerot in
  ``switches``. Any extra config MUST name an existing ``parent`` and MUST
  declare a ``startup`` blob that differs from its parent's; a child that
  switches rerot on requires ``startup.rerot == "on"`` with a parent whose
  ``startup.rerot == "off"``.
* RERoT rigor: an explicit ``rerot_frontier`` selects RERoT even when
  ``rerot=false``, so the harness checks ABSENCE, not truthiness. Every
  request payload sent (and every offline row's request echo, when present)
  MUST contain zero rerot keys (``rerot`` or any ``rerot_*``/``rerot-*``
  key); any occurrence aborts the run (exit 1).
* ``repetitions`` MUST be >= 5 (PREFILL 18.2: warmup, then at least 5
  independent alternating measurements).
* ``thresholds`` with all three keys is REQUIRED; the run is judged against
  these pre-declared values, never against post-hoc numbers.
* Fixed budgets/seeds/template (``seed``, ``max_tokens``, ``temperature``,
  ``template``, ``model``) are common to every configuration BY
  CONSTRUCTION: one builder makes every payload, and the harness asserts
  byte-identity across configurations per item. Any divergence is a
  pair-equality violation and aborts the run (exit 1).
* ``cache_dir`` values MUST be pairwise distinct: configurations never share
  a persistent-KV cache directory. Weights may be shared; runtime state may
  not.
* ``capacity_mode`` groups the performance reading: ``fixed-capacity`` means
  every ``fixed``-group config declares an identical ``capacity`` dict
  (K/B/P/batch/concurrency) and answers "is the algorithm itself cheaper";
  ``auto-fit`` records each side's independently fitted capacity and answers
  "what does the user actually get (TTFT, fitted slots, OOM risk)".
  ``both`` requires both readings to be present per config pair.
* Auth resolution per live config: ``api_key_env`` (env var name) first,
  then ``api_key``, then ``--api-key``, then ``$LLAMA_API_KEY``. Secrets are
  never written to any output file and never enter fingerprints.

Dataset JSONL schema (one object per line; stable ``item_id`` per case)::

    {"item_id": "aime24-001", "task": "aime24", "length_bucket": "4K",
     "prompt": "Find ...", "expected": "42",
     "scorer": {"type": "exact", "extract": "boxed", "normalize": "strip"},
     "skip_reason": null}

* ``task`` groups quality (e.g. ``aime24``, ``math500``, ``retrieval``,
  ``needle``, ``multiturn``, ``codeqa``, ``prod``). Tasks in
  ``retrieval``/``needle``/``ruler``/``longbench-retrieval`` use the
  ``retrieval_max_drop`` threshold; every other task uses
  ``per_task_max_drop``.
* ``length_bucket`` groups length-specific cliffs (``4K``/``16K``/``32K``/
  ``64K``/``128K`` or any stable label set used by the dataset).
* ``expected`` is the reference answer used by built-in scorers.
* ``scorer`` selects grading (see below). ``skip_reason`` (non-null) marks a
  pre-declared dataset-level skip: excluded from accuracy denominators,
  reported, never counted as a pass.

Answer scorers (explicit extraction first, then grading; unknown scorer
types and external-scorer failures are recorded as ERROR, never as success):

* ``exact``: ``{"type": "exact", "extract": "raw|boxed|answer_tag",
  "normalize": "strip|lower|spaceless"}``. Compares the extracted span
  against ``expected``.
* ``numeric``: ``{"type": "numeric", "extract": "...", "abs_tol": 1e-6,
  "rel_tol": 1e-9}``. Parses the LAST decimal/scientific number in the
  extracted span; passes when ``abs(a-b) <= abs_tol + rel_tol*|b|``.
* ``regex``: ``{"type": "regex", "extract": "raw",
  "pattern": "...", "flags": "IGNORECASE?"}``. Passes on ``re.search`` hit.
* ``external``: ``{"type": "external", "command": [...], "version": "...",
  "timeout": 60}``. Versioned domain-grading command for tasks needing real
  grading (proof checking, code execution, LLM judges). ``{output}``,
  ``{expected}``, ``{item_id}`` and ``{task}`` placeholders are substituted.
  The command MUST print JSON ``{"score": 0|1}`` on stdout; nonzero exit,
  timeout, bad JSON or missing ``score`` is an ERROR, never a default pass.
  There is deliberately NO fake math grader: AIME/MATH items must either use
  ``numeric``/``exact`` with an explicit ``extract`` step or route to an
  external versioned grader.

Offline ``responses_jsonl`` schema (HARNESS-DEFINED evidence rows, one file
per configuration -- this is NOT the MatrixHarness schema and MUST NOT be
treated as interchangeable with it; align field names with the MatrixHarness
owner before consuming its output)::

    {"item_id": "aime24-001", "config": "A", "rep": 0,
     "output": "...\\\\boxed{42}...",
     "finish_reason": "stop", "http_status": 200,
     "usage": {"completion_tokens": 64},
     "timings": {"predicted_n": 64, "predicted_ms": 1200.0,
                 "prompt_n": 128, "prompt_ms": 300.0}}

  Accepted aliases: ``item_id``/``case_id``/``case``/``id`` for the item,
  ``output``/``completion``/``assembled_text``/``text`` for the completion
  text, ``config``/``config_id``/``variant`` for the configuration,
  ``latency_s`` as fallback wall time, and an inline ``response_json`` body
  (parsed for text/timings/usage/finish when top-level fields are absent).
  Rows marked ``warmup: true`` are excluded from scores and perf.
  Aligned with scripts/flashprefill-matrix.py (additive only, no renames
  either side): rows carry ``rep`` (default 0); each row fills its
  (config, item, rep) slot and repeated rounds use distinct rep. Rows echo
  the verbatim sent payload so the zero-rerot-keys assertion is
  non-vacuous. ``response_json`` stays inline; ``raw_file`` is SSE-pointer
  only and is not consumed.
  A row MAY carry its sent request under ``request``/``payload``; when
  present the harness asserts it holds zero rerot keys (same rigor as live
  payloads). Missing rows are NOT RUN evidence gaps, never silent passes.

Per-config ``resources_jsonl`` (optional) supplies resource time series that
the harness merges verbatim into ``memory.jsonl`` with a ``config`` tag,
e.g. ``{"t": 12.3, "vram_bytes": 1, "rss_bytes": 2, "fitted_slots": 3}``.
Absent series are reported as NOT RUN; the harness never synthesizes them.
Likewise ``timings.ttft_ms`` is taken from server ``timings.prompt_ms`` when
present and recorded as NOT RUN (null) otherwise; TTFT is never fabricated
from wall time.

Output directory layout (PREFILL 18.6)::

    manifest.json              # provenance, fingerprints, capacity, thresholds
    commands.txt               # exact harness invocation + endpoints (no secrets)
    requests.jsonl             # every measured + warmup request (secret-free)
    responses.jsonl            # every raw response / replayed evidence / error
    scores.jsonl               # per-item scorer version, extraction, score
    timings.jsonl              # warmup flag, wall/TTFT/tokens, raw samples
    memory.jsonl               # supplied resource series or NOT RUN stub
    metrics-before-<CFG>.txt / metrics-after-<CFG>.txt  (live only; else stub)
    binary-libraries.sha256    # provenance passthrough or NOT RUN stub
    summary.json               # verdict, CIs, paired diffs, cliffs, checks

``summary.json`` carries per-configuration accuracy with Wilson 95% CIs,
paired differences vs ``A`` (discordant b/c counts, delta with 95% CI),
per-task and per-length accuracy tables with cliff flags, small-n warnings
(n < 30, e.g. AIME: flips are listed item-by-item instead of trusting bare
percentages), C-vs-A isolation notes (pre-existing Tri issues are reported
separately and never used to excuse D), and per-configuration perf medians
with raw samples. Status vocabulary: ``PASS``/``FAIL``/``SKIP`` per unit,
``NOT RUN`` for missing evidence, overall verdict ``PASS``/``FAIL``/
``INCOMPLETE``. INCOMPLETE (any missing cell, any SKIP/NOT RUN unit) can
never pass. SKIP cells are excluded from accuracy denominators and are
listed; they are never counted as passes.

Exit codes: 0 = PASS; 2 = FAIL or INCOMPLETE (details in summary.json);
1 = harness error (bad plan/dataset, pair-equality violation, shared cache
directory, unknown scorer type, duplicate evidence rows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
HARNESS_VERSION = "1.0.0"
HARNESS_NAME = "flashprefill-quality"

REQUIRED_CONFIGS = ("A", "B", "C", "D")
RETRIEVAL_TASKS = frozenset({"retrieval", "needle", "ruler", "longbench-retrieval"})
MIN_REPETITIONS = 5
SMALL_N = 30
WILSON_Z = 1.96

ITEM_ID_KEYS = ("item_id", "case_id", "case", "id")
TEXT_KEYS = ("output", "completion", "assembled_text", "text")
CONFIG_KEYS = ("config", "config_id", "variant")

STATUS_SCORED = "SCORED"
STATUS_ERROR = "ERROR"
STATUS_SKIP = "SKIP"
STATUS_NOT_RUN = "NOT RUN"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCOMPLETE = "INCOMPLETE"


class HarnessError(RuntimeError):
    """Fatal plan/dataset/evidence problem: exit 1."""


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HarnessError(f"cannot load JSON {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError as error:
                    raise HarnessError(
                        f"{path}:{lineno}: invalid JSON: {error}"
                    ) from error
                if not isinstance(obj, dict):
                    raise HarnessError(f"{path}:{lineno}: row must be an object")
                rows.append(obj)
    except OSError as error:
        raise HarnessError(f"cannot read {path}: {error}") from error
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def first_present(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ordered = sorted(xs)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def percentile(xs: list[float], pct: float) -> float | None:
    if not xs:
        return None
    ordered = sorted(xs)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def wilson(k: int, n: int, z: float = WILSON_Z) -> dict[str, float | None]:
    """Wilson score 95% interval for k/n (None when n == 0, never fabricated)."""
    if n <= 0:
        return {"acc": None, "ci_lo": None, "ci_hi": None, "n": 0}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return {
        "acc": p,
        "ci_lo": max(0.0, center - half),
        "ci_hi": min(1.0, center + half),
        "n": n,
    }


def paired_diff_ci(
    b: int, c: int, n: int, z: float = WILSON_Z
) -> dict[str, float | int | None]:
    """Paired accuracy delta (X minus A) with a normal-approx 95% CI.

    b = X right / A wrong, c = X wrong / A right, n = paired items.
    """
    if n <= 0:
        return {"delta": None, "ci_lo": None, "ci_hi": None, "b": b, "c": c, "n": n}
    delta = (b - c) / n
    discord = b + c
    if discord == 0:
        return {"delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "b": b, "c": c, "n": n}
    var = (discord - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(max(0.0, var))
    return {
        "delta": delta,
        "ci_lo": delta - z * se,
        "ci_hi": delta + z * se,
        "b": b,
        "c": c,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Extraction and scorers
# ---------------------------------------------------------------------------

def extract_span(output: str, method: str) -> tuple[str, str | None]:
    """Return (extracted, failure_reason). Empty extraction is reported, not hidden."""
    if method in ("raw", "", None):
        return output.strip(), None
    if method == "boxed":
        spans: list[str] = []
        for match in re.finditer(r"\\boxed\{", output):
            depth = 0
            start = match.end()
            i = match.start()
            # Brace-match from the opening brace of \boxed{.
            j = match.end() - 1
            depth = 0
            while j < len(output):
                if output[j] == "{":
                    depth += 1
                elif output[j] == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append(output[start:j])
                        break
                j += 1
        if not spans:
            return "", "boxed-extraction-empty"
        return spans[-1].strip(), None
    if method == "answer_tag":
        matches = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if not matches:
            return "", "answer-tag-extraction-empty"
        return matches[-1].strip(), None
    raise HarnessError(
        f"unknown extract method {method!r}; expected raw|boxed|answer_tag"
    )


def normalize_text(text: str, mode: str) -> str:
    if mode in ("strip", "", None):
        return text.strip()
    if mode == "lower":
        return text.strip().casefold()
    if mode == "spaceless":
        return re.sub(r"\s+", "", text.strip())
    raise HarnessError(
        f"unknown normalize mode {mode!r}; expected strip|lower|spaceless"
    )


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_last_number(text: str) -> tuple[float | None, str | None]:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None, "no-number-found"
    try:
        return float(matches[-1]), None
    except ValueError:
        return None, "number-parse-failed"


def score_exact(extracted: str, expected: str, scorer: dict[str, Any]) -> int:
    mode = scorer.get("normalize", "strip")
    return (
        1
        if normalize_text(extracted, mode) == normalize_text(expected, mode)
        else 0
    )


def score_numeric(extracted: str, expected: str, scorer: dict[str, Any]) -> tuple[int | None, str | None]:
    got, reason = parse_last_number(extracted)
    want, reason2 = parse_last_number(expected)
    if got is None:
        return None, reason or "no-number-in-output"
    if want is None:
        return None, "no-number-in-expected"
    abs_tol = float(scorer.get("abs_tol", 1e-6))
    rel_tol = float(scorer.get("rel_tol", 1e-9))
    if abs(got - want) <= abs_tol + rel_tol * abs(want):
        return 1, None
    return 0, None


def score_regex(extracted: str, scorer: dict[str, Any]) -> tuple[int, str | None]:
    pattern = scorer.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise HarnessError("regex scorer requires a non-empty 'pattern' string")
    flags = 0
    raw_flags = scorer.get("flags", "")
    if isinstance(raw_flags, str) and "IGNORECASE" in raw_flags:
        flags |= re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as error:
        raise HarnessError(f"regex scorer bad pattern: {error}") from error
    return (1, None) if compiled.search(extracted) else (0, None)


def score_external(
    output: str,
    item: dict[str, Any],
    scorer: dict[str, Any],
    default_timeout: float,
) -> tuple[int | None, str | None, str]:
    """Run a versioned external grader. Any failure is ERROR, never a pass."""
    command = scorer.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) for part in command
    ):
        raise HarnessError("external scorer requires 'command' as a non-empty string list")
    version = scorer.get("version")
    if not isinstance(version, str) or not version:
        raise HarnessError("external scorer requires an explicit 'version' string")
    timeout = float(scorer.get("timeout", default_timeout))
    subs = {
        "{output}": output,
        "{expected}": str(item.get("expected", "")),
        "{item_id}": str(item.get("item_id", "")),
        "{task}": str(item.get("task", "")),
    }
    argv = []
    for part in command:
        for key, value in subs.items():
            part = part.replace(key, value)
        argv.append(part)
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"external-scorer-launch-failed: {error}", version
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:500]
        return None, f"external-scorer-exit-{proc.returncode}: {detail}", version
    try:
        payload = json.loads((proc.stdout or "").strip())
    except ValueError:
        return None, "external-scorer-bad-json-stdout", version
    if not isinstance(payload, dict) or payload.get("score") not in (0, 1, True, False):
        return None, "external-scorer-missing-score", version
    reported_version = payload.get("version", version)
    return (1 if payload["score"] in (1, True) else 0), None, str(reported_version)


def score_item(
    item: dict[str, Any],
    output_text: str | None,
    scorer_error: str | None,
    default_timeout: float,
) -> dict[str, Any]:
    """Grade one response. Transport errors stay ERROR; scoring never invents success."""
    scorer = item.get("scorer", {})
    if not isinstance(scorer, dict):
        raise HarnessError(f"item {item.get('item_id')}: 'scorer' must be an object")
    scorer_type = scorer.get("type")
    base: dict[str, Any] = {
        "item_id": item.get("item_id"),
        "task": item.get("task"),
        "length_bucket": item.get("length_bucket"),
        "scorer_type": scorer_type,
        "expected": item.get("expected", ""),
    }
    if scorer_error is not None:
        base.update(
            {
                "score": None,
                "status": STATUS_ERROR,
                "extracted": "",
                "scorer_version": scorer.get("version", "builtin-1"),
                "failure_reason": scorer_error,
                "budget_exhausted": False,
            }
        )
        return base
    if scorer_type not in ("exact", "numeric", "regex", "external"):
        # No fake grading, no default success: unknown graders are errors.
        base.update(
            {
                "score": None,
                "status": STATUS_ERROR,
                "extracted": "",
                "scorer_version": scorer.get("version", "builtin-1"),
                "failure_reason": f"unknown-scorer-type: {scorer_type!r}",
                "budget_exhausted": False,
            }
        )
        return base
    text = output_text if isinstance(output_text, str) else ""
    if scorer_type == "external":
        score, reason, version = score_external(text, item, scorer, default_timeout)
        base.update(
            {
                "score": score,
                "status": STATUS_SCORED if reason is None else STATUS_ERROR,
                "extracted": text[:2000],
                "scorer_version": version,
                "failure_reason": reason,
            }
        )
        return base
    extract_method = scorer.get("extract", "raw")
    extracted, extraction_reason = extract_span(text, extract_method)
    if scorer_type == "exact":
        score = score_exact(extracted, str(item.get("expected", "")), scorer)
        reason = extraction_reason  # empty extraction scores 0, reason preserved
        base.update(
            {
                "score": score,
                "status": STATUS_SCORED,
                "extracted": extracted[:2000],
                "scorer_version": "builtin-exact-1",
                "failure_reason": reason,
            }
        )
        return base
    if scorer_type == "numeric":
        score, reason = score_numeric(extracted, str(item.get("expected", "")), scorer)
        if extraction_reason is not None and score is None:
            reason = extraction_reason + ";" + (reason or "")
        base.update(
            {
                "score": score,
                "status": STATUS_SCORED if score is not None else STATUS_ERROR,
                "extracted": extracted[:2000],
                "scorer_version": "builtin-numeric-1",
                "failure_reason": reason,
            }
        )
        return base
    score, reason = score_regex(extracted, scorer)
    base.update(
        {
            "score": score,
            "status": STATUS_SCORED,
            "extracted": extracted[:2000],
            "scorer_version": "builtin-regex-1",
            "failure_reason": reason or extraction_reason,
        }
    )
    return base


# ---------------------------------------------------------------------------
# HTTP helpers (same urllib convention as rerot-throughput-gate.py)
# ---------------------------------------------------------------------------

def post_json(
    url: str, payload: dict[str, Any], api_key: str, timeout: float
) -> tuple[float, int | None, dict[str, Any], str | None]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        return time.monotonic() - started, error.code, {}, raw[:4000]
    except (OSError, urllib.error.URLError) as error:
        return time.monotonic() - started, None, {}, f"transport: {error}"[:4000]
    wall = time.monotonic() - started
    try:
        return wall, status, json.loads(raw), None
    except ValueError:
        return wall, status, {}, raw[:4000]


def fetch_metrics_text(base_url: str, api_key: str, timeout: float) -> tuple[str, bool]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/metrics",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace"), True
    except (OSError, urllib.error.URLError) as error:
        return f"NOT RUN: metrics unavailable: {error}", False


def parse_chat_response(body: dict[str, Any], raw: str | None) -> tuple[str | None, str | None, dict[str, Any], dict[str, Any], str | None]:
    """Return (output_text, finish_reason, usage, timings, error)."""
    if not body:
        return None, None, {}, {}, raw or "empty-response-body"
    choices = body.get("choices", [])
    text: str | None = None
    finish: str | None = None
    if choices and isinstance(choices[0], dict):
        first = choices[0]
        message = first.get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            text = message["content"]
        elif isinstance(first.get("text"), str):
            text = first["text"]
        finish = first.get("finish_reason")
    for key in ("response", "content", "output"):
        if text is None and isinstance(body.get(key), str):
            text = body[key]
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    timings = body.get("timings", {})
    if not isinstance(timings, dict):
        timings = {}
    error: str | None = None
    if text is None:
        err_obj = body.get("error")
        error = json.dumps(err_obj, ensure_ascii=False)[:1000] if err_obj else "no-text-in-response"
    return text, finish, usage, timings, error


# ---------------------------------------------------------------------------
# Plan / dataset validation
# ---------------------------------------------------------------------------

def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def validate_plan(plan: Any) -> dict[str, Any]:
    require(isinstance(plan, dict), "plan must be a JSON object")
    for key in ("plan_id", "model", "template", "seed", "max_tokens", "temperature",
                "repetitions", "capacity_mode", "thresholds", "configs"):
        require(key in plan, f"plan missing required key: {key}")
    require(isinstance(plan["configs"], dict), "plan.configs must be an object")
    for cfg in REQUIRED_CONFIGS:
        require(cfg in plan["configs"], f"plan.configs missing required config {cfg!r}")
    reps = plan["repetitions"]
    require(
        isinstance(reps, int) and reps >= MIN_REPETITIONS,
        f"plan.repetitions must be an integer >= {MIN_REPETITIONS} "
        f"(PREFILL 18.2: >=5 independent alternating measurements)",
    )
    require(
        plan["capacity_mode"] in ("fixed-capacity", "auto-fit", "both"),
        "plan.capacity_mode must be fixed-capacity|auto-fit|both",
    )
    thresholds = plan["thresholds"]
    require(isinstance(thresholds, dict), "plan.thresholds must be an object")
    for key in ("overall_max_drop", "per_task_max_drop", "retrieval_max_drop"):
        require(key in thresholds, f"plan.thresholds missing required key: {key}")
        value = thresholds[key]
        require(
            isinstance(value, (int, float)) and 0 <= value <= 1,
            f"plan.thresholds.{key} must be a fraction in [0, 1]",
        )
    warmup = plan.get("warmup_requests", 1)
    require(isinstance(warmup, int) and warmup >= 0, "plan.warmup_requests must be >= 0")
    require(
        "allowed_switch_keys" not in plan,
        "plan.allowed_switch_keys is REMOVED: switches live in endpoint startup "
        "provenance, never in request fields; delete the key",
    )
    cache_dirs: dict[str, str] = {}
    startups: dict[str, str] = {}
    switch_blobs: dict[str, str] = {}
    for name, cfg in plan["configs"].items():
        require(isinstance(cfg, dict), f"config {name!r} must be an object")
        require(isinstance(cfg.get("label", name), str), f"config {name!r}: label must be a string")
        require(isinstance(cfg.get("switches", {}), dict), f"config {name!r}: switches must be an object")
        require(
            "request_overrides" not in cfg,
            f"config {name!r}: request_overrides is REMOVED (no supported "
            "per-request API exists for FlashPrefill/Tri policy; declare the "
            "difference in startup argv instead)",
        )
        startup = cfg.get("startup")
        require(isinstance(startup, dict), f"config {name!r}: explicit startup provenance is required")
        argv = startup.get("argv")
        require(
            isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv),
            f"config {name!r}: startup.argv must be a non-empty string list "
            "(the literal server startup command)",
        )
        for token in argv:
            require(
                not re.search(r"api[-_]?key|secret|password", token, re.IGNORECASE),
                f"config {name!r}: startup.argv must not contain secrets "
                "(redact API keys before recording provenance)",
            )
        rerot_state = startup.get("rerot")
        require(
            rerot_state in ("off", "on"),
            f"config {name!r}: startup.rerot must be 'off'|'on' (machine-readable "
            "RERoT assertion for this endpoint)",
        )
        cache_dir = cfg.get("cache_dir")
        require(
            isinstance(cache_dir, str) and cache_dir,
            f"config {name!r}: explicit cache_dir is required "
            "(configurations must never share a persistent-KV cache directory)",
        )
        cache_dirs[name] = cache_dir
        startups[name] = canonical(startup)
        switch_blobs[name] = canonical(cfg.get("switches", {}))
        group = cfg.get("capacity_group", "fixed")
        require(group in ("fixed", "auto-fit"), f"config {name!r}: capacity_group must be fixed|auto-fit")
        live = cfg.get("responses_jsonl") in (None, "")
        if live:
            require(
                cfg.get("base_url") or plan.get("base_url"),
                f"config {name!r}: live mode needs base_url (config or plan level)",
            )
        if name in REQUIRED_CONFIGS:
            require("parent" not in cfg, f"base config {name!r} must not declare a parent")
            require(
                rerot_state == "off",
                f"base config {name!r}: startup.rerot must be 'off' for the "
                "A/B/C/D comparison",
            )
            for key, value in cfg.get("switches", {}).items():
                require(
                    not ("rerot" in str(key).lower() and str(value).lower() in ("on", "true", "1", "yes", "enabled")),
                    f"base config {name!r}: switches must not claim RERoT on",
                )
        else:
            require(
                isinstance(cfg.get("parent"), str) and cfg["parent"] in plan["configs"],
                f"optional config {name!r}: must declare a 'parent' in plan.configs",
            )
            require(
                startups[name] != startups[cfg["parent"]],
                f"optional config {name!r}: startup must differ from parent "
                f"{cfg['parent']!r} (identical endpoints prove nothing)",
            )
            if rerot_state == "on":
                require(
                    plan["configs"][cfg["parent"]].get("startup", {}).get("rerot") == "off",
                    f"optional config {name!r}: rerot-on child requires a rerot-off parent",
                )
    if len(set(cache_dirs.values())) != len(cache_dirs):
        dupes = sorted({v for v in cache_dirs.values() if list(cache_dirs.values()).count(v) > 1})
        raise HarnessError(f"shared cache_dir across configurations (forbidden): {dupes}")
    base_startups = {startups[n] for n in REQUIRED_CONFIGS}
    require(
        len(base_startups) == len(REQUIRED_CONFIGS),
        "A/B/C/D startup provenance must be pairwise distinct "
        "(identical startup commands cannot evidence an algorithm comparison)",
    )
    require(
        len({switch_blobs[n] for n in REQUIRED_CONFIGS}) == len(REQUIRED_CONFIGS),
        "A/B/C/D switches identity must be pairwise distinct",
    )
    if plan["capacity_mode"] in ("fixed-capacity", "both"):
        fixed = {
            name: cfg
            for name, cfg in plan["configs"].items()
            if cfg.get("capacity_group", "fixed") == "fixed"
        }
        capacities = {canonical(cfg.get("capacity")) for cfg in fixed.values()}
        require(
            all(cfg.get("capacity") is not None for cfg in fixed.values()),
            "fixed-capacity mode: every fixed-group config must declare 'capacity'",
        )
        require(
            len(capacities) == 1,
            "fixed-capacity mode: fixed-group configs must declare IDENTICAL "
            f"capacity (K/B/P/batch/concurrency); saw {len(capacities)} variants",
        )
    return plan


def validate_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(rows, "dataset is empty")
    seen: set[str] = set()
    for lineno, row in enumerate(rows, 1):
        for key in ("item_id", "task", "length_bucket", "prompt", "expected", "scorer"):
            require(key in row, f"dataset line {lineno}: missing key {key!r}")
        require(isinstance(row["item_id"], str) and row["item_id"], f"dataset line {lineno}: bad item_id")
        require(row["item_id"] not in seen, f"dataset: duplicate item_id {row['item_id']!r}")
        seen.add(row["item_id"])
        require(isinstance(row["scorer"], dict), f"dataset line {lineno}: scorer must be an object")
        stype = row["scorer"].get("type")
        require(
            stype in ("exact", "numeric", "regex", "external"),
            f"dataset line {lineno}: unknown scorer type {stype!r} "
            "(no fake grading: fix the dataset, the harness will not default to success)",
        )
    return rows


def render_prompt(template: Any, prompt: str) -> str:
    if not isinstance(template, str) or not template:
        return prompt
    if "{prompt}" in template:
        return template.replace("{prompt}", prompt)
    return template + "\n\n" + prompt


def build_common_base(plan: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": plan["model"],
        "messages": [{"role": "user", "content": render_prompt(plan.get("template"), item["prompt"])}],
        "temperature": plan["temperature"],
        "seed": plan["seed"],
        "max_tokens": plan["max_tokens"],
        "stream": False,
    }


REROT_KEY_RE = re.compile(r"^rerot($|[_-])", re.IGNORECASE)


def assert_no_rerot_keys(payload: dict[str, Any], where: str) -> None:
    """An explicit rerot_frontier enables RERoT even with rerot=false, so the
    harness forbids the whole key family on the wire instead of checking
    truthiness."""
    bad = sorted(k for k in payload if REROT_KEY_RE.match(k))
    if bad:
        raise HarnessError(f"{where}: forbidden rerot request keys (would implicitly enable RERoT): {bad}")


def check_pair_equality(
    payloads: dict[str, dict[str, Any]],
    live_configs: list[str],
) -> list[str]:
    """Request payloads MUST be byte-identical across configurations.

    Switches live in endpoint startup provenance (asserted against declared
    startup argv); sampling/model/template/budget can never differ because
    one builder makes every payload. Any divergence aborts the run.
    """
    violations: list[str] = []
    if not live_configs:
        return violations
    reference = canonical(payloads[live_configs[0]])
    for name in live_configs[1:]:
        if canonical(payloads[name]) != reference:
            keys: set[str] = set(payloads[live_configs[0]]) | set(payloads[name])
            violations.extend(
                sorted(
                    k
                    for k in keys
                    if canonical(payloads[live_configs[0]].get(k)) != canonical(payloads[name].get(k))
                )
            )
    return sorted(set(violations))


def config_fingerprint(
    plan: dict[str, Any],
    name: str,
    dataset_sha: str,
    scorer_versions: list[str],
) -> str:
    cfg = plan["configs"][name]
    identity = {
        "harness": f"{HARNESS_NAME}-{HARNESS_VERSION}",
        "plan_id": plan["plan_id"],
        "model": plan["model"],
        "template": plan.get("template"),
        "seed": plan["seed"],
        "max_tokens": plan["max_tokens"],
        "temperature": plan["temperature"],
        "config": name,
        "label": cfg.get("label", name),
        "parent": cfg.get("parent"),
        "switches": cfg.get("switches", {}),
        "startup": cfg.get("startup", {}),
        "base_url": cfg.get("base_url") or plan.get("base_url"),
        "endpoint_path": cfg.get("endpoint_path", plan.get("endpoint_path", "/v1/chat/completions")),
        "cache_dir": cfg.get("cache_dir"),
        "capacity_group": cfg.get("capacity_group", "fixed"),
        "capacity": cfg.get("capacity"),
        "offline": bool(cfg.get("responses_jsonl")),
        "dataset_sha256": dataset_sha,
        "scorer_versions": sorted(set(scorer_versions)),
        "thresholds": plan["thresholds"],
    }
    return sha256_text(canonical(identity))


def scorer_identity(scorer: dict[str, Any]) -> str:
    stype = scorer.get("type", "?")
    if stype == "external":
        return f"external:{scorer.get('version', '?')}"
    return f"builtin-{stype}-1"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def resolve_api_key(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    env_name = cfg.get("api_key_env")
    if isinstance(env_name, str) and env_name and os.environ.get(env_name):
        return os.environ[env_name]
    if isinstance(cfg.get("api_key"), str) and cfg["api_key"]:
        return cfg["api_key"]
    if args.api_key:
        return args.api_key
    return os.environ.get("LLAMA_API_KEY", "")


def load_offline_evidence(
    plan: dict[str, Any], name: str
) -> dict[tuple[str, int], dict[str, Any]]:
    path = plan["configs"][name].get("responses_jsonl")
    rows = load_jsonl(Path(path))
    table: dict[tuple[str, int], dict[str, Any]] = {}
    for lineno, row in enumerate(rows, 1):
        item_id = first_present(row, ITEM_ID_KEYS)
        if not isinstance(item_id, str) or not item_id:
            raise HarnessError(f"{path}:{lineno}: row lacks item/case id")
        rep = row.get("rep", 0)
        try:
            rep = int(rep)
        except (TypeError, ValueError) as error:
            raise HarnessError(f"{path}:{lineno}: bad rep: {error}") from error
        key = (item_id, rep)
        if key in table:
            raise HarnessError(f"{path}:{lineno}: duplicate evidence for {key}")
        table[key] = row
    return table


def budget_exhausted(
    finish_reason: str | None, usage: dict[str, Any], max_tokens: int
) -> bool:
    if finish_reason == "length":
        return True
    try:
        if int(usage.get("completion_tokens", 0)) >= int(max_tokens) and int(max_tokens) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired FlashPrefill quality/performance harness (PREFILL 18-19): "
            "drives A/B/C/D FullKV/Tri dense/sparse configurations plus optional "
            "paired RERoT/MTP configurations over an explicit JSONL dataset, "
            "scores every raw response, and judges pre-declared thresholds. "
            "Missing evidence blocks PASS; SKIP is never counted as PASS."
        )
    )
    parser.add_argument("--plan", required=True, type=Path, help="Plan JSON (budgets, seeds, template, configs, thresholds).")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset JSONL (item/task/length/expected/scorer).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Independent output directory for this run.")
    parser.add_argument("--base-url", default=None, help="Default base URL for live configs lacking one.")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""), help="Default API key (or LLAMA_API_KEY).")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP seconds timeout (default 300).")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan/dataset/pair equality and print the planned matrix without any requests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        eprint("error: --timeout must be positive")
        return 1

    plan = validate_plan(load_json(args.plan))
    dataset = validate_dataset(load_jsonl(args.dataset))
    dataset_sha = sha256_file(args.dataset)
    scorer_versions = sorted({scorer_identity(row["scorer"]) for row in dataset})

    config_names = list(plan["configs"].keys())
    live_configs = [n for n in config_names if plan["configs"][n].get("responses_jsonl") in (None, "")]
    offline_configs = [n for n in config_names if n not in live_configs]
    fingerprints = {
        name: config_fingerprint(plan, name, dataset_sha, scorer_versions)
        for name in config_names
    }

    # Build and verify the paired request matrix BEFORE execution. One builder
    # makes every payload; switches live in endpoint startup provenance, never
    # in request fields. Identity across configurations is still asserted per
    # item so any future divergence aborts instead of silently comparing
    # different requests.
    planned_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for item in dataset:
        base = build_common_base(plan, item)
        assert_no_rerot_keys(base, f"item {item['item_id']}")
        planned_payloads[item["item_id"]] = {name: dict(base) for name in live_configs}
    pair_violations: list[str] = []
    for item in dataset:
        payloads = {name: planned_payloads[item["item_id"]][name] for name in live_configs}
        for bad in check_pair_equality(payloads, live_configs):
            if bad not in pair_violations:
                pair_violations.append(bad)
    if pair_violations:
        eprint(f"error: pair-equality violation (payloads differ across configs): {sorted(pair_violations)}")
        return 1

    reps: int = plan["repetitions"]
    warmup_n: int = plan.get("warmup_requests", 1)
    planned_cells = len(config_names) * len(dataset) * reps
    print(
        f"plan={plan['plan_id']} configs={config_names} "
        f"live={live_configs or 'none'} offline={offline_configs or 'none'} "
        f"items={len(dataset)} reps={reps} warmup={warmup_n} "
        f"planned_cells={planned_cells}"
    )
    if args.dry_run:
        print("dry-run: matrix validated (pair equality, caches, capacities, thresholds). No requests issued.")
        for name in config_names:
            print(f"  {name}: fingerprint={fingerprints[name][:16]}... cache={plan['configs'][name]['cache_dir']}")
        return 0

    outdir = args.output_dir
    try:
        outdir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        eprint(f"error: cannot create --output-dir {outdir}: {error}")
        return 1

    commands_txt = (
        f"harness: {HARNESS_NAME}-{HARNESS_VERSION} schema={SCHEMA_VERSION}\n"
        f"argv: {' '.join(sys.argv)}\n"
        f"plan: {args.plan}\n"
        f"dataset: {args.dataset} sha256={dataset_sha}\n"
        f"timeout: {args.timeout}\n"
    )
    for name in config_names:
        cfg = plan["configs"][name]
        endpoint = (cfg.get("base_url") or plan.get("base_url") or args.base_url or "http://127.0.0.1:8080").rstrip("/")
        path = cfg.get("endpoint_path", plan.get("endpoint_path", "/v1/chat/completions"))
        src = f"offline:{cfg['responses_jsonl']}" if name in offline_configs else f"live:{endpoint}{path}"
        startup_argv = " ".join(cfg.get("startup", {}).get("argv", []))
        commands_txt += (
            f"config {name} ({cfg.get('label', name)}): {src} cache={cfg['cache_dir']}\n"
            f"  startup: {startup_argv}\n"
        )
    (outdir / "commands.txt").write_text(commands_txt, encoding="utf-8")

    offline_tables = {name: load_offline_evidence(plan, name) for name in offline_configs}

    requests_rows: list[dict[str, Any]] = []
    responses_rows: list[dict[str, Any]] = []
    timings_rows: list[dict[str, Any]] = []
    scores_rows: list[dict[str, Any]] = []
    metrics_notes: dict[str, str] = {}
    fallback_reasons: set[str] = set()
    transport_errors = 0
    sequence = 0

    def record_cell(
        item: dict[str, Any],
        name: str,
        rep: int,
        payload: dict[str, Any] | None,
        endpoint: str,
        wall: float | None,
        status: int | None,
        body: dict[str, Any],
        raw: str | None,
        transport_error: str | None,
        warmup: bool,
    ) -> None:
        nonlocal sequence
        item_id = item["item_id"]
        case_id = f"{item_id}::rep{rep}" if not warmup else f"{item_id}::warmup{sequence}"
        if not warmup:
            requests_rows.append(
                {
                    "sequence": sequence,
                    "case_id": case_id,
                    "item_id": item_id,
                    "rep": rep,
                    "config": name,
                    "fingerprint": fingerprints[name],
                    "endpoint": endpoint,
                    "request": payload,
                    "warmup": False,
                }
            )
        if transport_error is not None:
            responses_rows.append(
                {
                    "sequence": sequence,
                    "case_id": case_id,
                    "item_id": item_id,
                    "rep": rep,
                    "config": name,
                    "http_status": status,
                    "output_text": None,
                    "finish_reason": None,
                    "usage": {},
                    "timings": {},
                    "client_wall_s": wall,
                    "budget_exhausted": False,
                    "error": transport_error,
                    "warmup": warmup,
                    "raw": (raw or "")[:2000],
                }
            )
            timings_rows.append(
                {
                    "sequence": sequence,
                    "case_id": case_id,
                    "item_id": item_id,
                    "rep": rep,
                    "config": name,
                    "warmup": warmup,
                    "client_wall_s": wall,
                    "prompt_ms": None,
                    "prompt_n": None,
                    "predicted_ms": None,
                    "predicted_n": None,
                    "ttft_ms": None,
                    "ttft_status": STATUS_NOT_RUN,
                    "tokens_per_s": None,
                }
            )
            if not warmup:
                graded = score_item(item, None, f"transport: {transport_error}"[:500], args.timeout)
                graded.update(
                    {"case_id": case_id, "rep": rep, "config": name, "budget_exhausted": False}
                )
                scores_rows.append(graded)
            sequence += 1
            return
        text, finish, usage, timings, parse_error = parse_chat_response(body, raw)
        if isinstance(body, dict) and body.get("fallback") is not None:
            fallback_reasons.add(str(body.get("fallback"))[:200])
        exhausted = budget_exhausted(finish, usage, plan["max_tokens"])
        responses_rows.append(
            {
                "sequence": sequence,
                "case_id": case_id,
                "item_id": item_id,
                "rep": rep,
                "config": name,
                "http_status": status,
                "output_text": text,
                "finish_reason": finish,
                "usage": usage,
                "timings": timings,
                "client_wall_s": wall,
                "budget_exhausted": exhausted,
                "error": parse_error,
                "warmup": warmup,
                "raw": body if len(canonical(body)) <= 20000 else {"truncated": True, "keys": sorted(body.keys())},
            }
        )
        prompt_ms = timings.get("prompt_ms")
        tps = None
        try:
            n = float(timings.get("predicted_n", 0))
            ms = float(timings.get("predicted_ms", 0))
            if n > 0 and ms > 0:
                tps = n / (ms / 1000.0)
        except (TypeError, ValueError):
            tps = None
        if tps is None and wall and text:
            try:
                n2 = float(usage.get("completion_tokens", 0))
                tps = n2 / wall if wall > 0 and n2 > 0 else None
            except (TypeError, ValueError):
                tps = None
        timings_rows.append(
            {
                "sequence": sequence,
                "case_id": case_id,
                "item_id": item_id,
                "rep": rep,
                "config": name,
                "warmup": warmup,
                "client_wall_s": wall,
                "prompt_ms": prompt_ms,
                "prompt_n": timings.get("prompt_n"),
                "predicted_ms": timings.get("predicted_ms"),
                "predicted_n": timings.get("predicted_n"),
                "ttft_ms": prompt_ms if isinstance(prompt_ms, (int, float)) else None,
                "ttft_status": "PRESENT" if isinstance(prompt_ms, (int, float)) else STATUS_NOT_RUN,
                "tokens_per_s": tps,
            }
        )
        if not warmup:
            graded = score_item(item, text, parse_error, args.timeout)
            graded.update(
                {"case_id": case_id, "rep": rep, "config": name, "budget_exhausted": exhausted}
            )
            scores_rows.append(graded)
        sequence += 1

    # Warmup per live configuration (excluded from quality, flagged in timings).
    first_items = [row for row in dataset if not row.get("skip_reason")]
    warmup_item = first_items[0] if first_items else dataset[0]
    for name in live_configs:
        cfg = plan["configs"][name]
        base_url = (cfg.get("base_url") or plan.get("base_url") or args.base_url or "http://127.0.0.1:8080").rstrip("/")
        path = cfg.get("endpoint_path", plan.get("endpoint_path", "/v1/chat/completions"))
        endpoint = base_url + path
        api_key = resolve_api_key(cfg, args)
        if not api_key:
            eprint(f"error: live config {name!r} has no API key (--api-key, LLAMA_API_KEY, or plan api_key_env)")
            return 1
        before, ok = fetch_metrics_text(base_url, api_key, args.timeout)
        (outdir / f"metrics-before-{name}.txt").write_text(before + ("\n" if not before.endswith("\n") else ""), encoding="utf-8")
        metrics_notes[name] = "metrics-collected" if ok else "metrics-before-NOT RUN"
        payload = build_common_base(plan, warmup_item)
        assert_no_rerot_keys(payload, f"warmup {name}")
        for _ in range(warmup_n):
            wall, status, body, raw = post_json(endpoint, payload, api_key, args.timeout)
            if status is None:
                eprint(f"warning: warmup transport failure for {name}: {(raw or '')[:200]}")
                record_cell(warmup_item, name, -1, payload, endpoint, wall, status, body, raw, raw, True)
            else:
                record_cell(warmup_item, name, -1, payload, endpoint, wall, status, body, raw, None, True)
    for name in offline_configs:
        (outdir / f"metrics-before-{name}.txt").write_text(
            "NOT RUN: offline evidence mode; metrics were not collected by this harness.\n",
            encoding="utf-8",
        )
        metrics_notes[name] = "metrics-NOT RUN-offline"

    # Alternating measured loop: rep-major, configuration order rotated per rep.
    for rep in range(reps):
        order = config_names[rep % len(config_names):] + config_names[: rep % len(config_names)]
        for item in dataset:
            if item.get("skip_reason"):
                for name in config_names:
                    scores_rows.append(
                        {
                            "case_id": f"{item['item_id']}::rep{rep}",
                            "item_id": item["item_id"],
                            "rep": rep,
                            "config": name,
                            "task": item.get("task"),
                            "length_bucket": item.get("length_bucket"),
                            "scorer_type": item.get("scorer", {}).get("type"),
                            "scorer_version": item.get("scorer", {}).get("version", "builtin-1"),
                            "expected": item.get("expected", ""),
                            "extracted": "",
                            "score": None,
                            "status": STATUS_SKIP,
                            "failure_reason": None,
                            "budget_exhausted": False,
                            "skip_reason": item.get("skip_reason"),
                        }
                    )
                continue
            for name in order:
                cfg = plan["configs"][name]
                if name in offline_tables:
                    row = offline_tables[name].get((item["item_id"], rep))
                    if row is None:
                        case_id = f"{item['item_id']}::rep{rep}"
                        responses_rows.append(
                            {
                                "sequence": sequence,
                                "case_id": case_id,
                                "item_id": item["item_id"],
                                "rep": rep,
                                "config": name,
                                "http_status": None,
                                "output_text": None,
                                "finish_reason": None,
                                "usage": {},
                                "timings": {},
                                "client_wall_s": None,
                                "budget_exhausted": False,
                                "error": "missing-offline-evidence",
                                "warmup": False,
                                "raw": {},
                            }
                        )
                        scores_rows.append(
                            {
                                "case_id": case_id,
                                "item_id": item["item_id"],
                                "rep": rep,
                                "config": name,
                                "task": item.get("task"),
                                "length_bucket": item.get("length_bucket"),
                                "scorer_type": item.get("scorer", {}).get("type"),
                                "scorer_version": item.get("scorer", {}).get("version", "builtin-1"),
                                "expected": item.get("expected", ""),
                                "extracted": "",
                                "score": None,
                                "status": STATUS_NOT_RUN,
                                "failure_reason": "missing-offline-evidence",
                                "budget_exhausted": False,
                            }
                        )
                        timings_rows.append(
                            {
                                "sequence": sequence,
                                "case_id": case_id,
                                "item_id": item["item_id"],
                                "rep": rep,
                                "config": name,
                                "warmup": False,
                                "client_wall_s": None,
                                "prompt_ms": None,
                                "prompt_n": None,
                                "predicted_ms": None,
                                "predicted_n": None,
                                "ttft_ms": None,
                                "ttft_status": STATUS_NOT_RUN,
                                "tokens_per_s": None,
                            }
                        )
                        sequence += 1
                        continue
                    if row.get("warmup") is True:
                        # Provider-side warmup evidence: not part of the
                        # measured matrix, excluded from scores and perf.
                        continue
                    text = first_present(row, TEXT_KEYS)
                    if not isinstance(text, str):
                        text = None
                    timings = row.get("timings", {})
                    if not isinstance(timings, dict):
                        timings = {}
                    usage = row.get("usage", {})
                    if not isinstance(usage, dict):
                        usage = {}
                    finish = row.get("finish_reason")
                    inline = row.get("response_json")
                    if isinstance(inline, dict):
                        inline_text, inline_finish, inline_usage, inline_timings, _ = parse_chat_response(inline, None)
                        if text is None:
                            text = inline_text
                        if finish is None:
                            finish = inline_finish
                        if not usage:
                            usage = inline_usage
                        if not timings:
                            timings = inline_timings
                    status = row.get("http_status", 200)
                    err = row.get("error")
                    parse_error = None
                    if text is None and err is None:
                        parse_error = "offline-row-has-no-output-text"
                    elif isinstance(err, str) and err:
                        parse_error = f"offline-error: {err}"[:500]
                    for echo_key in ("request", "payload"):
                        echo = row.get(echo_key)
                        if isinstance(echo, dict):
                            assert_no_rerot_keys(echo, f"offline {name} {item['item_id']}::{echo_key}")
                    endpoint = f"offline:{cfg.get('responses_jsonl')}"
                    payload = build_common_base(plan, item)
                    assert_no_rerot_keys(payload, f"item {item['item_id']}")
                    requests_rows.append(
                        {
                            "sequence": sequence,
                            "case_id": f"{item['item_id']}::rep{rep}",
                            "item_id": item["item_id"],
                            "rep": rep,
                            "config": name,
                            "fingerprint": fingerprints[name],
                            "endpoint": endpoint,
                            "request": payload,
                            "warmup": False,
                            "note": "offline evidence: switch asserted against declared startup provenance, not wire fields",
                        }
                    )
                    wall = row.get("client_wall_s", row.get("latency_s"))
                    try:
                        wall = float(wall) if wall is not None else None
                    except (TypeError, ValueError):
                        wall = None
                    exhausted = budget_exhausted(finish, usage, plan["max_tokens"])
                    responses_rows.append(
                        {
                            "sequence": sequence,
                            "case_id": f"{item['item_id']}::rep{rep}",
                            "item_id": item["item_id"],
                            "rep": rep,
                            "config": name,
                            "http_status": status,
                            "output_text": text,
                            "finish_reason": finish,
                            "usage": usage,
                            "timings": timings,
                            "client_wall_s": wall,
                            "budget_exhausted": exhausted,
                            "error": parse_error,
                            "warmup": False,
                            "raw": row,
                        }
                    )
                    prompt_ms = timings.get("prompt_ms")
                    tps = None
                    try:
                        n = float(timings.get("predicted_n", 0))
                        ms = float(timings.get("predicted_ms", 0))
                        if n > 0 and ms > 0:
                            tps = n / (ms / 1000.0)
                    except (TypeError, ValueError):
                        tps = None
                    timings_rows.append(
                        {
                            "sequence": sequence,
                            "case_id": f"{item['item_id']}::rep{rep}",
                            "item_id": item["item_id"],
                            "rep": rep,
                            "config": name,
                            "warmup": False,
                            "client_wall_s": wall,
                            "prompt_ms": prompt_ms,
                            "prompt_n": timings.get("prompt_n"),
                            "predicted_ms": timings.get("predicted_ms"),
                            "predicted_n": timings.get("predicted_n"),
                            "ttft_ms": prompt_ms if isinstance(prompt_ms, (int, float)) else None,
                            "ttft_status": "PRESENT" if isinstance(prompt_ms, (int, float)) else STATUS_NOT_RUN,
                            "tokens_per_s": tps,
                        }
                    )
                    graded = score_item(item, text, parse_error, args.timeout)
                    graded.update(
                        {
                            "case_id": f"{item['item_id']}::rep{rep}",
                            "rep": rep,
                            "config": name,
                            "budget_exhausted": exhausted,
                        }
                    )
                    scores_rows.append(graded)
                    sequence += 1
                    continue
                base_url = (cfg.get("base_url") or plan.get("base_url") or args.base_url or "http://127.0.0.1:8080").rstrip("/")
                path = cfg.get("endpoint_path", plan.get("endpoint_path", "/v1/chat/completions"))
                endpoint = base_url + path
                api_key = resolve_api_key(cfg, args)
                if not api_key:
                    eprint(f"error: live config {name!r} has no API key (--api-key, LLAMA_API_KEY, or plan api_key_env)")
                    return 1
                payload = planned_payloads[item["item_id"]][name]
                wall, status, body, raw = post_json(endpoint, payload, api_key, args.timeout)
                if status is None:
                    transport_errors += 1
                    record_cell(item, name, rep, payload, endpoint, wall, status, body, raw, raw, False)
                elif status != 200:
                    transport_errors += 1
                    record_cell(item, name, rep, payload, endpoint, wall, status, body, raw, f"http-{status}: {(raw or '')[:500]}", False)
                else:
                    record_cell(item, name, rep, payload, endpoint, wall, status, body, raw, None, False)

    for name in live_configs:
        cfg = plan["configs"][name]
        base_url = (cfg.get("base_url") or plan.get("base_url") or args.base_url or "http://127.0.0.1:8080").rstrip("/")
        after, ok = fetch_metrics_text(base_url, resolve_api_key(cfg, args), args.timeout)
        (outdir / f"metrics-after-{name}.txt").write_text(after + ("\n" if not after.endswith("\n") else ""), encoding="utf-8")
        if not ok:
            metrics_notes[name] += ";metrics-after-NOT RUN"
    for name in offline_configs:
        (outdir / f"metrics-after-{name}.txt").write_text(
            "NOT RUN: offline evidence mode; metrics were not collected by this harness.\n",
            encoding="utf-8",
        )

    # Resource series passthrough (never synthesized).
    memory_rows: list[dict[str, Any]] = []
    for name in config_names:
        src = plan["configs"][name].get("resources_jsonl")
        if src in (None, ""):
            continue
        for row in load_jsonl(Path(src)):
            tagged = dict(row)
            tagged["config"] = name
            memory_rows.append(tagged)
    if memory_rows:
        write_jsonl(outdir / "memory.jsonl", memory_rows)
    else:
        (outdir / "memory.jsonl").write_text(
            canonical({"status": STATUS_NOT_RUN, "reason": "no per-config resources_jsonl supplied; resource time series not measured by this harness"}) + "\n",
            encoding="utf-8",
        )

    write_jsonl(outdir / "requests.jsonl", requests_rows)
    write_jsonl(outdir / "responses.jsonl", responses_rows)
    write_jsonl(outdir / "timings.jsonl", timings_rows)
    write_jsonl(outdir / "scores.jsonl", scores_rows)

    provenance = plan.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {"provenance_raw": provenance}
    binary_sha = provenance.get("binary_sha256") or plan.get("binary_sha256")
    if binary_sha:
        (outdir / "binary-libraries.sha256").write_text(f"{binary_sha}  <operator-attested-binary>\n", encoding="utf-8")
    else:
        (outdir / "binary-libraries.sha256").write_text(
            "NOT RUN: no binary hash declared in plan.provenance; operator must attest binary + libraries.\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness": f"{HARNESS_NAME}-{HARNESS_VERSION}",
        "plan_id": plan["plan_id"],
        "model": plan["model"],
        "template": plan.get("template"),
        "seed": plan["seed"],
        "max_tokens": plan["max_tokens"],
        "temperature": plan["temperature"],
        "dataset": {"path": str(args.dataset), "sha256": dataset_sha, "items": len(dataset)},
        "scorer_versions": scorer_versions,
        "repetitions": reps,
        "warmup_requests": warmup_n,
        "capacity_mode": plan["capacity_mode"],
        "thresholds": plan["thresholds"],
        "configs": {
            name: {
                "label": plan["configs"][name].get("label", name),
                "parent": plan["configs"][name].get("parent"),
                "fingerprint": fingerprints[name],
                "switches": plan["configs"][name].get("switches", {}),
                "startup": plan["configs"][name].get("startup", {}),
                "cache_dir": plan["configs"][name]["cache_dir"],
                "capacity_group": plan["configs"][name].get("capacity_group", "fixed"),
                "capacity": plan["configs"][name].get("capacity"),
                "mode": "offline" if name in offline_configs else "live",
                "metrics": metrics_notes.get(name, ""),
            }
            for name in config_names
        },
        "pair_equality": {
            "mode": "identical-wire-payloads+startup-provenance",
            "request_fields": "byte-identical across configs (switches asserted against declared startup argv, never sent as request fields)",
            "offline_schema": "harness-defined evidence rows; NOT the MatrixHarness schema (align with its owner before interchanging)",
        },
        "fallback_reasons_observed": sorted(fallback_reasons),
        "provenance": provenance,
    }
    (outdir / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Analysis: quality with Wilson CIs, paired diffs, cliffs; perf medians.
    # ------------------------------------------------------------------
    rep0 = [s for s in scores_rows if s.get("rep") == 0 and s.get("status") != STATUS_SKIP]
    by_config: dict[str, list[dict[str, Any]]] = {n: [] for n in config_names}
    for row in rep0:
        if row.get("config") in by_config and row.get("status") in (STATUS_SCORED, STATUS_ERROR, STATUS_NOT_RUN):
            by_config[row["config"]].append(row)

    def accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [r for r in rows if r.get("status") == STATUS_SCORED]
        k = sum(1 for r in scored if r.get("score") == 1)
        stat = wilson(k, len(scored))
        stat["k"] = k
        stat["errors"] = sum(1 for r in rows if r.get("status") == STATUS_ERROR)
        stat["not_run"] = sum(1 for r in rows if r.get("status") == STATUS_NOT_RUN)
        stat["budget_exhausted"] = sum(1 for r in rows if r.get("budget_exhausted"))
        return stat

    quality: dict[str, Any] = {name: accuracy(rows) for name, rows in by_config.items()}

    score_lookup = {(s["config"], s["item_id"]): s for s in rep0}
    paired: dict[str, Any] = {}
    pairs_to_check = [n for n in config_names if n != "A"]
    for name in pairs_to_check:
        parent = plan["configs"][name].get("parent", "A")
        ref = parent if parent in by_config else "A"
        items = [i for i in dataset if not i.get("skip_reason")]
        b = c = n = 0
        flips: list[dict[str, Any]] = []
        for item in items:
            x = score_lookup.get((name, item["item_id"]))
            a = score_lookup.get((ref, item["item_id"]))
            if x is None or a is None:
                continue
            if x.get("status") != STATUS_SCORED or a.get("status") != STATUS_SCORED:
                continue
            n += 1
            xs, ar = (1 if x.get("score") == 1 else 0), (1 if a.get("score") == 1 else 0)
            if xs == 1 and ar == 0:
                b += 1
            elif xs == 0 and ar == 1:
                c += 1
            if xs != ar:
                flips.append({"item_id": item["item_id"], "task": item.get("task"), name: xs, ref: ar})
        stat = paired_diff_ci(b, c, n)
        stat["reference"] = ref
        stat["flips"] = flips
        paired[f"{name}-vs-{ref}"] = stat

    def group_table(key: str) -> dict[str, Any]:
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in rep0:
            g = str(row.get(key, "?"))
            groups.setdefault(g, {n: [] for n in config_names})
            if row.get("config") in groups[g]:
                groups[g][row["config"]].append(row)
        table = {}
        for g, per_cfg in sorted(groups.items()):
            accs = {n: accuracy(rows) for n, rows in per_cfg.items()}
            small_n = any((accs.get("A", {}).get("n", 0) or 0) < SMALL_N for _ in [0])
            table[g] = {"accuracy": accs, "small_n_warning": bool(small_n)}
        return table

    by_task = group_table("task")
    by_length = group_table("length_bucket")

    thresholds = plan["thresholds"]
    cliffs: list[dict[str, Any]] = []
    for g, entry in by_task.items():
        accs = entry["accuracy"]
        base = (accs.get("A") or {}).get("acc")
        if base is None:
            continue
        limit = thresholds["retrieval_max_drop"] if g in RETRIEVAL_TASKS else thresholds["per_task_max_drop"]
        for name in pairs_to_check:
            acc = (accs.get(name) or {}).get("acc")
            if acc is None:
                continue
            drop = base - acc
            if drop > limit:
                cliffs.append({"group": f"task:{g}", "config": name, "ref": "A", "drop": drop, "limit": limit})
    for g, entry in by_length.items():
        accs = entry["accuracy"]
        base = (accs.get("A") or {}).get("acc")
        if base is None:
            continue
        for name in pairs_to_check:
            acc = (accs.get(name) or {}).get("acc")
            if acc is None:
                continue
            drop = base - acc
            if drop > thresholds["per_task_max_drop"]:
                cliffs.append({"group": f"length:{g}", "config": name, "ref": "A", "drop": drop, "limit": thresholds["per_task_max_drop"]})

    perf: dict[str, Any] = {}
    for name in config_names:
        samples = [
            t for t in timings_rows
            if t.get("config") == name and not t.get("warmup") and t.get("client_wall_s") is not None
        ]
        walls = [float(t["client_wall_s"]) for t in samples]
        tps_vals = [float(t["tokens_per_s"]) for t in samples if t.get("tokens_per_s") is not None]
        ttft_vals = [float(t["ttft_ms"]) for t in samples if isinstance(t.get("ttft_ms"), (int, float))]
        groups: dict[str, list[float]] = {}
        for t in samples:
            item_id = t.get("item_id", "?")
            groups.setdefault(item_id, []).append(float(t["client_wall_s"]))
        perf[name] = {
            "n": len(walls),
            "wall_s": {"median": median(walls), "p95": percentile(walls, 95), "min": min(walls) if walls else None, "max": max(walls) if walls else None, "raw": walls},
            "tokens_per_s": {"median": median(tps_vals), "p95": percentile(tps_vals, 95), "raw": tps_vals, "status": "PRESENT" if tps_vals else STATUS_NOT_RUN},
            "ttft_ms": {"median": median(ttft_vals), "raw": ttft_vals, "status": "PRESENT" if ttft_vals else STATUS_NOT_RUN},
            "capacity_group": plan["configs"][name].get("capacity_group", "fixed"),
            "capacity": plan["configs"][name].get("capacity"),
        }

    # Completeness: every planned config x item x rep cell needs a SCORED/ERROR
    # score row (SKIP rows only cover pre-declared dataset skips).
    expected_cells = {(n, i["item_id"], r) for n in config_names for i in dataset for r in range(reps)}
    present_cells = set()
    for s in scores_rows:
        if s.get("status") in (STATUS_SCORED, STATUS_ERROR, STATUS_SKIP):
            present_cells.add((s.get("config"), s.get("item_id"), s.get("rep")))
        elif s.get("status") == STATUS_NOT_RUN:
            present_cells.add((s.get("config"), s.get("item_id"), s.get("rep")))
    missing = sorted(expected_cells - {(c, i, r) for (c, i, r) in present_cells})
    not_run_cells = [
        {"config": s.get("config"), "item_id": s.get("item_id"), "rep": s.get("rep")}
        for s in scores_rows if s.get("status") == STATUS_NOT_RUN
    ]
    skipped_cells = sum(1 for s in scores_rows if s.get("status") == STATUS_SKIP)
    complete = not missing and not not_run_cells

    checks: dict[str, bool] = {}
    checks["matrix_complete"] = bool(complete)
    checks["zero_transport_errors"] = transport_errors == 0 and all(
        quality[n]["errors"] == 0 for n in config_names
    )
    checks["pair_equality_ok"] = not pair_violations
    checks["caches_isolated"] = True  # enforced at plan validation; re-stated for the record
    overall_ok = True
    for key, stat in paired.items():
        name = key.split("-vs-")[0]
        ref_acc = quality.get(stat["reference"], {}).get("acc")
        acc = quality.get(name, {}).get("acc")
        if ref_acc is None or acc is None:
            overall_ok = False
            continue
        if ref_acc - acc > thresholds["overall_max_drop"]:
            overall_ok = False
    checks["overall_within_threshold"] = bool(overall_ok)
    checks["no_task_or_length_cliffs"] = not cliffs
    # C-vs-A isolation: report the pre-existing Tri gap separately; it neither
    # excuses D nor fails B on its own (thresholds still apply to every pair).
    c_gap = None
    if quality.get("A", {}).get("acc") is not None and quality.get("C", {}).get("acc") is not None:
        c_gap = quality["A"]["acc"] - quality["C"]["acc"]

    config_status = {}
    for name in config_names:
        rows = by_config[name]
        if not rows and not [s for s in scores_rows if s.get("config") == name]:
            config_status[name] = STATUS_NOT_RUN
        elif any(s.get("status") == STATUS_NOT_RUN for s in scores_rows if s.get("config") == name):
            config_status[name] = STATUS_NOT_RUN
        elif name not in REQUIRED_CONFIGS and not rows:
            config_status[name] = STATUS_SKIP
        else:
            config_status[name] = VERDICT_PASS if all(checks.values()) else VERDICT_FAIL

    if not complete or any(v in (STATUS_NOT_RUN, STATUS_SKIP) for v in config_status.values()):
        verdict = VERDICT_INCOMPLETE
    elif all(checks.values()):
        verdict = VERDICT_PASS
    else:
        verdict = VERDICT_FAIL

    summary = {
        "schema_version": SCHEMA_VERSION,
        "harness": f"{HARNESS_NAME}-{HARNESS_VERSION}",
        "plan_id": plan["plan_id"],
        "verdict": verdict,
        "checks": checks,
        "config_status": config_status,
        "fingerprints": fingerprints,
        "thresholds": thresholds,
        "quality_rep0": quality,
        "paired_diffs": paired,
        "by_task": by_task,
        "by_length": by_length,
        "cliffs": cliffs,
        "tri_c_vs_a_gap": c_gap,
        "tri_isolation_note": (
            "C-vs-A gap is the pre-existing Tri sparsity effect; D is judged "
            "against BOTH C (FlashPrefill-on-Tri delta) and A (cumulative user "
            "impact). A small D-vs-C delta never excuses a large D-vs-A gap."
        ),
        "perf": perf,
        "completeness": {
            "complete": bool(complete),
            "planned_cells": planned_cells,
            "missing_cells": missing[:100],
            "missing_count": len(missing),
            "not_run_cells": not_run_cells[:100],
            "not_run_count": len(not_run_cells),
            "skipped_cells": skipped_cells,
            "transport_errors": transport_errors,
        },
        "counts": {
            "requests": len(requests_rows),
            "responses": len(responses_rows),
            "scores": len(scores_rows),
            "timings": len(timings_rows),
        },
        "files": [
            "manifest.json", "commands.txt", "requests.jsonl", "responses.jsonl",
            "scores.jsonl", "timings.jsonl", "memory.jsonl",
            "binary-libraries.sha256", "summary.json",
        ] + [f"metrics-before-{n}.txt" for n in config_names] + [f"metrics-after-{n}.txt" for n in config_names],
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"verdict={verdict} complete={str(complete).lower()} out={outdir}")
    for name in config_names:
        q = quality[name]
        acc = q["acc"]
        acc_s = f"{acc:.4f}" if acc is not None else STATUS_NOT_RUN
        print(f"  {name}: acc={acc_s} n={q['n']} errors={q['errors']} status={config_status[name]}")
    for key, stat in paired.items():
        d = stat["delta"]
        d_s = f"{d:+.4f}" if d is not None else STATUS_NOT_RUN
        print(f"  {key}: delta={d_s} b/c={stat['b']}/{stat['c']} n={stat['n']}")
    if cliffs:
        for cliff in cliffs:
            print(f"  CLIFF: {cliff['group']} {cliff['config']} drop={cliff['drop']:.4f} limit={cliff['limit']:.4f}")
    if verdict == VERDICT_INCOMPLETE:
        print(f"INCOMPLETE: {len(missing)} missing + {len(not_run_cells)} NOT RUN cells block PASS", file=sys.stderr)
        return 2
    if verdict != VERDICT_PASS:
        for check, ok in checks.items():
            if not ok:
                print(f"FAIL: {check}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
