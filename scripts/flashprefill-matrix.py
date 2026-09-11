#!/usr/bin/env python3
"""FlashPrefill full-compatibility matrix runner (PREFILL sections 17 and 18.6).

Standalone CLI. Explicit JSON inputs, stable case IDs, independent output
directories, PASS/FAIL/SKIP statuses, nonzero exit on failure. Standard library
only. Never fabricates missing metrics and never treats SKIP as PASS.

Inputs (both required, JSON files)
----------------------------------
--matrix <path>: servers, cases with ordered steps, expectations, comparisons.
--requests <path>: fixed named request templates referenced by cases.

Matrix schema (schema_version must be 1; unknown keys are rejected)::

    {
      "schema_version": 1,
      "meta": { ...free-form provenance... },
      "servers": [
        {
          "id": "off-baseline",
          "launch": {
            "argv": ["./build/bin/llama-server", "-m", "/models/m.gguf",
                     "--host", "127.0.0.1", "--port", "18086", "--metrics"],
            "port": 18086,
            "startup_timeout_s": 120.0,
            "shutdown_timeout_s": 30.0
          },
          "request_timeout_s": 300.0,
          "metrics_timeout_s": 30.0,
          "api_key": "",
          "cache_domain": "off-cache-a"
        },
        {
          "id": "candidate",
          "endpoint": {"base_url": "http://127.0.0.1:18087"},
          "request_timeout_s": 300.0,
          "metrics_timeout_s": 30.0,
          "cache_domain": "cand-cache-a"
        }
      ],
      "cases": [
        {
          "id": "off-short",
          "server": "off-baseline",
          "description": "optional",
          "steps": [
            {"kind": "request", "request": "short-greedy",
             "timeout_s": 120.0, "warmup": true},
            {"kind": "concurrent_requests", "requests": ["a", "b"],
             "timeout_s": 300.0, "max_workers": 2},
            {"kind": "stream_request", "request": "stream-basic",
             "timeout_s": 120.0, "expect_stream": "success"},
            {"kind": "slot_save", "slot_id": 0, "filename": "c.bin",
             "timeout_s": 60.0},
            {"kind": "slot_restore", "slot_id": 0, "filename": "c.bin",
             "timeout_s": 60.0},
            {"kind": "slot_erase", "slot_id": 0, "timeout_s": 60.0},
            {"kind": "cancel_stream", "request": "stream-basic",
             "timeout_s": 60.0, "cancel_after_s": 1.0},
            {"kind": "sleep", "seconds": 1.0},
            {"kind": "health_check", "timeout_s": 30.0}
          ],
          "expect": {
            "require_metrics": {
              "flashprefill_sparse_rows_total": {"min_delta": 1.0},
              "flashprefill_dense_rows_total{reason=\\"decode\\"}": {"min_delta": 0.0}
            },
            "forbid_metrics_increase": ["rerot_hard_aborts"],
            "invariants": {"visible_ge_exact": true,
                           "corrected_positive_if_sparse": true},
            "dense_reasons_allowed": ["decode", "short_context"],
            "expect_dense_reason": null,
            "off_baseline": false,
            "old_binary_baseline": false,
            "match_case": null,
            "rerot": {"min_completed_episodes": 0,
                      "require_no_hard_aborts": true},
            "skip_if_metrics_missing": []
          }
        }
      ],
      "comparisons": [
        {"id": "off-vs-candidate", "baseline_case": "off-short",
         "candidate_case": "cand-short", "require_exact_match": true,
         "require_resource_evidence": true}
      ]
    }

Server entries take exactly one of ``endpoint`` or ``launch``. ``launch`` servers
are owned children: the harness rejects occupied ports before spawning, polls
``/health`` until ready, captures the child log, and terminates only that child.
Endpoint servers are never started, stopped, or killed. Every server declares an
explicit ``cache_domain`` (non-empty opaque label naming its cache/startup
provenance domain); the full launch argv (startup policy) is recorded in the
manifest. Cache isolation is enforced, not advisory: the two cases of any
``comparisons`` entry or ``match_case`` pair must run on different servers with
different ``cache_domain`` values, otherwise the matrix is rejected before
anything runs. Same-server cases share state by design (slot/cache continuity
for state-chain tests). Every numeric timeout is
explicit and bounded (0 < t <= 3600, sleep <= 300). ``meta`` is free-form
provenance (upstream SHA, model/calibration hashes, hardware notes) recorded
verbatim into the manifest. Request ``payload`` objects are passed through
verbatim to the server (server validates them); the harness never adds,
removes, or overrides payload fields (no cache_prompt/id_slot injection or any
other invented request override).

Requests schema (schema_version must be 1; unknown keys are rejected)::

    {
      "schema_version": 1,
      "requests": [
        {"id": "short-greedy", "endpoint": "/completion",
         "payload": {"prompt": "hi", "n_predict": 16, "temperature": 0.0,
                     "seed": 42, "stream": false, "cache_prompt": false},
         "timeout_s": 120.0, "description": "optional",
         "expected_substring": "optional-scoring-hint", "rep": 0}
      ]
    }

Allowed request endpoints: /completion, /v1/completions,
/v1/chat/completions, /embeddings, /v1/embeddings, /rerank, /reranking,
/tokenize, /detokenize. ``kind: request`` and ``concurrent_requests`` require
non-streaming payloads; ``stream_request`` and ``cancel_stream`` require
``stream: true`` payloads on a streaming endpoint. Step ``timeout_s`` bounds the
step; the referenced request ``timeout_s`` is recorded as evidence and the
smaller of the two is enforced per HTTP call.

Metric selectors
----------------
Bare ``flashprefill_sparse_rows_total`` sums every series with that name
(total across bounded labels). ``name{reason="decode"}`` selects only series
whose labels cover the filter. Missing selectors never count as zero: a
required/forbidden/invariant/rerot check naming an absent metric FAILs, unless
its bare name is listed in ``skip_if_metrics_missing``, in which case the case
is SKIP. Raw ``/metrics`` bodies are preserved verbatim; deltas cover bounded
labels (e.g. ``dense_rows_total{reason=...}``, ``pool_rebuild_total{reason=...}``).

Old binary baseline vs new-binary OFF (never conflated)
-------------------------------------------------------
``old_binary_baseline: true`` marks a case running on the pre-FlashPrefill
binary, which cannot expose ``flashprefill_*`` counters. Total absence of those
counters there is expected and passes by construction; their presence FAILs as
unexpected. Such cases must not reference ``flashprefill_*`` in
``require_metrics``/``forbid_metrics_increase``, must leave both invariants off,
must leave dense-reason constraints null, and must not set ``off_baseline``
(rejected at validation). ``off_baseline: true`` is the opposite claim about a
new binary with FlashPrefill OFF: the counters must be present and every
``flashprefill_*`` delta must be zero; absent counters FAIL. An old-baseline
case therefore never blocks OFF regression by construction, and comparisons may
pair an old-baseline case (no counters) with a new-OFF case (zero deltas).

Streaming contract (per-endpoint, matching tools/server current semantics)
--------------------------------------------------------------------------
Raw ``/completion`` streams carry no ``[DONE]`` sentinel: the stream ends after
the single ``stop:true`` event. Success there requires exactly one ``stop:true``
event and zero ``[DONE]`` lines. OpenAI-compatible endpoints (``/v1/completions``,
``/v1/chat/completions``) terminate with ``data: [DONE]``: success there requires
exactly one ``finish_reason != null`` choice event and exactly one ``[DONE]`` line.
``expect_stream: "error_or_cancel"`` accepts HTTP errors or truncated streams
and checks they are recorded, not disguised as success. RERoT/trace-like events
must not carry content deltas; any such leak FAILs the case. Cancel steps open
a stream, read until ``cancel_after_s``, then close client-side and verify the
server is still healthy.

Outputs (under --output-dir)
----------------------------
manifest.json, commands.txt, requests.jsonl, responses.jsonl, timings.jsonl,
scores.jsonl, memory.jsonl, summary.json, binary-libraries.sha256,
metrics-before-<server>.txt / metrics-after-<server>.txt (plus top-level
metrics-before.txt / metrics-after.txt copies when exactly one server exists),
servers/<id>/server.log for owned launches (plus top-level server.log copy for
a single owned server), cases/<case-id>/metrics-before.txt|after.txt,
deltas.json, sse-<seq>.txt raw streams. summary.json carries per-case
PASS/FAIL/SKIP, per-comparison results, and an overall verdict.

Cross-harness join contract (aligned with scripts/flashprefill-quality.py)
--------------------------------------------------------------------------
Every requests/responses/timings/scores row carries ``case`` (quality config
slot), ``request`` (quality item slot), and ``rep`` (quality repetition slot,
default 0; repeated rounds must use distinct ``rep`` values). Every
responses.jsonl row echoes the verbatim sent ``payload`` object so consumers
can assert on request fields (e.g. zero rerot keys) without joining; every
timings.jsonl row carries ``seq`` for an exact 1:1 join with responses.jsonl
on ``seq`` (per-request perf may also be read from inline ``response_json``
timings/usage). Rows with ``warmup:true`` are advisory and skipped by scorers.
``response_json`` stays inline; ``raw_file`` only points at raw SSE evidence.
``warmup`` defaults to false everywhere. Output directories are independent
per harness; no filenames are shared across harnesses.

Exit codes: 0 when no case/comparison FAILs (SKIP allowed), 2 when any FAILs,
1 on harness/validation/IO errors.

Limitations needing integration (not handled by this harness)
-------------------------------------------------------------
* The harness does not build binaries, choose production alpha/policy values,
  score task quality, or run benchmarks; quality gates live elsewhere.
* Cache isolation is enforced for A/B pairs: every server carries an explicit
  ``cache_domain``, and the two cases of any comparison or ``match_case`` pair
  must run on different servers with different ``cache_domain`` values
  (rejected at validation). Same-server cases share state by design for
  state-chain tests. The manifest records config identity (argv/endpoint hash),
  launch argv, /props capacity, and per-request cache_prompt/id_slot evidence.
* Old-binary baselines (``old_binary_baseline: true``) are expected to expose
  no ``flashprefill_*`` counters and pass by construction; new-binary OFF
  (``off_baseline: true``) must expose them with zero deltas, else FAIL.
* Tri/MTP domain checks are expressed through generic require_metrics on the
  server's real counter names; only RERoT episode/abort helpers are built in.
* GPU/driver/device facts and model/calibration hashes come from matrix ``meta``
  as stated by the caller; the harness records /props but never probes GPUs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MAX_TIMEOUT_S = 3600.0
MAX_SLEEP_S = 300.0
MAX_ID_LEN = 128

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(\{.*\})?\s+([^\s]+)\s*$"
)

ALLOWED_REQUEST_ENDPOINTS = (
    "/completion",
    "/v1/completions",
    "/v1/chat/completions",
    "/embeddings",
    "/v1/embeddings",
    "/rerank",
    "/reranking",
    "/tokenize",
    "/detokenize",
)
STREAMING_ENDPOINTS = (
    "/completion",
    "/v1/completions",
    "/v1/chat/completions",
)

STEP_KINDS = (
    "request",
    "concurrent_requests",
    "stream_request",
    "slot_save",
    "slot_restore",
    "slot_erase",
    "cancel_stream",
    "sleep",
    "health_check",
)

EXPECT_KEYS = frozenset(
    {
        "require_metrics",
        "forbid_metrics_increase",
        "invariants",
        "dense_reasons_allowed",
        "expect_dense_reason",
        "off_baseline",
        "old_binary_baseline",
        "match_case",
        "rerot",
        "skip_if_metrics_missing",
    }
)
INVARIANT_KEYS = frozenset({"visible_ge_exact", "corrected_positive_if_sparse"})
REROT_KEYS = frozenset({"min_completed_episodes", "require_no_hard_aborts"})

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"


class HarnessError(Exception):
    """Fatal harness/validation error (exit 1)."""


def eprint(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def shell_quote(argv: list[str]) -> str:
    out: list[str] = []
    for tok in argv:
        if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", tok):
            out.append(tok)
        else:
            out.append("'" + tok.replace("'", "'\\''") + "'")
    return " ".join(out)


def redact_key(value: str) -> str:
    if not value:
        return "<empty>"
    return "<redacted:%d>" % len(value)


# --------------------------------------------------------------------------
# JSON loading and structural validation
# --------------------------------------------------------------------------

def load_json_strict(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HarnessError(f"cannot read {path}: {exc}") from exc

    def _reject_constant(raw: str) -> Any:
        raise HarnessError(f"{path}: non-finite JSON constant {raw!r} is rejected")

    try:
        doc = json.loads(text, parse_constant=_reject_constant)
    except HarnessError:
        raise
    except Exception as exc:
        raise HarnessError(f"{path}: invalid JSON: {exc}") from exc
    assert_finite(doc, str(path))
    return doc


def assert_finite(node: Any, ctx: str) -> None:
    if isinstance(node, float):
        if not math.isfinite(node):
            raise HarnessError(f"{ctx}: non-finite number is rejected")
    elif isinstance(node, dict):
        for key, val in node.items():
            if not isinstance(key, str):
                raise HarnessError(f"{ctx}: non-string object key is rejected")
            assert_finite(val, ctx)
    elif isinstance(node, (list, tuple)):
        for val in node:
            assert_finite(val, ctx)


def check_dict_keys(obj: Any, allowed: frozenset[str] | set[str], ctx: str) -> None:
    if not isinstance(obj, dict):
        raise HarnessError(f"{ctx}: expected object")
    for key in obj:
        if key not in allowed:
            raise HarnessError(f"{ctx}: unknown key {key!r}")


def check_id(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_ID_LEN:
        raise HarnessError(f"{ctx}: id must be a non-empty string (<= {MAX_ID_LEN})")
    if not ID_RE.match(value):
        raise HarnessError(f"{ctx}: id {value!r} must match [A-Za-z0-9._-] starting alnum")
    return value


def check_timeout(value: Any, ctx: str, maximum: float = MAX_TIMEOUT_S) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessError(f"{ctx}: timeout must be a number")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0 or fval > maximum:
        raise HarnessError(f"{ctx}: timeout must satisfy 0 < t <= {maximum}")
    return fval


def check_nonneg_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessError(f"{ctx}: expected non-negative integer")
    return value


def validate_requests_doc(doc: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(doc, dict):
        raise HarnessError("requests: top level must be an object")
    check_dict_keys(doc, {"schema_version", "requests"}, "requests")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError("requests: schema_version must be 1")
    raw = doc.get("requests")
    if not isinstance(raw, list) or not raw:
        raise HarnessError("requests: 'requests' must be a non-empty list")
    index: dict[str, dict[str, Any]] = {}
    for pos, entry in enumerate(raw):
        ctx = f"requests[{pos}]"
        if not isinstance(entry, dict):
            raise HarnessError(f"{ctx}: must be an object")
        check_dict_keys(
            entry,
            {"id", "endpoint", "payload", "timeout_s", "description", "expected_substring", "rep"},
            ctx,
        )
        rid = check_id(entry.get("id"), ctx + ".id")
        if rid in index:
            raise HarnessError(f"{ctx}: duplicate request id {rid!r}")
        endpoint = entry.get("endpoint")
        if endpoint not in ALLOWED_REQUEST_ENDPOINTS:
            raise HarnessError(f"{ctx}: endpoint must be one of {list(ALLOWED_REQUEST_ENDPOINTS)}")
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            raise HarnessError(f"{ctx}: payload must be an object")
        timeout = check_timeout(entry.get("timeout_s"), ctx + ".timeout_s")
        if "description" in entry and not isinstance(entry["description"], str):
            raise HarnessError(f"{ctx}: description must be a string")
        if "expected_substring" in entry and not isinstance(entry["expected_substring"], str):
            raise HarnessError(f"{ctx}: expected_substring must be a string")
        rep = entry.get("rep", 0)
        if isinstance(rep, bool) or not isinstance(rep, int) or rep < 0 or rep > 9999:
            raise HarnessError(f"{ctx}: rep must be an integer 0..9999")
        index[rid] = {
            "id": rid,
            "endpoint": endpoint,
            "payload": payload,
            "timeout_s": timeout,
            "description": entry.get("description", ""),
            "expected_substring": entry.get("expected_substring"),
            "rep": rep,
        }
    return index


def validate_server_entry(entry: Any, pos: int, seen: set[str], ports: set[int]) -> dict[str, Any]:
    ctx = f"servers[{pos}]"
    if not isinstance(entry, dict):
        raise HarnessError(f"{ctx}: must be an object")
    check_dict_keys(
        entry, {"id", "endpoint", "launch", "request_timeout_s", "metrics_timeout_s", "api_key", "cache_domain"}, ctx
    )
    sid = check_id(entry.get("id"), ctx + ".id")
    cache_domain = check_id(entry.get("cache_domain"), ctx + ".cache_domain")
    if sid in seen:
        raise HarnessError(f"{ctx}: duplicate server id {sid!r}")
    seen.add(sid)
    has_endpoint = "endpoint" in entry
    has_launch = "launch" in entry
    if has_endpoint == has_launch:
        raise HarnessError(f"{ctx}: exactly one of 'endpoint' or 'launch' is required")
    request_timeout = check_timeout(entry.get("request_timeout_s"), ctx + ".request_timeout_s")
    metrics_timeout = check_timeout(entry.get("metrics_timeout_s"), ctx + ".metrics_timeout_s")
    api_key = entry.get("api_key", "")
    if not isinstance(api_key, str):
        raise HarnessError(f"{ctx}: api_key must be a string")
    out: dict[str, Any] = {
        "id": sid,
        "request_timeout_s": request_timeout,
        "metrics_timeout_s": metrics_timeout,
        "api_key": api_key,
        "cache_domain": cache_domain,
    }
    if has_endpoint:
        ep = entry["endpoint"]
        if not isinstance(ep, dict):
            raise HarnessError(f"{ctx}.endpoint: must be an object")
        check_dict_keys(ep, {"base_url"}, ctx + ".endpoint")
        base_url = ep.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise HarnessError(f"{ctx}.endpoint.base_url: must be an http(s) URL")
        out["endpoint"] = {"base_url": base_url.rstrip("/")}
        out["launch"] = None
    else:
        launch = entry["launch"]
        if not isinstance(launch, dict):
            raise HarnessError(f"{ctx}.launch: must be an object")
        check_dict_keys(
            launch, {"argv", "port", "startup_timeout_s", "shutdown_timeout_s"}, ctx + ".launch"
        )
        argv = launch.get("argv")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(t, str) or not t for t in argv
        ):
            raise HarnessError(f"{ctx}.launch.argv: must be a non-empty list of strings")
        port = launch.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or port < 1 or port > 65535:
            raise HarnessError(f"{ctx}.launch.port: must be an integer 1..65535")
        if port in ports:
            raise HarnessError(f"{ctx}: duplicate launch port {port}")
        ports.add(port)
        startup = check_timeout(launch.get("startup_timeout_s"), ctx + ".launch.startup_timeout_s")
        shutdown = check_timeout(launch.get("shutdown_timeout_s"), ctx + ".launch.shutdown_timeout_s")
        out["endpoint"] = None
        out["launch"] = {
            "argv": list(argv),
            "port": port,
            "startup_timeout_s": startup,
            "shutdown_timeout_s": shutdown,
        }
    return out


def validate_step(step: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise HarnessError(f"{ctx}: must be an object")
    if "kind" not in step:
        raise HarnessError(f"{ctx}: missing 'kind'")
    kind = step.get("kind")
    if kind not in STEP_KINDS:
        raise HarnessError(f"{ctx}: unknown kind {kind!r}")
    allowed = {"kind"}
    out: dict[str, Any] = {"kind": kind}
    if kind == "request":
        allowed |= {"request", "timeout_s", "warmup"}
        if not isinstance(step.get("request"), str) or not step["request"]:
            raise HarnessError(f"{ctx}: request id must be a non-empty string")
        out["request"] = step["request"]
        out["timeout_s"] = check_timeout(step.get("timeout_s"), ctx + ".timeout_s")
        out["warmup"] = bool(step.get("warmup", False))
        if "warmup" in step and not isinstance(step["warmup"], bool):
            raise HarnessError(f"{ctx}: warmup must be a boolean")
    elif kind == "concurrent_requests":
        allowed |= {"requests", "timeout_s", "max_workers", "warmup"}
        reqs = step.get("requests")
        if not isinstance(reqs, list) or len(reqs) < 2 or any(
            not isinstance(r, str) or not r for r in reqs
        ):
            raise HarnessError(f"{ctx}: requests must be a list of >= 2 request ids")
        if len(set(reqs)) != len(reqs):
            raise HarnessError(f"{ctx}: duplicate request ids in concurrent_requests")
        out["requests"] = list(reqs)
        out["timeout_s"] = check_timeout(step.get("timeout_s"), ctx + ".timeout_s")
        mw = step.get("max_workers", len(reqs))
        if isinstance(mw, bool) or not isinstance(mw, int) or mw < 1 or mw > 64:
            raise HarnessError(f"{ctx}: max_workers must be an integer 1..64")
        out["max_workers"] = mw
        out["warmup"] = bool(step.get("warmup", False))
        if "warmup" in step and not isinstance(step["warmup"], bool):
            raise HarnessError(f"{ctx}: warmup must be a boolean")
    elif kind == "stream_request":
        allowed |= {"request", "timeout_s", "expect_stream", "warmup"}
        if not isinstance(step.get("request"), str) or not step["request"]:
            raise HarnessError(f"{ctx}: request id must be a non-empty string")
        out["request"] = step["request"]
        out["timeout_s"] = check_timeout(step.get("timeout_s"), ctx + ".timeout_s")
        exp = step.get("expect_stream", "success")
        if exp not in ("success", "error_or_cancel"):
            raise HarnessError(f"{ctx}: expect_stream must be 'success' or 'error_or_cancel'")
        out["expect_stream"] = exp
        out["warmup"] = bool(step.get("warmup", False))
        if "warmup" in step and not isinstance(step["warmup"], bool):
            raise HarnessError(f"{ctx}: warmup must be a boolean")
    elif kind in ("slot_save", "slot_restore"):
        allowed |= {"slot_id", "filename", "timeout_s"}
        out["slot_id"] = check_nonneg_int(step.get("slot_id"), ctx + ".slot_id")
        fn = step.get("filename")
        if not isinstance(fn, str) or not fn or "/" in fn or "\\" in fn or len(fn) > 128:
            raise HarnessError(f"{ctx}: filename must be a bare file name (<= 128 chars)")
        out["filename"] = fn
        out["timeout_s"] = check_timeout(step.get("timeout_s"), ctx + ".timeout_s")
    elif kind == "slot_erase":
        allowed |= {"slot_id", "timeout_s"}
        out["slot_id"] = check_nonneg_int(step.get("slot_id"), ctx + ".slot_id")
        out["timeout_s"] = check_timeout(step.get("timeout_s"), ctx + ".timeout_s")
    elif kind == "cancel_stream":
        allowed |= {"request", "timeout_s", "cancel_after_s"}
        if not isinstance(step.get("request"), str) or not step["request"]:
            raise HarnessError(f"{ctx}: request id must be a non-empty string")
        out["request"] = step["request"]
        out["timeout_s"] = check_timeout(step.get("timeout_s"), ctx + ".timeout_s")
        ca = step.get("cancel_after_s")
        if isinstance(ca, bool) or not isinstance(ca, (int, float)):
            raise HarnessError(f"{ctx}: cancel_after_s must be a number")
        caf = float(ca)
        if not math.isfinite(caf) or caf < 0.0 or caf > MAX_SLEEP_S:
            raise HarnessError(f"{ctx}: cancel_after_s must satisfy 0 <= t <= {MAX_SLEEP_S}")
        if caf >= out["timeout_s"]:
            raise HarnessError(f"{ctx}: cancel_after_s must be smaller than timeout_s")
        out["cancel_after_s"] = caf
    elif kind == "sleep":
        allowed |= {"seconds"}
        sec = step.get("seconds")
        if isinstance(sec, bool) or not isinstance(sec, (int, float)):
            raise HarnessError(f"{ctx}: seconds must be a number")
        secf = float(sec)
        if not math.isfinite(secf) or secf < 0.0 or secf > MAX_SLEEP_S:
            raise HarnessError(f"{ctx}: seconds must satisfy 0 <= t <= {MAX_SLEEP_S}")
        out["seconds"] = secf
    elif kind == "health_check":
        allowed |= {"timeout_s"}
        out["timeout_s"] = check_timeout(step.get("timeout_s"), ctx + ".timeout_s")
    check_dict_keys(step, allowed, ctx)
    return out


def validate_expect(raw: Any, ctx: str) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise HarnessError(f"{ctx}: expect must be an object")
    check_dict_keys(raw, EXPECT_KEYS, ctx)
    out: dict[str, Any] = {}
    req = raw.get("require_metrics", {})
    if not isinstance(req, dict):
        raise HarnessError(f"{ctx}.require_metrics: must be an object")
    norm_req: dict[str, dict[str, float]] = {}
    for sel, cond in req.items():
        if not isinstance(sel, str) or not sel:
            raise HarnessError(f"{ctx}.require_metrics: selector must be a non-empty string")
        parse_metric_selector(sel, ctx + ".require_metrics")
        if not isinstance(cond, dict) or not cond:
            raise HarnessError(f"{ctx}.require_metrics[{sel!r}]: must be a non-empty object")
        for ck in cond:
            if ck not in ("min_delta", "max_delta", "exact_delta"):
                raise HarnessError(f"{ctx}.require_metrics[{sel!r}]: unknown key {ck!r}")
        for ck, cv in cond.items():
            if isinstance(cv, bool) or not isinstance(cv, (int, float)) or not math.isfinite(cv):
                raise HarnessError(f"{ctx}.require_metrics[{sel!r}].{ck}: must be finite")
        norm_req[sel] = {k: float(v) for k, v in cond.items()}
    out["require_metrics"] = norm_req
    forb = raw.get("forbid_metrics_increase", [])
    if not isinstance(forb, list) or any(not isinstance(s, str) or not s for s in forb):
        raise HarnessError(f"{ctx}.forbid_metrics_increase: must be a list of strings")
    for sel in forb:
        parse_metric_selector(sel, ctx + ".forbid_metrics_increase")
    if len(set(forb)) != len(forb):
        raise HarnessError(f"{ctx}.forbid_metrics_increase: duplicate selectors")
    out["forbid_metrics_increase"] = list(forb)
    inv = raw.get("invariants", {})
    if not isinstance(inv, dict):
        raise HarnessError(f"{ctx}.invariants: must be an object")
    for k in inv:
        if k not in INVARIANT_KEYS:
            raise HarnessError(f"{ctx}.invariants: unknown key {k!r}")
        if not isinstance(inv[k], bool):
            raise HarnessError(f"{ctx}.invariants.{k}: must be a boolean")
    out["invariants"] = {k: bool(inv.get(k, False)) for k in INVARIANT_KEYS}
    dra = raw.get("dense_reasons_allowed", None)
    if dra is not None:
        if not isinstance(dra, list) or any(not isinstance(r, str) or not r for r in dra):
            raise HarnessError(f"{ctx}.dense_reasons_allowed: must be null or list of strings")
        out["dense_reasons_allowed"] = list(dra)
    else:
        out["dense_reasons_allowed"] = None
    edr = raw.get("expect_dense_reason", None)
    if edr is not None and (not isinstance(edr, str) or not edr):
        raise HarnessError(f"{ctx}.expect_dense_reason: must be null or a non-empty string")
    out["expect_dense_reason"] = edr
    ob = raw.get("off_baseline", False)
    if not isinstance(ob, bool):
        raise HarnessError(f"{ctx}.off_baseline: must be a boolean")
    out["off_baseline"] = ob
    oldb = raw.get("old_binary_baseline", False)
    if not isinstance(oldb, bool):
        raise HarnessError(f"{ctx}.old_binary_baseline: must be a boolean")
    out["old_binary_baseline"] = oldb
    if ob and oldb:
        raise HarnessError(f"{ctx}: off_baseline and old_binary_baseline are mutually exclusive")
    if oldb:
        for sel in list(norm_req) + forb:
            if sel.split("{", 1)[0].startswith("flashprefill_"):
                raise HarnessError(f"{ctx}: old_binary_baseline cases must not reference {sel!r}")
        if any(inv.values()):
            raise HarnessError(f"{ctx}: old_binary_baseline cases must leave invariants off")
        if dra is not None or edr is not None:
            raise HarnessError(f"{ctx}: old_binary_baseline cases must leave dense-reason constraints null")
    mc = raw.get("match_case", None)
    if mc is not None and (not isinstance(mc, str) or not mc):
        raise HarnessError(f"{ctx}.match_case: must be null or a case id string")
    out["match_case"] = mc
    rerot = raw.get("rerot", {})
    if not isinstance(rerot, dict):
        raise HarnessError(f"{ctx}.rerot: must be an object")
    for k in rerot:
        if k not in REROT_KEYS:
            raise HarnessError(f"{ctx}.rerot: unknown key {k!r}")
    mep = rerot.get("min_completed_episodes", 0)
    if isinstance(mep, bool) or not isinstance(mep, int) or mep < 0:
        raise HarnessError(f"{ctx}.rerot.min_completed_episodes: must be an integer >= 0")
    rnh = rerot.get("require_no_hard_aborts", False)
    if not isinstance(rnh, bool):
        raise HarnessError(f"{ctx}.rerot.require_no_hard_aborts: must be a boolean")
    out["rerot"] = {"min_completed_episodes": mep, "require_no_hard_aborts": rnh}
    skipm = raw.get("skip_if_metrics_missing", [])
    if not isinstance(skipm, list) or any(not isinstance(s, str) or not s for s in skipm):
        raise HarnessError(f"{ctx}.skip_if_metrics_missing: must be a list of strings")
    for name in skipm:
        if "{" in name or "}" in name:
            raise HarnessError(f"{ctx}.skip_if_metrics_missing: entries must be bare metric names")
    out["skip_if_metrics_missing"] = list(skipm)
    return out


def validate_matrix_doc(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise HarnessError("matrix: top level must be an object")
    check_dict_keys(doc, {"schema_version", "meta", "servers", "cases", "comparisons"}, "matrix")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError("matrix: schema_version must be 1")
    meta = doc.get("meta", {})
    if not isinstance(meta, dict):
        raise HarnessError("matrix: meta must be an object")
    raw_servers = doc.get("servers")
    if not isinstance(raw_servers, list) or not raw_servers:
        raise HarnessError("matrix: 'servers' must be a non-empty list")
    seen_servers: set[str] = set()
    ports: set[int] = set()
    servers = [validate_server_entry(e, i, seen_servers, ports) for i, e in enumerate(raw_servers)]
    raw_cases = doc.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise HarnessError("matrix: 'cases' must be a non-empty list")
    seen_cases: set[str] = set()
    cases: list[dict[str, Any]] = []
    server_ids = {s["id"] for s in servers}
    for i, entry in enumerate(raw_cases):
        ctx = f"cases[{i}]"
        if not isinstance(entry, dict):
            raise HarnessError(f"{ctx}: must be an object")
        check_dict_keys(entry, {"id", "server", "description", "steps", "expect"}, ctx)
        cid = check_id(entry.get("id"), ctx + ".id")
        if cid in seen_cases:
            raise HarnessError(f"{ctx}: duplicate case id {cid!r}")
        seen_cases.add(cid)
        srv = entry.get("server")
        if srv not in server_ids:
            raise HarnessError(f"{ctx}: unknown server {srv!r}")
        if "description" in entry and not isinstance(entry["description"], str):
            raise HarnessError(f"{ctx}: description must be a string")
        steps_raw = entry.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise HarnessError(f"{ctx}: steps must be a non-empty list")
        steps = [validate_step(s, f"{ctx}.steps[{j}]") for j, s in enumerate(steps_raw)]
        expect = validate_expect(entry.get("expect"), ctx + ".expect")
        cases.append(
            {
                "id": cid,
                "server": srv,
                "description": entry.get("description", ""),
                "steps": steps,
                "expect": expect,
            }
        )
    raw_cmp = doc.get("comparisons", [])
    if not isinstance(raw_cmp, list):
        raise HarnessError("matrix: comparisons must be a list")
    seen_cmp: set[str] = set()
    comparisons: list[dict[str, Any]] = []
    for i, entry in enumerate(raw_cmp):
        ctx = f"comparisons[{i}]"
        if not isinstance(entry, dict):
            raise HarnessError(f"{ctx}: must be an object")
        check_dict_keys(
            entry,
            {"id", "baseline_case", "candidate_case", "require_exact_match", "require_resource_evidence"},
            ctx,
        )
        cmp_id = check_id(entry.get("id"), ctx + ".id")
        if cmp_id in seen_cmp:
            raise HarnessError(f"{ctx}: duplicate comparison id {cmp_id!r}")
        seen_cmp.add(cmp_id)
        for fk in ("baseline_case", "candidate_case"):
            if entry.get(fk) not in seen_cases:
                raise HarnessError(f"{ctx}: unknown case {entry.get(fk)!r} in {fk}")
        for fk in ("require_exact_match", "require_resource_evidence"):
            if not isinstance(entry.get(fk), bool):
                raise HarnessError(f"{ctx}: {fk} must be a boolean")
        comparisons.append(
            {
                "id": cmp_id,
                "baseline_case": entry["baseline_case"],
                "candidate_case": entry["candidate_case"],
                "require_exact_match": entry["require_exact_match"],
                "require_resource_evidence": entry["require_resource_evidence"],
            }
        )
    # cross-check match_case references
    for case in cases:
        mc = case["expect"]["match_case"]
        if mc is not None and mc not in seen_cases:
            raise HarnessError(f"case {case['id']!r}: unknown match_case {mc!r}")
        if mc == case["id"]:
            raise HarnessError(f"case {case['id']!r}: match_case must differ from own id")
    # cache-isolation enforcement: A/B pairs must run on different servers with
    # different cache_domain values, so shared prompt-cache state can never
    # leak between baseline and candidate legs.
    case_index = {c["id"]: c for c in cases}
    server_domains = {s["id"]: s["cache_domain"] for s in servers}

    def _require_isolated(a: str, b: str, ctx: str) -> None:
        srv_a = case_index[a]["server"]
        srv_b = case_index[b]["server"]
        if srv_a == srv_b:
            raise HarnessError(
                f"{ctx}: cases {a!r} and {b!r} must run on different servers "
                f"(both on {srv_a!r}); shared state would void isolation"
            )
        if server_domains[srv_a] == server_domains[srv_b]:
            raise HarnessError(
                f"{ctx}: servers {srv_a!r} and {srv_b!r} share cache_domain "
                f"{server_domains[srv_a]!r}; A/B legs need distinct cache domains"
            )

    for case in cases:
        mc = case["expect"]["match_case"]
        if mc is not None:
            _require_isolated(case["id"], mc, f"case {case['id']!r} match_case")
    for cmp in comparisons:
        _require_isolated(
            cmp["baseline_case"], cmp["candidate_case"], f"comparison {cmp['id']!r}"
        )
    return {"meta": meta, "servers": servers, "cases": cases, "comparisons": comparisons}


def validate_cross_refs(matrix: dict[str, Any], requests: dict[str, dict[str, Any]]) -> None:
    for case in matrix["cases"]:
        for j, step in enumerate(case["steps"]):
            ctx = f"case {case['id']!r} steps[{j}]"
            if step["kind"] == "request" and step["request"] not in requests:
                raise HarnessError(f"{ctx}: unknown request {step['request']!r}")
            elif step["kind"] == "concurrent_requests":
                for rid in step["requests"]:
                    if rid not in requests:
                        raise HarnessError(f"{ctx}: unknown request {rid!r}")
            elif step["kind"] in ("stream_request", "cancel_stream"):
                if step["request"] not in requests:
                    raise HarnessError(f"{ctx}: unknown request {step['request']!r}")


# --------------------------------------------------------------------------
# Prometheus metrics with bounded labels
# --------------------------------------------------------------------------

def parse_metric_selector(sel: str, ctx: str) -> tuple[str, dict[str, str] | None]:
    sel = sel.strip()
    if not sel:
        raise HarnessError(f"{ctx}: empty metric selector")
    if "{" not in sel:
        if not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", sel):
            raise HarnessError(f"{ctx}: bad metric name {sel!r}")
        return sel, None
    m = re.fullmatch(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}", sel)
    if not m:
        raise HarnessError(f"{ctx}: bad labeled selector {sel!r}")
    name, body = m.group(1), m.group(2).strip()
    filt: dict[str, str] = {}
    if body:
        for part in split_label_list(body, ctx):
            km = re.fullmatch(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*', part)
            if not km:
                raise HarnessError(f"{ctx}: bad label matcher {part!r}")
            filt[km.group(1)] = unescape_label(km.group(2))
    return name, filt


def split_label_list(body: str, ctx: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    in_str = False
    esc = False
    for ch in body:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                cur.append(ch)
            elif ch == ",":
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
    if in_str:
        raise HarnessError(f"{ctx}: unterminated label string in {body!r}")
    tail = "".join(cur)
    if tail.strip():
        parts.append(tail)
    return [p for p in parts if p.strip()]


def unescape_label(val: str) -> str:
    return val.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")


def parse_labels_block(block: str) -> dict[str, str]:
    inner = block[1:-1].strip()
    labels: dict[str, str] = {}
    if not inner:
        return labels
    for part in split_label_list(inner, "metrics"):
        km = re.fullmatch(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*', part)
        if not km:
            continue
        labels[km.group(1)] = unescape_label(km.group(2))
    return labels


def canon_labels(labels: dict[str, str]) -> str:
    return ",".join(f'{k}="{labels[k]}"' for k in sorted(labels))


def series_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    return name + "{" + canon_labels(labels) + "}"


def parse_metrics_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition; keep labeled and unlabeled series."""
    series: dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = METRIC_LINE_RE.match(s)
        if not m:
            continue
        name, block, raw_val = m.group(1), m.group(2), m.group(3)
        labels = parse_labels_block(block) if block else {}
        try:
            if raw_val in ("+Inf", "Inf", "inf"):
                val = math.inf
            elif raw_val in ("-Inf", "-inf"):
                val = -math.inf
            elif raw_val in ("NaN", "nan"):
                val = math.nan
            else:
                val = float(raw_val)
        except ValueError:
            continue
        key = series_key(name, labels)
        series[key] = val
    return series


def decompose_series_key(key: str) -> tuple[str, dict[str, str]]:
    if "{" not in key:
        return key, {}
    m = re.fullmatch(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\{(.*)\}", key)
    if not m:
        return key, {}
    name, body = m.group(1), m.group(2)
    labels: dict[str, str] = {}
    if body.strip():
        for part in split_label_list(body, "series"):
            km = re.fullmatch(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*', part)
            if km:
                labels[km.group(1)] = unescape_label(km.group(2))
    return name, labels


def selector_delta(
    before: dict[str, float], after: dict[str, float], selector: str
) -> dict[str, Any]:
    name, filt = parse_metric_selector(selector, "delta")
    b_match: dict[str, float] = {}
    a_match: dict[str, float] = {}
    for key, val in before.items():
        n, labs = decompose_series_key(key)
        if n != name:
            continue
        if filt is None or all(labs.get(k) == v for k, v in filt.items()):
            b_match[key] = val
    for key, val in after.items():
        n, labs = decompose_series_key(key)
        if n != name:
            continue
        if filt is None or all(labs.get(k) == v for k, v in filt.items()):
            a_match[key] = val
    missing = not b_match and not a_match
    b_total = sum(v for v in b_match.values() if math.isfinite(v))
    a_total = sum(v for v in a_match.values() if math.isfinite(v))
    nonfinite = any(not math.isfinite(v) for v in list(b_match.values()) + list(a_match.values()))
    return {
        "selector": selector,
        "missing": missing,
        "before_series": b_match,
        "after_series": a_match,
        "before_total": b_total,
        "after_total": a_total,
        "delta": (a_total - b_total) if not missing else None,
        "before_missing": not bool(b_match),
        "after_missing": not bool(a_match),
        "nonfinite": nonfinite,
    }


def dense_reason_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    reasons: dict[str, float] = {}
    names: set[str] = set()
    for store in (before, after):
        for key in store:
            n, labs = decompose_series_key(key)
            if n == "flashprefill_dense_rows_total" and "reason" in labs:
                names.add(labs["reason"])
    for reason in names:
        info = selector_delta(before, after, f'flashprefill_dense_rows_total{{reason="{reason}"}}')
        if not info["missing"]:
            reasons[reason] = info["delta"] if info["delta"] is not None else 0.0
    return reasons


# --------------------------------------------------------------------------
# HTTP helpers (standard library only)
# --------------------------------------------------------------------------

def auth_headers(api_key: str) -> dict[str, str]:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def http_get_text(base_url: str, route: str, api_key: str, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(
        base_url + route, headers=dict(auth_headers(api_key)), method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return exc.code, body


def http_get_json(base_url: str, route: str, api_key: str, timeout: float) -> tuple[int, Any]:
    status, text = http_get_text(base_url, route, api_key, timeout)
    try:
        return status, json.loads(text) if text.strip() else None
    except Exception:
        return status, None


def http_post_json(
    url: str, payload: dict[str, Any], api_key: str, timeout: float
) -> tuple[int | None, bytes, Any, str | None]:
    data = json.dumps(payload).encode("utf-8")
    headers = dict(auth_headers(api_key))
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, raw, json.loads(raw.decode("utf-8", errors="replace")), None
            except Exception:
                return resp.status, raw, None, None
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace")) if raw else None
        except Exception:
            parsed = None
        return exc.code, raw, parsed, None
    except Exception as exc:
        return None, b"", None, f"{type(exc).__name__}: {exc}"


def read_sse_stream(
    url: str, payload: dict[str, Any], api_key: str, timeout: float
) -> dict[str, Any]:
    """POST a streaming request and consume the SSE body.

    Returns raw lines, data events, terminal/DONE counts, assembled text, and
    any transport/HTTP error. Never raises on protocol content; raises only on
    local IO misuse (handled by callers as step errors).
    """
    started = time.monotonic()
    deadline = started + timeout
    data = json.dumps(payload).encode("utf-8")
    headers = dict(auth_headers(api_key))
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "text/event-stream"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    raw_lines: list[str] = []
    events: list[Any] = []
    done_count = 0
    pending_data: list[str] = []
    http_status: int | None = None

    def flush_block() -> None:
        nonlocal done_count
        if not pending_data:
            return
        blob = "\n".join(pending_data)
        pending_data.clear()
        if blob.strip() == "[DONE]":
            done_count += 1
            events.append("[DONE]")
            return
        try:
            events.append(json.loads(blob))
        except Exception:
            events.append({"_raw": blob})

    try:
        resp = urllib.request.urlopen(req, timeout=max(1.0, deadline - time.monotonic()))
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        return {
            "http_status": exc.code,
            "transport_error": None,
            "raw_text": raw.decode("utf-8", errors="replace"),
            "raw_lines": [],
            "events": [],
            "done_count": 0,
            "latency_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "http_status": None,
            "transport_error": f"{type(exc).__name__}: {exc}",
            "raw_text": "",
            "raw_lines": [],
            "events": [],
            "done_count": 0,
            "latency_s": time.monotonic() - started,
        }
    try:
        http_status = getattr(resp, "status", 200)
        with resp:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "http_status": http_status,
                        "transport_error": "TimeoutError: SSE read deadline exceeded",
                        "raw_text": "\n".join(raw_lines),
                        "raw_lines": raw_lines,
                        "events": events,
                        "done_count": done_count,
                        "latency_s": time.monotonic() - started,
                    }
                try:
                    sock = getattr(resp, "fp", None)
                    raw_sock = getattr(sock, "raw", None)
                    if raw_sock is not None and hasattr(raw_sock, "_sock"):
                        raw_sock._sock.settimeout(min(5.0, remaining))
                except Exception:
                    pass
                line_b = resp.readline()
                if not line_b:
                    break
                try:
                    line = line_b.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    line = ""
                raw_lines.append(line)
                if line == "" :
                    flush_block()
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    val = line[5:].lstrip()
                    if val == "[DONE]":
                        done_count += 1
                        events.append("[DONE]")
                    else:
                        pending_data.append(val)
                        # llama.cpp emits one JSON per data line; flush eagerly
                        # but keep multi-line data blocks correct via blank-line flush.
                        if len(pending_data) == 1:
                            # peek: keep single-line fast path by flushing now;
                            # a following data: line without blank separator starts
                            # a new event, which is also correct for this server.
                            flush_block()
                    continue
                if line.startswith("event:"):
                    continue
                # Unknown SSE field: preserve raw, ignore for assembly.
            flush_block()
    except Exception as exc:
        return {
            "http_status": http_status,
            "transport_error": f"{type(exc).__name__}: {exc}",
            "raw_text": "\n".join(raw_lines),
            "raw_lines": raw_lines,
            "events": events,
            "done_count": done_count,
            "latency_s": time.monotonic() - started,
        }
    return {
        "http_status": http_status,
        "transport_error": None,
        "raw_text": "\n".join(raw_lines),
        "raw_lines": raw_lines,
        "events": events,
        "done_count": done_count,
        "latency_s": time.monotonic() - started,
    }


def assemble_stream(
    endpoint: str, events: list[Any]
) -> dict[str, Any]:
    """Assemble text and count terminal signals; detect trace/content leaks."""
    chunks: list[str] = []
    finish_count = 0
    trace_like = 0
    leak = False
    leak_detail = ""
    for ev in events:
        if ev == "[DONE]":
            continue
        if not isinstance(ev, dict):
            continue
        keys_lower = {str(k).lower() for k in ev}
        is_trace_like = any("rerot" in k or "trace" in k for k in keys_lower)
        if endpoint == "/completion":
            if "content" in ev and isinstance(ev["content"], str):
                if is_trace_like and ev["content"]:
                    trace_like += 1
                    leak = True
                    leak_detail = "trace-like /completion event carried content"
                else:
                    chunks.append(ev["content"])
            if ev.get("stop") is True:
                finish_count += 1
        elif endpoint in ("/v1/completions", "/v1/chat/completions"):
            choices = ev.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0] if isinstance(choices[0], dict) else {}
                if endpoint == "/v1/completions":
                    text = first.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                    fr = first.get("finish_reason")
                    if fr is not None:
                        finish_count += 1
                else:
                    delta = first.get("delta", {})
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str):
                            if is_trace_like and content:
                                trace_like += 1
                                leak = True
                                leak_detail = "trace-like chat event carried delta.content"
                            else:
                                chunks.append(content)
                    fr = first.get("finish_reason")
                    if fr is not None:
                        finish_count += 1
            else:
                # Non-choice event (usage, trace, ping payload): must not carry
                # completion text. Flag only when it actually does.
                blob = json.dumps(ev, ensure_ascii=False)
                if is_trace_like:
                    trace_like += 1
                    for probe in ("delta", "\"content\""):
                        if probe in blob:
                            # Confirm non-empty content value before failing.
                            try:
                                if re.search(r'"content"\s*:\s*"[^"]', blob):
                                    leak = True
                                    leak_detail = "non-choice trace event carried content"
                            except Exception:
                                pass
        else:
            continue
    return {
        "assembled_text": "".join(chunks),
        "finish_count": finish_count,
        "trace_like_events": trace_like,
        "trace_leak": leak,
        "trace_leak_detail": leak_detail,
    }


def extract_nonstream_text(endpoint: str, body: Any) -> str | None:
    try:
        if endpoint == "/completion" and isinstance(body, dict):
            content = body.get("content")
            return content if isinstance(content, str) else None
        if endpoint == "/v1/completions" and isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                text = choices[0].get("text")
                return text if isinstance(text, str) else None
        if endpoint == "/v1/chat/completions" and isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and isinstance(p.get("text"), str)
                        ]
                        return "".join(parts)
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Owned server management (children only)
# --------------------------------------------------------------------------

def port_occupied(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


class OwnedServer:
    def __init__(self, cfg: dict[str, Any], log_path: Path) -> None:
        self.cfg = cfg
        self.log_path = log_path
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_fp: Any = None
        self.start_error: str | None = None
        self.exit_code: int | None = None

    def start(self) -> bool:
        launch = self.cfg["launch"]
        port = launch["port"]
        if port_occupied(port):
            self.start_error = f"port {port} occupied: refusing to launch or touch existing listener"
            return False
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.log_fp = open(self.log_path, "ab")
        except OSError as exc:
            self.start_error = f"cannot open log {self.log_path}: {exc}"
            return False
        try:
            self.proc = subprocess.Popen(
                launch["argv"],
                stdout=self.log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.start_error = f"spawn failed: {type(exc).__name__}: {exc}"
            return False
        base_url = f"http://127.0.0.1:{port}"
        api_key = self.cfg.get("api_key", "")
        deadline = time.monotonic() + launch["startup_timeout_s"]
        last_status: Any = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.exit_code = self.proc.returncode
                self.start_error = f"child exited during startup with code {self.exit_code}"
                return False
            try:
                status, _ = http_get_text(base_url, "/health", api_key, 5.0)
                last_status = status
                if status == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        if self.proc.poll() is not None:
            self.exit_code = self.proc.returncode
            self.start_error = f"child exited during startup with code {self.exit_code}"
        else:
            self.start_error = (
                f"health check timeout after {launch['startup_timeout_s']}s "
                f"(last status {last_status})"
            )
        return False

    def poll_exit(self) -> int | None:
        if self.proc is None:
            return self.exit_code
        code = self.proc.poll()
        if code is not None:
            self.exit_code = code
        return self.exit_code

    def stop(self, timeout: float) -> int | None:
        if self.proc is None:
            return self.exit_code
        if self.proc.poll() is not None:
            self.exit_code = self.proc.returncode
        else:
            try:
                self.proc.terminate()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=timeout)
                self.exit_code = self.proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=10)
                    self.exit_code = self.proc.returncode
                except Exception:
                    self.exit_code = self.proc.poll()
        try:
            if self.log_fp is not None:
                self.log_fp.flush()
                self.log_fp.close()
        except OSError:
            pass
        self.log_fp = None
        return self.exit_code


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------

class Evidence:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cases_dir = root / "cases"
        self.servers_dir = root / "servers"
        self.seq = 0
        self.f_requests: Any = None
        self.f_responses: Any = None
        self.f_timings: Any = None
        self.f_scores: Any = None
        self.f_memory: Any = None

    def open(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.servers_dir.mkdir(parents=True, exist_ok=True)
        self.f_requests = open(self.root / "requests.jsonl", "a", encoding="utf-8")
        self.f_responses = open(self.root / "responses.jsonl", "a", encoding="utf-8")
        self.f_timings = open(self.root / "timings.jsonl", "a", encoding="utf-8")
        self.f_scores = open(self.root / "scores.jsonl", "a", encoding="utf-8")
        self.f_memory = open(self.root / "memory.jsonl", "a", encoding="utf-8")

    def close(self) -> None:
        for fh in (self.f_requests, self.f_responses, self.f_timings, self.f_scores, self.f_memory):
            try:
                if fh is not None:
                    fh.flush()
                    fh.close()
            except OSError:
                pass

    def emit(self, fh: Any, obj: dict[str, Any]) -> None:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq


def git_info() -> tuple[str | None, str | None]:
    commit: str | None = None
    dirty: str | None = None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        if proc.returncode == 0 and proc.stdout.strip():
            commit = proc.stdout.strip()
    except Exception:
        commit = None
    try:
        proc = subprocess.run(
            ["git", "diff", "--binary"], capture_output=True, timeout=20
        )
        if proc.returncode == 0:
            dirty = sha256_bytes(proc.stdout or b"")
    except Exception:
        dirty = None
    return commit, dirty


# --------------------------------------------------------------------------
# Step execution
# --------------------------------------------------------------------------

def effective_api_key(server_cfg: dict[str, Any], cli_key: str) -> str:
    if server_cfg.get("api_key"):
        return server_cfg["api_key"]
    return cli_key


def server_base_url(server_cfg: dict[str, Any]) -> str:
    if server_cfg["endpoint"] is not None:
        return server_cfg["endpoint"]["base_url"]
    return f"http://127.0.0.1:{server_cfg['launch']['port']}"


def do_nonstream_request(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    status, raw, parsed, transport_error = http_post_json(base_url + endpoint, payload, api_key, timeout)
    latency = time.monotonic() - started
    ok = status == 200 and transport_error is None and parsed is not None
    text = extract_nonstream_text(endpoint, parsed) if ok else None
    return {
        "http_status": status,
        "transport_error": transport_error,
        "raw_bytes": len(raw),
        "raw_text": raw.decode("utf-8", errors="replace") if raw else "",
        "response_json": parsed,
        "assembled_text": text,
        "latency_s": latency,
        "ok": ok,
        "finish_count": None,
        "done_count": None,
    }


def run_case(  # noqa: C901 - sequential step dispatch with per-kind evidence
    case: dict[str, Any],
    server_cfg: dict[str, Any],
    requests: dict[str, dict[str, Any]],
    api_key: str,
    ev: Evidence,
    case_dir: Path,
) -> dict[str, Any]:
    base_url = server_base_url(server_cfg)
    step_results: list[dict[str, Any]] = []
    texts_in_order: list[str | None] = []
    case_ok = True
    case_error: str | None = None
    for idx, step in enumerate(case["steps"]):
        kind = step["kind"]
        if kind == "sleep":
            time.sleep(step["seconds"])
            step_results.append({"kind": kind, "ok": True, "seconds": step["seconds"]})
            continue
        if kind == "health_check":
            try:
                status, _ = http_get_text(base_url, "/health", api_key, step["timeout_s"])
            except Exception as exc:
                status = None
                case_ok = False
                step_results.append(
                    {"kind": kind, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            ok = status == 200
            if not ok:
                case_ok = False
            step_results.append({"kind": kind, "ok": ok, "http_status": status})
            continue
        if kind in ("slot_save", "slot_restore", "slot_erase"):
            slot_id = step["slot_id"]
            action = {"slot_save": "save", "slot_restore": "restore", "slot_erase": "erase"}[kind]
            url = f"{base_url}/slots/{slot_id}?action={action}"
            payload: dict[str, Any] = {}
            if kind in ("slot_save", "slot_restore"):
                payload = {"filename": step["filename"]}
            seq = ev.next_seq()
            started = time.monotonic()
            status, raw, parsed, transport_error = http_post_json(
                url, payload, api_key, step["timeout_s"]
            )
            latency = time.monotonic() - started
            ok = status == 200 and transport_error is None and isinstance(parsed, dict)
            if not ok:
                case_ok = False
            else:
                # Validate the documented response shape without inventing data.
                if kind == "slot_save" and ("n_saved" not in parsed or "filename" not in parsed):
                    ok = False
                    case_ok = False
                elif kind == "slot_restore" and (
                    "n_restored" not in parsed or "filename" not in parsed
                ):
                    ok = False
                    case_ok = False
                elif kind == "slot_erase" and "n_erased" not in parsed:
                    ok = False
                    case_ok = False
            ev.emit(
                ev.f_requests,
                {
                    "seq": seq, "case": case["id"], "step_index": idx, "step_kind": kind,
                    "request": None, "rep": None, "endpoint": f"/slots/{slot_id}?action={action}",
                    "payload": payload, "timeout_s": step["timeout_s"], "warmup": False,
                    "started_at": now_iso(), "server": server_cfg["id"],
                },
            )
            ev.emit(
                ev.f_responses,
                {
                    "seq": seq, "case": case["id"], "step_index": idx, "request": None,
                    "rep": None, "endpoint": f"/slots/{slot_id}?action={action}",
                    "payload": payload,
                    "http_status": status, "ok": ok, "latency_s": latency, "warmup": False,
                    "assembled_text": None, "raw_bytes": len(raw),
                    "response_json": parsed,
                    "error": transport_error,
                },
            )
            ev.emit(
                ev.f_timings,
                {
                    "seq": seq, "case": case["id"], "step_index": idx, "request": None,
                    "rep": None, "warmup": False,
                    "latency_s": latency, "client_wall_s": latency,
                    "timestamp": now_iso(), "server": server_cfg["id"], "ok": ok,
                },
            )
            step_results.append(
                {"kind": kind, "ok": ok, "http_status": status, "slot_id": slot_id,
                 "error": transport_error}
            )
            continue
        if kind == "request":
            req = requests[step["request"]]
            payload = req["payload"]
            if payload.get("stream") is True:
                case_ok = False
                step_results.append(
                    {"kind": kind, "ok": False, "request": req["id"],
                     "error": "non-stream step references a stream:true payload; use stream_request"}
                )
                texts_in_order.append(None)
                continue
            call_timeout = min(step["timeout_s"], req["timeout_s"])
            seq = ev.next_seq()
            ev.emit(
                ev.f_requests,
                {
                    "seq": seq, "case": case["id"], "step_index": idx, "step_kind": kind,
                    "request": req["id"], "rep": req.get("rep", 0),
                    "endpoint": req["endpoint"], "payload": payload,
                    "timeout_s": call_timeout, "warmup": step.get("warmup", False),
                    "started_at": now_iso(), "server": server_cfg["id"],
                },
            )
            res = do_nonstream_request(base_url, req["endpoint"], payload, api_key, call_timeout)
            ok = res["ok"]
            if not ok:
                case_ok = False
            ev.emit(
                ev.f_responses,
                {
                    "seq": seq, "case": case["id"], "step_index": idx, "request": req["id"],
                    "rep": req.get("rep", 0), "endpoint": req["endpoint"],
                    "payload": payload,
                    "http_status": res["http_status"], "ok": ok,
                    "latency_s": res["latency_s"], "warmup": step.get("warmup", False),
                    "assembled_text": res["assembled_text"], "raw_bytes": res["raw_bytes"],
                    "response_json": res["response_json"], "error": res["transport_error"],
                },
            )
            ev.emit(
                ev.f_timings,
                {
                    "seq": seq, "case": case["id"], "step_index": idx, "request": req["id"],
                    "rep": req.get("rep", 0),
                    "warmup": step.get("warmup", False), "latency_s": res["latency_s"],
                    "client_wall_s": res["latency_s"], "timestamp": now_iso(),
                    "server": server_cfg["id"], "ok": ok,
                },
            )
            write_score(ev, case["id"], idx, req, res["assembled_text"], res["response_json"], ok)
            texts_in_order.append(res["assembled_text"])
            step_results.append(
                {"kind": kind, "ok": ok, "request": req["id"],
                 "http_status": res["http_status"], "error": res["transport_error"]}
            )
            continue
        if kind == "concurrent_requests":
            reqs = [requests[r] for r in step["requests"]]
            for req in reqs:
                if req["payload"].get("stream") is True:
                    case_ok = False
                    step_results.append(
                        {"kind": kind, "ok": False,
                         "error": f"concurrent step references stream:true payload {req['id']!r}"}
                    )
                    break
            else:
                wall_start = time.monotonic()
                seqs = [ev.next_seq() for _ in reqs]
                for seq, req in zip(seqs, reqs):
                    ev.emit(
                        ev.f_requests,
                        {
                            "seq": seq, "case": case["id"], "step_index": idx,
                            "step_kind": kind, "request": req["id"], "rep": req.get("rep", 0),
                            "endpoint": req["endpoint"],
                            "payload": req["payload"],
                            "timeout_s": min(step["timeout_s"], req["timeout_s"]),
                            "warmup": step.get("warmup", False), "started_at": now_iso(),
                            "server": server_cfg["id"],
                        },
                    )

                def _one(arg: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any]]:
                    seq, req = arg
                    call_timeout = min(step["timeout_s"], req["timeout_s"])
                    res = do_nonstream_request(
                        base_url, req["endpoint"], req["payload"], api_key, call_timeout
                    )
                    return seq, req, res

                results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=step["max_workers"]
                ) as pool:
                    futs = [pool.submit(_one, (seq, req)) for seq, req in zip(seqs, reqs)]
                    for fut in concurrent.futures.as_completed(futs):
                        try:
                            results.append(fut.result())
                        except Exception as exc:  # pragma: no cover - defensive
                            case_ok = False
                            step_results.append(
                                {"kind": kind, "ok": False,
                                 "error": f"worker exception: {type(exc).__name__}: {exc}"}
                            )
                wall = time.monotonic() - wall_start
                sub_ok_all = True
                for seq, req, res in sorted(results, key=lambda t: t[0]):
                    ok = res["ok"]
                    sub_ok_all = sub_ok_all and ok
                    ev.emit(
                        ev.f_responses,
                        {
                            "seq": seq, "case": case["id"], "step_index": idx,
                            "request": req["id"], "rep": req.get("rep", 0),
                            "endpoint": req["endpoint"],
                            "payload": req["payload"],
                            "http_status": res["http_status"], "ok": ok,
                            "latency_s": res["latency_s"],
                            "warmup": step.get("warmup", False),
                            "assembled_text": res["assembled_text"],
                            "raw_bytes": res["raw_bytes"], "response_json": res["response_json"],
                            "error": res["transport_error"],
                        },
                    )
                    ev.emit(
                        ev.f_timings,
                        {
                            "seq": seq, "case": case["id"], "step_index": idx,
                            "request": req["id"], "rep": req.get("rep", 0),
                            "warmup": step.get("warmup", False), "latency_s": res["latency_s"],
                            "client_wall_s": wall, "timestamp": now_iso(),
                            "server": server_cfg["id"], "ok": ok,
                        },
                    )
                    write_score(
                        ev, case["id"], idx, req, res["assembled_text"],
                        res["response_json"], ok,
                    )
                    texts_in_order.append(res["assembled_text"])
                if not sub_ok_all:
                    case_ok = False
                step_results.append(
                    {"kind": kind, "ok": sub_ok_all,
                     "requests": [r["id"] for r in reqs], "client_wall_s": wall}
                )
                continue
            texts_in_order.extend([None] * len(step["requests"]))
            continue
        if kind in ("stream_request", "cancel_stream"):
            req = requests[step["request"]]
            payload = req["payload"]
            if payload.get("stream") is not True:
                case_ok = False
                step_results.append(
                    {"kind": kind, "ok": False, "request": req["id"],
                     "error": "stream step references a non-stream payload; set stream:true"}
                )
                texts_in_order.append(None)
                continue
            if req["endpoint"] not in STREAMING_ENDPOINTS:
                case_ok = False
                step_results.append(
                    {"kind": kind, "ok": False, "request": req["id"],
                     "error": f"endpoint {req['endpoint']} does not support streaming"}
                )
                texts_in_order.append(None)
                continue
            seq = ev.next_seq()
            call_timeout = min(step["timeout_s"], req["timeout_s"])
            ev.emit(
                ev.f_requests,
                {
                    "seq": seq, "case": case["id"], "step_index": idx, "step_kind": kind,
                    "request": req["id"], "rep": req.get("rep", 0),
                    "endpoint": req["endpoint"], "payload": payload,
                    "timeout_s": call_timeout, "warmup": step.get("warmup", False),
                    "started_at": now_iso(), "server": server_cfg["id"],
                },
            )
            if kind == "stream_request":
                res = read_sse_stream(base_url + req["endpoint"], payload, api_key, call_timeout)
                sse_file = case_dir / f"sse-{seq}.txt"
                try:
                    sse_file.write_text(res.get("raw_text", ""), encoding="utf-8")
                except OSError as exc:
                    case_ok = False
                    step_results.append(
                        {"kind": kind, "ok": False, "request": req["id"],
                         "error": f"cannot write SSE evidence: {exc}"}
                    )
                    texts_in_order.append(None)
                    continue
                asm = assemble_stream(req["endpoint"], res.get("events", []))
                expect_stream = step["expect_stream"]
                http_ok = res.get("http_status") == 200 and res.get("transport_error") is None
                # Raw /completion streams end after stop:true with no [DONE]
                # sentinel; OpenAI-compatible streams end with data: [DONE].
                want_done = 0 if req["endpoint"] == "/completion" else 1
                terminal_ok = (
                    asm["finish_count"] == 1 and res.get("done_count") == want_done
                )
                if expect_stream == "success":
                    ok = http_ok and terminal_ok and not asm["trace_leak"]
                    err = None
                    if not http_ok:
                        err = (
                            f"HTTP {res.get('http_status')} "
                            f"{res.get('transport_error') or ''}".strip()
                        )
                    elif asm["trace_leak"]:
                        err = asm["trace_leak_detail"]
                    elif not terminal_ok:
                        err = (
                            "stream terminal contract violated: "
                            f"endpoint={req['endpoint']} "
                            f"finish={asm['finish_count']} done={res.get('done_count')} "
                            f"(want_done={want_done})"
                        )
                else:
                    # error_or_cancel: record the outcome, require evidence that
                    # it was NOT disguised as a clean success.
                    clean = http_ok and terminal_ok and not asm["trace_leak"]
                    if clean:
                        ok = False
                        err = "expected error/cancel stream but observed a clean success stream"
                    else:
                        ok = http_ok or res.get("http_status") is not None or bool(
                            res.get("transport_error")
                        ) or bool(res.get("raw_text"))
                        err = None if ok else "no stream evidence captured"
                        if asm["trace_leak"]:
                            ok = False
                            err = asm["trace_leak_detail"]
                if not ok:
                    case_ok = False
                ev.emit(
                    ev.f_responses,
                    {
                        "seq": seq, "case": case["id"], "step_index": idx,
                        "request": req["id"], "rep": req.get("rep", 0),
                        "endpoint": req["endpoint"],
                        "payload": payload,
                        "http_status": res.get("http_status"), "ok": ok,
                        "latency_s": res.get("latency_s", 0.0),
                        "warmup": step.get("warmup", False),
                        "assembled_text": asm["assembled_text"],
                        "raw_bytes": len(res.get("raw_text", "")),
                        "raw_file": str(sse_file.name),
                        "finish_count": asm["finish_count"],
                        "done_count": res.get("done_count"),
                        "trace_like_events": asm["trace_like_events"],
                        "error": err or res.get("transport_error"),
                    },
                )
                ev.emit(
                    ev.f_timings,
                    {
                        "seq": seq, "case": case["id"], "step_index": idx,
                        "request": req["id"], "rep": req.get("rep", 0),
                        "warmup": step.get("warmup", False),
                        "latency_s": res.get("latency_s", 0.0),
                        "client_wall_s": res.get("latency_s", 0.0),
                        "timestamp": now_iso(), "server": server_cfg["id"], "ok": ok,
                    },
                )
                write_score(
                    ev, case["id"], idx, req, asm["assembled_text"], None, ok,
                    stream_info={
                        "finish_count": asm["finish_count"],
                        "done_count": res.get("done_count"),
                    },
                )
                texts_in_order.append(asm["assembled_text"])
                step_results.append(
                    {"kind": kind, "ok": ok, "request": req["id"],
                     "http_status": res.get("http_status"),
                     "finish_count": asm["finish_count"],
                     "done_count": res.get("done_count"), "error": err}
                )
            else:  # cancel_stream
                cancel_res = run_cancel_stream(
                    base_url + req["endpoint"], payload, api_key,
                    call_timeout, step["cancel_after_s"],
                )
                sse_file = case_dir / f"sse-{seq}-cancel.txt"
                try:
                    sse_file.write_text(cancel_res.get("partial_text", ""), encoding="utf-8")
                except OSError as exc:
                    case_ok = False
                    step_results.append(
                        {"kind": kind, "ok": False, "request": req["id"],
                         "error": f"cannot write cancel evidence: {exc}"}
                    )
                    texts_in_order.append(None)
                    continue
                ok = cancel_res["ok"]
                if not ok:
                    case_ok = False
                # A cancel must not break the server for the next request.
                try:
                    h_status, _ = http_get_text(base_url, "/health", api_key, 15.0)
                except Exception:
                    h_status = None
                if h_status != 200:
                    ok = False
                    case_ok = False
                ev.emit(
                    ev.f_responses,
                    {
                        "seq": seq, "case": case["id"], "step_index": idx,
                        "request": req["id"], "rep": req.get("rep", 0),
                        "endpoint": req["endpoint"],
                        "payload": payload,
                        "http_status": cancel_res.get("http_status"), "ok": ok,
                        "latency_s": cancel_res.get("elapsed_s", 0.0),
                        "warmup": False, "assembled_text": None,
                        "raw_bytes": cancel_res.get("partial_bytes", 0),
                        "raw_file": str(sse_file.name),
                        "cancelled": True,
                        "health_after_cancel": h_status,
                        "error": cancel_res.get("error"),
                    },
                )
                ev.emit(
                    ev.f_timings,
                    {
                        "seq": seq, "case": case["id"], "step_index": idx,
                        "request": req["id"], "rep": req.get("rep", 0),
                        "warmup": False, "latency_s": cancel_res.get("elapsed_s", 0.0),
                        "client_wall_s": cancel_res.get("elapsed_s", 0.0),
                        "timestamp": now_iso(), "server": server_cfg["id"], "ok": ok,
                    },
                )
                texts_in_order.append(None)
                step_results.append(
                    {"kind": kind, "ok": ok, "request": req["id"],
                     "partial_bytes": cancel_res.get("partial_bytes", 0),
                     "health_after_cancel": h_status, "error": cancel_res.get("error")}
                )
            continue
        raise HarnessError(f"case {case['id']!r}: unhandled step kind {kind!r}")  # pragma: no cover
    return {
        "steps": step_results,
        "texts": texts_in_order,
        "transport_ok": case_ok,
        "transport_error": case_error,
    }


def run_cancel_stream(
    url: str, payload: dict[str, Any], api_key: str, timeout: float, cancel_after_s: float
) -> dict[str, Any]:
    started = time.monotonic()
    data = json.dumps(payload).encode("utf-8")
    headers = dict(auth_headers(api_key))
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "text/event-stream"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    partial = b""
    http_status: int | None = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        http_status = getattr(resp, "status", 200)
        with resp:
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= cancel_after_s:
                    break
                if elapsed >= timeout:
                    return {
                        "ok": False, "http_status": http_status,
                        "partial_bytes": len(partial), "partial_text": "",
                        "elapsed_s": time.monotonic() - started,
                        "error": "cancel window exceeded step timeout before cancel",
                    }
                try:
                    chunk = resp.read(4096)
                except Exception as exc:
                    partial_note = f"read error before cancel: {type(exc).__name__}: {exc}"
                    return {
                        "ok": bool(partial), "http_status": http_status,
                        "partial_bytes": len(partial),
                        "partial_text": partial.decode("utf-8", errors="replace"),
                        "elapsed_s": time.monotonic() - started,
                        "error": partial_note if not partial else None,
                    }
                if not chunk:
                    break
                partial += chunk
                if time.monotonic() - started >= cancel_after_s:
                    break
        elapsed = time.monotonic() - started
        # Client-side close == cancel. Any bytes (or a clean immediate finish
        # for tiny outputs) count as evidence; zero bytes with an open stream
        # still proves the cancel path was exercised.
        return {
            "ok": True, "http_status": http_status, "partial_bytes": len(partial),
            "partial_text": partial.decode("utf-8", errors="replace"),
            "elapsed_s": elapsed, "error": None,
        }
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return {
            "ok": True, "http_status": exc.code, "partial_bytes": len(body),
            "partial_text": body.decode("utf-8", errors="replace"),
            "elapsed_s": time.monotonic() - started,
            "error": None,  # HTTP error during cancel window is itself cancel evidence
        }
    except Exception as exc:
        # Connection aborted by server on cancel is acceptable evidence only
        # when we already issued the request; otherwise it is a real failure.
        return {
            "ok": False, "http_status": http_status, "partial_bytes": len(partial),
            "partial_text": partial.decode("utf-8", errors="replace"),
            "elapsed_s": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_score(
    ev: Evidence,
    case_id: str,
    step_index: int,
    req: dict[str, Any],
    assembled_text: str | None,
    response_json: Any,
    ok: bool,
    stream_info: dict[str, Any] | None = None,
) -> None:
    hint = req.get("expected_substring")
    if hint is None:
        ev.emit(
            ev.f_scores,
            {
                "case": case_id, "request": req["id"], "rep": req.get("rep", 0),
                "step_index": step_index,
                "scorer": "matrix-harness-v1", "score": None, "status": "skip",
                "reason": "no expected_substring defined; structural http/stream checks only",
            },
        )
        return
    if not ok or not isinstance(assembled_text, str):
        ev.emit(
            ev.f_scores,
            {
                "case": case_id, "request": req["id"], "rep": req.get("rep", 0),
                "step_index": step_index,
                "scorer": "matrix-harness-v1", "score": 0.0, "status": "fail",
                "reason": "request failed or produced no text",
            },
        )
        return
    hit = hint in assembled_text
    ev.emit(
        ev.f_scores,
        {
            "case": case_id, "request": req["id"], "rep": req.get("rep", 0),
            "step_index": step_index,
            "scorer": "matrix-harness-v1", "score": 1.0 if hit else 0.0,
            "status": "pass" if hit else "fail",
            "reason": "expected_substring present" if hit else "expected_substring absent",
        },
    )


# --------------------------------------------------------------------------
# Expectation evaluation (no missing-metric-to-zero)
# --------------------------------------------------------------------------

def evaluate_case_expectations(
    case: dict[str, Any],
    before: dict[str, float],
    after: dict[str, float],
    transport_ok: bool,
    texts: list[str | None],
    peer_texts: dict[str, list[str | None]],
) -> tuple[str, list[str], dict[str, Any]]:
    expect = case["expect"]
    reasons: list[str] = []
    missing: list[str] = []
    checks: dict[str, Any] = {}
    failed = False

    def note_missing(selector: str) -> None:
        bare = selector.split("{", 1)[0]
        missing.append(selector)
        if bare in expect["skip_if_metrics_missing"] or selector in expect["skip_if_metrics_missing"]:
            checks.setdefault("skipped_missing", []).append(selector)
        else:
            reasons.append(f"missing_metric: {selector}")
            checks.setdefault("missing_failed", []).append(selector)

    if not transport_ok:
        failed = True
        reasons.append("transport: one or more steps failed (see responses.jsonl)")
    checks["transport_ok"] = transport_ok

    # Old-binary baseline: absence of every flashprefill_* counter is expected
    # and passes by construction; presence FAILs as unexpected. Validation
    # already bans flashprefill_* references, invariants, dense constraints and
    # off_baseline for such cases, so only transport, non-flashprefill metric
    # checks, rerot helpers and match_case below still apply.
    if expect["old_binary_baseline"]:
        fp_present = sorted(
            {
                decompose_series_key(k)[0]
                for k in set(before) | set(after)
                if decompose_series_key(k)[0].startswith("flashprefill_")
            }
        )
        if fp_present:
            failed = True
            reasons.append(
                f"old_binary_baseline: unexpected flashprefill_* counters: {fp_present}"
            )
            checks["old_baseline_no_counters"] = "present-unexpected"
        else:
            checks["old_baseline_no_counters"] = "absent-expected"

    # Required metric deltas.
    for sel, cond in expect["require_metrics"].items():
        info = selector_delta(before, after, sel)
        if info["missing"]:
            note_missing(sel)
            checks[f"require:{sel}"] = "missing"
            continue
        if info["nonfinite"]:
            failed = True
            reasons.append(f"nonfinite_metric: {sel}")
            checks[f"require:{sel}"] = "nonfinite"
            continue
        delta = info["delta"]
        assert delta is not None
        ok = True
        if "min_delta" in cond and not delta >= cond["min_delta"] - 1e-12:
            ok = False
        if "max_delta" in cond and not delta <= cond["max_delta"] + 1e-12:
            ok = False
        if "exact_delta" in cond and not abs(delta - cond["exact_delta"]) <= 1e-9:
            ok = False
        checks[f"require:{sel}"] = {"delta": delta, "cond": cond, "ok": ok}
        if not ok:
            failed = True
            reasons.append(f"metric_delta: {sel} delta={delta} cond={cond}")

    # Forbidden increases.
    for sel in expect["forbid_metrics_increase"]:
        info = selector_delta(before, after, sel)
        if info["missing"]:
            note_missing(sel)
            checks[f"forbid:{sel}"] = "missing"
            continue
        if info["nonfinite"]:
            failed = True
            reasons.append(f"nonfinite_metric: {sel}")
            checks[f"forbid:{sel}"] = "nonfinite"
            continue
        delta = info["delta"]
        assert delta is not None
        ok = delta <= 1e-12
        checks[f"forbid:{sel}"] = {"delta": delta, "ok": ok}
        if not ok:
            failed = True
            reasons.append(f"forbidden_increase: {sel} delta={delta}")

    # Invariants: visible >= exact; sparse rows imply corrected blocks.
    if expect["invariants"]["visible_ge_exact"]:
        vinf = selector_delta(before, after, "flashprefill_visible_tokens_total")
        einf = selector_delta(before, after, "flashprefill_exact_tokens_total")
        if vinf["missing"]:
            note_missing("flashprefill_visible_tokens_total")
            checks["invariant:visible_ge_exact"] = "missing-visible"
        elif einf["missing"]:
            note_missing("flashprefill_exact_tokens_total")
            checks["invariant:visible_ge_exact"] = "missing-exact"
        elif vinf["nonfinite"] or einf["nonfinite"]:
            failed = True
            reasons.append("nonfinite_metric in visible/exact invariant")
            checks["invariant:visible_ge_exact"] = "nonfinite"
        else:
            assert vinf["delta"] is not None and einf["delta"] is not None
            ok = vinf["delta"] + 1e-12 >= einf["delta"]
            checks["invariant:visible_ge_exact"] = {
                "visible_delta": vinf["delta"], "exact_delta": einf["delta"], "ok": ok
            }
            if not ok:
                failed = True
                reasons.append(
                    f"invariant visible_ge_exact violated: visible={vinf['delta']} "
                    f"exact={einf['delta']}"
                )
    if expect["invariants"]["corrected_positive_if_sparse"]:
        sinf = selector_delta(before, after, "flashprefill_sparse_rows_total")
        cinf = selector_delta(before, after, "flashprefill_corrected_blocks_total")
        if sinf["missing"]:
            note_missing("flashprefill_sparse_rows_total")
            checks["invariant:corrected_positive_if_sparse"] = "missing-sparse"
        elif cinf["missing"]:
            note_missing("flashprefill_corrected_blocks_total")
            checks["invariant:corrected_positive_if_sparse"] = "missing-corrected"
        elif sinf["nonfinite"] or cinf["nonfinite"]:
            failed = True
            reasons.append("nonfinite_metric in sparse/corrected invariant")
            checks["invariant:corrected_positive_if_sparse"] = "nonfinite"
        else:
            assert sinf["delta"] is not None and cinf["delta"] is not None
            ok = not (sinf["delta"] > 1e-12 and cinf["delta"] <= 1e-12)
            checks["invariant:corrected_positive_if_sparse"] = {
                "sparse_delta": sinf["delta"], "corrected_delta": cinf["delta"], "ok": ok
            }
            if not ok:
                failed = True
                reasons.append(
                    f"invariant corrected_positive_if_sparse violated: sparse={sinf['delta']} "
                    f"corrected={cinf['delta']}"
                )

    # Dense reasons.
    if expect["dense_reasons_allowed"] is not None or expect["expect_dense_reason"] is not None:
        dinf = selector_delta(before, after, "flashprefill_dense_rows_total")
        if dinf["missing"]:
            # Only a failure when the case constrains dense routing; a sparse-only
            # case on a server without dense counters leaves this unchecked only
            # when no dense expectation was stated.
            if expect["expect_dense_reason"] is not None:
                note_missing("flashprefill_dense_rows_total")
                checks["dense"] = "missing"
            else:
                checks["dense"] = "missing-unchecked"
        elif dinf["nonfinite"]:
            failed = True
            reasons.append("nonfinite_metric: flashprefill_dense_rows_total")
            checks["dense"] = "nonfinite"
        else:
            per_reason = dense_reason_deltas(before, after)
            checks["dense"] = {"total_delta": dinf["delta"], "per_reason": per_reason}
            if expect["dense_reasons_allowed"] is not None and (dinf["delta"] or 0.0) > 1e-12:
                for reason, dval in per_reason.items():
                    if dval > 1e-12 and reason not in expect["dense_reasons_allowed"]:
                        failed = True
                        reasons.append(
                            f"dense_reason: unexpected reason {reason!r} delta={dval} "
                            f"(allowed {expect['dense_reasons_allowed']})"
                        )
            if expect["expect_dense_reason"] is not None:
                got = per_reason.get(expect["expect_dense_reason"], 0.0)
                if got <= 1e-12:
                    failed = True
                    reasons.append(
                        f"dense_reason: expected reason {expect['expect_dense_reason']!r} "
                        f"with positive delta, got {got} (per_reason {per_reason})"
                    )

    # OFF baseline: every flashprefill_* delta must be zero.
    if expect["off_baseline"]:
        fp_names = sorted(
            {
                decompose_series_key(k)[0]
                for k in set(before) | set(after)
                if decompose_series_key(k)[0].startswith("flashprefill_")
            }
        )
        if not fp_names:
            for probe in (
                "flashprefill_sparse_rows_total",
                "flashprefill_corrected_blocks_total",
                "flashprefill_dense_rows_total",
            ):
                note_missing(probe)
            checks["off_baseline"] = "missing-counters"
        else:
            nonzero: list[str] = []
            for name in fp_names:
                info = selector_delta(before, after, name)
                if info["missing"] or info["delta"] is None:
                    continue
                if abs(info["delta"]) > 1e-12:
                    nonzero.append(f"{name} delta={info['delta']}")
            checks["off_baseline"] = {"counters": fp_names, "nonzero": nonzero}
            if nonzero:
                failed = True
                reasons.append(f"off_baseline violated: {'; '.join(nonzero)}")

    # RERoT helpers.
    rerot = expect["rerot"]
    if rerot["min_completed_episodes"] > 0 or rerot["require_no_hard_aborts"]:
        if rerot["min_completed_episodes"] > 0:
            einf = selector_delta(before, after, "rerot_completed_episode_total")
            if einf["missing"]:
                note_missing("rerot_completed_episode_total")
                checks["rerot:episodes"] = "missing"
            elif einf["nonfinite"]:
                failed = True
                reasons.append("nonfinite_metric: rerot_completed_episode_total")
                checks["rerot:episodes"] = "nonfinite"
            else:
                assert einf["delta"] is not None
                ok = einf["delta"] + 1e-12 >= rerot["min_completed_episodes"]
                checks["rerot:episodes"] = {"delta": einf["delta"], "ok": ok}
                if not ok:
                    failed = True
                    reasons.append(
                        f"rerot: completed_episodes delta={einf['delta']} < "
                        f"min {rerot['min_completed_episodes']}"
                    )
        if rerot["require_no_hard_aborts"]:
            hinf = selector_delta(before, after, "rerot_hard_aborts")
            if hinf["missing"]:
                note_missing("rerot_hard_aborts")
                checks["rerot:hard_aborts"] = "missing"
            else:
                assert hinf["delta"] is not None
                ok = hinf["delta"] <= 1e-12
                checks["rerot:hard_aborts"] = {"delta": hinf["delta"], "ok": ok}
                if not ok:
                    failed = True
                    reasons.append(f"rerot: hard_aborts delta={hinf['delta']}")

    # Exact-output match against a peer case (OFF baseline comparison helper).
    if expect["match_case"] is not None:
        peer = expect["match_case"]
        if peer not in peer_texts:
            failed = True
            reasons.append(f"match_case: peer {peer!r} has no recorded outputs yet")
            checks["match_case"] = "peer-not-run"
        else:
            mine = [t for t in texts]
            theirs = peer_texts[peer]
            ok = mine == theirs
            checks["match_case"] = {
                "peer": peer, "ok": ok, "n_mine": len(mine), "n_peer": len(theirs)
            }
            if not ok:
                failed = True
                reasons.append(
                    f"match_case: assembled outputs differ from case {peer!r} "
                    f"({len(mine)} vs {len(theirs)} texts; see responses.jsonl)"
                )

    only_skippable_missing = bool(missing) and not failed and bool(
        checks.get("skipped_missing")
    ) and not checks.get("missing_failed")
    if failed:
        status = STATUS_FAIL
    elif only_skippable_missing:
        # Every missing metric was explicitly allowlisted for SKIP, and every
        # observed check passed: report SKIP, never PASS.
        status = STATUS_SKIP
        reasons.append(f"skip: missing metrics allowlisted: {sorted(set(missing))}")
    elif missing:
        # Defensive: missing entries always resolve above; reaching here means a
        # bookkeeping error, which must not become a silent PASS.
        status = STATUS_FAIL
        reasons.append(f"missing_metric unresolved: {sorted(set(missing))}")
    else:
        status = STATUS_PASS
    return status, reasons, checks


# --------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashPrefill full-compatibility matrix runner (PREFILL 17/18.6). "
        "See module docstring for the --matrix/--requests schemas, output layout, "
        "streaming contract, and exit codes.",
    )
    parser.add_argument("--matrix", type=Path, required=True, help="Matrix JSON file (servers/cases/comparisons).")
    parser.add_argument("--requests", type=Path, required=True, help="Requests JSON file (fixed named request set).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Independent output directory.")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""),
                        help="Default Bearer key (default: $LLAMA_API_KEY or empty).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty output directory.")
    return parser.parse_args(argv)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise HarnessError(f"--output-dir {path} exists and is not a directory")
        entries = list(path.iterdir())
        if entries and not overwrite:
            raise HarnessError(
                f"--output-dir {path} is non-empty; pass --overwrite to reuse it"
            )
    else:
        try:
            path.mkdir(parents=True)
        except OSError as exc:
            raise HarnessError(f"cannot create --output-dir {path}: {exc}") from exc


def fetch_metrics_snapshot(base_url: str, api_key: str, timeout: float) -> tuple[str, dict[str, float], str | None]:
    try:
        status, text = http_get_text(base_url, "/metrics", api_key, timeout)
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"
    if status != 200:
        return text, {}, f"GET /metrics returned HTTP {status}"
    return text, parse_metrics_text(text), None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ensure_output_dir(args.output_dir, args.overwrite)
    except HarnessError as exc:
        eprint(f"error: {exc}")
        return 1
    out = args.output_dir
    try:
        matrix_doc = load_json_strict(args.matrix)
        requests_doc = load_json_strict(args.requests)
        matrix = validate_matrix_doc(matrix_doc)
        requests = validate_requests_doc(requests_doc)
        validate_cross_refs(matrix, requests)
    except HarnessError as exc:
        eprint(f"error: {exc}")
        return 1

    server_index = {s["id"]: s for s in matrix["servers"]}
    cli_key = args.api_key or ""
    ev = Evidence(out)
    try:
        ev.open()
    except OSError as exc:
        eprint(f"error: cannot open evidence files: {exc}")
        return 1

    invocation = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
    matrix_sha = sha256_file(args.matrix)
    requests_sha = sha256_file(args.requests)
    commit, dirty = git_info()

    # Config identity per server (endpoint URL or canonical launch argv+port).
    for srv in matrix["servers"]:
        if srv["endpoint"] is not None:
            ident = json.dumps({"endpoint": srv["endpoint"]}, sort_keys=True).encode()
        else:
            ident = json.dumps({"launch": srv["launch"]}, sort_keys=True).encode()
        srv["config_identity_sha256"] = sha256_bytes(ident)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "invocation": invocation,
        "matrix_file": str(args.matrix),
        "requests_file": str(args.requests),
        "matrix_sha256": matrix_sha,
        "requests_sha256": requests_sha,
        "git_commit": commit,
        "git_diff_sha256": dirty,
        "meta": matrix["meta"],
        "servers": [
            {
                "id": s["id"],
                "mode": "endpoint" if s["endpoint"] else "launch",
                "endpoint": s["endpoint"],
                "launch_argv": s["launch"]["argv"] if s["launch"] else None,
                "launch_port": s["launch"]["port"] if s["launch"] else None,
                "cache_domain": s["cache_domain"],
                "config_identity_sha256": s["config_identity_sha256"],
                "request_timeout_s": s["request_timeout_s"],
                "metrics_timeout_s": s["metrics_timeout_s"],
            }
            for s in matrix["servers"]
        ],
        "cases": [{"id": c["id"], "server": c["server"]} for c in matrix["cases"]],
        "notes": (
            "Same-server cases share server state by design (slot/cache/sequence "
            "continuity). A/B pairs (comparisons, match_case) are validated to run "
            "on different servers with different cache_domain values; launch argv, "
            "config identity, /props capacity, and per-request cache_prompt/id_slot "
            "are recorded as the isolation evidence. Payloads are sent verbatim; "
            "the harness never overrides request fields."
        ),
    }
    try:
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        eprint(f"error: cannot write manifest: {exc}")
        ev.close()
        return 1

    owned: dict[str, OwnedServer] = {}
    server_failed: dict[str, str] = {}
    server_base: dict[str, str] = {}
    server_metrics_before_raw: dict[str, str] = {}
    server_metrics_after_raw: dict[str, str] = {}
    server_props: dict[str, Any] = {}
    overall_rc = 0

    def cleanup_owned() -> None:
        for sid, handle in owned.items():
            cfg = server_index[sid]
            try:
                handle.stop(cfg["launch"]["shutdown_timeout_s"] if cfg["launch"] else 10.0)
            except Exception:
                pass

    try:
        # --- start/fetch servers -------------------------------------------------
        for srv in matrix["servers"]:
            sid = srv["id"]
            key = effective_api_key(srv, cli_key)
            if srv["launch"] is not None:
                log_path = out / "servers" / sid / "server.log"
                handle = OwnedServer(srv, log_path)
                print(f"[matrix] starting owned server {sid!r} on port {srv['launch']['port']} ...")
                ok = handle.start()
                owned[sid] = handle
                if not ok:
                    server_failed[sid] = handle.start_error or "unknown startup failure"
                    eprint(f"error: owned server {sid!r}: {server_failed[sid]}")
                    continue
                base = f"http://127.0.0.1:{srv['launch']['port']}"
                server_base[sid] = base
                print(f"[matrix] server {sid!r} ready at {base}")
            else:
                base = srv["endpoint"]["base_url"]
                server_base[sid] = base
                try:
                    status, _ = http_get_text(base, "/health", key, srv["metrics_timeout_s"])
                except Exception as exc:
                    server_failed[sid] = f"endpoint unreachable: {type(exc).__name__}: {exc}"
                    eprint(f"error: endpoint server {sid!r}: {server_failed[sid]}")
                    continue
                if status != 200:
                    server_failed[sid] = f"endpoint /health returned HTTP {status}"
                    eprint(f"error: endpoint server {sid!r}: {server_failed[sid]}")
                    continue
            # props (best effort) + run-level metrics-before snapshot
            try:
                p_status, p_body = http_get_json(server_base[sid], "/props", key, srv["metrics_timeout_s"])
                server_props[sid] = {"http_status": p_status, "body": p_body}
            except Exception as exc:
                server_props[sid] = {"http_status": None, "error": f"{type(exc).__name__}: {exc}"}
            raw, parsed, err = fetch_metrics_snapshot(
                server_base[sid], key, srv["metrics_timeout_s"]
            )
            server_metrics_before_raw[sid] = raw
            sdir = out / "servers" / sid
            sdir.mkdir(parents=True, exist_ok=True)
            try:
                (sdir / "metrics-before.txt").write_text(raw, encoding="utf-8")
                (sdir / "props-before.json").write_text(
                    json.dumps(server_props[sid], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                raise HarnessError(f"cannot write server evidence for {sid!r}: {exc}")
            if err:
                eprint(f"warning: server {sid!r} metrics-before failed: {err}")
            # memory snapshot at start
            try:
                _, slots_body = http_get_json(
                    server_base[sid], "/slots", key, srv["metrics_timeout_s"]
                )
            except Exception:
                slots_body = None
            ev.emit(
                ev.f_memory,
                {
                    "timestamp": now_iso(), "scope": "server-start", "server": sid,
                    "case": None,
                    "metrics_snapshot": {
                        k: v for k, v in list(parsed.items())[:256]
                    } if parsed else {},
                    "metrics_error": err, "slots": slots_body,
                },
            )

        # --- run cases in matrix order ------------------------------------------
        case_results: list[dict[str, Any]] = []
        peer_texts: dict[str, list[str | None]] = {}
        for case in matrix["cases"]:
            cid = case["id"]
            sid = case["server"]
            srv = server_index[sid]
            key = effective_api_key(srv, cli_key)
            cdir = out / "cases" / cid
            cdir.mkdir(parents=True, exist_ok=True)
            print(f"[matrix] case {cid!r} on server {sid!r} ...")
            if sid in server_failed:
                status = STATUS_FAIL
                reasons = [f"server_unavailable: {server_failed[sid]}"]
                checks = {"server": "unavailable"}
                before_parsed: dict[str, float] = {}
                after_parsed: dict[str, float] = {}
                (cdir / "metrics-before.txt").write_text("", encoding="utf-8")
                (cdir / "metrics-after.txt").write_text("", encoding="utf-8")
                (cdir / "deltas.json").write_text(
                    json.dumps({"error": reasons[0]}, indent=2) + "\n", encoding="utf-8"
                )
                peer_texts[cid] = []
                case_results.append(
                    {
                        "id": cid, "server": sid, "status": status, "reasons": reasons,
                        "checks": checks, "metrics_delta": {},
                        "missing_metrics": [], "n_steps": len(case["steps"]),
                    }
                )
                print(f"[matrix] case {cid!r}: {status} ({reasons[0]})")
                continue
            # owned child may have died mid-run: failed exit is a case failure.
            if sid in owned:
                exit_code = owned[sid].poll_exit()
                if exit_code is not None:
                    msg = f"owned server exited with code {exit_code} before case {cid!r}"
                    server_failed[sid] = msg
                    eprint(f"error: {msg}")
                    status = STATUS_FAIL
                    reasons = [f"server_unavailable: {msg}"]
                    peer_texts[cid] = []
                    case_results.append(
                        {
                            "id": cid, "server": sid, "status": status, "reasons": reasons,
                            "checks": {"server": "exited"}, "metrics_delta": {},
                            "missing_metrics": [], "n_steps": len(case["steps"]),
                        }
                    )
                    print(f"[matrix] case {cid!r}: {status} ({reasons[0]})")
                    continue
            before_raw, before_parsed, before_err = fetch_metrics_snapshot(
                server_base[sid], key, srv["metrics_timeout_s"]
            )
            try:
                (cdir / "metrics-before.txt").write_text(before_raw, encoding="utf-8")
            except OSError as exc:
                raise HarnessError(f"cannot write case evidence for {cid!r}: {exc}")
            if before_err:
                eprint(f"warning: case {cid!r} metrics-before failed: {before_err}")
            run = run_case(case, srv, requests, key, ev, cdir)
            after_raw, after_parsed, after_err = fetch_metrics_snapshot(
                server_base[sid], key, srv["metrics_timeout_s"]
            )
            try:
                (cdir / "metrics-after.txt").write_text(after_raw, encoding="utf-8")
            except OSError as exc:
                raise HarnessError(f"cannot write case evidence for {cid!r}: {exc}")
            if after_err:
                eprint(f"warning: case {cid!r} metrics-after failed: {after_err}")
            if before_err or after_err:
                # Without both snapshots no delta claim is allowed.
                status = STATUS_FAIL
                reasons = [
                    "metrics_unavailable: "
                    + "; ".join(x for x in (before_err, after_err) if x)
                ]
                checks: dict[str, Any] = {"metrics": "unavailable"}
                deltas_out: dict[str, float] = {}
                missing_out: list[str] = []
                if "flashprefill_sparse_rows_total" not in (before_err or ""):
                    pass
                # honor explicit skip allowlist for snapshot failures
                skip_names = case["expect"]["skip_if_metrics_missing"]
                if skip_names and (before_err or after_err):
                    status = STATUS_SKIP
                    reasons = [f"skip: metrics snapshot failed but allowlisted: {reasons[0]}"]
            else:
                status, reasons, checks = evaluate_case_expectations(
                    case, before_parsed, after_parsed, run["transport_ok"],
                    run["texts"], peer_texts,
                )
                # Full labeled deltas for evidence (bounded labels included).
                deltas_out = {}
                for k in set(before_parsed) | set(after_parsed):
                    b = before_parsed.get(k)
                    a = after_parsed.get(k)
                    if b is None and a is not None:
                        deltas_out[k] = a  # new series; flagged in deltas.json
                    elif b is not None and a is None:
                        deltas_out[k] = -b
                    elif b is not None and a is not None:
                        d = a - b
                        if d != 0.0 or k.startswith(("flashprefill_", "rerot_", "llamacpp:")):
                            deltas_out[k] = d
                missing_out = sorted(
                    {m.split("{", 1)[0] for m in checks.get("missing_failed", [])}
                    | {m.split("{", 1)[0] for m in checks.get("skipped_missing", [])}
                )
            # record dense-reason + fallback evidence explicitly
            per_reason: dict[str, float] = {}
            if not (before_err or after_err):
                per_reason = dense_reason_deltas(before_parsed, after_parsed)
            try:
                (cdir / "deltas.json").write_text(
                    json.dumps(
                        {
                            "case": cid, "server": sid,
                            "dense_reason_deltas": per_reason,
                            "deltas": deltas_out if not (before_err or after_err) else {},
                            "checks": checks,
                        },
                        ensure_ascii=False, indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                raise HarnessError(f"cannot write deltas for {cid!r}: {exc}")
            # memory snapshot after case
            try:
                _, slots_body = http_get_json(
                    server_base[sid], "/slots", key, srv["metrics_timeout_s"]
                )
            except Exception:
                slots_body = None
            keep = {
                k: v for k, v in after_parsed.items()
                if k.startswith("flashprefill_") or k.startswith("rerot_")
            }
            ev.emit(
                ev.f_memory,
                {
                    "timestamp": now_iso(), "scope": "case-after", "server": sid,
                    "case": cid, "metrics_snapshot": dict(list(keep.items())[:256]),
                    "metrics_error": after_err, "slots": slots_body,
                },
            )
            peer_texts[cid] = run["texts"]
            case_results.append(
                {
                    "id": cid, "server": sid, "status": status, "reasons": reasons,
                    "checks": checks,
                    "metrics_delta": deltas_out if not (before_err or after_err) else {},
                    "missing_metrics": missing_out if not (before_err or after_err) else [],
                    "n_steps": len(case["steps"]),
                }
            )
            print(f"[matrix] case {cid!r}: {status}" + (f" ({'; '.join(reasons)})" if reasons else ""))

        # --- run-level metrics-after + comparisons --------------------------------
        for srv in matrix["servers"]:
            sid = srv["id"]
            if sid in server_failed:
                continue
            key = effective_api_key(srv, cli_key)
            raw, _, err = fetch_metrics_snapshot(server_base[sid], key, srv["metrics_timeout_s"])
            server_metrics_after_raw[sid] = raw
            try:
                (out / "servers" / sid / "metrics-after.txt").write_text(raw, encoding="utf-8")
                p_status, p_body = http_get_json(
                    server_base[sid], "/props", key, srv["metrics_timeout_s"]
                )
                (out / "servers" / sid / "props-after.json").write_text(
                    json.dumps({"http_status": p_status, "body": p_body}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                raise HarnessError(f"cannot write server after-evidence for {sid!r}: {exc}")
            if err:
                eprint(f"warning: server {sid!r} metrics-after failed: {err}")

        comparison_results: list[dict[str, Any]] = []
        results_by_id = {r["id"]: r for r in case_results}
        for cmp in matrix["comparisons"]:
            bid, cid_ = cmp["baseline_case"], cmp["candidate_case"]
            b_texts = peer_texts.get(bid)
            c_texts = peer_texts.get(cid_)
            reasons_c: list[str] = []
            ok = True
            if results_by_id[bid]["status"] != STATUS_PASS or results_by_id[cid_]["status"] != STATUS_PASS:
                ok = False
                reasons_c.append(
                    f"needs both cases PASS (baseline {results_by_id[bid]['status']}, "
                    f"candidate {results_by_id[cid_]['status']})"
                )
            if cmp["require_exact_match"] and b_texts is not None and c_texts is not None:
                if b_texts != c_texts:
                    ok = False
                    reasons_c.append(
                        f"exact outputs differ ({len(b_texts or [])} vs {len(c_texts or [])} "
                        "texts; see responses.jsonl)"
                    )
            if cmp["require_resource_evidence"]:
                # Both cases must own timings + metric snapshots, not just averages.
                for check_id in (bid, cid_):
                    cdir = out / "cases" / check_id
                    for fname in ("metrics-before.txt", "metrics-after.txt", "deltas.json"):
                        fp = cdir / fname
                        try:
                            if not fp.exists() or fp.stat().st_size == 0 and fname != "deltas.json":
                                ok = False
                                reasons_c.append(f"resource evidence missing: {check_id}/{fname}")
                        except OSError:
                            ok = False
                            reasons_c.append(f"resource evidence unreadable: {check_id}/{fname}")
            comparison_results.append(
                {
                    "id": cmp["id"], "baseline_case": bid, "candidate_case": cid_,
                    "status": STATUS_PASS if ok else STATUS_FAIL,
                    "reasons": reasons_c,
                    "require_exact_match": cmp["require_exact_match"],
                    "require_resource_evidence": cmp["require_resource_evidence"],
                }
            )
            print(f"[matrix] comparison {cmp['id']!r}: {comparison_results[-1]['status']}")

        any_fail = any(r["status"] == STATUS_FAIL for r in case_results) or any(
            r["status"] == STATUS_FAIL for r in comparison_results
        )
        overall = STATUS_FAIL if any_fail else STATUS_PASS
        summary = {
            "schema_version": SCHEMA_VERSION,
            "overall": overall,
            "created_at": now_iso(),
            "cases": case_results,
            "comparisons": comparison_results,
            "servers": [
                {
                    "id": s["id"],
                    "failed": s["id"] in server_failed,
                    "failure": server_failed.get(s["id"]),
                    "owned_exit_code": owned[s["id"]].poll_exit() if s["id"] in owned else None,
                }
                for s in matrix["servers"]
            ],
        }
        try:
            (out / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise HarnessError(f"cannot write summary: {exc}")

        # commands.txt + binary hashes + top-level convenience copies
        lines = [
            f"# flashprefill-matrix invocation ({now_iso()})",
            shell_quote(invocation),
            "",
            f"# matrix: {args.matrix} sha256={matrix_sha}",
            f"# requests: {args.requests} sha256={requests_sha}",
            "",
        ]
        for srv in matrix["servers"]:
            if srv["launch"] is not None:
                lines.append(f"# owned server {srv['id']} (port {srv['launch']['port']})")
                lines.append(shell_quote(srv["launch"]["argv"]))
            else:
                lines.append(f"# endpoint server {srv['id']}: {srv['endpoint']['base_url']}")
            lines.append("")
        try:
            (out / "commands.txt").write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            raise HarnessError(f"cannot write commands.txt: {exc}")

        hash_lines: list[str] = []
        for srv in matrix["servers"]:
            if srv["launch"] is not None:
                binary = Path(srv["launch"]["argv"][0])
                if binary.exists() and binary.is_file():
                    try:
                        hash_lines.append(f"{sha256_file(binary)}  {binary}")
                    except OSError as exc:
                        hash_lines.append(f"# {binary}: unreadable ({exc})")
                else:
                    hash_lines.append(
                        f"# server {srv['id']}: no local binary at {binary} (not hashed, not fabricated)"
                    )
            else:
                hash_lines.append(
                    f"# server {srv['id']}: endpoint {srv['endpoint']['base_url']} (no local binary)"
                )
        try:
            (out / "binary-libraries.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
        except OSError as exc:
            raise HarnessError(f"cannot write binary-libraries.sha256: {exc}")

        # Top-level 18.6 copies for single-server runs; index otherwise.
        try:
            if len(matrix["servers"]) == 1:
                only = matrix["servers"][0]["id"]
                sdir = out / "servers" / only
                for name in ("metrics-before.txt", "metrics-after.txt"):
                    src = sdir / name
                    if src.exists():
                        shutil.copyfile(src, out / name)
                if only in owned:
                    src_log = sdir / "server.log"
                    if src_log.exists():
                        shutil.copyfile(src_log, out / "server.log")
                else:
                    (out / "server.log").write_text(
                        f"# endpoint server {only}: no owned log\n", encoding="utf-8"
                    )
            else:
                (out / "metrics-before.txt").write_text(
                    "# multi-server run: per-server raw files under servers/<id>/metrics-before.txt\n"
                    + "".join(f"# - {s['id']}\n" for s in matrix["servers"]),
                    encoding="utf-8",
                )
                (out / "metrics-after.txt").write_text(
                    "# multi-server run: per-server raw files under servers/<id>/metrics-after.txt\n"
                    + "".join(f"# - {s['id']}\n" for s in matrix["servers"]),
                    encoding="utf-8",
                )
                (out / "server.log").write_text(
                    "# multi-server run: owned logs under servers/<id>/server.log\n",
                    encoding="utf-8",
                )
        except OSError as exc:
            raise HarnessError(f"cannot write top-level evidence copies: {exc}")

        manifest["completed_at"] = now_iso()
        manifest["summary"] = summary
        manifest["server_props_before"] = server_props
        try:
            (out / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise HarnessError(f"cannot finalize manifest: {exc}")

        overall_rc = 0 if overall != STATUS_FAIL else 2
        print(f"[matrix] overall: {overall} (results in {out})")
        return overall_rc
    except HarnessError as exc:
        eprint(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        eprint("interrupted; cleaning up owned servers")
        return 1
    finally:
        try:
            ev.close()
        except Exception:
            pass
        for sid, handle in owned.items():
            cfg = server_index.get(sid, {})
            launch = cfg.get("launch") if isinstance(cfg, dict) else None
            timeout = launch.get("shutdown_timeout_s", 10.0) if launch else 10.0
            try:
                handle.stop(timeout)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
