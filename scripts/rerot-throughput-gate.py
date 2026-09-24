#!/usr/bin/env python3
"""RERoT throughput gate: same-build useful-token comparison (RERoT OFF vs ON).

C02 fixes (攻坚总计划 §2.4 / §12):
  1. Per-sample judgment: every correctness check is evaluated for EVERY round;
     a single bad round fails the gate (no last-round masking of earlier aborts).
  2. Gauge/counter separation: gauges (rerot_pens_allocated, rerot_batch_pens, ...)
     are snapshotted and reported, but NEVER used for before/after delta checks.
     Multi-Lane evidence comes from the counters rerot_parallel_model_tokens /
     rerot_parallel_seconds (event evidence), and from
     rerot_parallel_peak_lanes_total (request-scoped peak lanes, counter).
  3. Missing metrics are reported as `missing` and yield evidence-incomplete
     (exit 3), never a fabricated 0.0.
  4. No parallel token/time evidence -> `parallel_work_exercised_paired` is
     NOT_EXERCISED and the verdict is evidence-incomplete, never silently
     substituted by the aggregate number.
  5. Route gate: if --enable-operator-route-gate is set but the route counter
     is absent from /metrics, the gate reports evidence-incomplete instead of
     inventing a 0 hit count.
  6. timings.prompt_ms is reported as prompt_ms, not ttft_ms. TTFT requires
     streaming sampling; see scripts/rerot-ttft-sample.py. The gate requires
     non-streaming requests and records ttft_ms as null.
  7. Comparison kinds are explicit: same-build scheduling strategy (default)
     vs implementation-version comparison (--baseline-base-url).
  8. Paired statistics: every (serial, rerot) pair is retained; the primary
     verdict uses each pair's sampled-token rate, not model tokens inflated
     by forced frames or side medians. Parallel-phase model-token rate is
     diagnostic only: its numerator differs from the serial sampled-token rate.
  9. Historical baseline responses are never part of the formal verdict
     unless --baseline-manifest matches the live server manifest; even then
     they are reported as historical_reference and excluded from the paired
     verdict (which requires same-run fresh pairs).
 10. Timeout help text: the client-side HTTP deadline does not cancel GPU
     work and is not a device-recovery mechanism.

Exit codes: 0 PASS, 1 operational error, 2 FAIL (checks), 3 evidence-incomplete.

Note on rerot_q_prep_rows accounting (攻坚总计划 §2.4): q_prep_rows is
accumulated in llama-graph.cpp graph construction as group_cap * heads, where
group_cap is the 32-aligned capacity bucket — NOT dispatched live rows. It is
not exported to /metrics and this gate deliberately does not use it. Any
future exporter must divide by live_groups only after accounting for graph
replay counts; the capacity-based value overstates processed rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_FAIL = 2
EXIT_EVIDENCE_INCOMPLETE = 3

CHECK_OK = "ok"
CHECK_FAIL = "fail"
CHECK_NOT_EXERCISED = "not_exercised"  # route not hit / no parallel phase
CHECK_EVIDENCE_INCOMPLETE = "evidence_incomplete"  # required metric missing

# TODO: Enable when P0 low-level operator route counters land in /metrics.
# Once implemented, verify operator route hit metrics (e.g. Vulkan RERoT kernel hits).
ENABLE_OPERATOR_ROUTE_GATE = False


def validate_dag_payload(payload: Any) -> tuple[bool, str]:
    """Validate that the request payload contains a DAG or multi-Lane structure.

    Returns (is_valid, human_readable_diagnostic_if_invalid).
    """
    if not isinstance(payload, dict):
        return False, "payload root is not a JSON object"

    # Format 1: Explicit routing strategy specification
    strategy = payload.get("strategy")
    if strategy == "dag":
        dag_body = payload.get("payload")
        if not isinstance(dag_body, dict):
            return False, "explicit 'strategy' is 'dag' but 'payload' object is missing or not a dict"
        questions = dag_body.get("questions")
        if not isinstance(questions, list) or len(questions) < 2:
            return False, f"explicit DAG 'questions' must be a list with at least 2 items, got {questions!r}"
        for idx, q in enumerate(questions):
            if not isinstance(q, dict) or not q.get("id"):
                return False, f"DAG question item #{idx} missing required 'id' field"
        return True, ""

    # Format 2: Explicit DAG/lanes field in request
    for key in ("dag", "rerot_dag"):
        if key in payload:
            dag_obj = payload[key]
            if not isinstance(dag_obj, dict):
                return False, f"'{key}' must be a JSON object"
            nodes = dag_obj.get("nodes") or dag_obj.get("questions") or dag_obj.get("lanes")
            if not isinstance(nodes, list) or len(nodes) < 2:
                return False, f"'{key}' must define at least 2 concurrent nodes/lanes in list, got {nodes!r}"
            return True, ""

    if "lanes" in payload:
        lanes = payload["lanes"]
        if not isinstance(lanes, list) or len(lanes) < 2:
            return False, f"'lanes' must be a list with at least 2 items, got {lanes!r}"
        return True, ""

    # Format 3: Chat completion request targeting RERoT
    # Must have messages and explicitly opt into RERoT (rerot=True or rerot_frontier)
    messages = payload.get("messages")
    if isinstance(messages, list):
        if not messages:
            return False, "'messages' list is empty"
        # RERoT must be enabled or intended in the request
        rerot_flag = payload.get("rerot")
        has_frontier = "rerot_frontier" in payload
        if rerot_flag is False and not has_frontier:
            return False, "request explicitly disables RERoT ('rerot': false) without 'rerot_frontier'"

        # Verify multi-task / DAG instruction content across user messages
        user_texts = [
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "system") and isinstance(m.get("content"), str)
        ]
        combined = "\n".join(user_texts)
        if not combined.strip():
            return False, "request messages contain no text content"

        # Check for multi-task / DAG indicators (e.g. numbered items 1. 2., DAG keywords, subtasks, etc.)
        has_dag_keyword = any(k in combined.lower() for k in ("dag", "lane", "subtask", "sub-task", "multi-task"))
        has_multi_items = (
            ("1." in combined and "2." in combined)
            or ("1、" in combined and "2、" in combined)
            or ("Question A" in combined and "Question B" in combined)
            or ("task" in combined.lower() and len(user_texts) > 1)
            or ("独立" in combined)
            or ("并行" in combined)
        )
        if not (has_dag_keyword or has_multi_items or len(messages) >= 3):
            return False, (
                "request payload lacks DAG/multi-Lane decomposition structure: prompt must contain "
                "multiple subtasks (e.g. '1.', '2.', 'Question A', 'Question B') or DAG/Lane keywords"
            )
        return True, ""

    return False, (
        "payload lacks recognized DAG/multi-Lane structure: neither explicit DAG specification "
        "('strategy'='dag', 'dag', 'lanes') nor multi-task chat messages found"
    )


# Counter metrics: monotonic accumulators; before/after delta across one request
# equals that request's contribution (exactly one episode per round is asserted).
COUNTER_METRICS: tuple[str, ...] = (
    "rerot_completed_episode_total",
    "rerot_completed_model_tokens",
    "rerot_parallel_model_tokens",
    "rerot_completed_episode_seconds",
    "rerot_parallel_seconds",
    "rerot_public_tokens",
    "rerot_private_tokens",
    "rerot_pending_tokens",
    "rerot_hard_aborts",
    "rerot_final_fences",
    "rerot_frontier_rows",
    "rerot_six_row_batches",
    # Request-scoped multi-Lane peak pens, summed over completed episodes.
    # Present when the server exports it (C02); absent on older builds ->
    # reported missing (evidence-incomplete) rather than 0.
    "rerot_parallel_peak_lanes_total",
)

# Gauge metrics: instantaneous state; reported as snapshots, NEVER delta-checked.
GAUGE_METRICS: tuple[str, ...] = (
    "rerot_people_capacity",
    "rerot_people_resident",
    "rerot_pens_capacity",
    "rerot_pens_allocated",
    "rerot_pens_running",
    "rerot_batch_people",
    "rerot_batch_pens",
    "rerot_brain_bytes",
    "rerot_hand_bytes",
)

# Counters required for a meaningful verdict. Absence -> evidence-incomplete.
REQUIRED_COUNTERS: tuple[str, ...] = (
    "rerot_completed_episode_total",
    "rerot_completed_model_tokens",
    "rerot_parallel_model_tokens",
    "rerot_completed_episode_seconds",
    "rerot_parallel_seconds",
    "rerot_public_tokens",
    "rerot_private_tokens",
    "rerot_pending_tokens",
    "rerot_hard_aborts",
    "rerot_final_fences",
)

# Optional: when the route gate is enabled, this counter must be registered by
# the server (P0 operator route counters). Absent -> evidence-incomplete, never 0.
ROUTE_HIT_METRIC = "rerot_operator_route_hits"


def request_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> tuple[float, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    return time.monotonic() - started, json.loads(body)


def verify_server_health(base_url: str, api_key: str, timeout: float) -> None:
    """Pre-flight health check to verify server availability before issuing load.

    Fails closed with a human-readable error if /health is unreachable or non-200.
    """
    health_url = f"{base_url.rstrip('/')}/health"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(health_url, headers=headers)
    check_timeout = min(timeout, 10.0) if timeout > 0 else 10.0
    try:
        with urllib.request.urlopen(req, timeout=check_timeout) as response:
            status = getattr(response, "status", response.getcode())
            body = response.read().decode("utf-8", errors="replace")
            if status != 200:
                raise RuntimeError(
                    f"server health check failed: non-200 status {status} from {health_url}: {body.strip()}"
                )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"server health check failed: HTTP {error.code} from {health_url}: {body.strip()}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"server health check failed: unreachable {health_url}: {error.reason}"
        ) from error


def fetch_metrics(base_url: str, api_key: str, timeout: float) -> dict[str, float]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/metrics",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")

    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:") or "{" in line:
            continue
        name, raw_value = line.split(None, 1)
        values[name.removeprefix("llamacpp:")] = float(raw_value)
    return values


def counter_delta(before: dict[str, float], after: dict[str, float], name: str) -> tuple[float, bool]:
    """Delta for a counter metric.

    Returns (value, present). present=False means the counter is not registered
    in /metrics on either scrape; the value is then 0.0 and MUST NOT be used as
    evidence (callers report evidence-incomplete instead).
    """
    if name not in after and name not in before:
        return 0.0, False
    return after.get(name, 0.0) - before.get(name, 0.0), True


def fetch_props(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/props",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_build_number(build_info: Any) -> int | None:
    """Extract the build number from the /props build_info field.

    build_info is typically the llama_print_build_info() text containing a
    line like 'build: 4665 (ffffffff)'. Returns None when unparseable.
    """
    if isinstance(build_info, int):
        return build_info
    if isinstance(build_info, str):
        for part in build_info.replace("(", " ").replace(")", " ").split():
            if part.isdigit() and len(part) >= 4:
                return int(part)
    return None


def manifest_from_props(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "build_number": parse_build_number(props.get("build_info")),
        "model_alias": props.get("model_alias"),
        "model_ftype": props.get("model_ftype"),
        "model_sha256": props.get("model_sha256"),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to load manifest {path}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"manifest {path} must be a JSON object")
    return data


def manifests_match(server: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    # build_number and model_sha256 are the hard identity keys. model_alias and
    # model_ftype are soft: a missing side is reported, not rejected.
    for key in ("build_number", "model_sha256"):
        s = server.get(key)
        b = baseline.get(key)
        if s is None or b is None:
            mismatches.append(f"'{key}' missing on one side (server={s!r}, baseline={b!r})")
        elif s != b:
            mismatches.append(f"'{key}' differs (server={s!r}, baseline={b!r})")
    for key in ("model_alias", "model_ftype"):
        if server.get(key) != baseline.get(key):
            mismatches.append(f"'{key}' differs (server={server.get(key)!r}, baseline={baseline.get(key)!r})")
    return not mismatches, mismatches


def serial_throughput(response: dict[str, Any]) -> float:
    timings = response.get("timings", {})
    tokens = float(timings.get("predicted_n", 0))
    seconds = float(timings.get("predicted_ms", 0)) / 1000.0
    if tokens <= 0 or seconds <= 0:
        raise RuntimeError("serial response lacks positive timings.predicted_n/predicted_ms")
    return tokens / seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Same-build scheduling-strategy gate: run the same deterministic request with "
            "RERoT OFF/ON on ONE server and fail unless the RERoT path matches the single-Lane "
            "path on paired throughput with every round semantically valid. "
            "Implementation-version comparison (two servers) is a separate mode: "
            "--baseline-base-url."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""))
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument(
        "--baseline-response",
        type=Path,
        help=(
            "Reuse an existing RERoT-OFF response instead of issuing the OFF request for "
            "round 0. Pairs built on it are reported as historical_reference and are EXCLUDED "
            "from the formal paired verdict; use --baseline-manifest to validate that the "
            "response came from the same model/build."
        ),
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        help=(
            "JSON manifest of the build/model that produced --baseline-response "
            "(build_number, model_sha256, model_alias, model_ftype). Compared against the "
            "live server /props; on mismatch the baseline is not used at all."
        ),
    )
    parser.add_argument(
        "--baseline-base-url",
        default=None,
        help=(
            "Implementation-version comparison mode: issue serial (RERoT-OFF) requests against "
            "this server and RERoT-ON requests against --base-url. Without it, both sides run "
            "on --base-url (same-build scheduling-strategy comparison)."
        ),
    )
    parser.add_argument(
        "--baseline-api-key",
        default=None,
        help="API key for --baseline-base-url (defaults to --api-key).",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help=(
            "Client-side HTTP request deadline in seconds (default: 180.0). "
            "This timeout does NOT cancel in-flight GPU work on the server and is not a "
            "device-recovery or watchdog mechanism; server-side drain/safety is managed by "
            "the server itself."
        ),
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.0,
        help=(
            "Required lower bound for every paired useful-token throughput ratio. "
            "1.0 means the active pens must at least match simple; 1.02 requires 2%% gain."
        ),
    )
    parser.add_argument(
        "--min-peak-lanes", type=int, default=2,
        help="Require this many physical lanes to be active during the RERoT request.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of paired measurement rounds. When rounds > 1, executes in A/B/B/A pattern (§17).",
    )
    # TODO: Enable operator route gate once P0 low-level operator route counters land in /metrics.
    parser.add_argument(
        "--enable-operator-route-gate",
        action="store_true",
        default=ENABLE_OPERATOR_ROUTE_GATE,
        help=(
            "TODO: Assert low-level operator route hit metrics once P0 counters land; "
            "disabled by default. If enabled while the server does not export "
            "'rerot_operator_route_hits', the gate reports evidence-incomplete instead of "
            "fabricating a zero hit count."
        ),
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or LLAMA_API_KEY is required")
    if args.min_ratio <= 0:
        parser.error("--min-ratio must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be >= 1")
    if args.min_peak_lanes < 2:
        parser.error("--min-peak-lanes must be >= 2")
    if args.baseline_manifest and not args.baseline_response:
        parser.error("--baseline-manifest requires --baseline-response")
    if args.baseline_base_url is not None and args.baseline_response is not None:
        parser.error("--baseline-response cannot be combined with --baseline-base-url")
    return args


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p95_idx = min(n - 1, int(0.95 * n))
    return {
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "median": median,
        "p95": s[p95_idx],
    }


def extract_rerot_token_breakdown(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage", {})
    rerot_meta = usage.get("rerot", {})
    timings = response.get("timings", {})

    # C02 #6: timings.prompt_ms is the prompt COMPUTE time, not time-to-first-
    # token. TTFT needs streaming sampling (scripts/rerot-ttft-sample.py); the
    # gate requires non-streaming requests and cannot observe it.
    prompt_ms = timings.get("prompt_ms", 0.0)

    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "probe_tokens": rerot_meta.get("probe_tokens", 0),
        "frame_tokens": rerot_meta.get("frame_tokens", 0),
        "actual_replayed_tokens": rerot_meta.get("actual_replayed_tokens", 0),
        "sampled_tokens": rerot_meta.get("sampled_tokens", 0),
        "useful_body_tokens": rerot_meta.get("sampled_tokens", 0),
        "prompt_ms": prompt_ms,
        "ttft_ms": None,  # not observable with non-streaming requests
    }


def evaluate_round_checks(
    d: dict[str, float],
    present: dict[str, bool],
    payload: Any,
    is_dag_payload: bool,
    dag_structure_err: str,
    route_gate_enabled: bool,
    min_peak_lanes: int,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Per-round correctness checks.

    Returns (checks, details, missing). Checks use only counter deltas whose
    counters were present in the scrape; any required-metric absence yields
    CHECK_EVIDENCE_INCOMPLETE for the dependent checks and a missing entry.
    """
    checks: dict[str, str] = {}
    details: dict[str, str] = {}
    missing: list[str] = []

    def counter_evidence(name: str) -> tuple[float, bool]:
        if not present.get(name, False):
            missing.append(name)
            return 0.0, False
        return d.get(name, 0.0), True

    visible_tokens = d.get("rerot_public_tokens", 0.0) + d.get("rerot_private_tokens", 0.0) + d.get("rerot_pending_tokens", 0.0)

    # ---- Per-sample episode accounting (counters) ----
    ep, ep_ok = counter_evidence("rerot_completed_episode_total")
    if not ep_ok:
        checks["one_completed_episode"] = CHECK_EVIDENCE_INCOMPLETE
        details["one_completed_episode"] = f"required counter rerot_completed_episode_total missing from /metrics"
    else:
        checks["one_completed_episode"] = CHECK_OK if ep == 1 else CHECK_FAIL
        details["one_completed_episode"] = f"rerot_completed_episode_total delta must equal 1, observed {ep:g}"

    aborts, aborts_ok = counter_evidence("rerot_hard_aborts")
    if not aborts_ok:
        checks["no_hard_abort"] = CHECK_EVIDENCE_INCOMPLETE
        details["no_hard_abort"] = "required counter rerot_hard_aborts missing from /metrics"
    else:
        checks["no_hard_abort"] = CHECK_OK if aborts == 0 else CHECK_FAIL
        details["no_hard_abort"] = f"rerot_hard_aborts delta must equal 0, observed {aborts:g}"

    fences, fences_ok = counter_evidence("rerot_final_fences")
    if not fences_ok:
        checks["one_final_fence"] = CHECK_EVIDENCE_INCOMPLETE
        details["one_final_fence"] = "required counter rerot_final_fences missing from /metrics"
    else:
        checks["one_final_fence"] = CHECK_OK if fences == 1 else CHECK_FAIL
        details["one_final_fence"] = f"rerot_final_fences delta must equal 1, observed {fences:g}"

    m_tok, m_tok_ok = counter_evidence("rerot_completed_model_tokens")
    vis_pub, vis_pub_ok = counter_evidence("rerot_public_tokens")
    vis_priv, vis_priv_ok = counter_evidence("rerot_private_tokens")
    vis_pend, vis_pend_ok = counter_evidence("rerot_pending_tokens")
    if m_tok_ok and vis_pub_ok and vis_priv_ok and vis_pend_ok:
        visible = vis_pub + vis_priv + vis_pend
        checks["visibility_accounting_exact"] = CHECK_OK if visible == m_tok else CHECK_FAIL
        details["visibility_accounting_exact"] = (
            f"visible tokens ({visible:g}) must equal completed model tokens ({m_tok:g})"
        )
    else:
        checks["visibility_accounting_exact"] = CHECK_EVIDENCE_INCOMPLETE
        details["visibility_accounting_exact"] = (
            f"visibility accounting requires rerot_{'public,private,pending' if not vis_pub_ok else 'completed_model'}"
            f"_tokens (missing={[n for n in missing if 'tokens' in n]})"
        )

    # ---- Multi-Lane evidence (counters only; gauges never gate) ----
    par_tok, par_tok_ok = counter_evidence("rerot_parallel_model_tokens")
    par_sec, par_sec_ok = counter_evidence("rerot_parallel_seconds")
    peak_pens, peak_pens_ok = counter_evidence("rerot_parallel_peak_lanes_total")
    if not par_tok_ok or not par_sec_ok:
        checks["multi_lane_parallel_work"] = CHECK_EVIDENCE_INCOMPLETE
        details["multi_lane_parallel_work"] = (
            "required counters rerot_parallel_model_tokens/rerot_parallel_seconds missing from /metrics"
        )
    elif par_tok <= 0 or par_sec <= 0:
        checks["multi_lane_parallel_work"] = CHECK_NOT_EXERCISED
        details["multi_lane_parallel_work"] = (
            "multi-Lane phase produced no parallel model tokens/seconds this round: "
            "the request did not exercise parallel execution"
        )
    else:
        checks["multi_lane_parallel_work"] = CHECK_OK
        details["multi_lane_parallel_work"] = (
            f"parallel counters positive (tokens={par_tok:g}, seconds={par_sec:g})"
        )
    if peak_pens_ok:
        checks["multi_lane_peak_lanes"] = CHECK_OK if peak_pens >= min_peak_lanes else CHECK_FAIL
        details["multi_lane_peak_lanes"] = (
            f"request-scoped peak lanes (counter delta) must be >= {min_peak_lanes}, observed {peak_pens:g}"
        )
    else:
        # Older servers do not export this counter. A six-lane claim needs
        # positive evidence; the generic two-lane gate keeps its old policy.
        checks["multi_lane_peak_lanes"] = (
            CHECK_EVIDENCE_INCOMPLETE if min_peak_lanes > 2 else CHECK_NOT_EXERCISED
        )
        details["multi_lane_peak_lanes"] = (
            "counter rerot_parallel_peak_lanes_total not exported by this server build; "
            "multi-Lane evidence relies on parallel token counters only"
        )
    if min_peak_lanes >= 6:
        six_batches, six_ok = counter_evidence("rerot_six_row_batches")
        checks["six_pens_in_one_decode"] = (
            CHECK_OK if six_ok and six_batches > 0 else
            CHECK_FAIL if six_ok else CHECK_EVIDENCE_INCOMPLETE
        )
        details["six_pens_in_one_decode"] = (
            f"at least one successful decode slice must contain six pen rows; observed {six_batches:g}"
            if six_ok else "rerot_six_row_batches counter is missing"
        )

    # ---- Payload structure (identical every round; kept per-round for uniform reporting) ----
    checks["request_dag_structure"] = CHECK_OK if is_dag_payload else CHECK_FAIL
    details["request_dag_structure"] = (
        "payload verified as DAG/multi-Lane structure"
        if is_dag_payload
        else f"missing required DAG/multi-Lane structure in request: {dag_structure_err}"
    )

    # ---- Operator route gate (C02 #5) ----
    if route_gate_enabled:
        hits, hits_ok = counter_evidence(ROUTE_HIT_METRIC)
        if not hits_ok:
            checks["operator_route_hit"] = CHECK_EVIDENCE_INCOMPLETE
            details["operator_route_hit"] = (
                f"route gate enabled but counter {ROUTE_HIT_METRIC} is not registered in /metrics "
                "(P0 route counters not landed); no fabricated hit count"
            )
        else:
            checks["operator_route_hit"] = CHECK_OK if hits > 0 else CHECK_FAIL
            details["operator_route_hit"] = f"operator route hits must be > 0, observed {hits:g}"
    else:
        checks["operator_route_hit"] = CHECK_NOT_EXERCISED
        details["operator_route_hit"] = "route gate disabled (default); enable with --enable-operator-route-gate"

    return checks, details, missing


def main() -> int:
    args = parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    if payload.get("stream") is True:
        raise RuntimeError("throughput gate requires a non-streaming request")

    # C02 #7: explicit comparison kinds.
    if args.baseline_base_url:
        comparison_kind = "implementation_version"
    else:
        comparison_kind = "same_build_scheduling_strategy"

    serial_base_url = args.baseline_base_url or args.base_url
    serial_api_key = args.baseline_api_key or args.api_key
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    serial_endpoint = f"{serial_base_url.rstrip('/')}/v1/chat/completions"

    serial_payload = dict(payload)
    serial_payload.update({"stream": False, "rerot": False, "rerot_trace": False})
    # An explicit frontier selects RERoT even when rerot=false.
    serial_payload.pop("rerot_frontier", None)
    rerot_payload = dict(payload)
    rerot_payload.update({"stream": False, "rerot": True, "rerot_trace": False})

    # Pre-flight health check before issuing any benchmark load (fail-closed if server is down)
    verify_server_health(args.base_url, args.api_key, args.timeout)
    if args.baseline_base_url:
        verify_server_health(args.baseline_base_url, serial_api_key, args.timeout)

    # C02 #9: historical baseline validation against the live server manifest.
    historical_baseline: dict[str, Any] | None = None
    baseline_match_ok = True
    baseline_mismatch_reasons: list[str] = []
    if args.baseline_response:
        server_props = fetch_props(args.base_url, args.api_key, args.timeout)
        server_manifest = manifest_from_props(server_props)
        if args.baseline_manifest:
            baseline_manifest = load_manifest(args.baseline_manifest)
            baseline_match_ok, baseline_mismatch_reasons = manifests_match(server_manifest, baseline_manifest)
        else:
            baseline_mismatch_reasons.append(
                "--baseline-manifest not provided; manifest identity unverifiable"
            )
        if not baseline_match_ok:
            raise RuntimeError(
                "historical baseline rejected: server/baseline manifest mismatch: "
                + "; ".join(baseline_mismatch_reasons)
                + ". Re-measure the serial side instead of reusing this response."
            )
        historical_baseline = json.loads(args.baseline_response.read_text(encoding="utf-8"))
        if not isinstance(historical_baseline, dict):
            raise RuntimeError("--baseline-response must contain a JSON object")

    # §17 A/B/B/A paired measurement pattern
    # When rounds > 1, interleave execution order to neutralize warm-cache drift
    serial_samples: list[dict[str, Any]] = []
    rerot_samples: list[dict[str, Any]] = []
    serial_tps_list: list[float] = []
    aggregate_tps_list: list[float] = []
    parallel_tps_list: list[float] = []
    pairs: list[dict[str, Any]] = []
    round_checks: list[dict[str, Any]] = []
    all_missing: set[str] = set()
    any_round_failed = False
    any_evidence_incomplete = False

    is_dag_payload, dag_structure_err = validate_dag_payload(payload)

    total_pairs = args.rounds
    for round_idx in range(total_pairs):
        # A/B/B/A ordering pattern for round parity
        run_serial_first = (round_idx % 2 == 0)

        def do_serial() -> tuple[float | None, dict[str, Any], float]:
            if historical_baseline is not None and round_idx == 0:
                s_resp = historical_baseline
                s_wall = None
            else:
                s_wall, s_resp = request_json(serial_endpoint, serial_payload, serial_api_key, args.timeout)
            s_tps = serial_throughput(s_resp)
            return s_wall, s_resp, s_tps

        def do_rerot() -> tuple[float, dict[str, Any], dict[str, float], dict[str, bool], dict[str, float], dict[str, float], float, float]:
            before_m = fetch_metrics(args.base_url, args.api_key, args.timeout)
            r_wall, r_resp = request_json(endpoint, rerot_payload, args.api_key, args.timeout)
            after_m = fetch_metrics(args.base_url, args.api_key, args.timeout)
            d: dict[str, float] = {}
            present: dict[str, bool] = {}
            for name in COUNTER_METRICS:
                value, ok = counter_delta(before_m, after_m, name)
                d[name] = value
                present[name] = ok
            gauge_before = {name: before_m.get(name) for name in GAUGE_METRICS if name in before_m}
            gauge_after = {name: after_m.get(name) for name in GAUGE_METRICS if name in after_m}

            ep_sec = d["rerot_completed_episode_seconds"]
            m_tok = d["rerot_completed_model_tokens"]
            par_sec = d["rerot_parallel_seconds"]
            par_tok = d["rerot_parallel_model_tokens"]
            if not present["rerot_completed_episode_seconds"] or not present["rerot_completed_model_tokens"]:
                raise RuntimeError(
                    "completed RERoT episode did not publish positive token/time counters "
                    "(counters missing from /metrics; cannot compute throughput)"
                )
            if ep_sec <= 0 or m_tok <= 0:
                raise RuntimeError("completed RERoT episode did not publish positive token/time counters")
            sampled = r_resp.get("usage", {}).get("rerot", {}).get("sampled_tokens")
            predicted = r_resp.get("timings", {}).get("predicted_n")
            if sampled is None or predicted is None or sampled != predicted:
                raise RuntimeError("RERoT sampled-token count disagrees with timings.predicted_n")
            # The six-pen target concerns useful sampled tokens, not forced
            # frame tokens. Frame/model work still consumes the elapsed time.
            agg_tps = serial_throughput(r_resp)
            # C02 #4: no silent fallback to aggregate when the parallel phase
            # was not exercised. parallel_exercised drives the verdict state.
            if not present["rerot_parallel_seconds"] or not present["rerot_parallel_model_tokens"]:
                parallel_exercised = False
                par_tps = float("nan")
            elif par_sec <= 0 or par_tok <= 0:
                parallel_exercised = False
                par_tps = float("nan")
            else:
                parallel_exercised = True
                par_tps = par_tok / par_sec
            return r_wall, r_resp, d, present, gauge_before, gauge_after, agg_tps, (par_tps if parallel_exercised else float("nan"))

        if run_serial_first:
            sw, sr, stps = do_serial()
            rw, rr, d, present, gb, ga, atps, ptps = do_rerot()
        else:
            rw, rr, d, present, gb, ga, atps, ptps = do_rerot()
            sw, sr, stps = do_serial()

        serial_samples.append({
            "round": round_idx,
            "client_wall_seconds": sw,
            "tokens_per_second": stps,
            "answer": sr.get("choices", [{}])[0].get("message", {}).get("content"),
        })
        serial_tps_list.append(stps)
        parallel_exercised = ptps == ptps  # NaN check
        rerot_samples.append({
            "round": round_idx,
            "client_wall_seconds": rw,
            "answer": rr.get("choices", [{}])[0].get("message", {}).get("content"),
            "aggregate_tokens_per_second": atps,
            "model_tokens_per_second": d["rerot_completed_model_tokens"] / d["rerot_completed_episode_seconds"],
            "parallel_tokens_per_second": ptps if parallel_exercised else None,
            "parallel_exercised": parallel_exercised,
            "token_breakdown": extract_rerot_token_breakdown(rr),
            "metrics_delta": d,
            "metrics_present": present,
            "gauge_before": gb,
            "gauge_after": ga,
        })
        aggregate_tps_list.append(atps)
        if parallel_exercised:
            parallel_tps_list.append(ptps)

        checks, details, missing = evaluate_round_checks(
            d, present, payload, is_dag_payload, dag_structure_err,
            getattr(args, "enable_operator_route_gate", False),
            args.min_peak_lanes,
        )
        all_missing.update(missing)
        round_status = "ok"
        for name, state in checks.items():
            if state == CHECK_FAIL:
                round_status = "fail"
                any_round_failed = True
            elif state == CHECK_EVIDENCE_INCOMPLETE:
                if round_status == "ok":
                    round_status = "evidence_incomplete"
                any_evidence_incomplete = True
        round_checks.append({
            "round": round_idx,
            "status": round_status,
            "checks": checks,
            "check_details": details,
            "missing_metrics": sorted(missing),
        })

        is_historical = historical_baseline is not None and round_idx == 0
        pairs.append({
            "round": round_idx,
            "historical": is_historical,
            "serial_tokens_per_second": stps,
            "aggregate_tokens_per_second": atps,
            "parallel_tokens_per_second": ptps if parallel_exercised else None,
            "parallel_exercised": parallel_exercised,
            "aggregate_ratio": atps / stps,
            "parallel_ratio": (ptps / stps) if parallel_exercised else None,
            "serial_wall_seconds": sw,
            "rerot_wall_seconds": rw,
        })

    # ---- Sampled aggregate statistics (verdict is paired). Model-token
    # throughput includes forced frames and remains diagnostic only. ----
    serial_stats = calculate_percentiles(serial_tps_list)
    aggregate_stats = calculate_percentiles(aggregate_tps_list)
    parallel_stats = calculate_percentiles(parallel_tps_list)

    # ---- Paired statistics (C02 #8) ----
    formal_pairs = [p for p in pairs if not p["historical"]]
    historical_pairs = [p for p in pairs if p["historical"]]
    paired_aggregate_ratios = [p["aggregate_ratio"] for p in formal_pairs]
    paired_parallel_ratios = [p["parallel_ratio"] for p in formal_pairs if p["parallel_ratio"] is not None]

    if paired_aggregate_ratios:
        ratio_stats_agg = calculate_percentiles(paired_aggregate_ratios)
        paired_aggregate_pass = ratio_stats_agg["min"] >= args.min_ratio
        paired_aggregate_state = CHECK_OK if paired_aggregate_pass else CHECK_FAIL
    else:
        ratio_stats_agg = {}
        paired_aggregate_pass = False
        paired_aggregate_state = CHECK_EVIDENCE_INCOMPLETE

    ratio_stats_par = calculate_percentiles(paired_parallel_ratios) if paired_parallel_ratios else {}
    paired_parallel_state = (
        CHECK_EVIDENCE_INCOMPLETE if not formal_pairs else
        CHECK_OK if len(paired_parallel_ratios) == len(formal_pairs) else
        CHECK_NOT_EXERCISED
    )

    verdict_checks = {
        "per_round_all_ok": CHECK_OK if (not any_round_failed and not any_evidence_incomplete) else
        (CHECK_EVIDENCE_INCOMPLETE if not any_round_failed else CHECK_FAIL),
        "aggregate_no_slower_than_serial_paired": paired_aggregate_state,
        "parallel_work_exercised_paired": paired_parallel_state,
        "request_dag_structure": CHECK_OK if is_dag_payload else CHECK_FAIL,
    }

    # Overall exit code precedence: operational error already raised (1);
    # hard failure beats evidence-incomplete (2 > 3).
    if any_round_failed or paired_aggregate_state == CHECK_FAIL:
        passed = False
        status = "fail"
        exit_code = EXIT_FAIL
    elif (any_evidence_incomplete or paired_aggregate_state == CHECK_EVIDENCE_INCOMPLETE or
          paired_parallel_state in (CHECK_EVIDENCE_INCOMPLETE, CHECK_NOT_EXERCISED)):
        passed = False
        status = "evidence_incomplete"
        exit_code = EXIT_EVIDENCE_INCOMPLETE
    else:
        passed = True
        status = "pass"
        exit_code = EXIT_PASS

    result = {
        "schema_version": 3,
        "status": status,
        "passed": passed,
        "comparison": {
            "kind": comparison_kind,
            "base_url": args.base_url,
            "baseline_base_url": args.baseline_base_url,
            "historical_baseline_used": historical_baseline is not None,
            "historical_baseline_manifest_match": baseline_match_ok,
        },
        "rounds": args.rounds,
        "request": rerot_payload,
        "minimum_ratio": args.min_ratio,
        "statistics": {
            "serial_tps": serial_stats,
            "aggregate_tps": aggregate_stats,
            "parallel_tps": parallel_stats,
            "paired_aggregate_ratio": ratio_stats_agg,
            "paired_parallel_ratio": ratio_stats_par,
        },
        "serial": {
            "tokens_per_second": serial_stats.get("median"),
            "samples": serial_samples,
        },
        "rerot": {
            "aggregate_tokens_per_second": aggregate_stats.get("median"),
            "parallel_tokens_per_second": parallel_stats.get("median"),
            "samples": rerot_samples,
        },
        "pairs": pairs,
        "historical_reference": {
            "pairs": historical_pairs,
            "note": (
                "pairs built on --baseline-response are reported here and are excluded "
                "from the formal paired verdict"
            ),
        },
        "round_checks": round_checks,
        "verdict_checks": verdict_checks,
        "missing_metrics": sorted(all_missing),
        "not_exercised": (
            "parallel work not exercised in every formal round (no multi-Lane "
            "token/time evidence); useful-rate verdict is incomplete"
            if paired_parallel_state == CHECK_NOT_EXERCISED
            else None
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"rounds={args.rounds} "
        f"serial_median={serial_stats.get('median', float('nan')):.3f} tok/s "
        f"rerot_aggregate_median={aggregate_stats.get('median', float('nan')):.3f} tok/s "
        f"rerot_parallel_median={parallel_stats.get('median', float('nan')):.3f} tok/s "
        f"paired_aggregate_ratio_median={ratio_stats_agg.get('median', float('nan')):.3f} "
        f"status={status}"
    )
    if not passed:
        for round_rec in round_checks:
            if round_rec["status"] == "fail":
                for name, state in round_rec["checks"].items():
                    if state == CHECK_FAIL:
                        print(
                            f"FAIL round={round_rec['round']}: {name} "
                            f"({round_rec['check_details'].get(name, 'no detail')})",
                            file=sys.stderr,
                        )
        for name, state in verdict_checks.items():
            if state in (CHECK_FAIL, CHECK_EVIDENCE_INCOMPLETE, CHECK_NOT_EXERCISED):
                print(f"{state.upper()}: {name}", file=sys.stderr)
        if all_missing:
            print(f"MISSING metrics (evidence-incomplete): {sorted(all_missing)}", file=sys.stderr)
        if paired_parallel_state == CHECK_NOT_EXERCISED:
            print("NOT_EXERCISED: parallel work in one or more formal rounds", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR)
