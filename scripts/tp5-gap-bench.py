#!/usr/bin/env python3
"""Single-request, RERoT-off TP5 experiment. Preserve raw output and owned teardown.

No speed/semantic qualification is inferred by this runner. Read unique outputs
and use independent paired process blocks before promoting a configuration.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request

CANONICAL = "请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。"
GUARD = "用三句话解释哈希表与数组的主要区别。"


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def model_stat_identity(path):
    match = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", path.name)
    files = sorted(path.parent.glob(match.group(1) + "-?????-of-?????.gguf")) if match else [path]
    if match and len(files) != int(match.group(2)):
        raise RuntimeError("Incomplete model shard set")
    return {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                     "inode": p.stat().st_ino} for p in files}


def gpu_state():
    cards = []
    for p in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        if "-" in p.parent.name:
            continue
        try:
            if (p / "vendor").read_text().strip() != "0x1002":
                continue
            item = {"card": p.parent.name, "bdf": p.resolve().name}
            for key in ("device", "gpu_busy_percent", "mem_info_vram_used", "mem_info_vram_total",
                        "current_link_speed", "current_link_width", "pp_dpm_sclk", "pp_dpm_mclk"):
                q = p / key
                item[key] = q.read_text().strip() if q.exists() else None
            cards.append(item)
        except OSError as e:
            raise RuntimeError(f"Cannot read GPU state: {p}: {e}") from e
    return cards


def idle_gate():
    for _ in range(20):
        cards = gpu_state()
        if len(cards) != 5 or any(x["device"] != "0x73bf" for x in cards):
            raise RuntimeError(f"Need five physical Navi21 GPUs: {cards}")
        if all(int(x["gpu_busy_percent"]) == 0 and int(x["mem_info_vram_used"]) < 50 * 1024**2 for x in cards):
            return cards
        time.sleep(1)
    raise RuntimeError("GPUs not idle; no reset or recovery is attempted")


def build_gate():
    """Reject observed compilation at request boundaries, not just at startup."""
    for name in ("ninja", "cc1plus", "cc1", "glslc", "clang", "clang++"):
        if subprocess.run(["pgrep", "-x", name], stdout=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError(f"Build process active: {name}")


def api(port, path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def library_identity(pid):
    paths = set()
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        p = Path(fields[-1])
        if p.is_file() and ("libggml" in p.name or "libllama" in p.name or "libvulkan" in p.name):
            paths.add(p)
    return {str(p): sha(p) for p in sorted(paths)}


def request_payload(prompt, system_prompt=None, max_tokens=400):
    """Build a frozen request without changing the historical default workload."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ValueError("System prompt must not be empty when supplied")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return {"messages": messages, "temperature": 0, "seed": 42,
            "max_tokens": max_tokens, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}


def evaluate_length_gate(requests, length_min, length_max):
    """Pre-registered length gate over measured-phase public completion tokens.

    Warmup and guard requests stay recorded by the runner but are never
    counted here: only the measured phase is judged. The gate is evidence,
    not a promotion; status remains MEASURED_NOT_QUALIFIED either way and
    raw outputs are always retained.
    """
    measured = [r for r in requests if r.get("phase") == "measured"]
    per_request = [{"idx": r["idx"], "n": r["n"],
                    "in_range": length_min <= r["n"] <= length_max} for r in measured]
    return {"length_min": length_min, "length_max": length_max, "phase_scope": "measured",
            "passed": bool(measured) and all(x["in_range"] for x in per_request),
            "requests": per_request}


def _parse_kv_fields(line):
    """Shared key=value scanner for retained profile log lines.

    Integer-looking values become int, other numerics float, everything else
    keeps its raw token; nothing is coerced or dropped.
    """
    fields = {}
    for m in re.finditer(r"(\w+)=(\S+)", line):
        raw = m.group(2)
        try:
            fields[m.group(1)] = int(raw) if raw.isdigit() else float(raw)
        except ValueError:
            fields[m.group(1)] = raw
    return fields


def _cycle_ms(fields, key):
    value = fields.get(key)
    return value / 1000.0 if isinstance(value, (int, float)) else None


def extract_cycle_ledger(requests, horizon):
    """Best-effort MTP cycle ledger (AGENTS.md §7.2) from retained evidence.

    horizon=0 writes NOT_APPLICABLE. With MTP enabled, per-cycle
    [tp5-mtp-cycle] lines retained on each request item provide D/V/S/H/K,
    and timings counters (draft_n_accepted) are a fallback source for the
    accepted-prefix distribution. Anything no maintained source exposes is
    reported as the string NOT_COLLECTED; numbers are never fabricated.
    """
    if horizon == 0:
        return {"accepted_prefix_distribution": "NOT_APPLICABLE",
                "cycle_D_V_S_H_K": "NOT_APPLICABLE"}
    measured = [r for r in requests if r.get("phase") == "measured"]
    per_cycle = []
    for r in measured:
        for line in r.get("mtp_cycle_lines", []):
            fields = _parse_kv_fields(line)
            per_cycle.append({"D_ms": _cycle_ms(fields, "draft_us"),
                              "V_ms": _cycle_ms(fields, "target_us"),
                              "S_ms": _cycle_ms(fields, "catchup_us"),
                              "H_ms": _cycle_ms(fields, "handoff_us"),
                              "K": fields.get("accepted_tokens"),
                              "final_tokens": fields.get("final_tokens")})
    if per_cycle:
        k_list = [c["K"] for c in per_cycle if c["K"] is not None]
        histogram = {}
        for k in k_list:
            histogram[str(k)] = histogram.get(str(k), 0) + 1
        accepted = {"K_per_cycle": k_list, "histogram": histogram} if k_list else "NOT_COLLECTED"
        cycle = {"cycles_observed": len(per_cycle), "per_cycle": per_cycle}
    else:
        accepted = "NOT_COLLECTED"
        cycle = "NOT_COLLECTED"
    if accepted == "NOT_COLLECTED":
        values = [r.get("timings", {}).get("draft_n_accepted") for r in measured]
        if values and all(v is not None for v in values):
            accepted = {"source": "timings.draft_n_accepted_per_request", "values": values}
    return {"accepted_prefix_distribution": accepted, "cycle_D_V_S_H_K": cycle}


AGENTS_MINIMUM_FIELDS = (
    "contract_version", "workload_hash", "quality_gate_version", "timing_boundary",
    "all_model_shard_SHA256", "draft_SHA256", "source_dirty_identity",
    "binary_SHA256", "actual_loaded_DSO_SHA256", "rank_to_BDF",
    "resolved_env_and_CLI", "process_id", "paired_block", "phase",
    "public_completion_tokens", "finish_reason", "complete_text_hash_and_raw_body",
    "generation_ms", "prompt_ms", "first_token_ms", "request_wall_ms", "startup_ms",
    "mtp_horizon", "accepted_prefix_distribution", "cycle_D_V_S_H_K",
    "actual_pipeline_hits", "active_rows", "capacity_rows", "bank_epoch",
    "fallback", "timeout", "GPU_fault", "HTTP_error", "teardown", "quality_verdict",
)

RANK_STAGE_LINE_SPECS = {
    # tag -> (category, required numeric keys), following the actual print
    # sites: per-exec aggregates (ggml-tp5-profile.cpp print_summary),
    # per-stage relay rows (ggml-vulkan-collective.cpp relay_stage_detail),
    # definition-time rows, and per-graph rows (src/llama-context.cpp).
    "[tp5-profile]":           ("exec_summary",   ("exec",)),
    "[tp5-relay-profile]":     ("exec_summary",   ("exec",)),
    "[tp5-latebind-profile]":  ("exec_summary",   ("exec",)),
    "[tp5-rerot-profile]":     ("exec_summary",   ("exec",)),
    "[tp5-gdn-profile]":       ("exec_summary",   ("exec",)),
    "[tp5-relay-stage]":       ("relay_stage",    ("exec", "stage")),
    "[tp5-latebind-stage]":    ("latebind_stage", ("exec", "stage")),
    "[tp5-linear-definition]": ("definition",     ("rank", "stages")),
    "[tp5-dependency-ledger]": ("dependency",     ("rank", "stages")),
    "[tp5-mtp-graph]":         ("graph",          ("type", "reuse")),
}

# AGENTS.md §7.3 dimensions versus what the current source actually prints.
# The collector parses only real lines; dimensions without a print site are
# declared here instead of being fabricated.
RANK_STAGE_FIELD_BOUNDARIES = {
    "role": "per-stage role NOT_PRINTED; [tp5-mtp-graph] frame_role exists at graph level only",
    "phase": "request phase (warmup/measured/guard) attributed by the collector; [tp5-mtp-graph] execution_phase printed at graph level",
    "rank": "[tp5-linear-definition]/[tp5-dependency-ledger] print rank=; [tp5-relay-stage] prints a per-rank ready_rank_us list",
    "stage": "[tp5-relay-stage]/[tp5-latebind-stage] print stage=",
    "active_rows": "NOT_PRINTED per stage (relay submit takes active_rows but never logs it); [tp5-mtp-graph] ubatch_tokens is graph-level token rows, different granularity",
    "capacity_rows": "NOT_PRINTED per stage; [tp5-mtp-graph] target_capacity is predefined target admission, different semantics",
    "M_N_K": "NOT_PRINTED in profile lines",
    "weight_type": "NOT_PRINTED in profile lines",
    "pipeline": "per-stage native pipeline NOT_PRINTED; [tp5-profile] creplay/plan are per-exec aggregate counters; relay direct=1/0 is route selection, not pipeline",
    "moe_experts": "NOT_PRINTED: expert selection is not in profile lines",
    "bank_epoch": "[tp5-relay-stage] epoch printed (bank=epoch&1 derivable); reader-done completion relationship NOT_PRINTED",
}


def parse_rank_stage_line(line):
    """Parse one GGML_TP5_PROFILE-family log line into (category, record).

    Returns None when the line carries no recognized tag, or when the tag is
    recognized but a required key is missing or non-numeric (malformed):
    such lines are skipped and counted by the caller, never repaired into a
    fabricated record. ready_rank_us expands its comma list to floats; a
    single numeric value becomes a one-element list.
    """
    tag = next((t for t in RANK_STAGE_LINE_SPECS if t in line), None)
    if tag is None:
        return None
    category, required = RANK_STAGE_LINE_SPECS[tag]
    fields = _parse_kv_fields(line)
    for key in required:
        value = fields.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    raw_ranks = fields.get("ready_rank_us")
    if raw_ranks is not None:
        if isinstance(raw_ranks, (int, float)):
            fields["ready_rank_us"] = [float(raw_ranks)]
        elif isinstance(raw_ranks, str):
            try:
                fields["ready_rank_us"] = [float(x) for x in raw_ranks.split(",")]
            except ValueError:
                return None
        else:
            return None
    fields["tag"] = tag
    return category, fields


def _numeric_stats(values):
    if not values:
        return None
    return {"n": len(values), "min": min(values), "max": max(values),
            "mean": sum(values) / len(values)}


def extract_rank_stage_ledger(request_segments, profile_mode, out_dir=None):
    """AGENTS.md §7.3 rank/stage ledger from per-request server.log segments.

    request_segments is an iterable of (idx, phase, lines). Only fields the
    source actually prints are collected; §7.3 dimensions without a print
    site stay declared in field_boundaries rather than fabricated. When no
    profile line appears in any segment the ledger stays NOT_COLLECTED with
    a mode-specific reason. Full per-row detail is written to rank-stage.json
    in out_dir when any row parsed; the returned summary keeps aggregates,
    per-request counts, slowest-rank localization and malformed samples.
    """
    categories = ("exec_summary", "relay_stage", "latebind_stage",
                  "definition", "dependency", "graph")
    rows = {c: [] for c in categories}
    malformed = []
    per_request = {}
    total = parsed = 0
    for idx, phase, lines in request_segments:
        counts = {}
        for line in lines:
            tag = next((t for t in RANK_STAGE_LINE_SPECS if t in line), None)
            if tag is None:
                continue
            total += 1
            outcome = parse_rank_stage_line(line)
            if outcome is None:
                malformed.append({"request_idx": idx, "tag": tag, "line": line[:200]})
                continue
            category, record = outcome
            record["request_idx"] = idx
            record["phase"] = phase
            rows[category].append(record)
            parsed += 1
            counts[tag] = counts.get(tag, 0) + 1
        if counts:
            per_request[str(idx)] = dict(counts, phase=phase)
    if parsed == 0 and not malformed:
        if profile_mode == "none":
            reason = ("--profile none leaves GGML_TP5_PROFILE and GGML_TP5_MTP_PROFILE "
                      "unset; no rank/stage lines are expected")
        elif profile_mode == "cycle":
            reason = ("zero [tp5-*] rank/stage lines in request log segments under "
                      "--profile cycle: cycle sets only GGML_TP5_MTP_PROFILE=1 (cycle "
                      "ledger); rank/stage rows additionally need GGML_TP5_PROFILE=1 and "
                      "GGML_TP5_PROFILE_RELAY_STAGES=1 as set by --profile full or --env")
        elif profile_mode == "kernels":
            reason = ("zero [tp5-*] rank/stage lines in request log segments under "
                      "--profile kernels: kernels sets only GGML_VK_PERF_LOGGER=1")
        else:
            reason = ("zero [tp5-*] profile lines observed despite --profile full; the "
                      "profiled code paths may not have executed in this run")
        return {"status": "NOT_COLLECTED", "reason": reason,
                "field_boundaries": RANK_STAGE_FIELD_BOUNDARIES}

    relay = rows["relay_stage"]
    relay_stats = {}
    for key in ("total_us", "ready_all_us", "arm_us", "cpu_data_us", "generation_us"):
        stats = _numeric_stats([r[key] for r in relay if isinstance(r.get(key), (int, float))])
        if stats:
            relay_stats[key] = stats
    slowest = None
    rank_lists = [r["ready_rank_us"] for r in relay if r.get("ready_rank_us")]
    if rank_lists:
        n_ranks = max(len(x) for x in rank_lists)
        rank_max = [max(x[i] for x in rank_lists if len(x) > i) for i in range(n_ranks)]
        slowest = {"max_ready_rank_us": rank_max, "slowest_rank": rank_max.index(max(rank_max))}

    exec_rows = rows["exec_summary"]
    graph_rows = rows["graph"]
    derived = {}
    creplay = [r["creplay"] for r in exec_rows if isinstance(r.get("creplay"), str)]
    plan = [r["plan"] for r in exec_rows if isinstance(r.get("plan"), str)]
    if creplay or plan:
        derived["actual_pipeline_hits"] = {"compute_replay": creplay, "collective_plan": plan}
    ubatch = [r["ubatch_tokens"] for r in graph_rows if isinstance(r.get("ubatch_tokens"), (int, float))]
    if ubatch:
        derived["active_rows_ubatch_tokens"] = ubatch
    capacity = [r["target_capacity"] for r in graph_rows if isinstance(r.get("target_capacity"), (int, float))]
    if capacity:
        derived["capacity_rows_target_capacity"] = capacity
    epochs = [r["epoch"] for r in relay if isinstance(r.get("epoch"), (int, float))]
    if epochs:
        derived["relay_epoch_range"] = [min(epochs), max(epochs)]

    ledger = {
        "status": "COLLECTED",
        "source": "GGML_TP5_PROFILE-family lines parsed from request log segments (request-{idx}.log)",
        "scope": ("request segments only (warmup/measured/guard); lines outside request "
                  "segments are counted separately by the runner against the full server.log"),
        "lines_total": total,
        "lines_parsed": parsed,
        "lines_malformed": len(malformed),
        "per_request": per_request,
        "counts": {c: len(rows[c]) for c in categories},
        "relay_stage_stats": relay_stats,
        "slowest_rank": slowest,
        "malformed_samples": malformed[:5],
        "derived": derived,
        "field_boundaries": RANK_STAGE_FIELD_BOUNDARIES,
    }
    if out_dir is not None:
        detail = dict(ledger)
        detail.pop("malformed_samples", None)
        detail["malformed"] = malformed
        for c in categories:
            detail[c] = rows[c]
        save(Path(out_dir) / "rank-stage.json", detail)
        ledger["full_detail_file"] = "rank-stage.json"
    return ledger


def audit_minimum_fields(result, manifest=None, mapped_libraries=None):
    """Cross-check result.json against the AGENTS.md §5 minimum field list.

    Mechanically collectable values are copied from runner state; every field
    without a maintained collector is reported as NOT_COLLECTED, never
    fabricated and never silently dropped.
    """
    nc = "NOT_COLLECTED"
    requests = result.get("requests") or []
    measured = [r for r in requests if r.get("phase") == "measured"]
    manifest = manifest or {}

    def timing_series(key):
        if not measured:
            return nc
        values = [r.get("timings", {}).get(key) for r in measured]
        return values if all(v is not None for v in values) else nc

    gpu_before = result.get("gpu_before") or []
    ledger = result.get("rank_stage_ledger")
    derived = ledger.get("derived", {}) if isinstance(ledger, dict) else {}
    values = {
        "contract_version": nc,
        "workload_hash": manifest.get("prompt_sha256", nc),
        "quality_gate_version": nc,
        "timing_boundary": nc,
        "all_model_shard_SHA256": nc,
        "draft_SHA256": nc,
        "source_dirty_identity": nc,
        "binary_SHA256": manifest.get("bin_sha256", nc),
        "actual_loaded_DSO_SHA256": mapped_libraries if isinstance(mapped_libraries, dict) else nc,
        "rank_to_BDF": [{"card": x.get("card"), "bdf": x.get("bdf")} for x in gpu_before] or nc,
        "resolved_env_and_CLI": {"argv": manifest.get("argv"), "env": manifest.get("env")} if manifest else nc,
        "process_id": result.get("process_id", nc),
        "paired_block": nc,
        "phase": [r.get("phase") for r in measured] or nc,
        "public_completion_tokens": [r.get("n") for r in measured] or nc,
        "finish_reason": [r.get("finish") for r in measured] or nc,
        "complete_text_hash_and_raw_body": ({"per_request_content_sha256": [r.get("content_sha256") for r in measured],
                                             "raw_body_files": [f"request-{r.get('idx')}.json" for r in measured]}
                                            if measured else nc),
        "generation_ms": [r.get("predicted_ms") for r in measured] or nc,
        "prompt_ms": timing_series("prompt_ms"),
        "first_token_ms": timing_series("first_token_ms"),
        "request_wall_ms": [r.get("wall_ms") for r in measured] or nc,
        "startup_ms": result.get("loading_ms", nc),
        "mtp_horizon": result.get("mtp_horizon", nc),
        "accepted_prefix_distribution": result.get("accepted_prefix_distribution", nc),
        "cycle_D_V_S_H_K": result.get("cycle_D_V_S_H_K", nc),
        "actual_pipeline_hits": derived.get("actual_pipeline_hits", nc),
        "active_rows": ({"source": "[tp5-mtp-graph] ubatch_tokens (graph-level rows; per-stage active_rows has no print site)",
                         "values": derived["active_rows_ubatch_tokens"]}
                        if "active_rows_ubatch_tokens" in derived else nc),
        "capacity_rows": ({"source": "[tp5-mtp-graph] target_capacity (predefined admission; per-stage capacity_rows has no print site)",
                           "values": derived["capacity_rows_target_capacity"]}
                          if "capacity_rows_target_capacity" in derived else nc),
        "bank_epoch": nc,
        "fallback": [r.get("fallback_lines") for r in measured] or nc,
        "timeout": nc,
        "GPU_fault": nc,
        "HTTP_error": nc,
        "teardown": result.get("shutdown", nc),
        "quality_verdict": nc,
    }
    notes = {
        "all_model_shard_SHA256": "manifest.json carries stat identity; full shard SHA256 is computed offline by DevOps before GPU runs (AGENTS.md §4.2)",
        "draft_SHA256": "same offline policy as model shards",
        "actual_pipeline_hits": "per-exec creplay/plan counters parsed from [tp5-profile] lines (needs GGML_TP5_PROFILE=1, --profile full); no per-stage print site",
        "active_rows": "graph-level ubatch_tokens from [tp5-mtp-graph]; per-stage active_rows has no print site in current source",
        "capacity_rows": "graph-level target_capacity from [tp5-mtp-graph]; per-stage capacity_rows has no print site in current source",
        "bank_epoch": "relay epoch printed per [tp5-relay-stage] row (bank=epoch&1 derivable); reader-done completion has no print site",
    }
    return {"checklist": list(AGENTS_MINIMUM_FIELDS), "fields": values,
            "not_collected": [k for k in AGENTS_MINIMUM_FIELDS if values[k] == nc],
            "checklist_covered": set(values) == set(AGENTS_MINIMUM_FIELDS),
            "notes": notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=1, choices=range(0, 17))
    ap.add_argument("--latebind", choices=("off", "exact", "aggressive", "q8first", "pack-once", "ap", "l2", "l2-ap"), default="off")
    ap.add_argument("--mmvq", choices=("disabled", "auto", "forced"), default="disabled")
    ap.add_argument("--threads", type=int, default=22)
    ap.add_argument("--placement", choices=("tp5", "layer"), default="tp5",
                    help="Same five devices: TP5 tensor graph or ordinary layer placement")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--profile", choices=("none", "cycle", "full", "kernels"), default="none")
    ap.add_argument("--env", action="append", default=[], help="Explicit single-factor KEY=VALUE; recorded in manifest")
    ap.add_argument("--skip-guard", action="store_true")
    ap.add_argument("--prompt-file", type=Path, help="Explicit frozen UTF-8 workload, separate from the default calibration prompt")
    ap.add_argument("--system-prompt-file", type=Path,
                    help="Frozen UTF-8 system instruction for the primary workload only; guard stays unchanged")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="Primary request safety cap, not a forced output length; natural stop remains required")
    ap.add_argument("--length-min", type=int, default=160,
                    help="Pre-registered length gate lower bound on measured public completion tokens")
    ap.add_argument("--length-max", type=int, default=240,
                    help="Pre-registered length gate upper bound on measured public completion tokens")
    a = ap.parse_args()
    canonical = a.prompt_file.read_text(encoding="utf-8").strip() if a.prompt_file else CANONICAL
    system_prompt = a.system_prompt_file.read_text(encoding="utf-8").strip() if a.system_prompt_file else None
    try:
        primary_payload = request_payload(canonical, system_prompt, a.max_tokens)
    except ValueError as e:
        ap.error(str(e))
    if min(a.threads, a.repeats) < 1 or a.warmup < 0 or a.ctx < 256:
        ap.error("threads/repeats must be positive, warmup nonnegative")
    if a.length_min < 0 or a.length_max < a.length_min:
        ap.error("length-min must be nonnegative and not above length-max")
    a.bin, a.model, a.draft, a.out = (p.resolve() for p in (a.bin, a.model, a.draft, a.out))
    for p in (a.bin, a.model, a.draft):
        if not p.is_file():
            ap.error(f"Missing input: {p}")
    a.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "STARTED", "qualification": "NOT_REVIEWED", "profile": a.profile}
    manifest = None
    mapped_libraries = None
    proc = None
    with open("/var/tmp/tp5-gpu.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result["gpu_before"] = idle_gate()
            build_gate()
            env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
            env.update({"LD_LIBRARY_PATH": str(a.bin.parent), "RADV_DEBUG": "nobolist",
                        "GGML_VK_DISABLE_MMVQ": "1" if a.mmvq == "disabled" else "0",
                        "GGML_VK_FORCE_MMVQ": "1" if a.mmvq == "forced" else "0"})
            lb = a.latebind if a.latebind in ("off", "exact") else "aggressive"
            if a.latebind in ("q8first", "ap", "l2-ap"):
                env["GGML_TP5_LATEBIND_Q8_ORDER"] = "first"
            if a.latebind in ("pack-once", "ap", "l2-ap"):
                env["GGML_TP5_LATEBIND_PACK_ONCE"] = "1"
            if a.latebind in ("l2", "l2-ap"):
                env["GGML_TP5_LATEBIND_L2"] = "td"
            if a.profile in ("cycle", "full", "kernels"):
                env["GGML_TP5_MTP_PROFILE"] = "1"
            if a.profile == "full":
                env.update({"GGML_TP5_PROFILE": "1", "GGML_TP5_GPU_TIMING": "1", "GGML_TP5_PROFILE_RELAY_STAGES": "1"})
            if a.profile == "kernels":
                env["GGML_VK_PERF_LOGGER"] = "1"
            for setting in a.env:
                key, sep, value = setting.partition("=")
                if not sep or not key.startswith("GGML_") or "REROT" in key:
                    raise ValueError(f"Expected non-RERoT GGML_* KEY=VALUE: {setting}")
                env[key] = value
            if env.get("GGML_DECODE_COST", "0") not in ("0", "off"):
                result["profile"] = "host-cost"
            cmd = [str(a.bin), "-m", str(a.model), "-dev", "Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4",
                   "--split-mode", "tensor" if a.placement == "tp5" else "layer", "--fit", "off", "-ngl", "999", "-c", str(a.ctx), "-b", "32", "-ub", "32",
                   "--no-mmap", "--no-host", "-np", "1", "-t", str(a.threads), "--host", "127.0.0.1", "--port", str(a.port),
                   ]
            if a.placement == "tp5":
                cmd += ["--tp5", "qwen4exp-af", "--tp5-latebind", lb]
            elif a.latebind != "off":
                raise ValueError("Layer placement has no TP5 LateBind; specify --latebind off")
            if a.horizon:
                cmd += ["-md", str(a.draft), "--spec-type", "draft-mtp", "--spec-draft-n-max", str(a.horizon), "--spec-draft-p-min", "0"]
            else:
                cmd += ["--spec-type", "none"]
            manifest = {"argv": cmd, "env": env, "bin_sha256": sha(a.bin),
                                          "harness_sha256": sha(Path(__file__)),
                                          "model_stat_identity": model_stat_identity(a.model),
                                          "draft_stat_identity": model_stat_identity(a.draft),
                                          "prompt_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                                          "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest() if system_prompt is not None else None,
                                          "primary_payload": primary_payload,
                                          "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            save(a.out / "manifest.json", manifest)
            log_path = a.out / "server.log"
            loading_begin = time.monotonic_ns()
            with log_path.open("wb") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, start_new_session=True)
            result["process_id"] = proc.pid
            deadline = time.monotonic() + 240
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited during loading: {proc.returncode}")
                try:
                    if api(a.port, "/health", timeout=1).get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Server loading deadline exceeded")
                time.sleep(0.25)
            result["loading_ms"] = (time.monotonic_ns() - loading_begin) / 1e6
            build_gate()
            mapped_libraries = library_identity(proc.pid)
            save(a.out / "mapped-libraries.json", mapped_libraries)
            requests = []
            prompts = [("warmup", canonical)] * a.warmup + [("measured", canonical)] * a.repeats
            if not a.skip_guard:
                prompts.append(("guard", GUARD))
            for idx, (phase, prompt) in enumerate(prompts):
                build_gate()
                payload = request_payload(prompt) if phase == "guard" else primary_payload
                begin = log_path.stat().st_size
                t = time.monotonic_ns()
                try:
                    response = api(a.port, "/v1/chat/completions", payload)
                except urllib.error.HTTPError as e:
                    save(a.out / f"request-{idx}-error.json", {
                        "payload": payload, "phase": phase, "http_status": e.code,
                        "body": e.read().decode("utf-8", errors="replace"),
                        "wall_ms": (time.monotonic_ns() - t) / 1e6,
                        "log_begin": begin, "log_end": log_path.stat().st_size})
                    raise
                wall_ms = (time.monotonic_ns() - t) / 1e6
                end = log_path.stat().st_size
                save(a.out / f"request-{idx}.json", {"payload": payload, "response": response, "wall_ms": wall_ms,
                                                  "log_begin": begin, "log_end": end, "phase": phase})
                ch = response["choices"][0]
                text = ch["message"].get("content") or ""
                timings = response["timings"]
                n = response["usage"]["completion_tokens"]
                with log_path.open("rb") as f:
                    f.seek(begin)
                    segment = f.read(end - begin).decode("utf-8", errors="replace")
                (a.out / f"request-{idx}.log").write_text(segment)
                issues = []
                if ch["finish_reason"] != "stop":
                    issues.append("NOT_NATURAL_STOP")
                if len(text.strip()) < 20 or re.search(r"(?:^|\n)\s*[-*]\s*\n\s*[-*]\s*\n", text):
                    issues.append("DEGENERATE_OUTPUT")
                item = {"idx": idx, "phase": phase, "n": n, "predicted_ms": timings["predicted_ms"],
                        "tps": n * 1000 / timings["predicted_ms"], "wall_ms": wall_ms,
                        "finish": ch["finish_reason"], "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "timings": timings, "issues": issues,
                        "fallback_lines": [x for x in segment.splitlines() if "fallback" in x.lower()],
                        "mtp_cycle_lines": [x for x in segment.splitlines() if "[tp5-mtp-cycle]" in x]}
                requests.append(item)
                save(a.out / "requests.json", requests)
                build_gate()
                print(json.dumps({"out": a.out.name, **{k: item[k] for k in ("idx", "phase", "n", "tps", "issues")}}, ensure_ascii=False), flush=True)
                if issues:
                    raise RuntimeError(f"Mechanical output failure: {issues}; raw output retained")
            measured = [r for r in requests if r["phase"] == "measured"]
            cycle_ledger = extract_cycle_ledger(requests, a.horizon)
            segments = [(r["idx"], r["phase"],
                         (a.out / f"request-{r['idx']}.log").read_text(encoding="utf-8", errors="replace").splitlines())
                        for r in requests]
            rank_stage_ledger = extract_rank_stage_ledger(segments, a.profile, a.out)
            server_log_tag_lines = sum(1 for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                                       if any(t in line for t in RANK_STAGE_LINE_SPECS))
            if server_log_tag_lines != rank_stage_ledger.get("lines_total", 0):
                rank_stage_ledger["server_log_tag_lines_total"] = server_log_tag_lines
                rank_stage_ledger["outside_request_segments"] = server_log_tag_lines - rank_stage_ledger.get("lines_total", 0)
            result.update({"status": "MEASURED_NOT_QUALIFIED" if result["profile"] == "none" else "DIAGNOSTIC_NOT_RANKED",
                           "pooled_tps": 1000 * sum(r["n"] for r in measured) / sum(r["predicted_ms"] for r in measured),
                           "median_tps": statistics.median(r["tps"] for r in measured), "requests": requests,
                           "length_gate": evaluate_length_gate(requests, length_min=a.length_min, length_max=a.length_max),
                           "mtp_horizon": a.horizon,
                           "accepted_prefix_distribution": cycle_ledger["accepted_prefix_distribution"],
                           "cycle_D_V_S_H_K": cycle_ledger["cycle_D_V_S_H_K"],
                           "rank_stage_ledger": rank_stage_ledger})
        except Exception as e:
            result.update({"status": "FAILED", "error": repr(e)})
        finally:
            if proc is not None:
                start = time.monotonic()
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                forced = False
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    forced = True
                    proc.kill()
                    rc = proc.wait(timeout=10)
                result["shutdown"] = {"returncode": rc, "forced": forced, "seconds": time.monotonic() - start}
                if forced or rc != 0:
                    result["status"] = "FAILED"
                    result["teardown_error"] = "Non-normal owned teardown"
                    result.setdefault("error", "Non-normal owned teardown")
            result["gpu_after"] = gpu_state()
            # AGENTS.md §7.3 rank/stage ledger: parsed from
            # GGML_TP5_PROFILE-family lines in request log segments once the
            # run reaches collection; §7.3 dimensions without print sites are
            # declared in field_boundaries instead of fabricated.
            result.setdefault("mtp_horizon", a.horizon)
            result.setdefault("rank_stage_ledger",
                              {"status": "NOT_COLLECTED",
                               "reason": "run ended before rank/stage collection (failure or early exit)"})
            result.setdefault("length_gate", {"length_min": a.length_min, "length_max": a.length_max,
                                              "phase_scope": "measured", "passed": False,
                                              "requests": "NOT_COLLECTED"})
            result.setdefault("minimum_fields", audit_minimum_fields(result, manifest, mapped_libraries))
            save(a.out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("requests", "gpu_before", "gpu_after")}, ensure_ascii=False), flush=True)
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
