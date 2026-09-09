#!/usr/bin/env python3
"""
scripts/xkv/off-regression.py
-----------------------------
P0 Tri/XKV OFF zero-regression harness: compares a baseline binary against a
candidate binary with XKV explicitly OFF on identical model/prompt/seed/
template/sampling/budget/backend/KV/recurrent/RAM/preemption settings.

For each side it captures the binary SHA256, full argv/env fingerprint, output
tokens/text, exit/status, throughput, RSS/VRAM observations, KV/recurrent
fitting logs, prompt-cache state bytes/checksum, and scorer/XKV allocation
logs and metrics. It then asserts:

  - greedy token sequence exactly equal (token ids when emitted, else exact
    generated text);
  - no XKV store/scorer/scratch/sparse semantics in logs, metrics, or
    allocations (any active XKV/Tri signal FAILs);
  - legacy KV/recurrent fit and preemption event trace equivalent under
    configurable numeric tolerances;
  - throughput/VRAM/RSS within explicit thresholds (regression FAILs);
  - production runs use two distinct binaries (identical hashes are rejected
    unless --allow-same-binary-test, which marks the evidence synthetic).

Missing model/calibration/binaries/prompts or any missing required observation
yields NOT_EVALUATED and a nonzero exit (fail-closed, never a synthetic pass).
Only the Python standard library is used. No shell: subprocess argv arrays
only. Output is written atomically; --resume reuses completed per-side raw
records instead of re-executing.

Evidence schema: --schema prints EVIDENCE_SCHEMA_JSON; --help embeds it.
$schema id "xkv-evidence/v1" is shared with run-evaluation-matrix.py; key
names mirror compression-gate.py evidence fields where they overlap.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

VERSION = 1
SCHEMA_ID = "xkv-evidence/v1"

EVIDENCE_SCHEMA_JSON = {
    "$schema": SCHEMA_ID,
    "description": (
        "OFF zero-regression evidence. Consumed by reviewers and optionally "
        "attached to compression-gate.py runs. Keys mirror gate evidence "
        "fields where they overlap. Missing/unmeasured keys are the string "
        "'not_evaluated'. Synthetic runs carry __synthetic_test_fixture__=true "
        "and are rejected by production gates."
    ),
    "keys": {
        "off_regression_version": "int schema version (1)",
        "off_regression_pass": "bool true iff every verdict is PASS",
        "evidence_kind": "'real' or 'synthetic'",
        "xkv_requested_profile": "always 'OFF' for this harness",
        "xkv_effective_profile": "'OFF' when no XKV semantics observed, else 'not_evaluated'",
        "calibration_sha256": "str hex | 'not_evaluated'",
        "source_fingerprint": "git commit or 'not_evaluated'",
        "profile_fingerprint": "sha256 of the shared OFF settings dict",
        "verdicts": "dict of PASS|FAIL|NOT_EVALUATED verdict objects",
        "verdict_summary": "{PASS,FAIL,NOT_EVALUATED} counts",
        "manifests": "per-side immutable run manifests (hashes, argv, env, ...)",
        "observations": "per-side captured outputs and parsed metrics",
        "comparisons": "per-check numeric detail (deltas, tolerances)",
    },
}

# Machine-readable log protocol emitted by instrumented binaries / fixtures.
# Real llama binaries fall back to generic patterns documented in parse_log().
TOKEN_LINE = "OUTPUT_TOKENS:"
TEXT_LINE = "OUTPUT_TEXT:"
TPS_LINE = "THROUGHPUT_TPS:"
KV_LINE = "KV_CAPACITY_TOKENS:"
REC_SLOTS_LINE = "RECURRENT_SLOTS:"
REC_PENS_LINE = "RECURRENT_PENS:"
CACHE_BYTES_LINE = "PROMPT_CACHE_BYTES:"
CACHE_SHA_LINE = "PROMPT_CACHE_SHA256:"
RSS_LINE = "RSS_BYTES:"
VRAM_LINE = "VRAM_BYTES:"
PREEMPT_LINE = "PREEMPT_EVENT:"

# Any active XKV/Tri/sparse/factor signal in OFF mode is a hard FAIL.
FORBIDDEN_PATTERNS = (
    r"xkv[_-]?store",
    r"xkv[_-]?scorer",
    r"xkv[_-]?scratch",
    r"xkv[_-]?alloc",
    r"tri[_-]?scorer",
    r"tri[_-]?scratch",
    r"triattention\s+drain",
    r"triattention\s+maintenance",
    r"triattention\s+floor",
    r"tri_drain_total\s*[:=]\s*[1-9]",
    r"tri_maintenance_total\s*[:=]\s*[1-9]",
    r"tri_cells_(before|after|freed)\s*[:=]\s*[1-9]",
    r"sparse\s*(store|state|semantics|gap)",
    r"factor[_-]?(quant|encode|bytes)",
    r"landmark[_-]?(encode|quant|bytes)",
    r"sr_selected_rows\s*[:=]\s*[1-9]",
    r"dense_tiled_reconstruction\s*[:=]\s*true",
)
FORBIDDEN_RES = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]

# Legacy preemption/fallback events whose ordered trace must be equivalent.
TRACE_PATTERNS = (
    ("idle-demotion", re.compile(r"idle\s*demotion", re.IGNORECASE)),
    ("active-preemption", re.compile(r"active\s*(slot\s*)?preemption", re.IGNORECASE)),
    ("prefill-batch-limit", re.compile(r"(batch\s*limit|limiting\s*next\s*batch)", re.IGNORECASE)),
    ("kv-floor-exhausted", re.compile(r"(kv\s*floor\s*exhausted|floor\s*exhausted)", re.IGNORECASE)),
    ("recurrent-victim", re.compile(r"recurrent.*(victim|fallback|handling)", re.IGNORECASE)),
    ("slot-evict", re.compile(r"slot\s*evict", re.IGNORECASE)),
    ("context-exceeded", re.compile(r"context\s*size\s*exceeded", re.IGNORECASE)),
)

GENERIC_TPS_RES = (
    re.compile(r"([0-9]+\.[0-9]+)\s*tok/s", re.IGNORECASE),
    re.compile(r"throughput[^0-9]{0,20}([0-9]+\.[0-9]+)", re.IGNORECASE),
)
GENERIC_KV_RES = (
    re.compile(r"unified\s*KV\s*capacity\s*[:=]\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"\bKV\s*(size|capacity)\s*[:=]?\s*([0-9]+)", re.IGNORECASE),
)
GENERIC_REC_RES = (
    re.compile(r"recurrent\s*capacity\s*[:=]?\s*([0-9]+)", re.IGNORECASE),
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str):
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


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
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
            f.flush()
            os.fsync(f.fileno())
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
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "not_evaluated"


def load_metrics_sidecar(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {"_missing": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {"_malformed": path}
    except (OSError, ValueError):
        return {"_malformed": path}


def parse_log(stdout_b: bytes, stderr_b: bytes) -> dict:
    """Parse protocol lines first, then generic llama-style fallbacks."""
    text = (stdout_b + b"\n" + stderr_b).decode("utf-8", errors="replace")
    obs: dict = {
        "tokens": None,
        "text": None,
        "throughput_tps": None,
        "kv_capacity_tokens": None,
        "recurrent_slots": None,
        "recurrent_pens": None,
        "prompt_cache_bytes": None,
        "prompt_cache_sha256": None,
        "rss_bytes": None,
        "vram_bytes": None,
        "preempt_trace": [],
        "token_ids_unavailable": False,
    }
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(TOKEN_LINE):
            try:
                obs["tokens"] = [int(x) for x in s[len(TOKEN_LINE):].split(",") if x.strip() != ""]
            except ValueError:
                obs["tokens"] = None
        elif s.startswith(TEXT_LINE):
            obs["text"] = s[len(TEXT_LINE):].lstrip()
        elif s.startswith(TPS_LINE):
            try:
                obs["throughput_tps"] = float(s[len(TPS_LINE):].strip())
            except ValueError:
                pass
        elif s.startswith(KV_LINE):
            try:
                obs["kv_capacity_tokens"] = int(s[len(KV_LINE):].strip())
            except ValueError:
                pass
        elif s.startswith(REC_SLOTS_LINE):
            try:
                obs["recurrent_slots"] = int(s[len(REC_SLOTS_LINE):].strip())
            except ValueError:
                pass
        elif s.startswith(REC_PENS_LINE):
            try:
                obs["recurrent_pens"] = int(s[len(REC_PENS_LINE):].strip())
            except ValueError:
                pass
        elif s.startswith(CACHE_BYTES_LINE):
            try:
                obs["prompt_cache_bytes"] = int(s[len(CACHE_BYTES_LINE):].strip())
            except ValueError:
                pass
        elif s.startswith(CACHE_SHA_LINE):
            obs["prompt_cache_sha256"] = s[len(CACHE_SHA_LINE):].strip()
        elif s.startswith(RSS_LINE):
            try:
                obs["rss_bytes"] = int(s[len(RSS_LINE):].strip())
            except ValueError:
                pass
        elif s.startswith(VRAM_LINE):
            try:
                obs["vram_bytes"] = int(s[len(VRAM_LINE):].strip())
            except ValueError:
                pass
        elif s.startswith(PREEMPT_LINE):
            obs["preempt_trace"].append(s[len(PREEMPT_LINE):].strip())
    # Generic fallbacks from real llama logs.
    if obs["throughput_tps"] is None:
        for rx in GENERIC_TPS_RES:
            m = rx.search(text)
            if m:
                try:
                    obs["throughput_tps"] = float(m.group(1))
                    break
                except ValueError:
                    continue
    if obs["kv_capacity_tokens"] is None:
        for rx in GENERIC_KV_RES:
            m = rx.search(text)
            if m:
                try:
                    obs["kv_capacity_tokens"] = int(m.group(m.lastindex or 1))
                    break
                except ValueError:
                    continue
    if obs["recurrent_slots"] is None:
        for rx in GENERIC_REC_RES:
            m = rx.search(text)
            if m:
                try:
                    obs["recurrent_slots"] = int(m.group(1))
                    break
                except ValueError:
                    continue
    if obs["tokens"] is None and obs["text"] is None:
        # No structured output at all: text fallback over full stdout, so a
        # real binary that just prints completion text is still comparable.
        raw = stdout_b.decode("utf-8", errors="replace")
        if raw.strip():
            obs["text"] = raw
            obs["token_ids_unavailable"] = True
        else:
            obs["token_ids_unavailable"] = True
    elif obs["tokens"] is None:
        obs["token_ids_unavailable"] = True
    # Legacy trace fallback: map generic log lines to canonical event kinds.
    if not obs["preempt_trace"]:
        for line in text.splitlines():
            for kind, rx in TRACE_PATTERNS:
                if rx.search(line):
                    obs["preempt_trace"].append(kind)
                    break
    obs["combined_log"] = text
    return obs


def find_forbidden(text: str, metrics: dict) -> list:
    hits = []
    for rx in FORBIDDEN_RES:
        m = rx.search(text)
        if m:
            hits.append(m.group(0)[:120])
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        kl = k.lower()
        if kl.startswith("xkv_") and isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
            hits.append(f"metric {k}={v}")
        if kl.startswith("tri_") and isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
            hits.append(f"metric {k}={v}")
        if isinstance(v, str) and v not in ("none", "off", "disabled", "not_evaluated"):
            if kl in ("xkv_effective_profile", "xkv_effective_storage_profile") and v:
                hits.append(f"metric {k}={v}")
    return hits


def num_equal(a, b, rel_tol: float, abs_tol: float) -> bool:
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa == fb:
        return True
    return abs(fa - fb) <= max(abs_tol, rel_tol * max(abs(fa), abs(fb)))


class OffRegression:
    def __init__(self, args):
        self.args = args
        self.settings = {
            "model": args.model,
            "template": args.template or "",
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "n_predict": args.n_predict,
            "n_ctx": args.n_ctx,
            "total_kv": args.total_kv,
            "ubatch": args.ubatch,
            "batch": args.batch,
            "backend": args.backend,
            "recurrent_people": args.recurrent_people,
            "recurrent_pens": args.recurrent_pens,
            "ram_swap": args.ram_swap,
            "xkv": "off",
            "triattention": "off",
            "extra_args": list(args.extra_arg or []),
        }
        self.settings_fp = sha256_hex(canonical(self.settings))

    def _argv_for(self, binary: str, prompt: str) -> list:
        argv = [binary, "-m", self.args.model, "-p", prompt,
                "--seed", str(self.args.seed),
                "--temp", str(self.args.temperature),
                "--top-p", str(self.args.top_p),
                "--top-k", str(self.args.top_k),
                "-n", str(self.args.n_predict),
                "-c", str(self.args.n_ctx)]
        if self.args.template:
            argv += ["--template", self.args.template]
        if self.args.total_kv:
            argv += ["--total-kv", self.args.total_kv]
        if self.args.ubatch:
            argv += ["-ub", str(self.args.ubatch)]
        if self.args.batch:
            argv += ["-b", str(self.args.batch)]
        argv += ["--xkv", "off"]
        for e in (self.args.extra_arg or []):
            argv.append(e)
        return argv

    def _side_path(self, side: str) -> str:
        return os.path.join(self.args.workdir, f"{side}.raw.json")

    def run_side(self, side: str, binary: str, prompt: str) -> dict:
        """Execute one side (or reuse a resume record). Never uses a shell."""
        raw_path = self._side_path(side)
        if self.args.resume and os.path.exists(raw_path):
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if (isinstance(rec, dict)
                        and rec.get("settings_fingerprint") == self.settings_fp
                        and rec.get("manifest", {}).get("binary_sha256") == sha256_file(binary)
                        and rec.get("prompt_sha256") == sha256_hex(prompt.encode("utf-8"))):
                    return rec
            except (OSError, ValueError):
                pass
        argv = self._argv_for(binary, prompt)
        start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.time()
        timed_out = False
        stdout_b, stderr_b = b"", b""
        returncode = None
        error = None
        try:
            p = subprocess.run(argv, capture_output=True, timeout=self.args.timeout)
            stdout_b, stderr_b = p.stdout or b"", p.stderr or b""
            returncode = p.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            out = e.stdout if isinstance(e.stdout, bytes) else b""
            err = e.stderr if isinstance(e.stderr, bytes) else b""
            stdout_b = out + b"\nTIMEOUT"
            stderr_b = err
            error = f"timeout after {self.args.timeout}s"
        except OSError as e:
            error = f"os_error: {e}"
        end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_bytes(os.path.join(self.args.workdir, f"{side}.stdout"), stdout_b)
        atomic_write_bytes(os.path.join(self.args.workdir, f"{side}.stderr"), stderr_b)
        manifest = {
            "side": side,
            "binary_path": binary,
            "binary_sha256": sha256_file(binary),
            "model_path": self.args.model,
            "model_sha256": sha256_file(self.args.model),
            "calibration_path": self.args.calibration or "",
            "calibration_sha256": sha256_file(self.args.calibration) if self.args.calibration else None,
            "commit": git_commit(),
            "profile_fingerprint": self.settings_fp,
            "settings": self.settings,
            "argv": argv,
            "env": dict(sorted(os.environ.items())),
            "start": start,
            "end": end,
            "wall_s": time.time() - t0,
            "returncode": returncode,
            "timed_out": timed_out,
            "terminated_by_signal": (-returncode if isinstance(returncode, int) and returncode < 0 else None),
            "error": error,
            "timeout_s": self.args.timeout,
        }
        obs = parse_log(stdout_b, stderr_b)
        side_metrics = load_metrics_sidecar(
            self.args.metrics_baseline if side == "baseline" else self.args.metrics_candidate)
        # Merge numeric sidecar metrics into observations when the log lacks them.
        for k in ("throughput_tps", "kv_capacity_tokens", "recurrent_slots",
                  "recurrent_pens", "prompt_cache_bytes", "prompt_cache_sha256",
                  "rss_bytes", "vram_bytes"):
            if obs.get(k) is None and k in side_metrics and not k.startswith("_"):
                obs[k] = side_metrics[k]
        obs["metrics_sidecar"] = side_metrics
        obs["forbidden_hits"] = find_forbidden(obs.get("combined_log", ""), side_metrics)
        rec = {
            "manifest": manifest,
            "observations": {k: v for k, v in obs.items() if k != "combined_log"},
            "log_tail": obs.get("combined_log", "")[-4000:],
            "prompt_sha256": sha256_hex(prompt.encode("utf-8")),
            "settings_fingerprint": self.settings_fp,
        }
        rec["full_log_path"] = os.path.join(self.args.workdir, f"{side}.stdout")
        atomic_write_json(raw_path, rec)
        return rec

    def compare(self, base_rec: dict, cand_rec: dict) -> dict:
        verdicts: dict = {}
        reasons: list = []
        comparisons: dict = {}
        synthetic = bool(self.args.allow_same_binary_test)

        def _fail(key, reason):
            verdicts[key] = {"status": "FAIL", "reason": reason}
            reasons.append(f"[{key}] {reason}")

        def _pass(key, detail=None):
            v = {"status": "PASS"}
            if detail:
                v.update(detail)
            verdicts[key] = v

        def _na(key, reason):
            verdicts[key] = {"status": "NOT_EVALUATED", "reason": reason}
            reasons.append(f"[{key}] NOT_EVALUATED: {reason}")

        bman, cman = base_rec["manifest"], cand_rec["manifest"]
        bobs, cobs = base_rec["observations"], cand_rec["observations"]

        # 1. Artifact presence (binaries/model/prompts must exist with hashes).
        missing = []
        for label, man in (("baseline-binary", bman), ("candidate-binary", cman)):
            if not man.get("binary_sha256"):
                missing.append(label)
        if not bman.get("model_sha256") or not cman.get("model_sha256"):
            missing.append("model")
        if self.args.calibration and (not bman.get("calibration_sha256")
                                      or not cman.get("calibration_sha256")):
            missing.append("calibration")
        if missing:
            for key in ("binary_identity", "greedy_token_equality", "no_xkv_semantics",
                        "kv_recurrent_fit_equivalence", "preemption_trace_equivalence",
                        "prompt_cache_equivalence", "performance_regression", "exit_status"):
                _na(key, f"missing required artifact: {', '.join(sorted(set(missing)))}")
            verdicts["artifact_presence"] = {
                "status": "NOT_EVALUATED",
                "reason": f"missing required artifact: {', '.join(sorted(set(missing)))}",
            }
            reasons.append(f"[artifact_presence] NOT_EVALUATED: missing {sorted(set(missing))}")
            return self._report(verdicts, reasons, comparisons, synthetic,
                                base_rec, cand_rec,
                                "synthetic" if synthetic else "real")
        verdicts["artifact_presence"] = {"status": "PASS"}

        # 2. Binary identity: production runs must use distinct binaries.
        if bman["binary_sha256"] == cman["binary_sha256"]:
            if self.args.allow_same_binary_test:
                synthetic = True
                verdicts["binary_identity"] = {
                    "status": "PASS",
                    "synthetic": True,
                    "reason": "same binary explicitly allowed via --allow-same-binary-test",
                    "sha256": bman["binary_sha256"],
                }
            else:
                _fail("binary_identity",
                      "baseline and candidate binaries have identical SHA256; "
                      "production OFF regression must compare distinct builds "
                      "(use --allow-same-binary-test only for synthetic self-test)")
        else:
            _pass("binary_identity", {"baseline_sha256": bman["binary_sha256"],
                                      "candidate_sha256": cman["binary_sha256"]})

        # 3. Exit/status first: abnormal termination poisons later comparisons.
        for label, man in (("baseline", bman), ("candidate", cman)):
            if man.get("timed_out") or man.get("error") or man.get("returncode") != 0:
                _fail("exit_status",
                      f"{label}: returncode={man.get('returncode')} timed_out={man.get('timed_out')} "
                      f"signal={man.get('terminated_by_signal')} error={man.get('error')}")
                break
        else:
            _pass("exit_status", {"baseline_returncode": bman["returncode"],
                                  "candidate_returncode": cman["returncode"]})

        # 4. Greedy token sequence exactly equal.
        if bobs.get("tokens") is not None and cobs.get("tokens") is not None:
            comparisons["tokens_equal"] = bobs["tokens"] == cobs["tokens"]
            comparisons["baseline_token_count"] = len(bobs["tokens"])
            comparisons["candidate_token_count"] = len(cobs["tokens"])
            if bobs["tokens"] == cobs["tokens"]:
                _pass("greedy_token_equality",
                      {"token_count": len(bobs["tokens"]),
                       "token_ids_available": True})
            else:
                _fail("greedy_token_equality",
                      f"greedy token mismatch: baseline={bobs['tokens'][:32]} "
                      f"candidate={cobs['tokens'][:32]}")
        elif bobs.get("text") is not None and cobs.get("text") is not None:
            comparisons["tokens_equal"] = bobs["text"] == cobs["text"]
            comparisons["token_ids_available"] = False
            if bobs["text"] == cobs["text"]:
                _pass("greedy_token_equality",
                      {"text_bytes": len(bobs["text"].encode('utf-8')),
                       "token_ids_available": False,
                       "note": "token ids not emitted; exact generated-text fallback"})
            else:
                _fail("greedy_token_equality",
                      f"generated-text mismatch: baseline={bobs['text'][:200]!r} "
                      f"candidate={cobs['text'][:200]!r}")
        else:
            _na("greedy_token_equality", "no output tokens or text captured on one or both sides")

        # 5. No XKV semantics.
        hits = list(bobs.get("forbidden_hits", [])) + list(cobs.get("forbidden_hits", []))
        comparisons["forbidden_hits"] = hits
        if "exit_status" in verdicts and verdicts["exit_status"]["status"] != "PASS" and not hits:
            _na("no_xkv_semantics", "abnormal termination; log scan inconclusive")
        elif hits:
            _fail("no_xkv_semantics", f"unexpected XKV/Tri/sparse semantics: {hits[:8]}")
        else:
            # Both logs must be non-empty to make the negative claim meaningful.
            if not bobs.get("text") and bobs.get("tokens") is None:
                _na("no_xkv_semantics", "baseline produced no observable output; cannot assert absence")
            elif not cobs.get("text") and cobs.get("tokens") is None:
                _na("no_xkv_semantics", "candidate produced no observable output; cannot assert absence")
            else:
                _pass("no_xkv_semantics", {"forbidden_hits": []})

        # 6. Legacy KV/recurrent fit equivalence.
        fit_keys = ("kv_capacity_tokens", "recurrent_slots", "recurrent_pens")
        fit_missing = [k for k in fit_keys if bobs.get(k) is None or cobs.get(k) is None]
        comparisons["fit"] = {k: {"baseline": bobs.get(k), "candidate": cobs.get(k)} for k in fit_keys}
        if fit_missing:
            _na("kv_recurrent_fit_equivalence",
                f"missing fit observations: {fit_missing}")
        elif all(num_equal(bobs[k], cobs[k], self.args.fit_rel_tol, self.args.fit_abs_tol) for k in fit_keys):
            _pass("kv_recurrent_fit_equivalence",
                  {"rel_tol": self.args.fit_rel_tol, "abs_tol": self.args.fit_abs_tol,
                   "values": comparisons["fit"]})
        else:
            _fail("kv_recurrent_fit_equivalence",
                  f"fitting mismatch beyond tol rel={self.args.fit_rel_tol} abs={self.args.fit_abs_tol}: "
                  f"{comparisons['fit']}")

        # 7. Preemption trace equivalence (ordered kinds must match exactly).
        bt, ct = bobs.get("preempt_trace", []), cobs.get("preempt_trace", [])
        comparisons["preempt_trace"] = {"baseline": bt, "candidate": ct}
        if verdicts.get("exit_status", {}).get("status") != "PASS":
            _na("preemption_trace_equivalence", "abnormal termination; trace inconclusive")
        elif bt == ct:
            _pass("preemption_trace_equivalence", {"events": len(bt)})
        else:
            _fail("preemption_trace_equivalence",
                  f"preemption trace differs: baseline={bt} candidate={ct}")

        # 8. Prompt-cache state equivalence.
        if bobs.get("prompt_cache_bytes") is None or cobs.get("prompt_cache_bytes") is None \
                or not bobs.get("prompt_cache_sha256") or not cobs.get("prompt_cache_sha256"):
            _na("prompt_cache_equivalence", "missing prompt-cache bytes/checksum on one or both sides")
            comparisons["prompt_cache"] = {"baseline_bytes": bobs.get("prompt_cache_bytes"),
                                           "candidate_bytes": cobs.get("prompt_cache_bytes"),
                                           "checksum_equal": None}
        else:
            ceq = bobs["prompt_cache_sha256"] == cobs["prompt_cache_sha256"]
            beq = num_equal(bobs["prompt_cache_bytes"], cobs["prompt_cache_bytes"],
                            self.args.fit_rel_tol, self.args.fit_abs_tol)
            comparisons["prompt_cache"] = {"baseline_bytes": bobs["prompt_cache_bytes"],
                                           "candidate_bytes": cobs["prompt_cache_bytes"],
                                           "checksum_equal": ceq}
            if ceq and beq:
                _pass("prompt_cache_equivalence", comparisons["prompt_cache"])
            else:
                _fail("prompt_cache_equivalence",
                      f"prompt-cache mismatch: bytes {bobs['prompt_cache_bytes']} vs "
                      f"{cobs['prompt_cache_bytes']} checksum_equal={ceq}")

        # 9. Performance regression (explicit thresholds).
        tps_b, tps_c = bobs.get("throughput_tps"), cobs.get("throughput_tps")
        vram_b, vram_c = bobs.get("vram_bytes"), cobs.get("vram_bytes")
        rss_b, rss_c = bobs.get("rss_bytes"), cobs.get("rss_bytes")
        perf_missing = []
        if tps_b is None or tps_c is None:
            perf_missing.append("throughput_tps")
        if vram_b is None or vram_c is None:
            perf_missing.append("vram_bytes")
        if rss_b is None or rss_c is None:
            perf_missing.append("rss_bytes")
        comparisons["performance"] = {
            "throughput_tps": {"baseline": tps_b, "candidate": tps_c},
            "vram_bytes": {"baseline": vram_b, "candidate": vram_c},
            "rss_bytes": {"baseline": rss_b, "candidate": rss_c},
            "max_throughput_drop_pct": self.args.max_throughput_drop_pct,
            "max_vram_growth_pct": self.args.max_vram_growth_pct,
            "max_rss_growth_pct": self.args.max_rss_growth_pct,
        }
        if perf_missing:
            _na("performance_regression",
                f"missing performance observations: {perf_missing}")
        else:
            drop = (tps_b - tps_c) / tps_b * 100.0 if tps_b else None
            vgrow = (vram_c - vram_b) / vram_b * 100.0 if vram_b else 0.0
            rgrow = (rss_c - rss_b) / rss_b * 100.0 if rss_b else 0.0
            comparisons["performance"]["throughput_drop_pct"] = drop
            comparisons["performance"]["vram_growth_pct"] = vgrow
            comparisons["performance"]["rss_growth_pct"] = rgrow
            problems = []
            if drop is not None and drop > self.args.max_throughput_drop_pct:
                problems.append(f"throughput drop {drop:.2f}% > {self.args.max_throughput_drop_pct}%")
            if vgrow > self.args.max_vram_growth_pct:
                problems.append(f"VRAM growth {vgrow:.2f}% > {self.args.max_vram_growth_pct}%")
            if rgrow > self.args.max_rss_growth_pct:
                problems.append(f"RSS growth {rgrow:.2f}% > {self.args.max_rss_growth_pct}%")
            if problems:
                _fail("performance_regression", "; ".join(problems))
            else:
                _pass("performance_regression",
                      {"throughput_drop_pct": drop, "vram_growth_pct": vgrow,
                       "rss_growth_pct": rgrow})
        return self._report(verdicts, reasons, comparisons, synthetic,
                            base_rec, cand_rec,
                            "synthetic" if synthetic else "real")

    def _report(self, verdicts, reasons, comparisons, synthetic,
                base_rec, cand_rec, kind: str) -> dict:
        counts = {"PASS": 0, "FAIL": 0, "NOT_EVALUATED": 0}
        for v in verdicts.values():
            counts[v.get("status", "FAIL")] = counts.get(v.get("status", "FAIL"), 0) + 1
        passed = counts["FAIL"] == 0 and counts["NOT_EVALUATED"] == 0
        eff = "OFF" if (verdicts.get("no_xkv_semantics", {}).get("status") == "PASS") else "not_evaluated"
        report = {
            "off_regression_version": VERSION,
            "off_regression_pass": passed,
            "evidence_kind": kind,
            "xkv_requested_profile": "OFF",
            "xkv_effective_profile": eff,
            "calibration_sha256": (base_rec["manifest"].get("calibration_sha256")
                                   or "not_evaluated"),
            "source_fingerprint": base_rec["manifest"].get("commit") or "not_evaluated",
            "profile_fingerprint": self.settings_fp,
            "verdict_summary": counts,
            "reasons": reasons,
            "verdicts": verdicts,
            "comparisons": comparisons,
            "thresholds": {
                "fit_rel_tol": self.args.fit_rel_tol,
                "fit_abs_tol": self.args.fit_abs_tol,
                "max_throughput_drop_pct": self.args.max_throughput_drop_pct,
                "max_vram_growth_pct": self.args.max_vram_growth_pct,
                "max_rss_growth_pct": self.args.max_rss_growth_pct,
            },
            "manifests": {"baseline": base_rec["manifest"], "candidate": cand_rec["manifest"]},
            "observations": {"baseline": base_rec["observations"],
                             "candidate": cand_rec["observations"]},
        }
        if synthetic:
            report["__synthetic_test_fixture__"] = True
            report["note"] = "SYNTHETIC TEST FIXTURE ONLY - NOT REAL EMPIRICAL EVIDENCE"
        return report


# ---------------------------------------------------------------- self-test

FIXTURE_MAIN = """\
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    beh = json.load(f)
sys.stdout.write(beh.get("stdout", ""))
sys.stderr.write(beh.get("stderr", ""))
sys.exit(beh.get("exit_code", 0))
"""


def _write_fixture(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(FIXTURE_MAIN)
    os.chmod(path, 0o755)


def _behavior(stdout: str = "", stderr: str = "", exit_code: int = 0) -> dict:
    return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}


def _good_log(suffix: str = "", tps: float = 100.0, vram: int = 1000000,
              kv: int = 69376, extra: str = "") -> str:
    return (
        f"{TEXT_LINE}hello world{suffix}\n"
        f"{TOKEN_LINE}1,2,3,4,5\n"
        f"{TPS_LINE}{tps}\n"
        f"{KV_LINE}{kv}\n"
        f"{REC_SLOTS_LINE}6\n"
        f"{REC_PENS_LINE}18\n"
        f"{CACHE_BYTES_LINE}8192\n"
        f"{CACHE_SHA_LINE}abc123\n"
        f"{RSS_LINE}2000000\n"
        f"{VRAM_LINE}{vram}\n"
        f"{PREEMPT_LINE}prefill-batch-limit n=512\n"
        f"{PREEMPT_LINE}recurrent-victim slot=2\n"
        f"{extra}"
    )


def _mkargs(tmp, **kw) -> argparse.Namespace:
    base = dict(
        baseline_binary="", candidate_binary="",
        model=os.path.join(tmp, "m.gguf"), calibration="",
        prompt="hello", prompt_file="", template="t", seed=7,
        temperature=0.0, top_p=1.0, top_k=1, n_predict=32,
        n_ctx=4096, total_kv="", ubatch=0, batch=0, backend="vulkan",
        recurrent_people=0, recurrent_pens=0, ram_swap="",
        extra_arg=[], metrics_baseline="", metrics_candidate="",
        output=os.path.join(tmp, "report.json"), workdir=os.path.join(tmp, "work"),
        timeout=60, resume=False, allow_same_binary_test=False,
        fit_rel_tol=0.0, fit_abs_tol=0, max_throughput_drop_pct=5.0,
        max_vram_growth_pct=5.0, max_rss_growth_pct=5.0,
        self_test=False, schema=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def run_self_test() -> dict:
    checks: dict = {}
    tmp = tempfile.mkdtemp(prefix="offreg-selftest-")
    try:
        with open(os.path.join(tmp, "m.gguf"), "wb") as f:
            f.write(b"fake-model")
        fix_a = os.path.join(tmp, "fake_a.py")
        fix_b = os.path.join(tmp, "fake_b.py")
        _write_fixture(fix_a)
        _write_fixture(fix_b)

        def run_case(name: str, beh_b: dict, beh_c: dict, extra_args: dict,
                     expect_pass: bool, expect_verdict: str,
                     expect_status: str) -> dict:
            d = os.path.join(tmp, name)
            os.makedirs(d, exist_ok=True)
            pb = os.path.join(d, "b.json")
            pc = os.path.join(d, "c.json")
            with open(pb, "w") as f:
                json.dump(beh_b, f)
            with open(pc, "w") as f:
                json.dump(beh_c, f)
            bb = extra_args.get("baseline_binary", fix_a)
            cb = extra_args.get("candidate_binary", fix_b)
            argv_b = [sys.executable, bb, pb]
            argv_c = [sys.executable, cb, pc]

            # Wrap: point the harness at tiny shims whose argv embeds the
            # behavior file, by writing shim scripts.
            def shim(path, argv):
                with open(path, "w") as f:
                    f.write("import sys, subprocess; "
                            f"sys.exit(subprocess.run({argv!r}).returncode)")
                os.chmod(path, 0o755)
            # Simpler: harness builds its own argv from binary; emulate by
            # making the 'binary' a fixture invoked with prompt appended is
            # wrong. Instead drive OffRegression.run_side via monkeypatched
            # argv builder below.
            args = _mkargs(tmp, workdir=d, **{k: v for k, v in extra_args.items()
                                              if k in ("allow_same_binary_test", "model",
                                                       "fit_rel_tol", "fit_abs_tol",
                                                       "max_throughput_drop_pct",
                                                       "max_vram_growth_pct",
                                                       "max_rss_growth_pct")})
            reg = OffRegression(args)
            orig = reg._argv_for

            def patched(binary, prompt):
                if binary in ("BASE", "CAND"):
                    beh = pb if binary == "BASE" else pc
                    fix = fix_a
                    return [sys.executable, fix, beh]
                return orig(binary, prompt)
            reg._argv_for = patched  # type: ignore
            brec = reg.run_side("baseline", "BASE", "hello")
            crec = reg.run_side("candidate", "CAND", "hello")
            if extra_args.get("same_binary"):
                brec["manifest"]["binary_sha256"] = "selftest-same-sha"
                crec["manifest"]["binary_sha256"] = "selftest-same-sha"
            else:
                brec["manifest"]["binary_sha256"] = "selftest-sha-base-" + name
                crec["manifest"]["binary_sha256"] = "selftest-sha-cand-" + name
            rep = reg.compare(brec, crec)
            got = rep["verdicts"].get(expect_verdict, {}).get("status")
            assert got == expect_status, (
                f"{name}: verdict {expect_verdict}={got!r}, want {expect_status!r}; "
                f"reasons={rep['reasons']}")
            assert rep["off_regression_pass"] is expect_pass, (
                f"{name}: pass={rep['off_regression_pass']!r}, want {expect_pass!r}")
            return rep

        rep = run_case("pass",
                       _behavior(stdout=_good_log()), _behavior(stdout=_good_log()),
                       {}, True, "greedy_token_equality", "PASS")
        assert rep["evidence_kind"] == "real"
        checks["pass"] = {"status": "PASS"}

        run_case("token-mismatch",
                 _behavior(stdout=_good_log()),
                 _behavior(stdout=_good_log().replace(f"{TOKEN_LINE}1,2,3,4,5",
                                                     f"{TOKEN_LINE}1,2,3,4,6")),
                 {}, False, "greedy_token_equality", "FAIL")
        checks["token_mismatch"] = {"status": "PASS"}

        run_case("unexpected-xkv",
                 _behavior(stdout=_good_log()),
                 _behavior(stdout=_good_log(extra="xkv_scorer allocated 4096 bytes\n"
                                                  "tri_drain_total: 3\n")),
                 {}, False, "no_xkv_semantics", "FAIL")
        checks["unexpected_xkv"] = {"status": "PASS"}

        run_case("fit-preempt-mismatch",
                 _behavior(stdout=_good_log()),
                 _behavior(stdout=_good_log(kv=100).replace(
                     f"{PREEMPT_LINE}recurrent-victim slot=2",
                     f"{PREEMPT_LINE}active slot preemption slot=0")),
                 {}, False, "kv_recurrent_fit_equivalence", "FAIL")
        checks["fitting_preemption_mismatch"] = {"status": "PASS"}

        run_case("perf-regression",
                 _behavior(stdout=_good_log(tps=100.0, vram=1000000)),
                 _behavior(stdout=_good_log(tps=50.0, vram=2000000)),
                 {}, False, "performance_regression", "FAIL")
        checks["performance_regression"] = {"status": "PASS"}

        # Missing artifact: nonexistent model => every verdict NOT_EVALUATED.
        args = _mkargs(tmp, workdir=os.path.join(tmp, "missing"),
                       model=os.path.join(tmp, "no-such.gguf"))
        reg = OffRegression(args)
        brec = {"manifest": {"binary_sha256": None, "model_sha256": None,
                             "calibration_sha256": None, "commit": "not_evaluated"},
                "observations": {}}
        rep = reg.compare(brec, brec)
        assert rep["verdicts"]["greedy_token_equality"]["status"] == "NOT_EVALUATED", rep["verdicts"]
        assert rep["off_regression_pass"] is False
        checks["missing_artifact"] = {"status": "PASS"}

        # Same binary rejected in production, allowed only as synthetic.
        run_case("same-binary-reject",
                 _behavior(stdout=_good_log()), _behavior(stdout=_good_log()),
                 {"same_binary": True}, False, "binary_identity", "FAIL")
        checks["same_binary_production_rejection"] = {"status": "PASS"}

        rep = run_case("same-binary-synthetic",
                       _behavior(stdout=_good_log()), _behavior(stdout=_good_log()),
                       {"same_binary": True, "allow_same_binary_test": True},
                       True, "binary_identity", "PASS")
        assert rep["evidence_kind"] == "synthetic", rep["evidence_kind"]
        assert rep.get("__synthetic_test_fixture__") is True
        checks["same_binary_synthetic_marked"] = {"status": "PASS"}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return {"status": "PASS", "checks": checks}


# ---------------------------------------------------------------- CLI

SCHEMA_HELP = (
    "Evidence schema (--schema prints EVIDENCE_SCHEMA_JSON): "
    "off_regression_pass true iff every verdict is PASS; verdicts cover "
    "artifact_presence, binary_identity, exit_status, greedy_token_equality, "
    "no_xkv_semantics, kv_recurrent_fit_equivalence, "
    "preemption_trace_equivalence, prompt_cache_equivalence, "
    "performance_regression. Missing observations are NOT_EVALUATED "
    "(fail-closed). Same-binary runs require --allow-same-binary-test and "
    "are marked synthetic (__synthetic_test_fixture__). $schema id "
    "'xkv-evidence/v1' is shared with run-evaluation-matrix.py."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P0 Tri/XKV OFF zero-regression: baseline vs candidate binaries. " + SCHEMA_HELP)
    p.add_argument("--baseline-binary", default="")
    p.add_argument("--candidate-binary", default="")
    p.add_argument("--model", default="")
    p.add_argument("--calibration", default="")
    p.add_argument("--prompt", default="hello")
    p.add_argument("--prompt-file", default="")
    p.add_argument("--template", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--n-predict", type=int, default=512)
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--total-kv", default="")
    p.add_argument("--ubatch", type=int, default=0)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--backend", default="vulkan")
    p.add_argument("--recurrent-people", type=int, default=0)
    p.add_argument("--recurrent-pens", type=int, default=0)
    p.add_argument("--ram-swap", default="")
    p.add_argument("--extra-arg", action="append", default=[],
                   help="Extra argv element applied identically to both sides (repeatable)")
    p.add_argument("--metrics-baseline", default="")
    p.add_argument("--metrics-candidate", default="")
    p.add_argument("--output", default="docs/xkv/off_regression_report.json")
    p.add_argument("--workdir", default="docs/xkv/off_regression_work")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-same-binary-test", action="store_true",
                   help="Allow identical binary hashes; marks evidence synthetic (self-test only)")
    p.add_argument("--fit-rel-tol", type=float, default=0.0)
    p.add_argument("--fit-abs-tol", type=float, default=0)
    p.add_argument("--max-throughput-drop-pct", type=float, default=5.0)
    p.add_argument("--max-vram-growth-pct", type=float, default=5.0)
    p.add_argument("--max-rss-growth-pct", type=float, default=5.0)
    p.add_argument("--self-test", action="store_true")
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
    if not args.baseline_binary or not args.candidate_binary:
        print(json.dumps({
            "off_regression_version": VERSION,
            "off_regression_pass": False,
            "evidence_kind": "real",
            "verdicts": {"artifact_presence": {
                "status": "NOT_EVALUATED",
                "reason": "missing --baseline-binary and/or --candidate-binary"}},
            "verdict_summary": {"PASS": 0, "FAIL": 0, "NOT_EVALUATED": 1},
            "reasons": ["[artifact_presence] NOT_EVALUATED: missing binaries"],
        }, indent=2, sort_keys=True))
        sys.exit(2)
    prompt = args.prompt
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            report = {
                "off_regression_version": VERSION,
                "off_regression_pass": False,
                "evidence_kind": "real",
                "verdicts": {"artifact_presence": {
                    "status": "NOT_EVALUATED",
                    "reason": f"prompt file missing: {args.prompt_file}"}},
                "verdict_summary": {"PASS": 0, "FAIL": 0, "NOT_EVALUATED": 1},
                "reasons": [f"[artifact_presence] NOT_EVALUATED: {args.prompt_file}"],
            }
            atomic_write_json(args.output, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            sys.exit(2)
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    reg = OffRegression(args)
    os.makedirs(args.workdir, exist_ok=True)
    base_rec = reg.run_side("baseline", args.baseline_binary, prompt)
    cand_rec = reg.run_side("candidate", args.candidate_binary, prompt)
    report = reg.compare(base_rec, cand_rec)
    atomic_write_json(args.output, report)
    print(json.dumps({k: report[k] for k in
                      ("off_regression_pass", "evidence_kind", "verdict_summary",
                       "reasons", "verdicts")}, indent=2, sort_keys=True))
    sys.exit(0 if report["off_regression_pass"] else 1)


if __name__ == "__main__":
    main()
