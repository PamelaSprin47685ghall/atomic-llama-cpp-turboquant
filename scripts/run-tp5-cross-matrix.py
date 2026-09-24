#!/usr/bin/env python3
"""run-tp5-cross-matrix.py: Bounded, process-owned clean-room ABBA benchmark harness for TP5.

Guarantees & Invariants:
1. Process lifecycle: owned process group (os.setsid), graceful SIGTERM -> bounded SIGKILL. Never pkill.
2. Verified health of owned PID specifically; fail-closed if process exits or health fails.
3. Clean environment: strict minimal whitelist (PATH, HOME, USER, SHELL, LANG, LC_ALL, LD_LIBRARY_PATH).
   RADV_DEBUG is explicitly configured per variant (default "nobolist"), never silently inherited.
   LD_PRELOAD and unapproved profiling/tracing variables are purged.
4. Exact content check:
   - Full byte-for-byte match against canonical 1..60 counting string (170 chars).
   - finish_reason == "stop".
   - prompt_n == 31.
   - For count-60, both predicted_n and completion_tokens must equal 171, including MTP.
     Longer counting outputs are exploratory and do not enter the 171-token gate.
   - completion_tokens MUST be explicitly reported in usage payload; fail-closed with NO fallback.
5. Timing & Throughput:
   - High-resolution monotonic timing (time.perf_counter) strictly spanning the entire request
     including resp.read().
   - Decode throughput = 171 / (timings.predicted_ms / 1000) for count-60 (primary metric).
   - Client wall throughput is retained as a secondary diagnostic, never the >100 gate.
6. Server log route verification:
   - Parse server log to verify actual route activation:
     * relay vs timeline
     * linear lowering: verifies presence of `[tp5-linear-definition]` in server logs when linear lowering is evaluated.
       Note: collective.cpp line 7020 strictly gates linear lowering on RELAY + F32 (`c.sync_mode == RELAY && c.wire == F32`).
       Therefore linear lowering is evaluated as `relay-f32-linear-lowering` vs `golden-relay-f32`.
     * Never claims route hit from env flag alone without log proof.
7. Mapped DSO hashes: read /proc/<pid>/maps directly from the live running owned process.
8. Git provenance: records HEAD commit, porcelain status (including untracked files), diff SHA256,
   and hash of scripts/run-tp5-cross-matrix.py itself.
9. GPU state: snapshots before/after each trial (DPM table, current clock, VRAM, busy state).
10. Dry-run safety: never fabricates success or measurements; explicitly marks status="DRY_RUN_NOT_MEASURED"
    with empty sample vectors and zero/null ratio.
11. Conservative Student-t paired statistics on actual measured samples using Python stdlib.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_BIN = "/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5/bin/llama-server"
DEFAULT_MODEL = "/home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf"
DEFAULT_MTP_MODEL = "/home/kunweiz/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT_BASE = 8600

# Canonical counting prompt and exact ground-truth answer
CANONICAL_COUNT_STR = ",".join(str(i) for i in range(1, 61))  # 170 characters exactly
EXPECTED_PROMPT_N = 31
EXPECTED_PREDICTED_N_TARGET = 171

PROMPT_PAYLOAD = {
    "messages": [{"role": "user", "content": "Count from 1 to 60. Use commas without spaces. Output only the numbers."}],
    "temperature": 0,
    "seed": 42,
    "max_tokens": 192,
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "stream": False,
}

ENV_WHITELIST_KEYS = ["PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LD_LIBRARY_PATH"]


def get_available_variants(model_path: str, mtp_model_path: str) -> Dict[str, Dict[str, Any]]:
    """Define canonical benchmark matrix variants."""
    base_common_env = {
        "RADV_DEBUG": "nobolist",
        "GGML_TP5_ISOLATE_BO": "1",
        "GGML_TP5_CHAIN_CACHE": "1",
        "GGML_TP5_MERGE_SUBMIT": "1",
        "GGML_TP5_SPIN_MAX": "100000000",
        "GGML_TP5_RELAY_HANDOFF_TIMEOUT_MS": "5000",
        "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1",
        "GGML_VK_ALLOW_GRAPHICS_QUEUE": "1",
        "GGML_VK_CMD_REPLAY": "1",
        "GGML_TP5_REPLICATE_ATTN": "1",
        "GGML_VK_DISABLE_PRODUCER_WIRE": "1",
        "GGML_VK_DISABLE_MMVQ": "1",
    }

    variants: Dict[str, Dict[str, Any]] = {}

    # 1. Golden Baseline: Relay F32, no MTP
    golden_env = dict(base_common_env)
    golden_env.update({"GGML_TP5_WIRE": "f32", "GGML_TP5_SYNC": "relay", "GGML_TP5_RELAY": "on",
                       "GGML_TP5_LINEAR_LOWERING": "0"})
    variants["golden-relay-f32"] = {
        "name": "golden-relay-f32",
        "description": "Golden baseline: relay sync, wire f32, no MTP (historical 52.47/45.5 tok/s)",
        "env": golden_env,
        "cli_args": ["--tp5", "qwen4exp-af", "--tp5-sync", "relay", "--tp5-wire", "f32", "--spec-type", "none"],
        "is_mtp": False,
        "requires_linear": False,
    }

    # 2. Relay F32 + Linear Lowering (Valid linear lowering gate: RELAY + F32 only)
    linear_f32_env = dict(golden_env)
    linear_f32_env["GGML_TP5_LINEAR_LOWERING"] = "1"
    variants["relay-f32-linear-lowering"] = {
        "name": "relay-f32-linear-lowering",
        "description": "Relay F32 with GGML_TP5_LINEAR_LOWERING=1 (lowering gate: RELAY+F32 only)",
        "env": linear_f32_env,
        "cli_args": ["--tp5", "qwen4exp-af", "--tp5-sync", "relay", "--tp5-wire", "f32", "--spec-type", "none"],
        "is_mtp": False,
        "requires_linear": True,
    }
    linear_no_isolation = dict(linear_f32_env)
    linear_no_isolation.pop("GGML_TP5_ISOLATE_BO")
    variants["relay-f32-linear-lowering-isolation-off"] = {
        "name": "relay-f32-linear-lowering-isolation-off",
        "description": "Relay F32 linear lowering with launcher-default Vulkan device policy",
        "env": linear_no_isolation,
        "cli_args": list(variants["relay-f32-linear-lowering"]["cli_args"]),
        "is_mtp": False,
        "requires_linear": True,
    }

    # 3. Relay F16 Control: pure target, no MTP
    relay_f16_env = dict(base_common_env)
    relay_f16_env.update({"GGML_TP5_WIRE": "f16", "GGML_TP5_SYNC": "relay", "GGML_TP5_RELAY": "on"})
    variants["relay-f16-target"] = {
        "name": "relay-f16-target",
        "description": "Relay F16 pure target: sync relay, wire f16, no MTP",
        "env": relay_f16_env,
        "cli_args": ["--tp5", "qwen4exp-af", "--tp5-sync", "relay", "--tp5-wire", "f16", "--spec-type", "none"],
        "is_mtp": False,
        "requires_linear": False,
    }
    variants["native-default"] = {
        "name": "native-default",
        "description": "Only --tp5 qwen4exp-af; no inherited or explicit TP5/MMVQ knob",
        "env": {"RADV_DEBUG": "nobolist"},
        "cli_args": ["--tp5", "qwen4exp-af"],
        "expected_sync": "relay",
        "expected_wire": "f32",
        "is_mtp": False,
        "requires_linear": True,
    }
    variants["native-linear-off"] = {
        **variants["native-default"],
        "name": "native-linear-off",
        "description": "Qualified native F32 route with only GGML_TP5_LINEAR_LOWERING=0",
        "env": {"RADV_DEBUG": "nobolist", "GGML_TP5_LINEAR_LOWERING": "0"},
        "requires_linear": False,
    }
    variants["native-wire-f16"] = {
        **variants["native-default"],
        "name": "native-wire-f16",
        "description": "Qualified native RELAY route with only GGML_TP5_WIRE=f16; different wire arithmetic",
        "env": {"RADV_DEBUG": "nobolist", "GGML_TP5_WIRE": "f16"},
        "expected_wire": "f16",
        "requires_linear": False,
    }
    variants["native-launcher-device"] = {
        **variants["native-default"],
        "name": "native-launcher-device",
        "description": "Native default plus production launcher's graphics queue and device-local VRAM policy",
        "env": {
            "RADV_DEBUG": "nobolist",
            "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1",
            "GGML_VK_ALLOW_GRAPHICS_QUEUE": "1",
        },
    }

    # 4. Control Timeline F16: historical baseline
    timeline_env = dict(base_common_env)
    timeline_env.update({"GGML_TP5_WIRE": "f16", "GGML_TP5_SYNC": "timeline", "GGML_TP5_RELAY": "off"})
    variants["control-timeline-f16"] = {
        "name": "control-timeline-f16",
        "description": "Control baseline: sync timeline, wire f16, relay off (historical --baseline)",
        "env": timeline_env,
        "cli_args": ["--tp5", "qwen4exp-af", "--tp5-sync", "timeline", "--tp5-wire", "f16", "--spec-type", "none"],
        "is_mtp": False,
        "requires_linear": False,
    }

    for base_name in ("relay-f16-target", "control-timeline-f16", "golden-relay-f32"):
        source = variants[base_name]
        stock_env = dict(source["env"])
        stock_env.pop("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM")
        stock_env.pop("GGML_VK_ALLOW_GRAPHICS_QUEUE")
        stock_env.pop("GGML_TP5_ISOLATE_BO")
        name = base_name.replace("-target", "") + "-stock-device"
        variants[name] = {
            "name": name,
            "description": f"{base_name} with Vulkan default BDA/descriptor/queue/memory choices",
            "env": stock_env,
            "cli_args": list(source["cli_args"]),
            "is_mtp": False,
            "requires_linear": False,
        }

        for knob, label in (
            ("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "host-visible-vram-on"),
            ("GGML_VK_ALLOW_GRAPHICS_QUEUE", "compute-queue-only"),
            ("GGML_TP5_ISOLATE_BO", "isolation-off"),
        ):
            ablated_env = dict(source["env"])
            ablated_env.pop(knob)
            ablated_name = f"{base_name}-{label}"
            variants[ablated_name] = {
                "name": ablated_name,
                "description": f"{base_name} with only {knob} unset before device discovery",
                "env": ablated_env,
                "cli_args": list(source["cli_args"]),
                "is_mtp": False,
                "requires_linear": False,
            }

    for name, sync, wire, base in (
        ("star-f16-target", "star", "f16", timeline_env),
        ("timeline-f32-target", "timeline", "f32", timeline_env),
    ):
        chosen = dict(base, GGML_TP5_SYNC=sync, GGML_TP5_WIRE=wire, GGML_TP5_RELAY="off")
        variants[name] = {
            "name": name,
            "description": f"{sync} {wire} pure target",
            "env": chosen,
            "cli_args": ["--tp5", "qwen4exp-af", "--tp5-sync", sync, "--tp5-wire", wire, "--spec-type", "none"],
            "is_mtp": False,
            "requires_linear": False,
        }

    for name, env_key, value, baseline, cli_extra in (
        ("relay-f16-chain-cache-off", "GGML_TP5_CHAIN_CACHE", "0", "relay-f16-target", []),
        ("relay-f16-no-replay", "GGML_VK_CMD_REPLAY", "0", "relay-f16-target", ["--tp5-no-replay"]),
        ("relay-f16-replicate-off", "GGML_TP5_REPLICATE_ATTN", "0", "relay-f16-target", []),
        ("relay-f16-isolate-off", "GGML_TP5_ISOLATE_BO", "0", "relay-f16-target", []),
        ("timeline-f16-merge-off", "GGML_TP5_MERGE_SUBMIT", "0", "control-timeline-f16", []),
        ("timeline-f16-isolate-off", "GGML_TP5_ISOLATE_BO", "0", "control-timeline-f16", []),
    ):
        source = variants[baseline]
        variants[name] = {
            "name": name,
            "description": f"{baseline} with {env_key}={value}",
            "env": dict(source["env"], **{env_key: value}),
            "cli_args": source["cli_args"] + cli_extra,
            "is_mtp": False,
            "requires_linear": False,
        }

    # 5. Relay F16 + MTP (Speculative Decoding)
    variants["relay-f16-mtp"] = {
        "name": "relay-f16-mtp",
        "description": "Relay F16 with draft-mtp (speculative decoding, draft-n-max 3)",
        "env": dict(relay_f16_env),
        "cli_args": [
            "--tp5", "qwen4exp-af",
            "--tp5-sync", "relay",
            "--tp5-wire", "f16",
            "-md", mtp_model_path,
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3",
            "--spec-draft-p-min", "0.0",
        ],
        "is_mtp": True,
        "requires_linear": False,
    }
    variants["native-mtp-single-draft"] = {
        **variants["native-launcher-device"],
        "name": "native-mtp-single-draft",
        "description": "RELAY/F32 five-rank target with local Vulkan0 n=6 MTP draft; unprofiled Decode measurement",
        "env": dict(variants["native-launcher-device"]["env"],
                    GGML_TP5_WIRE="f32", GGML_TP5_LINEAR_LOWERING="1",
                    GGML_TP5_MTP_DRAFT_SINGLE_GPU="1",
                    GGML_TP5_SPIN_MAX="10000000"),
        "cli_args": variants["native-default"]["cli_args"] + [
            "-md", mtp_model_path,
            "--spec-type", "draft-mtp",
            "--spec-draft-device", "Vulkan0",
            "--spec-draft-n-max", "6",
            "--spec-draft-p-min", "0.0",
        ],
        "is_mtp": True,
        "requires_linear": True,
    }
    variants["native-mtp-serial-readback"] = {
        **variants["native-mtp-single-draft"],
        "name": "native-mtp-serial-readback",
        "description": "Same MTP binary with rank-local pinned readback disabled; one-factor W1 control",
        "env": dict(variants["native-mtp-single-draft"]["env"], GGML_META_ASYNC_READBACK="0"),
    }
    variants["native-mtp-materialized-draft"] = {
        **variants["native-mtp-single-draft"],
        "name": "native-mtp-materialized-draft",
        "description": "Same MTP binary with legacy CPU top-k draft candidates; confidence-free selection control",
        "env": dict(variants["native-mtp-single-draft"]["env"], GGML_MTP_DRAFT_RAW_GREEDY="0"),
    }
    variants["native-mtp-bo-isolated"] = {
        **variants["native-mtp-single-draft"],
        "name": "native-mtp-bo-isolated",
        "description": "One-factor RADV nobolist experiment; disables device-wide BDA/indexing/CM2",
        "env": dict(variants["native-mtp-single-draft"]["env"], GGML_TP5_ISOLATE_BO="1"),
    }
    variants["native-mtp-latebind-exact"] = {
        **variants["native-mtp-single-draft"],
        "name": "native-mtp-latebind-exact",
        "description": "Experimental five-rank RELAY/F32 MTP with exact-Q HC LateBind",
        "cli_args": variants["native-mtp-single-draft"]["cli_args"] + ["--tp5-latebind", "exact"],
    }

    # 6. Relay F16 + MMVQ Auto Ablation
    mmvq_env = dict(relay_f16_env)
    mmvq_env["GGML_VK_DISABLE_MMVQ"] = "0"
    variants["relay-f16-mmvq-auto"] = {
        "name": "relay-f16-mmvq-auto",
        "description": "Relay F16 with MMVQ auto (GGML_VK_DISABLE_MMVQ=0)",
        "env": mmvq_env,
        "cli_args": ["--tp5", "qwen4exp-af", "--tp5-sync", "relay", "--tp5-wire", "f16", "--spec-type", "none"],
        "is_mtp": False,
        "requires_linear": False,
    }

    # 7. Relay F16 with Producer Wire enabled
    producer_wire_env = dict(relay_f16_env)
    producer_wire_env["GGML_VK_DISABLE_PRODUCER_WIRE"] = "0"
    variants["relay-f16-wire-producer"] = {
        "name": "relay-f16-wire-producer",
        "description": "Relay F16 with producer wire enabled (GGML_VK_DISABLE_PRODUCER_WIRE=0)",
        "env": producer_wire_env,
        "cli_args": ["--tp5", "qwen4exp-af", "--tp5-sync", "relay", "--tp5-wire", "f16", "--spec-type", "none"],
        "is_mtp": False,
        "requires_linear": False,
    }

    return variants


def check_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def find_free_port(host: str, start_port: int, max_attempts: int = 100) -> int:
    for p in range(start_port, start_port + max_attempts):
        if check_port_free(host, p):
            return p
    raise RuntimeError(f"Could not find available port on {host} in range [{start_port}, {start_port + max_attempts})")


def compute_sha256(file_path: str) -> Optional[str]:
    p = Path(file_path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def get_git_provenance(repo_dir: str) -> Dict[str, Any]:
    prov: Dict[str, Any] = {
        "head_commit": None,
        "is_dirty": False,
        "diff_sha256": None,
        "script_sha256": None,
        "porcelain_status": None,
    }
    script_path = Path(__file__).resolve()
    prov["script_sha256"] = compute_sha256(str(script_path))
    try:
        res = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        prov["head_commit"] = res.stdout.strip()

        st_res = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"], capture_output=True, text=True, check=True)
        prov["porcelain_status"] = st_res.stdout.strip()
        prov["is_dirty"] = bool(st_res.stdout.strip())

        diff_res = subprocess.run(["git", "-C", repo_dir, "diff", "--no-ext-diff", "--binary", "HEAD", "--"], capture_output=True, check=True)
        prov["diff_sha256"] = hashlib.sha256(diff_res.stdout).hexdigest()
    except Exception as e:
        prov["error"] = str(e)
    return prov


def get_mapped_dso_hashes_from_proc(pid: int) -> Dict[str, Optional[str]]:
    mapped_paths: Set[str] = set()
    maps_file = Path(f"/proc/{pid}/maps")
    if not maps_file.is_file():
        return {}

    try:
        with open(maps_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    path = parts[5]
                    if path.startswith("/") and os.path.isfile(path):
                        if any(key in path for key in ["llama", "ggml", "vulkan", "radv"]):
                            mapped_paths.add(path)
    except Exception:
        return {}

    return {p: compute_sha256(p) for p in sorted(mapped_paths)}


def get_gpu_snapshots() -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    card_dirs = sorted(glob.glob("/sys/class/drm/card[0-9]*"))
    for cdir in card_dirs:
        dev_dir = os.path.join(cdir, "device")
        dpm_file = os.path.join(dev_dir, "pp_dpm_sclk")
        if not os.path.exists(dpm_file):
            continue
        try:
            bdf = os.path.basename(os.path.realpath(dev_dir))
            snap: Dict[str, Any] = {
                "card": os.path.basename(cdir),
                "bdf": bdf,
                "pci_inode": os.stat(os.path.realpath(dev_dir)).st_ino,
                "sclk_dpm": [],
                "vram_used": None,
                "gpu_busy_percent": None,
            }
            with open(dpm_file, "r") as f:
                snap["sclk_dpm"] = [line.strip() for line in f.readlines()]

            vram_file = os.path.join(dev_dir, "mem_info_vram_used")
            if os.path.exists(vram_file):
                with open(vram_file, "r") as f:
                    snap["vram_used"] = int(f.read().strip())

            busy_file = os.path.join(dev_dir, "gpu_busy_percent")
            if os.path.exists(busy_file):
                with open(busy_file, "r") as f:
                    snap["gpu_busy_percent"] = int(f.read().strip())

            snapshots.append(snap)
        except Exception:
            continue
    return snapshots


def gpu_identity(snapshots: List[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    return sorted((s["card"], s["bdf"], s["pci_inode"]) for s in snapshots)


class OwnedServerProcess:
    def __init__(self, cmd: List[str], env: Dict[str, str], log_file: Path):
        self.cmd = cmd
        self.env = env
        self.log_file = log_file
        self.proc: Optional[subprocess.Popen] = None
        self.pgid: Optional[int] = None
        self.pid: Optional[int] = None

    def start(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        out_fp = open(self.log_file, "wb")
        self.proc = subprocess.Popen(
            self.cmd,
            env=self.env,
            stdout=out_fp,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            close_fds=True,
        )
        self.pid = self.proc.pid
        self.pgid = os.getpgid(self.pid)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def terminate_gracefully(self, timeout_sec: float = 6.0):
        if self.proc is None or self.proc.poll() is not None:
            return
        pgid = self.pgid
        if pgid is None:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as e:
            print(f"Warning: SIGTERM to pgid {pgid} failed: {e}", file=sys.stderr)

        start = time.perf_counter()
        while time.perf_counter() - start < timeout_sec:
            if self.proc.poll() is not None:
                return
            time.sleep(0.1)

        try:
            print(f"[!] Server pgid={pgid} did not exit in {timeout_sec}s; escalating to SIGKILL", file=sys.stderr)
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=3.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            print(f"Warning: SIGKILL to pgid {pgid} failed: {e}", file=sys.stderr)


def wait_for_server_healthy(host: str, port: int, owned_proc: OwnedServerProcess, timeout_sec: float = 120.0) -> bool:
    url = f"http://{host}:{port}/health"
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_sec:
        if not owned_proc.is_running():
            print(f"[!] Owned server pid={owned_proc.pid} died prematurely (code={owned_proc.proc.returncode if owned_proc.proc else 'unknown'})", file=sys.stderr)
            return False
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def bounded_timeout(deadline: Optional[float], per_phase: float) -> float:
    return per_phase if deadline is None else max(0.0, min(per_phase, deadline - time.perf_counter()))


def verify_server_log_routes(log_path: Path, requires_linear: bool, expected_sync: str, expected_wire: str,
                             expected_mmvq: Optional[str]) -> Dict[str, Any]:
    """Inspect server log for physical route / linear lowering hits."""
    res = {
        "log_exists": log_path.is_file(),
        "linear_definition_hit": False,
        "route_verified": False,
        "mmvq_policy_seen": False,
        "mmvq_policy_unobservable": expected_mmvq is None,
        "error": None,
    }
    if not log_path.is_file():
        res["error"] = "Log file does not exist"
        return res

    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if "[tp5-linear-definition]" in line:
                    res["linear_definition_hit"] = True
                if "ggml-vulkan-collective: init " in line and f"wire={expected_wire} sync={expected_sync}" in line:
                    res["route_verified"] = True
                if expected_mmvq is not None and "[vulkan-mmvq-policy] " in line:
                    if f"mode={expected_mmvq}" not in line:
                        res["error"] = f"Unexpected MMVQ device policy (expected {expected_mmvq}): {line.strip()}"
                    else:
                        res["mmvq_policy_seen"] = True

        if res["error"]:
            return res
        if not res["route_verified"]:
            res["error"] = f"Missing observed collective init wire={expected_wire} sync={expected_sync}"
        elif expected_mmvq is not None and not res["mmvq_policy_seen"]:
            res["error"] = f"Missing effective MMVQ policy={expected_mmvq} device log"
        elif requires_linear and not res["linear_definition_hit"]:
            res["error"] = "Variant requires linear lowering but '[tp5-linear-definition]' was never logged by server."
        elif not requires_linear and res["linear_definition_hit"]:
            res["error"] = "Control variant unexpectedly hit '[tp5-linear-definition]'"
    except Exception as e:
        res["error"] = str(e)
    return res


def execute_bench_request(host: str, port: int, owned_proc: OwnedServerProcess, is_mtp: bool,
                          count_to: int, timeout_sec: float = 120.0) -> Dict[str, Any]:
    """Execute canonical count request with monotonic timing including read, and strict validation.
    
    For count-60, both MTP and pure Target must report exactly 171 predicted
    and committed tokens, positive finite predicted_ms, and naturally stopped
    exact content. Missing usage.completion_tokens always fails closed.
    """
    if timeout_sec <= 0:
        return {"success": False, "error_message": "Matrix wall-clock deadline exceeded before HTTP request"}
    if not owned_proc.is_running():
        return {
            "success": False,
            "error_message": f"Server process died before/during request (code={owned_proc.proc.returncode if owned_proc.proc else 'unknown'})",
        }

    url = f"http://{host}:{port}/v1/chat/completions"
    payload = dict(PROMPT_PAYLOAD)
    if count_to != 60:
        payload["messages"] = [{"role": "user", "content":
            f"Count from 1 to {count_to}. Use commas without spaces. Output only the numbers."}]
        payload["max_tokens"] = max(192, 4 * count_to)
    expected_content = CANONICAL_COUNT_STR if count_to == 60 else ",".join(str(i) for i in range(1, count_to + 1))
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    def expire_request(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"HTTP total response deadline exceeded ({timeout_sec:.1f}s)")

    t0 = time.perf_counter()
    previous_alarm = signal.signal(signal.SIGALRM, expire_request)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status_code = resp.status
            body_bytes = resp.read()
            # Monotonic time measured strictly AFTER full response payload is received
            elapsed = time.perf_counter() - t0

            if status_code != 200:
                return {
                    "success": False,
                    "status_code": status_code,
                    "client_wall_sec": elapsed,
                    "request": payload,
                    "raw_response": body_bytes.decode("utf-8", errors="replace"),
                    "error_message": f"Non-200 HTTP status: {status_code}",
                }

            body = json.loads(body_bytes.decode("utf-8"))

            timings = body.get("timings", {})
            pred_tps = float(timings.get("predicted_per_second", 0.0))
            pred_n = int(timings.get("predicted_n", 0))
            pred_ms = float(timings.get("predicted_ms", 0.0))
            prompt_n = int(timings.get("prompt_n", 0))

            usage = body.get("usage", {})
            if "completion_tokens" not in usage:
                # Inviolable constraint: fail closed if completion_tokens is absent (NO fallback to predicted_n)
                return {
                    "success": False,
                    "status_code": status_code,
                    "client_wall_sec": elapsed,
                    "request": payload,
                    "raw_response": body,
                    "error_message": "Missing 'completion_tokens' in usage payload; fail-closed.",
                }
            completion_tokens = int(usage["completion_tokens"])

            choices = body.get("choices", [{}])
            choice = choices[0] if choices else {}
            finish_reason = str(choice.get("finish_reason", ""))
            message = choice.get("message", {})
            content = str(message.get("content", ""))

            exact_content_match = (content == expected_content)

            # The fixed prompt_n/predicted_n oracle is specific to count-60.
            # Longer shapes still require the entire exact response and stop.
            pred_n_valid = pred_n == EXPECTED_PREDICTED_N_TARGET if count_to == 60 else pred_n > 0
            completion_n_valid = completion_tokens == EXPECTED_PREDICTED_N_TARGET if count_to == 60 else completion_tokens > 0
            prompt_n_valid = prompt_n == EXPECTED_PROMPT_N if count_to == 60 else prompt_n > 0

            is_valid = (
                exact_content_match
                and finish_reason == "stop"
                and prompt_n_valid
                and pred_n_valid
                and completion_n_valid
                and math.isfinite(pred_ms) and pred_ms > 0
            )

            committed_tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
            decode_tps = completion_tokens * 1000.0 / pred_ms if math.isfinite(pred_ms) and pred_ms > 0 else 0.0

            err = None
            if not is_valid:
                err = (
                    f"Validation failed: exact_content_match={exact_content_match} (len={len(content)} vs {len(expected_content)}), "
                    f"finish_reason={finish_reason}, prompt_n={prompt_n}, pred_n={pred_n} (valid={pred_n_valid}, is_mtp={is_mtp}), "
                    f"completion_tokens={completion_tokens} (valid={completion_n_valid}), predicted_ms={pred_ms}"
                )

            return {
                "success": is_valid,
                "status_code": status_code,
                "client_wall_sec": elapsed,
                "reported_predicted_tps": pred_tps,
                "decode_tps": decode_tps,
                "predicted_ms": pred_ms,
                "prompt_n": prompt_n,
                "predicted_n": pred_n,
                "completion_tokens": completion_tokens,
                "committed_tps": committed_tps,
                "finish_reason": finish_reason,
                "exact_content_match": exact_content_match,
                "request": payload,
                "raw_response": body,
                "error_message": err,
            }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "status_code": getattr(e, "code", 0),
            "client_wall_sec": elapsed,
            "reported_predicted_tps": 0.0,
            "decode_tps": 0.0,
            "predicted_ms": 0.0,
            "prompt_n": 0,
            "predicted_n": 0,
            "completion_tokens": 0,
            "committed_tps": 0.0,
            "finish_reason": "exception",
            "exact_content_match": False,
            "request": payload,
            "raw_response": {},
            "error_message": f"HTTP request exception: {str(e)}",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)


def build_clean_environment(overrides: Dict[str, str], base_binary_dir: str) -> Dict[str, str]:
    clean_env: Dict[str, str] = {}
    for k in ENV_WHITELIST_KEYS:
        if k in os.environ:
            clean_env[k] = os.environ[k]

    current_ld = clean_env.get("LD_LIBRARY_PATH", "")
    clean_env["LD_LIBRARY_PATH"] = f"{base_binary_dir}:{current_ld}" if current_ld else base_binary_dir

    clean_env.update(overrides)
    return clean_env


def compute_paired_statistics(samples_a: List[float], samples_b: List[float]) -> Dict[str, Any]:
    n = len(samples_a)
    if n != len(samples_b) or n < 2:
        return {"error": f"Need at least 2 paired samples, got {n}"}

    mean_a = statistics.mean(samples_a)
    mean_b = statistics.mean(samples_b)
    stdev_a = statistics.stdev(samples_a) if n > 1 else 0.0
    stdev_b = statistics.stdev(samples_b) if n > 1 else 0.0

    diffs = [b - a for a, b in zip(samples_a, samples_b)]
    mean_diff = statistics.mean(diffs)
    stdev_diff = statistics.stdev(diffs) if n > 1 else 0.0
    se_diff = stdev_diff / math.sqrt(n)

    ratios = [(b / a) for a, b in zip(samples_a, samples_b) if a > 0]
    mean_ratio = statistics.mean(ratios) if ratios else 0.0
    stdev_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
    se_ratio = stdev_ratio / math.sqrt(len(ratios)) if ratios else 0.0

    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        20: 2.086, 25: 2.060, 30: 2.042,
    }
    df = n - 1
    t_val = t_table.get(df, 1.96 if df > 30 else 2.571)

    return {
        "n_pairs": n,
        "mean_a": mean_a,
        "stdev_a": stdev_a,
        "mean_b": mean_b,
        "stdev_b": stdev_b,
        "mean_diff_b_minus_a": mean_diff,
        "ci_95_diff": [mean_diff - t_val * se_diff, mean_diff + t_val * se_diff],
        "mean_ratio_b_over_a": mean_ratio,
        "ci_95_ratio": [mean_ratio - t_val * se_ratio, mean_ratio + t_val * se_ratio],
        "noninferiority_0_99_met": (mean_ratio - t_val * se_ratio) >= 0.99,
    }


def run_single_variant_trial(
    variant: Dict[str, Any],
    bin_path: str,
    model_path: str,
    host: str,
    port: int,
    log_dir: Path,
    trial_tag: str,
    dry_run: bool = False,
    repeats: int = 1,
    count_to: int = 60,
    legacy_mmvq_policy_unobservable: bool = False,
    deadline: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "variant": variant["name"],
        "trial_tag": trial_tag,
        "count_to": count_to,
        "port": port,
        "is_mtp": variant.get("is_mtp", False),
        "requires_linear": variant.get("requires_linear", False),
        "metrics": [],
        "mapped_dso_hashes": {},
        "gpu_snapshot_before": [],
        "gpu_snapshot_after": [],
        "log_route_verification": {},
        "success": False,
        "log_path": None,
        "log_sha256": None,
    }

    if dry_run:
        print(f"[DRY-RUN] Simulated trial for '{variant['name']}' [{trial_tag}] on port {port}. NOT_MEASURED.")
        result["dry_run"] = True
        result["status"] = "NOT_MEASURED"
        result["success"] = False  # Never claim success or fabricate samples in dry run
        return result

    if deadline is not None and time.perf_counter() >= deadline:
        result["error"] = "Matrix wall-clock deadline exceeded before starting trial"
        return result
    if not check_port_free(host, port):
        result["error"] = f"Port {port} is occupied on {host}; fail-closed."
        return result

    bin_dir = str(Path(bin_path).parent.resolve())
    env = build_clean_environment(variant["env"], bin_dir)

    cmd = [
        bin_path,
        "-m", model_path,
        "-dev", "Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4",
        "--split-mode", "tensor",
        "--fit", "off",
        "-ngl", "999",
        "-c", str(256 if count_to == 60 else 768),
        "-b", "32",
        "-ub", "32",
        "--no-mmap",
        "--no-host",
        "-np", "1",
        "--host", host,
        "--port", str(port),
    ]
    cmd.extend(variant["cli_args"])

    log_file = log_dir / f"server_{variant['name']}_{trial_tag}.log"
    result["log_path"] = str(log_file)
    result["command"] = cmd
    result["env_snapshot"] = {k: v for k, v in env.items() if not k.startswith("_")}

    result["gpu_snapshot_before"] = get_gpu_snapshots()
    if len(gpu_identity(result["gpu_snapshot_before"])) != 5:
        result["error"] = "Five GPU identities were not readable before server startup"
        return result

    server = OwnedServerProcess(cmd, env, log_file)
    try:
        server.start()
        print(f"[*] Started owned server pid={server.pid} pgid={server.pgid} for {variant['name']} [{trial_tag}] on port {port}")

        if not wait_for_server_healthy(host, port, owned_proc=server,
                                       timeout_sec=bounded_timeout(deadline, 120.0)):
            result["error"] = "Server failed health check or died within deadline"
            print(f"[!] {result['error']}", file=sys.stderr)
            return result

        # Read mapped DSO hashes directly from /proc/<pid>/maps while server is healthy
        if server.pid:
            result["mapped_dso_hashes"] = get_mapped_dso_hashes_from_proc(server.pid)

        # Warmup request
        print(f"[*] Running warmup request for {variant['name']} [{trial_tag}]...")
        warmup = execute_bench_request(host, port, owned_proc=server, is_mtp=variant.get("is_mtp", False),
                                       count_to=count_to, timeout_sec=bounded_timeout(deadline, 120.0))
        result["warmup"] = warmup
        if deadline is not None and time.perf_counter() >= deadline:
            result["error"] = "Matrix wall-clock deadline exceeded during warmup"
            return result
        if not warmup["success"]:
            result["error"] = f"Warmup request failed: {warmup.get('error_message')}"
            print(f"[!] {result['error']}", file=sys.stderr)
            return result

        print(f"[+] Warmup passed (decode_tps={warmup['decode_tps']:.2f}, client_wall_tps={warmup['committed_tps']:.2f})")

        # Measured repeats
        collected: List[Dict[str, Any]] = []
        result["metrics"] = collected
        for r in range(repeats):
            time.sleep(0.5)
            metric = execute_bench_request(host, port, owned_proc=server, is_mtp=variant.get("is_mtp", False),
                                           count_to=count_to, timeout_sec=bounded_timeout(deadline, 120.0))
            if deadline is not None and time.perf_counter() >= deadline:
                result["error"] = f"Matrix wall-clock deadline exceeded during repeat {r+1}"
                return result
            if not metric["success"]:
                result["error"] = f"Repeat {r+1}/{repeats} failed: {metric.get('error_message')}"
                print(f"[!] {result['error']}", file=sys.stderr)
                return result
            print(f"    Repeat {r+1}/{repeats}: decode_tps={metric['decode_tps']:.2f}, client_wall_tps={metric['committed_tps']:.2f}, wall={metric['client_wall_sec']:.3f}s")
            collected.append(metric)

        # Post-flight: verify route logs for linear lowering / relay
        route_ver = verify_server_log_routes(
            log_file, variant.get("requires_linear", False),
            variant.get("expected_sync", variant["env"].get("GGML_TP5_SYNC", "")),
            variant.get("expected_wire", variant["env"].get("GGML_TP5_WIRE", "")),
            None if legacy_mmvq_policy_unobservable else variant.get(
                "expected_mmvq", "auto" if variant["env"].get("GGML_VK_DISABLE_MMVQ") == "0" else "disabled"))
        result["log_route_verification"] = route_ver
        if route_ver["error"]:
            result["error"] = f"Log route verification failed: {route_ver.get('error')}"
            print(f"[!] {result['error']}", file=sys.stderr)
            return result

        result["success"] = True
        return result
    finally:
        server.terminate_gracefully()
        result["log_sha256"] = compute_sha256(str(log_file))
        result["gpu_snapshot_after"] = get_gpu_snapshots()
        if gpu_identity(result["gpu_snapshot_before"]) != gpu_identity(result["gpu_snapshot_after"]):
            result["success"] = False
            result["error"] = "GPU PCI device identity changed during trial; discard measurements after recovery"
        if deadline is not None and time.perf_counter() >= deadline:
            result["success"] = False
            result["error"] = "Matrix wall-clock deadline exceeded before trial teardown completed"


def run_abba_cross_matrix(
    variant_a_name: str,
    variant_b_name: str,
    blocks: int,
    bin_path: str,
    model_path: str,
    mtp_model_path: str,
    host: str,
    base_port: int,
    log_dir: Path,
    dry_run: bool,
    repeats: int,
    count_to: int,
    bin_path_b: Optional[str] = None,
    legacy_mmvq_policy_unobservable: bool = False,
    max_wall_seconds: float = 1800.0,
    pilot: bool = False,
) -> Dict[str, Any]:
    if blocks < 5 and not (dry_run or pilot):
        raise ValueError(f"formal hardware comparison requires at least 5 complete ABBA blocks; use --pilot for {blocks} exploratory blocks")
    if blocks <= 0:
        raise ValueError(f"blocks must be positive, got {blocks}")
    if repeats <= 0:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if count_to not in (60, 100):
        raise ValueError(f"only audited count lengths 60 or 100 are accepted, got {count_to}")

    all_variants = get_available_variants(model_path, mtp_model_path)
    if variant_a_name not in all_variants:
        raise ValueError(f"Variant A '{variant_a_name}' not recognized. Available: {list(all_variants.keys())}")
    if variant_b_name not in all_variants:
        raise ValueError(f"Variant B '{variant_b_name}' not recognized. Available: {list(all_variants.keys())}")

    # A lowering-factor experiment must keep sync and F32 math fixed.
    if variant_b_name == "relay-f32-linear-lowering" and variant_a_name != "golden-relay-f32":
        raise ValueError("relay-f32-linear-lowering requires the matched golden-relay-f32 control")

    var_a = all_variants[variant_a_name]
    var_b = all_variants[variant_b_name]

    repo_dir = str(Path(__file__).parent.parent.resolve())
    git_prov = get_git_provenance(repo_dir)

    results_data: Dict[str, Any] = {
        "variant_a": var_a,
        "variant_b": var_b,
        "server_binary": str(Path(bin_path).resolve()),
        "server_binary_sha256": compute_sha256(bin_path),
        "server_binary_b": str(Path(bin_path_b or bin_path).resolve()),
        "server_binary_b_sha256": compute_sha256(bin_path_b or bin_path),
        "legacy_mmvq_policy_unobservable": legacy_mmvq_policy_unobservable,
        "blocks_requested": blocks,
        "pilot": pilot,
        "max_wall_seconds": max_wall_seconds,
        "count_to": count_to,
        "blocks_completed": 0,
        "repeats_per_trial": repeats,
        "git_provenance": git_prov,
        "trials": [],
        "paired_samples_a_committed": [],
        "paired_samples_b_committed": [],
        "paired_samples_a_decode": [],
        "paired_samples_b_decode": [],
        "stats_committed_tps": {},
        "stats_decode_tps": {},
        "status": "in_progress",
    }

    if dry_run:
        print(f"\n================================================================================")
        print(f"[*] Starting DRY RUN ABBA Matrix Check (NOT_MEASURED, no fake success)")
        print(f"    Variant A: {var_a['name']}")
        print(f"    Variant B: {var_b['name']}")
        print(f"    Blocks: {blocks}")
        print(f"================================================================================\n")
        results_data["status"] = "DRY_RUN_NOT_MEASURED"
        for block_idx in range(blocks):
            for slot_tag, variant in [("A1", var_a), ("B1", var_b), ("B2", var_b), ("A2", var_a)]:
                results_data["trials"].append({
                    "variant": variant["name"],
                    "trial_tag": f"blk{block_idx+1}_{slot_tag}",
                    "server_binary": str(Path(bin_path if slot_tag.startswith("A") else bin_path_b or bin_path).resolve()),
                    "status": "NOT_MEASURED",
                    "success": False,
                })
        return results_data

    print(f"\n================================================================================")
    print(f"[*] Starting Clean-Room ABBA Matrix Benchmark")
    print(f"    Variant A: {var_a['name']} ({var_a['description']})")
    print(f"    Variant B: {var_b['name']} ({var_b['description']})")
    print(f"    Blocks: {blocks} (total {blocks * 4} trials in ABBA order)")
    print(f"    Repeats per trial: {repeats}")
    print(f"================================================================================\n")

    port = base_port

    samples_a_dec: List[float] = []
    samples_b_dec: List[float] = []
    samples_a_com: List[float] = []
    samples_b_com: List[float] = []
    baseline_gpu_identity: Optional[List[Tuple[str, str, int]]] = None
    deadline = time.perf_counter() + max_wall_seconds

    for block_idx in range(blocks):
        print(f"\n>>> Starting Block {block_idx + 1}/{blocks} (ABBA sequence: A1, B1, B2, A2)")
        sequence: List[Tuple[str, Dict[str, Any]]] = [
            ("A1", var_a),
            ("B1", var_b),
            ("B2", var_b),
            ("A2", var_a),
        ]

        block_results: Dict[str, Any] = {}

        for slot_tag, variant in sequence:
            trial_tag = f"blk{block_idx+1}_{slot_tag}"
            trial_port = find_free_port(host, port)
            trial_res = run_single_variant_trial(
                variant=variant,
                bin_path=bin_path if slot_tag.startswith("A") else bin_path_b or bin_path,
                model_path=model_path,
                host=host,
                port=trial_port,
                log_dir=log_dir,
                trial_tag=trial_tag,
                dry_run=dry_run,
                repeats=repeats,
                count_to=count_to,
                legacy_mmvq_policy_unobservable=legacy_mmvq_policy_unobservable,
                deadline=deadline,
            )
            results_data["trials"].append(trial_res)

            current_gpu_identity = gpu_identity(trial_res["gpu_snapshot_before"])
            if baseline_gpu_identity is None:
                baseline_gpu_identity = current_gpu_identity
                results_data["baseline_gpu_identity"] = current_gpu_identity
            elif current_gpu_identity != baseline_gpu_identity:
                trial_res["success"] = False
                trial_res["error"] = "GPU PCI device identity changed between trials; discard cross-reset ABBA block"

            if not trial_res["success"]:
                results_data["status"] = "failed"
                results_data["failure_reason"] = f"Trial {trial_tag} ({variant['name']}) failed closed: {trial_res.get('error')}"
                print(f"\n[FATAL] {results_data['failure_reason']}", file=sys.stderr)
                return results_data

            block_results[slot_tag] = trial_res

        # The four trials are one independent ABBA block, not two independent
        # samples. Collapsing within a block avoids overstating CI precision.
        a_metrics = block_results["A1"]["metrics"] + block_results["A2"]["metrics"]
        b_metrics = block_results["B1"]["metrics"] + block_results["B2"]["metrics"]
        samples_a_com.append(statistics.mean(m["committed_tps"] for m in a_metrics))
        samples_b_com.append(statistics.mean(m["committed_tps"] for m in b_metrics))
        samples_a_dec.append(statistics.mean(m["decode_tps"] for m in a_metrics))
        samples_b_dec.append(statistics.mean(m["decode_tps"] for m in b_metrics))

        results_data["blocks_completed"] += 1

    results_data["paired_samples_a_committed"] = samples_a_com
    results_data["paired_samples_b_committed"] = samples_b_com
    results_data["paired_samples_a_decode"] = samples_a_dec
    results_data["paired_samples_b_decode"] = samples_b_dec

    if pilot:
        # A short directed experiment reports observations, not a confidence
        # interval or a default-promotion decision.
        results_data["status"] = "PILOT_COMPLETE_NOT_ACCEPTANCE"
    else:
        results_data["stats_committed_tps"] = compute_paired_statistics(samples_a_com, samples_b_com)
        results_data["stats_decode_tps"] = compute_paired_statistics(samples_a_dec, samples_b_dec)
        results_data["status"] = "success"

    return results_data


def print_summary_table(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY (Status: {results.get('status')})")
    print("=" * 80)

    if results.get("status") == "DRY_RUN_NOT_MEASURED":
        print("Dry run completed. Process flow and arguments validated. NO MEASUREMENTS TAKEN.")
        print(f"Requested blocks: {results.get('blocks_requested')}")
        print("=" * 80)
        return

    if results.get("status") == "SMOKE_PASS":
        print("Route and exact-response smoke passed. NO PAIRED PERFORMANCE WIN CLAIM.")
        if results.get("count_to") == 60:
            for metric in results["trials"][0]["metrics"]:
                print(f"171-token Decode: {metric['decode_tps']:.2f} tok/s (predicted_ms={metric['predicted_ms']:.3f})")
        print("=" * 80)
        return

    if results.get("status") == "PILOT_COMPLETE_NOT_ACCEPTANCE":
        a = results["paired_samples_a_decode"]
        b = results["paired_samples_b_decode"]
        print(f"Exploratory {results['blocks_completed']}-block ABBA; Decode A={statistics.mean(a):.2f}, B={statistics.mean(b):.2f} tok/s")
        print("No confidence interval or default-promotion claim from reduced rounds.")
        print("=" * 80)
        return

    if results.get("status") != "success":
        print(f"Benchmark Failed: {results.get('failure_reason')}")
        print("=" * 80)
        return

    var_a = results["variant_a"]["name"]
    var_b = results["variant_b"]["name"]

    com = results["stats_committed_tps"]
    dec = results["stats_decode_tps"]

    print(f"Completed Blocks: {results['blocks_completed']} (Total {len(results['trials'])} isolated process runs)")
    print(f"Variant A: {var_a}")
    print(f"Variant B: {var_b}")
    print("-" * 80)
    if results.get("count_to", 60) == 60:
        print("171-TOKEN DECODE THROUGHPUT (171 / timings.predicted_ms) [PRIMARY METRIC]:")
    else:
        print("LONG-OUTPUT DECODE THROUGHPUT (not the 171-token acceptance gate):")
    print(f"  {var_a}: Mean = {dec['mean_a']:.2f} tok/s (stdev: {dec['stdev_a']:.2f})")
    print(f"  {var_b}: Mean = {dec['mean_b']:.2f} tok/s (stdev: {dec['stdev_b']:.2f})")
    print(f"  Difference (B - A): {dec['mean_diff_b_minus_a']:+.2f} tok/s [95% CI: {dec['ci_95_diff'][0]:.2f}, {dec['ci_95_diff'][1]:.2f}]")
    print(f"  Paired Ratio (B / A): {dec['mean_ratio_b_over_a']:.4f} [95% CI: {dec['ci_95_ratio'][0]:.4f}, {dec['ci_95_ratio'][1]:.4f}]")
    print(f"  Non-inferiority (Lower CI >= 0.99): {dec['noninferiority_0_99_met']}")
    print("-" * 80)
    print(f"CLIENT WALL THROUGHPUT (secondary diagnostic): {var_a}={com['mean_a']:.2f}, {var_b}={com['mean_b']:.2f} tok/s")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Safe, bounded, process-owned ABBA cross-matrix benchmark harness for TP5."
    )
    parser.add_argument("--variant-a", "-a", default="golden-relay-f32", help="Variant A name (baseline/control)")
    parser.add_argument("--variant-b", "-b", default="relay-f16-target", help="Variant B name (treatment/candidate)")
    parser.add_argument("--blocks", "-k", type=int, default=5, help="Number of ABBA blocks (default: 5, yielding 20 trials)")
    parser.add_argument("--pilot", action="store_true", help="Allow fewer than 5 ABBA blocks for a quick exploratory comparison, without an acceptance claim")
    parser.add_argument("--repeats", "-r", type=int, default=1, help="Number of request repeats per trial (default: 1)")
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0,
                        help="Maximum matrix wall time in seconds (default: 1800; each request is also capped at 120)")
    parser.add_argument("--count-to", type=int, choices=(60, 100), default=60,
                        help="Exact counting workload: 60 uses the fixed 171-token gate; 100 tests longer decode")
    parser.add_argument("--bin", default=DEFAULT_BIN, help="Path to llama-server binary")
    parser.add_argument("--bin-b", help="Optional server binary for B trials (e.g. isolated historical factor)")
    parser.add_argument("--legacy-mmvq-policy-unobservable", action="store_true",
                        help="Old binaries lack the MMVQ policy marker; record this gap instead of silently requiring it")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model GGUF")
    parser.add_argument("--mtp-model", default=DEFAULT_MTP_MODEL, help="Path to MTP draft model GGUF")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT_BASE, help="Base port for search")
    parser.add_argument("--log-dir", help="Directory for server logs (default: unique alongside the output JSON)")
    parser.add_argument("--output", "-o", help="Save complete results JSON to path")
    parser.add_argument("--dry-run", action="store_true", help="Simulate process flow and arguments without GPU execution (marks NOT_MEASURED)")
    parser.add_argument("--smoke", action="store_true", help="One owned trial for route/content preflight; never estimates a performance win")
    parser.add_argument("--list-variants", action="store_true", help="List all defined matrix variants and exit")

    args = parser.parse_args()

    if args.list_variants:
        variants = get_available_variants(args.model, args.mtp_model)
        print("Available Matrix Variants:")
        for name, spec in variants.items():
            print(f"  {name:26s} : {spec['description']}")
        sys.exit(0)

    if args.blocks <= 0:
        parser.error(f"--blocks must be >= 1, got {args.blocks}")
    if args.repeats <= 0:
        parser.error(f"--repeats must be >= 1, got {args.repeats}")
    if args.max_wall_seconds <= 0:
        parser.error(f"--max-wall-seconds must be positive, got {args.max_wall_seconds}")
    if args.smoke and args.dry_run:
        parser.error("--smoke and --dry-run are mutually exclusive")

    log_dir = Path(args.log_dir) if args.log_dir else (
        Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_logs") if args.output else
        Path("/tmp/tp5_cross_matrix_logs") / f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}")
    if args.smoke:
        variants = get_available_variants(args.model, args.mtp_model)
        if args.variant_a not in variants:
            parser.error(f"unknown smoke variant {args.variant_a}")
        log_dir.mkdir(parents=True, exist_ok=True)
        trial = run_single_variant_trial(variants[args.variant_a], args.bin, args.model, args.host,
                                         find_free_port(args.host, args.port), log_dir,
                                         f"smoke_{args.count_to}", repeats=args.repeats, count_to=args.count_to,
                                         legacy_mmvq_policy_unobservable=args.legacy_mmvq_policy_unobservable,
                                         deadline=time.perf_counter() + args.max_wall_seconds)
        results = {
            "status": "SMOKE_PASS" if trial["success"] else "failed",
            "failure_reason": trial.get("error"), "count_to": args.count_to,
            "git_provenance": get_git_provenance(str(Path(__file__).parent.parent.resolve())),
            "server_binary_sha256": compute_sha256(args.bin), "trials": [trial],
        }
    else:
        results = run_abba_cross_matrix(
            variant_a_name=args.variant_a, variant_b_name=args.variant_b, blocks=args.blocks,
            bin_path=args.bin, model_path=args.model, mtp_model_path=args.mtp_model,
            host=args.host, base_port=args.port, log_dir=log_dir, dry_run=args.dry_run,
            repeats=args.repeats, count_to=args.count_to, bin_path_b=args.bin_b,
            legacy_mmvq_policy_unobservable=args.legacy_mmvq_policy_unobservable,
            max_wall_seconds=args.max_wall_seconds,
            pilot=args.pilot,
        )

    print_summary_table(results)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Detailed output saved to: {args.output}")

    if results.get("status") not in ("success", "DRY_RUN_NOT_MEASURED", "SMOKE_PASS", "PILOT_COMPLETE_NOT_ACCEPTANCE"):
        sys.exit(1)


if __name__ == "__main__":
    main()
