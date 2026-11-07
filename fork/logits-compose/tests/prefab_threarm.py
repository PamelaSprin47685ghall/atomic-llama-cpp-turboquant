#!/usr/bin/env python3
"""Three-arm (stock / stats-gate / prefab-on) verification for the logits collector.

Runs the same teacher-forced draft decode through tq-logits-compose --mtp-ctx
under five environment arms on one machine, then checks the structural claims
of the whole-graph MTP prefab lane against the organic per-context counters
(public llama_prefab_* C API, dumped as JSON by the collector) and the raw
TP5LOG2 tapes.

Arm definitions (each a full process, fresh env each time; the CLI arguments
and model are identical, only the environment differs):
  stock  no prefab env at all
  off    GGML_TP5_PREFAB_STATS=1 only (counters armed, lane off)
  slot   on + LLAMA_PREFAB_MTP=1: whole-graph slot over the exact live n_kv
         (no pad, no compose) -- negative control, build parity with stock
  on     GGML_TP5_PREFAB_STATS=1 + GGML_TP5_PREFAB_AB=1 +
         LLAMA_PREFAB_NKV_PAD=1: production A x B compose over the padded KV
         window -- the only arm with a real reuse win
  dual   on + --dual (two private MTP contexts over one model, interleaved)

Checks (fail-closed, no thresholds): bitwise tape equality across arms,
structural counter identities per arm, and the headline real build-count
decrease (stock builds > on builds).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PREFAB_ENV_KEYS = (
    "GGML_TP5_PREFAB_STATS",
    "LLAMA_PREFAB_MTP",
    "GGML_TP5_PREFAB_AB",
    "GGML_TP5_PREFAB_IN_GRAPH",
    "GGML_TP5_PREFAB_NOCLONE",
    "LLAMA_PREFAB_NKV_PAD",
    "LLAMA_PREFAB_MTP_LIFETIME",
)


def run_arm(binary, name, model, prompt, out_dir, env_extra, extra):
    env = dict(os.environ)
    for key in PREFAB_ENV_KEYS:
        env.pop(key, None)
    env.update(env_extra)
    tape = out_dir / f"{name}.bin"
    cmd = [binary, "--model", model, "--prompt-file", prompt, "--output", str(tape),
           "--mtp-ctx", "--prefab-stats", str(out_dir / f"{name}.json")] + extra
    with (out_dir / f"{name}.stdout").open("w") as so, (out_dir / f"{name}.stderr").open("w") as se:
        proc = subprocess.run(cmd, env=env, stdout=so, stderr=se)
    if proc.returncode != 0:
        raise RuntimeError(f"arm '{name}' exited {proc.returncode}; see {out_dir / (name + '.stderr')}")
    return tape


def parse_summary(out_dir, name):
    with (out_dir / f"{name}.stdout").open() as f:
        for line in f:
            line = line.strip()
            if line.startswith('{"format"'):
                return json.loads(line)
    raise RuntimeError(f"arm '{name}' printed no TP5LOG2 summary")


def parse_stats(out_dir, name):
    with (out_dir / f"{name}.json").open() as f:
        return json.load(f)


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{status} {label}{(' :: ' + detail) if detail else ''}")
    return bool(condition)


def tape_equal(a, b):
    return a.read_bytes() == b.read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", required=True, help="tq-logits-compose binary")
    ap.add_argument("--model", required=True, help="Qwen4Exp GGUF with a nextn/MTP head")
    ap.add_argument("--prompt-file", required=True)
    # --threads defaults to 1 for bitwise cross-arm comparison: on CPU,
    # ggml_compute_forward_flash_attn_ext_f16 (ggml-cpu/ops.cpp:9197) splits KV
    # across threads with chunk_size = ceil(nek1 / nth) when nek1 >= 512, so
    # nth > 1 changes the online-softmax reduction tree between nek1=512 (stock)
    # and nek1=1024 (padded prefab window). With nth=1 every arm accumulates
    # [0, live_n_kv) in one sequential pass and skips -INF pad columns.
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--out-dir", required=True, help="new directory for tapes, stats and logs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    tail = ["--threads", str(args.threads), "--gpu-layers", str(args.gpu_layers)]

    base_env = {"GGML_TP5_PREFAB_STATS": "1"}
    arms = [
        ("stock", {}),
        ("off", base_env),
        ("slot", dict(base_env, LLAMA_PREFAB_MTP="1")),
        ("on", dict(base_env, GGML_TP5_PREFAB_AB="1", LLAMA_PREFAB_NKV_PAD="1")),
        ("dual", dict(base_env, GGML_TP5_PREFAB_AB="1", LLAMA_PREFAB_NKV_PAD="1")),
    ]
    tapes = {}
    for name, env_extra in arms:
        extra = list(tail) + (["--dual"] if name == "dual" else [])
        tapes[name] = run_arm(args.binary, name, args.model, args.prompt_file, out_dir, env_extra, extra)

    stats = {name: parse_stats(out_dir, name) for name, _ in arms}
    summaries = {name: parse_summary(out_dir, name) for name, _ in arms}

    ok = True

    # 1. bitwise tape equality across every arm and both dual contexts
    ok &= check("off tape bitwise == stock tape", tape_equal(tapes["off"], tapes["stock"]))
    ok &= check("slot tape bitwise == stock tape", tape_equal(tapes["slot"], tapes["stock"]))
    ok &= check("on tape bitwise == stock tape", tape_equal(tapes["on"], tapes["stock"]))
    ok &= check("dual tape A bitwise == stock tape", tape_equal(tapes["dual"], tapes["stock"]))
    ok &= check("dual tape B bitwise == stock tape", tape_equal(out_dir / "dual.bin.dual", tapes["stock"]))

    # 2. stock arm (no prefab env): lane never engaged and stats gate is off,
    # so every counter field stays zero (hot path untouched). The prompt must
    # also cross at least one 256-cell stock KV bucket (calls > 256) so the
    # armed stock baseline (`off`) performs >= 2 builds inside the 1024-cell
    # prefab window.
    s_stats = stats["stock"]["contexts"][0]["stats"]
    calls = summaries["stock"]["decode_calls"]
    ok &= check("prompt crosses the 256-cell stock KV bucket (decode_calls > 256)",
                calls > 256, f"decode_calls={calls}")
    ok &= check("stock: lane never engaged", stats["stock"]["contexts"][0]["is_enabled"] is False)
    ok &= check("stock: stats gate off leaves every counter zero",
                all(v == 0 for v in s_stats.values()),
                f"non_zero={[k for k, v in s_stats.items() if v != 0]}")

    # 3. off arm (GGML_TP5_PREFAB_STATS=1 only): the armed stock baseline.
    # Lane never engages, every ubatch runs on the stock path, and crossing the
    # 256-cell KV bucket triggers >= 2 real builds on the stock builder.
    o_stats_raw = stats["off"]["contexts"][0]["stats"]
    ok &= check("off: lane never engaged", stats["off"]["contexts"][0]["is_enabled"] is False)
    ok &= check("off: builds happened on the stock path across KV buckets",
                o_stats_raw["n_stock_builds"] >= 2,
                f"n_stock_builds={o_stats_raw['n_stock_builds']}")
    ok &= check("off: every ubatch ran on the stock path and no envelope built",
                o_stats_raw["n_ubatch_stock"] == calls and o_stats_raw["n_envelope_builds"] == 0,
                f"n_ubatch_stock={o_stats_raw['n_ubatch_stock']} calls={calls}")

    # 4. slot arm (negative control): lane engaged, exact-width slot rebuilt per
    # n_kv growth, so real build parity with stock -- counters measure reality
    l_stats = stats["slot"]["contexts"][0]["stats"]
    ok &= check("slot: lane engaged", stats["slot"]["contexts"][0]["is_enabled"] is True)
    ok &= check("slot: every ubatch reused or rebuilt the slot",
                l_stats["n_envelope_reuses"] + l_stats["n_envelope_allocs"] == calls)
    ok &= check("slot: no stock fallback",
                l_stats["n_ubatch_stock"] == 0 and l_stats["n_mtp_declines"] == 0)
    ok &= check("slot: build parity with stock (no pad window, no compose)",
                l_stats["n_stock_builds"] == o_stats_raw["n_stock_builds"],
                f"slot_builds={l_stats['n_stock_builds']} off_stock_builds={o_stats_raw['n_stock_builds']}")

    # 5. on arm: production compose + padded window; slot reused, no fallback
    o_stats = stats["on"]["contexts"][0]["stats"]
    ok &= check("on: lane engaged", stats["on"]["contexts"][0]["is_enabled"] is True)
    ok &= check("on: every ubatch reused or rebuilt the slot",
                o_stats["n_envelope_reuses"] + o_stats["n_envelope_allocs"] == calls,
                f"reuses={o_stats['n_envelope_reuses']} allocs={o_stats['n_envelope_allocs']} calls={calls}")
    ok &= check("on: no stock fallback",
                o_stats["n_ubatch_stock"] == 0 and o_stats["n_mtp_declines"] == 0)
    ok &= check("on: every rebuild composed (A x B)",
                o_stats["n_mtp_compose_hits"] + o_stats["n_stock_builds"] == o_stats["n_envelope_allocs"])
    ok &= check("on: one A fetch per B emit",
                o_stats["n_mtp_a_compiles"] + o_stats["n_mtp_a_reuses"] == o_stats["n_mtp_b_builds"])
    ok &= check("on: A prologue compiled once (single envelope width)",
                o_stats["n_mtp_a_compiles"] == 1)

    # 6. headline: real build-count decrease (the reason the lane exists)
    ok &= check("on: strictly fewer graph builds than stock",
                o_stats["n_stock_builds"] + o_stats["n_envelope_allocs"] < o_stats_raw["n_stock_builds"],
                f"off_stock_builds={o_stats_raw['n_stock_builds']} "
                f"on_builds={o_stats['n_stock_builds'] + o_stats['n_envelope_allocs']}")

    # 7. dual: two private contexts, each with its own counters, identical logits
    ok &= check("dual: both contexts engaged",
                all(c["is_enabled"] for c in stats["dual"]["contexts"]) and len(stats["dual"]["contexts"]) == 2)
    ok &= check("dual: each context counted its own ubatches",
                stats["dual"]["contexts"][0]["stats"]["n_ubatch_prefab"] == calls
                and stats["dual"]["contexts"][1]["stats"]["n_ubatch_prefab"] == calls)

    print("VERDICT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
