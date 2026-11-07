#!/usr/bin/env python3
"""Check one padmax_cpu_probe stderr log against the runtime-length oracle.

Usage: check_probe.py LOG MODE ARM
  MODE: nkv | cap | seq
  ARM : unset | off | on

The oracle is the contract of the runtime-length lane:
  * off   : explicit gate=0, stock widths and capacities.
  * unset : qualified CPU contexts match on; multi-stream contexts match off.
  * on    : GGML_TP5_UNION_BINDER=1. One envelope build at the requested
            capacity; every true used width binds in place. There is no
            padded control arm: the exact width is the only width.

The capacity values are hard gates: a hidden pad or headroom fails the check
even when the logits happen to match.

The ARM line is the probe's own report of the environment it was started
with (`ARM binder=<unset|0|1> ntok=... reuse_disable=...`). The probe reads
those variables once and never writes them, so the line is the arm the runner
injected; this checker refuses an arm whose line names a different binder
value. Every successful step must also print one NKV_EXACT line whose
mask_ne0, k_ne2 and v_nkv all equal the 1-based position after the step: the
three n_kv runtime roots must be observed AND execute at the used span.

The NKV_EXACT line also carries the pad-tail field, `pad_tail=<n>` or
`pad_tail=none`. It is the width of the node that carries
GGML_TENSOR_FLAG_REDUCE_PAD_TAIL, or `none` when no node of the graph
carries the flag. On this probe's graph (flash attention off, so the ordered
attention lane) the producer skips the mask-carrying softmax because it
already carries ATTN_SOFT_MAX (llama-context.cpp:4720), and its value-product
branch needs the softmax to carry REDUCE_PAD_TAIL first, so no node carries
it and `none` is the correct value in every arm. The field is required to be
present, because a missing field means the probe stopped observing it. A
width that IS present must equal the position, as the shapes do.

The mask row of the step is read after llama_decode returned: the probe
prints MASK_ROW MODE=... step=... width=... row=... visible=... ok=...
and this checker requires ok=1 on every step. That is the content oracle the
probe lacked: a mask filled at a wider width leaves columns outside the
executed span unwritten, and a stale mask shows a value that is neither 0
nor -inf.

The stock build count is asserted as a property, not a constant. `builds` is
sum(STEP BUILD) with BUILD = (rc == 0 && d_reused == 0), and d_reused comes
from the scheduler's own reuse accounting (llama_perf_context, n_reused). That
accounting is not reducible to a closed form from the probe source
(padmax_cpu_probe.cpp run_plan) or from llama_context's graph-build sites
(llama-context.cpp:5104/5158/5266/6499/7524), so the exact number would be a
guess. The property that is derivable is: the stock control builds at least
once (it has to produce a graph) and does not rebuild on every decode (a
per-step rebuild means reuse was lost). Commit 0cec65d made the same change in
tests/test-nkv-exact-width.cpp.

Every arm of every mode goes through the same oracle below; there is no
per-arm special case. The `seq` arm is a non-unified multi-stream cache, which
the unset gate does not qualify (llama-context.cpp:499), so its unset run is
byte-for-byte the explicit-off run and must show no binder stats: that falls
out of the per-mode `unset_arm` field. `seq` + `on` does reach the binder and
is checked against the shape-analysis refusal (see the block below).
"""
import re
import sys

# ctx tuple = (n_ctx, n_ctx_seq, n_ctx_kv)
MODES = {
    "nkv": {
        "steps": 512,
        "records": 512,
        "unset_arm": "on",
        "ctx": {
            "off":   (512, 512, 512),
            "on":    (512, 512, 512),
        },
        "cap":   {"on": 512},
        "binds": {"on": 512},
    },
    "cap": {
        "steps": 300,
        "records": 300,
        "unset_arm": "on",
        "ctx": {
            "off":   (512, 512, 512),
            "on":    (300, 300, 300),
        },
        "cap":   {"on": 300},
        "binds": {"on": 300},
    },
    "seq": {
        "steps": 501,
        "records": 501,
        "unset_arm": "off",
        "ctx": {
            "off":   (1024, 512, 512),
            "on":    (1001, 501, 501),
        },
        "cap":   {"on": 501},
        "binds": {"on": 501},
    },
}

STEP_RE = re.compile(
    r"^STEP=(\d+) ATTEMPT=(\d+) MODE=(\S+) SEQ=(-?\d+) NTOK=(\d+) "
    r"POS_BEFORE=(-?\d+) POS_AFTER=(-?\d+) NOUT=(\d+) RC=(-?\d+) "
    r"BUILD=(\d+) BUILD_TOTAL=(\d+) REUSED_TOTAL=(\d+)$")
CTX_RE = re.compile(r"^CTX_API MODE=(\S+) N_CTX=(\d+) N_CTX_SEQ=(\d+) N_CTX_KV=(\d+)$")
ARM_RE = re.compile(r"^ARM binder=(\S+) ntok=(\S+) reuse_disable=(\S+)$")
NKV_EXACT_RE = re.compile(
    r"^NKV_EXACT MODE=(\S+) step=(\d+) pos=(\d+) mask_ne0=(-?\d+) "
    r"k_ne2=(-?\d+) v_nkv=(-?\d+) pad_tail=(\S+)$")
MASK_ROW_RE = re.compile(
    r"^MASK_ROW MODE=(\S+) step=(\d+) width=(\d+) row=(\d+) "
    r"visible=(\d+) ok=(\d+)$")
PERF_RE = re.compile(
    r"^RUN_PERF MODE=(\S+) DECODE_CALLS=(\d+) N_REUSED=(-?\d+) BUILDS=(-?\d+) "
    r"REBUILDS_AFTER_FIRST=(-?\d+) ANOMALIES=(\d+)$")
KV_RE = re.compile(r"(\w+)=(-?\d+)")

BAD_PREFIXES = (
    "DECODE_FAIL", "MISSING_LOGITS", "CONTEXT_FAIL", "NO_MEMORY",
    "MODEL_LOAD_FAIL", "BAD_VOCAB", "UBATCH_ANOMALY", "POS_ANOMALY",
    "UNEXPECTED_SUCCESS", "FAIL:",
)

# The binder= value each arm must print. The probe reads the variable once and
# never writes it, so this is the arm the runner injected.
ARM_BINDER = {"unset": "unset", "off": "0", "on": "1"}

# pad_tail= has two states and they are different facts: an integer width, or
# the word "none" for "no node carried the flag". This checker keeps them
# apart and requires the field on every step.
PAD_TAIL_NONE = "none"


def fail(msg):
    print(f"FAIL {msg}")
    return 1


def parse_stats(line):
    idx = line.find("NKV_BINDER_STATS")
    if idx < 0:
        return None
    return {k: int(v) for k, v in KV_RE.findall(line[idx + len("NKV_BINDER_STATS"):])}


def parse_pad_tail(text):
    """The pad-tail field: an int width, or the word "none"."""
    if text == PAD_TAIL_NONE:
        return PAD_TAIL_NONE
    try:
        return int(text, 10)
    except ValueError:
        return None


def main():
    if len(sys.argv) != 4:
        print("usage: check_probe.py LOG MODE ARM", file=sys.stderr)
        return 2
    path, mode, arm = sys.argv[1:4]
    if mode not in MODES or arm not in ("unset", "off", "on"):
        print(f"bad mode/arm {mode}/{arm}", file=sys.stderr)
        return 2
    spec = MODES[mode]
    # The unset arm is the default the context picks for itself. For a
    # qualified single-stream CPU cache that default is the binder on
    # (llama-context.cpp:496-509 cpu_default); for a non-unified multi-stream
    # cache it is off (the same predicate requires kv_unified || n_seq_max == 1).
    # The mode's `ctx` table records which one applies, so no per-mode branch.
    expected_arm = spec["unset_arm"] if arm == "unset" else arm

    steps = []
    stats = None
    ctx_api = None
    perf = None
    harness_end = None
    arm_line = None
    nkv_exact = []
    mask_rows = []
    bad = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = ARM_RE.match(line)
            if m:
                arm_line = {"binder": m.group(1), "ntok": m.group(2),
                            "reuse_disable": m.group(3)}
                continue
            m = NKV_EXACT_RE.match(line)
            if m:
                pad_tail = parse_pad_tail(m.group(7))
                if pad_tail is None:
                    # a pad_tail token that is neither an integer nor "none"
                    bad.append(line)
                    continue
                nkv_exact.append({
                    "mode": m.group(1), "step": int(m.group(2)),
                    "pos": int(m.group(3)), "mask_ne0": int(m.group(4)),
                    "k_ne2": int(m.group(5)), "v_nkv": int(m.group(6)),
                    "pad_tail": pad_tail,
                })
                continue
            m = MASK_ROW_RE.match(line)
            if m:
                mask_rows.append({
                    "mode": m.group(1), "step": int(m.group(2)),
                    "width": int(m.group(3)), "row": int(m.group(4)),
                    "visible": int(m.group(5)), "ok": int(m.group(6)),
                })
                continue
            m = STEP_RE.match(line)
            if m:
                steps.append({
                    "step": int(m.group(1)), "attempt": int(m.group(2)),
                    "mode": m.group(3), "seq": int(m.group(4)),
                    "ntok": int(m.group(5)), "pos_before": int(m.group(6)),
                    "pos_after": int(m.group(7)), "nout": int(m.group(8)),
                    "rc": int(m.group(9)), "build": int(m.group(10)),
                    "build_total": int(m.group(11)), "reused_total": int(m.group(12)),
                })
                continue
            m = CTX_RE.match(line)
            if m:
                ctx_api = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                continue
            m = PERF_RE.match(line)
            if m:
                perf = {
                    "decode_calls": int(m.group(2)), "n_reused": int(m.group(3)),
                    "builds": int(m.group(4)), "rebuilds": int(m.group(5)),
                    "anomalies": int(m.group(6)),
                }
                continue
            if "NKV_BINDER_STATS" in line:
                stats = parse_stats(line)
                continue
            if line.startswith("HARNESS_END "):
                harness_end = line
                continue
            if line.startswith(BAD_PREFIXES):
                bad.append(line)
            elif line.startswith("NKV_EXACT ") or line.startswith("MASK_ROW "):
                # a malformed line of either family must not pass as "absent"
                bad.append(line)

    if bad:
        return fail(f"mode={mode} arm={arm} harness_errors={bad[:3]}")
    if arm_line is None:
        return fail(f"mode={mode} arm={arm} missing ARM line")
    if arm_line["binder"] != ARM_BINDER[arm]:
        return fail(f"mode={mode} arm={arm} ARM binder={arm_line['binder']} "
                    f"want={ARM_BINDER[arm]}")
    if harness_end is None or "OK=1" not in harness_end:
        return fail(f"mode={mode} arm={arm} harness_end={harness_end!r}")
    m = re.search(r"RECORDS=(\d+)", harness_end)
    if not m or int(m.group(1)) != spec["records"]:
        return fail(f"mode={mode} arm={arm} records={m.group(1) if m else None} "
                    f"want={spec['records']}")
    if len(steps) != spec["steps"]:
        return fail(f"mode={mode} arm={arm} steps={len(steps)} want={spec['steps']}")
    if ctx_api is None:
        return fail(f"mode={mode} arm={arm} missing CTX_API")
    want_ctx = spec["ctx"][expected_arm]
    if ctx_api != want_ctx:
        return fail(f"mode={mode} arm={arm} ctx={ctx_api} want={want_ctx} "
                    f"(exact capacity must not be padded)")
    if perf is None:
        return fail(f"mode={mode} arm={arm} missing RUN_PERF")

    # every step must succeed, record a single output and advance one position
    for i, st in enumerate(steps):
        if st["step"] != i or st["attempt"] != i:
            return fail(f"mode={mode} arm={arm} step index {st['step']}/{st['attempt']} want {i}")
        if st["mode"] != mode or st["nout"] != 1 or st["ntok"] != 1:
            return fail(f"mode={mode} arm={arm} step={i} mode/nout/ntok="
                        f"{st['mode']}/{st['nout']}/{st['ntok']}")
        if st["rc"] != 0:
            return fail(f"mode={mode} arm={arm} step={i} rc={st['rc']}")
        if st["pos_after"] != st["pos_before"] + 1:
            return fail(f"mode={mode} arm={arm} step={i} pos {st['pos_before']}->{st['pos_after']}")
    # The three n_kv runtime roots, observed through the eval callback on
    # every successful step: the mask (llama-graph.cpp:91-93), the K view
    # (llama-kv-cache.cpp:2177) and the V view (the layout-dependent
    # dimension). Each must execute at the 1-based position after the step.
    if len(nkv_exact) != spec["steps"]:
        return fail(f"mode={mode} arm={arm} NKV_EXACT lines={len(nkv_exact)} "
                    f"want={spec['steps']}")
    for i, row in enumerate(nkv_exact):
        if row["mode"] != mode or row["step"] != i or row["pos"] != i + 1:
            return fail(f"mode={mode} arm={arm} NKV_EXACT row {i}: {row}")
        if not (row["mask_ne0"] == row["k_ne2"] == row["v_nkv"] == row["pos"]):
            return fail(f"mode={mode} arm={arm} executed width at step={i} "
                        f"mask={row['mask_ne0']} k={row['k_ne2']} "
                        f"v={row['v_nkv']} pos={row['pos']}")
        # the pad tail is a width when present, and must match the position;
        # "none" is the ordered lane and is the expected value on this graph
        if row["pad_tail"] != PAD_TAIL_NONE and row["pad_tail"] != row["pos"]:
            return fail(f"mode={mode} arm={arm} pad tail at step={i} "
                        f"pad_tail={row['pad_tail']} pos={row['pos']}")
    # The mask row of every step must hold 0.0f or -INFINITY below the
    # executed width, with at least one visible column. The probe reads it
    # after llama_decode returned and prints the verdict; this checker does
    # not re-read the buffer, it refuses a step whose verdict is not ok.
    if len(mask_rows) != spec["steps"]:
        return fail(f"mode={mode} arm={arm} MASK_ROW lines={len(mask_rows)} "
                    f"want={spec['steps']}")
    for i, row in enumerate(mask_rows):
        if row["mode"] != mode or row["step"] != i:
            return fail(f"mode={mode} arm={arm} MASK_ROW row {i}: {row}")
        if row["ok"] != 1 or row["visible"] < 1:
            return fail(f"mode={mode} arm={arm} mask row at step={i} "
                        f"ok={row['ok']} visible={row['visible']}")
        if row["width"] != i + 1:
            return fail(f"mode={mode} arm={arm} mask row width at step={i} "
                        f"width={row['width']} want={i + 1}")
    if perf["decode_calls"] != spec["steps"]:
        return fail(f"mode={mode} arm={arm} decode_calls={perf['decode_calls']}")
    if perf["anomalies"] != 0:
        return fail(f"mode={mode} arm={arm} anomalies={perf['anomalies']}")

    build_steps = sum(st["build"] for st in steps)
    if perf["builds"] != build_steps:
        return fail(f"mode={mode} arm={arm} RUN_PERF builds={perf['builds']} "
                    f"step_builds={build_steps}")
    if perf["n_reused"] != spec["steps"] - build_steps:
        return fail(f"mode={mode} arm={arm} n_reused={perf['n_reused']} "
                    f"want={spec['steps'] - build_steps}")

    if expected_arm == "off":
        if stats is not None:
            return fail(f"mode={mode} arm={arm} printed NKV_BINDER_STATS (gate must be off)")
        # property, not a guessed count: at least one build, and not one per
        # decode (see the module docstring)
        if not (1 <= build_steps < spec["steps"]):
            return fail(f"mode={mode} arm={arm} builds={build_steps} "
                        f"want 1 <= builds < {spec['steps']}")
        print(f"PASS mode={mode} arm={arm} steps={len(steps)} records={spec['records']} "
              f"builds={build_steps} ctx={ctx_api} no_binder_stats")
        return 0

    if stats is None:
        return fail(f"mode={mode} arm={arm} missing NKV_BINDER_STATS")
    if mode == "seq":
        # Only seq+on reaches here: the unset arm of a non-unified cache maps to
        # "off" (llama-context.cpp:499, cpu_default needs kv_unified ||
        # n_seq_max == 1), so it allocates no binder state and returns above.
        #
        # seq+on does reach the binder, and the refusal is observed in two
        # stages, both in llama-context.cpp:
        #   :5928  the envelope build still happens (record_build does not test
        #          n_stream), so env_builds > 0 and graph_builds > 0;
        #   :5971  union_shape_analyze fails for a multi-stream cache (the mask
        #          root has ne[3] == n_stream, which the symbolic map cannot
        #          derive), so shape_fail > 0, env_disabled becomes 1 and the
        #          context rebuilds at the stock width (:6007-6037).
        # The live bind refusal at :4867 (refused_skeleton) is only reached
        # when a record exists, and the shape failure above prevents one, so
        # it must NOT be required here.
        if (stats.get("env_disabled") != 1 or stats.get("shape_fail", 0) < 1 or
                stats.get("patch_binds") != 0 or stats.get("env_builds", 0) < 1):
            return fail(f"mode=seq arm={arm} multi-stream guard not enforced: {stats}")
        print(f"PASS mode=seq arm={arm} multi-stream refused builds={build_steps} stats={stats}")
        return 0
    # there is one lane: the NKV_BINDER_STATS line no longer carries an
    # exact field, so nothing to compare here
    if stats.get("cap") != spec["cap"][expected_arm]:
        return fail(f"mode={mode} arm={arm} cap={stats.get('cap')} "
                    f"want={spec['cap'][expected_arm]} (requested capacity, no pad)")
    if stats.get("graph_builds") != 1:
        return fail(f"mode={mode} arm={arm} graph_builds={stats.get('graph_builds')} want=1")
    if stats.get("env_builds") != 1:
        return fail(f"mode={mode} arm={arm} env_builds={stats.get('env_builds')} want=1")
    if stats.get("patch_binds") != spec["binds"][expected_arm]:
        return fail(f"mode={mode} arm={arm} patch_binds={stats.get('patch_binds')} "
                    f"want={spec['binds'][expected_arm]}")
    # The first decode runs rebind_any() before any record exists: n_kv is
    # refused by the envelope, n_outputs has no closure, rs_z has no recurrent
    # context.  Those three are the only expected fallbacks; every later bind
    # must succeed.  They are not graph rebuilds.
    if stats.get("fallbacks") > 3:
        return fail(f"mode={mode} arm={arm} fallbacks={stats.get('fallbacks')} want<=3")
    if stats.get("attempts") != stats.get("patch_binds", -1) + stats.get("fallbacks", -1):
        return fail(f"mode={mode} arm={arm} attempts={stats.get('attempts')} "
                    f"patch_binds+fallbacks")
    for key in ("refused_align", "refused_skeleton", "refused_relay",
                "refused_pipeline", "refused_reader", "refused_backend",
                "refused_iswa", "refused_nkv_unknown", "refused_nout"):
        if stats.get(key, 0) != 0:
            return fail(f"mode={mode} arm={arm} {key}={stats.get(key)} want=0")
    if stats.get("refused_envelope", 0) > 1:
        return fail(f"mode={mode} arm={arm} refused_envelope={stats.get('refused_envelope')} want<=1")
    if stats.get("refused_rs_z", 0) > 1:
        return fail(f"mode={mode} arm={arm} refused_rs_z={stats.get('refused_rs_z')} want<=1")
    if stats.get("shape_fail", 0) != 0:
        return fail(f"mode={mode} arm={arm} shape_fail={stats.get('shape_fail')}")
    if stats.get("env_disabled", 0) != 0:
        return fail(f"mode={mode} arm={arm} env_disabled={stats.get('env_disabled')}")
    if perf["builds"] != stats.get("graph_builds"):
        return fail(f"mode={mode} arm={arm} RUN_PERF builds={perf['builds']} "
                    f"stats graph_builds={stats.get('graph_builds')}")

    print(f"PASS mode={mode} arm={arm} steps={len(steps)} records={spec['records']} "
          f"ctx={ctx_api} cap={stats.get('cap')} graph_builds={stats.get('graph_builds')} "
          f"env_builds={stats.get('env_builds')} patch_binds={stats.get('patch_binds')} "
          f"fallbacks={stats.get('fallbacks')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
