#!/usr/bin/env python3
"""Emit one TSV row per (mode, arm) from the padmax probe stderr logs.

Usage: summarize.py LOGS_DIR OUT.tsv

The columns are exactly what padmax_cpu_probe.cpp and the context print:
  CTX_API MODE=... N_CTX=... N_CTX_SEQ=... N_CTX_KV=...      (run_plan)
  RUN_PERF MODE=... DECODE_CALLS=... N_REUSED=... BUILDS=... (run_plan)
  STEP=... (one per decode)                                  (run_plan)
  HARNESS_END OK=... RECORDS=...                             (main)
  NKV_BINDER_STATS ...                                       (llama_free)
There is no `exact` column: the executed n_kv width is the true used span on
every arm, so there is nothing to distinguish. There are no `pad` or `exact`
arms either; run_probe.sh runs unset, off and on only.
"""
import os
import re
import sys

CTX_RE = re.compile(r"^CTX_API MODE=(\S+) N_CTX=(\d+) N_CTX_SEQ=(\d+) N_CTX_KV=(\d+)$")
PERF_RE = re.compile(r"^RUN_PERF MODE=(\S+) DECODE_CALLS=(\d+) N_REUSED=(-?\d+) "
                     r"BUILDS=(-?\d+) REBUILDS_AFTER_FIRST=(-?\d+) ANOMALIES=(\d+)$")
STEP_RE = re.compile(r"^STEP=(\d+) ATTEMPT=(\d+) MODE=(\S+) SEQ=(-?\d+) NTOK=(\d+) "
                     r"POS_BEFORE=(-?\d+) POS_AFTER=(-?\d+) NOUT=(\d+) RC=(-?\d+) "
                     r"BUILD=(\d+) BUILD_TOTAL=(\d+) REUSED_TOTAL=(\d+)$")
KV_RE = re.compile(r"(\w+)=(-?\d+)")

COLUMNS = ["mode", "arm", "n_ctx", "n_ctx_seq", "n_ctx_kv", "cap", "cap_nout",
           "graph_builds", "env_builds", "patch_binds", "fallbacks", "attempts",
           "builds", "n_reused", "records", "steps"]

# NKV_BINDER_STATS fields copied into the table. Every one of them is printed
# by llama-context.cpp when the binder is enabled.
STATS_COLUMNS = ("cap", "cap_nout", "graph_builds", "env_builds",
                 "patch_binds", "fallbacks", "attempts")

# the arms run_probe.sh actually runs
ARMS = ("unset", "off", "on")


def parse(path):
    row = {k: "-" for k in COLUMNS}
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = CTX_RE.match(line)
            if m:
                row["n_ctx"], row["n_ctx_seq"], row["n_ctx_kv"] = m.group(2), m.group(3), m.group(4)
                continue
            m = PERF_RE.match(line)
            if m:
                row["builds"], row["n_reused"] = m.group(4), m.group(3)
                continue
            m = STEP_RE.match(line)
            if m:
                row["steps"] = (int(row["steps"]) if row["steps"] != "-" else 0) + 1
                continue
            if "NKV_BINDER_STATS" in line:
                stats = {k: v for k, v in KV_RE.findall(line[line.find("NKV_BINDER_STATS"):])}
                for key in STATS_COLUMNS:
                    if key in stats:
                        row[key] = stats[key]
                continue
            if line.startswith("HARNESS_END "):
                m = re.search(r"RECORDS=(\d+)", line)
                if m:
                    row["records"] = m.group(1)
    return row


def main():
    if len(sys.argv) != 3:
        print("usage: summarize.py PROBE_DIR OUT.tsv", file=sys.stderr)
        return 2
    logs_dir, out_path = sys.argv[1:3]
    rows = []
    for mode in ("nkv", "cap", "seq"):
        for arm in ARMS:
            path = os.path.join(logs_dir, f"{arm}-{mode}.stderr.log")
            if not os.path.exists(path):
                continue
            row = parse(path)
            row["mode"], row["arm"] = mode, arm
            rows.append(row)
    with open(out_path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in COLUMNS) + "\n")
    print(f"TABLE {out_path} rows={len(rows)}")
    # No row means no (mode, arm) log was found: nothing was compared. That
    # is UNPROVEN (exit 2), not success.
    if not rows:
        print(f"UNPROVEN: no (mode, arm) log found under {logs_dir}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
