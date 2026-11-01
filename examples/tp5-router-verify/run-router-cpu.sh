#!/usr/bin/env bash
# Router scratch (TP5 MoE router retention) — real-consumer CPU verification.
#
# Runs, in order, on the 512-expert production router shape with every hidden
# dimension reduced to the fixture minimum (n_embd=8, 2 trunk + 1 NextN block):
#
#   F0  fixture   llama-tp5-router-fixture  -> model.gguf (self-checks the load
#                 and a real 512/3-row router graph on the CPU backend)
#   L1  model     llama-tp5-router-verify   -> 1/2/3/4-row logits, hashed
#   L2  plan      same binary, --plan-rows   -> gallocr plan audit + compute
#   legs          OFF (all knobs unset, then =0) and ON (all three =1), so the
#                 OFF-vs-stock check is "knob off changes nothing" and the ON
#                 check is retention firing, no overlap, and identical output.
#
# C4b (promotion) grades an owner-side join: llama_context prints the identity of
# the graph it submits (`router_scratch plan identity ... match=<0|1>`) and the
# allocator prints the identity of the plan it audited (`... graph_uid= plan_id=
# fp=`). Both lines are graded together; a missing line, match=0, or a plan_id
# that moved after the reserve stays UNPROVEN and fails the run.
#
# Nothing here interprets the output for you: every check prints PASS/FAIL with
# the evidence line it used. CPU only, single thread, no GPU.
#
# Build (Devops, serial). Out-of-tree and absolute; this script never compiles and
# never calls build-slot (see the note above run()). A slot rc of 75/124/127 in the
# build step is a failure to be reported, not a tool result: 75 = queue timeout,
# 124 = compile timeout, 127 = the wrapper could not exec its argv.
#   make build dirs:
#     src=/home/kunweiz/work/merge-router-20261031 ; build=$src/build-router-cpu
#     cmake -S "$src" -B "$build" -DGGML_CUDA=OFF -DGGML_HIP=OFF \
#           -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
#     /home/kunweiz/.local/bin/build-slot cmake --build "$build" -j2 \
#           --target llama-tp5-router-fixture llama-tp5-router-verify
#   then run:  ./run-router-cpu.sh "$build/bin" /abs/out-dir
#
# Candidate and the 4496493dd reference share the same sources for the fixture
# and verifier (same seed -> identical GGUF), so their FIXTURE/VERIFY lines are
# directly comparable; the only difference is the router-scratch delta.
#
# The L1 model leg is the target: it runs llama_decode on the fixture and its
# result is a hard criterion. It needs the shared prerequisite fix for the
# hybrid graph input (see /home/kunweiz/work/router-hybrid-input-fix-20261031),
# applied to BOTH the candidate and the 4496493dd reference tree, otherwise
# set_inputs aborts in llama_kv_cache::set_input_k_idxs (ggml-backend.cpp:319).
#
# MODEL_LEG defaults to 1 (fail-closed: the model leg runs and any decode
# failure is a FAIL). MODEL_LEG=0 is a diagnostic escape hatch only: the
# verifier then runs with --no-model, the model leg is printed as SKIPPED and
# the run cannot report CRITERIA MET (exit 3), so it can never stand in for the
# real L1 result.
#
# Usage: ./run-router-cpu.sh <bin-dir> <out-dir>
#   Both are made absolute before use; out-dir is created if missing.
#
# Exit status: 0 = every criterion met, 1 = FAIL (including "a leg did not run"),
#              3 = diagnostic mode (MODEL_LEG != 1; never a verdict), 64 = bad usage,
#              2 = a required binary is missing.
#
# Fail-closed rules learned from the rc=127 run: the fixture and verifier are executed
# directly, never wrapped, and a non-zero exit stops the run before any criterion is
# derived from their (possibly empty) output files. Only cmake holds a build slot.

set -uo pipefail

usage() { printf 'usage: %s <bin-dir> <out-dir>\n  (BIN_DIR must contain llama-tp5-router-fixture and llama-tp5-router-verify)\n' "$0" >&2; }

[ $# -eq 2 ] || { usage; exit 64; }
[ -d "$1" ] || { printf '%s: not a directory: %s\n' "$0" "$1" >&2; exit 64; }
# absolute, so a failed exec can never be confused with a relative-path resolution
BIN=$(cd -- "$1" && pwd) || exit 64
OUT=$2
case $OUT in
    /*) ;;
    *) OUT=$(pwd)/$OUT ;;
esac
FIXTURE=$BIN/llama-tp5-router-fixture
VERIFY=$BIN/llama-tp5-router-verify
MODEL=$OUT/router-fixture.gguf
MODEL_LEG=${MODEL_LEG:-1}

mkdir -p "$OUT"
LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
fail=0

note() { printf '%s\n' "$*" | tee -a "$LOG"; }

# MODEL_LEG=1 (default) runs the real model leg. Anything else is a diagnostic
# mode: --no-model, the leg printed as skipped, and exit 3 (never CRITERIA MET).
VERIFY_MODEL_ARGS=()
MODEL_LEG_BLOCKED=0
if [ "$MODEL_LEG" != "1" ]; then
    VERIFY_MODEL_ARGS=(--no-model)
    MODEL_LEG_BLOCKED=1
    note "DIAGNOSTIC MODE (MODEL_LEG=$MODEL_LEG): the model leg is skipped; this run cannot report CRITERIA MET"
fi

# The fixture and the verifier are short-lived release binaries and are executed
# directly, never through build-slot: that wrapper runs its command under
# `setsid timeout`, and when the exec fails it returns 127, replacing the tool's own
# status (observed: four legs all rc=127 while the same argv executed directly gave 0).
# Compilation keeps the slot, and the script does not compile.
#
# run <argv...>: direct execution, never through a wrapper. The caller checks rc.
#
# $1 may be a shell function (`scrubbed`), so it is executed as given and a real
# exec failure surfaces as its own status; the binaries themselves are checked by
# the `for b in` loop below. Testing `-x "$1"` here rejected the function name and
# turned every leg into rc=127.
run() {
    "$@"
}

# Verdict helper: a missing leg output is missing evidence, not agreement.
require_file() { # <path> <what>
    if [ ! -s "$1" ]; then
        note "  FAIL: $2 produced no output ($1) — the leg did not run"
        fail=1
        return 1
    fi
    return 0
}

# The five knobs, always scrubbed first so a stale shell cannot leak in. Used as
# `run scrubbed <argv...>`; `run` executes its argv directly, so a function name is
# a valid $1 (testing `-x "$1"` here rejected it and made every leg rc=127).
scrubbed() { env -u GGML_TP5_MOE_ROUTER_SCRATCH -u GGML_TP5_ROUTER_SCRATCH_AUDIT -u GGML_TP5_ROUTER_SCRATCH_DEBUG -u GGML_TP5_ROUTER_SCRATCH_DIAG -u GGML_TP5_ROUTER_SCRATCH_DIAG_FILE "$@"; }

for b in "$FIXTURE" "$VERIFY"; do
    [ -x "$b" ] || { note "MISSING BINARY: $b"; exit 2; }
done

note "== F0: fixture =="
: > "$OUT/fixture.txt"; : > "$OUT/fixture.err"
run scrubbed env "$FIXTURE" -o "$MODEL" > "$OUT/fixture.txt" 2> "$OUT/fixture.err"
rc=$?
if [ $rc -ne 0 ]; then
    note "  FAIL: fixture generator rc=$rc (see $OUT/fixture.err)"
    sed 's/^/    /' "$OUT/fixture.err" | tee -a "$LOG"
    note "== RESULT: FAIL — the fixture could not be produced; nothing below can be judged =="
    note "log: $LOG"
    exit 1
fi
require_file "$OUT/fixture.txt" "the fixture generator"
grep -E '^FIXTURE' "$OUT/fixture.txt" | tee -a "$LOG"
grep -q 'contract problems=0' "$OUT/fixture.txt" || {
    note "  FAIL: the generated file does not satisfy the Qwen4Exp loader contract"
    grep 'contract_problem' "$OUT/fixture.txt" | sed 's/^/    /' | tee -a "$LOG"
    fail=1
}
grep -q 'router_shape=1' "$OUT/fixture.txt" || { note "  FAIL: fixture self-check did not see the 512/3-row router graph"; fail=1; }
grep -q 'model_load=1' "$OUT/fixture.txt"   || { note "  FAIL: fixture does not load through llama_model"; fail=1; }
note "  fixture sha256: $(sha256sum -- "$MODEL" | cut -c1-32)"

# ---------------------------------------------------------------- L1 model leg
# L1/L2 leg. A non-zero rc stops the run: a leg that did not execute must never be
# turned into evidence by reading the (empty) files it would have written.
leg() { # <tag> <ENV=..>... -- <verify args...>
    local tag=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    note "--- $tag: env=[${envs[*]:-none}] args=[$*]"
    rm -f "$OUT/$tag.txt" "$OUT/$tag.err"
    run scrubbed env "${envs[@]}" "$VERIFY" -m "$MODEL" "${VERIFY_MODEL_ARGS[@]}" "$@" \
        > "$OUT/$tag.txt" 2> "$OUT/$tag.err"
    local rc=$?
    if [ $rc -ne 0 ]; then
        note "    rc=$rc (FAIL)"
        sed 's/^/    /' "$OUT/$tag.err" | tail -20 | tee -a "$LOG"
        note "  FAIL: leg '$tag' exited rc=$rc — its result is unknown, not negative"
        fail=1
        return 1
    fi
    if ! require_file "$OUT/$tag.txt" "leg '$tag'"; then
        return 1
    fi
    note "    rc=0 retain=$(grep -c 'router_scratch retain layer' "$OUT/$tag.err") \
overlap=$(grep -c 'OVERLAPS' "$OUT/$tag.txt") audit_err=$(grep -c 'overlaps' "$OUT/$tag.err")"
    grep -E '^VERIFY_(ENV|MODEL|ROWS|PLAN_SUMMARY)' "$OUT/$tag.txt" | sed 's/^/    /' | tee -a "$LOG"
    return 0
}

# No derived criterion may be evaluated when a leg did not run: stop with a verdict
# that says exactly that, instead of comparing empty files and printing "identical".
leg_failed() { # <tag>
    note "== RESULT: FAIL — leg '$1' did not run; no criterion above or below it is"
    note "   meaningful, so the run stops here rather than reinterpreting missing output."
    note "log: $LOG"
    exit 1
}

note "== L1/L2: OFF (stock semantics) =="
leg off_1 GGML_TP5_ROUTER_SCRATCH_AUDIT=1 GGML_TP5_ROUTER_SCRATCH_DIAG=1 GGML_TP5_ROUTER_SCRATCH_DIAG_FILE="$OUT/diag_off_1.txt" \
          -- --rows 1,2,3,4 --plan-rows 3 || leg_failed off_1
note "== L1/L2: OFF with every knob set to 0 =="
leg off_0 GGML_TP5_MOE_ROUTER_SCRATCH=0 GGML_TP5_ROUTER_SCRATCH_AUDIT=0 \
          GGML_TP5_ROUTER_SCRATCH_DEBUG=0 -- --rows 1,2,3,4 --plan-rows 3 || leg_failed off_0
note "== L1/L2: ON =="
leg on_1 GGML_TP5_MOE_ROUTER_SCRATCH=1 GGML_TP5_ROUTER_SCRATCH_AUDIT=1 \
         GGML_TP5_ROUTER_SCRATCH_DEBUG=1 \
         GGML_TP5_ROUTER_SCRATCH_DIAG=1 GGML_TP5_ROUTER_SCRATCH_DIAG_FILE="$OUT/diag_on_1.txt" \
         -- --rows 1,2,3,4 --plan-rows 3 || leg_failed on_1

# ---------------------------------------------------------------- criteria
if [ "$MODEL_LEG_BLOCKED" = "1" ]; then
    note "== C1/C2/C3/C4: model leg SKIPPED (diagnostic mode) — nothing here is"
    note "   reported as passing, and the run ends with exit 3, never CRITERIA MET."
    grep -h '^VERIFY_MODEL_LEG' "$OUT"/off_1.txt "$OUT"/on_1.txt | sed 's/^/    /' | tee -a "$LOG"
else
note "== C1: OFF(=unset) bitwise == OFF(=0) on the model leg =="
c1_a=$(grep -cE '^VERIFY_ROWS' "$OUT/off_1.txt")
c1_b=$(grep -cE '^VERIFY_ROWS' "$OUT/off_0.txt")
if [ "$c1_a" -eq 0 ] || [ "$c1_b" -eq 0 ]; then
    note "  FAIL: no logits rows to compare (off_1=$c1_a off_0=$c1_b): empty output is not agreement"
    fail=1
elif diff <(grep -E '^VERIFY_(ROW|ROWS)' "$OUT/off_1.txt") <(grep -E '^VERIFY_(ROW|ROWS)' "$OUT/off_0.txt") > /dev/null; then
    note "  PASS: identical logits hashes"
else
    note "  FAIL: unset and =0 differ"; fail=1
fi

note "== C2: OFF vs ON model logits bitwise (expect identical: flag-only change) =="
if [ "$(grep -cE '^VERIFY_ROWS' "$OUT/on_1.txt")" -eq 0 ]; then
    note "  FAIL: the ON leg produced no logits rows"
    fail=1
elif diff <(grep -E '^VERIFY_(ROW|ROWS)' "$OUT/off_1.txt") <(grep -E '^VERIFY_(ROW|ROWS)' "$OUT/on_1.txt") > /dev/null; then
    note "  PASS: identical"
else
    note "  DIFFERS (see below) — a difference here is a real finding, not a gate on its own"
    diff <(grep -E '^VERIFY_(ROW|ROWS)' "$OUT/off_1.txt") <(grep -E '^VERIFY_(ROW|ROWS)' "$OUT/on_1.txt") | sed 's/^/    /' | tee -a "$LOG"
    fail=1
fi

note "== C3: retention actually fired on real Qwen4Exp layers (ON only) =="
on_layers=$(grep -c 'router_scratch retain layer' "$OUT/on_1.err")
off_layers=$(grep -c 'router_scratch retain layer' "$OUT/off_1.err")
note "  ON retain lines=$on_layers  OFF retain lines=$off_layers"
[ "$off_layers" -eq 0 ] || { note "  FAIL: retention fired with the gate off"; fail=1; }
note "  expected ON: 2 (trunk blocks 0,1 at 2..4 rows; the NextN block is not in the decode graph)"
if [ "$on_layers" -ge 2 ]; then
    note "  PASS: retention fired on the multi-row decode"
else
    note "  MISS/FAIL: retention did not fire on the multi-row decode (see $OUT/on_1.err)"; fail=1
fi
grep 'router_scratch retain layer' "$OUT/on_1.err" | sed 's/^/    /' | tee -a "$LOG"

note "== C4: row window (N1 must never be retained) =="
# a dedicated 1-row run: the model marks 2..4 rows only, so nothing may be retained
rm -f "$OUT/on_n1.txt" "$OUT/on_n1.err"
run scrubbed env GGML_TP5_MOE_ROUTER_SCRATCH=1 GGML_TP5_ROUTER_SCRATCH_DEBUG=1 \
    GGML_TP5_ROUTER_SCRATCH_AUDIT=1 \
    GGML_TP5_ROUTER_SCRATCH_DIAG=1 GGML_TP5_ROUTER_SCRATCH_DIAG_FILE="$OUT/diag_on_n1.txt" \
    "$VERIFY" -m "$MODEL" "${VERIFY_MODEL_ARGS[@]}" --rows 1 --plan-rows 1 \
    > "$OUT/on_n1.txt" 2> "$OUT/on_n1.err"
n1_rc=$?
if [ $n1_rc -ne 0 ] || ! require_file "$OUT/on_n1.txt" "the 1-row leg"; then
    leg_failed "on_n1 (rc=$n1_rc)"
fi
grep -E '^VERIFY_(ROWS|PLAN_SUMMARY)' "$OUT/on_n1.txt" | sed 's/^/    /' | tee -a "$LOG"
n1_retain=$(grep -c 'router_scratch retain layer' "$OUT/on_n1.err")
note "  1-row retain lines=$n1_retain (expect 0: the gate starts at 2 rows)"
[ "$n1_retain" -eq 0 ] || { note "  FAIL: retention fired for a single row"; fail=1; }

fi

note "== C4b: promotion gate — audited plan is the submitted decode graph =="
# The cb_eval numbers stay triage: they cannot join anything. The join is the
# owner's identity line plus the allocator's evidence line, both on stderr.
diag_on=$(grep -h '^VERIFY_REAL_DIAG ' "$OUT/on_1.txt" | tail -1)
diag_off=$(grep -h '^VERIFY_REAL_DIAG ' "$OUT/off_1.txt" | tail -1)
note "  ON : ${diag_on:-<missing>}"
note "  OFF: ${diag_off:-<missing>}"

c4b_fail() { note "  FAIL: $*"; fail=1; }

diag_triage() { note "  diag (triage): $*"; }

# C4b grades the owner-side join: (graph_uid, plan_id).
#
# The allocator audit (ggml_gallocr_check_no_reuse_plan in
# ggml_gallocr_reserve_n_impl) is production work over the complete node+leaf
# plan; it now records and prints the identity of the plan it judged
# (`graph_uid=`, `plan_id=`, `fp=`). llama_context::graph_reserve records what the
# scheduler stamped on the reserved graph, and llama_context::graph_compute prints
# `router_scratch plan identity ... match=<0|1>` for a graph it submits with
# retained inputs (that graph is not allocated again, so the plan the allocator
# holds is the plan it is replayed against). C4b requires that line to say
# match=1 on the ON leg, with the same graph_uid/plan_id as the allocator's clean
# line. Anything else -- a missing line, match=0, a plan_id that moved after the
# reserve (a re-reserve of the same graph object), a clean line from the (size)
# measurement entry, or no clean line at all -- stays UNPROVEN and fails.
on_lines=$(grep -h 'retention audit' "$OUT/on_1.err" | sed 's/^.*retention audit /retention audit /')
off_lines=$(grep -h 'retention audit' "$OUT/off_1.err" 2>/dev/null | sed 's/^.*retention audit /retention audit /' || true)
n1_lines=$(grep -h 'retention audit' "$OUT/on_n1.err" 2>/dev/null | sed 's/^.*retention audit /retention audit /' || true)
note "  ON audit lines:"; printf '%s\n' "${on_lines:-<none>}" | sed 's/^/    /' | tee -a "$LOG"
note "  OFF audit lines:"; printf '%s\n' "${off_lines:-<none>}" | sed 's/^/    /' | tee -a "$LOG"
note "  N1 audit lines:";  printf '%s\n' "${n1_lines:-<none>}" | sed 's/^/    /' | tee -a "$LOG"

# the log handler may prepend a level tag (default callback does not, llama's may), so
# never anchor at the line start; the tail anchor stays
RE_CLEAN='retention audit clean \(reserve\): [0-9]+ retained allocations, [0-9]+ bytes, rows=(mixed|[0-9]+), nodes=[0-9]+, graph_uid=[0-9]+, plan_id=[0-9]+, fp=[0-9a-f]+$'
RE_CLEAN_SIZE='retention audit clean \(size\): '
RE_ENABLED='retention audit enabled \(reserve\): 0 retained allocations, nodes=[0-9]+, graph_uid=[0-9]+, plan_id=[0-9]+, fp=[0-9a-f]+$'

# The only hard failures here are contract violations the allocator itself refuses
# the plan for: those are functional errors regardless of which graph was audited.
if printf '%s\n' "$on_lines" | grep -qE "retained tensor '[^']*' \[[^]]*\] (overlaps|is a view)|is a view, it cannot be retained"; then
    c4b_fail "the production audit reported a retention contract violation (see the ON stderr)"
fi

# Everything below is a DIAGNOSTIC OBSERVATION, not a pass/fail signal.
# - a (size) measurement reserve may legitimately carry the same structural marks; it
#   simply cannot vouch for the executed graph, so it is recorded and never graded;
# - rows=mixed, a clean line on the OFF leg, an OFF/1-row clean line, or a missing
#   zero line each mean "this run tells us nothing about that statement", not "the
#   feature is broken". They are printed with UNPROVEN and never set `fail`.
n_size=$(printf '%s\n' "$on_lines" | grep -c 'retention audit clean \(size\): ')
[ "$n_size" -gt 0 ] && diag_triage "ON leg has $n_size clean (size) line(s): expected for a structural measurement reserve, no evidentiary weight"

if printf '%s\n' "$on_lines" | grep -E "$RE_CLEAN" | grep -Eq 'rows=mixed'; then
    diag_triage "a clean (reserve) line reports rows=mixed: the retained set spans shapes, so this reserve is not a single decode shape"
fi
if printf '%s\n' "$off_lines" | grep -Eq "$RE_ENABLED"; then
    diag_triage "OFF: reserve audit ran and vouched for 0 retained allocations (consistent with the gate being off)"
else
    diag_triage "OFF: no 'enabled (reserve): 0 retained allocations' line, so nothing is claimed about the OFF reserve"
fi
if printf '%s\n' "$off_lines" | grep -q "$RE_CLEAN"; then
    diag_triage "OFF: a clean (reserve) line appeared; it cannot be attributed to a gated-off decode plan, so it is not evidence"
fi
if printf '%s\n' "$n1_lines" | grep -Eq "$RE_ENABLED"; then
    diag_triage "1-row: reserve audit ran and vouched for 0 retained allocations (consistent with the 2..4-row window)"
else
    diag_triage "1-row: no 'enabled (reserve): 0 retained allocations' line, so nothing is claimed about that reserve"
fi
if printf '%s\n' "$n1_lines" | grep -q "$RE_CLEAN"; then
    diag_triage "1-row: a clean (reserve) line appeared; it cannot be attributed to the 1-row decode plan, so it is not evidence"
fi

# ---- the join: one owner line, one allocator line, one plan ----
# llama_context prints one identity line per decode submission (not only the
# retained-input one), AFTER the scheduler ran, so the allocator already holds the
# plan that submission used. The last line is the final decode step and pairs with
# the last clean (reserve) line.
id_lines=$(grep -h 'router_scratch plan identity ' "$OUT/on_1.err" 2>/dev/null \
    | sed 's/^.*router_scratch plan identity //' || true)
n_id=$(printf '%s\n' "$id_lines" | grep -c 'exec_uid=' || true)
id_last=$(printf '%s\n' "$id_lines" | grep 'exec_uid=' | tail -1)
note "  ON identity lines=$n_id"
[ -z "$id_last" ] || note "  last: $id_last"

# Field extraction is anchored on the delimiter, never on the bare name: the
# identity line carries `plan_id=`, `audit_plan_id=` and `recorded_plan_id=`, and an
# unanchored match would return all three (as `exec_fp=` would also match `fp=`).
field() { # <line> <name>
    # The line may start with the field itself (`exec_uid=` after the prefix is
    # stripped), so pad one space and anchor on space-or-start; without the pad the
    # leading field can never match. `-*` keeps a negative value (`retained=-1`)
    # readable instead of collapsing to the empty string; every other value this
    # reads is decimal or hex.
    printf ' %s\n' "$1" | sed -n "s/.*[ =]$2=\(-*[0-9a-f][0-9a-f]*\).*/\1/p"
}

# The clean (reserve) line that belongs to the decode: `graph_uid=0` means the graph
# never passed through ggml_backend_sched_split_graph, which is true of the L2
# synthetic plan leg (`run_plan_leg` builds an 11-node graph directly on a gallocr).
# L2 runs AFTER the model leg and writes to the same stderr, so an unqualified tail
# would pick its line and compare it against the decode's identity line.
last_clean=$(printf '%s\n' "$on_lines" | grep -E "$RE_CLEAN" | grep -v 'graph_uid=0,' | tail -1)
clean_uid=$(field "$last_clean" graph_uid)
clean_pid=$(field "$last_clean" plan_id)
clean_fp=$(field  "$last_clean" fp)
# retained count the audit vouched for on that plan; the identity line must agree
clean_retained=$(printf '%s\n' "$last_clean" | sed -n 's/.*clean (reserve): \([0-9][0-9]*\) retained allocations.*/\1/p')

id_uid=$(field   "$id_last" exec_uid)
id_fp=$(field    "$id_last" exec_fp)
id_pid=$(field   "$id_last" plan_id)
id_retained=$(field "$id_last" retained)
id_audit_pid=$(field "$id_last" audit_plan_id)
id_match=$(field "$id_last" match)

if [ "$n_id" -eq 0 ]; then
    c4b_fail "no 'router_scratch plan identity' line on the ON leg: the owner did not join the submitted graph to a plan (UNPROVEN)"
elif [ "$id_match" != "1" ]; then
    c4b_fail "the owner reported match=${id_match:-<none>}: the plan the allocator holds is not the plan the decode submits (UNPROVEN)"
elif [ -z "$last_clean" ]; then
    c4b_fail "no well-formed 'retention audit clean (reserve): ... graph_uid= plan_id= fp=' line: nothing audited the plan (UNPROVEN)"
elif [ "$id_uid" != "$clean_uid" ] || [ "$id_pid" != "$clean_pid" ] || [ "$id_fp" != "$clean_fp" ]; then
    c4b_fail "identity mismatch: exec graph_uid=${id_uid:-<none>} plan_id=${id_pid:-<none>} fp=${id_fp:-<none>}" \
             "vs audit graph_uid=${clean_uid:-<none>} plan_id=${clean_pid:-<none>} fp=${clean_fp:-<none>}"
elif [ -z "$id_retained" ] || [ "$id_retained" -lt 2 ]; then
    c4b_fail "the submitted plan carries ${id_retained:-<none>} retained router projections, expected the 2 trunk MoE layers"
elif [ "$clean_retained" != "$id_retained" ]; then
    c4b_fail "the audit vouched for ${clean_retained:-<none>} retained allocations but the submitted plan carries ${id_retained}: the clean line is not this plan"
elif [ "$id_audit_pid" != "$id_pid" ]; then
    c4b_fail "the allocator's accepted plan (audit_plan_id=${id_audit_pid:-<none>}) is not the plan it holds" \
             "(plan_id=${id_pid:-<none>}): a later planning pass was not audited"
else
    note "  PASS: the audited plan is the submitted graph (graph_uid=$clean_uid plan_id=$clean_pid fp=$clean_fp retained=$clean_retained)"
fi

# Every multi-row submission must reach the same conclusion, not just the last one:
# `--rows 1,2,3,4` decodes 1 row (nothing retained, retained=0) and then 2/3/4 rows
# (2 retained projections each). Grading only the last line would let the other
# submissions drift. A negative `retained` means the pass that planned that
# submission was never audited, so it is a failure wherever it appears.
id_total=$(printf '%s\n' "$id_lines" | grep -c 'exec_uid=' || true)
id_retained_pos=$(printf '%s\n' "$id_lines" | grep 'exec_uid=' | grep -Ec ' retained=[1-9][0-9]*' || true)
id_match_one=$(printf '%s\n' "$id_lines" | grep 'exec_uid=' | grep -Ec ' match=1$' || true)
id_retained_neg=$(printf '%s\n' "$id_lines" | grep 'exec_uid=' | grep -Ec ' retained=-[0-9]' || true)
note "  ON identity: lines=$id_total retained>0=$id_retained_pos match=1=$id_match_one retained<0=$id_retained_neg"
note "  expected: 4 lines (rows 1,2,3,4), the 3 multi-row ones retained=2 match=1, none unaudited"
[ "$id_retained_neg" -eq 0 ] || c4b_fail "$id_retained_neg submission(s) were planned without an audit (retained=-1)"
[ "$id_retained_pos" -ge 3 ] || c4b_fail "only $id_retained_pos multi-row submission(s) carry retained scratch, expected 3 (rows 2,3,4)"
[ "$id_match_one" -eq "$id_retained_pos" ] || c4b_fail "$id_retained_pos submissions carry retained scratch but only $id_match_one reported match=1"

# OFF and 1-row legs: no retained plan may be claimed there. `match=1` is already
# impossible without retention, and the held-plan record is checked too, so a
# producer that dropped the retention condition could not slip through.
for leg in off_1 on_n1; do
    n_leg=$(grep -c 'router_scratch plan identity ' "$OUT/$leg.err" 2>/dev/null || true)
    # anchored: `match=` must not pick up the tail of another field
    leg_match=$(grep -h 'router_scratch plan identity ' "$OUT/$leg.err" 2>/dev/null | grep -Ec ' match=1$' || true)
    # `retained=` may be negative (nothing known) or 0; only a positive count is a claim
    leg_retained=$(grep -h 'router_scratch plan identity ' "$OUT/$leg.err" 2>/dev/null \
        | sed -n 's/.* retained=\(-*[0-9][0-9]*\).*/\1/p' | awk '$1 > 0' | wc -l)
    note "  $leg identity lines=$n_leg match=1 lines=$leg_match retained>0 lines=$leg_retained (expect 0)"
    [ "$leg_match" -eq 0 ] || c4b_fail "$leg reported match=1, so a plan was claimed without retention"
    [ "$leg_retained" -eq 0 ] || c4b_fail "$leg reported a retained plan, but nothing may be retained there"
done

diag_triage "cb_eval line is informational only: $diag_on"

note "== C5: plan-level disjointness and real hit (L2, synthetic) =="
# every check below reads on_1.txt: if the ON leg produced no plan summary, none of them
# may be evaluated (an empty file would otherwise read as "no overlap, no problem")
if [ "$(grep -c '^VERIFY_PLAN_SUMMARY' "$OUT/on_1.txt")" -eq 0 ]; then
    note "  FAIL: the ON leg produced no VERIFY_PLAN_SUMMARY: the L2 plan leg did not run"
    fail=1
fi
grep -E '^VERIFY_PLAN ' "$OUT/on_1.txt" | sed 's/^/    /' | tee -a "$LOG"
grep -q 'OVERLAPS retained block' "$OUT/on_1.txt" && { note "  FAIL: an allocation overlaps the retained extent"; fail=1; }
# L2 is a synthetic graph: its clean audit is necessary but never sufficient. C4b
# above is the promotion leg; it grades the model leg, not this one.
# a retain leg whose retained tensor never got storage proves nothing: fail closed
if grep -q 'materialized=0' "$OUT/on_1.txt"; then
    note "  FAIL: the retained tensor was not materialized (view or unallocated) — retain leg inconclusive"
    grep 'retention not materialized' "$OUT/on_1.txt" | sed 's/^/    /' | tee -a "$LOG"
    fail=1
fi
retained_bytes=$(grep -o 'retained_bytes=[0-9]*' "$OUT/on_1.txt" | tail -1 | cut -d= -f2)
note "  retained_bytes=$retained_bytes (512 x rows x 4 B)"
if [ -n "$retained_bytes" ] && [ "$retained_bytes" -ge 512 ]; then
    note "  PASS: the retained allocation is present in the live plan"
else
    note "  FAIL: no retained allocation in the plan"; fail=1
fi
grep -q 'computed=1' "$OUT/on_1.txt" || { note "  FAIL: the ON plan graph did not compute"; fail=1; }
grep -q 'OVERLAPS retained block' "$OUT/off_1.txt" && { note "  FAIL: OFF plan overlaps (allocator bug, not retention)"; fail=1; }

note "== C6: gallocr fail-closed audit printed no violation (AUDIT=1 on the ON leg) =="
# The allocator emits `retained tensor '<t>' [<op>] overlaps '<other>' [<op>] in buffer
# <n>`, `... overlaps leaf '<leaf>' ...` and `... is a view, it cannot be retained`.
# Match on the literal tail and quote-bounded name, not on `'.*'`: the grep this script
# runs under does not match that construct against these lines (measured).
RE_VIOLATION="retained tensor '[^']*' \[[^]]*\] (overlaps|is a view)|is a view, it cannot be retained"
if grep -qE "$RE_VIOLATION" "$OUT/on_1.err"; then
    note "  FAIL: gallocr audit reported a retention contract violation"; fail=1
else
    note "  PASS: no gallocr violation report"
fi
# `in buffer <n>` is the galloc buffer id, not a device
if grep -qE "$RE_VIOLATION" "$OUT/on_1.err"; then
    note "  triage: verbatim violation line(s) for the allocator owner:"
    grep -nE "$RE_VIOLATION" "$OUT/on_1.err" | sed 's/^/    /' | tee -a "$LOG"
fi
# the audit exits immediately on a violation, so any audit line AFTER it belongs to a
# later reserve; the retained tensor name in the violation is the reliable shape clue
if grep -qE "$RE_VIOLATION" "$OUT/on_1.err"; then
    viol_at=$(grep -nE "$RE_VIOLATION" "$OUT/on_1.err" | head -1 | cut -d: -f1)
    note "  triage: audit lines before the violation (line $viol_at) belong to an earlier reserve;"
    note "          the violated reserve printed none, so take the shape from the retained tensor name"
fi
# the INFO evidence lines were moved out of GGML_LOG_DEBUG, so they are visible in release
grep -q 'retention audit clean (reserve)' "$OUT/on_1.err" \
    && note "  note: audit clean line present (graded by C4b above)" \
    || note "  note: no clean line on the ON leg (see C4b output above)"

if [ "$MODEL_LEG_BLOCKED" = "1" ]; then
    note "== RESULT: FAIL — model leg skipped in diagnostic mode; L2 alone is not the L1 target =="
    note "log: $LOG"
    exit 3
fi

note "== RESULT: $([ $fail -eq 0 ] && echo 'CRITERIA MET' || echo 'FAIL — see above') =="
note "log: $LOG"
exit $fail
