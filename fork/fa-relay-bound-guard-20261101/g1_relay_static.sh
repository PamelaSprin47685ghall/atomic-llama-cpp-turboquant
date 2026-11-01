#!/bin/bash
# G1 static gate for the FA src[5] RELAY-BOUND lane.
#
#   bash g1_relay_static.sh [SRC_TREE] [DEST_TREE]
#
# Runs on CPU only: no compiler, no GPU, no model. Every check prints one line and the script
# prints exactly one terminal line, "STATIC-OK" or "STATIC-FAIL <reason>", and exits 0 only on
# STATIC-OK. A step that cannot run is a FAIL, never a skip.
#
# Author note: this script has never been executed by its author (no compile/test rule). Devops
# runs it; see LIMITS.md for what it can and cannot establish.
set -u

SRC=${1:-/home/kunweiz/work/merge-fa-20261031}
HERE=$(cd "$(dirname "$0")" && pwd)
DST=${2:-$HERE/relay-patch}
GEN=$HERE/prepare_relay_bound_guard.py
LOG=$(mktemp)
trap 'rm -f "$LOG" "$LOG.cu" "$LOG.hip" "$LOG.src.hip"' EXIT

say()  { printf '%s\n' "$*" >>"$LOG"; }
fail() { say "FAIL: $*"; printf 'STATIC-FAIL %s\n' "$*"; exit 1; }

[ -f "$GEN" ] || fail "generator missing: $GEN"
[ -d "$SRC" ] || fail "source tree missing: $SRC"

# 1. generator: pinned-hash copy + anchored edits. Any drift or double anchor aborts non-zero.
say "== generator"
python3 "$GEN" "$SRC" "$DST" >>"$LOG" 2>&1 || fail "generator refused (see log)"

# 2. the patched files must exist and differ from the source exactly where claimed
say "== patched set"
for f in ggml/src/ggml-cuda/fattn-common.cuh ggml/src/ggml-cuda/relay.hip; do
    [ -f "$DST/$f" ] || fail "missing patched file $f"
    cmp -s "$SRC/$f" "$DST/$f" && fail "$f was not modified"
done
# relay.hip is the only file whose non-lane content must be untouched: the patch adds one block
# in submit() and changes nothing else. Count the added lines outside that block.
[ -s "$DST/CANDIDATE.diff" ] || fail "CANDIDATE.diff missing/empty"

# Code-only view of a file: drop whole-line // comments and trailing // comments. Every count
# below uses it, so documentation text can never satisfy (or break) a structural check.
codeview() { sed -e 's@[[:space:]]*//.*$@@' "$1"; }

# 3. lane symbols, counted PER FILE and PER ROLE, on the code-only view. A single aggregate count
#    is not acceptable: an earlier version summed a definition, a dead helper and one call to
#    reach its expected total while the other call site was never counted at all.
say "== symbols"
_cu=$DST/ggml/src/ggml-cuda/fattn-common.cuh
_hip=$DST/ggml/src/ggml-cuda/relay.hip
codeview "$_cu" >"$LOG.cu"; codeview "$_hip" >"$LOG.hip"

# 3a. the activation predicate exists exactly once (definition) and is CALLED at least once.
n=$(grep -cE '^[^/]*bool ggml_cuda_fattn_relay_bound_active\(' "$LOG.cu")
[ "$n" -eq 1 ] || fail "relay_bound_active must be DEFINED once in fattn-common.cuh (got $n)"
n=$(grep -c 'ggml_cuda_fattn_relay_bound_active(' "$LOG.cu")
[ "$n" -ge 2 ] || fail "relay_bound_active must be called at least once besides its definition (got $n)"

# 3b. every call of the activation predicate must be a live call, i.e. it must appear in a
#     conditional or a return, never only inside a helper nobody calls.
n=$(grep -c 'if (ggml_cuda_fattn_relay_bound_active(' "$LOG.cu")
[ "$n" -ge 1 ] || fail "relay_bound_active is never used in a branch (dead code)"

# 3c. the env gate is read exactly once (inside its own definition). In fattn-common.cuh the
#     predicate therefore has exactly 2 live occurrences in code: its definition line and the
#     single call inside active(). relay.hip's submission scan calls it once, counted separately.
n=$(grep -c 'getenv("GGML_TP5_FA_RELAY_BOUND")' "$LOG.cu")
[ "$n" -eq 1 ] || fail "GGML_TP5_FA_RELAY_BOUND must be read once, in fattn-common.cuh (got $n)"
n=$(grep -c 'ggml_cuda_fattn_relay_bound_enabled()' "$LOG.cu")
[ "$n" -eq 2 ] || fail "fattn-common.cuh must mention relay_bound_enabled twice in code (definition + the call inside active()), got $n"
n=$(grep -c 'ggml_cuda_fattn_relay_bound_enabled()' "$LOG.hip")
[ "$n" -eq 1 ] || fail "relay.hip must CALL relay_bound_enabled exactly once (got $n)"

# 3d. the classifier: one definition, and one call at each boundary (launch_fattn + submit).
n=$(grep -cE '^[^/]*bool ggml_cuda_fattn_relay_bound_in_graph\(' "$LOG.cu")
[ "$n" -eq 1 ] || fail "in_graph classifier must be DEFINED once (got $n)"
n=$(grep -c 'ggml_cuda_fattn_relay_bound_in_graph(' "$LOG.cu")
[ "$n" -eq 2 ] || fail "in_graph must be defined once and called once in fattn-common.cuh (got $n)"
n=$(grep -c 'ggml_cuda_fattn_relay_bound_in_graph(' "$LOG.hip")
[ "$n" -eq 1 ] || fail "relay submit gate must classify exactly once in code (got $n)"

# 4. the relay gate must sit before begin(): no epoch, no capture, no launch on refusal.
#    Only the gate's own guard counts, so a mention of the classifier in a comment is ignored.
say "== ordering"
awk '/^bool submit\(void \* opaque/{insubmit=1}
     insubmit&&/ggml_cuda_fattn_relay_bound_in_graph\(node\)/{gate=NR}
     insubmit&&/^    if \(!begin\(p\)\)/{begin=NR; exit}
     END{exit !(gate&&begin&&gate<begin)}' "$LOG.hip" || fail "relay gate is not before begin()"

# 5. the device check must precede every launch, every pool allocation and every K/V conversion
#    inside launch_fattn. The check is identified by its own guard line, and the first
#    "to_fp16(" / "ggml_cuda_kernel_launch(" / pool alloc after it must not precede it.
say "== launch ordering"
awk '/^void launch_fattn\(/{inl=1}
     inl&&/if \(ggml_cuda_fattn_relay_bound_active\(dst\)\)/{chk=NR}
     inl&&/to_fp16\(/{conv=NR}
     inl&&/ggml_cuda_kernel_launch\(/{kl=NR}
     inl&&/pool_alloc</{pa=NR}
     END{exit !(chk&&conv&&kl&&pa&&chk<conv&&chk<kl&&chk<pa)}' "$LOG.cu" \
    || fail "device check is not before pool allocation/conversion/first launch"

# 6. no D2H and no stream synchronization may be introduced by this lane
say "== no host round trip"
if grep -q "cudaMemcpy.*DeviceToHost\|cudaStreamSynchronize\|cudaStreamQuery\|hipStreamSynchronize\|hipMemcpy.*DeviceToHost" "$LOG.cu"; then
    fail "a host round trip appeared in fattn-common.cuh"
fi
# The pinned relay.hip mentions hipPointerGetAttributes once in code (prepare, relay.hip:1055); the
# lane must not change that count. Compared in the code view on both sides.
codeview "$SRC/ggml/src/ggml-cuda/relay.hip" >"$LOG.src.hip"
[ "$(grep -c 'hipPointerGetAttributes' "$LOG.hip")" -eq \
  "$(grep -c 'hipPointerGetAttributes' "$LOG.src.hip")" ] \
  || fail "the relay lane added a device attribute query"

# 7. the gate must be opt-in: an exact-"1" env read, never a default-on
say "== opt-in"
grep -q 'getenv("GGML_TP5_FA_RELAY_BOUND")' "$LOG.cu" \
    || fail "gate is not an exact env read"
grep -q 'strcmp(env, "1") == 0' "$LOG.cu" \
    || fail "gate is not exact-\"1\""

# 8. NaN must not be used as the refusal signal. The patched header contains no NaN literal at all
#    (measured on the pinned file: 0 matches), so this asserts absence rather than attributing it.
say "== no NaN signal"
if grep -q "NAN\|nanf\|0x7fc00000" "$LOG.cu"; then
    fail "relay lane emits a NaN as its refusal signal"
fi

# 9. the stock contract lines must be untouched
say "== stock untouched"
grep -q "GGML_TP5_FA_NKV_PARAM: runtime KV length %d exceeds built KV width %d" \
    "$LOG.cu" || fail "stock op-param abort was removed"
grep -q "GGML_ASSERT(n_kv_raw >= 0 && n_kv_raw <= nek1);" "$SRC/ggml/src/ggml-cpu/ops.cpp" \
    || fail "CPU consumer contract missing in the pinned tree"

printf 'STATIC-OK\n'
