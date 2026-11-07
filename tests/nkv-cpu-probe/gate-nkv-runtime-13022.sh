#!/usr/bin/env bash
# Gate for pillar 1: n_kv, n_outputs, rs_z and n_tokens as runtime parameters
# of ONE fixed-topology graph.
#
# Run ONLY on the authorized CPU build host (13022). Builds and runs:
#   - test-nkv-runtime-param-hash   (no model)  -> Gate HASH
#   - test-nkv-runtime-param        (model)     -> Gate LIVE
#   - test-union-four-param-hash    (no model)  -> Gate FOURHASH
#   - test-union-four-param-matrix  (model)     -> Gate MATRIX
#   - the five test-cpu-attn-pad arms (no model) -> Gate KERNEL
#   - test-union-binder-stock-equiv (model)     -> Gate BSE
#
# The four-parameter cross matrix is test-union-four-param-matrix and only
# that: it is the single registration of the n_kv x n_outputs x rs_z x
# n_tokens claim on one graph. The retired test-ntok-cross-matrix is NOT run
# here; its claim moved into the four-parameter matrix, and running both would
# be two tests of the same claim, which is how they drift apart.
#
# The KERNEL gate runs the five arms of test-cpu-attn-pad. HASH and
# LIVE prove the bind and the hash; MATRIX proves the two arms took different
# paths. None of them executes the CPU kernels that do the trim. A regression
# in the true-len reduction, in the softmax tail zeroing, or in the measured
# work counter would leave all three green. KERNEL closes that hole by
# requiring each arm's own PASS line, which the arm prints only after every
# assertion in it returned true.
#
#   MODEL=/path/to/tiny-dense-llama.gguf RUN_DIR=/path/to/new-dir \
#       bash tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh
#
# Configure and build run through /home/kunweiz/.local/bin/build-slot (override
# with BUILD_SLOT=...). The ctest runs stay outside the slot.
#
# The LIVE gate requires bind_steps>0 and distinct_bound>1 on purpose. A red
# LIVE gate with bind_steps=0 means the binder never bound below the build
# envelope for this model/context pair (the used span never fell short of the
# capacity, or the binder refused and rebuilt every step). That is a finding
# about the pair, not a licence to relax the regex: the hash is only proven to
# be re-taken when it was re-taken over real binds.
#
# RUN_DIR must not exist. Outputs:
#   identity.txt        revision, branch, model sha256, ccache, nproc, host
#   configure.log       cmake configure output
#   build.log           cmake build output
#   hash.log            test-nkv-runtime-param-hash (no model)
#   live.log            test-nkv-runtime-param (model, 300 steps + mask case)
#   fourhash.log        test-union-four-param-hash (no model)
#   matrix.log          test-union-four-param-matrix (model)
#   kernel.log          the five test-cpu-attn-pad arms (no model)
#   bse.log             test-union-binder-stock-equiv (model)
#   GATE_SUMMARY        GATE_HASH=... GATE_LIVE=... GATE_FOURHASH=...
#                       GATE_MATRIX=... GATE_KERNEL=... GATE_BSE=... each PASS|FAIL
#
# GATE_MATRIX reads the FOUR_MATRIX line and rejects, in this order:
#   - a line whose shape is not the pinned one (missing field, ok=0, a
#     non-hex hash, rebuilds != 0, stock_rebuilds == 0);
#   - bitwise=<a>/<b> with a != b (the two arms did not agree everywhere);
#   - bitwise=<a>/<b> with b != cells - nout_zero (the denominator is not the
#     cells that requested an output row). There is no separate "a < cells"
#     comparison: a == b and b == cells - nout_zero already pin both counts,
#     and the zero-output cell is excluded from both sides by construction.
#   - stock_rebuilds < cells (the stock arm did not rebuild per cell);
#   - rebuilds == stock_rebuilds (both arms took the same path);
#   - a missing FOUR_AXES line, a missing FOUR_MASK_NEG ok=1, a missing
#     FOUR_MASK_ROW_NEG ok=1, a missing FOUR_END ok=1, or any FAIL: line.
#     The two mask lines are separate on purpose: FOUR_MASK_NEG is the
#     conjunction of the fill's width guard and the read-back's corrupted-row
#     arm, so it stays ok=1 if the second arm is deleted. FOUR_MASK_ROW_NEG
#     is printed only by that second arm, and it is what goes missing.
# The two counts are compared in shell, not by a regex backreference: POSIX
# ERE has no backreferences, and GNU grep's \N is an extension a non-GNU grep
# reads as a literal N.
#
# A red MATRIX gate is a finding about the binder, not a licence to relax the
# regex.
#
# GATE_KERNEL is the only gate that executes the CPU kernels that do the trim.
# A green HASH/LIVE/FOURHASH/MATRIX says nothing about them.
# GATE_BSE reads the binder-vs-stock lines. It starts FAIL and becomes PASS
# only after every check below passed; a PASS default would report success for
# a block that was skipped or deleted. The checks, in order:
#   - BSE_KERNEL_SEL ok=0 -- the exact lane is only legitimate once kernel
#     selection is independent of the runtime length, and the test's probe for
#     that decoupling is fail-closed;
#   - BSE_STOCK_ARM -- the stock context must report NO binder runtime stats
#     (it holds no record), or the two arms are the same binder context;
#   - BSE_SINGLE -- the 300-step single-token arm: on_decode_builds == 1,
#     off_decode_builds >= steps and != 1, rebuild_fallbacks == 0,
#     pad_fallbacks == 0, and rows=a/b with a == b;
#   - BSE_MULTI -- the 65-step 4-token arm, same shape and same numeric
#     requirements, 260 rows;
#   - BSE_BITWISE -- rows=a/b with a == b and b == the sum of the two arms'
#     rows (a comparison that covered less than every produced row is not the
#     claim);
#   - BSE_NEG ok=1 differ>0 -- the falsifiability arm did not differ, so the
#     bitwise comparison cannot fail and a green bitwise line would be vacuous;
#   - a missing BSE_END ok=1, or any FAIL: line.
# The numeric comparisons are made in shell for the same reason as the MATRIX
# counts: POSIX ERE has no backreferences.
#
# A red BSE gate is a finding about the binder, not a licence to relax the
# regex. On a tree without the kernel-selection decoupling the gate is red by
# design and says so.
#
# Exit status, the tri-state contract shared with gate-verdict.sh:
#   0  PASS: all six gates passed (HASH, LIVE, FOURHASH, MATRIX, KERNEL, BSE
#      -- the same six keys GATE_SUMMARY carries).
#   1  FAIL: at least one gate failed an equality.
#   2  UNPROVEN: no gate failed, and at least one gate could not prove its
#      claim, or an input was missing. Nothing was compared, or the run could
#      not start. A caller must not read 2 as a failure of the code under test.
set -uo pipefail

# The tri-state vocabulary, resolved from this file.
_gate_verdict_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=tests/nkv-cpu-probe/gate-verdict.sh
. "$_gate_verdict_dir/gate-verdict.sh"
unset _gate_verdict_dir

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${MODEL:?set MODEL to a tiny CPU GGUF: a dense LLAMA for the two-axis matrix, or a MAMBA/RWKV/QWEN35 model for all three axes}"
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
if [[ -e $RUN_DIR ]]; then
    echo "UNPROVEN: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi
if [[ ! -f $MODEL || ! -s $MODEL ]]; then
    echo "UNPROVEN: MODEL missing or empty: $MODEL" >&2
    exit 2
fi
mkdir -- "$RUN_DIR" || { echo "UNPROVEN: cannot create RUN_DIR" >&2; exit 2; }
model=$(realpath -- "$MODEL")
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build

{
    printf 'source=%s\n' "$root"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'branch=';   git -C "$root" rev-parse --abbrev-ref HEAD
    printf 'model_path=%s\n' "$model"
    printf 'model_sha256='; sha256sum "$model" | cut -d' ' -f1
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
} > "$run_dir/identity.txt"

# Configure is not a compile: it runs outside build-slot. Every cmake --build
# goes through build-slot; the tests below run outside it too.
: "${BUILD_SLOT:=/home/kunweiz/.local/bin/build-slot}"
if [[ ! -x $BUILD_SLOT ]]; then
    echo "UNPROVEN: build-slot not executable: $BUILD_SLOT" >&2
    exit 2
fi

# Run one build-slot command, resumably. A 75 (queue cap) or 124 (run cap) is
# the environment refusing, not a result: retry, then report it as UNPROVEN.
# Returns 0 on success, 75/124 when the slot refused every attempt, and 1 on a
# real cmake/compiler error.
slot_run() { # log cmd...
    local log=$1
    shift
    local attempt rc last=0
    for attempt in $(seq 1 20); do
        printf '=== build-slot attempt %d at %s\n' "$attempt" "$(date --iso-8601=seconds)" >> "$log"
        "$BUILD_SLOT" "$@" >> "$log" 2>&1
        rc=$?
        printf 'BUILD_RC=%d attempt=%d\n' "$rc" "$attempt" >> "$log"
        last=$rc
        case $rc in
            0) return 0 ;;
            75) sleep 20 ;;
            124) : ;;
            *) return 1 ;;
        esac
    done
    return "$last"
}

# Configure is not a compile: it runs outside build-slot, whose only compiler
# work here would be cmake's seconds-long try_compile probes. A 75/124/143 is a
# transient environment signal, not a configure verdict: retry, then report it
# as UNPROVEN. Returns 0 on success, 75/124/143 when every attempt was
# refused, and 1 on a real cmake error.
configure_run() { # log args...
    local log=$1
    shift
    local attempt rc last=0
    for attempt in $(seq 1 3); do
        printf '=== configure attempt %d at %s\n' "$attempt" "$(date --iso-8601=seconds)" >> "$log"
        cmake "$@" >> "$log" 2>&1
        rc=$?
        printf 'CONFIGURE_RC=%d attempt=%d\n' "$rc" "$attempt" >> "$log"
        last=$rc
        case $rc in
            0) return 0 ;;
            75|124|143) sleep 5 ;;
            *) return 1 ;;
        esac
    done
    return "$last"
}

configure_run "$run_dir/configure.log" -S "$root" -B "$build_dir" \
        -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
        -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_VARLEN_TEST_MODEL="$model" \
        -DLLAMA_NTOK_TEST_MODEL="$model" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cfg_rc=$?
if [[ $cfg_rc -eq 75 || $cfg_rc -eq 124 || $cfg_rc -eq 143 ]]; then
    printf 'GATE_HASH=unproven GATE_LIVE=unproven GATE_FOURHASH=unproven GATE_MATRIX=unproven GATE_KERNEL=unproven GATE_BSE=unproven\n' > "$run_dir/GATE_SUMMARY"
    echo "UNPROVEN: configure transient rc=$cfg_rc; see $run_dir/configure.log" >&2
    exit 2
fi
if [[ $cfg_rc -ne 0 ]]; then
    printf 'GATE_HASH=FAIL GATE_LIVE=FAIL GATE_FOURHASH=FAIL GATE_MATRIX=FAIL GATE_KERNEL=FAIL GATE_BSE=FAIL\n' > "$run_dir/GATE_SUMMARY"
    echo "FAIL: configure; see $run_dir/configure.log" >&2
    exit 1
fi

slot_run "$run_dir/build.log" cmake --build "$build_dir" -j"$(nproc)" \
        --target test-nkv-runtime-param test-union-four-param-matrix test-cpu-attn-pad \
                test-union-binder-stock-equiv
bld_rc=$?
if [[ $bld_rc -eq 75 || $bld_rc -eq 124 ]]; then
    printf 'GATE_HASH=unproven GATE_LIVE=unproven GATE_FOURHASH=unproven GATE_MATRIX=unproven GATE_KERNEL=unproven GATE_BSE=unproven\n' > "$run_dir/GATE_SUMMARY"
    if [[ $bld_rc -eq 75 ]]; then
        echo "UNPROVEN: build build_slot_queue; see $run_dir/build.log" >&2
    else
        echo "UNPROVEN: build build_slot_timeout; see $run_dir/build.log" >&2
    fi
    exit 2
fi
if [[ $bld_rc -ne 0 ]]; then
    printf 'GATE_HASH=FAIL GATE_LIVE=FAIL GATE_FOURHASH=FAIL GATE_MATRIX=FAIL GATE_KERNEL=FAIL GATE_BSE=FAIL\n' > "$run_dir/GATE_SUMMARY"
    echo "FAIL: build; see $run_dir/build.log" >&2
    exit 1
fi

run_test() { # name log
    local name=$1 log=$2 rc=0
    local raw="$run_dir/ctest-$name.raw"
    # ctest -V prefixes every line of a test's own output with "<index>: ".
    # The gate's greps are anchored at the start of the line, so they can only
    # read the test's output after that prefix is stripped; without this every
    # anchored gate is red by construction, whatever the test printed. The
    # strip is plumbing, not a relaxation: every regex below is unchanged.
    env -i PATH="$PATH" HOME="$HOME" \
        ctest --test-dir "$build_dir" --no-tests=error \
              --output-on-failure -R "^${name}$" -V > "$raw" 2>&1 || rc=$?
    sed -E 's/^[[:space:]]*[0-9]+: //' "$raw" >> "$log"
    rm -f -- "$raw"
    return "$rc"
}

# Every gate starts UNPROVEN. A block promotes itself to PASS only after all
# of its checks ran and held. A FAIL is sticky.
gate_hash=$GATE_UNPROVEN
gate_live=$GATE_UNPROVEN
gate_fourhash=$GATE_UNPROVEN
gate_matrix=$GATE_UNPROVEN
gate_kernel=$GATE_UNPROVEN
gate_bse=$GATE_UNPROVEN
: > "$run_dir/hash.log"
: > "$run_dir/live.log"
: > "$run_dir/fourhash.log"
: > "$run_dir/matrix.log"
: > "$run_dir/kernel.log"

# ---------------------------------------------------------------- Gate HASH
# No model. The synthetic n_kv values include 64, 256, 257, 300 and 1024, so
# the hash is exercised below, at and above the 256-cell bucket; the n_outputs
# and n_rs values include 1, 257 and 300/301, so those two axes cross the same
# boundary. The cross product is 5 x 4 x 4 = 80 combinations.
gate_init
if ! run_test test-nkv-runtime-param-hash "$run_dir/hash.log"; then
    gate_fail hash_rc "Gate HASH: ctest failed" "$run_dir/hash.log"
fi
if ! grep -qE '^NKV_RT_HASH_ARM ok=1 cases=80 widths=5$' "$run_dir/hash.log"; then
    gate_unproven hash_arm "Gate HASH: no NKV_RT_HASH_ARM ok=1 cases=80 widths=5" "$run_dir/hash.log"
fi
if ! grep -qE '^NKV_RT_TRIPARAM ok=1 combos=80 roots=3$' "$run_dir/hash.log"; then
    gate_unproven hash_triparam "Gate HASH: no NKV_RT_TRIPARAM ok=1 combos=80 roots=3" "$run_dir/hash.log"
fi
if ! grep -qE '^NKV_RT_END ok=1$' "$run_dir/hash.log"; then
    gate_unproven hash_end "Gate HASH: no NKV_RT_END ok=1" "$run_dir/hash.log"
fi
if grep -qE '^FAIL:' "$run_dir/hash.log"; then
    gate_fail hash_fail "Gate HASH: FAIL line present" \
        "$(grep -E '^FAIL:' "$run_dir/hash.log" | tail -n 1)"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_hash=$gate_verdict

# ---------------------------------------------------------------- Gate LIVE
# Model. 300 steps cross 256. The test's own assertions are the gate; this
# script only re-reads the machine-readable lines so a silently truncated run
# cannot pass.
gate_init
if ! run_test test-nkv-runtime-param "$run_dir/live.log"; then
    gate_fail live_rc "Gate LIVE: ctest failed" "$run_dir/live.log"
fi
# bind_steps > 0 and distinct_bound > 1 are the D1 requirement: the hash must
# have been re-taken over real binds, not captured once at build time.
if ! grep -qE '^NKV_RT_LIVE ok=1 steps=300 hash=[0-9a-f]{16} min_width=1 max_width=300 bind_steps=[1-9][0-9]* distinct_bound=([2-9]|[1-9][0-9]+)$' "$run_dir/live.log"; then
    gate_unproven live_line "Gate LIVE: no NKV_RT_LIVE ok=1 steps=300 with min_width=1 max_width=300, bind_steps>0, distinct_bound>1" \
        "$run_dir/live.log"
fi
if ! grep -qE '^NKV_RT_END ok=1$' "$run_dir/live.log"; then
    gate_unproven live_end "Gate LIVE: no NKV_RT_END ok=1" "$run_dir/live.log"
fi
# The measured CPU attention work must follow the runtime width. On this lane
# the bind narrows the mask and the K/V views, so the ordered softmax executes
# exactly the runtime width and covers exactly that extent: the no-pad
# identity holds on every measured step. no_pad is a measured count, not a
# flag; a run whose ordered reduction never ran reports 0 here and fails, even
# if every width assertion above passed.
if ! grep -qE '^NKV_RT_TRUELEN ok=1 steps=300 no_pad=300 pad_steps=0$' "$run_dir/live.log"; then
    gate_unproven live_truelen "Gate LIVE: no NKV_RT_TRUELEN ok=1 with no_pad=300 pad_steps=0 (the ordered lane did not report the no-pad identity on every step)" \
        "$run_dir/live.log"
fi
if grep -qE '^FAIL:' "$run_dir/live.log"; then
    gate_fail live_fail "Gate LIVE: FAIL line present" \
        "$(grep -E '^FAIL:' "$run_dir/live.log" | tail -n 1)"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_live=$gate_verdict

# ------------------------------------------------------------ Gate FOURHASH
# No model. The four-parameter hash arm: the width-table hash is stable across
# 6 n_kv x 4 n_outputs x 6 n_rs x 7 token widths = 1008 combinations, and the
# SINGLE GRAPH CACHE KEY is stable across the seven token widths only with the
# token lane's exclusion. cases=1008 and widths=7 are pinned: a run that
# silently shrank the sweep must not pass.
#
# The counts are derived from the arrays in part1(), not guessed:
#   kv_v[]  = { 1, 8, 64, 256, 257, 300 }                    6
#   out_v[] = { 0, 1, 2, 257 }                               4
#   nrs_v[] = { 8, 9, 255, 256, 257, 260 }                   6
#   tok_v[] = { 1, 3, 2, 4, 7, 5, 300 }                      7   (n_tok_v derives from it)
#   widths[] in (b) and (c) is tok_v[]                       7
gate_init
: > "$run_dir/fourhash.log"
if ! run_test test-union-four-param-hash "$run_dir/fourhash.log"; then
    gate_fail fourhash_rc "Gate FOURHASH: ctest failed" "$run_dir/fourhash.log"
fi
if ! grep -qE '^FOUR_HASH_ARM ok=1 cases=1008 widths=7 key_cases=7$' "$run_dir/fourhash.log"; then
    gate_unproven fourhash_arm "Gate FOURHASH: no FOUR_HASH_ARM ok=1 cases=1008 widths=7 key_cases=7" \
        "$run_dir/fourhash.log"
fi
# The falsifiability arm: a runtime parameter that leaks into the hash must be
# refused. Without this line the equality sweep could pass vacuously.
if ! grep -qE '^FOUR_LEAK_NEG ok=1$' "$run_dir/fourhash.log"; then
    gate_unproven fourhash_leak "Gate FOURHASH: no FOUR_LEAK_NEG ok=1 (the hash is not falsifiable)" \
        "$run_dir/fourhash.log"
fi
if ! grep -qE '^FOUR_LEAK_NEG_CASE name=mask_rows .*refused=1$' "$run_dir/fourhash.log"; then
    gate_unproven fourhash_leak_mask "Gate FOURHASH: no FOUR_LEAK_NEG_CASE name=mask_rows with refused=1" \
        "$run_dir/fourhash.log"
fi
if ! grep -qE '^FOUR_LEAK_NEG_CASE name=token_dim armed_agree=1 disarmed_differ=1 ' "$run_dir/fourhash.log"; then
    gate_unproven fourhash_leak_token "Gate FOURHASH: no FOUR_LEAK_NEG_CASE name=token_dim with armed_agree=1 and disarmed_differ=1" \
        "$run_dir/fourhash.log"
fi
if ! grep -qE '^FOUR_END ok=1$' "$run_dir/fourhash.log"; then
    gate_unproven fourhash_end "Gate FOURHASH: no FOUR_END ok=1" "$run_dir/fourhash.log"
fi
if grep -qE '^FAIL:' "$run_dir/fourhash.log"; then
    gate_fail fourhash_fail "Gate FOURHASH: FAIL line present" \
        "$(grep -E '^FAIL:' "$run_dir/fourhash.log" | tail -n 1)"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_fourhash=$gate_verdict

# -------------------------------------------------------------- Gate MATRIX
# The four-parameter cross matrix (tests/test-union-four-param-matrix.cpp).
# The axis line must be present (rs_z=1 on a recurrent/hybrid model, rs_z=0
# with a reason on a dense one), the binder arm must report ZERO rebuilds, the
# stock arm must report at least one per cell, the two rebuild counts must
# differ, and every cell must be bitwise identical to the stock arm.
#
# rebuilds= and stock_rebuilds= are measured deltas of each context's own
# model.build_graph() counter, not a flag the test sets. Requiring both, and
# requiring them to differ, is what proves the two arms took different paths:
# a stock arm that reused the graph would report stock_rebuilds=0 and fail
# here, and two equal counts would mean both arms did the same thing.
gate_init
: > "$run_dir/matrix.log"
if ! run_test test-union-four-param-matrix "$run_dir/matrix.log"; then
    gate_fail matrix_rc "Gate MATRIX: ctest failed" "$run_dir/matrix.log"
fi
# The axis line carries the census: n_kv is the span of the executed width,
# n_outputs / rs_z / n_tokens are distinct-value counts. n_outputs and n_tokens
# must have moved (>= 2 distinct values each) or the matrix did not cross those
# axes; rs_z is 0 on a dense model (inert, with a reason) and >= 2 on a
# recurrent one. The counts are checked in shell below.
if ! grep -qE '^FOUR_AXES n_kv=[1-9][0-9]* n_outputs=[1-9][0-9]* rs_z=[0-9]+ n_tokens=[1-9][0-9]* reason=[a-z_]+$' "$run_dir/matrix.log"; then
    gate_unproven matrix_axes "Gate MATRIX: no FOUR_AXES line with the census" "$run_dir/matrix.log"
else
    axes_line=$(grep -E '^FOUR_AXES ' "$run_dir/matrix.log" | tail -n 1)
    a_nout=$(sed -E 's/^.* n_outputs=([0-9]+) .*$/\1/' <<< "$axes_line")
    a_rsz=$(sed -E 's/^.* rs_z=([0-9]+) .*$/\1/' <<< "$axes_line")
    a_ntok=$(sed -E 's/^.* n_tokens=([0-9]+) .*$/\1/' <<< "$axes_line")
    if [[ $a_nout -lt 2 ]]; then
        gate_fail matrix_axes_nout "Gate MATRIX: the n_outputs axis never moved" \
            "$a_nout distinct: $axes_line"
    fi
    if [[ $a_ntok -lt 2 ]]; then
        gate_fail matrix_axes_ntok "Gate MATRIX: the n_tokens axis never moved" \
            "$a_ntok distinct: $axes_line"
    fi
    if [[ $a_rsz -eq 1 ]]; then
        gate_fail matrix_axes_rsz "Gate MATRIX: rs_z=1 is not a census (0 = inert, >=2 = moved)" \
            "$axes_line"
    fi
fi
# The mask observation, in three parts, all required:
#
#   FOUR_MASK_NEG ok=1      part 3 proved the mask paths can FAIL, in two arms
#                           that reach different code. Arm (b): the fill's
#                           width guard aborts the child on an over-wide
#                           runtime n_kv, and the child's abort text names it.
#                           Arm (c): after a fill that succeeded, the child
#                           takes the live row (the row fetch must report
#                           success) and calls the public read-back
#                           (llama_union_mask_row_check, which runs
#                           check_mask_row); the read-back must return
#                           LLAMA_GATE_PASS on the clean row, LLAMA_GATE_FAIL
#                           after one column is corrupted (the child prints
#                           "the mask row disagrees with the KV cells" and
#                           exits 0), and LLAMA_GATE_PASS again after the
#                           column is restored. If the row is wider than the
#                           fill, the tail columns must be -INFINITY. A
#                           read-back that always refused fails the clean row;
#                           one that always accepted fails the corrupted row.
#   FOUR_MASK_UNPROVEN ok=1 part 3(d) proved the three verdicts are distinct
#                           and reachable. ok=0 is a FAIL; a missing line is
#                           UNPROVEN, because the arm did not run.
#   mask_cells=cells        the read-back ran on every cell of the matrix, and
#                           mask_rows=ok/checked agreed everywhere. This is
#                           what makes mask_n_kv == runtime_n_kv falsifiable:
#                           without the read-back both numbers come from the
#                           width the binder wrote.
if ! grep -qE '^FOUR_MASK_NEG ok=1$' "$run_dir/matrix.log"; then
    gate_unproven matrix_mask_neg "Gate MATRIX: no FOUR_MASK_NEG ok=1 (a mask failure path did not fire)" \
        "$run_dir/matrix.log"
fi
if grep -qE '^FOUR_MASK_UNPROVEN ok=0$' "$run_dir/matrix.log"; then
    gate_fail matrix_mask_unproven "Gate MATRIX: FOUR_MASK_UNPROVEN ok=0 (the tri-state arm did not prove the three verdicts)" \
        "$(grep -E '^FOUR_MASK_UNPROVEN ok=0$' "$run_dir/matrix.log" | tail -n 1)"
fi
# The read-back's own arm, on its own line. FOUR_MASK_NEG is the conjunction
# of both arms, so it stays ok=1 when the row arm is deleted; this line is the
# one that goes missing. Two arms, two evidence lines.
if ! grep -qE '^FOUR_MASK_ROW_NEG ok=1$' "$run_dir/matrix.log"; then
    gate_unproven matrix_mask_row_neg "Gate MATRIX: no FOUR_MASK_ROW_NEG ok=1 (the read-back cannot be shown to fail)" \
        "$run_dir/matrix.log"
fi
if ! grep -qE '^FOUR_MASK_UNPROVEN ok=1$' "$run_dir/matrix.log"; then
    gate_unproven matrix_mask_unproven "Gate MATRIX: no FOUR_MASK_UNPROVEN ok=1 (the tri-state arm did not run)" \
        "$run_dir/matrix.log"
fi
# The two bitwise counts are compared in shell, not by a regex backreference:
# POSIX ERE has no backreferences, and GNU grep's \N is an extension that a
# non-GNU grep reads as a literal N (or rejects the pattern outright). The
# shape check below is portable; every numeric comparison is made in shell.
#
# THE DENOMINATOR: bitwise=<a>/<b> counts only the cells that requested an
# output row. A cell with n_outputs = 0 produces two EMPTY logits buffers, and
# memcmp on two empty ranges returns 0, so counting it would be a comparison
# that compared nothing. The zero-output cell is therefore excluded from BOTH
# sides and reported as nout_zero=<k> instead. The invariant is
#
#     b == cells - nout_zero      and      a == b
#
# and it is checked below. Requiring b == cells would fail on every run that
# contains the zero-output cell, which four_cells() always does. The shape
# regex already requires both counts >= 1, so a 0/0 line is UNPROVEN here.
#
# THE MASK CENSUS: mask_cells must equal cells (the read-back ran on every
# cell), mask_rows must be a/b with a == b and b >= cells (at least one row
# compared per cell), and mask_cells_all must equal cells (every cell compared
# ALL of its rows, not only the last one). All three are checked in shell
# below; the regex only pins the shape.
#
# THE ROW DENOMINATOR: mask_expect is the number of rows the batches actually
# owned -- n_tokens * mask_filled summed over the cells, which the test
# accumulates as it walks them. `b >= cells` is satisfied by a read-back that
# compares exactly one row per cell, so it cannot show that every row was
# looked at; `b == mask_expect` can. Both are required: the equality pins the
# count, the lower bound still catches a run whose expectation itself
# collapsed.
matrix_line=$(grep -E '^FOUR_MATRIX ' "$run_dir/matrix.log" | tail -n 1 || true)
if ! grep -qE '^FOUR_MATRIX ok=1 cells=[1-9][0-9]* hash=[0-9a-f]{16} rebuilds=0 stock_rebuilds=[1-9][0-9]* bitwise=[1-9][0-9]*/[1-9][0-9]* nout_zero=[0-9]+ mask_cells=[1-9][0-9]* mask_rows=[1-9][0-9]*/[1-9][0-9]* mask_expect=[1-9][0-9]* mask_cells_all=[1-9][0-9]*$' "$run_dir/matrix.log"; then
    gate_unproven matrix_line "Gate MATRIX: no FOUR_MATRIX ok=1 with rebuilds=0, stock_rebuilds>0, bitwise=n/n, nout_zero=k and the mask census" \
        "$run_dir/matrix.log"
else
    m_cells=$(sed -E 's/^.* cells=([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_rebuilds=$(sed -E 's/^.* rebuilds=([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_stock=$(sed -E 's/^.* stock_rebuilds=([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_a=$(sed -E 's/^.* bitwise=([0-9]+)\/[0-9]+ .*$/\1/' <<< "$matrix_line")
    m_b=$(sed -E 's/^.* bitwise=[0-9]+\/([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_zero=$(sed -E 's/^.* nout_zero=([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_mcells=$(sed -E 's/^.* mask_cells=([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_mok=$(sed -E 's/^.* mask_rows=([0-9]+)\/[0-9]+ .*$/\1/' <<< "$matrix_line")
    m_mchk=$(sed -E 's/^.* mask_rows=[0-9]+\/([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_mexp=$(sed -E 's/^.* mask_expect=([0-9]+) .*$/\1/' <<< "$matrix_line")
    m_mall=$(sed -E 's/^.* mask_cells_all=([0-9]+)$/\1/' <<< "$matrix_line")
    m_expect=$(( m_cells - m_zero ))
    if [[ -z $m_cells || -z $m_a || -z $m_b || -z $m_zero || -z $m_mcells || -z $m_mok || -z $m_mchk || -z $m_mexp || -z $m_mall ]]; then
        gate_unproven matrix_fields "Gate MATRIX: a field did not parse" "$matrix_line"
    elif [[ $m_zero -lt 1 ]]; then
        gate_fail matrix_nout_zero "Gate MATRIX: no zero-output cell ran; the boundary is untested" \
            "nout_zero=$m_zero: $matrix_line"
    elif [[ $m_a != "$m_b" ]]; then
        gate_fail matrix_bitwise "Gate MATRIX: bitwise counts differ" \
            "$m_a/$m_b: $matrix_line"
    elif [[ $m_b -ne $m_expect ]]; then
        gate_fail matrix_bitwise_denominator "Gate MATRIX: bitwise denominator != cells - nout_zero" \
            "$m_b != $m_cells - $m_zero = $m_expect: $matrix_line"
    elif [[ $m_mcells -ne $m_cells ]]; then
        gate_fail matrix_mask_cells "Gate MATRIX: the mask read-back did not run on every cell" \
            "$m_mcells of $m_cells cells: $matrix_line"
    elif [[ $m_mchk -ne $m_mexp ]]; then
        gate_fail matrix_mask_expect "Gate MATRIX: the mask read-back compared a different number of rows than the batches own" \
            "$m_mchk rows, want $m_mexp: $matrix_line"
    elif [[ $m_mok -ne $m_mchk ]]; then
        gate_fail matrix_mask_rows "Gate MATRIX: the mask read-back disagreed on some rows" \
            "$m_mok of $m_mchk rows: $matrix_line"
    elif [[ $m_mchk -lt $m_cells ]]; then
        gate_fail matrix_mask_floor "Gate MATRIX: fewer mask rows compared than cells" \
            "$m_mchk rows, $m_cells cells: $matrix_line"
    elif [[ $m_mall -ne $m_cells ]]; then
        gate_fail matrix_mask_cells_all "Gate MATRIX: not every cell compared every mask row" \
            "$m_mall of $m_cells cells: $matrix_line"
    elif [[ $m_stock -lt $m_cells ]]; then
        gate_fail matrix_stock_rebuilds "Gate MATRIX: the stock arm rebuilt fewer times than cells" \
            "$m_stock time(s) for $m_cells cells: $matrix_line"
    elif [[ $m_rebuilds == "$m_stock" ]]; then
        gate_fail matrix_arms_same "Gate MATRIX: both arms rebuilt the same number of times; they took the same path" \
            "$m_rebuilds time(s): $matrix_line"
    fi
fi
# The zero-output cell must have its own evidence: the decode returned 0 and
# the binder bound n_outputs = 0. This is the claim the bitwise exclusion
# would otherwise leave unproven.
if ! grep -qE '^FOUR_NOUT_ZERO cell=[0-9]+ rc=0 bound_n_outputs=0 n_tokens=[1-9][0-9]*$' "$run_dir/matrix.log"; then
    gate_unproven matrix_nout_zero_line "Gate MATRIX: no FOUR_NOUT_ZERO line with rc=0 and bound_n_outputs=0" \
        "$run_dir/matrix.log"
fi
if ! grep -qE '^FOUR_END ok=1$' "$run_dir/matrix.log"; then
    gate_unproven matrix_end "Gate MATRIX: no FOUR_END ok=1" "$run_dir/matrix.log"
fi
if grep -qE '^FAIL:' "$run_dir/matrix.log"; then
    gate_fail matrix_fail_line "Gate MATRIX: FAIL line present" \
        "$(grep -E '^FAIL:' "$run_dir/matrix.log" | tail -n 1)"
fi
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_matrix=$gate_verdict
# -------------------------------------------------------------- Gate KERNEL
# The four arms of test-cpu-attn-pad, no model. HASH, LIVE and MATRIX prove
# the bind, the hash and the two arms' divergence, but none of them executes
# the CPU kernels that do the trim. A regression in the true-len reduction,
# in the softmax tail zeroing, or in the measured work counter would leave
# all three green.
#
# Each arm prints its PASS line only after every assertion in it returned
# true, so the line is the arm's own verdict. ctest already enforces the same
# line through PASS_REGULAR_EXPRESSION; the greps below re-read it from the
# log so a run whose ctest invocation was skipped cannot pass on exit 0 alone.
#
# The default arm prints two lines; only the second, which covers the
# 608-width sweep, is required here, because it is the one the registration
# names.
gate_init
for kernel_test in test-cpu-attn-true-len test-cpu-attn-true-len-counters \
                   test-cpu-attn-masked-gemm test-cpu-attn-f16-kqv test-cpu-attn-pad; do
    if ! run_test "$kernel_test" "$run_dir/kernel.log"; then
        gate_fail kernel_rc "Gate KERNEL: $kernel_test returned non-zero" "$run_dir/kernel.log"
    fi
done
while IFS= read -r kernel_line; do
    if ! grep -qF -- "$kernel_line" "$run_dir/kernel.log"; then
        gate_unproven kernel_pass_line "Gate KERNEL: missing PASS line: $kernel_line" "$run_dir/kernel.log"
    fi
done <<'EOF'
PASS: TRUE_LEN_REDUCE matches a live-width graph bitwise
PASS: TRUE_LEN_REDUCE counters follow the runtime length
PASS: masked-tail SGEMM reduces over the live length only
PASS: F16 ordered kqv rows match reference width bitwise
PASS: 608 widths, three reference widths each, guarded active rows
EOF
if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_kernel=$gate_verdict

# ----------------------------------------------------------------- Gate BSE
# The FINAL gate of the exact n_kv lane (tests/test-union-binder-stock-equiv.cpp):
# both positive arms must produce bitwise identical logits to a stock context at
# every row of every step, while building exactly ONE graph for the whole decode
# against the stock arm's one-per-step. Arm 1 is a 300-step single-token decode
# (300 rows); arm 2 is a 65-step 4-token decode at positions 0..259, all four
# rows marked, so its span crosses 256 at the last step (260 rows).
#
# BSE_KERNEL_SEL is the decoupling gate. The exact lane is only legitimate when
# kernel selection is independent of the runtime length. The test's probe for
# that is fail-closed, so on a tree WITHOUT the kernel-selection decoupling this
# gate is EXPECTED red with reason=kernel_sel_coupled. That is the correct
# verdict, not a bug to work around: do not relax the regex, and do not delete
# the check to make the summary green.
#
# The probe symbol must be defined in include/llama.h, the only llama header
# the test TU includes. A definition anywhere else (src/llama-union-binder.h
# included) cannot turn this gate green, and the test deliberately does not
# include the internal header. That is the fail-closed property: the macro has
# to be visible to a consumer of the public header, not smuggled into an
# internal one.
#
# Every numeric claim is compared in shell, not by a regex backreference, for
# the same reason as the MATRIX counts above. The capture groups in the sed
# expressions are load-bearing: without them the substitution matches but
# returns the whole line, and `bash -n` cannot see that.
#
# gate_bse starts UNPROVEN and is promoted to PASS only at the end of this
# block: an early return, a deleted block, or an unexecuted check leaves it
# UNPROVEN. An explicit ok=0 line is a FAIL.
#
# The pinned constants (300 rows / 300 steps, 260 rows / 65 steps) are the
# test's own design constants, not measurements: a shorter single-token run
# would not cross 256, and a shorter 4-token run would not reach 260. Pinning
# them here keeps a silently truncated run from passing.
gate_init
: > "$run_dir/bse.log"
if ! run_test test-union-binder-stock-equiv "$run_dir/bse.log"; then
    gate_fail bse_rc "Gate BSE: ctest test-union-binder-stock-equiv failed" "$run_dir/bse.log"
fi

# --- the decoupling gate. ok=1 only when the decoupling patch defines the probe
# symbol; reason names which answer was given. An explicit ok=0 is a FAIL; a
# missing line is UNPROVEN.
if ! grep -qE '^BSE_KERNEL_SEL ok=1 reason=kernel_sel_len_indep$' "$run_dir/bse.log"; then
    if grep -qE '^BSE_KERNEL_SEL ok=0 reason=kernel_sel_coupled$' "$run_dir/bse.log"; then
        gate_fail bse_kernel_sel_coupled \
            "Gate BSE: kernel selection still depends on the runtime length; the exact lane cannot be claimed" \
            "$run_dir/bse.log"
    else
        gate_unproven bse_kernel_sel "Gate BSE: no BSE_KERNEL_SEL ok=1 line" "$run_dir/bse.log"
    fi
fi

# --- the stock arm must not be a binder context. llama_get_nkv_runtime_stats()
# returns false there because no binder record is ever taken with graph reuse
# disabled; binder_stats=1 means the two arms are the same binder context and
# the comparison proves nothing.
if ! grep -qE '^BSE_STOCK_ARM ok=1 binder_stats=0$' "$run_dir/bse.log"; then
    gate_unproven bse_stock_arm \
        "Gate BSE: no BSE_STOCK_ARM ok=1 binder_stats=0 line; the stock arm is not proven distinct" \
        "$run_dir/bse.log"
fi

# --- the two positive arms: shape first, numbers below.
if ! grep -qE '^BSE_SINGLE ok=1 rows=[0-9]+/[0-9]+ steps=[0-9]+ on_decode_builds=1 off_decode_builds=[0-9]+ rebuild_fallbacks=0 pad_fallbacks=0$' "$run_dir/bse.log"; then
    gate_unproven bse_single \
        "Gate BSE: no BSE_SINGLE ok=1 line with on_decode_builds=1, rebuild_fallbacks=0, pad_fallbacks=0" \
        "$run_dir/bse.log"
fi
if ! grep -qE '^BSE_MULTI ok=1 rows=[0-9]+/[0-9]+ steps=[0-9]+ on_decode_builds=1 off_decode_builds=[0-9]+ rebuild_fallbacks=0 pad_fallbacks=0$' "$run_dir/bse.log"; then
    gate_unproven bse_multi \
        "Gate BSE: no BSE_MULTI ok=1 line with on_decode_builds=1, rebuild_fallbacks=0, pad_fallbacks=0" \
        "$run_dir/bse.log"
fi

s_a=; s_b=; s_steps=; s_off=
m_a=; m_b=; m_steps=; m_off=
bse_single=$(grep -E '^BSE_SINGLE ' "$run_dir/bse.log" | tail -n 1 || true)
if [[ -n $bse_single ]]; then
    s_a=$(sed -E 's/^.* rows=([0-9]+)\/[0-9]+ .*$/\1/' <<< "$bse_single")
    s_b=$(sed -E 's/^.* rows=[0-9]+\/([0-9]+) .*$/\1/' <<< "$bse_single")
    s_steps=$(sed -E 's/^.* steps=([0-9]+) .*$/\1/' <<< "$bse_single")
    s_off=$(sed -E 's/^.* off_decode_builds=([0-9]+) .*$/\1/' <<< "$bse_single")
    if [[ -z $s_a || -z $s_b || -z $s_steps || -z $s_off ]]; then
        gate_unproven bse_single_fields "Gate BSE: BSE_SINGLE fields did not parse" "$bse_single"
    else
        if [[ $s_a != "$s_b" ]]; then
            gate_fail bse_single_rows "Gate BSE: BSE_SINGLE rows differ" "$s_a/$s_b: $bse_single"
        fi
        if [[ $s_b -ne 300 || $s_steps -ne 300 ]]; then
            gate_fail bse_single_size "Gate BSE: BSE_SINGLE is not the 300-row 300-step arm" \
                "rows=$s_b steps=$s_steps: $bse_single"
        fi
        if [[ $s_off -lt $s_steps ]]; then
            gate_fail bse_single_off "Gate BSE: the single stock arm built fewer graphs than steps" \
                "$s_off graph(s) for $s_steps steps: $bse_single"
        fi
        if [[ $s_off -eq 1 ]]; then
            gate_fail bse_single_off_one "Gate BSE: the single stock arm built 1 graph; the two arms took the same path" \
                "$bse_single"
        fi
    fi
fi
bse_multi=$(grep -E '^BSE_MULTI ' "$run_dir/bse.log" | tail -n 1 || true)
if [[ -n $bse_multi ]]; then
    m_a=$(sed -E 's/^.* rows=([0-9]+)\/[0-9]+ .*$/\1/' <<< "$bse_multi")
    m_b=$(sed -E 's/^.* rows=[0-9]+\/([0-9]+) .*$/\1/' <<< "$bse_multi")
    m_steps=$(sed -E 's/^.* steps=([0-9]+) .*$/\1/' <<< "$bse_multi")
    m_off=$(sed -E 's/^.* off_decode_builds=([0-9]+) .*$/\1/' <<< "$bse_multi")
    if [[ -z $m_a || -z $m_b || -z $m_steps || -z $m_off ]]; then
        gate_unproven bse_multi_fields "Gate BSE: BSE_MULTI fields did not parse" "$bse_multi"
    else
        if [[ $m_a != "$m_b" ]]; then
            gate_fail bse_multi_rows "Gate BSE: BSE_MULTI rows differ" "$m_a/$m_b: $bse_multi"
        fi
        if [[ $m_b -ne 260 || $m_steps -ne 65 ]]; then
            gate_fail bse_multi_size "Gate BSE: BSE_MULTI is not the 260-row 65-step arm" \
                "rows=$m_b steps=$m_steps: $bse_multi"
        fi
        if [[ $m_off -lt $m_steps ]]; then
            gate_fail bse_multi_off "Gate BSE: the multi stock arm built fewer graphs than steps" \
                "$m_off graph(s) for $m_steps steps: $bse_multi"
        fi
        if [[ $m_off -eq 1 ]]; then
            gate_fail bse_multi_off_one "Gate BSE: the multi stock arm built 1 graph; the two arms took the same path" \
                "$bse_multi"
        fi
    fi
fi

# --- the aggregate bitwise line: every row of both arms, compared.
bse_bitwise=$(grep -E '^BSE_BITWISE ' "$run_dir/bse.log" | tail -n 1 || true)
if ! grep -qE '^BSE_BITWISE ok=1 rows=[0-9]+/[0-9]+$' "$run_dir/bse.log"; then
    gate_unproven bse_bitwise "Gate BSE: no BSE_BITWISE ok=1 rows=a/b line" "$run_dir/bse.log"
else
    bse_a=$(sed -E 's/^.* rows=([0-9]+)\/[0-9]+$/\1/' <<< "$bse_bitwise")
    bse_b=$(sed -E 's/^.* rows=[0-9]+\/([0-9]+)$/\1/' <<< "$bse_bitwise")
    if [[ -z $bse_a || -z $bse_b ]]; then
        gate_unproven bse_bitwise_fields "Gate BSE: BSE_BITWISE fields did not parse" "$bse_bitwise"
    elif [[ $bse_a != "$bse_b" ]]; then
        gate_fail bse_bitwise_rows "Gate BSE: bitwise counts differ" "$bse_a/$bse_b: $bse_bitwise"
    elif [[ -z ${s_a:-} || -z ${m_a:-} || $bse_a -ne $(( s_a + m_a )) ]]; then
        gate_fail bse_bitwise_sum "Gate BSE: BSE_BITWISE rows are not the sum of the two arms" \
            "$bse_a != ${s_a:-?} + ${m_a:-?}: $bse_bitwise"
    fi
fi

# --- the falsifiability arm. Without it a green bitwise line could be a
# comparison that cannot fail.
if ! grep -qE '^BSE_NEG ok=1 differ=[1-9][0-9]*$' "$run_dir/bse.log"; then
    gate_unproven bse_neg \
        "Gate BSE: no BSE_NEG ok=1 differ>0 line; the bitwise comparison is not falsifiable" \
        "$run_dir/bse.log"
fi
if ! grep -qE '^BSE_END ok=1$' "$run_dir/bse.log"; then
    gate_unproven bse_end "Gate BSE: no BSE_END ok=1" "$run_dir/bse.log"
fi
if grep -qE '^FAIL:' "$run_dir/bse.log"; then
    gate_fail bse_fail "Gate BSE: FAIL line present" \
        "$(grep -E '^FAIL:' "$run_dir/bse.log" | tail -n 1)"
fi

if [[ -z $gate_reason ]]; then
    gate_pass
fi
gate_bse=$gate_verdict

# ---------------------------------------------------------------- summary
verdict_upper() { # v
    case $1 in
        "$GATE_FAIL")     printf 'FAIL' ;;
        "$GATE_PASS")     printf 'PASS' ;;
        "$GATE_UNPROVEN") printf 'UNPROVEN' ;;
        *)                printf 'UNKNOWN' ;;
    esac
}

printf 'GATE_HASH=%s GATE_LIVE=%s GATE_FOURHASH=%s GATE_MATRIX=%s GATE_KERNEL=%s GATE_BSE=%s\n' \
    "$(verdict_upper "$gate_hash")" "$(verdict_upper "$gate_live")" \
    "$(verdict_upper "$gate_fourhash")" "$(verdict_upper "$gate_matrix")" \
    "$(verdict_upper "$gate_kernel")" "$(verdict_upper "$gate_bse")" > "$run_dir/GATE_SUMMARY"
cat "$run_dir/GATE_SUMMARY"

# The exit status is the worst of the six: FAIL > UNPROVEN > PASS.
overall=$GATE_PASS
for v in "$gate_hash" "$gate_live" "$gate_fourhash" "$gate_matrix" "$gate_kernel" "$gate_bse"; do
    if [[ $v -eq $GATE_FAIL ]]; then
        overall=$GATE_FAIL
        break
    fi
    if [[ $v -eq $GATE_UNPROVEN ]]; then
        overall=$GATE_UNPROVEN
    fi
done
# The exit status is not the enum value: PASS is 0, FAIL is 1, UNPROVEN is 2.
case $overall in
    "$GATE_PASS") exit 0 ;;
    "$GATE_FAIL") exit 1 ;;
    *)            exit 2 ;;
esac
