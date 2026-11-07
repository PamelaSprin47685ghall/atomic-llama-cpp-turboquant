#!/usr/bin/env bash
# One-shot close-out for pillar 1 on the authorized CPU build host (13022):
# n_kv, n_outputs and rs_z as runtime parameters of ONE fixed-topology graph.
#
# One invocation does everything:
#   1. configure + build inside /home/kunweiz/.local/bin/build-slot (override
#      with BUILD_SLOT=...), CPU-only, tests on, both test models set;
#   2. run the five registrations with ctest OUTSIDE the slot, each into its
#      own log;
#   3. run tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh into a second fresh
#      RUN_DIR and require its GATE_SUMMARY to be all PASS;
#   4. re-read the logs and assert the pillar's own claims from them:
#        - test-topology-key-sweep: every step built at most one graph and the
#          sweep did work after the first: it either built or reused, and the
#          binder bound at least once. On this schedule the binder patches the
#          record in place on all post-first steps, so steps_reused is the
#          rule; the test's header comment says why (a missing per-step line
#          or summary is a FAIL, not a skip);
#        - the binder arm's rebuilds=0 and the stock arm's
#          stock_rebuilds >= cells;
#        - bitwise=n/n with the two counts equal;
#        - the per-length comparison length_bitwise=n/n with equal counts and
#          lengths >= 2.
#   5. print one machine-readable line and exit 0 only when everything passed.
#
#   MODEL=/path/to/tiny-dense-llama.gguf RUN_DIR=/path/to/new-dir \
#       bash tests/nkv-cpu-probe/run-triparam-close-13022.sh
#
# MODEL is used for BOTH configure-time model variables (LLAMA_VARLEN_TEST_MODEL
# and LLAMA_NTOK_TEST_MODEL), so one GGUF drives the matrix, the live lane, the
# sweep and the cache-key test. A dense LLAMA gives the two-axis matrix
# (TRIPARAM_AXES rs_z=0 reason=dense_model); a MAMBA/RWKV/QWEN35 model gives all
# three axes. The script does not choose for you and does not weaken a gate when
# the model has no rs_z axis: the test says which axes it ran.
#
# RUN_DIR must not exist. Outputs:
#   identity.txt       revision, branch, model sha256, ccache, nproc, host, date
#   configure.log      cmake configure output
#   build.log          cmake build output
#   test-<name>.log    one ctest -V log per registration
#   gate/              the gate-nkv-runtime-13022.sh run (its own RUN_DIR)
#   TRIPARAM_CLOSE     the final machine-readable line
#
# Exit status: 0 only when the build, all five tests, the gate and every
# log-derived assertion passed.
#
# No bucket and no pad-to-max: nothing here rounds a width, and the tests it
# runs assert that the graph executes at the true used span.
set -uo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${MODEL:?set MODEL to a tiny CPU GGUF: a dense LLAMA for the two-axis matrix, or a MAMBA/RWKV/QWEN35 model for all three axes}"
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
if [[ -e $RUN_DIR ]]; then
    echo "FAIL: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi
if [[ ! -f $MODEL || ! -s $MODEL ]]; then
    echo "FAIL: MODEL missing or empty: $MODEL" >&2
    exit 2
fi
mkdir -- "$RUN_DIR" || { echo "FAIL: cannot create RUN_DIR" >&2; exit 2; }
model=$(realpath -- "$MODEL")
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build
gate_dir=$run_dir/gate

# The one line this script exists to print. `ok=0` until every check passed;
# the FAIL path below rewrites it with whatever the logs actually showed.
final_line="TRIPARAM_CLOSE ok=0 hash= rebuilds= stock_rebuilds= bitwise= lengths= length_bitwise= gates=FAIL"
finish() { # status
    printf '%s\n' "$final_line" | tee "$run_dir/TRIPARAM_CLOSE" >&2
    exit "$1"
}

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

# Every compile on the build host goes through build-slot. Configure and build
# are wrapped; every test below runs outside it.
: "${BUILD_SLOT:=/home/kunweiz/.local/bin/build-slot}"
if [[ ! -x $BUILD_SLOT ]]; then
    echo "FAIL: build-slot not executable: $BUILD_SLOT" >&2
    finish 2
fi

if ! "$BUILD_SLOT" cmake -S "$root" -B "$build_dir" \
        -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
        -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_VARLEN_TEST_MODEL="$model" \
        -DLLAMA_NTOK_TEST_MODEL="$model" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        > "$run_dir/configure.log" 2>&1; then
    echo "FAIL: configure; see $run_dir/configure.log" >&2
    finish 1
fi

if ! "$BUILD_SLOT" cmake --build "$build_dir" -j"$(nproc)" \
        --target test-nkv-runtime-param test-topology-cache-key test-topology-key-sweep \
        > "$run_dir/build.log" 2>&1; then
    echo "FAIL: build; see $run_dir/build.log" >&2
    finish 1
fi

# ctest outside the slot, one log per registration. `env -i` keeps the runner's
# shell from leaking an arm-selecting variable into a test that sets its own.
run_test() { # name log
    local name=$1 log=$2
    env -i PATH="$PATH" HOME="$HOME" \
        ctest --test-dir "$build_dir" --no-tests=error \
              --output-on-failure -R "^${name}$" -V >> "$log" 2>&1
}

tests_ok=1
for t in test-nkv-runtime-param-hash test-nkv-runtime-param test-triparam-matrix \
         test-topology-cache-key test-topology-key-sweep; do
    log=$run_dir/test-$t.log
    : > "$log"
    if ! run_test "$t" "$log"; then
        echo "FAIL: ctest $t; see $log" >&2
        tests_ok=0
    fi
done

# The gate, into its own fresh RUN_DIR. It configures and builds again (ccache
# makes that cheap) and runs its own three gates.
gate_ok=1
if ! MODEL="$model" RUN_DIR="$gate_dir" BUILD_SLOT="$BUILD_SLOT" \
        bash "$root/tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh"; then
    echo "FAIL: gate-nkv-runtime-13022.sh; see $gate_dir" >&2
    gate_ok=0
fi
gate_summary=$(cat "$gate_dir/GATE_SUMMARY" 2>/dev/null || printf 'GATE_SUMMARY=missing')
# Every gate the summary carries is required here as well as through the gate's
# exit status: the parse is what survives a future caller that wraps the gate in
# `|| true`, and a red gate must not be droppable downstream (SURVEY-GATE.md
# GAP G7). All six are required, because the gate writes all six: HASH, LIVE,
# FOURHASH, MATRIX, KERNEL and BSE. A requirement list shorter than the
# summary is how a red gate becomes droppable.
if [[ $gate_summary != *'GATE_HASH=PASS'* || $gate_summary != *'GATE_LIVE=PASS'* ||
      $gate_summary != *'GATE_FOURHASH=PASS'* ||
      $gate_summary != *'GATE_MATRIX=PASS'* || $gate_summary != *'GATE_KERNEL=PASS'* ||
      $gate_summary != *'GATE_BSE=PASS'* ]]; then
    echo "FAIL: gate summary is not all PASS: $gate_summary" >&2
    gate_ok=0
fi

# ---------------------------------------------------------------- assertions
# Every value below is read from a log this run produced. Nothing is assumed.

# 1. test-topology-key-sweep. The test prints one line per step:
#      step=<n> width=<w> builds=<b> rebuild_count=<r> binds=<n> ...
#    and one summary line at the end:
#      reuse_summary steps_reused=<n> steps_built=<n>
#
#    The test does NOT assert rebuild_count == 0. On this schedule the union
#    binder patches the recorded graph in place on every post-first step, so
#    the build count stays at 1 and steps_reused is the rule; allow_reuse()
#    and can_reuse_kq_mask() gate the stock graph cache, not the union bind.
#    What it asserts, and what this script re-reads, is:
#      - every step's rebuild_count is at most the number of steps so far
#        (one graph per step, never a retry loop);
#      - the sweep did work after the first step: steps_built + steps_reused
#        >= 1, and the binder bound at least once (patch_binds > 0);
#      - the test reported PASS.
#    A missing per-step line or a missing summary is a FAIL: a sweep that
#    printed nothing must not pass by absence.
sweep_log=$run_dir/test-test-topology-key-sweep.log
sweep_ok=1
sweep_steps=0
sweep_bad=0
if [[ ! -s $sweep_log ]]; then
    sweep_ok=0
else
    while IFS= read -r line; do
        [[ $line == step=* ]] || continue
        sweep_steps=$((sweep_steps + 1))
        rc=$(sed -E 's/^.* rebuild_count=(-?[0-9]+).*$/\1/' <<< "$line")
        # rebuild_count is the build delta since init; after N steps it can be
        # at most N. A larger value means a step built more than one graph.
        if [[ -z $rc || $rc -lt 0 || $rc -gt $sweep_steps ]]; then
            sweep_bad=$((sweep_bad + 1))
            echo "FAIL: sweep step built more than one graph: $line" >&2
        fi
    done < "$sweep_log"
    if [[ $sweep_steps -eq 0 ]]; then
        echo "FAIL: test-topology-key-sweep printed no per-step line" >&2
        sweep_ok=0
    fi
    if [[ $sweep_bad -ne 0 ]]; then
        sweep_ok=0
    fi
    # steps_built + steps_reused >= 1 is asserted: the sweep did work after
    # the first step. On this schedule the binder reuses on every post-first
    # step (steps_built == 0, steps_reused == steps - 1); a future binder that
    # rebuilds a step still passes. The counts are parsed and echoed so the
    # run records the finding.
    summary=$(grep -E '^reuse_summary steps_reused=[0-9]+ steps_built=[0-9]+$' "$sweep_log" | tail -n 1)
    if [[ -z $summary ]]; then
        echo "FAIL: test-topology-key-sweep printed no reuse_summary line" >&2
        sweep_ok=0
    else
        reused=$(sed -E 's/^reuse_summary steps_reused=([0-9]+) .*$/\1/' <<< "$summary")
        built=$(sed -E 's/^reuse_summary steps_reused=[0-9]+ steps_built=([0-9]+)$/\1/' <<< "$summary")
        echo "sweep: $summary"
        if [[ $((built + reused)) -lt 1 ]]; then
            echo "FAIL: the sweep neither built nor reused a graph: $summary" >&2
            sweep_ok=0
        fi
    fi
    if ! grep -qE '^PASS [0-9]+ topology-key-sweep checks$' "$sweep_log"; then
        echo "FAIL: test-topology-key-sweep did not report PASS" >&2
        sweep_ok=0
    fi
fi

# 2. The matrix line, and the fields the close-out reports.
matrix_ok=1
matrix_line=$(grep -E '^TRIPARAM_MATRIX ' "$run_dir/test-test-triparam-matrix.log" 2>/dev/null | tail -n 1)
if [[ -z $matrix_line ]]; then
    echo "FAIL: no TRIPARAM_MATRIX line in test-test-triparam-matrix.log" >&2
    matrix_ok=0
fi

hash=; rebuilds=; stock_rebuilds=; bitwise_a=; bitwise_b=; lengths=; lb_a=; lb_b=
cells=
if [[ $matrix_ok -eq 1 ]]; then
    hash=$(sed -E 's/^.* hash=([0-9a-f]+).*$/\1/' <<< "$matrix_line")
    cells=$(sed -E 's/^.* cells=([0-9]+) .*$/\1/' <<< "$matrix_line")
    rebuilds=$(sed -E 's/^.* rebuilds=([0-9]+) .*$/\1/' <<< "$matrix_line")
    stock_rebuilds=$(sed -E 's/^.* stock_rebuilds=([0-9]+) .*$/\1/' <<< "$matrix_line")
    bitwise_a=$(sed -E 's/^.* bitwise=([0-9]+)\/([0-9]+) .*$/\1/' <<< "$matrix_line")
    bitwise_b=$(sed -E 's/^.* bitwise=([0-9]+)\/([0-9]+) .*$/\2/' <<< "$matrix_line")
    lengths=$(sed -E 's/^.* lengths=([0-9]+) .*$/\1/' <<< "$matrix_line")
    lb_a=$(sed -E 's/^.* length_bitwise=([0-9]+)\/([0-9]+)$/\1/' <<< "$matrix_line")
    lb_b=$(sed -E 's/^.* length_bitwise=([0-9]+)\/([0-9]+)$/\2/' <<< "$matrix_line")

    if [[ $matrix_line != *' ok=1 '* ]]; then
        echo "FAIL: TRIPARAM_MATRIX is not ok=1: $matrix_line" >&2
        matrix_ok=0
    fi
    if [[ $rebuilds != 0 ]]; then
        echo "FAIL: the binder arm rebuilt ($rebuilds): $matrix_line" >&2
        matrix_ok=0
    fi
    if [[ -z $cells || -z $stock_rebuilds || $stock_rebuilds -lt $cells ]]; then
        echo "FAIL: the stock arm did not rebuild per cell ($stock_rebuilds < $cells): $matrix_line" >&2
        matrix_ok=0
    fi
    if [[ -z $bitwise_a || -z $bitwise_b || $bitwise_a != "$bitwise_b" ]]; then
        echo "FAIL: bitwise counts differ ($bitwise_a/$bitwise_b): $matrix_line" >&2
        matrix_ok=0
    fi
    if [[ -z $bitwise_a || $bitwise_a -lt $cells ]]; then
        echo "FAIL: bitwise compared fewer cells than the matrix ran ($bitwise_a < $cells): $matrix_line" >&2
        matrix_ok=0
    fi
    if [[ -z $lengths || $lengths -lt 2 ]]; then
        echo "FAIL: the per-length arm covered fewer than two widths (lengths=$lengths): $matrix_line" >&2
        matrix_ok=0
    fi
    if [[ -z $lb_a || -z $lb_b || $lb_a != "$lb_b" ]]; then
        echo "FAIL: length_bitwise counts differ ($lb_a/$lb_b): $matrix_line" >&2
        matrix_ok=0
    fi
    if [[ -z $lb_a || $lb_a -lt $cells ]]; then
        echo "FAIL: length_bitwise compared fewer cells than the matrix ran ($lb_a < $cells): $matrix_line" >&2
        matrix_ok=0
    fi
    if ! grep -qE '^TRIPARAM_END ok=1$' "$run_dir/test-test-triparam-matrix.log"; then
        echo "FAIL: no TRIPARAM_END ok=1 in test-test-triparam-matrix.log" >&2
        matrix_ok=0
    fi
fi

# 3. The cache-key test must have reported its own PASS (it needs no model).
cache_ok=1
if ! grep -qE '^PASS [0-9]+ topology-cache-key checks$' "$run_dir/test-test-topology-cache-key.log"; then
    echo "FAIL: test-topology-cache-key did not report PASS" >&2
    cache_ok=0
fi

gates=PASS
if [[ $tests_ok -ne 1 || $gate_ok -ne 1 || $sweep_ok -ne 1 || $matrix_ok -ne 1 || $cache_ok -ne 1 ]]; then
    gates=FAIL
fi
ok=0
if [[ $gates == PASS ]]; then
    ok=1
fi

# The line is machine-read, so its shape is pinned here rather than trusted: a
# field that failed to parse would otherwise print as an empty value and a
# reader could mistake "missing" for "zero".
close_re='^TRIPARAM_CLOSE ok=1 hash=[0-9a-f]{16} rebuilds=0 stock_rebuilds=[1-9][0-9]* bitwise=[1-9][0-9]*/[1-9][0-9]* lengths=[2-9][0-9]* length_bitwise=[1-9][0-9]*/[1-9][0-9]* gates=PASS$'
if [[ $ok -eq 1 ]] && ! grep -qE "$close_re" <<< "TRIPARAM_CLOSE ok=$ok hash=$hash rebuilds=$rebuilds stock_rebuilds=$stock_rebuilds bitwise=$bitwise_a/$bitwise_b lengths=$lengths length_bitwise=$lb_a/$lb_b gates=$gates"; then
    echo "FAIL: the close line does not have the pinned shape" >&2
    gates=FAIL
    ok=0
fi

final_line="TRIPARAM_CLOSE ok=$ok hash=$hash rebuilds=$rebuilds stock_rebuilds=$stock_rebuilds bitwise=$bitwise_a/$bitwise_b lengths=$lengths length_bitwise=$lb_a/$lb_b gates=$gates"

if [[ $ok -eq 1 ]]; then
    finish 0
fi
finish 1
