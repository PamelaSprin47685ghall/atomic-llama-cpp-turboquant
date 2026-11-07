#!/usr/bin/env bash
# Gate for the kernel-selection length independence of the CPU non-FA lane.
#
# The claim under test: for a fixed topology the CPU kernel variant selected
# for a node depends only on topology constants (types, head_dim, n_embd,
# block size, ISA, precision, thread count), never on the runtime KV length.
# tests/test-cpu-kernel-sel-len-indep.cpp is the falsifiable form of it:
#
#   STOCK a graph built at REF_W and then PATCHED in place down to width n
#         (ne/nb rewritten, no flags) must equal a graph built at n. A complete
#         patch is an identity by construction, so this arm checks that the
#         patch is complete; the substantive equality claim rests on the
#         selection-site table in ANALYSIS-EQUALITY.md section C.
#   STRIDE the same patch with ne[] narrowed and nb[] left at the envelope
#         stride must be DETECTED (ggml asserts on the inconsistent layout, so
#         this arm runs in a child process and an abort counts as detection).
#   OLD   two unmarked widths straddling a tinyBLAS tile boundary
#         (m % 16 / 8 / 4, m = ne01) must DIFFER somewhere (otherwise STOCK
#         cannot see a selection difference and the test fails by name)
#
# The test carries no marking at all: REDUCE_PAD_TAIL / MASKED_TAIL_GEMM are
# gone from the n_kv lane, so an arm that exercised them would prove nothing.
#
# Run ONLY on the authorized CPU build host. Configure and build go through
# build-slot; the test runs outside it.
#
#   RUN_DIR=/path/to/new-dir bash tests/nkv-cpu-probe/gate-kernel-sel-len-13022.sh
#
# RUN_DIR must not exist. Outputs:
#   identity.txt    revision, branch, ccache, nproc, host, date
#   configure.log   cmake configure output
#   build.log       cmake build output
#   kernel_sel.log  the test's own output
#   GATE_SUMMARY    GATE_KERNEL_SEL=PASS|FAIL
#
# Exit status:
#   0  the test ran and reported ok=1
#   1  a real failure (configure, build, or the test)
#   2  a missing input or an existing RUN_DIR
#   3  NOT_RUN: build-slot refused to queue (75) or asked for a narrower
#      build (124). Neither is a test failure, and neither is a pass.
#
# A red gate is a finding about the kernel selection, not a licence to relax
# the assertions.
set -uo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
if [[ -e $RUN_DIR ]]; then
    echo "FAIL: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi
mkdir -- "$RUN_DIR" || { echo "FAIL: cannot create RUN_DIR" >&2; exit 2; }
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build

write_summary() { # value
    printf 'GATE_KERNEL_SEL=%s\n' "$1" > "$run_dir/GATE_SUMMARY"
}

{
    printf 'source=%s\n' "$root"
    printf 'revision='; git -C "$root" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
    printf 'branch=';   git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown\n'
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
} > "$run_dir/identity.txt"

: "${BUILD_SLOT:=/home/kunweiz/.local/bin/build-slot}"
if [[ ! -x $BUILD_SLOT ]]; then
    echo "FAIL: build-slot not executable: $BUILD_SLOT" >&2
    write_summary NOT_RUN
    exit 3
fi

"$BUILD_SLOT" cmake -S "$root" -B "$build_dir" \
    -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
    -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
    -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    > "$run_dir/configure.log" 2>&1
rc=$?
if [[ $rc -eq 75 ]]; then
    echo "NOT_RUN: build-slot refused to queue (rc=75)" >&2
    write_summary NOT_RUN
    exit 3
fi
if [[ $rc -eq 124 ]]; then
    echo "NOT_RUN: build-slot asked for a narrower build (rc=124)" >&2
    write_summary NOT_RUN
    exit 3
fi
if [[ $rc -ne 0 ]]; then
    echo "FAIL: configure (rc=$rc); see $run_dir/configure.log" >&2
    write_summary FAIL
    exit 1
fi

"$BUILD_SLOT" cmake --build "$build_dir" -j2 \
    --target test-cpu-kernel-sel-len-indep \
    > "$run_dir/build.log" 2>&1
rc=$?
if [[ $rc -eq 75 ]]; then
    echo "NOT_RUN: build-slot refused to queue (rc=75)" >&2
    write_summary NOT_RUN
    exit 3
fi
if [[ $rc -eq 124 ]]; then
    echo "NOT_RUN: build-slot asked for a narrower build (rc=124)" >&2
    write_summary NOT_RUN
    exit 3
fi
if [[ $rc -ne 0 ]]; then
    echo "FAIL: build (rc=$rc); see $run_dir/build.log" >&2
    write_summary FAIL
    exit 1
fi

# The test runs outside the slot: it is a CPU-only unit test with no build.
#
# ctest -V prefixes every line of a test's own output with "<index>: ". The
# greps below are anchored at the start of the line, so the log they read is
# the stripped one; without this no KERNEL_SEL_ line could ever match and the
# gate would be red by construction. The strip is plumbing: every regex is
# unchanged.
env -i PATH="$PATH" HOME="$HOME" \
    ctest --test-dir "$build_dir" --no-tests=error \
          --output-on-failure -R '^test-cpu-kernel-sel-len-indep$' -V \
          > "$run_dir/kernel_sel.log.raw" 2>&1
rc=$?
sed -E 's/^[[:space:]]*[0-9]+: //' "$run_dir/kernel_sel.log.raw" > "$run_dir/kernel_sel.log"
rm -f -- "$run_dir/kernel_sel.log.raw"
if [[ $rc -ne 0 ]]; then
    echo "FAIL: test-cpu-kernel-sel-len-indep (rc=$rc); see $run_dir/kernel_sel.log" >&2
    write_summary FAIL
    exit 1
fi

# The summary line is the contract; a run without it proved nothing.
#
#   stock=<a>/<r>   a complete patch must agree on EVERY case: a == r, r > 0
#   stride=<d>/<r>  an incomplete patch (ne narrowed, nb left at the envelope
#                   stride) must be DETECTED on at least one case: d >= 1
#   old=<d>/<r>     two widths straddling a tinyBLAS tile class must differ on
#                   at least one case: d >= 1
#
# The gate does NOT accept "most cases agreed": stock must be exactly a == r.
# bash ERE has no backreferences, so the equality is asserted here with two
# captures and a string compare rather than in one regex.
if ! grep -qE '^KERNEL_SEL_END ok=1 stock=[0-9]+/[0-9]+ stride=[0-9]+/[0-9]+ old=[0-9]+/[0-9]+$' \
        "$run_dir/kernel_sel.log"; then
    echo "FAIL: no KERNEL_SEL_END ok=1 line with three d/r arms; see $run_dir/kernel_sel.log" >&2
    write_summary FAIL
    exit 1
fi
end_line=$(grep -E '^KERNEL_SEL_END ' "$run_dir/kernel_sel.log" | tail -n 1)
stock_a=$(sed -E 's/.* stock=([0-9]+)\/([0-9]+) .*/\1/' <<<"$end_line")
stock_r=$(sed -E 's/.* stock=([0-9]+)\/([0-9]+) .*/\2/' <<<"$end_line")
stride_d=$(sed -E 's/.* stride=([0-9]+)\/([0-9]+) .*/\1/' <<<"$end_line")
stride_r=$(sed -E 's/.* stride=([0-9]+)\/([0-9]+) .*/\2/' <<<"$end_line")
old_d=$(sed -E 's/.* old=([0-9]+)\/([0-9]+)$/\1/' <<<"$end_line")
old_r=$(sed -E 's/.* old=([0-9]+)\/([0-9]+)$/\2/' <<<"$end_line")
if [[ -z $stock_a || -z $stock_r || -z $stride_d || -z $stride_r || -z $old_d || -z $old_r ]]; then
    echo "FAIL: could not parse the arm counts from: $end_line" >&2
    write_summary FAIL
    exit 1
fi
if [[ $stock_r -eq 0 ]]; then
    echo "FAIL: stock arm ran no cases" >&2
    write_summary FAIL
    exit 1
fi
if [[ $stock_a != "$stock_r" ]]; then
    echo "FAIL: a complete patch is not an identity: stock=${stock_a}/${stock_r}" >&2
    write_summary FAIL
    exit 1
fi
if [[ $stride_r -eq 0 || $stride_d -eq 0 ]]; then
    echo "FAIL: an incomplete patch was not detected: stride=${stride_d}/${stride_r}" >&2
    write_summary FAIL
    exit 1
fi
if [[ $old_r -eq 0 || $old_d -eq 0 ]]; then
    echo "FAIL: the tile-boundary falsifier never differed: old=${old_d}/${old_r}" >&2
    write_summary FAIL
    exit 1
fi
if grep -qE '^KERNEL_SEL_FAIL ' "$run_dir/kernel_sel.log"; then
    echo "FAIL: a KERNEL_SEL_FAIL line is present:" >&2
    grep -E '^KERNEL_SEL_FAIL ' "$run_dir/kernel_sel.log" >&2 || true
    write_summary FAIL
    exit 1
fi
# A STOCK disagreement is the headline finding and must FAIL CLOSED: this branch
# exits 1 on its own, so the gate can never print a finding and then report PASS.
if grep -qE '^KERNEL_SEL_STOCK .* differ=1$' "$run_dir/kernel_sel.log"; then
    echo "FAIL: the completely patched graph and the graph built at that width disagree:" >&2
    grep -E '^KERNEL_SEL_STOCK .* differ=1$' "$run_dir/kernel_sel.log" | head -n 10 >&2 || true
    write_summary FAIL
    exit 1
fi

write_summary PASS
exit 0
