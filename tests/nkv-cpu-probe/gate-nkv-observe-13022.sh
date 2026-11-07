#!/usr/bin/env bash
# n_kv observation gate: the three binder arms and the three negative arms of
# test-nkv-exact-width, each its own ctest result.
#
# Run ONLY on the authorized CPU build host (13022). Configure and build go
# through /home/kunweiz/.local/bin/build-slot (override with BUILD_SLOT=...);
# every ctest invocation stays OUTSIDE the slot.
#
#   MODEL=/path/tiny-dense-llama.gguf \
#   RUN_DIR=/path/to/new-dir \
#       bash tests/nkv-cpu-probe/gate-nkv-observe-13022.sh
#
# What this gate proves, per ctest name:
#
#   p1-nkv-exact-unset      exit 0; NKV_ARM binder=unset; cases single,
#                           multi_seq, multi_stream, widths run (skipped=0)
#                           and `off` skipped=1; every NKV_EXACT line at the
#                           used span
#   p1-nkv-exact-off        exit 0; NKV_ARM binder=0; case off runs and the
#                           four binder cases are skipped=1
#   p1-nkv-exact-on         exit 0; NKV_ARM binder=1; same matrix as unset
#   p1-nkv-exact-negative-mask
#   p1-nkv-exact-negative-k  exit 1; NKV_ARM binder=unset; a
#   p1-nkv-exact-negative-v  "FAIL: missing tensor at step=0" line; and
#                           NKV_NEGATIVE ok=1 omitted=<root>
#
# The arm is injected by the CMake registration (tests/CMakeLists.txt,
# NKV_ARM_ENV), not by the test. The gate re-reads the NKV_ARM line, so a
# binary that ignored the runner's environment and took one path for all three
# arms fails here: two arms would print the same binder= value.
#
# A negative arm that exits 0 is a FAIL, and so is a negative arm that fails
# without the named missing-tensor line: the first means the gate cannot see a
# missing observation, the second means it failed for some other reason.
#
# Exit status, the tri-state contract shared with gate-verdict.sh:
#   0  PASS: every arm passed.
#   1  FAIL: at least one arm failed an equality or a negative arm did not
#      detect its omission.
#   2  UNPROVEN: no arm failed, and at least one arm could not prove its
#      claim (MODEL unset, a skipped test, a missing evidence line).
#      Nothing was compared, or the run could not start.
#
# Outputs in RUN_DIR:
#   identity.txt     revision, branch, model path and sha256, ccache, nproc
#   configure.log    cmake configure output
#   build.log        cmake build output
#   <name>.log       ctest -V output for that arm
#   GATE_SUMMARY     one NKV_<ARM>=pass|fail|unproven field per arm
set -uo pipefail

# The tri-state vocabulary, resolved from this file.
_gate_verdict_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=tests/nkv-cpu-probe/gate-verdict.sh
. "$_gate_verdict_dir/gate-verdict.sh"
unset _gate_verdict_dir

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
model=${MODEL:-}
: "${RUN_DIR:?set RUN_DIR to a new output directory}"

if [[ -z $model ]]; then
    echo "UNPROVEN: MODEL is unset; nothing was configured, built or run"
    printf 'NKV_EXACT_UNSET=unproven NKV_EXACT_OFF=unproven NKV_EXACT_ON=unproven '
    printf 'NKV_EXACT_NEGATIVE_MASK=unproven NKV_EXACT_NEGATIVE_K=unproven '
    printf 'NKV_EXACT_NEGATIVE_V=unproven NKV_EXACT_NEGATIVE_PAD_TAIL=unproven\n'
    exit 2
fi
if [[ ! -f $model || ! -s $model ]]; then
    echo "UNPROVEN: MODEL missing or empty: $model" >&2
    exit 2
fi
if [[ -e $RUN_DIR ]]; then
    echo "UNPROVEN: RUN_DIR already exists: $RUN_DIR" >&2
    exit 2
fi
mkdir -- "$RUN_DIR" || { echo "UNPROVEN: cannot create RUN_DIR" >&2; exit 2; }
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
build_dir=$run_dir/build
model=$(realpath -- "$model")

# ---------------------------------------------------------------- arms
# ctest name | summary key | expected binder= value | expected case list
# A negative arm has an empty case list: it stops at the first step.
arms=(
    "p1-nkv-exact-unset|NKV_EXACT_UNSET|unset|single,multi_seq,multi_stream,widths"
    "p1-nkv-exact-off|NKV_EXACT_OFF|0|off"
    "p1-nkv-exact-on|NKV_EXACT_ON|1|single,multi_seq,multi_stream,widths"
    "p1-nkv-exact-negative-mask|NKV_EXACT_NEGATIVE_MASK|unset|"
    "p1-nkv-exact-negative-k|NKV_EXACT_NEGATIVE_K|unset|"
    "p1-nkv-exact-negative-v|NKV_EXACT_NEGATIVE_V|unset|"
    "p1-nkv-exact-negative-pad-tail|NKV_EXACT_NEGATIVE_PAD_TAIL|unset|"
)

is_negative() { # name
    [[ $1 == p1-nkv-exact-negative-* ]]
}

# ---------------------------------------------------------------- identity
{
    printf 'source=%s\n' "$root"
    printf 'revision='; git -C "$root" rev-parse HEAD
    printf 'branch=';   git -C "$root" rev-parse --abbrev-ref HEAD
    printf 'MODEL=%s\n' "$model"
    printf 'MODEL_sha256='; sha256sum "$model" | cut -d' ' -f1
    printf 'ccache='; (ccache --version 2>/dev/null | head -n 1) || printf 'unavailable\n'
    printf 'nproc='; nproc
    printf 'host='; uname -n
    printf 'date='; date --iso-8601=seconds
} > "$run_dir/identity.txt"

# ---------------------------------------------------------------- configure
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

fail_all() { # reason
    {
        for row in "${arms[@]}"; do
            IFS='|' read -r _ key _ _ <<< "$row"
            printf '%s=fail\n' "$key"
        done
    } > "$run_dir/GATE_SUMMARY"
    echo "FAIL: $1" >&2
    exit 1
}

unproven_all() { # reason
    {
        for row in "${arms[@]}"; do
            IFS='|' read -r _ key _ _ <<< "$row"
            printf '%s=unproven\n' "$key"
        done
    } > "$run_dir/GATE_SUMMARY"
    echo "UNPROVEN: $1" >&2
    exit 2
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

# Same backend-off set as tests/nkv-cpu-probe/gate-p1-merge-13022.sh.
# Configure is not a compile: it runs outside build-slot.
configure_run "$run_dir/configure.log" -S "$root" -B "$build_dir" \
        -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
        -DGGML_WEBGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_VARLEN_TEST_MODEL="$model" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cfg_rc=$?
if [[ $cfg_rc -eq 75 || $cfg_rc -eq 124 || $cfg_rc -eq 143 ]]; then
    unproven_all "configure transient rc=$cfg_rc; see $run_dir/configure.log"
fi
if [[ $cfg_rc -ne 0 ]]; then
    fail_all "configure; see $run_dir/configure.log"
fi

slot_run "$run_dir/build.log" cmake --build "$build_dir" -j"$(nproc)" \
        --target test-nkv-exact-width
bld_rc=$?
if [[ $bld_rc -eq 75 || $bld_rc -eq 124 ]]; then
    if [[ $bld_rc -eq 75 ]]; then
        unproven_all "build build_slot_queue; see $run_dir/build.log"
    else
        unproven_all "build build_slot_timeout; see $run_dir/build.log"
    fi
fi
if [[ $bld_rc -ne 0 ]]; then
    fail_all "build; see $run_dir/build.log"
fi

# ---------------------------------------------------------------- run
declare -A result

for row in "${arms[@]}"; do
    IFS='|' read -r name key want_binder want_cases <<< "$row"
    log=$run_dir/$name.log
    : > "$log"
    # ${CTEST} rather than a bare `ctest`: the heredocs in this file contain
    # `ctest -R '^<name>$'` as prose, and a bare call inside a loop would make
    # those lines look like invocations to a careless reader.
    if "${CTEST:-ctest}" --test-dir "$build_dir" --no-tests=error \
             --output-on-failure -R "^${name}$" -V >> "$log" 2>&1; then
        ctest_ok=1
    else
        ctest_ok=0
    fi
    result[$name]=UNPROVEN

    if is_negative "$name"; then
        # A correct negative run exits 0 and prints the named reason: the
        # test records the missing observation and only then prints
        # NKV_NEGATIVE ok=1. A run that fails for any other reason (model
        # load, context, wrong arm) exits 1 WITHOUT that record, so it is not
        # proof of falsifiability and must not pass here either.
        if [[ $ctest_ok -eq 0 ]]; then
            echo "$name: ctest failed; the negative run did not prove the omission" >&2
            result[$name]=FAIL
            continue
        fi
    elif [[ $ctest_ok -eq 0 ]]; then
        echo "$name: ctest failed" >&2
        result[$name]=FAIL
        continue
    fi

    # exactly one NKV_ARM line, naming the arm the registration set. The
    # test refuses an unknown gate value with exit 2 and no NKV_ARM line, so
    # this also catches a registration that stopped injecting the arm.
    n_arm=$(grep -c '^NKV_ARM ' "$log" || true)
    if [[ ${n_arm:-0} -ne 1 ]]; then
        echo "$name: ${n_arm:-0} NKV_ARM line(s), want 1" >&2
        result[$name]=FAIL
        continue
    fi
    got_binder=$(sed -n 's/^NKV_ARM binder=\([^ ]*\) .*/\1/p' "$log")
    if [[ $got_binder != "$want_binder" ]]; then
        echo "$name: NKV_ARM binder=$got_binder, want $want_binder" >&2
        result[$name]=FAIL
        continue
    fi

    if is_negative "$name"; then
        root_name=${name##*-}
        case $root_name in
            mask)     want_root=attn_inp_kq_mask ;;
            k)        want_root=union_nkv_k ;;
            v)        want_root=union_nkv_v ;;
            pad-tail) want_root=reduce_pad_tail ;;
            *)        want_root="" ;;
        esac
        if [[ -z $want_root ]]; then
            echo "$name: unknown negative root '$root_name'" >&2
            result[$name]=FAIL
            continue
        fi
        # The NKV_ARM line must agree with the root this run omits: the test
        # derives its evidence from --negative, and the arm line must name the
        # same root, or the log is self-contradictory.
        got_negative=$(sed -n 's/^NKV_ARM binder=[^ ]* ntok=[^ ]* reuse_disable=[^ ]* negative=\([^ ]*\)$/\1/p' "$log")
        if [[ $got_negative != "$want_root" ]]; then
            echo "$name: NKV_ARM negative=$got_negative, want $want_root" >&2
            result[$name]=FAIL
            continue
        fi
        # The named failure line, for THIS root. The three n_kv roots print
        # "mask=1 k=0 v=0" style flags; the pad tail is not one of them, so it
        # has its own line and its own text.
        case $want_root in
            attn_inp_kq_mask) want_line="^FAIL: missing tensor at step=0 mask=1 k=0 v=0 " ;;
            union_nkv_k)      want_line="^FAIL: missing tensor at step=0 mask=0 k=1 v=0 " ;;
            union_nkv_v)      want_line="^FAIL: missing tensor at step=0 mask=0 k=0 v=1 " ;;
            reduce_pad_tail)  want_line="^FAIL: missing pad tail at step=0 (negative: omitted reduce_pad_tail " ;;
            *)                want_line="" ;;
        esac
        if ! grep -q -- "$want_line" "$log"; then
            echo "$name: no '${want_line}' line" >&2
            result[$name]=FAIL
            continue
        fi
        if ! grep -q "^NKV_NEGATIVE ok=1 omitted=${want_root}$" "$log"; then
            echo "$name: no NKV_NEGATIVE ok=1 omitted=${want_root} line" >&2
            result[$name]=FAIL
            continue
        fi
        # The undetected-omission text must NOT appear: it is what the test
        # prints when the run passed despite the omission.
        if grep -q "negative run omitting ${want_root} passed" "$log"; then
            echo "$name: the omission went undetected" >&2
            result[$name]=FAIL
            continue
        fi
        # Every check above passed. Promote only from UNPROVEN, so a path
        # that set FAIL cannot be overwritten.
        if [[ ${result[$name]} == UNPROVEN ]]; then
            result[$name]=PASS
        fi
        continue
    fi

    # positive arm: the case list, in order, and one footer per case. Every
    # case is printed in every arm; the ones this arm skips must say so, and
    # a skipped case must not carry step rows.
    mapfile -t got_cases < <(sed -n 's/^NKV_CASE name=\([^ ]*\) .*/\1/p' "$log")
    want_list=()
    IFS=',' read -r -a want_list <<< "$want_cases"
    if [[ ${#got_cases[@]} -ne ${#want_list[@]} ]]; then
        echo "$name: ${#got_cases[@]} NKV_CASE lines, want ${#want_list[@]}" >&2
        result[$name]=FAIL
        continue
    fi
    for i in "${!want_list[@]}"; do
        if [[ ${got_cases[$i]} != "${want_list[$i]}" ]]; then
            echo "$name: case #$i is '${got_cases[$i]}', want '${want_list[$i]}'" >&2
            result[$name]=FAIL
        fi
    done
    if ! grep -q "^NKV_EXACT_END ok=1 cases=${#want_list[@]} " "$log"; then
        echo "$name: missing NKV_EXACT_END ok=1 cases=${#want_list[@]}" >&2
        result[$name]=FAIL
    fi
    # the arm's own cases ran; every other case says skipped=1
    for case_name in "${want_list[@]}"; do
        if ! grep -q "^NKV_CASE name=${case_name} .* skipped=0$" "$log"; then
            echo "$name: case ${case_name} did not run in this arm" >&2
            result[$name]=FAIL
        fi
    done
    for case_name in single multi_seq multi_stream widths off; do
        case " ${want_list[*]} " in
            *" $case_name "*) continue ;;
        esac
        if ! grep -q "^NKV_CASE name=${case_name} .* skipped=1$" "$log"; then
            echo "$name: case ${case_name} is neither run nor marked skipped" >&2
            result[$name]=FAIL
        fi
    done
    if grep -q '^FAIL:' "$log"; then
        echo "$name: printed a FAIL: line but exited 0" >&2
        result[$name]=FAIL
    fi
    # Every check above passed. Promote only from UNPROVEN.
    if [[ ${result[$name]} == UNPROVEN ]]; then
        result[$name]=PASS
    fi
done

# The three positive arms must be three DIFFERENT runs. unset and on share a
# case list, so the only thing that separates them in the log is the arm line;
# that is checked above. Here the gate states the property it relies on: the
# three binder= values are not all equal.
b_unset=$(sed -n 's/^NKV_ARM binder=\([^ ]*\) .*/\1/p' "$run_dir/p1-nkv-exact-unset.log" 2>/dev/null || true)
b_off=$(sed -n 's/^NKV_ARM binder=\([^ ]*\) .*/\1/p' "$run_dir/p1-nkv-exact-off.log" 2>/dev/null || true)
b_on=$(sed -n 's/^NKV_ARM binder=\([^ ]*\) .*/\1/p' "$run_dir/p1-nkv-exact-on.log" 2>/dev/null || true)
if [[ $b_unset == "$b_off" || $b_off == "$b_on" || $b_unset == "$b_on" ]]; then
    echo "arms are not distinct: unset=$b_unset off=$b_off on=$b_on" >&2
    result[p1-nkv-exact-unset]=FAIL
    result[p1-nkv-exact-off]=FAIL
    result[p1-nkv-exact-on]=FAIL
fi

# ---------------------------------------------------------------- summary
# The result array carries the three names. The summary prints them in lower
# case, so an arm that could not prove its claim reads as `unproven`, never as
# `fail`.
result_name() { # value
    case $1 in
        PASS) printf 'pass' ;;
        FAIL) printf 'fail' ;;
        *)    printf 'unproven' ;;
    esac
}

: > "$run_dir/GATE_SUMMARY"
for row in "${arms[@]}"; do
    IFS='|' read -r name key _ _ <<< "$row"
    printf '%s=%s\n' "$key" "$(result_name "${result[$name]:-UNPROVEN}")" >> "$run_dir/GATE_SUMMARY"
done
cat "$run_dir/GATE_SUMMARY"

# The exit status is the worst arm: FAIL > UNPROVEN > PASS.
overall=0
for row in "${arms[@]}"; do
    IFS='|' read -r name _ _ _ <<< "$row"
    case ${result[$name]:-UNPROVEN} in
        FAIL)
            overall=1
            break
            ;;
        PASS)
            ;;
        *)
            if [[ $overall -eq 0 ]]; then
                overall=2
            fi
            ;;
    esac
done
exit "$overall"
