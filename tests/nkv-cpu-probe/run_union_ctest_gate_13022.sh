#!/usr/bin/env bash
# Run on the authorized CPU host only. RUN_DIR must be a new directory.
#
# Master gate for pillar 1: one command, one aggregate JSON, for every
# union/onegraph test this repository registers.
#
#   1. pre-flight: `ctest -N` over the whole build tree, then every test in
#      DECLARED below must appear in that listing. A test that is declared but
#      not registered fails the gate here, before anything runs. The listing is
#      the authority for registration; `ctest --no-tests=error` cannot tell a
#      missing test from a passing one.
#   2. run: exactly the declared tests, through ctest, with a JUnit XML file
#      ctest writes itself. No other test in the tree is executed.
#   3. summarize: union_ctest_gate.py reads the listing, the JUnit XML and each
#      test's own artifact, and writes $RUN_DIR/union-ctest-gate.json.
#
# The layer above the existing gates changes none of them: the driver's eight
# gate keys and run_union_varlen_e2e_13022.sh's own pre-flight stay as they are.
# This script only proves that the tests are registered, runs them and records
# what happened.
#
# The build is the caller's: this script configures nothing and builds nothing.
# It takes an existing build tree with -DLLAMA_BUILD_TESTS=ON and the union
# tests registered; the 13022 command in README.md shows the full configure.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${RUN_DIR:?set RUN_DIR to a new output directory}"
: "${BUILD_DIR:?set BUILD_DIR to an existing build tree with the union tests registered}"
if [[ -e $RUN_DIR ]]; then
    echo "FAIL: RUN_DIR already exists: $RUN_DIR" >&2
    echo "      every run must write into a new directory; choose another RUN_DIR." >&2
    exit 1
fi
mkdir -- "$RUN_DIR"
# Both `cd`s run outside set -e with their status captured: a bad path must
# reach an explicit FAIL: line, not kill the script with cd's own status. The
# RUN_DIR cd cannot fail in practice (mkdir just created the directory), but it
# is checked anyway because the cost is one comparison and the failure mode is
# silent.
set +e
run_dir=$(cd -- "$RUN_DIR" && pwd -P)
run_dir_rc=$?
build_dir=$(cd -- "$BUILD_DIR" && pwd -P)
build_dir_rc=$?
set -e
if (( run_dir_rc != 0 )); then
    echo "FAIL: cannot enter RUN_DIR $RUN_DIR (cd exited $run_dir_rc)" >&2
    exit 1
fi
if (( build_dir_rc != 0 )); then
    echo "FAIL: BUILD_DIR is not a directory: $BUILD_DIR (cd exited $build_dir_rc)" >&2
    echo "      BUILD_DIR must be an existing build tree configured with" >&2
    echo "      -DLLAMA_BUILD_TESTS=ON and the union tests registered." >&2
    exit 1
fi

# The summarizer is invoked as `python3 -` with its code on stdin, so the
# script stays self-contained: one file to copy, no sibling module to keep in
# step. It is the only code that decides the pre-flight, so the listing and the
# decision cannot disagree.
#
# Its own availability is checked here, before the first python3 call, because
# the config builder below is also python3. Without this, a missing summarizer
# or a python3 that is not on PATH would surface as python's exit 1 or 127 and
# the message would say the pre-flight failed when in fact the summarizer never
# ran. The two are different problems and must read differently.
summarizer=$root/tests/nkv-cpu-probe/union_ctest_gate.py
if [[ ! -r $summarizer ]]; then
    echo "FAIL: summarizer not found at $summarizer" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo 'FAIL: python3 not on PATH; the summarizer cannot run' >&2
    exit 1
fi

# The declared set: the single source of truth for what must be registered.
# A test named here that `ctest -N` does not list is a failure, never a skip.
# Each line is: test name, then the artifacts that test must leave behind as
# "kind:path". The path may use $RUN_DIR; it is expanded below.
#
# test-union-binder-gate is registered unconditionally by tests/CMakeLists.txt.
# The other five need a caller-supplied GGUF, so they are registered only under
# the same switches the 13022 runners pass:
#   test-union-varlen-e2e   LLAMA_UNION_VARLEN_E2E=ON
#   test-pad-lane-ab-gate   LLAMA_PAD_LANE_AB_GATE=ON
#   test-ntok-onegraph      LLAMA_NTOK_TEST_MODEL non-empty
#   test-varlen-output-*    LLAMA_NTOK_TEST_MODEL non-empty
DECLARED=(
    'test-union-binder-gate'
    'test-union-varlen-e2e|e2e-json:$RUN_DIR/union-varlen-e2e.json'
    'test-pad-lane-ab-gate|pad-gate-json:$RUN_DIR/pad-lane-ab-gate.json'
    'test-ntok-onegraph'
    'test-varlen-output-fixed|rows-bin:$BUILD_DIR/tests/varlen-fixed-on.bin|rows-bin:$BUILD_DIR/tests/varlen-fixed-off.bin'
    'test-varlen-output-variable|rows-bin:$BUILD_DIR/tests/varlen-variable-on.bin|rows-bin:$BUILD_DIR/tests/varlen-variable-off.bin'
)

names=()
for entry in "${DECLARED[@]}"; do
    names+=("${entry%%|*}")
done

# `ctest -N` over the whole tree, not over a regex: the listing must show what
# the build tree registers, and a regex would hide a test that is missing
# because it was added under another name. The listing runs outside set -e and
# its status is captured, because a build tree ctest cannot read must reach the
# explicit diagnostic below rather than kill the script with ctest's status.
# Redirect-then-cat keeps that status unambiguous; a pipe into tee under
# pipefail would report the pipe, not ctest.
list_file=$run_dir/ctest-list.txt
# ctest writes its own JUnit record of the run, and the summarizer reads status
# and wall time from it rather than scraping the console output. The ctest log
# is kept beside it as the human-readable record.
junit=$run_dir/ctest-junit.xml
log=$run_dir/ctest.log
set +e
ctest --test-dir "$build_dir" -N >"$list_file" 2>&1
list_rc=$?
set -e
cat "$list_file"
if (( list_rc != 0 )); then
    echo "FAIL: ctest -N exited $list_rc; cannot confirm the union tests are registered" >&2
fi

# The config is JSON built by python3, not by string concatenation in shell, so
# a path with a quote or a backslash cannot corrupt it. It is written before the
# pre-flight, because the pre-flight is the summarizer reading this config.
config=$run_dir/summarizer-config.json
python3 -c '
import json
import sys

out, run_dir, build_dir, list_rc, listing, junit = sys.argv[1:7]
declared = sys.argv[7:]
tests = []
for item in declared:
    fields = item.replace("$RUN_DIR", run_dir).replace("$BUILD_DIR", build_dir).split("|")
    artifacts = []
    for field in fields[1:]:
        kind, _, path = field.partition(":")
        artifacts.append([kind, path])
    tests.append({"name": fields[0], "log": run_dir + "/ctest.log", "artifacts": artifacts})
config = {
    "run_dir": run_dir,
    "build_dir": build_dir,
    "preflight_rc": int(list_rc),
    "preflight_list": listing,
    "run_rc": 0,
    "junit": junit,
    "argv": declared,
    "tests": tests,
}
with open(out, "w") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
' "$config" "$run_dir" "$build_dir" "$list_rc" "$list_file" "$junit" "${DECLARED[@]}"

summarize() {
    python3 - "$config" <"$summarizer"
}

# Pre-flight: the summarizer reads the listing and fails when a declared test is
# absent, or when ctest -N itself failed. Its exit status here is the gate's
# status; nothing runs on a fail, and the aggregate still lands.
set +e
summarize
preflight_rc=$?
set -e
if (( preflight_rc != 0 )); then
    printf 'not run: pre-flight failed; see %s\n' "$run_dir/union-ctest-gate.json" >"$log"
    cat "$log"
    printf 'GATE_RC=%d JSON=%s LOG=%s\n' "$preflight_rc" "$run_dir/union-ctest-gate.json" "$log"
    exit "$preflight_rc"
fi

# Run exactly the declared tests. -R is anchored to the declared names, so a
# test registered under another name is never run by this gate. The JUnit file
# is written by ctest itself; the summarizer reads status and wall time from it
# rather than scraping the console output.
regex='^('"$(IFS='|'; printf '%s' "${names[*]}")"')$'
set +e
# Remove inherited runtime feature switches: the tests set their own arms.
env -i PATH="$PATH" HOME="$HOME" \
    ctest --test-dir "$build_dir" -R "$regex" --no-tests=error -V \
    --output-junit "$junit" >"$log" 2>&1
run_rc=$?
set -e
cat "$log"

# The config gains the run's status; the summarizer then reads the listing, the
# JUnit XML and each test's own artifact, and writes the aggregate.
python3 -c '
import json
import sys

path, run_rc = sys.argv[1], int(sys.argv[2])
with open(path, "r") as handle:
    config = json.load(handle)
config["run_rc"] = run_rc
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
' "$config" "$run_rc"
set +e
summarize
gate_rc=$?
set -e
printf 'GATE_RC=%d JSON=%s LOG=%s\n' "$gate_rc" "$run_dir/union-ctest-gate.json" "$log"
exit "$gate_rc"
