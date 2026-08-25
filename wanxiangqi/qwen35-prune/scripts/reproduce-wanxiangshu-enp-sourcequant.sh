#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  reproduce-wanxiangshu-enp-sourcequant.sh TEACHER.gguf BAOMU.md [OUT_DIR]

Environment:
  THREADS=16          planner/apply thread count
  BUILD_DIR=build     CMake build directory (relative to repo root or absolute)
  FORCE=0             set to 1 to remove this script's known output files first
  CMAKE_ARGS=...      extra arguments appended to the CMake configure command

The script verifies the published teacher/corpus/trie SHA-256 values, builds
the fork-isolated calibration/prune tools, regenerates the trie imatrix and ENP
plan, applies source-quant expert pruning with GDN/vocab preserved, runs the
sanctioned finalize verifier, and checks the final GGUF byte hash.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
TEACHER="$(realpath "$1")"
CORPUS="$(realpath "$2")"
OUT_DIR="${3:-$ROOT/wanxiangqi/.reproduce-output/wanxiangshu-enp-sourcequant-v2}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(realpath "$OUT_DIR")"

# The final model is ~11.1 GiB and the imatrix/plan add a few hundred MiB.
# Fail before a long calibration run if the destination cannot safely hold the
# complete atomic-finalize pipeline.
free_kib="$(df -Pk "$OUT_DIR" | awk 'NR==2 {print $4}')"
min_free_kib=$((14 * 1024 * 1024))
if (( free_kib < min_free_kib )); then
    printf 'Insufficient free space in %s: need at least 14 GiB, have %.2f GiB\n' \
        "$OUT_DIR" "$(awk -v k="$free_kib" 'BEGIN {print k / 1024 / 1024}')" >&2
    exit 1
fi

THREADS="${THREADS:-16}"
BUILD_DIR="${BUILD_DIR:-build}"
FORCE="${FORCE:-0}"
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$ROOT/$BUILD_DIR"
fi

TRIE="$ROOT/wanxiangqi/qwen35-prune/evidence/wanxiangshu-teacher-trie-v1"
IMATRIX="$OUT_DIR/wanxiangshu-enp-uniform-imatrix-v1.gguf"
PLAN="$OUT_DIR/wanxiangshu-enp-uniform-plan-v1.json"
TMP="$OUT_DIR/wanxiangshu-enp-uniform-sourcequant-v2.gguf.tmp"
FINAL="$OUT_DIR/wanxiangshu-enp-uniform-sourcequant-v2.gguf"

EXPECTED_TEACHER_SHA="3e13c52d562b1c97998c3cf3954b99f7e8156ca6a1d8b2db76fe0499d53a95c4"
EXPECTED_CORPUS_SHA="01ffbc268ef1395fd65a4bae04ab4282a13bd71e30bdbcb1b1ed8734b2ec266a"
EXPECTED_FINAL_SHA="7d8b84d27aa9f3c92ddd494388f0d7e4e2c7a13933793be4b15159cdaea61200"

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

require_sha() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256_of "$path")"
    if [[ "$actual" != "$expected" ]]; then
        printf 'SHA-256 mismatch: %s\n  expected %s\n  actual   %s\n' "$path" "$expected" "$actual" >&2
        exit 1
    fi
}

require_sha "$TEACHER" "$EXPECTED_TEACHER_SHA"
require_sha "$CORPUS" "$EXPECTED_CORPUS_SHA"
require_sha "$TRIE/format.json" "7f6256c4b5999c1a4fa19e169db24b27900f1e9d56b2ced2e20b390044d34548"
require_sha "$TRIE/nodes-000000.bin" "15d10275d4c5f572029f1e662e6af56246985dd7d8867ccd5b9593c4139fe686"
require_sha "$TRIE/requests-000000.bin" "df4c450217eed60b917d6882ae996f2cad389ca4fffdf55c511f88515f44533f"

known_outputs=(
    "$IMATRIX"
    "$PLAN"
    "$PLAN.checkpoint.json"
    "$PLAN.checkpoint.json.lock"
    "$TMP"
    "$TMP.manifest.json"
    "$TMP.checkpoint.json"
    "$TMP.checkpoint.json.lock"
    "$FINAL"
    "$FINAL.manifest.json"
)

for path in "${known_outputs[@]}"; do
    if [[ -e "$path" ]]; then
        if [[ "$FORCE" == "1" ]]; then
            rm -f -- "$path"
        else
            printf 'Refusing to overwrite %s (set FORCE=1 for a clean rerun)\n' "$path" >&2
            exit 1
        fi
    fi
done

extra_cmake_args=()
if [[ -n "${CMAKE_ARGS:-}" ]]; then
    # Intentional shell-style splitting for the conventional CMAKE_ARGS env.
    # shellcheck disable=SC2206
    extra_cmake_args=(${CMAKE_ARGS})
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "${extra_cmake_args[@]}"
cmake --build "$BUILD_DIR" --target llama-qwen35-imatrix llama-qwen35-prune -j"$THREADS"

IMATRIX_BIN="$BUILD_DIR/bin/llama-qwen35-imatrix"
PRUNE_BIN="$BUILD_DIR/bin/llama-qwen35-prune"

"$IMATRIX_BIN" \
    -m "$TEACHER" \
    -o "$IMATRIX" \
    -c 512 \
    -b 2048 \
    -ub 512 \
    --threads "$THREADS" \
    --no-ppl \
    --request-token-trie "$TRIE" \
    --prune-qk-gram \
    --prune-vocab-counts \
    --prune-mtp \
    --prune-enp \
    --prune-enp-legacy-uniform

"$PRUNE_BIN" plan \
    "$TEACHER" \
    "$IMATRIX" \
    "$PLAN" \
    --threads "$THREADS"

"$PRUNE_BIN" apply \
    "$TEACHER" \
    "$IMATRIX" \
    "$PLAN" \
    "$TMP" \
    --threads "$THREADS" \
    --keep-gdn \
    --keep-vocab \
    --expert-source-quant

"$PRUNE_BIN" finalize \
    "$TEACHER" \
    "$TMP" \
    "$PLAN" \
    "$CORPUS" \
    "$FINAL"

actual_final_sha="$(sha256_of "$FINAL")"
if [[ "$actual_final_sha" != "$EXPECTED_FINAL_SHA" ]]; then
    printf 'Final GGUF differs from the published artifact.\n  expected %s\n  actual   %s\n' \
        "$EXPECTED_FINAL_SHA" "$actual_final_sha" >&2
    exit 1
fi

printf 'Reproduction OK\n%s  %s\n' "$actual_final_sha" "$FINAL"

