#!/usr/bin/env bash
set -u
ROOT=/home/kunweiz/Desktop/atomic-llama-cpp-turboquant
BUILD=$ROOT/build-vulkan-localhost
LOG=$ROOT/PLAN/experiment-queue/evidence-dry/1gpu-c06-c08-c10-revalidation.log
export LLAMA_SOURCE_ROOT=$ROOT
{
  echo "=== 1GPU revalidation C06/C08/C10 ==="
  echo "host=$(hostname) date=$(date -Iseconds)"
  echo "build=$BUILD"
  echo
  for t in test-rerot-math test-rerot-span-expand test-rerot-q-prep test-rerot-q3-shared-kv-2reader; do
    echo "=== RUN: $t ==="
    if [[ ! -x $BUILD/bin/$t ]]; then
      echo "RESULT: SKIP missing binary"
      continue
    fi
    set +e
    $BUILD/bin/$t
    ec=$?
    set -e
    echo "EXIT_$t:$ec"
    if [[ $ec -eq 0 ]]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
    echo
  done
  echo "=== RUN: test-rerot-span-expand with LLAMA_REROT_GPU_SPAN_EXPAND=1 ==="
  set +e
  LLAMA_REROT_GPU_SPAN_EXPAND=1 $BUILD/bin/test-rerot-span-expand
  ec=$?
  set -e
  echo "EXIT_span_expand_gpu:$ec"
  if [[ $ec -eq 0 ]]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
} 2>&1 | tee "$LOG"
