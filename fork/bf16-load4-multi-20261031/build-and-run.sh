#!/bin/bash
# Build and run the opt-in exact BF16 load4 N2/N3 micro (129 gfx1030, 1/1 lane).
# Never touches the selected installs: it only links against an existing libggml
# and compiles the candidate source into a private directory.
#
# usage: build-and-run.sh <tree> <install-prefix> [device] [repeats]
set -euo pipefail
tree=${1:?tree with ggml/src (candidate worktree or lane src copy)}
install=${2:?install prefix holding lib/libggml-hip.so*}
device=${3:-0}
repeats=${4:-200}
here=$(cd "$(dirname "$0")" && pwd)
rocm=/opt/rocm/core-10.0
out=$here/build
mkdir -p "$out"

# Private copy of the patched source; the tree itself is never modified.
cp -f "$tree/ggml/src/ggml-cuda/mmvf.cu" "$out/mmvf.multi.cu"

"$rocm/lib/llvm/bin/clang++" \
  -O3 -DNDEBUG -std=gnu++17 -funsafe-math-optimizations --offload-arch=gfx1030 -fPIC \
  -DGGML_USE_HIP -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED \
  -DGGML_CUDA_NO_PEER_COPY -DGGML_HIP_NO_MMQ_MFMA -DGGML_HIP_NO_VMM \
  -DGGML_SCHED_MAX_COPIES=4 -DUSE_PROF_API=1 -D_GNU_SOURCE \
  -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I"$tree/ggml/include" -iquote "$tree/ggml/src/ggml-cuda" \
  -DTP5_MMVF_SOURCE="\"$out/mmvf.multi.cu\"" \
  -x hip "$here/micro.cpp" -o "$out/micro" \
  -L"$install/lib" -lggml-hip -lggml-base \
  -Wl,-rpath,"$install/lib:$rocm/lib" --hip-link -lamdhip64

"$out/micro" "$device" "$repeats" | tee "$out/micro.jsonl"
grep -q '"kind":"summary"' "$out/micro.jsonl"
grep '"kind":"summary"' "$out/micro.jsonl" | grep -q '"status":"PASS"'
echo "MICRO_BITWISE_PASS device=$device repeats=$repeats"
