#!/bin/bash
# Lane legs for the opt-in exact BF16 load4 N2/N3 path (129 / gfx1030, 1/1 lane).
# OFF  = installed libggml-hip (stock model path) + GGML_TP5_BF16_LOAD4_MULTI_STATS=1
# ON   = same install, libggml-hip replaced by the relinked base+mmvf candidate
# Both legs carry the identical ISO environment; the only differences are the hip
# library and GGML_TP5_BF16_LOAD4_MULTI (unset vs 1).
#
# usage: legs.sh <lane-root> <candidate-lib-dir> <run.py> <out-dir> <leg> <rounds>
# The caller holds both GPU flocks (E lock, then /tmp lock) and owns leg order.
set -euo pipefail
root=${1:?lane root}; libs=${2:?dir with libggml-hip.so.0.18.1}; run=${3:?run.py}; out=${4:?out dir}; leg=${5:?leg}; rounds=${6:-2}
bin_dir=$root/install/bin
[ -x "$bin_dir/tq-tp5-compose" ] || { echo "missing $bin_dir/tq-tp5-compose" >&2; exit 2; }

case $leg in
  OFF)
    lib_dir=$root/install/lib
    iso="GGML_TP5_BF16_LOAD4_MULTI_STATS=1"
    ;;
  ON)
    lib_dir=$root/install/lib-on
    mkdir -p "$lib_dir"
    cp -a "$root/install/lib/." "$lib_dir/"
    cp -f "$libs/libggml-hip.so.0.18.1" "$lib_dir/libggml-hip.so.0.18.1"
    iso="GGML_TP5_BF16_LOAD4_MULTI=1 GGML_TP5_BF16_LOAD4_MULTI_STATS=1"
    ;;
  *) echo "leg must be OFF or ON" >&2; exit 2 ;;
esac

python3 "$run" "$leg" "$out" "$lib_dir" 32 "$rounds" $iso
