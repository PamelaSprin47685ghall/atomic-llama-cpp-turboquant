#!/usr/bin/env python3
"""Build the mixer-shape N=1 reference micro and print its run matrix.

Authored, NOT executed.  This script only assembles argv from the lane's own
recorded recipes, so the compiler flags stay identical to the M3 micro that the
trunk baselines were produced with:

  compile flags : mtp-m3-trio-20261028/build/ggml-cuda-m3t.argv.json (clang++
                  from /opt/rocm/core-10.0, gfx1030, the GGML_* defines)
  link inputs   : the same file is used only for -L/-l values; the micro links
                  against the lane install/lib of whichever library is under
                  test, plus /opt/rocm/core-10.0/lib

Usage:
  build_mixer_n1ref.py <outdir> <libdir>

Writes <outdir>/m3-mixer-n1ref and <outdir>/build.argv.json for the record.
Nothing in the official lanes is touched.  Devops owns execution (GPU flock);
this file is not run by the engineer.
"""
import json
import os
import shlex
import sys

E = "/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024"
TEMPLATE = E + "/mtp-m3-trio-20261028/build/ggml-cuda-m3t.argv.json"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3-mixer-n1ref.cpp")


def flags_from_template():
    """Every recorded flag except the ones that name lane output files.

    The previous filter kept only -I/-D/-O/-f/-W/--offload-arch/-std and thereby
    dropped the lane's `-x hip`, so the generated command could not find
    hip/hip_runtime.h.  Fidelity is now the rule: drop only what names an output
    (-o, -c, -MF, -MT, -MD, -MTdeps) and keep everything else verbatim.
    """
    argv = json.load(open(TEMPLATE))["argv"]
    drop_value_after = {"-o", "-c", "-MF", "-MT", "-MD", "-MTdeps"}
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in drop_value_after:
            i += 2
            continue
        if a.startswith("-Wl,--dependency-file"):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    outdir, libdir = argv[1], argv[2]
    os.makedirs(outdir, exist_ok=True)
    cc = "/opt/rocm/core-10.0/lib/llvm/bin/clang++"
    # The template already carries the lane's -x hip, -std, -O, -DNDEBUG,
    # --offload-arch and include set; do not add a second copy of any of them.
    args = [cc] + flags_from_template() + [
        SRC,
        "-o", os.path.join(outdir, "m3-mixer-n1ref"),
        "-L" + libdir,
        "-L/opt/rocm/core-10.0/lib",
        "-lggml", "-lggml-base", "-lggml-hip",
        "-Wl,-rpath," + libdir,
        "-Wl,-rpath,/opt/rocm/core-10.0/lib",
    ]
    json.dump({"argv": args}, open(os.path.join(outdir, "build.argv.json"), "w"), indent=1)
    print(shlex.join(args))
    print()
    print("# run matrix (Devops, under both GPU flocks), binary from the line above:")
    print("#   MIXER=1 %s <libdir> <nt> <sites> <prefix> [dev]" % os.path.join(outdir, "m3-mixer-n1ref"))
    print("#   1. N=1 reference (mixer shape):")
    print("#      MIXER=1 M3_ROW0=0 m3-mixer-n1ref <libdir> 2 1 ref-mixer-multi")
    print("#      -> writes ref-mixer-multi.nt1.0/.1.bin (per-row single-row images)")
    print("#   2. candidate (mixer only) legs, same libdir:")
    print("#      MIXER=1 GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1 m3-mixer-n1ref <libdir> 2 1 on-mixer")
    print("#      MIXER=1 GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1 GGML_TP5_HC_N3_TAIL_DIAG_SCOPE=mixer m3-mixer-n1ref <libdir> 2 1 diag-mixer")
    print("#      MIXER=1 GGML_TP5_HC_N3_TRIO=1 m3-mixer-n1ref <libdir> 2 1 off-mixer")
    print("#   3. trunk control (MIXER unset) on the same libdir, to prove the split:")
    print("#      GGML_TP5_HC_N3_TRIO=1 m3-mixer-n1ref <libdir> 3 1 on-trunk")
    print("#      GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1 m3-mixer-n1ref <libdir> 3 1 full-trunk")
    print("#   4. compare with diag/m3-n1-rowref.py against the .nt1.<r>.bin images")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
