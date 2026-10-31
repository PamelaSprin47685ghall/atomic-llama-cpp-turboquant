#!/usr/bin/env python3
"""Build the mixer-shape N=1 reference micro and print compile + link commands.

Authored, NOT executed.  The recorded HIP command
(mtp-m3-trio-20261028/build/ggml-cuda-m3t.argv.json) is parsed with the driver's
own argument rules rather than whitelisted per flag, because a whitelist already
lost `-x hip` once.  Output is two commands:

  compile : <clang++> <every recorded flag except output naming> -c <src>
            -o <obj> -MMD -MF <obj>.d
  link    : <clang++> <obj> -L<libdir> -L<rocm> -Wl,-rpath,<libdir>
            -Wl,-rpath,<rocm> -Wl,-rpath-link,<libdir> -lggml -lggml-base
            -lggml-hip -o <bin>

Exactly one compiler per command, exactly one `-o`/`-c`/`-MF`, the source never
after `-MMD` with no destination, and the private directory always before the
libraries.  The script asserts all of that and refuses to print a link command for
a `--libdir` that does not exist.

Usage: build_mixer_n1ref.py <outdir> <libdir>
Devops owns execution (GPU flock); this file is not run by the engineer.
"""
import json
import os
import shlex
import sys

E = "/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024"
TEMPLATE = E + "/mtp-m3-trio-20261028/build/ggml-cuda-m3t.argv.json"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3-mixer-n1ref.cpp")
CLANG = "/opt/rocm/core-10.0/lib/llvm/bin/clang++"


def split_argv(argv):
    """(compiler, [flags], source) using the driver's own argument grouping.

    Output naming (-o/-MF/-MT/-MQ/-MD/-MMD) and its value are dropped; every other
    element is kept, including `-x hip`, `-std=...`, `--offload-arch=...`, `-I`
    and `-iquote` with their values.
    """
    if not argv:
        raise ValueError("empty recorded command")
    cc = argv[0]
    with_value = {"-o", "-c", "-MF", "-MT", "-MQ", "-MD", "-MMD", "-x", "-include",
                  "-iquote", "-isystem", "-I", "-D", "-U", "-L", "-l", "-std",
                  "--offload-arch", "-arch"}
    output_only = {"-o", "-MF", "-MT", "-MQ", "-MD", "-MMD"}
    flags = []
    source = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in output_only:
            i += 2
            continue
        if a in with_value:
            flags.append(a)
            if i + 1 < len(argv):
                flags.append(argv[i + 1])
            i += 2
            continue
        if a.startswith("-"):
            flags.append(a)
        elif source is None:
            source = a
        else:
            raise ValueError("unexpected second operand %r" % a)
        i += 1
    if source is None:
        raise ValueError("recorded command has no source operand")
    return cc, flags, source


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    outdir, libdir = argv[1], argv[2]
    os.makedirs(outdir, exist_ok=True)
    if not os.path.isdir(libdir):
        print("# ERROR: --libdir %s does not exist; refusing to print a link command" % libdir)
        return 2
    _cc, flags, _src = split_argv(json.load(open(TEMPLATE))["argv"])
    obj = os.path.join(outdir, "m3-mixer-n1ref.o")
    compile_argv = [CLANG] + flags + [
        "-c", SRC,
        "-o", obj, "-MMD", "-MF", obj + ".d",
    ]
    link_argv = [CLANG, obj,
                 "-L" + libdir, "-L/opt/rocm/core-10.0/lib",
                 "-Wl,-rpath," + libdir, "-Wl,-rpath,/opt/rocm/core-10.0/lib",
                 "-Wl,-rpath-link," + libdir,
                 "-lggml", "-lggml-base", "-lggml-hip",
                 "-o", os.path.join(outdir, "m3-mixer-n1ref")]
    assert compile_argv.count(CLANG) == 1 and compile_argv[0] == CLANG
    assert compile_argv.count("-c") == 1 and compile_argv.count("-o") == 1
    assert compile_argv.count("-MF") == 1 and compile_argv.index("-c") < compile_argv.index("-o")
    first_lib = next(i for i, a in enumerate(link_argv) if a.startswith("-l"))
    assert link_argv.count(CLANG) == 1 and link_argv.index("-L" + libdir) < first_lib
    json.dump({"compile": compile_argv, "link": link_argv},
              open(os.path.join(outdir, "build.argv.json"), "w"), indent=1)
    print(shlex.join(compile_argv))
    print(shlex.join(link_argv))
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
