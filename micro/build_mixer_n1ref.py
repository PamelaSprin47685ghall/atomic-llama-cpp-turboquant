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


# Switches that take the next argv element, grouped by what the value is.  The
# recorded command uses the separated form (-o <file>, -x hip, -iquote <dir>), so a
# flat walk cannot tell a switch's value from the translation unit; the previous
# version mistook the -MF dependency file for the source.
VALUE_FILE = {"-o", "-MF", "-MT", "-MQ"}          # dropped: output naming
VALUE_KEEP = {"-x", "-iquote", "-isystem", "-include", "-I", "-L", "--sysroot",
              "-resource-dir", "-D", "-U", "-l", "-arch", "-Xlinker", "-Xclang", "-mllvm"}
OUTPUT_ONLY = {"-MD", "-MMD"}
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu", ".c++", ".C")


def split_argv(argv, what="command"):
    """(compiler, flag groups, option map, the one source, the recorded -MF value).

    Raises on ambiguity (second positional, non-source positional, missing value,
    two `-c`), so a command that cannot be compiled is never printed.
    """
    if not argv:
        raise ValueError("%s: empty recorded command" % what)
    compiler = argv[0]
    if compiler.startswith("-"):
        raise ValueError("%s: first element is not a compiler: %r" % (what, compiler))
    flags = []
    options = {}
    source = None
    dep = None
    have_c = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in VALUE_FILE:
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                raise ValueError("%s: switch %r has no value" % (what, a))
            if a == "-MF":
                dep = argv[i + 1]
            i += 2
            continue
        if a in VALUE_KEEP:
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                raise ValueError("%s: switch %r has no value" % (what, a))
            flags.append([a, argv[i + 1]])
            options[a] = argv[i + 1]
            i += 2
            continue
        if a in OUTPUT_ONLY:
            i += 1
            continue
        if a == "-c":
            if have_c:
                raise ValueError("%s: more than one -c" % what)
            have_c = True
            i += 1
            continue
        if a.startswith("-"):
            flags.append([a])
            i += 1
            continue
        if source is not None:
            raise ValueError("%s: second positional operand %r (first was %r)" % (what, a, source))
        if not a.endswith(SOURCE_SUFFIXES):
            raise ValueError("%s: positional operand %r is not a source file" % (what, a))
        source = a
        i += 1
    if source is None:
        raise ValueError("%s: no translation unit in recorded command" % what)
    if not have_c:
        raise ValueError("%s: no -c in recorded command" % what)
    return compiler, flags, options, source, dep


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    outdir, libdir = argv[1], argv[2]
    os.makedirs(outdir, exist_ok=True)
    if not os.path.isdir(libdir):
        print("# ERROR: --libdir %s does not exist; refusing to print a link command" % libdir)
        return 2
    try:
        _cc, groups, options, _src, _dep = split_argv(json.load(open(TEMPLATE))["argv"], "HIP template")
    except ValueError as exc:
        print("# ERROR: %s" % exc)
        return 2
    if "-x" not in options or "-iquote" not in options:
        print("# ERROR: HIP template lost -x/-iquote; refusing to emit a command that cannot "
              "find the HIP headers")
        return 2
    flags = [x for g in groups for x in g]
    for forbidden in ("-c", "-o", "-MF", "-MT", "-MQ", "-MD", "-MMD"):
        if forbidden in flags:
            print("# ERROR: flag tail still contains %r" % forbidden)
            return 2
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
    if compile_argv[0] != CLANG or compile_argv.count(CLANG) != 1:
        print("# ERROR: compile argv must start with exactly one compiler")
        return 2
    if compile_argv.count("-c") != 1 or compile_argv.count("-o") != 1 or compile_argv.count("-MF") != 1:
        print("# ERROR: compile argv must carry exactly one -c/-o/-MF")
        return 2
    if not (compile_argv.index("-c") < compile_argv.index("-o") < compile_argv.index("-MMD")
            < compile_argv.index("-MF")):
        print("# ERROR: compile argv order must be -c < src, -o, -MMD, -MF")
        return 2
    first_lib = next((i for i, a in enumerate(link_argv) if a.startswith("-l")), None)
    if first_lib is None or link_argv.index("-L" + libdir) > first_lib:
        print("# ERROR: -L<libdir> must precede the libraries")
        return 2
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
