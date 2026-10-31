#!/usr/bin/env python3
"""Build the production-path M3 mixer test; print compile and link commands.

The previous revisions of the recipes were wrong in four ways that only show up
when the generated argv is executed exactly as printed (Devops
`lead-m3mixer-verify-20261031`):

  1. the compiler appeared twice (it was taken from both recorded commands) and
     ld.lld rejected the second copy as `unknown file type`;
  2. the compile line had the `.cpp` after `-MMD`, with no `-o`/`-MF`, so the
     translation unit was treated as a flag tail and the dependency file had no
     destination;
  3. the link line had no `-L`/`-rpath` entry for the private library directory,
     so `-lllama` resolved against whatever the loader path happened to hold;
  4. one rewrote `-o`/`-c`/`-MF` positions instead of rebuilding the argv in the
     order the recorded commands use.

The fix is structural: each recorded argv is parsed into (compiler, flags,
source) with the driver's own argument rules (a switch that takes a value is kept
together with it, and `-o/-MF/-MT/-MQ/-MD/-MMD` are dropped), and every output is
re-emitted explicitly: one compiler, the deduplicated flag union, then
`-c <src> -o <obj> -MMD -MF <obj>.d`.  The link line is not mined from the
recorded one; it is assembled from the objects, the private `-L`/`-rpath`/
`-rpath-link` entries, the library list taken from the recorded link command, and
`-o <bin>` -- so no recorded absolute output path can leak into it.

Output is two command groups, never one merged argv: one compile per translation
unit (each with exactly one compiler, `-c <src>`, `-o <obj>`, `-MMD -MF <obj>.d`)
followed by one link (objects, then `-L<libdir>` / `-rpath` / `-rpath-link`, then
the library list, then `-o <bin>`).  The candidate `libllama` must exist in
`<libdir>`; the script warns when the directory is absent or a source is missing
and exits non-zero instead of printing a command that cannot work.

Authored, NOT executed by the engineer.
"""
import json
import os
import re
import shlex
import sys

E = "/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024"
HIP_TEMPLATE = E + "/mtp-m3-trio-20261028/build/ggml-cuda-m3t.argv.json"
LLAMA_TEMPLATE = E + "/mtp-hc-row-tail-candidate-20261025/commands/qwen4exp.cpp.argv.json"
LLAMA_LINK = E + "/mtp-hc-row-tail-candidate-20261025/commands/llama-link.argv.json"
HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(HERE)

CLANG = "/opt/rocm/core-10.0/lib/llvm/bin/clang++"

# Switches that consume the next argv element, grouped by what the value is.
# Listed because the recorded commands use the separated form (`-o <file>`,
# `-x hip`, `-iquote <dir>`), so a flat walk cannot tell a switch's value from the
# translation unit; the previous parser guessed and mistook the `.d` dependency
# file for the source.
VALUE_FILE = {"-o", "-MF", "-MT", "-MQ"}           # drop: output naming
VALUE_LANG = {"-x"}                                # keep: switch + value
VALUE_PATH = {"-iquote", "-isystem", "-include", "-I", "-L", "--sysroot",
              "-resource-dir"}                     # keep: switch + value
VALUE_TEXT = {"-D", "-U", "-l", "-arch"}           # keep: switch + value
VALUE_EXTRA = {"-Xlinker", "-Xclang", "-mllvm"}    # keep: switch + value
VALUE_ALL = VALUE_FILE | VALUE_LANG | VALUE_PATH | VALUE_TEXT | VALUE_EXTRA
OUTPUT_ONLY = {"-MD", "-MMD"}                      # drop: bare output switches
# Accepted translation-unit suffixes for the recorded compile command.
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu", ".c++", ".C")

PROBE_HOOK_H = """
#if defined(M3_PROBE_GRAPH_HOOK)
    // Test-only. The owning context stays in ctx_compute; this view points at a
    // caller-owned context and the getters prefer it only while it is non-null.
    ggml_context * ctx_compute_view = nullptr;
#endif
"""

PROBE_GETTER_PATCH = """
    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const {
#if defined(M3_PROBE_GRAPH_HOOK)
        if (ctx_compute_view != nullptr) {
            return ctx_compute_view;
        }
#endif
        return ctx_compute.get();
    }
    void reset_context_view(ggml_context * c, ggml_cgraph * g) { ctx_compute_view = c; gf = g; }
"""


def split_argv(argv, what="command"):
    """Parse an explicitly recorded compiler command.

    Returns (compiler, flags, options, source, dep):
      compiler : argv[0]
      flags    : switch groups to keep, each as a list, in the recorded order
      options  : {switch: value} for the value-carrying switches that were kept
      source   : the ONE translation unit (validated by suffix and existence)
      dep      : the value of -MF when recorded, else None

    Exactly one `-c` and one source are required.  Civilised ambiguity is refused:
    a second positional operand, a source after -MMD with no -MF value, or a source
    with an unexpected suffix raises instead of producing a command that cannot be
    compiled (the failure modes Devops hit twice).
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
        if a in VALUE_ALL:
            if i + 1 >= len(argv):
                raise ValueError("%s: switch %r has no value" % (what, a))
            value = argv[i + 1]
            if value.startswith("-"):
                raise ValueError("%s: switch %r value looks like a flag: %r" % (what, a, value))
            if a in VALUE_FILE:
                if a == "-MF":
                    dep = value
                # -o/-MT/-MQ name lane outputs; they are re-created here.
            else:
                flags.append([a, value])
                options[a] = value
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
        # positional operand: must be the single translation unit
        if source is not None:
            raise ValueError("%s: second positional operand %r (first was %r)" % (what, a, source))
        if not a.endswith(SOURCE_SUFFIXES):
            raise ValueError("%s: positional operand %r is not a source file; the recorded "
                             "command must name exactly one translation unit" % (what, a))
        source = a
        i += 1
    if source is None:
        raise ValueError("%s: no translation unit in recorded command" % what)
    if not have_c:
        raise ValueError("%s: no -c in recorded command" % what)
    return compiler, flags, options, source, dep


def dedup_groups(groups):
    """Deduplicate switch groups (each group is a list) while keeping order."""
    out = []
    seen = set()
    for g in groups:
        key = tuple(g)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(g))
    return out


def compile_commands(outdir):
    """One command per probe TU, each with exactly one compiler and explicit outputs.

    The recorded HIP and model commands are parsed only for their flags/options:
    neither template's own translation unit or dependency file is carried over, and
    `-c` appears exactly once in the emitted command.
    """
    _, hip_flags, hip_opts, hip_src, _ = split_argv(json.load(open(HIP_TEMPLATE))["argv"], "HIP template")
    _, llama_flags, llama_opts, llama_src, _ = split_argv(json.load(open(LLAMA_TEMPLATE))["argv"], "model template")
    if not os.path.isfile(hip_src) or not os.path.isfile(llama_src):
        raise ValueError("recorded templates name sources that do not exist: %r / %r"
                         % (hip_src, llama_src))
    # The probe must be compiled as HIP C++ with both header families visible.
    for required in ("-x", "-iquote"):
        if required not in hip_opts:
            raise ValueError("HIP template lost %r; refusing to emit a command that cannot "
                             "find the HIP headers" % required)
    # Include search order matters: the HIP template's -iquote/-I come first so the
    # probe sees the HIP-facing ggml headers, then the model template's include set.
    # The deduplicated flag tail must not contain -c/-o/-MF: those are re-emitted per
    # source below, exactly once.
    flags = dedup_groups(hip_flags + llama_flags)
    flat = [x for g in flags for x in g]
    for forbidden in ("-c", "-o", "-MF", "-MT", "-MQ", "-MD", "-MMD"):
        if forbidden in flat:
            raise ValueError("flag tail still contains %r; refusing to emit two output "
                             "specifications" % forbidden)
    flat = flat + ["-DM3_PROBE_GRAPH_HOOK"]
    cmds = []
    emitted = set()
    for src in ("m3-mixer-prod-path.cpp", "m3-probe-hc-mix.cpp"):
        if src in emitted:
            raise ValueError("duplicate source %r" % src)
        emitted.add(src)
        obj = os.path.join(outdir, os.path.splitext(src)[0] + ".o")
        cmds.append([CLANG] + flat + [
            "-c", os.path.join(HERE, src),
            "-o", obj, "-MMD", "-MF", obj + ".d",
        ])
    return cmds


def link_command(outdir, libdir):
    """Rebuild the link line instead of mining the recorded one.

    The recorded llama link argv names the lane's own output and library by
    absolute path; carrying it over verbatim is what produced a link line without
    the private `-L`.  Only its library list and rpath style are reused, and the
    private directory is placed before every `-l` flag.
    """
    recorded = json.load(open(LLAMA_LINK))["argv"]
    libs = []
    for a in recorded:
        if a.startswith("-l") and a not in libs:
            libs.append(a)
        elif a.endswith(".so") or ".so." in a:
            base = os.path.basename(a)
            lib = re.sub(r"^lib", "", base)
            lib = lib.split(".so")[0]
            flag = "-l" + lib
            if flag not in libs:
                libs.append(flag)
    if not libs:
        libs = ["-lllama", "-lggml", "-lggml-base", "-lggml-hip"]
    objs = [os.path.join(outdir, "m3-mixer-prod-path.o"),
            os.path.join(outdir, "m3-probe-hc-mix.o")]
    return [CLANG] + objs + [
        f"-L{libdir}",
        f"-L/opt/rocm/core-10.0/lib",
        f"-Wl,-rpath,{libdir}",
        f"-Wl,-rpath,/opt/rocm/core-10.0/lib",
        "-Wl,-rpath-link," + libdir,
    ] + libs + [
        "-o", os.path.join(outdir, "m3-mixer-prod-path"),
    ]


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    outdir, libdir = argv[1], argv[2]
    os.makedirs(outdir, exist_ok=True)
    if not os.path.isdir(libdir):
        # Fail closed: a link line printed for a directory that does not exist is
        # exactly the defect this revision is fixing.
        print("# ERROR: --libdir %s does not exist; refusing to print a link command" % libdir)
        return 2
    if not any(os.path.exists(os.path.join(libdir, n))
               for n in ("libllama.so", "libllama.so.0", "libllama.so.0.0.2451")):
        print("# ERROR: %s holds no libllama; the probe must link the candidate tower" % libdir)
        return 2
    missing = [s for s in ("m3-mixer-prod-path.cpp", "m3-probe-hc-mix.cpp")
               if not os.path.isfile(os.path.join(HERE, s))]
    if missing:
        print("# ERROR: missing sources: %s" % ", ".join(missing))
        return 2
    try:
        compiles = compile_commands(outdir)
        link = link_command(outdir, libdir)
    except ValueError as exc:
        # Ambiguous or unusable recorded command: stop before printing anything.
        print("# ERROR: %s" % exc)
        return 2
    for c in compiles:
        if c[0] != CLANG or c.count(CLANG) != 1:
            print("# ERROR: compile argv does not start with exactly one compiler")
            return 2
        if c.count("-c") != 1 or c.count("-o") != 1 or c.count("-MF") != 1:
            print("# ERROR: compile argv must carry exactly one -c/-o/-MF")
            return 2
        if not (c.index("-c") < c.index("-o") < c.index("-MMD") < c.index("-MF")):
            print("# ERROR: compile argv order must be -c < src, -o, -MMD, -MF")
            return 2
        srcs = [a for a in c if a.endswith(SOURCE_SUFFIXES) and os.path.isfile(a)]
        if len(srcs) != 1:
            print("# ERROR: compile argv names %d existing sources: %s" % (len(srcs), srcs))
            return 2
    if link[0] != CLANG or link.count(CLANG) != 1:
        print("# ERROR: link argv does not start with exactly one compiler")
        return 2
    first_lib = next((i for i, a in enumerate(link) if a.startswith("-l")), None)
    if first_lib is None or link.index(f"-L{libdir}") > first_lib:
        print("# ERROR: -L<libdir> must precede the libraries")
        return 2
    if f"-Wl,-rpath,{libdir}" not in link or f"-Wl,-rpath-link,{libdir}" not in link:
        print("# ERROR: link argv lacks the private dir rpath entries")
        return 2
    json.dump({"compile": compiles, "link": link,
               "hook_header_patch": PROBE_HOOK_H,
               "hook_getter_patch": PROBE_GETTER_PATCH},
              open(os.path.join(outdir, "build.argv.json"), "w"), indent=1)
    for c in compiles:
        print(shlex.join(c))
    print(shlex.join(link))
    print()
    print("# invariants enforced before printing (violations exit 2 with no command):")
    print("#   * exactly one compiler per command;")
    print("#   * exactly one -c/-o/-MF, ordered -c < src, -o, -MMD, -MF;")
    print("#   * exactly one existing source in each compile command;")
    print("#   * -L<libdir> / -Wl,-rpath / -Wl,-rpath-link for the private dir, before")
    print("#     the first -l flag;")
    print("#   * the recorded templates must name an existing single .cu/.cpp with -x")
    print("#     and -iquote present, else the script refuses instead of guessing.")
    print()
    print("# run matrix (Devops, both GPU flocks):")
    print("# 1. N=1 reference, MIXER site, production gate OFF:")
    print("#    GGML_TP5_HC_N3_TRIO=1 M3_TAIL_DIAG=1 ./m3-mixer-prod-path <libdir> 2 mixer ref-mixer")
    print("# 2. candidate mixer legs (exact-1 gate, then the diag scope):")
    print("#    GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1 ./m3-mixer-prod-path <libdir> 2 mixer on-mixer")
    print("#    GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_DIAG_SCOPE=mixer ./m3-mixer-prod-path <libdir> 2 mixer diag-mixer")
    print("# 3. trunk control on the same stack (inject live, il=0):")
    print("#    GGML_TP5_HC_N3_TRIO=1 ./m3-mixer-prod-path <libdir> 3 trunk ref-trunk")
    print("#    GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1 ./m3-mixer-prod-path <libdir> 3 trunk on-trunk")
    print("# 4. compare with diag/m3-n1-rowref.py <leg-prefix> <ref-prefix>")
    print("# 5. witness: ON mixer prints lo_retained=1, OFF prints 0, trunk prints")
    print("#    xn ret=0 in its M3_TAIL_BUILD line.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
