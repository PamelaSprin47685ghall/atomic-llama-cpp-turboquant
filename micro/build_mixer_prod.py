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

# Switches that consume the next argv element.
WITH_VALUE = {
    "-o", "-c", "-MF", "-MT", "-MQ", "-MD", "-MMD", "-x", "-include", "-iquote",
    "-isystem", "-I", "-D", "-U", "-L", "-l", "-std", "--offload-arch", "-arch",
    "-Xlinker", "-Xclang", "-mllvm", "--sysroot", "-resource-dir",
}
# Output-only switches: dropped when re-emitting a compile command.
OUTPUT_ONLY = {"-o", "-MF", "-MT", "-MQ", "-MD", "-MMD"}

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


def split_argv(argv):
    """(compiler, [flags...], source) with the driver's own argument grouping.

    A switch that takes a value is emitted together with its value (so `-x hip`
    and `-iquote <dir>` survive as pairs); output naming switches are dropped with
    their value, and the single positional operand is the source.
    """
    if not argv:
        raise ValueError("empty recorded command")
    cc = argv[0]
    flags = []
    source = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in OUTPUT_ONLY:
            i += 2  # drop the switch and its value
            continue
        if a in WITH_VALUE:
            flags.append(a)
            if i + 1 < len(argv):
                flags.append(argv[i + 1])
                i += 2
                continue
            i += 1
            continue
        if a.startswith(("-I", "-D", "-U", "-L", "-l", "-Wl,", "-Xlinker")) and len(a) > 2:
            flags.append(a)
            i += 1
            continue
        if a.startswith("-"):
            flags.append(a)
        elif source is None:
            source = a
        else:
            raise ValueError("unexpected second operand %r in recorded command" % a)
        i += 1
    if source is None:
        raise ValueError("recorded command has no source operand")
    return cc, flags, source


def dedup_groups(flat):
    """Dedup a flat argv whose value-carrying switches are still adjacent."""
    out = []
    seen = set()
    i = 0
    while i < len(flat):
        a = flat[i]
        if a in WITH_VALUE and i + 1 < len(flat):
            group = (a, flat[i + 1])
            i += 2
        else:
            group = (a,)
            i += 1
        if group in seen:
            continue
        seen.add(group)
        out.extend(group)
    return out


def compile_commands(outdir):
    """One command per TU, each with exactly one compiler and explicit outputs."""
    _, hip_flags, _ = split_argv(json.load(open(HIP_TEMPLATE))["argv"])
    _, llama_flags, _ = split_argv(json.load(open(LLAMA_TEMPLATE))["argv"])
    # Exactly one driver compiles both probe TUs: the lane's HIP clang++ (the only
    # one of the two recorded compilers that can see HIP headers), with the HIP
    # flag set followed by the model TU's include/define/warning set.  Taking the
    # compiler from the templates is what previously put a second compiler into
    # the argv; it is now fixed at CLANG and asserted below.
    # Value-carrying switches must keep their values adjacent; dedup therefore works
    # on switch groups, not on flat elements.
    flags = dedup_groups(hip_flags + llama_flags) + ["-DM3_PROBE_GRAPH_HOOK"]
    cmds = []
    for src in ("m3-mixer-prod-path.cpp", "m3-probe-hc-mix.cpp"):
        obj = os.path.join(outdir, os.path.splitext(src)[0] + ".o")
        # -c first, then the source, then the output naming: the source is never
        # emitted after -MMD and the dependency file always has a destination.
        cmds.append([CLANG] + flags + [
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
    compiles = compile_commands(outdir)
    link = link_command(outdir, libdir)
    for c in compiles:
        assert c[0] == CLANG and c.count(CLANG) == 1, "duplicate compiler in compile argv"
        assert c.count("-c") == 1 and c.count("-o") == 1 and c.count("-MF") == 1
        assert c.index("-c") < c.index("-o") < c.index("-MMD") < c.index("-MF")
    assert link[0] == CLANG and link.count(CLANG) == 1, "duplicate compiler in link argv"
    first_lib = next((i for i, a in enumerate(link) if a.startswith("-l")), None)
    assert first_lib is not None, "link argv carries no library flag"
    assert link.index(f"-L{libdir}") < first_lib, "-L<libdir> must precede the libraries"
    assert f"-Wl,-rpath,{libdir}" in link and f"-Wl,-rpath-link,{libdir}" in link
    json.dump({"compile": compiles, "link": link,
               "hook_header_patch": PROBE_HOOK_H,
               "hook_getter_patch": PROBE_GETTER_PATCH},
              open(os.path.join(outdir, "build.argv.json"), "w"), indent=1)
    for c in compiles:
        print(shlex.join(c))
    print(shlex.join(link))
    print()
    print("# invariants this script now guarantees (assert them mechanically before")
    print("# running, e.g. by reading build.argv.json):")
    print("#   * every command has exactly one compiler in position 0;")
    print("#   * every compile command is `-c <src> ... -o <obj> -MMD -MF <obj>.d`:")
    print("#     exactly one '-c', exactly one '-o', exactly one '-MF', and the")
    print("#     source never appears after '-MMD';")
    print("#   * the link command contains '-L<libdir>' and '-Wl,-rpath,<libdir>'")
    print("#     before the first '-l' flag, and '-Wl,-rpath-link,<libdir>';")
    print("#   * no argv element repeats the compiler path; these four are asserted")
    print("#     by the script itself before anything is printed.")
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
