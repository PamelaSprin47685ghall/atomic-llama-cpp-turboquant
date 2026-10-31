#!/usr/bin/env python3
"""Build the production-path M3 mixer test and print its faithful run matrix.

Two outputs:

  1. m3-mixer-prod-path — links against the candidate libllama of this branch and
     the lane install.  It creates the model, the Q8_0 HC tensors and the graph
     context, then calls the REAL llama_model_qwen4exp::graph::build_hc_mix via
     micro/m3-probe-hc-mix.cpp (a subclass that exposes the protected member) with
     `il = 0` trunk weights (inject live) or the output-mixer path (inject
     nullptr).  It never sets GGML_TENSOR_FLAG_NO_REUSE itself: the flag comes from
     the library under test, and the harness reads it back.

  2. the patch needed to let a caller supply the graph context
     (micro/m3-probe-overlay.h documents it): src/llama-graph.h gains

         #if defined(M3_PROBE_GRAPH_HOOK)
             void reset_context_view(ggml_context * c, ggml_cgraph * g);
         #endif

     and src/llama-graph.cpp its body (sets a non-owning view + gf).  The probe TU
     defines M3_PROBE_GRAPH_HOOK, so the production build is untouched and the
     probe needs no re-parenting of llama internals.

Compiler flags come from the llama tower's own recorded command
(mtp-hc-row-tail-candidate-20261025/commands/qwen4exp.cpp.argv.json), i.e. the
same -I/-D/-std set that compiles src/models/qwen4exp.cpp, because the probe TU
includes models.h/llama-graph.h/llama-model.h and must see exactly the headers
the model TU saw.  The HIP runtime headers come from the separate recorded HIP
command (mtp-m3-trio-20261028/build/ggml-cuda-m3t.argv.json): that command starts
with `-o <obj> -x hip -c -MF <dep>`; the earlier recipe kept a hand-picked subset
of its flags and thereby dropped `-x hip`, so the generated command could not find
hip/hip_runtime.h.  Fidelity is now the rule: for the HIP command only the output
naming (-o/-c/-MF/-MT/-MD) is dropped and every other argv element is preserved
verbatim, including `-x hip`; the harness is a .cpp compiled as HIP C++ because it
transitively pulls the HIP-facing ggml headers.

Usage: build_mixer_prod.py <outdir> <libdir>
Nothing here is executed by the engineer.
"""
import json
import os
import shlex
import sys

E = "/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024"
HIP_TEMPLATE = E + "/mtp-m3-trio-20261028/build/ggml-cuda-m3t.argv.json"
LLAMA_TEMPLATE = E + "/mtp-hc-row-tail-candidate-20261025/commands/qwen4exp.cpp.argv.json"
LLAMA_LINK = E + "/mtp-hc-row-tail-candidate-20261025/commands/llama-link.argv.json"
HERE = os.path.dirname(os.path.abspath(__file__))
# The candidate tree this file lives in (micro/ is one level below its root); the
# hook blocks below are applied to that tree's src/llama-graph.h by the operator.
WORKTREE = os.path.dirname(HERE)

# Hook block inserted into class llm_graph_result, right after the ctor
# declaration.  The owning context stays in ctx_compute; this view points at the
# device test's context and the getters prefer it only while it is non-null, so a
# build without the macro (and every production call with the view unset) behaves
# bit-for-bit as shipped.
PROBE_HOOK_H = """
#if defined(M3_PROBE_GRAPH_HOOK)
    ggml_context * ctx_compute_view = nullptr; // test-only, non-owning, default null
#endif
"""

# Replacements for the two shipped getters (they are tiny inline bodies; the
# patch is shown verbatim so a reviewer can check the null-view fallback).
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


def faithful_flags(template):
    """Every recorded flag, minus the ones that name lane-private outputs."""
    argv = json.load(open(template))["argv"]
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
    hip = faithful_flags(HIP_TEMPLATE)
    llama = faithful_flags(LLAMA_TEMPLATE)
    # drop the template's source operand and dependency switches; the probe
    # sources are appended explicitly below.
    def strip_sources(args):
        out = []
        skip = False
        for a in args:
            if skip:
                skip = False
                continue
            if a in ("-include",):
                out.append(a); skip = True; continue
            if a.endswith((".cpp", ".cu", ".c", ".cc")):
                continue
            out.append(a)
        return out
    hip = strip_sources(hip)
    llama = strip_sources(llama)
    # Source operands were stripped above; the probe sources are appended here.
    # The HIP command's `-x hip` is retained in `hip`, so the harness is compiled as
    # HIP C++ exactly as the lane compiles its own HIP TU.
    cmd = [cc] + llama + hip + [
        "-DM3_PROBE_GRAPH_HOOK",
        os.path.join(HERE, "m3-mixer-prod-path.cpp"),
        os.path.join(HERE, "m3-probe-hc-mix.cpp"),
        # Deliberately NOT compiled here: qwen4exp.cpp defines the whole model
        # library's symbols, and linking it beside libllama would duplicate them.
        # The probe instead exposes the protected member by subclassing the real
        # type declared in models.h and calls the definition that ships inside the
        # candidate libllama below, so the code under test is the shipped one.
        "-o", os.path.join(outdir, "m3-mixer-prod-path"),
        "-L" + libdir, "-L/opt/rocm/core-10.0/lib",
        "-lggml", "-lggml-base", "-lggml-hip",
        # The candidate libllama: llama_model_load_from_file, the graph context and
        # the production build_hc_mix definition all come from here.
        "-lllama",
        "-Wl,-rpath," + libdir, "-Wl,-rpath,/opt/rocm/core-10.0/lib",
    ]
    json.dump({"argv": cmd,
               "hook_header_patch": PROBE_HOOK_H,
               "hook_getter_patch": PROBE_GETTER_PATCH},
              open(os.path.join(outdir, "build.argv.json"), "w"), indent=1)
    print(shlex.join(cmd))
    print()
    print("# the only source change the probe needs, in the candidate tree (no model"
          " behaviour change, only compiled when M3_PROBE_GRAPH_HOOK is defined):")
    print("#   src/llama-graph.h : the hook member + the get_ctx()/reset_context_view()")
    print("#   replacements above; no change to the graph ctor or to any model code.")
    print()
    print("# run matrix (Devops, both GPU flocks; <bin> = path above):")
    print("# 1. N=1 reference images for the MIXER site, production gate OFF:")
    print("#    GGML_TP5_HC_N3_TRIO=1 M3_TAIL_DIAG=1 <bin> <libdir> 2 mixer ref-mixer")
    print("#    -> ref-mixer.nt1.0.bin, ref-mixer.nt1.1.bin + M3_TAIL_BUILD witness lines")
    print("# 2. candidate mixer legs (exact-1 gate, then the two diag scopes):")
    print("#    GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1 <bin> <libdir> 2 mixer on-mixer")
    print("#    GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_DIAG_SCOPE=mixer <bin> <libdir> 2 mixer diag-mixer")
    print("# 3. trunk control on the same stack (inject live, il=0):")
    print("#    GGML_TP5_HC_N3_TRIO=1 <bin> <libdir> 3 trunk ref-trunk")
    print("#    GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1 <bin> <libdir> 3 trunk on-trunk")
    print("# 4. compare rows with the N=1 images:")
    print("#    python3 diag/m3-n1-rowref.py <leg-prefix-dir> <ref-prefix-dir>")
    print("# 5. witness check: ON mixer leg must print lo ret=1 and xn ret=1, OFF must")
    print("#    print lo ret=0, and the trunk legs must print xn ret=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
