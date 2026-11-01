#!/bin/bash
# Build the candidate libggml-hip (base libs + mmvf.cu recompiled with the
# opt-in exact N2/N3 load4 branch) without touching the selected installs.
#
# Fidelity ladder (G2):
#   link-relinkref : lane link argv with the snapshot mmvf object must reproduce
#                    the installed libggml-hip byte-for-byte.
#   compile-control: the unpatched source compiled with the lane compile argv
#                    must reproduce the snapshot mmvf object.
#   compile-on/off : patched object; OFF-vs-ON device text must differ only in
#                    the two multi-column kernels.
#
# usage: build.sh <tree> <i-base-prefix> <out-dir> [mmvf-compile-argv.json]
set -euo pipefail
tree=${1:?candidate tree with ggml/src}
ibase=${2:?install prefix with the lane base library (lib/libggml-hip.so*)}
out=${3:?output dir}
argv_json=${4:-}
mkdir -p "$out/obj" "$out/lib"
rocm=/opt/rocm/core-10.0

snap=$tree/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-ready-snapshot-consolidated-20261025
if [ ! -d "$snap" ]; then
  snap=${SNAP_DIR:-/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-ready-snapshot-consolidated-20261025}
fi
link_argv="$snap/commands/hip-link.argv.json"
snap_obj="$snap/objects/hip-current-040-mmvf.cu.o"
[ -f "$link_argv" ] || { echo "missing $link_argv" >&2; exit 2; }
[ -f "$snap_obj" ] || { echo "missing $snap_obj" >&2; exit 2; }

# Compile argv: explicit override, else the recorded lane template edited for
# this tree and output. Both are printed into build-receipt.json.
if [ -n "$argv_json" ]; then
  cc_json=$argv_json
else
  cc_json=$out/compile-mmvf.argv.json
  python3 - "$tree" "$out" > "$cc_json" <<'PY'
import json, sys
tree, out = sys.argv[1], sys.argv[2]
argv = [
    "/opt/rocm/core-10.0/lib/llvm/bin/clang++",
    "-DGGML_BACKEND_BUILD", "-DGGML_BACKEND_SHARED", "-DGGML_CUDA_NO_PEER_COPY",
    "-DGGML_HIP_NO_MMQ_MFMA", "-DGGML_HIP_NO_VMM", "-DGGML_SCHED_MAX_COPIES=4",
    "-DGGML_SHARED", "-DGGML_USE_HIP", "-DUSE_PROF_API=1", "-D_GNU_SOURCE",
    "-D_XOPEN_SOURCE=600", "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1",
    "-Dggml_hip_EXPORTS",
    f"-I{tree}/ggml/src/ggml-hip/..",
    f"-I{tree}/ggml/src/../include",
    "-funsafe-math-optimizations", "-O3", "-DNDEBUG", "-std=gnu++17",
    "--offload-arch=gfx1030", "-fPIC",
    "-o", f"{out}/obj/PLACEHOLDER.o",
    "-x", "hip", "-c", "PLACEHOLDER.cc",
    "-iquote", f"{tree}/ggml/src/ggml-cuda",
]
json.dump({"argv": argv, "source": "edited lane template (hc-down-batch-candidate-20261024/compile-mmvf.cu.argv.json)"},
          sys.stdout, indent=1)
PY
fi

python3 - "$cc_json" "$snap" "$tree" "$out" "$ibase" "$link_argv" "$snap_obj" <<'PY'
import hashlib, json, os, subprocess, sys, time
cc_json, snap, tree, out, ibase, link_argv, snap_obj = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
log = []

def run(name, argv):
    started = time.time()
    proc = subprocess.run(argv, capture_output=True)
    open(f"{out}/{name}.stderr", "wb").write(proc.stderr)
    json.dump({"argv": argv}, open(f"{out}/{name}.argv.json", "w"), indent=1)
    log.append([name, proc.returncode, round(time.time() - started, 1)])
    print(log[-1], flush=True)
    if proc.returncode:
        sys.stderr.write(proc.stderr.decode(errors="replace")[-4000:])
        raise SystemExit("FAIL " + name)

def compile_mmvf(src, tag):
    argv = json.load(open(cc_json))["argv"]
    argv = [f"{out}/obj/{tag}-mmvf.cu.o" if a == f"{out}/obj/PLACEHOLDER.o" else
            src if a == "PLACEHOLDER.cc" else a for a in argv]
    assert src in argv and f"{out}/obj/{tag}-mmvf.cu.o" in argv
    run(f"compile-{tag}", argv)
    return f"{out}/obj/{tag}-mmvf.cu.o"

def link(obj, tag):
    raw = json.load(open(link_argv))
    argv = raw["argv"] if isinstance(raw, dict) else raw
    out_lib = f"{out}/lib/libggml-hip-{tag}.so"
    argv = [out_lib if i == argv.index('-o') + 1 else obj if a == snap_obj else a for i, a in enumerate(argv)]
    assert argv.count(out_lib) == 1 and argv.count(obj) == 1 and (obj == snap_obj or snap_obj not in argv)
    run(f"link-{tag}", argv)
    return out_lib

# Control object from the pristine overlay bytes in git, so the candidate tree's
# own mmvf.cu is never overwritten. Keep the basename `mmvf.cu` so the ELF
# STT_FILE symbol matches the snapshot object byte-for-byte.
os.makedirs(f"{out}/src-control", exist_ok=True)
os.makedirs(f"{out}/src-on", exist_ok=True)
control_src = f"{out}/src-control/mmvf.cu"
with open(control_src, "wb") as f:
    f.write(subprocess.check_output(["git", "-C", tree, "show", "f1c5fafc0:ggml/src/ggml-cuda/mmvf.cu"]))
on_src = f"{out}/src-on/mmvf.cu"
subprocess.run(["cp", "-f", f"{tree}/ggml/src/ggml-cuda/mmvf.cu", on_src], check=True)

relink = link(snap_obj, "relinkref")
control = compile_mmvf(control_src, "control")
control_lib = link(control, "control")
cand = compile_mmvf(on_src, "on")
cand_lib = link(cand, "on")

installed = [f"{ibase}/lib/{n}" for n in sorted(os.listdir(f"{ibase}/lib")) if n.startswith("libggml-hip.so")]
receipt = {
    "steps": log,
    "argv": {"mmvf_compile": cc_json, "hip_link": link_argv},
    "sha256": {
        "installed_hip": {p: sha(p) for p in installed},
        "relinkref_lib": sha(relink),
        "snapshot_mmvf_obj": sha(snap_obj),
        "control_mmvf_obj": sha(control),
        "control_lib": sha(control_lib),
        "on_mmvf_obj": sha(cand),
        "on_lib": sha(cand_lib),
        "control_src": sha(control_src),
        "on_src": sha(on_src),
    },
    "fidelity": {
        "relinkref_eq_installed": any(sha(relink) == sha(p) for p in installed),
        "control_obj_eq_snapshot": sha(control) == sha(snap_obj),
        "control_lib_eq_installed": any(sha(control_lib) == sha(p) for p in installed),
        "on_obj_differs_from_snapshot": sha(cand) != sha(snap_obj),
    },
}
json.dump(receipt, open(f"{out}/build-receipt.json", "w"), indent=1)
print(json.dumps(receipt["fidelity"], indent=1))
if not (receipt["fidelity"]["relinkref_eq_installed"] and receipt["fidelity"]["control_obj_eq_snapshot"]):
    raise SystemExit("G2 fidelity failed: baseline does not reproduce")
open(f"{out}/BUILD_OK", "w").write(time.strftime('%F %T UTC', time.gmtime()) + "\n")
PY

# Device-ISA census: the candidate adds two load4 kernels and keeps every other
# load4 / mul_mat_vec_f instantiation. Info only; the numbers go to isa.log.
mkdir -p "$out/isa"
for f in "$out/obj/on-mmvf.cu.o" "$out/obj/control-mmvf.cu.o"; do
  "$rocm/lib/llvm/bin/llvm-objcopy" --dump-section .hip_fatbin="$f.fatbin" "$f" "$f.tmp"
  "$rocm/lib/llvm/bin/llvm-objdump" -d --no-show-raw-insn "$f.fatbin" > "$out/isa/$(basename "$f").s" 2>/dev/null || true
  rm -f "$f.tmp"
done
grep -o 'mul_mat_vec_bf16_load4[a-z_]*' "$out/isa/on-mmvf.cu.o.s" | sort | uniq -c > "$out/isa/on-kernels.txt" || true
grep -o 'mul_mat_vec_bf16_load4[a-z_]*' "$out/isa/control-mmvf.cu.o.s" | sort | uniq -c > "$out/isa/control-kernels.txt" || true
echo "candidate libs: $out/lib/libggml-hip-on.so (base libs + mmvf recompiled)" | tee "$out/isa/README"
