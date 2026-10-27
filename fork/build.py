#!/usr/bin/env python3
"""Verify reviewed A changes, install A, then build each B and C independently."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(command):
    command = [str(part) for part in command]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "vulkan", "cuda", "hip"), default="cpu")
    parser.add_argument("--build-dir", type=Path, default=Path("build-clean-fork"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tools", action="store_true", help="also build upstream llama-server and llama-bench")
    parser.add_argument("--tp5", action="store_true", help="build the five-rank gfx1030 HIP profile")
    parser.add_argument("--hip-root", type=Path, help="ROCm prefix; otherwise query hipconfig --path")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if (args.tp5 or args.hip_root) and args.backend != "hip":
        parser.error("--tp5 and --hip-root require --backend hip")
    root = Path(__file__).resolve().parent.parent
    build = args.build_dir.resolve()
    prefix = build / "install"
    upstream = build / "upstream"
    guard = [sys.executable, root / "fork/check-upstream.py"]
    run(guard)
    configure = [
        "cmake", "-S", root, "-B", upstream,
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        f"-DGGML_VULKAN={'ON' if args.backend == 'vulkan' else 'OFF'}",
        f"-DGGML_CUDA={'ON' if args.backend == 'cuda' else 'OFF'}",
        f"-DGGML_HIP={'ON' if args.backend == 'hip' else 'OFF'}",
        f"-DLLAMA_BUILD_COMMON={'ON' if args.tools else 'OFF'}",
        f"-DLLAMA_BUILD_TOOLS={'ON' if args.tools else 'OFF'}",
        f"-DLLAMA_BUILD_SERVER={'ON' if args.tools else 'OFF'}",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_TOOLS_INSTALL=OFF", "-DLLAMA_TESTS_INSTALL=OFF",
    ]
    if args.backend == "hip":
        hip_root = args.hip_root
        if hip_root is None:
            hip_root = Path(subprocess.check_output(["hipconfig", "--path"], text=True).strip())
        hip_root = hip_root.resolve()
        compiler = hip_root / "lib/llvm/bin/clang++"
        if not compiler.is_file():
            parser.error(f"HIP compiler does not exist: {compiler}")
        inherited_prefix = os.environ.get("CMAKE_PREFIX_PATH", "").replace(os.pathsep, ";")
        configure.extend([
            f"-DCMAKE_PREFIX_PATH={hip_root};{inherited_prefix}",
            f"-DCMAKE_HIP_COMPILER={compiler}",
            f"-DCMAKE_INSTALL_RPATH=$ORIGIN;{hip_root / 'lib'}",
        ])
        if args.tp5:
            configure.extend(["-DCMAKE_HIP_ARCHITECTURES=gfx1030", "-DGGML_HIP_GRAPHS=OFF",
                              "-DGGML_HIP_MMQ_MFMA=OFF", "-DGGML_CUDA_FA=ON", "-DGGML_CUDA_NO_PEER_COPY=ON"])
    run(configure)
    # A installs mtmd whenever its tools directory is enabled.
    targets = ["llama"] + (["llama-server", "llama-bench", "mtmd"] if args.tools else [])
    run(["cmake", "--build", upstream, "--parallel", args.jobs, "--target", *targets])
    run(guard)
    run(["cmake", "--install", upstream])

    def module(name, installed_dependencies=False, install=True):
        module_build = build / "modules" / name
        command = ["cmake", "-S", root / "fork" / name, "-B", module_build,
                   "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}",
                   "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
                   "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
                   f"-DBUILD_TESTING={'OFF' if args.no_test else 'ON'}"]
        if installed_dependencies:
            command.append(f"-DCMAKE_PREFIX_PATH={prefix}")
        if name == "scheduling":
            command.append(f"-DTQ_SCHEDULING_BUILD_TESTS={'OFF' if args.no_test else 'ON'}")
        elif name == "rerot":
            command.append(f"-DTQ_REROT_BUILD_TESTS={'OFF' if args.no_test else 'ON'}")
        elif name == "relay-probe":
            command.extend([
                "-DTQ_RELAY_PROBE_BUILD_HIP=OFF",
                f"-DTQ_RELAY_PROBE_BUILD_TESTS={'OFF' if args.no_test else 'ON'}",
            ])
        run(command)
        run(["cmake", "--build", module_build, "--parallel", args.jobs])
        if not args.no_test:
            run(["ctest", "--test-dir", module_build, "--output-on-failure", "--no-tests=error"])
        if install:
            run(["cmake", "--install", module_build])

    for name in ("sampling", "corpus", "scheduling", "text", "calibration", "relay", "rerot"):
        module(name)
    # Check probe configuration without enabling another backend toolchain.
    # Build and install the HIP platform probe separately for hardware qualification.
    module("relay-probe", install=False)
    for name in ("compose", "corpus-compose", "batch-compose", "logits-compose", "imatrix-compose",
                 "rerot-compose", "tp5-compose"):
        module(name, installed_dependencies=True)
    if not args.no_test:
        module("tests/a-seams", installed_dependencies=True, install=False)
    run(guard)
    print(f"Installed independent B/C packages and executables in {prefix}")
    if args.tools:
        print(f"Upstream tools: {upstream / 'bin'}")


if __name__ == "__main__":
    main()
