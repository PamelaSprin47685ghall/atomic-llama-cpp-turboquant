#!/usr/bin/env python3
"""Build the unchanged upstream package, then build and test the extensions."""

import argparse
from pathlib import Path
import subprocess


def run(command):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "vulkan", "cuda", "hip"), default="cpu")
    parser.add_argument("--build-dir", type=Path, default=Path("build-clean-fork"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tools", action="store_true", help="also build upstream llama-server and llama-bench")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parent.parent
    build = args.build_dir.resolve()
    prefix = build / "install"
    upstream = build / "upstream"
    extension = build / "extension"
    run([
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
        "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_TOOLS_INSTALL=OFF",
        "-DLLAMA_TESTS_INSTALL=OFF",
    ])
    targets = ["llama"]
    if args.tools:
        targets.extend(["llama-server", "llama-bench"])
    run(["cmake", "--build", upstream, "--parallel", args.jobs, "--target", *targets])
    run(["cmake", "--install", upstream])
    run([
        "cmake", "-S", root / "fork", "-B", extension,
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DBUILD_TESTING={'OFF' if args.no_test else 'ON'}",
    ])
    run(["cmake", "--build", extension, "--parallel", args.jobs])
    if not args.no_test:
        run(["ctest", "--test-dir", extension, "--output-on-failure"])
    run(["cmake", "--install", extension])
    print(f"Built {prefix / 'bin' / 'llama-compose'}")
    if args.tools:
        print(f"Upstream tools: {upstream / 'bin'}")


if __name__ == "__main__":
    main()
