#!/usr/bin/env python3
"""Build A, each B, and each C from separate source copies, then consume relocated packages."""

import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory for isolated builds and logs")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--a-prefix", type=Path, help="reuse a previously verified installed A package")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []

    def run(name, command):
        command = [str(part) for part in command]
        print("+ " + " ".join(command), flush=True)
        with (output / f"{name}.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, cwd=output)
        records.append({"name": name, "command": command, "returncode": result.returncode})
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")

    def configure(name, source, build, prefix, dependencies=(), extra=()):
        command = ["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                   f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_INSTALL_RPATH=$ORIGIN",
                   "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"]
        if dependencies:
            command.append("-DCMAKE_PREFIX_PATH=" + ";".join(str(part) for part in dependencies))
        run(name + "-configure", command + list(extra))

    base = json.loads((root / "fork/upstream.json").read_text())["upstream"]["commit"]
    prefix_a = output / "package-a"
    if args.a_prefix:
        shutil.copytree(args.a_prefix.resolve(), prefix_a)
    else:
        archive = subprocess.check_output(["git", "-C", str(root), "archive", base])
        source_a = output / "source-a"
        source_a.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            # The archive is the pinned Git commit. Python's data filter also rejects
            # archive paths that escape this newly created source directory.
            tar.extractall(source_a, filter="data")
        if (source_a / "fork").exists():
            raise RuntimeError("upstream archive unexpectedly contains the extension")
        build_a = output / "build-a"
        configure("a", source_a, build_a, prefix_a, extra=[
            "-DGGML_NATIVE=OFF", "-DGGML_VULKAN=OFF", "-DGGML_CUDA=OFF", "-DGGML_HIP=OFF",
            "-DLLAMA_BUILD_COMMON=ON", "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_BUILD_TOOLS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_TESTS_INSTALL=OFF", "-DLLAMA_TOOLS_INSTALL=OFF",
        ])
        run("a-build", ["cmake", "--build", build_a, "--parallel", args.jobs, "--target", "test-sampling"])
        run("a-test", ["ctest", "--test-dir", build_a, "--output-on-failure", "--no-tests=error", "-R", "^test-sampling$"])
        run("a-install", ["cmake", "--install", build_a])
        shutil.rmtree(source_a)

    def module(name, dependencies):
        source = output / f"source-{name}"
        build = output / f"build-{name}"
        prefix = output / f"package-{name}"
        shutil.copytree(root / "fork" / name, source)
        configure(name, source, build, prefix, dependencies, ["-DBUILD_TESTING=ON"])
        run(name + "-build", ["cmake", "--build", build, "--parallel", args.jobs])
        run(name + "-test", ["ctest", "--test-dir", build, "--output-on-failure", "--no-tests=error"])
        run(name + "-install", ["cmake", "--install", build])
        shutil.rmtree(source)
        return prefix

    sampling = module("sampling", ())
    corpus = module("corpus", ())
    compose = module("compose", (prefix_a, sampling))
    corpus_compose = module("corpus-compose", (prefix_a, corpus))

    # Copy only installed products to a new prefix. Source copies are already
    # gone. The final consumer must resolve public headers and exported packages.
    delivered = output / "delivered"
    for prefix in (prefix_a, sampling, corpus, compose, corpus_compose):
        shutil.copytree(prefix, delivered, dirs_exist_ok=True)
        shutil.rmtree(prefix)
    consumer = output / "consumer"
    shutil.copytree(root / "fork/tests/package-consumer", consumer)
    configure("consumer", consumer, output / "build-consumer", output / "unused", (delivered,))
    run("consumer-build", ["cmake", "--build", output / "build-consumer", "--parallel", args.jobs])
    run("consumer-test", ["ctest", "--test-dir", output / "build-consumer", "--output-on-failure", "--no-tests=error"])
    run("installed-compose", [delivered / "bin/llama-compose", "--help"])
    run("installed-corpus", [delivered / "bin/llama-corpus", "--help"])
    result = {"upstream": base, "a_archive_has_no_b_or_c": not bool(args.a_prefix),
              "isolated_modules": ["sampling", "corpus", "compose", "corpus-compose"],
              "source_copies_removed_before_consumption": True,
              "relocated_installed_consumer_passed": True, "commands": len(records)}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
