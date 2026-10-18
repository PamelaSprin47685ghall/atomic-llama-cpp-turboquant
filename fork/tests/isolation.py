#!/usr/bin/env python3
"""Build pinned A plus verified overlays, isolated B/C, and relocated consumers."""

import argparse
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory for isolated builds and logs")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--a-prefix", type=Path, help="reuse installed A; source provenance is explicitly unverified")
    parser.add_argument("--model", type=Path, help="optional real GGUF for relocated batch execution and phase comparison")
    parser.add_argument("--gpu-layers", type=int, default=0)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.model and not args.model.is_file():
        parser.error("--model must be an existing GGUF file")
    root = Path(__file__).resolve().parents[2]
    boundary = runpy.run_path(str(root / "fork/check-upstream.py"))
    snapshot = boundary["verify"](root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "verified-boundary.json").write_text(json.dumps(snapshot["summary"], indent=2) + "\n")
    (output / "upstream-exceptions.json").write_bytes(snapshot["manifest_bytes"])
    records = []
    source_copies = []
    build_copies = []

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
                   "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
                   "-DCMAKE_PREFIX_PATH=" + ";".join(str(part) for part in dependencies)]
        run(name + "-configure", command + list(extra))
        build_copies.append(build)

    prefix_a = output / "package-a"
    if args.a_prefix:
        shutil.copytree(args.a_prefix.resolve(), prefix_a)
        provenance = "reused_unverified"
    else:
        # Never archive HEAD or the live worktree. Overlays use owned bytes already
        # checked against the manifest, including the recorded Git executable mode.
        archive = boundary["git"](root, "archive", snapshot["base"])
        source_a = output / "source-a"
        source_a.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            tar.extractall(source_a, filter="data")
        if (source_a / "fork").exists():
            raise RuntimeError("pinned upstream archive unexpectedly contains B or C")
        boundary["apply_verified_overlays"](snapshot, source_a)
        source_copies.append(source_a)
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
        shutil.rmtree(build_a)
        provenance = "built_from_pinned_archive_and_verified_overlay_bytes"

    def module(name, dependencies, source_relative=None, install=True):
        source = output / f"source-{name}"
        build = output / f"build-{name}"
        prefix = output / f"package-{name}"
        shutil.copytree(root / "fork" / (source_relative or name), source)
        source_copies.append(source)
        configure(name, source, build, prefix, dependencies, ["-DBUILD_TESTING=ON"])
        run(name + "-build", ["cmake", "--build", build, "--parallel", args.jobs])
        run(name + "-test", ["ctest", "--test-dir", build, "--output-on-failure", "--no-tests=error"])
        if install:
            run(name + "-install", ["cmake", "--install", build])
        shutil.rmtree(source)
        shutil.rmtree(build)
        return prefix

    # The seams are exercised only through installed public A headers and exports.
    module("a-seams", (prefix_a,), "tests/a-seams", install=False)
    sampling = module("sampling", ())
    corpus = module("corpus", ())
    scheduling = module("scheduling", ())
    compose = module("compose", (prefix_a, sampling))
    corpus_compose = module("corpus-compose", (prefix_a, corpus))
    batch_compose = module("batch-compose", (prefix_a, scheduling))
    logits_compose = module("logits-compose", (prefix_a,))
    real_test = output / "real-batch.py"
    real_logits_test = output / "real-logits.py"
    logits_comparator = output / "compare-logits.py"
    if args.model:
        shutil.copyfile(root / "fork/batch-compose/tests/real-batch.py", real_test)
        shutil.copyfile(root / "fork/logits-compose/tests/test_real_logits.py", real_logits_test)
        shutil.copyfile(root / "fork/tools/compare-logits.py", logits_comparator)

    delivered = output / "delivered"
    for prefix in (prefix_a, sampling, corpus, scheduling, compose, corpus_compose, batch_compose, logits_compose):
        shutil.copytree(prefix, delivered, dirs_exist_ok=True)
        shutil.rmtree(prefix)
    for path in source_copies + build_copies:
        if path.exists():
            raise RuntimeError(f"source or build copy was not removed: {path}")
    build_copies.clear()

    consumer = output / "consumer"
    shutil.copytree(root / "fork/tests/package-consumer", consumer)
    configure("consumer", consumer, output / "build-consumer", output / "unused", (delivered,))
    run("consumer-build", ["cmake", "--build", output / "build-consumer", "--parallel", args.jobs])
    run("consumer-test", ["ctest", "--test-dir", output / "build-consumer", "--output-on-failure", "--no-tests=error"])

    # A new consumer sees the relocated scheduling package, not its sources.
    scheduling_consumer = output / "scheduling-consumer"
    scheduling_consumer.mkdir()
    (scheduling_consumer / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(TQRelocatedScheduling LANGUAGES CXX)
enable_testing()
find_package(TQScheduling CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE TQScheduling::scheduling)
add_test(NAME relocated-scheduling COMMAND consumer)
''')
    (scheduling_consumer / "main.cpp").write_text('''#include <tq/scheduling/admission.h>
#include <tq/scheduling/prefill.h>
int main() {
    using namespace tq::scheduling;
    admission_budget budget(512, 1);
    const auto held = budget.try_acquire(7, 256);
    if (!held || budget.reserved_tokens() != 256 || budget.try_acquire(9, 256)) return 3;
    if (!budget.release(*held) || budget.release(*held) || budget.available_tokens() != 512) return 4;
    prefill_selector selector;
    auto prompt = selector.choose({{7, stage::generation}, {9, stage::prompt}});
    if (prompt.next != stage::prompt || prompt.prompt_slot != 9 || !prompt.generation_slots.empty()) return 1;
    auto resume = selector.choose({{7, stage::generation}, {9, stage::generation}});
    return resume.next == stage::generation && resume.generation_slots == std::vector<slot_id>{7, 9} ? 0 : 2;
}
''')
    configure("scheduling-consumer", scheduling_consumer, output / "build-scheduling-consumer", output / "unused", (delivered,))
    run("scheduling-consumer-build", ["cmake", "--build", output / "build-scheduling-consumer", "--parallel", args.jobs])
    run("scheduling-consumer-test", ["ctest", "--test-dir", output / "build-scheduling-consumer", "--output-on-failure", "--no-tests=error"])
    for label, binary in (("compose", "llama-compose"), ("corpus", "llama-corpus"),
                          ("batch", "tq-batch-compose"), ("logits", "tq-logits-compose")):
        run("installed-" + label, [delivered / "bin" / binary, "--help"])
    if args.model:
        run("relocated-real-batch", [sys.executable, real_test, "--binary", delivered / "bin/tq-batch-compose",
                                    "--model", args.model.resolve(), "--gpu-layers", args.gpu_layers,
                                    "--output-dir", output / "relocated-real-model"])
        run("relocated-real-logits", [sys.executable, real_logits_test, "--binary", delivered / "bin/tq-logits-compose",
                                     "--model", args.model.resolve(), "--gpu-layers", args.gpu_layers,
                                     "--comparator", logits_comparator, "--batch-binary", delivered / "bin/tq-batch-compose",
                                     "--output-dir", output / "relocated-real-logits"])
    result = {"upstream": snapshot["base"], "a_source_provenance": provenance,
              "a_archive_has_no_b_or_c": None if args.a_prefix else True,
              "a_reviewed_overlays_applied": None if args.a_prefix else snapshot["summary"]["reviewed_overlays"],
              "a_reuse_limit": "Installed A source, pin, overlays, and build provenance are not verified." if args.a_prefix else None,
              "a_public_seams_tests_passed": True,
              "isolated_modules": ["sampling", "corpus", "scheduling", "compose", "corpus-compose", "batch-compose", "logits-compose"],
              "source_and_build_copies_removed_before_consumption": True,
              "relocated_installed_consumers_passed": True,
              "relocated_real_batch": "passed" if args.model else "skipped_no_model",
              "relocated_real_logits": "passed" if args.model else "skipped_no_model",
              "limits": ["Help/package checks alone do not establish model execution or throughput.",
                         "Separate source copies and installed consumption do not constitute an OS sandbox."],
              "commands": len(records)}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
