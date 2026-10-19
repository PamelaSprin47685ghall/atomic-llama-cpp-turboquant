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
    rerot_options = parser.add_argument_group("optional relocated RERoT execution")
    rerot_options.add_argument("--rerot-model", type=Path,
                               help="real GGUF supported by the reader-view seam and assistant-eot profile")
    rerot_options.add_argument("--rerot-chat-template", help="explicit native chat template for the RERoT model")
    rerot_options.add_argument("--rerot-ubatch-sizes", default="128,1",
                               help="comma-separated microbatch sizes, each at most 512")
    rerot_options.add_argument("--rerot-phase-modes", default="release,retain",
                               help="release, retain, or release,retain")
    rerot_options.add_argument("--rerot-split-mode", choices=("none", "layer", "row"), default="none")
    rerot_options.add_argument("--rerot-devices", help="explicit A offload device names")
    rerot_options.add_argument("--rerot-require-multi-device", action="store_true",
                               help="require real model buffers on at least two non-CPU devices")
    rerot_options.add_argument("--rerot-timeout", type=int, default=600,
                               help="timeout in seconds for each real RERoT run")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.model and not args.model.is_file():
        parser.error("--model must be an existing GGUF file")
    if args.rerot_model and not args.rerot_model.is_file():
        parser.error("--rerot-model must be an existing GGUF file")
    if args.rerot_model and args.gpu_layers < 0:
        parser.error("RERoT --gpu-layers must be non-negative")
    if args.rerot_timeout < 1:
        parser.error("--rerot-timeout must be positive")
    try:
        rerot_ubatches = [int(value) for value in args.rerot_ubatch_sizes.split(",")]
    except ValueError:
        parser.error("--rerot-ubatch-sizes requires comma-separated positive integers")
    if (not all(1 <= value <= 512 for value in rerot_ubatches) or
            len(set(rerot_ubatches)) != len(rerot_ubatches)):
        parser.error("--rerot-ubatch-sizes must be unique values from 1 through 512")
    rerot_phases = args.rerot_phase_modes.split(",")
    if (len(set(rerot_phases)) != len(rerot_phases) or
            not set(rerot_phases).issubset({"release", "retain"})):
        parser.error("--rerot-phase-modes requires unique release or retain values")
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

    def remove_copy(path):
        # Every removable tree is a direct child that this run created. Never
        # remove the user's repository or the input supplied with --a-prefix.
        if (path.parent != output or path.resolve().parent != output or path.is_symlink() or
                not path.name.startswith(("source-", "build-", "package-"))):
            raise RuntimeError(f"refusing to remove a directory outside this run's copies: {path}")
        shutil.rmtree(path)

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
        remove_copy(source_a)
        remove_copy(build_a)
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
        remove_copy(source)
        remove_copy(build)
        return prefix

    # The seams are exercised only through installed public A headers and exports.
    module("a-seams", (prefix_a,), "tests/a-seams", install=False)
    sampling = module("sampling", ())
    corpus = module("corpus", ())
    scheduling = module("scheduling", ())
    text_package = module("text", ())
    calibration = module("calibration", ())
    relay = module("relay", ())
    rerot = module("rerot", ())
    compose = module("compose", (prefix_a, sampling))
    corpus_compose = module("corpus-compose", (prefix_a, corpus))
    batch_compose = module("batch-compose", (prefix_a, scheduling, text_package))
    logits_compose = module("logits-compose", (prefix_a,))
    imatrix_compose = module("imatrix-compose", (prefix_a, calibration))
    rerot_compose = module("rerot-compose", (prefix_a, rerot, relay, text_package, sampling))
    conversion_test = output / "imatrix-conversion.py"
    shutil.copyfile(root / "fork/imatrix-compose/tests/test_conversion.py", conversion_test)
    rerot_cli_test = output / "rerot-cli.py"
    shutil.copyfile(root / "fork/rerot-compose/tests/test_cli.py", rerot_cli_test)
    real_test = output / "real-batch.py"
    real_logits_test = output / "real-logits.py"
    real_stream_test = output / "test_streaming.py"
    logits_comparator = output / "compare-logits.py"
    if args.model:
        shutil.copyfile(root / "fork/batch-compose/tests/real-batch.py", real_test)
        shutil.copyfile(root / "fork/logits-compose/tests/test_real_logits.py", real_logits_test)
        shutil.copyfile(root / "fork/batch-compose/tests/test_streaming.py", real_stream_test)
        shutil.copyfile(root / "fork/tools/compare-logits.py", logits_comparator)
    real_rerot_test = output / "real-rerot-whiteboard.py"
    if args.rerot_model:
        shutil.copyfile(root / "fork/rerot-compose/tests/real-whiteboard.py", real_rerot_test)

    delivered = output / "delivered"
    for prefix in (prefix_a, sampling, corpus, scheduling, text_package, calibration, relay, rerot,
                   compose, corpus_compose, batch_compose, logits_compose, imatrix_compose, rerot_compose):
        shutil.copytree(prefix, delivered, dirs_exist_ok=True)
        remove_copy(prefix)
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

    # All implementation and test-module build copies and original package
    # prefixes are gone. This new source contains only a public B consumer.
    rerot_consumer = output / "rerot-consumer"
    shutil.copytree(root / "fork/rerot/tests/installed-consumer", rerot_consumer)
    configure("rerot-consumer", rerot_consumer, output / "build-rerot-consumer", output / "unused", (delivered,))
    run("rerot-consumer-build", ["cmake", "--build", output / "build-rerot-consumer", "--parallel", args.jobs])
    run("rerot-consumer-test", ["ctest", "--test-dir", output / "build-rerot-consumer", "--output-on-failure", "--no-tests=error"])
    for label, binary in (("compose", "llama-compose"), ("corpus", "llama-corpus"),
                          ("batch", "tq-batch-compose"), ("logits", "tq-logits-compose"),
                          ("imatrix", "tq-imatrix-compose"), ("rerot", "tq-rerot-compose")):
        run("installed-" + label, [delivered / "bin" / binary, "--help"])
    run("relocated-imatrix-conversion", [sys.executable, conversion_test,
                                         "--binary", delivered / "bin/tq-imatrix-compose"])
    run("relocated-rerot-cli", [sys.executable, rerot_cli_test,
                                "--binary", delivered / "bin/tq-rerot-compose"])
    if args.model:
        run("relocated-real-batch", [sys.executable, real_test, "--binary", delivered / "bin/tq-batch-compose",
                                    "--model", args.model.resolve(), "--gpu-layers", args.gpu_layers,
                                    "--output-dir", output / "relocated-real-model"])
        run("relocated-real-logits", [sys.executable, real_logits_test, "--binary", delivered / "bin/tq-logits-compose",
                                     "--model", args.model.resolve(), "--gpu-layers", args.gpu_layers,
                                     "--comparator", logits_comparator, "--batch-binary", delivered / "bin/tq-batch-compose",
                                     "--output-dir", output / "relocated-real-logits"])
        run("relocated-real-stream", [sys.executable, real_stream_test, "--binary", delivered / "bin/tq-batch-compose",
                                     "--model", args.model.resolve(), "--gpu-layers", args.gpu_layers,
                                     "--output-dir", output / "relocated-real-stream"])
    if args.rerot_model:
        command = [sys.executable, real_rerot_test,
                   "--binary", delivered / "bin/tq-rerot-compose",
                   "--model", args.rerot_model.resolve(), "--gpu-layers", args.gpu_layers,
                   "--ubatch-sizes", ",".join(str(value) for value in rerot_ubatches),
                   "--phase-modes", args.rerot_phase_modes,
                   "--split-mode", args.rerot_split_mode, "--timeout", args.rerot_timeout,
                   "--output", output / "relocated-real-rerot"]
        if args.rerot_chat_template is not None:
            command.extend(("--chat-template", args.rerot_chat_template))
        if args.rerot_devices is not None:
            command.extend(("--devices", args.rerot_devices))
        if args.rerot_require_multi_device:
            command.append("--require-multi-device")
        run("relocated-real-rerot", command)
    result = {"upstream": snapshot["base"], "a_source_provenance": provenance,
              "a_archive_has_no_b_or_c": None if args.a_prefix else True,
              "a_reviewed_overlays_applied": None if args.a_prefix else snapshot["summary"]["reviewed_overlays"],
              "a_reuse_limit": "Installed A source, pin, overlays, and build provenance are not verified." if args.a_prefix else None,
              "a_public_seams_tests_passed": True,
              "isolated_modules": ["sampling", "corpus", "scheduling", "text", "calibration", "relay", "rerot",
                                   "compose", "corpus-compose", "batch-compose", "logits-compose", "imatrix-compose",
                                   "rerot-compose"],
              "source_and_build_copies_removed_before_consumption": True,
              "relocated_installed_consumers_passed": True,
              "relocated_imatrix_conversion": "passed",
              "relocated_rerot_consumer": "passed",
              "relocated_rerot_cli": "passed",
              "relocated_real_rerot": "passed" if args.rerot_model else "skipped_no_rerot_model",
              "relocated_real_rerot_receipt": "relocated-real-rerot/receipt.json" if args.rerot_model else None,
              "relocated_real_batch": "passed" if args.model else "skipped_no_model",
              "relocated_real_logits": "passed" if args.model else "skipped_no_model",
              "relocated_real_stream": "passed" if args.model else "skipped_no_model",
              "limits": ["Help/package checks alone do not establish model execution or throughput.",
                         "Imatrix conversion uses synthetic format fixtures, not model calibration or quantization quality measurements.",
                         "RERoT event-contract qualification does not establish attention numerical equivalence, TP5, MMVQ dispatch, LateBind, MTP, or performance benefit.",
                         "Separate source copies and installed consumption do not constitute an OS sandbox."],
              "commands": len(records)}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
