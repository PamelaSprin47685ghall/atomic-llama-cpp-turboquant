#!/usr/bin/env python3
"""Interrupt real installed inference after observed concurrent BODY work.

This is a process-boundary test, separate from attention and device-drain
qualification. SIGINT must reach C's cancellation path. Closing the sole
stdout reader must reach C's output-error path, not terminate it by SIGPIPE.
The receipt retains raw observed output, commands, artifact hashes, exact
injection timing, exit codes, and any forced cleanup. No model is downloaded.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import subprocess
import sys
import time


class CheckError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise CheckError(message)


def integer(value, minimum=1):
    parsed = int(value)
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum}")
    return parsed


def fingerprint(path):
    path = path.resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
         f"artifact changed during hashing: {path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


def model_files(path):
    match = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", path.name)
    if not match:
        return [path]
    count = int(match.group(3))
    need(1 <= count <= 10000, "invalid GGUF shard count")
    return [path.with_name(f"{match.group(1)}-{index:05d}-of-{count:05d}.gguf")
            for index in range(1, count + 1)]


class Observation:
    """Validate the observed prefix without assuming an atomic stdout write."""

    def __init__(self):
        self.events = 0
        self.config = None
        self.frontier = 0
        self.revision = None
        self.readers = {}
        self.body_stages = set()
        self.tokens = set()
        self.live_body_cohorts = 0
        self.cohort_counted = False
        self.terminal = None

    def accept(self, line):
        event = json.loads(line.decode("utf-8", errors="strict"))
        need(isinstance(event, dict) and isinstance(event.get("event"), str), "invalid JSON event")
        need(self.terminal is None, "output continued after a terminal event")
        self.events += 1
        kind = event["event"]
        if kind == "config":
            need(self.events == 1 and self.config is None, "config was not the first and only config event")
            need(event.get("provider") == "native-a" and event.get("submission_domains") == 1,
                 "the process did not use the native A provider")
            need(event.get("workers") == 2 and event.get("parallel") == 2,
                 "the requested pair of concurrent writers was not configured")
            self.config = event
            return
        if kind in ("error", "result"):
            self.terminal = event
            return
        need(self.config is not None, "execution event appeared before configuration")
        if kind == "frontier":
            need(event.get("frontier") == self.frontier + 1, "skipped or repeated committed frontier")
            read = event.get("read_revision")
            need(type(read) is int and read >= 1 and (self.revision is None or read == self.revision),
                 "invalid frozen read revision")
            need(event.get("revision") == read + 1, "frontier did not publish exactly one revision")
            readers = event.get("readers")
            need(isinstance(readers, list) and 1 <= len(readers) <= 2, "invalid reader cohort")
            stages = [reader.get("stage") for reader in readers]
            need(all(type(stage) is int and 1 <= stage <= 3 for stage in stages)
                 and len(set(stages)) == len(stages), "invalid or duplicated reader stage")
            need(all(type(reader.get("starting")) is bool for reader in readers), "missing reader phase")
            self.frontier = event["frontier"]
            self.revision = event["revision"]
            self.readers = {reader["stage"]: reader for reader in readers}
            self.body_stages = set()
            self.tokens = set()
            self.cohort_counted = False
        elif kind == "token":
            stage = event.get("stage")
            need(stage in self.readers and stage not in self.tokens, "foreign or duplicate committed token")
            need(not self.readers[stage]["starting"], "FRAME was reported as a BODY token")
            need(event.get("frontier") == self.frontier and event.get("revision") == self.revision,
                 "token does not belong to its published cohort")
            need(type(event.get("token")) is int and event["token"] >= 0
                 and type(event.get("source_end")) is bool, "invalid token event")
            self.tokens.add(stage)
            if not event["source_end"]:
                self.body_stages.add(stage)
            # Both workers must have actually advanced a non-EOG BODY token
            # in this cohort. Inventory, FRAME work, or two sequential stages
            # are insufficient evidence for the injection trigger.
            if not self.cohort_counted and self.body_stages == {1, 2}:
                self.live_body_cohorts += 1
                self.cohort_counted = True
        else:
            need(kind in ("text", "sealed", "compute_release", "logits"), "unknown execution event")

    def summary(self):
        return {"events": self.events, "committed_frontiers": self.frontier,
                "last_committed_revision": self.revision,
                "live_body_cohorts": self.live_body_cohorts, "terminal_event": self.terminal}


def capture(command, output, name, kind, startup_timeout, stop_timeout, trigger_cohorts):
    stdout_path = output / f"{name}.stdout.jsonl"
    stderr_path = output / f"{name}.stderr.log"
    observed = Observation()
    started = time.monotonic()
    receipt = {"name": name, "kind": kind, "argv": command, "status": "RUNNING",
               "started_at": datetime.now(timezone.utc).isoformat(), "injection": None,
               "startup_timeout_seconds": startup_timeout, "stop_timeout_seconds": stop_timeout,
               "timed_out": False, "forced_cleanup": [], "violations": [],
               "stdout_line_arrival_seconds": []}
    pending = bytearray()
    parse_ok = True
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr,
                                   stdin=subprocess.DEVNULL, shell=False, start_new_session=True)
        receipt["pid"] = process.pid
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        stdout_open = True

        def close_stdout():
            nonlocal stdout_open
            if stdout_open:
                selector.unregister(process.stdout)
                process.stdout.close()
                stdout_open = False

        def cleanup(reason):
            if process.poll() is None:
                # Popen created a new session. Only this test's process group
                # is eligible for timeout cleanup; no unrelated server is hit.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    receipt["forced_cleanup"].append({"signal": "SIGKILL", "reason": reason,
                                                      "at_seconds": time.monotonic() - started})
                except ProcessLookupError:
                    pass

        try:
            while process.poll() is None or stdout_open:
                elapsed = time.monotonic() - started
                injected = receipt["injection"]
                deadline = startup_timeout if injected is None else injected["at_seconds"] + stop_timeout
                if elapsed > deadline:
                    receipt["timed_out"] = True
                    receipt["violations"].append("injection trigger timed out" if injected is None
                                                  else "controlled exit exceeded the post-injection timeout")
                    cleanup("timeout")
                    break
                if not stdout_open:
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                    continue
                for key, _ in selector.select(timeout=0.1):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        close_stdout()
                        continue
                    stdout.write(data)
                    stdout.flush()
                    if not parse_ok:
                        continue
                    pending.extend(data)
                    need(len(pending) <= 8 * 1024 * 1024, "one stdout event exceeded the bounded parser capacity")
                    while b"\n" in pending:
                        line, _, tail = pending.partition(b"\n")
                        pending[:] = tail
                        receipt["stdout_line_arrival_seconds"].append(time.monotonic() - started)
                        try:
                            observed.accept(line)
                        except (CheckError, ValueError, UnicodeError, TypeError, KeyError) as error:
                            receipt["violations"].append("invalid observed event prefix: " + str(error))
                            parse_ok = False
                            cleanup("invalid event stream")
                            break
                    # Parse the complete chunk before injection: all these
                    # bytes were already read. Never label buffered output as
                    # a new publication caused after the external signal.
                    if (parse_ok and receipt["injection"] is None and observed.terminal is None
                            and observed.live_body_cohorts >= trigger_cohorts):
                        need(process.poll() is None, "process exited before fault injection")
                        action = {"kind": kind, "at_seconds": time.monotonic() - started,
                                  "after_observed_events": observed.events,
                                  "after_committed_frontier": observed.frontier,
                                  "after_committed_revision": observed.revision,
                                  "after_live_body_cohorts": observed.live_body_cohorts}
                        if kind == "sigint":
                            process.send_signal(signal.SIGINT)
                            action["signal"] = "SIGINT"
                        else:
                            close_stdout()
                            action["operation"] = "close sole stdout read end"
                        receipt["injection"] = action
            process.wait(timeout=5)
        except (CheckError, OSError, subprocess.TimeoutExpired) as error:
            receipt["violations"].append(str(error))
            cleanup("driver error")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                receipt["violations"].append("could not reap the owned process after SIGKILL")
        finally:
            close_stdout()
            selector.close()
    receipt["returncode"] = process.returncode
    receipt["seconds"] = time.monotonic() - started
    receipt["seconds_after_injection"] = (receipt["seconds"] - receipt["injection"]["at_seconds"]
                                          if receipt["injection"] else None)
    receipt["observed"] = observed.summary()
    receipt["partial_stdout_tail_bytes"] = len(pending)
    receipt["stdout_intentionally_closed"] = kind == "broken-pipe" and receipt["injection"] is not None
    receipt["stdout"] = fingerprint(stdout_path)
    receipt["stderr"] = fingerprint(stderr_path)
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    offloads = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", stderr_text)
    devices = set()
    for match in re.finditer(r"([A-Za-z][A-Za-z0-9_.:-]*)\s+model buffer size\s*=\s*([0-9.]+)\s*MiB",
                             stderr_text):
        if not match.group(1).upper().startswith("CPU") and float(match.group(2)) > 0:
            devices.add(match.group(1))
    receipt["model_placement_observed"] = {
        "offloaded_layers": max((int(actual) for actual, _ in offloads), default=0),
        "non_cpu_model_buffer_devices": sorted(devices),
        "evidence": "actual positive model load buffer/offload logs; enumeration alone is excluded",
    }
    try:
        need(receipt["injection"] is not None, "no fault was injected after concurrent BODY work")
        need(not receipt["timed_out"] and not receipt["forced_cleanup"], "process required forced cleanup")
        need(observed.live_body_cohorts >= trigger_cohorts, "actual concurrent BODY trigger was not reached")
        need(observed.terminal is None or observed.terminal.get("event") != "result",
             "the interrupted run emitted a completed result")
        if kind == "sigint":
            need(process.returncode == 130, f"SIGINT should exit through C with 130, observed {process.returncode}")
            need(not pending, "SIGINT left a truncated JSON event on the still-open output pipe")
            terminal = observed.terminal
            need(terminal is not None and terminal.get("event") == "error"
                 and terminal.get("complete") is False and "cancelled" in terminal.get("message", ""),
                 "SIGINT did not emit a terminal, incomplete cancellation error")
            need("RERoT: cancelled" in stderr_text, "SIGINT cancellation diagnostic was absent from stderr")
        else:
            need(process.returncode == 1, f"broken pipe should reach C's error path with 1, observed {process.returncode}")
            need("RERoT: output write failed after the last committed frontier" in stderr_text,
                 "broken-pipe output-error diagnostic was absent from stderr")
    except CheckError as error:
        receipt["violations"].append(str(error))
    receipt["status"] = "FAIL" if receipt["violations"] else "PASS"
    receipt["not_qualified_by_this_run"] = [
        "exact native in-flight state at signal delivery", "independent device-drain proof",
        "attention numerics", "absence of unobserved stdout after the broken-pipe injection",
    ]
    return receipt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing receipts are never overwritten")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--source-commit", default="UNSPECIFIED")
    parser.add_argument("--cases", default="sigint,broken-pipe")
    parser.add_argument("--repeat", type=integer, default=1)
    parser.add_argument("--after-live-cohorts", type=integer, default=2)
    parser.add_argument("--startup-timeout", type=integer, default=300)
    parser.add_argument("--stop-timeout", type=integer, default=30)
    parser.add_argument("--threads", type=integer, default=4)
    parser.add_argument("--gpu-layers", type=lambda value: integer(value, 0), default=0)
    parser.add_argument("--split-mode", choices=("none", "layer", "row"), default="none")
    parser.add_argument("--devices")
    parser.add_argument("--ubatch-size", type=integer, default=128)
    return parser.parse_args()


def main():
    args = arguments()
    need(os.name == "posix", "this failure driver requires POSIX signals and pipe semantics")
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / "receipt.json"
    receipt = {"schema": "tq-rerot-real-process-failure/1", "status": "RUNNING",
               "source_commit_provided": args.source_commit, "platform": platform.platform(),
               "scope": "actual C process cancellation and broken-pipe cleanup after concurrent BODY work",
               "runs": []}

    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        kinds = args.cases.split(",")
        need(len(kinds) == len(set(kinds)) and all(kind in ("sigint", "broken-pipe") for kind in kinds),
             "--cases must contain distinct sigint and/or broken-pipe names")
        need(args.ubatch_size <= 512, "--ubatch-size exceeds the fixture's batch capacity of 512")
        receipt["driver"] = fingerprint(Path(__file__))
        receipt["binary"] = fingerprint(binary)
        receipt["model_files"] = [fingerprint(path) for path in model_files(model)]
        receipt["additional_artifacts"] = [fingerprint(path) for path in args.artifact]
        names = ("GGML_VK_DISABLE_F16", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                 "CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES")
        receipt["environment_allowlist"] = list(names)
        receipt["execution_environment"] = {name: os.environ[name] for name in names if name in os.environ}
        receipt["fixture"] = {"workers": 2, "parallel": 2, "worker_tokens": 512,
                              "final_tokens": 512, "after_live_body_cohorts": args.after_live_cohorts}
        command = [str(binary), "--model", str(model), "--json",
                   "--prompt", "This is a short copying exercise on a shared whiteboard. Both writers may give the same answer; agreement is expected. Give the requested brief reply and finish your turn.",
                   "--stage", "writer_a=Reply with only this sentence: Shared work brings us to the same answer.",
                   "--stage", "writer_b=Reply with only this sentence: Shared work brings us to the same answer.",
                   "--synthesis", "Reply with the single word complete.", "--parallel", "2",
                   "--worker-tokens", "512", "--final-tokens", "512", "--ctx-size", "4096",
                   "--total-kv", "4096", "--batch-size", "512", "--ubatch-size", str(args.ubatch_size),
                   "--threads", str(args.threads), "--gpu-layers", str(args.gpu_layers),
                   "--split-mode", args.split_mode, "--temperature", "0", "--seed", "0"]
        if args.devices:
            command.extend(("--devices", args.devices))
        for kind in kinds:
            for repeat in range(1, args.repeat + 1):
                name = f"{len(receipt['runs']) + 1:02d}-{kind}-{repeat}"
                receipt["runs"].append({"name": name, "argv": command, "status": "RUNNING"})
                save()
                print(f"RUN {name}", flush=True)
                run = capture(command, output, name, kind, args.startup_timeout,
                              args.stop_timeout, args.after_live_cohorts)
                if args.gpu_layers > 0 and run["model_placement_observed"]["offloaded_layers"] == 0:
                    run["violations"].append("GPU execution was requested but no actual layer offload was observed")
                    run["status"] = "FAIL"
                receipt["runs"][-1] = run
                save()
                print(f"{run['status']} {name}: exit {run['returncode']}; "
                      f"post-injection {run['seconds_after_injection']} s", flush=True)
        need(bool(receipt["runs"]) and all(run["status"] == "PASS" for run in receipt["runs"]),
             "one or more actual failure-injection runs failed")
        receipt["status"] = "PASS"
    except (CheckError, OSError, ValueError) as error:
        receipt["status"] = "FAIL"
        receipt["failure"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr)
    finally:
        save()
    print(str(receipt_path), flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
