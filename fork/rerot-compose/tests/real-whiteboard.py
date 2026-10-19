#!/usr/bin/env python3
"""Run real installed inference and check the incremental whiteboard contract.

The receipt distinguishes event-contract evidence from attention correctness.
Run the real A reader-view numerical tests as a separate release gate. Neither
enumerated devices nor a requested split mode prove a multi-device execution.
Phase and repetition controls compare only the same execution shape. Cross-
shape greedy equivalence is an observation unless explicitly required.
No model is downloaded, and no generated response is substituted for a run.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import subprocess
import sys
import threading
import time


class ContractError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ContractError(message)


def number(value, minimum=0):
    return type(value) is int and value >= minimum


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
    need(1 <= count <= 10000, "invalid model shard count")
    return [path.with_name(f"{match.group(1)}-{index:05d}-of-{count:05d}.gguf")
            for index in range(1, count + 1)]


def capture(command, output, name, timeout):
    """Drain stdout as it is emitted; keep stderr separate from JSON events."""
    stdout_path = output / f"{name}.stdout.jsonl"
    stderr_path = output / f"{name}.stderr.log"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    lines = queue.Queue()
    arrivals = []
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr,
                                   stdin=subprocess.DEVNULL, shell=False)

        def drain():
            try:
                for line in iter(process.stdout.readline, b""):
                    lines.put((time.monotonic() - started, line))
            finally:
                lines.put(None)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        while True:
            if not timed_out and time.monotonic() - started > timeout:
                process.kill()
                timed_out = True
            try:
                item = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            arrival, line = item
            stdout.write(line)
            arrivals.append(arrival)
        process.wait()
        reader.join()
        process.stdout.close()
    return {
        "argv": command, "started_at": started_at,
        "seconds": time.monotonic() - started,
        "returncode": process.returncode, "timed_out": timed_out,
        "stdout": fingerprint(stdout_path), "stderr": fingerprint(stderr_path),
        "line_arrival_seconds": arrivals,
    }


def read_events(path):
    events = []
    with Path(path).open(encoding="utf-8", errors="strict") as source:
        for index, line in enumerate(source, 1):
            need(bool(line.strip()), f"empty JSON event at line {index}")
            event = json.loads(line)
            need(isinstance(event, dict) and isinstance(event.get("event"), str),
                 f"invalid event at line {index}")
            events.append(event)
    return events


def expand(reader, context):
    """Recover key identities independently of reader-relative positions."""
    need(number(reader.get("keys"), 1) and reader["keys"] <= context,
         "reader key count exceeds its declared context")
    need(isinstance(reader.get("spans"), list), "reader spans are missing")
    groups = defaultdict(list)
    order = []
    virtual = 0
    all_keys = set()
    for span in reader["spans"]:
        for name in ("stage", "run", "offset", "storage", "virtual"):
            need(number(span.get(name)), f"invalid span {name}")
        need(number(span.get("count"), 1), "empty or invalid span")
        need(span["virtual"] == virtual, "reader view has a gap or overlap")
        need(virtual + span["count"] <= reader["keys"], "span exceeds view bounds")
        owner = span["stage"]
        if not order or order[-1] != owner:
            need(owner not in order, "one owner's tape is interleaved with another owner")
            order.append(owner)
        for offset in range(span["count"]):
            key = (span["run"], span["offset"] + offset, span["storage"] + offset)
            identity = (owner, key[0], key[1])
            need(identity not in all_keys, "a visible key was counted more than once")
            all_keys.add(identity)
            groups[owner].append(key)
        virtual += span["count"]
    need(virtual == reader["keys"], "key count differs from the actual span lengths")
    need(reader.get("query_virtual") == virtual - 1, "own query is not at the end of its view")
    return dict(groups), order


def validate_trace(events, workers, parallel, edges, min_growth):
    """Check publication with prior committed keys, never with current peers."""
    need(bool(events) and events[0]["event"] == "config", "missing initial config")
    config = events[0]
    synthesis = workers + 1
    need(config.get("workers") == workers, "configured worker count differs from the request")
    need(config.get("parallel") == min(workers, parallel), "resident worker limit changed")
    need(config.get("synthesis_stage") == synthesis, "synthesis identity changed")
    need(config.get("provider") == "native-a" and config.get("submission_domains") == 1,
         "unexpected provider or fabricated per-device submission receipts")
    need(number(config.get("logical_context"), 1), "invalid logical context")
    need(number(config.get("physical_kv"), 1), "invalid physical KV capacity")
    need(number(config.get("base_tokens"), 1), "empty base prefix")
    need(number(config.get("logical_sequences"), synthesis + 1),
         "logical sequence identities cannot hold the requested stages")

    groups = []
    result = None
    for index, event in enumerate(events[1:], 1):
        kind = event["event"]
        need(result is None, "an event was emitted after the completed result")
        if kind == "frontier":
            groups.append((event, []))
        elif kind == "result":
            need(index == len(events) - 1 and event.get("complete") is True,
                 "result is incomplete or not terminal")
            result = event
        elif kind == "error":
            raise ContractError("native run failed: " + str(event.get("message")))
        elif kind == "compute_release":
            need(number(event.get("before")) and number(event.get("after")),
                 "invalid compute release observation")
        else:
            need(kind in ("token", "text", "sealed", "logits") and bool(groups),
                 f"unexpected or premature event: {kind}")
            groups[-1][1].append(event)
    need(result is not None and bool(groups), "run has no completed result")

    full = defaultdict(list)
    public = defaultdict(list)
    bodies = defaultdict(int)
    texts = defaultdict(str)
    finished_text = set()
    sealed = set()
    alive = {}
    started = set()
    base = None
    revision = None
    last_body = {}
    previously_visible = {}
    growth = defaultdict(int)
    body_reads = 0
    maximum_readers = 0
    maximum_rows = 0
    trace_tokens = []
    stage_frames = {}

    for expected_frontier, (frontier, after) in enumerate(groups, 1):
        stamp = f"frontier {expected_frontier}"
        need(frontier.get("frontier") == expected_frontier, stamp + ": skipped or duplicate commit")
        read_revision = frontier.get("read_revision")
        need(number(read_revision, 1), stamp + ": invalid read revision")
        need(revision is None or read_revision == revision, stamp + ": stale or future read revision")
        need(frontier.get("revision") == read_revision + 1, stamp + ": commit did not advance once")
        revision = frontier["revision"]
        readers = frontier.get("readers")
        need(isinstance(readers, list) and 1 <= len(readers) <= config["parallel"],
             stamp + ": invalid cohort size")
        ids = [reader.get("stage") for reader in readers]
        pens = [reader.get("pen") for reader in readers]
        need(all(number(stage, 1) and stage <= synthesis for stage in ids), stamp + ": invalid reader")
        need(len(set(ids)) == len(ids), stamp + ": duplicate reader")
        need(all(number(pen) and pen < config["parallel"] for pen in pens), stamp + ": invalid pen")
        need(len(set(pens)) == len(pens), stamp + ": two readers share a resident pen")
        need(set(alive).issubset(ids), stamp + ": a live insertion point was omitted")
        tokens = {}
        ends = set()
        for event in after:
            if event["event"] != "token":
                continue
            stage = event.get("stage")
            need(stage in ids and stage not in tokens, stamp + ": duplicate or foreign token event")
            need(event.get("frontier") == expected_frontier and event.get("revision") == revision,
                 stamp + ": token was published with the wrong commit")
            need(number(event.get("token")) and type(event.get("source_end")) is bool,
                 stamp + ": invalid token or source-end provenance")
            tokens[stage] = event
            if event["source_end"]:
                ends.add(stage)

        additions = {}
        for reader in readers:
            stage, pen = reader["stage"], reader["pen"]
            starting = reader.get("starting")
            need(type(starting) is bool, stamp + ": starting flag is absent")
            need(stage not in sealed, stamp + ": sealed stage ran again")
            if starting:
                need(stage not in started and pen not in alive.values(), stamp + ": early pen reuse")
                need(stage not in tokens, stamp + ": FRAME was reported as a body token")
                if stage == synthesis:
                    need(set(range(1, synthesis)).issubset(sealed), stamp + ": synthesis started early")
                for parent, child in edges:
                    need(child != stage or parent in sealed, stamp + ": dependency started before source end")
            else:
                need(alive.get(stage) == pen, stamp + ": a bound reader changed pens")
                need(stage in tokens, stamp + ": a running reader lacks its one token")

            visible, order = expand(reader, config["logical_context"])
            need(order[0] == 0 and order[-1] == stage, stamp + ": prefix or own tape is misplaced")
            if base is None:
                base = visible.get(0)
                need(base is not None and len(base) == config["base_tokens"], stamp + ": base count mismatch")
            need(visible.get(0) == base, stamp + ": base keys changed")
            own = visible.get(stage, [])
            need(own[:len(full[stage])] == full[stage], stamp + ": own committed keys changed")
            added = own[len(full[stage]):]
            need(len(added) >= 1 if starting else len(added) == 1,
                 stamp + ": body step did not advance exactly one key")
            additions[stage] = added
            if starting:
                stage_frames[stage] = len(added)
            for peer in set(public) | set(visible):
                if peer in (0, stage):
                    continue
                need(1 <= peer <= workers, stamp + ": unexpected peer identity")
                need(visible.get(peer, []) == public[peer],
                     stamp + f": reader {stage} did not see exactly the prior committed public keys of {peer}")
                pair = (stage, peer)
                count = len(public[peer])
                previous = previously_visible.get(pair)
                if peer in alive and bodies[peer] > 0:
                    body_reads += 1
                    if (previous is not None and previous[0] == expected_frontier - 1 and
                            count == previous[1] + 1 and last_body.get(peer) == expected_frontier - 1):
                        growth[pair] += 1
                previously_visible[pair] = (expected_frontier, count)
            for parent, child in edges:
                if parent in order and child in order:
                    need(order.index(parent) < order.index(child), stamp + ": visible order violates the DAG")

        need(frontier.get("rows") == sum(map(len, additions.values())),
             stamp + ": physical row count does not match the cohort writes")
        maximum_readers = max(maximum_readers, len(readers))
        maximum_rows = max(maximum_rows, frontier["rows"])
        for reader in readers:
            stage = reader["stage"]
            started.add(stage)
            alive[stage] = reader["pen"]
            full[stage].extend(additions[stage])
            if stage not in ends:
                public[stage].extend(additions[stage])
            if stage in tokens:
                event = tokens[stage]
                trace_tokens.append((expected_frontier, stage, event["token"], event["source_end"]))
                if not event["source_end"]:
                    bodies[stage] += 1
                    last_body[stage] = expected_frontier
        seen_tokens = set()
        step_sealed = set()
        for event in after:
            stage = event.get("stage")
            if event["event"] == "token":
                seen_tokens.add(stage)
            elif event["event"] == "text":
                need(stage in seen_tokens and stage not in finished_text,
                     stamp + ": text preceded commit or followed stream completion")
                need(isinstance(event.get("text"), str) and type(event.get("finished")) is bool,
                     stamp + ": invalid text event")
                need(event.get("offset") == len(texts[stage].encode("utf-8")),
                     stamp + ": text byte offset changed")
                texts[stage] += event["text"]
                if event["finished"]:
                    need(stage in ends, stamp + ": stream finished without native source end")
                    finished_text.add(stage)
            elif event["event"] == "sealed":
                need(stage in ends and stage not in step_sealed and stage in finished_text,
                     stamp + ": early or duplicate retirement")
                step_sealed.add(stage)
        need(step_sealed == ends, stamp + ": source-end token was not retired exactly once")
        for stage in ends:
            sealed.add(stage)
            del alive[stage]

    need(not alive and sealed == set(range(1, synthesis + 1)), "not every source retired")
    need(result.get("revision") == revision, "result revision differs from the last commit")
    need(result.get("text") == texts[synthesis], "result does not equal the committed text stream")
    need(result.get("body_tokens") == bodies[synthesis], "result token count includes uncommitted tokens")
    if min_growth:
        need(max(growth.values(), default=0) >= min_growth,
             "insufficient coverage: no concurrent reader observed the required peer token increments")
    return {
        "status": "PASS", "scope": "real model execution and emitted whiteboard contract",
        "frontiers": len(groups), "maximum_readers": maximum_readers,
        "maximum_rows": maximum_rows, "body_tokens_by_stage": dict(bodies),
        "frame_tokens_by_stage": stage_frames, "peer_body_reads_while_running": body_reads,
        "peer_growth_witnesses": [
            {"reader": pair[0], "peer": pair[1], "increments": count}
            for pair, count in sorted(growth.items())
        ],
        "config": config, "token_trace": trace_tokens,
        "attention_numerical_equivalence": "NOT_EVALUATED_BY_EVENT_TRACE",
    }


def placement(stderr):
    offloads = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", stderr)
    devices = set()
    for match in re.finditer(
            r"([A-Za-z][A-Za-z0-9_.:-]*)\s+model buffer size\s*=\s*([0-9.]+)\s*MiB",
            stderr):
        if not match.group(1).upper().startswith("CPU") and float(match.group(2)) > 0:
            devices.add(match.group(1))
    return {
        "offloaded_layers": max((int(a) for a, _ in offloads), default=0),
        "model_buffer_devices": sorted(devices),
        "evidence": "actual model load log; device enumeration alone is excluded",
    }


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


class TrajectoryControls:
    """Compare observed runs without assuming native cross-shape invariance."""

    def __init__(self, require_cross_shape):
        self.require_cross_shape = require_cross_shape
        self.groups = {}
        self.group_receipts = []
        self.cross_baseline = None
        self.results = {"phase_memory_invariance": [], "same_shape_reproducibility": []}
        self.cross_observations = []

    def check(self, group_id, ubatch, phase, trajectory, run, observed_placement, native_shape=None):
        name = run["name"]
        placement_profile = {key: observed_placement[key]
                             for key in ("offloaded_layers", "model_buffer_devices")}
        run["control_group"] = group_id
        group = self.groups.get(group_id)
        if group is None:
            public = {"id": group_id, "ubatch_size": ubatch,
                      "actual_model_placement": placement_profile,
                      "native_context_shape": native_shape,
                      "baseline_run": name, "runs": [name], "phases": [phase]}
            group = {"public": public, "trajectory": trajectory, "phases": {phase: (name, trajectory)}}
            self.groups[group_id] = group
            self.group_receipts.append(public)
            run["same_shape_control"] = {"status": "BASELINE", "reference_run": None}
            if self.cross_baseline is None:
                self.cross_baseline = (group_id, name, trajectory)
                run["cross_shape_observation"] = {"status": "BASELINE"}
            else:
                reference_group, reference_name, reference = self.cross_baseline
                equal = trajectory == reference
                observation = {"reference_group": reference_group, "group": group_id,
                               "reference_run": reference_name, "run": name,
                               "greedy_trajectory_equal": equal}
                self.cross_observations.append(observation)
                run["cross_shape_observation"] = observation
                need(equal or not self.require_cross_shape,
                     f"{name}: explicitly required cross-shape greedy equivalence failed")
            return

        public = group["public"]
        public["runs"].append(name)
        need(placement_profile == public["actual_model_placement"],
             f"{name}: actual model placement changed within the requested execution profile")
        need(native_shape == public["native_context_shape"],
             f"{name}: native context shape changed within one control group")
        if phase in group["phases"]:
            kind = "same_shape_reproducibility"
            reference_name, reference = group["phases"][phase]
        else:
            kind = "phase_memory_invariance"
            reference_name, reference = public["baseline_run"], group["trajectory"]
            group["phases"][phase] = (name, trajectory)
            public["phases"].append(phase)
        equal = trajectory == reference
        result = {"group": group_id, "reference_run": reference_name, "run": name,
                  "status": "PASS" if equal else "FAIL", "greedy_trajectory_equal": equal}
        self.results[kind].append(result)
        run["same_shape_control"] = {"kind": kind, **result}
        need(equal, f"{name}: {kind} changed the greedy trajectory within one execution shape")

    def report(self, shape_count, phase_count, repetitions):
        expected = {
            "phase_memory_invariance": shape_count * max(0, phase_count - 1),
            "same_shape_reproducibility": shape_count * phase_count * max(0, repetitions - 1),
        }
        report = {}
        for kind, results in self.results.items():
            failed = any(item["status"] == "FAIL" for item in results)
            complete = expected[kind] > 0 and len(results) == expected[kind]
            report[kind] = {"status": "FAIL" if failed else "PASS" if complete else "NOT_QUALIFIED",
                            "expected_comparisons": expected[kind], "comparisons": results}
        cross_expected = max(0, shape_count - 1)
        cross_complete = cross_expected > 0 and len(self.cross_observations) == cross_expected
        cross_failed = any(not item["greedy_trajectory_equal"] for item in self.cross_observations)
        if not self.require_cross_shape:
            status = "NOT_QUALIFIED"
        elif cross_failed:
            status = "FAIL"
        elif cross_complete:
            status = "PASS"
        else:
            status = "NOT_QUALIFIED"
        report["cross_shape_greedy_equivalence"] = {
            "status": status, "explicitly_required": self.require_cross_shape,
            "expected_comparisons": cross_expected, "observations": self.cross_observations,
            "scope": "observed executed shapes only; no general native batch-invariance claim",
        }
        return report


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory for full run logs and the JSON receipt")
    parser.add_argument("--artifact", action="append", type=Path, default=[],
                        help="Additional installed A/B shared library to hash; repeat")
    parser.add_argument("--source-commit", default="UNSPECIFIED")
    parser.add_argument("--prompt", default=(
        "This is a short copying exercise on a shared whiteboard. "
        "Both writers may give the same answer; agreement is expected. "
        "Give the requested brief reply and finish your turn."))
    parser.add_argument("--stage", action="append",
                        help="Replace the default three-worker fixture; repeat NAME=INSTRUCTION")
    parser.add_argument("--after", action="append", default=[], help="CHILD=PARENT; repeat")
    parser.add_argument("--synthesis", default="Reply with the single word complete.")
    parser.add_argument("--chat-template")
    parser.add_argument("--parallel", type=positive, default=2)
    parser.add_argument("--ctx-size", type=positive, default=4096)
    parser.add_argument("--total-kv", type=positive)
    parser.add_argument("--batch-size", type=positive, default=512)
    parser.add_argument("--ubatch-sizes", default="128",
                        help="Comma-separated real microbatch sizes; e.g. 128,1")
    parser.add_argument("--phase-modes", default="release",
                        help="release, retain, or release,retain")
    parser.add_argument("--repeat", type=positive, default=1)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--gpu-layers", type=nonnegative, default=0)
    parser.add_argument("--devices")
    parser.add_argument("--split-mode", choices=("none", "layer", "row"), default="none")
    parser.add_argument("--worker-tokens", type=positive, default=512,
                        help="Maximum native body tokens, including any model reasoning; exhaustion fails")
    parser.add_argument("--final-tokens", type=positive, default=512,
                        help="Maximum synthesis tokens, including any model reasoning; exhaustion fails")
    parser.add_argument("--seed", type=nonnegative, default=0)
    parser.add_argument("--min-peer-growth", type=nonnegative, default=2,
                        help="Required successive peer body increments while both readers are active")
    parser.add_argument("--require-multi-device", action="store_true",
                        help="Fail unless real model buffer logs show at least two non-CPU devices")
    parser.add_argument("--require-cross-shape-equivalence", action="store_true",
                        help="Also require identical greedy trajectories across at least two requested microbatch sizes")
    parser.add_argument("--timeout", type=positive, default=600, help="Seconds per actual model run")
    return parser.parse_args()


def main():
    args = arguments()
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output = args.output.resolve()
    # Preserve previous evidence: never overwrite an earlier qualification.
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / "receipt.json"
    receipt = {
        "schema": "tq-rerot-real-whiteboard/2", "status": "RUNNING",
        "source_commit_provided": args.source_commit,
        "platform": platform.platform(), "python": sys.version,
        "scope": "real installed C whiteboard contract plus requested phase/repetition controls within each execution shape",
        "not_qualified_by_this_harness": [
            "reader attention numerical equivalence",
            "whole-model TP5", "LateBind", "MTP", "MMVQ kernel dispatch", "performance benefit",
        ],
        "runs": [],
    }
    controls = TrajectoryControls(args.require_cross_shape_equivalence)
    shape_count = phase_count = expected_runs = 0

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=True) + "\n",
                                encoding="utf-8")

    save()
    try:
        receipt["binary"] = fingerprint(binary)
        receipt["model_shards"] = [fingerprint(path) for path in model_files(model)]
        receipt["additional_artifacts"] = [fingerprint(path) for path in args.artifact]
        names = ("GGML_TQ_NATIVE", "GGML_CUDA_FUSE_CHAIN", "GGML_CUDA_Q8CACHE",
                 "GGML_CUDA_DISABLE_GRAPHS", "ROCR_VISIBLE_DEVICES",
                 "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES",
                 "GGML_VK_DISABLE_F16")
        receipt["execution_environment"] = {name: os.environ[name] for name in names if name in os.environ}
        # Deliberately restricted to precision switches confirmed in the
        # tested native provider. Do not collect the process environment.
        precision_names = ("GGML_VK_DISABLE_F16",)
        receipt["precision_environment_allowlist"] = list(precision_names)
        receipt["precision_environment"] = {name: os.environ[name]
                                            for name in precision_names if name in os.environ}
        # Agreement is legitimate in a shared whiteboard. Conflicting copying
        # instructions add an unrelated reasoning task and are not needed to
        # prove incremental peer visibility. Native reasoning tokens count
        # against the budget even when the requested final answer is short.
        stages = args.stage or [
            "writer_a=Reply with only this sentence: Shared work brings us to the same answer.",
            "writer_b=Reply with only this sentence: Shared work brings us to the same answer.",
            "ready=Reply with the single word ready.",
        ]
        after = args.after if args.stage else ["ready=writer_a", "ready=writer_b", *args.after]
        receipt["fixture"] = {
            "instructions": "caller supplied" if args.stage else "cooperative shared whiteboard",
            "worker_token_budget": args.worker_tokens,
            "synthesis_token_budget": args.final_tokens,
            "end_condition": "native model EOG; token-budget exhaustion is an error",
            "reasoning_tokens_count_toward_budget": True,
            "different_worker_wording_required": False,
        }
        stage_names = [item.split("=", 1)[0] for item in stages]
        need(len(set(stage_names)) == len(stages), "duplicate fixture stage")
        need(all("=" in item and item.split("=", 1)[1] for item in stages), "invalid fixture stage")
        mapped = {name: index for index, name in enumerate(stage_names, 1)}
        edges = []
        for item in after:
            child, parent = item.split("=", 1)
            need(parent in mapped and child in mapped, "fixture edge names an unknown stage")
            edges.append((mapped[parent], mapped[child]))
        need(args.min_peer_growth == 0 or min(len(stages), args.parallel) >= 2,
             "peer growth requires at least two resident workers")
        ubatches = [positive(value) for value in args.ubatch_sizes.split(",")]
        need(len(set(ubatches)) == len(ubatches) and max(ubatches) <= args.batch_size,
             "microbatch sizes are duplicated or exceed batch capacity")
        need(not args.require_cross_shape_equivalence or len(ubatches) >= 2,
             "cross-shape equivalence requires at least two actual microbatch sizes")
        phases = args.phase_modes.split(",")
        need(len(set(phases)) == len(phases) and set(phases).issubset({"release", "retain"}),
             "invalid or duplicate phase mode")
        shape_count, phase_count = len(ubatches), len(phases)
        expected_runs = shape_count * phase_count * args.repeat
        common = [str(binary), "--model", str(model), "--prompt", args.prompt,
                  "--synthesis", args.synthesis, "--parallel", str(args.parallel),
                  "--ctx-size", str(args.ctx_size), "--batch-size", str(args.batch_size),
                  "--threads", str(args.threads), "--gpu-layers", str(args.gpu_layers),
                  "--split-mode", args.split_mode, "--worker-tokens", str(args.worker_tokens),
                  "--final-tokens", str(args.final_tokens), "--seed", str(args.seed),
                  "--temperature", "0", "--json"]
        for stage in stages:
            common.extend(("--stage", stage))
        for edge in after:
            common.extend(("--after", edge))
        for flag, value in (("--total-kv", args.total_kv), ("--devices", args.devices),
                            ("--chat-template", args.chat_template)):
            if value is not None:
                common.extend((flag, str(value)))
        profile = {"common_argv": common, "binary": receipt["binary"],
                   "model_shards": receipt["model_shards"],
                   "additional_artifacts": receipt["additional_artifacts"],
                   "execution_environment": receipt["execution_environment"]}
        profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode("utf-8")).hexdigest()
        receipt["execution_profile"] = {"sha256": profile_hash, **profile}
        case_number = 0
        for phase in phases:
            for ubatch in ubatches:
                for repeat in range(args.repeat):
                    case_number += 1
                    name = f"{case_number:02d}-ubatch-{ubatch}-{phase}-{repeat + 1}"
                    command = [*common, "--ubatch-size", str(ubatch),
                               "--phase-memory" if phase == "release" else "--no-phase-memory"]
                    print(f"RUN {name}", flush=True)
                    run = capture(command, output, name, args.timeout)
                    run["name"] = name
                    run["ubatch_size"] = ubatch
                    run["phase_mode"] = phase
                    run["repetition"] = repeat + 1
                    run["precision_environment"] = dict(receipt["precision_environment"])
                    receipt["runs"].append(run)
                    save()
                    run["contract_status"] = "FAIL"
                    need(not run["timed_out"], f"{name}: real inference timed out")
                    events = read_events(run["stdout"]["path"])
                    if run["returncode"] != 0:
                        error = next((item.get("message") for item in reversed(events)
                                      if item["event"] == "error"), "no structured error")
                        raise ContractError(f"{name}: inference exited {run['returncode']}: {error}")
                    run["contract"] = validate_trace(events, len(stages), args.parallel, edges,
                                                     args.min_peer_growth)
                    run["contract_status"] = "PASS"
                    stderr = Path(run["stderr"]["path"]).read_text(encoding="utf-8", errors="replace")
                    run["placement"] = placement(stderr)
                    # These are resolved native context values printed by A,
                    # not an assumption based on the requested CLI switches.
                    native_shape = {}
                    for key in ("n_batch", "n_ubatch"):
                        values = {int(value) for value in re.findall(r"\b" + key + r"\s*=\s*(\d+)\b", stderr)}
                        need(len(values) == 1, f"{name}: no single resolved native {key} was observed")
                        native_shape[key] = values.pop()
                    need(native_shape["n_ubatch"] == ubatch,
                         f"{name}: native microbatch capacity differs from the requested comparison shape")
                    run["native_context_shape"] = native_shape
                    if args.gpu_layers > 0:
                        need(run["placement"]["offloaded_layers"] > 0,
                             f"{name}: requested GPU layers but no positive actual offload was observed")
                    if args.require_multi_device:
                        need(len(run["placement"]["model_buffer_devices"]) >= 2,
                             f"{name}: actual multi-device model placement was not observed")
                    trajectory = run["contract"].pop("token_trace")
                    encoded = json.dumps(trajectory, separators=(",", ":")).encode("ascii")
                    run["token_trajectory_sha256"] = hashlib.sha256(encoded).hexdigest()
                    group_id = f"{profile_hash[:16]}-ubatch-{ubatch}"
                    controls.check(group_id, ubatch, phase, trajectory, run, run["placement"], native_shape)
                    run["status"] = "PASS"
                    save()
                    print(f"PASS {name}: {run['contract']['frontiers']} frontiers; "
                          f"{run['contract']['peer_body_reads_while_running']} live peer reads", flush=True)
        need(bool(receipt["runs"]), "no actual inference case ran")
        receipt["status"] = "PASS"
    except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
        receipt["status"] = "FAIL"
        receipt["failure"] = str(error)
        if receipt["runs"] and receipt["runs"][-1].get("status") != "PASS":
            receipt["runs"][-1]["status"] = "FAIL"
        print(f"FAIL: {error}", file=sys.stderr)
    finally:
        receipt["control_groups"] = controls.group_receipts
        receipt["qualifications"] = controls.report(shape_count, phase_count, args.repeat)
        contract_passes = sum(run.get("contract_status") == "PASS" for run in receipt["runs"])
        contract_failed = any(run.get("contract_status") == "FAIL" or run["returncode"] != 0 or run["timed_out"]
                              for run in receipt["runs"])
        receipt["qualifications"]["incremental_whiteboard_contract"] = {
            "status": "FAIL" if contract_failed else "PASS" if expected_runs > 0 and contract_passes == expected_runs
                      else "NOT_QUALIFIED",
            "expected_runs": expected_runs, "passed_runs": contract_passes,
        }
        save()
    print("QUALIFICATIONS " + json.dumps({key: value["status"]
                                         for key, value in receipt["qualifications"].items()}, sort_keys=True),
          flush=True)
    print(str(receipt_path), flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
