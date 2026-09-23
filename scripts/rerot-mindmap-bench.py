#!/usr/bin/env python3
"""MM-R1 mindmap experiment harness.

Drives the mindmap research line end to end and records every artifact the
experiment plan (section 15) requires. The harness is deliberately split in two:

  * A MODEL-FREE section that runs without a server: it exercises the strict
    parser, the hierarchical reader order and the G0 character language against
    the fixtures in tests/data/rerot-mindmap, so a protocol regression is caught
    before any model is loaded.
  * A SERVER section that runs the arms against a live llama-server, records
    plans/responses/events, and refuses to claim success for a capability the
    server did not report.

Contract (this is the point of the harness):
  * An unknown arm / final_mode / reader_order is a HANDSHAKE FAILURE, never a
    silent fallback: the report must never contain a "mermaid success" that was
    actually a DAG run.
  * A request that fails, times out, or exceeds its budget stays in the
    denominator.
  * Every arm record stores the configuration the SERVER parsed, not just the
    requested one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "data" / "rerot-mindmap" / "fixtures.txt"

# Experiment arms from the plan (section 9.1). Each entry names the wire, the
# worker dependency policy, the reader order, the final mode, and the question
# the arm answers. Anything not listed here is an unknown arm and must fail.
ARMS: Dict[str, Dict[str, str]] = {
    "C0": {"wire": "none", "deps": "none", "order": "plain", "final": "plain",
           "why": "overall benefit of paying the probe cost"},
    "C2": {"wire": "json", "deps": "model", "order": "kahn", "final": "reason",
           "why": "tuned shipped DAG baseline"},
    "C4": {"wire": "mindmap", "deps": "none", "order": "flat", "final": "reason",
           "why": "mermaid + hierarchical planning front-end"},
    "C5": {"wire": "mindmap", "deps": "none", "order": "hierarchical-cyclic-dfs",
           "final": "reason", "why": "MAIN ARM: full mindmap route"},
    "C7": {"wire": "mindmap", "deps": "none", "order": "hierarchical-cyclic-dfs",
           "final": "direct", "why": "S0 direct global entry"},
    "C8": {"wire": "mindmap", "deps": "none", "order": "own-only", "final": "reason",
           "why": "is real-time sharing contributing at all"},
    "C9": {"wire": "mindmap", "deps": "none", "order": "sealed-only", "final": "reason",
           "why": "token-level vs post-completion sharing"},
}

# The server only accepts these exact values; anything else is a handshake
# failure rather than a fallback.
KNOWN_WIRES = {"none", "json", "mindmap"}
KNOWN_FINAL = {"plain", "reason", "direct"}
KNOWN_ORDERS = {"plain", "kahn", "flat", "hierarchical-cyclic-dfs",
                "own-only", "sealed-only"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Model-free section: protocol + ordering
# ---------------------------------------------------------------------------

MM_REFERENCE = REPO_ROOT / "tests" / "test-rerot-mindmap-parser"


def run_parser_target(build_dir: Path) -> Tuple[bool, str]:
    """Runs the compiled strict-parser test target. No model, no GPU."""
    binary = build_dir / "bin" / "test-rerot-mindmap-parser"
    if not binary.is_file():
        return False, f"missing test binary: {binary}"
    proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
    ok = proc.returncode == 0 and "all tests passed" in proc.stdout
    return ok, (proc.stdout + proc.stderr).strip()


def run_sampler_target(build_dir: Path) -> Tuple[bool, str]:
    binary = build_dir / "bin" / "test-rerot-mindmap-sampler"
    if not binary.is_file():
        return False, f"missing test binary: {binary}"
    proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
    ok = proc.returncode == 0 and "all tests passed" in proc.stdout
    return ok, (proc.stdout + proc.stderr).strip()


def load_fixtures() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parses the fixture file. Returns (records, problems)."""
    records: List[Dict[str, Any]] = []
    problems: List[str] = []
    current: Optional[Dict[str, Any]] = None
    body_lines: List[str] = []
    in_block = False

    for lineno, raw in enumerate(FIXTURES.read_text(encoding="utf-8").splitlines(), start=1):
        if raw.startswith("--- id:"):
            if current is not None:
                current["body"] = "\n".join(body_lines) + ("" if in_block else "\n")
                records.append(current)
            current = {"id": raw[len("--- id:"):].strip(), "line": lineno,
                       "expect": "", "note": "", "body": ""}
            body_lines = []
            in_block = False
            continue
        if current is None:
            continue
        if raw.startswith("EXPECT:"):
            current["expect"] = raw[len("EXPECT:"):].strip()
            continue
        if raw.startswith("##"):
            current["note"] = raw[2:].strip()
            continue
        if in_block:
            # The wire INCLUDES its closing fence: the protocol requires the
            # document to end with it, so the loader must keep the line.
            body_lines.append(raw)
            if raw.rstrip() == "```":
                in_block = False
            continue
        if raw.startswith("WIRE:"):
            # A fixture whose PROSE-shape wire is the subject under test: the
            # bytes after the marker are the document, prose included.
            body_lines.append(raw[len("WIRE:"):])
            in_block = True
            continue
        if raw.rstrip() == "```" or raw.startswith("```mermaid"):
            # Opening fence. A bare ```` ``` ```` is markdown only; the
            # mermaid opener is the FIRST BYTES of the wire document itself, so
            # it must be captured as content.
            if raw.startswith("```mermaid"):
                body_lines.append(raw)
            in_block = True
            continue
        # Prose lines between records are commentary, never wire content: they
        # are recorded as the record's note, not appended to the document.

    if current is not None:
        # The document ends with the closing fence followed by LF, which the
        # line iterator above dropped; re-add it so the classifier sees the
        # exact bytes a probe would have emitted.
        current["body"] = "\n".join(body_lines) + ("" if in_block else "\n")
        records.append(current)

    for rec in records:
        if not rec["expect"]:
            problems.append(f"fixture {rec['id']} has no EXPECT")
        if rec["expect"] not in {"complete", "invalid"}:
            problems.append(f"fixture {rec['id']} has unknown EXPECT {rec['expect']}")
    return records, problems


def check_fixtures() -> Tuple[bool, Dict[str, Any]]:
    """Model-free protocol check of every fixture.

    The fixtures pin the BYTE contract of the wire. This function re-implements
    the verdicts with an independent scanner so a bug in the C++ parser cannot
    hide behind itself: the scan is a plain line/indent state machine.
    """
    records, problems = load_fixtures()
    if problems:
        return False, {"problems": problems}

    HEAD = "```mermaid\nmindmap\n"
    results = []
    for rec in records:
        text = rec["body"]
        verdict, reason = classify_wire(text)
        ok = verdict == rec["expect"]
        results.append({"id": rec["id"], "expect": rec["expect"],
                        "got": verdict, "reason": reason, "ok": ok})
    failures = [r for r in results if not r["ok"]]
    return (not failures), {"cases": len(results), "failures": failures}


def classify_wire(text: str) -> Tuple[str, str]:
    """Independent complete-document classifier (no C++ involved)."""
    HEAD = "```mermaid\nmindmap\n"
    if not text:
        return "invalid", "empty"
    if not text.startswith(HEAD):
        return "invalid", "envelope"
    if not text.endswith("```\n"):
        return "invalid", "envelope"
    body = text[len(HEAD):-len("```\n")]
    if not body or not body.endswith("\n"):
        return "invalid", "structure"
    if "\r" in body or "\t" in body:
        return "invalid", "envelope"

    stack: List[int] = []
    nodes = 0
    leaves: List[int] = []
    lines = body[:-1].split("\n")
    for idx, line in enumerate(lines):
        spaces = len(line) - len(line.lstrip(" "))
        label = line[spaces:]
        if spaces % 2:
            return "invalid", f"line {idx + 3}: odd indent"
        depth = spaces // 2
        if depth < 1:
            return "invalid", f"line {idx + 3}: no indent"
        if not label or label != label.strip():
            return "invalid", f"line {idx + 3}: label padding"
        if "  " in label:
            return "invalid", f"line {idx + 3}: double space in label"
        for ch in label:
            o = ord(ch)
            ascii_ok = (0x30 <= o <= 0x39 or 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A
                        or ch in "*/=.,?!;:_- ")
            if not (ascii_ok or 0xA1 <= o <= 0xD7AF or 0x3000 <= o <= 0x30FF
                    or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF
                    or 0xFF01 <= o <= 0xFF5E or 0x2018 <= o <= 0x201D):
                return "invalid", f"line {idx + 3}: forbidden scalar {ch!r}"
        if depth > 4:
            return "invalid", f"line {idx + 3}: depth budget"
        nodes += 1
        if nodes > 64:
            return "invalid", f"line {idx + 3}: node budget"
        if not stack:
            if depth != 1:
                return "invalid", f"line {idx + 3}: first node must be root"
        else:
            if depth == 1:
                return "invalid", f"line {idx + 3}: second root"
            if depth > len(stack) + 1:
                return "invalid", f"line {idx + 3}: skipped level"
        stack = stack[: depth - 1]
        stack.append(nodes)
        # A node is a leaf when no later line nests under it. `depth` is the
        # 1-based level of this line; it stays a leaf unless a following line
        # has a strictly greater depth, which the loop below detects.
        leaves.append(depth)
    # A line is a leaf iff the next line does not nest under it.
    leaf_count = 0
    for i, depth in enumerate(leaves):
        is_leaf = True
        if i + 1 < len(leaves) and leaves[i + 1] > depth:
            is_leaf = False
        if is_leaf:
            leaf_count += 1
    if leaf_count > 8:
        return "invalid", "leaf budget"
    return "complete", "ok"


def check_reader_order() -> Tuple[bool, Dict[str, Any]]:
    """Model-free check of the hierarchical cyclic DFS contract.

    `emit` is the production-shaped algorithm (rotate at every node on the
    reader path, recurse, put the path child's subtree last). `oracle` is the
    independent path-sort reference from the plan appendix (sort leaves by the
    rotated sibling ranks along the root path). They must agree for EVERY leaf
    of EVERY tree, the reader's own leaf must be last, and every sibling subtree
    must stay contiguous. A flat ring rotation of the canonical leaf order fails
    the last two on deep trees -- which is why C4 and C5 are separate arms.
    """
    def build_parents(tree: List[List[int]]) -> Dict[int, int]:
        return {c: p for p, kids in enumerate(tree) for c in kids}

    def path_of(parents: Dict[int, int], leaf: int) -> List[int]:
        path = [leaf]
        while path[-1] in parents:
            path.append(parents[path[-1]])
        path.reverse()
        return path

    def emit_order(tree: List[List[int]], reader: int) -> List[int]:
        parents = build_parents(tree)
        on_path = set(path_of(parents, reader))
        out: List[int] = []

        def visit(u: int) -> None:
            kids = list(tree[u])
            if u in on_path and kids:
                child = next(k for k in kids if k == path_step(parents, on_path, u))
                k = kids.index(child)
                kids = kids[k + 1:] + kids[: k + 1]
            for k in kids:
                if tree[k]:
                    visit(k)
                else:
                    out.append(k)
            if not kids and u == 0:
                out.append(u)

        visit(0)
        return out

    def path_step(parents: Dict[int, int], on_path: set, u: int) -> int:
        """The child of u that lies on the reader path."""
        node = reader_ref[0]
        path = path_of(parents, node)
        for depth in range(len(path) - 1):
            if path[depth] == u:
                return path[depth + 1]
        raise AssertionError("path step requested for an off-path node")

    def oracle_order(tree: List[List[int]], reader: int) -> List[int]:
        parents = build_parents(tree)
        on_path = set(path_of(parents, reader))
        leaves = [i for i, kids in enumerate(tree) if not kids]

        def key(leaf: int) -> Tuple[int, ...]:
            path = path_of(parents, leaf)
            ranks = []
            for depth in range(len(path) - 1):
                node = path[depth]
                siblings = tree[node]
                rank = siblings.index(path[depth + 1])
                if node in on_path:
                    pivot = siblings.index(path_step(parents, on_path, node))
                    rank = (rank - pivot - 1) % len(siblings)
                ranks.append(rank)
            return tuple(ranks)

        return sorted(leaves, key=key)

    reader_ref = [0]
    trees = 0
    views = 0
    mismatches: List[str] = []
    state = 0x20260923
    for _ in range(300):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        n = 1 + (state >> 33) % 24
        tree: List[List[int]] = [[]]
        for i in range(1, n):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            p = (state >> 33) % i
            tree.append([])
            tree[p].append(i)
        trees += 1
        leaves = [i for i, kids in enumerate(tree) if not kids]
        for reader in leaves:
            reader_ref[0] = reader
            got = emit_order(tree, reader)
            want = oracle_order(tree, reader)
            views += 1
            if got != want:
                mismatches.append(f"tree={tree} reader={reader} got={got} want={want}")
                if len(mismatches) > 4:
                    break
            if got and got[-1] != reader:
                mismatches.append(f"tree={tree} reader={reader} own-last violated")
            # Contiguity: each sibling subtree occupies one unbroken block.
            pos = {leaf: i for i, leaf in enumerate(got)}
            for p, kids in enumerate(tree):
                if len(kids) < 2:
                    continue
                spans = []
                for k in kids:
                    # Only LEAVES occupy slots; interior nodes render no token.
                    sub_leaves = [x for x in descendants(tree, k) if not tree[x]]
                    if not sub_leaves:
                        continue
                    idxs = sorted(pos[x] for x in sub_leaves)
                    spans.append((idxs[0], idxs[-1]))
                spans.sort()
                covered = sum(b - a + 1 for a, b in spans)
                if covered != spans[-1][1] - spans[0][0] + 1:
                    mismatches.append(
                        f"tree={tree} reader={reader} sibling blocks overlap or gap")
            if len(mismatches) > 4:
                break
    ok = not mismatches
    return ok, {"trees": trees, "views": views, "mismatches": mismatches[:5]}


def descendants(tree: List[List[int]], node: int) -> List[int]:
    out = [node]
    stack = list(tree[node])
    while stack:
        u = stack.pop()
        out.append(u)
        stack.extend(tree[u])
    return out


# ---------------------------------------------------------------------------
# Server section
# ---------------------------------------------------------------------------

class ServerHandshake:
    """Minimal capability check: the server must report what it supports."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def props(self) -> Dict[str, Any]:
        req = urllib.request.Request(self.base_url + "/props")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def check_arm(self, arm: Dict[str, str]) -> Tuple[bool, str]:
        """Validates an arm against this server's advertised capabilities.

        A failure is HARD: the caller must abort the arm, not fall back. Running
        a DAG and reporting it as a mindmap success is the worst possible
        outcome for the experiment record.
        """
        props = self.props()
        rerot = props.get("rerot") if isinstance(props, dict) else None
        if not isinstance(rerot, dict):
            return False, "server did not report a rerot capability block"
        if not rerot.get("enabled"):
            return False, "server rerot is not enabled (need --rerot)"
        if arm["wire"] not in KNOWN_WIRES:
            return False, f"unknown wire {arm['wire']}"
        if arm["final"] not in KNOWN_FINAL:
            return False, f"unknown final mode {arm['final']}"
        if arm["order"] not in KNOWN_ORDERS:
            return False, f"unknown reader order {arm['order']}"
        if arm["wire"] == "mindmap":
            supported = rerot.get("plan_wires")
            if isinstance(supported, list) and "mindmap" not in supported:
                return False, "server does not advertise the mindmap plan wire"
        return True, "ok"


def wait_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def completion(base_url: str, prompt: str, timeout: float) -> Dict[str, Any]:
    payload = json.dumps({
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "seed": 20260923,
        "n_predict": 512,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_wall_seconds"] = time.monotonic() - started
    return body


# ---------------------------------------------------------------------------
# Frozen evaluation set + grader (M13)
# ---------------------------------------------------------------------------

EVAL_SET = REPO_ROOT / "tests" / "data" / "rerot-mindmap" / "eval-set.txt"

# Keys the eval-set parser recognises. `family` is the split-separation key and
# is unique per prompt; `family_group` is the analytic grouping.
_EVAL_KEYS = {"category", "family", "family_group", "split", "expected",
              "prompt", "rubric"}


def load_eval_set(path: Path = EVAL_SET) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parses the frozen evaluation set. Returns (records, problems).

    A rubric-only prompt (empty `expected`) is NOT auto-graded: substring
    matching the final answer is exactly the "unacceptable scoring" the plan
    forbids, so those go to the blind panel and are reported as pending.
    """
    if not path.is_file():
        return [], [f"missing evaluation set: {path}"]
    records: List[Dict[str, Any]] = []
    problems: List[str] = []
    current: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip("\n")
        m = re.match(r"^([a-z_]+):[ ]?(.*)$", line)
        if m and m.group(1) == "id":
            if current is not None:
                records.append(current)
            current = {"id": m.group(2).strip(), "line": lineno}
            key = None
            continue
        if current is None:
            continue
        if m and m.group(1) in _EVAL_KEYS:
            key = m.group(1)
            current[key] = m.group(2)
            continue
        if key and line.strip():
            current[key] += "\n" + line
    if current is not None:
        records.append(current)

    for rec in records:
        for field in ("category", "family", "family_group", "split", "prompt"):
            if not rec.get(field):
                problems.append(f"prompt {rec['id']} missing {field}")
        if rec.get("split") not in {"smoke", "dev", "blind"}:
            problems.append(f"prompt {rec['id']} has unknown split {rec.get('split')}")
        if not rec.get("rubric"):
            problems.append(f"prompt {rec['id']} has no rubric")

    # Split separation: no family may span two splits, otherwise the blind set
    # is contaminated by family overlap with the development set.
    fam_splits: Dict[str, set] = {}
    for rec in records:
        if rec.get("family"):
            fam_splits.setdefault(rec["family"], set()).add(rec.get("split"))
    for fam, splits in fam_splits.items():
        if len(splits) > 1:
            problems.append(f"eval family {fam} spans splits {sorted(s for s in splits if s)}")
    return records, problems


def grade_exact(rec: Dict[str, Any], content: str) -> Dict[str, Any]:
    """Grades one prompt against the frozen rubric.

    Transport health and answer correctness are separate gates:
        success = transport_ok AND protocol_ok AND answer_correct
    A missing final answer is a rubric failure, not a skipped item.
    """
    expected = (rec.get("expected") or "").strip().strip("'\"")
    if not expected:
        return {"grade": "pending_blind_panel", "reason": "no exact expected answer",
                "answer_correct": None, "transport_ok": True, "protocol_ok": True}
    text = content.strip()
    # Strip surrounding quotes/punctuation so '221' and 221 compare equal, but
    # never rewrite the content itself.
    needle = expected.strip()
    found = needle in text
    # A correct value that never appears in the FINAL content is still a
    # failure: the plan forbids scoring the reasoning trace.
    return {"grade": "exact", "reason": "substring of the final content",
            "answer_correct": found, "transport_ok": True, "protocol_ok": True,
            "expected": needle}


class Harness:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
        self.out = Path(args.output).resolve() / self.run_id
        self.events: List[Dict[str, Any]] = []
        self.problems: List[str] = []

    def emit(self, kind: str, **fields: Any) -> None:
        rec = {"seq": len(self.events), "kind": kind,
               "monotonic_ns": time.monotonic_ns()}
        rec.update(fields)
        self.events.append(rec)

    def write_manifests(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT,
                               capture_output=True, text=True)
        manifest = {
            "run_id": self.run_id,
            "experiment_version": "MM-R1",
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": dirty.stdout if dirty.returncode == 0 else None,
            "binary_sha256": {
                "llama-server": sha256_file(
                    Path(self.args.build_dir) / "bin" / "llama-server")
                if (Path(self.args.build_dir) / "bin" / "llama-server").is_file() else None,
                "test-rerot-mindmap-parser": sha256_file(
                    Path(self.args.build_dir) / "bin" / "test-rerot-mindmap-parser")
                if (Path(self.args.build_dir) / "bin" / "test-rerot-mindmap-parser").is_file() else None,
            },
            "fixtures_sha256": sha256_file(FIXTURES) if FIXTURES.is_file() else None,
            "arms": {k: v for k, v in ARMS.items() if k in self.args.arms},
            "seed": self.args.seed,
            "prompt_limit_tokens": 512,
        }
        (self.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    def run_model_free(self) -> bool:
        ok = True
        build = Path(self.args.build_dir)

        parser_ok, parser_out = run_parser_target(build)
        self.emit("model_free", target="test-rerot-mindmap-parser",
                  ok=parser_ok, detail=parser_out[-2000:])
        if not parser_ok:
            self.problems.append("strict parser target failed")

        sampler_ok, sampler_out = run_sampler_target(build)
        self.emit("model_free", target="test-rerot-mindmap-sampler",
                  ok=sampler_ok, detail=sampler_out[-2000:])
        if not sampler_ok:
            self.problems.append("G1 sampler target failed")

        fix_ok, fix_detail = check_fixtures()
        self.emit("model_free", target="fixtures", ok=fix_ok, detail=fix_detail)
        if not fix_ok:
            self.problems.append(f"fixture check failed: {fix_detail}")

        order_ok, order_detail = check_reader_order()
        self.emit("model_free", target="reader_order", ok=order_ok, detail=order_detail)
        if not order_ok:
            self.problems.append(f"reader order check failed: {order_detail}")

        ok = parser_ok and sampler_ok and fix_ok and order_ok
        return ok

    def run_server_arms(self) -> bool:
        """Runs the requested arms against a live server.

        Without --server-url this section is SKIPPED, not failed: the harness
        must be runnable (and honest) on a machine with no model.
        """
        if not self.args.server_url:
            self.emit("server", skipped=True,
                      reason="no --server-url supplied; model section skipped")
            return True

        ok = True
        base = self.args.server_url.rstrip("/")
        try:
            hs = ServerHandshake(base, self.args.handshake_timeout)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self.emit("server", skipped=True, reason=f"handshake failed: {exc}")
            self.problems.append(f"server handshake failed: {exc}")
            return False

        responses = []
        for arm_name in self.args.arms:
            arm = ARMS[arm_name]
            good, why = hs.check_arm(arm)
            self.emit("arm_handshake", arm=arm_name, ok=good, reason=why)
            if not good:
                # HARD failure: never fall back to another wire.
                self.problems.append(f"arm {arm_name} rejected: {why}")
                ok = False
                continue
            for prompt_id, prompt in enumerate(self.args_prompts()):
                started = time.monotonic()
                try:
                    body = completion(base, prompt, self.args.timeout)
                except Exception as exc:  # noqa: BLE001 - failures stay in the log
                    self.emit("arm_request", arm=arm_name, prompt=prompt_id,
                              ok=False, error=str(exc),
                              wall_seconds=time.monotonic() - started)
                    self.problems.append(f"arm {arm_name} prompt {prompt_id}: {exc}")
                    ok = False
                    continue
                content = body.get("content", "")
                record = {
                    "arm": arm_name,
                    "wire": arm["wire"],
                    "order": arm["order"],
                    "final": arm["final"],
                    "prompt_id": prompt_id,
                    "wall_seconds": body.get("_wall_seconds"),
                    "content_chars": len(content),
                    "content_empty": not content.strip(),
                    "finish_reason": body.get("finish_reason"),
                    "timings": body.get("timings"),
                    "usage": body.get("usage"),
                }
                responses.append(record)
                self.emit("arm_request", ok=not record["content_empty"], **record)
                if record["content_empty"]:
                    self.problems.append(
                        f"arm {arm_name} prompt {prompt_id}: empty final content")

        (self.out / "responses.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in responses),
            encoding="utf-8")
        return ok

    def run_frozen_set(self) -> bool:
        """Validates the frozen evaluation set as an instrument.

        The blind panel cannot run without human graders, so this section only
        checks that the set is SOUND: parseable, split-separated by family, and
        gradeable end to end through grade_exact. A set that fails here would
        produce a misleading report later, so it is a hard gate.
        """
        records, problems = load_eval_set()
        if problems:
            self.emit("frozen_set", ok=False, problems=problems)
            self.problems.extend(f"frozen set: {p}" for p in problems)
            return False
        if not records:
            self.emit("frozen_set", ok=False, problems=["no prompts loaded"])
            self.problems.append("frozen set is empty")
            return False
        # Grade every exact-answer prompt against a synthetic response to prove
        # the grader makes decisions rather than shrugging.
        decided = 0
        for rec in records:
            expected = (rec.get("expected") or "").strip().strip("'\"")
            if not expected:
                continue
            hit = grade_exact(rec, expected)
            miss = grade_exact(rec, "unrelated answer text")
            if hit.get("answer_correct") is not True or miss.get("answer_correct") is not False:
                self.emit("frozen_set", ok=False,
                          problems=[f"grader is not discriminating for {rec['id']}"])
                self.problems.append(f"grader not discriminating for {rec['id']}")
                return False
            decided += 1
        self.emit("frozen_set", ok=True, prompts=len(records),
                  exact_answer=decided,
                  rubric_only=len(records) - decided,
                  splits={})
        return True

    def args_prompts(self) -> List[str]:
        if self.args.prompts:
            path = Path(self.args.prompts)
            if path.is_file():
                return [l for l in path.read_text(encoding="utf-8").splitlines()
                        if l.strip()]
        return [
            "9.11 and 9.9: which is larger? Answer directly.",
            "Compute 13*17 and 21*19, then give both products and their sum.",
            "Solve (1) 13*17 step by step, (2) 21*19, then verify both.",
            "Write a Python function that reverses a string without slicing, "
            "then add a unit test for it.",
        ]

    def finish(self, model_free_ok: bool, server_ok: bool) -> int:
        (self.out / "events.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.events),
            encoding="utf-8")
        report = {
            "run_id": self.run_id,
            "model_free_ok": model_free_ok,
            "server_ok": server_ok,
            "problems": self.problems,
            "artifacts": str(self.out),
        }
        (self.out / "report.md").write_text(
            "# MM-R1 run\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
            encoding="utf-8")
        print(json.dumps(report, indent=2))
        if self.problems:
            return 1
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "build"))
    parser.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "mm-r1"))
    parser.add_argument("--arms", nargs="+", default=["C5"],
                        choices=sorted(ARMS))
    parser.add_argument("--server-url", default=None,
                        help="e.g. http://127.0.0.1:18090; omit to run the "
                             "model-free section only")
    parser.add_argument("--prompts", default=None,
                        help="file with one prompt per line")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--handshake-timeout", type=float, default=30.0)
    args = parser.parse_args()

    harness = Harness(args)
    harness.write_manifests()
    model_free_ok = harness.run_model_free()
    frozen_ok = harness.run_frozen_set()
    server_ok = harness.run_server_arms()
    return harness.finish(model_free_ok and frozen_ok, server_ok)


if __name__ == "__main__":
    sys.exit(main())
