# Router scratch candidate — Devops execution-level verification plan

Status: **PARTIAL**. Every number below is an expectation, not a result.
Only Devops compiles/tests. Nothing in this file has been executed.
No static check is claimed for the current sources either: they were written
and re-read, not compiled (an earlier revision of this file claimed
`-fsyntax-only` runs on files that have since been rewritten).

The CPU leg of §2 is superseded by `ROUTER_SCRATCH_CPU_FIXTURE.md`: that
fixture is a real 512/top-10 Qwen4Exp GGUF with shrunk hidden widths, so the
CPU leg no longer needs a production checkpoint. §3 (GPU) is unchanged and
still needs the target `swift-q2_0` GGUF on 13022.

## 1. Candidate under test

Tree: `/home/kunweiz/work/merge-router-20261031`, branch
`candidates/router-scratch-20261031`, baseline commit `4496493dd`
("Preserve router scratch candidate", itself on `f1c5fafc0…` archive baseline).
Working-tree deltas on top of `4496493dd`:

| File | Change |
|---|---|
| `ggml/src/ggml-alloc.c` | `GGML_TENSOR_FLAG_NO_REUSE` plan-lifecycle (already in 4496493dd) + plan-level disjointness audit `ggml_gallocr_check_no_reuse_plan` (this session; audits retained allocations only and recognises the in-place continuation, see `ROUTER_SCRATCH_CPU_FIXTURE.md` §5) |
| `src/models/qwen4exp.cpp` | retention row window bounded to `2 <= cur->ne[1] <= 4`, contiguity/shape fail-closed checks, `GGML_TP5_ROUTER_SCRATCH_DEBUG` per-layer retention log (this session) |
| `ggml/src/ggml-cuda/topk-moe.cu` | multi-row fire log in `ggml_cuda_op_topk_moe` (this session) |
| `examples/tp5-router-verify/` | real-consumer N1/N2/N3/N4 verifier, the minimal 512/top-10 fixture generator, the CPU driver script and the CMake registration (this session) |

Contract preserved: `ggml/src/ggml-cuda/ggml-cuda.cu:4467`
(`ggml_nrows(...) == 1 && is_topk_moe`) is untouched — the nrows==1 alias
exception is neither relaxed nor removed. No `ggml_set_output` was added
(`grep -n ggml_set_output src/models/qwen4exp.cpp` shows only the pre-existing
model-output logits call). No quant/type/precision change anywhere in the diff.

Mechanism: with `GGML_TP5_MOE_ROUTER_SCRATCH=1`, each Qwen4Exp layer with a
multi-row (2–4) 512/top-10 router marks its direct F32 router projection
(softmax `src[0]`) `NO_REUSE`. The gallocr then (a) never reuses it as an
in-place parent, (b) never frees it during graph allocation, (c) refuses a
saved plan whose flag state mismatches the current graph, and (d) — with
`GGML_TP5_ROUTER_SCRATCH_AUDIT=1` — fails the whole reserve if the finished
plan places any other allocation inside the retained extent (an in-place
continuation of the retained tensor, e.g. the fused top-k route sharing its
logits bytes, is no violation and is audited once, at the owning node). The fused
`topk_moe` consumer reads only `logits` and writes `weights`/`ids`; with the
projection guaranteed unshared, the `ggml_cuda_check_fusion_memory_ranges`
veto (which today disables the fusion for `nrows > 1`) passes on real
pointers, so no consumer-side alias exception is needed.

Env knobs (lane rule: only exact `"1"` enables; unset/`""`/`0`/`true`/`2` = off):

| Env | Effect |
|---|---|
| `GGML_TP5_MOE_ROUTER_SCRATCH` | enables retention; OFF ≡ stock bit-for-bit (no flags set, identical graph and plan) |
| `GGML_TP5_ROUTER_SCRATCH_AUDIT` | plan-level overlap audit, fail-closed (reserve returns false + `GGML_LOG_ERROR`) |
| `GGML_TP5_ROUTER_SCRATCH_DEBUG` | per-layer retention log (llama side) and multi-row fusion fire log (CUDA side) |

## 2. CPU leg (Devops, serial; no GPU)

Superseded by `ROUTER_SCRATCH_CPU_FIXTURE.md`, which needs no production
checkpoint: it generates a real 512-expert / top-10 Qwen4Exp GGUF with every
hidden width at its floor (`n_embd = 8`, 2 trunk + 1 NextN block, ~1.3 MB),
then checks OFF-vs-stock, retention firing, N1/N2/N3/N4 row windows, plan
disjointness against physical pointers and the gallocr's own fail-closed audit.

```sh
src=/home/kunweiz/work/merge-router-20261031
build=$src/build-router-cpu
cmake -S "$src" -B "$build" -DGGML_CUDA=OFF -DGGML_HIP=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
/home/kunweiz/.local/bin/build-slot cmake --build "$build" -j2 \
      --target llama-tp5-router-fixture llama-tp5-router-verify
"$src"/examples/tp5-router-verify/run-router-cpu.sh "$build/bin" <out-dir>
```

Pass criteria live in that document (§3) and are printed by the script as
`CRITERIA MET` or `FAIL` together with the evidence line. Retained:
`OFF ≡ stock` and `OFF ≡ ON` are bitwise on the six-plus model-logits hashes,
retention fires on 2..4 rows and only there, and the retained extent shares its
block with nothing but its own in-place continuation.

Limits: CPU execution never fires the CUDA fusion and cannot prove device
dispatch or cross-warp lifetime; it proves the allocator/retention lifecycle
and the model-level numerics of the unfused path under retention. The fixture's
trunk blocks are all linear-attention blocks, so the full-attention and QSA
indexer paths are not part of the CPU claim (see
`ROUTER_SCRATCH_CPU_FIXTURE.md` §2.2).

## 3. GPU leg (13022, Devops; dual lock, 1/1 lane; UNPROVEN)

Per handoff §8.7, clone an existing lane's `patch/` (e.g.
`mtp-lb-default-on-hip-20261030`), keep `run.py`'s RSZ_RT pop, and set the
ISO for the ON legs:

```
GGML_TP5_MOE_ROUTER_SCRATCH=1
GGML_TP5_ROUTER_SCRATCH_AUDIT=1
GGML_TP5_ROUTER_SCRATCH_DEBUG=1
```

Legs OFF_1/ON_1/OFF_2/ON_2 with OFF = all three envs unset. Gates mapped to
G1–G4:

- G1: patch applies clean; no selected-install files touched.
- G2: relink reproduces the baseline library hash for OFF; OFF logits
  bitwise == pristine stock logits (harness sha).
- G3: in ON stderr, `grep -c "router_scratch retain layer"` = 48 per verify
  graph, `grep "topk_moe fused n_rows=3" | grep -c "no_reuse=1"` > 0
  (fusion actually consumed the scratch), the allocator line
  `retention audit clean (reserve): N retained allocations, … , rows=R,
  graph_uid=…, plan_id=…, fp=…` is present with N = 2 trunk layers × the row
  count and R equal to that row count (a `(size)` line does not count), the
  matching `router_scratch plan identity … match=1` line carries the same
  graph_uid/plan_id/fp, and there is no `overlaps` / `view, it cannot be
  retained` line. Do NOT accept the `cb_eval` observer line
  (`VERIFY_REAL_DIAG`) as this evidence: it is CPU-only, unavailable on relay,
  changes scheduler timing and only sees executed split nodes.
- G4: OFF vs ON logits soft gate (numeric-order-only: mean KL < 1e-6,
  max row KL < 1e-5, top1 ≥ 99.9%, flips only at margin ≤ 0.10); N1 legs
  bitwise unchanged vs stock (nrows==1 exception intact); no RSZ_RT in any leg.

Fusion-firing alone is not a speed claim; ms/tok deltas are informational
until a formal 512×2 run is authorized.

## 3b. Overlap evidence: allocator audit is real but not promotable; cb_eval is not evidence

**Production allocator audit (real work, insufficient on its own).**
`ggml/src/ggml-alloc.c:ggml_gallocr_check_no_reuse_plan()` runs inside
`ggml_gallocr_reserve_n_impl()`, i.e. in the same reserve the production run
performs, over the **complete node + leaf plan** including cross-backend input
copies that the scheduler creates in `sched->graph`. It needs no eval callback,
adds no synchronize, and cannot alter scheduling. Gate:
`GGML_TP5_ROUTER_SCRATCH_AUDIT=1` (exact) together with
`GGML_TP5_MOE_ROUTER_SCRATCH=1`.

| Line | Meaning |
|---|---|
| `retention audit clean (reserve): N retained allocations, B bytes, rows=R, nodes=M, graph_uid=U, plan_id=P, fp=F` | whole plan audited, no retained extent overlaps another node/leaf allocation. `entry` is `reserve` for `ggml_gallocr_reserve_n`, `size` for `ggml_gallocr_reserve_n_size` (measurement, allocates nothing); `rows` = common `ne[1]` of the retained set (`mixed` if they disagree); `U`/`P`/`F` identify the planned graph and the planning pass |
| `retention audit enabled (reserve): 0 retained allocations, nodes=M, graph_uid=U, plan_id=P, fp=F` | audit ran, nothing retained |
| `retained tensor '…' overlaps '…' / overlaps leaf '…'` | contract violation; reserve returns false (fail-closed) |
| `retained tensor '…' is a view, it cannot be retained` | malformed retention; also fail-closed |

**How it is joined to the submitted graph.** The audit records the identity of the
plan it judged and prints it (`graph_uid=`, `plan_id=`, `fp=`), and
`llama_context` reads it back: `graph_reserve` records the uid the scheduler
stamped on the reserved graph plus the allocator's `plan_id`/`fp`, and
`graph_compute` prints `router_scratch plan identity … match=<0|1>` for a graph it
submits with retained inputs. The join key is the pair `(graph_uid, plan_id)`: a
re-reserve of the same graph object keeps its uid but moves `plan_id`, so the pair
separates "same graph" from "same plan". The `(size)` measurement entry is excluded
by `entry_size=0`. Promotion requires `match=1` **and** the identity fields equal to
the last clean `(reserve)` line; the earlier "phase tag" sketch stays withdrawn
because it declared an intent instead of recording an identity.

**Observation only (NOT promotion evidence).**
`examples/tp5-router-verify/tp5-router-scratch-diag.h` installs
`cparams.cb_eval`. Its limits are real and must be stated:

- unavailable on the `tp5_relay` path — llama rejects a non-null `cb_eval`
  ("no sampler, callback or alternate compute policy");
- a non-null `cb_eval` disables the single-split `defer_input_flush` fast path
  and the callback branch synchronizes the split backend per batched range, so
  the schedule differs from production;
- the callback is invoked for `split->graph` nodes only; cross-backend input
  copies and their dependency nodes live in `sched->graph` and are never
  visible. `DISJOINT` therefore means "disjoint from the executed nodes", not
  "disjoint from every allocation in the plan".

Its `VERIFY_REAL_DIAG … promotion_evidence=0 scope=cpu_non_relay_executed_nodes_only`
line may be used for triage (is the flag still on an executed node? which layer
by name? did the step execute it?) and **must not on its own pass the retention
gate or promote the candidate**. Any `OVERLAP`/`UNRESOLVED` line it emits is
still a legitimate red flag, but absence of them is not proof.

**Current verdict.** C4b in `run-router-cpu.sh` grades the join described above:
`match=1` with matching identity fields on the ON leg passes; a missing identity
line, `match=0`, a mismatch, or no well-formed clean line stays **UNPROVEN** and
fails. Nothing here has been run: the candidate is still unbuilt.

## 4. Explicitly unproven (carry forward)

- Candidate binaries have never been built or run (CPU or GPU).
- Real-consumer fusion firing, allocation disjointness in a live backend, and
  N1/N2/N3 model-level numerics are all UNPROVEN until §2/§3 run.
- The fixture and the rewritten verifier are themselves UNEXECUTED: the
  fixture has never been written, loaded, or decoded from; whether the stock
  model path accepts `n_embd = 8` / two linear trunk blocks end to end is the
  first thing the CPU leg reports.
- Cross-warp lifetime of the retained scratch is untestable without HIP.
- The bounded row window (`2..4`) is a deliberate narrowing of the earlier
  `> 1` sketch to avoid long-prompt retention footprints (512 × n_ubatch × 4 B
  × 48 layers held per plan); it changes no N1 behavior and no numerics.
