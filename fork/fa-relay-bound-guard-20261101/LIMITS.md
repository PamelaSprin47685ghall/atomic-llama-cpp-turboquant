# LIMITS — FA src[5] bound safety on the relay replay path

Date 2026-11-01. Tree read: `/home/kunweiz/work/merge-fa-20261031` @ `03024e721`.
Static reading and static edits only: **nothing here was compiled, linked, run on GPU or
measured.** Every "confirmed" line below is a source fact with a file:line; every other line is
labelled unproven or refused. Where this file disagrees with `DESIGN.md`, `MINIMAL-DESIGN.md` or
`REVIEW.md`, **this file wins** — those describe the host-selector lane, this one describes the
relay lane and what it cannot have.

---

## 0. Answer in one paragraph

The relay lane can be made to **refuse** an untrustworthy bound before it submits anything, and it
can be made to **stop the process** at the launch that would consume a bound whose identity is
wrong. It cannot be given a per-replay device verdict that is *acted upon before consumption*
without either a host round trip or a change to the relay's own completion protocol. A NaN output
is not a refusal: nothing between the FA output and the sampled token checks finiteness
(`RETRACTION.md` §1, `llama-sampler.cpp:295-325` softmax and `:648-652` chain apply; `grep
isfinite|isnan src/llama-sampler.cpp` → 0 matches). So the lane is implemented as
**refuse-before-submit + refuse-at-launch + no new host round trip**, and the remaining staleness
class is named in §4 rather than papered over.

---

## 1. What the relay actually does per replay (the ordering that decides everything)

| step | site | fact |
|---|---|---|
| host refreshes the bound | `llama-graph.cpp:650` → `llama-kv-cache.cpp:2715-2725` | `ggml_backend_tensor_set(dst, &n_kv_eff, 0, 4)`; on the meta buffer this is a **mirrored, blocking** write to every rank (`ggml-backend-meta.cpp:1544-1549` → `ggml-cuda.cu:917-918`: `cudaMemcpyAsync(..., cudaStreamPerThread)` + `cudaStreamSynchronize`). Complete before the submission is enqueued. |
| scheduler uploads inputs | `ggml-backend.cpp:1836-1875` | a `GGML_TENSOR_FLAG_INPUT` source is copied to the split backend's device buffer, waiting on the previous submission of that backend first (`:1840-1846`). |
| relay submits | `relay.hip:1481-1557` (`submit`) | one call per submission; it validates topology/bindings, then either captures one epoch or launches a previously captured one. |
| relay launches | `relay.hip:1300-1312` | `hipGraphLaunch(exec[r], lanes[0].stream)`; on failure `break` — *"Never fall back after a possibly accepted launch"*. |
| relay returns | `relay.hip:1335-1348` | seals, records native tails (`hipEventRecord`), returns `p.status.sealed`. **No `hipStreamQuery`, no `hipStreamSynchronize`, no D2H on this path.** |
| consumer reads | `llama-context.cpp:4053-4060`, `:1776-1781` | the caller takes a D2H of logits (`ggml_backend_tensor_get_async`) and its own `ggml_backend_sched_synchronize`. |

Two consequences, both of which are load-bearing:

1. **A device flag written by a kernel of epoch *N* is not observable on the host until epoch *N*
   has been drained.** The relay deliberately offers no host hook between launch and drain
   (`relay.hip:1335-1348`), so "arm → kernel validates → host reads flag" would either need a new
   synchronize inside `submit` (forbidden: it would serialize five ranks per step) or would read
   the flag one epoch late — which is precisely the defect `REVIEW.md` §1.1 rejected in the
   host-selector candidate.
2. **The bound the epoch consumes is already on the device before the epoch is enqueued**, so the
   *identity* of the tensor (which buffer, which slot) can be checked on the host per submission,
   even though its *value* cannot be.

---

## 2. The two producer classes, and why only one survives

**Class A — host-produced per-eval bound** (`GGML_TP5_FA_RTLEN` / `GGML_TP5_FA_CHECKED_TAIL`):
`llama-graph.cpp:3282-3286` creates one `i32[1]` tensor per graph, `ggml_set_input`, refreshed per
eval by `set_input_nkv_eff` and attached to every plain FA node as `src[5]`
(`llama-graph.cpp:3499-3501`). One tensor per graph, one writer, written before submission.
→ **Admitted.** Its staleness is bounded by "the value written for this eval", which is exactly the
contract the graph-input copy already enforces.

**Class B — kernel-produced bound inside the same graph** (`GGML_TP5_FA_CHECKED_TAIL` streamed
chunk staging): `fattn.cu:2276-2281` builds a stack `ggml_tensor chunk_nkv_eff` with `buffer ==
nullptr` and `data == chunk_bounds.ptr + chunk`, produced per chunk by
`kv_stream_chunk_bound_dual<<<1,1>>>` (`fattn.cu:2242-2247`) on the compute stream, immediately
before that chunk's FA work. Inside one *direct* submission the producer/consumer order is a stream
order and is fine. **On a replay it is not provable from the provider's own data**: the recorded
launch re-executes the producer, but the relay's frame key (`relay.hip:1366-1369`) covers tensor
metadata, not the ordering between a pool slot's writer and its readers across stages.
→ **Refused for relay replay**, in `relay.hip submit()` before `begin()`, i.e. before any epoch,
capture or launch exists (`g1_relay_static.sh` §4 asserts this ordering).

Note this class cannot reach the relay today anyway: `llama-context.cpp:248-272` refuses
`params.kv_stream_arena_mib != 0` on the relay lane, so the streamed staging does not run there.
The refusal is therefore a guard on a *future* configuration, not a fix for a live bug. It is
written because the ticket asks for "refuse before consumption on every replay", and a guard that
only exists while an unrelated admission check happens to hold is not a guarantee.

---

## 3. What the patch actually enforces

Three edits (`prepare_relay_bound_guard.py`), all behind `GGML_TP5_FA_RELAY_BOUND=1`:

| # | site | effect | when |
|---|---|---|---|
| R1 | `relay.hip submit()`, before `begin()` | scans the stage graphs; a `FLASH_ATTN_EXT` node whose `src[5]` is class B ⇒ `fail(ENOTSUP)` and `return false` | host, pre-submit; the caller then aborts the submission (`ggml-backend-meta.cpp:3797-3801`) |
| R2 | **head of `launch_fattn()`** — before the pool allocations, before every K/V conversion, before the first kernel launch | class B ⇒ `GGML_ABORT`; `src[5]` present but not an allocated single-scalar tensor ⇒ `GGML_ABORT` | host, inside the launch call, before any K/V work of the node |
| R3 | same site | **nothing about the value** | — |

R2's placement is the head of the function, not the line where `n_kv_eff_t` is captured: that
line sits after the K→fp16 conversion (`fattn-common.cuh:1605` in the pinned file), and a first
revision of this lane put the check there. Devops measured it (VERIFY-RELAY-BOUND-GUARD-20261101
§3 step 5, §8) and it is fixed. G1 step 5 now asserts `check < pool_alloc < to_fp16 <
kernel_launch` on a comment-stripped copy of the file, so the same mistake cannot pass again.

**R1 and R2 cover disjoint classes, and this must not be read as two chances at one.**

| boundary | class it refuses | reachable when |
|---|---|---|
| R1 (submit) | class B only — a `src[5]` produced by a kernel of this graph | every submission through the relay provider |
| R2 (launch) | (a) a `src[5]` that is not a readable single allocated int32 scalar; (b) class B, as a backstop | (a) any caller; (b) only a caller that reaches `launch_fattn` **without** going through `submit()` — for the relay lane R1 has already refused it |

Both use the same classifier (`ggml_cuda_fattn_relay_bound_in_graph`), so they cannot drift; the
overlap is deliberate (belt and braces for (b)) and the comments say so.

**One predicate, called by both boundaries.** `ggml_cuda_fattn_relay_bound_active(dst)` is the
only place the three conditions of §3a are evaluated: R2 branches on it directly, and the
classifier `ggml_cuda_fattn_relay_bound_in_graph()` begins by calling it, so R1's per-node test
goes through it too. A first revision defined `active()` and then never called it — dead code
whose existence made the G1 symbol count look right (Devops finding on v2). It is now the single
entry point, and `g1_relay_static.sh` step 3 asserts per file and per role that it is defined
once, called at least once, and used in a branch.

`GGML_ABORT` (`ggml.c:252-272`) prints and calls `abort()`; a registered abort callback cannot
return (`ggml.c:264-271` calls the callback and then still reaches `abort()`), so there is no
path where the process continues after R2.

---

## 3a. Activation — when this lane does nothing at all

Everything in §3 is driven by one predicate, `ggml_cuda_fattn_relay_bound_active(dst)`, and every
check is **dead** unless all three of its conditions hold:

1. `GGML_TP5_FA_RELAY_BOUND=1` (this lane's own gate);
2. one of `GGML_TP5_FA_CHECKED_TAIL=1`, `GGML_TP5_FA_RTLEN=1`, `GGML_TP5_FA_NKV_PARAM=1` —
   without one of these `src[5]` is never produced and `ggml_cuda_fattn_rtlen_tensor()` returns
   null (`llama-graph.cpp:3282-3286`, `:3499-3501`);
3. the node is a plain `FLASH_ATTN_EXT` carrying an `i32[1]` `src[5]`.

The relay lane as it runs today sets none of these (`relay.hip` reads only
`GGML_TP5_RELAY_PRIORITY_PAIR`), so **the guard does not protect the relay configuration that is
actually being run**, and no relay run may be cited as evidence about it. This is opt-in by
design, and Devops' finding F-A is correct: it has to be said out loud rather than implied.

---

## 4. The class that remains — stated plainly

**S1 — legal-but-stale value.** If the host writes a bound that is within `[0, built]` but belongs
to a different request/step than the one the epoch is about to consume, this lane does not detect
it. The kernels clamp with `min(bound, built)` (`fattn-tile.cuh:869`, `fattn-vec.cuh:54`), so the
result is a *plausible* attention over the wrong number of rows — the one outcome the ticket
forbids — and the fp64 gate would only catch it afterwards.

Why it is not closed here, honestly:

* closing it needs the value, and reading the value on the host is a D2H (forbidden), while
  reading it on the device and acting on it needs a verdict the same grid can honour — which means
  the kernel must *refuse to compute*, and "refuse" in a kernel means either a poisoned output
  (not a refusal, §0) or a device-side trap. A trap is a new failure mode for a five-rank
  captured graph (a rank dies inside a graph exec; the other four are already launched) and it is
  not proposed.
* the ordering that would make the value provably current is exactly the one the relay does not
  publish: `submit()` returns `sealed`, not `drained` (`relay.hip:1348`), and the drain is the
  caller's `synchronize()` in `llama-context.cpp:1781`.

So S1 stays open. It is narrower than before this lane — class B is refused and class A has a
single writer per eval — but it is not zero, and no PASS may be claimed for it.

**S2 — the value's identity is not revalidated per replay.** The bound tensor's address is part
of the frame key (`relay.hip:1366-1369`), so a *different* tensor forces a re-capture rather than a
silent reuse. That is the strongest statement available: it means "the same slot", not "a fresh
value".

---

## 5. Refused designs, with the reason (so nobody re-proposes them)

| design | why it is refused |
|---|---|
| NaN as the fail-closed signal | no consumer rejects NaN before sampling (`RETRACTION.md` §1). Post-hoc detection. |
| per-launch device flag + host readback in `submit()` | needs a synchronize inside `submit()`; the relay has none by construction (`relay.hip:1335-1348`) and adding one serializes five ranks per step. |
| flag readback one submission late | that is `REVIEW.md` §1.1, already rejected: the consumer can read the tensor before the verdict. |
| host D2H of the bound before each submission | the round trip this lane exists to avoid; also unbounded per step. |
| a shared slot across windows | two windows are two contexts/processes; a shared static was the r14 defect. The lane uses no shared state at all. |
| scan the graphs in `submit()` and *also* trust a host value | a host scan proves topology, never freshness; it would dress up S1 as solved. |
| raise the stock clamp to an abort inside the kernel | changes the numerics contract of a landed lane (`min()` clamping is documented at `fattn-tile.cuh:868-875`) for a case the CPU consumer already asserts on (`ggml-cpu/ops.cpp:8577`, `:9192`). |

---

## 6. Concrete, implementable boundaries (if this is to be closed properly)

These are the only shapes that would close S1, each with what it costs. None is implemented here.

1. **Publish a per-replay completion boundary in the relay.** `submit()` would return
   `drained` (i.e. do the `hipStreamQuery`/event drain it currently refuses) at least for
   `n_seq_max == 1` decode. Cost: a per-step host wait on five ranks; kills the async protocol
   the lane was built for. Only acceptable if the caller already waits (it does, for logits), in
   which case the drain is free — **this is the one candidate worth measuring.**
2. **Make the bound part of the frame key by value.** Capture the bound's content into the
   signature; a changed value forces re-capture. Cost: one re-capture per step, i.e. the capture
   win disappears.
3. **Move the bound out of a mutable tensor**: compute the per-request bound inside the epoch with
   a kernel that *precedes* the FA nodes in the same stage graph and is covered by the frame key.
   That is class B again, and it needs the ordering proof class B lacks.

---

## 7. What Devops can and cannot conclude from the gate

`g1_relay_static.sh` proves, and only proves: the generator applies with every anchor unique and
every pinned hash matching; the patched files differ only as listed; R1 sits before `begin()`; R2
sits before the pool allocation, every K/V conversion and the first launch; no
`cudaMemcpy*DeviceToHost`/`cudaStreamSynchronize`/`cudaStreamQuery` was added and no
`hipPointerGetAttributes` was added to `relay.hip`; the gate is an exact-`"1"` env read; no NaN
literal appears in the patched FA header; the stock op-param abort and the CPU consumer assertion
are still present.

Every one of those counts runs on a **comment-stripped copy** of the file (`codeview()`), because
the first revision of this gate counted comment text: a comment naming the classifier made step 3
read 3 instead of 2, and a comment naming `hipStreamSynchronize` tripped step 6. Both were false
positives in the CHECK, not in the patch; both are fixed, and Devops' §3 measurement is what
found them.

It cannot prove: that the code compiles (G2; `relay.hip` needs `hipcc` and was not compiled), that
the classifier never misclassifies a real node (G3 micro), that the guard is active in any given
run (§3a — by default it is not), or anything about S1. A `STATIC-OK` line is a statement about the
patch text, not about the lane's safety.
