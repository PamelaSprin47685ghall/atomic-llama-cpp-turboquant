# TQRerotCompose

TQRerotCompose is the C layer for token-frontier shared-whiteboard generation.
It uses one installed A model/context, the independent `TQRerot` state library,
`TQRelay` submission accounting, `TQText` output assembly, and `TQSampling`
numerical selection. It has no private
A headers and does not copy a server loop into B.

This release implements an explicit task DAG with the `assistant-eot` framing
profile. Each stage is a complete native assistant turn. An edge delays the
start of its child until its parent has sealed. It does not restrict attention
to ancestors: all committed public work is available to every running reader,
including work from a completed non-ancestor.

## Build

The repository pipeline installs A, each B library, and then each C consumer:

```sh
python3 fork/build.py --backend vulkan --build-dir build-rerot --jobs 4
build-rerot/install/bin/tq-rerot-compose --help
```

Use `--backend cpu` for a CPU build. CUDA and HIP use the corresponding A build
options and need their own toolchains and runtime qualification.

The C source also builds in isolation from installed packages:

```sh
cmake -S fork/rerot-compose -B build-rerot-compose \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH='/installed/A;/installed/TQRerot;/installed/TQRelay;/installed/TQText;/installed/TQSampling'
cmake --build build-rerot-compose --parallel 4
ctest --test-dir build-rerot-compose --output-on-failure
```

The installed A must include the reviewed reader-view and recurrent-state
interfaces and `batch_invariant_matmul` context request declared in its public
`llama.h`. The context parameter struct has changed. Rebuild consumers against
that installed header and library together.

## Run

```sh
build-rerot/install/bin/tq-rerot-compose \
  --model /models/instruct.gguf \
  --prompt 'What is 17 times 19? Give the exact result.' \
  --stage 'multiply=Compute the product. Reply with the integer only.' \
  --stage 'check=Check the product by an independent method. Reply with the integer only.' \
  --stage 'review=Compare the preceding calculations. Reply with the agreed integer only.' \
  --after review=multiply --after review=check \
  --synthesis 'Answer the original question with the integer only.' \
  --parallel 2 --ctx-size 8192 --total-kv 8192 \
  --worker-tokens 512 --final-tokens 512 \
  --batch-size 512 --ubatch-size 128 \
  --gpu-layers 99 --devices Vulkan0 --split-mode layer \
  --seed 0 --temperature 0 --json
```

`--stage` and `--after` can repeat. Stage names use ASCII letters, digits,
underscore, hyphen, and dot. Each `--after CHILD=PARENT` names a start dependency.
Unknown names, duplicate names, cycles, missing inputs, invalid numeric values,
and insufficient configured capacity are errors. Each worker gets its own
sampler and native sequence identity. A physical pen does not select its RNG.

Each stage retains its candidate storage across sampling calls. C uses the
installed `TQSampling` shortlist for a greedy row with a unique maximum. For
stochastic sampling, compaction requires `1 <= top-k <= 128`, distinct retained
logits, and a unique cutoff logit across the full vocabulary. These conditions
give the same ordered candidate row as native top-k. Retained ties, cutoff ties,
and larger stochastic top-k values use the original full row in ascending token
order. Native greedy or native top-k/temperature/distribution still selects and
accepts every token with the same stage-local RNG; the one-candidate case retains
A's ordinary no-draw behavior. Non-finite input logits remain errors.

The `sealed` JSON event reports the actual `sampling.compact_rows` and
`sampling.full_rows` counts for that stage. There is no extra sampler mode flag.
`tests/test-sampling.cpp` pairs this adapter with the former full-candidate
native chain through unique rows, ties, changing vocabulary sizes, invalid
inputs, and interleaved stages. This is sampler equivalence evidence; model
throughput needs a separate measurement.

The default is CPU execution (`--gpu-layers 0`). Device names are the public A
backend names. `--devices` accepts a comma-separated list. `none`, `layer`, and
`row` are passed to A's model placement policy. A owns the actual devices,
transfers, and scheduler; C does not invent per-device completion receipts.

The JSON `config` event keeps the requested split mode and device names separate
from the actual context. `devices` lists available backends; the
`native_context.model_devices` entries identify the devices selected by A's
model loader. That selection does not prove that every device holds layers or
ran a kernel. `native_context` also reports effective K/V types, actual batch
and microbatch sizes, the F32 reader policy, configured Flash Attention, and
the scheduler's current `pipeline_parallel_enabled` value. Its
`batch_invariant_matmul_requested` field comes from A's public getter and
records the retained context request. It does not observe the effective
precision or pipeline of every graph operation.

Layer placement alone does not enable pipeline parallelism. A also checks
device count, full offload, asynchronous compute and event support. It can
disable the pipeline if graph allocation fails. C therefore reads the current
pipeline value again at every committed `frontier` and in the final `result`.
An enabled scheduler flag describes the execution configuration; overlap and
performance still require measurements from the actual devices.

On the qualified RX 6800 Vulkan device, `none` and `layer` completed real
whiteboard execution. `row` failed at native model load because Vulkan0 does
not support split buffers. This is an explicit backend limitation, and no
automatic split-mode fallback is performed. Multi-device PP still needs its
own execution evidence; see [validation](../VALIDATION.md).

The native chat template must be supported by the public A renderer. A stage
frame consists of the native assistant close, a user handoff containing that
stage's instruction, and an assistant open. Tokenization must preserve the
shared base prefix exactly. An incompatible template is rejected. An explicit
`--chat-template` can select a supported renderer.

## Shared-whiteboard semantics

All bound running stages advance at one token per frontier. They first read
the same committed revision. Each current token is visible to its own query;
another stage's current token is not visible during that forward. Once every
member has completed its real forward, B publishes all public tokens together.
Every member's next forward immediately sees those newly committed peer tokens.

| Boundary | What a reader can see |
| --- | --- |
| Start of frontier `f` | Every stage's public prefix committed through `f - 1` |
| Forward in `f` | That frozen peer world plus its own causal current token |
| Publication of `f` | The whole cohort's successfully computed public tokens |
| Forward in `f + 1` | All peer tokens just published at `f`, even if those peers are still running |

The token sampled from a forward is pending until its next forward computes
the corresponding KV. It is not displayed or published early. A reader view
owns the exact prefix lengths; an old view cannot grow after a later commit.

A new stage's FRAME may contain several tokens. C executes it in token waves,
with at most one current query for each native sequence in each A call. Earlier
own FRAME tokens become resident for its later causal queries. All these waves
keep the original frozen peer revision and form one B publication transaction.
This does not turn BODY sharing into whole-response sharing.

Readers use the DAG's deterministic reader-relative order, with their own stage
last. That order assigns the keys their virtual positions; a shared key stays
stored at its writer position. A reconstructs the effective query phase where
required and places the resulting visible scores in the corresponding physical
KV slots. Hidden slots receive negative infinity. One global softmax per
query/head then feeds the native batched V operation. Logical reader order and
physical reduction order therefore have separate roles; spans are not
normalized independently.

## Ownership and capacity

There is one native context and one physical attention KV pool. The base prompt
is evaluated once in sequence 0 and is never advanced again. Native sequence
copy shares that immutable prefix and establishes each new stage's recurrent
lineage. C never exposes native state pointers to B.

For `W` workers and at most `P` active pens, C requests `W + 2` logical sequence
identities and `P + 1` native recurrent-state slots. The additional recurrent
slot holds the immutable base. Sealing a stage releases only its recurrent or
convolution state. Its public attention keys retain their logical owner for
later readers. A new stage receives a fresh lineage from the base when a pen
becomes available. Synthesis is a new stage after all workers seal.

The `config` event reports `physical_recurrent_slots_requested = P + 1` alongside
the logical sequence capacity. The field describes the request made to A; it
does not measure allocated bytes or claim that an attention-only model owns
that many recurrent buffers.

Pens stay bound until source end. The DAG can contain more workers than pens.
Changing `P` can change which work has become visible at a given forward, so
generation is not specified to be identical across different pen counts.

`--ctx-size` is the reader's logical position limit. `--total-kv` is the physical
unified pool capacity. C checks a conservative bound for the full episode,
including frames, all permitted BODY tokens, and private source-end keys. It
does not silently evict public work, truncate the task graph, or auto-fit a
different policy.

At a FRAME-to-BODY transition, the default `--phase-memory` mode synchronizes A
and releases transient compute allocation through its public phase interface.
The KV pool and recurrent states keep their owners. `--no-phase-memory` retains
compute allocation for comparison. Later graph construction allocates its
required transient workspace again.

## Failure and output

Native EOG closes a complete assistant stage in this profile. Its source-end KV
is private and is not part of another reader's whiteboard. Reaching a token
limit without native source end is an incomplete episode and returns an error.
A token limit is not converted into a fabricated successful source end.

The C loop has one caller for the native context. Each submit creates a local
`tq::relay::run_submissions(call, 1)` ledger for one opaque A submission domain.
Its receipt stays alive through actual `llama_synchronize`, completion, and
owned-view cleanup, including the failure path. Completed attempt history is
released after each call. The executor keeps the unique, increasing call ID and
its sticky failure state across calls. A logical cohort size does not imply
that many device submissions or TP ranks.

A decode failure may have written earlier native microbatches. C drains the
actual work, aborts the unpublished B frontier, and terminates the context. It
does not retry the frontier or claim device rollback. Previously committed
text remains a valid prefix.

SIGINT and SIGTERM handlers only set a `sig_atomic_t` flag. Normal C code checks
it before the next native submission, after synchronized Prefill chunks and
FRAME waves, and after sampling but before B commit. A long FRAME does not keep
submitting its remaining waves once cancellation is observed. The current
native call completes and drains before its owners are released; no signal
handler calls A, B, or C++ cleanup code. Cancellation exits with status 130.

Without `--json`, stdout contains the synthesis text. With `--json`, stdout is
NDJSON with `config`, `compute_release`, `frontier`, `token`, `text`, `sealed`,
and `result` events. `frontier.read_revision` identifies the frozen input;
`revision` identifies publication. Each reader includes its stage, pen, virtual
query position, and exact visible span identities, including its own private
rows. A final `result` with
`complete: true` appears only after the synthesis stage has produced source end.
When stdout remains writable, errors emit an `error` event with `complete: false`.
Ordinary errors return status 1, and native diagnostics stay on stderr. Where
SIGPIPE exists, C ignores it so a broken pipe reaches the normal ostream error
and cleanup path. A broken output stream is terminal; its JSON error event
cannot be guaranteed to reach that stream. Even `--help` flushes before deciding
its exit status.

## Backend and optimization boundary

C sets `llama_context_params.batch_invariant_matmul = true` before creating
the native context, so the request applies from the immutable prefix prefill
through stage FRAME and BODY execution. A's projection builders request
`GGML_PREC_BATCH_INVARIANT` (mode 20) for applicable quantized dense `MUL_MAT`
and expert-indexed `MUL_MAT_ID` operations with F32 activations. Enabling the
request only after prefill would leave the shared base state dependent on
the earlier batch policy. While enabled, this projection request takes
precedence over a builder's requested accumulator precision.

The policy preserves the tested backend's native single-query quantized
arithmetic across the query batch. Resident reader K/V remain F32; projection
arithmetic retains its native activation policy, including Vulkan Q8
activation conversion and MMVQ where the device, weight type, and K select
that route. It does not globally disable quantized kernels or force all
linear operations to true-F32 arithmetic. A's default context parameter is
`false`, which retains ordinary optimized quantized matrix selection for
consumers that do not request this consistency policy. CPU and Vulkan
qualification is limited to the tested profiles and recorded devices.

The reader implementation uses existing GGML operations and the installed A
scheduler. QKV and FFN still receive the native batch's multiple rows. A first
computes native batched KQ with the original cache strides. Spans whose effective
query position matches the native query position reuse those scores. Other
spans reconstruct Q from its pre-RoPE source and original parameters, then use
the selected resident K rows to calculate their scores.

The helper scatters only the visible scores back into physical KV slots and
retains the native padding and reduction order through one global softmax. It
then uses native batched V, including A's ordinary layout conversion when
needed. V is not gathered per reader, and no reader owns a persistent KV copy.
Preserving these operation shapes is part of the numerical design; it does not
promise identical results for every model, batch shape, or backend.

This release supports **resident F32 K and F32 V only**, with FA disabled by C.
The public A capability check rejects F16 or mixed-type reader contexts, and C
requests F32 explicitly. Ordinary A decoding retains its existing F16 support.
The F16 reader experiments failed the strict state tests: the CPU reorder/restore
comparison reached absolute logits error `0.11799526`, and the Vulkan row
permutation comparison reached `0.0003576`. Those receipts remain evidence for
a separate precision-policy rewrite covering physical permutation and native
single-sequence restore reallocation. Explaining floating-point differences
does not waive the numerical gate or qualify this reader profile.

The consistency policy can change projection latency. The
[qualification procedure](QUALIFICATION.md#projection-performance-and-dispatch)
separates exact operator comparisons, actual MMVQ dispatch evidence, normal
Release timing, and whole-model generation measurements. Record the measured
cost for the tested shape and candidate; an operator correctness PASS supplies
no throughput claim. Small ubatches also change the rows available to native
matrix and vector kernels.

Current capability checks reject unsupported attention architectures, streamed
KV, sliding-window caches, embeddings, non-default context roles, and unsupported
KV types. No raw private tensor layout is passed through C to emulate support.

The repository's TP5 relay library supplies reusable submission and ownership
primitives. It does not yet supply a whole-model TP5 provider. This consumer's
`--tp5`, `--latebind`, and `--mtp` flags therefore fail explicitly. The current
native A route can use A's own placement options; this is not TP5 execution.
Hardware qualification of PP needs the actual multi-device configuration.

Before a future LateBind or MTP provider can run with this whiteboard, its cached
work must include stage identity, reader revision, key layout, and live state
generation in its validity checks. While peers are live, accepting several draft
tokens against one stale peer view would violate the required token frontier.
The provider must validate each accepted frontier against the newly published
peer world and keep real submitted work owned until completion. These paths
remain rewrite candidates, with no unverified implementation copied here.

The former server's automatic planner, reasoning-end/tool router, and heuristic
template probes are separate from the explicit `assistant-eot` profile. They
are not represented as migrated or qualified by this executable.

## Qualification

The [qualification procedure](QUALIFICATION.md) gives the full source-isolation
command, the CPU/Vulkan dense and expert projection gates, strict reader and
eight-episode generation matrices, cancellation checks, capability evidence,
and rules for retaining previous failed receipts. These are separate gates:
a relocated help check, a protocol trace, and a numerical oracle each establish
their own facts.

`tests/test_cli.py` checks the actual executable's help and refusal paths
without loading a model. `tests/real-whiteboard.py` requires a real model and
retains commands, raw stdout/stderr, elapsed time, binary/model hashes, and
frontier validation results:

```sh
python3 fork/rerot-compose/tests/real-whiteboard.py \
  --binary build-rerot/install/bin/tq-rerot-compose \
  --model /models/instruct.gguf \
  --output build-rerot-whiteboard-proof \
  --parallel 2 --ubatch-sizes 128,1 --phase-modes release,retain --repeat 2 \
  --require-cross-shape-equivalence
```

The real harness requires repeated visibility growth while a peer is still
running, exact peer key identities, correct pen lifetime, dependency admission,
private source-end exclusion, and a complete synthesis. Its default four-worker
fixture requires a completed non-ancestor witness independently of which
writer ends first. It also checks the requested phase mode, every actual
FRAME-to-BODY release with zero remaining compute bytes, and the absence of
release in retain mode. Protocol validation
alone does not prove attention numerics. The public A seam tests separately
compare full-vocabulary logits and an independent small-model attention oracle.

### Current qualification — 2026-10-19

Qualification belongs to the source and installed artifacts recorded by each
receipt. The final recorded CPU and Vulkan scope has passed its source,
operator, reader, complete-generation, and process-failure gates. The
[qualification evidence index](../audit/rerot-hardening-qualification-20261019.json)
binds the source, installed artifacts, model, devices, commands, and receipts;
it also distinguishes normal Release measurements from DEBUG diagnostics.
These results qualify those recorded artifacts and workloads. Earlier
candidates retain their own results below, including failed trajectories.

| Recorded final gate | Completed result | Scope |
| --- | --- | --- |
| CPU source isolation and relocation | PASS | `isolation-expert` rebuilt A from the fixed upstream and 34 reviewed overlays without `--a-prefix`. Source/build copies were removed before relocated consumers and four real generation cases passed. Those four cases establish their contract/phase scope; the separate strict eight-case matrix follows below. |
| Pure CPU public A seams | 11/11 CTests PASS | The final relocated `delivered` CPU installation, including the public dense and expert projection gates. |
| Public A seams against `install-vulkan-expert` | 20/20 CTests PASS | All tests registered in that installed-candidate seam build; backend-specific scope follows each test. |
| CPU expert-indexed projections | 1,114 graphs and 431 bitwise comparisons PASS | CPU backend selected from `install-vulkan-expert`; independent native DEFAULT N=1 references, four quantized types, shared/per-slot activations, padded ID rows, repeated and reordered expert IDs, bias/scale, and the 65,540-slot dispatch boundary. |
| Vulkan expert-indexed projections | 1,114 graphs and 431 bitwise comparisons PASS | Vulkan backend selected from the same installation; the same public `MUL_MAT_ID` fixture. CPU and Vulkan are each compared with their own native references. |
| Vulkan dense projections | 597 graphs and 90 bitwise comparisons, including 18 bias graphs, PASS | Independent native DEFAULT N=1 references; IQ4_NL, Q8_0, and Q4_K across the fixture's query counts, head layouts, and orders. |
| Strict CPU full-model reader | 33 groups and 12 cross-shape comparisons PASS; same-shape and cross-shape equality maxima both 0 | The final pure CPU installation; 72 decode calls and 169 rows. The 33 groups include 31 equality comparisons and two causal checks. |
| Strict Vulkan full-model reader | 33 groups and 12 cross-shape comparisons PASS; same-shape and cross-shape equality maxima both `1.52587890625e-5` | The final Vulkan installation; the 33 groups include the causal checks. CPU and Vulkan are each checked against their own controls. |
| Independent CPU attention oracle | 16 cases PASS; reader maximum F64 error `6.75223405188774e-7`, ordinary maximum `3.7782220707605063e-7`; seven unsupported-profile checks PASS | Actual tiny-GGUF execution using the final pure CPU installation. |
| Independent Vulkan attention oracle | 16 cases PASS; reader maximum F64 error `6.539480768563699e-7`, ordinary maximum `6.109807473286821e-7` | Actual tiny-GGUF execution using the final Vulkan installation; earlier oracle maxima remain historical values. |
| C suites | Final CPU 5/5 and Vulkan-linked C 5/5 CTests PASS, including real native CPU cancellation | Each suite uses its recorded installation. The CPU abort test retains its CPU execution scope even when linked to the Vulkan-enabled A installation. |
| Complete CPU whiteboard matrix | 8/8 PASS; all four qualification categories PASS | Final pure CPU execution: four workers and two pens, `ubatch=128,1 × release,retain × repeat=2`, with strict cross-shape equivalence. Each run recorded 764 committed frontiers and 208 live peer reads. |
| Complete Vulkan whiteboard matrix | 8/8 PASS; all four qualification categories PASS | `whiteboard-vulkan-expert`: four workers and two pens, `ubatch=128,1 × release,retain × repeat=2`, with strict cross-shape equivalence. Each run recorded 748 committed frontiers and 208 live peer reads. |
| CPU process failure checks | SIGINT PASS: exit 130 in `0.1603012448` s after injection; broken pipe PASS: exit 1 in `0.2155467896` s after injection | Actual process termination on the final pure CPU installation. |
| Vulkan process failure checks | SIGINT PASS: exit 130 in `0.0998307341` s after injection; broken pipe PASS: exit 1 in `0.1001606318` s after injection | Actual process termination on the final Vulkan installation; this does not identify the exact GPU instruction at signal delivery. |

Separate DEBUG receipts also passed the MMVQ checker for `auto`, `force`,
`disable`, and both-controls-present policies. They observed disable precedence
and the IQ4_NL native fallback. The expert fixture's DEBUG run completed all
1,114 graphs and recorded eight actual pairs of Y dispatch sizes `65535 + 5`
at the 65,540-slot boundary. This supplies dispatch evidence alongside the
numeric fixture; the ordinary fixture footer by itself does not observe
dispatch or fusion.

#### Measured projection cost

Two normal Release dense timing runs reversed which mode started first. Each
completed nine pairs, 18 mode runs, and 288 graphs, with one first computation
and 15 warm computations per mode. The resident `SCALE -> MUL_MAT` workload
used K=2048, M=1025, and IQ4_NL, Q8_0, and Q4_K weights. Across those types and
both orderings, the ratio of mode-20 to DEFAULT warm medians was:

| Query columns | Mode-20 / DEFAULT median ratio | Observed cost on these graphs |
| --- | --- | --- |
| 9 | `0.4694–0.4991` | Mode 20 took about 50–53% less time. |
| 32 | `1.0926–1.3114` | Mode 20 took about 9–31% more time. |
| 128 | `1.8571–2.0191` | Mode 20 took about 86–102% more time. |

The separate MMVQ auto/disable comparison used four processes in ABBA order.
For Q4_K at N=8, the recorded warm medians were `63.05–63.54` microseconds with
auto and `77.14–77.38` microseconds with MMVQ disabled. Other cases showed
variation or no advantage, so these results do not establish a general MMVQ
speedup. Both timing experiments measure their resident operator graphs,
including submission and synchronization. They are not full-model throughput
measurements, and the dense mode ratios do not measure expert-ID latency.
The evidence index retains the raw samples, build/environment identity, and
device scope; DEBUG timings are kept separate from these Release results.

#### Earlier diagnostics and candidate results

The GPU prefix diagnostic completed four 32-row groups with zero
greedy-token mismatches in each group. Maximum absolute logits differences
were `6.008148193359375e-5` without tracing and `8.106231689453125e-5` with
tracing. These are separate diagnostic observations, not a completed strict
whiteboard matrix or a performance comparison. The subsequent complete matrix
has its own passing receipt in `whiteboard-vulkan-expert`; its qualification
comes from the full run, not this diagnostic.

The following results retain their earlier candidate identities:

| Gate | Completed result | Scope |
| --- | --- | --- |
| Independent B | Native CTests 4/4 PASS | Public BODY rejection and existing DAG, transaction, bounds, and independent numerical tests; sanitizer receipts retain their own source identity. |
| Count-fixed CPU public A seams | 8/8 PASS | That CPU candidate's registered public A seam tests. |
| Count-fixed CPU F32 reader | 31 same-shape equality comparisons plus 2 causal groups, and 12 cross-shape comparisons PASS; maximum absolute logits difference 0 in the equality comparisons | Actual full LFM2.5 model under the original strict gates. The 33 groups include the 2 causal groups; they are not 33 equality comparisons. |
| Vulkan F32 reader | 33 strict same-shape groups and 12 cross-shape comparisons PASS; maximum absolute logits difference `1.52587890625e-5` | Actual GPU reader execution; qualified for the recorded model, shapes, device, and candidate. |
| Independent GPU F64 oracle | 16 cases PASS in the default Vulkan precision environment; maximum reader logits absolute error `5.1333685324728151e-7` | Real tiny-GGUF execution compared with an independent F64 calculation, including ordinary calibration; this is separate from full-model reader comparisons. |
| Vulkan F32 layout | 76 graphs PASS; maximum absolute error against the independent F64 fixture 0 | Contiguous and strided F32 operands, native precision modes, GQA, serial/batched queries, and the added medium/large tile cases. |
| Vulkan F32 shape and rounding | 141 graphs PASS; 38 bitwise comparisons, including 6 bias graphs | Rounding-sensitive PEDANTIC fixtures across query counts, strides, and query order. These counts precede the later head-only-padding extension. |
| Native sampler equivalence | 7,671 paired comparisons PASS | Compact/full transitions, ties, changing vocabulary, invalid-input recovery, and independent stage RNG streams. |
| C test suites | Count-fixed CPU 5/5 PASS, including real native cancellation; earlier Vulkan candidate 4/4 PASS | The tests actually registered for each build; the Vulkan count does not claim the optional real CPU-abort case ran on GPU. |
| Count-fixed CPU complete whiteboard matrix | 8/8 PASS; all four qualification categories PASS | `whiteboard-cpu-count`: four workers and two pens, `ubatch=128,1 × release,retain × repeat=2`, with strict cross-shape equivalence. Each run recorded 746 committed frontiers and 208 live peer reads. |
| Count-fixed CPU process failure checks | SIGINT PASS: exit 130, recorded time 0.1781924 s; broken pipe PASS: exit 1, recorded time 0.2316818 s | `interrupt-cpu-count`; actual process termination after interruption or output failure. |
| Earlier Vulkan shape-policy complete whiteboard matrix | FAIL on the first `ubatch=1` run after two `ubatch=128` release runs passed | The completed greedy trajectory differed across shapes. The two passing runs each recorded 1,197 frontiers and 208 live peer reads; that full eight-case matrix did not pass. |
| Vulkan F32 head-only-padding extension | Rebuilt candidate PASS: 165 graphs, 62 bitwise comparisons including 6 bias graphs; F64 maximum absolute error `0.00096392687544266664` | Layouts `0/1/2/3/5/6/7`. Native shape/layout pairs are bitwise equal; error against the independent F64 calculation is a separate measure. Log: `f32-head-padding-vulkan-after.log`. |

The original `whiteboard-cpu-strict/receipt.json` remains a FAIL: the observer
reached its 512-token budget without source end. The next default fixture only
clarified ownership of the observer's current task and the literal word
"observed". The successful CPU rerun retained all four workers, the dependency
graph, non-ancestor visibility, phase gates, 512-token budgets, and native EOG
completion. It does not overwrite the failed receipt.

The distinct full-generation Vulkan failure is retained at
`build-rerot-hardening-20261019/whiteboard-vulkan-shape-policy/receipt.json`.
It leaves the reader-state and independent F64 results above intact within
their own tested scope. The dense-plus-expert consistency candidate has passed
the new operator gates and its own complete-generation matrix. That new result
does not change the older failing trajectory or its receipt.

The head-only-padding counterexample is also retained: the previous candidate
failed at layout 5, index 198, with `267.139038` instead of `80.018539`. The
expanded test's passing rebuild is the new result for that boundary. The
count-fixed CPU library rebuilt and passed its reported CPU gates:
8 public A seam tests, 5 C tests, the strict reader comparisons, all 8 complete
episodes in `whiteboard-cpu-count`, and SIGINT/broken-pipe checks in
`interrupt-cpu-count`. These results remain tied to that tested CPU candidate.

Earlier F16, cross-shape, and GPU-oracle failures remain dated records for their
original candidates in [VALIDATION.md](../VALIDATION.md). The new F32 passes
qualify only their executed scope. The independent tiny-GGUF oracle is not a
full 8B-model F64 reference, and none of these correctness gates measures a
throughput gain or establishes multi-device PP, whole-model TP5, MTP, or
LateBind execution.

Final qualification: **PASS for the recorded CPU/Vulkan scope**. The
[evidence index](../audit/rerot-hardening-qualification-20261019.json) identifies
the final source, artifact, and receipt set. The reviewed 34-entry upstream
exception manifest has SHA256
`d52ab562be8ac3fcc331fdc731267fd18341539f980f72a6f80257855b3ae715`.
These completed gates retain their exact model, provider, layout, device, and
measurement scope; they do not promote unexecuted backends or future source
revisions. Historical failures and earlier candidate measurements remain
unchanged.
