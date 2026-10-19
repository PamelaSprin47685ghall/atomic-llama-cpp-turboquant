# TQRerotCompose

TQRerotCompose is the C layer for token-frontier shared-whiteboard generation.
It uses one installed A model/context, the independent `TQRerot` state library,
`TQRelay` submission accounting, and `TQText` output assembly. It has no private
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
  -DCMAKE_PREFIX_PATH='/installed/A;/installed/TQRerot;/installed/TQRelay;/installed/TQText'
cmake --build build-rerot-compose --parallel 4
ctest --test-dir build-rerot-compose --output-on-failure
```

The installed A must include the reviewed reader-view and recurrent-state
interfaces declared in its public `llama.h`. The context parameter struct has
changed. Rebuild consumers against that installed header and library together.

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

The default is CPU execution (`--gpu-layers 0`). Device names are the public A
backend names. `--devices` accepts a comma-separated list. `none`, `layer`, and
`row` are passed to A's model placement policy. A owns the actual devices,
transfers, and scheduler; C does not invent per-device completion receipts.

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

This is a correctness reference route. It is not a claim of a specialized
RERoT kernel speedup. Ordinary quantized linear dispatch remains available to A,
but an MMVQ performance claim needs an observed kernel hit and a measured
comparison. Small ubatches can also reduce the rows that reach those kernels.

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

`tests/test_cli.py` checks the actual executable's help and refusal paths
without loading a model. `tests/real-whiteboard.py` requires a real model and
retains commands, raw stdout/stderr, elapsed time, binary/model hashes, and
frontier validation results:

```sh
python3 fork/rerot-compose/tests/real-whiteboard.py \
  --binary build-rerot/install/bin/tq-rerot-compose \
  --model /models/instruct.gguf \
  --output build-rerot-whiteboard-proof \
  --ubatch-sizes 128,1 --phase-modes release,retain
```

The real harness requires repeated visibility growth while a peer is still
running, exact peer key identities, correct pen lifetime, dependency admission,
private source-end exclusion, and a complete synthesis. Protocol validation
alone does not prove attention numerics. The public A seam tests separately
compare full-vocabulary logits and an independent small-model attention oracle.

The independent B passed all four CTest groups in Release, UBSan, and ASan.
The final physical-score/native-KQ/native-V helper passed the 16-case CPU oracle:
legal single-layer GGUF fixtures with zero FFN weights exercise Llama NORMAL,
Qwen2 NEOX, GQA, RoPE amplitude, query permutation, and serial execution of the
same frozen cohort against independent F64 forward/logits calculations. The
earlier eight-case receipt keeps its original helper identity. No A
partial-rotary or GPTNeoX qualification is implied.

Actual LFM2.5 F32 reader state testing passed all 33 strict groups: 31
full-logits equivalence groups and two peer causal-effect groups. Across the
equivalence groups, the maximum absolute logits difference was 0 on CPU and
`0.0000133514404296875` on Vulkan, including the added same-shape
reversed-serial and visible-serial comparisons. The final grouping and receipts
are recorded in [VALIDATION.md](../VALIDATION.md). These are scoped numerical
and state results. Complete C generation, installation, signal/output failure,
and multi-device PP qualification each require their own run receipts; local
oracle success does not supply them. See [VALIDATION.md](../VALIDATION.md) for
the exact qualified build, hardware, commands, and remaining limitations. The
GPU independent F64 oracle still fails its ordinary-calibration gate; the
Vulkan state comparisons do not grant independent GPU mathematical qualification.
