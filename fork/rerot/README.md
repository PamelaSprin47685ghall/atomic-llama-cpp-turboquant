# TQRerot

TQRerot owns the logical part of a shared KV whiteboard. It is a standalone
C++17 library with no dependency on llama.cpp, GGML, Relay, or a device SDK.

This is the RERoT **B** library. A real application **C** must execute model
work in **A**, keep one physical KV owner, and consume these reader views in
real attention. This library does not implement a model or claim device
qualification from its unit tests.

## Build and use

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /path/to/prefix
```

A consumer uses the installed package:

```cmake
find_package(TQRerot CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE TQRerot::rerot)
```

Public headers are `tq/rerot/plan.h`, `tq/rerot/episode.h`, and
`tq/rerot/view.h`. They use only the C++ standard library.

`TQ_REROT_BUILD_TESTS` defaults to CMake's `BUILD_TESTING` setting. An
explicit `-DTQ_REROT_BUILD_TESTS=ON` or `OFF` overrides that default. The
installed package includes this README and the MIT license.

The independent installed consumer uses only `find_package()` and public
headers. After installation, build and run it against the prefix:

```sh
cmake -S tests/installed-consumer -B build-installed-consumer \
  -DCMAKE_PREFIX_PATH=/path/to/prefix
cmake --build build-installed-consumer
ctest --test-dir build-installed-consumer --output-on-failure
```

It checks two-worker token publication and next-frontier visibility through
the installed library. It does not execute model or device work.

## Identities and ownership

`Plan` owns worker names, intents, dependencies, and input order. Worker IDs
are 1 through N, planner ID is 0, and synthesis ID is N + 1. A physical pen
is a temporary binding. It does not select a recurrent seed or an RNG stream.

`Episode` is a single-writer object. C must serialize calls to it. A `Step`
ticket is tied to its Episode instance. Moving the Episode preserves this
identity. A moved-from Episode supports destruction and move assignment.

C owns token data, model configuration, one physical KV owner, C_base,
stage-local recurrent/conv state, complete sampler state, template events,
device resources, and all actual work. C must derive each new stage from
C_base, not from whichever pen became free.

The prefix passed to `Episode` describes already computed C_base KV. Run IDs
are nonzero and unique across the episode. Writer storage ranges are
nonnegative half-open intervals. New stages start at or after the prefix's
storage end. A run can grow only at its owner's tape tail, with unchanged
kind, visibility, and contiguous writer positions.

## One token, one publication boundary

1. `begin_step()` retains all running bindings. It fills free pens from the
   eligible queue in plan order. A worker keeps its pen until native source
   end. Unbound workers hold no model state in this library.
2. C supplies the complete FRAME for a new stage through `prepare()`. A
   running stage supplies exactly one input token. C still owns the bytes.
3. `prepare()` returns logical `WriteSpan` ranges. C binds them to its actual
   KV rows. It obtains one `ReadView` for each input query row.
4. C executes all members against the frozen peer world. A row may read its
   own current key and causal history, including private or pending input.
   It cannot read another member's new key from this step.
5. C calls `complete()` only after that member's real operation succeeds.
   `commit()` requires every member, then publishes all increments together.
6. The next `begin_step()` sees each newly committed public token. There is
   no extra lag, no whole-BODY delay, and no wait for peer task completion.

The member input token is not the token merely sampled from its output
logits. C must not publish a sampled token before A has computed its KV.

Within a step, changing member preparation or completion order does not
change the published whiteboard. Changing the configured pen capacity can
change when queued workers start and what completed peer work they see.
This is the documented P-bound scheduling rule.

`source_end` must be a private segment with explicit current-source
provenance. Only a successful worker source-end commit seals a worker and
unlocks its successors. Source-end KV remains in the source state but does
not appear in a foreign view. Runtime FRAME tokens cannot seal a worker.
After all workers seal, a new synthesis stage starts from C_base. Its source
end finishes this library's episode; C owns the subsequent output transition.

The first version requires a template whose source end can be represented
without splitting one model token into distinct body and end KV rows. C must
reject an unsupported boundary instead of guessing from arbitrary text.

## Views and attention

A `ReadView` owns all visible prefix lengths. Its read revision and spans do
not change after another member completes or the episode commits. The view
is independent of the Episode's lifetime.

The order is a cycle-preferred topological order: dependencies first, with
the active worker last. Public FRAME/BODY from every started peer remains
eligible, including sealed peers that are not hard ancestors. The synthesis
view uses plan order subject to dependencies. Private/control/source-end
rows remain local to their stage. DAG edges are start barriers, not
information isolation.

Each span contains a logical run, stage, token offset, count, writer storage
position, and reader virtual position. It contains no physical cell index.
C/A binds `(run, offset)` to a retained physical allocation generation. A
physical relocation must not change logical order or phase.

For writer storage position s, reader position v, and query position q:

```text
stored K = R(s) K_raw
effective Q = R(q + s - v) Q_raw
```

All visible spans enter **one global softmax per query/head**. Readers share
KV data, not softmax statistics. `expand(view, max_entries)` produces checked
per-key logical entries and effective positions for a straightforward C/A
consumer. An application can consume spans directly with the same contract.

Turbo storage needs its actual coordinate transform. Rotate Q before the
storage-domain WHT. For a quantized oracle, use the same dequantized stored
K; quantizing raw K again at another RoPE position changes the problem.

## Failure and resource limits

Invalid input throws `std::invalid_argument` or `std::out_of_range`.
Inconsistent lifecycle calls throw `std::logic_error`. Numeric exhaustion
throws `std::overflow_error`; explicit resource limits throw
`std::length_error`. These checks remain enabled in release builds.

Argument or allocation failure leaves the affected operation unpublished.
If actual A work fails, C calls `abort()`. The episode then accepts no more
views or work and publishes no pending token. C must drain all actual
submitted work before `confirm_drained()`. That acknowledgement is not a
device synchronization primitive. Aborted episodes cannot resume.

This version requires all referenced KV to remain resident. It has no
persistent restore, KV eviction, context shift, compressed factor store,
automatic routing probe, HTTP server, or device implementation. The library
does not restore a sampler from a seed or pretend to roll back A by clearing
logical metadata.

## MTP, MMVQ, and LateBind

The view stamp binds the episode, stage, frontier, and frozen read revision.
C must add its own model/state/allocation identities to execution records.
It must invalidate prepared numerical data when visible prefixes change,
even if the structural DAG is unchanged.

With active peers, a draft cannot be accepted across several whiteboard
frontiers using one old view. C must publish and revalidate at each frontier,
or supply an independently verified dependency-aware joint procedure. This
library enforces the one-token running-step boundary.

MMVQ row packing and LateBind buffer reuse must preserve logical stage and
read identity. Padding is not a live member. An implementation flag is not
evidence that a real model executed a particular numerical path.

## Tests

The tests cover all four-worker DAGs and legal reader states, publication
and abort counterexamples, own/foreign visibility, P-bound scheduling,
numeric bounds, and an independent F64 attention oracle. The oracle uses
stored quantized keys, explicit key rephasing, partial rotary dimensions,
both rotary pair layouts, GQA, and an orthogonal storage transform. It also
rejects a per-span-softmax substitute.

These tests establish the B contracts. Real A execution, templates,
recurrent/sampler state, single-card/PP/TP5 execution, and actual performance
need their own evidence in the C application.
