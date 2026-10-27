# AXIOM

1. Users need pending prompts to pause ongoing generation and then let it
   resume from the same sequence positions. Reason: the original priority must
   be demonstrated by real model work, not a launcher or a simulated trace.
2. Developers need a replaceable interaction layer between installed model and
   scheduling libraries. Reason: A and B must remain independently usable.
3. Users need truthful memory controls. Reason: a release switch or physical KV
   option is useful only when A actually consumes it and reports failure.
4. A fixed set of resident slots must serve a longer request queue. Reason: request count must not force equivalent GPU residency.
5. A request needs valid incremental text and precise stopping without losing sampled token accounting. Reason: token boundaries can split a character or a stop string.

# LEMMA

1. From AXIOM 1–2: C owns requests, token positions, pending samples and sampler
   lifetimes. It sends value snapshots to B, then submits B's chosen work to A.
   B chooses priority; C does not reimplement that choice.
2. From AXIOM 1: C may advance positions only after A returns success. It must
   keep ongoing request state while other prompts run, consume each sampled
   token as later model input, and release finished sequences after all batch
   outputs have been read.
3. From AXIOM 3: phase changes synchronize A; prompt-to-generation release calls
   the real public seam. Explicit physical capacity is passed unchanged; the
   resulting logical and physical sizes come from A's getters.
4. From AXIOM 4: C keeps request identities separate from A sequence slots. B reserves a complete token bound before C starts work. C consumes every batch output and successfully removes the finished A sequence before it returns the reservation. A failure is fatal and does not trigger an unverified retry.
5. From AXIOM 4 and 5: each admitted request owns a TQText stream. C creates it at admission and releases it at completion, so matching state follows resident capacity. C keeps the sampled stopping token, consumes terminal text, and submits no later token for that request.

# SPEC

1. From LEMMA 1: build uses only installed `llama`, `TQScheduling`, and `TQText` CMake
   packages. Source includes are public A/B headers; neither source tree is an
   implicit dependency. Proof: the archive/build/install/relocation runs in
   `../VALIDATION.md`; source/build copies are removed before the final consumer runs.
2. From LEMMA 1–2: `composition::run`, `consume`, and `sample` own all in-flight
   state. `tests/real-batch.py` validates actual interleaving, resume positions,
   fed-back token ids, and successful results with real A and B.
   Proof: CPU and Vulkan runs passed for three requests, 22 actual batches
   and 145 rows per run; release on/off tokens and input rows matched.
3. From LEMMA 3: `change_phase` checks the release result, and context creation
   uses `n_ctx_kv` plus actual getters. The same integration test compares real
   greedy output tokens and rows with release on/off and rejects actual invalid
   A configurations. Proof: real CPU/Vulkan getters reported logical 512 / physical 1024,
   successful releases returned compute bytes to zero, and the invalid
   configuration and missing-model executions failed explicitly. See
   `../VALIDATION.md` for the complete scope and raw-log locations.
4. From AXIOM 2 and LEMMA 1: each C instance owns one model, context, batch and
   request set; B takes value snapshots and returns owned values. No globals
   store mutable request state and no A/B callbacks cross the boundary.
   Proof scope: source ownership plus B's instance/input/result tests; the
   integration test uses separate processes for on/off. Concurrent C instances
   in one process are not a public contract of this CLI.
5. From LEMMA 5: `tests/test_streaming.py` uses actual A generation to check same-geometry stream on/off runs, cross-delta stops, stopping-token retention, no decode after completion, byte offsets, terminal events, and slot reuse. It observes output while the process runs. B separately proves arbitrary UTF-8 partition and end-of-input behavior. See `../VALIDATION.md` for executed hardware and limits.

# Exceptions

| Item | Consumer | Reason retained | Removal condition |
|---|---|---|---|
| Explicit A seams for compute release/bytes and independent physical KV | This CLI | A owns private graph and KV allocation; C cannot truthfully release it itself | Equivalent upstream public behavior lands, or the feature is removed |

This executable does not claim the stock HTTP server is detached from A. That
program's limited scheduling and memory patches are separately recorded by the
root migration guide. No HTTP, speculative decoding, stochastic RNG resume,
unified-pool auto-fit, or RERoT shared-KV behavior is inferred from this CLI.
