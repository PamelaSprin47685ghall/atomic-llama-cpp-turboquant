# AXIOM

1. Generated token pieces can split UTF-8 characters and stop strings. Users need valid text without leaking a stop prefix that later completes.
2. One mutable object must own one text stream. A tokenizer, sampler, request queue, output sink, or sibling library must not be required to maintain its invariants.
3. Delivery boundaries must not change the selected stop or final text. Input and result ownership must remain explicit.

# LEMMA

1. From AXIOM 1 and 3: keep only undecided bytes, repair invalid bytes consistently, and choose stops by first completion position with a documented tie rule.
2. From AXIOM 2: receive byte messages and return owned deltas; keep no caller pointers, service callbacks, token IDs, or I/O handles.
3. From AXIOM 2 and 3: make termination explicit, keep byte offsets monotonic, and make repeated completion unable to duplicate output.

# SPEC

1. From LEMMA 1: `generation_stream` preserves valid UTF-8 and replaces each invalid byte with U+FFFD. The first completed stop wins; simultaneous completions choose the longest string, then registration order. Proof: complete-input oracle, exhaustive partitions, and generated adversarial cases in `tests/stream-test.cpp`.
2. From LEMMA 1 and 2: stop count and byte limits are checked before copying. Prefix tables and a fixed ring hold bounded matching state without emitted history. Output allocation and byte capacity are checked before mutation. Proof: constructor boundary and long-prefix tests, plus the allocation-before-mutation path in `src/stream.cpp`.
3. From LEMMA 2 and 3: separate objects are independent; input and result edits cannot change state. Moving transfers ownership and leaves the source terminal. Proof: interleaving, ownership, move, and terminal-operation tests.
4. From AXIOM 2: the installed header, static library, and CMake package require only C++17 and the standard library. Model behavior, JSON framing, token accounting, and output errors belong to the C consumer and need separate validation.

# Exceptions

None. A caller serializes operations on one stream. No asynchronous, shared-instance, callback, persistent-state, network-delivery, or model-lifetime contract is implied.
