# AXIOM

1. Users need the old fork's Prefill priority to survive upstream updates.
   Reason: a pending prompt must pause generation without importing the old
   server, kernels, or experimental reasoning system.
2. Developers must build and use the selector without a model engine.
   Reason: scheduling policy has a complete value contract of its own.
3. Consumers need fixed token and resident-slot budgets. Reason: a waiting request must not begin model work before its full known demand is reserved.

# LEMMA

1. From AXIOM 1: keep the active prompt until completion; then scan caller slot
   order. Generation is eligible only when no prompt remains. Adding FIFO or a
   fairness timer would change the requested behavior.
2. From AXIOM 2: receive ids and stages, return owned ids, and keep no callbacks
   or references to a model, composition object, input, or result.
3. From AXIOM 2-3: admission owns accounting and issues immutable reservation identities. C owns model memory and releases it before returning a reservation. Admission and selection are independent objects.

# SPEC

1. From LEMMA 1: `prefill_selector::choose` implements sticky prompt selection,
   slot order, generation resumption, and the possible starvation of generation.
   Proof: `tests/prefill-test.cpp` covers arrivals, removed/completed prompts,
   unchanged generation order, empty input, and repeated prompt arrivals.
2. From LEMMA 2: two selectors do not share state; external input/result edits
   cannot change a selector. Invalid snapshots do not commit state changes.
   Proof: ownership, instance, duplicate-id, and invalid-stage tests in that file.
3. From AXIOM 2: only C++17 and the standard library are required. The installed
   header/archive/CMake package suffice for an unrelated consumer.
   Proof: independent configure/build/test/install and a relocated installed
   consumer passed; see `../VALIDATION.md`.
   Runtime model integration is a separate consumer's proof, not this library's.
4. From LEMMA 3: stale or foreign reservations cannot release a new slot occupant. Invalid requests and capacity rejection preserve accounting. Proof: `tests/admission-test.cpp` covers separate owners, reuse, moved-from values, duplicate release, exact capacity, and arithmetic at UINT64_MAX.

# Exceptions

None. There is no asynchronous, shared-instance, or callback contract. A caller
must serialize operations on one selector; separate selectors are independent.
