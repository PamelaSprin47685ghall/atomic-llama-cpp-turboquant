# Alloc Ledger

# AXIOM

1. [User] An allocation ledger must report the same slot, generation, and lifetime facts that it enforces. Reason: a plan built on different facts than the allocator is not a plan.
2. [Developer] This library must work without a model runtime, a GPU, or a placement policy. Reason: slot, lease, window, and lifetime bookkeeping is a complete responsibility of its own.

# LEMMA

1. [AXIOM 1] Arena storage is caller-owned. Arena never calls malloc or free and never places data inside an extent. Derivation: the caller owns the physical buffer and can free it independently.
2. [AXIOM 2] Placement stays in ggml_gallocr. This library only decides slot order, lease identity, and reuse permission. Derivation: a second placement policy would duplicate the real allocator.
3. [AXIOM 1] A lease is a value capability. Every state change revalidates arena id, window, arena generation, slot generation, extent, lifetime, and reuse class. Derivation: an old or forged lease cannot change state.
4. [AXIOM 2] Plan cache and domain expose plain values. No callback, I/O, or llama/HIP type crosses the boundary. Derivation: tests and callers can own results.

# SPEC

1. [LEMMA 1] Slot reuse requires the new begin to be at or after the previous lease end. Live and quarantined extents cannot overlap a new extent. Proof: tests/arena-test.cpp.
2. [LEMMA 3] A repeated, stale, cross-window, or changed lease is rejected without a state change. Proof: tests/arena-test.cpp and tests/domain-test.cpp.
3. [LEMMA 1, 2] Frame and staging slots wait for drain. Cold slots return at once. A fault quarantines the slot and adds one cold reserve unit. Proof: tests/domain-test.cpp.
4. [LEMMA 4] A plan key has seven fields. Footprint is an entry value, not a key field. The cache keeps eight entries, evicts the least recently used unpinned entry, and never inserts the same key twice. Proof: tests/plan-cache-test.cpp.
5. [LEMMA 4] `LeaseEvent` and `PlanEvent` are in-memory values. They are trivially copyable and do no I/O. They are not external schema records: WS0 owns `alloc.schema.json`, `plan.schema.json`, and `leg.schema.json`; the C adapter writes the JSON. B defines no `LegManifest`. Proof: tests/events-test.cpp and README.md.
6. [AXIOM 1] Live bytes, high water, leases, releases, and quarantines are tracked per lifetime and in total. Proof: tests/arena-test.cpp and tests/domain-test.cpp.
7. [LEMMA 4] `PlanKey::describe()` and `PlanKey::hash()` give the C adapter a stable external `plan_key` string. The PlanKind to WS0 action mapping lives in README.md. Proof: tests/plan-cache-test.cpp.
