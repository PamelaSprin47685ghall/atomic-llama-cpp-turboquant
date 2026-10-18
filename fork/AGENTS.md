[GUIDE: Apply ASD-STE100 to source code. Use Mise-en-place for design. Start with complete, independent responsibilities and compose them explicitly. Prefer autonomous objects and messages. Adapt Gall's Law, Chesterton's Fence, and Poka-yoke. Use Goodhart's Law and Via Negativa when a target metric hides the real goal. Build the real thing. Try to falsify it.]

# Scope

This file governs the maintained extension project under `fork/`. The repository outside this directory is upstream A, with the small, explicit compatibility changes recorded in `upstream-exceptions.json`. The upstream root instructions still apply to A. Rules for each extension library belong in that library's `AGENTS.md`; do not repeat GUIDE there.

## AXIOM

1. Users need working inference and useful local features. Reason: a small diff that removes the useful behavior is not sufficient.
2. Developers need inexpensive upstream updates. Reason: changes to model, cache, backend, and server internals caused the previous semantic merge burden.
3. Each library must work as a complete unit with only its declared dependencies. Reason: maintaining one responsibility must not require understanding its siblings.
4. Each mutable object owns its state. Reason: external aliases must not let another participant break its invariants.
5. Claims must match observed evidence and its scope. Reason: a microbenchmark does not prove model speed, and a build does not prove runtime correctness.
6. Use the least complexity that satisfies these needs. Reason: abstractions and compatibility paths also need maintenance.

## LEMMA

1. [AXIOM 1, 2, 3] Start A from the exact selected upstream tree. Prefer independent B libraries and explicit C composition. Keep an A change only when a named consumer needs behavior that its public API cannot provide. Derivation: a caller cannot safely release a private allocator or change physical KV capacity by a launch flag alone; a small public seam preserves these useful behaviors without importing the old runtime.
2. [AXIOM 3] A and B do not call, include, link, or receive service callbacks from each other. C may call their public APIs. Derivation: service injection can hide a dependency without removing the responsibility that causes it.
3. [AXIOM 1, 5, 6] Keep a feature only with a concrete consumer, valid behavior, and a maintenance reason. Record exclusions with evidence. Derivation: neither old code volume nor a zero-patch count measures user value.
4. [AXIOM 4] Return owned result values or immutable views with a declared lifetime. Derivation: one writer per state permits independent verification.
5. [AXIOM 1, 5] An unavailable optimization must use the documented reference route or report failure. Derivation: partial execution or changed semantics cannot stand in for successful execution.

## SPEC

1. [LEMMA 1, 2] Every changed path outside `fork/` is recorded with an exact content digest, consumer, reason, and removal condition in `upstream-exceptions.json`. The boundary check rejects unrecorded changes. B builds, tests, and installs from a copy containing only B. C builds from a copy containing only C plus installed A and B packages. Proof: see `VALIDATION.md`; unexecuted cases must remain explicit.
2. [LEMMA 2, 4] B behavior tests require no A services. Tests cover separate instances and input/result ownership. C tests use the real A implementation and B. Proof: see each module's rules and `VALIDATION.md`.
3. [LEMMA 3, 5] `MIGRATION.md` records kept, excluded, and deferred behavior separately. Historical measurements retain their original identity and do not qualify this branch. Proof: source manifest and recorded commands in `VALIDATION.md`.
4. [LEMMA 4, 5] Sampling tests compare selected tokens and subsequent stochastic behavior after compact sampling. Invalid inputs must not mutate sampling state. Proof: `compose/tests/test-sampler.cpp` and `VALIDATION.md`.

## Exceptions

- The 2026-10-18 small-feature audit supersedes the first iteration's zero-A rule. Consumers: phased inference, a physical KV pool separate from the logical window, and the original server's scheduling loop. Reason: the user permits a few A changes and explicitly requires these useful small behaviors. Each seam remains an independent commit and is checked by exact file digests. Removal condition: upstream provides the required public operation or the consumer moves entirely into C. The original server's private loop is not claimed to be an independent B library.
- Upstream A contains its own internal dependencies and rules. Consumers: the original upstream applications and C. Reason: A is an external participant, not rewritten by this migration. Removal condition: an upstream release removes those dependencies; do not copy them into B.
- Public upstream C/C++ APIs are versioned by the pinned commit in `upstream.json`. Consumer: C. Reason: upstream does not promise a stable binary ABI across all commits. Removal condition: a supported stable ABI replaces the pin. Rebuild and rerun C tests when updating A.
- AVX2 is a compiler-specific implementation inside B sampling. Consumer: the numerical scan. Reason: retain the original useful optimization while preserving a scalar path. Removal condition: a portable equivalent has verified performance or this path is no longer useful. Details: `sampling/AGENTS.md`.
- The V1 corpus contract preserves token paths, request multiplicity, kind and timestamp, but has no tokenizer identity. Consumer: prefix replay and old Wanxiangqi corpora. Removal condition: a versioned tokenizer identity and migration contract are implemented. Details: `corpus/AGENTS.md`.
- Two-file corpus saves assume exclusive synchronous access, without crash durability. Consumer: corpus CLI builders. Removal condition: concurrent writers or durable transactions become a demonstrated requirement. Details: `corpus/AGENTS.md`.
- Public sequence snapshots are conditional on model/backend support. Consumer: `llama-corpus`. Reason: C cannot repair private A state; it fails explicitly on unsupported save/restore and offers full replay. Removal condition: the supported upstream API guarantees the required state operation. Details: `corpus-compose/AGENTS.md`.

No hidden source rewrite, link interposition, private-header bridge, or generated patch may be added to claim that A is unchanged. Any needed exception requires a named consumer, evidence, and a removal condition here.
