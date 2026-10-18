# Composition C

See `../AGENTS.md` for the project axioms and derivation rules.

## AXIOM

1. Users need the selected token and sampler state to match the upstream reference. Reason: a faster row scan must not change later generation.
2. Users need a runnable consumer of the extension. Reason: an unused adapter is not a delivered feature.

## LEMMA

1. [AXIOM 1; parent LEMMA 1, 2] C owns the upstream sampler chain, the choice of full or compact rows, and the conversion of B candidates to A token data. A and B do not know this workflow.
2. [AXIOM 1; parent LEMMA 5] Compact rows are eligible only for the explicit top-k-first greedy chain. Ties use the full row; invalid distributions are rejected before the chain changes.
3. [AXIOM 2; parent LEMMA 1] The command-line consumer uses public llama APIs for model loading, tokenization, decode, and sampling. It must run against the unmodified A library.

## SPEC

1. [LEMMA 1, 2] Compare real A chains with compact sampling on mixed unique/tie rows, then switch to stochastic sampling and compare subsequent tokens. Test clone, reset, invalid-input state preservation, and separate instances. Proof: `tests/test-sampler.cpp`; execution record in `../VALIDATION.md`.
2. [LEMMA 1, 3] Build this module from only its source and installed dependencies, then build a consumer from only installed packages. Proof: the isolation validation recorded in `../VALIDATION.md`.
3. [LEMMA 3] Run `llama-compose` on an actual GGUF in full and compact modes and compare token IDs. Report CPU and GPU coverage separately. Proof: `../VALIDATION.md`.

## Exceptions

None within the implemented sampling workflow. This module does not implement the old fork's shared-attention RERoT, TP5, or device-hidden runtime. See `../MIGRATION.md` for their status.
