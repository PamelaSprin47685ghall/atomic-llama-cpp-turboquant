# AXIOM

1. [User] Recorded token requests must remain usable without model-internal hooks. Reason: replay is useful even when the model backend changes.
2. [Developer] The composer owns traversal, not trie invariants or model state interpretation. Reason: each participant must remain independently usable.

# LEMMA

1. [AXIOM 1, 2; parent LEMMA 2, 3] C asks B for owned paths and child lists, and asks A to decode tokens or save and restore its own sequence state. Inference: no private model or trie fields are needed.
2. [AXIOM 1] Prefix reuse must preserve request-weighted log loss. Inference: compare with independent full request replay through the same real model.

# SPEC

1. [LEMMA 1] Build with only this source directory and installed llama and TQCorpus packages. Proof: ../tests/isolation.py and ../VALIDATION.md.
2. [LEMMA 1] Opaque sequence snapshots and traversal frames belong to C. No callback is installed into A or B. Proof: src/main.cpp.
3. [LEMMA 2] Real-model validation compares decoded-token counts and request-weighted log loss with full replay, and checks invalid context limits. Proof: ../tests/real-model.py and ../VALIDATION.md.

# Exceptions

1. V1 trie files have no tokenizer fingerprint. Consumer: old recorded corpora. A vocabulary size only bounds token IDs; it does not prove tokenizer identity. Remove this exception when a versioned fingerprint format and migration tool replace V1.
2. The public llama sequence-state API may not support every architecture or configuration. Consumer: prefix replay. Fail on an unsupported snapshot; use explicit full replay. Remove only when the upstream contract guarantees this operation for the relevant model.
