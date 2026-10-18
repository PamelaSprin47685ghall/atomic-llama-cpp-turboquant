# AXIOM

1. [User] Recorded token requests must retain their paths, order and multiplicity. Reason: changing them changes the scoring workload.
2. [Developer] The trie must own its invariants and be usable without a model runtime. Reason: request storage and model execution are separate responsibilities.

# LEMMA

1. [AXIOM 1, 2] B owns node IDs, parent ordering, token range, request depths and serialization. Derivation: consumers must not repair B's internal state.
2. [AXIOM 2; parent LEMMA 2, 4] Public queries return owned values; no llama or ggml dependency or service callback is allowed. Derivation: B can answer direct requests independently.
3. [AXIOM 1] A failed append or an attempted save to an existing target must not change recorded requests. Derivation: failures cannot silently replace a user's workload.

# SPEC

1. [LEMMA 1] Use explicit little-endian conversion for CLTNOD01/CLTREQ01 version 1; reject malformed records, forward/cyclic parents, invalid tokens, leaves and depths. Proof: src/trie.cpp and tests/trie_test.cpp.
2. [LEMMA 2, 3] Test separate instances, copies, input/result ownership, append rejection and existing-directory protection. Proof: tests/trie_test.cpp; checks remain active in Release.
3. [LEMMA 2] Build, test and install this source alone, then consume its relocated package. Proof: ../tests/isolation.py and ../VALIDATION.md.
4. [LEMMA 1, 2] The actual C consumer uses A to tokenize and replay requests, including repeated and prefix requests. Proof: ../tests/real-model.py and ../VALIDATION.md.

# Exceptions

- V1 contains no tokenizer fingerprint. Consumers: existing Wanxiangqi token corpora. Vocabulary size checks token bounds only. Remove when a versioned identity format and migration tool replace V1.
- V1 reserved capture metadata is not part of this library's token-path contract. Consumer: prefix replay. Load ignores it and save writes zero; it is not a byte-for-byte capture archive. Remove when a real metadata consumer requires a specified ownership and round-trip contract.
- Saving two files is synchronous and assumes exclusive access to the new target directory. Consumers: command-line corpus builders. Existing directories are rejected and partial writes are cleaned up; process crashes are not a durable transaction. Remove when concurrent writers or crash durability become a real requirement.
