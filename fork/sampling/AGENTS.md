# AXIOM

1. [User] A numerical shortlist must preserve its stated values and limits. Reason: a fast but changed distribution is not the same feature.
2. [Developer] This library must be usable without any model runtime. Reason: numerical selection has a complete responsibility of its own.

# LEMMA

1. [AXIOM 1] Return the complete min(count, k) shortlist only for a unique finite maximum; return an empty unusable result outside the contract. Derivation: C can then decide whether the documented fast route applies without observing partial work.
2. [AXIOM 2; parent LEMMA 2, 4] Accept values and return owned values. Do not include, link, or receive callbacks from llama or ggml. Derivation: B can respond independently and cannot mutate another participant's state.
3. [AXIOM 1, 2] Keep a scalar implementation and check CPU support before AVX2. Derivation: the public numerical behavior must not require a specific CPU.

# SPEC

1. [LEMMA 1, 3] Compare the shortlist with full sorting across ties, masks, invalid values, vector tails and k limits. Proof: tests/shortlist-test.cpp, including explicit Release-safe checks.
2. [LEMMA 2] Input mutation or destruction and result mutation do not affect another result. Proof: the ownership cases in tests/shortlist-test.cpp.
3. [LEMMA 2] Build, test and install this source directory without A or C. Consume its relocated installed package. Proof: ../tests/isolation.py and ../VALIDATION.md.
4. [LEMMA 1] C must run its real sampler chain and retain random state; B does not claim that a shortlist alone proves full sampling equivalence. Proof: ../compose/tests/test-sampler.cpp and ../tests/real-model.py.

# Exceptions

- AVX2 uses compiler intrinsics on supported x86-64 GCC/Clang builds. Consumer: the numerical scan. Reason: retain the measured original scan optimization. Removal condition: a standard portable implementation has verified equivalent performance, or this path ceases to be useful. The scalar implementation remains independently testable.
