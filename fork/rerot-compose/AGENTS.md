# AXIOM

1. Each active insertion point must read its peers' last committed token on
   its next forward step. Reason: RERoT is a shared whiteboard, not independent
   completions that are concatenated after they finish.
2. The native model owns its cache, graph, and recurrent state. Reason: C must
   not duplicate A's model implementation or copy a cache per reader.
3. A failed forward must not publish part of a whiteboard update. Reason:
   device execution can change state before a failure is reported.
4. Users need a complete installed program with explicit capabilities.
   Reason: a protocol test cannot replace useful model execution.

# LEMMA

1. From AXIOM 1 and 2: C calls the independent RERoT plan and episode owner,
   then passes owned reader views to A's public attention interface. A and B
   do not call each other.
2. From AXIOM 1 and 3: one body step advances each bound stage by one token.
   C completes every actual A submission, checks all outputs, then asks B to
   commit. Partial execution makes the episode fail-stop.
3. From AXIOM 2 and 3: C owns the mapping of logical stages to A sequence
   handles, stage-local samplers, output cursors, and native completion facts.
   TQRelay owns submission accounting; it is not proof of GPU completion.
4. From AXIOM 4: the command line accepts an explicit task graph and uses the
   model's native chat framing. Missing execution capabilities fail before
   inference. Five simulated receipts must never stand in for TP5 execution.

# SPEC

1. From LEMMA 1: this directory builds against installed public llama,
   TQRerot, TQRelay, TQText, and TQSampling packages. No private A headers or sibling source
   directories are build inputs. Proof is recorded in ../VALIDATION.md after
   the isolated install and relocated-consumer runs.
2. From LEMMA 2: real model tests check incremental visibility, cohort row
   mapping, source-end retirement, native state isolation, and failures.
   Each device and optimization result retains its actual execution scope.
3. From LEMMA 3: a submission is registered before llama_decode can have side
   effects. Its completion is recorded only after llama_synchronize. Failure
   prevents later whiteboard publication or state reuse.
4. From LEMMA 4: tests reject unsupported modes explicitly, and the installed
   help and README state the same capabilities as the executable.
5. From LEMMA 3: compact sampling must preserve the native token order and RNG
   sequence. A tied greedy maximum, a stochastic retained/cutoff tie, or an
   unsupported shortlist size uses the original full row. Non-finite rows fail
   before the sampler changes. Proof: `tests/test-sampling.cpp` compares the
   actual A chains and B selection through mixed compact and fallback calls.

# Exceptions

- A's public reader-view seam is pinned with the llama package. Consumer:
  this executable. Reason: the original sequence API cannot express different
  rotary positions for readers of the same physical keys. Removal condition:
  the selected upstream provides equivalent public behavior.
