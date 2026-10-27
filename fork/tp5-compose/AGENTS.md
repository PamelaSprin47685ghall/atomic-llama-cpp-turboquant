# TQTP5Compose

## AXIOM

1. Users need real five-rank model inference, not a collective demo. Reason: kernel flags and device utilization alone do not establish a usable model.
2. Failure must not release in-flight native owners. Reason: a cancelled result does not stop DMA or state writes.
3. Numerical, execution and per-request timing claims need separate evidence. Reason: a successful platform probe cannot qualify a model.

## LEMMA

1. [AXIOM 1] C composes A's public opt-in execution capability, TQRelay's ownership messages, TQScheduling's reservations, TQText's output stream and TQSampling's finite-greedy selection. Derivation: A owns model/native semantics; B participants remain independently usable without each other.
2. [AXIOM 1, 2] Each mutable context has one model worker and independent inputs/state. A separate CPU service progresses every active window before and throughout device submission and drain. Derivation: one owner prevents input aliasing; independent service prevents GPU/CPU completion cycles.
3. [AXIOM 2] An unknown native completion requires retention, never fallback or destructive cleanup. Derivation: logical failure does not prove native accesses ended.

## SPEC

1. [LEMMA 1, 2] The CLI loads one real Qwen3-8B Q8_0 model across exactly five HIP ranks, runs independent requests through one or two isolated windows, computes CPU sums in rank order, and samples complete finite logits. Proof: `tests/real-tp5.py` exercises installed executable and records actual masks/counters, full logits, row identity and final reservations. Only executed records grant qualification.
2. [LEMMA 2, 3] CPU input retirement never grants output retirement; output is reusable only after all actual last readers. Failure stops new publication; unproved drain terminates the CLI GPU domain without freeing native owners. Proof: runtime's ledger transitions, explicit abort/delay cases and provider native receipts; see `../VALIDATION.md` for actual scope.
3. [LEMMA 1] This package builds from installed A/B artifacts and includes no private A header. Proof: standalone installed-consumer acceptance, recorded after execution.

## Exceptions

A's opt-in Meta/HIP relay interface is a maintained extension, registered in `../upstream-exceptions.json`. Consumer: this package. Removal: upstream supplies equivalent prepared five-rank execution, stable CPU payload views, publication, error and native-drain facts.
