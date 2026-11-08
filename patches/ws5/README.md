# WS5 — pillar 5: graph / plan / frame persistence

Branch `feat/tp5-mtp-ws5-plan-frame`, worktree `/home/kunweiz/work/r28b-ws5-impl`.
Base `71544695b`, F2 merge `2a29eba88` (`feat/tp5-mtp-ws2-alloc-ledger` @ `ae9c13065`, read-only).

## OFF is stock

With every switch unset or `0`, the branch is behavior-identical to stock:

- two captured frames (the old linear 2-frame cache),
- no plan cache, monotonic `plan_generation` on every prepare,
- the fixed 384.25 MiB channel pool (`256 * 1573888 B`),
- the original `map<(plan,channel), channel_state>` runtime ownership,
- every graph `uid` stays zero, so the relay takes the full `describe_frame`
  signature path.

`GGML_TP5_GRAPH_UID` is **off by default**. The ON leg must set it to `1`
explicitly. This keeps the F2/OFF leg equivalent to stock.

## Switches

| Switch | Default | ON leg | Meaning |
|---|---|---|---|
| `GGML_TP5_PLAN_UNION` | `0` | `1` | K=8 plan cache + reuse of a plan generation |
| `GGML_TP5_RELAY_FRAMES` | unset/`0` → 2 | `8` | frame capacity 2..8; `4` is the structural tier; illegal values fail `create` |
| `GGML_TP5_POOL_EXACT` | `0` → 384 MiB | `1` | exact per-plan layout, 8 MiB baseline, 68 MiB budget, fail closed |
| `GGML_TP5_RELAY_SLOT_CHANNELS` | follows `PLAN_UNION` | `1` | 8 plan slots x 256 channel states per window |
| `GGML_TP5_GRAPH_UID` | `0` | `1` | llama assigns a graph uid after each build |
| `GGML_TP5_RELAY_VERIFY_UID` | `0` | `1` | uid fast path still computes and asserts the signature |
| `GGML_TP5_RELAY_PLAN_STATS` | `0` | `1` | print plan/frame/pool counters |
| `GGML_TP5_RELAY_PLAN_FLUSH` | `0` | `1` | diagnostic: flush every plan on each prepare |
| `GGML_TP5_LLAMA_PLAN_CACHE` | `0` | `1..8` | llama-side per-shape `llm_graph_result`+scheduler cache; the value is the slot count; relay-only |
| `GGML_TP5_RELAY_OUTPUT_ROWS` | `0` | first catch-up rows | relay-only host output preallocation bound; 0 keeps `max(n_outputs_max, n_seq_max)` |

Recommended ON configuration for the five-shape MTP working set:

```
GGML_TP5_PLAN_UNION=1
GGML_TP5_RELAY_FRAMES=8
GGML_TP5_POOL_EXACT=1
GGML_TP5_RELAY_SLOT_CHANNELS=1
GGML_TP5_GRAPH_UID=1
```

`GGML_TP5_PLAN_UNION=1` together with `GGML_TP5_RELAY_SLOT_CHANNELS=0` is an
illegal matrix entry; the runtime patch fails closed on it.

## llama-side per-shape cache

`GGML_TP5_LLAMA_PLAN_CACHE=<slots>` (default 0, relay-only) keeps up to
`slots` `(llm_graph_result, ggml_backend_sched)` pairs keyed by
`(n_tokens, n_seq_tokens, n_outputs, n_seqs, n_seqs_unq, equal_seqs,
n_kv_target, gtype, alloc_gen)`. A hit skips both `model.build_graph` and the
scheduler's reset/placement; `can_reuse()` is still the second-level check, and
a key collision rebuilds in the same slot (`rekey`). A new `alloc_gen`
(scheduler recreation or phase switch) flushes the cache. The cache is per
context, so two windows never share a slot.

Cost: every slot owns a scheduler, and each scheduler allocates its own compute
buffers. Slot count trades host/device memory for build reuse; the five-shape
relay working set needs at least five slots to avoid LRU thrash. The flag is
off by default, so the stock and relay-OFF legs keep the single scheduler.

## First catch-up output rows

The relay constructor preallocates the host output buffer for
`max(cparams.n_outputs_max, n_seq_max())`. A first catch-up batch larger than
that (the ALLf run used 13 rows with `n_outputs_max = 4`) still grows once.
`GGML_TP5_RELAY_OUTPUT_ROWS` lets the owner name that bound before context
creation; the constructor then reserves it and the first decode does not grow.
The preallocation is allowed above `cparams.n_outputs_max`; every decode call
keeps the contract assert. A late `set_embeddings_nextn` or sampler change can
still grow once; that is the documented exception, and the old buffer is
retired rather than freed (`buf_output_retired`), so no `hipHostFree` runs
while kernels may be launching.

## Plan/frame counters

`TP5_PLAN_STATS destroy ...` is logged **before** the teardown flush, so
`live` and `frames` are the steady-state values. The frame fields are
`frame_hits`, `frame_misses`, `frame_evict`, `frame_inst`, `frame_destroy`.
A nonzero `frame_inst` proves the capture path ran; a nonzero `frame_hits`
proves the LRU served a repeat. `frames_cleared` counts frames dropped by an
exact-pool layout growth, not by the destroy flush.

## Plan identity: coarse llama key + terminal binding digest

The seven llama fields plus `channels` are coarse: a graph rebuild can keep
all of them identical while the terminal tensors move or change layout. The
relay therefore folds a digest of every terminal binding (`tensor` and `data`
pointers, `ne`, `nb`, for all channels and ranks) into its internal plan key.
"Same llama key, different terminal binding" is then a cache miss with its
own slot.

A key hit must not be compared against the provider's *current* channels:
after a tail switch between the 180- and 181-channel plans, the current
channels describe the other plan, so `same_plan` is false even though the
cached plan is the right one. A cached key hit reuses that plan's generation
and rebinds the channels; a genuinely different binding has a different
digest and is a miss.

The digest is only a fast filter, so every plan slot also stores an exact
terminal binding snapshot: terminal and data pointers plus `ne`/`nb` for every
channel and rank (819 KB per provider for 8 slots). A hit compares that
snapshot in full before reusing the generation. A 64-bit digest collision that
names a different real binding therefore fails closed with
`relay plan key collided with a different terminal binding`, not a silent
reuse. The snapshot is written on insert and overwritten when the slot is
reused; the explicit key->channels vs prepared-count contradiction remains a
key-consistency failure. The digest and snapshot are used only when the plan
cache is enabled and a key is present, so flag 0 keeps the legacy path.

The provider-create line ends with `key_digest=1` when the build contains this
fix. That marker is build freshness only, not a correctness proof: the
correctness evidence is the full-snapshot comparison in the code and the CPU
tests below. A rerun log without the marker used an older installed
`libggml-hip.so`.

## Two plan-key domains (not the same type)

| Type | Fields | Owner | Purpose |
|---|---|---|---|
| `tq::alloc::PlanKey` | 7: model, window, n_tokens, n_outputs, rs_class, n_kv_bucket, alloc_gen | `fork/alloc-ledger/include/tq/alloc/plan_cache.h` (F2) | allocation-ledger identity and arena placement |
| `ggml_relay_plan::plan_key` | the same 7 + `channels` = 8 | `ggml/include/ggml-relay-plan.h` (WS5) | relay graph plan identity and frame/generation reuse |

They are deliberately separate. The relay adds `channels` because one
allocation plan can back relay graphs with different channel tables (180 vs
181). The C boundary type `struct ggml_relay_plan_key` (in `ggml-relay.h`) maps
the relay domain only. `ggml_relay_plan::plan_key::from_c`/`to_c` are the
adapter; no F2 file is modified.

## Cross-owner patches

Apply from the worktree root, in any order:

```
git apply patches/ws5/ggml-backend-meta.cpp.patch   # Meta: key storage, prepare_ex, setter, snapshot
git apply patches/ws5/runtime.cpp.patch             # compose runtime: channel slots + PLAN_RETIRED
git apply patches/ws5/ggml-backend.cpp.patch        # scheduler: preserve a caller-assigned uid (proposal)
```

The Meta patch is required for a complete link: `libllama.so` references
`ggml_backend_meta_set_plan_key`. Without it the shared library still links
(unresolved symbols are tolerated), but the first relay decode cannot resolve
the symbol.

## Build and test

```
cmake -S . -B build-ws5-20261107 -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_HIP=OFF
cmake --build build-ws5-20261107 -j44 --target llama test-relay-plan-policy test-shape-cache
./build-ws5-20261107/bin/test-relay-plan-policy
./build-ws5-20261107/bin/test-shape-cache
```

HIP compile of the provider (no device needed):

```
hipcc -x hip -std=gnu++17 --offload-arch=gfx1030 -O1 -c \
  -DGGML_USE_HIP -DGGML_BACKEND_BUILD -DGGML_SHARED -DGGML_HIP_NO_MMQ_MFMA \
  -DGGML_HIP_NO_VMM -DGGML_SCHED_MAX_COPIES=4 -D__HIP_PLATFORM_AMD__=1 \
  -D__HIP_ROCclr__=1 -DUSE_PROF_API=1 \
  -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
  ggml/src/ggml-cuda/relay.hip -o /tmp/relay.o
```

The policy test pins: plan-cache hit/evict/pin/flush and channels-in-key; the
five-shape frame LRU (cap 8 = five cold misses, cap 4 worst case 9/12); the T4
pool layout (`71,209,984 B <= 68 MiB`) and the fail-closed 65536 bound; channel
slot binding/retirement/dual-window isolation and the 1448/2896 bounds; and the
legacy switch defaults.

`test-shape-cache` pins the llama-side cache policy: same-shape second hit,
per-field key identity, no reuse across `alloc_gen`, flush, LRU eviction,
rekey after a `can_reuse` failure, dual-window isolation and the flag/capacity
parsing (0 = off, 1..8 = slots). The runtime integration is relay-only and
flag-gated, so a CPU build cannot execute it end to end; the GPU gates below
must cover it.
