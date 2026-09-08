# Nanbeige attention regression checks

These checks audit the repository; they do not install binaries or change a
running service. A passed kernel test is not a production quality approval.

## Build and run the focused checks

Use an existing Vulkan build, or configure with `GGML_VULKAN=ON` and
`LLAMA_BUILD_TESTS=ON`. Non-system Vulkan and SPIR-V header installations need
`Vulkan_INCLUDE_DIR` and `CMAKE_PREFIX_PATH`, respectively.

```sh
cmake --build build --parallel 3 --target \
    test-turbo-quant test-triattention-score test-backend-ops \
    test-flashprefill-routing test-flashprefill-select test-flashprefill-attn \
    test-flashprefill-state test-kv-cells test-backend-memmove

ctest --test-dir build --output-on-failure --timeout 180 \
    -R 'test-(nanbeige-audit|turbo-quant|triattention-score|flashprefill-.*|kv-cells|backend-memmove)$'

build/bin/test-flashprefill-select --backend Vulkan --required-backend
build/bin/test-flashprefill-attn --required-backend Vulkan
build/bin/test-backend-ops -b Vulkan0 -o SET_ROWS_TURBO
build/bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT \
    -p 'nh=8,nr23=\[6,1\].*type_K=turbo'
```

`test-backend-ops -b` matches the **exact device name**. `-b Vulkan` can skip
every device and still exit successfully; use the enumerated name, normally
`Vulkan0`, and check that the expected cases actually ran. The FlashPrefill
commands above explicitly fail rather than silently substitute CPU or skip
an unavailable backend.

On the RTX 3070 Ti test host, the Turbo cooperative-matrix patch was checked
with 32 dense attention cases (GQA=6, K=Turbo2/3/4, V=Turbo2/3/4/F16/Q8_0,
query batches 1/128 plus the K4/V2 and K4/Q8 256-token prefill shapes),
33 Turbo SET_ROWS cases, and 81 CPU round trips.
These are operator correctness checks, not model-quality scores.

The routing test also calls the real cache planner and graph packer in 36
model-free shapes: GQA=1/6/8, KV heads=1/8, query tiles=5/64/128, and one/two
visibility domains. It checks each row's exact physical key set with sparse
holes and a nonzero physical offset. Ordinary identity Q groups do not
produce unused RoPE inputs; phase-bearing graphs retain their position data.

## Bounded model probes

The standard-library-only runner starts a private loopback listener with an
ephemeral authentication key and stops it in a `finally` block. It stores
per-request responses, checked answers, timing, sampled GPU memory, and server
logs. Finish the build before probing. For binaries inside a CMake build tree,
the runner first checks compiler dependency files for server, common, mtmd,
llama and ggml (including built backend) objects. It rejects newer or missing
local dependencies and duplicate members in the server/common static archives. This
prevents hand-run object relinks or direct `link.txt`/`ar qc` invocations from
silently mixing incompatible C++ layouts. The result records `checked_objects`;
this timestamp check covers the available `*.o.d` files, not a reproducible-build
attestation. When the CMake source is a Git checkout, the runner also executes
`llama-server --version` with the probe's runtime environment and requires its
reported commit to resolve to that checkout's current HEAD. A stale, missing or
ambiguous commit, a failed version command, or a HEAD change during this check
fails before model fingerprinting or server startup. The failed JSON is retained.
`build_preflight.runtime_identity` records the version, HEAD and porcelain Git
status; a matching commit with dirty source is **not** a clean-HEAD build claim.
Source archives without Git metadata explicitly record that identity comparison
was skipped. Installed binaries outside a CMake build tree still require an
operator identity check. A normal, completed target build remains necessary;
neither matching version text nor timestamps prove ABI consistency.
The runner then fingerprints the executable
and local shared libraries before/after each run and rejects changed artifacts;
these hashes do not cover driver or system libraries outside the binary directory.
It records the process exit status **before** its cleanup signal, so a server
crash is distinguishable from the runner stopping a still-live process after
an HTTP failure. GPU memory is a sampled, device-wide observation, not an exact allocation
peak or proof that all graph operations stayed on the GPU.

```sh
python3 tests/test-nanbeige-audit.py -v
python3 scripts/nanbeige-audit.py probe \
    --server build/bin/llama-server --model "$MODEL" --out build/probe-results \
    --configs build/probe-configs.json --corpus "$HELD_OUT_TEXT"
```

Example matrix (not an optimized deployment recommendation):

```json
[
  {"name":"fullkv", "ctx":8192, "kv":8192,
   "batch":256, "ubatch":256, "k":"f16", "v":"f16"},
  {"name":"turbo4-turbo2", "ctx":8192, "kv":8192,
   "batch":256, "ubatch":256, "k":"turbo4", "v":"turbo2"}
]
```

An incorrect checked answer, truncated chat result, incomplete generation
budget, missing/nonfinite timing, or unhealthy final response makes the run
exit nonzero. Per-case evidence is saved even when checks fail. `--no-chat`
disables short QA checks; it must not be described as a quality evaluation.
The three built-in numeric questions are smoke checks, not a benchmark suite.
For prompt-contained retrieval, `--needle-tokens N` keeps the historical
early-needle fixture by default. Repeat `--needle-position` to cover multiple
locations without changing the corpus or answer key, for example:

```sh
--needle-tokens 8192 \
--needle-position early --needle-position middle --needle-position late
```

The middle fixture splits the same tokenized filler in half before
detokenization, so early/middle/late use the same held-out source tokens.
Duplicate positions and positions without a positive needle length are rejected.

Use `--repeats 3` or more for performance comparisons. The result's
`timing_summary.first_request` preserves the process-first request separately;
`repeated_requests` reports count and min/median/max rates for later requests.
All original responses remain present. A one-request probe has null repeated
rates, not an invented warmed-up measurement. Driver disk caches can survive
process restarts, so "first request" does not by itself prove a cold cache.
Compare identical prompts, physical KV, flags and sampling, with no concurrent
build or inference workload. Profiling modes alter scheduling; do not compare
their wall times directly with normal execution.

A matrix entry can set `"require_tri_drain": true` to require an observed
drain that actually frees physical cells. `"max_tri_score_ms": 0` additionally
checks the floor-only fast path, where hard guards already fill the target
and no candidate ranking is needed. These gates parse the real server log,
check before/after/freed accounting, and save the observed events. Do not use
the zero-score bound for general ranked eviction, which legitimately scores
candidates. HTTP success alone does not satisfy a requested pressure gate.

Use `"require_tri_scoring": true` when the test is specifically meant to
exercise calibration/ranking. It requires a real drain with `score_ms > 0`, so
the KV=512 recent-window floor-only case cannot be mistaken for scorer coverage.
For Nanbeige, use physical KV at least 4096 and a prompt longer than physical KV
so the 3/32 target exceeds the fixed recent-128 guard.

Set `"require_flashprefill_plan": true` to require completed FlashPrefill
plan work. The runner enables `/metrics`, saves the response, and validates
finite integral row/block/token counters. Exact-all plans count as real
execution; a switch that ultimately routes all work to ordinary attention
does not. This proves path coverage, not approximate-attention quality.

For the graph integration regression, use a 768-token prompt with
`ctx=4096`, `kv=512`, `batch=ubatch=128`, K4/V2, actual model calibration,
and both evidence gates. The small KV ensures a real drain; early prompt
batches also exercise tensor views narrower than the allocated cache.
FlashPrefill flags for that **exact-all correctness** probe are:

```text
--flashprefill required --flashprefill-min-kv 128
--flashprefill-dense-tail-tiles 0 --flashprefill-full-attn-layers 0
--flashprefill-exact-all
```

Keep `max_tri_score_ms=0` only for this floor-only scenario. It tests the
equivalent fast path when hard guards consume the entire target; it does
not disable scoring when any non-protected candidate must be retained.

For calibration preparation, use `prepare` instead of `probe` with an explicit
**training** corpus. The runner tokenizes with the supplied model and writes a
prefix-deduplicated trie for `wanxiangqi-trie-triattention-calib`. Each matrix
entry has a separate output directory. Never reuse a trie containing another
model's token IDs, and keep evaluation text separate from calibration text.
The bounded preparation uses at most eight 1,024-token documents; it is not
evidence of representative production calibration.

Nanbeige 4.2 has 22 physical weight layers executed twice: size its KV for
44 logical attention layers, 8 KV heads and head dimension 128. Logical context
and physical unified KV are separate capacities. Reducing K/V precision is
not a substitute for correct memory accounting, and TriAttention's fixed
3/32 policy must not be silently relaxed to conceal quality failures.

Production acceptance still needs model-level FullKV/Tri/FlashPrefill A/B,
nontrivial retrieval and generation tasks, shared-prefix pressure, state
save/restore, streaming, and sustained load. Preserve raw outputs and failures;
neither `/health` nor a successful build replaces those gates.

## September 2026 optimization audit (RTX 3070 Ti)

The Vulkan ATTN path now partitions its complete coverage validation across
lanes, and uses an online M/L/O numerator for value heads up to four components
per lane (256 at the current 64-lane workgroup). Larger heads retain the
general two-pass path. This adds no persistent scratch and does not change
the eviction policy, selected-plan semantics, or the V storage domain.
Tests cover the 248/256/264 boundary, correction on/off, Q8 boundary layers,
Turbo K/V, split-K/MERGE and truly empty partials. Invalid small plans must
signal failure; an error-header offset bug was reproduced and fixed.

SELECT dispatch is bounded independently of descriptor validation and uses
grid-stride traversal. CPU/GPU tests cover more actual tiles than dispatched
workgroups. This removes excess launches but did **not** materially accelerate
the selector in the measured workload; serial selection remains a bottleneck.

Paired non-profiled model measurements used the same IQ4_NL Nanbeige model,
768 prompt tokens, 8 greedy generated tokens, 512 physical KV cells, context
4096, batch/ubatch 128, Turbo4/Turbo2 (default Q8 boundary V preserved), fixed
TriAttention 3/32 and forced sparse coverage. `GGML_VK_DISABLE_COOPMAT2=1`
was held equal. Sparse coverage used alpha=1, no sink/window blocks and no
dense tail: these are adversarial coverage settings, **not recommendations**.
Each process ran three requests. Below are medians of requests 2 and 3;
the first request and all raw outputs were retained separately.

| Binary order | Prompt tokens/s | Generated tokens/s | Sampled device VRAM MiB |
| --- | ---: | ---: | ---: |
| Preserved two-pass reference | 191.35 | 58.00 | 2281 |
| Online value accumulation | 251.50 | 58.05 | 2281 |
| Preserved reference rechecked | 190.57 | 58.29 | 2281 |

The paired prefill improvement is about 31%; it is not a decode speedup or a
claim to outperform ordinary dense attention. Plan counters were unchanged,
but greedy text changed (`its itss` versus `its its outputs` in the repetitive
synthetic probe). Floating-point accumulation is not bit-identical. Passing
operator tolerances does not establish unchanged model quality.

Additional checked QA plus 768-token retrieval passed 4/4 with FullKV. The
512-cell Tri configurations failed retrieval with **both** the preserved
reference and the optimized binary; failures included a PEG chat-format HTTP
500 and a repeated-output budget exhaustion. These failures remain visible,
block quality approval for those configurations, and were not worked around
by increasing the retention ratio or accepting malformed responses.

The default cooperative-matrix2-enabled configuration also completed three
requests each at batch/ubatch 128 and 256. The 256 case had a roughly 22-second
first prompt versus 0.35 seconds on repeats (about 2195 prompt tokens/s).
The old failure was not reproduced; first-use overhead still needs separate
operational attention. This is not evidence that every default-backend path
or production load is validated.

Local evidence is under `build-audit-vulkan/continuation/`: `cm2-warm`,
`two-pass-normal`, `online-normal`, `two-pass-recheck`, `quality-reference`,
`quality-smoke`, and `online-ctest.log`. Model results retain binary
fingerprints, flags and per-request responses; successful pressure/sparse
probes additionally record the requested plan/reclaim evidence.
Nothing in this audit installs a binary or restarts the existing service.
