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
logs. Finish the build before probing. The runner fingerprints the executable
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

A matrix entry can set `"require_tri_drain": true` to require an observed
drain that actually frees physical cells. `"max_tri_score_ms": 0` additionally
checks the floor-only fast path, where hard guards already fill the target
and no candidate ranking is needed. These gates parse the real server log,
check before/after/freed accounting, and save the observed events. Do not use
the zero-score bound for general ranked eviction, which legitimately scores
candidates. HTTP success alone does not satisfy a requested pressure gate.

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
