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
with 24 dense attention cases (GQA=6, K=Turbo2/3/4, V=Turbo2/3/4/F16,
query batches 1/128), 33 Turbo SET_ROWS cases, and 81 CPU round trips.
These are operator correctness checks, not model-quality scores.

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
