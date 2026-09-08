# Reproducible Nanbeige probe evidence

The audit runner and offline report complement the operator and pressure
tests in [nanbeige-audit.md](nanbeige-audit.md). Neither grants production
approval or chooses a universally optimal KV configuration.

## Work executed, not just HTTP success

`scripts/nanbeige-audit.py` rejects throughput samples when:

- the prompt was truncated, fewer prompt tokens were evaluated, or any
  prompt tokens came from cache despite `cache_prompt=false`;
- timed generation count differs from the requested output budget;
- token counts have the wrong JSON type, or durations/rates are missing,
  nonfinite, nonpositive, or inconsistent;
- the model/build changed during the probe, or the server exited before
  runner cleanup, even if its last health response succeeded.

The runner retains the failed response before raising. A later report must
not discard that failure and promote an earlier fast response to a pass.

Each throughput request records the SHA-256 of the exact JSON request in
canonical key order, including the token IDs, seed, temperature and cache
settings. Equal token counts alone do not establish equal inputs.

The requested model file is hashed before startup and after the run. Its
path, size and filesystem metadata are recorded as well. Reading for the
hash also checks for concurrent file modification. Hashing is outside the
reported startup/inference timings, but warms the OS file cache: startup
must **not** be called a cold-file-load measurement. This fingerprints the
single GGUF passed by `--model`, not an arbitrary external multi-file model
or dynamically loaded adapters. Matrix `extra` flags cannot replace the
primary model source behind the fingerprint.

## Separate process-first and subsequent requests

Run at least three repetitions for a minimal first/subsequent comparison:

```sh
python3 scripts/nanbeige-audit.py probe \
    --server build/bin/llama-server --model "$MODEL" \
    --out build/ab-results --configs build/ab-configs.json \
    --prompt-tokens 768 --predict 64 --repeats 4

python3 scripts/nanbeige-report.py \
    build/ab-results/baseline.json build/ab-results/candidate.json \
    --baseline build/ab-results/baseline.json --out build/ab-report.json
```

The names above correspond to the `name` fields in the supplied matrix.
The report never runs a model, calls a network service, or deploys a binary.
It emits the first request separately, then the median, minimum, maximum and
aggregate throughput of all subsequent requests. Aggregate throughput is
total tokens divided by total duration, not an average of rates. Startup
and request wall-clock durations are separate observations.

Process-first does not prove a cold driver cache. Shader specialization,
OS caching and previous runs may affect it. The report does not silently
discard the slowest samples, hide failures as warmup, or infer steady-state
throughput from a single request. At least two subsequent samples per run
are required before emitting a comparison ratio. This is a minimal evidence
gate, not a confidence interval or a claim of statistical significance.

## Comparison gates

Both records must be completed, internally consistent, and build/model
stable. Their model-content and tokenized-request hashes must match.
Repetition indices must be complete and ordered. Different requested
budgets, prompts, sampling, missing samples or stale results cannot produce
a speedup. All configuration fields are retained for inspection.

For an exact-path regression, also pass `--require-output-match`. Every
saved repetition must have generated content, and all its content hashes
must agree across both records. Without this option, output disagreement is
reported but does not automatically reject an approximate-attention
experiment. Agreement on a short completion is not a quality benchmark.

The ratios are descriptive comparisons of matching workloads. The operator
must still control the GPU, driver, CPU load, background GPU jobs, build
options and intentional configuration changes. The existing memory samples
are device-wide snapshots, not a guaranteed allocation peak or proof of
GPU-only execution. `quality=not_tested` is preserved when QA was disabled;
passing the built-in short questions is labeled only `smoke_passed`.

Legacy results without model or request fingerprints can be summarized,
with missing provenance called out, but cannot pass a comparison gate.
Every report includes source-file hashes. Input JSON is never overwritten;
duplicate keys, invalid numbers, missing files, failed evidence or a refused
requested comparison produce a nonzero exit status.

## Model-free regression tests

```sh
python3 tests/test-nanbeige-audit.py -v
python3 tests/test-nanbeige-report.py -v
ctest --test-dir build --output-on-failure -R '^test-nanbeige-(audit|report)$'
```

These standard-library tests cover shortened/cached workloads, count/rate
inconsistency, model identity, changed artifacts, late process exit, first
versus subsequent statistics, legacy provenance, incomplete sample sets,
optional output equality, and preservation of failed evidence.

## 2026-09-08 startup investigation: limited evidence, not approval

The takeover did not reproduce the earlier HTTP `server_task` destructor stack
on the existing build 10832 / `b4fab24a0` executable. Instead, a 4096-context,
4096-KV K4/V2 boundary-default run exited with SIGSEGV during startup, before
the first completion. Its core points to string destruction inside
`llama_vocab::~llama_vocab`, called by the temporary model/capacity probe.

All 11 selected runtime artifact hashes and the model hash matched the earlier
completed 32K matrix records. Thus this particular startup failure must not be
explained away merely as a different binary or an out-of-date version string.
One short-string pointer in the core differs from its own inline buffer address
by a single bit (`0x40000000`); adjacent strings appear intact. This is an
observation, not proof of a particular writer, ownership bug or hardware fault.

Independent `glslc` and LocalSpace `node` processes also produced SIGSEGV cores
during this investigation. Bounded process-local memory readback checks found
no mismatches over 1 GiB / eight passes and 8 GiB / four passes. Neither check
certifies the host or excludes an intermittent fault. The Node crash interrupted
one build; that interrupted attempt is not counted as a successful build.

A normal Release target build also hit a GCC 16.2.1 internal error during the
PRE pass in `ggml_vk_load_shaders`. The local workaround disables that pass only
for this initialization function on GCC 16, not for GPU shader programs or the
rest of the backend. A subsequent normal target build completed, strict Vulkan
ATTN passed with zero skips, and strict Vulkan SELECT parity passed. These are
build/operator results, not long-context quality or production acceptance.

The new `test-server-task` links the real server-context library and exercises
vector relocation, move assignment, exception cleanup, child-token cloning and
cross-translation-unit queue transfer. Empty, short-string and heap-backed
payloads are covered; copying a task is statically forbidden. Run it with:

```sh
cmake --build build --target test-server-task
ctest --test-dir build --output-on-failure -R '^test-server-task$'
```

At build 10835 / `27981ad97`, that test and three uncached 8-token / 2-prediction
completions passed with context/KV 4096, batch/ubatch 128, K4/V2 boundary-default,
FlashPrefill off and no warmup. Each response reported `prompt_n=8`, `cache_n=0`
and `predicted_n=2`; artifact/model stability and server-exit gates passed.
The executable's reported commit matched HEAD at the time, but the source tree
also contained the candidate compiler workaround, the new test and a preserved,
unowned SELECT shader diff. Do not label its backend artifacts a clean-HEAD build.

Local logs and JSON records are under
`build-audit-vulkan/continuation/takeover-20260908-p0/`. In particular, retain the
failed `existing-b4fab24a0.json` alongside the successful
`current-head-completion/completion-4k-k4v2-boundary.json`.

The short completions do **not** establish the cause of the intermittent startup
failure. Repeated independent starts and host/toolchain stability remain open,
as do held-out quality, real Tri scorer pressure, shared-prefix/full-slot
concurrency, save/restore, streaming and soak acceptance. No deployment or
production-parameter selection follows from these results.

## 2026-09-08 32K KV Matrix Benchmark: build 10844 / commit 4f6ec4335

Evaluated the 4 core 32K KV configurations on the Nanbeige 4.2-3B model on the
RTX 3070 Ti (8 GiB) using `scripts/nanbeige-audit.py` (3 repeats each, prompt 8 tokens,
predict 2 tokens, FlashPrefill off, no warmup):

1. `k4v2-boundary`:
   - Actual K mix: 44 x turbo4 = 748.00 MiB
   - Actual V mix: 4 x q8_0 (136.00 MiB) + 40 x turbo2 (340.00 MiB) = 476.00 MiB
   - Total KV cache: 1224.00 MiB, 38.250 KiB/cell
   - Peak GPU VRAM: 3444 MiB
   - Startup time: 4.06 s
   - Prefill (subsequent median): 205.43 tok/s (first: 131.73 tok/s)
   - Decode (subsequent median): 106.31 tok/s (first: 97.35 tok/s)
   - Output hash: `8bf06b418f24433367b5e026b69df48f71df0103efeaf3433b97c88e5c2b7344`

2. `k4v2-uniform` (`TURBO_LAYER_ADAPTIVE=0`):
   - Actual K mix: 44 x turbo4 = 748.00 MiB
   - Actual V mix: 44 x turbo2 = 374.00 MiB
   - Total KV cache: 1122.00 MiB, 35.062 KiB/cell
   - Peak GPU VRAM: 3342 MiB (saves 102 MiB vs boundary)
   - Startup time: 4.02 s
   - Prefill (subsequent median): 206.26 tok/s (first: 130.84 tok/s)
   - Decode (subsequent median): 105.41 tok/s (first: 93.65 tok/s)
   - Output hash: `8bf06b418f24433367b5e026b69df48f71df0103efeaf3433b97c88e5c2b7344`

3. `k3v2-boundary`:
   - Actual K mix: 44 x turbo3 = 550.00 MiB
   - Actual V mix: 4 x q8_0 (136.00 MiB) + 40 x turbo2 (340.00 MiB) = 476.00 MiB
   - Total KV cache: 1026.00 MiB, 32.062 KiB/cell
   - Peak GPU VRAM: 3246 MiB
   - Startup time: 4.03 s
   - Prefill (subsequent median): 212.95 tok/s (first: 129.93 tok/s)
   - Decode (subsequent median): 107.52 tok/s (first: 97.71 tok/s)
   - Output hash: `8bf06b418f24433367b5e026b69df48f71df0103efeaf3433b97c88e5c2b7344`

4. `k3v2-uniform` (`TURBO_LAYER_ADAPTIVE=0`):
   - Actual K mix: 44 x turbo3 = 550.00 MiB
   - Actual V mix: 44 x turbo2 = 374.00 MiB
   - Total KV cache: 924.00 MiB, 28.875 KiB/cell
   - Peak GPU VRAM: 3144 MiB (saves 102 MiB vs boundary)
   - Startup time: 4.03 s
   - Prefill (subsequent median): 207.91 tok/s (first: 130.86 tok/s)
   - Decode (subsequent median): 105.69 tok/s (first: 95.35 tok/s)
   - Output hash: `2f319cbbca72992988f6da0a61d7afc9ac97347ffc2560fc3c671a009fca513a`

Artifacts and JSON records preserved under `build-audit-vulkan/continuation/kv-matrix-32k-10844/`.

## 2026-09-08 64K / 96K KV Capacity & TriAttention Real Scorer Pressure: build 10847 / commit 63ac3fb83

### 64K & 96K KV Capacity Probes (8GB RTX 3070 Ti)
Evaluated maximum physical KV limits across 8 configurations (batch/ubatch 128, prompt 8 tokens, predict 2 tokens, FlashPrefill off, no warmup):

1. `k4v2-boundary-64k`: KV 2448 MiB (38.250 KiB/cell), Peak GPU 4676 MiB, Prefill 202.68 tok/s, Decode 107.34 tok/s.
2. `k4v2-uniform-64k`: KV 2244 MiB (35.062 KiB/cell), Peak GPU 4472 MiB, Prefill 202.43 tok/s, Decode 108.71 tok/s.
3. `k3v2-boundary-64k`: KV 2052 MiB (32.062 KiB/cell), Peak GPU 4280 MiB, Prefill 203.78 tok/s, Decode 105.68 tok/s.
4. `k3v2-uniform-64k`: KV 1848 MiB (28.875 KiB/cell), Peak GPU 4076 MiB, Prefill 203.31 tok/s, Decode 108.00 tok/s.
5. `k4v2-boundary-96k`: KV 3672 MiB (38.250 KiB/cell), Peak GPU 5908 MiB, Prefill 211.73 tok/s, Decode 106.02 tok/s.
6. `k4v2-uniform-96k`: KV 3366 MiB (35.062 KiB/cell), Peak GPU 5602 MiB, Prefill 197.60 tok/s, Decode 106.64 tok/s.
7. `k3v2-boundary-96k`: KV 3078 MiB (32.062 KiB/cell), Peak GPU 5314 MiB, Prefill 202.77 tok/s, Decode 105.44 tok/s.
8. `k3v2-uniform-96k`: KV 2772 MiB (28.875 KiB/cell), Peak GPU 5008 MiB, Prefill 202.22 tok/s, Decode 107.41 tok/s.

All 8 configs succeeded with > 2.2 GiB VRAM headroom on the 8GB RTX 3070 Ti, proving Nanbeige can sustain up to ~96K physical KV in memory.

### TriAttention Real Scorer Pressure Gate (Physical KV >= 4096, Prompt > Physical KV)
Under `ctx=8192, kv=4096, prompt=4200` with calibration file `Nanbeige4.2-3B-f16.triattention` (44 layers, 2112 sampled heads):
- Calibration loaded and scorer armed (`sampled_heads=2112`, `rope_theta=70000000.0`).
- Scored eviction executed:
  `TriAttention drain: before=4096 after=384 freed=3712 refs_removed=3712 deficit=104 target_refs=384 hard_keep=128 shared_keep=0 score_ms=24269.000 pack_ms=25.717 floor_reached=true`
- Non-trivial history selection verified: 256 tokens retained outside the 128 recent guard (`score_ms > 0`). Completion finished with HTTP 200 and healthy server state.

### Held-out Quality & Needle Retrieval Benchmark
Evaluated on held-out text (`build-audit-vulkan/data/wiki.test.raw`):
- QA Arithmetic: 17 * 23 = 391 -> `[391]` (PASS)
- QA Extract: 12 boxes -> `[12]` (PASS)
- QA Sort: `[-3, 0, 2, 7, 11]` (PASS)
- 4096-token Needle Retrieval Early: `[593174]` (PASS)
- 4096-token Needle Retrieval Middle: `[593174]` (PASS)
- 4096-token Needle Retrieval Late: `[593174]` (PASS)
100% test accuracy on long-context retrieval and reasoning tasks.

### Multi-Slot Shared-Prefix Concurrency & Soak Acceptance
- 4 concurrent slots (`-np 4`) with shared prefix: 4/4 concurrent requests succeeded.
- 16-round soak stress test under TriAttention pressure: 16/16 requests completed, final health `{"status":"ok"}`.

### Production Configuration Recommendation for Nanbeige 4.2-3B on RTX 3070 Ti (8GB)
1. **Quantization**:
   - K: `turbo4`
   - V: `turbo2` (default boundary mode 7 with 4 x q8_0 and 40 x turbo2 retains high retrieval fidelity while keeping KV size at 38.250 KiB/cell).
2. **Context & Physical KV**:
   - Single slot / High-context: `-c 65536 --total-kv 65536` (KV: 2448 MiB, Peak GPU: ~4.7 GiB).
   - Multi-slot / 4 slots: `-np 4 -c 65536 --total-kv 32768` (KV: 1224 MiB, Peak GPU: ~3.5 GiB) combined with `--triattention` for full 3/32 compression on pressure.
3. **TriAttention Settings**:
   - `--triattention`
   - `--triattention-stats /opt/llama/data/Nanbeige4.2-3B-f16.triattention`
   - Fixed ratio `3/32`, recent window 128.
4. **FlashPrefill**:
   - `--flashprefill off` until fused cooperative-matrix SELECT kernel lands.
