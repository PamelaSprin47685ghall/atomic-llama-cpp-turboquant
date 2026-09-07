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
