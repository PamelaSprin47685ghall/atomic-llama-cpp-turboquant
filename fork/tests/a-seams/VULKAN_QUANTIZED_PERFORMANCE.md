# Quantized projection mode timing

`test-vulkan-quantized-shape-invariance --performance` compares two public
per-node modes in the same installed Vulkan library:

| Mode | Requested behavior |
| --- | --- |
| `native_default` | `GGML_PREC_DEFAULT`; retain the normal native matrix selection. |
| `batch_invariant_native_quantized` | `GGML_PREC_BATCH_INVARIANT`; use the supported native single-query quantized arithmetic across the query batch. |

The latter can retain native Q8 activation quantization where the device/type/K
policy selects it. It is distinct from the true-F32 precision mode. The timing
result alone does not establish which shader was dispatched.

## Scope and correctness

The performance branch uses IQ4_NL, Q8_0 and Q4_K weights, K=2048, M=1025 and
N=9/32/128. M=1025 exercises a realistic projection size and a partial output
tile. These dimensions cross the previous small-batch vector boundary. They
are synthetic operator shapes, not a model workload or a model throughput test.

Each mode owns a two-node resident `SCALE -> MUL_MAT` graph. A pair shares the
same allocated weight and source buffers. The source contains twice the fixed
activation, and every graph executes native SCALE by 0.5 before multiplication.
This includes any activation conversion that the native backend performs after
the producer writes the activation. There is no host loop over individual
queries in this branch; one graph computes all N output columns.

The timer includes graph compute, submission and completion synchronization.
Tensor/context creation, allocations, host uploads, output readback, output
validation and progress markers are outside the timed interval. Shared inputs
and intermediate/output buffers remain resident across the samples.

Each mode reports its first graph computation separately, followed by 15 warm
computations of that same graph. The first computation is not a claim that GPU
caches, device pipelines or the whole process are cold. Warm samples are
consecutive within a mode so switching graph objects cannot evict the native
graph recording between its samples. Mode order alternates between cases;
`--performance-first` reverses that order for a second invocation.

Every output must be finite. A first result must contain a nontrivial value,
and all 15 later results must match that mode's first result bit for bit. These
are within-mode stability checks. They do not establish an independent error
bound for the default F16 matrix route or serial equivalence for either mode.
The existing correctness invocation remains the separate exact native
single-column reference gate:

```sh
"$TQ_A_SEAMS_DIR/test-vulkan-quantized-shape-invariance" Vulkan0
```

That correctness invocation intentionally contains reference computations at
N=1. The performance branch does not run them. `--native-matmul` remains an
explicit control for the correctness invocation; it cannot be combined with
`--performance` because the latter always compares both modes.

## Controlled comparison

Build the executable against the exact normal installed A candidate used for
the correctness gate. Use the Release library with Vulkan DEBUG,
CHECK_RESULTS, MEMORY_DEBUG and SHADER_DEBUG_INFO disabled. The public GGML API
does not expose these build settings, so preserve the build-cache and resolved
library evidence externally. Do not infer a normal build from the output.

Run one GPU job at a time. Keep the source, executable, resolved libraries,
driver, device selection and runtime settings identical for both invocations.
The following commands are sequential and reverse the first mode. Set
`TQ_A_SEAMS_DIR` to the installed seam executable directory and `TQ_RECEIPTS`
to the new evidence directory before running them.

```sh
env -u GGML_VK_FORCE_MMVQ -u GGML_VK_DISABLE_MMVQ -u GGML_VK_DISABLE_F16 \
    -u GGML_VK_PERF_LOGGER -u GGML_VK_PERF_LOGGER_CONCURRENT \
    -u GGML_VK_PERF_LOGGER_FREQUENCY -u GGML_VK_PIPELINE_STATS \
    -u GGML_VK_SYNC_LOGGER -u GGML_VK_MEMORY_LOGGER \
    -u GGML_VK_DISABLE_FUSION -u GGML_VK_SERIALIZE_SUBMISSIONS \
    "$TQ_A_SEAMS_DIR/test-vulkan-quantized-shape-invariance" Vulkan0 \
    --performance --performance-first default \
    > "$TQ_RECEIPTS/quantized-performance-default-first.jsonl" \
    2> "$TQ_RECEIPTS/quantized-performance-default-first.stderr"

env -u GGML_VK_FORCE_MMVQ -u GGML_VK_DISABLE_MMVQ -u GGML_VK_DISABLE_F16 \
    -u GGML_VK_PERF_LOGGER -u GGML_VK_PERF_LOGGER_CONCURRENT \
    -u GGML_VK_PERF_LOGGER_FREQUENCY -u GGML_VK_PIPELINE_STATS \
    -u GGML_VK_SYNC_LOGGER -u GGML_VK_MEMORY_LOGGER \
    -u GGML_VK_DISABLE_FUSION -u GGML_VK_SERIALIZE_SUBMISSIONS \
    "$TQ_A_SEAMS_DIR/test-vulkan-quantized-shape-invariance" Vulkan0 \
    --performance --performance-first batch-invariant \
    > "$TQ_RECEIPTS/quantized-performance-invariant-first.jsonl" \
    2> "$TQ_RECEIPTS/quantized-performance-invariant-first.stderr"
```

The executable refuses active runtime performance/sync/memory/statistics
logging controls. Environment switches are presence-based, so setting them to
`0` does not make them unset. Other relevant native settings are recorded in
the configuration event rather than changed by the fixture.

## Evidence interpretation

One successful invocation contains a configuration event, 18 mode results,
nine comparison results and a completion footer with nine cases and 288
completed graphs. A failure or a missing footer does not qualify the run.

Retain every warm sample, the median, p25/p75 and the first-graph duration.
`batch_invariant_over_native_default` is the ratio of the paired warm medians;
values greater than one indicate additional time for the invariant mode.
Compare both orderings for each type and shape and report dispersion. Do not
reduce these operator results to a single whole-model speedup. The reported
maximum output difference between the modes is diagnostic and has no pass
threshold in this timing branch.

Record exact final source identity, executable and resolved library hashes,
build cache, device/driver identity, environment, command lines and return
codes beside the new receipts. Keep earlier failure receipts unchanged.

Actual shader dispatch evidence is a separate diagnostic run using the same
source in an isolated DEBUG backend. Only actual
`ggml_vk_dispatch_pipeline(...)` records inside successful
`TQ_QUANT_PERF_BEGIN`/`TQ_QUANT_PERF_END ... ok` intervals establish dispatch;
pipeline creation names alone do not. The interval's type/shape/mode identity
must match the graph. A DEBUG or timestamp-logger run is not comparable to the
normal-library timings above. The existing MMVQ trace checker covers its own
N<=8 fixture and does not parse these performance intervals.

This evidence can establish the cost of the explicitly selected projection
policy on these resident graphs. Whole-model trajectory, attention visibility,
phase release and native interruption remain separate qualification gates.
