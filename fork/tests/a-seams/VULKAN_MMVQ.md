# Vulkan MMVQ dispatch and timing qualification

`test-vulkan-mmvq` is an installed-public-GGML operator fixture. It constructs
SCALE followed by MUL_MAT on the selected Vulkan device. It does not load a
model, use private headers, change a precision policy, or replace a native
quantized kernel. Its evidence has three separate scopes:

| Evidence | What it establishes |
| --- | --- |
| Successful fixture output | The completed operator outputs satisfy the independent numerical bounds below. |
| `check-vulkan-mmvq.py` PASS | A DEBUG backend recorded the expected native pipeline for every completed case, including every warm graph. |
| Normal Release fixture timings | Resident two-operation graph latency on the recorded device and build. |

The fixture does not establish a full model's dispatch, token throughput, or
RERoT frontier equivalence. Those require the real model receipts and their
existing strict gates. Preserve old failed receipts. Save this qualification in
a new evidence directory with its own exact source and binary identities.

## Matrix and execution

The fixed matrix is K=2048, 1025 output rows, and batch columns 1, 2, and 8 for
each of Q4_K, Q4_0, and IQ4_NL. The row count exercises a final partial workgroup.
Eight columns are within this checkout's `mul_mat_vec_max_cols = 8` limit.
Weights use `ggml_quantize_chunk`, followed by the public type trait's
`to_float` conversion. The oracle uses the values actually encoded in the
weights; it does not treat weight-quantization loss as a backend error.

On the RX 6800, the Q4_K and Q4_0 cases at this K are intended MMVQ witnesses.
The actual DEBUG dispatch is the acceptance condition. If integer-dot support
or a shader build feature is absent, a flag cannot qualify MMVQ. IQ4_NL is an
explicit negative control: the current Q8_1 pipeline selector has no IQ4_NL
entry and must select its native quantized-weight/F32-activation pipeline,
including with `GGML_VK_FORCE_MMVQ` present.

| Weight type | Expected MMVQ dispatch | Native F32 activation control |
| --- | --- | --- |
| Q4_K | `mul_mat_vec_q4_k_q8_1_f32` | `mul_mat_vec_q4_k_f32_f32` |
| Q4_0 | `mul_mat_vec_q4_0_q8_1_f32` | `mul_mat_vec_q4_0_f32_f32` |
| IQ4_NL | No Q8_1 pipeline | `mul_mat_vec_iq4_nl_f32_f32` |

The first graph is recorded separately. There are 15 subsequent timed graphs
per case by default, with a bounded `--samples 1..101` option. Each interval
contains graph compute and backend synchronization. Allocation, input upload,
host reference work, and output readback/validation are outside the interval.
Every output is checked after every graph. A failed status, non-finite value,
unqualified numerical result, or incomplete run prevents successful completion.

SCALE multiplies a resident source by exactly 0.5 on every graph. This writes
the activation that MUL_MAT consumes, as in a model graph. The trace gate
requires both the SCALE dispatch and, on MMVQ, one activation-quantization
dispatch for every compute. A benchmark that silently reuses converted
activations across these writes does not pass. The measured graph includes
SCALE, native matvec, optional Q8 conversion, host submission, and synchronization;
it is not a single-shader latency measurement. Weights remain resident and can
be in the device cache. This small resident fixture cannot establish full-model
memory bandwidth or performance.

## Independent numerical bounds

The oracle is a double-precision sum of public dequantized weights times the
original binary32 activations. It does not call CPU or Vulkan matrix multiply.
It also constructs a Q8 activation reference and an explicit error interval.
The derivation is tied to this checkout's `quantize_q8_1.comp`,
`mul_mat_vecq.comp`, `mul_mat_vecq_funcs.glsl`, `mul_mat_vec.comp`, and
`mul_mat_vec_base.glsl`. Revisit it if their quantization or reduction contracts
change. It is a deterministic finite-input fixture, not a proof over every
possible tensor value or GPU floating-point implementation.

For each 32-value block, let A be the largest activation magnitude and D=A/127.
The source shader computes binary32 d=A/127 and `round(x * (1/d))`, then stores
both d and `d * sum(q)` in binary16. The fixture's extrema and scale are exact
binary32 values. Its other values are D times an integer plus or minus 1/4.
Every intended integer is therefore at least 1/4 away from a rounding tie.
The fixture checks that this margin exceeds `32 * u32 * 127`, where
`u32 = 2^-24`. That envelope covers the shader's binary32 division, reciprocal,
and multiply errors under the native normal-number arithmetic contract; it is
larger than the combined 2.5-ULP division allowances and rounded multiply.

The stored binary16 scale Dh lies in D plus or minus

```
delta_D = D * (8*u32 + 2^-10 * (1 + 8*u32)) + 2^-24.
```

Eight binary32 unit roundoffs cover the division. One full binary16 ULP covers
either nearest or directed scale conversion, and the additive binary16 spacing
covers subnormal rounding. The fixture's scales are normal and finite; the
additive term makes the bound conservative. The selected scales fall at one
quarter or three quarters of a binary16 ULP, so scale rounding is exercised.

Half the blocks have integer sum zero and half have integer sum 128. For these
normal finite values, `half(d * 128) = 128 * half(d)` under the same conversion
mode, because multiplication by 128 is an exact power-of-two change. Thus the
independently stored sum equals Dh times the integer sum. This matters: both
Q4_0 and Q4_K consume the stored sum for their affine correction. Nonzero-sum
blocks make an omitted correction detectable. This fixture does not cover the
additional sum/scale discrepancy for arbitrary non-power-of-two integer sums.

For each output, let W be the publicly dequantized weights and let q be the
known, stable activation integers. Define

```
R = sum_i W_i * x_i
Q = sum_blocks D_block * sum_i_in_block(W_i * q_i)
E_activation = sum_blocks delta_D_block * abs(sum_i_in_block(W_i * q_i)).
```

The Q8 path must lie within `E_activation + E_arithmetic` of Q. Its error
relative to the independent original-activation reference R is bounded by
`abs(Q - R) + E_activation + E_arithmetic`. This uses the measured dequantized
weights and the actual fixture integers. It does not introduce an empirical
absolute tolerance to make an output pass.

`E_arithmetic` also accounts for native affine-factor evaluation, not just the
final dequantized dot product. In each four-bit weight group, L distinct
dequantized values span at least L-1 integer code steps. The value range and the
binary32 dequantization rounding bound therefore give an upper bound on the
step. Q4_0 has offset magnitude eight times its step. A Q4_K group's minimum
code is at most 16-L, which bounds its minimum offset from the same public
values. Constant or invalid groups are rejected by this particular fixture.
IQ4_NL has no affine offset. No private packed block fields are read.

The resulting absolute expression envelope is the sum of
`(abs(W_i) + 2*offset_bound)` times an activation magnitude bound, including a
small explicit binary32 reference-conversion allowance. The native shaders
form at most K/8 nonzero partials per output; local dot/scaling and reduction
contribute fewer than 32 additional rounding stages. With
`gamma(n,u) = n*u/(1-n*u)`, the arithmetic allowance uses
`gamma(K/8+32, 2^-24)` plus `gamma(2*K+16, 2^-53)` for the independent double
reference. Integer dot products cannot overflow for these values.

The fixture reports separate F32 and Q8 validity and bound ratios. A DEBUG
receipt must pass the reference corresponding to the actual dispatched
pipeline. IQ4_NL and the explicit disabled control require the F32 reference
even without trace output. The fixture also requires the bounds to reject an
all-zero result. First-column differences across batches are reported as
diagnostics; this does not replace the real model's stricter shape gate.

## Isolated DEBUG dispatch probe

Use the finalized, manifest-verified A source. Keep a separate Release build
and install prefix for DEBUG. Repeat the normal build's configuration, changing
`GGML_VULKAN_DEBUG=ON` only. Keep `GGML_VULKAN_CHECK_RESULTS`,
`GGML_VULKAN_MEMORY_DEBUG`, and `GGML_VULKAN_SHADER_DEBUG_INFO` off. CHECK_RESULTS
adds an independent CPU execution path and is not part of this probe. The
DEBUG macro exposes the existing dispatch logger; no backend or shader source
change is needed to add instrumentation.

A minimal configuration corresponding to `fork/build.py` is below. Include any
additional recorded compiler/backend options from the normal build. Set the
task paths to fresh directories before running it.

```bash
python3 fork/check-upstream.py
cmake -S "$TASK_SOURCE" -B "$TASK_TRACE_ROOT/a" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF \
  -DGGML_VULKAN_DEBUG=ON -DGGML_VULKAN_CHECK_RESULTS=OFF \
  -DGGML_VULKAN_MEMORY_DEBUG=OFF -DGGML_VULKAN_SHADER_DEBUG_INFO=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_TOOLS_INSTALL=OFF -DLLAMA_TESTS_INSTALL=OFF \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_INSTALL_PREFIX="$TASK_TRACE_ROOT/install"
cmake --build "$TASK_TRACE_ROOT/a" --parallel 4 --target llama
cmake --install "$TASK_TRACE_ROOT/a"
python3 fork/check-upstream.py
cmake -S "$TASK_SOURCE/fork/tests/a-seams" -B "$TASK_TRACE_ROOT/fixture" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TASK_TRACE_ROOT/install"
cmake --build "$TASK_TRACE_ROOT/fixture" --parallel 4 --target test-vulkan-mmvq
```

Record the exact source manifest, CMake caches, compiler, device/driver identity,
fixture digest, and actual resolved A library digests. Ensure the fixture loads
this isolated install, without another build's libraries in its search path.
Normal and DEBUG binaries have different identities and separate receipts.

Run each policy in a fresh process: the backend reads the controls during device
initialization. Both controls are presence-based, so a value of `0` still sets
the control. If both are present, disable wins. An example Bash matrix, after
setting `TASK_TRACE_BIN` and creating a new `TASK_RECEIPT` directory, is:

```bash
set -euo pipefail
task_clean=(env -u GGML_VK_DISABLE_MMVQ -u GGML_VK_FORCE_MMVQ
  -u GGML_VK_PERF_LOGGER -u GGML_VK_PERF_LOGGER_CONCURRENT
  -u GGML_VK_PERF_LOGGER_FREQUENCY -u GGML_VK_PIPELINE_STATS
  -u GGML_VK_SYNC_LOGGER -u GGML_VK_MEMORY_LOGGER)

"${task_clean[@]}" "$TASK_TRACE_BIN" --device Vulkan0 --samples 2 \
  > "$TASK_RECEIPT/trace-auto.jsonl" 2> "$TASK_RECEIPT/trace-auto.stderr"
python3 fork/tests/a-seams/check-vulkan-mmvq.py \
  --events "$TASK_RECEIPT/trace-auto.jsonl" --stderr "$TASK_RECEIPT/trace-auto.stderr" \
  --expect mmvq --policy auto > "$TASK_RECEIPT/trace-auto-receipt.json"

"${task_clean[@]}" GGML_VK_FORCE_MMVQ=1 "$TASK_TRACE_BIN" --device Vulkan0 --samples 2 \
  > "$TASK_RECEIPT/trace-force.jsonl" 2> "$TASK_RECEIPT/trace-force.stderr"
python3 fork/tests/a-seams/check-vulkan-mmvq.py \
  --events "$TASK_RECEIPT/trace-force.jsonl" --stderr "$TASK_RECEIPT/trace-force.stderr" \
  --expect mmvq --policy force > "$TASK_RECEIPT/trace-force-receipt.json"

"${task_clean[@]}" GGML_VK_DISABLE_MMVQ=1 "$TASK_TRACE_BIN" --device Vulkan0 --samples 2 \
  > "$TASK_RECEIPT/trace-disable.jsonl" 2> "$TASK_RECEIPT/trace-disable.stderr"
python3 fork/tests/a-seams/check-vulkan-mmvq.py \
  --events "$TASK_RECEIPT/trace-disable.jsonl" --stderr "$TASK_RECEIPT/trace-disable.stderr" \
  --expect f32 --policy disable > "$TASK_RECEIPT/trace-disable-receipt.json"
```

The checker reads only actual `ggml_vk_dispatch_pipeline(...)` lines between
fixture graph markers. Pipeline creation logs, option echoes, and ordinary
native tensor callbacks are not dispatch proof. The logger runs before the
Vulkan bind/dispatch calls, so a record alone is insufficient. Successful end
markers, all completed graphs, finite full outputs, and the appropriate
independent numerical check are required as well. The parser tests use
synthetic logs and qualify only the evidence gate:

```bash
python3 fork/tests/a-seams/test-vulkan-mmvq-trace.py
```

## Controlled normal-build timing

Build this same fixture against the finalized normal Release A install, with
all four build diagnostics off. Use the same device and matrix, no unrelated
GPU load, and the clean environment above. Run fresh processes for auto,
force, and disable, using `--samples 15`. Save each stdout/stderr pair under a
new name. Repeat the three policies in a rotated order for a small number of
rounds when comparing them, so one policy does not always receive the coldest
device. Report all raw samples, median, interquartile range, first-compute
time, order, and exact build/device identities. Do not discard slow samples or
mix DEBUG measurements into normal-build comparisons. The public API does not
expose the backend's DEBUG compile option; build provenance supplies that fact.

If GPU timestamps are needed to explain a difference, use a separate diagnostic
run with `GGML_VK_PERF_LOGGER=1` and `GGML_VK_PERF_LOGGER_FREQUENCY=1`.
The existing non-concurrent logger inserts synchronization between graph ops,
then submits, waits for completion, and reads timestamp queries. Its MUL_MAT
record can contain both activation conversion and matvec dispatch. It reports
op/fusion timing, not a pipeline name or an isolated matvec shader time. The
concurrent mode changes grouping. Keep either mode's measurements separate
from the normal uninstrumented resident-graph timings.
