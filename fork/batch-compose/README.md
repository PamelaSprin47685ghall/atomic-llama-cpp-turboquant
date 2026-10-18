# TQBatchCompose

This is a real causal text-generation consumer of an installed llama library
(A) and installed TQScheduling and TQText (B). It owns the request tokens, positions,
sequence ids, greedy samplers, and their interaction. Neither A nor B holds or
calls the other.

The runner accepts one raw prompt per line. It admits requests in input order within fixed token and resident-slot budgets. It serializes admitted prompt batches with sticky Prefill priority. When no admitted prompt remains, it sends one generated token from each active request in a shared Decode batch.
It does not provide an HTTP server or carry the old RERoT, MTP, or KV-retry code.

## Build from installed packages

```sh
cmake -S fork/batch-compose -B build-batch-compose \
  -DCMAKE_PREFIX_PATH="/path/to/installed-A;/path/to/installed-TQScheduling;/path/to/installed-TQText" \
  -DCMAKE_INSTALL_PREFIX="$PWD/install-batch-compose"
cmake --build build-batch-compose
cmake --install build-batch-compose
```

The imported A target is `llama`. This consumer requires the clean fork's
explicit A compatibility seams: `llama_context_params.n_ctx_kv`,
`llama_n_ctx_kv`, `llama_memory_release_compute`, and `llama_memory_compute_bytes`.
There is no fallback to
A's source tree, private headers, or a stub model. An unmodified upstream
installation without these seams will not build this consumer.

## Use and inspect actual batches

Put at least three prompts in `prompts.txt`, then run:

```sh
install-batch-compose/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt \
  --predict 32 --inject-after 2 --json > batches.jsonl
```

The first two requests begin normally. After two successful generation batches, the remaining prompts become eligible for admission. Each newly admitted prompt pauses generation until its Prefill work finishes. Capacity can delay admission; decoding then continues until a slot and its token reservation are released. The first two requests retain their positions and continue afterwards. Without `--inject-after`, every input line is ready at
startup. If an initial request reaches EOG before the selected injection point,
the requested two-decoder demonstration fails visibly. `--ignore-eog` is useful
for a fixed-token validation or benchmark.

`--ctx-size` is the logical per-sequence bound. `--total-kv` selects the
physical shared KV capacity; it does not shrink that logical bound. By default, the physical capacity is the larger of the logical bound and the sum of the largest concurrent request demands. An explicit number is sent
to A as given. A can reject an unsupported or insufficient configuration;
this runner does not silently preempt, retry, evict, or change it.

`--parallel N` limits resident requests and A's sequence slots. The default is the smaller of the input request count and `--batch-size`. A prompt file can contain more requests than slots or batch rows. Request IDs remain stable while A sequence slots are reused; JSON records both identities.

Before Prefill, B reserves `prompt_tokens + predict - 1` tokens for the request's complete upper bound. A full token or slot budget leaves the request queued in input order. Later requests do not bypass it. The bound is conservative when EOG ends a request early. A successful request releases its actual sequence state after all batch outputs have been consumed; only then does C release B's reservation. Failed A calls end the run without optimistic release or retry. This is known-demand admission, not device auto-fit or cache eviction.

For example, a long input file can run through two resident slots:

```sh
install-batch-compose/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt \
  --ctx-size 2048 --total-kv 4096 --parallel 2 --predict 64 --json
```

`--inject-after` requires at least two resident slots and enough token capacity for both initial requests. Invalid option combinations are rejected; an injection cannot wait forever behind a one-slot limit.

`--phase-memory` is on by default. Each prompt-to-generation transition calls
A's public compute-release seam, and failure ends the run. The runner also
synchronizes on phase changes and after actual decode calls. The
`--no-phase-memory` mode executes the same schedule while omitting release, so
its real outputs and allocations can be compared. A successful release call
does not by itself prove a particular number of bytes were returned by a GPU
allocator; measure the backend for that claim.

JSON Lines include:

- `config`: requested capacity and A's actual public getters. GPU layers are
  labeled **requested**, since this runner does not claim actual offload from a flag.
- `phase`: a completed synchronization, the actual compute-release result, and
  A's compute-byte getter before and after the call.
- `inject`: the point when additional requests become eligible.
- `admit`, `wait`, `release`: actual token reservations, capacity pressure, sequence-slot identities, and their generations.
- `batch`: the exact rows submitted to `llama_decode`, its return code, and
  samples read from those successful outputs. A failed decode's submitted rows
  are not counted as committed request positions.
- `result`: generated token ids, text, prompt size, committed decode rows, and
  finish reason. The final sampled token need not be fed back into A, so a
  length-limited request uses `prompt_tokens + generated_tokens - 1` decode rows.
- `error`: a fatal failure. No complete result is fabricated for a failed run.

Every request has its own greedy sampler. A batch's output rows are consumed
before any completed sequence is removed from memory. Token ids are exact;
Generated text uses TQText's per-invalid-byte U+FFFD rule. Every published text delta is valid UTF-8, including at the final token boundary.

Sampling rejects NaN, positive infinity, and rows with no finite candidate before mutating the sampler. Negative infinity remains a valid exclusion. A sample records its best and second finite logits; the second value is null when absent. These values are diagnostics, not evidence that different batch shapes must produce identical tokens.

## Incremental text and stop strings

```sh
install-batch-compose/bin/tq-batch-compose \
  --model /path/to/model.gguf --prompts prompts.txt --parallel 2 \
  --predict 64 --stop '<end>' --stop '###' --json --stream
```

`--stop TEXT` is repeatable. It takes a literal nonempty UTF-8 string, including a value such as `--help`; values are not reinterpreted as options. Matching is case-sensitive on rendered UTF-8 bytes. The first string to complete wins. If several complete at the same byte position, the longest wins; equal strings use configuration order. This avoids changing the selected stop when a tokenizer or caller divides text into different pieces. No Unicode normalization is applied.

The configured stop and any later bytes from its token are suppressed. An undecided stop prefix stays inside B until it completes or stops being a prefix. At EOG or the token limit, C finishes B, which emits any remaining nonmatching prefix and replaces each incomplete UTF-8 byte. A replacement can itself complete a configured stop. The stop count and byte limits come from TQText: 32 strings, 4096 bytes each, and 65536 bytes in total.

The `tokens` result retains every sampled non-EOG token, including tokens whose text is suppressed by a stop. `--ignore-eog` retains EOG tokens too; configured stops still apply. A stopped request reports `finish_reason: "stop"` and its zero-based `stop_index`. Once the match is known, C submits no more decode work for that request. A stop completed by final UTF-8 repair may include bytes from a token that was already decoded. C consumes all outputs of the current batch before it frees any completed sequence. Early completion also returns its admission reservation, allowing a queued request to use that slot.

`--stream` requires `--json` and adds `text` events after the corresponding successful `batch` event. It does not alter model scheduling or sampling. Each event includes:

| Field | Meaning |
|---|---|
| `request_id` | Stable input request identity |
| `offset_bytes` | Number of rendered UTF-8 bytes emitted before this delta |
| `text` | New valid UTF-8 text; it may be empty at completion |
| `generated_tokens` | Actual sampled tokens counted by this request so far |
| `finished` | Whether no more text or model work will follow for this request |
| `finish_reason`, `stop_index` | Terminal reason and matched pattern index, otherwise null |

There is exactly one terminal text event per successful request, before its sequence is released. Concatenating a request's deltas gives its final `result.text`, with contiguous byte offsets. C flushes each text event and treats an output-stream error as fatal. A successful flush does not prove that another process or network peer received the text. The final `result` events remain in input order after all requests complete.

Only resident requests allocate text matching state. Completed and queued requests retain their ordinary input/result data, without a copied stop table for every queue entry. This CLI does not implement HTTP cancellation, an asynchronous output queue, or transport retry semantics.

## Real integration verification

```sh
cmake -S fork/batch-compose -B build-batch-compose \
  -DCMAKE_PREFIX_PATH="/path/to/installed-A;/path/to/installed-TQScheduling;/path/to/installed-TQText" \
  -DTQ_BATCH_COMPOSE_TEST_MODEL=/path/to/model.gguf
cmake --build build-batch-compose
ctest --test-dir build-batch-compose --output-on-failure
```

`tests/real-batch.py` uses real A and B. It checks two requests are actually
decoding before injection, prompt work takes priority, positions continue
without duplicates, sampled tokens are actually fed back, and all requests
finish. It compares every greedy output token and submitted row with release
on versus off, then checks actual A failures for invalid KV configuration and a
missing model. Raw traces and stderr are retained. Set
`TQ_BATCH_COMPOSE_TEST_GPU_LAYERS` to request the GPU variant.

The test is opt-in because it needs a supplied model. Building the binary or
running B's unit tests is not proof of real model integration. Throughput,
backend peak allocation, stochastic sampling, cancellation/retry, and the
stock server's separate compatibility patches need their own evidence.

`tests/test_admission.py` records real queued inference with more requests than batch rows. It checks every reservation, slot generation, actual sequence ID, request position, fed-back token, completion, and release under both token and slot pressure. Test artifacts use a new subdirectory for each run, so repeated CTest runs preserve earlier evidence.

The same original nine prompts run with phase memory on and off for each of three configurations. Each pair must preserve actual row shapes, tokens, and results. Two serial configurations also compare different KV capacities. A six-request canary repeats the first three prompts in the same sequence slots and checks dirty reuse against their fresh first wave. Changing the active batch shape is recorded separately; the pinned runtime already produces different greedy tokens in the archived pre-admission C for this fixture. `tests/diagnose_shapes.py` retains that counterexample without claiming its numerical cause. The request runner treats failures as fatal for the whole job; it has no per-request cancellation or retry API.
