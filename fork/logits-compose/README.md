# Full-logits collection

`tq-logits-compose` collects actual, full-vocabulary logits from an autoregressive decoder-only text model. It uses installed public llama and GGML APIs. It has no dependency on a private model graph, on the old TP5 runtime, or on another extension library.

This is a diagnostic collector. It does not sample tokens, apply a chat template, score model quality, or measure inference speed.

## Build and install

Requirements: C++17, CMake 3.16 or newer, an installed A package, and a POSIX system with `openat` and `linkat`. Windows is not supported by this package. The parent directory must support hard links between regular files.

```sh
cmake -S fork/logits-compose -B build-logits \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/a/install \
  -DCMAKE_INSTALL_PREFIX=/absolute/path/to/install
cmake --build build-logits -j 4
ctest --test-dir build-logits --output-on-failure
cmake --install build-logits
```

The executable is `bin/tq-logits-compose`. Public A APIs are tied to the upstream revision recorded by this fork. Rebuild this package when updating A; a stable cross-version binary ABI is not assumed.

## Collect and compare

Create one UTF-8 text file containing the complete input sequence. Newlines are input text, not separate requests. Then run:

```sh
tq-logits-compose --model model.gguf --prompt-file input.txt \
  --output reference.bin --ctx-size 2048 --threads 4

tq-logits-compose --model model.gguf --prompt-file input.txt \
  --output candidate.bin --ctx-size 2048 --threads 4 --gpu-layers 99

python3 fork/tools/compare-logits.py reference.bin candidate.bin \
  --atol 0.0001 --rtol 0.00001 --output comparison.json
```

The tolerance above is an example, not a universal accuracy threshold. Choose and record tolerances for the actual model and change under examination. CPU and GPU logits need not be bitwise equal.

The tokenizer inserts configured BOS/EOS tokens by default; `--no-add-special` disables insertion. Special-token spellings in the file are always treated as text. The resulting exact token IDs are stored in the tape. Empty files, invalid UTF-8, and inputs that do not fit the requested context fail. The collector never truncates or shifts the input context.

`--prefill 1` is the default. It saves a row after each input token. Row `i` is the model output after consuming input token `i`; it predicts the next token. It is not the probability vector for the token already consumed. All remaining input tokens come from the file, regardless of the model's preferred next token or EOG output.

For a prefill/decode trace, set `--prefill N`. The first `N` tokens are evaluated in chunks of at most `--batch-size` tokens. Only the final prefix row is saved. Each remaining input token is then evaluated in a single-token call and its row is saved. Tape row `r` follows input position `N - 1 + r`. Zero remaining decode steps is valid. Changes to chunk geometry can change floating-point results and must be recorded when interpreting a comparison.

Input is limited to 16 MiB, with at most 2,000,000 tokens and vocabulary entries. The default context is 2048 tokens. CPU threads must fit the installed public `GGML_MAX_N_THREADS` limit. The default file limit is 1 GiB; `--max-output-bytes N` permits a different explicit limit. The complete file size is checked before context creation. The collector stores input text, token IDs, and one bounded batch. The writer uses a 16 KiB conversion buffer and holds no saved logit history. A owns model, KV, graph, and current logit memory.

## Format and provenance

The file is compatible with the existing `TP5LOG2` comparator. All fields are little endian:

| Field | Representation |
|---|---|
| Magic | Eight bytes: `TP5LOG2` and zero |
| Dimensions | Four uint32 values: vocabulary, prefill tokens, decode steps, total input tokens |
| Input | `total` int32 token IDs |
| Logits | `decode_steps + 1` rows of `vocabulary` IEEE binary32 values |

`total = prefill + decode_steps`. Every input token is range checked. Every saved logit must be finite. The writer refuses missing or extra rows and publication before all declared rows are present.

On success, stdout contains one JSON object with the actual vocabulary, token count, row count, bytes, decode calls and context size, plus batch/thread settings, requested GPU layers, and tokenizer flags. Backend logs go to stderr. Requested GPU layers do not prove actual offload; inspect and retain the backend logs.

The historical tape header has no model, tokenizer, build, hardware, or context-policy identity. Do not infer those identities from equal token IDs and vocabulary sizes. Retain the command, stdout summary, stderr log, model and input hashes, build revision, and relevant backend environment together with each tape. The collector does not silently extend the binary format with unrecognized metadata and does not claim that a seed controls its teacher-forced inputs.

## Output and failures

The output path must be new, including when it names a symlink or directory. The collector holds the output directory open, creates a mode-0600 sibling temporary with exclusive creation, streams the tape, syncs and closes it, then publishes it with a no-clobber hard link. A competing creator wins without being overwritten. The destination becomes visible only after all rows are complete.

Caught input, model, decode, numeric, or I/O errors remove the collector's temporary. They do not publish a partial tape. Exit code 0 means a complete tape was published. Invalid arguments or unsupported input use 2; model, decode, numeric, and I/O errors use 1. If stdout fails after publication, the program returns 1 and reports that the complete tape was already published; that complete file is preserved. The containing directory must be controlled by the caller; this is not a defense against another process with permission to replace arbitrary entries in that directory. Publication is atomic but directory metadata is not synced for crash durability. A process kill or machine failure can leave a `.tq-logits-*.tmp` file; remove it only after confirming that its writer is no longer running.

## Verification

The writer test compares exact little-endian bytes, including signed zero, and checks size limits, invalid dimensions/tokens, nonfinite values, missing and extra rows, existing paths, dangling symlinks, competing writers, a temporary-like destination name, and a real file-size-limit write failure. These tests exercise the actual filesystem writer without a model. The CLI test invokes the real executable and checks parsing, UTF-8 and size limits, output preservation, and model-load cleanup. Python 3 is required when building these tests.

Model validation must additionally run this executable against a real GGUF, inspect the declared input tokens and row counts, and compare repeated or independent reference logits. A passing writer test alone does not qualify model behavior or a GPU backend.

Run the optional real-model qualification with Python 3.11 or newer:

```sh
python3 fork/logits-compose/tests/test_real_logits.py \
  --binary /absolute/path/to/tq-logits-compose \
  --model /absolute/path/to/model.gguf \
  --comparator fork/tools/compare-logits.py \
  --batch-binary /absolute/path/to/tq-batch-compose \
  --output-dir build-logits-real
```

This retains repeated full tapes, scalar-geometry suffix and prefill-only controls, a chunked-prefix tape, comparison output, and all commands and backend logs. The optional batch program checks the submitted prompt tokens and final maximum logit, and confirms that the selected token is within 1e-5 of that maximum, using an independent C consumer with the same prompt geometry. It does not obtain a second full-vocabulary row from that program. Repeated tapes and scalar suffix/prefix controls compare every vocabulary value. It also checks output preservation and, on Linux, a failed stdout sink after complete publication. Add `--gpu-layers 99` for a separate GPU run. It records cross-geometry differences as diagnostics and does not require CPU/GPU equality.
