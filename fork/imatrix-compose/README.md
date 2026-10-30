# Calibration to imatrix

`tq-imatrix-compose` converts WXQCAL03 routed-expert statistics to the GGUF imatrix consumed by A's `llama-quantize`. It uses the independently installed TQCalibration reader and the installed public GGUF/GGML API. It does not load a model, create an inference context, or initialize a GPU.

The converter preserves measured raw square sums and expert sample counts. It does not compute a new calibration or establish model quality. It replaces the old converter's hardcoded paths, silent partial acceptance, and implicit expert filling with explicit contracts.

## Build and install

Requirements: C++17, CMake 3.16 or newer, A's installed `ggml` package, installed TQCalibration, and a little-endian POSIX system with `openat` and `linkat`. The destination filesystem must support hard links. Windows and big-endian A GGUF output are outside the current qualification.

```sh
cmake -S fork/imatrix-compose -B build-imatrix \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/install \
  -DCMAKE_INSTALL_PREFIX=/absolute/path/to/install
cmake --build build-imatrix -j 4
ctest --test-dir build-imatrix --output-on-failure
cmake --install build-imatrix
```

The installed executable is `bin/tq-imatrix-compose`. A's public ABI is tied to this fork's pinned revision. Rebuild and requalify C after updating A. The package needs no NumPy or Python runtime to convert; Python 3 is required for its tests.

C links `ggml::ggml-base` for public metadata functions and TQCalibration for parsing. It uses the installed package's `GGML_INCLUDE_DIR` because the pinned base target does not export its header path. It does not link llama or backend runtime libraries. A's `ggml` package configuration still discovers the backends enabled in that installation; retain a complete A installation when configuring C.

## Convert

```sh
tq-imatrix-compose --input calibration.bin --output imatrix.gguf \
  --dataset 'my-selected-calibration-range'

llama-quantize --imatrix imatrix.gguf input-model.gguf output-model.gguf Q4_K_M
```

Choose a quantization type and real source model appropriate to the calibration. The example command is not a quality recommendation. The converter cannot verify model or tokenizer identity because WXQCAL03 carries neither. Retain the model/input hashes, producer revision, collection command, corpus selection, and context policy beside the files. A dataset label and matching shapes do not establish that identity.

The output must be a new path. Existing regular files, directories, and symlinks are refused, including the input path. The input must remain unchanged throughout conversion.

## Exact mapping

For each layer `N`, the converter writes six F32 tensors. Array order is expert-major, matching A's per-expert normalization.

| WXQCAL03 statistic | GGUF name | GGML shape, fastest dimension first |
|---|---|---|
| `exp_diag` | `blk.N.ffn_gate_exps.weight.in_sum2` and `blk.N.ffn_up_exps.weight.in_sum2` | embedding width, experts |
| `exp_n` | The same two weights with `.counts` | 1, experts |
| `hid_sum` | `blk.N.ffn_down_exps.weight.in_sum2` | hidden width, experts |
| `hid_n` | The down weight with `.counts` | 1, experts |

A divides each expert's raw sums by its count. The converter does not divide measured sums in advance. Both required statistic flags must be present. Squared sums must remain finite and nonnegative in F32; overflow and nonzero-to-zero underflow fail. The converter also rejects a nonzero F32 sum whose division by the F32 count would become zero in A. Ordinary F64-to-F32 rounding is part of this export format and is not claimed to be bitwise preservation of the input statistics.

Sample counts must be exact F32 integers and at most INT32_MAX. Counts that would round are rejected; they are not clamped or rescaled. `imatrix.chunk_count` is the positive decoded-position counter, bounded by INT32_MAX for A's actual consumer. `imatrix.chunk_size` is 1. These fields do not change A's per-expert normalization and do not prove coverage of every corpus request.

## Selected-range completeness

Default conversion requires the complete byte to be true, a positive declared total, and `decoded == total`. A structurally valid file may still fail this policy. The old producer's complete byte meant that traversal returned without an error. Its total could be a selected node cap, and its final batch could overshoot that cap.

`--allow-partial` accepts these range limitations explicitly. The GGUF retains the original complete byte, decoded/total values, context limit, reset count, flags, and sampling parameters. `tq.calibration.partial_reasons` records every applicable reason:

- `collector_reported_incomplete`
- `declared_total_is_zero`
- `decoded_below_declared_total`
- `decoded_above_declared_total`

Overshoot is retained as overshoot; the converter does not truncate statistics or repair counters. This option does not bypass structural, numeric, population-count, or byte-limit failures. All output is described as a declared selected range; full-corpus coverage remains unverified.

## Unseen experts

The default `--unseen reject` rejects any expert with no observed input or hidden samples. A zero count must already have zero corresponding source statistics.

For an intentional estimate, use `--unseen global-mean`. Each missing expert row receives the pooled profile `sum(observed expert square sums) / sum(observed expert counts)` and an output count of 1. Input and hidden populations are treated separately. A layer with no observed expert in either required population still fails.

The output records the chosen policy, imputed input/hidden expert counts, and exact IDs. An ID is a uint64 with the layer ID in its high 32 bits and expert ID in its low 32 bits. Each list is limited to 65536 records. `tq.calibration.imputed_rows_are_measurements` is false. The replacement count is an encoding for the estimate, not an observed sample. Measured rows retain their original sums and counts.

## Memory and output lifecycle

Default limits are 256 MiB of input data per layer, 16 GiB total input, and 2 GiB final output. Use `--max-layer-bytes`, `--max-input-bytes`, and `--max-output-bytes` to set explicit alternatives. The reader also limits the file to 4096 layers; GGUF metadata is limited to 16 MiB. These are resource limits, not recommended calibration sizes.

Only one parsed layer is kept at a time. Covariance is validated in 64 KiB chunks and discarded. F32 payload conversion uses a 16 KiB buffer and a private disk spool; it does not retain all output arrays. A holds only tensor metadata with data allocation disabled. Mean estimates need at most one input-width and one hidden-width vector. Imputation records and GGUF metadata are separately bounded.

Two private mode-0600 files are created in the destination directory: a payload spool and a final staging file. They coexist during the final copy, so conversion needs about twice the final payload size in free disk space, plus metadata. After all layers, statistics, and the exact source end have been validated, the converter asks A to encode the GGUF metadata and copies the spool behind it. It checks each tensor's offset and payload size against A's public GGUF directory. The completed staging file is synced and closed, then published with a no-clobber hard link. A competing creator is never overwritten.

Caught failures before publication remove this conversion's private files without publishing its output. Existing destination entries, including a competing writer's completed output, are preserved. The caller must control the directory; hostile replacement by another process with directory write permission is outside this contract. Directory metadata is not synced for crash durability. A process kill or system crash can leave `.tq-imatrix-*.tmp` files; remove them only after confirming that no writer is running.

Exit 0 means a complete GGUF was published and its stdout JSON summary was delivered. Invalid arguments, malformed or unsupported calibration, and policy rejection return 2. The reader also reports some input read failures as invalid calibration, so they can return 2. Failures to open the input or output directory, caught output I/O failures, and other runtime failures return 1. If only stdout fails after publication, the complete GGUF remains and the error says so. The summary contains counts and scope indicators, not model quality or throughput claims.

## Verification

`test_conversion.py` invokes the actual executable, reads its actual GGUF bytes with an independent standard-library reader, and checks tensor shape/order/values, typed metadata, partial reasons, explicit mean estimates, malformed numeric data, exact-count limits, existing files, symlinks, competing writers, resource-limit write failures, and source preservation. Linux also checks failed stdout delivery through `/dev/full`.

These are format and filesystem tests. The release integration must additionally use the unchanged old writer to generate a production-format fixture and pass the exported imatrix to the actual A loader/quantizer. Such a fixture qualifies interoperability and real format consumption, not trained-model calibration or quantization quality. No historical calibration file is bundled or presumed available.

### Unchanged-writer receipt

`tests/old-writer-fixture.cpp` supplies two small layers with all three statistic flags to the original `wxq::write_calib` implementation. Compile it with `calib-dump.cpp` and `calib-dump.h` from the selected old revision, without modifying those files. Set `OLD_LIB` to that archive's `wanxiangqi/lib` directory. Record the commit, compiler command, fixture driver digest, and both old-source SHA-256 digests at build time; keep this build record with the receipt.

```sh
mkdir -p build-old-writer
sha256sum "$OLD_LIB/calib-dump.cpp" "$OLD_LIB/calib-dump.h" \
  fork/imatrix-compose/tests/old-writer-fixture.cpp
c++ -std=c++17 -O2 -I "$OLD_LIB" \
  fork/imatrix-compose/tests/old-writer-fixture.cpp \
  "$OLD_LIB/calib-dump.cpp" -o build-old-writer/old-writer-fixture
```

Set `OLD_REVISION`, `OLD_WRITER_CPP_SHA256`, and `OLD_WRITER_H_SHA256` to the values in that build record, and `INSTALL_PREFIX` to the C installation. The qualification directory must not exist yet.

```sh
python3 fork/imatrix-compose/tests/check_old_writer.py \
  --producer build-old-writer/old-writer-fixture \
  --converter "$INSTALL_PREFIX/bin/tq-imatrix-compose" \
  --writer-revision "$OLD_REVISION" \
  --writer-source-sha256 "$OLD_WRITER_CPP_SHA256" \
  --writer-header-sha256 "$OLD_WRITER_H_SHA256" \
  --output-dir build-old-writer/qualification \
  > build-old-writer/receipt.json
```

The check runs both actual executables, reads all twelve GGUF tensors, and preserves the production-format input. Its JSON records the actual commands, binary digests, input/output digests, converter summary, and old source identity. The old revision and source digests are explicitly caller-supplied build provenance; the check does not reconstruct the producer binary from source or authenticate an old archive. This receipt alone does not qualify A's imatrix loader. Retain the separate actual `llama-quantize` invocation, exit status, and output with the release validation record.
