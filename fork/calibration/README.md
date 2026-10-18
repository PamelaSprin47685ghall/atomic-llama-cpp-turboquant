# TQCalibration

TQCalibration is an independent C++17 reader for the WXQCAL03 routed-expert calibration format. It uses only the standard library. It has no dependency on llama, GGML, a model, or another extension library.

## Build and install

```sh
cmake -S fork/calibration -B build-calibration \
  -DCMAKE_INSTALL_PREFIX=/absolute/path/to/install
cmake --build build-calibration -j 4
ctest --test-dir build-calibration --output-on-failure
cmake --install build-calibration
```

Consumers use `find_package(TQCalibration CONFIG REQUIRED)` and link `TQCalibration::calibration`.

## Read owned layer values

```cpp
#include <tq/calibration/reader.h>
#include <fstream>

std::ifstream input("calibration.bin", std::ios::binary);
tq::calibration::reader source(input);
const auto header = source.header();
while (auto layer = source.next()) {
    // Consume or move the owned layer arrays here.
}
```

The input stream must outlive the reader and must not be used by another reader during parsing. A reader cannot be copied or moved. `header()` returns a value. Each returned layer owns its arrays and can outlive the reader. Retaining every returned layer is the caller's memory choice; `next()` itself holds no layer history.

The reader checks the exact file end before returning the last layer. For an empty dump, the first `next()` checks the end. Subsequent calls return no value. A parse failure makes that reader terminal. It cannot be resumed after the caller catches an exception. Errors are standard C++ exceptions with a reason; incomplete input is not silently returned as a completed layer.

## Format

The supported on-disk byte order is little endian. The reader decodes it explicitly, including on a big-endian host. Old writers emitted payload arrays in host byte order despite the format's little-endian declaration. Files made by a big-endian old writer are outside this contract; the format has no byte-order marker that can safely identify them.

| Section | Fields |
|---|---|
| Magic | Eight bytes `WXQCAL03` |
| Header | Six uint32 values: layer count, maximum context, activation stride, activation prefix tokens, routing stride, flags |
| Progress | Three uint64 values: decoded positions, declared total, context resets; one complete byte and seven zero bytes |
| Layer | Four uint32 values: layer ID, experts, embedding width, hidden width; three uint64 observation counts: input, routing, covariance |
| Always present | uint64 routing counts per expert; binary64 input square sums per embedding coordinate |
| EXPERT_INPUT, flag 4 | uint64 sample counts per expert; binary64 first-moment sums and square sums, each expert-major |
| HIDDEN, flag 2 | uint64 sample counts per expert; binary64 hidden square sums, expert-major |
| COVARIANCE, flag 1 | binary64 full square covariance accumulation, embedding width squared |

Unknown flags, invalid complete bytes, nonzero padding, repeated layer IDs, invalid geometry, truncated arrays, and trailing data fail. Covariance is read and validated in a fixed 64 KiB buffer; it is not stored in the returned layer. All values must be finite. Square sums and covariance diagonals must be nonnegative. Signed first moments and off-diagonal covariance entries are valid. This reader does not prove covariance symmetry or positive semidefiniteness.

Each expert count must not exceed the observation count for its population. Activation and routing observation counts must not exceed the declared decoded positions. Covariance observations must not exceed activation observations and must be zero if covariance is absent. Zero samples require zero corresponding statistics. These rules reject inconsistent data even if its complete byte is set.

Routing and activation samples are distinct populations. The reader does not require routing counts to equal expert-input or hidden counts. The routing observation count records token rows; summing routing counts also includes top-k expert multiplicity. Stride zero remains valid: the old producer used it to sample every batch. An activation prefix of zero means the full observed batch. A maximum context of zero means no collector-imposed limit.

## Limits and completeness

Defaults are 4096 layers, 256 MiB of on-disk data per layer, and 16 GiB for the file. The layer limit includes the fixed layer header and covariance bytes, even though covariance is streamed. Size arithmetic and limits are checked before array allocation. The caller may provide a `read_limits` value to choose smaller or larger bounds.

The parser accepts a structurally valid partial or overshot progress range. It preserves `complete`, `decoded`, and `total` exactly; the consuming application decides whether that range is usable. A true complete byte alone does not prove full-corpus coverage. The old producer could select a capped range, finish its final batch beyond that cap, or stop with an incomplete checkpoint.

The companion C application `tq-imatrix-compose` applies the imatrix export policy. It is a separate installed consumer, not a dependency of this library.

## Tests

The standalone test reads real byte sequences and rejects every truncated prefix of a valid file. It also covers independent populations, signed first moments, output ownership, repeated IDs, malformed flags and geometry, numeric errors, zero-count inconsistencies, byte budgets, covariance truncation, terminal failure, empty dumps, and partial progress preservation. Production-writer compatibility is an additional integration check; a generated fixture alone is not a model calibration or quantization quality result.
