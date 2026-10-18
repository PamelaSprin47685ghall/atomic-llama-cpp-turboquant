# Meta async transfer regression

## Purpose

The Meta backend splits one tensor across child backends. A byte offset in the full tensor is not the same byte offset in each smaller shard. The synchronous set/get paths already convert the logical chunk index to the local shard offset. The asynchronous paths rejected every nonzero offset and still passed the full tensor offset to each shard.

The fix removes the two zero-offset restrictions and applies the existing local offset calculation to both asynchronous paths. It changes four lines in `ggml/src/ggml-backend-meta.cpp` and adds four replacement lines. It adds no public API, backend interface field, queue, cache, or allocation policy.

The regression uses public GGML APIs and two real CPU child backends. The expected result is a flat host byte array. It tests each transfer direction separately, so a write error cannot cancel a matching read error.

## Transfer contract

- The tensor must be contiguous. The split state must have one segment and one repetition.
- Split axes 0, 1, and 2 remain supported. For a split axis `a`, both offset and size must be multiples of `tensor->nb[a + 1]`. This unit is a full contiguous chunk that contains the split dimension.
- A mirrored tensor uses the same byte offset on each child. It does not need the split-chunk alignment, but it still has the normal tensor bounds and layout requirements.
- Axis 3, partial-sum tensors, noncontiguous tensors, and multiple segments or repetitions are outside this asynchronous contract. The patch does not extend those cases. Existing assertion failures for invalid calls remain in place.
- Async set borrows the caller's source bytes. The caller must keep them alive and unchanged until synchronization finishes.
- Async get borrows the caller's destination. The caller must keep it alive and wait for synchronization before reading or reusing it.
- `ggml_backend_synchronize(meta)` waits for all child backends. The test calls it before inspecting results and before destroying its buffers or backend.

This is not a snapshot API. It does not copy source bytes into a new owned transaction at submission time.

## Cases

`test-meta-transfers.cpp` runs this matrix:

| Dimension | Cases |
| --- | --- |
| Split layout | Axis 0, axis 1, axis 2, mirrored |
| Storage type | F32, F16 |
| Range | All six outer chunks; two chunks starting at chunk 1; the last chunk |
| Operation | Async get with independent sync initialization; async set with independent sync verification |

Each split uses unequal shard extents of 3 and 7. The complete matrix contains 48 cases. `--zero-only` selects 16 controls. `--get-only` and `--set-only` each select 24 cases.

The test checks the requested bytes, the complete tensor after the operation, leading and trailing destination guards, and the unchanged source of async set. It checks range arithmetic before allocating guards or forming subrange pointers. All checks remain active in Release builds.

There is no model, shader, mocked child backend, or fallback to a handwritten copy implementation in this test. A missing CPU backend is a test failure.

## Reproduce before and after

Use two separate A install prefixes. Build the baseline prefix from commit `3746ebf82585f47c88fb48097f6ae45cb972b031`, before the Meta offset fix. Build the fixed prefix from the branch containing this fix. Build the same current regression source against each prefix.

Set `TQ_META_BASELINE_PREFIX` and `TQ_META_FIXED_PREFIX` to the absolute paths of those installs. Run these commands from the repository root:

```bash
cmake -S fork/tests/a-seams -B build-meta-before \
  -DCMAKE_PREFIX_PATH="$TQ_META_BASELINE_PREFIX"
cmake --build build-meta-before --parallel 4 --target test-meta-transfers
```

On Linux, the following check disables core files and requires the two baseline failures. It must run against the baseline installation:

```bash
python3 - <<'PY'
from pathlib import Path
import resource
import signal
import subprocess

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
binary = Path('build-meta-before/test-meta-transfers').resolve()
for option, expected in [('--zero-only', 0),
                         ('--get-only', -signal.SIGABRT),
                         ('--set-only', -signal.SIGABRT)]:
    result = subprocess.run([str(binary), option], capture_output=True, timeout=60)
    if result.returncode != expected:
        raise SystemExit(f'{option}: expected {expected}, got {result.returncode}')
    if option == '--zero-only':
        if b'PASS 16 real meta transfer cases' not in result.stdout:
            raise SystemExit('the zero-offset controls did not run')
    elif b'offset == 0' not in result.stderr:
        raise SystemExit(f'{option}: failure did not reach the old offset assertion')
    print(option, result.returncode)
PY
```

Then build and run against the fixed install:

```bash
cmake -S fork/tests/a-seams -B build-meta-after \
  -DCMAKE_PREFIX_PATH="$TQ_META_FIXED_PREFIX"
cmake --build build-meta-after --parallel 4 --target test-meta-transfers
build-meta-after/test-meta-transfers
```

The final command must return 0 and print `PASS 48 real meta transfer cases on two CPU children`.

Removing only the two assertions is not a complete fix: it leaves the wrong local offsets. The middle-range cases compare each direction with independently initialized or read tensor contents and check that untouched chunks remain unchanged.

## Recorded run on 2026-10-18

The run used Linux x86-64 and GCC 16.2.1 in `/home/kunweiz/Desktop/atomic-llama-cpp-turboquant-clean-20261018`.

The baseline source file had 2,421 lines and SHA-256 `d5b2ef371ec4c2faf44c3b286136b0e645a4e58fecc8164f3c575fd65905c884`. The baseline executable was `build-release-seams-cpu/test-meta-transfers`, linked to the A installation in `build-clean-fork/install` before that installation was rebuilt with the fix.

| Run | Observed result |
| --- | --- |
| Baseline `--zero-only` | Return 0; 16 controls passed |
| Baseline `--get-only` | Return -6 (`SIGABRT`); failed at the old nonzero-offset restriction |
| Baseline `--set-only` | Return -6 (`SIGABRT`); failed at the old nonzero-offset restriction |
| Fixed CPU seam suite | 4/4 CTest tests passed, including all 48 Meta cases |
| Fixed Vulkan-linked seam suite | 5/5 CTest tests passed, including all 48 Meta cases |

Baseline subprocesses used `resource.RLIMIT_CORE = (0, 0)`. Their logs are in `build-release-meta-before/{zero-control,get-offset,set-offset}.{stdout,stderr}`, with a `summary.json` receipt.

After the patch, the CPU run rebuilt target `llama` in `build-clean-fork/upstream`, installed it to `build-clean-fork/install`, then ran:

```bash
cmake --build build-release-seams-cpu --parallel 4
ctest --test-dir build-release-seams-cpu --output-on-failure
```

The Vulkan run rebuilt targets `llama-server`, `llama-bench`, and `mtmd` in `build-clean-vulkan/upstream`, installed A to `build-clean-vulkan/install`, then ran:

```bash
cmake --build build-release-seams-vulkan --parallel 4
ctest --test-dir build-release-seams-vulkan --output-on-failure
```

The explicit range and guard-allocation checks removed GCC's `-Wstringop-overflow` warning without a diagnostic suppression. The install prefixes were reused after rebuilding; their current binaries are the fixed version, not preserved baseline binaries. Use separate prefixes to repeat the before/after experiment.

## Hardware and evidence limits

Both Meta runs use two CPU children, including the test linked through the Vulkan-enabled installation. CPU transfers exercise the actual Meta routing and byte layout, but do not prove overlap or deferred host-pointer use on a GPU queue.

The surrounding Vulkan suite also exercises its own existing GPU tests. That does not turn this Meta test into a Vulkan, CUDA, multi-GPU, or heterogeneous-device transfer test. Those configurations remain unqualified by this regression. No model throughput or end-to-end latency claim follows from these results.

## Other audited Meta candidates

The source review also checked the premises of the old graph-cache and pinned-readback changes from [the backend audit](../../audit/audit-final-backend-findings.md).

- **Graph view cache identity (FBE15):** this A initializes `backend_ctx->uid` to zero and never assigns it later. Its subgraph rebuild condition therefore stays true. Importing the old cache-key refinement alone would not fix an active cache path. A future cache feature must define and test parent/view identity together with its actual reuse behavior.
- **Pinned readback retirement (FBE12):** this A has no pending pinned-readback transaction. The old retirement code assumes the separate fast path and its retained buffers already exist. That feature needs its own measured benefit and complete ownership contract before migration.

The async offset correction (FBE11) applies to an existing public operation and has a direct failing regression. It does not require either of those additional mechanisms.
