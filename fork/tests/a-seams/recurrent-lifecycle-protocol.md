# Recurrent sequence lifecycle qualification

This test checks the public memory operations used when C reuses a sequence slot. It uses a real recurrent or hybrid model and installed A packages. It does not replace A with a mock or access private cache headers.

## Why the archived M33 clear path is not added

Current A already clears each owned recurrent buffer directly in the constructor and in `llama_memory_recurrent::clear(true)`. Normal `seq_rm` and `seq_keep` operations invalidate metadata. `find_slot` chooses an unreferenced row and records state-copy sources; `llm_graph_context::build_rs` initializes the zero row inside the execution graph and gathers the required states.

The archived M33 optimization reduced the cost of extra eager row clearing in the old fork. Adding that clearing to current A would add device writes and fences to the normal release path. The old timing improvement does not measure an improvement over current A. Keep A's existing ownership and initialization path, and qualify the operations that C actually uses.

The recurrent source and header need no change for this qualification.

## Contract exercised

The test session owns its context, pending batch storage, and host snapshots. It keeps batch storage alive until the next synchronization boundary. State mutation and snapshot operations synchronize the context before they use memory. A submitted decode can be followed by removal or clearing without an intervening logits read; the following operation owns the fence.

The test requests three sequence slots, `n_ctx = 768`, unified KV, batches up to 64 rows, two CPU threads, and `n_rs_seq = 0`. It checks the actual sequence-slot and rollback settings.

- Reuse each of three slots twice, checking exact snapshots of both surviving neighbors before and after refill.
- Copy a shared prefix, release references, and advance one child. Check that untouched siblings retain exact snapshots.
- Reject unsupported partial rollback without changing the saved sequence.
- Restore a valid saved snapshot after unrelated data occupied the same slot.
- Feed the same three continuation tokens with the same one-sequence batch shape. Require finite logits and a maximum absolute difference of at most `1e-5`.
- Submit work, clear metadata or data, and refill. Require the original snapshot bytes and logits.

Comparisons do not assume that different batch widths produce identical greedy tokens.

## Reproduce

Use the clean worktree as the current directory. These commands use the installed CPU and Vulkan A packages produced by `fork/build.py`. Set the model path to a local copy; the test does not download models.

```sh
TQ_LIFECYCLE_MODEL=/absolute/path/LFM2.5-8B-A1B-UD-IQ4_NL.gguf

cmake -S fork/tests/a-seams -B build-release-seams-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build-clean-fork/install" \
  -DTQ_RECURRENT_TEST_MODEL="$TQ_LIFECYCLE_MODEL"
cmake --build build-release-seams-cpu --parallel 4
ctest --test-dir build-release-seams-cpu \
  -R '^recurrent-lifecycle$' --output-on-failure

cmake -S fork/tests/a-seams -B build-release-seams-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build-clean-vulkan/install" \
  -DTQ_RECURRENT_TEST_MODEL="$TQ_LIFECYCLE_MODEL" \
  -DTQ_RECURRENT_TEST_GPU_LAYERS=99
cmake --build build-release-seams-vulkan --parallel 4
ctest --test-dir build-release-seams-vulkan \
  -R '^recurrent-lifecycle$' --output-on-failure
```

CTest records stdout, stderr, and its executed command in each build directory's `Testing/Temporary/LastTest.log`. Verify the GPU offload line as well as the test exit status.

## Observed results: 2026-10-18

The commands above passed with `/home/kunweiz/models/lfm25/LFM2.5-8B-A1B-UD-IQ4_NL.gguf` on CPU and an RX 6800 with Vulkan. Both runs produced:

```json
{"status":"pass","neighbor_snapshot_checks":24,"shared_snapshot_checks":8,"continuation_max_abs":0,"decode_calls":24,"input_rows":189}
```

The Vulkan log confirms 25 of 25 model layers offloaded. The later complete seam suites also passed: 4 of 4 tests on CPU and 5 of 5 on Vulkan. These results qualify the exercised LFM2.5 hybrid convolution/KV lifecycle with rollback disabled. They do not establish a reset speedup or a reduction in peak device memory.

## Limits

Valid snapshots saved and restored in the same process are covered. Malformed snapshot restore is not transactional: current recurrent `state_read_meta` removes the destination before all incoming metadata is validated, and later failures can remove the destination after partial work. A caller must not assume that a failed restore preserves a previous live sequence. This is a source-level limitation; this protocol does not claim an executed malformed-state test or a fix for it.

GDN ingredient replay, rollback snapshots, MTP, PLE, and other recurrent architectures are not qualified by this model run. Cross-context shared arenas, concurrent access to one context, and crash-durable state files are outside this test. A model-specific pass must not be reported as support for all models or backends.
