# Compute workspace seam validation

`compute-memory.cpp` is a standalone C++17 test executable. Link the patched,
installed A's public ggml API and CPU backend. It takes no arguments and does
not need a model. Exit 0 means all checks passed; checks remain enabled with
`NDEBUG`. This CPU test validates allocator ownership and graph lifetime, not
GPU memory residency or the CUDA arena's separate layout protocol.

It queues a graph and a host readback, then immediately releases the scheduler
workspace. It verifies the readback, zero allocator-owned bytes, idempotent
release, unchanged external input bytes and scheduler/backend/callback state,
smaller allocation for the next small graph, and regrowth for a later large
graph. The test always builds fresh graph metadata after a release.

For a real model, compare two contexts with identical model/parameters/prompt,
sampling chain, and seed. Only the experimental context calls
`llama_memory_release_compute` at a completed prompt boundary. Record
`llama_memory_compute_bytes` immediately before release, immediately after, and
after the next one-token decode. Check release returns true, after-release
bytes are zero, tokens/logits match the baseline, and KV sequence position and
capacity are unchanged. Repeat with generation followed by another prompt.
Keep host logits returned by the last decode valid across release. Exercise
the project's enabled GPU backend separately with real asynchronous work.

Unsupported-mode checks: NULL must return false; NULL byte query returns zero.
An active optimizer, metadata-only context, and CUDA joint KV/compute arena
must return false before synchronization or mutation. For a configured CUDA
arena, compare subsequent output with a baseline to verify rejection leaves
the arena valid. Do not claim this guard has been tested on CUDA merely because
the CPU allocator test passed.

The byte query reports current buffers owned by this graph allocator. It
excludes model weights, KV/recurrent state, outputs, the MoE provider cache,
driver caches, and allocator memory retained by the device driver. A zero
result is not a measurement of physical free VRAM. Access to a given context
or scheduler must remain serialized; these APIs do not add a concurrent
decode/trim protocol and must not be invoked from an evaluation callback.
