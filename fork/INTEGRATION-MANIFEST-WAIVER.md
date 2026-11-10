# Integration Manifest Waiver — `integ/tp5-mtp-r28b`

Status 2026-11-10: the full registration resolves the guard;
`python3 fork/check-upstream.py` now exits 0. Sections A-F retain the historical
record of the pre-registration drift and the WS-owned registrations.

- Base: `71544695b`
- Merged: WS2 `ae9c13065`, WS3 `6d0614915`, WS4 `15fd113e2`, WS5 `9cd97421e` (parent `4648777f7`)
- Applied: `patches/ws5/{ggml-backend-meta.cpp,runtime.cpp,ggml-backend.cpp}.patch`
- Guard (2026-11-10): `python3 fork/check-upstream.py` rc=0
- Historical pre-registration failure: `boundary check failed: reviewed content or mode differs: ggml/src/ggml-cuda/mmvf.cu`

## B8. Full manifest registration on p5unified 2026-11-10

The real upstream pin remains `cda0f9256665845cd9d6f3a0176880883f653a3d` in
`fork/upstream.json` and `fork/upstream-exceptions.json`. No synthetic baseline
is introduced and no undeclared path remains.

The earlier c9715b052 registration (314 entries) is the base. This round
re-signs every entry whose bytes changed in the unified p5unified tree and adds
the 26 non-`fork/` paths that became changed after that registration. Every
entry carries the real SHA-256 and mode read from the current worktree, plus
named consumers, reason, and removal condition. The manifest now has 340
entries.

Newly registered paths in this round:

- build: `CMakeLists.txt`, `ggml/src/CMakeLists.txt`
- docs: `docs/meta-host-view-copy.md`
- examples/tools: `examples/idle/idle.cpp`, `tools/completion/completion.cpp`,
  `tools/cvector-generator/cvector-generator.cpp`, `tools/llama-bench/llama-bench.cpp`
- provider/DSO lifecycle: `ggml/src/ggml-backend-dl.cpp`,
  `ggml/src/ggml-backend-dl.h`, `ggml/src/ggml-backend-module.cpp`,
  `ggml/src/ggml-backend-module.h`, `ggml/src/ggml-backend-reg.cpp`,
  `ggml/src/ggml-metal/ggml-metal.cpp`, `ggml/src/ggml-opencl/ggml-opencl.cpp`,
  `ggml/src/ggml-zdnn/ggml-zdnn.cpp`
- relay/cleanup/state: `ggml/src/ggml-relay-token-guard.h`,
  `src/llama-impl.h`, `src/llama-io.h`, `src/llama-relay-free.h`,
  `src/tp5-cleanup-owner.h`
- CPU tests: `tests/test-meta-host-view-copy.cpp`,
  `tests/test-sched-receipt.cpp`, `tests/test-tp5-context-cleanup-concurrent.cpp`,
  `tests/test-tp5-hc-replicated-compute.cpp`,
  `tests/test-tp5-relay-free-decision.cpp`,
  `tests/test-tp5-staging-owner-runtime.cpp`

`.gitattributes` marks the retained patch/text artifacts as whitespace-exempt:
`audit/m3-tail-storage.patch.txt`, the WS0 overlay patches,
`patches/ws5/*.patch`, `fork/logits-compose/README.md`, and the
`fork/mtp-progress-20261031` archive. Source and test files are not exempted.

Guard result: `changed_upstream_paths=340`, `extension_paths=366`,
`exception_manifest_sha256=c768ef406b4d9c619c618c0f4eaf1315a62d11904b3a04c77003959f354c44ec`.
No source or test file was modified, added, or removed by the manifest round.

## A. Pre-existing mismatches (historical)

These files were already stale at the integration base and were not touched by
WS2/WS3/WS4/WS5. The table below preserves the pre-registration state; B8
re-signed these entries to the current worktree bytes.

| path | recorded | actual | category |
|---|---|---|---|
| `ggml/src/ggml-cuda/mmvf.cu` | `bac6f2bcc571b53d` | `fedc9845cd888afd` | Pre-existing CUDA/HIP A overlays |
| `ggml/include/ggml-backend.h` | `4dc9e1da84bd9b45` | `423b22be2af4cb2a` | Pre-existing GGML public-header A overlays |
| `ggml/include/ggml.h` | `3a5f74e8d196bb3d` | `166592e1bd2c6670` | Pre-existing GGML public-header A overlays |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `8cfe7ee653073919` | `68c5e873c8179784` | Pre-existing CPU attention/SIMD A overlays |
| `ggml/src/ggml-cpu/ops.cpp` | `e5b6affc76f02ec3` | `748308ad652cb3b6` | Pre-existing CPU attention/SIMD A overlays |
| `ggml/src/ggml.c` | `5f49b8219ad06040` | `e71c6027c9a67696` | Other pre-existing A paths |
| `include/llama.h` | `bad66f3437131f82` | `97faf1a6dfe6d18e` | Other pre-existing A paths |
| `src/models/qwen4exp.cpp` | `1238129e2a822b48` | `6de26b7ff01f3375` | Pre-existing TP5 model A overlays |
| `src/llama-cparams.h` | `f36eb2da24e040e9` | `151a9d0197a73da8` | Other pre-existing A paths |
| `src/llama-graph.cpp` | `3d72bbced50b8118` | `49e380a7b5ce41e5` | Other pre-existing A paths |
| `src/llama-graph.h` | `087693cb84344308` | `967d9ba0b4d703bd` | Other pre-existing A paths |
| `src/llama-memory-recurrent.cpp` | `cd13ab45e4d44346` | `7bc3ffec20148c0e` | Pre-existing recurrent/model-loader A overlays |
| `src/llama-memory-recurrent.h` | `8f79b71f2935ee2a` | `cfc9daadb544f5a3` | Pre-existing recurrent/model-loader A overlays |
| `src/llama-model.cpp` | `64edc5905857f88e` | `706420b172655379` | Other pre-existing A paths |
| `ggml/src/ggml-cuda/mmvf.cuh` | `a5192767d8034085` | `436c0d6963bca942` | Pre-existing CUDA/HIP A overlays |
| `ggml/src/ggml-alloc.c` | `94ce30f2e57da646` | `effc1dcb97ab008a` | Other pre-existing A paths |
| `ggml/src/ggml-cuda/mmvq.cu` | `8b79fc059843010d` | `67cf17349c30a65f` | Pre-existing CUDA/HIP A overlays |

## A0. The original mismatches at the integration base

At `71544695b` the manifest had 25 mismatches. Six of them are
WS4/WS5-owned files whose delta is re-signed in B2; the rest are untouched.

| path | recorded at base | actual at base | disposition |
|---|---|---|---|
| `ggml/src/ggml-cuda/mmvf.cu` | `bac6f2bcc571b53d` | `fedc9845cd888afd` | untouched (Section A) |
| `ggml/src/ggml-cuda/relay.hip` | `ac13937177faa06b` | `1badd280b9a826d4` | re-signed in B2 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ed4085720505729a` | `e5647253f50af967` | re-signed in B2 |
| `ggml/include/ggml-backend.h` | `4dc9e1da84bd9b45` | `423b22be2af4cb2a` | untouched (Section A) |
| `ggml/include/ggml.h` | `3a5f74e8d196bb3d` | `166592e1bd2c6670` | untouched (Section A) |
| `ggml/src/ggml-backend-meta.cpp` | `cfbc9a166e42f384` | `ae54d2b301c43526` | re-signed in B2 |
| `ggml/src/ggml-backend.cpp` | `797281a796648103` | `b728749ed376f1d9` | re-signed in B2 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `8cfe7ee653073919` | `68c5e873c8179784` | untouched (Section A) |
| `ggml/src/ggml-cpu/ops.cpp` | `e5b6affc76f02ec3` | `748308ad652cb3b6` | untouched (Section A) |
| `ggml/src/ggml.c` | `5f49b8219ad06040` | `e71c6027c9a67696` | untouched (Section A) |
| `include/llama.h` | `bad66f3437131f82` | `97faf1a6dfe6d18e` | untouched (Section A) |
| `src/models/qwen4exp.cpp` | `1238129e2a822b48` | `6de26b7ff01f3375` | untouched (Section A) |
| `src/llama-context.cpp` | `ca6286855864e316` | `29fa8694d1b357d8` | re-signed in B2 |
| `src/llama-context.h` | `a097c9435ccb2472` | `03ed46b16546c9ec` | re-signed in B2 |
| `src/llama-cparams.h` | `f36eb2da24e040e9` | `151a9d0197a73da8` | untouched (Section A) |
| `src/llama-graph.cpp` | `3d72bbced50b8118` | `49e380a7b5ce41e5` | untouched (Section A) |
| `src/llama-graph.h` | `087693cb84344308` | `967d9ba0b4d703bd` | untouched (Section A) |
| `src/llama-kv-cache.cpp` | `839df30582f517ee` | `45da363f0b0a0492` | re-signed in B2 |
| `src/llama-kv-cache.h` | `e230833526353e57` | `50e565be4a4b467e` | re-signed in B2 |
| `src/llama-memory-recurrent.cpp` | `cd13ab45e4d44346` | `7bc3ffec20148c0e` | untouched (Section A) |
| `src/llama-memory-recurrent.h` | `8f79b71f2935ee2a` | `cfc9daadb544f5a3` | untouched (Section A) |
| `src/llama-model.cpp` | `64edc5905857f88e` | `706420b172655379` | untouched (Section A) |
| `ggml/src/ggml-cuda/mmvf.cuh` | `a5192767d8034085` | `436c0d6963bca942` | untouched (Section A) |
| `ggml/src/ggml-alloc.c` | `94ce30f2e57da646` | `effc1dcb97ab008a` | untouched (Section A) |
| `ggml/src/ggml-cuda/mmvq.cu` | `8b79fc059843010d` | `67cf17349c30a65f` | untouched (Section A) |

## B. Re-signed entries

### B1. Matched at base, changed by this integration (normal re-sign)

| path | old hash | new hash | source |
|---|---|---|---|
| `ggml/include/ggml-relay.h` | `7fdd8249b94a7072` | `6515be02a00f7755` | WS5 |
| `ggml/src/ggml-relay-impl.h` | `527c06d6540832d1` | `5bffa8b906d2ad78` | WS5 |
| `ggml/CMakeLists.txt` | `64e32884913d8a7b` | `f29244c173f414a2` | WS5 |

### B2. Already stale at base, re-signed only for the WS delta

The old recorded hash was already wrong before this integration. The new hash signs the
WS4/WS5 delta only; the prior drift is not reviewed or blessed by this act.

| path | old recorded | actual at base | new hash | source |
|---|---|---|---|---|
| `ggml/src/ggml-cuda/relay.hip` | `ac13937177faa06b` | `1badd280b9a826d4` | `8ca6d29c1d2088c3` | WS4/WS5 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ed4085720505729a` | `e5647253f50af967` | `feb717e88f4af933` | WS4/WS5 |
| `ggml/src/ggml-backend-meta.cpp` | `cfbc9a166e42f384` | `ae54d2b301c43526` | `1fb3fdd97cb368fb` | WS4/WS5 |
| `ggml/src/ggml-backend.cpp` | `797281a796648103` | `b728749ed376f1d9` | `07e8059e51f8eb3a` | WS4/WS5 |
| `src/llama-context.cpp` | `ca6286855864e316` | `29fa8694d1b357d8` | `633c469274e2cd1f` | WS4/WS5 |
| `src/llama-context.h` | `a097c9435ccb2472` | `03ed46b16546c9ec` | `6bb331b8054be7c1` | WS4/WS5 |
| `src/llama-kv-cache.cpp` | `839df30582f517ee` | `45da363f0b0a0492` | `8f5a8f81ef6e3943` | WS3 |
| `src/llama-kv-cache.h` | `e230833526353e57` | `50e565be4a4b467e` | `449b9f8b397c7509` | WS3 |

### B3. Refinement round 2026-11-09: KV decode scratch/snapshot ownership

Base `5a34498fd` (`feat/pillar5-refine-20261109`). Cherry-picks
`c6e985353` -> `92864c701` and `0bfb6a75f` -> `94e508c4d`; the follow-up
`40ea8d93b` restores the context copy/move deletion, clarifies the exception
boundary, and records the final digests. These three entries are re-signed for
this round only. The older B2 hashes for the two KV files are left as history;
this section does not rewrite them, and no Section A/C drift is re-signed.

| path | old recorded at 5a34498fd | new hash | source / responsibility |
|---|---|---|---|
| `src/llama-kv-cache.cpp` | `5e0eb2c88b7f2377cd1e1f5768026a2b360fa2d90cc7c71315b962a877e5019f` | `b69703a4140951c6b5526b13419f0c928795d2d8ae3ccd6d5cc8defd2abe335c` | `c6e985353`, `0bfb6a75f`, `40ea8d93b`: `init_batch()` holds the decode scratch in a guard and returns it on every non-transfer exit; `prepare_into()` reserves the recovery vectors before taking a snapshot slot; `prepare_reference()`/`prepare_into()` document that the reverse restore is the normal-path rollback and that an allocation failure after `apply_ubatch()` is a C++ exception escaping `llama_decode()`, not a `llama_decode()` return. The documented abort (2) / fatal (< -1) partial-memory rule covers the normal return path; it does not make that exception a supported return. A no-throw rollback is impossible because `llama_kv_cells::set()` re-inserts into `std::set`/`std::map` and can allocate. |
| `src/llama-kv-cache.h` | `0fbe78bb2693efebd6fd4cc1d27e4012170d9f8badba3fafaf24082ec3fc825d` | `5c8099c830c0f928f2946c77b16b9509f06638dbb879ed75ca27a5bd03cbb775` | `c6e985353`, `0bfb6a75f`, `40ea8d93b`: `snapshot_pool::reserve()` reserves `storage` and `free` before publishing a slot, so `release()` does not allocate in the noexcept guard destructor; the decode-scratch pool comment states that the pool synchronizes the scratch buffers only, not the KV cells/heads. The context copy/move deletion is retained/restored because the context owns the exclusive decode scratch; a copy would create a second owner and return the same scratch twice, and a move would leave the source still holding it. The type-shape static asserts removed by `0bfb6a75f` are not restored; the class-level deletion is the ownership invariant. |
| `tests/test-tp5-ubatch-pool.cpp` | `4b0a4b63716fba5fc7b8fb3f39b0aa471fc982a0fec429c0c66bffc402721999` | `bac8b583b2794ad9850a1c59195fe6a12a1acfd4252ab46f2eb5c97b88f06b34` | `c6e985353`: adds `test_decode_scratch_pool_concurrency()` with eight threads checking exclusive hand-out, full return and exact fallback counting; `0bfb6a75f` removes the context copy/move static asserts. CPU-only; registered as `test-tp5-ubatch-pool` with label `main`. |

Ownership and boundary: A owns the KV cache scratch/snapshot pool; the named
consumer is `tq-tp5-compose (WS3 host pool)` and the CPU test is
`llama batch allocator tests`. No public API, B module, or C runtime changes.
The pool flag is `TP5_DECODE_SCRATCH_POOL` (plus the existing
`TP5_KV_SNAPSHOT_POOL`); both stay default-off. The install feature-string gate
now lists `TP5_DECODE_SCRATCH_POOL` for `install/lib/libllama`.

Verification: the three actual file digests match the table. The narrow CPU
build reused `/tmp/r28b-pillar5-cpu` (reconfigured with
`LLAMA_BUILD_TESTS=ON LLAMA_BUILD_COMMON=ON`, CUDA/HIP OFF), built only
`test-tp5-ubatch-pool`, and ran it once: `OK failures=0 allocs=1682 frees=532`,
119 `PASS`, 0 `FAIL`, including the new
`scratch pool concurrent: exclusive slot`, `every slot returned`,
`capacity was sufficient`, and both `scratch pool exhausted` checks; exit 0.
`python3 fork/check-upstream.py` still fails first at
`ggml/src/ggml-cuda/mmvf.cu`; that is the pre-existing red item recorded in
Section A, not a new mismatch from this round.

### B4. Multi-stream restore round 2026-11-09

Base `a93914c36` (`feat/pillar5-refine-20261109`). Cherry-picks
`e126dbf3f` from `feat/pillar5-refine-kv-polish-20261109`. The cherry-pick
conflicted in `src/llama-kv-cache.cpp` and `fork/upstream-exceptions.json`;
`src/llama-kv-cache.h` and the test merged cleanly. The resolution keeps
`24d086a80`'s context copy/move deletion and `40ea8d93b`'s exact exception
boundary, and fuses the multi-stream restore description. The resolved hunks
contain comment and manifest text only; the restore logic and the test are
the source commit's. These three entries are re-signed for this round only.
The older B3 hashes are left as history; this section does not rewrite them,
and no Section A/C drift is re-signed.

| path | old recorded at a93914c36 | new hash | source / responsibility |
|---|---|---|---|
| `src/llama-kv-cache.cpp` | `b69703a4140951c6b5526b13419f0c928795d2d8ae3ccd6d5cc8defd2abe335c` | `1b50c714a6ca94a841c3c6cc27954bb942fef9ad188226241bf31321a926f7d0` | `e126dbf3f`: both prepare paths call the inline `restore_cells()`; it restores head `v_heads_old[sinfo.strm[s]]` and cells `v_cells_old[s]`. The fused comment states that the reverse restore runs on success and on the find_slot failure path, and that the exception window starts after the first `apply_ubatch()`; it keeps the exact C++ exception boundary from `40ea8d93b`. |
| `src/llama-kv-cache.h` | `5c8099c830c0f928f2946c77b16b9509f06638dbb879ed75ca27a5bd03cbb775` | `beaa815a0ea1638c6ef9b94b6e958586557cfb1270f40a8092f8a1b2ae0a5e9e` | `e126dbf3f`: exposes the inline `restore_cells()` helper. The context copy/move deletion from `24d086a80` is retained unchanged. |
| `tests/test-tp5-ubatch-pool.cpp` | `bac8b583b2794ad9850a1c59195fe6a12a1acfd4252ab46f2eb5c97b88f06b34` | `ffd14c9077c1e690653bbb0a9f15e4a94128b3661e7e41822ac90ad3537d2326` | `e126dbf3f`: adds `test_restore_cells_multi_stream()` with stream orders (3,1) and (1,3), two reverse-ordered trial placements, and a find_slot-failure-shaped restore. |

Verification: the source branch reported the narrow `test-tp5-ubatch-pool`
check passing; it was not rerun here because the resolution changed only
comments and manifest/waiver text. The three actual file digests match the
table, and `python3 fork/check-upstream.py` still fails first at
`ggml/src/ggml-cuda/mmvf.cu`, the pre-existing Section A red item; no new
mismatch and no historical red item was re-signed.

Known uncovered boundary: the added test calls `restore_cells()` directly
with synthetic cells; it does not execute `prepare_reference()` or
`prepare_into()` through a real multi-stream `llama_decode()`. The reachable
case is `kv_unified=false` with non-contiguous, out-of-order sequence ids
(for example {2,3}); the unified path keeps `sinfo.strm[s] == s` and does not
exercise the index mismatch.

### B5. Staging version and reachability round 2026-11-09

Base `e4b838303` (`feat/pillar5-refine-20261109`). Cherry-picks
`97e824de2` -> `bfc762daa` -> `8b6c28403` from
`feat/pillar5-refine-attrib`. The three commits applied without textual
conflicts. The target-side Meta rejection aggregation still had to be
checked: the registration function keeps the shared `candidate_decision`
path, rejects a short table after reading only `abi_version`, does not
replace an existing registration, and reports `existing registration
unchanged` versus `no prior registration (staging unregistered)`. The dead
`no_slot_quarantined` / `no_slot_other` counters are gone from the pool,
the dump and the tests.

The automatic merge left the six staging manifest digests at their old
values. A whole-file digest cannot sign a partial change: four of the six
files were already stale against the parent manifest at `e4b838303`, so
recording their final digest would hide that pre-existing difference. The
table separates the two proven cases. `staging.cu` and the Meta partial
test matched their parent manifest digests; their final digests are signed
for this round. `docs`, `ggml-backend-meta.cpp`, `staging.h` and the pool
test did not match at `e4b838303`; their parent manifest digests are
restored and remain red items. No other entry is touched, and no
historical red item is re-signed.

| path | parent manifest at e4b838303 | actual at e4b838303 | final hash | disposition |
|---|---|---|---|---|
| `docs/tp5-staging-submission.md` | `9b5a226c66d9e76317dae10b06849d13e35448b55d2d3910c5dc6e2f1b3e8bca` | `e73853524455aed75f703a64056f6500e69959cba1e7174b7b274771d08c6905` | `e31dca0f0df99096408c8a68b8470b862e2627a35a8113708b773a7e0ac91713` | parent stale; keep the parent digest as a red item |
| `ggml/src/ggml-backend-meta.cpp` | `30086f0db3ed3d29be3652eef950fa850e6ccaa9fef561715ca93444aa5134a3` | `59192e302292197dd85921fb852bc4922c66ed173b73ebc2d4504aeba5b51aad` | `02b3de5a50b7f187bbc5f9fb3bd87dd23d9f4b797db7a3861bed6acf53174254` | parent stale; keep the parent digest as a red item |
| `ggml/src/ggml-cuda/staging.cu` | `06703caf8deb82f13406b3b30d086b63ee0d82da063e388214f7cb0b14eea2d0` | `06703caf8deb82f13406b3b30d086b63ee0d82da063e388214f7cb0b14eea2d0` | `130f2192a26ceec1bb6103a1ba40d879b13b0c39361242646be5ffe0360581a6` | parent matched; sign the final digest |
| `ggml/src/ggml-cuda/staging.h` | `2726b628aebec43b200298e7c9605509abb2836d1e759122d22969f2c1afb4ec` | `5ee59204e738201b10cd9e51eebbea6b5afcca089d623afff6a7fcb2072b110b` | `afccd87ef5ab7429663b45ae623cebcf0d54a4d82c179785d3e81c98d428875b` | parent stale; keep the parent digest as a red item |
| `tests/test-tp5-staging-meta-partial.cpp` | `96f27f3ce1ebc2f1f8782f6c2c6137f9735ca024b10b0f7d8bf0308ed4618664` | `96f27f3ce1ebc2f1f8782f6c2c6137f9735ca024b10b0f7d8bf0308ed4618664` | `d46e164fc08c907902df46779e81bd474cdb062db86a5c26ee46c19c2dd909c3` | parent matched; sign the final digest |
| `tests/test-tp5-staging-pool.cpp` | `aa3b45e969a6163e2434ddb5d147396fe41e8cc2926452ca5e9243660be580b6` | `43accbb88d30a0ede86611b42a585b36d5ffa6cccad86dd427a3b73bc067ea6a` | `39c226c9b9517582dd360bb1ae39bab16e5c9c0b5cafd90c68c375f6bacfdd36` | parent stale; keep the parent digest as a red item |

The four restored entries are expected to remain mismatches against the
final files; that records the pre-existing drift instead of blessing it.
The two signed entries are the only manifest changes in this round. No
manifest path is new, so there is no new-file registration.

Verification: the parent-match column comes from `git show
e4b838303:<path>` against the parent manifest, and the final hashes are
the current files. JSON is valid. The source branch reported the
short-table ASan check and the staging pool CPU harness passing; neither
was rerun here. The two test files and the registration function are
byte-identical to the source branch's final versions; the only
`ggml-backend-meta.cpp` difference from that branch is the target-side
rejection aggregation block. The known guard was not rerun after restoring
the four red items; the earlier B4 run first failed at
`ggml/src/ggml-cuda/mmvf.cu`, and that entry is unchanged.

Known uncovered boundary: no target-side compile or ASan/CPU rerun, so the
target's aggregation block and its interaction with the registration are
verified statically, not by execution. The version check is only exercised
for the short v1/v2 tables by the source test; no real mixed v2/v3 HIP
provider was exercised. GPU/HIP staging was not run.

### B6. MTP batch pool and batch-caller migration round 2026-11-09

Base `393d917` (`feat/pillar5-refine-20261109`). Cherry-picks
`410ffb70c` -> `390556d62` -> `07e19d229` -> `d7d2cf87a` from
`feat/pillar5-refine-mtp-pool`. Two files conflicted:
`fork/tp5-compose/tests/test-host-pool.cpp` (both sides inserted tests at
the same points; keep the piece-scratch test and every MTP/failure test,
and restore the function brace git moved outside the conflict) and
`fork/tp5-compose/CMakeLists.txt` (keep `TP5_TEST_VOCAB_DEFAULT` and merge
the `llama` + `${CMAKE_DL_LIBS}` link). The other 62 files merged
automatically.

Manifest disposition. `include/llama.h` and `src/llama-batch.cpp` matched
their parent manifest digests at `393d917`; their final digests are signed
after review. The three A entries below also matched at `393d917`, but the
four commits changed their files; they keep the parent digest as red items
and were not re-signed.

| path | parent manifest at 393d917 | current actual | disposition |
|---|---|---|---|
| `src/llama-context.cpp` | `0d758756c2aef1947a8d92f372b13346350052f87d352ef48397876fd3ebfa84` | `cd2c18cb4997c9b5d30e4d10429957977aecee5114ee0e461c7e9dc579771aae` | keep parent digest; red item; not re-signed |
| `tests/test-tp5-prefab-subenvelope.cpp` | `37a4d9d30f7b171eeca3157f78261e44018f4a886e7b2784a8e8d2ae8f5683c6` | `36d3c3f049c8e594f39bdf0330ecc5db147c2eda82d7d0b5512332eae09db2e3` | keep parent digest; red item; not re-signed |
| `tools/server/server-context.cpp` | `a5892b7e9101f08a05769387758e736c038bc22fd70afad43d9b07f013041e2d` | `28518ef8ed529962a82d55cf04d9a39c737493b0c40307e33d4e859c7d1affc6` | keep parent digest; red item; not re-signed |

| path | parent manifest at 393d917 | final hash |
|---|---|---|
| `include/llama.h` | `58840cdf4f40fb0b3e912a7b2817d4a7ceaf28a180f5b6ddcda46212b588c4ad` | `db1c349d56e1e7a5d8b7efdb8f50a56fe076f500c5dbefb45311b422080d2585` |
| `src/llama-batch.cpp` | `4519681d562c9702bbb0e69ba61b4917407a5250b994bcb32142696be1fcd815` | `42468b792b218a0ac33f2cfe7607a8d26d87979a8c107c070a38a32ae0e74bd7` |

The remaining 58 changed paths have no manifest entry. They are listed
below as new/unregistered red items; no entry was added, and no historical
red item was re-signed. The B4/B5 staging red items (`docs`, `meta.cpp`,
`staging.h`, `test-pool`) and the Section A `mmvf.cu` drift are unchanged.

`common/` (1):

- `common/speculative.cpp`

`examples/` (12):

- `examples/batched/batched.cpp`
- `examples/diffusion/diffusion.cpp`
- `examples/embedding/embedding.cpp`
- `examples/llama.android/lib/src/main/cpp/ai_chat.cpp`
- `examples/lookahead/lookahead.cpp`
- `examples/lookup/lookup.cpp`
- `examples/parallel/parallel.cpp`
- `examples/passkey/passkey.cpp`
- `examples/retrieval/retrieval.cpp`
- `examples/speculative-simple/speculative-simple.cpp`
- `examples/speculative/speculative.cpp`
- `examples/turboanchorkv/turboanchorkv.cpp`

`fork/` (12):

- `fork/batch-compose/src/main.cpp`
- `fork/rerot-compose/src/native.cpp`
- `fork/tests/a-seams/recurrent-lifecycle-test.cpp`
- `fork/tests/a-seams/rerot-attention-oracle.cpp`
- `fork/tests/a-seams/rerot-native-trace.cpp`
- `fork/tests/a-seams/rerot-reader-views.cpp`
- `fork/tp5-compose/CMakeLists.txt`
- `fork/tp5-compose/README.md`
- `fork/tp5-compose/src/host_pool.h`
- `fork/tp5-compose/src/runtime.cpp`
- `fork/tp5-compose/tests/fail-malloc.c`
- `fork/tp5-compose/tests/test-host-pool.cpp`

`tests/` (27):

- `tests/nkv-cpu-probe/nout_runtime_probe.cpp`
- `tests/nkv-cpu-probe/pad_lane_ab_gate.cpp`
- `tests/nkv-cpu-probe/padmax_cpu_probe.cpp`
- `tests/nkv-cpu-probe/union_varlen_e2e.cpp`
- `tests/nkv-cpu-probe/varlen_output_probe.cpp`
- `tests/rsz/graph_reuse_probe.c`
- `tests/test-backend-sampler.cpp`
- `tests/test-fused-qkv-refusal.cpp`
- `tests/test-llama-archs.cpp`
- `tests/test-nkv-exact-width.cpp`
- `tests/test-nkv-runtime-param.cpp`
- `tests/test-ntok-bind-replay.cpp`
- `tests/test-ntok-cross-matrix.cpp`
- `tests/test-ntok-onegraph.cpp`
- `tests/test-ntok-strict.cpp`
- `tests/test-recurrent-state-rollback.cpp`
- `tests/test-rsz-hybrid-compose.cpp`
- `tests/test-rsz-multiseq-decode.cpp`
- `tests/test-rsz-nkv-rsz-topology.cpp`
- `tests/test-rsz-runtime-param.cpp`
- `tests/test-rsz-union-combo.cpp`
- `tests/test-save-load-state.cpp`
- `tests/test-state-restore-fragmented.cpp`
- `tests/test-topology-key-sweep.cpp`
- `tests/test-union-binder-rsz.cpp`
- `tests/test-union-binder-stock-equiv.cpp`
- `tests/test-union-four-param-matrix.cpp`

`tools/` (6):

- `tools/batched-bench/batched-bench.cpp`
- `tools/imatrix/imatrix.cpp`
- `tools/mtmd/mtmd-cli.cpp`
- `tools/mtmd/mtmd-helper.cpp`
- `tools/perplexity/perplexity.cpp`
- `tools/results/results.cpp`

Contract checks: `llama_batch_is_complete()` documents that `n_tokens`
must be the capacity passed to `llama_batch_init()` (or a smaller number
of initialized `seq_id` entries); a larger value reads past `seq_id`.
The inspected call sites pass the same capacity expression used at init;
the pool passes `1` for `llama_batch_init(1, wide, 1)`, and no inspected
call passes a larger running count. `llama_batch_init()` initializes the
outer `seq_id` array with a null sentinel, checks every allocation and
size multiplication, and returns null fields on failure;
`llama_batch_free()` stops at the first null `seq_id` entry. The pool
discards a partial batch, leaves the slot unready, and retries; the
fallback token is freed once. CMake enables C, builds the test-only
`fail-malloc` interposer, runs the test through `cmake -E env` with
`LD_PRELOAD` and `TP5_REQUIRE_FAIL_MALLOC=1`, and keeps the existing
`TP5_TEST_VOCAB_DEFAULT` define and `TQText::text` link.

Verification: the narrow CPU check built `tq_tp5_host_pool_tests` and
`tp5_fail_malloc` from this tree against the freshly rebuilt `libllama`
(build commit `0804b304b`), loaded with `LD_PRELOAD`. Normal run: 73
`PASS`, 0 `FAIL`, 3 `SKIP`, exit 0. With the fail-malloc interposer and
`TP5_REQUIRE_FAIL_MALLOC=1`: 112 `PASS`, 0 `FAIL`, 0 `SKIP`, exit 0. The
source branch's earlier normal 67 `PASS` / 3 `SKIP` and interposer 103
`PASS` / 0 `FAIL` were not repeated as the same check. The guard was not
rerun; the red items above remain, so this integration is not clean.

Known uncovered boundary: no full link/build of every migrated caller, no
GPU/full-suite run, and no execution of the 58 unregistered paths or the
three red A files beyond static review. The narrow check covers the
host-pool test and its CMake wiring only. A future caller that passes a
value larger than the init capacity would still read past `seq_id`,
because `llama_batch` does not store the capacity.

### B7. Synchronous Meta source identity round 2026-11-09

Base `2dda3c6fa`. Cherry-picks `0969d889a` -> `d384aaae1` from
`feat/pillar5-refine-sync-src-identity`. One code conflict remained in the
`GGML_BACKEND_SPLIT_AXIS_MIRRORED` eligibility hunk of
`ggml/src/ggml-backend-meta.cpp`: the target's v3/rejection-aggregation
comment and the source's `staging_meta` shape check touched the same lines.
The resolution keeps the destination-copy warning and adds the source
identity path. The test file conflicted only on the fake-table comment; the
v3 comment and `kind_count = kKindCount` are kept. Docs and CMake merged.

The source branch's fake table was v2; the integrated
`tests/test-tp5-staging-sync-src.cpp` now initializes the v3 `kind_count`
and describes the table as v3. A destination view
(`dst->view_src != nullptr`) is rejected by the `cpy_tensor` preflight and
falls back to `set_tensor` at its offset. A fatal or partial submission is
sticky: `ggml_backend_tensor_copy` is `void`, the public
`submission_pending`/`submission_error` state carries the fault, the next
graph compute returns `GGML_STATUS_FAILED`, and the blocking fallback never
replays the destination.

Manifest disposition: no digest was updated or added. The five changed
paths are red items; the existing B5/B6 entries for `docs` and
`ggml-backend-meta.cpp` were already stale and stay at their recorded
digests. `ggml-backend.cpp` matched its parent manifest before this round
but is not signed now; `tests/CMakeLists.txt` was already stale; the new
test has no entry.

| path | parent manifest at 2dda3c6fa | current actual | disposition |
|---|---|---|---|
| `docs/tp5-staging-submission.md` | `9b5a226c66d9e76317dae10b06849d13e35448b55d2d3910c5dc6e2f1b3e8bca` | `d7a8d24f07b932d043ff78e09a483000319834b99a5ff7445a4e61dc96fede20` | already-stale red item; not re-signed |
| `ggml/src/ggml-backend-meta.cpp` | `30086f0db3ed3d29be3652eef950fa850e6ccaa9fef561715ca93444aa5134a3` | `6e1f907ad24161cb25c1d643318d9952fef085a9af8baa645362256dbdf9a158` | already-stale red item; not re-signed |
| `ggml/src/ggml-backend.cpp` | `325d55028535ca47823cf7ceaa95f185101857cd11b8549b73c70536d249cfcb` | `20ffa660f6db39485d24b1c22a22117fb03966f5b02d91b6cafb21331151b45e` | parent-matched, now red; not signed |
| `tests/CMakeLists.txt` | `9a94fd693d250398dd67475384024d4f24d5f721eef206c8a64d1e82dc4f4bca` | `10430b1d117a9d7f62dcfcdb019abfaa97aaee0b94f2992e2fc11afd425752e2` | already-stale red item; not re-signed |
| `tests/test-tp5-staging-sync-src.cpp` | no entry | `6cea0b6b4533b46b8c19ea14994ac12cb8498b0e0c5e7d8093b88273986301af` | new/unregistered red item |

Verification: built `test-tp5-staging-sync-src` from this tree against the
integrated `ggml` and ran it once. Result:
`TP5_STAGING_SYNC_SRC_TEST PASS`, exit 0. No GPU, full suite, other
staging tests, or guard rerun.

Known uncovered boundary: the test uses the fake staging table and CPU Meta
children; it does not exercise a real HIP provider, the async lane, or a
full scheduler/model run. The other staging tests were not rerun. The guard
was not rerun; the red items above remain, so this integration is not clean.

### B8. Relay polish v3 manifest reconciliation 2026-11-09

Scope: only the five relay manifest entries below were re-signed. No source
file, test, or other manifest entry was changed.

Evidence chain from `agent://polish-manifest-reconcile` (t2):

- At `d401ea986` all five manifest entries matched the actual files and
  mode `100644`; the report lists the last manifest sign for each path and
  no unsigned change between that sign and `d401ea986`.
- Between `d401ea986` and HEAD only `5a34498fd` touches these five files.
  This round verified that `git log 5a34498fd..HEAD -- <five paths>` is
  empty and `git diff 5a34498fd HEAD -- <five paths>` is empty.
- For each path the HEAD git blob equals the v3 patch post-image blob, so
  the current bytes are the reviewed patch output, not a later edit.
- The exported v3 patch artifact
  `/home/kunweiz/work/r28b-relay-polish-export-20261109T101312Z/relay-polish-v3.patch`
  has SHA256 `5bc1ad65f9f5cb2c29932c358e548a9e955dac4ec8972ea1f9d30b39e9a66468`,
  matching the commit message and sidecar. The report records apply-check,
  apply, reverse-check, byte-identical roundtrip, `test-relay-plan-policy`
  (normal and ASan+UBSan), `test-relay-verify-counters`,
  `test-shape-cache`, and `hipcc gfx1030` compile receipts.

| path | d401 manifest | new signed sha256 | HEAD / post-image blob |
|---|---|---|---|
| `ggml/include/ggml-relay-plan.h` | `73220f73122e0343b7261c28f0f92a0769863426cb9180b9f6f97b337709d97f` | `296a0257682c3d7844fd6ccef4a6eb891b82ec3c25c0c492cb71c3afc985142d` | `1f9ceffd42c7769410b2f50b065382f0fa9b8fba` |
| `ggml/include/ggml-relay.h` | `6515be02a00f7755e88b7b5fc7a247e18d96e2e34c508a409ae81dbfed2019c6` | `41f97fa931e5a823261b329bb29d672ea232ae34d5790b7b8ed974a906b427eb` | `69dfb15173156e18c8d3f51c9f681877a6bb42b6` |
| `ggml/src/ggml-cuda/relay.hip` | `600d494e7c6f8c7360b9eab6ff96406a107c0e351c37abaaa92083cba8e8b259` | `6f608d139b422a5cf39397b6ea8ed81d6bbbd8aa7a78adb8dfefd4fb759242a0` | `af6eca18bfff7c2b2f89a6cb18910a5145a1bbae` |
| `patches/ws5/README.md` | `3d91142be018064d463453f794225e283909eccc75d5dba26bf8d1f3e4a17c4d` | `04df9c7459d371f22cc140982697fd803bca6dd8149a153ae3a4a4b34afb7f8c` | `9b99636f7008fba1b9b5a75544e143b3d9439802` |
| `tests/test-relay-plan-policy.cpp` | `e1b0332d786946d9e0cbbdb95cd983625a2ee53529e4b37289055f1491498119` | `4d849f8de99a5e0364cea495d6e5428e7d2cf5abdc10ab9b37007a3855f39efc` | `0d955dfb8f46ae2d484eb0b299690d0df65df8c1` |

This is patch-level span review, not per-file runtime qualification.
`test-relay-plan-policy.cpp` has a direct test receipt; `relay.hip` has a
`hipcc` compile receipt; `ggml-relay-plan.h`, `ggml-relay.h` and
`patches/ws5/README.md` rely on the whole-patch review and their consumers.
Do not call these five entries runtime-qualified.

Remaining unresolved mismatches: this round does not claim the guard
passes and did not rerun it. All other manifest debt recorded in B5-B7
remains, including the staging red items (`docs`,
`ggml-backend-meta.cpp`, `staging.h`, `test-tp5-staging-pool.cpp`), the
three A red items (`src/llama-context.cpp`,
`tools/server/server-context.cpp`, `tests/test-tp5-prefab-subenvelope.cpp`),
the B6 MTP paths (including the 58 unregistered paths), the B7 paths
(`ggml-backend.cpp`, `tests/CMakeLists.txt`,
`tests/test-tp5-staging-sync-src.cpp`), and the pre-existing Section A
drift (`mmvf.cu` and the other listed A paths). No entry outside the five
rows was re-signed.

Verification: JSON is valid; the five actual file SHA256 values match the
new entries; the diff is exactly five SHA256 lines. The guard was not
rerun; no GPU or full suite was run.

### B9. Batch allocr coupling cleanup and split reserve round 2026-11-10

Scope: `src/llama-batch.cpp`, `src/llama-batch.h` and
`tests/test-batch-alloc.cpp`. The three manifest entries are re-signed for
this round; no other entry is rewritten.

`llama_batch_allocr::clear()` used to clear all 256x256 coupling bits on every
batch, and the memory divergence check scanned the full matrix even when the
batch had no multi-sequence token. The allocator now records the
`(dependent, primary)` pairs whose `seq_cpl` bit is set, clears only those
bits, and resets `has_cpl` in `clear()`. The per-sequence position and index
state is still cleared in full. The divergence check runs only when `has_cpl`
is set. `split_simple()` reserves `min(n_ubatch, remaining tokens)`, the exact
upper bound of its chunk.

Invariants kept:

- `seq_pos` is fully cleared at the start of `init()`; after it, sequence `s`
  has entries iff it appears in the current batch.
- Every `seq_cpl` bit is recorded exactly once in `seq_cpl_pairs`, and the
  pair is recorded before its bit is set. A failed `emplace_back` therefore
  leaves no untracked bit, and any pair recorded by a failed `init()` is
  cleared by the next `clear()`.
- `has_cpl` is false after `clear()` and true iff at least one pair was
  recorded.
- `split_simple()` behavior is unchanged; `used` is still required to be a
  prefix of true entries, which its callers preserve.

| path | old recorded | new signed sha256 |
|---|---|---|
| `src/llama-batch.cpp` | `42468b792b218a0ac33f2cfe7607a8d26d87979a8c107c070a38a32ae0e74bd7` | `e21f7dea65beba1cb980e69987d29f44b7917c2da202d127f3cd465efd4c2d19` |
| `src/llama-batch.h` | `e0440479841dedd26b6e89878dc67fae0e9592955eaf50662981b5eb56c7f144` | `a8884a5b545bc235b3ee8a76f135d428182427237a469089cf49c17cc1f32e89` |
| `tests/test-batch-alloc.cpp` | `061661c773d486eaa3ecc91f2a45e6a166656e489040ba6aa12eb7e46d85d190` | `af28ae435a5f6ec972e81c9e6d1ba6528780204d09b01ecbff04840537565475` |

New tests: `init.coupled_bits_cleared_between_batches` (a stale pair from an
accepted batch and from a rejected coupled batch must not reject a later
uncoupled or differently coupled batch), and
`split.split_equal_sequential_after_coupled_batch` (`has_cpl` does not leak
across batches). `init.coupled_multiple_pairs_cleared_between_batches` records
two different pairs in one batch and then makes both dependent sequences
diverge, so a `clear()` that drops either recorded pair is rejected. The
former `init.seq_pos_cleared_for_absent_sequences` test was dropped: `clear()`
still resets `seq_pos` in full, but that path predates this round and the test
exercised none of the changed code.

Verification: the initial round built only `test-batch-alloc` in a fresh
CPU-only Ninja build (`LLAMA_BUILD_TESTS=ON`, `LLAMA_BUILD_COMMON=ON`,
CUDA/HIP off) and ran it once: 44 tests, 474 assertions, 0 failures,
0 exceptions, exit 0. The exception-safety repair swaps one test, so the tree
still registers 44 tests; the assertion total changes and was not re-measured.
DevOps runs the single CPU verification of the repaired tree. No GPU, full
suite, benchmark or speed claim. The guard was not rerun; all debt recorded in
B5-B8 and Section A remains.

Final repaired-tree verification: a fresh CPU-only build in
`/tmp/tmp.Tpt83qyFof` ran `test-batch-alloc` once: 44 tests, 470 assertions,
0 failures, exit 0. `bad_alloc` was not directly injected. No GPU. The four
uncommitted staging files were not exercised; their runtime remains
unverified.

## C. Pre-existing undeclared A paths

172 paths remain undeclared after registering the WS2/WS3/WS4/WS5 paths.
All predate this integration and none is modified to make a digest match.

### Other pre-existing A paths (6)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `AUDIT-INVENTORY.md`
- `COMPILE-AUDIT.md`
- `EXIT-CODE-AUDIT.md`
- `SYNC.md`
- `docs/backend/DUALWIN-COMPOSE-EVENT.md`
- `ggml/src/ggml-backend-impl.h`

### Pillar-1 NKV/union probe and gate scripts (35)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `tests/nkv-cpu-probe/README.md`
- `tests/nkv-cpu-probe/check_probe.py`
- `tests/nkv-cpu-probe/compare_records.py`
- `tests/nkv-cpu-probe/gate-kernel-sel-len-13022.sh`
- `tests/nkv-cpu-probe/gate-nkv-observe-13022.sh`
- `tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh`
- `tests/nkv-cpu-probe/gate-p1-ctest-pack-13022.sh`
- `tests/nkv-cpu-probe/gate-p1-merge-13022.sh`
- `tests/nkv-cpu-probe/gate-verdict.sh`
- `tests/nkv-cpu-probe/nkv-exact-width-report.py`
- `tests/nkv-cpu-probe/nout_runtime_probe.cpp`
- `tests/nkv-cpu-probe/one-shot-cpu-13022.sh`
- `tests/nkv-cpu-probe/pad_lane_ab_gate.cpp`
- `tests/nkv-cpu-probe/padmax_cpu_probe.cpp`
- `tests/nkv-cpu-probe/reproduce-ntok-13022.sh`
- `tests/nkv-cpu-probe/reproduce_nout_tail_cpu.sh`
- `tests/nkv-cpu-probe/run-nkv-exact-width-arms-13022.sh`
- `tests/nkv-cpu-probe/run-triparam-close-13022.sh`
- `tests/nkv-cpu-probe/run_ntok_cross_matrix_13022.sh`
- `tests/nkv-cpu-probe/run_ntok_onegraph_13022.sh`
- `tests/nkv-cpu-probe/run_pad_lane_ab_gate_13022.sh`
- `tests/nkv-cpu-probe/run_probe.sh`
- `tests/nkv-cpu-probe/run_union_ctest_gate_13022.sh`
- `tests/nkv-cpu-probe/run_union_varlen_e2e_13022.sh`
- `tests/nkv-cpu-probe/run_varlen_output_probe.sh`
- `tests/nkv-cpu-probe/summarize.py`
- `tests/nkv-cpu-probe/test_compare_records.py`
- `tests/nkv-cpu-probe/test_nkv_exact_width_report.py`
- `tests/nkv-cpu-probe/test_union_ctest_gate.py`
- `tests/nkv-cpu-probe/union-gate-verdict-selftest.sh`
- `tests/nkv-cpu-probe/union-mainline-gate-r24o.sh`
- `tests/nkv-cpu-probe/union_arm.hpp`
- `tests/nkv-cpu-probe/union_ctest_gate.py`
- `tests/nkv-cpu-probe/union_varlen_e2e.cpp`
- `tests/nkv-cpu-probe/varlen_output_probe.cpp`

### Pillar-1 union/NKV/RSZ CPU tests (32)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `tests/test-cpu-attn-mask.cpp`
- `tests/test-cpu-attn-pad.cpp`
- `tests/test-cpu-kernel-sel-len-indep.cpp`
- `tests/test-flash-attn-bias.cpp`
- `tests/test-fused-qkv-refusal.cpp`
- `tests/test-gdn-ingredient-replay.cpp`
- `tests/test-nkv-exact-width.cpp`
- `tests/test-nkv-runtime-param.cpp`
- `tests/test-ntok-bind-replay.cpp`
- `tests/test-ntok-cross-matrix.cpp`
- `tests/test-ntok-onegraph.cpp`
- `tests/test-ntok-replan.cpp`
- `tests/test-ntok-strict.cpp`
- `tests/test-rs-ring-relay.cpp`
- `tests/test-rsz-hybrid-compose.cpp`
- `tests/test-rsz-hybrid-selection-write.cpp`
- `tests/test-rsz-key-injection.cpp`
- `tests/test-rsz-lane-rows.cpp`
- `tests/test-rsz-multiseq-decode.cpp`
- `tests/test-rsz-nkv-rsz-topology.cpp`
- `tests/test-rsz-relay-onegraph.cpp`
- `tests/test-rsz-runtime-param.cpp`
- `tests/test-rsz-selection-write.cpp`
- `tests/test-rsz-union-combo.cpp`
- `tests/test-topology-cache-key.cpp`
- `tests/test-topology-hash.cpp`
- `tests/test-topology-key-sweep.cpp`
- `tests/test-union-binder-gate.cpp`
- `tests/test-union-binder-rsz.cpp`
- `tests/test-union-binder-stock-equiv.cpp`
- `tests/test-union-four-param-matrix.cpp`
- `tests/test-union-shape-guard.cpp`

### Pre-existing CPU attention/SIMD A overlays (10)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `ggml/src/ggml-cpu/CMakeLists.txt`
- `ggml/src/ggml-cpu/attn-ordered.cpp`
- `ggml/src/ggml-cpu/attn-ordered.h`
- `ggml/src/ggml-cpu/ggml-cpu-attn-stats.h`
- `ggml/src/ggml-cpu/ggml-cpu-impl.h`
- `ggml/src/ggml-cpu/ggml-cpu.cpp`
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp`
- `ggml/src/ggml-cpu/llamafile/sgemm.h`
- `ggml/src/ggml-cpu/vec.cpp`
- `ggml/src/ggml-cpu/vec.h`

### Pre-existing CUDA/HIP A overlays (1)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `ggml/src/ggml-cuda/cpy.cu`

### Pre-existing GGML public-header A overlays (3)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `ggml/include/ggml-alloc.h`
- `ggml/include/ggml-cpu.h`
- `ggml/include/tp5-meta-submit-ready.h`

### Pre-existing TP5 model A overlays (5)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `src/models/delta-net-base.cpp`
- `src/models/mamba-base.cpp`
- `src/models/models.h`
- `src/models/rwkv6-base.cpp`
- `src/models/rwkv7-base.cpp`

### Pre-existing TP5 prefab/union/topology A overlays (14)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `src/llama-nout-state.cpp`
- `src/llama-nout-state.h`
- `src/llama-prefab-env.h`
- `src/llama-prefab-hotpath.h`
- `src/llama-prefab-iface.h`
- `src/llama-prefab-mtp.cpp`
- `src/llama-prefab-splice.h`
- `src/llama-prefab.cpp`
- `src/llama-prefab.h`
- `src/llama-topology-hash.h`
- `src/llama-union-binder.h`
- `src/llama-union-shape-guard.h`
- `src/llama-union-topology.h`
- `src/llama-union-width-table.h`

### Pre-existing recurrent/model-loader A overlays (2)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `src/llama-memory-hybrid-idx.cpp`
- `src/llama-memory-hybrid-idx.h`

### RSZ/scheduler probe tests (4)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `tests/rsz/graph_reuse_probe.c`
- `tests/rsz/make_tiny_rsz_mamba.py`
- `tests/scheduler-split-copy/CMakeLists.txt`
- `tests/scheduler-split-copy/test-scheduler-split-copy.cpp`

### Repository metadata (1)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `tests/.gitignore`

### TP5 audit/diagnostic evidence (14)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `audit/AUDIT-AND-VERIFICATION.md`
- `audit/m3-tail-storage.patch.txt`
- `audit/verify/VERIFY-M3-ROWS-G2-20261101.txt`
- `audit/verify/VERIFY-M3-ROWS-G3G4-20261101.txt`
- `diag/m3-locate-divergence.py`
- `diag/m3-n1-rowref.py`
- `diag/m3-split-trunk-mixer.py`
- `diag/m3-tail-stamp-diff.py`
- `diag/m3-tail-stamp.ggml-cuda.diff`
- `micro/build_mixer_n1ref.py`
- `micro/build_mixer_prod.py`
- `micro/m3-mixer-n1ref.cpp`
- `micro/m3-mixer-prod-path.cpp`
- `micro/m3-probe-hc-mix.cpp`

### WS0 async-protocol overlay evidence (45)

Source: pre-existing at the integration base; not touched by WS2/WS3/WS4/WS5.
Non-blocking: manifest debt only; the build and CPU tests do not depend on this digest.

- `overlay/mtp-async-protocol-candidate-20261027/PROVENANCE.md`
- `overlay/mtp-async-protocol-candidate-20261027/STABLE-FOR-FREEZE.md`
- `overlay/mtp-async-protocol-candidate-20261027/archive-complete.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/base/ggml-backend-meta.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/base/ggml-backend-meta.ldl.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/base/runtime.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/base/runtime.ldl.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/base/tp5-meta-submit-ready.h`
- `overlay/mtp-async-protocol-candidate-20261027/build-receipt.json`
- `overlay/mtp-async-protocol-candidate-20261027/build.py`
- `overlay/mtp-async-protocol-candidate-20261027/candidate.json`
- `overlay/mtp-async-protocol-candidate-20261027/command-template-NOT-RUN.json`
- `overlay/mtp-async-protocol-candidate-20261027/compile-meta.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/compile.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/dw-dev-overlap.h`
- `overlay/mtp-async-protocol-candidate-20261027/dw-thru.h`
- `overlay/mtp-async-protocol-candidate-20261027/dw_device_wait.h`
- `overlay/mtp-async-protocol-candidate-20261027/dw_event_trace.h`
- `overlay/mtp-async-protocol-candidate-20261027/ggml-backend-meta.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/handoff-micro.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/headers/README.md`
- `overlay/mtp-async-protocol-candidate-20261027/headers/base/ggml-backend.h`
- `overlay/mtp-async-protocol-candidate-20261027/headers/base/llama.h`
- `overlay/mtp-async-protocol-candidate-20261027/headers/ggml-backend.h`
- `overlay/mtp-async-protocol-candidate-20261027/headers/ggml-backend.patch`
- `overlay/mtp-async-protocol-candidate-20261027/headers/llama.h`
- `overlay/mtp-async-protocol-candidate-20261027/headers/llama.patch`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/PROVENANCE.md`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/control-link.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/hip-link.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay-compile.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay.hip`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay.ldl.hip`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay.ldl.patch`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay.orig.hip`
- `overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay.patch`
- `overlay/mtp-async-protocol-candidate-20261027/link-base.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/link.argv.json`
- `overlay/mtp-async-protocol-candidate-20261027/meta.ldl.patch`
- `overlay/mtp-async-protocol-candidate-20261027/meta.patch`
- `overlay/mtp-async-protocol-candidate-20261027/pin-micro.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/runtime.cpp`
- `overlay/mtp-async-protocol-candidate-20261027/runtime.ldl.patch`
- `overlay/mtp-async-protocol-candidate-20261027/runtime.patch`
- `overlay/mtp-async-protocol-candidate-20261027/tp5-meta-submit-ready.h`

## D. WS registrations added by this integration

| path | consumer |
|---|---|
| `ggml/include/ggml-relay-plan.h` | tq-tp5-compose pillar-5 plan cache and frame LRU (WS5), ggml relay plan-key provider ABI |
| `ggml/src/ggml-cuda/staging.cu` | tq-tp5-compose persistent pinned H2D staging pool (WS4) |
| `ggml/src/ggml-cuda/staging.h` | tq-tp5-compose persistent pinned H2D staging pool (WS4), WS4 staging pool CPU test |
| `tests/test-tp5-staging-pool.cpp` | WS4 staging pool CPU test |
| `tests/test-relay-plan-policy.cpp` | WS5 relay plan policy CPU test |
| `ggml/include/ggml-relay-verify.h` | WS5 relay verification counters (`verify_req/res/exec`), header-only |
| `tests/test-relay-verify-counters.cpp` | WS5 relay verification counter CPU test |
| `patches/ws5/ggml-backend-meta.cpp.patch` | integ/tp5-mtp-r28b integration maintainers |
| `patches/ws5/runtime.cpp.patch` | integ/tp5-mtp-r28b integration maintainers |
| `patches/ws5/ggml-backend.cpp.patch` | integ/tp5-mtp-r28b integration maintainers |

## E. Defaults and domain separation

- `ggml_relay_plan::config::graph_uid` defaults to `false`, but it does not gate the
  relay uid fast path. With the switch unset, `ggml_backend_sched_split_graph()` still
  assigns a uid to every graph whose uid is 0, and the relay provider receives the Meta
  backend's internal child graphs, whose uids are assigned on every Meta rebuild. The
  switch only changes when the llama-level parent graph receives its uid. The OFF leg
  keeps 2 frames, the 384 MiB pool, no plan cache and map channels; the relay uid fast
  path still applies unless `GGML_TP5_RELAY_VERIFY_UID=1` requests the describe. The ON
  leg must set `GGML_TP5_GRAPH_UID=1` explicitly.
- `GGML_TP5_RELAY_FRAMES` defaults to `2`; the ON leg must set `8` explicitly.
- `GGML_TP5_POOL_EXACT` defaults to `0`; the ON leg must set `1` explicitly.
- `GGML_TP5_PLAN_UNION` defaults to `0`; the ON leg must set `1` explicitly.
- WS5 `plan_key` has nine fields: the seven B fields, `channels`, and an internal
  terminal-binding digest; B `PlanKey` has seven fields, and the C boundary carries the
  eight caller-visible fields. They are separate domains. The C adapter maps them
  explicitly; the types are not merged.
  `patches/ws5/README.md` records the legacy defaults, the uid fast-path caveat and the
  two-domain mapping.

### E.1 Documentation-only round

The comment above `tp5_assign_graph_uid()` in `src/llama-context.cpp` was corrected
without a behavior change. That path is a pre-existing stale manifest item: the digest
recorded at the `0aca90f70` baseline does not match the committed file, so this round
does not re-sign the whole file and does not claim the old overlay is certified.

The three relay-owned documentation/test paths
(`ggml/include/ggml-relay-plan.h`, `tests/test-relay-plan-policy.cpp`,
`patches/ws5/README.md`) are re-signed against the audited `d401->67c` baseline. Their
signed digests record byte identity and the reviewed comment change, not runtime
qualification. `ggml/include/ggml-relay.h` and `ggml/src/ggml-cuda/relay.hip` were not
changed and keep the B8 signatures. No other manifest entry was updated, and the guard
was not rerun; the B5-B7 red items remain.

### E.2 Relay finish failure round 2026-11-09

Cherry-pick `0a6a0cba3`, independently reviewed as t19
(`agent://kernel6-review`). `finish()` now clears `sealed` when the plan-pin
release or the final `observe()` fails, so a true `finish()`/`submit` return
is the only continuation permit. A false return is the abort path; native
owners stay retained until `idle()` proves completion. The README
finish-contract section is merged with the existing graph_uid correction.

Manifest: only `ggml/include/ggml-relay.h`,
`ggml/src/ggml-cuda/relay.hip`, `ggml/src/ggml-relay-impl.h` and
`patches/ws5/README.md` were re-signed. Each matched its manifest before the
merge and the t19 review covers the changed span. The E.1 note that
`ggml-relay.h` and `relay.hip` kept the B8 signatures described the
documentation-only round; this finish round changes and re-signs them.
`src/llama-context.cpp` remains at its stale `0aca90f70` digest; no other
entry was updated.

Verification boundary: the real `observe()` failure and
`plan_cache.release` failure branches were not dynamically executed. There
was no HIP fault injection. The HIP compile and the existing CPU
policy/verify/shape tests each cover only their own scope; they do not prove
the runtime failure branches. t19 also records the residual P3 window where
an asynchronous abort arrives after `observe()` and before the final failed
check, and recommends a pure CPU predicate test; that test is not added here.
The guard was not rerun; no GPU or full suite was run.

### E.3 MTP target checkpoint round 2026-11-09

Cherry-pick `62c700ac9`, independently reviewed as t7
(`agent://mirror-port-review`). The checkpoint replaces the
`clear()` + `resize()` vector with retained storage, an explicit snapshot
size and a valid flag. The non-native-trim verify path captures at epoch
start, restores a rejected prefix, invalidates on full acceptance and at the
next epoch start; the native-trim path never captures.

The test's only assertion of the vector `resize()` implementation detail
(`checkpoint: grow zero-initializes only the new tail`) was removed. The
remaining test exercises the actual `capture`/`restore`/`invalidate`
semantics: same-size retained bytes, growth prefix, shrink prefix, short
capture, short restore, exception unwinding, invalidation and zero size. The
integrated test has 27 checks, not the source commit's 28.

DevOps ran the checkpoint target once as an equivalent single-file compile
against the main tree at `e49cac490`: the target has no macros or external
libraries, so `c++ -std=gnu++17` with the `src` include reproduces its CMake
properties; the binary ran once with 27 PASS, 0 FAIL and exit 0. Commands and
output are recorded at `agent://kernel-devops-lock` (t9). This is a CPU
header-semantics check only, not a CMake/CTest wiring or integration run.

Runtime integration was not executed. No real MTP run, no HIP, no ASan/LSan,
no full suite and no guard rerun. The first capture in each service round and
any growth still zero-fill the new storage; same-size or shrink reuse avoids
that zero fill. The checkpoint is per service round, not retained across
rounds; its capacity is reused only within a service round.

Manifest: the five changed paths have no manifest entry and none was added.
No existing entry matched and was reviewed here, so no digest was updated;
the B5-B7 red items remain. The guard was not rerun.

### E.4 MTP greedy-cache unit check round 2026-11-09

The integrated tree at `a34ef6275` was checked once for the
`tq_tp5_mtp_greedy_cache_tests` target by an equivalent compile/run in an
exclusive directory: the test and `shortlist.cpp` were compiled separately
with the target's `cxx_std_17`, `src` include and `TQSampling` static link,
then linked and run once. Result: 32 PASS, 0 FAIL, exit 0. Commands and
output are recorded at `agent://kernel-devops-lock` (t10).

This is a cache-class unit test only. It does not compile or execute
`runtime.cpp` and does not cover the runtime wiring, cancellation/exception
paths, `no_sync_logits_calls`, `capture_logits`, a real model or end-to-end
run, GPU/HIP, or performance. None of the merged greedy-cache paths has a
manifest entry, so no digest was updated; the B5-B7 red items remain.

### E.5 Installed-consumer isolation round 2026-11-09

Cherry-pick `2903d14cd` -> `4bfae1eb7` -> `34cc1672f`, reviewed as t26
(`agent://kernel6-review`). The obsolete `tp5-meta-submit-ready.h` and the
wiring-only `tests/test-public-tp5-api.cpp` are absent from the final tree;
the checkpoint, T3 greedy-cache and relay changes remain. NextN set/ith get
are public in `include/llama.h`; the Meta submit hooks are `GGML_API` in
`ggml/include/ggml-backend.h`; the hybrid-index trace is a versioned public
query. `common/speculative.cpp` keeps `llama-ext.h` only for the remaining
private APIs (layer_inp, target_layer_ids, mtp_chain,
get_embeddings_nextn), not for the public NextN pair.

`patches/ws5/README.md` now marks its `git apply` commands as fixed-base
historical provenance, not applicable to the current tree; the patch bytes
are unchanged.

Manifest: `include/llama.h`, `ggml/include/ggml-backend.h` and
`patches/ws5/README.md` matched their baseline digests and were reviewed;
their digests are updated. `docs/tp5-staging-submission.md` and
`ggml/src/ggml-backend-meta.cpp` were already stale and keep their old
digests. No other path has an entry, and none was added.

Evidence scope: the installed `find_package`/symbol checks are temporary
staged evidence, not a repository CTest. The permanent behavior evidence is
the real single-sequence CPU qwen4exp trace test. GPU/HIP and Windows export
paths were not run. The guard was not rerun.

### E.6 SWA KV prepare-trial round 2026-11-09

Cherry-pick `e1112f844`, reviewed as t4 (`agent://audit-cpu-boundary`). The
pooled prepare skips the last trial only when `n_swa == 0` and
`swa_type == LLAMA_SWA_TYPE_NONE`; with SWA active every ubatch is trialled,
so `apply_ubatch()`'s `seq_rm` purge happens during prepare and matches
`prepare_reference()`. The real-cache 512/400/900 counterexample compares
reference and pooled cache state: cell 390 is purged, cell 400 is restored,
and the head returns to 399. The non-SWA control checks cache-state
equivalence only; it does not prove the last trial is skipped. The existing
exception boundary is unchanged: an allocation failure after an earlier apply
can unwind past `restore_cells`; it is not a no-throw rollback.

The candidate CPU test is included but was not run in this integration tree.
No GPU, full suite or guard rerun.

Manifest: `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h` and
`tests/test-tp5-ubatch-pool.cpp` matched their baseline digests and were
reviewed; their digests are updated. Other stale entries keep their old
digests.

### E.7 Legacy relay channel-map reclamation round 2026-11-09

Cherry-pick `3e65d33a1` -> `97fd640ec` -> `c3e8ce6fa` -> `4fc8cfcaf` ->
`0c3204879`, reviewed as t10 (`agent://mirror-port-review`). `bank_released`
and `no_reservations` are host-ledger facts only, not native-drain claims;
`held` and `not_ready` entries are retained, and a stale `INPUT_READY` is
rejected. The map uses the real TQRelay ledger API; no A dependency was
invented. The integrated tree at `e7d0a0d2` built and ran
`tq-tp5-relay-channel-map` against the current TQRelay source installed in
an exclusive prefix: 1/1 passed, exit 0. The first build against the existing
install prefix failed because that TQRelay lacked
`retirement()`/`retirement_state`; rebuilding TQRelay from the current
source tree and putting that prefix first made the target compile and the
test pass. Commands and output are recorded at `agent://safe-devops` (t2).
This is a host CPU narrow test only: it does not cover courier, GPU/HIP,
native drain, MTP, full-model or five-rank execution. The seven changed
paths have no manifest entry, so no digest was updated.

### E.8 MoE session policy and KV phase transition round 2026-11-09

Cherry-pick net effect `09044ca64` -> `8cfc3db76`, reviewed as t7
(`agent://audit-model-boundary`). The final net keeps the `09044ca` ON
semantics: an explicit `on` is a request; an ineligible model resolves to
`off`, a provider with no session is dormant, and the context continues with
`requested=... resolved=...` logging. The intermediate `7f6144cc6` hard
rejection is not part of this net. A recreated scheduler re-applies the
context's resolved mode/budget.

The prefab/phase exclusion restates the existing prefab eligibility
invariant (prefab rejects a context with KV-stream targets) and is a
defensive fail-closed guard; it adds no new user-facing restriction. The
phase switch refuses any unproven owner before reuse/reset, keeps the
committed physical split when only scheduler/graph reserve fails, and marks
the arena inconsistent if rollback itself fails.

Manifest evidence is the historical snapshot at `c75c6f8c1`, not the current
main tree: `common/arg.cpp`
`e89c65062fb9f16e8116abbcc922e5409328c69a0e3bdfb6d3aaa45bbe12b158`,
`src/llama-context.cpp`
`09798de83a217edd3fb233d2d5e8c8e62f3bc2422d4169711e9b21079d07a12c`,
`src/llama-context.h`
`90a79a6bd978e5e1ad8daf12c6971b617b381a9d3f573c15c5af5c488984f995`,
`src/llama-model.cpp`
`5a0baa4c949f2fae13012d1320b32477d5b87ae73e5e669d5e23873537ccb0d4` and
`src/llama-model.h`
`96ed16c351c50596a3e8bff522da82d7a19fb23f799939263dcea3e11d49a345`.
The manifest records `common/arg.cpp`
`d395a1d926266275ec5df0c6d7e9e72bed221a5e9e7c4b74d84475470c19377f`,
`src/llama-context.cpp`
`0d758756c2aef1947a8d92f372b13346350052f87d352ef48397876fd3ebfa84` and
`src/llama-context.h`
`50cf52d1b0bcc367f8626c66fd07e5a7de2d57c740cc1161469674d43fe0880b`; those
entries remain stale. The new phase header and MoE tests have no entry. No
digest was re-signed; later main-tree edits do not make this snapshot
current. `include/llama.h` matched before this round, but the ON comment
changes it; its manifest entry is now stale and is not re-signed here.

Verification boundary: CPU `test-moe-cache-session` (real scheduler/provider
session lifecycle), `test-moe-cache-context-policy` (real
`llama_init_from_model`, model-gated) and the existing staging scheduler test
are included but were not run here. The context-level ON provider-dormant
path and the CUDA phase-arena failure side effects remain unverified: the CPU
backend has no real MoE provider, and no device fault injection was run. No
GPU, full suite or guard rerun.

### E.9 MTP multi-row stop contract round 2026-11-09

Cherry-pick net effect `67c71b1da` -> `b3c6cf681`, reviewed as t27
(`agent://kernel6-review`). A verification stop includes its row in the
committed prefix and suppresses every later push. The target trim uses the
stopped request's first row position (`stopped_source_position`) and
`valid_rows`, not `batch.pos[0]`; the original replay mismatch gate is
retained, with the stop row required to match. `ids.size() == 1 && local == 0`
restates the existing T4 smoke option contract and adds no new restriction on
legal inputs. The draft stop cleanup synchronizes a pending draft decode and
trims from that request source; it is code-inspected, not exercised here.

DevOps ran `tq-tp5-commit-stop` on this tree (`addd2ca70`) once in the
exclusive directory `/tmp/tmp.7jHPfKmkip`: it built and installed the real
TQText source into `$D/prefix` (the existing install's `stream.h` differed
from the current source tree, which contains commit `6dee933f9`), configured
`fork/tp5-compose` with
`CMAKE_PREFIX_PATH="$D/prefix;<existing install>"`, built
`tq_tp5_commit_stop_tests`, and ran
`ctest -R '^tq-tp5-commit-stop$'`: 1/1 Passed, exit 0. Commands and output are
recorded at `agent://safe-devops` (t3).

This is helper/stream evidence only. It exercises the production
`commit_stop.h` helpers and the real `generation_stream`; it does not execute
`run()`, a model, checkpoint restore/replay, `llama_memory_seq_rm`, teacher
forcing or a draft context, and it does not prove runtime/draft/teacher or
GPU/end-to-end MTP behavior. No GPU, full suite or guard rerun. The five
changed paths have no manifest entry, so no digest was updated.

### E.10 Relay abort lock-free claim round 2026-11-09

Cherry-pick the final reviewed net effect `09044ca64` -> `cd319f4ad`
(the requested `9f310f9f` is the clean ancestor; t12 reviewed the later
`abort_requested` code at `762b83642`, and `cd319f4ad` changes only README
wording). The abort path is a lock-free claim: one 64-bit `abort_claim` word
holds the sticky abort bit and the host-success counter. `finish()`/`publish()`
claim a success by CAS; `abort()` stores the first pending error and sets the
sticky bit with one `fetch_or`; `snapshot()` uses a no-op CAS.
`prepare()`/`submit()`/`begin()` check the claim and pending error before any
plan-cache mutation or plan pin. First abort error is sticky, the provider is
fail-stop, and a new provider does not make old caller buffers reusable.

The public relay ABI 1 is unchanged: the header changes are comments only, no
struct layout or function signature changed. No shim was added. The existing
finish/sealed fix, graph-UID documentation, install-isolation annotations,
checkpoint/T3/ledger changes and other merged features are preserved.

Manifest: `ggml/include/ggml-relay.h`, `ggml/src/ggml-cuda/relay.hip`,
`ggml/src/ggml-relay-impl.h` and `patches/ws5/README.md` matched their
baseline digests and were reviewed; their hashes are updated. Other stale
entries keep their old digests.

DevOps ran one `hipcc -x hip -std=gnu++17 --offload-arch=gfx1030 -O1 -fPIC -c`
single-file compile of `ggml/src/ggml-cuda/relay.hip` at `430a4f02c` in the
exclusive directory `/tmp/tmp.IwJ4WT5WDi`: exit 0, 5 seconds. Commands and
flags are recorded at `agent://safe-devops` (t4). This proves source
compilation/codegen for gfx1030 only; it does not link, load a model, run
kernels, or test abort ordering/visibility/concurrency/performance.

The documented evidence also includes the static RMW-order, call and owner
audits. Device abort-word visibility and RMW concurrency were not run; no
runtime abort/finish behavior test exists. Old caller-owned buffers still wait
for real native completion or remain retained with the old owner; this is
retention, not a GPU completion claim. No GPU, full suite or guard rerun.

### E.11 Relay failed-drain and seal-history round 2026-11-09

Merge the final net `6596b6f29` -> `6035c35d` (`1af7d0907` -> `19419fda7`
-> `6035c35d`). `sealed` is the historical seal fact: `fail()` no longer
clears it, so `sealed && failed` is legal and means a sealed epoch later
failed; callers must check `failed`/`poll() == -1`, never `sealed` alone.
`finish()` clears `sealed` only when its own seal claim does not succeed, i.e.
when that finish did not seal the epoch.

On a failed or abort-claimed path, `snapshot()` calls `observe(p)` before
returning the sticky status, so native tails and `finished` continue to
advance without publishing success. During capture, `consume_abort()` runs
after the EndCapture loop, again before instantiate, and after the blocking
instantiate jobs; no new instantiate/launch starts after a visible abort. The
instantiate counter counts only jobs that actually succeeded. The runtime
drain stops waiting on a stopped/failed service and can exit once `finished`
advances; `llama_free()` retains only on a failed snapshot or an unfinished
epoch, not on sticky `failed` alone. The intermediate `19419` behavior of
permanent retain on failed is not present.

Manifest: `ggml/include/ggml-relay.h`, `ggml/src/ggml-cuda/relay.hip` and
`patches/ws5/README.md` matched their baseline digests and were reviewed;
their hashes are updated. `src/llama-context.cpp` remains a historical stale
entry and was not re-signed; `fork/tp5-compose/src/runtime.cpp` has no entry.

DevOps compiled `relay.hip` at `015dec2042` with
`hipcc -x hip -std=gnu++17 --offload-arch=gfx1030 -O1 -fPIC -c` in the
exclusive directory `/tmp/tmp.PhrAVeJBTH`: exit 0, 6 seconds, no warnings.
This was compile-only: no link, model load, kernel run, or GPU use. The full
command is recorded at `agent://safe-devops`. It does not prove device
execution, concurrency, memory ordering, snapshot/abort linearization or any
runtime/provider behavior.

Verification boundary: t4 (`agent://audit-integrated-relay`) is the static
audit. No GPU, full suite or guard rerun. Capture-phase abort, actual
instantiate counts and the wait/drain stop path were not executed; device
abort-word visibility and RMW concurrency remain unverified. t4 also records
the pre-existing staging `submission_failed` gap in `llama_free`, which this
net does not change.

### E.12 K-shift FAILED/NO_UPDATE round 2026-11-10

Integrate `3ac5c64d0` -> `b62af685b` -> `1b2139dad` -> `c22ca654e`. The
target tree did not contain the prior `2c98090a` prerequisite; the four-commit
`memory_update` changes were adapted to the target's pre-`2c98090a` body,
including the status/failure flags and the `has_unproven` pre/post guards
required by the four-commit diff. The separate `2c98090a` destructor
null-scheduler guard is not part of this integration.

`llama_memory_apply_status` distinguishes `NO_UPDATE`, `APPLIED` and `FAILED`.
`llama_kv_cache::update()` returns it; `apply()` keeps bool and exposes
`get_update_status()`. Setup/alloc failure restores the pre-shift cells; a
compute failure sets `shift_failed` until `clear(true)`. `state_write` and
`state_read_sinfo` guard `shift_failed` at their entry so an empty stream
cannot bypass the meta guard. Composite status combines FAILED > APPLIED >
NO_UPDATE.

A pure no-update decode returns at `init_update` with NO_UPDATE before
`apply`, so it does not reach the composite decision and does not add a
reserve. The recurrent default APPLIED mattered only when `init_update`
returned SUCCESS (a K-shift or stream copy) and `apply` was reached: in the
all-shared mirror case the KV apply can return NO_UPDATE, and the old
composite APPLIED forced a graph reserve. E.13 removes that unnecessary
reserve; a K-shift or stream copy remains APPLIED and still reserves.

The manual adaptation initially missed c22's post-apply main-scheduler
synchronize. It is now restored after apply/status and before
`graph_reserve`: a completion wait only, not proof of asynchronous success.
`has_unproven()` remains a staging/relay submission check, not a complete GPU
completion proof. A null `init_update` or catchable init exception maps to
-2; a catchable exception from `apply()` or `get_update_status()` maps to -3;
no exception path is treated as success.

The real CPU test `tests/test-kv-update-status.cpp` reports 20 checks passing
on the isolated branch; it was not rerun in this main tree. No GPU, full suite
or guard rerun.

Manifest: `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`, `src/llama-memory.h`
and `patches/ws5/README.md` matched their baseline digests and were reviewed;
their hashes are updated, and `tests/test-kv-update-status.cpp` is added as a
new entry. `src/llama-context.cpp`, `src/llama-context.h` and
`tests/CMakeLists.txt` were already stale and remain unsigned red items.

### E.13 Recurrent update NO_UPDATE and post-apply wait 2026-11-10

Cherry-pick `198668702` (based on `c22ca654e`) and restore the c22 post-apply
main-scheduler wait. Recurrent update contexts are always NO_UPDATE:
`init_update()` constructs `llama_memory_recurrent_context` with
`LLAMA_MEMORY_STATUS_NO_UPDATE` and no ubatches, and `apply()` is a documented
no-op. The override returns NO_UPDATE for that context and APPLIED for a batch
context, so hybrid/hybrid-iswa/hybrid-idx can skip `graph_reserve` when the
attention side's apply also has no work, specifically the all-shared mirror
case. DSV4 needs no extra override; its sub-contexts are KV contexts.

Scope: a pure no-update decode exits at `init_update` NO_UPDATE before
`apply`, so the recurrent default APPLIED never caused an extra reserve there.
The override matters only when `init_update` SUCCESS reaches `apply`; a K-shift
or stream copy remains APPLIED and still reserves. The local narrow check only
constructed the update context and exercised `get_update_status` and the
combine order; it did not run a hybrid end-to-end model, and no test was
committed for it.

Post-apply wait: `memory_update` synchronizes the main scheduler before
`graph_reserve`, so an asynchronous K-shift graph cannot be reset/reused while
in flight. The wait is completion-only, not an async-success claim;
`has_unproven()` is not a complete GPU completion proof. Null `init_update`
and catchable init exceptions map to -2; catchable `apply`/status exceptions
map to -3.

Manifest: `src/llama-memory-recurrent.h` and `patches/ws5/README.md` remain
stale/unsigned after this follow-up; `src/llama-context.cpp/.h` and
`tests/CMakeLists.txt` remain the existing red items. No digest was re-signed.
No GPU, full suite or guard rerun. More complex staging receipts remain a
separate line.

### E.14 Unified memory_update failure wait 2026-11-10

Implemented in an isolated worktree based on `69a2871c0`. Every `apply()`
exit — false, FAILED, std/non-std throw, post-apply unproven, or NO_UPDATE —
now goes through the main-scheduler wait before returning or entering
`graph_reserve`. Failure paths return -3 without a reserve and leave the
context/owner alive; a null `init_update`, `init_full` or `graph_reserve`
maps to -2 and returns false. Catch handlers cover both `std::exception` and
unknown exceptions.

The wait is a void completion barrier only; it is not an async-success
receipt. `has_unproven()` remains a staging/relay submission check, not a
complete GPU completion proof. Staging completion receipts remain a separate
contract and are not faked here.

Narrow verification (CPU, isolated worktree, not the main tree): built
`test-kv-update-status` and ran it once with
`/tmp/tmp.xRCpsZ1lzq/build/tests/test-models/llama-dense.gguf`:
`KV_UPDATE_STATUS_TEST PASS checks=20`, exit 0. The 20 checks are the
llama-dense CPU result; they are not a main-tree, GPU or end-to-end result.

Retained failed run (not rerun, not counted as pass): command
`/tmp/tmp.35y02xHpoy/build/bin/test-kv-update-status
/home/kunweiz/work/r28b-f3-draftonly-20261109/build-f3-fixtures/tiny-target.gguf`,
exit 1. Actual captured error/check lines:

```
state_seq_get_data: error saving state: unexpectedly reached end of buffer
state_seq_set_data: error loading state: unexpectedly reached end of buffer
graph_compute: ggml_backend_sched_graph_compute_async failed with error -1
process_ubatch: failed to compute graph, compute status: -1
decode: removing memory module entries for seq_id = 0, pos = [2, +inf)
llama_decode: failed to decode, ret = -3
FAIL .../tests/test-kv-update-status.cpp:255: baseline state get
FAIL .../tests/test-kv-update-status.cpp:269: cells restored after K-shift compute failure
FAIL .../tests/test-kv-update-status.cpp:281: empty-stream state write refused after failed update
FAIL .../tests/test-kv-update-status.cpp:283: empty-stream state read refused after failed update
KV_UPDATE_STATUS_TEST FAIL checks=20 failures=4
```

The complete terminal output was not saved to a file; the lines above are the
actual captured failure output. The specific root cause is unknown/not
proven: the failure first appears at baseline state get, but this record does
not assert fixture mismatch or a code regression. This failed run is not
evidence of success.

An intermediate corrected attempt (state buffers sized by
`llama_state_seq_get_size`, before the no-shift skip) ran qwen4exp and still
reported `KV_UPDATE_STATUS_TEST FAIL checks=22 failures=4`: the failed-ubatch
position check, nonempty state read, and both empty-stream state checks. That
run showed `seq_add` was ignored and the mock failure hit decode, not the
K-shift update. It is retained as diagnosis evidence, not a pass.

Follow-up correction (test-only): state buffers are now sized by
`llama_state_seq_get_size`. The test checks `llama_memory_can_shift()` before
any core path: an unsupported model prints `SKIP` and returns 77 without
running the core checks or printing PASS. The earlier qwen4exp `SKIP ... PASS
checks=13` output was before this capability/exit-77 change and is superseded;
it is not a core PASS. The qwen4exp exit-77 path was not rerun after this
change.

For a shift-capable fixture, the test now requires `seq_add` to actually move
the sequence position before running the K-shift FAILED checks; a no-op is a
failure, not a skip. Dense llama run after the change:
`KV_UPDATE_STATUS_TEST PASS checks=21`, exit 0, with the movement assertion and
the K-shift/state-guard checks executed. The earlier 20-check dense result is
historical; the current test has 21 checks. Root cause of the original
qwen4exp empty-stream failures is that no K-shift update occurred and the
state guards were not reached; this does not prove a state-guard bug.

Manifest: no digest was re-signed. `src/llama-context.cpp/.h`,
`src/llama-memory-recurrent.h` and `patches/ws5/README.md` remain red/unsigned.

### E.15 KV prepare journal and ownership round 2026-11-10

Integrated the final net `99f1d48f9` -> `758b81a98` (journal branch plus the
unbound-destination ownership fix). This is a net application, not a replay
of the used-node pool/revert history pair.

Prepare now activates a per-stream cell journal. The journal records the full
write domain, including SWA `seq_rm` purge cells outside `sinfo.idxs`, and
replays in reverse on a throw; heads and the stream `has_shift` state are
restored. The normal restore uses the per-trial `v_cells` snapshot when the
journal entry predates the trial, otherwise the journal entry. K-shift
`shift_failed`, update status, shift backup and the state guards remain the
explicit async-failure path; the journal covers prepare exceptions, not
native async success.

`llama_kv_cells` now requires an unbound destination for both assignment
operators. Copy construction/assignment never inherit the source journal;
move construction/assignment transfer and clear the source journal; a bound
destination asserts. Production mutates live `v_cells` in place and uses
unbound prepare snapshots; vector growth/snapshot reuse only assign unbound
elements. This is the ownership fix, not a change to the journal data model.

Scope: `test-kv-cells-journal-ownership` is header-only synthetic ownership/
OOM coverage; it was run once and passed, including the 262145-cell boundary.
`test-tp5-ubatch-pool` uses a synthetic
real `llama_kv_cache` with OOM sweeps (SWA/non-SWA, persistent, begin_trial,
same-cell) and capacity/microbench reporting; it is not a full model run.
`test-kv-update-status` uses a real CPU dense Llama context. SWA purge tests
use synthetic cache construction; `n_swa=0` covers the journal replay branch,
not an SWA purge. No full runtime model, GPU, or staging receipt was run.

Narrow verification in this isolated worktree, once each:
`test-tp5-ubatch-pool` reported `OK failures=0 allocs=24313 frees=23163`,
exit 0 (synthetic real-cache OOM/SWA/non-SWA sweep, including the host
breakdown and cells-address checks; not a full model);
`test-kv-update-status` with `tests/test-models/llama-dense.gguf` reported
`KV_UPDATE_STATUS_TEST PASS checks=21`, exit 0 (real CPU dense context).
`test-kv-cells-journal-ownership` passed once with the 262145-cell boundary
(the two child GGML_ASSERT messages are the expected bound-destination death
tests). No GPU/full suite was run.

The K-shift shift backup is now strong on allocation failure:
`shift_backup_save()` copies `v_cells` and `v_heads` into locals and swaps only
after both copies succeed, so a throw leaves the committed backup empty and a
retry starts from a complete snapshot. `shift_backup_restore()` swaps instead
of assigning, so it cannot fail halfway. The new
`test_shift_backup_strong_oom` cases check the failed save leaves no partial
backup and that a retry restores the complete cells/heads state. Journal
replay failure still aborts via `GGML_ASSERT`; it is never reported as
success. `test-tp5-ubatch-pool` now reports
`OK failures=0 allocs=24313 frees=23163`, exit 0, including those cases. The
K-shift status test was not rerun after this backup-only hardening.

The cell-journal entry index is no longer an 18-bit field. `seen_` is one
64-bit word per cell: 32-bit generation in the high half and a 32-bit entry
index in the low half; `trial_seen_` remains a 4 B generation word. The
generation wrap clears at 2^32; the entry count is bounded by the actual
entries vector and the 32-bit index, not an artificial 262144 limit. The hot
path is one 64-bit load; the public `bytes()` budget counts 8 B for `seen_`
and 4 B for `trial_seen_` (12 B/cell, 3 MiB/stream at 262144 cells). The old
2.2 MiB/1,105,920-byte capacity figures predate the widening and were not
rerun.

`prepare_journal_begin()` now asserts that `v_cells[s]` is not already
journal-bound, so a reentrant prepare fails loudly instead of silently
resetting another writer's trial. The journal replay still aborts via
`GGML_ASSERT` on a failed restore; it never reports success.

`test-kv-cells-journal-ownership` includes a 262145-distinct-cell boundary
case (one past the old limit) and passed once; its bound-destination death
tests intentionally print child abort messages. The 12 B/cell budget is the
current public number; the old capacity/microbench observations are historical.

Host accounting is exposed separately as
`llama_get_host_memory_breakdown()` in `llama-ext.h`; it does not change
`llama_get_memory_breakdown()`, whose map remains device/buffer-type memory
only. The host struct reports `journal`, `trial_index`, `snapshots` and
`scratch` payload capacity: `cell_journal::bytes()` is now called, the
snapshot pools/fallback slots and shift backup are counted once, the flat
trial index is counted from the scratch blocks, and scratch control vectors
are separate. `slot_info` now reports its nested stream and per-stream index
heap, so snapshot slots and decode scratches no longer lose those payloads.
`cell_journal::bytes()` also counts the set/map nodes the journal extracted
from live cells, matching the payload that `llama_kv_cells::host_bytes()`
stops counting after the extract. `heads_before` belongs to the prepare
snapshot and is counted in `snapshots`, next to the per-slot `v_heads_old`
arrays. Main resident `v_cells` metadata is not included in this
journal/prepare account. These are payload capacities, not allocator
bookkeeping. hybrid/hybrid-iswa/hybrid-idx aggregate their KV sub-caches;
dsa/msa/dsv4 aggregate their KV members; recurrent contributes zero.

This is a C++ extension, not a new field in the device-buffer map. Call it
only while the memory is quiescent, and serialize it with every memory
mutation: decode/encode, set_input, `llama_memory_clear`, the `seq_*` family,
state_read/state_write, update/shift, prepare/replay, and pool
acquire/release. It reads the live host containers without locks.
`llama_synchronize()` alone does not make the query safe. A const context
pointer is not a read-only guarantee: a concurrent mutation can reallocate
the containers while the query walks them. A batch context from init_batch
owns the split ubatches and slot infos: the pooled path moves them out of the
cache scratch, while the non-pooled or pool-exhausted path allocates them
locally. While any such context is alive, the cache account cannot see that
payload and `scratch` is a lower bound. An init_update or init_full context
does not take the decode scratch payload. Do not query while any memory
context is in use, and serialize the query with its creation and destruction.
The account also excludes resident live `v_cells`/`v_heads`, recurrent
state, compressor state, PLE history, rotation matrices, and every host
container that no journal/prepare owner reports.

`shift_backup_restore()` now swaps cell payloads in place instead of swapping
the `v_cells` vector, so `get_cells()` references and the cell objects keep
their addresses. `get_cells()` has no production caller that holds a
reference across restore; the test access path checks the address and the
restored state explicitly.

Public-consumer verification: the qwen4exp tiny hybrid context reports an
empty journal at construction, then a nonzero journal after one decode; this
exercises `llama_get_host_memory_breakdown()` and the hybrid aggregation. The
test still exits 77/SKIP because that cache cannot shift, so it does not print
the core K-shift PASS. The dense host field check is in the same test but was
not rerun after this public-check addition.

The host-account regression lives in `tests/test-tp5-ubatch-pool.cpp` and
runs without a model file: it checks reserved-capacity slot-info payload,
non-empty snapshot and fallback slots, the journal node payload for entries,
trials and discards, a real prepare, and the `heads_before` classification.
The model-backed public-API state check is in `tests/test-kv-update-status.cpp`:
an empty journal at construction and a nonzero journal after one decode, when
`LLAMA_NTOK_TEST_MODEL` is set.

CPU microbenchmark (agent://safe-devops t8; raw values in
`/tmp/tmp.7Kd6QbMNvC/results.csv`): it compares the 18-bit index direct
parent `fa400274f` against the 32-bit `03331f2d6`, not a completely
unoptimized original. It is a synthetic, predeclared 3-group ABBA run (12
independent processes, same `g++ -std=c++17 -O2 -DNDEBUG` flags; only the
header differs). Medians, old/new: 8192-cell small case 50.080/50.904 ms;
262145-cell boundary record 46.427/46.300 ms; 16385 `begin()` calls
144.586/101.744 us. The small and record deltas are within run-to-run noise.
The `begin()` difference is consistent but absolute cost is only about
0.10-0.17 ms per 16385 begins, and it is the old 16384th-begin full `seen_`
fill path, not a general throughput result. The old arm cannot record the
262145th entry, so that record boundary is not a paired same-workload
comparison. No GPU, model tok/s, or significant-gain claim.

Combined main-tree CPU check: a fresh CPU-only build in
`/tmp/tmp.sguI26rjTw` built once; `test-kv-cells-pos-lookup` and
`test-tp5-ubatch-pool` each ran once and passed. No real model or GPU was
used. The four uncommitted staging files were not exercised; their runtime
behavior remains unverified.

Manifest: `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`,
`src/llama-kv-cells.h` and `tests/test-tp5-ubatch-pool.cpp` were updated to
their actual post-integration hashes. `tests/test-kv-cells-journal-ownership.cpp`
has no manifest entry and was not added. Existing K-shift entries and stale
red items were not re-signed.

### E.16 ON_DEVICE checkpoint contract repair round 2026-11-09

Fixes candidate `507dded7c` and the first review round. The host sequence blob
keeps its existing 8 B magic/seq_id header, so existing host states remain
readable and writable and host restore remains legal across contexts. The
device blob has a private 28 B versioned header (device magic, seq_id, version,
context identity, generation); it is context-bound. A device blob passed with
host or PARTIAL_ONLY flags is rejected before any memory module sees it, and
vice versa. The sequence file format (v3) and the whole-context state format
are unchanged. The device blob is a private context-local handle: a
prior-version device blob cannot reference a live `mem_storage` snapshot in a
new context, so the version bump returns 0 instead of silently using a
different snapshot. Host blobs and file v3 remain compatible.

Boundaries and transactions:

- get_size/get_data/set_data and sequence save/load file allow seq_id == -1
  (all sequences, matching the internal sequence state paths) and reject
  seq_id < -1 or >= n_seq_max with 0 instead of reaching a memory-module
  assert.
- The recurrent single-sequence restore checks `cell_count <= size` before
  clearing the destination or reserving a ubatch.
- A single-sequence KV restore clears the destination seq first, so an empty or
  shorter source cannot leave stale attention.
- Device state copies try the destination buffer `cpy_tensor` (bool) and fall
  back to `ggml_backend_tensor_copy`. A `cpy_tensor` true only means the
  backend accepted or enqueued the copy; it is not a device-completion
  receipt. The capture and restore `finalize` paths synchronize the scheduler
  backends before the new snapshot is published or the restore returns, so the
  previous snapshot is not released while a copy can still write it. A Meta
  destination is split into its real child buffers and each child is copied
  with the child buffer's `cpy_tensor` (blocking fallback); the staging owner
  is not used. The Meta child copy preflights every child (exact split state,
  simple view, buffer/data, type/shape/nbytes) before any side effect: false
  means preflight/unsupported with the destination untouched. A false
  preflight is no-side-effect unsupported: the ON_DEVICE call returns 0 and
  does not mark unproven; the caller must explicitly request the host format
  (host flags) for a host blob. The ON_DEVICE blob is a device metadata handle,
  not the full host state; the host blob is a separate format whose size
  includes the tensor bytes. A transposed v/s state writes a row-interior
  range; the range is copied through a host buffer instead of a Meta view,
  because `init_tensor_impl` scales a view offset by the child split and would
  address the wrong element. Meta get/set also gained an arbitrary axis-0/1
  byte-range path that walks the segment/child blocks and copies only the
  requested overlap. A copy that throws after it started is partial/unknown:
  the ON_DEVICE restore returns 0 and retains the snapshot, its view contexts
  and the destination cells; the capture returns 0 with the old snapshot and
  retains the new staging owner in `mem_storage_retired`. Both are fail-stop;
  no fallback runs and unproven is not cleared. Non-Meta device errors keep
  clear-on-failure; a cross-context device blob is rejected by its context id.
  A CUDA_CHECK abort is not a returned status. The generic
  `ggml_backend_tensor_copy` fallback checks its malloc. A partial restore sets
  `state_copy_unproven` and marks the memory unproven; the outer catch keys on
  the reader's unproven state, including the non-row-aligned host path.
  `has_unproven()` makes `llama_free` retain the whole context (source,
  destination, backends, model); the state API returns 0. The memory modules
  and the outer state catch check `llama_state_copy_unproven()` and skip
  `seq_rm`/`clear`, so the destination cells remain owned by the retained
  context. Public memory mutators refuse while the memory is unproven.
- A device capture builds a fresh buffer map and swaps it in after the
  CUDA/CPU/Meta child copy and the completion synchronization return. A
  failed preflight leaves the previous snapshot and generation intact and
  discards the new staging with no side effect. A capture that throws after a
  copy started leaves the previous snapshot intact, retains the new staging
  and marks the context fail-stop. The old and new buffers coexist until the
  swap, so peak device memory is about twice the snapshot size; the 512 MiB
  cap is gone and no fixed cap replaces it.
- `llama_tp5_context_staging_drain` from `/home/kunweiz/work/r28b-pillar5-sync-owner-v4-e123`
  @7fac3140f is not used. Static comparison: with no relay backend it returns
  DRAINED without querying the memory backend, while an ordinary checkpoint
  D2D copy can still be async there; with a relay backend it drains the
  staging provider/owner, not the CUDA buffer `cpy_tensor` blocking copy. The
  CUDA copy is not owned by that staging owner. The checkpoint does not
  integrate it. The Meta destination is split into real child buffers and each
  child is copied with the child buffer's `cpy_tensor`; the staging owner is
  not involved in those child copies. The staging candidate is pending review
  and is not a dependency. The staging owner confirms that 7fac3140f is
  superseded by an uncommitted rework, that staging DRAINED must not be used
  for an ordinary D2D checkpoint copy, and that an explicit memory-backend
  producer registration or UNKNOWN may be added later; the checkpoint does not
  need it for the child split.
- t9 M1 (a completed Meta child copy marked unproven by
  `state_copy_events_wait`) is stale for this worktree: the current
  `src/llama-context.cpp` has no `state_copy_events_wait` or
  `ggml_backend_event`; the Meta path uses `ggml_backend_buffer_copy_tensor`.
  The current Meta CPU test exercises that path. t9 M2/M3/M5 (view/split
  preflight and partial-child fallback) are fixed by the full preflight, the
  no-false-after-side-effect copy loop, and the explicit unsupported contract
  (ON_DEVICE returns 0; the caller requests the host format explicitly); the
  actual state API Meta test covers a bad-split context, cross-context device
  rejection and host restore. M4 (CUDA_CHECK abort) is a backend fatal, not a
  returned status; M6 GPU/HIP remains uncovered.
- Host-flag and file restores have no rollback and allocate no full host
  backup. A failed host/file restore returns 0/false and the destination
  follows the memory clear-on-failure policy. Device unproven copies use the
  fail-stop retention above.
- The 512 MiB host-size proxy is gone; the runtime requires a positive host
  state size and reports host and metadata bytes separately. Allocation
  failure is a failed capture and preserves the old snapshot.
- `common_prompt_checkpoint`'s four update/load methods and their bool callers
  are owned by the context-lifecycle branch; this branch reverted its 0-size
  edit and does not change them. The `empty()`/`data_spec` semantics remain
  with that branch. The save/load test rejects an empty device state.
- The unreachable live-checkpoint guard was removed.
- Device views convert byte counts through the type block size (quantized
  tensors), and the recurrent multi-range device guard now checks the tensor
  range vector, not the metadata range vector.

Verification: CPU-only `llama` build in an exclusive mktemp directory (r40).
The narrow state test binary was run once with the tiny Mamba fixture: 83
PASS, 0 FAIL. It includes the original 14 Meta buffer cases, the existing CPU
state tests, and the actual state API Meta cases: normal Meta memory
capture/restore; a second bad-split context device capture/restore;
cross-context device blob rejected; host-format restore across contexts; the
normal state unchanged and the context usable. The ON_DEVICE blob remains the
device metadata handle; the host blob is a separate format. The test binary
compiles; the runtime, `common.cpp` and `test-save-load-state.cpp` passed
`-fsyntax-only` checks. The earlier host/-1 and full 69/64 runs predate this
change and are not rerun. GPU/HIP Meta child copies and the CUDA_CHECK fatal
path are not run.

Not covered: GPU/HIP Meta child copy completion and the CUDA_CHECK fatal abort
(a backend fatal, not a returned status), the Meta partial-copy mark/retention
(the child throw is not triggered), different buft/quantized child layouts,
the staging owner's receipt (not used; staging candidate pending), allocation,
a real TP5 MTP run, strided non-contiguous source views, the recurrent
multi-range guard (code-reviewed only), host/file target preservation
(clear-on-failure, no rollback), the staging owner's ordinary-copy receipt (the
staging candidate is pending review and its drain is not used), old device
blobs from prior library versions (the private device handle is versioned and
rejected), `common_prompt_checkpoint`'s bool API/`empty()`/`data_spec` and its
callers (owned by the context-lifecycle branch; integration must use that bool
API), ASan/LSan, the full suite and the manifest guard (not rerun), and the
existing model file/generation tests (not rerun).

Follow-up repair (static only): the three review gaps are addressed. A
row-interior `v_trans`/`s_trans` range no longer builds a Meta view; Meta
get/set gained an arbitrary axis-0/1 byte-range path and the state copy moves
that range through a host buffer. A `cpy_tensor` true is treated as an accepted
submission, and both finalize paths synchronize the scheduler backends before
the snapshot is published or the restore returns. A capture that throws after
a copy started retains its staging owner and marks the context fail-stop; a
restore that throws after a copy started retains the snapshot, view contexts
and destination cells, with the outer catch keyed on the reader's unproven
state. The new Meta range test covers a 20-element window starting inside the
first child and spanning into the second. None of this follow-up was compiled
or run here: the CPU/Meta test, the state test and the real TP5 MTP path remain
for safe-devops. The manifest digests are unchanged.

Manifest: `src/llama-io.h`, `src/llama-impl.h`, `src/llama-memory-hybrid.cpp`,
`src/llama-memory-hybrid-idx.cpp`, `tests/test-save-load-state.cpp` and
`fork/tp5-compose/tests/test-mtp-state-real.cpp` have no entries;
`include/llama.h`, `src/llama-context.cpp`, `src/llama-context.h`,
`src/llama-kv-cache.cpp`, `src/llama-memory-recurrent.cpp`,
`src/llama-memory.h`, `ggml/src/ggml-backend-meta.cpp`,
`ggml/src/ggml-backend.cpp` and `ggml/src/ggml-backend-impl.h` have existing
entries whose digests were not updated. The B5-B7 red items remain.

## F. Why this does not block the integration

- The guard is a manifest boundary check, not a compiler or runtime test. Its failure
  names pre-existing drift; the integration does not add new unreviewed A files beyond
  the registered WS paths.
- No unrelated A file was edited to repair a digest. No pre-existing hash was silently
  overwritten; the B2 table records every re-sign and its prior drift.
- Build and CPU test receipts are in `/home/kunweiz/work/r28b-integ-build`.

Resolution: run a dedicated manifest audit, decide each pre-existing path, then delete
this waiver. Until then, treat the guard failure as known debt.
