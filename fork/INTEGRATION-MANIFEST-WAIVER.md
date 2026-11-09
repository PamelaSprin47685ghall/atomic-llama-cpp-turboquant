# Integration Manifest Waiver — `integ/tp5-mtp-r28b`

Status: the upstream guard still fails on pre-existing drift. This waiver records every
pre-existing mismatch and undeclared A path, the WS-owned registrations, and why the
guard failure does not block this integration's build or CPU tests.

- Base: `71544695b`
- Merged: WS2 `ae9c13065`, WS3 `6d0614915`, WS4 `15fd113e2`, WS5 `9cd97421e` (parent `4648777f7`)
- Applied: `patches/ws5/{ggml-backend-meta.cpp,runtime.cpp,ggml-backend.cpp}.patch`
- Guard: `python3 fork/check-upstream.py` rc=1
- Guard stderr: `boundary check failed: reviewed content or mode differs: ggml/src/ggml-cuda/mmvf.cu`

## A. Pre-existing mismatches not re-signed

These files were already stale at the integration base and are not touched by WS2/WS3/WS4/WS5.
Their digests are left exactly as recorded; this integration does not bless them.

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

- `ggml_relay_plan::config::graph_uid` defaults to `false`. The OFF leg therefore returns
  to uid 0, the full signature, 2 frames, the 384 MiB pool, no plan cache and map
  channels. The ON leg must set `GGML_TP5_GRAPH_UID=1` explicitly.
- `GGML_TP5_RELAY_FRAMES` defaults to `2`; the ON leg must set `8` explicitly.
- `GGML_TP5_POOL_EXACT` defaults to `0`; the ON leg must set `1` explicitly.
- `GGML_TP5_PLAN_UNION` defaults to `0`; the ON leg must set `1` explicitly.
- WS5 `plan_key` has nine fields: the seven B fields, `channels`, and an internal
  terminal-binding digest; B `PlanKey` has seven fields, and the C boundary carries the
  eight caller-visible fields. They are separate domains. The C adapter maps them
  explicitly; the types are not merged.
  `patches/ws5/README.md` records OFF≡stock and the two-domain mapping.

## F. Why this does not block the integration

- The guard is a manifest boundary check, not a compiler or runtime test. Its failure
  names pre-existing drift; the integration does not add new unreviewed A files beyond
  the registered WS paths.
- No unrelated A file was edited to repair a digest. No pre-existing hash was silently
  overwritten; the B2 table records every re-sign and its prior drift.
- Build and CPU test receipts are in `/home/kunweiz/work/r28b-integ-build`.

Resolution: run a dedicated manifest audit, decide each pre-existing path, then delete
this waiver. Until then, treat the guard failure as known debt.
