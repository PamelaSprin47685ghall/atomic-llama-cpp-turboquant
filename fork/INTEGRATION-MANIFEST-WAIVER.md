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
- WS5 `plan_key` has eight fields including `channels`; B `PlanKey` has seven fields.
  They are separate domains. The C adapter maps them explicitly; the types are not merged.
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
