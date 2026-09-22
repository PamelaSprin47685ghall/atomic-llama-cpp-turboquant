# 1GPU session evidence summary

Date: 2026-09-22T16:24:45
Host: JYSZ (1x AMD Radeon RX 6800)
Build: build-vulkan-localhost

## Scope
- Offline + 1GPU unit/prototype validation only.
- **No 5GPU claims.** C11/C12 remain blocked on multi-GPU hardware.

## Results (revalidation log)
Source: `PLAN/experiment-queue/evidence-dry/1gpu-c06-c08-c10-revalidation.log`

| Target | Result | Notes |
|---|---|---|
| test-rerot-math | PASS | 0 failures |
| test-rerot-span-expand | PASS | CPU + Vulkan bit-identical |
| test-rerot-span-expand + LLAMA_REROT_GPU_SPAN_EXPAND=1 | PASS | GPU path exercised |
| test-rerot-q-prep | PASS | CPU vs reference; GPU skipped (default off) |
| test-rerot-q3-shared-kv-2reader | PASS | CPU golden + glslc; GPU dispatch deferred |
| C03 teardown mocks | PASS | scripts/test-safe-teardown-mock.py + test-experiment-queue-safe-teardown-mock.py |

## Delivered offline ceilings
- **C06**: graph wire in `src/llama-graph.cpp` / `.h`; env `LLAMA_REROT_GPU_SPAN_EXPAND=1`; default dense CPU path unchanged; default enable NOT authorized.
- **C08**: shader/route/CPU forward landed; env `LLAMA_REROT_GPU_Q_PREP=1`; NORMAL rotate reference fixed; real GPU dispatch opt-in/queued.
- **C10**: shader stub + CPU golden harness; pipeline registered; no production GGML_OP dispatch.
- **C03**: experiment-queue + safe-teardown mocks.

## Explicitly untested / blocked
- 5GPU T1C / R14-TM5 paired adjudication (C11/C12)
- C08/C10 production default-enable
- End-to-end model throughput on multi-GPU

## C11/C12 offline (no hardware verdict)
- T1C transport enum + publish-order mock: PASS (`scripts/test-t1c-publish-order-mock.py`)
- R07 small-batch shape fixtures: `PLAN/fixtures/r07-small-batch-shapes.json`
- Pending ABBA packs (NOT_APPROVED, NO_WINNER_CLAIMED):
  `PLAN/experiment-queue/pending-packs/r14-tm5-c06-span-expand-abba.json`,
  `PLAN/experiment-queue/pending-packs/r14-tm5-c08-q-prep-abba.json`
