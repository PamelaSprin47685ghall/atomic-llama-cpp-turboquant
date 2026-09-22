# 5GPU blocked scope (dev host)

Host: 1x RX 6800 only. Cannot run 5-card TP5 / multi-rank harnesses.

## Still deliverable offline on this host
- C11 definition-time transport enum / fixture packs / mock publish-order cases — DELIVERED
  (`PLAN/fixtures/t1c-transport-enum.json`, `PLAN/fixtures/r07-small-batch-shapes.json`,
  `scripts/test-t1c-publish-order-mock.py` PASS)
- C12 pending experiment packs + L1/L2 candidate freeze without fabricating winners — DELIVERED
  (`PLAN/experiment-queue/pending-packs/r14-tm5-*-abba.json`,
  `scripts/test-pending-abba-packs-mock.py` PASS; verdict PENDING_5GPU / NO_WINNER_CLAIMED)
- All C01-C10 offline ceilings not requiring 5 devices

## Explicitly blocked until 5GPU window
- T1C three-transport same-arithmetic multi-rank route evidence on real devices
- R14/TM5 paired ABBA hardware verdicts that need multi-GPU timing
- test-vulkan-tp5-relay multi-rank partial-submit beyond single-device mocks

## Rule
Do not mark C11/C12 hardware evidence complete on this machine. Publish pending packs only.
