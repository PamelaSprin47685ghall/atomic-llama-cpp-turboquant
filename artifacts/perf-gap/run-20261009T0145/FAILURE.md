# Baseline screen failure

The first registered screen did not reach a measured result. Warm requests returned
182 tokens / 62.5499 tps and 176 tokens / 64.9544 tps. The third request failed at
position 159, stage 1, epoch 47423, bank 0: `STAR payload timeout on rank 0`.
The process then reported ErrorDeviceLost and exited -6. It was not force-killed.
Kernel logs show four Navi21 rings reset automatically at 01:47:43 UTC; no manual
reset, clock, driver or watchdog changes were made. The entire sweep stopped.

These two warm results are not performance qualification. Earlier eight-process
qualification remains historical evidence, not proof that this failure cannot occur.
Position/shape change is a diagnostic clue, not an established cause.

Next step is an error-only two-bank state snapshot plus gated host phase timers.
The failed screen is preserved and will not be replaced or pooled with later tests.
Any repaired-source performance needs fresh process pairs and output review.
