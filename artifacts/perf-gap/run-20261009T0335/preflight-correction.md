# Synthetic-service preflight correction

Two attempts were refused before GPU initialization by the existing relay test's
idle gate (`relay-micro-1.log`, `relay-aligned-2560.log`). No payload was submitted,
no protocol timeout and no reset occurred. The 256-element standalone probe did
run: 96 stages x16 replays, all 1,986,560 element checks passed; wall 4403.457 us per
chain includes numerical-reference checks, so is NOT a hardware latency floor.

The wrapper wrongly assumed `gpu_state()` enforces idleness; it only returns a
snapshot. The wrapper now enforces exactly five cards, busy=0 and VRAM<128MiB,
with two observations before proceeding. No gate/tolerance was relaxed. Further
service results go to `service-v2/`; refused files remain unchanged and are never
pooled as timing points. The successful 256-element result is not replaced.
