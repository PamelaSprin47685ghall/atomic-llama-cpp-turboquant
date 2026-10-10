# LateBind squeeze — round 2: corrected verdict and methodology repair

Date: 2026-09-28. Base: `/var/tmp/tp5-candidate-k1a` (dirty, uncommitted) → build `/var/tmp/tp5-final-build`.
Workload: natural text, `请用八句话简要介绍健康减重原则、每周运动安排、饮食建议和睡眠误区。`,
prompt_n=31, temp=0, seed=42, stream=false. No counting-task substitution.

## 0. Why this round exists

Round 1 (13 paired arms, `run-20260928T0130/paired-arms/`) concluded "LateBind is negative".
The user rejected the methodology: comparisons were not tip-to-tip ("针尖对麦芒"), and a negative
verdict about a *mechanism* may only be a negative verdict about one *implementation*.
Re-audit found four concrete defects in that round. All four are fixed here and the mechanism is
re-opened.

## 1. Methodology defects found in round 1 (all confirmed from the archived logs)

| # | Defect | Evidence |
|---|---|---|
| D1 | LateBind arms were measured **without the GPU-side ledger** (`GGML_TP5_GPU_TIMING` unset), so `late_pre`/`p2` overlap was never observed — only inferred from CPU counters. | `grep GGML_TP5_GPU_TIMING` absent from `tp5-paired-arms.sh`; no `[tp5-relay-gpu]` lines in `B_exact.log`. |
| D2 | Arm-internal spread was never reported next to the effect. A_reference handoff was 25.7 ms vs 30.5 ms (19 % spread) while effects of 6.8 ms were claimed. | `A_reference.log` relay-profile lines. |
| D3 | `exact` produced **different text** (279 B vs 308 B, different think-block) → not the same arithmetic, so "+49 % cost" was a different-work comparison, not the same-work one the verdict needed. | `B_exact_r3.json` vs `A_reference_r3.json`. |
| D4 | The **fast** LateBind modes were never runnable: both `aggressive-q8` and `p1a-nosidecar-q8` SIGSEGV during warmup. The verdict was therefore taken on the slow fallback only. | `D2_aggr_noprofile` / `E_p1a`: `exit_status=139` after `[tp5-latebind] ... mode=aggressive-q8`. |
| D5 | The E_p1a arm's binary **predated the P1A `late_count=0` fix** (binary 01:13:48 < source 01:14:26) — the run was invalid even as a crash test. | file mtimes. |
| D6 | Chain captures used a fixed index (`GGML_TP5_GPU_TIMING=6`) across arms that were in **different phases**, so the "1 row 95 µs/stage → 3 rows 130 µs/stage" inference compared unlike chains. | `M1_target_only.log` / `M3_h2.log` both captured chain #6. |
| D7 | A "systematic per-rank imbalance (rank4 = 1.7×)" was concluded from a **cold head**. Splitting the same capture by head/mid/tail shows the imbalance is confined to stages 0..31; stages 32..95 are balanced to 2 %. | `tp5-stage-cranks.py M3_h2.log`: head rank0 5122 / rank2 12237 / rank4 12448; mid 4865..4981; tail 4889..5000; median max/mean per-stage ratio **1.021**. |
| D8 | Same-work figures were quoted as **means** of `target_us` while the distribution carries rare ~460–480 ms outliers (the chain re-recorded for the timing capture). Same h=1 config, two scripts: mean 28971 vs 39615 (+37 %) but **median 25247 vs 23864 (−5 %)**. Round 1's "exact = +49 %" was such a mean-based, cross-config number; the outlier-free, same-rows figure is **+19.3 %** (28457 vs 23864 median). | `[tp5-mtp-cycle]` distributions in `M2_h1.log`, `R_h1_ref.log`, `R_h1_exact.log`. |

Repairs applied:

- Captured chains now **self-label**: `[tp5-relay-gpu] ... rows= seqs= phase=` (`ggml-vulkan-collective.cpp`).
- Opt-in `GGML_TP5_CHAIN_TRACE=1` prints one line per chain (`[tp5-chain]`) so a capture index can be
  aimed at a chain of known phase instead of guessed.
- The reference arm is the **best measured horizon** (h=1), not the inherited default (h=2).
- Every candidate arm is now paired with the GPU ledger and a drift control on the same binary.
- Same-work numbers are reported as **median + p10/p90 over `[tp5-mtp-cycle]`**, never means, and only
  between arms with the same row count.

### 2b. Corrected same-work table (medians, same row counts only)

| arm | rows/cycle | target_us median | p10–p90 | draft median | tokens/cycle | cycle | tok/s |
|---|---|---|---|---|---|---|---|
| M1 `--spec-type none` | 1 | (no ledger) | — | — | 1.0 | 18.38 ms | 54.4 |
| M2 h=1 | 2 | **25247** | 23785–27169 | 2439 | 1.753 | 28.1 ms | **62.4** |
| M3 h=2 | 3 | 33825 | 30013–35404 | 4336 | 2.152 | 38.6 ms | 55.7 |
| M4 h=4 | 5 | 45651 | 44596–47307 | 8106 | — | 54.2 ms | — |
| R_h1_ref (sharp-1) | 2 | 23864 | 23206–26123 | 2444 | 1.753 | 26.7 ms | 65.6 |
| R_h1_exact (sharp-1) | 2 | 28457 | 26489–33735 | 2370 | — | 30.8 ms | — |



## 2. Row-scaling on natural text (same binary, same env, one factor per arm)

| arm | rows/cycle | tok/s r1 | tok/s r2 | target_us/cycle | draft_us/cycle | acc/cycle |
|---|---|---|---|---|---|---|
| M1 `--spec-type none` | 1 | 54.39 | 54.31 | — | — | — |
| **M2 `-n-max 1` (h=1)** | 2 | **58.39** | **58.39** | 28971 | 2431 | 0.753 |
| M3 `-n-max 2` (h=2) | 3 | 51.76 | 54.60 | 37646 | 4343 | 1.152 |
| M4 `-n-max 4` (h=4) | 5 | 43.97 | 44.10 | 51187 | 8144 | — |

- h=1 is the optimum: **+7.4 % over no-MTP**; h=2 (the inherited default) is **−2 %**; h=4 is −19 %.
- Marginal row cost: 1→2 rows **+10.6 ms**, 2→3 rows **+8.7 ms**, 3→5 rows **+6.8 ms/row** against a
  full single-row decode of **18.4 ms** — i.e. an extra verified row costs ~half a full row.
- Consequence: MTP on natural text is economical only while `tokens/cycle` grows faster than the row cost.
  At p≈0.75 the ceiling is h=1. The previous default (h=2) was tuned on the counting workload.

## 3. LateBind internals (source-verified this round)

- `mode=exact-f32` = the **f32 fallback**: its per-site publication keeps a
  `COMPUTE→HOST` domain barrier ("Exact fallback publishes from imported system RAM and therefore
  keeps the established Vulkan HOST-domain barrier") at every late site.
- `late_q8_fast` (aggressive / P1A) exists precisely to remove that barrier: "Aggressive Q8 writes
  cached device-coherent bcast VRAM instead; its payload/ready protocol is shader-ordered and
  CPU-visible **without** a queue-wide COMPUTE→HOST execution dependency."
- Per-chain counts confirm the dispatch-side win of the late path, and confirm the cost is the protocol:

| per chain | A_reference | B_exact |
|---|---|---|
| dispatches | 4603 | **2926 (−36 %)** |
| barriers | 3996 | **2506 (−37 %)** |
| late sites | 0 | 94 |
| CPU `sidecar_wait_us` | 0 | **6830** |
| CPU `handoff_us` | 25.7–30.5 ms | 36.9–38.6 ms |

  `ready_us + sidecar_wait_us ≈ handoff_us` → the Q wait is **fully serialized** into the per-stage CPU
  handoff loop, ≈72 µs/site × 94 sites.

- Doorbell contract: "word 0 is Y, word 2 is Q. Both are data readiness, never CPU queue scheduling."

## 4. Verdicts

### 4c. Tip-to-tip table, round 2 (same script, same binary, same n_predict=171, medians)

| arm | r1 tok/s | r2 tok/s | median | note |
|---|---|---|---|---|
| `S_ref` reference | 59.23 | 56.86 | **57.7** | sharp arm (no LateBind) |
| `M2_h1` reference (other script) | 58.39 | 58.39 | **58.4** | independent confirmation, 1 % apart |
| `S_ref_hcq8` (`GGML_VK_HC_DOT=q8`) | 58.21 | 57.25 | 57.7 | **silently runs mode=reference, late=0** |
| `S_exact` exact-f32 | 44.17 | 37.12 | 40.7 | fallback implementation |
| `S_aggr` aggressive-q8 | 2.93 | 2.94 | 2.9 | first time it runs at all (4a) |
| `M3_h2` h=2 | 51.76 | 54.60 | 51.8 | wrong horizon |

### 4d. D9 — a short `n_predict` does not measure steady throughput

`R_h1_ref` (n_predict=60) reported 34.2 tok/s while `S_ref` (n_predict=171) reports 57.7 tok/s for the
**same configuration**. Per-request overhead dominates a 60-token measurement. Round 2's
"instrumentation costs 20 %" was this artifact, not instrumentation. All throughput arms now use
n_predict=171 and report both requests.

### 4e. `ready_us` is a sum, not a blocking event

`tp5_relay_submit_epoch_chain` accumulates `prof->relay_ready_wait_us += step.wait_us` over **all**
stages (`ggml-vulkan-collective.cpp:6272`). The "handoff is 69–75 % of the cycle" framing in round 1
treated it as one blocking wait; it is the cumulative per-stage CPU wait and is not on the critical
path by itself. Withdrawn.

### 4f. Stage semantics are now identified, not guessed

Capture-time stage identity (`[tp5-stage-map]`, added this round) gives, for every one of the 96
stages, the last node of that subgraph: `96 = 48 layers x 2`. Even stage `2L` = layer `L`'s attention
subgraph (`linear_attn_out-L` for 36 GDN layers, `attn_output-L` for the 12 full-attention layers at
3,7,11,...,47 — the 3:1 hybrid), odd stage `2L+1` = `ffn_out-L`. So stages 0..30 are **layers 0..15**.

### 4g. The head serialisation is real (verified against the timer definition)

Timer slots are recorded around specific command ranges: `q+1..q+2` is the model compute region and
the peer wait lives in `q..q+1` (p2). Therefore a rank cannot inflate "compute" by waiting.
Steady capture (chain 130, no-MTP, 1 row):

| window | r0 | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| stages 0..30 compute | 3099 | 3095 | **7339** | 3113 | **10222** |
| stages 0..30 p2 (wait) | 14847 | 14691 | 10151 | 14375 | **1745** |
| stages 31..95 compute | 6217 | 6214 | 6241 | 6238 | 6279 |
| stages 31..95 p2 | 2046 | 2053 | 2021 | 2023 | 1757 |

r0/r1/r3 run at the same per-stage rate in head (100 us) and tail (96 us); r2/r4 carry extra work in
layers 0..15 only. Independent corroboration: `ready_us=12.3 ms` for the same arm matches the head's
serialised compute. Open: whether that extra work is node-count (staging/sharding) or per-op cost.

### 4h. The aggressive-q8 cost is a memory-placement defect, not an arithmetic verdict

With the 4a fix the aggressive path runs, and its Q-sidecar CPU reduce reads the f16 payload out of
the **host-mapped VRAM** (`late_ptrs[i] = late_bcast_f16[i]`), one 2-byte element at a time:
`late_count = rows x streams(4) x late_rank(320) = 1280` per rank, x5 ranks = 6400 PCIe transactions
per stage, x96 stages. Measured `cpu_data_us = 328781` / `sidecar_data_us = 328009` per chain
(~7 MB/s) versus `exact-f32` reading the same logical payload from system RAM in `cpu_data_us = 945`.
The bcast allocation asks for `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|DEVICE_COHERENT_AMD` with **no
`HOST_CACHED_BIT`**, so the aperture is CPU-uncached; `[tp5-relay-config] bcast_cache=cached` refers
to GPU-side caching. Under test: `GGML_TP5_BCAST_HOST_CACHED=1` (`/var/tmp/tp5-bcast-cached.sh`).

### 4i. Run isolation is mandatory (self-inflicted wedge)

bg_259 (`tp5-sharp-2.sh`) was still running while later jobs started; every script's
`pkill -9 -f llama-server` killed the other run's server, and two servers on five GPUs wedged all
five gfx rings (`amdgpu ... ring gfx_0.0.0 timeout` -> reset). Arms affected: `S_aggr_hcq8` r1/r2
(0-byte JSON), `S_p1a` (DIED), bg_260 and bg_262. All scripts now take
`flock -n /var/tmp/tp5-gpu.lock` and refuse to start if a `llama-server` is already running.

### 4j. The "head window" is a per-chain transient, not a structural imbalance (RETRACTION)

Round 2 concluded that rank2/rank4 "carry 2.4-3.3x the compute of the other ranks in layers 0-15".
Three independent censuses refute the structural reading:

| check | result |
|---|---|
| command ledger per chain (`[tp5-linear-definition]`) | `dispatches=4529 barriers=3997 copies=191` **identical on all five ranks** (steady chains `1234/1429/96`) |
| node census per stage (`[tp5-stage-nodes]`) | `s=0 76 76 76 76 76`, `s=1 89 89 89 89 89`, ... **identical on all five ranks for every stage** |
| kernel policy per rank | `[vulkan-mmvq-policy] mode=forced` / `[vulkan-argmax-policy] hierarchical=enabled` identical on all five |
| PCIe link | all five Gen3 x16, single NUMA node |

The steady capture (`tp5-stagename/nomtp_map.log`) shows the real shape: **stages 0-16 are elevated on
every rank at once** (stage 1 = 522/524/522/529/526 us, < 1.5 % spread), the ranks then leave that
regime at different stage indices (r0 at 17, r1-r4 by ~20-30), and the tail is flat at 87/104 us with
r4/r0 = 1.01. The "3.16x" at stage 17 is the crossing of two decaying curves. In the earlier capture
(same chain index 130) the elevation was confined to r2/r4 over stages 0-30. **A transient whose onset,
extent and per-rank ordering change between identical runs is machine state, not work.**

Cost: head 0..30 sums 7354/8130/7815/7834/9936 us against a flat tail of ~6240 us per rank, i.e.
~2-4 ms per chain (10-15 % of a 26.7 ms chain).

### 4k. Machine state: OverDrive is not applied on all five cards

Idle sample, `amd-smi monitor -u -m -p -g all`:

| amd-smi gpu | BDF | gfx_clk | max_power | `pp_dpm_sclk` top level |
|---|---|---|---|---|
| 0 | 0f:00 (Vulkan4) | 500 | 203 W | 2: 2475 MHz |
| 1 | 09:00 (Vulkan2) | 500 | 203 W | **2: 1200 MHz** |
| 2 | 0c:00 (Vulkan3) | 500 | 203 W | 2: 2475 MHz |
| 3 | 06:00 (Vulkan1) | 2475 | 227 W | 2: 2475 MHz |
| 4 | 03:00 (Vulkan0) | 2475 | 227 W | 2: 2475 MHz |

`power_dpm_force_performance_level=high` on all five, but only two cards carry the 227 W OverDrive cap;
the other three stay at 203 W and the card at **09:00 never received the OverDrive commit at all** -
its highest DPM level is still the stock 1200 MHz, exactly the condition `restore-gpu-highspeed.sh` was
written to repair. Vulkan2 = 09:00 = the rank that looked "elevated" in the first capture. In-load
sampling (`tp5-clock-load.sh`, `A_before` vs `B_after` around a re-applied OverDrive) decides whether
this is the cause of the transient.

### 4l. Why LateBind loses on this base (mechanism, not opinion)

The p2 region is **slack, not latency**. Per-chain ledger, h=1 verify chain, reference:

| rank | span | p2 (wait) | compute |
|---|---|---|---|
| 0 | 36395 | 21288 | 14097 |
| 1 | 36234 | 21153 | 14076 |
| 2 | 36111 | 17381 | 17636 |
| 3 | 35959 | 20812 | 14116 |
| **4** | **28060** | **5366** | **21452** |

Rank 4 never waits; the other four wait ~21 ms for it, so the cycle is set by **rank 4's serial
compute** and removing other ranks' slack buys nothing. `exact-f32` shows the consequence: it converts
17.3 ms of r0 p2 slack into 14.4 ms of **late_pre critical-path work** (`S_exact`: r0 p2 3982,
late_pre 14386, span 31870); r0's span improves 4.5 ms but r4's grows from 28060 to 31261 - the
critical rank gets worse and throughput drops 30 %.

- **reference** 57.7-62.5 tok/s (three independent runs, median ~58.7) - the shipped fast path.
- **exact-f32** 40.7 tok/s: one COMPUTE->HOST barrier per late site. It cannot condemn the mechanism
  (it is a fallback implementation), but it proves the mechanism's own cost lands on the critical path.
- **aggressive-q8** 2.9 tok/s: not an arithmetic verdict. The CPU reduce reads each rank's f16 sidecar
  from **host-mapped VRAM** (`late_ptrs[i] = late_bcast_f16[i]`, dsts are those same VRAM addresses):
  `late_count` = rows x streams(4) x late_rank(320) = 1280 per rank, 6400 uncached reads per stage x 96
  stages = 328 ms/chain at ~7 MB/s. Reads round-trip, mirrored writes are posted - which is exactly why
  the f32/exact path (sources in system RAM `star_host_aligned`) costs only 945 us/chain.
  `GGML_TP5_BCAST_HOST_CACHED=1` was implemented and measured this round: **the memory type does not
  exist on gfx1030** (no DEVICE_LOCAL|HOST_CACHED), the allocation fails and silently falls back
  (`bcast_cache=cached host_cached=0`, `cpu_data_us` unchanged at 327783). A CPU-visible sidecar must
  therefore live in **system RAM (GTT, host-cached)**, not in VRAM, or the CPU must not read it at all.
- **P1A (no sidecar)** now fails structurally rather than by descriptor: `[tp5-relay-chain] handoff
  failed stage=1 epoch=98 bank=1` -> `graph_compute failed with error -1` -> SIGSEGV.

### 4m. `GGML_VK_HC_DOT=q8` silently disables LateBind

`S_ref_hcq8` reports `mode=reference desc=reference-no-latebind late=0 q8_fast=0` although launched with
`--tp5-latebind aggressive`: the HC Q8DOT pipeline is preferred by the graph builder and `hc_quant_sub`
stays empty, so the late path never arms. Throughput equals plain reference (57.7/57.7; `A3_ref_hc`
58.7/62.5). There is no fail-closed check for this coupling - every future LateBind arm must assert
`mode=`/`late=96` from the ledger, not just the CLI flag.

### 4n. (RETRACTED by 4o) The head penalty looked like a CLOCK ramp

> **Retracted.** The correlation below held for the `bg_266` run only. `bg_267` reproduced the same
> x3.37 head on a card that reports 2475 MHz, the ALU benchmark shows card 09:00 at 92 % of the best
> card, and `pp_dpm_sclk` turned out to have no predictive power at all. Read 4o before quoting any
> number from this section. The one part that still stands is the *diagnostic requirement*: the slow
> rank moves between arms, so every earlier ratio must be re-measured on a state-checked arm.

Same binary, same workload, same `n_predict`, same phase; the only factor between the arms below is
the per-card SCLK state. Metric is the per-stage `compute_us` sum over stages 0..30 ("head") and 31..95
("tail"), from the existing capture instrument:

| arm | head 0..30 by rank | tail 31..95 by rank |
|---|---|---|
| governed (`A_before`) | **10429 / 10369 / 10625** / 3118 / 3054 | 6473 / 6340 / 6745 / 6242 / 6108 |
| OD re-applied (`B_after`) | **10204 / 10552 / 10458** / 3123 / 3055 | 6422 / 6380 / 6472 / 6243 / 6113 |

Re-applying the documented OverDrive state changes **nothing** (`restore-gpu-highspeed.sh` ends with
`echo high`, which releases any level selection). The split is 3 slow / 2 fast in both arms.

Mapping and the third independent signal (1 Hz `amd-smi` sampling *while the requests were decoding*):

| amd-smi | BDF | Vulkan/rank | in-load SCLK median | frac < 1500 MHz | head sum |
|---|---|---|---|---|---|
| gpu0 | 03:00 | V0 / r0 | **500** | 11/15 | 10429 |
| gpu1 | 06:00 | V1 / r1 | **500** | 11/15 | 10369 |
| gpu2 | 09:00 | V2 / r2 | **500** | 11/15 | 10625 |
| gpu3 | 0c:00 | V3 / r3 | 2475 | 1/15 | **3118** |
| gpu4 | 0f:00 | V4 / r4 | 2475 | 1/15 | **3054** |

1:1, no exceptions. `pp_dpm_sclk` agrees: the two fast cards carried `*` on `2475Mhz` even at idle,
the other three sat on `0: 500Mhz`. All five have `power_dpm_force_performance_level=high` and
`power_dpm_profile_mode=0 BOOTUP_DEFAULT`, i.e. the *policy* is identical; only the card state differs.

Shape of the penalty (steady 1-row capture, per-stage `compute_us`): a **constant x3.37 multiplier that
decays per rank after a rank-specific stage index** (r0 escapes at 17, r2/r3 at 19, r1 at 20, r4 at 30),
identical on attention (even) and FFN (odd) stages. 3.37 = 2475/734 MHz. Once a rank escapes, every
rank's per-stage value is *identical* (stage 95: 118/117/118/118/120) - so there is no structural
per-rank or per-card work difference at all, and no operator/kernel difference.

- **The chain floor is 96 x ~98 us = 9.4 ms = ~105 tok/s for the 1-row no-MTP path.** Observed 16.9 ms
  (full penalty) / 18.4 ms measured per token. The whole gap is the first ~10 ms of clock ramp.
- amdgpu's governor needs roughly a 10 ms load window before it raises SCLK; the chain is only 9-21 ms
  and the inter-chain gap is longer, so three cards never leave the idle clock inside a chain.
- The node/command censuses (`[tp5-stage-nodes]`, `[tp5-linear-definition]`) were already identical
  per rank; this is the mechanism that those censuses left unexplained.

### 4o. The clock-table reading does NOT explain the slow rank (retracts the causal claim in 4n)

The same single-factor A/B was repeated with a proper concurrent sampler (1201 Hz during decode) and
the acceptance test is whether the *slow rank* is the *low-clock card*:

| card | in-load `pp_dpm_sclk` median | frac < 1500 | head 0..30 | tail 31..95 |
|---|---|---|---|---|
| 0f:00 (r0) | 2475 | 0.28 | 3124 | 6261 |
| **06:00 (r1)** | **2475** | 0.43 | **10704** | 6676 |
| **09:00 (r2)** | **1200** | **1.00** | 5024 | **10067** |
| 0c:00 (r3) | 2475 | 0.28 | 3146 | 6294 |
| 03:00 (r4) | 2475 | 0.28 | 3079 | 6157 |

`06:00` reports 2475 MHz and carries the slow head; `09:00` reports 1200 MHz and carries a slow *tail*,
not the head. The table has no predictive power. On top of that the instrument is visibly broken:
`min = 0`, `p10 = 0`, and `frac<1500 = 0.28` even on the cards that are demonstrably fine, i.e. it
reports the `S: 0Mhz` sleep row roughly a quarter of the time while the GPU is decoding.

Two further facts kill the clock reading as a mechanism:

- The user's own arbiter, the 1 s sustained ALU benchmark (`/var/tmp/tp5-alu1s`), gives
  `0f:00 18.03 / 09:00 16.53 / 0c:00 18.43 / 06:00 17.33 / 03:00 17.36 TFLOPS` - card 09:00 is 92 % of
  the best card, so its clock, whatever the table says, is not what limits the chain.
- Re-applying OD and pinning changed nothing twice (`A_high` vs `B_pinned`: 40.15-40.71 vs 40.02-40.23
  tok/s; head/tail sums identical to within noise). `amdgpu_pm_info` exposes the real SMU state
  (`PSTATE_SCLK`) and it does not track the `pp_dpm_sclk` table either.

What does survive, and is the same in every capture:

- node census identical per rank per stage (`s=0: 76 76 76 76 76`, `s=1: 89 89 89 89 89`, ...),
- command ledger identical per rank (`dispatches=4529 barriers=3997 copies=191`),
- one stage band (head **or** tail, it moves between runs) running at **x3.37** for every rank, with
  *identical* per-stage values across ranks everywhere else (stage 95: 118/117/118/118/120),
- end-to-end throughput stable inside a run (40.0-40.7 over six requests) and across arms, while the
  *identity* of the slow rank changes (r0/r1/r2 in `bg_266`, r1-head + r2-tail in `bg_267`),

i.e. a shared, short-lived stall that gates whichever rank arrives at it, not a per-card property.
The candidate that fits all of the above is the host relay: one CPU star-reduce per stage, whose
backlog at chain start is exactly a decaying, shared stall that hits the rank gated by the ready word.
`bg_268` turns on `GGML_TP5_PROFILE_RELAY_STAGES=1` and compares the per-stage `ready_all_us`,
`cpu_data_us` and `total_us` of stages 0..30 against 31..95 to settle it.

**Still true from 4n**: the arm-to-arm clock/state split is real (three of five cards sat at a 500 MHz
median in load in the `bg_266` run) and it does move between arms, so **every earlier ratio in this file
that depended on which rank was slow must be re-measured on a state-checked arm.** The specific causal
chain "500 MHz -> x3.37 -> 10.4 ms head" is however **withdrawn**: `bg_267` reproduced the x3.37 head on
a card that reports 2475 MHz.

**Diagnostic requirement from now on**: every arm records the in-load clock table *and* the head/tail
sums; an arm whose slow rank does not match the stated cause is not used as evidence for it.

### 4p. The bottleneck is ONE rank, stably: rank2 = Vulkan2 = 09:00, +56 % per stage

`GGML_TP5_PROFILE_RELAY_STAGES=1` prints `ready_rank_us=` - the CPU-observed time from arming the
relay to each rank's ready word, per rank, per stage. That is the per-stage per-rank latency the chain
is actually gated by, and it supersedes every earlier per-stage metric in this file:

| rank | card | mean | median | min | max |
|---|---|---|---|---|---|
| 0 | 03:00 | 124.58 | 129.53 | 0.17 | 214.85 |
| 1 | 06:00 | 124.47 | 128.70 | 0.26 | 218.82 |
| **2** | **09:00** | **193.81** | **203.64** | 34.66 | 351.15 |
| 3 | 0c:00 | 125.14 | 128.50 | 76.87 | 214.38 |
| 4 | 0f:00 | 123.15 | 128.46 | 108.13 | 209.13 |

**Slowest-rank histogram over the 96 stages: rank2 = 95, rank4 = 1, everyone else 0.** It is not a
race and it is not a head/tail effect - per rank, head(0..30) / tail(31..95) is 1.01-1.05, i.e. this run
has **no head penalty at all**. The bands reported in 4n/4o were the same single-rank deficit showing
up wherever the capture happened to land.

Arithmetic: 96 x 193.8 us = **18.6 ms per chain**, which is exactly the best no-MTP token time ever
measured here (18.4 ms = 54.4 tok/s). With rank2 at par (124 us): 96 x 124 = **11.9 ms -> ~84 tok/s**.
The whole 100 tok/s question is currently one card.

### 4q. LateBind's ceiling is 0.4 ms/chain, measured on the shipped path (not the fallback)

Same per-stage instrument, mean over the chain:

| band | ready_first | ready_all | skew | arm | cpu_data | generation | total |
|---|---|---|---|---|---|---|---|
| head 0..30 | 121.89 | 199.72 | 77.80 | 0.066 | **3.886** | 0.152 | 203.91 |
| tail 31..95 | 121.02 | 192.84 | 71.79 | 0.058 | **3.761** | 0.120 | 196.85 |
| ratio | 1.01 | 1.04 | 1.08 | 1.14 | **1.03** | 1.27 | 1.04 |

`cpu_data_us` is the CPU's actual reduce time: **3.9 us per stage, flat, 2 % of the ~200 us relay
step, 0.37 ms per chain**. Everything else in the step is waiting for the five ranks to publish
(`ready_all` = 193-200 us = 97 %).

So the maximum LateBind can ever recover here is ~0.4 ms out of an 18-25 ms chain (~1.5 %), *before*
paying for the mechanism - and at `h=1` there is not even that much, because this number already
includes the CPU's full per-stage cost on the shipped path. That is a stronger statement than the
earlier negative verdicts, and it does not depend on the `exact-f32` fallback at all.

Instrumentation cost note: this arm ran at 36.4 tok/s against 40.5 tok/s for the identical
configuration without `GGML_TP5_PROFILE_RELAY_STAGES` (~11 % - 192 fprintf pairs per chain on the hot
CPU path). Never compare a throughput number across instrument states.

### 4a. Root cause of the aggressive-q8 SIGSEGV (found, fixed, rebuilt)

Real backtrace under gdb (`/var/tmp/tp5-aggr-gdb/app-aggressive.log`):

```
#0  libvulkan_radeon.so                       <- crash inside RADV
#1  tp5_update_storage_set(..., count=2)      ggml-vulkan-collective.cpp:2963
#2  tp5_record_plan(...)                      ggml-vulkan-collective.cpp:4109
#3  ggml_backend_vk_tp5_submit_epoch_chain    ggml-vulkan-collective.cpp:7583
#4  ggml_backend_meta_graph_compute / llama_decode (warmup graph)
```

The q8 weight-repack path allocates its descriptor sets as
`plan.late_pack_ds.resize(2 * c.n_ranks)` with slots `[0, n_ranks) = down pack`,
`[n_ranks, 2*n_ranks) = up pack`, and the record site indexes
`plan.late_pack_ds[i + which * c.n_ranks]` for `which = 0,1`. **Only the down half was ever
allocated**, and the record-site guard tested `late_pack_ds[i]` only. So on the first replay of
`cmd_late_pre` the up-half update passed `dstSet = VK_NULL_HANDLE` to `vkUpdateDescriptorSets`
(no validation layer active) -> SIGSEGV in the driver.

Fix (semantic, not a rollback): allocate the up-pack set as well, and extend the guard to require
both sets before the pack dispatch. `late_down_packed_buf`/`late_up_packed_buf` were already
allocated, so the descriptor set was the only missing piece.

Consequence: **the fast LateBind modes (aggressive-q8, P1A) were never runnable because of this
single missed allocation** — not because of a mechanism limit. The earlier "LateBind is negative"
verdict, and the round-1 numbers behind it, say nothing about the fast modes.

### 4b. First-31-stage asymmetry (new, steady-state, reproduced twice)

In a *steady* captured chain (chain #130 of r2) the first 31 stages are asymmetric on one rank:

| stages | rank0 | rank1 | rank2 | rank3 | rank4 |
|---|---|---|---|---|---|
| 0..30 | ~100 µs | ~100 | ~100 (525–683 at s=1,2) | ~100 | **~355** (525–683 at s=1,2) |
| 31..95 | ~100 | ~100 | ~100 | ~100 | ~100 |

Per-stage max/mean > 1.5 for exactly stages 0..30 and never afterwards; head waste ≈ 4.9 ms of a
27.3 ms chain (≈18 %). The 3-row capture shows the same window with **two** ranks slow (rank2 and
rank4 at 2.4×). Cause not yet identified: `GGML_TP5_PROFILE_RELAY_STAGES=1` gives `elems=`/`direct=`
per stage and is the next measurement (`/var/tmp/tp5-stagemap.sh`).


- **`exact-f32` is withdrawn as a performance candidate** — but for a *structural* reason, not a
  mechanism limit: it is the fallback that pays one HOST-domain barrier per late site
  (94 sites × ~72 µs = 6.8 ms/chain) while doing 36 % *fewer* dispatches than the reference.
  Using it to judge LateBind was the round-1 error (D4).
- **`aggressive-q8` / `p1a-nosidecar-q8` remain unjudged**: they have never run. The crash is a
  prerequisite, not a result. Suspect: the plan selects the q8dot HC-down consumer
  (`q=q8dot-approx`) while `device->pipeline_hc_down_silu_q8dot` is only created under
  `GGML_VK_HC_DOT=q8`, and the collective's q8 sidecar pipelines only under `q8_wave32`
  (integer dot + 32-lane subgroup) — a silent capability coupling with no fail-closed check at
  plan activation.
- **`h=1` replaces `h=2` as the natural-text configuration** pending an adaptive-horizon policy.

## 5. Open items (in priority order)

1. gdb backtrace of the aggressive SIGSEGV (`/var/tmp/tp5-gdb2.sh`, unique port, core enabled).
2. `GGML_VK_HC_DOT=q8` coupling test (`/var/tmp/tp5-hcdot-test.sh`): if aggressive runs with that
   variable set, the root cause is configuration coupling and the fix is a fail-closed check plus
   auto-enable, not a rewrite.
3. Phase-labeled steady captures of a 1-row decode chain vs a 2-row verify chain to put the
   10.6 ms marginal row on a named side (MoE expert reads vs per-stage compute).
4. Only after (1)–(3): re-run the tip-to-tip pair `h=1 reference` vs `h=1 aggressive` with the ledger,
   ≥4 steady requests, drift control, and the same-work metric.

## 5b. Corrections to round-2 claims (found by the new analyzer)

- **Retracted: "systematic rank imbalance, rebalancing is the biggest lever."** In the captured chain the
  slow-rank effect exists only in the head third (stages 0..31, e.g. rank2/rank4 ≈ 12.3/12.4 ms vs
  rank0/1/3 ≈ 5.1 ms) and disappears in mid/tail (all ranks within 2 %). Rank4 also carries the least
  `p2` wait (3.4 ms head) precisely because it is the last producer. Steady-state per-stage budget:
  `compute ≈ 155 µs`, `p2 ≈ 60 µs`, `p1 ≈ 0.1 µs`, `late_pre = 0` on all five ranks.
- Consequence for priorities: the steady chain is (a) per-stage compute ≈ 96 × 155 µs ≈ 14.9 ms and
  (b) doorbell/handoff ≈ 96 × 60 µs ≈ 5.8 ms. Shard rebalancing is off the table as a steady-state lever.
- Hardware check (non-intrusive): all five cards report `power_dpm_force_performance_level=high`, identical
  OD tables, and `mclk = 1000 MHz`. card2 (BDF 09:00, = Vulkan2 = rank2) is the only card whose
  `pp_dpm_sclk` table caps at `2: 1200Mhz` instead of 2475 MHz; per the operator, sysfs sclk is known to be
  stale on that card, so this is recorded as **unresolved environment asymmetry, not established cause**.
  Vulkan index → BDF map (vulkaninfo UUIDs): Vulkan0=03:00, V1=06:00, V2=09:00, V3=0c:00, V4=0f:00.

## 6. Evidence


- Row scaling: `/var/tmp/tp5-row-arms/{M1_target_only,M2_h1,M3_h2,M4_h4}.{log,_r*.json}`
- Phase map: `/var/tmp/tp5-phase-map.sh` → `/var/tmp/tp5-phase-map/`
- Sharp pair + ledger: `/var/tmp/tp5-sharp-1.sh` → `/var/tmp/tp5-sharp/`
- HC_DOT coupling: `/var/tmp/tp5-hcdot-test.sh` → `/var/tmp/tp5-hcdot-test/`
- gdb: `/var/tmp/tp5-gdb2.sh` → `/var/tmp/tp5-aggr-gdb/`
