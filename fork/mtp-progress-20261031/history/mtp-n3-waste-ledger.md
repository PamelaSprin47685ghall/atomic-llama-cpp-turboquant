# Qwen3.8-Flash-Next MTP (TP5, 2 windows, depth 2 → N3 verify): N3 waste ledger
2026-10-27. Read-only. No builds, no GPU runs, nothing promoted. Selected install is unchanged.
Root: `~/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024` (EV below).
Rules followed: microbench or mock first, no formal 512x2, no changes to `q2-unpack-joint-l2-a-install` or `single-sync-compose-install`.

## 0. Correction to the earlier report
Earlier I listed "MoE +7.0 ms", "linear_attn +2.1 ms" and "~9 ms device gap" as if they were separate buckets. They are the same time counted twice. In `gpu-wait-n3-vs-n1-analysis.json`, the "gap" between relay stages is compute plus launch, not idle time.
The real rank0 split of N3−N1 is 30.50 − 18.71 = **11.8 ms**:
- gap/compute: +9.08 ms (25.32 vs 16.24). By stage that is ffn_moe +7.03 ms and linear_attn +2.09 ms.
- relay wait: +2.54 ms (3.96 vs 1.42). On rank4 it is only +0.78 ms.
- relay consume: +0.46 ms (1.34 vs 0.88).

## 1. Smoking gun: the N3 graph is 54% bigger because every t1-only fusion turns off
Native captured graphs, same provider, rank0 (`mtp-critical-attribution-20261027/native32x2/graph-0x63862cf3c100-{1,2}.csv`):
**N1 = 2476 nodes, N3 = 3825 nodes, +1349 nodes.** Node-family diff (N1 → N3):

| family | N1 | N3 | what is lost |
|---|---:|---:|---|
| router: `topk_moe_cuda<512>` | 48 | 0 | fused softmax+top10+norm becomes soft_max 48 + argsort 48 + get_rows ~49 + sum/reduce ~48 + clamp 48 + div 48 + cpy_scalar_transpose 48. **About +290 nodes** |
| HC: `hc_down_silu_t1_q8`, `hc_owner_tail4_t1_q8`, `hc_combine4_t1` | 3×96 | 0 | becomes mmvq q8_0 196 + op_mul +231 + op_add +182 + hc_tail4_rows 97 + repeat +96 + reduce_rows +96 + mmv_f f32 96. **About +700 nodes**, the largest family |
| `quantize_q8_1` | 241 | 532 | +291 standalone re-quantize nodes |
| linear_attn/GDN | — | — | softplus +36, l2_norm +36, silu +62, concat_cont→non_cont 37. **About +170** |
| Q2 down: `q2_down_weighted_sum_k128_t1` | 48 | 0 | becomes `mtp_q2_down_subwarp` 48 + `mul_mat_vec_q_moe` 95. Similar count, unfused weighted-sum |
| bf16 mmv: `mul_mat_vec_bf16_load4` | 181 | 0 | becomes generic `mul_mat_vec_f<bf16>` 181. Same count, slower generic body |

Why the fusions turn off:
- `ggml/src/ggml-cuda/ggml-cuda.cu:4467` allows the topk_moe overlap exception only when `ggml_nrows==1`. For N3 the allocator aliases the normalized output 128 B into row0's probabilities, so the overlap check at 5401/5416 vetoes the fusion. This is the same "alias veto" that router-exact hit.
- `mmvf.cu:1102/1298/1331` have `ne11 == 1` guards, so the HC t1 kernels (`mmvf.cu:160,179,311,362`, `binbcast.cu:435,476`) and bf16_load4 are N1-only.
- `ggml-cuda/mmvq-q8-f32.hip:172,813` hardcodes `q2_down_weighted_sum_k128_t1` for t1 only.

Dispatch cost per node: synthetic captured dependent chain (`mtp-critical-chain-20261026/dispatch-pacing-fit.json`) costs **2.4–3.2 µs per node** on these RX6800s (3016 nodes ≈ 7.4 ms). So the +1349 nodes cost **≈3.3–4.3 ms from launch/admission pacing alone**, before counting the extra HBM round trips of the intermediates.

## 2. Waste ledger: per-window steady cycle, per verify (2.34 tok/verify)
Formal row-tail on512x2: 9000 ms / 219 verifies = **41.1 ms per cycle → 56.8 tok/s**.

| # | bucket | class | ms per verify | evidence | tok/s if this bucket alone → 0 | cumulative |
|---|---|---|---:|---|---:|---:|
| F | cold graph build/alloc per (ctx,N). N1 is rebuilt twice; N2/N1 rebuild again at the tail; 130–230 ms each | removable host scheduling | **5.6** (1.22 s / 219) | `formal-cold-interval-counterfactual`; `TP5_DIAG_GRAPH` in `hc-mean-rows-candidate-20261024/on32x2/stderr.log` | 65.9 | 65.9 |
| E | draft + commit + submit serialized after the verify | removable draft/commit not overlapped | **~5.0** (35.5 − 30.5) | `cycle-timing`: draft ~5 ms gross, commit median 8.6 ms (instrumented) | 64.8 | 76.7 |
| A | +1349 graph nodes from lost t1 fusions (pacing only) | removable inter-kernel/launch + unfused router/HC duplication | **3.3–4.3** | graph diff above + dispatch-floor | 61.9–63.6 | 86–89 |
| B | N3 rank imbalance / relay wait (rank spread in N3 1.8 ms vs 0.54 ms in N1) | dual-request + split-draft-device interference, relay handshake | **1.3–2.5** | gpu-wait rank medians | 58.8–60.6 | 90–98 |
| D1 | unfused bodies: intermediate round trips, generic mmv_f bf16, standalone quantize | removable unfused body | **~2–4 (not measured)** | difference left after A/B/C | — | ~100–115 |
| D2 | real 3-row work: up to 3× unique routed experts, 3-row FA, 3-step GDN | **necessary, keep** | **~1.5–3 (estimate)** | latency-bound decode; byte budget is small | — | — |
| C | relay consume of 3× F32 rows | necessary under the F32-wire contract | 0.46 | gpu-wait | — | — |

Reading the cumulative column: removing F, E, A and B lands at **~90–98 tok/s**. Removing part of D1 as well crosses 100. Ashley's ~100 ceiling is reachable without touching acceptance.
Also seen, but too contaminated to count:
- An instrumented run (36 tok/s) shows ~30% of calls with submit ≈34 ms, i.e. one window blocking for the other window's whole graph.
- `historical-on32-n3-graph-spans` shows the same pattern: compute_us alternates 40 ms vs 1.4 ms between ctxs.
This is a candidate interference bucket that has not been measured in formal runs.

## 3. Ranked cuts
1. **F – prewarm every row shape (N1, N2, N3) per ctx before the first decode and pin them in the graph cache.** No kernel change. About +9 tok/s (→~66). The cost moves into TTFT.
2. **A+D1 – N=3-capable versions of the t1 fusions, in this order:**
   - (a) HC trio (+~700 nodes)
   - (b) topk_moe for N=3 (+~290 nodes). Fix it by giving the router output a separate scratch buffer so it no longer aliases row0 probabilities, then call the existing multi-row `topk_moe_cuda`. Bonus: N3 routing becomes per-row identical to the N1 kernel.
   - (c) fuse quantize_q8_1 into consumers (+291)
   - (d) `q2_down_weighted_sum` for t3
   - (e) bf16_load4 for ne11 ≤ 3
   - (f) GDN softplus/l2/concat
3. **E – start draft step 1 on the row-0 hidden state while commit runs; let the next N3 submit overlap the 4.4 ms tail.** Up to +10 tok/s at the end of the chain.
4. **B – rebalance rank work for N3 (rank4/rank2 are slowest) and check whether split-draft devices steal from the other window's verify.**

## 4. Microbench / mock gates (fastest first; no formal 512x2)
- **M1 (mock, ~minutes, single GPU): replay the real N1 vs N3 node lists as a synthetic captured graph.**
  - Extend `mtp-blueprint-20261026/mtp-critical-chain-20261026/dispatch-floor.hip`. Read the node list (grid/block per node) from `native32x2/graph-…-1.csv` (2476) and `-2.csv` (3825), launch no-op kernels with identical grid/block, and keep a linear dependency chain.
  - Measure 1 stream and 2 streams (two windows).
  - Gate: Δ(N3−N1) ≥ 3 ms → A is confirmed as the top kernel-side cut. < 1 ms → drop A and go to D1.
  - Second variant: "N3 with the HC trio restored" (subtract the about 700 HC nodes, add 3×96) gives the expected win of cut 2a before any real kernel is written.
- **M2 (micro, exact): topk_moe N=3 with separate scratch vs the current 7-kernel chain.** → **DONE 2026-10-27: PASS (exact + 0.25×). See "M2 results" below.**
  - Real shape: logits [512×3], top10, clamp/scale args from the graph. Extend `router-snapshot/micro.cpp`, call `ggml-cuda/topk-moe.cu`'s kernel with a distinct output buffer.
  - Bitwise compare: ids, weights, and N1 single-row results row by row.
  - Gate: exact, and captured-graph time ≤ 0.5× the chain's.
- **M3 (micro): `hc_down_silu`/`hc_owner_tail4`/`hc_combine4` rows=3 kernels vs the current generic chain**, in a captured graph on one device. Start from `mmvf.cu:160/179` and `binbcast.cu:435`. Bitwise vs the generic N3 chain. Time with the fixed-slot observer (`mtp-fixedslot-observer-20261027/body-local.cpp`, cost gate already passed at 1.009×).
- **M4 (mock, no kernels): cold-build prewarm.**
  - Instrumentation is already present: env `TP5_DIAG_GRAPH`.
  - Use a short 8x2 or 32x2 diagnostic in a NEW isolated candidate dir, prewarm patched in.
  - Gate: zero `build_us>0` lines after the first decode, and interval[0..3] < 50 ms.
  - This is a short diagnostic, not formal.

## 5. Candidate evidence quality (claims of gap reduction already in tree)
- router-two-stage: numerically exact. Formal pair gave 0.926/0.973 and the reverse pair 1.03/1.006, with a concurrent-build confound. → noise, no gain. It never addressed the HC nodes.
- router-snapshot: single-CTA micro, 24.7 vs 23–24 µs. → no gain. It measured the body only, not node/dispatch pacing.
- compact-relay-pool: All511 regressed in the OFF-ON pair. → no gain.
- mtp-critical-attribution (gate0 in-body): observer perturbation 18.3×. → timings invalid. The node census is valid, and it is the source of §1.
- fixed-slot observer: local cost gate passes (1.009×); never integrated into a model. → use it as the timer for M3, not as a speedup.
None of these targeted graph node count or cold builds, which are the biggest buckets above.

## 6. Do NOT touch
- The selected installs `q2-unpack-joint-l2-a-install` and `single-sync-compose-install`, and the row-tail selected config.
- Contracts: 2 independent requests, KV 262144, FA on, F32 relay, same prompts and weights.
- Already ruled out, do not retry: dense tuning, expert-group/GU-tile microkernels, shared-context nseq2, paired projection, hc-combine-order (numeric failure), mapped-label observer.

## Dual-window theory vs implementation (2026-10-27)

Theory: hide all transfer if T≤C by phase-offsetting windows (A in C ⇒ B in T).

Measured MTP steady (timing.cycles mock):
- C≈36ms, T_commit≈8.7ms ⇒ T≤C holds, yet uncovered commit ≈7.1ms/cycle
- commit∩peer_wait cover only ~0.20–0.31
- commit∩commit median 0.73 (same-phase T)
- submit≥10ms spikes almost all during peer_wait (~33ms ≈ peer C)
- wait∩wait p90=0.95 (lockstep compute)

Implementation smell:
- `TP5_MTP_READY_PAIR` uses `pair_.meet()` at Meta submit-ready → **same-phase barrier**, not half-cycle offset
- `TP5_MTP_POST_SUBMIT_UNLOCK=0` on qualified row-tail; unlock=1 candidate did not beat unlock=0 (~50 vs ~51 tok/s) and worsened TTFT

Bug stance: jitter/spikes treated as bugs; do not dismiss as noise.

## M1 results (2026-10-27, mock only, single GPU, no model, no 512x2)
Dir: `mtp-endtoend-20261024/mtp-n3-dispatch-floor-m1-20261027/`. Real rank0 node lists from `native32x2/graph-0x63862cf3c100-{1,2}.csv` (window1 = 0x63863a0924c0) were replayed as empty kernels with identical grid/block/shared and 227 D2D memcpy nodes, in a linear captured hipGraph. 40 reps after 5 warmups. Each line below is dev0 / dev2 / dev4, p50 in ms.

| plan | nodes | 1-stream | tiny (1 block × 64 threads, dev0) | 2-stream union (dev0) |
|---|---:|---:|---:|---:|
| N1 | 2476 | 6.51 / 7.06 / 6.49 | 6.10 | 6.81 |
| N3 | 3825 | **10.45 / 11.26 / 10.42** | 9.62 | 10.84 |
| N3 with router restored (−336 nodes, +48 topk_moe) | 3537 | 9.75 / 10.51 / 9.73 | 8.70 | — |
| N3 with HC trio restored (−654 nodes, +288 trio) | 3459 | 9.58 / 10.60 / 9.57 | 8.51 | — |
| N3 with both restored | 3171 | 8.89 / 9.60 / 8.86 | 7.80 | 9.40 |

- **Gate PASS:** N3−N1 = **3.9–4.2 ms** of pure launch/admission floor, which clears the ≥3 ms bar. Bucket A is confirmed as the first kernel-side cut. Cost is ≈2.7 µs per node. Most of it is node count, not waves: the tiny variant still shows 3.5 ms.
- Expected gains:
  - router fusion for N=3: −0.69 ms
  - HC trio for N=3: −0.87 ms (even though the restored trio has more waves)
  - both: −1.56 ms
- After both, 2.4 ms still separates N3 from N1. That is the remaining +~650 nodes: quantize_q8_1 +291, bf16→generic mmv_f, GDN softplus/l2/concat.
- The absolute floor is large: an empty N1 graph is already 6.5 ms per rank, about 35% of the real N1 critical path (18.7 ms). An empty N3 graph is about 34% of 30.5 ms.
- **2-stream:** two windows' graphs on the same GPU overlap almost perfectly (union 10.84 vs 10.45 for one). The dispatch floor is per-chain latency, not a device throughput limit.
- **M1b** (`m1-contend`): a 400-node draft graph launched while the peer's N3 verify is running busy kernels (0/4000/8000 cycles per node) takes 1.11 → 1.24 → 1.35 ms. The verify slows by <1%, and stream priority makes no difference. The hardware can hide T under the peer's C.

## Dual-window protocol carpet scan + fixes (isolated: `mtp-async-protocol-candidate-20261027`, built, NOT run, NOT promoted)
Source of the selected binary 1160f624 is `mtp-post-submit-unlock-candidate-20261025/runtime.cpp`. The selected row-tail config runs `TP5_MTP_READY_PAIR=1`. New binary sha 91dbeb47. Libraries are copied from row-tail (hip c4a1634f, llama bc534172), and libggml-base is relinked with the patched Meta. Each knob can be set to 0 to roll back.

Contradictions found and fixed:
1. **Same-phase barrier ×3 per verify:** `phase_pair.meet()` before the target, at Meta submit-ready, and after the target, plus a forced synchronous draft drain before the target. Fixed with `TP5_MTP_ASYNC_PROTOCOL=1`, which ignores READY_PAIR and POST_SUBMIT_UNLOCK.
2. **Doorbell inside meet():** `condition_variable::wait_for(1ms)`. Replaced with a spin. Host micro: release p50 3.6 µs → 0.25 µs.
3. **Cross-window submission gate on std::mutex (futex sleep), held for the whole `llama_decode`** including host prepare. Fixed two ways:
   - The lock is now a FIFO ticket spin lock. Handoff p50 1.0 µs → 0.11 µs.
   - `TP5_MTP_SUBMIT_ONLY_GATE=1` holds the lock only between the Meta submit-ready and submitted hooks, i.e. only around the async 5-rank enqueue. The rollback-replay decode is gated the same way.
4. **One shared CPU relay courier thread round-robins both windows,** so A's fold/publish serializes with B's. Fixed with `TP5_RELAY_SERVICE_PER_WINDOW=1`: one courier thread per window.
5. **HIP ScheduleAuto:** host stream/event waits can block on an interrupt. The ggml-cuda spin flag is only set for cc 12.1. Fixed with `TP5_HIP_SCHEDULE_SPIN=1`, which sets hipDeviceScheduleSpin on all devices via dlsym before backend init.
6. **Meta relay-snapshot try-lock retry used `std::this_thread::yield()`.** Now uses a pause spin.

Remaining (not fixed, with reasons):
- R1: the draft's own readback (`llama_get_logits_ith(dctx)`) is synchronous on the window's own chain. That is a real data dependency, not a cross-window violation. Hiding it needs speculative pipelining (bucket E).
- R2: the relay device wait kernel polls with `s_sleep(2)` backoff. That is still a device busy-wait, so it is acceptable.
- R3: `split_state_mutex` in Meta prepare. It covers short cache lookups and only matters if two prepares overlap.
- R4: `hipGraphInstantiate` via std::async on relay rebuild. Cold path only, bucket F.
- **WARNING (historical evidence):** in the `mtp-ready-pair-candidate-20261025` A/B, READY_PAIR=0 was *slower* than READY_PAIR=1 (35.1/35.6 vs 41.4/42.0 tok/s; draft_first 2.5–3.5 s vs 0.9 s; submission_queue 0.73/1.75 s vs 0.28 s). M1b shows the GPU itself can overlap a draft under the peer's verify. So that regression came from host-side serialization that meet() was masking: whole-decode lock, shared courier, blocking waits. Fixes 3–5 target exactly that. Fix 1 without 3–5 would likely regress.
- Math nuance: removing the barriers only recovers peer-wait (E[max]−E[own]) plus cross-window serialization. A window's own C→T chain (verify → draft → next submit) is a true dependency, and dual windows cannot hide it from that window's own per-token latency.

### Scan pass 2 (lens per Ashley: CPU fold path is just transit latency, so it is not a violation; violations = sleep/doorbell/same-phase meet)
7. **The relay device wait kernel slept with `__builtin_amdgcn_s_sleep(2)` in its poll loop**, so the GPU wave left busy-wait. Fixed: the sleep is removed (`hip-relay/relay.patch`).
   - libggml-hip was relinked from row-epilogue's exact object set, swapping only `relay.hip.o`.
   - A control relink with the original object reproduced the selected **c4a1634f bit-for-bit**, which proves provenance.
   - The new lib is 202a55b5 and lives in the candidate `install/lib`.
- Under the new lens, fix 4 (one courier per window) stays: it removes a cross-window serialization; the reduce math is unchanged. CPU fold latency by itself is reclassified as not a violation.
- Rechecked, consistent with the protocol:
  - relay `observe()` polls `hipEventQuery`;
  - every host wait loop uses `pause_cpu`;
  - the relay provider mutex is per window, so nothing crosses windows;
  - Meta synchronize holds no lock;
  - the watchdog's 10 ms sleep is off the critical path.
- Remaining violations in the hot path: **none found**. Not violations: the draft readback (own dependency) and the cold rebuild via std::async.

### CPU budget (Ashley: CPU idles when not relaying, and ngram competes for cores, so do not busy-wait across all cores)
8. **Spin confined to one dedicated physical core per window**, controlled by `TP5_SPIN_CORES` (auto, `0` to disable, or explicit `21:43,20:42`).
   - Auto layout: window0 = CPU 21 for the worker and its hyperthread sibling 43 for the courier; window1 = 20 and 42.
   - At startup, the process (and every thread it creates after that: HIP runtime, ggml/OpenMP pools, main, watchdog) is excluded from those 4 CPUs. That leaves 40 CPUs free for ngram and other work.
   - Each per-window courier is pinned to its sibling CPU for its whole life.
   - The window worker migrates onto its dedicated CPU only while busy-waiting: the verify wait, the rollback-replay wait, and a contended ticket-lock spin. It goes back to the shared mask for host prepare and draft, so OpenMP teams it forks never inherit a one-CPU mask.
   - Micro: the pin+unpin round trip is p50 3.4 µs, p99 20 µs, max 71 µs. That is 2 syscalls per ~41 ms cycle. Watch the p99 as a possible jitter source; if it shows up, the alternative is to pin the worker for its whole life.
   - Still outside the dedicated cores: HIP ScheduleSpin inside the draft-readback sync (≤ ~4 ms per cycle on the worker's shared-mask CPU), and ggml-cpu/OpenMP spin-waits if any CPU ops run (`--threads 22`). Not patched.
- Candidate binary is now **331a0ae9**, libggml-hip 202a55b5. Built, NOT run, NOT promoted.

## duplicate-compute 邪修 (carpet scan, 2026-10-27, E5 executor; read-only except one isolated candidate, no GPU touched)
Lens (Ashley): duplicating compute to cut single-path latency is 邪修. Compute once, pay the communication, and let the dual window hide T (T≤C holds).
Sources: selected lineage `mtp-hc-row-tail-candidate-20261025/sources/{llama-model.cpp,qwen4exp.cpp}`, `mtp-ready-snapshot-consolidated-20261025/sources/ggml/src/ggml-cuda/mmvq.cu`, row-epilogue `ggml-cuda.cu`, async `runtime.cpp`, rank0 native graphs (`mtp-blueprint-20261026/mtp-critical-attribution-20261027/native32x2`), GGUF tensor sizes (header parse).
Selected env (row-tail on512x2 `command.json`): `TP5_HC_REPLICATED_COMPUTE=1`, `GGML_TP5_HC_ROUTER_OWNER=0`, `GGML_TP5_SHARED_EXPERT_SHARD=0`, `TP5_MTP_SPLIT_DRAFT_DEVICES=1`, `GGML_TP5_MTP_RS_RING=1`, `LLAMA_GDN_REPLAY=0`.

Rank0 graph census confirms that every rank runs the full mirrored ops listed below:
- `mul_mat_vec_q[320]` / `[10240]`: full HC down and up
- `mul_mat_vec_f[512]` ×48: full router
- `mul_mat_vec_q[12288]` ×12, `[512]` ×24, and `flash_attn_ext_vec` over 24 heads: full attention
- `mul_mat_vec_q[640]`: full shared expert

Bytes below are per rank per verify. "Wasted" = 4/5 of the bytes, i.e. what the other four ranks read for nothing. Time assumes ~400–450 GB/s effective; it is an **estimate, not measured**.

| ID | file:line | duplicate | bytes/rank/verify (wasted) | correctness-required? | kill | patched? |
|---|---|---|---|---|---|---|
| **DC1** | `llama-model.cpp:387-399` (flag), `:699-717` (MIRRORED return); OFF path `:745-775` | `TP5_HC_REPLICATED_COMPUTE=1` mirrors all 97 HC pairs (194 Q8 matrices), so all 5 ranks compute every pair. candidate.json reason: "Remove 97 HC single-owner broadcast boundaries" (181→84 channels) | ~644 MiB Q8 (**~515 MiB ≈ 1.2 ms**) | **NO.** `numeric-comparison.json`: 64 rows / 15.9M F32 logits **bitwise equal** to the static-owner reference; formal ON/OFF trajectories equal. Pure latency hide → 邪修 | Set `TP5_HC_REPLICATED_COMPUTE=0`. That gives the existing static-owner layout: owner = pair_index%5, so the 97 pairs are spread round-robin (~19–20 per rank). The owner does the full-K reduction, which keeps it **bitwise identical**. Accept +97 relay channels and run it on the async-protocol candidate (per-window courier, submit-only gate) so the peer window hides T. Not the balanced 64-wide intra-pair shard: that one is not bitwise. | flag-only, no patch needed |
| DC1-pre | `mtp-hc-replicated-compute-candidate-20261025/formal-comparison.json` | the OFF arm had 56–59 s TTFT and up to 3.19 s pauses | — | — | Root-cause the OFF cold path (bucket F, 181-channel graph build) **before** the A/B. Otherwise the kill will look like a regression for the wrong reason. | — |
| **DC5** | default MIRRORED `llama-model.cpp:901`; router-owner `qwen4exp.cpp:38-48,1491-1520` | F32 router `ffn_gate_inp [2560,512]` ×48 runs on all 5 ranks (`HC_ROUTER_OWNER=0`) | 240 MiB (**192 ≈ 0.45 ms**) | NO | Split the router by **expert axis** (103/103/102/102/102). Each logit is still a full-K dot on one rank, so it stays **bitwise**. Then all-gather 512×F32 = 2 KiB/row → topk. The existing `HC_ROUTER_OWNER=1` is T1-only and rejects non-T1 in Meta, so it must be extended to N3 first. | no (needs Meta join) |
| **DC6** | `llama-model.cpp:406-445,725-733`; `qwen4exp.cpp:51,1570` | shared expert (gate/up/down 640) mirrored, `SHARED_EXPERT_SHARD=0` | ~101 MiB (**81 ≈ 0.2 ms**) | NO | `GGML_TP5_SHARED_EXPERT_SHARD=1` already exists: 128 per rank, and the partials join at the **existing** MoE boundary, so it adds **0 new channels**. Not bitwise (partial-sum order), so it needs a KL/top1 gate. Never A/B'd in MTP. | flag-only |
| **DC7** | `llama-model.cpp:1099-1125` | Qwen4Exp has n_head_kv=2 < 5 ranks, so **all 12 full-attention layers** run MIRRORED on every rank: q/k/v/o, indexer, FA, and a mirrored KV cache | ~335 MiB (**268 ≈ 0.6 ms**) + 5× FA + 5× KV memory | NO (the comment says it only avoids zero-width KV-group slices) | Give each whole attention layer one owner, round-robin (12 layers → 2–3 per rank), plus one broadcast join, the same way as the HC static-owner. That is **bitwise** and frees 4/5 of the full-attention KV. Alternative: split by Q-head with KV replicated. | no (Meta layout + join, bigger) |
| **DC8** | `mmvq.cu:1500` (`ids == nullptr` excludes the q8 cache); N3 graph `quantize_q8_1[10x1x3]` → `mul_mat_vec_q_moe` twice per layer | In N3, gate+up GLU fusion is off (`ggml-cuda.cu:2856-2862`, ncols_dst==1 only), so gate_exps and up_exps **each re-quantize the same src1** | 48 nodes ≈ 0.13 ms floor | NO (quantize does not depend on ids) | Let unfused MUL_MAT_ID use the q8 cache | **YES → `mtp-dupcompute-candidate-20261027`** |
| DC8b | N3 graph idx 508: `quantize_q8_1[10x3x1]` before shexp | same FFN input x quantized a 3rd time in a different view (ne11/ne12 swapped, same bytes); cache key is the tensor pointer | 48 nodes | NO | Key the cache on data+contiguous rows. This needs a 2-slot cache, because the down_exps quantize evicts the slot. That changes the `common.cuh` struct, which is shared with ggml-cuda.cu.o and relay.hip.o, so it means a rebuild. Deferred. | no |

Checked and **not** duplicate compute:
- Reject rollback replay (`runtime.cpp:1604-1740`, re-decodes the accepted prefix) is **inactive**. With RS_RING the native suffix trim runs instead (`:1545-1600`, "extra_target_replay=0"). Compliant.
- Draft KV drop / recompute after verify (`runtime.cpp:1533-1541`): the inputs differ (real target hidden vs draft hidden), so this is a true dependency, not a duplicate.
- `TP5_MTP_DRAFT_KV_ONLY=1` and `LAST_DRAFT_LOGITS=1` already remove work, which is the right direction.
- Mirrored residual/HC elementwise ops (rms_norm 10240, bin_bcast) are cheap. Removing them would need a reduce-scatter/all-gather per layer, which costs more T than it saves. Keep.
- No extra verify and no "extra draft to fill bubbles" exists (DC3 from KTrader: clean).

**DC4 `TP5_MTP_SPLIT_DRAFT_DEVICES=1` (bucket B):** this is placement, not duplication. Window0's draft runs on ROCm0 and window1's on ROCm3, both verify ranks. The draft is HBM-bound: the self-contained NextN is ~1.6 GB with 450+225 MiB of experts. M1b only used ALU busy kernels, so it **does not cover bandwidth steal** from the peer's verify shard.
- History: split ON vs OFF showed no gain (24.0/33.2 vs 33.9/34.6; ready-split 39.5/51.0 vs 40.5/43.9).
- Proposed M5 (mock): replay N3 with HBM-streaming mock kernels (draft-sized reads) on dev0/dev3 during the peer's verify, and measure the peer's rank spread.
- **dev2 jitter:** dev2 is the slowest even in the empty-kernel M1 (11.26 vs 10.42 ms) and is not a draft device. So dev2's lag is its own launch floor, not draft steal. Watch it separately (PCIe/IRQ affinity).

Total duplicate (DC1+5+6+7): **~1.06 GiB/rank/verify wasted ≈ 2.3–2.6 ms of device time per rank per verify (estimate)**. Under dual window that is ~7–8% of the ~30.5 ms verify C, plus a 4/5 cut of the HC/router/attn HBM traffic on every rank. After the kills, T rises (+97 HC channels, +12 attn joins, +48 router gathers ≈ 157 joins). It has to be hidden by the peer window, which requires the async-protocol fixes (per-window courier, submit-only gate). Do not A/B DC1 on the old READY_PAIR binary.

**Isolated candidate built** at `mtp-endtoend-20261024/mtp-dupcompute-candidate-20261027/` (DC8):
- `mmvq.patch` adds `GGML_CUDA_Q8CACHE_IDS=1`, default OFF. Only unfused MUL_MAT_ID is affected; the N1 fused path is unchanged. It is host-only code: device ISA md5 79dbf136 is identical.
- Control compile matches host .text and ISA. The control relink reproduces async hip **202a55b5 bit-for-bit**.
- Candidate `lib/libggml-hip.so.0.18.1` = **336ef572**. Built, NOT run, NOT promoted. selected untouched.
- Gate (needs flock EX on `ple-candidate-gpu.lock`): 32x2 full-logit bitwise vs flag=0, plus q8_cache_hits +48 per verify.

Suggested order (micro/mock first, no 512x2):
1. DC8 bitwise gate
2. DC1 OFF on the async candidate, after DC1-pre cold fix (bitwise, flag-only)
3. DC6 flag (numeric gate)
4. DC5 expert-axis router
5. DC7 attention owner

## M2 results (2026-10-27, micro only, no model, no 512x2, GPU under flock EX `ple-candidate-gpu.lock`)

Isolated dir `mtp-blueprint-20261026/router-m2-topk-scratch-20261027/`. It contains `micro.cpp`, `micro_graph.cpp`, `compare.py`, `breakdown.py`, `out/`, `graph.jsonl`, and `prof/`. It links a sha-identical copy of `router-snapshot/install/lib` (libggml-hip c4a1634f). selected was not touched.

**Method.** The micro builds the real `build_moe_ffn` router through the public ggml API: soft_max → argsort_top_k(10) → get_rows → sum_rows → clamp(6.1035e-5) → div → reshape, no bias, w_scale=1. Both paths therefore run the **production library kernels**.
- Fused path: `topk_moe_cuda<512,false>` with separate scratch. Every tensor has its own storage, so the output cannot alias the probs.
- Chain path: `GGML_CUDA_DISABLE_FUSION=1`. The rocprofv3 kernel trace confirms 6 kernels per router: soft_max_f32<512>, k_argsort_f32_i32, k_get_rows_float, reduce_rows_f32, op_clamp, k_bin_bcast<div>. The ledger's 7th "cpy_scalar_transpose" is not part of the router fusion set, so it is not counted here.
- N1-per-row reference: each row is run alone as a single-row graph (the nrows==1 exception → topk_moe), which is exactly the production N1 path.

**Bitwise (5 devices × N=1..4 × 6 input patterns** (gauss, all-tie, heavy-tie, peaky, 2-spike+ties, near-uniform) **= 120 cases, 300 rows)**

| path | ids ≠ N1-row | weight bits ≠ N1-row |
|---|---:|---:|
| fused N3 (separate scratch) | **0** | **0 / 900** |
| fused all N1..4 | 0 | 0 / 3000 |
| current chain N3 | 0 | **376 / 900 (≤3 ulp)** |
| current chain N1..4 | 0 | 1050 / 3000 (≤3 ulp) |

→ Today's N3 chain does **not** reproduce N1 numerics: about 42% of the N3 router weights differ by 1–3 ulp from what the same row would get under N1. The expert IDs never differ, ties included. Restoring the fusion makes N3 per row **bitwise identical to N1**, which also removes an N1/N3 numerical inconsistency.

**Captured hipGraph time.** The real `ggml_backend_graph_compute_async` is captured on the backend's own stream and then replayed with hipGraphLaunch. Per router = (t(K=96) − t(K=48))/48, p50 of 21×100 replays.

| dev | N | fused µs | chain µs | ratio | graph kernel nodes per router |
|---|---|---:|---:|---:|---|
| 0 | 3 | **6.71** | 26.78 | **0.250×** | 1 vs 6 |
| 0 | 1 | 6.70 | 26.86 | 0.249× | |
| 2 | 3 | **7.23** | 29.43 | **0.246×** | 1 vs 6 |
| 2 | 1 | 6.89 | 28.78 | 0.239× | |

Gate ≤ 0.5×: **PASS, at 0.25×.** The fused N3 cost equals N1, so the 3 rows are free and the remaining cost is just the dispatch floor of about 4.5 µs per kernel node on gfx1030. Without graph capture (direct launch, the production mode, because this libggml-hip was built **without** `GGML_HIP_GRAPHS`: no warmup strings in the .so, and DISABLE_GRAPHS made no difference) the numbers are fused 7.76 vs chain 32.19 µs per router, also 0.24×.

**Per-token saving at 48 routers/graph:** (26.78 − 6.71) × 48 ≈ **0.96 ms** on dev0 and **1.06 ms** on dev2, per N3 graph per device (−240 kernel nodes). That is above M1's 0.69 ms estimate from empty-kernel replay, because the real chain bodies cost more than empty kernels. It is pure removal of duplicated dispatch; no compute is added.

**Findings for the integration patch (not written yet)**
1. In a standalone gallocr graph the N3 fusion **does** fire. The production veto comes only from the full-model allocator layout: the weights output lands 128 B inside row-0 probs, and `ggml-cuda.cu:4467/5401`. The fix is in allocation, not in the kernel. The kernel is row-independent (one warp per row reads its whole row before writing), but rows 1/2 write into row 0's region, so the alias is a real cross-warp race. Separate scratch is required; widening the `nrows==1` exception is not allowed.
2. **Do not** implement the separate scratch with `ggml_set_output()` on the weights or on the ids view. Either flag alone kills the fusion even for N1, I measured this but did not trace the cause; likely suspects are the OUTPUT/use-count checks in `ggml_can_fuse_subgraph`. Options: a dedicated router-output tensor or buffer per layer (allocator hint/no-reuse), or allocating the weights before the probs in the graph so gallocr cannot place them inside the probs.
3. Harness bug, not production: `ggml_cuda_topk_moe_fusion` reads `nodes[node_idx]` one past the final RESHAPE without a bound check and segfaults if the router is the last node in a graph. In production there is always a consumer. A guard node was added in the micro.

Next: write the allocator-side scratch patch in an isolated candidate. The gate is the N3 graph node count −240 with topk_moe×48, plus full-logit bitwise of the N3 verify rows vs N1-per-row (the router part is now proven exact).

## M4: cold-shape prewarm (2026-10-27). Ran a short diagnostic, not formal. Gate FAILED. Part of bucket F is removed.
Dir: `mtp-endtoend-20261024/mtp-cold-prewarm-m4-20261027/`. Not promoted. Selected installs untouched (libs are copies).
Every GPU run held a flock on `ple-candidate-gpu.lock`.

**Build**
- Runtime: the control rebuild reproduces the selected **1160f624 bit-exact**. The M4 binary is cead8d6b.
- `TP5_PREWARM_SHAPES=1` (default off):
  - Before the first real decode, each window runs these target shapes 3× each: N2-out, N1-out, N3-out, N1-noout, N4-noout. The 3 passes cover direct, capture and replay.
  - It also runs the draft N1 no-out and out shapes 3× each.
  - Then it calls `llama_memory_clear(data=true)` on both contexts.
- relay.hip: the frame cache goes from 2 to **8 entries with LRU** (libggml-hip 706967b5). Before this, 2 frames could not hold 5 shapes, so a revisited shape always fell back to direct dispatch plus a new capture.

**Numerics:** with prewarm on vs off, the logits file (`report.json.logits.f32le`) is **bitwise identical** and the emitted tokens are identical.

**32x2 TP5_DIAG_GRAPH (base = selected bin + libs, prewarm off; M4 = prewarm on + 8 frames)**

| | req0 intervals[0..3] ms | req1 intervals[0..3] ms | 31-interval span req0 / req1 | TTFT |
|---|---|---|---|---|
| base | 351 / 313 / 5 / **137** | 306 / 0.6 / 4.6 / **138** | 1256 / 938 ms | 2.3 / 2.6 s |
| M4 | **200 / 117** / 4.6 / 37 | **117** / 0.6 / 4.6 / 37 | **813 / 652 ms** | 27.8 / 28.0 s (prewarm 26.8 s, including about 16 s of first-ever HIP kernel load) |

- What prewarm removes: the 137 ms spike at interval 3 (mode1 capture of N3), the first-use kernel load of N3, and the direct→capture cycle at every later revisit. After the first token, N3 and N1-out now come in as relay `mode=2` (replay hit).
- **Gate result: FAIL.**
  - After the first decode, `build_us>0` still appears 3× per ctx: N1-noout at pos 4, N1-out at pos 11/12, and the first N3.
  - Interval 0 is 117–200 ms, above the 50 ms bar.

**Root cause: prewarm alone cannot satisfy the gate.** Every cache layer holds only one entry per shape, so each change of shape rebuilds on the host:
1. llama `gf_res_prev` is single-entry. Any change in N or n_outputs means build ~2 ms + `sched_alloc_graph` **~40 ms**.
2. Meta keeps a single plan keyed by `cgraph->uid`. A new uid means plan **~26 ms**. Its double-buffered `stc_compute` (KV buffer included) caps it at 2 live plans.
3. On a changed definition, the relay signature/key costs **10–16 ms** even when the frame then hits.

So each transition costs about **80 ms of host time** whether or not prewarm ran. With `READY_PAIR` on, the peer window also absorbs it (the M4 N3 transition shows submit_us 95 ms).
After the first token, the only remaining transition in the main flow is N1-out → N3 (interval 0). In 512x2 the tail adds N3 → N2/N1 transitions; by the same mechanism they should drop from ~170 to ~80 ms, but this is not measured.

**Next cut to pass the gate (pick one)**
- (a) **Shape unification (runtime only):**
  - Process the prompt in N3 chunks with all rows output. Skip emission for prompt rows; catch-up still reads each row's hidden.
  - At the tail, keep N3 and cap emission at the budget instead of shrinking to N2/N1.
  - The result is one shape for the whole request: zero transitions, and the gate passes by construction. Cost: lm_head on prompt rows, plus ≤2 wasted rows per request at the tail. Prompt numerics then follow the 3-row path that verify already uses.
- (b) **Multi-entry caches:**
  - llama: one `sched`+`gf_res` per shape, swapped in on a hit.
  - Meta: a per-uid plan cache with private `stc` containers per plan.
  - relay: a per-uid signature cache.
  - This is correct but invasive (libllama, Meta, relay).
- Also seen: the draft context has a single-entry llama cache too, and alternates N1 no-out/out several times per cycle (state-only rows vs output rows). This is a small per-cycle rebuild hidden inside bucket E. It is not instrumented (DIAG_GRAPH only prints for relay contexts).

## M3 results (2026-10-27): HC trio at N=3, real rows fusion (micro only, single GPU, no model, no 512x2)
Dir: `mtp-endtoend-20261024/mtp-hc-n3-trio-m3-20261027/` (isolated). GPU lock `ple-candidate-gpu.lock` held (LOCK_EX) for every GPU run. Selected installs untouched.
- **Provenance:** the selected libggml-hip `c4a1634f` was relinked bit-for-bit from row-epilogue objects (control). Recompiling `mmvf.cu` and `ggml-cuda.cu` from the recorded sources gives identical host `.text` and gfx1030 device `.text`. Candidate lib = control + patched `mmvf.cu` / `ggml-cuda.cu` only. The patch is behind `TP5_HC_N3_TRIO=1` (default off) and `TP5_HC_N3_TRIO_PARTS` (bitmask: 1 = down, 2 = tail, 4 = combine).
- **What the patch does.** Real N=2..4 kernels: each weight is read once per site for all rows. There is no N1×3 replay; the earlier three_T1 hack lost because it re-read the weights.
  1. `hc_down_silu_rows_q8_f32<R>` replaces quantize_q8_1 → mmvq → k_elem_chain.
  2. `hc_owner_tail4_rows_q8_f32<R>` replaces quantize_q8_1 → mmvq → hc_tail4_rows. It still writes `up` if `up` is OUTPUT, so the qwen4exp graph needs no change.
  3. Combine without the graph reorder that caused the old numeric failure:
     - The REPEAT is deferred. The caller loop treats `try_fuse == -1` as "handled".
     - MUL+ADD runs as `hc_combine4_rows`.
     - Guards: use-closure via `ggml_can_fuse_subgraph_ext{i,j,j+1}`; nothing in (i,j) writes block_out; no event nodes in the window.
- **Harness.** `m3-micro.cpp` runs the model's exact HC op sequence (rms_norm·w → down/scale/silu → up/sigmoid/mul/4 views/3 adds/scale; inject bf16 → sigmoid/scale → repeat/mul/add) through the real ggml-cuda backend, with the selected env (FUSE_HC_DOWN_SILU, OWNER_TAIL, KEEP_LO, ROW_EPILOGUE). It is captured as a hipGraph on the backend stream and timed with events, p50 of 200 replays.
  - Timing pattern follows the fixed-slot observer (captured replay, ABAB order).
  - Not used: the `tp5_body_binding` per-wave timer itself. It only exists for `mul_mat_vec_q_moe`.
- **Graph size per site:** 12 kernels → **6** (= N1). For 96 sites: 1152 → 576.

| N=3, 96 sites, dev0 | p50 ms | Δ vs off | numerics |
|---|---:|---:|---|
| off (current generic chain) | 7.20–7.29 | — | — |
| trio, rows kernels (exact) | 6.57–6.58 | **−0.65** | lo/mixed **bitwise = N1 t1 per row**; combine bitwise = generic |
| trio, down blocked RB=4 + tail rows (best) | **5.65** | **−1.6 (−22%)** | down ≤1 ulp vs N1 (max 1.2e-7); tail/combine exact |
| trio, RB=4 down + tail J=2 | 5.77 | −1.46 | same as above |
| N1 reference, same 96 sites | 3.83 | — | — |

- Per part, alone (exact kernels): down −0.59 ms, combine −0.67 ms, tail rows +0.52 ms alone. All three together: −0.71 ms. Not additive: the rows tail body (18.8 µs) is slower than mmvq 14.3 + quantize + tail.
- rocprofv3 kernel bodies per site:
  - generic: mmvq 14.3 µs ×2 + quantize 1.3 ×2 + elem 1.4 + tail 1.5 + bin_bcast 1.3 ×3, with ~3.9 µs between kernels.
  - rows: down 17.8, tail 18.8, combine 1.2.
  - t1 (N1): 10.2 / 10.2 / 1.0.
- **Numerics vs the current N3:** the generic N3 down/up quantize activations to q8_1, so N3-off already differs from N1 per row (lo relative L2 3.9e-3, mixed 3.0e-4). The trio uses the N1 F32-activation math. That makes N3 rows equal N1 (exact config), which is more accurate but **not bitwise equal to the current N3 logits**. After 96 chained sites, x differs from off by relative L2 1.9e-4. Matching the q8_1 chain bitwise would need quantize fused into the producer (cut 2c), not duplicated per block.
- **Remaining N3−N1 gap per site** (59 vs 40 µs): the rows bodies are ~1.7× the t1 bodies (L2-bound activation re-reads, partly fixed by RB=4), plus the inject bf16 generic mmv_f (7.3 vs 3.9 µs, cut 2e).
- **Expected model effect:** every rank runs all 96 HC sites in the N3 verify. Estimate −0.65 ms (exact) to −1.6 ms (RB=4) per verify on the critical path, about +1.5–4% tok/s (≈57 → 58–59). This is the micro body + dispatch only; M1 predicted −0.87 ms for dispatch alone.
- NOT run in the model, NOT promoted. Next gate: short N3 diagnostic (8x2/32x2, isolated) with `TP5_HC_N3_TRIO=1 M3_DOWN_RB=4`. Check node census 3825 → ~3250 and acceptance/logit drift against the replicated32x2 reference.

## M3 short model diagnostic (2026-10-27, 32x2 + 8x2, isolated, GPU flock EX, NOT promoted)
Dir: `mtp-hc-n3-trio-m3-20261027/diag/`. Same layout as M4's `rundiag`: the selected bin (1160f624) plus copies of the selected libs. Only `libggml-hip` is swapped, to the M3 cand `6a19849d`. Selected installs were not touched. No formal 512x2.
Scripts: `m3diag.py`, `m3plain.py`, `m3gpu{,2,3,4}.sh`, `cmplogits.py`, `census.py`. Compare outputs were computed on the box from copies of the logits.

**Reference.** `replicated32x2` = base32x2 = native32x2. All three are bitwise identical (this is the selected config).

**Determinism / no-op checks**
- M3 lib with trio OFF (`m3off32x2`, `m3off_rep32x2`) is bitwise equal to replicated32x2. The OFF path is a true no-op.
- Trio ON is reproducible run to run, bitwise: exact == exact_rep, and RB4 == RB4_rep. **No race was found.**
- Window 1 vs window 0: RB4 vs exact is bitwise for req0 (all 32 rows) but differs from req1 row 1 onward. rocprof shows `hc_down_silu_rows_blk<3,4>` running on both windows' queues, on all 5 agents. Both variants are deterministic, so this is ulp drift and not a race. The likely mechanism: the generic tail re-quantizes `lo` to q8_1, which absorbs 1-ulp changes in `lo` except at rare rounding ties.

**Numerics (soft gate: KL/top1). PASS: no coarse error.**

| cand vs ref (replicated32x2) | nonfinite | req0 top1 | req0 KL max / mean | req1 top1 | req1 KL max / mean | rel L2 max | first token divergence |
|---|---|---|---|---|---|---|---|
| exact (TRIO=1) | 0 | 32/32 | 0.0115 / 0.0020 | 32/32 | 0.0055 / 0.0011 | 0.16 | none (64/64 tokens equal) |
| RB4 (TRIO=1 M3_DOWN_RB=4) | 0 | 32/32 | 0.0115 / 0.0020 | 26/27 | 0.0065 / 0.0012 | 0.16 | req1 @26 (ref top1−top2 margin 0.044) |

- Acceptance is unchanged: 18/26 draft tokens accepted in both windows for off, exact and RB4. verified_rounds are 13/13.
- **Envelope check.** I added a plain N1 no-MTP run with the selected libs (`plainN1_32x2`). Its distance to the replicated ref is KL max 0.008 / 0.067 and rel L2 0.09 / 0.13. Trio's distance to plain is KL max 0.005 / 0.054. So the trio's drift is inside the natural N1-vs-N3 envelope; it is not a new error.
- Bitwise vs ref is 0 rows. That is expected: the trio runs N1 F32-activation math, while the current N3 path uses q8_1 activations. The N4 prompt row 0 also changes, because the trio covers R=2..4.

**Node census** (rocprofv3 kernel trace, 8x2, same decode schedule both runs: N1×17, N2×2, N3×5, N4×2; kernels per agent)
- off 78574 → RB4 75122: **−3452 per agent**, which is −3.96 kernels per HC site (871 sites in multi-row graphs). Identical on all 5 agents.
  - down part: mmvq −878, quantize −878, elem_chain −871, `hc_down_silu_rows` +871.
  - combine part: bin_bcast −2544, `hc_combine4_rows` +848.
- **The tail part is a silent no-op in the model:** `hc_owner_tail4_rows` fired at only 7 of 871 sites.
  - `taildiag8x2` (debug lib `diag/build/libggml-hip-taildiag.so`, env `M3_TAIL_DIAG=1`) shows the reject at the final overlap/hazard guard (`up` is OUTPUT, flags=18).
  - Only the first 400 rejects were logged, and all were rows=4 (prompt). The N3 sites were not logged, but the census shows they also fall back.
  - The guard is correct, not a bug: the alias it detects would be a real read/write race. The generic `hc_tail4_rows` path keeps running, so results stay correct. This is the same allocator-alias class as M2.
- Net effect: the N3 graph goes **~3825 → ~3440 per rank** (−385 per multi-row graph). The target was ~3250 (−575); the missing −2 per site is the tail. Fixing it needs allocator-side storage for `hc_mixed` vs `lo`/`xn`/`up`, not a weaker guard.

**Timing** (short diagnostic, noise level, not a claim): sum of intervals[4:] per request in ms:
- off 434/469 and 451/488
- exact 434/469 and 434/467
- RB4 430/463 and 426/462

**Verdict:** soft gate PASS. No NaN, no shape errors, no race, the OFF path is bitwise, and acceptance is unchanged. The down and combine parts are live. The tail part does not fire in the model (guard-correct fallback). **Not promoted.** Next: an allocator fix for the tail alias (like M2 scratch); then re-census, target −575.
