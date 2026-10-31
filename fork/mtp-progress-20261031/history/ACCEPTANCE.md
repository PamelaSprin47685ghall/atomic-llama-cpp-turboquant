# TP5 MTP Acceptance Framework & Decision Tree
**Lock ID:** `ple-candidate-gpu.lock`  
**Target:** Qwen3.8-Flash-Next MTP (TP5, 5× AMD Radeon RX 6800 / gfx1030, depth-2 $N=3$ verify)  
**Authoritative Standards:** Ashley Standing Rules (13022 Sync, 2026-10-27)  
**Primary Evidence Ledger:** `src/mtp-n3-waste-ledger.md`

---

## 1. Authoritative Rules & Operational Principles

* **Rule A (Soft KL/Top-1 Gate):** Small numeric, order, or rounding differences are **ACCEPTED** (`src/mtp-n3-waste-ledger.md:3`).
* **Rule B (Coarse Error Rejection):** Bugs, wrong implementations, coarse/gross errors (NaN/Inf, wrong shapes, silent no-op fusions, broken reductions, top-1 collapse from bad code, data races) are **REJECTED fail-closed** (`src/mtp-n3-waste-ledger.md:3`).
* **Rule C (Intentional Degrade Bar):** Intentional numeric shortcuts that deliberately worsen accuracy to buy speed (e.g., postdown Q8 truncation, top-8 capacity drop) are **REJECTED**. Degrades are not tradeoffs (`src/mtp-n3-waste-ledger.md:52`).
* **Rule D (Natural Tradeoffs):** Natural speed/accuracy tradeoffs (multi-row kernels reading weights once, accumulator reassociation, TP partial-sum ordering) are **ACCEPTED** if within the soft bar (`src/mtp-n3-waste-ledger.md:240-245`).
* **Rule E (Historically Slower Approx Bar):** Historically slower approximations are **REJECTED** regardless of claims: MMVQ rows2, MoE top8, HC Q8 shortcuts (`src/mtp-n3-waste-ledger.md:52,92`).
* **Rule F (Microbench / Mock First):** All candidates must prove isolated speedup and correctness in microbenchmarks or synthetic mocks before any multi-kernel integration; **strictly no formal 512×2 runs until absolute confidence** (`src/mtp-n3-waste-ledger.md:3,64`).
* **Rule G (Environment & Safety Guardrails):** Selected installs are immutable; host 12022 is code-write only; the `nrows == 1` fusion exception must never be widened to paper over memory races (`src/mtp-n3-waste-ledger.md:3,90,261`).

---

## 2. Pipeline Architecture: Gates G1–G4

Execution proceeds sequentially. Every gate operates **fail-closed**: `PASS` advances to the next gate; `FAIL` halts immediately. **No GPU runs occur until G1 and G2 pass.**

```
[Candidate Patch]
       │
       ▼
┌──────────────┐
│  G1: Static  │ ── FAIL (Syntax error, EACCES, synthetic diff, race) ──► [HALT: REJECT]
└──────┬───────┘
       │ PASS
       ▼
┌──────────────┐
│  G2: Build   │ ── FAIL (Relink failure, ISA drift, touch selected)  ──► [HALT: REJECT]
└──────┬───────┘
       │ PASS (Acquire `flock -x ple-candidate-gpu.lock`)
       ▼
┌──────────────┐
│  G3: Micro   │ ── FAIL (Silent no-op, body slower, interval >50ms)  ──► [HALT: REJECT]
└──────┬───────┘
       │ PASS
       ▼
┌──────────────┐
│ G4: Numerics │ ── FAIL (Rule B bug, Rule C degrade, Rule E veto, KL) ──► [HALT: REJECT]
└──────┬───────┘
       │ PASS
       ▼
[Promotion / Integration Ready]
```

### Stage G1: Static Verification Gate
* **Requirements:** Patch applies cleanly without conflict; no `EACCES` permission barriers; no synthetic or fabricated diffs.
* **Concurrency Audit:** Verify no memory aliasing or cross-warp write hazards (e.g., M2 weights-in-probs race at `src/mtp-n3-waste-ledger.md:27,261`).
* **Host Guard:** Host 12022 is code-write only. Synthetic reconstructions are forbidden and purged.

### Stage G2: Build & Relink Gate
* **Requirements:** Clean compilation and link of candidate `libggml-hip.so.0.18.1` against baseline control ref `c4a1634f`.
* **ISA Audit:** Host-only changes must verify identical device ISA checksums (e.g., DC8 control match `79dbf136`; `src/mtp-n3-waste-ledger.md:215-216`).
* **Install Isolation:** Selected production installs (`q2-unpack-joint-l2-a-install`, `single-sync-compose-install`) must remain strictly untouched (`src/mtp-n3-waste-ledger.md:90`).

### Stage G3: Microbenchmark / Mock Performance Gate
* **Lock Prerequisite:** Exclusive GPU access via `flock -x ple-candidate-gpu.lock` (`src/mtp-n3-waste-ledger.md:218,269`).
* **Execution:** Replay synthetic captured graphs (M1), microbench individual kernel CTAs (M2/M3), or run shape diagnostics (M4).
* **Speedup Floor:** Kernel body timing must prove real speedup, not merely lower node counts. Baseline p50 of 21×100 replays:
  * Router fusion: `topk_moe_cuda<512>` must achieve $\le 0.50\times$ chain latency (target: 6.71 µs vs 26.78 µs chain on dev0, $0.250\times$; `src/mtp-n3-waste-ledger.md:251`).
  * Shape transition interval 0: must execute within $<50\text{ ms}$ (`src/mtp-n3-waste-ledger.md:291`).
* **No-Op Guard:** Catch silent no-ops where node reduction does not decrease runtime (e.g., `m3-tail-noop-reject` where multi-row tail was 18.8 µs slower than generic mmvq+quantize+tail).

### Stage G4: Numerical Acceptance Gate
Full numerical triage tree applied to candidate outputs before end-to-end integration.

---

## 3. Numerical Gate Triage Decision Tree (G4)

```
[Candidate Output Logits]
  │
  ├─► [1. Rule B Check: Gross Defect / Bug?]
  │     ├── Contains NaN or Inf?
  │     ├── Wrong tensor shapes, strides, or batch padding?
  │     ├── Silent no-op fusion (nodes unchanged or hit counter +0)?
  │     ├── Broken reduction or top-1 collapse (variance σ <= 1e-4)?
  │     ├── Concurrency hazard (aliased scratch, cross-warp race)?
  │     └── Out-of-scope graph walk (escaped layer bounds)?
  │           └─► YES ──► REJECT (fail-closed, Rule B; abort immediately)
  │
  ├─► [2. Rule C Check: Intentional Numeric Degrade?]
  │     ├── Deliberate precision drop (e.g. postdown Q8 truncation)?
  │     └── Capacity truncation (e.g. top-10 to top-8 experts)?
  │           └─► YES ──► REJECT (Rule C; intentional degrade != tradeoff)
  │
  ├─► [3. Rule E Check: Historically Slower Approximation?]
  │     ├── MMVQ rows2, MoE top8, or HC Q8 shortcuts?
  │     └── GU-tile microkernel or hc-combine-order reordering?
  │           └─► YES ──► REJECT (Rule E / ledger:92 do-not mandate)
  │
  └─► [4. Soft Bar Qualification (Rules A & D)]
        │
        ├── Check Reference Baseline:
        │     └── Evaluated against N1-per-row ground truth (NOT generic N3 chain)
        │
        ├── Classify Optimization:
        │     ├─► Class "exact":
        │     │     └── Bitwise identity required: 0/N differing bits vs N1
        │     │
        │     ├─► Class "numeric-order-only" (natural reassociation / multi-row):
        │     │     ├── Mean forward KL < 1.0e-6 vs reference
        │     │     ├── Max per-row KL < 1.0e-5
        │     │     ├── Token Top-1 agreement >= 99.9%
        │     │     └── Zero top-1 flips outside near-tie margin (top1/top2 delta <= 0.10)
        │     │
        │     └─► Class "behavior-changing" (whole-request shape unification):
        │           ├── Pad / cap trim logic proven bitwise exact separately
        │           ├── Mean row KL in range 1.0e-4 .. 1.0e-3
        │           ├── Max row KL <= 0.021
        │           └── Top-1 flips justified by tie-margin analysis (margin <= 0.10)
        │
        └── Threshold Passed?
              ├── YES ──► PASS (Accept candidate under Soft Gate)
              └── NO  ──► REJECT (Halt fail-closed)
```

---

## 4. Soft Bar Acceptance Thresholds

| Classification | Scope / Examples | Bitwise Identity | Mean Forward KL | Max Row KL | Top-1 Agreement | Boundary & Margin Condition |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`exact`** | DC1 static owner (`ledger:192`), M2 fused router weights (`ledger:240`), M3 lo/mixed exact (`ledger:13`) | **Required** (0 bit diffs) | 0.0 | 0.0 | 100.0% | Bitwise parity against single-token $N=1$ per-row reference. |
| **`numeric-order-only`** | M3 HC trio RB=4 (`ledger:13`), DC6 shared-expert shard (`ledger:195`), DC8 Q8 cache (`ledger:197`), Q2 down t3 (`ledger:29`), BF16 load4 (`ledger:28`) | Not required ($\le 1\text{ ULP}$ acceptable) | $< 1.0 \times 10^{-6}$ | $< 1.0 \times 10^{-5}$ | $\ge 99.9\%$ | Zero token flips except where baseline top-1 / top-2 margin is $\le 0.10$; zero NaNs; logit $\sigma > 1.0 \times 10^{-4}$. |
| **`behavior-changing`** | M4a fixed-$N=3$ shape unification (`ledger:302-305`), E-bucket draft pipelining (`ledger:61`) | Separate exactness on trim | $\sim 10^{-4} - 10^{-3}$ | $\le 0.021$ | Matches control (except near-ties) | Pad/cap logic verified bitwise exact (16/16 rows); flips confined to tight margin ($\le 0.10$); post-token build stall $= 0\text{ µs}$. |

---

## 5. Grounded Casebook: Precedents & Empirical Evidence

### PASS Precedents
1. **M2 Router Fusion (Micro PASS, `src/mtp-n3-waste-ledger.md:227-266`):**
   * *Evidence:* Dev0 fused `topk_moe_cuda<512>` achieved **6.71 µs vs 26.78 µs** chain ($0.250\times$, p50 of 21×100 replays; `ledger:251`), removing 240 kernel nodes across 48 layers for **−0.96 ms** device time (`ledger:258`).
   * *Numerics:* Restores $N=1$ bitwise parity (0/900 weight bits, 0/900 ID diffs; `ledger:240`). Unfused $N=3$ chain had drifted by $\le 3\text{ ULPs}$ on 376/900 weights (~42%; `ledger:242`).
   * *Safety Guard:* Allocator alias into row-0 probs (`ledger:27`) resolved via pinned logits scratch (`ggml-alloc.c:631/645/692`), rejecting `ggml_set_output` on probs (`ggml.c:7825`). Layer-scope assertion (`rc=1` on backward walk) caught cross-layer bug before promotion.
2. **M3 HC Trio (Micro PASS, `src/mtp-n3-waste-ledger.md:13,74,119`):**
   * *Evidence:* Native multi-row kernels (`hc_down_silu_rows_q8_f32`, `hc_owner_tail4_rows_q8_f32`, `hc_combine4_rows`) reduced 12 kernels to 6 per site (1152 $\to$ 576 nodes; `ledger:13`).
   * *Numerics:* Exact mode: bitwise identical to $N=1$ per-row reference (−0.65 ms). Fastest `RB=4` mode: down-projection within $\le 1\text{ ULP}$ ($1.2 \times 10^{-7}$ max drift), tail/combine exact (−1.6 ms, −22%). Natural read-once weight tradeoff accepted under Rule D. Combine preserves graph order, preventing the historical `hc-combine-order` failure (`ledger:92`).
3. **M4a Fixed-$N=3$ Shape Unification (Diagnostic PASS, `src/mtp-n3-waste-ledger.md:302-305`):**
   * *Evidence:* Bypasses single-entry host cache rebuilds (~80 ms stall; `ledger:294-298`). Post-first-token `build_us = 0`.
   * *Numerics:* Not bitwise vs control (prompt KV/GDN incorporates $N=3$ multi-row drift). Mean KL $\sim 10^{-4} - 10^{-3}$, max KL 0.021; 1 top-1 flip at token 25 where control margin was 0.10. Suffix trim/cap logic bitwise exact (16/16 rows). Accepted under Rule A/D.

### FAIL / REJECT Precedents
1. **M4 Cold-Shape Prewarm (FAIL Gate, `src/mtp-n3-waste-ledger.md:267-300`):**
   * *Evidence:* Numerics bitwise clean, but `build_us > 0` recurred 3× per context; interval 0 remained 117–200 ms (violating $<50\text{ ms}$ bar; `ledger:291`). Host single-entry caches (`gf_res_prev` ~40 ms, Meta plan ~26 ms, relay key 10–16 ms; `ledger:294-296`) cannot be bypassed by prewarm. Halted fail-closed.
2. **DC8 Q8 Cache for IDs (BLOCKED / BUILT-NOT-RUN, `src/mtp-n3-waste-ledger.md:214-218`):**
   * *Evidence:* Candidate built (`libggml-hip` sha `336ef572`, ISA `79dbf136`; `ledger:216-217`) but unrun. **Rule: BUILT, NOT RUN $\ne$ PASS.** Bars defined (top-1 $\ge 99.9\%$, mean KL $< 10^{-6}$, hits $\ge 48$; `ledger:218`), but unmeasured claims remain `[UNPROVEN]`. Synthetic diff reconstruction strictly forbidden.
3. **Intentional Degrades & Slower Approximations (REJECT Precedents, `src/mtp-n3-waste-ledger.md:52,92`):**
   * *Postdown Q8 / Top-8 / HC Q8 Shortcuts:* Intentional truncation is an accuracy degrade, not a tradeoff (Rule C).
   * *MMVQ rows2:* Historically slower; barred from revival under Rule E.
   * *`hc-combine-order` / GU-Tile Microkernels:* Explicitly barred under ledger §6 (`ledger:92`) due to gross numerical divergence.
4. **`m3-tail-noop-reject` (Silent No-Op REJECT):**
   * *Evidence:* Multi-row tail kernel body was 18.8 µs slower than generic chain (14.3 µs MMVQ + quantize + tail). Microbench body timing must catch regressions that pass node-count checks.

---

## 6. Hard Blockers & Inviolable Guardrails

1. **Selected Installs Untouched:** Production directories `q2-unpack-joint-l2-a-install` and `single-sync-compose-install` must never be modified, overwritten, or relinked (`src/mtp-n3-waste-ledger.md:90`).
2. **No Widening of `nrows == 1`:** The `nrows == 1` exception in `ggml-cuda.cu:4467` must never be widened to $N > 1$ without dedicated allocator scratch. Widening without scratch reintroduces cross-warp memory races (`src/mtp-n3-waste-ledger.md:27,261`).
3. **No Premature 512×2 Runs:** Multi-sequence full benchmarks ($512\times 2$) are prohibited until isolated microbenchmarks and diagnostics provide absolute confidence (`src/mtp-n3-waste-ledger.md:3,64`).
4. **Exclusive GPU Lock:** Every GPU execution must be serialized under `flock -x ple-candidate-gpu.lock` (`src/mtp-n3-waste-ledger.md:218,269`).
5. **Host 12022 Static Scope:** Host 12022 is restricted to code-write only; no GPU executions or direct installation modifications are permitted.
6. **Built $\ne$ Pass:** A compiled artifact without completed GPU gate evaluation is unverified and cannot be promoted.

---

## 7. Reporting & Audit Duties

* **Line-Grounded Citations:** Every assertion, result, and gate decision must cite concrete repository evidence using `file:line` format (e.g., `src/mtp-n3-waste-ledger.md:251`).
* **Tagging Unverified Claims:** Any projection, unbenchmarked kernel, or unverified output must be explicitly marked `[UNPROVEN]`.
* **Fail-Closed Assertions:** Guards must fail with non-zero exit codes (`rc != 0`) and clear diagnostic messages (e.g., M2 graph reserve guard aborting with `rc=1` on multi-row logits count). Guard aborts count as successful defense, not framework defects.
