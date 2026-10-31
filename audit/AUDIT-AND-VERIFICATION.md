# M3 N-row owner-tail storage: audit + Devops verification plan (12022, static only)

Status: **PARTIAL**. Patch authored and audited; NOT compiled, NOT tested (contract: only Devops may build/run).
Nothing in `$E` was modified; selected installs untouched; no GPU touched.

Worktree: `/home/kunweiz/work/m3tail-storage-20261031`, branch `candidates/m3tail-storage-20261031`
Base: `f1c5fafc0` (= 13022 snapshot of clean/fork-20261018 `829d05f9e`, the base the M3 trio waves ran on).
Patch: `audit/m3-tail-storage.patch.txt` (`git diff f1c5fafc0`); commits `2ef9cf2e3`, `1d529d64a`,
`2ae2c7d9d`; this audit file (and the patch copy) sits in the audit commit on top and is not part of the code diff.

Operational note for anyone cloning the trio lane: the **shipping** trio binary
(`libggml-hip` `11800fe6…`, source `47abaa3b…`) reads `GGML_TP5_HC_N3_TRIO` /
`GGML_TP5_HC_N3_DOWN_RB` / `GGML_TP5_HC_N3_TAIL_J` (lane `patch/patch.diff`), whereas the
2026-10-27 diag/census binary (`ggml-cuda.tailwhy.cu` `22e88007…`) still reads the old
`TP5_HC_N3_TRIO` without the `GGML_` prefix. Use the names of the binary under test; the new
llama gate uses the `GGML_`-prefixed `GGML_TP5_HC_N3_TRIO`, so a leg that silently sets only the
old spelling would take the OFF path and invalidate the reach measurement.

## 1. The allocation chain that defeats the existing tail guard

Chain (per HC site, `qwen4exp.cpp build_hc_mix`, nt = x->ne[2]):

```
lo   = silu(scale(build_lora_mm(w_down, xn), 1/4))     # [320, nt]  F32
gate = sigmoid(build_lora_mm(w_up, lo))                # last "ordinary" consumer of lo
gated = mul(xn, gate) -> reshape [n_embd,4,nt]
streams/ADD/ADD/ADD/SCALE -> mixed / hc_mixed          # [n_embd, nt] F32
```

`ggml-alloc.c`:
* after `ggml_cuda`'s UP node, `lo`'s `n_children`/`n_views` reach 0 while it is still OWN →
  `ggml_gallocr_free_node(lo)` → block goes to `ggml_dyn_tallocr_free_bytes`;
* `ggml_dyn_tallocr_free_bytes` merges contiguous neighbours (first/next/prev), so the `lo`
  block coalesces with the neighbouring up/gated blocks into one large free block;
* the `hc_mixed` chain operates per row on `nt` rows of 10 KiB: `ADD` (ADD is `can_inplace`) reuses
  the first operand in place, so the 4-row `gated` block is *already* the destination, and the
  stream views alias it (`view_src == gated`, `has_owner_tail_hazard` counts 5);
* `SCALE` (also `can_inplace`) is not in place: `scale_buffer` has no kernel for F32 dst ≠ src,
  so it allocates a fresh dst from the merged free block — i.e. exactly inside the `lo` range;
* `ggml-cuda.cu:5649` `ggml_cuda_fuse_hc_owner_tail4_rows` — a **correct** fail-closed guard —
  then sees `overlaps(out=hc_mixed, lo)` (`out->data` ∈ `[lo, lo+ggml_nbytes(lo))`) and the trio
  silently falls back to the generic 12-node chain. `nrows==1` is irrelevant: the t1 fusion has its
  own guard instance (`ggml_cuda.cu:5215`), which is why N=1 fires and N=2..4 do not.

Evidence chain (all pre-existing, re-read here):
| claim | evidence |
|---|---|
| tail fires 0/3870 with trio, 3840/3870 reject `why=1` (out↔lo) | `$E/mtp-m3-tail-alloc-20261027/whyctl8x2/stderr.log` (`M3_TAIL_WHY why=1 rows=2/3/4` = 960/1920/960, `why=0` = 30) |
| same config + `TP5_HC_N3_TAIL_KEEP=1` → 3870/3870 fire | `…/why8x2/stderr.log` (`why=0` 3870, `reject` 40 = a foreign Q8_0 MUL_MAT shape, line 5612) |
| those two legs ran the **diag** DSO, not the shipping trio DSO | each leg's `install/lib/libggml-hip.so.0.18.1` = `cf14e803…` (`libggml-hip-tailwhy.so`); only the **llama** side differed (`92e48ec2…` pin vs nothing) — see §2a |
| pin is lifetime-only, OFF bitwise, determinism 4/4 | `$E/mtp-m3-tail-alias-20261028/GATE.md`, `SUMMARY.md` |
| guard lines | `$E/mtp-m3-trio-20261028/tree/ggml/src/ggml-cuda/ggml-cuda.cu:5649` (`out,lo` / `out,xn` / `out,weights` / up aliases / `has_owner_tail_hazard`) |
| allocator rules | worktree `ggml/src/ggml-alloc.c:642-697` (retained parent not reused in place, retained node never freed), `:307-345` (free/merge), `:22-50` (`can_inplace`), `ggml/src/ggml-cuda/scale.cu` F32 dst≠src → no kernel |

So the *mechanism* was real; what was missing was (a) an independent-storage primitive that does
not change fusion or Meta decisions, and (b) coverage of the mixer sites, see §3.

## 2. The fix

`GGML_TENSOR_FLAG_NO_REUSE` (new, = 32) = "allocator retains this non-view tensor and forbids
reuse through its views" (`ggml/include/ggml.h`), with the allocator contract taken verbatim from
the preserved router-scratch candidate `4496493dd` so both branches carry one copy of the code:
`ggml-alloc.c` skips retained parents for in-place reuse (also via `view_src`), never frees a
retained node, and records `no_reuse` per graph node/src so `ggml_gallocr_needs_realloc` re-plans
when the flag differs.

`build_hc_mix` sets the flag, default off, on the guard-identical Q8_0 shape only:

```
GGML_TP5_HC_N3_TAIL_KEEP == "1"  AND  GGML_TP5_HC_N3_TRIO == "1"   (strict, byte-exact "1")
hc == 4 && n_embd == 2560 && 2 <= nt <= 4
w_down/w_up: Q8_0, op==NONE, no view, (hc_dim,320)/(320,hc_dim)
lo: SILU unary, F32 [320,nt,1,1], no view, contiguous            -> lo->flags |= NO_REUSE
if inject == nullptr (head mixer, MTP head mixer)                -> xn storage owner |= NO_REUSE
```

Why this is safe (static argument):
1. **No value change.** No node, copy, kernel or op_params is added; the fused kernels write
   `hc_mixed` and read `lo`/`xn`/weights, and per-row arithmetic is the t1 arithmetic
   (`numeric-check.json`: trio vs N1 `lo`/`mixed` bitdiff 0 over 960/7680 values).
2. **Storage disjointness is now structural.** A retained block is never freed, so no later
   allocation can land in it, and a retained parent is never reused in place, so the tail
   destination is allocated from a block disjoint from `lo` (and from `xn` where retained).
   The backend guard is *not* relaxed: residual hazards (`up` aliases, event nodes, non-retained
   `xn` at inject sites) still veto and fall back.
3. **Fusion independence.** `ggml_can_fuse_subgraph_ext` keys on `COMPUTE`/`OUTPUT` only
   (`ggml.c:7817-7827`); `lo`/`xn` stay external inputs. No `OUTPUT` flag is added, so
   `hc_has_external_alias`, `has_owner_tail_hazard` (`view_counts` only) and Meta's HC owner /
   late-join plan (`ggml-backend-meta.cpp:2617,3150` key on `OUTPUT`) keep their previous decisions.
4. **Shape/type guards mirror the HIP guard** so the flag is only set where fusion is even possible.
5. **OFF ≡ trio alone ≡ stock for graphs without the trio.**
6. **N=1 untouched.** `nt == 1` never enters the new branch; the qualified T1 path keeps
   `GGML_TP5_HC_KEEP_LO` (an `OUTPUT` pin, default-off-safe, unchanged). `GGML_TP5_HC_KEEP_LO=1`
   in the qualified row-tail config therefore cannot interact: it cannot fire for nt ≥ 2.
7. **Retention cost** 320·nt·4 B per site (96 sites): 240/360/480 KiB for nt = 2/3/4; plus, at the
   two mixers, one `[4·n_embd, nt]` F32 row batch (4·2560·4·4 B = 160 KiB for nt=4) — the same
   order as the M2 router pin (+0.32 MiB) and the T1 `KEEP_LO` pin.

## 2a. Which DSO is the real M3 fusion reference (receipt cross-check)

`mtp-m3-trio-20261028/build-receipt.json` mixes three different `libggml-hip` hashes; only one of
them is the shipping trio DSO the model legs and Devops' 8x2 ran:

| key | path | what it is | role |
|---|---|---|---|
| `lib-m3t` `11800fe6…` | `mtp-m3-trio-20261028/build/libggml-hip.so.0.18.1` = `…/build/libggml-hip.repro.so.0.18.1` = `…/install/lib/libggml-hip.so.0.18.1` (all three byte-identical) | cleaned trio, sources `47abaa3b…` (ggml-cuda.cu) + `4cf2acc4…` (mmvf.cu), objects `2b52b9ac…`/`16ed7a02…` | **the shipping/perf reference**: micro 96-site −0.66 ms, `m3t_on32x2` and the Devops 8x2 leg |
| `lib-m3-measured` `6a19849d…` | `mtp-hc-n3-trio-m3-20261027/install/lib/libggml-hip.so.0.18.1` | earlier trio build with the old `TP5_HC_N3_*` env names (same origin, +unused helper) | equality anchor only: `m3t_on32x2` is whole-logits **bitwise** equal to `m3exact32x2` on this lib; never the perf leg |
| `lib-selected` `c4a1634f…` | `mtp-hc-row-tail-candidate-20261025/install/lib/libggml-hip.so.0.18.1` = `mtp-hc-n3-trio-m3-20261027/ctl-lib/libggml-hip.so.*` | stock relink control (no trio code) | OFF/control: G2 relink target, `m3off*`/`replay` legs |
| — | `mtp-m3-tail-alloc-20261027/build/libggml-hip-tailwhy.so` `cf14e803…` | trio + log-only `m3_tail_why`/`m3_tail_reject` diag (source `ggml-cuda.tailwhy.cu` `22e88007…`), old `TP5_HC_N3_TRIO` spelling; kernels otherwise = trio | **the only DSO that can prove reach** (the shipping DSO has no diag output) |
| — | `mtp-m3-tail-alias-20261028/build/libggml-hip-diag.so` `aeeb21e3…` | trio + alias-lane `diag-only.ggml-cuda.diff` hook, cleaned `GGML_TP5_HC_N3_TRIO` gate; installed in that lane's `runs/*/install/lib` | reach for the **OUTPUT-pin** fix variant |

Consequences for the Devops 8x2 result: two legs rc0 + 16 tokens equal on `11800fe6` proves the
gate is a no-op/soft-drift clean model run, but **cannot** distinguish "tail fused" from "tail fell
back to the generic 12-node chain" — the shipping DSO has no counters and both paths produce the
same tokens inside the soft gate. Reach must come from a diag DSO (§5 G3-1), and only a diag build
that keeps the trio kernels identical to `11800fe6` may be used for that claim (diff its
`ggml-cuda.cu`/`mmvf.cu` objects against trio objects, or its kernel bodies against `47abaa3b…`).

## 3. Known residual risk (must be read, not hidden)

* The 2026-10-27 `why8x2` run (trio + old `OUTPUT`-on-`lo` pin) fired 30 times at the **no-inject
  head mixer** (`out=result_norm`, `n_nodes=39`, rows=3 x10) and 30 times at the out-embedding
  mixer (`out=result_output`, rows=2 x10, rows=3 x20) — i.e. the old pin was sufficient there in
  that run, even though `xn` was unpinned (`xn=hc_norm`). Retaining `lo` also moves that site's
  arena layout, so the destination could land in `xn`'s freed block instead. Hence `xn` (its
  storage owner) is retained at exactly those sites (a view cannot be retained;
  galloc frees a view owner when its last view loses its last consumer, cf. `:811-823`).
* Retaining `xn`'s owner pins the model-wide `hc_norm` row batch, which later layers reuse as their
  input-embedding copy (layer 0 writes it, layers 1.. keep reusing that storage): the memory delta
  is bounded (≈ +0.16 MiB) but the plan is *not* the same shape as the old pin's.
* If the guard still vetoes at any site, behaviour is the **generic fallback** (correct, slower),
  not a wrong result: the guard is fail-closed and unmodified.

## 4. Merging note

`ggml/include/ggml.h` (`NO_REUSE`), `ggml/src/ggml-alloc.c` are byte-identical to
`candidates/router-scratch-20261031` (`4496493dd`); `git diff candidates/router-scratch-20261031
candidates/m3tail-storage-20261031 -- ggml/` must be empty. Merge M2 first, or merge both with the
`ggml/` hunks taken once. The `qwen4exp.cpp` hunks are disjoint from M2's (`build_layer_ffn`).

## 5. Devops verification plan (13022, GPU flock EX; 12022 = compile/static only)

Pinned inputs (verify by sha256 before use):

| artifact | sha256 |
|---|---|
| trio HIP source `mtp-m3-trio-20261028/tree/ggml/src/ggml-cuda/ggml-cuda.cu` | `47abaa3b…` (= `src-ggml-cuda` in its receipt) |
| trio `mmvf.cu` | `4cf2acc4…` |
| trio lib `build/libggml-hip.so.0.18.1` | `11800fe6…` |
| trio `patch/patch.diff` | `1b5e6ba8…` |
| diag hip `mtp-m3-tail-alloc-20261027/ggml-cuda.tailwhy.cu` | `22e88007…` |
| llama base `mtp-hc-row-tail-candidate-20261025/sources/qwen4exp.cpp` | `bd619a5b…` |

### G1 static
1. `git diff candidates/router-scratch-20261031 candidates/m3tail-storage-20261031 -- ggml/` empty.
2. `git diff f1c5fafc0 HEAD -- src/models/qwen4exp.cpp` == `audit/m3-tail-storage.patch.txt`;
   only `build_hc_mix`, one helper, no node/op/op_params added; no `ggml_set_output` added;
   no change to `GGML_TP5_HC_KEEP_LO`, `GGML_TP5_HC_ROW_EPILOGUE`, `NO_REUSE` semantics, or the
   HIP guard; `nrows==1` exception untouched.
3. `grep -n "NO_REUSE" ggml/src/ggml-alloc.c` → 4 sites only (in-place skip, free skip, dst/src
   capture, `needs_realloc`); no other consumer of flag 32 anywhere (`grep -rn NO_REUSE ggml/ src/`).

### G2 build (Devops only)
* `libllama-nt` from `qwen4exp.cpp` with the row-tail recipe
  `mtp-hc-row-tail-candidate-20261025/commands/qwen4exp.cpp.argv.json` (+`llama-link.argv.json`),
  objects out to the lane; expected new libllama sha (record it; must differ from
  `mtp-m3-tail-alias` `libllama-ta` `fd06757b…` only because of this diff).
* `libggml-base-nt` from `ggml-alloc.c` / `ggml.c` using
  `mtp-ready-snapshot-consolidated-20261025/commands/base-link.argv.json`, recompiling
  `base-000-ggml.c.o` (`ggml.c`) and `base-002-ggml-alloc.c.o` (`ggml-alloc.c`) with their
  recorded argv; `ggml.h` is header-only for the enum.
* Because `ggml-alloc.c`/`ggml.c` are relinked into the base library, **the device ISA must be
  unchanged**: relink `libggml-hip` from the **unmodified** trio objects
  (`11800fe6…` / `mmvf.o 2b52b9ac…`) and confirm `sha256(libggml-hip) == 11800fe6…`
  (= ref `mtp-m3-trio-20261028` relink). STOP if not (G2 fail-closed).
* Selected installs untouched; no RSZ_RT in any leg.

### G3 micro/mock (before any long leg)

1. **Reach, minimal legs (2 legs, 8x2, one diag DSO + one llama):** the shipping trio DSO
   `11800fe6…` emits no diag, so reach needs a *modified* HIP DSO. Do **not** inherit the
   2026-10-27 `libggml-hip-tailwhy.so` (`cf14e803…`): it predates the cleaned trio (old
   `TP5_HC_N3_TRIO` spelling, rr=1 kernels, different object hashes), so its firing counts are
   evidence for the *mechanism*, not for `11800fe6`. Minimal diag build:
   * sources: `mtp-m3-trio-20261028/tree/ggml/src/ggml-cuda/ggml-cuda.cu` (`47abaa3b…`) with only
     the alias-lane `mtp-m3-tail-alias-20261028/patch/diag-only.ggml-cuda.diff` hook applied
     (it replaces the veto `if` at trio line ~5649 by the same veto that logs
     `M3_TAIL_WHY why=0/1..7`, gated by `M3_TAIL_DIAG=1`) — the trio `mmvf.cu` and kernels stay
     byte-identical, `diff` the applied hunk to prove only the log branch was added;
   * link with the trio link argv (`mtp-m3-trio-20261028/build/link-m3t.argv.json`), trio
     `mmvf.m3t.o` (`2b52b9ac…`) untouched → record `libggml-hip-diag2` sha; the trio kernels inside
     must match `11800fe6` (same mmvf object, only `ggml_cuda_try_fuse` diff);
   * llama side: the **new** `libllama-nt` (this patch) for ON, and either the trio-lane
     `install/lib` copy of the row-tail libllama or the same `libllama-nt` for the control;
   * DSO identity check in each leg's `install/lib`: `libggml-hip.so.0.18.1` = diag2 sha,
     `libllama.so.0.0.2451` = `libllama-nt` sha, `ldd`/ld probe resolves inside the leg dir
     (the alias lane's `micro2-ldprobe` pattern); never the selected `c4a1634f` unless the leg is
     explicitly the stock control.
   * Legs (8x2, `TP5_DIAG_GRAPH=1`, `M3_TAIL_DIAG=1`, row-tail env):
     `nt_off8x2` = `GGML_TP5_HC_N3_TRIO=1`;
     `nt_on8x2`  = `GGML_TP5_HC_N3_TRIO=1 GGML_TP5_HC_N3_TAIL_KEEP=1`.
   * PASS = ON `M3_TAIL_WHY why=0` totals **3870** with `why≠0` = 0 (rows 2/3/4 = 970/1940/960),
     including the two no-inject mixer sites (`out=result_norm` and `out=result_output`), and
     `reject` only the 40 foreign Q8_0-shape line; OFF must still show `why=1` ≈ 3840 / `why=0` = 30.
   * A cheaper *negative* control without a second build: the stock DSO `c4a1634f…` (trio env
     unset) must show the same kernel census without `hc_tail_rows`; use it only to prove the
     census tool itself is not blind.
   * If a reading of the runner scripts is done first: `mtp-m3-tail-alloc-20261027/rundiag.py`
     selects the HIP DSO by tag (`m3` → `mtp-hc-n3-trio-m3-20261027/install/lib` `6a19849d…`,
     which is the *old* measured trio, **not** `11800fe6`; `tailwhy` → `cf14e803…`), and copies
     `tailkeep` `libllama` `92e48ec2…`; the alias lane's runner (`patch/ta_run.py`,
     tags `m3t`/`diag`/`ta`) picks `11800fe6…` + `fd06757b…` or its diag `aeeb21e3…`. So a reach
     run that merely reuses a historical runner silently measures a different DSO than the
     shipping one — the leg's `install/lib` sha and `command.json` `"hip"` field are the only
     admissible proof of which DSO ran.
2. **Gate-exactness (no GPU needed):** `GGML_TP5_HC_N3_TAIL_KEEP` set to `""`, `0`, `01`, `true`,
   `" 1"`, `2` with trio=1 → identical `why` histogram to the OFF leg; `GGML_TP5_HC_N3_TRIO=0` with
   `TAIL_KEEP=1` → OFF leg.
3. **Compute-buffer delta:** read `Meta() compute buffer` MiB from stderr, OFF vs ON; expected
   ≈ +0.4–0.6 MiB; must not exceed the row-tail config's headroom (record it).
4. **Kernel census** (rocprofv3, 8x2, per agent), the decisive *timing* cross-check: the retained
   allocation shifts the arena plan, so record the histogram and require
   `hc_down_rows` = `hc_tail_rows` = `hc_combine_rows` ≈ 871 (i.e. no generic
   `hc_tail4_rows_generic`, no generic `mul_mat_vec`+`quantize_q8_1` tail), and compare TOTAL
   against `mtp-m3-tail-alloc-20261027/census-off-vs-keep-8x2.json` (`TOTAL [78574, 73207, −5367]`
   for the old pin). A TOTAL **above** the pin's 73207 by more than the guard-checked non-tail
   drift seen there (`topk_moe +31`, `bin_bcast −2623`, `combine_rows +864` are layout luck, not
   semantics) means the new layout lost fusions and must be diagnosed before any timing claim.
   Note the census direction: column 1 is the pre-fix profile, column 2 the pinned one, and
   `hc_tail4_rows_generic` drops 871 → 0 in the pinned profile.

### G4 numerics (soft gate, 32x2, `cmplogits.py` as in the alias lane)
Legs: `ref` = `mtp-hc-row-tail-candidate-20261025` row-tail on32x2 (or `replicated32x2` +
`m3t_off` as in the alias lane), `nt_off32x2` (trio only), `nt_on32x2` (trio+tailkeep),
`nt_on_rep32x2` (repeat), plus `ta_on32x2` (old OUTPUT pin, for the delta vs the historical fix).
Acceptance:
* `nt_off32x2` **bitwise** == ref trio-only leg (OFF equals trio alone);
* `GGML_TP5_HC_N3_TAIL_KEEP=1` without trio **bitwise** == ref (gate);
* `nt_on_rep32x2` == `nt_on32x2` **bitwise** (determinism, no race);
* `nt_on32x2` vs `nt_off32x2`: 0 non-finite, top1 equal wherever the margin > 0.10, KL mean < 1e-3,
  max < 0.03 and drift toward N1 (`mtp-hc-row-tail` N1 envelope max KL 0.067) — i.e. the same
  soft-gate class the alias lane accepted, and each near-tie flip must be listed with its margin;
* no new `TP5_MTP_*` counters regress; no RSZ_RT; no `LATEBIND` interaction (LATEBIND unset).
* Formal 512x2 only if G3/G4 pass; expected outcome is "correct + not slower", a speed claim needs
  the paired ABAB median (`tB_*`/`tC_*` in `patch/gpu_ta.sh`) and a noise band.

### G5 promotion
* Merge prerequisite: §4 (single `ggml/` copy).
* Keep default **off** (`GGML_TP5_HC_N3_TAIL_KEEP` unset) until G3(1) shows `why=0` == full fire
  count and G4 passes; then it may be promoted as the *storage* half of M3 (the trio stays a
  separate knob because its steady-state gain was NO-PROMOTE in `mtp-m3-tail-alias-20261028`).

## 5a. 2026-10-31 Devops reach run: what moved, and the layered diagnosis

Real data: `/home/kunweiz/work/lead-m3cand-reach-20261031/{VERIFY-M3CAND-REACH.txt,runs/cand_{off,on}8x2}`,
`runs/PER-ROW-KL.json`.  Legs: candidate `libllama` `fbfa019e…` (this patch's setters), candidate
`libggml-base` `e8445fbe…` (NO_REUSE allocator, control relink `69545831` exact), trio
`libggml-hip` `11800fe6…` **unmodified** (`1edd2678…` is its diag2 copy; device `.text` identical),
runner `1160f624…`; identical in both legs; only `GGML_TP5_HC_N3_TAIL_KEEP=1` differs.  Re-read
here (read-only, stdlib math):

| quantity | value | note |
|---|---|---|
| reach | 774/774 sites `why=0`, 0 other why | the retention contract removed every `out↔lo` veto, incl. mixer sites |
| site split | rows 4 = 192, rows 3 = 388, rows 2 = 194 | 96 + 194 + 96 + 194 + 194 per graph set |
| graph sequence | 382 graphs, **identical** shape list in both legs | fused probe (`node=23`, `n_nodes=35`) vs generic (`node=16`, `n_nodes=28`) both fire |
| emitted tokens | 16/16 equal, top1 16/16, 0 flips | no sampled-token change |
| KL (their 4-dp Triton-style) | mean 1.816e-3, max 1.543e-2 | recomputed here as **1.816e-3 / 1.543e-2** with a pure-python reference softmax |
| per-row KL | 3 rows ≈ 1e-8…1e-7, 13 rows 1e-4…1.5e-2 | non-uniform: not a constant rescale |
| logits' own scale | `maxabs` logit 0.56–1.80 on **every** row incl. the 1e-8 ones | so the rows that stayed put did NOT change logits — the KL is all in softmax weights |
| top-8 index sets | agree 6–8/8, disagreements listed per row | index churn coexists with sub-1e-7 KL rows |
| MTP activity | identical counters on both legs (`drafted 7/5`, `accepted 3/4`) | no "different number of verify rows" explanation |

**Which comparison is meaningful.**  This pairing is *tail fused vs tail not fused*, i.e. the same
class the alias lane already ran (`ta_trio32x2` vs `ta_on32x2`, 32x2, 64 rows): there the
tail-vs-tail-off delta was KL mean 0.00195/0.00106, max 0.0115/0.0055 with **token divergence at
rows 2 and 26**, and it was accepted as soft drift.  The new run is *smaller* (mean 1.816e-3, max
1.5e-2, no token change), so the KL value alone is not a regression of the same quantity; the
gate that rejects it ("numeric-order-only, mean < 1e-6") cannot be met by any tail-fused run and
was not met by the alias run either.  The residual open question is the **non-uniform** spread:
13/16 rows move while 3 stay at 1e-8, and 8/16 rows disagree on top-8 indices.  A "layout only"
story (same kernels, different addresses) predicts *either* a uniform ulp-level shift *or* a
specific set of guard-driven fallbacks at exactly the sites whose aliases the old pin created —
so the next step is not a bigger KL table but **placement + per-layer capture** (§5b).

**Layered attribution order** (each layer is independent, stop at the first that separates):
1. placement: per-site (buffer, offset) of `up/lo/xn/out/weights`, off vs on — tool
   `diag/m3-tail-stamp-diff.py` over `M3TAIL_STAMP=1` logs from `diag/m3-tail-stamp.ggml-cuda.diff`;
2. kernel identity: per-site fired-kernel name vs the generic chain (node index 23 fused /
   16 generic already known), plus the census of §5-G3(4);
3. value provenance: per-layer boundary capture, off vs on, using the existing
   `--capture-logits`/`GGML_TP5_MTP_*_TRACE` plumbing or a layer-tap list; the first layer whose
   `hc_mixed` rows differ names the tensor;
4. only then a numerics verdict.

`diag/m3-locate-divergence.py` is the read-only companion that turns one captured pair into
row-level evidence (verified on this capture: reproduces their mean/max and flags the 8/16
top-8 disagreements with the per-row `|dlogit|` on the shared index set).

## 5b. Deterministic layered diagnostic (code delivered, not run)

Files (branch `candidates/m3tail-storage-20261031`):

| file | what it is |
|---|---|
| `diag/m3-tail-stamp.ggml-cuda.diff` | log-only replacement of the trio's final veto block. Same `why` 0..7 decision, and with `M3TAIL_STAMP=1` it appends one line per site: `why,node,rows` plus `(buffer,offset)` of `up/lo/xn/out/weights` (`offset` = `data - ggml_backend_buffer_get_base(buffer)`, so it is stable across legs). It touches nothing else; the guard still fails closed. |
| `diag/m3-tail-stamp-diff.py` | read-only stdlib comparator: aligns the two logs by `(node,rows)` sequence, reports the count with *identical* placement and lists the first differing `(tensor, offset)` per site; flags a `why=0` site whose `lo`/`out` still overlap (diag-build bug); maps the first differing sites onto `--rows-jsonl` resume positions. |
| `diag/m3-locate-divergence.py` | read-only stdlib comparator for one captured pair: per row token-KL, top-8 index agreement, and the `|Δlogit|` on the shared top-8 index set (non-zero ⇒ the logits moved *before* softmax), next to the cycle log's `(rows,position)` so prefill/catch-up rows are not mixed with verify rows. |

Pairing that makes the diagnosis decisive with one extra leg pair (all under flock, `predict 8`, same
argv as `lead-m3cand-reach-20261031/run-cand8x2.py`):

| leg | env | question it answers |
|---|---|---|
| A `ref` | stock pair (`libggml-hip` `c4a1634f`, unmodified libllama `e959386a`, no TP5 HC knobs) | ground truth for row 0 |
| B `trio_off` | trio `11800fe6` + candidate libllama + `GGML_TP5_HC_N3_TRIO=1` | already run (this run's OFF) |
| C `tail_on` | + `GGML_TP5_HC_N3_TAIL_KEEP=1` | already run (this run's ON) |
| D `tail_on_stamp` | C + diag2 `M3TAIL_STAMP=1 M3TAIL_STAMP_PATH=…`, and B likewise | placement diff `m3-tail-stamp-diff.py` |
| E `lo_only` (optional) | C built with the `qwen4exp_hc_n3_retain_xn` call removed (code change, not an env value: the gate is strict `==1`) | separates "lo retention" from "xn retention" if D shows only mixer sites differing |

Order of reading: B vs A must stay at the alias lane's trio-only class (mean ~1.95e-3/1.06e-3 rows
8–40) — if it moves, the candidate llama/base relink changed something unrelated and the run is not
about the tail at all.  Then C vs B per layer; then D's placement diff.  A verdict of
"correct, soft drift of the same class as the alias lane" needs rows 8–40 and per-row flips with
margins; a verdict of "bug" needs a site whose `why=0` but tensors overlap, or a layer whose
captured tensor differs beyond the ulp class.

## 5c. Trunk / mixer isolation and the N=1 row-reference plan (code in this branch)

**Gate matrix (handoff-conformant, restored after review).**  `GGML_TP5_HC_N3_TAIL_KEEP`
enables retention **only** for the literal `"1"` and only together with `GGML_TP5_HC_N3_TRIO == "1"`;
`unset`, `""`, `"0"`, `"01"`, `"true"`, `" 1"` and every other spelling are OFF, and no value is
rejected (an exception would itself be an OFF-matrix violation).  Production coverage is therefore
unchanged and complete: trunk **and** mixer sites, with the mixer half still failing closed when
`xn`'s owner cannot be retained.

**What the previous revision got wrong** (kept as evidence, commit `7732fdaca`): it overloaded the
production variable with the values `trunk` / `mixer` and threw on anything else, which both broke
the exact-`1` off-matrix and let a mistyped value abort a leg instead of running it OFF.  That is
reverted in `fe2e48a11`; the isolation now lives in a separate, explicitly diagnostic variable:

| variable | values | effect | promotion input? |
|---|---|---|---|
| `GGML_TP5_HC_N3_TAIL_KEEP` | `"1"` + `TRIO=="1"` → on; anything else → off | complete retention (trunk + mixer) | yes |
| `GGML_TP5_HC_N3_TAIL_DIAG_SCOPE` | `"trunk"` → trunk lo retention only; `"mixer"` → mixer lo+xn retention only; unset / anything else → off | diagnostic only, cannot turn retention on by itself | **never** |

The two branches are evaluated separately in `build_hc_mix`, so the diagnostic cannot narrow `"1"`,
and every receipt for a diagnostic leg must list both variables verbatim.

**Why the baselines are split.**  The M3 micro baseline
(`mtp-hc-n3-trio-m3-20261027/src/m3-micro.cpp`) always calls the inject projection, so it only ever
builds a **trunk** site; `numeric-check.json` (`trio_vs_N1_rows` bitdiff 0, `trio_vs_N3off` max 3.4e-4)
therefore certifies trunk sites only.  The 2026-10-31 reach log shows the fix also fires at the two
`inject == nullptr` mixer sites, which have no reference — hence `micro/m3-mixer-n1ref.cpp`
(the same block with the inject consumer dropped, the real mixer's F32 output projection added, and
per-row `nt=1` images written with the trunk micro's `M3_ROW0` seeding) plus
`micro/build_mixer_n1ref.py` (compiler flags taken verbatim from the lane's recorded
`ggml-cuda-m3t.argv.json`, so no flag is invented).  Both are authored, not built.

**Reading order** (Devops, GPU flock; nothing here is a threshold change):

1. `diag/m3-n1-rowref.py <candidate> <n1-reference>` per row: `dlogit_max`, float32 **ulp step**,
   `bitdiff`, `top1`.  The reference is either a `--predict 1` leg or the `chk-N1-r*.bin` /
   `.nt1.<r>.bin` images; the tool refuses anything else, so an off/off pair cannot pose as N=1.
   Accepted class is the one the micro already measured: HC values 0 bit, propagated logits 1–3 ulp,
   top1 equal.
2. `diag/m3-split-trunk-mixer.py <leg> <stamp>`: rows split into N=1 vs multi-row; site graph sizes
   (3 = trunk complete, 2 = trunk without tail, 1 = mixer graph).  Trunk verdict from the `trunk`
   diagnostic leg, mixer verdict from the `mixer` diagnostic leg — never a pooled average.
3. `diag/m3-tail-stamp-diff.py`: placement (buffer, offset) per site, first differing tensor.
4. `diag/m3-locate-divergence.py`: per row, whether the difference is already in the logits
   (top-8 index sets equal but `|Δlogit| > 0`) or only in the softmax tail.

## 5d. Fixes after the Devops harness audit (2026-10-31, `lead-m3mixer-verify-20261031`)

Devops found two real defects in the first harness revision; both are fixed and both are recorded
here as evidence rather than edited away:

1. **Recipe flag loss.**  `build_mixer_n1ref.py` whitelisted `-I/-D/-O/-f/-W/--offload-arch/-std`
   and thereby dropped the lane's `-x hip`, so the generated command failed on
   `hip/hip_runtime.h` (`build-harness.log`).  Both recipes now copy **every** recorded argv element
   and drop only output naming (`-o/-c/-MF/-MT/-MD`, `-Wl,--dependency-file`); the n1ref recipe also
   stopped re-adding `-fPIC/-O3/--offload-arch` the template already carries.
2. **Wrong subject under test.**  The first harness rebuilt the HC block with local ops and **F32**
   weights, so it never entered `llama_model_qwen4exp::graph::build_hc_mix`, never met the
   production `Q8_0` door and could not observe the retention decision.  The replacement is
   `micro/m3-mixer-prod-path.cpp`:
   * weights exactly as the model loads them — `hc_*_down` Q8_0 `[10240,320]`, `hc_*_up` Q8_0
     `[320,10240]`, `hc_*_inject` Q8_0 `[10240,4]` (trunk) or absent (mixer);
   * the site is built by the **shipped** builder through `micro/m3-probe-hc-mix.cpp`, a subclass of
     `llama_model_qwen4exp::graph` that exposes the one protected member; the definition called is the
     one inside the candidate `libllama`, not a copy;
   * the harness never sets `GGML_TENSOR_FLAG_NO_REUSE`; it reads `lo->flags` back and prints
     `lo_retained`, so a leg **witnesses** that the production gate decided it;
   * per-row `nt = 1` images are written by the same builder with the `M3_ROW0` seeding, which is the
     `chk-N1-r<k>` convention, for `diag/m3-n1-rowref.py`.
   The only test-only code change is a hook pair in `src/llama-graph.h` under
   `M3_PROBE_GRAPH_HOOK` (a non-owning `ctx_compute_view` plus `reset_context_view`), printed verbatim
   by `micro/build_mixer_prod.py`; without the macro nothing changes and the null-view path is the
   shipped one.

Production diagnostics added (inert unless the env value is exactly `"1"`): `build_hc_mix` now
labels `hc_lo`/`hc_up` and calls `qwen4exp_hc_mix_probe::hc_mix_built`, which emits
`M3_TAIL_BUILD` (scope, il, nt, and the retained flags of `lo` and `xn` read from the tensors) under
`M3_TAIL_DIAG=1` and per-site placement rows under `M3TAIL_STAMP=1`.  Names and logging only; no
value, op, flag or gate changes, and the production gate stays exact `"1"` with the OFF matrix
(`""`/`0`/`01`/`true`/anything) intact.

Gate matrix after this round (unchanged from §5c): `TAIL_KEEP` exact `"1"` + `TRIO` `"1"`; every other
value OFF; the diagnostic scope is a separate variable and is never a promotion input.

## 5e. Recipe corrections after the second Devops audit (2026-10-31)

Devops re-ran the printed argv verbatim and reported four concrete defects; all four are fixed in
`micro/build_mixer_prod.py` and `micro/build_mixer_n1ref.py`, and the fixes are structural rather
than positional:

| defect (executed argv) | cause | fix |
|---|---|---|
| compiler appeared twice → `ld.lld: unknown file type` | the compiler was taken from both recorded commands and concatenated | exactly one compiler per command, fixed to the lane's HIP `clang++`; asserted before printing |
| `.cpp` after `-MMD` with no `-o`/`-MF` | template flags were concatenated and the source appended at the end | every compile is `-c <src> -o <obj> -MMD -MF <obj>.d`, asserted (`-c < -o < -MMD < -MF`) |
| link line had no `-L`/rpath for the private dir | the recorded link argv was mined element-wise and its absolute lane paths dropped | link is rebuilt: objects, `-L<libdir>`, `-Wl,-rpath,<libdir>`, `-Wl,-rpath-link,<libdir>`, then the recorded library list, then `-o <bin>`; asserted that `-L<libdir>` precedes the first `-l` |
| flag handling was position-based | `-o/-c/-MF` were rewritten in place | argv is parsed into (compiler, flags, source) with the driver's own grouping: value-carrying switches stay with their value, output naming is dropped, nothing else is touched (`-x hip`, `-iquote <dir>`, `-std=…`, `--offload-arch=…` all survive as groups) |

Additionally the recipes now fail closed: a `--libdir` that does not exist, or that holds no
`libllama`, or a missing probe source stops the script with `rc=2` and no command is printed
(previously a link line was printed for a directory that could not exist).  Nothing was built or
run for this revision: the production harness (`m3-mixer-prod-path`) and both recipes are
**authored, NOT verified**, and G2 is **not** claimed.

## 5f. Third Devops audit: the parser mistook the dependency file for the source

Devops executed the recipe from `da7651743` and hit `AssertionError` at the single-`-c` check, then
`clang: error: cannot specify -o when generating multiple output files`.  Root cause, reproduced by
reading the two recorded argv verbatim: both use the **separated** form

```
<cc> ... -o <obj> <obj> -x hip -c <src>.cu -iquote <dir> -MMD -MF <obj>.d
```

so a parser that keeps "the first operand that is not a switch" as the source picks the value of
`-o` and then the `.d` value of `-MF`, leaving `-c` and the real `.cu` inside the flag list — hence
two `-c`s, two inputs, and no usable command (§5e's assertions fired for exactly that reason).

Fix (both recipes, `micro/build_mixer_prod.py`, `micro/build_mixer_n1ref.py`):

* the recorded argv is parsed with explicit switch semantics — every value-carrying switch is
  listed by what its value *is*: output naming (`-o/-MF/-MT/-MQ`, dropped), language (`-x hip`),
  search (`-iquote/-isystem/-include/-I/-L/--sysroot`), defines (`-D/-U/-l/-arch`), driver
  pass-through (`-Xlinker/-Xclang/-mllvm`), and bare output switches (`-MD/-MMD`);
* a value that looks like a switch, a missing value, a second `-c`, a second positional operand, or
  a positional operand that is not a `.c/.cc/.cpp/.cxx/.cu/.c++` file each **raise** — the script
  prints `# ERROR: …` and exits 2 without emitting any command;
* the emitted compile command contains exactly one `-c` (the probe), one `-o`, one `-MMD`, one
  `-MF <obj>.d`, the recorded flag tail with no output switch left in it (checked), and exactly one
  existing source (checked by existence, not by suffix alone);
* the HIP template must still provide `-x` and `-iquote`, and the model template's source must
  exist, otherwise the script refuses instead of guessing;
* the assertions were replaced by explicit checks with a printed reason, because an `AssertionError`
  in a script whose job is to emit a command is indistinguishable from a real build failure.

Nothing was executed for this revision: `m3-mixer-prod-path`, `m3-mixer-n1ref` and both recipes are
**authored, NOT verified**, and **G2 is not claimed**.

## 5g. Fourth Devops audit: hook closure and the seven-argument call

Two independent blockers, both inside the delivery (`lead-m3mixer3-verify-20261031`):

1. **The hook was printed, not delivered.**  `build_mixer_prod.py` only *emitted text* for
   `src/llama-graph.h` while compiling with `-DM3_PROBE_GRAPH_HOOK`, so the frozen tree had no
   `reset_context_view` and the probe TU could not compile.  Fixed by making the change a real file
   edit in this tree, guarded so production is untouched:

   ```cpp
   ggml_context * get_ctx() const {
   #if defined(M3_PROBE_GRAPH_HOOK)
       if (ctx_compute_view != nullptr) { return ctx_compute_view; }
   #endif
       return ctx_compute.get();
   }
   #if defined(M3_PROBE_GRAPH_HOOK)
       void reset_context_view(ggml_context * c, ggml_cgraph * g) { ctx_compute_view = c; gf = g; }
       ggml_context * ctx_compute_view = nullptr;   // non-owning, next to ctx_compute
   #endif
   ```

   The macro is defined **only** by the probe's compile line, so a normal build sees the shipped
   header byte for byte; there is no permanent test-only API on the production class (the member
   and both functions disappear without the macro, and the null-view path *is* the shipped path).
   The recipe no longer prints a patch: it **verifies** that the tree's header carries the token
   trio and that the guard appears at least twice, and refuses to emit a `-D` command otherwise.
2. **Wrong arity at the call site.**  The trampoline called `hc_mix(x, w_norm, w_down, w_up,
   &inject, il)` (6 arguments) while the production signature is
   `build_hc_mix(x, w_norm, w_down, w_up, w_inject, ggml_tensor ** inject, il)` (7).  Fixed: the
   weight is now passed explicitly as `w_inject` — never `NULL` in place of a required weight — and
   `inject` is null *exactly* for the mixer site, which is the production way of selecting it.  The
   subclass declaration is spelled with the same parameter list so a future drift is a compile error.

Nothing was compiled, run or syntax-checked for this revision by me: the probe TU, the harness and
both recipes remain **authored, NOT verified**, and **G2 is not claimed**.

## 5h. Fifth review: remove the macro-level layout change; sole ownership of the test context

The independent scheduler review is right on both counts, and the previous revision is withdrawn as
evidence of a usable build:

* `M3_PROBE_GRAPH_HOOK` added a member to `llm_graph_result` and changed the inline `get_ctx()`.
  Because the macro was defined for the probe TUs only, the class had **two layouts** and `get_ctx()`
  two definitions across the binary (ODR/ABI split), and — worse — the production builder runs in the
  library TU, where `build_hc_mix` reaches the context through `res->get_ctx()`; a shadow member the
  probe could see but the library could not would make the hook useless even if it linked.
* The fix is not a better macro, it is no macro: `src/llama-graph.h` is now **byte-identical to the
  production header** (`git diff f1c5fafc0 -- src/llama-graph.h` is empty) and every TU of the probe
  is compiled with exactly the library's flags.

Ownership, the part that needed care:

* `llm_graph_result::ctx_compute` is a `ggml_context_ptr` = `std::unique_ptr<ggml_context,
  ggml_context_deleter>` and the deleter is `ggml_free` (`ggml/include/ggml-cpp.h:17`).  The shipped
  `reset()` does exactly `ctx_compute.reset(ggml_init(...))`, so a result can be handed a
  caller-created context through the same unique_ptr type and will free it on destruction with the
  same deleter — one owner, one free.
* The probe therefore does:

  ```cpp
  std::unique_ptr<llm_graph_result> res(new llm_graph_result(4096,
          std::unique_ptr<ggml_context, ggml_context_deleter>(g_probe.ctx)));
  g_probe.ctx = nullptr;                 // ownership moved, nothing left to free twice
  params.res = res.get();                // the library reads get_ctx()/get_gf() from it
  ...                                    // build_hc_mix(...) — the shipped definition
  g_probe.res = res.release();           // outlives the call; caller reads tensors back
  ```

  `m3_probe_release()` deletes it, `m3_probe_attach()` refuses to install a second context while one
  is outstanding, and a build whose context, weights or input do not come from the handed-over
  context is refused (checked via the backend buffer's `context` tag, since the public API does not
  expose a tensor's context).  The harness allocates weights, input and graph in that one context, so
  there is a single allocation site and a single free site.
* **No production API was added**: nothing new in `llama-graph.h`, `llama-impl.h` or the model header;
  the probe only subclasses `llama_model_qwen4exp::graph` (which exposes one protected member to a
  derived type) and uses the public `llm_graph_context` constructor.

Every claim in this section is a static reading of the code.  Nothing was compiled, run or
syntax-checked; **G2 is not claimed** and the probe/harness/recipes remain authored-only.

## 5i. Sixth review: use the shipped `llm_graph_result(max_nodes)` as-is

Devops' G2 attempt found two hard interface errors in §5h, both correct:

* `llm_graph_result` has **one** public constructor, `llm_graph_result(int64_t max_nodes)`, whose
  body calls `reset()` and creates its own context and graph.  The "adopting" constructor assumed in
  §5h does not exist, and adding one would be a production API/ABI change.
* `ggml_backend_buffer::context` is the *backend's* private context (for a CPU buffer, a malloc base
  address), not a `ggml_context *`; the `in_ctx` precondition was therefore always false.

The probe and harness are rewritten to the real interface:

* the probe constructs `llm_graph_result(4096)` and uses **the context and graph that result already
  owns** (`res->get_ctx()`, `res->get_gf()`), which is exactly the pair the library dereferences
  inside `build_hc_mix`; the probe and the library therefore share one context by construction;
* the weights (`F32 [n_embd,4]`, `Q8_0 [10240,320]`, `Q8_0 [320,10240]`, `Q8_0 [10240,4]` for the
  trunk) and the input `F32 [n_embd,4,nt]` are created **in that context**, with `no_alloc = true`
  (the context the result creates is a no_alloc context), so nothing is allocated outside it;
* the harness attaches backend storage through the **public** tensor allocator —
  `ggml_backend_buft_alloc_buffer` + `struct ggml_tallocr` (`ggml-alloc.h:10-22`) +
  `ggml_tallocr_alloc` — one buffer for all probe tensors, freed before the result;
* release is `delete res`, the only owner; there is no adoption, no deleter swap, no
  `buffer->context`, and no private header (`ggml-backend-impl.h` is no longer included);
* `src/llama-graph.h` remains byte-identical to production and the recipe still adds no `-D`.

Witnesses, stated honestly:

* **Witness A (diagnostic only):** `M3_TAIL_BUILD` from the model TU (scope, il, nt, tensor names,
  retained flag bits) plus the harness line with `lo_flags`; neither llama nor ggml exposes a public
  query for the retention decision and this delivery adds none, so a flag bit is *logged*, not
  asserted.
* **Witness B (the numeric check):** `diag/m3-n1-rowref.py` over `<prefix>.rows.bin` vs
  `<prefix>.nt1.<r>.bin`, per row ulp/bitdiff/top1.

Nothing in this section was compiled, run or syntax-checked: **G2 remains unproven**.

## 5j. Seventh review: opaque cgraph, and a witness that can tell "not retained" from "gate off"

1. **Compile error.**  `ggml_cgraph` is opaque in the public headers, so `gf->n_nodes` does not
   compile from the probe.  Fixed by the documented accessor: `ggml_graph_n_nodes(gf)`; no private
   header is included anywhere in the probe or harness.

2. **The witness was not discriminating.**  A log line carrying `lo->flags` proves nothing on its own:
   the same zero bits appear when the library decided not to retain *and* when the gate never armed.
   Two read-only instruments now separate those cases, both emitted by the code under test and both
   inert unless asked for:

   | instrument | where | what it shows | how it discriminates |
   |---|---|---|---|
   | `M3_ALLOC_COUNTERS` | `ggml/src/ggml-alloc.c`: counted at the allocator's own decisions (`ggml_gallocr_free_node` retention skip; in-place refusal for a retained parent; in-place refusal through a retained view owner), printed at the end of `ggml_gallocr_alloc_graph` | `retained_free_skip`, `no_reuse_inplace_skip`, `view_reuse_skip` | ON leg: `retained_free_skip > 0` (and `view_reuse_skip > 0` at a mixer site).  OFF leg and every mis-armed spelling (`unset`, `""`, `0`, `01`, `true`): all three are **0**.  A non-zero count can only be produced by a tensor that really carries `GGML_TENSOR_FLAG_NO_REUSE`, so "gate armed" and "gate off" are distinguishable by data, not by reading a log's wording |
   | `M3_TAIL_BUILD` | `src/models/qwen4exp.cpp` (model TU, namespace scope) | per built site: scope (trunk/mixer), `il`, `nt`, tensor names, and the flag bits of `lo` and `xn` | says *which* site was built and at which `nt`, so a count from the first instrument can be attributed to a site; it is deliberately not the criterion |
   | row images | harness | `*.rows.bin` vs `*.nt1.<r>.bin` | the numeric witness (`diag/m3-n1-rowref.py`, per-row ulp/bitdiff/top1) |

   The counters live in `ggml-alloc.c` (allocator build), are `static` (no exported symbol, no API, no
   header change), are written only in the free/reuse branches, and are read by nothing: the
   allocation plan is identical whether the switch is on or off.  "Permanent test-only API" is
   therefore not introduced — the only externally visible surface is a log line gated on an
   environment variable that is off by default.

3. **What a leg must show to be admissible** (Devops):
   a. `libggml-base` rebuilt from this HEAD (the counters live there), same lane argv;
   b. ON leg `TAIL_KEEP=1`: `retained_free_skip > 0`, `view_reuse_skip > 0` for the mixer site;
   c. OFF leg (`TRIO=1` only) **and** the gate-exactness legs (`""`, `0`, `01`, `true`): all counters 0;
   d. `M3_TAIL_BUILD` present once per site in both legs with the expected scope/`nt`;
   e. then `diag/m3-n1-rowref.py` per row.
   If (b) fails while (d) shows the site was built, that is a real finding about the gate, not a
   numerics question; if (c) differs from OFF, the gate matrix is broken and the leg is void.

Nothing in this section was compiled, run or syntax-checked: **G2 remains unproven**.

## 6. Evidence boundary (what is and is not proven)

* **Proven (pre-existing evidence re-read):** the veto is `out↔lo` overlap produced by galloc's
  merge + `SCALE` allocation, 3840/3870 rejects, and a pin removes it (3870 fires, OFF bitwise).
* **Proven here (static):** the new flag's allocator contract is exactly "never freed, never reused
  in place", no consumer in this diff alters values or fusion eligibility; OFF path is a strict
  superset-free no-op; qualified N1 config does not set the trio and cannot reach the new branch.
* **NOT proven:** that under the new primitive the destination is disjoint at **all 3870** sites
  (needs G3-1), the exact compute-buffer delta (G3-3), kernel-census equivalence (G3-4), and every
  G4 numerics number. No build, no syntax check, no test was run for this report by contract.
* **2026-10-31 update (Devops data, re-read only):** reach 774/774 with the unmodified trio proven;
  the trunk/mixer isolation and the production-path mixer test (§5c/§5d) are code-only in this
  branch and have NOT been built or run; the first harness revision was rejected by Devops for
  dropping `-x hip` and for not driving the production builder (§5d), and the recipes were again
  rejected for a duplicated compiler, a misordered `-MMD`, and a link line without the private
  `-L` (§5e), and the parser likewise mistook the `-MF` dependency file for the translation unit
  (§5f), and the probe's hook was printed rather than delivered while its call site used the wrong
  arity (§5g); the hook itself was then withdrawn because a macro-guarded member change splits the
  class layout across TUs and cannot reach the library anyway (§5h adds real sole ownership of the
  test context instead). The adopting constructor and the `buffer->context` precondition it used
  do not exist in the shipped interface, so §5i replaces both with the real
  `llm_graph_result(max_nodes)` and the public `ggml_tallocr` path; the probe also used the opaque
  `ggml_cgraph` struct and its witness could not separate "not retained" from "gate off" (§5j adds
  `ggml_graph_n_nodes` and allocator-side retention counters). Nothing has been executed, so G2
  remains unproven; the production gate is exact `"1"` again and the previous
  `trunk`/`mixer`-inside-`TAIL_KEEP` revision is reverted (kept as evidence above);
  token stream unchanged; KL mean 1.816e-3 / max 1.543e-2 (reproduced).  Cause not yet localised;
  §5a/§5b give the placement-stamp and per-layer capture plan.  The gate stays closed: promotion is
  NOT claimed and no threshold was relaxed.
