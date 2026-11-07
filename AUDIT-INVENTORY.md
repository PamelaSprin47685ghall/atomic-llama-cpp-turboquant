# AUDIT-INVENTORY — "returned 0 / empty set / never executed / UNPROVEN read as PASS"

Survey phase. No source file was changed in that phase. Nothing was compiled.
This is the survey record. The dispositions and the fixing commits are in
`RESULT.md` section 4.

## 0. Identity

| field | value |
|---|---|
| source HEAD | `400f948736478f45cefbc43f2e72fbb11943f1f0` |
| source branch | `e5-union-mainline-r24o-synth-r24o` |
| source date | 2026-11-04 23:20:58 +0800 |
| source subject | `fix(union): stop the bind dispatch after the first kind that ran` |
| clone path | `/home/kunweiz/work/union-gate-return0-audit-r24q-20261104-1625/src` |
| audit branch | `e5-union-gate-return0-audit-r24q` (created at that HEAD, not pushed) |
| origin | `/home/kunweiz/work/union-mainline-r24o-synth-r24o-20261104-1430/src` (read-only) |

## 1. Scope boundary

Searched, in full:

| tree | files | what was looked for |
|---|---|---|
| `src/*.cpp`, `src/*.h`, `include/llama.h` | 79 | `_check` / `_gate` / `_verify` / `_probe` definitions and calls; every `int`/`bool` whose 0/non-0 is judged at the call site; counters compared with `== 0` |
| `tests/*.cpp` | 95 | every `ok=%d` print, every `checked`/`compared`/`expect` counter, every loop whose body may run zero times, every `return x == 0` |
| `tests/nkv-cpu-probe/*.cpp` | 5 | the probe binaries' `HARNESS_END` / `PROBE_RC` / JSON verdicts |
| `tests/nkv-cpu-probe/*.sh` | 18 | `|| true`, `set +e`, `[ -z "$x" ] && …`, `grep -q … && ok=1`, counters initialised to PASS, ignored `rc=$?`, command-not-found, missing log line treated as skip |
| `tests/nkv-cpu-probe/*.py` | 8 | empty-input verdicts, `sys.exit` polarity, report writers |
| `ggml/src/ggml-cpu/*.c`, `*.cpp` | 11 | the attention-work counters and their getter |
| `tests/CMakeLists.txt` | 1 | `PASS_REGULAR_EXPRESSION`, `FAIL_REGULAR_EXPRESSION`, `WILL_FAIL`, `SKIP_RETURN_CODE`, `add_test` whose target may not exist |

Excluded, with reason:

| tree | reason |
|---|---|
| `examples/`, `tools/`, `common/`, `vendor/`, `models/`, `docs/`, `scripts/`, `ci/` | not on the union gate path; no gate verdict is produced there |
| `ggml/src` outside `ggml-cpu/` | the union lane is CPU-only and non-FA; the only ggml symbol the gates read is `ggml_backend_cpu_attn_reduce_stats` |
| `src/llama-arch.cpp`, `src/llama-vocab.cpp`, model loaders | no gate verdict; a 0 there is data, not a pass |
| `third_party/` | absent from this tree |

Counts: 4025 tracked files; 217 in scope; 51 files opened and read; 26 pattern searches; 26 findings; 24 benign sites recorded.

Method note: the audit reads code and text only. Every "what the current code does" statement below is read from the file, not executed.

## 2. Findings

Severity is the chance that a wrong result passes today, not the size of the fix.
`live` = reachable on the current tree. `latent` = the shape is present, but the current control flow prevents it.

### Class A — a C++ return code must become an explicit tri-state PASS / FAIL / UNPROVEN

| id | file:line | snippet | what the code does | why it is a false-pass | fix |
|---|---|---|---|---|---|
| A1 | `src/llama-context.cpp:8639-8696` | `int llama_union_mask_row_check(const llama_context * ctx) {` … `return verdict > 0 ? 1 : 0;` | Returns 1 when the row agrees. Returns 0 for a disagreement **and** for six "nothing was checked" paths: no live mask input, no F32 mask, `n_tokens == 0`, `width == 0` or wider than `ne[0]`, `rebuild_fallbacks != 0`, no KV cache. | The verdict is two-valued over a three-valued fact. A caller cannot tell "the mask is wrong" from "the check did not run". The header states this as intended (`include/llama.h:848-849`). No current caller reads 0 as PASS: the two call sites require `== 1` first. So the collapse is fail-closed today, but the API cannot express UNPROVEN, and `mask_row_refused` is the only place the refusal survives. | A |
| A2 | `include/llama.h:837-849` | `It returns 0 on success` … `It returns 1 when the row agrees, 0 when it does not or when there is nothing to check.` | Two adjacent `int` functions in one comment block use opposite success polarity: `llama_union_kq_mask_row` returns 0 for success; `llama_union_mask_row_check` returns 1 for success. | A caller that swaps the two, or copies a `!= 0` guard from one to the other, inverts the verdict silently. The compiler cannot help: both are `int`. This is the exact "returned 0" trap, one layer up from the mask check itself. | A |
| A3 | `src/llama-union-topology.h:151-171` | `h.mix_u64((uint64_t) nodes.size());` … `return true;` | `hash_nodes` accepts an empty node list, hashes "0 nodes", and returns true. | Two empty graphs hash equal, so a hash-equality claim over nothing passes. `llama_union_width_table::build()` refuses `build_n_kv == 0`, so the width-table path cannot reach it today. A direct caller of `hash_nodes` can. | A |

### Class B — a shell gate must classify the line and fail closed on missing or unproven

| id | file:line | snippet | what the code does | why it is a false-pass | fix |
|---|---|---|---|---|---|
| B1 | `tests/nkv-cpu-probe/run_pad_lane_ab_gate_13022.sh:103-109` | `else` / `echo "NOTE: driver wrote no JSON report at $json"` / `exit "$rc"` | A missing report file prints a NOTE and the script exits with the driver's status. | A driver that exits 0 without writing its report passes this script. The aggregate gate (`union_ctest_gate.py`, artifact `pad-gate-json`) catches it, so the pair is covered. The runner alone is not. | B |
| B2 | `tests/nkv-cpu-probe/run_union_varlen_e2e_13022.sh:110-116` | `echo "NOTE: driver wrote no JSON report at $json" \| tee -a "$log"` / `exit "$rc"` | Same shape as B1 for `union-varlen-e2e.json`. | Same: exit 0 plus no report passes the runner. Covered by the `e2e-json` artifact check, not by the runner. | B |
| B3 | `tests/nkv-cpu-probe/run_probe.sh:143-150` | `if diff -u <(stats_line "$LOGS/unset-$mode.stderr.log") <(stats_line "$LOGS/on-$mode.stderr.log")` … `PASS identical` | Compares the `NKV_BINDER_STATS` lines of two arms with `diff`. | Two empty captures compare equal, so "no stats line in either log" prints `STATS … PASS identical`. `stats_line` returns empty when the line is absent. The arms make it unreachable today (both are binder-enabled), and a failed arm already sets `overall=1`. The check itself does not test that it compared anything. | B |
| B4 | `tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh:183-189` | `gate_hash=PASS` … `gate_kernel=PASS` / `# fail-closed: the BSE block below sets PASS only after every check passed` / `gate_bse=FAIL` | Five gate verdicts start at PASS; only `gate_bse` starts at FAIL. | A deleted, skipped or early-returned block leaves its verdict at PASS. The script's own comment names the correct shape and applies it to one of six gates. All six blocks are unconditional today, so this is latent. | B |
| B5 | `tests/nkv-cpu-probe/gate-p1-merge-13022.sh:108-112` | `gate_a=PASS` … `gate_e=PASS` | Same shape: five verdicts start at PASS and are demoted by checks. | A future `if` around a block, or a deleted block, leaves PASS. Latent today. | B |
| B6 | `tests/nkv-cpu-probe/gate-nkv-observe-13022.sh:156` | `result[$name]=PASS` | The per-arm verdict is set to PASS immediately after the ctest exit status, before the `NKV_ARM`, case-list and evidence checks run. | The verdict is a default that later checks demote. Any new `continue` inserted before a demotion passes the arm. The existing checks are correct; the initialiser is the hazard. Latent. | B |
| B7 | `tests/nkv-cpu-probe/gate-p1-ctest-pack-13022.sh:205` | `result[$name]=PASS` | Same: PASS on exit 0, then `require_line` demotes. | Same hazard. `require_line` returns early when the result is not PASS, so a NOT_RUN stays NOT_RUN. Latent. | B |
| B8 | `tests/nkv-cpu-probe/run_varlen_output_probe.sh:22-24` | `if cmp -s "$OUT/$mode-on.bin" "$OUT/$mode-off.bin"; then cmp_rc=0; else cmp_rc=$?; fi` | Compares two record files with `cmp -s`. | Two zero-byte files compare equal. The probe writes one record per step and requires `pass_steps == steps`, so an empty pair is not reachable while the probe exits 0. The script has no record-count floor of its own. Latent. | B |
| B9 | `tests/nkv-cpu-probe/summarize.py:85-90` | `print(f"TABLE {out_path} rows={len(rows)}")` / `return 0` | Writes the table and returns 0 even when `rows` is empty. | "No (mode, arm) log found" is reported as success. `run_probe.sh:154` only checks the exit status. A missing log already fails its arm, so the aggregate verdict is still FAIL. Latent. | B |

### Class C — a test counter or arm must prove non-zero work

| id | file:line | snippet | what the code does | why it is a false-pass | fix |
|---|---|---|---|---|---|
| C1 | `tests/test-nkv-runtime-param.cpp:840-861` | `if (st.mask_filled == 0)` … `if (st.mask_n_kv != st.runtime_n_kv)` | The live n_kv arm asserts that a mask was filled and that the fill width equals the executed width. It never reads `mask_row_checked`, `mask_row_ok`, `mask_row_refused` or `mask_rows_all`. | `NKV_RT_LIVE ok=1` and `NKV_RT_TRUELEN ok=1` can print with the mask read-back never having run. The read-back is the only independent observation of the mask; the fill width and the executed width both come from the same binder value. The matrix test asserts the floor (`test-union-four-param-matrix.cpp:1728`); this test does not. | C |
| C2 | `tests/test-nkv-runtime-param.cpp:1066` | `if (st.mask_filled == 0 \|\| st.mask_refused != 0 \|\| st.mask_n_kv != st.runtime_n_kv)` | The part-4 positive arm has the same three conditions and no row floor. | `TRIPARAM_MASK_NEG ok=1` can print with no row compared. | C |
| C3 | `tests/test-union-binder-stock-equiv.cpp:529-547` | `if (st.mask_filled == 0)` … `if (st.mask_n_kv != st.runtime_n_kv)` | The BSE per-step stats block asserts fill and width, not the read-back. | `BSE_SINGLE ok=1` and `BSE_MULTI ok=1` can print with `mask_row_checked == 0`. The gate then accepts `BSE_END ok=1`. | C |
| C4 | `tests/test-nkv-exact-width.cpp:1627-1629` | `const bool all_ok = ok && g_all_ok;` / `"NKV_EXACT_END ok=%d cases=%d steps=%d\n"` | The footer's `ok` is the conjunction of the cases that ran. It has no non-zero floor on `g_cases_run` or `g_steps_run`. | If every case were skipped, the line reads `ok=1 cases=0 steps=0`. `parse_binder_arm` rejects unknown arms and every arm owns at least one case, so this is unreachable today. The gate (`gate-nkv-observe-13022.sh:261`) requires `cases=<expected>`, so the gate is stronger than the test. Latent. | C |
| C5 | `src/llama-graph.cpp:371-374` | `if (mask == nullptr \|\| mask->buffer == nullptr) { continue; }` | A registered mask with no buffer is skipped without incrementing `mask_refused` or `mask_filled`. | `mask_rows_all` is computed as `rows_examined == n_tok * mask_filled` (`:435-437`), so the skipped mask is absent from both sides and the measurement still reads 1. The lane registers exactly one mask today. Latent. | C |
| C6 | `tests/test-rsz-lane-rows.cpp:107-111`, `tests/test-union-binder-gate.cpp:107-114`, `tests/test-rsz-selection-write.cpp:467-472` (and the same shape in `test-rsz-hybrid-selection-write`, `test-rs-ring-relay`, `test-rsz-key-injection`, `test-rsz-relay-onegraph`, `test-union-binder-rsz`, `test-rsz-multiseq-decode`) | `if (g_fail != 0) { … FAIL … }` / `… PASS` | The verdict is a failure counter only. No counter records that a check ran. | A deleted `CHECK` line is invisible: `g_fail` stays 0 and the test prints PASS. All checks are straight-line today, so no loop can run zero times. This is a review-detectable gap, not a runtime one. | C |

### Class D — CMake registration / `PASS_REGULAR_EXPRESSION` defect

| id | file:line | snippet | what the code does | why it is a false-pass | fix |
|---|---|---|---|---|---|
| D1 | `tests/CMakeLists.txt:230` | `PASS_REGULAR_EXPRESSION "FOUR_MATRIX ok=1 cells=124 … mask_cells_all=[1-9][0-9]*"` | The registration names the part-2 line only. `--matrix-only` runs parts 1, 2 **and** 3 (`test-union-four-param-matrix.cpp:2320-2325`). With `PASS_REGULAR_EXPRESSION` set, ctest decides PASS/FAIL from the regex and ignores the exit code. | If part 3 fails (either mask-negative arm), `FOUR_MATRIX ok=1` is already on the log, so `ctest -R '^test-union-four-param-matrix$'` reports PASS while the binary exits 1. The source comment at `:2290-2293` claims "its exit code and its regex can never disagree". They can. The shell gate catches it (`FOUR_MASK_NEG`, `FOUR_MASK_ROW_NEG`, `FOUR_END`, any `FAIL:`); the ctest registration does not. **Live.** | D |
| D2 | `tests/CMakeLists.txt:382` | `PASS_REGULAR_EXPRESSION "BSE_END ok=1"` | The regex names the conjunction line. The exit code is ignored. | `BSE_END ok=%d` is printed at `test-union-binder-stock-equiv.cpp:1272` and the process returns on the next line, so the window is one statement. A crash between the print and the return passes ctest. Narrow. | D |
| D3 | `tests/CMakeLists.txt:1102-1117` | `if (LLAMA_NTOK_TEST_MODEL)` / `if (TARGET test-ntok-onegraph AND TARGET test-cpu-kernel-sel-len-indep)` / `add_test(NAME test-union-mainline-gate …)` | The one aggregate union gate is registered only when a model is configured and both conditional targets exist. | When the guard is false, no gate exists. A ctest run then reports success: absence of a gate is indistinguishable from a green gate. `RESULT.md` §13.6 names this risk. | D |
| D4 | `tests/CMakeLists.txt:562` vs `tests/nkv-cpu-probe/union-mainline-gate-r24o.sh:38-40` | `set(LLAMA_NTOK_TEST_ROUNDS "50" CACHE STRING …)` vs `rounds defaults to 2 when the argument is absent or empty: the CMake registration passes ${LLAMA_NTOK_TEST_ROUNDS}, which is empty unless the caller set it.` | The cache default is `50`, so the script always receives `50` and its `rounds=${6:-2}` default never applies. | Not a false-pass. A stale contract: the script documents a default it never uses, and a reader cannot tell which round count the gate ran. | D |
| D5 | `tests/nkv-cpu-probe/run-triparam-close-13022.sh:125-126` | `for t in test-nkv-runtime-param-hash test-nkv-runtime-param test-triparam-matrix \` | The loop runs `test-triparam-matrix`, which is retired (`tests/CMakeLists.txt:194`). `ctest -R` with `--no-tests=error` fails, so `tests_ok=0` and the script always prints `TRIPARAM_CLOSE ok=0`. | Not a false-pass: it fails closed. The claim it used to carry is now unverified by this script, and the script can never go green. Dead gate. | D |
| D6 | `tests/nkv-cpu-probe/one-shot-cpu-13022.sh:136` | `-R '^(…\|test-triparam-matrix)$'` | Same retired name in the ctest regex. `TRIPARAM_TRUELEN` is then `MISSING`, so `ok=0` always. | Same: fails closed, dead gate. | D |
| D7 | `tests/nkv-cpu-probe/run_ntok_cross_matrix_13022.sh:85` | `--target test-ntok-cross-matrix \` | Builds a target that is no longer declared (`test-ntok-cross-matrix.cpp` exists but has no `add_executable`). | `cmake --build` fails and the script exits 1. Fails closed; the script is dead. | D |
| D8 | `tests/nkv-cpu-probe/README.md:350` | `` `test-triparam-matrix` runs the same binary with `--matrix-only` … so that registration's exit code and its `PASS_REGULAR_EXPRESSION` always agree `` | The README still describes a live registration and asserts the invariant that D1 breaks. | Not a code defect. It is the document that would tell the next reader the invariant holds. It does not. | D |

## 3. Benign — checked and deliberately not changed

These were read and are correct, or are unreachable, or are the right shape already. The reason is given so the next reader does not audit them again.

| site | why no change |
|---|---|
| `src/llama-kv-cache.cpp:2817-2913` `check_mask_row` | Correct tri-state: 1 agree, 0 disagree, -1 refuse. `width == 0` and every unusable layout return -1, never 1. The header (`llama-kv-cache.h:340-353`) documents all three. |
| `src/llama-graph.cpp:409-424` | Maps the three verdicts correctly: `> 0` counts checked+ok, `< 0` counts refused, `== 0` counts checked and logs the disagreement. |
| `src/llama-graph.cpp:435-437` `mask_rows_all` | `rows_expected != 0 && rows_examined == rows_expected`. A zero-row batch reads 0, not 1. |
| `src/llama-context.cpp:2026-2039` `nkv_runtime_stats` | Returns false when no record exists. The mask counters stay 0 when no runtime input exists, and the comment says 0 reads as "nothing was filled". The callers that matter assert a floor. |
| `src/llama-context.cpp:8674` `rebuild_fallbacks` guard | When `llama_get_union_diagnostics` returns false the guard does not fire, but the binder is then off and `union_live_mask_input` returns null, so the function has already returned 0. Unreachable. |
| `src/llama-context.cpp:5469` `mixed_depth` loop | `for (… && recr_chk != nullptr; …)`. If the loop body runs zero times, `mixed_depth` stays false, but `rsz_rt_zero_view_rows` is then 0 and `llama_rsz_rt_check_row` returns `envelope`, so the lane is refused. Fails closed. |
| `src/llama-context.cpp:7238-7250` `llama_rsz_rt_relay_gate` | A pure conjunction of five explicit inputs; the refusal is counted by name. No default. |
| `src/llama-nout-state.h:217-264` | `llama_rsz_rt_row_check` is an enum with `ok = 0`; callers compare `== ok`, never `== 0`. The polarity is explicit in the type. |
| `src/llama-union-binder.h:232-236, 290-302` | `llama_union_rebind_result` is an explicit tri-state (`declined` / `refused` / `applied`). `rebind_any` returns false only when no kind ran, and the caller then counts a rebuild fallback, which the tests require to be 0. |
| `src/llama-union-width-table.h:397-415` | `set_runtime_n_kv(n_kv, arena_n_kv)` refuses 0 and any width above the live arena width the caller supplies (not the build width: the token lane builds narrow and a successful replan widens the arena); `effective_n_kv` never returns 0 for a built table. |
| `ggml/src/ggml-cpu/ggml-cpu.cpp:311-329` | A non-CPU backend reports zeros by design; every consumer of the counters asserts `delta != 0` before using them (`test-nkv-runtime-param.cpp:755`, `test-cpu-attn-pad.cpp:417`). |
| `tests/test-union-four-param-matrix.cpp:1728-1763` | The reference shape: `mask_row_checked == 0` fails, `mask_row_ok != checked` fails, `mask_row_refused != 0` fails, `mask_rows_all != 1` fails, `checked == n_tokens * mask_filled` fails. |
| `tests/test-union-four-param-matrix.cpp:2086, 2115-2123` | The negative child requires 1 on the clean row before it requires 0 on the corrupt row. That precondition is what stops a "nothing to check" 0 from counting as detection. |
| `tests/test-union-four-param-matrix.cpp:1869-1889` | `mask_cells != cells.size()`, `mask_rows_checked == 0`, `mask_rows_ok != checked`, `mask_cells_all != mask_cells` all fail. |
| `tests/test-union-binder-stock-equiv.cpp:685-689, 1205-1216` | `rows != bse_steps` and `rows != bse_neg_steps` floors; `differ == 0` fails the negative control. |
| `tests/test-cpu-kernel-sel-len-indep.cpp:326, 353, 391` | `stock_cases == 0`, `stride_ran == 0`, `old_ran == 0` all fail. |
| `tests/test-cpu-attn-pad.cpp:417-444` | `d_rows != columns` fails; `columns` is 1 or 4. |
| `tests/test-ntok-onegraph.cpp:157` | `pos != total_tokens \|\| checked_rows != total_tokens` fails. |
| `tests/nkv-cpu-probe/check_probe.py:286-316` | `len(nkv_exact) != steps`, `len(mask_rows) != steps`, `row["ok"] != 1`, `row["visible"] < 1` all fail. |
| `tests/nkv-cpu-probe/compare_records.py:50-52` | `if not left or not right: FAIL no records to compare`. |
| `tests/nkv-cpu-probe/nkv-exact-width-report.py:459-461` | `if not statuses: return 3`; `overall.ok` requires `bool(ran)` and no `not_run`. |
| `tests/nkv-cpu-probe/union_ctest_gate.py:73-74, 213-234` | `ORDER` puts `preflight_failed` and `not_registered` first; `run_rc != 0` fails even when every declared test passed. |
| `tests/nkv-cpu-probe/union-mainline-gate-r24o.sh` | Fully fail-closed: missing line is a FAIL, `FAIL:` anywhere is a FAIL, every numeric equality is shell arithmetic with a floor (`mchk >= mcells`, `x >= 1`, `stock_r > 0`). The `|| true` uses are absent. |
| `tests/nkv-cpu-probe/gate-kernel-sel-len-13022.sh:171-190` | `stock_r == 0`, `stride_r == 0`, `stride_d == 0`, `old_r == 0`, `old_d == 0` all fail. The `sed` captures are guarded by the shape grep above them. |

## 4. The `llama_union_mask_row_check` record, in full

Definition, `src/llama-context.cpp:8639-8696`:

```c
int llama_union_mask_row_check(const llama_context * ctx) {
    const llm_graph_input_nkv_runtime * rt = union_live_mask_input(ctx);
    ggml_tensor * mask = union_live_mask_tensor(rt);
    if (mask == nullptr) {
        return 0;
    }
    const llama_ubatch & ubatch = ctx->get_ubatch_inputs_last();
    if (ubatch.n_tokens == 0) {
        return 0;
    }
    const uint32_t width = rt->mask_n_kv;
    if (width == 0 || (int64_t) width > mask->ne[0]) {
        return 0;
    }
    const uint32_t tok = ubatch.n_tokens - 1;

    // A step whose bind was refused and whose graph was then rebuilt has a
    // patch that may have landed on a graph nobody executes (the pad-tail and
    // shape patches are written inside union_rebind_apply, before the record
    // is updated and before the post-bind can_reuse refusal). The rebuilt
    // graph is a different object, so the hazard is a stale RECORD, not a
    // stale node -- but a read-back taken across that boundary would be
    // comparing a row against cells the record no longer describes. Refuse
    // instead of reporting agreement: 0 here means "not checked", and the
    // matrix's per-cell assertion treats 0 as a failure.
    llama_union_diagnostics diag = {};
    if (llama_get_union_diagnostics(ctx, &diag) && diag.rebuild_fallbacks != 0) {
        return 0;
    }

    // The cache is resolved from the context's own memory, which lives as long
    // as the context. The per-decode memory CONTEXT is not consulted: it is
    // freed when decode returns, and this function is called after that.
    llama_kv_cache * kv = union_kv_cache_of(ctx->get_memory());
    if (kv == nullptr) {
        return 0;
    }
    // The same two rules the fill applies: the context's causal flag and the
    // cache's own SWA type, not defaults.
    const bool swa = kv->get_swa_type() != LLAMA_SWA_TYPE_NONE;
    int64_t first_bad = -1;
    // The check's three verdicts map onto this function's two: 1 stays 1
    // (agreed), 0 stays 0 (disagreed), and -1 (refused to model the layout)
    // also returns 0 -- a refusal is never a pass. The distinction is kept in
    // the stats (`mask_row_refused`), which is where a caller can see it.
    const int verdict = kv->check_mask_row(mask, &ubatch, tok, (const float *) mask->data, width,
            ctx->get_cparams().causal_attn, swa, &first_bad);
    return verdict > 0 ? 1 : 0;
}
```

Call site 1, `tests/test-union-four-param-matrix.cpp:2082-2091`:

```c
    // The same public function the arm's verdict comes from, on the CLEAN
    // row, must agree first. The stats above are the fill's own counters; a
    // read-back that returned 0 unconditionally would satisfy the corruption
    // arm below for the wrong reason. Two-sided measurement, one process.
    if (llama_union_mask_row_check(ctx) != 1) {
        std::fprintf(stderr, "FAIL: the read-back refused a clean mask row\n");
        llama_free(ctx);
        live_close(m);
        return 12;
    }
```

Call site 2, `tests/test-union-four-param-matrix.cpp:2113-2131`:

```c
    mask_data[width - 1] = 1.0f;

    const int ok = llama_union_mask_row_check(ctx);
    llama_free(ctx);
    live_close(m);
    if (ok != 0) {
        // The read-back accepted a row the fill never wrote: the arm fails,
        // and the exit code carries it.
        std::fprintf(stderr, "FAIL: the read-back accepted a mask row that disagrees with the cells\n");
        return 11;
    }
    // Exit 0 means the corruption WAS detected. The value line records both
    // verdicts as numbers, so the pair is on the record without reading prose;
    // the sentence below is the phrase the parent greps for, and without it
    // the parent cannot tell detection from a child that died before it
    // checked anything.
    std::fprintf(stderr, "FOUR_MASK_ROW_NEG_CHILD clean=1 corrupt=0\n");
    std::fprintf(stderr, "the mask row disagrees with the KV cells\n");
    return 0;
```

Declaration and contract, `include/llama.h:843-859`:

```c
    // `llama_union_mask_row_check` re-runs the read-back on the LAST row of
    // the last batch (token n_tokens - 1; one row, because that row covers the
    // whole step and because multi-row coverage is the fill's job, reported
    // through the stats above). Every column below the fill width must be
    // exactly 0.0f or -INFINITY and must agree with the visibility the KV
    // cache's cells imply. It returns 1 when the row agrees, 0 when it does
    // not or when there is nothing to check.
    …
    LLAMA_API int llama_union_kq_mask_row  (const struct llama_context * ctx, uint32_t row, float ** data, int64_t * ne0);
    LLAMA_API int llama_union_mask_row_check(const struct llama_context * ctx);
```

Correction to the ticket's premise: this function returns **1** for "rows agree", not 0. The defect is the opposite collapse: 0 means both "disagrees" and "nothing to check", so UNPROVEN is reported as FAIL. No caller reads 0 as PASS. The two call sites require `== 1` first, which is what makes the negative arm sound: the clean-row call proves the state was live before the corrupt-row call.

## 5. Convergence — what the shared refs already contain

Read-only cross-checks. Nothing in them was modified.

| path | state | content on this defect class |
|---|---|---|
| `/home/kunweiz/work/union-mainline-r24o-synth-r24o-20261104-1430/RESULT.md` | present, 38 KB | §9 NOT RUN: no compiler, no cmake, no ctest, no gate executed, no model loaded; every gate line is NOT RUN. §13 residual risks: (2) `BSE_KERNEL_SEL` is a declaration, not a proof; (3) `rebuilds == 1` depends on the token lane and on the first build being the only one; (6) the aggregate gate registers only when both conditional targets exist, so a configuration that skips `if (UNIX)` or the ggml-cpu guard "silently has no union gate". Item 6 is finding D3. Items 2 and 3 are about proof strength, not about 0-as-pass. |
| `/home/kunweiz/work/e5-refs-r24q/r24q-refs/` | present: `p1234-13022-r24j-RESULT.md`, `p1234-10022/{RESULT.md,patch.diff}`, `hip-hostside/{HOSTSPLIT.md,OVERLAP-DESIGN.md,RESULT.md,patches/}` | No audit of the "return 0 / empty set / never executed / UNPROVEN as PASS" class. The word `unproven` appears once (`p1234-13022-r24j-RESULT.md:362`, about `mtp_a_epochs` as an A-result-cache epoch). The patches contain fail-closed predicates, not a site inventory. |
| `/home/kunweiz/work/e5-refs-1225/x/r1225/` | present: `loot/` with 19 project directories, `loot-*/RESULT.md`, bundles | No audit of this class. The closest artefact is `loot-pad-to-max-residual/RESULT.md` §1, a site-audit table of a different class (256-cell padding sites) in the house style `# / site / current shape / class / action`. It cites `AUDIT.md @ 8765b8c`, but no `AUDIT.md` exists anywhere under `loot/` (`find . -name 'AUDIT*.md'` returns nothing). |
| `/home/kunweiz/work/e5-refs-0825/e5-refs-0825/loot-dw-spans-async-hipcc-0723/AUDIT.md` (referenced by `loot-dw-rollup-meter-ab/RESULT.md:310`) | not checked — outside the two paths named in the ticket | — |

Convergence statement: no earlier audit of this defect class exists in the named refs. There is nothing to converge with, and nothing to fork. The house style to match is the `# / site / current shape / class / action` table in `loot-pad-to-max-residual/RESULT.md` §1; this inventory uses the ticket's columns and adds severity.

## 6. What this inventory does not claim

1. No gate was run. Every "live" severity above is a reading of the control flow, not an observation.
2. No compiler was invoked. A finding's line numbers are from HEAD `400f9487` and may move after an edit.
3. The `ggml` kernels outside `ggml-cpu/` were not searched.
4. `test-ntok-cross-matrix.cpp` and the three scripts that name it (D5, D6, D7) are dead. This inventory records them; it does not propose to resurrect the test.
