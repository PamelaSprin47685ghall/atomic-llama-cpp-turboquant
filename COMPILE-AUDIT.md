# COMPILE-AUDIT — static compilability self-audit

Nothing was compiled. Every statement below comes from reading the tree.
Host rules: no compiler, no cmake, no ctest, no build-slot, no gate run.
`bash -n` and a Python AST parse were the only checks executed.

Files changed on the branch: 42 (38 code and build files, 3 audit
documents, 1 README), `git diff --name-only 400f9487..HEAD`. This table
covers the 38 code and build files plus this document (39 rows); the other
two audit documents and the README are documentation-only.

## Per-file table

| file | what was checked | verdict |
|---|---|---|
| `include/llama.h` | `enum llama_gate_verdict` is inside `extern "C"` (line 52) and is C-compatible. `llama_union_kq_mask_row` declares `bool`; the definition at `src/llama-context.cpp:8611` matches in return type, parameter list, parameter names, and `const`. `llama_union_mask_row_check` declares `enum llama_gate_verdict`; the definition at `src/llama-context.cpp:8639` matches. No default arguments on either. No `printf` added. | clean — COMPILE_DEFERRED_13022 |
| `src/llama-context.cpp` | Both definitions match the header (above). The enum is visible through `#include "llama-context.h"` (line 1) → `llama.h` (line 3), and directly at line 28. The `verdict` variable is `const enum llama_gate_verdict`; `return verdict;` matches the return type. No `printf` added or changed. | clean — COMPILE_DEFERRED_13022 |
| `src/llama-context.h` | `llama_ubatch ubatch_inputs_last = {};` at line 611. `llama_ubatch` is an aggregate with a `std::shared_ptr<data_t>` member and no user-provided constructor, so `= {}` value-initialises it: scalars 0, pointers null, shared_ptr default. Valid C++11. | clean — COMPILE_DEFERRED_13022 |
| `src/llama-graph.cpp` | The enum is visible through `#include "llama-graph.h"` (line 4) → `llama-batch.h` (line 4) → `llama.h`, and directly through `#include "llama-batch.h"` (line 8). The `switch (verdict)` at line 418 has one case per enumerator (`PASS`, `FAIL`, `UNPROVEN`) and no `default`; no enumerator is unhandled. The changed `LLAMA_LOG_ERROR` at line 425 uses `%s` for `__func__` and `mask->name`, `%u` for `tok` (`uint32_t`), `%" PRId64 "` for `first_bad` (`int64_t`), `%d` for `runtime` (`int32_t`). The new log at line 380 uses two `%s`. | clean — COMPILE_DEFERRED_13022 |
| `src/llama-kv-cache.cpp` | `llama_kv_cache::check_mask_row` at line 2817 and `llama_kv_cache_context::check_mask_row` at line 4142 match their declarations in `llama-kv-cache.h` in return type, parameter list, names, and trailing `const`. Every `return` yields an enumerator. No `printf` added. | clean — COMPILE_DEFERRED_13022 |
| `src/llama-kv-cache.h` | Both declarations return `enum llama_gate_verdict`. The enum is visible through `#include "llama-batch.h"` (line 3) → `llama.h` (line 3). Parameter lists match the definitions (checked side by side). No default arguments. | clean — COMPILE_DEFERRED_13022 |
| `src/llama-union-topology.h` | `hash_nodes` keeps its signature and adds `if (nodes.empty()) return false;`. `std::vector::empty()` is available through `<vector>` (line 63). No enum, no `printf`. Every caller handles `false` (checked in phase 2). | clean — COMPILE_DEFERRED_13022 |
| `tests/CMakeLists.txt` | `add_test` names are unique (no duplicates from `grep -oE "add_test\(NAME …" | sort | uniq -d`). Every `$<TARGET_FILE:…>` in the union gate block names a target declared earlier: `test-union-four-param-matrix` 222 < 1116, `test-union-binder-stock-equiv` 381 < 1117, `test-ntok-onegraph` 568 < 579, `test-cpu-kernel-sel-len-indep` 1081 < 1119. `if (TARGET …)` at 1113 is after both declarations. `FAIL_REGULAR_EXPRESSION` at 235 and 391 are semicolon lists. The selftest is inside `if (UNIX)` at 1138, has `LABELS "cpu;union"` and `TIMEOUT 300`, and has no `PASS_REGULAR_EXPRESSION`. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/gate-verdict.sh` | `bash -n` OK. No shebang, no `set -e`. `gate_finish` returns 0/1/2, not the enum value. `case` patterns use quoted variable expansions, which are literal in bash. No variable is read before `gate_init` on any documented path. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/union-mainline-gate-r24o.sh` | `bash -n` OK. The library path is `$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/gate-verdict.sh`, so it resolves when sourced. `gate_init` runs at source time. `run_verdict`/`run_reason` are initialised before the `if`. `agg_verdict`/`agg_reason` are initialised at top level. `cleanup` reads `${gate_verdict:-…}`. No `set -e`. `bash -n` OK. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/union-gate-verdict-selftest.sh` | `bash -n` OK. Sources both files from `here`, computed from `BASH_SOURCE[0]`. `tmp` is set before the trap. `check_case` locals are assigned. The six `want_re` strings are EREs with no backreferences. Not executed. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh` | `bash -n` OK. The library path uses `BASH_SOURCE[0]`. Six verdicts initialised to `$GATE_UNPROVEN`. Each block calls `gate_init`, then `gate_pass` under `[[ -z $gate_reason ]]`, then assigns its verdict. The final `case $overall` maps 1→0, 0→1, 2→2. No `set -e`. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/gate-p1-merge-13022.sh` | `bash -n` OK. Same library path and init pattern for five gates. `plan_check`/`exact_check` are assigned from awk; if awk prints nothing, the `%`/`#` expansions yield empty and `[[ -eq 0 ]]` treats empty as 0 (fail-closed). The final `case $overall` maps 1→0, 0→1, 2→2. No `set -e`. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/gate-nkv-observe-13022.sh` | `bash -n` OK. The library path uses `BASH_SOURCE[0]`. `result[$name]` starts at `UNPROVEN`; every failure path sets `FAIL` and continues; the two success paths promote only from `UNPROVEN`. `result_name` maps the three names. Exit 0/1/2. No `set -e`. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/gate-p1-ctest-pack-13022.sh` | `bash -n` OK. Same result-array pattern. `require_line` returns early unless the result is `PASS`. The summary prints `pass|fail|unproven`. Exit 0/1/2. No `set -e`. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/run_pad_lane_ab_gate_13022.sh` | `bash -n` OK. `set -euo pipefail`; the new missing-JSON block ends in `exit 2`, so no non-zero command is left unguarded. `printf` specifiers: `%s` path, `%d` rc. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/run_union_varlen_e2e_13022.sh` | `bash -n` OK. Same shape and same `printf` specifiers as the pad-lane runner. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/run_probe.sh` | `bash -n` OK. `stats_unset`/`stats_on` are assigned before the emptiness test. `sum_rc` is assigned before the `if`. `overall` starts at 0 and is only raised. No `set -e`. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/run_varlen_output_probe.sh` | `bash -n` OK. `status` starts at 0; the probe-failure path sets 1 and continues; the empty-file path sets 2 only from 0. `printf` specifiers: `%s` mode, `%d` rc, `%d` cmp status. No `set -e`. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/summarize.py` | `python3 -c 'import ast; ast.parse(...)'` OK. `logs_dir` and `rows` are in scope at the new `if not rows`. `return 2` is an int. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/union_ctest_gate.py` | AST parse OK. The new f-strings read `report['verdict']` and `entry['name'/'registered'/'status'/'wall_time_s']`, all present in the `summarize()` return. `return report["exit_code"]` unchanged. | clean — COMPILE_DEFERRED_13022 |
| `tests/nkv-cpu-probe/pad_lane_ab_gate.cpp` | The new `const char * gate_verdict` is declared before the `fprintf`. The `GATE %s …` argument list was re-counted: `%s` gate_verdict, `%s` pass/fail, `%d` `bound_matching` (int), `%d` steps (int), `%s`, `%d`, `%d`, `%d`, `%d` (bools as `? 1 : 0`), `%d` `equal_rows` (int), `%d` steps, `%d` `ok ? 1 : 0`. `<cstdio>` is included. | fixed (fix 1) |
| `tests/nkv-cpu-probe/union_varlen_e2e.cpp` | The new `const char * gate_verdict` is declared before the `fprintf`. The `GATE %s …` list was re-counted: `%s` verdict, `%d` `executed` (int), `%d` steps (int), `%d` `equal_steps` (int), `%d` `executed`, `%llu` `on_builds` (`uint64_t`, cast to `unsigned long long`), `%llu` `off_builds` (same), `%d` bools as `? 1 : 0`, `%d` `tok_steps_matching` (int), `%d` `tok_steps_checked` (int), `%d` `tok_moved ? 1 : 0`, `%s` `multi_status`. `<cstdio>` is included. | fixed (fix 1) |
| `tests/test-flash-attn-bias.cpp` | `error_stats` is now called only when both outputs are non-empty and the same size; it reads `ref[i]` for every `i` in `got`, so the size check is required. `max_abs`/`mean_abs`/`max_rel`/`mean_rel`/`rmse` are initialised to 0.0. `std::isfinite` needs `<cmath>`, included at line 9. The `printf` list was re-counted: `%s` `c.name`, `%lld` `(long long)c.nq`, `%zu` `flash.allocated_bytes` (`size_t`), `%zu` `dense.allocated_bytes`, `%.6f` ratio (double), `%.9g` max_abs, `%.9g` mean_abs, `%s` verdict. | fixed (fix 2) |
| `tests/test-gdn-ingredient-replay.cpp` | `have_output` takes `const std::vector<float> &` and returns `bool`; `<vector>` and `<cstdio>` are included. All seven `run_gdn` calls are followed by `if (!have_output(...)) return 1;` before any slice or comparison. The helper prints one `FAIL` line with no format specifiers. | fixed (phase 4) |
| `tests/test-nkv-exact-width.cpp` | `footer_ok` is declared at line 1632, before the `if (g_omit_root >= 0)` block, so it is in scope at the final `return footer_ok ? 0 : 1;` (line 1692). `g_cases_run`/`g_steps_run` are `int` (lines 243-244), so `%d` is correct in the new `FAIL` line and in `NKV_EXACT_END`. `<cstdio>` is included. | fixed (C1) |
| `tests/test-nkv-runtime-param.cpp` | The new floor lines use `%d` for `step` (int) and `%u` for `mask_filled`/`mask_row_ok`/`mask_row_checked`/`mask_row_refused`/`mask_rows_all` (`uint32_t` in `include/llama.h:781-801`). The part-4 line uses `%u` for the four `uint32_t` fields. `<cstdio>` and `<cstdint>` are included. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-union-binder-stock-equiv.cpp` | The new floor lines use `%d` for `step` (int) and `%u` for the `uint32_t` stats fields. The multi arm repeats the same types. `<cstdio>` and `<cstdint>` are included. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-union-four-param-matrix.cpp` | `enum llama_gate_verdict` is visible through `#include "llama.h"` (line 169). The four `const enum llama_gate_verdict` locals are declared before use. Every enum print uses `(int)` or `int(...)`: `%d` for `(int) ok`, `(int) before`, `(int) clean`, `(int) corrupt`, `(int) LLAMA_GATE_*`, `int(verdicts_ok)`. The new child's `rc` is `int` (`%d`), `named ? "present" : "MISSING"` is `const char *` (`%s`). `<cstdio>`, `<cstdint>`, `<vector>` are included. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-rsz-lane-rows.cpp` | `void check_arm(bool, bool, uint32_t, int32_t, bool)` at 35 and `void check_row(int32_t, uint32_t, llama_rsz_rt_row_check)` at 48 each increment `g_checks_run` once (2 sites). Every `++g_fail` (44, 55) is inside one of them. Floor `if` at 110; FAIL print 115; PASS print 118, text unchanged. `#if`/`#endif` 0/0; `<cstdio>` included. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-union-binder-gate.cpp` | `void check(bool, const char *, const char *, bool)` at 25 increments `g_checks_run` once. The only `++g_fail` (30) is inside it. Floor 93; FAIL 98; PASS 101, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-rsz-selection-write.cpp` | `#define CHECK(cond, ...)` at 85 increments `g_checks_run` once (line 86). The only `++g_fail` (91) is inside the macro. Floor 469; FAIL 474; PASS 477, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-rsz-hybrid-selection-write.cpp` | `#define CHECK(cond, ...)` at 92 increments once (line 93). The only `++g_fail` (98) is inside the macro. Floor 621; FAIL 626; PASS 629, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-rs-ring-relay.cpp` | `#define CHECK(cond, ...)` at 136 increments once (line 137). `++g_fail` at 142 is inside the macro. `++g_fail` at 844 is in `main`, on the sweep's `if (!f.ok())` fixture failure; it is not a check and cannot cause a false floor, because the earlier sections and the sweep run CHECKs. Floor 940; FAIL 945; PASS 948, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-rsz-key-injection.cpp` | `#define CHECK(cond, ...)` at 79 increments once (line 80). The only `++g_fail` (85) is inside the macro. Floor 589; FAIL 594; PASS 597, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-rsz-relay-onegraph.cpp` | `#define CHECK(cond, ...)` at 103 increments once (line 104). The only `++g_fail` (109) is inside the macro. Floor 454; FAIL 459; PASS 462, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-union-binder-rsz.cpp` | `#define CHECK(cond, ...)` at 53 (increment 54), `static bool compare_logits(...)` at 316 (increment 318), `static bool check_logits_nonzero(...)` at 352 (increment 354), `static bool compare_states(...)` at 392 (increment 394): 4 sites. Every `++g_fail` (59, 324, 344, 357, 368, 386, 400, 406, 416, 425) is inside one of them. Floor 832; FAIL 837; PASS 840, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `tests/test-rsz-multiseq-decode.cpp` | `#define CHECK(cond, ...)` at 50 (increment 51), `static bool logits_match(...)` at 139 (increment 142), `static bool logits_differ(...)` at 179 (increment 182), `static bool compare_all_states(...)` at 201 (increment 203), `static bool compare_seq_state(...)` at 241 (increment 244): 5 sites. Every `++g_fail` (56, 148, 172, 188, 196, 207, 212, 222, 231, 249, 257, 262, 272, 280, 294) is inside one of them. Floor 622; FAIL 627; PASS 630, text unchanged. 0/0; `<cstdio>`. | clean — COMPILE_DEFERRED_13022 |
| `COMPILE-AUDIT.md` | Documentation only. No code, no build input, no compile surface. | clean — COMPILE_DEFERRED_13022 |

## Shell syntax results

| file | `bash -n` |
|---|---|
| `gate-verdict.sh` | OK |
| `union-mainline-gate-r24o.sh` | OK |
| `union-gate-verdict-selftest.sh` | OK |
| `gate-nkv-runtime-13022.sh` | OK |
| `gate-p1-merge-13022.sh` | OK |
| `gate-nkv-observe-13022.sh` | OK |
| `gate-p1-ctest-pack-13022.sh` | OK |
| `run_pad_lane_ab_gate_13022.sh` | OK |
| `run_union_varlen_e2e_13022.sh` | OK |
| `run_probe.sh` | OK |
| `run_varlen_output_probe.sh` | OK |

## Python parse results

| file | parse |
|---|---|
| `summarize.py` | OK |
| `union_ctest_gate.py` | OK |

## What could still break on 13022

1. **The selftest's expected reason strings** (`tests/nkv-cpu-probe/union-gate-verdict-selftest.sh:113-136`).
   Cases 2-6 assert exact `reason=` slugs. If `check_four_matrix` reports a
   different first failure than traced (for example `matrix_arms_same`
   before `matrix_stock_rebuilds`), the case fails. This is the intended
   falsifiability, but it is the most likely red on the first run. The
   check order in `check_four_matrix` is the dependency.
2. **`std::isfinite` and `%zu` on a non-glibc toolchain**
   (`tests/test-flash-attn-bias.cpp:479` and `:484`). `std::isfinite` is
   C++11 and `<cmath>` is included, but a toolchain that defines
   `isfinite` as a macro can reject the `std::` qualification. `%zu` for
   `size_t` is C99/C++11; a pre-C99 MSVC would reject it. The 13022 host
   is Linux, so the risk is low.
3. **The `switch` with no `default`** (`src/llama-graph.cpp:418`). All
   three enumerators are covered today. If a fourth value is added to
   `enum llama_gate_verdict` without a case, a build with `-Wswitch` and
   `-Werror` (LLAMA_FATAL_WARNINGS=ON) fails here. That is the intended
   signal, but it is a compile break, not a test failure.

## Checks that could not be established

None. Every changed file is covered by at least one static check above.
The two facts that only a compiler can settle are: (a) the exact warning
set of the 13022 build (`-Wswitch-enum`, `-Werror`), and (b) that the
`llama_ubatch` aggregate stays value-initialisable if a member gains a
user-provided constructor. Both are stated as risks, not as verified.
