# EXIT-CODE-AUDIT — printed verdict vs process exit status

Bounded audit of the class in C1: a run prints a verdict that the process
exit status does not carry.

## Scope and method

Searched, in full:

| tree | files |
|---|---|
| `tests/*.cpp` | 95 |
| `tests/nkv-cpu-probe/*.cpp` | 5 |
| `tests/nkv-cpu-probe/*.sh` | 20 |
| `tests/nkv-cpu-probe/*.py` | 8 |

Total: 128 files.

Method, per file:

1. every line that prints a verdict was extracted: `ok=%d`, `ok=1`, `ok=0`,
   `PASS`, `FAIL`, `HARNESS_END`, `PROBE_RC`, `*_END ok=`, `GATE_*`,
   `OVERALL`, `RUN_RC`, `DRIVER_RC`;
2. the expression the process returns or exits with was read;
3. a site is flagged when a run can print a non-passing verdict and exit 0,
   or print a passing verdict and exit non-zero.

Counts: 525 verdict print lines in 40 C++ files; 24 summary sites in the 20
shell scripts and 8 Python files. Every summary line and every `FAIL` print
with no nearby failure action was read in full. The remaining per-case
diagnostics were classified by the extraction: they either set a failure
flag, increment a failure counter, or return a value the caller checks.

## Sites where the printed verdict and the exit status disagree

| file:line | printed verdict | returned expression | verdict |
|---|---|---|---|
| `tests/test-nkv-exact-width.cpp:1637`, `:1692` | `NKV_EXACT_END ok=0` for a zero-work run (`footer_ok` false) | `return all_ok ? 0 : 1;` — `all_ok` was still true, so exit 0 | **fixed**: `return footer_ok ? 0 : 1;`. The negative-omission path returns before this line (three `return 1` and one `return 0`), so it is unaffected. |
| `tests/test-gdn-ingredient-replay.cpp:97` | `FAIL: gated_delta_net graph compute failed` | `return {};` from `run_gdn`. Every caller sliced the empty vector (`out.begin() + offset`) and compared unrelated memory, so the run could print FAIL and exit 0. | **fixed**: `have_output()` after each of the 7 calls returns 1 on an empty result. The comparison is no longer reached. |
| `tests/nkv-cpu-probe/pad_lane_ab_gate.cpp:688`, `:703` | `GATE PASS` when `ok` is true | `return ok && wrote ? 0 : 1;` — exit 1 when the JSON write failed | **fixed** (7980453): the verdict is `PASS` only when `ok && wrote`, `FAIL reason=json_write` when the write failed, `FAIL` otherwise. The exit expression is unchanged. |
| `tests/nkv-cpu-probe/union_varlen_e2e.cpp:1258`, `:1270` | `GATE PASS` when `ok` is true | `return ok && wrote ? 0 : 1;` — exit 1 when the JSON write failed | **fixed** (7980453): same expression and same unchanged exit. |
| `tests/test-flash-attn-bias.cpp:469`, `:473` | `PASS` printed unconditionally in the `memory` suite | `return 0;` | **fixed** (7980453): `FAIL` when an output is empty or mismatched or a reported statistic is not finite, `UNPROVEN reason=no_tolerance` otherwise, exit 1 only for the FAIL case. `error_stats()` now runs only when the two outputs are non-empty and the same size. |
| `tests/test-laya-backend-precision.cpp:249`, `:296`, `:297` | per-case `FAIL` for a device that is not gated | `return failed ? 1 : 0;` — `failed` is set only for a gated device, so exit 0 | **reported, not fixed**: the run verdict is the summary line, which agrees with the exit. The device line says `- reference only, not gated`, so the per-case `FAIL` is a diagnostic, not a run verdict. Making it fail would change the gating policy. |
| `tests/test-moe-cache.cpp:3407` | `cache-backend-unload: OK` printed unconditionally | `return ok ? 0 : 1;` | **reported, not fixed**: a step status, not the run verdict. The run verdict is the exit code, which carries `ok`. The line does not claim the run passed. |
| `tests/nkv-cpu-probe/union_ctest_gate.py:282` | `GATE FAIL test=<name>` printed for every test when the aggregate failed, including tests that passed | `return report["exit_code"]` — 1 for the aggregate fail | **fixed** (7980453): one `GATE <VERDICT>` line, then one `TEST <name> registered=… status=… wall=…` line per test. The JSON report and the return value are unchanged. |

## Summary sites checked and consistent

Grouped by shape. Each line prints a verdict and the process returns the same
expression.

| shape | sites |
|---|---|
| `*_END ok=<n>` and `return ok ? 0 : 1` | `tests/test-cpu-kernel-sel-len-indep.cpp:397`, `tests/test-nkv-runtime-param.cpp:1167`, `:1210`, `tests/test-ntok-cross-matrix.cpp:1170`, `:1207`, `tests/test-union-binder-stock-equiv.cpp:1334`, `tests/test-union-four-param-matrix.cpp:2468`, `tests/nkv-cpu-probe/nout_runtime_probe.cpp:528`, `tests/nkv-cpu-probe/padmax_cpu_probe.cpp:640`, `tests/nkv-cpu-probe/varlen_output_probe.cpp:385` |
| `g_fail` counter, `FAIL %d` then `return 1`, `PASS` then `return 0` | `tests/test-rs-ring-relay.cpp:939`, `tests/test-rsz-hybrid-selection-write.cpp:620`, `tests/test-rsz-key-injection.cpp:588`, `tests/test-rsz-lane-rows.cpp:108`, `tests/test-rsz-multiseq-decode.cpp:617`, `tests/test-rsz-relay-onegraph.cpp:453`, `tests/test-rsz-selection-write.cpp:468`, `tests/test-union-binder-gate.cpp:92`, `tests/test-union-binder-rsz.cpp:828`, `tests/test-topology-cache-key.cpp:365`, `tests/test-topology-hash.cpp:664`, `tests/test-topology-key-sweep.cpp:286` |
| `ok ? "PASS" : "FAIL"` and `return ok ? 0 : 1` | `tests/test-rsz-runtime-param.cpp:757`, `tests/test-rsz-hybrid-compose.cpp:704`, `tests/test-rsz-union-combo.cpp:545`, `tests/test-ntok-onegraph.cpp:220`, `tests/test-flash-attn-bias.cpp:536`, `tests/nkv-cpu-probe/pad_lane_ab_gate.cpp:688` (the `ok` half) |
| `rc` accumulated with `rc \|= ok ? 0 : 1`, final `OK/FAIL` from `rc` | `tests/test-laya-backend-parity.cpp:300`, `:325`, `:336` |
| `n_fail` / `total_fail` / `failures` counter | `tests/test-laya-convert.cpp:701`, `tests/test-quant-type-selection.cpp:472`, `tests/test-moe-cache-fit.cpp:10`, `:166` |
| `ok &= run_*` composition, per-step `OK/FAIL` from the same `ok` | `tests/test-moe-cache.cpp` (21 sites: `:270`, `:514`, `:706`, `:1034`, `:1157`, `:1615`, `:1802`, `:1856`, `:1959`, `:2235`, `:2304`, `:2369`, `:2438`, `:2478`, `:2560`, `:2606`, `:2768`, `:2864`, `:2925`, `:3032`, `:3434`) |
| `PASSED` after `GGML_ASSERT`-guarded bodies | `tests/test-alloc.cpp:590` |
| `n_ok == device count`, per-backend status from the same `ok` | `tests/test-backend-ops.cpp:11873`, `:11886` |
| shell: `gate_finish` prints and returns the same verdict | `tests/nkv-cpu-probe/gate-verdict.sh:93`, `tests/nkv-cpu-probe/union-mainline-gate-r24o.sh:700`, `tests/nkv-cpu-probe/union-gate-verdict-selftest.sh:84` |
| shell: `GATE_SUMMARY` and the exit computed from the same values | `gate-nkv-runtime-13022.sh:716`, `gate-p1-merge-13022.sh:465`, `gate-nkv-observe-13022.sh:332`, `gate-p1-ctest-pack-13022.sh:296`, `gate-kernel-sel-len-13022.sh:206` |
| shell: `OVERALL` / `RUN_RC` / `DRIVER_RC` / `GATE_RC` and the same exit | `run-nkv-exact-width-arms-13022.sh:388`, `run_probe.sh:173`, `run_varlen_output_probe.sh:40`, `run_pad_lane_ab_gate_13022.sh:108`, `run_union_varlen_e2e_13022.sh:119`, `run_union_ctest_gate_13022.sh:208`, `one-shot-cpu-13022.sh:218`, `run-triparam-close-13022.sh:71` |
| Python: `PASS`/`FAIL` and the same return | `check_probe.py:338`, `:407`, `compare_records.py:51`, `:77`, `nkv-exact-width-report.py:495` |

## Notes

1. **Exit 3.** `gate-kernel-sel-len-13022.sh` and
   `run-nkv-exact-width-arms-13022.sh` (with `nkv-exact-width-report.py`)
   still use exit 3 for NOT_RUN. In both, the printed verdict and the exit
   agree. They are not consumers of the two scripts whose status changed in
   C4, and no consumer of those two reads 3. They are left unchanged; a
   later pass can move them to the shared exit 2.
2. **Vacuous PASS lines.** `tests/test-moe-cache.cpp:3407` prints a
   passing word without a check behind it. It is not a print/exit
   mismatch. `tests/test-flash-attn-bias.cpp:469` had the same shape and
   is fixed (7980453).
3. **Not inspected in depth.** The 525 verdict print lines include per-case
   diagnostics inside loops. They were classified by the extraction and by
   reading every `FAIL` print with no nearby failure action. A diagnostic
   that sets no flag and is not a summary line would not change the exit;
   none was found beyond the gdn case.
