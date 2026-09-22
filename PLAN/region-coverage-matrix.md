# TP5 TM3 Region Coverage Matrix

## Overview
Based on evaluation of `src/llama-context.cpp:2409-2475` (`llama_context::evaluate_target_capacity_admission`), `docs/TP5-TARGET-CAPACITY-PLAN.md` §2 & §7, and unit tests in `tests/test-target-capacity.cpp`.

## Admission Function Analysis (`src/llama-context.cpp:2409-2475`)
- Target capacity admission criteria:
  - `!target_enabled`: fail-closed, exact token rows returned (`entered = false`).
  - `hparams.ple_n_heads > 0` (PLE): rejected (`"model with PLE"`).
  - `ubatch.n_seqs != 1 || ubatch.n_seqs_unq != 1` (multi-sequence): rejected (`"multi-sequence"`).
  - `hparams.dsv4_compress_ratios[i] > 0` (QSA): rejected (`"model with QSA"`).
  - `hparams.n_embd_r() > 0 || hparams.n_embd_s() > 0 || hparams.is_recr(il)` (GDN/recurrent): rejected (`"model with GDN/recurrent layers"`).
  - Otherwise (admissible single-seq dense transformer): `entered = true`, `capacity_rows = verify_tokens`, `capacity_outputs = verify_tokens`.

## Coverage Matrix (Regions 0 - 5)

| Region | Region Name | Claimed | Verified | Actual Dispatch Evidence | Gap |
|---|---|---|---|---|---|
| **0** | **Input / Position** | Input shell capacity-sized; active prefix written; inactive suffix zeroed; bounds check throws on overflow (`active > capacity`). Dynamic reuse matches capacity. | **CPU Verified** (`test_target_out_ids_capacity`, `test_target_input_shell_bounds`, `test_target_reuse_key_dynamic`, `test_target_switch_off_status_quo`, `test_target_embd_h_bounds`) | CPU test execution in `tests/test-target-capacity.cpp` passing 100%. `llama-context.cpp:2409-2475` fail-closed rules validated in `test_target_fail_closed_contract`. | Device-side kernel execution for Region 0 covered in Vulkan command replay and mesh suites, but no single end-to-end full GPU session run for TARGET capacity mode. |
| **1** | **Attention (Dense / FA / QSA)** | Dense transformer single-seq capacity shapes expand to `capacity_rows` (`llama_ubatch_expand_capacity`). KQ mask & KV cache tail sanitization isolates inactive slots. QSA fail-closed rejected. | **CPU Verified (Contract & KV Tail)** (`test_target_attn_kv_capacity_shapes`, `test_target_kv_tail_safety_contract`, `test_target_fail_closed_contract`); **GPU Verified for Flash Attn tail** (`test-vulkan-flash-attn-capacity.cpp`) | `llama_ubatch_expand_capacity` verified. `test-vulkan-flash-attn-capacity.cpp` lines 238-319 prove tail poison (-999 vs +888) yields bit-identical outputs on GPU. | Real model multi-head attention graph execution in full context pipeline under dynamic active rows still requires end-to-end GPU integration. |
| **2** | **GDN / Recurrent State** | GDN/recurrent models are fail-closed rejected in production admission (`llama_context::evaluate_target_capacity_admission`). Scheme B contract and convolution tail extraction isolate inactive rows. | **CPU Verified** (`test_target_fail_closed_contract`, `test_target_gdn_conv_state_tail_safety`, `test_target_gdn_scheme_b_contract`) | Real forward `ggml_graph_compute` on CPU in `test_target_gdn_scheme_b_contract` confirms inactive garbage rows have 0 effect on output state. Admission fail-closed verified. | In-flight admission blocks GDN in TARGET capacity mode until Stage 2 trunk lowering is ready. |
| **3** | **MoE Routing & Gating** | MoE routing tensors dimensioned to `capacity_rows`. Active token router outputs bit-identical regardless of inactive row garbage. | **CPU Verified** (`test_target_moe_contract`) | `test_target_moe_contract` verifies forward execution with -999.0f vs +888.0f inactive row inputs produces bit-identical active logits. | GPU kernel execution for fused MoE under variable active rows not yet exercised in TARGET phase. |
| **4** | **Terminal Producer (Result Output)** | Result output reaches capacity elements (`n_vocab * capacity_outputs`). Inactive output rows are zeroed/sanitized, preventing poison propagation. Active rows produce unpolluted results. | **CPU Verified** (`test_target_terminal_producer_contract`, `test_target_poison_inactive_output`) | `test_target_poison_inactive_output` tests dense transformer with active=2, capacity=4, output buffer pre-poisoned with sentinel; asserts active rows produce valid outputs and inactive rows [2, 4) are strictly zeroed. | Direct Vulkan LateBind dispatch and multi-card RESULT handoff relies on mesh collective tests. |
| **5** | **RELAY / Consumer** | RELAY direct payload elements calculated by `active_rows * width`. Inactive rows never transmitted across mesh. Legacy P1 fallback disallowed when `active < capacity`. | **GPU Verified in Mesh Suite** (`test-vulkan-tp5-mesh.cpp`, `test-tp5-plan`) | `test-vulkan-tp5-mesh.cpp` verifies active rows [0, 2) valid, inactive rows [2, 4) poisoned, with active element transmission across 5 cards. | Full TARGET end-to-end multi-rank generation loop with dynamic candidate verification tokens. |

## Poison & Rollback Regression Analysis
- **Existing Poison Injections in Repo**:
  - `tests/test-vulkan-flash-attn-capacity.cpp`: Poison values `-999.0f` vs `+888.0f` injected into inactive KV tail tokens `[n_past + active, n_past + capacity)` to prove bit-identical active-query GPU outputs.
  - `tests/test-vulkan-tp5-mesh.cpp`: Active rows `[0, 2)` populated with valid numbers, inactive rows `[2, 4)` injected with poison; verifies collective communication only transmits active elements.
  - `tests/test-target-capacity.cpp`: Inactive inputs injected with garbage `-999.0f` vs `+888.0f` in GDN Scheme B and MoE contract tests.
- **Added Regression Test**:
  - Added `test_target_poison_inactive_output` to `tests/test-target-capacity.cpp`.
  - Simplest admissible configuration: Single sequence, dense transformer, `capacity_rows = 4`, `active_rows = 2`.
  - Validates: Pre-poisons output memory (`-77777.0f`) and input tail (`-99999.0f`), executes forward projection, verifies active rows are unpolluted and inactive rows `[2, 4)` are zeroed.
  - Compiles and passes with `cmake --build build-tp5 --target test-target-capacity`.
