# reader-audit — unconditional reader refusal in the union-binder rebind paths

Workspace: `/home/kunweiz/work/lead-fa-lower-20261030` (isolated; cumulative
baseline = stock `f1c5faf` + `lead-fa-boundary-20261030/patch.diff` + the CPU
worker's `fix-cpu-nkv-failclosed.patch`). No compilation, tests, or GPU were
run in this session; every claim below is static source-level evidence with
file/line pointers, not executed verification.

## 1. What changed (this workspace, `src/llama-context.cpp` only)

1. `union_rebind_n_kv` — the attention-view reader refusal that the baseline
   kept inside `if (st.cov_enabled)` is lifted out and extended with a
   graph-side detector (post-edit block at lines ~2910-2932).
2. `union_rebind_outputs` — the n_outputs rebind had NO reader check at all in
   the baseline; it gains the same unconditional refusal (block at lines
   ~3126-3147).
3. New file-local helper `llama_graph_result_has_attn_readers` (before
   `union_rebind_n_kv`): walks `res->inputs` and reports reader-carrying
   attention inputs from the LIVE graph's captured state, so a graph built
   under attention views stays refused even after the views are cleared from
   the cache.

Each refusal is fail-closed: refuse → `++c_fallbacks` → caller full-rebuilds
at the current frontier (the fresh build re-resolves the reader spans at the
current cell state). Counter gating is preserved exactly: `c_refused_view`
stays r8 cov-family (never written with the cov gate off); `c_fallbacks` (r7
field) accounts every refusal unconditionally.

## 2. Detector coverage (all input variants, evidence-based)

`llama_graph_result_has_attn_readers` walks `llm_graph_result::inputs`
(`src/llama-graph.h:970`) and checks exactly two shapes:

- `llm_graph_input_attn_kv` — the only input class with a `readers` member
  (`src/llama-graph.h:381`; `reader_span_input` at 356-365). Populated only by
  `build_attn_inp_kv_impl` (`src/llama-graph.cpp:3066-3084`) when the cache
  has attention views at build time.
- `llm_graph_input_mem_hybrid` — embeds that same attn_kv input
  (`src/llama-graph.h:692`); checked through its public `inp_attn` member.

Every other input class in `src/llama-graph.h` (embd, embd_h, pos,
pos_bucket, pos_bucket_kv, out_ids, mean, cls, rs, cross_embd, attn_no_cache,
attn_k, attn_k_dsa, attn_kv_iswa, attn_k_iswa, dsv4, sampling) carries no
reader state — header-verified: `readers` appears only inside
`llm_graph_input_attn_kv` (line 381). `llm_graph_input_attn_kv_msa` (line
479) derives from attn_kv, so the base-class cast catches it; its builder
(`src/llama-graph.cpp:3857-3883`) never populates readers, and its
`set_input` runs the base `set_reader_input()` (no-op over an empty vector).

Reader-graph dispatch chain: a reader graph always FAILS the stock reuse path
(`llm_graph_input_attn_kv::can_reuse`, `src/llama-graph.cpp:644` — refuses
when readers are present OR the cache has attention views) → the dispatch
falls to the binder rebind → the detector above refuses the in-place patch →
caller full-rebuilds at the current frontier. Plain REUSE of a reader graph
is unreachable by construction.

## 3. Exact variable-width gate semantics (static source reading; nothing executed)

- **Mask width** (`build_attn_inp_kq_mask`, `src/llama-graph.cpp:101-129`):
  narrowed to the request's true length only when the caller passes
  `n_kv_read > 0` — that happens only in the no-reader branch of
  `build_attn_inp_kv_impl` (`src/llama-graph.cpp:3091-3093`) under
  `cparams.flash_attn && inp->fa_true_len && llama_graph_fa_nkv_param_enabled()`;
  the builder itself accepts only `(0, get_n_kv()]` (`:116-117`), anything
  else falls back to the stock cache width. The reader branch (views present)
  builds NO mask at all (`:3085-3095`).
- **K/V read views** (`build_attn`, `src/llama-graph.cpp:3291-3292`): narrowed
  to `self_nkv_req` under the same combined condition PLUS `readers.empty()`;
  otherwise stock cache width.
- **Op-param lane** (`GGML_TP5_FA_NKV_PARAM` exact "1",
  `llama_graph_fa_nkv_param_enabled`, `src/llama-graph.cpp:58-64`): publishes
  the per-request runtime KV length into every plain-FA node's op params slot
  5 (`:2843-2847`). OFF (unset or any other value) leaves slot 5 untouched
  (stock byte pattern).
- **RTLEN tensor lane** (`GGML_TP5_FA_RTLEN` exact "1", aliased by
  `GGML_TP5_FA_CHECKED_TAIL`, `src/llama-graph.cpp:43-53`): independent gate;
  creates the i32[1] `self_nkv_eff` input (`:3101-3105`), attached to each FA
  node's `src[5]` (`:3318-3320`). CPU consumption is a DIRECT validated use:
  the value is asserted in `[0, k->ne[1]]` and then used as-is — no clamp
  (`src/ggml/src/ggml-cpu/ops.cpp:8572-8577`); any out-of-range value aborts.
  CUDA-side, the launcher re-validates the raw op param and aborts on
  `> K->ne[1]` or `< -1` (`fattn.cu:1842-1851`).
- **Reader refusal** (this fix): independent of all FA gates — it fires on any
  binder rebind of a reader-carrying graph, keyed on the live graph's
  captured input state PLUS the cache's current attention-view state.

## 4. 13022 validation plan (suggested, unexecuted)

1. **Refusal fires**: attention views set + frontier move (seq cancel or
   shift) with `GGML_TP5_NKV_TRUEWIDTH=1`, `GGML_TP5_NKV_COV=1` → expect
   `NKV_BINDER_STATS ... refused_view >= 1`, fresh build per move, logits
   matching a non-binder run.
2. **Views cleared after build**: build a reader graph, `clear_attention_views`,
   force a frontier move + rebind attempt → expect refusal (graph-side
   detector) and a full rebuild; no stale captured span may reach compute.
3. **Cov-off accounting delta**: same script with `GGML_TP5_NKV_COV` unset →
   `c_fallbacks` +1 per reader-graph rebind attempt, `c_refused_view` stays 0
   (gated), logits still correct.
4. **OFF ≡ stock (static + runtime)**: all binder gates off → no
   `NKV_BINDER_STATS` line; logits identical to stock `f1c5faf` on the same
   prompt set. Static half is §2's header enumeration; the runtime half needs
   the 13022 run to mean anything.
5. **Non-reader regression**: plain decode loop without attention views → the
   graph carries no reader inputs, the detector walk is a no-op, and binder
   rebinds proceed per the unchanged skeleton/width checks; counters
   `c_attempts`, `c_applied`, `c_fallbacks` therefore behave exactly as in a
   pre-fix build.
