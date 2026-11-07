# CPU exact n_kv probe

The binder defaults on only for the CPU configuration described in
`src/llama-union-binder.h`: LLAMA without ALiBi, experts, relative attention
buckets or backend samplers. Explicit opt-in also permits the recurrent and
hybrid architectures listed in `union_arch_supported()`, subject to their
graph and backend guards. A live bind still checks for one scheduler split,
no relay or pipeline, one KV stream, compatible tensor shapes and allocation
envelopes. A refusal uses the stock rebuild path. Explicit
`GGML_TP5_UNION_BINDER=0` disables the binder; explicit `1` opts in. Unset
does not enable it for a GPU or accelerator context.

Every explicit `GGML_TP5_*` on/off knob in this lane follows one contract:
the value must be exactly `1`. Unset, `""`, `0`, `true`, `01`, ` 1`, `2` and
every other string are OFF. `GGML_TP5_UNION_BINDER` alone keeps its unset
CPU default; the sub-lanes below are off unless set to `1`.

`GGML_TP5_RSZ_RUNTIME_PARAM=1` (with the binder on) makes `rs_z`, the
recurrent-state cell that `build_rs()` zeroes, a runtime input write instead
of a graph key: the zero view covers every state row, the selection is an
I32 `[1]` input, and a moved `rs_z` neither binds nor rebuilds. It is refused
by name, never silently, when the context cannot carry it: no binder
(`rsz_rt_refused_gate`), no recurrent state (`_arch`), a zero view the shape
pass did not prove (`_no_rs_z`), `rs_z < 0` (`_cell`), `rs_z` outside the
zero view at decode time (`_envelope`), a missing or malformed selection
tensor (`_shape`), or `n_rs_seq > 0`, where the state rows are widened by
rollback snapshots and a row is no longer a cell (`_n_rs_seq`). Each refusal
increments exactly one named counter plus `rsz_rt_fallbacks`; all of them are
in `llama_union_diagnostics` and in the `NKV_BINDER_STATS` line.

The two decisions of that lane are named functions in
`src/llama-nout-state.h`, so they can be checked without a model:
`llama_rsz_rt_armable(gate_ok, has_recr, n_rs_seq, rs_z)` is the arming rule
in `graph_params()` (it is false for every `n_rs_seq > 0`, which is the
`_n_rs_seq` refusal), and `llama_rsz_rt_check_row(rs_z, zero_view_rows)` is
the decode-time check that splits `_cell` (`rs_z < 0`) from `_envelope`
(`rs_z >= zero_view_rows`). `test-rsz-lane-rows` pins both in every CPU
build.

The selection write itself has one definition: `union_rsz_bind_index()`
(`src/llama-graph.cpp`). It writes the row the gathers will read into the I32
`[1]` tensor `s_zero`, computed by
`llama_memory_recurrent_context::get_rs_zero_row(seq)` as
`rs_idx[seq]*mem_size + rs_z` -- the cell index `rs_z` alone is not the row,
because `s_copy()` returns `idx*size + src0`. It reads `rs_idx` without
consuming it; `s_copy()` is the one consumer. Four `set_input()` sites call it,
each right after its `s_copy` loop and before `consume_replay_len()`: the
standalone recurrent input, the hybrid input, the hybrid K-cache input, and the
hybrid iSWA input. The call is a no-op when `s_zero` is null, which is the stock
build. One graph carries one row, so a ubatch whose sequences disagree on the
row is refused before `set_inputs()` (`_mixed`), not asserted: `rs_idx` is per
sequence and a sequential ubatch split can mix rollback depths. The hybrid iSWA
site is unreachable by construction: `rsz_gate_ok` (`src/llama-context.cpp`)
excludes `union_is_hybrid_iswa(mctx)`, so a context with that memory never arms
the lane; the call is there so the rule has one form at every site, not because
a reachable path needs it. `test-rsz-selection-write` drives the helper itself
over a minimal recurrent memory and pins the null-index no-op, the write, the
rewrite after a moved `rs_z`, and that the write touches only the row.

## Test fixtures

`test-rsz-runtime-param` and `test-rsz-union-combo` take a tiny PURE
recurrent GGUF (`LLAMA_RSZ_TEST_MODEL`). On such a fixture
`union_nkv_now()` returns 0 -- it reads only a `llama_kv_cache_context` or a
hybrid attention sub-cache -- so the n_kv width lane never moves there and
those two tests cross `rs_z` with `rs_head` only.

`test-rsz-hybrid-compose` takes a tiny HYBRID GGUF
(`LLAMA_RSZ_HYBRID_TEST_MODEL`): a model whose memory is
`llama_memory_hybrid`, so it has a recurrent sub-cache (the `rs_z` lane) and
an attention KV cache (the n_kv lane) at the same time. The arches that
produce that memory in this tree are QWEN35 / QWEN35MOE / QWEN3NEXT /
QWEN4EXP (per-layer split), NEMOTRON_H / NEMOTRON_H_MOE, and JAMBA /
FALCON_H1 / PLAMO2 / GRANITE_HYBRID / LFM2 / KIMI_LINEAR
(`llm_arch_is_hybrid()`, `llama-arch.cpp`; the memory filters are in
`llama-model.cpp`). A model with a sliding window (`swa_type` not NONE)
gets `llama_memory_hybrid_iswa` instead, which the binder refuses, so the
fixture must not set a SWA pattern. The fixture used by this lane is
`tiny-qwen35-hybrid.gguf`: 2 layers, layer 0 gated-delta-net (recurrent),
layer 1 full attention, 262-token SPM vocabulary, 151104 bytes.

The test asserts what actually moves: the live n_kv is the true used span of
the attention cache (not a multiple of 256) while the stock arm stays at the
256-cell floor, and the two arms must disagree. On the same steps `rs_z` moves,
the live graph is reused, and logits and every recurrent state cell stay
bitwise equal to stock.

`test-rsz-nkv-rsz-topology` takes the same hybrid GGUF and the same
`LLAMA_RSZ_HYBRID_TEST_MODEL` variable. It crosses `rs_z` and the n_kv width
on the same steps: two sequences with distinct prefill lengths (5 and 3), then
one token per sequence per step, with seq 0 released and rejoined at position
0 on the odd steps (so `rs_z` moves) while seq 1 keeps growing the attention
cache (so the bound n_kv moves). It asserts topology invariance: after the
warm decode `llama_graph_build_count()` does not move on any step where both
parameters move, while the stock arm rebuilds on every one of those steps. It
also requires at least two distinct values of each parameter, reads the bound
n_kv from the live graph input and the selection row from the live graph's
`s_zero` (host-backed), and requires that row to equal the live zero row
`rs_idx[seq]*mem_size + rs_z` -- the check that catches a hybrid `set_input()`
that does not write the row.
Logits and recurrent cells stay bitwise equal to stock, and the named
refusals sum to `rsz_rt_fallbacks`.

`test-rsz-selection-write` needs no model file and no fixture variable. It
builds a one-layer, two-cell `llama_memory_recurrent` on the CPU buffer type
and a real `llama_memory_recurrent_context` over it, then calls
`union_rsz_bind_index()` directly: the row must receive the live zero row, a
moved `rs_z` must rewrite it (the property the hybrid sites used to miss), a
null row must be a no-op, and nothing but the row may change.

The four model tests exit 77 (ctest `SKIP_RETURN_CODE`) when their fixture
variable is empty, and are then not registered at all.

Build `padmax_cpu_probe.cpp` against a CPU-only libllama, then run, for example:

```sh
WT=/path/to/workdir LIB=/path/to/cpu/libdir MODEL=/path/to/tiny-llama.gguf \
  PROBE=/path/to/padmax_cpu_probe bash tests/nkv-cpu-probe/run_probe.sh all
```

The three arms are unset, explicit off and explicit on. There is one lane:
the executed n_kv width is the true used span, and the binder patches it in
place. `nkv` crosses the 256-cell boundary; `cap` exercises a non-aligned
capacity; `seq` checks that the unset default rejects a multi-stream context
and that explicit opt-in still refuses a multi-stream bind. The runner
records each arm's logits, context sizes, builds, binder counts and pairwise
comparisons. It does not establish correctness for GPU, relay, pipeline, or
other models.

The fixed-token-width n_outputs contract is separate from variable-token-width
reuse. The latter remains a reuse key and is not part of the fixed gate.

Behaviour change in this change set: the executed n_kv width is the true used
span on every path, including the gate-off path, while the gate-off capacity
sites (n_ctx, n_ctx_seq, n_ctx_kv) are still rounded up to 256 cells. The
256-cell bucket was graph-reuse padding; the reuse key now handles width
changes, so the bucket is gone. As a result the gate-off path is no longer
byte-identical to the pre-change binary for an n_ctx that is not a multiple of
256: it executes a different width and can rebuild once when the width moves.
It is not stock.

The padded control arm no longer exists, so no test asserts a padded executed
width any more. A regression that reintroduces padding is caught only by
`test-nkv-exact-width` and Gate D of `gate-p1-merge-13022.sh`, which read the
executed mask and K/V widths per step.

`test-nkv-exact-width` runs five cases. Each case is bracketed by
`NKV_CASE name=… n_seq=… unified=… steps=… arm=… skipped=…` and
`NKV_CASE_END name=… ok=… steps=… skipped=…`, and the run ends with
`NKV_EXACT_END ok=… cases=… steps=…`.

**The arm belongs to the runner.** The test never sets, unsets or overrides
`GGML_TP5_UNION_BINDER`, `LLAMA_EXPERIMENTAL_VARLEN_NTOK` or
`LLAMA_GRAPH_REUSE_DISABLE`. It reads them once at start and prints

```
NKV_ARM binder=<unset|0|1> ntok=<unset|0|1> reuse_disable=<unset|0|1> negative=<none|omit-*>
```

which is the arm identity every gate keys on. The runner
(`run-nkv-exact-width-arms-13022.sh`) starts one process per arm with `env`,
and the CMake registrations (`p1-nkv-exact-*`) inject the same variables with
`cmake -E env`. A case whose arm is not the running one prints `skipped=1`,
touches nothing and counts as neither a pass nor a failure:

| case | unset | on | off | why |
|---|---|---|---|---|
| `single` | run | run | skip | its `binder_ok()` assertions need binder state |
| `multi_seq` | run | run | skip | same |
| `multi_stream` | run | run | skip | it asserts that the qualified-CPU default refuses a multi-stream cache; with the gate explicitly off there is nothing to refuse |
| `widths` | run | run | skip | its `binder_ok()` assertions need binder state |
| `off` | skip | skip | run | the stock control; its capacity-pad assertions only hold with the gate explicitly off |

`unset` and `on` both run the binder, through different branches of
`llama_context::llama_context` (`llama_union_binder_enabled(nullptr,
cpu_default)` versus the explicit `"1"`); the `NKV_ARM` line is what makes
them distinguishable in a log.

`--negative omit-mask|omit-k|omit-v|omit-pad-tail` drops one observation from
the eval callback's matching set. For `omit-mask` that means the consumer walk
is suppressed, because the mask has no node of its own. For `omit-pad-tail` it
means the pad-tail width is never recorded, which is what an unsuppressed
producer that set no flag would look like. The case that runs must then fail
with
`FAIL: missing tensor at step=0 mask=… k=… v=… (negative: omitted <root> …)`,
where the flag of the omitted root is 1 and the other two are 0, or with
`FAIL: missing pad tail at step=0 (negative: omitted reduce_pad_tail …)` for
the pad tail. The process prints `NKV_NEGATIVE ok=1 omitted=<root>` and exits
0 only when it recorded that missing observation. A nonzero exit alone is not
proof, because a model load failure, a context failure or a wrong arm also
exits nonzero:

| exit | last line | meaning |
|---|---|---|
| 0 | `NKV_NEGATIVE ok=1 omitted=<root>` | the omission was detected: the gate is falsifiable |
| 1 | `FAIL: negative run omitting <root> passed` | the omission went undetected: the gate is worthless |
| 1 | `FAIL: negative run omitting <root> failed without the named 'missing tensor' line` | it failed for some other reason; proves nothing |
| 2 | — | usage error (bad `--negative` target, unknown arm value) |

The four roots are `attn_inp_kq_mask`, `union_nkv_k`, `union_nkv_v` and
`reduce_pad_tail`. The pad tail has its own evidence record because it is not
one of the three n_kv roots: it is a width on a node, and it can be missing
while the mask is present.

`gate-nkv-observe-13022.sh` runs the three arms and the four negative arms,
one `ctest -R '^<name>$'` each, with one `GATE_SUMMARY` key per arm
(`NKV_EXACT_NEGATIVE_PAD_TAIL` for the fourth). For a negative arm it requires
the named failure line for that root, the `NKV_NEGATIVE ok=1 omitted=<root>`
line, and the absence of the undetected-omission text. The CMake registrations
pin the same reason with
`PASS_REGULAR_EXPRESSION "NKV_NEGATIVE ok=1 omitted=<root>"` and
`FAIL_REGULAR_EXPRESSION "negative run omitting <root> passed"`, so a
negative ctest PASS is a PASS only for the right reason.

| case | context | what it shows |
|---|---|---|
| `single` | unified, n_ctx = n_ctx_kv = 300, 300 steps | one `NKV_EXACT` line per step; the mask and both K/V views execute at the 1-based position; one graph build |
| `multi_seq` | unified, n_seq_max = 2, 9 steps | two sequences with different used spans share one stream; the width is `max(seq_pos_max + 1)` over the sequences, clamped to the capacity; a `seq_rm` then moves the width down |
| `multi_stream` | non-unified, n_seq_max = 2, 4 steps | the unset gate does not qualify a multi-stream cache, so the run stays stock at the 256-cell **capacity** pad and no binder state exists; the mask row checked is the last token row of stream `s`, which is row `s` because one token per stream makes `ne[1] == 1` |
| `widths` | unified, capacity 257, 257 steps | the width crosses 1, 255, 256 and 257 and ends at the full physical capacity |
| `off` | explicit `GGML_TP5_UNION_BINDER=0`, 300 steps | stock control: the capacity is `GGML_PAD(n_ctx, 256)`, but the executed width is the **true used span** (`get_n_kv()` rounds by `n_pad`, which is 1 for a plain cache; the 256-cell bucket floor was deleted), and the path builds at least once without rebuilding per step |

**How the mask is observed.** `attn_inp_kq_mask` is a graph *input*
(`ggml_new_tensor_4d` + `ggml_set_input`), not a node. The eval callback is
invoked over `split->graph.nodes` only (`ggml/src/ggml-backend.cpp:2039`); an
input travels in `split->inputs[]` (`:1873`) and the callback never sees it.
Naming the mask cannot change that. So the callback reaches the mask through
its CONSUMERS: `SOFT_MAX` `src[1]`, `FLASH_ATTN_EXT` and
`FLASH_ATTN_EXT_BANDED` `src[3]`, and `MUL_MAT` `src[2]` when an attention
flag is set. That is the same reachability the production binder uses
(`llama-context.cpp:4721`).

The callback records only the mask's shape and a pointer to it. The buffer is
read after `llama_decode` returned, so no other thread can be inside the
consuming kernel. The read asserts that the mask is F32 (this test disables
flash attention), that every column below the executed width holds exactly
`0.0f` or `-INFINITY`, and that at least one column is visible. Which column
is visible is not asserted: the physical cell index is not exposed
(`find_slot` allocates the lowest free cell, so after a `seq_rm` the cell index
is not the decode index), and reconstructing it would be a guess. A mask that
hid the query's own cell but kept some other cell visible would therefore pass.

**The attention lane, and why `pad_tail=none` is correct here.** The binder
has two ways to tell a CPU attention node how wide the reduction is: the
ordered lane (`GGML_TENSOR_FLAG_ATTN_SOFT_MAX` / `ATTN_MASK`, which reads the
mask and skips masked columns by value) and the padded lane
(`GGML_TENSOR_FLAG_REDUCE_PAD_TAIL` with `op_params[2]`). This graph is on the
ordered lane, so no node carries the pad-tail flag. The chain, read in this
tree:

1. `build_attn_mha` takes its non-flash branch (`src/llama-graph.cpp:2995`)
   because the test disables flash attention.
2. It calls `ggml_mul_mat_set_attn_mask(kqv, kq_mask)`
   (`src/llama-graph.cpp:3051`) because the mask is set and V is F16/F32.
3. That function sets `ATTN_MASK` on `kqv` and also `ATTN_SOFT_MAX` on
   `kqv->src[1]`, i.e. on `kq_soft_max` (`ggml/src/ggml.c:3397-3399`).
4. The binder's softmax producer requires `(flags & ATTN_SOFT_MAX) == 0`
   (`src/llama-context.cpp:4719-4724`), so it skips that node.
5. Its other branch needs a `MUL_MAT` whose `src[1]` is a `SOFT_MAX` that
   already carries `REDUCE_PAD_TAIL` (`:4725-4733`), so it does not fire.
6. The replan path (`:5298-5313`) has the same two conditions and the same
   outcome.

So `pad_tail=none` is the correct value in every arm, and a width that does
appear is checked against the position rather than assumed away. What the
test asserts instead is the lane itself: the mask-carrying softmax carries
`ATTN_SOFT_MAX` and not `REDUCE_PAD_TAIL` (the two are mutually exclusive,
`ggml/src/ggml.c:1339`), and the value product carries `ATTN_MASK`. That is
falsifiable in three ways, and the `--negative omit-pad-tail` arm proves the
observation can be missed.

This is **not** an independent kernel measurement of the executed width. On
the ordered lane the kernels take the mask by pointer and the width from
`ne00`; there is no second number to compare against the position.

**The `pad_tail=` and `NKV_RT` fields.** Every step line of
`test-nkv-exact-width` (`NKV_EXACT`, `NKV_MULTI`, `NKV_WIDTH`,
`NKV_MULTI_STREAM`, `OFF_ARM`) ends with `pad_tail=<n>` or `pad_tail=none`,
the width of the node that carries `REDUCE_PAD_TAIL` or the word `none` when
no node does. `padmax_cpu_probe` prints the same token on its `NKV_EXACT`
line. The exact-width test also prints one `NKV_RT` line per step:

```
NKV_RT step=<n> runtime=<r> bound=<n> pos=<p> patch_binds=<k>
```

`runtime_n_kv` and `bound_n_kv` come from `llama_get_nkv_runtime_stats`
(`include/llama.h:689-697`). They are the binder's own bookkeeping, not
kernel evidence, and are labelled as such. `patch_binds` is printed because
the pad-tail patch and the shape patch are written inside one bind
(`src/llama-context.cpp:4719-4733`), before `record_bound` is set (`:4750`);
a refused bind can therefore leave a patch on a graph that is then rebuilt,
which is why every asserted step also requires `rebuild_fallbacks == 0`.

Explicitly not claimed by this test, because no public CPU input reaches them:
the `result > min_size` guard of `get_n_kv` (unified caches have one stream, so
the width is `min(capacity, used_max_p1) <= min_size` by construction; the
binder refuses multi-stream before the guard matters); the `v_trans` align
branch (needs a quantized V block size > 1, while this test uses F32/F16); and
a width that moves down through `seq_rm` alone (the freed cells are reused
from the stream head, so the used span does not shrink). Bitwise equality
between the binder arm and the stock arm is also not claimed: their executed
widths differ by design.

## n_kv, n_outputs and rs_z as runtime parameters (pillar 1)

The binder lane above keeps the executed width exact by patching shapes in
place. That makes the width a *shape* the graph is rewritten to; pillar 1 asks
for the width to be a *value* the graph reads. Both halves now exist and are
separately checkable:

| piece | file | what it is |
|---|---|---|
| runtime width input | `src/llama-graph.cpp` (`llm_graph_input_nkv_runtime`, `build_inp_nkv_runtime`) | one I32 `[1]` graph input named `union_nkv_runtime`; its value is rewritten before every compute from `llama_kv_cache_context::get_n_kv_runtime()` |
| the consumer | `llm_graph_input_nkv_runtime::set_input` + `llama_kv_cache::set_input_kq_mask(..., n_kv)` | the input reads its own value back and fills every registered KQ mask at that width, so the mask fill width is the runtime value, not the mask tensor's shape. Only `build_attn_inp_kv` creates the input and registers its mask; the other attention builders (V-less, MSA, iSWA) register nothing. The K/V views are **not** consumed this way; their width is the binder's in-place view patch |
| width table | `src/llama-union-width-table.h` | records which dimension of which node carries each runtime parameter: n_kv (`union_nkv_k`, `union_nkv_v` in both layouts, `attn_inp_kq_mask`), n_outputs (`union_out_ids` row 0), rs_z (`union_rs_copy`, `union_rs_copy_extra`, `union_rs_copy_dst`, `union_rs_head_view`); fails closed when a root's shape disagrees with the recorded value |
| topology hash | `src/llama-union-topology.h` | FNV-1a 64 over node count, per-node op type, flags, all op params (except the view-offset copy that ggml also stores there), non-runtime shape dims, name, and every edge as an index; the n_kv, n_outputs and rs_z dimensions are excluded. Taken at the record build and **re-taken after every applied bind**; a bind whose hash cannot be re-taken is refused and the caller rebuilds |
| one combined bind | `llama_union_binder::rebind_any` + `union_rebind_apply` (`src/llama-context.cpp`) | a registered kind is a gate, not a lane: the first kind whose preconditions hold calls `union_rebind_apply`, which evaluates the symbolic shape map at all three targets in ONE patch. A kind whose own target did not move cannot mask a delta in another kind, and a step that moves all three needs one bind |
| public read-out | `llama_get_nkv_runtime_stats()` | `topology_hash`, `build_n_kv`, `runtime_n_kv`, `bound_n_kv`, `build_n_outputs`, `bound_n_outputs`, `runtime_n_outputs`, `bound_n_rs`, `bound_rs_head`, `bound_rs_z`, `topology_hashed`, `has_nkv_dims`, `has_nout_dims`, `has_nrs_dims` |

`test-nkv-runtime-param` checks both. Its `--hash-only` mode (registered as
`test-nkv-runtime-param-hash`, no model needed) builds synthetic skeletons of
all three lanes at 5 n_kv values (64, 256, 257, 300, 1024), 4 n_outputs values
(1, 2, 257, 300) and 4 n_rs values (8, 9, 260, 301) — 80 combinations — and
asserts the hash does not move, then asserts it *does* move for an extra node,
a changed op parameter, a rewired edge, a changed non-runtime dimension, a
renamed node and a moved non-runtime view offset. Its model mode
(`test-nkv-runtime-param`, registered only when `LLAMA_VARLEN_TEST_MODEL` is
set; a missing or unreadable model is an error, never a skip) decodes 300
steps with the binder on and asserts, after every step, that the topology hash
and the build width are unchanged and that `runtime_n_kv` equals the 1-based
position — crossing 256 on the way — **and** that at least one step executed
below the build envelope, that `bound_n_kv` moved between steps and that more
than one distinct `bound_n_kv` value was seen. The last three are what make the
hash a re-taken value rather than a constant captured at build time; the
`NKV_RT_LIVE` line prints `bind_steps` and `distinct_bound` for the gate.

Part 3 of the same model run is the three-parameter cross matrix. One context
decodes a fixed token history as batches of 1..4 tokens with the binder on, so
n_kv, n_outputs and (on a recurrent or hybrid model) the rs_z lane all move
between cells; a stock context replays the same history with
`LLAMA_GRAPH_REUSE_DISABLE=1`, so it rebuilds for every cell. Every cell must
report the first cell's topology hash, an unchanged `build_n_kv`,
`build_n_outputs` and `bound_n_rs` (no rebuild), the executed parameters the
cell asked for, and logits byte-identical to the stock row (`memcmp`). The
`rs_z` axis needs a recurrent or hybrid model (MAMBA / RWKV, or QWEN35 /
QWEN3NEXT); a pure dense Llama has no recurrent state, so the test prints
`TRIPARAM_AXES ... rs_z=0 reason=dense_model` and runs the two-axis matrix. It
never reports a three-axis pass it did not run. The registration
`test-union-four-param-matrix` runs the same binary with `--matrix-only`,
which runs part 1, part 2 and part 3. Its `PASS_REGULAR_EXPRESSION` names
the `FOUR_MATRIX` line from part 2 only, and ctest ignores the exit code
when a PASS regex is set, so the registration also carries
`FAIL_REGULAR_EXPRESSION` for a `FAIL:` line, `FOUR_END ok=0` and
`FOUR_MASK_UNPROVEN ok=0`. The live n_kv lane keeps its own registration.

What this does not prove: **no ggml kernel reads the runtime width.** The
consumer is the host-side mask fill. Flash attention on GPU/HIP still needs a
device-side width parameter, and `GGML_TP5_FA_MASK_KVLEN=1` remains prohibited
there. Bitwise equality against a binder-off arm is also not claimed: the
gate-off arm runs the stock width.

## Three-arm runner and JSON report

`test-nkv-exact-width` runs its five cases in one process and forces the
binder off inside the process for its own `off` case. That cannot show what
the same binary does when `GGML_TP5_UNION_BINDER` is unset, explicitly `0` or
explicitly `1` at process start. `run-nkv-exact-width-arms-13022.sh` starts
the same binary once per arm, each in a stripped environment. `ARMS` defaults
to seven arms, three positive and four negative:

| arm | environment at process start | what it is |
|---|---|---|
| `unset` | `GGML_TP5_UNION_BINDER` unset | the default CPU path under test |
| `off` | `GGML_TP5_UNION_BINDER=0` | the stock control arm |
| `on` | `GGML_TP5_UNION_BINDER=1` | explicit opt-in |
| `negative-mask` | binder unset, `--negative omit-mask` | the mask omission must be detected |
| `negative-k` | binder unset, `--negative omit-k` | the K omission must be detected |
| `negative-v` | binder unset, `--negative omit-v` | the V omission must be detected |
| `negative-pad-tail` | binder unset, `--negative omit-pad-tail` | the pad-tail omission must be detected |

The default is the literal line in the script:
`ARMS=${ARMS:-unset off on negative-mask negative-k negative-v negative-pad-tail}`.

Every arm also gets `HIP_VISIBLE_DEVICES=` and `ROCR_VISIBLE_DEVICES=` empty
(CPU only), `GGML_TP5_RSZ_RT` unset (the forbidden promotion path must never
appear), and `LLAMA_EXPERIMENTAL_VARLEN_NTOK` and
`LLAMA_GRAPH_REUSE_DISABLE` unset (this is the non-token lane with the
default reuse key).

```sh
BIN=/path/to/test-nkv-exact-width \
MODEL=/path/to/tiny-dense-llama.gguf \
RUN_DIR=/path/to/new-dir \
  bash tests/nkv-cpu-probe/run-nkv-exact-width-arms-13022.sh
```

`RUN_DIR` must not exist. The runner writes `<arm>.log` (stdout+stderr),
`<arm>.rc`, `report.json` and `identity.txt` (binary and model sha256,
revision, branch, host, date, nproc, ccache version). `ARMS` defaults to the
seven arms in the table above; an arm left out is reported `NOT_RUN`, never
`PASS`.

Exit status is `0` only when every arm ran, every arm's rc is 0 and the report
says `OK` for all of them. `1` is a real failure. `3` is `NOT_RUN`: `BIN` or
`MODEL` missing, or an arm skipped. A missing input writes a report whose
every arm is `NOT_RUN` with a reason; nothing is faked.

### Report schema

`nkv-exact-width-report.py` parses the test's stderr into JSON. It measures
nothing itself.

```json
{
  "arms": {
    "<arm>": {
      "arm": "<arm>", "status": "OK|FAIL|NOT_RUN", "ran": true,
      "log": "<path>", "rc": 0, "reason": null,
      "problems": ["..."],
      "cases": [{"name": "single", "n_seq": 1, "unified": 1,
                 "declared_steps": 300, "observed_steps": 300,
                 "ok": 1, "footer_steps": 300, "arm": "binder",
                 "skipped": "0"}],
      "case_count": 5,
      "arm_env": {"binder": "unset", "ntok": "unset",
                  "reuse_disable": "unset", "negative": "none"},
      "end": {"ok": 1, "cases": 5, "steps": 867},
      "multi_stream_context": {"n_ctx": 300, "n_ctx_seq": 256,
                               "n_ctx_kv": 256, "n_seq_max": 2,
                               "stock_width": 256},
      "steps": {"NKV_EXACT": [...], "NKV_MULTI": [...],
                "NKV_MULTI_STREAM": [...], "NKV_MULTI_STREAM_BIND": [...],
                "NKV_WIDTH": [...], "NKV_RT": [...], "OFF_ARM": [...]},
      "pad_tail": [null, null, 0, "none", "..."],
      "off_arm_total": {"builds": 1, "steps": 300},
      "off_arm_reuse": {"ok": 1, "builds": 1, "steps": 300},
      "fail_lines": ["..."]
    }
  },
  "summary": {"<arm>": "OK|FAIL|NOT_RUN"},
  "overall": {"ran": ["unset"], "not_run": ["on"], "ok": false}
}
```

`--expect-cases` and `--expect-steps` pin the case list and per-case step
counts (300, 9, 4, 257, 300 from the test source). A different count is a
failure, not a new baseline. The runner always passes both.

### NOT_RUN semantics

An arm whose log file does not exist is emitted with `"ran": false`,
`"status": "NOT_RUN"` and a `"reason"`, and carries no case list, no step
rows and no counts. A field whose line family is absent from an existing log
is `null`, never `0`: a zero in this report always means "measured as zero".
The writer exits `0` only when every arm that ran is `OK` and every arm that
did not run is `NOT_RUN`; `3` when any arm is `NOT_RUN`; `1` on any failure.
A missing arm never becomes a pass.

### What the seven arms do and do not prove

The three positive arms prove that the same binary, started three ways,
executes the widths the test asserts in each arm, and that the stock control
keeps its graph reuse. `off` is the stock-control arm: it is the path with no
binder state, not a second copy of the `unset` arm. The four negative arms
prove that each observation can be missed: each one suppresses one root and
must fail with its own named line and its own `NKV_NEGATIVE` record.

They do **not** prove bitwise equality between arms. The arms differ in
binder state and in allocated capacity (the `off` arm pads `n_ctx` to
`GGML_PAD(n_ctx, 256)`), and they reach the used span by different paths, so
comparing their logits bitwise would be a false gate. They no longer differ in
executed width: the 256-cell bucket floor is gone, so the `off` arm executes
the true used span too. The same limit is stated in the header of
`tests/test-nkv-exact-width.cpp`.

Nothing in this change set was compiled or run on 12022. The unit tests for
the report writer are pure Python and run there; the runner, the report over
real logs, and the CTest registration are unverified until they run on the
authorized CPU build host.

The token binder is opt-in with `LLAMA_EXPERIMENTAL_VARLEN_NTOK=1`. It is
limited to one CPU backend, a dense LLAMA model, plain single-stream F16/F32
KV, and non-flash attention. Unsupported, recurrent and hybrid graphs fall
back; `NKV_BINDER_STATS` reports token and recurrent/hybrid fallbacks. It
builds at the live KV, token and output widths and replans tensor allocations
at each changed layout. It never reserves a wider token or output width as
headroom. Replan uses a separate arena until commit, so peak memory can
contain both old and new allocations. Token-only changes have their own binder
dispatch even when n_kv is unchanged; independent sequence-token widths are not
supported by this lane. Neither the new path nor F16 masked reductions have
passed the full logits gate here.

Set `LLAMA_VARLEN_TEST_MODEL` to a tiny LLAMA GGUF at CMake configure time to
register the paired output probes and `test-ntok-strict`. The strict test
runs widths 1, 3, 2, 4 for 200 cycles with output counts 1, 0, 1, 4;
it checks active KV, mask, token and output-index plan sizes, arena capacity,
one graph build, and every requested logits row bitwise.
The ON arm of `test-ntok-strict` enables the binder; the OFF arm disables it
and uses the stock path. There is no separate exact flag: the executed n_kv
width is the true used span on both arms. This is a proposed bitwise
comparison, not a passed numerical qualification.

The previous CPU F32 non-flash sweep failed the bytewise gate. The applied
stock width is now used only as a reference reduction width; this source change
has not been compiled or validated.

The registered `test-nout-tail-fixed` and `test-nout-tail-instrumented`
require `NOUT_TAIL_MODEL` to name a local, supported tiny LLAMA GGUF; a missing
model is an error, not a skip. On authorized 13022, reproduce both and the
CPU attention checks with:

```sh
NOUT_TAIL_MODEL=/path/to/tiny-llama.gguf \
  bash tests/nkv-cpu-probe/reproduce_nout_tail_cpu.sh
```

When `LLAMA_NTOK_TEST_MODEL` is set, CTest also registers fixed and variable
paired probes with `LLAMA_EXPERIMENTAL_VARLEN_NTOK=1` so token and output
count changes are checked together.

Both modes require one ON decode build, 200 OFF decode builds, and all logits
rows to match byte-for-byte. The candidate remains unqualified for production:
the gates below are CPU-only evidence, and no run recorded in this repository
has yet been reproduced on the authorized CPU host with the current exact-lane
keys. Earlier instrumented runs reported extra builds and a bitwise mismatch;
those runs predate the current gate set and are not evidence either way for
it. Per-step counts use the actual graph-build counter, including fallback
builds. `max_possible_outputs` is a legal bound, not an allocation
measurement.

The `test-ntok-onegraph` target runs the real single-stream CPU LLAMA decode
path for 50 rounds of widths 1, 3, 2, 4 by default. Every row's logits and
final normalized activation are compared byte-for-byte with binder OFF. It
checks every decode's cumulative actual build count, token binding, and whole-
graph fallback count. It also checks the exact n_kv lane: `bound_n_kv` must
equal the cumulative token count after each step, and `pad_fallbacks` must not
move from its post-initialization value. Initialization reserve builds are
reported separately. Per-kind dispatch declines are reported, not treated as
whole-graph fallbacks. Only the final normalized activation is observed, not
all intermediate layers. The graph already marks that tensor as an output; the
diagnostic copies it after synchronization without enabling embeddings or an
eval callback.


The `test-ntok-bind-replay` target is the token-lane regression. ONE context
decodes the width schedule 1, 3, 2, 4 for three rounds (twelve steps by
default) with `GGML_TP5_UNION_BINDER=1` and
`LLAMA_EXPERIMENTAL_VARLEN_NTOK=1`. It requires an in-place bind on every step
after the record build (`patch_binds` strictly increases), zero graph rebuilds
after the first decode (`llama_graph_build_count()` delta 0 and
`rebuild_fallbacks` 0), `bound_tokens` equal to the batch width, and
`bound_n_kv` equal to the used span. Because `patch_binds` increments before
the binder's `can_reuse()` post-check, the test also requires
`record_valid && !envelope_disabled` on every step and on the final step: a
counted-but-rejected bind cannot pass. `has_nkv_dims` must be set, which is the
observable that the K and V write views are recorded; flash attention is off,
so the V write is the reshaped 2-D cache-write view, and the width changes
exercise that reshape. Every captured logits row must be byte-identical to a
per-length rebuild baseline: a FRESH context per captured step with
`LLAMA_GRAPH_REUSE_DISABLE=1`, replaying the same prefix. The baseline's build
count is sampled after every decode and the MINIMUM per-decode delta must be
at least 1, so the per-length rebuild is measured, not assumed. It prints
`NTOK_BIND`, `NTOK_BASE`, `NTOK_REPLAY` and `NTOK_END` lines; `NTOK_END`
carries the final bind, rebuild, bitwise, record and envelope counts. The
target is registered only when `LLAMA_NTOK_TEST_MODEL` names a dense LLAMA
GGUF, and a missing model is an error, never a skip. It is an INDEPENDENT
CTest, not a change to an existing gate: `tests/nkv-cpu-probe/gate-p1-merge-13022.sh`
is left unchanged. Run it directly with
`ctest --test-dir build -R '^test-ntok-bind-replay$' --no-tests=error -V`.

**The set_rows write binding.** The KV store is `ggml_set_rows(cache, cur,
idxs)`: the result is a view of the cache storage, `src[0]` is the value
tensor, `src[1]` the row indices and `src[2]` the cache. The binder narrows
the read views (`union_nkv_k`, `union_nkv_v`) and the KQ mask to the live
span; it must NOT narrow the write destination, whose row dimension is the
physical capacity. The write is bound on the value tensor's token axis and on
the index input. In the transposed-V layout (flash attention off) `cpy_v`
flattens `(n_embd_head, n_head, n_tokens)` to `(1, n_embd_gqa*n_tokens)`, so
the token axis is axis 1 with factor `n_embd_gqa`; `cpy_v` tags it with
`LLM_TOKEN_AXIS_MARKER | 1` and the shape pass derives
`ne[1] = n_tokens * n_embd_gqa`. Without the tag the axis search can pick the
degenerate trailing axis 2, and the CPU kernel then iterates `i02` over tokens
while adding a full cache span (`dst->nb[2]`) per plane, writing outside the
cache. The symbolic shape pass refuses a `SET_ROWS` whose source carries a
runtime coefficient on `ne[2]`/`ne[3]` for the same reason. It also refuses a
dependent coefficient on the destination's `ne`/`nb` (the cache allocation
envelope) and on the index tensor's plane extents, and it re-checks the
constructor and kernel relations after copying the destination shape. The tag
is applied only when `strict_ntok` is set (the token lane); every other lane
is byte-identical to before.

Both tail gates use two separate CPU contexts with identical weights and tokens,
one sequence, a single four-token ubatch per decode, no flash attention,
and active output counts 4, 1, 3, 2 repeated for 200 steps. The first graph
is built for four actually requested outputs; there is no output padding or
extra output headroom. Later requests shrink and grow within this observed
high-water width. The last requested row is always the last batch token.
Every requested logits row, including that tail row, must match bitwise
between explicit binder ON and OFF. The gate checks both the batch-token
index and `llama_get_logits_ith(-1)`, finite outputs, a changing tail across
steps, and nonzero shrink and growth coverage.

The uninstrumented gate has no graph evaluation callback. The instrumented
gate observes `result_norm` and `result_output` on both arms, requires exactly
one visit to each, checks their actual output-row dimensions and every row
bitwise, and matches the captured final logits row to the public API. A
missing node, missing row, mismatch, or callback that prevents reuse fails
the gate. Instrumentation can change scheduling; its success does not replace
the uninstrumented gate, and both are mandatory in the script.

For each gate, the per-step decode build counts come from
`llama_graph_build_count()` deltas and require one ON decode build and 200 OFF
decode builds. This is supplemented by the ON context's actual
`NKV_BINDER_STATS`: exactly one `graph_builds`, one envelope build
and shape analysis, no shape failure or disabled envelope, output cap 4,
nonzero in-place shrink and growth coverage, and no grow rebuild. Missing
statistics fail closed. The graph-build counter catches internal fallback
rebuilds that `n_reused` alone cannot count. Initialization reserve graphs are
outside these decode counters. The OFF context has no binder stats.

The unified gate for this merged line is
`tests/nkv-cpu-probe/gate-p1-merge-13022.sh`; it runs the build-count, tail and
strict no-headroom gates in one configure/build and writes `GATE_SUMMARY`.

The older runner still offers `fixed`, `variable`, or `all`; default is
`fixed`. `variable` retains its original one-ON-build criterion, which is
incompatible with the current token-width reuse key. It is a separate
diagnostic, not a prerequisite of the fixed n_outputs gate. Its optional
`NKV_CAPTURE_KQV=1` observes masked-KV attention rather than output-tail
rows and can affect scheduling.

The non-flash CPU F16/F32 value product carries the softmax mask as an
explicit graph input. A mask entry equal to negative infinity excludes the
corresponding score and V; a zero probability alone does not. The paired
softmax and value product visit visible indices in increasing order. The value
product rounds probabilities to F16 when V is F16 and accumulates in double;
F32 products also accumulate in double. An all-masked row without a sink
produces positive zero. This deliberately changes the stock SIMD/SGEMM
reduction order in both arms; it does not emulate the stock dispatch.

The F16/F32 K by F32 Q score product is also marked with the explicit mask
when this graph arm is eligible. It skips masked K and computes each visible
dot in fixed scalar order, rather than selecting a width-dependent upstream
KQ implementation. Identical visible K, Q, mask, values and floating-point
environment therefore give the same ordered attention operations on both
arms. End-to-end equality and actual binder execution are still unproved until
the gates run; this source-level argument is not runtime evidence.

The ordered kernel has strict floating-point compilation options, including
disabled contraction and LTO on GCC/Clang-family compilers. Unsupported
compilers are rejected at configuration. Floating-point environment, compiler
behavior, CPU architectures and other graph nodes still need qualification.
The scheduler requires an explicit backend capability for all ordered nodes;
the CPU backend supplies it for ordinary layouts and rejects extra-buffer
inputs. Unsupported devices can cause CPU copies or a failed dispatch. This
can significantly reduce GPU throughput and is not a GPU implementation.

`test-cpu-attn-pad --f16-kqv` compares marked F16 products at multiple widths.
`test-cpu-attn-mask` covers F16/F32 V and masks, masked NaNs and garbage,
per-query and broadcast masks, half probability conversion, signed zero,
all-masked rows, sinks, unmasked zero-probability NaNs, and exact-capacity
storage with a guarded final boundary. `test-cpu-attn-mask-widths` compares
ordered live and reference-width softmax and products. `test-cpu-attn-ordered-kq`
exercises the marked score path for F16/F32 K and masks, multiple widths and
queries, with NaNs in masked K and an independent visible-dot reference.
`test-cpu-attn-true-len` and `test-cpu-attn-true-len-counters` cover the
runtime-length reduction: the first proves the softmax tail past the bound
length is zeroed and that the `--true-len` control isolates the flag, the
second counts the executed elements through a real CPU backend. The separate
`test-cpu-attn-mask-kq-dispatch` retains the unmarked upstream dispatch
diagnostic; it is not a prerequisite for the marked-path gate. The upstream KQ
matmul can expose width-dependent dispatch. Flash tests compare masked poisoned
and clean inputs; flash split-KV dispatch and its reduction can still change by
width. None of these tests proves the actual binder or an end-to-end model.
Flash split-KV behavior, non-CPU backends and performance remain unqualified.
No test, compilation, backend fallback, other architecture or performance
result is claimed without running it.

## The shared arm machinery (`union_arm.hpp`)

Three probes compare a union-binder context against a stock context on one
model. They used to carry three copies of the same block, and two of those
copies disagreed about which environment variables an arm owns — which is
exactly how an arm quietly becomes the same path as its control.
`union_arm.hpp` is that block, header-only and `inline`, so no probe gains a
translation unit and every probe still links `llama` only.

| Piece | What it is |
| --- | --- |
| `union_arm::arm_env` | all six variables an arm may touch, applied as one block. A null member **unsets**, so no arm inherits a setting from the arm built before it. `binder_arm()` / `stock_arm()` are the two arms every probe compares |
| `open_model` / `close_model` / `cpu_params` / `make_context` | the CPU-only model and context setup the binder requires (one unified stream, one sequence, no flash attention, no offload) |
| `build_delta` | scope form of `llama_graph_build_count()`. This is the measurement that makes two arms provably different paths |
| `rows_equal` | size-checked `memcmp` over a captured logits block; a block captured at a different row count is never equal |
| `fail` / `gate_fail` | the one `FAIL:` / `GATE_FAIL=` printer, so a shell gate matches one shape |

A probe keeps its own schedule, its own gates and its own report; only the
machinery is shared.

The environment, and where each variable is read:

| Variable | Read at | Selects |
| --- | --- | --- |
| `GGML_TP5_UNION_BINDER` | `llama_context` constructor | binder arm vs stock |
| `LLAMA_GRAPH_REUSE_DISABLE` | `llama_context` constructor | stock arm that rebuilds every step |
| `GGML_TP5_UNION_BINDER_EXACT` | `llama_context` constructor | `0` selects the stock padded control lane (the A/B arm) |
| `LLAMA_NKV_MASK_OVERRIDE` | `llm_graph_input_nkv_runtime::set_input` | mask-width refusal negative case |
| `LLAMA_EXPERIMENTAL_VARLEN_NTOK` | `llama_context` constructor | token lane |

## n_outputs runtime parameter probe

`nout_runtime_probe.cpp` (target `test-nout-runtime-probe`) measures the
n_outputs lane on its own. Two CPU contexts share one model: the ON arm is
`union_arm::arm_env::binder_arm()`, the OFF arm is `stock_arm()`. Both receive
the same batch at every step.

The cycle submits 4 tokens per step with active output counts 1, 3, 2, 4, 1, 3,
2, 4, 2, 3 and a moving subset of batch positions marked as outputs. The
growth arm then submits 5 tokens per step and requests 5, 3, 5 outputs with
`n_outputs_max = 5`: 5 is above the cycle's high-water of 4, so the first of
those is the envelope-expanding rebuild and the two after it bind in place. A
final step requests 6 outputs, above `n_outputs_max`, and must be refused.

Gates, all fail-closed: G1 the OFF arm does not report an enabled binder (a
stock context allocates no binder state, so "no state" is correct); G2 the ON
arm reports an enabled binder with `envelope_disabled` clear; G3 the ON decode
build total is exactly 2; G4 no cycle step after the first adds an ON build;
G5 the growth step adds exactly one; G6 no step after it adds any; G7 every
step executes on both arms; G8 every requested logits row is byte-identical
between the arms; G9 the above-max request fails on both arms, builds nothing
on ON and leaves the KV position untouched; G10 the OFF arm follows the stock
reuse rule; G11 the whole token budget is submitted.

The stock reuse rule is derived from the schedule, not guessed. With no binder
state the OFF arm reuses whenever `res->can_reuse()` holds, and
`llm_graph_input_out_ids::can_reuse` compares the bound width against the
active count. Inside the cycle the skeleton is constant, so stock rebuilds
exactly when the active count changes. Over the whole run the probe also
requires the OFF arm to build strictly more graphs than the ON arm.

What it does not claim: the public API exposes no bound-output count.
`llama_union_diagnostics::bound_tokens` is the bound **token** width, so it is
reported and never asserted. `c_nout_grow_binds`, `c_nout_grow_rebuilds`,
`c_refused_nout`, `record_cap_nout`, `record_bound_nout` and `nout_high_water`
are internal and are not read.

Registered under the existing `LLAMA_NTOK_TEST_MODEL` gate: the executable is
always built, the ctest entry appears when that variable names a dense LLAMA
GGUF.

## Aggregate union/onegraph gate

`run_union_ctest_gate_13022.sh` is the master gate for pillar 1. It takes an
existing build tree, proves every union/onegraph test is registered, runs
exactly those tests through ctest, and writes one aggregate JSON. It is a layer
above the existing gates: the driver's eight gate keys and
`run_union_varlen_e2e_13022.sh`'s own pre-flight are unchanged.

What the aggregate proves, and what it does not: it proves that each declared
test is registered, that ctest recorded it as passed, and that it left the
artifact it owns, with that artifact saying the tool passed. It does **not**
re-derive any numerical verdict. The ctest status is the test process's own
exit status; for `test-varlen-output-*` the artifact check is structural only
(a record file of the right shape), and for `test-union-varlen-e2e` it is the
driver's own `verdict: "pass"` and `exit_code: 0`. The numerical verdict stays
with each test's own gate and its own report; this aggregate only makes sure
that gate ran, passed, and left its evidence behind.

On authorized 13022, with a new output directory and a build tree that has the
union tests registered:

```sh
RUN_DIR=/home/kunweiz/work/union-ctest-gate-13022-$(date +%Y%m%d-%H%M) \
BUILD_DIR=/path/to/the/configured/build \
  bash "$(git rev-parse --show-toplevel)/tests/nkv-cpu-probe/run_union_ctest_gate_13022.sh"
```

The script configures nothing and builds nothing. The build tree must come from
a configure with `-DLLAMA_BUILD_TESTS=ON`, `-DLLAMA_UNION_VARLEN_E2E=ON`,
`-DLLAMA_NTOK_TEST_MODEL=<tiny LLAMA GGUF>` and
`-DLLAMA_UNION_VARLEN_E2E_JSON=$RUN_DIR/union-varlen-e2e.json`; the full
configure, the build target list and the one-command form are in RESULT.md of
the run that added this section and in the `### 13022 command` section above.

The build targets are the `add_executable` names from `tests/CMakeLists.txt`,
not the `add_test` names: `test-varlen-output-probe` is one executable that both
`test-varlen-output-fixed` and `test-varlen-output-variable` run, and each of
those two writes its records to `${CMAKE_CURRENT_BINARY_DIR}` of `tests/`, that
is `<build>/tests/varlen-<mode>-{on,off}.bin` — the paths `DECLARED` expects.

### The declared set

`DECLARED` at the top of the script is the single source of truth for which
tests must be registered:

| Test | Registered by |
| --- | --- |
| `test-union-binder-gate` | always, by `tests/CMakeLists.txt` |
| `test-union-varlen-e2e` | `LLAMA_UNION_VARLEN_E2E=ON` |
| `test-ntok-onegraph` | non-empty `LLAMA_NTOK_TEST_MODEL` |
| `test-ntok-bind-replay` | non-empty `LLAMA_NTOK_TEST_MODEL` |
| `test-varlen-output-fixed` | non-empty `LLAMA_NTOK_TEST_MODEL` |
| `test-varlen-output-variable` | non-empty `LLAMA_NTOK_TEST_MODEL` |

Each line also names the artifacts that test must leave behind:
`test-union-varlen-e2e` its `union-varlen-e2e/3` report, and each
`test-varlen-output-*` its two `.bin` record files in the build tree's `tests/`
directory.

`test-union-ctest-gate`, the summarizer's own fixture test, is deliberately
**not** in `DECLARED`. It is a fixture test of this summarizer, not a union
runtime test: it needs no model and no build, it is registered by the same
`LLAMA_UNION_VARLEN_E2E` switch, and it runs in the ordinary `ctest` suite.
Putting it in the declared set would make the gate depend on the test that
tests the gate, and would report a summarizer bug as a union-runtime failure.

### Pre-flight, run, summarize

Pre-flight is `ctest --test-dir "$BUILD_DIR" -N` over the whole tree, kept in
`$RUN_DIR/ctest-list.txt`. The listing, not `ctest --no-tests=error`, is the
authority for registration: a declared test that the listing does not contain
fails the gate before anything runs, with `failure:
"not_registered: <test>"` and exit 1. A non-zero `ctest -N` fails the same way
with `preflight_failed`.

Two input checks run before that, so an unusable invocation reads as itself
rather than as a pre-flight failure. A `BUILD_DIR` that is missing or is not a
directory stops with `FAIL: BUILD_DIR is not a directory: <path> (cd exited N)`
and exit 1, instead of dying on `cd` under `set -e` with no message. A missing
`union_ctest_gate.py` stops with `FAIL: summarizer not found at <path>`, and a
`python3` that is not on `PATH` stops with `FAIL: python3 not on PATH; the
summarizer cannot run`; both are checked before the first `python3` call, so
they are never reported as `not run: pre-flight failed`.

The run is `ctest -R '^(<declared names>)$' --no-tests=error -V` under
`env -i PATH HOME`, with `--output-junit "$RUN_DIR/ctest-junit.xml"` so status
and wall time come from ctest's own record rather than from scraping the
console. `-R` is anchored to the declared names, so no other test is run.

`union_ctest_gate.py` then reads the listing, the JUnit XML and each test's
artifact and writes `$RUN_DIR/union-ctest-gate.json`. It is invoked as
`python3 -` with its code on stdin, so the script stays self-contained.

### Aggregate JSON (`union-ctest-gate/1`)

```json
{
  "schema": "union-ctest-gate/1",
  "verdict": "pass | fail",
  "failure": null,
  "argv": ["string"],
  "exit_code": 0,
  "run_dir": "string",
  "build_dir": "string",
  "preflight": {"rc": 0, "listing": "string", "registered": ["string"]},
  "ctest": {"rc": 0, "junit": "string"},
  "declared": ["string"],
  "tests": [
    {
      "name": "string",
      "registered": true,
      "status": "passed | failed | not_run",
      "exit_code": 0,
      "wall_time_s": 0.0,
      "log": "string",
      "artifacts": [{"kind": "e2e-json", "path": "string", "ok": true, "checked": true}],
      "problems": ["string"]
    }
  ],
  "not_run": "string"
}
```

`status` is what ctest recorded: `passed`, `failed`, or `not_run` for a test
ctest skipped or could not start. `exit_code` is `0` only for `passed`; the
per-test exit code is not in the JUnit file, so it is derived from the status
and never invented. `registered` is the pre-flight listing's answer.

`artifacts` records each declared artifact. An artifact is evidence only when
`checked` is true, and `checked` is true only for a test ctest recorded as
`passed`: for any other status the file may be stale from an earlier run, so it
is recorded as `ok: false, checked: false` with the problem `not_checked` and is
never read. For a passed test, a missing file is `missing_artifact` and a file
that exists but is not the owning tool's passing output is `bad_artifact`
(wrong schema, or for the e2e report a `verdict` other than `pass` or an
`exit_code` other than `0`). Neither is ever a silent omission.

`failure` names the first failing condition, in this order:
`preflight_failed`, `not_registered`, `not_run`, `failed`, `missing_artifact`,
`bad_artifact`. A missing registration outranks everything else, because a
test that never registered was never run; a test's own failure outranks its
artifact, because the artifact was not read.

The summarizer's exit status is its own vocabulary: `0` pass, `1` gate fail,
`2` it cannot read its own inputs (bad usage, unreadable or incomplete config),
in which case no aggregate is written.

### Summarizer unit test

`test_union_ctest_gate.py` (target name `test-union-ctest-gate`) feeds the
summarizer synthetic listings, synthetic JUnit XML and synthetic artifacts. It
needs no GGUF model, no compiled binary and no build tree: it is plain
`python3`, stdlib only. It is registered under `LLAMA_UNION_VARLEN_E2E` in
`tests/CMakeLists.txt` and proves that all-present-and-passing gives `pass`,
that a missing registration gives `fail` naming the test, that a missing
artifact gives `fail`, that wrong artifact content gives `bad_artifact`, that a
failed test gives `fail`, that a failed test's stale artifact is recorded as
`not_checked` rather than as evidence, that an e2e report whose own verdict is
`fail` gives `bad_artifact`, that a registered test with no JUnit case gives
`not_run`, and that a failed pre-flight outranks later problems.
