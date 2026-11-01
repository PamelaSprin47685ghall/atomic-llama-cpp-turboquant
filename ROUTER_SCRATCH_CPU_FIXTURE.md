# Router scratch — CPU fixture and verifier (real consumer)

Status: **UNEXECUTED**. Everything below is code plus the exact commands to run
it. No build, run or result is claimed here; only Devops builds and runs
(serial, `build-slot`, CPU only).

## 1. What is new

Tree: `/home/kunweiz/work/merge-router-20261031`
(deltas of record: `git status` / `git diff` in that tree).

| File | Kind | Purpose |
|---|---|---|
| `examples/tp5-router-verify/tp5-router-fixture.cpp` | new | writes the minimal real Qwen4Exp 512/top-10 checkpoint and self-checks it |
| `examples/tp5-router-verify/tp5-router-verify.cpp` | rewritten | N1/N2/N3/N4 model leg + one-layer plan leg, hashed evidence |
| `examples/tp5-router-verify/run-router-cpu.sh` | new | drives fixture + verifier OFF/ON and checks the criteria |
| `examples/tp5-router-verify/CMakeLists.txt` | changed | two targets: `llama-tp5-router-fixture`, `llama-tp5-router-verify` (both registered up front; no manual linking) |
| `ggml/src/ggml-alloc.c` | changed | plan audit fixed to audit *retained* allocations only (see §5) |

## 2. Fixture: what is real and what is shrunk

Both example sources include public headers only (`llama.h`, `gguf.h`,
`ggml.h`, `ggml-backend.h`, `ggml-alloc.h`) and start the CPU backend through
the registry (`ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)`),
so they also build against a pristine stock tree where `src/` is not on the
include path. The fixture writes no `attention.indexer.types` array: only
`glm-dsa` reads that key (optionally); qwen4exp never does.

`llama-tp5-router-fixture -o m.gguf` writes a normal GGUF that the **stock**
`llama_model` loader reads. It is not a mock: there is no branch in the model
or backend code that knows about it, and the decode reads real F32 weights.

Real (production values):

- `n_expert = 512`, `n_expert_used = 10` — the hard-wired gate in
  `src/models/qwen4exp.cpp:1502-1504`
  (`n_expert == 512 && n_expert_used == 10 && 2 <= cur->ne[1] <= 4`);
- `hyper_connection.count = 4` (production HC4);
- the router projection is one `MUL_MAT` of `ffn_gate_inp` `[n_embd, 512]`
  against the layer input, followed by `SOFT_MAX` `[512, rows]`, `TOP_K 10`
  and the routed `MUL_MAT_ID` — the same node sequence `build_layer_ffn`
  emits, in the same order, so the retention's shape predicate
  (`SOFT_MAX` with `ne[0] == 512`, `ne[1] == rows`, a direct contiguous F32
  `MUL_MAT` input) matches exactly once per layer;
- 2 trunk blocks + 1 `NextN` block (`block_count = 3`,
  `nextn_predict_layers = 1`), so `n_layer() = 2` (trunk) and
  `n_layer_all = 3`; `attention.recurrent_layers = [1,1,0]` — the trunk blocks
  are gated-delta-net (linear-attention) layers, and the NextN block is full
  attention, which is what the loader computes itself when the key is absent
  (`is_recr(i) = i < n_layer() && (i+1) % interval != 0`, `qwen4exp.cpp:235-243`).
  With the key present it is used as-is, so an all-ones array would wrongly make
  blk.2 recurrent;
- every tensor the model's `load_arch_tensors` asks for, with the shapes it
  asks for, per layer branch (`qwen4exp.cpp:309-403`):
  * **trunk blocks 0..1** — `is_recr` true, gated-delta-net branch
    (`:350-360`): the nine tensors below;
  * **block 2 (NextN)** — past the trunk, so `is_recr`/`is_ple` are false and
    it takes the **full-attention branch** (`:337-349`), exactly like the
    comment at `:312-313` says: `attn_qkv.weight`
    `[n_embd, 2*n_embd_head_k*n_head + 2*n_embd_k_gqa]` (from
    `create_tensor_qkv`, q and gate interleaved, then k, then v),
    `attn_output.weight` `[n_embd_head_k*n_head, n_embd]`,
    `attn_q_norm` / `attn_k_norm` `[n_embd_head_k]`,
    `indexer.q_proj` `[n_embd, indexer_n_head*idx_dim]`, `indexer.k_proj`
    `[n_embd, idx_dim]`, `indexer.q_norm` / `indexer.k_norm` `[idx_dim]`, plus
    the same two HC mixers and the same MoE/shared-expert set, plus the NextN
    head tensors `nextn.enorm` `[n_embd]`, `nextn.hnorm` `[hc_dim]`,
    `nextn.eh_proj` `[2*n_embd, n_embd]`. The `nextn.hc_head_*`,
    `nextn.embed_tokens` and `nextn.shared_head_head` tensors are
    `TENSOR_NOT_REQUIRED` and are not written.
  `n_embd_head_k = n_embd / n_head = 8` and `n_embd_k_gqa = n_embd_v_gqa = 8`
  (one kv head); the fixture writes `attention.key_length`/`value_length`
  explicitly so no default can move them.
  The trunk per-layer names are the `llama-arch.cpp` mappings, not the
  graph names: `blk.N.attn_qkv.weight` `[n_embd, 2*key+value]`,
  `attn_gate.weight` `[n_embd, value]`, `ssm_conv1d.weight`
  `[conv, 2*key+value]`, `ssm_dt.bias` / `ssm_a` `[ssm_dt_rank]` (`ssm_a` is
  `LLM_TENSOR_SSM_A_NOSCAN`), `ssm_beta` / `ssm_alpha` `[n_embd, ssm_dt_rank]`,
  `ssm_norm.weight` `[head_v]`, `ssm_out.weight` `[value, n_embd]`, the eight
  `hc_*` mixers, `ffn_gate_inp` `[n_embd, 512]`, `ffn_down_exps`
  `[n_ff_exp, n_embd, 512]`, `ffn_gate_exps` + `ffn_up_exps`
  `[n_embd, n_ff_exp, 512]` (or the merged `ffn_gate_up_exps`
  `[n_embd, 2*n_ff_exp, 512]`), the four `ffn_*_shexp`, and on the NextN block
  `nextn.enorm` / `nextn.hnorm` / `nextn.eh_proj`;
- plus the expert/HC/SSM/indexer/rope metadata the loader requires.

The SSM dimensions are taken from `src/models/qwen4exp.cpp:319-325`, not
guessed: `head_k=head_v=ssm_d_state=2`, `n_k_heads=ssm_n_group=1`,
`n_v_heads=ssm_dt_rank=2`, so `key_dim=2`, `value_dim=head_v*n_v_heads=4`,
`conv_dim=2*key_dim+value_dim=8`; `attn_qkv` is `[8, 2*2+4]=[8, 8]`,
`ssm_conv1d` `[2, 8]`, `ssm_out` `[4, 8]`, `ssm_norm` `[2]`, and the recurrent
state `n_embd_s() = ssm_d_state*ssm_d_inner = 2*4 = 8` (with
`ssm_d_inner = 4 = 2*ssm_dt_rank`). Every one of these is what the loader's
`create_tensor` calls compare against; the dimensions are derived from the
same formulas in the generator, not written as literals.

Contract check: when the fixture is configured inside this tree
(`examples/CMakeLists.txt` sets `TP5_FIXTURE_CONTRACT`), the generator runs
`router-fixture-contract.cpp` first. That unit instantiates the real
`llama_model_qwen4exp`, calls its own `load_hparams` / `load_tensors` on the
file through a real `llama_model_loader` (no data read: `no_alloc`), then
verifies the model-level tensors, every per-layer tensor listed above,
`rope.dimension_sections` length 4, `compress_ratios`/`recurrent_layers`
length = block count, `expert_count == 512`, `expert_used_count == 10`, and
that every stored tensor is non-empty F32. Problems are printed as
`FIXTURE contract_problem <text>` and make the generator exit non-zero, so a
wrong name, shape or array length is named at generation time instead of
surfacing as a loader exception later. A stock tree that merely copies this
directory does not build the checker (it needs `src/` headers).

Shrunk (only hidden widths):

| key | fixture | production |
|---|---|---|
| `embedding_length` | 8 | 2560 |
| `hyper_connection.low_rank` | 2 | 320 |
| `expert_feed_forward_length` | 2 | (full) |
| `expert_shared_feed_forward_length` | 2 | |
| `ssm.inner_size / state_size / time_step_rank / group_count / conv_kernel` | 4 / 2 / 2 / 1 / 2 | 10240 / 128 / 48 / 16 / 4 |
| `attention.head_count(_kv)` | 1 / 1 | |
| `context_length` | 64 | |
| `vocab_size` | 128 | |
| `block_count` | 3 | 48 + 1 |

`n_embd = 8` is the floor: `f32` `MUL_MAT` needs `ne[0]` a multiple of 8, and
`rope.dimension_sections` is `[2,2,2,2]` so the (unused) rotary width divides.
The file is ~1.3 MB.

Determinism: all bytes come from one xorshift stream seeded with `--seed`
(default 20261031); two runs produce the same file.

Ownership: the tensors are created in an owning (`no_alloc = false`)
`ggml_context` sized for the model bytes, filled in place, and `gguf` only
stores the pointer (`gguf_set_tensor_data`), so the context outlives the last
`fwrite`. The writer checks before opening the file (nothing is open at that
point, so a failure only releases the two contexts) that the gguf layout is
contiguous (`GGML_PAD(ggml_get_meta_size, ggml_get_alignment)` then every
tensor at its recorded offset) and that the summed tensor bytes equal the
planned size, so a shifted or truncated model fails loudly instead of being
written.

### 2.1 Blocked: lowering the contract further

The contract **cannot** be shrunk further without patching the production code
in `src/models/qwen4exp.cpp`:

- `build_layer_ffn:1502-1504` hard-codes `n_expert == 512 && n_expert_used == 10`.
  Any file with fewer experts/selected experts never enters the retention block
  (`retain_router` is false), so it cannot exercise `GGML_TENSOR_FLAG_NO_REUSE`,
  the plan audit, or the fusion predicate at all. A smaller-router fixture would
  be a different (dense-MoE) code path and would not verify this candidate;
- `1505/1527-1539` require exactly one `SOFT_MAX` of shape
  `[512, rows, 1, 1]` with a direct, contiguous, non-view F32 `MUL_MAT` input
  per layer; that is why the fixture keeps the router at full width and keeps
  `n_embd` a multiple of 8 instead of dropping to, say, 4.

Everything else *is* already at its floor. Reaching a router of, e.g., 8
experts/top-2 would need the production gate itself changed — which changes the
behaviour under test — so it is out of scope for a verification fixture.

### 2.2 Not covered

- The **full-attention** block is not built by this fixture: the interval is 8
  and there are two trunk layers, so both are gated-delta-net layers. That is
  deliberate — a full-attention trunk layer needs
  `hparams.n_head() * n_embd_head == n_embd` because `graph()` builds the
  initial residual with a 2-D reshape of a 3-D `[n_embd, 1, n_tokens]`
  tensor (`src/models/qwen4exp.cpp`, `res_hc`), which aborts for the 1-head,
  `n_embd=8` fixture. The same reshape is why the 2-token leg only works with a
  single head (`n_embd_head == n_embd`). Under `GGML_TP5_MOE_ROUTER_SCRATCH`
  nothing in the attention path is touched, so this does not weaken the CPU
  claim; it does mean the fixture cannot serve a future full-attention
  retention check without raising `n_embd` to `n_head * n_embd_head`.
- The **QSA indexer** path is inert for this fixture: `compress_ratios` is 0
  on every block, so `build_layer_attn` takes its dense branch
  (`qsa = get_idx() != nullptr && compress_ratios[il] > 0` is false). The
  indexer tensors are present only because `load_arch_tensors` asks for them
  on full-attention layers, and there are none here.
- The **NextN** block is loaded (so the checkpoint is complete) and its MoE
  tensors exist, but the decode graph is the trunk graph, so no MTP block is
  executed. If a run ever enables `embeddings_nextn`, the same 2-D `res_hc`
  reshape applies.
- CUDA/HIP: no device code is exercised. The fusion dispatch
  (`ggml/src/ggml-cuda/ggml-cuda.cu:4467`) and cross-warp lifetime are still
  only provable on 13022 (see `ROUTER_SCRATCH_VERIFY.md` §3).

## 3. What the verifier does

One binary, both legs, the environment decides the behaviour
(`GGML_TP5_MOE_ROUTER_SCRATCH`, `GGML_TP5_ROUTER_SCRATCH_AUDIT`,
`GGML_TP5_ROUTER_SCRATCH_DEBUG`; only the exact string `"1"` enables).

**L1 model leg** — `llama_decode` on the fixture with 1, 2, 3 and 4 rows
(`--rows`, default `1,2,3,4`). Retention needs 2..4 rows, so 2/3/4 fire and the
1-row leg must not. Every row's logits are hashed (FNV-1a over the `n_vocab`
floats) and printed:

```
VERIFY_ROW rows=<n> row=<i> sha=<hex>
VERIFY_ROWS rows=<n> decode=0 finite=1 top1=<id> sha=<hex>
```

**L2 plan leg** — one trunk layer's MoE rebuilt node-for-node, with the
production Mixtral layout: the layer input is `[n_embd, 1, rows]`
(`llama-graph.cpp:2203`), the router projection runs on its flattened
`[n_embd, rows]` view ("`MUL_MAT [n_embd,512] -> SOFT_MAX -> TOP_K`"), ids are
`[10, rows]` and `MUL_MAT_ID` gets the 3-D input so `ids->ne[1] == x->ne[2]`
holds, followed by the shared expert;
run twice: once with the router projection marked `GGML_TENSOR_FLAG_NO_REUSE`
(the model's mark), once without. It reserves a real `ggml_gallocr` plan,
audits physical pointers, computes the graph and hashes the output:

```
VERIFY_PLAN retained_ptr=<p> bytes=<512*rows*4> view_src=0 no_reuse=1
VERIFY_PLAN rows=3 reserved=1 audit_clean=1 overlaps=0 reuses_retained=1 retained_bytes=6144 ...
VERIFY_PLAN_SUMMARY retain=1 rows=3 reserved=1 audit_clean=1 overlaps=0 retained_bytes=6144 other_allocs=7 computed=1 sha=<hex>
```

`reuses_retained=1` means some other node shares the retained block — the
in-place top-k route. That is the real consumer: the fused `topk_moe` reads
`logits` and writes `weights`, so with the same layout ggml's in-place pass
continues it in the retained storage instead of overwriting it later.
`overlaps=0` is the disjointness claim. The checks are:

1. **OFF ≡ OFF**: `--rows` hashes identical with all knobs unset vs all three
   knobs `=0`.
2. **OFF ≡ ON**: model-logits hashes identical OFF vs ON. OFF sets no flag, so
   the graph and plan must be bit-identical; a difference is a finding.
3. **Real-consumer hit**: `router_scratch retain layer` lines appear in ON (2,
   one per trunk block, for rows 2/3/4) and zero in OFF.
4. **N1 untouched**: a 1-row-only ON run prints zero retain lines.
5. **Plan**: ON shows no `OVERLAPS`, `retained_bytes >= 512*rows*4`,
   `computed=1`, OFF shows no overlap either.
6. **gallocr audit** (`AUDIT=1`): no `retained tensor '...' overlaps` error.

## 4. Commands (Devops, serial, CPU only)

```sh
src=/home/kunweiz/work/merge-router-20261031
build=$src/build-router-cpu
cmake -S "$src" -B "$build" -DGGML_CUDA=OFF -DGGML_HIP=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
/home/kunweiz/.local/bin/build-slot cmake --build "$build" -j2 \
      --target llama-tp5-router-fixture llama-tp5-router-verify

$src/examples/tp5-router-verify/run-router-cpu.sh "$build/bin" \
      /home/kunweiz/work/lead-router-verify-20261031/out-fixture
```

`run-router-cpu.sh` scrubs the three knobs on every invocation, runs fixture →
OFF(unset) → OFF(=0) → ON, then prints `CRITERIA MET` or `FAIL` with the
evidence lines. Stop on build-slot exit 75/124.

Prerequisite: the fixture generator and the verifier need the candidate tree
only. For an OFF-vs-**pristine stock** comparison, build the stock tree at
`4496493dd` with the *same* two new sources copied in (they touch no model
API), then diff the `VERIFY_ROW(S)` lines: the fixture file is identical
(byte-for-byte, same seed), so the hashes are directly comparable.

## 5. Change to `ggml/src/ggml-alloc.c` (needed for a sound audit)

The first audit walked every node whose `dst.no_reuse` flag was set. That is
unsound because `ggml_tensor` flags are sticky: `ggml_top_k` overwrites its
`NO_REUSE` logits input in place (same shape, in-place legal), the child then
keeps the parent's flag, and the audit reported a false
`retained tensor 'ffn_moe_logits' [TOP_K] overlaps ...`. The audit now follows
`dst.no_reuse` (copied at reserve time) and recognises the one legitimate
sharing — an in-place continuation, i.e. the same storage as a `NO_REUSE` parent
that itself is the earlier retained producer — and audits the storage once, at
the node that first owns it. Any other overlap still fails the reserve (this is
the `reserve` return value, so `llama_decode` fails loudly). A retained tensor
that is a view is refused outright.

The audit lives after `ggml_gallocr_hash_get` (all helpers it calls precede
it), so no forward declaration is involved.

Also changed: `dst.size_max` now records the aligned extent the plan actually
reserves (what `dyn_tallocr` hands out and what `ggml_gallocr_init_tensor`
compares against), so the audit compares reserved blocks, not tensor sizes.

Neither change alters an OFF plan: with no flag set, `dst.no_reuse` is false for
every node and the audit returns immediately.

## 6. Retracted: there is no missing `QWEN4EXP` rope case

Earlier revisions of this document claimed `llama_model_rope_type` lacked a
`LLM_ARCH_QWEN4EXP` case, and a patch adding one was produced. **That was
wrong and is withdrawn** (the patch file is deleted): both trees already list
`case LLM_ARCH_QWEN4EXP:` in the `LLAMA_ROPE_TYPE_IMROPE` group
(`src/llama-model.cpp:3267`), and applying the patch would have been a
duplicate case. Devops' build proved it.

The real cause of the rev8 abort was in the **contract driver**, not in the
model loader:

- `llama_model_create(arch, params)` (`src/llama-model.cpp:343`) returns
  `llama_model_mapping(arch, params)` and then stamps `model->arch = arch`.
  That is the only place the arch is set; `llama.cpp:316` goes through it.
- Every arch model (including `llama_model_qwen4exp`,
  `src/models/models.h:2243`) constructs via
  `llama_model_base -> llama_model(params)`, which leaves
  `arch = LLM_ARCH_UNKNOWN` (`src/llama-model.h:609`).
- The checker constructed `llama_model_qwen4exp model(mparams)` **directly**,
  so `arch` stayed `LLM_ARCH_UNKNOWN`, `load_hparams` ran on it and its last
  line `hparams.rope_type = llama_model_rope_type(this)`
  (`src/llama-model.cpp:1669`) took the `LLM_ARCH_UNKNOWN: GGML_ABORT` branch.
  That is the `llama-model.cpp:3281: unknown architecture` in the rev8 log.

Fix, in the driver only: the checker now creates the model with
`llama_model_create(LLM_ARCH_QWEN4EXP, mparams)` — the same entry point the
production load path uses — then fails closed if the factory returned null, if
`model->arch` is not `LLM_ARCH_QWEN4EXP`, or if the type is not
`llama_model_qwen4exp`. It then runs the loader passes in the production order
(`src/llama.cpp:337-356`): `load_hparams` -> `load_vocab` -> `load_stats` ->
`load_tensors`. `load_vocab` is what reads `qwen4exp.vocab_size`; skipping it
left `n_vocab = 0` and made `load_arch_tensors` ask for `token_embd [8, 0]`
(rev9 evidence: `lead-router-rope-verify-20261031/VERIFY-ROUTER-REV9-CONTRACT.txt`).
Every pass is gated on the previous one, exceptions become `hparams:` /
`vocab:` / `tensors:` `contract_problem`s, an empty vocabulary is its own
failure, and nothing is carried over from a failed pass. Teardown is one path:
the model is deleted, then the gguf context it was built for. The production
rope switch is untouched.

## 7. OFF/stock comparison baseline

- **Stock reference**: `4496493dd` (`git archive 4496493dd`, or the existing
  `/home/kunweiz/work/lead-router-verify-20261031/stock` copy). No shared
  prerequisite patch: the rope case exists in both trees.
- **Candidate**: this tree as-is; deltas of record vs stock are
  `ggml/src/ggml-alloc.c`, `src/models/qwen4exp.cpp`,
  `ggml/src/ggml-cuda/topk-moe.cu`, `examples/CMakeLists.txt` +
  `examples/tp5-router-verify/`.
- **Fixture file**: produced by the same source with the same `--seed` in both
  builds, so the GGUF is byte-identical and every `FIXTURE contract ...` /
  `VERIFY_ROW(S)` line is directly comparable.
- **What OFF must show**: with no knob set, `ggml-alloc.c` sets no flag, so the
  plan is identical to stock and the model leg's rows are bit-identical;
  `run-router-cpu.sh` compares the hashes and prints `FAIL` on any difference.
- **What ON must show**: retention lines only for 2..4 rows, the gallocr audit
  silent, no `OVERLAPS` in the plan leg, `retained_bytes > 0`, and the same
  model logits as OFF (a flag-only change cannot alter arithmetic).
- **L1 needs §8's prerequisite patch applied to both trees** to produce logits at
  all. Both sides carry the same patch, so it cancels out; without it the decode
  aborts and the run fails (no criterion is ever reported as passed).
- **Contract gate**: the in-tree fixture build runs
  `router-fixture-contract.cpp`, which drives the real `llama_model_qwen4exp`
  loader over the just-written file in production order (hparams, vocab, stats,
  tensors). A loader failure — metadata key, array length, empty vocabulary,
  tensor name/shape — is reported as `contract_problem` and the
  generator exits non-zero; it never continues from a half-loaded model and
  never reports success for one. A stock tree that merely copies this directory
  does not build the checker (it needs `src/`).

## 8. Blocker: qwen4exp decode aborts in `llm_graph_input_mem_hybrid::set_input`

The L1 model leg (`llama_decode` on the fixture) aborts in production code
before any compute:

```
ggml_backend_buffer_is_host <- llama_kv_cache::set_input_k_idxs (llama-kv-cache.cpp:2375)
  <- llm_graph_input_mem_hybrid::set_input (llama-graph.cpp:1129-1131)
  <- llm_graph_result::set_inputs <- llama_context::process_ubatch <- llama_decode
```

Read of the call chain:

- `llm_graph_input_mem_hybrid::set_input` unconditionally drives the **attention**
  kv-cache context (`llama-graph.h:648-674`, `inp_attn` typed
  `llm_graph_input_attn_kv *`; `llama-memory.h:143-152` derives that from
  `llama_kv_cache_context`).
- `llama_kv_cache::set_input_k_idxs` hard-asserts `ggml_backend_buffer_is_host(dst->buffer)`
  (and `set_input_v_idxs` / `set_input_kq_mask` likewise at
  `llama-kv-cache.cpp:2398` / `:2647`).
- A qwen4exp decode graph has **no attention layer** here: `is_recr(il)` is true
  for both trunk blocks and the NextN block is outside `n_layer()`, so the
  hybrid memory's layer filters (`llama-model.cpp:2764-2772`) put no layer in
  the attention cache.
  `build_inp_mem_hybrid` (`llama-graph.cpp:3900-3909`) still creates
  `self_k_idxs` / `self_v_idxs` / `self_kq_mask`, the graph never consumes them,
  the scheduler leaves them unallocated, and `set_input` aborts. The same shape
  occurs for the 1-row and the 2..4-row decode; the fixture's own probe and the
  L2 plan leg are unaffected (they never build a llama context).
- Not a fixture-architecture violation that the fixture should paper over: a
  model whose only layers are recurrent (or recurrent + an MTP block) is a legal
  qwen4exp configuration, and the graph builder knows it — `llm_graph_input_attn_kv`
  already guards exactly this for the mask (`if (self_kq_mask && self_kq_mask->buffer)`,
  `llama-graph.cpp:516`), and the file carries the same FIXME at
  `llama-context.cpp:2635`.
- The NULL tensor is the hybrid **attention** input, not the recurrent
  `s_copy` path: only `llm_graph_input_attn_kv` carries a
  `llama_kv_cache_context *` and calls `set_input_k_idxs` on it
  (`llama-graph.cpp:511-512`, reached from the hybrid input's
  `set_input`). The other `set_input_k_idxs` callers are `:565`
  (`llm_graph_input_attn_k`) and `:596` (MSA idx), neither of which a qwen4exp
  graph builds. Devops' rev10f backtrace is the authority for the frame order.
- Production fix, written as its own change in a separate worktree:
  `/home/kunweiz/work/router-hybrid-input-fix-20261031/` (`README.txt`,
  post-fix `llama-graph.cpp` and `llama-graph.h`). The design is uniform, not one
  guard per input class:
  - `llm_graph_result::set_graph_marks()` (called from `set_outputs`, i.e. at the
    end of `build_graph`, **before** `ggml_backend_sched_alloc_graph` and thus
    before pass 5 rewrites `node->src[]` with per-backend copies) marks each
    input used iff the completed graph depends on one of its tensors
    (`graph_tensor_is_dependency`: a node source, or the storage a
    view/reshape/permute/transpose reads). Once per graph result — a rebuild
    recomputes, a reused graph keeps it; nothing scans the graph per decode step;
  - `set_inputs` calls `set_input` only for marked inputs; `llm_graph_input_rs`,
    the three hybrid wrappers and `llm_graph_input_sampling` override
    `graph_used()` to always run, because their set_input also does memory /
    sampler bookkeeping;
  - inside each input a tensor that exists but is unallocated while the input is
    used aborts (fail closed); a tensor the graph does not use is skipped. The
    hybrid inputs additionally feed k_idxs / v_idxs / kq_mask / k_rot / v_rot
    (base and SWA separately for iswa) by their own recorded relation, assert
    allocation when consumed, and feed readers exactly when the attention path
    runs;
  - `mem_hybrid`, `mem_hybrid_k` (MLA) and `mem_hybrid_iswa` are all covered by
    the same relation (the latter two were the earlier gaps);
  - `llm_graph_input_out_ids` / pos_bucket / mean / cls / attn_no_cache /
    attn_cross keep their host-buffer asserts, now reached only when used.
  Apply it to **both** the candidate and the `4496493dd` reference tree as a
  shared prerequisite, so it cancels out in the comparison.

`run-router-cpu.sh` runs the model leg by default (`MODEL_LEG=1`, fail-closed:
any decode failure is a FAIL). `MODEL_LEG=0` is a diagnostic escape hatch only —
the verifier then runs `--no-model`, prints
`VERIFY_MODEL_LEG skipped=1 reason=llm_graph_input_mem_hybrid_set_input_assert`,
and the script exits 3 without ever printing `CRITERIA MET`. It cannot stand in
for the real L1 result.

## 9. Retention evidence: which leg can prove what

**Placement comes first.** `ggml_gallocr_alloc_graph_impl` allocates every non-view
`NO_REUSE` node (and such a node's sources) before the first temporary, and
`ggml_gallocr_free_node` never frees it. Without that, a retained router projection
was placed at an address a temporary (`node_9`) had already been freed from — legal
for the dyn_tallocr, an overlap for the contract — and the audit then refused the
plan, so the audit's own record could never describe a plan the decode ran.

**Empty tensors are not unverifiable neighbours.** A tensor with any `ne[] == 0` has
no bytes and no address range, and every backend's kernel is a no-op for it, so it
cannot overlap a retained extent. The audit skips such a neighbour instead of
counting it as "in-domain with no recorded size". A real Qwen4Exp decode graph
carries several (the `states_extra` nodes, shape `[state_size, 0]`, built when
`n_rs == n_seqs`); counting them refused the plan outright, which is why no clean
line ever appeared for the 303-node graph. The retained tensor itself may never be
empty.

**The join key is `(graph_uid, plan_id)`.** `ggml_gallocr_check_no_reuse_plan` runs
inside `ggml_gallocr_reserve_n_impl`, i.e. on the reserve path llama takes, over the
complete node + leaf plan of `sched->graph` including every cross-backend copy, with
no callback and no extra synchronize. With `GGML_TP5_ROUTER_SCRATCH_AUDIT=1` it
records the identity of the plan it judged and prints it on the evidence line:

| Line | Meaning |
|---|---|
| `retention audit clean (reserve): N retained allocations, B bytes, rows=R, nodes=M, graph_uid=U, plan_id=P, fp=F` | this plan was audited; none of its retained extents overlaps another node or leaf allocation. `entry` is `reserve` for `ggml_gallocr_reserve_n` and `size` for `ggml_gallocr_reserve_n_size` (a measurement that allocates nothing); `rows` is the common `ne[1]` of the retained set (`mixed` if they disagree); `U` is `ggml_cgraph.uid` of the planned graph, `P` the planning pass on that allocator, `F` the FNV-1a of its node array |
| `retention audit enabled (reserve): 0 retained allocations, nodes=M, graph_uid=U, plan_id=P, fp=F` | the audit ran and vouched for nothing |
| `retained tensor '…' overlaps '…' / overlaps leaf '…' / is a view, it cannot be retained` | contract violation; the reserve returns false and the plan is never run |

`llama_context` supplies the other half, and it does not use `graph_reserve` for
this: the plan a decode submits is produced by `ggml_gallocr_reserve_n` reached from
`ggml_backend_sched_alloc_graph` inside `process_ubatch`, not by the sizing reserve.
So the owner records at the allocation and checks at the submission:

- the allocator keeps, next to the live plan identity, the identity of the last plan
  the retention audit **accepted with retained allocations** (`audit_clean_plan_id`,
  `audit_clean_graph_uid`, `audit_clean_fp`, node/leaf counts) plus `retained`, the
  retained count of the plan now held. The record is written only by such an audit
  and is **kept** across later passes: every planning pass advances `plan_id`, so a
  refused or unaudited pass separates the two ids by itself. A plan with zero
  retained allocations (the 1-row and knob-off legs) never updates the record, so
  nothing there can be reported as an audited retained plan.
- `process_ubatch`, right after `ggml_backend_sched_alloc_graph` succeeds, records
  the allocator's `plan_id`/`n_nodes` and the graph's `uid`/`fp` (`record_exec_plan`).
- `graph_compute`, after the scheduler call returns and for **every** submission
  (a CPU context has no relay backend, so a retained-input-only condition would
  never print there), prints:

```
llama_context::graph_compute: router_scratch plan identity exec_uid=U exec_fp=F exec_nodes=M plan_uid=U plan_fp=F plan_id=P plan_nodes=M retained=N audit_plan_id=P audit_uid=U audit_fp=F recorded_plan_id=P recorded_uid=U recorded_fp=F entry_size=0 match=<0|1>
```

`match=1` requires all of: a record taken at allocation; a non-zero `plan_id` and
`graph_uid`; `retained > 0` (the plan carries the router scratch this gate exists
for); `audit_plan_id == plan_id` with the audit's uid, fp and node/leaf counts equal
to the live plan's (the accepted plan is the plan held); the submitted
graph's non-zero uid and its fingerprint equal to the allocator's, with node/leaf
counts equal to `sched->graph`'s; a `(reserve)` entry rather than a `(size)`
measurement; and the allocation-time record equal to the live identity (a plan
replaced between allocation and submission gives `match=0`). The uid the allocator
sees is `sched->graph.uid`, which `ggml_backend_sched_split_graph` copies from the
graph it received (`ggml_graph_cpy` does not carry a uid, and the split views keep
uid 0). C4b requires `match=1` on the ON leg **and** that `exec_uid`/`plan_id`/`fp`
equal the last clean `(reserve)` line of the **scheduled** graph — a line with
`graph_uid=0` is the L2 synthetic plan leg, which runs after the model leg and writes
to the same stderr — `graph_uid`/`plan_id`/`fp`, **and** that the
line's `audit_plan_id` equals its `plan_id`; a missing identity line, `match=0`, a
mismatched identity, an unaudited later plan, or no well-formed clean line stays
UNPROVEN and fails. The gate also requires `retained >= 2` (both trunk MoE layers)
and the clean line's own retained count to equal the identity line's, and it grades
**every** submission of the ON leg, not just the last: each multi-row decode
(`--rows 1,2,3,4` → 1 row unretained, then 2/3/4 rows retained) must report
`match=1`, the number of `match=1` lines must equal the number of `retained>0` lines
and be at least 3, and no line may report `retained=-1` (a plan that was never
audited). On the OFF and 1-row legs no identity line may report `match=1` or a
positive `retained`.

A pass that is refused keeps the previous accepted record, so the identity line of a
submission that used a refused plan reports `audit_plan_id != plan_id` and
`match=0`: the plan in the buffers is not one the audit vouched for.

Why the uid alone would not do: `ggml_backend_sched_split_graph` stamps the graph it
receives, so a re-reserve of the same graph object keeps its uid while the plan is a
different one. `plan_id` moves on every planning pass, so the pair separates "same
graph" from "same plan". Why the `(size)` entry is excluded: it measures, it does
not reserve for execution.

Scope of the claim, stated so it is not overread: this proves the plan the
allocator holds at submission is the plan it audited, for the graph `llama_context`
submits with retained inputs. It does not prove kernel dispatch (GPU fusion needs
its own line) and does not cover Meta backend sub-graphs, which carry their own
uids and no allocator plan.

The earlier design sketch in `HANDOFF-ROUTER-CPU-FIXTURE-REV10.txt` rev10x (a
"phase" tag on the gallocr) remains withdrawn: it declared an intent instead of
recording an identity. This route records the identity the allocator actually
produced and has the owner read it back.

**What is not a failure.** A `(size)` measurement reserve has the same graph
structure as the plan it measures, so the retention marks there are legitimate; a
`clean (size)` line is recorded and never graded — it simply cannot vouch for the
executed graph. In the same way, a missing zero line on the OFF or 1-row leg, a
`rows=mixed` clean line, or a clean line on a leg where none is expected each mean
"this run says nothing about that statement", i.e. UNPROVEN, not a defect. A
contract violation (`retained tensor … overlaps …`, view-as-retained) is a hard
failure regardless, because the allocator refused the plan.

**`cb_eval` is observation only, never evidence.** `tp5-router-scratch-diag.h`
installs `cparams.cb_eval` behind `GGML_TP5_ROUTER_SCRATCH_DIAG=1` and reports
`RETAIN`/`DISJOINT`/`OVERLAP`/`STEP`/`summary`. Three limits are real, all
confirmed in source:

- a non-null `cb_eval` disables the single-split `defer_input_flush` fast path
  (`ggml/src/ggml-backend.cpp:1980`) and the callback branch synchronizes the
  split backend per batched range, so the run is **not** production timing;
- the callback sees `split->graph` nodes only; cross-backend input copies live in
  `sched->graph`, so `DISJOINT` means "disjoint from the executed nodes", not
  "disjoint from every allocation in the plan";
- the `tp5_relay` path rejects a non-null `cb_eval` outright
  (`src/llama-context.cpp:216-218`), so the observer does not exist there at all.

Its counts (including the 2 layers × 4 rows = 8 shape) are printed as `triage:`
and may be used to explain a failure by layer name and step, but they cannot pass
or promote anything, and `overlaps=0` there must never substitute for the audit
line.

**L2 synthetic plan (supporting evidence only).** It builds one layer's MoE by hand
with the real allocator API — `ggml_gallocr_reserve_n_size` to measure, then
`ggml_gallocr_reserve_n` + `ggml_gallocr_alloc_graph`. It audits `pg.graph->nodes`,
mirrors `build_moe_ffn`'s consumer set (`ids` straight from `ggml_top_k`), and treats
**any** sharing of the retained extent — exact or partial — as FAIL. A clean L2 is
necessary and never sufficient on its own.

## 10. Unverified claims## 10. Unverified claims## 10. Unverified claims## 10. Unverified claims

- Nothing in this document has been built or run.
- The fixture loads and reaches `build_layer_ffn` with `cur->ne[1]` 2..4 in the
  *decode* graph: expected, not observed. If a stock decode of the fixture
  fails, `run-router-cpu.sh` prints the `llama_decode` rc and the failing leg,
  and that is the next thing to look at.
- OFF-vs-ON bitwise equality of logits is the expectation for a flag-only
  change; it is checked, not assumed.
- No GPU, no fused `topk_moe`, no `n_rows > 1` dispatch is covered here.
