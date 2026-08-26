# qwen35-prune

This directory contains the Qwen3.5-MoE / Ornith-1.5-35B-A3B compression
tooling used in this repository.  The project is now split into two explicit
release targets so that GDN research does not block a usable test build.

## Reproduce the published Wanxiangshu ENP source-quant artifact

The exact request-token prefix trie used for the published experiment is
versioned under `evidence/wanxiangshu-teacher-trie-v1/`. Given the matching
teacher GGUF and BAOMU audit corpus, the complete calibration -> plan -> apply
-> verify/finalize pipeline is one command:

```bash
wanxiangqi/qwen35-prune/scripts/reproduce-wanxiangshu-enp-sourcequant.sh \
    /path/to/Ornith-1.5-35B-A3B-Q4_K_M.gguf \
    /path/to/BAOMU.md \
    /path/to/output-directory
```

The script pins input/trie hashes and fails unless the final GGUF SHA-256 is
`7d8b84d27aa9f3c92ddd494388f0d7e4e2c7a13933793be4b15159cdaea61200`.
See `EVIDENCE_WANXIANGSHU_ENP_SOURCEQUANT.md` for the provenance and validation
record.

The published v2 artifact intentionally remains reproducible with its original
512-position uniform-without-replacement ENP calibration. New `--prune-enp`
runs use a task-blind streaming geometric medoid coreset instead; the
reproduction script passes `--prune-enp-legacy-uniform` explicitly so the
historical byte hash does not change.

### Geometric ENP coreset

The default ENP collector traverses every calibration hidden state once and
maintains at most 1024 actual hidden-state medoids per layer. Coreset formation
does not inspect task labels, router frequency, Q2 error, or any downstream
surrogate. Each cluster stores its population weight plus conservative
original-2048D L2 transport and radius bounds. The planner evaluates the
unaltered standard ENP projection score on the representatives and tries nested
64/128/256/512/1024-point coarsenings until the selected-vs-pruned confidence
intervals separate.

Geometric-plan records report `enp_top256_certified`, transport/radius
statistics, the adaptive sizes attempted, and
`enp_full_population_fallback_required` when the stored coreset cannot certify
the 256/257 boundary. Such a record is deliberately not described as having a
unique certified survivor identity. Production `apply`/`finalize` refuse to
materialize a geometric expert-pruned model while any main-model expert remains
unresolved, so source-index tie-breaking can never silently turn an approximate
diagnostic plan into a published model.

The certificate uses a local Lipschitz bound where an entire coreset cell stays
away from zero expert output. If that local directional bound is unavailable,
the planner falls back to the global ENP score magnitude bound for that cell,
so intervals stay finite and rigorous rather than relying on the numerical
`1e-8` denominator. An exact full-population run distinguishes a genuine
256/257 tie (`enp_exact_boundary_tie`) from an approximation that merely needs
more representatives.

`--prune-enp-coreset-points N` raises or lowers the stored maximum. If an
expert remains unresolved, rerunning calibration with a larger `N` tightens
the geometric approximation; setting `N` at least as large as the calibration
population makes every hidden state a medoid, gives zero transport/radius, and
therefore reduces the planner to exact full-population standard ENP for the
final fallback. A future targeted second traversal can make this cheaper by
re-evaluating only unresolved experts; the current implementation deliberately
requires an explicit larger/full-population calibration rather than pretending
that discarded hidden states can be recovered from the coreset.

## Release split

### 1.0 test target: expert + vocabulary compression, GDN unchanged

The 1.0 test line compresses only the parts that are already sufficiently
engineered for external evaluation:

- routed-expert width: `512 -> 256`;
- vocabulary: variable-size Chinese/English-preserving language filtering (no fixed target size);
- hidden size: unchanged at 2048;
- **Gated Delta Net (GDN): unchanged**;
- GDN state/head dimension remains `128`;
- GDN inner dimension remains `4096`;
- stock `qwen35moe` inference architecture is retained.

For 1.0, every tensor classified as GDN is copied byte-for-byte from the
teacher.  In the current model this is 150 tensors.  GDN compression is not a
1.0 release criterion and must not delay 1.0 testing.

The current BAOMU-full 1.0 test artifact is documented in
[`RELEASE_1_0_TEST.md`](RELEASE_1_0_TEST.md).

Current external-test artifact:

```text
/home/kunweiz/models/Ornith-1.5-35B-A3B-BAOMU-v1.0-KEEPGDN.gguf
SHA256 5f78c258708f97448088172775ac4b3b0b215bd48586203a0887d1afea4ba6cd
```

The local held-out smoke for this exact payload is NLL `2.6563` / PPL
`14.2435` over 95 next-token targets.  This is a test-build gate, not a broad
OOS quality claim.

### 2.0 target: GDN 128D -> 64D

GDN compression is the 2.0 target:

- 16 Q/K heads: `128 -> 64`;
- 32 V heads: `128 -> 64`;
- GDN state dimension: `128 -> 64`;
- GDN inner dimension: `4096 -> 2048`;
- stock `qwen35moe` graph must still be usable; no custom inference operator.

2.0 may use calibration, but calibration is treated conservatively.  The
current design constraints are:

- prefer one teacher calibration pass plus `o(1 model forward)` local work;
- Q, K and V must be optimized as a coupled recurrent system rather than as
  three independent truncations;
- low-dimensional/shrunk statistics are preferred over high-capacity fitting;
- local recurrence replay is allowed because it is cheap relative to a full
  model forward;
- held-out perplexity is a required quality gate;
- out-of-sample/domain stability is required; a method that wins only on its
  calibration text is rejected;
- high-freedom interpolation, repeated whole-model student replay and broad
  backpropagation are not first-line methods.

The older full-domain theory note is retained as useful 2.0 background in
[`GDN_UNIFORM_APPROXIMATION_PROBLEM.md`](GDN_UNIFORM_APPROXIMATION_PROBLEM.md),
but a uniform theorem is no longer a prerequisite for shipping 1.0 or for
experimenting with 2.0.

## 1.0 engineering path

The relevant production/test path is deliberately small:

```text
plan                    # expert/vocab plan from imatrix
apply --keep-gdn        # expert/vocab rewrite; exact-copy all GDN tensors
score-ppl               # single-model held-out NLL/PPL smoke/audit
audit-tokenizer         # tokenizer audit
```

The `apply --keep-gdn` switch is the defining 1.0 behavior.  It preserves both
GDN metadata and GDN tensor payloads:

```text
qwen35moe.ssm.state_size = 128
qwen35moe.ssm.inner_size = 4096
```

and leaves the teacher shapes such as:

```text
attn_qkv.weight     [2048, 8192]
attn_gate.weight    [2048, 4096]
ssm_conv1d.weight   [4, 8192]
ssm_norm.weight     [128]
ssm_out.weight      [4096, 2048]
```

unchanged.

## Expert compression

Routed experts are reduced from width 512 to 256 with the Q2-aware planning
and writer path.  The planner records routing coverage, selection swaps and
packing swaps, while the writer records quantization error statistics and
payload hashes.

The current BAOMU-full **test** plan deliberately permits uncovered experts.
This is why the 1.0 artifact is called a test build rather than a final
release.  The full BAOMU routing audit reports 253 zero-routing experts across
the complete 41-block inventory, with most of the zeros concentrated in the
MTP/next-token-prediction block.  Such experts use the tool's deterministic
weight-only fallback.  External OOS testing is therefore important before a
final 1.x release is declared.

## Vocabulary compression

The vocabulary path no longer targets a fixed token count. It keeps tokens that
are Chinese, English/Latin-script, ASCII/code/protocol syntax, neutral Unicode
(symbols, math, emoji), special/control tokens, and the complete 256-byte
fallback alphabet. It removes only tokens whose decoded text is confidently in
a non-Chinese/non-English writing system, then restores any removed token that
is required as a BPE ancestor of a retained token. Ambiguous tokens are kept.
The final vocabulary size is therefore data-independent and is recorded in the
plan/manifest rather than forced to 36,096.

## Integrity and reproducibility

The tool records source/imatrix hashes, file identities, tensor payload hashes
and an output manifest.  Expert planning/apply operations support checkpoints.

For a 1.0 test artifact, the essential release checks are:

1. stock llama.cpp/qwen35moe can load the generated GGUF;
2. model metadata reports GDN `128/4096`;
3. all 150 GDN payload hashes equal their source hashes;
4. expert width is 256 and the variable pruned vocabulary size matches the plan/manifest;
5. tokenizer metadata is internally consistent;
6. a held-out PPL smoke completes with finite NLL/PPL;
7. model SHA-256 and manifest are supplied to the tester.

## 2.0 GDN experimental code

The directory also contains GDN research code.  It is not part of the 1.0
release gate.  Current components include:

- `gdn-geometry-plan.{h,cpp}`: shrunk second-moment geometry and deterministic
  top-k/pivoted-Cholesky candidate construction;
- `gdn-joint-replay.{h,cpp}`: local 64D recurrence replay for synchronized QKV
  candidate scoring;
- `gdn-function-projection.{h,cpp}`: empirical function/readout fitting;
- `gdn-observable-risk.{h,cpp}`: teacher-KL/NLL diagnostics;
- `gdn-adjoint.{h,cpp}`: gradient/adjoint experiments.

Important empirical finding from the current 2.0 work: independently selected
Q/K and V subsets can each improve held-out perplexity when changed alone yet
degrade it when combined.  Consequently the 2.0 optimizer must treat QKV as a
coupled reduction problem.  This is one reason GDN compression remains a 2.0
research target rather than being forced into 1.0.

The current `ssm_out` writer for 2.0 supports arbitrary V coordinates by
dequantizing the teacher row, gathering the requested columns and requantizing
with the stock tensor type.  The verifier reproduces that operation and
reports the induced numerical error.  This correctness work remains useful to
2.0 but is irrelevant to 1.0 because 1.0 exact-copies `ssm_out`.

## CLI surface

The binary is `build/bin/llama-qwen35-prune`.  Important commands include:

```text
inspect
plan
apply
score-ppl
audit-tokenizer
verify
finalize

# 2.0 / GDN research
plan-gdn-geometry-v1
plan-gdn-joint-v1
rewrite-gdn-inplace
score-gdn-requantization
verify-gdn-layer
fit-gdn-function-projection
rewrite-gdn-function-projection
select-gdn-function-projection-global
select-gdn-qk-atom-exchange-global
score-gdn-function-risk
```

The GDN commands above are experimental and must not be treated as 1.0 release
requirements.

## Focused build/tests

From the repository root:

```bash
cmake --build build --target \
    llama-qwen35-prune \
    test-q2-opt-signed \
    test-q2-opt-backend \
    test-gdn-geometry-plan \
    test-gdn-joint-replay
```

For 1.0, Q2/expert, tokenizer, stock-load and PPL checks are release-relevant;
GDN tests validate 2.0 research code only and do not block 1.0 delivery.
