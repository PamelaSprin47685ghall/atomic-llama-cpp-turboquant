# qwen35-prune 1.0 test build

## Scope

This is the first external-testable 1.0 line.  It intentionally **does not
compress GDN**.  The purpose is to validate the already mature expert and
vocabulary compression paths independently of the unresolved GDN 128D -> 64D
research problem.

The test model has:

| component | teacher | 1.0 test |
|---|---:|---:|
| routed expert width | 512 | 256 |
| vocabulary | 248,320 | 36,096 |
| hidden size | 2,048 | 2,048 |
| GDN state/head dim | 128 | **128 (unchanged)** |
| GDN inner dim | 4,096 | **4,096 (unchanged)** |

Inference remains stock `qwen35moe`.

## Artifact

Canonical generated test model:

```text
/home/kunweiz/models/Ornith-1.5-35B-A3B-BAOMU-v1.0-KEEPGDN.gguf
```

This path is a hard link to the fully generated keep-GDN control artifact, so
it does not consume another 5.53 GiB copy.  Exact file size is
`5940219968` bytes.  The reuse is deliberate: the completed control artifact
has the same expert plan, Q2 statistics, tokenizer merges, protected tokens
and byte-fallback set as the BAOMU full-en/zh keep-GDN build.  Its only
plan-level difference is an experimental GDN record, which is ignored because
`--keep-gdn` is active.  The final GGUF SHA also matches the historical
BAOMU-full keep-GDN manifests.

SHA-256:

```text
5f78c258708f97448088172775ac4b3b0b215bd48586203a0887d1afea4ba6cd
```

Associated files:

```text
/home/kunweiz/models/Ornith-1.5-35B-A3B-BAOMU-v1.0-KEEPGDN.gguf.manifest.json
/home/kunweiz/models/Ornith-1.5-35B-A3B-BAOMU-v1.0-KEEPGDN.gguf.sha256
```

## Calibration provenance

The expert/vocabulary plan uses the BAOMU-full imatrix with 21,504 expected
calibration tokens (`168 x 128` chunks).  The imatrix used by the build is:

```text
/home/kunweiz/models/baomu-full-imatrix.gguf
```

The expert/vocabulary decisions are those of the existing BAOMU-full combined
plan.  The equivalent production command is:

```bash
./build/bin/llama-qwen35-prune apply \
  /home/kunweiz/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf \
  /home/kunweiz/models/baomu-full-imatrix.gguf \
  /tmp/gdn-atom-plan-v2.json \
  /home/kunweiz/models/Ornith-1.5-35B-A3B-BAOMU-v1.0-KEEPGDN.gguf.tmp \
  --threads 16 --keep-gdn
```

`--keep-gdn` is mandatory for the 1.0 line.

## Verified invariants

The generated model was successfully loaded by the current stock qwen35moe
runtime.  Runtime metadata reported:

```text
qwen35moe.ssm.state_size = 128
qwen35moe.ssm.inner_size = 4096
n_vocab = 36096
expert_feed_forward_length = 256
```

The release manifest records:

- `diagnostic_keep_gdn = true`;
- the GDN policy class contains 150 tensors and keep-GDN preserves their
  original type/shape/payload path;
- 35,799 tokenizer merges;
- 289 protected tokens;
- all 256 byte-fallback tokens retained.

Q2 expert encoding statistics also passed the aggregate ordering invariant
`signed <= positive-only <= stock` for weighted SSE.

## Perplexity smoke

The 1.0 test model completed a held-out BAOMU smoke using the single-model
`score-ppl` path:

```text
tokens      = 128
score_from  = 32
samples     = 95
NLL         = 2.6563 nats/token
PPL         = 14.2435
```

The same load reported a 5.53 GiB / 18.13 B-parameter model with GDN
`128/4096`, expert width 256 and vocabulary 36,096.  This is a structural and
quality smoke result, not the final quality claim.  External machines should
run longer and genuinely OOS corpora,
especially code, English, Chinese, math and long-context prompts.

The full BAOMU tokenizer audit also passed the configured priority hard
limits:

```text
byte fallback round-trip = PASS
byte fallback tokens     = 256
BAOMU source tokens      = 21628
BAOMU reduced tokens     = 21632
BAOMU token ratio        = 1.0001849454
zh token ratio           = 1.0000
en token ratio           = 1.0540540541
priority hard limit      = PASS
```

Code/math/tool/unicode probes can expand more strongly under the reduced
vocabulary; those ratios are diagnostics for external OOS testing rather than
release blockers for this BAOMU-oriented 1.0 test artifact.

## Why this is still called a test build

The BAOMU-full plan currently has `allow_uncovered_experts=true`.  Its routing
audit contains 253 zero-routing experts across the complete 41-block
inventory, heavily concentrated in the MTP block.  Uncovered experts use the
deterministic weight-only fallback.  Therefore this artifact is suitable for
cross-machine/OOS evaluation, but it should not yet be called the final 1.x
release.

The main external questions for 1.0 are:

1. Does expert width 512 -> 256 retain quality across domains not present in
   BAOMU?
2. Does the 36,096-token vocabulary behave correctly on multilingual/code
   text and chat templates?
3. Are long-context behavior and generation stability acceptable when GDN is
   untouched?
4. Do any uncovered/rare experts produce clear OOS regressions?

## 2.0 is separate

GDN 128D -> 64D is explicitly the **2.0 target**.  2.0 research may use
one-pass calibration, shrunk statistics, synchronized QKV recurrence replay,
perplexity gates and OOS stability checks.  None of that work blocks the 1.0
test artifact above.
