# Wanxiangshu ENP source-quant publication evidence

This note pins the inputs and expected published artifact for the clean
projection-ENP / source-quant ablation.

## Published inputs and outputs

```text
teacher GGUF sha256:
3e13c52d562b1c97998c3cf3954b99f7e8156ca6a1d8b2db76fe0499d53a95c4

BAOMU audit corpus sha256:
01ffbc268ef1395fd65a4bae04ab4282a13bd71e30bdbcb1b1ed8734b2ec266a

original calibration imatrix sha256:
619b2510e4e2500d4617cace3dfeafe5f9050ca94027e130041c5c5af30513a3

original production plan sha256:
ba292e7ba59f2cadf3d884cfe0fc9721a6e9c996fcdceacb9ab3052961d1b3b2

published final GGUF sha256:
7d8b84d27aa9f3c92ddd494388f0d7e4e2c7a13933793be4b15159cdaea61200
```

The versioned trie is in
`evidence/wanxiangshu-teacher-trie-v1/`. The ENP bottom-K sampler now uses the
fixed seed `0x2fe24846bc52ca37`, reconstructed from the seed metadata stored in
the published imatrix, so reruns do not depend on `std::random_device`.

The calibration executable observes Qwen3.5's existing `attn_post_norm`
callback tensor. `build_layer_ffn(attn_post_norm, il)` passes that exact tensor
to `build_moe_ffn()`, so this is the same expert-input hidden state previously
observed through the temporary core `prune_enp_hidden` callback, without a
fork-only llama-core API or graph tap.

The original imatrix/plan file hashes may differ on a fresh run because their
provenance includes paths/tool-build metadata. The reproducibility gate is the
final GGUF byte hash above, together with the structural verifier.

## Published validation

- changed tensors: `123` (`41 x gate/up/down expert tensors`)
- copied tensors: `630`
- expert quantization: source tensor quant type preserved
- Q2 blocks: `0`
- GDN: unchanged (`rel_rms=0` under verifier)
- vocab: unchanged (`248320 -> 248320`)
- tokenizer-domain ratios: all `1.0`, hard-limit PASS
- expert payload verified: `9,978,249,216` bytes
- BAOMU 95-target comparison: teacher PPL `90.3761`, candidate PPL `116.735`
- structured behavior gate: emitted legal `<function=read>` and OMP executed
  `Read .`; no prior `write`/system-prompt-copy/parameter-loop failure

Run `scripts/reproduce-wanxiangshu-enp-sourcequant.sh` to rebuild the complete
artifact from the teacher GGUF, the committed trie, and the BAOMU audit corpus.
That historical reproduction path now passes `--prune-enp-legacy-uniform`
explicitly. The default `--prune-enp` collector has moved to the newer
task-blind geometric-medoid coreset, so research improvements cannot silently
change the published v2 calibration or final byte hash.

After the source isolation into `wanxiangqi/`, a real-model smoke traversal of
the shortest committed request branch (220 tokens) produced ENP samples for
blk.0 through blk.39, `prune.enp.population.blk.0 = 220`,
`prune.token_count = 220`, and seed components
`51767,48210,18502,12258`, matching the seed metadata in the published
calibration file.  The relocated prune verifier also re-verified the published
final GGUF with `verify OK`.

