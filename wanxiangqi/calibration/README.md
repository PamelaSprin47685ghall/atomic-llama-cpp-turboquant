# Wanxiangshu teacher prefix-trie evidence

This directory is the versioned request-token prefix trie used as the finite
calibration population for the published Wanxiangshu ENP source-quant
ablation. It is intentionally committed as raw trie data, not just as a
digest, so the calibration population can be independently traversed and
audited.

The trie was captured by `llama-server --run-dump` using the format documented
in `wanxiangqi/docs/request-corpus-dump.md`.

## Inventory

- unique persisted trie nodes: `270952`
- request records: `42`
- request kind: completion (`1`) for all 42 records
- shortest request leaf: `220` tokens
- longest request leaf: `133136` tokens
- sum of request-leaf lengths: `3487264` tokens
- unique trie population used by ENP: `270952` hidden-state positions

The ENP imatrix population entry is also `270952`, providing a direct
cross-check that calibration traversed every unique live trie node once.

## SHA-256

```text
7f6256c4b5999c1a4fa19e169db24b27900f1e9d56b2ced2e20b390044d34548  format.json
15d10275d4c5f572029f1e662e6af56246985dd7d8867ccd5b9593c4139fe686  nodes-000000.bin
df4c450217eed60b917d6882ae996f2cad389ca4fffdf55c511f88515f44533f  requests-000000.bin
```

The binary files contain raw token IDs. Given the matching tokenizer they can
be decoded back into request text; treat this evidence directory as source
data rather than anonymized statistics.

