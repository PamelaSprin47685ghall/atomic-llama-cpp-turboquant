# Wanxiangqi fork extensions

This directory contains the Wanxiangqi model pruning, ENP optimization, and server token dump extensions.
The goal is to keep upstream-owned files as clean as possible: code in `common/`, `tools/`, and `tests/` contains only the minimal registration or call-site hooks required.

## Layout

- `common/`: fork-only common parameter/CLI extensions and speculative runtime.
- `server/`: persistent request-token prefix trie and hybrid slot-memory fix.
- `tests/`: tests for Wanxiangqi server extensions.
- `docs/`: fork-specific format documentation.
- `qwen35-prune/`: Qwen3.5/Ornith ENP, pruning, GDN experiments, tests,
  publication evidence, and reproducibility scripts.

## Published Wanxiangshu ENP source-quant artifact

The exact prefix trie used for calibration is committed under
`qwen35-prune/evidence/wanxiangshu-teacher-trie-v1/`.  It contains 270,952
unique trie nodes and 42 request records.

Given the pinned teacher GGUF and BAOMU corpus, regenerate the final artifact
with:

```bash
wanxiangqi/qwen35-prune/scripts/reproduce-wanxiangshu-enp-sourcequant.sh \
  /path/to/Ornith-1.5-35B-A3B-Q4_K_M.gguf \
  /path/to/BAOMU.md
```

By default generated files stay under `wanxiangqi/.reproduce-output/`, which
is intentionally ignored by Git.  The script validates the teacher, corpus,
and committed trie hashes and requires the final GGUF SHA-256 to equal the
published value.

