# Wanxiangqi fork extensions

This directory contains Wanxiangqi fork extensions with minimal upstream hooks.
The goal is to keep upstream-owned files as clean as possible: code in `common/`, `tools/`, and `tests/` contains only the minimal registration or call-site hooks required.

## Layout

- `common/`: fork-only common parameter/CLI extensions and repetition-guard utilities.
- `server/`: persistent request-token prefix trie dump for calibration capture.
- `tests/`: tests for Wanxiangqi server extensions.
- `docs/`: fork-specific format documentation.
- `qwen35-prune/evidence/`: retained calibration data only (prune/recurrent experiments removed).

## Calibration data (retained)

The request-token prefix trie used for calibration is retained under
`qwen35-prune/evidence/wanxiangshu-teacher-trie-v1/`. It contains 270,952
unique trie nodes and 42 request records. The binary files are versioned
for future use; the pruning and recurrent experiment code has been removed
due to poor results.

- `format.json` - trie format descriptor
- `nodes-000000.bin` - token trie nodes (binary)
- `requests-000000.bin` - request records (binary)
- `README.md` - SHA-256 and inventory

The trie was captured by `llama-server --run-dump` using the format documented
in `wanxiangqi/docs/request-corpus-dump.md`.

## Note

Prune (ENP/GDN/Qwen3.5-MoE) and recurrent shared-bank MoE experiments were
removed. Only the calibration trie is kept for potential future use.
