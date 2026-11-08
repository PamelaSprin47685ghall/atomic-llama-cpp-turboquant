# TP5 integration checklist

## Build and install (every merge)

1. Build the upstream tree, then install it:
   `cmake --build upstream --parallel 32 --target llama ...`
   `cmake --install upstream`
2. Rebuild and install the compose tree:
   `cmake --build compose --parallel 16`
   `cmake --install compose`
3. Rebuild and install every module used by compose:
   `sampling`, `scheduling`, `text`, `relay` (build dir, then `cmake --install`).
4. Never reuse a compose binary older than the installed `libllama`.

## Install consistency gate

Run before any GPU smoke. A stale `install/bin/tq-tp5-compose` with a newer
`install/lib/libllama` is a hard stop.

- `install/bin/tq-tp5-compose` and `build/compose/tq-tp5-compose` must have the
  same mtime and size. `sha256sum` may differ only by the install RPATH rewrite:
  check `cmp -l ... | wc -l` is small and every differing byte lies in
  `RUNPATH` (`readelf -d`). Current expected deltas: compose 48 bytes,
  `libllama` 73 bytes, `libggml-hip` 49 bytes.
- `readlink -f install/lib/libllama.so` and `libggml-hip.so` must point at the
  same soversion as `build/upstream/bin/`.
- `LD_LIBRARY_PATH=install/lib:/opt/rocm/core-10.0/lib ldd install/bin/tq-tp5-compose`
  must resolve `libllama`, `libggml`, `libggml-base`, `libggml-cpu` and
  `libggml-hip` from `install/lib`, with no `not found`.
- Feature strings:
  - `install/bin/tq-tp5-compose`: `GGML_TP5_PLAN_UNION`, `TP5_MTP_BATCH_POOL`,
    `TP5_TEXT_POOL` (the other WS switches live in the libraries).
  - `install/lib/libllama`: `TP5_UBATCH_POOL`, `TP5_KV_SNAPSHOT_POOL`,
    `TP5_UBATCH_TRACE`.
  - `install/lib/libggml-hip`: `GGML_TP5_STAGING`, `GGML_TP5_PLAN_UNION`,
    `GGML_TP5_RELAY_FRAMES`, `GGML_TP5_POOL_EXACT`, `GGML_TP5_GRAPH_UID`.
  - A missing string in the artifact that owns it means the install is stale:
    stop, reinstall, re-check. Do not debug pool code against a stale install.
- H1 trace gate: with `TP5_UBATCH_POOL=1 TP5_UBATCH_TRACE=1`, stderr must carry
  non-empty `TP5_UBATCH` trace lines before decode. Empty trace means the
  compose/lib pair is still mismatched.

## Smoke order after an install

`H1 (ubatch pool + trace)` -> `H2 (kv snapshot)` -> `H3 (mtp batch)` ->
`H4 (text pool)` -> `H (four together)` -> `ALL (H+S+P)`.

Each arm: 32x2, `--capture-logits`, 262144 ctx, FA on, F32 relay, same 34-value
environment, `--deadline 600`. A pass requires `rc=0` and logits sha
`8a701790e146becbab8733b476a2203432c1b5cb4b3fa41b443ea028dc436fb3`.
