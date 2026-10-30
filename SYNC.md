# Engine source snapshot (synced 2026-10-27 from 13022)

- Source: kunweiz@13022:~/atomic-llama-cpp-turboquant, branch clean/fork-20261018, HEAD 829d05f9e
  (worktree == HEAD for engine files; only an untracked-doc deletion was dirty).
- Method: `git archive HEAD` (all 4440 tracked files: src/, ggml/ incl. ggml-cuda + ggml-hip, include/, common/, fork/ incl. fork/tp5-compose, tools/, vendor/ ...).
  No build-*/ dirs, no models, no .git history. Local git repo re-initialised here (commit "snapshot ...") so `git diff` yields patch.diff.
- overlay/mtp-async-protocol-candidate-20261027/: source-only copy of
  13022:~/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/mtp-async-protocol-candidate-20261027
  (runtime.cpp = the 2177-line TP5 runtime used in /home/hermes/work/lb-impl-kvslot/src/runtime.cpp, md5 613a05b7;
  ggml-backend-meta.cpp + meta.patch, hip-relay/relay.hip + relay.patch, tp5-meta-submit-ready.h, runtime.patch, build argv json).
  Binaries/.o/install/ NOT synced.
- Key files: src/llama-model.cpp, src/llama-context.cpp, src/llama-graph.cpp, src/llama-memory-*.cpp, src/llama-kv-cache*.cpp,
  ggml/src/ggml-backend.cpp, ggml/src/ggml-backend-meta.cpp, ggml/src/ggml-cuda/{ggml-cuda.cu,relay.hip}, ggml/src/ggml-hip/, fork/tp5-compose/src/runtime.cpp.
- Read-only reference copy: do not build here (build-slot 3-min cap). Make edits in a per-run worktree copy.
