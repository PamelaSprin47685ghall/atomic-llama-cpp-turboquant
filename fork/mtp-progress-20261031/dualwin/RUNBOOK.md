# RUNBOOK — layered verification (NOT executed)

Nothing in this file was executed. The active rules forbid compilation, tests,
mocks, harnesses and GPU work on 12022, and 13022 HIP requires Ashley's
authorization (no burn, no GPU lock). The commands below are the repo-supported
routes, with their prerequisites stated. Where a flag cannot be derived from the
repository, that is said explicitly instead of guessing.

Set once:

```
R=/home/kunweiz/work/lead-device-wait-20261030
W=$R/wt
BS=/home/kunweiz/.local/bin/build-slot      # required for any allowed compile
```

`BS` returns exit 75 under load; record it and stop — do not re-queue. Never set
`GGML_TP5_RSZ_RT`. Never modify sampling / `greedy.selected`.

## Layer 0 — patch provenance (git only, no compiler)

Uses a unique temporary directory; no fixed path is removed.

```
D=$(mktemp -d "${TMPDIR:-/tmp}/dw-stock.XXXXXX")
git -C "$W" archive HEAD | tar -x -C "$D"
git -C "$D" init -q
git -C "$D" add -A
git -C "$D" -c user.email=verify@local -c user.name=verify commit -qm stock
git -C "$D" apply --check "$R/patch.diff"        # expect 0
git -C "$D" apply         "$R/patch.diff"        # expect 0
cmp "$D/fork/tp5-compose/src/runtime.cpp" "$W/fork/tp5-compose/src/runtime.cpp"
cmp "$D/ggml/src/ggml-backend-meta.cpp"   "$W/ggml/src/ggml-backend-meta.cpp"
git -C "$D" apply -R --check "$R/patch.diff"     # expect 0
test ! -e "$D/fork/tp5-compose/src/dw_device_wait.h"   # removed file must be absent
```

Net increment against the predecessor candidate (copy of the 085001 tree):

```
P=$(mktemp -d "${TMPDIR:-/tmp}/dw-prev.XXXXXX")
cp -a /home/kunweiz/work/dualwin-device-wait-live-r8-085001/wt/. "$P"/
git -C "$P" apply --check "$R/net-increment.patch"     # expect 0
```

## Layer 1 — CPU build of the compose package (repo-supported route)

Prerequisites (all must already exist; these are **not** derivable on 12022):
CMake >= 3.21, a C++17 compiler, and installed CMake packages for
`llama`, `TQRelay`, `TQSampling`, `TQScheduling`, `TQText`
(`fork/tp5-compose/CMakeLists.txt` does `find_package(llama CONFIG REQUIRED)`,
`find_package(TQRelay CONFIG REQUIRED)`, and the same for `TQSampling`,
`TQScheduling` and `TQText`).
The installed `llama` must be the one built from this tree so that the
new `llama_tp5_*` symbols exist.

```
B=$(mktemp -d "${TMPDIR:-/tmp}/dw-cpu.XXXXXX")
"$BS" cmake -S "$W/fork/tp5-compose" -B "$B" \
      -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"$BS" cmake --build "$B" --target tq_tp5_runtime -j "${JOBS:-4}"   # library only
# for Layer 4 also build the CLI (output name is tq-tp5-compose):
"$BS" cmake --build "$B" --target tq_tp5_compose  -j "${JOBS:-4}"
```

Every `cmake` invocation goes through `$BS`, including configure: configure runs
compiler detection and probe compiles. Read-only helpers (`git`, `tar`, `cmp`,
`jq`, `mktemp`, `nvidia-smi`-style queries) do not need `$BS`.

`PREFIX` is the install prefix holding the five packages. Without it the package
cannot configure; a hand-written `-I` list is **not** an equivalent check — the
real include set comes from those packages' exported target usage requirements.

Single-translation-unit syntax check with the real flags (no guessing): extract
the exact command CMake used, then rerun it through `$BS` with `-fsyntax-only`
and without the output file.

```
jq -r '.[] | select(.file | endswith("fork/tp5-compose/src/runtime.cpp")) | .command' \
   "$B/compile_commands.json"
# prefix the printed command with "$BS" and replace "-c ... -o ..." by "-fsyntax-only"
```

For `src/llama-context.cpp` and `ggml/src/ggml-backend-meta.cpp`, configure the
main tree (also through `$BS`) and use *its* `compile_commands.json` the same way:

```
L=$(mktemp -d "${TMPDIR:-/tmp}/dw-llama.XXXXXX")
"$BS" cmake -S "$W" -B "$L" -DLLAMA_CURL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
jq -r '.[] | select(.file | endswith("src/llama-context.cpp")) | .command'        "$L/compile_commands.json"
jq -r '.[] | select(.file | endswith("ggml/src/ggml-backend-meta.cpp")) | .command' "$L/compile_commands.json"
```

A CPU-only configure does not compile the CUDA/HIP sources; `ggml-backend-meta.cpp`
is host code and does compile there. `src/llama-context.cpp` also compiles
without HIP.

## Layer 2 — header-only gate check (needs only a compiler)

`fork/tp5-compose/src/dw_compose_event.h` includes only `<cstdlib>`/`<cstring>`,
so this TU is self-contained:

```
G=$(mktemp -d "${TMPDIR:-/tmp}/dw-gate.XXXXXX")
cat > "$G/gate.cpp" <<'EOF'
#include "dw_compose_event.h"
#include <cstdio>
int main() { std::printf("compose_gate=%d\n", int(dw_compose_event::gate())); return 0; }
EOF
"$BS" g++ -std=c++17 -I"$W/fork/tp5-compose/src" "$G/gate.cpp" -o "$G/gate"
env -u GGML_TP5_DW_COMPOSE_EVENT "$G/gate"   # expect compose_gate=0
GGML_TP5_DW_COMPOSE_EVENT=0 "$G/gate"        # expect 0
GGML_TP5_DW_COMPOSE_EVENT=1 "$G/gate"        # expect 1
GGML_TP5_DW_COMPOSE_EVENT=10 "$G/gate"       # expect 0
```

This proves env parsing only; it cannot exercise record/wait.

## Layer 3 — HIP syntax (authorized host, Ashley's approval)

Prerequisites: ROCm with `hipcc`, a GPU-visible `/dev/kfd`, and the site's
`-DGGML_HIP=ON` configuration. Do not take a GPU lock for a compile; use `$BS`.

Route: configure with HIP and `CMAKE_EXPORT_COMPILE_COMMANDS=ON`, then take the
exact per-file commands. The architecture flags (`CMAKE_HIP_ARCHITECTURES`,
`GPU_TARGETS`) and ROCm paths are site values and are **not** guessed here.

```
H=$(mktemp -d "${TMPDIR:-/tmp}/dw-hip.XXXXXX")
"$BS" cmake -S "$W" -B "$H" -DGGML_HIP=ON -DLLAMA_CURL=OFF \
      -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
jq -r '.[] | select(.file | endswith("ggml/src/ggml-cuda/ggml-cuda.cu")) | .command' "$H/compile_commands.json"
jq -r '.[] | select(.file | endswith("ggml/src/ggml-cuda/relay.hip"))   | .command' "$H/compile_commands.json"
# any `cmake --build` here also runs through "$BS";
# prefix each printed command with "$BS" and replace "-c ... -o ..." by "-fsyntax-only"
```

The HIP configure probes the toolchain (`hipcc`, device detection), so it must go
through `$BS` as well; `jq` and the trace post-processing do not.

If `jq` is absent, use `grep -n` on `compile_commands.json`; the point is to use
the generated command, not a reconstructed include list.

## Layer 4 — device behavior with the real CLI

Prerequisites: a binary built from this tree (`tq-tp5-compose`), the real model
file, five HIP ranks, and Ashley's explicit authorization for the GPU lock.
CLI syntax below is from `fork/tp5-compose/src/main.cpp` (`usage()`); `--model`,
`--prompt` (1..4×) and `--output` are required, `--mode` defaults to `relay`,
`--windows` defaults to 2.

Three arms, each a fresh process, same model/prompts/seed-free greedy decoding:

```
OUT=$(mktemp -d "${TMPDIR:-/tmp}/dw-run.XXXXXX")
BIN="$B/tq-tp5-compose"                    # build-tree binary (OUTPUT_NAME from CMakeLists)
MODEL=/path/to/model.gguf

# stock binary (from the pristine stock build), no gates
env -u GGML_TP5_DW_COMPOSE_EVENT "$STOCK_BIN" \
  --model "$MODEL" --prompt "A" --prompt "B" --mode relay --windows 2 \
  --predict 8 --ctx-size 512 --threads 4 --deadline 300 --capture-logits \
  --output "$OUT/stock.json"

# candidate, gate OFF
env -u GGML_TP5_DW_COMPOSE_EVENT "$BIN" \
  --model "$MODEL" --prompt "A" --prompt "B" --mode relay --windows 2 \
  --predict 8 --ctx-size 512 --threads 4 --deadline 300 --capture-logits \
  --output "$OUT/off.json"

# candidate, gate ON
GGML_TP5_DW_COMPOSE_EVENT=1 "$BIN" \
  --model "$MODEL" --prompt "A" --prompt "B" --mode relay --windows 2 \
  --predict 8 --ctx-size 512 --threads 4 --deadline 300 --capture-logits \
  --output "$OUT/on.json"
```

Each `--output` also writes `<output>.logits.f32le` and
`<output>.logits.rows.jsonl`: the suffix is appended to the full output path that
was given (`fork/tp5-compose/src/main.cpp` builds `report + ".logits.f32le"` and
`report + ".logits.rows.jsonl"`). With `--output "$OUT/stock.json"` the files are
`$OUT/stock.json.logits.f32le` and `$OUT/stock.json.logits.rows.jsonl`. Compare:

```
cmp "$OUT/stock.json.logits.f32le" "$OUT/off.json.logits.f32le"   # OFF must equal stock
cmp "$OUT/off.json.logits.f32le"   "$OUT/on.json.logits.f32le"    # ON must equal OFF
cmp "$OUT/stock.json.logits.rows.jsonl" "$OUT/off.json.logits.rows.jsonl"
```

Also compare the emitted token ids per request from the three JSON reports; the
JSON carries timing fields, so do not `cmp` the JSON itself.

The repo's own acceptance driver is
`fork/tp5-compose/tests/real-tp5.py`. `--binary`, `--model` and `--output-dir`
are required, and `--output-dir` **must not exist** (it creates all evidence
exclusively). `--profile` is `qwen3-q8` (default, `--ctx-size` defaults to 512) or
`swift` (`--ctx-size 262144`, fixed prompt A, one window); `--cases` defaults to
`main,slow,cancel,abort`. One fresh run per arm:

```
ARM=off
env -u GGML_TP5_DW_COMPOSE_EVENT python3 "$W/fork/tp5-compose/tests/real-tp5.py" \
  --binary "$BIN" --model "$MODEL" --output-dir "$OUT/real-$ARM" \
  --profile qwen3-q8 --predict 8 --ctx-size 512 --threads 4 --flash-attn off --deadline 300

ARM=on
GGML_TP5_DW_COMPOSE_EVENT=1 python3 "$W/fork/tp5-compose/tests/real-tp5.py" \
  --binary "$BIN" --model "$MODEL" --output-dir "$OUT/real-$ARM" \
  --profile qwen3-q8 --predict 8 --ctx-size 512 --threads 4 --flash-attn off --deadline 300
```

For the stock arm use `--binary "$STOCK_BIN"` and no gate variable. The driver
runs real device work under the same authorization as Layer 4; it is not a
compile, so `$BS` does not apply. It does not know the compose gate, so the arm's
environment is the only difference.

Receipt checks on the ON arm (`TP5_DW_COMPOSE_EVENT …` lines on stderr):

- `output_event_syncs == host_sync_replaced > 0`;
- `output_event_syncs + fallback_no_output_barrier == epochs`;
- `no_sync_logits_calls > 0` when the workload has output rows;
- `epochs` equals the OFF arm's decode count for the same workload;
- on a forced-fallback arm (e.g. a child without an event interface)
  `output_event_syncs == 0`, `fallback_no_output_barrier == epochs`, and tokens
  still match stock.

Any token/logits difference, cross-window contamination, deadlock, timeout or
`require` failure is NO-SHIP.

## Layer 4b — device trace (profiler flags are version-specific)

The profiler option set differs between ROCm releases, so this RUNBOOK does not
invent flags. Inspect the installed tool first and use the options it documents
for HIP API activity and kernel dispatch:

```
rocprof --help 2>&1 | head -80          # legacy rocprof
rocprofv3 --help 2>&1 | head -80        # rocprofv3
```

Run the ON arm under the selected profiler with the same CLI as Layer 4 and save
the raw trace, the exact command line, the profiler version and any "records
lost" message. Required trace facts:

1. for each epoch, the `hipMemcpyAsync`/`hipMemcpy2DAsync` of the logits on the
   child stream, then `hipEventRecord` of the private event on the same stream;
2. the host `hipEventSynchronize` before the first logits read;
3. the relay lane's kernels/`hipEventRecord(tail)` on that same stream, showing
   the event is later than every contribution of that rank;
4. the child stream ids match the split state's contributing children.

Only kernel `[start,end)` intersection of the two windows on the same device can
support an overlap claim; host stamps and receipt counters cannot. Without this
trace the result stays PARTIAL and no latency or overlap benefit may be claimed.
