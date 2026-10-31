# Provider identity for window attribution — static review and minimal patch

**Status: STATIC ONLY.** No compilation, fixture execution or syntax check was
run on host 12022 after the no-build/no-test directive. No HIP/GPU command was
run at any point. `selected/` installs were never modified. The two originals in
this directory (`REPORT.md`, `patch.diff`) were not overwritten.

## Deliverables

| file | purpose | sha256 |
|---|---|---|
| `provider-getter.patch` | real minimal patch: public getter + Meta implementation + runtime MAP emission (3 files, 25 added lines) | `21754a675810add69ce1ef32f2c8a00e04697b71aecf310a3429d22e9daac73d` |
| `impl/ggml/include/ggml-relay.h` | patched header | `125ba96887935403004803be51f0005ee2774df99bd838999526a31bd2adaf7e` |
| `impl/ggml/src/ggml-backend-meta.cpp` | patched Meta backend | `e0c440948496bfc8581234618e4d531c32765ff562120afbc1cb7b3ccc9080f1` |
| `impl/overlay/mtp-async-protocol-candidate-20261027/runtime.cpp` | patched runtime source | `fdcb4b96eb494ffc8750ab81f439fe12be77108464a6338d2331e2e95953a5c7` |
| `verify/provider-getter-cpu/provider-getter-cpu.cpp` | CPU contract fixture source (fake child devices + fake relay interface; no HIP) | `60faa15731871122b108bc6863a7ffe975730d5df0c552bd0a6a1ad53d0be9ee` |
| `verify/provider-getter-cpu/build.sh` | build/run steps for an allowed host | — |

Baseline: local clean production-equivalent tree
`/home/kunweiz/work/thru-ab-live-r8-061501/src`, HEAD
`a1f1d2c9401b67da7588e189ab94ec62b0878cd5`.
Blob hashes: `ggml-relay.h` `cff5609d…`, `ggml-backend-meta.cpp` `1417e8ef…`,
`overlay/.../runtime.cpp` `82a2b353…`.

Originals in this directory (read-only candidates, unchanged): `REPORT.md`
`c7ebcbdbb3761695d342d5b66e3c3de808b3896f365a4c51fd48ba9de1048cc5`,
`patch.diff` `e1b92683d5750329d55608b2fcb2b07da8196687638db1a68d2d1e5a8557b634`.

## 1. What was executed before the directive (and what was not)

Commands were run under `/home/kunweiz/.local/bin/build-slot` only.

1. `build.sh meta` — **rc=1**, wrong source path, compiler did not start.
2. `build.sh meta` — **rc=0** with warning `no previous declaration for
   ggml_backend_relay_provider`; includes resolved the *unpatched* source
   header. Object produced.
3. `build.sh meta` — **rc=0**, no output, with the patched header first on the
   include path; object replaced. This is the only build of the patched Meta TU.
4. `build.sh lib` — **rc=0**; relinked `libggml-base-provider.so.0.18.1` from
   the existing CPU-only `ggml-base` objects plus the patched Meta object.
5. `build.sh test` — **rc=1**; the fixture TU resolved the unpatched header and
   therefore lacked the declaration. **No test binary was produced.**
   The pre-ban shared library records SONAME `libggml-base-provider.so.0`, but
   no `.so.0` symlink was created at that time, so it was never loadable as-is
   and was never loaded. `build.sh` now creates that symlink in the lib step.

**No fixture was ever executed.** No GPU/HIP build or run occurred.

After the directive: zero compiler invocations, zero test runs, zero
`-fsyntax-only`. The object and shared library produced in steps 3–4 are
quarantined in `stopped-build-artifacts/`; they are build artifacts, not test
evidence, and may be stale.

## 2. Lifecycle evidence (real production code)

The identity chain is:

```
runtime window i
  -> contexts[i] (llama_context, owner)
  -> backend_tp5_relay (Meta ggml_backend_t)
  -> backend_ctx->relay (relay provider object p)
  -> provider=%p printed by the relay evidence lines
```

Evidence, all in the baseline tree:

1. **Window creation.** `overlay/.../runtime.cpp:1699` creates one
   `llama_context` per window; `:1731-1743` takes
   `llama_tp5_relay_backend(context[wi])` and stores it in `native[wi]`.
   `contexts` and `native` are declared at `:1683-1684`.
2. **Per-context ownership.** `src/llama-context.cpp:505-525` creates the Meta
   backend for `tp5-relay` / `tp5-relay-native`, validates it with
   `ggml_backend_relay_snapshot`, and stores it in `backend_tp5_relay`.
   `backends` is a `std::vector<ggml_backend_ptr>` member
   (`src/llama-context.h:427`), so the backend is owned by that context.
3. **Provider creation.** `ggml/src/ggml-backend-meta.cpp:1897-1973` sets
   `relay_mode` and stores the exact `relay_i->create(...)` result in
   `backend_ctx->relay`; `valid` is true only when that result is non-null.
   `relay` is assigned nowhere else except `nullptr` in
   `ggml_backend_meta_free` (`:2019-2033`), so the pointer is stable for the
   backend's lifetime.
4. **What the relay prints.** `ggml/src/ggml-cuda/relay.hip:757-976`
   (`create`) returns `provider * p`; `:970-974` prints `provider=%p` with
   `(void *) p` when the priority pair is enabled. The span/overlap dump in
   `overlay/.../hip-relay/relay.hip:1662` and in the HIP-spans patch
   (`pw.../relay.hip` `TP5_DW_HIP_SPANS/SPAN` lines) prints `(void *) &p`
   where `p` is `provider &`. `&p == p` for the same object, so all evidence
   lines and `ctx->relay` are the **same numeric identity**. They are not the
   Meta backend pointer.
5. **MAP emission is inside the owner lifetime.** The runtime joins its
   workers at `runtime.cpp:3133-3135`, then emits
   `TP5_DW_HIP_OVERLAP_MAP` at `:3203-3208` while `contexts` is still alive.
   `contexts.clear()` is at `:3286` (patched file `:3297`), i.e. after the MAP
   line. Clearing destroys each `llama_context`, which frees its Meta backend
   through `ggml_backend_meta_free`, which calls `relay_i->destroy` and sets
   `relay = nullptr`.
6. **Span dump ordering is not an attribution mechanism.** The HIP-spans dump
   runs inside `destroy` (`meter_flush_dump` before `delete p`), during
   `contexts.clear()` — after the MAP line. The joiner is offline and matches
   by pointer identity; it must never use line order. This is consistent with
   the no-order-inference rule.

**Conclusion:** `ggml_backend_relay_provider(native[wi])` is called while the
provider object is guaranteed alive, and returns exactly the pointer printed by
the span producers. A null return can only mean "no live relay provider"; the
MAP key is then omitted rather than guessed.

## 3. Patch contents


`provider-getter.patch` (3 files; no joiner files, no selected installs):

1. `ggml/include/ggml-relay.h` — declares
   `GGML_API const void * ggml_backend_relay_provider(ggml_backend_t backend);`
   as an identity token: never dereference/free/retain past backend lifetime;
   null for null/non-Meta/non-relay backends; no implicit synchronization with
   teardown. No struct or ABI layout change; a new exported symbol only.
2. `ggml/src/ggml-backend-meta.cpp` — implementation next to the existing
   `ggml_backend_relay_snapshot` wrapper:
   `ggml_backend_relay_meta_context(backend)` (already checks
   `ggml_backend_is_meta`, `relay_mode` and non-null `relay`) then return
   `ctx->relay`.
3. `overlay/.../runtime.cpp` — after printing `native=%p`, call the getter and
   print ` provider=%p` **only when non-null**, before the `pci_bdf*` entries.
   If every MAP line omits the key, the joiner uses its pre-existing
   pointer/shared logic; a partial mixture is refused, never downgraded.

The 13022 lane compiles an assembled `src/runtime.hsp.cpp`, which is not
present in the local tree. The source-of-truth hunk is the overlay change above;
the original `patch.diff` hunk against `src/runtime.hsp.cpp` inserts the same
provider block at the same MAP printer (its context at `:3528` is the
HIP-spans-shifted copy). Apply the overlay hunk to the source and re-assemble
the lane copy; do not hand-edit a frozen lane directory as the only delivery.

## Dependencies

- C++17 compiler. The getter TU is `ggml/src/ggml-backend-meta.cpp`, compiled
  into `libggml-base`. The pre-ban evidence compiled that TU and relinked a
  CPU-only `ggml-base`; production must rebuild/replace the `ggml-base` library
  used by the runtime, not only the runtime binary.
- `ggml-relay.h` must be on the Meta and runtime include paths; the overlay
  runtime already includes it.
- **13022 lane-generated file:** `src/runtime.hsp.cpp` is generated and is not
  present in this worktree. `provider-getter.patch` targets the source
  `overlay/.../runtime.cpp`. At lane assembly the same provider block must be
  applied to the generated `runtime.hsp.cpp`, then diffed against the source
  intent on 13022. This work cannot claim the generated file was patched,
  compiled or executed.
- CPU fixture: internal headers `ggml-backend-impl.h` and
  `ggml-relay-impl.h`, plus CPU-only `ggml-base` objects. No HIP headers.
- Offline joiner: optional `provider=` between `native=` and `pci_bdf*`. A log
  without the key is **not** automatically shared: the joiner attempts pointer
  mode first and only falls back to shared when no span provider matches a MAP
  `native=`.
- No HIP/ROCm toolchain, GPU device, or `selected/` install is used.

## 4. Joiner contract (owned by the joiner worker; not edited here)

No file under `overlay/.../dw-hip-join.h` or `dw-hip-join-tests.sh` was
modified by this work. The runtime side only adds an optional key. The identity
chain requires the joiner to keep these rules (already present in the 13022
candidate):

- `mode=provider` requires **every** `TP5_DW_HIP_OVERLAP_MAP` line to carry
  `provider=`; all span `provider=` values must match exactly one MAP provider.
- Missing provider key on any MAP line, two windows claiming one provider, a
  provider matching both `provider=` and another window's `native=`, duplicate
  window/provider mappings, and duplicate MAP keys must refuse.
- Only proven provider/pointer modes may emit the original
  `TP5_DW_HIP_OVERLAP_JOINED_PAIR` tag with
  `pair_scope=same_pci_shared_epoch`. Shared/unattributed evidence must stay on
  its own diagnostic tag/scope, so the old gate cannot pass on it.
- Old logs with no `provider=` are not automatically "shared": the joiner
  first tries pointer mode (each span provider must match a MAP `native=`),
  and only when no native matches does it fall back to shared diagnostics.
  This patch does not retroactively attribute any old log; the original 13022
  ON/OFF logs still produce no `_JOINED_PAIR` and no
  `pair_scope=same_pci_shared_epoch`.

- In relay execution every window owns a separate context, Meta backend and
  relay provider (`src/llama-context.cpp:505-525` runs per context;
  `runtime.cpp:1731-1743` takes each window's backend). A null getter on a
  relay window, or a partial MAP mixture, is therefore a source defect, not an
  expected mode; the joiner's refusal is the correct fail-closed outcome.

**Peer evidence (joiner-owned, not produced here):** `dw-hip-join.h` sha256
`ad0eccede3f90368…`; `dw-hip-join-tests.sh` `e4750bd69d76452c…`;
`joiner-cumulative.patch` `9feaa328…`; `joiner-hardening.patch` `212730da…`.
The joiner's CPU suite was 62/62 **before** the 12022 no-build/no-test directive
and was not rebuilt or rerun after it. This work did not modify any joiner file.

## 5. Verification commands (for an allowed host; NOT RUN on 12022)

The fixture is a CPU-only contract test: fake child devices/backends and a fake
relay interface drive the real Meta backend getter path. It proves null
returns, positive identity equality with `relay_i->create`, stability across
calls, distinct identities for two relay backends, and destroy-at-free. It does
**not** prove HIP behavior and is not device evidence.

```sh
# Commands below are for an ALLOWED host. Do not run any on 12022.
# build.sh contains direct compilers and does NOT acquire a build slot; every
# step must be wrapped by build-slot.

# A. Standalone impl/ fixture (current build.sh default):
#    compiles impl/ggml/src/ggml-backend-meta.cpp against impl/ggml/include,
#    using the baseline tree only for other headers/objects.
cd /home/kunweiz/work/lead-provider-map-20261030
/home/kunweiz/.local/bin/build-slot bash verify/provider-getter-cpu/build.sh meta
/home/kunweiz/.local/bin/build-slot bash verify/provider-getter-cpu/build.sh lib
/home/kunweiz/.local/bin/build-slot bash verify/provider-getter-cpu/build.sh test
./verify/provider-getter-cpu/provider-getter-cpu   # expect PASS, rc=0
# The lib step creates libggml-base-provider.so.0 -> .so.0.18.1 because the
# executable records SONAME libggml-base-provider.so.0. Without that symlink
# the binary fails to load even when the link step succeeded.

# B. Applied-tree variant (override form; use exact same build-slot wrappers):
#    apply the patch, build that tree's own ggml-base with its build system,
#    then point the fixture at that tree. Do not reuse impl/.
cd /path/to/patched/tree
patch -p1 < /path/to/provider-getter.patch
TREE=/path/to/patched/tree
TREE_BUILD=/path/to/patched/tree/build/ggml/src/CMakeFiles/ggml-base.dir
cd /home/kunweiz/work/lead-provider-map-20261030
SRC="$TREE" BASE_BUILD="$TREE_BUILD" \
META_SRC="$TREE/ggml/src/ggml-backend-meta.cpp" \
HEADER_INC="$TREE/ggml/include" \
  /home/kunweiz/.local/bin/build-slot bash verify/provider-getter-cpu/build.sh meta
SRC="$TREE" BASE_BUILD="$TREE_BUILD" \
  /home/kunweiz/.local/bin/build-slot bash verify/provider-getter-cpu/build.sh lib
SRC="$TREE" HEADER_INC="$TREE/ggml/include" \
  /home/kunweiz/.local/bin/build-slot bash verify/provider-getter-cpu/build.sh test
# The generated 13022 file src/runtime.hsp.cpp still needs the same provider
# block applied and statically diffed on 13022; this worktree has no copy.
```

## 6. Claim scope

The getter and MAP `provider=` establish **window ownership of a relay
provider**. They do not turn provider span intersections into kernel-level
co-execution: the numbers remain same-device, single-clock intersections of
enqueue-to-completion device-stream envelopes (`clock=hip_event`), and
cross-device intervals are never compared. Attribution is by pointer equality,
never by log order, priority or creation order.

## 7. Residual risks / open items

- The fixture has **not** been compiled since the final include-path fix and has
  **never** been executed. Treat it as an unverified harness until an allowed
  host runs the commands above.
- The runtime TU was not compiled after the change (and must not be on 12022).
  The pre-ban Meta TU compile (rc=0) is the only compiler evidence for the
  getter; the runtime edit is statically reviewed only.
- The 13022 assembled `src/runtime.hsp.cpp` is remote/frozen; the hunk must be
  applied at assembly time, not inferred from the overlay alone.
- The getter returns an identity token, not a lifetime-extending handle. Any
  future caller that stores it beyond `ggml_backend_free()` violates the
  documented contract.
