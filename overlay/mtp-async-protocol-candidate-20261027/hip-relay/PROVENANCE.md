# Relay candidate provenance — hip-relay/relay.hip

## Layering (three stages, both edges reproducible from the files here)

    relay.orig.hip --(relay.ldl.patch)--> relay.ldl.hip --(relay.patch)--> relay.hip

| stage | file | sha256 | what it is |
|---|---|---|---|
| 0 baseline | `relay.orig.hip` | `8bf84b65894146cd8d0858f2cff95121c110e8ac03881f292b3a22d7fc8de2a7` | the TRUE baseline this candidate changes (HIP relay source at `mtp-relay-wide-read-candidate-20261025` ancestry; it still has `__builtin_amdgcn_s_sleep(2)` at line 232) |
| 1 busy-wait | `relay.ldl.hip` | `1badd280b9a826d47182182a0058c9a0a4cdda362c2e9f8ee44b0f414fe3d389` | baseline + the ONE archived edit (s_sleep -> busy-wait). This is exactly what `relay.orig.hip + relay.ldl.patch` yields |
| 2 delivered | `relay.hip` | `1badd280b9a826d47182182a0058c9a0a4cdda362c2e9f8ee44b0f414fe3d389` | stage 1 only: the rank-event round was retracted, so this edge is empty (a zero-byte `relay.patch`) |
| patch 0->1 | `relay.ldl.patch` | `403d3a0ff40021be9123bc42193557101d473545404fbed91f2d4bf87692fb02` | the patch the lane archived at the time (1 line). Kept VERBATIM and under its own name |
| patch 1->2 | `relay.patch` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | stage 1 -> delivered (host-gate adoption + rank events) |

Earlier the directory shipped `relay.patch` (= `relay.ldl.patch` content) next to a `relay.hip`
that also contained stage 2, so `orig + patch` produced `1badd280`, NOT the delivered file
(Devops' 198-line gap). Both edges are now separate, named files.

## Where the extra 198 lines came from (read-only trace)

`relay.hip` minus `relay.ldl.hip` is 189 added + 1 replaced lines in 7 places, and every one
belongs to the two dualwin host-gate adoptions already present on the host side of this same
candidate (`runtime.cpp`), i.e. the relay is the device sink of gates that ship in the host
binary:

| site (delivered line) | lines | owner | host counterpart |
|---|---|---|---|
| 29-69 | 41 | `DUALWIN-SERIALCUT-TRIM` gate: `dw_trim_active` = (`GGML_TP5_DW_SERIALCUT`=="1" && `GGML_TP5_DW_SERIALCUT_TRIM`=="1") read once, plus `tp5_diag_graph_env` (read once) | `runtime.cpp` line 1889/1901 (`serialcut_env`, `trim_env`, `serialcut_gate`, `submit_only_gate`) |
| 445-459 | 15 | `DUALWIN-OVERLAP-METER` per-rank device state (`meter_state/spans/span_ns/span_max_ns/skipped`) | `runtime.cpp` line 444 (`overlap_meter_req`, `GGML_TP5_DW_OVERLAP_METER`) |
| 1542-1670 | 129 | `DUALWIN-OVERLAP-METER` helpers: `meter_harvest`, `meter_submit_begin/end`, `meter_flush_dump` (hipEventQuery/Record/ElapsedTime, direct-mode only) | same gate; host dumps `TP5_DW_OVERLAP_METER … host_counter=1` |
| 1721 | 1 (replace) | TRIM-only `TP5_DIAG_RELAY` scan: `dw_trim_active ? tp5_diag_graph_env : std::getenv("TP5_DIAG_GRAPH")` | TRIM semantics documented in `runtime.cpp` 1901+ |
| 1742, 1760, 1995 | 3 | call sites: `meter_submit_begin(p)`, `meter_submit_end(p)`, `meter_flush_dump(*p)` | — |

## DUALWIN-RANK-EVENTS (RETRACTED - edge 1->2 is empty)

An intermediate revision added a second event per rank lane (`lane::publish`), a `rank_events_live`
window, a provider vtable member `rank_events` (ABI 3) and a publication-count guard on
`cpu_reductions`. All of it is withdrawn:

- a HIP `hipEvent_t` is not a `ggml_backend_event_t` (that type is a struct whose first member is a
  device pointer, dereferenced by `ggml_backend_event_synchronize`), so lending one as the other is
  a type confusion that crashes on first use;
- gating `++p.status.cpu_reductions` on a provider-side flag changed when the counter advances, and
  that counter is the fork/overlay courier's retirement baseline - an unrelated semantic change;
- the only consumer was a cross-window wait, which is the same-phase lockstep the dual-window proof
  forbids.

`relay.hip` is therefore byte-identical to `relay.ldl.hip` (the busy-wait stage), provider ABI is 2,
and `relay.patch` is empty. The file is kept so the chain `relay.orig.hip -> relay.ldl.hip ->
relay.hip` stays reproducible with the same command.

Semantics, as the source states them:
- **SERIALCUT-TRIM (relay side) does nothing unless both env values are exactly "1"; with TRIM
  off the per-submit `getenv("TP5_DIAG_GRAPH")` is kept verbatim.** Its only effect is reading
  that diagnostic variable once instead of per submit (documented in the source as a semantic
  change under TRIM only: a later `setenv` is not observed).
- **OVERLAP-METER is measurement only**: `meter_active` is initialised from
  `GGML_TP5_DW_OVERLAP_METER` and every HIP failure only disables the meter; it never changes
  submitted work. In non-direct mode it counts `meter_skipped_mode` and records nothing.
- Neither adoption touches the F32 fold/publish protocol, the borrow lifetime, or the
  `s_sleep` line.

## What the archived build did, and what it did not

- `relay-compile.argv.json` compiles `hip-relay/relay.hip` (path) and writes
  `hip-relay/relay.hip.o`; `hip-link.argv.json` links that object into a private
  `libggml-hip.so`. Devops' per-image comparison shows the candidate bundle differs from the
  unmodified control in exactly one image — the relay kernel (5768 B vs 5912 B) — i.e. the
  busy-wait edit. The gate blocks are host-side `getenv` reads and event bookkeeping; they add
  no device kernel, so the device-ISA comparison cannot see them either way.
- The lane's own `build-receipt.json` predates this stage (its 2026-10-27 build hashed the
  earlier sources); it is left untouched and makes no claim about `relay.hip`.

## Verify (CPU; runs no build — for Devops)

    cd <this dir>
    # edge 0->1 (one line, no offsets needed)
    sed 's|__builtin_amdgcn_s_sleep(2);|// ASYNC-PROTOCOL: pure busy-wait, no wave sleep.|' \
        relay.orig.hip | sha256sum                                   # 1badd280…
    patch -o /tmp/b.hip -p0 relay.orig.hip -i relay.ldl.patch && sha256sum /tmp/b.hip relay.ldl.hip
    # edge 1->2
    patch -o /tmp/f.hip -p0 relay.ldl.hip -i relay.patch && sha256sum /tmp/f.hip relay.hip   # d5fb4a02…
    patch -o /tmp/r.hip -R -p0 relay.hip -i relay.patch && sha256sum /tmp/r.hip relay.ldl.hip
