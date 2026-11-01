#!/usr/bin/env python3
"""Apply the FA src[5] RELAY-REPLAY-BOUND guard candidate to a PRIVATE copy of the pinned tree.

    ./prepare_relay_bound_guard.py [SRC_TREE] [DEST_TREE]
      SRC_TREE   default /home/kunweiz/work/merge-fa-20261031   (read-only input)
      DEST_TREE  default <this dir>/relay-patch                  (private output)

Never writes into SRC_TREE. Copies exactly the files it patches plus the include closure,
sha256-checks the pinned ones against revision 03024e721, applies anchored replacements (each
asserted to occur exactly once) and writes a unified diff.

RUN BY DEVOPS ONLY. The author did not run this file, did not compile, and did not test.

WHAT THIS LANE IS (and is not)
------------------------------
The question is the one the src[5] lane keeps hitting: on the FA src[5] runtime-KV-length channel,
can a per-replay, fail-closed, no-D2H/no-sync guarantee be given for the relay (five-rank,
capture/replay) consumer?

Answer, from reading this tree: **not as a device-side signal**, and this script does not
pretend otherwise. What it implements is the strongest boundary that survives the relay's own
ordering rules:

  R1  a bound that can only be produced by a kernel of the same graph is REFUSED for relay replay
      before any submission (relay.hip submit(), the single place every submission passes);
  R2  a host-produced per-eval bound (GGML_TP5_FA_RTLEN / GGML_TP5_FA_CHECKED_TAIL, the only
      producer class this tree has for the relay's own graph) must be a device-resident tensor
      whose value the device check below can actually reach;
  R3  the device check runs INSIDE launch_fattn, on the compute stream, before any K/V, mask or
      bound-consuming work, and on a mismatch it terminates the process (GGML_ABORT) instead of
      emitting a value. NaN is never the signal: nothing in this tree rejects a NaN logit row
      before sampling (see RETRACTION.md section 1).

R3 is a process abort, not a "rejected result", and the two are not the same thing. The abort is
correct for a corrupted invariant (a bound outside [0, built]) and it is the only mechanism in
this framework that stops a replay before its output is consumed without a host round trip.
What it cannot do is turn a *late* bound into an *early* one: a value that is legal but stale is
still consumed. R1/R2 remove the classes where staleness is reachable for the relay lane
(cross-window slot reuse, a kernel-produced slot, a host-resident bound); they do not make the
remaining class fresh, because the relay has no per-replay host barrier (see LIMITS.md).

The relay provider itself is NOT modified: it has no src[5] knowledge, and adding a per-replay
host scan of every stage graph would add a host walk per replay without proving freshness. That
is a design decision recorded in LIMITS.md, not an omission.

Phases are separable so review does not depend on running anything:

  python3 prepare_relay_bound_guard.py --copy-only  SRC DEST   (byte-identical pinned copies)
  python3 prepare_relay_bound_guard.py --patch-only DEST       (anchored edits + diff)
"""
import os
import shutil
import subprocess
import sys

_argv = [a for a in sys.argv[1:] if not a.startswith("--")]
MODE = "all"
for _a in sys.argv[1:]:
    if _a in ("--copy-only", "--patch-only", "--all"):
        MODE = _a.lstrip("-").replace("-only", "")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = _argv[0] if len(_argv) > 0 else "/home/kunweiz/work/merge-fa-20261031"
DST = _argv[1] if len(_argv) > 1 else os.path.join(HERE, "relay-patch")

PINNED_REV = "03024e721ad446613d38ea128d0c6f17dea0a391"

# Files this lane patches, with the sha256 they must have in the pinned tree.
PINNED = {
    "ggml/src/ggml-cuda/fattn-common.cuh":
        "bf48c4be6b09ccecbb03ad5a05637d47e0ac428c520a912e04014050c8eeae9a",
    "ggml/src/ggml-cuda/fattn.cu":
        "3c826daa02cfa8e83331d256dd286a227abb75c8904d84e8eca52ecd093693de",
    "ggml/src/ggml-cuda/relay.hip":
        "ac13937177faa06bda6a7f1abecdae6efab787732eb5a39d709c93264aeb1ba6",
}

# Consumers of the modified headers: a TU that re-includes a patched header must be the private
# copy or the edit is invisible to it (same closure rule as prepare_src5_guard.py).
CLOSURE = [
    "fattn.cuh", "fattn-tile.cuh", "fattn-vec.cuh", "fattn-mma-f16.cuh",
    "fattn-mma-turbo.cuh", "fattn-banded.cuh", "fattn-swizzle.cuh",
    "lightning-indexer.cu",
]

# ---------------------------------------------------------------- helper block --
# Inserted immediately before the `typedef float (*vec_dot_KQ_t)(` line in fattn-common.cuh, i.e.
# after ggml_cuda_fattn_rtlen_tensor(), whose contract it extends.
HELPER_ANCHOR = "typedef float (*vec_dot_KQ_t)("

HELPER_CODE = r'''// ---------------------------------------------------------------------------
// FA src[5] RELAY lane (candidate). Why the relay consumer cannot have a device-side
// "validated" flag, and what is enforced instead.
//
// The relay submits one captured epoch per rank and replays it. Three facts decide the
// boundary, all read out of this tree:
//
//  * the relay has no per-replay host barrier: submit() -> finish() launches the epoch and
//    returns; there is no cudaStreamQuery/hipStreamSynchronize on that path (relay.hip
//    finish(), "Never fall back after a possibly accepted launch"), so a flag written by a
//    kernel of this epoch is not observable by the host before the epoch is consumed;
//  * the bound is a host-produced graph input refreshed by set_input_nkv_eff() through a
//    blocking ggml_backend_tensor_set on the per-thread stream (llama-kv-cache.cpp:2724),
//    i.e. it is complete before the epoch is enqueued, not inside it;
//  * nothing downstream rejects a NaN row: llama_sampler.cpp has no isfinite/isnan
//    (RETRACTION.md section 1), so "poison the output" is detection after consumption, not
//    a fail-closed verdict.
//
// Therefore the only two boundaries this lane can enforce without a host round trip are:
//   - the SUBMISSION boundary (host, before begin()): the identity of every src[5] in the stage
//     graphs is checkable there, so a bound produced by a kernel of the same epoch -- whose
//     freshness the relay cannot prove across stages -- is refused outright
//     (ggml_cuda_fattn_relay_bound_in_graph);
//   - the LAUNCH boundary (host, before the first kernel of the node): the tensor must still be
//     a single allocated int32 scalar, so a replay cannot consume a slot that stopped being one.
//
// What this does NOT give, and must not be read as giving: a verdict on the VALUE. Reading it on
// the host is the D2H this lane forbids; reading it on the device and acting on it needs a
// device-side trap that a five-rank captured graph cannot honour. The value therefore keeps the
// stock contract (min() clamp in the kernel, GGML_ASSERT in the CPU consumer, never widened),
// and the legal-but-stale class is named as open in LIMITS.md section 4 rather than dressed up.
//
// ACTIVATION (stated because it is not obvious from the call sites): every check in this lane is
// dead unless ALL of the following hold for the process:
//   1. GGML_TP5_FA_RELAY_BOUND == "1"                       (this lane's own gate), and
//   2. one of GGML_TP5_FA_CHECKED_TAIL / GGML_TP5_FA_RTLEN / GGML_TP5_FA_NKV_PARAM == "1"
//      (otherwise src[5] is never produced and ggml_cuda_fattn_rtlen_tensor() is null), and
//   3. the node is a plain FLASH_ATTN_EXT carrying an i32[1] src[5].
// A lane that runs without 1 and 2 is NOT protected by this guard, and no relay run may be
// cited as evidence about it. See LIMITS.md section 3a.
static inline bool ggml_cuda_fattn_relay_bound_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_TP5_FA_RELAY_BOUND");
        return env != nullptr && strcmp(env, "1") == 0;
    }();
    return enabled;
}

// The single activation predicate for this lane, and the only one both boundaries call.
// True exactly when ALL of LIMITS.md section 3a hold for this node:
//   1. GGML_TP5_FA_RELAY_BOUND == "1";
//   2. one of GGML_TP5_FA_CHECKED_TAIL / GGML_TP5_FA_RTLEN / GGML_TP5_FA_NKV_PARAM == "1", i.e.
//      a bound lane is armed at all;
//   3. the node is a plain FLASH_ATTN_EXT carrying an i32[1] src[5].
// OFF: false on a host constant, and every caller's block is then dead code.
static inline bool ggml_cuda_fattn_relay_bound_active(const ggml_tensor * dst) {
    return ggml_cuda_fattn_relay_bound_enabled() &&
           (ggml_cuda_fattn_rtlen_enabled() || ggml_cuda_fattn_nkv_param_enabled()) &&
           ggml_cuda_fattn_rtlen_tensor(dst) != nullptr;
}

// True when the lane is active AND src[5] was produced by a kernel of the same graph. The only
// such producer this tree has is the streamed chunk staging: it builds a stack
// `ggml_tensor chunk_nkv_eff` whose `buffer` is left null and whose `data` is a slot of the
// per-launch `chunk_bounds` pool (fattn.cu:2276-2281). The host-produced class is the opposite:
// a plain GGML_TENSOR_FLAG_INPUT tensor allocated into a graph buffer (llama-graph.cpp:3283 +
// ggml_set_input). Classifying by "no buffer, or not marked INPUT" therefore separates them
// without guessing at addresses.
static inline bool ggml_cuda_fattn_relay_bound_in_graph(const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_relay_bound_active(dst)) {
        return false;
    }
    const ggml_tensor * t = ggml_cuda_fattn_rtlen_tensor(dst);
    return (t->flags & GGML_TENSOR_FLAG_INPUT) == 0 || t->buffer == nullptr;
}

'''

# --------------------------------------------------------------- device check --
# Inserted at the HEAD of launch_fattn(), i.e. before the pool allocations, before every K/V
# conversion and before the first kernel launch. It must not be placed where n_kv_eff_t is
# already captured: that line sits after the K to fp16 conversion (a first version of this lane
# did exactly that, and the ordering check below caught it).
DEVICE_ANCHOR = """void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const bool use_sparse,
    const int warp_size = WARP_SIZE,
    float * partial_dst = nullptr, float2 * partial_meta = nullptr
) {
"""

DEVICE_CODE = r'''    // FA src[5] RELAY lane (candidate): the launch boundary, taken before any K/V work.
    // Only two things can be decided here without reading the bound's value on the host:
    //  - the slot must be a single int32 scalar allocated in a buffer, otherwise the kernel
    //    below cannot read it as a runtime KV length at all;
    //  - a slot produced by a kernel of this same graph is refused: its freshness depends on a
    //    producer position inside the epoch that the relay does not publish. For graphs that go
    //    through relay submit() this class is already refused there, so the two aborts cover
    //    DISJOINT classes and this one is the backstop for a caller that bypasses submit().
    // The VALUE keeps the stock contract (min() clamp in the kernel, GGML_ASSERT in the CPU
    // consumer, never widened); no D2H and no stream synchronization is taken here.
    // Activation is ggml_cuda_fattn_relay_bound_active(), the same predicate R1 uses and the
    // same three conditions LIMITS.md section 3a states. OFF: it is false on a host constant and
    // the whole block is dead.
    if (ggml_cuda_fattn_relay_bound_active(dst)) {
        const ggml_tensor * bound_t = ggml_cuda_fattn_rtlen_tensor(dst);
        if (ggml_cuda_fattn_relay_bound_in_graph(dst)) {
            GGML_ABORT("GGML_TP5_FA_RELAY_BOUND: src[5] is produced inside this graph "
                       "(streamed chunk staging); the relay cannot prove that slot is fresh "
                       "at replay time");
        }
        if (bound_t->buffer == nullptr || bound_t->ne[0] != 1 || bound_t->data == nullptr) {
            GGML_ABORT("GGML_TP5_FA_RELAY_BOUND: src[5] is not a single allocated int32 "
                       "scalar; the relay replay cannot read it as a runtime KV length");
        }
    }
'''

# ------------------------------------------------------------------ relay gate --
# Inserted in relay.hip submit(), the single place every relay submission passes, after the
# idle/failed/arity guard and before begin() (i.e. before any epoch is created and before any
# launch). A refusal here is a host-side refusal of the whole submission.
RELAY_ANCHOR = """    uint64_t compute_nodes[ranks] = {};
    try {
"""

# relay.hip does not pull the FA headers, so the lane's classifier has to be made visible. This
# is the only include change; it is anchored on relay.cuh so a reordered include block is a
# refusal rather than a silent miss.
RELAY_INCLUDE_ANCHOR = """#include "relay.cuh"
"""

RELAY_INCLUDE_CODE = """#include "relay.cuh"
// FA src[5] RELAY lane (candidate): the pre-submission gate classifies src[5] with the same
// predicate the launcher uses, so both sides cannot drift apart.
#include "fattn-common.cuh"
"""

RELAY_CODE = r'''    // FA src[5] RELAY lane (candidate): refuse a submission whose graphs carry a src[5] bound
    // that was produced inside the graph (class B). This runs before begin(), so on a refusal
    // no epoch, no capture and no launch exists at all.
    // This gate and the launch_fattn abort cover DISJOINT classes, not the same one twice:
    //   - here: class B, i.e. a src[5] produced by a kernel of this graph;
    //   - in launch_fattn: a src[5] that is not a readable single allocated int32 scalar, plus
    //     class B again as a backstop for a caller that does not submit through this provider.
    // A class-A bound (the host-produced per-eval graph input) passes both, by design.
    // The scan is entered only when the lane is on for the process (relay_bound_enabled); each
    // node is then classified with ggml_cuda_fattn_relay_bound_in_graph(), which
    // itself applies the per-node activation predicate. So this gate cannot be "on" for a node
    // the launch side would leave alone, and vice versa.
    if (ggml_cuda_fattn_relay_bound_enabled()) {
        for (size_t stage = 0; stage < stages; ++stage) {
            for (uint32_t r = 0; r < ranks; ++r) {
                const ggml_cgraph * g = graphs[stage * ranks + r];
                if (g == nullptr) {
                    continue;
                }
                for (int i = 0; i < g->n_nodes; ++i) {
                    const ggml_tensor * node = g->nodes[i];
                    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT) {
                        continue;
                    }
                    if (ggml_cuda_fattn_relay_bound_in_graph(node)) {
                        fail(p, ENOTSUP, "relay: FLASH_ATTN_EXT src[5] is produced inside the graph; "
                                         "refused before submission");
                        return false;
                    }
                }
            }
        }
    }
'''


def sha(path):
    return subprocess.check_output(["sha256sum", path]).split()[0].decode()


def patch(rel, pairs):
    p = os.path.join(DST, rel)
    s = open(p).read()
    for old, new, tag in pairs:
        n = s.count(old)
        assert n == 1, "%s: anchor %s count %d" % (rel, tag, n)
        s = s.replace(old, new)
    open(p, "w").write(s)


def main():
    os.makedirs(DST, exist_ok=True)
    if MODE in ("copy", "all"):
        copy_phase()
    if MODE in ("patch", "all"):
        patch_phase()


def copy_phase():
    for rel, want in PINNED.items():
        src = os.path.join(SRC, rel)
        if want is not None:
            got = sha(src)
            if got != want:
                sys.exit("SOURCE-DRIFT %s\n  expected %s\n  got      %s" % (rel, want, got))
        dst = os.path.join(DST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    inc_src = os.path.join(SRC, "ggml/src/ggml-cuda")
    inc_dst = os.path.join(DST, "ggml/src/ggml-cuda")
    for h in CLOSURE:
        src = os.path.join(inc_src, h)
        if not os.path.isfile(src):
            sys.exit("CLOSURE-MISSING %s" % h)
        shutil.copy2(src, os.path.join(inc_dst, h))

    with open(os.path.join(DST, "CLOSURE.txt"), "w") as fh:
        fh.write("pinned revision: %s\n" % PINNED_REV)
        fh.write("copied from %s\n" % SRC)
        fh.write("patched: %s\n" % ", ".join(PINNED))
        fh.write("verbatim closure: %s\n" % ", ".join(CLOSURE))
    print("copied to", DST)


def patch_phase():
    # 1. predicates + in-graph classifier (fattn-common.cuh)
    p = os.path.join(DST, "ggml/src/ggml-cuda/fattn-common.cuh")
    s = open(p).read()
    assert s.count(HELPER_ANCHOR) == 1, "helper anchor"
    assert s.index("static inline const ggml_tensor * ggml_cuda_fattn_rtlen_tensor(") < \
        s.index(HELPER_ANCHOR), "helper must follow the rtlen accessor"
    s = s.replace(HELPER_ANCHOR, HELPER_CODE + HELPER_ANCHOR)
    open(p, "w").write(s)

    # 2. device check at the head of launch_fattn (fattn-common.cuh): before the pool
    #    allocations, before every K/V conversion, before the first kernel launch.
    patch("ggml/src/ggml-cuda/fattn-common.cuh", [
        (DEVICE_ANCHOR, DEVICE_ANCHOR + DEVICE_CODE, "device-check"),
    ])

    # 3. pre-submission refusal in the relay provider (relay.hip)
    patch("ggml/src/ggml-cuda/relay.hip", [
        (RELAY_INCLUDE_ANCHOR, RELAY_INCLUDE_CODE, "relay-include"),
        (RELAY_ANCHOR, RELAY_CODE + RELAY_ANCHOR, "relay-submit-gate"),
    ])

    with open(os.path.join(DST, "CANDIDATE.diff"), "w") as f:
        for rel in PINNED:
            d = subprocess.run(["diff", "-u", os.path.join(SRC, rel), os.path.join(DST, rel)],
                               stdout=subprocess.PIPE).stdout.decode()
            f.write(d)
    print("candidate written to", DST)


if __name__ == "__main__":
    main()
