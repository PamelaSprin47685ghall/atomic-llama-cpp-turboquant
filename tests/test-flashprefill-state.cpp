// test-flashprefill-state.cpp — FlashPrefill V2 state/membership/policy unit coverage.
//
// Ownership: StateCases (released from ServerRouting; consolidated here per
// parent directive — no separate cases header/source, no CMake change: this
// target is already registered by the foreach loop in tests/CMakeLists.txt).
// Scope: the real cell owner (src/llama-kv-cells.h) plus public policy types
// (include/llama-flashprefill.h), internal checked policy
// (src/llama-flashprefill.h), and wire metadata
// (src/llama-flashprefill-layout.h). No server impl, KV tensors, graph, GPU,
// model, backend, or threads needed.
//   - NOT repeated here: defaults, role gates, tail edges, GQA packing,
//     slice contracts, validation edges, fingerprint determinism — all live
//     in tests/test-flashprefill-routing.cpp. Only mode separation,
//     invalid-config rejection, and the serial-comparison rule are pinned
//     here, which the routing file does not cover.
//   - Envelope/sidecar consolidation is StatePolicy's: the envelope codec,
//     match domains, cache-key discrimination, and bounded diagnostics run
//     here via flashprefill_state_envelope_tests::run_tests() (their cases,
//     their ownership — called, not copied). This file additionally pins the
//     fingerprint mode-separation and serial-domain rules the envelope
//     consults. No production helpers are added here
//     (plan_usable/same_serial_domain below are test-local predicates).
// Prior routing-plumbing/default tautologies (boundary freeze, row
// construction, dense-label pinning, counter-discipline pinning, OFF-default
// pinning) were removed rather than preserved: they belong to the routing
// file or pin defaults without distinguishing a state failure mode.
//
// Status: NOT RUN (compile-only per task constraints).
//
// Optional real-context integration (chains B/C/F/H): enabled ONLY with
// explicit flags --model <path> [--backend <name>] [--required-backend
// <name>]. Remaining argv passes through to common_params_parse (real
// --flashprefill/--device/-c CLI kept); the three flags are consumed here.
// No flags: CPU-only owner/runtime cases above, no model needed.
//   B: real seq-state checkpoint bytes + real KV draft appends + partial
//      seq_rm rollback vs an untouched control and a restored+replayed seq
//      (single model: no draft model exists, so common_speculative is N/A;
//      the checkpoint travels as production seq-state bytes).
//   C: save full seq state, erase, fragment with other seqs, restore onto
//      remapped cells, compare vs untouched control (non-contiguous slots).
//   F: constrained slots (n_ctx=256): oversized append fails nonzero,
//      smaller retry after freeing completes.
//   H: reuse-after-release is covered by C's erase/restore leg.
//   R1 (Graph final-audit): padded-n_kv graph switch — totalKV1024, idle
//      seq x768 resident, active seq 64x4 (=256) vs immutable min_kv192
//      (BN64/alpha1/sink0/window0/tail0, frozen interval [0,256)). Padded
//      n_kv is a constant 1024 throughout (768+256 at the end); dense,
//      dense, transition, sparse must follow the active seq's resident
//      count via ledger plan/sparse/visible counters, with no false
//      required error on the dense legs and never a scalar route mirror.
//      The transition chunk allows mixed/all-exact (only its last query
//      reaches 192 legal tokens); a no-skip fourth chunk trips as
//      insufficient coverage, not an algorithm failure. Own immutable
//      context. The 768-token idle seed decodes in n_batch/ubatch chunks.
//   R2 (Met final-audit): full_attn_layers=UINT32_MAX all-dense —
//      REQUIRED finishes finite deterministic logits; no orphan plan work,
//      no false NO_PLAN failure. Own immutable context.
// Each --model case (B/C/F block, R1, R2) gets a fresh immutable-policy
// context, freed before the next case so VRAM never holds two contexts.
// Graph final-audit ordinary foreign-isolation fixture (CPU, no model):
// test_ordinary_foreign_isolation calls the actual ordinary planner seam
// with A cells held fixed while B-only cells (incl. an extreme image
// pattern) are added; A fragments/cell_refs/uses/groups must be identical,
// F uninflated, no foreign-forced unsupported, shared refs exactly once.
// Backend/model absence contract (never fake-covered):
//   integration requested without --model, unreadable model, init failure,
//   or unavailable --required-backend -> SKIP note, exit NONZERO.
//   unavailable merely-preferred --backend    -> SKIP note, exit follows
//   the CPU suite. Tiny controlled capacity (n_ctx=1024, n_parallel=4,
//   kv_unified). Fixed developmental policy (logged, always): REQUIRED,
//   min_kv=0, dense_tail_tiles=0, sink_blocks=0, window_blocks=0,
//   block_k=64, alpha=1.0 (RECORDED EXPERIMENTAL OVERRIDE requesting maximum
//   sparsity pressure; it does NOT force skips on any model: tied/all-equal
//   block energies keep all under the inclusive >= threshold — that all-keep
//   behavior is pinned by the selector equal-energy tests, not contradicted
//   here. A run with no skips trips the coverage gate below as insufficient
//   coverage, never as an algorithm failure. Defaults like min_kv=1024 or
//   tail=8 would route every probe dense before selection is even reached).
//   Prompts are 256 tokens (four full BN64 historical blocks) so the
//   restored suffix presents multiple selectable/correctable fragments.
//   Compared histories always share identical logical intervals, packing,
//   and chunking (same batch sizes/positions on both sides), so plan
//   selection — a logical-block function of identical resident content —
//   cannot diverge between control and restored legs. Every decode uses versioned
//   llama_decode_with_flashprefill descriptors (PREFILL rows with frozen
//   intervals for prompts/suffixes, DECODE rows for appends). Restored
//   suffixes are re-prefilled as PREFILL rows (a legal post-restore role);
//   the metrics ledger around them must show
//   eligible/sparse/selected/corrected/visible growth, and equal states
//   must agree on full finite logits vectors (1e-5 abs: identical ops on
//   identical bytes reproduce bit-near-identically; any state divergence
//   moves logits by orders more), not merely on sampled tokens.
//
// Chain mapping (PREFILL.md 17.3): every chain below names the TRUE API
// exercised here. Anything else is listed under NOT COVERED, never mapped
// onto a different action.
//   A drain-save-restore-append: owner compact remap + cp/set save/restore
//     remap (harness: state-a slot save/erase/restore/append).
//   B MTP checkpoint/partial-accept: server_rerot_runtime save_episode /
//     load_episode accept-vs-reject incl. the opaque mtp_blob checkpoint
//     (harness: no representable leg — INCOMPLETE there).
//   C RAM cell reuse: owner clear/reuse + shared-ref removal (harness:
//     state-c interleaved-reuse/restore/append leg; demotion proper has no
//     endpoint — PARTIAL there).
//   D RERoT publication: rerot_set/publish/reclassify/collect + generation
//     invalidation (harness: no representable leg — INCOMPLETE there).
//   E context shift: server_rerot_runtime::context_shift +
//     server_rerot_truncate_oldest_public incl. barrier/epoch/no-op/unknown
//     rules (harness: no representable leg — INCOMPLETE there).
//   F capacity retry/streaming: checked packing/scratch/wire/budget
//     capacity rejects (harness: state-f streaming terminal leg; shortage
//     inducement has no endpoint — PARTIAL there).
//   G pen suspend/resume: suspend_pen/resume_pen + scheduler counts across
//     a frontier (harness: no representable leg — INCOMPLETE there).
//   H cancel/reuse: owner clear/reuse + serial-domain rule (harness:
//     state-h cancel/reuse/health leg).
// NOT COVERED here (need a live context/decode; integration-level):
//   per-seq stamp roll-forward (llama_rerot_context_apply_shift), MTP draft
//   staleness predicate (llama_rerot_mtp_is_stale — covered with a stub
//   context in tests/test-rerot-runtime.cpp, not repeated), recurrent-state
//   rewrite observation under shift, GPU pool/plan content.
//
// Real counters referenced (in-memory, ServerRouting; no Prometheus series
// yet — never asserted here, inspected in server logs):
//   fp_eligible_rows, fp_dense_by_reason[7], fp_policy_fingerprint,
//   sidecar "<slotfile>.flashprefill"
//   ("flashprefill_slot_v1 <fp-hex> <mode>").

// This test uses explicit CHECK reports (not assert) as its failure
// mechanism, but the owner below uses assert() internally: keep those active
// when the surrounding project is configured with CMAKE_BUILD_TYPE=Release.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../ggml/include/ggml-backend.h"
#include "../ggml/include/ggml-flashprefill.h"
#include "../src/llama-flashprefill-fixture.h"
#include "../src/llama-flashprefill-layout.h"
#include "../src/llama-flashprefill.h"
#include "../src/llama-kv-cells.h"
#include "../tools/server/server-rerot.h"
#include "llama-flashprefill.h"
#include "test-flashprefill-state-envelope.h"

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-context.h"
#include "../src/llama-flashprefill-metrics.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

namespace {

// Test-local pending-tag factory (real owner metadata, no lifecycle copy).
llama_kv_rerot_meta make_pending_meta(
        uint64_t episode, uint32_t node, uint32_t run, uint64_t frontier) {
    llama_kv_rerot_meta m;
    m.episode_id     = episode;
    m.node_id        = node;
    m.run_id         = run;
    m.visibility     = llama_rerot_visibility::pending_record;
    m.publish_epoch  = 0;
    m.frontier       = frontier;
    return m;
}

// Test-local plan-freshness predicate over the owner's (stamp, generation)
// identity. Disabled (0) and saturated (MAX) keys never authorize reuse;
// otherwise the stamped pair must match exactly (composite owner ops may
// advance by more than one, so callers compare inequality, never +1).
bool plan_usable(uint64_t plan_stamp, uint64_t plan_gen,
        uint64_t owner_stamp, uint64_t owner_gen) {
    static const uint64_t kMax = (std::numeric_limits<uint64_t>::max)();
    if (plan_stamp == 0 || plan_stamp == kMax) {
        return false;
    }
    if (plan_gen == kMax || owner_stamp == kMax || owner_gen == kMax) {
        return false;
    }
    return plan_stamp == owner_stamp && plan_gen == owner_gen;
}

// Test-local same-context domain over ContextIntegration's per-context
// serial (flashprefill_context_serial(); 0 == "unknown", every OFF
// context). Durable restore must additionally match the policy
// fingerprint — the serial alone is not a policy check (StatePolicy's
// sidecar persists both).
bool same_serial_domain(uint64_t a, uint64_t b) {
    return a != 0 && a == b;
}

// Caller-supplied identity triple for the RERoT episode codec (fixed test
// values; production derives them from the live model via
// llama_rerot_context_fingerprints).
server_rerot_state_fingerprints test_fingerprints() {
    server_rerot_state_fingerprints fp;
    fp.caps =
        LLAMA_REROT_STATE_CAP_REROT |
        LLAMA_REROT_STATE_CAP_REROT_TREE |
        LLAMA_REROT_STATE_CAP_REROT_PRIVATE |
        LLAMA_REROT_STATE_CAP_REROT_MTP |
        LLAMA_REROT_STATE_CAP_HYBRID_REC |
        LLAMA_REROT_STATE_CAP_SPARSE_KV |
        LLAMA_REROT_STATE_CAP_TRIATTENTION;
    fp.model_fp = 0x1122334455667788ULL;
    fp.rope_fp  = 0x8877665544332211ULL;
    fp.tri_fp   = 0x0f1e2d3c4b5a6978ULL;
    return fp;
}

// Minimal commit scaffolding over the runtime planner (same call pattern as
// tests/test-rerot-runtime.cpp; nullptr-memory runtimes, no decode).
bool commit_private_token(
        server_rerot_runtime & runtime, uint64_t episode_id,
        llama_rerot_node_id node_id, llama_pos pos) {
    const auto plan = runtime.plan_private_token(episode_id, node_id, pos);
    return plan.has_value() && runtime.commit_token(episode_id, node_id, *plan);
}

bool commit_generated_token(
        server_rerot_runtime & runtime, uint64_t episode_id,
        llama_rerot_node_id node_id, llama_pos pos, const std::string & bytes) {
    const auto plan = runtime.plan_generated_token(episode_id, node_id, pos, bytes);
    return plan.has_value() && runtime.commit_token(episode_id, node_id, *plan);
}

// Admit one child lane through heading/publish/private/commit/admission,
// returning false (never throwing) when any step refuses.
bool start_child_lane(
        server_rerot_runtime & runtime, uint64_t episode_id,
        int slot, llama_seq_id exec_seq, llama_rerot_node_id expected_node) {
    llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
    if (!runtime.admit_next_child(episode_id, slot, exec_seq, &admitted) ||
            admitted != expected_node) {
        return false;
    }
    auto * node = runtime.node(episode_id, admitted);
    if (node == nullptr) {
        return false;
    }
    const auto heading = runtime.plan_heading_token(episode_id, admitted, node->storage_pos_next);
    if (!heading.has_value() || !heading->is_heading) {
        return false;
    }
    if (!runtime.commit_token(episode_id, admitted, *heading)) {
        return false;
    }
    if (!runtime.publish_heading(episode_id, admitted, heading->run_id)) {
        return false;
    }
    node = runtime.node(episode_id, admitted);
    if (node == nullptr) {
        return false;
    }
    if (!commit_private_token(runtime, episode_id, admitted, node->storage_pos_next)) {
        return false;
    }
    return runtime.complete_admission(episode_id, admitted);
}

// Chains A/B/C (shared prefix): removing branch A's reference to a shared
// cell must not change branch B's pool inputs (count + positions).
void test_shared_ref_remove_keeps_peer() {
    llama_kv_cells cells;
    cells.resize(8);
    cells.set_generation_enabled(true);

    // Shared prefix 0..2 (A=0, B=1); A-private 3; B-private 4.
    for (uint32_t i = 0; i < 3; ++i) {
        cells.pos_set(i, 100 + (llama_pos) i);
        cells.seq_add(i, 0);
        cells.seq_add(i, 1);
    }
    cells.pos_set(3, 200);
    cells.seq_add(3, 0);
    cells.pos_set(4, 300);
    cells.seq_add(4, 1);
    CHECK(cells.seq_get_used(0) == 4);
    CHECK(cells.seq_get_used(1) == 4);

    const uint64_t gen_before = cells.get_generation();
    // Branch A drops the shared prefix (e.g., Tri maintenance scoped to A).
    for (uint32_t i = 0; i < 3; ++i) {
        CHECK(!cells.seq_rm(i, 0)); // peer keeps each cell alive
    }
    CHECK(cells.get_generation() != gen_before);

    // B's derived-cache inputs are unchanged: same count, same positions,
    // still resident; A keeps only its private cell.
    CHECK(cells.seq_get_used(1) == 4);
    CHECK(cells.seq_get_used(0) == 1);
    for (uint32_t i = 0; i < 3; ++i) {
        CHECK(!cells.is_empty(i));
        CHECK(cells.seq_has(i, 1));
        CHECK(!cells.seq_has(i, 0));
        CHECK(cells.pos_get(i) == 100 + (llama_pos) i);
    }
    CHECK(cells.seq_has(3, 0));
    CHECK(cells.seq_has(4, 1));
}

// Chain B (pollution probe): huge values in cells no reader-A row may see
// must not enter A's descriptors, pool counts, or min/max bounds.
void test_foreign_poison_excluded() {
    llama_kv_cells cells;
    cells.resize(8);
    cells.set_generation_enabled(true);

    cells.pos_set(0, 10);
    cells.seq_add(0, 0);
    cells.pos_set(1, 11);
    cells.seq_add(1, 0);
    CHECK(cells.seq_get_used(0) == 2);

    // Another reader's cell carries huge values plus an active RERoT tag.
    cells.pos_set(2, (llama_pos) 1000000000);
    cells.seq_add(2, 7);
    cells.ext_set(2, {(llama_pos) 1000000, (llama_pos) -1000000});
    cells.rerot_set(2, make_pending_meta(55, 1, 2, 9));

    CHECK(cells.seq_get_used(0) == 2);
    CHECK(cells.pos_get(0) == 10);
    CHECK(cells.pos_get(1) == 11);
    CHECK(cells.seq_pos_min(0) == 10);
    CHECK(cells.seq_pos_max(0) == 11);
    uint32_t a_count = 0;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, 0)) {
            CHECK(cells.pos_get(i) < (llama_pos) 1000000);
            ++a_count;
        }
    }
    CHECK(a_count == 2);
}

// Chains A/C (compaction): the physical remap preserves logical values,
// positions, per-sequence membership, extents, and RERoT metadata.
void test_compact_remap_preserves() {
    llama_kv_cells cells;
    cells.resize(12);
    cells.set_generation_enabled(true);

    const uint32_t phys[] = {0, 1, 2, 5, 6, 10};
    for (uint32_t k = 0; k < 6; ++k) {
        const uint32_t i = phys[k];
        cells.pos_set(i, 10 * (llama_pos) i + 3);
        cells.seq_add(i, 0);
        cells.ext_set(i, {(llama_pos) i, (llama_pos) -((llama_pos) i)});
    }
    cells.seq_add(6, 1); // one shared cell
    const llama_kv_rerot_meta tag = make_pending_meta(77, 4, 5, 3);
    cells.rerot_set(5, tag);

    std::vector<llama_pos> poss;
    std::vector<bool>      has1;
    for (uint32_t k = 0; k < 6; ++k) {
        poss.push_back(cells.pos_get(phys[k]));
        has1.push_back(cells.seq_has(phys[k], 1));
    }
    const llama_kv_cell_ext ext5 = cells.ext_get(5);
    CHECK(cells.seq_get_used(0) == 6);
    CHECK(cells.seq_get_used(1) == 1);

    const uint64_t gen_before = cells.get_generation();
    kv_pack_plan plan = cells.make_pack_plan();
    CHECK(plan.retained_count == 6);
    CHECK(cells.get_generation() == gen_before); // planning is read-only
    cells.apply_pack(plan);
    CHECK(cells.get_generation() != gen_before);

    // Dense [0, 6) in ascending source order: phys 5 -> dense 3.
    CHECK(cells.get_used() == 6);
    CHECK(cells.used_min() == 0);
    CHECK(cells.used_max_p1() == 6);
    for (uint32_t k = 0; k < 6; ++k) {
        CHECK(!cells.is_empty(k));
        CHECK(cells.pos_get(k) == poss[k]);
        CHECK(cells.seq_has(k, 0));
        CHECK(cells.seq_has(k, 1) == has1[k]);
    }
    CHECK(cells.seq_get_used(0) == 6);
    CHECK(cells.seq_get_used(1) == 1);
    CHECK(cells.ext_get(3).x == ext5.x && cells.ext_get(3).y == ext5.y);
    CHECK(cells.rerot_get(3) == tag);
}

// Chain B (MTP partial accept): rejected draft content — including huge
// speculative values written before the rollback decision — must not leak
// into the rebuild; reuse after rollback carries no residue.
void test_draft_rollback_excludes_rejected() {
    llama_kv_cells cells;
    cells.resize(8);
    cells.set_generation_enabled(true);

    cells.pos_set(0, 0);
    cells.seq_add(0, 0);
    cells.pos_set(1, 1);
    cells.seq_add(1, 0);
    // Two drafts append as new refs.
    cells.pos_set(2, 2);
    cells.seq_add(2, 0);
    cells.pos_set(3, 3);
    cells.seq_add(3, 0);
    CHECK(cells.seq_get_used(0) == 4);

    // Partial accept: draft 2 stays; draft 3's bytes turn huge, then the
    // rollback frees the cell.
    cells.pos_add(3, (llama_pos) 999999999);
    CHECK(cells.pos_get(3) == (llama_pos) 1000000002);
    CHECK(cells.seq_rm(3, 0)); // sole ref removed: cell freed
    CHECK(cells.is_empty(3));
    CHECK(cells.seq_get_used(0) == 3);

    // Reuse of the freed physical cell carries no residue.
    cells.pos_set(3, 30);
    cells.seq_add(3, 0);
    CHECK(cells.pos_get(3) == 30);
    CHECK(cells.seq_get_used(0) == 4);

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i)) {
            CHECK(cells.pos_get(i) < (llama_pos) 1000000);
        }
    }
    CHECK(cells.pos_get(0) == 0 && cells.pos_get(1) == 1 && cells.pos_get(2) == 2);
}

// Chains C/H (slot reuse, cancel release): clear is an epoch boundary —
// refs gone, generation bumped, identity preserved — and the next request
// inherits nothing.
void test_clear_reuse() {
    llama_kv_cells cells;
    cells.resize(6);
    cells.set_generation_enabled(true);

    cells.pos_set(0, 5);
    cells.seq_add(0, 0);
    cells.pos_set(1, 6);
    cells.seq_add(1, 1);
    const uint64_t stamp = cells.get_generation_stamp();
    const uint64_t gen   = cells.get_generation();

    cells.reset();
    CHECK(cells.get_used() == 0);
    CHECK(cells.seq_get_used(0) == 0);
    CHECK(cells.seq_get_used(1) == 0);
    CHECK(cells.seq_pos_min(0) == -1);
    CHECK(cells.get_generation_enabled());
    CHECK(cells.get_generation_stamp() == stamp);
    CHECK(cells.get_generation() != gen);
    CHECK(cells.get_generation() != 0);

    cells.pos_set(0, 50);
    cells.seq_add(0, 2);
    CHECK(cells.seq_get_used(2) == 1);
    CHECK(cells.seq_get_used(0) == 0);
    CHECK(cells.seq_pos_min(2) == 50);
    CHECK(!cells.seq_has(0, 0) && !cells.seq_has(0, 1));
}

// Chains A/C (RAM save/restore): metadata round-trips through the existing
// cp/set snapshot primitives (the same pair the state metadata path is
// built on), including restore onto remapped physical indices.
void test_save_restore_remapped() {
    llama_kv_cells cells;
    cells.resize(8);
    cells.set_generation_enabled(true);

    for (uint32_t i = 0; i < 4; ++i) {
        cells.pos_set(i, 10 + (llama_pos) i);
        cells.seq_add(i, 0);
    }
    const llama_kv_cells saved = cells.cp(0, 4);

    for (uint32_t i = 0; i < 4; ++i) {
        cells.rm(i);
    }
    CHECK(cells.get_used() == 0);
    CHECK(cells.seq_get_used(0) == 0);

    const uint64_t gen_erased = cells.get_generation();
    cells.set(0, saved); // restore in place
    CHECK(cells.get_generation() != gen_erased);
    CHECK(cells.seq_get_used(0) == 4);
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(!cells.is_empty(i));
        CHECK(cells.pos_get(i) == 10 + (llama_pos) i);
        CHECK(cells.seq_has(i, 0));
    }

    // Remapped restore: the same logical snapshot lands on different
    // physical cells (e.g., RAM restore after reuse moved the low cells).
    const std::vector<uint32_t> idxs{4, 5, 6, 7};
    cells.set(idxs, saved);
    CHECK(cells.seq_get_used(0) == 8);
    for (uint32_t i = 0; i < 4; ++i) {
        cells.rm(i);
    }
    CHECK(cells.get_used() == 4);
    CHECK(cells.seq_get_used(0) == 4);
    for (uint32_t k = 0; k < 4; ++k) {
        CHECK(!cells.is_empty(4 + k));
        CHECK(cells.pos_get(4 + k) == 10 + (llama_pos) k);
        CHECK(cells.seq_has(4 + k, 0));
    }
    // Indexed snapshot of the remapped range carries the same raw state
    // (snapshots hold pos/ext/seq/rerot rows; used/seq_pos are rebuilt by
    // set(), so only the rows are pinned here).
    const llama_kv_cells by_idx = cells.cp(idxs);
    for (uint32_t k = 0; k < 4; ++k) {
        CHECK(by_idx.pos_get(k) == 10 + (llama_pos) k);
        CHECK(by_idx.seq_has(k, 0));
    }
}

// Chains D/E/G (RERoT publish, context shift, pen preemption): every owner
// mutation class — including tag/publish/reclassify view changes, packing,
// and restore — invalidates earlier plans; MAX/zero keys never authorize
// reuse. Stable run identity still resolves while the plan is stale.
void test_generation_view_invalidation() {
    // Disabled by default: untracked, never a reusable plan key.
    llama_kv_cells untracked;
    untracked.resize(4);
    CHECK(!untracked.get_generation_enabled());
    CHECK(untracked.get_generation() == 0);
    CHECK(untracked.get_generation_stamp() == 0);
    CHECK(!plan_usable(0, 0, 0, 0));

    llama_kv_cells cells;
    cells.resize(6);
    cells.set_generation_enabled(true);
    const uint64_t stamp = cells.get_generation_stamp();
    CHECK(stamp != 0);

    uint64_t g = cells.get_generation();
    CHECK(plan_usable(stamp, g, stamp, g));
    cells.pos_set(0, 1);
    CHECK(!plan_usable(stamp, g, stamp, cells.get_generation()));
    g = cells.get_generation();
    cells.seq_add(0, 0);
    CHECK(!plan_usable(stamp, g, stamp, cells.get_generation()));
    g = cells.get_generation();
    cells.ext_set(0, {(llama_pos) 7, (llama_pos) 8});
    CHECK(!plan_usable(stamp, g, stamp, cells.get_generation()));

    // View change invalidates: pending tag -> public commit -> reclassify.
    g = cells.get_generation();
    cells.rerot_set(0, make_pending_meta(9, 1, 1, 0));
    CHECK(!plan_usable(stamp, g, stamp, cells.get_generation()));
    CHECK(cells.rerot_has_active());
    std::vector<uint32_t> found;
    CHECK(cells.rerot_collect_run(9, 1, found) == 1 && found.size() == 1 && found[0] == 0);
    g = cells.get_generation();
    CHECK(cells.rerot_publish(0, 9, 1, 42));
    CHECK(!plan_usable(stamp, g, stamp, cells.get_generation()));
    found.clear();
    CHECK(cells.rerot_collect_run(9, 1, found) == 1 && found.size() == 1 && found[0] == 0);
    g = cells.get_generation();
    CHECK(cells.rerot_reclassify(0, 9, 1,
            llama_rerot_visibility::public_live,
            llama_rerot_visibility::private_control, 0));
    CHECK(!plan_usable(stamp, g, stamp, cells.get_generation()));

    // Pack planning is read-only; applying the pack invalidates.
    cells.pos_set(3, 30);
    cells.seq_add(3, 0);
    g = cells.get_generation();
    kv_pack_plan plan = cells.make_pack_plan();
    CHECK(cells.get_generation() == g);
    CHECK(plan_usable(stamp, g, stamp, cells.get_generation()));
    cells.apply_pack(plan);
    CHECK(!plan_usable(stamp, g, stamp, cells.get_generation()));

    // Saturated keys are always-invalid, never reusable.
    static const uint64_t kMax = (std::numeric_limits<uint64_t>::max)();
    CHECK(!plan_usable(kMax, 1, kMax, 1));
    CHECK(!plan_usable(stamp, kMax, stamp, kMax));
    CHECK(!plan_usable(stamp, g, stamp, kMax));
}

// Chain F (capacity shortage): checked arithmetic rejects overflow and
// garbage capacities with explicit errors — never silent truncation — and
// OFF sizes zero.
void test_overflow_capacity_rejects() {
    uint32_t total = 0, tiles = 0;
    CHECK(llama_flashprefill::packed_layout_checked((1u << 24), 256u, 128u, &total, &tiles) ==
            LLAMA_FLASHPREFILL_ERR_OVERFLOW);
    CHECK(llama_flashprefill::packed_layout_checked(64u, 4u, 128u, &total, &tiles) ==
            LLAMA_FLASHPREFILL_OK);
    CHECK(total == 256u && tiles == 2u);
    CHECK(llama_flashprefill::packed_layout_checked(1u, 1u, 128u, nullptr, &tiles) ==
            LLAMA_FLASHPREFILL_ERR_NULL);

    const llama_flashprefill_config off = llama_flashprefill_default_config();
    CHECK(!llama_flashprefill_is_enabled(&off));
    llama_flashprefill::scratch_inputs in{};
    in.n_fragments  = 8;
    in.n_kv_heads   = 2;
    in.d_k          = 64;
    in.d_v          = 64;
    in.n_packed_rows = 128;
    in.n_splits     = 1;
    in.mean_bytes   = 4;
    uint64_t bytes = 0;
    CHECK(llama_flashprefill::scratch_bytes_checked(&off, &in, &bytes) ==
            LLAMA_FLASHPREFILL_OK && bytes == 0);
    llama_flashprefill_config auto_cfg = llama_flashprefill_default_config();
    auto_cfg.mode = LLAMA_FLASHPREFILL_MODE_AUTO;
    CHECK(llama_flashprefill_validate_config(&auto_cfg) == LLAMA_FLASHPREFILL_OK);
    CHECK(llama_flashprefill::scratch_bytes_checked(&auto_cfg, &in, &bytes) ==
            LLAMA_FLASHPREFILL_OK && bytes > 0);
    llama_flashprefill::scratch_inputs huge = in;
    huge.n_fragments = (1u << 24) + 1u;
    CHECK(llama_flashprefill::scratch_bytes_checked(&auto_cfg, &huge, &bytes) ==
            LLAMA_FLASHPREFILL_ERR_COUNT);
    llama_flashprefill::scratch_inputs bad_mean = in;
    bad_mean.mean_bytes = 3;
    CHECK(llama_flashprefill::scratch_bytes_checked(&auto_cfg, &bad_mean, &bytes) ==
            LLAMA_FLASHPREFILL_ERR_FLAG);
    CHECK(llama_flashprefill::scratch_bytes_checked(&auto_cfg, nullptr, &bytes) ==
            LLAMA_FLASHPREFILL_ERR_NULL);

    // Wire metadata (ggml-side schema, WireReference-owned; consumed here):
    // words/init/set/validate reject oversize/negative caps, short buffers,
    // over-cap counts, bad flags, out-of-range refs, and corrupt magic with
    // explicit codes — never silent truncation.
    int64_t words = 0;
    CHECK(ggml_flashprefill_metadata_words(2, 2, 2, 8, &words) ==
            GGML_FLASHPREFILL_OK && words > 32);
    CHECK(ggml_flashprefill_metadata_words(-1, 2, 2, 8, &words) ==
            GGML_FLASHPREFILL_ERR_BAD_ARG);
    CHECK(ggml_flashprefill_metadata_words(2, 2, 2, 8, nullptr) ==
            GGML_FLASHPREFILL_ERR_BAD_ARG);
    std::vector<int32_t> meta((size_t) words, 0);
    CHECK(ggml_flashprefill_metadata_init(meta.data(), words, 2, 2, 2, 8, 64, 64, 2, 1, 4) ==
            GGML_FLASHPREFILL_OK);
    CHECK(ggml_flashprefill_metadata_validate(meta.data(), words) ==
            GGML_FLASHPREFILL_OK);
    CHECK(ggml_flashprefill_metadata_set_counts(meta.data(), words, 0, 0, 0, 0) ==
            GGML_FLASHPREFILL_OK);
    CHECK(ggml_flashprefill_metadata_set_counts(meta.data(), words, 3, 0, 0, 0) ==
            GGML_FLASHPREFILL_ERR_CAP_EXCEEDED);
    CHECK(ggml_flashprefill_metadata_set_frag(meta.data(), words, 0, 0, 0, 0, 0, 0) ==
            GGML_FLASHPREFILL_OK);
    CHECK(ggml_flashprefill_metadata_set_frag(meta.data(), words, 0, 0, 0, 0, 0, 0x40) ==
            GGML_FLASHPREFILL_ERR_BAD_FLAG);
    CHECK(ggml_flashprefill_metadata_set_frag(meta.data(), words, 9, 0, 0, 0, 0, 0) ==
            GGML_FLASHPREFILL_ERR_BAD_RANGE);
    CHECK(ggml_flashprefill_metadata_init(meta.data(), words - 1, 2, 2, 2, 8, 64, 64, 2, 1, 4) ==
            GGML_FLASHPREFILL_ERR_BAD_LAYOUT);
    CHECK(ggml_flashprefill_metadata_init(nullptr, words, 2, 2, 2, 8, 64, 64, 2, 1, 4) ==
            GGML_FLASHPREFILL_ERR_BAD_ARG);
    meta[0] = 0;
    CHECK(ggml_flashprefill_metadata_validate(meta.data(), words) ==
            GGML_FLASHPREFILL_ERR_BAD_MAGIC);
    // Wire config validation names the offending field.
    struct ggml_flashprefill_config wcfg = ggml_flashprefill_config_default();
    int32_t bad_field = -1;
    CHECK(ggml_flashprefill_config_validate(&wcfg, &bad_field) == GGML_FLASHPREFILL_OK);
    struct ggml_flashprefill_config wbad = wcfg;
    wbad.alpha = 0.0f;
    CHECK(ggml_flashprefill_config_validate(&wbad, &bad_field) ==
            GGML_FLASHPREFILL_ERR_BAD_CONFIG);
    CHECK(bad_field == GGML_FLASHPREFILL_FIELD_ALPHA);
    CHECK(ggml_flashprefill_config_validate(nullptr, &bad_field) ==
            GGML_FLASHPREFILL_ERR_BAD_ARG);

    // Compact-layout fragment budget (CacheFragments): checked 64-bit math
    // with hard I32 wire-domain failure, never a saturating clamp.
    llama_flashprefill_fragment_budget budget{};
    std::string budget_error;
    CHECK(llama_flashprefill_fragment_budget_for(65536, 128, 1, 2, 4, 8, &budget, &budget_error));
    CHECK(budget.n_fragments == 1030 && budget.n_uses == 8240 && budget.n_exact_rows == 131840);
    CHECK(!llama_flashprefill_fragment_budget_for(65536, 0, 1, 2, 4, 8, &budget, &budget_error));
    CHECK(!budget_error.empty());
    budget_error.clear();
    CHECK(!llama_flashprefill_fragment_budget_for(UINT32_MAX, 64, 1, 64, 0, 1, &budget, &budget_error));
    CHECK(!budget_error.empty());
    CHECK(!llama_flashprefill_fragment_budget_for(65536, 128, 1, 2, 4, 8, nullptr, &budget_error));
}

// All durable legs (A/C/H restores, RAM round-trips): mode is part of the
// policy identity, invalid configs never mint one, and the serial rule keeps
// cross-context restores conservative even when fingerprints match.
void test_policy_identity_separates() {
    const llama_flashprefill_config off = llama_flashprefill_default_config();
    llama_flashprefill_config auto_cfg = off;
    auto_cfg.mode = LLAMA_FLASHPREFILL_MODE_AUTO;
    uint64_t fp_off = 0, fp_auto = 0;
    CHECK(llama_flashprefill_fingerprint(&off, "model-A", nullptr, &fp_off) ==
            LLAMA_FLASHPREFILL_OK);
    CHECK(llama_flashprefill_fingerprint(&auto_cfg, "model-A", nullptr, &fp_auto) ==
            LLAMA_FLASHPREFILL_OK);
    CHECK(fp_off != fp_auto);

    llama_flashprefill_config bad = auto_cfg;
    bad.alpha = 0.0f;
    uint64_t fp_bad = 0;
    CHECK(llama_flashprefill_fingerprint(&bad, "model-A", nullptr, &fp_bad) !=
            LLAMA_FLASHPREFILL_OK);

    CHECK(same_serial_domain(7, 7));
    CHECK(!same_serial_domain(7, 8));
    CHECK(!same_serial_domain(0, 0));
    CHECK(!same_serial_domain(0, 7));
}

// Chain B (MTP checkpoint accept/reject): the real episode codec round-trip
// preserves the opaque mtp_blob checkpoint plus sampler bytes and view
// stamps; wrong-identity and truncated blobs are refused, never partially
// restored. nullptr-memory runtimes; no decode.
void test_rerot_checkpoint_accept_reject() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t ep = runtime.adopt_root(280, 281, 0, 0, 10, 77);
    CHECK(ep == 77);
    CHECK(commit_private_token(runtime, ep, 0, 10));
    CHECK(commit_generated_token(runtime, ep, 0, 11,
        "<ol><li>Persist child A</li><li>Persist child B</li></ol>"));
    CHECK(runtime.finish_frontier(ep).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(ep, 0));
    CHECK(start_child_lane(runtime, ep, 0, 0, 1));

    auto * lane = runtime.node(ep, 1);
    CHECK(lane != nullptr);
    if (lane != nullptr) {
        lane->sampler_blob = {1, 2, 3, 4};
        lane->mtp_blob     = {9, 8, 7};
        lane->view_stamp   = {4, 5, 6};
    }
    const auto fp = test_fingerprints();
    std::vector<uint8_t> blob;
    std::string error;
    CHECK(runtime.save_episode(ep, fp, &blob, &error));
    CHECK(!blob.empty() && error.empty());

    // Accept: logical tree plus opaque checkpoint state restore verbatim.
    server_rerot_runtime restored(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    uint64_t restored_id = 0;
    CHECK(restored.load_episode(blob.data(), blob.size(), fp, &restored_id, &error));
    CHECK(restored_id == ep);
    const auto * rlane = restored.node(restored_id, 1);
    CHECK(rlane != nullptr);
    CHECK(rlane && rlane->sampler_blob == std::vector<uint8_t>({1, 2, 3, 4}));
    CHECK(rlane && rlane->mtp_blob == std::vector<uint8_t>({9, 8, 7}));
    CHECK(rlane && rlane->view_stamp.topology_epoch == 4);
    CHECK(rlane && rlane->view_stamp.publish_epoch == 5);
    CHECK(rlane && rlane->view_stamp.layout_epoch == 6);
    CHECK(restored.episode(restored_id) != nullptr);
    CHECK(restored.episode(restored_id)->document.validate(&error));

    // Reject: wrong model identity is named, never best-effort restored.
    server_rerot_runtime wrong(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    auto bad_fp = fp;
    ++bad_fp.model_fp;
    error.clear();
    CHECK(!wrong.load_episode(blob.data(), blob.size(), bad_fp, nullptr, &error));
    CHECK(error.find("model fingerprint mismatch") != std::string::npos);

    // Reject: truncated bytes never restore, with a reason.
    std::vector<uint8_t> cut(blob.begin(), blob.end() - (blob.size() > 16 ? 16 : 0));
    error.clear();
    CHECK(!wrong.load_episode(cut.data(), cut.size(), fp, nullptr, &error));
    CHECK(!error.empty());
}

// Chain E (context shift): the real runtime shift entry drops only the
// oldest unpinned public region, bumps layout/publish epochs, raises the
// barrier, leaves the pinned prefix and active run intact, and is a stable
// no-op once quiesced; unknown episodes are refused without mutation.
void test_context_shift_runtime_api() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
    const uint64_t ep = runtime.adopt_root(9, 9, 0, 0, 0, 900);
    auto * episode = runtime.episode(ep);
    CHECK(episode != nullptr);
    episode->base_prefix_end = 10;
    episode->publish_epoch   = 3;
    episode->layout_epoch    = 7;
    const auto prefix = episode->document.append_run(
        0, llama_rerot_visibility::public_live, 0, 10, 1);
    const auto old_public = episode->document.append_run(
        0, llama_rerot_visibility::public_live, 10, 4, 2);
    const auto active_public = episode->document.append_run(
        0, llama_rerot_visibility::public_live, 14, 3, 3);
    CHECK(prefix != LLAMA_REROT_RUN_INVALID);
    CHECK(old_public != LLAMA_REROT_RUN_INVALID);
    CHECK(active_public != LLAMA_REROT_RUN_INVALID);
    auto * lane0 = runtime.node(ep, 0);
    CHECK(lane0 != nullptr);
    if (lane0 != nullptr) {
        lane0->public_run = active_public;
    }

    server_rerot_shift_result result;
    std::string error;
    CHECK(runtime.context_shift(ep, 3, &result, &error));
    CHECK(error.empty());
    CHECK(result.tokens_removed == 4);
    CHECK(result.runs_truncated == 1 && result.runs_emptied == 1);
    CHECK(result.new_layout_epoch == 8 && result.new_publish_epoch == 4);
    CHECK(episode->topology_barrier_pending);
    CHECK(episode->document.run(prefix)->token_count == 10);
    CHECK(episode->document.run(old_public)->token_count == 0);
    CHECK(episode->document.run(active_public)->token_count == 3);

    // Quiesced shift removes nothing and moves no epoch.
    const uint64_t layout_after  = episode->layout_epoch;
    const uint64_t publish_after = episode->publish_epoch;
    CHECK(runtime.context_shift(ep, 100, &result, &error));
    CHECK(result.tokens_removed == 0);
    CHECK(episode->layout_epoch == layout_after);
    CHECK(episode->publish_epoch == publish_after);

    CHECK(!runtime.context_shift(999, 3, nullptr, nullptr));
    CHECK(runtime.erase_episode(ep));
}

// Chain G (pen suspend/resume): frontier-boundary yield parks the lane
// (pen/slot released, node ready_suspended, frontier completable without
// it) and resume rebinds a pen back to running; erase drains all counts.
void test_pen_suspend_resume() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
    runtime.set_pen_capacity(2);
    CHECK(runtime.pen_capacity() == 2);
    CHECK(runtime.pens_allocated() == 0 && runtime.pens_suspended() == 0);

    const uint64_t ep = runtime.adopt_root(1, 1, 0, 10, 0, 1000);
    CHECK(runtime.pens_allocated() == 1);
    CHECK(commit_generated_token(runtime, ep, 0, 0, "<ol><li>Child 1</li><li>Child 2</li></ol>"));
    runtime.finish_frontier(ep);
    CHECK(runtime.freeze_fork_parent(ep, 0));
    CHECK(runtime.pens_allocated() == 0);

    CHECK(runtime.schedule_pens({ep}) == 2);
    CHECK(runtime.pens_allocated() == 2 && runtime.pens_running() == 0);
    CHECK(runtime.complete_admission(ep, 1));
    CHECK(runtime.complete_admission(ep, 2));
    CHECK(runtime.pens_running() == 2);

    const auto * ep_ptr = runtime.episode(ep);
    CHECK(ep_ptr->running.count(1) == 1 && ep_ptr->running.count(2) == 1);
    CHECK(ep_ptr->suspended.empty());

    // Child 1 yields pen 0 at the frontier boundary.
    CHECK(runtime.suspend_pen(0));
    CHECK(runtime.pens_suspended() == 1);
    CHECK(runtime.pens_running() == 1 && runtime.pens_allocated() == 1);
    CHECK(ep_ptr->running.count(1) == 0 && ep_ptr->running.count(2) == 1);
    CHECK(ep_ptr->suspended.count(1) == 1);
    CHECK(runtime.node(ep, 1)->pen_id == -1);
    CHECK(runtime.node(ep, 1)->physical_slot == -1);
    CHECK(ep_ptr->document.node(1)->state == llama_rerot_node_state::ready_suspended);
    CHECK(runtime.has_ready_nodes(ep));

    // A frontier completes without the suspended lane; it stays parked.
    const auto f1 = runtime.finish_frontier(ep);
    CHECK(!f1.natural_final());
    CHECK(f1.final_node == LLAMA_REROT_NODE_INVALID);

    // Resume rebinds pen 0 and returns the lane to running.
    const auto pen0 = runtime.allocate_pen(ep, ep, 1);
    CHECK(pen0.has_value() && *pen0 == 0);
    CHECK(runtime.resume_pen(ep, 1, 0, 10));
    CHECK(runtime.pens_suspended() == 0);
    CHECK(runtime.pens_running() == 2 && runtime.pens_allocated() == 2);
    CHECK(ep_ptr->suspended.empty());
    CHECK(ep_ptr->running.count(1) == 1);
    CHECK(runtime.node(ep, 1)->pen_id == 0);

    CHECK(runtime.erase_episode(ep));
    CHECK(runtime.pens_allocated() == 0 && runtime.pens_suspended() == 0);
}

// ---- RERoT query-dependent virtualization fixture (chains D/E) ----
//
// Contract under test (CacheFragments correction): the old oracle filters
// gated future/private/untagged seq ownership BEFORE dense virtualize, so a
// reader-owned future gated cell that precedes another visible run in
// ordered_runs occupies NO virtual slot, and every later run's virtual base
// shifts down by the filtered (legal-prefix) count. The new compact table
// must expand to the same per-query (physical, effective) sets as
// llama_rerot_build_query_layout.
//
// New-side entry: CacheFragments' pure, model-free fixture table
// (src/llama-flashprefill-fixture.h, unconditional include — the header is
// landed). The old-oracle pins below run regardless and would catch a
// fixture-construction error first.

static llama_kv_rerot_meta fp_tag(
        uint64_t episode, uint32_t node, uint32_t run,
        llama_rerot_visibility vis, uint64_t pub_epoch, uint64_t frontier) {
    llama_kv_rerot_meta m;
    m.episode_id    = episode;
    m.node_id       = node;
    m.run_id        = run;
    m.visibility    = vis;
    m.publish_epoch = pub_epoch;
    m.frontier      = frontier;
    return m;
}

static llama_rerot_reader_state fp_reader(
        uint64_t episode, uint32_t reader_node, uint32_t query_run,
        uint64_t frontier, const std::vector<llama_rerot_run_id> & ordered) {
    llama_rerot_reader_state r;
    r.episode_id     = episode;
    r.reader         = reader_node;
    r.query_run      = query_run;
    r.frontier       = frontier;
    r.topology_epoch = 2;
    r.publish_epoch  = 3;
    r.layout_epoch   = 4;
    r.frontier_mode  = LLAMA_REROT_FRONTIER_STRONG;
    r.ordered_runs   = ordered;
    return r;
}

// One key record per resident owner cell, with per-query seq ownership baked
// in — the same derivation the cache performs (owned_by_reader from sequence
// references), so shared-view queries with different seqs get different keys.
static std::vector<llama_rerot_key_record> fp_keys_for_seq(
        const llama_kv_cells & cells, llama_seq_id qseq) {
    std::vector<llama_rerot_key_record> keys;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i)) {
            continue;
        }
        llama_rerot_key_record k;
        k.key_index      = i;
        k.storage_pos    = cells.pos_get(i);
        k.owned_by_reader = cells.seq_has(i, qseq);
        k.meta           = cells.rerot_get(i);
        keys.push_back(k);
    }
    return keys;
}

struct fp_q_expectation {
    llama_pos query_virtual = -1;
    std::vector<std::pair<uint32_t, llama_pos>> phys_eff; // sorted
};

static fp_q_expectation fp_run_oracle(
        const llama_rerot_reader_state & reader, llama_pos qpos,
        const std::vector<llama_rerot_key_record> & keys) {
    fp_q_expectation out;
    const llama_rerot_query_layout layout = llama_rerot_build_query_layout(reader, qpos, keys);
    out.query_virtual = layout.query_virtual_pos;
    for (const auto & e : layout.entries) {
        out.phys_eff.emplace_back(e.key_index, layout.groups[e.group_index].effective_pos);
    }
    std::sort(out.phys_eff.begin(), out.phys_eff.end());
    return out;
}

static bool fp_contains_phys(const fp_q_expectation & exp, uint32_t phys) {
    for (const auto & p : exp.phys_eff) {
        if (p.first == phys) {
            return true;
        }
    }
    return false;
}

static std::vector<llama_pos> fp_distinct_effectives(const fp_q_expectation & exp) {
    std::vector<llama_pos> eff;
    for (const auto & p : exp.phys_eff) {
        if (eff.empty() || eff.back() != p.second) {
            eff.push_back(p.second);
        }
    }
    return eff;
}

// New compact table (want_exact_rows=true oracle path) must expand to the
// same per-query (physical, effective) sets and query-virtual positions.
static bool fp_layout_matches_oracle(
        const llama_flashprefill_layout & built, uint32_t q,
        const fp_q_expectation & exp, std::string * error) {
    const auto fail = [&](const std::string & msg) {
        if (error) { *error = msg; }
        return false;
    };
    if (q >= built.queries.size() || q + 1 >= built.exact_offsets.size()) {
        return fail("fixture: query out of range in built layout");
    }
    if (built.queries[q].query_virtual_pos != exp.query_virtual) {
        return fail("fixture: query_virtual_pos mismatch");
    }
    std::vector<std::pair<uint32_t, llama_pos>> got;
    for (uint32_t e = built.exact_offsets[q]; e < built.exact_offsets[q + 1]; ++e) {
        if (e >= built.exact_rows.size() || e >= built.exact_groups.size()) {
            return fail("fixture: exact row out of range");
        }
        const uint32_t g = built.exact_groups[e];
        if (g >= built.groups.size()) {
            return fail("fixture: exact group out of range");
        }
        got.emplace_back(built.exact_rows[e], built.groups[g].effective_pos);
    }
    std::sort(got.begin(), got.end());
    if (got != exp.phys_eff) {
        return fail("fixture: expanded (physical, effective) set mismatch");
    }
    return true;
}

// Scenario V: the reader's own future gated cell (private, storage beyond
// every query) sits in ordered_runs BEFORE another visible run. It must
// occupy no virtual slot: the later run's virtual base counts only the
// legal prefix (base + earlier legal runs).
static void test_rerot_virtualization_future_gated() {
    // runs: 11 = public visible, 12 = reader-private future, 13 = reader's
    // frontier-equal public (gated), 14 = reader-private query run.
    llama_kv_cells cells;
    cells.resize(8);
    cells.set_generation_enabled(true);
    const uint32_t seq = 5;
    auto put = [&](uint32_t i, llama_pos pos) {
        cells.pos_set(i, pos);
        cells.seq_add(i, seq);
    };
    put(0, 0);
    put(1, 1);
    put(2, 2); cells.rerot_set(2, fp_tag(1, 2, 11, llama_rerot_visibility::public_live, 2, 9));
    put(3, 3); cells.rerot_set(3, fp_tag(1, 2, 11, llama_rerot_visibility::public_live, 2, 9));
    put(4, 20); cells.rerot_set(4, fp_tag(1, 1, 12, llama_rerot_visibility::private_control, 0, 10));
    put(5, 4); cells.rerot_set(5, fp_tag(1, 1, 13, llama_rerot_visibility::public_live, 3, 10));
    put(6, 7); cells.rerot_set(6, fp_tag(1, 1, 13, llama_rerot_visibility::public_live, 3, 10));
    put(7, 6); cells.rerot_set(7, fp_tag(1, 1, 14, llama_rerot_visibility::private_control, 0, 10));

    const llama_rerot_reader_state reader = fp_reader(1, 1, 14, 10, {11, 12, 13, 14});
    const auto keys = fp_keys_for_seq(cells, seq);

    // Query at the reader's own latest cell: everything causal is legal
    // except the future gated cell; the query key itself is found.
    const fp_q_expectation exp6 = fp_run_oracle(reader, 6, keys);
    CHECK(exp6.phys_eff.size() == 6); // base 2 + runA 2 + runB 1 + runQ 1
    CHECK(!fp_contains_phys(exp6, 4));
    CHECK(exp6.query_virtual == 5);
    // Hand-verified effectives: qvirt 5 spreads base/runA/runB over 5 and
    // isolates the query cell at 6 (5+6-5). The query key IS found (own
    // private cell at storage 6), so this is its dense virtual index, not
    // the legal total 6. A raw-size run_v0 would put qvirt at 7 and shift
    // every group (base over 7, query cell at 6).
    CHECK(fp_distinct_effectives(exp6) == std::vector<llama_pos>({5, 6}));

    // Earlier query: the gated runB tail and the query run drop out too.
    const fp_q_expectation exp5 = fp_run_oracle(reader, 5, keys);
    CHECK(exp5.phys_eff.size() == 5); // base 2 + runA 2 + runB 1
    CHECK(!fp_contains_phys(exp5, 4) && !fp_contains_phys(exp5, 6) && !fp_contains_phys(exp5, 7));
    CHECK(exp5.query_virtual == 5); // no matching key: next-after-visible
    CHECK(fp_distinct_effectives(exp5) == std::vector<llama_pos>({5}));

    llama_ubatch ub{};
    ub.n_tokens = 2;
    llama_pos pos_arr[2] = {6, 5};
    int32_t nseq_arr[2] = {1, 1};
    llama_seq_id s0[1] = {(llama_seq_id) seq};
    llama_seq_id s1[1] = {(llama_seq_id) seq};
    llama_seq_id * seq_arr[2] = {s0, s1};
    ub.pos = pos_arr;
    ub.n_seq_id = nseq_arr;
    ub.seq_id = seq_arr;
    std::vector<llama_rerot_reader_state> views(8);
    views[seq] = reader;
    llama_flashprefill_layout_params params;
    params.want_exact_rows = true;
    llama_flashprefill_layout built;
    std::string error;
    CHECK(llama_flashprefill_fixture_build_rerot(cells, views, ub, params, built, &error));
    CHECK(fp_layout_matches_oracle(built, 0, exp6, &error));
    CHECK(fp_layout_matches_oracle(built, 1, exp5, &error));
    if (!error.empty()) {
        std::fprintf(stderr, "fixture mismatch: %s\n", error.c_str());
    }
}

// Scenario S: two queries share one reader view but own different untagged
// base cells; the query-run private cell is owned by one seq only (found vs
// next-after query-virtual contrast).
static void test_rerot_virtualization_shared_view() {
    llama_kv_cells cells;
    cells.resize(8);
    cells.set_generation_enabled(true);
    cells.pos_set(0, 0); cells.seq_add(0, 5);
    cells.pos_set(1, 1); cells.seq_add(1, 5); cells.seq_add(1, 6);
    cells.pos_set(2, 2); cells.seq_add(2, 6);
    cells.pos_set(3, 3); cells.seq_add(3, 5);
    cells.rerot_set(3, fp_tag(2, 9, 21, llama_rerot_visibility::public_live, 2, 5));
    cells.pos_set(4, 4); cells.seq_add(4, 6);
    cells.rerot_set(4, fp_tag(2, 9, 21, llama_rerot_visibility::public_live, 2, 5));
    cells.pos_set(5, 1); cells.seq_add(5, 5);
    cells.rerot_set(5, fp_tag(2, 1, 22, llama_rerot_visibility::private_control, 0, 10));

    // runs: 21 = foreign public (FULL: visible regardless of ownership),
    // 22 = reader-private query run.
    const llama_rerot_reader_state reader = fp_reader(2, 1, 22, 10, {21, 22});

    const fp_q_expectation exp5 = fp_run_oracle(reader, 1, fp_keys_for_seq(cells, 5));
    CHECK(exp5.phys_eff.size() == 5); // base {0,1} + FULL {3,4} + query cell {5}
    CHECK(fp_contains_phys(exp5, 5) && !fp_contains_phys(exp5, 2));
    CHECK(exp5.query_virtual == 4); // own query key found

    const fp_q_expectation exp6 = fp_run_oracle(reader, 2, fp_keys_for_seq(cells, 6));
    CHECK(exp6.phys_eff.size() == 4); // base {1,2} + FULL {3,4}, query cell unowned
    CHECK(!fp_contains_phys(exp6, 5) && fp_contains_phys(exp6, 2) && !fp_contains_phys(exp6, 0));
    CHECK(exp6.query_virtual == 4); // no matching key: next-after-visible

    llama_ubatch ub{};
    ub.n_tokens = 2;
    llama_pos pos_arr[2] = {1, 2};
    int32_t nseq_arr[2] = {1, 1};
    llama_seq_id s0[1] = {5};
    llama_seq_id s1[1] = {6};
    llama_seq_id * seq_arr[2] = {s0, s1};
    ub.pos = pos_arr;
    ub.n_seq_id = nseq_arr;
    ub.seq_id = seq_arr;
    std::vector<llama_rerot_reader_state> views(8);
    views[5] = reader;
    views[6] = reader;
    llama_flashprefill_layout_params params;
    params.want_exact_rows = true;
    llama_flashprefill_layout built;
    std::string error;
    CHECK(llama_flashprefill_fixture_build_rerot(cells, views, ub, params, built, &error));
    CHECK(fp_layout_matches_oracle(built, 0, exp5, &error));
    CHECK(fp_layout_matches_oracle(built, 1, exp6, &error));
    if (!error.empty()) {
        std::fprintf(stderr, "fixture mismatch: %s\n", error.c_str());
    }
}

// ---- Real-context integration helpers (chains B/C/F; --model mode only) ----

static llama_flashprefill_row fp_prefill_row(int32_t seq, int32_t pos, int32_t begin, int32_t end) {
    llama_flashprefill_row r{};
    r.version       = LLAMA_FLASHPREFILL_ROW_VERSION;
    r.struct_size   = (uint32_t) sizeof(r);
    r.role          = LLAMA_FLASHPREFILL_ROLE_PREFILL;
    r.seq_id        = seq;
    r.reader_id     = LLAMA_FLASHPREFILL_READER_NONE;
    r.logical_pos   = pos;
    r.prefill_begin = begin;
    r.prefill_end   = end;
    r.prefill_known = true;
    return r;
}

static llama_flashprefill_row fp_decode_row(int32_t seq, int32_t pos) {
    llama_flashprefill_row r{};
    r.version       = LLAMA_FLASHPREFILL_ROW_VERSION;
    r.struct_size   = (uint32_t) sizeof(r);
    r.role          = LLAMA_FLASHPREFILL_ROLE_DECODE;
    r.seq_id        = seq;
    r.reader_id     = LLAMA_FLASHPREFILL_READER_NONE;
    r.logical_pos   = pos;
    r.prefill_begin = LLAMA_FLASHPREFILL_SEQ_UNKNOWN;
    r.prefill_end   = LLAMA_FLASHPREFILL_SEQ_UNKNOWN;
    r.prefill_known = false;
    return r;
}

// One single-seq batch plus its borrowed exec view (n_rows == n_tokens).
// Last row carries logits when the caller samples next.
struct fp_seq_batch {
    llama_batch batch;
    std::vector<llama_flashprefill_row> rows;
    llama_flashprefill_exec exec{};
};

static fp_seq_batch fp_make_batch(
        const std::vector<llama_token> & toks, llama_seq_id seq,
        llama_pos pos0, bool prefill, int32_t interval_end) {
    fp_seq_batch b;
    b.batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        const bool last = (i + 1 == toks.size());
        common_batch_add(b.batch, toks[i], pos0 + (llama_pos) i, {seq}, last);
        b.rows.push_back(prefill
            ? fp_prefill_row(seq, (int32_t) (pos0 + (llama_pos) i), 0, interval_end)
            : fp_decode_row(seq, (int32_t) (pos0 + (llama_pos) i)));
    }
    b.exec.version     = LLAMA_FLASHPREFILL_EXEC_VERSION;
    b.exec.struct_size = (uint32_t) sizeof(b.exec);
    b.exec.n_rows      = (uint32_t) b.rows.size();
    b.exec.reserved0   = 0;
    b.exec.rows        = b.rows.data();
    return b;
}

static void fp_free_batch(fp_seq_batch & b) {
    llama_batch_free(b.batch);
    b.rows.clear();
}

static int fp_decode_exec(llama_context * ctx, fp_seq_batch & b) {
    return llama_decode_with_flashprefill(ctx, b.batch, &b.exec);
}

// Last-row logits snapshot (copied synchronously: the next decode on any
// seq overwrites the row). Caller decodes, snapshots, decodes the peer,
// snapshots, then compares.
static std::vector<float> fp_last_logits(llama_context * ctx) {
    std::vector<float> out;
    const llama_model * model = llama_get_model(ctx);
    const int n = model == nullptr ? 0 : llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * l = n <= 0 ? nullptr : llama_get_logits_ith(ctx, -1);
    CHECK(n > 0 && l != nullptr);
    if (n <= 0 || l == nullptr) {
        return out;
    }
    out.assign(l, l + n);
    return out;
}

// Same restored state executed as the identical operator sequence on
// identical bytes on one device must reproduce bit-near-identical logits:
// 1e-5 abs covers backend accumulation variance (reduction order, precise
// vs fast math); any state divergence (wrong restore/rollback/stale pool)
// moves logits by orders more. Finiteness is required separately: NaN
// poison (e.g., from a stale pool) must hard-fail even if argmax coincides.
static void fp_assert_logits_match(const std::vector<float> & a, const std::vector<float> & b) {
    CHECK(a.size() == b.size() && !a.empty());
    if (a.size() != b.size() || a.empty()) {
        return;
    }
    float maxdiff = 0.0f;
    size_t amax = 0, bmax = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return;
        }
        const float d = std::fabs(a[i] - b[i]);
        if (d > maxdiff) {
            maxdiff = d;
        }
        if (a[i] > a[amax]) {
            amax = i;
        }
        if (b[i] > b[bmax]) {
            bmax = i;
        }
    }
    CHECK(maxdiff <= 1e-5f);
    CHECK(amax == bmax);
}

// Restored-suffix sparse-path evidence from the context metrics ledger
// (read-only snapshot; the server owns the drain watermark — never
// consumed here). All five must grow across the suffix prefills: eligible
// (new-path presentation) plus plan-confirmed sparse/selected/corrected/
// visible GPU work. This is an explicit model-integration COVERAGE GATE
// (labeled as such): it asserts the run actually exercised sparse work.
// Tripping it means insufficient coverage on this model/input (tied
// energies kept all, or no docked plans — see the printed NO_PLAN
// evidence), never by itself an algorithm failure. Proof is ledger-actual
// only — no routing-function pin stands in for backend execution (backend
// capability itself stays provisional-true in tree;
// ContextIntegration/GraphIntegration own the real query).
static void fp_assert_sparse_suffix_grew(
        const llama_flashprefill_metrics::accum & before,
        const llama_flashprefill_metrics::accum & after) {
    namespace fpmet = llama_flashprefill_metrics;
    std::fprintf(stderr,
        "integration metrics delta: eligible %llu->%llu sparse %llu->%llu selected %llu->%llu corrected %llu->%llu visible %llu->%llu noplan %llu->%llu\n",
        (unsigned long long) before.eligible_rows, (unsigned long long) after.eligible_rows,
        (unsigned long long) before.sparse_rows, (unsigned long long) after.sparse_rows,
        (unsigned long long) before.selected_blocks, (unsigned long long) after.selected_blocks,
        (unsigned long long) before.corrected_blocks, (unsigned long long) after.corrected_blocks,
        (unsigned long long) before.visible_tokens, (unsigned long long) after.visible_tokens,
        (unsigned long long) before.dense_rows[fpmet::DENSE_BUCKET_NO_PLAN],
        (unsigned long long) after.dense_rows[fpmet::DENSE_BUCKET_NO_PLAN]);
    CHECK(after.eligible_rows > before.eligible_rows);
    CHECK(after.sparse_rows > before.sparse_rows);
    CHECK(after.selected_blocks > before.selected_blocks);
    CHECK(after.corrected_blocks > before.corrected_blocks);
    CHECK(after.visible_tokens > before.visible_tokens);
}

static void fp_clear_all_seqs(llama_memory_t mem) {
    for (llama_seq_id s = 0; s < 4; ++s) {
        CHECK(llama_memory_seq_rm(mem, s, -1, -1));
    }
}

// Chain B over a real context: seq-state checkpoint + real KV draft appends
// + partial (range) rollback. Invariant: rollback-to-accepted IDENTICAL to
// restore-checkpoint-then-replay-accepted (both compared after one shared
// next token via full finite logits vectors).
static void test_integration_seq_checkpoint_draft_rollback(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    fp_clear_all_seqs(mem);

    // 256-token prompt = four full BN64 historical fragments: multiple
    // candidate blocks for selection to discriminate. Whether skips occur
    // stays model/input-dependent (ties keep all) and is gated, not
    // assumed, by the metrics assertions below.
    const std::vector<llama_token> prompt(256, (llama_token) 11);
    auto b0 = fp_make_batch(prompt, 0, 0, true, 256);
    CHECK(fp_decode_exec(ctx, b0) == 0);
    const std::vector<float> l0 = fp_last_logits(ctx);
    fp_free_batch(b0);
    auto b1 = fp_make_batch(prompt, 1, 0, true, 256);
    CHECK(fp_decode_exec(ctx, b1) == 0);
    const std::vector<float> l1 = fp_last_logits(ctx);
    fp_free_batch(b1);
    fp_assert_logits_match(l0, l1); // identical histories, identical vectors

    std::vector<uint8_t> ckpt(llama_state_seq_get_size(ctx, 1));
    CHECK(!ckpt.empty());
    if (ckpt.empty()) {
        return;
    }
    CHECK(llama_state_seq_get_data(ctx, ckpt.data(), ckpt.size(), 1) == ckpt.size());

    // Temporary drafts: real KV appends beyond the checkpoint.
    const std::vector<llama_token> drafts = {21, 22, 23, 24, 25};
    auto bd = fp_make_batch(drafts, 1, 256, false, 0);
    CHECK(fp_decode_exec(ctx, bd) == 0);
    fp_free_batch(bd);

    // Partial rollback: accept 256,257 only, drop the rejected tail.
    CHECK(llama_memory_seq_rm(mem, 1, 258, -1));

    // Restore the checkpoint elsewhere and replay only the accepted prefix.
    // Same tokens/positions/interval/chunking as the accepted leg, so plan
    // selection cannot diverge between the two histories.
    CHECK(llama_state_seq_set_data(ctx, ckpt.data(), ckpt.size(), 2) == ckpt.size());
    const std::vector<llama_token> accepted = {21, 22};
    auto ba = fp_make_batch(accepted, 2, 256, false, 0);
    CHECK(fp_decode_exec(ctx, ba) == 0);
    fp_free_batch(ba);

    // Restored suffix as real prefill rows (legal post-restore role): both
    // seqs present 32 sparse-eligible rows. Metrics ledger around them.
    std::vector<llama_token> suffix;
    for (llama_token t = 61; t < 93; ++t) {
        suffix.push_back(t);
    }
    const auto m_before = ctx->flashprefill_metrics_snapshot();
    auto bs1 = fp_make_batch(suffix, 1, 258, true, 290);
    CHECK(fp_decode_exec(ctx, bs1) == 0);
    fp_free_batch(bs1);
    auto bs2 = fp_make_batch(suffix, 2, 258, true, 290);
    CHECK(fp_decode_exec(ctx, bs2) == 0);
    fp_free_batch(bs2);
    const auto m_after = ctx->flashprefill_metrics_snapshot();
    fp_assert_sparse_suffix_grew(m_before, m_after);

    // One shared next token on both accepted+suffixed histories.
    const std::vector<llama_token> next = {30};
    auto bn1 = fp_make_batch(next, 1, 290, false, 0);
    CHECK(fp_decode_exec(ctx, bn1) == 0);
    const std::vector<float> r1 = fp_last_logits(ctx);
    fp_free_batch(bn1);
    auto bn2 = fp_make_batch(next, 2, 290, false, 0);
    CHECK(fp_decode_exec(ctx, bn2) == 0);
    const std::vector<float> r2 = fp_last_logits(ctx);
    fp_free_batch(bn2);
    fp_assert_logits_match(r1, r2);
}

// Chain C over a real context: save, erase, fragment with other seqs,
// restore onto remapped (possibly non-contiguous) cells, re-prefill the
// suffix through the sparse path, compare full logits vs the untouched
// control after one shared next token.
static void test_integration_save_erase_fragment_restore(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    fp_clear_all_seqs(mem);

    const std::vector<llama_token> prompt(256, (llama_token) 11);
    auto b0 = fp_make_batch(prompt, 0, 0, true, 256);
    CHECK(fp_decode_exec(ctx, b0) == 0);
    fp_free_batch(b0);
    auto b1 = fp_make_batch(prompt, 1, 0, true, 256);
    CHECK(fp_decode_exec(ctx, b1) == 0);
    fp_free_batch(b1);

    std::vector<uint8_t> saved(llama_state_seq_get_size(ctx, 1));
    CHECK(!saved.empty());
    if (saved.empty()) {
        return;
    }
    CHECK(llama_state_seq_get_data(ctx, saved.data(), saved.size(), 1) == saved.size());
    CHECK(llama_memory_seq_rm(mem, 1, -1, -1)); // erase: leaves a hole

    // Fragment the cache with other seqs before restoring.
    auto f2 = fp_make_batch(std::vector<llama_token>(40, (llama_token) 9), 2, 0, true, 40);
    CHECK(fp_decode_exec(ctx, f2) == 0);
    fp_free_batch(f2);
    auto f3 = fp_make_batch(std::vector<llama_token>(40, (llama_token) 8), 3, 0, true, 40);
    CHECK(fp_decode_exec(ctx, f3) == 0);
    fp_free_batch(f3);

    CHECK(llama_state_seq_set_data(ctx, saved.data(), saved.size(), 1) == saved.size());

    std::vector<llama_token> suffix;
    for (llama_token t = 61; t < 93; ++t) {
        suffix.push_back(t);
    }
    const auto m_before = ctx->flashprefill_metrics_snapshot();
    auto bs0 = fp_make_batch(suffix, 0, 256, true, 288);
    CHECK(fp_decode_exec(ctx, bs0) == 0);
    fp_free_batch(bs0);
    auto bs1 = fp_make_batch(suffix, 1, 256, true, 288);
    CHECK(fp_decode_exec(ctx, bs1) == 0);
    fp_free_batch(bs1);
    const auto m_after = ctx->flashprefill_metrics_snapshot();
    fp_assert_sparse_suffix_grew(m_before, m_after);

    const std::vector<llama_token> next = {30};
    auto bn0 = fp_make_batch(next, 0, 288, false, 0);
    CHECK(fp_decode_exec(ctx, bn0) == 0);
    const std::vector<float> r0 = fp_last_logits(ctx);
    fp_free_batch(bn0);
    auto bn1 = fp_make_batch(next, 1, 288, false, 0);
    CHECK(fp_decode_exec(ctx, bn1) == 0);
    const std::vector<float> r1 = fp_last_logits(ctx);
    fp_free_batch(bn1);
    fp_assert_logits_match(r0, r1);
}

// Chain F over a real context: constrained slots (n_ctx=1024). Two
// near-capacity seqs, then an oversized append must fail nonzero (shortage,
// no crash); freeing one seq lets a smaller retry complete.
static void test_integration_constrained_retry(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    fp_clear_all_seqs(mem);

    auto a0 = fp_make_batch(std::vector<llama_token>(440, (llama_token) 11), 0, 0, true, 440);
    CHECK(fp_decode_exec(ctx, a0) == 0);
    fp_free_batch(a0);
    auto a1 = fp_make_batch(std::vector<llama_token>(440, (llama_token) 11), 1, 0, true, 440);
    CHECK(fp_decode_exec(ctx, a1) == 0);
    fp_free_batch(a1);

    // 880/1024 cells used: a further 440-token append cannot fit.
    auto big = fp_make_batch(std::vector<llama_token>(440, (llama_token) 11), 2, 0, true, 440);
    CHECK(fp_decode_exec(ctx, big) != 0);
    fp_free_batch(big);

    CHECK(llama_memory_seq_rm(mem, 1, -1, -1));
    auto small = fp_make_batch(std::vector<llama_token>(200, (llama_token) 11), 2, 0, true, 200);
    CHECK(fp_decode_exec(ctx, small) == 0);
    fp_free_batch(small);
}

// Developmental policy shared by the B/C/F legs (fixed fixture values).
static llama_flashprefill_config fp_policy_developmental() {
    llama_flashprefill_config c = llama_flashprefill_default_config();
    c.mode             = LLAMA_FLASHPREFILL_MODE_REQUIRED;
    c.min_kv           = 0;
    c.dense_tail_tiles = 0;
    c.sink_blocks      = 0;
    c.window_blocks    = 0;
    c.block_k          = 64;
    c.alpha            = 1.0f; // EXPERIMENTAL: request max pressure (ties still keep all)
    c.mean_correction  = true;
    c.exact_all        = false;
    return c;
}

// Audit R1 policy: identical except the resident gate under test.
static llama_flashprefill_config fp_policy_audit_r1() {
    llama_flashprefill_config c = fp_policy_developmental();
    c.min_kv = 192;
    return c;
}

// Audit R2 policy: fully explicit, not inherited field-by-field — min_kv=0
// and dense_tail_tiles=0 with enough BN64 history (192 = three blocks) so
// these rows would be FP-eligible but for full_attn_layers=MAX. (A base
// cloned at default min_kv=1024 would only ever test SHORT, never the
// orphan/full-prefix path.) Experimental params recorded as in development.
static llama_flashprefill_config fp_policy_audit_r2() {
    llama_flashprefill_config c = llama_flashprefill_default_config();
    c.mode                 = LLAMA_FLASHPREFILL_MODE_REQUIRED;
    c.min_kv               = 0;
    c.dense_tail_tiles     = 0;
    c.sink_blocks          = 0;
    c.window_blocks        = 0;
    c.block_k              = 64;
    c.alpha                = 1.0f; // EXPERIMENTAL: request max pressure (ties still keep all)
    c.full_attn_layers     = UINT32_MAX;
    c.mean_correction      = true;
    c.exact_all            = false;
    return c;
}

// Init one immutable-policy context for an audit case. Null (with SKIP
// note) when the required scenario cannot run. The holder frees model +
// context on destruction: cases are scoped so VRAM never holds two.
static common_init_result_ptr fp_init_audit_ctx(common_params p, const char * tag) {
    if (llama_flashprefill_validate_config(&p.flashprefill) != LLAMA_FLASHPREFILL_OK) {
        std::fprintf(stderr, "SKIP: %s policy invalid (required scenario misconfigured)\n", tag);
        return nullptr;
    }
    common_init_result_ptr h = common_init_from_params(p);
    if (!h || !h->model() || !h->context()) {
        std::fprintf(stderr, "SKIP: %s model init failed (required runtime scenario absent)\n", tag);
        return nullptr;
    }
    return h;
}

// REQUIRED-mode decodes must never throw on legitimate dense legs (short
// context, full-prefix deny, ...): a false required error would escape as
// an exception, not a return code. Guarded decode converts that into a
// CHECK failure.
static int fp_decode_guarded(llama_context * ctx, fp_seq_batch & b, const char * what) {
    try {
        return fp_decode_exec(ctx, b);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "CHECK failed: %s threw: %s\n", what, e.what());
        ++failures;
    } catch (...) {
        std::fprintf(stderr, "CHECK failed: %s threw (unknown exception)\n", what);
        ++failures;
    }
    return -999;
}

// Chunked prefill decode for long seeds: never hand llama_decode more than
// the context batch limits in one call. Same tokens/positions/interval as a
// single batch would carry; chunking is a transport detail, not content.
static void fp_decode_prefill_chunked(
        llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & toks,
        llama_pos pos0, int32_t interval_end, const char * what) {
    uint32_t lim = llama_n_batch(ctx);
    const uint32_t lim_u = llama_n_ubatch(ctx);
    if (lim_u < lim) {
        lim = lim_u;
    }
    if (lim < 1) {
        lim = 1;
    }
    for (size_t off = 0; off < toks.size(); ) {
        size_t n = toks.size() - off;
        if (n > lim) {
            n = lim;
        }
        std::vector<llama_token> part(toks.begin() + (ptrdiff_t) off, toks.begin() + (ptrdiff_t) (off + n));
        auto b = fp_make_batch(part, seq, pos0 + (llama_pos) off, true, interval_end);
        CHECK(fp_decode_guarded(ctx, b, what) == 0);
        fp_free_batch(b);
        off += n;
    }
}

// Audit R1 (Graph): the padded-physical n_kv must not decide the dense to
// sparse switch. totalKV 1024 with idle seq1 x768 resident: padded n_kv is a
// constant 1024 across every active chunk below (768+256=1024 at the end),
// while active seq0 grows 64 -> 128 -> 192 -> 256 past immutable min_kv=192.
// Chunks 1-2 stay dense (legitimate short context, REQUIRED must not falsely
// error). Chunk 3 is the transition: only its last query reaches 192 legal
// causal tokens, and dense-forced earlier rows may keep the whole packed
// tile exact — mixed or all-exact both allowed, no SHORT-silence guarantee.
// Chunk 4 is fully past the gate and must show actual FP plan/sparse/
// visible work; a no-skip outcome trips as insufficient coverage (labeled),
// never as an algorithm failure. Ledger counters only — never a scalar
// route mirror.
static void test_integration_padded_graph_switch(llama_context * ctx) {
    namespace fpmet = llama_flashprefill_metrics;
    llama_memory_t mem = llama_get_memory(ctx);
    fp_clear_all_seqs(mem);

    // Idle seed in batch-limit chunks (768 must not exceed n_batch/ubatch).
    fp_decode_prefill_chunked(ctx, 1, std::vector<llama_token>(768, (llama_token) 7), 0, 768, "r1 idle seed");

    auto m = ctx->flashprefill_metrics_snapshot();
    const llama_token chunk_tok[4] = {11, 12, 13, 14};
    for (int c = 0; c < 4; ++c) {
        auto ch = fp_make_batch(
            std::vector<llama_token>(64, chunk_tok[c]), 0, (llama_pos) (64 * c), true, 256);
        CHECK(fp_decode_guarded(ctx, ch, "r1 active chunk") == 0);
        fp_free_batch(ch);
        const auto m_next = ctx->flashprefill_metrics_snapshot();
        CHECK(m_next.eligible_rows > m.eligible_rows);
        if (c < 2) {
            // Resident 64/128 < min_kv 192: dense short context, sparse silent.
            CHECK(m_next.sparse_rows == m.sparse_rows);
            CHECK(m_next.dense_rows[fpmet::DENSE_BUCKET_SHORT] > m.dense_rows[fpmet::DENSE_BUCKET_SHORT]);
        } else if (c == 2) {
            // Transition: only the last query (pos 191) reaches 192 legal
            // causal tokens. Mixed or all-exact allowed here.
        } else {
            // Fully past the gate: actual FP plan/sparse/visible work.
            const bool grew = m_next.sparse_rows > m.sparse_rows
                && m_next.visible_tokens > m.visible_tokens
                && m_next.selected_blocks > m.selected_blocks
                && m_next.corrected_blocks > m.corrected_blocks;
            if (!grew) {
                std::fprintf(stderr, "COVERAGE: r1 fourth chunk produced no sparse work on this model/input (ties keep all) — insufficient coverage, not an algorithm failure\n");
            }
            CHECK(m_next.sparse_rows > m.sparse_rows);
            CHECK(m_next.visible_tokens > m.visible_tokens);
            CHECK(m_next.selected_blocks > m.selected_blocks);
            CHECK(m_next.corrected_blocks > m.corrected_blocks);
        }
        m = m_next;
    }
}

// Audit R2 (Met): full_attn_layers=UINT32_MAX keeps every eligible full
// layer on the old dense path by design (FULL_PREFIX early deny builds no
// companion metadata because no layer could consume it). REQUIRED mode must
// still finish with finite, deterministic logits — and the ledger must show
// no orphan plan work and no false NO_PLAN failure.
static void test_integration_full_prefix_dense(llama_context * ctx) {
    namespace fpmet = llama_flashprefill_metrics;
    llama_memory_t mem = llama_get_memory(ctx);
    fp_clear_all_seqs(mem);

    const std::vector<llama_token> prompt(192, (llama_token) 11);
    const auto m0 = ctx->flashprefill_metrics_snapshot();
    auto b0 = fp_make_batch(prompt, 0, 0, true, 192);
    CHECK(fp_decode_guarded(ctx, b0, "r2 prefill seq0") == 0);
    const std::vector<float> l0 = fp_last_logits(ctx);
    fp_free_batch(b0);
    auto b1 = fp_make_batch(prompt, 1, 0, true, 192);
    CHECK(fp_decode_guarded(ctx, b1, "r2 prefill seq1") == 0);
    const std::vector<float> l1 = fp_last_logits(ctx);
    fp_free_batch(b1);
    const auto m1 = ctx->flashprefill_metrics_snapshot();

    fp_assert_logits_match(l0, l1); // old dense path finishes finite + deterministic
    CHECK(m1.dense_rows[fpmet::DENSE_BUCKET_NO_PLAN] == m0.dense_rows[fpmet::DENSE_BUCKET_NO_PLAN]);
    uint64_t inv0 = 0, inv1 = 0, pool0 = 0, pool1 = 0;
    for (uint32_t i = 0; i < fpmet::PLAN_REASON_COUNT; ++i) {
        inv0 += m0.plan_invalidations[i];
        inv1 += m1.plan_invalidations[i];
    }
    for (uint32_t i = 0; i < fpmet::POOL_REASON_COUNT; ++i) {
        pool0 += m0.pool_rebuild[i];
        pool1 += m1.pool_rebuild[i];
    }
    CHECK(inv1 == inv0);    // no orphan plan work
    CHECK(pool1 == pool0);  // no pool built on the all-dense path
    CHECK(m1.sparse_rows == m0.sparse_rows); // nothing executed sparse
    CHECK(m1.dense_rows[fpmet::DENSE_BUCKET_FULL_PREFIX] > m0.dense_rows[fpmet::DENSE_BUCKET_FULL_PREFIX]);
}

// Real-context integration driver. Returns failures to add (scenario CHECKs
// add to the global count directly): missing model/init/required-backend
// SKIP nonzero; merely-preferred missing backend SKIP zero.
static int fp_run_integration(
        const std::vector<char *> & fwd_args, const std::string & model_path, bool have_model,
        const std::string & backend_name, const std::string & required_backend) {
    if (!have_model) {
        std::fprintf(stderr, "SKIP: integration flags without --model; no model run performed (required scenario absent)\n");
        return 1;
    }
    FILE * f = std::fopen(model_path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "SKIP: model file unreadable: %s (required runtime scenario absent)\n", model_path.c_str());
        return 1;
    }
    std::fclose(f);

    ggml_backend_load_all();
    const std::string want_backend = !required_backend.empty() ? required_backend : backend_name;
    ggml_backend_dev_t dev = nullptr;
    bool cpu_forced = false;
    if (!want_backend.empty()) {
        if (want_backend == "cpu" || want_backend == "none") {
            cpu_forced = true;
        } else {
            dev = ggml_backend_dev_by_name(want_backend.c_str());
            if (dev == nullptr) {
                std::fprintf(stderr, "SKIP: backend '%s' unavailable%s\n",
                    want_backend.c_str(), required_backend.empty() ? "" : " (required)");
                return required_backend.empty() ? 0 : 1;
            }
        }
    }

    common_params params;
    std::vector<char *> pargs = fwd_args;
    if (!common_params_parse((int) pargs.size(), pargs.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    // Shared base for every case below; n_ctx and the immutable policy are
    // set per case (policies must differ, so contexts cannot be shared).
    params.n_parallel   = 4;
    params.kv_unified   = true;
    params.sampling.seed = 42;
    if (cpu_forced) {
        params.devices = {nullptr};
    } else if (dev != nullptr) {
        params.devices = {dev, nullptr};
    }

    common_init();

    // B/C/F development context (own immutable policy; the holder is
    // destroyed at block end so later cases never share VRAM with it).
    {
        common_params p = params;
        p.n_ctx = 1024;
        // Fixed developmental policy for the sparse-path legs (always
        // applied and logged: integration is a deterministic fixture, not
        // a policy sweep). Defaults would route everything dense.
        p.flashprefill = fp_policy_developmental();
        std::fprintf(stderr,
            "integration: developmental policy REQUIRED min_kv=0 tail=0 sink=0 window=0 block_k=64 alpha=1.0(EXPERIMENTAL max-pressure; ties keep all) n_ctx=1024 n_parallel=4\n");
        common_init_result_ptr h = fp_init_audit_ctx(p, "bcf");
        if (!h) {
            return 1;
        }
        llama_context * ctx = h->context();
        std::fprintf(stderr, "integration: policy fingerprint 0x%016llx\n",
            (unsigned long long) llama_flashprefill_policy_fingerprint(ctx));

        test_integration_seq_checkpoint_draft_rollback(ctx);
        test_integration_save_erase_fragment_restore(ctx);
        test_integration_constrained_retry(ctx);
    }

    // Audit R1 (Graph): padded-n_kv graph switch on its own immutable
    // context; freed before R2 so VRAM never doubles.
    {
        common_params p = params;
        p.n_ctx = 1024;
        p.flashprefill = fp_policy_audit_r1();
        common_init_result_ptr h = fp_init_audit_ctx(p, "r1");
        if (!h) {
            return 1;
        }
        test_integration_padded_graph_switch(h->context());
    }

    // Audit R2 (Met): full-prefix dense on its own immutable context.
    {
        common_params p = params;
        p.n_ctx = 512;
        p.flashprefill = fp_policy_audit_r2();
        common_init_result_ptr h = fp_init_audit_ctx(p, "r2");
        if (!h) {
            return 1;
        }
        test_integration_full_prefix_dense(h->context());
    }
    return 0;
}

// ---- Ordinary-planner foreign-isolation fixture (Graph final audit) ----
//
// Production bug under fix: the ordinary planner drew membership signatures
// from live[] = ALL resident seqs, so foreign idle/image/fragmentation cells
// perturbed A's descriptor/admission though no use ever consumes them —
// B-only image-pattern cells forced whole-build INELIGIBLE and B-only
// fragments inflated F. Fixed behavior: the candidate pool is restricted to
// active query primary-seq membership before pool, so with A legal cells
// held fixed, altering/adding B-only cells (including extreme ext/image
// patterns) leaves A fragments/cell_refs/uses/groups identical (physical
// map literal — no pack runs here), foreign refs force neither unsupported
// nor F inflation, and shared A+B refs appear once.
//
// New-side entry: llama_flashprefill_fixture_build_ordinary (fixture header;
// fixed causal 1-D text policy, want_exact_rows honored). No hand-rolled
// cell-enumeration proof: the built layouts themselves are compared.
static bool fp_fragment_equal(const llama_flashprefill_fragment & a, const llama_flashprefill_fragment & b) {
    return a.domain == b.domain && a.stream == b.stream && a.boundary_partial == b.boundary_partial &&
           a.contiguous == b.contiguous && a.cell_begin == b.cell_begin &&
           a.cell_ref_offset == b.cell_ref_offset && a.token_count == b.token_count &&
           a.logical_block == b.logical_block && a.logical_begin == b.logical_begin &&
           a.logical_end == b.logical_end && a.members == b.members && a.episode_id == b.episode_id &&
           a.run_id == b.run_id && a.visibility == b.visibility && a.run_rank == b.run_rank &&
           a.virtual_pos0 == b.virtual_pos0 && a.phase_bias == b.phase_bias && a.gated == b.gated;
}

// Full built-payload equality for one query view: fragments, backing refs,
// phase groups, compact uses, and oracle exact rows. Freshness stamps and
// eligibility bookkeeping are owner-build metadata, not payload.
static bool fp_ord_payload_equal(
        const llama_flashprefill_layout & a, const llama_flashprefill_layout & b, std::string * error) {
    const auto fail = [&](const std::string & msg) {
        if (error) { *error = msg; }
        return false;
    };
    if (a.queries.size() != b.queries.size()) {
        return fail("ordinary fixture: query count mismatch");
    }
    for (size_t i = 0; i < a.queries.size(); ++i) {
        const auto & x = a.queries[i];
        const auto & y = b.queries[i];
        if (x.query_index != y.query_index || x.seq_id != y.seq_id || x.stream != y.stream ||
            x.query_pos != y.query_pos || x.query_virtual_pos != y.query_virtual_pos) {
            return fail("ordinary fixture: query row mismatch");
        }
    }
    if (a.fragments.size() != b.fragments.size()) {
        return fail("ordinary fixture: fragment count (F) mismatch — foreign refs inflated F");
    }
    for (size_t i = 0; i < a.fragments.size(); ++i) {
        if (!fp_fragment_equal(a.fragments[i], b.fragments[i])) {
            return fail("ordinary fixture: fragment descriptor mismatch");
        }
    }
    if (a.cell_refs != b.cell_refs) {
        return fail("ordinary fixture: cell_refs mismatch");
    }
    if (a.groups.size() != b.groups.size()) {
        return fail("ordinary fixture: group count mismatch");
    }
    for (size_t i = 0; i < a.groups.size(); ++i) {
        if (a.groups[i].query_index != b.groups[i].query_index ||
            a.groups[i].effective_pos != b.groups[i].effective_pos) {
            return fail("ordinary fixture: group mismatch");
        }
    }
    if (a.group_offsets != b.group_offsets || a.uses.size() != b.uses.size() ||
        a.use_offsets != b.use_offsets) {
        return fail("ordinary fixture: uses mismatch");
    }
    for (size_t i = 0; i < a.uses.size(); ++i) {
        const auto & x = a.uses[i];
        const auto & y = b.uses[i];
        if (x.query != y.query || x.fragment != y.fragment || x.group != y.group ||
            x.sub_off != y.sub_off || x.sub_count != y.sub_count || x.flags != y.flags) {
            return fail("ordinary fixture: use record mismatch");
        }
    }
    if (a.exact_rows != b.exact_rows || a.exact_groups != b.exact_groups ||
        a.exact_flags != b.exact_flags || a.exact_offsets != b.exact_offsets) {
        return fail("ordinary fixture: oracle exact rows mismatch");
    }
    return true;
}

// Count occurrences of one physical cell across every fragment's member
// range (contiguous range or ref span).
static uint32_t fp_count_phys_in_fragments(const llama_flashprefill_layout & l, uint32_t phys) {
    uint32_t n = 0;
    for (const auto & f : l.fragments) {
        if (f.contiguous) {
            if (phys >= f.cell_begin && phys < f.cell_begin + f.token_count) {
                ++n;
            }
        } else {
            for (uint32_t k = 0; k < f.token_count; ++k) {
                if (f.cell_ref_offset + k < l.cell_refs.size() && l.cell_refs[f.cell_ref_offset + k] == phys) {
                    ++n;
                }
            }
        }
    }
    return n;
}

static void test_ordinary_foreign_isolation() {
    // A (seq 5): text cells pos 0..127 with a resident hole at 40,41
    // (exercises the cell_refs path); pos 10,11 shared with B (seq 6).
    // B-only: far-region cells pos 1000..1063 plus an extreme image-pattern
    // cell (ext.y far beyond pos) that must neither poison the build nor
    // inflate F once the pool is restricted to active query membership.
    llama_kv_cells base;
    base.resize(130);
    base.set_generation_enabled(true);
    for (uint32_t i = 0; i < 128; ++i) {
        if (i == 40 || i == 41) {
            continue; // resident hole: fragmentation, not a member
        }
        base.pos_set(i, (llama_pos) i);
        base.seq_add(i, 5);
    }
    base.seq_add(10, 6);
    base.seq_add(11, 6);

    llama_kv_cells with_foreign = base; // copy keeps A physical indices
    with_foreign.resize(200);
    for (uint32_t k = 0; k < 64; ++k) {
        const uint32_t i = 130 + k;
        with_foreign.pos_set(i, (llama_pos) (1000 + k));
        with_foreign.seq_add(i, 6);
    }
    with_foreign.pos_set(194, 2000);
    with_foreign.seq_add(194, 6);
    with_foreign.ext_set(194, {(llama_pos) 0, (llama_pos) 3000}); // image pattern: ext.y >> pos

    // One stream; queries ask seq 5 only. B (seq 6) is never queried.
    std::vector<uint32_t> seq_to_stream(8, 0);
    llama_ubatch ub{};
    ub.n_tokens = 2;
    llama_pos pos_arr[2] = {120, 121};
    int32_t nseq_arr[2] = {1, 1};
    llama_seq_id s0[1] = {5};
    llama_seq_id s1[1] = {5};
    llama_seq_id * seq_arr[2] = {s0, s1};
    ub.pos = pos_arr;
    ub.n_seq_id = nseq_arr;
    ub.seq_id = seq_arr;
    llama_kv_cells_vec v0{base};
    llama_kv_cells_vec v1{with_foreign};

    llama_flashprefill_layout_params params;
    params.block_k = 64;
    params.want_exact_rows = true;
    llama_flashprefill_layout l0;
    llama_flashprefill_layout l1;
    std::string error;
    CHECK(llama_flashprefill_fixture_build_ordinary(v0, seq_to_stream, ub, params, l0, &error));
    CHECK(llama_flashprefill_fixture_build_ordinary(v1, seq_to_stream, ub, params, l1, &error));
    if (!error.empty()) {
        std::fprintf(stderr, "ordinary fixture build error: %s\n", error.c_str());
    }
    // Foreign B-only cells (including the image-pattern one) change nothing
    // observable for A: same fragments (no F inflation), refs, uses, groups.
    // A foreign-forced INELIGIBLE fails here instead of passing silently.
    CHECK(fp_ord_payload_equal(l0, l1, &error));
    if (!error.empty()) {
        std::fprintf(stderr, "ordinary fixture mismatch: %s\n", error.c_str());
    }
    // Shared A+B refs survive exactly once (dedup, not dropped, not doubled).
    CHECK(fp_count_phys_in_fragments(l1, 10) == 1);
    CHECK(fp_count_phys_in_fragments(l1, 11) == 1);
}

} // namespace

static bool fp_take_value(int & i, int argc, char ** argv, const char * flag, std::string & out) {
    const std::string a = argv[i];
    if (a == flag && i + 1 < argc) {
        out = argv[++i];
        return true;
    }
    const std::string pref = std::string(flag) + "=";
    if (a.compare(0, pref.size(), pref) == 0) {
        out = a.substr(pref.size());
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    std::puts("=== FlashPrefill state/membership/policy test ===");
    test_shared_ref_remove_keeps_peer();
    test_foreign_poison_excluded();
    test_compact_remap_preserves();
    test_draft_rollback_excludes_rejected();
    test_clear_reuse();
    test_save_restore_remapped();
    test_generation_view_invalidation();
    test_overflow_capacity_rejects();
    test_policy_identity_separates();
    test_rerot_checkpoint_accept_reject();
    test_context_shift_runtime_api();
    test_pen_suspend_resume();
    test_rerot_virtualization_future_gated();
    test_rerot_virtualization_shared_view();
    test_ordinary_foreign_isolation();
    failures += flashprefill_state_envelope_tests::run_tests();

    // Optional real-context integration: consumed flags never reach common.
    std::string backend_name, required_backend, model_path;
    bool have_model = false;
    std::vector<char *> fwd;
    fwd.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string v;
        if (fp_take_value(i, argc, argv, "--backend", v)) {
            backend_name = v;
        } else if (fp_take_value(i, argc, argv, "--required-backend", v)) {
            required_backend = v;
        } else {
            // --model/-m (either spelling) and everything else passes
            // through to common_params_parse; record a model request when
            // the pair carries a value.
            const std::string a = argv[i];
            if ((a == "--model" || a == "-m") && i + 1 < argc) {
                model_path = argv[i + 1];
                have_model = true;
            } else if (a.compare(0, 8, "--model=") == 0) {
                model_path = a.substr(8);
                have_model = true;
            }
            fwd.push_back(argv[i]);
        }
    }
    if (have_model || !backend_name.empty() || !required_backend.empty()) {
        failures += fp_run_integration(fwd, model_path, have_model, backend_name, required_backend);
    } else {
        std::puts("(no --model/--backend/--required-backend: model integration skipped, CPU-only cases above)");
    }

    std::printf("=== Results: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
