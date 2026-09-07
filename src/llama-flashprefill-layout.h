// llama-flashprefill-layout.h — owner-derived FlashPrefill legal-fragment planning types.
//
// Owner: CacheFragments. Short-lived planning output derived from llama_kv_cells
// (the sole cell owner) plus the cache-side RERoT tag/view state. This header
// defines NO new cell lifecycle: layouts hold physical indices + stamps only,
// never pointers into GPU memory, and are rebuilt per graph (conservative) or
// whenever the owner stamps change. There is deliberately NO pooled-mean
// mirror here: means/pools are derived GPU work, rebuilt per graph by later
// stages, never persisted across restore.
//
// Address spaces (never conflated):
//   physical cell index -> reads K/V bytes (exact_rows, fragment ranges/refs)
//   logical position    -> ordinary causal/SWA legality (fragments' logical span)
//   storage position    -> K writer RoPE phase (never re-phased; K stays put)
//   virtual position    -> RERoT reader-relative layout (fragment virtual span,
//                          query_virtual_pos, phase_bias = storage - virtual)
//
// Query phase groups use the SAME structure as the existing graph RoPE input
// (llm_graph_input_attn_rerot q_indices/q_pos feed): groups[g] carries the
// raw-Q RoPE position (effective_pos) for that group, and each exact row
// consumes exact_groups[e]. All groups of one query merge into ONE global
// online softmax downstream. K is never re-RoPE'd.
//
// Wire packing (frozen ggml/include/ggml-flashprefill.h v1, never included
// here; the packer is GraphIntegration, not this header):
//   fragments[i] -> wire fragment[8]: cell_off/cell_count from the contiguous
//     range or (cell_refs base + cell_ref_offset); logical_block from
//     logical_block (always regular BN identities, never INT32_MIN); domain
//     from wire_domain() below (wire leaves numbering open; confirmed values
//     pending WireReference, never guessed elsewhere); flags bit0 from
//     boundary_partial.
//   uses[i] -> wire use[8]: frag_id/query/group/sub_off/sub_count/flags pack
//     directly; tile_id/kv_head fan out downstream from the GQA packing
//     (block_q, Hq, Hkv); source_query derives from queries[use.query]
//     .query_index (packed-row domain is query_index * G + subhead).
//   queries[q] + groups -> wire row[8]: logical_pos is query_pos; prompt
//     bounds come from the execution descriptor (ContextIntegration), never
//     guessed here; flags bit0 DENSE_FORCE is select-stage policy downstream.
//   Q tensor F32 [Dk, n_groups, Hq, 1] with n_groups = groups.size()
//     (without RERoT that is one group per query); O tensor
//     [Dv, Hq, n_output_queries, 1]. Pool F32 [Dk+Dv, Hkv, Fcap] over
//     fragments. Reference math is natural-log (proxy_logit = s_dot + ln(n));
//     proxy multiplicity n_J = token_count. Production NEVER expands exact
//     token rows: the packer consumes uses (built O(Q*F)); exact_rows exist
//     only for the explicit oracle/debug path (want_exact_rows=true).
//
// Limits (fail-closed, explicit):
//   L1  ordinary prefill rows only. Text-pattern partial IMRoPE (Qwen35/Ornith
//       text: n_pos==4 rows carrying [p,p,p,0]) is 1D-equivalent and stays
//       eligible. Genuinely non-text 2-D/M-RoPE rows, image-pattern resident
//       cells under a 2-D ubatch (the stock mask's tiebreak would diverge),
//       ALiBi models, and non-causal+SWA are capability-ineligible, never
//       silently approximated (multimodal/embedding/rerank keep stock path).
//       The RERoT indexed path consumes the scalar slot-0 coordinate exactly
//       like the old builder, so it carries no pattern gate.
//   L2  RERoT path requires unified KV (production configuration) and
//       replicates rerot_build_query_layout visibility exactly; RERoT+SWA is
//       capability-ineligible (the old indexed path has no SWA filter, so a
//       fragment path must not invent one).
//   L3  RERoT and ordinary rows never share one layout (matches the existing
//       rerot_batch_active refusal; mixed ubatches propagate that throw).
//   L4  exact_rows are bounded by params.exact_cap; overflow is a hard error,
//       never silent truncation. Production sparse select should pass
//       want_exact_rows=false (fragments+groups only, no O(Q*K) output).
//   L5  staleness is fail-closed: layouts are keyed by (ubatch role,
//       capacities) and validated against the owner cache epoch plus the
//       per-stream CellGeneration (stamp, generation) pairs. Saturated
//       (MAX) stamps/generations are always-invalid. clear/reclaim/compact/
//       restore/publish/view/tag/write mutations all advance the stamps, so a
//       layout built before any of them never validates afterwards.
//   L6  multi-stream (non-unified) ordinary planning is explicit per stream
//       (query stream comes from seq_to_stream); it is supported, not
//       rejected. Only RERoT demands unified.
//   L7  MTP draft/verify, speculative replay, frontier generation, decode,
//       embedding, rerank, multimodal, and unknown roles are dense by role,
//       never planned sparse (recurrent/active-lane MTP restrictions kept).
//   L8  graph ordering: requires_cache_writes=true means pool/select/attn
//       nodes must be built AFTER the cpy_k/cpy_v expands in the same graph
//       (the layout sees post-apply metadata; K/V bytes land via those
//       nodes; ggml carries the edge through the shared K/V tensors).

#pragma once

#include "../include/llama-flashprefill.h" // frozen v1 roles/routes (values only)

#include "llama.h"         // llama_pos, llama_seq_id, LLAMA_MAX_SEQ
#include "llama-rerot.h"   // reader/run/visibility/group types
#include "llama-batch.h"   // llama_ubatch (borrowed rows for model-free builds)
#include "llama-kv-cells.h" // llama_kv_cells (sole cell owner; borrowed, never retained)
#include "llama-hparams.h"  // llama_swa_type (ordinary SWA legality domain)

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Build outcome: OK (eligible layout), HARD_ERROR (corrupt/overflow/capacity:
// fail the planning call), INELIGIBLE (legitimate stock-path content such as
// image rows/cells: dense with the reason, never an error).
enum class llama_flashprefill_build_status : uint8_t {
    OK         = 0,
    HARD_ERROR = 1,
    INELIGIBLE = 2,
};

// ---------------------------------------------------------------------------
// Build parameters (caller-supplied; BN comes from the wire/policy config via
// the caller, never guessed here)
// ---------------------------------------------------------------------------

struct llama_flashprefill_layout_params {
    uint32_t block_k = 128; // BN: logical K block, must be > 0
    bool causal = true;     // ordinary-path causal legality switch (from cparams)
    bool want_exact_rows = false; // oracle/debug only: expand deduped exact token
                                  // rows (O(E) output). Production stays false and
                                  // consumes uses (O(Q*F), no E expansion).
    uint32_t exact_cap = (1u << 24); // hard cap on oracle exact rows; overflow =>
                                     // checked error, never silent truncation

    static constexpr uint32_t CAP_BUCKET = 256; // capacity bucketing for graph reuse

    static uint32_t bucket_for(uint32_t n) {
        const uint64_t b = (uint64_t(n) + CAP_BUCKET - 1) / CAP_BUCKET * CAP_BUCKET;
        return b > UINT32_MAX ? UINT32_MAX : (uint32_t) b;
    }
};

// ---------------------------------------------------------------------------
// Use record: one (query, fragment) legality decision (production payload)
// ---------------------------------------------------------------------------
//
// The packer turns each use into exactly one wire use[8] per (tile, kv_head)
// fan-out. At most one use exists per (query, fragment) pair (wire key
// uniqueness on (source_query, tile, kv_head, frag_id) is then satisfiable by
// construction). sub_off/sub_count select a single contiguous member subrange
// of the fragment in MEMBER ORDER (positions/storage ascending, so every
// query's legal set is a prefix): full-fragment subsets may proxy downstream;
// partial subsets are MANDATORY exact (flags bit0). A missing (query,
// fragment) pair means no legal members (never an empty use: sub_count > 0).
//
// Owner-canonical emission order (builder invariant, relied on by the packer):
// uses are query-major — use_offsets[q]..use_offsets[q+1] holds query q's uses
// with fragment ids strictly ascending within each query. Tile/kv_head order
// is NOT established here: tiles (packed-Q blocking) and kv_head fan-out do
// not exist at planning time, so (tile, kv_head) sorting happens pack-side
// (GraphIntegration) before select; the wire schema itself stays
// order-agnostic.

struct llama_flashprefill_use {
    uint32_t query    = 0; // index into layout.queries (dense 0..nq-1)
    uint32_t fragment = 0; // frag_id into layout.fragments
    uint32_t group    = 0; // q_group: consuming Q phase group for this pair
    uint32_t sub_off  = 0; // absolute offset inside the fragment's member
                           // range (cell_begin-based or cell_refs-based)
    uint32_t sub_count = 0; // > 0, inside the fragment range
    uint32_t flags    = 0; // bit0 MANDATORY (partial subset, must classify exact)

    static constexpr uint32_t FLAG_MANDATORY = (1u << 0);
};

// ---------------------------------------------------------------------------
// Fragment descriptor (one legal compensation/exact unit)
// ---------------------------------------------------------------------------

enum class llama_flashprefill_fragment_domain : uint8_t {
    ORDINARY   = 0, // resident owned cells in one logical BN block + membership
    REROT_BASE = 1, // untagged serial prefix, one virtual BN + membership + phase
    REROT_RUN  = 2, // one run slice: uniform visibility class + phase + virtual BN
};

struct llama_flashprefill_fragment {
    llama_flashprefill_fragment_domain domain = llama_flashprefill_fragment_domain::ORDINARY;
    uint32_t stream = 0;
    // True when this fragment straddles a legality boundary for at least one
    // query row (partial causal/SWA edge, or gated RERoT membership): such
    // rows may only consume it as a MANDATORY exact subset, never as a whole-
    // fragment proxy. Wire fragment flags bit0 (BOUNDARY_PARTIAL) source.
    // FULL RERoT-run fragments visible to every sharing query are false.
    bool boundary_partial = false;

    // Physical addressing: exactly one form is valid.
    bool     contiguous     = false;
    uint32_t cell_begin     = 0;          // iff contiguous: members are [begin, begin+token_count)
    uint32_t cell_ref_offset = UINT32_MAX; // iff !contiguous: offset into layout.cell_refs, span token_count

    uint32_t token_count = 0;   // n_J: actual valid tokens, never padded capacity
    uint32_t logical_block = 0; // BN identity (ordinary: logical pos/BN; RERoT: virtual pos/BN)
    llama_pos logical_begin = 0; // inclusive min coordinate in the fragment's domain
    llama_pos logical_end   = 0; // exclusive max+1 in the fragment's domain

    // Ownership signature for ordinary and RERoT-base domains. A query row may
    // use this fragment only when its seq id is a member. RERoT-run FULL
    // fragments (view-visible regardless of ownership) leave this empty.
    std::bitset<LLAMA_MAX_SEQ> members;

    // RERoT domains only:
    uint64_t episode_id = 0;
    llama_rerot_run_id run_id = LLAMA_REROT_RUN_INVALID;
    llama_rerot_visibility visibility = llama_rerot_visibility::normal;
    uint32_t run_rank = 0;      // rank inside the reader's ordered_runs
    llama_pos virtual_pos0 = 0; // first TABLE virtual position covered (logical
                                // identity; per-query virtuals are derived in
                                // the emission pass, never stored here)
    int64_t phase_bias = 0;     // table-phase constant P = storage(member) minus
                                // piece-local member index, uniform over this
                                // fragment by split construction. Per-query
                                // effective position is
                                //   eff = query_virtual + P - prefix_count
                                // where prefix_count is the legal member count
                                // before this fragment for that query.
    bool gated = true;          // true => per-query ownership+causal gate still applies;
                                // false => visible to every query sharing the reader view
};

// ---------------------------------------------------------------------------
// Query row descriptor (who each planned row is)
// ---------------------------------------------------------------------------

struct llama_flashprefill_query {
    uint32_t query_index = 0; // row in the source ubatch
    llama_seq_id seq_id = -1; // first seq id of the row (matches KQ-mask convention)
    uint32_t stream = 0;      // owning stream via seq_to_stream
    llama_pos query_pos = 0;  // logical/storage query coordinate
    llama_pos query_virtual_pos = -1; // RERoT only; -1 for ordinary rows
};

// ---------------------------------------------------------------------------
// Topology key (graph-reuse contribution; freshness tracked separately)
// ---------------------------------------------------------------------------

struct llama_flashprefill_build_key {
    uint32_t ubatch_index = 0; // index into the mctx-owned ubatch vector
    int32_t  role = LLAMA_FLASHPREFILL_ROLE_UNKNOWN;
    uint32_t block_k = 0;
    uint32_t causal = 0;
    uint32_t want_exact_rows = 0;
    uint32_t n_tokens = 0;
    uint32_t exact_cap_bucket = 0;
    uint32_t group_cap_bucket = 0;
    uint32_t is_rerot = 0;

    bool operator==(const llama_flashprefill_build_key & o) const {
        return ubatch_index == o.ubatch_index && role == o.role && block_k == o.block_k &&
               causal == o.causal && want_exact_rows == o.want_exact_rows && n_tokens == o.n_tokens &&
               exact_cap_bucket == o.exact_cap_bucket && group_cap_bucket == o.group_cap_bucket &&
               is_rerot == o.is_rerot;
    }

    bool operator!=(const llama_flashprefill_build_key & o) const { return !(*this == o); }
};

// Per-stream CellGeneration stamps captured at build (freshness, not topology).
struct llama_flashprefill_cell_stamp {
    uint64_t stamp = 0;
    uint64_t generation = 0;
};

// ---------------------------------------------------------------------------
// Layout (short-lived; owned by the mctx; validated before every use)
// ---------------------------------------------------------------------------

struct llama_flashprefill_layout {
    bool eligible = false;
    int32_t dense_reason = LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF; // frozen route when !eligible
    std::string error; // hard-error text; empty when none (ineligible-by-policy is not an error)

    // Build scalars (packer inputs alongside the vectors; counts are sizes).
    uint32_t block_k    = 0; // BN used for logical_block identities
    uint32_t is_rerot   = 0; // 0 ordinary, 1 RERoT
    uint32_t causal     = 1; // legality shape applied (RERoT always causal-shaped)
    uint32_t swa_window = 0; // 0 when no SWA
    uint32_t swa_type   = 0; // llama_swa_type value

    std::vector<llama_flashprefill_query>    queries;    // [Q] query rows
    std::vector<llama_flashprefill_fragment> fragments;  // [F] legal fragments
    std::vector<uint32_t> cell_refs; // backing store for non-contiguous fragments

    std::vector<llama_rerot_attn_group> groups; // [G] raw-Q RoPE groups (effective_pos per group)
    std::vector<uint32_t> group_offsets; // [Q+1] per-query group ranges (groups[g].query_index agrees)

    // Production payload: compact per-(query, fragment) uses (built O(Q*F),
    // never expanded to token rows). use_offsets[q]..use_offsets[q+1] are the
    // uses of query q; at most one use per (query, fragment) pair.
    std::vector<llama_flashprefill_use> uses; // [U]
    std::vector<uint32_t> use_offsets; // [Q+1] per-query use ranges

    // Oracle payload only (want_exact_rows=true): exact token rows expanded
    // FROM uses (same sets, per-query deduped). Empty in production.
    std::vector<uint32_t> exact_rows;    // [E] physical cell indices
    std::vector<uint32_t> exact_groups;  // [E] consuming Q-group per exact row
    std::vector<uint32_t> exact_flags;   // [E] bit0 MANDATORY (partial subset row)
    std::vector<uint32_t> exact_offsets; // [Q+1] per-query exact ranges

    // Ordering + staleness contract for graph integration:
    uint64_t cells_epoch = 0; // owner cache-side epoch at build
    std::vector<llama_flashprefill_cell_stamp> cell_stamps; // per-stream (stamp, generation) at build
    uint32_t n_kv_at_build = 0; // resident scan width; graph must build against the same n_kv
    bool requires_cache_writes = true; // pool/select/attn nodes follow cpy_k/cpy_v expands (L8)

    // Proposed wire domain numbering (wire fragment word[3]; GGML schema v1
    // leaves numbering open, so WireReference confirms or remaps on pack).
    static uint32_t wire_domain(llama_flashprefill_fragment_domain d) {
        return d == llama_flashprefill_fragment_domain::REROT_BASE ? 1u :
               d == llama_flashprefill_fragment_domain::REROT_RUN  ? 2u : 0u;
    }

    uint32_t n_queries()   const { return (uint32_t) queries.size(); }
    uint32_t n_fragments() const { return (uint32_t) fragments.size(); }
    uint32_t n_groups()    const { return (uint32_t) groups.size(); }
    uint32_t n_uses()      const { return (uint32_t) uses.size(); }

    void clear() {
        eligible = false;
        dense_reason = LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
        error.clear();
        block_k = 0;
        is_rerot = 0;
        causal = 1;
        swa_window = 0;
        swa_type = 0;
        queries.clear();
        fragments.clear();
        cell_refs.clear();
        groups.clear();
        group_offsets.clear();
        uses.clear();
        use_offsets.clear();
        exact_rows.clear();
        exact_groups.clear();
        exact_flags.clear();
        exact_offsets.clear();
        cells_epoch = 0;
        cell_stamps.clear();
        n_kv_at_build = 0;
        requires_cache_writes = true;
    }

    // Structural self-check: I32 wire ranges on all sizes, offset
    // monotonicity/exactness, group/query agreement, fragment ranges/refs in
    // bounds, use subranges inside fragments with (query, fragment)
    // uniqueness, and — when present — per-row exact physical dedup.
    // scan_width is the resident width (cells.size() of the widest stream).
    // Returns false with a reason on any violation.
    bool validate(uint32_t scan_width, std::string * error = nullptr) const;
};

// ---------------------------------------------------------------------------
// Model-free RERoT table planning (owner- and fixture-shared entry)
// ---------------------------------------------------------------------------
//
// Derives legal fragments + per-query uses/groups for ONE planning call over
// resident cells, reader views, and caller-owned ubatch rows. All inputs are
// borrowed (cells, views, ubatch row pointers must outlive the call); no
// model, cache, tensors, or backends are involved, so state fixtures call
// this directly with synthetic cells/views/rows. The owner calls it once per
// distinct reader view. Visibility, dense per-query virtualization, and
// effective-phase grouping replicate llama_rerot_build_query_layout exactly;
// any semantic change there must be mirrored here (the state-test oracle
// expands uses back to per-query (key, effective) maps and compares).
llama_flashprefill_build_status llama_flashprefill_build_rerot_plan(
    const llama_kv_cells & cells,
    const std::vector<llama_rerot_reader_state> & views,
    const llama_ubatch & ubatch,
    const llama_flashprefill_layout_params & params,
    llama_flashprefill_layout & out,
    std::string * error = nullptr);

// ---------------------------------------------------------------------------
// Model-free ordinary table planning (owner- and fixture-shared entry)
// ---------------------------------------------------------------------------
//
// Ordinary-path counterpart of the RERoT entry above: same borrowing rules,
// no model/cache/tensors/backends. Membership signatures draw exclusively
// from the ubatch's queried primary seqs (seq_id[q][0] per stream), so
// foreign idle/private KV is skipped before any text/layout check and never
// splits fragments, trips text-compat, or shifts counts. SWA/causal legality
// replicates set_input_kq_mask exactly (1-D positions; 2-D rows need the
// text-pattern gate via require_text_cells, owned by the caller like the
// owner dispatcher).
llama_flashprefill_build_status llama_flashprefill_build_ordinary_plan(
    const llama_kv_cells_vec & v_cells,
    const std::vector<uint32_t> & seq_to_stream,
    uint32_t n_swa,
    llama_swa_type swa_type,
    bool causal,
    bool require_text_cells,
    const llama_ubatch & ubatch,
    const llama_flashprefill_layout_params & params,
    llama_flashprefill_layout & out,
    std::string * error = nullptr);

// ---------------------------------------------------------------------------
// Fragment budget for fit reserve sizing (FittingIntegration; pure, checked)
// ---------------------------------------------------------------------------
//
// Worst-case fragment/use counts WITHOUT assuming F = ceil(K/BN): every BN
// block may split further along membership/phase/visibility seams
// (split_allowance per block covers sequence-membership signatures, RERoT
// phase-bias changes, and visibility/gating cuts), and run_allowance covers
// RERoT run-table heads plus the untagged-base structure. Uses are bounded by
// F * q_cap (at most one use per (query, fragment) pair). Oracle exact rows
// are bounded by F * block_k: every fragment holds members of a single BN
// identity, so token_count can only exceed block_k through duplicate logical
// positions (vision repeats), which the builder rejects fail-closed instead
// of truncating.
//
// Checked 64-bit math: any overflow or I32 wire-domain breach is a hard
// failure (false + reason), never a saturating clamp — a clamped budget would
// let fit reserve less than the worst case and OOM at runtime. The budget
// sizes the reserve; the builder still enforces its exact_cap independently
// and errors rather than overflowing it.

struct llama_flashprefill_fragment_budget {
    uint32_t n_fragments = 0; // F worst case at capacity
    uint32_t n_uses      = 0; // U worst case (<= F * q_cap)
    uint32_t n_exact_rows = 0; // E worst case, oracle only (<= F * block_k)
};

inline bool llama_flashprefill_fragment_budget_for(
        uint32_t k_cap,
        uint32_t block_k,
        uint32_t n_streams,
        uint32_t split_allowance,
        uint32_t run_allowance,
        uint32_t q_cap,
        llama_flashprefill_fragment_budget * out,
        std::string * error = nullptr) {
    if (!out) {
        if (error) { *error = "flashprefill budget: null output"; }
        return false;
    }
    *out = llama_flashprefill_fragment_budget{};
    if (block_k == 0 || n_streams == 0) {
        if (error) { *error = "flashprefill budget: block_k and n_streams must be non-zero"; }
        return false;
    }
    const uint64_t blocks = uint64_t(k_cap) / uint64_t(block_k) + 1u; // +1 partial tail
    const uint64_t frags = blocks * uint64_t(split_allowance ? split_allowance : 1u) *
                           uint64_t(n_streams) + uint64_t(run_allowance);
    const uint64_t uses  = frags * uint64_t(q_cap);
    const uint64_t exact = frags * uint64_t(block_k);
    // Wire I32 domain: metadata counts, plan per-pair capacities, and every
    // I32 word derived from them must fit int32.
    if (frags > uint64_t(INT32_MAX) || uses > uint64_t(INT32_MAX) || exact > uint64_t(INT32_MAX)) {
        if (error) { *error = "flashprefill budget exceeds I32 wire domain: lower K/block policy or raise split discipline"; }
        return false;
    }
    out->n_fragments = uint32_t(frags);
    out->n_uses      = uint32_t(uses);
    out->n_exact_rows = uint32_t(exact);
    return true;
}

// ---------------------------------------------------------------------------
// Shared admission allowances (single definition for graph.cpp/fit.cpp)
// ---------------------------------------------------------------------------
//
// DEVELOPMENT, unbenchmarked guard — never a measured cutoff. These frozen
// allowances size the admission reserve (worst-case F/U/E) identically in
// graph and fit code; both call admission_budget_for() below instead of
// duplicating the values. Pinned to the layout/wire version: changing any
// allowance is a policy change and requires a version bump (see
// ggml/include/ggml-flashprefill.h GGML_FLASHPREFILL_VERSION). Values
// already match both callers; no caller keeps a private copy.

static constexpr uint32_t LLAMA_FLASHPREFILL_ADMISSION_SPLIT = 4u; // membership/phase/visibility splits per BN block
static constexpr uint32_t LLAMA_FLASHPREFILL_ADMISSION_RUN = 65u; // RERoT run-table headroom (run_allowance units)
static constexpr uint32_t LLAMA_FLASHPREFILL_ADMISSION_STREAMS_UNIFIED = 1u; // unified admission stream count

inline bool llama_flashprefill_admission_budget_for(
        uint32_t k_cap,
        uint32_t block_k,
        uint32_t n_streams,
        uint32_t q_cap,
        llama_flashprefill_fragment_budget * out,
        std::string * error = nullptr) {
    if (!llama_flashprefill_fragment_budget_for(
            k_cap, block_k, n_streams,
            LLAMA_FLASHPREFILL_ADMISSION_SPLIT, LLAMA_FLASHPREFILL_ADMISSION_RUN,
            q_cap, out, error)) {
        return false;
    }
    // Structural clamp: fragments cannot exceed resident tokens. Keeps the
    // small-K reserve pool sensible (never a second KV mirror: this only
    // tightens the count bound, never stores per-token data).
    if (out->n_fragments > k_cap) {
        const uint64_t f = uint64_t(k_cap);
        const uint64_t u = f * uint64_t(q_cap);
        const uint64_t e = f * uint64_t(block_k);
        // Clamped products only shrink from already-validated values, so they
        // stay inside the I32 wire domain by construction; re-check anyway.
        if (u > uint64_t(INT32_MAX) || e > uint64_t(INT32_MAX)) {
            if (error) { *error = "flashprefill admission budget exceeds I32 wire domain after clamp"; }
            return false;
        }
        out->n_fragments = uint32_t(f);
        out->n_uses      = uint32_t(u);
        out->n_exact_rows = uint32_t(e);
    }
    return true;
}
