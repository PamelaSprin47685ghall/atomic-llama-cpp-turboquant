#pragma once

// ggml-vulkan-landmark.h — Native Vulkan phase-aware quantized landmark scoring,
// deterministic bounded global top-k selection, fragment→row CSR expansion, and
// deterministic multi-segment merge for XKV (v2).
//
// Graph contract:
//  - Code streams, query vectors, positions, metadata, and CSR outputs are backend-resident
//    scheduler-visible tensors (arena slices);
//  - ggml op_params carry POD scalars only (64 bytes), no host pointers;
//  - Vulkan path keeps code streams and tables device-resident, bounded caller-owned
//    scratch, fail-closed semantics, one-shot status;
//  - Q8_0 and Turbo4_0 quantized landmark codecs are explicit and capability-gated;
//  - Vector pairing is NeoX HALF / front-back per AGENTS.md (position sections may be
//    interleaved in IMRoPE, but ggml vector pairing is always HALF);
//  - Deterministic top-k: strictly score-descending, stable global-id-ascending tie-break;
//  - Future speculative queries never influence earlier queries (strict per-query isolation);
//  - Illegal / all-masked inputs return valid empty CSR state without NaN;
//  - Zero or oversized budget fails closed (never select-all);
//  - Bounded boundary refinement enforces refine_cap and records cap-hit (never fallback).
//
// History independence (v2): scratch is O(nq*(2*top_k + refine_cap + TILE + const)).
// There is NO nq*n_frags score matrix anywhere. Fragments stream in fixed TILE_FRAGS
// tiles; each per-query workgroup holds a carry of at most top_k (idx+score) plus at
// most refine_cap rows. n_frags itself is unbounded (only tile residency is bounded).
//
// Capability maxima (v2): the CLI may request any positive sr_budget/refine cap, but the
// native op supports budgets dynamically through scratch only up to the explicit maxima
// below; anything above is rejected BEFORE graph build (supports()==false, oracle==false,
// status ERR_BUDGET). No shader silently truncates: every fixed local array is backed by
// its maximum, larger requests fail closed. Boundary tests cover max and max+1.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_XKV_LANDMARK_VERSION 2
#define GGML_XKV_LANDMARK_VERSION_MIN 1  // v1 params accepted as legacy (flags=0, stride-4 meta)

// RoPE modes (matches ggml-xkv.h)
#define GGML_XKV_LANDMARK_ROPE_HALF        0  // NeoX front-back half: [re(0:Fc), im(Fc:2Fc)]
#define GGML_XKV_LANDMARK_ROPE_INTERLEAVED 1  // GPT-J / IMRoPE pairs: (0,1), (2,3), ...

// ---- explicit capability maxima: shader/descriptor budgets (v2) ----
// Score kernel register file: full landmark decode needs head_dim floats/thread.
// Larger requests are rejected before build, never truncated.
#define GGML_XKV_LANDMARK_MAX_HEAD_DIM   256
#define GGML_XKV_LANDMARK_MAX_PADDED_DIM 256
#define GGML_XKV_LANDMARK_MAX_ROTARY_DIM 256
// Selection carry per query: top_k (idx+score) + take + refined|hit + refine rows.
#define GGML_XKV_LANDMARK_MAX_TOP_K      128
#define GGML_XKV_LANDMARK_MAX_REFINE_CAP 1024
// Fan-out bounds.
#define GGML_XKV_LANDMARK_MAX_QUERIES    4096
#define GGML_XKV_LANDMARK_MAX_Q_HEADS    32
// Fixed streaming tile: fragments per tile per query workgroup (history-independent).
#define GGML_XKV_LANDMARK_TILE_FRAGS     64
// Rows op: max explicit rows per fragment (contiguous count or sparse list length).
#define GGML_XKV_LANDMARK_MAX_FRAG_ROWS  1024
// Merge op: max candidate sets fused in one call (segments/spans).
#define GGML_XKV_LANDMARK_MAX_SEGMENTS   64
// Push-constant budget for every landmark kernel: 64 bytes (16 words).
// (19 words: 16-word v1 POD + fstride, qstride, flags).
#define GGML_XKV_LANDMARK_PUSH_BYTES     76
// Descriptor budgets: score 8 bindings, select 7 bindings, rows 8 bindings, merge 6.

// Status codes emitted to status tensor I32[4]
#define GGML_XKV_LANDMARK_STATUS_OK               0
#define GGML_XKV_LANDMARK_STATUS_ERR_BUDGET       1 // zero/oversized cap or above maxima
#define GGML_XKV_LANDMARK_STATUS_ERR_INPUT        2 // null or invalid input dimensions
#define GGML_XKV_LANDMARK_STATUS_ERR_WORKSPACE    3 // scratch buffer too small
#define GGML_XKV_LANDMARK_STATUS_ERR_UNSUPPORTED  4 // unsupported codec or params

// Metadata strides.
// frag_meta legacy stride-4: [row_begin,row_count,segment_id,legal_flags].
// frag_meta extended stride-6: + [row_off,group_id]; row_off indexes frag_row_ids
//   for sparse/partial fragments (row_off<0 => contiguous [row_begin,row_count)).
// query_meta legacy stride-4: [causal_limit_pos,query_pos,n_q_heads,kv_head].
// query_meta extended stride-6: + [epoch,vis_group]; epoch enables per-query
//   generation gating against frag_gen (see select v2 builder).
#define GGML_XKV_LANDMARK_FRAG_META_STRIDE   4
#define GGML_XKV_LANDMARK_FRAG_META_STRIDE_X 6
#define GGML_XKV_LANDMARK_QUERY_META_STRIDE  4
#define GGML_XKV_LANDMARK_QUERY_META_STRIDE_X 6
// v3 device path: frag_meta stride-8 [row_begin,row_count,segment_id,legal_flags,
// row_off,group_id,fpos,gen]. Positions AND generations ride in the meta so the
// frag_positions tensor is a 1-element dummy (ignored) and no extra src slot is
// needed (10/10 full). frag_row_off/row_ids stay rows-op-only. Query meta stays 4/6.
#define GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3 8

// ---- phase-domain contract (authoritative: PHASED-DIRECT) ----
// Production landmarks are row-wise forward-RoPE-then-mean: each row is phased at
// its OWN storage position before the mean, so the selector MUST score the stored
// landmark directly and MUST NOT apply representative-position RoPE (that would
// double-phase and diverge whenever positions vary within the chunk). frag_positions
// (v2) / meta fpos (v3) are causal-gating-only inputs.
// Experimental canonical mode (FLAG_CANONICAL_XPHASE, host oracle only, device
// rejects): a single-position phase is mathematically defined ONLY for uniform
// single-row fragments (row_count==1 with exact rope tables whose fingerprint
// matches the build-time fingerprint). Anything else (multi-row frags, fp mismatch,
// missing tables) fails closed. ggml_xkv_landmark_phase_fingerprint() computes the
// fingerprint both sides compare.
GGML_API uint64_t ggml_xkv_landmark_phase_fingerprint(
        const float * omega_mag_2fc, uint32_t fc,
        uint32_t rope_mode, uint32_t rotary_dim, uint32_t head_dim,
        uint32_t landmark_type);

// Flags in params._reserved (v2; v1 legacy == 0)
#define GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS  (1u << 0) // rows op: use frag_row_ids
#define GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL   (1u << 1) // select: enforce frag_gen vs query epoch
#define GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE (1u << 2) // experimental: single-position
                                              // phase of canonical landmarks (host
                                              // oracle only; device rejects). See phase
                                              // domain contract below.

// POD parameters for GGML_OP_XKV_LANDMARK (64 bytes, fits in GGML_MAX_OP_PARAMS=64).
// Unchanged layout from v1 so existing dispatch memcpy stays valid; _reserved is
// now flags, and caps are enforced against the maxima above.
typedef struct ggml_xkv_landmark_params {
    uint32_t version;         // GGML_XKV_LANDMARK_VERSION (1 accepted legacy)
    uint32_t n_queries;       // number of queries (>= 1, <= MAX_QUERIES)
    uint32_t n_frags;         // total candidate fragments across DDVR spans/segments (unbounded)
    uint32_t head_dim;        // landmark vector dim (<= MAX_HEAD_DIM)
    uint32_t padded_dim;      // padded row width (<= MAX_PADDED_DIM; 32-align Q8, 128-mult T4)
    uint32_t rotary_dim;      // RoPE dim (0 = none; even, <= head_dim, <= MAX_ROTARY_DIM)
    uint32_t rope_mode;       // HALF / INTERLEAVED
    uint32_t landmark_type;   // GGML_TYPE_Q8_0 (8) or GGML_TYPE_TURBO4_0 (44)
    uint32_t top_k;           // per-query selection budget (>0, <= max_top_k, <= MAX_TOP_K)
    uint32_t max_top_k;       // capacity bound (<= MAX_TOP_K)
    uint32_t refine_cap;      // boundary refinement cap (<= MAX_REFINE_CAP; 0 = none)
    uint32_t frag_size;       // default rows per fragment for contiguous fallback
    float    scale;           // attention scale (e.g. 1/sqrt(head_dim))
    uint32_t n_q_heads;       // GQA group size (>= 1, <= MAX_Q_HEADS)
    uint32_t n_rows_total;    // total rows (contiguous clipping bound)
    uint32_t _reserved;       // v2 flags (SPARSE_ROWS / PERQ_LEGAL); v1 == 0
} ggml_xkv_landmark_params;

#ifdef __cplusplus
static_assert(sizeof(ggml_xkv_landmark_params) == 64, "ggml_xkv_landmark_params must be 64 bytes");
#endif

// ---- scratch layouts (v2, history-independent) ----
// Select scratch (words), NO n_frags term:
//   CARRY_IDX  nq*top_k      uint   per-query carry fragment ids (sorted desc/asc)
//   CARRY_SC   nq*top_k      float  per-query carry scores
//   TAKE       nq            uint   per-query take counts (pass1 -> pass2 fold)
//   REF        nq            uint   per-query refined|hit-bit (pass1 -> pass2)
//   LEGAL      nq            uint   per-query legal-fragment counts (legality CSR counts)
//   ROWBUF     nq*refine_cap uint   per-query sorted unique refine rows (cap==0: absent)
//   TILE       nq*TILE_FRAGS float  per-query tile score staging (score kernel)
//   +64 margin words. Select pass1 never aliases another region.
GGML_API bool ggml_xkv_landmark_workspace_bytes(
        const ggml_xkv_landmark_params * params,
        size_t * out_bytes,
        char * err, size_t err_size);

// Documented word offsets into the select scratch (for graph-runtime consumers that
// read per-query legality CSR counts / takes directly off device memory).
GGML_API bool ggml_xkv_landmark_scratch_map(
        const ggml_xkv_landmark_params * params,
        size_t * carry_idx_words, size_t * carry_sc_words,
        size_t * take_words, size_t * ref_words, size_t * legal_words,
        size_t * rowbuf_words, size_t * tile_words,
        char * err, size_t err_size);

// Graph builder (scheduler-visible tensors, no host pointers in op_params).
// Node model: dst = csr_indices I32[top_k, n_queries]; srcs as listed.
// frag_meta accepts stride 4 (contiguous legacy) or 6 (sparse-capable + group);
// query_meta accepts stride 4 or 6 (epoch-gated legality iff flags&PERQ_LEGAL).
// scratch must be >= workspace_bytes (history-independent).
GGML_API struct ggml_tensor * ggml_xkv_landmark(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * landmarks,
        struct ggml_tensor  * frag_positions,
        struct ggml_tensor  * frag_meta,
        struct ggml_tensor  * query_meta,
        struct ggml_tensor  * rope_tables,
        struct ggml_tensor  * scratch,
        struct ggml_tensor  * csr_ptrs,
        struct ggml_tensor  * topk_scores,
        struct ggml_tensor  * status,
        const ggml_xkv_landmark_params * params);

// Lightweight validation hook (shapes/types/params/caps/scratch, no contents).
// Rejects above-maximum budgets BEFORE graph build (fail-closed, never truncate).
GGML_API bool ggml_xkv_landmark_supports(
        const struct ggml_tensor * q,
        const struct ggml_tensor * landmarks,
        const struct ggml_tensor * frag_positions,
        const struct ggml_tensor * frag_meta,
        const struct ggml_tensor * query_meta,
        const struct ggml_tensor * rope_tables,
        const struct ggml_tensor * scratch,
        const struct ggml_tensor * csr_ptrs,
        const struct ggml_tensor * topk_scores,
        const struct ggml_tensor * status,
        const struct ggml_tensor * dst_csr_indices,
        const ggml_xkv_landmark_params * params,
        char * err, size_t err_size);

// CPU independent reference oracle: tiled streaming carry (mirrors the device exactly,
// including tile order and carry merge), exact phase-aware decode, deterministic
// score-desc/global-id-asc top-k, per-query CSR, bounded refinement. No nq*nf buffer.
GGML_API bool ggml_xkv_landmark_cpu_oracle(
        const float   * q_data,
        const void    * landmarks_data, enum ggml_type landmark_type,
        const int32_t * frag_positions,
        const int32_t * frag_meta,
        const int32_t * query_meta,
        const float   * rope_tables,
        const ggml_xkv_landmark_params * params,
        int32_t       * out_csr_ptrs,
        int32_t       * out_csr_indices,
        float         * out_topk_scores,
        int32_t       * out_status,
        char * err, size_t err_size);

// Extended oracle: explicit frag/query meta strides (4 legacy, 6 extended) and optional
// per-fragment generation gate (frag_gen, non-NULL iff flags&PERQ_LEGAL). Same streaming
// semantics as the legacy entry; legacy == _x with strides 4/4 and NULL generation.
GGML_API bool ggml_xkv_landmark_cpu_oracle_x(
        const float   * q_data,
        const void    * landmarks_data, enum ggml_type landmark_type,
        const int32_t * frag_positions,
        const int32_t * frag_meta, int32_t frag_meta_stride,
        const int32_t * query_meta, int32_t query_meta_stride,
        const int32_t * frag_gen, // optional per-frag generation (NULL unless PERQ_LEGAL)
        const float   * rope_tables,
        const ggml_xkv_landmark_params * params,
        uint64_t expected_phase_fp, // required iff FLAG_CANONICAL_XPHASE (else ignored)
        int32_t       * out_csr_ptrs,
        int32_t       * out_csr_indices,
        float         * out_topk_scores,
        int32_t       * out_status,
        char * err, size_t err_size);

// v3 oracle: stride-8 frag meta carries fpos+gen ([6],[7]); frag_positions and
// frag_gen MUST be NULL (ambiguity rejected). query_meta stride 4 or 6 as usual.
// expected_phase_fp is consumed only iff FLAG_CANONICAL_XPHASE is set (else ignored).
GGML_API bool ggml_xkv_landmark_v3_cpu_oracle(
        const float   * q_data,
        const void    * landmarks_data, enum ggml_type landmark_type,
        const int32_t * frag_meta, // stride-8, n_frags rows
        const int32_t * query_meta, int32_t query_meta_stride,
        const float   * rope_tables,
        const ggml_xkv_landmark_params * params,
        uint64_t expected_phase_fp,
        int32_t       * out_csr_ptrs,
        int32_t       * out_csr_indices,
        float         * out_topk_scores,
        int32_t       * out_status,
        char * err, size_t err_size);

// ---- fragment→row CSR expansion (device row-ref feed for reconstruct, no D2H) ----
// Expands per-query selected fragment ids into explicit row-ref CSR consumable by
// GGML_OP_XKV_RECONSTRUCT without host round-trip. Deterministic: selected frags
// ascending, rows in stored order per frag (contiguous [row_begin,row_count) unless
// flags&SPARSE_ROWS, then frag_row_ids[row_off .. +row_len)). Row refs use the
// GGML_XKV_REF_STRIDE-4 layout [a_row,group,layer_slot,head] with per-frag
// (group,arena,head) from frag_kv [3,n_frags] (frag_kv[1] is the arena ordinal)
// and positions from row_pos.
// Mapped mode embeds [parent_query,ddvr_slot] before each selected-fragment list;
// expanded queries must be parent-major and DDVR slots strictly increasing per
// parent. The output rows remain compact while row_ptrs describes parent queries.
// Bounded: each expanded query emits <= refine_cap rows (zero rejected); overflow
// sets cap-hit, output is clamped, and there is no select-all fallback.
// Per-arena contract (device-native reconstruct without D2H): arena_filter selects
// one arena (frag_kv[1]==arena_filter; UINT32_MAX == all). Filtering happens before
// cap accounting. refs[0] is arena-local (global_row-global_row_base, validated
// < arena_row_count). Outputs are a compact ordinal tile window
// [output_row_begin, output_row_begin+tile_capacity) over the flattened parent-major
// rows, where tile_capacity=min(1024,n_queries*refine_cap). rec_idx is tile-local,
// row_ptrs counts only this window, entries use source=2 (COLD) with valid masking.
GGML_API bool ggml_xkv_landmark_rows_workspace_bytes(
        uint32_t n_queries, uint32_t top_k, uint32_t refine_cap,
        size_t * out_bytes, char * err, size_t err_size);

struct ggml_xkv_landmark_rows_params; // defined below (64B device op params)

GGML_API bool ggml_xkv_landmark_rows_cpu_oracle(
        const int32_t * sel_indices, // [top_k(+2), n_queries] map + frag ids (-1 sentinel)
        const int32_t * frag_meta, // [fstride, n_frags], fstride = params->fstride
        const int32_t * frag_row_off, // [n_frags] or NULL (contiguous)
        const int32_t * frag_row_ids, // sparse row pool or NULL
        const int32_t * frag_kv,      // [3, n_frags] (group,arena,head)
        const int32_t * row_pos,      // [n_rows_total] storage pos per row id (NULL => frag_pos)
        const int32_t * frag_positions,
        const struct ggml_xkv_landmark_rows_params * params,
        int32_t * out_row_ptrs,  // [n_parent_queries+1] mapped, else [n_queries+1]
        int32_t * out_row_refs,  // [4, tile_capacity]
        int32_t * out_row_pos,   // [tile_capacity]
        int32_t * out_row_entries, // [4, tile_capacity] [source=2,rec_idx,slot,valid] or NULL
        int32_t * out_row_status,// I32[4] [code,emitted,cap_hits,total_filtered]
        char * err, size_t err_size);

// ---- deterministic multi-segment merge (online global top-k) ----
// Merges n_sets per-query candidate lists (global frag ids + scores, -1/-inf
// sentinels for short lists) into one top_k list per query, score-desc /
// global-id-asc. Streaming equivalent: merging per-segment top-k lists in any
// segment order yields the same result as one global select over the union
// (segments are disjoint id ranges; ties broken by global id). One-shot fail-closed.
GGML_API bool ggml_xkv_landmark_merge_cpu_oracle(
        const int32_t * set_indices, // [set_cap, n_queries, n_sets] global frag ids
        const float   * set_scores,  // same shape
        uint32_t n_queries, uint32_t n_sets, uint32_t set_cap, uint32_t top_k,
        int32_t * out_indices, // [top_k, n_queries]
        float   * out_scores,  // same
        char * err, size_t err_size);

// Merge CPU oracle with explicit per-set global ID offsets (base mapping) and deterministic status.
GGML_API bool ggml_xkv_landmark_merge_with_base_cpu_oracle(
        const int32_t  * set_indices,
        const float    * set_scores,
        const uint32_t * set_base,
        uint32_t n_queries, uint32_t n_sets, uint32_t set_cap, uint32_t top_k,
        int32_t * out_indices, float * out_scores, int32_t * out_status,
        char * err, size_t err_size);

// ---- executable rows / merge device ops (GGML_OP_XKV_LANDMARK_ROWS & _MERGE) ----
// Fully registered in GGML enum, RPC wire protocol, and CPU/Vulkan backends.
// Rows expands device selected fragment IDs into compact row refs + positions
// + attention entries without D2H. Merge fuses multi-segment candidate lists
// into a single global top-k per query with set_base global ID mapping.

// POD params for the rows device op (64 bytes = 16 words). Vulkan derives a
// separate 64B push struct copying words 0..14 with word 15 as status_flag.
typedef struct ggml_xkv_landmark_rows_params {
    uint32_t version;       // GGML_XKV_LANDMARK_VERSION
    uint32_t n_queries;     // E: number of selector queries (expanded (parent, slot))
    uint32_t top_k;
    uint32_t n_frags;
    uint32_t refine_cap;    // per-query row bound (>0, <= MAX_REFINE_CAP)
    uint32_t fstride;       // frag_meta stride (4 or 6)
    uint32_t flags;         // FLAG_SPARSE_ROWS
    uint32_t max_frag_rows; // explicit per-frag row bound (<= MAX_FRAG_ROWS)
    uint32_t n_rows_total;  // exact row_pos domain, required
    uint32_t n_parent_queries; // N in mapped mode; zero in identity mode
    uint32_t has_query_map; // 1 => sel_idx first dimension is top_k + 2
    uint32_t arena_filter;  // frag_kv[1] arena ordinal; UINT32_MAX == all arenas
    uint32_t global_row_base; // arena window base in global row ids
    uint32_t arena_row_count;  // rows in this arena (refs[0] validated < count)
    uint32_t output_row_begin;  // compact ordinal tile start (windowing)
    uint32_t _reserved;     // zero
} ggml_xkv_landmark_rows_params;

// POD params for the merge device op (32 bytes = 8 words, matches shader push).
typedef struct ggml_xkv_landmark_merge_params {
    uint32_t version;       // GGML_XKV_LANDMARK_VERSION
    uint32_t n_queries;
    uint32_t n_sets;        // <= MAX_SEGMENTS
    uint32_t top_k;
    uint32_t set_cap;       // per-set list length (<= MAX_TOP_K)
    uint32_t has_set_base;  // 1 if set_base tensor provided, 0 if IDs already global
    uint32_t _pad0, _pad1;  // zero
} ggml_xkv_landmark_merge_params;

#ifdef __cplusplus
static_assert(sizeof(ggml_xkv_landmark_rows_params) == 64, "rows params must be 64 bytes");
static_assert(sizeof(ggml_xkv_landmark_merge_params) == 32, "merge params must be 32 bytes");
#endif

// Rows graph builder: dst = row_refs I32[4, tile_capacity] (device-written),
// tile_capacity = min(1024, n_queries*refine_cap).
// src0 sel_idx I32[top_k(+2),E] (optional parent/slot map + device frag CSR),
// src1 frag_meta, src2 frag_row_off,
// src3 frag_row_ids, src4 frag_kv I32[3,nf] [group,arena,head], src5 row_pos I32[n_rows_total],
// src6 row_ptrs I32[N+1], src7 row_out_pos I32[tile_capacity],
// src8 row_entries I32[4, tile_capacity] [source=2, reconstructed_index, ddvr_slot, valid],
// src9 row_status I32[4] (WRITTEN).
GGML_API struct ggml_tensor * ggml_xkv_landmark_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * sel_idx,
        struct ggml_tensor  * frag_meta,
        struct ggml_tensor  * frag_row_off,
        struct ggml_tensor  * frag_row_ids,
        struct ggml_tensor  * frag_kv,
        struct ggml_tensor  * row_pos,
        struct ggml_tensor  * row_ptrs,
        struct ggml_tensor  * row_out_pos,
        struct ggml_tensor  * row_entries,
        struct ggml_tensor  * row_status,
        const ggml_xkv_landmark_rows_params * params);

GGML_API bool ggml_xkv_landmark_rows_supports(
        const struct ggml_tensor * sel_idx,
        const struct ggml_tensor * frag_meta,
        const struct ggml_tensor * frag_row_off,
        const struct ggml_tensor * frag_row_ids,
        const struct ggml_tensor * frag_kv,
        const struct ggml_tensor * row_pos,
        const struct ggml_tensor * row_ptrs,
        const struct ggml_tensor * row_refs,
        const struct ggml_tensor * row_out_pos,
        const struct ggml_tensor * row_entries,
        const struct ggml_tensor * row_status,
        const ggml_xkv_landmark_rows_params * params,
        char * err, size_t err_size);

// Merge graph builder: dst = out_idx I32[top_k,nq] (device-written).
// src0 set_idx I32[set_cap,nq,n_sets] (candidate ids, -1 sentinels), src1 set_sc
// F32 (same shape), src2 set_base I32[n_sets] (per-set base offset, or dummy/null),
// src3 out_sc F32[top_k,nq] (WRITTEN), src4 status I32[4] (WRITTEN).
GGML_API struct ggml_tensor * ggml_xkv_landmark_merge(
        struct ggml_context * ctx,
        struct ggml_tensor  * set_idx,
        struct ggml_tensor  * set_sc,
        struct ggml_tensor  * set_base,
        struct ggml_tensor  * out_sc,
        struct ggml_tensor  * status,
        const ggml_xkv_landmark_merge_params * params);

GGML_API bool ggml_xkv_landmark_merge_supports(
        const struct ggml_tensor * set_idx,
        const struct ggml_tensor * set_sc,
        const struct ggml_tensor * set_base,
        const struct ggml_tensor * out_sc,
        const struct ggml_tensor * status,
        const struct ggml_tensor * dst_out_idx,
        const ggml_xkv_landmark_merge_params * params,
        char * err, size_t err_size);

#ifdef __cplusplus
}
#endif
