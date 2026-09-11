#pragma once

// ggml-xkv.h — native XKV selected-row factor reconstruction op (v1).
//
// First correct native path (see local://xkv-vulkan-runtime.md):
//  - reconstructs only selected K/V rows from encoded A_K/B_K/A_V/B_V,
//    never projected Q (Q·B is invalid: B does not commute with RoPE);
//  - dequantizes/dots A/B along padded rank in their validated common
//    Turbo rotation domain ((A H^T)(B H^T)^T = A B^T), no decoded B mirror;
//  - applies exact partial/interleaved text RoPE to reconstructed canonical
//    K at storage position; canonical V output;
//  - emits bounded selected dense tensors only.
//
// Graph contract:
//  - code streams, refs, positions, group metadata are backend-resident
//    scheduler-visible tensors (arena slices);
//  - ggml op_params carry POD scalars/offsets/fingerprints only, no host
//    pointers;
//  - Vulkan path keeps code streams device-resident, single dispatch per op;
//  - unsupported codec/layout fails explicitly, never silently selects all.
//
// Owned by XkvVulkanFactorOps. Avoid llama-kv-cache/graph/server/state/store.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_XKV_VERSION 2

#define GGML_XKV_ROPE_HALF        0  // NeoX front-back half: [re(0:Fc), im(Fc:2Fc)]
#define GGML_XKV_ROPE_INTERLEAVED 1  // GPT-J pairs: (0,1),(2,3),...

// 8 ints per group in group_meta I32 tensor [8, n_groups]:
//   [0]=n_layers, [1]=n_heads, [2]=b_k_offset, [3]=b_k_rows,
//   [4]=b_v_offset, [5]=b_v_rows, [6]=rank_k, [7]=rank_v
// 5 ints per owning layer in layer_meta I32 tensor [5, n_layers], keyed by
// layer_slot. Store groups carry explicit per-layer maps (aliases share one
// offset; tail/unequal K-V layers have their own dims); the op NEVER derives
// layer slices from a uniform layer_slot*n_heads*dim formula:
//   [0]=offset_k (absolute B_K row), [1]=dim_k,
//   [2]=offset_v (absolute B_V row), [3]=dim_v,
//   [4]=n_kv_heads (head bound for this layer_slot)
#define GGML_XKV_LAYER_META_STRIDE 5
#define GGML_XKV_GROUP_META_STRIDE 8


// 4 ints per selection in refs I32 tensor [4, n_sel]:
//   [0]=a_row, [1]=group, [2]=layer_slot, [3]=head
#define GGML_XKV_REF_STRIDE 4

// One op per exact segment-group (v1): n_groups must be 1. Independently
// encoded groups / adaptive (tail) ranks run as separate ops with their own
// exact rank/descriptor params — never one A stream spanning groups.
// group_meta carries the group envelope (ranks, B-row coverage); the exact
// per-ref slice always comes from layer_meta (absolute offsets/dims/heads).
// RoPE is table-driven: the rope_tables tensor carries precomputed
// per-frequency effective omega + magnitude scale (exact ggml_rope_ext/YaRN
// equivalent, partial IMRoPE HALF pairing); scalar theta is NOT consumed.

// Rope tables layout: F32 vector [2*Fc], Fc=rotary_dim/2; first Fc entries
// are effective omega per frequency, next Fc are magnitude scales
// (sqrt(freq_scale_sq)). rotary_dim==0 => zero-element rope tensor.

// POD op params (52 bytes, fits in GGML_MAX_OP_PARAMS=64). No pointers.
typedef struct ggml_xkv_reconstruct_params {
    uint32_t version;      // must be GGML_XKV_VERSION
    uint32_t n_sel;        // number of selected rows
    uint32_t n_groups;     // must be 1 (one exact segment-group per op)
    uint32_t rank_k;       // logical K rank of this exact group
    uint32_t rank_v;       // logical V rank of this exact group
    uint32_t dim_k;        // K head dim of this exact group
    uint32_t dim_v;        // V head dim of this exact group
    uint32_t rotary_dim;   // RoPE rotary dim (even, <= dim_k; 0 = no RoPE)
    uint32_t rope_mode;    // GGML_XKV_ROPE_HALF or _INTERLEAVED
    uint32_t seed_k;       // Turbo table seed for K pair (factor codec K seed)
    uint32_t seed_v;       // Turbo table seed for V pair (factor codec V seed = K seed + 1)
    uint64_t fp_combined;  // FNV over per-stream (table_fp, format_rev)
} ggml_xkv_reconstruct_params;

// Landmark selection config (POD, host-side helper, shares deterministic top-k).
// Query input q is already post/effective-position RoPE: scoring phases ONLY
// the canonical landmark K to its storage position (never rotates Q again).
typedef struct ggml_xkv_landmark_config {
    uint32_t dim;          // landmark vector dim (== dim_k of its head)
    uint32_t rotary_dim;   // RoPE dim for phase-aware scoring (0 = no RoPE)
    uint32_t rope_mode;    // HALF / INTERLEAVED
    uint32_t top_k;        // bounded selection count (0 = fail, never select-all)
    uint32_t refine_max_rows; // boundary refinement cap (0 = none)
} ggml_xkv_landmark_config;

// ---- fingerprints / sizing (CPU + Vulkan share) ----
GGML_API uint64_t ggml_xkv_table_fingerprint(enum ggml_type type, uint32_t seed);
GGML_API uint64_t ggml_xkv_format_revision(enum ggml_type type);
GGML_API uint64_t ggml_xkv_fp_combined(enum ggml_type a_k, enum ggml_type b_k,
                                       enum ggml_type a_v, enum ggml_type b_v,
                                       uint32_t seed_k, uint32_t seed_v);
GGML_API int32_t  ggml_xkv_padded_rank(enum ggml_type type, uint32_t logical_rank);
GGML_API bool     ggml_xkv_codec_supported(enum ggml_type type);

// ---- graph builder (no host pointers in op_params) ----
GGML_API struct ggml_tensor * ggml_xkv_reconstruct(
        struct ggml_context * ctx,

        struct ggml_tensor  * a_k,        // [padded_rank_k, n_rows_a]
        struct ggml_tensor  * b_k,        // [padded_rank_k, total_feat_k]
        struct ggml_tensor  * a_v,        // [padded_rank_v, n_rows_a]
        struct ggml_tensor  * b_v,        // [padded_rank_v, total_feat_v]
        struct ggml_tensor  * refs,       // I32 [4, n_sel]
        struct ggml_tensor  * positions,  // I32 [n_sel]
        struct ggml_tensor  * group_meta, // I32 [8, n_groups]
        struct ggml_tensor  * layer_meta, // I32 [5, n_layers]
        struct ggml_tensor  * rope_tables, // F32 [2*Fc] (Fc=rotary_dim/2; empty if 0)
        const ggml_xkv_reconstruct_params * params);

// dst layout: F32 [dim_k+dim_v, n_sel]; rows [0,dim_k) = RoPE'd K,
// rows [dim_k,dim_k+dim_v) = canonical V.

// ---- validation ----
// Lightweight shape/type/params check (no tensor contents; usable in backend
// support hooks). Shared gate allows mixed Turbo/canonical (CPU canonical
// decode); the Vulkan hook narrows with an explicit mixed reject.
// Returns true if dispatchable, false with err message.
GGML_API bool ggml_xkv_reconstruct_supports(
        const struct ggml_tensor * a_k,
        const struct ggml_tensor * b_k,
        const struct ggml_tensor * a_v,
        const struct ggml_tensor * b_v,
        const struct ggml_tensor * refs,
        const struct ggml_tensor * positions,
        const struct ggml_tensor * group_meta,
        const struct ggml_tensor * layer_meta,
        const struct ggml_tensor * rope_tables,
        const struct ggml_tensor * dst,
        const ggml_xkv_reconstruct_params * params,
        char * err, size_t err_size);

// Full validation incl. per-group rank/feature slices, bounds, padded rows,
// fingerprints. Host pointers to refs/positions/group_meta contents required
// (CPU oracle + tests). Vulkan dispatch validates shapes/fingerprints on host
// and bounds in-shader (NaN poison, no host mirror of code streams).
GGML_API bool ggml_xkv_reconstruct_validate_full(
        const struct ggml_tensor * a_k,
        const struct ggml_tensor * b_k,
        const struct ggml_tensor * a_v,
        const struct ggml_tensor * b_v,
        const int32_t * refs_data,      // [4*n_sel] or NULL to skip content check
        const int32_t * positions_data, // [n_sel] or NULL
        const int32_t * group_meta_data,// [8*n_groups] or NULL
        const int32_t * layer_meta_data,// [5*n_layers] or NULL to skip content check
        int64_t n_layers,               // layer_meta rows (0 to skip)
        const float * rope_data,        // [2*Fc] omega/mag, or NULL to skip table check
        int64_t rope_nelements,         // 2*Fc (0 when rotary_dim==0)
        const ggml_xkv_reconstruct_params * params,
        char * err, size_t err_size);

// ---- CPU oracle (encoded streams -> dense selected K/V) ----
// All code-stream bytes + refs/positions/group_meta are host-readable here.
// Returns true on success, false with err on invalid/unsupported (fail-closed).
// Used by CPU backend forward and by backend tests as the encoded oracle.
GGML_API bool ggml_xkv_reconstruct_oracle(
        const void * a_k_data, enum ggml_type a_k_type, int64_t a_k_ne0, int64_t a_k_ne1,
        size_t a_k_nb0, size_t a_k_nb1,
        const void * b_k_data, enum ggml_type b_k_type, int64_t b_k_ne0, int64_t b_k_ne1,
        size_t b_k_nb0, size_t b_k_nb1,
        const void * a_v_data, enum ggml_type a_v_type, int64_t a_v_ne0, int64_t a_v_ne1,
        size_t a_v_nb0, size_t a_v_nb1,
        const void * b_v_data, enum ggml_type b_v_type, int64_t b_v_ne0, int64_t b_v_ne1,
        size_t b_v_nb0, size_t b_v_nb1,
        const int32_t * refs_data, const int32_t * positions_data, const int32_t * group_meta_data,
        const int32_t * layer_meta_data, int64_t n_layers,
        const float * rope_data, int64_t rope_nelements,
        const ggml_xkv_reconstruct_params * params,
        float * dst_data, int64_t dst_ne0, int64_t dst_ne1, size_t dst_nb0, size_t dst_nb1,
        char * err, size_t err_size);

// Bounded-scratch core for production compute (no heap): scratch must hold
// at least (2*max(prk,prv) + dim_k + dim_v) floats; scratch_floats carries
// its length. Reference-only ggml_xkv_reconstruct_oracle() heap-allocates
// internally and calls this core; backend forwards must call the core with
// graph workspace (params->wdata).
GGML_API bool ggml_xkv_reconstruct_core(
        const void * a_k_data, enum ggml_type a_k_type, int64_t a_k_ne0, int64_t a_k_ne1,
        size_t a_k_nb0, size_t a_k_nb1,
        const void * b_k_data, enum ggml_type b_k_type, int64_t b_k_ne0, int64_t b_k_ne1,
        size_t b_k_nb0, size_t b_k_nb1,
        const void * a_v_data, enum ggml_type a_v_type, int64_t a_v_ne0, int64_t a_v_ne1,
        size_t a_v_nb0, size_t a_v_nb1,
        const void * b_v_data, enum ggml_type b_v_type, int64_t b_v_ne0, int64_t b_v_ne1,
        size_t b_v_nb0, size_t b_v_nb1,
        const int32_t * refs_data, const int32_t * positions_data, const int32_t * group_meta_data,
        const int32_t * layer_meta_data, int64_t n_layers,
        const float * rope_data, int64_t rope_nelements,
        const ggml_xkv_reconstruct_params * params,
        float * dst_data, int64_t dst_ne0, int64_t dst_ne1, size_t dst_nb0, size_t dst_nb1,
        float * scratch, size_t scratch_floats,
        char * err, size_t err_size);

// Minimum scratch floats for the core (fail-closed on overflow).
GGML_API bool ggml_xkv_core_scratch_floats(const ggml_xkv_reconstruct_params * params,
        int64_t prk, int64_t prv, size_t * out_floats, char * err, size_t err_size);

// ---- landmark score / deterministic top-k / refinement (CPU) ----
// Phase-aware: landmarks are canonical pre-RoPE quantized vectors [padded_dim,
// n_frag]; query q is ALREADY post/effective-position RoPE and is used as-is.
// Only the canonical landmark is phased (via rope omega/mag tables) to its
// storage position before the dot. Deterministic top-k: score desc,
// index asc tie-break (matches ggml_top_k determinism requirement).
// top_k==0 or top_k>n_frag or unsupported landmark type => fail (never select-all).
GGML_API bool ggml_xkv_landmark_score(
        const float * q, uint32_t dim,
        const void * land_data, enum ggml_type land_type, int64_t land_ne0, int64_t land_ne1,
        size_t land_nb0, size_t land_nb1,
        const int32_t * frag_positions,
        const float * rope_omega, const float * rope_mag, uint32_t rope_fc,
        const ggml_xkv_landmark_config * cfg,
        float * scores_out, // [n_frag]
        char * err, size_t err_size);

GGML_API bool ggml_xkv_landmark_topk(
        const float * scores, uint32_t n_frag,
        const ggml_xkv_landmark_config * cfg,
        uint32_t * indices_out, // [top_k]
        float * top_scores_out, // [top_k] or NULL
        char * err, size_t err_size);

// Bounded boundary refinement: expand each selected fragment to include up to
// `refine_max_rows` neighboring selected-row candidates, scored at full
// precision via the reconstruct oracle path is the caller's responsibility;
// this helper only validates caps and expands indices deterministically.
// refine_max_rows==0 => output equals input. Cap hit is reported, never silent.
GGML_API bool ggml_xkv_landmark_refine(
        const uint32_t * selected, uint32_t n_selected,
        uint32_t n_rows_total, uint32_t n_frag, uint32_t frag_size,
        const ggml_xkv_landmark_config * cfg,
        uint32_t * refined_out, uint32_t * n_refined_out, uint32_t refined_cap,
        bool * cap_hit_out,
        char * err, size_t err_size);

// ---- code-stream upload / residency contract (narrow) ----
// Store-side encoded_matrix owns host vector bytes. Graph-op tensor inputs
// must be backend-resident with no duplicate full host mirror:
//  - reference mode may retain host bytes (host-resident tensors);
//  - Vulkan tq profile must upload EXACT bytes (ggml_row_size(type,ne0)*ne1,
//    no extra padding) into owned backend tensors ONCE per immutable segment
//    bundle, verify, then release the host vector. Graphs reuse the tensors
//    across dispatches: never reupload per decode. State save may perform an
//    explicit synchronized readback only on request.
typedef enum ggml_xkv_residency {
    GGML_XKV_RES_REFERENCE_HOST = 0, // retain host bytes
    GGML_XKV_RES_DEVICE_OWNED   = 1, // upload exact bytes, release host
} ggml_xkv_residency;

// Exact device bytes for a code stream (== ggml_row_size(type, padded_ne0) * n_rows).
// Returns 0 with err on invalid type/shape (fail-closed).
GGML_API size_t ggml_xkv_exact_bytes(enum ggml_type type, int64_t padded_ne0, int64_t n_rows,
                                      char * err, size_t err_size);

// Capability: whether a codec supports the requested residency on this build.
// Reference host supports all ggml_xkv_codec_supported() types. Device-owned
// supports F32/F16/Q8_0/Turbo2/3/4 (the reconstruct kernels); anything else
// is explicitly unsupported, never silently mirrored.
GGML_API bool ggml_xkv_residency_supported(enum ggml_type type, ggml_xkv_residency res);

// ---- dual-source indexed attention (Stage 2, one op per KV head) ----
//
// P0 ARCHITECTURE (XKV-SR §7.2): production NEVER materializes a dense cold
// union (DENSE cold can be 262K rows). Production chains TILED dispatches:
// each tile reconstructs only a workspace-bounded tile (existing
// ggml_xkv_reconstruct with tile-sized n_sel) and this op folds the tile
// into a persistent (m, l, o) carry — one global online softmax across all
// tiles. Hot rows participate via tile-0 entries in the same accumulator;
// sink applies once (first tile). A full selected-union K/V tensor exists
// only for the CPU oracle / small tests, never production DENSE/Vulkan
// (the Vulkan gate rejects carry-less unions beyond the tile bound below).
//
// Canonical domain first: hot Turbo K/V decode to canonical (inverse codec
// rotation), cold K/V are canonical reconstruct outputs, Q is canonical
// post-RoPE; V accumulates canonical. No WHT-shortcut assumptions; any future
// pair-domain fast path requires proven compatibility, never assumed.
//
// Single FP32 online softmax over the union of bounded hot rows and
// reconstructed cold rows for each (query, GQA head). Hot rows are read at
// exact physical (row, stream, kv_head); cold rows are Stage-1 reconstruct
// outputs (one tile, or a small test union). All K/V/Q inputs below are
// canonical; hot Turbo rows are inverse-rotated on decode inside the op.
//
// Entries (I32): entry_stride==4 -> [source,row,group,valid] with
//   source = GGML_XKV_ATTN_SOURCE_HOT(1)/COLD(2), row = hot physical row or
//   cold union row, group = index into q ne[2], valid==0 masks the entry;
// entry_stride==2 -> [key,group] with key<hot_rows selecting hot row=key,
//   otherwise cold row=key-hot_rows (key<0 is malformed).
// Offsets (I32[n_queries+1]): entries [offsets[q],offsets[q+1]) belong to
// query q; capacity padding at/after offsets[n_queries] is strictly ignored.
// Per-query duplicate (source,row) pairs accumulate deterministically in
// entry order unless GGML_XKV_ATTN_FLAG_REJECT_DUPES is set (then malformed).
// Sink (optional F32[GQA]) is applied exactly once per (query, GQA head)
// (first tile / single-shot only).
// All-masked/empty queries write zero (never NaN). Malformed offsets,
// source, index, or group write the status code, NaN-poison dst, and fail.
#define GGML_XKV_ATTN_VERSION 1
#define GGML_XKV_ATTN_SOURCE_HOT  1
#define GGML_XKV_ATTN_SOURCE_COLD 2
#define GGML_XKV_ATTN_ENTRY_STRIDE 4
#define GGML_XKV_ATTN_FLAG_REJECT_DUPES 1u
#define GGML_XKV_ATTN_FLAG_FIRST_TILE 2u
#define GGML_XKV_ATTN_FLAG_FINAL_TILE 4u
// Carry-less dense unions beyond this many cold rows are rejected on Vulkan
// (production must chain workspace-bounded tiles with a carry instead).
#define GGML_XKV_ATTN_DENSE_COLD_MAX 1024u

// POD op params (64 bytes == GGML_MAX_OP_PARAMS). No pointers.
typedef struct ggml_xkv_attention_params {
    uint32_t version;       // must be GGML_XKV_ATTN_VERSION
    float    scale;         // 1/sqrt(Dk); divided by softcap internally iff softcap>0
    float    logit_softcap; // 0 = none, else softcap*tanh(score/softcap)
    uint32_t n_queries;     // queries; dst ne[2]
    uint32_t n_groups;      // DDVR groups; q ne[2]
    uint32_t gqa_ratio;     // Q heads per KV head; q ne[1], dst ne[1]
    uint32_t dim_k;         // K width; q ne[0], k_hot ne[0], k_cold ne[0]
    uint32_t dim_v;         // V width; v_hot ne[0], v_cold ne[0], dst ne[0]
    uint32_t hot_rows;      // hot physical rows per stream
    uint32_t n_cold;        // cold union rows (k_cold/v_cold ne[1])
    uint32_t n_entries;     // entries ne[1] (capacity; offsets bound validity)
    uint32_t kv_head;       // hot KV head index into k_hot/v_hot ne[1]
    uint32_t stream;        // hot stream index into k_hot/v_hot ne[3]
    uint32_t flags;         // GGML_XKV_ATTN_FLAG_*; FIRST iff carry is null
    uint32_t reserved0;     // zero
    uint32_t reserved1;     // zero
} ggml_xkv_attention_params;

// Graph builder (no host pointers in op_params). The accumulator is an
// explicit DAG value, never a writable input side effect. `carry` is null for
// FIRST_TILE and is the preceding tile's dst otherwise. Status I32[1] is a
// diagnostic written by the op (0 ok, else first error code); sinks are read
// only by FIRST_TILE and may be NULL.
GGML_API struct ggml_tensor * ggml_xkv_attention(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k_hot,
        struct ggml_tensor  * v_hot,
        struct ggml_tensor  * k_cold,
        struct ggml_tensor  * v_cold,
        struct ggml_tensor  * entries,
        struct ggml_tensor  * offsets,
        struct ggml_tensor  * sinks,
        struct ggml_tensor  * status,
        struct ggml_tensor  * carry,     // previous F32 [Dv+2,GQA,NQ] or NULL on FIRST
        const ggml_xkv_attention_params * params);
// dst: F32 [Dv+2,GQA,n_queries], layout [O(0:Dv), m, l]. Intermediate
// tiles store unnormalized O; FINAL_TILE stores O/l (or zero for all-mask).
// The caller takes a strided view of the first Dv rows after the final tile.

// Lightweight shape/type/params check (no tensor contents; usable in backend
// support hooks). Returns true if dispatchable, false with err message.
GGML_API bool ggml_xkv_attention_supports(
        const struct ggml_tensor * q,
        const struct ggml_tensor * k_hot,
        const struct ggml_tensor * v_hot,
        const struct ggml_tensor * k_cold,
        const struct ggml_tensor * v_cold,
        const struct ggml_tensor * entries,
        const struct ggml_tensor * offsets,
        const struct ggml_tensor * sinks,
        const struct ggml_tensor * status,
        const struct ggml_tensor * carry,
        const struct ggml_tensor * dst,
        const ggml_xkv_attention_params * params,
        char * err, size_t err_size);

// Independent CPU oracle (raw pointers/strides; used by the CPU backend
// forward and by backend tests). Fail-closed: on malformed input writes the
// status code (when status_data!=NULL), NaN-poisons the failing query row,
// returns false (later queries uncomputed: check status, never consume dst).
// The graph forward additionally NaN-poisons the whole dst on false.
// tmp must hold ggml_xkv_attn_tmp_floats(params) floats (may be NULL iff 0).
GGML_API bool ggml_xkv_attention_oracle(
        const float * q_data, int64_t q_ne0, int64_t q_ne1, int64_t q_ne2,
        size_t q_nb0, size_t q_nb1, size_t q_nb2,
        const void * k_hot_data, enum ggml_type k_hot_type,
        int64_t kh_ne0, int64_t kh_ne1, int64_t kh_ne2, int64_t kh_ne3,
        size_t kh_nb0, size_t kh_nb1, size_t kh_nb2, size_t kh_nb3,
        const void * v_hot_data, enum ggml_type v_hot_type,
        int64_t vh_ne0, int64_t vh_ne1, int64_t vh_ne2, int64_t vh_ne3,
        size_t vh_nb0, size_t vh_nb1, size_t vh_nb2, size_t vh_nb3,
        const void * k_cold_data, enum ggml_type k_cold_type,
        int64_t kc_ne0, int64_t kc_ne1, size_t kc_nb0, size_t kc_nb1,
        const void * v_cold_data, enum ggml_type v_cold_type,
        int64_t vc_ne0, int64_t vc_ne1, size_t vc_nb0, size_t vc_nb1,
        const int32_t * entries_data, int64_t entries_ne0, int64_t entries_ne1,
        const int32_t * offsets_data, int64_t offsets_ne0,
        const float * sinks_data,
        int32_t * status_data,
        const float * carry_data, // previous [O(0:Dv),m,l], null only on FIRST_TILE
        const ggml_xkv_attention_params * params,
        float * dst_data, int64_t dst_ne0, int64_t dst_ne1, int64_t dst_ne2,
        size_t dst_nb0, size_t dst_nb1, size_t dst_nb2,
        float * tmp, size_t tmp_floats,
        char * err, size_t err_size);

// Minimum tmp floats for the oracle (fail-closed on overflow).
GGML_API bool ggml_xkv_attn_tmp_floats(const ggml_xkv_attention_params * params,
        size_t * out_floats, char * err, size_t err_size);

#ifdef __cplusplus
}
#endif
