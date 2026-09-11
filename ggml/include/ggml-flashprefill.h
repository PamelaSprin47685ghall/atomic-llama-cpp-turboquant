#pragma once

// FlashPrefill V2 GGML wire schema + CPU-independent reference math, v1.
//
// Two independent contracts exist: the llama policy/role API in
// include/llama-flashprefill.h and this file. Neither includes the other;
// the scalar config fields match semantically (names, units, defaults).
// Frozen v1: do not renumber fields or offsets without coordination.
//
// Conventions: metadata/plan tensors are I32 1-D. Sizes and offsets are
// checked in int64_t and narrowed to int32 only after validation. Pool is
// F32 with ne = [Dk+Dv, Hkv, Fcap]: K means in rows [0,Dk), V means in
// rows [Dk,Dk+Dv). Dk may differ from Dv. Reference math uses the
// natural-log domain: proxy_logit = s_dot + ln(n); an exp2 port must use
// scaled_dot_log2 + log2(n) consistently across select, merge, and attend.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_FLASHPREFILL_API __declspec(dllexport) extern
#        else
#            define GGML_FLASHPREFILL_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_FLASHPREFILL_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_FLASHPREFILL_API extern
#endif

#define GGML_FLASHPREFILL_VERSION 1

#define GGML_FLASHPREFILL_METADATA_MAGIC ((int32_t)0x4650314D) // 'FP1M'
#define GGML_FLASHPREFILL_PLAN_MAGIC ((int32_t)0x46503150)     // 'FP1P'

#define GGML_FLASHPREFILL_ROLE_EXACT 0
#define GGML_FLASHPREFILL_ROLE_PROXY 1

#define GGML_FLASHPREFILL_OP_POOL 0
#define GGML_FLASHPREFILL_OP_SELECT 1
#define GGML_FLASHPREFILL_OP_ATTN 2

#define GGML_FLASHPREFILL_MODE_OFF 0
#define GGML_FLASHPREFILL_MODE_AUTO 1
#define GGML_FLASHPREFILL_MODE_REQUIRED 2

#define GGML_FLASHPREFILL_TAIL_SCOPE_CALL 0
#define GGML_FLASHPREFILL_TAIL_SCOPE_LOGICAL_PROMPT 1

#define GGML_FLASHPREFILL_HEADER_WORDS 32
#define GGML_FLASHPREFILL_FRAG_WORDS 8
#define GGML_FLASHPREFILL_ROW_WORDS 8
#define GGML_FLASHPREFILL_USE_WORDS 8
#define GGML_FLASHPREFILL_PLAN_HEADER_WORDS 24

#define GGML_FLASHPREFILL_DEFAULT_MODE GGML_FLASHPREFILL_MODE_OFF
#define GGML_FLASHPREFILL_DEFAULT_ALPHA 0.1f
#define GGML_FLASHPREFILL_DEFAULT_BLOCK_Q 128u
#define GGML_FLASHPREFILL_DEFAULT_BLOCK_K 128u
#define GGML_FLASHPREFILL_DEFAULT_SINK_BLOCKS 2u
#define GGML_FLASHPREFILL_DEFAULT_WINDOW_BLOCKS 4u
#define GGML_FLASHPREFILL_DEFAULT_DENSE_TAIL_TILES 8u
#define GGML_FLASHPREFILL_DEFAULT_MIN_KV 1024u
#define GGML_FLASHPREFILL_DEFAULT_FULL_ATTN_LAYERS 0u
#define GGML_FLASHPREFILL_DEFAULT_TAIL_SCOPE GGML_FLASHPREFILL_TAIL_SCOPE_LOGICAL_PROMPT

#define GGML_FLASHPREFILL_BLOCK_Q_MIN 1u
#define GGML_FLASHPREFILL_BLOCK_Q_MAX 256u
#define GGML_FLASHPREFILL_BLOCK_K_MIN 64u
#define GGML_FLASHPREFILL_BLOCK_K_MAX 1024u
#define GGML_FLASHPREFILL_BLOCK_K_STEP 64u
#define GGML_FLASHPREFILL_COUNT_WINDOW_MAX 1048576u
#define GGML_FLASHPREFILL_HEAD_DIM_MAX 65536

#define GGML_FLASHPREFILL_USE_FLAG_MANDATORY ((int32_t)(1 << 0))
#define GGML_FLASHPREFILL_FRAG_FLAG_BOUNDARY_PARTIAL ((int32_t)(1 << 0))
#define GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE ((int32_t)(1 << 0))

// Source slots for the later op-registration wave (GGML_MAX_SRC of 10 kept).
//   POOL:   src[0]=K, src[1]=V, src[2]=metadata(I32)          -> pool F32
//   SELECT: src[0]=Q, src[1]=pool(F32), src[2]=metadata(I32)  -> plan I32
//   ATTN:   src[0]=Q, src[1]=K, src[2]=V, src[3]=pool(F32),
//           src[4]=plan(I32), src[5]=metadata(I32),
//           src[6]=sinks (optional, NULL when absent)          -> O
// Q tensor: ne = [Dk, n_groups, Hq, 1] (existing RERoT convention; Gmax=1
// without RERoT; dims echo metadata n_groups/n_q_heads). The Q vector for
// head h under effective group g is element (d, g, h, 0). O tensor:
// ne = [Dv, Hq, n_output_queries, 1]; output for (query q, head h) is
// element (d, h, q, 0).
#define GGML_FLASHPREFILL_POOL_N_SRC 3
#define GGML_FLASHPREFILL_SELECT_N_SRC 3
#define GGML_FLASHPREFILL_ATTN_N_SRC 7
#define GGML_FLASHPREFILL_MAX_SRC_USED 7

// Errors: 0 ok, else one of these.
#define GGML_FLASHPREFILL_OK 0
#define GGML_FLASHPREFILL_ERR_BAD_ARG 1
#define GGML_FLASHPREFILL_ERR_OVERFLOW 2
#define GGML_FLASHPREFILL_ERR_BAD_MAGIC 3
#define GGML_FLASHPREFILL_ERR_BAD_VERSION 4
#define GGML_FLASHPREFILL_ERR_BAD_LAYOUT 5
#define GGML_FLASHPREFILL_ERR_CAP_EXCEEDED 6
#define GGML_FLASHPREFILL_ERR_BAD_RANGE 7
#define GGML_FLASHPREFILL_ERR_BAD_FLAG 8
#define GGML_FLASHPREFILL_ERR_DUP_KEY 9
#define GGML_FLASHPREFILL_ERR_DUP_TOKEN 10
#define GGML_FLASHPREFILL_ERR_ORPHAN_USE 11
#define GGML_FLASHPREFILL_ERR_PARTIAL_AS_PROXY 12
#define GGML_FLASHPREFILL_ERR_PROXY_NOT_FULL 13
#define GGML_FLASHPREFILL_ERR_MANDATORY_AS_PROXY 14
#define GGML_FLASHPREFILL_ERR_DENSE_ROW_PROXY 15
#define GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE 16
#define GGML_FLASHPREFILL_ERR_PLAN_ROLE_MISMATCH 17
#define GGML_FLASHPREFILL_ERR_BAD_CONFIG 18
#define GGML_FLASHPREFILL_ERR_OVER_CAPACITY 19
#define GGML_FLASHPREFILL_ERR_BAD_INPUT 20

GGML_FLASHPREFILL_API const char * ggml_flashprefill_strerror(int32_t err);

// Scalar policy (mirrors the llama-side policy header semantically).
struct ggml_flashprefill_config {
    uint32_t mode;             // 0=off, 1=auto, 2=required
    float    alpha;            // (0,1] energy threshold, not a sparsity ratio
    uint32_t block_q;          // packed Q rows per tile, 1..256, any value
    uint32_t block_k;          // logical K block, multiple of 64, <= 1024
    uint32_t sink_blocks;      // forced-exact prefix blocks
    uint32_t window_blocks;    // forced-exact local blocks (+partial boundary)
    uint32_t dense_tail_tiles; // dense Q tiles at the tail
    uint32_t min_kv;           // eligibility floor on resident-visible tokens
    uint32_t full_attn_layers; // first N eligible full-attention layers dense
    uint32_t tail_scope;       // 0=call, 1=logical-prompt
    bool     mean_correction;  // production requires true
    bool     exact_all;        // debug: new path, exact compute
};

#define GGML_FLASHPREFILL_FIELD_NONE 0
#define GGML_FLASHPREFILL_FIELD_MODE 1
#define GGML_FLASHPREFILL_FIELD_ALPHA 2
#define GGML_FLASHPREFILL_FIELD_BLOCK_Q 3
#define GGML_FLASHPREFILL_FIELD_BLOCK_K 4
#define GGML_FLASHPREFILL_FIELD_SINK_BLOCKS 5
#define GGML_FLASHPREFILL_FIELD_WINDOW_BLOCKS 6
#define GGML_FLASHPREFILL_FIELD_DENSE_TAIL_TILES 7
#define GGML_FLASHPREFILL_FIELD_TAIL_SCOPE 8

GGML_FLASHPREFILL_API struct ggml_flashprefill_config ggml_flashprefill_config_default(void);
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_config_validate(
        const struct ggml_flashprefill_config * cfg, int32_t * bad_field);

// Metadata tensor (I32, 1-D). Canonical v1 layout; offsets are cross-checked.
// header [0..31]:
//   [0] magic [1] version [2] header_words(32)
//   [3] n_frag [4] f_cap [5] n_row [6] r_cap
//   [7] n_use [8] u_cap [9] n_cell [10] c_cap
//   [11] frag_off(=32) [12] row_off(=32+8*f_cap)
//   [13] use_off(=row_off+8*r_cap) [14] cell_off(=use_off+8*u_cap)
//   [15] frag_words(8) [16] row_words(8) [17] use_words(8)
//   [18] dk [19] dv [20] n_kv_heads [21] total_words(=cell_off+c_cap)
//   [22] n_groups (Gmax, >=1) [23] n_q_heads (>=1) [24..31] reserved (0)
// row [8]: [0] source_query (output packed-row index) [1] kv_head
//   [2] logical_pos [3] tile_id [4] prompt_begin (incl) [5] prompt_end (excl)
//   [6] flags (bit0 DENSE_FORCE) [7] q_head
// use [8]: [0] frag_id [1] tile_id [2] kv_head [3] q_group (effective phase
//   for this query+fragment) [4] sub_off (absolute cell-table offset inside
//   the fragment range) [5] sub_count (>0, inside the fragment range)
//   [6] flags (bit0 MANDATORY: must classify exact) [7] source_query
// fragment [8]: [0] cell_off [1] cell_count [2] logical_block
//   (INT32_MIN if irregular) [3] domain [4] flags [5..7] reserved
// cell table [c_cap]: physical cell indices (>= 0).
// A use matches rows on (source_query, tile_id, kv_head); one use serves
// every subhead row sharing that triple. Key uniqueness: rows on
// (source_query, q_head); uses on (source_query, tile_id, kv_head, frag_id)
// (one effective phase per query+fragment). Validation: counts<=caps;
// sub ranges inside fragments; sub_count>0; key uniqueness; every use
// matches >=1 row; kv_head<n_kv_heads; q_head<n_q_heads; q_group<n_groups;
// prompt_begin<prompt_end; per matched (source_query, tile, kv_head) triple
// the union of sub cells is duplicate-free (no repeated physical token per
// row; E(r)/U(r) disjoint by construction); DENSE_FORCE rows match only
// MANDATORY uses. Residency/visibility/causality membership itself is the
// owner's claim; the validator checks its encoding.

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_words(
        int64_t f_cap, int64_t r_cap, int64_t u_cap, int64_t c_cap,
        int64_t * out_words);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_init(
        int32_t * meta, int64_t n_words,
        int64_t f_cap, int64_t r_cap, int64_t u_cap, int64_t c_cap,
        int32_t dk, int32_t dv, int32_t n_kv_heads,
        int32_t n_groups, int32_t n_q_heads);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_set_counts(
        int32_t * meta, int64_t n_words,
        int64_t n_frag, int64_t n_row, int64_t n_use, int64_t n_cell);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_set_frag(
        int32_t * meta, int64_t n_words, int64_t idx,
        int64_t cell_off, int64_t cell_count,
        int32_t logical_block, int32_t domain, int32_t flags);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_set_row(
        int32_t * meta, int64_t n_words, int64_t idx,
        int32_t source_query, int32_t kv_head, int32_t logical_pos,
        int32_t tile_id, int32_t prompt_begin, int32_t prompt_end,
        int32_t flags, int32_t q_head);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_set_use(
        int32_t * meta, int64_t n_words, int64_t idx,
        int32_t frag_id, int32_t tile_id, int32_t kv_head, int32_t q_group,
        int64_t sub_off, int64_t sub_count, int32_t flags, int32_t source_query);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_set_cell(
        int32_t * meta, int64_t n_words, int64_t idx, int32_t cell);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_n_tiles(
        const int32_t * meta, int64_t n_words, int64_t * out_n_tiles);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_metadata_validate(
        const int32_t * meta, int64_t n_words);

// Pool element (f,h,d): pool[(f*Hkv + h)*(Dk+Dv) + d], F32.
GGML_FLASHPREFILL_API int64_t ggml_flashprefill_pool_nelements(
        int32_t dk, int32_t dv, int64_t n_heads, int64_t f_cap);

// Plan tensor (I32, 1-D; SELECT output, ATTN input). Per-(tile,kv_head) the
// exact and proxy use-index lists are both explicit, so the classified
// complement is never inferred. Exact-all uses the same schema with empty
// proxy lists. Capacity is per (tile,head) pair (max_sel_pair), sized from
// the worst-case pair — never U_global multiplied over tiles.
// header [0..23]:
//   [0] magic [1] version [2] header_words(24)
//   [3] n_tiles [4] n_kv_heads [5] max_sel_pair
//   [6] exact_off(=24) [7] proxy_off(=24+T*H*max_sel_pair)
//   [8] counts_off(=proxy_off+T*H*max_sel_pair)
//   [9] total_words(=counts_off+2*T*H)
//   [10] error [11] sparse_rows [12] dense_rows
//   [13] selected_total (exact uses) [14] corrected_total (proxy uses)
//   [15] req_exact_all echo
//   [16..17] visible_tokens i64 lo/hi (sum of sub_count over ALL metadata uses)
//   [18..19] exact_tokens i64 lo/hi (sum of sub_count over exact-listed uses)
//   [20..23] reserved (0)
// exact/proxy tables [T*H*max_sel_pair]: use indices, -1 past per-pair counts.
// counts table [2*T*H]: per (tile,head) n_exact, n_proxy.
// Work classification: a packed row is sparse iff >= 1 matched metadata use
// is proxy-listed or missing from both plan lists; exact_all, all-exact,
// and no-matched-use rows are dense. Unlisted uses fail visible as sparse,
// never as fake-dense.
// Units: sparse/dense_rows count packed (source_query, q_head) execution
// rows (qheads included); visible/exact_tokens count each use-record once
// (subhead fan-out is not multiplied). Ratios must keep one unit:
// sparse_rows/(sparse_rows+dense_rows), or exact_tokens/visible_tokens.
// Error + counters ride the plan header (single-output op model: no side
// stats tensor). Entry order within each list is increasing use index.

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_words(
        int64_t n_tiles, int64_t n_heads, int64_t max_sel_pair, int64_t * out_words);

// Sizing invariant: max_sel_pair must be the MAXIMUM actual per-(tile,head)
// use count in the metadata, never the global n_use. This helper computes
// that maximum (0 when n_use==0). Owners size the plan with it, check the
// resulting worst-case allocation against budget, and resource-gate high
// fragmentation before dispatch — never silently truncate.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_max_sel_for_meta(
        const int32_t * meta, int64_t meta_words, int64_t * out_max);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_init(
        int32_t * plan, int64_t n_words,
        int64_t n_tiles, int64_t n_heads, int64_t max_sel_pair, int32_t req_exact_all);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_set_counts(
        int32_t * plan, int64_t n_words,
        int64_t tile, int64_t head, int64_t n_exact, int64_t n_proxy);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_set_entry(
        int32_t * plan, int64_t n_words,
        int64_t tile, int64_t head, int32_t role, int64_t slot, int64_t use_idx);

// Fills header words [11..14] and [16..19] from the actual tables and use
// records (checked 64-bit accumulation; never from capacities). The error
// word is left untouched.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_accumulate_stats(
        int32_t * plan, int64_t plan_words,
        const int32_t * meta, int64_t meta_words);

struct ggml_flashprefill_plan_stats {
    int32_t error;
    int32_t sparse_rows;
    int32_t dense_rows;
    int32_t selected_total;
    int32_t corrected_total;
    int32_t req_exact_all;
    int64_t visible_tokens;
    int64_t exact_tokens;
};

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_get_stats(
        const int32_t * plan, int64_t n_words,
        struct ggml_flashprefill_plan_stats * out);

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_validate(
        const int32_t * plan, int64_t n_words);

// Against metadata: per (tile,head) the plan's exact+proxy use sets must
// equal the metadata use set exactly once each; proxy entries must be
// full-fragment and non-mandatory; mandatory uses must be exact; triples
// touching a DENSE_FORCE row must be all-mandatory.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_plan_validate_against_metadata(
        const int32_t * plan, int64_t plan_words,
        const int32_t * meta, int64_t meta_words);

// Explicit op params: 16 int32 words (<= GGML_MAX_OP_PARAMS). BM/BN, tail
// scope, and sink/window/tail counts are owner policy already materialized
// into mandatory uses; params carry only live numerics.
struct ggml_flashprefill_op_params {
    int32_t op;             // POOL / SELECT / ATTN
    float   alpha;          // SELECT threshold, inclusive >=
    float   scale;          // kq_scale from the graph (SELECT + ATTN)
    float   softcap;        // 0 = none; else tanh(x/cap)*cap in score domain
    int32_t exact_all;      // SELECT: classify every use exact
    int32_t dk;
    int32_t dv;
    int32_t version;        // GGML_FLASHPREFILL_VERSION
    int32_t mean_correction;// ATTN: 1 apply proxy correction, 0 ablation
    int32_t reserved[7];
};

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_op_params_pack(
        const struct ggml_flashprefill_op_params * p, int32_t out16[16]);
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_op_params_unpack(
        const int32_t in16[16], struct ggml_flashprefill_op_params * p);

// Reference math on plain pointers (no ggml types, no allocation). Later CPU
// kernels decode cache bytes of any dtype into scratch and call these;
// ports must reproduce them within the agreed oracle tolerances. Internal
// accumulation is double, rounded once to float on store.

// kbar/vbar = means over idx[0..n); row strides in floats; idx >= 0.
// Requires n>=1, dk/dv>=1.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_ref_means(
        const float * K, int64_t stride_k,
        const float * V, int64_t stride_v,
        const int32_t * idx, int64_t n,
        int32_t dk, int32_t dv,
        float * kbar, float * vbar);

// Tile max-energy selector for one (tile, kv_head): energy aggregates per
// fragment across matched packed rows (not per use/phase).
// qpair[(r*n_cand + j)*stride_qp .. +dk] is the Q vector of packed row r
// under candidate j's effective phase (gathered by the caller from Q).
// cand_full[j] = 1 iff every matched use of that fragment is a full-fragment
// subset. Single energy rule: partial candidates (cand_full == 0) are
// mandatory-exact and excluded from energy entirely — their column is never
// read or scored, so a boundary mean holding future/invisible tokens cannot
// move M, Smax, or any keep decision. Fully legal candidates score only
// valid pairs: pair_valid[r*n_cand + j] = 1 iff row r may legally see
// candidate j's whole mean; invalid pairs are never read. Production SELECT
// (CPU/GPU) enforces the identical exclusion. keep[j]=(S[j] >= alpha*max S[j]),
// inclusive, over scored candidates. exact = unscored|kept|mandatory|partial
// |(exact_all ? all : {}); proxy = rest (hence always full, scored, kept,
// non-mandatory). Deterministic: increasing candidate order.
// Over-capacity returns OVER_CAPACITY, never truncates.
// Requires n_rows>=1, n_cand>=1, mask strictly 0/1, finite scored inputs,
// 0<alpha<=1. An all-zero mask is legal (no legal mean, e.g. all-mandatory
// partial rows): every candidate classifies exact with zero sparse work,
// capacity check still enforced. Sink-free scores: model sinks enter only
// the attention merge, not selection.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_ref_select(
        const float * qpair, int64_t stride_qp, int32_t n_rows,
        const float * kbar, int64_t stride_kb,
        const int32_t * cand_frag, const int32_t * cand_mandatory,
        const int32_t * cand_full, const int32_t * pair_valid, int32_t n_cand,
        int32_t dk, float scale, float softcap, float alpha, int32_t exact_all,
        int32_t cap_exact, int32_t cap_proxy,
        int32_t * out_exact, int32_t * n_exact_out,
        int32_t * out_proxy, int32_t * n_proxy_out);

GGML_FLASHPREFILL_API double ggml_flashprefill_softcap_apply(double x, double cap);

// Stable accumulator: m=max logit, l=sum exp(logit-m), o=sum exp*mass*value.
// Caller owns o (dv floats). Empty-split identity: m=-inf, l=0.
struct ggml_flashprefill_mlo_state {
    double  m;
    double  l;
    int32_t dv;
};

GGML_FLASHPREFILL_API int32_t ggml_flashprefill_mlo_init(
        struct ggml_flashprefill_mlo_state * s, float * o, int32_t dv);
// One term with multiplicity count>=1 at z=logit+ln(count): exact tokens
// use count=1, proxy blocks count=n_J (never 0).
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_mlo_add(
        struct ggml_flashprefill_mlo_state * s, float * o,
        double logit, const float * v, int64_t count);
// Model softmax sink term: exp mass without value contribution.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_mlo_add_sink(
        struct ggml_flashprefill_mlo_state * s, float * o, double sink_logit);
// Merge b into a (o_out may alias oa). Either side may be empty. Sink mass
// must have been added in exactly one input side, never both.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_mlo_merge(
        struct ggml_flashprefill_mlo_state * sa, const float * oa,
        const struct ggml_flashprefill_mlo_state * sb, const float * ob,
        struct ggml_flashprefill_mlo_state * s_out, float * o_out);
// out=o/l; empty (l==0) yields zeros, never NaN.
GGML_FLASHPREFILL_API int32_t ggml_flashprefill_mlo_finalize(
        const struct ggml_flashprefill_mlo_state * s, const float * o, float * out);

#ifdef __cplusplus
}
#endif
