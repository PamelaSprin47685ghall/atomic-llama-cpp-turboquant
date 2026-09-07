#ifndef FLASHPREFILL_INTERFACE_H
#define FLASHPREFILL_INTERFACE_H
// FlashPrefill V2 Vulkan interface (owned by VulkanShaders).
// Joint contract with VulkanDispatch (bindings/push) + WireReference
// (frozen ggml/include/ggml-flashprefill.h v1 offsets) + OpRegistration
// (GGML_OP 104/105/106, shader names flashprefill_pool/select/attn).
//
// C/GLSL-safe: C++ sees uint32_t structs + static_asserts; GLSL sees
// uint/float macros + field macros with identical order/size. Push blocks
// are defined via FP_*_PUSH_FIELDS so host and device cannot drift.
// K/V stay at bindings 1/2 to reuse flash_attn_dequant.glsl; Q=0, pool=3,
// plan=4 (unused in POOL/SELECT pipelines), meta=5, sinks=6, dst=7.
// SELECT dispatch is two-phased inside one op (VulkanDispatch):
//   phase 0 = init (parallel slice zeroing, header only by workgroup 0),
//   barrier (ggml_vk_sync_buffers, no host sync),
//   phase 1 = select (only atomicMax error / atomicAdd counters, never clears).
// ATTN only reads plan error, never writes it. Workgroup barrier is never
// global; no cross-workgroup sync assumed. No subgroup-size assumption.

#ifdef __cplusplus
#include <cstdint>
#define FP_UINT uint32_t
#define FP_FLOAT float
#define FP_INT int32_t
#else
#define FP_UINT uint
#define FP_FLOAT float
#define FP_INT int
#endif

// ---- frozen wire magics/version/words (mirror ggml-flashprefill.h) ----
#define FP_MAGIC_META 0x4650314D
#define FP_MAGIC_PLAN 0x46503150
#define FP_VERSION 1
#define FP_META_HDR_WORDS 32
#define FP_FRAG_WORDS 8
#define FP_ROW_WORDS 8
#define FP_USE_WORDS 8
#define FP_PLAN_HDR_WORDS 24

// ---- op ids (mirror GGML_OP 104/105/106 via OpRegistration) ----
#define FP_OP_POOL 0
#define FP_OP_SELECT 1
#define FP_OP_ATTN 2

// ---- role ids ----
#define FP_ROLE_EXACT 0
#define FP_ROLE_PROXY 1

// ---- flags ----
#define FP_USE_FLAG_MANDATORY 1
#define FP_FRAG_FLAG_BOUNDARY_PARTIAL 1
#define FP_ROW_FLAG_DENSE_FORCE 1

// ---- bindings (joint with VulkanDispatch) ----
#define FP_BIND_Q 0
#define FP_BIND_K 1
#define FP_BIND_V 2
#define FP_BIND_POOL 3
#define FP_BIND_PLAN 4
#define FP_BIND_META 5
#define FP_BIND_SINKS 6
#define FP_BIND_DST 7

// ---- spec constant ids (FaType/BlockBytes match FA ids 12..15 for host reuse) ----
#define FP_SPEC_WGSIZE 0
#define FP_SPEC_FATYPE_K 12
#define FP_SPEC_FATYPE_V 13
#define FP_SPEC_FBLOCKB_K 14
#define FP_SPEC_FBLOCKB_V 15

// ---- FaType values (mirror enum ggml_type, keep in sync with ggml.h) ----
#define FP_FA_TYPE_F32 0u
#define FP_FA_TYPE_F16 1u
#define FP_FA_TYPE_Q4_0 2u
#define FP_FA_TYPE_Q4_1 3u
#define FP_FA_TYPE_Q5_0 6u
#define FP_FA_TYPE_Q5_1 7u
#define FP_FA_TYPE_Q8_0 8u
#define FP_FA_TYPE_IQ4_NL 20u
#define FP_FA_TYPE_BF16 30u
#define FP_FA_TYPE_TURBO2_0 42u
#define FP_FA_TYPE_TURBO3_0 43u
#define FP_FA_TYPE_TURBO4_0 44u

// ---- metadata header word indices ----
#define FP_MH_MAGIC 0
#define FP_MH_VERSION 1
#define FP_MH_HDR_WORDS 2
#define FP_MH_N_FRAG 3
#define FP_MH_F_CAP 4
#define FP_MH_N_ROW 5
#define FP_MH_R_CAP 6
#define FP_MH_N_USE 7
#define FP_MH_U_CAP 8
#define FP_MH_N_CELL 9
#define FP_MH_C_CAP 10
#define FP_MH_FRAG_OFF 11
#define FP_MH_ROW_OFF 12
#define FP_MH_USE_OFF 13
#define FP_MH_CELL_OFF 14
#define FP_MH_FRAG_WORDS 15
#define FP_MH_ROW_WORDS 16
#define FP_MH_USE_WORDS 17
#define FP_MH_DK 18
#define FP_MH_DV 19
#define FP_MH_N_HKV 20
#define FP_MH_TOTAL 21
#define FP_MH_N_GROUPS 22
#define FP_MH_N_QHEADS 23

// ---- frag word indices ----
#define FP_FR_CELL_OFF 0
#define FP_FR_CELL_CNT 1
#define FP_FR_LBLOCK 2
#define FP_FR_DOMAIN 3
#define FP_FR_FLAGS 4

// ---- row word indices (frozen: source_query,kv_head,logical_pos,tile,pbegin,pend,flags,q_head) ----
#define FP_RW_SRC_Q 0
#define FP_RW_KV_HEAD 1
#define FP_RW_LOG_POS 2
#define FP_RW_TILE 3
#define FP_RW_PBEGIN 4
#define FP_RW_PEND 5
#define FP_RW_FLAGS 6
#define FP_RW_Q_HEAD 7

// ---- use word indices (frozen: frag,tile,kvhead,q_group,suboff,count,flags,src_q) ----
#define FP_UW_FRAG 0
#define FP_UW_TILE 1
#define FP_UW_KV_HEAD 2
#define FP_UW_Q_GROUP 3
#define FP_UW_SUB_OFF 4
#define FP_UW_SUB_CNT 5
#define FP_UW_FLAGS 6
#define FP_UW_SRC_Q 7

// ---- plan header word indices (24 words) ----
#define FP_PH_MAGIC 0
#define FP_PH_VERSION 1
#define FP_PH_HDR_WORDS 2
#define FP_PH_N_TILES 3
#define FP_PH_N_HEADS 4
#define FP_PH_MAX_SEL 5
#define FP_PH_EXACT_OFF 6
#define FP_PH_PROXY_OFF 7
#define FP_PH_COUNTS_OFF 8
#define FP_PH_TOTAL 9
#define FP_PH_ERROR 10
#define FP_PH_SPARSE_ROWS 11
#define FP_PH_DENSE_ROWS 12
#define FP_PH_SELECTED 13
#define FP_PH_CORRECTED 14
#define FP_PH_REQ_EXACT_ALL 15
#define FP_PH_VIS_LO 16
#define FP_PH_VIS_HI 17
#define FP_PH_EXACT_TOK_LO 18
#define FP_PH_EXACT_TOK_HI 19

// ---- error codes (mirror ggml-flashprefill.h) ----
#define FP_OK 0
#define FP_ERR_BAD_ARG 1
#define FP_ERR_OVERFLOW 2
#define FP_ERR_BAD_MAGIC 3
#define FP_ERR_BAD_VERSION 4
#define FP_ERR_BAD_LAYOUT 5
#define FP_ERR_CAP_EXCEEDED 6
#define FP_ERR_BAD_RANGE 7
#define FP_ERR_BAD_FLAG 8
#define FP_ERR_DUP_KEY 9
#define FP_ERR_DUP_TOKEN 10
#define FP_ERR_ORPHAN_USE 11
#define FP_ERR_PARTIAL_AS_PROXY 12
#define FP_ERR_PROXY_NOT_FULL 13
#define FP_ERR_MANDATORY_AS_PROXY 14
#define FP_ERR_DENSE_ROW_PROXY 15
#define FP_ERR_BAD_PLAN_COVERAGE 16
#define FP_ERR_PLAN_ROLE_MISMATCH 17
#define FP_ERR_BAD_CONFIG 18
#define FP_ERR_OVER_CAPACITY 19
#define FP_ERR_BAD_INPUT 20

// ---- selection legality invariant (Main + WireReference, frozen single rule) ----
// A partial-boundary fragment's whole-fragment mean may include
// future/invisible tokens for some tile rows. Single rule: a candidate not
// full across the tile (ANY use carrying the fragment in the pair is a
// partial subset, cand_full==0) is mandatory exact AND its entire column is
// excluded from tile-max/energy reads entirely (never read). Fully legal
// candidates (cand_full==1) score valid row pairs only (present,
// full-fragment uses). This conservative partial-boundary extension
// prevents any pollutant future/hidden input from altering legal
// plan/output and matches CPU. Unscored (excluded) fragments are exact;
// all-excluded pairs are all-exact, finite, zero sparse work. The CPU
// reference encodes the same exclusion as its pair_valid mask; GPU
// enforces the identical exclusion on-device (no partial-mean reads in the
// energy path). One policy everywhere; no dual policies.

// ---- emission-order contracts (frozen: wire order-agnostic) ----
// CacheFragments guarantees uses query-major with fragment ids strictly
// ascending within each query (header comment landed; tiles/kv_head do not
// exist at planning time). The (tile, kv_head) contiguous grouping that
// SELECT binary-searches is a pack-side sort before select dispatch
// (GraphIntegration owns packed-Q blocking + head fan-out). SELECT
// validates boundaries/in-range and raises BAD_LAYOUT on violation: loud
// failure, never silent subsets. A bounded linear-scan fallback (single
// global pass into exact-slice workspace, P<=max_sel) may replace the
// failure if the packer sort ever slips; until then fail-loud stands.

// ---- split/merge protocol (Guide8.1/Phase4, split1 + bounded split) ----
// ATTN phase (push split_phase): 0 = final split1 (writes O to dst, sink
// always carried when present); 1 = partial (writes M/L/O to scratch,
// sink carried ONLY when split_id == sink_split, unique assignment).
// Scratch layouts, row-major, fit-counted (notify Dispatch/fit):
//   M/L: [n_rows x n_splits] floats at bindings 8/9.
//   O:   [n_rows x n_splits x Dv] floats at binding 10.
// Scratch floats total = n_rows * n_splits * (2 + Dv).
// MERGE reads S partials per row and applies reference mlo_merge
// equations (empty identity l==0, sink mass already unique in exactly one
// partial, never re-added). CPU mlo_* is the math reference only; Vulkan
// split executes on GPU via ATTN-partial + MERGE.

// ---- op_params word indices (16 words, word8 = mean_correction) ----
#define FP_OPP_OP 0
#define FP_OPP_ALPHA 1
#define FP_OPP_SCALE 2
#define FP_OPP_SOFTCAP 3
#define FP_OPP_EXACT_ALL 4
#define FP_OPP_DK 5
#define FP_OPP_DV 6
#define FP_OPP_VERSION 7
#define FP_OPP_MEAN_CORR 8

// ---- push layouts: caps/bounds only (no actual counts).
// Host fills these from tensor shapes (ne/nb), allocation sizes, and
// op_params scalars. Actual counts (n_frag/n_row/n_use/n_cell/n_tiles,
// max_sel, n_groups/n_q_heads) live only in device headers at dispatch
// time; shaders read them from device and validate actual<=cap on-device.
// This preserves the no-NNZ-sync / no-readback / no-host-KV-readback
// constraints. Dispatch grids use caps (f_cap/r_cap x heads) with shader
// guards (f>=n_frag / t>=T / r>=n_row return). ----
#define FP_POOL_PUSH_FIELDS \
    FP_UINT dk; \
    FP_UINT dv; \
    FP_UINT h_cap; \
    FP_UINT f_cap; \
    FP_UINT meta_words; \
    FP_UINT n_kv; \
    FP_UINT k_nb1; \
    FP_UINT k_nb2; \
    FP_UINT v_nb1; \
    FP_UINT v_nb2; \
    FP_UINT version; \
    FP_UINT reserved0; \
    FP_UINT reserved1; \
    FP_UINT reserved2; \
    FP_UINT reserved3; \
    FP_UINT reserved4;

#define FP_SELECT_PUSH_FIELDS \
    FP_UINT phase; \
    FP_UINT dk; \
    FP_UINT dv_cap; \
    FP_UINT f_cap; \
    FP_UINT r_cap; \
    FP_UINT u_cap; \
    FP_UINT c_cap; \
    FP_UINT meta_words; \
    FP_UINT plan_words; \
    FP_UINT q_nb1; \
    FP_UINT q_nb2; \
    FP_UINT q_groups_cap; \
    FP_UINT q_heads_cap; \
    FP_UINT hkv_cap; \
    FP_UINT exact_all; \
    FP_UINT version; \
    FP_FLOAT alpha; \
    FP_FLOAT scale; \
    FP_FLOAT softcap; \
    FP_UINT reserved0; \
    FP_UINT reserved1; \
    FP_UINT reserved2; \
    FP_UINT reserved3; \
    FP_UINT reserved4;

#define FP_ATTN_PUSH_FIELDS \
    FP_UINT dk; \
    FP_UINT dv; \
    FP_UINT meta_words; \
    FP_UINT plan_words; \
    FP_UINT n_kv; \
    FP_UINT f_cap; \
    FP_UINT r_cap; \
    FP_UINT u_cap; \
    FP_UINT c_cap; \
    FP_UINT n_output_cap; \
    FP_UINT q_nb1; \
    FP_UINT q_nb2; \
    FP_UINT k_nb1; \
    FP_UINT k_nb2; \
    FP_UINT v_nb1; \
    FP_UINT v_nb2; \
    FP_UINT q_groups_cap; \
    FP_UINT q_heads_cap; \
    FP_UINT hkv_cap; \
    FP_UINT sinks_present; \
    FP_UINT version; \
    FP_UINT mean_correction; \
    FP_FLOAT scale; \
    FP_FLOAT softcap; \
    FP_UINT split_phase; \
    FP_UINT split_id; \
    FP_UINT n_splits; \
    FP_UINT sink_split; \
    FP_UINT reserved4; \
    FP_UINT reserved5; \
    FP_UINT reserved6; \
    FP_UINT reserved7;

#define FP_MERGE_PUSH_FIELDS \
    FP_UINT dv; \
    FP_UINT r_cap; \
    FP_UINT meta_words; \
    FP_UINT plan_words; \
    FP_UINT n_splits_cap; \
    FP_UINT version; \
    FP_UINT reserved0; \
    FP_UINT reserved1; \
    FP_UINT reserved2; \
    FP_UINT reserved3; \
    FP_UINT reserved4; \
    FP_UINT reserved5; \
    FP_UINT reserved6; \
    FP_UINT reserved7; \
    FP_UINT reserved8; \
    FP_UINT reserved9;

#ifdef __cplusplus
struct FpPoolPush { FP_POOL_PUSH_FIELDS };
struct FpSelectPush { FP_SELECT_PUSH_FIELDS };
struct FpAttnPush { FP_ATTN_PUSH_FIELDS };
struct FpMergePush { FP_MERGE_PUSH_FIELDS };
static_assert(sizeof(FpPoolPush) == 64, "pool push 64B");
static_assert(sizeof(FpSelectPush) == 96, "select push 96B");
static_assert(sizeof(FpAttnPush) == 128, "attn push 128B");
static_assert(sizeof(FpMergePush) == 64, "merge push 64B");
#endif

#ifndef __cplusplus
// ---- GLSL-only helpers (types.glsl must be included before this header) ----
#define FP_CEIL_DIV(a, b) (((a) + (b) - 1u) / (b))

// Canonical FA_TYPE_* names expected by flash_attn_dequant.glsl (which
// switches on FaTypeK/FaTypeV with these enumerants mirroring ggml_type).
// Single shared definition: FP_FA_TYPE_* own the values, FA_TYPE_* alias.
#define FA_TYPE_F32 FP_FA_TYPE_F32
#define FA_TYPE_F16 FP_FA_TYPE_F16
#define FA_TYPE_Q4_0 FP_FA_TYPE_Q4_0
#define FA_TYPE_Q4_1 FP_FA_TYPE_Q4_1
#define FA_TYPE_Q5_0 FP_FA_TYPE_Q5_0
#define FA_TYPE_Q5_1 FP_FA_TYPE_Q5_1
#define FA_TYPE_Q8_0 FP_FA_TYPE_Q8_0
#define FA_TYPE_IQ4_NL FP_FA_TYPE_IQ4_NL
#define FA_TYPE_BF16 FP_FA_TYPE_BF16
#define FA_TYPE_TURBO2_0 FP_FA_TYPE_TURBO2_0
#define FA_TYPE_TURBO3_0 FP_FA_TYPE_TURBO3_0
#define FA_TYPE_TURBO4_0 FP_FA_TYPE_TURBO4_0

// GLSL 450 (vulkan1.2 target) provides isnan/isinf but no isfinite
// overload: single shared finite check for select/attn/merge.
bool fp_isfinite(float x) { return !isnan(x) && !isinf(x); }

uint fp_block_elems(uint ty) {
    if (ty == FP_FA_TYPE_F32) return 4u;
    if (ty == FP_FA_TYPE_F16) return 1u;
    if (ty == FP_FA_TYPE_Q4_0) return uint(QUANT_K_Q4_0);
    if (ty == FP_FA_TYPE_Q4_1) return uint(QUANT_K_Q4_1);
    if (ty == FP_FA_TYPE_Q5_0) return uint(QUANT_K_Q5_0);
    if (ty == FP_FA_TYPE_Q5_1) return uint(QUANT_K_Q5_1);
    if (ty == FP_FA_TYPE_Q8_0) return uint(QUANT_K_Q8_0);
    if (ty == FP_FA_TYPE_IQ4_NL) return uint(QUANT_K_IQ4_NL);
    if (ty == FP_FA_TYPE_BF16) return 1u;
    if (ty == FP_FA_TYPE_TURBO2_0) return uint(QUANT_K_TURBO2_0);
    if (ty == FP_FA_TYPE_TURBO3_0) return uint(QUANT_K_TURBO3_0);
    if (ty == FP_FA_TYPE_TURBO4_0) return uint(QUANT_K_TURBO4_0);
    return 1u;
}

bool fp_type_needs_shmem(uint ty) {
    return ty == FP_FA_TYPE_IQ4_NL;
}

float fp_softcap_apply(float x, float cap) {
    if (cap == 0.0f) return x;
    return tanh(x / cap) * cap;
}

// 64-bit counter note: SELECT accumulates visible/exact token incidences
// with two 32-bit atomicAdds (lo + carry to hi) per bounded per-pair sum.
// No helper here: GLSL disallows passing SSBOs as function parameters, so
// each shader performs the lo/carry/hi sequence inline on its plan buffer.
#endif // __cplusplus (GLSL section)

#endif // FLASHPREFILL_INTERFACE_H
