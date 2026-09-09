#pragma once

// ggml-xkv-landmark-build.h — Bounded native landmark construction from FINAL
// encoded A_K/B_K streams (v1).
//
// Reconstructs canonical K chunk-wise (never whole-K), applies forward RoPE
// to EACH row at its own storage position, mean-pools each fixed chunk, and
// encodes each output row directly as Q8_0 or Turbo4_0. Mirrors the
// build_landmarks_bounded factory contract (src/llama-xkv-landmark.cpp):
// per-chunk rows decode A rows to P=padded canonical floats (Turbo: inverse
// WHT inside via CANONICAL domain), decode B rows per layer slice, dot over
// LOGICAL rank, forward-phase per head, mean-pool, quant-encode, then
// authoritative post-encode decode-compare error bound
// (eb = 1.10 * ||dec - mean||_2 over logical D; floor 1e-4 non-F32) and FNV-1a
// source fingerprint over phased chunk floats.
//
// Node model (single-dst + writable src convention):
//   dst    = landmarks packed [padded_dim, n_chunks], type Q8_0 or Turbo4_0
//   src[0] = a_k       token-major [pad_rank, n_rows_a] (F32/F16/Q8_0/T2/3/4)
//   src[1] = b_k       feature-major [pad_rank, b_rows] (same type class)
//   src[2] = rows      I32 [n_rows] surviving A row indices
//   src[3] = positions I32 or I64 [n_rows] per-row storage positions
//   src[4] = layer_meta I32 [6, n_layers]: [feature_offset, feature_dim,
//            head_dim (0=>feature_dim), rotary_dim (0=>no RoPE, even),
//            rope_mode (0=HALF,1=INTERLEAVED), rope_offset (float index)]
//   src[5] = rope_tables F32 [rope_nelems]: [omega(Fc_total), mag(Fc_total)]
//            mag is the DIRECT multiplier (cf. xkv_rope_apply /
//            xkv_canonicalize.comp sc convention)
//   src[6] = scratch   U8/I8/F32 [exact bytes from scratch_bytes]
//   src[7] = eb        F32 [n_chunks] (WRITTEN per-chunk error bound)
//   src[8] = srcfp     I64 [n_chunks] (WRITTEN FNV-1a over phased floats)
//   src[9] = status    I32 [4] (WRITTEN: [0]=code, [1]=n_chunks_built,
//            [2]=0, [3]=0; failure leaves dst/eb/srcfp untouched)
//
// Owned by XkvNativeLandmarkBuild. Shared enum/switch/CMake merge coordinated
// with XkvVulkanFactorizer (factor) + XkvVulkanLandmark (selector).

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_XKV_LANDMARK_BUILD_VERSION 1

// Status codes emitted to status[0]
#define GGML_XKV_LANDMARK_BUILD_STATUS_OK          0
#define GGML_XKV_LANDMARK_BUILD_STATUS_ERR_BUDGET  1 // zero rows/chunks/tokens
#define GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT   2 // bad dims/shapes/contents
#define GGML_XKV_LANDMARK_BUILD_STATUS_ERR_WORKSPACE 3 // scratch too small
#define GGML_XKV_LANDMARK_BUILD_STATUS_ERR_UNSUPPORTED 4 // bad codec/layout

// layer_meta stride: [off, dim, head_dim, rotary_dim, rope_mode, rope_offset]
#define GGML_XKV_LANDMARK_BUILD_LAYER_META_STRIDE 6

// POD params for GGML_OP_XKV_LANDMARK_BUILD (64 bytes = 16 x u32)
typedef struct ggml_xkv_landmark_build_params {
    uint32_t version;       // must be GGML_XKV_LANDMARK_BUILD_VERSION
    uint32_t n_rows;        // surviving rows (>0)
    uint32_t n_chunks;      // ceil(n_rows/chunk_tokens)
    uint32_t chunk_tokens;  // rows per chunk (>0)
    uint32_t total_dim;     // D = sum feature_dim (>0, <=4096 for Ornith W=4*1024)
    uint32_t padded_dim;    // pad_D for landmark codec
    uint32_t rank;          // logical rank (>0)
    uint32_t pad_rank;      // padded rank P (>= rank)
    uint32_t n_layers;      // layer slices (>=1)
    uint32_t max_feature_dim; // max feature_dim across layers (0 = total_dim)
    uint32_t landmark_type; // Q8_0 (8) or TURBO4_0 (44)
    uint32_t a_type;        // ggml_type of A_K stream
    uint32_t b_type;        // ggml_type of B_K stream
    uint32_t seed;          // deterministic seed (info; quant is seedless)
    uint32_t phase_lo;      // phase-tx fingerprint lo (mixed into srcfp seed)
    uint32_t phase_hi;      // phase-tx fingerprint hi
} ggml_xkv_landmark_build_params;

#ifdef __cplusplus
static_assert(sizeof(ggml_xkv_landmark_build_params) == 64,
    "ggml_xkv_landmark_build_params must be 64 bytes");
#endif

// Exact checked scratch bytes (chunk-local float arena + decode tmp + slack).
// Fail-closed (false) on any invalid param/overflow.
GGML_API bool ggml_xkv_landmark_build_scratch_bytes(
    const ggml_xkv_landmark_build_params * params,
    size_t * out_bytes,
    char * err, size_t err_size);

// Lightweight validation (shapes/types/params/scratch size; no contents).
GGML_API bool ggml_xkv_landmark_build_supports(
    const struct ggml_tensor * a_k,
    const struct ggml_tensor * b_k,
    const struct ggml_tensor * rows,
    const struct ggml_tensor * positions,
    const struct ggml_tensor * layer_meta,
    const struct ggml_tensor * rope_tables,
    const struct ggml_tensor * scratch,
    const struct ggml_tensor * eb,
    const struct ggml_tensor * srcfp,
    const struct ggml_tensor * status,
    const struct ggml_tensor * dst,
    const ggml_xkv_landmark_build_params * params,
    char * err, size_t err_size);

// Graph builder (no host pointers in op_params; backend tensors only).
GGML_API struct ggml_tensor * ggml_xkv_landmark_build(
    struct ggml_context * ctx,
    struct ggml_tensor  * a_k,
    struct ggml_tensor  * b_k,
    struct ggml_tensor  * rows,
    struct ggml_tensor  * positions,
    struct ggml_tensor  * layer_meta,
    struct ggml_tensor  * rope_tables,
    struct ggml_tensor  * scratch,
    struct ggml_tensor  * eb,
    struct ggml_tensor  * srcfp,
    struct ggml_tensor  * status,
    const ggml_xkv_landmark_build_params * params);

// Independent CPU reference oracle: decode-product-phase-mean-encode.
// All streams host-readable here. Writes dst bytes + eb + srcfp + status.
// Leaves dst/eb/srcfp untouched on failure. No NaN on success.
GGML_API bool ggml_xkv_landmark_build_cpu_oracle(
    const void * a_k_data, enum ggml_type a_k_type, int64_t a_k_ne0, int64_t a_k_ne1,
    size_t a_k_nb0, size_t a_k_nb1,
    const void * b_k_data, enum ggml_type b_k_type, int64_t b_k_ne0, int64_t b_k_ne1,
    size_t b_k_nb0, size_t b_k_nb1,
    const int32_t * rows_data, int64_t n_rows,
    const int64_t * pos_data, int pos_is_64,
    const int32_t * layer_meta_data, int64_t n_layers,
    const float * rope_data, int64_t rope_nelements,
    const ggml_xkv_landmark_build_params * params,
    void * dst_data, int64_t dst_ne0, int64_t dst_ne1, size_t dst_nb0, size_t dst_nb1,
    float * eb_data, uint64_t * fp_data, int32_t * status_data,
    void * scratch, size_t scratch_bytes,
    char * err, size_t err_size);

#ifdef __cplusplus
}
#endif
