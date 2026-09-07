#pragma once

// ggml-xkv-factor.h — Native XKV device-local matrix factorization & canonicalization ops (v1).
//
// Composable GGML native primitives plus graph builder API for deterministic
// randomized SVD factorization of one canonical group matrix X[n,m], plus a
// device canonicalize-hot op, plus contiguous Turbo2/3/4/Q8_0/F16 factor
// encoding into final backend packed streams (exact existing quant
// kernels/layout/fingerprints). Full Vulkan algorithm stays device-local with
// device-local barriers; one final host sync per sealed group/bundle. Only a
// tiny scalar status word is host-readable; no D2H of K/V/factors.
//
// Node model (ggml single-dst + writable src convention, cf. OPT_STEP_ADAMW):
//  - GGML_OP_XKV_FACTORIZE: dst = A packed (type_a [pad_r_a, n]);
//      src[0] = X F32[m,n] canonical staging; src[1] = scratch F32 (exact bytes
//      from ggml_xkv_factorize_scratch_bytes); src[2] = B packed (type_b
//      [pad_r_b, m], WRITTEN by the op); src[3] = status I32[1] (WRITTEN: 0 ok).
//    K and V run as two independent ops with their own seeds/params.
//  - GGML_OP_XKV_CANONICALIZE: dst = X F32[total_feat, n_rows];
//      src[0] = hot_kv (F32/F16/Q8_0/Turbo2/3/4); src[1] = rows I32[n_rows]
//      (selected physical rows); src[2] = positions I32[n_rows];
//      src[3] = rope_tables F32[2*Fc] (empty iff rotary_dim==0);
//      src[4] = hadamard F32[H,H] (empty iff hadamard_dim==0, skip rotation);
//      src[5] = status I32[1] (WRITTEN: 0 ok).
//    Applies dequant -> inverse attention rotation (if any) -> exact
//    table-driven inverse text RoPE (HALF or INTERLEAVED) for K; V skips RoPE.
//
// Owned by XkvVulkanFactorizer. Shared enum/switch/CMake merge owned by
// XkvVulkanAttention (do not edit shared regions here).

#include "ggml.h"
#include "ggml-xkv.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_XKV_FACTOR_VERSION 1

// Balance modes (match src/llama-xkv-factor.cpp factor_balance semantics)
#define GGML_XKV_BALANCE_UPSTREAM 0 // A = U*S,   B^T = V
#define GGML_XKV_BALANCE_SQRT     1 // A = U*S^1/2, B^T = V*S^1/2
#define GGML_XKV_BALANCE_DIAGONAL 2 // sqrt then equalize per-column norms

// POD params for GGML_OP_XKV_FACTORIZE (52 bytes, fits GGML_MAX_OP_PARAMS=64)
typedef struct ggml_xkv_factorize_params {
    uint32_t version;          // GGML_XKV_FACTOR_VERSION
    uint32_t rows_n;           // n (tokens)
    uint32_t cols_m;           // m (concatenated features)
    uint32_t requested_rank;   // target rank r
    uint32_t oversampling;     // oversampling p (fixed, e.g. 16)
    uint32_t power_iterations; // power iterations q (>= 1)
    uint32_t balance_mode;     // GGML_XKV_BALANCE_*
    uint32_t type_a;           // output ggml_type for A stream
    uint32_t type_b;           // output ggml_type for B stream
    uint32_t seed_low;         // low 32 bits of deterministic Rademacher seed
    uint32_t seed_high;        // high 32 bits
    uint32_t pad_r_a;          // padded rank for A (0 = requested_rank)
    uint32_t pad_r_b;          // padded rank for B (0 = requested_rank)
    uint32_t _reserved;        // zero
} ggml_xkv_factorize_params;

// POD params for GGML_OP_XKV_CANONICALIZE (48 bytes)
typedef struct ggml_xkv_canonicalize_params {
    uint32_t version;      // GGML_XKV_FACTOR_VERSION
    uint32_t n_rows;       // selected tokens
    uint32_t n_layers;     // owning layers in group
    uint32_t n_heads;      // KV heads per layer
    uint32_t head_dim;     // head feature dim
    uint32_t padded_head_dim; // per-head stored width (Turbo: round128(head_dim); else head_dim)
    uint32_t total_feat;   // m = n_layers*n_heads*head_dim
    uint32_t rotary_dim;   // 0 = no RoPE (V path); even otherwise
    uint32_t rope_mode;    // GGML_XKV_ROPE_HALF / _INTERLEAVED
    uint32_t input_type;   // ggml_type of hot input
    uint32_t is_k;         // 1 = apply inverse RoPE (K), 0 = skip (V)
    uint32_t hadamard_dim; // 0 = skip inverse attention rotation
} ggml_xkv_canonicalize_params;

// Exact scratch bytes (F32 floats + order u32) for factorize: Y/Q(n*l) +
// Z(m*l) + C(l*m) + G(l*l) + V(l*l) + S(l) + order(l) + Af(n*r) + BTf(m*r),
// l = min(min(n,m), r+oversampling). Fail-closed on overflow/bad params.
// T-1 byte short must fail supports (exactness test relies on it).
GGML_API bool ggml_xkv_factorize_scratch_bytes(
    const ggml_xkv_factorize_params * params,
    size_t * out_bytes,
    char * err, size_t err_size);

// Graph builders (no host pointers in op_params; backend tensors only)
GGML_API struct ggml_tensor * ggml_xkv_factorize(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    struct ggml_tensor  * scratch,
    struct ggml_tensor  * b_out,
    struct ggml_tensor  * status,
    const ggml_xkv_factorize_params * params);

GGML_API struct ggml_tensor * ggml_xkv_canonicalize(
    struct ggml_context * ctx,
    struct ggml_tensor  * hot_kv,
    struct ggml_tensor  * rows,
    struct ggml_tensor  * positions,
    struct ggml_tensor  * rope_tables,
    struct ggml_tensor  * hadamard,
    struct ggml_tensor  * status,
    const ggml_xkv_canonicalize_params * params);

// Lightweight validation (shapes/types/params/scratch size; no contents)
GGML_API bool ggml_xkv_factorize_supports(
    const struct ggml_tensor * x,
    const struct ggml_tensor * scratch,
    const struct ggml_tensor * b_out,
    const struct ggml_tensor * status,
    const struct ggml_tensor * dst_a,
    const ggml_xkv_factorize_params * params,
    char * err, size_t err_size);

GGML_API bool ggml_xkv_canonicalize_supports(
    const struct ggml_tensor * hot_kv,
    const struct ggml_tensor * rows,
    const struct ggml_tensor * positions,
    const struct ggml_tensor * rope_tables,
    const struct ggml_tensor * hadamard,
    const struct ggml_tensor * status,
    const struct ggml_tensor * dst,
    const ggml_xkv_canonicalize_params * params,
    char * err, size_t err_size);

// Independent CPU reference oracle (distinct implementation mirroring the
// device algorithm: counter Rademacher, CGS2x2, Jacobi, sorted/sign-canonical
// triplets, rank-deficiency guards, balance modes, exact quant encode).
// Writes packed A/B streams + singular values; leaves outputs untouched on
// failure (returns false + err). No NaN on success.
//
// Device-side residual contract (status tensor): the factorize status must
// be I32[>=10]: [0]=code, [1..3]=A+B residual, [4..6]=A-only, [7..9]=B-only
// (each frob_orig, frob_err, max_abs_err as float bits). Modes decode FINAL
// streams (A-only pairs packed A with FP BT, B-only FP Af with packed B).
typedef struct ggml_xkv_residual {
    float frob_orig; // ||X||_F
    float frob_err;  // ||X - recon||_F
    float max_err;   // max |X - recon|
} ggml_xkv_residual;
GGML_API bool ggml_xkv_factorize_cpu_oracle(
    const float * x_data,
    uint32_t n, uint32_t m,
    const ggml_xkv_factorize_params * params,
    void * out_a, void * out_b,
    float * out_s,
    char * err, size_t err_size);

// Extended oracle with tiled final-codec residuals (nullable each):
// ab = decode(A).decode(B), a_only = decode(A).BT_fp, b_only = Af_fp.decode(B).
// Residuals run over a deterministic strided row sample (S = min(n,128),
// stride = n/S, rows k*stride) bit-identical to the device residual steps;
// small-n cases cover all rows. Relative error convention matches
// factor_error_report (llama side).
GGML_API bool ggml_xkv_factorize_cpu_oracle_resid(
    const float * x_data,
    uint32_t n, uint32_t m,
    const ggml_xkv_factorize_params * params,
    void * out_a, void * out_b,
    float * out_s,
    ggml_xkv_residual * out_r_ab,
    ggml_xkv_residual * out_r_a,
    ggml_xkv_residual * out_r_b,
    char * err, size_t err_size);

// CPU canonicalize oracle (dequant + inv rotation + inv RoPE), same contract.
GGML_API bool ggml_xkv_canonicalize_cpu_oracle(
    const void * hot_data, enum ggml_type hot_type,
    const int32_t * rows, const void * positions_data, int pos_is_64, uint32_t n_rows,
    const float * rope_tables, uint32_t rope_nelements,
    const float * hadamard, uint32_t hadamard_dim,
    const ggml_xkv_canonicalize_params * params,
    float * out_x,
    char * err, size_t err_size);

#ifdef __cplusplus
}
#endif
