// ggml-xkv-factor.cpp — Native XKV device-local factorization/canonicalize ops (v1).
//
// CPU independent oracle + validation + graph builders. Vulkan backend shares
// validation; device kernels live in xkv_factorize.comp / xkv_canonicalize.comp.
// No host pointers in op_params. Scratch arenas stay device-resident on Vulkan.

#include "ggml-xkv-factor.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

static void xkv_err(char * err, size_t n, const char * msg) {
    if (err && n) {
        snprintf(err, n, "%s", msg);
    }
}

// Counter-based deterministic Rademacher using 32-bit-only integer hashing
// (exact on every Vulkan device, no int64 required). Counter = (k, l, j):
// feature index k, oversampled width l, projection column j. Identical on CPU
// oracle and xkv_factorize.comp (fused: no persistent Omega).
static uint32_t xkv_fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

static double xkv_rademacher_at(uint32_t seed_lo, uint32_t seed_hi,
                                  uint32_t k, uint32_t l, uint32_t j) {
    uint32_t h = seed_lo ^ (k * 2654435761u) ^ (j * 2246822519u) ^
                   (l * 3266489917u) ^ (seed_hi * 668265263u);
    h = xkv_fmix32(h == 0u ? 0x9E3779B9u : h);
    return (h & 1u) ? 1.0 : -1.0;
}


static uint64_t xkv_eff_l(const ggml_xkv_factorize_params * p) {
    uint64_t n = p->rows_n, m = p->cols_m;
    uint64_t mn = std::min(n, m);
    uint64_t l = (uint64_t)p->requested_rank + (uint64_t)p->oversampling;
    return std::min(mn, l);
}

} // anonymous namespace

bool ggml_xkv_factorize_scratch_bytes(const ggml_xkv_factorize_params * params,
                                      size_t * out_bytes,
                                      char * err, size_t err_size) {
    if (!params || !out_bytes) {
        xkv_err(err, err_size, "xkv_factorize: null params or out_bytes");
        return false;
    }
    if (params->version != GGML_XKV_FACTOR_VERSION) {
        xkv_err(err, err_size, "xkv_factorize: bad version");
        return false;
    }
    if (params->rows_n == 0 || params->cols_m == 0 || params->requested_rank == 0) {
        xkv_err(err, err_size, "xkv_factorize: invalid dimensions");
        return false;
    }
    if (params->power_iterations == 0) {
        xkv_err(err, err_size, "xkv_factorize: power_iterations must be >= 1");
        return false;
    }
    if (params->balance_mode > GGML_XKV_BALANCE_DIAGONAL) {
        xkv_err(err, err_size, "xkv_factorize: unknown balance mode");
        return false;
    }
    uint64_t n = params->rows_n, m = params->cols_m;
    uint64_t r = std::min<uint64_t>(params->requested_rank, std::min(n, m));
    uint64_t l = xkv_eff_l(params);
    // Device kernel bounds: single-workgroup stages assume l <= 512; u32
    // push-constant arithmetic bounds n, m.
    if (n > 8192 || m > 8192 || l > 512 || l == 0) {
        xkv_err(err, err_size, "xkv_factorize: dims exceed device bounds (n,m<=8192, l<=512)");
        return false;
    }
    // floats: Y/Q(nl) + Z(ml) + C(lm) + G(ll) + V(ll) + S(l) + order(l) + Af(nr) + BTf(mr)
    __uint128_t floats = (__uint128_t)n * l + (__uint128_t)m * l + (__uint128_t)l * m
        + (__uint128_t)l * l + (__uint128_t)l * l + (__uint128_t)l
        + (__uint128_t)l + (__uint128_t)n * r + (__uint128_t)m * r
        + (__uint128_t)64 * 10; // RES: bounded residual band partials (G<=64)
    __uint128_t bytes = floats * 4;
    if (bytes > SIZE_MAX) {
        xkv_err(err, err_size, "xkv_factorize: scratch overflow");
        return false;
    }
    *out_bytes = (size_t)bytes;
    return true;
}

bool ggml_xkv_factorize_supports(const struct ggml_tensor * x,
                                 const struct ggml_tensor * scratch,
                                 const struct ggml_tensor * b_out,
                                 const struct ggml_tensor * status,
                                 const struct ggml_tensor * dst_a,
                                 const ggml_xkv_factorize_params * params,
                                 char * err, size_t err_size) {
    if (!x || !scratch || !b_out || !status || !dst_a || !params) {
        xkv_err(err, err_size, "xkv_factorize: null tensors or params");
        return false;
    }
    if (params->version != GGML_XKV_FACTOR_VERSION) {
        xkv_err(err, err_size, "xkv_factorize: bad descriptor version");
        return false;
    }
    if (params->rows_n == 0 || params->cols_m == 0 || params->requested_rank == 0) {
        xkv_err(err, err_size, "xkv_factorize: invalid dimensions");
        return false;
    }
    if (params->power_iterations == 0) {
        xkv_err(err, err_size, "xkv_factorize: power_iterations must be >= 1");
        return false;
    }
    if (params->balance_mode > GGML_XKV_BALANCE_DIAGONAL) {
        xkv_err(err, err_size, "xkv_factorize: unknown balance mode");
        return false;
    }
    if (x->type != GGML_TYPE_F32 || x->ne[0] != (int64_t)params->cols_m ||
        x->ne[1] != (int64_t)params->rows_n) {
        xkv_err(err, err_size, "xkv_factorize: X must be contiguous F32[m,n]");
        return false;
    }
    if (x->nb[0] != sizeof(float)) {
        xkv_err(err, err_size, "xkv_factorize: X must be contiguous");
        return false;
    }
    if (x->nb[1] != (size_t)params->cols_m * sizeof(float)) {
        xkv_err(err, err_size, "xkv_factorize: X rows must be contiguous");
        return false;
    }
    if (!ggml_xkv_codec_supported(dst_a->type) || !ggml_xkv_codec_supported(b_out->type)) {
        xkv_err(err, err_size, "xkv_factorize: unsupported codec (only F32/F16/Q8_0/Turbo2/3/4)");
        return false;
    }
    uint64_t n = params->rows_n;
    uint64_t mm = params->cols_m;
    uint32_t r = (uint32_t)std::min<uint64_t>(params->requested_rank, std::min(n, mm));
    uint32_t pra = params->pad_r_a ? params->pad_r_a : r;
    uint32_t prb = params->pad_r_b ? params->pad_r_b : r;
    if ((int32_t)pra != ggml_xkv_padded_rank(dst_a->type, r)) {
        xkv_err(err, err_size, "xkv_factorize: dst_a ne0 != ggml_xkv_padded_rank(type_a, r)");
        return false;
    }
    if ((int32_t)prb != ggml_xkv_padded_rank(b_out->type, r)) {
        xkv_err(err, err_size, "xkv_factorize: b_out ne0 != ggml_xkv_padded_rank(type_b, r)");
        return false;
    }
    if (dst_a->ne[0] != (int64_t)pra || dst_a->ne[1] != (int64_t)n) {
        xkv_err(err, err_size, "xkv_factorize: dst_a must be [pad_r_a,n]");
        return false;
    }
    if (b_out->ne[0] != (int64_t)prb || b_out->ne[1] != (int64_t)mm) {
        xkv_err(err, err_size, "xkv_factorize: b_out must be [pad_r_b,m]");
        return false;
    }
    size_t eb_a = ggml_row_size(dst_a->type, dst_a->ne[0]);
    size_t eb_b = ggml_row_size(b_out->type, b_out->ne[0]);
    if (dst_a->nb[1] != eb_a || b_out->nb[1] != eb_b) {
        xkv_err(err, err_size, "xkv_factorize: packed streams must be contiguous");
        return false;
    }
    if (status->type != GGML_TYPE_I32 || ggml_nelements(status) < 10) {
        // [0]=code, [1..3]=AB residual, [4..6]=A-only, [7..9]=B-only.
        xkv_err(err, err_size, "xkv_factorize: status must be I32[>=10]");
        return false;
    }
    size_t need = 0;
    if (!ggml_xkv_factorize_scratch_bytes(params, &need, err, err_size)) return false;
    size_t have = 0;
    if (scratch->type == GGML_TYPE_F32) {
        have = (size_t)ggml_nelements(scratch) * sizeof(float);
    } else {
        have = ggml_nbytes(scratch);
    }
    if (have < need) {
        xkv_err(err, err_size, "xkv_factorize: scratch smaller than exact requirement (no T-1)");
        return false;
    }
    return true;
}

bool ggml_xkv_canonicalize_supports(const struct ggml_tensor * hot_kv,
                                    const struct ggml_tensor * rows,
                                    const struct ggml_tensor * positions,
                                    const struct ggml_tensor * rope_tables,
                                    const struct ggml_tensor * hadamard,
                                    const struct ggml_tensor * status,
                                    const struct ggml_tensor * dst,
                                    const ggml_xkv_canonicalize_params * params,
                                    char * err, size_t err_size) {
    if (!hot_kv || !rows || !positions || !rope_tables || !hadamard || !status || !dst || !params) {
        xkv_err(err, err_size, "xkv_canon: null tensors or params");
        return false;
    }
    if (params->version != GGML_XKV_FACTOR_VERSION) {
        xkv_err(err, err_size, "xkv_canon: bad descriptor version");
        return false;
    }
    if (params->n_rows == 0 || params->total_feat == 0 || params->head_dim == 0 ||
        params->padded_head_dim == 0) {
        xkv_err(err, err_size, "xkv_canon: invalid dimensions");
        return false;
    }
    // Device kernel stages one head in registers: bound the stored width.
    if (params->head_dim > 512 || params->padded_head_dim > 512) {
        xkv_err(err, err_size, "xkv_canon: head widths exceed 512 device bound");
        return false;
    }
    if ((uint64_t)params->n_layers * params->n_heads * params->head_dim != params->total_feat) {
        xkv_err(err, err_size, "xkv_canon: total_feat != n_layers*n_heads*head_dim");
        return false;
    }
    enum ggml_type ht = (enum ggml_type)params->input_type;
    if (hot_kv->type != ht || !ggml_xkv_codec_supported(ht)) {
        xkv_err(err, err_size, "xkv_canon: unsupported hot codec");
        return false;
    }
    bool ht_turbo = (ht == GGML_TYPE_TURBO2_0 || ht == GGML_TYPE_TURBO3_0 || ht == GGML_TYPE_TURBO4_0);
    uint32_t want_pad = ht_turbo ? ((params->head_dim + 127) / 128 * 128) : params->head_dim;
    if (params->padded_head_dim != want_pad) {
        xkv_err(err, err_size, "xkv_canon: padded_head_dim must be round128(head_dim) for Turbo, head_dim otherwise");
        return false;
    }
    if (ht == GGML_TYPE_Q8_0 && (params->padded_head_dim % 32) != 0) {
        xkv_err(err, err_size, "xkv_canon: Q8_0 head width must be a multiple of 32");
        return false;
    }
    {
        // Hot tensor is typed [nvec*padded_head_dim, n_phys]: per-head blocks
        // keep their own norms; Turbo blocks stay 128-aligned.
        uint64_t nvec = (uint64_t)params->n_layers * params->n_heads;
        if (hot_kv->ne[0] != (int64_t)(nvec * params->padded_head_dim)) {
            xkv_err(err, err_size, "xkv_canon: hot ne0 != nvec*padded_head_dim");
            return false;
        }
        size_t exp_b1 = ggml_row_size(ht, hot_kv->ne[0]);
        if (hot_kv->nb[1] != exp_b1) {
            xkv_err(err, err_size, "xkv_canon: hot rows must be contiguous");
            return false;
        }
    }
    if (rows->type != GGML_TYPE_I32 || rows->ne[0] != (int64_t)params->n_rows) {
        xkv_err(err, err_size, "xkv_canon: rows must be I32[n_rows]");
        return false;
    }
    if ((positions->type != GGML_TYPE_I32 && positions->type != GGML_TYPE_I64) ||
        positions->ne[0] != (int64_t)params->n_rows) {
        xkv_err(err, err_size, "xkv_canon: positions must be I32 or I64 [n_rows]");
        return false;
    }
    if (params->rotary_dim > params->head_dim || (params->rotary_dim & 1u) != 0) {
        xkv_err(err, err_size, "xkv_canon: rotary_dim must be even and <= head_dim");
        return false;
    }
    if (params->rope_mode != GGML_XKV_ROPE_HALF && params->rope_mode != GGML_XKV_ROPE_INTERLEAVED) {
        xkv_err(err, err_size, "xkv_canon: bad rope_mode");
        return false;
    }
    {
        uint32_t fc = params->rotary_dim / 2;
        int64_t want = fc == 0 ? 0 : (int64_t)fc * 2;
        if (rope_tables->type != GGML_TYPE_F32 || ggml_nelements(rope_tables) != want) {
            xkv_err(err, err_size, "xkv_canon: rope_tables must be F32[2*Fc]");
            return false;
        }
    }
    if (params->hadamard_dim == 0) {
        if (ggml_nelements(hadamard) != 0) {
            xkv_err(err, err_size, "xkv_canon: hadamard must be empty when hadamard_dim==0");
            return false;
        }
    } else {
        if (hadamard->type != GGML_TYPE_F32 || hadamard->ne[0] != (int64_t)params->hadamard_dim ||
            hadamard->ne[1] != (int64_t)params->hadamard_dim) {
            xkv_err(err, err_size, "xkv_canon: hadamard must be F32[H,H]");
            return false;
        }
        if (params->head_dim % params->hadamard_dim != 0) {
            xkv_err(err, err_size, "xkv_canon: head_dim must be a multiple of hadamard_dim");
            return false;
        }
    }
    if (status->type != GGML_TYPE_I32 || ggml_nelements(status) < 1) {
        xkv_err(err, err_size, "xkv_canon: status must be I32[>=1]");
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != (int64_t)params->total_feat ||
        dst->ne[1] != (int64_t)params->n_rows) {
        xkv_err(err, err_size, "xkv_canon: dst must be F32[total_feat,n_rows]");
        return false;
    }
    return true;
}

struct ggml_tensor * ggml_xkv_factorize(struct ggml_context * ctx,
                                        struct ggml_tensor * x,
                                        struct ggml_tensor * scratch,
                                        struct ggml_tensor * b_out,
                                        struct ggml_tensor * status,
                                        const ggml_xkv_factorize_params * params) {
    if (!ctx || !x || !scratch || !b_out || !status || !params) return nullptr;
    uint64_t n = params->rows_n;
    uint64_t m = params->cols_m;
    uint32_t r = (uint32_t)std::min<uint64_t>(params->requested_rank, std::min(n, m));
    uint32_t pra = params->pad_r_a ? params->pad_r_a : r;
    enum ggml_type ta = (enum ggml_type)params->type_a;
    size_t row_bytes = ggml_row_size(ta, pra);
    struct ggml_tensor * dst_a = ggml_new_tensor_2d(ctx, ta, (int64_t)pra, (int64_t)n);
    if (!dst_a) return nullptr;
    // Exact packed layout: row stride == ggml_row_size (no extra padding)
    dst_a->op = GGML_OP_XKV_FACTORIZE;
    memcpy(dst_a->op_params, params, sizeof(*params));
    dst_a->src[0] = x;
    dst_a->src[1] = scratch;
    dst_a->src[2] = b_out;
    dst_a->src[3] = status;
    (void)row_bytes;
    return dst_a;
}

struct ggml_tensor * ggml_xkv_canonicalize(struct ggml_context * ctx,
                                           struct ggml_tensor * hot_kv,
                                           struct ggml_tensor * rows,
                                           struct ggml_tensor * positions,
                                           struct ggml_tensor * rope_tables,
                                           struct ggml_tensor * hadamard,
                                           struct ggml_tensor * status,
                                           const ggml_xkv_canonicalize_params * params) {
    if (!ctx || !hot_kv || !rows || !positions || !rope_tables || !hadamard || !status || !params) {
        return nullptr;
    }
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
        (int64_t)params->total_feat, (int64_t)params->n_rows);
    if (!dst) return nullptr;
    dst->op = GGML_OP_XKV_CANONICALIZE;
    memcpy(dst->op_params, params, sizeof(*params));
    dst->src[0] = hot_kv;
    dst->src[1] = rows;
    dst->src[2] = positions;
    dst->src[3] = rope_tables;
    dst->src[4] = hadamard;
    dst->src[5] = status;
    return dst;
}

// Two-pass CGS2 orthonormalization (double accumulator; rank-deficiency guard
// zeroes degenerate columns instead of producing NaN).
static void qr_cgs2(double * Q, uint64_t n, uint64_t l) {
    for (uint64_t j = 0; j < l; ++j) {
        for (int pass = 0; pass < 2; ++pass) {
            for (uint64_t j0 = 0; j0 < j; ++j0) {
                double dot = 0.0;
                for (uint64_t i = 0; i < n; ++i) dot += Q[i * l + j0] * Q[i * l + j];
                for (uint64_t i = 0; i < n; ++i) Q[i * l + j] -= dot * Q[i * l + j0];
            }
        }
        double norm = 0.0;
        for (uint64_t i = 0; i < n; ++i) { double v = Q[i * l + j]; norm += v * v; }
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (uint64_t i = 0; i < n; ++i) Q[i * l + j] /= norm;
        } else {
            for (uint64_t i = 0; i < n; ++i) Q[i * l + j] = 0.0;
        }
    }
}

// One-sided cyclic Jacobi SVD on tall matrix U(n x m); V(m x m) accumulates
// right rotations. Returns singular values via column norms.
static bool jacobi_svd_tall(double * U, double * V, uint64_t n, uint64_t m,
                            std::vector<double> & S, char * err, size_t esz) {
    const double tol = 1e-9;
    const uint32_t sweeps = 30;
    if (n == 0 || m == 0) { S.assign(m, 0.0); return true; }
    std::vector<double> col(m, 0.0);
    for (uint64_t j = 0; j < m; ++j) {
        double s = 0.0;
        for (uint64_t i = 0; i < n; ++i) { double v = U[i * m + j]; s += v * v; }
        col[j] = s;
    }
    bool conv = false;
    for (uint32_t sw = 0; sw < sweeps; ++sw) {
        double maxc = 0.0;
        uint32_t rots = 0;
        for (uint64_t j1 = 0; j1 < m; ++j1) {
            for (uint64_t j2 = j1 + 1; j2 < m; ++j2) {
                double dot = 0.0;
                for (uint64_t i = 0; i < n; ++i) dot += U[i * m + j1] * U[i * m + j2];
                double den = std::sqrt(col[j1] * col[j2]);
                if (den > 1e-15) maxc = std::max(maxc, std::abs(dot) / den);
                if (std::abs(dot) <= tol * den || den < 1e-15) continue;
                rots++;
                double tau = (col[j1] - col[j2]) / (2.0 * dot);
                double t = tau >= 0 ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                    : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                for (uint64_t i = 0; i < n; ++i) {
                    double u1 = U[i * m + j1], u2 = U[i * m + j2];
                    U[i * m + j1] = c * u1 + s * u2;
                    U[i * m + j2] = -s * u1 + c * u2;
                }
                for (uint64_t i = 0; i < m; ++i) {
                    double v1 = V[i * m + j1], v2 = V[i * m + j2];
                    V[i * m + j1] = c * v1 + s * v2;
                    V[i * m + j2] = -s * v1 + c * v2;
                }
                col[j1] = std::max(0.0, col[j1] + t * dot);
                col[j2] = std::max(0.0, col[j2] - t * dot);
            }
        }
        for (uint64_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (uint64_t i = 0; i < n; ++i) { double v = U[i * m + j]; s += v * v; }
            col[j] = s;
        }
        if (maxc < tol || rots == 0) { conv = true; break; }
    }
    if (!conv && m > 1) {
        xkv_err(err, esz, "xkv_factorize: Jacobi did not converge");
        return false;
    }
    S.assign(m, 0.0);
    for (uint64_t j = 0; j < m; ++j) {
        double s = 0.0;
        for (uint64_t i = 0; i < n; ++i) { double v = U[i * m + j]; s += v * v; }
        S[j] = std::sqrt(s);
    }
    return true;
}

// Whole-stream decode to padded float rows (ROTATED Turbo domain, matching
// the production reader dot over full padded rank).
static bool xkv_decode_stream(enum ggml_type t, const void * bytes, uint32_t rows,
                              uint32_t pr, std::vector<float> & out, char * err, size_t esz) {
    out.assign((size_t)rows * pr, 0.0f);
    size_t rb = ggml_row_size(t, pr);
    std::vector<float> pad(pr);
    bool is_turbo = (t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0);
    for (uint32_t i = 0; i < rows; ++i) {
        const uint8_t * src = (const uint8_t *)bytes + (size_t)i * rb;
        if (t == GGML_TYPE_F32) {
            memcpy(pad.data(), src, (size_t)pr * 4);
        } else if (is_turbo) {
            if (!ggml_dequantize_turbo_row(t, src, pad.data(), pr, 128, GGML_TURBO_DECODE_ROTATED)) {
                xkv_err(err, esz, "xkv_factorize: turbo decode failed");
                return false;
            }
        } else {
            const auto * tr = ggml_get_type_traits(t);
            if (!tr || !tr->to_float) {
                xkv_err(err, esz, "xkv_factorize: no decoder");
                return false;
            }
            tr->to_float(src, pad.data(), pr);
        }
        memcpy(out.data() + (size_t)i * pr, pad.data(), (size_t)pr * 4);
    }
    return true;
}

// Tiled final-codec residual: X[n*m] vs P[n*pr] x Q[m*pr] dotted over pr.
// Fills frob_orig/frob_err/max_err (double accumulation, row tiles).
static void xkv_tiled_residual(const float * X, const float * P, const float * Q,
                               uint32_t n, uint64_t m, uint32_t pr,
                               ggml_xkv_residual * out) {
    double so = 0.0, se = 0.0, mx = 0.0;
    const uint64_t TILE = 64;
    for (uint64_t r0 = 0; r0 < n; r0 += TILE) {
        uint64_t r1 = std::min<uint64_t>(r0 + TILE, n);
        for (uint64_t i = r0; i < r1; ++i) {
            for (uint64_t j = 0; j < m; ++j) {
                double rec = 0.0;
                const float * prow = P + i * pr;
                const float * qrow = Q + j * pr;
                for (uint32_t k = 0; k < pr; ++k) rec += (double)prow[k] * (double)qrow[k];
                double o = (double)X[i * m + j];
                double e = o - rec;
                so += o * o;
                se += e * e;
                double ae = e < 0 ? -e : e;
                if (ae > mx) mx = ae;
            }
        }
    }
    out->frob_orig = (float)std::sqrt(so);
    out->frob_err = (float)std::sqrt(se);
    out->max_err = (float)mx;
}

bool ggml_xkv_factorize_cpu_oracle_resid(const float * x_data,
                                   uint32_t n, uint32_t m,
                                   const ggml_xkv_factorize_params * params,
                                   void * out_a, void * out_b,
                                   float * out_s,
                                   ggml_xkv_residual * out_r_ab,
                                   ggml_xkv_residual * out_r_a,
                                   ggml_xkv_residual * out_r_b,
                                   char * err, size_t err_size) {
    if (!x_data || !params || !out_a || !out_b) {
        xkv_err(err, err_size, "xkv_factorize: null oracle inputs");
        return false;
    }
    if (params->version != GGML_XKV_FACTOR_VERSION || n == 0 || m == 0 ||
        params->requested_rank == 0 || params->power_iterations == 0 ||
        params->balance_mode > GGML_XKV_BALANCE_DIAGONAL) {
        xkv_err(err, err_size, "xkv_factorize: invalid oracle params");
        return false;
    }
    for (uint64_t i = 0; i < (uint64_t)n * m; ++i) {
        if (!std::isfinite(x_data[i])) {
            xkv_err(err, err_size, "xkv_factorize: NaN/Inf input");
            return false;
        }
    }
    uint32_t r = std::min<uint32_t>(params->requested_rank, std::min(n, m));
    uint64_t l = xkv_eff_l(params);
    uint32_t seed_lo = params->seed_low;
    uint32_t seed_hi = params->seed_high;
    uint32_t lu = (uint32_t)l;

    std::vector<float> A((size_t)n * r, 0.0f), BT((size_t)m * r, 0.0f);
    std::vector<float> Sr(r, 0.0f);

    if (std::min(n, m) <= 64) {
        bool tall = n >= m;
        uint64_t tn = tall ? n : m, tm = tall ? m : n;
        std::vector<double> U(tn * tm, 0.0), V(tm * tm, 0.0);
        if (tall) {
            for (uint64_t i = 0; i < tn * tm; ++i) U[i] = (double)x_data[i];
        } else {
            for (uint32_t rr = 0; rr < n; ++rr)
                for (uint32_t cc = 0; cc < m; ++cc)
                    U[(size_t)cc * n + rr] = (double)x_data[(size_t)rr * m + cc];
        }
        for (uint64_t i = 0; i < tm; ++i) V[i * tm + i] = 1.0;
        std::vector<double> Sd;
        if (!jacobi_svd_tall(U.data(), V.data(), tn, tm, Sd, err, err_size)) return false;
        std::vector<uint64_t> ord(tm);
        for (uint64_t i = 0; i < tm; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](uint64_t a, uint64_t b) { return Sd[a] > Sd[b]; });
        double smax = Sd[ord[0]];
        double guard = smax * 1e-12;
        for (uint32_t k = 0; k < r; ++k) {
            uint64_t src = ord[k];
            double sv = Sd[src];
            Sr[k] = (float)sv;
            double inv = (sv > guard && sv > 0.0) ? 1.0 / sv : 0.0;
            if (tall) {
                for (uint32_t i = 0; i < n; ++i) A[(size_t)i * r + k] = (float)(U[(size_t)i * m + src] * inv);
                for (uint32_t j = 0; j < m; ++j) BT[(size_t)j * r + k] = (float)V[(size_t)j * m + src];
            } else {
                for (uint32_t i = 0; i < n; ++i) A[(size_t)i * r + k] = (float)V[(size_t)i * n + src];
                for (uint32_t j = 0; j < m; ++j) BT[(size_t)j * r + k] = (float)(U[(size_t)j * n + src] * inv);
            }
        }
    } else {
        // Fused counter-Rademacher projection: Y = X * Omega (no Omega store)
        std::vector<double> Q((size_t)n * l, 0.0);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < l; ++j) {
                double s = 0.0;
                for (uint32_t k = 0; k < m; ++k) {
                    s += (double)x_data[(size_t)i * m + k] *
                         xkv_rademacher_at(seed_lo, seed_hi, k, lu, (uint32_t)j);
                }
                Q[(size_t)i * l + j] = s;
            }
        }
        qr_cgs2(Q.data(), n, l);
        std::vector<double> Z((size_t)m * l, 0.0), Yp((size_t)n * l, 0.0);
        for (uint32_t it = 0; it < params->power_iterations; ++it) {
            for (uint32_t i = 0; i < m; ++i) {
                for (uint64_t j = 0; j < l; ++j) {
                    double s = 0.0;
                    for (uint32_t k = 0; k < n; ++k) s += (double)x_data[(size_t)k * m + i] * Q[(size_t)k * l + j];
                    Z[(size_t)i * l + j] = s;
                }
            }
            qr_cgs2(Z.data(), m, l);
            for (uint32_t i = 0; i < n; ++i) {
                for (uint64_t j = 0; j < l; ++j) {
                    double s = 0.0;
                    for (uint32_t k = 0; k < m; ++k) s += (double)x_data[(size_t)i * m + k] * Z[(size_t)k * l + j];
                    Yp[(size_t)i * l + j] = s;
                }
            }
            Q.swap(Yp);
            qr_cgs2(Q.data(), n, l);
        }
        Z.clear(); Z.shrink_to_fit();
        Yp.clear(); Yp.shrink_to_fit();
        // Reduced core: CT = (Q^T X)^T (m x l), Gram-free Jacobi on CT
        std::vector<double> CT((size_t)m * l, 0.0);
        for (uint32_t i = 0; i < m; ++i) {
            for (uint64_t j = 0; j < l; ++j) {
                double s = 0.0;
                for (uint32_t k = 0; k < n; ++k) s += Q[(size_t)k * l + j] * (double)x_data[(size_t)k * m + i];
                CT[(size_t)i * l + j] = s;
            }
        }
        std::vector<double> VT(l * l, 0.0);
        for (uint64_t i = 0; i < l; ++i) VT[i * l + i] = 1.0;
        std::vector<double> Sc;
        if (!jacobi_svd_tall(CT.data(), VT.data(), m, l, Sc, err, err_size)) return false;
        // Sorted triplets: stable desc sort (index asc tie-break => repeated
        // singular values deterministic), then sign canonicalization below.
        std::vector<uint64_t> ord(l);
        for (uint64_t i = 0; i < l; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](uint64_t a, uint64_t b) { return Sc[a] > Sc[b]; });
        double smax = Sc[ord[0]];
        double guard = smax * 1e-12;
        for (uint32_t k = 0; k < r; ++k) {
            uint64_t src = ord[k];
            double sv = Sc[src];
            Sr[k] = (float)sv;
            double inv = (sv > guard && sv > 0.0) ? 1.0 / sv : 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                double u = 0.0;
                for (uint64_t j0 = 0; j0 < l; ++j0) u += Q[(size_t)i * l + j0] * VT[j0 * l + src];
                A[(size_t)i * r + k] = (float)u;
            }
            for (uint32_t j = 0; j < m; ++j) BT[(size_t)j * r + k] = (float)(CT[(size_t)j * l + src] * inv);
        }
    }

    // Sign canonicalization: flip (A_col, BT_col) when max-|.| element of A col is negative
    for (uint32_t k = 0; k < r; ++k) {
        float best = 0.0f;
        bool neg = false;
        for (uint32_t i = 0; i < n; ++i) {
            float v = A[(size_t)i * r + k];
            if (std::fabs(v) > std::fabs(best)) { best = v; neg = (v < 0.0f); }
        }
        if (neg) {
            for (uint32_t i = 0; i < n; ++i) A[(size_t)i * r + k] = -A[(size_t)i * r + k];
            for (uint32_t j = 0; j < m; ++j) BT[(size_t)j * r + k] = -BT[(size_t)j * r + k];
        }
    }

    if (params->balance_mode == GGML_XKV_BALANCE_UPSTREAM) {
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t k = 0; k < r; ++k) A[(size_t)i * r + k] *= Sr[k];
    } else {
        std::vector<float> sq(r);
        for (uint32_t k = 0; k < r; ++k) sq[k] = std::sqrt(std::max(0.0f, Sr[k]));
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t k = 0; k < r; ++k) A[(size_t)i * r + k] *= sq[k];
        for (uint32_t j = 0; j < m; ++j)
            for (uint32_t k = 0; k < r; ++k) BT[(size_t)j * r + k] *= sq[k];
        if (params->balance_mode == GGML_XKV_BALANCE_DIAGONAL) {
            for (uint32_t k = 0; k < r; ++k) {
                double na = 0.0, nb = 0.0;
                for (uint32_t i = 0; i < n; ++i) { double v = A[(size_t)i * r + k]; na += v * v; }
                for (uint32_t j = 0; j < m; ++j) { double v = BT[(size_t)j * r + k]; nb += v * v; }
                na = std::sqrt(na);
                nb = std::sqrt(nb);
                if (na > 1e-12 && nb > 1e-12) {
                    double d = std::sqrt(nb / na);
                    for (uint32_t i = 0; i < n; ++i) A[(size_t)i * r + k] = (float)(A[(size_t)i * r + k] * d);
                    for (uint32_t j = 0; j < m; ++j) BT[(size_t)j * r + k] = (float)(BT[(size_t)j * r + k] / d);
                }
            }
        }
    }

    for (uint64_t i = 0; i < (uint64_t)n * r; ++i) {
        if (!std::isfinite(A[i])) { xkv_err(err, err_size, "xkv_factorize: non-finite A"); return false; }
    }
    for (uint64_t i = 0; i < (uint64_t)m * r; ++i) {
        if (!std::isfinite(BT[i])) { xkv_err(err, err_size, "xkv_factorize: non-finite BT"); return false; }
    }

    // Contiguous encode into packed streams via exact ggml kernels
    enum ggml_type ta = (enum ggml_type)params->type_a;
    enum ggml_type tb = (enum ggml_type)params->type_b;
    uint32_t pra = params->pad_r_a ? params->pad_r_a : r;
    uint32_t prb = params->pad_r_b ? params->pad_r_b : r;
    std::vector<float> pad(std::max(pra, prb), 0.0f);
    auto encode_rows = [&](const std::vector<float> & S, uint32_t rows, uint32_t rank,
                           enum ggml_type t, uint32_t pr, void * out) -> bool {
        size_t rb = ggml_row_size(t, pr);
        if (t == GGML_TYPE_F32) {
            for (uint32_t i = 0; i < rows; ++i) {
                float * d = (float *)((uint8_t *)out + (size_t)i * rb);
                memcpy(d, S.data() + (size_t)i * rank, (size_t)rank * 4);
                if (pr > rank) memset(d + rank, 0, (size_t)(pr - rank) * 4);
            }
            return true;
        }
        const auto * tr = ggml_get_type_traits(t);
        bool is_turbo = (t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0);
        for (uint32_t i = 0; i < rows; ++i) {
            memcpy(pad.data(), S.data() + (size_t)i * rank, (size_t)rank * 4);
            if (pr > rank) memset(pad.data() + rank, 0, (size_t)(pr - rank) * 4);
            uint8_t * d = (uint8_t *)out + (size_t)i * rb;
            if (is_turbo) {
                if (!ggml_quantize_turbo_row(t, pad.data(), d, pr, 128)) {
                    xkv_err(err, err_size, "xkv_factorize: turbo encode failed");
                    return false;
                }
            } else {
                if (!tr || !tr->from_float_ref) {
                    xkv_err(err, err_size, "xkv_factorize: no encoder");
                    return false;
                }
                tr->from_float_ref(pad.data(), d, pr);
            }
        }
        return true;
    };
    if (!encode_rows(A, n, r, ta, pra, out_a)) return false;
    if (!encode_rows(BT, m, r, tb, prb, out_b)) return false;
    if (out_s) memcpy(out_s, Sr.data(), (size_t)r * 4);
    // Final-codec tiled residuals (nullable each). FP sides zero-extended to
    // padded width so the dot matches the production reader contract.
    if (out_r_ab || out_r_a || out_r_b) {
        std::vector<float> Ad, Bd;
        if (!xkv_decode_stream(ta, out_a, n, pra, Ad, err, err_size)) return false;
        if (!xkv_decode_stream(tb, out_b, m, prb, Bd, err, err_size)) return false;
        std::vector<float> Afz((size_t)n * pra, 0.0f), Bz((size_t)m * prb, 0.0f);
        for (uint32_t i = 0; i < n; ++i)
            memcpy(Afz.data() + (size_t)i * pra, A.data() + (size_t)i * r, (size_t)r * 4);
        for (uint32_t j = 0; j < m; ++j)
            memcpy(Bz.data() + (size_t)j * prb, BT.data() + (size_t)j * r, (size_t)r * 4);
        // NOTE: mixed pairs require equal padded widths (Turbo pairs share
        // type by validation; F32/F16/Q8 pads equal when types match). If the
        // pair pads differ, the production dot is undefined: report failure.
        if (pra != prb) {
            xkv_err(err, err_size, "xkv_factorize: mixed padded widths unsupported");
            return false;
        }
        // Exact all-row tiled residual (no sampling; matches device steps 12/13).
        if (out_r_ab) xkv_tiled_residual(x_data, Ad.data(), Bd.data(), n, m, pra, out_r_ab);
        if (out_r_a)  xkv_tiled_residual(x_data, Ad.data(), Bz.data(), n, m, pra, out_r_a);
        if (out_r_b)  xkv_tiled_residual(x_data, Afz.data(), Bd.data(), n, m, pra, out_r_b);
    }
    return true;
}

bool ggml_xkv_factorize_cpu_oracle(const float * x_data,
                                   uint32_t n, uint32_t m,
                                   const ggml_xkv_factorize_params * params,
                                   void * out_a, void * out_b,
                                   float * out_s,
                                   char * err, size_t err_size) {
    return ggml_xkv_factorize_cpu_oracle_resid(x_data, n, m, params, out_a, out_b,
        out_s, nullptr, nullptr, nullptr, err, err_size);
}

// ---- canonicalize oracle ----

// Non-Turbo whole-row decode (F32/F16/Q8_0), mirroring xkv_dequant_row in
// ggml-xkv-reconstruct.cpp. Turbo uses ggml_dequantize_turbo_row CANONICAL.
static bool xkv_dequant_row(const uint8_t * base, enum ggml_type t,
                            float * out, uint32_t dim, char * err, size_t esz) {
    if (t == GGML_TYPE_F32) {
        memcpy(out, base, (size_t)dim * 4);
        return true;
    }
    if (t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0) {
        const auto * tr = ggml_get_type_traits(t);
        if (!tr || !tr->to_float) {
            xkv_err(err, esz, "xkv_canon: no decoder for hot type");
            return false;
        }
        tr->to_float(base, out, dim);
        return true;
    }
    xkv_err(err, esz, "xkv_canon: unexpected Turbo in plain dequant");
    return false;
}

bool ggml_xkv_canonicalize_cpu_oracle(const void * hot_data, enum ggml_type hot_type,
                                      const int32_t * rows, const void * positions_data, int pos_is_64, uint32_t n_rows,
                                      const float * rope_tables, uint32_t rope_nelements,
                                      const float * hadamard, uint32_t hadamard_dim,
                                      const ggml_xkv_canonicalize_params * params,
                                      float * out_x,
                                      char * err, size_t err_size) {
    if (!hot_data || !rows || !positions_data || !params || !out_x) {
        xkv_err(err, err_size, "xkv_canon: null oracle inputs");
        return false;
    }
    if (params->version != GGML_XKV_FACTOR_VERSION || params->n_rows == 0 ||
        params->total_feat == 0 || params->head_dim == 0 || params->padded_head_dim == 0) {
        xkv_err(err, err_size, "xkv_canon: invalid oracle params");
        return false;
    }
    if (params->head_dim > 512 || params->padded_head_dim > 512) {
        xkv_err(err, err_size, "xkv_canon: head widths exceed 512 device bound");
        return false;
    }
    if (n_rows != params->n_rows) {
        xkv_err(err, err_size, "xkv_canon: n_rows mismatch");
        return false;
    }
    uint32_t n = params->n_rows, hd = params->head_dim, phd = params->padded_head_dim;
    uint32_t nvec = params->n_layers * params->n_heads;
    if ((uint64_t)nvec * hd != params->total_feat) {
        xkv_err(err, err_size, "xkv_canon: total_feat mismatch");
        return false;
    }
    bool is_turbo = (hot_type == GGML_TYPE_TURBO2_0 || hot_type == GGML_TYPE_TURBO3_0 ||
                       hot_type == GGML_TYPE_TURBO4_0);
    size_t head_bytes = ggml_row_size(hot_type, phd);
    size_t hot_row_bytes = head_bytes * nvec;
    std::vector<float> head(phd), tmp(hadamard ? params->hadamard_dim : 0);
    uint32_t fc = params->rotary_dim / 2;
    const float * omega = rope_tables;
    const float * mag = rope_tables ? rope_tables + fc : nullptr;
    if (params->rotary_dim > 0 && rope_nelements != 2u * fc) {
        xkv_err(err, err_size, "xkv_canon: rope table size mismatch");
        return false;
    }
    for (uint32_t i = 0; i < n; ++i) {
        int32_t pr = rows[i];
        if (pr < 0) {
            xkv_err(err, err_size, "xkv_canon: negative physical row");
            return false;
        }
        const uint8_t * row_base = (const uint8_t *)hot_data + (size_t)pr * hot_row_bytes;
        float pos = pos_is_64 ? (float)((const int64_t *)positions_data)[i]
                              : (float)((const int32_t *)positions_data)[i];
        for (uint32_t v = 0; v < nvec; ++v) {
            const uint8_t * hsrc = row_base + (size_t)v * head_bytes;
            float * hdst = out_x + (size_t)i * nvec * hd + (size_t)v * hd;
            // Per-head decode into canonical domain (Turbo: exact kernel,
            // CANONICAL domain inverts the WHT; mirrors dequant_k_head_from_rows)
            if (is_turbo) {
                if (!ggml_dequantize_turbo_row(hot_type, hsrc, head.data(), phd, 128,
                                               GGML_TURBO_DECODE_CANONICAL)) {
                    xkv_err(err, err_size, "xkv_canon: turbo canonical decode failed");
                    return false;
                }
            } else if (!xkv_dequant_row(hsrc, hot_type, head.data(), phd, err, err_size)) {
                return false;
            }
            memcpy(hdst, head.data(), (size_t)hd * 4);
            // Inverse attention rotation per hadamard_dim block (exact H multiply)
            if (hadamard_dim > 0) {
                if (!hadamard) { xkv_err(err, err_size, "xkv_canon: null hadamard"); return false; }
                for (uint32_t b = 0; b < hd; b += hadamard_dim) {
                    float * blk = hdst + b;
                    for (uint32_t ii = 0; ii < hadamard_dim; ++ii) {
                        double s = 0.0;
                        const float * hr = hadamard + (size_t)ii * hadamard_dim;
                        for (uint32_t jj = 0; jj < hadamard_dim; ++jj) s += (double)hr[jj] * blk[jj];
                        tmp[ii] = (float)s;
                    }
                    memcpy(blk, tmp.data(), (size_t)hadamard_dim * 4);
                }
            }
            // Exact inverse text RoPE for K (HALF or INTERLEAVED pairing)
            if (params->is_k && params->rotary_dim > 0) {
                if (params->rope_mode == GGML_XKV_ROPE_HALF) {
                    for (uint32_t f = 0; f < fc; ++f) {
                        float ang = omega[f] * pos;
                        float c = cosf(ang), s = sinf(ang);
                        float sc = (mag && mag[f] > 0.0f) ? mag[f] : 1.0f;
                        float re = hdst[f] / sc, im = hdst[f + fc] / sc;
                        hdst[f] = re * c + im * s;
                        hdst[f + fc] = im * c - re * s;
                    }
                } else {
                    for (uint32_t f = 0; f < fc; ++f) {
                        float ang = omega[f] * pos;
                        float c = cosf(ang), s = sinf(ang);
                        float sc = (mag && mag[f] > 0.0f) ? mag[f] : 1.0f;
                        float re = hdst[2 * f] / sc, im = hdst[2 * f + 1] / sc;
                        hdst[2 * f] = re * c + im * s;
                        hdst[2 * f + 1] = im * c - re * s;
                    }
                }
            }
        }
    }
    for (uint64_t i = 0; i < (uint64_t)n * nvec * hd; ++i) {
        if (!std::isfinite(out_x[i])) {
            xkv_err(err, err_size, "xkv_canon: non-finite output");
            return false;
        }
    }
    return true;
}
