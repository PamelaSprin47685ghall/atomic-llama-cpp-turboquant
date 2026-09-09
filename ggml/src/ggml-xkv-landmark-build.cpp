// ggml-xkv-landmark-build.cpp — Bounded native landmark construction from
// FINAL encoded A_K/B_K streams + CPU oracle + validation + graph builder.
//
// Mirrors the build_landmarks_bounded factory contract: decode A rows to
// padded canonical floats (Turbo via CANONICAL domain), decode B rows per
// layer slice, dot over LOGICAL rank, forward-phase EACH row at its own
// storage position per layer/head (never mean-then-rotate), mean-pool each
// fixed chunk, encode each output row directly as Q8_0 or Turbo4_0, then
// authoritative post-encode decode-compare error bound + FNV-1a fingerprint
// over provenance, exact source row/positions, and the encoded landmark row.
// Bounded chunk-local scratch only; no whole-K.

#include "ggml-xkv-landmark-build.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

static void xb_err(char * err, size_t n, const char * m) {
    if (err && n) {
        size_t i = 0;
        while (m[i] && i + 1 < n) { err[i] = m[i]; ++i; }
        err[i] = 0;
    }
}

static uint64_t xb_fnv_byte(uint64_t h, uint8_t v) {
    return (h ^ v) * UINT64_C(0x100000001b3);
}

static uint64_t xb_fnv_u32(uint64_t h, uint32_t v) {
    h = xb_fnv_byte(h, (uint8_t) v);
    h = xb_fnv_byte(h, (uint8_t) (v >> 8));
    h = xb_fnv_byte(h, (uint8_t) (v >> 16));
    return xb_fnv_byte(h, (uint8_t) (v >> 24));
}

static bool xb_add(size_t a, size_t b, size_t * o) {
    __uint128_t s = (__uint128_t)a + b;
    if (s > (size_t)-1) return false;
    *o = (size_t)s;
    return true;
}
static bool xb_mul(size_t a, size_t b, size_t * o) {
    __uint128_t p = (__uint128_t)a * b;
    if (p > (size_t)-1) return false;
    *o = (size_t)p;
    return true;
}
static bool xb_mul_u32(uint32_t a, uint32_t b, size_t * o) {
    return xb_mul((size_t)a, (size_t)b, o);
}

static bool xb_is_turbo(enum ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}
static bool xb_codec_ok(enum ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || xb_is_turbo(t);
}
static bool xb_landmark_ok(enum ggml_type t) {
    return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_TURBO4_0;
}

// Padded landmark width: Q8_0 -> round32, Turbo4_0 -> round128.
static bool xb_padded_dim(uint32_t dim, enum ggml_type lt, uint32_t * out) {
    if (dim == 0 || !out) return false;
    if (lt == GGML_TYPE_Q8_0) {
        uint64_t p = ((uint64_t)dim + 31) / 32 * 32;
        if (p == 0 || p > UINT32_MAX) return false;
        *out = (uint32_t)p;
        return true;
    }
    if (lt == GGML_TYPE_TURBO4_0) {
        uint64_t p = ((uint64_t)dim + 127) / 128 * 128;
        if (p == 0 || p > UINT32_MAX) return false;
        *out = (uint32_t)p;
        return true;
    }
    return false;
}

static bool xb_check_params(const ggml_xkv_landmark_build_params * p, char * err, size_t n) {
    if (!p) { xb_err(err, n, "xkv lmbuild: null params"); return false; }
    if (p->version != GGML_XKV_LANDMARK_BUILD_VERSION) { xb_err(err, n, "xkv lmbuild: bad version"); return false; }
    if (p->n_rows == 0 || p->chunk_tokens == 0 || p->n_chunks == 0) { xb_err(err, n, "xkv lmbuild: empty rows/chunks"); return false; }
    uint64_t nc = ((uint64_t)p->n_rows + p->chunk_tokens - 1) / p->chunk_tokens;
    if (nc != p->n_chunks) { xb_err(err, n, "xkv lmbuild: n_chunks != ceil(n_rows/chunk)"); return false; }
    if (p->total_dim == 0 || p->total_dim > 4096) { xb_err(err, n, "xkv lmbuild: bad total_dim"); return false; }
    if (p->rank == 0 || p->pad_rank < p->rank || p->pad_rank > 2048) { xb_err(err, n, "xkv lmbuild: bad rank"); return false; }
    if (p->n_layers == 0 || p->n_layers > 128) { xb_err(err, n, "xkv lmbuild: bad n_layers"); return false; }
    if (!xb_landmark_ok((enum ggml_type)p->landmark_type)) { xb_err(err, n, "xkv lmbuild: landmark must be Q8_0/Turbo4_0"); return false; }
    if (!xb_codec_ok((enum ggml_type)p->a_type) || !xb_codec_ok((enum ggml_type)p->b_type)) { xb_err(err, n, "xkv lmbuild: unsupported factor codec"); return false; }
    if (xb_is_turbo((enum ggml_type)p->a_type) || xb_is_turbo((enum ggml_type)p->b_type)) {
        if (p->a_type != p->b_type) { xb_err(err, n, "xkv lmbuild: turbo K pair must match"); return false; }
        if (p->pad_rank % 128 != 0) { xb_err(err, n, "xkv lmbuild: turbo pad_rank %128"); return false; }
    }
    if ((enum ggml_type)p->a_type == GGML_TYPE_Q8_0 || (enum ggml_type)p->b_type == GGML_TYPE_Q8_0) {
        // Q8 rows decode per-32 blocks; padded rank must be a whole block count.
        if (p->pad_rank % 32 != 0) { xb_err(err, n, "xkv lmbuild: q8 pad_rank %32"); return false; }
    }
    uint32_t pd = 0;
    if (!xb_padded_dim(p->total_dim, (enum ggml_type)p->landmark_type, &pd) || pd != p->padded_dim) {
        xb_err(err, n, "xkv lmbuild: padded_dim mismatch"); return false;
    }
    return true;
}

// Canonical decode of one packed row (P floats). Turbo via CANONICAL domain.
static bool xb_decode_row(enum ggml_type t, const uint8_t * src_row, float * dst,
        uint32_t P, uint8_t * tmp, size_t tmp_bytes, char * err, size_t esz) {
    size_t rb = ggml_row_size(t, (int64_t)P);
    if (rb == 0) { xb_err(err, esz, "xkv lmbuild: row size"); return false; }
    if (t == GGML_TYPE_F32) {
        memcpy(dst, src_row, (size_t)P * sizeof(float));
        return true;
    }
    if (t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0) {
        const struct ggml_type_traits * tr = ggml_get_type_traits(t);
        if (!tr || !tr->to_float) { xb_err(err, esz, "xkv lmbuild: missing to_float"); return false; }
        if (!tmp || tmp_bytes < rb) { xb_err(err, esz, "xkv lmbuild: decode tmp short"); return false; }
        memcpy(tmp, src_row, rb);
        tr->to_float(tmp, dst, (int64_t)P);
        return true;
    }
    if (xb_is_turbo(t)) {
        if (!ggml_dequantize_turbo_row(t, src_row, dst, (int64_t)P, 128, GGML_TURBO_DECODE_CANONICAL)) {
            xb_err(err, esz, "xkv lmbuild: turbo decode failed"); return false;
        }
        return true;
    }
    xb_err(err, esz, "xkv lmbuild: unsupported decode type");
    return false;
}

// Forward RoPE per head: HALF front-back pairs, INTERLEAVED adjacent pairs,
// mag = direct multiplier, tail passthrough. Matches xkv_rope_apply.
static void xb_rope_head(float * v, uint32_t head_dim, uint32_t rotary_dim,
        uint32_t mode, int64_t pos, const float * omega, const float * mag) {
    if (rotary_dim == 0 || rotary_dim > head_dim || !omega) return;
    uint32_t fc = rotary_dim / 2;
    float fpos = (float)pos;
    if (mode == 0) {
        for (uint32_t f = 0; f < fc; ++f) {
            float ang = omega[f] * fpos;
            float c = cosf(ang), s = sinf(ang);
            float mg = mag ? mag[f] : 1.0f;
            float re = v[f], im = v[f + fc];
            v[f]      = (re * c - im * s) * mg;
            v[f + fc] = (re * s + im * c) * mg;
        }
    } else {
        for (uint32_t f = 0; f < fc; ++f) {
            float ang = omega[f] * fpos;
            float c = cosf(ang), s = sinf(ang);
            float mg = mag ? mag[f] : 1.0f;
            float re = v[2 * f], im = v[2 * f + 1];
            v[2 * f]     = (re * c - im * s) * mg;
            v[2 * f + 1] = (re * s + im * c) * mg;
        }
    }
}

// Max B-tile feature width across layers (for scratch sizing / validation).
static bool xb_max_dim(const int32_t * lm, int64_t n_layers, uint32_t * out, char * err, size_t esz) {
    uint32_t mx = 0;
    int64_t run = 0;
    for (int64_t i = 0; i < n_layers; ++i) {
        const int32_t * L = lm + i * GGML_XKV_LANDMARK_BUILD_LAYER_META_STRIDE;
        int64_t off = L[0], dim = L[1], hd = L[2], rd = L[3], mode = L[4], ro = L[5];
        if (dim <= 0 || dim > 4096) { xb_err(err, esz, "xkv lmbuild: bad layer dim"); return false; }
        if (off != run) { xb_err(err, esz, "xkv lmbuild: slices must be contiguous from 0"); return false; }
        uint32_t ehd = hd <= 0 ? (uint32_t)dim : (uint32_t)hd;
        if (ehd == 0 || (uint32_t)dim % ehd != 0) { xb_err(err, esz, "xkv lmbuild: dim % head"); return false; }
        if (rd < 0 || rd > (int64_t)ehd || (rd & 1)) { xb_err(err, esz, "xkv lmbuild: bad rotary_dim"); return false; }
        if (mode < 0 || mode > 1) { xb_err(err, esz, "xkv lmbuild: bad rope_mode"); return false; }
        if (ro < 0) { xb_err(err, esz, "xkv lmbuild: bad rope_offset"); return false; }
        run += dim;
        if (run > 4096) { xb_err(err, esz, "xkv lmbuild: total width overflow"); return false; }
        if ((uint32_t)dim > mx) mx = (uint32_t)dim;
    }
    if (run == 0) { xb_err(err, esz, "xkv lmbuild: empty width"); return false; }
    *out = mx;
    return true;
}

bool ggml_xkv_landmark_build_scratch_bytes(
        const ggml_xkv_landmark_build_params * p,
        size_t * out_bytes, char * err, size_t err_size) {
    if (!p || !out_bytes) { xb_err(err, err_size, "xkv lmbuild: null pointer"); return false; }
    if (!xb_check_params(p, err, err_size)) return false;
    uint32_t max_feature_dim = p->max_feature_dim ? p->max_feature_dim : p->total_dim;
    if (max_feature_dim == 0 || max_feature_dim > p->total_dim) { xb_err(err, err_size, "xkv lmbuild: bad max_feature_dim"); return false; }
    // floats: tile(maxDim*P) + arow(P) + recon/ phased/acc/mean (4*D) + dec(pad_D)
    size_t tile = 0, f4 = 0, floats = 0;
    if (!xb_mul_u32(max_feature_dim, p->pad_rank, &tile)) { xb_err(err, err_size, "xkv lmbuild: scratch overflow"); return false; }
    if (!xb_mul((size_t)p->total_dim, 4, &f4)) { xb_err(err, err_size, "xkv lmbuild: scratch overflow"); return false; }
    if (!xb_add(tile, p->pad_rank, &floats) || !xb_add(floats, f4, &floats) || !xb_add(floats, p->padded_dim, &floats)) {
        xb_err(err, err_size, "xkv lmbuild: scratch overflow"); return false;
    }
    size_t fbytes = 0;
    if (!xb_mul(floats, sizeof(float), &fbytes)) { xb_err(err, err_size, "xkv lmbuild: scratch overflow"); return false; }
    // tmp: worst packed row among A_K / B_K (F16/Q8 staging; turbo decodes direct)
    size_t ra = ggml_row_size((enum ggml_type)p->a_type, (int64_t)p->pad_rank);
    size_t rb = ggml_row_size((enum ggml_type)p->b_type, (int64_t)p->pad_rank);
    if (ra == 0 || rb == 0) { xb_err(err, err_size, "xkv lmbuild: row size"); return false; }
    size_t tmp = ra > rb ? ra : rb;
    size_t need = 0;
    if (!xb_add(fbytes, tmp, &need) || !xb_add(need, 160, &need)) { xb_err(err, err_size, "xkv lmbuild: scratch overflow"); return false; }
    *out_bytes = need;
    return true;
}

bool ggml_xkv_landmark_build_supports(
        const struct ggml_tensor * a_k, const struct ggml_tensor * b_k,
        const struct ggml_tensor * rows, const struct ggml_tensor * positions,
        const struct ggml_tensor * layer_meta, const struct ggml_tensor * rope_tables,
        const struct ggml_tensor * scratch, const struct ggml_tensor * eb,
        const struct ggml_tensor * srcfp, const struct ggml_tensor * status,
        const struct ggml_tensor * dst, const ggml_xkv_landmark_build_params * p,
        char * err, size_t err_size) {
    if (!a_k || !b_k || !rows || !positions || !layer_meta || !rope_tables ||
        !scratch || !eb || !srcfp || !status || !dst || !p) {
        xb_err(err, err_size, "xkv lmbuild: null tensor/params"); return false;
    }
    if (!xb_check_params(p, err, err_size)) return false;
    if (a_k->type != (enum ggml_type)p->a_type || b_k->type != (enum ggml_type)p->b_type) {
        xb_err(err, err_size, "xkv lmbuild: tensor/type param mismatch"); return false;
    }
    if (a_k->ne[0] != (int64_t)p->pad_rank || b_k->ne[0] != (int64_t)p->pad_rank) {
        xb_err(err, err_size, "xkv lmbuild: ne0 != pad_rank"); return false;
    }
    if (a_k->ne[1] < (int64_t)p->n_rows || b_k->ne[1] < (int64_t)p->total_dim) {
        xb_err(err, err_size, "xkv lmbuild: stream rows short"); return false;
    }
    size_t ra = ggml_row_size(a_k->type, a_k->ne[0]);
    size_t rb = ggml_row_size(b_k->type, b_k->ne[0]);
    if (ra == 0 || rb == 0 || a_k->nb[1] != ra || b_k->nb[1] != rb) {
        xb_err(err, err_size, "xkv lmbuild: streams must be packed contiguous"); return false;
    }
    if (rows->type != GGML_TYPE_I32 || rows->ne[0] < (int64_t)p->n_rows) {
        xb_err(err, err_size, "xkv lmbuild: rows must be I32[n_rows]"); return false;
    }
    if ((positions->type != GGML_TYPE_I32 && positions->type != GGML_TYPE_I64) ||
        positions->ne[0] < (int64_t)p->n_rows) {
        xb_err(err, err_size, "xkv lmbuild: positions must be I32/I64[n_rows]"); return false;
    }
    if (layer_meta->type != GGML_TYPE_I32 ||
        layer_meta->ne[0] != GGML_XKV_LANDMARK_BUILD_LAYER_META_STRIDE ||
        layer_meta->ne[1] < (int64_t)p->n_layers) {
        xb_err(err, err_size, "xkv lmbuild: layer_meta must be I32[6,n_layers]"); return false;
    }
    if (rope_tables->type != GGML_TYPE_F32) { xb_err(err, err_size, "xkv lmbuild: rope must be F32"); return false; }
    if (dst->type != (enum ggml_type)p->landmark_type ||
        dst->ne[0] != (int64_t)p->padded_dim || dst->ne[1] < (int64_t)p->n_chunks) {
        xb_err(err, err_size, "xkv lmbuild: dst must be [padded_dim,n_chunks]"); return false;
    }
    size_t rd = ggml_row_size(dst->type, dst->ne[0]);
    if (rd == 0 || dst->nb[1] != rd) { xb_err(err, err_size, "xkv lmbuild: dst must be packed"); return false; }
    if (eb->type != GGML_TYPE_F32 || eb->ne[0] < (int64_t)p->n_chunks) {
        xb_err(err, err_size, "xkv lmbuild: eb must be F32[n_chunks]"); return false;
    }
    if (srcfp->type != GGML_TYPE_I64 || srcfp->ne[0] < (int64_t)p->n_chunks) {
        xb_err(err, err_size, "xkv lmbuild: srcfp must be I64[n_chunks]"); return false;
    }
    if (status->type != GGML_TYPE_I32 || ggml_nelements(status) < 4) {
        xb_err(err, err_size, "xkv lmbuild: status must be I32[4]"); return false;
    }
    size_t need = 0;
    if (!ggml_xkv_landmark_build_scratch_bytes(p, &need, err, err_size)) return false;
    if (ggml_nbytes(scratch) < need) { xb_err(err, err_size, "xkv lmbuild: scratch too small"); return false; }
    return true;
}

struct ggml_tensor * ggml_xkv_landmark_build(
        struct ggml_context * ctx, struct ggml_tensor * a_k, struct ggml_tensor * b_k,
        struct ggml_tensor * rows, struct ggml_tensor * positions,
        struct ggml_tensor * layer_meta, struct ggml_tensor * rope_tables,
        struct ggml_tensor * scratch, struct ggml_tensor * eb,
        struct ggml_tensor * srcfp, struct ggml_tensor * status,
        const ggml_xkv_landmark_build_params * p) {
    if (!ctx || !a_k || !b_k || !rows || !positions || !layer_meta || !rope_tables ||
        !scratch || !eb || !srcfp || !status || !p) return nullptr;
    if (p->version != GGML_XKV_LANDMARK_BUILD_VERSION) return nullptr;
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, (enum ggml_type)p->landmark_type,
        (int64_t)p->padded_dim, (int64_t)p->n_chunks);
    if (!dst) return nullptr;
    static_assert(sizeof(ggml_xkv_landmark_build_params) <= GGML_MAX_OP_PARAMS, "lmbuild params overflow");
    memcpy(dst->op_params, p, sizeof(*p));
    dst->op = (enum ggml_op)(GGML_OP_XKV_LANDMARK_BUILD);
    dst->src[0] = a_k; dst->src[1] = b_k; dst->src[2] = rows;
    dst->src[3] = positions; dst->src[4] = layer_meta; dst->src[5] = rope_tables;
    dst->src[6] = scratch; dst->src[7] = eb; dst->src[8] = srcfp;
    dst->src[9] = status;
    return dst;
}

// ---- CPU oracle ----
bool ggml_xkv_landmark_build_cpu_oracle(
        const void * a_k_data, enum ggml_type a_k_type, int64_t a_k_ne0, int64_t a_k_ne1,
        size_t a_k_nb0, size_t a_k_nb1,
        const void * b_k_data, enum ggml_type b_k_type, int64_t b_k_ne0, int64_t b_k_ne1,
        size_t b_k_nb0, size_t b_k_nb1,
        const int32_t * rows_data, int64_t n_rows,
        const int64_t * pos_data, int pos_is_64,
        const int32_t * layer_meta_data, int64_t n_layers,
        const float * rope_data, int64_t rope_nelements,
        const ggml_xkv_landmark_build_params * p,
        void * dst_data, int64_t dst_ne0, int64_t dst_ne1, size_t dst_nb0, size_t dst_nb1,
        float * eb_data, uint64_t * fp_data, int32_t * status_data,
        void * scratch, size_t scratch_bytes,
        char * err, size_t err_size) {
    auto set_st = [&](int32_t c, int32_t n) {
        if (status_data) { status_data[0] = c; status_data[1] = n; status_data[2] = 0; status_data[3] = 0; }
    };
    if (!a_k_data || !b_k_data || !rows_data || !pos_data || !layer_meta_data ||
        !dst_data || !eb_data || !fp_data || !status_data || !scratch || !p) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
        xb_err(err, err_size, "xkv lmbuild oracle: null pointer"); return false;
    }
    // pos_data points to I32 or I64 array depending on pos_is_64.
    if (!xb_check_params(p, err, err_size)) { set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0); return false; }
    if (a_k_type != (enum ggml_type)p->a_type || b_k_type != (enum ggml_type)p->b_type) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
        xb_err(err, err_size, "xkv lmbuild oracle: type mismatch"); return false;
    }
    if (a_k_ne0 != (int64_t)p->pad_rank || b_k_ne0 != (int64_t)p->pad_rank ||
        n_rows != (int64_t)p->n_rows || n_layers != (int64_t)p->n_layers ||
        dst_ne0 != (int64_t)p->padded_dim || dst_ne1 < (int64_t)p->n_chunks) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
        xb_err(err, err_size, "xkv lmbuild oracle: shape mismatch"); return false;
    }
    if (a_k_ne1 < (int64_t)p->n_rows || b_k_ne1 < (int64_t)p->total_dim) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
        xb_err(err, err_size, "xkv lmbuild oracle: stream short"); return false;
    }
    size_t ra = ggml_row_size(a_k_type, a_k_ne0), rb = ggml_row_size(b_k_type, b_k_ne0);
    size_t rd = ggml_row_size((enum ggml_type)p->landmark_type, dst_ne0);
    if (ra == 0 || rb == 0 || rd == 0 || a_k_nb1 != ra || b_k_nb1 != rb || dst_nb1 != rd) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
        xb_err(err, err_size, "xkv lmbuild oracle: stride mismatch"); return false;
    }
    (void)a_k_nb0; (void)b_k_nb0; (void)dst_nb0;
    uint32_t mx = 0;
    {
        char e2[256] = {0};
        // validate slices + width + rope offsets against rope_nelements
        int64_t run = 0;
        for (int64_t i = 0; i < n_layers; ++i) {
            const int32_t * L = layer_meta_data + (size_t)i * 6;
            if (L[0] != run || L[1] <= 0) {
                set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
                xb_err(err, err_size, "xkv lmbuild oracle: layer slices"); return false;
            }
            uint32_t ehd = L[2] <= 0 ? (uint32_t)L[1] : (uint32_t)L[2];
            if (ehd == 0 || (uint32_t)L[1] % ehd != 0 || L[3] < 0 || L[3] > (int32_t)ehd || (L[3] & 1)) {
                set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
                xb_err(err, err_size, "xkv lmbuild oracle: bad head/rotary"); return false;
            }
            if (L[4] < 0 || L[4] > 1 || L[5] < 0) {
                set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
                xb_err(err, err_size, "xkv lmbuild oracle: bad rope meta"); return false;
            }
            if (L[3] > 0) {
                int64_t need = (int64_t)L[5] + L[3];
                if (!rope_data || rope_nelements < need) {
                    set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
                    xb_err(err, err_size, "xkv lmbuild oracle: rope short"); return false;
                }
            }
            run += L[1];
            if ((uint32_t)L[1] > mx) mx = (uint32_t)L[1];
        }
        if (run != (int64_t)p->total_dim) {
            set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
            xb_err(err, err_size, "xkv lmbuild oracle: width != D"); return false;
        }
        (void)e2;
    }
    size_t need = 0;
    if (!ggml_xkv_landmark_build_scratch_bytes(p, &need, err, err_size)) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0); return false;
    }
    if (scratch_bytes < need) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_WORKSPACE, 0);
        xb_err(err, err_size, "xkv lmbuild oracle: scratch short"); return false;
    }
    // Carve: tile(mx*P) | arow(P) | recon(D) | phased(D) | acc(D) | mean(D) | dec(pad_D) | tmp
    uint8_t * base = (uint8_t *)scratch;
    size_t off = 0;
    auto take = [&](size_t bytes, size_t align) -> void * {
        size_t a = (off + align - 1) & ~(align - 1);
        if (a > scratch_bytes || bytes > scratch_bytes - a) return nullptr;
        void * r = base + a;
        off = a + bytes;
        return r;
    };
    const uint32_t P = p->pad_rank, D = p->total_dim, PD = p->padded_dim;
    float * tile   = (float *)take((size_t)mx * P * sizeof(float), 16);
    float * arow   = (float *)take((size_t)P * sizeof(float), 16);
    float * recon  = (float *)take((size_t)D * sizeof(float), 16);
    float * phased = (float *)take((size_t)D * sizeof(float), 16);
    float * acc    = (float *)take((size_t)D * sizeof(float), 16);
    float * mean   = (float *)take((size_t)D * sizeof(float), 16);
    float * dec    = (float *)take((size_t)PD * sizeof(float), 16);
    size_t tmp_sz = scratch_bytes > off ? scratch_bytes - off : 0;
    uint8_t * tmp = (uint8_t *)take(tmp_sz, 64);
    if (!tile || !arow || !recon || !phased || !acc || !mean || !dec || (!tmp && tmp_sz)) {
        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_WORKSPACE, 0);
        xb_err(err, err_size, "xkv lmbuild oracle: carve failed"); return false;
    }
    // rows/positions bounds
    for (int64_t i = 0; i < n_rows; ++i) {
        if (rows_data[i] < 0 || (int64_t)rows_data[i] >= a_k_ne1) {
            set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
            xb_err(err, err_size, "xkv lmbuild oracle: row oob"); return false;
        }
    }
    const uint8_t * ab = (const uint8_t *)a_k_data;
    const uint8_t * bb = (const uint8_t *)b_k_data;
    uint8_t * db = (uint8_t *)dst_data;
    const int32_t * pos32 = (const int32_t *)pos_data;
    enum ggml_type lt = (enum ggml_type)p->landmark_type;
    const struct ggml_type_traits * q8tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
    uint64_t phase_seed = ((uint64_t)p->phase_hi << 32) | p->phase_lo;

    for (uint32_t c = 0; c < p->n_chunks; ++c) {
        size_t r0 = (size_t)c * p->chunk_tokens;
        size_t r1 = r0 + p->chunk_tokens;
        if (r1 > (size_t)p->n_rows) r1 = p->n_rows;
        for (uint32_t d = 0; d < D; ++d) acc[d] = 0.0f;
        for (size_t ri = r0; ri < r1; ++ri) {
            int32_t ar = rows_data[ri];
            int64_t pos = pos_is_64 ? pos_data[ri] : (int64_t)pos32[ri];
            if (!xb_decode_row(a_k_type, ab + (size_t)ar * ra, arow, P, tmp, tmp_sz, err, err_size)) {
                set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0); return false;
            }
            for (int64_t li = 0; li < n_layers; ++li) {
                const int32_t * L = layer_meta_data + (size_t)li * 6;
                uint32_t foff = (uint32_t)L[0], fdim = (uint32_t)L[1];
                uint32_t ehd = L[2] <= 0 ? fdim : (uint32_t)L[2];
                uint32_t rdim = (uint32_t)L[3];
                uint32_t mode = (uint32_t)L[4];
                uint32_t roff = (uint32_t)L[5];
                for (uint32_t d = 0; d < fdim; ++d) {
                    if (!xb_decode_row(b_k_type, bb + ((size_t)foff + d) * rb,
                            tile + (size_t)d * P, P, tmp, tmp_sz, err, err_size)) {
                        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0); return false;
                    }
                }
                for (uint32_t d = 0; d < fdim; ++d) {
                    const float * bf = tile + (size_t)d * P;
                    double s = 0.0;
                    for (uint32_t k = 0; k < p->rank; ++k) s += (double)arow[k] * (double)bf[k];
                    recon[foff + d] = (float)s;
                }
                // forward-phase per head at this row's own position
                uint32_t nh = fdim / ehd;
                const float * om = nullptr, * mg = nullptr;
                if (rdim > 0) {
                    if (!rope_data) {
                        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
                        xb_err(err, err_size, "xkv lmbuild oracle: rope missing"); return false;
                    }
                    om = rope_data + roff;
                    mg = rope_data + roff + rdim / 2;
                    // bounds already checked per layer above
                }
                for (uint32_t h = 0; h < nh; ++h) {
                    float * rp = recon + foff + (size_t)h * ehd;
                    float * pp = phased + foff + (size_t)h * ehd;
                    if (rdim == 0) {
                        for (uint32_t j = 0; j < ehd; ++j) pp[j] = rp[j];
                    } else {
                        for (uint32_t j = 0; j < ehd; ++j) pp[j] = rp[j];
                        xb_rope_head(pp, ehd, rdim, mode, pos, om, mg);
                    }
                }
                for (uint32_t d = 0; d < fdim; ++d) {
                    float v = phased[foff + d];
                    if (!std::isfinite(v)) {
                        set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT, 0);
                        xb_err(err, err_size, "xkv lmbuild oracle: non-finite"); return false;
                    }
                    acc[foff + d] += v;
                }
            }
        }
        float inv = 1.0f / (float)(r1 - r0);
        for (uint32_t d = 0; d < D; ++d) mean[d] = acc[d] * inv;
        uint8_t * out = db + (size_t)c * rd;
        if (lt == GGML_TYPE_Q8_0) {
            if (!q8tr || !q8tr->from_float_ref || !q8tr->to_float) {
                set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_UNSUPPORTED, 0);
                xb_err(err, err_size, "xkv lmbuild oracle: q8 traits"); return false;
            }
            if (ggml_blck_size(GGML_TYPE_Q8_0) != 32) {
                set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_UNSUPPORTED, 0);
                xb_err(err, err_size, "xkv lmbuild oracle: q8 block"); return false;
            }
            size_t blk_bytes = ggml_row_size(GGML_TYPE_Q8_0, 32);
            uint32_t nb = PD / 32;
            float inb[32], dcb[32];
            uint8_t qb[64];
            for (uint32_t b = 0; b < nb; ++b) {
                for (uint32_t j = 0; j < 32; ++j) {
                    uint32_t idx = b * 32 + j;
                    inb[j] = idx < D ? mean[idx] : 0.0f;
                }
                q8tr->from_float_ref(inb, qb, 32);
                memcpy(out + (size_t)b * blk_bytes, qb, blk_bytes);
            }
            // authoritative: decode just-written bytes, compare over logical D
            double max_abs = 0.0;
            for (uint32_t b = 0; b < nb; ++b) {
                q8tr->to_float(out + (size_t)b * blk_bytes, dcb, 32);
                for (uint32_t j = 0; j < 32; ++j) {
                    uint32_t idx = b * 32 + j;
                    if (idx >= D) break;
                    max_abs = std::max(max_abs, std::fabs((double)dcb[j] - (double)mean[idx]));
                }
            }
            const double raw = std::sqrt((double) D) * max_abs;
            float ebv = std::nextafter((float) (raw * (1.0 + 8.0 * FLT_EPSILON) + 1e-6),
                                       std::numeric_limits<float>::infinity());
            if (!(ebv >= 1e-6f) || !std::isfinite(ebv)) ebv = 1e-4f;
            eb_data[c] = ebv;
        } else if (lt == GGML_TYPE_TURBO4_0) {
            size_t gb = ggml_row_size(GGML_TYPE_TURBO4_0, 128);
            if (gb == 0 || PD % 128 != 0) {
                set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_UNSUPPORTED, 0);
                xb_err(err, err_size, "xkv lmbuild oracle: turbo group"); return false;
            }
            uint32_t ng = PD / 128;
            float ing[128], dcg[128];
            for (uint32_t g = 0; g < ng; ++g) {
                for (uint32_t j = 0; j < 128; ++j) {
                    uint32_t idx = g * 128 + j;
                    ing[j] = idx < D ? mean[idx] : 0.0f;
                }
                if (!ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, ing, out + (size_t)g * gb, 128, 128)) {
                    set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_UNSUPPORTED, 0);
                    xb_err(err, err_size, "xkv lmbuild oracle: turbo quant"); return false;
                }
            }
            double max_abs = 0.0;
            for (uint32_t g = 0; g < ng; ++g) {
                if (!ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, out + (size_t)g * gb, dcg,
                        128, 128, GGML_TURBO_DECODE_CANONICAL)) {
                    set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_UNSUPPORTED, 0);
                    xb_err(err, err_size, "xkv lmbuild oracle: turbo bound"); return false;
                }
                for (uint32_t j = 0; j < 128; ++j) {
                    uint32_t idx = g * 128 + j;
                    if (idx >= D) break;
                    max_abs = std::max(max_abs, std::fabs((double)dcg[j] - (double)mean[idx]));
                }
            }
            const double raw = std::sqrt((double) D) * max_abs;
            float ebv = std::nextafter((float) (raw * (1.0 + 8.0 * FLT_EPSILON) + 1e-6),
                                       std::numeric_limits<float>::infinity());
            if (!(ebv >= 1e-6f) || !std::isfinite(ebv)) ebv = 1e-4f;
            eb_data[c] = ebv;
        } else {
            set_st(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_UNSUPPORTED, 0);
            xb_err(err, err_size, "xkv lmbuild oracle: bad landmark type"); return false;
        }
        uint64_t src_fp = UINT64_C(0xcbf29ce484222325);
        src_fp = xb_fnv_u32(src_fp, p->seed);
        src_fp = xb_fnv_u32(src_fp, (uint32_t) phase_seed);
        src_fp = xb_fnv_u32(src_fp, (uint32_t) (phase_seed >> 32));
        src_fp = xb_fnv_u32(src_fp, c);
        for (size_t ri = r0; ri < r1; ++ri) {
            src_fp = xb_fnv_u32(src_fp, (uint32_t) rows_data[ri]);
            const int64_t pos = pos_is_64 ? pos_data[ri] : (int64_t) pos32[ri];
            src_fp = xb_fnv_u32(src_fp, (uint32_t) pos);
            src_fp = xb_fnv_u32(src_fp, (uint32_t) ((uint64_t) pos >> 32));
        }
        for (size_t i = 0; i < rd; ++i) src_fp = xb_fnv_byte(src_fp, out[i]);
        fp_data[c] = src_fp != 0 ? src_fp : 1;
    }
    set_st(GGML_XKV_LANDMARK_BUILD_STATUS_OK, (int32_t)p->n_chunks);
    return true;
}
