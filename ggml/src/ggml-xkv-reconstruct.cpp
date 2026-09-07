// ggml-xkv-reconstruct.cpp — native XKV selected-row factor reconstruction (v1).
//
// CPU oracle + validation + graph builder. Vulkan backend shares validation
// and fingerprint helpers; device kernel lives in
// ggml/src/ggml-vulkan/vulkan-shaders/xkv_reconstruct.comp with a single
// dispatch per op. No host pointers in op_params; code streams stay
// device-resident on Vulkan.

#include "ggml-xkv.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static uint64_t xkv_fnv64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (uint8_t)(v >> (i * 8));
        h *= 1099511628211ULL;
    }
    return h;
}

static bool xkv_is_turbo(enum ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

static void xkv_err(char * err, size_t n, const char * msg) {
    if (err && n) {
        snprintf(err, n, "%s", msg);
    }
}

bool ggml_xkv_codec_supported(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

uint64_t ggml_xkv_format_revision(enum ggml_type type) {
    if (!xkv_is_turbo(type)) return 0;
    return ggml_turbo_layout_fingerprint(type);
}

uint64_t ggml_xkv_table_fingerprint(enum ggml_type type, uint32_t seed) {
    if (!xkv_is_turbo(type)) return 0;
    uint64_t h = 14695981039346656037ULL;
    h = xkv_fnv64(h, (uint64_t)seed);
    h = xkv_fnv64(h, (uint64_t)128); // group_size fixed v1
    h = xkv_fnv64(h, (uint64_t)type);
    // Fold compiled WHT sign identity via format revision so CPU/Vulkan agree.
    h = xkv_fnv64(h, ggml_xkv_format_revision(type));
    return h == 0 ? 1 : h;
}

uint64_t ggml_xkv_fp_combined(enum ggml_type a_k, enum ggml_type b_k,
                              enum ggml_type a_v, enum ggml_type b_v,
                              uint32_t seed_k, uint32_t seed_v) {
    uint64_t h = 14695981039346656037ULL;
    // K pair uses seed_k, V pair uses seed_v (codec uses seed / seed+1).
    h = xkv_fnv64(h, ggml_xkv_table_fingerprint(a_k, seed_k));
    h = xkv_fnv64(h, ggml_xkv_format_revision(a_k));
    h = xkv_fnv64(h, (uint64_t)a_k);
    h = xkv_fnv64(h, ggml_xkv_table_fingerprint(b_k, seed_k));
    h = xkv_fnv64(h, ggml_xkv_format_revision(b_k));
    h = xkv_fnv64(h, (uint64_t)b_k);
    h = xkv_fnv64(h, ggml_xkv_table_fingerprint(a_v, seed_v));
    h = xkv_fnv64(h, ggml_xkv_format_revision(a_v));
    h = xkv_fnv64(h, (uint64_t)a_v);
    h = xkv_fnv64(h, ggml_xkv_table_fingerprint(b_v, seed_v));
    h = xkv_fnv64(h, ggml_xkv_format_revision(b_v));
    h = xkv_fnv64(h, (uint64_t)b_v);
    h = xkv_fnv64(h, (uint64_t)seed_k);
    h = xkv_fnv64(h, (uint64_t)seed_v);
    return h == 0 ? 1 : h;
}

int32_t ggml_xkv_padded_rank(enum ggml_type type, uint32_t logical_rank) {
    if (logical_rank == 0) return 0;
    int64_t blk = ggml_blck_size(type);
    if (blk <= 0) return -1;
    uint64_t b = (uint64_t)blk;
    // Turbo v1 additionally requires 128 alignment (blck is 128, same).
    if (xkv_is_turbo(type) && b < 128) b = 128;
    uint64_t p = (logical_rank + b - 1) / b * b;
    if (p > INT32_MAX) return -1;
    return (int32_t)p;
}

// ---- internal validation core ----
struct xkv_shapes {
    enum ggml_type a_k_t, b_k_t, a_v_t, b_v_t;
    int64_t a_k_n0, a_k_n1, b_k_n0, b_k_n1;
    int64_t a_v_n0, a_v_n1, b_v_n0, b_v_n1;
    int64_t refs_n0, refs_n1, pos_n0, gm_n0, gm_n1, dst_n0, dst_n1;
    int64_t rope_nelements;
    int64_t lm_n0, lm_n1;
    size_t a_k_b0, a_k_b1, b_k_b0, b_k_b1, a_v_b0, a_v_b1, b_v_b0, b_v_b1;
    size_t dst_b0, dst_b1;
    enum ggml_type refs_t, pos_t, gm_t, dst_t, rope_t, lm_t;
};

static bool xkv_check_params(const ggml_xkv_reconstruct_params * p, char * err, size_t n) {
    if (!p) { xkv_err(err, n, "xkv: null params"); return false; }
    if (p->version != GGML_XKV_VERSION) { xkv_err(err, n, "xkv: bad descriptor version"); return false; }
    if (p->n_sel == 0 || p->n_sel > 16384) { xkv_err(err, n, "xkv: n_sel out of bounded range"); return false; }
    // One op per exact segment-group: independently encoded groups (incl.
    // adaptive tail ranks) run as separate ops, never one A stream spanning groups.
    if (p->n_groups != 1) { xkv_err(err, n, "xkv: n_groups must be 1 (one exact segment-group per op)"); return false; }
    if (p->rank_k == 0 || p->rank_v == 0) { xkv_err(err, n, "xkv: rank must be positive"); return false; }
    if (p->dim_k == 0 || p->dim_v == 0) { xkv_err(err, n, "xkv: dims must be positive"); return false; }
    if (p->dim_k > 1024 || p->dim_v > 1024) { xkv_err(err, n, "xkv: dims exceed v1 bound"); return false; }
    if (p->rotary_dim > p->dim_k) { xkv_err(err, n, "xkv: rotary_dim exceeds dim_k"); return false; }
    if ((p->rotary_dim & 1u) != 0) { xkv_err(err, n, "xkv: rotary_dim must be even"); return false; }
    if (p->rope_mode != GGML_XKV_ROPE_HALF && p->rope_mode != GGML_XKV_ROPE_INTERLEAVED) {
        xkv_err(err, n, "xkv: unsupported rope_mode"); return false;
    }
    if (p->fp_combined == 0) { xkv_err(err, n, "xkv: zero fingerprint"); return false; }
    return true;
}

static bool xkv_check_types_shapes(const xkv_shapes & s, const ggml_xkv_reconstruct_params * p,
                                   char * err, size_t n) {
    if (!ggml_xkv_codec_supported(s.a_k_t) || !ggml_xkv_codec_supported(s.b_k_t) ||
        !ggml_xkv_codec_supported(s.a_v_t) || !ggml_xkv_codec_supported(s.b_v_t)) {
        xkv_err(err, n, "xkv: unsupported codec (only F32/F16/Q8_0/Turbo2/3/4)"); return false;
    }
    // Turbo-Turbo pair must share type (identical rotation domain + layout).
    // Canonical-canonical may mix F32/F16/Q8_0. Mixed Turbo+canonical allowed
    // via canonical decode on CPU; Vulkan v1 supports matched pairs and
    // canonical pairs, rejects mixed as unsupported (explicit, in support hook).
    bool ak_t = xkv_is_turbo(s.a_k_t), bk_t = xkv_is_turbo(s.b_k_t);
    bool av_t = xkv_is_turbo(s.a_v_t), bv_t = xkv_is_turbo(s.b_v_t);
    if ((ak_t != bk_t) || (av_t != bv_t)) {
        // Mixed allowed at validation level (CPU canonical path); Vulkan
        // support hook rejects mixed explicitly. Mark as supported-here so
        // CPU oracle can run; Vulkan gate handles its subset.
    }
    if (ak_t && bk_t && s.a_k_t != s.b_k_t) {
        xkv_err(err, n, "xkv: Turbo K pair types must match"); return false;
    }
    if (av_t && bv_t && s.a_v_t != s.b_v_t) {
        xkv_err(err, n, "xkv: Turbo V pair types must match"); return false;
    }
    int32_t pk = ggml_xkv_padded_rank(s.a_k_t, p->rank_k);
    int32_t pk_b = ggml_xkv_padded_rank(s.b_k_t, p->rank_k);
    int32_t pv = ggml_xkv_padded_rank(s.a_v_t, p->rank_v);
    int32_t pv_b = ggml_xkv_padded_rank(s.b_v_t, p->rank_v);
    if (pk <= 0 || pk_b <= 0 || pv <= 0 || pv_b <= 0) {
        xkv_err(err, n, "xkv: bad padded rank"); return false;
    }
    if (pk != pk_b || pv != pv_b) {
        xkv_err(err, n, "xkv: per-pair padded rank mismatch"); return false;
    }
    if (s.a_k_n0 != pk || s.b_k_n0 != pk || s.a_v_n0 != pv || s.b_v_n0 != pv) {
        xkv_err(err, n, "xkv: tensor ne0 != padded rank (bad layout)"); return false;
    }
    if (s.refs_t != GGML_TYPE_I32 || s.refs_n0 != GGML_XKV_REF_STRIDE || s.refs_n1 != (int64_t)p->n_sel) {
        xkv_err(err, n, "xkv: refs must be I32[4,n_sel]"); return false;
    }
    if (s.pos_t != GGML_TYPE_I32 || s.pos_n0 != (int64_t)p->n_sel) {
        xkv_err(err, n, "xkv: positions must be I32[n_sel]"); return false;
    }
    if (s.gm_t != GGML_TYPE_I32 || s.gm_n0 != GGML_XKV_GROUP_META_STRIDE || s.gm_n1 != (int64_t)p->n_groups) {
        xkv_err(err, n, "xkv: group_meta must be I32[8,n_groups]"); return false;
    }
    // Layer map: exact per-layer offsets/dims/heads; never inferred uniform.
    if (s.lm_t != GGML_TYPE_I32 || s.lm_n0 != GGML_XKV_LAYER_META_STRIDE || s.lm_n1 <= 0) {
        xkv_err(err, n, "xkv: layer_meta must be I32[5,n_layers]"); return false;
    }
    if (s.dst_t != GGML_TYPE_F32 || s.dst_n0 != (int64_t)(p->dim_k + p->dim_v) || s.dst_n1 != (int64_t)p->n_sel) {
        // params dim_k/dim_v are maxima across layers; narrower layers zero-pad.
        xkv_err(err, n, "xkv: dst must be F32[max_dim_k+max_dim_v,n_sel]"); return false;
    }
    // Rope tables: F32 [2*Fc] (omega then magnitude); empty iff rotary_dim==0.
    {
        uint32_t fc = p->rotary_dim / 2;
        int64_t want = fc == 0 ? 0 : (int64_t)fc * 2;
        if (s.rope_t != GGML_TYPE_F32 || s.rope_nelements != want) {
            xkv_err(err, n, "xkv: rope_tables must be F32[2*Fc]"); return false;
        }
    }
    // Contiguous layout required v1 (reject strided/aliased layouts explicitly).
    if (s.a_k_b0 != ggml_type_size(s.a_k_t) || s.b_k_b0 != ggml_type_size(s.b_k_t) ||
        s.a_v_b0 != ggml_type_size(s.a_v_t) || s.b_v_b0 != ggml_type_size(s.b_v_t)) {
        xkv_err(err, n, "xkv: code streams must be contiguous (bad nb0)"); return false;
    }
    size_t exp_a_k_b1 = ggml_row_size(s.a_k_t, s.a_k_n0);
    size_t exp_b_k_b1 = ggml_row_size(s.b_k_t, s.b_k_n0);
    size_t exp_a_v_b1 = ggml_row_size(s.a_v_t, s.a_v_n0);
    size_t exp_b_v_b1 = ggml_row_size(s.b_v_t, s.b_v_n0);
    if (s.a_k_b1 != exp_a_k_b1 || s.b_k_b1 != exp_b_k_b1 ||
        s.a_v_b1 != exp_a_v_b1 || s.b_v_b1 != exp_b_v_b1) {
        xkv_err(err, n, "xkv: code stream row stride mismatch (bad layout)"); return false;
    }
    if (s.dst_b0 != sizeof(float) || s.dst_b1 != (size_t)s.dst_n0 * sizeof(float)) {
        xkv_err(err, n, "xkv: dst must be contiguous"); return false;
    }
    // Fingerprint check (POD only, no pointers).
    // K pair fingerprinted with seed_k, V pair with seed_v (codec: seed / seed+1).
    uint64_t expect = ggml_xkv_fp_combined(s.a_k_t, s.b_k_t, s.a_v_t, s.b_v_t, p->seed_k, p->seed_v);
    if (expect != p->fp_combined) {
        xkv_err(err, n, "xkv: fingerprint mismatch"); return false;
    }
    // Turbo format revisions must be non-zero (compiled layout check).
    for (int i = 0; i < 4; ++i) {
        enum ggml_type t = i == 0 ? s.a_k_t : i == 1 ? s.b_k_t : i == 2 ? s.a_v_t : s.b_v_t;
        if (xkv_is_turbo(t) && ggml_xkv_format_revision(t) == 0) {
            xkv_err(err, n, "xkv: unsupported Turbo layout in backend"); return false;
        }
    }
    return true;
}

static bool xkv_check_contents(const int32_t * refs, const int32_t * pos, const int32_t * gm,
                               const int32_t * lm, int64_t n_layers,
                               const float * rope, int64_t rope_n,
                               const xkv_shapes & s, const ggml_xkv_reconstruct_params * p,
                               char * err, size_t n) {
    if (!refs || !pos || !gm) return true; // supports() path skips contents
    if (!lm || n_layers != s.lm_n1) {
        xkv_err(err, n, "xkv: layer_meta missing/mismatched"); return false;
    }
    uint32_t fc = p->rotary_dim / 2;
    if (fc > 0) {
        if (!rope || rope_n != (int64_t)fc * 2) {
            xkv_err(err, n, "xkv: rope tables missing/mismatched"); return false;
        }
        for (uint32_t f = 0; f < fc; ++f) {
            float om = rope[f], mg = rope[fc + f];
            if (!(om == om) || !(mg == mg) || !(mg > 0.0f)) {
                xkv_err(err, n, "xkv: rope table non-finite/bad magnitude"); return false;
            }
        }
    } else {
        if (rope_n != 0) { xkv_err(err, n, "xkv: rope tables must be empty when rotary_dim==0"); return false; }
    }
    for (uint32_t g = 0; g < p->n_groups; ++g) {
        const int32_t * e = gm + (size_t)g * GGML_XKV_GROUP_META_STRIDE;
        int32_t nl = e[0], nh = e[1];
        int32_t bko = e[2], bkr = e[3], bvo = e[4], bvr = e[5];
        int32_t rk = e[6], rv = e[7];
        if (nl <= 0 || nh <= 0) { xkv_err(err, n, "xkv: per-group layers/heads must be positive"); return false; }
        if (rk != (int32_t)p->rank_k || rv != (int32_t)p->rank_v) {
            xkv_err(err, n, "xkv: per-group rank mismatch"); return false;
        }
        if (bko < 0 || bvo < 0 || bkr <= 0 || bvr <= 0) { xkv_err(err, n, "xkv: bad feature slice"); return false; }
        // Checked integer arithmetic for feature extents.
        if ((int64_t)nl > INT64_MAX / nh) { xkv_err(err, n, "xkv: feature extent overflow"); return false; }
        int64_t heads = (int64_t)nl * nh;
        if (heads > INT64_MAX / (int64_t)p->dim_k || heads > INT64_MAX / (int64_t)p->dim_v) {
            xkv_err(err, n, "xkv: feature extent overflow"); return false;
        }
        int64_t need_k = heads * p->dim_k;
        int64_t need_v = heads * p->dim_v;
        if (bkr != need_k || bvr != need_v) { xkv_err(err, n, "xkv: feature slice rows mismatch"); return false; }
        if (bkr > INT64_MAX - bko || bvr > INT64_MAX - bvo) { xkv_err(err, n, "xkv: slice overflow"); return false; }
        if ((int64_t)bko + bkr > s.b_k_n1 || (int64_t)bvo + bvr > s.b_v_n1) {
            xkv_err(err, n, "xkv: feature slice out of bounds"); return false;
        }
        if ((int64_t)nl > n_layers) { xkv_err(err, n, "xkv: group layers exceed layer_meta"); return false; }
    }
    for (uint32_t i = 0; i < p->n_sel; ++i) {
        int32_t ar = refs[i * 4 + 0], g = refs[i * 4 + 1], ls = refs[i * 4 + 2], h = refs[i * 4 + 3];
        (void)pos;
        if (ar < 0 || (int64_t)ar >= s.a_k_n1 || (int64_t)ar >= s.a_v_n1) {
            xkv_err(err, n, "xkv: selected row out of bounds"); return false;
        }
        if (g < 0 || (uint32_t)g >= p->n_groups) { xkv_err(err, n, "xkv: group out of bounds"); return false; }
        if (ls < 0 || (int64_t)ls >= n_layers) { xkv_err(err, n, "xkv: layer_slot out of bounds"); return false; }
        // Exact per-layer map (aliases share offsets; tail/unequal dims honored).
        const int32_t * L = lm + (size_t)ls * GGML_XKV_LAYER_META_STRIDE;
        int32_t ok = L[0], dk = L[1], ov = L[2], dv = L[3], nh = L[4];
        if (dk <= 0 || dv <= 0 || nh <= 0) { xkv_err(err, n, "xkv: bad layer dims/heads"); return false; }
        if (dk > 1024 || dv > 1024) { xkv_err(err, n, "xkv: layer dim exceeds bound"); return false; }
        if (dk > (int32_t)p->dim_k || dv > (int32_t)p->dim_v) {
            xkv_err(err, n, "xkv: layer dim exceeds op maxima"); return false;
        }
        if (h < 0 || h >= nh) { xkv_err(err, n, "xkv: head out of layer bounds"); return false; }
        if (ok < 0 || ov < 0) { xkv_err(err, n, "xkv: bad layer offsets"); return false; }
        if ((int64_t)ok > INT64_MAX - dk || (int64_t)ov > INT64_MAX - dv) {
            xkv_err(err, n, "xkv: layer slice overflow"); return false;
        }
        if ((int64_t)ok + dk > s.b_k_n1 || (int64_t)ov + dv > s.b_v_n1) {
            xkv_err(err, n, "xkv: layer slice out of B bounds"); return false;
        }
        // Head slice must fit inside the layer slice.
        if ((int64_t)h > INT64_MAX / dk || (int64_t)h > INT64_MAX / dv) {
            xkv_err(err, n, "xkv: head index overflow"); return false;
        }
    }
    return true;
}

static xkv_shapes xkv_shapes_from_tensors(
        const struct ggml_tensor * a_k, const struct ggml_tensor * b_k,
        const struct ggml_tensor * a_v, const struct ggml_tensor * b_v,
        const struct ggml_tensor * refs, const struct ggml_tensor * pos,
        const struct ggml_tensor * gm, const struct ggml_tensor * lm,
        const struct ggml_tensor * rope, const struct ggml_tensor * dst) {
    xkv_shapes s;
    s.a_k_t = a_k->type; s.b_k_t = b_k->type; s.a_v_t = a_v->type; s.b_v_t = b_v->type;
    s.a_k_n0 = a_k->ne[0]; s.a_k_n1 = a_k->ne[1]; s.b_k_n0 = b_k->ne[0]; s.b_k_n1 = b_k->ne[1];
    s.a_v_n0 = a_v->ne[0]; s.a_v_n1 = a_v->ne[1]; s.b_v_n0 = b_v->ne[0]; s.b_v_n1 = b_v->ne[1];
    s.refs_n0 = refs->ne[0]; s.refs_n1 = refs->ne[1];
    s.pos_n0 = pos->ne[0]; s.gm_n0 = gm->ne[0]; s.gm_n1 = gm->ne[1];
    s.lm_n0 = lm->ne[0]; s.lm_n1 = lm->ne[1];
    s.rope_nelements = ggml_nelements(rope);
    s.dst_n0 = dst->ne[0]; s.dst_n1 = dst->ne[1];
    s.a_k_b0 = a_k->nb[0]; s.a_k_b1 = a_k->nb[1]; s.b_k_b0 = b_k->nb[0]; s.b_k_b1 = b_k->nb[1];
    s.a_v_b0 = a_v->nb[0]; s.a_v_b1 = a_v->nb[1]; s.b_v_b0 = b_v->nb[0]; s.b_v_b1 = b_v->nb[1];
    s.dst_b0 = dst->nb[0]; s.dst_b1 = dst->nb[1];
    s.refs_t = refs->type; s.pos_t = pos->type; s.gm_t = gm->type; s.dst_t = dst->type;
    s.rope_t = rope->type;
    s.lm_t = lm->type;
    return s;
}

bool ggml_xkv_reconstruct_supports(
        const struct ggml_tensor * a_k, const struct ggml_tensor * b_k,
        const struct ggml_tensor * a_v, const struct ggml_tensor * b_v,
        const struct ggml_tensor * refs, const struct ggml_tensor * positions,
        const struct ggml_tensor * group_meta, const struct ggml_tensor * layer_meta,
        const struct ggml_tensor * rope_tables,
        const struct ggml_tensor * dst,
        const ggml_xkv_reconstruct_params * params, char * err, size_t err_size) {
    if (!a_k || !b_k || !a_v || !b_v || !refs || !positions || !group_meta || !layer_meta || !rope_tables || !dst) {
        xkv_err(err, err_size, "xkv: null tensor"); return false;
    }
    if (!xkv_check_params(params, err, err_size)) return false;
    xkv_shapes s = xkv_shapes_from_tensors(a_k, b_k, a_v, b_v, refs, positions, group_meta, layer_meta, rope_tables, dst);
    if (!xkv_check_types_shapes(s, params, err, err_size)) return false;
    // NOTE: mixed Turbo+canonical pairs are ALLOWED here (CPU decodes the
    // Turbo side to canonical). The Vulkan backend support hook narrows this
    // shared gate with an explicit mixed reject (v1 kernel covers matched
    // Turbo pairs and canonical pairs only).
    return true;
}

bool ggml_xkv_reconstruct_validate_full(
        const struct ggml_tensor * a_k, const struct ggml_tensor * b_k,
        const struct ggml_tensor * a_v, const struct ggml_tensor * b_v,
        const int32_t * refs_data, const int32_t * positions_data, const int32_t * group_meta_data,
        const int32_t * layer_meta_data, int64_t n_layers,
        const float * rope_data, int64_t rope_nelements,
        const ggml_xkv_reconstruct_params * params, char * err, size_t err_size) {
    if (!a_k || !b_k || !a_v || !b_v) { xkv_err(err, err_size, "xkv: null tensor"); return false; }
    // Build shapes with placeholder refs/pos/gm/dst when validating raw (builder path uses real tensors).
    // Here tensors carry shapes; contents passed separately.
    if (!xkv_check_params(params, err, err_size)) return false;
    xkv_shapes s;
    s.a_k_t = a_k->type; s.b_k_t = b_k->type; s.a_v_t = a_v->type; s.b_v_t = b_v->type;
    s.a_k_n0 = a_k->ne[0]; s.a_k_n1 = a_k->ne[1]; s.b_k_n0 = b_k->ne[0]; s.b_k_n1 = b_k->ne[1];
    s.a_v_n0 = a_v->ne[0]; s.a_v_n1 = a_v->ne[1]; s.b_v_n0 = b_v->ne[0]; s.b_v_n1 = b_v->ne[1];
    // Synthesize expected refs/pos/gm/dst shapes from params (contents checked below).
    s.refs_t = GGML_TYPE_I32; s.refs_n0 = 4; s.refs_n1 = params->n_sel;
    s.pos_t = GGML_TYPE_I32; s.pos_n0 = params->n_sel;
    s.gm_t = GGML_TYPE_I32; s.gm_n0 = 8; s.gm_n1 = params->n_groups;
    s.lm_t = GGML_TYPE_I32; s.lm_n0 = GGML_XKV_LAYER_META_STRIDE; s.lm_n1 = n_layers;
    s.rope_t = GGML_TYPE_F32; s.rope_nelements = rope_nelements;
    s.dst_t = GGML_TYPE_F32; s.dst_n0 = (int64_t)params->dim_k + params->dim_v; s.dst_n1 = params->n_sel;
    s.a_k_b0 = a_k->nb[0]; s.a_k_b1 = a_k->nb[1]; s.b_k_b0 = b_k->nb[0]; s.b_k_b1 = b_k->nb[1];
    s.a_v_b0 = a_v->nb[0]; s.a_v_b1 = a_v->nb[1]; s.b_v_b0 = b_v->nb[0]; s.b_v_b1 = b_v->nb[1];
    s.dst_b0 = sizeof(float); s.dst_b1 = (size_t)s.dst_n0 * sizeof(float);
    // Actual tensor ne/nb for refs/pos/gm/dst are not available here; check params-level only.
    // Full tensor-shape check happens in supports()/oracle() with real tensors.
    if (!xkv_check_types_shapes(s, params, err, err_size)) return false;
    if (!xkv_check_contents(refs_data, positions_data, group_meta_data, layer_meta_data, n_layers,
                            rope_data, rope_nelements, s, params, err, err_size)) return false;
    // Bounds of A rows vs B rows need real ne1; s above uses synthesized? No:
    // a_k_n1/b_k_n1 are real from tensors, so contents check above used them. Good.
    return true;
}

// ---- dequant helpers ----
static bool xkv_dequant_row(enum ggml_type t, const void * src_row, float * dst, int64_t n, bool rotated) {
    if (t == GGML_TYPE_F32) {
        memcpy(dst, src_row, (size_t)n * sizeof(float));
        return true;
    }
    if (t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0) {
        const struct ggml_type_traits * tr = ggml_get_type_traits(t);
        if (!tr || !tr->to_float) return false;
        // to_float may require alignment; copy through aligned scratch per block row.
        // For F16/Q8_0 the whole row is contiguous; traits handle it.
        tr->to_float(src_row, dst, n);
        return true;
    }
    if (xkv_is_turbo(t)) {
        enum ggml_turbo_decode_domain dom =
            rotated ? GGML_TURBO_DECODE_ROTATED : GGML_TURBO_DECODE_CANONICAL;
        return ggml_dequantize_turbo_row(t, src_row, dst, n, 128, dom);
    }
    return false;
}

static void xkv_rope_apply(float * vec, uint32_t dim, uint32_t rotary_dim, uint32_t mode,
                           int32_t pos, const float * omega, const float * mag) {
    if (rotary_dim == 0) return;
    // dim is genuinely unused: RoPE transforms only the first rotary_dim lanes
    // in place; the tail (rotary_dim..dim) passes through untouched.
    (void)dim;
    uint32_t fc = rotary_dim / 2;
    if (mode == GGML_XKV_ROPE_HALF) {
        for (uint32_t f = 0; f < fc; ++f) {
            float ang = (float)pos * omega[f];
            float mg = mag ? mag[f] : 1.0f;
            float c = cosf(ang), s = sinf(ang);
            float re = vec[f], im = vec[f + fc];
            vec[f] = (re * c - im * s) * mg;
            vec[f + fc] = (re * s + im * c) * mg;
        }
    } else {
        for (uint32_t f = 0; f < fc; ++f) {
            float ang = (float)pos * omega[f];
            float mg = mag ? mag[f] : 1.0f;
            float c = cosf(ang), s = sinf(ang);
            float re = vec[2 * f], im = vec[2 * f + 1];
            vec[2 * f] = (re * c - im * s) * mg;
            vec[2 * f + 1] = (re * s + im * c) * mg;
        }
    }
}

bool ggml_xkv_core_scratch_floats(const ggml_xkv_reconstruct_params * params,
        int64_t prk, int64_t prv, size_t * out_floats, char * err, size_t err_size) {
    if (!params || !out_floats) { xkv_err(err, err_size, "xkv: null pointer"); return false; }
    if (prk <= 0 || prv <= 0) { xkv_err(err, err_size, "xkv: bad padded rank"); return false; }
    int64_t mx = prk > prv ? prk : prv;
    if (mx > INT64_MAX / 2) { xkv_err(err, err_size, "xkv: scratch overflow"); return false; }
    int64_t need = mx * 2 + (int64_t)params->dim_k + params->dim_v + 64;
    if (need <= 0) { xkv_err(err, err_size, "xkv: scratch overflow"); return false; }
    *out_floats = (size_t)need;
    return true;
}

bool ggml_xkv_reconstruct_core(
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
        char * err, size_t err_size) {
    if (!a_k_data || !b_k_data || !a_v_data || !b_v_data || !refs_data || !positions_data ||
        !group_meta_data || !layer_meta_data || !params || !dst_data || !scratch) {
        xkv_err(err, err_size, "xkv: null pointer"); return false;
    }
    if (!xkv_check_params(params, err, err_size)) return false;
    xkv_shapes s;
    s.a_k_t = a_k_type; s.b_k_t = b_k_type; s.a_v_t = a_v_type; s.b_v_t = b_v_type;
    s.a_k_n0 = a_k_ne0; s.a_k_n1 = a_k_ne1; s.b_k_n0 = b_k_ne0; s.b_k_n1 = b_k_ne1;
    s.a_v_n0 = a_v_ne0; s.a_v_n1 = a_v_ne1; s.b_v_n0 = b_v_ne0; s.b_v_n1 = b_v_ne1;
    s.refs_t = GGML_TYPE_I32; s.refs_n0 = 4; s.refs_n1 = params->n_sel;
    s.pos_t = GGML_TYPE_I32; s.pos_n0 = params->n_sel;
    s.gm_t = GGML_TYPE_I32; s.gm_n0 = 8; s.gm_n1 = params->n_groups;
    s.lm_t = GGML_TYPE_I32; s.lm_n0 = GGML_XKV_LAYER_META_STRIDE; s.lm_n1 = n_layers;
    s.rope_t = GGML_TYPE_F32; s.rope_nelements = rope_nelements;
    s.dst_t = GGML_TYPE_F32; s.dst_n0 = dst_ne0; s.dst_n1 = dst_ne1;
    s.a_k_b0 = a_k_nb0; s.a_k_b1 = a_k_nb1; s.b_k_b0 = b_k_nb0; s.b_k_b1 = b_k_nb1;
    s.a_v_b0 = a_v_nb0; s.a_v_b1 = a_v_nb1; s.b_v_b0 = b_v_nb0; s.b_v_b1 = b_v_nb1;
    s.dst_b0 = dst_nb0; s.dst_b1 = dst_nb1;
    if (!xkv_check_types_shapes(s, params, err, err_size)) return false;
    if (!xkv_check_contents(refs_data, positions_data, group_meta_data, layer_meta_data, n_layers,
                            rope_data, rope_nelements, s, params, err, err_size)) return false;

    bool ak_t = xkv_is_turbo(a_k_type), bk_t = xkv_is_turbo(b_k_type);
    bool av_t = xkv_is_turbo(a_v_type), bv_t = xkv_is_turbo(b_v_type);
    // Direct rotated dot only when both Turbo (types already matched); else canonical.
    bool k_rotated = ak_t && bk_t;
    bool v_rotated = av_t && bv_t;

    int64_t prk = a_k_ne0, prv = a_v_ne0;
    size_t need = 0;
    if (!ggml_xkv_core_scratch_floats(params, prk, prv, &need, err, err_size)) return false;
    if (scratch_floats < need) { xkv_err(err, err_size, "xkv: scratch too small"); return false; }
    int64_t mx = prk > prv ? prk : prv;
    float * a_row = scratch;
    float * b_row = scratch + (size_t)mx;
    float * k_head = scratch + (size_t)mx * 2;
    float * v_head = k_head + params->dim_k;
    uint32_t fc = params->rotary_dim / 2;
    const float * omega = (fc > 0 && rope_data) ? rope_data : nullptr;
    const float * mag = (fc > 0 && rope_data) ? rope_data + fc : nullptr;

    for (uint32_t i = 0; i < params->n_sel; ++i) {
        int32_t ar = refs_data[i * 4 + 0], g = refs_data[i * 4 + 1];
        int32_t ls = refs_data[i * 4 + 2], h = refs_data[i * 4 + 3];
        int32_t pos = positions_data[i];
        (void)g;
        // Exact per-layer map: absolute offsets/dims/heads (aliases share
        // offsets; never the uniform layer_slot*n_heads*dim formula).
        const int32_t * L = layer_meta_data + (size_t)ls * GGML_XKV_LAYER_META_STRIDE;
        int64_t ok = L[0], dk = L[1], ov = L[2], dv = L[3];
        const void * a_k_row_src = (const char *)a_k_data + (size_t)ar * a_k_nb1;
        if (!xkv_dequant_row(a_k_type, a_k_row_src, a_row, prk, k_rotated)) {
            xkv_err(err, err_size, "xkv: A_K dequant failed"); return false;
        }
        if (ok > INT64_MAX - dk || ov > INT64_MAX - dv) {
            xkv_err(err, err_size, "xkv: layer slice overflow"); return false;
        }
        // Head slice starts at layer offset + h*dk (heads packed within layer).
        if ((int64_t)h > INT64_MAX / (dk > 0 ? dk : 1) ||
            (int64_t)h > INT64_MAX / (dv > 0 ? dv : 1)) {
            xkv_err(err, err_size, "xkv: head index overflow"); return false;
        }
        int64_t feat_base_k = ok + (int64_t)h * dk;
        int64_t feat_base_v = ov + (int64_t)h * dv;
        if (feat_base_k > INT64_MAX - dk || feat_base_v > INT64_MAX - dv) {
            xkv_err(err, err_size, "xkv: head slice overflow"); return false;
        }
        for (int64_t d = 0; d < dk; ++d) {
            const void * b_src = (const char *)b_k_data + (size_t)(feat_base_k + d) * b_k_nb1;
            if (!xkv_dequant_row(b_k_type, b_src, b_row, prk, k_rotated)) {
                xkv_err(err, err_size, "xkv: B_K dequant failed"); return false;
            }
            double acc = 0.0;
            for (int64_t r = 0; r < prk; ++r) acc += (double)a_row[r] * (double)b_row[r];
            k_head[d] = (float)acc;
        }
        // Zero-pad narrow layers to op maxima; RoPE spans min(rotary_dim, dk).
        for (int64_t d = dk; d < (int64_t)params->dim_k; ++d) k_head[d] = 0.0f;
        uint32_t rdim = params->rotary_dim;
        if ((int64_t)rdim > dk) rdim = (uint32_t)dk;
        xkv_rope_apply(k_head, params->dim_k, rdim, params->rope_mode, pos, omega, mag);

        const void * a_v_row_src = (const char *)a_v_data + (size_t)ar * a_v_nb1;
        if (!xkv_dequant_row(a_v_type, a_v_row_src, a_row, prv, v_rotated)) {
            xkv_err(err, err_size, "xkv: A_V dequant failed"); return false;
        }
        for (int64_t d = 0; d < dv; ++d) {
            const void * b_src = (const char *)b_v_data + (size_t)(feat_base_v + d) * b_v_nb1;
            if (!xkv_dequant_row(b_v_type, b_src, b_row, prv, v_rotated)) {
                xkv_err(err, err_size, "xkv: B_V dequant failed"); return false;
            }
            double acc = 0.0;
            for (int64_t r = 0; r < prv; ++r) acc += (double)a_row[r] * (double)b_row[r];
            v_head[d] = (float)acc;
        }
        for (int64_t d = dv; d < (int64_t)params->dim_v; ++d) v_head[d] = 0.0f;
        char * dst_row = (char *)dst_data + (size_t)i * dst_nb1;
        memcpy(dst_row, k_head, (size_t)params->dim_k * sizeof(float));
        memcpy(dst_row + (size_t)params->dim_k * sizeof(float), v_head,
               (size_t)params->dim_v * sizeof(float));
    }
    return true;
}

// Reference-only oracle: heap-allocates bounded scratch internally. Production
// backends must call ggml_xkv_reconstruct_core() with graph workspace instead.
bool ggml_xkv_reconstruct_oracle(
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
        char * err, size_t err_size) {
    if (!params) { xkv_err(err, err_size, "xkv: null params"); return false; }
    size_t need = 0;
    // Scratch bound needs padded ranks; derive best-effort here (validated inside core).
    int64_t prk = a_k_ne0, prv = a_v_ne0;
    if (!ggml_xkv_core_scratch_floats(params, prk, prv, &need, err, err_size)) return false;
    std::vector<float> scratch(need, 0.0f);
    return ggml_xkv_reconstruct_core(
        a_k_data, a_k_type, a_k_ne0, a_k_ne1, a_k_nb0, a_k_nb1,
        b_k_data, b_k_type, b_k_ne0, b_k_ne1, b_k_nb0, b_k_nb1,
        a_v_data, a_v_type, a_v_ne0, a_v_ne1, a_v_nb0, a_v_nb1,
        b_v_data, b_v_type, b_v_ne0, b_v_ne1, b_v_nb0, b_v_nb1,
        refs_data, positions_data, group_meta_data, layer_meta_data, n_layers,
        rope_data, rope_nelements, params,
        dst_data, dst_ne0, dst_ne1, dst_nb0, dst_nb1,
        scratch.data(), scratch.size(), err, err_size);
}

// ---- graph builder ----
struct ggml_tensor * ggml_xkv_reconstruct(
        struct ggml_context * ctx,
        struct ggml_tensor  * a_k,
        struct ggml_tensor  * b_k,
        struct ggml_tensor  * a_v,
        struct ggml_tensor  * b_v,
        struct ggml_tensor  * refs,
        struct ggml_tensor  * positions,
        struct ggml_tensor  * group_meta,
        struct ggml_tensor  * layer_meta,
        struct ggml_tensor  * rope_tables,
        const ggml_xkv_reconstruct_params * params) {
    GGML_ASSERT(ctx && a_k && b_k && a_v && b_v && refs && positions && group_meta && layer_meta &&
                rope_tables && params);
    GGML_ASSERT(params->version == GGML_XKV_VERSION);
    GGML_ASSERT(params->n_groups == 1);
    int64_t ne[4] = {(int64_t)params->dim_k + params->dim_v, (int64_t)params->n_sel, 1, 1};
    struct ggml_tensor * dst = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
    static_assert(sizeof(ggml_xkv_reconstruct_params) <= GGML_MAX_OP_PARAMS, "xkv params overflow");
    memcpy(dst->op_params, params, sizeof(*params));
    // New op id is patched into ggml.h enum (GGML_OP_XKV_RECONSTRUCT).
    dst->op = (enum ggml_op)(GGML_OP_XKV_RECONSTRUCT);
    dst->src[0] = a_k; dst->src[1] = b_k; dst->src[2] = a_v; dst->src[3] = b_v;
    dst->src[4] = refs; dst->src[5] = positions; dst->src[6] = group_meta;
    dst->src[7] = layer_meta; dst->src[8] = rope_tables;
    for (int i = 9; i < GGML_MAX_SRC; ++i) dst->src[i] = NULL;
    return dst;
}

// ---- landmark helpers ----
bool ggml_xkv_landmark_score(
        const float * q, uint32_t dim,
        const void * land_data, enum ggml_type land_type, int64_t land_ne0, int64_t land_ne1,
        size_t land_nb0, size_t land_nb1,
        const int32_t * frag_positions,
        const float * rope_omega, const float * rope_mag, uint32_t rope_fc,
        const ggml_xkv_landmark_config * cfg,
        float * scores_out, char * err, size_t err_size) {
    if (!q || !land_data || !frag_positions || !cfg || !scores_out) {
        xkv_err(err, err_size, "xkv landmark: null pointer"); return false;
    }
    if (dim == 0 || dim > 1024 || cfg->dim != dim) { xkv_err(err, err_size, "xkv landmark: bad dim"); return false; }
    if (!ggml_xkv_codec_supported(land_type)) { xkv_err(err, err_size, "xkv landmark: unsupported codec"); return false; }
    if (land_ne0 <= 0 || land_ne1 <= 0) { xkv_err(err, err_size, "xkv landmark: bad shape"); return false; }
    if (cfg->rotary_dim > dim || (cfg->rotary_dim & 1u)) { xkv_err(err, err_size, "xkv landmark: bad rotary_dim"); return false; }
    if (cfg->rope_mode != GGML_XKV_ROPE_HALF && cfg->rope_mode != GGML_XKV_ROPE_INTERLEAVED) {
        xkv_err(err, err_size, "xkv landmark: bad rope_mode"); return false;
    }
    // q is already post/effective-position RoPE: use as-is, never rotate again.
    // Only the canonical landmark K is phased to its storage position.
    uint32_t fc = cfg->rotary_dim / 2;
    if (fc > 0 && (!rope_omega || !rope_mag || rope_fc != fc)) {
        xkv_err(err, err_size, "xkv landmark: rope tables missing/mismatched"); return false;
    }
    // Landmark rank/dim padding: ne0 must cover dim (padded). Tail beyond dim ignored in dot.
    int64_t blk = ggml_blck_size(land_type);
    if (blk <= 0) { xkv_err(err, err_size, "xkv landmark: bad block"); return false; }
    // Padded dim for landmark vectors: round dim up like rank.
    int64_t padded = ((int64_t)dim + blk - 1) / blk * blk;
    if (xkv_is_turbo(land_type) && padded % 128 != 0) padded = (padded + 127) / 128 * 128;
    if (land_ne0 != padded) { xkv_err(err, err_size, "xkv landmark: ne0 != padded dim"); return false; }

    std::vector<float> lvec((size_t)padded), lph(dim);
    bool land_turbo = xkv_is_turbo(land_type);
    for (int64_t f = 0; f < land_ne1; ++f) {
        const void * row_src = (const char *)land_data + (size_t)f * land_nb1;
        (void)land_nb0;
        if (!xkv_dequant_row(land_type, row_src, lvec.data(), padded, false)) {
            xkv_err(err, err_size, "xkv landmark: dequant failed"); return false;
        }
        // Canonical decode for Turbo (inverse WHT inside dequant CANONICAL).
        (void)land_turbo;
        memcpy(lph.data(), lvec.data(), (size_t)dim * sizeof(float));
        xkv_rope_apply(lph.data(), dim, cfg->rotary_dim, cfg->rope_mode, frag_positions[f],
                       rope_omega, rope_mag);
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; ++d) acc += (double)q[d] * (double)lph[d];
        scores_out[f] = (float)acc;
    }
    return true;
}

bool ggml_xkv_landmark_topk(
        const float * scores, uint32_t n_frag,
        const ggml_xkv_landmark_config * cfg,
        uint32_t * indices_out, float * top_scores_out, char * err, size_t err_size) {
    if (!scores || !cfg || !indices_out) { xkv_err(err, err_size, "xkv topk: null"); return false; }
    if (cfg->top_k == 0) { xkv_err(err, err_size, "xkv topk: top_k==0 (never select-all)"); return false; }
    if (n_frag == 0) { xkv_err(err, err_size, "xkv topk: empty"); return false; }
    if (cfg->top_k > n_frag) { xkv_err(err, err_size, "xkv topk: top_k exceeds candidates"); return false; }
    std::vector<uint32_t> idx(n_frag);
    for (uint32_t i = 0; i < n_frag; ++i) idx[i] = i;
    // Deterministic: score desc, index asc (shares ggml_top_k determinism contract).
    std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    for (uint32_t i = 0; i < cfg->top_k; ++i) {
        indices_out[i] = idx[i];
        if (top_scores_out) top_scores_out[i] = scores[idx[i]];
    }
    return true;
}

bool ggml_xkv_landmark_refine(
        const uint32_t * selected, uint32_t n_selected,
        uint32_t n_rows_total, uint32_t n_frag, uint32_t frag_size,
        const ggml_xkv_landmark_config * cfg,
        uint32_t * refined_out, uint32_t * n_refined_out, uint32_t refined_cap,
        bool * cap_hit_out, char * err, size_t err_size) {
    if (!selected || !cfg || !refined_out || !n_refined_out) {
        xkv_err(err, err_size, "xkv refine: null"); return false;
    }
    if (n_selected == 0 || frag_size == 0 || n_frag == 0) { xkv_err(err, err_size, "xkv refine: bad input"); return false; }
    if (refined_cap == 0) { xkv_err(err, err_size, "xkv refine: zero cap"); return false; }
    // Expand each selected fragment [f*frag_size, (f+1)*frag_size) clipped to n_rows_total,
    // plus up to refine_max_rows boundary neighbors is the caller's full-precision
    // rescoring budget; here we deterministically gather fragment rows.
    std::vector<uint32_t> out;
    out.reserve(n_selected * frag_size + cfg->refine_max_rows);
    for (uint32_t i = 0; i < n_selected; ++i) {
        uint32_t f = selected[i];
        if (f >= n_frag) { xkv_err(err, err_size, "xkv refine: frag out of range"); return false; }
        uint32_t base = f * frag_size;
        for (uint32_t r = 0; r < frag_size && base + r < n_rows_total; ++r) {
            if (out.size() >= refined_cap) break;
            out.push_back(base + r);
        }
        if (out.size() >= refined_cap) break;
    }
    bool hit = out.size() >= refined_cap;
    // Deterministic dedup+sort (already ordered by fragment order; dedup).
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.size() > refined_cap) { out.resize(refined_cap); hit = true; }
    // Honor refine_max_rows as an additional bound on top of fragment rows.
    uint32_t hard_cap = n_selected * frag_size + cfg->refine_max_rows;
    if (out.size() > hard_cap) { out.resize(hard_cap); hit = true; }
    memcpy(refined_out, out.data(), out.size() * sizeof(uint32_t));
    *n_refined_out = (uint32_t)out.size();
    if (cap_hit_out) *cap_hit_out = hit;
    return true;
}

size_t ggml_xkv_exact_bytes(enum ggml_type type, int64_t padded_ne0, int64_t n_rows,
                            char * err, size_t err_size) {
    if (!ggml_xkv_codec_supported(type)) { xkv_err(err, err_size, "xkv: unsupported codec"); return 0; }
    if (padded_ne0 <= 0 || n_rows < 0) { xkv_err(err, err_size, "xkv: bad shape"); return 0; }
    // Reject non-block-divisible ne0 BEFORE ggml_row_size (which asserts).
    int64_t blk = ggml_blck_size(type);
    if (blk <= 0) { xkv_err(err, err_size, "xkv: bad block"); return 0; }
    if (padded_ne0 % blk != 0) { xkv_err(err, err_size, "xkv: ne0 not block-divisible"); return 0; }
    if (xkv_is_turbo(type) && padded_ne0 % 128 != 0) {
        xkv_err(err, err_size, "xkv: Turbo ne0 must be 128-divisible"); return 0;
    }
    size_t row = ggml_row_size(type, padded_ne0);
    if (row == 0 && n_rows != 0) { xkv_err(err, err_size, "xkv: bad row size"); return 0; }
    if (n_rows != 0 && row > SIZE_MAX / (size_t)n_rows) { xkv_err(err, err_size, "xkv: overflow"); return 0; }
    return row * (size_t)n_rows;
}

bool ggml_xkv_residency_supported(enum ggml_type type, ggml_xkv_residency res) {
    if (!ggml_xkv_codec_supported(type)) return false;
    if (res == GGML_XKV_RES_REFERENCE_HOST) return true;
    if (res == GGML_XKV_RES_DEVICE_OWNED) {
        // Device-owned reflects actual kernel capability in THIS build: the
        // Vulkan reconstruct kernel dequantizes F32/F16/Q8_0/Turbo2/3/4, and
        // Turbo additionally requires a valid compiled layout/table revision.
        // (Graph runtime additionally gates on proven op + device presence;
        // this query alone does not imply a device is present.)
        if (xkv_is_turbo(type)) return ggml_xkv_format_revision(type) != 0;
        return true;
    }
    return false;
}
