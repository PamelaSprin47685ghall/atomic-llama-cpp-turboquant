// ggml-xkv-attention.cpp — native XKV dual-source indexed attention (Stage 2, v1).
//
// P0 ARCHITECTURE (XKV-SR §7.2): production chains TILED dispatches, each
// folding one workspace-bounded cold tile into a persistent (m, l, o) carry —
// one global online softmax, never a materialized dense union. Single-shot
// (carry NULL) exists only for the CPU oracle / small tests.
//
// Canonical domain first: hot Turbo K/V decode to canonical (inverse codec
// rotation via ggml_dequantize_turbo_row CANONICAL), cold K/V are canonical
// reconstruct outputs, Q is canonical post-RoPE, V accumulates canonical.

#include "ggml-xkv.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>

static void xattn_err(char * err, size_t n, const char * msg) {
    if (err && n) snprintf(err, n, "%s", msg);
}

static bool xattn_hot_type_ok(enum ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 ||
           t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

static bool xattn_cold_type_ok(enum ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16;
}

static bool xattn_is_turbo(enum ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

bool ggml_xkv_attn_tmp_floats(const ggml_xkv_attention_params * params,
        size_t * out_floats, char * err, size_t err_size) {
    if (!params || !out_floats) { xattn_err(err, err_size, "xkv attn: null pointer"); return false; }
    if (params->version != GGML_XKV_ATTN_VERSION) { xattn_err(err, err_size, "xkv attn: bad version"); return false; }
    if (params->dim_k == 0 || params->dim_v == 0) { xattn_err(err, err_size, "xkv attn: zero dims"); return false; }
    if (params->dim_k > 1024 || params->dim_v > 1024) { xattn_err(err, err_size, "xkv attn: dims exceed bound"); return false; }
    uint64_t need = (uint64_t)params->dim_k + (uint64_t)params->dim_v;
    if (need > (size_t)-1 / sizeof(float)) { xattn_err(err, err_size, "xkv attn: tmp overflow"); return false; }
    *out_floats = (size_t)need;
    return true;
}

bool ggml_xkv_attention_supports(
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
        char * err, size_t err_size) {
    if (!q || !k_hot || !v_hot || !k_cold || !v_cold || !entries || !offsets || !status || !dst || !params) {
        xattn_err(err, err_size, "xkv attn: null tensor/params"); return false;
    }
    if (params->version != GGML_XKV_ATTN_VERSION) { xattn_err(err, err_size, "xkv attn: bad version"); return false; }
    if (params->reserved0 != 0 || params->reserved1 != 0) { xattn_err(err, err_size, "xkv attn: reserved nonzero"); return false; }
    if (params->n_queries == 0 || params->n_groups == 0 || params->gqa_ratio == 0) {
        xattn_err(err, err_size, "xkv attn: zero queries/groups/gqa"); return false;
    }
    if (params->dim_k == 0 || params->dim_k > 1024 || params->dim_v == 0 || params->dim_v > 1024) {
        xattn_err(err, err_size, "xkv attn: bad dims"); return false;
    }
    constexpr uint32_t known_flags = GGML_XKV_ATTN_FLAG_REJECT_DUPES |
                                     GGML_XKV_ATTN_FLAG_FIRST_TILE |
                                     GGML_XKV_ATTN_FLAG_FINAL_TILE;
    if ((params->flags & ~known_flags) != 0) {
        xattn_err(err, err_size, "xkv attn: unknown flags"); return false;
    }
    const bool first_tile = (params->flags & GGML_XKV_ATTN_FLAG_FIRST_TILE) != 0;
    if (first_tile == (carry != nullptr)) {
        xattn_err(err, err_size, "xkv attn: FIRST_TILE requires null carry; later tiles require carry"); return false;
    }
    if (q->type != GGML_TYPE_F32) { xattn_err(err, err_size, "xkv attn: q must be F32"); return false; }
    if (!xattn_hot_type_ok(k_hot->type)) { xattn_err(err, err_size, "xkv attn: unsupported hot K type"); return false; }
    if (!xattn_hot_type_ok(v_hot->type)) { xattn_err(err, err_size, "xkv attn: unsupported hot V type"); return false; }
    if (!xattn_cold_type_ok(k_cold->type)) { xattn_err(err, err_size, "xkv attn: cold K must be F32/F16"); return false; }
    if (!xattn_cold_type_ok(v_cold->type)) { xattn_err(err, err_size, "xkv attn: cold V must be F32/F16"); return false; }
    // Quantized hot rows come in whole blocks (Turbo pads to 128 graph-side).
    if (k_hot->type == GGML_TYPE_Q8_0 && params->dim_k % 32u != 0) {
        xattn_err(err, err_size, "xkv attn: Q8 hot K dim not multiple of 32"); return false;
    }
    if (xattn_is_turbo(k_hot->type) && params->dim_k % 128u != 0) {
        xattn_err(err, err_size, "xkv attn: Turbo hot K dim not multiple of 128"); return false;
    }
    if (v_hot->type == GGML_TYPE_Q8_0 && params->dim_v % 32u != 0) {
        xattn_err(err, err_size, "xkv attn: Q8 hot V dim not multiple of 32"); return false;
    }
    if (xattn_is_turbo(v_hot->type) && params->dim_v % 128u != 0) {
        xattn_err(err, err_size, "xkv attn: Turbo hot V dim not multiple of 128"); return false;
    }
    if (q->ne[0] != (int64_t)params->dim_k || q->ne[1] != (int64_t)params->gqa_ratio ||
        q->ne[2] != (int64_t)params->n_groups || q->ne[3] != 1) {
        xattn_err(err, err_size, "xkv attn: q must be [Dk, GQA, n_groups]"); return false;
    }
    // Both CPU and Vulkan kernels index Dk directly and encode higher Q and
    // storage strides in 32-bit byte push constants. Reject non-representable
    // or non-element-aligned views instead of truncating `nb` in dispatch.
    if (q->nb[0] != sizeof(float) || q->nb[1] % sizeof(float) != 0 ||
        q->nb[2] % sizeof(float) != 0 || q->nb[1] > UINT32_MAX || q->nb[2] > UINT32_MAX) {
        xattn_err(err, err_size, "xkv attn: unsupported Q stride"); return false;
    }
    if (k_hot->ne[0] != (int64_t)params->dim_k || k_hot->ne[2] != (int64_t)params->hot_rows) {
        xattn_err(err, err_size, "xkv attn: k_hot layout mismatch"); return false;
    }
    if (v_hot->ne[0] != (int64_t)params->dim_v || v_hot->ne[2] != (int64_t)params->hot_rows) {
        xattn_err(err, err_size, "xkv attn: v_hot layout mismatch"); return false;
    }
    if (params->kv_head >= (uint64_t)k_hot->ne[1] || params->kv_head >= (uint64_t)v_hot->ne[1]) {
        xattn_err(err, err_size, "xkv attn: kv_head out of range"); return false;
    }
    if (params->stream >= (uint64_t)k_hot->ne[3] || params->stream >= (uint64_t)v_hot->ne[3]) {
        xattn_err(err, err_size, "xkv attn: stream out of range"); return false;
    }
    if (k_cold->ne[0] != (int64_t)params->dim_k || k_cold->ne[1] != (int64_t)params->n_cold) {
        xattn_err(err, err_size, "xkv attn: k_cold must be [Dk, n_cold]"); return false;
    }
    if (v_cold->ne[0] != (int64_t)params->dim_v || v_cold->ne[1] != (int64_t)params->n_cold) {
        xattn_err(err, err_size, "xkv attn: v_cold must be [Dv, n_cold]"); return false;
    }
    if (k_hot->nb[1] > UINT32_MAX || k_hot->nb[2] > UINT32_MAX || k_hot->nb[3] > UINT32_MAX ||
        v_hot->nb[1] > UINT32_MAX || v_hot->nb[2] > UINT32_MAX || v_hot->nb[3] > UINT32_MAX ||
        k_cold->nb[1] > UINT32_MAX || v_cold->nb[1] > UINT32_MAX) {
        xattn_err(err, err_size, "xkv attn: storage stride exceeds native bound"); return false;
    }
    if (entries->type != GGML_TYPE_I32 || (entries->ne[0] != 2 && entries->ne[0] != GGML_XKV_ATTN_ENTRY_STRIDE)) {
        xattn_err(err, err_size, "xkv attn: entries must be I32[S,E], S==2 or 4"); return false;
    }
    if (entries->nb[0] != sizeof(int32_t) || entries->nb[1] != (size_t)entries->ne[0] * sizeof(int32_t)) {
        xattn_err(err, err_size, "xkv attn: entries must be contiguous"); return false;
    }
    if (entries->ne[1] != (int64_t)params->n_entries) {
        xattn_err(err, err_size, "xkv attn: entries capacity mismatch"); return false;
    }
    if (offsets->type != GGML_TYPE_I32 || !ggml_is_vector(offsets) ||
        offsets->ne[0] != (int64_t)params->n_queries + 1 || offsets->nb[0] != sizeof(int32_t)) {
        xattn_err(err, err_size, "xkv attn: offsets must be I32[n_queries+1]"); return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] != (int64_t)params->gqa_ratio ||
                  sinks->nb[0] != sizeof(float))) {
        xattn_err(err, err_size, "xkv attn: sinks must be F32[GQA]"); return false;
    }
    if (status->type != GGML_TYPE_I32 || status->ne[0] < 1 || status->nb[0] != sizeof(int32_t)) {
        xattn_err(err, err_size, "xkv attn: status must be I32[>=1]"); return false;
    }
    if (carry) {
        if (carry->type != GGML_TYPE_F32 || carry->ne[0] != (int64_t)params->dim_v + 2 ||
            carry->ne[1] != (int64_t)params->gqa_ratio || carry->ne[2] != (int64_t)params->n_queries ||
            !ggml_is_contiguous(carry)) {
            xattn_err(err, err_size, "xkv attn: carry must be contiguous F32[Dv+2, GQA, NQ]"); return false;
        }
    }
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != (int64_t)params->dim_v + 2 ||
        dst->ne[1] != (int64_t)params->gqa_ratio || dst->ne[2] != (int64_t)params->n_queries ||
        !ggml_is_contiguous(dst)) {
        xattn_err(err, err_size, "xkv attn: dst must be F32[Dv+2, GQA, n_queries]"); return false;
    }
    return true;
}

struct ggml_tensor * ggml_xkv_attention(
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
        struct ggml_tensor  * carry,
        const ggml_xkv_attention_params * params) {
    GGML_ASSERT(ctx && q && k_hot && v_hot && k_cold && v_cold && entries && offsets && status && params);
    GGML_ASSERT(params->version == GGML_XKV_ATTN_VERSION);
    GGML_ASSERT(params->reserved0 == 0 && params->reserved1 == 0);
    GGML_ASSERT(((params->flags & GGML_XKV_ATTN_FLAG_FIRST_TILE) != 0) == (carry == nullptr));
    int64_t ne[4] = {(int64_t)params->dim_v + 2, (int64_t)params->gqa_ratio, (int64_t)params->n_queries, 1};
    struct ggml_tensor * dst = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    static_assert(sizeof(ggml_xkv_attention_params) <= GGML_MAX_OP_PARAMS, "xkv attn params overflow");
    memcpy(dst->op_params, params, sizeof(*params));
    dst->op = (enum ggml_op)(GGML_OP_XKV_ATTENTION);
    dst->src[0] = q; dst->src[1] = k_hot; dst->src[2] = v_hot; dst->src[3] = k_cold;
    dst->src[4] = v_cold; dst->src[5] = entries; dst->src[6] = offsets; dst->src[7] = sinks;
    dst->src[8] = status; dst->src[9] = carry;
    return dst;
}

// ---- oracle ----
static void xattn_poison(float * dst, size_t n) {
    uint32_t nan = 0x7fc00000u;
    float f;
    memcpy(&f, &nan, sizeof(f));
    for (size_t i = 0; i < n; ++i) dst[i] = f;
}

// Canonical decode: F32 memcpy, F16/Q8 via traits, Turbo via inverse rotation
// (GGML_TURBO_DECODE_CANONICAL). No WHT-shortcut: production dots canonical.
static bool xattn_dequant_row(const void * src, enum ggml_type type, float * dst, int64_t n) {
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t)n * sizeof(float));
        return true;
    }
    if (xattn_is_turbo(type)) {
        return ggml_dequantize_turbo_row(type, src, dst, n, 128, GGML_TURBO_DECODE_CANONICAL);
    }
    const auto * tr = ggml_get_type_traits(type);
    if (!tr || !tr->to_float) return false;
    tr->to_float(src, dst, n);
    return true;
}

bool ggml_xkv_attention_oracle(
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
        const float * carry_data,
        const ggml_xkv_attention_params * params,
        float * dst_data, int64_t dst_ne0, int64_t dst_ne1, int64_t dst_ne2,
        size_t dst_nb0, size_t dst_nb1, size_t dst_nb2,
        float * tmp, size_t tmp_floats,
        char * err, size_t err_size) {
    if (!params || !q_data || !offsets_data || !dst_data) {
        xattn_err(err, err_size, "xkv attn oracle: null pointer"); return false;
    }
    if ((params->hot_rows > 0 && (!k_hot_data || !v_hot_data)) ||
        (params->n_cold > 0 && (!k_cold_data || !v_cold_data)) ||
        (params->n_entries > 0 && !entries_data)) {
        xattn_err(err, err_size, "xkv attn oracle: null nonempty source"); return false;
    }
    const uint32_t DK = params->dim_k, DV = params->dim_v;
    const uint32_t GQA = params->gqa_ratio, NQ = params->n_queries, NG = params->n_groups;
    const bool first_tile = (params->flags & GGML_XKV_ATTN_FLAG_FIRST_TILE) != 0;
    const bool final_tile = (params->flags & GGML_XKV_ATTN_FLAG_FINAL_TILE) != 0;
    const bool use_carry = !first_tile;
    const bool apply_sink = sinks_data && first_tile;
    if (first_tile == (carry_data != nullptr)) {
        xattn_err(err, err_size, "xkv attn oracle: invalid carry chain"); return false;
    }
    bool shape_ok =
        params->version == GGML_XKV_ATTN_VERSION && params->reserved0 == 0 && params->reserved1 == 0 &&
        q_ne0 == (int64_t)DK && q_ne1 == (int64_t)GQA && q_ne2 == (int64_t)NG &&
        kh_ne0 == (int64_t)DK && kh_ne2 == (int64_t)params->hot_rows &&
        params->kv_head < (uint64_t)kh_ne1 && params->stream < (uint64_t)kh_ne3 &&
        vh_ne0 == (int64_t)DV && vh_ne2 == (int64_t)params->hot_rows &&
        params->kv_head < (uint64_t)vh_ne1 && params->stream < (uint64_t)vh_ne3 &&
        kc_ne0 == (int64_t)DK && kc_ne1 == (int64_t)params->n_cold &&
        vc_ne0 == (int64_t)DV && vc_ne1 == (int64_t)params->n_cold &&
        (entries_ne0 == 2 || entries_ne0 == GGML_XKV_ATTN_ENTRY_STRIDE) &&
        entries_ne1 == (int64_t)params->n_entries &&
        offsets_ne0 == (int64_t)NQ + 1 &&
        dst_ne0 == (int64_t)DV + 2 && dst_ne1 == (int64_t)GQA && dst_ne2 == (int64_t)NQ &&
        xattn_hot_type_ok(k_hot_type) && xattn_hot_type_ok(v_hot_type) &&
        xattn_cold_type_ok(k_cold_type) && xattn_cold_type_ok(v_cold_type);
    size_t need = 0;
    if (!ggml_xkv_attn_tmp_floats(params, &need, err, err_size)) return false;
    if (!shape_ok || !tmp || tmp_floats < need) {
        if (status_data) status_data[0] = 6;
        xattn_err(err, err_size, "xkv attn oracle: descriptor/tmp mismatch");
        return false;
    }
    float * k_tmp = tmp;
    float * v_tmp = tmp + DK;

    const float eff_scale = params->logit_softcap > 0.0f ? params->scale / params->logit_softcap : params->scale;
    const bool reject_dupes = (params->flags & GGML_XKV_ATTN_FLAG_REJECT_DUPES) != 0;
    const int64_t S = entries_ne0;
    const size_t carry_stride = (size_t)(DV + 2);

    if (status_data) {
        if (first_tile) {
            status_data[0] = 0;
        } else if (status_data[0] != 0) {
            xattn_err(err, err_size, "xkv attn oracle: prior tile failed");
            return false;
        }
    }
    for (uint32_t qq = 0; qq < NQ; ++qq) {
        const int32_t begin = offsets_data[qq];
        const int32_t end = offsets_data[qq + 1];
        if (begin < 0 || end < begin || end > (int64_t)params->n_entries) {
            if (status_data && status_data[0] == 0) status_data[0] = 1;
            for (uint32_t g = 0; g < GQA; ++g) {
                float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(dst_data) +
                        (size_t)qq * dst_nb2 + (size_t)g * dst_nb1);
                xattn_poison(out, DV);
            }
            xattn_err(err, err_size, "xkv attn oracle: malformed offsets");
            return false;
        }
        for (uint32_t g = 0; g < GQA; ++g) {
            float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(dst_data) +
                    (size_t)qq * dst_nb2 + (size_t)g * dst_nb1);
            const float * cy = use_carry ? carry_data + ((size_t)qq * GQA + g) * carry_stride : nullptr;
            float max_score, sum_exp;
            if (first_tile) {
                max_score = -INFINITY;
                sum_exp = 0.0f;
                for (uint32_t d = 0; d < DV; ++d) v_tmp[d] = 0.0f;
            } else {
                for (uint32_t d = 0; d < DV; ++d) v_tmp[d] = cy[d];
                max_score = cy[DV];
                sum_exp = cy[DV + 1];
            }

            for (int32_t e = begin; e < end; ++e) {
                int32_t source, row, group;
                if (S == GGML_XKV_ATTN_ENTRY_STRIDE) {
                    source = entries_data[(size_t)e * 4 + 0];
                    row    = entries_data[(size_t)e * 4 + 1];
                    group  = entries_data[(size_t)e * 4 + 2];
                    if (entries_data[(size_t)e * 4 + 3] == 0) continue; // masked
                } else {
                    const int32_t key = entries_data[(size_t)e * 2 + 0];
                    group = entries_data[(size_t)e * 2 + 1];
                    if (key < 0) {
                        if (status_data && status_data[0] == 0) status_data[0] = 3;
                        xattn_poison(out, DV);
                        xattn_err(err, err_size, "xkv attn oracle: malformed index");
                        return false;
                    }
                    if ((uint64_t)key < params->hot_rows) { source = GGML_XKV_ATTN_SOURCE_HOT; row = key; }
                    else { source = GGML_XKV_ATTN_SOURCE_COLD; row = key - (int32_t)params->hot_rows; }
                }
                if (source != GGML_XKV_ATTN_SOURCE_HOT && source != GGML_XKV_ATTN_SOURCE_COLD) {
                    if (status_data && status_data[0] == 0) status_data[0] = 2;
                    xattn_poison(out, DV);
                    xattn_err(err, err_size, "xkv attn oracle: malformed source");
                    return false;
                }
                if (group < 0 || (uint64_t)group >= NG) {
                    if (status_data && status_data[0] == 0) status_data[0] = 4;
                    xattn_poison(out, DV);
                    xattn_err(err, err_size, "xkv attn oracle: malformed group");
                    return false;
                }
                const bool is_hot = (source == GGML_XKV_ATTN_SOURCE_HOT);
                if (row < 0 || (is_hot ? (uint64_t)row >= params->hot_rows
                                       : (uint64_t)row >= params->n_cold)) {
                    if (status_data && status_data[0] == 0) status_data[0] = 3;
                    xattn_poison(out, DV);
                    xattn_err(err, err_size, "xkv attn oracle: malformed index");
                    return false;
                }
                if (reject_dupes) {
                    for (int32_t d2 = begin; d2 < e; ++d2) {
                        int32_t s2, r2;
                        if (S == GGML_XKV_ATTN_ENTRY_STRIDE) {
                            if (entries_data[(size_t)d2 * 4 + 3] == 0) continue;
                            s2 = entries_data[(size_t)d2 * 4 + 0];
                            r2 = entries_data[(size_t)d2 * 4 + 1];
                        } else {
                            const int32_t k2 = entries_data[(size_t)d2 * 2 + 0];
                            if (k2 < 0) continue;
                            if ((uint64_t)k2 < params->hot_rows) { s2 = GGML_XKV_ATTN_SOURCE_HOT; r2 = k2; }
                            else { s2 = GGML_XKV_ATTN_SOURCE_COLD; r2 = k2 - (int32_t)params->hot_rows; }
                        }
                        if (s2 == source && r2 == row) {
                            if (status_data && status_data[0] == 0) status_data[0] = 5;
                            xattn_poison(out, DV);
                            xattn_err(err, err_size, "xkv attn oracle: duplicate entry");
                            return false;
                        }
                    }
                }
                const float * qv = (const float *)((const char *)q_data + (size_t)group * q_nb2 + (size_t)g * q_nb1);
                if (is_hot) {
                    const void * kv = (const char *)k_hot_data + (size_t)params->kv_head * kh_nb1 +
                                      (size_t)row * kh_nb2 + (size_t)params->stream * kh_nb3;
                    if (!xattn_dequant_row(kv, k_hot_type, k_tmp, DK)) {
                        if (status_data && status_data[0] == 0) status_data[0] = 6;
                        xattn_poison(out, DV);
                        xattn_err(err, err_size, "xkv attn oracle: hot K decode failed");
                        return false;
                    }
                } else {
                    const void * kc = (const char *)k_cold_data + (size_t)row * kc_nb1;
                    if (!xattn_dequant_row(kc, k_cold_type, k_tmp, DK)) {
                        if (status_data && status_data[0] == 0) status_data[0] = 6;
                        xattn_poison(out, DV);
                        xattn_err(err, err_size, "xkv attn oracle: cold K decode failed");
                        return false;
                    }
                }
                double dot = 0.0;
                for (uint32_t d = 0; d < DK; ++d) dot += (double)qv[d] * (double)k_tmp[d];
                float score = (float)(dot * (double)eff_scale);
                if (params->logit_softcap > 0.0f) score = params->logit_softcap * tanhf(score);
                if (!std::isfinite(score)) continue; // non-finite treated as masked
                const float old_max = max_score;
                if (!(max_score > -INFINITY)) {
                    max_score = score;
                    sum_exp = 1.0f;
                    const void * vd = is_hot
                        ? (const void *)((const char *)v_hot_data + (size_t)params->kv_head * vh_nb1 +
                                         (size_t)row * vh_nb2 + (size_t)params->stream * vh_nb3)
                        : (const void *)((const char *)v_cold_data + (size_t)row * vc_nb1);
                    if (!xattn_dequant_row(vd, is_hot ? v_hot_type : v_cold_type, v_tmp, DV)) {
                        if (status_data && status_data[0] == 0) status_data[0] = 6;
                        xattn_poison(out, DV);
                        xattn_err(err, err_size, "xkv attn oracle: V decode failed");
                        return false;
                    }
                } else {
                    float old_scale = 1.0f, value_scale = 1.0f;
                    if (score > max_score) {
                        old_scale = expf(old_max - score);
                        max_score = score;
                    } else {
                        value_scale = expf(score - max_score);
                    }
                    const void * vd = is_hot
                        ? (const void *)((const char *)v_hot_data + (size_t)params->kv_head * vh_nb1 +
                                         (size_t)row * vh_nb2 + (size_t)params->stream * vh_nb3)
                        : (const void *)((const char *)v_cold_data + (size_t)row * vc_nb1);
                    // `k_tmp` is only DK floats and cannot hold V when DV > DK.
                    // Decode the incoming V into this tile's distinct dst row;
                    // it is scratch until final/intermediate output is committed.
                    if (!xattn_dequant_row(vd, is_hot ? v_hot_type : v_cold_type, out, DV)) {
                        if (status_data && status_data[0] == 0) status_data[0] = 6;
                        xattn_poison(out, DV);
                        xattn_err(err, err_size, "xkv attn oracle: V decode failed");
                        return false;
                    }
                    const float * vrow = out;
                    for (uint32_t d = 0; d < DV; ++d) v_tmp[d] = v_tmp[d] * old_scale + value_scale * vrow[d];
                    sum_exp = sum_exp * old_scale + value_scale;
                }
            }
            if (apply_sink) {
                const float sscore = sinks_data[g];
                if (std::isfinite(sscore) && sscore > -1e30f) {
                    if (!(max_score > -INFINITY)) {
                        max_score = sscore;
                        sum_exp = 1.0f;
                    } else if (sscore > max_score) {
                        const float f = expf(max_score - sscore);
                        max_score = sscore;
                        for (uint32_t d = 0; d < DV; ++d) v_tmp[d] *= f;
                        sum_exp = sum_exp * f + 1.0f;
                    } else {
                        sum_exp += expf(sscore - max_score);
                    }
                }
            }
            if (final_tile && sum_exp > 0.0f && std::isfinite(sum_exp)) {
                const float inv = 1.0f / sum_exp;
                for (uint32_t d = 0; d < DV; ++d) out[d] = v_tmp[d] * inv;
            } else if (final_tile) {
                for (uint32_t d = 0; d < DV; ++d) out[d] = 0.0f;
            } else {
                for (uint32_t d = 0; d < DV; ++d) out[d] = v_tmp[d];
            }
            out[DV] = max_score;
            out[DV + 1] = sum_exp;
        }
    }
    return true;
}
