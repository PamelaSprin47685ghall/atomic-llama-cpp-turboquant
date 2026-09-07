// ggml-cpu-xkv-attention.cpp — CPU forward for GGML_OP_XKV_ATTENTION (v1).
//
// Single-threaded (n_tasks=1). Canonical domain throughout (hot Turbo rows
// inverse-rotated on decode inside the oracle). Production compute uses bounded
// graph workspace (wdata) as the oracle tmp: no per-op heap. Tiled carry
// chains pass persistent (m, l, o) through immutable prior-dst carry values.
// Fail-closed by NaN-poisoning dst on invalid input so graph compute never
// throws across the C ABI.

#include "ggml-cpu-impl.h"
#include "ops.h"

#include "ggml-xkv.h"

#include <cmath>
#include <cstring>

void ggml_compute_forward_xkv_attention(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }
    const struct ggml_tensor * q        = dst->src[0];
    const struct ggml_tensor * k_hot    = dst->src[1];
    const struct ggml_tensor * v_hot    = dst->src[2];
    const struct ggml_tensor * k_cold   = dst->src[3];
    const struct ggml_tensor * v_cold   = dst->src[4];
    const struct ggml_tensor * entries  = dst->src[5];
    const struct ggml_tensor * offsets  = dst->src[6];
    const struct ggml_tensor * sinks    = dst->src[7];
    const struct ggml_tensor * status   = dst->src[8];
    const struct ggml_tensor * carry    = dst->src[9];

    ggml_xkv_attention_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    uint32_t nan = 0x7fc00000u;
    float fnan;
    memcpy(&fnan, &nan, sizeof(fnan));
    auto poison = [&]() {
        const size_t n = (size_t)ggml_nelements(dst);
        float * d = (float *)dst->data;
        for (size_t i = 0; i < n; ++i) d[i] = fnan;
        if (status && status->data) ((int32_t *)status->data)[0] = 6;
    };

    char err[256] = {0};
    if (!ggml_xkv_attention_supports(q, k_hot, v_hot, k_cold, v_cold, entries,
                                     offsets, sinks, status, carry, dst, &p, err, sizeof(err))) {
        poison();
        return;
    }
    size_t need = 0;
    if (!ggml_xkv_attn_tmp_floats(&p, &need, err, sizeof(err))) {
        poison();
        return;
    }
    float * tmp = need ? (float *)params->wdata : nullptr;

    int32_t * status_data = (status && status->data) ? (int32_t *)status->data : nullptr;
    const float * carry_data = (carry && carry->data) ? (const float *)carry->data : nullptr;
    bool ok = ggml_xkv_attention_oracle(
        (const float *)q->data, q->ne[0], q->ne[1], q->ne[2], q->nb[0], q->nb[1], q->nb[2],
        k_hot->data, k_hot->type, k_hot->ne[0], k_hot->ne[1], k_hot->ne[2], k_hot->ne[3],
        k_hot->nb[0], k_hot->nb[1], k_hot->nb[2], k_hot->nb[3],
        v_hot->data, v_hot->type, v_hot->ne[0], v_hot->ne[1], v_hot->ne[2], v_hot->ne[3],
        v_hot->nb[0], v_hot->nb[1], v_hot->nb[2], v_hot->nb[3],
        k_cold->data, k_cold->type, k_cold->ne[0], k_cold->ne[1], k_cold->nb[0], k_cold->nb[1],
        v_cold->data, v_cold->type, v_cold->ne[0], v_cold->ne[1], v_cold->nb[0], v_cold->nb[1],
        (const int32_t *)entries->data, entries->ne[0], entries->ne[1],
        (const int32_t *)offsets->data, offsets->ne[0],
        sinks ? (const float *)sinks->data : nullptr,
        status_data, carry_data, &p,
        (float *)dst->data, dst->ne[0], dst->ne[1], dst->ne[2],
        dst->nb[0], dst->nb[1], dst->nb[2],
        tmp, need, err, sizeof(err));
    if (!ok) {
        // Fail-closed across the C ABI: whole dst NaN (later queries uncomputed).
        const size_t n = (size_t)ggml_nelements(dst);
        float * d = (float *)dst->data;
        for (size_t i = 0; i < n; ++i) d[i] = fnan;
        if (status_data && status_data[0] == 0) status_data[0] = 6;
    }
}
