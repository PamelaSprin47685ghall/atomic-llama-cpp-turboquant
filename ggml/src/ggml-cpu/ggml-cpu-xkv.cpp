// ggml-cpu-xkv.cpp — CPU forward for GGML_OP_XKV_RECONSTRUCT (v1).
//
// Single-threaded (n_tasks=1). Production compute uses bounded graph workspace
// (params->wdata) via ggml_xkv_reconstruct_core(): no per-op heap. Fail-closed
// by NaN-poisoning dst on invalid input so graph compute never throws across
// the C ABI. Fault tests exercise validate/core directly for precise errors.

#include "ggml-cpu-impl.h"
#include "ops.h"

#include "ggml-xkv.h"

#include <cmath>
#include <cstring>

void ggml_compute_forward_xkv_reconstruct(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }
    const struct ggml_tensor * a_k = dst->src[0];
    const struct ggml_tensor * b_k = dst->src[1];
    const struct ggml_tensor * a_v = dst->src[2];
    const struct ggml_tensor * b_v = dst->src[3];
    const struct ggml_tensor * refs = dst->src[4];
    const struct ggml_tensor * positions = dst->src[5];
    const struct ggml_tensor * group_meta = dst->src[6];
    const struct ggml_tensor * layer_meta = dst->src[7];
    const struct ggml_tensor * rope_tables = dst->src[8];

    ggml_xkv_reconstruct_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    auto poison = [&]() {
        float * out = (float *)dst->data;
        int64_t n = ggml_nelements(dst);
        for (int64_t i = 0; i < n; ++i) out[i] = NAN;
    };

    if (!a_k || !b_k || !a_v || !b_v || !refs || !positions || !group_meta || !layer_meta ||
        !rope_tables) {
        poison();
        return;
    }

    char err[256] = {0};
    if (!ggml_xkv_reconstruct_supports(a_k, b_k, a_v, b_v, refs, positions, group_meta,
                                       layer_meta, rope_tables, dst, &p, err, sizeof(err))) {
        poison();
        return;
    }

    // Bounded workspace (no heap in production compute).
    size_t need = 0;
    if (!ggml_xkv_core_scratch_floats(&p, a_k->ne[0], a_v->ne[0], &need, err, sizeof(err))) {
        poison();
        return;
    }
    if (!params->wdata || params->wsize < need * sizeof(float)) {
        poison();
        return;
    }

    const float * rope_data = (const float *)rope_tables->data;
    int64_t rope_n = ggml_nelements(rope_tables);

    bool ok = ggml_xkv_reconstruct_core(
        a_k->data, a_k->type, a_k->ne[0], a_k->ne[1], a_k->nb[0], a_k->nb[1],
        b_k->data, b_k->type, b_k->ne[0], b_k->ne[1], b_k->nb[0], b_k->nb[1],
        a_v->data, a_v->type, a_v->ne[0], a_v->ne[1], a_v->nb[0], a_v->nb[1],
        b_v->data, b_v->type, b_v->ne[0], b_v->ne[1], b_v->nb[0], b_v->nb[1],
        (const int32_t *)refs->data, (const int32_t *)positions->data,
        (const int32_t *)group_meta->data, (const int32_t *)layer_meta->data, layer_meta->ne[1],
        rope_data, rope_n, &p,
        (float *)dst->data, dst->ne[0], dst->ne[1], dst->nb[0], dst->nb[1],
        (float *)params->wdata, need, err, sizeof(err));
    if (!ok) {
        poison();
    }
}
