// ggml-cpu-xkv-landmark.cpp — CPU forward for GGML_OP_XKV_LANDMARK, _ROWS, and _MERGE.
//
// Single-threaded (n_tasks=1). Wraps reference oracles.
// Failures fail closed through status tensors. Never throws across C ABI.

#include "ggml-cpu-impl.h"
#include "ops.h"

#include "ggml-cpu-xkv-landmark.h"
#include "ggml-vulkan-landmark.h"

#include <cmath>
#include <cstring>

void ggml_compute_forward_xkv_landmark(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) return;

    const struct ggml_tensor * q         = dst->src[0];
    const struct ggml_tensor * landmarks = dst->src[1];
    const struct ggml_tensor * fpos     = dst->src[2];
    const struct ggml_tensor * fmeta    = dst->src[3];
    const struct ggml_tensor * qmeta    = dst->src[4];
    const struct ggml_tensor * rope     = dst->src[5];
    const struct ggml_tensor * scratch  = dst->src[6];
    struct ggml_tensor * ptrs   = dst->src[7];
    struct ggml_tensor * scores = dst->src[8];
    struct ggml_tensor * status = dst->src[9];

    ggml_xkv_landmark_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    auto fail = [&](int32_t code) {
        if (status && status->data) ((int32_t *)status->data)[0] = code;
    };

    if (!q || !landmarks || !fpos || !fmeta || !qmeta || !rope || !scratch ||
        !ptrs || !scores || !status) {
        fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        return;
    }
    char err[256] = {0};
    if (!ggml_xkv_landmark_supports(q, landmarks, fpos, fmeta, qmeta, rope,
            scratch, ptrs, scores, status, dst, &p, err, sizeof(err))) {
        fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        return;
    }

    const float * rope_data = ggml_nelements(rope) ? (const float *)rope->data : nullptr;

    bool ok = ggml_xkv_landmark_cpu_oracle(
        (const float *)q->data, landmarks->data, landmarks->type,
        (const int32_t *)fpos->data, (const int32_t *)fmeta->data,
        (const int32_t *)qmeta->data, rope_data, &p,
        (int32_t *)ptrs->data, (int32_t *)dst->data,
        (float *)scores->data, (int32_t *)status->data,
        err, sizeof(err));
    if (!ok) {
        if (status && status->data && ((const int32_t *)status->data)[0] == 0) {
            fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        }
    }
}

void ggml_compute_forward_xkv_landmark_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) return;

    const struct ggml_tensor * sel_idx     = dst->src[0];
    const struct ggml_tensor * frag_meta   = dst->src[1];
    const struct ggml_tensor * frag_row_off= dst->src[2];
    const struct ggml_tensor * frag_row_ids= dst->src[3];
    const struct ggml_tensor * frag_kv     = dst->src[4];
    const struct ggml_tensor * row_pos     = dst->src[5];
    struct ggml_tensor * row_ptrs          = dst->src[6];
    struct ggml_tensor * row_out_pos       = dst->src[7];
    struct ggml_tensor * row_entries       = dst->src[8];
    struct ggml_tensor * row_status        = dst->src[9];

    ggml_xkv_landmark_rows_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    auto fail = [&](int32_t code) {
        if (row_status && row_status->data) ((int32_t *)row_status->data)[0] = code;
    };

    if (!sel_idx || !frag_meta || !frag_kv || !row_pos || !row_ptrs || !row_out_pos || !row_entries || !row_status) {
        fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        return;
    }

    char err[256] = {0};
    if (!ggml_xkv_landmark_rows_supports(sel_idx, frag_meta, frag_row_off, frag_row_ids,
            frag_kv, row_pos, row_ptrs, dst, row_out_pos, row_entries, row_status, &p, err, sizeof(err))) {
        fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        return;
    }

    const int32_t * off_data = (frag_row_off && ggml_nelements(frag_row_off) > 1) ? (const int32_t *)frag_row_off->data : nullptr;
    const int32_t * ids_data = (frag_row_ids && ggml_nelements(frag_row_ids) > 1) ? (const int32_t *)frag_row_ids->data : nullptr;

    bool ok = ggml_xkv_landmark_rows_cpu_oracle(
        (const int32_t *)sel_idx->data,
        (const int32_t *)frag_meta->data,
        off_data, ids_data,
        (const int32_t *)frag_kv->data,
        (const int32_t *)row_pos->data,
        (const int32_t *)row_pos->data,
        &p,
        (int32_t *)row_ptrs->data,
        (int32_t *)dst->data,
        (int32_t *)row_out_pos->data,
        row_entries ? (int32_t *)row_entries->data : nullptr,
        (int32_t *)row_status->data,
        err, sizeof(err));
    if (!ok) {
        if (row_status && row_status->data && ((const int32_t *)row_status->data)[0] == 0) {
            fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        }
    }
}

void ggml_compute_forward_xkv_landmark_merge(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) return;

    const struct ggml_tensor * set_idx  = dst->src[0];
    const struct ggml_tensor * set_sc   = dst->src[1];
    const struct ggml_tensor * set_base = dst->src[2];
    struct ggml_tensor * out_sc         = dst->src[3];
    struct ggml_tensor * status         = dst->src[4];

    ggml_xkv_landmark_merge_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    auto fail = [&](int32_t code) {
        if (status && status->data) ((int32_t *)status->data)[0] = code;
    };

    if (!set_idx || !set_sc || !out_sc || !status) {
        fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        return;
    }

    char err[256] = {0};
    if (!ggml_xkv_landmark_merge_supports(set_idx, set_sc, set_base, out_sc, status, dst, &p, err, sizeof(err))) {
        fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        return;
    }

    const uint32_t * base_data = (p.has_set_base != 0 && set_base) ? (const uint32_t *)set_base->data : nullptr;

    bool ok = ggml_xkv_landmark_merge_with_base_cpu_oracle(
        (const int32_t *)set_idx->data,
        (const float *)set_sc->data,
        base_data,
        p.n_queries, p.n_sets, p.set_cap, p.top_k,
        (int32_t *)dst->data,
        (float *)out_sc->data,
        (int32_t *)status->data,
        err, sizeof(err));
    if (!ok) {
        if (status && status->data && ((const int32_t *)status->data)[0] == 0) {
            fail(GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
        }
    }
}
