// ggml-cpu-xkv-landmark-build.cpp — CPU forward for
// GGML_OP_XKV_LANDMARK_BUILD (v1).
//
// Single-threaded (n_tasks=1). Wraps ggml_xkv_landmark_build_cpu_oracle;
// validation failures fail closed through the I32[4] status word without
// touching dst/eb/srcfp ("unpublished"). Never throws across the C ABI.

#include "ggml-cpu-impl.h"
#include "ops.h"

#include "ggml-cpu-xkv-landmark-build.h"
#include "ggml-xkv-landmark-build.h"

#include <cstring>

void ggml_compute_forward_xkv_landmark_build(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }
    const struct ggml_tensor * a_k   = dst->src[0];
    const struct ggml_tensor * b_k   = dst->src[1];
    const struct ggml_tensor * rows  = dst->src[2];
    const struct ggml_tensor * pos   = dst->src[3];
    const struct ggml_tensor * lmeta = dst->src[4];
    const struct ggml_tensor * rope  = dst->src[5];
    const struct ggml_tensor * scr   = dst->src[6];
    struct ggml_tensor * eb     = dst->src[7];
    struct ggml_tensor * srcfp  = dst->src[8];
    struct ggml_tensor * status = dst->src[9];

    ggml_xkv_landmark_build_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    auto fail = [&](int32_t code) {
        if (status && status->data) {
            ((int32_t *)status->data)[0] = code;
        }
    };

    if (!a_k || !b_k || !rows || !pos || !lmeta || !rope || !scr || !eb || !srcfp || !status) {
        fail(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT);
        return;
    }
    char err[256] = {0};
    if (!ggml_xkv_landmark_build_supports(a_k, b_k, rows, pos, lmeta, rope,
            scr, eb, srcfp, status, dst, &p, err, sizeof(err))) {
        fprintf(stderr, "CPU FORWARD SUPPORTS FAILED: %s\n", err);
        // Distinguish workspace shorts from bad inputs for the status word.
        fail(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT);
        // supports() already validated scratch size; re-probe exact bytes to
        // report WORKSPACE precisely when that is the only failure.
        size_t need = 0;
        // best-effort: if params+meta parse, check scratch shortfall
        if (ggml_nbytes(scr) == 0) fail(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_WORKSPACE);
        (void)need;
        return;
    }

    const float * rope_data = ggml_nelements(rope) ? (const float *)rope->data : nullptr;
    int pos_is_64 = (pos->type == GGML_TYPE_I64) ? 1 : 0;

    bool ok = ggml_xkv_landmark_build_cpu_oracle(
        a_k->data, a_k->type, a_k->ne[0], a_k->ne[1], a_k->nb[0], a_k->nb[1],
        b_k->data, b_k->type, b_k->ne[0], b_k->ne[1], b_k->nb[0], b_k->nb[1],
        (const int32_t *)rows->data, rows->ne[0],
        (const int64_t *)pos->data, pos_is_64,
        (const int32_t *)lmeta->data, lmeta->ne[1],
        rope_data, (int64_t)ggml_nelements(rope),
        &p,
        dst->data, dst->ne[0], dst->ne[1], dst->nb[0], dst->nb[1],
        (float *)eb->data, (uint64_t *)srcfp->data, (int32_t *)status->data,
        scr->data, ggml_nbytes(scr),
        err, sizeof(err));
    if (!ok) {
        if (status && status->data && ((const int32_t *)status->data)[0] == 0) {
            fail(GGML_XKV_LANDMARK_BUILD_STATUS_ERR_INPUT);
        }
    }
}
