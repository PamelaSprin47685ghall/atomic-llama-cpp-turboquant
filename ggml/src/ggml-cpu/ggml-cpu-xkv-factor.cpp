// ggml-cpu-xkv-factor.cpp — CPU forwards for GGML_OP_XKV_FACTORIZE / GGML_OP_XKV_CANONICALIZE.
//
// Single-threaded (n_tasks=1). The oracle is heap-based (independent reference
// algorithm); no graph-workspace sizing needed on CPU. Failure contract:
// outputs are left untouched ("unpublished") and the status word carries
// nonzero (1 = validation, 2 = compute/encode). Success writes status 0.
// Never throws across the C ABI.

#include "ggml-cpu-impl.h"
#include "ggml-cpu-xkv-factor.h"
#include "ops.h"

#include "ggml-xkv-factor.h"

#include <cmath>
#include <cstring>
#include <vector>

void ggml_compute_forward_xkv_factorize(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }
    const struct ggml_tensor * x       = dst->src[0];
    const struct ggml_tensor * scratch = dst->src[1];
    struct ggml_tensor * b_out  = dst->src[2];
    struct ggml_tensor * status = dst->src[3];

    ggml_xkv_factorize_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    auto fail = [&](int32_t code) {
        if (status && status->data) {
            ((int32_t *)status->data)[0] = code;
        }
    };

    if (!x || !scratch || !b_out || !status) {
        fail(1);
        return;
    }
    char err[256] = {0};
    if (!ggml_xkv_factorize_supports(x, scratch, b_out, status, dst, &p, err, sizeof(err))) {
        fail(1);
        return;
    }

    uint32_t n = p.rows_n, m = p.cols_m;
    uint32_t r = p.requested_rank;
    uint64_t mn = std::min<uint64_t>(n, m);
    if (r > mn) r = (uint32_t)mn;
    std::vector<float> svals(r, 0.0f);
    ggml_xkv_residual r_ab = {0, 0, 0}, r_a = {0, 0, 0}, r_b = {0, 0, 0};

    bool ok = ggml_xkv_factorize_cpu_oracle_resid(
        (const float *)x->data, n, m, &p,
        dst->data, b_out->data, svals.data(),
        &r_ab, &r_a, &r_b, err, sizeof(err));
    if (ok && status && status->data) {
        // status[0] set by fail() below; residuals as float bits.
        float * words = (float *)status->data;
        words[1] = r_ab.frob_orig; words[2] = r_ab.frob_err; words[3] = r_ab.max_err;
        words[4] = r_a.frob_orig;  words[5] = r_a.frob_err;  words[6] = r_a.max_err;
        words[7] = r_b.frob_orig;  words[8] = r_b.frob_err;  words[9] = r_b.max_err;
    }
    if (ok && scratch && scratch->data) {
        // Write S telemetry to scratch at exact off_s (mirrors Vulkan step 7)
        uint64_t l = (uint64_t)r + p.oversampling;
        uint64_t mn_u64 = std::min<uint64_t>(n, m);
        if (l > mn_u64) l = mn_u64;
        uint64_t off_s = (uint64_t)n * l + (uint64_t)m * l + (uint64_t)l * m + 2ULL * l * l;
        float * scr_f = (float *)scratch->data;
        for (uint32_t i = 0; i < r; ++i) scr_f[off_s + i] = svals[i];
    }
    fail(ok ? 0 : 2);
}

void ggml_compute_forward_xkv_canonicalize(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }
    const struct ggml_tensor * hot     = dst->src[0];
    const struct ggml_tensor * rows    = dst->src[1];
    const struct ggml_tensor * pos     = dst->src[2];
    const struct ggml_tensor * rope    = dst->src[3];
    const struct ggml_tensor * had     = dst->src[4];
    struct ggml_tensor * status = dst->src[5];

    ggml_xkv_canonicalize_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    auto fail = [&](int32_t code) {
        if (status && status->data) {
            ((int32_t *)status->data)[0] = code;
        }
    };

    if (!hot || !rows || !pos || !rope || !had || !status) {
        fail(1);
        return;
    }
    char err[256] = {0};
    if (!ggml_xkv_canonicalize_supports(hot, rows, pos, rope, had, status, dst, &p, err, sizeof(err))) {
        fail(1);
        return;
    }

    const float * rope_data = ggml_nelements(rope) ? (const float *)rope->data : nullptr;
    const float * had_data  = ggml_nelements(had) ? (const float *)had->data : nullptr;
    int pos_is_64 = (pos->type == GGML_TYPE_I64) ? 1 : 0;

    bool ok = ggml_xkv_canonicalize_cpu_oracle(
        hot->data, hot->type,
        (const int32_t *)rows->data, pos->data, pos_is_64, p.n_rows,
        rope_data, (uint32_t)ggml_nelements(rope),
        had_data, p.hadamard_dim, &p,
        (float *)dst->data, err, sizeof(err));
    fail(ok ? 0 : 2);
}
