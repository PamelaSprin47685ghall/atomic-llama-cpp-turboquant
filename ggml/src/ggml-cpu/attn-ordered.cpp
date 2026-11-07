#include "attn-ordered.h"
#include "vec.h"

#include <cmath>

// A finite-only or reassociating compiler can discard masks or move terms
// across them. This file is compiled with strict floating-point options.
#if defined(__FAST_MATH__)
#error "ordered CPU attention requires strict floating-point arithmetic"
#endif

static inline float attn_mask_value(const void * mask, ggml_type type, int64_t i) {
    if (type == GGML_TYPE_F16) {
        return GGML_COMPUTE_FP16_TO_FP32(((const ggml_fp16_t *) mask)[i]);
    }
    return ((const float *) mask)[i];
}

void ggml_cpu_attn_ordered_dot(int64_t n, ggml_type value_type, const void * values,
                               const float * weights, ggml_type mask_type, const void * mask,
                               float * result) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (attn_mask_value(mask, mask_type, i) == -INFINITY) {
            continue;
        }
        float value;
        float weight = weights[i];
        if (value_type == GGML_TYPE_F16) {
            value = GGML_COMPUTE_FP16_TO_FP32(((const ggml_fp16_t *) values)[i]);
            // Match the F16 value product's probability conversion boundary.
            weight = GGML_COMPUTE_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(weight));
        } else {
            value = ((const float *) values)[i];
        }
        // Also evaluate unmasked zero weights: 0 * NaN/Inf is not a mask.
        sum += (double) value * (double) weight;
    }
    *result = (float) sum;
}

void ggml_cpu_attn_ordered_score(int64_t n, ggml_type key_type, const void * key,
                                 const float * query, ggml_type mask_type, const void * mask,
                                 int64_t key_index, float * result) {
    if (attn_mask_value(mask, mask_type, key_index) == -INFINITY) {
        // The paired softmax applies the mask after any score transforms.
        // Do not read K or Q for this key, including stale nonfinite K.
        *result = 0.0f;
        return;
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float k = key_type == GGML_TYPE_F16
            ? GGML_COMPUTE_FP16_TO_FP32(((const ggml_fp16_t *) key)[i])
            : ((const float *) key)[i];
        sum += (double) k * (double) query[i];
    }
    *result = (float) sum;
}

void ggml_cpu_attn_ordered_softmax(int64_t n, float * dst, const float * scores,
                                   ggml_type mask_type, const void * mask,
                                   float scale, float slope, const float * sink) {
    float max = -INFINITY;
    bool visible = false;
    for (int64_t i = 0; i < n; ++i) {
        const float m = attn_mask_value(mask, mask_type, i);
        if (m == -INFINITY) {
            dst[i] = 0.0f;
            continue;
        }
        visible = true;
        const float score = scores[i] * scale + slope * m;
        dst[i] = score;
        if (std::isnan(score) || score > max) {
            max = score;
        }
    }
    if (sink && (std::isnan(*sink) || *sink > max)) {
        max = *sink;
    }
    if (!visible && !sink) {
        return;
    }

    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (attn_mask_value(mask, mask_type, i) == -INFINITY) {
            continue;
        }
        const float weight = std::exp(dst[i] - max);
        dst[i] = weight;
        sum += (double) weight;
    }
    if (sink) {
        sum += (double) std::exp(*sink - max);
    }
    const float inv_sum = (float) (1.0 / sum);
    for (int64_t i = 0; i < n; ++i) {
        if (attn_mask_value(mask, mask_type, i) != -INFINITY) {
            dst[i] *= inv_sum;
        }
    }
}
