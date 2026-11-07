#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cpu_attn_ordered_dot(int64_t n, enum ggml_type value_type, const void * values,
                               const float * weights, enum ggml_type mask_type, const void * mask,
                               float * result);

void ggml_cpu_attn_ordered_score(int64_t n, enum ggml_type key_type, const void * key,
                                 const float * query, enum ggml_type mask_type, const void * mask,
                                 int64_t key_index, float * result);

void ggml_cpu_attn_ordered_softmax(int64_t n, float * dst, const float * scores,
                                   enum ggml_type mask_type, const void * mask,
                                   float scale, float slope, const float * sink);

#ifdef __cplusplus
}
#endif
