// ggml-cpu-xkv-landmark.h — forward declaration for the XKV landmark CPU
// kernel (wired into ggml-cpu.c by the owner of that switch; keeps ops.h untouched).

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;
struct ggml_tensor;

void ggml_compute_forward_xkv_landmark(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_xkv_landmark_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_xkv_landmark_merge(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
