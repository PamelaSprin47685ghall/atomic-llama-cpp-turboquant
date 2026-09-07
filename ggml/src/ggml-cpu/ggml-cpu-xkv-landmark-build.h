// ggml-cpu-xkv-landmark-build.h — forward declaration for the XKV
// landmark-build CPU kernel (wired into ggml-cpu.c by the shared-switch
// integration owner; keeps ops.h untouched).

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;
struct ggml_tensor;

void ggml_compute_forward_xkv_landmark_build(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
