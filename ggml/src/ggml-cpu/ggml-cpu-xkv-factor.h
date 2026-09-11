// ggml-cpu-xkv-factor.h — forward declarations for the XKV factorize/
// canonicalize CPU kernels (wired into ggml-cpu.c by the shared-switch
// integration owner; keeps ops.h untouched).

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;
struct ggml_tensor;

void ggml_compute_forward_xkv_factorize(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_xkv_canonicalize(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
