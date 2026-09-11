#pragma once

#include "common.cuh"

bool ggml_cuda_xkv_factorize_supports(const ggml_tensor * op);
void ggml_cuda_xkv_factorize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_xkv_canonicalize_supports(const ggml_tensor * op);
void ggml_cuda_xkv_canonicalize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_xkv_landmark_build_supports(const ggml_tensor * op);
void ggml_cuda_xkv_landmark_build(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
