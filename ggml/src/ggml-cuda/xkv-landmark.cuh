#pragma once

#include "common.cuh"

bool ggml_cuda_xkv_landmark_supports(const ggml_tensor * op);
void ggml_cuda_xkv_landmark(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_xkv_landmark_rows_supports(const ggml_tensor * op);
void ggml_cuda_xkv_landmark_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_xkv_landmark_merge_supports(const ggml_tensor * op);
void ggml_cuda_xkv_landmark_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
