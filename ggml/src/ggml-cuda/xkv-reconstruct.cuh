#pragma once

#include "common.cuh"

bool ggml_cuda_xkv_reconstruct_supports(const ggml_tensor * op);
void ggml_cuda_xkv_reconstruct(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
