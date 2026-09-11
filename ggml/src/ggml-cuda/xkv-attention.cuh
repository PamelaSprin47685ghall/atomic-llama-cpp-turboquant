#pragma once

#include "common.cuh"

bool ggml_cuda_xkv_attention_supports(const ggml_tensor * op);
void ggml_cuda_xkv_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
