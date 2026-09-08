#pragma once

#include "common.cuh"

void ggml_cuda_flash_prefill_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_flash_prefill_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_flash_prefill_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
