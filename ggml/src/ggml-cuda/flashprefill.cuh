#pragma once

#include "common.cuh"

bool ggml_cuda_flash_prefill_pool_supported(int device, const ggml_tensor * dst);
bool ggml_cuda_flash_prefill_select_supported(int device, const ggml_tensor * dst);
bool ggml_cuda_flash_prefill_attn_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_prefill_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_flash_prefill_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_flash_prefill_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
