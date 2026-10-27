#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_banded_supported(int device, const ggml_tensor * dst);

// F32 Q/math: head128/GQA4 requires F32 K/V; Swift head256/GQA12 accepts
// matching F16 or F32 K/V through the same fused online kernel.
bool ggml_cuda_flash_attn_ext_f32_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
