#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)
// Original-byte Q2_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL/Q4_0/Q5_0/Q8_0
// [K,N,A,B] x F32[K,M,C,D] -> F32[N,M,C,D], C%A == D%B == 0.
// K contains whole quantization blocks. F32 element and all row/batch strides
// are explicit, aligned, and may include padding. No activation quantization
// or full-weight dequantization; accumulators and the reduction tree are F32.
bool ggml_cuda_mul_mat_q8_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_q8_f32(ggml_backend_cuda_context & ctx,
                              const ggml_tensor *         src0,
                              const ggml_tensor *         src1,
                              ggml_tensor *               dst);

// Device-routed Q2_0[K,N,E] x F32[K,R,T], I32 IDs[U,T] -> F32[N,U,T].
// U%R == 0; activation expert slot is u%R. IDs never leave the device. An
// out-of-range device ID emits NaN for the existing upper-layer finite check.
bool ggml_cuda_mul_mat_id_q2_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1,
                                         const ggml_tensor * ids, const ggml_tensor * dst);
// Reuse the short down-projection kernel when its complete layout is supported.
// Both native and decoded-F32 policies may select it; false submits no work.
bool ggml_cuda_try_mul_mat_id_q2_f32_short(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                          const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
// Exact T1 K128/M2560/top10 down plus routing-weighted expert reduction.
// The caller closes the complete graph chain; false submits no work.
bool ggml_cuda_try_mul_mat_id_q2_down_sum(ggml_backend_cuda_context & ctx,
        const ggml_tensor * down, const ggml_tensor * routing, ggml_tensor * out);
void ggml_cuda_mul_mat_id_q2_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                               const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
#endif

// The caller must first apply upstream subgraph-use and allocation checks.
// Unsupported shapes retain the separate original-byte/F32 projections.
bool ggml_cuda_try_mul_mat_id_q2_f32_glu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * dst);
