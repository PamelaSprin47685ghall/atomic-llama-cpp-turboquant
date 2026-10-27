#include "mmvq-q8-f32.cuh"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "unary.cuh"

#include <climits>
#include <limits>
#include <type_traits>

namespace {
constexpr int q8_f32_lanes          = 32;
constexpr int q8_f32_rows_per_block = 4;
constexpr int q8_f32_values         = 4;
constexpr int q8_f32_threads        = q8_f32_lanes * q8_f32_rows_per_block;

template <ggml_type type>
__device__ __forceinline__ float decode(const char * row, int64_t i);

template <ggml_type type>
__device__ __forceinline__ float4 decode_four(const char * row, int64_t i) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        // Preserve the original Q8_0 packed decoder, including its two-byte
        // alignment requirement and explicit RN scale multiplication.
        const auto & block = reinterpret_cast<const block_q8_0 *>(row)[i / QK8_0];
        const int codes = get_int_b2(block.qs, int(i % QK8_0) / q8_f32_values);
        const float scale = __half2float(block.d);
        return make_float4(
            __fmul_rn(scale, float(int8_t(codes >>  0))),
            __fmul_rn(scale, float(int8_t(codes >>  8))),
            __fmul_rn(scale, float(int8_t(codes >> 16))),
            __fmul_rn(scale, float(int8_t(codes >> 24))));
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const auto & block = reinterpret_cast<const block_q2_0 *>(row)[i / QK2_0];
        const int codes = block.qs[(i % QK2_0) / q8_f32_values];
        const float scale = __half2float(block.d);
        return make_float4(
            __fmul_rn(float(((codes >> 0) & 3) - 1), scale),
            __fmul_rn(float(((codes >> 2) & 3) - 1), scale),
            __fmul_rn(float(((codes >> 4) & 3) - 1), scale),
            __fmul_rn(float(((codes >> 6) & 3) - 1), scale));
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        const auto & b = reinterpret_cast<const block_q3_K *>(row)[i / QK_K];
        const int j = int(i % QK_K);
        const int s = j / 16;
        const int us = ((b.scales[s & 7] >> (4 * (s >> 3))) & 15) |
                       (((b.scales[8 + (s & 3)] >> (2 * (s >> 2))) & 3) << 4);
        const uint32_t codes = uint32_t(get_int_b2(b.qs, ((j / 128) * 32 + j % 32) / 4));
        const uint32_t high = uint32_t(get_int_b2(b.hmask, (j % 32) / 4));
        const float d = __fmul_rn(__half2float(b.d), float(us - 32));
        float values[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int q = ((codes >> (8*c + 2*((j % 128) / 32))) & 3) -
                          (((high >> (8*c + j/32)) & 1) ? 0 : 4);
            values[c] = __fmul_rn(d, float(q));
        }
        return make_float4(values[0], values[1], values[2], values[3]);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        using block = typename std::conditional<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>::type;
        const auto & b = reinterpret_cast<const block *>(row)[i / QK_K];
        const int j = int(i % QK_K);
        uint8_t sc, m;
        get_scale_min_k4(j / 32, b.scales, sc, m);
        const float d = __fmul_rn(__half2float(__low2half(b.dm)), float(sc));
        const float min = __fmul_rn(__half2float(__high2half(b.dm)), float(m));
        const uint32_t codes = uint32_t(get_int_b2(b.qs, ((j / 64) * 32 + j % 32) / 4));
        uint32_t high = 0;
        if constexpr (type == GGML_TYPE_Q5_K) {
            high = uint32_t(get_int_b2(b.qh, (j % 32) / 4));
        }
        float values[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            int q = (codes >> (8*c + ((j % 64) / 32)*4)) & 15;
            if constexpr (type == GGML_TYPE_Q5_K) {
                q |= ((high >> (8*c + j/32)) & 1) << 4;
            }
            values[c] = __fadd_rn(__fmul_rn(d, float(q)), -min);
        }
        return make_float4(values[0], values[1], values[2], values[3]);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const auto & b = reinterpret_cast<const block_q6_K *>(row)[i / QK_K];
        const int j = int(i % QK_K);
        const int group = (j % 128) / 32;
        const uint32_t low = uint32_t(get_int_b2(b.ql, ((j / 128) * 64 + (group & 1) * 32 + j % 32) / 4));
        const uint32_t high = uint32_t(get_int_b2(b.qh, ((j / 128) * 32 + j % 32) / 4));
        const float d = __fmul_rn(__half2float(b.d), float(b.scales[j / 16]));
        float values[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int q = ((low >> (8*c + (group / 2)*4)) & 15) |
                          (((high >> (8*c + 2*group)) & 3) << 4);
            values[c] = __fmul_rn(d, float(q - 32));
        }
        return make_float4(values[0], values[1], values[2], values[3]);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const auto & b = reinterpret_cast<const block_iq4_xs *>(row)[i / QK_K];
        const int j = int(i % QK_K);
        const int s = j / 32;
        const int scale = ((b.scales_l[s / 2] >> (4 * (s % 2))) & 15) |
                          (((b.scales_h >> (2 * s)) & 3) << 4);
        const float d = __fmul_rn(__half2float(b.d), float(scale - 32));
        const uint32_t codes = uint32_t(get_int_b2(b.qs, (s * 16 + j % 16) / 4));
        float values[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int q = (codes >> (8*c + ((j % 32) / 16)*4)) & 15;
            values[c] = __fmul_rn(d, float(kvalues_iq4nl[q]));
        }
        return make_float4(values[0], values[1], values[2], values[3]);
    } else {
        return make_float4(decode<type>(row, i + 0), decode<type>(row, i + 1),
                           decode<type>(row, i + 2), decode<type>(row, i + 3));
    }
}

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float q2_k128_lane_sum(const char * row, int lane, float4 input) {
    const float4 weight = decode_four<GGML_TYPE_Q2_0>(row, 4 * lane);
    float p0, p1, p2, p3, sum;
    // Match the retained gfx1030 K128 ISA: four FMA(+0) chains followed by
    // ((p3 + p1) + p0) + p2. HIP RN intrinsics alone do not lock this tree.
    asm("v_fma_f32 %0, %1, %2, 0" : "=v"(p0) : "v"(weight.x), "v"(input.x));
    asm("v_fma_f32 %0, %1, %2, 0" : "=v"(p1) : "v"(weight.y), "v"(input.y));
    asm("v_fma_f32 %0, %1, %2, 0" : "=v"(p2) : "v"(weight.z), "v"(input.z));
    asm("v_fma_f32 %0, %1, %2, 0" : "=v"(p3) : "v"(weight.w), "v"(input.w));
    asm("v_add_f32 %0, %1, %2" : "=v"(sum) : "v"(p3), "v"(p1));
    asm("v_add_f32 %0, %1, %2" : "=v"(sum) : "v"(sum), "v"(p0));
    asm("v_add_f32 %0, %1, %2" : "=v"(sum) : "v"(sum), "v"(p2));
    return sum;
}

__global__ void mul_mat_id_q2_k128_rows2(
        const char * weights, const char * input, const char * ids, char * output,
        int nexperts, size_t weight_row, size_t weight_expert,
        size_t input_slot, size_t id_slot, size_t output_slot, size_t output_row) {
    const int lane = threadIdx.x % q8_f32_lanes;
    // The exact 2560-row admission and 320-block grid contain only full pairs.
    const int row = 8 * blockIdx.x + 2 * (threadIdx.x / q8_f32_lanes);
    const size_t slot = blockIdx.y;
    const int expert = *reinterpret_cast<const int32_t *>(ids + slot * id_slot);
    char * dst = output + slot * output_slot + size_t(row) * output_row;
    if (expert < 0 || expert >= nexperts) {
        if (lane == 0) {
            *reinterpret_cast<float *>(dst) = __int_as_float(0x7fc00000);
            *reinterpret_cast<float *>(dst + output_row) = __int_as_float(0x7fc00000);
        }
        return;
    }
    float4 values;
    const auto * x_values = reinterpret_cast<const float *>(input + slot * input_slot);
    // Preserve the existing four-byte input alignment, including padded slots.
    __builtin_memcpy(&values, x_values + 4 * lane, sizeof(values));
    const char * wr = weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
    float sum0 = q2_k128_lane_sum(wr, lane, values);
    float sum1 = q2_k128_lane_sum(wr + weight_row, lane, values);
#pragma unroll
    for (int offset = q8_f32_lanes / 2; offset > 0; offset >>= 1) {
        const float other0 = __shfl_xor_sync(0xffffffff, sum0, offset, q8_f32_lanes);
        const float other1 = __shfl_xor_sync(0xffffffff, sum1, offset, q8_f32_lanes);
        asm("v_add_f32 %0, %1, %2" : "=v"(sum0) : "v"(sum0), "v"(other0));
        asm("v_add_f32 %0, %1, %2" : "=v"(sum1) : "v"(sum1), "v"(other1));
    }
    if (lane == 0) {
        *reinterpret_cast<float *>(dst) = sum0;
        *reinterpret_cast<float *>(dst + output_row) = sum1;
    }
}

// One wave owns one selected expert slot and eight output rows. The ten waves
// compute every original dot exactly once, then join their weighted results.
__global__ void q2_down_weighted_sum_k128_t1(
        const char * weights, const char * input, const char * ids,
        const char * routing, char * output, int nexperts,
        size_t weight_row, size_t weight_expert, size_t input_slot,
        size_t id_slot, size_t routing_slot, size_t output_row) {
    const int lane = threadIdx.x % q8_f32_lanes;
    const int slot = threadIdx.x / q8_f32_lanes;
    const int row = 8 * blockIdx.x;
    __shared__ float weighted[10][8];

    const int expert = *reinterpret_cast<const int32_t *>(ids + size_t(slot) * id_slot);
    float sum0;
    float sum1;
    float sum2;
    float sum3;
    float sum4;
    float sum5;
    float sum6;
    float sum7;
    if (expert < 0 || expert >= nexperts) {
        sum0 = __int_as_float(0x7fc00000);
        sum1 = sum0;
        sum2 = sum0;
        sum3 = sum0;
        sum4 = sum0;
        sum5 = sum0;
        sum6 = sum0;
        sum7 = sum0;
    } else {
        float4 values;
        const float * x = reinterpret_cast<const float *>(input + size_t(slot) * input_slot);
        __builtin_memcpy(&values, x + 4 * lane, sizeof(values));
        const char * wr = weights + size_t(expert) * weight_expert + size_t(row) * weight_row;
        sum0 = q2_k128_lane_sum(wr, lane, values);
        sum1 = q2_k128_lane_sum(wr + weight_row, lane, values);
        sum2 = q2_k128_lane_sum(wr + 2 * weight_row, lane, values);
        sum3 = q2_k128_lane_sum(wr + 3 * weight_row, lane, values);
        sum4 = q2_k128_lane_sum(wr + 4 * weight_row, lane, values);
        sum5 = q2_k128_lane_sum(wr + 5 * weight_row, lane, values);
        sum6 = q2_k128_lane_sum(wr + 6 * weight_row, lane, values);
        sum7 = q2_k128_lane_sum(wr + 7 * weight_row, lane, values);
#pragma unroll
        for (int offset = q8_f32_lanes / 2; offset > 0; offset >>= 1) {
            sum0 += __shfl_xor_sync(0xffffffff, sum0, offset, q8_f32_lanes);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, offset, q8_f32_lanes);
            sum2 += __shfl_xor_sync(0xffffffff, sum2, offset, q8_f32_lanes);
            sum3 += __shfl_xor_sync(0xffffffff, sum3, offset, q8_f32_lanes);
            sum4 += __shfl_xor_sync(0xffffffff, sum4, offset, q8_f32_lanes);
            sum5 += __shfl_xor_sync(0xffffffff, sum5, offset, q8_f32_lanes);
            sum6 += __shfl_xor_sync(0xffffffff, sum6, offset, q8_f32_lanes);
            sum7 += __shfl_xor_sync(0xffffffff, sum7, offset, q8_f32_lanes);
        }
    }
    if (lane == 0) {
        const float alpha = *reinterpret_cast<const float *>(routing + size_t(slot) * routing_slot);
        weighted[slot][0] = alpha * sum0;
        weighted[slot][1] = alpha * sum1;
        weighted[slot][2] = alpha * sum2;
        weighted[slot][3] = alpha * sum3;
        weighted[slot][4] = alpha * sum4;
        weighted[slot][5] = alpha * sum5;
        weighted[slot][6] = alpha * sum6;
        weighted[slot][7] = alpha * sum7;
    }
    // Invalid IDs must also reach this barrier; their NaN contribution keeps
    // the existing finite-check behavior without reading invalid weight memory.
    __syncthreads();
    if (threadIdx.x < 8) {
        const int column = threadIdx.x;
        float result = weighted[0][column];
#pragma unroll
        for (int s = 1; s < 10; ++s) {
            result += weighted[s][column];
        }
        *reinterpret_cast<float *>(output + size_t(row + column) * output_row) = result;
    }
}

// Preserve the four accumulator chains' actual gfx1030 combine order. The
// three RN-looking additions in the generic source are reassociated by HIP.
__device__ __forceinline__ float q2_f32_combine(float4 sums) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(sums.w), "v"(sums.y));
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(result), "v"(sums.x));
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(result), "v"(sums.z));
    return result;
}

__global__ void mul_mat_id_q2_f32_swiglu(
        const char * gate_weights, const char * up_weights, const char * input,
        const char * ids, char * output, int nexperts,
        size_t weight_row, size_t weight_expert, size_t input_slot,
        size_t id_slot, size_t output_slot) {
    const int lane = threadIdx.x % q8_f32_lanes;
    const int row = q8_f32_rows_per_block * blockIdx.x + threadIdx.x / q8_f32_lanes;
    const size_t slot = blockIdx.y;
    const int expert = *reinterpret_cast<const int32_t *>(ids + slot * id_slot);
    float * dst = reinterpret_cast<float *>(output + slot * output_slot) + row;
    if (expert < 0 || expert >= nexperts) {
        if (lane == 0) {
            *dst = __int_as_float(0x7fc00000);
        }
        return;
    }
    const size_t weight_offset = size_t(expert) * weight_expert + size_t(row) * weight_row;
    const char * gate_row = gate_weights + weight_offset;
    const char * up_row = up_weights + weight_offset;
    const float * x = reinterpret_cast<const float *>(input + slot * input_slot);
    float4 gate_sum = make_float4(0, 0, 0, 0);
    float4 up_sum = make_float4(0, 0, 0, 0);
#pragma unroll 1
    for (int i = q8_f32_values * lane; i < 2560; i += q8_f32_values * q8_f32_lanes) {
        float4 values;
        __builtin_memcpy(&values, x + i, sizeof(values));
        const float4 gate = decode_four<GGML_TYPE_Q2_0>(gate_row, i);
        const float4 up = decode_four<GGML_TYPE_Q2_0>(up_row, i);
        gate_sum.x = __fmaf_rn(gate.x, values.x, gate_sum.x);
        gate_sum.y = __fmaf_rn(gate.y, values.y, gate_sum.y);
        gate_sum.z = __fmaf_rn(gate.z, values.z, gate_sum.z);
        gate_sum.w = __fmaf_rn(gate.w, values.w, gate_sum.w);
        up_sum.x = __fmaf_rn(up.x, values.x, up_sum.x);
        up_sum.y = __fmaf_rn(up.y, values.y, up_sum.y);
        up_sum.z = __fmaf_rn(up.z, values.z, up_sum.z);
        up_sum.w = __fmaf_rn(up.w, values.w, up_sum.w);
    }
    float gate = q2_f32_combine(gate_sum);
    float up = q2_f32_combine(up_sum);
#pragma unroll
    for (int offset = q8_f32_lanes / 2; offset > 0; offset >>= 1) {
        const float gate_other = __shfl_xor_sync(0xffffffff, gate, offset, q8_f32_lanes);
        const float up_other = __shfl_xor_sync(0xffffffff, up, offset, q8_f32_lanes);
        asm("v_add_f32 %0, %1, %2" : "=v"(gate) : "v"(gate), "v"(gate_other));
        asm("v_add_f32 %0, %1, %2" : "=v"(up) : "v"(up), "v"(up_other));
    }
    if (lane == 0) {
        *dst = ggml_cuda_op_silu_single(gate) * up;
    }
}


#endif


template <ggml_type type, int tokens>
__global__ void  mul_mat_q8_f32(const char * weights,
                                const char * input,
                                char *       output,
                                int64_t      k,
                                int64_t      n,
                                size_t       weight_stride,
                                size_t       input_stride,
                                size_t       output_stride) {
    // The HIP target otherwise enables unsafe FP reassociation globally. Keep the
    // four F32 accumulator chains and the wave addition tree in their specified order.
#pragma clang fp reassociate(off) contract(off)
    const int     lane  = threadIdx.x % q8_f32_lanes;
    const int64_t row_i = int64_t(blockIdx.x) * q8_f32_rows_per_block + threadIdx.x / q8_f32_lanes;
    if (row_i >= n) {
        return;
    }
    const char * row = weights + size_t(row_i) * weight_stride;
    const size_t first_token = size_t(blockIdx.y) * tokens;
    float sums[tokens][q8_f32_values] = {};

    // One wave owns a row. Each lane reads four adjacent codes and F32 inputs,
    // reusing the decoded weights for all columns. Every admitted format has
    // a block size divisible by four, so these groups have no partial K tail.
    for (int64_t i = q8_f32_values * lane; i < k; i += q8_f32_values * q8_f32_lanes) {
        const float4 weight = decode_four<type>(row, i);
#pragma unroll
        for (int token = 0; token < tokens; ++token) {
            const auto * x = reinterpret_cast<const float *>(input + (first_token + token) * input_stride);
            // A padded token stride need only preserve four-byte alignment.
            // memcpy permits a vector load without asserting float4 alignment.
            float4 values;
            __builtin_memcpy(&values, x + i, sizeof(values));
            sums[token][0] = __fmaf_rn(weight.x, values.x, sums[token][0]);
            sums[token][1] = __fmaf_rn(weight.y, values.y, sums[token][1]);
            sums[token][2] = __fmaf_rn(weight.z, values.z, sums[token][2]);
            sums[token][3] = __fmaf_rn(weight.w, values.w, sums[token][3]);
        }
    }

    // The native MMVQ wave-shuffle tree needs no LDS or block barrier when each
    // wave owns its output row. Use explicit RN additions under the HIP flags.
#pragma unroll
    for (int token = 0; token < tokens; ++token) {
        float sum = __fadd_rn(__fadd_rn(sums[token][0], sums[token][2]),
                              __fadd_rn(sums[token][1], sums[token][3]));
#pragma unroll
        for (int offset = q8_f32_lanes / 2; offset > 0; offset /= 2) {
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset, q8_f32_lanes));
        }
        if (lane == 0) {
            auto * y      = reinterpret_cast<float *>(output + (first_token + token) * output_stride);
            y[row_i] = sum;
        }
    }
}

template <ggml_type type, int tokens>
void launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * y,
            size_t input_stride, size_t output_stride, int64_t columns, int64_t first_column) {
    const dim3 blocks(unsigned((w->ne[1] + q8_f32_rows_per_block - 1) / q8_f32_rows_per_block),
                      unsigned(columns / tokens));
    mul_mat_q8_f32<type, tokens><<<blocks, q8_f32_threads, 0, ctx.stream(ctx.device, 0)>>>(
        static_cast<const char *>(w->data), static_cast<const char *>(x->data) + size_t(first_column) * input_stride,
        static_cast<char *>(y->data) + size_t(first_column) * output_stride,
        w->ne[0], w->ne[1], w->nb[1], input_stride, output_stride);
}
// Scalar indexing of the existing GGML decoders. The K-format mappings below
// are the same bit/scale mappings as dequantize.cuh, without an intermediate
// decoded block. Explicit RN operations prohibit the global unsafe HIP flags
// from folding scale/min arithmetic into the dot product.
template <ggml_type type>
__device__ __forceinline__ float decode(const char * row, int64_t i) {
    if constexpr (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q8_0) {
        float2 v;
        if constexpr (type == GGML_TYPE_Q8_0) {
            dequantize_q8_0(row, i / QK8_0, int(i % QK8_0) & ~1, v);
            return (i & 1) ? v.y : v.x;
        } else {
            if constexpr (type == GGML_TYPE_Q4_0) {
                dequantize_q4_0(row, i / QK4_0, int(i % 16), v);
            } else {
                dequantize_q5_0(row, i / QK5_0, int(i % 16), v);
            }
            return (i % 32 < 16) ? v.x : v.y;
        }
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        const auto & b = reinterpret_cast<const block_iq4_nl *>(row)[i / QK4_NL];
        const int j = int(i % QK4_NL);
        const int q = (b.qs[j % 16] >> (j / 16 * 4)) & 15;
        return __fmul_rn(__half2float(b.d), float(kvalues_iq4nl[q]));
    } else {
        const int j = int(i % QK_K);
        if constexpr (type == GGML_TYPE_Q3_K) {
            const auto & b = reinterpret_cast<const block_q3_K *>(row)[i / QK_K];
            const int s = j / 16;
            const int us = ((b.scales[s & 7] >> (4 * (s >> 3))) & 15) |
                           (((b.scales[8 + (s & 3)] >> (2 * (s >> 2))) & 3) << 4);
            const int q = ((b.qs[(j / 128) * 32 + j % 32] >> (2 * ((j % 128) / 32))) & 3) -
                          ((b.hmask[j % 32] & (1 << (j / 32))) ? 0 : 4);
            return __fmul_rn(__fmul_rn(__half2float(b.d), float(us - 32)), float(q));
        } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            using block = typename std::conditional<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>::type;
            const auto & b = reinterpret_cast<const block *>(row)[i / QK_K];
            uint8_t sc, m;
            get_scale_min_k4(j / 32, b.scales, sc, m);
            int q = (b.qs[(j / 64) * 32 + j % 32] >> ((j % 64) / 32 * 4)) & 15;
            if constexpr (type == GGML_TYPE_Q5_K) {
                q |= (b.qh[j % 32] & (1 << (j / 32))) ? 16 : 0;
            }
            const float d = __fmul_rn(__half2float(__low2half(b.dm)), float(sc));
            const float min = __fmul_rn(__half2float(__high2half(b.dm)), float(m));
            return __fadd_rn(__fmul_rn(d, float(q)), -min);
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            const auto & b = reinterpret_cast<const block_q6_K *>(row)[i / QK_K];
            const int group = (j % 128) / 32;
            const int l = j % 32;
            const int lo = b.ql[(j / 128) * 64 + (group & 1) * 32 + l];
            const int hi = b.qh[(j / 128) * 32 + l];
            const int q = ((lo >> ((group / 2) * 4)) & 15) | (((hi >> (2 * group)) & 3) << 4);
            return __fmul_rn(__fmul_rn(__half2float(b.d), float(b.scales[j / 16])), float(q - 32));
        } else {
            static_assert(type == GGML_TYPE_IQ4_XS, "unsupported original-byte decoder");
            const auto & b = reinterpret_cast<const block_iq4_xs *>(row)[i / QK_K];
            const int s = j / 32;
            const int scale = ((b.scales_l[s / 2] >> (4 * (s % 2))) & 15) |
                              (((b.scales_h >> (2 * s)) & 3) << 4);
            const int q = (b.qs[s * 16 + j % 16] >> ((j % 32) / 16 * 4)) & 15;
            return __fmul_rn(__fmul_rn(__half2float(b.d), float(scale - 32)), float(kvalues_iq4nl[q]));
        }
    }
}

struct tensor_layout {
    int64_t ne[4];
    size_t nb[4];
};

tensor_layout layout(const ggml_tensor * t) {
    tensor_layout result;
    for (int d = 0; d < 4; ++d) {
        result.ne[d] = t->ne[d];
        result.nb[d] = t->nb[d];
    }
    return result;
}

template <ggml_type type, int tokens, bool indexed, bool contiguous_input = false>
__global__ void mul_mat_mixed_f32(const char * weights, const char * input, const char * ids, char * output,
                                tensor_layout w, tensor_layout x, tensor_layout id, tensor_layout y) {
#pragma clang fp reassociate(off) contract(off)
    static_assert(!contiguous_input || (indexed && type == GGML_TYPE_Q2_0 && tokens == 1),
                  "contiguous input specialization is only for indexed Q2_0");
    const int lane = threadIdx.x % q8_f32_lanes;
    const int64_t row = int64_t(blockIdx.x) * q8_f32_rows_per_block + threadIdx.x / q8_f32_lanes;
    if (row >= w.ne[1]) {
        return;
    }
    size_t wo, xo, yo;
    int64_t first_token;
    if constexpr (indexed) {
        const int64_t slot = blockIdx.y;
        const int64_t token = blockIdx.z;
        const int expert = *reinterpret_cast<const int32_t *>(ids + size_t(slot) * id.nb[0] + size_t(token) * id.nb[1]);
        yo = size_t(slot) * y.nb[1] + size_t(token) * y.nb[2];
        if (expert < 0 || expert >= w.ne[2]) {
            if (lane == 0) {
                *reinterpret_cast<float *>(output + yo + size_t(row) * y.nb[0]) = __int_as_float(0x7fc00000);
            }
            return;
        }
        wo = size_t(expert) * w.nb[2];
        // ID admission bounds slot and the positive divisor to 65535.
        xo = size_t(uint32_t(slot) % uint32_t(x.ne[1])) * x.nb[1] + size_t(token) * x.nb[2];
        first_token = 0;
    } else {
        const int64_t b2 = blockIdx.z % x.ne[2];
        const int64_t b3 = blockIdx.z / x.ne[2];
        wo = size_t(b2 / (x.ne[2] / w.ne[2])) * w.nb[2] + size_t(b3 / (x.ne[3] / w.ne[3])) * w.nb[3];
        xo = size_t(b2) * x.nb[2] + size_t(b3) * x.nb[3];
        yo = size_t(b2) * y.nb[2] + size_t(b3) * y.nb[3];
        first_token = int64_t(blockIdx.y) * tokens;
    }
    const char * wr = weights + wo + size_t(row) * w.nb[1];
    float sums[tokens][q8_f32_values] = {};
    // Admission requires complete four-value groups; the outer bound also
    // covers partial waves without a redundant bound on every group element.
    for (int64_t i = q8_f32_values * lane; i < w.ne[0]; i += q8_f32_values * q8_f32_lanes) {
        float input_values[q8_f32_values];
        if constexpr (contiguous_input) {
            // Admission guarantees a complete group and four-byte alignment.
            // memcpy does not impose the stronger alignment of a float4 pointer.
            const auto * x_values = reinterpret_cast<const float *>(input + xo);
            __builtin_memcpy(input_values, x_values + i, sizeof(input_values));
        }
        int q2_codes = 0;
        float q2_scale = 0.0f;
        if constexpr (type == GGML_TYPE_Q2_0) {
            // Q2_0 packs four consecutive c-1 symbols into one byte. Since i
            // is a multiple of four and QK2_0 is 64, this group never crosses
            // a block: load its original byte and convert its half scale once.
            const auto & block = reinterpret_cast<const block_q2_0 *>(wr)[i / QK2_0];
            q2_codes = block.qs[(i % QK2_0) / q8_f32_values];
            q2_scale = __half2float(block.d);
        }
#pragma unroll
        for (int v = 0; v < q8_f32_values; ++v) {
            float weight;
            if constexpr (type == GGML_TYPE_Q2_0) {
                weight = __fmul_rn(float(((q2_codes >> (2 * v)) & 3) - 1), q2_scale);
            } else {
                weight = decode<type>(wr, i + v);
            }
#pragma unroll
            for (int t = 0; t < tokens; ++t) {
                if (indexed || first_token + t < x.ne[1]) {
                    const size_t token_offset = indexed ? 0 : size_t(first_token + t) * x.nb[1];
                    float value;
                    if constexpr (contiguous_input) {
                        value = input_values[v];
                    } else {
                        value = *reinterpret_cast<const float *>(input + xo + token_offset + size_t(i + v) * x.nb[0]);
                    }
                    sums[t][v] = __fmaf_rn(weight, value, sums[t][v]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
        float sum = __fadd_rn(__fadd_rn(sums[t][0], sums[t][2]), __fadd_rn(sums[t][1], sums[t][3]));
#pragma unroll
        for (int offset = q8_f32_lanes / 2; offset > 0; offset /= 2) {
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset, q8_f32_lanes));
        }
        if (lane == 0 && (indexed || first_token + t < x.ne[1])) {
            const size_t token_offset = indexed ? 0 : size_t(first_token + t) * y.nb[1];
            *reinterpret_cast<float *>(output + yo + token_offset + size_t(row) * y.nb[0]) = sum;
        }
    }
}

bool mixed_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0: return true;
        default: return false;
    }
}

// Check the complete strided address span without overflow before any launch.
// Higher dimensions can be permuted views; do not assume contiguous batches.
bool valid_layout(const ggml_tensor * t, size_t element, int64_t block_size, size_t alignment) {
    if (reinterpret_cast<uintptr_t>(t->data) % alignment != 0) {
        return false;
    }
    size_t span = element;
    for (int d = 0; d < 4; ++d) {
        if (t->ne[d] <= 0 || t->nb[d] < element || t->nb[d] % alignment != 0) {
            return false;
        }
        if (d == 0 && t->ne[0] % block_size != 0) {
            return false;
        }
        const uint64_t count = uint64_t(t->ne[d] / (d == 0 ? block_size : 1) - 1);
        if (count > (std::numeric_limits<size_t>::max() - span) / t->nb[d]) {
            return false;
        }
        span += size_t(count) * t->nb[d];
    }
    return true;
}

bool nonoverlapping_output(const ggml_tensor * t) {
    // Order axes by byte stride, allowing permuted destinations while rejecting
    // aliases that would make independent output waves race with one another.
    size_t span = sizeof(float);
    unsigned used = 0;
    for (int i = 0; i < 4; ++i) {
        int axis = -1;
        for (int d = 0; d < 4; ++d) {
            if (!(used & (1u << d)) && (axis < 0 || t->nb[d] < t->nb[axis])) {
                axis = d;
            }
        }
        used |= 1u << axis;
        if (t->ne[axis] > 1 && t->nb[axis] < span) {
            return false;
        }
        span += size_t(t->ne[axis] - 1) * t->nb[axis];
    }
    return true;
}

bool valid_operands(const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * y) {
    if (!w || !x || !y || !mixed_type(w->type) || x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32) {
        return false;
    }
    const size_t bytes = ggml_type_size(w->type);
    const int64_t block_size = ggml_blck_size(w->type);
    // Q4_K/Q5_K access their packed half2 scale/min pair, not two independent
    // half values. Every row/batch and the base must preserve that alignment.
    const size_t alignment = w->type == GGML_TYPE_Q4_K ? alignof(block_q4_K) :
                             w->type == GGML_TYPE_Q5_K ? alignof(block_q5_K) : alignof(ggml_half);
    return valid_layout(w, bytes, block_size, alignment) && w->nb[0] == bytes &&
           w->ne[0] % q8_f32_values == 0 &&
           valid_layout(x, sizeof(float), 1, alignof(float)) &&
           valid_layout(y, sizeof(float), 1, alignof(float)) && nonoverlapping_output(y) &&
           w->nb[1] >= size_t(w->ne[0] / block_size) * bytes &&
           w->ne[1] <= INT_MAX && w->ne[0] == x->ne[0] && y->ne[0] == w->ne[1];
}

template <ggml_type type, int tokens>
void launch_mixed(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * y) {
    const dim3 blocks(unsigned((w->ne[1] + q8_f32_rows_per_block - 1) / q8_f32_rows_per_block),
                      unsigned((x->ne[1] + tokens - 1) / tokens), unsigned(x->ne[2] * x->ne[3]));
    mul_mat_mixed_f32<type, tokens, false><<<blocks, q8_f32_threads, 0, ctx.stream(ctx.device, 0)>>>(
        static_cast<const char *>(w->data), static_cast<const char *>(x->data), nullptr, static_cast<char *>(y->data),
        layout(w), layout(x), {}, layout(y));
}

// Axes 1..3 may be flattened only if their actual addresses are an arithmetic
// progression. Ignore singleton-axis strides; the first non-singleton axis
// supplies the column stride, which may include padding. Bound the flattened
// count so four-column groups fit the same 65535-entry grid.y admission limit.
bool compact_column_stride(const ggml_tensor * t, size_t & stride) {
    constexpr int64_t max_columns = 4 * int64_t(65535);
    int64_t columns = 1;
    stride = 0;
    for (int d = 1; d < 4; ++d) {
        if (t->ne[d] == 1) {
            continue;
        }
        if (t->ne[d] > max_columns / columns) {
            return false;
        }
        if (stride == 0) {
            stride = t->nb[d];
        } else if (size_t(columns) > std::numeric_limits<size_t>::max() / stride ||
                   t->nb[d] != size_t(columns) * stride) {
            return false;
        }
        columns *= t->ne[d];
    }
    if (stride == 0) {
        stride = t->nb[1];
    }
    return true;
}

template <ggml_type type>
void launch_compact_tokens(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                           ggml_tensor * y, size_t input_stride, size_t output_stride, int64_t columns) {
    // Full groups need no token bounds or layout arguments in the device ABI.
    // A separate exact-size tail preserves every output without reading padding.
    const int64_t full = columns / 4 * 4;
    if (full != 0) {
        launch<type, 4>(ctx, w, x, y, input_stride, output_stride, full, 0);
    }
    switch (columns - full) {
        case 1: launch<type, 1>(ctx, w, x, y, input_stride, output_stride, 1, full); break;
        case 2: launch<type, 2>(ctx, w, x, y, input_stride, output_stride, 2, full); break;
        case 3: launch<type, 3>(ctx, w, x, y, input_stride, output_stride, 3, full); break;
    }
}

template <ggml_type type>
void launch_mixed_tokens(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * y) {
    switch (x->ne[1]) {
        case 1: launch_mixed<type, 1>(ctx, w, x, y); break;
        case 2: launch_mixed<type, 2>(ctx, w, x, y); break;
        case 3: launch_mixed<type, 3>(ctx, w, x, y); break;
        default: launch_mixed<type, 4>(ctx, w, x, y); break;
    }
}
}  // namespace

bool ggml_cuda_mul_mat_q8_f32_supported(const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * y) {
    if (!valid_operands(w, x, y) || x->ne[1] > 4 * int64_t(65535) ||
        x->ne[2] > 65535 / x->ne[3] || x->ne[2] % w->ne[2] != 0 || x->ne[3] % w->ne[3] != 0) {
        return false;
    }
    return y->ne[1] == x->ne[1] && y->ne[2] == x->ne[2] && y->ne[3] == x->ne[3];
}

void ggml_cuda_mul_mat_q8_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * w,
                            const ggml_tensor * x, ggml_tensor * y) {
    GGML_ASSERT(ctx.quantized_matmul_f32 && ctx.curr_stream_no == 0);
    GGML_ASSERT(ggml_cuda_mul_mat_q8_f32_supported(w, x, y));
    size_t input_stride, output_stride;
    // Shared 2D weights and contiguous element axes are necessary, but not
    // sufficient: prove the complete input/output column address mapping too.
    const bool compact = w->ne[2] == 1 && w->ne[3] == 1 &&
                         x->nb[0] == sizeof(float) && y->nb[0] == sizeof(float) &&
                         compact_column_stride(x, input_stride) && compact_column_stride(y, output_stride);
    const int64_t columns = compact ? x->ne[1] * x->ne[2] * x->ne[3] : 0;
    switch (w->type) {
#define MIXED_CASE(t) \
        case t: \
            if (compact) { \
                launch_compact_tokens<t>(ctx, w, x, y, input_stride, output_stride, columns); \
            } else { \
                launch_mixed_tokens<t>(ctx, w, x, y); \
            } \
            break
        MIXED_CASE(GGML_TYPE_Q2_0);
        MIXED_CASE(GGML_TYPE_Q3_K);
        MIXED_CASE(GGML_TYPE_Q4_K);
        MIXED_CASE(GGML_TYPE_Q5_K);
        MIXED_CASE(GGML_TYPE_Q6_K);
        MIXED_CASE(GGML_TYPE_IQ4_XS);
        MIXED_CASE(GGML_TYPE_IQ4_NL);
        MIXED_CASE(GGML_TYPE_Q4_0);
        MIXED_CASE(GGML_TYPE_Q5_0);
        MIXED_CASE(GGML_TYPE_Q8_0);
#undef MIXED_CASE
        default: GGML_ABORT("unsupported original-byte F32 projection type");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mul_mat_id_q2_f32_supported(const ggml_tensor * w, const ggml_tensor * x,
                                         const ggml_tensor * ids, const ggml_tensor * y) {
    return valid_operands(w, x, y) && w->type == GGML_TYPE_Q2_0 && ids && ids->type == GGML_TYPE_I32 &&
           valid_layout(ids, sizeof(int32_t), 1, alignof(int32_t)) &&
           w->ne[2] <= INT_MAX && w->ne[3] == 1 && x->ne[3] == 1 && y->ne[3] == 1 &&
           ids->ne[2] == 1 && ids->ne[3] == 1 && ids->ne[0] <= 65535 && ids->ne[1] <= 65535 &&
           ids->ne[0] % x->ne[1] == 0 && ids->ne[1] == x->ne[2] &&
           y->ne[1] == ids->ne[0] && y->ne[2] == ids->ne[1];
}

bool ggml_cuda_try_mul_mat_id_q2_f32_short(ggml_backend_cuda_context & ctx, const ggml_tensor * w,
                                          const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * y) {
    if (ctx.curr_stream_no != 0 || !ggml_cuda_mul_mat_id_q2_f32_supported(w, x, ids, y)) {
        return false;
    }
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    const auto & device = ggml_cuda_info().devices[ctx.device];
    if (device.cc == GGML_CUDA_CC_RDNA2 && device.warp_size == q8_f32_lanes &&
        w->ne[0] == 128 && w->ne[1] == 2560 && ids->ne[0] == 10 && ids->ne[1] == 1 &&
        w->ne[2] <= std::numeric_limits<int>::max() &&
        x->nb[0] == sizeof(float) && (x->ne[1] == 1 || x->ne[1] == ids->ne[0])) {
        // The single-token down projection reuses each F32 input for two rows.
        // All other shapes and input mappings retain their existing kernel.
        const size_t input_slot = x->ne[1] == 1 ? 0 : x->nb[1];
        const dim3 paired_blocks(320, 10);
        mul_mat_id_q2_k128_rows2<<<paired_blocks, q8_f32_threads, 0, ctx.stream(ctx.device, 0)>>>(
            static_cast<const char *>(w->data), static_cast<const char *>(x->data),
            static_cast<const char *>(ids->data), static_cast<char *>(y->data),
            int(w->ne[2]), w->nb[1], w->nb[2], input_slot, ids->nb[0], y->nb[1], y->nb[0]);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
#else
    GGML_UNUSED(ctx); GGML_UNUSED(w); GGML_UNUSED(x); GGML_UNUSED(ids); GGML_UNUSED(y);
#endif
    return false;
}


bool ggml_cuda_try_mul_mat_id_q2_down_sum(ggml_backend_cuda_context & ctx,
        const ggml_tensor * down, const ggml_tensor * routing, ggml_tensor * out) {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    if (ctx.quantized_matmul_f32 || ctx.curr_stream_no != 0 ||
            !down || down->op != GGML_OP_MUL_MAT_ID || !routing || !out) {
        return false;
    }
    const ggml_tensor * w = down->src[0];
    const ggml_tensor * x = down->src[1];
    const ggml_tensor * ids = down->src[2];
    const auto & device = ggml_cuda_info().devices[ctx.device];
    if (device.cc != GGML_CUDA_CC_RDNA2 || device.warp_size != q8_f32_lanes ||
            !w || !x || !ids || !w->data || !x->data || !ids->data ||
            !down->data || !routing->data || !out->data ||
            !ggml_cuda_mul_mat_id_q2_f32_supported(w, x, ids, down) ||
            w->ne[0] != 128 || w->ne[1] != 2560 || ids->ne[0] != 10 || ids->ne[1] != 1 ||
            x->nb[0] != sizeof(float) || (x->ne[1] != 1 && x->ne[1] != 10) ||
            routing->type != GGML_TYPE_F32 || routing->ne[0] != 1 || routing->ne[1] != 10 ||
            routing->ne[2] != 1 || routing->ne[3] != 1 ||
            !valid_layout(routing, sizeof(float), 1, alignof(float)) ||
            out->type != GGML_TYPE_F32 || out->ne[0] != 2560 ||
            out->ne[1] != 1 || out->ne[2] != 1 || out->ne[3] != 1 ||
            !ggml_is_contiguous(out) ||
            !valid_layout(out, sizeof(float), 1, alignof(float))) {
        return false;
    }
    // Fusion extends every source lifetime to the final write, so even exact
    // in-place output aliasing with an activation or a broadcast is unsafe.
    const auto disjoint = [out](const ggml_tensor * src) {
        const uintptr_t d = reinterpret_cast<uintptr_t>(out->data);
        const uintptr_t s = reinterpret_cast<uintptr_t>(src->data);
        return d <= s ? ggml_nbytes(out) <= s - d : ggml_nbytes(src) <= d - s;
    };
    if (!disjoint(w) || !disjoint(x) || !disjoint(ids) || !disjoint(routing)) {
        return false;
    }
    const size_t input_slot = x->ne[1] == 1 ? 0 : x->nb[1];
    q2_down_weighted_sum_k128_t1<<<2560 / 8, 10 * q8_f32_lanes, 0, ctx.stream(ctx.device, 0)>>>(
        static_cast<const char *>(w->data), static_cast<const char *>(x->data),
        static_cast<const char *>(ids->data), static_cast<const char *>(routing->data),
        static_cast<char *>(out->data), int(w->ne[2]), w->nb[1], w->nb[2],
        input_slot, ids->nb[0], routing->nb[1], out->nb[0]);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(down); GGML_UNUSED(routing); GGML_UNUSED(out);
    return false;
#endif
}

void ggml_cuda_mul_mat_id_q2_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * w,
                               const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * y) {
    GGML_ASSERT(ctx.quantized_matmul_f32 && ctx.curr_stream_no == 0);
    GGML_ASSERT(ggml_cuda_mul_mat_id_q2_f32_supported(w, x, ids, y));
    if (ggml_cuda_try_mul_mat_id_q2_f32_short(ctx, w, x, ids, y)) {
        return;
    }
    const dim3 blocks(unsigned((w->ne[1] + q8_f32_rows_per_block - 1) / q8_f32_rows_per_block),
                      unsigned(ids->ne[0]), unsigned(ids->ne[1]));
    if (x->nb[0] == sizeof(float)) {
        mul_mat_mixed_f32<GGML_TYPE_Q2_0, 1, true, true><<<blocks, q8_f32_threads, 0, ctx.stream(ctx.device, 0)>>>(
            static_cast<const char *>(w->data), static_cast<const char *>(x->data), static_cast<const char *>(ids->data),
            static_cast<char *>(y->data), layout(w), layout(x), layout(ids), layout(y));
    } else {
        mul_mat_mixed_f32<GGML_TYPE_Q2_0, 1, true><<<blocks, q8_f32_threads, 0, ctx.stream(ctx.device, 0)>>>(
            static_cast<const char *>(w->data), static_cast<const char *>(x->data), static_cast<const char *>(ids->data),
            static_cast<char *>(y->data), layout(w), layout(x), layout(ids), layout(y));
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_try_mul_mat_id_q2_f32_glu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    const auto & device = ggml_cuda_info().devices[ctx.device];
    if (!ctx.quantized_matmul_f32 || ctx.curr_stream_no != 0 ||
        device.cc != GGML_CUDA_CC_RDNA2 || device.warp_size != q8_f32_lanes ||
        !gate || !up || !dst || gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
        dst->op != GGML_OP_GLU || ggml_get_glu_op(dst) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(dst, 1) != 0 || dst->src[0] != gate || dst->src[1] != up ||
        gate->src[1] != up->src[1] || gate->src[2] != up->src[2]) {
        return false;
    }
    const ggml_tensor * wg = gate->src[0];
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * x = up->src[1];
    const ggml_tensor * ids = up->src[2];
    if (!ggml_cuda_mul_mat_id_q2_f32_supported(wg, x, ids, dst) ||
        !ggml_cuda_mul_mat_id_q2_f32_supported(wu, x, ids, dst) ||
        !ggml_are_same_shape(wg, wu) || !ggml_are_same_stride(wg, wu) ||
        wg->ne[0] != 2560 || wg->ne[1] != 128 || ids->ne[0] != 10 || ids->ne[1] != 1 ||
        x->nb[0] != sizeof(float) || (x->ne[1] != 1 && x->ne[1] != ids->ne[0]) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    // The caller also applies the upstream graph-use and allocation checks.
    // Retain a direct guard for leaf/input buffers, including an external x.
    const auto disjoint = [dst](const ggml_tensor * src) {
        if (!dst->data || !src->data) {
            return false;
        }
        const uintptr_t d = reinterpret_cast<uintptr_t>(dst->data);
        const uintptr_t s = reinterpret_cast<uintptr_t>(src->data);
        return d <= s ? ggml_nbytes(dst) <= s - d : ggml_nbytes(src) <= d - s;
    };
    if (!disjoint(wg) || !disjoint(wu) || !disjoint(x) || !disjoint(ids)) {
        return false;
    }
    const size_t input_slot = x->ne[1] == 1 ? 0 : x->nb[1];
    const dim3 blocks(128 / q8_f32_rows_per_block, 10);
    mul_mat_id_q2_f32_swiglu<<<blocks, q8_f32_threads, 0, ctx.stream(ctx.device, 0)>>>(
        static_cast<const char *>(wg->data), static_cast<const char *>(wu->data),
        static_cast<const char *>(x->data), static_cast<const char *>(ids->data),
        static_cast<char *>(dst->data), int(wg->ne[2]), wg->nb[1], wg->nb[2],
        input_slot, ids->nb[0], dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(gate); GGML_UNUSED(up); GGML_UNUSED(dst);
    return false;
#endif
}
