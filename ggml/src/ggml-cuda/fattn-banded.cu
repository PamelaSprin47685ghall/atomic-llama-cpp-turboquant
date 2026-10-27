#include "common.cuh"
#include "fattn-banded.cuh"
#include "fattn.cuh"

#include <cstdint>
#include <type_traits>

static __device__ __forceinline__ float fattn_banded_load(
        const char * ptr, const int type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return __half2float(*(const half *) ptr);
        case GGML_TYPE_BF16: {
            // Read BF16 as raw bits so this kernel needs no native BF16 support; all math stays FP32.
            const uint32_t bits = uint32_t(*(const uint16_t *) ptr) << 16;
            return __uint_as_float(bits);
        }
        default:
            return 0.0f;
    }
}

template<int D, int WARPS_PER_BLOCK, bool PLAIN_F16 = false>
static __global__ void flash_attn_ext_banded_f32(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        float      * __restrict__ dst,
        float scale,
        int type_k,
        int type_v,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_batch,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb0,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb0,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3) {
    constexpr int values_per_lane = D / WARP_SIZE;
    static_assert(D == 64 || D == 128 || D == 256, "F32 FA supports head dimensions 64, 128 and 256");
    static_assert(D % WARP_SIZE == 0, "head dimension must be divisible by warp size");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t iq = int64_t(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (iq >= n_q || ih >= n_head_q || ib >= n_batch) {
        return;
    }

    const int64_t ih_kv = ih / (n_head_q / n_head_kv);
    const char * q_row = q + uint64_t(iq)*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3;

    float q_reg[values_per_lane];
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        q_reg[j] = *(const float *)(q_row + uint64_t(d)*sizeof(float));
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t q_offset = n_kv - n_q;

    for (int64_t ik = 0; ik < n_kv; ++ik) {
        float mask_bias = 0.0f;
        if (mask) {
            if (lane == 0) {
                const char * mask_value = mask + uint64_t(ik)*sizeof(half) +
                    uint64_t(iq)*m_nb1 + uint64_t(PLAIN_F16 ? 0 : ih % mask_ne2)*m_nb2 +
                    uint64_t(PLAIN_F16 ? 0 : ib % mask_ne3)*m_nb3;
                mask_bias = __half2float(*(const half *) mask_value);
            }
            mask_bias = __shfl_sync(0xffffffff, mask_bias, 0, WARP_SIZE);
            // An excluded key must not consume K/V bandwidth or contaminate
            // the row with nonfinite data hidden behind its -INFINITY mask.
            if (mask_bias == -INFINITY) {
                continue;
            }
        }
        const char * k_row = k + uint64_t(ik)*k_nb1 + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float kk = fattn_banded_load(
                k_row + uint64_t(d)*(PLAIN_F16 ? sizeof(half) : k_nb0),
                PLAIN_F16 ? GGML_TYPE_F16 : type_k);
            if constexpr (PLAIN_F16) {
                // Keep the qualified gfx1030 product and seven ordered FMAs.
                dot = j == 0 ? __fmul_rn(q_reg[j], kk) : __fmaf_rn(q_reg[j], kk, dot);
            } else {
                dot += q_reg[j] * kk;
            }
        }
        dot = warp_reduce_sum(dot);

        float score = PLAIN_F16 ? __fmul_rn(dot, scale) : dot * scale;
        if (lane == 0) {
            const int64_t rel_dist = iq + q_offset - ik;
            if (!PLAIN_F16 && rel_dist >= 0 && rel_dist < rel_extent) {
                const char * rel_value = rel +
                    uint64_t(rel_dist)*r_nb0 + uint64_t(ih)*r_nb1 +
                    uint64_t(iq)*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
                score += fattn_banded_load(rel_value, type_rel);
            }
            if (mask) {
                score = PLAIN_F16 ? __fadd_rn(score, mask_bias) : score + mask_bias;
            }
        }
        score = __shfl_sync(0xffffffff, score, 0, WARP_SIZE);

        if (score == -INFINITY) {
            continue;
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (score > row_max) {
            old_scale = expf(row_max - score);
            row_max = score;
        } else {
            value_scale = expf(score - row_max);
        }

        const char * v_row = v + uint64_t(ik)*v_nb1 + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float vv = fattn_banded_load(
                v_row + uint64_t(d)*(PLAIN_F16 ? sizeof(half) : v_nb0),
                PLAIN_F16 ? GGML_TYPE_F16 : type_v);
            // The original machine code rounds out*old_scale before this FMA.
            out[j] = PLAIN_F16 ? __fmaf_rn(vv, value_scale, __fmul_rn(out[j], old_scale)) :
                                 out[j]*old_scale + vv*value_scale;
        }
        row_sum = PLAIN_F16 ? __fmaf_rn(row_sum, old_scale, value_scale) :
                              row_sum*old_scale + value_scale;
    }

    const float inv_sum = row_sum > 0.0f ? 1.0f/row_sum : 0.0f;
    float * dst_row = dst + ((ib*n_q + iq)*n_head_q + ih)*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        dst_row[d] = out[j]*inv_sum;
    }
}

static bool fattn_banded_type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

bool ggml_cuda_flash_attn_ext_banded_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_BANDED) {
        return false;
    }

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[5];
    if (!q || !k || !v || !rel || q->type != GGML_TYPE_F32) {
        return false;
    }
    if (!fattn_banded_type_supported(k->type) ||
        !fattn_banded_type_supported(v->type) ||
        !fattn_banded_type_supported(rel->type)) {
        return false;
    }
    if ((q->ne[0] != 64 && q->ne[0] != 128) || v->ne[0] != q->ne[0] || k->ne[0] != q->ne[0]) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || k->ne[2] != v->ne[2]) {
        return false;
    }
    if (q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) || rel->nb[0] != ggml_type_size(rel->type)) {
        return false;
    }
    if (rel->ne[0] <= 0 || rel->ne[1] != q->ne[2] || rel->ne[2] != q->ne[1] ||
        (rel->ne[3] != 1 && rel->ne[3] != q->ne[3])) {
        return false;
    }
    return !m || (m->type == GGML_TYPE_F16 && ggml_is_contiguous(m) &&
        q->ne[2] % m->ne[2] == 0 && q->ne[3] % m->ne[3] == 0);
#endif
}

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_banded_supported(ctx.device, dst));

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[5];

    // route F16/BF16 K/V to the MMA kernel; keep this FP32 kernel for mixed types and strided rel
    if (k->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F32 &&
        rel->type == GGML_TYPE_F32 && ggml_is_contiguous(rel) &&
        // MMA ABI indexes rel by Q's batch: a singleton rel batch must take the stride-aware fallback
        rel->ne[3] == q->ne[3] && rel->ne[0] <= (1 << 20)) {
        ggml_cuda_flash_attn_ext(ctx, dst);
        return;
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    // the tensor extent (not op_params) is authoritative after graph cloning
    const int64_t rel_extent = rel->ne[0];

    constexpr int warps_per_block = 4;
    const dim3 blocks((q->ne[1] + warps_per_block - 1) / warps_per_block, q->ne[2], q->ne[3]);
    const dim3 threads(warps_per_block * WARP_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

#define LAUNCH_BANDED(D) \
    flash_attn_ext_banded_f32<D, warps_per_block><<<blocks, threads, 0, stream>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        m ? (const char *) m->data : nullptr, (const char *) rel->data, (float *) dst->data, \
        scale, k->type, v->type, rel->type, q->ne[1], k->ne[1], q->ne[2], k->ne[2], q->ne[3], \
        rel_extent, m ? m->ne[2] : 1, m ? m->ne[3] : 1, \
        q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3], \
        v->nb[0], v->nb[1], v->nb[2], v->nb[3], \
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3])

    if (q->ne[0] == 64) {
        LAUNCH_BANDED(64);
    } else {
        LAUNCH_BANDED(128);
    }
#undef LAUNCH_BANDED
}

#if defined(GGML_USE_HIP)
static constexpr int fattn_fused_head_size = 256;
static constexpr int fattn_fused_heads = 24;
static constexpr int fattn_fused_gqa = 12;
static constexpr int fattn_fused_wave = 32;
static constexpr int fattn_fused_waves = 4;
static constexpr int fattn_fused_threads = fattn_fused_wave * fattn_fused_waves;
static constexpr int fattn_fused_tile_keys = 32;

static __global__ void flash_attn_ext_f32_fused_tile(
        const char * __restrict__ query, const char * __restrict__ keys,
        const char * __restrict__ values, const half * __restrict__ mask,
        float * __restrict__ output, const float scale, const int key_count,
        const uint64_t query_head_stride, const uint64_t key_row_stride,
        const uint64_t key_head_stride, const uint64_t value_row_stride,
        const uint64_t value_head_stride) {
    static_assert(fattn_fused_head_size == 2 * fattn_fused_threads,
                  "each thread owns exactly two output dimensions");
    const int thread = threadIdx.x;
    const int lane = thread % fattn_fused_wave;
    const int wave = thread / fattn_fused_wave;
    const int head = blockIdx.x;
    const int kv_head = head / fattn_fused_gqa;
    const char * query_row = query + uint64_t(head) * query_head_stride;
    const char * key_head = keys + uint64_t(kv_head) * key_head_stride;
    const char * value_head = values + uint64_t(kv_head) * value_head_stride;

    // Every wave keeps the original eight Q values per lane. The four waves
    // produce different keys, without changing any key's dot-product tree.
    float query_values[fattn_fused_head_size / fattn_fused_wave];
#pragma unroll
    for (int j = 0; j < fattn_fused_head_size / fattn_fused_wave; ++j) {
        const int d = lane + j * fattn_fused_wave;
        query_values[j] = *(const float *) (query_row + uint64_t(d) * sizeof(float));
    }
    float output_first = 0.0f;
    float output_second = 0.0f;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    __shared__ float scores_and_old_scales[fattn_fused_tile_keys];
    __shared__ float value_scales[fattn_fused_tile_keys];
    __shared__ float inverse_sum;

    for (int tile_first = 0; tile_first < key_count; tile_first += fattn_fused_tile_keys) {
        const int remaining = key_count - tile_first;
        const int tile_count = remaining < fattn_fused_tile_keys ? remaining : fattn_fused_tile_keys;

        // Phase 1: four independent score producers, with the original
        // lane products, seven FMAs, and complete XOR16/8/4/2/1 tree.
#pragma unroll 1
        for (int offset = wave; offset < tile_count; offset += fattn_fused_waves) {
            const int key = tile_first + offset;
            float mask_bias = 0.0f;
            if (mask) {
                if (lane == 0) {
                    mask_bias = __half2float(mask[key]);
                }
                mask_bias = __shfl_sync(0xffffffff, mask_bias, 0, fattn_fused_wave);
                if (mask_bias == -INFINITY) {
                    if (lane == 0) {
                        scores_and_old_scales[offset] = -INFINITY;
                    }
                    continue;
                }
            }
            const char * key_row = key_head + uint64_t(key) * key_row_stride;
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < fattn_fused_head_size / fattn_fused_wave; ++j) {
                const int d = lane + j * fattn_fused_wave;
                const float value = __half2float(*(const half *) (key_row + uint64_t(d) * sizeof(half)));
                dot = j == 0 ? __fmul_rn(query_values[j], value) : __fmaf_rn(query_values[j], value, dot);
            }
#pragma unroll
            for (int offset_xor = 16; offset_xor > 0; offset_xor >>= 1) {
                dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffff, dot, offset_xor, fattn_fused_wave));
            }
            if (lane == 0) {
                // The current gfx1030 ISA contracts the masked score to FMA.
                scores_and_old_scales[offset] = mask ? __fmaf_rn(dot, scale, mask_bias) : __fmul_rn(dot, scale);
            }
        }
        __syncthreads();

        // Phase 2: wave zero computes the same per-key maxima and scales.
        // row_max never becomes NaN: the original update is a strict >.
        // Seed every prefix with that carry, then keep the earlier input on
        // ties. This retains NaN handling and the first maximum's zero sign.
        if (wave == 0) {
            const float carry = __shfl_sync(0xffffffff, row_max, 0, fattn_fused_wave);
            const float score = lane < tile_count ? scores_and_old_scales[lane] : -INFINITY;
            float prefix = score > carry ? score : carry;
#pragma unroll
            for (int distance = 1; distance < fattn_fused_wave; distance <<= 1) {
                const float left = __shfl_up_sync(0xffffffff, prefix, distance, fattn_fused_wave);
                if (lane >= distance) {
                    prefix = prefix > left ? prefix : left;
                }
            }
            // Every lane, including the tail, participates in every shuffle.
            const float preceding = __shfl_up_sync(0xffffffff, prefix, 1, fattn_fused_wave);
            const float previous = lane == 0 ? carry : preceding;
            const float tile_max = __shfl_sync(0xffffffff, prefix, tile_count - 1, fattn_fused_wave);
            if (lane == 0) {
                row_max = tile_max;
            }
            if (lane < tile_count) {
                if (score == -INFINITY) {
                    scores_and_old_scales[lane] = -1.0f;
                    value_scales[lane] = 0.0f;
                } else {
                    float old_scale = 1.0f;
                    float value_scale = 1.0f;
                    if (score > previous) {
                        old_scale = expf(previous - score);
                    } else {
                        value_scale = expf(score - previous);
                    }
                    scores_and_old_scales[lane] = old_scale;
                    value_scales[lane] = value_scale;
                }
            }
        }
        __syncthreads();

        // Keep the row sum's original FMA order while other waves consume V.
        // Both paths only read the scales; the last tile barrier publishes
        // inverse_sum and protects those scales from the next tile's writers.
        if (thread == 0) {
#pragma unroll 1
            for (int offset = 0; offset < tile_count; ++offset) {
                const float old_scale = scores_and_old_scales[offset];
                if (old_scale == -1.0f) {
                    continue;
                }
                row_sum = __fmaf_rn(row_sum, old_scale, value_scales[offset]);
            }
            if (tile_first + tile_count == key_count) {
                inverse_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
            }
        }

        // Phase 3: all 128 threads consume coefficients in the original key
        // order. A valid zero coefficient must still load V and execute FMA.
#pragma unroll 1
        for (int offset = 0; offset < tile_count; ++offset) {
            const float old_scale = scores_and_old_scales[offset];
            if (old_scale == -1.0f) {
                continue;
            }
            const float value_scale = value_scales[offset];
            const char * value_row = value_head + uint64_t(tile_first + offset) * value_row_stride;
            const float first = __half2float(*(const half *) (value_row + uint64_t(thread) * sizeof(half)));
            const float second = __half2float(*(const half *) (value_row + uint64_t(thread + fattn_fused_threads) * sizeof(half)));
            output_first = __fmaf_rn(first, value_scale, __fmul_rn(output_first, old_scale));
            output_second = __fmaf_rn(second, value_scale, __fmul_rn(output_second, old_scale));
        }
        // Every consumer must finish before any producer overwrites the tile.
        __syncthreads();
    }
    output[head * fattn_fused_head_size + thread] = output_first * inverse_sum;
    output[head * fattn_fused_head_size + thread + fattn_fused_threads] = output_second * inverse_sum;
}

static bool ggml_cuda_flash_attn_ext_f32_fused_tile(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float scale) {
    const ggml_tensor * query = dst->src[0];
    const ggml_tensor * keys = dst->src[1];
    const ggml_tensor * values = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const auto & device = ggml_cuda_info().devices[ctx.device];
    // The existing supported predicate owns all other type/layout/bias rules.
    if (device.cc != GGML_CUDA_CC_RDNA2 || device.warp_size != fattn_fused_wave ||
        query->ne[0] != fattn_fused_head_size ||
        query->ne[1] != 1 || query->ne[2] != fattn_fused_heads || query->ne[3] != 1 ||
        keys->type != GGML_TYPE_F16 || values->type != GGML_TYPE_F16 ||
        keys->ne[2] != 2 || values->ne[2] != 2 ||
        keys->ne[3] != 1 || values->ne[3] != 1 ||
        keys->ne[1] < 1 || keys->ne[1] > 262144 ||
        (mask && (mask->ne[2] != 1 || mask->ne[3] != 1))) {
        return false;
    }
    flash_attn_ext_f32_fused_tile<<<fattn_fused_heads, fattn_fused_threads, 0, ctx.stream()>>>(
        (const char *) query->data, (const char *) keys->data, (const char *) values->data,
        mask ? (const half *) mask->data : nullptr, (float *) dst->data, scale, int(keys->ne[1]),
        query->nb[2], keys->nb[1], keys->nb[2], values->nb[1], values->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif

bool ggml_cuda_flash_attn_ext_f32_supported(const ggml_tensor * dst) {
#if !defined(FLASH_ATTN_AVAILABLE) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_ATTN_EXT || dst->type != GGML_TYPE_F32 || dst->src[4] != nullptr) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    if (!q || !k || !v || q->type != GGML_TYPE_F32) {
        return false;
    }
    // The existing head128 profile retains F32 K/V. Swift's larger mirrored
    // cache stores matching F16 or F32 operands; typed loads promote to F32
    // without rounding Q or changing the fused online-softmax arithmetic.
    const bool profile = (q->ne[0] == 128 && q->ne[2] == 4*k->ne[2] &&
                          k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32) ||
                         (q->ne[0] == 256 && q->ne[2] == 12*k->ne[2] &&
                          (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) &&
                          v->type == k->type);
    if (!profile || k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] ||
            q->ne[1] < 1 || q->ne[1] > 4 || k->ne[1] < 1 || k->ne[1] != v->ne[1] ||
            k->ne[2] < 1 || k->ne[2] != v->ne[2] ||
            q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
            q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
            v->nb[0] != ggml_type_size(v->type)) {
        return false;
    }
    float max_bias;
    float logit_softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(logit_softcap));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    return !m || (m->type == GGML_TYPE_F16 && ggml_is_contiguous(m) &&
        m->ne[0] >= k->ne[1] && m->ne[1] >= q->ne[1] && m->ne[2] == 1 && m->ne[3] == 1);
#endif
}

void ggml_cuda_flash_attn_ext_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_f32_supported(dst));
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

#if defined(GGML_USE_HIP)
    if (ggml_cuda_flash_attn_ext_f32_fused_tile(ctx, dst, scale)) {
        return;
    }
#endif

    // Reuse the banded kernel's fused F32 online softmax with its relative-logit
    // branch disabled by an empty extent. F16 K/V decode in the typed loads;
    // Q stays F32. No QK matrix, conversion buffer, or global scratch.
    constexpr int warps_per_block = 4;
    const dim3 threads(warps_per_block * WARP_SIZE, 1, 1);
    const auto launch = [&](auto dimension, auto plain_f16) {
        const dim3 blocks((q->ne[1] + warps_per_block - 1) / warps_per_block, q->ne[2], q->ne[3]);
        flash_attn_ext_banded_f32<decltype(dimension)::value, warps_per_block, decltype(plain_f16)::value>
            <<<blocks, threads, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            m ? (const char *) m->data : nullptr, nullptr, (float *) dst->data,
            scale, k->type, v->type, GGML_TYPE_F32,
            q->ne[1], k->ne[1], q->ne[2], k->ne[2], q->ne[3], 0, 1, 1,
            q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3],
            v->nb[0], v->nb[1], v->nb[2], v->nb[3],
            m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0,
            0, 0, 0, 0, 1);
    };
    if (q->ne[0] == 256) {
        // This public profile has F32 Q, no relative logits, and a scalar mask
        // head/batch. Compile its known F16 loads so all lane values can load
        // together, without a dynamic type branch and wait for every element.
        // Explicit contraction order is qualified against the original gfx1030
        // kernel. Other devices retain their existing floating-point behavior.
        if (k->type == GGML_TYPE_F16 && ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_RDNA2) {
            launch(std::integral_constant<int, 256>{}, std::true_type{});
        } else {
            launch(std::integral_constant<int, 256>{}, std::false_type{});
        }
    } else {
        launch(std::integral_constant<int, 128>{}, std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());
}
