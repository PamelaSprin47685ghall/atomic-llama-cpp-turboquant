#include "common.cuh"
#include "fattn-rerot.cuh"
#include "turbo-quant.cuh"

#include <cstdint>
#include <cmath>
#include <cstring>

static __device__ __forceinline__ float fattn_rerot_load_element(
        const char * __restrict__ ptr, int type, int d) {
    switch (type) {
        case GGML_TYPE_F32:
            return ((const float *) ptr)[d];
        case GGML_TYPE_F16:
            return __half2float(((const half *) ptr)[d]);
        case GGML_TYPE_BF16: {
            const uint32_t bits = uint32_t(((const uint16_t *) ptr)[d]) << 16;
            return __uint_as_float(bits);
        }
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * blk = (const block_q8_0 *) ptr + (d / QK8_0);
            const int off = d % QK8_0;
            return __half2float(blk->d) * float(blk->qs[off]);
        }
        case GGML_TYPE_Q4_0: {
            const block_q4_0 * blk = (const block_q4_0 *) ptr + (d / QK4_0);
            const int off = d % QK4_0;
            // Canonical Q4_0 (ggml-quants.c dequantize_row_q4_0): qs[j] low nibble -> j,
            // high nibble -> j+16. NOT interleaved.
            const uint8_t byte = blk->qs[off & 15];
            const int val = (off < 16) ? (byte & 0x0F) : (byte >> 4);
            return __half2float(blk->d) * float(val - 8);
        }
        case GGML_TYPE_Q4_1: {
            // Canonical Q4_1 (dequantize_row_q4_1): same j/j+16 split, y = nibble*d + m.
            const block_q4_1 * blk = (const block_q4_1 *) ptr + (d / QK4_1);
            const int off = d % QK4_1;
            const float scale = __low2float(blk->dm);
            const float min_val = __high2float(blk->dm);
            const uint8_t byte = blk->qs[off & 15];
            const int nibble = (off < 16) ? (byte & 0x0F) : (byte >> 4);
            return scale * float(nibble) + min_val;
        }
        case GGML_TYPE_Q5_0: {
            // Canonical Q5_0 (dequantize_row_q5_0): j/j+16 split, 5th bit = bit off of qh.
            const block_q5_0 * blk = (const block_q5_0 *) ptr + (d / QK5_0);
            const int off = d % QK5_0;
            const float scale = __half2float(blk->d);
            uint32_t qh;
            memcpy(&qh, blk->qh, sizeof(qh));
            const uint8_t byte = blk->qs[off & 15];
            const int lo = (off < 16) ? (byte & 0x0F) : (byte >> 4);
            const int hi = (qh >> off) & 1;
            return scale * float((lo | (hi << 4)) - 16);
        }
        case GGML_TYPE_Q5_1: {
            // Canonical Q5_1 (dequantize_row_q5_1): j/j+16 split, y = quant*d + m.
            const block_q5_1 * blk = (const block_q5_1 *) ptr + (d / QK5_1);
            const int off = d % QK5_1;
            const float scale = __low2float(blk->dm);
            const float min_val = __high2float(blk->dm);
            uint32_t qh;
            memcpy(&qh, blk->qh, sizeof(qh));
            const uint8_t byte = blk->qs[off & 15];
            const int lo = (off < 16) ? (byte & 0x0F) : (byte >> 4);
            const int hi = (qh >> off) & 1;
            return scale * float(lo | (hi << 4)) + min_val;
        }
        case GGML_TYPE_IQ4_NL: {
            // Canonical IQ4_NL (dequantize_row_iq4_nl): low nibbles -> [0,16), high -> [16,32).
            const block_iq4_nl * blk = (const block_iq4_nl *) ptr + (d / QK4_NL);
            const int off = d % QK4_NL;
            const float scale = __half2float(blk->d);
            const uint8_t byte = blk->qs[off & 15];
            const int nibble = (off < 16) ? (byte & 0x0F) : (byte >> 4);
            return scale * float(kvalues_iq4nl[nibble]);
        }
        case GGML_TYPE_TURBO2_0: {
            const block_turbo2_0 * blk = (const block_turbo2_0 *) ptr + (d / QK_TURBO2);
            const int off = d % QK_TURBO2;
            const float norm = __half2float(blk->norm);
            return turbo2_dequant_element(blk, off, norm);
        }
        case GGML_TYPE_TURBO3_0: {
            const block_turbo3_0 * blk = (const block_turbo3_0 *) ptr + (d / QK_TURBO3);
            const int off = d % QK_TURBO3;
            const float norm = __half2float(blk->norm);
            return turbo3_dequant_element(blk, off, norm);
        }
        case GGML_TYPE_TURBO4_0: {
            const block_turbo4_0 * blk = (const block_turbo4_0 *) ptr + (d / QK_TURBO4);
            const int off = d % QK_TURBO4;
            const float norm = __half2float(blk->norm);
            return turbo4_dequant_element(blk, off, norm);
        }
        default:
            return 0.0f;
    }
}

static constexpr int REROT_MAX_VALS_PER_LANE = 16; // 16 * 32 = 512 max DV

template<int WARPS_PER_BLOCK>
static __global__ void flash_attn_ext_rerot_kernel(
        const char * __restrict__ q_groups,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const int32_t * __restrict__ entries,
        const int32_t * __restrict__ offsets,
        const float * __restrict__ sinks,
        float * __restrict__ dst,
        float scale,
        float logit_softcap,
        int type_k,
        int type_v,
        int64_t DK,
        int64_t DV,
        int64_t n_queries,
        int64_t n_groups,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_k,
        int64_t n_head_v,
        int64_t n_entries,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t dst_nb1,
        uint64_t dst_nb2) {
    __shared__ float s_q[WARPS_PER_BLOCK][512];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t iq = int64_t(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int64_t ih = blockIdx.y;

    if (iq >= n_queries || ih >= n_head_q) {
        return;
    }

    const int64_t ih_k = ih / (n_head_q / n_head_k);
    const int64_t ih_v = ih / (n_head_q / n_head_v);

    const int32_t begin = offsets[iq];
    const int32_t end   = offsets[iq + 1];

    const int n_vals = (DV + WARP_SIZE - 1) / WARP_SIZE;
    float out[REROT_MAX_VALS_PER_LANE];
#pragma unroll
    for (int j = 0; j < REROT_MAX_VALS_PER_LANE; ++j) {
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    int32_t loaded_group = -1;

    // CPU parity (ggml_compute_forward_flash_attn_ext_rerot asserts the range
    // and every entry): a malformed [begin,end) or any OOB entry poisons the
    // whole (iq,ih) row with quiet NaN. Empty-but-valid ranges (begin == end)
    // keep the sink-only/zero behavior below. Device-side only, no host sync.
    bool row_poisoned = !(begin >= 0 && end >= begin && end <= n_entries);

    if (!row_poisoned) {
        for (int32_t ie = begin; ie < end; ++ie) {
            const int32_t key_index   = entries[2 * ie + 0];
            const int32_t group_index = entries[2 * ie + 1];

            if (key_index < 0 || key_index >= n_kv || group_index < 0 || group_index >= n_groups) {
                row_poisoned = true;
                break;
            }

            // Run-length reuse: reload Q vector into shared memory only when group_index changes
            if (loaded_group != group_index) {
                const char * q_ptr = q_groups + uint64_t(group_index) * q_nb1 + uint64_t(ih) * q_nb2;
                for (int d = lane; d < DK; d += WARP_SIZE) {
                    s_q[warp][d] = *(const float *)(q_ptr + uint64_t(d) * sizeof(float));
                }
                __syncwarp();
                loaded_group = group_index;
            }

            const char * k_ptr = k + uint64_t(key_index) * k_nb1 + uint64_t(ih_k) * k_nb2;
            float dot = 0.0f;
            for (int d = lane; d < DK; d += WARP_SIZE) {
                const float k_val = fattn_rerot_load_element(k_ptr, type_k, d);
                dot += s_q[warp][d] * k_val;
            }
            dot = warp_reduce_sum(dot);

            float score = dot * scale;
            if (logit_softcap != 0.0f) {
                score = logit_softcap * tanhf(score);
            }

            float old_scale = 1.0f;
            float value_scale = 1.0f;
            if (score > row_max) {
                old_scale = expf(row_max - score);
                row_max = score;
            } else {
                value_scale = expf(score - row_max);
            }

            const char * v_ptr = v + uint64_t(key_index) * v_nb1 + uint64_t(ih_v) * v_nb2;
            for (int j = 0; j < n_vals; ++j) {
                const int d = lane + j * WARP_SIZE;
                const float v_val = (d < DV) ? fattn_rerot_load_element(v_ptr, type_v, d) : 0.0f;
                out[j] = out[j] * old_scale + v_val * value_scale;
            }
            row_sum = row_sum * old_scale + value_scale;
        }
    }

    if (row_poisoned) {
        // Exact quiet-NaN row (0x7fc00000): write directly and return so the
        // sinks path and final rescale cannot alter the bit pattern.
        const float qnan = __uint_as_float(0x7fc00000u);
        char * dst_ptr = (char *) dst + uint64_t(ih) * dst_nb1 + uint64_t(iq) * dst_nb2;
        for (int j = 0; j < n_vals; ++j) {
            const int d = lane + j * WARP_SIZE;
            if (d < DV) {
                ((float *) dst_ptr)[d] = qnan;
            }
        }
        return;
    }

    if (sinks) {
        const float sink_score = sinks[ih];
        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (sink_score > row_max) {
            old_scale = expf(row_max - sink_score);
            row_max = sink_score;
        } else {
            value_scale = expf(sink_score - row_max);
        }
        for (int j = 0; j < n_vals; ++j) {
            out[j] = out[j] * old_scale;
        }
        row_sum = row_sum * old_scale + value_scale;
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;
    char * dst_ptr = (char *) dst + uint64_t(ih) * dst_nb1 + uint64_t(iq) * dst_nb2;
    for (int j = 0; j < n_vals; ++j) {
        const int d = lane + j * WARP_SIZE;
        if (d < DV) {
            ((float *) dst_ptr)[d] = out[j] * inv_sum;
        }
    }
}

static bool fattn_rerot_type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

bool ggml_cuda_flash_attn_ext_rerot_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_REROT) {
        return false;
    }

    const ggml_tensor * q_groups = dst->src[0];
    const ggml_tensor * k        = dst->src[1];
    const ggml_tensor * v        = dst->src[2];
    const ggml_tensor * entries  = dst->src[3];
    const ggml_tensor * offsets  = dst->src[4];
    const ggml_tensor * sinks    = dst->src[5];

    if (!q_groups || !k || !v || !entries || !offsets) {
        return false;
    }
    if (q_groups->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!fattn_rerot_type_supported(k->type) || !fattn_rerot_type_supported(v->type)) {
        return false;
    }
    if (entries->type != GGML_TYPE_I32 || offsets->type != GGML_TYPE_I32) {
        return false;
    }
    if (entries->ne[0] != 2 || !ggml_is_vector(offsets) || offsets->ne[0] < 2) {
        return false;
    }
    // Fail closed on non-packed dim-0 strides for the F32/I32 tensors the
    // kernel indexes by element (mirrors the Vulkan capability gate).
    // q_groups stays F32 nb[0]==4 through rope/pad/cont/WHT/permute, and
    // entries/offsets are fresh contiguous I32 span tensors, so this gate
    // cannot reject shapes current producers emit.
    if (q_groups->nb[0] != sizeof(float) ||
        entries->nb[0] != sizeof(int32_t) ||
        offsets->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (q_groups->ne[0] != k->ne[0]) {
        return false;
    }
    const int64_t DK = k->ne[0];
    const int64_t DV = v->ne[0];
    if (DK <= 0 || DK > 512 || DV <= 0 || DV > 512) {
        return false;
    }
    // The kernel indexes K/V as [D, n_kv, n_head] (ne[3] must be 1) and the
    // per-element dequant helpers divide D by the type block size, so D must
    // be a whole number of blocks. Both hold for every in-tree producer.
    // NOTE: no nb[0] stride gate: ggml_new_tensor_impl sets nb[0] to the
    // block byte size for quantized types (not an element stride), so K/V
    // keep no nb[0] gate. The F32/I32 tensors (q_groups/entries/offsets)
    // are gated above instead.
    if (k->ne[3] != 1 || v->ne[3] != 1) {
        return false;
    }
    if (DK % ggml_blck_size(k->type) != 0 || DV % ggml_blck_size(v->type) != 0) {
        return false;
    }
    if (q_groups->ne[2] % k->ne[2] != 0 || q_groups->ne[2] % v->ne[2] != 0) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] != q_groups->ne[2])) {
        return false;
    }
    return true;
#endif
}

void ggml_cuda_flash_attn_ext_rerot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_rerot_supported(ctx.device, dst));

    const ggml_tensor * q_groups = dst->src[0];
    const ggml_tensor * k        = dst->src[1];
    const ggml_tensor * v        = dst->src[2];
    const ggml_tensor * entries  = dst->src[3];
    const ggml_tensor * offsets  = dst->src[4];
    const ggml_tensor * sinks    = dst->src[5];

    float scale = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const int64_t DK        = k->ne[0];
    const int64_t DV        = v->ne[0];
    const int64_t n_queries = offsets->ne[0] - 1;
    const int64_t n_groups  = q_groups->ne[1];
    const int64_t n_kv      = k->ne[1];
    const int64_t n_head_q  = q_groups->ne[2];
    const int64_t n_head_k  = k->ne[2];
    const int64_t n_head_v  = v->ne[2];
    const int64_t n_entries = entries->ne[1];

    constexpr int warps_per_block = 4;
    const dim3 blocks((n_queries + warps_per_block - 1) / warps_per_block, n_head_q, 1);
    const dim3 threads(warps_per_block * WARP_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

    flash_attn_ext_rerot_kernel<warps_per_block><<<blocks, threads, 0, stream>>>(
        (const char *) q_groups->data,
        (const char *) k->data,
        (const char *) v->data,
        (const int32_t *) entries->data,
        (const int32_t *) offsets->data,
        sinks ? (const float *) sinks->data : nullptr,
        (float *) dst->data,
        scale, logit_softcap,
        k->type, v->type,
        DK, DV,
        n_queries, n_groups, n_kv,
        n_head_q, n_head_k, n_head_v, n_entries,
        q_groups->nb[1], q_groups->nb[2],
        k->nb[1], k->nb[2],
        v->nb[1], v->nb[2],
        dst->nb[1], dst->nb[2]);
}
