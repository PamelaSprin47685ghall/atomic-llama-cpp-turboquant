#include "xkv-factorize.cuh"
#include "turbo-quant.cuh"
#include "ggml-xkv.h"
#include "ggml-xkv-factor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

struct cuda_xkv_canon_push {
    uint32_t n;
    uint32_t nvec;
    uint32_t hd;
    uint32_t phd;
    uint32_t rotary_dim;
    uint32_t rope_mode;
    uint32_t input_type;
    uint32_t is_k;
    uint32_t hadamard_dim;
    uint32_t hot_row_bytes;
    uint32_t hot_nphys;
    uint32_t pos_is_64;
    uint32_t status_idx;
};

static __device__ __forceinline__ float canon_rd_half(const uint8_t * hot, uint32_t off) {
    uint16_t h = (uint16_t)hot[off] | ((uint16_t)hot[off + 1u] << 8u);
    __half val = *reinterpret_cast<const __half *>(&h);
    return __half2float(val);
}

static __device__ __forceinline__ float canon_rd_f32(const uint8_t * hot, uint32_t off) {
    uint32_t w = (uint32_t)hot[off] |
                 ((uint32_t)hot[off + 1u] << 8u) |
                 ((uint32_t)hot[off + 2u] << 16u) |
                 ((uint32_t)hot[off + 3u] << 24u);
    return *reinterpret_cast<const float *>(&w);
}

static __global__ void k_xkv_canonicalize_validate(
        int32_t * __restrict__ st,
        cuda_xkv_canon_push p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bool ok = (p.n > 0u && p.nvec > 0u && p.hd > 0u && p.phd > 0u &&
                   p.phd >= p.hd && p.hot_row_bytes > 0u &&
                   (p.rotary_dim <= p.hd) && ((p.rotary_dim & 1u) == 0u) &&
                   (p.rope_mode <= 1u) &&
                   (p.input_type == GGML_TYPE_F32 || p.input_type == GGML_TYPE_F16 ||
                    p.input_type == GGML_TYPE_Q8_0 || p.input_type == GGML_TYPE_TURBO2_0 ||
                    p.input_type == GGML_TYPE_TURBO3_0 || p.input_type == GGML_TYPE_TURBO4_0) &&
                   (p.hadamard_dim == 0u || (p.hd % p.hadamard_dim) == 0u));
        if (ok && (p.input_type == GGML_TYPE_TURBO2_0 ||
                   p.input_type == GGML_TYPE_TURBO3_0 ||
                   p.input_type == GGML_TYPE_TURBO4_0)) {
            ok = (p.phd % 128u) == 0u;
        }
        if (ok && p.input_type == GGML_TYPE_Q8_0) {
            ok = (p.phd % 32u) == 0u;
        }
        st[p.status_idx] = ok ? 0 : 1;
    }
}

static __global__ void k_xkv_canonicalize_work(
        const uint8_t * __restrict__ hot,
        const int32_t * __restrict__ rows,
        const void    * __restrict__ pos_raw,
        const float   * __restrict__ rope,
        const float   * __restrict__ had,
        float         * __restrict__ outx,
        int32_t       * __restrict__ st,
        cuda_xkv_canon_push p) {
    if (st[p.status_idx] != 0) {
        return;
    }

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = p.n * p.nvec;
    if (gid >= total) {
        return;
    }

    uint32_t i = gid / p.nvec;
    uint32_t v = gid % p.nvec;
    int32_t pr = rows[i];
    if (pr < 0 || (uint32_t)pr >= p.hot_nphys) {
        atomicExch(&st[p.status_idx], 2);
        return;
    }

    uint32_t t = p.input_type;
    uint32_t head_bytes = 0u;
    if (t == GGML_TYPE_F32) head_bytes = p.phd * 4u;
    else if (t == GGML_TYPE_F16) head_bytes = p.phd * 2u;
    else if (t == GGML_TYPE_Q8_0) head_bytes = (p.phd / 32u) * 34u;
    else if (t == GGML_TYPE_TURBO2_0) head_bytes = (p.phd / 128u) * 34u;
    else if (t == GGML_TYPE_TURBO3_0) head_bytes = (p.phd / 128u) * 50u;
    else head_bytes = (p.phd / 128u) * 68u;

    uint32_t base = (uint32_t)pr * p.hot_row_bytes + v * head_bytes;

    if (p.phd > 512u) {
        atomicExch(&st[p.status_idx], 2);
        return;
    }

    float buf[512];
    if (t == GGML_TYPE_F32) {
        for (uint32_t j = 0u; j < p.phd; ++j) {
            buf[j] = canon_rd_f32(hot, base + j * 4u);
        }
    } else if (t == GGML_TYPE_F16) {
        for (uint32_t j = 0u; j < p.phd; ++j) {
            buf[j] = canon_rd_half(hot, base + j * 2u);
        }
    } else if (t == GGML_TYPE_Q8_0) {
        for (uint32_t b = 0u; b < p.phd / 32u; ++b) {
            float d = canon_rd_half(hot, base + b * 34u);
            for (uint32_t j = 0u; j < 32u; ++j) {
                int8_t q = (int8_t)hot[base + b * 34u + 2u + j];
                buf[b * 32u + j] = d * (float)q;
            }
        }
    } else {
        bool isT2 = (t == GGML_TYPE_TURBO2_0);
        bool isT3 = (t == GGML_TYPE_TURBO3_0);
        const float inv_sqrt_128 = 0.08838834764831845f;
        for (uint32_t g = 0u; g < p.phd / 128u; ++g) {
            uint32_t boff = base + g * (isT2 ? 34u : (isT3 ? 50u : 68u));
            float norm = canon_rd_half(hot, boff);
            for (uint32_t j = 0u; j < 128u; ++j) {
                float cv = 0.0f;
                if (isT2) {
                    uint32_t qb = (uint32_t)hot[boff + 2u + j / 4u];
                    cv = TURBO_CENTROIDS_2BIT[(qb >> (2u * (j % 4u))) & 3u];
                } else if (isT3) {
                    uint32_t qb = (uint32_t)hot[boff + 2u + j / 4u];
                    uint32_t sb = (uint32_t)hot[boff + 34u + j / 8u];
                    uint32_t idx = ((qb >> (2u * (j % 4u))) & 3u) | (((sb >> (j % 8u)) & 1u) << 2u);
                    cv = TURBO_CENTROIDS_3BIT[idx];
                } else {
                    uint32_t qb = (uint32_t)hot[boff + 4u + j / 2u];
                    uint32_t idx = (j % 2u == 0u) ? (qb & 15u) : (qb >> 4u);
                    cv = TURBO_CENTROIDS_4BIT[idx];
                }
                buf[g * 128u + j] = norm * cv;
            }
            // Exact inverse WHT (swapped signs) -> canonical domain
            for (uint32_t j = 0u; j < 128u; ++j) {
                buf[g * 128u + j] *= TURBO_WHT_SIGNS2[j];
            }
            for (uint32_t h = 1u; h < 128u; h *= 2u) {
                for (uint32_t k = 0u; k < 128u; k += h * 2u) {
                    for (uint32_t j = k; j < k + h; ++j) {
                        float a = buf[g * 128u + j];
                        float b = buf[g * 128u + j + h];
                        buf[g * 128u + j]     = a + b;
                        buf[g * 128u + j + h] = a - b;
                    }
                }
            }
            for (uint32_t j = 0u; j < 128u; ++j) {
                buf[g * 128u + j] *= inv_sqrt_128 * TURBO_WHT_SIGNS1[j];
            }
        }
    }

    uint32_t out_base = i * (p.nvec * p.hd) + v * p.hd;

    // Inverse attention rotation per hadamard block
    uint32_t Hd = p.hadamard_dim;
    if (Hd > 0u) {
        for (uint32_t b = 0u; b < p.hd; b += Hd) {
            for (uint32_t ii = 0u; ii < Hd; ++ii) {
                float acc = 0.0f;
                for (uint32_t jj = 0u; jj < Hd; ++jj) {
                    acc += had[ii * Hd + jj] * buf[b + jj];
                }
                outx[out_base + b + ii] = acc;
            }
        }
    } else {
        for (uint32_t j = 0u; j < p.hd; ++j) {
            outx[out_base + j] = buf[j];
        }
    }

    // Exact inverse text RoPE for K
    if (p.is_k != 0u && p.rotary_dim > 0u) {
        float pos = (p.pos_is_64 != 0u)
            ? (float)(((const int64_t *)pos_raw)[i])
            : (float)(((const int32_t *)pos_raw)[i]);
        uint32_t Fc = p.rotary_dim / 2u;
        if (p.rope_mode == 0u) {
            for (uint32_t f = 0u; f < Fc; ++f) {
                float ang = rope[f] * pos;
                float c = cosf(ang), sn = sinf(ang);
                float sc = rope[Fc + f] > 0.0f ? rope[Fc + f] : 1.0f;
                float re = outx[out_base + f] / sc;
                float im = outx[out_base + f + Fc] / sc;
                outx[out_base + f]      = re * c + im * sn;
                outx[out_base + f + Fc] = im * c - re * sn;
            }
        } else {
            for (uint32_t f = 0u; f < Fc; ++f) {
                float ang = rope[f] * pos;
                float c = cosf(ang), sn = sinf(ang);
                float sc = rope[Fc + f] > 0.0f ? rope[Fc + f] : 1.0f;
                float re = outx[out_base + 2u * f] / sc;
                float im = outx[out_base + 2u * f + 1u] / sc;
                outx[out_base + 2u * f]      = re * c + im * sn;
                outx[out_base + 2u * f + 1u] = im * c - re * sn;
            }
        }
    }

    for (uint32_t j = 0u; j < p.hd; ++j) {
        float w = outx[out_base + j];
        if (isnan(w) || isinf(w)) {
            atomicExch(&st[p.status_idx], 3);
            return;
        }
    }
}

bool ggml_cuda_xkv_canonicalize_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_CANONICALIZE) {
        return false;
    }
    if (!op->src[0] || !op->src[1] || !op->src[2] || !op->src[3] ||
        !op->src[4] || !op->src[5]) {
        return false;
    }
    ggml_xkv_canonicalize_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    return ggml_xkv_canonicalize_supports(
        op->src[0], op->src[1], op->src[2], op->src[3],
        op->src[4], op->src[5], op, &p, err, sizeof(err));
}

void ggml_cuda_xkv_canonicalize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * hot         = dst->src[0];
    const ggml_tensor * rows        = dst->src[1];
    const ggml_tensor * positions   = dst->src[2];
    const ggml_tensor * rope_tables = dst->src[3];
    const ggml_tensor * hadamard    = dst->src[4];
    const ggml_tensor * status      = dst->src[5];

    ggml_xkv_canonicalize_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    cuda_xkv_canon_push pc = {};
    pc.n             = p.n_rows;
    pc.nvec          = p.n_layers * p.n_heads;
    pc.hd            = p.head_dim;
    pc.phd           = p.padded_head_dim;
    pc.rotary_dim    = p.rotary_dim;
    pc.rope_mode     = p.rope_mode;
    pc.input_type    = p.input_type;
    pc.is_k          = p.is_k;
    pc.hadamard_dim  = p.hadamard_dim;
    pc.hot_row_bytes = (uint32_t)ggml_row_size(hot->type, hot->ne[0]);
    pc.hot_nphys     = (uint32_t)hot->ne[1];
    pc.pos_is_64     = (positions->type == GGML_TYPE_I64) ? 1u : 0u;
    pc.status_idx    = 0u;

    cudaStream_t stream = ctx.stream();

    // Step 0: Validation
    k_xkv_canonicalize_validate<<<1, 1, 0, stream>>>(
        (int32_t *)status->data,
        pc
    );

    // Step 1: Work
    uint32_t total = pc.n * pc.nvec;
    constexpr int block_size = 64;
    int num_blocks = (total + block_size - 1) / block_size;
    if (num_blocks > 0) {
        k_xkv_canonicalize_work<<<num_blocks, block_size, 0, stream>>>(
            (const uint8_t *)hot->data,
            (const int32_t *)rows->data,
            positions->data,
            (const float   *)rope_tables->data,
            (const float   *)hadamard->data,
            (float         *)dst->data,
            (int32_t       *)status->data,
            pc
        );
    }
    CUDA_CHECK(cudaGetLastError());
}
