#include "xkv-attention.cuh"
#include "turbo-quant.cuh"
#include "ggml-xkv.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

struct cuda_xkv_attention_push {
    uint32_t dim_k;
    uint32_t dim_v;
    uint32_t gqa;
    uint32_t n_groups;
    uint32_t n_queries;
    uint32_t hot_rows;
    uint32_t n_cold;
    uint32_t n_entries;
    uint32_t kv_head;
    uint32_t stream;
    uint32_t flags;         // bit0 reject_dupes, bit1 entries_2wide, bit2 has_sinks,
                            // bit3 first_tile, bit4 has_carry, bit5 final_tile
    uint32_t entry_stride;  // 2 or 4
    float scale;
    float softcap;
    uint32_t k_hot_type;    // 0=F32 1=F16 8=Q8_0 42=T2 43=T3 44=T4
    uint32_t v_hot_type;
    uint32_t k_cold_type;   // 0=F32 1=F16
    uint32_t v_cold_type;
    uint32_t q_nb1;         // byte strides
    uint32_t q_nb2;
    uint32_t kh_nb1;
    uint32_t kh_nb2;
    uint32_t kh_nb3;
    uint32_t vh_nb1;
    uint32_t vh_nb2;
    uint32_t vh_nb3;
    uint32_t kc_nb1;
    uint32_t vc_nb1;
};

static __device__ __forceinline__ float u8f16(uint8_t b0, uint8_t b1) {
    uint16_t h = (uint16_t)b0 | ((uint16_t)b1 << 8u);
    __half val = *reinterpret_cast<const __half *>(&h);
    return __half2float(val);
}

static __device__ __forceinline__ float u8f32(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    uint32_t w = (uint32_t)b0 | ((uint32_t)b1 << 8u) | ((uint32_t)b2 << 16u) | ((uint32_t)b3 << 24u);
    return *reinterpret_cast<const float *>(&w);
}

static __device__ __forceinline__ void poison_row(float * dst, uint32_t q, uint32_t h, const cuda_xkv_attention_push & p) {
    uint32_t base = (q * p.gqa + h) * (p.dim_v + 2u);
    uint32_t nan_bits = 0x7fc00000u;
    const float nan_val = *reinterpret_cast<const float *>(&nan_bits);
    for (uint32_t d = 0u; d < p.dim_v + 2u; ++d) {
        dst[base + d] = nan_val;
    }
}

// Decode one hot row (canonical domain) into K[dim]. Turbo types inverse-rotate.
static __device__ void hot_row_decode(
        uint32_t which, // 0 for K, 1 for V
        const uint8_t * kh,
        const uint8_t * vh,
        uint32_t base,
        uint32_t type,
        uint32_t dim,
        float * K) {
    const uint8_t * src = (which == 0u) ? kh : vh;
    if (type == GGML_TYPE_F32) {
        for (uint32_t d = 0u; d < dim; ++d) {
            K[d] = u8f32(src[base + d * 4u], src[base + d * 4u + 1u], src[base + d * 4u + 2u], src[base + d * 4u + 3u]);
        }
        return;
    }
    if (type == GGML_TYPE_F16) {
        for (uint32_t d = 0u; d < dim; ++d) {
            K[d] = u8f16(src[base + d * 2u], src[base + d * 2u + 1u]);
        }
        return;
    }
    if (type == GGML_TYPE_Q8_0) {
        for (uint32_t d = 0u; d < dim; ++d) {
            uint32_t b = d / 32u;
            uint32_t j = d % 32u;
            uint32_t boff = base + b * 34u;
            float dd = u8f16(src[boff], src[boff + 1u]);
            int8_t qv = (int8_t)src[boff + 2u + j];
            K[d] = dd * (float)qv;
        }
        return;
    }

    // Turbo2/3/4: centroid dequant per 128-group, then exact inverse WHT
    // (D(s2) -> butterfly -> x/√128 x s1), mirroring ggml_turbo_wht_inverse_row.
    uint32_t ng = dim / 128u;
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (uint32_t g = 0u; g < ng; ++g) {
        float tmp[128];
        if (type == GGML_TYPE_TURBO2_0) {
            uint32_t boff = base + g * 34u;
            float norm = u8f16(src[boff], src[boff + 1u]);
            for (uint32_t j = 0u; j < 128u; ++j) {
                uint32_t qb = (uint32_t)src[boff + 2u + j / 4u];
                tmp[j] = norm * TURBO_CENTROIDS_2BIT[(qb >> (2u * (j % 4u))) & 3u];
            }
        } else if (type == GGML_TYPE_TURBO3_0) {
            uint32_t boff = base + g * 50u;
            float norm = u8f16(src[boff], src[boff + 1u]);
            for (uint32_t j = 0u; j < 128u; ++j) {
                uint32_t qb = (uint32_t)src[boff + 2u + j / 4u];
                uint32_t sb = (uint32_t)src[boff + 2u + 32u + j / 8u];
                uint32_t low2 = (qb >> (2u * (j % 4u))) & 3u;
                uint32_t hi1  = (sb >> (j % 8u)) & 1u;
                tmp[j] = norm * TURBO_CENTROIDS_3BIT[low2 | (hi1 << 2u)];
            }
        } else {
            uint32_t boff = base + g * 68u;
            float norm = u8f16(src[boff], src[boff + 1u]);
            for (uint32_t j = 0u; j < 128u; ++j) {
                uint32_t qb = (uint32_t)src[boff + 4u + j / 2u];
                tmp[j] = norm * TURBO_CENTROIDS_4BIT[(qb >> (4u * (j % 2u))) & 15u];
            }
        }

        for (uint32_t i = 0u; i < 128u; ++i) {
            tmp[i] *= TURBO_WHT_SIGNS2[i];
        }
        for (uint32_t hh = 1u; hh < 128u; hh *= 2u) {
            for (uint32_t i = 0u; i < 128u; i += hh * 2u) {
                for (uint32_t j = i; j < i + hh; ++j) {
                    float a = tmp[j];
                    float c = tmp[j + hh];
                    tmp[j]      = a + c;
                    tmp[j + hh] = a - c;
                }
            }
        }
        for (uint32_t i = 0u; i < 128u; ++i) {
            K[g * 128u + i] = tmp[i] * inv_sqrt_128 * TURBO_WHT_SIGNS1[i];
        }
    }
}

static __device__ __forceinline__ float cold_ld(
        uint32_t which, // 0 for K, 1 for V
        const uint8_t * kc,
        const uint8_t * vc,
        uint32_t base,
        uint32_t r,
        uint32_t type) {
    const uint8_t * src = (which == 0u) ? kc : vc;
    if (type == GGML_TYPE_F32) {
        return u8f32(src[base + r * 4u], src[base + r * 4u + 1u], src[base + r * 4u + 2u], src[base + r * 4u + 3u]);
    }
    return u8f16(src[base + r * 2u], src[base + r * 2u + 1u]);
}

static __global__ void k_xkv_attention(
        const float   * __restrict__ q_data,
        const uint8_t * __restrict__ kh,
        const uint8_t * __restrict__ vh,
        const uint8_t * __restrict__ kc,
        const uint8_t * __restrict__ vc,
        const int32_t * __restrict__ entries,
        const int32_t * __restrict__ offsets,
        const float   * __restrict__ sinks,
        int32_t       * __restrict__ status,
        float         * __restrict__ dst,
        const float   * __restrict__ carry,
        cuda_xkv_attention_push p) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = p.n_queries * p.gqa;
    if (gid >= total) {
        return;
    }
    uint32_t q = gid / p.gqa;
    uint32_t h = gid % p.gqa;

    if (p.entry_stride != 2u && p.entry_stride != 4u) {
        atomicCAS(&status[0], 0, 6);
        poison_row(dst, q, h, p);
        return;
    }
    bool wide2 = (p.entry_stride == 2u);
    bool reject_dupes = (p.flags & 1u) != 0u;
    bool has_sinks    = (p.flags & 4u) != 0u;
    bool first_tile   = (p.flags & 8u) != 0u;
    bool has_carry    = (p.flags & 16u) != 0u;
    bool final_tile   = (p.flags & 32u) != 0u;
    bool apply_sink   = has_sinks && first_tile;
    float eff = p.softcap > 0.0f ? p.scale / p.softcap : p.scale;

    // Carry orders tiles. A sticky failure from an earlier tile invalidates
    // the whole chain; do not consume or normalize its poisoned carry.
    if (!first_tile && status[0] != 0) {
        poison_row(dst, q, h, p);
        return;
    }

    int32_t ob = offsets[q];
    int32_t oe = offsets[q + 1u];
    if (ob < 0 || oe < ob || (uint32_t)oe > p.n_entries) {
        atomicCAS(&status[0], 0, 1);
        poison_row(dst, q, h, p);
        return;
    }

    float M;
    float L;
    float O[1024];
    if (!has_carry || first_tile) {
        M = -1.7014117e38f;
        L = 0.0f;
        for (uint32_t d = 0u; d < p.dim_v; ++d) {
            O[d] = 0.0f;
        }
    } else {
        uint32_t cb = (q * p.gqa + h) * (p.dim_v + 2u);
        for (uint32_t d = 0u; d < p.dim_v; ++d) {
            O[d] = carry[cb + d];
        }
        M = carry[cb + p.dim_v];
        L = carry[cb + p.dim_v + 1u];
    }

    float QV[1024];
    float KV[1024];
    float VV[1024];
    int32_t staged_group = -1;

    for (int32_t e = ob; e < oe; ++e) {
        uint32_t ue = (uint32_t)e;
        int32_t source;
        int32_t row;
        int32_t group;
        if (wide2) {
            int32_t key = entries[ue * 2u + 0u];
            group = entries[ue * 2u + 1u];
            if (key < 0) {
                atomicCAS(&status[0], 0, 3);
                poison_row(dst, q, h, p);
                return;
            }
            if ((uint32_t)key < p.hot_rows) {
                source = 1;
                row = key;
            } else {
                source = 2;
                row = key - (int32_t)p.hot_rows;
            }
        } else {
            source = entries[ue * 4u + 0u];
            row    = entries[ue * 4u + 1u];
            group  = entries[ue * 4u + 2u];
            if (entries[ue * 4u + 3u] == 0) {
                continue;
            }
        }

        if (source != 1 && source != 2) {
            atomicCAS(&status[0], 0, 2);
            poison_row(dst, q, h, p);
            return;
        }
        if (group < 0 || (uint32_t)group >= p.n_groups) {
            atomicCAS(&status[0], 0, 4);
            poison_row(dst, q, h, p);
            return;
        }
        bool is_hot = (source == 1);
        if (row < 0 || (is_hot ? (uint32_t)row >= p.hot_rows : (uint32_t)row >= p.n_cold)) {
            atomicCAS(&status[0], 0, 3);
            poison_row(dst, q, h, p);
            return;
        }

        if (reject_dupes) {
            for (int32_t d2 = ob; d2 < e; ++d2) {
                uint32_t u2 = (uint32_t)d2;
                int32_t s2;
                int32_t r2;
                if (wide2) {
                    int32_t k2 = entries[u2 * 2u + 0u];
                    if (k2 < 0) continue;
                    if ((uint32_t)k2 < p.hot_rows) {
                        s2 = 1;
                        r2 = k2;
                    } else {
                        s2 = 2;
                        r2 = k2 - (int32_t)p.hot_rows;
                    }
                } else {
                    if (entries[u2 * 4u + 3u] == 0) continue;
                    s2 = entries[u2 * 4u + 0u];
                    r2 = entries[u2 * 4u + 1u];
                }
                if (s2 == source && r2 == row) {
                    atomicCAS(&status[0], 0, 5);
                    poison_row(dst, q, h, p);
                    return;
                }
            }
        }

        if (group != staged_group) {
            uint32_t qb = (uint32_t)group * p.q_nb2 + h * p.q_nb1;
            for (uint32_t d = 0u; d < p.dim_k; ++d) {
                QV[d] = q_data[qb / 4u + d];
            }
            staged_group = group;
        }

        if (is_hot) {
            uint32_t base = p.kv_head * p.kh_nb1 + (uint32_t)row * p.kh_nb2 + p.stream * p.kh_nb3;
            hot_row_decode(0u, kh, vh, base, p.k_hot_type, p.dim_k, KV);
        } else {
            uint32_t base = (uint32_t)row * p.kc_nb1;
            for (uint32_t d = 0u; d < p.dim_k; ++d) {
                KV[d] = cold_ld(0u, kc, vc, base, d, p.k_cold_type);
            }
        }

        float dot = 0.0f;
        for (uint32_t d = 0u; d < p.dim_k; ++d) {
            dot += QV[d] * KV[d];
        }
        float s = dot * eff;
        if (p.softcap > 0.0f) {
            s = p.softcap * tanhf(s);
        }
        if (isnan(s) || isinf(s)) {
            continue; // non-finite treated as masked
        }

        if (is_hot) {
            uint32_t base = p.kv_head * p.vh_nb1 + (uint32_t)row * p.vh_nb2 + p.stream * p.vh_nb3;
            hot_row_decode(1u, kh, vh, base, p.v_hot_type, p.dim_v, VV);
        } else {
            uint32_t base = (uint32_t)row * p.vc_nb1;
            for (uint32_t d = 0u; d < p.dim_v; ++d) {
                VV[d] = cold_ld(1u, kc, vc, base, d, p.v_cold_type);
            }
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (M <= -1.7014117e38f / 2.0f) {
            M = s;
            L = 1.0f;
            for (uint32_t d = 0u; d < p.dim_v; ++d) {
                O[d] = VV[d];
            }
        } else {
            if (s > M) {
                old_scale = expf(M - s);
                M = s;
            } else {
                value_scale = expf(s - M);
            }
            for (uint32_t d = 0u; d < p.dim_v; ++d) {
                O[d] = O[d] * old_scale + value_scale * VV[d];
            }
            L = L * old_scale + value_scale;
        }
    }

    if (apply_sink) {
        float ss = sinks[h];
        if (ss == ss && ss > -1e30f) {
            if (M <= -1.7014117e38f / 2.0f) {
                M = ss;
                L = 1.0f;
            } else if (ss > M) {
                float f = expf(M - ss);
                M = ss;
                for (uint32_t d = 0u; d < p.dim_v; ++d) {
                    O[d] *= f;
                }
                L = L * f + 1.0f;
            } else {
                L += expf(ss - M);
            }
        }
    }

    uint32_t obase = (q * p.gqa + h) * (p.dim_v + 2u);
    if (final_tile && L > 0.0f && L == L) {
        float inv = 1.0f / L;
        for (uint32_t d = 0u; d < p.dim_v; ++d) {
            dst[obase + d] = O[d] * inv;
        }
    } else if (final_tile) {
        for (uint32_t d = 0u; d < p.dim_v; ++d) {
            dst[obase + d] = 0.0f;
        }
    } else {
        for (uint32_t d = 0u; d < p.dim_v; ++d) {
            dst[obase + d] = O[d];
        }
    }
    dst[obase + p.dim_v] = M;
    dst[obase + p.dim_v + 1u] = L;
}

bool ggml_cuda_xkv_attention_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_ATTENTION) {
        return false;
    }
    // q, k_hot, v_hot, k_cold, v_cold, entries, offsets, sinks?, status, carry?.
    if (!op->src[0] || !op->src[1] || !op->src[2] || !op->src[3] ||
        !op->src[4] || !op->src[5] || !op->src[6] || !op->src[8]) {
        return false;
    }
    ggml_xkv_attention_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    if (!ggml_xkv_attention_supports(op->src[0], op->src[1], op->src[2],
        op->src[3], op->src[4], op->src[5], op->src[6], op->src[7], op->src[8], op->src[9], op, &p,
        err, sizeof(err))) {
        return false;
    }
    // Index/output tensors must be densely packed for flat kernel indexing.
    if (!ggml_is_contiguous(op->src[5]) || !ggml_is_contiguous(op->src[6]) ||
        !ggml_is_contiguous(op->src[8]) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (op->src[7] && !ggml_is_contiguous(op->src[7])) return false;
    if (op->src[9] && !ggml_is_contiguous(op->src[9])) return false;

    return true;
}

void ggml_cuda_xkv_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k_hot   = dst->src[1];
    const ggml_tensor * v_hot   = dst->src[2];
    const ggml_tensor * k_cold  = dst->src[3];
    const ggml_tensor * v_cold  = dst->src[4];
    const ggml_tensor * entries = dst->src[5];
    const ggml_tensor * offsets = dst->src[6];
    const ggml_tensor * sinks   = dst->src[7];
    ggml_tensor       * status  = dst->src[8];
    const ggml_tensor * carry   = dst->src[9];

    ggml_xkv_attention_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    cuda_xkv_attention_push pc = {};
    pc.dim_k = p.dim_k;
    pc.dim_v = p.dim_v;
    pc.gqa = p.gqa_ratio;
    pc.n_groups = p.n_groups;
    pc.n_queries = p.n_queries;
    pc.hot_rows = p.hot_rows;
    pc.n_cold = p.n_cold;
    pc.n_entries = p.n_entries;
    pc.kv_head = p.kv_head;
    pc.stream = p.stream;
    pc.flags = (p.flags & GGML_XKV_ATTN_FLAG_REJECT_DUPES ? 1u : 0u) |
               (entries->ne[0] == 2 ? 2u : 0u) |
               (sinks ? 4u : 0u) |
               (p.flags & GGML_XKV_ATTN_FLAG_FIRST_TILE ? 8u : 0u) |
               (carry ? 16u : 0u) |
               (p.flags & GGML_XKV_ATTN_FLAG_FINAL_TILE ? 32u : 0u);
    pc.entry_stride = (uint32_t)entries->ne[0];
    pc.scale = p.scale;
    pc.softcap = p.logit_softcap;
    pc.k_hot_type = (uint32_t)k_hot->type;
    pc.v_hot_type = (uint32_t)v_hot->type;
    pc.k_cold_type = (uint32_t)k_cold->type;
    pc.v_cold_type = (uint32_t)v_cold->type;
    pc.q_nb1 = (uint32_t)q->nb[1];
    pc.q_nb2 = (uint32_t)q->nb[2];
    pc.kh_nb1 = (uint32_t)k_hot->nb[1];
    pc.kh_nb2 = (uint32_t)k_hot->nb[2];
    pc.kh_nb3 = (uint32_t)k_hot->nb[3];
    pc.vh_nb1 = (uint32_t)v_hot->nb[1];
    pc.vh_nb2 = (uint32_t)v_hot->nb[2];
    pc.vh_nb3 = (uint32_t)v_hot->nb[3];
    pc.kc_nb1 = (uint32_t)k_cold->nb[1];
    pc.vc_nb1 = (uint32_t)v_cold->nb[1];

    cudaStream_t stream = ctx.stream();

    // The first tile initializes the sticky status word. Later tiles preserve
    // an earlier failure; their explicit carry dependency orders the write.
    if (p.flags & GGML_XKV_ATTN_FLAG_FIRST_TILE) {
        CUDA_CHECK(cudaMemsetAsync(status->data, 0, sizeof(int32_t), stream));
    }

    const uint32_t rows = p.n_queries * p.gqa_ratio;
    constexpr int block_size = 64;
    int num_blocks = (rows + block_size - 1) / block_size;

    const float * q_data = (const float *)q->data;
    const uint8_t * kh_data = (const uint8_t *)k_hot->data;
    const uint8_t * vh_data = (const uint8_t *)v_hot->data;
    const uint8_t * kc_data = (const uint8_t *)k_cold->data;
    const uint8_t * vc_data = (const uint8_t *)v_cold->data;
    const int32_t * ent_data = (const int32_t *)entries->data;
    const int32_t * off_data = (const int32_t *)offsets->data;
    const float * snk_data = sinks ? (const float *)sinks->data : nullptr;
    int32_t * st_data = (int32_t *)status->data;
    float * dst_data = (float *)dst->data;
    const float * car_data = carry ? (const float *)carry->data : nullptr;

    if (num_blocks > 0) {
        k_xkv_attention<<<num_blocks, block_size, 0, stream>>>(
            q_data,
            kh_data,
            vh_data,
            kc_data,
            vc_data,
            ent_data,
            off_data,
            snk_data,
            st_data,
            dst_data,
            car_data,
            pc
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
