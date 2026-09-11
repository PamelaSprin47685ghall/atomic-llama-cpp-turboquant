// xkv-landmark-build.cu — Native on-device CUDA landmark construction.
//
// Mirrors ggml/src/ggml-vulkan/vulkan-shaders/xkv_landmark_build.comp:
// chunk-parallel reconstruction from FINAL encoded A_K/B_K streams with
// per-row forward RoPE, chunk mean-pooling, and Q8_0 / Turbo4_0 quant
// encoding in the same stream. All device-local; host reads only status
// (+ eb/srcfp telemetry via backend copies owned by the seal bridge).
// No D2H of K/V, no CPU oracle, no cudaMemcpy in this path.
//
// Status: [0]=code (0 ok, else GGML_XKV_LANDMARK_BUILD_STATUS_*),
// [1]=n_chunks_built, [2]=[3]=0. Failure leaves dst/eb/srcfp untouched
// (step 0 zeroes eb/srcfp only on success; steps 1/2 early-out).

#include "xkv-factorize.cuh"
#include "turbo-quant.cuh"
#include "ggml-xkv.h"
#include "ggml-xkv-landmark-build.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstring>

struct cuda_xkv_lmbuild_push {
    uint32_t n_rows, n_chunks, chunk_tokens;
    uint32_t total_dim, padded_dim, rank, pad_rank, n_layers;
    uint32_t landmark_type, a_type, b_type;
    uint32_t a_stride, b_stride, dst_stride;
    uint32_t max_dim, pos_is_64, rope_nelems;
    uint32_t status_idx;
    uint32_t seed_lo, phase_lo, phase_hi;
};

static __device__ __forceinline__ float lm_rd_half_ak(const uint8_t * ak, uint32_t off) {
    uint16_t h = (uint16_t)ak[off] | ((uint16_t)ak[off + 1u] << 8u);
    return __half2float(__ushort_as_half(h));
}

static __device__ __forceinline__ float lm_rd_f32_ak(const uint8_t * ak, uint32_t off) {
    uint32_t w = (uint32_t)ak[off] | ((uint32_t)ak[off + 1u] << 8u) |
                 ((uint32_t)ak[off + 2u] << 16u) | ((uint32_t)ak[off + 3u] << 24u);
    return *reinterpret_cast<const float *>(&w);
}

static __device__ __forceinline__ float lm_rd_half_bk(const uint8_t * bk, uint32_t off) {
    uint16_t h = (uint16_t)bk[off] | ((uint16_t)bk[off + 1u] << 8u);
    return __half2float(__ushort_as_half(h));
}

static __device__ __forceinline__ float lm_rd_f32_bk(const uint8_t * bk, uint32_t off) {
    uint32_t w = (uint32_t)bk[off] | ((uint32_t)bk[off + 1u] << 8u) |
                 ((uint32_t)bk[off + 2u] << 16u) | ((uint32_t)bk[off + 3u] << 24u);
    return *reinterpret_cast<const float *>(&w);
}

static __device__ __forceinline__ void lm_rope_trig_cw(float pos_f, float omega,
        float & out_c, float & out_s) {
    const float INV_TWO_PI = 0.15915494309189535f;
    const float TWO_PI_HI  = 6.2831854820251465f;
    const float TWO_PI_LO  = -1.748455588302094e-7f;
    float ang = pos_f * omega;
    float k_cycles = roundf(ang * INV_TWO_PI);
    float red_ang = fmaf(k_cycles, -TWO_PI_HI, ang);
    red_ang = fmaf(k_cycles, -TWO_PI_LO, red_ang);
    out_c = cosf(red_ang);
    out_s = sinf(red_ang);
}

static __device__ float lm_reconstruct_value(
        const uint8_t * __restrict__ ak,
        const uint8_t * __restrict__ bk,
        int arow, uint32_t feature,
        cuda_xkv_lmbuild_push p) {
    if (arow < 0 || feature >= p.total_dim) {
        return 0.0f;
    }
    uint32_t abase = (uint32_t)arow * p.a_stride;
    uint32_t bbase = feature * p.b_stride;
    float value = 0.0f;
    for (uint32_t k = 0u; k < p.rank; ++k) {
        float av;
        if (p.a_type == GGML_TYPE_F32) {
            av = lm_rd_f32_ak(ak, abase + k * 4u);
        } else if (p.a_type == GGML_TYPE_F16) {
            av = lm_rd_half_ak(ak, abase + k * 2u);
        } else if (p.a_type == GGML_TYPE_Q8_0) {
            uint32_t b = k / 32u, j = k % 32u;
            float d = lm_rd_half_ak(ak, abase + b * 34u);
            int8_t q = (int8_t)ak[abase + b * 34u + 2u + j];
            av = d * (float)q;
        } else if (p.a_type == GGML_TYPE_TURBO2_0) {
            uint32_t g = k / 128u, j = k % 128u;
            uint32_t off = abase + g * 34u;
            float norm = lm_rd_half_ak(ak, off);
            uint32_t packed = (uint32_t)ak[off + 2u + j / 4u];
            uint32_t qi = (packed >> (2u * (j % 4u))) & 3u;
            av = norm * TURBO_CENTROIDS_2BIT[qi];
        } else if (p.a_type == GGML_TYPE_TURBO3_0) {
            uint32_t g = k / 128u, j = k % 128u;
            uint32_t off = abase + g * 50u;
            float norm = lm_rd_half_ak(ak, off);
            uint32_t qb = (uint32_t)ak[off + 2u + j / 4u];
            uint32_t sb = (uint32_t)ak[off + 34u + j / 8u];
            uint32_t qi = ((qb >> (2u * (j % 4u))) & 3u) | (((sb >> (j % 8u)) & 1u) << 2u);
            av = norm * TURBO_CENTROIDS_3BIT[qi];
        } else if (p.a_type == GGML_TYPE_TURBO4_0) {
            uint32_t g = k / 128u, j = k % 128u;
            uint32_t off = abase + g * 68u;
            float norm = lm_rd_half_ak(ak, off);
            uint32_t packed = (uint32_t)ak[off + 4u + j / 2u];
            uint32_t qi = (j & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
            av = norm * TURBO_CENTROIDS_4BIT[qi];
        } else {
            av = 0.0f;
        }
        float bv;
        if (p.b_type == GGML_TYPE_F32) {
            bv = lm_rd_f32_bk(bk, bbase + k * 4u);
        } else if (p.b_type == GGML_TYPE_F16) {
            bv = lm_rd_half_bk(bk, bbase + k * 2u);
        } else if (p.b_type == GGML_TYPE_Q8_0) {
            uint32_t b = k / 32u, j = k % 32u;
            float d = lm_rd_half_bk(bk, bbase + b * 34u);
            int8_t q = (int8_t)bk[bbase + b * 34u + 2u + j];
            bv = d * (float)q;
        } else if (p.b_type == GGML_TYPE_TURBO2_0) {
            uint32_t g = k / 128u, j = k % 128u;
            uint32_t off = bbase + g * 34u;
            float norm = lm_rd_half_bk(bk, off);
            uint32_t packed = (uint32_t)bk[off + 2u + j / 4u];
            uint32_t qi = (packed >> (2u * (j % 4u))) & 3u;
            bv = norm * TURBO_CENTROIDS_2BIT[qi];
        } else if (p.b_type == GGML_TYPE_TURBO3_0) {
            uint32_t g = k / 128u, j = k % 128u;
            uint32_t off = bbase + g * 50u;
            float norm = lm_rd_half_bk(bk, off);
            uint32_t qb = (uint32_t)bk[off + 2u + j / 4u];
            uint32_t sb = (uint32_t)bk[off + 34u + j / 8u];
            uint32_t qi = ((qb >> (2u * (j % 4u))) & 3u) | (((sb >> (j % 8u)) & 1u) << 2u);
            bv = norm * TURBO_CENTROIDS_3BIT[qi];
        } else if (p.b_type == GGML_TYPE_TURBO4_0) {
            uint32_t g = k / 128u, j = k % 128u;
            uint32_t off = bbase + g * 68u;
            float norm = lm_rd_half_bk(bk, off);
            uint32_t packed = (uint32_t)bk[off + 4u + j / 2u];
            uint32_t qi = (j & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
            bv = norm * TURBO_CENTROIDS_4BIT[qi];
        } else {
            bv = 0.0f;
        }
        value += av * bv;
    }
    return value;
}

static __device__ __forceinline__ int64_t lm_load_position(
        const uint32_t * __restrict__ pos_raw, uint32_t row_index, uint32_t pos_is_64) {
    if (pos_is_64 == 0u) {
        return (int64_t)(int32_t)pos_raw[row_index];
    }
    uint32_t lo = pos_raw[2u * row_index];
    int64_t hi = (int64_t)(int32_t)pos_raw[2u * row_index + 1u];
    return (hi << 32) | (int64_t)(uint64_t)lo;
}

static __device__ __forceinline__ uint64_t lm_fnv_byte(uint64_t h, uint32_t v) {
    return (h ^ (uint64_t)(v & 255u)) * 0x100000001b3ULL;
}

static __device__ __forceinline__ uint64_t lm_fnv_u32(uint64_t h, uint32_t v) {
    h = lm_fnv_byte(h, v);
    h = lm_fnv_byte(h, v >> 8u);
    h = lm_fnv_byte(h, v >> 16u);
    return lm_fnv_byte(h, v >> 24u);
}

// ---- step 0: validate + zero telemetry ----
static __global__ void k_xkv_lmbuild_validate(
        const int32_t * __restrict__ lmeta,
        uint32_t      * __restrict__ ebits,
        uint32_t      * __restrict__ srcfp,
        int32_t       * __restrict__ st,
        cuda_xkv_lmbuild_push p) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    bool ok = (p.n_rows > 0u && p.n_chunks > 0u && p.chunk_tokens > 0u &&
               p.total_dim > 0u && p.total_dim <= 4096u && p.padded_dim >= p.total_dim &&
               p.rank > 0u && p.pad_rank >= p.rank && p.n_layers > 0u &&
               (p.landmark_type == GGML_TYPE_Q8_0 || p.landmark_type == GGML_TYPE_TURBO4_0) &&
               (p.a_type == GGML_TYPE_F32 || p.a_type == GGML_TYPE_F16 ||
                p.a_type == GGML_TYPE_Q8_0 || p.a_type == GGML_TYPE_TURBO2_0 ||
                p.a_type == GGML_TYPE_TURBO3_0 || p.a_type == GGML_TYPE_TURBO4_0) &&
               (p.b_type == GGML_TYPE_F32 || p.b_type == GGML_TYPE_F16 ||
                p.b_type == GGML_TYPE_Q8_0 || p.b_type == GGML_TYPE_TURBO2_0 ||
                p.b_type == GGML_TYPE_TURBO3_0 || p.b_type == GGML_TYPE_TURBO4_0));
    if (ok) {
        uint32_t run = 0u;
        for (uint32_t li = 0u; li < p.n_layers; ++li) {
            int32_t foff = lmeta[li * 6u + 0u], fdim = lmeta[li * 6u + 1u];
            int32_t ehd = lmeta[li * 6u + 2u], rdim = lmeta[li * 6u + 3u];
            int32_t rmode = lmeta[li * 6u + 4u], roff = lmeta[li * 6u + 5u];
            if (foff != (int32_t)run || fdim <= 0) {
                ok = false; break;
            }
            if (ehd <= 0 || (fdim % ehd) != 0 || rdim < 0 || rdim > ehd || (rdim & 1) != 0) {
                ok = false; break;
            }
            if (rmode < 0 || rmode > 1 || roff < 0) {
                ok = false; break;
            }
            if (rdim > 0 && (uint32_t)(roff + rdim) > p.rope_nelems) {
                ok = false; break;
            }
            run += (uint32_t)fdim;
        }
        if (ok && run != p.total_dim) {
            ok = false;
        }
    }
    st[p.status_idx] = ok ? 0 : 4;
    st[p.status_idx + 1u] = 0;
    if (ok) {
        for (uint32_t c = 0u; c < p.n_chunks; ++c) {
            ebits[c] = 0u;
            srcfp[2u * c] = 0u;
            srcfp[2u * c + 1u] = 0u;
        }
    }
}

// ---- step 1: parallel reconstruction + cooperative block quantization ----
static __global__ void k_xkv_lmbuild_build(
        const uint8_t * __restrict__ ak,
        const uint8_t * __restrict__ bk,
        const int32_t * __restrict__ rows,
        const uint32_t * __restrict__ pos_raw,
        const int32_t * __restrict__ lmeta,
        const float   * __restrict__ rope,
        uint32_t      * __restrict__ ebits,
        uint8_t       * __restrict__ dst,
        int32_t       * __restrict__ st,
        cuda_xkv_lmbuild_push p) {
    __shared__ float sh_v[128];
    __shared__ float sh_red[128];
    __shared__ uint32_t sh_u[128];
    uint32_t lid = threadIdx.x;
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t blk_sz = (p.landmark_type == GGML_TYPE_Q8_0) ? 32u : 128u;
    uint32_t nb = p.padded_dim / blk_sz;
    uint32_t wgid = blockIdx.x;
    if (wgid >= p.n_chunks * nb) {
        return;
    }
    uint32_t cid = wgid / nb;
    uint32_t bid = wgid % nb;

    uint32_t D = p.total_dim;
    uint32_t feat = bid * blk_sz + lid;
    bool valid_feat = (lid < blk_sz) && (feat < D);

    uint32_t foff = 0u, ehd = 0u, rdim = 0u, rmode = 0u, roff = 0u;
    if (valid_feat) {
        for (uint32_t li = 0u; li < p.n_layers; ++li) {
            uint32_t lo = (uint32_t)lmeta[li * 6u + 0u], ld = (uint32_t)lmeta[li * 6u + 1u];
            if (feat >= lo && feat < lo + ld) {
                foff = lo;
                ehd = (uint32_t)lmeta[li * 6u + 2u];
                rdim = (uint32_t)lmeta[li * 6u + 3u];
                rmode = (uint32_t)lmeta[li * 6u + 4u];
                roff = (uint32_t)lmeta[li * 6u + 5u];
                break;
            }
        }
    }

    uint32_t r0 = cid * p.chunk_tokens;
    uint32_t r1 = r0 + p.chunk_tokens;
    if (r1 > p.n_rows) {
        r1 = p.n_rows;
    }

    float acc_feat = 0.0f;
    bool bad = false;
    for (uint32_t ri = r0; ri < r1; ++ri) {
        int32_t ar = rows[ri];
        if (ar < 0) {
            bad = true; break;
        }
        float dot_val = valid_feat ? lm_reconstruct_value(ak, bk, ar, feat, p) : 0.0f;
        if (valid_feat) {
            float phased_val = dot_val;
            if (rdim > 0u) {
                int64_t pos = lm_load_position(pos_raw, ri, p.pos_is_64);
                float fpos = (float)pos;
                uint32_t fc = rdim / 2u;
                uint32_t head_local = (feat - foff) % ehd;
                if (head_local < rdim) {
                    uint32_t head_base = feat - head_local;
                    if (rmode == 0u) {
                        if (head_local < fc) {
                            float re = dot_val;
                            float im = lm_reconstruct_value(ak, bk, ar, head_base + head_local + fc, p);
                            float om = rope[roff + head_local];
                            float mg = rope[roff + fc + head_local];
                            if (mg <= 0.0f) {
                                mg = 1.0f;
                            }
                            float c, sn;
                            lm_rope_trig_cw(fpos, om, c, sn);
                            phased_val = (re * c - im * sn) * mg;
                        } else {
                            float im = dot_val;
                            uint32_t f = head_local - fc;
                            float re = lm_reconstruct_value(ak, bk, ar, head_base + f, p);
                            float om = rope[roff + f];
                            float mg = rope[roff + fc + f];
                            if (mg <= 0.0f) {
                                mg = 1.0f;
                            }
                            float c, sn;
                            lm_rope_trig_cw(fpos, om, c, sn);
                            phased_val = (re * sn + im * c) * mg;
                        }
                    } else {
                        uint32_t pair_idx = head_local / 2u;
                        float om = rope[roff + pair_idx];
                        float mg = rope[roff + fc + pair_idx];
                        if (mg <= 0.0f) {
                            mg = 1.0f;
                        }
                        float c, sn;
                        lm_rope_trig_cw(fpos, om, c, sn);
                        if ((head_local & 1u) == 0u) {
                            float re = dot_val;
                            float im = lm_reconstruct_value(ak, bk, ar, feat + 1u, p);
                            phased_val = (re * c - im * sn) * mg;
                        } else {
                            float im = dot_val;
                            float re = lm_reconstruct_value(ak, bk, ar, feat - 1u, p);
                            phased_val = (re * sn + im * c) * mg;
                        }
                    }
                }
            }
            if (isnan(phased_val) || isinf(phased_val)) {
                bad = true;
            }
            acc_feat += phased_val;
        }
    }
    if (bad) {
        atomicMax(&st[p.status_idx], 2);
        return;
    }

    float inv_rows = 1.0f / (float)(r1 - r0);
    float mean_val = valid_feat ? (acc_feat * inv_rows) : 0.0f;
    sh_v[lid] = mean_val;
    __syncthreads();

    uint32_t dbase = cid * p.dst_stride;
    const float inv_sqrt_128 = 0.08838834764831845f;

    if (p.landmark_type == GGML_TYPE_Q8_0) {
        if (lid < 32u) {
            sh_red[lid] = fabsf(mean_val);
        } else {
            sh_red[lid] = 0.0f;
        }
        __syncthreads();
        for (uint32_t s = 16u; s > 0u; s >>= 1u) {
            if (lid < s) {
                sh_red[lid] = fmaxf(sh_red[lid], sh_red[lid + s]);
            }
            __syncthreads();
        }
        float amax = sh_red[0];
        float dd = amax / 127.0f;
        uint16_t dd_bits = __half_as_ushort(__float2half(dd));
        float stored_dd = __half2float(__ushort_as_half(dd_bits));
        float id = (dd != 0.0f) ? (1.0f / dd) : 0.0f;
        if (lid == 0u) {
            dst[dbase + bid * 34u] = (uint8_t)(dd_bits & 255u);
            dst[dbase + bid * 34u + 1u] = (uint8_t)((dd_bits >> 8u) & 255u);
        }
        float err_upper = 0.0f;
        if (lid < 32u) {
            int q = (int)roundf(mean_val * id);
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
            dst[dbase + bid * 34u + 2u + lid] = (uint8_t)((uint32_t)q & 255u);
            float decv = (float)q * stored_dd;
            if (feat < D) {
                float diff = decv - mean_val;
                err_upper = fabsf(diff) +
                    4.0f * 1.1920928955078125e-7f * (fabsf(decv) + fabsf(mean_val) + 1.0f);
            }
        }
        sh_red[lid] = err_upper;
        __syncthreads();
        for (uint32_t s = 64u; s > 0u; s >>= 1u) {
            if (lid < s) {
                sh_red[lid] = fmaxf(sh_red[lid], sh_red[lid + s]);
            }
            __syncthreads();
        }
        if (lid == 0u) {
            atomicMax(&ebits[cid], __float_as_uint(sh_red[0]));
        }
    } else {
        sh_red[lid] = mean_val * mean_val;
        __syncthreads();
        for (uint32_t s = 64u; s > 0u; s >>= 1u) {
            if (lid < s) {
                sh_red[lid] += sh_red[lid + s];
            }
            __syncthreads();
        }
        float nrm = sqrtf(sh_red[0]);
        float inv_nrm = (nrm > 1e-10f) ? (1.0f / nrm) : 0.0f;
        float rot = mean_val * inv_nrm * TURBO_WHT_SIGNS1[lid];
        sh_v[lid] = rot;
        __syncthreads();
        for (uint32_t h = 1u; h < 128u; h <<= 1u) {
            float a = sh_v[lid];
            float b = sh_v[lid ^ h];
            __syncthreads();
            sh_v[lid] = ((lid & h) == 0u) ? (a + b) : (b - a);
            __syncthreads();
        }
        float wht_val = sh_v[lid] * inv_sqrt_128 * TURBO_WHT_SIGNS2[lid];
        uint32_t ix = 0u;
        for (uint32_t i = 0u; i < 15u; ++i) {
            if (wht_val >= TURBO_MID_4BIT[i]) {
                ix = i + 1u;
            }
        }
        sh_u[lid] = ix;
        sh_red[lid] = TURBO_CENTROIDS_4BIT[ix] * TURBO_CENTROIDS_4BIT[ix];
        __syncthreads();
        for (uint32_t s = 64u; s > 0u; s >>= 1u) {
            if (lid < s) {
                sh_red[lid] += sh_red[lid + s];
            }
            __syncthreads();
        }
        float rn = sqrtf(sh_red[0]);
        float cnorm = (rn > 1e-10f) ? (nrm / rn) : nrm;
        uint16_t cnorm_bits = __half_as_ushort(__float2half(cnorm));
        float stored_cnorm = __half2float(__ushort_as_half(cnorm_bits));
        uint32_t boff = dbase + bid * 68u;
        if (lid == 0u) {
            dst[boff] = (uint8_t)(cnorm_bits & 255u);
            dst[boff + 1u] = (uint8_t)((cnorm_bits >> 8u) & 255u);
            dst[boff + 2u] = 0u;
            dst[boff + 3u] = 0u;
        }
        if (lid < 64u) {
            uint32_t lo = sh_u[2u * lid] & 15u, hi = sh_u[2u * lid + 1u] & 15u;
            dst[boff + 4u + lid] = (uint8_t)(lo | (hi << 4u));
        }
        float c_val = stored_cnorm * TURBO_CENTROIDS_4BIT[ix] * TURBO_WHT_SIGNS2[lid];
        sh_v[lid] = c_val;
        __syncthreads();
        for (uint32_t h = 1u; h < 128u; h <<= 1u) {
            float a = sh_v[lid];
            float b = sh_v[lid ^ h];
            __syncthreads();
            sh_v[lid] = ((lid & h) == 0u) ? (a + b) : (b - a);
            __syncthreads();
        }
        float dec_val = sh_v[lid] * inv_sqrt_128 * TURBO_WHT_SIGNS1[lid];
        float diff = (feat < D) ? (dec_val - mean_val) : 0.0f;
        sh_red[lid] = (feat < D) ? fabsf(diff) +
            4.0f * 1.1920928955078125e-7f * (fabsf(dec_val) + fabsf(mean_val) + 1.0f) : 0.0f;
        __syncthreads();
        for (uint32_t s = 64u; s > 0u; s >>= 1u) {
            if (lid < s) {
                sh_red[lid] = fmaxf(sh_red[lid], sh_red[lid + s]);
            }
            __syncthreads();
        }
        if (lid == 0u) {
            atomicMax(&ebits[cid], __float_as_uint(sh_red[0]));
        }
    }
}

// ---- step 2: finalize bound + fingerprint ----
static __global__ void k_xkv_lmbuild_finalize(
        const int32_t  * __restrict__ rows,
        const uint32_t * __restrict__ pos_raw,
        uint32_t       * __restrict__ ebits,
        uint32_t       * __restrict__ srcfp,
        const uint8_t  * __restrict__ dst,
        int32_t        * __restrict__ st,
        cuda_xkv_lmbuild_push p) {
    uint32_t lid = threadIdx.x;
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t cid = blockIdx.x;
    if (cid >= p.n_chunks || lid != 0u) {
        return;
    }
    float max_abs = __uint_as_float(ebits[cid]);
    float bound = sqrtf((float)p.total_dim) * max_abs;
    bound = bound * (1.0f + 8.0f * 1.1920928955078125e-7f) + 1.0e-6f;
    if (!(bound >= 1.0e-6f) || isnan(bound) || isinf(bound)) {
        bound = 1.0e-4f;
    }
    ebits[cid] = __float_as_uint(bound);

    uint64_t h = 0xcbf29ce484222325ULL;
    h = lm_fnv_u32(h, p.seed_lo);
    h = lm_fnv_u32(h, p.phase_lo);
    h = lm_fnv_u32(h, p.phase_hi);
    h = lm_fnv_u32(h, cid);
    uint32_t r0 = cid * p.chunk_tokens;
    uint32_t r1 = r0 + p.chunk_tokens;
    if (r1 > p.n_rows) {
        r1 = p.n_rows;
    }
    for (uint32_t ri = r0; ri < r1; ++ri) {
        h = lm_fnv_u32(h, (uint32_t)rows[ri]);
        if (p.pos_is_64 != 0u) {
            h = lm_fnv_u32(h, pos_raw[2u * ri]);
            h = lm_fnv_u32(h, pos_raw[2u * ri + 1u]);
        } else {
            h = lm_fnv_u32(h, pos_raw[ri]);
            int32_t pv = (int32_t)pos_raw[ri];
            h = lm_fnv_u32(h, (uint32_t)(pv < 0 ? 0xFFFFFFFFu : 0u));
        }
    }
    uint32_t dbase = cid * p.dst_stride;
    for (uint32_t i = 0u; i < p.dst_stride; ++i) {
        h = lm_fnv_byte(h, (uint32_t)dst[dbase + i]);
    }
    if (h == 0ULL) {
        h = 1ULL;
    }
    srcfp[2u * cid] = (uint32_t)(h & 0xFFFFFFFFULL);
    srcfp[2u * cid + 1u] = (uint32_t)(h >> 32u);
    if (cid == 0u) {
        st[p.status_idx + 1u] = (int32_t)p.n_chunks;
    }
}

bool ggml_cuda_xkv_landmark_build_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_LANDMARK_BUILD) {
        return false;
    }
    for (int i = 0; i < 10; ++i) {
        if (!op->src[i]) {
            return false;
        }
    }
    ggml_xkv_landmark_build_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    return ggml_xkv_landmark_build_supports(
        op->src[0], op->src[1], op->src[2], op->src[3], op->src[4],
        op->src[5], op->src[6], op->src[7], op->src[8], op->src[9],
        op, &p, err, sizeof(err));
}

void ggml_cuda_xkv_landmark_build(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a_k         = dst->src[0];
    const ggml_tensor * b_k         = dst->src[1];
    const ggml_tensor * rows        = dst->src[2];
    const ggml_tensor * positions   = dst->src[3];
    const ggml_tensor * layer_meta  = dst->src[4];
    const ggml_tensor * rope_tables = dst->src[5];
    const ggml_tensor * eb          = dst->src[7];
    const ggml_tensor * srcfp       = dst->src[8];
    const ggml_tensor * status      = dst->src[9];

    ggml_xkv_landmark_build_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    uint32_t max_dim = p.max_feature_dim ? p.max_feature_dim : p.total_dim;

    cuda_xkv_lmbuild_push pc = {};
    pc.n_rows = p.n_rows; pc.n_chunks = p.n_chunks; pc.chunk_tokens = p.chunk_tokens;
    pc.total_dim = p.total_dim; pc.padded_dim = p.padded_dim;
    pc.rank = p.rank; pc.pad_rank = p.pad_rank; pc.n_layers = p.n_layers;
    pc.landmark_type = p.landmark_type; pc.a_type = p.a_type; pc.b_type = p.b_type;
    pc.a_stride = (uint32_t)ggml_row_size(a_k->type, a_k->ne[0]);
    pc.b_stride = (uint32_t)ggml_row_size(b_k->type, b_k->ne[0]);
    pc.dst_stride = (uint32_t)ggml_row_size(dst->type, dst->ne[0]);
    pc.max_dim = max_dim;
    pc.pos_is_64 = (positions->type == GGML_TYPE_I64) ? 1u : 0u;
    pc.rope_nelems = (uint32_t)ggml_nelements(rope_tables);
    pc.status_idx = 0u;
    pc.seed_lo = p.seed;
    pc.phase_lo = p.phase_lo;
    pc.phase_hi = p.phase_hi;

    cudaStream_t stream = ctx.stream();

    // step 0: validate
    k_xkv_lmbuild_validate<<<1, 1, 0, stream>>>(
        (const int32_t *)layer_meta->data,
        (uint32_t      *)eb->data,
        (uint32_t      *)srcfp->data,
        (int32_t       *)status->data,
        pc
    );

    // step 1: one 128-thread block per (chunk, codec-block)
    {
        uint32_t block = (p.landmark_type == GGML_TYPE_Q8_0) ? 32u : 128u;
        uint32_t n_blocks = p.padded_dim / block;
        uint64_t n_workgroups = (uint64_t)p.n_chunks * n_blocks;
        if (n_workgroups > 0 && n_workgroups <= UINT32_MAX) {
            k_xkv_lmbuild_build<<<(uint32_t)n_workgroups, 128, 0, stream>>>(
                (const uint8_t *)a_k->data,
                (const uint8_t *)b_k->data,
                (const int32_t *)rows->data,
                (const uint32_t *)positions->data,
                (const int32_t *)layer_meta->data,
                (const float   *)rope_tables->data,
                (uint32_t      *)eb->data,
                (uint8_t       *)dst->data,
                (int32_t       *)status->data,
                pc
            );
        }
    }

    // step 2: finalize bound + fingerprint (one block per chunk)
    k_xkv_lmbuild_finalize<<<p.n_chunks, 128, 0, stream>>>(
        (const int32_t  *)rows->data,
        (const uint32_t *)positions->data,
        (uint32_t       *)eb->data,
        (uint32_t       *)srcfp->data,
        (const uint8_t  *)dst->data,
        (int32_t        *)status->data,
        pc
    );
    CUDA_CHECK(cudaGetLastError());
}
