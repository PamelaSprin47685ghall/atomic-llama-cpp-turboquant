// xkv-factorize.cu — Native on-device CUDA factorization core.
//
// Mirrors ggml/src/ggml-vulkan/vulkan-shaders/xkv_factorize.comp step for
// step (same push layout, same serial summation order, same counter
// Rademacher, CGS2x2, cyclic Jacobi, sign/balance, exact quant encode,
// all-row tiled final-codec residuals). All stages run on-device on the
// backend stream; the host reads only the scalar status word. No D2H of
// K/V/factors, no CPU oracle, no cudaMemcpy of any kind in this path.
//
// GEMM steps (Z = X^T Q, Y = X Z, C = Q^T X, G = C C^T) use cuBLAS SGEMM
// (already linked via the backend context handle). The fused Y = X*Omega
// projection, CGS2/Jacobi orthogonalization, factor assembly, encodes and
// residuals use dedicated kernels preserving the shader's summation order.
//
// Status word: 0 success, 1 validation fail, 2 compute/encode fail,
// 4 residual-bad (matches Vulkan steps 12/13). Outputs stay unpublished
// on failure (encode/residual kernels early-out on nonzero status; the
// validation kernel runs first on the same stream).

#include "xkv-factorize.cuh"
#include "turbo-quant.cuh"
#include "ggml-xkv.h"
#include "ggml-xkv-factor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include <cstring>

struct cuda_xkv_factor_push {
    uint32_t n, m, r, l;
    uint32_t seed_lo, seed_hi;
    uint32_t balance;
    uint32_t type_a, type_b;
    uint32_t pad_a, pad_b;
    uint32_t off_Y, off_Z, off_C;
    uint32_t off_G, off_V, off_S;
    uint32_t off_ORD, off_AF, off_BTF;
    uint32_t status_idx;
    uint32_t qr_rows;
    uint32_t aux; // QR base (off_Y/off_Z) or residual band count G
};

// ---- counter Rademacher (bit-exact with CPU oracle / Vulkan) ----
static __device__ __forceinline__ uint32_t xkv_fmix32_cuda(uint32_t h) {
    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;
    return h;
}

static __device__ __forceinline__ float xkv_radem_cuda(uint32_t k, uint32_t j,
        uint32_t seed_lo, uint32_t seed_hi, uint32_t l) {
    uint32_t h = seed_lo ^ (k * 2654435761u) ^ (j * 2246822519u)
               ^ (l * 3266489917u) ^ (seed_hi * 668265263u);
    h = xkv_fmix32_cuda(h == 0u ? 0x9E3779B9u : h);
    return (h & 1u) != 0u ? 1.0f : -1.0f;
}

static __device__ __forceinline__ uint32_t xkv_rowbytes_cuda(uint32_t t, uint32_t pr) {
    if (t == GGML_TYPE_F32)      return pr * 4u;
    if (t == GGML_TYPE_F16)      return pr * 2u;
    if (t == GGML_TYPE_Q8_0)     return (pr / 32u) * 34u;
    if (t == GGML_TYPE_TURBO2_0) return (pr / 128u) * 34u;
    if (t == GGML_TYPE_TURBO3_0) return (pr / 128u) * 50u;
    return (pr / 128u) * 68u; // TURBO4_0
}

static __device__ __forceinline__ float xkv_load_half_cuda(const uint8_t * base, uint32_t off) {
    uint16_t h = (uint16_t)base[off] | ((uint16_t)base[off + 1u] << 8u);
    return __half2float(__ushort_as_half(h));
}

static __device__ __forceinline__ float xkv_rhalf_away_cuda(float v) {
    return v >= 0.0f ? floorf(v + 0.5f) : -floorf(-v + 0.5f);
}

// Packed-stream decode (ROTATED Turbo domain = production dot contract)
static __device__ __forceinline__ float xkv_decA_cuda(const uint8_t * oa,
        uint32_t i, uint32_t k, uint32_t t, uint32_t pr) {
    uint32_t base = i * xkv_rowbytes_cuda(t, pr);
    if (t == GGML_TYPE_F32) {
        uint32_t o = base + k * 4u;
        uint32_t w = (uint32_t)oa[o] | ((uint32_t)oa[o+1u] << 8u) |
                     ((uint32_t)oa[o+2u] << 16u) | ((uint32_t)oa[o+3u] << 24u);
        return *reinterpret_cast<const float *>(&w);
    }
    if (t == GGML_TYPE_F16) {
        return xkv_load_half_cuda(oa, base + k * 2u);
    }
    if (t == GGML_TYPE_Q8_0) {
        uint32_t b = k / 32u, j = k % 32u;
        uint32_t bo = base + b * 34u;
        float d = xkv_load_half_cuda(oa, bo);
        int8_t q = (int8_t)oa[bo + 2u + j];
        return d * (float)q;
    }
    uint32_t g = k / 128u, j = k % 128u;
    uint32_t bb = (t == GGML_TYPE_TURBO2_0) ? 34u : ((t == GGML_TYPE_TURBO3_0) ? 50u : 68u);
    uint32_t bo = base + g * bb;
    float norm = xkv_load_half_cuda(oa, bo);
    float cv = 0.0f;
    if (t == GGML_TYPE_TURBO2_0) {
        uint32_t qb = (uint32_t)oa[bo + 2u + j / 4u];
        cv = TURBO_CENTROIDS_2BIT[(qb >> (2u * (j % 4u))) & 3u];
    } else if (t == GGML_TYPE_TURBO3_0) {
        uint32_t qb = (uint32_t)oa[bo + 2u + j / 4u];
        uint32_t sb = (uint32_t)oa[bo + 34u + j / 8u];
        uint32_t idx = ((qb >> (2u * (j % 4u))) & 3u) | (((sb >> (j % 8u)) & 1u) << 2u);
        cv = TURBO_CENTROIDS_3BIT[idx];
    } else {
        uint32_t qb = (uint32_t)oa[bo + 4u + j / 2u];
        uint32_t idx = ((j % 2u) == 0u) ? (qb & 15u) : (qb >> 4u);
        cv = TURBO_CENTROIDS_4BIT[idx];
    }
    return norm * cv;
}

static __device__ __forceinline__ float xkv_decB_cuda(const uint8_t * ob,
        uint32_t j, uint32_t k, uint32_t t, uint32_t pr) {
    return xkv_decA_cuda(ob, j, k, t, pr);
}

// ---- step 0: validate ----
static __global__ void k_xkv_factor_validate(
        const float * __restrict__ x,
        float       * __restrict__ s,
        int32_t     * __restrict__ st,
        cuda_xkv_factor_push p) {
    (void) x; (void) s;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bool ok = (p.n > 0u && p.m > 0u && p.r > 0u && p.l > 0u && p.l <= 512u &&
                   p.r <= p.n && p.r <= p.m &&
                   (p.balance <= 2u) &&
                   (p.pad_a == p.r || p.pad_a > p.r) && (p.pad_b == p.r || p.pad_b > p.r) &&
                   (p.pad_a == p.pad_b));
        st[p.status_idx] = ok ? 0 : 1;
    }
}

// ---- step 1: Y = X*Omega fused ----
static __global__ void k_xkv_factor_proj_y(
        const float * __restrict__ x,
        float       * __restrict__ s,
        const int32_t * __restrict__ st,
        cuda_xkv_factor_push p) {
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = p.n * p.l;
    if (gid >= total) {
        return;
    }
    uint32_t i = gid / p.l, j = gid % p.l;
    float acc = 0.0f;
    for (uint32_t k = 0u; k < p.m; ++k) {
        acc += x[i * p.m + k] * xkv_radem_cuda(k, j, p.seed_lo, p.seed_hi, p.l);
    }
    s[p.off_Y + gid] = acc;
}

// ---- step 2: CGS2x2 in place on T[qr_rows x l] at aux; single block ----
static __global__ void k_xkv_factor_cgs2(
        float         * __restrict__ s,
        int32_t       * __restrict__ st,
        cuda_xkv_factor_push p) {
    __shared__ float sh_red[256];
    __shared__ float sh_bc;
    uint32_t lid = threadIdx.x;
    if (gridDim.x != 1u) {
        if (lid == 0u) {
            st[p.status_idx] = 1;
        }
        return;
    }
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t K = p.qr_rows;
    uint32_t TB = p.aux;
    uint32_t l = p.l;
    for (uint32_t j = 0u; j < l; ++j) {
        for (uint32_t pass = 0u; pass < 2u; ++pass) {
            for (uint32_t j0 = 0u; j0 < j; ++j0) {
                float part = 0.0f;
                for (uint32_t i = lid; i < K; i += blockDim.x) {
                    part += s[TB + i * l + j0] * s[TB + i * l + j];
                }
                sh_red[lid] = part;
                __syncthreads();
                if (lid == 0u) {
                    float d = 0.0f;
                    for (uint32_t t = 0u; t < blockDim.x; ++t) {
                        d += sh_red[t];
                    }
                    sh_bc = d;
                }
                __syncthreads();
                float d = sh_bc;
                for (uint32_t i = lid; i < K; i += blockDim.x) {
                    s[TB + i * l + j] -= d * s[TB + i * l + j0];
                }
                __syncthreads();
            }
        }
        float part = 0.0f;
        for (uint32_t i = lid; i < K; i += blockDim.x) {
            float v = s[TB + i * l + j];
            part += v * v;
        }
        sh_red[lid] = part;
        __syncthreads();
        if (lid == 0u) {
            float nn = 0.0f;
            for (uint32_t t = 0u; t < blockDim.x; ++t) {
                nn += sh_red[t];
            }
            sh_bc = sqrtf(nn);
        }
        __syncthreads();
        float nn = sh_bc;
        if (nn > 1e-12f) {
            for (uint32_t i = lid; i < K; i += blockDim.x) {
                s[TB + i * l + j] /= nn;
            }
        } else {
            for (uint32_t i = lid; i < K; i += blockDim.x) {
                s[TB + i * l + j] = 0.0f;
            }
        }
        __syncthreads();
    }
}

// ---- step 7: parallel round-robin cyclic Jacobi on G + V; single block ----
static __global__ void k_xkv_factor_jacobi(
        float         * __restrict__ s,
        int32_t       * __restrict__ st,
        cuda_xkv_factor_push p) {
    __shared__ float  sh_red[256];
    __shared__ uint32_t sh_np;
    __shared__ uint32_t sh_P[256];
    __shared__ uint32_t sh_Q[256];
    __shared__ float  sh_C[256];
    __shared__ float  sh_Sv[256];
    __shared__ float  sh_keys[512];
    __shared__ uint32_t sh_idx[512];
    __shared__ uint32_t sh_done;
    uint32_t lid = threadIdx.x;
    if (gridDim.x != 1u) {
        if (lid == 0u) {
            st[p.status_idx] = 1;
        }
        return;
    }
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t l = p.l;
    uint32_t L2 = l + (l & 1u);
    uint32_t npair = L2 / 2u;
    for (uint32_t i = lid; i < l * l; i += blockDim.x) {
        uint32_t rr = i / l, cc = i % l;
        s[p.off_V + i] = (rr == cc) ? 1.0f : 0.0f;
    }
    __syncthreads();
    if (lid == 0u) {
        sh_done = 0u;
    }
    __syncthreads();
    for (uint32_t sweep = 0u; sweep < 20u; ++sweep) {
        for (uint32_t t = 0u; t + 1u < L2; ++t) {
            if (lid == 0u) {
                uint32_t cnt = 0u;
                uint32_t a0 = t, b0 = L2 - 1u;
                if (a0 < l && b0 < l) {
                    sh_P[cnt] = a0; sh_Q[cnt] = b0; cnt++;
                }
                for (uint32_t k = 1u; k < npair; ++k) {
                    uint32_t a = (t + k) % (L2 - 1u);
                    uint32_t b = (t + L2 - 1u - k) % (L2 - 1u);
                    if (a < l && b < l) {
                        sh_P[cnt] = a; sh_Q[cnt] = b; cnt++;
                    }
                }
                sh_np = cnt;
            }
            __syncthreads();
            uint32_t cnt = sh_np;
            if (cnt == 0u) {
                __syncthreads();
                continue;
            }
            if (lid < cnt) {
                uint32_t a = sh_P[lid], b = sh_Q[lid];
                float app = s[p.off_G + a * l + a];
                float aqq = s[p.off_G + b * l + b];
                float apq = s[p.off_G + a * l + b];
                float tau = (aqq - app) / (2.0f * apq + 1e-30f);
                float tt = (tau >= 0.0f ? 1.0f : -1.0f) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
                float c = 1.0f / sqrtf(1.0f + tt * tt);
                sh_C[lid] = c; sh_Sv[lid] = tt * c;
            }
            __syncthreads();
            for (uint32_t e = lid; e < cnt * l; e += blockDim.x) {
                uint32_t pi = e / l;
                uint32_t cc = e % l;
                uint32_t a = sh_P[pi], b = sh_Q[pi];
                float c = sh_C[pi], sv = sh_Sv[pi];
                float gai = s[p.off_G + cc * l + a];
                float gbi = s[p.off_G + cc * l + b];
                s[p.off_G + cc * l + a] = c * gai - sv * gbi;
                s[p.off_G + cc * l + b] = sv * gai + c * gbi;
                float vai = s[p.off_V + cc * l + a];
                float vbi = s[p.off_V + cc * l + b];
                s[p.off_V + cc * l + a] = c * vai - sv * vbi;
                s[p.off_V + cc * l + b] = sv * vai + c * vbi;
            }
            __syncthreads();
            for (uint32_t e = lid; e < cnt * l; e += blockDim.x) {
                uint32_t pi = e / l;
                uint32_t cc = e % l;
                uint32_t a = sh_P[pi], b = sh_Q[pi];
                float c = sh_C[pi], sv = sh_Sv[pi];
                float gaj = s[p.off_G + a * l + cc];
                float gbj = s[p.off_G + b * l + cc];
                s[p.off_G + a * l + cc] = c * gaj - sv * gbj;
                s[p.off_G + b * l + cc] = sv * gaj + c * gbj;
            }
            __syncthreads();
        }
        float poff = 0.0f, pdiag = 0.0f;
        uint32_t total = l * l;
        for (uint32_t e = lid; e < total; e += blockDim.x) {
            uint32_t i = e / l, j = e % l;
            if (i < j) {
                poff = fmaxf(poff, fabsf(s[p.off_G + i * l + j]));
            } else if (i == j) {
                pdiag = fmaxf(pdiag, fabsf(s[p.off_G + i * l + i]));
            }
        }
        sh_red[lid] = poff;
        sh_C[lid] = pdiag;
        __syncthreads();
        if (lid == 0u) {
            float mx = 0.0f, md = 0.0f;
            for (uint32_t t = 0u; t < blockDim.x; ++t) {
                mx = fmaxf(mx, sh_red[t]); md = fmaxf(md, sh_C[t]);
            }
            sh_done = (mx <= 1e-5f * md || md == 0.0f) ? 1u : 0u;
        }
        __syncthreads();
        if (sh_done != 0u) {
            break;
        }
    }
    if (sh_done == 0u) {
        if (lid == 0u) {
            st[p.status_idx] = 2;
        }
        return;
    }
    for (uint32_t i = lid; i < l; i += blockDim.x) {
        float d = s[p.off_G + i * l + i];
        sh_keys[i] = sqrtf(fmaxf(d, 0.0f));
        sh_idx[i] = i;
    }
    __syncthreads();
    for (uint32_t ph = 0u; ph < l; ++ph) {
        uint32_t start = ph & 1u;
        for (uint32_t i = start + lid * 2u; i + 1u < l; i += blockDim.x * 2u) {
            float a = sh_keys[i], b = sh_keys[i + 1u];
            uint32_t ia = sh_idx[i], ib = sh_idx[i + 1u];
            if (b > a || (b == a && ib < ia)) {
                sh_keys[i] = b; sh_keys[i + 1u] = a;
                sh_idx[i] = ib; sh_idx[i + 1u] = ia;
            }
        }
        __syncthreads();
    }
    for (uint32_t i = lid; i < l; i += blockDim.x) {
        s[p.off_S + i] = sh_keys[i];
        uint32_t idx = sh_idx[i];
        s[p.off_ORD + i] = *reinterpret_cast<float *>(&idx);
    }
}

// ---- step 8: FACTORS Af / BTf ----
static __global__ void k_xkv_factor_factors(
        float             * __restrict__ s,
        const int32_t     * __restrict__ st,
        cuda_xkv_factor_push p) {
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = p.n, m = p.m, r = p.r, l = p.l;
    uint32_t total = (n + m) * r;
    if (gid >= total) {
        return;
    }
    float smax = s[p.off_S];
    float guard = smax * 1e-12f;
    if (gid < n * r) {
        uint32_t i = gid / r, k = gid % r;
        uint32_t src = *reinterpret_cast<const uint32_t *>(&s[p.off_ORD + k]);
        float acc = 0.0f;
        for (uint32_t j = 0u; j < l; ++j) {
            acc += s[p.off_Y + i * l + j] * s[p.off_V + j * l + src];
        }
        s[p.off_AF + gid] = acc;
    } else {
        uint32_t g2 = gid - n * r;
        uint32_t j = g2 / r, k = g2 % r;
        uint32_t src = *reinterpret_cast<const uint32_t *>(&s[p.off_ORD + k]);
        float sv = s[p.off_S + k];
        float acc2 = 0.0f;
        for (uint32_t t = 0u; t < l; ++t) {
            acc2 += s[p.off_C + t * m + j] * s[p.off_V + t * l + src];
        }
        s[p.off_BTF + g2] = (sv > guard && sv > 0.0f) ? acc2 / sv : 0.0f;
    }
}

// ---- step 9: SIGN + BALANCE (one thread per column; grid covers r) ----
static __global__ void k_xkv_factor_sign_balance(
        float         * __restrict__ s,
        int32_t       * __restrict__ st,
        cuda_xkv_factor_push p) {
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= p.r) {
        return;
    }
    uint32_t n = p.n, m = p.m, r = p.r;
    float best = 0.0f;
    bool neg = false;
    for (uint32_t i = 0u; i < n; ++i) {
        float v = s[p.off_AF + i * r + k];
        if (fabsf(v) > fabsf(best)) {
            best = v; neg = (v < 0.0f);
        }
    }
    float sg = neg ? -1.0f : 1.0f;
    float sv = s[p.off_S + k];
    if (p.balance == 0u) {
        for (uint32_t i = 0u; i < n; ++i) {
            s[p.off_AF + i * r + k] *= sg * sv;
        }
        if (sg < 0.0f) {
            for (uint32_t j = 0u; j < m; ++j) {
                s[p.off_BTF + j * r + k] = -s[p.off_BTF + j * r + k];
            }
        }
    } else {
        float sq = sqrtf(fmaxf(sv, 0.0f));
        for (uint32_t i = 0u; i < n; ++i) {
            s[p.off_AF + i * r + k] *= sg * sq;
        }
        for (uint32_t j = 0u; j < m; ++j) {
            s[p.off_BTF + j * r + k] *= sg * sq;
        }
        if (p.balance == 2u) {
            float na = 0.0f, nb = 0.0f;
            for (uint32_t i = 0u; i < n; ++i) {
                float v = s[p.off_AF + i * r + k]; na += v * v;
            }
            for (uint32_t j = 0u; j < m; ++j) {
                float v = s[p.off_BTF + j * r + k]; nb += v * v;
            }
            na = sqrtf(na); nb = sqrtf(nb);
            if (na > 1e-12f && nb > 1e-12f) {
                float d = sqrtf(nb / na);
                for (uint32_t i = 0u; i < n; ++i) {
                    s[p.off_AF + i * r + k] *= d;
                }
                for (uint32_t j = 0u; j < m; ++j) {
                    s[p.off_BTF + j * r + k] /= d;
                }
            }
        }
    }
}

// ---- steps 10/11: contiguous row encode (one thread per row) ----
static __global__ void k_xkv_factor_encode(
        float             * __restrict__ s,
        uint8_t           * __restrict__ oa,
        uint8_t           * __restrict__ ob,
        const int32_t     * __restrict__ st,
        cuda_xkv_factor_push p,
        uint32_t isA) {
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rows = isA ? p.n : p.m;
    if (gid >= rows) {
        return;
    }
    uint32_t pr = isA ? p.pad_a : p.pad_b;
    uint32_t tcode = isA ? p.type_a : p.type_b;
    uint32_t foff = isA ? p.off_AF : p.off_BTF;
    uint32_t r = p.r;
    uint8_t * out = isA ? oa : ob;
    auto store_half = [out](uint32_t off, float v) {
        uint16_t h = __half_as_ushort(__float2half(v));
        out[off]     = (uint8_t)(h & 255u);
        out[off + 1] = (uint8_t)((h >> 8u) & 255u);
    };
    if (tcode == GGML_TYPE_F32) {
        for (uint32_t j = 0u; j < pr; ++j) {
            float v = (j < r) ? s[foff + gid * r + j] : 0.0f;
            uint32_t w = *reinterpret_cast<uint32_t *>(&v);
            uint32_t base = gid * pr * 4u + j * 4u;
            out[base]     = (uint8_t)(w & 255u);
            out[base + 1] = (uint8_t)((w >> 8u) & 255u);
            out[base + 2] = (uint8_t)((w >> 16u) & 255u);
            out[base + 3] = (uint8_t)(w >> 24u);
        }
        return;
    }
    if (tcode == GGML_TYPE_F16) {
        for (uint32_t j = 0u; j < pr; ++j) {
            float v = (j < r) ? s[foff + gid * r + j] : 0.0f;
            store_half(gid * pr * 2u + j * 2u, v);
        }
        return;
    }
    if (tcode == GGML_TYPE_Q8_0) {
        uint32_t nb = pr / 32u;
        for (uint32_t b = 0u; b < nb; ++b) {
            float amax = 0.0f;
            for (uint32_t j = 0u; j < 32u; ++j) {
                uint32_t c = b * 32u + j;
                float v = (c < r) ? s[foff + gid * r + c] : 0.0f;
                amax = fmaxf(amax, fabsf(v));
            }
            float d = amax / 127.0f;
            float id = (d != 0.0f) ? 1.0f / d : 0.0f;
            uint32_t base = gid * (nb * 34u);
            store_half(base + b * 34u, d);
            for (uint32_t j = 0u; j < 32u; ++j) {
                uint32_t c = b * 32u + j;
                float v = (c < r) ? s[foff + gid * r + c] : 0.0f;
                int q = (int)xkv_rhalf_away_cuda(v * id);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                out[base + b * 34u + 2u + j] = (uint8_t)((uint32_t)q & 255u);
            }
        }
        return;
    }
    // Turbo2/3/4: per-128-group L2 norm -> normalize -> fwd WHT ->
    // nearest centroid -> corrected norm.
    bool isT2 = (tcode == GGML_TYPE_TURBO2_0);
    bool isT3 = (tcode == GGML_TYPE_TURBO3_0);
    uint32_t ng = pr / 128u;
    uint32_t grp_bytes = isT2 ? 34u : (isT3 ? 50u : 68u);
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (uint32_t g = 0u; g < ng; ++g) {
        float buf[128];
        float nrm2 = 0.0f;
        for (uint32_t j = 0u; j < 128u; ++j) {
            uint32_t c = g * 128u + j;
            float v = (c < r) ? s[foff + gid * r + c] : 0.0f;
            buf[j] = v;
            nrm2 += v * v;
        }
        float gn = sqrtf(nrm2);
        float inv = (gn > 1e-10f) ? 1.0f / gn : 0.0f;
        for (uint32_t j = 0u; j < 128u; ++j) {
            buf[j] *= inv;
        }
        for (uint32_t j = 0u; j < 128u; ++j) {
            buf[j] *= TURBO_WHT_SIGNS1[j];
        }
        for (uint32_t h = 1u; h < 128u; h *= 2u) {
            for (uint32_t i = 0u; i < 128u; i += h * 2u) {
                for (uint32_t j = i; j < i + h; ++j) {
                    float a = buf[j], bb = buf[j + h];
                    buf[j] = a + bb; buf[j + h] = a - bb;
                }
            }
        }
        for (uint32_t j = 0u; j < 128u; ++j) {
            buf[j] *= inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
        }
        float rsq = 0.0f;
        uint32_t cidx[128];
        for (uint32_t j = 0u; j < 128u; ++j) {
            float v = buf[j];
            uint32_t ix = 0u;
            if (isT2) {
                ix = v < TURBO_MID_2BIT[0] ? 0u : (v < TURBO_MID_2BIT[1] ? 1u : (v < TURBO_MID_2BIT[2] ? 2u : 3u));
                rsq += TURBO_CENTROIDS_2BIT[ix] * TURBO_CENTROIDS_2BIT[ix];
            } else if (isT3) {
                ix = v < TURBO_MID_3BIT[0] ? 0u : (v < TURBO_MID_3BIT[1] ? 1u : (v < TURBO_MID_3BIT[2] ? 2u :
                     (v < TURBO_MID_3BIT[3] ? 3u : (v < TURBO_MID_3BIT[4] ? 4u : (v < TURBO_MID_3BIT[5] ? 5u :
                     (v < TURBO_MID_3BIT[6] ? 6u : 7u))))));
                rsq += TURBO_CENTROIDS_3BIT[ix] * TURBO_CENTROIDS_3BIT[ix];
            } else {
                ix = turbo_nearest_centroid_4bit(v);
                rsq += TURBO_CENTROIDS_4BIT[ix] * TURBO_CENTROIDS_4BIT[ix];
            }
            cidx[j] = ix;
        }
        float rn = sqrtf(rsq);
        float corr = (rn > 1e-10f) ? gn / rn : gn;
        uint32_t base = gid * (ng * grp_bytes) + g * grp_bytes;
        store_half(base, corr);
        if (isT2) {
            for (uint32_t j = 0u; j < 32u; ++j) {
                uint32_t qacc = 0u;
                for (uint32_t k = 0u; k < 4u; ++k) {
                    qacc |= ((cidx[j * 4u + k] & 3u) << (k * 2u));
                }
                out[base + 2u + j] = (uint8_t)(qacc & 255u);
            }
        } else if (isT3) {
            for (uint32_t j = 0u; j < 32u; ++j) {
                uint32_t qacc = 0u;
                for (uint32_t k = 0u; k < 4u; ++k) {
                    qacc |= ((cidx[j * 4u + k] & 3u) << (k * 2u));
                }
                out[base + 2u + j] = (uint8_t)(qacc & 255u);
            }
            for (uint32_t j = 0u; j < 16u; ++j) {
                uint32_t sacc = 0u;
                for (uint32_t k = 0u; k < 8u; ++k) {
                    if (((cidx[j * 8u + k]) & 4u) != 0u) {
                        sacc |= (1u << k);
                    }
                }
                out[base + 34u + j] = (uint8_t)(sacc & 255u);
            }
        } else {
            store_half(base + 2u, 0.0f);
            for (uint32_t j = 0u; j < 64u; ++j) {
                uint32_t lo = cidx[j * 2u] & 15u, hi = cidx[j * 2u + 1u] & 15u;
                out[base + 4u + j] = (uint8_t)((lo | (hi << 4u)) & 255u);
            }
        }
    }
}

// ---- step 12: EXACT all-row tiled final-codec residual band partials ----
static __global__ void k_xkv_factor_resid_bands(
        const float   * __restrict__ x,
        float         * __restrict__ s,
        const uint8_t * __restrict__ oa,
        const uint8_t * __restrict__ ob,
        int32_t       * __restrict__ st,
        cuda_xkv_factor_push p) {
    __shared__ float sh_bc;
    __shared__ float sh_C[256];
    __shared__ float sh_keys[512];
    uint32_t lid = threadIdx.x;
    uint32_t G = p.aux;
    uint32_t pr = p.pad_a; // pad_a == pad_b enforced in step 0
    uint32_t off_RES = p.off_BTF + p.m * p.r;
    if (gridDim.x != G || G < 1u || G > 64u) {
        if (lid == 0u) {
            st[p.status_idx] = 1;
        }
        return;
    }
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t g = blockIdx.x;
    uint32_t n = p.n, m = p.m, r = p.r;
    float ad[256];
    float bd[256];
    float sum_o = 0.0f, co = 0.0f;
    float sum_ab = 0.0f, cab = 0.0f;
    float sum_a = 0.0f, ca = 0.0f;
    float sum_b = 0.0f, cb = 0.0f;
    float mx_ab = 0.0f, mx_a = 0.0f, mx_b = 0.0f;
    uint32_t bad = 0u;
    uint32_t stride = G * 256u;
    for (uint32_t i = g * 256u + lid; i < n; i += stride) {
        for (uint32_t j = 0u; j < m; ++j) {
            float o = x[i * m + j];
            if (isnan(o) || isinf(o)) {
                bad = 1u; continue;
            }
            float dab = 0.0f, da = 0.0f, db = 0.0f;
            for (uint32_t c0 = 0u; c0 < pr; c0 += 256u) {
                uint32_t K = pr - c0 < 256u ? pr - c0 : 256u;
                for (uint32_t t = 0u; t < K; ++t) {
                    uint32_t k = c0 + t;
                    ad[t] = xkv_decA_cuda(oa, i, k, p.type_a, p.pad_a);
                    bd[t] = xkv_decB_cuda(ob, j, k, p.type_b, p.pad_b);
                }
                for (uint32_t t = 0u; t < K; ++t) {
                    uint32_t k = c0 + t;
                    float afp = (k < r) ? s[p.off_AF + i * r + k] : 0.0f;
                    float bfp = (k < r) ? s[p.off_BTF + j * r + k] : 0.0f;
                    float av = ad[t], bv = bd[t];
                    if (isnan(av) || isinf(av) || isnan(bv) || isinf(bv) ||
                        isnan(afp) || isinf(afp) || isnan(bfp) || isinf(bfp)) {
                        bad = 1u;
                    }
                    dab += av * bv;
                    da += av * bfp;
                    db += afp * bv;
                }
            }
            float y0 = o * o - co; float t0 = sum_o + y0; co = (t0 - sum_o) - y0; sum_o = t0;
            float eab = o - dab;
            float y1 = eab * eab - cab; float t1 = sum_ab + y1; cab = (t1 - sum_ab) - y1; sum_ab = t1;
            float ea = o - da;
            float y2 = ea * ea - ca; float t2 = sum_a + y2; ca = (t2 - sum_a) - y2; sum_a = t2;
            float eb = o - db;
            float y3 = eb * eb - cb; float t3 = sum_b + y3; cb = (t3 - sum_b) - y3; sum_b = t3;
            mx_ab = fmaxf(mx_ab, fabsf(eab));
            mx_a = fmaxf(mx_a, fabsf(ea));
            mx_b = fmaxf(mx_b, fabsf(eb));
        }
    }
    sh_keys[lid] = sum_o; sh_keys[256u + lid] = sum_ab;
    __syncthreads();
    if (lid == 0u) {
        float ao = 0.0f, c0 = 0.0f, ab = 0.0f, c1 = 0.0f;
        for (uint32_t t = 0u; t < 256u; ++t) {
            float y = sh_keys[t] - c0; float tt = ao + y; c0 = (tt - ao) - y; ao = tt;
            y = sh_keys[256u + t] - c1; tt = ab + y; c1 = (tt - ab) - y; ab = tt;
        }
        sh_bc = ao;
        sh_C[0] = ab;
    }
    __syncthreads();
    float band_o = sh_bc;
    float band_ab = sh_C[0];
    sh_keys[lid] = sum_a; sh_keys[256u + lid] = sum_b;
    __syncthreads();
    if (lid == 0u) {
        float a = 0.0f, c0 = 0.0f, b = 0.0f, c1 = 0.0f;
        for (uint32_t t = 0u; t < 256u; ++t) {
            float y = sh_keys[t] - c0; float tt = a + y; c0 = (tt - a) - y; a = tt;
            y = sh_keys[256u + t] - c1; tt = b + y; c1 = (tt - b) - y; b = tt;
        }
        sh_bc = a;
        sh_C[0] = b;
    }
    __syncthreads();
    float band_a = sh_bc;
    float band_b = sh_C[0];
    sh_keys[lid] = mx_ab;
    __syncthreads();
    if (lid == 0u) {
        float mm = 0.0f;
        for (uint32_t t = 0u; t < 256u; ++t) {
            mm = fmaxf(mm, sh_keys[t]);
        }
        sh_bc = mm;
    }
    __syncthreads();
    float band_mab = sh_bc;
    sh_keys[lid] = mx_a;
    __syncthreads();
    if (lid == 0u) {
        float mm = 0.0f;
        for (uint32_t t = 0u; t < 256u; ++t) {
            mm = fmaxf(mm, sh_keys[t]);
        }
        sh_bc = mm;
    }
    __syncthreads();
    float band_ma = sh_bc;
    sh_keys[lid] = mx_b;
    __syncthreads();
    if (lid == 0u) {
        float mm = 0.0f;
        for (uint32_t t = 0u; t < 256u; ++t) {
            mm = fmaxf(mm, sh_keys[t]);
        }
        sh_bc = mm;
    }
    __syncthreads();
    float band_mb = sh_bc;
    if (bad != 0u || isnan(sum_o + sum_ab + sum_a + sum_b) || isinf(sum_o + sum_ab + sum_a + sum_b)) {
        bad = 1u;
    }
    __syncthreads();
    if (lid == 0u) {
        uint32_t base = off_RES + g * 10u;
        s[base + 0u] = band_o;
        s[base + 1u] = band_ab;
        s[base + 2u] = band_a;
        s[base + 3u] = band_b;
        s[base + 4u] = band_mab;
        s[base + 5u] = band_ma;
        s[base + 6u] = band_mb;
        s[base + 7u] = 0.0f;
        s[base + 8u] = 0.0f;
        s[base + 9u] = (bad != 0u) ? 1.0f : 0.0f;
    }
}

// ---- step 13: fold G band partials -> status[1..9] ----
static __global__ void k_xkv_factor_resid_fold(
        float         * __restrict__ s,
        int32_t       * __restrict__ st,
        cuda_xkv_factor_push p) {
    uint32_t lid = threadIdx.x;
    if (gridDim.x != 1u) {
        if (lid == 0u) {
            st[p.status_idx] = 1;
        }
        return;
    }
    if (st[p.status_idx] != 0) {
        return;
    }
    uint32_t G = p.aux;
    if (G < 1u || G > 64u) {
        if (lid == 0u) {
            st[p.status_idx] = 1;
        }
        return;
    }
    if (lid != 0u) {
        return;
    }
    uint32_t off_RES = p.off_BTF + p.m * p.r;
    float so = 0.0f, co = 0.0f;
    float sab = 0.0f, cab = 0.0f, sa = 0.0f, caa = 0.0f, sb = 0.0f, cbb = 0.0f;
    float mab = 0.0f, ma = 0.0f, mb = 0.0f;
    uint32_t anybad = 0u;
    for (uint32_t g = 0u; g < G; ++g) {
        uint32_t base = off_RES + g * 10u;
        float y;
        y = s[base + 0u] - co;  float t = so + y;  co = (t - so) - y;  so = t;
        y = s[base + 1u] - cab; t = sab + y; cab = (t - sab) - y; sab = t;
        y = s[base + 2u] - caa; t = sa + y;  caa = (t - sa) - y;  sa = t;
        y = s[base + 3u] - cbb; t = sb + y;  cbb = (t - sb) - y;  sb = t;
        mab = fmaxf(mab, s[base + 4u]);
        ma = fmaxf(ma, s[base + 5u]);
        mb = fmaxf(mb, s[base + 6u]);
        if (s[base + 9u] > 0.0f) {
            anybad = 1u;
        }
        if (isnan(so + sab + sa + sb) || isinf(so + sab + sa + sb)) {
            anybad = 1u;
        }
    }
    if (anybad != 0u) {
        st[p.status_idx] = 4;
    } else {
        uint32_t * stu = reinterpret_cast<uint32_t *>(st);
        stu[p.status_idx + 1u] = *reinterpret_cast<uint32_t *>(&(so = sqrtf(so)));
        float v_ab = sqrtf(sab), v_a = sqrtf(sa), v_b = sqrtf(sb);
        stu[p.status_idx + 2u] = *reinterpret_cast<uint32_t *>(&v_ab);
        stu[p.status_idx + 3u] = *reinterpret_cast<uint32_t *>(&mab);
        stu[p.status_idx + 4u] = *reinterpret_cast<uint32_t *>(&so);
        stu[p.status_idx + 5u] = *reinterpret_cast<uint32_t *>(&v_a);
        stu[p.status_idx + 6u] = *reinterpret_cast<uint32_t *>(&ma);
        stu[p.status_idx + 7u] = *reinterpret_cast<uint32_t *>(&so);
        stu[p.status_idx + 8u] = *reinterpret_cast<uint32_t *>(&v_b);
        stu[p.status_idx + 9u] = *reinterpret_cast<uint32_t *>(&mb);
    }
}

bool ggml_cuda_xkv_factorize_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_FACTORIZE) {
        return false;
    }
    if (!op->src[0] || !op->src[1] || !op->src[2] || !op->src[3]) {
        return false;
    }
    ggml_xkv_factorize_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    return ggml_xkv_factorize_supports(
        op->src[0], op->src[1], op->src[2], op->src[3], op, &p, err, sizeof(err));
}

void ggml_cuda_xkv_factorize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * scratch = dst->src[1];
    const ggml_tensor * b_out   = dst->src[2];
    const ggml_tensor * status  = dst->src[3];

    ggml_xkv_factorize_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    uint32_t n = p.rows_n, m = p.cols_m;
    uint32_t r = p.requested_rank;
    uint32_t mn = n < m ? n : m;
    if (r > mn) {
        r = mn;
    }
    uint32_t l = r + p.oversampling;
    if (l > mn) {
        l = mn;
    }

    cuda_xkv_factor_push pc = {};
    pc.n = n; pc.m = m; pc.r = r; pc.l = l;
    pc.seed_lo = p.seed_low; pc.seed_hi = p.seed_high;
    pc.balance = p.balance_mode;
    pc.type_a = (uint32_t)dst->type; pc.type_b = (uint32_t)b_out->type;
    pc.pad_a = (uint32_t)dst->ne[0]; pc.pad_b = (uint32_t)b_out->ne[0];
    pc.off_Y = 0;
    pc.off_Z = pc.off_Y + n * l;
    pc.off_C = pc.off_Z + m * l;
    pc.off_G = pc.off_C + l * m;
    pc.off_V = pc.off_G + l * l;
    pc.off_S = pc.off_V + l * l;
    pc.off_ORD = pc.off_S + l;
    pc.off_AF = pc.off_ORD + l;
    pc.off_BTF = pc.off_AF + n * r;
    pc.status_idx = 0;
    pc.qr_rows = n;
    pc.aux = pc.off_Y;

    cudaStream_t stream = ctx.stream();
    const float * xd = (const float *)x->data;
    float * sd = (float *)scratch->data;
    uint8_t * oa = (uint8_t *)dst->data;
    uint8_t * ob = (uint8_t *)b_out->data;
    int32_t * st = (int32_t *)status->data;

    constexpr int kBlock = 256;
    auto grid = [](uint64_t total) {
        return (int)((total + kBlock - 1) / kBlock);
    };

    // step 0: validate
    k_xkv_factor_validate<<<1, 1, 0, stream>>>(xd, sd, st, pc);

    // step 1: Y = X*Omega fused
    k_xkv_factor_proj_y<<<grid((uint64_t)n * l), kBlock, 0, stream>>>(xd, sd, st, pc);

    // cuBLAS for the matmul stages (row-major C[M,N] = A[M,K] * B[K,N]
    // mapped to column-major calls on the same storage)
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float one = 1.0f, zero = 0.0f;
    float * Y = sd + pc.off_Y;
    float * Z = sd + pc.off_Z;
    float * C = sd + pc.off_C;
    float * G = sd + pc.off_G;

    for (uint32_t it = 0; it <= p.power_iterations; ++it) {
        if (it > 0) {
            // step 3: Z[m,l] = X^T * Q (Q at off_Y).
            // Row-major Z = X^T*Q <-> col-major Z^T = Q^T_col * X_col^T.
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)l, (int)m, (int)n, &one, Y, (int)l, xd, (int)m, &zero, Z, (int)l));
            // QR(Z)
            pc.qr_rows = m; pc.aux = pc.off_Z;
            k_xkv_factor_cgs2<<<1, kBlock, 0, stream>>>(sd, st, pc);
            // step 4: Y[n,l] = X * Z
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)l, (int)n, (int)m, &one, Z, (int)l, xd, (int)m, &zero, Y, (int)l));
        }
        // QR(Y)
        pc.qr_rows = n; pc.aux = pc.off_Y;
        k_xkv_factor_cgs2<<<1, kBlock, 0, stream>>>(sd, st, pc);
    }

    // step 5: C[l,m] = Q^T * X
    // Row-major C = Q^T*X <-> col-major C^T = X_col * Q_col^T.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        (int)m, (int)l, (int)n, &one, xd, (int)m, Y, (int)l, &zero, C, (int)m));
    // step 6: G[l,l] = C * C^T (symmetric).
    // Row-major G = C*C^T <-> col-major G = C_col^T * C_col.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)l, (int)l, (int)m, &one, C, (int)m, C, (int)m, &zero, G, (int)l));

    // step 7: Jacobi + sort
    k_xkv_factor_jacobi<<<1, kBlock, 0, stream>>>(sd, st, pc);
    // step 8: factors
    k_xkv_factor_factors<<<grid((uint64_t)(n + m) * r), kBlock, 0, stream>>>(sd, st, pc);
    // step 9: sign + balance
    k_xkv_factor_sign_balance<<<grid(r), kBlock, 0, stream>>>(sd, st, pc);
    // steps 10/11: encode A/B
    k_xkv_factor_encode<<<grid(n), kBlock, 0, stream>>>(sd, oa, ob, st, pc, 1u);
    k_xkv_factor_encode<<<grid(m), kBlock, 0, stream>>>(sd, oa, ob, st, pc, 0u);
    // steps 12/13: exact all-row tiled final-codec residual
    {
        uint32_t Gc = (n + 255u) / 256u;
        if (Gc > 64u) {
            Gc = 64u;
        }
        if (Gc == 0u) {
            Gc = 1u;
        }
        pc.aux = Gc;
        k_xkv_factor_resid_bands<<<Gc, kBlock, 0, stream>>>(xd, sd, oa, ob, st, pc);
        k_xkv_factor_resid_fold<<<1, kBlock, 0, stream>>>(sd, st, pc);
    }
    CUDA_CHECK(cudaGetLastError());
}
