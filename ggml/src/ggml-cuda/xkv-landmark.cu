// xkv-landmark.cu — Native on-device CUDA landmark scoring, deterministic top-k selection,
// fragment->row CSR expansion, and multi-segment merge.
//
// Ported from Vulkan compute shaders (xkv_landmark_score.comp, xkv_landmark_select.comp,
// xkv_landmark_rows.comp, xkv_landmark_merge.comp) and CPU oracle (ggml-vulkan-landmark.cpp).
//
// Supports:
//   GGML_OP_XKV_LANDMARK
//   GGML_OP_XKV_LANDMARK_ROWS
//   GGML_OP_XKV_LANDMARK_MERGE

#include "xkv-landmark.cuh"
#include "turbo-quant.cuh"
#include "ggml-vulkan-landmark.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

struct cuda_op_xkv_landmark_push {
    uint32_t n_queries, n_frags, head_dim, padded_dim;
    uint32_t rotary_dim, rope_mode, landmark_type, top_k;
    uint32_t max_top_k, refine_cap, frag_size;
    float    scale;
    uint32_t n_q_heads, n_rows_total, landmark_row_stride_bytes, status_flag;
    uint32_t fstride, qstride, flags;
};

// Constant tables matching xkv_landmark_score.comp
static __constant__ float CUDA_XC4[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

static __constant__ float CUDA_XS1[128] = {
    -1.f, 1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, -1.f, 1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f, -1.f, 1.f, 1.f, -1.f, -1.f, -1.f,
    -1.f, 1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f,
     1.f, 1.f, 1.f, -1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, 1.f, 1.f, 1.f, 1.f, -1.f, 1.f,
    -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, 1.f, 1.f,
     1.f, -1.f, -1.f, 1.f, 1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f,
    -1.f, 1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, 1.f, 1.f, 1.f, -1.f, 1.f, -1.f, 1.f,
     1.f, -1.f, 1.f, 1.f, -1.f, -1.f, -1.f, -1.f, -1.f, 1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f
};

static __constant__ float CUDA_XS2[128] = {
     1.f, 1.f, 1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f,
     1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, 1.f, 1.f, 1.f, -1.f,
     1.f, -1.f, 1.f, 1.f, 1.f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, 1.f, 1.f,
     1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f,
    -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, -1.f,
     1.f, -1.f, 1.f, 1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f,
    -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f
};

static __device__ __forceinline__ float cuda_unpack_f16(uint16_t h) {
    __half val = *reinterpret_cast<const __half *>(&h);
    return __half2float(val);
}

// ------------------------------------------------------------------------------------------------
// KERNEL: k_xkv_landmark_score
// 1 block of 64 threads per query
// ------------------------------------------------------------------------------------------------
static __global__ void k_xkv_landmark_score(
        const float   * __restrict__ q_data,
        const uint8_t * __restrict__ land,
        const int32_t * __restrict__ frag_pos,
        const int32_t * __restrict__ frag_meta,
        const int32_t * __restrict__ query_meta,
        const float   * __restrict__ rope,
        uint32_t      * __restrict__ scratch,
        int32_t       * __restrict__ status,
        cuda_op_xkv_landmark_push p) {
    uint32_t qq = blockIdx.x;
    uint32_t lid = threadIdx.x;
    if (qq >= p.n_queries) return;

    uint32_t nq = p.n_queries;
    uint32_t tk = p.top_k;
    uint32_t ci  = 0u;
    uint32_t cs  = nq * tk;
    uint32_t tak = cs + nq * tk;
    uint32_t ref = tak + nq;
    uint32_t leg = ref + nq;
    uint32_t my_ci = ci + qq * tk;
    uint32_t my_cs = cs + qq * tk;

    // Fail-closed cap/param checks
    if (tk == 0u || tk > p.max_top_k || tk > p.n_frags || tk > 128u) {
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 1; }
        return;
    }
    if (p.landmark_type != 8u && p.landmark_type != 44u) {
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 4; }
        return;
    }
    if (p.head_dim == 0u || p.head_dim > 256u || p.padded_dim < p.head_dim || p.padded_dim > 256u) {
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 4; }
        return;
    }
    if (p.landmark_type == 44u && (p.padded_dim % 128u) != 0u) {
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 4; }
        return;
    }
    if (p.n_q_heads == 0u || p.n_q_heads > 32u) {
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 2; }
        return;
    }
    if (p.fstride != 4u && p.fstride != 6u && p.fstride != 8u) {
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 2; }
        return;
    }
    if (p.qstride != 4u && p.qstride != 6u) {
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 2; }
        return;
    }
    if ((p.flags & 4u) != 0u) {
        // Canonical experimental mode is host-oracle-only; device rejects
        if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 4; }
        return;
    }

    // Strided landmark rows: per-head views into one sealed full-group tensor
    {
        uint32_t enc_bytes = (p.landmark_type == 8u) ? (p.padded_dim / 32u) * 34u : (p.padded_dim / 128u) * 68u;
        uint32_t blk_bytes = (p.landmark_type == 8u) ? 34u : 68u;
        if (p.landmark_row_stride_bytes < enc_bytes ||
            (p.landmark_row_stride_bytes % blk_bytes) != 0u) {
            if (lid == 0u) { scratch[tak + qq] = 0u; scratch[leg + qq] = 0u; status[0] = 4; }
            return;
        }
    }

    int32_t causal = query_meta[qq * p.qstride + 0u];
    int32_t vis_group = (p.qstride == 6u) ? query_meta[qq * 6u + 4u] : -1;
    uint32_t hd = p.head_dim;
    uint32_t pd = p.padded_dim;

    __shared__ float s_land[256];
    __shared__ float s_red[64];
    __shared__ float s_head_best;
    __shared__ uint32_t s_is_legal;

    uint32_t ncarry = 0u;
    uint32_t nlegal = 0u;
    bool failed = false;

    for (uint32_t f = 0u; f < p.n_frags && !failed; ++f) {
        if (lid == 0u) {
            int32_t fl = frag_meta[f * p.fstride + 3u];
            int32_t spos = (p.fstride == 8u) ? frag_meta[f * 8u + 6u] : frag_pos[f];
            int32_t fgrp = (p.fstride >= 6u) ? frag_meta[f * p.fstride + 5u] : -1;

            bool ok = ((fl & 1) != 0) && (causal < 0 || spos <= causal);

            // v3: per-query eligibility bitset
            if (ok && p.fstride == 8u) {
                uint32_t words_per_q = (p.n_frags + 31u) / 32u;
                uint32_t w = (uint32_t)frag_pos[qq * words_per_q + f / 32u];
                if ((w & (1u << (f % 32u))) == 0u) ok = false;
            }

            // Real DDVR group matching
            if (ok && vis_group >= 0 && fgrp >= 0) {
                if (fgrp != vis_group) ok = false;
            }

            s_is_legal = ok ? 1u : 0u;
        }
        __syncthreads();
        if (s_is_legal == 0u) continue;
        if (lid == 0u) nlegal++;

        // Cooperative 64-lane decode from the strided full-group row
        uint32_t bb = f * p.landmark_row_stride_bytes;
        if (p.landmark_type == 8u) {
            // Q8_0: 34B per 32 elements
            for (uint32_t d = lid; d < pd; d += 64u) {
                uint32_t b = d / 32u;
                uint32_t j = d % 32u;
                uint32_t boff = bb + b * 34u;
                uint16_t h = (uint16_t)land[boff] | ((uint16_t)land[boff + 1u] << 8u);
                float dd = cuda_unpack_f16(h);
                int32_t qv = (int32_t)(int8_t)land[boff + 2u + j];
                s_land[d] = dd * (float)qv;
            }
        } else {
            // Turbo4_0
            uint32_t ngroups = pd / 128u;
            for (uint32_t g = 0u; g < ngroups; ++g) {
                uint32_t boff = bb + g * 68u;
                uint16_t h = (uint16_t)land[boff] | ((uint16_t)land[boff + 1u] << 8u);
                float norm = cuda_unpack_f16(h);
                for (uint32_t k = 0u; k < 2u; ++k) {
                    uint32_t j = lid * 2u + k;
                    uint32_t qb = (uint32_t)land[boff + 4u + j / 2u];
                    uint32_t idx = ((j & 1u) != 0u) ? ((qb >> 4u) & 15u) : (qb & 15u);
                    s_land[g * 128u + j] = norm * CUDA_XC4[idx] * CUDA_XS2[j];
                }
            }
            __syncthreads();
            for (uint32_t g = 0u; g < ngroups; ++g) {
                uint32_t goff = g * 128u;
                for (uint32_t hh = 1u; hh < 128u; hh *= 2u) {
                    uint32_t pair = (lid / hh) * (2u * hh) + (lid % hh);
                    float a = s_land[goff + pair];
                    float c = s_land[goff + pair + hh];
                    s_land[goff + pair]      = a + c;
                    s_land[goff + pair + hh] = a - c;
                    __syncthreads();
                }
                for (uint32_t k = 0u; k < 2u; ++k) {
                    uint32_t j = lid * 2u + k;
                    s_land[goff + j] *= (0.08838834764831845f * CUDA_XS1[j]);
                }
            }
        }
        __syncthreads();

        if (lid == 0u) s_head_best = -3.402823466e+38f;
        __syncthreads();

        // Cooperative dot product and tree reduction
        for (uint32_t qh = 0u; qh < p.n_q_heads; ++qh) {
            uint32_t qbase = (qq * p.n_q_heads + qh) * hd;
            float my_dot = 0.0f;
            for (uint32_t d = lid; d < hd; d += 64u) {
                my_dot += q_data[qbase + d] * s_land[d];
            }
            s_red[lid] = my_dot;
            __syncthreads();
            for (uint32_t s = 32u; s > 0u; s >>= 1u) {
                if (lid < s) s_red[lid] += s_red[lid + s];
                __syncthreads();
            }
            if (lid == 0u) {
                float sc = s_red[0] * p.scale;
                if (isnan(sc) || isinf(sc)) failed = true;
                if (sc > s_head_best) s_head_best = sc;
            }
            __syncthreads();
            if (failed) break;
        }

        // Lane 0: deterministic carry insertion
        if (lid == 0u && !failed) {
            float best = s_head_best;
            uint32_t pos = ncarry;
            for (uint32_t i = 0u; i < ncarry; ++i) {
                uint32_t ef = scratch[my_ci + i];
                float es = __uint_as_float(scratch[my_cs + i]);
                if (best > es || (best == es && f < ef)) { pos = i; break; }
            }
            if (pos < tk) {
                uint32_t up = ncarry < tk ? ncarry : tk - 1u;
                for (uint32_t i = up; i > pos; --i) {
                    scratch[my_ci + i] = scratch[my_ci + i - 1u];
                    scratch[my_cs + i] = scratch[my_cs + i - 1u];
                }
                scratch[my_ci + pos] = f;
                scratch[my_cs + pos] = __float_as_uint(best);
                if (ncarry < tk) ncarry++;
            }
        }
        __syncthreads();
    }

    if (lid == 0u) {
        if (failed) {
            scratch[tak + qq] = 0u;
            scratch[leg + qq] = 0u;
            status[0] = 2;
        } else {
            scratch[tak + qq] = ncarry;
            scratch[leg + qq] = nlegal;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// KERNEL: k_xkv_landmark_select_pass0 (Per-query emit & bounded refinement)
// ------------------------------------------------------------------------------------------------
static __global__ void k_xkv_landmark_select_pass0(
        const int32_t * __restrict__ frag_meta,
        int32_t       * __restrict__ csr_indices,
        float         * __restrict__ topk_scores,
        int32_t       * __restrict__ status,
        uint32_t      * __restrict__ scratch,
        cuda_op_xkv_landmark_push p) {
    uint32_t qq = blockIdx.x;
    uint32_t lid = threadIdx.x;
    if (qq >= p.n_queries || lid != 0u) return;

    uint32_t nq = p.n_queries;
    uint32_t tk = p.top_k;
    uint32_t cap = p.refine_cap;
    uint32_t ci  = 0u;
    uint32_t cs  = nq * tk;
    uint32_t tak = cs + nq * tk;
    uint32_t ref = tak + nq;
    uint32_t leg = ref + nq;
    uint32_t row = leg + nq;

    if (tk == 0u || tk > p.max_top_k || tk > p.n_frags) {
        scratch[tak + qq] = 0u;
        scratch[ref + qq] = 0u;
        status[0] = 1;
        return;
    }

    if (status[0] != 0) {
        for (uint32_t i = 0u; i < tk; ++i) {
            csr_indices[qq * tk + i] = -1;
            topk_scores[qq * tk + i] = -1e30f;
        }
        scratch[tak + qq] = 0u;
        scratch[ref + qq] = 0u;
        return;
    }

    uint32_t base_i = ci + qq * tk;
    uint32_t base_s = cs + qq * tk;
    uint32_t nsel = scratch[tak + qq];
    if (nsel > tk) nsel = tk;

    for (uint32_t i = 0u; i < tk; ++i) {
        if (i < nsel) {
            csr_indices[qq * tk + i] = (int32_t)scratch[base_i + i];
            topk_scores[qq * tk + i] = __uint_as_float(scratch[base_s + i]);
        } else {
            csr_indices[qq * tk + i] = -1;
            topk_scores[qq * tk + i] = -1e30f;
        }
    }

    // Bounded refinement
    uint32_t refined = 0u;
    uint32_t hit = 0u;
    uint32_t rowb = row + qq * cap;
    if (cap > 0u && nsel > 0u) {
        for (uint32_t i = 1u; i < nsel; ++i) {
            uint32_t k = scratch[base_i + i];
            uint32_t j = i;
            while (j > 0u && scratch[base_i + j - 1u] > k) {
                scratch[base_i + j] = scratch[base_i + j - 1u];
                --j;
            }
            scratch[base_i + j] = k;
        }
        uint32_t total = 0u;
        uint32_t uniq = 0u;
        for (uint32_t s = 0u; s < nsel; ++s) {
            uint32_t fidx = scratch[base_i + s];
            int32_t rb = frag_meta[fidx * p.fstride + 0u];
            int32_t rc = frag_meta[fidx * p.fstride + 1u];
            if (rc <= 0) rc = (int32_t)p.frag_size;
            for (int32_t r = 0; r < rc; ++r) {
                int32_t rr = rb + r;
                if (rr < 0 || (uint32_t)rr >= p.n_rows_total) continue;
                total++;
                if (uniq < cap) {
                    uint32_t urow = (uint32_t)rr;
                    uint32_t upos = uniq;
                    for (uint32_t i = 0u; i < uniq; ++i) {
                        uint32_t er = scratch[rowb + i];
                        if (er == urow) { upos = uniq + 1u; break; }
                        if (er > urow) { upos = i; break; }
                    }
                    if (upos <= uniq) {
                        if (upos < uniq) {
                            for (uint32_t i = uniq; i > upos; --i) scratch[rowb + i] = scratch[rowb + i - 1u];
                        }
                        scratch[rowb + upos] = urow;
                        uniq++;
                    }
                }
            }
        }
        if (total > cap) hit = 1u;
        refined = uniq < cap ? uniq : cap;
    }
    scratch[ref + qq] = refined | (hit << 31u);
}

// ------------------------------------------------------------------------------------------------
// KERNEL: k_xkv_landmark_select_pass1 (Prefix fold & status)
// ------------------------------------------------------------------------------------------------
static __global__ void k_xkv_landmark_select_pass1(
        int32_t  * __restrict__ csr_ptrs,
        int32_t  * __restrict__ status,
        uint32_t * __restrict__ scratch,
        cuda_op_xkv_landmark_push p) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;

    uint32_t nq = p.n_queries;
    uint32_t tk = p.top_k;
    uint32_t cs = nq * tk;
    uint32_t tak = cs + nq * tk;
    uint32_t ref = tak + nq;
    uint32_t leg = ref + nq;

    if (tk == 0u || tk > p.max_top_k || tk > p.n_frags) {
        status[0] = 1;
        status[1] = 0; status[2] = 0; status[3] = 0;
        return;
    }

    int32_t acc = 0;
    for (uint32_t q = 0u; q < nq; ++q) {
        csr_ptrs[q] = acc;
        acc += (int32_t)scratch[tak + q];
    }
    csr_ptrs[nq] = acc;

    uint32_t tot_ref = 0u;
    uint32_t tot_hit = 0u;
    uint32_t tot_leg = 0u;
    for (uint32_t q = 0u; q < nq; ++q) {
        uint32_t rw = scratch[ref + q];
        tot_ref += (rw & 0x7FFFFFFFu);
        tot_hit += (rw >> 31u);
        tot_leg += scratch[leg + q];
    }

    if (status[0] == 0) {
        status[0] = 0;
    }
    status[1] = (int32_t)tot_ref;
    status[2] = (int32_t)tot_hit;
    status[3] = (int32_t)tot_leg;
}

// ------------------------------------------------------------------------------------------------
// GGML_OP_XKV_LANDMARK Host Dispatch
// ------------------------------------------------------------------------------------------------
bool ggml_cuda_xkv_landmark_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_LANDMARK) return false;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!op->src[i]) return false;
    }
    ggml_xkv_landmark_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    if (!ggml_xkv_landmark_supports(op->src[0], op->src[1], op->src[2],
        op->src[3], op->src[4], op->src[5], op->src[6], op->src[7],
        op->src[8], op->src[9], op, &p, err, sizeof(err))) {
        return false;
    }
    if (p.landmark_type != (uint32_t)GGML_TYPE_Q8_0 &&
        p.landmark_type != (uint32_t)GGML_TYPE_TURBO4_0) {
        return false;
    }
    if (p.padded_dim == 0 || p.padded_dim > 256) return false;
    if (p.landmark_type == (uint32_t)GGML_TYPE_TURBO4_0 && p.padded_dim % 128u != 0) {
        return false;
    }
    if (p.refine_cap >= 0x80000000u) return false;
    return true;
}

void ggml_cuda_xkv_landmark(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * landmarks     = dst->src[1];
    const ggml_tensor * frag_positions = dst->src[2];
    const ggml_tensor * frag_meta     = dst->src[3];
    const ggml_tensor * query_meta    = dst->src[4];
    const ggml_tensor * rope_tables   = dst->src[5];
    const ggml_tensor * scratch       = dst->src[6];
    ggml_tensor       * csr_ptrs      = dst->src[7];
    ggml_tensor       * topk_scores   = dst->src[8];
    ggml_tensor       * status        = dst->src[9];

    ggml_xkv_landmark_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    cuda_op_xkv_landmark_push pc = {};
    pc.n_queries = p.n_queries; pc.n_frags = p.n_frags;
    pc.head_dim = p.head_dim; pc.padded_dim = p.padded_dim;
    pc.rotary_dim = p.rotary_dim; pc.rope_mode = p.rope_mode;
    pc.landmark_type = p.landmark_type; pc.top_k = p.top_k;
    pc.max_top_k = p.max_top_k; pc.refine_cap = p.refine_cap; pc.frag_size = p.frag_size;
    pc.scale = p.scale; pc.n_q_heads = p.n_q_heads; pc.n_rows_total = p.n_rows_total;
    pc.landmark_row_stride_bytes = (uint32_t)landmarks->nb[1];
    pc.status_flag = 0;
    pc.fstride = (uint32_t)frag_meta->ne[0];
    pc.qstride = (uint32_t)query_meta->ne[0];
    pc.flags = p._reserved;

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(status->data, 0, 4 * sizeof(int32_t), stream));

    k_xkv_landmark_score<<<p.n_queries, 64, 0, stream>>>(
        (const float   *)q->data,
        (const uint8_t *)landmarks->data,
        (const int32_t *)frag_positions->data,
        (const int32_t *)frag_meta->data,
        (const int32_t *)query_meta->data,
        (const float   *)rope_tables->data,
        (uint32_t      *)scratch->data,
        (int32_t       *)status->data,
        pc
    );

    k_xkv_landmark_select_pass0<<<p.n_queries, 1, 0, stream>>>(
        (const int32_t *)frag_meta->data,
        (int32_t       *)dst->data,
        (float         *)topk_scores->data,
        (int32_t       *)status->data,
        (uint32_t      *)scratch->data,
        pc
    );

    k_xkv_landmark_select_pass1<<<1, 1, 0, stream>>>(
        (int32_t  *)csr_ptrs->data,
        (int32_t  *)status->data,
        (uint32_t *)scratch->data,
        pc
    );
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// KERNEL: k_xkv_landmark_rows (Fragment->Row CSR Expansion)
// ------------------------------------------------------------------------------------------------
struct cuda_op_xkv_landmark_rows_push {
    uint32_t version;
    uint32_t n_queries;
    uint32_t top_k;
    uint32_t n_frags;
    uint32_t refine_cap;
    uint32_t fstride;
    uint32_t flags;
    uint32_t max_frag_rows;
    uint32_t n_rows_total;
    uint32_t n_parent_queries;
    uint32_t has_query_map;
    uint32_t arena_filter;
    uint32_t global_row_base;
    uint32_t arena_row_count;
    uint32_t output_row_begin;
    uint32_t status_flag;
};

static __device__ void cuda_insert_query_row(
        int32_t * s_row, int32_t * s_hed, int32_t * s_slot, int32_t * s_pos,
        uint32_t cap, int32_t lrow, int32_t hed, int32_t slot, int32_t gpos,
        uint32_t & uniq, bool & overflow) {
    uint32_t upos = uniq;
    for (uint32_t i = 0u; i < uniq; ++i) {
        int32_t er = s_row[i];
        int32_t es = s_slot[i];
        if (er == lrow && es == slot) return;
        if (er > lrow || (er == lrow && es > slot)) {
            upos = i;
            break;
        }
    }

    uint32_t old_uniq = uniq;
    if (uniq < cap) {
        ++uniq;
    } else {
        overflow = true;
        if (upos >= cap) return;
        old_uniq = cap - 1u;
    }
    for (uint32_t i = old_uniq; i > upos; --i) {
        s_row[i]  = s_row[i - 1u];
        s_hed[i]  = s_hed[i - 1u];
        s_slot[i] = s_slot[i - 1u];
        s_pos[i]  = s_pos[i - 1u];
    }
    s_row[upos]  = lrow;
    s_hed[upos]  = hed;
    s_slot[upos] = slot;
    s_pos[upos]  = gpos;
}

static __global__ void k_xkv_landmark_rows(
        const int32_t * __restrict__ sel_idx,
        const int32_t * __restrict__ frag_meta,
        const int32_t * __restrict__ frag_row_off,
        const int32_t * __restrict__ frag_row_ids,
        const int32_t * __restrict__ frag_kv,
        const int32_t * __restrict__ row_pos,
        uint32_t      * __restrict__ row_ptrs,
        int32_t       * __restrict__ row_refs,
        int32_t       * __restrict__ row_out_pos,
        int32_t       * __restrict__ row_entries,
        int32_t       * __restrict__ row_status,
        cuda_op_xkv_landmark_rows_push p) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    if (p.status_flag == 0u) return;

    uint32_t E = p.n_queries;
    uint32_t N = (p.n_parent_queries > 0u) ? p.n_parent_queries : E;
    uint32_t cap = p.refine_cap;

    bool filter_all = (p.arena_filter == 0xFFFFFFFFu);
    if (row_status[0] != 0 || p.top_k == 0u || p.top_k > 128u ||
        cap == 0u || cap > 1024u || E == 0u || N == 0u || N > E ||
        p.has_query_map > 1u ||
        (p.has_query_map == 0u && p.n_parent_queries != 0u) ||
        (p.has_query_map != 0u && p.n_parent_queries == 0u) ||
        (p.fstride != 4u && p.fstride != 6u) ||
        (filter_all && (p.global_row_base != 0u || p.arena_row_count != p.n_rows_total)) ||
        (!filter_all && (p.arena_row_count == 0u || p.arena_row_count > p.n_rows_total ||
                         p.global_row_base >= p.n_rows_total ||
                         p.global_row_base + p.arena_row_count > p.n_rows_total))) {
        row_status[0] = 1;
        row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
        return;
    }

    for (uint32_t n = 0u; n <= N; ++n) {
        row_ptrs[n] = 0u;
    }

    int32_t prev_parent = -1;
    int32_t prev_slot = -1;
    uint32_t sel_stride = p.top_k + (p.has_query_map != 0u ? 2u : 0u);

    if (p.has_query_map != 0u) {
        for (uint32_t e = 0u; e < E; ++e) {
            int32_t parent = sel_idx[e * sel_stride + 0u];
            int32_t slot = sel_idx[e * sel_stride + 1u];
            if (parent < 0 || (uint32_t)parent >= N || slot < 0 ||
                parent < prev_parent || (parent == prev_parent && slot <= prev_slot)) {
                row_status[0] = 2;
                row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                return;
            }
            prev_parent = parent;
            prev_slot = slot;
        }
    }

    uint32_t tile_cap = E * cap;
    if (tile_cap > 1024u) tile_cap = 1024u;

    uint32_t win_start = p.output_row_begin;
    uint32_t win_end = win_start + tile_cap;

    uint32_t running_total = 0u;
    uint32_t written = 0u;
    uint32_t total_hits = 0u;

    uint32_t sel[128];
    __shared__ int32_t s_row[1024];
    __shared__ int32_t s_hed[1024];
    __shared__ int32_t s_slot[1024];
    __shared__ int32_t s_pos[1024];

    for (uint32_t e = 0u; e < E; ++e) {
        int32_t parent = (p.has_query_map != 0u) ? sel_idx[e * sel_stride + 0u] : (int32_t)e;
        int32_t mapped_slot = (p.has_query_map != 0u) ? sel_idx[e * sel_stride + 1u] : 0;
        uint32_t sel_offset = (p.has_query_map != 0u) ? 2u : 0u;

        uint32_t nsel = 0u;
        for (uint32_t i = 0u; i < p.top_k; ++i) {
            int32_t f = sel_idx[e * sel_stride + sel_offset + i];
            if (f < 0) break;
            if ((uint32_t)f >= p.n_frags) {
                row_status[0] = 2;
                row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                return;
            }
            sel[nsel++] = (uint32_t)f;
        }

        for (uint32_t i = 1u; i < nsel; ++i) {
            uint32_t k = sel[i];
            uint32_t j = i;
            while (j > 0u && sel[j - 1u] > k) { sel[j] = sel[j - 1u]; --j; }
            sel[j] = k;
        }

        uint32_t uniq = 0u;
        bool overflow = false;

        for (uint32_t s = 0u; s < nsel; ++s) {
            uint32_t f = sel[s];
            uint32_t frag_arena = (uint32_t)frag_kv[f * 3u + 1u];
            if (!filter_all && frag_arena != p.arena_filter) continue;

            int32_t hed = frag_kv[f * 3u + 2u];
            int32_t slot_val = (p.has_query_map != 0u) ? mapped_slot :
                ((p.fstride >= 6u) ? frag_meta[f * p.fstride + 5u] : 0);
            if (slot_val < 0) {
                row_status[0] = 2;
                row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                return;
            }

            bool use_list = (p.flags & 1u) != 0u;
            int32_t list_off = 0, list_nxt = 0;
            int32_t rb = 0, rc = 0;

            if (use_list) {
                list_off = frag_row_off[f];
                list_nxt = frag_row_off[f + 1u];
                if (list_off >= 0 && list_nxt > list_off) {
                    if ((uint32_t)(list_nxt - list_off) > p.max_frag_rows) {
                        row_status[0] = 2;
                        row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                        return;
                    }
                } else {
                    use_list = false;
                    rb = frag_meta[f * p.fstride + 0u];
                    rc = frag_meta[f * p.fstride + 1u];
                    if (rc <= 0 || (uint32_t)rc > p.max_frag_rows) {
                        row_status[0] = 2;
                        row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                        return;
                    }
                }
            } else {
                rb = frag_meta[f * p.fstride + 0u];
                rc = frag_meta[f * p.fstride + 1u];
                if (rc <= 0 || (uint32_t)rc > p.max_frag_rows) {
                    row_status[0] = 2;
                    row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                    return;
                }
            }

            if (!use_list) {
                if (rb < 0 || ((uint32_t)rb + (uint32_t)rc) > p.n_rows_total) {
                    row_status[0] = 2;
                    row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                    return;
                }
                for (int32_t r = 0; r < rc; ++r) {
                    int32_t grow = rb + r;
                    if ((uint32_t)grow < p.global_row_base ||
                        (uint32_t)grow >= p.global_row_base + p.arena_row_count) {
                        row_status[0] = 2;
                        row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                        return;
                    }
                    int32_t lrow = (int32_t)((uint32_t)grow - p.global_row_base);
                    int32_t gpos = row_pos[(uint32_t)grow];
                    cuda_insert_query_row(s_row, s_hed, s_slot, s_pos, cap, lrow, hed, slot_val, gpos, uniq, overflow);
                }
            } else {
                for (int32_t k = list_off; k < list_nxt; ++k) {
                    int32_t grow = frag_row_ids[(uint32_t)k];
                    if (grow < 0) continue;
                    if ((uint32_t)grow >= p.n_rows_total) {
                        row_status[0] = 2;
                        row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                        return;
                    }
                    if ((uint32_t)grow < p.global_row_base ||
                        (uint32_t)grow >= p.global_row_base + p.arena_row_count) {
                        row_status[0] = 2;
                        row_status[1] = 0; row_status[2] = 0; row_status[3] = 0;
                        return;
                    }
                    int32_t lrow = (int32_t)((uint32_t)grow - p.global_row_base);
                    int32_t gpos = row_pos[(uint32_t)grow];
                    cuda_insert_query_row(s_row, s_hed, s_slot, s_pos, cap, lrow, hed, slot_val, gpos, uniq, overflow);
                }
            }
        }

        if (overflow) ++total_hits;

        uint32_t q_start = running_total;
        uint32_t q_end = running_total + uniq;
        running_total += uniq;

        uint32_t isect_start = q_start > win_start ? q_start : win_start;
        uint32_t isect_end = q_end < win_end ? q_end : win_end;

        if (isect_start < isect_end) {
            uint32_t take = isect_end - isect_start;
            uint32_t q_off = isect_start - q_start;
            for (uint32_t i = 0u; i < take; ++i) {
                uint32_t dst_idx = written + i;
                uint32_t src_idx = q_off + i;
                row_refs[dst_idx * 4u + 0u] = s_row[src_idx];
                row_refs[dst_idx * 4u + 1u] = 0;
                row_refs[dst_idx * 4u + 2u] = 0;
                row_refs[dst_idx * 4u + 3u] = s_hed[src_idx];
                row_out_pos[dst_idx] = s_pos[src_idx];
                row_entries[dst_idx * 4u + 0u] = 2; // GGML_XKV_ATTN_SOURCE_COLD
                row_entries[dst_idx * 4u + 1u] = (int32_t)dst_idx;
                row_entries[dst_idx * 4u + 2u] = s_slot[src_idx];
                row_entries[dst_idx * 4u + 3u] = 1;
            }
            row_ptrs[(uint32_t)parent + 1u] += take;
            written += take;
        }
    }

    for (uint32_t n = 0u; n < N; ++n) {
        row_ptrs[n + 1u] += row_ptrs[n];
    }

    for (uint32_t j = written; j < tile_cap; ++j) {
        row_refs[j * 4u + 0u] = 0;
        row_refs[j * 4u + 1u] = 0;
        row_refs[j * 4u + 2u] = 0;
        row_refs[j * 4u + 3u] = 0;
        row_out_pos[j] = 0;
        row_entries[j * 4u + 0u] = 2;
        row_entries[j * 4u + 1u] = (int32_t)j;
        row_entries[j * 4u + 2u] = 0;
        row_entries[j * 4u + 3u] = 0;
    }

    row_status[0] = 0;
    row_status[1] = (int32_t)written;
    row_status[2] = (int32_t)total_hits;
    row_status[3] = (int32_t)running_total;
}

bool ggml_cuda_xkv_landmark_rows_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_LANDMARK_ROWS) return false;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!op->src[i]) return false;
    }
    ggml_xkv_landmark_rows_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    return ggml_xkv_landmark_rows_supports(op->src[0], op->src[1], op->src[2],
        op->src[3], op->src[4], op->src[5], op->src[6], op, op->src[7],
        op->src[8], op->src[9], &p, err, sizeof(err));
}

void ggml_cuda_xkv_landmark_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel_idx     = dst->src[0];
    const ggml_tensor * frag_meta   = dst->src[1];
    const ggml_tensor * frag_row_off= dst->src[2];
    const ggml_tensor * frag_row_ids= dst->src[3];
    const ggml_tensor * frag_kv     = dst->src[4];
    const ggml_tensor * row_pos     = dst->src[5];
    ggml_tensor       * row_ptrs    = dst->src[6];
    ggml_tensor       * row_out_pos = dst->src[7];
    ggml_tensor       * row_entries = dst->src[8];
    ggml_tensor       * row_status  = dst->src[9];

    ggml_xkv_landmark_rows_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    cuda_op_xkv_landmark_rows_push pc;
    memcpy(&pc, &p, sizeof(p));
    pc.status_flag = 1;

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(row_status->data, 0, 4 * sizeof(int32_t), stream));

    k_xkv_landmark_rows<<<1, 1, 0, stream>>>(
        (const int32_t *)sel_idx->data,
        (const int32_t *)frag_meta->data,
        (const int32_t *)frag_row_off->data,
        (const int32_t *)frag_row_ids->data,
        (const int32_t *)frag_kv->data,
        (const int32_t *)row_pos->data,
        (uint32_t      *)row_ptrs->data,
        (int32_t       *)dst->data,
        (int32_t       *)row_out_pos->data,
        (int32_t       *)row_entries->data,
        (int32_t       *)row_status->data,
        pc
    );
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// KERNEL: k_xkv_landmark_merge (Multi-segment merge)
// ------------------------------------------------------------------------------------------------
static __global__ void k_xkv_landmark_merge(
        const int32_t  * __restrict__ set_idx,
        const float    * __restrict__ set_sc,
        const uint32_t * __restrict__ set_base,
        int32_t        * __restrict__ out_idx,
        float          * __restrict__ out_sc,
        int32_t        * __restrict__ status,
        ggml_xkv_landmark_merge_params p) {
    uint32_t qq = blockIdx.x;
    uint32_t lid = threadIdx.x;
    if (qq >= p.n_queries || lid != 0u) return;

    uint32_t ns = p.n_sets;
    uint32_t sc = p.set_cap;
    uint32_t tk = p.top_k;

    if (ns == 0u || ns > 64u || sc == 0u || sc > 128u || tk == 0u || tk > 128u) {
        status[0] = 1;
        status[1] = 0; status[2] = 0; status[3] = 0;
        return;
    }
    if ((uint64_t)tk > (uint64_t)ns * (uint64_t)sc) {
        status[0] = 1;
        status[1] = 0; status[2] = 0; status[3] = 0;
        return;
    }

    uint32_t took_id[128];
    float took_sc[128];
    uint32_t ntook = 0u;
    bool failed = false;

    for (uint32_t s = 0u; s < ns && !failed; ++s) {
        uint32_t base_offset = (p.has_set_base != 0u && set_base) ? set_base[s] : 0u;

        for (uint32_t i = 0u; i < sc; ++i) {
            uint32_t at = (s * p.n_queries + qq) * sc + i;
            int32_t local_id = set_idx[at];
            float v = set_sc[at];

            if (local_id < 0) continue;
            if (isnan(v)) { failed = true; break; }
            if (v == __uint_as_float(0xFF800000u)) continue;

            uint32_t global_id = base_offset + (uint32_t)local_id;

            bool dup = false;
            for (uint32_t t = 0u; t < ntook; ++t) {
                if (took_id[t] == global_id) { dup = true; break; }
            }
            if (dup) continue;

            uint32_t pos = ntook;
            for (uint32_t t = 0u; t < ntook; ++t) {
                if (v > took_sc[t] || (v == took_sc[t] && global_id < took_id[t])) {
                    pos = t; break;
                }
            }

            if (pos < tk) {
                uint32_t up = ntook < tk ? ntook : tk - 1u;
                for (uint32_t t = up; t > pos; --t) {
                    took_id[t] = took_id[t - 1u];
                    took_sc[t] = took_sc[t - 1u];
                }
                took_id[pos] = global_id;
                took_sc[pos] = v;
                if (ntook < tk) ntook++;
            }
        }
    }

    if (failed) {
        for (uint32_t i = 0u; i < tk; ++i) {
            out_idx[qq * tk + i] = -1;
            out_sc[qq * tk + i] = -1e30f;
        }
        status[0] = 2;
        status[1] = 0; status[2] = 0; status[3] = 0;
        return;
    }

    for (uint32_t i = 0u; i < tk; ++i) {
        if (i < ntook) {
            out_idx[qq * tk + i] = (int32_t)took_id[i];
            out_sc[qq * tk + i] = took_sc[i];
        } else {
            out_idx[qq * tk + i] = -1;
            out_sc[qq * tk + i] = -1e30f;
        }
    }

    status[0] = 0;
    status[1] = (int32_t)ntook;
    status[2] = 0;
    status[3] = 0;
}

bool ggml_cuda_xkv_landmark_merge_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_LANDMARK_MERGE) return false;
    // 5 src slots per ggml_xkv_landmark_merge (idx,sc,base,outsc,status); src[5..9] are always NULL.
    for (int i = 0; i < 5; ++i) {
        if (i == 2) continue; // set_base can be null/dummy
        if (!op->src[i]) return false;
    }
    ggml_xkv_landmark_merge_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    return ggml_xkv_landmark_merge_supports(op->src[0], op->src[1], op->src[2],
        op->src[3], op->src[4], op, &p, err, sizeof(err));
}

void ggml_cuda_xkv_landmark_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * set_idx  = dst->src[0];
    const ggml_tensor * set_sc   = dst->src[1];
    const ggml_tensor * set_base = dst->src[2];
    ggml_tensor       * out_sc   = dst->src[3];
    ggml_tensor       * status   = dst->src[4];

    ggml_xkv_landmark_merge_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(status->data, 0, 4 * sizeof(int32_t), stream));

    const uint32_t * base_ptr = (p.has_set_base != 0 && set_base) ? (const uint32_t *)set_base->data : nullptr;

    k_xkv_landmark_merge<<<p.n_queries, 1, 0, stream>>>(
        (const int32_t *)set_idx->data,
        (const float   *)set_sc->data,
        base_ptr,
        (int32_t       *)dst->data,
        (float         *)out_sc->data,
        (int32_t       *)status->data,
        p
    );
    CUDA_CHECK(cudaGetLastError());
}
