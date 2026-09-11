#include "xkv-reconstruct.cuh"
#include "turbo-quant.cuh"
#include "ggml-xkv.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

struct cuda_xkv_reconstruct_push {
    uint32_t n_sel;
    uint32_t n_groups;
    uint32_t rank_k;
    uint32_t rank_v;
    uint32_t dim_k;
    uint32_t dim_v;
    uint32_t rotary_dim;
    uint32_t rope_mode;
    uint32_t a_k_type;
    uint32_t b_k_type;
    uint32_t a_v_type;
    uint32_t b_v_type;
    uint32_t a_k_stride;
    uint32_t b_k_stride;
    uint32_t a_v_stride;
    uint32_t b_v_stride;
    uint32_t prk;
    uint32_t prv;
    uint32_t n_rows_a;
    uint32_t b_k_rows;
    uint32_t b_v_rows;
};

// Cody-Waite range reduction of pos * omega to [-pi, pi]
static __device__ __forceinline__ void rope_trig_cody_waite_cuda(float pos_f, float omega, float & out_c, float & out_s) {
    const float INV_TWO_PI = 0.15915494309189535f; // 1 / (2*pi)
    const float TWO_PI_HI  = 6.2831854820251465f;  // high 24 bits
    const float TWO_PI_LO  = -1.748455588302094e-7f; // low bits
    float ang = pos_f * omega;
    float k_cycles = roundf(ang * INV_TWO_PI);
    float red_ang = fmaf(k_cycles, -TWO_PI_HI, ang);
    red_ang = fmaf(k_cycles, -TWO_PI_LO, red_ang);
    out_c = cosf(red_ang);
    out_s = sinf(red_ang);
}

static __device__ __forceinline__ float dequant_element(
        const uint8_t * __restrict__ base,
        uint32_t type,
        uint32_t r) {
    if (type == GGML_TYPE_F32) {
        return ((const float *) base)[r];
    }
    if (type == GGML_TYPE_F16) {
        return __half2float(((const __half *) base)[r]);
    }
    if (type == GGML_TYPE_Q8_0) {
        const uint32_t b = r / 32u;
        const uint32_t j = r % 32u;
        const uint8_t * boff = base + b * 34u;
        float d = __half2float(*((const __half *) boff));
        int8_t q = (int8_t)(boff[2u + j]);
        return d * (float) q;
    }
    if (type == GGML_TYPE_TURBO2_0) {
        const uint32_t b = r / 128u;
        const uint32_t j = r % 128u;
        const uint8_t * boff = base + b * 34u;
        float norm = __half2float(*((const __half *) boff));
        uint8_t qb = boff[2u + j / 4u];
        uint32_t idx = (qb >> (2u * (j % 4u))) & 3u;
        return norm * TURBO_CENTROIDS_2BIT[idx];
    }
    if (type == GGML_TYPE_TURBO3_0) {
        const uint32_t b = r / 128u;
        const uint32_t j = r % 128u;
        const uint8_t * boff = base + b * 50u;
        float norm = __half2float(*((const __half *) boff));
        uint8_t qb = boff[2u + j / 4u];
        uint8_t sb = boff[2u + 32u + j / 8u];
        uint32_t low2 = (qb >> (2u * (j % 4u))) & 3u;
        uint32_t hi1 = (sb >> (j % 8u)) & 1u;
        return norm * TURBO_CENTROIDS_3BIT[low2 | (hi1 << 2u)];
    }
    if (type == GGML_TYPE_TURBO4_0) {
        const uint32_t b = r / 128u;
        const uint32_t j = r % 128u;
        const uint8_t * boff = base + b * 68u;
        float norm = __half2float(*((const __half *) boff));
        uint8_t qb = boff[4u + j / 2u];
        uint32_t idx = (qb >> (4u * (j % 2u))) & 15u;
        return norm * TURBO_CENTROIDS_4BIT[idx];
    }
    return 0.0f;
}

static __global__ void k_xkv_reconstruct(
        const uint8_t * __restrict__ ak,
        const uint8_t * __restrict__ bk,
        const uint8_t * __restrict__ av,
        const uint8_t * __restrict__ bv,
        const int32_t * __restrict__ refs,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ group_meta,
        const int32_t * __restrict__ layer_meta,
        const float   * __restrict__ rope,
        float         * __restrict__ dst,
        cuda_xkv_reconstruct_push p) {
    uint32_t sel = blockIdx.x * blockDim.x + threadIdx.x;
    if (sel >= p.n_sel) {
        return;
    }

    uint32_t out_dim = p.dim_k + p.dim_v;
    uint32_t dst_base = sel * out_dim;

    int32_t a_row = refs[sel * 4u + 0u];
    int32_t grp   = refs[sel * 4u + 1u];
    int32_t ls    = refs[sel * 4u + 2u];
    int32_t hd    = refs[sel * 4u + 3u];
    int32_t pos   = positions[sel];

    bool bad = false;
    if (a_row < 0 || (uint32_t) a_row >= p.n_rows_a) { bad = true; }
    if (grp < 0 || (uint32_t) grp >= p.n_groups) { bad = true; }
    if (ls < 0) { bad = true; }

    int32_t ok = 0, dk = 0, ov = 0, dv = 0, nh = 0;
    if (!bad) {
        ok = layer_meta[(uint32_t) ls * 5u + 0u];
        dk = layer_meta[(uint32_t) ls * 5u + 1u];
        ov = layer_meta[(uint32_t) ls * 5u + 2u];
        dv = layer_meta[(uint32_t) ls * 5u + 3u];
        nh = layer_meta[(uint32_t) ls * 5u + 4u];
        if (dk <= 0 || dv <= 0 || nh <= 0) { bad = true; }
        if (hd < 0 || hd >= nh) { bad = true; }
    }

    int32_t feat_k = 0, feat_v = 0;
    if (!bad) {
        feat_k = ok + hd * dk;
        feat_v = ov + hd * dv;
        if (feat_k < 0 || ok < 0 || (uint32_t)(feat_k + dk) > p.b_k_rows) { bad = true; }
        if (feat_v < 0 || ov < 0 || (uint32_t)(feat_v + dv) > p.b_v_rows) { bad = true; }
        if ((uint32_t) dk > p.dim_k || (uint32_t) dv > p.dim_v) { bad = true; }
    }

    if (bad) {
        for (uint32_t d = 0u; d < out_dim; ++d) {
            dst[dst_base + d] = NAN;
        }
        return;
    }

    const uint8_t * a_k_base = ak + (uint64_t) a_row * p.a_k_stride;
    for (int32_t d = 0; d < dk; ++d) {
        const uint8_t * b_base = bk + (uint64_t)(feat_k + d) * p.b_k_stride;
        float acc = 0.0f;
        for (uint32_t r = 0u; r < p.prk; ++r) {
            acc += dequant_element(a_k_base, p.a_k_type, r) * dequant_element(b_base, p.b_k_type, r);
        }
        dst[dst_base + (uint32_t) d] = acc;
    }
    for (int32_t d = dk; d < (int32_t) p.dim_k; ++d) {
        dst[dst_base + (uint32_t) d] = 0.0f;
    }

    // Table-driven exact partial RoPE
    if (p.rotary_dim > 0u) {
        uint32_t fc0 = p.rotary_dim / 2u;
        uint32_t fc = fc0;
        if (fc > (uint32_t) dk / 2u) {
            fc = (uint32_t) dk / 2u;
        }
        if (p.rope_mode == 0u) { // GGML_XKV_ROPE_HALF
            for (uint32_t f = 0u; f < fc; ++f) {
                float omega = rope[f];
                float mg = rope[fc0 + f];
                float c, s;
                rope_trig_cody_waite_cuda((float) pos, omega, c, s);
                float re = dst[dst_base + f];
                float im = dst[dst_base + f + fc];
                dst[dst_base + f] = (re * c - im * s) * mg;
                dst[dst_base + f + fc] = (re * s + im * c) * mg;
            }
        } else { // GGML_XKV_ROPE_INTERLEAVED
            for (uint32_t f = 0u; f < fc; ++f) {
                float omega = rope[f];
                float mg = rope[fc0 + f];
                float c, s;
                rope_trig_cody_waite_cuda((float) pos, omega, c, s);
                float re = dst[dst_base + 2u * f];
                float im = dst[dst_base + 2u * f + 1u];
                dst[dst_base + 2u * f] = (re * c - im * s) * mg;
                dst[dst_base + 2u * f + 1u] = (re * s + im * c) * mg;
            }
        }
    }

    const uint8_t * a_v_base = av + (uint64_t) a_row * p.a_v_stride;
    for (int32_t d = 0; d < dv; ++d) {
        const uint8_t * b_base = bv + (uint64_t)(feat_v + d) * p.b_v_stride;
        float acc = 0.0f;
        for (uint32_t r = 0u; r < p.prv; ++r) {
            acc += dequant_element(a_v_base, p.a_v_type, r) * dequant_element(b_base, p.b_v_type, r);
        }
        dst[dst_base + p.dim_k + (uint32_t) d] = acc;
    }
    for (int32_t d = dv; d < (int32_t) p.dim_v; ++d) {
        dst[dst_base + p.dim_k + (uint32_t) d] = 0.0f;
    }
}

bool ggml_cuda_xkv_reconstruct_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_XKV_RECONSTRUCT) {
        return false;
    }
    if (!op->src[0] || !op->src[1] || !op->src[2] || !op->src[3] ||
        !op->src[4] || !op->src[5] || !op->src[6] || !op->src[7] || !op->src[8]) {
        return false;
    }
    ggml_xkv_reconstruct_params p;
    memcpy(&p, op->op_params, sizeof(p));
    char err[256] = {0};
    if (!ggml_xkv_reconstruct_supports(op->src[0], op->src[1], op->src[2],
        op->src[3], op->src[4], op->src[5], op->src[6], op->src[7], op->src[8], op, &p,
        err, sizeof(err))) {
        return false;
    }

    auto is_turbo = [](enum ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    bool kak = is_turbo(op->src[0]->type), kbk = is_turbo(op->src[1]->type);
    bool kav = is_turbo(op->src[2]->type), kbv = is_turbo(op->src[3]->type);
    if ((kak != kbk) || (kav != kbv)) {
        return false;
    }

    auto is_supported_type = [](enum ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 ||
               t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    if (!is_supported_type(op->src[0]->type) || !is_supported_type(op->src[1]->type) ||
        !is_supported_type(op->src[2]->type) || !is_supported_type(op->src[3]->type)) {
        return false;
    }

    return true;
}

void ggml_cuda_xkv_reconstruct(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a_k = dst->src[0];
    const ggml_tensor * b_k = dst->src[1];
    const ggml_tensor * a_v = dst->src[2];
    const ggml_tensor * b_v = dst->src[3];
    const ggml_tensor * refs = dst->src[4];
    const ggml_tensor * positions = dst->src[5];
    const ggml_tensor * group_meta = dst->src[6];
    const ggml_tensor * layer_meta = dst->src[7];
    const ggml_tensor * rope_tables = dst->src[8];

    ggml_xkv_reconstruct_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    cuda_xkv_reconstruct_push push;
    push.n_sel = p.n_sel;
    push.n_groups = p.n_groups;
    push.rank_k = p.rank_k;
    push.rank_v = p.rank_v;
    push.dim_k = p.dim_k;
    push.dim_v = p.dim_v;
    push.rotary_dim = p.rotary_dim;
    push.rope_mode = p.rope_mode;
    push.a_k_type = (uint32_t) a_k->type;
    push.b_k_type = (uint32_t) b_k->type;
    push.a_v_type = (uint32_t) a_v->type;
    push.b_v_type = (uint32_t) b_v->type;
    push.a_k_stride = (uint32_t) ggml_row_size(a_k->type, a_k->ne[0]);
    push.b_k_stride = (uint32_t) ggml_row_size(b_k->type, b_k->ne[0]);
    push.a_v_stride = (uint32_t) ggml_row_size(a_v->type, a_v->ne[0]);
    push.b_v_stride = (uint32_t) ggml_row_size(b_v->type, b_v->ne[0]);
    push.prk = (uint32_t) a_k->ne[0];
    push.prv = (uint32_t) a_v->ne[0];
    push.n_rows_a = (uint32_t) a_k->ne[1];
    push.b_k_rows = (uint32_t) b_k->ne[1];
    push.b_v_rows = (uint32_t) b_v->ne[1];

    cudaStream_t stream = ctx.stream();
    constexpr int block_size = 64;
    int num_blocks = (p.n_sel + block_size - 1) / block_size;

    k_xkv_reconstruct<<<num_blocks, block_size, 0, stream>>>(
        (const uint8_t *) a_k->data,
        (const uint8_t *) b_k->data,
        (const uint8_t *) a_v->data,
        (const uint8_t *) b_v->data,
        (const int32_t *) refs->data,
        (const int32_t *) positions->data,
        (const int32_t *) group_meta->data,
        (const int32_t *) layer_meta->data,
        (const float *)   rope_tables->data,
        (float *)         dst->data,
        push
    );
    CUDA_CHECK(cudaGetLastError());
}
