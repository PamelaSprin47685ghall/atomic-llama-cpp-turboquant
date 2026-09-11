#include "flashprefill.cuh"
#include "turbo-quant.cuh"
#include "ggml-flashprefill.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cfloat>

// Supported KV cache element types on CUDA FlashPrefill
static bool ggml_cuda_fp_kv_type_supported(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            return true;
        default:
            return false;
    }
}

// Unpack op_params helper
struct ggml_cuda_fp_params {
    int32_t op = 0;
    float alpha = 0.0f;
    float scale = 0.0f;
    float softcap = 0.0f;
    int32_t exact_all = 0;
    int32_t dk = 0;
    int32_t dv = 0;
    int32_t mean_correction = 0;
};

static bool ggml_cuda_fp_unpack_params(const ggml_tensor * node, int32_t expect_op, ggml_cuda_fp_params & out) {
    int32_t w[16];
    memcpy(w, node->op_params, sizeof(w));
    if (w[0] != expect_op || w[7] != (int32_t) GGML_FLASHPREFILL_VERSION) {
        return false;
    }
    for (int i = 9; i < 16; i++) {
        if (w[i] != 0) {
            return false;
        }
    }
    memcpy(&out.alpha, &w[1], sizeof(float));
    memcpy(&out.scale, &w[2], sizeof(float));
    memcpy(&out.softcap, &w[3], sizeof(float));
    out.exact_all = w[4];
    out.dk = w[5];
    out.dv = w[6];
    out.mean_correction = w[8];
    return true;
}

static bool ggml_cuda_fp_float_ok(float x) {
    return !std::isnan(x) && !std::isinf(x);
}

bool ggml_cuda_flash_prefill_pool_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_PREFILL_POOL) {
        return false;
    }
    const ggml_tensor * K = dst->src[0];
    const ggml_tensor * V = dst->src[1];
    const ggml_tensor * MT = dst->src[2];
    if (!K || !V || !MT) {
        return false;
    }
    ggml_cuda_fp_params pp;
    if (!ggml_cuda_fp_unpack_params(dst, GGML_FLASHPREFILL_OP_POOL, pp)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (MT->type != GGML_TYPE_I32 || !ggml_is_contiguous(MT) || MT->ne[0] < GGML_FLASHPREFILL_HEADER_WORDS) {
        return false;
    }
    if (pp.dk < 1 || pp.dv < 1) {
        return false;
    }
    if (K->ne[0] != pp.dk || V->ne[0] != pp.dv) {
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[1] < 1) {
        return false;
    }
    if (K->ne[2] != V->ne[2] || K->ne[2] < 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (dst->ne[0] != (int64_t) pp.dk + pp.dv || dst->ne[1] != K->ne[2] || dst->ne[2] < 1 || dst->ne[3] != 1) {
        return false;
    }
    if (!ggml_cuda_fp_kv_type_supported(K->type) || !ggml_cuda_fp_kv_type_supported(V->type)) {
        return false;
    }
    if ((K->type == GGML_TYPE_BF16) != (V->type == GGML_TYPE_BF16)) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }
    return true;
#endif
}

bool ggml_cuda_flash_prefill_select_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_PREFILL_SELECT) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * pool = dst->src[1];
    const ggml_tensor * MT = dst->src[2];
    if (!Q || !pool || !MT) {
        return false;
    }
    ggml_cuda_fp_params pp;
    if (!ggml_cuda_fp_unpack_params(dst, GGML_FLASHPREFILL_OP_SELECT, pp)) {
        return false;
    }
    if (dst->type != GGML_TYPE_I32 || !ggml_is_contiguous(dst) || dst->ne[0] < GGML_FLASHPREFILL_PLAN_HEADER_WORDS) {
        return false;
    }
    if (MT->type != GGML_TYPE_I32 || !ggml_is_contiguous(MT) || MT->ne[0] < GGML_FLASHPREFILL_HEADER_WORDS) {
        return false;
    }
    if (pp.dk < 1 || pp.dv < 1 || (pp.exact_all != 0 && pp.exact_all != 1)) {
        return false;
    }
    if (!ggml_cuda_fp_float_ok(pp.alpha) || pp.alpha <= 0.0f || pp.alpha > 1.0f) {
        return false;
    }
    if (!ggml_cuda_fp_float_ok(pp.scale) || !ggml_cuda_fp_float_ok(pp.softcap) || pp.softcap < 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || pool->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != pp.dk || Q->ne[1] < 1 || Q->ne[2] < 1 || Q->ne[3] != 1) {
        return false;
    }
    if (pool->ne[0] != (int64_t) pp.dk + pp.dv || pool->ne[1] < 1 || pool->ne[3] != 1) {
        return false;
    }
    if (Q->ne[2] % pool->ne[1] != 0) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % sizeof(float) != 0 || Q->nb[2] % sizeof(float) != 0) {
        return false;
    }
    return true;
#endif
}

bool ggml_cuda_flash_prefill_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_PREFILL_ATTN) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * pool = dst->src[3];
    const ggml_tensor * plan = dst->src[4];
    const ggml_tensor * MT = dst->src[5];
    const ggml_tensor * sinks = dst->src[6];
    if (!Q || !K || !V || !pool || !plan || !MT) {
        return false;
    }
    ggml_cuda_fp_params pp;
    if (!ggml_cuda_fp_unpack_params(dst, GGML_FLASHPREFILL_OP_ATTN, pp)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (plan->type != GGML_TYPE_I32 || !ggml_is_contiguous(plan) || plan->ne[0] < GGML_FLASHPREFILL_PLAN_HEADER_WORDS) {
        return false;
    }
    if (MT->type != GGML_TYPE_I32 || !ggml_is_contiguous(MT) || MT->ne[0] < GGML_FLASHPREFILL_HEADER_WORDS) {
        return false;
    }
    if (pp.dk < 1 || pp.dv < 1 || (pp.mean_correction != 0 && pp.mean_correction != 1)) {
        return false;
    }
    if (!ggml_cuda_fp_float_ok(pp.scale) || !ggml_cuda_fp_float_ok(pp.softcap) || pp.softcap < 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || pool->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_cuda_fp_kv_type_supported(K->type) || !ggml_cuda_fp_kv_type_supported(V->type)) {
        return false;
    }
    if ((K->type == GGML_TYPE_BF16) != (V->type == GGML_TYPE_BF16)) {
        return false;
    }
    if (Q->ne[0] != pp.dk || Q->ne[1] < 1 || Q->ne[2] < 1 || Q->ne[3] != 1) {
        return false;
    }
    if (K->ne[0] != pp.dk || V->ne[0] != pp.dv) {
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[1] < 1) {
        return false;
    }
    if (K->ne[2] != V->ne[2] || K->ne[2] < 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (pool->ne[0] != (int64_t) pp.dk + pp.dv || pool->ne[1] != K->ne[2] || pool->ne[3] != 1) {
        return false;
    }
    if (Q->ne[2] % pool->ne[1] != 0) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] < Q->ne[2])) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % sizeof(float) != 0 || Q->nb[2] % sizeof(float) != 0) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }
    return true;
#endif
}

// -------------------------------------------------------------------------
// Device Helpers
// -------------------------------------------------------------------------

static __device__ __forceinline__ float fp_load_element(
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
        case GGML_TYPE_Q4_0: {
            const block_q4_0 * blk = (const block_q4_0 *) ptr + (d / QK4_0);
            const int off = d % QK4_0;
            const float scale = __half2float(blk->d);
            const uint8_t byte_val = blk->qs[off / 2];
            const int nibble = (off % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
            return scale * float(nibble - 8);
        }
        case GGML_TYPE_Q4_1: {
            const block_q4_1 * blk = (const block_q4_1 *) ptr + (d / QK4_1);
            const int off = d % QK4_1;
            const float scale = __low2float(blk->dm);
            const float min_val = __high2float(blk->dm);
            const uint8_t byte_val = blk->qs[off / 2];
            const int nibble = (off % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
            return scale * float(nibble) + min_val;
        }
        case GGML_TYPE_Q5_0: {
            const block_q5_0 * blk = (const block_q5_0 *) ptr + (d / QK5_0);
            const int off = d % QK5_0;
            const float scale = __half2float(blk->d);
            uint32_t qh;
            memcpy(&qh, blk->qh, sizeof(qh));
            const uint8_t byte_val = blk->qs[off / 2];
            const int lo = (off % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
            const int hi = (qh >> off) & 1;
            const int quant = lo | (hi << 4);
            return scale * float(quant - 16);
        }
        case GGML_TYPE_Q5_1: {
            const block_q5_1 * blk = (const block_q5_1 *) ptr + (d / QK5_1);
            const int off = d % QK5_1;
            const float scale = __low2float(blk->dm);
            const float min_val = __high2float(blk->dm);
            uint32_t qh;
            memcpy(&qh, blk->qh, sizeof(qh));
            const uint8_t byte_val = blk->qs[off / 2];
            const int lo = (off % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
            const int hi = (qh >> off) & 1;
            const int quant = lo | (hi << 4);
            return scale * float(quant) + min_val;
        }
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * blk = (const block_q8_0 *) ptr + (d / QK8_0);
            const int off = d % QK8_0;
            const float scale = __half2float(blk->d);
            return scale * float(blk->qs[off]);
        }
        case GGML_TYPE_IQ4_NL: {
            // 32-element block, fp16 scale d, 4-bit indices into the 16-entry
            // kvalues_iq4nl table (mirrors Vulkan flash_attn_dequant.glsl and
            // the CPU to_float): low nibbles hold elements [0,16), high [16,32).
            const block_iq4_nl * blk = (const block_iq4_nl *) ptr + (d / QK4_NL);
            const int off = d % QK4_NL;
            const float scale = __half2float(blk->d);
            const uint8_t byte_val = blk->qs[off & 15];
            const int nibble = (off < 16) ? (byte_val & 0x0F) : (byte_val >> 4);
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

static __device__ __forceinline__ float fp_softcap_apply(float x, float cap) {
    if (cap == 0.0f) return x;
    return tanhf(x / cap) * cap;
}

// -------------------------------------------------------------------------
// POOL Kernel
// -------------------------------------------------------------------------
// Grid: (f_cap, h_cap), block: 128 or 256 threads.
// Each block computes mean for (fragment f, head h) across Dk+Dv dimensions.

__global__ void flash_prefill_pool_kernel(
        const char * __restrict__ K,
        const char * __restrict__ V,
        const int32_t * __restrict__ meta,
        float * __restrict__ pool,
        uint32_t dk,
        uint32_t dv,
        uint32_t f_cap,
        uint32_t h_cap,
        uint32_t meta_words,
        uint32_t n_kv,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t v_nb1,
        uint64_t v_nb2,
        int type_k,
        int type_v) {
    const uint32_t f = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t bsize = blockDim.x;

    if (f >= f_cap || h >= h_cap) return;

    const uint32_t dtot = dk + dv;
    bool header_ok = true;
    uint32_t n_frag = 0;
    uint32_t cell_base = 0;
    uint32_t cell_cnt = 0;

    if (meta_words < GGML_FLASHPREFILL_HEADER_WORDS) {
        header_ok = false;
    } else {
        if (meta[0] != GGML_FLASHPREFILL_METADATA_MAGIC || meta[1] != GGML_FLASHPREFILL_VERSION) header_ok = false;
        if (meta[2] != GGML_FLASHPREFILL_HEADER_WORDS || meta[15] != GGML_FLASHPREFILL_FRAG_WORDS) header_ok = false;
        if (meta[16] != GGML_FLASHPREFILL_ROW_WORDS || meta[17] != GGML_FLASHPREFILL_USE_WORDS) header_ok = false;
        n_frag = uint32_t(meta[3]);
        uint32_t mf_fcap = uint32_t(meta[4]);
        uint32_t mf_dk = uint32_t(meta[18]);
        uint32_t mf_dv = uint32_t(meta[19]);
        uint32_t H = uint32_t(meta[20]);
        if (mf_dk != dk || mf_dv != dv || H != h_cap || mf_fcap != f_cap) header_ok = false;
        if (uint32_t(meta[11]) != 32u) header_ok = false;
        uint32_t n_use = uint32_t(meta[7]);
        uint32_t u_cap = uint32_t(meta[8]);
        uint32_t n_cell = uint32_t(meta[9]);
        uint32_t c_cap = uint32_t(meta[10]);
        uint32_t n_row = uint32_t(meta[5]);
        uint32_t r_cap = uint32_t(meta[6]);
        if (n_frag > f_cap || n_use > u_cap || n_cell > c_cap || n_row > r_cap) header_ok = false;
        uint32_t row_off = uint32_t(meta[12]);
        uint32_t use_off = uint32_t(meta[13]);
        uint32_t ctab = uint32_t(meta[14]);
        uint32_t total = uint32_t(meta[21]);
        if (row_off != 32u + 8u * f_cap) header_ok = false;
        if (use_off != row_off + 8u * r_cap) header_ok = false;
        if (ctab != use_off + 8u * u_cap) header_ok = false;
        if (total != ctab + c_cap || meta_words < total) header_ok = false;
        if (dk == 0 || dv == 0 || H == 0 || n_kv == 0) header_ok = false;
    }

    float bad_val = header_ok ? 0.0f : __int_as_float(0x7FC00000);
    if (!header_ok || f >= n_frag) {
        for (uint32_t d = tid; d < dtot; d += bsize) {
            pool[(f * h_cap + h) * dtot + d] = bad_val;
        }
        return;
    }

    uint32_t frag_off = uint32_t(meta[11]);
    uint32_t ctab = uint32_t(meta[14]);
    uint32_t c_cap = uint32_t(meta[10]);
    uint32_t fb = frag_off + f * 8u;
    int32_t fco = meta[fb + 0];
    int32_t fcc = meta[fb + 1];
    bool frag_ok = true;
    if (fco < 0 || fcc <= 0 || uint32_t(fco) + uint32_t(fcc) > c_cap) {
        frag_ok = false;
    } else {
        cell_base = ctab + uint32_t(fco);
        cell_cnt = uint32_t(fcc);
    }

    if (frag_ok) {
        for (uint32_t i = 0; i < cell_cnt; ++i) {
            int32_t cell = meta[cell_base + i];
            if (cell < 0 || uint32_t(cell) >= n_kv) {
                frag_ok = false;
                break;
            }
        }
    }

    if (!frag_ok) {
        for (uint32_t d = tid; d < dtot; d += bsize) {
            pool[(f * h_cap + h) * dtot + d] = 0.0f;
        }
        return;
    }

    const float inv_cnt = 1.0f / float(cell_cnt);
    for (uint32_t d = tid; d < dtot; d += bsize) {
        float sum = 0.0f;
        if (d < dk) {
            for (uint32_t i = 0; i < cell_cnt; ++i) {
                uint32_t tok = uint32_t(meta[cell_base + i]);
                const char * k_ptr = K + uint64_t(tok) * k_nb1 + uint64_t(h) * k_nb2;
                sum += fp_load_element(k_ptr, type_k, int(d));
            }
        } else {
            uint32_t vd = d - dk;
            for (uint32_t i = 0; i < cell_cnt; ++i) {
                uint32_t tok = uint32_t(meta[cell_base + i]);
                const char * v_ptr = V + uint64_t(tok) * v_nb1 + uint64_t(h) * v_nb2;
                sum += fp_load_element(v_ptr, type_v, int(vd));
            }
        }
        pool[(f * h_cap + h) * dtot + d] = sum * inv_cnt;
    }
}

// -------------------------------------------------------------------------
// SELECT Kernels: Phase 0 and Phase 1
// -------------------------------------------------------------------------

static __device__ int select_use_pair_cmp(const int32_t * meta, uint32_t idx, uint32_t use_off, int t, int h) {
    int ut = meta[use_off + idx * 8u + 1]; // FP_UW_TILE
    int uh = meta[use_off + idx * 8u + 2]; // FP_UW_KV_HEAD
    if (ut != t) return ut < t ? -1 : 1;
    if (uh != h) return uh < h ? -1 : 1;
    return 0;
}

static __device__ int select_row_pair_cmp(const int32_t * meta, uint32_t idx, uint32_t row_off, int t, int h) {
    int rt = meta[row_off + idx * 8u + 3]; // FP_RW_TILE
    int rh = meta[row_off + idx * 8u + 1]; // FP_RW_KV_HEAD
    if (rt != t) return rt < t ? -1 : 1;
    if (rh != h) return rh < h ? -1 : 1;
    return 0;
}

static __device__ uint32_t select_lower_bound_use(const int32_t * meta, uint32_t n_use, uint32_t use_off, int t, int h) {
    uint32_t lo = 0, hi = n_use;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (select_use_pair_cmp(meta, mid, use_off, t, h) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static __device__ uint32_t select_upper_bound_use(const int32_t * meta, uint32_t n_use, uint32_t use_off, int t, int h) {
    uint32_t lo = 0, hi = n_use;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (select_use_pair_cmp(meta, mid, use_off, t, h) <= 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static __device__ uint32_t select_lower_bound_row(const int32_t * meta, uint32_t n_row, uint32_t row_off, int t, int h) {
    uint32_t lo = 0, hi = n_row;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (select_row_pair_cmp(meta, mid, row_off, t, h) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static __device__ uint32_t select_upper_bound_row(const int32_t * meta, uint32_t n_row, uint32_t row_off, int t, int h) {
    uint32_t lo = 0, hi = n_row;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (select_row_pair_cmp(meta, mid, row_off, t, h) <= 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void flash_prefill_select_phase0_kernel(
        const int32_t * __restrict__ meta,
        int32_t * __restrict__ plan,
        uint32_t dk,
        uint32_t dv_cap,
        uint32_t f_cap,
        uint32_t r_cap,
        uint32_t u_cap,
        uint32_t c_cap,
        uint32_t meta_words,
        uint32_t plan_words,
        uint32_t q_groups_cap,
        uint32_t q_heads_cap,
        uint32_t hkv_cap,
        uint32_t exact_all) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (plan_words < GGML_FLASHPREFILL_PLAN_HEADER_WORDS) return;

    for (uint32_t i = 0; i < GGML_FLASHPREFILL_PLAN_HEADER_WORDS; ++i) plan[i] = 0;
    plan[0] = GGML_FLASHPREFILL_PLAN_MAGIC;
    plan[1] = GGML_FLASHPREFILL_VERSION;
    plan[2] = GGML_FLASHPREFILL_PLAN_HEADER_WORDS;
    plan[9] = int32_t(plan_words);
    plan[15] = int32_t(exact_all);
    plan[10] = GGML_FLASHPREFILL_ERR_BAD_LAYOUT; // default to bad layout until verified

    if (meta_words < GGML_FLASHPREFILL_HEADER_WORDS) return;
    if (meta[0] != GGML_FLASHPREFILL_METADATA_MAGIC || meta[1] != GGML_FLASHPREFILL_VERSION) return;

    uint32_t n_frag = uint32_t(meta[3]);
    uint32_t mf_fcap = uint32_t(meta[4]);
    uint32_t n_row = uint32_t(meta[5]);
    uint32_t mf_rcap = uint32_t(meta[6]);
    uint32_t n_use = uint32_t(meta[7]);
    uint32_t mf_ucap = uint32_t(meta[8]);
    uint32_t n_cell = uint32_t(meta[9]);
    uint32_t mf_ccap = uint32_t(meta[10]);
    uint32_t frag_off = uint32_t(meta[11]);
    uint32_t row_off = uint32_t(meta[12]);
    uint32_t use_off = uint32_t(meta[13]);
    uint32_t ctab = uint32_t(meta[14]);
    uint32_t dk_dev = uint32_t(meta[18]);
    uint32_t Hdev = uint32_t(meta[20]);
    uint32_t n_groups_dev = uint32_t(meta[22]);
    uint32_t n_qheads_dev = uint32_t(meta[23]);

    if (dk_dev != dk || n_groups_dev > q_groups_cap || n_qheads_dev > q_heads_cap) return;
    if (Hdev == 0 || Hdev != hkv_cap || n_qheads_dev == 0 || n_qheads_dev % Hdev != 0 || n_groups_dev == 0) return;
    if (mf_fcap != f_cap || mf_rcap > r_cap || mf_ucap > u_cap || mf_ccap > c_cap) return;
    if (n_frag > f_cap || n_row > r_cap || n_use > u_cap || n_cell > c_cap) return;
    if (frag_off != 32u || row_off != 32u + 8u * f_cap) return;
    if (use_off != row_off + 8u * mf_rcap) return;
    if (ctab != use_off + 8u * mf_ucap) return;
    if (meta_words < ctab + mf_ccap) return;

    uint32_t T = 0;
    for (uint32_t r = 0; r < n_row; ++r) {
        int32_t rt = meta[row_off + r * 8u + 3];
        int32_t rh = meta[row_off + r * 8u + 1];
        if (rt < 0 || rh < 0 || uint32_t(rh) >= Hdev) return;
        if (r > 0 && select_row_pair_cmp(meta, r - 1, row_off, rt, rh) > 0) return;
        T = max(T, uint32_t(rt) + 1u);
    }
    for (uint32_t u = 0; u < n_use; ++u) {
        int32_t ut = meta[use_off + u * 8u + 1];
        int32_t uh = meta[use_off + u * 8u + 2];
        if (ut < 0 || uh < 0 || uint32_t(uh) >= Hdev) return;
        if (u > 0 && select_use_pair_cmp(meta, u - 1, use_off, ut, uh) > 0) return;
        T = max(T, uint32_t(ut) + 1u);
    }

    uint32_t payload = plan_words - GGML_FLASHPREFILL_PLAN_HEADER_WORDS;
    if (T == 0 || T > r_cap + 1 || T > payload / 2u / Hdev) return;
    uint32_t pairs = T * Hdev;
    uint32_t denom = 2u * pairs;
    if (payload % denom != 0) return;
    uint32_t max_sel = payload / denom - 1u;
    uint32_t proxy_off = GGML_FLASHPREFILL_PLAN_HEADER_WORDS + pairs * max_sel;
    uint32_t counts_off = proxy_off + pairs * max_sel;

    plan[3] = int32_t(T);
    plan[4] = int32_t(Hdev);
    plan[5] = int32_t(max_sel);
    plan[6] = int32_t(GGML_FLASHPREFILL_PLAN_HEADER_WORDS);
    plan[7] = int32_t(proxy_off);
    plan[8] = int32_t(counts_off);
    plan[10] = GGML_FLASHPREFILL_OK;
}

__global__ void flash_prefill_select_phase1_kernel(
        const float * __restrict__ Q,
        const float * __restrict__ pool,
        const int32_t * __restrict__ meta,
        int32_t * __restrict__ plan,
        uint32_t dk,
        uint32_t dv_cap,
        uint32_t f_cap,
        uint32_t r_cap,
        uint32_t u_cap,
        uint32_t c_cap,
        uint32_t meta_words,
        uint32_t plan_words,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint32_t q_groups_cap,
        uint32_t q_heads_cap,
        uint32_t hkv_cap,
        uint32_t exact_all,
        float alpha,
        float scale,
        float softcap) {
    if (threadIdx.x != 0) return;
    if (plan_words < GGML_FLASHPREFILL_PLAN_HEADER_WORDS) return;

    if (plan[0] != GGML_FLASHPREFILL_PLAN_MAGIC ||
        plan[1] != GGML_FLASHPREFILL_VERSION ||
        plan[2] != GGML_FLASHPREFILL_PLAN_HEADER_WORDS ||
        atomicAdd(&plan[10], 0) != GGML_FLASHPREFILL_OK) {
        return;
    }

    uint32_t T = uint32_t(plan[3]);
    uint32_t H = uint32_t(plan[4]);
    uint32_t max_sel = uint32_t(plan[5]);
    uint32_t exact_off = uint32_t(plan[6]);
    uint32_t proxy_off = uint32_t(plan[7]);
    uint32_t counts_off = uint32_t(plan[8]);

    if (T == 0 || H == 0 || H != hkv_cap) return;
    if (alpha <= 0.0f || alpha > 1.0f) {
        atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_CONFIG);
        return;
    }

    uint32_t h = blockIdx.y;
    if (h >= H) return;

    uint32_t n_frag = uint32_t(meta[3]);
    uint32_t n_row = uint32_t(meta[5]);
    uint32_t n_use = uint32_t(meta[7]);
    uint32_t frag_off = uint32_t(meta[11]);
    uint32_t row_off = uint32_t(meta[12]);
    uint32_t use_off = uint32_t(meta[13]);
    uint32_t n_groups_dev = uint32_t(meta[22]);
    uint32_t n_qheads_dev = uint32_t(meta[23]);
    uint32_t dtot = dk + dv_cap;

    for (uint32_t t = blockIdx.x; t < T; t += gridDim.x) {
        uint32_t pair = t * H + h;
        for (uint32_t s = 0; s < max_sel; ++s) {
            plan[exact_off + pair * max_sel + s] = -1;
            plan[proxy_off + pair * max_sel + s] = -1;
        }
        plan[counts_off + pair * 2u + 0] = 0;
        plan[counts_off + pair * 2u + 1] = 0;

        uint32_t u_lo = select_lower_bound_use(meta, n_use, use_off, int(t), int(h));
        uint32_t u_hi = select_upper_bound_use(meta, n_use, use_off, int(t), int(h));
        uint32_t r_lo = select_lower_bound_row(meta, n_row, row_off, int(t), int(h));
        uint32_t r_hi = select_upper_bound_row(meta, n_row, row_off, int(t), int(h));

        if (u_lo > 0 && select_use_pair_cmp(meta, u_lo - 1, use_off, int(t), int(h)) == 0) {
            atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_LAYOUT);
            return;
        }
        if (u_hi < n_use && select_use_pair_cmp(meta, u_hi, use_off, int(t), int(h)) == 0) {
            atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_LAYOUT);
            return;
        }
        if (r_lo > 0 && select_row_pair_cmp(meta, r_lo - 1, row_off, int(t), int(h)) == 0) {
            atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_LAYOUT);
            return;
        }
        if (r_hi < n_row && select_row_pair_cmp(meta, r_hi, row_off, int(t), int(h)) == 0) {
            atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_LAYOUT);
            return;
        }

        for (uint32_t u = u_lo; u < u_hi; ++u) {
            if (select_use_pair_cmp(meta, u, use_off, int(t), int(h)) != 0) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_LAYOUT);
                return;
            }
        }

        uint32_t P = u_hi - u_lo;
        uint32_t Rp = r_hi - r_lo;
        if (P > max_sel) {
            atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_OVER_CAPACITY);
            return;
        }

        uint32_t pair_vis = 0;
        for (uint32_t k = 0; k < P; ++k) {
            uint32_t u = u_lo + k;
            uint32_t ub = use_off + u * 8u;
            int32_t frag = meta[ub + 0];
            int32_t qg = meta[ub + 3];
            int32_t so = meta[ub + 4];
            int32_t sc = meta[ub + 5];
            int32_t fl = meta[ub + 6];
            int32_t sq = meta[ub + 7];
            if (frag < 0 || uint32_t(frag) >= n_frag || qg < 0 || uint32_t(qg) >= n_groups_dev) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_RANGE);
                return;
            }
            if (sc <= 0 || so < 0 || (fl & ~GGML_FLASHPREFILL_USE_FLAG_MANDATORY) != 0 || sq < 0) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_RANGE);
                return;
            }
            uint32_t fb = frag_off + uint32_t(frag) * 8u;
            int32_t fco = meta[fb + 0];
            int32_t fcc = meta[fb + 1];
            int32_t ffl = meta[fb + 4];
            if (fco < 0 || fcc <= 0 || uint32_t(fco) + uint32_t(fcc) > c_cap) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_RANGE);
                return;
            }
            if ((ffl & ~GGML_FLASHPREFILL_FRAG_FLAG_BOUNDARY_PARTIAL) != 0) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_FLAG);
                return;
            }
            if (uint32_t(so) + uint32_t(sc) > c_cap || so < fco || uint32_t(so) + uint32_t(sc) > uint32_t(fco) + uint32_t(fcc)) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_RANGE);
                return;
            }

            bool matched = false;
            bool dense_touch = false;
            for (uint32_t j = 0; j < Rp; ++j) {
                uint32_t rb = row_off + (r_lo + j) * 8u;
                if (meta[rb + 0] != sq) continue;
                matched = true;
                if ((meta[rb + 6] & GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE) != 0) dense_touch = true;
            }
            if (!matched) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_ORPHAN_USE);
                return;
            }
            if (dense_touch && (fl & GGML_FLASHPREFILL_USE_FLAG_MANDATORY) == 0) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_DENSE_ROW_PROXY);
                return;
            }
            pair_vis += uint32_t(sc);
        }

        for (uint32_t j = 0; j < Rp; ++j) {
            uint32_t rb = row_off + (r_lo + j) * 8u;
            int32_t sq = meta[rb + 0];
            int32_t pb = meta[rb + 4];
            int32_t pe = meta[rb + 5];
            int32_t fl = meta[rb + 6];
            int32_t qh = meta[rb + 7];
            if (qh < 0 || uint32_t(qh) >= n_qheads_dev || sq < 0 || pb < 0 || pe <= pb) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_RANGE);
                return;
            }
            if ((fl & ~GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE) != 0) {
                atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_FLAG);
                return;
            }
        }

        if (P == 0) {
            atomicAdd(&plan[12], int32_t(Rp));
            continue;
        }

        if (exact_all != 0) {
            for (uint32_t k = 0; k < P; ++k) {
                plan[exact_off + pair * max_sel + k] = int32_t(u_lo + k);
            }
            plan[counts_off + pair * 2u + 0] = int32_t(P);
            plan[counts_off + pair * 2u + 1] = 0;
            atomicAdd(&plan[13], int32_t(P));
            atomicAdd(&plan[12], int32_t(Rp));
            uint32_t old_lo = uint32_t(atomicAdd(&plan[16], int32_t(pair_vis)));
            if (old_lo + pair_vis < old_lo) atomicAdd(&plan[17], 1);
            uint32_t old_e = uint32_t(atomicAdd(&plan[18], int32_t(pair_vis)));
            if (old_e + pair_vis < old_e) atomicAdd(&plan[19], 1);
            continue;
        }

        uint32_t F = 0;
        for (uint32_t k = 0; k < P; ++k) {
            uint32_t u = u_lo + k;
            int32_t frag = meta[use_off + u * 8u + 0];
            bool seen = false;
            for (uint32_t d = 0; d < F; ++d) {
                if (plan[exact_off + pair * max_sel + d] == frag) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                plan[exact_off + pair * max_sel + F] = frag;
                plan[proxy_off + pair * max_sel + F] = 0;
                F++;
            }
        }

        float M = -1e30f;
        bool have = false;
        for (uint32_t d = 0; d < F; ++d) {
            int32_t frag = plan[exact_off + pair * max_sel + d];
            bool col_full = true;
            for (uint32_t k = 0; k < P; ++k) {
                uint32_t u = u_lo + k;
                uint32_t uub = use_off + u * 8u;
                if (meta[uub + 0] != frag) continue;
                int32_t so = meta[uub + 4];
                int32_t sc = meta[uub + 5];
                uint32_t fb = frag_off + uint32_t(frag) * 8u;
                bool full = (so == meta[fb + 0]) && (sc == meta[fb + 1]);
                if (!full) {
                    col_full = false;
                    break;
                }
            }
            if (!col_full) continue;

            for (uint32_t k = 0; k < P; ++k) {
                uint32_t u = u_lo + k;
                uint32_t uub = use_off + u * 8u;
                if (meta[uub + 0] != frag) continue;
                int32_t usq = meta[uub + 7];
                int32_t qg = meta[uub + 3];
                for (uint32_t j = 0; j < Rp; ++j) {
                    uint32_t rb = row_off + (r_lo + j) * 8u;
                    if (meta[rb + 0] != usq) continue;
                    uint32_t rqh = uint32_t(meta[rb + 7]);
                    float dot = 0.0f;
                    const float * q_vec = (const float *)((const char *)Q + uint64_t(qg) * q_nb1 + uint64_t(rqh) * q_nb2);
                    const float * pb_vec = pool + (uint32_t(frag) * H + h) * dtot;
                    for (uint32_t dd = 0; dd < dk; ++dd) {
                        dot += q_vec[dd] * pb_vec[dd];
                    }
                    float z = fp_softcap_apply(scale * dot, softcap);
                    if (std::isnan(z) || std::isinf(z)) {
                        atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_INPUT);
                        return;
                    }
                    if (!have) {
                        M = z;
                        have = true;
                        float cur = __int_as_float(plan[proxy_off + pair * max_sel + d]);
                        cur += 1.0f;
                        plan[proxy_off + pair * max_sel + d] = __float_as_int(cur);
                    } else if (z > M) {
                        float rescale = expf(M - z);
                        for (uint32_t dd2 = 0; dd2 < F; ++dd2) {
                            float s = __int_as_float(plan[proxy_off + pair * max_sel + dd2]);
                            s *= rescale;
                            plan[proxy_off + pair * max_sel + dd2] = __float_as_int(s);
                        }
                        M = z;
                        float cur = __int_as_float(plan[proxy_off + pair * max_sel + d]);
                        cur += 1.0f;
                        plan[proxy_off + pair * max_sel + d] = __float_as_int(cur);
                    } else {
                        float cur = __int_as_float(plan[proxy_off + pair * max_sel + d]);
                        cur += expf(z - M);
                        plan[proxy_off + pair * max_sel + d] = __float_as_int(cur);
                    }
                }
            }
        }

        if (!have) {
            for (uint32_t k = 0; k < P; ++k) {
                plan[exact_off + pair * max_sel + k] = int32_t(u_lo + k);
            }
            for (uint32_t k = 0; k < max_sel; ++k) {
                plan[proxy_off + pair * max_sel + k] = -1;
            }
            plan[counts_off + pair * 2u + 0] = int32_t(P);
            plan[counts_off + pair * 2u + 1] = 0;
            atomicAdd(&plan[13], int32_t(P));
            atomicAdd(&plan[12], int32_t(Rp));
            uint32_t et_all = 0;
            for (uint32_t k = 0; k < P; ++k) {
                et_all += uint32_t(meta[use_off + (u_lo + k) * 8u + 5]);
            }
            uint32_t old_lo0 = uint32_t(atomicAdd(&plan[16], int32_t(pair_vis)));
            if (old_lo0 + pair_vis < old_lo0) atomicAdd(&plan[17], 1);
            uint32_t old_e0 = uint32_t(atomicAdd(&plan[18], int32_t(et_all)));
            if (old_e0 + et_all < old_e0) atomicAdd(&plan[19], 1);
            continue;
        }

        float maxS = 0.0f;
        for (uint32_t d = 0; d < F; ++d) {
            float s = __int_as_float(plan[proxy_off + pair * max_sel + d]);
            maxS = fmaxf(maxS, s);
        }
        float thresh = alpha * maxS;

        for (uint32_t k = 0; k < P; ++k) {
            uint32_t u = u_lo + k;
            uint32_t ub = use_off + u * 8u;
            int32_t frag = meta[ub + 0];
            int32_t sc = meta[ub + 5];
            int32_t so = meta[ub + 4];
            int32_t fl = meta[ub + 6];
            bool mandatory = (fl & GGML_FLASHPREFILL_USE_FLAG_MANDATORY) != 0;
            uint32_t fb = frag_off + uint32_t(frag) * 8u;
            bool full = (so == meta[fb + 0]) && (sc == meta[fb + 1]);
            bool col_full = true;
            for (uint32_t kc = 0; kc < P; ++kc) {
                uint32_t uc = u_lo + kc;
                if (meta[use_off + uc * 8u + 0] != frag) continue;
                int32_t soc = meta[use_off + uc * 8u + 4];
                int32_t scc = meta[use_off + uc * 8u + 5];
                if ((soc != meta[fb + 0]) || (scc != meta[fb + 1])) {
                    col_full = false;
                    break;
                }
            }

            uint32_t ord = 0;
            for (uint32_t kk = 0; kk < P; ++kk) {
                uint32_t uu = u_lo + kk;
                int32_t ff = meta[use_off + uu * 8u + 0];
                bool seen = false;
                for (uint32_t k2 = 0; k2 < kk; ++k2) {
                    if (meta[use_off + (u_lo + k2) * 8u + 0] == ff) { seen = true; break; }
                }
                if (!seen) {
                    if (ff == frag) break;
                    ord++;
                }
            }
            float Sf = __int_as_float(plan[proxy_off + pair * max_sel + ord]);
            bool keep = (Sf >= thresh);
            bool exact = mandatory || !full || !col_full || keep;
            plan[exact_off + pair * max_sel + k] = exact ? 1 : 0;
        }

        uint32_t e = 0, px = 0, etok = 0;
        for (uint32_t k = 0; k < P; ++k) {
            uint32_t u = u_lo + k;
            if (plan[exact_off + pair * max_sel + k] != 0) {
                if (e >= max_sel) {
                    atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_OVER_CAPACITY);
                    return;
                }
                plan[exact_off + pair * max_sel + e] = int32_t(u);
                e++;
                etok += uint32_t(meta[use_off + u * 8u + 5]);
            } else {
                if (px >= max_sel) {
                    atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_OVER_CAPACITY);
                    return;
                }
                plan[proxy_off + pair * max_sel + px] = int32_t(u);
                px++;
            }
        }

        for (uint32_t k = e; k < max_sel; ++k) plan[exact_off + pair * max_sel + k] = -1;
        for (uint32_t k = px; k < max_sel; ++k) plan[proxy_off + pair * max_sel + k] = -1;
        plan[counts_off + pair * 2u + 0] = int32_t(e);
        plan[counts_off + pair * 2u + 1] = int32_t(px);
        atomicAdd(&plan[13], int32_t(e));
        atomicAdd(&plan[14], int32_t(px));

        uint32_t n_sparse = 0;
        for (uint32_t j = 0; j < Rp; ++j) {
            uint32_t rb = row_off + (r_lo + j) * 8u;
            int32_t rsq = meta[rb + 0];
            bool is_sparse = false;
            for (uint32_t kk = 0; kk < px; ++kk) {
                uint32_t uu = uint32_t(plan[proxy_off + pair * max_sel + kk]);
                if (meta[use_off + uu * 8u + 7] == rsq) { is_sparse = true; break; }
            }
            if (is_sparse) n_sparse++;
        }
        atomicAdd(&plan[11], int32_t(n_sparse));
        atomicAdd(&plan[12], int32_t(Rp - n_sparse));

        uint32_t old_lo = uint32_t(atomicAdd(&plan[16], int32_t(pair_vis)));
        if (old_lo + pair_vis < old_lo) atomicAdd(&plan[17], 1);
        uint32_t old_e = uint32_t(atomicAdd(&plan[18], int32_t(etok)));
        if (old_e + etok < old_e) atomicAdd(&plan[19], 1);
    }
}

// -------------------------------------------------------------------------
// ATTN Kernel
// -------------------------------------------------------------------------
// 1 warp (32 threads) per metadata row (= 1 output (source_query, q_head)).
// Cooperative dot reduction and online softmax accumulation.

template<int WARPS_PER_BLOCK>
__global__ void flash_prefill_attn_kernel(
        const float * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const float * __restrict__ pool,
        int32_t * __restrict__ plan,
        const int32_t * __restrict__ meta,
        const float * __restrict__ sinks,
        float * __restrict__ dst,
        uint32_t dk,
        uint32_t dv,
        uint32_t meta_words,
        uint32_t plan_words,
        uint32_t n_kv,
        uint32_t f_cap,
        uint32_t r_cap,
        uint32_t u_cap,
        uint32_t c_cap,
        uint32_t n_output_cap,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint32_t q_groups_cap,
        uint32_t q_heads_cap,
        uint32_t hkv_cap,
        uint32_t sinks_present,
        uint32_t mean_correction,
        float scale,
        float softcap,
        int type_k,
        int type_v) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const uint32_t r = blockIdx.x * WARPS_PER_BLOCK + warp;

    __shared__ uint32_t sh_coverage_bad[WARPS_PER_BLOCK];

    bool bad = false;
    int32_t bad_code = GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    if (meta_words < GGML_FLASHPREFILL_HEADER_WORDS || plan_words < GGML_FLASHPREFILL_PLAN_HEADER_WORDS) {
        bad = true;
    }

    uint32_t n_frag = 0, n_row = 0, n_use = 0, frag_off = 0, row_off = 0, use_off = 0, ctab = 0;
    uint32_t Hkv = 0, n_groups = 0, Hq = 0;
    uint32_t T = 0, max_sel = 0, exact_off = 24u, proxy_off = 24u, counts_off = 24u;

    if (!bad) {
        if (meta[0] != GGML_FLASHPREFILL_METADATA_MAGIC || meta[1] != GGML_FLASHPREFILL_VERSION) bad = true;
        n_frag = uint32_t(meta[3]);
        n_row = uint32_t(meta[5]);
        n_use = uint32_t(meta[7]);
        uint32_t mf_fcap = uint32_t(meta[4]);
        uint32_t mf_rcap = uint32_t(meta[6]);
        uint32_t mf_ucap = uint32_t(meta[8]);
        uint32_t mf_ccap = uint32_t(meta[10]);
        frag_off = uint32_t(meta[11]);
        row_off = uint32_t(meta[12]);
        use_off = uint32_t(meta[13]);
        ctab = uint32_t(meta[14]);

        if (meta[2] != GGML_FLASHPREFILL_HEADER_WORDS || meta[15] != 8 || meta[16] != 8 || meta[17] != 8) bad = true;
        if (mf_fcap != f_cap || frag_off != 32u || row_off != 32u + 8u * mf_fcap) bad = true;
        if (use_off != row_off + 8u * mf_rcap || ctab != use_off + 8u * mf_ucap) bad = true;
        if (meta_words < ctab + mf_ccap) bad = true;
        if (uint32_t(meta[18]) != dk || uint32_t(meta[19]) != dv) bad = true;
        Hkv = uint32_t(meta[20]);
        n_groups = uint32_t(meta[22]);
        Hq = uint32_t(meta[23]);
        if (n_frag > f_cap || n_row > r_cap || n_use > u_cap || uint32_t(meta[9]) > c_cap) {
            bad = true;
            bad_code = GGML_FLASHPREFILL_ERR_CAP_EXCEEDED;
        }
        if (Hkv > hkv_cap || n_groups > q_groups_cap || Hq > q_heads_cap) bad = true;
        if (plan[0] != GGML_FLASHPREFILL_PLAN_MAGIC || plan[1] != GGML_FLASHPREFILL_VERSION) bad = true;
        T = uint32_t(plan[3]);
        uint32_t Hplan = uint32_t(plan[4]);
        max_sel = uint32_t(plan[5]);
        exact_off = uint32_t(plan[6]);
        proxy_off = uint32_t(plan[7]);
        counts_off = uint32_t(plan[8]);
        if (Hplan != Hkv || T == 0 || Hkv == 0) bad = true;
        if (exact_off != 24u || proxy_off != 24u + T * Hkv * max_sel) bad = true;
        if (counts_off != proxy_off + T * Hkv * max_sel) bad = true;
        if (plan[10] != GGML_FLASHPREFILL_OK) {
            bad = true;
            bad_code = plan[10];
        }
    }

    if (bad) {
        if (plan_words >= GGML_FLASHPREFILL_PLAN_HEADER_WORDS && lane == 0) {
            atomicMax(&plan[10], bad_code);
        }
        return;
    }

    if (r >= n_row) return;

    uint32_t rb = row_off + r * 8u;
    int32_t src_q = meta[rb + 0];
    int32_t kv_head = meta[rb + 1];
    int32_t tile = meta[rb + 3];
    int32_t q_head = meta[rb + 7];

    bool row_bad = false;
    if (src_q < 0 || uint32_t(src_q) >= n_output_cap) row_bad = true;
    if (kv_head < 0 || uint32_t(kv_head) >= Hkv) row_bad = true;
    if (q_head < 0 || uint32_t(q_head) >= Hq) row_bad = true;
    if (tile < 0 || uint32_t(tile) >= T) row_bad = true;

    if (row_bad) {
        if (lane == 0) atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_RANGE);
        return;
    }

    uint32_t u_tile = uint32_t(tile), u_kv = uint32_t(kv_head), u_hq = uint32_t(q_head), u_sq = uint32_t(src_q);
    uint32_t pair = u_tile * Hkv + u_kv;
    int32_t n_exact = plan[counts_off + pair * 2u + 0];
    int32_t n_proxy = plan[counts_off + pair * 2u + 1];

    float * out_ptr = dst + dv * (u_hq + Hq * u_sq);

    if (n_exact < 0 || n_proxy < 0 || uint32_t(n_exact) > max_sel || uint32_t(n_proxy) > max_sel) {
        for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
            out_ptr[d] = 0.0f;
        }
        if (lane == 0) atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE);
        return;
    }

    if (lane == 0) {
        sh_coverage_bad[warp] = 0;
    }
    __syncwarp();

    for (uint32_t u = lane; u < n_use; u += WARP_SIZE) {
        uint32_t ub = use_off + u * 8u;
        if (meta[ub + 1] != int32_t(u_tile)) continue;
        if (meta[ub + 2] != int32_t(u_kv)) continue;
        if (meta[ub + 7] != int32_t(u_sq)) continue;
        int found = 0;
        for (int k = 0; k < n_exact; ++k) {
            if (plan[exact_off + pair * max_sel + uint32_t(k)] == int32_t(u)) found++;
        }
        for (int k = 0; k < n_proxy; ++k) {
            if (plan[proxy_off + pair * max_sel + uint32_t(k)] == int32_t(u)) found++;
        }
        if (found != 1) {
            atomicOr(&sh_coverage_bad[warp], 1u);
        }
    }
    __syncwarp();

    if (sh_coverage_bad[warp] != 0) {
        for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
            out_ptr[d] = 0.0f;
        }
        if (lane == 0) atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE);
        return;
    }

    float sink_logit = 0.0f;
    bool have_sink = false;
    if (sinks_present != 0) {
        sink_logit = sinks[u_hq];
        if (std::isnan(sink_logit) || std::isinf(sink_logit)) {
            for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
                out_ptr[d] = 0.0f;
            }
            if (lane == 0) atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_BAD_INPUT);
            return;
        }
        have_sink = true;
    }

    uint32_t dtot = dk + dv;
    float M = -1e30f;
    float L = 0.0f;
    bool have = false;

    // Pass 1: compute M and L
    for (int k = 0; k < n_exact; ++k) {
        int32_t ui = plan[exact_off + pair * max_sel + uint32_t(k)];
        uint32_t ub = use_off + uint32_t(ui) * 8u;
        if (meta[ub + 7] != int32_t(u_sq)) continue;
        uint32_t qg = uint32_t(meta[ub + 3]);
        uint32_t so = uint32_t(meta[ub + 4]);
        int32_t sc = meta[ub + 5];
        const float * q_vec = (const float *)((const char *)Q + uint64_t(qg) * q_nb1 + uint64_t(u_hq) * q_nb2);

        for (int j = 0; j < sc; ++j) {
            int32_t cell = meta[ctab + so + uint32_t(j)];
            const char * k_ptr = K + uint64_t(cell) * k_nb1 + uint64_t(u_kv) * k_nb2;

            float dot = 0.0f;
            for (uint32_t d = lane; d < dk; d += WARP_SIZE) {
                dot += q_vec[d] * fp_load_element(k_ptr, type_k, int(d));
            }
            dot = warp_reduce_sum(dot);

            float z = fp_softcap_apply(scale * dot, softcap);
            float m_new = have ? fmaxf(M, z) : z;
            float keep = have ? expf(M - m_new) : 0.0f;
            float weight = expf(z - m_new);
            L = L * keep + weight;
            M = m_new;
            have = true;
        }
    }

    if (mean_correction != 0) {
        for (int k = 0; k < n_proxy; ++k) {
            int32_t ui = plan[proxy_off + pair * max_sel + uint32_t(k)];
            uint32_t ub = use_off + uint32_t(ui) * 8u;
            if (meta[ub + 7] != int32_t(u_sq)) continue;
            uint32_t qg = uint32_t(meta[ub + 3]);
            uint32_t frag = uint32_t(meta[ub + 0]);
            int32_t sc = meta[ub + 5];
            int32_t so = meta[ub + 4];
            uint32_t pfb = frag_off + frag * 8u;
            bool pfull = (so == meta[pfb + 0]) && (sc == meta[pfb + 1]);
            bool pmand = (meta[ub + 6] & GGML_FLASHPREFILL_USE_FLAG_MANDATORY) != 0;
            if (!pfull || pmand) {
                for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
                    out_ptr[d] = 0.0f;
                }
                if (lane == 0) atomicMax(&plan[10], GGML_FLASHPREFILL_ERR_MANDATORY_AS_PROXY);
                return;
            }

            const float * q_vec = (const float *)((const char *)Q + uint64_t(qg) * q_nb1 + uint64_t(u_hq) * q_nb2);
            const float * kbar_vec = pool + (frag * Hkv + u_kv) * dtot;

            float dot = 0.0f;
            for (uint32_t d = lane; d < dk; d += WARP_SIZE) {
                dot += q_vec[d] * kbar_vec[d];
            }
            dot = warp_reduce_sum(dot);

            float z = fp_softcap_apply(scale * dot, softcap) + logf(float(sc));
            float m_new = have ? fmaxf(M, z) : z;
            float keep = have ? expf(M - m_new) : 0.0f;
            float weight = expf(z - m_new);
            L = L * keep + weight;
            M = m_new;
            have = true;
        }
    }

    if (have_sink) {
        float m_new = have ? fmaxf(M, sink_logit) : sink_logit;
        float keep = have ? expf(M - m_new) : 0.0f;
        L = L * keep + expf(sink_logit - m_new);
        M = m_new;
        have = true;
    }

    if (!have || L <= 0.0f) {
        for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
            out_ptr[d] = 0.0f;
        }
        return;
    }

    // Pass 2: accumulate weighted V vectors into out_ptr
    for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
        out_ptr[d] = 0.0f;
    }

    const float inv_L = 1.0f / L;

    for (int k = 0; k < n_exact; ++k) {
        int32_t ui = plan[exact_off + pair * max_sel + uint32_t(k)];
        uint32_t ub = use_off + uint32_t(ui) * 8u;
        if (meta[ub + 7] != int32_t(u_sq)) continue;
        uint32_t qg = uint32_t(meta[ub + 3]);
        uint32_t so = uint32_t(meta[ub + 4]);
        int32_t sc = meta[ub + 5];
        const float * q_vec = (const float *)((const char *)Q + uint64_t(qg) * q_nb1 + uint64_t(u_hq) * q_nb2);

        for (int j = 0; j < sc; ++j) {
            int32_t cell = meta[ctab + so + uint32_t(j)];
            const char * k_ptr = K + uint64_t(cell) * k_nb1 + uint64_t(u_kv) * k_nb2;

            float dot = 0.0f;
            for (uint32_t d = lane; d < dk; d += WARP_SIZE) {
                dot += q_vec[d] * fp_load_element(k_ptr, type_k, int(d));
            }
            dot = warp_reduce_sum(dot);

            float z = fp_softcap_apply(scale * dot, softcap);
            float weight = expf(z - M) * inv_L;

            const char * v_ptr = V + uint64_t(cell) * v_nb1 + uint64_t(u_kv) * v_nb2;
            for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
                out_ptr[d] += weight * fp_load_element(v_ptr, type_v, int(d));
            }
        }
    }

    if (mean_correction != 0) {
        for (int k = 0; k < n_proxy; ++k) {
            int32_t ui = plan[proxy_off + pair * max_sel + uint32_t(k)];
            uint32_t ub = use_off + uint32_t(ui) * 8u;
            if (meta[ub + 7] != int32_t(u_sq)) continue;
            uint32_t qg = uint32_t(meta[ub + 3]);
            uint32_t frag = uint32_t(meta[ub + 0]);
            int32_t sc = meta[ub + 5];

            const float * q_vec = (const float *)((const char *)Q + uint64_t(qg) * q_nb1 + uint64_t(u_hq) * q_nb2);
            const float * kbar_vec = pool + (frag * Hkv + u_kv) * dtot;

            float dot = 0.0f;
            for (uint32_t d = lane; d < dk; d += WARP_SIZE) {
                dot += q_vec[d] * kbar_vec[d];
            }
            dot = warp_reduce_sum(dot);

            float z = fp_softcap_apply(scale * dot, softcap) + logf(float(sc));
            float weight = expf(z - M) * inv_L;

            const float * vbar_vec = pool + (frag * Hkv + u_kv) * dtot + dk;
            for (uint32_t d = lane; d < dv; d += WARP_SIZE) {
                out_ptr[d] += weight * vbar_vec[d];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Host Wrappers
// -------------------------------------------------------------------------

void ggml_cuda_flash_prefill_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * K = dst->src[0];
    const ggml_tensor * V = dst->src[1];
    const ggml_tensor * MT = dst->src[2];

    ggml_cuda_fp_params pp;
    GGML_ASSERT(ggml_cuda_fp_unpack_params(dst, GGML_FLASHPREFILL_OP_POOL, pp));

    const uint32_t f_cap = (uint32_t) dst->ne[2];
    const uint32_t h_cap = (uint32_t) dst->ne[1];
    const uint32_t dk = (uint32_t) pp.dk;
    const uint32_t dv = (uint32_t) pp.dv;
    const uint32_t meta_words = (uint32_t) MT->ne[0];
    const uint32_t n_kv = (uint32_t) K->ne[1];

    dim3 block(128, 1, 1);
    dim3 grid(f_cap, h_cap, 1);

    flash_prefill_pool_kernel<<<grid, block, 0, stream>>>(
        (const char *) K->data,
        (const char *) V->data,
        (const int32_t *) MT->data,
        (float *) dst->data,
        dk, dv, f_cap, h_cap, meta_words, n_kv,
        K->nb[1], K->nb[2], V->nb[1], V->nb[2],
        K->type, V->type
    );
    CUDA_CHECK(cudaGetLastError());
}

static uint32_t ggml_cuda_fp_meta_count_bound(int64_t meta_words) {
    if (meta_words <= GGML_FLASHPREFILL_HEADER_WORDS) return 0u;
    uint64_t b = ((uint64_t) meta_words - GGML_FLASHPREFILL_HEADER_WORDS) / GGML_FLASHPREFILL_FRAG_WORDS;
    return b > UINT32_MAX ? UINT32_MAX : (uint32_t) b;
}

static uint32_t ggml_cuda_fp_select_grid_x(int64_t meta_words) {
    uint64_t b = (uint64_t) ggml_cuda_fp_meta_count_bound(meta_words) + 1u;
    if (b < 1u) b = 1u;
    return b > 1024u ? 1024u : (uint32_t) b;
}

void ggml_cuda_flash_prefill_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * pool = dst->src[1];
    const ggml_tensor * MT = dst->src[2];

    ggml_cuda_fp_params pp;
    GGML_ASSERT(ggml_cuda_fp_unpack_params(dst, GGML_FLASHPREFILL_OP_SELECT, pp));

    const uint32_t bound = ggml_cuda_fp_meta_count_bound(MT->ne[0]);
    const uint32_t dk = (uint32_t) pp.dk;
    const uint32_t dv_cap = (uint32_t) pp.dv;
    const uint32_t f_cap = (uint32_t) pool->ne[2];
    const uint32_t r_cap = bound;
    const uint32_t u_cap = bound;
    const uint32_t c_cap = MT->ne[0] > GGML_FLASHPREFILL_HEADER_WORDS ? (uint32_t)(MT->ne[0] - GGML_FLASHPREFILL_HEADER_WORDS) : 0u;
    const uint32_t meta_words = (uint32_t) MT->ne[0];
    const uint32_t plan_words = (uint32_t) dst->ne[0];
    const uint32_t q_groups_cap = (uint32_t) Q->ne[1];
    const uint32_t q_heads_cap = (uint32_t) Q->ne[2];
    const uint32_t hkv_cap = (uint32_t) pool->ne[1];
    const uint32_t exact_all = (uint32_t) pp.exact_all;

    // Phase 0: 1 block, 1 thread
    flash_prefill_select_phase0_kernel<<<1, 1, 0, stream>>>(
        (const int32_t *) MT->data,
        (int32_t *) dst->data,
        dk, dv_cap, f_cap, r_cap, u_cap, c_cap,
        meta_words, plan_words,
        q_groups_cap, q_heads_cap, hkv_cap, exact_all
    );
    CUDA_CHECK(cudaGetLastError());

    // Phase 1: grid(select_grid_x, hkv_cap)
    dim3 grid1(ggml_cuda_fp_select_grid_x(MT->ne[0]), hkv_cap, 1);
    flash_prefill_select_phase1_kernel<<<grid1, 1, 0, stream>>>(
        (const float *) Q->data,
        (const float *) pool->data,
        (const int32_t *) MT->data,
        (int32_t *) dst->data,
        dk, dv_cap, f_cap, r_cap, u_cap, c_cap,
        meta_words, plan_words,
        Q->nb[1], Q->nb[2],
        q_groups_cap, q_heads_cap, hkv_cap, exact_all,
        pp.alpha, pp.scale, pp.softcap
    );
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_prefill_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * pool = dst->src[3];
    const ggml_tensor * plan = dst->src[4];
    const ggml_tensor * MT = dst->src[5];
    const ggml_tensor * sinks = dst->src[6];

    ggml_cuda_fp_params pp;
    GGML_ASSERT(ggml_cuda_fp_unpack_params(dst, GGML_FLASHPREFILL_OP_ATTN, pp));

    const uint32_t bound = ggml_cuda_fp_meta_count_bound(MT->ne[0]);
    const uint32_t dk = (uint32_t) pp.dk;
    const uint32_t dv = (uint32_t) pp.dv;
    const uint32_t meta_words = (uint32_t) MT->ne[0];
    const uint32_t plan_words = (uint32_t) plan->ne[0];
    const uint32_t n_kv = (uint32_t) K->ne[1];
    const uint32_t f_cap = (uint32_t) pool->ne[2];
    const uint32_t r_cap = bound;
    const uint32_t u_cap = bound;
    const uint32_t c_cap = MT->ne[0] > GGML_FLASHPREFILL_HEADER_WORDS ? (uint32_t)(MT->ne[0] - GGML_FLASHPREFILL_HEADER_WORDS) : 0u;
    const uint32_t n_output_cap = (uint32_t) dst->ne[2];
    const uint32_t q_groups_cap = (uint32_t) Q->ne[1];
    const uint32_t q_heads_cap = (uint32_t) Q->ne[2];
    const uint32_t hkv_cap = (uint32_t) pool->ne[1];
    const uint32_t sinks_present = sinks ? 1u : 0u;
    const uint32_t mean_corr = (uint32_t) pp.mean_correction;

    const uint64_t total_rows = (uint64_t) dst->ne[2] * (uint64_t) dst->ne[1];
    constexpr int WARPS_PER_BLOCK = 4;
    const uint32_t n_blocks = (uint32_t)((total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    flash_prefill_attn_kernel<WARPS_PER_BLOCK><<<n_blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
        (const float *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        (const float *) pool->data,
        (int32_t *) plan->data,
        (const int32_t *) MT->data,
        sinks ? (const float *) sinks->data : nullptr,
        (float *) dst->data,
        dk, dv, meta_words, plan_words, n_kv,
        f_cap, r_cap, u_cap, c_cap, n_output_cap,
        Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2],
        q_groups_cap, q_heads_cap, hkv_cap,
        sinks_present, mean_corr,
        pp.scale, pp.softcap,
        K->type, V->type
    );
    CUDA_CHECK(cudaGetLastError());
}
