// Included by qwen4_attention_projections.comp after its buffer declarations.
// Keep the four-projection fusion; replace only selected contractions. All
// selected matrices share ONE canonical Q8_1 input prepared by the backend.
// Reuse the built-in MMVQ weight unpack/dot code, not another quant decoder.

#extension GL_EXT_integer_dot_product : require

layout(std430, binding = 9) readonly buffer CommonInputQ8 {
    block_q8_1_x4 common_q8[];
};
layout(std430, binding = 0) readonly buffer Weights0Packed32 {
    block_q5_K_packed32 weights0_packed32[];
};
#ifdef GDN_PROJECTIONS
layout(std430, binding = 1) readonly buffer Weights1Packed32 {
    block_q5_K_packed32 weights1_packed32[];
};
#endif

int32_t cache_b_qs[4];
vec2 cache_b_ds;

#define FLOAT_TYPE float
#define FLOAT_TYPEV2 vec2
#define DATA_A_Q5_K 1
#define data_a_packed32 weights0_packed32
#define repack4 qwen_q5_0_repack4
#define get_dm_scale qwen_q5_0_dm_scale
#define mmvq_dot_product qwen_q5_0_dot
#include "mul_mat_vecq_funcs.glsl"
#undef data_a_packed32
#undef repack4
#undef get_dm_scale
#undef mmvq_dot_product
#ifdef GDN_PROJECTIONS
#define data_a_packed32 weights1_packed32
#define repack4 qwen_q5_1_repack4
#define get_dm_scale qwen_q5_1_dm_scale
#define mmvq_dot_product qwen_q5_1_dot
#include "mul_mat_vecq_funcs.glsl"
#undef data_a_packed32
#undef repack4
#undef get_dm_scale
#undef mmvq_dot_product
#endif
#undef DATA_A_Q5_K

#define DATA_A_Q6_K 1
#define data_a WEIGHTS_Q6_A
#define data_a_packed16 WEIGHTS_Q6_A_PACKED16
#define repack4 qwen_q6_a_repack4
#define get_d_scale qwen_q6_a_d_scale
#define mmvq_dot_product qwen_q6_a_dot
#include "mul_mat_vecq_funcs.glsl"
#undef data_a
#undef data_a_packed16
#undef repack4
#undef get_d_scale
#undef mmvq_dot_product
#define data_a WEIGHTS_Q6_B
#define data_a_packed16 WEIGHTS_Q6_B_PACKED16
#define repack4 qwen_q6_b_repack4
#define get_d_scale qwen_q6_b_d_scale
#define mmvq_dot_product qwen_q6_b_dot
#include "mul_mat_vecq_funcs.glsl"
#undef data_a
#undef data_a_packed16
#undef repack4
#undef get_d_scale
#undef mmvq_dot_product
#undef DATA_A_Q6_K

void qwen_mmvq_iter(uint matrix, uint first_row, uint num_rows, uint col,
                    inout float a[NUM_ROWS_QUANT], inout float b[NUM_ROWS_QUANT]) {
        const uint ib = col / 32u;
        const uint half_block = (col % 32u) / 16u;
        cache_b_ds = vec2(common_q8[ib / 4u].ds[ib % 4u]);
        [[unroll]] for (uint q = 0u; q < 4u; ++q) {
            cache_b_qs[q] = common_q8[ib / 4u].qs[(ib % 4u) * 8u + half_block * 4u + q];
        }
        [[unroll]] for (uint r = 0u; r < num_rows; ++r) {
            const uint weight_block = ((first_row + r) * p.width + col) / 32u;
            if (matrix == 2u) {
                a[r] += qwen_q6_a_dot(weight_block, half_block);
                b[r] += qwen_q6_b_dot(weight_block, half_block);
            } else if (matrix == 0u) {
                a[r] += qwen_q5_0_dot(weight_block, half_block);
            }
#ifdef GDN_PROJECTIONS
            else {
                a[r] += qwen_q5_1_dot(weight_block, half_block);
            }
#endif
        }
}

// matrix=0/1: one Q5 matrix; matrix=2: the paired Q6 matrices. The
// workgroup/row distribution and all surrounding graph fusion stay unchanged.
void qwen_compute_mmvq(uint matrix, uint first_row, uint num_rows, uint tid) {
    float a[NUM_ROWS_QUANT];
    float b[NUM_ROWS_QUANT];
    [[unroll]] for (uint r = 0u; r < NUM_ROWS_QUANT; ++r) {
        a[r] = 0.0;
        b[r] = 0.0;
    }
    // Match the built-in MMVQ 4/2/1 K-loop, including each lane's tail.
    // This is finite arithmetic iteration, not a synchronization spin.
    const uint stride = 16u * BLOCK_SIZE;
    uint col = 16u * tid;
    uint remaining = p.width / stride;
    if (remaining * stride + col < p.width) ++remaining;
    for (; remaining >= 4u; remaining -= 4u) {
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            qwen_mmvq_iter(matrix, first_row, num_rows, col, a, b);
            col += stride;
        }
    }
    if (remaining >= 2u) {
        qwen_mmvq_iter(matrix, first_row, num_rows, col, a, b);
        col += stride;
        qwen_mmvq_iter(matrix, first_row, num_rows, col, a, b);
        col += stride;
        remaining -= 2u;
    }
    if (remaining != 0u) qwen_mmvq_iter(matrix, first_row, num_rows, col, a, b);
    [[unroll]] for (uint r = 0u; r < num_rows; ++r) {
        a[r] = subgroupAdd(a[r]);
        b[r] = subgroupAdd(b[r]);
    }
    // Pipeline creation requires one full subgroup per workgroup.
    if (tid == 0u) {
        [[unroll]] for (uint r = 0u; r < num_rows; ++r) {
            if (matrix == 2u) {
                OUT_Q6_A[first_row + r] = a[r];
                OUT_Q6_B[first_row + r] = b[r];
            } else if (matrix == 0u) {
                out0[first_row + r] = a[r];
            }
#ifdef GDN_PROJECTIONS
            else {
                out1[first_row + r] = a[r];
            }
#endif
        }
    }
}
