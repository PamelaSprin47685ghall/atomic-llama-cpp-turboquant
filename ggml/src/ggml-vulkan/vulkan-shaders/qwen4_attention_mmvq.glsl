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

void qwen_mmvq_iter_multi(uint matrix, uint first_row, uint num_rows, uint col,
                          inout float a[4][NUM_ROWS_QUANT], inout float b[4][NUM_ROWS_QUANT]) {
    const uint ib = col / 32u;
    const uint half_block = (col % 32u) / 16u;

    // 1. Preload multi-token activations into registers (single read pass, zero duplication)
    int32_t act_qs[4][4];
    vec2    act_ds[4];
    [[unroll]] for (uint t = 0u; t < p.n_tokens; ++t) {
        const uint tok_ib = t * (p.width / 32u) + ib;
        act_ds[t] = vec2(common_q8[tok_ib / 4u].ds[tok_ib % 4u]);
        [[unroll]] for (uint q = 0u; q < 4u; ++q) {
            act_qs[t][q] = common_q8[tok_ib / 4u].qs[(tok_ib % 4u) * 8u + half_block * 4u + q];
        }
    }

    // 2. Weights unpacked ONCE per matrix row and broadcast across all tokens simultaneously
    [[unroll]] for (uint r = 0u; r < num_rows; ++r) {
        const uint weight_block = ((first_row + r) * p.width + col) / 32u;
        if (matrix == 2u) {
            const i32vec4 qs_a = qwen_q6_a_repack4(weight_block, half_block * 4u);
            const float d_scale_a = qwen_q6_a_d_scale(weight_block, half_block * 4u);
            const i32vec4 qs_b = qwen_q6_b_repack4(weight_block, half_block * 4u);
            const float d_scale_b = qwen_q6_b_d_scale(weight_block, half_block * 4u);

            [[unroll]] for (uint t = 0u; t < p.n_tokens; ++t) {
                int32_t sum_a = dotPacked4x8EXT(qs_a.x, act_qs[t][0]) +
                                dotPacked4x8EXT(qs_a.y, act_qs[t][1]) +
                                dotPacked4x8EXT(qs_a.z, act_qs[t][2]) +
                                dotPacked4x8EXT(qs_a.w, act_qs[t][3]);
                a[t][r] += float(act_ds[t].x) * d_scale_a * float(sum_a);

                int32_t sum_b = dotPacked4x8EXT(qs_b.x, act_qs[t][0]) +
                                dotPacked4x8EXT(qs_b.y, act_qs[t][1]) +
                                dotPacked4x8EXT(qs_b.z, act_qs[t][2]) +
                                dotPacked4x8EXT(qs_b.w, act_qs[t][3]);
                b[t][r] += float(act_ds[t].x) * d_scale_b * float(sum_b);
            }
        } else if (matrix == 0u) {
            const i32vec4 qs_0 = qwen_q5_0_repack4(weight_block, half_block * 4u);
            const vec2 dm_scale = qwen_q5_0_dm_scale(weight_block, half_block * 4u);

            [[unroll]] for (uint t = 0u; t < p.n_tokens; ++t) {
                int32_t sum_0 = dotPacked4x8EXT(qs_0.x, act_qs[t][0]) +
                                dotPacked4x8EXT(qs_0.y, act_qs[t][1]) +
                                dotPacked4x8EXT(qs_0.z, act_qs[t][2]) +
                                dotPacked4x8EXT(qs_0.w, act_qs[t][3]);
                a[t][r] += float(act_ds[t].x) * dm_scale.x * float(sum_0) - dm_scale.y * float(act_ds[t].y / 2.0);
            }
        }
#ifdef GDN_PROJECTIONS
        else {
            const i32vec4 qs_1 = qwen_q5_1_repack4(weight_block, half_block * 4u);
            const vec2 dm_scale = qwen_q5_1_dm_scale(weight_block, half_block * 4u);

            [[unroll]] for (uint t = 0u; t < p.n_tokens; ++t) {
                int32_t sum_1 = dotPacked4x8EXT(qs_1.x, act_qs[t][0]) +
                                dotPacked4x8EXT(qs_1.y, act_qs[t][1]) +
                                dotPacked4x8EXT(qs_1.z, act_qs[t][2]) +
                                dotPacked4x8EXT(qs_1.w, act_qs[t][3]);
                a[t][r] += float(act_ds[t].x) * dm_scale.x * float(sum_1) - dm_scale.y * float(act_ds[t].y / 2.0);
            }
        }
#endif
    }
}

void qwen_compute_mmvq(uint matrix, uint first_row, uint num_rows, uint tid) {
    float a[4][NUM_ROWS_QUANT];
    float b[4][NUM_ROWS_QUANT];
    [[unroll]] for (uint t = 0u; t < 4u; ++t) {
        [[unroll]] for (uint r = 0u; r < NUM_ROWS_QUANT; ++r) {
            a[t][r] = 0.0;
            b[t][r] = 0.0;
        }
    }

    const uint stride = 16u * BLOCK_SIZE;
    uint col = 16u * tid;
    uint remaining = p.width / stride;
    if (remaining * stride + col < p.width) ++remaining;
    for (; remaining >= 4u; remaining -= 4u) {
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            qwen_mmvq_iter_multi(matrix, first_row, num_rows, col, a, b);
            col += stride;
        }
    }
    if (remaining >= 2u) {
        qwen_mmvq_iter_multi(matrix, first_row, num_rows, col, a, b);
        col += stride;
        qwen_mmvq_iter_multi(matrix, first_row, num_rows, col, a, b);
        col += stride;
        remaining -= 2u;
    }
    if (remaining != 0u) qwen_mmvq_iter_multi(matrix, first_row, num_rows, col, a, b);

    [[unroll]] for (uint t = 0u; t < p.n_tokens; ++t) {
        [[unroll]] for (uint r = 0u; r < num_rows; ++r) {
            a[t][r] = subgroupAdd(a[t][r]);
            b[t][r] = subgroupAdd(b[t][r]);
        }
        if (tid == 0u) {
            [[unroll]] for (uint r = 0u; r < num_rows; ++r) {
                if (matrix == 2u) {
                    OUT_Q6_A[t * OUT_Q6_A_STRIDE + first_row + r] = a[t][r];
                    OUT_Q6_B[t * OUT_Q6_B_STRIDE + first_row + r] = b[t][r];
                } else if (matrix == 0u) {
                    out0[t * p.rows0 + first_row + r] = a[t][r];
                }
#ifdef GDN_PROJECTIONS
                else {
                    out1[t * p.rows1 + first_row + r] = a[t][r];
                }
#endif
            }
        }
    }
}
