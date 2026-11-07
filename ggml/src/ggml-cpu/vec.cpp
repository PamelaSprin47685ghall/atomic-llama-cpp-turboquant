#include "vec.h"

#include <cassert>
#include <cstring>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
   assert(nrc == 1);
   GGML_UNUSED(nrc);
   GGML_UNUSED(bx);
   GGML_UNUSED(by);
   GGML_UNUSED(bs);

#if defined(GGML_SIMD)
    float sumf = 0.0f;

    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers

        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t sum1 = svdup_n_f32(0.0f);
        svfloat32_t sum2 = svdup_n_f32(0.0f);
        svfloat32_t sum3 = svdup_n_f32(0.0f);
        svfloat32_t sum4 = svdup_n_f32(0.0f);
        svfloat32_t sum5 = svdup_n_f32(0.0f);
        svfloat32_t sum6 = svdup_n_f32(0.0f);
        svfloat32_t sum7 = svdup_n_f32(0.0f);
        svfloat32_t sum8 = svdup_n_f32(0.0f);
        svfloat32_t ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8;
        svfloat32_t ay1,ay2,ay3,ay4,ay5,ay6,ay7,ay8;
        for (int i = 0; i < np; i += ggml_f32_step) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            sum2 = GGML_F32_VEC_FMA(sum2, ax2, ay2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            sum3 = GGML_F32_VEC_FMA(sum3, ax3, ay3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            sum4 = GGML_F32_VEC_FMA(sum4, ax4, ay4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            sum5 = GGML_F32_VEC_FMA(sum5, ax5, ay5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            sum6 = GGML_F32_VEC_FMA(sum6, ax6, ay6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            sum7 = GGML_F32_VEC_FMA(sum7, ax7, ay7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            sum8 = GGML_F32_VEC_FMA(sum8, ax8, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        const int np2 = (n & ~(ggml_f32_epr - 1));
        for (int i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmla on available elements only
        if (np2 < n) {
            svbool_t pg = svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            sum1 = svmla_f32_m(pg, sum1, ax1, ay1);
        }
        // reduce sum1,sum2 to sum1
        GGML_F32_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
    #elif defined(__riscv_v_intrinsic)
        int vl = __riscv_vsetvlmax_e32m8();
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m8_t vsum;
        vfloat32m8_t ax;
        vfloat32m8_t ay;
        vsum = __riscv_vfmv_v_f_f32m8_tu(vsum, 0.0f, vl);
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            ax = __riscv_vle32_v_f32m8_tu(ax, &x[i], vl);
            ay = __riscv_vle32_v_f32m8_tu(ay, &y[i], vl);
            vsum = __riscv_vfmacc_vv_f32m8_tu(vsum, ax, ay, vl);
        }
        vl = __riscv_vsetvlmax_e32m8();
        vs = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vs, vl);
        sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #endif
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

// Match the native dot order at a padded width without reading the tail.
// The omitted weights and values are zero; only the last SIMD block needs a
// temporary copy. Full zero blocks do not change finite accumulators.
void ggml_vec_dot_f32_padded(int n, int padded, float * s, const float * x, const float * y) {
    GGML_ASSERT(n >= 0 && padded >= n);
    if (n == padded) {
        ggml_vec_dot_f32(n, s, 0, x, 0, y, 0, 1);
        return;
    }
#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
    const int epr = ggml_cpu_get_sve_cnt() * 8 / 32;
    const int step = 8 * epr;
    const int ref_np = padded & ~(step - 1);
    const int ref_np2 = padded & ~(epr - 1);
    svfloat32_t s1 = svdup_n_f32(0.0f), s2 = svdup_n_f32(0.0f);
    svfloat32_t s3 = svdup_n_f32(0.0f), s4 = svdup_n_f32(0.0f);
    svfloat32_t s5 = svdup_n_f32(0.0f), s6 = svdup_n_f32(0.0f);
    svfloat32_t s7 = svdup_n_f32(0.0f), s8 = svdup_n_f32(0.0f);
#define GGML_SVE_PAD_DOT(acc, pos) do { \
        const int idx = (pos); \
        if (idx < n) { \
            const svbool_t pg = svwhilelt_b32(idx, n); \
            const svfloat32_t ax = svld1_f32(pg, x + idx); \
            const svfloat32_t ay = svld1_f32(pg, y + idx); \
            acc = svsel_f32(pg, GGML_F32_VEC_FMA(acc, ax, ay), acc); \
        } \
    } while (0)
    for (int i = 0; i < ref_np && i < n; i += step) {
        GGML_SVE_PAD_DOT(s1, i);           GGML_SVE_PAD_DOT(s2, i + epr);
        GGML_SVE_PAD_DOT(s3, i + 2*epr);   GGML_SVE_PAD_DOT(s4, i + 3*epr);
        GGML_SVE_PAD_DOT(s5, i + 4*epr);   GGML_SVE_PAD_DOT(s6, i + 5*epr);
        GGML_SVE_PAD_DOT(s7, i + 6*epr);   GGML_SVE_PAD_DOT(s8, i + 7*epr);
    }
    for (int i = ref_np; i < ref_np2 && i < n; i += epr) {
        GGML_SVE_PAD_DOT(s1, i);
    }
#undef GGML_SVE_PAD_DOT
    if (ref_np2 < padded && ref_np2 < n) {
        const svbool_t pg = svwhilelt_b32(ref_np2, n);
        const svfloat32_t ax = svld1_f32(pg, x + ref_np2);
        const svfloat32_t ay = svld1_f32(pg, y + ref_np2);
        s1 = svmla_f32_m(pg, s1, ax, ay);
    }
    float sumf = 0.0f;
    GGML_F32_VEC_REDUCE(sumf, s1, s2, s3, s4, s5, s6, s7, s8);
    *s = sumf;
#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
    int vl = __riscv_vsetvlmax_e32m8();
    vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m8_t sum, ax, ay;
    sum = __riscv_vfmv_v_f_f32m8_tu(sum, 0.0f, vl);
    for (int i = 0; i < n;) {
        const int ref_vl = __riscv_vsetvl_e32m8(padded - i);
        vl = __riscv_vsetvl_e32m8(MIN(ref_vl, n - i));
        ax = __riscv_vle32_v_f32m8_tu(ax, x + i, vl);
        ay = __riscv_vle32_v_f32m8_tu(ay, y + i, vl);
        sum = __riscv_vfmacc_vv_f32m8_tu(sum, ax, ay, vl);
        i += ref_vl;
    }
    vl = __riscv_vsetvlmax_e32m8();
    vs = __riscv_vfredusum_vs_f32m8_f32m1(sum, vs, vl);
    *s = __riscv_vfmv_f_s_f32m1_f32(vs);
#elif defined(GGML_SIMD)
    const int ref_np = padded & ~(GGML_F32_STEP - 1);
    int np = MIN(ref_np, n & ~(GGML_F32_STEP - 1));
    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; ++j) {
            const GGML_F32_VEC ax = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            const GGML_F32_VEC ay = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            sum[j] = GGML_F32_VEC_FMA(sum[j], ax, ay);
        }
    }
    if (ref_np > n && np < n) {
        float tx[GGML_F32_STEP] = {};
        float ty[GGML_F32_STEP] = {};
        std::memcpy(tx, x + np, (size_t) (n - np) * sizeof(float));
        std::memcpy(ty, y + np, (size_t) (n - np) * sizeof(float));
        for (int j = 0; j < GGML_F32_ARR; ++j) {
            const GGML_F32_VEC ax = GGML_F32_VEC_LOAD(tx + j * GGML_F32_EPR);
            const GGML_F32_VEC ay = GGML_F32_VEC_LOAD(ty + j * GGML_F32_EPR);
            sum[j] = GGML_F32_VEC_FMA(sum[j], ax, ay);
        }
        np = n;
    }
    float sumf = 0.0f;
    GGML_F32_VEC_REDUCE(sumf, sum);
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }
    *s = sumf;
#else
    ggml_vec_dot_f32(n, s, 0, x, 0, y, 0, 1);
#endif
}

void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int i = 0;
    ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfwma)
    size_t vl = __riscv_vsetvlmax_e32m4();

    // initialize accumulators to all zeroes
    vfloat32m4_t vsum0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vsum1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    // calculate step size
    const size_t epr = __riscv_vsetvlmax_e16m2();
    const size_t step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16 *)&x[i], epr);
        vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((const __bf16 *)&y[i], epr);
        vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, epr);
        __asm__ __volatile__ ("" ::: "memory");

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16 *)&x[i + epr], epr);
        vbfloat16m2_t ay1 = __riscv_vle16_v_bf16m2((const __bf16 *)&y[i + epr], epr);
        vsum1 = __riscv_vfwmaccbf16_vv_f32m4(vsum1, ax1, ay1, epr);
        __asm__ __volatile__ ("" ::: "memory");
    }

    // accumulate in 1 register
    vsum0 = __riscv_vfadd_vv_f32m4(vsum0, vsum1, vl);

    // leftovers
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16 *)&x[i], vl);
        vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((const __bf16 *)&y[i], vl);
        vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, vl);
    }

    // reduce
    vl = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t redsum = __riscv_vfredusum_vs_f32m4_f32m1(vsum0, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
    sumf += __riscv_vfmv_f_s_f32m1_f32(redsum);

#elif defined(__POWER9_VECTOR__) || defined(__VXE__) || defined(__VXE2__)
    const int np = (n & ~(GGML_BF16_STEP - 1));
    if (np > 0) {
        GGML_F32_VEC sum[4] = {GGML_F32_VEC_ZERO};
        for (; i < np; i += GGML_BF16_STEP) {
            GGML_BF16_VEC vx0 = GGML_BF16_VEC_LOAD(x + i);
            GGML_BF16_VEC vx1 = GGML_BF16_VEC_LOAD(x + i + 8);
            GGML_BF16_VEC vy0 = GGML_BF16_VEC_LOAD(y + i);
            GGML_BF16_VEC vy1 = GGML_BF16_VEC_LOAD(y + i + 8);
            GGML_BF16_FMA_LO(sum[0], vx0, vy0);
            GGML_BF16_FMA_HI(sum[1], vx0, vy0);
            GGML_BF16_FMA_LO(sum[2], vx1, vy1);
            GGML_BF16_FMA_HI(sum[3], vx1, vy1);
        }
        GGML_F32x4_REDUCE_4(sumf, sum[0], sum[1], sum[2], sum[3]);
    }
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;


#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        const int ggml_f16_epr = svcnth();
        const int ggml_f16_step = 8 * ggml_f16_epr;
        const int np = n - (n % ggml_f16_step);
        const int np2 = n - (n % ggml_f16_epr);

        svfloat32_t sum1_lo = svdup_n_f32(0.0f);
        svfloat32_t sum1_hi = svdup_n_f32(0.0f);
        svfloat32_t sum2_lo = svdup_n_f32(0.0f);
        svfloat32_t sum2_hi = svdup_n_f32(0.0f);
        svfloat32_t sum3_lo = svdup_n_f32(0.0f);
        svfloat32_t sum3_hi = svdup_n_f32(0.0f);
        svfloat32_t sum4_lo = svdup_n_f32(0.0f);
        svfloat32_t sum4_hi = svdup_n_f32(0.0f);

        for (int i = 0; i < np; i += ggml_f16_step) {
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_LOAD(x + i + 0 * ggml_f16_epr, 0), GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0));
            ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_LOAD(x + i + 1 * ggml_f16_epr, 1), GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1));
            ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_LOAD(x + i + 2 * ggml_f16_epr, 2), GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2));
            ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_LOAD(x + i + 3 * ggml_f16_epr, 3), GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3));
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_LOAD(x + i + 4 * ggml_f16_epr, 4), GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4));
            ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_LOAD(x + i + 5 * ggml_f16_epr, 5), GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5));
            ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_LOAD(x + i + 6 * ggml_f16_epr, 6), GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6));
            ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_LOAD(x + i + 7 * ggml_f16_epr, 7), GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7));
        }

        for (int i = np; i < np2; i += ggml_f16_epr) {
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_LOAD(x + i, 0), GGML_F16x_VEC_LOAD(y + i, 0));
        }

        if (np2 < n) {
            const svbool_t pg = svwhilelt_b16(np2, n);
            const svfloat16_t rx = svld1_f16(pg, (const __fp16 *)(x + np2));
            const svfloat16_t ry = svld1_f16(pg, (const __fp16 *)(y + np2));

            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, rx, ry);
        }

        sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum2_lo);
        sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum2_hi);
        sum3_lo = svadd_f32_m(DEFAULT_PG32, sum3_lo, sum4_lo);
        sum3_hi = svadd_f32_m(DEFAULT_PG32, sum3_hi, sum4_hi);
        sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum3_lo);
        sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum3_hi);

        sumf = ggml_sve_sum_f32x2(sum1_lo, sum1_hi);
    #elif defined(__riscv_v_intrinsic)
        #if defined(__riscv_zvfh)
            int vl = __riscv_vsetvlmax_e32m2();
            vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m2_t vsum;
            vfloat16m1_t ax;
            vfloat16m1_t ay;
            vsum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
            for (int i = 0; i < n; i += vl) {
                vl = __riscv_vsetvl_e16m1(n - i);
                ax = __riscv_vle16_v_f16m1_tu(ax, (const _Float16 *)&x[i], vl);
                ay = __riscv_vle16_v_f16m1_tu(ay, (const _Float16 *)&y[i], vl);
                vsum = __riscv_vfwmacc_vv_f32m2_tu(vsum, ax, ay, vl);
            }
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t ac0 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(vsum, 0), __riscv_vget_v_f32m2_f32m1(vsum, 1), vl);
            vs = __riscv_vfredusum_vs_f32m1_f32m1(ac0, vs, vl);
            sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
        #else
            for (int i = 0; i < n; ++i) {
                sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
            }
        #endif // __riscv_zvfh
    #else
        const int np = (n & ~(GGML_F16_STEP - 1));

        GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

        GGML_F16_VEC ax[GGML_F16_ARR];
        GGML_F16_VEC ay[GGML_F16_ARR];

        for (int i = 0; i < np; i += GGML_F16_STEP) {
            for (int j = 0; j < GGML_F16_ARR; j++) {
                ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
                ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

                sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F16_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
        }
        // if you hit this, you are likely running outside the FP range
        assert(!isnan(sumf) && !isinf(sumf));
    #endif
#else
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
    }
#endif // GGML_SIMD

    *s = sumf;
}

void ggml_vec_dot_f16_padded(int n, int padded, float * s, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    GGML_ASSERT(n >= 0 && padded >= n);
    if (n == padded) {
        ggml_vec_dot_f16(n, s, 0, const_cast<ggml_fp16_t *>(x), 0, const_cast<ggml_fp16_t *>(y), 0, 1);
        return;
    }
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)
    const int ref_np = padded & ~(GGML_F16_STEP - 1);
    int np = MIN(ref_np, n & ~(GGML_F16_STEP - 1));
    GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };
    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; ++j) {
            const GGML_F16_VEC ax = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            const GGML_F16_VEC ay = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            sum[j] = GGML_F16_VEC_FMA(sum[j], ax, ay);
        }
    }
    if (ref_np > n && np < n) {
        ggml_fp16_t tx[GGML_F16_STEP] = {};
        ggml_fp16_t ty[GGML_F16_STEP] = {};
        std::memcpy(tx, x + np, size_t(n - np)*sizeof(*x));
        std::memcpy(ty, y + np, size_t(n - np)*sizeof(*y));
        for (int j = 0; j < GGML_F16_ARR; ++j) {
            const GGML_F16_VEC ax = GGML_F16_VEC_LOAD(tx + j*GGML_F16_EPR, j);
            const GGML_F16_VEC ay = GGML_F16_VEC_LOAD(ty + j*GGML_F16_EPR, j);
            sum[j] = GGML_F16_VEC_FMA(sum[j], ax, ay);
        }
        np = n;
    }
    ggml_float sumf = 0.0;
    GGML_F16_VEC_REDUCE(sumf, sum);
    for (int i = np; i < n; ++i) {
        sumf += (ggml_float) (GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
    }
    *s = sumf;
#elif !defined(GGML_SIMD)
    ggml_vec_dot_f16(n, s, 0, const_cast<ggml_fp16_t *>(x), 0, const_cast<ggml_fp16_t *>(y), 0, 1);
#else
    GGML_ABORT("padded F16 dot is not supported on this CPU");
#endif
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, ggml_v_silu(pg, svld1_f32(pg, x + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_silu(vld1q_f32(x + i)));
    }
#elif defined(__riscv_v_intrinsic)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vy = ggml_v_silu_m2(vx, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(ggml_v_silu(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(ggml_v_silu(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(ggml_v_silu(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
    }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, svmul_f32_x(pg, ggml_v_silu(pg, svld1_f32(pg, x + i)), svld1_f32(pg, g + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_silu(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#elif defined(__riscv_v_intrinsic)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vg = __riscv_vle32_v_f32m2(&g[i], vl);
        vfloat32m2_t vy = __riscv_vfmul_vv_f32m2(ggml_v_silu_m2(vx, vl), vg, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]) * g[i];
    }
}

ggml_float ggml_vec_cvar_f32(const int n, float * y, const float * x, const float mean) {
    int i = 0;
    ggml_float sum = 0;
// TODO: optimize to process the remaining elements in groups using the smaller vector sizes from AVX2 and SSE
// ref: https://github.com/ggml-org/llama.cpp/pull/15953#pullrequestreview-3310928344
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = _mm512_sub_ps(_mm512_loadu_ps(x + i),
                                   _mm512_set1_ps(mean));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(_mm512_mul_ps(val, val));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = _mm256_sub_ps(_mm256_loadu_ps(x + i),
                                   _mm256_set1_ps(mean));
        _mm256_storeu_ps(y + i, val);
        val = _mm256_mul_ps(val,val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = _mm_sub_ps(_mm_loadu_ps(x + i),
                                _mm_set1_ps(mean));
        _mm_storeu_ps(y + i, val);
        val = _mm_mul_ps(val, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif  // __AVX__ || __AVX2__ || __AVX512F__
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = vsubq_f32(vld1q_f32(x + i),
                                    vdupq_n_f32(mean));
        vst1q_f32(y + i, val);
        val = vmulq_f32(val, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#elif defined(__VXE__) || defined(__VXE2__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = vec_sub(vec_xl(0, x + i), vec_splats(mean));
        vec_xst(val, 0, y + i);
        val = vec_mul(val, val);
        sum += (ggml_float)vec_hsum_f32x4(val);
    }
#elif defined(__riscv_v_intrinsic)
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t val = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], vl), mean, vl);
        __riscv_vse32_v_f32m2(&y[i], val, vl);
        val = __riscv_vfmul_vv_f32m2(val, val, vl);
        vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);
    }
    sum = (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#endif
    for (; i < n; ++i) {
        float val = x[i] - mean;
        y[i] = val;
        val *= val;
        sum += (ggml_float)val;
    }
    return sum/n;
}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svfloat32_t val = ggml_v_expf(pg, svsub_f32_x(pg, svld1_f32(pg, x + i),
                                                svdup_n_f32_x(pg, max)));
        svst1_f32(pg, y + i, val);
        sum += (ggml_float)svaddv_f32(pg, val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#elif defined(__riscv_v_intrinsic)
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
    for (int avl; i < n; i += avl) {
        avl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t val = ggml_v_expf_m2(__riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], avl), max, avl), avl);
        __riscv_vse32_v_f32m2(&y[i], val, avl);
        vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, avl);
    }
    return (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (ggml_float)val;
        y[i] = val;
    }
    return sum;
}

// Preserve the padded SIMD exp and sum order for the live lanes. The final
// vector is copied to a small -inf-filled buffer, never read past x or y.
ggml_float ggml_vec_soft_max_f32_padded(int n, int padded, float * y, const float * x, float max) {
    GGML_ASSERT(n >= 0 && padded >= n);
    if (n == padded) {
        return ggml_vec_soft_max_f32(n, y, x, max);
    }
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    constexpr int lanes = 16;
#elif defined(__AVX2__) && defined(__FMA__)
    constexpr int lanes = 8;
#elif defined(__SSE2__) || (defined(__ARM_NEON) && defined(__aarch64__) && !defined(__ARM_FEATURE_SVE))
    constexpr int lanes = 4;
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    // SVE has at most 64 F32 lanes. Preserve the reference predicate on the
    // last vector and use -inf only for its missing live inputs.
    const int lanes = svcntw();
    GGML_ASSERT(lanes <= 64);
    const int np = n - n % lanes;
    ggml_float sum = ggml_vec_soft_max_f32(np, y, x, max);
    if (np < n) {
        float tx[64], ty[64];
        const int width = MIN(lanes, padded - np);
        std::memcpy(tx, x + np, (size_t) (n - np) * sizeof(float));
        for (int i = n - np; i < width; ++i) tx[i] = -INFINITY;
        sum += ggml_vec_soft_max_f32(width, ty, tx, max);
        std::memcpy(y + np, ty, (size_t) (n - np) * sizeof(float));
    }
    return sum;
#elif defined(__riscv_v_intrinsic)
    // RVV VLEN is at most 65536 bits; an m2 F32 vector has at most 4096 lanes.
    float tx[4096];
    GGML_ASSERT(__riscv_vsetvlmax_e32m2() <= 4096);
    vfloat64m1_t sum = __riscv_vfmv_v_f_f64m1(0, 1);
    for (int i = 0; i < n;) {
        const int vl = __riscv_vsetvl_e32m2(padded - i);
        const int live = MIN(vl, n - i);
        const float * src = x + i;
        if (live < vl) {
            std::memcpy(tx, src, (size_t) live * sizeof(float));
            for (int j = live; j < vl; ++j) tx[j] = -INFINITY;
            src = tx;
        }
        const vfloat32m2_t val = ggml_v_expf_m2(
            __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(src, vl), max, vl), vl);
        __riscv_vse32_v_f32m2(y + i, val, live);
        sum = __riscv_vfwredusum_vs_f32m2_f64m1(val, sum, vl);
        i += vl;
    }
    return (ggml_float) __riscv_vfmv_f_s_f64m1_f64(sum);
#else
    return ggml_vec_soft_max_f32(n, y, x, max);
#endif
#if (defined(__AVX512F__) && defined(__AVX512DQ__)) || (defined(__AVX2__) && defined(__FMA__)) || defined(__SSE2__) || (defined(__ARM_NEON) && defined(__aarch64__) && !defined(__ARM_FEATURE_SVE))
    const int ref_np = padded - padded % lanes;
    int np = MIN(ref_np, n - n % lanes);
    ggml_float sum = ggml_vec_soft_max_f32(np, y, x, max);
    if (ref_np > n && np < n) {
        float tx[lanes], ty[lanes];
        std::memcpy(tx, x + np, (size_t) (n - np) * sizeof(float));
        for (int i = n - np; i < lanes; ++i) tx[i] = -INFINITY;
        sum += ggml_vec_soft_max_f32(lanes, ty, tx, max);
        std::memcpy(y + np, ty, (size_t) (n - np) * sizeof(float));
        np = n;
    }
    for (int i = np; i < n; ++i) {
        const float val = expf(x[i] - max);
        sum += (ggml_float) val;
        y[i] = val;
    }
    return sum;
#endif
}

ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (ggml_float)expf(val);
    }
    return sum = (ggml_float)logf(sum);
}
