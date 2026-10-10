#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

// Same epoch/extent and per-workgroup xor contract as the established checker.
// Combine vectors BEFORE the horizontal xor, instead of crossing an AVX2
// function boundary and horizontally reducing each individual 32-byte load.
__attribute__((target("avx2")))
inline bool vk_tp5_qcheck_avx2(const uint32_t * entries, const uint32_t * words,
                              bool f16, uint32_t epoch, size_t n_words, uint32_t rank_dim,
                              uint32_t * pending_workgroup = nullptr, uint32_t rows_per_wg = 16u) {
    if (f16 && ((rows_per_wg!=4u && rows_per_wg!=8u && rows_per_wg!=16u) || rank_dim%rows_per_wg)) return false;
    const uint32_t n_wg = f16 ? rank_dim / rows_per_wg : 1u;
    const size_t vector_words = f16 ? rank_dim / 2u : n_words;
    if (!n_wg || !vector_words) return false;
    for (uint32_t wg = 0; wg < n_wg; ++wg) {
        std::atomic_thread_fence(std::memory_order_acquire);
        alignas(16) uint32_t entry[4];
        _mm_store_si128((__m128i *)entry,_mm_load_si128((const __m128i *)(entries+4u*wg)));
        if (entry[1] != epoch || entry[2] != n_words) {
            if (pending_workgroup) *pending_workgroup = wg;
            return false;
        }
        __m256i accum = _mm256_setzero_si256();
        uint32_t scalar = 0;
        const size_t step_words = f16 ? rows_per_wg/2u : n_words;
        for (size_t vector=0;vector<n_words/vector_words;++vector) {
            const uint32_t * p = words+vector*vector_words+(f16?wg*step_words:0u);
            size_t i=0;
            for (;i+8u<=step_words;i+=8u) accum=_mm256_xor_si256(accum,_mm256_loadu_si256((const __m256i *)(p+i)));
            for (;i<step_words;++i) scalar^=p[i];
        }
        __m128i x=_mm_xor_si128(_mm256_castsi256_si128(accum),_mm256_extracti128_si256(accum,1));
        x=_mm_xor_si128(x,_mm_shuffle_epi32(x,_MM_SHUFFLE(1,0,3,2)));
        x=_mm_xor_si128(x,_mm_shuffle_epi32(x,_MM_SHUFFLE(2,3,0,1)));
        scalar^=(uint32_t)_mm_cvtsi128_si32(x);
        if (scalar != entry[0]) {
            if (pending_workgroup) *pending_workgroup=wg;
            return false;
        }
    }
    return true;
}
