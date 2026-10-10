#pragma once
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

// Only a unique finite maximum may bypass the ordinary greedy sampler chain.
// Ties and non-finite rows retain that chain's sorting/masking semantics.
inline llama_token common_unique_finite_argmax(const float * logits, llama_token n) {
    if (!logits || n <= 0) return LLAMA_TOKEN_NULL;
    float maximum = -std::numeric_limits<float>::infinity();
    llama_token best = LLAMA_TOKEN_NULL;
    bool unique = false;
    for (llama_token i = 0; i < n; ++i) {
        const float x = logits[i];
        if (std::isnan(x)) return LLAMA_TOKEN_NULL;
        if (x > maximum) {
            maximum = x;
            best = i;
            unique = true;
        } else if (x == maximum) {
            unique = false;
        }
    }
    return unique && std::isfinite(maximum) ? best : LLAMA_TOKEN_NULL;
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
inline bool common_greedy_heap_scan_avx2(const float * logits, llama_token n, llama_token first,
                                       std::vector<llama_token_data> & out) {
    const auto greater = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
    llama_token i = first;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(logits + i);
        if (_mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q))) return false;
        unsigned mask = (unsigned) _mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_set1_ps(out.front().logit), _CMP_GT_OQ));
        while (mask) {
            const unsigned lane = (unsigned) __builtin_ctz(mask);
            mask &= mask - 1;
            const float x = logits[i + lane];
            if (x <= out.front().logit) continue; // another lane may have raised the threshold
            std::pop_heap(out.begin(), out.end(), greater);
            out.back() = {llama_token(i + lane), x, 0};
            std::push_heap(out.begin(), out.end(), greater);
        }
    }
    for (; i < n; ++i) {
        const float x = logits[i];
        if (std::isnan(x)) return false;
        if (x <= out.front().logit) continue;
        std::pop_heap(out.begin(), out.end(), greater);
        out.back() = {i, x, 0};
        std::push_heap(out.begin(), out.end(), greater);
    }
    return true;
}
#endif

// Exact sorted top-k logits, built without an O(vocabulary) token-data array.
// The caller must prove top-k is the first active filter and then run the REAL
// sampler chain on this prefix (including dist's size==1/no-RNG behavior).
inline bool common_greedy_compact_top_k(const float * logits, llama_token n, int k,
                                      std::vector<llama_token_data> & out) {
    if (!logits || n < 1 || k < 2 || k > 128) return false;
    const size_t count = std::min<size_t>(n, k);
    out.clear();
    out.reserve(count);
    const auto greater = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };
    for (llama_token i = 0; i < (llama_token) count; ++i) {
        const float x = logits[i];
        if (std::isnan(x)) return false;
        out.push_back({i, x, 0});
        std::push_heap(out.begin(), out.end(), greater);
    }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2")) {
        if (!common_greedy_heap_scan_avx2(logits, n, (llama_token) count, out)) return false;
    } else
#endif
    {
        for (llama_token i = (llama_token) count; i < n; ++i) {
            const float x = logits[i];
            if (std::isnan(x)) return false;
            if (x > out.front().logit) {
                std::pop_heap(out.begin(), out.end(), greater);
                out.back() = {i, x, 0};
                std::push_heap(out.begin(), out.end(), greater);
            }
        }
    }
    std::sort_heap(out.begin(), out.end(), greater);
    return std::isfinite(out.front().logit) &&
           (out.size() == 1 || out[0].logit > out[1].logit);
}
