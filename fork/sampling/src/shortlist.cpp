#include "tq/sampling/shortlist.h"

#include <algorithm>
#include <cmath>

#if !defined(TQ_SAMPLING_SCALAR_ONLY) && defined(__x86_64__) && \
        (defined(__GNUC__) || defined(__clang__))
#define TQ_SAMPLING_HAS_AVX2_SCAN 1
#include <immintrin.h>
#endif

namespace tq::sampling {
namespace {

struct greater_logit {
    bool operator()(const candidate & left, const candidate & right) const {
        return left.logit > right.logit;
    }
};

void replace_minimum(std::vector<candidate> & output, std::int32_t id, float logit) {
    std::pop_heap(output.begin(), output.end(), greater_logit{});
    output.back() = {id, logit};
    std::push_heap(output.begin(), output.end(), greater_logit{});
}

bool scan_scalar(const float * logits, std::int32_t count, std::int32_t first,
                 std::vector<candidate> & output) {
    for (std::int32_t i = first; i < count; ++i) {
        const float value = logits[i];
        if (std::isnan(value)) {
            return false;
        }
        if (value > output.front().logit) {
            replace_minimum(output, i, value);
        }
    }
    return true;
}

bool scan_greedy_scalar(const float * logits, std::int32_t count, std::int32_t first,
                        candidate & selected) {
    for (std::int32_t i = first; i < count; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) {
            return false;
        }
        if (value > selected.logit) {
            selected = {i, value};
        }
    }
    return true;
}

#if defined(TQ_SAMPLING_HAS_AVX2_SCAN)
__attribute__((target("avx2")))
bool scan_avx2(const float * logits, std::int32_t count, std::int32_t first,
               std::vector<candidate> & output) {
    std::int32_t i = first;
    // Subtract before the check to avoid signed overflow near INT32_MAX.
    for (; count - i >= 8; i += 8) {
        const __m256 values = _mm256_loadu_ps(logits + i);
        if (_mm256_movemask_ps(_mm256_cmp_ps(values, values, _CMP_UNORD_Q))) {
            return false;
        }
        const __m256 threshold = _mm256_set1_ps(output.front().logit);
        unsigned mask = static_cast<unsigned>(
            _mm256_movemask_ps(_mm256_cmp_ps(values, threshold, _CMP_GT_OQ)));
        while (mask != 0) {
            const auto lane = static_cast<std::int32_t>(__builtin_ctz(mask));
            mask &= mask - 1;
            const float value = logits[i + lane];
            // A previous lane may have raised the heap threshold.
            if (value > output.front().logit) {
                replace_minimum(output, i + lane, value);
            }
        }
    }
    return scan_scalar(logits, count, i, output);
}

__attribute__((target("avx2")))
bool scan_greedy_avx2(const float * logits, std::int32_t count, candidate & selected) {
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    std::int32_t i = 0;
    for (; count - i >= 8; i += 8) {
        const __m256 values = _mm256_loadu_ps(logits + i);
        const __m256i nonfinite = _mm256_cmpeq_epi32(
            _mm256_and_si256(_mm256_castps_si256(values), exponent), exponent);
        if (_mm256_movemask_ps(_mm256_castsi256_ps(nonfinite))) {
            return false;
        }
        const __m256 threshold = _mm256_set1_ps(selected.logit);
        unsigned mask = static_cast<unsigned>(
            _mm256_movemask_ps(_mm256_cmp_ps(values, threshold, _CMP_GT_OQ)));
        while (mask != 0) {
            const auto lane = static_cast<std::int32_t>(__builtin_ctz(mask));
            mask &= mask - 1;
            const float value = logits[i + lane];
            // Visit ids in order. An earlier lane can raise the maximum.
            if (value > selected.logit) {
                selected = {i + lane, value};
            }
        }
    }
    return scan_greedy_scalar(logits, count, i, selected);
}
#endif

} // namespace

result select(const float * logits, std::int32_t count, int k) {
    if (logits == nullptr || count < 1 || k < 2 || k > 128) {
        return {};
    }

    result selected;
    const auto size = static_cast<std::int32_t>(std::min<std::int32_t>(count, k));
    selected.candidates.reserve(static_cast<std::size_t>(size));
    for (std::int32_t i = 0; i < size; ++i) {
        if (std::isnan(logits[i])) {
            return {};
        }
        selected.candidates.push_back({i, logits[i]});
        std::push_heap(selected.candidates.begin(), selected.candidates.end(), greater_logit{});
    }

    bool valid = false;
#if defined(TQ_SAMPLING_HAS_AVX2_SCAN)
    if (__builtin_cpu_supports("avx2")) {
        valid = scan_avx2(logits, count, size, selected.candidates);
    } else
#endif
    {
        valid = scan_scalar(logits, count, size, selected.candidates);
    }
    if (!valid) {
        return {};
    }

    std::sort_heap(selected.candidates.begin(), selected.candidates.end(), greater_logit{});
    const float maximum = selected.candidates.front().logit;
    if (!std::isfinite(maximum) ||
            (selected.candidates.size() > 1 && maximum <= selected.candidates[1].logit)) {
        return {};
    }
    selected.usable = true;
    return selected;
}

greedy_result select_finite_greedy(const float * logits, std::int32_t count) {
    if (logits == nullptr || count < 1) {
        return {};
    }
    candidate selected{0, logits[0]};
    bool valid = false;
#if defined(TQ_SAMPLING_HAS_AVX2_SCAN)
    if (__builtin_cpu_supports("avx2")) {
        valid = scan_greedy_avx2(logits, count, selected);
    } else
#endif
    {
        valid = scan_greedy_scalar(logits, count, 0, selected);
    }
    return valid ? greedy_result{true, selected} : greedy_result{};
}

} // namespace tq::sampling
