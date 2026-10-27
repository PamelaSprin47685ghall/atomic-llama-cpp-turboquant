#include "ordered-fold.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    include <emmintrin.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#    include <immintrin.h>
#    define TQ_F16C_TARGET __attribute__((target("avx2,f16c")))
#    define TQ_F16C_DISPATCH 1
#endif

namespace tq::tp5::detail {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "ordered reduction requires IEEE binary32");

fold_result ordered_fold_f32(const float * const inputs[5], float * output, std::size_t elements) noexcept {
    std::size_t element = 0;
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    const __m128i exponent = _mm_set1_epi32(0x7f800000);
    const auto nonfinite = [exponent](__m128 value) {
        return _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(value), exponent), exponent);
    };
    for (; elements - element >= 4; element += 4) {
        const __m128 rank0 = _mm_loadu_ps(inputs[0] + element);
        const __m128 rank1 = _mm_loadu_ps(inputs[1] + element);
        const __m128 rank2 = _mm_loadu_ps(inputs[2] + element);
        const __m128 rank3 = _mm_loadu_ps(inputs[3] + element);
        const __m128 rank4 = _mm_loadu_ps(inputs[4] + element);
        const __m128i invalid = _mm_or_si128(
            _mm_or_si128(nonfinite(rank0), nonfinite(rank1)),
            _mm_or_si128(_mm_or_si128(nonfinite(rank2), nonfinite(rank3)), nonfinite(rank4)));
        if (_mm_movemask_epi8(invalid)) {
            return fold_result::nonfinite_input;
        }
        // SIMD lanes are independent elements, never different ranks. Keep each
        // addition separate so every rank boundary rounds to binary32.
        __m128 sum = _mm_add_ps(rank0, rank1);
        sum = _mm_add_ps(sum, rank2);
        sum = _mm_add_ps(sum, rank3);
        sum = _mm_add_ps(sum, rank4);
        if (_mm_movemask_epi8(nonfinite(sum))) {
            return fold_result::nonfinite_sum;
        }
        _mm_storeu_ps(output + element, sum);
    }
#endif
    for (; element < elements; ++element) {
        float sum = inputs[0][element];
        if (!std::isfinite(sum)) {
            return fold_result::nonfinite_input;
        }
        for (std::size_t rank = 1; rank < 5; ++rank) {
            const float value = inputs[rank][element];
            if (!std::isfinite(value)) {
                return fold_result::nonfinite_input;
            }
            sum += value;
        }
        if (!std::isfinite(sum)) {
            return fold_result::nonfinite_sum;
        }
        output[element] = sum;
    }
    return fold_result::ok;
}

namespace {
constexpr std::uint32_t all_ranks = 31;

template <typename T>
std::size_t active_inputs(const T * const inputs[5], const T * active[5],
                          std::uint32_t mask) noexcept {
    std::size_t count = 0;
    for (std::size_t rank = 0; rank < 5; ++rank) {
        if (mask & (1u << rank)) {
            active[count++] = inputs[rank];
        }
    }
    return count;
}

bool half_nonfinite(std::uint16_t bits) noexcept {
    return (bits & 0x7c00u) == 0x7c00u;
}

float widen_half(std::uint16_t bits) noexcept {
    static_assert(sizeof(ggml_fp16_t) == sizeof(bits), "binary16 wire width differs");
    ggml_fp16_t value;
    std::memcpy(&value, &bits, sizeof(value));
    return ggml_fp16_to_fp32(value);
}

std::uint16_t narrow_half(float value) noexcept {
    const ggml_fp16_t half = ggml_fp32_to_fp16(value);
    std::uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

// IEEE additions cannot turn a nonfinite operand into a finite final sum.
// Keep successful folds to one final check; classify only a failed chunk.
fold_result classify_half_failure(const std::uint16_t * const active[5], std::size_t count,
                                  std::size_t first, std::size_t elements) noexcept {
    for (std::size_t rank = 0; rank < count; ++rank) {
        for (std::size_t i = first; i < first + elements; ++i) {
            if (half_nonfinite(active[rank][i])) {
                return fold_result::nonfinite_input;
            }
        }
    }
    return fold_result::nonfinite_sum;
}

fold_result classify_float_failure(const float * const active[5], std::size_t count,
                                   std::size_t first, std::size_t elements) noexcept {
    for (std::size_t rank = 0; rank < count; ++rank) {
        for (std::size_t i = first; i < first + elements; ++i) {
            if (!std::isfinite(active[rank][i])) {
                return fold_result::nonfinite_input;
            }
        }
    }
    return fold_result::nonfinite_sum;
}

fold_result fold_half_scalar_active(const std::uint16_t * const active[5], std::size_t count,
                                    std::uint16_t * output, std::size_t elements) noexcept {
    for (std::size_t element = 0; element < elements; ++element) {
        float sum = widen_half(active[0][element]);
        for (std::size_t rank = 1; rank < count; ++rank) {
            sum += widen_half(active[rank][element]);
        }
        if (!std::isfinite(sum)) {
            return classify_half_failure(active, count, element, 1);
        }
        const auto packed = narrow_half(sum);
        if (half_nonfinite(packed)) {
            return fold_result::nonfinite_pack;
        }
        output[element] = packed;
    }
    return fold_result::ok;
}

fold_result copy_half_finite(const std::uint16_t * input, std::uint16_t * output,
                             std::size_t elements) noexcept {
    std::size_t element = 0;
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    const __m128i exponent = _mm_set1_epi16(0x7c00);
    for (; elements - element >= 8; element += 8) {
        const __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + element));
        if (_mm_movemask_epi8(_mm_cmpeq_epi16(_mm_and_si128(value, exponent), exponent))) {
            return fold_result::nonfinite_input;
        }
        _mm_storeu_si128(reinterpret_cast<__m128i *>(output + element), value);
    }
#endif
    for (; element < elements; ++element) {
        const auto value = input[element];
        if (half_nonfinite(value)) {
            return fold_result::nonfinite_input;
        }
        output[element] = value;
    }
    return fold_result::ok;
}

#if defined(TQ_F16C_DISPATCH)
TQ_F16C_TARGET
fold_result fold_half_vector_active(const std::uint16_t * const active[5], std::size_t count,
                                    std::uint16_t * output, std::size_t elements) noexcept {
    const __m128i half_exponent = _mm_set1_epi16(0x7c00);
    const __m256i float_exponent = _mm256_set1_epi32(0x7f800000);
    std::size_t element = 0;
    for (; elements - element >= 8; element += 8) {
        const __m128i first = _mm_loadu_si128(reinterpret_cast<const __m128i *>(active[0] + element));
        __m256 sum = _mm256_cvtph_ps(first);
        for (std::size_t rank = 1; rank < count; ++rank) {
            const __m128i next = _mm_loadu_si128(reinterpret_cast<const __m128i *>(active[rank] + element));
            sum = _mm256_add_ps(sum, _mm256_cvtph_ps(next));
        }
        const __m256i sum_bits = _mm256_castps_si256(sum);
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(
                _mm256_and_si256(sum_bits, float_exponent), float_exponent))) {
            return classify_half_failure(active, count, element, 8);
        }
        const __m128i packed = _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        if (_mm_movemask_epi8(_mm_cmpeq_epi16(_mm_and_si128(packed, half_exponent), half_exponent))) {
            return fold_result::nonfinite_pack;
        }
        _mm_storeu_si128(reinterpret_cast<__m128i *>(output + element), packed);
    }
    const std::uint16_t * tail[5]{};
    for (std::size_t rank = 0; rank < count; ++rank) {
        tail[rank] = active[rank] + element;
    }
    return fold_half_scalar_active(tail, count, output + element, elements - element);
}
#endif
}  // namespace

fold_result ordered_fold_f32_masked(const float * const inputs[5], float * output,
                                   std::size_t elements, std::uint32_t computed_mask) noexcept {
    if (computed_mask & ~all_ranks) {
        return fold_result::invalid_mask;
    }
    if (computed_mask == all_ranks) {
        return ordered_fold_f32(inputs, output, elements);
    }
    const float * active[5]{};
    const auto count = active_inputs(inputs, active, computed_mask);
    if (!count) {
        std::fill_n(output, elements, 0.0f);
        return fold_result::ok;
    }
    std::size_t element = 0;
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    const __m128i exponent = _mm_set1_epi32(0x7f800000);
    for (; elements - element >= 4; element += 4) {
        __m128 sum = _mm_loadu_ps(active[0] + element);
        for (std::size_t rank = 1; rank < count; ++rank) {
            sum = _mm_add_ps(sum, _mm_loadu_ps(active[rank] + element));
        }
        if (_mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(sum), exponent), exponent))) {
            return classify_float_failure(active, count, element, 4);
        }
        _mm_storeu_ps(output + element, sum);
    }
#endif
    for (; element < elements; ++element) {
        float sum = active[0][element];
        for (std::size_t rank = 1; rank < count; ++rank) {
            sum += active[rank][element];
        }
        if (!std::isfinite(sum)) {
            return classify_float_failure(active, count, element, 1);
        }
        output[element] = sum;
    }
    return fold_result::ok;
}

fold_result ordered_fold_f16_scalar(const std::uint16_t * const inputs[5], std::uint16_t * output,
                                   std::size_t elements, std::uint32_t computed_mask) noexcept {
    if (computed_mask & ~all_ranks) {
        return fold_result::invalid_mask;
    }
    const std::uint16_t * active[5]{};
    const auto count = active_inputs(inputs, active, computed_mask);
    if (!count) {
        std::fill_n(output, elements, std::uint16_t(0));
        return fold_result::ok;
    }
    if (count == 1) {
        return copy_half_finite(active[0], output, elements);
    }
    return fold_half_scalar_active(active, count, output, elements);
}

fold_result ordered_fold_f16(const std::uint16_t * const inputs[5], std::uint16_t * output,
                            std::size_t elements, std::uint32_t computed_mask) noexcept {
#if defined(TQ_F16C_DISPATCH)
    if ((computed_mask & ~all_ranks) == 0 && computed_mask && (computed_mask & (computed_mask - 1))) {
        static const bool available = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
        if (available) {
            const std::uint16_t * active[5]{};
            const auto count = active_inputs(inputs, active, computed_mask);
            return fold_half_vector_active(active, count, output, elements);
        }
    }
#endif
    return ordered_fold_f16_scalar(inputs, output, elements, computed_mask);
}

}  // namespace tq::tp5::detail
