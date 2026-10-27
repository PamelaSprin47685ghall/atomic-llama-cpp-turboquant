#pragma once

#include <cstddef>
#include <cstdint>

namespace tq::tp5::detail {

enum class fold_result { ok, nonfinite_input, nonfinite_sum, nonfinite_pack, invalid_mask };

// Each element is (((rank0 + rank1) + rank2) + rank3) + rank4 in F32.
// The caller owns five valid input spans and one output span. All inputs and
// final sums must be finite; on failure, output is not ready for publication.
fold_result ordered_fold_f32(const float * const inputs[5], float * output, std::size_t elements) noexcept;

// A zero bit is a provider-proved exact-zero contribution. All five native
// producer/consumer facts remain required by the caller. Active ranks are folded
// in ascending order; a single rank is validated and copied, and mask zero
// produces positive zero. The original full-mask F32 implementation is reused.
fold_result ordered_fold_f32_masked(const float * const inputs[5], float * output,
                                   std::size_t elements, std::uint32_t computed_mask) noexcept;

// Wire values are IEEE binary16 bit patterns. Each active input is widened to
// F32, rank sums round in F32, and only the final sum is rounded to binary16 RN.
// A nonfinite packed result is an error and must never be published.
fold_result ordered_fold_f16(const std::uint16_t * const inputs[5], std::uint16_t * output,
                            std::size_t elements, std::uint32_t computed_mask) noexcept;

// Portable fallback, exercised directly by CPU tests independently of dispatch.
fold_result ordered_fold_f16_scalar(const std::uint16_t * const inputs[5], std::uint16_t * output,
                                   std::size_t elements, std::uint32_t computed_mask) noexcept;

}  // namespace tq::tp5::detail
