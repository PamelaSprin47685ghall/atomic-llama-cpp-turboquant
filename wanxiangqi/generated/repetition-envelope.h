#pragma once

#include <cstddef>

namespace wanxiangqi_repetition_envelope {
inline constexpr std::size_t vocabulary_size = 199998;
inline constexpr std::size_t half_life = 256;
inline constexpr double lambda = 0.99729605608547012;
inline constexpr double normal_weighted_distinct_count = 105.50161324208266;
inline constexpr double central_probability = 0.94999999999999996;
inline constexpr double lower_quantile_probability = 0.025000000000000022;
inline constexpr double upper_quantile_probability = 0.97499999999999998;
inline constexpr double lower_weighted_distinct_count = 34.929298958513719;
inline constexpr double upper_weighted_distinct_count = 170.57944475033941;
inline constexpr std::size_t source_files = 2974;
inline constexpr std::size_t corpus_tokens = 10957521;
}
