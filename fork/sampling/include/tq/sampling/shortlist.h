#pragma once

#include <cstdint>
#include <vector>

namespace tq::sampling {

struct candidate {
    std::int32_t id;
    float logit;
};

struct result {
    bool usable = false;
    std::vector<candidate> candidates;
};

struct greedy_result {
    bool usable = false;
    candidate selected{-1, 0.0f};
};

// Return min(count, k) candidates in descending logit order.
// The caller must provide count readable floats and 2 <= k <= 128.
// A usable row has a unique finite maximum. Negative infinity masks are valid.
// Ties at the maximum, NaN, positive infinity, and invalid arguments return an
// empty, unusable result. The result owns its values and does not keep logits.
// Equal logits below the maximum have no specified token order.
result select(const float * logits, std::int32_t count, int k);

// Return the maximum from a row of finite logits. Equal maxima, including
// signed zeros, select the smallest id and retain that input value's bits.
// The caller must provide count readable floats. Null, count <= 0, NaN and
// either infinity return the default unusable result. The result owns its value.
greedy_result select_finite_greedy(const float * logits, std::int32_t count);

} // namespace tq::sampling
