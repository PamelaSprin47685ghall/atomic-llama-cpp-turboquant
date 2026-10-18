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

// Return min(count, k) candidates in descending logit order.
// The caller must provide count readable floats and 2 <= k <= 128.
// A usable row has a unique finite maximum. Negative infinity masks are valid.
// Ties at the maximum, NaN, positive infinity, and invalid arguments return an
// empty, unusable result. The result owns its values and does not keep logits.
// Equal logits below the maximum have no specified token order.
result select(const float * logits, std::int32_t count, int k);

} // namespace tq::sampling
