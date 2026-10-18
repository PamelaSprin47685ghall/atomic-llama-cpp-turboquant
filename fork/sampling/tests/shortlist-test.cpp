#include <functional>
#include "tq/sampling/shortlist.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool value, const std::string & message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void require_unusable(const tq::sampling::result & value, const std::string & message) {
    require(!value.usable, message + ": row was accepted");
    require(value.candidates.empty(), message + ": partial output escaped");
}

void compare_reference(const std::vector<float> & logits, int k) {
    const auto selected = tq::sampling::select(logits.data(),
        static_cast<std::int32_t>(logits.size()), k);
    require(selected.usable, "valid row was rejected");
    std::vector<float> expected = logits;
    std::sort(expected.begin(), expected.end(), std::greater<float>{});
    expected.resize(std::min(expected.size(), static_cast<std::size_t>(k)));
    require(selected.candidates.size() == expected.size(), "candidate count changed");
    std::vector<bool> seen(logits.size(), false);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto & value = selected.candidates[i];
        require(value.id >= 0 && static_cast<std::size_t>(value.id) < logits.size(), "invalid id");
        require(!seen[static_cast<std::size_t>(value.id)], "duplicate id");
        seen[static_cast<std::size_t>(value.id)] = true;
        require(value.logit == logits[static_cast<std::size_t>(value.id)], "id and logit differ");
        require(value.logit == expected[i], "top-k differs from the full sort");
    }
}

void check_invalid_arguments() {
    const float value = 1.0f;
    require_unusable(tq::sampling::select(nullptr, 1, 2), "null pointer");
    require_unusable(tq::sampling::select(&value, 0, 2), "empty row");
    require_unusable(tq::sampling::select(&value, -1, 2), "negative count");
    for (const int k : {-1, 0, 1, 129}) {
        require_unusable(tq::sampling::select(&value, 1, k), "unsupported k");
    }
}

void check_masks_and_nonfinite_rows() {
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> logits(257, -infinity);
    require_unusable(tq::sampling::select(logits.data(), 257, 40), "all masks");
    logits[256] = -5.0f;
    compare_reference(logits, 40);
    compare_reference({-1.0f}, 128);
    require_unusable(tq::sampling::select(&infinity, 1, 2), "single positive infinity");
    require_unusable(tq::sampling::select(&nan, 1, 2), "single NaN");

    for (const std::size_t index : {0u, 1u, 7u, 8u, 15u, 16u, 31u, 32u, 64u, 128u, 256u}) {
        logits.assign(257, -10.0f);
        logits[128] = 10.0f;
        logits[index] = nan;
        require_unusable(tq::sampling::select(logits.data(), 257, 2), "NaN position");
        require_unusable(tq::sampling::select(logits.data(), 257, 40), "NaN heap or tail");
        logits[index] = infinity;
        require_unusable(tq::sampling::select(logits.data(), 257, 2), "positive infinity position");
        require_unusable(tq::sampling::select(logits.data(), 257, 40), "positive infinity heap or tail");
    }
}

void check_ties() {
    for (const std::size_t count : {2u, 8u, 9u, 17u, 33u, 65u, 257u}) {
        std::vector<float> logits(count, -3.0f);
        logits[0] = 7.0f;
        logits[count - 1] = 7.0f;
        require_unusable(tq::sampling::select(logits.data(),
            static_cast<std::int32_t>(count), 2), "maximum tie");
        require_unusable(tq::sampling::select(logits.data(),
            static_cast<std::int32_t>(count), 128), "maximum tie with large k");
    }
    compare_reference({9.0f, 8.0f, 8.0f, 8.0f, 8.0f, 7.0f}, 3);
    compare_reference({1.0f, -0.0f, 0.0f, -1.0f}, 3);
    const float zeros[] = {-0.0f, 0.0f};
    require_unusable(tq::sampling::select(zeros, 2, 2), "signed zero tie");
}

void check_full_sort_reference() {
    std::uint32_t state = 123456789u;
    for (const std::size_t count : {1u, 2u, 3u, 7u, 8u, 9u, 16u, 17u, 31u, 32u, 33u,
                                   129u, 137u, 1001u, 32000u}) {
        for (int pass = 0; pass < 4; ++pass) {
            std::vector<float> logits(count);
            for (float & value : logits) {
                state = state * 1664525u + 1013904223u;
                value = static_cast<float>(state % 1024u) / 16.0f - 32.0f;
            }
            const std::size_t best = pass == 0 ? 0 : pass == 1 ? count - 1 : count / 2;
            logits[best] = 1000.0f;
            if (pass == 3) {
                std::sort(logits.begin(), logits.end());
            }
            for (const int k : {2, 3, 8, 40, 128}) {
                compare_reference(logits, k);
            }
        }
    }
}

void check_ownership() {
    tq::sampling::result selected;
    {
        std::vector<float> logits = {9.0f, 6.0f, 7.0f, 8.0f};
        selected = tq::sampling::select(logits.data(), 4, 3);
        std::fill(logits.begin(), logits.end(), -100.0f);
    }
    require(selected.usable, "owned row was rejected");
    require(selected.candidates.size() == 3, "owned candidate count changed");
    require(selected.candidates[0].id == 0 && selected.candidates[0].logit == 9.0f,
        "result depends on input storage");
    require(selected.candidates[1].id == 3 && selected.candidates[1].logit == 8.0f,
        "result lost its token mapping");
    const auto copy = selected;
    selected.candidates[0].logit = -20.0f;
    require(copy.candidates[0].logit == 9.0f, "copied result shares mutable storage");
}

} // namespace

int main() {
    try {
        check_invalid_arguments();
        check_masks_and_nonfinite_rows();
        check_ties();
        check_full_sort_reference();
        check_ownership();
        std::cout << "Sampling boundary tests passed.\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Sampling boundary test failed: " << error.what() << '\n';
        return 1;
    }
}
