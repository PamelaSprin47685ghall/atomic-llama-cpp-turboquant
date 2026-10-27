#include "tq/sampling/shortlist.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
              std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24,
              "finite greedy tests require IEEE binary32");

void require(bool value, const std::string & message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void require_unusable(const tq::sampling::greedy_result & result, const std::string & label) {
    require(!result.usable, label + ": invalid row was accepted");
    require(result.selected.id == -1, label + ": partial token escaped");
    require(float_bits(result.selected.logit) == 0u, label + ": default logit changed");
}

struct reference_result {
    bool usable = false;
    std::int32_t id = -1;
    float logit = 0.0f;
};

reference_result scalar_reference(const float * logits, std::int32_t count) {
    if (logits == nullptr || count <= 0) {
        return {};
    }
    // Keep the original C caller's scalar rule, including its strict comparison.
    std::int32_t selected = 0;
    for (std::int32_t token = 0; token < count; ++token) {
        const float value = logits[token];
        if (!std::isfinite(value)) {
            return {};
        }
        if (value > logits[selected]) {
            selected = token;
        }
    }
    return {true, selected, logits[selected]};
}

tq::sampling::greedy_result compare_reference(const float * logits, std::int32_t count,
                                              const std::string & label) {
    const auto expected = scalar_reference(logits, count);
    const auto actual = tq::sampling::select_finite_greedy(logits, count);
    require(actual.usable == expected.usable, label + ": usable state differs");
    if (!expected.usable) {
        require_unusable(actual, label);
    } else {
        require(actual.selected.id == expected.id, label + ": token differs from scalar scan");
        require(float_bits(actual.selected.logit) == float_bits(expected.logit),
                label + ": selected logit bits differ");
    }
    return actual;
}

tq::sampling::greedy_result compare_reference(const std::vector<float> & logits,
                                              const std::string & label) {
    return compare_reference(logits.data(), static_cast<std::int32_t>(logits.size()), label);
}

void check_invalid_arguments() {
    require_unusable(tq::sampling::greedy_result{}, "default result");
    const float value = 1.0f;
    for (const std::int32_t count : {-1, 0, 1, 17}) {
        compare_reference(nullptr, count, "null pointer count " + std::to_string(count));
    }
    for (const std::int32_t count : {0, -1, std::numeric_limits<std::int32_t>::min()}) {
        compare_reference(&value, count, "invalid count " + std::to_string(count));
    }
    compare_reference(std::vector<float>{}, "empty vector");
}

void check_nonfinite_values() {
    const std::array<float, 4> invalid = {
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        float_from_bits(0x7fc00001u),
        float_from_bits(0xffc12345u),
    };
    std::vector<float> baseline(23);
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        baseline[i] = static_cast<float>(i) - 11.0f;
    }
    for (const float value : invalid) {
        const std::string label = "nonfinite bits " + std::to_string(float_bits(value));
        compare_reference(&value, 1, label + " singleton");
        auto logits = baseline;
        // Cover every lane in two vectors and every position in a seven-value tail.
        for (std::size_t i = 0; i < logits.size(); ++i) {
            logits[i] = value;
            compare_reference(logits, label + " at " + std::to_string(i));
            logits[i] = baseline[i];
        }
        std::fill(logits.begin(), logits.end(), value);
        compare_reference(logits, label + " whole row");
        for (const std::size_t index : {8u, 22u}) {
            logits.assign(23, -1.0f);
            logits[0] = std::numeric_limits<float>::max();
            logits[index] = value;
            compare_reference(logits, label + " after maximum finite value");
        }
    }
}

void check_ties_and_finite_bits() {
    for (const std::size_t count : {1u, 2u, 7u, 8u, 9u, 15u, 16u, 17u, 33u}) {
        compare_reference(std::vector<float>(count, -3.0f),
                          "equal logits count " + std::to_string(count));
    }
    std::vector<float> tied(17, -2.0f);
    for (const std::size_t i : {3u, 7u, 8u, 15u, 16u}) {
        tied[i] = 7.0f;
    }
    compare_reference(tied, "maximum ties within vectors and at the tail");

    for (const std::uint32_t first_zero : {0u, 0x80000000u}) {
        std::vector<float> zeros(17, -1.0f);
        zeros[3] = float_from_bits(first_zero);
        for (const std::size_t i : {4u, 7u, 8u, 15u, 16u}) {
            zeros[i] = float_from_bits(first_zero ^ 0x80000000u);
        }
        compare_reference(zeros, "signed zero maximum bits " + std::to_string(first_zero));
        for (std::size_t i = 0; i < zeros.size(); ++i) {
            zeros[i] = float_from_bits(first_zero ^ ((i % 2) ? 0x80000000u : 0u));
        }
        compare_reference(zeros, "all zero ties bits " + std::to_string(first_zero));
    }

    for (const std::uint32_t bits : {0u, 0x80000000u, 0x00000001u, 0x80000001u,
                                     0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
                                     0x3f800001u, 0xbf800001u, 0x7f7fffffu, 0xff7fffffu}) {
        const float value = float_from_bits(bits);
        compare_reference(&value, 1, "finite singleton bits " + std::to_string(bits));
    }
}

void check_ordered_and_nonmonotone_rows() {
    for (const std::size_t count : {8u, 17u, 33u, 248320u}) {
        std::vector<float> logits(count);
        for (std::size_t i = 0; i < count; ++i) {
            logits[i] = static_cast<float>(i) - 16.0f;
        }
        compare_reference(logits, "ascending count " + std::to_string(count));
        std::reverse(logits.begin(), logits.end());
        compare_reference(logits, "descending count " + std::to_string(count));
    }
    // Several lanes exceed the old maximum. Each update must check the new maximum.
    compare_reference({-20.0f, 11.0f, 3.0f, 19.0f, 7.0f, 5.0f, 17.0f, 9.0f},
                      "nonmonotone candidates in one vector");
    compare_reference({-20.0f, 11.0f, 3.0f, 19.0f, 7.0f, 5.0f, 17.0f, 9.0f,
                       2.0f, 29.0f, 23.0f, 31.0f, 25.0f, 21.0f, 27.0f, 24.0f, 30.0f},
                      "nonmonotone candidates across vectors and tail");
}

void check_unaligned_rows_and_tails() {
    std::uint32_t state = 123456789u;
    const float padding = std::numeric_limits<float>::quiet_NaN();
    for (const std::size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 248320u}) {
        std::vector<float> logits(count);
        for (float & value : logits) {
            state = state * 1664525u + 1013904223u;
            std::uint32_t bits = state;
            if ((bits & 0x7f800000u) == 0x7f800000u) {
                bits &= ~0x00800000u;
            }
            value = float_from_bits(bits);
        }
        // One allocation makes these offsets cover every float alignment modulo 32.
        std::vector<float> storage(count + 15, padding);
        for (std::size_t offset = 0; offset < 8; ++offset) {
            std::fill(storage.begin(), storage.end(), padding);
            std::copy(logits.begin(), logits.end(), storage.begin() + offset);
            compare_reference(storage.data() + offset, static_cast<std::int32_t>(count),
                              "count " + std::to_string(count) + " offset " + std::to_string(offset));
        }
    }
}

void check_ownership() {
    tq::sampling::greedy_result first;
    tq::sampling::greedy_result second;
    {
        std::vector<float> logits(17, -2.0f);
        logits[3] = float_from_bits(0x80000000u);
        logits[8] = 0.0f;
        first = compare_reference(logits, "first owned result");
        std::fill(logits.begin(), logits.end(), -100.0f);
        logits[6] = 2.0f;
        second = compare_reference(logits, "independent owned result");
    }
    require(first.usable && first.selected.id == 3 && float_bits(first.selected.logit) == 0x80000000u,
            "first result changed after input mutation or destruction");
    require(second.usable && second.selected.id == 6 && float_bits(second.selected.logit) == float_bits(2.0f),
            "second result changed after input destruction");
    const auto copy = first;
    first.usable = false;
    first.selected.id = 12;
    first.selected.logit = 100.0f;
    require(copy.usable && copy.selected.id == 3 && float_bits(copy.selected.logit) == 0x80000000u,
            "copied result shares mutable state");
    require(second.usable && second.selected.id == 6 && float_bits(second.selected.logit) == float_bits(2.0f),
            "result mutation changed another result");
}

} // namespace

int main() {
    try {
        check_invalid_arguments();
        check_nonfinite_values();
        check_ties_and_finite_bits();
        check_ordered_and_nonmonotone_rows();
        check_unaligned_rows_and_tails();
        check_ownership();
        std::cout << "Finite greedy boundary tests passed.\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Finite greedy boundary test failed: " << error.what() << '\n';
        return 1;
    }
}
