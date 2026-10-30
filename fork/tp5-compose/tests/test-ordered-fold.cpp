#include "ordered-fold.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
using tq::tp5::detail::fold_result;
using tq::tp5::detail::ordered_fold_f32;
using tq::tp5::detail::ordered_fold_f32_masked;
using tq::tp5::detail::ordered_fold_f16;
using tq::tp5::detail::ordered_fold_f16_scalar;
std::uint64_t wire_cases = 0;

constexpr std::size_t capacity = 80;
using rank_storage = std::array<std::array<float, capacity>, 5>;

void check(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(std::uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float reference(const float * const inputs[5], std::size_t element) {
    // Volatile stores make this a scalar, explicitly rounded F32 oracle even
    // when the compiler can vectorize the surrounding test loops.
    volatile float sum = inputs[0][element];
    for (std::size_t rank = 1; rank < 5; ++rank) {
        sum = sum + inputs[rank][element];
    }
    return sum;
}

void finite_cases() {
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float normal = std::numeric_limits<float>::min();
    const float largest = std::numeric_limits<float>::max();
    const std::array<std::array<float, 5>, 10> patterns{{
        {{-0.0f, -0.0f, -0.0f, -0.0f, -0.0f}},
        {{0.0f, -0.0f, 0.0f, -0.0f, -0.0f}},
        {{0x1p24f, 1.0f, -0x1p24f, 1.0f, 1.0f}},
        {{0x1p100f, 1.0f, -0x1p100f, 1.0f, 0.0f}},
        {{largest, -largest, 1.0f, 2.0f, 3.0f}},
        {{normal, -normal, tiny, tiny, -tiny}},
        {{-0x1p24f, -1.0f, 0x1p24f, -1.0f, -1.0f}},
        {{1.0f, -1.0f, -0.0f, -0.0f, -0.0f}},
        {{0x1.000002p0f, 0x1p-24f, -1.0f, 0x1p-24f, -0x1p-23f}},
        {{-largest, largest, -3.0f, 2.0f, 1.0f}},
    }};
    const std::array<std::size_t, 14> lengths{{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63}};
    for (std::size_t phase = 0; phase < 16; ++phase) {
        for (const auto elements : lengths) {
            alignas(16) rank_storage storage{};
            alignas(16) std::array<float, capacity> output;
            output.fill(from_bits(0x4a123456));
            auto expected = output;
            const std::size_t output_offset = phase / 4;
            const float * inputs[5];
            for (std::size_t rank = 0; rank < 5; ++rank) {
                const std::size_t offset = (phase + rank) % 4;
                inputs[rank] = storage[rank].data() + offset;
                for (std::size_t element = 0; element < elements; ++element) {
                    storage[rank][offset + element] = patterns[(element + phase) % patterns.size()][rank];
                }
            }
            const auto before = storage;
            for (std::size_t element = 0; element < elements; ++element) {
                expected[output_offset + element] = reference(inputs, element);
            }
            check(ordered_fold_f32(inputs, output.data() + output_offset, elements) == fold_result::ok,
                  "finite ordered fold was rejected");
            for (std::size_t element = 0; element < capacity; ++element) {
                check(bits(output[element]) == bits(expected[element]),
                      "ordered fold differs bitwise from scalar rank-order F32 or changed output padding");
            }
            check(std::memcmp(storage.data(), before.data(), sizeof(storage)) == 0,
                  "ordered fold modified an input");
        }
    }

    // Independently generated finite bit patterns include mixed signs and
    // exponents. Bound magnitudes so any rank order remains finite.
    alignas(16) rank_storage storage{};
    alignas(16) std::array<float, capacity> output{};
    const float * inputs[5];
    std::uint32_t state = 0x34a97f25;
    for (std::size_t rank = 0; rank < 5; ++rank) {
        inputs[rank] = storage[rank].data() + rank % 4;
        for (std::size_t element = 0; element < 67; ++element) {
            state = state * 1664525u + 1013904223u;
            const auto value = (state & 0x807fffffu) | (((state >> 24) % 200u + 1u) << 23);
            storage[rank][rank % 4 + element] = from_bits(value);
        }
    }
    check(ordered_fold_f32(inputs, output.data() + 1, 67) == fold_result::ok,
          "finite generated fold was rejected");
    for (std::size_t element = 0; element < 67; ++element) {
        check(bits(output[element + 1]) == bits(reference(inputs, element)),
              "generated fold differs bitwise from scalar rank-order F32");
    }
}

void nonfinite_cases() {
    // Include both infinity signs, signed quiet NaNs with payloads, and a
    // signaling NaN. Every rank and every SIMD lane/tail position must reject.
    const std::array<std::uint32_t, 5> invalid{{0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f800001}};
    for (const auto value : invalid) {
        for (std::size_t rank = 0; rank < 5; ++rank) {
            for (std::size_t bad = 0; bad < 11; ++bad) {
                alignas(16) rank_storage storage{};
                alignas(16) std::array<float, capacity> output{};
                const float * inputs[5];
                for (std::size_t r = 0; r < 5; ++r) {
                    inputs[r] = storage[r].data() + r % 4;
                }
                storage[rank][rank % 4 + bad] = from_bits(value);
                check(ordered_fold_f32(inputs, output.data() + 1, 11) == fold_result::nonfinite_input,
                      "nonfinite rank input was not rejected");
            }
        }
    }
}

void overflow_cases() {
    // Overflow at each addition boundary, with either sign. Later cancelling
    // values cannot repair an infinite intermediate in the ordered F32 fold.
    for (const float extreme : {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()}) {
        for (std::size_t rank = 1; rank < 5; ++rank) {
            for (std::size_t bad = 0; bad < 11; ++bad) {
                alignas(16) rank_storage storage{};
                alignas(16) std::array<float, capacity> output{};
                const float * inputs[5];
                for (std::size_t r = 0; r < 5; ++r) {
                    inputs[r] = storage[r].data() + r % 4;
                }
                storage[0][bad] = extreme;
                storage[rank][rank % 4 + bad] = extreme;
                for (std::size_t r = rank + 1; r < 5; ++r) {
                    storage[r][r % 4 + bad] = -extreme;
                }
                check(ordered_fold_f32(inputs, output.data() + 1, 11) == fold_result::nonfinite_sum,
                      "finite-input ordered overflow was not rejected");
            }
        }
    }
}

// Independent binary16 oracle: exact exponent arithmetic and explicit nearest
// integer ties-to-even, not ggml conversion or F16C intrinsics.
float half_value(std::uint16_t h) {
    const unsigned e = (h >> 10) & 31u;
    const unsigned m = h & 1023u;
    float value = e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(1024u + m), int(e) - 25);
    return h & 0x8000u ? -value : value;
}

std::uint16_t half_reference(float value) {
    const std::uint16_t sign = std::signbit(value) ? 0x8000u : 0;
    const long double a = std::fabs(static_cast<long double>(value));
    if (a >= 65520.0L) {
        return sign | 0x7c00u;
    }
    if (a == 0) {
        return sign;
    }
    const auto nearest_even = [](long double scaled) {
        const long double floor = std::floor(scaled);
        auto n = static_cast<unsigned>(floor);
        const long double fraction = scaled - floor;
        return n + unsigned(fraction > 0.5L || (fraction == 0.5L && (n & 1u)));
    };
    if (a < std::ldexp(1.0L, -14)) {
        return sign | std::uint16_t(nearest_even(std::ldexp(a, 24)));
    }
    int exponent = 0;
    std::frexp(a, &exponent);
    unsigned rounded = nearest_even(std::ldexp(a, 11 - exponent));
    if (rounded == 2048u) {
        rounded = 1024u;
        ++exponent;
    }
    return sign | std::uint16_t(((exponent + 14) << 10) | (rounded - 1024u));
}

void masked_f32_cases() {
    const std::array<float, 8> values{{-0.0f, 0.0f, 0x1p24f, -0x1p24f, 1.0f, -1.0f, 0x1p-24f, 0x1p-126f}};
    for (std::uint32_t mask = 0; mask < 32; ++mask) {
        rank_storage storage{};
        const float * inputs[5]{};
        std::array<float, capacity> output{};
        for (std::size_t rank = 0; rank < 5; ++rank) {
            if (mask & (1u << rank)) {
                inputs[rank] = storage[rank].data() + rank % 4;
            }
            for (std::size_t i = 0; i < 67; ++i) {
                storage[rank][rank % 4 + i] = values[(i + rank) % values.size()];
            }
        }
        const auto before = storage;
        check(ordered_fold_f32_masked(inputs, output.data(), 67, mask) == fold_result::ok,
              "masked F32 fold rejected finite input");
        ++wire_cases;
        for (std::size_t i = 0; i < 67; ++i) {
            bool first = true;
            volatile float sum = 0.0f;
            for (std::size_t rank = 0; rank < 5; ++rank) {
                if (mask & (1u << rank)) {
                    sum = first ? inputs[rank][i] : sum + inputs[rank][i];
                    first = false;
                }
            }
            check(bits(output[i]) == bits(sum), "masked F32 rank order or zero/copy path differs");
        }
        check(std::memcmp(storage.data(), before.data(), sizeof(storage)) == 0, "masked F32 modified input");
    }
    const float * inputs[5]{};
    float sentinel = 7.0f;
    check(ordered_fold_f32_masked(inputs, &sentinel, 1, 32) == fold_result::invalid_mask && sentinel == 7.0f,
          "invalid F32 mask was not rejected before writes");
    // Both cold failure classes, including a SIMD lane and a scalar tail.
    for (const std::size_t bad : {std::size_t(1), std::size_t(8)}) {
        rank_storage storage{};
        inputs[0] = storage[0].data(); inputs[4] = storage[4].data();
        storage[0][bad] = std::numeric_limits<float>::infinity();
        std::array<float, capacity> out{};
        check(ordered_fold_f32_masked(inputs, out.data(), 9, 17) == fold_result::nonfinite_input,
              "masked F32 nonfinite input classification differs");
        storage[0][bad] = storage[4][bad] = std::numeric_limits<float>::max();
        check(ordered_fold_f32_masked(inputs, out.data(), 9, 17) == fold_result::nonfinite_sum,
              "masked F32 overflow classification differs");
        wire_cases += 2;
    }
}

void half_finite_cases() {
    using half_storage = std::array<std::array<std::uint16_t, capacity>, 5>;
    const std::array<std::array<std::uint16_t, 5>, 9> patterns{{
        {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000}},
        {{0x0000, 0x8000, 0x0000, 0x8000, 0x8000}},
        {{0x3c00, 0x1000, 0x0000, 0x0000, 0x0000}},
        {{0x3c01, 0x1000, 0x0000, 0x0000, 0x0000}},
        {{0x7bff, 0xfbff, 0x3c00, 0x4000, 0x4200}},
        {{0x0001, 0x0001, 0x8001, 0x0001, 0x8001}},
        {{0x0400, 0x8400, 0x0001, 0x0001, 0x8001}},
        {{0x7800, 0x0001, 0xf800, 0x0001, 0x0000}},
        {{0x3555, 0xb555, 0x3aab, 0xbaab, 0x0003}},
    }};
    const std::array<std::size_t, 15> lengths{{0,1,2,3,4,5,7,8,9,15,16,17,31,33,63}};
    for (std::uint32_t mask = 0; mask < 32; ++mask) {
        for (std::size_t phase = 0; phase < 8; ++phase) {
            for (const auto n : lengths) {
                half_storage storage{};
                const std::uint16_t * inputs[5]{};
                std::array<std::uint16_t, capacity> out, scalar, expected;
                out.fill(0x52a5); scalar = expected = out;
                const auto output_offset = phase % 4;
                for (std::size_t rank = 0; rank < 5; ++rank) {
                    const auto offset = (phase + rank) % 4;
                    if (mask & (1u << rank)) {
                        inputs[rank] = storage[rank].data() + offset;
                    }
                    for (std::size_t i = 0; i < n; ++i) {
                        storage[rank][offset + i] = patterns[(i + phase) % patterns.size()][rank];
                    }
                }
                const auto before = storage;
                for (std::size_t i = 0; i < n; ++i) {
                    bool first = true;
                    volatile float sum = 0.0f;
                    for (std::size_t rank = 0; rank < 5; ++rank) {
                        if (mask & (1u << rank)) {
                            const float value = half_value(inputs[rank][i]);
                            sum = first ? value : sum + value;
                            first = false;
                        }
                    }
                    expected[output_offset + i] = half_reference(sum);
                    check((expected[output_offset + i] & 0x7c00u) != 0x7c00u, "finite oracle test unexpectedly overflowed");
                }
                check(ordered_fold_f16(inputs, out.data()+output_offset, n, mask) == fold_result::ok,
                      "dispatched F16 finite fold rejected");
                check(ordered_fold_f16_scalar(inputs, scalar.data()+output_offset, n, mask) == fold_result::ok,
                      "portable F16 finite fold rejected");
                wire_cases += 2;
                check(out == expected && scalar == expected,
                      "F16 fold differs from independent F32-rank-order/RN oracle or changed output padding");
                check(storage == before, "F16 fold modified input");
            }
        }
    }
}

void half_all_codes_single_rank() {
    for (const std::uint32_t rank : {0u, 4u}) {
        for (std::uint32_t raw = 0; raw <= 65535; ++raw) {
            const auto value = std::uint16_t(raw);
            const std::uint16_t * inputs[5]{};
            inputs[rank] = &value;
            std::uint16_t output = 0x52a5;
            const auto status = ordered_fold_f16(inputs, &output, 1, 1u << rank);
            if ((raw & 0x7c00u) == 0x7c00u) {
                check(status == fold_result::nonfinite_input && output == 0x52a5,
                      "single-rank nonfinite half was copied");
            } else {
                check(status == fold_result::ok && output == raw,
                      "single-rank half copy changed finite bits");
                check(half_reference(half_value(value)) == value, "independent half oracle roundtrip differs");
            }
            ++wire_cases;
        }
    }
}

void half_error_cases() {
    using half_storage = std::array<std::array<std::uint16_t, capacity>, 5>;
    for (const std::uint16_t invalid : {std::uint16_t(0x7c00), std::uint16_t(0xfc00),
                                      std::uint16_t(0x7e01), std::uint16_t(0xfe23), std::uint16_t(0x7c01)}) {
        for (std::size_t rank = 0; rank < 5; ++rank) {
            for (std::size_t bad = 0; bad < 17; ++bad) {
                half_storage storage{};
                const std::uint16_t * inputs[5];
                for (std::size_t r = 0; r < 5; ++r) inputs[r] = storage[r].data();
                storage[rank][bad] = invalid;
                std::array<std::uint16_t, capacity> output{};
                check(ordered_fold_f16(inputs, output.data()+1, 17, 31) == fold_result::nonfinite_input,
                      "dispatched F16 nonfinite input classification differs");
                check(ordered_fold_f16_scalar(inputs, output.data()+1, 17, 31) == fold_result::nonfinite_input,
                      "portable F16 nonfinite input classification differs");
                wire_cases += 2;
            }
        }
    }
    for (const std::uint16_t sign : {std::uint16_t(0), std::uint16_t(0x8000)}) {
        for (std::size_t bad = 0; bad < 17; ++bad) {
            half_storage storage{};
            const std::uint16_t * inputs[5]{storage[0].data(), storage[1].data(), nullptr, nullptr, nullptr};
            storage[0][bad] = sign | 0x7bff; // 65504
            storage[1][bad] = sign | 0x4c00; // 16: tie at 65520 rounds to infinity
            std::array<std::uint16_t, capacity> output{};
            check(ordered_fold_f16(inputs, output.data()+1, 17, 3) == fold_result::nonfinite_pack,
                  "dispatched F16 pack overflow was accepted");
            check(ordered_fold_f16_scalar(inputs, output.data()+1, 17, 3) == fold_result::nonfinite_pack,
                  "portable F16 pack overflow was accepted");
            wire_cases += 2;
        }
    }
    const std::uint16_t * inputs[5]{};
    std::uint16_t sentinel = 0x52a5;
    check(ordered_fold_f16(inputs, &sentinel, 1, 32) == fold_result::invalid_mask && sentinel == 0x52a5,
          "invalid F16 mask was not rejected before writes");
    ++wire_cases;
}

}  // namespace

int main() {
    try {
        finite_cases();
        nonfinite_cases();
        overflow_cases();
        masked_f32_cases();
        half_finite_cases();
        half_all_codes_single_rank();
        half_error_cases();
        std::printf("{\"status\":\"PASS\",\"wire_cases\":%llu,\"original_f32_tests\":\"PASS\","
                    "\"all_masks\":32,\"half_codes_per_single_rank\":65536,\"single_ranks\":2,"
                    "\"independent_half_oracle\":true,\"scalar_and_dispatched_paths\":true}\n",
                    static_cast<unsigned long long>(wire_cases));
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ordered wire fold: %s\n", error.what());
        return 1;
    }
}
