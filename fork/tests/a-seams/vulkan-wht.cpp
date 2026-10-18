#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

struct test_case {
    const char * name;
    int group;
    int direction;
    int64_t rows;
    bool scaled;
};

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// All shapes obey the unchanged Vulkan support contract ne[0] % 128 == 0.
// Explicit groups 32 and 64 are legal public ggml_turbo_wht arguments even
// when the tensor row width is 128; this test does not expand GPU support.
std::vector<float> run(ggml_backend_t backend, const test_case & tc) {
    constexpr int64_t width = 128;
    constexpr size_t graph_capacity = 16;
    const ggml_init_params params = {
        graph_capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false),
        nullptr,
        true,
    };
    context_ptr context(ggml_init(params), ggml_free);
    require(context != nullptr, "metadata allocation failed");
    auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, tc.rows);
    auto * scale = tc.scaled ? ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, tc.group) : nullptr;
    auto * output = ggml_turbo_wht(context.get(), input, tc.direction, tc.group, scale);
    ggml_set_input(input);
    if (scale) {
        ggml_set_input(scale);
    }
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    ggml_build_forward_expand(graph, output);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "tensor allocation failed");

    const size_t count = static_cast<size_t>(width * tc.rows);
    std::vector<float> source(count);
    for (size_t i = 0; i < count; ++i) {
        // Exactly representable inputs, with different content in adjacent
        // subgroups and rows. A wrong 128-wide transform cannot pass by symmetry.
        const int32_t value = static_cast<int32_t>((i * 73 + (i / 32) * 19) % 509) - 254;
        source[i] = static_cast<float>(value) / 128.0f;
    }
    ggml_backend_tensor_set(input, source.data(), 0, source.size() * sizeof(float));
    if (scale) {
        std::vector<float> scales(static_cast<size_t>(tc.group));
        for (size_t i = 0; i < scales.size(); ++i) {
            scales[i] = 0.5f + static_cast<float>((i * 11) % 29) / 32.0f;
        }
        ggml_backend_tensor_set(scale, scales.data(), 0, scales.size() * sizeof(float));
    }

    // Missing output groups must fail deterministically, not happen to match
    // fresh zeroed allocator pages. In particular rows=513 exercises Y-split
    // host geometry for the old 32/64-group dispatch.
    std::vector<float> result(count, std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(output, result.data(), 0, result.size() * sizeof(float));

    // Direct backend execution deliberately avoids the scheduler: Vulkan must
    // execute the WHT itself. There is no CPU fallback that could mask a bug.
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "direct backend graph computation failed");
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    return result;
}

bool compare(const test_case & tc, const std::vector<float> & expected, const std::vector<float> & actual) {
    require(expected.size() == actual.size(), "output size mismatch");
    size_t bad = 0;
    size_t first_bad = 0;
    float max_abs = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        const float tolerance = 2.0e-5f + 2.0e-5f * std::abs(expected[i]);
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) || error > tolerance) {
            if (bad++ == 0) {
                first_bad = i;
            }
        }
        if (std::isfinite(error)) {
            max_abs = std::max(max_abs, error);
        }
    }
    std::cout << (bad == 0 ? "PASS " : "FAIL ") << tc.name
              << " group=" << tc.group << " direction=" << tc.direction
              << " rows=" << tc.rows << " scaled=" << tc.scaled
              << " compared=" << actual.size() << " mismatches=" << bad
              << " max_abs=" << max_abs;
    if (bad) {
        std::cout << " first=" << first_bad << " cpu=" << expected[first_bad]
                  << " vulkan=" << actual[first_bad];
    }
    std::cout << '\n';
    return bad == 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 2) {
            throw std::runtime_error("usage: test-vulkan-wht [device-index]");
        }
        size_t device = 0;
        if (argc == 2) {
            const std::string value = argv[1];
            size_t end = 0;
            const unsigned long parsed = std::stoul(value, &end);
            require(!value.empty() && value[0] != '-' && end == value.size(), "invalid device index");
            device = static_cast<size_t>(parsed);
        }
        backend_ptr cpu(ggml_backend_cpu_init(), ggml_backend_free);
        backend_ptr vulkan(ggml_backend_vk_init(device), ggml_backend_free);
        require(cpu != nullptr, "CPU backend initialization failed");
        require(vulkan != nullptr, "Vulkan backend initialization failed; no CPU fallback is allowed");
        ggml_backend_cpu_set_n_threads(cpu.get(), 2);

        const test_case cases[] = {
            {"control_128_forward", 128, 0, 4, false},
            {"control_128_inverse", 128, 1, 4, false},
            {"group_64_forward", 64, 0, 4, false},
            {"group_64_inverse", 64, 1, 4, false},
            {"group_32_forward", 32, 0, 4, false},
            {"group_32_inverse", 32, 1, 4, false},
            {"grid_64_forward", 64, 0, 513, false},
            {"grid_32_forward", 32, 0, 513, false},
            {"scale_128_forward", 128, 0, 4, true},
            {"scale_128_inverse", 128, 1, 4, true},
            {"scale_64_forward", 64, 0, 4, true},
            {"scale_32_inverse", 32, 1, 4, true},
        };
        size_t failed = 0;
        for (const auto & tc : cases) {
            const auto expected = run(cpu.get(), tc);
            const auto actual = run(vulkan.get(), tc);
            failed += !compare(tc, expected, actual);
        }
        std::cout << "Vulkan WHT device=" << device << " cases=" << (sizeof(cases) / sizeof(cases[0]))
                  << " failed=" << failed << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "FAIL Vulkan WHT: " << error.what() << '\n';
        return 1;
    }
}
