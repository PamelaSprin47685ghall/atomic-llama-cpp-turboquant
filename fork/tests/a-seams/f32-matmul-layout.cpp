#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr std::int64_t width = 64;
// Four query heads share two KV heads, including in the single-token calls.
constexpr std::int64_t kv_heads = 2;
constexpr std::int64_t query_heads = 4;
// F16 drops this increment. F32 retains it after the integer terms cancel.
constexpr float small = 1.0f / 8192.0f;

struct Input {
    std::int64_t tokens;
    std::int64_t rows;
    std::vector<float> a;
    std::vector<float> b;
};

Input make_input(bool change_a, std::int64_t tokens, std::int64_t first_token, std::int64_t rows = 5) {
    Input input{tokens, rows, std::vector<float>(kv_heads * rows * width),
        std::vector<float>(query_heads * tokens * width)};
    for (std::int64_t head = 0; head < kv_heads; ++head) {
        for (std::int64_t row = 0; row < rows; ++row) {
            const float scale = static_cast<float>(1u << (head + row % 3));
            for (std::int64_t k = 0; k < width; ++k) {
                const float value = k % 2 == 0 ? 1.0f + (change_a ? small : 0.0f) : -1.0f;
                input.a[(head * rows + row) * width + k] = scale * value;
            }
        }
    }
    for (std::int64_t head = 0; head < query_heads; ++head) {
        for (std::int64_t token = 0; token < tokens; ++token) {
            // Keep the larger dot products exact in F32. The first two scales
            // remain 1 and 2, as in the single-token and two-token cases.
            const auto token_scale = 1u << ((first_token + token) % 5);
            const float scale = static_cast<float>((head + 1) * token_scale);
            for (std::int64_t k = 0; k < width; ++k) {
                const float value = !change_a && k % 2 == 0 ? 1.0f + small : 1.0f;
                input.b[(head * tokens + token) * width + k] = scale * value;
            }
        }
    }
    return input;
}

std::vector<float> reference(const Input & input, bool round_a, bool round_b) {
    std::vector<float> result(query_heads * input.tokens * input.rows);
    const auto half = [](float value) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)); };
    for (std::int64_t head = 0; head < query_heads; ++head) {
        for (std::int64_t token = 0; token < input.tokens; ++token) {
            for (std::int64_t row = 0; row < input.rows; ++row) {
                double sum = 0.0;
                for (std::int64_t k = 0; k < width; ++k) {
                    const auto kv_head = head / (query_heads / kv_heads);
                    const auto a = input.a[(kv_head * input.rows + row) * width + k];
                    const auto b = input.b[(head * input.tokens + token) * width + k];
                    sum += static_cast<double>(round_a ? half(a) : a) * (round_b ? half(b) : b);
                }
                result[(head * input.tokens + token) * input.rows + row] = static_cast<float>(sum);
            }
        }
    }
    return result;
}

std::vector<float> compute(ggml_backend_t backend, const Input & input, bool stride_a,
                           bool stride_b, ggml_prec precision) {
    constexpr std::size_t graph_size = 16;
    const ggml_init_params params{16 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_size, false), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        throw std::runtime_error("cannot create the graph context");
    }
    const auto make_tensor = [&](const std::vector<float> & data, std::int64_t columns,
                                 std::int64_t heads, bool strided) {
        const std::int64_t stored_width = width + (strided ? 5 : 0);
        const std::int64_t stored_columns = columns + (strided ? 1 : 0);
        auto * storage = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32,
            stored_width, stored_columns, heads);
        auto * view = strided ? ggml_view_3d(context.get(), storage, width, columns, heads,
            stored_width * sizeof(float), stored_width * stored_columns * sizeof(float), 0) : storage;
        std::vector<float> packed(heads * stored_columns * stored_width, 17.0f);
        for (std::int64_t head = 0; head < heads; ++head) {
            for (std::int64_t column = 0; column < columns; ++column) {
                std::copy_n(data.data() + (head * columns + column) * width, width,
                    packed.data() + (head * stored_columns + column) * stored_width);
            }
        }
        return std::make_tuple(storage, view, std::move(packed));
    };
    auto [a_storage, a, a_data] = make_tensor(input.a, input.rows, kv_heads, stride_a);
    auto [b_storage, b, b_data] = make_tensor(input.b, input.tokens, query_heads, stride_b);
    if (ggml_is_contiguous(a) == stride_a || ggml_is_contiguous(b) == stride_b) {
        throw std::runtime_error("test input has the wrong tensor strides");
    }
    auto * output = ggml_mul_mat(context.get(), a, b);
    ggml_mul_mat_set_prec(output, precision);
    if (!ggml_backend_dev_supports_op(ggml_backend_get_device(backend), output)) {
        throw std::runtime_error("backend does not support the requested F32 matrix product");
    }
    auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("cannot allocate the graph tensors");
    }
    ggml_backend_tensor_set(a_storage, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b_storage, b_data.data(), 0, b_data.size() * sizeof(float));
    const auto status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("matrix compute failed: " + std::to_string(static_cast<int>(status)));
    }
    std::vector<float> result(query_heads * input.tokens * input.rows);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    return result;
}

float compare(const std::vector<float> & actual, const std::vector<float> & expected,
              const std::string & name) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + ": wrong result size");
    }
    float maximum = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || error > 1e-6f) {
            throw std::runtime_error(name + ": error at " + std::to_string(i) +
                ", expected " + std::to_string(expected[i]) + ", got " + std::to_string(actual[i]));
        }
        maximum = std::max(maximum, error);
    }
    return maximum;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 2) {
            throw std::invalid_argument("usage: test-f32-matmul [device]");
        }
        const char * device = argc == 2 ? argv[1] : "Vulkan0";
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(device, nullptr), ggml_backend_free);
        if (!backend) {
            throw std::runtime_error(std::string("backend is unavailable: ") + device);
        }
        float maximum = 0.0f;
        std::size_t graphs = 0;
        for (const auto precision : {GGML_PREC_DEFAULT, GGML_PREC_F32, GGML_PREC_F32_PEDANTIC}) {
            const std::string precision_name = precision == GGML_PREC_DEFAULT ? "default" :
                precision == GGML_PREC_F32 ? "f32" : "pedantic";
            for (const bool change_a : {true, false}) {
                const auto input = make_input(change_a, 2, 0);
                const auto expected = reference(input, false, false);
                const auto rounded = reference(input, change_a, !change_a);
                for (std::size_t i = 0; i < expected.size(); ++i) {
                    if (std::abs(expected[i] - rounded[i]) < 0.001f) {
                        throw std::runtime_error("test input does not detect half-precision staging");
                    }
                }
                const auto contiguous = compute(backend.get(), input, false, false, precision);
                ++graphs;
                maximum = std::max(maximum,
                    compare(contiguous, expected, precision_name + ", contiguous host oracle"));
                for (unsigned layout = 0; layout < 4; ++layout) {
                    const bool stride_a = (layout & 1u) != 0;
                    const bool stride_b = (layout & 2u) != 0;
                    const std::string name = precision_name + ", " + (change_a ? "A" : "B") +
                        " precision, layout " + std::to_string(layout);
                    if (layout != 0) {
                        const auto batched = compute(backend.get(), input, stride_a, stride_b, precision);
                        ++graphs;
                        maximum = std::max(maximum, compare(batched, contiguous, name + ", batch 2"));
                    }
                    std::vector<float> serial(contiguous.size());
                    for (std::int64_t token = 0; token < 2; ++token) {
                        const auto value = compute(backend.get(), make_input(change_a, 1, token, input.rows),
                            stride_a, stride_b, precision);
                        ++graphs;
                        for (std::int64_t head = 0; head < query_heads; ++head) {
                            std::copy_n(value.data() + head * input.rows, input.rows,
                                serial.data() + (head * 2 + token) * input.rows);
                        }
                    }
                    maximum = std::max(maximum, compare(serial, contiguous, name + ", serial 1"));
                }
            }
        }
        // Cross the 64- and 128-row tile edges with padded input storage.
        const std::pair<std::int64_t, std::int64_t> tile_shapes[] = {{65, 40}, {129, 65}};
        for (const auto & shape : tile_shapes) {
            const auto input = make_input(true, shape.second, 0, shape.first);
            const auto expected = reference(input, false, false);
            const auto contiguous = compute(backend.get(), input, false, false, GGML_PREC_F32_PEDANTIC);
            const auto strided = compute(backend.get(), input, true, true, GGML_PREC_F32_PEDANTIC);
            graphs += 2;
            const auto name = "pedantic tile " + std::to_string(input.rows) + "x" + std::to_string(input.tokens);
            maximum = std::max(maximum, compare(contiguous, expected, name + ", contiguous"));
            maximum = std::max(maximum, compare(strided, expected, name + ", padded"));
        }
        std::cout << "device=" << device << " F32 strided matmul graphs=" << graphs
                  << " max_error=" << maximum << '\n';
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
