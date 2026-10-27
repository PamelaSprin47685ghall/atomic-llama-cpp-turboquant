#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct fixture {
    static constexpr int width = 2048;
    static constexpr int outputs = 32;
    static constexpr int columns = 7;
    std::vector<float> weights;
    std::vector<float> input;

    fixture() : weights(width * outputs), input(width * columns) {
        uint32_t state = 0x38d741c5;
        const auto next = [&state]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return float(int32_t(state & 0xffff) - 32768) / 8192.0f;
        };
        for (auto & value : weights) { value = next(); }
        for (auto & value : input) { value = next(); }
        // This exact result cannot survive an F16 operand conversion.
        std::fill(weights.begin(), weights.begin() + width, 0.0f);
        weights[0] = 65537.0f;
        weights[1] = -65536.0f;
        for (int column = 0; column < columns; ++column) {
            input[column * width] = float(column + 1);
            input[column * width + 1] = float(column + 1);
        }
    }
};

std::vector<float> execute(ggml_backend_t backend, const fixture & data,
                           int first, int count, int padding, ggml_prec precision, int threads) {
    ggml_init_params params{};
    params.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    require(context != nullptr, "could not create graph context");
    auto * weights = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, fixture::width, fixture::outputs);
    auto * storage = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, fixture::width + padding, count);
    auto * input = padding == 0 ? storage :
        ggml_view_2d(context.get(), storage, fixture::width, count, storage->nb[1], 0);
    auto * output = ggml_mul_mat(context.get(), weights, input);
    ggml_mul_mat_set_prec(output, precision);
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "could not allocate graph tensors");
    ggml_backend_tensor_set(weights, data.weights.data(), 0, data.weights.size() * sizeof(float));
    std::vector<float> padded(size_t(fixture::width + padding) * count, 19.0f);
    for (int column = 0; column < count; ++column) {
        std::copy_n(data.input.data() + (first + column) * fixture::width, fixture::width,
                    padded.data() + column * (fixture::width + padding));
    }
    ggml_backend_tensor_set(storage, padded.data(), 0, padded.size() * sizeof(float));
    ggml_backend_cpu_set_n_threads(backend, threads);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CPU graph failed");
    std::vector<float> values(size_t(fixture::outputs) * count);
    ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    for (const auto value : values) { require(std::isfinite(value), "non-finite CPU result"); }
    return values;
}

double verify_oracle(const fixture & data, const std::vector<float> & actual, int count) {
    double maximum = 0;
    const double epsilon = std::numeric_limits<float>::epsilon();
    const double gamma = fixture::width * epsilon / (1 - fixture::width * epsilon);
    for (int column = 0; column < count; ++column) {
        require(actual[column * fixture::outputs] == float(column + 1), "F32 precision sentinel failed");
        for (int row = 0; row < fixture::outputs; ++row) {
            double expected = 0;
            double magnitude = 0;
            for (int k = 0; k < fixture::width; ++k) {
                const double product = double(data.weights[row * fixture::width + k]) *
                                       double(data.input[column * fixture::width + k]);
                expected += product;
                magnitude += std::abs(product);
            }
            const double error = std::abs(double(actual[column * fixture::outputs + row]) - expected);
            require(error <= gamma * magnitude + epsilon * std::abs(expected), "F32 dot exceeds error bound");
            maximum = std::max(maximum, error);
        }
    }
    return maximum;
}
} // namespace

int main() {
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        require(backend != nullptr, "could not create CPU backend");
        const fixture data;
        std::vector<float> reference;
        for (int column = 0; column < fixture::columns; ++column) {
            auto one = execute(backend.get(), data, column, 1, 0, GGML_PREC_F32_PEDANTIC, 1);
            reference.insert(reference.end(), one.begin(), one.end());
        }
        size_t cases = 0;
        double oracle_maximum = 0;
        double default_maximum = 0;
        for (const int count : {2, 3, 7}) {
            for (const int padding : {0, 5}) {
                for (const int threads : {1, 2}) {
                    const auto actual = execute(backend.get(), data, 0, count, padding,
                                                GGML_PREC_F32_PEDANTIC, threads);
                    require(std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) == 0,
                            "PEDANTIC F32 dots changed across batch, stride, or thread count");
                    oracle_maximum = std::max(oracle_maximum, verify_oracle(data, actual, count));
                    ++cases;
                }
            }
            // DEFAULT is measured only. Its reduction contract remains unchanged.
            const auto ordinary = execute(backend.get(), data, 0, count, 0, GGML_PREC_DEFAULT, 2);
            for (size_t i = 0; i < ordinary.size(); ++i) {
                default_maximum = std::max(default_maximum, std::abs(double(ordinary[i]) - reference[i]));
            }
        }
        std::cout << std::setprecision(17)
                  << "{\"status\":\"PASS\",\"scope\":\"CPU F32 x F32 PEDANTIC dot contract\",\"cases\":" << cases
                  << ",\"width\":" << fixture::width << ",\"output_rows\":" << fixture::outputs
                  << ",\"batch_columns\":[2,3,7],\"row_padding\":[0,5],\"threads\":[1,2]"
                  << ",\"exact_serial_equivalence\":true,\"f32_precision_sentinel\":true"
                  << ",\"double_oracle_max_abs\":" << oracle_maximum
                  << ",\"default_vs_pedantic_max_abs\":" << default_maximum << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "CPU F32 precision test failed: " << error.what() << '\n';
        return 1;
    }
}
