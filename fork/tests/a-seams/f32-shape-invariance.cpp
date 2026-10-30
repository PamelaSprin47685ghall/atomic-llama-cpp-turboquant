#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using values = std::vector<float>;

void require(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string json_string(const std::string & value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 0x20) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else { result += char(c); }
    }
    result += '"';
    return result;
}

struct geometry {
    int width;
    int rows;
    int columns;
    int kv_heads;
    int query_heads;
};

struct fixture {
    geometry shape;
    values a;
    values b;

    explicit fixture(geometry selected)
        : shape(selected),
          a(size_t(shape.width) * shape.rows * shape.kv_heads),
          b(size_t(shape.width) * shape.columns * shape.query_heads) {
        uint32_t state = 0xb279f531;
        const auto next = [&state]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Division by an odd integer gives varied F32 mantissas. These
            // products and sums require rounding, unlike the staging sentinel.
            return float(int32_t(state & 0xffffu) - 32768) / 10009.0f;
        };
        for (float & item : a) { item = next(); }
        for (float & item : b) { item = next(); }
    }
};

float bias_value(int index, int row) {
    return float((row + 7 * index) % 17 - 8) / float(11 + 2 * index);
}

values compute(ggml_backend_t backend, const fixture & data, const std::vector<int> & order,
               bool stride_a, bool stride_b, int bias_count = 0, bool head_padding_only = false) {
    const auto & g = data.shape;
    const int columns = int(order.size());
    require(columns > 0, "empty query order");
    const ggml_init_params params{
        16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    require(context != nullptr, "could not create graph context");

    const auto tensor = [&](const values & input, int rows, int heads, bool strided) {
        const int stored_width = g.width + (strided && !head_padding_only ? 5 : 0);
        const int stored_rows = rows + (strided ? 1 : 0);
        auto * storage = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, stored_width, stored_rows, heads);
        auto * view = strided ? ggml_view_3d(context.get(), storage, g.width, rows, heads,
                                            storage->nb[1], storage->nb[2], 0) : storage;
        values packed(size_t(stored_width) * stored_rows * heads, 19.0f);
        for (int head = 0; head < heads; ++head) {
            for (int row = 0; row < rows; ++row) {
                std::copy_n(input.data() + (size_t(head) * rows + row) * g.width, g.width,
                            packed.data() + (size_t(head) * stored_rows + row) * stored_width);
            }
        }
        return std::make_tuple(storage, view, std::move(packed));
    };
    values reordered(size_t(g.width) * columns * g.query_heads);
    for (int head = 0; head < g.query_heads; ++head) {
        for (int column = 0; column < columns; ++column) {
            require(order[size_t(column)] >= 0 && order[size_t(column)] < g.columns, "invalid query order");
            std::copy_n(data.b.data() + (size_t(head) * g.columns + order[size_t(column)]) * g.width,
                        g.width, reordered.data() + (size_t(head) * columns + column) * g.width);
        }
    }
    auto [a_storage, a, a_data] = tensor(data.a, g.rows, g.kv_heads, stride_a);
    auto [b_storage, b, b_data] = tensor(reordered, columns, g.query_heads, stride_b);
    auto * output = ggml_mul_mat(context.get(), a, b);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32_PEDANTIC);
    require(ggml_backend_supports_op(backend, output), "backend rejected F32 matrix product");
    std::vector<ggml_tensor *> biases;
    std::vector<values> bias_data;
    for (int index = 0; index < bias_count; ++index) {
        auto * bias = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, g.rows, columns, g.query_heads);
        values packed(size_t(g.rows) * columns * g.query_heads);
        for (size_t i = 0; i < packed.size(); ++i) {
            packed[i] = bias_value(index, int(i % size_t(g.rows)));
        }
        biases.push_back(bias);
        bias_data.push_back(std::move(packed));
        output = index == 0 ? ggml_add(context.get(), output, bias) : ggml_add(context.get(), bias, output);
    }
    auto * graph = ggml_new_graph_custom(context.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "could not allocate graph tensors");
    ggml_backend_tensor_set(a_storage, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b_storage, b_data.data(), 0, b_data.size() * sizeof(float));
    for (size_t index = 0; index < biases.size(); ++index) {
        ggml_backend_tensor_set(biases[index], bias_data[index].data(), 0, bias_data[index].size() * sizeof(float));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "matrix graph failed");
    values result(size_t(g.rows) * columns * g.query_heads);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    for (float value : result) { require(std::isfinite(value), "non-finite matrix result"); }
    return result;
}

void exact(const values & actual, const values & reference, const std::string & name) {
    require(actual.size() == reference.size(), "comparison shape mismatch");
    if (std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) == 0) { return; }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &reference[i], sizeof(float)) != 0) {
            throw std::runtime_error(name + " is not bitwise invariant at " + std::to_string(i) +
                ": actual=" + std::to_string(actual[i]) + " reference=" + std::to_string(reference[i]));
        }
    }
}

double oracle(const fixture & data, const values & actual) {
    const auto & g = data.shape;
    const double unit_roundoff = std::numeric_limits<float>::epsilon() / 2.0;
    const double gamma = (2.0 * g.width * unit_roundoff) / (1.0 - 2.0 * g.width * unit_roundoff);
    double maximum = 0;
    for (int head = 0; head < g.query_heads; ++head) {
        const int kv_head = head / (g.query_heads / g.kv_heads);
        for (int column = 0; column < g.columns; ++column) {
            for (int row = 0; row < g.rows; ++row) {
                double expected = 0;
                double magnitude = 0;
                for (int k = 0; k < g.width; ++k) {
                    const double product = double(data.a[(size_t(kv_head) * g.rows + row) * g.width + k]) *
                                           double(data.b[(size_t(head) * g.columns + column) * g.width + k]);
                    expected += product;
                    magnitude += std::abs(product);
                }
                const double error = std::abs(double(actual[(size_t(head) * g.columns + column) * g.rows + row]) -
                                              expected);
                require(error <= gamma * magnitude + unit_roundoff * std::abs(expected),
                        "matrix product exceeds the F32 forward error bound");
                maximum = std::max(maximum, error);
            }
        }
    }
    return maximum;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        require(argc <= 2, "usage: test-f32-shape-invariance [device]");
        const char * requested = argc == 2 ? argv[1] : "CPU";
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(requested, nullptr), ggml_backend_free);
        require(backend != nullptr, std::string("backend is unavailable: ") + requested);
        const char * device = ggml_backend_dev_name(ggml_backend_get_device(backend.get()));
        // Odd K and M tails, the router geometry, an eligible split-K shape,
        // and larger M/N tile boundaries. Three cases broadcast four query
        // heads over two KV heads.
        const geometry shapes[] = {
            {513, 33, 3, 2, 4},
            {2048, 32, 2, 1, 1},
            {2048, 32, 33, 2, 4},
            {65, 65, 65, 2, 4},
        };
        size_t graphs = 0;
        size_t comparisons = 0;
        size_t bias_graphs = 0;
        double maximum = 0;
        for (const auto & g : shapes) {
            const fixture data(g);
            values reference(size_t(g.rows) * g.columns * g.query_heads);
            for (int column = 0; column < g.columns; ++column) {
                const auto one = compute(backend.get(), data, {column}, false, false);
                ++graphs;
                for (int head = 0; head < g.query_heads; ++head) {
                    std::copy_n(one.data() + size_t(head) * g.rows, g.rows,
                                reference.data() + (size_t(head) * g.columns + column) * g.rows);
                }
            }
            maximum = std::max(maximum, oracle(data, reference));
            for (unsigned layout = 0; layout < 8; ++layout) {
                if (layout == 4) { continue; } // no padded operand: duplicates layout zero
                for (const bool reverse : {false, true}) {
                    std::vector<int> order(size_t(g.columns));
                    std::iota(order.begin(), order.end(), 0);
                    if (reverse) { std::reverse(order.begin(), order.end()); }
                    // Bit 2 keeps rows contiguous while padding only the head
                    // planes. This must also take the packed F32 staging route.
                    const auto actual = compute(backend.get(), data, order, (layout & 1) != 0,
                                                (layout & 2) != 0, 0, (layout & 4) != 0);
                    ++graphs;
                    values canonical(actual.size());
                    for (int head = 0; head < g.query_heads; ++head) {
                        for (int column = 0; column < g.columns; ++column) {
                            std::copy_n(actual.data() + (size_t(head) * g.columns + column) * g.rows, g.rows,
                                        canonical.data() + (size_t(head) * g.columns + order[size_t(column)]) * g.rows);
                        }
                    }
                    exact(canonical, reference, "PEDANTIC K=" + std::to_string(g.width) +
                        " M=" + std::to_string(g.rows) + " N=" + std::to_string(g.columns) +
                        " layout=" + std::to_string(layout) + (reverse ? " reversed" : " ordered"));
                    ++comparisons;
                }
            }
            if (g.query_heads == 1) {
                // N=1 and no outer heads matches the Vulkan MMV bias-fusion
                // predicate. A PEDANTIC MM dispatch must still execute every
                // one- or two-bias ADD node, with both operand orders covered.
                for (int count = 1; count <= 2; ++count) {
                    values expected = reference;
                    for (int index = 0; index < count; ++index) {
                        for (size_t i = 0; i < expected.size(); ++i) {
                            expected[i] += bias_value(index, int(i % size_t(g.rows)));
                        }
                    }
                    for (int column = 0; column < g.columns; ++column) {
                        const auto actual = compute(backend.get(), data, {column}, false, false, count);
                        const values one(expected.begin() + size_t(column) * g.rows,
                                         expected.begin() + size_t(column + 1) * g.rows);
                        exact(actual, one, "PEDANTIC serial product with " + std::to_string(count) + " biases");
                        ++graphs;
                        ++bias_graphs;
                        ++comparisons;
                    }
                    std::vector<int> order(size_t(g.columns));
                    std::iota(order.begin(), order.end(), 0);
                    const auto actual = compute(backend.get(), data, order, false, false, count);
                    exact(actual, expected, "PEDANTIC batched product with " + std::to_string(count) + " biases");
                    ++graphs;
                    ++bias_graphs;
                    ++comparisons;
                }
            }
        }
        std::cout << std::setprecision(17)
                  << "{\"status\":\"PASS\",\"scope\":\"F32 PEDANTIC batch, stride, and query-order invariance\""
                  << ",\"device\":" << json_string(device)
                  << ",\"graphs\":" << graphs << ",\"exact_comparisons\":" << comparisons
                  << ",\"bias_graphs\":" << bias_graphs
                  << ",\"layout_codes\":[0,1,2,3,5,6,7],\"head_only_padding\":true"
                  << ",\"geometries\":[[513,33,3,2,4],[2048,32,2,1,1],[2048,32,33,2,4],[65,65,65,2,4]]"
                  << ",\"geometry_fields\":[\"K\",\"M\",\"N\",\"kv_heads\",\"query_heads\"]"
                  << ",\"f64_oracle_max_abs\":" << maximum << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "F32 shape test failed: " << error.what() << '\n';
        return 1;
    }
}
