#include "ggml.h"
#include "ggml-backend.h"
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

constexpr int64_t width = 256;
constexpr int64_t columns = 2;
constexpr int64_t outputs = 32;

enum class graph_kind { original_reference, scaled_reference, disjoint_control, inplace_activation };

struct result {
    std::vector<float> first;
    std::vector<float> scale;
    std::vector<float> last;
    std::vector<float> storage;
};

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> input_values(bool disjoint, float factor = 1.0f) {
    std::vector<float> values(static_cast<size_t>((width + 1) * columns));
    for (int64_t row = 0; row < columns; ++row) {
        for (int64_t col = 0; col <= width; ++col) {
            const float value = col == width ? -17.0f :
                0.25f + static_cast<float>((col * 13 + row * 7 + (disjoint ? 11 : 0)) % 47) / 64.0f;
            values[static_cast<size_t>(row * (width + 1) + col)] = factor * value;
        }
    }
    return values;
}

template<typename T>
void set(ggml_tensor * tensor, const std::vector<T> & values) {
    require(tensor != nullptr && ggml_nbytes(tensor) == values.size() * sizeof(T), "tensor size mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(T));
}

std::vector<float> read(ggml_tensor * tensor) {
    require(tensor != nullptr && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor),
            "readback requires a contiguous F32 tensor");
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

void poison(ggml_tensor * tensor) {
    set(tensor, std::vector<float>(static_cast<size_t>(ggml_nelements(tensor)),
                                  std::numeric_limits<float>::quiet_NaN()));
}

result run(ggml_backend_t backend, graph_kind kind) {
    constexpr size_t capacity = 32;
    const ggml_init_params params = {
        capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, false), nullptr, true,
    };
    context_ptr context(ggml_init(params), ggml_free);
    require(context != nullptr, "metadata allocation failed");
    auto * ctx = context.get();
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width + 1, columns);
    auto * view = ggml_view_2d(ctx, storage, width, columns, static_cast<size_t>(width + 1) * sizeof(float), sizeof(float));
    auto * weights0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, width, outputs);
    auto * first = ggml_mul_mat(ctx, weights0, view);
    ggml_tensor * weights1 = nullptr;
    ggml_tensor * scale_input = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * last = nullptr;
    const bool combined = kind == graph_kind::disjoint_control || kind == graph_kind::inplace_activation;
    require(ggml_is_contiguous(storage) && !ggml_is_contiguous(view), "invalid activation storage geometry");
    require(view->view_offs == sizeof(float), "activation view must have a nonzero storage offset");
    ggml_set_input(storage);
    ggml_set_input(weights0);
    ggml_set_output(first);
    if (combined) {
        scale_input = kind == graph_kind::inplace_activation ? storage :
                      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width + 1, columns);
        ggml_set_input(scale_input);
        scale = ggml_scale_inplace(ctx, scale_input, 2.0f);
        weights1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, width, outputs);
        // Keep the original view pointer. Reading from a newly created scale
        // node would change the activation cache key and hide source mutation.
        last = ggml_mul_mat(ctx, weights1, view);
        require(first->src[1] == view && last->src[1] == view, "matmuls must share one activation view");
        require(scale->view_src == scale_input, "SCALE must update its input storage in place");
        ggml_set_input(weights1);
        ggml_set_output(scale);
        ggml_set_output(last);
    }

    auto * graph = ggml_new_graph_custom(ctx, capacity, false);
    std::vector<ggml_tensor *> order;
    for (auto * output : {first, scale, last}) {
        if (output) {
            require(ggml_backend_supports_op(backend, output), "Vulkan does not support a probe operation");
            ggml_build_forward_expand(graph, output);
            order.push_back(output);
        }
    }
    size_t next = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_VIEW) continue;
        require(next < order.size() && node == order[next], "unexpected public graph operation order");
        ++next;
    }
    require(next == order.size(), "missing public graph operation");

    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "tensor allocation failed");
    set(storage, input_values(false, kind == graph_kind::scaled_reference ? 2.0f : 1.0f));
    if (kind == graph_kind::disjoint_control) set(scale_input, input_values(true));
    std::vector<ggml_fp16_t> weights(static_cast<size_t>(width * outputs));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = ggml_fp32_to_fp16(0.03125f + static_cast<float>((i * 7 + i / width) % 19) / 128.0f);
    }
    set(weights0, weights);
    if (weights1) set(weights1, weights);
    poison(first);
    if (last) poison(last);
    // SCALE aliases initialized input. Poisoning that output would destroy the
    // input; its complete readback must instead equal the known doubled data.
    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    require(status == GGML_STATUS_SUCCESS, "direct Vulkan graph computation failed");
    result output;
    output.first = read(first);
    output.storage = read(storage);
    if (scale) output.scale = read(scale);
    if (last) output.last = read(last);
    return output;
}

bool compare(const std::string & label, const std::vector<float> & expected, const std::vector<float> & actual) {
    require(!expected.empty() && expected.size() == actual.size(), "comparison size mismatch");
    size_t bad = 0;
    size_t first = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(static_cast<double>(expected[i]) - actual[i]);
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) || expected[i] != actual[i]) {
            if (bad++ == 0) first = i;
        }
        if (std::isfinite(error)) max_abs = std::max(max_abs, error);
    }
    std::cout << (bad ? "FAIL " : "PASS ") << label << " compared=" << actual.size()
              << " mismatches=" << bad << " max_abs=" << max_abs << " tolerance=0";
    if (bad) std::cout << " first=" << first << " expected=" << expected[first] << " actual=" << actual[first];
    std::cout << '\n';
    return bad == 0;
}

bool check(ggml_backend_t backend, graph_kind kind, const result & original, const result & scaled) {
    const auto actual = run(backend, kind);
    const bool mutation = kind == graph_kind::inplace_activation;
    const std::string label = mutation ? "inplace_activation" : "disjoint_scale_control";
    bool pass = compare(label + "/first_vs_original_input", original.first, actual.first);
    pass &= compare(label + "/last_vs_independent_input", mutation ? scaled.first : original.first, actual.last);
    pass &= compare(label + "/scale_full_storage", input_values(!mutation, 2.0f), actual.scale);
    pass &= compare(label + "/activation_storage", input_values(false, mutation ? 2.0f : 1.0f), actual.storage);
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        require(argc <= 3, "usage: test-vulkan-activation-cache [device-index] [all|control|inplace]");
        size_t device = 0;
        if (argc >= 2) {
            const std::string value = argv[1];
            size_t end = 0;
            const unsigned long parsed = std::stoul(value, &end);
            require(!value.empty() && value[0] >= '0' && value[0] <= '9' && end == value.size(), "invalid device index");
            require(parsed <= std::numeric_limits<size_t>::max(), "device index out of range");
            device = static_cast<size_t>(parsed);
        }
        const std::string mode = argc == 3 ? argv[2] : "all";
        require(mode == "all" || mode == "control" || mode == "inplace", "unknown probe mode");
        backend_ptr backend(ggml_backend_vk_init(device), ggml_backend_free);
        require(backend != nullptr && ggml_backend_is_vk(backend.get()), "Vulkan initialization failed");
        std::cout << "Vulkan activation cache backend=" << ggml_backend_name(backend.get())
                  << " device=" << device << " mode=" << mode << " view_offset_bytes=" << sizeof(float) << '\n';
        const auto original = run(backend.get(), graph_kind::original_reference);
        const auto scaled = run(backend.get(), graph_kind::scaled_reference);
        require(original.first.size() == static_cast<size_t>(outputs * columns) && original.first.size() == scaled.first.size(),
                "invalid independent matmul controls");
        for (size_t i = 0; i < original.first.size(); ++i) {
            require(std::isfinite(original.first[i]) && original.first[i] > 0 &&
                    std::isfinite(scaled.first[i]) && scaled.first[i] > original.first[i],
                    "independent original and scaled matmul controls must differ");
        }
        bool pass = compare("reference/original_storage", input_values(false), original.storage);
        pass &= compare("reference/scaled_storage", input_values(false, 2.0f), scaled.storage);
        size_t cases = 0;
        size_t failed = 0;
        if (mode == "all" || mode == "control") {
            ++cases;
            failed += !check(backend.get(), graph_kind::disjoint_control, original, scaled);
        }
        if (mode == "all" || mode == "inplace") {
            ++cases;
            failed += !check(backend.get(), graph_kind::inplace_activation, original, scaled);
        }
        std::cout << "Vulkan activation cache cases=" << cases << " failed=" << failed << '\n';
        return pass && failed == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "FAIL Vulkan activation cache: " << error.what() << '\n';
        return 1;
    }
}
