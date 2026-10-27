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

constexpr int64_t mm_k = 256;
constexpr int64_t mm_m = 32;
constexpr int64_t mm_n = 2;
constexpr int64_t fa_width = 64;
constexpr int64_t fa_queries = 32;
constexpr int64_t fa_keys = 4096;

enum class layout { matmul_only, middle_only, sandwich };
enum class middle_op { softmax_small, softmax_large, flash_unmasked, flash_masked, flash_masked_no_split };

struct result {
    std::vector<float> before;
    std::vector<float> middle;
    std::vector<float> after;
};

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const char * name(middle_op op) {
    switch (op) {
        case middle_op::softmax_small: return "softmax_16384_control";
        case middle_op::softmax_large: return "softmax_32768";
        case middle_op::flash_unmasked: return "flash_unmasked_control";
        case middle_op::flash_masked: return "flash_masked_4096x32";
        case middle_op::flash_masked_no_split: return "flash_masked_no_split_target_qh256_kvh1";
    }
    throw std::runtime_error("invalid middle operation");
}

bool is_softmax(middle_op op) {
    return op == middle_op::softmax_small || op == middle_op::softmax_large;
}

int64_t softmax_columns(middle_op op) {
    return op == middle_op::softmax_small ? 16384 : 32768;
}

int64_t query_heads(middle_op op) {
    return op == middle_op::flash_masked_no_split ? 256 : 1;
}

float no_split_scale(size_t channel) {
    const float scales[] = {0.25f, 0.5f, 1.0f};
    return scales[channel % 3];
}

template<typename T>
void set(ggml_tensor * tensor, const std::vector<T> & values) {
    require(tensor != nullptr && ggml_nbytes(tensor) == values.size() * sizeof(T),
            "input tensor size mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(T));
}

std::vector<float> poisoned(ggml_tensor * tensor) {
    require(tensor != nullptr && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor),
            "invalid output tensor");
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)),
                              std::numeric_limits<float>::quiet_NaN());
    set(tensor, values);
    return values;
}

// The reference and sandwich use the same GPU, shapes, inputs and public ops.
// No scheduler or CPU fallback can replace an operation. Distinct weight
// tensors have equal contents; both matmuls consume the identical strided view.
result run(ggml_backend_t backend, middle_op op, layout graph_layout) {
    constexpr size_t capacity = 64;
    const ggml_init_params params = {
        capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, false),
        nullptr,
        true,
    };
    context_ptr context(ggml_init(params), ggml_free);
    require(context != nullptr, "metadata allocation failed");
    ggml_context * ctx = context.get();
    ggml_tensor * activation = nullptr;
    ggml_tensor * weights0 = nullptr;
    ggml_tensor * weights1 = nullptr;
    ggml_tensor * before = nullptr;
    ggml_tensor * middle = nullptr;
    ggml_tensor * after = nullptr;
    ggml_tensor * softmax_input = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * mask = nullptr;

    if (graph_layout != layout::middle_only) {
        activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mm_k + 1, mm_n);
        auto * view = ggml_view_2d(ctx, activation, mm_k, mm_n,
                                  static_cast<size_t>(mm_k + 1) * sizeof(float), 0);
        require(!ggml_is_contiguous(view), "matmul input must require strided access");
        weights0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mm_k, mm_m);
        before = ggml_mul_mat(ctx, weights0, view);
        ggml_set_input(activation);
        ggml_set_input(weights0);
        ggml_set_output(before);
        if (graph_layout == layout::sandwich) {
            weights1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mm_k, mm_m);
            after = ggml_mul_mat(ctx, weights1, view);
            ggml_set_input(weights1);
            ggml_set_output(after);
        }
    }
    if (graph_layout != layout::matmul_only) {
        if (is_softmax(op)) {
            softmax_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, softmax_columns(op), 1);
            middle = ggml_soft_max(ctx, softmax_input);
            ggml_set_input(softmax_input);
        } else {
            q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, fa_width, fa_queries, query_heads(op), 1);
            k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, fa_width, fa_keys, 1, 1);
            v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, fa_width, fa_keys, 1, 1);
            if (op == middle_op::flash_masked || op == middle_op::flash_masked_no_split) {
                mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, fa_keys, fa_queries, 1, 1);
                ggml_set_input(mask);
            }
            middle = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(middle, GGML_PREC_F32);
            ggml_set_input(q);
            ggml_set_input(k);
            ggml_set_input(v);
        }
        ggml_set_output(middle);
    }

    auto * graph = ggml_new_graph_custom(ctx, capacity, false);
    std::vector<ggml_tensor *> expected_order;
    for (auto * output : {before, middle, after}) {
        if (output) {
            require(ggml_backend_supports_op(backend, output), "Vulkan does not support a test operation");
            ggml_build_forward_expand(graph, output);
            expected_order.push_back(output);
        }
    }
    size_t next = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_VIEW) {
            continue;
        }
        require(next < expected_order.size() && node == expected_order[next],
                "unexpected public graph operation order");
        ++next;
    }
    require(next == expected_order.size(), "missing public graph operation");

    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "tensor allocation failed");
    if (activation) {
        std::vector<float> input(static_cast<size_t>((mm_k + 1) * mm_n));
        for (int64_t row = 0; row < mm_n; ++row) {
            for (int64_t col = 0; col <= mm_k; ++col) {
                input[static_cast<size_t>(row * (mm_k + 1) + col)] = col == mm_k ? -17.0f :
                    0.25f + static_cast<float>((col * 13 + row * 7) % 47) / 64.0f;
            }
        }
        set(activation, input);
        std::vector<ggml_fp16_t> weights(static_cast<size_t>(mm_k * mm_m));
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = ggml_fp32_to_fp16(0.03125f + static_cast<float>((i * 7 + i / mm_k) % 19) / 128.0f);
        }
        set(weights0, weights);
        if (weights1) {
            set(weights1, weights);
        }
    }
    if (softmax_input) {
        set(softmax_input, std::vector<float>(static_cast<size_t>(softmax_columns(op)), 0.0f));
    }
    if (q) {
        set(q, std::vector<float>(static_cast<size_t>(fa_width * fa_queries * query_heads(op)), 0.0f));
        set(k, std::vector<ggml_fp16_t>(static_cast<size_t>(fa_width * fa_keys), ggml_fp32_to_fp16(0.0f)));
        std::vector<ggml_fp16_t> values(static_cast<size_t>(fa_width * fa_keys));
        for (size_t i = 0; i < values.size(); ++i) {
            float value = 0.125f + static_cast<float>(i % fa_width) / 128.0f;
            if (op == middle_op::flash_masked_no_split) {
                // Equal visible bands have values 0 and 2*scale. Powers of two
                // keep their sums exact; wrong key-block selection changes the mean.
                const auto key = i / fa_width;
                const float scale = no_split_scale(i % fa_width);
                value = key < 1024 ? 0.0f : key < 2048 ? 2.0f * scale : 8.0f * scale;
            }
            values[i] = ggml_fp32_to_fp16(value);
        }
        set(v, values);
        if (mask) {
            std::vector<ggml_fp16_t> masks(static_cast<size_t>(fa_keys * fa_queries));
            for (size_t i = 0; i < masks.size(); ++i) {
                masks[i] = ggml_fp32_to_fp16(i % fa_keys < fa_keys / 2 ? 0.0f :
                                            -std::numeric_limits<float>::infinity());
            }
            set(mask, masks);
        }
    }

    result output;
    if (before) output.before = poisoned(before);
    if (middle) output.middle = poisoned(middle);
    if (after) output.after = poisoned(after);
    const auto status = ggml_backend_graph_compute(backend, graph);
    // Synchronize even on a failed submission before destroying tensor buffers.
    ggml_backend_synchronize(backend);
    require(status == GGML_STATUS_SUCCESS, "direct Vulkan graph computation failed");
    if (before) ggml_backend_tensor_get(before, output.before.data(), 0, output.before.size() * sizeof(float));
    if (middle) ggml_backend_tensor_get(middle, output.middle.data(), 0, output.middle.size() * sizeof(float));
    if (after) ggml_backend_tensor_get(after, output.after.data(), 0, output.after.size() * sizeof(float));
    return output;
}

bool compare(const std::string & label, const std::vector<float> & expected,
             const std::vector<float> & actual, double tolerance = 0.0) {
    require(!expected.empty() && expected.size() == actual.size(), "comparison size mismatch");
    size_t bad = 0;
    size_t first = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(static_cast<double>(expected[i]) - actual[i]);
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) || error > tolerance) {
            if (bad++ == 0) first = i;
        }
        if (std::isfinite(error)) max_abs = std::max(max_abs, error);
    }
    std::cout << (bad ? "FAIL " : "PASS ") << label << " compared=" << actual.size()
              << " mismatches=" << bad << " max_abs=" << max_abs << " tolerance=" << tolerance;
    if (bad) {
        std::cout << " first=" << first << " expected=" << expected[first] << " actual=" << actual[first];
    }
    std::cout << '\n';
    return bad == 0;
}

bool check(ggml_backend_t backend, middle_op op, const std::vector<float> & matmul_reference) {
    const result isolated = run(backend, op, layout::middle_only);
    const result combined = run(backend, op, layout::sandwich);
    const std::string label = name(op);
    bool pass = compare(label + "/first_vs_isolated_matmul", matmul_reference, combined.before);
    pass &= compare(label + "/last_vs_isolated_matmul", matmul_reference, combined.after);
    pass &= compare(label + "/middle_vs_isolated", isolated.middle, combined.middle);
    std::vector<float> known(isolated.middle.size());
    for (size_t i = 0; i < known.size(); ++i) {
        known[i] = is_softmax(op) ? 1.0f / static_cast<float>(softmax_columns(op)) :
                   0.125f + static_cast<float>(i % fa_width) / 128.0f;
        if (op == middle_op::flash_masked_no_split) {
            // Q.K is zero: mean(1024 zeros, 1024 values of 2*scale) = scale.
            known[i] = no_split_scale(i % fa_width);
        }
    }
    pass &= compare(label + "/middle_closed_form", known, isolated.middle, is_softmax(op) ? 0.0 : 2.0e-4);
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        require(argc <= 3, "usage: test-vulkan-scratch-reuse [device-index] [all|controls|softmax|flash-mask|flash-no-split]");
        size_t device = 0;
        if (argc >= 2) {
            const std::string value = argv[1];
            size_t end = 0;
            const unsigned long parsed = std::stoul(value, &end);
            require(!value.empty() && value[0] >= '0' && value[0] <= '9' && end == value.size(),
                    "invalid device index");
            require(parsed <= std::numeric_limits<size_t>::max(), "device index out of range");
            device = static_cast<size_t>(parsed);
        }
        const std::string mode = argc == 3 ? argv[2] : "all";
        require(mode == "all" || mode == "controls" || mode == "softmax" || mode == "flash-mask" || mode == "flash-no-split",
                "unknown test mode");
        backend_ptr backend(ggml_backend_vk_init(device), ggml_backend_free);
        require(backend != nullptr && ggml_backend_is_vk(backend.get()), "Vulkan initialization failed");
        std::cout << "Vulkan scratch reuse backend=" << ggml_backend_name(backend.get())
                  << " device=" << device << " mode=" << mode << '\n';
        const auto reference = run(backend.get(), middle_op::softmax_small, layout::matmul_only).before;
        require(std::all_of(reference.begin(), reference.end(), [](float x) { return std::isfinite(x) && x > 0.0f; }),
                "isolated matmul control is not finite and positive");
        size_t cases = 0;
        size_t failed = 0;
        for (auto op : {middle_op::softmax_small, middle_op::softmax_large,
                        middle_op::flash_unmasked, middle_op::flash_masked, middle_op::flash_masked_no_split}) {
            const bool control = op == middle_op::softmax_small || op == middle_op::flash_unmasked;
            if (mode != "all" && !(mode == "controls" && control) &&
                !(mode == "softmax" && op == middle_op::softmax_large) &&
                !(mode == "flash-mask" && op == middle_op::flash_masked) &&
                !(mode == "flash-no-split" && op == middle_op::flash_masked_no_split)) {
                continue;
            }
            ++cases;
            failed += !check(backend.get(), op, reference);
        }
        std::cout << "Vulkan scratch reuse cases=" << cases << " failed=" << failed << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "FAIL Vulkan scratch reuse: " << error.what() << '\n';
        return 1;
    }
}
