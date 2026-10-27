#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
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
    return result + '"';
}

std::string environment(const char * name) {
    const char * value = std::getenv(name);
    return value ? json_string(value) : "null";
}

struct geometry {
    int width;
    int rows;
    int experts;
    int slots;
    int tokens;
    int activation_slots;
    int id_stride;
};

struct variant {
    const char * name;
    bool bias;
    bool scale;
};

// These immutable tensors remain resident through all graphs for one fixture.
// A reference graph and a cohort graph use the same quantized weight bytes.
struct fixture {
    ggml_type type;
    geometry shape;
    values activations;
    std::vector<int32_t> ids;
    values bias_data;
    values scale_data;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{
        nullptr, ggml_backend_buffer_free};
    ggml_tensor * weights = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * scale = nullptr;

    fixture(ggml_backend_t backend, ggml_type selected_type, geometry selected_shape)
        : type(selected_type), shape(selected_shape),
          activations(size_t(shape.width) * shape.activation_slots * shape.tokens),
          ids(size_t(shape.id_stride) * shape.tokens),
          bias_data(size_t(shape.rows) * shape.experts), scale_data(size_t(shape.slots)) {
        const auto & g = shape;
        require(g.width % ggml_blck_size(type) == 0, "invalid quantized row width");
        require((g.activation_slots == 1 || g.activation_slots == g.slots) &&
                g.id_stride >= g.slots && g.slots > 1 && g.slots <= g.experts,
                "invalid expert selection geometry");
        uint32_t state = 0x537a9de1;
        const auto next = [&state]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return float(int32_t(state & 0xffffu) - 32768) / 10009.0f;
        };
        values original(size_t(g.width) * g.rows * g.experts);
        for (float & value : original) { value = next(); }
        for (float & value : activations) { value = next(); }
        for (int token = 0; token < g.tokens; ++token) {
            for (int slot = 0; slot < g.id_stride; ++slot) {
                // Padding contains valid but different IDs. Reading it as a
                // selected slot must fail numerically, without an invalid ID.
                int expert = (3 * token + slot + 4) % g.experts;
                if (slot < g.slots) {
                    int rank = slot == g.slots - 1 ? 0 : slot;
                    if ((token & 1) && rank != 0) { rank = g.slots - 1 - rank; }
                    expert = (5 * token + token / 3 + 3 * rank + 1) % g.experts;
                }
                ids[size_t(token) * g.id_stride + slot] = expert;
            }
        }
        for (int expert = 0; expert < g.experts; ++expert) {
            for (int row = 0; row < g.rows; ++row) {
                bias_data[size_t(expert) * g.rows + row] = float((7 * row + 11 * expert) % 37 - 18) / 32.0f;
            }
        }
        for (int slot = 0; slot < g.slots; ++slot) {
            // Distinct signed powers of two expose incorrect slot indexing;
            // their exact scaling also permits a host F32 oracle after ADD_ID.
            scale_data[size_t(slot)] = std::ldexp((slot & 1) ? -1.0f : 1.0f, slot - 1);
        }
        std::vector<uint8_t> quantized(ggml_row_size(type, g.width) * g.rows * g.experts);
        values imatrix;
        if (ggml_quantize_requires_imatrix(type)) { imatrix.assign(size_t(g.width), 1.0f); }
        const size_t written = ggml_quantize_chunk(type, original.data(), quantized.data(), 0,
            int64_t(g.rows) * g.experts, g.width, imatrix.empty() ? nullptr : imatrix.data());
        require(written == quantized.size(), "quantized expert byte count mismatch");
        const ggml_init_params params{3 * ggml_tensor_overhead(), nullptr, true};
        context.reset(ggml_init(params));
        require(context != nullptr, "could not create resident tensor context");
        weights = ggml_new_tensor_3d(context.get(), type, g.width, g.rows, g.experts);
        bias = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, g.rows, g.experts);
        scale = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 1, g.slots, 1);
        ggml_set_name(weights, "id-expert-weights");
        ggml_set_name(bias, "id-expert-bias");
        ggml_set_name(scale, "id-shared-slot-scale");
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(buffer != nullptr, "could not allocate resident tensors");
        ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
        ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
        ggml_backend_tensor_set(scale, scale_data.data(), 0, scale_data.size() * sizeof(float));
    }

    std::string label() const {
        return std::string(ggml_type_name(type)) + " K=" + std::to_string(shape.width) +
            " M=" + std::to_string(shape.rows) + " E=" + std::to_string(shape.experts) +
            " I=" + std::to_string(shape.slots) + " source_N=" + std::to_string(shape.tokens) +
            " B_slots=" + std::to_string(shape.activation_slots) +
            " ids_stride=" + std::to_string(shape.id_stride);
    }
};

values compute(ggml_backend_t backend, const fixture & data, const std::vector<int> & order,
               ggml_prec precision, variant selected) {
    const auto & g = data.shape;
    const int count = int(order.size());
    require(count > 0, "empty query order");
    const ggml_init_params params{
        10 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    require(context != nullptr, "could not create graph context");
    auto * activations = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, g.width, g.activation_slots, count);
    auto * ids_storage = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, g.id_stride, count);
    auto * ids = ggml_view_2d(context.get(), ids_storage, g.slots, count, size_t(g.id_stride) * sizeof(int32_t), 0);
    values reordered(size_t(g.width) * g.activation_slots * count);
    std::vector<int32_t> reordered_ids(size_t(g.id_stride) * count);
    for (int token = 0; token < count; ++token) {
        const int original = order[size_t(token)];
        require(original >= 0 && original < g.tokens, "invalid query order");
        const size_t activation_size = size_t(g.width) * g.activation_slots;
        std::copy_n(data.activations.data() + size_t(original) * activation_size, activation_size,
                    reordered.data() + size_t(token) * activation_size);
        std::copy_n(data.ids.data() + size_t(original) * g.id_stride, g.id_stride,
                    reordered_ids.data() + size_t(token) * g.id_stride);
    }
    auto * output = ggml_mul_mat_id(context.get(), data.weights, activations, ids);
    // Omitting the DEFAULT setter is intentional: --native-matmul can link to
    // the old library, whose setter accepts MUL_MAT but rejects MUL_MAT_ID.
    // Every single-column reference also follows that unchanged native API.
    if (precision != GGML_PREC_DEFAULT) { ggml_mul_mat_set_prec(output, precision); }
    require(ggml_backend_supports_op(backend, output), data.label() + " backend rejected MUL_MAT_ID");
    if (selected.bias) {
        output = ggml_add_id(context.get(), output, data.bias, ids);
        require(ggml_backend_supports_op(backend, output), "backend rejected ADD_ID");
    }
    if (selected.scale) {
        output = ggml_mul(context.get(), output, data.scale);
        require(ggml_backend_supports_op(backend, output), "backend rejected per-slot MUL");
    }
    require(output->type == GGML_TYPE_F32 && output->ne[0] == g.rows && output->ne[1] == g.slots &&
            output->ne[2] == count && output->ne[3] == 1, "unexpected indirect product shape");
    ggml_set_input(activations);
    ggml_set_input(ids_storage);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "could not allocate graph tensors");
    ggml_backend_tensor_set(activations, reordered.data(), 0, reordered.size() * sizeof(float));
    ggml_backend_tensor_set(ids_storage, reordered_ids.data(), 0, reordered_ids.size() * sizeof(int32_t));
    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    require(status == GGML_STATUS_SUCCESS, data.label() + " indirect graph failed");
    values result(size_t(g.rows) * g.slots * count);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    for (float value : result) { require(std::isfinite(value), "non-finite indirect graph result"); }
    return result;
}

values expected_for(const fixture & data, const values & reference, const std::vector<int> & order,
                    variant selected) {
    const auto & g = data.shape;
    values expected(size_t(g.rows) * g.slots * order.size());
    for (size_t token = 0; token < order.size(); ++token) {
        const int original = order[token];
        for (int slot = 0; slot < g.slots; ++slot) {
            const int expert = data.ids[size_t(original) * g.id_stride + slot];
            for (int row = 0; row < g.rows; ++row) {
                float value = reference[(size_t(original) * g.slots + slot) * g.rows + row];
                if (selected.bias) { value += data.bias_data[size_t(expert) * g.rows + row]; }
                if (selected.scale) { value *= data.scale_data[size_t(slot)]; }
                expected[(token * g.slots + slot) * g.rows + row] = value;
            }
        }
    }
    return expected;
}

void exact(const values & actual, const values & expected, const std::string & label) {
    require(actual.size() == expected.size(), "comparison shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            uint32_t actual_bits;
            uint32_t expected_bits;
            std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
            std::memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
            std::ostringstream error;
            error << std::setprecision(9) << label << " mismatch at " << i << ": actual=" << actual[i]
                  << " reference=" << expected[i] << " bits=0x" << std::hex << actual_bits
                  << "/0x" << expected_bits;
            throw std::runtime_error(error.str());
        }
    }
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const char * requested = "Vulkan0";
        ggml_prec precision = GGML_PREC_BATCH_INVARIANT;
        bool selected_device = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--native-matmul") { precision = GGML_PREC_DEFAULT; }
            else if (arg == "--help") {
                std::cout << "usage: test-mul-mat-id-shape-invariance [device] [--native-matmul]\n";
                return 0;
            } else {
                require(!selected_device && !arg.empty() && arg.front() != '-',
                        "unknown argument; use --help for supported options");
                requested = argv[i];
                selected_device = true;
            }
        }
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(requested, nullptr), ggml_backend_free);
        require(backend != nullptr, std::string("backend is unavailable: ") + requested);
        const ggml_type types[] = {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K};
        const geometry shapes[] = {
            {2048, 33, 8, 4, 9, 1, 4},
            {768, 65, 8, 3, 32, 3, 8},
            {2048, 32, 16, 4, 128, 1, 4},
        };
        const variant variants[] = {{"plain", false, false}, {"bias", true, false},
                                    {"scale", false, true}, {"bias_scale", true, true}};
        size_t graphs = 0;
        size_t reference_graphs = 0;
        size_t native_control_graphs = 0;
        size_t cohort_graphs = 0;
        size_t boundary_graphs = 0;
        size_t comparisons = 0;
        size_t bias_graphs = 0;
        size_t scale_graphs = 0;
        for (const auto type : types) {
            for (const auto & g : shapes) {
                const fixture data(backend.get(), type, g);
                values reference(size_t(g.rows) * g.slots * g.tokens);
                for (int token = 0; token < g.tokens; ++token) {
                    const auto one = compute(backend.get(), data, {token}, GGML_PREC_DEFAULT, variants[0]);
                    ++graphs;
                    ++reference_graphs;
                    std::copy(one.begin(), one.end(), reference.begin() + size_t(token) * g.slots * g.rows);
                }
                require(std::any_of(reference.begin(), reference.end(),
                                    [](float value) { return std::abs(value) > 1.0f; }),
                        "degenerate native single-column reference");
                // Check the independent host ADD_ID/MUL oracle against native
                // single-column graph semantics before judging a cohort.
                for (const auto selected : variants) {
                    if (!selected.bias && !selected.scale) { continue; }
                    const std::vector<int> order{g.tokens - 1};
                    const auto actual = compute(backend.get(), data, order, GGML_PREC_DEFAULT, selected);
                    ++graphs;
                    ++native_control_graphs;
                    bias_graphs += selected.bias;
                    scale_graphs += selected.scale;
                    exact(actual, expected_for(data, reference, order, selected),
                        data.label() + " native N=1 control " + selected.name);
                    ++comparisons;
                }
                for (const int count : {2, 8, 9, 32, 128}) {
                    if (count > g.tokens) { continue; }
                    for (const bool reverse : {false, true}) {
                        std::vector<int> order(static_cast<size_t>(count));
                        std::iota(order.begin(), order.end(), 0);
                        if (reverse) { std::reverse(order.begin(), order.end()); }
                        for (const auto selected : variants) {
                            const auto actual = compute(backend.get(), data, order, precision, selected);
                            ++graphs;
                            ++cohort_graphs;
                            bias_graphs += selected.bias;
                            scale_graphs += selected.scale;
                            exact(actual, expected_for(data, reference, order, selected),
                                data.label() + " N=" + std::to_string(count) +
                                (reverse ? " reversed " : " ordered ") + selected.name);
                            ++comparisons;
                        }
                    }
                }
            }
        }
        {
            // Four slots times 16385 tokens exceeds a 65535-workgroup Y
            // limit by five slots. A chunk can end inside a token. Periodic
            // inputs need only seven native N=1 references, not 16385.
            const geometry g{32, 3, 8, 4, 7, 1, 8};
            const fixture data(backend.get(), GGML_TYPE_Q8_0, g);
            values reference(size_t(g.rows) * g.slots * g.tokens);
            for (int token = 0; token < g.tokens; ++token) {
                const auto one = compute(backend.get(), data, {token}, GGML_PREC_DEFAULT, variants[0]);
                ++graphs;
                ++reference_graphs;
                std::copy(one.begin(), one.end(), reference.begin() + size_t(token) * g.slots * g.rows);
            }
            require(std::any_of(reference.begin(), reference.end(),
                                [](float value) { return std::abs(value) > 1.0f; }),
                    "degenerate dispatch-boundary reference");
            for (const auto selected : variants) {
                if (!selected.bias && !selected.scale) { continue; }
                const std::vector<int> order{g.tokens - 1};
                const auto actual = compute(backend.get(), data, order, GGML_PREC_DEFAULT, selected);
                ++graphs;
                ++native_control_graphs;
                bias_graphs += selected.bias;
                scale_graphs += selected.scale;
                exact(actual, expected_for(data, reference, order, selected),
                    data.label() + " boundary native N=1 control " + selected.name);
                ++comparisons;
            }
            for (const bool reverse : {false, true}) {
                std::vector<int> order(16385);
                for (size_t token = 0; token < order.size(); ++token) { order[token] = int(token % g.tokens); }
                if (reverse) { std::reverse(order.begin(), order.end()); }
                for (const auto selected : variants) {
                    const auto actual = compute(backend.get(), data, order, precision, selected);
                    ++graphs;
                    ++boundary_graphs;
                    bias_graphs += selected.bias;
                    scale_graphs += selected.scale;
                    exact(actual, expected_for(data, reference, order, selected),
                        data.label() + " N=16385" + (reverse ? " reversed " : " ordered ") + selected.name);
                    ++comparisons;
                }
            }
        }
        ggml_quantize_free();
        std::cout << "{\"status\":\"pass\",\"device\":"
                  << json_string(ggml_backend_dev_name(ggml_backend_get_device(backend.get())))
                  << ",\"backend\":" << json_string(ggml_backend_name(backend.get()))
                  << ",\"matmul_id_mode\":" << json_string(precision == GGML_PREC_DEFAULT ?
                      "native_default" : "batch_invariant_native_quantized")
                  << ",\"reference\":\"native_default_single_column\",\"default_setter_omitted\":true"
                  << ",\"types\":[\"iq3_s\",\"iq4_nl\",\"q8_0\",\"q4_K\"]"
                  << ",\"graphs\":" << graphs << ",\"reference_graphs\":" << reference_graphs
                  << ",\"native_control_graphs\":" << native_control_graphs
                  << ",\"cohort_graphs\":" << cohort_graphs << ",\"bitwise_comparisons\":" << comparisons
                  << ",\"boundary_graphs\":" << boundary_graphs << ",\"boundary_flattened_slots\":65540"
                  << ",\"bias_graphs\":" << bias_graphs << ",\"scale_graphs\":" << scale_graphs
                  << ",\"cohort_sizes\":[2,8,9,32,128],\"orders\":[\"ordered\",\"reversed\"]"
                  << ",\"shared_and_per_slot_activations\":true,\"padded_id_rows\":true"
                  << ",\"repeated_and_varying_expert_ids\":true,\"shared_per_slot_scale\":true"
                  << ",\"uniform_imatrix_when_required\":true"
                  << ",\"disable_mmvq\":" << environment("GGML_VK_DISABLE_MMVQ")
                  << ",\"force_mmvq\":" << environment("GGML_VK_FORCE_MMVQ")
                  << ",\"disable_f16\":" << environment("GGML_VK_DISABLE_F16")
                  << ",\"disable_fusion\":" << environment("GGML_VK_DISABLE_FUSION")
                  << ",\"actual_dispatch_observed\":false,\"model_or_performance_claim\":false}\n";
        return 0;
    } catch (const std::exception & error) {
        ggml_quantize_free();
        std::cerr << "mul-mat-id-shape-invariance: " << error.what() << '\n';
        return 1;
    }
}
