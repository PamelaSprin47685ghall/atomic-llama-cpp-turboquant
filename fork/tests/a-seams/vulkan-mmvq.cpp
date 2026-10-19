#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int64_t width = 2048;
constexpr std::int64_t rows = 1025; // Include the final partial row workgroup.
constexpr std::int64_t max_columns = 8;
constexpr std::int64_t q8_block = 32;
constexpr double unit_f32 = 0x1p-24;
constexpr double unit_f64 = 0x1p-53;

double gamma(std::int64_t count, double unit) {
    return count * unit / (1.0 - count * unit);
}

std::string quoted(const char * value) {
    if (value == nullptr) {
        return "null";
    }
    std::string result = "\"";
    for (const unsigned char c : std::string(value)) {
        if (c == '\\' || c == '"') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 32) {
            const char hex[] = "0123456789abcdef";
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

struct Activations {
    std::vector<float> values = std::vector<float>(std::size_t(width * max_columns));
    std::vector<int> integers = std::vector<int>(std::size_t(width * max_columns));
    std::vector<double> scales = std::vector<double>(std::size_t(width * max_columns / q8_block));
};

Activations make_activations() {
    Activations result;
    for (std::int64_t column = 0; column < max_columns; ++column) {
        for (std::int64_t block = 0; block < width / q8_block; ++block) {
            const auto offset = column * width + block * q8_block;
            // Quarter/three-quarter binary16 ULPs exercise scale rounding.
            // All values and the two exact extrema are binary32-exact.
            const float scale = std::ldexp(1.0f + float(1 + 2 * (block % 2)) / 4096.0f,
                                          -6 - int((block + column) % 4));
            result.scales[std::size_t(offset / q8_block)] = scale;
            for (int pair = 0; pair < 16; ++pair) {
                const auto hash = mix(std::uint32_t(offset + 17 * pair + 31));
                int first = pair == 0 ? 127 : 1 + int(hash % 119);
                int second = -first;
                float residual = pair == 0 ? 0.0f : ((hash & 256u) ? 0.25f : -0.25f);
                // Include an affine-weight witness. The integer block sum is
                // 128 in half the blocks and zero in the other half.
                if (pair == 1 && (block + column) % 2 != 0) {
                    first = second = 64;
                }
                result.integers[std::size_t(offset + 2 * pair)] = first;
                result.integers[std::size_t(offset + 2 * pair + 1)] = second;
                result.values[std::size_t(offset + 2 * pair)] = scale * (first + residual);
                result.values[std::size_t(offset + 2 * pair + 1)] = scale * (second - residual);
            }
            int integer_sum = 0;
            for (std::int64_t i = 0; i < q8_block; ++i) {
                const double ratio = result.values[std::size_t(offset + i)] / double(scale);
                const int q = result.integers[std::size_t(offset + i)];
                // Quantizer division, reciprocal and multiply cannot reach a
                // rounding tie within the documented 32u binary32 envelope.
                if (std::abs(ratio - q) + 32 * unit_f32 * 127 >= 0.5) {
                    throw std::runtime_error("activation does not have a stable Q8 integer");
                }
                integer_sum += q;
            }
            if (integer_sum != 0 && integer_sum != 128) {
                throw std::runtime_error("activation has an invalid Q8 block sum");
            }
        }
    }
    return result;
}

struct Weights {
    ggml_type type;
    std::vector<std::uint8_t> packed;
    std::vector<float> values;
    std::vector<double> offset_bounds;
};

Weights make_weights(ggml_type type) {
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float || width % traits->blck_size != 0) {
        throw std::runtime_error("public quantization traits are unavailable");
    }
    Weights result{type, std::vector<std::uint8_t>(ggml_row_size(type, width) * rows),
        std::vector<float>(std::size_t(width * rows)),
        std::vector<double>(std::size_t(width * rows / q8_block))};
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t k = 0; k < width; ++k) {
            const auto hash = mix(std::uint32_t(row * width + k + 0x12345));
            result.values[std::size_t(row * width + k)] =
                (int(hash % 8191) - 4095) / 4096.0f + float((row % 7) - 3) / 32.0f;
        }
    }
    const std::vector<float> importance(std::size_t(width), 1.0f);
    const auto bytes = ggml_quantize_chunk(type, result.values.data(), result.packed.data(),
        0, rows, width, ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
    if (bytes != result.packed.size()) {
        throw std::runtime_error("public quantizer returned an unexpected byte count");
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        traits->to_float(result.packed.data() + row * ggml_row_size(type, width),
                        result.values.data() + row * width, width);
    }
    for (std::size_t block = 0; block < result.offset_bounds.size(); ++block) {
        std::array<float, q8_block> sorted{};
        std::copy_n(result.values.data() + block * q8_block, q8_block, sorted.data());
        for (float value : sorted) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("weight dequantization returned a non-finite value");
            }
        }
        if (type == GGML_TYPE_IQ4_NL) {
            continue; // Nonlinear code points, with no affine offset.
        }
        std::sort(sorted.begin(), sorted.end());
        const auto distinct = std::unique(sorted.begin(), sorted.end()) - sorted.begin();
        if (distinct < 2 || distinct > 16) {
            throw std::runtime_error("weight block cannot bound its affine terms");
        }
        const double minimum = sorted[0];
        const double maximum = sorted[std::size_t(distinct - 1)];
        const double magnitude = std::max(std::abs(minimum), std::abs(maximum));
        const double error = unit_f32 * magnitude / (1 - unit_f32);
        // L distinct four-bit codes span at least L-1 integer steps. For
        // Q4_K the lowest code is at most 16-L. This bounds the affine offset
        // using only public dequantized values, without reading packed fields.
        const double step = (maximum - minimum + 2 * error) / (distinct - 1);
        result.offset_bounds[block] = type == GGML_TYPE_Q4_0 ? 8 * step :
            std::max(0.0, step * (16 - distinct) - minimum + error);
    }
    return result;
}

struct Reference {
    double native = 0;
    double quantized = 0;
    double arithmetic = 0;
    double activation = 0;
};

std::vector<Reference> make_reference(const Weights & weights, const Activations & inputs,
                                      std::int64_t columns) {
    std::vector<Reference> result(std::size_t(rows * columns));
    // The native shaders form at most K/8 nonzero partials. Local dot/scaling,
    // dequantized-reference rounding and the final reduction add fewer than
    // 32 rounding stages. Integer dot products themselves are exact here.
    const double arithmetic_gamma = gamma(width / 8 + 32, unit_f32);
    const double reference_gamma = gamma(2 * width + 16, unit_f64);
    for (std::int64_t column = 0; column < columns; ++column) {
        for (std::int64_t row = 0; row < rows; ++row) {
            auto & ref = result[std::size_t(column * rows + row)];
            double expression_magnitude = 0;
            for (std::int64_t block = 0; block < width / q8_block; ++block) {
                const auto x_base = column * width + block * q8_block;
                const auto w_base = row * width + block * q8_block;
                const double d = inputs.scales[std::size_t(x_base / q8_block)];
                // Eight binary32 unit roundoffs cover amax/127; one full
                // binary16 ULP covers scale storage with either nearest or
                // directed rounding. The additive term covers subnormals.
                const double d_error = d * (8 * unit_f32 + 0x1p-10 * (1 + 8 * unit_f32)) + 0x1p-24;
                const double offset = weights.offset_bounds[std::size_t(w_base / q8_block)];
                double integer_dot = 0;
                for (std::int64_t i = 0; i < q8_block; ++i) {
                    const double w = weights.values[std::size_t(w_base + i)];
                    const double x = inputs.values[std::size_t(x_base + i)];
                    const double q = inputs.integers[std::size_t(x_base + i)];
                    ref.native += w * x;
                    integer_dot += w * q;
                    const double x_bound = weights.type == GGML_TYPE_IQ4_NL ? std::abs(x) :
                        std::max(std::abs(x), (d + d_error) * std::abs(q));
                    expression_magnitude += (std::abs(w) * (1 + 4 * unit_f32) + 2 * offset) * x_bound;
                }
                ref.quantized += d * integer_dot;
                ref.activation += d_error * std::abs(integer_dot);
            }
            // The Q8 sums are zero or 128. Half(d * 128) is exactly
            // 128 * half(d) for these normal finite scales, under the same
            // conversion mode. Thus the stored affine correction equals the
            // correction in the independently dequantized weight dot product.
            ref.arithmetic = (arithmetic_gamma + reference_gamma) * expression_magnitude;
            if (!std::isfinite(ref.native) || !std::isfinite(ref.quantized) ||
                !(ref.arithmetic > 0) || !std::isfinite(ref.activation)) {
                throw std::runtime_error("invalid independent reference bound");
            }
        }
    }
    return result;
}

struct Check {
    bool native = true;
    bool quantized = true;
    double maximum_error = 0;
    double maximum_native_ratio = 0;
    double maximum_quantized_ratio = 0;
    double maximum_raw_bound = 0;
};

Check check(const std::vector<float> & values, const std::vector<Reference> & reference) {
    if (values.size() != reference.size()) {
        throw std::runtime_error("result size does not match the reference");
    }
    Check result;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error("non-finite native result at " + std::to_string(i));
        }
        const auto & ref = reference[i];
        const double error = std::abs(double(values[i]) - ref.native);
        const double native_ratio = error / ref.arithmetic;
        const double quantized_ratio = std::abs(double(values[i]) - ref.quantized) /
            (ref.activation + ref.arithmetic);
        result.native &= native_ratio <= 1;
        result.quantized &= quantized_ratio <= 1;
        result.maximum_error = std::max(result.maximum_error, error);
        result.maximum_native_ratio = std::max(result.maximum_native_ratio, native_ratio);
        result.maximum_quantized_ratio = std::max(result.maximum_quantized_ratio, quantized_ratio);
        result.maximum_raw_bound = std::max(result.maximum_raw_bound,
            std::abs(ref.quantized - ref.native) + ref.activation + ref.arithmetic);
    }
    return result;
}

struct Result {
    Check check;
    std::vector<float> values;
    double first_compute_us = 0;
    std::vector<double> warm_us;
};

Result compute(ggml_backend_t backend, const Weights & weights, const Activations & inputs,
               std::int64_t columns, int samples, const std::string & id, bool require_native) {
    const auto reference = make_reference(weights, inputs, columns);
    const auto zero = check(std::vector<float>(reference.size(), 0.0f), reference);
    if (zero.native || zero.quantized) {
        throw std::runtime_error("reference bound cannot reject an all-zero result");
    }
    constexpr std::size_t graph_capacity = 8;
    const ggml_init_params params{16 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_capacity, false), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    if (!context) {
        throw std::runtime_error("cannot create the public graph context");
    }
    auto * w = ggml_new_tensor_2d(context.get(), weights.type, width, rows);
    auto * source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, columns);
    auto * x = ggml_scale(context.get(), source, 0.5f);
    auto * output = ggml_mul_mat(context.get(), w, x);
    ggml_set_name(w, (id + "-weights").c_str());
    ggml_set_name(source, (id + "-input").c_str());
    ggml_set_name(x, (id + "-activation-write").c_str());
    ggml_set_name(output, (id + "-output").c_str());
    ggml_set_input(source);
    ggml_set_output(output);
    const auto device = ggml_backend_get_device(backend);
    if (!ggml_backend_dev_supports_op(device, x) || !ggml_backend_dev_supports_op(device, output)) {
        throw std::runtime_error("Vulkan cannot execute the requested public graph");
    }
    auto * graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_n_nodes(graph) != 2) {
        throw std::runtime_error("qualification graph must contain SCALE and MUL_MAT");
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("cannot allocate the device graph tensors");
    }
    std::vector<float> source_values(inputs.values.begin(), inputs.values.begin() + width * columns);
    for (float & value : source_values) {
        value *= 2.0f;
    }
    ggml_backend_tensor_set(w, weights.packed.data(), 0, weights.packed.size());
    ggml_backend_tensor_set(source, source_values.data(), 0, source_values.size() * sizeof(float));
    Result result;
    result.values.resize(reference.size());
    result.warm_us.reserve(std::size_t(samples));
    for (int iteration = 0; iteration <= samples; ++iteration) {
        ggml_backend_synchronize(backend);
        std::cerr << "TQ_MMVQ_BEGIN " << id << ' ' << iteration << std::endl;
        const auto start = std::chrono::steady_clock::now();
        const auto status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto stop = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed, status=" + std::to_string(int(status)));
        }
        ggml_backend_tensor_get(output, result.values.data(), 0, result.values.size() * sizeof(float));
        const auto checked = check(result.values, reference);
        result.check.native &= checked.native;
        result.check.quantized &= checked.quantized;
        result.check.maximum_error = std::max(result.check.maximum_error, checked.maximum_error);
        result.check.maximum_native_ratio = std::max(result.check.maximum_native_ratio, checked.maximum_native_ratio);
        result.check.maximum_quantized_ratio = std::max(result.check.maximum_quantized_ratio, checked.maximum_quantized_ratio);
        result.check.maximum_raw_bound = std::max(result.check.maximum_raw_bound, checked.maximum_raw_bound);
        if ((require_native && !result.check.native) || (!result.check.native && !result.check.quantized)) {
            throw std::runtime_error(id + ": independent bound failed at iteration " +
                std::to_string(iteration) + ", F32 ratio=" + std::to_string(checked.maximum_native_ratio) +
                ", Q8 ratio=" + std::to_string(checked.maximum_quantized_ratio));
        }
        const double elapsed = std::chrono::duration<double, std::micro>(stop - start).count();
        if (iteration == 0) {
            result.first_compute_us = elapsed;
        } else {
            result.warm_us.push_back(elapsed);
        }
        // Completion means status, synchronization and every output passed.
        std::cerr << "TQ_MMVQ_END " << id << ' ' << iteration << " ok" << std::endl;
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double index = fraction * double(values.size() - 1);
    const auto lower = std::size_t(index);
    const auto upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (index - double(lower));
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string device_name = "Vulkan0";
        int samples = 15;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "usage: test-vulkan-mmvq [--device Vulkan0] [--samples 1..101]\n";
                return 0;
            }
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing option value");
            }
            if (arg == "--device") {
                device_name = argv[++i];
            } else if (arg == "--samples") {
                const std::string value = argv[++i];
                std::size_t used = 0;
                samples = std::stoi(value, &used);
                if (used != value.size() || samples < 1 || samples > 101) {
                    throw std::invalid_argument("samples must be in 1..101");
                }
            } else {
                throw std::invalid_argument("unknown option: " + arg);
            }
        }
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(device_name.c_str(), nullptr), ggml_backend_free);
        if (!backend || !ggml_backend_is_vk(backend.get())) {
            throw std::runtime_error("requested device is not an available Vulkan backend");
        }
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(ggml_backend_get_device(backend.get()), &props);
        const bool disabled = std::getenv("GGML_VK_DISABLE_MMVQ") != nullptr;
        const bool forced = std::getenv("GGML_VK_FORCE_MMVQ") != nullptr;
        std::cout << std::boolalpha << std::setprecision(17)
            << "{\"event\":\"config\",\"schema\":1,\"device\":" << quoted(props.name)
            << ",\"description\":" << quoted(props.description) << ",\"device_id\":" << quoted(props.device_id)
            << ",\"backend\":" << quoted(ggml_backend_name(backend.get()))
            << ",\"width\":" << width << ",\"rows\":" << rows << ",\"samples\":" << samples
            << ",\"policy\":\"" << (disabled ? "disable" : forced ? "force" : "auto")
            << "\",\"dispatch\":\"unobserved_by_public_api\",\"env\":{";
        const char * controls[] = {"GGML_VK_DISABLE_MMVQ", "GGML_VK_FORCE_MMVQ",
            "GGML_VK_PERF_LOGGER", "GGML_VK_PERF_LOGGER_CONCURRENT", "GGML_VK_PERF_LOGGER_FREQUENCY",
            "GGML_VK_DISABLE_FUSION", "GGML_VK_SERIALIZE_SUBMISSIONS", "GGML_VK_PIPELINE_STATS",
            "GGML_VK_SYNC_LOGGER", "GGML_VK_MEMORY_LOGGER", "GGML_VK_VISIBLE_DEVICES",
            "GGML_VK_ASYNC_USE_TRANSFER_QUEUE"};
        for (std::size_t i = 0; i < sizeof(controls) / sizeof(controls[0]); ++i) {
            std::cout << (i == 0 ? "" : ",") << quoted(controls[i]) << ':' << quoted(std::getenv(controls[i]));
        }
        std::cout << "}}\n";
        const auto inputs = make_activations();
        for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL}) {
            const auto weights = make_weights(type);
            std::vector<float> first_column;
            for (const auto columns : {1, 2, 8}) {
                const std::string id = std::string(ggml_type_name(type)) + "-b" + std::to_string(columns);
                const auto result = compute(backend.get(), weights, inputs, columns, samples, id,
                    disabled || type == GGML_TYPE_IQ4_NL);
                double shape_difference = 0;
                if (first_column.empty()) {
                    first_column = result.values;
                } else {
                    for (std::int64_t row = 0; row < rows; ++row) {
                        shape_difference = std::max(shape_difference,
                            std::abs(double(result.values[std::size_t(row)]) - first_column[std::size_t(row)]));
                    }
                }
                std::cout << "{\"event\":\"case\",\"id\":" << quoted(id.c_str())
                    << ",\"type\":" << quoted(ggml_type_name(type)) << ",\"columns\":" << columns
                    << ",\"weight_bytes\":" << weights.packed.size()
                    << ",\"checked_outputs\":" << rows * columns * (samples + 1)
                    << ",\"completed_graphs\":" << samples + 1
                    << ",\"f32_valid\":" << result.check.native << ",\"q8_valid\":" << result.check.quantized
                    << ",\"max_abs_error\":" << result.check.maximum_error
                    << ",\"max_f32_bound_ratio\":" << result.check.maximum_native_ratio
                    << ",\"max_q8_bound_ratio\":" << result.check.maximum_quantized_ratio
                    << ",\"max_raw_q8_bound\":" << result.check.maximum_raw_bound
                    << ",\"first_column_shape_difference\":" << shape_difference
                    << ",\"first_compute_us\":" << result.first_compute_us
                    << ",\"warm_median_us\":" << percentile(result.warm_us, 0.5)
                    << ",\"warm_p25_us\":" << percentile(result.warm_us, 0.25)
                    << ",\"warm_p75_us\":" << percentile(result.warm_us, 0.75)
                    << ",\"warm_us\":[";
                for (std::size_t i = 0; i < result.warm_us.size(); ++i) {
                    std::cout << (i == 0 ? "" : ",") << result.warm_us[i];
                }
                std::cout << "]}\n";
            }
        }
        ggml_quantize_free();
        std::cout << "{\"event\":\"complete\",\"status\":\"ok\",\"cases\":9,\"completed_graphs\":"
                  << 9 * (samples + 1) << "}\n";
    } catch (const std::exception & error) {
        ggml_quantize_free();
        std::cerr << "test-vulkan-mmvq: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
