#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#if defined(TQ_QUANTIZED_TEST_VULKAN)
#include <ggml-vulkan.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <numeric>
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

struct geometry {
    int width;
    int rows;
    int columns;
    int weight_heads;
    int query_heads;
};

struct fixture {
    ggml_type type;
    geometry shape;
    std::vector<uint8_t> weights;
    values activations;

    fixture(ggml_type selected_type, geometry selected_shape)
        : type(selected_type), shape(selected_shape),
          weights(ggml_row_size(type, shape.width) * shape.rows * shape.weight_heads),
          activations(size_t(shape.width) * shape.columns * shape.query_heads) {
        require(shape.width % ggml_blck_size(type) == 0, "invalid quantized row width");
        require(!ggml_quantize_requires_imatrix(type), "fixture requires an importance matrix");
        values original(size_t(shape.width) * shape.rows * shape.weight_heads);
        uint32_t state = 0x537a9de1;
        const auto next = [&state]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return float(int32_t(state & 0xffffu) - 32768) / 10009.0f;
        };
        for (float & value : original) { value = next(); }
        for (float & value : activations) { value = next(); }
        const auto written = ggml_quantize_chunk(type, original.data(), weights.data(), 0,
            int64_t(shape.rows) * shape.weight_heads, shape.width, nullptr);
        require(written == weights.size(), "quantized fixture byte count mismatch");
    }
};

float bias_value(int index, int row, int column, int head) {
    return float((row + 3 * column + 5 * head + 7 * index) % 23 - 11) / float(13 + 2 * index);
}

values compute(ggml_backend_t backend, const fixture & data, const std::vector<int> & order,
               ggml_prec precision, int bias_count = 0) {
    const auto & g = data.shape;
    const int columns = int(order.size());
    require(columns > 0, "empty query order");
    const ggml_init_params params{
        12 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    require(context != nullptr, "could not create graph context");
    auto * weights = ggml_new_tensor_3d(context.get(), data.type, g.width, g.rows, g.weight_heads);
    auto * activations = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, g.width, columns, g.query_heads);
    values reordered(size_t(g.width) * columns * g.query_heads);
    for (int head = 0; head < g.query_heads; ++head) {
        for (int column = 0; column < columns; ++column) {
            const int original_column = order[size_t(column)];
            require(original_column >= 0 && original_column < g.columns, "invalid query order");
            std::copy_n(data.activations.data() + (size_t(head) * g.columns + original_column) * g.width,
                        g.width, reordered.data() + (size_t(head) * columns + column) * g.width);
        }
    }
    auto * output = ggml_mul_mat(context.get(), weights, activations);
    // The invariant mode preserves the backend's native one-column profile;
    // it does not promise unquantized F32 activation arithmetic.
    ggml_mul_mat_set_prec(output, precision);
    require(ggml_backend_supports_op(backend, output), "backend rejected quantized matrix product");
    std::vector<ggml_tensor *> biases;
    std::vector<values> bias_data;
    for (int index = 0; index < bias_count; ++index) {
        auto * bias = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, g.rows, columns, g.query_heads);
        values packed(size_t(g.rows) * columns * g.query_heads);
        for (int head = 0; head < g.query_heads; ++head) {
            for (int column = 0; column < columns; ++column) {
                for (int row = 0; row < g.rows; ++row) {
                    packed[(size_t(head) * columns + column) * g.rows + row] =
                        bias_value(index, row, order[size_t(column)], head);
                }
            }
        }
        biases.push_back(bias);
        bias_data.push_back(std::move(packed));
        output = index == 0 ? ggml_add(context.get(), output, bias) : ggml_add(context.get(), bias, output);
    }
    auto * graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "could not allocate graph tensors");
    ggml_backend_tensor_set(weights, data.weights.data(), 0, data.weights.size());
    ggml_backend_tensor_set(activations, reordered.data(), 0, reordered.size() * sizeof(float));
    for (size_t index = 0; index < biases.size(); ++index) {
        ggml_backend_tensor_set(biases[index], bias_data[index].data(), 0, bias_data[index].size() * sizeof(float));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "quantized matrix graph failed");
    values result(size_t(g.rows) * columns * g.query_heads);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    for (float value : result) { require(std::isfinite(value), "non-finite quantized matrix result"); }
    return result;
}

void exact(const values & actual, const values & expected, const std::string & label) {
    require(actual.size() == expected.size(), "comparison shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            throw std::runtime_error(label + " is not bitwise invariant at " + std::to_string(i) +
                ": actual=" + std::to_string(actual[i]) + " reference=" + std::to_string(expected[i]));
        }
    }
}

values expected_for(const fixture & data, const values & reference, const std::vector<int> & order,
                    int bias_count) {
    const auto & g = data.shape;
    const int columns = int(order.size());
    values expected(size_t(g.rows) * columns * g.query_heads);
    for (int head = 0; head < g.query_heads; ++head) {
        for (int column = 0; column < columns; ++column) {
            const int original_column = order[size_t(column)];
            for (int row = 0; row < g.rows; ++row) {
                float value = reference[(size_t(head) * g.columns + original_column) * g.rows + row];
                for (int index = 0; index < bias_count; ++index) {
                    value += bias_value(index, row, original_column, head);
                }
                expected[(size_t(head) * columns + column) * g.rows + row] = value;
            }
        }
    }
    return expected;
}

const char * mode_name(ggml_prec precision) {
    return precision == GGML_PREC_DEFAULT ? "native_default" : "batch_invariant_native_quantized";
}

std::string nullable_json(const char * value) {
    return value ? json_string(value) : "null";
}

void check_performance_environment() {
    // These controls add diagnostic work or alter submission timing. Build-time
    // DEBUG/CHECK_RESULTS flags are recorded separately with the build receipt.
    for (const char * name : {"GGML_VK_PERF_LOGGER", "GGML_VK_PERF_LOGGER_CONCURRENT",
            "GGML_VK_PIPELINE_STATS", "GGML_VK_SYNC_LOGGER", "GGML_VK_MEMORY_LOGGER"}) {
        require(std::getenv(name) == nullptr,
                std::string("performance mode requires the diagnostic control unset: ") + name);
    }
}

struct performance_inputs {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{
        nullptr, ggml_backend_buffer_free};
    ggml_tensor * weights = nullptr;
    ggml_tensor * source = nullptr;

    performance_inputs(ggml_backend_t backend, const fixture & data, int columns) {
        const auto & g = data.shape;
        require(g.weight_heads == 1 && g.query_heads == 1 && columns <= g.columns,
                "invalid resident performance shape");
        const ggml_init_params params{2 * ggml_tensor_overhead(), nullptr, true};
        context.reset(ggml_init(params));
        require(context != nullptr, "could not create resident input context");
        weights = ggml_new_tensor_2d(context.get(), data.type, g.width, g.rows);
        source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, g.width, columns);
        ggml_set_name(weights, "quantized-performance-weights");
        ggml_set_name(source, "quantized-performance-source");
        ggml_set_input(source);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(buffer != nullptr, "could not allocate resident performance inputs");
        values uploaded(data.activations.begin(), data.activations.begin() + size_t(g.width) * columns);
        for (float & value : uploaded) { value *= 2.0f; }
        ggml_backend_tensor_set(weights, data.weights.data(), 0, data.weights.size());
        ggml_backend_tensor_set(source, uploaded.data(), 0, uploaded.size() * sizeof(float));
    }
};

struct performance_graph {
    ggml_backend_t backend;
    ggml_prec precision;
    std::string id;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{
        nullptr, ggml_backend_buffer_free};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;
    values cold_output;
    values observed;
    double cold_us = 0.0;
    std::vector<double> warm_us;

    performance_graph(ggml_backend_t selected_backend, const performance_inputs & inputs,
                      ggml_prec selected_precision, const std::string & case_id)
        : backend(selected_backend), precision(selected_precision), id(case_id) {
        constexpr size_t capacity = 8;
        const ggml_init_params params{
            8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, false), nullptr, true};
        context.reset(ggml_init(params));
        require(context != nullptr, "could not create resident performance graph");
        auto * activation = ggml_scale(context.get(), inputs.source, 0.5f);
        output = ggml_mul_mat(context.get(), inputs.weights, activation);
        ggml_mul_mat_set_prec(output, precision);
        ggml_set_name(activation, (id + "-" + mode_name(precision) + "-activation-write").c_str());
        ggml_set_name(output, (id + "-" + mode_name(precision) + "-output").c_str());
        ggml_set_output(output);
        require(ggml_backend_supports_op(backend, activation) &&
                ggml_backend_supports_op(backend, output), "backend rejected performance graph");
        graph = ggml_new_graph_custom(context.get(), capacity, false);
        ggml_build_forward_expand(graph, output);
        require(ggml_graph_n_nodes(graph) == 2, "performance graph must be SCALE then MUL_MAT");
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(buffer != nullptr, "could not allocate resident performance outputs");
        observed.resize(size_t(ggml_nelements(output)));
        warm_us.reserve(15);
    }

    void run() {
        // Run the same graph consecutively so another graph cannot evict its
        // native recording cache between warm samples. Both modes share the
        // exact same resident weight and source buffers.
        for (int iteration = 0; iteration <= 15; ++iteration) {
            ggml_backend_synchronize(backend);
            std::cerr << "TQ_QUANT_PERF_BEGIN " << id << ' ' << mode_name(precision) << ' '
                      << iteration << std::endl;
            const auto start = std::chrono::steady_clock::now();
            const auto status = ggml_backend_graph_compute(backend, graph);
            ggml_backend_synchronize(backend);
            const auto stop = std::chrono::steady_clock::now();
            require(status == GGML_STATUS_SUCCESS, "resident performance graph failed");
            const double elapsed = std::chrono::duration<double, std::micro>(stop - start).count();
            require(std::isfinite(elapsed) && elapsed > 0.0, "invalid graph timing sample");
            ggml_backend_tensor_get(output, observed.data(), 0, observed.size() * sizeof(float));
            for (float value : observed) {
                require(std::isfinite(value), "non-finite resident performance output");
            }
            if (iteration == 0) {
                require(std::any_of(observed.begin(), observed.end(),
                                   [](float value) { return std::abs(value) > 1.0f; }),
                        "degenerate resident performance output");
                cold_output = observed;
                cold_us = elapsed;
            } else {
                // This is a within-mode stability gate, not a numerical oracle
                // for the default F16 matrix path or a serial-equivalence gate.
                exact(observed, cold_output, id + " " + mode_name(precision) + " warm output");
                warm_us.push_back(elapsed);
            }
            std::cerr << "TQ_QUANT_PERF_END " << id << ' ' << mode_name(precision) << ' '
                      << iteration << " ok" << std::endl;
        }
    }
};

double timing_percentile(std::vector<double> values, double fraction) {
    require(!values.empty(), "empty timing distribution");
    std::sort(values.begin(), values.end());
    const double index = fraction * double(values.size() - 1);
    const size_t lower = size_t(index);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (index - double(lower));
}

void report_performance_mode(const performance_graph & result, const fixture & data,
                             int columns, int order) {
    std::cout << "{\"event\":\"performance_mode\",\"id\":" << json_string(result.id)
              << ",\"type\":" << json_string(ggml_type_name(data.type))
              << ",\"width\":" << data.shape.width << ",\"rows\":" << data.shape.rows
              << ",\"columns\":" << columns << ",\"mode\":" << json_string(mode_name(result.precision))
              << ",\"order\":" << order << ",\"weight_bytes\":" << data.weights.size()
              << ",\"completed_graphs\":16,\"finite_nonzero_stable\":true"
              << ",\"cold_graph_us\":" << result.cold_us
              << ",\"warm_median_us\":" << timing_percentile(result.warm_us, 0.5)
              << ",\"warm_p25_us\":" << timing_percentile(result.warm_us, 0.25)
              << ",\"warm_p75_us\":" << timing_percentile(result.warm_us, 0.75)
              << ",\"warm_us\":[";
    for (size_t i = 0; i < result.warm_us.size(); ++i) {
        std::cout << (i ? "," : "") << result.warm_us[i];
    }
    std::cout << "]}\n";
}

void run_performance(ggml_backend_t backend, ggml_prec first_mode) {
#if defined(TQ_QUANTIZED_TEST_VULKAN)
    require(ggml_backend_is_vk(backend), "performance comparison requires a native Vulkan device");
#else
    require(false, "performance comparison requires a Vulkan-enabled test installation");
#endif
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    std::cout << std::boolalpha << std::setprecision(17)
              << "{\"event\":\"performance_config\",\"schema\":1"
              << ",\"scope\":\"resident_batched_SCALE_MUL_MAT\""
              << ",\"device\":" << nullable_json(props.name)
              << ",\"description\":" << nullable_json(props.description)
              << ",\"device_id\":" << nullable_json(props.device_id)
              << ",\"backend\":" << nullable_json(ggml_backend_name(backend))
              << ",\"first_mode\":" << json_string(mode_name(first_mode))
              << ",\"width\":2048,\"rows\":1025,\"columns\":[9,32,128]"
              << ",\"cold_graphs_per_mode\":1,\"warm_graphs_per_mode\":15"
              << ",\"shared_resident_inputs\":true,\"mode_order_alternates_by_case\":true"
              << ",\"serial_equivalence\":\"not_checked_by_performance_mode\""
              << ",\"actual_pipeline\":\"unobserved_by_public_api\",\"env\":{";
    const char * controls[] = {"GGML_VK_DISABLE_MMVQ", "GGML_VK_FORCE_MMVQ", "GGML_VK_DISABLE_F16",
        "GGML_VK_PERF_LOGGER", "GGML_VK_PERF_LOGGER_CONCURRENT", "GGML_VK_PERF_LOGGER_FREQUENCY",
        "GGML_VK_DISABLE_FUSION", "GGML_VK_SERIALIZE_SUBMISSIONS", "GGML_VK_PIPELINE_STATS",
        "GGML_VK_SYNC_LOGGER", "GGML_VK_MEMORY_LOGGER", "GGML_VK_VISIBLE_DEVICES",
        "GGML_VK_ASYNC_USE_TRANSFER_QUEUE"};
    for (size_t i = 0; i < sizeof(controls) / sizeof(controls[0]); ++i) {
        std::cout << (i ? "," : "") << json_string(controls[i]) << ':' << nullable_json(std::getenv(controls[i]));
    }
    std::cout << "}}\n";
    size_t cases = 0;
    for (const auto type : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
        const fixture data(type, {2048, 1025, 128, 1, 1});
        for (int columns : {9, 32, 128}) {
            const std::string id = std::string(ggml_type_name(type)) + "-k2048-m1025-n" + std::to_string(columns);
            const performance_inputs inputs(backend, data, columns);
            performance_graph native(backend, inputs, GGML_PREC_DEFAULT, id);
            performance_graph invariant(backend, inputs, GGML_PREC_BATCH_INVARIANT, id);
            const bool invariant_first = (first_mode == GGML_PREC_BATCH_INVARIANT) != ((cases % 2) != 0);
            performance_graph * order[] = {invariant_first ? &invariant : &native,
                                           invariant_first ? &native : &invariant};
            for (int index = 0; index < 2; ++index) {
                order[index]->run();
                report_performance_mode(*order[index], data, columns, index);
            }
            double difference = 0.0;
            for (size_t i = 0; i < native.cold_output.size(); ++i) {
                difference = std::max(difference,
                    std::abs(double(native.cold_output[i]) - double(invariant.cold_output[i])));
            }
            std::cout << "{\"event\":\"performance_comparison\",\"id\":" << json_string(id)
                      << ",\"batch_invariant_over_native_default\":"
                      << timing_percentile(invariant.warm_us, 0.5) / timing_percentile(native.warm_us, 0.5)
                      << ",\"max_abs_mode_difference\":" << difference
                      << ",\"mode_difference_is_diagnostic\":true}\n";
            ++cases;
        }
    }
    std::cout << "{\"event\":\"performance_complete\",\"status\":\"ok\",\"cases\":" << cases
              << ",\"mode_runs\":" << cases * 2 << ",\"completed_graphs\":" << cases * 32 << "}\n";
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const char * requested = "Vulkan0";
        ggml_prec precision = GGML_PREC_BATCH_INVARIANT;
        ggml_prec performance_first = GGML_PREC_DEFAULT;
        bool selected_device = false;
        bool performance = false;
        bool selected_performance_first = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--native-matmul") {
                precision = GGML_PREC_DEFAULT;
            } else if (arg == "--performance") {
                performance = true;
            } else if (arg == "--performance-first") {
                require(i + 1 < argc, "--performance-first requires default or batch-invariant");
                const std::string value = argv[++i];
                require(value == "default" || value == "batch-invariant",
                        "--performance-first requires default or batch-invariant");
                performance_first = value == "default" ? GGML_PREC_DEFAULT : GGML_PREC_BATCH_INVARIANT;
                selected_performance_first = true;
            } else if (arg == "--help") {
                std::cout << "usage: test-vulkan-quantized-shape-invariance [device] [--native-matmul]\n"
                          << "       test-vulkan-quantized-shape-invariance [device] --performance "
                             "[--performance-first default|batch-invariant]\n";
                return 0;
            } else {
                require(!selected_device && argv[i][0] != '-',
                        "unknown argument; use --help for supported options");
                requested = argv[i];
                selected_device = true;
            }
        }
        require(!selected_performance_first || performance, "--performance-first requires --performance");
        require(!performance || precision == GGML_PREC_BATCH_INVARIANT,
                "--performance compares both modes and cannot be combined with --native-matmul");
        if (performance) { check_performance_environment(); }
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(requested, nullptr), ggml_backend_free);
        require(backend != nullptr, std::string("backend is unavailable: ") + requested);
        if (performance) {
            run_performance(backend.get(), performance_first);
            ggml_quantize_free();
            return 0;
        }
        const char * device = ggml_backend_dev_name(ggml_backend_get_device(backend.get()));
        const ggml_type types[] = {GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K};
        const geometry shapes[] = {
            {2048, 33, 32, 1, 1},
            {768, 65, 9, 2, 4},
            {2048, 32, 128, 1, 1},
        };
        const int cohort_sizes[] = {2, 8, 9, 32, 128};
        size_t graphs = 0;
        size_t comparisons = 0;
        size_t bias_graphs = 0;
        for (const auto type : types) {
            for (const auto & g : shapes) {
                const fixture data(type, g);
                values reference(size_t(g.rows) * g.columns * g.query_heads);
                for (int column = 0; column < g.columns; ++column) {
                    // The reference is the unchanged native one-column path,
                    // independent of the candidate's requested batch policy.
                    const auto one = compute(backend.get(), data, {column}, GGML_PREC_DEFAULT);
                    ++graphs;
                    for (int head = 0; head < g.query_heads; ++head) {
                        std::copy_n(one.data() + size_t(head) * g.rows, g.rows,
                            reference.data() + (size_t(head) * g.columns + column) * g.rows);
                    }
                }
                require(std::any_of(reference.begin(), reference.end(),
                                    [](float value) { return std::abs(value) > 1.0f; }),
                        "degenerate quantized reference");
                const std::string label = std::string(ggml_type_name(type)) +
                    " K=" + std::to_string(g.width) + " M=" + std::to_string(g.rows) +
                    " weight_heads=" + std::to_string(g.weight_heads) +
                    " query_heads=" + std::to_string(g.query_heads);
                for (const int count : cohort_sizes) {
                    if (count > g.columns) { continue; }
                    for (const bool reverse : {false, true}) {
                        std::vector<int> order(static_cast<size_t>(count));
                        std::iota(order.begin(), order.end(), 0);
                        if (reverse) { std::reverse(order.begin(), order.end()); }
                        const auto actual = compute(backend.get(), data, order, precision);
                        ++graphs;
                        exact(actual, expected_for(data, reference, order, 0),
                            label + " N=" + std::to_string(count) + (reverse ? " reversed" : " ordered"));
                        ++comparisons;
                    }
                }
                if (g.weight_heads == 1 && g.columns == 32) {
                    // The single-query graph is eligible for one/two ADD
                    // fusion. The batched graph must perform the same adds.
                    for (int bias_count = 1; bias_count <= 2; ++bias_count) {
                        for (int count : {1, 2, 32}) {
                            std::vector<int> order(static_cast<size_t>(count));
                            std::iota(order.begin(), order.end(), 0);
                            std::reverse(order.begin(), order.end());
                            const auto actual = compute(backend.get(), data, order, precision, bias_count);
                            ++graphs;
                            ++bias_graphs;
                            exact(actual, expected_for(data, reference, order, bias_count),
                                label + " N=" + std::to_string(count) +
                                " biases=" + std::to_string(bias_count));
                            ++comparisons;
                        }
                    }
                }
            }
        }
        ggml_quantize_free();
        std::cout << "{\"status\":\"pass\",\"device\":" << json_string(device)
                  << ",\"matmul_mode\":" << json_string(precision == GGML_PREC_BATCH_INVARIANT ?
                      "batch_invariant_native_quantized" : "native_default")
                  << ",\"reference\":\"native_default_single_column\",\"types\":[\"iq4_nl\",\"q8_0\",\"q4_K\"]"
                  << ",\"contiguous_operands\":true,\"head_broadcast\":true"
                  << ",\"cohort_sizes\":[2,8,9,32,128],\"graphs\":" << graphs
                  << ",\"bitwise_comparisons\":" << comparisons
                  << ",\"bias_graphs\":" << bias_graphs
                  << ",\"force_mmvq\":" << (std::getenv("GGML_VK_FORCE_MMVQ") ? "true" : "false")
                  << ",\"disable_mmvq\":" << (std::getenv("GGML_VK_DISABLE_MMVQ") ? "true" : "false")
                  << "}\n";
        return 0;
    } catch (const std::exception & error) {
        ggml_quantize_free();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
