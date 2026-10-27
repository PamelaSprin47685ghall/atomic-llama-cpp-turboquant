#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <array>
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
constexpr int64_t source_rows = 3;
constexpr int64_t destination_rows = 7;
constexpr uint8_t guard = 0xa5;
const std::array<int64_t, source_rows> identity = {0, 1, 2};
const std::array<int64_t, source_rows> scatter = {5, 1, 3};

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// Each source value is exactly representable as both F16 and F32. Using the
// same GPU's identity write as the byte reference isolates source-type and
// row-addressing behavior from differences between quantization algorithms.
std::vector<uint8_t> run(ggml_backend_t backend, ggml_type destination_type,
                         ggml_type source_type, ggml_type index_type,
                         int64_t rows, const std::array<int64_t, source_rows> & indices) {
    constexpr size_t capacity = 16;
    const ggml_init_params params = {
        capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, false), nullptr, true,
    };
    context_ptr context(ggml_init(params), ggml_free);
    require(context != nullptr, "metadata allocation failed");
    auto * destination = ggml_new_tensor_2d(context.get(), destination_type, width, rows);
    auto * source = ggml_new_tensor_2d(context.get(), source_type, width, source_rows);
    auto * index = ggml_new_tensor_1d(context.get(), index_type, source_rows);
    auto * output = ggml_set_rows(context.get(), destination, source, index);
    require(output->op == GGML_OP_SET_ROWS, "unexpected operation");
    require(ggml_backend_supports_op(backend, output), "Vulkan rejects this SET_ROWS type combination");
    ggml_set_input(destination);
    ggml_set_input(source);
    ggml_set_input(index);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(context.get(), capacity, false);
    ggml_build_forward_expand(graph, output);
    require(ggml_graph_n_nodes(graph) == 1 && ggml_graph_node(graph, 0) == output,
            "unexpected SET_ROWS graph");
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "tensor allocation failed");

    std::vector<uint8_t> bytes(ggml_nbytes(destination), guard);
    ggml_backend_tensor_set(destination, bytes.data(), 0, bytes.size());
    std::vector<float> floats(static_cast<size_t>(width * source_rows));
    std::vector<ggml_fp16_t> halves(floats.size());
    for (size_t i = 0; i < floats.size(); ++i) {
        const int value = static_cast<int>((i * 73 + (i / width) * 31) % 251) - 125;
        floats[i] = static_cast<float>(value) / 64.0f;
        halves[i] = ggml_fp32_to_fp16(floats[i]);
        require(ggml_fp16_to_fp32(halves[i]) == floats[i], "source is not exactly half-representable");
    }
    if (source_type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(source, floats.data(), 0, floats.size() * sizeof(float));
    } else {
        require(source_type == GGML_TYPE_F16, "unexpected source type");
        ggml_backend_tensor_set(source, halves.data(), 0, halves.size() * sizeof(ggml_fp16_t));
    }
    for (int64_t row : indices) require(row >= 0 && row < rows, "test row index out of range");
    if (index_type == GGML_TYPE_I64) {
        ggml_backend_tensor_set(index, indices.data(), 0, indices.size() * sizeof(int64_t));
    } else {
        require(index_type == GGML_TYPE_I32, "unexpected index type");
        std::array<int32_t, source_rows> indices32;
        std::transform(indices.begin(), indices.end(), indices32.begin(),
                       [](int64_t value) { return static_cast<int32_t>(value); });
        ggml_backend_tensor_set(index, indices32.data(), 0, indices32.size() * sizeof(int32_t));
    }

    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    require(status == GGML_STATUS_SUCCESS, "direct Vulkan SET_ROWS computation failed");
    ggml_backend_tensor_get(destination, bytes.data(), 0, bytes.size());
    return bytes;
}

bool compare(const std::string & label, const std::vector<uint8_t> & expected,
             const std::vector<uint8_t> & actual, size_t row_bytes) {
    require(expected.size() == actual.size() && !expected.empty(), "comparison size mismatch");
    size_t bad = 0;
    size_t first = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i] && bad++ == 0) first = i;
    }
    std::cout << (bad ? "FAIL " : "PASS ") << label << " compared_bytes=" << actual.size()
              << " mismatches=" << bad << " row_bytes=" << row_bytes;
    if (bad) {
        std::cout << " first_row=" << first / row_bytes << " first_byte=" << first % row_bytes
                  << " expected=" << static_cast<unsigned>(expected[first])
                  << " actual=" << static_cast<unsigned>(actual[first]);
    }
    std::cout << '\n';
    return bad == 0;
}

bool check(ggml_backend_t backend, ggml_type destination_type, ggml_type index_type) {
    const size_t row_bytes = ggml_row_size(destination_type, width);
    require(row_bytes > 0, "empty quantized row");
    const auto packed = run(backend, destination_type, GGML_TYPE_F32, index_type, source_rows, identity);
    require(packed.size() == row_bytes * source_rows, "identity output size mismatch");
    for (size_t row = 0; row < source_rows; ++row) {
        const auto first = packed.begin() + static_cast<ptrdiff_t>(row * row_bytes);
        const auto last = first + static_cast<ptrdiff_t>(row_bytes);
        require(std::any_of(first, last, [](uint8_t value) { return value != guard; }),
                "identity SET_ROWS did not write an output row");
        for (size_t earlier = 0; earlier < row; ++earlier) {
            require(!std::equal(first, last, packed.begin() + static_cast<ptrdiff_t>(earlier * row_bytes)),
                    "identity rows must differ to detect a wrong row permutation");
        }
    }
    std::vector<uint8_t> expected(row_bytes * destination_rows, guard);
    for (size_t row = 0; row < source_rows; ++row) {
        std::copy_n(packed.begin() + static_cast<ptrdiff_t>(row * row_bytes), row_bytes,
                    expected.begin() + static_cast<ptrdiff_t>(scatter[row] * row_bytes));
    }
    const std::string label = std::string(ggml_type_name(destination_type)) + "/" + ggml_type_name(index_type);
    bool pass = true;
    for (auto source_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        const auto actual = run(backend, destination_type, source_type, index_type, destination_rows, scatter);
        pass &= compare(label + "/" + ggml_type_name(source_type), expected, actual, row_bytes);
    }
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        require(argc <= 2, "usage: test-vulkan-set-rows-types [device-index]");
        size_t device = 0;
        if (argc == 2) {
            const std::string value = argv[1];
            size_t end = 0;
            const unsigned long parsed = std::stoul(value, &end);
            require(!value.empty() && value[0] >= '0' && value[0] <= '9' && end == value.size(),
                    "invalid device index");
            require(parsed <= std::numeric_limits<size_t>::max(), "device index out of range");
            device = static_cast<size_t>(parsed);
        }
        backend_ptr backend(ggml_backend_vk_init(device), ggml_backend_free);
        require(backend != nullptr && ggml_backend_is_vk(backend.get()), "Vulkan initialization failed");
        std::cout << "Vulkan SET_ROWS backend=" << ggml_backend_name(backend.get()) << " device=" << device
                  << " width=" << width << " source_rows=" << source_rows
                  << " scattered_rows=" << source_rows << " preserved_rows=" << destination_rows - source_rows << '\n';
        size_t cases = 0;
        size_t failed = 0;
        for (auto destination_type : {GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0}) {
            for (auto index_type : {GGML_TYPE_I32, GGML_TYPE_I64}) {
                ++cases;
                failed += !check(backend.get(), destination_type, index_type);
            }
        }
        std::cout << "Vulkan SET_ROWS cases=" << cases << " compared_type_pairs=" << cases * 2
                  << " failed=" << failed << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "FAIL Vulkan SET_ROWS: " << error.what() << '\n';
        return 1;
    }
}
