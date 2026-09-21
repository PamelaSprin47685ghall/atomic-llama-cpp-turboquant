#pragma once

#include "../include/llama-predefined-hidden.h"
#include "ggml-cpp.h"
#include "ggml-device-copy.h"

#include <array>
#include <cstdint>
#include <algorithm>
#include <limits>

// All references point to persistent storage, never a temporary graph tensor.
// Source dependencies are completed by the existing context synchronization
// before crossing context owners. Copies within an owner share its queue.
struct llama_predefined_hidden_store {
    ggml_backend_t executor = nullptr;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    // RESULT + two single-row states. INPUT is a logical destination range,
    // not another maximum-sized matrix.
    std::array<ggml_tensor *, 3> tensors{};
    struct bound_range {
        const ggml_tensor * tensor = nullptr;
        size_t src_offset = 0;
        uint32_t dst_row = 0;
        uint32_t rows = 0;
    };
    std::array<bound_range, GGML_DEVICE_COPY_MAX_RANGES> input{};
    size_t input_count = 0;
    uint32_t width = 0;
    uint32_t capacity = 0;
    uint32_t result_capacity = 0;
    uint64_t generation = 0;
    // Witnessed completion generation: tracks the latest generation of this store
    // that has been proven complete via a host-level synchronize().
    // Reset to 0 whenever a new decode/capture begins or upon reset.
    uint64_t synchronized_generation = 0;
    uint32_t captured_rows = 0;
    uint32_t valid_rows = 0;
    uint32_t input_rows = 0;
    bool bound_input = false;
    bool pending = false;
    bool host_current = false;
};

// Borrowed for one synchronous set_inputs() call. Only capacity backing is
// referenced; changing the active range never changes tensor ne/nb.
struct llama_device_hidden_input {
    ggml_backend_t executor = nullptr;
    std::array<ggml_device_copy_range, GGML_DEVICE_COPY_MAX_RANGES> ranges{};
    size_t count = 0;
    uint32_t rows = 0;
};

inline bool llama_predefined_hidden_generation_matches(
        const llama_predefined_hidden_store & source, unsigned slot, uint64_t expected) {
    if (slot >= source.tensors.size()) return false;
    return slot == LLAMA_PREDEFINED_H_RESULT
        ? source.valid_rows != 0 && expected != 0 && expected == source.generation
        : expected == 0;
}

// Source-row identity is explicit. A slice never treats maximum capacity as
// useful work, guesses an offset from token count, or rewrites tensor ne/nb.
inline bool llama_predefined_hidden_slice(
        const llama_predefined_hidden_store::bound_range * ranges, size_t count,
        uint32_t input_rows, uint32_t width, const int32_t * source_rows, uint32_t rows,
        ggml_backend_t executor, llama_device_hidden_input & out) {
    if (!ranges || count == 0 || count > GGML_DEVICE_COPY_MAX_RANGES || !source_rows ||
        rows == 0 || width == 0 || source_rows[0] < 0) return false;
    const uint32_t first = uint32_t(source_rows[0]);
    if (first > input_rows || rows > input_rows - first ||
        uint64_t(width) * sizeof(float) > SIZE_MAX) return false;
    for (uint32_t i = 0; i < rows; ++i) {
        if (source_rows[i] < 0 || uint32_t(source_rows[i]) != first + i) return false;
    }
    const uint32_t end = first + rows;
    const size_t row_bytes = size_t(width) * sizeof(float);
    llama_device_hidden_input next{};
    next.executor = executor;
    next.rows = rows;
    uint32_t covered = 0;
    uint32_t plan_end = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & r = ranges[i];
        if (!r.tensor || !r.rows || r.dst_row != plan_end || r.dst_row > input_rows ||
            r.rows > input_rows - r.dst_row || size_t(r.rows) > (SIZE_MAX - r.src_offset) / row_bytes) return false;
        plan_end += r.rows;
        const uint32_t begin = std::max(first, r.dst_row);
        const uint32_t stop = std::min(end, plan_end);
        if (begin >= stop) continue;
        if (begin - first != covered || next.count >= next.ranges.size()) return false;
        next.ranges[next.count++] = {r.tensor, nullptr,
            r.src_offset + size_t(begin - r.dst_row) * row_bytes,
            size_t(begin - first) * row_bytes, size_t(stop - begin) * row_bytes};
        covered += stop - begin;
    }
    if (covered != rows || plan_end != input_rows) return false;
    out = next;
    return true;
}
