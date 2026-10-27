#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "../../../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr size_t guard_bytes = 32;
constexpr uint8_t guard_value = 0xa5;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct split_config {
    ggml_backend_meta_split_axis axis;
    const char * name;
    int64_t ne[4];
};

ggml_backend_meta_split_state split_tensor(const ggml_tensor *, void * user_data) {
    const auto & config = *static_cast<const split_config *>(user_data);
    ggml_backend_meta_split_state result = {config.axis, {0}, {1}, 1};
    if (config.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        result.ne[0] = 3;
        result.ne[1] = 7;
    }
    return result;
}

struct tensor_fixture {
    ggml_context * context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor * tensor = nullptr;

    tensor_fixture() = default;
    tensor_fixture(const tensor_fixture &) = delete;
    tensor_fixture & operator=(const tensor_fixture &) = delete;

    ~tensor_fixture() {
        if (backend != nullptr) {
            ggml_backend_synchronize(backend);
        }
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (context != nullptr) {
            ggml_free(context);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    void init(ggml_backend_dev_t cpu, split_config & config, ggml_type type) {
        ggml_backend_dev_t children[] = {cpu, cpu};
        const auto device = ggml_backend_meta_device(children, 2, split_tensor, &config);
        require(device != nullptr, "cannot create the meta device");
        backend = ggml_backend_dev_init(device, nullptr);
        require(backend != nullptr, "cannot create the meta backend");
        context = ggml_init({16 * 1024, nullptr, true});
        require(context != nullptr, "cannot create the tensor context");
        tensor = ggml_new_tensor_4d(context, type, config.ne[0], config.ne[1], config.ne[2], config.ne[3]);
        require(tensor != nullptr, "cannot create the tensor");
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(context, ggml_backend_dev_buffer_type(device));
        require(buffer != nullptr, "cannot allocate the split tensor");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
};

std::vector<uint8_t> make_values(const ggml_tensor * tensor, int seed) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    const size_t count = static_cast<size_t>(ggml_nelements(tensor));
    for (size_t i = 0; i < count; ++i) {
        const float value = float(int((i * 17) % 127) - 63 + seed) * 0.125f;
        if (tensor->type == GGML_TYPE_F32) {
            std::memcpy(result.data() + i * sizeof(value), &value, sizeof(value));
        } else {
            const ggml_fp16_t half = ggml_fp32_to_fp16(value);
            std::memcpy(result.data() + i * sizeof(half), &half, sizeof(half));
        }
    }
    return result;
}

std::vector<uint8_t> guarded_buffer(size_t bytes) {
    if (bytes > size_t(std::numeric_limits<std::ptrdiff_t>::max()) - 2 * guard_bytes) {
        throw std::length_error("the transfer is too large for a guarded buffer");
    }
    return std::vector<uint8_t>(bytes + 2 * guard_bytes, guard_value);
}

void check_range(const std::vector<uint8_t> & data, size_t offset, size_t bytes) {
    require(offset <= data.size() && bytes <= data.size() - offset, "the test transfer is outside the tensor");
}

void check_guards(const std::vector<uint8_t> & data) {
    for (size_t i = 0; i < guard_bytes; ++i) {
        require(data[i] == guard_value, "the transfer changed the leading guard");
        require(data[data.size() - 1 - i] == guard_value, "the transfer changed the trailing guard");
    }
}

void check_full_tensor(tensor_fixture & fixture, const std::vector<uint8_t> & expected) {
    auto actual = guarded_buffer(expected.size());
    ggml_backend_tensor_get(fixture.tensor, actual.data() + guard_bytes, 0, expected.size());
    check_guards(actual);
    require(std::memcmp(actual.data() + guard_bytes, expected.data(), expected.size()) == 0,
            "the tensor contents differ from the expected flat array");
}

void run_get(tensor_fixture & fixture, size_t offset, size_t bytes) {
    const auto expected = make_values(fixture.tensor, 0);
    check_range(expected, offset, bytes);
    ggml_backend_tensor_set(fixture.tensor, expected.data(), 0, expected.size());
    auto actual = guarded_buffer(bytes);
    ggml_backend_tensor_get_async(fixture.backend, fixture.tensor, actual.data() + guard_bytes, offset, bytes);
    ggml_backend_synchronize(fixture.backend);
    check_guards(actual);
    require(std::memcmp(actual.data() + guard_bytes, expected.data() + offset, bytes) == 0,
            "async get returned the wrong logical range");
    check_full_tensor(fixture, expected);
}

void run_set(tensor_fixture & fixture, size_t offset, size_t bytes) {
    auto expected = make_values(fixture.tensor, 0);
    check_range(expected, offset, bytes);
    ggml_backend_tensor_set(fixture.tensor, expected.data(), 0, expected.size());
    const auto replacement = make_values(fixture.tensor, 100);
    check_range(replacement, offset, bytes);
    auto source = guarded_buffer(bytes);
    std::memcpy(source.data() + guard_bytes, replacement.data() + offset, bytes);
    const auto source_before = source;
    std::memcpy(expected.data() + offset, replacement.data() + offset, bytes);
    ggml_backend_tensor_set_async(fixture.backend, fixture.tensor, source.data() + guard_bytes, offset, bytes);
    ggml_backend_synchronize(fixture.backend);
    require(source == source_before, "async set changed the caller's source bytes");
    check_full_tensor(fixture, expected);
}

ggml_backend_meta_split_state split_copy_fixture(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state split = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    if (std::strcmp(tensor->name, "copy_weights") == 0) {
        split.axis = GGML_BACKEND_SPLIT_AXIS_1;
        split.ne[0] = split.ne[1] = 2;
    }
    return split;
}

void run_host_copy_consumer(ggml_backend_dev_t cpu) {
    tensor_fixture fixture;
    ggml_backend_dev_t children[] = {cpu, cpu};
    const auto device = ggml_backend_meta_device(children, 2, split_copy_fixture, nullptr);
    require(device != nullptr, "cannot create copy fixture device");
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    require(fixture.backend != nullptr, "cannot create copy fixture backend");
    fixture.context = ggml_init({128 * 1024, nullptr, true});
    require(fixture.context != nullptr, "cannot create copy fixture context");
    fixture.tensor = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 4);
    auto * weights = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 4);
    ggml_set_name(weights, "copy_weights");
    fixture.buffer = ggml_backend_alloc_ctx_tensors_from_buft(fixture.context, ggml_backend_dev_buffer_type(device));
    require(fixture.buffer != nullptr, "cannot allocate copy fixture inputs");
    ggml_backend_buffer_set_usage(fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * graph = ggml_new_graph(fixture.context);
    auto * result = ggml_silu(fixture.context, ggml_mul_mat(fixture.context, weights, fixture.tensor));
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    fixture.allocator = ggml_gallocr_new(ggml_backend_dev_buffer_type(device));
    require(fixture.allocator != nullptr && ggml_gallocr_alloc_graph(fixture.allocator, graph),
            "cannot allocate copy consumer graph");
    float w[16];
    for (int row = 0; row < 4; ++row) {
        for (int k = 0; k < 4; ++k) {
            w[row * 4 + k] = float((row + 1) * (k + 2)) / 16.0f;
        }
    }
    ggml_backend_tensor_set(weights, w, 0, sizeof(w));
    const auto copy = fixture.backend->iface.cpy_tensor_async;
    require(copy != nullptr, "Meta host copy callback is unavailable");
    std::vector<uint8_t> expected;
    auto check_consumer = [&]() {
        require(ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS,
                "host copy consumer execution failed");
        float actual[16];
        ggml_backend_tensor_get(result, actual, 0, sizeof(actual));
        // Output rows are split across the two children, so checking every row
        // observes a real consumer of each mirrored upload, not only rank zero.
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    float value;
                    std::memcpy(&value, expected.data() + (column * 4 + k) * sizeof(value), sizeof(value));
                    sum += w[row * 4 + k] * value;
                }
                const float reference = sum / (1.0f + std::exp(-sum));
                require(std::isfinite(actual[column * 4 + row]) &&
                            std::fabs(actual[column * 4 + row] - reference) <= 1e-6f,
                        "a child consumer observed stale or aliased host-copy data");
            }
        }
    };
    {
        tensor_fixture source;
        source.backend = ggml_backend_dev_init(cpu, nullptr);
        require(source.backend != nullptr, "cannot create host source backend");
        source.context = ggml_init({16 * 1024, nullptr, true});
        require(source.context != nullptr, "cannot create host source context");
        source.tensor = ggml_new_tensor_2d(source.context, GGML_TYPE_F32, 4, 4);
        source.buffer = ggml_backend_alloc_ctx_tensors(source.context, source.backend);
        require(source.buffer != nullptr, "cannot allocate host source tensor");
        for (int seed : {0, 48}) {
            expected = make_values(source.tensor, seed);
            ggml_backend_tensor_set(source.tensor, expected.data(), 0, expected.size());
            require(copy(source.backend, fixture.backend, source.tensor, fixture.tensor),
                    "contiguous host-to-mirrored copy was rejected");
            ggml_backend_synchronize(fixture.backend);
            check_full_tensor(source, expected);
            check_full_tensor(fixture, expected);

            // The source remains caller-owned and can be reused after completion.
            const auto replacement = make_values(source.tensor, seed + 17);
            ggml_backend_tensor_set(source.tensor, replacement.data(), 0, replacement.size());
            check_full_tensor(fixture, expected);
            check_consumer();
        }

        const auto source_before = make_values(source.tensor, 65);
        require(!copy(source.backend, fixture.backend, source.tensor, weights), "split copy target was accepted");
        ggml_backend_synchronize(fixture.backend);
        float weights_after[16];
        ggml_backend_tensor_get(weights, weights_after, 0, sizeof(weights_after));
        require(std::memcmp(weights_after, w, sizeof(w)) == 0, "rejected split copy changed destination bytes");
        ggml_tensor strided_src = *source.tensor;
        ggml_tensor strided_dst = *fixture.tensor;
        strided_src.nb[1] += sizeof(float);
        strided_dst.nb[1] += sizeof(float);
        require(!copy(source.backend, fixture.backend, &strided_src, &strided_dst), "strided copy was accepted");
        ggml_tensor other_layout = *source.tensor;
        other_layout.ne[0] = 3;
        require(!copy(source.backend, fixture.backend, &other_layout, fixture.tensor), "different layouts were accepted");
        require(!copy(source.backend, source.backend, source.tensor, source.tensor), "CPU destination backend was accepted");

        static split_config foreign_config = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, "foreign", {4, 4, 1, 1}};
        tensor_fixture foreign;
        foreign.init(cpu, foreign_config, GGML_TYPE_F32);
        const auto foreign_before = make_values(foreign.tensor, 99);
        ggml_backend_tensor_set(foreign.tensor, foreign_before.data(), 0, foreign_before.size());
        require(!copy(source.backend, fixture.backend, source.tensor, foreign.tensor), "foreign Meta buffer was accepted");
        require(!copy(foreign.backend, fixture.backend, foreign.tensor, fixture.tensor), "Meta buffer was treated as a host pointer");
        ggml_backend_synchronize(fixture.backend);
        ggml_backend_synchronize(foreign.backend);
        check_full_tensor(foreign, foreign_before);
        check_full_tensor(source, source_before);
        check_full_tensor(fixture, expected);
    }
    // Destroying the completed source must not invalidate either child's data.
    check_full_tensor(fixture, expected);
    check_consumer();
}

void run_scheduler_empty_input_consumer(ggml_backend_dev_t cpu) {
    tensor_fixture fixture;
    tensor_fixture source;
    ggml_backend_dev_t children[] = {cpu, cpu};
    const auto device = ggml_backend_meta_device(children, 2, split_copy_fixture, nullptr);
    require(device != nullptr, "cannot create scheduler fixture device");
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    source.backend = ggml_backend_dev_init(cpu, nullptr);
    require(fixture.backend != nullptr && source.backend != nullptr, "cannot create scheduler fixture backends");
    fixture.context = ggml_init({128 * 1024, nullptr, true});
    source.context = ggml_init({16 * 1024, nullptr, true});
    require(fixture.context != nullptr && source.context != nullptr, "cannot create scheduler fixture contexts");
    fixture.tensor = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 4);
    ggml_set_name(fixture.tensor, "copy_weights");
    auto * empty_destination = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 0);
    fixture.buffer = ggml_backend_alloc_ctx_tensors_from_buft(fixture.context, ggml_backend_dev_buffer_type(device));
    require(fixture.buffer != nullptr, "cannot allocate scheduler fixture weights");
    ggml_backend_buffer_set_usage(fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    source.tensor = ggml_new_tensor_2d(source.context, GGML_TYPE_F32, 4, 4);
    auto * bias = ggml_new_tensor_2d(source.context, GGML_TYPE_F32, 4, 4);
    ggml_set_input(source.tensor);
    ggml_set_input(bias);
    // Host metadata can be a non-INPUT view. The zero-row view carries no
    // bytes; the nonempty view must still be uploaded before computation.
    auto * empty = ggml_view_2d(source.context, source.tensor, 4, 0, source.tensor->nb[1], 0);
    auto * bias_view = ggml_view_2d(source.context, bias, 4, 4, bias->nb[1], 0);
    source.buffer = ggml_backend_alloc_ctx_tensors(source.context, source.backend);
    require(source.buffer != nullptr, "cannot allocate scheduler host inputs");
    auto * empty_copy = ggml_cpy(fixture.context, empty, empty_destination);
    auto * added = ggml_add(fixture.context, source.tensor, bias_view);
    auto * product = ggml_mul_mat(fixture.context, fixture.tensor, added);
    auto * result = ggml_silu(fixture.context, product);
    ggml_set_output(result);
    auto * graph = ggml_new_graph(fixture.context);
    ggml_build_forward_expand(graph, empty_copy);
    ggml_build_forward_expand(graph, result);
    ggml_backend_t backends[] = {fixture.backend, source.backend};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true), ggml_backend_sched_free);
    require(scheduler != nullptr, "cannot create scheduler for empty-input consumer");
    for (auto * node : {empty_copy, added, product, result}) {
        ggml_backend_sched_set_tensor_backend(scheduler.get(), node, fixture.backend);
    }
    require(ggml_backend_sched_alloc_graph(scheduler.get(), graph), "cannot allocate scheduler consumer graph");
    float w[16];
    for (int row = 0; row < 4; ++row) {
        for (int k = 0; k < 4; ++k) {
            w[row * 4 + k] = float((row + 1) * (k + 2)) / 16.0f;
        }
    }
    ggml_backend_tensor_set(fixture.tensor, w, 0, sizeof(w));
    for (int seed : {0, 5, 11}) {
        float x[16], b[16];
        for (int i = 0; i < 16; ++i) {
            x[i] = float((i * 3 + seed) % 17 - 8) / 16.0f;
            b[i] = float((i + seed) % 7 - 3) / 32.0f;
        }
        ggml_backend_tensor_set(source.tensor, x, 0, sizeof(x));
        ggml_backend_tensor_set(bias, b, 0, sizeof(b));
        require(ggml_backend_sched_graph_compute_async(scheduler.get(), graph) == GGML_STATUS_SUCCESS,
                "scheduler empty-input consumer execution failed");
        // The graph call must finish its uploads before returning, even though
        // compute may remain pending. Reuse both caller-owned sources now.
        const float replacement[16] = {};
        ggml_backend_tensor_set(source.tensor, replacement, 0, sizeof(replacement));
        ggml_backend_tensor_set(bias, replacement, 0, sizeof(replacement));
        ggml_backend_sched_synchronize(scheduler.get());
        float actual[16];
        ggml_backend_tensor_get(result, actual, 0, sizeof(actual));
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    const int i = column * 4 + k;
                    sum += w[row * 4 + k] * (x[i] + b[i]);
                }
                const float reference = sum / (1.0f + std::exp(-sum));
                require(std::isfinite(actual[column * 4 + row]) &&
                            std::fabs(actual[column * 4 + row] - reference) <= 1e-6f,
                        "scheduler consumer lost a host input around an empty view or retained caller storage");
            }
        }
    }
}

ggml_backend_meta_split_state split_expert_fixture(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state split = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    if (std::strcmp(tensor->name, "expert_weights") == 0 || std::strcmp(tensor->name, "expert_input") == 0) {
        split.axis = GGML_BACKEND_SPLIT_AXIS_0;
        split.ne[0] = split.ne[1] = 2;
    }
    return split;
}

void run_exported_expert_sum(ggml_backend_dev_t cpu, int exported_stage) {
    tensor_fixture fixture;
    ggml_backend_dev_t children[] = {cpu, cpu};
    const auto device = ggml_backend_meta_device(children, 2, split_expert_fixture, nullptr);
    require(device != nullptr, "cannot create expert fixture device");
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    require(fixture.backend != nullptr, "cannot create expert fixture backend");
    fixture.context = ggml_init({128 * 1024, nullptr, true});
    require(fixture.context != nullptr, "cannot create expert fixture context");
    auto * weights = ggml_new_tensor_3d(fixture.context, GGML_TYPE_F32, 4, 3, 2);
    auto * input = ggml_new_tensor_3d(fixture.context, GGML_TYPE_F32, 4, 2, 1);
    auto * ids = ggml_new_tensor_2d(fixture.context, GGML_TYPE_I32, 2, 1);
    auto * mix = ggml_new_tensor_3d(fixture.context, GGML_TYPE_F32, 1, 2, 1);
    ggml_set_name(weights, "expert_weights");
    ggml_set_name(input, "expert_input");
    fixture.buffer = ggml_backend_alloc_ctx_tensors_from_buft(fixture.context, ggml_backend_dev_buffer_type(device));
    require(fixture.buffer != nullptr, "cannot allocate expert fixture inputs");
    ggml_backend_buffer_set_usage(fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * graph = ggml_new_graph(fixture.context);
    auto * experts = ggml_mul_mat_id(fixture.context, weights, input, ids);
    ggml_build_forward_expand(graph, experts);
    auto * weighted = ggml_mul(fixture.context, experts, mix);
    auto * perm = ggml_permute(fixture.context, weighted, 1, 0, 2, 3);
    auto * cont = ggml_cont(fixture.context, perm);
    auto * sum = ggml_sum_rows(fixture.context, cont);
    auto * result = ggml_reshape_2d(fixture.context, sum, 3, 1);
    ggml_tensor * stages[] = {nullptr, experts, weighted, perm, cont, sum};
    if (exported_stage != 0) {
        ggml_set_output(stages[exported_stage]);
    }
    ggml_set_output(result);
    auto * consumer = ggml_silu(fixture.context, result);
    ggml_set_output(consumer);
    ggml_build_forward_expand(graph, consumer);
    fixture.allocator = ggml_gallocr_new(ggml_backend_dev_buffer_type(device));
    require(fixture.allocator != nullptr && ggml_gallocr_alloc_graph(fixture.allocator, graph),
            "cannot allocate expert compute graph");
    float w[24], x[8], expected_experts[6] = {}, expected_weighted[6], expected_cont[6], expected_sum[3] = {};
    const int32_t selected[] = {0, 1};
    const float mixture[] = {0.25f, 0.75f};
    for (int expert = 0; expert < 2; ++expert) {
        for (int row = 0; row < 3; ++row) {
            for (int k = 0; k < 4; ++k) {
                w[(expert * 3 + row) * 4 + k] = float((expert + 1) * (row + 1) * (k + 1)) / 16.0f;
            }
        }
    }
    for (int slot = 0; slot < 2; ++slot) {
        for (int k = 0; k < 4; ++k) {
            x[slot * 4 + k] = float((slot + 2) * (k - 1)) / 8.0f;
        }
        for (int row = 0; row < 3; ++row) {
            for (int k = 0; k < 4; ++k) {
                expected_experts[slot * 3 + row] += w[(slot * 3 + row) * 4 + k] * x[slot * 4 + k];
            }
            const float value = expected_experts[slot * 3 + row] * mixture[slot];
            expected_weighted[slot * 3 + row] = value;
            expected_cont[row * 2 + slot] = value;
            expected_sum[row] += value;
        }
    }
    ggml_backend_tensor_set(weights, w, 0, sizeof(w));
    ggml_backend_tensor_set(input, x, 0, sizeof(x));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    ggml_backend_tensor_set(mix, mixture, 0, sizeof(mixture));
    require(ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS, "expert fixture execution failed");
    float actual[6] = {};
    ggml_backend_tensor_get(result, actual, 0, sizeof(expected_sum));
    if (std::memcmp(actual, expected_sum, sizeof(expected_sum)) != 0) {
        std::fprintf(stderr, "expert stage=%d observed=%g,%g,%g expected=%g,%g,%g\n", exported_stage,
                     actual[0], actual[1], actual[2], expected_sum[0], expected_sum[1], expected_sum[2]);
    }
    require(std::memcmp(actual, expected_sum, sizeof(expected_sum)) == 0, "expert final sum differs from scalar reference");
    ggml_backend_tensor_get(consumer, actual, 0, sizeof(expected_sum));
    for (int row = 0; row < 3; ++row) {
        const float expected = expected_sum[row] / (1.0f + std::exp(-expected_sum[row]));
        require(std::isfinite(actual[row]) && std::fabs(actual[row] - expected) <= 1e-6f,
                "expert nonlinear consumer observed a partial sum");
    }
    if (exported_stage != 0) {
        const auto * expected = exported_stage == 1 ? expected_experts :
                               exported_stage <= 3 ? expected_weighted :
                               exported_stage == 4 ? expected_cont : expected_sum;
        const size_t bytes = exported_stage == 5 ? sizeof(expected_sum) : sizeof(expected_experts);
        ggml_backend_tensor_get(stages[exported_stage], actual, 0, bytes);
        require(std::memcmp(actual, expected, bytes) == 0, "exported expert intermediate lost contributions");
    }
}

} // namespace

int main(int argc, char ** argv) {
    bool do_get = true;
    bool do_set = true;
    bool zero_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--get-only") == 0 && do_get && do_set) {
            do_set = false;
        } else if (std::strcmp(argv[i], "--set-only") == 0 && do_get && do_set) {
            do_get = false;
        } else if (std::strcmp(argv[i], "--zero-only") == 0 && !zero_only) {
            zero_only = true;
        } else {
            std::fprintf(stderr, "usage: %s [--get-only | --set-only] [--zero-only]\n", argv[0]);
            return 1;
        }
    }

    try {
        const auto registry = ggml_backend_cpu_reg();
        require(registry != nullptr, "CPU registry is unavailable");
        const auto cpu = ggml_backend_reg_dev_get(registry, 0);
        require(cpu != nullptr, "CPU device is unavailable");
        size_t cases = 0;
        if (do_get && do_set && !zero_only) {
        run_host_copy_consumer(cpu);
        ++cases;
        run_scheduler_empty_input_consumer(cpu);
        ++cases;
        // The raw MUL_MAT_ID result is publicly typed PARTIAL and cannot be
        // read directly. Its weighted/permuted consumers are mirrored views.
        for (int stage : {0, 2, 3, 4, 5}) {
            run_exported_expert_sum(cpu, stage);
            ++cases;
        }
        }
        static split_config configs[] = {
            {GGML_BACKEND_SPLIT_AXIS_0, "axis0", {10, 6, 1, 1}},
            {GGML_BACKEND_SPLIT_AXIS_1, "axis1", {5, 10, 6, 1}},
            {GGML_BACKEND_SPLIT_AXIS_2, "axis2", {5, 2, 10, 6}},
            {GGML_BACKEND_SPLIT_AXIS_MIRRORED, "mirrored", {10, 6, 1, 1}},
        };
        for (auto & config : configs) {
            for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                tensor_fixture fixture;
                fixture.init(cpu, config, type);
                const int chunk_axis = config.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ? 0 : int(config.axis);
                const size_t chunk_bytes = fixture.tensor->nb[chunk_axis + 1];
                const size_t ranges[][2] = {{0, 6}, {1, 2}, {5, 1}};
                for (const auto & range : ranges) {
                    if (zero_only && range[0] != 0) {
                        continue;
                    }
                    for (const bool is_set : {false, true}) {
                        if ((is_set && !do_set) || (!is_set && !do_get)) {
                            continue;
                        }
                        std::fprintf(stderr, "RUN %s %s %s start=%zu count=%zu\n", config.name, ggml_type_name(type),
                                     is_set ? "set" : "get", range[0], range[1]);
                        if (is_set) {
                            run_set(fixture, range[0] * chunk_bytes, range[1] * chunk_bytes);
                        } else {
                            run_get(fixture, range[0] * chunk_bytes, range[1] * chunk_bytes);
                        }
                        ++cases;
                    }
                }
            }
        }
        require(cases != 0, "no transfer case ran");
        std::printf("PASS %zu real meta contract cases on two CPU children\n", cases);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
