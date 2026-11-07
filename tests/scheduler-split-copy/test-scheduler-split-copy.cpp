#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// CPU execution and allocation with isolated buffer types. This forces the
// scheduler's real cross-backend copies without a GPU or a simulated copy.
bool not_host(ggml_backend_buffer_type_t) { return false; }
ggml_backend_buffer_type_t own_buffer(ggml_backend_dev_t device) {
    return static_cast<ggml_backend_buffer_type_t>(device->context);
}
bool own_buffer_only(ggml_backend_dev_t device, ggml_backend_buffer_type_t buffer_type) {
    return buffer_type == static_cast<ggml_backend_buffer_type_t>(device->context);
}

struct isolated_cpu {
    ggml_backend_buffer_type buffer_type;
    ggml_backend_device device;
    ggml_backend backend;

    explicit isolated_cpu(ggml_backend_t cpu) :
        buffer_type(*ggml_backend_cpu_buffer_type()),
        device(*ggml_backend_get_device(cpu)),
        backend(*cpu) {
        buffer_type.iface.is_host = not_host;
        buffer_type.device = &device;
        device.iface.get_buffer_type = own_buffer;
        device.context = &buffer_type;
        device.iface.supports_buft = own_buffer_only;
        backend.device = &device;
    }
};

constexpr int64_t rows = 3;
constexpr int64_t capacity = 8;

struct shape {
    std::array<int64_t, GGML_MAX_DIMS> ne;
    std::array<size_t, GGML_MAX_DIMS> nb;
};

shape contiguous_shape(int64_t width) {
    const size_t row_bytes = static_cast<size_t>(width) * sizeof(float);
    return {{width, rows, 1, 1}, {sizeof(float), row_bytes, row_bytes * rows, row_bytes * rows}};
}

void set_shape(ggml_tensor * tensor, const shape & s) {
    std::memcpy(tensor->ne, s.ne.data(), sizeof(tensor->ne));
    std::memcpy(tensor->nb, s.nb.data(), sizeof(tensor->nb));
}

ggml_backend_sched_tensor_layout layout(ggml_tensor * tensor, const shape & s) {
    ggml_backend_sched_tensor_layout result = {};
    result.tensor = tensor;
    std::memcpy(result.ne, s.ne.data(), sizeof(result.ne));
    std::memcpy(result.nb, s.nb.data(), sizeof(result.nb));
    return result;
}

bool has_shape(const ggml_tensor * tensor, const shape & s) {
    return std::memcmp(tensor->ne, s.ne.data(), sizeof(tensor->ne)) == 0 &&
           std::memcmp(tensor->nb, s.nb.data(), sizeof(tensor->nb)) == 0;
}

void run(bool parallel, bool single_split, bool tracking = true) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    require(cpu != nullptr, "CPU backend initialization failed");
    isolated_cpu first(cpu), second(cpu);
    ggml_backend_t backends[] = {&first.backend, &second.backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 3, 128, parallel, false);
    require(sched != nullptr, "scheduler initialization failed");
    if (tracking) {
        require(ggml_backend_sched_set_split_copy_tracking(sched, true), "could not enable split-copy tracking");
    }

    ggml_context * inputs = ggml_init({16384, nullptr, true});
    ggml_context * ctx = ggml_init({65536, nullptr, true});
    require(inputs && ctx, "context initialization failed");
    auto * x = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, capacity, rows);
    auto * y = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, capacity, rows);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    require(input_buffer != nullptr, "input allocation failed");

    auto * out1 = ggml_add(ctx, x, y);
    auto * out2 = ggml_sub(ctx, x, y);
    ggml_set_output(out1);
    ggml_set_output(out2);
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out1);
    ggml_build_forward_expand(graph, out2);
    ggml_backend_sched_set_tensor_backend(sched, out1, &first.backend);
    ggml_backend_sched_set_tensor_backend(sched, out2, single_split ? &first.backend : &second.backend);
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation failed");
    require(ggml_backend_sched_get_n_splits(sched) == (single_split ? 1 : 2), "unexpected split count");
    require(ggml_backend_sched_tensor_has_split_copy(sched, x) &&
            ggml_backend_sched_tensor_has_split_copy(sched, y), "expected actual split copies");
    ggml_tensor * copies[] = {out1->src[0], out1->src[1], out2->src[0], out2->src[1]};
    require(copies[0] != x && copies[1] != y && copies[2] != x && copies[3] != y,
            "split sources were not rewritten to copies");
    require(single_split ? copies[0] == copies[2] && copies[1] == copies[3]
                         : copies[0] != copies[2] && copies[1] != copies[3],
            "split destinations have unexpected copy sharing");
    require(parallel ? ggml_backend_sched_get_n_copies(sched) > 1 : ggml_backend_sched_get_n_copies(sched) == 1,
            "unexpected copy slot count");

    auto check_copies = [&](const shape & expected) {
        for (const auto * copy : copies) {
            require(has_shape(copy, expected), "split-copy shape or stride differs");
        }
    };
    auto compute = [&](const shape & s, int seed) {
        const size_t count = static_cast<size_t>(s.ne[0] * rows);
        std::vector<float> a(count), b(count), actual(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = float(static_cast<int>(i) + seed);
            b[i] = float(2 * static_cast<int>(i) - seed);
        }
        ggml_backend_tensor_set(x, a.data(), 0, count * sizeof(float));
        ggml_backend_tensor_set(y, b.data(), 0, count * sizeof(float));
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        for (int output = 0; output < 2; ++output) {
            auto * tensor = output == 0 ? out1 : out2;
            require(has_shape(tensor, s), "output shape or stride differs");
            ggml_backend_tensor_get(tensor, actual.data(), 0, count * sizeof(float));
            for (size_t i = 0; i < count; ++i) {
                const float expected = output == 0 ? a[i] + b[i] : a[i] - b[i];
                require(actual[i] == expected, "split-copy output content differs");
            }
        }
    };

    shape current = contiguous_shape(capacity);
    check_copies(current);
    compute(current, 0);
    if (!tracking) {
        const shape next = contiguous_shape(3);
        ggml_backend_sched_tensor_layout updates[] = {layout(x, next), layout(y, next)};
        require(!ggml_backend_sched_update_split_copies(sched, updates, 2, false), "untracked dry run accepted");
        require(!ggml_backend_sched_update_split_copies(sched, updates, 2, true), "untracked apply accepted");
        check_copies(current);
        require(has_shape(x, current) && has_shape(y, current), "untracked rejection changed source metadata");
        compute(current, 29);
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        ggml_free(inputs);
        ggml_backend_free(cpu);
        return;
    }
    for (int64_t width : std::array<int64_t, 3>{3, 6, capacity}) {
        const shape next = contiguous_shape(width);
        ggml_backend_sched_tensor_layout updates[] = {layout(x, next), layout(y, next)};
        require(ggml_backend_sched_update_split_copies(sched, updates, 2, false), "valid layout rejected by dry run");
        check_copies(current);
        require(has_shape(x, current) && has_shape(y, current), "dry run changed source metadata");
        require(ggml_backend_sched_update_split_copies(sched, updates, 2, true), "valid layout rejected by apply");
        check_copies(next);
        require(has_shape(x, current) && has_shape(y, current), "apply changed source metadata");
        set_shape(x, next);
        set_shape(y, next);
        set_shape(out1, next);
        set_shape(out2, next);
        ggml_backend_sched_refresh_split_copy_views(sched);
        current = next;
        compute(current, static_cast<int>(width));
    }

    // A valid first entry followed by an out-of-envelope second entry must not
    // update any copies, even on the committing call.
    const shape smaller = contiguous_shape(4);
    const shape too_large = contiguous_shape(capacity + 1);
    ggml_backend_sched_tensor_layout invalid[] = {layout(x, smaller), layout(y, too_large)};
    require(!ggml_backend_sched_update_split_copies(sched, invalid, 2, false), "oversized dry run accepted");
    check_copies(current);
    require(!ggml_backend_sched_update_split_copies(sched, invalid, 2, true), "oversized apply accepted");
    check_copies(current);
    require(has_shape(x, current) && has_shape(y, current), "rejection changed source metadata");
    compute(current, 19);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx);
    ggml_free(inputs);
    ggml_backend_free(cpu);
}

void run_zero_copy() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    require(cpu != nullptr, "CPU backend initialization failed");
    ggml_backend_t backends[] = {cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 128, false, false);
    require(sched != nullptr, "scheduler initialization failed");
    require(ggml_backend_sched_set_split_copy_tracking(sched, true), "could not enable split-copy tracking");

    ggml_context * inputs = ggml_init({16384, nullptr, true});
    ggml_context * ctx = ggml_init({65536, nullptr, true});
    require(inputs && ctx, "context initialization failed");
    ggml_tensor * x = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, capacity, rows);
    ggml_set_input(x);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    require(input_buffer != nullptr, "input allocation failed");
    ggml_tensor * out = ggml_add(ctx, x, x);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation failed");
    require(ggml_backend_sched_get_n_splits(sched) == 1 &&
            !ggml_backend_sched_tensor_has_split_copy(sched, x) && out->src[0] == x,
            "expected one CPU split without a split copy");

    const shape next = contiguous_shape(3);
    ggml_backend_sched_tensor_layout update = layout(x, next);
    require(ggml_backend_sched_update_split_copies(sched, &update, 1, false), "zero-copy dry run rejected");
    require(ggml_backend_sched_update_split_copies(sched, &update, 1, true), "zero-copy apply rejected");
    require(has_shape(x, contiguous_shape(capacity)), "zero-copy apply changed source metadata");
    set_shape(x, next);
    set_shape(out, next);
    ggml_backend_sched_refresh_split_copy_views(sched);
    std::array<float, 3 * rows> values = {}, actual = {};
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    ggml_backend_tensor_set(x, values.data(), 0, sizeof(values));
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "zero-copy compute failed");
    require(has_shape(out, next), "zero-copy output shape or stride differs");
    ggml_backend_tensor_get(out, actual.data(), 0, sizeof(actual));
    for (size_t i = 0; i < values.size(); ++i) {
        require(actual[i] == 2 * values[i], "zero-copy output content differs");
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx);
    ggml_free(inputs);
    ggml_backend_free(cpu);
}

} // namespace

int main() {
    try {
        run(false, false);
        run(true, false);
        run(false, true);
        run(true, true);
        run(false, true, false);
        run_zero_copy();
        std::puts("scheduler split-copy regression passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "scheduler split-copy regression failed: %s\n", error.what());
        return 1;
    }
}
