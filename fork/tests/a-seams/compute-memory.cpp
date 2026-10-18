#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using scheduler_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;

constexpr int64_t large_count = 1 << 20;
constexpr int64_t small_count = 128;
constexpr size_t graph_capacity = 32;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

context_ptr make_context(size_t tensors, bool graph) {
    const ggml_init_params params = {
        tensors * ggml_tensor_overhead() +
            (graph ? ggml_graph_overhead_custom(graph_capacity, false) : 0),
        nullptr,
        true,
    };
    context_ptr result(ggml_init(params), ggml_free);
    require(result != nullptr, "metadata context allocation failed");
    return result;
}

bool observe(ggml_tensor *, bool ask, void * user_data) {
    if (!ask) {
        ++*static_cast<size_t *>(user_data);
    }
    return true;
}

// Queue the readback before release so the release operation itself must
// finish both compute and output transfer. Old graph tensors are never read
// after release; only the owned host readback and separate input buffer are.
size_t run_graph(ggml_backend_sched_t sched, ggml_backend_t backend,
                 ggml_tensor * input, const std::vector<float> & source,
                 int64_t count, bool release_after_queue) {
    auto metadata = make_context(graph_capacity, true);
    auto * view = ggml_view_1d(metadata.get(), input, count, 0);
    auto * output = ggml_scale(metadata.get(), view, 2.0f);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(metadata.get(), graph_capacity, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_sched_reset(sched);
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation failed");
    const size_t owned = ggml_backend_sched_get_buffer_size(sched, backend);
    require(owned > 0, "graph did not own compute storage");
    require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS,
            "graph computation failed");
    std::vector<float> result(static_cast<size_t>(count));
    ggml_backend_tensor_get_async(backend, output, result.data(), 0, result.size() * sizeof(float));
    if (release_after_queue) {
        ggml_backend_sched_release_buffers(sched);
    } else {
        ggml_backend_sched_synchronize(sched);
    }
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] != 2.0f * source[i]) {
            throw std::runtime_error("graph output mismatch at " + std::to_string(i));
        }
    }
    return owned;
}

void verify_input(ggml_tensor * input, const std::vector<float> & source) {
    std::vector<float> result(source.size());
    ggml_backend_tensor_get(input, result.data(), 0, result.size() * sizeof(float));
    require(result == source, "external input storage was changed or released");
}

} // namespace

int main() {
    try {
        backend_ptr backend(ggml_backend_cpu_init(), ggml_backend_free);
        require(backend != nullptr, "CPU backend initialization failed");
        auto weights = make_context(1, false);
        auto * input = ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, large_count);
        buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(weights.get(), backend.get()),
                          ggml_backend_buffer_free);
        require(buffer != nullptr, "external input allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> source(static_cast<size_t>(large_count));
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<float>(static_cast<int>(i % 251) - 125) * 0.125f;
        }
        ggml_backend_tensor_set(input, source.data(), 0, source.size() * sizeof(float));

        ggml_backend_t backends[] = {backend.get()};
        scheduler_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, graph_capacity, false, true),
                            ggml_backend_sched_free);
        require(sched != nullptr, "scheduler initialization failed");
        size_t callbacks = 0;
        ggml_backend_sched_set_eval_callback(sched.get(), observe, &callbacks);
        const auto identity = sched.get();
        const int copies = ggml_backend_sched_get_n_copies(sched.get());

        const size_t large = run_graph(sched.get(), backend.get(), input, source, large_count, true);
        require(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == 0,
                "release retained allocator-owned storage");
        require(callbacks > 0, "evaluation callback did not execute");
        verify_input(input, source);
        const size_t callbacks_after_large = callbacks;

        ggml_backend_sched_release_buffers(sched.get());
        require(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == 0,
                "repeated release is not idempotent");
        require(sched.get() == identity && ggml_backend_sched_get_n_backends(sched.get()) == 1 &&
                    ggml_backend_sched_get_backend(sched.get(), 0) == backend.get() &&
                    ggml_backend_sched_get_n_copies(sched.get()) == copies,
                "release changed scheduler/backend identity or copy configuration");

        const size_t small = run_graph(sched.get(), backend.get(), input, source, small_count, false);
        require(small < large, "small graph retained the prefill high-water allocation");
        require(callbacks > callbacks_after_large, "release discarded the evaluation callback");
        ggml_backend_sched_release_buffers(sched.get());
        require(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == 0,
                "small graph release retained storage");

        const size_t regrown = run_graph(sched.get(), backend.get(), input, source, large_count, true);
        require(regrown > small, "allocator did not grow for a later large graph");
        require(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == 0,
                "final release retained storage");
        verify_input(input, source);
        std::cout << "PASS compute memory: large=" << large << " released=0 small=" << small
                  << " regrown=" << regrown << " final=0 callbacks=" << callbacks << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL compute memory: " << error.what() << '\n';
        return 1;
    }
}
