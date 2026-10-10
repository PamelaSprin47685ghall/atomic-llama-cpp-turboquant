// Read+write streaming calibration, not GEMV performance or model tok/s.
// Four independent 64 MiB inputs and outputs exceed the RX 6800 cache.
// Run serially under the TP5 GPU lock, never alongside a model benchmark.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2 || argv[1][0] < '0' || argv[1][0] > '4' || argv[1][1] != '\0') return 2;
    const size_t device = static_cast<size_t>(argv[1][0] - '0');
    auto * reg = ggml_backend_vk_reg();
    if (ggml_backend_reg_dev_count(reg) <= device) return 2;
    auto * backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, device), nullptr);
    if (!backend) return 2;
    constexpr size_t parts = 4, elements = 16 * 1024 * 1024;
    constexpr float scale = 1.0001f;
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    if (!ctx) { ggml_backend_free(backend); return 2; }
    auto * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_tensor * src[parts];
    ggml_tensor * dst[parts];
    for (size_t i = 0; i < parts; ++i) {
        src[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        dst[i] = ggml_scale(ctx, src[i], scale);
        ggml_build_forward_expand(graph, dst[i]);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); ggml_backend_free(backend); return 2; }
    std::vector<float> host(elements);
    for (size_t i = 0; i < elements; ++i) host[i] = float(i % 1024) / 1024.0f;
    for (auto * t : src) ggml_backend_tensor_set(t, host.data(), 0, elements * sizeof(float));
    std::vector<double> times;
    int result = 0;
    for (int iteration = 0; iteration < 24; ++iteration) {
        const auto begin = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { result = 1; break; }
        ggml_backend_synchronize(backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (iteration >= 4) times.push_back(ms);
    }
    for (auto * t : dst) {
        float observed = 0;
        ggml_backend_tensor_get(t, &observed, (elements - 1) * sizeof(float), sizeof(float));
        if (std::fabs(observed - host.back() * scale) > 1e-5f) result = 1;
    }
    if (!result && times.size() == 20) {
        std::sort(times.begin(), times.end());
        const double median = 0.5 * (times[9] + times[10]);
        const size_t bytes = 2 * parts * elements * sizeof(float);
        std::printf("{\"device\":%zu,\"status\":\"STREAM_CALIBRATION_NOT_MODEL\",\"bytes_read_plus_written\":%zu,"
                    "\"median_ms\":%.9f,\"min_ms\":%.9f,\"p90_ms\":%.9f,\"effective_GB_s\":%.6f,\"samples_ms\":[",
                    device, bytes, median, times.front(), times[17], bytes / (median * 1e6));
        for (size_t i = 0; i < times.size(); ++i) std::printf("%s%.9f", i ? "," : "", times[i]);
        std::printf("]}\n");
    }
    ggml_backend_synchronize(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}
