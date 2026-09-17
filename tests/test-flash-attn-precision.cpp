#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

static bool check_f32_accumulation(int64_t n_kv) {
    ggml_context_ptr ctx(ggml_init({ 4 * 1024 * 1024, nullptr, false }));
    GGML_ASSERT(ctx);
    constexpr int64_t d = 64;
    ggml_tensor *     q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, 1, 1, 1);
    ggml_tensor *     k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, d, n_kv, 1, 1);
    ggml_tensor *     v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, d, n_kv, 1, 1);
    std::fill_n(static_cast<float *>(q->data), d, 0.0f);
    static_cast<float *>(q->data)[0] = 1.0f;
    std::fill_n(static_cast<ggml_fp16_t *>(k->data), d * n_kv, ggml_fp32_to_fp16(0.0f));

    // A constant V must survive every normalized weighting unchanged. Increasing
    // scores exercise both accumulator rescaling and addition; this value is
    // exactly representable in F16 but loses precision in an F16 accumulator.
    constexpr float expected = 1.0009765625f;
    for (int64_t j = 0; j < n_kv; ++j) {
        static_cast<ggml_fp16_t *>(k->data)[j * d] = ggml_fp32_to_fp16(float(j) / 64.0f);
        for (int64_t i = 0; i < d; ++i) {
            static_cast<ggml_fp16_t *>(v->data)[j * d + i] = ggml_fp32_to_fp16(i % 2 ? -expected : expected);
        }
    }

    ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx.get(), graph, 4) == GGML_STATUS_SUCCESS);

    float max_abs = 0.0f;
    for (int64_t i = 0; i < d; ++i) {
        const float actual = static_cast<const float *>(out->data)[i];
        max_abs = std::isfinite(actual) ? std::max(max_abs, std::abs(actual - (i % 2 ? -expected : expected))) :
                                          std::numeric_limits<float>::infinity();
    }
    const bool pass = max_abs < 3e-6f;
    std::printf("F32 attention n_kv=%lld max_abs=%.9g %s\n", (long long) n_kv, max_abs, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    const bool direct   = check_f32_accumulation(256);
    const bool split_kv = check_f32_accumulation(1024);
    return direct && split_kv ? 0 : 1;
}
