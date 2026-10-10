// CPU-only HC conversion helper using the repository's actual GGML quantizer.
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int tp5_hc_quantize(const void * input, int type, int64_t rows, int64_t width,
                                 void * output, double * max_abs, double * relative_l2) {
    if (!input || !output || !max_abs || !relative_l2 || rows <= 0 || width <= 0 || width % 32) return 1;
    const int64_t n = rows * width;
    if (n / width != rows || n > (1LL << 32)) return 1;
    std::vector<float> x(n);
    if (type == GGML_TYPE_BF16) ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(input), x.data(), n);
    else if (type == GGML_TYPE_F16) ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(input), x.data(), n);
    else if (type == GGML_TYPE_F32) std::memcpy(x.data(), input, n * sizeof(float));
    else return 1;
    for (float v : x) if (!std::isfinite(v)) return 1;
    const size_t bytes = ggml_quantize_chunk(GGML_TYPE_Q8_0, x.data(), output, 0, rows, width, nullptr);
    if (bytes != size_t(n / 32) * 34) return 1;
    double error = 0, norm = 0;
    *max_abs = 0;
    const auto * encoded = static_cast<const uint8_t *>(output);
    for (int64_t block = 0; block < n / 32; ++block) {
        ggml_fp16_t h;
        std::memcpy(&h, encoded + 34 * block, sizeof(h));
        const double scale = ggml_fp16_to_fp32(h);
        const auto * q = reinterpret_cast<const int8_t *>(encoded + 34 * block + 2);
        for (int j = 0; j < 32; ++j) {
            const double original = x[32 * block + j];
            const double delta = scale * q[j] - original;
            error += delta * delta;
            norm += original * original;
            *max_abs = std::max(*max_abs, std::fabs(delta));
        }
    }
    *relative_l2 = std::sqrt(error / std::max(norm, 1e-60));
    return 0;
}
