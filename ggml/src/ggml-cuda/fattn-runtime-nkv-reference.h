#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <limits>

// Reference only: physical stride is independent of the logical KV extent.
// This does not bind the GPU launch or update captured graph parameters.
inline bool fattn_runtime_nkv_enabled() {
    const char * value = std::getenv("GGML_TP5_FA_RUNTIME_NKV");
    return value && std::strcmp(value, "1") == 0;
}

inline float fattn_runtime_nkv_reference(const float * logits, const float * values,
                                         std::size_t n_kv, std::size_t value_stride) {
    if (n_kv == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t base = 0; base < n_kv; base += 256) {
        const std::size_t end = base + ((n_kv - base < 256) ? n_kv - base : 256);
        for (std::size_t i = base; i < end; ++i) maximum = std::fmax(maximum, logits[i]);
    }
    float numerator = 0.0f, denominator = 0.0f;
    for (std::size_t base = 0; base < n_kv; base += 256) {
        const std::size_t end = base + ((n_kv - base < 256) ? n_kv - base : 256);
        for (std::size_t i = base; i < end; ++i) {
            const float weight = std::exp(logits[i] - maximum);
            numerator += weight * values[i * value_stride];
            denominator += weight;
        }
    }
    return numerator / denominator;
}
