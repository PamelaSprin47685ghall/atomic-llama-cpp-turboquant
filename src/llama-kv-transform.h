#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size);
GGML_API void turbo_cpu_fwht(float * x, int group_size);
GGML_API void ggml_turbo_wht_row(float * x, int group_size);
GGML_API void ggml_turbo_wht_inverse_row(float * x, int group_size);
#ifdef __cplusplus
}
#endif

inline bool llama_kv_is_turbo(ggml_type type) {
    return type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

inline bool llama_kv_needs_inverse_turbo(ggml_type type) {
#if defined(TURBO4_USE_4BIT) && !TURBO4_USE_4BIT
    return type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0;
#else
    return llama_kv_is_turbo(type);
#endif
}

// The optional attention Hadamard is NOT TurboQuant's signed WHT. It is
// self-inverse and is applied before padding/quantization, per attention head.
inline void llama_kv_hadamard(float * data, uint32_t head_dim, uint32_t nrot) {
    if (nrot == 0) {
        return;
    }
    if ((nrot & (nrot - 1)) != 0 || head_dim % nrot != 0) {
        throw std::runtime_error("invalid KV attention Hadamard geometry");
    }
    const float scale = 1.0f / std::sqrt((float) nrot);
    for (uint32_t off = 0; off < head_dim; off += nrot) {
        for (uint32_t step = 1; step < nrot; step *= 2) {
            for (uint32_t i = 0; i < nrot; i += 2 * step) {
                for (uint32_t j = 0; j < step; ++j) {
                    const float a = data[off + i + j];
                    const float b = data[off + i + j + step];
                    data[off + i + j] = a + b;
                    data[off + i + j + step] = a - b;
                }
            }
        }
        for (uint32_t i = 0; i < nrot; ++i) {
            data[off + i] *= scale;
        }
    }
}

// Restore actual post-RoPE K, including Turbo's storage transform. F32's
// type traits intentionally have no to_float callback.
inline void llama_kv_decode_key(ggml_type type, const void * src, float * dst, uint32_t n) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, n * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(type);
        if (!traits || !traits->to_float || n % ggml_blck_size(type) != 0) {
            throw std::runtime_error("unsupported K cache dequantization geometry");
        }
        traits->to_float(src, dst, n);
    }
    if (llama_kv_needs_inverse_turbo(type)) {
        for (uint32_t off = 0; off < n; off += 128) {
            turbo_cpu_fwht_inverse(dst + off, 128);
        }
    }
}

// A position shift changes phase, not magnitude. In particular YaRN's
// attention scale must NOT be multiplied into an already-scaled stored K.
inline void llama_kv_shift_key(float * key, uint32_t rotary_dim, bool neox,
                               const float * omega, int32_t delta) {
    for (uint32_t f = 0; f < rotary_dim / 2; ++f) {
        const uint32_t re = neox ? f : 2 * f;
        const uint32_t im = neox ? f + rotary_dim / 2 : 2 * f + 1;
        const double phase = (double) omega[f] * delta;
        const float c = (float) std::cos(phase);
        const float s = (float) std::sin(phase);
        const float a = key[re], b = key[im];
        key[re] = a * c - b * s;
        key[im] = a * s + b * c;
    }
}
