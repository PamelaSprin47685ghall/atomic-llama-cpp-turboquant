#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdlib>

// Resolve the layer-adaptive KV policy for one cache instance. This must not
// be process-global: a process may create caches with different requested V
// types (fit probes, tests, multiple models/contexts). A function-local static
// would make the first cache silently dictate all later caches.
static inline int llama_turbo_layer_adaptive_mode(
        ggml_type type_v,
        uint32_t n_layer,
        const char * env_value) {
    if (env_value != nullptr) {
        return std::atoi(env_value);
    }
    return type_v == GGML_TYPE_TURBO2_0 && n_layer >= 8 ? 7 : 0;
}
