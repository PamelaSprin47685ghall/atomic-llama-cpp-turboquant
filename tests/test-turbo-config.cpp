// Layer-adaptive Turbo KV policy resolution is per cache instance. Keep this
// test independent from model loading so fit probes and mixed-cache processes
// can exercise the policy without constructing a full llama_context.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/llama-turbo-config.h"

#include <cassert>

int main() {
    // Auto Boundary-V applies only to sufficiently deep turbo2-V caches.
    assert(llama_turbo_layer_adaptive_mode(GGML_TYPE_TURBO2_0, 44, nullptr) == 7);
    assert(llama_turbo_layer_adaptive_mode(GGML_TYPE_TURBO2_0, 7,  nullptr) == 0);

    // A later cache in the same process must resolve from its own requested
    // V type, not inherit the first cache's auto mode.
    assert(llama_turbo_layer_adaptive_mode(GGML_TYPE_TURBO3_0, 44, nullptr) == 0);
    assert(llama_turbo_layer_adaptive_mode(GGML_TYPE_TURBO4_0, 44, nullptr) == 0);

    // Explicit policy continues to override auto selection exactly as before.
    assert(llama_turbo_layer_adaptive_mode(GGML_TYPE_TURBO2_0, 44, "0") == 0);
    assert(llama_turbo_layer_adaptive_mode(GGML_TYPE_TURBO3_0, 44, "5") == 5);
    assert(llama_turbo_layer_adaptive_mode(GGML_TYPE_TURBO4_0, 44, "7") == 7);
    return 0;
}
