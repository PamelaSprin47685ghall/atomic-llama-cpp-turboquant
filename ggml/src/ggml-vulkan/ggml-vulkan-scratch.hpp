#pragma once

#include <cstdint>

// Round allocation capacity, not dispatch geometry. Recomputing the number
// of splits from a rounded KV length can produce FEWER splits than the
// shader writes (e.g. KV=129, alignment=32: five actual splits, four rounded).
// Keep modest spare capacity so decode does not retire the entire replay
// cache for a small scratch growth. Never exceed the device's range limit.
inline uint64_t ggml_vk_scratch_capacity(uint64_t required, uint64_t limit) {
    constexpr uint64_t granule = 64 * 1024;
    if (required == 0 || required > limit) {
        return required;
    }
    const uint64_t padding = (0 - required) & (granule - 1);
    return padding <= limit - required ? required + padding : required;
}
