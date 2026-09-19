#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../ggml/src/ggml-vulkan/ggml-vulkan-scratch.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>

int main() {
    constexpr uint64_t limit = 256 * 1024 * 1024;
    constexpr uint64_t granule = 64 * 1024;
    assert(ggml_vk_scratch_capacity(0, limit) == 0);
    assert(ggml_vk_scratch_capacity(1, limit) == granule);
    assert(ggml_vk_scratch_capacity(granule, limit) == granule);
    assert(ggml_vk_scratch_capacity(granule + 1, limit) == 2 * granule);
    assert(ggml_vk_scratch_capacity(limit - 1, limit) == limit);
    assert(ggml_vk_scratch_capacity(limit + 1, limit) == limit + 1); // caller rejects
    assert(ggml_vk_scratch_capacity(1025, 1027) == 1025); // non-aligned limit
    const auto max = std::numeric_limits<uint64_t>::max();
    assert(ggml_vk_scratch_capacity(max - 1, max) == max - 1); // cannot wrap

    uint64_t checked = 0;
    for (uint32_t alignment : {16u, 32u, 64u, 128u, 256u}) {
        for (uint32_t target : {2u, 4u, 5u, 8u, 12u, 16u, 24u, 40u, 60u, 120u}) {
            for (uint32_t kv = 1; kv <= 8192; ++kv) {
                const auto split_kv = ((std::max(1u, kv / target) + alignment - 1) / alignment) * alignment;
                const auto splits = (kv + split_kv - 1) / split_kv;
                for (uint64_t rows : {1u, 32u, 512u}) {
                    const uint64_t required = splits > 1 ? (128 + 2) * sizeof(float) * rows * splits * 5 : 0;
                    const auto capacity = ggml_vk_scratch_capacity(required, limit);
                    assert(capacity >= required);
                    if (required <= limit) {
                        assert(capacity <= limit);
                        assert(capacity - required < granule);
                    }
                    ++checked;
                }
            }
        }
    }
    printf("test-vulkan-scratch: %llu split-K capacity cases passed (no GPU required).\n",
           (unsigned long long) checked);
}
