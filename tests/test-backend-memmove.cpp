#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " #cond " at line " << __LINE__ << "\n"; \
        return 1; \
    } \
} while (0)

static void fill_pattern(std::vector<uint8_t> & data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (uint8_t) ((i * 37 + 11) & 0xff);
    }
}

static int test_buffer(ggml_backend_buffer_type_t buft) {
    ggml_init_params params = {};
    params.mem_size = 2 * ggml_tensor_overhead() + 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64); // 256 bytes
    CHECK(ggml_nbytes(tensor) == 256);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    CHECK(buffer);

    std::vector<uint8_t> original(256);
    fill_pattern(original);

    // Contiguous overlapping downward move.
    ggml_backend_tensor_set(tensor, original.data(), 0, original.size());
    std::vector<uint8_t> expected = original;
    memmove(expected.data() + 16, expected.data() + 64, 128);

    ggml_backend_tensor_memmove_region contiguous = {};
    contiguous.tensor = tensor;
    contiguous.src_offset = 64;
    contiguous.dst_offset = 16;
    contiguous.size = 128;
    contiguous.n_copies = 1;

    CHECK(ggml_backend_tensor_memmove_regions_supported(&contiguous, 1));
    CHECK(ggml_backend_tensor_memmove_regions(&contiguous, 1));

    std::vector<uint8_t> actual(original.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    CHECK(actual == expected);

    // Strided row moves in one backend submission.
    ggml_backend_tensor_set(tensor, original.data(), 0, original.size());
    expected = original;
    for (size_t row = 0; row < 4; ++row) {
        memmove(expected.data() + row * 48, expected.data() + 16 + row * 48, 16);
    }

    ggml_backend_tensor_memmove_region strided = {};
    strided.tensor = tensor;
    strided.src_offset = 16;
    strided.dst_offset = 0;
    strided.size = 16;
    strided.n_copies = 4;
    strided.src_stride = 48;
    strided.dst_stride = 48;

    CHECK(ggml_backend_tensor_memmove_regions_supported(&strided, 1));
    CHECK(ggml_backend_tensor_memmove_regions(&strided, 1));
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    CHECK(actual == expected);

    // Two-byte and byte-unaligned ranges use native compute on Vulkan.
    ggml_backend_tensor_set(tensor, original.data(), 0, original.size());
    ggml_backend_tensor_memmove_region unaligned = {};
    unaligned.tensor = tensor;
    unaligned.src_offset = 4;
    unaligned.dst_offset = 0;
    unaligned.size = 2;
    unaligned.n_copies = 1;
    CHECK(ggml_backend_tensor_memmove_regions_supported(&unaligned, 1));
    CHECK(ggml_backend_tensor_memmove_regions(&unaligned, 1));
    expected = original;
    memmove(expected.data(), expected.data() + 4, 2);
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    CHECK(actual == expected);

    for (size_t src : {size_t(1), size_t(13), size_t(64)}) {
        for (size_t dst : {size_t(2), size_t(17), size_t(80)}) {
            ggml_backend_tensor_set(tensor, original.data(), 0, original.size());
            unaligned.src_offset = src;
            unaligned.dst_offset = dst;
            unaligned.size = 31;
            unaligned.n_copies = 3;
            unaligned.src_stride = unaligned.dst_stride = 48;
            expected = original;
            for (size_t c = 0; c < 3; ++c) {
                memmove(expected.data() + dst + c * 48, expected.data() + src + c * 48, 31);
            }
            CHECK(ggml_backend_tensor_memmove_regions(&unaligned, 1));
            ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
            CHECK(actual == expected);
        }
    }
    // A later invalid region must reject the WHOLE batch without mutation.
    ggml_backend_tensor_set(tensor, original.data(), 0, original.size());
    auto bad = contiguous;
    bad.src_offset = SIZE_MAX - 1;
    ggml_backend_tensor_memmove_region batch[] = {contiguous, bad};
    CHECK(!ggml_backend_tensor_memmove_regions_supported(batch, 2));
    CHECK(!ggml_backend_tensor_memmove_regions(batch, 2));
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    CHECK(actual == original);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::cout << "PASS: " << ggml_backend_buft_name(buft) << " memmove regions\n";
    return 0;
}

int main() {
    ggml_backend_load_all();
    CHECK(test_buffer(ggml_backend_cpu_buffer_type()) == 0);
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg && std::string(ggml_backend_reg_name(reg)).find("Vulkan") != std::string::npos) {
        CHECK(test_buffer(ggml_backend_dev_buffer_type(dev)) == 0);
    } else {
        std::cout << "SKIP: Vulkan portion (CPU tests ran)\n";
    }
    return 0;
}
