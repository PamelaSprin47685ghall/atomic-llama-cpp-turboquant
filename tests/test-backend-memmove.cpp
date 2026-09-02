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

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        std::cout << "SKIP: no GPU backend\n";
        return 0;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (!reg_name || std::string(reg_name).find("Vulkan") == std::string::npos) {
        std::cout << "SKIP: GPU backend is not Vulkan\n";
        return 0;
    }

    ggml_init_params params = {};
    params.mem_size = 2 * ggml_tensor_overhead() + 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64); // 256 bytes
    CHECK(ggml_nbytes(tensor) == 256);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
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

    // Vulkan transfer copies require four-byte alignment. Unsupported regions
    // must fail preflight and leave data untouched.
    ggml_backend_tensor_set(tensor, original.data(), 0, original.size());
    ggml_backend_tensor_memmove_region unaligned = {};
    unaligned.tensor = tensor;
    unaligned.src_offset = 4;
    unaligned.dst_offset = 0;
    unaligned.size = 2;
    unaligned.n_copies = 1;
    CHECK(!ggml_backend_tensor_memmove_regions_supported(&unaligned, 1));
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    CHECK(actual == original);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::cout << "PASS: Vulkan backend memmove regions\n";
    return 0;
}
