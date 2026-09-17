#include "../src/llama-model-loader.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <cstdio>
#include <memory>
#include <vector>

static void check_lazy_rows(bool mmap_model, bool bind_lazy, bool mixed, llama_lazy_mode mode = LLAMA_LAZY_MODE_ON) {
    const char *       name  = "per_layer_token_embd.weight";
    const int64_t      width = 16;
    const int64_t      rows  = 512;
    std::vector<float> expected(width * rows);
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = float(i) * 0.125f - 17.0f;

    ggml_context_ptr source(ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true }));
    auto *           table = ggml_new_tensor_2d(source.get(), GGML_TYPE_F32, width, rows);
    ggml_set_name(table, name);
    table->data = expected.data();
    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_val_str(metadata.get(), "general.architecture", "qwen4exp");
    gguf_add_tensor(metadata.get(), table);
    if (mixed) {
        auto * ordinary = ggml_new_tensor_1d(source.get(), GGML_TYPE_F32, width);
        ggml_set_name(ordinary, "ordinary.weight");
        ordinary->data = expected.data() + width;
        gguf_add_tensor(metadata.get(), ordinary);
    }
    const auto close_file = [](FILE * f) {
        fclose(f);
    };
    std::unique_ptr<FILE, decltype(close_file)> file(std::tmpfile(), close_file);
    GGML_ASSERT(file && gguf_write_to_file_ptr(metadata.get(), file.get(), false));
    rewind(file.get());

    std::vector<std::string> splits;
    llama_model_loader       loader(nullptr, nullptr, nullptr, "", splits, file.get(),
                              mmap_model ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE, false, false, false, nullptr,
                                    nullptr);
    ggml_context_ptr         target(ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true }));
    auto *                   lazy = ggml_new_tensor_2d(target.get(), GGML_TYPE_F32, width, rows);
    ggml_set_name(lazy, name);
    auto * ordinary = mixed ? ggml_new_tensor_1d(target.get(), GGML_TYPE_F32, width) : nullptr;
    if (ordinary)
        ggml_set_name(ordinary, "ordinary.weight");
    loader.lazy.mode = mode;
    GGML_ASSERT(loader.lazy.add(name, lazy, &loader.require_weight(name)));
    loader.init_mappings(false);

    // Match production's host-backed buffer, then let load_all_data perform
    // its final mmap trimming. The lazy-only file used to be unmapped here.
    auto *                               cpu     = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    auto &                               mapping = loader.mappings[0];
    std::vector<ggml_backend_buffer_ptr> lazy_buffers;
    if (bind_lazy)
        loader.init_lazy_tensors(target.get(), cpu, lazy_buffers);
    ggml_backend_buffer_ptr buffer(
        mmap_model ? ggml_backend_dev_buffer_from_host_ptr(cpu, mapping->addr(), mapping->size(), ggml_nbytes(lazy)) :
                     ggml_backend_alloc_ctx_tensors_from_buft(target.get(), ggml_backend_dev_buffer_type(cpu)));
    GGML_ASSERT(buffer || !lazy_buffers.empty());
    llama_buf_map buffers;
    if (buffer)
        buffers.emplace(0, buffer.get());
    GGML_ASSERT(loader.load_all_data(target.get(), buffers, nullptr, nullptr, nullptr));

    // Read rows on distant pages only AFTER loading/retirement, as inference
    // does. Exact values distinguish live mapped payload from stale pointers.
    std::vector<float> actual(width);
    for (int64_t row : { int64_t(0), rows / 2, rows - 1 }) {
        ggml_backend_tensor_get(lazy, actual.data(), row * width * sizeof(float), actual.size() * sizeof(float));
        for (int64_t col = 0; col < width; ++col) {
            GGML_ASSERT(actual[col] == expected[row * width + col]);
        }
    }
    if (ordinary) {
        ggml_backend_tensor_get(ordinary, actual.data(), 0, actual.size() * sizeof(float));
        for (int64_t col = 0; col < width; ++col)
            GGML_ASSERT(actual[col] == expected[width + col]);
    }
    fprintf(stderr, "test-model-lazy-mmap: mmap_model=%d bind_lazy=%d mixed=%d post-load rows match file exactly\n",
            int(mmap_model), int(bind_lazy), int(mixed));
}

int main() {
    if (!llama_mmap::SUPPORTED) {
        fprintf(stderr, "test-model-lazy-mmap: SKIP (mmap unavailable)\n");
        return 0;
    }
    ggml_backend_cpu_reg();
    check_lazy_rows(true, false, false);
    check_lazy_rows(false, false, true);
    check_lazy_rows(false, true, false);
    check_lazy_rows(false, true, true);
    // Direct-reader fallback platforms still consume an actual mapped table.
    check_lazy_rows(true, false, false, LLAMA_LAZY_MODE_DIRECT);
    return 0;
}
