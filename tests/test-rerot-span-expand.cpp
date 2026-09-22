// test-rerot-span-expand.cpp — test fixture for GGML_OP_REROT_SPAN_EXPAND
//
// Tests R13-A compact span expand bridge:
//  - CPU reference (llama_rerot_spans_expand from src/llama-rerot.h) vs GGML_OP_REROT_SPAN_EXPAND on CPU
//  - GPU/Vulkan path if available (enabled via LLAMA_REROT_GPU_SPAN_EXPAND=1)
//  - Tests fake span set: {key_start=100, count=8, group=1, entry_index=2} with 2 spill rows before and 3 after.

#include "llama.h"
#include "llama-rerot.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while (0)

static std::vector<int32_t> run_span_expand_graph(
        ggml_backend_t backend,
        const std::vector<uint32_t> & spans_vec,  // 4 * n_spans
        const std::vector<uint32_t> & prefix_vec, // n_spans + 1
        const std::vector<int32_t>  & spill_vec,  // 2 * n_spill
        int64_t n_spans,
        int64_t n_entries,
        int64_t n_spill) {

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    // n_spans==0 must use spans->ne[1]==0 so ggml_rerot_span_expand derives
    // op_params n_spans=0 (true pass-through). Do NOT pad to 1 row.
    struct ggml_tensor * t_spans = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, n_spans);
    struct ggml_tensor * t_prefix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_spans + 1);
    struct ggml_tensor * t_spill = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, std::max((int64_t)1, n_spill));

    struct ggml_tensor * t_out = ggml_rerot_span_expand(ctx, t_spans, t_prefix, t_spill, n_spans, n_entries);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    CHECK(ggml_gallocr_alloc_graph(galloc, gf));

    // Upload data
    if (n_spans > 0) {
        ggml_backend_tensor_set(t_spans, spans_vec.data(), 0, spans_vec.size() * sizeof(uint32_t));
    }
    ggml_backend_tensor_set(t_prefix, prefix_vec.data(), 0, prefix_vec.size() * sizeof(uint32_t));
    if (n_spill > 0) {
        ggml_backend_tensor_set(t_spill, spill_vec.data(), 0, spill_vec.size() * sizeof(int32_t));
    }

    CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> result(n_entries * 2);
    ggml_backend_tensor_get(t_out, result.data(), 0, result.size() * sizeof(int32_t));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return result;
}

int main() {
    ggml_backend_load_all();

    // 1. Construct the test fixture described in the prompt:
    // "builds a fake span set {key_start=100, count=8, group=1, entry_index=2} with 2 spill rows before and 3 after"
    // Total live rows: 2 (before) + 8 (span) + 3 (after) = 13 entries.
    // Spill rows: 2 + 3 = 5 rows.
    llama_rerot_attn_layout layout;
    layout.n_queries = 1;
    layout.groups = {{0, 0}, {0, 1}};
    layout.query_offsets = {0, 13};

    // 2 spill rows before
    layout.entries.push_back({10, 0});
    layout.entries.push_back({11, 0});
    // 8 span placeholders (entry_index 2..9)
    for (uint32_t k = 0; k < 8; ++k) {
        layout.entries.push_back({0, 0});
    }
    // 3 spill rows after (entry_index 10..12)
    layout.entries.push_back({20, 0});
    layout.entries.push_back({21, 0});
    layout.entries.push_back({22, 0});

    layout.spans.push_back({100, 8, 1, 2});

    // Coverage check
    auto cov = llama_rerot_spans_coverage(layout);
    CHECK(cov.total_rows == 13);
    CHECK(cov.span_rows == 8);
    CHECK(cov.spill_rows == 5);

    // CPU authoritative reference
    std::vector<llama_rerot_attn_entry> ref_entries = llama_rerot_spans_expand(layout);
    CHECK(ref_entries.size() == 13);

    // Verify ref contents
    CHECK(ref_entries[0].key_index == 10 && ref_entries[0].group_index == 0);
    CHECK(ref_entries[1].key_index == 11 && ref_entries[1].group_index == 0);
    for (uint32_t k = 0; k < 8; ++k) {
        CHECK(ref_entries[2 + k].key_index == 100 + k && ref_entries[2 + k].group_index == 1);
    }
    CHECK(ref_entries[10].key_index == 20 && ref_entries[10].group_index == 0);
    CHECK(ref_entries[11].key_index == 21 && ref_entries[11].group_index == 0);
    CHECK(ref_entries[12].key_index == 22 && ref_entries[12].group_index == 0);

    // Prepare tensors for GGML_OP_REROT_SPAN_EXPAND
    // spans: uvec4 {key_start, count, group_index, entry_index}
    std::vector<uint32_t> spans_buf = { 100, 8, 1, 2 };
    // prefix: prefix[0] = 0, prefix[1] = 8
    std::vector<uint32_t> prefix_buf = { 0, 8 };
    // spill: 5 rows (in entry order)
    std::vector<int32_t> spill_buf = {
        10, 0,
        11, 0,
        20, 0,
        21, 0,
        22, 0
    };

    // Run on CPU backend
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    CHECK(cpu_backend != nullptr);

    std::vector<int32_t> cpu_out = run_span_expand_graph(
        cpu_backend, spans_buf, prefix_buf, spill_buf, 1, 13, 5);

    CHECK(cpu_out.size() == 26);
    for (size_t i = 0; i < 13; ++i) {
        CHECK(cpu_out[2 * i + 0] == (int32_t) ref_entries[i].key_index);
        CHECK(cpu_out[2 * i + 1] == (int32_t) ref_entries[i].group_index);
    }
    std::printf("test-rerot-span-expand: CPU op matches llama_rerot_spans_expand reference bit-identically.\n");

    // Test n_spans == 0 pass-through on CPU
    {
        std::vector<uint32_t> empty_spans = {0, 0, 0, 0};
        std::vector<uint32_t> empty_prefix = {0};
        std::vector<int32_t> pass_spill = { 1, 0, 2, 0, 3, 0 };
        std::vector<int32_t> pass_out = run_span_expand_graph(
            cpu_backend, empty_spans, empty_prefix, pass_spill, 0, 3, 3);
        CHECK(pass_out.size() == 6);
        for (size_t i = 0; i < 6; ++i) {
            CHECK(pass_out[i] == pass_spill[i]);
        }
        std::printf("test-rerot-span-expand: CPU n_spans==0 pass-through verified.\n");
    }

    // Run on Vulkan backend if GPU available and LLAMA_REROT_GPU_SPAN_EXPAND=1
    setenv("LLAMA_REROT_GPU_SPAN_EXPAND", "1", 1);
    ggml_backend_t vk_backend = nullptr;
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
        vk_backend = ggml_backend_dev_init(gpu_dev, nullptr);
    }
#ifdef GGML_USE_VULKAN
    if (!vk_backend && ggml_backend_vk_get_device_count() > 0) {
        vk_backend = ggml_backend_vk_init(0);
    }
#endif

    if (vk_backend) {
        std::printf("test-rerot-span-expand: GPU backend available, running Vulkan op...\n");
        std::vector<int32_t> vk_out = run_span_expand_graph(
            vk_backend, spans_buf, prefix_buf, spill_buf, 1, 13, 5);

        CHECK(vk_out.size() == 26);
        for (size_t i = 0; i < 13; ++i) {
            CHECK(vk_out[2 * i + 0] == (int32_t) ref_entries[i].key_index);
            CHECK(vk_out[2 * i + 1] == (int32_t) ref_entries[i].group_index);
        }
        std::printf("test-rerot-span-expand: Vulkan GPU op matches reference bit-identically.\n");

        // Test n_spans == 0 pass-through on Vulkan
        {
            std::vector<uint32_t> empty_spans = {0, 0, 0, 0};
            std::vector<uint32_t> empty_prefix = {0};
            std::vector<int32_t> pass_spill = { 1, 0, 2, 0, 3, 0 };
            std::vector<int32_t> pass_out = run_span_expand_graph(
                vk_backend, empty_spans, empty_prefix, pass_spill, 0, 3, 3);
            CHECK(pass_out.size() == 6);
            for (size_t i = 0; i < 6; ++i) {
                CHECK(pass_out[i] == pass_spill[i]);
            }
            std::printf("test-rerot-span-expand: Vulkan n_spans==0 pass-through verified.\n");
        }

        ggml_backend_free(vk_backend);
    } else {
        std::printf("test-rerot-span-expand: GPU device not available, skipped Vulkan test.\n");
    }

    ggml_backend_free(cpu_backend);
    std::puts("test-rerot-span-expand: all tests OK");
    return 0;
}
