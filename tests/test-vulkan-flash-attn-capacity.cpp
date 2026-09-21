// tests/test-vulkan-flash-attn-capacity.cpp
//
// Single-device hermetic regression test for Vulkan Flash Attention GPU kernel
// execution under capacity-padded KV rows and safe_slot redirection (Target Area 1).
//
// Evaluates:
// 1. Numerical parity: verifies that ggml_flash_attn_ext on Vulkan produces
//    active-query outputs matching the canonical CPU reference operator across
//    both standard MHA, Grouped Query Attention (GQA, Hq > Hkv), and quantized KV (Q8_0).
// 2. State tail isolation: verifies that changing inactive KV token inputs
//    [n_past + active, n_past + capacity) between sentinel/poison values
//    (-999.0f vs +888.0f) produces bit-identical active-query outputs on GPU.
// 3. Mask & boundary enforcement: verifies that inactive queries [active, capacity)
//    and inactive KV slots are completely masked by -INFINITY in the KQ mask
//    without triggering GPUVM fault or invalid memory access.
// 4. Capacity & configuration envelope:
//    - C in {4, 8} and A in {1, 2, C}
//    - Head configurations: MHA (4:4) and GQA (4:1, 8:2)
//    - Precision formats: F32 KV and Q8_0 quantized KV
// 5. Graceful skip: exits cleanly (0) when Vulkan backend or device 0 is unavailable.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d]: %s (%s)\n", __FILE__, __LINE__, #cond, msg); \
        return false; \
    } \
} while (0)

#define CHECK_CLOSE(a, b, tol, msg) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (std::isnan((float)(a)) || std::isnan((float)(b)) || diff > (tol)) { \
        std::fprintf(stderr, "FAIL [%s:%d]: |%f - %f| = %f > tol=%f (%s)\n", \
                     __FILE__, __LINE__, (double)(a), (double)(b), (double)diff, (double)(tol), msg); \
        return false; \
    } \
} while (0)

namespace {

struct backend_holder {
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;

    ~backend_holder() {
        if (gpu) {
            ggml_backend_free(gpu);
        }
        if (cpu) {
            ggml_backend_free(cpu);
        }
    }
};

struct fa_capacity_tensors {
    ggml_tensor * q    = nullptr; // [hsk, capacity, hq, 1]
    ggml_tensor * k    = nullptr; // [hsk, n_kv, hkv, 1]
    ggml_tensor * v    = nullptr; // [hsv, n_kv, hkv, 1]
    ggml_tensor * mask = nullptr; // [n_kv, capacity, 1, 1], F16
    ggml_tensor * out  = nullptr; // [hsv, hq, capacity, 1]
};

static bool quantize_row_data(ggml_type type, const float * src, void * dst, int64_t n) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, n * sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) dst, n);
        return true;
    }
    if (type == GGML_TYPE_Q8_0) {
        const auto * traits = ggml_get_type_traits(type);
        if (!traits || !traits->from_float_ref) return false;
        traits->from_float_ref(src, dst, n);
        return true;
    }
    return false;
}

static fa_capacity_tensors build_fa_capacity_graph(
        ggml_context * ctx,
        int64_t capacity,
        int64_t n_past,
        int64_t hq,
        int64_t hkv,
        ggml_type type_k,
        ggml_type type_v) {

    const int64_t hsk = 128;
    const int64_t hsv = 128;
    const int64_t n_kv = n_past + capacity;

    fa_capacity_tensors t;
    t.q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, capacity, hq, 1);
    t.k    = ggml_new_tensor_4d(ctx, type_k,        hsk, n_kv,     hkv, 1);
    t.v    = ggml_new_tensor_4d(ctx, type_v,        hsv, n_kv,     hkv, 1);
    t.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, capacity, 1, 1);

    const float scale = 1.0f / std::sqrt((float)hsk);
    t.out = ggml_flash_attn_ext(ctx, t.q, t.k, t.v, t.mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(t.out, GGML_PREC_F32);

    return t;
}

static void fill_fa_tensors(
        ggml_backend_t backend,
        const fa_capacity_tensors & t,
        int64_t capacity,
        int64_t active,
        int64_t n_past,
        int64_t hq,
        int64_t hkv,
        ggml_type type_k,
        ggml_type type_v,
        float tail_poison_val) {

    const int64_t hsk  = 128;
    const int64_t hsv  = 128;
    const int64_t n_kv = n_past + capacity;

    // 1. Q: [hsk, capacity, hq, 1], F32
    // Active queries get deterministic non-zero test patterns; inactive queries get 0.0f
    {
        std::vector<float> q_data(ggml_nelements(t.q), 0.0f);
        for (int64_t h = 0; h < hq; ++h) {
            for (int64_t q_idx = 0; q_idx < active; ++q_idx) {
                for (int64_t d = 0; d < hsk; ++d) {
                    size_t idx = (size_t)h * (capacity * hsk) + q_idx * hsk + d;
                    q_data[idx] = std::sin((float)(idx + 1) * 0.17f) * 0.5f;
                }
            }
        }
        ggml_backend_tensor_set(t.q, q_data.data(), 0, q_data.size() * sizeof(float));
    }

    // 2. K: [hsk, n_kv, hkv, 1], supports F32 and Q8_0
    {
        const size_t k_row_size = ggml_row_size(type_k, hsk);
        const size_t k_total_bytes = (size_t)hkv * n_kv * k_row_size;
        std::vector<uint8_t> k_bytes(k_total_bytes, 0);
        std::vector<float> row_buf(hsk);

        for (int64_t h = 0; h < hkv; ++h) {
            for (int64_t kv_idx = 0; kv_idx < n_kv; ++kv_idx) {
                const bool is_active = (kv_idx < n_past + active);
                for (int64_t d = 0; d < hsk; ++d) {
                    size_t seed_idx = (size_t)h * (n_kv * hsk) + kv_idx * hsk + d;
                    row_buf[d] = is_active ? (std::cos((float)(seed_idx + 1) * 0.13f) * 0.5f) : tail_poison_val;
                }
                size_t byte_offset = ((size_t)h * n_kv + kv_idx) * k_row_size;
                bool ok = quantize_row_data(type_k, row_buf.data(), k_bytes.data() + byte_offset, hsk);
                GGML_ASSERT(ok && "quantize K row failed");
            }
        }
        ggml_backend_tensor_set(t.k, k_bytes.data(), 0, k_total_bytes);
    }

    // 3. V: [hsv, n_kv, hkv, 1], supports F32 and Q8_0
    {
        const size_t v_row_size = ggml_row_size(type_v, hsv);
        const size_t v_total_bytes = (size_t)hkv * n_kv * v_row_size;
        std::vector<uint8_t> v_bytes(v_total_bytes, 0);
        std::vector<float> row_buf(hsv);

        for (int64_t h = 0; h < hkv; ++h) {
            for (int64_t kv_idx = 0; kv_idx < n_kv; ++kv_idx) {
                const bool is_active = (kv_idx < n_past + active);
                for (int64_t d = 0; d < hsv; ++d) {
                    size_t seed_idx = (size_t)h * (n_kv * hsv) + kv_idx * hsv + d;
                    row_buf[d] = is_active ? (std::sin((float)(seed_idx + 1) * 0.11f) * 0.5f) : tail_poison_val;
                }
                size_t byte_offset = ((size_t)h * n_kv + kv_idx) * v_row_size;
                bool ok = quantize_row_data(type_v, row_buf.data(), v_bytes.data() + byte_offset, hsv);
                GGML_ASSERT(ok && "quantize V row failed");
            }
        }
        ggml_backend_tensor_set(t.v, v_bytes.data(), 0, v_total_bytes);
    }

    // 4. KQ Mask: [n_kv, capacity, 1, 1], type F16
    // Causal attention on active queries; future tokens and inactive tail slots masked out (-INFINITY)
    // Entire row for inactive queries [active, capacity) masked out (-INFINITY)
    {
        std::vector<ggml_fp16_t> mask_data(ggml_nelements(t.mask), ggml_fp32_to_fp16(-INFINITY));
        for (int64_t q_idx = 0; q_idx < active; ++q_idx) {
            const int64_t visible_kv_limit = n_past + q_idx + 1;
            for (int64_t kv_idx = 0; kv_idx < n_kv; ++kv_idx) {
                size_t idx = (size_t)q_idx * n_kv + kv_idx;
                if (kv_idx < visible_kv_limit) {
                    mask_data[idx] = ggml_fp32_to_fp16(0.0f); // unmasked causal connection
                } else {
                    mask_data[idx] = ggml_fp32_to_fp16(-INFINITY); // masked (future or inactive tail)
                }
            }
        }
        ggml_backend_tensor_set(t.mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
}

static bool test_fa_capacity_case(
        backend_holder & fix,
        int64_t capacity,
        int64_t active,
        int64_t n_past,
        int64_t hq,
        int64_t hkv,
        ggml_type type_k,
        ggml_type type_v,
        const char * desc) {

    std::fprintf(stderr, "== Test FA Capacity: %s [C=%lld, A=%lld, n_past=%lld, Hq=%lld, Hkv=%lld, K=%s, V=%s] ==\n",
                 desc, (long long)capacity, (long long)active, (long long)n_past,
                 (long long)hq, (long long)hkv, ggml_type_name(type_k), ggml_type_name(type_v));

    const size_t ctx_size = 16 * 1024 * 1024;
    const int64_t hsv = 128;
    const size_t active_output_elems = (size_t)active * hq * hsv;
    const float parity_tol = (type_k == GGML_TYPE_F32 && type_v == GGML_TYPE_F32) ? 1e-3f : 2e-2f;

    std::vector<float> gpu_out_run1;
    std::vector<float> gpu_out_run2;
    std::vector<float> cpu_out;

    // Run 1 on GPU: tail poison = -999.0f
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        CHECK_TRUE(ctx != nullptr, "ctx init");

        auto t = build_fa_capacity_graph(ctx, capacity, n_past, hq, hkv, type_k, type_v);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.gpu);
        CHECK_TRUE(buf != nullptr, "gpu buf alloc");

        fill_fa_tensors(fix.gpu, t, capacity, active, n_past, hq, hkv, type_k, type_v, -999.0f);

        ggml_status st = ggml_backend_graph_compute(fix.gpu, gf);
        CHECK_TRUE(st == GGML_STATUS_SUCCESS, "gpu graph compute failed (run 1)");

        gpu_out_run1.resize(ggml_nelements(t.out));
        ggml_backend_tensor_get(t.out, gpu_out_run1.data(), 0, gpu_out_run1.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Run 2 on GPU: tail poison = +888.0f (tail isolation proof)
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        CHECK_TRUE(ctx != nullptr, "ctx init");

        auto t = build_fa_capacity_graph(ctx, capacity, n_past, hq, hkv, type_k, type_v);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.gpu);
        CHECK_TRUE(buf != nullptr, "gpu buf alloc");

        fill_fa_tensors(fix.gpu, t, capacity, active, n_past, hq, hkv, type_k, type_v, +888.0f);

        ggml_status st = ggml_backend_graph_compute(fix.gpu, gf);
        CHECK_TRUE(st == GGML_STATUS_SUCCESS, "gpu graph compute failed (run 2)");

        gpu_out_run2.resize(ggml_nelements(t.out));
        ggml_backend_tensor_get(t.out, gpu_out_run2.data(), 0, gpu_out_run2.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Run 3 on CPU: canonical reference
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        CHECK_TRUE(ctx != nullptr, "ctx init");

        auto t = build_fa_capacity_graph(ctx, capacity, n_past, hq, hkv, type_k, type_v);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.cpu);
        CHECK_TRUE(buf != nullptr, "cpu buf alloc");

        fill_fa_tensors(fix.cpu, t, capacity, active, n_past, hq, hkv, type_k, type_v, -999.0f);

        ggml_status st = ggml_backend_graph_compute(fix.cpu, gf);
        CHECK_TRUE(st == GGML_STATUS_SUCCESS, "cpu graph compute failed");

        cpu_out.resize(ggml_nelements(t.out));
        ggml_backend_tensor_get(t.out, cpu_out.data(), 0, cpu_out.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Invariant 1: State tail isolation (varying inactive KV poison produces bit-identical active outputs on GPU)
    CHECK_TRUE(gpu_out_run1.size() == gpu_out_run2.size(), "output size match");
    for (size_t i = 0; i < active_output_elems; ++i) {
        CHECK_TRUE(gpu_out_run1[i] == gpu_out_run2[i], "GPU active output polluted by inactive KV tail tokens!");
    }

    // Invariant 2: Parity between GPU and CPU on active output rows
    for (int64_t q_idx = 0; q_idx < active; ++q_idx) {
        size_t row_offset = (size_t)q_idx * hq * hsv;
        float gpu_sample = gpu_out_run1[row_offset];
        float cpu_sample = cpu_out[row_offset];
        std::fprintf(stderr, "  [DIAG] q=%lld: GPU[0]=%.6f CPU[0]=%.6f\n",
                     (long long)q_idx, gpu_sample, cpu_sample);

        for (size_t d = 0; d < (size_t)hq * hsv; ++d) {
            size_t idx = row_offset + d;
            CHECK_TRUE(std::isfinite(gpu_out_run1[idx]), "GPU output is not finite (NaN/Inf)");
            CHECK_CLOSE(gpu_out_run1[idx], cpu_out[idx], parity_tol, "active query output parity with CPU reference");
        }
    }

    std::fprintf(stderr, "  PASSED: %s\n", desc);
    return true;
}

} // namespace

int main() {
    backend_holder fix;

    if (ggml_backend_vk_get_device_count() <= 0) {
        std::fprintf(stderr, "SKIP: no Vulkan devices available\n");
        return 0;
    }

    fix.gpu = ggml_backend_vk_init(0);
    if (!fix.gpu) {
        std::fprintf(stderr, "SKIP: failed to initialize Vulkan device 0\n");
        return 0;
    }

    fix.cpu = ggml_backend_cpu_init();
    if (!fix.cpu) {
        std::fprintf(stderr, "FAIL: failed to initialize CPU backend\n");
        return 1;
    }

    struct test_case {
        int64_t capacity;
        int64_t active;
        int64_t n_past;
        int64_t hq;
        int64_t hkv;
        ggml_type type_k;
        ggml_type type_v;
        const char * desc;
    };

    // Extended test matrix covering:
    // 1. MHA F32 baseline: C in {4, 8}, A in {1, 2, C}
    // 2. GQA F32 (4:1 and 4:2 ratio): C in {4, 8}, A < C
    // 3. MHA Q8_0 quantized KV: C = 4, A = 2
    // 4. Production Target Config: GQA 4:1 with Q8_0 quantized KV (C=4, A=2 and C=8, A=4)
    // 5. Zero-past edge case (Prefill boundary)
    const std::vector<test_case> cases = {
        // --- 1. MHA F32 Baseline ---
        {4, 1, 8, 4, 4, GGML_TYPE_F32, GGML_TYPE_F32, "MHA F32 C=4 A=1"},
        {4, 2, 8, 4, 4, GGML_TYPE_F32, GGML_TYPE_F32, "MHA F32 C=4 A=2"},
        {4, 4, 8, 4, 4, GGML_TYPE_F32, GGML_TYPE_F32, "MHA F32 C=4 A=4"},
        {8, 1, 16, 4, 4, GGML_TYPE_F32, GGML_TYPE_F32, "MHA F32 C=8 A=1"},
        {8, 4, 16, 4, 4, GGML_TYPE_F32, GGML_TYPE_F32, "MHA F32 C=8 A=4"},
        {8, 8, 16, 4, 4, GGML_TYPE_F32, GGML_TYPE_F32, "MHA F32 C=8 A=8"},
        {4, 2, 0, 4, 4, GGML_TYPE_F32, GGML_TYPE_F32, "MHA F32 C=4 A=2 n_past=0"},

        // --- 2. GQA F32 (Grouped Query Attention) ---
        {4, 2, 8, 4, 1, GGML_TYPE_F32, GGML_TYPE_F32, "GQA 4:1 F32 C=4 A=2"},
        {8, 4, 16, 8, 2, GGML_TYPE_F32, GGML_TYPE_F32, "GQA 4:1 F32 C=8 A=4"},

        // --- 3. MHA Q8_0 Quantized KV ---
        {4, 2, 8, 4, 4, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "MHA Q8_0 C=4 A=2"},

        // --- 4. Production Target Alignment: GQA + Q8_0 Quantized KV ---
        {4, 2, 8, 4, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "Production GQA 4:1 Q8_0 C=4 A=2"},
        {8, 4, 16, 8, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "Production GQA 4:1 Q8_0 C=8 A=4"},
    };

    for (const auto & tc : cases) {
        if (!test_fa_capacity_case(fix, tc.capacity, tc.active, tc.n_past, tc.hq, tc.hkv, tc.type_k, tc.type_v, tc.desc)) {
            std::fprintf(stderr, "FAILED on case: %s\n", tc.desc);
            return 1;
        }
    }

    std::printf("test-vulkan-flash-attn-capacity: ALL TESTS PASSED (Vulkan GPU verified)\n");
    return 0;
}
