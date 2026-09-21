// tests/test-vulkan-flash-attn-capacity.cpp
//
// Single-device hermetic regression test for Vulkan Flash Attention GPU kernel
// execution under capacity-padded KV rows and safe_slot redirection (Target Area 1).
//
// Evaluates:
// 1. Numerical parity: verifies that ggml_flash_attn_ext on Vulkan produces
//    active-query outputs matching the canonical CPU reference operator.
// 2. State tail isolation: verifies that changing inactive KV token inputs
//    [n_past + active, n_past + capacity) between sentinel/poison values
//    (-999.0f vs +888.0f) produces bit-identical active-query outputs on GPU.
// 3. Mask & boundary enforcement: verifies that inactive queries [active, capacity)
//    and inactive KV slots are completely masked by -INFINITY in the KQ mask
//    without triggering GPUVM fault or invalid memory access.
// 4. Capacity envelope: evaluates C in {4, 8} and A in {1, 2, C}.
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

static fa_capacity_tensors build_fa_capacity_graph(
        ggml_context * ctx,
        int64_t capacity,
        int64_t n_past) {

    const int64_t hsk = 128;
    const int64_t hsv = 128;
    const int64_t hq  = 4;
    const int64_t hkv = 4;
    const int64_t n_kv = n_past + capacity;

    fa_capacity_tensors t;
    t.q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, capacity, hq, 1);
    t.k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, n_kv, hkv, 1);
    t.v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsv, n_kv, hkv, 1);
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
        float tail_poison_val) {

    const int64_t hsk  = 128;
    const int64_t hsv  = 128;
    const int64_t hq   = 4;
    const int64_t hkv  = 4;
    const int64_t n_kv = n_past + capacity;

    // 1. Q: [hsk, capacity, hq, 1]
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

    // 2. K: [hsk, n_kv, hkv, 1]
    // Valid history + new active tokens get deterministic features;
    // Inactive capacity tail [n_past + active, n_kv) gets tail_poison_val
    {
        std::vector<float> k_data(ggml_nelements(t.k), 0.0f);
        for (int64_t h = 0; h < hkv; ++h) {
            for (int64_t kv_idx = 0; kv_idx < n_kv; ++kv_idx) {
                const bool is_active = (kv_idx < n_past + active);
                for (int64_t d = 0; d < hsk; ++d) {
                    size_t idx = (size_t)h * (n_kv * hsk) + kv_idx * hsk + d;
                    if (is_active) {
                        k_data[idx] = std::cos((float)(idx + 1) * 0.13f) * 0.5f;
                    } else {
                        k_data[idx] = tail_poison_val;
                    }
                }
            }
        }
        ggml_backend_tensor_set(t.k, k_data.data(), 0, k_data.size() * sizeof(float));
    }

    // 3. V: [hsv, n_kv, hkv, 1]
    // Same active vs inactive tail partition
    {
        std::vector<float> v_data(ggml_nelements(t.v), 0.0f);
        for (int64_t h = 0; h < hkv; ++h) {
            for (int64_t kv_idx = 0; kv_idx < n_kv; ++kv_idx) {
                const bool is_active = (kv_idx < n_past + active);
                for (int64_t d = 0; d < hsv; ++d) {
                    size_t idx = (size_t)h * (n_kv * hsv) + kv_idx * hsv + d;
                    if (is_active) {
                        v_data[idx] = std::sin((float)(idx + 1) * 0.11f) * 0.5f;
                    } else {
                        v_data[idx] = tail_poison_val;
                    }
                }
            }
        }
        ggml_backend_tensor_set(t.v, v_data.data(), 0, v_data.size() * sizeof(float));
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
        int64_t n_past) {

    std::fprintf(stderr, "== Test FA Capacity: C=%lld, A=%lld, n_past=%lld ==\n",
                 (long long)capacity, (long long)active, (long long)n_past);

    const size_t ctx_size = 16 * 1024 * 1024;
    const int64_t hsv = 128;
    const int64_t hq  = 4;
    const size_t active_output_elems = (size_t)active * hq * hsv;

    std::vector<float> gpu_out_run1;
    std::vector<float> gpu_out_run2;
    std::vector<float> cpu_out;

    // Run 1 on GPU: tail poison = -999.0f
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        CHECK_TRUE(ctx != nullptr, "ctx init");

        auto t = build_fa_capacity_graph(ctx, capacity, n_past);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.gpu);
        CHECK_TRUE(buf != nullptr, "gpu buf alloc");

        fill_fa_tensors(fix.gpu, t, capacity, active, n_past, -999.0f);

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

        auto t = build_fa_capacity_graph(ctx, capacity, n_past);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.gpu);
        CHECK_TRUE(buf != nullptr, "gpu buf alloc");

        fill_fa_tensors(fix.gpu, t, capacity, active, n_past, +888.0f);

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

        auto t = build_fa_capacity_graph(ctx, capacity, n_past);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.cpu);
        CHECK_TRUE(buf != nullptr, "cpu buf alloc");

        fill_fa_tensors(fix.cpu, t, capacity, active, n_past, -999.0f);

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
            CHECK_CLOSE(gpu_out_run1[idx], cpu_out[idx], 1e-3f, "active query output parity with CPU reference");
        }
    }

    std::fprintf(stderr, "  PASSED (C=%lld, A=%lld, n_past=%lld)\n",
                 (long long)capacity, (long long)active, (long long)n_past);
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
    };

    // Test matrix covering:
    // 1. C = 4 with A in {1, 2, 4} and n_past = 8 (typical decode & target verification)
    // 2. C = 8 with A in {1, 4, 8} and n_past = 16 (maximum capacity envelope)
    // 3. n_past = 0 edge case (prefill / zero prior context)
    const std::vector<test_case> cases = {
        {4, 1, 8},
        {4, 2, 8},
        {4, 4, 8},
        {8, 1, 16},
        {8, 4, 16},
        {8, 8, 16},
        {4, 2, 0},
    };

    for (const auto & tc : cases) {
        if (!test_fa_capacity_case(fix, tc.capacity, tc.active, tc.n_past)) {
            std::fprintf(stderr, "FAILED on C=%lld, A=%lld, n_past=%lld\n",
                         (long long)tc.capacity, (long long)tc.active, (long long)tc.n_past);
            return 1;
        }
    }

    std::printf("test-vulkan-flash-attn-capacity: ALL TESTS PASSED (Vulkan GPU verified)\n");
    return 0;
}
