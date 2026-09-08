// test-vulkan-moe-threshold.cpp
// Hermetic regression test verifying Vulkan MoE count_experts reset behavior
// across the 18/19 threshold (mul_mat_vec_id_hybrid_max_cols = 18).
// For supported F32/Q8_1 B, N <= 18 uses singleton routing (mode 3).
// Larger N, or staged F16/BF16 B (e.g. coopmat2), uses the generic route and
// requires mode 0 reset. In particular, small N must not request an F16-B
// singleton kernel: that kernel family only accepts F32/Q8_1 B.
// Tests sequence 18 -> 19 -> 20 -> 18 -> 19 and repeated executions with different
// expert id distributions to ensure route counters / dispatch offsets always start from 0.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " #cond " at line " << __LINE__ << "\n"; \
        return 1; \
    } \
} while (0)

namespace {

struct test_fixture {
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    ~test_fixture() {
        if (backend_gpu) {
            ggml_backend_free(backend_gpu);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

static bool run_moe_step(
        test_fixture & fix,
        ggml_type type_a,
        int n_mats,
        int n_used,
        int64_t m,
        int64_t n,
        int64_t k,
        const std::vector<int32_t> & ids_data,
        const std::vector<float> & a_data,
        const std::vector<float> & b_data,
        std::vector<float> & out_gpu,
        std::vector<float> & out_cpu) {

    const size_t ctx_size = 64 * 1024 * 1024;

    // 1. Compute on GPU
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }

        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n);
        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.backend_gpu);
        if (!buf) {
            ggml_free(ctx);
            return false;
        }

        // Set inputs
        if (type_a == GGML_TYPE_F32) {
            ggml_backend_tensor_set(as, a_data.data(), 0, a_data.size() * sizeof(float));
        } else if (type_a == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> a_f16(a_data.size());
            ggml_fp32_to_fp16_row(a_data.data(), a_f16.data(), a_data.size());
            ggml_backend_tensor_set(as, a_f16.data(), 0, a_f16.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

        ggml_backend_graph_compute(fix.backend_gpu, gf);

        out_gpu.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, out_gpu.data(), 0, out_gpu.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // 2. Compute on CPU
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }

        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n);
        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.backend_cpu);
        if (!buf) {
            ggml_free(ctx);
            return false;
        }

        if (type_a == GGML_TYPE_F32) {
            ggml_backend_tensor_set(as, a_data.data(), 0, a_data.size() * sizeof(float));
        } else if (type_a == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> a_f16(a_data.size());
            ggml_fp32_to_fp16_row(a_data.data(), a_f16.data(), a_data.size());
            ggml_backend_tensor_set(as, a_f16.data(), 0, a_f16.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

        ggml_backend_graph_compute(fix.backend_cpu, gf);

        out_cpu.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, out_cpu.data(), 0, out_cpu.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    return true;
}

// Three consumers share one expert-ID tensor, but the middle consumer stages
// non-contiguous B. Without coopmat2 this switches singleton -> generic ->
// singleton. The route cache must include that mode, not just IDs/grouped_max.
static bool run_mixed_layout_graph(ggml_backend_t backend, int n,
        const std::vector<float> & a_data, const std::vector<float> & b_data,
        const std::vector<int32_t> & ids_data, std::vector<std::vector<float>> & outputs) {
    constexpr int k = 64, m = 32, n_mats = 8, n_used = 2, stride = k + 4;
    ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    auto * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_mats);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
    auto * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n);
    auto * padded = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, stride * n_used * n);
    auto * strided = ggml_view_3d(ctx, padded, k, n_used, n,
            stride * sizeof(float), stride * n_used * sizeof(float), 0);
    ggml_tensor * results[] = {
        ggml_mul_mat_id(ctx, a, b, ids),
        ggml_mul_mat_id(ctx, a, strided, ids),
        ggml_mul_mat_id(ctx, a, b, ids),
    };
    auto * graph = ggml_new_graph(ctx);
    for (auto * result : results) ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    // Poison padding so a mistaken contiguous read is observable.
    std::vector<float> padded_data(stride * n_used * n, 10000.0f);
    for (int row = 0; row < n_used * n; ++row) {
        std::copy_n(b_data.data() + row * k, k, padded_data.data() + row * stride);
    }
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(padded, padded_data.data(), 0, padded_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
    const auto status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        outputs.resize(3);
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i].resize(ggml_nelements(results[i]));
            ggml_backend_tensor_get(results[i], outputs[i].data(), 0, outputs[i].size() * sizeof(float));
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev_gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev_gpu) {
        std::cout << "SKIP: no GPU device found\n";
        return 0;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev_gpu);
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (!reg_name || std::string(reg_name).find("Vulkan") == std::string::npos) {
        std::cout << "SKIP: GPU device is not Vulkan\n";
        return 0;
    }

    ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(dev_cpu != nullptr);

    test_fixture fix;
    fix.backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
    CHECK(fix.backend_gpu != nullptr);
    fix.backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
    CHECK(fix.backend_cpu != nullptr);

    const int n_mats = 8;
    const int n_used = 2; // top-2 routing
    const int64_t m = 32;
    const int64_t k = 64;

    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> dist_val(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> dist_expert(0, n_mats - 1);

    std::vector<float> a_data(k * m * n_mats);
    for (auto & v : a_data) v = dist_val(rng);

    // Cross the 18 threshold repeatedly, including the small-N generic fallback
    // on coopmat2 devices that stage B as F16.
    const std::vector<int> n_cols_seq = { 18, 19, 20, 18, 19, 18, 21 };

    for (size_t round = 0; round < 3; ++round) {
        std::cout << "--- Testing round " << round << " across 18/19 threshold ---\n";
        for (int n : n_cols_seq) {
            std::vector<int32_t> ids_data(n_used * n);
            for (int col = 0; col < n; ++col) {
                int32_t exp0 = dist_expert(rng);
                int32_t exp1 = (exp0 + 1 + (dist_expert(rng) % (n_mats - 1))) % n_mats;
                ids_data[col * n_used + 0] = exp0;
                ids_data[col * n_used + 1] = exp1;
            }

            std::vector<float> b_data(k * n_used * n);
            for (auto & v : b_data) v = dist_val(rng);

            std::vector<float> out_gpu;
            std::vector<float> out_cpu;
            CHECK(run_moe_step(fix, GGML_TYPE_F32, n_mats, n_used, m, n, k, ids_data, a_data, b_data, out_gpu, out_cpu));

            CHECK(out_gpu.size() == out_cpu.size());
            double sum_diff2 = 0.0;
            double sum_ref2 = 0.0;
            float max_diff = 0.0f;
            for (size_t i = 0; i < out_gpu.size(); ++i) {
                float diff = std::abs(out_gpu[i] - out_cpu[i]);
                max_diff = std::max(max_diff, diff);
                sum_diff2 += double(diff) * double(diff);
                sum_ref2  += double(out_cpu[i]) * double(out_cpu[i]);
            }
            double nmse = sum_ref2 > 0.0 ? (sum_diff2 / sum_ref2) : sum_diff2;

            std::cout << "  N=" << n << (n <= 18 ? " (singleton-eligible size) " : " (generic size) ")
                      << " max_abs_diff=" << max_diff << " nmse=" << nmse << "\n";
            CHECK(nmse < 5e-4);

            // Repeat the exact same N on GPU twice in a row to verify prealloc_moe_route cache/reuse
            std::vector<float> out_gpu2;
            std::vector<float> out_cpu2;
            CHECK(run_moe_step(fix, GGML_TYPE_F32, n_mats, n_used, m, n, k, ids_data, a_data, b_data, out_gpu2, out_cpu2));
            float repeat_diff = 0.0f;
            for (size_t i = 0; i < out_gpu.size(); ++i) {
                float diff = std::abs(out_gpu[i] - out_gpu2[i]);
                repeat_diff = std::max(repeat_diff, diff);
            }
            CHECK(repeat_diff == 0.0f);
        }
    }

    for (int n : {18, 19, 18}) {
        std::vector<float> b_data(k * n_used * n);
        for (auto & value : b_data) value = dist_val(rng);
        std::vector<int32_t> ids_data(n_used * n);
        for (int row = 0; row < n; ++row) {
            ids_data[2 * row] = row % n_mats;
            ids_data[2 * row + 1] = (row + 1) % n_mats;
        }
        std::vector<std::vector<float>> actual, reference;
        CHECK(run_mixed_layout_graph(fix.backend_gpu, n, a_data, b_data, ids_data, actual));
        CHECK(run_mixed_layout_graph(fix.backend_cpu, n, a_data, b_data, ids_data, reference));
        CHECK(actual.size() == 3 && reference.size() == 3);
        for (size_t consumer = 0; consumer < 3; ++consumer) {
            CHECK(actual[consumer].size() == reference[consumer].size());
            double error2 = 0.0, reference2 = 0.0;
            for (size_t i = 0; i < actual[consumer].size(); ++i) {
                CHECK(std::isfinite(actual[consumer][i]) && std::isfinite(reference[consumer][i]));
                const double delta = actual[consumer][i] - reference[consumer][i];
                error2 += delta * delta;
                reference2 += double(reference[consumer][i]) * reference[consumer][i];
            }
            CHECK(reference2 > 0.0 && error2 / reference2 < 5e-4);
        }
        std::cout << "  shared IDs, contiguous/strided/contiguous B, N=" << n << " PASS\n";
    }

    std::cout << "PASS: Vulkan MoE 18/19 threshold and mixed-layout route-cache regression test\n";
    return 0;
}
