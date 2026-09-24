// tests/test-vulkan-gdn-multistep.cpp
//
// Single-device hermetic regression test for Vulkan multistep variable-length
// GDN segment execution (qwen4_gdn_multistep_{prep,delta,norm}).
//
// Evaluates:
// 1. Numerical parity: compares fused multistep Vulkan output against the
//    canonical CPU reference operator on active rows [0, active).
// 2. State tail isolation: verifies that changing inactive token inputs [active, capacity)
//    produces bit-identical final recurrent states on GPU.
// 3. Step coverage: evaluates active steps A in {1, 2, 4} with capacity C = 4, and A in {1, 2, 4, 8} with C = 8.
// 4. Graceful skip: exits cleanly when Vulkan backend or device 0 is unavailable.

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

static inline void set_op_params_f32(struct ggml_tensor * tensor, uint32_t i, float value) {
    std::memcpy(&tensor->op_params[i], &value, sizeof(float));
}

#define CHECK_CLOSE(a, b, tol, msg) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (std::isnan(a) || std::isnan(b) || diff > (tol)) { \
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

struct gdn_test_tensors {
    ggml_tensor * conv_in     = nullptr; // [channels, C]
    ggml_tensor * conv_w      = nullptr; // [4, channels]
    ggml_tensor * raw_alpha   = nullptr; // [H_v, C]
    ggml_tensor * alpha_bias  = nullptr; // [H_v, 1]
    ggml_tensor * alpha_scale = nullptr; // [H_v, 1]
    ggml_tensor * raw_beta    = nullptr; // [H_v, C]
    ggml_tensor * state_in    = nullptr; // [S_v, S_v * H_v]
    ggml_tensor * cache_dst   = nullptr; // [S_v * H_v, 129]
    ggml_tensor * gamma       = nullptr; // [128, H_v]
    ggml_tensor * raw_z       = nullptr; // [128, H_v, C]
    ggml_tensor * out_w       = nullptr; // [head_values, head_values]

    // Graph outputs
    ggml_tensor * final_out   = nullptr; // node(26)
    ggml_tensor * state_out   = nullptr; // node(16)
};

static gdn_test_tensors build_gdn_27_node_graph(
        ggml_context * ctx,
        int64_t capacity,
        int64_t active_tokens,
        int64_t snapshots = 1,
        int64_t hk = 1,
        int64_t hv = 3,
        const std::vector<uint8_t> & headmap = {}) {

    const int64_t S_v = 128;
    const int64_t channels    = 128 * (2 * hk + hv); // 640
    const int64_t head_values = 128 * hv;             // 384
    const int64_t n_time      = capacity;

    gdn_test_tensors t;

    // External inputs
    t.conv_in     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_time + 3, channels, 1);
    t.conv_w      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    t.raw_alpha   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hv, n_time);
    t.alpha_bias  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hv, 1);
    t.alpha_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hv, 1);
    t.raw_beta    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hv, n_time);
    t.state_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S_v, S_v * hv);
    t.cache_dst   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_values, n_time + 128 * snapshots);
    t.gamma       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, hv);
    t.raw_z       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, hv, n_time);
    t.out_w       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_values, head_values);

    // 0. GGML_OP_SSM_CONV
    ggml_tensor * n0 = ggml_ssm_conv(ctx, t.conv_in, t.conv_w);

    // 1. GGML_OP_UNARY (SILU)
    ggml_tensor * n1 = ggml_silu(ctx, n0);

    // 2. GGML_OP_VIEW (Q_raw)
    ggml_tensor * n2 = ggml_view_3d(ctx, n1, 128, hk, n_time, 128 * sizeof(float), channels * sizeof(float), 0);

    // 3. GGML_OP_L2_NORM (Q_norm)
    ggml_tensor * n3 = ggml_l2_norm(ctx, n2, 1e-5f);
    set_op_params_f32(n3, 0, 1e-5f);

    // 4. GGML_OP_VIEW (K_raw)
    ggml_tensor * n4 = ggml_view_3d(ctx, n1, 128, hk, n_time, 128 * sizeof(float), channels * sizeof(float), 128 * hk * sizeof(float));

    // 5. GGML_OP_L2_NORM (K_norm)
    ggml_tensor * n5 = ggml_l2_norm(ctx, n4, 1e-5f);
    set_op_params_f32(n5, 0, 1e-5f);

    // 6. GGML_OP_VIEW (V_raw)
    ggml_tensor * n6 = ggml_view_3d(ctx, n1, 128, hv, n_time, 128 * sizeof(float), channels * sizeof(float), 256 * hk * sizeof(float));

    // 7. GGML_OP_RESHAPE
    ggml_tensor * n7 = ggml_reshape_3d(ctx, t.raw_alpha, hv, n_time, 1);

    // 8. GGML_OP_ADD
    ggml_tensor * n8 = ggml_add(ctx, n7, t.alpha_bias);

    // 9. GGML_OP_UNARY (SOFTPLUS)
    ggml_tensor * n9 = ggml_softplus(ctx, n8);

    // 10. GGML_OP_MUL
    ggml_tensor * n10 = ggml_mul(ctx, n9, t.alpha_scale);

    // 11. GGML_OP_RESHAPE
    ggml_tensor * n11 = ggml_reshape_4d(ctx, n10, 1, hv, n_time, 1);

    // 12. GGML_OP_RESHAPE
    ggml_tensor * n12 = ggml_reshape_4d(ctx, t.raw_beta, 1, hv, n_time, 1);

    // 13. GGML_OP_UNARY (SIGMOID)
    ggml_tensor * n13 = ggml_sigmoid(ctx, n12);

    // 14. GGML_OP_RESHAPE
    ggml_tensor * n14 = ggml_reshape_4d(ctx, t.state_in, S_v, S_v, hv, 1);

    // 15. GGML_OP_GATED_DELTA_NET (with active_tokens in op_params[2])
    ggml_tensor * n15 = ggml_gated_delta_net_ext(ctx, n3, n5, n6, n11, n13, n14, snapshots, active_tokens);
    if (!headmap.empty()) {
        GGML_ASSERT((int64_t) headmap.size() == hv);
        ggml_tp5_headmap_set(n15, headmap.data(), (int32_t) headmap.size());
    }
    const int64_t written = std::min(active_tokens == 0 ? capacity : active_tokens, snapshots);

    // 16. GGML_OP_VIEW (state_out_view) - offset strictly follows production delta-net-base.cpp: attn_score_elems * sizeof(float)
    ggml_tensor * n16 = ggml_view_4d(ctx, n15, S_v, S_v, hv, written,
                                     S_v * sizeof(float),
                                     S_v * S_v * sizeof(float),
                                     S_v * S_v * hv * sizeof(float),
                                     (size_t) n_time * head_values * sizeof(float));

    // 17. GGML_OP_VIEW (cache_dst_view)
    ggml_tensor * n17 = ggml_view_4d(ctx, t.cache_dst, S_v, S_v, hv, written,
                                     S_v * sizeof(float),
                                     S_v * S_v * sizeof(float),
                                     S_v * S_v * hv * sizeof(float),
                                     0);

    // 18. GGML_OP_CPY
    ggml_tensor * n18 = ggml_cpy(ctx, n16, n17);

    // 19. GGML_OP_VIEW (attn_view)
    ggml_tensor * n19 = ggml_view_4d(ctx, n15, 128, hv, n_time, 1,
                                     128 * sizeof(float),
                                     128 * hv * sizeof(float),
                                     128 * hv * n_time * sizeof(float),
                                     0);

    // 20. GGML_OP_RMS_NORM
    ggml_tensor * n20 = ggml_rms_norm(ctx, n19, 1e-6f);
    set_op_params_f32(n20, 0, 1e-6f);

    // 21. GGML_OP_MUL
    ggml_tensor * n21 = ggml_mul(ctx, n20, t.gamma);

    // 22. GGML_OP_RESHAPE
    ggml_tensor * n22 = ggml_reshape_4d(ctx, t.raw_z, 128, hv, n_time, 1);

    // 23. GGML_OP_UNARY (SIGMOID)
    ggml_tensor * n23 = ggml_sigmoid(ctx, n22);

    // 24. GGML_OP_MUL
    ggml_tensor * n24 = ggml_mul(ctx, n21, n23);

    // 25. GGML_OP_RESHAPE
    ggml_tensor * n25 = ggml_reshape_2d(ctx, n24, head_values, n_time);

    // 26. GGML_OP_MUL_MAT
    ggml_tensor * n26 = ggml_mul_mat(ctx, t.out_w, n25);

    t.final_out = n26;
    t.state_out = n18;

    return t;
}

static void fill_tensor_uniform(ggml_backend_t backend, ggml_tensor * t, float val) {
    std::vector<float> data(ggml_nelements(t), val);
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

static void fill_tensors(
        ggml_backend_t backend,
        const gdn_test_tensors & t,
        int64_t capacity,
        int64_t active,
        float inactive_garbage) {

    const int64_t channels = t.conv_in->ne[1];
    const int64_t hv       = t.raw_alpha->ne[0];

    // Conv weights: simple 4-tap box filter
    fill_tensor_uniform(backend, t.conv_w, 0.25f);

    // Distinct mapped Q/K head vectors make a wrong V->QK map observable.
    {
        const int64_t n_in_time = capacity + 3;
        std::vector<float> data(channels * n_in_time, 0.0f);
        for (int64_t step = 0; step < capacity; ++step) {
            float v = (step < active) ? 1.0f : inactive_garbage;
            for (int64_t c = 0; c < channels; ++c) {
                const float pattern = hv > 3 ? 1.0f + 0.08f * float(((c / 128) * 7 + (c % 128) * 3) % 17 - 8) : 1.0f;
                data[c * n_in_time + (step + 3)] = v * pattern;
            }
        }
        ggml_backend_tensor_set(t.conv_in, data.data(), 0, data.size() * sizeof(float));
    }

    // Alpha / Beta / Gate parameters
    fill_tensor_uniform(backend, t.alpha_bias, 0.0f);
    fill_tensor_uniform(backend, t.alpha_scale, -1.0f); // decay = exp(-softplus)
    {
        std::vector<float> alpha_data(hv * capacity, 0.5f);
        std::vector<float> beta_data(hv * capacity, 1.0f);
        for (int64_t step = active; step < capacity; ++step) {
            for (int64_t h = 0; h < hv; ++h) {
                alpha_data[step * hv + h] = inactive_garbage;
                beta_data[step * hv + h]  = inactive_garbage;
            }
        }
        ggml_backend_tensor_set(t.raw_alpha, alpha_data.data(), 0, alpha_data.size() * sizeof(float));
        ggml_backend_tensor_set(t.raw_beta, beta_data.data(), 0, beta_data.size() * sizeof(float));
    }

    if (hv > 3) {
        std::vector<float> state(ggml_nelements(t.state_in));
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] = 0.01f + 0.002f * float(int((i * 13 + i / (128 * 128) * 7) % 19) - 9);
        }
        ggml_backend_tensor_set(t.state_in, state.data(), 0, state.size() * sizeof(float));
    } else {
        fill_tensor_uniform(backend, t.state_in, 0.01f);
    }
    fill_tensor_uniform(backend, t.cache_dst, 0.0f);

    // Norm and Z gate
    fill_tensor_uniform(backend, t.gamma, 1.0f);
    fill_tensor_uniform(backend, t.raw_z, 0.0f); // sigmoid(0) = 0.5

    // Projection matrix: identity scale
    fill_tensor_uniform(backend, t.out_w, 0.05f);
}

static bool test_gdn_multistep_step(
        backend_holder & fix,
        int64_t capacity,
        int64_t active,
        int64_t snapshots = 1,
        int64_t hk = 1,
        int64_t hv = 3,
        const std::vector<uint8_t> & headmap = {}) {

    std::fprintf(stderr, "== Test GDN Multistep: Capacity C=%lld, Active A=%lld ==\n",
                 (long long)capacity, (long long)active);
    const int64_t effective_active = active == 0 ? capacity : active;

    const size_t ctx_size = 32 * 1024 * 1024;

    std::vector<float> gpu_out_run1;
    std::vector<float> gpu_state_run1;
    std::vector<float> gpu_state_run2;
    std::vector<float> cpu_out;
    std::vector<float> cpu_state;

    // Run 1 on GPU (inactive token garbage = -999.0f)
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        CHECK_TRUE(ctx != nullptr, "ctx init");

        auto t = build_gdn_27_node_graph(ctx, capacity, active, snapshots, hk, hv, headmap);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.state_out);
        ggml_build_forward_expand(gf, t.final_out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.gpu);
        CHECK_TRUE(buf != nullptr, "gpu buf alloc");

        fill_tensors(fix.gpu, t, capacity, effective_active, -999.0f);

        ggml_status st = ggml_backend_graph_compute(fix.gpu, gf);
        CHECK_TRUE(st == GGML_STATUS_SUCCESS, "gpu graph compute failed");

        gpu_out_run1.resize(ggml_nelements(t.final_out));
        gpu_state_run1.resize(ggml_nelements(t.state_out));
        ggml_backend_tensor_get(t.final_out, gpu_out_run1.data(), 0, gpu_out_run1.size() * sizeof(float));
        ggml_backend_tensor_get(t.state_out, gpu_state_run1.data(), 0, gpu_state_run1.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Run 2 on GPU (inactive token garbage = +888.0f) to prove tail isolation
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        CHECK_TRUE(ctx != nullptr, "ctx init");

        auto t = build_gdn_27_node_graph(ctx, capacity, active, snapshots, hk, hv, headmap);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.state_out);
        ggml_build_forward_expand(gf, t.final_out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.gpu);
        CHECK_TRUE(buf != nullptr, "gpu buf alloc");

        fill_tensors(fix.gpu, t, capacity, effective_active, +888.0f);

        ggml_status st = ggml_backend_graph_compute(fix.gpu, gf);
        CHECK_TRUE(st == GGML_STATUS_SUCCESS, "gpu graph compute failed");

        gpu_state_run2.resize(ggml_nelements(t.state_out));
        ggml_backend_tensor_get(t.state_out, gpu_state_run2.data(), 0, gpu_state_run2.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Run 3 on CPU Reference
    {
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        CHECK_TRUE(ctx != nullptr, "ctx init");

        auto t = build_gdn_27_node_graph(ctx, capacity, active, snapshots, hk, hv, headmap);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t.state_out);
        ggml_build_forward_expand(gf, t.final_out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, fix.cpu);
        CHECK_TRUE(buf != nullptr, "cpu buf alloc");

        fill_tensors(fix.cpu, t, capacity, effective_active, -999.0f);

        ggml_status st = ggml_backend_graph_compute(fix.cpu, gf);
        CHECK_TRUE(st == GGML_STATUS_SUCCESS, "cpu graph compute failed");

        cpu_out.resize(ggml_nelements(t.final_out));
        cpu_state.resize(ggml_nelements(t.state_out));
        ggml_backend_tensor_get(t.final_out, cpu_out.data(), 0, cpu_out.size() * sizeof(float));
        ggml_backend_tensor_get(t.state_out, cpu_state.data(), 0, cpu_state.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Invariant 1: Tail isolation (varying inactive tokens produces bit-identical final state on GPU)
    CHECK_TRUE(gpu_state_run1.size() == gpu_state_run2.size(), "state size match");
    for (size_t i = 0; i < gpu_state_run1.size(); ++i) {
        CHECK_TRUE(gpu_state_run1[i] == gpu_state_run2[i], "GPU final state polluted by inactive tokens!");
    }

    // Invariant 2: Parity between GPU and CPU on active output steps
    const int64_t head_values = 128 * hv;
    for (int64_t step = 0; step < effective_active; ++step) {
        float gpu_first = gpu_out_run1[step * head_values];
        float cpu_first = cpu_out[step * head_values];
        float gpu_mid   = gpu_out_run1[step * head_values + head_values / 2];
        float cpu_mid   = cpu_out[step * head_values + head_values / 2];
        std::fprintf(stderr, "  [DIAG] step %lld: GPU[0]=%.6f CPU[0]=%.6f | GPU[mid]=%.6f CPU[mid]=%.6f\n",
                     (long long)step, gpu_first, cpu_first, gpu_mid, cpu_mid);
        for (int64_t v = 0; v < head_values; ++v) {
            size_t idx = step * head_values + v;
            CHECK_CLOSE(gpu_out_run1[idx], cpu_out[idx], 1e-3f, "active step output parity with CPU reference");
        }
    }

    // Invariant 3: Parity between GPU and CPU final recurrent state
    for (size_t i = 0; i < gpu_state_run1.size(); ++i) {
        CHECK_CLOSE(gpu_state_run1[i], cpu_state[i], 1e-3f, "final recurrent state parity with CPU reference");
    }

    std::fprintf(stderr, "  PASSED (A=%lld, C=%lld)\n", (long long)active, (long long)capacity);
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
    };

    // A=0 is the op's full-row encoding; check both encodings against CPU.
    const std::vector<test_case> cases = {
        {4, 0},
        {4, 1},
        {4, 2},
        {4, 4},
        {7, 0},
        {7, 7},
        {8, 0},
        {8, 1},
        {8, 2},
        {8, 4},
        {8, 8},
    };

    for (const auto & tc : cases) {
        if (!test_gdn_multistep_step(fix, tc.capacity, tc.active)) {
            std::fprintf(stderr, "FAILED on active=%lld, capacity=%lld\n",
                         (long long)tc.active, (long long)tc.capacity);
            return 1;
        }
    }
    const std::vector<uint8_t> tp5_headmap{ 0, 1, 2, 3, 1, 2, 3, 0, 1, 2 };
    // The partial K4 case must leave padding untouched and reverse-map only two snapshots.
    for (int64_t active : { 0, 2, 4 }) {
        if (!test_gdn_multistep_step(fix, 4, active, 4, 4, 10, tp5_headmap)) {
            std::fprintf(stderr, "FAILED on mapped TP5 GDN K=4 active=%lld\n", (long long)active);
            return 1;
        }
    }
    if (!test_gdn_multistep_step(fix, 7, 0, 1, 4, 10, tp5_headmap)) {
        std::fprintf(stderr, "FAILED on mapped TP5 GDN K=1 full-active seven-row graph\n");
        return 1;
    }
    for (int64_t active : { 0, 3, 7 }) {
        if (!test_gdn_multistep_step(fix, 7, active, 7)) {
            std::fprintf(stderr, "FAILED on GDN K=7 rollback bank active=%lld\n", (long long)active);
            return 1;
        }
    }
    if (!test_gdn_multistep_step(fix, 7, 0, 7, 4, 10, tp5_headmap)) {
        std::fprintf(stderr, "FAILED on mapped TP5 GDN K=7 rollback bank\n");
        return 1;
    }

    std::printf("test-vulkan-gdn-multistep: ALL TESTS PASSED (Vulkan GPU verified)\n");
    return 0;
}
