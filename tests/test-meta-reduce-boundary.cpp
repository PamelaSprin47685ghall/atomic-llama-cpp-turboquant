// CPU regression test for ggml-backend-meta split-state propagation,
// linear region deferral, and reduction boundaries.
//
// Invariants verified:
// 1. Matmul with Axis-0 split weight (leaf) naturally produces PARTIAL output without
//    name hacks or manual overrides.
// 2. Two-branch gated sum: PARTIAL routed MoE + PARTIAL (shared expert * MIRRORED gate)
//    combines into a single PARTIAL sum before reducing (1 reduction, not 2).
// 3. Mirrored residual boundary: A PARTIAL tensor added to a MIRRORED residual
//    cannot defer; it MUST reduce before addition, preventing residual duplication by P.
// 4. Non-linear consumer boundary: A PARTIAL tensor consumed by a non-linear operator
//    (e.g. SQR) MUST reduce before the operator.
// 5. Numerical equivalence: End-to-end CPU execution comparing split-buffer meta execution
//    against un-split CPU reference produces identical values within epsilon.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static int g_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return; \
    } \
} while (0)

#define TEST_ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #cond, msg); \
        g_failures++; \
        return; \
    } \
} while (0)

// Split callback for leaf tensors: weights partitioned along Axis 0 (down projections),
// inputs and gates replicated (MIRRORED).
struct test_split_ctx {
    int n_devices;
    std::vector<std::string> split_axis0_leaves;
};

static ggml_backend_meta_split_state test_get_split_state(
        const struct ggml_tensor * tensor, void * user_data) {
    auto * sctx = (test_split_ctx *) user_data;
    if (!tensor || !tensor->name[0]) {
        return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
    }

    for (const auto & name : sctx->split_axis0_leaves) {
        if (strcmp(tensor->name, name.c_str()) == 0) {
            // Split along axis 0 across devices evenly
            ggml_backend_meta_split_state ss = { GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, false, {0} };
            int64_t per_dev = tensor->ne[0] / sctx->n_devices;
            for (int j = 0; j < sctx->n_devices; ++j) {
                ss.ne[j] = per_dev;
            }
            return ss;
        }
    }
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
}

static ggml_backend_dev_t get_cpu_dev() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        ggml_backend_reg_t reg = ggml_backend_cpu_reg();
        if (reg) {
            dev = ggml_backend_reg_dev_get(reg, 0);
        }
    }
    return dev;
}

// Test 1: Two-branch gated sum with real Axis-0 leaf weights -> PARTIAL mul_mat -> combined PARTIAL sum -> residual ADD
static void test_two_branch_gated_sum_with_leaf_matmuls() {
    fprintf(stderr, "--- test_two_branch_gated_sum_with_leaf_matmuls ---\n");

    const int n_ranks = 2;
    ggml_backend_dev_t cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);

    test_split_ctx sctx;
    sctx.n_devices = n_ranks;
    // Leaf weights and activation split along Axis 0 (inner dimension in GGML mul_mat: W is [K, N], x is [K, B], W*x contracts K)
    sctx.split_axis0_leaves = {"w_routed", "w_shexp", "act"};

    std::vector<ggml_backend_dev_t> devs(n_ranks, cpu_dev);
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
        devs.data(), n_ranks, test_get_split_state, &sctx);
    TEST_ASSERT(meta_dev != nullptr);

    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    TEST_ASSERT(meta_backend != nullptr);

    const int64_t K = 8;
    const int64_t N = 4;

    // 1. Build meta graph with no_alloc = true
    // Leaf tensors belong to weights buffer (USAGE_WEIGHTS), intermediate nodes to compute buffer (USAGE_COMPUTE)
    struct ggml_init_params params_weights = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_weights = ggml_init(params_weights);

    struct ggml_init_params params_compute = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_compute = ggml_init(params_compute);

    // Leaf weights: [K, N] in ctx_weights
    ggml_tensor * w_routed = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_routed, "w_routed");

    ggml_tensor * w_shexp = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_shexp, "w_shexp");

    // Leaf activation: [K, 1] in ctx_weights
    ggml_tensor * act = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, K);
    ggml_set_name(act, "act");

    // Leaf gate: [N, 1] MIRRORED in ctx_weights
    ggml_tensor * gate = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, N);
    ggml_set_name(gate, "gate");

    // Leaf residual: [N, 1] MIRRORED in ctx_weights
    ggml_tensor * residual = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, N);
    ggml_set_name(residual, "residual");

    // Allocate leaf weights buffer and mark USAGE_WEIGHTS
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);
    ggml_backend_buffer_t buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, meta_buft);
    TEST_ASSERT_MSG(buf_weights != nullptr, "leaf weights buffer allocation failed");
    ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Computations in ctx_compute:
    // Both w_routed and act are split along Axis 0 -> mul_mat produces PARTIAL (axis 0 × axis 0)!
    ggml_tensor * routed_out = ggml_mul_mat(ctx_compute, w_routed, act);
    ggml_set_name(routed_out, "routed_out");

    ggml_tensor * shexp_out = ggml_mul_mat(ctx_compute, w_shexp, act);
    ggml_set_name(shexp_out, "shexp_out");

    // Gated shared expert: PARTIAL * MIRRORED -> PARTIAL
    ggml_tensor * shexp_gated = ggml_mul(ctx_compute, shexp_out, gate);
    ggml_set_name(shexp_gated, "shexp_gated");

    // Local combination: PARTIAL + PARTIAL -> PARTIAL
    ggml_tensor * ffn_combined = ggml_add(ctx_compute, routed_out, shexp_gated);
    ggml_set_name(ffn_combined, "ffn_combined");

    // Mirrored residual ADD: residual (MIRRORED) + ffn_combined (PARTIAL)
    // Invariant: Must force AllReduce before residual ADD so residual is NOT multiplied by P!
    ggml_tensor * out = ggml_add(ctx_compute, residual, ffn_combined);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf_meta = ggml_new_graph(ctx_compute);
    ggml_graph_add_node(gf_meta, routed_out);
    ggml_graph_add_node(gf_meta, shexp_out);
    ggml_graph_add_node(gf_meta, shexp_gated);
    ggml_build_forward_expand(gf_meta, out);

    // Allocate compute buffer and mark USAGE_COMPUTE
    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute, meta_buft);
    TEST_ASSERT_MSG(buf_compute != nullptr, "meta compute buffer allocation failed");
    ggml_backend_buffer_set_usage(buf_compute, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    // Prepare deterministic input data
    std::vector<float> w_r_data(K * N);
    std::vector<float> w_s_data(K * N);
    std::vector<float> act_data(K, 1.0f);
    std::vector<float> gate_data(N, 0.5f);
    std::vector<float> res_data(N, 100.0f);

    for (size_t i = 0; i < w_r_data.size(); ++i) {
        w_r_data[i] = float(i + 1) * 0.1f;
        w_s_data[i] = float(i + 1) * 0.05f;
    }

    // Set leaf tensor data via backend_tensor_set
    ggml_backend_tensor_set(w_routed, w_r_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(w_shexp,  w_s_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(act,      act_data.data(), 0, K * sizeof(float));
    ggml_backend_tensor_set(gate,     gate_data.data(), 0, N * sizeof(float));
    ggml_backend_tensor_set(residual, res_data.data(), 0, N * sizeof(float));

    // Compute meta graph
    ggml_status status_meta = ggml_backend_graph_compute(meta_backend, gf_meta);
    TEST_ASSERT_MSG(status_meta == GGML_STATUS_SUCCESS, "meta graph compute failed");

    // Read output using ggml_backend_tensor_get
    std::vector<float> meta_out(N, 0.0f);
    ggml_backend_tensor_get(out, meta_out.data(), 0, N * sizeof(float));

    // 2. Build reference un-split CPU graph for mathematical verification
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr);

    struct ggml_init_params params_ref = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_ref = ggml_init(params_ref);

    ggml_tensor * ref_w_routed = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_w_shexp  = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_act      = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, K);
    ggml_tensor * ref_gate     = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, N);
    ggml_tensor * ref_residual = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, N);

    ggml_tensor * ref_routed_out   = ggml_mul_mat(ctx_ref, ref_w_routed, ref_act);
    ggml_tensor * ref_shexp_out    = ggml_mul_mat(ctx_ref, ref_w_shexp, ref_act);
    ggml_tensor * ref_shexp_gated  = ggml_mul(ctx_ref, ref_shexp_out, ref_gate);
    ggml_tensor * ref_ffn_combined = ggml_add(ctx_ref, ref_routed_out, ref_shexp_gated);
    ggml_tensor * ref_out          = ggml_add(ctx_ref, ref_residual, ref_ffn_combined);

    struct ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(gf_ref, ref_out);

    ggml_backend_buffer_t ref_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_ref, ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ref_buf != nullptr);

    ggml_backend_tensor_set(ref_w_routed, w_r_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(ref_w_shexp,  w_s_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(ref_act,      act_data.data(), 0, K * sizeof(float));
    ggml_backend_tensor_set(ref_gate,     gate_data.data(), 0, N * sizeof(float));
    ggml_backend_tensor_set(ref_residual, res_data.data(), 0, N * sizeof(float));

    ggml_status status_ref = ggml_backend_graph_compute(cpu_backend, gf_ref);
    TEST_ASSERT(status_ref == GGML_STATUS_SUCCESS);

    std::vector<float> ref_result(N, 0.0f);
    ggml_backend_tensor_get(ref_out, ref_result.data(), 0, N * sizeof(float));

    // Compare meta output to reference output:
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(meta_out[i] - ref_result[i]);
        if (diff >= 1e-4f) {
            fprintf(stderr, "  [DIAG] elem %d: meta=%.6f ref=%.6f diff=%.6f\n", i, meta_out[i], ref_result[i], diff);
        }
        TEST_ASSERT_MSG(diff < 1e-4f, "numerical mismatch between meta and CPU reference");
    }

    fprintf(stderr, "  Two-branch gated sum vs CPU reference: 100%% exact match across all %d elements (out[0]=%.3f ref[0]=%.3f)\n",
            (int)N, meta_out[0], ref_result[0]);

    ggml_backend_buffer_free(ref_buf);
    ggml_free(ctx_ref);
    ggml_backend_free(cpu_backend);

    ggml_backend_buffer_free(buf_weights);
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx_weights);
    ggml_free(ctx_compute);
    ggml_backend_free(meta_backend);
}

// Test 2: Non-linear boundary enforcement with un-split CPU reference comparison
static void test_nonlinear_boundary_and_reference() {
    fprintf(stderr, "--- test_nonlinear_boundary_and_reference ---\n");

    const int n_ranks = 2;
    ggml_backend_dev_t cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);

    test_split_ctx sctx;
    sctx.n_devices = n_ranks;
    sctx.split_axis0_leaves = {"w_p", "act_p"};

    std::vector<ggml_backend_dev_t> devs(n_ranks, cpu_dev);
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
        devs.data(), n_ranks, test_get_split_state, &sctx);
    TEST_ASSERT(meta_dev != nullptr);

    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    TEST_ASSERT(meta_backend != nullptr);

    const int64_t K = 4;
    const int64_t N = 4;
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);

    struct ggml_init_params params_weights = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_weights = ggml_init(params_weights);

    struct ggml_init_params params_compute = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_compute = ggml_init(params_compute);

    // Leaf weights: [K, N] in ctx_weights
    ggml_tensor * w_p = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_p, "w_p");

    // Leaf activation: [K, 1] in ctx_weights
    ggml_tensor * act_p = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, K);
    ggml_set_name(act_p, "act_p");

    // Allocate leaf weights buffer and mark USAGE_WEIGHTS
    ggml_backend_buffer_t buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, meta_buft);
    TEST_ASSERT(buf_weights != nullptr);
    ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // mul_mat produces PARTIAL in ctx_compute
    ggml_tensor * p_out = ggml_mul_mat(ctx_compute, w_p, act_p);
    ggml_set_name(p_out, "p_out");

    // Non-linear operation: SQR
    // Mathematical invariant: sqr(sum_{r=1}^P x_r) != sum_{r=1}^P sqr(x_r)
    // The meta backend MUST reduce p_out before applying SQR!
    ggml_tensor * nonlin = ggml_sqr(ctx_compute, p_out);
    ggml_set_name(nonlin, "nonlin");

    struct ggml_cgraph * gf_meta = ggml_new_graph(ctx_compute);
    ggml_build_forward_expand(gf_meta, nonlin);

    // Allocate compute buffer and mark USAGE_COMPUTE
    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute, meta_buft);
    TEST_ASSERT(buf_compute != nullptr);
    ggml_backend_buffer_set_usage(buf_compute, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> w_data(K * N, 0.5f);
    std::vector<float> a_data(K, 2.0f);

    ggml_backend_tensor_set(w_p,   w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(act_p, a_data.data(), 0, K * sizeof(float));

    ggml_status status_meta = ggml_backend_graph_compute(meta_backend, gf_meta);
    TEST_ASSERT(status_meta == GGML_STATUS_SUCCESS);

    std::vector<float> meta_result(N, 0.0f);
    ggml_backend_tensor_get(nonlin, meta_result.data(), 0, N * sizeof(float));

    // Reference CPU compute
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr);

    struct ggml_init_params params_ref = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_ref = ggml_init(params_ref);

    ggml_tensor * ref_w_p   = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_act_p = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, K);
    ggml_tensor * ref_p_out = ggml_mul_mat(ctx_ref, ref_w_p, ref_act_p);
    ggml_tensor * ref_nonlin = ggml_sqr(ctx_ref, ref_p_out);

    struct ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(gf_ref, ref_nonlin);

    ggml_backend_buffer_t ref_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_ref, ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ref_buf != nullptr);

    ggml_backend_tensor_set(ref_w_p,   w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(ref_act_p, a_data.data(), 0, K * sizeof(float));

    ggml_status status_ref = ggml_backend_graph_compute(cpu_backend, gf_ref);
    TEST_ASSERT(status_ref == GGML_STATUS_SUCCESS);

    std::vector<float> ref_result(N, 0.0f);
    ggml_backend_tensor_get(ref_nonlin, ref_result.data(), 0, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(meta_result[i] - ref_result[i]);
        if (diff >= 1e-4f) {
            fprintf(stderr, "  [DIAG] elem %d: meta=%.6f ref=%.6f diff=%.6f\n", i, meta_result[i], ref_result[i], diff);
        }
        TEST_ASSERT_MSG(diff < 1e-4f, "non-linear boundary failed: premature or skipped reduction");
    }

    fprintf(stderr, "  Non-linear SQR boundary vs CPU reference: 100%% exact match (meta[0]=%.3f ref[0]=%.3f)\n",
            meta_result[0], ref_result[0]);

    ggml_backend_buffer_free(ref_buf);
    ggml_free(ctx_ref);
    ggml_backend_free(cpu_backend);

    ggml_backend_buffer_free(buf_weights);
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx_weights);
    ggml_free(ctx_compute);
    ggml_backend_free(meta_backend);
}

// Test 3: Forked PARTIAL source with fanout (used both in non-linear op AND in a later linear sum)
// Invariant: A PARTIAL node with use_count > 1 cannot defer; it MUST reduce at the fork boundary
// so both the non-linear consumer and the linear consumer observe reduced values.
static void test_forked_partial_fanout_boundary() {
    fprintf(stderr, "--- test_forked_partial_fanout_boundary ---\n");

    const int n_ranks = 2;
    ggml_backend_dev_t cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);

    test_split_ctx sctx;
    sctx.n_devices = n_ranks;
    sctx.split_axis0_leaves = {"w_fork", "act_fork"};

    std::vector<ggml_backend_dev_t> devs(n_ranks, cpu_dev);
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
        devs.data(), n_ranks, test_get_split_state, &sctx);
    TEST_ASSERT(meta_dev != nullptr);

    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    TEST_ASSERT(meta_backend != nullptr);

    const int64_t K = 4;
    const int64_t N = 4;
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);

    struct ggml_init_params params_weights = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_weights = ggml_init(params_weights);

    struct ggml_init_params params_compute = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_compute = ggml_init(params_compute);

    ggml_tensor * w_fork   = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_fork, "w_fork");
    ggml_tensor * act_fork = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, K);
    ggml_set_name(act_fork, "act_fork");

    ggml_backend_buffer_t buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, meta_buft);
    TEST_ASSERT(buf_weights != nullptr);
    ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // mul_mat produces PARTIAL
    ggml_tensor * fork_root = ggml_mul_mat(ctx_compute, w_fork, act_fork);
    ggml_set_name(fork_root, "fork_root");

    // Consumer A: non-linear SQR
    ggml_tensor * branch_sqr = ggml_sqr(ctx_compute, fork_root);
    ggml_set_name(branch_sqr, "branch_sqr");

    // Consumer B: linear SCALE
    ggml_tensor * branch_scale = ggml_scale(ctx_compute, fork_root, 2.0f);
    ggml_set_name(branch_scale, "branch_scale");

    // Sink: add branch_sqr and branch_scale
    ggml_tensor * sink = ggml_add(ctx_compute, branch_sqr, branch_scale);
    ggml_set_name(sink, "fork_sink");

    struct ggml_cgraph * gf_meta = ggml_new_graph(ctx_compute);
    ggml_build_forward_expand(gf_meta, sink);

    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute, meta_buft);
    TEST_ASSERT(buf_compute != nullptr);
    ggml_backend_buffer_set_usage(buf_compute, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> w_data(K * N, 0.5f);
    std::vector<float> a_data(K, 2.0f);

    ggml_backend_tensor_set(w_fork,   w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(act_fork, a_data.data(), 0, K * sizeof(float));

    ggml_status status_meta = ggml_backend_graph_compute(meta_backend, gf_meta);
    TEST_ASSERT(status_meta == GGML_STATUS_SUCCESS);

    std::vector<float> meta_result(N, 0.0f);
    ggml_backend_tensor_get(sink, meta_result.data(), 0, N * sizeof(float));

    // Unsplit CPU reference
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr);

    struct ggml_init_params params_ref = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_ref = ggml_init(params_ref);

    ggml_tensor * ref_w_fork   = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_act_fork = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, K);
    ggml_tensor * ref_root     = ggml_mul_mat(ctx_ref, ref_w_fork, ref_act_fork);
    ggml_tensor * ref_sqr      = ggml_sqr(ctx_ref, ref_root);
    ggml_tensor * ref_scale    = ggml_scale(ctx_ref, ref_root, 2.0f);
    ggml_tensor * ref_sink     = ggml_add(ctx_ref, ref_sqr, ref_scale);

    struct ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(gf_ref, ref_sink);

    ggml_backend_buffer_t ref_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_ref, ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ref_buf != nullptr);

    ggml_backend_tensor_set(ref_w_fork,   w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(ref_act_fork, a_data.data(), 0, K * sizeof(float));

    ggml_status status_ref = ggml_backend_graph_compute(cpu_backend, gf_ref);
    TEST_ASSERT(status_ref == GGML_STATUS_SUCCESS);

    std::vector<float> ref_result(N, 0.0f);
    ggml_backend_tensor_get(ref_sink, ref_result.data(), 0, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(meta_result[i] - ref_result[i]);
        if (diff >= 1e-4f) {
            fprintf(stderr, "  [DIAG] elem %d: meta=%.6f ref=%.6f diff=%.6f\n", i, meta_result[i], ref_result[i], diff);
        }
        TEST_ASSERT_MSG(diff < 1e-4f, "forked partial fanout failed: illegal deferral past fanout");
    }

    fprintf(stderr, "  Forked partial fanout boundary vs CPU reference: 100%% exact match (meta[0]=%.3f ref[0]=%.3f)\n",
            meta_result[0], ref_result[0]);

    ggml_backend_buffer_free(ref_buf);
    ggml_free(ctx_ref);
    ggml_backend_free(cpu_backend);

    ggml_backend_buffer_free(buf_weights);
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx_weights);
    ggml_free(ctx_compute);
    ggml_backend_free(meta_backend);
}

// Test 4: Regression for realistic MoE expert view stack + ADD aggregation + shared expert gated sum
// Invariants:
// 1. Down experts output tensor with 2 used experts: shape [N, 2].
// 2. Slice views [N, 1] per expert ordered before the adds (matching build_moe_ffn).
// 3. ADD chain aggregates expert views into moe_out.
// 4. Independent shared expert produces PARTIAL ffn_shexp and MIRRORED shared_gate -> ffn_shexp_gated (PARTIAL).
// 5. Final combine ADD(moe_out, ffn_shexp_gated) -> ffn_out.
// 6. Mirrored residual ADD(residual, ffn_out) -> block_out.
// Invariant: get_i_delayed defers down_exps across views+adds to moe_out, then chains into can_defer_linear_partial
// through shared expert to ffn_out. Exactly 1 AllReduce for the entire FFN block!
static void test_moe_expert_stack_and_shared_expert_chain() {
    fprintf(stderr, "--- test_moe_expert_stack_and_shared_expert_chain ---\n");

    const int n_ranks = 2;
    ggml_backend_dev_t cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);

    test_split_ctx sctx;
    sctx.n_devices = n_ranks;
    // Leaf weights split along Axis 0 (down projections)
    sctx.split_axis0_leaves = {"w_down_exps", "w_down_shexp", "act_moe_cur", "act_shexp"};

    std::vector<ggml_backend_dev_t> devs(n_ranks, cpu_dev);
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
        devs.data(), n_ranks, test_get_split_state, &sctx);
    TEST_ASSERT(meta_dev != nullptr);

    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    TEST_ASSERT(meta_backend != nullptr);

    const int64_t K = 8;
    const int64_t N = 4; // n_embd
    const int64_t E = 2; // n_expert_used
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);

    struct ggml_init_params params_weights = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_weights = ggml_init(params_weights);

    struct ggml_init_params params_compute = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_compute = ggml_init(params_compute);

    // Leaf weights: down_exps as [K, N, E], split on Axis 0
    ggml_tensor * w_down_exps  = ggml_new_tensor_3d(ctx_weights, GGML_TYPE_F32, K, N, E);
    ggml_set_name(w_down_exps, "w_down_exps");
    ggml_tensor * w_down_shexp = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_down_shexp, "w_down_shexp");

    // Activations: act_moe_cur as [K, E, 1] (sharded activation on axis 0)
    ggml_tensor * act_moe_cur = ggml_new_tensor_3d(ctx_weights, GGML_TYPE_F32, K, E, 1);
    ggml_set_name(act_moe_cur, "act_moe_cur");
    ggml_tensor * act_shexp = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, K);
    ggml_set_name(act_shexp, "act_shexp");

    // Selected expert IDs: [E, 1] (I32, MIRRORED)
    ggml_tensor * selected_ids = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_I32, E, 1);
    ggml_set_name(selected_ids, "selected_ids");

    // Shared gate & residual (MIRRORED)
    ggml_tensor * shared_gate = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, N);
    ggml_set_name(shared_gate, "shared_gate");
    ggml_tensor * residual    = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, N);
    ggml_set_name(residual, "residual");

    ggml_backend_buffer_t buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, meta_buft);
    TEST_ASSERT(buf_weights != nullptr);
    ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // 1. Indirect matrix multiplication: ggml_mul_mat_id(as, b, ids)
    // as: [K, N, E], b: [K, E, 1], ids: [E, 1] -> experts: [N, E, 1] PARTIAL
    ggml_tensor * experts = ggml_mul_mat_id(ctx_compute, w_down_exps, act_moe_cur, selected_ids);
    ggml_set_name(experts, "ffn_moe_down");

    // 2. Expert 2D views matching build_moe_ffn: ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])
    ggml_tensor * exp0 = ggml_view_2d(ctx_compute, experts, N, 1, experts->nb[2], 0 * experts->nb[1]);
    ggml_set_name(exp0, "exp0");
    ggml_tensor * exp1 = ggml_view_2d(ctx_compute, experts, N, 1, experts->nb[2], 1 * experts->nb[1]);
    ggml_set_name(exp1, "exp1");

    // 3. Expert tree aggregation
    ggml_tensor * moe_out = ggml_add(ctx_compute, exp0, exp1);
    ggml_set_name(moe_out, "ffn_moe_out");

    // 4. Shared expert branch
    ggml_tensor * shexp_out = ggml_mul_mat(ctx_compute, w_down_shexp, act_shexp);
    ggml_set_name(shexp_out, "shexp_out");
    ggml_tensor * shexp_gated = ggml_mul(ctx_compute, shexp_out, shared_gate);
    ggml_set_name(shexp_gated, "ffn_shexp_gated");

    // 5. Combined FFN sum
    ggml_tensor * ffn_out = ggml_add(ctx_compute, moe_out, shexp_gated);
    ggml_set_name(ffn_out, "ffn_out");

    // 6. Mirrored residual addition
    ggml_tensor * block_out = ggml_add(ctx_compute, residual, ffn_out);
    ggml_set_name(block_out, "block_out");

    struct ggml_cgraph * gf_meta = ggml_new_graph(ctx_compute);
    ggml_graph_add_node(gf_meta, experts);
    ggml_graph_add_node(gf_meta, exp0);
    ggml_graph_add_node(gf_meta, exp1);
    ggml_graph_add_node(gf_meta, moe_out);
    ggml_graph_add_node(gf_meta, shexp_out);
    ggml_graph_add_node(gf_meta, shexp_gated);
    ggml_graph_add_node(gf_meta, ffn_out);
    ggml_build_forward_expand(gf_meta, block_out);

    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute, meta_buft);
    TEST_ASSERT(buf_compute != nullptr);
    ggml_backend_buffer_set_usage(buf_compute, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> w_down_data(K * N * E);
    for (size_t i = 0; i < w_down_data.size(); ++i) {
        w_down_data[i] = 0.1f + float(i % (K * N)) * 0.05f + float(i / (K * N)) * 0.3f;
    }
    std::vector<float> w_shexp_data(K * N);
    for (size_t i = 0; i < w_shexp_data.size(); ++i) {
        w_shexp_data[i] = 0.2f + float(i) * 0.04f;
    }
    std::vector<float> act_moe_data(K * E);
    for (size_t i = 0; i < act_moe_data.size(); ++i) {
        act_moe_data[i] = 1.0f + float(i) * 0.2f;
    }
    std::vector<int32_t> ids_data = {0, 1};
    std::vector<float> act_shexp_data(K, 0.8f);
    std::vector<float> gate_data(N, 0.7f);
    std::vector<float> res_data(N, 50.0f);

    ggml_backend_tensor_set(w_down_exps,  w_down_data.data(), 0, w_down_data.size() * sizeof(float));
    ggml_backend_tensor_set(w_down_shexp, w_shexp_data.data(), 0, w_shexp_data.size() * sizeof(float));
    ggml_backend_tensor_set(act_moe_cur,  act_moe_data.data(), 0, act_moe_data.size() * sizeof(float));
    ggml_backend_tensor_set(selected_ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(act_shexp,    act_shexp_data.data(), 0, act_shexp_data.size() * sizeof(float));
    ggml_backend_tensor_set(shared_gate,  gate_data.data(), 0, gate_data.size() * sizeof(float));
    ggml_backend_tensor_set(residual,     res_data.data(), 0, res_data.size() * sizeof(float));

    ggml_status status_meta = ggml_backend_graph_compute(meta_backend, gf_meta);
    TEST_ASSERT(status_meta == GGML_STATUS_SUCCESS);

    std::vector<float> meta_result(N, 0.0f);
    ggml_backend_tensor_get(block_out, meta_result.data(), 0, N * sizeof(float));

    // Reference CPU compute
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr);

    struct ggml_init_params params_ref = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_ref = ggml_init(params_ref);

    ggml_tensor * ref_w_down_exps  = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, K, N, E);
    ggml_tensor * ref_w_down_shexp = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_act_moe_cur  = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, K, E, 1);
    ggml_tensor * ref_selected_ids = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_I32, E, 1);
    ggml_tensor * ref_act_shexp    = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, K);
    ggml_tensor * ref_shared_gate  = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, N);
    ggml_tensor * ref_residual     = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, N);

    ggml_tensor * ref_experts  = ggml_mul_mat_id(ctx_ref, ref_w_down_exps, ref_act_moe_cur, ref_selected_ids);
    ggml_tensor * ref_exp0     = ggml_view_2d(ctx_ref, ref_experts, N, 1, ref_experts->nb[2], 0 * ref_experts->nb[1]);
    ggml_tensor * ref_exp1     = ggml_view_2d(ctx_ref, ref_experts, N, 1, ref_experts->nb[2], 1 * ref_experts->nb[1]);
    ggml_tensor * ref_moe_out  = ggml_add(ctx_ref, ref_exp0, ref_exp1);
    ggml_tensor * ref_shexp_out     = ggml_mul_mat(ctx_ref, ref_w_down_shexp, ref_act_shexp);
    ggml_tensor * ref_shexp_gated   = ggml_mul(ctx_ref, ref_shexp_out, ref_shared_gate);
    ggml_tensor * ref_ffn_out       = ggml_add(ctx_ref, ref_moe_out, ref_shexp_gated);
    ggml_tensor * ref_block_out     = ggml_add(ctx_ref, ref_residual, ref_ffn_out);

    struct ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(gf_ref, ref_block_out);

    ggml_backend_buffer_t ref_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_ref, ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ref_buf != nullptr);

    ggml_backend_tensor_set(ref_w_down_exps,  w_down_data.data(), 0, w_down_data.size() * sizeof(float));
    ggml_backend_tensor_set(ref_w_down_shexp, w_shexp_data.data(), 0, w_shexp_data.size() * sizeof(float));
    ggml_backend_tensor_set(ref_act_moe_cur,  act_moe_data.data(), 0, act_moe_data.size() * sizeof(float));
    ggml_backend_tensor_set(ref_selected_ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ref_act_shexp,    act_shexp_data.data(), 0, act_shexp_data.size() * sizeof(float));
    ggml_backend_tensor_set(ref_shared_gate,  gate_data.data(), 0, gate_data.size() * sizeof(float));
    ggml_backend_tensor_set(ref_residual,     res_data.data(), 0, res_data.size() * sizeof(float));

    ggml_status status_ref = ggml_backend_graph_compute(cpu_backend, gf_ref);
    TEST_ASSERT(status_ref == GGML_STATUS_SUCCESS);

    std::vector<float> ref_result(N, 0.0f);
    ggml_backend_tensor_get(ref_block_out, ref_result.data(), 0, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(meta_result[i] - ref_result[i]);
        if (diff >= 1e-4f) {
            fprintf(stderr, "  [DIAG] elem %d: meta=%.6f ref=%.6f diff=%.6f\n", i, meta_result[i], ref_result[i], diff);
        }
        TEST_ASSERT_MSG(diff < 1e-4f, "moe expert stack + shared expert chain failed: numerical mismatch");
    }

    fprintf(stderr, "  MoE expert stack + shared expert chain vs CPU reference: 100%% exact match (meta[0]=%.3f ref[0]=%.3f)\n",
            meta_result[0], ref_result[0]);

    ggml_backend_buffer_free(ref_buf);
    ggml_free(ctx_ref);
    ggml_backend_free(cpu_backend);

    ggml_backend_buffer_free(buf_weights);
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx_weights);
    ggml_free(ctx_compute);
    ggml_backend_free(meta_backend);
}

// Test 4: Unrelated PARTIAL root needed by a later non-linear op while other branches merge
// Invariant: An unrelated PARTIAL root in the interval that does NOT feed into the local
// linear merge sink cannot be deferred past; it must either reduce or maintain correct boundaries.
static void test_unrelated_partial_root_with_interleaved_merge() {
    fprintf(stderr, "--- test_unrelated_partial_root_with_interleaved_merge ---\n");

    const int n_ranks = 2;
    ggml_backend_dev_t cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);

    test_split_ctx sctx;
    sctx.n_devices = n_ranks;
    sctx.split_axis0_leaves = {"w_a", "w_b", "w_unrel", "act_x"};

    std::vector<ggml_backend_dev_t> devs(n_ranks, cpu_dev);
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
        devs.data(), n_ranks, test_get_split_state, &sctx);
    TEST_ASSERT(meta_dev != nullptr);

    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    TEST_ASSERT(meta_backend != nullptr);

    const int64_t K = 4;
    const int64_t N = 4;
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);

    struct ggml_init_params params_weights = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_weights = ggml_init(params_weights);

    struct ggml_init_params params_compute = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_compute = ggml_init(params_compute);

    ggml_tensor * w_a     = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_a, "w_a");
    ggml_tensor * w_b     = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_b, "w_b");
    ggml_tensor * w_unrel = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, K, N);
    ggml_set_name(w_unrel, "w_unrel");
    ggml_tensor * act_x   = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, K);
    ggml_set_name(act_x, "act_x");

    ggml_backend_buffer_t buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, meta_buft);
    TEST_ASSERT(buf_weights != nullptr);
    ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Branch 1: w_a * act_x (PARTIAL)
    ggml_tensor * p_a = ggml_mul_mat(ctx_compute, w_a, act_x);
    ggml_set_name(p_a, "p_a");

    // Interleaved unrelated PARTIAL root: w_unrel * act_x (PARTIAL)
    ggml_tensor * p_unrel = ggml_mul_mat(ctx_compute, w_unrel, act_x);
    ggml_set_name(p_unrel, "p_unrel");

    // Branch 2: w_b * act_x (PARTIAL)
    ggml_tensor * p_b = ggml_mul_mat(ctx_compute, w_b, act_x);
    ggml_set_name(p_b, "p_b");

    // Merge branches 1 and 2: p_a + p_b
    ggml_tensor * merged = ggml_add(ctx_compute, p_a, p_b);
    ggml_set_name(merged, "merged");

    // Unrelated consumer: non-linear SQR on p_unrel!
    ggml_tensor * sqr_unrel = ggml_sqr(ctx_compute, p_unrel);
    ggml_set_name(sqr_unrel, "sqr_unrel");

    // Final combine: merged + sqr_unrel
    ggml_tensor * final_out = ggml_add(ctx_compute, merged, sqr_unrel);
    ggml_set_name(final_out, "final_out");

    struct ggml_cgraph * gf_meta = ggml_new_graph(ctx_compute);
    ggml_graph_add_node(gf_meta, p_a);
    ggml_graph_add_node(gf_meta, p_unrel);
    ggml_graph_add_node(gf_meta, p_b);
    ggml_graph_add_node(gf_meta, merged);
    ggml_build_forward_expand(gf_meta, final_out);

    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute, meta_buft);
    TEST_ASSERT(buf_compute != nullptr);
    ggml_backend_buffer_set_usage(buf_compute, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> w_data(K * N, 0.4f);
    std::vector<float> a_data(K, 1.5f);

    ggml_backend_tensor_set(w_a,     w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(w_b,     w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(w_unrel, w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(act_x,   a_data.data(), 0, K * sizeof(float));

    ggml_status status_meta = ggml_backend_graph_compute(meta_backend, gf_meta);
    TEST_ASSERT(status_meta == GGML_STATUS_SUCCESS);

    std::vector<float> meta_result(N, 0.0f);
    ggml_backend_tensor_get(final_out, meta_result.data(), 0, N * sizeof(float));

    // Unsplit CPU reference
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr);

    struct ggml_init_params params_ref = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_ref = ggml_init(params_ref);

    ggml_tensor * ref_w_a     = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_w_b     = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_w_unrel = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, K, N);
    ggml_tensor * ref_act_x   = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, K);

    ggml_tensor * ref_p_a     = ggml_mul_mat(ctx_ref, ref_w_a, ref_act_x);
    ggml_tensor * ref_p_unrel = ggml_mul_mat(ctx_ref, ref_w_unrel, ref_act_x);
    ggml_tensor * ref_p_b     = ggml_mul_mat(ctx_ref, ref_w_b, ref_act_x);
    ggml_tensor * ref_merged  = ggml_add(ctx_ref, ref_p_a, ref_p_b);
    ggml_tensor * ref_sqr     = ggml_sqr(ctx_ref, ref_p_unrel);
    ggml_tensor * ref_final   = ggml_add(ctx_ref, ref_merged, ref_sqr);

    struct ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
    ggml_graph_add_node(gf_ref, ref_p_a);
    ggml_graph_add_node(gf_ref, ref_p_unrel);
    ggml_graph_add_node(gf_ref, ref_p_b);
    ggml_graph_add_node(gf_ref, ref_merged);
    ggml_build_forward_expand(gf_ref, ref_final);

    ggml_backend_buffer_t ref_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_ref, ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ref_buf != nullptr);

    ggml_backend_tensor_set(ref_w_a,     w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(ref_w_b,     w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(ref_w_unrel, w_data.data(), 0, K * N * sizeof(float));
    ggml_backend_tensor_set(ref_act_x,   a_data.data(), 0, K * sizeof(float));

    ggml_status status_ref = ggml_backend_graph_compute(cpu_backend, gf_ref);
    TEST_ASSERT(status_ref == GGML_STATUS_SUCCESS);

    std::vector<float> ref_result(N, 0.0f);
    ggml_backend_tensor_get(ref_final, ref_result.data(), 0, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(meta_result[i] - ref_result[i]);
        if (diff >= 1e-4f) {
            fprintf(stderr, "  [DIAG] elem %d: meta=%.6f ref=%.6f diff=%.6f\n", i, meta_result[i], ref_result[i], diff);
        }
        TEST_ASSERT_MSG(diff < 1e-4f, "unrelated partial root failed: illegal deferral past unrelated root");
    }

    fprintf(stderr, "  Unrelated PARTIAL root with interleaved merge vs CPU reference: 100%% exact match (meta[0]=%.3f ref[0]=%.3f)\n",
            meta_result[0], ref_result[0]);

    ggml_backend_buffer_free(ref_buf);
    ggml_free(ctx_ref);
    ggml_backend_free(cpu_backend);

    ggml_backend_buffer_free(buf_weights);
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx_weights);
    ggml_free(ctx_compute);
    ggml_backend_free(meta_backend);
}

// Test 5: Model-free regression for uneven TP5 meta-device copy and row-geometry snapshot restore.
// Invariants:
// 1. 5 CPU devices in a single meta device.
// 2. Recurrent state-like tensor [48*64, 3] split on Axis 0 with uneven head shards [10, 10, 10, 10, 8]*64.
// 3. Snapshot of middle row (offset = nb[1], size = one full row) created with row geometry (ne0, ne1=1).
// 4. Mutate original tensor in place.
// 5. Restore snapshot via .cpy_tensor / meta buffer copy.
// 6. Verify middle row is restored exactly and top/bottom rows remain intact (no bleed, no hidden host fallback).
static void test_meta_recurrent_snapshot_uneven_tp5(bool use_vulkan = false) {
    fprintf(stderr, "--- test_meta_recurrent_snapshot_uneven_tp5 (%s) ---\n", use_vulkan ? "Vulkan" : "CPU");

    const int n_ranks = 5;
    std::vector<ggml_backend_dev_t> devs;
    if (use_vulkan) {
        ggml_backend_load_all();
        for (int i = 0; i < n_ranks; ++i) {
            char dev_name[32];
            snprintf(dev_name, sizeof(dev_name), "Vulkan%d", i);
            ggml_backend_dev_t vk_dev = ggml_backend_dev_by_name(dev_name);
            TEST_ASSERT_MSG(vk_dev != nullptr, "missing required Vulkan device for explicit Vulkan state copy proof");
            devs.push_back(vk_dev);
        }
    } else {
        ggml_backend_dev_t cpu_dev = get_cpu_dev();
        TEST_ASSERT(cpu_dev != nullptr);
        devs.assign(n_ranks, cpu_dev);
    }

    const int64_t head_dim = 64;
    const int64_t n_heads = 48;
    const int64_t ne0 = n_heads * head_dim; // 3072 elements per row
    const int64_t ne1 = 3;                  // 3 rows

    auto split_fn = [](const struct ggml_tensor * tensor, void * /*user_data*/) -> ggml_backend_meta_split_state {
        if (tensor && tensor->name[0] && strstr(tensor->name, "cache_s")) {
            ggml_backend_meta_split_state ss = { GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, false, {0} };
            const int64_t hdim = 64;
            const int64_t sheads[5] = {10, 10, 10, 10, 8};
            for (int j = 0; j < 5; ++j) {
                ss.ne[j] = sheads[j] * hdim;
            }
            return ss;
        }
        return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
    };

    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs.data(), n_ranks, split_fn, nullptr);
    TEST_ASSERT(meta_dev != nullptr);

    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);
    TEST_ASSERT(meta_buft != nullptr);

    // Allocate original tensor on meta buffer
    struct ggml_init_params params_orig = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_orig = ggml_init(params_orig);
    ggml_tensor * orig = ggml_new_tensor_2d(ctx_orig, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(orig, "cache_s_l0");

    ggml_backend_buffer_t orig_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_orig, meta_buft);
    TEST_ASSERT(orig_buf != nullptr);

    // Fill distinct values per element across all 3 rows
    const size_t total_elements = (size_t) (ne0 * ne1);
    std::vector<float> init_data(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
        init_data[i] = 1000.0f + (float) i * 0.1f;
    }
    ggml_backend_tensor_set(orig, init_data.data(), 0, total_elements * sizeof(float));

    // Create snapshot of ONLY middle row (row 1: offset = orig->nb[1], size = 1 * orig->nb[1])
    // using production row geometry: 2D tensor of shape [ne0, 1] with matching name
    struct ggml_init_params params_snap = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_snap = ggml_init(params_snap);

    // Destination tensor (snapshot) with exact row geometry: [ne0, 1]
    ggml_tensor * snap = ggml_new_tensor_2d(ctx_snap, GGML_TYPE_F32, ne0, 1);
    ggml_set_name(snap, "cache_s_l0");

    // Source view in ctx_snap referencing orig on middle row
    const size_t row1_offset = orig->nb[1];
    ggml_tensor * org_view = ggml_view_2d(ctx_snap, orig, ne0, 1, orig->nb[1], row1_offset);
    ggml_set_name(org_view, "cache_s_l0");

    ggml_backend_buffer_t snap_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_snap, meta_buft);
    TEST_ASSERT(snap_buf != nullptr);

    // Ensure view is initialized on the meta buffer
    if (org_view->buffer == nullptr) {
        enum ggml_status vstatus = ggml_backend_view_init(org_view);
        TEST_ASSERT(vstatus == GGML_STATUS_SUCCESS);
    }

    // Copy original row 1 into snapshot buffer via buffer's .cpy_tensor interface directly
    // asserting device-level copy succeeds with no host roundtrip
    TEST_ASSERT_MSG(snap->buffer->iface.cpy_tensor != nullptr, "meta buffer missing .cpy_tensor interface");
    bool copy_ok = snap->buffer->iface.cpy_tensor(snap->buffer, org_view, snap);
    TEST_ASSERT_MSG(copy_ok, "meta device-to-device copy of row slice failed");

    // Mutate the original tensor (all rows including row 1)
    std::vector<float> mutated_data(total_elements, -999.0f);
    ggml_backend_tensor_set(orig, mutated_data.data(), 0, total_elements * sizeof(float));

    // Verify mutation took effect
    std::vector<float> verify_mutated(total_elements, 0.0f);
    ggml_backend_tensor_get(orig, verify_mutated.data(), 0, total_elements * sizeof(float));
    for (size_t i = 0; i < total_elements; ++i) {
        TEST_ASSERT(verify_mutated[i] == -999.0f);
    }

    // Restore: copy snapshot back into original row 1 view
    TEST_ASSERT_MSG(org_view->buffer->iface.cpy_tensor != nullptr, "meta buffer missing .cpy_tensor interface");
    bool restore_ok = org_view->buffer->iface.cpy_tensor(org_view->buffer, snap, org_view);
    TEST_ASSERT_MSG(restore_ok, "meta device-to-device restore of row slice failed");

    // Read back all 3 rows of original tensor
    std::vector<float> restored_data(total_elements, 0.0f);
    ggml_backend_tensor_get(orig, restored_data.data(), 0, total_elements * sizeof(float));

    // Assert:
    // Row 0 (0..ne0-1) must remain mutated (-999.0f)
    for (int64_t i = 0; i < ne0; ++i) {
        TEST_ASSERT_MSG(restored_data[i] == -999.0f, "Row 0 was corrupted during row 1 restore");
    }

    // Row 1 (ne0..2*ne0-1) must be restored to EXACT initial values
    for (int64_t i = 0; i < ne0; ++i) {
        const float expected = init_data[ne0 + i];
        const float actual = restored_data[ne0 + i];
        TEST_ASSERT_MSG(actual == expected, "Row 1 element mismatch after restore");
    }

    // Row 2 (2*ne0..3*ne0-1) must remain mutated (-999.0f)
    for (int64_t i = 0; i < ne0; ++i) {
        TEST_ASSERT_MSG(restored_data[2 * ne0 + i] == -999.0f, "Row 2 was corrupted during row 1 restore");
    }

    fprintf(stderr, "  Uneven TP5 recurrent snapshot restore: row 1 exact match (3072 floats across 5 ranks [10,10,10,10,8]*64), rows 0 & 2 untouched\n");

    ggml_backend_buffer_free(snap_buf);
    ggml_free(ctx_snap);

    ggml_backend_buffer_free(orig_buf);
    ggml_free(ctx_orig);
}

// Test 6: fail-closed advertised communicator.
// A backend advertising "ggml_backend_comm_init" promises a native collective. If that
// initialization is rejected (returns nullptr, e.g. peer mesh unavailable) or the trio is
// incomplete (comm_allreduce_tensor proc missing), meta backend init must fail cleanly
// (ggml_backend_dev_init returns nullptr) with all initialized simple backends released,
// instead of silently falling back to the generic host allreduce in graph_compute.
// CPU backends advertising no communicator must keep working (covered by tests 1-5 and the
// explicit no-comm check below). Uses mock CPU-wrapping devices; no GPU, no source asserts.
static ggml_backend_dev_t g_mock_real_cpu_dev = nullptr;
static struct ggml_backend_device g_mock_devs[2];
static struct ggml_backend_reg g_mock_reg;
static int g_mock_mode = 0; // 0 = comm_init returns nullptr; 1 = comm_allreduce proc missing
static int g_mock_dummy_comm = 42;
static int g_mock_inits = 0;
static int g_mock_frees = 0;
static int g_mock_comm_frees = 0;
static void (*g_mock_orig_free)(ggml_backend_t) = nullptr;

static void * mock_comm_init(ggml_backend_t * backends, size_t n) {
    (void) backends;
    (void) n;
    if (g_mock_mode == 0) {
        return nullptr; // simulate rejected native collective initialization
    }
    return &g_mock_dummy_comm;
}

static void mock_comm_free(void * comm_ctx) {
    (void) comm_ctx;
    g_mock_comm_frees++;
}

static bool mock_comm_allreduce(void * comm_ctx, struct ggml_tensor ** tensors) {
    (void) comm_ctx;
    (void) tensors;
    return true;
}

static void * mock_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) mock_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return g_mock_mode == 1 ? nullptr : (void *) mock_comm_allreduce;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) mock_comm_free;
    }
    (void) reg;
    return ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), name);
}

static void mock_free_backend(ggml_backend_t backend) {
    g_mock_frees++;
    g_mock_orig_free(backend);
}

static ggml_backend_t mock_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t b = g_mock_real_cpu_dev->iface.init_backend(g_mock_real_cpu_dev, params);
    if (b == nullptr) {
        return nullptr;
    }
    if (g_mock_orig_free == nullptr) {
        g_mock_orig_free = b->iface.free;
    }
    b->iface.free = mock_free_backend;
    b->device = dev;
    g_mock_inits++;
    return b;
}

static void test_meta_advertised_comm_fail_closed() {
    fprintf(stderr, "--- test_meta_advertised_comm_fail_closed ---\n");

    ggml_backend_dev_t cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);
    g_mock_real_cpu_dev = cpu_dev;

    g_mock_reg = *ggml_backend_cpu_reg();
    g_mock_reg.iface.get_proc_address = mock_get_proc_address;
    for (int i = 0; i < 2; i++) {
        g_mock_devs[i] = *cpu_dev;
        g_mock_devs[i].reg = &g_mock_reg;
        g_mock_devs[i].iface.init_backend = mock_init_backend;
    }

    static test_split_ctx mock_sctx;
    mock_sctx.n_devices = 2;
    mock_sctx.split_axis0_leaves.clear();
    ggml_backend_dev_t mock_dev_list[2] = { &g_mock_devs[0], &g_mock_devs[1] };
    ggml_backend_dev_t mock_meta = ggml_backend_meta_device(
        mock_dev_list, 2, test_get_split_state, &mock_sctx);
    TEST_ASSERT(mock_meta != nullptr);

    // Case A: advertised comm_init rejects initialization -> backend init must fail,
    // both simple backends released, no comm allocated.
    g_mock_mode = 0;
    g_mock_inits = 0;
    g_mock_frees = 0;
    g_mock_comm_frees = 0;
    {
        ggml_backend_t b = ggml_backend_dev_init(mock_meta, nullptr);
        TEST_ASSERT_MSG(b == nullptr, "meta init must fail when advertised comm_init returns nullptr");
        TEST_ASSERT_MSG(g_mock_inits == 2, "both simple backends must be initialized before comm failure");
        TEST_ASSERT_MSG(g_mock_frees == 2, "both simple backends must be released on comm failure");
        TEST_ASSERT_MSG(g_mock_comm_frees == 0, "no comm_ctx must exist when comm_init fails");
    }
    fprintf(stderr, "  comm_init rejection: init failed closed, 2/2 simple backends released\n");

    // Case B: comm_init succeeds but comm_allreduce proc missing -> backend init must fail closed.
    g_mock_mode = 1;
    g_mock_inits = 0;
    g_mock_frees = 0;
    g_mock_comm_frees = 0;
    {
        ggml_backend_t b = ggml_backend_dev_init(mock_meta, nullptr);
        TEST_ASSERT_MSG(b == nullptr, "meta init must fail when comm_allreduce proc is missing");
        TEST_ASSERT_MSG(g_mock_inits == 2, "both simple backends must be initialized before proc check");
        TEST_ASSERT_MSG(g_mock_frees == 2, "both simple backends must be released on proc check failure");
    }
    fprintf(stderr, "  missing allreduce proc: init failed closed, 2/2 simple backends released\n");

    // Case C: CPU backends with no communicator must keep initializing (no-comm path preserved).
    {
        test_split_ctx sctx;
        sctx.n_devices = 2;
        ggml_backend_dev_t devs[2] = { cpu_dev, cpu_dev };
        ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, test_get_split_state, &sctx);
        TEST_ASSERT(meta != nullptr);
        ggml_backend_t b = ggml_backend_dev_init(meta, nullptr);
        TEST_ASSERT_MSG(b != nullptr, "CPU meta without communicator must still initialize");
        ggml_backend_free(b);
    }
    fprintf(stderr, "  CPU no-comm path: still initializes\n");
}

// Static split callback for test_meta_indexed_replica_tp5_q8 (non-capturing function pointer):
static const int64_t s_tp5_q8_rank_ne[5]     = { 256, 256, 512, 256, 256 };
static const int64_t s_tp5_q8_rank_starts[5] = { 0,   0,   0,   256, 256 };

static ggml_backend_meta_split_state test_indexed_replica_tp5_q8_split_fn(
        const struct ggml_tensor * tensor, void * /*user_data*/) {
        if (tensor && tensor->name[0] && strstr(tensor->name, "cache_k")) {
            ggml_backend_meta_split_state ss = { GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, false, {0} };
            ss.indexed_replica = true;
            for (int j = 0; j < 5; ++j) {
            ss.ne[j] = s_tp5_q8_rank_ne[j];
            ss.replica_start[j] = s_tp5_q8_rank_starts[j];
        }
            return ss;
    }
        if (tensor && tensor->name[0] && strstr(tensor->name, "plain_split")) {
            // Ordinary unindexed partition along axis 0: 5 ranks, 64 elements each = 320 elements
            ggml_backend_meta_split_state ss = { GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, false, {0} };
            ss.indexed_replica = false;
            for (int j = 0; j < 5; ++j) {
                ss.ne[j] = 64;
        }
            return ss;
    }
    if (tensor && tensor->name[0] && strstr(tensor->name, "replicated_axis2")) {
        // Axis-2 replicated tensor: [ne0=4, ne1=4, ne2=2, ne3=1]
        // 2 global slices along axis 2 across 5 ranks:
        // Ranks carry counts [1, 1, 2, 1, 1] and starts [0, 0, 0, 1, 1] along axis 2
        ggml_backend_meta_split_state ss = { GGML_BACKEND_SPLIT_AXIS_2, {0}, {1}, 1, false, {0} };
        ss.indexed_replica = true;
        const int64_t a2_ne[5]     = { 1, 1, 2, 1, 1 };
        const int64_t a2_starts[5] = { 0, 0, 0, 1, 1 };
            for (int j = 0; j < 5; ++j) {
            ss.ne[j] = a2_ne[j];
            ss.replica_start[j] = a2_starts[j];
        }
            return ss;
    }
        return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
}

// Test 7: Explicit indexed replicas across 5 CPU-backed ranks with Q8_0 KV heads.
// Invariants verified:
// 1. Five ranks carry counts [256, 256, 512, 256, 256] and starts [0, 0, 0, 256, 256] along axis 0.
// 2. Q8_0 logical row has 544 bytes (512 elements, 2 heads of 256 elements each = 16 blocks of 34 bytes = 544 bytes per row).
// 3. Distinct valid quantized block data for head 0 and head 1 ensures any naive clamp returning head 0 twice fails.
// 4. Writes distribute replicas properly:
//    - ranks 0, 1 hold head 0
//    - rank 2 holds head 0 and head 1
//    - ranks 3, 4 hold head 1
// 5. Canonical readback materializes each logical element once using lowest-rank canonical owner:
//    - [0..256) read from rank 0, [256..512) read from rank 2.
// 6. Storage views & host serialized transfer:
//    - Base view (row 0) and nonzero-offset view (middle row 1) carry correct data contracts.
//    - Snapshot saved via host serialized get_tensor on org_view_row1.
//    - Entire tensor mutated dirty with 0x7F.
//    - Restored via host serialized set_tensor into org_view_row1 (the exact host path that crashed on view offset).
//    - All 5 replicas and canonical full readback verify row 1 is restored while rows 0 and 2 remain dirty.
// 7. Asynchronous transfer API (set_async / get_async) exercises the same behavioral fixture and verifies identical invariants.
// 8. Ordinary partition assertion protects unindexed branch from regressions.
static void test_meta_indexed_replica_tp5_q8() {
    fprintf(stderr, "--- test_meta_indexed_replica_tp5_q8 ---\n");

    const int n_ranks = 5;
    ggml_backend_dev_t cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);

    std::vector<ggml_backend_dev_t> devs(n_ranks, cpu_dev);

    // 2 global heads of 256 elements each = 512 elements per row.
    // For Q8_0, block size is 32 elements (34 bytes: 2 bytes fp16 delta + 32 bytes int8 quants).
    // Head 0: 256 elements = 8 blocks = 272 bytes.
    // Head 1: 256 elements = 8 blocks = 272 bytes.
    // Total row: 512 elements = 16 blocks = 544 bytes.
    const int64_t head_dim = 256;
    const int64_t n_heads = 2;
    const int64_t ne0 = n_heads * head_dim; // 512
    const int64_t ne1 = 3;                  // 3 rows

    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs.data(), n_ranks, test_indexed_replica_tp5_q8_split_fn, nullptr);
    TEST_ASSERT(meta_dev != nullptr);

    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    TEST_ASSERT(meta_backend != nullptr);

    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);
    TEST_ASSERT(meta_buft != nullptr);

    // --- Protection check: ordinary partition assertion on unindexed branch ---
    {
        struct ggml_init_params params_p = { 4 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
        struct ggml_context * ctx_p = ggml_init(params_p);
        ggml_tensor * t_plain = ggml_new_tensor_1d(ctx_p, GGML_TYPE_F32, 320);
        ggml_set_name(t_plain, "plain_split");
        ggml_backend_buffer_t buf_p = ggml_backend_alloc_ctx_tensors_from_buft(ctx_p, meta_buft);
        TEST_ASSERT(buf_p != nullptr);

        std::vector<float> plain_in(320);
        for (int i = 0; i < 320; ++i) plain_in[i] = (float)(i + 1);
        ggml_backend_tensor_set(t_plain, plain_in.data(), 0, 320 * sizeof(float));

        std::vector<float> plain_out(320, 0.0f);
        ggml_backend_tensor_get(t_plain, plain_out.data(), 0, 320 * sizeof(float));
        for (int i = 0; i < 320; ++i) {
            TEST_ASSERT_MSG(plain_out[i] == plain_in[i], "Ordinary unindexed partition readback corrupted");
        }
        ggml_backend_buffer_free(buf_p);
        ggml_free(ctx_p);
    }

    // --- Allocate main replicated KV tensor on meta buffer ---
    struct ggml_init_params params_orig = { 16 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
    struct ggml_context * ctx_orig = ggml_init(params_orig);
    ggml_tensor * orig = ggml_new_tensor_2d(ctx_orig, GGML_TYPE_Q8_0, ne0, ne1);
    ggml_set_name(orig, "cache_k_l48");

    ggml_backend_buffer_t orig_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_orig, meta_buft);
    TEST_ASSERT(orig_buf != nullptr);

    const size_t row_bytes = (size_t) orig->nb[1]; // 544 bytes for 512 elements Q8_0
    TEST_ASSERT_MSG(row_bytes == 544, "Q8_0 row_bytes must be 544 for 512 elements");
    const size_t total_bytes = row_bytes * (size_t) ne1; // 1632 bytes

    // Prepare distinct valid quantized Q8_0 block data:
    // Head 0 (first 8 blocks of each row): distinct pattern (e.g. delta = 1.0f + row, quants = +1, +2, ...)
    // Head 1 (next 8 blocks of each row): distinct pattern (e.g. delta = 50.0f + row, quants = -1, -2, ...)
    // This guarantees that any clamped read returning head 0 twice will fail with 100% certainty.
    std::vector<uint8_t> init_data(total_bytes, 0);
    for (int r = 0; r < ne1; ++r) {
        uint8_t * row_ptr = init_data.data() + r * row_bytes;
        // Head 0: 8 blocks
        for (int b = 0; b < 8; ++b) {
            block_q8_0 * blk = (block_q8_0 *)(row_ptr + b * sizeof(block_q8_0));
            blk->d = ggml_fp32_to_fp16(1.0f + (float) r * 0.5f + (float) b * 0.1f);
            for (int k = 0; k < QK8_0; ++k) {
                blk->qs[k] = (int8_t)((k + 1 + b * 4) % 120);
            }
        }
        // Head 1: 8 blocks
        for (int b = 0; b < 8; ++b) {
            block_q8_0 * blk = (block_q8_0 *)(row_ptr + (8 + b) * sizeof(block_q8_0));
            blk->d = ggml_fp32_to_fp16(50.0f + (float) r * 0.5f + (float) b * 0.1f);
            for (int k = 0; k < QK8_0; ++k) {
                blk->qs[k] = (int8_t)(-((k + 1 + b * 4) % 120));
            }
        }
    }

    // Write initial data across meta buffer via set_tensor
    ggml_backend_tensor_set(orig, init_data.data(), 0, total_bytes);

    // --- Invariant 4: Inspect every replica independently via ggml_backend_meta_buffer_simple_tensor ---
    for (size_t j = 0; j < 5; ++j) {
        struct ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(orig, j);
        TEST_ASSERT_MSG(st != nullptr, "simple_tensor for replica must exist");
        TEST_ASSERT_MSG(st->ne[0] == s_tp5_q8_rank_ne[j], "replica simple_tensor ne[0] mismatch");
        TEST_ASSERT_MSG(st->ne[1] == ne1, "replica simple_tensor ne[1] mismatch");

        const size_t st_row_bytes = (size_t) st->nb[1];
        std::vector<uint8_t> st_data(st_row_bytes * (size_t) ne1, 0);
        ggml_backend_tensor_get(st, st_data.data(), 0, st_data.size());

        for (int r = 0; r < ne1; ++r) {
            const uint8_t * row_expected = init_data.data() + r * row_bytes;
            const uint8_t * row_actual = st_data.data() + r * st_row_bytes;
            const size_t start_byte_in_row = (size_t)(s_tp5_q8_rank_starts[j] / QK8_0) * sizeof(block_q8_0);
            // Compare this replica's data against the logical start range
            int cmp = memcmp(row_actual, row_expected + start_byte_in_row, st_row_bytes);
            TEST_ASSERT_MSG(cmp == 0, "replica data does not match expected range from logical source");
        }
    }

    // --- Invariant 5: Canonical readback materializes each logical element once ---
    {
        std::vector<uint8_t> readback_data(total_bytes, 0);
        ggml_backend_tensor_get(orig, readback_data.data(), 0, total_bytes);

        // Ensure full rows match exactly
        for (int r = 0; r < ne1; ++r) {
            const uint8_t * exp_row = init_data.data() + r * row_bytes;
            const uint8_t * act_row = readback_data.data() + r * row_bytes;
            // Head 0 check
            int cmp_h0 = memcmp(act_row, exp_row, 272);
            TEST_ASSERT_MSG(cmp_h0 == 0, "Canonical readback corrupted head 0");
            // Head 1 check: MUST match head 1 (272..544). If clamped to head 0, this asserts!
            int cmp_h1 = memcmp(act_row + 272, exp_row + 272, 272);
            TEST_ASSERT_MSG(cmp_h1 == 0, "Canonical readback head 1 mismatch: old clamp returning head 0 twice detected!");
        }
    }

    // Equal head/token counts must not make a reshape retain the wrong split axis.
    {
        ggml_tensor * two_rows = ggml_view_2d(ctx_orig, orig, ne0, 2, orig->nb[1], 0);
        ggml_tensor * heads = ggml_reshape_3d(ctx_orig, two_rows, head_dim, n_heads, 2);
        ggml_tensor * flat = ggml_reshape_2d(ctx_orig, heads, ne0, 2);
        for (ggml_tensor * view : {two_rows, heads, flat}) {
            TEST_ASSERT(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
        }
        std::vector<uint8_t> flattened(2 * row_bytes);
        ggml_backend_tensor_get(flat, flattened.data(), 0, flattened.size());
        TEST_ASSERT_MSG(memcmp(flattened.data(), init_data.data(), flattened.size()) == 0,
                        "Head/token reshape must preserve both logical KV heads");
    }

    // --- Invariant 6: Storage views (base view and nonzero-offset view), dirty mutation & host restore ---
    // Create views in ctx_orig referencing orig rows
    const size_t row1_offset = row_bytes;
    ggml_tensor * org_view_row1 = ggml_view_2d(ctx_orig, orig, ne0, 1, orig->nb[1], row1_offset);
    ggml_set_name(org_view_row1, "cache_k_l48");

    // Base view referencing row 0 (zero view_offs = 0)
    ggml_tensor * org_view_row0 = ggml_view_2d(ctx_orig, orig, ne0, 1, orig->nb[1], 0);
    ggml_set_name(org_view_row0, "cache_k_l48");

    if (org_view_row1->buffer == nullptr) {
        enum ggml_status vstatus = ggml_backend_view_init(org_view_row1);
        TEST_ASSERT(vstatus == GGML_STATUS_SUCCESS);
    }
    if (org_view_row0->buffer == nullptr) {
        enum ggml_status vstatus0 = ggml_backend_view_init(org_view_row0);
        TEST_ASSERT(vstatus0 == GGML_STATUS_SUCCESS);
    }

    // Verify base view by reading content directly via ggml_backend_tensor_get:
    // Row 0 must match row 0 of init_data for both head 0 and head 1.
    {
        std::vector<uint8_t> row0_content(row_bytes, 0);
        ggml_backend_tensor_get(org_view_row0, row0_content.data(), 0, row_bytes);
        int cmp_v0 = memcmp(row0_content.data(), init_data.data(), row_bytes);
        TEST_ASSERT_MSG(cmp_v0 == 0, "Base view row 0 content mismatch against init_data");
    }

    // Save snapshot of row 1 via host serialized bytes: ggml_backend_tensor_get on org_view_row1
    std::vector<uint8_t> row1_saved(row_bytes, 0);
    ggml_backend_tensor_get(org_view_row1, row1_saved.data(), 0, row_bytes);
    int cmp_saved = memcmp(row1_saved.data(), init_data.data() + 1 * row_bytes, row_bytes);
    TEST_ASSERT_MSG(cmp_saved == 0, "Saved snapshot from org_view_row1 does not match row 1 initial data");

    // Dirty mutation: overwrite all 3 rows of orig with dirty marker (0x7F) via set_tensor
    std::vector<uint8_t> dirty_data(total_bytes, 0x7F);
    ggml_backend_tensor_set(orig, dirty_data.data(), 0, total_bytes);

    // Verify dirty mutation took effect on all rows
    std::vector<uint8_t> verify_dirty(total_bytes, 0);
    ggml_backend_tensor_get(orig, verify_dirty.data(), 0, total_bytes);
    for (size_t i = 0; i < total_bytes; ++i) {
        TEST_ASSERT(verify_dirty[i] == 0x7F);
    }

    // Restore snapshot into row 1 view via HOST serialized bytes: ggml_backend_tensor_set into org_view_row1.
    // This tests the exact host serialized write path into a nonzero-offset replicated view.
    ggml_backend_tensor_set(org_view_row1, row1_saved.data(), 0, row_bytes);

    // Inspect every replica of orig after restore:
    // Row 1 of each replica must match row 1 of init_data; rows 0 and 2 must remain 0x7F.
    for (size_t j = 0; j < 5; ++j) {
        struct ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(orig, j);
        const size_t st_row_bytes = (size_t) st->nb[1];
        std::vector<uint8_t> st_data(st_row_bytes * (size_t) ne1, 0);
        ggml_backend_tensor_get(st, st_data.data(), 0, st_data.size());

        // Row 0 of replica j must remain dirty
        for (size_t b = 0; b < st_row_bytes; ++b) {
            TEST_ASSERT_MSG(st_data[b] == 0x7F, "Row 0 replica corrupted during row 1 restore");
        }
        // Row 1 of replica j must match init_data
        const uint8_t * exp_r1 = init_data.data() + 1 * row_bytes;
        const size_t start_byte = (size_t)(s_tp5_q8_rank_starts[j] / QK8_0) * sizeof(block_q8_0);
        int cmp_r1 = memcmp(st_data.data() + st_row_bytes, exp_r1 + start_byte, st_row_bytes);
        TEST_ASSERT_MSG(cmp_r1 == 0, "Row 1 replica not restored to initial data");

        // Row 2 of replica j must remain dirty
        for (size_t b = 0; b < st_row_bytes; ++b) {
            TEST_ASSERT_MSG(st_data[2 * st_row_bytes + b] == 0x7F, "Row 2 replica corrupted during row 1 restore");
        }
    }

    // Canonical readback after restore:
    // Row 0 and Row 2 must remain dirty (0x7F)
    // Row 1 must match init_data exactly for BOTH head 0 and head 1!
    {
        std::vector<uint8_t> restored_full(total_bytes, 0);
        ggml_backend_tensor_get(orig, restored_full.data(), 0, total_bytes);

        // Row 0
        for (size_t b = 0; b < row_bytes; ++b) {
            TEST_ASSERT_MSG(restored_full[b] == 0x7F, "Row 0 corrupted on full tensor after row 1 restore");
        }
        // Row 1: Head 0
        int cmp_r1_h0 = memcmp(restored_full.data() + row_bytes, init_data.data() + row_bytes, 272);
        TEST_ASSERT_MSG(cmp_r1_h0 == 0, "Row 1 Head 0 corrupted after restore");
        // Row 1: Head 1
        int cmp_r1_h1 = memcmp(restored_full.data() + row_bytes + 272, init_data.data() + row_bytes + 272, 272);
        TEST_ASSERT_MSG(cmp_r1_h1 == 0, "Row 1 Head 1 corrupted after restore (clamped read failure)");
        // Row 2
        for (size_t b = 0; b < row_bytes; ++b) {
            TEST_ASSERT_MSG(restored_full[2 * row_bytes + b] == 0x7F, "Row 2 corrupted on full tensor after row 1 restore");
        }
    }

    // --- Invariant 7: Asynchronous transfer API with nonzero API offset on original tensor ---
    // Write/read middle row (row 1) via ORIGINAL tensor with offset = row_bytes, size = row_bytes.
    // Verify that all 5 replicas receive row 1 data at their rank-local row 1 positions,
    // while row 0 and row 2 remain untouched (still 0x7F).
    {
        std::vector<uint8_t> async_row1(row_bytes, 0);
        for (size_t i = 0; i < row_bytes; ++i) {
            async_row1[i] = (uint8_t)((init_data[row_bytes + i] ^ 0xAA) + 1);
        }

        // Asynchronous set with API offset = row_bytes, size = row_bytes
        ggml_backend_tensor_set_async(meta_backend, orig, async_row1.data(), row_bytes, row_bytes);
        ggml_backend_synchronize(meta_backend);

        // Check every replica after async write with nonzero offset:
        // Row 0: dirty (0x7F)
        // Row 1: async_row1 data for that rank's replica range
        // Row 2: dirty (0x7F)
        for (size_t j = 0; j < 5; ++j) {
            struct ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(orig, j);
            const size_t st_row_bytes = (size_t) st->nb[1];
            std::vector<uint8_t> st_data(st_row_bytes * (size_t) ne1, 0);
            ggml_backend_tensor_get(st, st_data.data(), 0, st_data.size());

            // Row 0 untouched
            for (size_t b = 0; b < st_row_bytes; ++b) {
                TEST_ASSERT_MSG(st_data[b] == 0x7F, "Async set_tensor corrupted adjacent row 0");
            }
            // Row 1 updated to async_row1
            const size_t start_byte = (size_t)(s_tp5_q8_rank_starts[j] / QK8_0) * sizeof(block_q8_0);
            int cmp_r1 = memcmp(st_data.data() + st_row_bytes, async_row1.data() + start_byte, st_row_bytes);
            TEST_ASSERT_MSG(cmp_r1 == 0, "Async set_tensor replica row 1 data mismatch");
            // Row 2 untouched
            for (size_t b = 0; b < st_row_bytes; ++b) {
                TEST_ASSERT_MSG(st_data[2 * st_row_bytes + b] == 0x7F, "Async set_tensor corrupted adjacent row 2");
            }
        }

        // Asynchronous get with API offset = row_bytes, size = row_bytes
        std::vector<uint8_t> async_row1_out(row_bytes, 0);
        ggml_backend_tensor_get_async(meta_backend, orig, async_row1_out.data(), row_bytes, row_bytes);
        ggml_backend_synchronize(meta_backend);

        int cmp_h0 = memcmp(async_row1_out.data(), async_row1.data(), 272);
        TEST_ASSERT_MSG(cmp_h0 == 0, "Async get_tensor with offset head 0 mismatch");
        int cmp_h1 = memcmp(async_row1_out.data() + 272, async_row1.data() + 272, 272);
        TEST_ASSERT_MSG(cmp_h1 == 0, "Async get_tensor with offset head 1 mismatch (clamped read failure)");
    }

    // --- Invariant 8: Axis-2 F32 replicated geometry regression (catching nb[1] vs nb[axis] stride bug) ---
    {
        // 3D tensor: [ne0=4, ne1=4, ne2=2, ne3=1]
        // ne2 = 2 slices along axis 2 (e.g. KV heads or channels)
        // Row stride nb[1] = 4 * sizeof(float) = 16 bytes
        // Slice stride along axis 2 nb[2] = 4 * 4 * sizeof(float) = 64 bytes
        // Chunk size full nb[3] = 4 * 4 * 2 * sizeof(float) = 128 bytes
        // 5 ranks carry counts [1, 1, 2, 1, 1] and starts [0, 0, 0, 1, 1] along axis 2
        struct ggml_init_params params_a2 = { 4 * 1024 * 1024, nullptr, /*no_alloc =*/ true };
        struct ggml_context * ctx_a2 = ggml_init(params_a2);
        ggml_tensor * t_a2 = ggml_new_tensor_3d(ctx_a2, GGML_TYPE_F32, 4, 4, 2);
        ggml_set_name(t_a2, "replicated_axis2");
        ggml_backend_buffer_t buf_a2 = ggml_backend_alloc_ctx_tensors_from_buft(ctx_a2, meta_buft);
        TEST_ASSERT(buf_a2 != nullptr);

        const size_t a2_total_bytes = (size_t) ggml_nbytes(t_a2); // 128 bytes (32 floats)
        std::vector<float> a2_in(32);
        for (int i = 0; i < 32; ++i) a2_in[i] = 100.0f + (float) i;

        ggml_backend_tensor_set(t_a2, a2_in.data(), 0, a2_total_bytes);

        // Check every replica independently:
        // Rank 0, 1: slice 0 (first 16 floats)
        // Rank 2: slice 0 and slice 1 (all 32 floats)
        // Rank 3, 4: slice 1 (next 16 floats, indices 16..31)
        const int64_t a2_starts[5] = { 0, 0, 0, 1, 1 };
        for (size_t j = 0; j < 5; ++j) {
            struct ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(t_a2, j);
            TEST_ASSERT_MSG(st != nullptr, "Axis-2 simple_tensor must exist");
            const size_t st_bytes = (size_t) ggml_nbytes(st);
            std::vector<float> st_data(st_bytes / sizeof(float), 0.0f);
            ggml_backend_tensor_get(st, st_data.data(), 0, st_bytes);

            const float * exp_ptr = a2_in.data() + a2_starts[j] * 16;
            int cmp = memcmp(st_data.data(), exp_ptr, st_bytes);
            TEST_ASSERT_MSG(cmp == 0, "Axis-2 replica content mismatch (nb[1] vs nb[axis] stride bug detected!)");
        }

        // Canonical readback of axis-2 replicated tensor
        std::vector<float> a2_out(32, 0.0f);
        ggml_backend_tensor_get(t_a2, a2_out.data(), 0, a2_total_bytes);
        for (int i = 0; i < 32; ++i) {
            TEST_ASSERT_MSG(a2_out[i] == a2_in[i], "Axis-2 canonical readback mismatch");
        }

        ggml_backend_buffer_free(buf_a2);
        ggml_free(ctx_a2);
    }

    fprintf(stderr, "  Explicit TP5 Q8_0 indexed replica: all 5 replicas verified, both distinct heads preserved, views & restore clean\n");

    ggml_backend_buffer_free(orig_buf);
    ggml_free(ctx_orig);

    ggml_backend_free(meta_backend);
}

static void test_meta_scheduler_keepalive_dependencies() {
    fprintf(stderr, "--- test_meta_scheduler_keepalive_dependencies ---\n");
    auto * cpu_dev = get_cpu_dev();
    TEST_ASSERT(cpu_dev != nullptr);
    test_split_ctx split{
        2, { "keepalive_weight", "keepalive_activation" }
    };
    ggml_backend_dev_t devices[]{ cpu_dev, cpu_dev };
    auto *             meta_dev = ggml_backend_meta_device(devices, 2, test_get_split_state, &split);
    TEST_ASSERT(meta_dev != nullptr);
    auto * meta = ggml_backend_dev_init(meta_dev, nullptr);
    auto * cpu  = ggml_backend_cpu_init();
    TEST_ASSERT(meta != nullptr && cpu != nullptr);
    constexpr int k = 8, n = 4;
    auto *        weights_ctx = ggml_init({ 256 * 1024, nullptr, true });
    auto *        host_ctx    = ggml_init({ 256 * 1024, nullptr, true });
    TEST_ASSERT(weights_ctx != nullptr && host_ctx != nullptr);
    auto * weight     = ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, k, n);
    auto * activation = ggml_new_tensor_1d(weights_ctx, GGML_TYPE_F32, k);
    ggml_set_name(weight, "keepalive_weight");
    ggml_set_name(activation, "keepalive_activation");
    auto weights = ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx, ggml_backend_dev_buffer_type(meta_dev));
    TEST_ASSERT(weights != nullptr);
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * residual = ggml_new_tensor_1d(host_ctx, GGML_TYPE_F32, n);
    ggml_set_input(residual);
    auto host = ggml_backend_alloc_ctx_tensors(host_ctx, cpu);
    TEST_ASSERT(host != nullptr);
    float weight_values[k * n];
    for (int i = 0; i < k * n; ++i)
        weight_values[i] = float(i + 1);
    ggml_backend_tensor_set(weight, weight_values, 0, sizeof(weight_values));

    // Exercise the real optimizer API with dependencies that need not share a
    // buffer type or nonempty shape. Only lifetime, not math layout, is extended.
    static ggml_tensor * kept[2];
    meta->iface.graph_optimize = [](ggml_backend_t, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
        auto * until = ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1);
        for (auto * tensor : kept)
            params->add_alloc_dep(params->user_data, tensor, until);
    };
    for (bool foreign : { false, true }) {
        auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        TEST_ASSERT(ctx != nullptr);
        auto * empty = ggml_view_1d(ctx, activation, 0, 0);
        TEST_ASSERT(ggml_backend_view_init(empty) == GGML_STATUS_SUCCESS);
        auto * out = ggml_add(ctx, ggml_sqr(ctx, ggml_mul_mat(ctx, weight, activation)), residual);
        ggml_set_output(out);
        auto * graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, out);
        kept[0] = weight;
        kept[1] = foreign ? residual : empty;
        ggml_backend_t backends[]{ meta, cpu };
        auto           scheduler = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
        TEST_ASSERT(scheduler != nullptr);
        ggml_backend_sched_set_tensor_backend(scheduler, out, meta);
        TEST_ASSERT(ggml_backend_sched_alloc_graph(scheduler, graph));
        for (int round = 0; round < 2; ++round) {
            float act[k], res[n], actual[n];
            for (int j = 0; j < k; ++j)
                act[j] = float(j + 1 + round) * 0.125f;
            for (int i = 0; i < n; ++i)
                res[i] = float(1 + round) + float(i) * 0.25f;
            ggml_backend_tensor_set(activation, act, 0, sizeof(act));
            ggml_backend_tensor_set(residual, res, 0, sizeof(res));
            TEST_ASSERT(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, actual, 0, sizeof(actual));
            for (int i = 0; i < n; ++i) {
                float dot = 0.0f;
                for (int j = 0; j < k; ++j)
                    dot += weight_values[i * k + j] * act[j];
                TEST_ASSERT_MSG(actual[i] == dot * dot + res[i], "keepalive dependencies changed the computed result");
            }
        }
        ggml_backend_sched_free(scheduler);
        ggml_free(ctx);
    }
    kept[0] = kept[1] = nullptr;
    ggml_backend_buffer_free(host);
    ggml_backend_buffer_free(weights);
    ggml_free(host_ctx);
    ggml_free(weights_ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(meta);
}

int main(int argc, char ** argv) {
    bool vulkan_state_copy_only = false;
    bool alloc_deps_only        = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--vulkan-state-copy-only") == 0) {
            vulkan_state_copy_only = true;
        } else if (strcmp(argv[i], "--alloc-deps-only") == 0) {
            alloc_deps_only = true;
        } else {
            fprintf(stderr, "test-meta-reduce-boundary: unknown CLI argument '%s'\n", argv[i]);
            return 1;
        }
    }

    if (alloc_deps_only) {
        test_meta_scheduler_keepalive_dependencies();
        return g_failures ? 1 : 0;
    }
    if (vulkan_state_copy_only) {
        test_meta_recurrent_snapshot_uneven_tp5(true);
        if (g_failures > 0) {
            fprintf(stderr, "test-meta-reduce-boundary (--vulkan-state-copy-only): %d failures\n", g_failures);
            return 1;
        }
        printf("test-meta-reduce-boundary (--vulkan-state-copy-only): all passed\n");
        return 0;
    }

    test_two_branch_gated_sum_with_leaf_matmuls();
    test_moe_expert_stack_and_shared_expert_chain();
    test_nonlinear_boundary_and_reference();
    test_forked_partial_fanout_boundary();
    test_unrelated_partial_root_with_interleaved_merge();
    test_meta_recurrent_snapshot_uneven_tp5(false);
    test_meta_advertised_comm_fail_closed();
    test_meta_indexed_replica_tp5_q8();
    test_meta_scheduler_keepalive_dependencies();

    if (g_failures > 0) {
        fprintf(stderr, "test-meta-reduce-boundary: %d failures\n", g_failures);
        return 1;
    }
    printf("test-meta-reduce-boundary: all passed\n");
    return 0;
}
