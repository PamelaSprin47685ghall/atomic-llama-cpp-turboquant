// tests/test-vulkan-command-replay.cpp
// Observable regression tests for Vulkan command replay.
// Exercises real Vulkan backend execution on valid single-token decode graphs
// (w[dim,dim] * x[dim,1] + b, i.e. MUL_MAT batch == 1 so replay is eligible):
// 1. Alternating input data on repeated executions (replayed commands must read new inputs).
// 2. Op parameter (scale) mutation (must invalidate cache, re-record, compute new values).
// 3. Shape mutation (different ne dims get distinct cache entries, both replay correctly).
// 4. Buffer rebind & aligned nonzero view offset with distinct expected values (must invalidate/re-record, still numerically correct).
// 5. Large working set (145 distinct decode subgraphs) executed twice to prove the
//    256-entry cache holds the full TP5 working set without premature eviction.
// 6. Exceed 256 cache capacity with 270 small graphs, then reuse an evicted graph with
//    changed inputs, proving proper resource lifecycle and correct new outputs.
// 7. Adjacent host transfers must not corrupt the earlier tensor; --transfer-only
//    runs this regression without executing any compute graph.
// 8. Split-K scratch growth across decode and non-recordable graphs: a large F32
//    M=32,N=509,K=2112 matmul forces scratch reallocation while the replay pool is live;
//    following decode graphs must re-record under the new generation and replay correctly.
// 9. Multi-expert single-token MUL_MAT_ID replays while expert IDs and inputs change;
//    selected-expert count is not the token-batch axis. --moe-only isolates this case.
// Bounded to single Vulkan device 0; total GPU allocation is kept <2 MiB per fixture
// with minimal CPU context overhead. Replay is verified via backend hit/miss counters
// (ggml_backend_vk_get_replay_stats) and all GPU results are compared against CPU oracle.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CLOSE(val_gpu, val_cpu, tol) do { \
    if (!std::isfinite((double)(val_gpu)) || !std::isfinite((double)(val_cpu))) { \
        fprintf(stderr, "NON-FINITE at %s:%d: gpu=%f, cpu=%f\n", \
                __FILE__, __LINE__, (double)(val_gpu), (double)(val_cpu)); \
        exit(1); \
    } \
    float diff = std::abs((val_gpu) - (val_cpu)); \
    if (!(diff <= (tol))) { \
        fprintf(stderr, "MISMATCH at %s:%d: gpu=%f, cpu=%f, diff=%f > tol=%f\n", \
                __FILE__, __LINE__, (double)(val_gpu), (double)(val_cpu), (double)diff, (double)(tol)); \
        exit(1); \
    } \
} while (0)

#define CHECK_STATUS(st, what) do { \
    if ((st) != GGML_STATUS_SUCCESS) { \
        fprintf(stderr, "FAILED %s at %s:%d: status=%d\n", (what), __FILE__, __LINE__, (int)(st)); \
        exit(1); \
    } \
} while (0)

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "ASSERTION FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

typedef void (*get_replay_stats_t)(ggml_backend_t, uint64_t *, uint64_t *);

struct test_env {
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    get_replay_stats_t get_stats = nullptr;

    bool init() {
        ggml_backend_load_all();
        ggml_backend_reg_t reg_gpu = ggml_backend_reg_by_name("Vulkan");
        if (!reg_gpu) {
            return false;
        }
        if (ggml_backend_reg_dev_count(reg_gpu) < 1) {
            return false;
        }
        // Bounded to single GPU device 0
        ggml_backend_dev_t dev_gpu = ggml_backend_reg_dev_get(reg_gpu, 0);
        if (!dev_gpu) {
            return false;
        }
        backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
        if (!backend_gpu) {
            return false;
        }
        get_stats = (get_replay_stats_t) ggml_backend_reg_get_proc_address(reg_gpu, "ggml_backend_vk_get_replay_stats");
        if (!get_stats) {
            return false;
        }
        ggml_backend_reg_t reg_cpu = ggml_backend_reg_by_name("CPU");
        if (!reg_cpu) {
            return false;
        }
        ggml_backend_dev_t dev_cpu = ggml_backend_reg_dev_get(reg_cpu, 0);
        if (!dev_cpu) {
            return false;
        }
        backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
        if (!backend_cpu) {
            return false;
        }
        return true;
    }

    ~test_env() {
        if (backend_gpu) {
            ggml_backend_free(backend_gpu);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

// Helper to build a valid single-token decode subgraph (w[dim,dim] * x[dim,1] + b[dim,1]).
// no_alloc=true on ggml_init, backend_alloc_ctx_tensors afterwards.
struct decode_subgraph_fixture {
    struct ggml_tensor * w = nullptr;
    struct ggml_tensor * x = nullptr;
    struct ggml_tensor * b = nullptr;
    struct ggml_tensor * out = nullptr;
    struct ggml_cgraph * gf = nullptr;

    void build(struct ggml_context * ctx, int dim = 16) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
        struct ggml_tensor * mm = ggml_mul_mat(ctx, w, x);
        out = ggml_add(ctx, mm, b);
        gf = ggml_new_graph_custom(ctx, 16, false);
        TEST_ASSERT(w && x && b && mm && out && gf);
        ggml_build_forward_expand(gf, out);
    }
};

static size_t decode_subgraph_context_overhead(size_t n_subgraphs) {
    const size_t per_graph = ggml_graph_overhead_custom(16, false) + 4 * ggml_tensor_overhead() + 256;
    return n_subgraphs * per_graph + 64 * 1024;
}

static void check_finite_vec(const std::vector<float> & v, const char * what) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite((double) v[i])) {
            fprintf(stderr, "NON-FINITE %s[%zu]=%f\n", what, i, (double) v[i]);
            exit(1);
        }
    }
}

// Test 1: Repeated execution with alternating inputs on the same decode cgraph.
// Verifies that replayed Vulkan command buffers correctly read the new input values.
// Stats affirm the complete lifecycle sequence: run 0 misses while first-ever scratch
// growth (add_rms_partials) coherently cancels the armed recording, run 1 misses while
// recording, runs 2..9 replay (8 hits). Totals: exactly 2 misses, 8 hits.
static void test_alternating_inputs(test_env & env) {
    printf("Running test_alternating_inputs (verifying decode replay computes new input data)...\n");

    const int dim = 16;
    size_t mem_size = decode_subgraph_context_overhead(1);
    struct ggml_init_params params = { mem_size, nullptr, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    struct ggml_context * ctx_cpu = ggml_init(params);
    TEST_ASSERT(ctx_gpu != nullptr && ctx_cpu != nullptr);

    decode_subgraph_fixture fix_gpu, fix_cpu;
    fix_gpu.build(ctx_gpu, dim);
    fix_cpu.build(ctx_cpu, dim);

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, env.backend_cpu);
    TEST_ASSERT(buf_gpu != nullptr && buf_cpu != nullptr);

    std::vector<float> w_init(dim * dim);
    std::vector<float> b_init(dim, 0.5f);
    for (int i = 0; i < dim * dim; ++i) {
        w_init[i] = ((float)(i % 7) - 3.0f) * 0.1f;
    }
    ggml_backend_tensor_set(fix_gpu.w, w_init.data(), 0, w_init.size() * sizeof(float));
    ggml_backend_tensor_set(fix_cpu.w, w_init.data(), 0, w_init.size() * sizeof(float));
    ggml_backend_tensor_set(fix_gpu.b, b_init.data(), 0, b_init.size() * sizeof(float));
    ggml_backend_tensor_set(fix_cpu.b, b_init.data(), 0, b_init.size() * sizeof(float));

    std::vector<float> x_val(dim);
    std::vector<float> res_gpu(dim);
    std::vector<float> res_cpu(dim);

    uint64_t hits_before = 0, misses_before = 0;
    env.get_stats(env.backend_gpu, &hits_before, &misses_before);

    // Run 10 iterations with alternating input vectors, affirming exact stats at each stage.
    for (int iter = 0; iter < 10; ++iter) {
        float factor = (float)(iter + 1) * 1.25f;
        for (int i = 0; i < dim; ++i) {
            x_val[i] = (float)i * factor + 1.0f;
        }

        ggml_backend_tensor_set(fix_gpu.x, x_val.data(), 0, dim * sizeof(float));
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fix_gpu.gf), "gpu alternating graph_compute");
        ggml_backend_tensor_get(fix_gpu.out, res_gpu.data(), 0, dim * sizeof(float));

        ggml_backend_tensor_set(fix_cpu.x, x_val.data(), 0, dim * sizeof(float));
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fix_cpu.gf), "cpu alternating graph_compute");
        ggml_backend_tensor_get(fix_cpu.out, res_cpu.data(), 0, dim * sizeof(float));

        check_finite_vec(res_gpu, "res_gpu");
        check_finite_vec(res_cpu, "res_cpu");
        for (int i = 0; i < dim; ++i) {
            CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
        }

        uint64_t h = 0, m = 0;
        env.get_stats(env.backend_gpu, &h, &m);
        if (iter == 0) {
            // First-ever graph execution grows scratch and cancels the armed recording.
            TEST_ASSERT(m - misses_before == 1);
            TEST_ASSERT(h == hits_before);
        } else if (iter == 1) {
            // Scratch already sized; this run records. Still a miss, no hit yet.
            TEST_ASSERT(m - misses_before == 2);
            TEST_ASSERT(h == hits_before);
        } else {
            // Replays the run-1 recording; every rerun is exactly one hit, zero new misses.
            TEST_ASSERT(m - misses_before == 2);
            TEST_ASSERT(h - hits_before == (uint64_t)(iter - 1));
        }
    }

    uint64_t hits_after = 0, misses_after = 0;
    env.get_stats(env.backend_gpu, &hits_after, &misses_after);
    // Complete sequence: 2 misses (growth-cancelled + recording) and 8 replay hits.
    TEST_ASSERT(misses_after - misses_before == 2);
    TEST_ASSERT(hits_after - hits_before == 8);
    printf("Replay stats verified: 8 hits, 2 misses (growth-cancel + record).\n");

    ggml_backend_buffer_free(buf_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);
    printf("test_alternating_inputs PASSED.\n");
}

// Test 2: Scale/op_params mutation triggers cache invalidation and produces correct values.
static void test_op_param_mutation(test_env & env) {
    printf("Running test_op_param_mutation...\n");

    const int dim = 16;
    size_t mem_size = decode_subgraph_context_overhead(1);
    struct ggml_init_params params = { mem_size, nullptr, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    struct ggml_context * ctx_cpu = ggml_init(params);
    TEST_ASSERT(ctx_gpu != nullptr && ctx_cpu != nullptr);

    // Decode graph: w[16,16] * x[16,1], then scaled by 2.0f.
    struct ggml_tensor * w_gpu = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_F32, dim, dim);
    struct ggml_tensor * x_gpu = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_F32, dim, 1);
    struct ggml_tensor * mm_gpu = ggml_mul_mat(ctx_gpu, w_gpu, x_gpu);
    struct ggml_tensor * out_gpu = ggml_scale(ctx_gpu, mm_gpu, 2.0f);
    struct ggml_cgraph * gf_gpu = ggml_new_graph_custom(ctx_gpu, 16, false);
    TEST_ASSERT(w_gpu && x_gpu && mm_gpu && out_gpu && gf_gpu);
    ggml_build_forward_expand(gf_gpu, out_gpu);

    struct ggml_tensor * w_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, dim, dim);
    struct ggml_tensor * x_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, dim, 1);
    struct ggml_tensor * mm_cpu = ggml_mul_mat(ctx_cpu, w_cpu, x_cpu);
    struct ggml_tensor * out_cpu = ggml_scale(ctx_cpu, mm_cpu, 2.0f);
    struct ggml_cgraph * gf_cpu = ggml_new_graph_custom(ctx_cpu, 16, false);
    TEST_ASSERT(w_cpu && x_cpu && mm_cpu && out_cpu && gf_cpu);
    ggml_build_forward_expand(gf_cpu, out_cpu);

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, env.backend_cpu);
    TEST_ASSERT(buf_gpu != nullptr && buf_cpu != nullptr);

    std::vector<float> w_val(dim * dim, 1.0f);
    std::vector<float> x_val(dim, 2.0f);
    std::vector<float> res_gpu(dim);
    std::vector<float> res_cpu(dim);

    ggml_backend_tensor_set(w_gpu, w_val.data(), 0, w_val.size() * sizeof(float));
    ggml_backend_tensor_set(w_cpu, w_val.data(), 0, w_val.size() * sizeof(float));
    ggml_backend_tensor_set(x_gpu, x_val.data(), 0, x_val.size() * sizeof(float));
    ggml_backend_tensor_set(x_cpu, x_val.data(), 0, x_val.size() * sizeof(float));

    // Run 1: scale = 2.0f. Output per row = 16 * 1.0 * 2.0 * 2.0 = 64.0f.
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_gpu), "gpu scale graph_compute run1");
    ggml_backend_tensor_get(out_gpu, res_gpu.data(), 0, dim * sizeof(float));

    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, gf_cpu), "cpu scale graph_compute run1");
    ggml_backend_tensor_get(out_cpu, res_cpu.data(), 0, dim * sizeof(float));

    check_finite_vec(res_gpu, "res_gpu run1");
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 64.0f, 1e-4f);
        CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
    }

    // Run 2: replay must hit with unchanged scale = 2.0f.
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_gpu), "gpu scale graph_compute run2");
    ggml_backend_tensor_get(out_gpu, res_gpu.data(), 0, dim * sizeof(float));
    check_finite_vec(res_gpu, "res_gpu run2");
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 64.0f, 1e-4f);
    }

    // Mutate scale to 5.0f: this mutates op_params, which MUST invalidate the replay cache
    // and re-record with the new scale instead of replaying 2.0f.
    float new_scale = 5.0f;
    memcpy(out_gpu->op_params, &new_scale, sizeof(float));
    memcpy(out_cpu->op_params, &new_scale, sizeof(float));

    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_gpu), "gpu scale graph_compute run3");
    ggml_backend_tensor_get(out_gpu, res_gpu.data(), 0, dim * sizeof(float));

    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, gf_cpu), "cpu scale graph_compute run3");
    ggml_backend_tensor_get(out_cpu, res_cpu.data(), 0, dim * sizeof(float));

    // Output per row = 16 * 1.0 * 2.0 * 5.0 = 160.0f.
    check_finite_vec(res_gpu, "res_gpu run3");
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 160.0f, 1e-4f);
        CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
    }

    ggml_backend_buffer_free(buf_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);
    printf("test_op_param_mutation PASSED.\n");
}

// Test 3: Shape mutation gets a distinct cache entry and both shapes replay correctly.
// Builds dim=16 and dim=8 decode graphs, interleaves executions, and verifies
// exact results plus replay hits for both shapes.
static void test_shape_mutation(test_env & env) {
    printf("Running test_shape_mutation...\n");

    size_t mem_size = decode_subgraph_context_overhead(2);
    struct ggml_init_params params = { mem_size, nullptr, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    struct ggml_context * ctx_cpu = ggml_init(params);
    TEST_ASSERT(ctx_gpu != nullptr && ctx_cpu != nullptr);

    decode_subgraph_fixture fix16_gpu, fix16_cpu, fix8_gpu, fix8_cpu;
    fix16_gpu.build(ctx_gpu, 16);
    fix16_cpu.build(ctx_cpu, 16);
    fix8_gpu.build(ctx_gpu, 8);
    fix8_cpu.build(ctx_cpu, 8);

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, env.backend_cpu);
    TEST_ASSERT(buf_gpu != nullptr && buf_cpu != nullptr);

    std::vector<float> w16(16 * 16, 0.5f), b16(16, 1.0f), x16(16, 2.0f);
    std::vector<float> w8(8 * 8, 0.5f), b8(8, 1.0f), x8(8, 2.0f);
    ggml_backend_tensor_set(fix16_gpu.w, w16.data(), 0, w16.size() * sizeof(float));
    ggml_backend_tensor_set(fix16_cpu.w, w16.data(), 0, w16.size() * sizeof(float));
    ggml_backend_tensor_set(fix16_gpu.b, b16.data(), 0, b16.size() * sizeof(float));
    ggml_backend_tensor_set(fix16_cpu.b, b16.data(), 0, b16.size() * sizeof(float));
    ggml_backend_tensor_set(fix16_gpu.x, x16.data(), 0, x16.size() * sizeof(float));
    ggml_backend_tensor_set(fix16_cpu.x, x16.data(), 0, x16.size() * sizeof(float));
    ggml_backend_tensor_set(fix8_gpu.w, w8.data(), 0, w8.size() * sizeof(float));
    ggml_backend_tensor_set(fix8_cpu.w, w8.data(), 0, w8.size() * sizeof(float));
    ggml_backend_tensor_set(fix8_gpu.b, b8.data(), 0, b8.size() * sizeof(float));
    ggml_backend_tensor_set(fix8_cpu.b, b8.data(), 0, b8.size() * sizeof(float));
    ggml_backend_tensor_set(fix8_gpu.x, x8.data(), 0, x8.size() * sizeof(float));
    ggml_backend_tensor_set(fix8_cpu.x, x8.data(), 0, x8.size() * sizeof(float));

    uint64_t hits_before = 0, misses_before = 0;
    env.get_stats(env.backend_gpu, &hits_before, &misses_before);

    std::vector<float> res_gpu(16), res_cpu(16);
    // Interleave dim=16 and dim=8 three times each: first pass records (2 misses),
    // subsequent passes replay (4 hits).
    for (int round = 0; round < 3; ++round) {
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fix16_gpu.gf), "gpu dim16 graph_compute");
        ggml_backend_tensor_get(fix16_gpu.out, res_gpu.data(), 0, 16 * sizeof(float));
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fix16_cpu.gf), "cpu dim16 graph_compute");
        ggml_backend_tensor_get(fix16_cpu.out, res_cpu.data(), 0, 16 * sizeof(float));
        check_finite_vec(res_gpu, "res_gpu dim16");
        for (int i = 0; i < 16; ++i) {
            // row = 16 * 0.5 * 2.0 + 1.0 = 17.0f
            CHECK_CLOSE(res_gpu[i], 17.0f, 1e-4f);
            CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
        }

        CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fix8_gpu.gf), "gpu dim8 graph_compute");
        ggml_backend_tensor_get(fix8_gpu.out, res_gpu.data(), 0, 8 * sizeof(float));
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fix8_cpu.gf), "cpu dim8 graph_compute");
        ggml_backend_tensor_get(fix8_cpu.out, res_cpu.data(), 0, 8 * sizeof(float));
        check_finite_vec(res_gpu, "res_gpu dim8");
        for (int i = 0; i < 8; ++i) {
            // row = 8 * 0.5 * 2.0 + 1.0 = 9.0f
            CHECK_CLOSE(res_gpu[i], 9.0f, 1e-4f);
            CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
        }
    }

    uint64_t hits_after = 0, misses_after = 0;
    env.get_stats(env.backend_gpu, &hits_after, &misses_after);
    // 2 distinct shapes recorded (2 misses), remaining 4 executions replayed (4 hits).
    TEST_ASSERT(misses_after - misses_before == 2);
    TEST_ASSERT(hits_after - hits_before == 4);

    ggml_backend_buffer_free(buf_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);
    printf("test_shape_mutation PASSED (distinct ne dims hold distinct replay entries).\n");
}

// Test 4: Same-graph view offset rebind invalidation.
// Evaluates a single decode graph on view window 0 (offset 0), verifies replay hit,
// then rebinds the SAME graph's view tensor to window 1 (aligned offset 256 bytes) with
// distinct expected values. Asserts cache invalidation (miss), fresh re-record, and subsequent replay hit.
static void test_view_offset_rebind(test_env & env) {
    printf("Running test_view_offset_rebind (same-graph view offset rebind invalidation)...\n");

    const int dim = 16;
    const size_t align_floats = 64;
    const size_t offset_bytes = align_floats * sizeof(float);

    size_t mem_size = decode_subgraph_context_overhead(1) + 2 * ggml_tensor_overhead();
    struct ggml_init_params params = { mem_size, nullptr, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    struct ggml_context * ctx_cpu = ggml_init(params);
    TEST_ASSERT(ctx_gpu != nullptr && ctx_cpu != nullptr);

    struct ggml_tensor * x_parent_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, align_floats + dim);
    struct ggml_tensor * w_gpu = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_F32, dim, dim);
    struct ggml_tensor * x_view_gpu = ggml_view_2d(ctx_gpu, x_parent_gpu, dim, 1, dim * sizeof(float), 0);
    struct ggml_tensor * mm_gpu = ggml_mul_mat(ctx_gpu, w_gpu, x_view_gpu);
    struct ggml_cgraph * gf_gpu = ggml_new_graph_custom(ctx_gpu, 16, false);
    TEST_ASSERT(x_parent_gpu && w_gpu && x_view_gpu && mm_gpu && gf_gpu);
    ggml_build_forward_expand(gf_gpu, mm_gpu);

    struct ggml_tensor * x_parent_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, align_floats + dim);
    struct ggml_tensor * w_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, dim, dim);
    struct ggml_tensor * x_view_cpu = ggml_view_2d(ctx_cpu, x_parent_cpu, dim, 1, dim * sizeof(float), 0);
    struct ggml_tensor * mm_cpu = ggml_mul_mat(ctx_cpu, w_cpu, x_view_cpu);
    struct ggml_cgraph * gf_cpu = ggml_new_graph_custom(ctx_cpu, 16, false);
    TEST_ASSERT(x_parent_cpu && w_cpu && x_view_cpu && mm_cpu && gf_cpu);
    ggml_build_forward_expand(gf_cpu, mm_cpu);

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, env.backend_cpu);
    TEST_ASSERT(buf_gpu != nullptr && buf_cpu != nullptr);

    std::vector<float> w_val(dim * dim, 1.0f);
    std::vector<float> parent_val(align_floats + dim, 0.0f);
    for (int i = 0; i < dim; ++i) {
        parent_val[i] = 2.0f;
        parent_val[align_floats + i] = 7.0f;
    }

    ggml_backend_tensor_set(w_gpu, w_val.data(), 0, w_val.size() * sizeof(float));
    ggml_backend_tensor_set(w_cpu, w_val.data(), 0, w_val.size() * sizeof(float));
    ggml_backend_tensor_set(x_parent_gpu, parent_val.data(), 0, parent_val.size() * sizeof(float));
    ggml_backend_tensor_set(x_parent_cpu, parent_val.data(), 0, parent_val.size() * sizeof(float));

    std::vector<float> res_gpu(dim);
    std::vector<float> res_cpu(dim);

    uint64_t hits0 = 0, misses0 = 0;
    env.get_stats(env.backend_gpu, &hits0, &misses0);

    // Pass 1: Run graph on window 0 (view_offs = 0). Record (miss). Expected = 16 * 1.0 * 2.0 = 32.0f.
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_gpu), "gpu view0 run1");
    ggml_backend_tensor_get(mm_gpu, res_gpu.data(), 0, dim * sizeof(float));
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, gf_cpu), "cpu view0 run1");
    ggml_backend_tensor_get(mm_cpu, res_cpu.data(), 0, dim * sizeof(float));
    check_finite_vec(res_gpu, "res_gpu view0 run1");
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 32.0f, 1e-4f);
        CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
    }

    // Pass 2: Replay window 0 on same graph. Must hit replay.
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_gpu), "gpu view0 run2");
    ggml_backend_tensor_get(mm_gpu, res_gpu.data(), 0, dim * sizeof(float));
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 32.0f, 1e-4f);
    }

    uint64_t hits1 = 0, misses1 = 0;
    env.get_stats(env.backend_gpu, &hits1, &misses1);
    TEST_ASSERT(misses1 - misses0 == 1);
    TEST_ASSERT(hits1 - hits0 == 1);

    // Pass 3: Rebind the SAME graph's view tensor to window 1 (view_offs = offset_bytes).
    // This MUST invalidate the existing replay entry on the same graph, causing a miss + re-record.
    x_view_gpu->view_offs = offset_bytes;
    x_view_gpu->data = (char *) x_parent_gpu->data + offset_bytes;
    x_view_cpu->view_offs = offset_bytes;
    x_view_cpu->data = (char *) x_parent_cpu->data + offset_bytes;

    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_gpu), "gpu view1 run3");
    ggml_backend_tensor_get(mm_gpu, res_gpu.data(), 0, dim * sizeof(float));
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, gf_cpu), "cpu view1 run3");
    ggml_backend_tensor_get(mm_cpu, res_cpu.data(), 0, dim * sizeof(float));
    check_finite_vec(res_gpu, "res_gpu view1 run3");
    // Expected = 16 * 1.0 * 7.0 = 112.0f.
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 112.0f, 1e-4f);
        CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
    }

    uint64_t hits2 = 0, misses2 = 0;
    env.get_stats(env.backend_gpu, &hits2, &misses2);
    TEST_ASSERT(misses2 - misses1 == 1); // Invalidated and re-recorded
    TEST_ASSERT(hits2 == hits1);

    // Pass 4: Replay window 1 on same graph. Must hit replay.
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_gpu), "gpu view1 run4");
    ggml_backend_tensor_get(mm_gpu, res_gpu.data(), 0, dim * sizeof(float));
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 112.0f, 1e-4f);
    }

    uint64_t hits3 = 0, misses3 = 0;
    env.get_stats(env.backend_gpu, &hits3, &misses3);
    TEST_ASSERT(hits3 - hits2 == 1); // Replay hit on window 1
    TEST_ASSERT(misses3 == misses2);

    ggml_backend_buffer_free(buf_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);
    printf("test_view_offset_rebind PASSED.\n");
}

// Test 5: Large working set of 145 distinct decode subgraphs executed repeatedly.
// Verifies that the replay cache capacity of 256 entries holds all 145 target decode subgraphs
// without premature eviction, and every subgraph executes correctly with replay hits.
// Memory is kept small: 145 small fixtures, total GPU memory <512 KB, CPU context <2 MiB.
static void test_large_subgraph_working_set(test_env & env) {
    const int NUM_SUBGRAPHS = 145;
    printf("Running test_large_subgraph_working_set with %d distinct decode subgraphs...\n", NUM_SUBGRAPHS);

    const int dim = 16;
    size_t mem_size = decode_subgraph_context_overhead(NUM_SUBGRAPHS);
    struct ggml_init_params params = { mem_size, nullptr, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    struct ggml_context * ctx_cpu = ggml_init(params);
    TEST_ASSERT(ctx_gpu != nullptr && ctx_cpu != nullptr);

    std::vector<decode_subgraph_fixture> fixs_gpu(NUM_SUBGRAPHS);
    std::vector<decode_subgraph_fixture> fixs_cpu(NUM_SUBGRAPHS);

    for (int s = 0; s < NUM_SUBGRAPHS; ++s) {
        fixs_gpu[s].build(ctx_gpu, dim);
        fixs_cpu[s].build(ctx_cpu, dim);
    }

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, env.backend_cpu);
    TEST_ASSERT(buf_gpu != nullptr && buf_cpu != nullptr);

    std::vector<float> w_data(dim * dim);
    std::vector<float> b_data(dim);
    std::vector<float> x_data(dim);

    for (int s = 0; s < NUM_SUBGRAPHS; ++s) {
        for (int i = 0; i < dim * dim; ++i) w_data[i] = 0.05f * (float)((s + i) % 5);
        for (int i = 0; i < dim; ++i) {
            b_data[i] = 0.1f * (float)(s % 3);
            x_data[i] = 1.0f + 0.01f * (float)s;
        }
        ggml_backend_tensor_set(fixs_gpu[s].w, w_data.data(), 0, w_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_cpu[s].w, w_data.data(), 0, w_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_gpu[s].b, b_data.data(), 0, b_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_cpu[s].b, b_data.data(), 0, b_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_gpu[s].x, x_data.data(), 0, x_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_cpu[s].x, x_data.data(), 0, x_data.size() * sizeof(float));
    }

    uint64_t hits_start = 0, misses_start = 0;
    env.get_stats(env.backend_gpu, &hits_start, &misses_start);

    // Pass 1: warm up all 145 subgraphs (records all 145 replay entries into cache).
    for (int s = 0; s < NUM_SUBGRAPHS; ++s) {
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fixs_gpu[s].gf), "gpu warmup graph_compute");
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fixs_cpu[s].gf), "cpu warmup graph_compute");
    }

    uint64_t hits_pass1 = 0, misses_pass1 = 0;
    env.get_stats(env.backend_gpu, &hits_pass1, &misses_pass1);
    // Pass 1 must record all 145 without hits.
    TEST_ASSERT(misses_pass1 - misses_start == NUM_SUBGRAPHS);
    TEST_ASSERT(hits_pass1 == hits_start);

    // Pass 2: replay all 145 subgraphs (must all hit cache with zero additional misses).
    std::vector<float> res_gpu(dim);
    std::vector<float> res_cpu(dim);
    for (int s = 0; s < NUM_SUBGRAPHS; ++s) {
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fixs_gpu[s].gf), "gpu replay graph_compute");
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fixs_cpu[s].gf), "cpu replay graph_compute");
        ggml_backend_tensor_get(fixs_gpu[s].out, res_gpu.data(), 0, dim * sizeof(float));
        ggml_backend_tensor_get(fixs_cpu[s].out, res_cpu.data(), 0, dim * sizeof(float));
        check_finite_vec(res_gpu, "res_gpu working set");
        check_finite_vec(res_cpu, "res_cpu working set");
        for (int i = 0; i < dim; ++i) {
            CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
        }
    }

    uint64_t hits_pass2 = 0, misses_pass2 = 0;
    env.get_stats(env.backend_gpu, &hits_pass2, &misses_pass2);
    // Pass 2 must record exactly 145 hits and 0 new misses.
    TEST_ASSERT(hits_pass2 - hits_pass1 == NUM_SUBGRAPHS);
    TEST_ASSERT(misses_pass2 == misses_pass1);
    printf("Large working set replay verified: %d/145 hits on pass 2, 0 evictions.\n", NUM_SUBGRAPHS);

    ggml_backend_buffer_free(buf_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);
    printf("test_large_subgraph_working_set PASSED.\n");
}

// Test 6: Capacity overflow & eviction recovery.
// Generates 270 distinct decode subgraphs to exceed the 256-entry cache capacity limit.
// Then re-executes early subgraphs (which were evicted) with modified input values,
// proving cache invalidation and clean re-recording after capacity eviction:
// 1. Re-recording evicted graph 0 registers an actual miss and computes correct new outputs.
// 2. Subsequent execution of graph 0 hits the newly recorded replay entry.
// Total GPU allocation is bounded <1 MiB, CPU context <4 MiB.
static void test_capacity_eviction_and_reuse(test_env & env) {
    const int NUM_GRAPHS = 270;
    printf("Running test_capacity_eviction_and_reuse with %d graphs (exceeding 256 cache capacity)...\n", NUM_GRAPHS);

    const int dim = 16;
    size_t mem_size = decode_subgraph_context_overhead(NUM_GRAPHS);
    struct ggml_init_params params = { mem_size, nullptr, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    struct ggml_context * ctx_cpu = ggml_init(params);
    TEST_ASSERT(ctx_gpu != nullptr && ctx_cpu != nullptr);

    std::vector<decode_subgraph_fixture> fixs_gpu(NUM_GRAPHS);
    std::vector<decode_subgraph_fixture> fixs_cpu(NUM_GRAPHS);
    for (int s = 0; s < NUM_GRAPHS; ++s) {
        fixs_gpu[s].build(ctx_gpu, dim);
        fixs_cpu[s].build(ctx_cpu, dim);
    }

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, env.backend_cpu);
    TEST_ASSERT(buf_gpu != nullptr && buf_cpu != nullptr);

    std::vector<float> w_data(dim * dim, 0.25f);
    std::vector<float> b_data(dim, 1.0f);
    std::vector<float> x_data(dim, 2.0f);

    for (int s = 0; s < NUM_GRAPHS; ++s) {
        ggml_backend_tensor_set(fixs_gpu[s].w, w_data.data(), 0, w_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_cpu[s].w, w_data.data(), 0, w_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_gpu[s].b, b_data.data(), 0, b_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_cpu[s].b, b_data.data(), 0, b_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_gpu[s].x, x_data.data(), 0, x_data.size() * sizeof(float));
        ggml_backend_tensor_set(fixs_cpu[s].x, x_data.data(), 0, x_data.size() * sizeof(float));
    }

    // Execute all 270 graphs sequentially.
    // Graphs 0..255 fill the cache; graph 256 triggers capacity eviction of old entries.
    for (int s = 0; s < NUM_GRAPHS; ++s) {
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fixs_gpu[s].gf), "gpu fill graph_compute");
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fixs_cpu[s].gf), "cpu fill graph_compute");
    }

    uint64_t hits_before_evicted = 0, misses_before_evicted = 0;
    env.get_stats(env.backend_gpu, &hits_before_evicted, &misses_before_evicted);

    // Now re-execute graph 0 (which was evicted) with NEW input values.
    // Expected output: row = 16 * 0.25 * 9.0 + 1.0 = 37.0f
    std::vector<float> new_x(dim, 9.0f);
    ggml_backend_tensor_set(fixs_gpu[0].x, new_x.data(), 0, dim * sizeof(float));
    ggml_backend_tensor_set(fixs_cpu[0].x, new_x.data(), 0, dim * sizeof(float));

    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fixs_gpu[0].gf), "gpu evicted graph_compute");
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fixs_cpu[0].gf), "cpu evicted graph_compute");

    uint64_t hits_after_evicted = 0, misses_after_evicted = 0;
    env.get_stats(env.backend_gpu, &hits_after_evicted, &misses_after_evicted);
    // Graph 0 was evicted when capacity exceeded 256; re-executing it MUST register a miss (re-record)
    TEST_ASSERT(misses_after_evicted - misses_before_evicted == 1);
    TEST_ASSERT(hits_after_evicted == hits_before_evicted);

    std::vector<float> res_gpu(dim);
    std::vector<float> res_cpu(dim);
    ggml_backend_tensor_get(fixs_gpu[0].out, res_gpu.data(), 0, dim * sizeof(float));
    ggml_backend_tensor_get(fixs_cpu[0].out, res_cpu.data(), 0, dim * sizeof(float));

    check_finite_vec(res_gpu, "res_gpu evicted reuse");
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 37.0f, 1e-4f);
        CHECK_CLOSE(res_gpu[i], res_cpu[i], 1e-4f);
    }

    // Replay graph 0 a second time: should hit the re-recorded cache and produce 37.0f
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fixs_gpu[0].gf), "gpu evicted replay");
    ggml_backend_tensor_get(fixs_gpu[0].out, res_gpu.data(), 0, dim * sizeof(float));
    for (int i = 0; i < dim; ++i) {
        CHECK_CLOSE(res_gpu[i], 37.0f, 1e-4f);
    }

    uint64_t hits_after_replay = 0, misses_after_replay = 0;
    env.get_stats(env.backend_gpu, &hits_after_replay, &misses_after_replay);
    // Replaying graph 0 must register a hit and zero new misses
    TEST_ASSERT(hits_after_replay - hits_after_evicted == 1);
    TEST_ASSERT(misses_after_replay == misses_after_evicted);

    ggml_backend_buffer_free(buf_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);
    printf("test_capacity_eviction_and_reuse PASSED.\n");
}

// Test 7: Deterministic host transfer ordering (producer-precheck regression).
// Writes 10KB input data (in) and 10KB bias data at distinct offsets, then
// reads back in intact. Guards against write/read staging corruption:
// both tensors must retain their own values after the second write.
static void test_host_transfer_ordering(test_env & env) {
    printf("Running test_host_transfer_ordering (10KB in + 10KB bias distinct offsets)...\n");

    const int n_elems = 2560;
    size_t mem_size = 128 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    TEST_ASSERT(ctx_gpu != nullptr);

    struct ggml_tensor * in_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, n_elems);
    struct ggml_tensor * bias_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, n_elems);
    TEST_ASSERT(in_gpu && bias_gpu);

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, env.backend_gpu);
    TEST_ASSERT(buf_gpu != nullptr);

    std::vector<float> in_vals(n_elems);
    std::vector<float> bias_vals(n_elems, 3.0f);
    for (int e = 0; e < n_elems; ++e) {
        in_vals[e] = 0.5f * (float)(((e % 13) * 3) % 19 - 9);
    }

    ggml_backend_tensor_set(in_gpu, in_vals.data(), 0, n_elems * sizeof(float));
    ggml_backend_tensor_set(bias_gpu, bias_vals.data(), 0, n_elems * sizeof(float));

    std::vector<float> readback(n_elems);
    ggml_backend_tensor_get(in_gpu, readback.data(), 0, n_elems * sizeof(float));
    check_finite_vec(readback, "readback in");
    for (int e = 0; e < n_elems; ++e) {
        CHECK_CLOSE(readback[e], in_vals[e], 1e-6f);
    }

    std::vector<float> readback_bias(n_elems);
    ggml_backend_tensor_get(bias_gpu, readback_bias.data(), 0, n_elems * sizeof(float));
    check_finite_vec(readback_bias, "readback bias");
    for (int e = 0; e < n_elems; ++e) {
        CHECK_CLOSE(readback_bias[e], 3.0f, 1e-6f);
    }

    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);
    printf("test_host_transfer_ordering PASSED (in intact after bias write, first 2048 exact).\n");
}

// Test 8: Large split-K preallocation trigger across decode and prefill graphs.
// Runs a recordable tiny decode graph to initialize the replay command pool and cache.
// Then runs a non-recordable F32 M=32, N=509, K=2112 graph that requires larger split-K scratch.
// Proves that mid-graph scratch reallocation in preallocate_buffers does not leave subctx->s
// reset/null across subsequent operations, and following decode graphs still replay correctly.
static void test_split_k_preallocate_lifecycle(test_env & env) {
    printf("Running test_split_k_preallocate_lifecycle (M=32, N=509, K=2112 split-K trigger)...\n");

    // Step 1: Run and cache a standard recordable decode graph, with deterministic weights/bias/x
    // and CPU oracle verification.
    const int decode_dim = 16;
    size_t mem_size_decode = decode_subgraph_context_overhead(1);
    struct ggml_init_params params_decode = { mem_size_decode, nullptr, true };
    struct ggml_context * ctx_decode_gpu = ggml_init(params_decode);
    struct ggml_context * ctx_decode_cpu = ggml_init(params_decode);
    TEST_ASSERT(ctx_decode_gpu != nullptr && ctx_decode_cpu != nullptr);

    decode_subgraph_fixture fix_decode;
    decode_subgraph_fixture fix_decode_ref;
    fix_decode.build(ctx_decode_gpu, decode_dim);
    fix_decode_ref.build(ctx_decode_cpu, decode_dim);
    ggml_backend_buffer_t buf_decode = ggml_backend_alloc_ctx_tensors(ctx_decode_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_decode_ref = ggml_backend_alloc_ctx_tensors(ctx_decode_cpu, env.backend_cpu);
    TEST_ASSERT(buf_decode != nullptr && buf_decode_ref != nullptr);

    std::vector<float> w_decode(decode_dim * decode_dim, 0.5f);
    std::vector<float> b_decode(decode_dim, 1.0f);
    std::vector<float> x_decode(decode_dim, 2.0f);
    ggml_backend_tensor_set(fix_decode.w, w_decode.data(), 0, w_decode.size() * sizeof(float));
    ggml_backend_tensor_set(fix_decode_ref.w, w_decode.data(), 0, w_decode.size() * sizeof(float));
    ggml_backend_tensor_set(fix_decode.b, b_decode.data(), 0, b_decode.size() * sizeof(float));
    ggml_backend_tensor_set(fix_decode_ref.b, b_decode.data(), 0, b_decode.size() * sizeof(float));
    ggml_backend_tensor_set(fix_decode.x, x_decode.data(), 0, x_decode.size() * sizeof(float));
    ggml_backend_tensor_set(fix_decode_ref.x, x_decode.data(), 0, x_decode.size() * sizeof(float));

    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fix_decode.gf), "gpu decode init graph_compute");
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fix_decode_ref.gf), "cpu decode init graph_compute");

    std::vector<float> res0(decode_dim);
    std::vector<float> res0_ref(decode_dim);
    ggml_backend_tensor_get(fix_decode.out, res0.data(), 0, decode_dim * sizeof(float));
    ggml_backend_tensor_get(fix_decode_ref.out, res0_ref.data(), 0, decode_dim * sizeof(float));
    check_finite_vec(res0, "res0 decode init");
    check_finite_vec(res0_ref, "res0_ref decode init");
    for (int i = 0; i < decode_dim; ++i) {
        // row = 16 * 0.5 * 2.0 + 1.0 = 17.0f
        CHECK_CLOSE(res0[i], 17.0f, 1e-4f);
        CHECK_CLOSE(res0[i], res0_ref[i], 1e-4f);
    }

    uint64_t hits_before = 0, misses_before = 0;
    env.get_stats(env.backend_gpu, &hits_before, &misses_before);

    // Step 2: Run an F32 M=32, N=509, K=2112 matmul graph (non-recordable, triggers split-K scratch growth)
    // with deterministic inputs and full GPU vs CPU oracle verification (<5MB GPU allocation).
    const int M = 32;
    const int N = 509;
    const int K = 2112;
    size_t mem_size_splitk = decode_subgraph_context_overhead(1) + ggml_tensor_overhead();
    struct ggml_init_params params_splitk = { mem_size_splitk, nullptr, true };
    struct ggml_context * ctx_splitk_gpu = ggml_init(params_splitk);
    struct ggml_context * ctx_splitk_cpu = ggml_init(params_splitk);
    TEST_ASSERT(ctx_splitk_gpu != nullptr && ctx_splitk_cpu != nullptr);

    struct ggml_tensor * w_splitk = ggml_new_tensor_2d(ctx_splitk_gpu, GGML_TYPE_F32, K, M);
    struct ggml_tensor * x_splitk = ggml_new_tensor_2d(ctx_splitk_gpu, GGML_TYPE_F32, K, N);
    struct ggml_tensor * mm_splitk = ggml_mul_mat(ctx_splitk_gpu, w_splitk, x_splitk);
    struct ggml_cgraph * gf_splitk = ggml_new_graph_custom(ctx_splitk_gpu, 16, false);
    TEST_ASSERT(w_splitk && x_splitk && mm_splitk && gf_splitk);
    ggml_build_forward_expand(gf_splitk, mm_splitk);

    struct ggml_tensor * w_splitk_ref = ggml_new_tensor_2d(ctx_splitk_cpu, GGML_TYPE_F32, K, M);
    struct ggml_tensor * x_splitk_ref = ggml_new_tensor_2d(ctx_splitk_cpu, GGML_TYPE_F32, K, N);
    struct ggml_tensor * mm_splitk_ref = ggml_mul_mat(ctx_splitk_cpu, w_splitk_ref, x_splitk_ref);
    struct ggml_cgraph * gf_splitk_ref = ggml_new_graph_custom(ctx_splitk_cpu, 16, false);
    TEST_ASSERT(w_splitk_ref && x_splitk_ref && mm_splitk_ref && gf_splitk_ref);
    ggml_build_forward_expand(gf_splitk_ref, mm_splitk_ref);

    ggml_backend_buffer_t buf_splitk = ggml_backend_alloc_ctx_tensors(ctx_splitk_gpu, env.backend_gpu);
    ggml_backend_buffer_t buf_splitk_ref = ggml_backend_alloc_ctx_tensors(ctx_splitk_cpu, env.backend_cpu);
    TEST_ASSERT(buf_splitk != nullptr && buf_splitk_ref != nullptr);

    // Exactly-representable integer inputs: every product and every partial sum over
    // K=2112 terms is an integer bounded by 2112*2*8 = 33792 < 2^24, so float accumulation
    // is bit-exact under ANY order (serial CPU, split-K partials, final reduction).
    // Any GPU/CPU deviation is therefore a genuine compute error, not rounding.
    std::vector<float> w_sk(K * M);
    std::vector<float> x_sk(K * N);
    for (int i = 0; i < K * M; ++i) w_sk[i] = (float)((i % 5) - 2);   // -2..2, distinct
    for (int i = 0; i < K * N; ++i) x_sk[i] = (float)((i % 16) - 8);  // -8..7, distinct
    ggml_backend_tensor_set(w_splitk, w_sk.data(), 0, w_sk.size() * sizeof(float));
    ggml_backend_tensor_set(w_splitk_ref, w_sk.data(), 0, w_sk.size() * sizeof(float));
    ggml_backend_tensor_set(x_splitk, x_sk.data(), 0, x_sk.size() * sizeof(float));
    ggml_backend_tensor_set(x_splitk_ref, x_sk.data(), 0, x_sk.size() * sizeof(float));

    // This triggers split-K scratch preallocation while replay_cmd_pool_init is true from step 1!
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, gf_splitk), "gpu split-k trigger graph_compute");
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, gf_splitk_ref), "cpu split-k trigger graph_compute");

    std::vector<float> res_sk(M * N);
    std::vector<float> res_sk_ref(M * N);
    ggml_backend_tensor_get(mm_splitk, res_sk.data(), 0, res_sk.size() * sizeof(float));
    ggml_backend_tensor_get(mm_splitk_ref, res_sk_ref.data(), 0, res_sk.size() * sizeof(float));
    check_finite_vec(res_sk, "res_sk split-k");
    check_finite_vec(res_sk_ref, "res_sk_ref split-k");
    for (size_t i = 0; i < res_sk.size(); ++i) {
        // Integer-exact under all accumulation orders: tight 1e-4 oracle bound.
        CHECK_CLOSE(res_sk[i], res_sk_ref[i], 1e-4f);
    }

    // Step 3: Vary the decode inputs after scratch growth and verify new values (no stale replay).
    std::vector<float> x_decode_new(decode_dim, 9.0f);
    ggml_backend_tensor_set(fix_decode.x, x_decode_new.data(), 0, decode_dim * sizeof(float));
    ggml_backend_tensor_set(fix_decode_ref.x, x_decode_new.data(), 0, decode_dim * sizeof(float));

    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fix_decode.gf), "gpu decode post-split-k run1");
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_cpu, fix_decode_ref.gf), "cpu decode post-split-k run1");
    ggml_backend_tensor_get(fix_decode.out, res0.data(), 0, decode_dim * sizeof(float));
    ggml_backend_tensor_get(fix_decode_ref.out, res0_ref.data(), 0, decode_dim * sizeof(float));
    check_finite_vec(res0, "res0 post-split-k run1");
    for (int i = 0; i < decode_dim; ++i) {
        // row = 16 * 0.5 * 9.0 + 1.0 = 73.0f
        CHECK_CLOSE(res0[i], 73.0f, 1e-4f);
        CHECK_CLOSE(res0[i], res0_ref[i], 1e-4f);
    }

    // Step 4: Run the decode graph once more with the same inputs; verifies replay with new scratch.
    CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, fix_decode.gf), "gpu decode post-split-k run2");
    ggml_backend_tensor_get(fix_decode.out, res0.data(), 0, decode_dim * sizeof(float));
    check_finite_vec(res0, "res0 post-split-k run2");
    for (int i = 0; i < decode_dim; ++i) {
        CHECK_CLOSE(res0[i], 73.0f, 1e-4f);
    }

    uint64_t hits_after = 0, misses_after = 0;
    env.get_stats(env.backend_gpu, &hits_after, &misses_after);
    // Decode init's miss happened before this snapshot. Split-K growth invalidates the
    // entry; run1 re-records under the new scratch generation (one miss), run2 replays it.
    TEST_ASSERT(misses_after - misses_before == 1);
    TEST_ASSERT(hits_after - hits_before == 1);

    ggml_backend_buffer_free(buf_splitk);
    ggml_backend_buffer_free(buf_splitk_ref);
    ggml_free(ctx_splitk_gpu);
    ggml_free(ctx_splitk_cpu);
    ggml_backend_buffer_free(buf_decode);
    ggml_backend_buffer_free(buf_decode_ref);
    ggml_free(ctx_decode_gpu);
    ggml_free(ctx_decode_cpu);

    printf("test_split_k_preallocate_lifecycle PASSED.\n");
}

static void test_moe_decode_replay(test_env & env) {
    constexpr int K = 32, M = 16, experts = 4, selected = 2;
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    TEST_ASSERT(ctx != nullptr);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, experts);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, selected, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected, 1);
    ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, env.backend_gpu);
    TEST_ASSERT(buffer != nullptr);

    std::vector<float> weights(K * M * experts), inputs(K * selected), output(M * selected);
    for (int e = 0; e < experts; ++e) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                weights[k + K * (m + M * e)] = float((e + 1) * ((k + 2 * m) % 5 - 2));
            }
        }
    }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(float));
    uint64_t hits_before = 0, misses_before = 0;
    env.get_stats(env.backend_gpu, &hits_before, &misses_before);
    for (int round = 0; round < 4; ++round) {
        const int32_t routes[selected] = {round % experts, (round + 2) % experts};
        for (int i = 0; i < K * selected; ++i) {
            inputs[i] = float((i + round) % 3 - 1);
        }
        ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));
        ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
        CHECK_STATUS(ggml_backend_graph_compute(env.backend_gpu, graph), "MoE decode replay");
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
        for (int lane = 0; lane < selected; ++lane) {
            for (int m = 0; m < M; ++m) {
                float expected = 0;
                for (int k = 0; k < K; ++k) {
                    expected += weights[k + K * (m + M * routes[lane])] * inputs[k + K * lane];
                }
                CHECK_CLOSE(output[m + M * lane], expected, 1e-4f);
            }
        }
    }
    uint64_t hits_after = 0, misses_after = 0;
    env.get_stats(env.backend_gpu, &hits_after, &misses_after);
    TEST_ASSERT(misses_after - misses_before == 1);
    TEST_ASSERT(hits_after - hits_before == 3);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("test_moe_decode_replay PASSED: changing experts and inputs, 3 replay hits.\n");
}

int main(int argc, char ** argv) {
    const bool transfer_only = argc == 2 && std::strcmp(argv[1], "--transfer-only") == 0;
    const bool moe_only = argc == 2 && std::strcmp(argv[1], "--moe-only") == 0;
    if (argc != 1 && !transfer_only && !moe_only) {
        std::fprintf(stderr, "Usage: %s [--transfer-only|--moe-only]\n", argv[0]);
        return 2;
    }
    // Enable Vulkan command replay for testing.
    setenv("GGML_VK_CMD_REPLAY", "1", 1);

    test_env env;
    if (!env.init()) {
        // Explicit absence notice: no Vulkan device, CPU backend, or replay stats
        // interface. Zero tests ran; this is a SKIP, not a pass.
        fprintf(stderr, "SKIP test-vulkan-command-replay: no usable Vulkan device 0, CPU backend, or replay stats proc (0 tests ran)\n");
        return 1;
    }

    if (transfer_only) {
        test_host_transfer_ordering(env);
        return 0;
    }
    if (moe_only) {
        test_moe_decode_replay(env);
        return 0;
    }

    int tests_run = 0;
    test_alternating_inputs(env);          ++tests_run;
    test_op_param_mutation(env);           ++tests_run;
    test_shape_mutation(env);              ++tests_run;
    test_view_offset_rebind(env);          ++tests_run;
    test_large_subgraph_working_set(env);    ++tests_run;
    test_capacity_eviction_and_reuse(env); ++tests_run;
    test_host_transfer_ordering(env); ++tests_run;
    test_split_k_preallocate_lifecycle(env); ++tests_run;
    test_moe_decode_replay(env); ++tests_run;

    printf("All Vulkan command replay observable integration tests completed successfully (%d tests ran).\n", tests_run);
    return 0;
}
