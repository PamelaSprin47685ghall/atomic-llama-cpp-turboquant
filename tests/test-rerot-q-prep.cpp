// test-rerot-q-prep.cpp — R02 fused Q-prep op vs llama_rerot_q_prep_reference
//
// - CPU: build tiny ggml_rerot_q_prep graph, compute on CPU backend, compare
//   active prefix to reference (atol) and assert NaN poison on [active,capacity).
// - GPU: only when LLAMA_REROT_GPU_Q_PREP=1 and a Vulkan device is present;
//   otherwise print skipped (may remain queued).
// - Modes: NORMAL path exercised; VISION is rejected by the reference/support
//   table (IMROPE=40 per ggml.h).

#include "llama-rerot-math.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cmath>
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

static constexpr float kAtol = 1e-5f;

static std::vector<float> run_q_prep_graph(
        ggml_backend_t backend,
        const std::vector<float> & q_raw,
        const std::vector<int32_t> & q_indices,
        const std::vector<int32_t> & q_pos,
        int64_t head_dim,
        int64_t heads,
        int64_t n_tokens,
        int64_t capacity,
        int32_t active,
        int n_dims,
        int mode,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int * sections) {

    struct ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    struct ggml_tensor * t_q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, n_tokens);
    struct ggml_tensor * t_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
    struct ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity); // n_pos=1

    struct ggml_tensor * t_out = ggml_rerot_q_prep(
        ctx, t_q, t_idx, t_pos, /*freq_factors=*/nullptr,
        active, n_dims, sections, mode, n_ctx_orig,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    CHECK(t_out != nullptr);
    CHECK(t_out->ne[0] == head_dim);
    CHECK(t_out->ne[1] == heads);
    CHECK(t_out->ne[2] == capacity);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    CHECK(ggml_gallocr_alloc_graph(galloc, gf));

    ggml_backend_tensor_set(t_q, q_raw.data(), 0, q_raw.size() * sizeof(float));
    ggml_backend_tensor_set(t_idx, q_indices.data(), 0, q_indices.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_pos, q_pos.data(), 0, q_pos.size() * sizeof(int32_t));

    CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> result(size_t(head_dim) * heads * capacity);
    ggml_backend_tensor_get(t_out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

static void compare_to_reference(
        const std::vector<float> & got,
        const std::vector<float> & ref,
        int64_t head_dim,
        int64_t heads,
        int64_t capacity,
        int64_t active) {
    CHECK(got.size() == ref.size());
    CHECK(got.size() == size_t(head_dim) * heads * capacity);

    for (int64_t g = 0; g < capacity; ++g) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t i = size_t((g * heads + h) * head_dim + d);
                if (g >= active) {
                    CHECK(std::isnan(got[i]));
                    CHECK(std::isnan(ref[i]));
                } else {
                    CHECK(!std::isnan(got[i]));
                    CHECK(!std::isnan(ref[i]));
                    CHECK(std::fabs(got[i] - ref[i]) <= kAtol);
                }
            }
        }
    }
}

int main() {
    ggml_backend_load_all();

    const int64_t head_dim = 8;
    const int64_t heads    = 2;
    const int64_t n_tokens = 5;
    const int64_t capacity = 4;
    const int32_t active   = 2;
    const int     n_rot    = 8;
    const int     mode     = 0; // NORMAL

    std::vector<float> q_raw(size_t(head_dim) * heads * n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                q_raw[size_t((t * heads + h) * head_dim + d)] =
                    0.1f * float(t * heads + h) + 0.01f * float(d);
            }
        }
    }
    std::vector<int32_t> q_indices = { 1, 3, 0, 2 };
    std::vector<int32_t> q_pos     = { 7, 42, 0, 0 };

    llama_rerot_q_prep_contract c;
    c.head_dim  = head_dim;
    c.heads     = heads;
    c.n_tokens  = n_tokens;
    c.capacity  = capacity;
    c.active    = active;
    c.n_pos     = 1;
    c.n_rot     = n_rot;
    c.rope_mode = mode;
    c.freq_base = 10000.0f;

    const auto ref = llama_rerot_q_prep_reference(
        q_raw.data(), q_indices.data(), q_pos.data(), c);

    // Strengthened poison-tail on the reference itself.
    {
        size_t poison = 0;
        for (int64_t g = active; g < capacity; ++g) {
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    CHECK(std::isnan(ref[size_t((g * heads + h) * head_dim + d)]));
                    ++poison;
                }
            }
        }
        CHECK(poison == size_t((capacity - active) * heads * head_dim));
    }

    // Support table: IMROPE=40, VISION=24 rejected.
    {
        const auto modes = llama_rerot_q_prep_supported_modes();
        CHECK(modes.size() == 4);
        CHECK(modes[0] == 0 && modes[1] == 2 && modes[2] == 8 && modes[3] == 40);
        for (int m : modes) {
            CHECK(m != 24);
        }
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    CHECK(cpu_backend != nullptr);

    std::vector<float> cpu_out = run_q_prep_graph(
        cpu_backend, q_raw, q_indices, q_pos,
        head_dim, heads, n_tokens, capacity, active,
        n_rot, mode, /*n_ctx_orig=*/0,
        /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
        /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
        /*beta_fast=*/32.0f, /*beta_slow=*/1.0f,
        /*sections=*/nullptr);

    compare_to_reference(cpu_out, ref, head_dim, heads, capacity, active);
    std::printf("test-rerot-q-prep: CPU op matches llama_rerot_q_prep_reference (atol=%g) + poison-tail.\n",
                double(kAtol));

    // GPU path: env-gated skip unless LLAMA_REROT_GPU_Q_PREP=1 and device present.
    const char * gpu_env = std::getenv("LLAMA_REROT_GPU_Q_PREP");
    const bool want_gpu = gpu_env && std::strcmp(gpu_env, "1") == 0;
#ifdef GGML_USE_VULKAN
    if (want_gpu) {
        ggml_backend_t vk = ggml_backend_vk_init(0);
        if (vk) {
            std::vector<float> vk_out = run_q_prep_graph(
                vk, q_raw, q_indices, q_pos,
                head_dim, heads, n_tokens, capacity, active,
                n_rot, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, nullptr);
            compare_to_reference(vk_out, ref, head_dim, heads, capacity, active);
            std::printf("test-rerot-q-prep: Vulkan GPU op matches reference.\n");
            ggml_backend_free(vk);
        } else {
            std::printf("test-rerot-q-prep: LLAMA_REROT_GPU_Q_PREP=1 but no Vulkan device; skipped.\n");
        }
    } else {
        std::printf("test-rerot-q-prep: GPU path skipped (LLAMA_REROT_GPU_Q_PREP!=1).\n");
    }
#else
    (void) want_gpu;
    std::printf("test-rerot-q-prep: GPU path skipped (GGML_USE_VULKAN not built).\n");
#endif

    ggml_backend_free(cpu_backend);
    std::puts("test-rerot-q-prep: all tests OK");
    return 0;
}
