// test-rerot-q3-shared-kv-2reader.cpp — C10 / Q3 two-reader shared-KV prototype
//
// Offline ceiling (AGENTS C10): shader compile + CPU golden compare.
// Does NOT require 5 GPUs. GPU runtime dispatch is not wired in this stub;
// when no Vulkan device is present the GPU leg is skipped with an explicit
// message (never forged as a pass via tolerance changes).
//
// Existing CPU coverage this harness builds on:
//   - tests/test-rerot-math.cpp::test_q3_shared_block_attention
//       (FP64 oracle, partial visibility, empty reader, merge-order)
//   - tests/test-rerot-shared-block.cpp
//       (DDVR A/B shadow, partial visibility, empty reader, R=1)
//
// Prototype artifacts:
//   - ggml/.../vulkan-shaders/rerot_shared_kv_2reader.comp
//   - registered in vulkan-shaders-gen.cpp as "rerot_shared_kv_2reader"
//   - pipeline create in ggml-vulkan.cpp (not production-dispatched)

#include "llama-rerot-math.h"

#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static double max_abs_diff(const std::vector<double> & a, const std::vector<double> & b) {
    CHECK(a.size() == b.size());
    double result = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

static std::vector<double> random_vector(std::mt19937_64 & rng, size_t n) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> out(n);
    for (double & x : out) {
        x = dist(rng);
    }
    return out;
}

// Independent full-softmax oracle over one reader's visible union (no online
// rescale) — same formulation as test-rerot-math.cpp.
static std::vector<double> full_softmax_oracle(
        const std::vector<double> & queries,
        const std::vector<double> & keys,
        const std::vector<double> & values,
        uint32_t reader,
        uint32_t head_dim,
        uint32_t value_dim,
        double scale) {
    const size_t n_block = keys.size() / head_dim;
    std::vector<double> scores(n_block);
    for (size_t j = 0; j < n_block; ++j) {
        const double * q = queries.data() + size_t(reader) * head_dim;
        const double * k = keys.data() + j * head_dim;
        double dot = 0.0;
        for (uint32_t e = 0; e < head_dim; ++e) {
            dot += q[e] * k[e];
        }
        scores[j] = scale * dot;
    }
    std::vector<double> out(value_dim, 0.0);
    double z = 0.0;
    for (size_t j = 0; j < n_block; ++j) {
        const double w = std::exp(scores[j]);
        z += w;
        for (uint32_t e = 0; e < value_dim; ++e) {
            out[e] += w * values[j * value_dim + e];
        }
    }
    for (uint32_t e = 0; e < value_dim; ++e) {
        out[e] /= z;
    }
    return out;
}

// CPU golden: llama_rerot_shared_block_attention vs independent oracle.
// Cases cover the C10 contract surface for R=2.
static void test_cpu_golden_two_reader_cases() {
    constexpr uint32_t n_readers = 2;
    constexpr uint32_t head_dim = 16;
    constexpr uint32_t value_dim = 8;
    constexpr uint32_t n_block = 32;
    const double scale = 1.0 / std::sqrt(double(head_dim));

    std::mt19937_64 rng(0xc10u);
    const auto queries = random_vector(rng, size_t(n_readers) * head_dim);
    const auto keys = random_vector(rng, size_t(n_block) * head_dim);
    const auto values = random_vector(rng, size_t(n_block) * value_dim);

    struct Case {
        const char * name;
        std::vector<uint8_t> visible; // [2]
        bool expect_empty_block;      // n_block forced to 0
    };
    const Case cases[] = {
        { "both_visible",          {1, 1}, false },
        { "partial_r0_only",       {1, 0}, false },
        { "partial_r1_only",       {0, 1}, false },
        { "neither_visible",       {0, 0}, false },
        { "empty_block_both",      {1, 1}, true  },
    };

    for (const Case & c : cases) {
        std::vector<double> k_use = keys;
        std::vector<double> v_use = values;
        if (c.expect_empty_block) {
            // Empty physical block: API rejects n_block==0; emulate by marking
            // both readers invisible (pipeline stub writes used=0 for NB==0).
            // CPU math path: empty visibility → empty states.
            std::vector<uint8_t> none = {0, 0};
            auto states = llama_rerot_shared_block_attention(
                queries, keys, values, n_readers, head_dim, value_dim, none, scale);
            CHECK(states[0].empty());
            CHECK(states[1].empty());
            std::printf("cpu_golden[%s]: empty states OK\n", c.name);
            continue;
        }

        auto states = llama_rerot_shared_block_attention(
            queries, k_use, v_use, n_readers, head_dim, value_dim, c.visible, scale);

        for (uint32_t r = 0; r < n_readers; ++r) {
            if (c.visible[r] == 0) {
                CHECK(states[r].empty());
                continue;
            }
            CHECK(!states[r].empty());
            const auto got = llama_rerot_attn_state_output(states[r]);
            const auto exp = full_softmax_oracle(
                queries, k_use, v_use, r, head_dim, value_dim, scale);
            const double err = max_abs_diff(got, exp);
            if (err >= 1e-12) {
                std::fprintf(stderr, "cpu_golden[%s] reader %u err=%.3g\n",
                             c.name, r, err);
            }
            CHECK(err < 1e-12);
        }
        std::printf("cpu_golden[%s]: matched independent softmax oracle\n", c.name);
    }

    // Multi-block merge: reader 0 sees only block A, reader 1 sees only block B
    // (partial visibility across tiles — the GPU tile loop's main case).
    {
        const auto keys_a = random_vector(rng, size_t(n_block) * head_dim);
        const auto vals_a = random_vector(rng, size_t(n_block) * value_dim);
        const auto keys_b = random_vector(rng, size_t(n_block) * head_dim);
        const auto vals_b = random_vector(rng, size_t(n_block) * value_dim);
        std::vector<uint8_t> vis_a = {1, 0};
        std::vector<uint8_t> vis_b = {0, 1};

        std::vector<llama_rerot_attn_state> merged(n_readers);
        {
            auto s = llama_rerot_shared_block_attention(
                queries, keys_a, vals_a, n_readers, head_dim, value_dim, vis_a, scale);
            merged[0] = s[0];
            CHECK(s[1].empty());
        }
        {
            auto s = llama_rerot_shared_block_attention(
                queries, keys_b, vals_b, n_readers, head_dim, value_dim, vis_b, scale);
            merged[1] = s[1];
            CHECK(s[0].empty());
        }

        const auto o0 = llama_rerot_attn_state_output(merged[0]);
        const auto e0 = full_softmax_oracle(
            queries, keys_a, vals_a, 0, head_dim, value_dim, scale);
        CHECK(max_abs_diff(o0, e0) < 1e-12);

        const auto o1 = llama_rerot_attn_state_output(merged[1]);
        const auto e1 = full_softmax_oracle(
            queries, keys_b, vals_b, 1, head_dim, value_dim, scale);
        CHECK(max_abs_diff(o1, e1) < 1e-12);
        std::printf("cpu_golden[cross_block_partial]: OK\n");
    }
}

// Offline shader compile evidence: glslc when available. Missing glslc is not
// a test failure — Vulkan build regenerates SPIR-V via vulkan-shaders-gen.
static void try_offline_shader_compile() {
    const char * shader = nullptr;
    const char * candidates[] = {
        "ggml/src/ggml-vulkan/vulkan-shaders/rerot_shared_kv_2reader.comp",
        "../ggml/src/ggml-vulkan/vulkan-shaders/rerot_shared_kv_2reader.comp",
        "../../ggml/src/ggml-vulkan/vulkan-shaders/rerot_shared_kv_2reader.comp",
    };
    // Prefer path relative to source tree via env override.
    if (const char * root = std::getenv("LLAMA_SOURCE_ROOT")) {
        static std::string from_env;
        from_env = std::string(root) + "/ggml/src/ggml-vulkan/vulkan-shaders/rerot_shared_kv_2reader.comp";
        shader = from_env.c_str();
        FILE * f = std::fopen(shader, "rb");
        if (!f) {
            shader = nullptr;
        } else {
            std::fclose(f);
        }
    }
    if (!shader) {
        for (const char * c : candidates) {
            FILE * f = std::fopen(c, "rb");
            if (f) {
                std::fclose(f);
                shader = c;
                break;
            }
        }
    }
    if (!shader) {
        std::printf("shader_compile: source .comp not found from cwd (ok; "
                    "vulkan-shaders-gen registers it at Vulkan build)\n");
        return;
    }

    // Presence of the stub file is the minimum offline artifact check.
    std::printf("shader_compile: found %s\n", shader);

    if (std::system("command -v glslc >/dev/null 2>&1") != 0) {
        std::printf("shader_compile: glslc not on PATH — skipped offline "
                    "SPIR-V (vulkan-shaders-gen covers build-time compile)\n");
        return;
    }

    std::string cmd = std::string("glslc -fshader-stage=compute -o /tmp/rerot_shared_kv_2reader.spv \"")
                    + shader + "\"";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: glslc compile of %s returned %d\n", shader, rc);
        ++g_failures;
        return;
    }
    std::printf("shader_compile: glslc OK -> /tmp/rerot_shared_kv_2reader.spv\n");
}

// GPU leg: skip cleanly when no device. Full Vulkan dispatch of this stub is
// intentionally not required for C10 offline (no GGML_OP yet).
static void try_gpu_leg_or_skip() {
    ggml_backend_load_all();

    ggml_backend_t vk = nullptr;
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
        vk = ggml_backend_dev_init(gpu_dev, nullptr);
    }
#ifdef GGML_USE_VULKAN
    if (!vk && ggml_backend_vk_get_device_count() > 0) {
        vk = ggml_backend_vk_init(0);
    }
#endif

    if (!vk) {
        std::printf("gpu_leg: no GPU/Vulkan device — SKIPPED (C10 offline OK; "
                    "does not require 5 GPUs)\n");
        return;
    }

    // Device present: still no production GGML_OP for this stub. Record
    // "queued for GPU compare" rather than fabricating a dispatch pass.
    std::printf("gpu_leg: Vulkan device available, but C10 stub has no GGML_OP "
                "dispatch yet — GPU numeric compare DEFERRED (pipeline "
                "registered as rerot_shared_kv_2reader; CPU golden is the "
                "offline closer)\n");
    ggml_backend_free(vk);
}

int main() {
    std::printf("=== C10 Q3 two-reader shared-KV prototype ===\n");
    std::printf("Existing CPU tests: test-rerot-math (Q3), test-rerot-shared-block\n");

    test_cpu_golden_two_reader_cases();
    try_offline_shader_compile();
    try_gpu_leg_or_skip();

    if (g_failures != 0) {
        std::fprintf(stderr, "=== %d failure(s) ===\n", g_failures);
        return 1;
    }
    std::puts("test-rerot-q3-shared-kv-2reader: all offline checks OK");
    return 0;
}
