#include "llama-rerot.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

static int g_failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ++g_failures; \
    } \
} while (0)

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    CHECK(a.size() == b.size());
    float result = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

static std::vector<float> random_vector(std::mt19937 & rng, size_t n) {
    std::normal_distribution<float> distribution(0.0f, 0.35f);
    std::vector<float> result(n);
    for (float & value : result) {
        value = distribution(rng);
    }
    return result;
}

static void run_equivalence_case(
        const llama_rerot_rope_config & config,
        uint32_t n_keys,
        uint32_t value_dim,
        const std::vector<llama_rerot_ddvr_span> & spans,
        int64_t query_position,
        uint32_t seed) {
    std::mt19937 rng(seed);
    const auto query = random_vector(rng, config.head_dim);
    const auto keys = random_vector(rng, size_t(n_keys) * config.head_dim);
    const auto values = random_vector(rng, size_t(n_keys) * value_dim);

    const auto materialized = llama_rerot_ddvr_attention_materialized(
        query, keys, values, value_dim, query_position, spans, config);
    const auto qside = llama_rerot_ddvr_attention_qside(
        query, keys, values, value_dim, query_position, spans, config);

    const float error = max_abs_diff(materialized, qside);
    if (error >= 2.5e-5f) {
        std::fprintf(stderr, "equivalence error = %.9g\n", error);
    }
    CHECK(error < 2.5e-5f);
}

static void test_half_rope_multiple_spans() {
    llama_rerot_rope_config config;
    config.head_dim = 32;
    config.rotary_dim = 24;
    config.theta = 10000.0;
    config.layout = llama_rerot_rope_layout::half;

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 3, 0, 0 },
        { 3, 4, 0, 3 },       // overlapping writer-local storage positions
        { 7, 2, 11, 7 },      // private/storage gap before this public run
    };
    run_equivalence_case(config, 9, 13, spans, 9, 0x1001u);
}

static void test_interleaved_rope() {
    llama_rerot_rope_config config;
    config.head_dim = 40;
    config.rotary_dim = 32;
    config.theta = 1000000.0;
    config.freq_scale = 0.75;
    config.layout = llama_rerot_rope_layout::interleaved;

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 1, 17, 0 },
        { 1, 5, 2, 1 },
        { 6, 3, 30, 6 },
    };
    run_equivalence_case(config, 9, 7, spans, 9, 0x2002u);
}

static void test_qwen35_text_imrope() {
    llama_rerot_rope_config config;
    config.head_dim = 256;
    config.rotary_dim = 64;
    config.theta = 10000000.0;
    config.layout = llama_rerot_rope_layout::interleaved;
    config.axis_pair_count = { 11, 11, 10, 0 };

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 4, 0, 0 },
        { 4, 2, 19, 4 },
        { 6, 5, 3, 6 },
    };
    run_equivalence_case(config, 11, 17, spans, 11, 0x350035u);
}

static void test_validation() {
    llama_rerot_rope_config config;
    config.head_dim = 8;
    config.rotary_dim = 8;

    const std::vector<float> query(8, 0.0f);
    const std::vector<float> keys(16, 0.0f);
    const std::vector<float> values(6, 0.0f);
    const std::vector<llama_rerot_ddvr_span> overlapping = {
        { 0, 2, 0, 0 },
        { 1, 1, 0, 2 },
    };

    bool threw = false;
    try {
        (void) llama_rerot_ddvr_attention_qside(
            query, keys, values, 3, 2, overlapping, config);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    std::fprintf(stderr, "=== RERoT DDVR Tests ===\n");
    test_half_rope_multiple_spans();
    test_interleaved_rope();
    test_qwen35_text_imrope();
    test_validation();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

