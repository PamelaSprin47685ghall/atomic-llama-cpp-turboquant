#include "llama-rerot.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// ---------------------------------------------------------------------------
// Independent F32 pure-math reference (test-local, never calls the core
// helpers under test). Proves the DDVR identity from scratch:
//
//   materialized:  scores[k] = <R(q_virt) Q, R(v_k) K_k>           (K rephased)
//   q-side:        scores[k] = <R(q_virt + s_k - v_k) Q, R(s_k) K_k> (K不动, Q per span)
//
// Both use ONE global softmax over all visible keys. A per-span softmax
// baseline is provided as a negative control and must differ.
// ---------------------------------------------------------------------------
namespace ddvr_ref {

struct RopeConfig {
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    double theta = 10000.0;
    double freq_scale = 1.0;
    bool interleaved = false;
    uint32_t axis_pairs[4] = {0, 0, 0, 0};
};

static RopeConfig from_core(const llama_rerot_rope_config & cfg) {
    RopeConfig out;
    out.head_dim = cfg.head_dim;
    out.rotary_dim = cfg.rotary_dim;
    out.theta = cfg.theta;
    out.freq_scale = cfg.freq_scale;
    out.interleaved = (cfg.layout == llama_rerot_rope_layout::interleaved);
    for (int i = 0; i < 4; ++i) {
        out.axis_pairs[i] = cfg.axis_pair_count[size_t(i)];
    }
    return out;
}

static uint32_t pair_axis(const RopeConfig & cfg, uint32_t pair) {
    const uint32_t n_pairs = cfg.rotary_dim / 2;
    const uint32_t configured = cfg.axis_pairs[0] + cfg.axis_pairs[1] + cfg.axis_pairs[2] + cfg.axis_pairs[3];
    if (configured == 0) {
        return 0; // 1D RoPE: every pair uses axis 0
    }
    uint32_t offset = 0;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint32_t next = offset + cfg.axis_pairs[axis];
        if (pair < next) {
            return axis;
        }
        offset = next;
    }
    (void) n_pairs;
    return 0;
}

// Explicit 4-coordinate RoPE. Qwen3.5 text passes (p, p, p, 0): the delta is
// applied to the first three coordinates only, the fourth stays zero.
static void rope_apply(float * vec, const RopeConfig & cfg, const int64_t pos[4]) {
    const uint32_t n_pairs = cfg.rotary_dim / 2;
    for (uint32_t pair = 0; pair < n_pairs; ++pair) {
        const uint32_t axis = pair_axis(cfg, pair);
        const double freq = std::pow(cfg.theta, -2.0 * double(pair) / double(cfg.rotary_dim)) * cfg.freq_scale;
        const double angle = double(pos[axis]) * freq;
        const float c = float(std::cos(angle));
        const float s = float(std::sin(angle));
        uint32_t first;
        uint32_t second;
        if (cfg.interleaved) {
            first = pair * 2;
            second = first + 1;
        } else {
            first = pair;
            second = pair + n_pairs;
        }
        const float x = vec[first];
        const float y = vec[second];
        vec[first] = x * c - y * s;
        vec[second] = x * s + y * c;
    }
}

static void rope_apply_text(float * vec, const RopeConfig & cfg, int64_t p) {
    const int64_t pos[4] = {p, p, p, 0};
    rope_apply(vec, cfg, pos);
}

static float dot_f64(const float * a, const float * b, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += double(a[i]) * double(b[i]);
    }
    return float(sum);
}

struct Problem {
    std::vector<float> raw_q; // [head_dim]
    std::vector<float> raw_k; // [n_keys * head_dim]
    std::vector<float> values; // [n_keys * value_dim]
    uint32_t value_dim = 0;
    int64_t query_virtual = 0;
    std::vector<llama_rerot_ddvr_span> spans;
    RopeConfig rope;
    float scale = 0.0f;
};

static std::vector<float> global_softmax_mix(const std::vector<float> & scores,
                                             const std::vector<float> & values,
                                             uint32_t value_dim) {
    std::vector<float> out(value_dim, 0.0f);
    if (scores.empty()) {
        return out;
    }
    const float m = *std::max_element(scores.begin(), scores.end());
    std::vector<double> w(scores.size());
    double denom = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        w[i] = std::exp(double(scores[i] - m));
        denom += w[i];
    }
    for (size_t i = 0; i < scores.size(); ++i) {
        const double weight = w[i] / denom;
        for (uint32_t d = 0; d < value_dim; ++d) {
            out[d] += float(weight * double(values[i * value_dim + d]));
        }
    }
    return out;
}

static std::vector<float> materialized(const Problem & prob) {
    const uint32_t hd = prob.rope.head_dim;
    const uint32_t n_keys = uint32_t(prob.raw_k.size() / hd);
    const float scale = prob.scale != 0.0f ? prob.scale : 1.0f / std::sqrt(float(hd));
    std::vector<float> q(prob.raw_q);
    rope_apply_text(q.data(), prob.rope, prob.query_virtual);
    std::vector<float> scores;
    scores.reserve(n_keys);
    std::vector<float> k(hd);
    for (const auto & span : prob.spans) {
        for (uint32_t local = 0; local < span.key_count; ++local) {
            const uint32_t key = span.key_begin + local;
            std::memcpy(k.data(), prob.raw_k.data() + size_t(key) * hd, size_t(hd) * sizeof(float));
            rope_apply_text(k.data(), prob.rope, span.virtual_pos0 + int64_t(local));
            scores.push_back(scale * dot_f64(q.data(), k.data(), hd));
        }
    }
    return global_softmax_mix(scores, prob.values, prob.value_dim);
}

static std::vector<float> qside(const Problem & prob) {
    const uint32_t hd = prob.rope.head_dim;
    const uint32_t n_keys = uint32_t(prob.raw_k.size() / hd);
    const float scale = prob.scale != 0.0f ? prob.scale : 1.0f / std::sqrt(float(hd));
    std::vector<float> scores;
    scores.reserve(n_keys);
    std::vector<float> q(hd);
    std::vector<float> k(hd);
    for (const auto & span : prob.spans) {
        std::memcpy(q.data(), prob.raw_q.data(), size_t(hd) * sizeof(float));
        // Per-span Q phase: K never moves; the span delta is folded into Q.
        rope_apply_text(q.data(), prob.rope, prob.query_virtual + span.storage_pos0 - span.virtual_pos0);
        for (uint32_t local = 0; local < span.key_count; ++local) {
            const uint32_t key = span.key_begin + local;
            std::memcpy(k.data(), prob.raw_k.data() + size_t(key) * hd, size_t(hd) * sizeof(float));
            rope_apply_text(k.data(), prob.rope, span.storage_pos0 + int64_t(local));
            scores.push_back(scale * dot_f64(q.data(), k.data(), hd));
        }
    }
    return global_softmax_mix(scores, prob.values, prob.value_dim);
}

// Negative control: independent softmax per span, then average the per-span
// outputs weighted by span mass. This is the forbidden implementation from
// the planning guide §9.2 and must NOT match the global-softmax result.
static std::vector<float> per_span_softmax(const Problem & prob) {
    const uint32_t hd = prob.rope.head_dim;
    const float scale = prob.scale != 0.0f ? prob.scale : 1.0f / std::sqrt(float(hd));
    std::vector<float> q(prob.raw_q);
    rope_apply_text(q.data(), prob.rope, prob.query_virtual);
    // First pass: per-span softmax outputs.
    std::vector<std::vector<float>> span_outs;
    std::vector<double> span_mass;
    std::vector<float> k(hd);
    for (const auto & span : prob.spans) {
        std::vector<float> scores;
        scores.reserve(span.key_count);
        for (uint32_t local = 0; local < span.key_count; ++local) {
            const uint32_t key = span.key_begin + local;
            std::memcpy(k.data(), prob.raw_k.data() + size_t(key) * hd, size_t(hd) * sizeof(float));
            rope_apply_text(k.data(), prob.rope, span.virtual_pos0 + int64_t(local));
            scores.push_back(scale * dot_f64(q.data(), k.data(), hd));
        }
        const float m = *std::max_element(scores.begin(), scores.end());
        double denom = 0.0;
        for (float s : scores) {
            denom += std::exp(double(s - m));
        }
        span_mass.push_back(denom);
        std::vector<float> out(prob.value_dim, 0.0f);
        for (uint32_t local = 0; local < span.key_count; ++local) {
            const uint32_t key = span.key_begin + local;
            const double w = std::exp(double(scores[local] - m)) / denom;
            for (uint32_t d = 0; d < prob.value_dim; ++d) {
                out[d] += float(w * double(prob.values[size_t(key) * prob.value_dim + d]));
            }
        }
        span_outs.push_back(std::move(out));
    }
    double total = 0.0;
    for (double m : span_mass) {
        total += m;
    }
    std::vector<float> out(prob.value_dim, 0.0f);
    for (size_t s = 0; s < span_outs.size(); ++s) {
        const double w = span_mass[s] / total;
        for (uint32_t d = 0; d < prob.value_dim; ++d) {
            out[d] += float(w * span_outs[s][d]);
        }
    }
    return out;
}

// GQA reference: n_head_q query heads share n_head_kv KV heads.
// raw_q: [hq * hd], raw_k: [hkv * n_keys * hd], values: [hkv * n_keys * vd].
static std::vector<float> gqa_materialized(const std::vector<float> & raw_q,
                                           const std::vector<float> & raw_k,
                                           const std::vector<float> & values,
                                           uint32_t n_keys,
                                           uint32_t value_dim,
                                           uint32_t n_head_q,
                                           uint32_t n_head_kv,
                                           int64_t query_virtual,
                                           const std::vector<llama_rerot_ddvr_span> & spans,
                                           const RopeConfig & rope,
                                           float scale,
                                           bool use_qside) {
    const uint32_t hd = rope.head_dim;
    const float sc = scale != 0.0f ? scale : 1.0f / std::sqrt(float(hd));
    std::vector<float> out(size_t(n_head_q) * value_dim, 0.0f);
    std::vector<float> q(hd);
    std::vector<float> k(hd);
    for (uint32_t hq = 0; hq < n_head_q; ++hq) {
        const uint32_t hkv = hq / (n_head_q / n_head_kv);
        std::vector<float> scores;
        scores.reserve(n_keys);
        std::vector<uint32_t> order;
        order.reserve(n_keys);
        if (!use_qside) {
            std::memcpy(q.data(), raw_q.data() + size_t(hq) * hd, size_t(hd) * sizeof(float));
            rope_apply_text(q.data(), rope, query_virtual);
            for (const auto & span : spans) {
                for (uint32_t local = 0; local < span.key_count; ++local) {
                    const uint32_t key = span.key_begin + local;
                    const float * ksrc = raw_k.data() + (size_t(hkv) * n_keys + key) * hd;
                    std::memcpy(k.data(), ksrc, size_t(hd) * sizeof(float));
                    rope_apply_text(k.data(), rope, span.virtual_pos0 + int64_t(local));
                    scores.push_back(sc * dot_f64(q.data(), k.data(), hd));
                    order.push_back(key);
                }
            }
        } else {
            for (const auto & span : spans) {
                std::memcpy(q.data(), raw_q.data() + size_t(hq) * hd, size_t(hd) * sizeof(float));
                rope_apply_text(q.data(), rope, query_virtual + span.storage_pos0 - span.virtual_pos0);
                for (uint32_t local = 0; local < span.key_count; ++local) {
                    const uint32_t key = span.key_begin + local;
                    const float * ksrc = raw_k.data() + (size_t(hkv) * n_keys + key) * hd;
                    std::memcpy(k.data(), ksrc, size_t(hd) * sizeof(float));
                    rope_apply_text(k.data(), rope, span.storage_pos0 + int64_t(local));
                    scores.push_back(sc * dot_f64(q.data(), k.data(), hd));
                    order.push_back(key);
                }
            }
        }
        const float m = *std::max_element(scores.begin(), scores.end());
        std::vector<double> w(scores.size());
        double denom = 0.0;
        for (size_t i = 0; i < scores.size(); ++i) {
            w[i] = std::exp(double(scores[i] - m));
            denom += w[i];
        }
        for (size_t i = 0; i < scores.size(); ++i) {
            const double weight = w[i] / denom;
            const uint32_t key = order[i];
            const float * vsrc = values.data() + (size_t(hkv) * n_keys + key) * value_dim;
            for (uint32_t d = 0; d < value_dim; ++d) {
                out[size_t(hq) * value_dim + d] += float(weight * double(vsrc[d]));
            }
        }
    }
    return out;
}

} // namespace ddvr_ref

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

    // Cross-check both core paths against the independent test-local
    // reference: catches shared-assumption bugs in either implementation.
    ddvr_ref::Problem prob;
    prob.raw_q = query;
    prob.raw_k = keys;
    prob.values = values;
    prob.value_dim = value_dim;
    prob.query_virtual = query_position;
    prob.spans = spans;
    prob.rope = ddvr_ref::from_core(config);
    const auto ref_mat = ddvr_ref::materialized(prob);
    const auto ref_q = ddvr_ref::qside(prob);
    const float err_mat = max_abs_diff(materialized, ref_mat);
    const float err_q = max_abs_diff(qside, ref_q);
    if (err_mat >= 2.5e-5f || err_q >= 2.5e-5f) {
        std::fprintf(stderr, "core-vs-ref errors: mat=%.9g qside=%.9g\n", err_mat, err_q);
    }
    CHECK(err_mat < 2.5e-5f);
    CHECK(err_q < 2.5e-5f);
    // Independent reference must itself satisfy the identity.
    CHECK(max_abs_diff(ref_mat, ref_q) < 2.5e-5f);
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

static void test_partial_rotary_dim() {
    // Partial rotary dim: only the first rotary_dim entries rotate; the tail
    // must be preserved verbatim by both paths.
    llama_rerot_rope_config config;
    config.head_dim = 32;
    config.rotary_dim = 16;
    config.theta = 10000.0;
    config.layout = llama_rerot_rope_layout::half;

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 2, 4, 0 },
        { 2, 3, 0, 2 },
        { 5, 2, 44, 5 },
    };
    run_equivalence_case(config, 7, 9, spans, 7, 0x9aau);
}

static void test_partial_rotary_dim_interleaved() {
    llama_rerot_rope_config config;
    config.head_dim = 24;
    config.rotary_dim = 8;
    config.theta = 5000.0;
    config.layout = llama_rerot_rope_layout::interleaved;

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 3, 9, 0 },
        { 3, 3, 1, 3 },
    };
    run_equivalence_case(config, 6, 5, spans, 6, 0x5432u);
}

static void test_imrope_fourth_axis_untouched() {
    // Qwen3.5 text IMRoPE: positions are (p, p, p, 0). The axis_pair_count
    // assigns zero pairs to axis 3, so shifting the fourth coordinate must
    // not change the result, while shifting any of the first three must.
    llama_rerot_rope_config config;
    config.head_dim = 64;
    config.rotary_dim = 32;
    config.theta = 10000000.0;
    config.layout = llama_rerot_rope_layout::interleaved;
    config.axis_pair_count = { 6, 5, 5, 0 };

    std::mt19937 rng(0x40a5u);
    const auto query = random_vector(rng, config.head_dim);

    std::vector<float> a = query;
    std::vector<float> b = query;
    std::string error;
    const llama_rerot_rope_pos pos_zero = {7, 7, 7, 0};
    const llama_rerot_rope_pos pos_fourth_shifted = {7, 7, 7, 7};
    CHECK(llama_rerot_rope_apply(a.data(), a.size(), pos_zero, config, &error));
    CHECK(llama_rerot_rope_apply(b.data(), b.size(), pos_fourth_shifted, config, &error));
    CHECK(max_abs_diff(a, b) == 0.0f);

    // The independent reference agrees: fourth-axis shift is a no-op.
    ddvr_ref::RopeConfig ref = ddvr_ref::from_core(config);
    std::vector<float> c = query;
    std::vector<float> d = query;
    const int64_t pz[4] = {7, 7, 7, 0};
    const int64_t ps[4] = {7, 7, 7, 7};
    ddvr_ref::rope_apply(c.data(), ref, pz);
    ddvr_ref::rope_apply(d.data(), ref, ps);
    CHECK(max_abs_diff(c, d) == 0.0f);

    // Full attention equivalence still holds with the text mapping.
    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 3, 2, 0 },
        { 3, 3, 25, 3 },
    };
    run_equivalence_case(config, 6, 8, spans, 6, 0x60a7u);
}

static void test_sparse_physical_indices() {
    // Sparse physical storage: large gaps between writer-local positions
    // (holes left by private runs / reclaimed cells). Virtual positions stay
    // dense; the Q-side delta absorbs the sparsity exactly.
    llama_rerot_rope_config config;
    config.head_dim = 16;
    config.rotary_dim = 16;
    config.theta = 10000.0;
    config.layout = llama_rerot_rope_layout::half;

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 2, 0, 0 },       // storage 0,1 -> virtual 0,1
        { 2, 3, 50, 2 },      // storage 50,51,52 -> virtual 2,3,4 (gap of 48)
        { 5, 2, 200, 5 },     // storage 200,201 -> virtual 5,6 (gap of 148)
    };
    run_equivalence_case(config, 7, 6, spans, 7, 0x5a5au);
}

static void test_private_gaps() {
    // A private run occupies storage [4,8) but is invisible to this reader:
    // the visible spans skip it (storage gap), virtual stays dense.
    llama_rerot_rope_config config;
    config.head_dim = 24;
    config.rotary_dim = 16;
    config.theta = 10000.0;
    config.layout = llama_rerot_rope_layout::interleaved;

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 4, 0, 0 },   // public run A: storage 0..3 -> virtual 0..3
        // storage 4..7 is a private gap (no keys here)
        { 4, 3, 8, 4 },   // public run B: storage 8..10 -> virtual 4..6
    };
    run_equivalence_case(config, 7, 11, spans, 7, 0x9aacu);
}

static void test_gqa() {
    // GQA: hq query heads share hkv KV heads. Each head pair must satisfy
    // materialized == q-side under one global softmax per head.
    llama_rerot_rope_config config;
    config.head_dim = 16;
    config.rotary_dim = 16;
    config.theta = 10000.0;
    config.layout = llama_rerot_rope_layout::half;

    constexpr uint32_t n_keys = 8;
    constexpr uint32_t value_dim = 10;
    constexpr uint32_t hq = 4;
    constexpr uint32_t hkv = 2;
    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 3, 5, 0 },
        { 3, 2, 0, 3 },   // overlapping storage with first span
        { 5, 3, 40, 5 },  // sparse gap
    };
    constexpr int64_t query_virtual = 8;

    std::mt19937 rng(0x616au);
    const auto raw_q = random_vector(rng, size_t(hq) * config.head_dim);
    const auto raw_k = random_vector(rng, size_t(hkv) * n_keys * config.head_dim);
    const auto values = random_vector(rng, size_t(hkv) * n_keys * value_dim);

    ddvr_ref::RopeConfig ref = ddvr_ref::from_core(config);
    const auto ref_mat = ddvr_ref::gqa_materialized(
        raw_q, raw_k, values, n_keys, value_dim, hq, hkv, query_virtual, spans, ref, 0.0f, false);
    const auto ref_q = ddvr_ref::gqa_materialized(
        raw_q, raw_k, values, n_keys, value_dim, hq, hkv, query_virtual, spans, ref, 0.0f, true);
    const float err = max_abs_diff(ref_mat, ref_q);
    if (err >= 2.5e-5f) {
        std::fprintf(stderr, "GQA equivalence error = %.9g\n", err);
    }
    CHECK(err < 2.5e-5f);

    // Each head must also agree with the single-head core helpers when the
    // per-head slices are fed through them.
    for (uint32_t h = 0; h < hq; ++h) {
        const uint32_t kv = h / (hq / hkv);
        std::vector<float> q_slice(raw_q.begin() + size_t(h) * config.head_dim,
                                   raw_q.begin() + size_t(h + 1) * config.head_dim);
        std::vector<float> k_slice(n_keys * config.head_dim);
        std::vector<float> v_slice(n_keys * value_dim);
        for (uint32_t k = 0; k < n_keys; ++k) {
            std::memcpy(k_slice.data() + size_t(k) * config.head_dim,
                        raw_k.data() + (size_t(kv) * n_keys + k) * config.head_dim,
                        size_t(config.head_dim) * sizeof(float));
            std::memcpy(v_slice.data() + size_t(k) * value_dim,
                        values.data() + (size_t(kv) * n_keys + k) * value_dim,
                        size_t(value_dim) * sizeof(float));
        }
        const auto core_mat = llama_rerot_ddvr_attention_materialized(
            q_slice, k_slice, v_slice, value_dim, query_virtual, spans, config);
        const auto core_q = llama_rerot_ddvr_attention_qside(
            q_slice, k_slice, v_slice, value_dim, query_virtual, spans, config);
        CHECK(max_abs_diff(core_mat, core_q) < 2.5e-5f);
        std::vector<float> ref_head(ref_mat.begin() + size_t(h) * value_dim,
                                    ref_mat.begin() + size_t(h + 1) * value_dim);
        CHECK(max_abs_diff(core_mat, ref_head) < 2.5e-5f);
    }
}

static void test_gqa_imrope() {
    // GQA combined with Qwen3.5 text IMRoPE.
    llama_rerot_rope_config config;
    config.head_dim = 32;
    config.rotary_dim = 16;
    config.theta = 10000000.0;
    config.layout = llama_rerot_rope_layout::interleaved;
    config.axis_pair_count = { 3, 3, 2, 0 };

    constexpr uint32_t n_keys = 6;
    constexpr uint32_t value_dim = 8;
    constexpr uint32_t hq = 4;
    constexpr uint32_t hkv = 1;
    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 2, 0, 0 },
        { 2, 4, 12, 2 },
    };
    std::mt19937 rng(0x3555u);
    const auto raw_q = random_vector(rng, size_t(hq) * config.head_dim);
    const auto raw_k = random_vector(rng, size_t(hkv) * n_keys * config.head_dim);
    const auto values = random_vector(rng, size_t(hkv) * n_keys * value_dim);
    ddvr_ref::RopeConfig ref = ddvr_ref::from_core(config);
    const auto ref_mat = ddvr_ref::gqa_materialized(
        raw_q, raw_k, values, n_keys, value_dim, hq, hkv, 6, spans, ref, 0.0f, false);
    const auto ref_q = ddvr_ref::gqa_materialized(
        raw_q, raw_k, values, n_keys, value_dim, hq, hkv, 6, spans, ref, 0.0f, true);
    CHECK(max_abs_diff(ref_mat, ref_q) < 2.5e-5f);
}

static void test_global_softmax_only() {
    // Proves ONE global softmax: the forbidden per-span-softmax baseline
    // must differ observably, while materialized == q-side tightly.
    llama_rerot_rope_config config;
    config.head_dim = 16;
    config.rotary_dim = 16;
    config.theta = 10000.0;
    config.layout = llama_rerot_rope_layout::half;

    const std::vector<llama_rerot_ddvr_span> spans = {
        { 0, 2, 0, 0 },
        { 2, 2, 30, 2 },
        { 4, 2, 7, 4 },
    };
    std::mt19937 rng(0x600bau);
    const auto query = random_vector(rng, config.head_dim);
    const auto keys = random_vector(rng, size_t(6) * config.head_dim);
    const auto values = random_vector(rng, size_t(6) * 8);

    const auto materialized = llama_rerot_ddvr_attention_materialized(
        query, keys, values, 8, 6, spans, config);
    const auto qside = llama_rerot_ddvr_attention_qside(
        query, keys, values, 8, 6, spans, config);
    CHECK(max_abs_diff(materialized, qside) < 2.5e-5f);

    ddvr_ref::Problem prob;
    prob.raw_q = query;
    prob.raw_k = keys;
    prob.values = values;
    prob.value_dim = 8;
    prob.query_virtual = 6;
    prob.spans = spans;
    prob.rope = ddvr_ref::from_core(config);
    const auto per_span = ddvr_ref::per_span_softmax(prob);
    const float gap = max_abs_diff(materialized, per_span);
    std::fprintf(stderr, "global-vs-perspan gap = %.9g (must be large)\n", gap);
    CHECK(gap > 1e-4f);
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
    test_partial_rotary_dim();
    test_partial_rotary_dim_interleaved();
    test_imrope_fourth_axis_untouched();
    test_sparse_physical_indices();
    test_private_gaps();
    test_gqa();
    test_gqa_imrope();
    test_global_softmax_only();
    test_validation();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
