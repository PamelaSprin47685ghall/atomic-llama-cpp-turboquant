#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Declarations for the correctness-only CPU segmented DDVR helpers in
// ggml/src/ggml-cpu/ggml-cpu-rerot.cpp (no GGML_OP, ordinary path untouched).
// Struct layouts must match that file exactly.
namespace ggml_cpu_rerot {
enum class RopeLayout : uint8_t { Half = 0, Interleaved = 1 };
enum class FrontierMode : uint8_t { Strong = 0, Lag1 = 1 };
struct RopeConfig {
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    double theta = 10000.0;
    double freq_scale = 1.0;
    RopeLayout layout = RopeLayout::Half;
    uint32_t axis_pair_count[4] = {0, 0, 0, 0};
};
struct Span {
    uint32_t key_begin = 0;
    uint32_t key_count = 0;
    int64_t storage_pos0 = 0;
    int64_t virtual_pos0 = 0;
};
bool RopeApply(float * vec, const RopeConfig & cfg, const int64_t pos[4]);
bool RopeApplyText(float * vec, const RopeConfig & cfg, int64_t pos);
bool DdvrMaterializedGqa(const float * raw_q, const float * raw_k, const float * values,
                         uint32_t n_keys, uint32_t value_dim, uint32_t n_head_q, uint32_t n_head_kv,
                         int64_t query_virtual_pos, const Span * spans, size_t n_spans,
                         const RopeConfig & cfg, float scale, const uint64_t * key_frontiers,
                         uint64_t query_frontier, FrontierMode mode, const uint8_t * extra_mask,
                         float * out);
bool DdvrQsideGqa(const float * raw_q, const float * raw_k, const float * values,
                  uint32_t n_keys, uint32_t value_dim, uint32_t n_head_q, uint32_t n_head_kv,
                  int64_t query_virtual_pos, const Span * spans, size_t n_spans,
                  const RopeConfig & cfg, float scale, const uint64_t * key_frontiers,
                  uint64_t query_frontier, FrontierMode mode, const uint8_t * extra_mask,
                  float * out);
} // namespace ggml_cpu_rerot

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static float value_q(int g, int h, int d) {
    return 0.03f * float(1 + d) + 0.11f * float(g) - 0.04f * float(h);
}

static float value_k(int key, int h, int d) {
    return -0.02f * float(1 + d) + 0.07f * float(key) + 0.05f * float(h);
}

static float value_v(int key, int h, int d) {
    return 0.13f * float(1 + d) - 0.03f * float(key) + 0.09f * float(h);
}

static std::vector<float> reference(
        int d,
        int dv,
        int hq,
        int hkv,
        int nq,
        const std::vector<int32_t> & entries,
        const std::vector<int32_t> & offsets,
        float scale,
        float softcap,
        const std::vector<float> & sinks) {
    std::vector<float> out(size_t(dv) * hq * nq, 0.0f);
    for (int iq = 0; iq < nq; ++iq) {
        for (int ih = 0; ih < hq; ++ih) {
            const int ihkv = ih / (hq / hkv);
            float max_score = -INFINITY;
            std::vector<float> scores;
            scores.reserve(offsets[iq + 1] - offsets[iq]);
            for (int ie = offsets[iq]; ie < offsets[iq + 1]; ++ie) {
                const int key = entries[2 * ie + 0];
                const int group = entries[2 * ie + 1];
                float dot = 0.0f;
                for (int id = 0; id < d; ++id) {
                    dot += value_q(group, ih, id) * value_k(key, ihkv, id);
                }
                float score = dot * scale;
                if (softcap != 0.0f) {
                    score = softcap * std::tanh(score / softcap);
                }
                scores.push_back(score);
                max_score = std::max(max_score, score);
            }
            if (!sinks.empty()) {
                max_score = std::max(max_score, sinks[ih]);
            }

            float denom = sinks.empty() ? 0.0f : std::exp(sinks[ih] - max_score);
            for (float score : scores) {
                denom += std::exp(score - max_score);
            }
            for (int ie = offsets[iq]; ie < offsets[iq + 1]; ++ie) {
                const int local = ie - offsets[iq];
                const int key = entries[2 * ie + 0];
                const float weight = std::exp(scores[local] - max_score) / denom;
                for (int id = 0; id < dv; ++id) {
                    out[(size_t(iq) * hq + ih) * dv + id] +=
                        weight * value_v(key, ihkv, id);
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Test-local independent DDVR reference (no dependency on the helpers under
// test, no llama headers): raw Q/K + spans + IMRoPE text + GQA + frontier
// mask + ONE global softmax. Mirrors the planning guide 9.1 identity.
// ---------------------------------------------------------------------------
namespace attn_ref {

struct RopeConfig {
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    double theta = 10000.0;
    double freq_scale = 1.0;
    bool interleaved = false;
    uint32_t axis_pairs[4] = {0, 0, 0, 0};
};

static uint32_t pair_axis(const RopeConfig & cfg, uint32_t pair) {
    const uint32_t configured = cfg.axis_pairs[0] + cfg.axis_pairs[1] + cfg.axis_pairs[2] + cfg.axis_pairs[3];
    if (configured == 0) {
        return 0;
    }
    uint32_t offset = 0;
    for (uint32_t a = 0; a < 4; ++a) {
        const uint32_t next = offset + cfg.axis_pairs[a];
        if (pair < next) {
            return a;
        }
        offset = next;
    }
    return 0;
}

static void rope_text(float * vec, const RopeConfig & cfg, int64_t p) {
    const int64_t pos[4] = {p, p, p, 0};
    const uint32_t n_pairs = cfg.rotary_dim / 2;
    for (uint32_t pair = 0; pair < n_pairs; ++pair) {
        const uint32_t axis = pair_axis(cfg, pair);
        const double freq = std::pow(cfg.theta, -2.0 * double(pair) / double(cfg.rotary_dim)) * cfg.freq_scale;
        const double angle = double(pos[axis]) * freq;
        const float c = float(std::cos(angle));
        const float s = float(std::sin(angle));
        uint32_t f;
        uint32_t t;
        if (cfg.interleaved) {
            f = pair * 2;
            t = f + 1;
        } else {
            f = pair;
            t = pair + n_pairs;
        }
        const float x = vec[f];
        const float y = vec[t];
        vec[f] = x * c - y * s;
        vec[t] = x * s + y * c;
    }
}

static float dot_f64(const float * a, const float * b, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += double(a[i]) * double(b[i]);
    }
    return float(sum);
}

struct Span {
    uint32_t key_begin = 0;
    uint32_t key_count = 0;
    int64_t storage_pos0 = 0;
    int64_t virtual_pos0 = 0;
};

// visible[key] precomputed by caller (frontier strong/lag1 + private mask).
// raw_q [hq*hd], raw_k [hkv*n_keys*hd], vals [hkv*n_keys*vd]. One global
// softmax per query head over visible keys.
static std::vector<float> materialized_gqa(const std::vector<float> & raw_q,
                                           const std::vector<float> & raw_k,
                                           const std::vector<float> & vals,
                                           uint32_t n_keys, uint32_t vd,
                                           uint32_t hq, uint32_t hkv,
                                           int64_t q_virtual,
                                           const std::vector<Span> & spans,
                                           const RopeConfig & rope,
                                           const std::vector<uint8_t> & visible) {
    const uint32_t hd = rope.head_dim;
    const float scale = 1.0f / std::sqrt(float(hd));
    std::vector<float> out(size_t(hq) * vd, 0.0f);
    std::vector<float> q(hd);
    std::vector<float> k(hd);
    for (uint32_t h = 0; h < hq; ++h) {
        const uint32_t kv = h / (hq / hkv);
        std::memcpy(q.data(), raw_q.data() + size_t(h) * hd, size_t(hd) * sizeof(float));
        rope_text(q.data(), rope, q_virtual);
        std::vector<float> scores;
        std::vector<uint32_t> order;
        for (const auto & sp : spans) {
            for (uint32_t l = 0; l < sp.key_count; ++l) {
                const uint32_t key = sp.key_begin + l;
                if (!visible[key]) {
                    continue;
                }
                std::memcpy(k.data(), raw_k.data() + (size_t(kv) * n_keys + key) * hd,
                            size_t(hd) * sizeof(float));
                rope_text(k.data(), rope, sp.virtual_pos0 + int64_t(l));
                scores.push_back(scale * dot_f64(q.data(), k.data(), hd));
                order.push_back(key);
            }
        }
        if (scores.empty()) {
            continue;
        }
        const float m = *std::max_element(scores.begin(), scores.end());
        std::vector<double> w(scores.size());
        double denom = 0.0;
        for (size_t i = 0; i < scores.size(); ++i) {
            w[i] = std::exp(double(scores[i] - m));
            denom += w[i];
        }
        for (size_t i = 0; i < scores.size(); ++i) {
            const double wt = w[i] / denom;
            const float * vsrc = vals.data() + (size_t(kv) * n_keys + order[i]) * vd;
            for (uint32_t d = 0; d < vd; ++d) {
                out[size_t(h) * vd + d] += float(wt * double(vsrc[d]));
            }
        }
    }
    return out;
}

} // namespace attn_ref

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    // Missing/nonfinite output is a failed numerical gate, not zero error.
    // std::max(m, NaN) keeps m and used to silently accept broken kernels.
    if (a.empty() || a.size() != b.size()) return INFINITY;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

static void test_error_metric_rejects_invalid_outputs() {
    CHECK(max_abs_diff({1.0f, -2.0f}, {1.0f, -2.0f}) == 0.0f);
    CHECK(max_abs_diff({1.0f, -2.0f}, {1.25f, -2.0f}) == 0.25f);
    CHECK(std::isinf(max_abs_diff({}, {})));
    CHECK(std::isinf(max_abs_diff({1.0f}, {})));
    CHECK(std::isinf(max_abs_diff({1.0f}, {1.0f, 2.0f})));
    for (const float bad : {NAN, INFINITY, -INFINITY}) {
        CHECK(std::isinf(max_abs_diff({bad}, {0.0f})));
        CHECK(std::isinf(max_abs_diff({0.0f}, {bad})));
        CHECK(std::isinf(max_abs_diff({bad}, {bad})));
    }
}

// Runs the indexed RERoT op (GGML_OP_FLASH_ATTN_EXT_REROT, ordinary path
// untouched) for one zero-sink configuration and returns the output.
static std::vector<float> run_indexed_op(int d, int dv, int ng, int nkv, int hq, int hkv, int nq,
                                         const std::vector<float> & q_data,
                                         const std::vector<float> & k_data,
                                         const std::vector<float> & v_data,
                                         const std::vector<int32_t> & entries,
                                         const std::vector<int32_t> & offsets,
                                         float scale,
                                         ggml_backend_t backend_override = nullptr,
                                         ggml_type k_type = GGML_TYPE_F32,
                                         ggml_type v_type = GGML_TYPE_F32,
                                         const std::vector<float> * sink_data = nullptr) {
    CHECK(nq == int(offsets.size()) - 1);
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        ++failures;
        return {};
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, ng, hq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), k_type, d, nkv, hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), v_type, dv, nkv, hkv, 1);
    ggml_tensor * e = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, entries.size() / 2);
    ggml_tensor * o = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, offsets.size());
    ggml_tensor * s = sink_data && !sink_data->empty()
        ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, sink_data->size())
        : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext_rerot(ctx.get(), q, k, v, e, o, s, scale, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_backend_t backend = backend_override;
    const bool owns_backend = backend == nullptr;
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        std::fprintf(stderr, "backend init failed\n");
        ++failures;
        return {};
    }
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "backend does not support rerot op\n");
        ++failures;
        if (owns_backend) {
            ggml_backend_free(backend);
        }
        return {};
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    std::vector<uint8_t> k_quant;
    std::vector<uint8_t> v_quant;
    if (k_type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    } else {
        k_quant.resize(ggml_nbytes(k));
        const size_t written = ggml_quantize_chunk(
            k_type, k_data.data(), k_quant.data(), 0, int64_t(nkv) * hkv, d, nullptr);
        CHECK(written == k_quant.size());
        ggml_backend_tensor_set(k, k_quant.data(), 0, k_quant.size());
    }
    if (v_type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    } else {
        v_quant.resize(ggml_nbytes(v));
        const size_t written = ggml_quantize_chunk(
            v_type, v_data.data(), v_quant.data(), 0, int64_t(nkv) * hkv, dv, nullptr);
        CHECK(written == v_quant.size());
        ggml_backend_tensor_set(v, v_quant.data(), 0, v_quant.size());
    }
    ggml_backend_tensor_set(e, entries.data(), 0, entries.size() * sizeof(int32_t));
    ggml_backend_tensor_set(o, offsets.data(), 0, offsets.size() * sizeof(int32_t));
    if (s) {
        ggml_backend_tensor_set(s, sink_data->data(), 0, sink_data->size() * sizeof(float));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        ++failures;
        buffer.reset();
        if (owns_backend) {
            ggml_backend_free(backend);
        }
        return {};
    }
    ggml_backend_synchronize(backend);

    std::vector<float> actual(ggml_nelements(out));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    buffer.reset();
    if (owns_backend) {
        ggml_backend_free(backend);
    }
    return actual;
}

static void test_indexed_basic() {
    std::puts("--- indexed basic (GQA + sinks + global softmax) ---");

    constexpr int d = 8;
    constexpr int dv = 6;
    constexpr int ng = 4;
    constexpr int nkv = 5;
    constexpr int hq = 4;
    constexpr int hkv = 2;
    constexpr int nq = 2;

    const std::vector<int32_t> entries = {
        0, 0,
        4, 0,
        2, 1,
        1, 2,
        4, 2,
        3, 3,
    };
    const std::vector<int32_t> offsets = { 0, 3, 6 };
    const std::vector<float> sinks = { -0.4f, -0.1f, 0.2f, 0.5f };
    constexpr float scale = 0.37f;
    constexpr float softcap = 1.7f;

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    CHECK(bool(ctx));
    if (!ctx) {
        return;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, ng, hq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, nkv, hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dv, nkv, hkv, 1);
    ggml_tensor * e = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, entries.size() / 2);
    ggml_tensor * o = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, offsets.size());
    ggml_tensor * s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, sinks.size());
    ggml_tensor * out = ggml_flash_attn_ext_rerot(ctx.get(), q, k, v, e, o, s, scale, softcap);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    CHECK(ggml_backend_supports_op(backend, out));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    CHECK(bool(buffer));

    std::vector<float> q_data(size_t(d) * ng * hq);
    std::vector<float> k_data(size_t(d) * nkv * hkv);
    std::vector<float> v_data(size_t(dv) * nkv * hkv);
    for (int h = 0; h < hq; ++h) {
        for (int g = 0; g < ng; ++g) {
            for (int id = 0; id < d; ++id) {
                q_data[(size_t(h) * ng + g) * d + id] = value_q(g, h, id);
            }
        }
    }
    for (int h = 0; h < hkv; ++h) {
        for (int key = 0; key < nkv; ++key) {
            for (int id = 0; id < d; ++id) {
                k_data[(size_t(h) * nkv + key) * d + id] = value_k(key, h, id);
            }
            for (int id = 0; id < dv; ++id) {
                v_data[(size_t(h) * nkv + key) * dv + id] = value_v(key, h, id);
            }
        }
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(e, entries.data(), 0, entries.size() * sizeof(int32_t));
    ggml_backend_tensor_set(o, offsets.data(), 0, offsets.size() * sizeof(int32_t));
    ggml_backend_tensor_set(s, sinks.data(), 0, sinks.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(ggml_nelements(out));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    const auto expected = reference(d, dv, hq, hkv, nq, entries, offsets, scale, softcap, sinks);
    CHECK(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 2e-5f) {
            std::fprintf(stderr, "mismatch[%zu]: actual=%g expected=%g\n", i, actual[i], expected[i]);
            ++failures;
        }
    }

    ggml_backend_free(backend);
}

// Builds Q groups for one query: q_group[span] = R(q_virtual + storage0 -
// virtual0) * raw_q, K rows at storage phase. Returns q_data in the indexed
// op layout [(h * ng + g) * d + id].
static std::vector<float> build_q_groups(const std::vector<float> & raw_q,
                                         int d, int hq,
                                         int64_t q_virtual,
                                         const std::vector<attn_ref::Span> & spans,
                                         const attn_ref::RopeConfig & rope) {
    const int ng = int(spans.size());
    std::vector<float> q_data(size_t(d) * ng * hq);
    std::vector<float> tmp(d);
    for (int h = 0; h < hq; ++h) {
        for (int g = 0; g < ng; ++g) {
            std::memcpy(tmp.data(), raw_q.data() + size_t(h) * d, size_t(d) * sizeof(float));
            attn_ref::rope_text(tmp.data(), rope, q_virtual + spans[size_t(g)].storage_pos0 - spans[size_t(g)].virtual_pos0);
            std::memcpy(q_data.data() + (size_t(h) * ng + g) * d, tmp.data(), size_t(d) * sizeof(float));
        }
    }
    return q_data;
}

static void test_ddvr_imrope_via_indexed_op() {
    std::puts("--- DDVR IMRoPE text via indexed op (GQA + sparse storage) ---");

    constexpr int d = 32;
    constexpr int dv = 8;
    constexpr int hq = 4;
    constexpr int hkv = 2;
    constexpr int nkv = 7;
    constexpr int64_t q_virtual = 7;

    attn_ref::RopeConfig rope;
    rope.head_dim = d;
    rope.rotary_dim = 16;
    rope.theta = 10000000.0;
    rope.interleaved = true;
    rope.axis_pairs[0] = 3;
    rope.axis_pairs[1] = 3;
    rope.axis_pairs[2] = 2;
    rope.axis_pairs[3] = 0;

    // Multi-span with overlapping storage + sparse gap (private run hole).
    const std::vector<attn_ref::Span> spans = {
        { 0, 2, 0, 0 },    // storage 0,1 -> virtual 0,1
        { 2, 3, 0, 2 },    // storage 0,1,2 -> virtual 2,3,4 (overlap)
        { 5, 2, 40, 5 },   // storage 40,41 -> virtual 5,6 (sparse gap)
    };
    const int ng = int(spans.size());

    std::mt19937 rng(0xdd01u);
    std::normal_distribution<float> dist(0.0f, 0.35f);
    std::vector<float> raw_q(size_t(hq) * d);
    std::vector<float> raw_k(size_t(hkv) * nkv * d);
    std::vector<float> raw_v(size_t(hkv) * nkv * dv);
    for (float & x : raw_q) {
        x = dist(rng);
    }
    for (float & x : raw_k) {
        x = dist(rng);
    }
    for (float & x : raw_v) {
        x = dist(rng);
    }

    // K rows carry storage-phase RoPE; V stays raw.
    std::vector<float> k_data(size_t(hkv) * nkv * d);
    std::vector<float> tmp(d);
    for (int h = 0; h < hkv; ++h) {
        for (int key = 0; key < nkv; ++key) {
            // Find the span owning this key for its storage position.
            int64_t spos = 0;
            for (const auto & sp : spans) {
                if (key >= int(sp.key_begin) && key < int(sp.key_begin + sp.key_count)) {
                    spos = sp.storage_pos0 + (key - int(sp.key_begin));
                }
            }
            std::memcpy(tmp.data(), raw_k.data() + (size_t(h) * nkv + key) * d, size_t(d) * sizeof(float));
            attn_ref::rope_text(tmp.data(), rope, spos);
            std::memcpy(k_data.data() + (size_t(h) * nkv + key) * d, tmp.data(), size_t(d) * sizeof(float));
        }
    }

    const std::vector<float> q_data = build_q_groups(raw_q, d, hq, q_virtual, spans, rope);

    // Entries in permuted (sparse) order: reversed within the single query
    // range. Global softmax must be order-invariant.
    std::vector<int32_t> entries;
    for (int g = ng - 1; g >= 0; --g) {
        for (int l = int(spans[size_t(g)].key_count) - 1; l >= 0; --l) {
            entries.push_back(int(spans[size_t(g)].key_begin) + l);
            entries.push_back(g);
        }
    }
    const std::vector<int32_t> offsets = { 0, int(entries.size() / 2) };
    const float scale = 1.0f / std::sqrt(float(d));

    const std::vector<float> actual = run_indexed_op(d, dv, ng, nkv, hq, hkv, 1, q_data, k_data, raw_v,
                                                     entries, offsets, scale);

    const std::vector<uint8_t> visible(nkv, 1);
    const std::vector<float> expected =
        attn_ref::materialized_gqa(raw_q, raw_k, raw_v, nkv, dv, hq, hkv, q_virtual, spans, rope, visible);
    CHECK(actual.size() == expected.size());
    const float err = max_abs_diff(actual, expected);
    if (err >= 1e-4f) {
        std::fprintf(stderr, "DDVR IMRoPE op-vs-ref error = %.9g\n", err);
    }
    CHECK(err < 1e-4f);
}

static void test_frontier_strong_vs_lag1() {
    std::puts("--- frontier strong vs lag1 mask ---");

    constexpr int d = 16;
    constexpr int dv = 6;
    constexpr int hq = 2;
    constexpr int hkv = 1;
    constexpr int nkv = 5;
    constexpr int64_t q_virtual = 5;

    attn_ref::RopeConfig rope;
    rope.head_dim = d;
    rope.rotary_dim = 16;
    rope.theta = 10000.0;
    rope.interleaved = false;

    const std::vector<attn_ref::Span> spans = {
        { 0, 3, 0, 0 },
        { 3, 2, 10, 3 },
    };
    const int ng = int(spans.size());

    // Staged visibility at query frontier 8:
    // - epoch 6 is visible to strong and lag1
    // - epoch 7 is visible to strong only
    // - epoch 8 is current write-stage peer data and visible to neither.
    const std::vector<uint64_t> frontiers = { 6, 6, 7, 8, 8 };
    constexpr uint64_t q_frontier = 8;

    std::mt19937 rng(0xf20au);
    std::normal_distribution<float> dist(0.0f, 0.35f);
    std::vector<float> raw_q(size_t(hq) * d);
    std::vector<float> raw_k(size_t(hkv) * nkv * d);
    std::vector<float> raw_v(size_t(hkv) * nkv * dv);
    for (float & x : raw_q) {
        x = dist(rng);
    }
    for (float & x : raw_k) {
        x = dist(rng);
    }
    for (float & x : raw_v) {
        x = dist(rng);
    }

    std::vector<float> k_data(size_t(hkv) * nkv * d);
    std::vector<float> tmp(d);
    for (int key = 0; key < nkv; ++key) {
        int64_t spos = key < 3 ? key : 10 + (key - 3);
        std::memcpy(tmp.data(), raw_k.data() + size_t(key) * d, size_t(d) * sizeof(float));
        attn_ref::rope_text(tmp.data(), rope, spos);
        std::memcpy(k_data.data() + size_t(key) * d, tmp.data(), size_t(d) * sizeof(float));
    }
    const std::vector<float> q_data = build_q_groups(raw_q, d, hq, q_virtual, spans, rope);
    const float scale = 1.0f / std::sqrt(float(d));

    auto run_with_mask = [&](const std::vector<uint8_t> & vis) {
        std::vector<int32_t> entries;
        for (int g = 0; g < ng; ++g) {
            for (uint32_t l = 0; l < spans[size_t(g)].key_count; ++l) {
                const int key = int(spans[size_t(g)].key_begin) + int(l);
                if (!vis[size_t(key)]) {
                    continue;
                }
                entries.push_back(key);
                entries.push_back(g);
            }
        }
        const std::vector<int32_t> offsets = { 0, int(entries.size() / 2) };
        return run_indexed_op(d, dv, ng, nkv, hq, hkv, 1, q_data, k_data, raw_v, entries, offsets, scale);
    };

    std::vector<uint8_t> vis_strong = { 1, 1, 1, 0, 0 };
    std::vector<uint8_t> vis_lag1   = { 1, 1, 0, 0, 0 };
    const std::vector<float> out_strong = run_with_mask(vis_strong);
    const std::vector<float> out_lag1 = run_with_mask(vis_lag1);

    const std::vector<float> exp_strong =
        attn_ref::materialized_gqa(raw_q, raw_k, raw_v, nkv, dv, hq, hkv, q_virtual, spans, rope, vis_strong);
    const std::vector<float> exp_lag1 =
        attn_ref::materialized_gqa(raw_q, raw_k, raw_v, nkv, dv, hq, hkv, q_virtual, spans, rope, vis_lag1);

    CHECK(out_strong.size() == exp_strong.size());
    CHECK(out_lag1.size() == exp_lag1.size());
    CHECK(max_abs_diff(out_strong, exp_strong) < 1e-4f);
    CHECK(max_abs_diff(out_lag1, exp_lag1) < 1e-4f);
    // The two frontier modes must observably differ on this problem.
    const float gap = max_abs_diff(out_strong, out_lag1);
    std::fprintf(stderr, "frontier strong-vs-lag1 gap = %.9g\n", gap);
    CHECK(gap > 1e-4f);

    // Cross-check the ggml-cpu reference helpers on the same problem.
    ggml_cpu_rerot::RopeConfig cpu_rope;
    cpu_rope.head_dim = d;
    cpu_rope.rotary_dim = 16;
    cpu_rope.theta = 10000.0;
    cpu_rope.freq_scale = 1.0;
    cpu_rope.layout = ggml_cpu_rerot::RopeLayout::Half;
    std::vector<ggml_cpu_rerot::Span> cpu_spans = { { 0, 3, 0, 0 }, { 3, 2, 10, 3 } };
    std::vector<float> cpu_out_strong(size_t(hq) * dv, 0.0f);
    std::vector<float> cpu_out_lag1(size_t(hq) * dv, 0.0f);
    std::vector<uint8_t> mask_all = vis_strong;
    std::vector<uint8_t> mask_lag = vis_lag1;
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(raw_q.data(), raw_k.data(), raw_v.data(), nkv, dv, hq, hkv,
                                       q_virtual, cpu_spans.data(), cpu_spans.size(), cpu_rope, 0.0f,
                                       nullptr, 0, ggml_cpu_rerot::FrontierMode::Strong, mask_all.data(),
                                       cpu_out_strong.data()));
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(raw_q.data(), raw_k.data(), raw_v.data(), nkv, dv, hq, hkv,
                                       q_virtual, cpu_spans.data(), cpu_spans.size(), cpu_rope, 0.0f,
                                       nullptr, 0, ggml_cpu_rerot::FrontierMode::Strong, mask_lag.data(),
                                       cpu_out_lag1.data()));
    CHECK(max_abs_diff(cpu_out_strong, exp_strong) < 1e-4f);
    CHECK(max_abs_diff(cpu_out_lag1, exp_lag1) < 1e-4f);

    // Frontier-epoch filtering inside the helper: same masks via epochs.
    std::vector<float> cpu_front_strong(size_t(hq) * dv, 0.0f);
    std::vector<float> cpu_front_lag1(size_t(hq) * dv, 0.0f);
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(raw_q.data(), raw_k.data(), raw_v.data(), nkv, dv, hq, hkv,
                                       q_virtual, cpu_spans.data(), cpu_spans.size(), cpu_rope, 0.0f,
                                       frontiers.data(), q_frontier, ggml_cpu_rerot::FrontierMode::Strong,
                                       nullptr, cpu_front_strong.data()));
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(raw_q.data(), raw_k.data(), raw_v.data(), nkv, dv, hq, hkv,
                                       q_virtual, cpu_spans.data(), cpu_spans.size(), cpu_rope, 0.0f,
                                       frontiers.data(), q_frontier, ggml_cpu_rerot::FrontierMode::Lag1,
                                       nullptr, cpu_front_lag1.data()));
    CHECK(max_abs_diff(cpu_front_strong, exp_strong) < 1e-4f);
    CHECK(max_abs_diff(cpu_front_lag1, exp_lag1) < 1e-4f);
}

static std::vector<float> run_standard_op(
        int d,
        int dv,
        int nkv,
        int hq,
        int hkv,
        const std::vector<float> & q_data,
        const std::vector<float> & k_data,
        const std::vector<float> & v_data,
        float scale,
        ggml_backend_t backend,
        ggml_type k_type,
        ggml_type v_type,
        const std::vector<float> & sink_data) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, 1, hq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), k_type, d, nkv, hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), v_type, dv, nkv, hkv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, nkv, 1, 1, 1);
    ggml_tensor * s = sink_data.empty()
        ? nullptr
        : ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, sink_data.size());
    ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, m, scale, 0.0f, 0.0f);
    if (s) {
        ggml_flash_attn_ext_add_sinks(out, s);
    }
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    CHECK(ggml_backend_supports_op(backend, out));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    std::vector<uint8_t> k_quant(ggml_nbytes(k));
    std::vector<uint8_t> v_quant(ggml_nbytes(v));
    ggml_quantize_chunk(k_type, k_data.data(), k_quant.data(), 0, int64_t(nkv) * hkv, d, nullptr);
    ggml_quantize_chunk(v_type, v_data.data(), v_quant.data(), 0, int64_t(nkv) * hkv, dv, nullptr);
    ggml_backend_tensor_set(k, k_quant.data(), 0, k_quant.size());
    ggml_backend_tensor_set(v, v_quant.data(), 0, v_quant.size());
    std::vector<ggml_fp16_t> mask(size_t(nkv), ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (s) {
        ggml_backend_tensor_set(s, sink_data.data(), 0, sink_data.size() * sizeof(float));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 24, false);
    ggml_build_forward_expand(graph, out);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> actual(ggml_nelements(out));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    return actual;
}

// Decode the same quantized bytes used by the GPU tests, but do not invoke
// either attention implementation when constructing the numerical oracle.
static std::vector<float> stored_values(const std::vector<float> & input, ggml_type type, int width) {
    if (type == GGML_TYPE_F32) return input;
    const size_t rows = input.size() / width;
    CHECK(rows * width == input.size());
    const size_t row_bytes = ggml_row_size(type, width);
    std::vector<uint8_t> encoded(rows * row_bytes);
    const size_t written = ggml_quantize_chunk(type, input.data(), encoded.data(), 0, rows, width, nullptr);
    const auto * traits = ggml_get_type_traits(type);
    CHECK(written == encoded.size() && traits && traits->to_float);
    if (written != encoded.size() || !traits || !traits->to_float) return {};
    std::vector<float> decoded(input.size());
    for (size_t row = 0; row < rows; ++row) {
        traits->to_float(encoded.data() + row * row_bytes, decoded.data() + row * width, width);
    }
    return decoded;
}

// Equal visible keys and equal Q must not acquire different rounding merely
// by selecting the indexed execution shape. In particular GGML_PREC_F32 must
// not round Q, softmax weights, or the PV accumulation to half precision.
static void test_attention_f32_precision(bool require_gpu = false) {
    ggml_backend_load_all();
    const auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) {
        std::puts("SKIP: no GPU backend for attention precision regression");
        CHECK(!require_gpu);
    }
    ggml_backend_t gpu = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    CHECK(!device || gpu != nullptr);
    constexpr int d = 256, hq = 16, hkv = 2;
    const float scale = 1.0f / std::sqrt(float(d));
    std::mt19937 rng(0x51a9u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (const int nkv : {33, 257}) {
        std::vector<float> q(d * hq), k(size_t(d) * nkv * hkv), v(k.size());
        for (auto & x : q) x = dist(rng) * 2.0f;
        for (auto & x : k) x = dist(rng);
        for (auto & x : v) x = dist(rng);
        std::vector<int32_t> entries;
        for (int i = 0; i < nkv; ++i) { entries.push_back(i); entries.push_back(0); }
        const std::vector<int32_t> offsets = {0, nkv};
        for (const bool turbo : {false, true}) {
            const auto kt = turbo ? GGML_TYPE_TURBO4_0 : GGML_TYPE_F16;
            const auto vt = turbo ? GGML_TYPE_TURBO2_0 : GGML_TYPE_F16;
            const auto stored_k = stored_values(k, kt, d);
            const auto stored_v = stored_values(v, vt, d);
            CHECK(stored_k.size() == k.size() && stored_v.size() == v.size());
            if (stored_k.size() != k.size() || stored_v.size() != v.size()) continue;
            // Both F16 and Turbo now use an independent f64 QK/softmax/PV
            // reference over their stored K/V, with the original F32 Q.
            std::vector<float> ref(q.size());
            for (int h = 0; h < hq; ++h) {
                const int kvh = h / (hq / hkv);
                std::vector<double> scores(nkv);
                for (int i = 0; i < nkv; ++i) {
                    double dot = 0.0;
                    for (int c = 0; c < d; ++c) dot += double(q[h*d+c]) *
                        stored_k[(size_t(kvh)*nkv+i)*d+c];
                    scores[i] = dot * scale;
                }
                const double maximum = *std::max_element(scores.begin(), scores.end());
                double sum = 0.0;
                for (auto & x : scores) { x = std::exp(x - maximum); sum += x; }
                for (int c = 0; c < d; ++c) {
                    double value = 0.0;
                    for (int i = 0; i < nkv; ++i) value += scores[i] / sum *
                        stored_v[(size_t(kvh)*nkv+i)*d+c];
                    ref[h*d+c] = float(value);
                }
            }
            const auto cpu = run_indexed_op(d, d, 1, nkv, hq, hkv, 1, q, k, v, entries, offsets, scale, nullptr, kt, vt);
            const float cpu_error = max_abs_diff(ref, cpu);
            std::printf("F32_CPU_PRECISION cache=%s keys=%d error=%g reference=f64\n",
                turbo ? "turbo" : "f16", nkv, cpu_error);
            CHECK(cpu_error < 2e-5f);
            // The CPU oracle gate also runs on hosts without a GPU.
            if (!gpu) continue;
            const auto ordinary = run_standard_op(d, d, nkv, hq, hkv, q, k, v, scale, gpu, kt, vt, {});
            const auto indexed = run_indexed_op(d, d, 1, nkv, hq, hkv, 1, q, k, v, entries, offsets, scale, gpu, kt, vt);
            CHECK(ref.size() == ordinary.size() && ref.size() == indexed.size());
            const float native_error = max_abs_diff(ref, ordinary);
            const float indexed_error = max_abs_diff(ref, indexed);
            const float shape_error = max_abs_diff(ordinary, indexed);
            std::printf("F32_PRECISION cache=%s keys=%d native=%g indexed=%g shape=%g cpu=%g reference=f64\n",
                turbo ? "turbo" : "f16", nkv, native_error, indexed_error, shape_error, cpu_error);
            CHECK(native_error < 2e-5f);
            CHECK(indexed_error < 2e-5f);
            CHECK(shape_error < 2e-5f);
        }
    }
    if (gpu) ggml_backend_free(gpu);
}

static void test_vulkan_indexed_parity() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) {
        std::puts("SKIP: no GPU backend for RERoT indexed parity");
        return;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend != nullptr);
    if (!backend) {
        return;
    }

    constexpr int d = 64;
    constexpr int dv = 64;
    constexpr int ng = 1;
    constexpr int nkv = 2;
    constexpr int hq = 1;
    constexpr int hkv = 1;
    const std::vector<int32_t> entries = { 0, 0, 1, 0 };
    const std::vector<int32_t> offsets = { 0, 2 };
    std::vector<float> q(size_t(d) * ng * hq, 0.0f);
    std::vector<float> k(size_t(d) * nkv * hkv, 0.0f);
    std::vector<float> v(size_t(dv) * nkv * hkv, 0.0f);
    q[0] = 1.0f;
    k[0] = 1.0f;
    v[0] = 1.0f;
    v[size_t(dv) + 1] = 1.0f;

    const auto cpu = run_indexed_op(
        d, dv, ng, nkv, hq, hkv, 1, q, k, v, entries, offsets, 1.0f);
    const auto gpu = run_indexed_op(
        d, dv, ng, nkv, hq, hkv, 1, q, k, v, entries, offsets, 1.0f, backend);
    CHECK(cpu.size() == gpu.size());
    if (cpu.size() == gpu.size()) {
        const float error = max_abs_diff(cpu, gpu);
        if (error >= 2.0e-4f) {
            std::fprintf(stderr, "GPU indexed RERoT parity error = %.9g\n", error);
        }
        CHECK(error < 2.0e-4f);
    }

    constexpr int td = 256;
    constexpr int tdv = 256;
    constexpr int tng = 1;
    constexpr int tnkv = 3;
    constexpr int thq = 2;
    constexpr int thkv = 1;
    const std::vector<int32_t> tentries = { 0, 0, 2, 0, 1, 0 };
    const std::vector<int32_t> toffsets = { 0, 3 };
    std::vector<float> tq(size_t(td) * tng * thq);
    std::vector<float> tk(size_t(td) * tnkv * thkv);
    std::vector<float> tv(size_t(tdv) * tnkv * thkv);
    for (size_t i = 0; i < tq.size(); ++i) tq[i] = std::sin(float(i + 1) * 0.013f);
    for (size_t i = 0; i < tk.size(); ++i) tk[i] = std::cos(float(i + 3) * 0.017f);
    for (size_t i = 0; i < tv.size(); ++i) tv[i] = std::sin(float(i + 5) * 0.019f);
    const auto kcpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        1.0f / std::sqrt(float(td)), nullptr, GGML_TYPE_TURBO4_0, GGML_TYPE_F32);
    const auto kgpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        1.0f / std::sqrt(float(td)), backend, GGML_TYPE_TURBO4_0, GGML_TYPE_F32);
    const float kerror = max_abs_diff(kcpu, kgpu);
    std::fprintf(stderr, "GPU Turbo4/F32 RERoT parity error = %.9g h0=[%.6g %.6g]/[%.6g %.6g] h1=[%.6g %.6g]/[%.6g %.6g]\n",
        kerror, kcpu[0], kcpu[1], kgpu[0], kgpu[1],
        kcpu[tdv], kcpu[tdv + 1], kgpu[tdv], kgpu[tdv + 1]);
    CHECK(kcpu.size() == kgpu.size());
    CHECK(kerror < 2.0e-4f);

    const auto fcpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        1.0f / std::sqrt(float(td)));
    const auto fgpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        1.0f / std::sqrt(float(td)), backend);
    const float ferror = max_abs_diff(fcpu, fgpu);
    std::fprintf(stderr, "GPU F32/F32 GQA RERoT parity error = %.9g h1=[%.6g %.6g]/[%.6g %.6g]\n",
        ferror, fcpu[tdv], fcpu[tdv + 1], fgpu[tdv], fgpu[tdv + 1]);
    CHECK(fcpu.size() == fgpu.size());
    CHECK(ferror < 2.0e-4f);

    const auto vcpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        1.0f / std::sqrt(float(td)), nullptr, GGML_TYPE_F32, GGML_TYPE_TURBO2_0);
    const auto vgpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        1.0f / std::sqrt(float(td)), backend, GGML_TYPE_F32, GGML_TYPE_TURBO2_0);
    const float verror = max_abs_diff(vcpu, vgpu);
    std::fprintf(stderr, "GPU F32/Turbo2 RERoT parity error = %.9g h0=[%.6g %.6g]/[%.6g %.6g] h1=[%.6g %.6g]/[%.6g %.6g]\n",
        verror, vcpu[0], vcpu[1], vgpu[0], vgpu[1],
        vcpu[tdv], vcpu[tdv + 1], vgpu[tdv], vgpu[tdv + 1]);
    CHECK(vcpu.size() == vgpu.size());
    CHECK(verror < 1.0e-3f);

    const float tscale = 1.0f / std::sqrt(float(td));
    const std::vector<float> sinks = { -0.3f, 0.2f };
    const auto tcpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        tscale, nullptr, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, &sinks);
    const auto tgpu = run_indexed_op(
        td, tdv, tng, tnkv, thq, thkv, 1, tq, tk, tv, tentries, toffsets,
        tscale, backend, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, &sinks);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    const auto scpu = run_standard_op(
        td, tdv, tnkv, thq, thkv, tq, tk, tv, tscale,
        cpu_backend, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, sinks);
    const auto sgpu = run_standard_op(
        td, tdv, tnkv, thq, thkv, tq, tk, tv, tscale,
        backend, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, sinks);
    ggml_backend_free(cpu_backend);
    CHECK(tcpu.size() == tgpu.size());
    if (tcpu.size() == tgpu.size()) {
        const float error = max_abs_diff(tcpu, tgpu);
        if (error >= 4.0e-3f) {
            std::fprintf(stderr, "GPU Turbo4/Turbo2 RERoT parity error = %.9g\n", error);
        }
        CHECK(error < 4.0e-3f);
    }

    CHECK(scpu.size() == tcpu.size());
    CHECK(sgpu.size() == tgpu.size());
    if (scpu.size() == tcpu.size()) {
        const float error = max_abs_diff(scpu, tcpu);
        std::fprintf(stderr, "CPU ordinary/indexed Turbo+sinks gap = %.9g\n", error);
        CHECK(error < 4.0e-3f);
    }
    if (sgpu.size() == tgpu.size()) {
        const float error = max_abs_diff(sgpu, tgpu);
        std::fprintf(stderr, "GPU ordinary/indexed Turbo+sinks gap = %.9g\n", error);
        CHECK(error < 4.0e-3f);
    }

    // Production heads and more than two 128-entry tiles exercise D_split,
    // online-softmax carry, multiple Q groups, and split-K reduction together.
    constexpr int lnkv = 257;
    constexpr int lng = 3;
    constexpr int lnq = 2;
    constexpr int lhq = 16;
    constexpr int lhkv = 2;
    std::vector<float> lq(size_t(td) * lng * lhq);
    std::vector<float> lk(size_t(td) * lnkv * lhkv);
    std::vector<float> lv(size_t(tdv) * lnkv * lhkv);
    for (size_t i = 0; i < lq.size(); ++i) lq[i] = std::sin(float(i + 7) * 0.007f);
    for (size_t i = 0; i < lk.size(); ++i) lk[i] = std::cos(float(i + 11) * 0.009f);
    for (size_t i = 0; i < lv.size(); ++i) lv[i] = std::sin(float(i + 13) * 0.011f);

    std::vector<int32_t> lentries;
    lentries.reserve(size_t(2 * lnq * lnkv));
    for (int e = 0; e < lnkv; ++e) {
        lentries.push_back((e * 37) % lnkv);
        lentries.push_back(e < 129 ? 0 : 1);
    }
    for (int e = 0; e < lnkv; ++e) {
        lentries.push_back(lnkv - 1 - ((e * 53) % lnkv));
        lentries.push_back(e < 170 ? 2 : 1);
    }
    const std::vector<int32_t> loffsets = { 0, lnkv, 2 * lnkv };
    const auto lcpu = run_indexed_op(
        td, tdv, lng, lnkv, lhq, lhkv, lnq, lq, lk, lv, lentries, loffsets,
        tscale, nullptr, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
    const auto lgpu = run_indexed_op(
        td, tdv, lng, lnkv, lhq, lhkv, lnq, lq, lk, lv, lentries, loffsets,
        tscale, backend, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
    CHECK(lcpu.size() == lgpu.size());
    if (lcpu.size() == lgpu.size()) {
        const float error = max_abs_diff(lcpu, lgpu);
        std::fprintf(stderr, "GPU multi-tile D_split RERoT parity error = %.9g\n", error);
        CHECK(error < 4.0e-3f);
    }

    ggml_backend_free(backend);
}

static void test_cpu_helper_matches_op() {
    std::puts("--- ggml-cpu DDVR helper vs indexed op ---");

    constexpr int d = 16;
    constexpr int dv = 10;
    constexpr int hq = 4;
    constexpr int hkv = 2;
    constexpr int nkv = 6;
    constexpr int64_t q_virtual = 6;

    attn_ref::RopeConfig rope;
    rope.head_dim = d;
    rope.rotary_dim = 8;
    rope.theta = 10000.0;
    rope.interleaved = true;

    const std::vector<attn_ref::Span> spans = {
        { 0, 4, 3, 0 },
        { 4, 2, 30, 4 },
    };
    const int ng = int(spans.size());

    std::mt19937 rng(0xce11u);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> raw_q(size_t(hq) * d);
    std::vector<float> raw_k(size_t(hkv) * nkv * d);
    std::vector<float> raw_v(size_t(hkv) * nkv * dv);
    for (float & x : raw_q) {
        x = dist(rng);
    }
    for (float & x : raw_k) {
        x = dist(rng);
    }
    for (float & x : raw_v) {
        x = dist(rng);
    }

    ggml_cpu_rerot::RopeConfig cpu_rope;
    cpu_rope.head_dim = d;
    cpu_rope.rotary_dim = 8;
    cpu_rope.theta = 10000.0;
    cpu_rope.freq_scale = 1.0;
    cpu_rope.layout = ggml_cpu_rerot::RopeLayout::Interleaved;
    const std::vector<ggml_cpu_rerot::Span> cpu_spans = { { 0, 4, 3, 0 }, { 4, 2, 30, 4 } };

    std::vector<float> cpu_mat(size_t(hq) * dv, 0.0f);
    std::vector<float> cpu_qside(size_t(hq) * dv, 0.0f);
    CHECK(ggml_cpu_rerot::DdvrMaterializedGqa(raw_q.data(), raw_k.data(), raw_v.data(), nkv, dv, hq,
                                              hkv, q_virtual, cpu_spans.data(), cpu_spans.size(),
                                              cpu_rope, 0.0f, nullptr, 0,
                                              ggml_cpu_rerot::FrontierMode::Strong, nullptr, cpu_mat.data()));
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(raw_q.data(), raw_k.data(), raw_v.data(), nkv, dv, hq, hkv,
                                       q_virtual, cpu_spans.data(), cpu_spans.size(), cpu_rope, 0.0f,
                                       nullptr, 0, ggml_cpu_rerot::FrontierMode::Strong, nullptr,
                                       cpu_qside.data()));
    const float err = max_abs_diff(cpu_mat, cpu_qside);
    if (err >= 2.5e-5f) {
        std::fprintf(stderr, "cpu helper mat-vs-qside error = %.9g\n", err);
    }
    CHECK(err < 2.5e-5f);

    // Indexed op must match the same reference.
    std::vector<float> k_data(size_t(hkv) * nkv * d);
    std::vector<float> tmp(d);
    for (int h = 0; h < hkv; ++h) {
        for (int key = 0; key < nkv; ++key) {
            int64_t spos = key < 4 ? 3 + key : 30 + (key - 4);
            std::memcpy(tmp.data(), raw_k.data() + (size_t(h) * nkv + key) * d, size_t(d) * sizeof(float));
            attn_ref::rope_text(tmp.data(), rope, spos);
            std::memcpy(k_data.data() + (size_t(h) * nkv + key) * d, tmp.data(), size_t(d) * sizeof(float));
        }
    }
    const std::vector<float> q_data = build_q_groups(raw_q, d, hq, q_virtual, spans, rope);
    std::vector<int32_t> entries;
    for (int g = 0; g < ng; ++g) {
        for (uint32_t l = 0; l < spans[size_t(g)].key_count; ++l) {
            entries.push_back(int(spans[size_t(g)].key_begin) + int(l));
            entries.push_back(g);
        }
    }
    const std::vector<int32_t> offsets = { 0, int(entries.size() / 2) };
    const float scale = 1.0f / std::sqrt(float(d));
    const std::vector<float> op_out =
        run_indexed_op(d, dv, ng, nkv, hq, hkv, 1, q_data, k_data, raw_v, entries, offsets, scale);
    CHECK(op_out.size() == cpu_mat.size());
    CHECK(max_abs_diff(op_out, cpu_mat) < 1e-4f);
    CHECK(max_abs_diff(op_out, cpu_qside) < 1e-4f);
}

static void test_ragged_multi_reader_grouped_attention() {
    std::puts("--- ragged multi-reader grouped attention (§B.7, §B.13 Phase 4) ---");

    constexpr int d = 16;
    constexpr int dv = 8;
    constexpr int hq = 4;
    constexpr int hkv = 2;

    // Ragged configuration: 3 people with 3, 1, 2 readers -> total 6 readers (no 3x3=9 padding!)
    const std::vector<uint32_t> person_offsets = { 0, 3, 4, 6 };
    const int nq = 6;

    // Person 0 keys: 0..7 (8 keys)
    // Person 1 keys: 8..11 (4 keys)
    // Person 2 keys: 12..17 (6 keys)
    const int nkv = 18;

    std::vector<int32_t> entries;
    std::vector<int32_t> offsets;
    offsets.push_back(0);

    // Build entries per reader respecting person boundaries
    // Readers 0, 1, 2 belong to Person 0 -> read keys 0..7
    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 8; ++k) {
            entries.push_back(k);
            entries.push_back(r); // group/reader local
        }
        offsets.push_back(int(entries.size() / 2));
    }

    // Reader 3 belongs to Person 1 -> reads keys 8..11
    for (int k = 8; k < 12; ++k) {
        entries.push_back(k);
        entries.push_back(3);
    }
    offsets.push_back(int(entries.size() / 2));

    // Readers 4, 5 belong to Person 2 -> read keys 12..17
    for (int r = 0; r < 2; ++r) {
        for (int k = 12; k < 18; ++k) {
            entries.push_back(k);
            entries.push_back(4 + r);
        }
        offsets.push_back(int(entries.size() / 2));
    }

    CHECK(offsets.size() == size_t(nq + 1));

    // Generate deterministic test vectors matching 4D tensor strides [(h*ng + g)*d + id]
    std::vector<float> q(size_t(d) * nq * hq);
    std::vector<float> k(size_t(d) * nkv * hkv);
    std::vector<float> v(size_t(dv) * nkv * hkv);

    for (int h = 0; h < hq; ++h) {
        for (int g = 0; g < nq; ++g) {
            for (int id = 0; id < d; ++id) {
                q[(size_t(h) * nq + g) * d + id] = value_q(g, h, id);
            }
        }
    }
    for (int h = 0; h < hkv; ++h) {
        for (int key = 0; key < nkv; ++key) {
            for (int id = 0; id < d; ++id) {
                k[(size_t(h) * nkv + key) * d + id] = value_k(key, h, id);
            }
            for (int id = 0; id < dv; ++id) {
                v[(size_t(h) * nkv + key) * dv + id] = value_v(key, h, id);
            }
        }
    }

    const float scale = 1.0f / std::sqrt(float(d));
    const auto ref_out = reference(d, dv, hq, hkv, nq, entries, offsets, scale, 0.0f, {});
    const auto op_out = run_indexed_op(d, dv, nq, nkv, hq, hkv, nq, q, k, v, entries, offsets, scale);

    CHECK(op_out.size() == ref_out.size());
    const float err = max_abs_diff(op_out, ref_out);
    CHECK(err < 1e-4f);
    std::printf("ragged multi-reader grouped attention CPU error = %.9g\n", err);

    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device) {
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (backend) {
            const auto gpu_op_out = run_indexed_op(d, dv, nq, nkv, hq, hkv, nq, q, k, v, entries, offsets, scale, backend);
            CHECK(gpu_op_out.size() == ref_out.size());
            const float gpu_err = max_abs_diff(gpu_op_out, ref_out);
            CHECK(gpu_err < 2e-4f);
            std::printf("ragged multi-reader grouped attention GPU parity error = %.9g\n", gpu_err);
            ggml_backend_free(backend);
        }
    }
}

static void test_full_size_multi_reader_attention() {
    constexpr int d = 256, dv = 256, hq = 16, hkv = 2, nq = 8, ng = 32, nkv = 385;
    const float scale = 1.0f / std::sqrt(float(d));
    std::vector<float> q(size_t(d) * ng * hq), k(size_t(d) * nkv * hkv), v(size_t(dv) * nkv * hkv);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i % 997) * 0.173f);
    for (size_t i = 0; i < k.size(); ++i) k[i] = std::cos(float(i % 991) * 0.137f);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(float(i % 983) * 0.193f);
    std::vector<int32_t> entries, offsets{0};
    for (int row = 0; row < nq; ++row) {
        for (int j = 0; j < nkv - 13 * row; ++j) {
            const int key = (37 * j + 11 * row) % nkv;
            entries.push_back(key);
            entries.push_back(3 * row + key % 3);
        }
        offsets.push_back(int(entries.size() / 2));
    }
    // Independent double-precision global softmax, not another backend call.
    std::vector<float> expected(size_t(dv) * hq * nq);
    for (int row = 0; row < nq; ++row) {
        for (int h = 0; h < hq; ++h) {
            const int kh = h / (hq / hkv);
            std::vector<double> scores;
            double maximum = -INFINITY;
            for (int e = offsets[row]; e < offsets[row + 1]; ++e) {
                const int key = entries[2 * e], group = entries[2 * e + 1];
                double score = 0.0;
                for (int c = 0; c < d; ++c) score += double(q[(size_t(h) * ng + group) * d + c]) * k[(size_t(kh) * nkv + key) * d + c];
                scores.push_back(score * scale);
                maximum = std::max(maximum, scores.back());
            }
            double denominator = 0.0;
            for (double & score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (int c = 0; c < dv; ++c) {
                double out = 0.0;
                for (int e = offsets[row]; e < offsets[row + 1]; ++e) out += scores[size_t(e - offsets[row])] * v[(size_t(kh) * nkv + entries[2 * e]) * dv + c];
                expected[(size_t(row) * hq + h) * dv + c] = float(out / denominator);
            }
        }
    }
    const auto cpu = run_indexed_op(d, dv, ng, nkv, hq, hkv, nq, q, k, v, entries, offsets, scale);
    CHECK(max_abs_diff(cpu, expected) < 2e-5f);
    ggml_backend_load_all();
    const auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) { std::puts("SKIP full-size attention GPU: no GPU"); return; }
    ggml_backend_t gpu = ggml_backend_dev_init(device, nullptr);
    CHECK(gpu != nullptr);
    if (!gpu) return;
    const auto actual = run_indexed_op(d, dv, ng, nkv, hq, hkv, nq, q, k, v, entries, offsets, scale, gpu);
    const auto qcpu = run_indexed_op(d, dv, ng, nkv, hq, hkv, nq, q, k, v, entries, offsets, scale, nullptr, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
    const auto qgpu = run_indexed_op(d, dv, ng, nkv, hq, hkv, nq, q, k, v, entries, offsets, scale, gpu, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
    const float ferror = max_abs_diff(actual, expected), qerror = max_abs_diff(qcpu, qgpu);
    CHECK(ferror < 2e-4f);
    CHECK(qerror < 1e-3f);
    std::printf("full-size 8-reader/16-head/385-key attention: f32=%g turbo4/turbo2=%g\n", ferror, qerror);
    ggml_backend_free(gpu);
}

static void test_parallel_delta_gdn(int N, bool correlated, int gate_case, bool on_cpu, bool density_norm = false) {
    std::puts(on_cpu ? "--- CPU RERoT Parallel Delta GDN ---" : "--- Vulkan RERoT Parallel Delta GDN ---");

    ggml_backend_load_all();
    ggml_backend_dev_t device =
        ggml_backend_dev_by_type(on_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) {
        std::puts("SKIP: requested backend unavailable for Parallel Delta GDN");
        return;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend != nullptr);
    if (!backend) {
        return;
    }

    constexpr int D = 128;
    constexpr int H = 1;
    std::vector<float> q((size_t) D * H * N, 0.0f);
    std::vector<float> k((size_t) D * H * N, 0.0f);
    std::vector<float> v((size_t) D * H * N, 0.0f);
    std::vector<float> g((size_t) H * N);
    std::vector<float> beta((size_t) H * N);
    std::vector<float> state((size_t) D * D * H);

    for (int seq = 0; seq < N; ++seq) {
        q[(size_t) seq * D + seq] = 1.0f;
        k[(size_t) seq * D + seq] = 1.0f;
        g[seq] = -0.1f * float(seq + 1);
        beta[seq] = 0.3f + 0.1f * float(seq % 6);
        if (correlated) {
            double norm = 0.0;
            for (int row = 0; row < D; ++row) {
                const float value = 0.8f + std::sin(float((seq + 1) * (row + 3)) * 0.17f);
                k[(size_t) seq * D + row] = value;
                q[(size_t) seq * D + row] = std::cos(float(seq + row) * 0.11f) / std::sqrt(float(D));
                norm += double(value) * value;
            }
            for (int row = 0; row < D; ++row) {
                k[(size_t) seq * D + row] /= std::sqrt(norm);
            }
        }
        for (int col = 0; col < D; ++col) {
            v[(size_t) seq * D + col] =
                0.01f * float(1 + col) + 0.2f * float(seq);
        }
    }
    for (int col = 0; col < D; ++col) {
        for (int row = 0; row < D; ++row) {
            const float value =
                0.0001f * float(1 + row + 2 * col);
            state[(size_t) col * D + row] = value;
        }
    }

    // A closed write gate must stay closed, including in the singleton
    // native path. Large but finite values make a hidden beta floor visible.
    if (gate_case == 1 || gate_case == 2) {
        const int writer = N == 1 ? 0 : 1;
        beta[writer] = gate_case == 2 ? 1.0e-8f : 0.0f;
        for (int col = 0; col < D; ++col) {
            v[(size_t) writer * D + col] = 1024.0f;
        }
    }
    if (gate_case == 3) {
        // Repeated evidence is not one native write: it increases effective
        // delta gain even though mean log-decay stays fixed. This is a
        // mathematical behavior check, not a claim of serial equivalence.
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        for (int seq = 0; seq < N; ++seq) {
            q[size_t(seq) * D] = k[size_t(seq) * D] = 1.0f;
            g[seq] = -0.1f;
            beta[seq] = 0.3f;
            for (int col = 0; col < D; ++col) v[size_t(seq) * D + col] = 0.01f * float(col + 1);
        }
    }
    if (gate_case == 4) {
        // Ill-conditioned, correlated rows with saturated but valid gates.
        // Runs without external model/snapshot fixtures in normal CTest.
        for (int seq = 0; seq < N; ++seq) {
            beta[seq] = 0.99f + 0.001f * float(seq % 9);
            double norm = 0.0;
            for (int row = 0; row < D; ++row) {
                const float value = 1.0f + 0.03f * std::sin(float((seq + 1) * (row + 3)) * 0.17f);
                k[size_t(seq) * D + row] = value;
                norm += double(value) * value;
            }
            for (int row = 0; row < D; ++row) k[size_t(seq) * D + row] /= std::sqrt(norm);
        }
    }

    // Exact regularized Parallel DeltaNet block update. The GPU solves the
    // same state-column system with matrix-free preconditioned CG; pivoted
    // elimination supplies an independent numerical reference.
    std::vector<float> native_state((size_t) N * state.size());
    for (int seq = 0; seq < N; ++seq) {
        for (size_t i = 0; i < state.size(); ++i) {
            const float overlay = N == 1 || seq % 2 != 0
                ? 0.015625f * float(int((i + seq) % 7) - 3)
                : 0.0f;
            native_state[(size_t) seq * state.size() + i] = state[i] + overlay;
        }
    }
    std::vector<float> expected_state((size_t) D * D, 0.0f);
    std::vector<double> density(N, 1.0);
    if (density_norm) {
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
            if (i == j || beta[j] == 0.0f) continue;
            double dot = 0.0, ni = 0.0, nj = 0.0;
            for (int row = 0; row < D; ++row) {
                const double ki = k[size_t(i) * D + row], kj = k[size_t(j) * D + row];
                dot += ki * kj; ni += ki * ki; nj += kj * kj;
            }
            if (ni * nj > 1e-20) density[i] += dot * dot / (ni * nj);
        }
    }
    std::vector<double> expected_output((size_t) D * N, 0.0);
    std::vector<float> expected_hand((size_t) D * D * N, 0.0f);
    double mean_log_decay = 0.0;
    for (int seq = 0; seq < N; ++seq) {
        mean_log_decay += g[seq];
    }
    const double base_alpha = std::exp(mean_log_decay / N);

    for (int col = 0; col < D; ++col) {
        std::vector<double> rhs((size_t) N, 0.0);
        for (int seq = 0; seq < N; ++seq) {
            double projected = 0.0;
            for (int row = 0; row < D; ++row) {
                projected += base_alpha *
                    state[(size_t) col * D + row] *
                    k[(size_t) seq * D + row];
            }
            rhs[(size_t) seq] =
                v[(size_t) seq * D + col] - projected;
        }

        std::vector<double> weights((size_t) N, 0.0);
        if (N == 1) {
            // Strict native-recurrence degeneration; no epsilon perturbation.
            weights[0] = beta[0] * rhs[0];
        } else {
            std::vector<double> matrix((size_t) N * N, 0.0);
            for (int i = 0; i < N; ++i) {
                if (beta[i] == 0.0f) {
                    matrix[(size_t) i * N + i] = 1.0;
                    rhs[(size_t) i] = 0.0;
                    continue;
                }
                for (int j = i; j < N; ++j) {
                    if (beta[j] == 0.0f) {
                        continue;
                    }
                    double dot = 0.0;
                    for (int row = 0; row < D; ++row) {
                        dot += k[(size_t) i * D + row] *
                            k[(size_t) j * D + row];
                    }
                    matrix[(size_t) i * N + j] = dot;
                    matrix[(size_t) j * N + i] = dot;
                }
                matrix[(size_t) i * N + i] +=
                    density[i] * ((1.0 - beta[i]) / beta[i] + 1.0e-4);
            }

            std::vector<double> solve_rhs = rhs;
            bool solve_ok = true;
            for (int i = 0; i < N; ++i) {
                int pivot_row = i;
                double pivot_abs =
                    std::abs(matrix[(size_t) i * N + i]);
                for (int row = i + 1; row < N; ++row) {
                    const double candidate =
                        std::abs(matrix[(size_t) row * N + i]);
                    if (candidate > pivot_abs) {
                        pivot_abs = candidate;
                        pivot_row = row;
                    }
                }
                if (pivot_abs < 1.0e-12) {
                    solve_ok = false;
                    break;
                }
                if (pivot_row != i) {
                    for (int column = i; column < N; ++column) {
                        std::swap(
                            matrix[(size_t) i * N + column],
                            matrix[(size_t) pivot_row * N + column]);
                    }
                    std::swap(
                        solve_rhs[(size_t) i],
                        solve_rhs[(size_t) pivot_row]);
                }
                const double pivot = matrix[(size_t) i * N + i];
                for (int row = i + 1; row < N; ++row) {
                    const double factor =
                        matrix[(size_t) row * N + i] / pivot;
                    for (int column = i; column < N; ++column) {
                        matrix[(size_t) row * N + column] -=
                            factor * matrix[(size_t) i * N + column];
                    }
                    solve_rhs[(size_t) row] -=
                        factor * solve_rhs[(size_t) i];
                }
            }
            if (solve_ok) {
                for (int i = N - 1; i >= 0; --i) {
                    double value = solve_rhs[(size_t) i];
                    for (int column = i + 1; column < N; ++column) {
                        value -= matrix[(size_t) i * N + column] *
                            weights[(size_t) column];
                    }
                    weights[(size_t) i] =
                        value / matrix[(size_t) i * N + i];
                }
            }
        }

        std::vector<double> overlay_projection((size_t) N, 0.0);
        std::vector<double> native_projection((size_t) N, 0.0);
        for (int seq = 0; seq < N; ++seq) {
            for (int row = 0; row < D; ++row) {
                const size_t index = (size_t) col * D + row;
                overlay_projection[seq] += std::exp(double(g[seq])) *
                    (double(native_state[(size_t) seq * state.size() + index]) - state[index]) *
                    k[(size_t) seq * D + row];
                native_projection[seq] += std::exp(double(g[seq])) *
                    double(native_state[(size_t) seq * state.size() + index]) *
                    k[(size_t) seq * D + row];
            }
        }
        for (int row = 0; row < D; ++row) {
            double merged =
                base_alpha * state[(size_t) col * D + row];
            for (int seq = 0; seq < N; ++seq) {
                merged += k[(size_t) seq * D + row] *
                    weights[(size_t) seq];
            }
            expected_state[(size_t) col * D + row] = float(merged);
            for (int seq = 0; seq < N; ++seq) {
                const size_t index = (size_t) col * D + row;
                const double local_hand = std::exp(double(g[seq])) *
                    (double(native_state[(size_t) seq * state.size() + index]) - state[index]) -
                    beta[seq] * k[(size_t) seq * D + row] * overlay_projection[seq];
                const double shared_write = merged - base_alpha * state[index];
                const double self_echo = N == 1 ? 0.0 :
                    k[(size_t) seq * D + row] * weights[(size_t) seq] - shared_write / double(N);
                const double hand = local_hand + self_echo;
                // Commit and current-token read are separate: persist the
                // shared block + centered echo, but emit this Lane's exact
                // native GDN transition for the current token.
                const double native_delta = beta[seq] *
                    (v[(size_t) seq * D + col] - native_projection[seq]);
                const double native_candidate = std::exp(double(g[seq])) *
                    double(native_state[(size_t) seq * state.size() + index]) +
                    k[(size_t) seq * D + row] * native_delta;
                expected_output[(size_t) seq * D + col] +=
                    native_candidate * q[(size_t) seq * D + row] / std::sqrt(double(D));
                expected_hand[
                    ((size_t) seq * D + col) * D + row] =
                    float(hand);
            }
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr(ggml_init(params));
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, 1, N);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, 1, N);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, 1, N);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, N);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, N);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H, 1);
    ggml_tensor * tn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H, N);
    ggml_tensor * out =
        ggml_gated_delta_net_rbb(ctx, tq, tk, tv, tg, tb, ts, tn);
    CHECK(ggml_get_op_params_i32(out, 2) == 1);
    ggml_set_op_params_i32(out, 2, density_norm ? 1 : 0);

    CHECK(ggml_backend_supports_op(backend, out));
    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buffer != nullptr);
    if (!buffer) {
        ggml_backend_free(backend);
        return;
    }

    ggml_backend_tensor_set(tq, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(tk, k.data(), 0, k.size() * sizeof(float));
    ggml_backend_tensor_set(tv, v.data(), 0, v.size() * sizeof(float));
    ggml_backend_tensor_set(tg, g.data(), 0, g.size() * sizeof(float));
    ggml_backend_tensor_set(tb, beta.data(), 0, beta.size() * sizeof(float));
    ggml_backend_tensor_set(ts, state.data(), 0, state.size() * sizeof(float));
    ggml_backend_tensor_set(tn, native_state.data(), 0, native_state.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    const size_t state_offset = (size_t) D * H * N;
    const size_t hand_offset = state_offset + expected_state.size();
    std::vector<float> actual(
        hand_offset + expected_hand.size());
    ggml_backend_tensor_get(
        out, actual.data(), 0, actual.size() * sizeof(float));

    float state_error = 0.0f;
    float output_error = 0.0f;
    float hand_error = 0.0f;
    for (size_t i = 0; i < expected_state.size(); ++i) {
        CHECK(std::isfinite(actual[state_offset + i]));
        state_error = std::max(
            state_error, std::abs(actual[state_offset + i] - expected_state[i]));
    }
    for (int seq = 0; seq < N; ++seq) {
        for (int col = 0; col < D; ++col) {
            const size_t index = (size_t) seq * D + col;
            CHECK(std::isfinite(actual[index]));
            output_error = std::max(
                output_error,
                float(std::abs(actual[index] - expected_output[index])));
        }
    }
    for (size_t i = 0; i < expected_hand.size(); ++i) {
        CHECK(std::isfinite(actual[hand_offset + i]));
        hand_error = std::max(
            hand_error,
            std::abs(actual[hand_offset + i] - expected_hand[i]));
    }
    CHECK(state_error < 2e-4f);
    CHECK(output_error < 2e-5f);
    CHECK(hand_error < 2e-4f);
    if (gate_case == 3) {
        const double beta_eff = N == 1 ? double(beta[0]) :
            N * double(beta[0]) / (N * double(beta[0]) + density[0] * (1.0 - beta[0] + 1e-4 * double(beta[0])));
        const double old = std::exp(double(g[0])) * state[0];
        CHECK(std::abs(actual[state_offset] - (old + beta_eff * (v[0] - old))) < 2e-6);
        std::printf("duplicate writers N=%d density_mode=%d: beta=%g effective_beta=%.9g\n", N, density_norm, beta[0], beta_eff);
    }
    std::printf(
        "Parallel Delta GDN N=%d correlated=%d gate_case=%d state error = %.9g, output error = %.9g, hand error = %.9g\n",
        N, correlated, gate_case, state_error, output_error, hand_error);

    buffer.reset();
    ggml_backend_free(backend);
}

// Replay real, strided input tensors captured immediately after the RBB op,
// before any persistent-state copy. CPU double Cholesky and Vulkan CG see
// identical values; this avoids confounding two autoregressive trajectories.
static int replay_rbb_snapshot(const std::string & directory) {
    struct input_data { int64_t ne[4]; std::vector<float> values; };
    std::array<input_data, 7> inputs;
    int32_t density_mode = 0;
    std::ifstream mode_file(directory + "/density-mode.txt");
    if (mode_file && (!(mode_file >> density_mode) || (density_mode != 0 && density_mode != 1))) return 2;
    std::ifstream meta(directory + "/tensors.tsv");
    if (!meta) return 2;
    for (int i = 0; i < 7; ++i) {
        std::string line, name, type;
        if (!std::getline(meta, line)) return 2;
        std::istringstream fields(line);
        size_t nb[4], bytes = 0;
        fields >> name >> type;
        for (auto & d : inputs[i].ne) fields >> d;
        for (auto & stride : nb) fields >> stride;
        fields >> bytes;
        if (!fields || name != "src" + std::to_string(i) || type != "f32" || bytes > 256 * 1024 * 1024) return 2;
        std::vector<uint8_t> raw(bytes);
        std::ifstream file(directory + "/" + name + ".bin", std::ios::binary);
        if (!file.read(reinterpret_cast<char *>(raw.data()), std::streamsize(bytes))) return 2;
        const auto * ne = inputs[i].ne;
        for (int d = 0; d < 4; ++d) if (ne[d] <= 0 || ne[d] > 65536) return 2;
        for (int64_t w = 0; w < ne[3]; ++w) for (int64_t z = 0; z < ne[2]; ++z)
            for (int64_t y = 0; y < ne[1]; ++y) for (int64_t x = 0; x < ne[0]; ++x) {
                const size_t off = size_t(w)*nb[3] + size_t(z)*nb[2] + size_t(y)*nb[1] + size_t(x)*nb[0];
                if (off + sizeof(float) > bytes) return 2;
                float value;
                std::memcpy(&value, raw.data() + off, sizeof(value));
                inputs[i].values.push_back(value);
            }
    }
    ggml_backend_load_all();
    auto run = [&](enum ggml_backend_dev_type type) {
        std::vector<float> values;
        const auto device = ggml_backend_dev_by_type(type);
        if (!device) { ++failures; return values; }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        ggml_context_ptr ctx(ggml_init({16 * 1024 * 1024, nullptr, true}));
        std::array<ggml_tensor *, 7> in;
        for (size_t i = 0; i < in.size(); ++i) in[i] = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, inputs[i].ne);
        ggml_tensor * result = ggml_gated_delta_net_rbb(ctx.get(), in[0], in[1], in[2], in[3], in[4], in[5], in[6]);
        ggml_set_op_params_i32(result, 2, density_mode);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        for (size_t i = 0; i < in.size(); ++i) ggml_backend_tensor_set(in[i], inputs[i].values.data(), 0, inputs[i].values.size() * sizeof(float));
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, result);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        const size_t dim = size_t(in[2]->ne[0]), heads = size_t(in[2]->ne[1]), n = size_t(in[2]->ne[3]);
        values.resize(dim * heads * n + dim * dim * heads * (n + 1));
        ggml_backend_tensor_get(result, values.data(), 0, values.size() * sizeof(float));
        buffer.reset();
        ggml_backend_free(backend);
        return values;
    };
    const auto cpu = run(GGML_BACKEND_DEVICE_TYPE_CPU), gpu = run(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (cpu.empty() || gpu.size() != cpu.size()) return 2;
    const size_t d = size_t(inputs[2].ne[0]), h = size_t(inputs[2].ne[1]), n = size_t(inputs[2].ne[3]);
    const size_t bounds[] = {0, d*h*n, d*h*n + d*d*h, cpu.size()};
    const char * names[] = {"output", "brain", "hand"};
    for (size_t part = 0; part < 3; ++part) {
        double err = 0.0, ref = 0.0, err2 = 0.0, ref2 = 0.0;
        for (size_t i = bounds[part]; i < bounds[part + 1]; ++i) {
            CHECK(std::isfinite(gpu[i]));
            const double delta = double(gpu[i]) - cpu[i];
            err = std::max(err, std::abs(delta)); ref = std::max(ref, std::abs(double(cpu[i])));
            err2 += delta * delta; ref2 += double(cpu[i]) * cpu[i];
        }
        CHECK(err <= 2e-4 * std::max(1.0, ref));
        std::printf("RBB replay %s: max_error=%g reference_max=%g relative_l2=%g\n", names[part], err, ref, std::sqrt(err2 / std::max(1e-30, ref2)));
    }
    return failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    test_error_metric_rejects_invalid_outputs();
    if (argc == 3 && std::strcmp(argv[1], "--rbb-replay") == 0) return replay_rbb_snapshot(argv[2]);
    if (argc == 2 && std::strcmp(argv[1], "--precision-only") == 0) {
        // An explicitly requested GPU gate cannot succeed without running.
        test_attention_f32_precision(true);
        return failures == 0 ? 0 : 1;
    }
    std::puts("=== RERoT indexed attention test ===");
    test_indexed_basic();
    test_ddvr_imrope_via_indexed_op();
    test_frontier_strong_vs_lag1();
    test_vulkan_indexed_parity();
    test_attention_f32_precision();
    test_cpu_helper_matches_op();
    test_ragged_multi_reader_grouped_attention();
    test_full_size_multi_reader_attention();
    for (const bool on_cpu : {true, false}) {
        test_parallel_delta_gdn(1, false, 3, on_cpu);
        test_parallel_delta_gdn(8, false, 3, on_cpu);
        test_parallel_delta_gdn(4, true, 4, on_cpu);
        test_parallel_delta_gdn(7, true, 4, on_cpu);
        test_parallel_delta_gdn(1, false, 3, on_cpu, true);
        test_parallel_delta_gdn(8, false, 3, on_cpu, true);
        test_parallel_delta_gdn(3, false, 0, on_cpu, true);
        test_parallel_delta_gdn(7, true, 4, on_cpu, true);
        test_parallel_delta_gdn(3, true, 1, on_cpu, true);
    }
    for (bool on_cpu : {true, false}) {
        test_parallel_delta_gdn(1, false, 0, on_cpu);
        test_parallel_delta_gdn(3, false, 0, on_cpu);
        test_parallel_delta_gdn(12, true, 0, on_cpu);
        test_parallel_delta_gdn(32, true, 0, on_cpu);
        test_parallel_delta_gdn(64, true, 0, on_cpu);
        test_parallel_delta_gdn(1, false, 1, on_cpu);
        test_parallel_delta_gdn(1, false, 2, on_cpu);
        test_parallel_delta_gdn(3, false, 1, on_cpu);
        test_parallel_delta_gdn(3, true, 2, on_cpu);
    }
    std::printf("=== Results: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
