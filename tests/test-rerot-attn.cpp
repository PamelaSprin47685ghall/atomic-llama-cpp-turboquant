#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

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

int main() {
    std::puts("=== RERoT indexed attention test ===");

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
        return 1;
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
    std::printf("=== Results: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
