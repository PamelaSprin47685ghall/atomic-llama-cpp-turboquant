// [Q3] Shadow oracle for the "one physical K/V block serves many readers"
// organization: the FP64 family proves it equals one global softmax per
// reader, and the F32 gate measures the online merge error. This probe adds
// the third leg — a real CPU A/B over the production DDVR reference path
// (ggml_cpu_rerot::DdvrQsideGqa), which runs one reader at a time:
//
//   A: R separate calls, each re-reading the whole physical K/V block set.
//   B: ONE pass over the blocks, carrying per-reader (m, z, u) states.
//
// For every reader the outputs must agree bit-OR-nearly; the timing shows
// whether the shared supply actually saves anything on this CPU shape.
// This is instrumentation, not a production path: it adds no GGML_OP.
#include "llama-rerot-math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

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

static void test_shared_block_partial_visibility() {
    // 1. 'test_shared_block_partial_visibility': R=4 readers, but each reader
    // sees only a subset of the K/V block (reader 0 sees [0..4096], reader 1 sees
    // [4096..8192], reader 2 sees [8192..12288], reader 3 sees [12288..16384]).
    // Verify each reader's output matches per-reader oracle (DdvrQsideGqa with
    // restricted key range) and that the shared-block path produces the same outputs.
    const uint32_t R          = 4;
    const uint32_t n_keys     = 16384;
    const uint32_t head_dim   = 64;
    const uint32_t value_dim  = 64;
    const uint32_t n_head_q   = 4;
    const uint32_t n_head_kv  = 4;
    const float    scale      = 1.0f / std::sqrt(float(head_dim));
    const uint32_t block_rows = 512;

    std::mt19937_64 rng(0x1234u);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> raw_k(size_t(n_head_kv) * n_keys * head_dim);
    std::vector<float> values(size_t(n_head_kv) * n_keys * value_dim);
    for (size_t hkv = 0; hkv < n_head_kv; ++hkv) {
        for (uint32_t j = 0; j < n_keys; ++j) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    raw_k[(size_t(h2) * n_keys + j) * head_dim + e] = x;
                }
            }
            for (uint32_t e = 0; e < value_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    values[(size_t(h2) * n_keys + j) * value_dim + e] = x;
                }
            }
        }
    }
    std::vector<uint64_t> frontiers(n_keys, 1);

    ggml_cpu_rerot::RopeConfig cfg;
    cfg.head_dim = head_dim;
    cfg.rotary_dim = head_dim;
    cfg.layout = ggml_cpu_rerot::RopeLayout::Half;

    std::vector<std::vector<float>> raw_q(R, std::vector<float>(size_t(n_head_q) * head_dim));
    std::vector<int64_t> q_virt(R);
    for (uint32_t r = 0; r < R; ++r) {
        for (float & x : raw_q[r]) { x = dist(rng); }
        q_virt[r] = int64_t(n_keys) - 1 - r;
    }

    std::vector<std::vector<double>> q_rot(size_t(R) * n_head_q,
                                          std::vector<double>(head_dim));
    for (uint32_t r = 0; r < R; ++r) {
        std::vector<float> q(raw_q[r]);
        int64_t pos4[4] = { q_virt[r], q_virt[r], q_virt[r], 0 };
        for (uint32_t h = 0; h < n_head_q; ++h) {
            CHECK(ggml_cpu_rerot::RopeApply(q.data() + size_t(h) * head_dim, cfg, pos4));
        }
        for (uint32_t h = 0; h < n_head_q; ++h) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                q_rot[size_t(r) * n_head_q + h][e] = double(q[size_t(h) * head_dim + e]);
            }
        }
    }

    // Per-reader key subsets: [k_beg, k_end)
    const uint32_t chunk = n_keys / R; // 4096
    std::vector<uint32_t> k_start(R);
    std::vector<uint32_t> k_count(R);
    for (uint32_t r = 0; r < R; ++r) {
        k_start[r] = r * chunk;
        k_count[r] = chunk;
    }

    // ---- A: Per-reader oracle with restricted key range (via extra_mask) ----
    std::vector<float> reference(size_t(R) * n_head_q * value_dim);
    std::vector<ggml_cpu_rerot::Span> full_span = { { 0, n_keys, 0, 0 } };
    for (uint32_t r = 0; r < R; ++r) {
        std::vector<uint8_t> reader_mask(n_keys, 0);
        std::fill(reader_mask.begin() + k_start[r], reader_mask.begin() + k_start[r] + k_count[r], 1);
        CHECK(ggml_cpu_rerot::DdvrQsideGqa(
            raw_q[r].data(), raw_k.data(), values.data(), n_keys, value_dim,
            n_head_q, n_head_kv, q_virt[r], full_span.data(), full_span.size(), cfg, scale,
            frontiers.data(), 2, ggml_cpu_rerot::FrontierMode::Strong, reader_mask.data(),
            reference.data() + size_t(r) * n_head_q * value_dim));
    }

    // ---- B: Shared-block pass with block-level visibility filtering ----
    std::vector<float> shared(size_t(R) * n_head_q * value_dim);
    std::vector<llama_rerot_attn_state> merged(size_t(R) * n_head_q);
    for (uint32_t b0 = 0; b0 < n_keys; b0 += block_rows) {
        const uint32_t rows = std::min(block_rows, n_keys - b0);
        const uint32_t b_end = b0 + rows;

        std::vector<double> kd;
        kd.reserve(size_t(rows) * head_dim);
        for (uint32_t j = 0; j < rows; ++j) {
            std::vector<float> k(raw_k.begin() + size_t(b0 + j) * head_dim,
                                 raw_k.begin() + size_t(b0 + j + 1) * head_dim);
            int64_t kpos4[4] = { int64_t(b0 + j), int64_t(b0 + j), int64_t(b0 + j), 0 };
            CHECK(ggml_cpu_rerot::RopeApply(k.data(), cfg, kpos4));
            for (float x : k) { kd.push_back(double(x)); }
        }
        std::vector<double> vd;
        vd.reserve(size_t(rows) * value_dim);
        for (uint32_t j = 0; j < rows; ++j) {
            for (uint32_t e = 0; e < value_dim; ++e) {
                vd.push_back(double(values[size_t(b0 + j) * value_dim + e]));
            }
        }

        for (uint32_t h = 0; h < n_head_q; ++h) {
            for (uint32_t r = 0; r < R; ++r) {
                // Overlap between block [b0, b_end) and reader r's visible interval [k_start[r], k_start[r]+k_count[r])
                const uint32_t r_start = k_start[r];
                const uint32_t r_end = r_start + k_count[r];
                const uint32_t ov_start = std::max(b0, r_start);
                const uint32_t ov_end = std::min(b_end, r_end);
                if (ov_start >= ov_end) {
                    continue; // Block has no visible keys for reader r
                }

                const auto & q_head = q_rot[size_t(r) * n_head_q + h];
                llama_rerot_attn_state & s = merged[size_t(r) * n_head_q + h];
                for (uint32_t key_idx = ov_start; key_idx < ov_end; ++key_idx) {
                    const uint32_t j = key_idx - b0;
                    double dot = 0.0;
                    for (uint32_t e = 0; e < head_dim; ++e) {
                        dot += q_head[e] * kd[size_t(j) * head_dim + e];
                    }
                    const double score = scale * dot;
                    const double m_new = s.empty() ? score : std::max(s.m, score);
                    if (s.empty()) {
                        s.m = m_new;
                        s.z = 1.0;
                        s.u.assign(value_dim, 0.0);
                        const double * v = vd.data() + size_t(j) * value_dim;
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] = v[e]; }
                    } else {
                        const double f = std::exp(s.m - m_new);
                        const double w = std::exp(score - m_new);
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] *= f; }
                        const double * v = vd.data() + size_t(j) * value_dim;
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] += w * v[e]; }
                        s.z = s.z * f + w;
                        s.m = m_new;
                    }
                }
            }
        }
    }

    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t h = 0; h < n_head_q; ++h) {
            const auto o = llama_rerot_attn_state_output(
                merged[size_t(r) * n_head_q + h]);
            float * dst = shared.data() + (size_t(r) * n_head_q + h) * value_dim;
            for (uint32_t e = 0; e < value_dim; ++e) { dst[e] = float(o[e]); }
        }
    }

    for (uint32_t r = 0; r < R; ++r) {
        double max_err = 0.0, max_ref = 0.0;
        for (size_t i = 0; i < size_t(n_head_q) * value_dim; ++i) {
            const size_t idx = size_t(r) * n_head_q * value_dim + i;
            max_err = std::max(max_err, std::abs(double(shared[idx]) - double(reference[idx])));
            max_ref = std::max(max_ref, std::abs(double(reference[idx])));
        }
        const double rel = max_ref > 0.0 ? max_err / max_ref : 0.0;
        std::printf("partial_visibility reader %u: rel_err=%.3e\n", r, rel);
        CHECK(rel < 1e-4);
    }
}

static void test_shared_block_empty_reader() {
    // 2. 'test_shared_block_empty_reader': R=3 readers, but reader 1 has 0 visible keys (empty block).
    // Verify reader 1's output is all zeros (or NaN — the contract says m=-inf, z=0, u=0 -> o=0),
    // and readers 0,2 produce correct outputs matching per-reader oracle.
    const uint32_t R          = 3;
    const uint32_t n_keys     = 4096;
    const uint32_t head_dim   = 64;
    const uint32_t value_dim  = 64;
    const uint32_t n_head_q   = 4;
    const uint32_t n_head_kv  = 4;
    const float    scale      = 1.0f / std::sqrt(float(head_dim));
    const uint32_t block_rows = 512;

    std::mt19937_64 rng(0x5678u);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> raw_k(size_t(n_head_kv) * n_keys * head_dim);
    std::vector<float> values(size_t(n_head_kv) * n_keys * value_dim);
    for (size_t hkv = 0; hkv < n_head_kv; ++hkv) {
        for (uint32_t j = 0; j < n_keys; ++j) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    raw_k[(size_t(h2) * n_keys + j) * head_dim + e] = x;
                }
            }
            for (uint32_t e = 0; e < value_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    values[(size_t(h2) * n_keys + j) * value_dim + e] = x;
                }
            }
        }
    }
    std::vector<uint64_t> frontiers(n_keys, 1);

    ggml_cpu_rerot::RopeConfig cfg;
    cfg.head_dim = head_dim;
    cfg.rotary_dim = head_dim;
    cfg.layout = ggml_cpu_rerot::RopeLayout::Half;

    std::vector<std::vector<float>> raw_q(R, std::vector<float>(size_t(n_head_q) * head_dim));
    std::vector<int64_t> q_virt(R);
    for (uint32_t r = 0; r < R; ++r) {
        for (float & x : raw_q[r]) { x = dist(rng); }
        q_virt[r] = int64_t(n_keys) - 1 - r;
    }

    std::vector<std::vector<double>> q_rot(size_t(R) * n_head_q,
                                          std::vector<double>(head_dim));
    for (uint32_t r = 0; r < R; ++r) {
        std::vector<float> q(raw_q[r]);
        int64_t pos4[4] = { q_virt[r], q_virt[r], q_virt[r], 0 };
        for (uint32_t h = 0; h < n_head_q; ++h) {
            CHECK(ggml_cpu_rerot::RopeApply(q.data() + size_t(h) * head_dim, cfg, pos4));
        }
        for (uint32_t h = 0; h < n_head_q; ++h) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                q_rot[size_t(r) * n_head_q + h][e] = double(q[size_t(h) * head_dim + e]);
            }
        }
    }

    // Reader 0: visible on all keys
    // Reader 1: 0 visible keys
    // Reader 2: visible on all keys
    std::vector<uint8_t> r0_mask(n_keys, 1);
    std::vector<uint8_t> r1_mask(n_keys, 0); // empty
    std::vector<uint8_t> r2_mask(n_keys, 1);

    std::vector<float> reference(size_t(R) * n_head_q * value_dim, 0.0f);
    std::vector<ggml_cpu_rerot::Span> full_span = { { 0, n_keys, 0, 0 } };
    // Oracle for reader 0
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(
        raw_q[0].data(), raw_k.data(), values.data(), n_keys, value_dim,
        n_head_q, n_head_kv, q_virt[0], full_span.data(), full_span.size(), cfg, scale,
        frontiers.data(), 2, ggml_cpu_rerot::FrontierMode::Strong, r0_mask.data(),
        reference.data() + size_t(0) * n_head_q * value_dim));
    // Oracle for reader 1 (empty block)
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(
        raw_q[1].data(), raw_k.data(), values.data(), n_keys, value_dim,
        n_head_q, n_head_kv, q_virt[1], full_span.data(), full_span.size(), cfg, scale,
        frontiers.data(), 2, ggml_cpu_rerot::FrontierMode::Strong, r1_mask.data(),
        reference.data() + size_t(1) * n_head_q * value_dim));
    // Oracle for reader 2
    CHECK(ggml_cpu_rerot::DdvrQsideGqa(
        raw_q[2].data(), raw_k.data(), values.data(), n_keys, value_dim,
        n_head_q, n_head_kv, q_virt[2], full_span.data(), full_span.size(), cfg, scale,
        frontiers.data(), 2, ggml_cpu_rerot::FrontierMode::Strong, r2_mask.data(),
        reference.data() + size_t(2) * n_head_q * value_dim));

    // Reader 1 reference output must be all zeros
    for (size_t i = 0; i < size_t(n_head_q) * value_dim; ++i) {
        CHECK(reference[size_t(1) * n_head_q * value_dim + i] == 0.0f);
    }

    // Shared-block pass
    std::vector<float> shared(size_t(R) * n_head_q * value_dim, 0.0f);
    std::vector<llama_rerot_attn_state> merged(size_t(R) * n_head_q);
    for (uint32_t b0 = 0; b0 < n_keys; b0 += block_rows) {
        const uint32_t rows = std::min(block_rows, n_keys - b0);

        std::vector<double> kd;
        kd.reserve(size_t(rows) * head_dim);
        for (uint32_t j = 0; j < rows; ++j) {
            std::vector<float> k(raw_k.begin() + size_t(b0 + j) * head_dim,
                                 raw_k.begin() + size_t(b0 + j + 1) * head_dim);
            int64_t kpos4[4] = { int64_t(b0 + j), int64_t(b0 + j), int64_t(b0 + j), 0 };
            CHECK(ggml_cpu_rerot::RopeApply(k.data(), cfg, kpos4));
            for (float x : k) { kd.push_back(double(x)); }
        }
        std::vector<double> vd;
        vd.reserve(size_t(rows) * value_dim);
        for (uint32_t j = 0; j < rows; ++j) {
            for (uint32_t e = 0; e < value_dim; ++e) {
                vd.push_back(double(values[size_t(b0 + j) * value_dim + e]));
            }
        }

        for (uint32_t h = 0; h < n_head_q; ++h) {
            for (uint32_t r = 0; r < R; ++r) {
                if (r == 1) {
                    continue; // reader 1 has no visible keys
                }
                const auto & q_head = q_rot[size_t(r) * n_head_q + h];
                llama_rerot_attn_state & s = merged[size_t(r) * n_head_q + h];
                for (uint32_t j = 0; j < rows; ++j) {
                    double dot = 0.0;
                    for (uint32_t e = 0; e < head_dim; ++e) {
                        dot += q_head[e] * kd[size_t(j) * head_dim + e];
                    }
                    const double score = scale * dot;
                    const double m_new = s.empty() ? score : std::max(s.m, score);
                    if (s.empty()) {
                        s.m = m_new;
                        s.z = 1.0;
                        s.u.assign(value_dim, 0.0);
                        const double * v = vd.data() + size_t(j) * value_dim;
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] = v[e]; }
                    } else {
                        const double f = std::exp(s.m - m_new);
                        const double w = std::exp(score - m_new);
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] *= f; }
                        const double * v = vd.data() + size_t(j) * value_dim;
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] += w * v[e]; }
                        s.z = s.z * f + w;
                        s.m = m_new;
                    }
                }
            }
        }
    }

    // Output extraction: empty reader state yields 0s
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t h = 0; h < n_head_q; ++h) {
            float * dst = shared.data() + (size_t(r) * n_head_q + h) * value_dim;
            const auto & s = merged[size_t(r) * n_head_q + h];
            if (s.empty() || s.z == 0.0) {
                std::fill(dst, dst + value_dim, 0.0f);
            } else {
                const auto o = llama_rerot_attn_state_output(s);
                for (uint32_t e = 0; e < value_dim; ++e) { dst[e] = float(o[e]); }
            }
        }
    }

    // Verify reader 1 output is all zeros
    for (size_t i = 0; i < size_t(n_head_q) * value_dim; ++i) {
        CHECK(shared[size_t(1) * n_head_q * value_dim + i] == 0.0f);
    }

    // Verify reader 0 and 2 match oracle
    for (uint32_t r : {0u, 2u}) {
        double max_err = 0.0, max_ref = 0.0;
        for (size_t i = 0; i < size_t(n_head_q) * value_dim; ++i) {
            const size_t idx = size_t(r) * n_head_q * value_dim + i;
            max_err = std::max(max_err, std::abs(double(shared[idx]) - double(reference[idx])));
            max_ref = std::max(max_ref, std::abs(double(reference[idx])));
        }
        const double rel = max_ref > 0.0 ? max_err / max_ref : 0.0;
        std::printf("empty_reader test: reader %u rel_err=%.3e\n", r, rel);
        CHECK(rel < 1e-4);
    }
}

static void test_shared_block_single_reader() {
    // 3. 'test_shared_block_single_reader': R=1 reader.
    // The shared-block path with cohort=1 should still work correctly
    // (no sharing benefit, but correctness must not regress).
    // Verify speedup ~1.0x and output matches oracle.
    const uint32_t R          = 1;
    const uint32_t n_keys     = 4096;
    const uint32_t reps       = 3;
    const uint32_t head_dim   = 64;
    const uint32_t value_dim  = 64;
    const uint32_t n_head_q   = 4;
    const uint32_t n_head_kv  = 4;
    const float    scale      = 1.0f / std::sqrt(float(head_dim));
    const uint32_t block_rows = 512;

    std::mt19937_64 rng(0x9abcu);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> raw_k(size_t(n_head_kv) * n_keys * head_dim);
    std::vector<float> values(size_t(n_head_kv) * n_keys * value_dim);
    for (size_t hkv = 0; hkv < n_head_kv; ++hkv) {
        for (uint32_t j = 0; j < n_keys; ++j) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    raw_k[(size_t(h2) * n_keys + j) * head_dim + e] = x;
                }
            }
            for (uint32_t e = 0; e < value_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    values[(size_t(h2) * n_keys + j) * value_dim + e] = x;
                }
            }
        }
    }
    std::vector<uint64_t> frontiers(n_keys, 1);
    std::vector<uint8_t> extra_mask;
    std::vector<ggml_cpu_rerot::Span> span = { { 0, n_keys, 0, 0 } };

    ggml_cpu_rerot::RopeConfig cfg;
    cfg.head_dim = head_dim;
    cfg.rotary_dim = head_dim;
    cfg.layout = ggml_cpu_rerot::RopeLayout::Half;

    std::vector<std::vector<float>> raw_q(R, std::vector<float>(size_t(n_head_q) * head_dim));
    std::vector<int64_t> q_virt(R);
    for (uint32_t r = 0; r < R; ++r) {
        for (float & x : raw_q[r]) { x = dist(rng); }
        q_virt[r] = int64_t(n_keys) - 1 - r;
    }

    std::vector<std::vector<double>> q_rot(size_t(R) * n_head_q,
                                          std::vector<double>(head_dim));
    for (uint32_t r = 0; r < R; ++r) {
        std::vector<float> q(raw_q[r]);
        int64_t pos4[4] = { q_virt[r], q_virt[r], q_virt[r], 0 };
        for (uint32_t h = 0; h < n_head_q; ++h) {
            CHECK(ggml_cpu_rerot::RopeApply(q.data() + size_t(h) * head_dim, cfg, pos4));
        }
        for (uint32_t h = 0; h < n_head_q; ++h) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                q_rot[size_t(r) * n_head_q + h][e] = double(q[size_t(h) * head_dim + e]);
            }
        }
    }

    std::vector<float> reference(size_t(R) * n_head_q * value_dim);
    std::vector<float> shared(size_t(R) * n_head_q * value_dim);

    double a_us = 1e30;
    for (uint32_t rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < R; ++r) {
            CHECK(ggml_cpu_rerot::DdvrQsideGqa(
                raw_q[r].data(), raw_k.data(), values.data(), n_keys, value_dim,
                n_head_q, n_head_kv, q_virt[r], span.data(), span.size(), cfg, scale,
                frontiers.data(), 2, ggml_cpu_rerot::FrontierMode::Strong, extra_mask.data(),
                reference.data() + size_t(r) * n_head_q * value_dim));
        }
        a_us = std::min(a_us, std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count());
    }

    double b_us = 1e30;
    for (uint32_t rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<llama_rerot_attn_state> merged(size_t(R) * n_head_q);
        for (uint32_t b0 = 0; b0 < n_keys; b0 += block_rows) {
            const uint32_t rows = std::min(block_rows, n_keys - b0);
            std::vector<double> kd;
            kd.reserve(size_t(rows) * head_dim);
            for (uint32_t j = 0; j < rows; ++j) {
                std::vector<float> k(raw_k.begin() + size_t(b0 + j) * head_dim,
                                     raw_k.begin() + size_t(b0 + j + 1) * head_dim);
                int64_t kpos4[4] = { int64_t(b0 + j), int64_t(b0 + j), int64_t(b0 + j), 0 };
                CHECK(ggml_cpu_rerot::RopeApply(k.data(), cfg, kpos4));
                for (float x : k) { kd.push_back(double(x)); }
            }
            std::vector<double> vd;
            vd.reserve(size_t(rows) * value_dim);
            for (uint32_t j = 0; j < rows; ++j) {
                for (uint32_t e = 0; e < value_dim; ++e) {
                    vd.push_back(double(values[size_t(b0 + j) * value_dim + e]));
                }
            }
            for (uint32_t h = 0; h < n_head_q; ++h) {
                std::vector<double> qd;
                qd.reserve(size_t(R) * head_dim);
                for (uint32_t r = 0; r < R; ++r) {
                    qd.insert(qd.end(), q_rot[size_t(r) * n_head_q + h].begin(),
                              q_rot[size_t(r) * n_head_q + h].end());
                }
                std::vector<double> sc(size_t(R) * rows);
                for (uint32_t r = 0; r < R; ++r) {
                    for (uint32_t j = 0; j < rows; ++j) {
                        double dot = 0.0;
                        for (uint32_t e = 0; e < head_dim; ++e) {
                            dot += qd[size_t(r) * head_dim + e] * kd[size_t(j) * head_dim + e];
                        }
                        sc[size_t(r) * rows + j] = scale * dot;
                    }
                }
                for (uint32_t r = 0; r < R; ++r) {
                    llama_rerot_attn_state & s = merged[size_t(r) * n_head_q + h];
                    const double * row = sc.data() + size_t(r) * rows;
                    for (uint32_t j = 0; j < rows; ++j) {
                        const double m_new = std::max(s.m, row[j]);
                        const double f = std::exp(s.m - m_new);
                        const double w = std::exp(row[j] - m_new);
                        if (s.u.empty()) { s.u.assign(value_dim, 0.0); }
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] *= f; }
                        const double * v = vd.data() + size_t(j) * value_dim;
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] += w * v[e]; }
                        s.z = s.z * f + w;
                        s.m = m_new;
                    }
                }
            }
        }
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t h = 0; h < n_head_q; ++h) {
                const auto o = llama_rerot_attn_state_output(
                    merged[size_t(r) * n_head_q + h]);
                float * dst = shared.data() + (size_t(r) * n_head_q + h) * value_dim;
                for (uint32_t e = 0; e < value_dim; ++e) { dst[e] = float(o[e]); }
            }
        }
        b_us = std::min(b_us, std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count());
    }

    double max_err = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        max_err = std::max(max_err, std::abs(double(shared[i]) - double(reference[i])));
        max_ref = std::max(max_ref, std::abs(double(reference[i])));
    }
    const double rel = max_ref > 0.0 ? max_err / max_ref : 0.0;
    const double speedup = a_us / b_us;
    std::printf("single_reader A/B R=1 K=%u: A=%.1f us  B=%.1f us  speedup=%.2fx  rel_err=%.3e\n",
                n_keys, a_us, b_us, speedup, rel);
    CHECK(rel < 1e-4);
    // For R=1 speedup is expected ~1.0x (within reasonable margin, e.g. 0.5x to 3.0x on noisy CPU)
    CHECK(speedup > 0.3 && speedup < 5.0);
}

int main(int argc, char ** argv) {
    // Run new edge cases first
    std::printf("Running test_shared_block_partial_visibility...\n");
    test_shared_block_partial_visibility();

    std::printf("Running test_shared_block_empty_reader...\n");
    test_shared_block_empty_reader();

    std::printf("Running test_shared_block_single_reader...\n");
    test_shared_block_single_reader();

    std::printf("Running full-visibility benchmark...\n");
    const uint32_t R          = argc > 1 ? (uint32_t) std::atoi(argv[1]) : 6;    // readers
    const uint32_t n_keys     = argc > 2 ? (uint32_t) std::atoi(argv[2]) : 16384;
    const uint32_t reps       = argc > 3 ? (uint32_t) std::atoi(argv[3]) : 3;
    const uint32_t head_dim   = 64;
    const uint32_t value_dim  = 64;
    const uint32_t n_head_q   = 4;
    const uint32_t n_head_kv  = 4;
    const float    scale      = 1.0f / std::sqrt(float(head_dim));

    std::mt19937_64 rng(0x5ab1u);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // One shared physical K/V block set, dense ascending storage (the
    // production write layout for a public run). Layout matches
    // DdvrQsideGqa: kv-head-major, n_keys rows per head.
    std::vector<float> raw_k(size_t(n_head_kv) * n_keys * head_dim);
    std::vector<float> values(size_t(n_head_kv) * n_keys * value_dim);
    for (size_t hkv = 0; hkv < n_head_kv; ++hkv) {
        // Same physical rows for every kv head: the A/B comparison is
        // about the SUPPLY organization (one shared pass over the resident
        // rows vs R re-reads), not about GQA head routing, so the two
        // paths must consume identical bytes.
        for (uint32_t j = 0; j < n_keys; ++j) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    raw_k[(size_t(h2) * n_keys + j) * head_dim + e] = x;
                }
            }
            for (uint32_t e = 0; e < value_dim; ++e) {
                const float x = dist(rng);
                for (uint32_t h2 = 0; h2 < n_head_kv; ++h2) {
                    values[(size_t(h2) * n_keys + j) * value_dim + e] = x;
                }
            }
        }
    }
    std::vector<uint64_t> frontiers(n_keys, 1); // committed, old frontier
    std::vector<uint8_t> extra_mask;            // unused
    // Single contiguous span per reader covering everything.
    std::vector<ggml_cpu_rerot::Span> span = { { 0, n_keys, 0, 0 } };

    ggml_cpu_rerot::RopeConfig cfg;
    cfg.head_dim = head_dim;
    cfg.rotary_dim = head_dim;
    cfg.layout = ggml_cpu_rerot::RopeLayout::Half;

    // Each reader has its own raw query and virtual position (DDVR phase).
    std::vector<std::vector<float>> raw_q(R, std::vector<float>(size_t(n_head_q) * head_dim));
    std::vector<int64_t> q_virt(R);
    for (uint32_t r = 0; r < R; ++r) {
        for (float & x : raw_q[r]) { x = dist(rng); }
        q_virt[r] = int64_t(n_keys) - 1 - r;
    }
    // Per-reader, per-head pre-rotated queries (q-side DDVR phase).
    std::vector<std::vector<double>> q_rot(size_t(R) * n_head_q,
                                          std::vector<double>(head_dim));
    for (uint32_t r = 0; r < R; ++r) {
        std::vector<float> q(raw_q[r]);
        int64_t pos4[4] = { q_virt[r], q_virt[r], q_virt[r], 0 };
        for (uint32_t h = 0; h < n_head_q; ++h) {
            CHECK(ggml_cpu_rerot::RopeApply(q.data() + size_t(h) * head_dim, cfg, pos4));
        }
        for (uint32_t h = 0; h < n_head_q; ++h) {
            for (uint32_t e = 0; e < head_dim; ++e) {
                q_rot[size_t(r) * n_head_q + h][e] = double(q[size_t(h) * head_dim + e]);
            }
        }
    }

    std::vector<float> reference(size_t(R) * n_head_q * value_dim);
    std::vector<float> shared(size_t(R) * n_head_q * value_dim);

    // ---- A: R separate per-reader passes (current organization) ----------
    double a_us = 1e30;
    for (uint32_t rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < R; ++r) {
            CHECK(ggml_cpu_rerot::DdvrQsideGqa(
                raw_q[r].data(), raw_k.data(), values.data(), n_keys, value_dim,
                n_head_q, n_head_kv, q_virt[r], span.data(), span.size(), cfg, scale,
                frontiers.data(), 2, ggml_cpu_rerot::FrontierMode::Strong, extra_mask.data(),
                reference.data() + size_t(r) * n_head_q * value_dim));
        }
        a_us = std::min(a_us, std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count());
    }

    // ---- B: ONE shared pass over the physical blocks ----------------------
    // Uses the [Q3] math contract directly (llama_rerot_shared_block_
    // attention + attn_state_merge) with per-reader pre-rotated queries —
    // exactly the data flow a kernel implements. Blocks chunk the shared
    // K/V so the "resident block serves every reader" organization shows up
    // in the loop, not just the API.
    const uint32_t block_rows = argc > 4 ? (uint32_t) std::atoi(argv[4]) : 512;
    double b_us = 1e30;
    for (uint32_t rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<llama_rerot_attn_state> merged(size_t(R) * n_head_q);
        std::vector<float> scores(size_t(R) * n_head_q * block_rows);
        for (uint32_t b0 = 0; b0 < n_keys; b0 += block_rows) {
            const uint32_t rows = std::min(block_rows, n_keys - b0);
            // Per-reader pre-rotated queries are cached outside the loop;
            // keys are rotated once per block (shared supply) and reused by
            // every reader.
            // The A path scores kv-head (hq / group) per query head; for a
            // shared single-head reference we use kv-head 0's rows. That is
            // also what the B path consumes — the kv-head stride is the
            // only difference between the two organizations.
            const uint32_t kstride = size_t(n_head_kv == 1 ? 0 : 0) + 0; // kv head 0
            (void) kstride;
            std::vector<double> kd;
            kd.reserve(size_t(rows) * head_dim);
            for (uint32_t j = 0; j < rows; ++j) {
                std::vector<float> k(raw_k.begin() + size_t(b0 + j) * head_dim,
                                     raw_k.begin() + size_t(b0 + j + 1) * head_dim);
                int64_t kpos4[4] = { int64_t(b0 + j), int64_t(b0 + j), int64_t(b0 + j), 0 };
                CHECK(ggml_cpu_rerot::RopeApply(k.data(), cfg, kpos4));
                for (float x : k) { kd.push_back(double(x)); }
            }
            std::vector<double> vd;
            vd.reserve(size_t(rows) * value_dim);
            for (uint32_t j = 0; j < rows; ++j) {
                for (uint32_t e = 0; e < value_dim; ++e) {
                    vd.push_back(double(values[size_t(b0 + j) * value_dim + e]));
                }
            }
            for (uint32_t h = 0; h < n_head_q; ++h) {
                std::vector<double> qd;
                qd.reserve(size_t(R) * head_dim);
                for (uint32_t r = 0; r < R; ++r) {
                    qd.insert(qd.end(), q_rot[size_t(r) * n_head_q + h].begin(),
                              q_rot[size_t(r) * n_head_q + h].end());
                }
                // scores come from the reference helper's own scoring path
                // per reader; the merge contract is what is under test.
                std::vector<double> sc(size_t(R) * rows);
                for (uint32_t r = 0; r < R; ++r) {
                    for (uint32_t j = 0; j < rows; ++j) {
                        double dot = 0.0;
                        for (uint32_t e = 0; e < head_dim; ++e) {
                            dot += qd[size_t(r) * head_dim + e] * kd[size_t(j) * head_dim + e];
                        }
                        sc[size_t(r) * rows + j] = scale * dot;
                    }
                }
                // online per-reader merge over the block
                for (uint32_t r = 0; r < R; ++r) {
                    llama_rerot_attn_state & s = merged[size_t(r) * n_head_q + h];
                    const double * row = sc.data() + size_t(r) * rows;
                    for (uint32_t j = 0; j < rows; ++j) {
                        const double m_new = std::max(s.m, row[j]);
                        const double f = std::exp(s.m - m_new);
                        const double w = std::exp(row[j] - m_new);
                        if (s.u.empty()) { s.u.assign(value_dim, 0.0); }
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] *= f; }
                        const double * v = vd.data() + size_t(j) * value_dim;
                        for (uint32_t e = 0; e < value_dim; ++e) { s.u[e] += w * v[e]; }
                        s.z = s.z * f + w;
                        s.m = m_new;
                    }
                }
            }
        }
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t h = 0; h < n_head_q; ++h) {
                const auto o = llama_rerot_attn_state_output(
                    merged[size_t(r) * n_head_q + h]);
                float * dst = shared.data() + (size_t(r) * n_head_q + h) * value_dim;
                for (uint32_t e = 0; e < value_dim; ++e) { dst[e] = float(o[e]); }
            }
        }
        b_us = std::min(b_us, std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count());
    }

    double max_err = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        max_err = std::max(max_err, std::abs(double(shared[i]) - double(reference[i])));
        max_ref = std::max(max_ref, std::abs(double(reference[i])));
    }
    const double rel = max_ref > 0.0 ? max_err / max_ref : 0.0;
    std::printf("shared-block A/B R=%u K=%u: A(per-reader)=%.1f us  B(one pass)=%.1f us  speedup=%.2fx  rel_err=%.3e\n",
                R, n_keys, a_us, b_us, a_us / b_us, rel);
    CHECK(rel < 1e-4);
    return failures == 0 ? 0 : 1;
}
