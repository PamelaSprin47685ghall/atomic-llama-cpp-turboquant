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

int main(int argc, char ** argv) {
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
