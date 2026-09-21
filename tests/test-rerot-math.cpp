// test-rerot-math.cpp — FP64/bit-exact oracles for the RERoT compute
// organization research line (2026-09-21 round; src/llama-rerot-math.*).
//
// Every test below compares the core helpers against an INDEPENDENT
// formulation written directly from the mathematical definitions in
// RERoT.md §2.3/§2.4 and the PQ2_0 decode table, never by calling the
// helpers under test. Families:
//
//   [Q3] shared-KV multi-reader block attention vs per-reader full softmax
//   [Q5] GDN base + low-rank vs dense §2.3 recurrence (state/output/dense)
//   [Q6] WY chunk fold vs step-by-step §2.3 recurrence (random chunks,
//        including alpha < 1 decay and heterogeneous beta)
//   [Q7] PQ2_0 bit-plane / LUT dot vs (code - 1) decode, all 4 codes present

#include "llama-rerot-math.h"

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

static double max_abs_diff(const std::vector<double> & a, const std::vector<double> & b) {
    CHECK(a.size() == b.size());
    double result = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

static std::vector<double> random_vector(std::mt19937_64 & rng, size_t n, double scale = 1.0) {
    std::normal_distribution<double> distribution(0.0, scale);
    std::vector<double> result(n);
    for (double & value : result) {
        value = distribution(rng);
    }
    return result;
}

// ---------------------------------------------------------------------------
// [Q3] Shared-KV multi-reader block attention
// ---------------------------------------------------------------------------

// Independent oracle: per reader, plain full softmax over the union of its
// visible blocks (no online max/rescaling at all).
static std::vector<double> q3_full_softmax_reference(
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

static void test_q3_shared_block_attention() {
    constexpr uint32_t n_readers = 5;
    constexpr uint32_t head_dim = 16;
    constexpr uint32_t value_dim = 12;

    std::mt19937_64 rng(0x5eedu);
    const double scale = 1.0 / std::sqrt(double(head_dim));

    // Two physical blocks; readers see different subsets (visibility
    // boundaries), including a reader that sees nothing.
    struct block {
        std::vector<double> keys;
        std::vector<double> values;
        std::vector<uint8_t> visible; // [n_readers]
    };
    std::vector<block> blocks;
    for (uint32_t b = 0; b < 2; ++b) {
        block blk;
        blk.keys = random_vector(rng, 24 * head_dim);
        blk.values = random_vector(rng, 24 * value_dim);
        blk.visible.assign(n_readers, 0);
        for (uint32_t r = 0; r < n_readers; ++r) {
            blk.visible[r] = ((r + b) % 3 == 0) ? 0 : 1;
        }
        blocks.push_back(std::move(blk));
    }
    // Reader 0 sees nothing at all.
    blocks[0].visible[0] = 0;
    blocks[1].visible[0] = 0;

    // Per-reader DDVR phases differ: each reader's query is pre-rotated by
    // the caller; here we emulate distinct phases with distinct query vectors.
    std::vector<double> queries;
    for (uint32_t r = 0; r < n_readers; ++r) {
        const auto q = random_vector(rng, head_dim);
        queries.insert(queries.end(), q.begin(), q.end());
    }

    // Shared-load execution: one pass per block, all visible readers consume.
    std::vector<llama_rerot_attn_state> merged(n_readers);
    for (const auto & blk : blocks) {
        auto states = llama_rerot_shared_block_attention(
            queries, blk.keys, blk.values, n_readers, head_dim, value_dim,
            blk.visible, scale);
        for (uint32_t r = 0; r < n_readers; ++r) {
            if (states[r].empty()) {
                continue; // this block is invisible to the reader
            }
            if (merged[r].empty()) {
                merged[r] = states[r]; // first visible block for this reader
                continue;
            }
            llama_rerot_attn_state_merge(merged[r], states[r]);
        }
    }

    for (uint32_t r = 0; r < n_readers; ++r) {
        // Rebuild the reader's visible union for the oracle.
        std::vector<double> union_keys;
        std::vector<double> union_values;
        for (const auto & blk : blocks) {
            if (!blk.visible[r]) {
                continue;
            }
            union_keys.insert(union_keys.end(), blk.keys.begin(), blk.keys.end());
            union_values.insert(union_values.end(), blk.values.begin(), blk.values.end());
        }
        if (union_keys.empty()) {
            CHECK(merged[r].empty());
            continue;
        }
        const auto expected = q3_full_softmax_reference(
            queries, union_keys, union_values, r, head_dim, value_dim, scale);
        const auto out = llama_rerot_attn_state_output(merged[r]);
        const double err = max_abs_diff(out, expected);
        if (err >= 1e-12) {
            std::fprintf(stderr, "Q3 reader %u error = %.3g\n", r, err);
        }
        CHECK(err < 1e-12);
    }

    // Merge order independence: block 0 then 1 vs 1 then 0 agree.
    {
        std::vector<llama_rerot_attn_state> fwd(n_readers), rev(n_readers);
        for (int order = 0; order < 2; ++order) {
            auto & target = order == 0 ? fwd : rev;
            for (size_t bi = 0; bi < blocks.size(); ++bi) {
                const size_t b = order == 0 ? bi : blocks.size() - 1 - bi;
                auto states = llama_rerot_shared_block_attention(
                    queries, blocks[b].keys, blocks[b].values, n_readers,
                    head_dim, value_dim, blocks[b].visible, scale);
                for (uint32_t r = 0; r < n_readers; ++r) {
                    if (states[r].empty()) {
                        continue; // invisible block for this reader
                    }
                    if (target[r].empty()) {
                        target[r] = states[r]; // first visible block
                        continue;
                    }
                    llama_rerot_attn_state_merge(target[r], states[r]);
                }
            }
        }
        for (uint32_t r = 0; r < n_readers; ++r) {
            if (fwd[r].empty()) {
                CHECK(rev[r].empty());
                continue;
            }
            const auto a = llama_rerot_attn_state_output(fwd[r]);
            const auto b = llama_rerot_attn_state_output(rev[r]);
            CHECK(max_abs_diff(a, b) < 1e-12);
        }
    }

    // Empty-state output must throw.
    bool threw = false;
    try {
        (void) llama_rerot_attn_state_output(llama_rerot_attn_state{});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// [Q5] GDN common base + per-lane low-rank increments
// ---------------------------------------------------------------------------

// Independent oracle: dense §2.3 recurrence on S = a*B + U V^T materialized
// each step (never touches the low-rank helpers).
static void q5_dense_step(
    std::vector<double> & s,          // [d_k * d_v]
    const std::vector<double> & k,
    const std::vector<double> & v,
    double alpha,
    double beta,
    uint32_t d_k,
    uint32_t d_v) {
    std::vector<double> sbar_t_k(d_v, 0.0);
    for (uint32_t e = 0; e < d_v; ++e) {
        double dot = 0.0;
        for (uint32_t j = 0; j < d_k; ++j) {
            dot += s[size_t(j) * d_v + e] * k[j];
        }
        sbar_t_k[e] = alpha * dot;
    }
    for (size_t idx = 0; idx < s.size(); ++idx) {
        s[idx] *= alpha;
    }
    for (uint32_t j = 0; j < d_k; ++j) {
        const double coeff = beta * k[j];
        for (uint32_t e = 0; e < d_v; ++e) {
            s[size_t(j) * d_v + e] += coeff * (v[e] - sbar_t_k[e]);
        }
    }
}

static void test_q5_gdn_lowrank() {
    constexpr uint32_t d_k = 8;
    constexpr uint32_t d_v = 6;
    constexpr uint32_t steps = 12;

    std::mt19937_64 rng(0x9d2fu);
    const auto base = random_vector(rng, size_t(d_k) * d_v, 0.5);
    const auto s0 = random_vector(rng, size_t(d_k) * d_v, 0.5);

    // Factored init: a = 1, r = 0, U/V absorb (S0 - B) exactly? The cleanest
    // exact init is a=0 with U V^T = S0 - 0*B via one rank-1... but S0 is
    // generic full-rank. Instead init with a=1, r=0 and S = B, then verify
    // against the dense oracle started from B.
    llama_rerot_gdn_lowrank_state state;
    state.a = 1.0;
    state.r = 0;

    std::vector<double> dense = base; // S = B

    std::vector<double> ks, vs, alphas, betas;
    for (uint32_t step = 0; step < steps; ++step) {
        const auto k = random_vector(rng, d_k, 0.7);
        const auto v = random_vector(rng, d_v, 0.7);
        std::uniform_real_distribution<double> alpha_dist(0.7, 1.0);
        std::uniform_real_distribution<double> beta_dist(0.2, 1.2);
        const double alpha = alpha_dist(rng);
        const double beta = beta_dist(rng);

        // Shared base projection: B^T k for this lane (one call, many lanes
        // in production — here one lane, still through the shared API).
        const auto proj = llama_rerot_gdn_base_project(base, k, d_k, d_v);
        CHECK(proj.size() == d_v);

        state = llama_rerot_gdn_lowrank_step(state, k, v, alpha, beta, proj);
        q5_dense_step(dense, k, v, alpha, beta, d_k, d_v);

        // Rank grows by exactly one per step.
        CHECK(state.r == step + 1);

        ks.insert(ks.end(), k.begin(), k.end());
        vs.insert(vs.end(), v.begin(), v.end());
        alphas.push_back(alpha);
        betas.push_back(beta);
    }

    // Dense equivalence after all steps.
    const auto dense_from_factored = llama_rerot_gdn_lowrank_dense(state, base, d_k, d_v);
    const double err = max_abs_diff(dense_from_factored, dense);
    if (err >= 1e-10) {
        std::fprintf(stderr, "Q5 dense error = %.3g\n", err);
    }
    CHECK(err < 1e-10);

    // Output equivalence at a random query.
    const auto q = random_vector(rng, d_k, 0.7);
    const auto base_proj_q = llama_rerot_gdn_base_project(base, q, d_k, d_v);
    const auto out_factored = llama_rerot_gdn_lowrank_output(state, q, base_proj_q);
    std::vector<double> out_dense(d_v, 0.0);
    for (uint32_t j = 0; j < d_k; ++j) {
        for (uint32_t e = 0; e < d_v; ++e) {
            out_dense[e] += dense[size_t(j) * d_v + e] * q[j];
        }
    }
    const double out_err = max_abs_diff(out_factored, out_dense);
    if (out_err >= 1e-10) {
        std::fprintf(stderr, "Q5 output error = %.3g\n", out_err);
    }
    CHECK(out_err < 1e-10);

    // Multi-lane shared-base projection: one call over many vectors equals
    // per-vector calls (the actual sharing contract).
    {
        std::vector<double> xs;
        for (uint32_t n = 0; n < 6; ++n) {
            const auto x = random_vector(rng, d_k, 0.9);
            xs.insert(xs.end(), x.begin(), x.end());
        }
        const auto batched = llama_rerot_gdn_base_project(base, xs, d_k, d_v);
        for (uint32_t n = 0; n < 6; ++n) {
            const auto single = llama_rerot_gdn_base_project(
                base,
                std::vector<double>(xs.begin() + size_t(n) * d_k,
                                    xs.begin() + size_t(n + 1) * d_k),
                d_k, d_v);
            CHECK(max_abs_diff(
                std::vector<double>(batched.begin() + size_t(n) * d_v,
                                    batched.begin() + size_t(n + 1) * d_v),
                single) == 0.0);
        }
    }

    // State-size mismatch must throw.
    bool threw = false;
    try {
        llama_rerot_gdn_lowrank_state bad;
        bad.r = 3;
        bad.u.assign(size_t(d_k) * 2, 0.0); // wrong rank
        const auto proj = llama_rerot_gdn_base_project(
            base, std::vector<double>(d_k, 0.0), d_k, d_v);
        (void) llama_rerot_gdn_lowrank_step(
            bad, std::vector<double>(d_k, 0.0), std::vector<double>(d_v, 0.0),
            0.9, 0.5, proj);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// [Q6] Known-token chunk recurrence (WY fold)
// ---------------------------------------------------------------------------

static void test_q6_chunk_fold() {
    constexpr uint32_t d_k = 6;
    constexpr uint32_t d_v = 5;

    std::mt19937_64 rng(0xc0ffeeu);
    for (int trial = 0; trial < 24; ++trial) {
        // Chunk sizes 1..8; alphas < 1 exercise decay, betas heterogeneous.
        const uint32_t t = 1 + (trial % 8);
        const auto s0 = random_vector(rng, size_t(d_k) * d_v, 0.6);
        std::vector<double> ks, vs, alphas, betas;
        for (uint32_t i = 0; i < t; ++i) {
            const auto k = random_vector(rng, d_k, 0.8);
            const auto v = random_vector(rng, d_v, 0.8);
            std::uniform_real_distribution<double> alpha_dist(0.6, 1.0);
            std::uniform_real_distribution<double> beta_dist(0.0, 1.5);
            ks.insert(ks.end(), k.begin(), k.end());
            vs.insert(vs.end(), v.begin(), v.end());
            alphas.push_back(alpha_dist(rng));
            betas.push_back(beta_dist(rng));
        }

        const auto chunk = llama_rerot_gdn_chunk_fold(d_k, d_v, ks, vs, alphas, betas);
        const auto folded = llama_rerot_gdn_chunk_apply(chunk, s0, d_k, d_v);
        const auto stepped = llama_rerot_gdn_steps_reference(s0, d_k, d_v, ks, vs, alphas, betas);

        const double err = max_abs_diff(folded, stepped);
        if (err >= 1e-10) {
            std::fprintf(stderr, "Q6 trial %d (T=%u) error = %.3g\n", trial, t, err);
        }
        CHECK(err < 1e-10);
    }

    // beta = 0 (pure decay, no update) must also fold exactly.
    {
        constexpr uint32_t t = 4;
        std::vector<double> ks, vs, alphas, betas;
        std::mt19937_64 rng2(7u);
        const auto s0 = random_vector(rng2, size_t(d_k) * d_v, 0.5);
        for (uint32_t i = 0; i < t; ++i) {
            const auto k = random_vector(rng2, d_k, 0.8);
            const auto v = random_vector(rng2, d_v, 0.8);
            ks.insert(ks.end(), k.begin(), k.end());
            vs.insert(vs.end(), v.begin(), v.end());
            alphas.push_back(0.8);
            betas.push_back(0.0);
        }
        const auto chunk = llama_rerot_gdn_chunk_fold(d_k, d_v, ks, vs, alphas, betas);
        const auto folded = llama_rerot_gdn_chunk_apply(chunk, s0, d_k, d_v);
        const auto stepped = llama_rerot_gdn_steps_reference(s0, d_k, d_v, ks, vs, alphas, betas);
        CHECK(max_abs_diff(folded, stepped) < 1e-12);

        // beta = 0 means M must be exactly G * I and Y exactly 0.
        const double g = 0.8 * 0.8 * 0.8 * 0.8;
        for (uint32_t j = 0; j < d_k; ++j) {
            for (uint32_t i = 0; i < d_k; ++i) {
                const double expected = (i == j) ? g : 0.0;
                CHECK(std::abs(chunk.m[size_t(j) * d_k + i] - expected) < 1e-12);
            }
        }
        for (double yv : chunk.y) {
            CHECK(yv == 0.0);
        }
    }

    // Empty chunk must throw.
    bool threw = false;
    try {
        (void) llama_rerot_gdn_chunk_fold(
            d_k, d_v, {}, {}, {}, {});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// [Q7] PQ2_0 bit-plane subset-sum inner product
// ---------------------------------------------------------------------------

static void test_q7_pq2_bitplane() {
    constexpr uint32_t n = 128;

    std::mt19937_64 rng(0x2b1du);
    std::uniform_int_distribution<int> code_dist(0, 3);
    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);

    // Random codes with ALL FOUR values present (the {3 -> +2} code is the
    // one a "ternary" assumption would silently drop).
    std::vector<uint8_t> qs(n / 4);
    bool has[4] = { false, false, false, false };
    for (uint32_t j = 0; j < n; ++j) {
        const int code = code_dist(rng);
        has[code] = true;
        qs[j / 4] |= uint8_t(code) << ((j % 4) * 2);
    }
    CHECK(has[0] && has[1] && has[2] && has[3]);

    const auto x = random_vector(rng, n, 1.5);
    (void) x_dist;

    // Independent oracle: decode (code - 1) and dot, exactly as
    // ggml's dequantize_row_pq2_0 defines the values.
    double expected = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
        const uint8_t code = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        expected += (double(code) - 1.0) * x[j];
    }

    const double bitplane = llama_rerot_pq2_bitplane_dot(qs.data(), x.data(), n);
    const double lut = llama_rerot_pq2_lut_dot(qs.data(), x.data(), n);

    CHECK(std::abs(bitplane - expected) < 1e-9);
    CHECK(std::abs(lut - expected) < 1e-9);
    // The two formulations of the same identity agree tightly.
    CHECK(std::abs(bitplane - lut) < 1e-9);

    // Bit-exactness against ggml's integer path: integral activations make
    // the subset sums exactly representable, so double accumulation must
    // equal the int accumulation ggml performs.
    {
        std::vector<double> xi(n);
        std::uniform_int_distribution<int> int_dist(-8, 8);
        for (uint32_t j = 0; j < n; ++j) {
            xi[j] = double(int_dist(rng));
        }
        int64_t int_oracle = 0;
        for (uint32_t j = 0; j < n; ++j) {
            const uint8_t code = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            int_oracle += int64_t(code) - 1;
            // multiply below
        }
        // redo properly: (code - 1) * x_j
        int_oracle = 0;
        for (uint32_t j = 0; j < n; ++j) {
            const uint8_t code = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            int_oracle += (int64_t(code) - 1) * int64_t(xi[j]);
        }
        CHECK(llama_rerot_pq2_bitplane_dot(qs.data(), xi.data(), n) == double(int_oracle));
        CHECK(llama_rerot_pq2_lut_dot(qs.data(), xi.data(), n) == double(int_oracle));
    }

    // All-codes-same sanity: code 3 everywhere -> (+2) * sum(x).
    {
        std::vector<uint8_t> q3(n / 4, 0xff);
        double sum = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            sum += x[j];
        }
        CHECK(std::abs(llama_rerot_pq2_bitplane_dot(q3.data(), x.data(), n) - 2.0 * sum) < 1e-9);
        CHECK(std::abs(llama_rerot_pq2_lut_dot(q3.data(), x.data(), n) - 2.0 * sum) < 1e-9);
    }

    // Invalid n must throw.
    bool threw = false;
    try {
        (void) llama_rerot_pq2_bitplane_dot(qs.data(), x.data(), 3);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    std::fprintf(stderr, "=== RERoT Math Reference Tests ===\n");
    test_q3_shared_block_attention();
    test_q5_gdn_lowrank();
    test_q6_chunk_fold();
    test_q7_pq2_bitplane();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
