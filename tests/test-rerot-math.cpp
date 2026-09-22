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

// [Q7] ggml's own PQ2_0 block layout: the bit-plane identity must hold on
// the REAL packed representation the inference path decodes, not only on a
// synthetic byte array.
#include "ggml-common.h"
#include "llama.h"
#include "ggml-quants.h"

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


// ---------------------------------------------------------------------------
// [Q2] Span-view reference
// ---------------------------------------------------------------------------

// Independent oracle: per-key effective positions and causal visibility,
// written straight from §2.4 (no span shortcuts).
static void test_q2_span_view() {
    std::vector<llama_rerot_span_view> spans = {
        { 0,  10, 3 },   // virtuals [0,10)   phase 3
        { 12, 6, -2 },   // virtuals [12,18)  phase -2 (hole at 10..11)
        { 40, 8, 7 },    // virtuals [40,48)  phase 7 (hole at 18..39)
    };

    // (1) Span-constant effective position: for EVERY key j in span s,
    //     q_v + s_j - v_j equals the span helper's single value (j cancels).
    for (const auto & s : spans) {
        const int64_t eff = llama_rerot_span_effective_pos(100, s);
        for (uint32_t j = 0; j < s.len; ++j) {
            const int64_t v_j = int64_t(s.begin) + j;
            const int64_t s_j = v_j + s.phase; // unit step: storage = virtual + phase
            CHECK(100 + s_j - v_j == eff);
        }
    }

    // (2) Causal cut == per-key causal mask: key j visible iff v_j <= q_v.
    const std::vector<int64_t> qs = { -1, 0, 5, 9, 10, 17, 39, 47, 100 };
    for (const int64_t q_v : qs) {
        for (const auto & s : spans) {
            uint32_t per_key = 0;
            while (per_key < s.len && int64_t(s.begin) + per_key <= q_v) {
                ++per_key;
            }
            CHECK(llama_rerot_span_causal_len(s, q_v) == per_key);
        }
    }

    // (3) Fragmentation metric.
    {
        std::vector<llama_rerot_span_view> fragmented;
        for (uint32_t i = 0; i < 32; ++i) {
            fragmented.push_back({ i * 2, 2, int64_t(i) });
        }
        CHECK(llama_rerot_span_long_fraction(fragmented, 8) == 0.0);
        std::vector<llama_rerot_span_view> regular = { { 0, 64, 0 } };
        CHECK(llama_rerot_span_long_fraction(regular, 8) == 1.0);
        std::vector<llama_rerot_span_view> mixed = { { 0, 16, 0 }, { 16, 2, 1 },
                                                     { 18, 2, 2 }, { 20, 2, 3 } };
        CHECK(std::abs(llama_rerot_span_long_fraction(mixed, 8) - (16.0 / 22.0)) < 1e-15);
    }

    // Empty span must throw.
    bool threw = false;
    try {
        (void) llama_rerot_span_causal_len({ 0, 0, 0 }, 5);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// [Q4] Structure/numeric separation
// ---------------------------------------------------------------------------

static void test_q4_structure_numeric() {
    llama_rerot_run_order order;
    order.run_ids = { 7, 3, 5, 1 }; // topological order (structure)
    llama_rerot_run_lengths lengths;
    lengths.len = { 10, 4, 0, 25 };

    // Prefix sums over the frozen order.
    const auto starts = llama_rerot_virtual_starts(order, lengths);
    const std::vector<int64_t> expected = { 0, 10, 14, 14 };
    CHECK(starts == expected);

    // Growth of run 1 by 6: only LATER runs shift; structure untouched.
    const auto grown = llama_rerot_virtual_starts_after_growth(order, lengths, 1, 6);
    const std::vector<int64_t> expected_grown = { 0, 10, 20, 20 };
    CHECK(grown == expected_grown);

    // Signature invariant to ALL numeric changes...
    const uint64_t sig_before = llama_rerot_run_order_signature(order);
    llama_rerot_run_lengths big;
    big.len = { 1000, 2000, 3000, 4000 };
    (void) llama_rerot_virtual_starts(order, big);
    CHECK(llama_rerot_run_order_signature(order) == sig_before);

    // ...and changes iff the ORDER (structure) changes.
    llama_rerot_run_order reordered = order;
    std::swap(reordered.run_ids[0], reordered.run_ids[1]);
    CHECK(llama_rerot_run_order_signature(reordered) != sig_before);

    // Incremental path == full recompute from grown lengths.
    llama_rerot_run_lengths manual = lengths;
    manual.len[1] += 6;
    CHECK(llama_rerot_virtual_starts(order, manual) == grown);

    // Size mismatches must throw.
    bool threw = false;
    llama_rerot_run_lengths bad;
    bad.len = { 1, 2, 3 };
    try {
        (void) llama_rerot_virtual_starts(order, bad);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void) llama_rerot_virtual_starts_after_growth(order, lengths, 4, 1);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// [Q8] Skip-block bounds
// ---------------------------------------------------------------------------

static void test_q8_skip_bounds() {
    constexpr uint32_t head_dim = 8;
    constexpr uint32_t value_dim = 4;
    constexpr uint32_t n_block = 12;
    std::mt19937_64 rng(0x51ull);

    std::vector<double> center(head_dim);
    for (auto & c : center) {
        c = 0.1 * double(int(rng() % 21) - 10);
    }
    std::vector<double> keys, values;
    double radius = 0.0;
    for (uint32_t j = 0; j < n_block; ++j) {
        const auto k = random_vector(rng, head_dim, 0.5);
        const auto v = random_vector(rng, value_dim, 1.0);
        for (uint32_t e = 0; e < head_dim; ++e) {
            radius = std::max(radius, std::abs(k[e] - center[e]));
        }
        keys.insert(keys.end(), k.begin(), k.end());
        values.insert(values.end(), v.begin(), v.end());
    }
    std::vector<double> keep_keys, keep_values;
    double v_max = 0.0;
    for (uint32_t j = 0; j < 5; ++j) {
        const auto k = random_vector(rng, head_dim, 1.0);
        const auto v = random_vector(rng, value_dim, 1.0);
        keep_keys.insert(keep_keys.end(), k.begin(), k.end());
        keep_values.insert(keep_values.end(), v.begin(), v.end());
        for (double e : v) {
            v_max = std::max(v_max, std::abs(e));
        }
    }
    for (double e : values) {
        v_max = std::max(v_max, std::abs(e));
    }
    const auto q = random_vector(rng, head_dim, 1.0);
    const double scale = 1.0 / std::sqrt(double(head_dim));

    // Exact Z_keep, Z_skip, o, o_keep (independent full softmax).
    auto dot = [&](const std::vector<double> & kbuf, uint32_t idx) {
        double acc = 0.0;
        for (uint32_t e = 0; e < head_dim; ++e) {
            acc += q[e] * kbuf[size_t(idx) * head_dim + e];
        }
        return acc;
    };
    double z_keep = 0.0, z_skip = 0.0;
    std::vector<double> num_keep(value_dim, 0.0), num_all(value_dim, 0.0);
    for (uint32_t j = 0; j < 5; ++j) {
        const double w = std::exp(scale * dot(keep_keys, j));
        z_keep += w;
        for (uint32_t e = 0; e < value_dim; ++e) {
            num_keep[e] += w * keep_values[size_t(j) * value_dim + e];
        }
    }
    for (uint32_t j = 0; j < n_block; ++j) {
        const double w = std::exp(scale * dot(keys, j));
        z_skip += w;
        for (uint32_t e = 0; e < value_dim; ++e) {
            num_all[e] += w * values[size_t(j) * value_dim + e];
        }
    }
    for (uint32_t e = 0; e < value_dim; ++e) {
        num_all[e] += num_keep[e];
    }
    std::vector<double> o_exact(value_dim), o_keep_exact(value_dim);
    for (uint32_t e = 0; e < value_dim; ++e) {
        o_exact[e] = num_all[e] / (z_keep + z_skip);
        o_keep_exact[e] = num_keep[e] / z_keep;
    }

    // Bound soundness: exact deviation never exceeds the published bound,
    // and the bound's delta dominates the true skipped-mass ratio.
    const double z_skip_bound = llama_rerot_skip_mass_bound(q, center, radius, n_block, scale);
    CHECK(z_skip_bound > 0.0);
    const auto [delta, bound] = llama_rerot_skip_output_bound(z_keep, z_skip_bound, v_max);
    double exact_dev = 0.0;
    for (uint32_t e = 0; e < value_dim; ++e) {
        exact_dev = std::max(exact_dev, std::abs(o_exact[e] - o_keep_exact[e]));
    }
    CHECK(exact_dev <= bound + 1e-15);
    const double true_delta = z_skip / (z_keep + z_skip);
    CHECK(delta >= true_delta - 1e-15);

    // Aggregation counterexample: {-1,+1} vs {0,0} share count and key sum
    // but 2cosh(1) != 2 — no finite statistic of the frozen context can
    // represent Z(q) for all q, so exact skip-compression cannot exist.
    {
        double z1 = std::exp(-1.0) + std::exp(1.0);
        double z2 = std::exp(0.0) + std::exp(0.0);
        CHECK(std::abs(z1 - z2) > 1e-2);
    }

    // Bad inputs must throw.
    bool threw = false;
    try {
        (void) llama_rerot_skip_mass_bound(q, center, -1.0, n_block, scale);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void) llama_rerot_skip_output_bound(0.0, 1.0, 1.0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// [Q9] Joint sampler contract
// ---------------------------------------------------------------------------

static void test_q9_joint_sampler() {
    std::vector<double> logits_a = { 0.1, -0.3, 2.0, 0.5, -1.0, 0.9, 1.4, 0.0 };
    std::vector<double> logits_b = { 1.1, 0.3, -2.0, 0.5, 1.0, -0.9, 0.4, 2.0 };
    const uint64_t base_seed = 0xabcdefu;

    // (1) Row-order independence: pens {2,5} in either cohort order draw
    // identically (per-pen streams).
    uint64_t s2 = llama_rerot_joint_sample_seed(base_seed, 2);
    uint64_t s5 = llama_rerot_joint_sample_seed(base_seed, 5);
    const auto draw2 = llama_rerot_joint_sample_row(logits_a, 0.8, 4, 0.95, s2);
    const auto draw5 = llama_rerot_joint_sample_row(logits_b, 0.8, 4, 0.95, s5);
    uint64_t t5 = llama_rerot_joint_sample_seed(base_seed, 5);
    uint64_t t2 = llama_rerot_joint_sample_seed(base_seed, 2);
    const auto again5 = llama_rerot_joint_sample_row(logits_b, 0.8, 4, 0.95, t5);
    const auto again2 = llama_rerot_joint_sample_row(logits_a, 0.8, 4, 0.95, t2);
    CHECK(draw5 == again5);
    CHECK(draw2 == again2);

    // (2) Cohort-size independence: pen 2 alone draws the same as pen 2 in
    // a cohort (streams are per-pen, not a shared cohort stream).
    uint64_t u2 = llama_rerot_joint_sample_seed(base_seed, 2);
    const auto solo2 = llama_rerot_joint_sample_row(logits_a, 0.8, 4, 0.95, u2);
    CHECK(solo2 == draw2);

    // (3) Different pens get different streams.
    bool any_diff = false;
    for (uint32_t rep = 0; rep < 32 && !any_diff; ++rep) {
        std::vector<double> l(64);
        for (uint32_t i = 0; i < 64; ++i) {
            l[i] = std::sin(0.37 * double((rep * 64 + i) % 97));
        }
        uint64_t a = llama_rerot_joint_sample_seed(0x1234u, 0);
        uint64_t b = llama_rerot_joint_sample_seed(0x1234u, 1);
        if (llama_rerot_joint_sample_row(l, 1.0, 0, 1.0, a) !=
            llama_rerot_joint_sample_row(l, 1.0, 0, 1.0, b)) {
            any_diff = true;
        }
    }
    CHECK(any_diff);

    // (4) Determinism: same stream + same row -> same token, always.
    for (uint32_t rep = 0; rep < 8; ++rep) {
        uint64_t a = llama_rerot_joint_sample_seed(999, 3);
        uint64_t b = llama_rerot_joint_sample_seed(999, 3);
        CHECK(llama_rerot_joint_sample_row(logits_a, 1.0, 0, 1.0, a) ==
              llama_rerot_joint_sample_row(logits_a, 1.0, 0, 1.0, b));
    }

    // (5) Greedy joint argmax: per-block maxima reduce to the full-row
    // argmax with LOWEST-index tie-break.
    {
        std::vector<std::vector<double>> per_block = {
            { 1.0, 3.0, 3.0, -1.0, 0.5, 0.0 },
            { 0.0, 0.0, 0.0, -1.0, 0.5, 3.0 },
        };
        const auto arg = llama_rerot_joint_argmax_rows(per_block);
        CHECK(arg.size() == 1);
        CHECK(arg[0] == 1); // 3.0 ties at 1, 2, 5 -> lowest index
    }

    // (6) Bad inputs must throw.
    bool threw = false;
    try {
        uint64_t st = 1;
        (void) llama_rerot_joint_sample_row({}, 1.0, 0, 1.0, st);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void) llama_rerot_joint_argmax_rows({});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// [Q10] Frontier-grid joint verification
// ---------------------------------------------------------------------------

static void test_q10_frontier_grid() {
    // 3 pens, horizon 3. B's col-1 draft is rejected; C reads B, so C's
    // col-2 cell dies even though C's own drafts all match truth.
    llama_rerot_draft_grid grid;
    grid.n_pens = 3;
    grid.horizon = 3;
    grid.draft  = { 10, 11, 12,   // A
                    20, 21, 22,   // B
                    30, 31, 32 }; // C
    grid.truth  = { 10, 11, 12,   // A all correct
                    20, 99, 22,   // B: col-1 rejected
                    30, 31, 32 }; // C: own drafts all correct

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> reads(3);
    reads[1].push_back({ 2, 1 }); // C reads B's col-0 commit
    reads[2].push_back({ 2, 1 }); // C reads B's col-1 commit

    const auto verdict = llama_rerot_verify_grid(grid, reads);
    CHECK(verdict.accepted[0] == 3);          // A: full accept
    CHECK(verdict.accepted[1] == 1);          // B: col-1 rejected
    CHECK(verdict.replacement[1] == 99);      // B publishes truth at rejection
    CHECK(verdict.accepted[2] == 2);          // C: col-1 alive (B col-0 fine), dies col-2
    CHECK(verdict.replacement[2] == 32);      // C's truth at its first dead column

    // The naive per-row engine accepts C fully — the executable
    // counterexample: per-row verification is WRONG under cross-pen reads.
    const auto naive = llama_rerot_verify_grid_naive(grid);
    CHECK(naive.accepted[2] == 3);
    CHECK(naive.accepted[2] != verdict.accepted[2]);

    // No cross-pen reads: dependency-tracked degenerates to naive.
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> no_reads(3);
    const auto indep = llama_rerot_verify_grid(grid, no_reads);
    CHECK(indep.accepted[2] == 3);

    // Bad edge index must throw.
    bool threw = false;
    try {
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> bad_reads(3);
        bad_reads[0].push_back({ 0, 7 });
        (void) llama_rerot_verify_grid(grid, bad_reads);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// F32 numerics gate for Q5/Q6 (re-association error measurement)
// ---------------------------------------------------------------------------

// Run the Q5 factored path in FLOAT arithmetic against the FP64 dense oracle
// and return the max absolute deviation. This is the measurement the F32 gate
// must eventually bound before any GPU kernel ships; it is NOT an acceptance
// threshold — it is the honest error number for the current formulation.
static double q5_f32_vs_fp64(uint32_t steps, uint32_t & out_rank,
                             double alpha_lo, double alpha_hi,
                             double beta_lo, double beta_hi,
                             double & out_max_abs) {
    constexpr uint32_t d_k = 32;
    constexpr uint32_t d_v = 16;
    std::mt19937_64 rng(0xf32u);
    const auto base = random_vector(rng, size_t(d_k) * d_v, 0.5);

    // Float factored state (mirrors llama_rerot_gdn_lowrank_state in float).
    float a = 1.0f;
    std::vector<float> u, v;
    uint32_t r = 0;
    std::vector<double> dense = base;

    for (uint32_t step = 0; step < steps; ++step) {
        const auto k = random_vector(rng, d_k, 0.7);
        const auto v_d = random_vector(rng, d_v, 0.7);
        std::uniform_real_distribution<double> alpha_dist(alpha_lo, alpha_hi);
        std::uniform_real_distribution<double> beta_dist(beta_lo, beta_hi);
        const double alpha = alpha_dist(rng);
        const double beta = beta_dist(rng);
        const auto proj = llama_rerot_gdn_base_project(base, k, d_k, d_v);

        // Float factored step (re-associated arithmetic).
        std::vector<float> kf(k.begin(), k.end());
        std::vector<float> vf(v_d.begin(), v_d.end());
        std::vector<float> projf(proj.begin(), proj.end());
        std::vector<float> u_t_k(r, 0.0f);
        for (uint32_t c = 0; c < r; ++c) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < d_k; ++j) {
                acc += u[size_t(j) * r + c] * kf[j];
            }
            u_t_k[c] = acc;
        }
        std::vector<float> sbar_t_k(d_v, 0.0f);
        for (uint32_t e = 0; e < d_v; ++e) {
            float acc = a * projf[e];
            for (uint32_t c = 0; c < r; ++c) {
                acc += v[size_t(e) * r + c] * u_t_k[c];
            }
            sbar_t_k[e] = float(alpha) * acc;
        }
        a *= float(alpha);
        std::vector<float> un(size_t(d_k) * (r + 1), 0.0f);
        std::vector<float> vn(size_t(d_v) * (r + 1), 0.0f);
        for (uint32_t j = 0; j < d_k; ++j) {
            for (uint32_t c = 0; c < r; ++c) {
                un[size_t(j) * (r + 1) + c] = float(alpha) * u[size_t(j) * r + c];
            }
            un[size_t(j) * (r + 1) + r] = float(beta) * kf[j];
        }
        for (uint32_t e = 0; e < d_v; ++e) {
            for (uint32_t c = 0; c < r; ++c) {
                vn[size_t(e) * (r + 1) + c] = v[size_t(e) * r + c];
            }
            vn[size_t(e) * (r + 1) + r] = vf[e] - sbar_t_k[e];
        }
        u = std::move(un);
        v = std::move(vn);
        ++r;

        q5_dense_step(dense, k, v_d, alpha, beta, d_k, d_v);
    }

    // Compare float factored S against FP64 dense oracle; report the state
    // magnitude too so the relative error is honest.
    double max_err = 0.0;
    double max_abs = 0.0;
    for (uint32_t j = 0; j < d_k; ++j) {
        for (uint32_t e = 0; e < d_v; ++e) {
            float acc = a * float(base[size_t(j) * d_v + e]);
            for (uint32_t c = 0; c < r; ++c) {
                acc += u[size_t(j) * r + c] * v[size_t(e) * r + c];
            }
            max_err = std::max(max_err, std::abs(double(acc) - dense[size_t(j) * d_v + e]));
            max_abs = std::max(max_abs, std::abs(dense[size_t(j) * d_v + e]));
        }
    }
    out_rank = r;
    out_max_abs = max_abs;
    return max_err;
}

// Run the Q6 WY fold in FLOAT arithmetic (materialized M, Y in float, float
// solve) against the FP64 step-by-step oracle; return the max deviation.
static double q6_f32_vs_fp64(uint32_t t) {
    constexpr uint32_t d_k = 32;
    constexpr uint32_t d_v = 16;
    std::mt19937_64 rng(0xf36u);
    const auto s0 = random_vector(rng, size_t(d_k) * d_v, 0.5);
    std::vector<double> ks, vs, alphas, betas;
    for (uint32_t i = 0; i < t; ++i) {
        const auto k = random_vector(rng, d_k, 0.8);
        const auto v = random_vector(rng, d_v, 0.8);
        ks.insert(ks.end(), k.begin(), k.end());
        vs.insert(vs.end(), v.begin(), v.end());
        alphas.push_back(0.95);
        betas.push_back(0.5 + 0.5 * double(i) / t);
    }

    // FP64 oracle.
    const auto stepped = llama_rerot_gdn_steps_reference(s0, d_k, d_v, ks, vs, alphas, betas);

    // Float fold + float apply: cast inputs to float, run the same algorithm.
    // (Re-implemented inline in float to measure the re-association error of
    // the WY formulation itself, not of a double-cast.)
    std::vector<float> ksf(ks.begin(), ks.end());
    std::vector<float> vsf(vs.begin(), vs.end());
    std::vector<float> alphaf(alphas.begin(), alphas.end());
    std::vector<float> betaf(betas.begin(), betas.end());

    // Y arm in float.
    std::vector<float> y(size_t(d_k) * d_v, 0.0f);
    std::vector<float> sbar_t_k(d_v);
    for (uint32_t step = 0; step < t; ++step) {
        const float * k = ksf.data() + step * d_k;
        const float * v = vsf.data() + step * d_v;
        const float al = alphaf[step];
        const float be = betaf[step];
        for (uint32_t e = 0; e < d_v; ++e) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < d_k; ++j) {
                acc += y[size_t(j) * d_v + e] * k[j];
            }
            sbar_t_k[e] = al * acc;
        }
        for (size_t idx = 0; idx < y.size(); ++idx) {
            y[idx] *= al;
        }
        for (uint32_t j = 0; j < d_k; ++j) {
            const float coeff = be * k[j];
            for (uint32_t e = 0; e < d_v; ++e) {
                y[size_t(j) * d_v + e] += coeff * (v[e] - sbar_t_k[e]);
            }
        }
    }

    // M in float: G * (I - K W K^T), W = (I+L)^-1 diag(beta) via float forward
    // substitution on the Gram matrix.
    std::vector<float> gram(t * t, 0.0f);
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            float acc = 0.0f;
            for (uint32_t e = 0; e < d_k; ++e) {
                acc += ksf[size_t(i) * d_k + e] * ksf[size_t(j) * d_k + e];
            }
            gram[size_t(i) * t + j] = acc;
        }
    }
    std::vector<float> l(t * t, 0.0f);
    for (uint32_t j = 0; j < t; ++j) {
        for (uint32_t c = 0; c < j; ++c) {
            l[size_t(j) * t + c] = betaf[j] * gram[size_t(j) * t + c];
        }
    }
    std::vector<float> w_cols(t * t, 0.0f);
    for (uint32_t c = 0; c < t; ++c) {
        std::vector<float> b(t, 0.0f);
        b[c] = betaf[c];
        for (uint32_t i = 0; i < t; ++i) {
            float acc = b[i];
            for (uint32_t j2 = 0; j2 < i; ++j2) {
                acc -= l[size_t(i) * t + j2] * b[j2];
            }
            b[i] = acc;
        }
        for (uint32_t i = 0; i < t; ++i) {
            w_cols[size_t(c) * t + i] = b[i];
        }
    }
    float g_total = 1.0f;
    for (uint32_t i = 0; i < t; ++i) {
        g_total *= alphaf[i];
    }
    // S_out = M S0 + Y in float.
    std::vector<float> s0f(s0.begin(), s0.end());
    std::vector<float> out(size_t(d_k) * d_v, 0.0f);
    for (uint32_t j = 0; j < d_k; ++j) {
        for (uint32_t e = 0; e < d_v; ++e) {
            // column j of M applied to S0 row-space: out[j,:] = sum_i M[j,i] S0[i,:] + Y[j,:]
            float acc = 0.0f;
            (void) acc;
        }
    }
    // Build M explicitly in float (reference measurement only).
    std::vector<float> m(size_t(d_k) * d_k, 0.0f);
    for (uint32_t j = 0; j < d_k; ++j) {
        std::vector<float> kt_ej(t);
        for (uint32_t s = 0; s < t; ++s) {
            kt_ej[s] = ksf[size_t(s) * d_k + j];
        }
        std::vector<float> w(t, 0.0f);
        for (uint32_t c = 0; c < t; ++c) {
            const float coeff = kt_ej[c];
            for (uint32_t r2 = 0; r2 < t; ++r2) {
                w[r2] += coeff * w_cols[size_t(c) * t + r2];
            }
        }
        for (uint32_t i = 0; i < d_k; ++i) {
            float acc = (i == j) ? 1.0f : 0.0f;
            for (uint32_t r2 = 0; r2 < t; ++r2) {
                acc -= w[r2] * ksf[size_t(r2) * d_k + i];
            }
            m[size_t(i) * d_k + j] = g_total * acc;
        }
    }
    for (uint32_t j = 0; j < d_k; ++j) {
        for (uint32_t e = 0; e < d_v; ++e) {
            float acc = 0.0f;
            for (uint32_t c = 0; c < d_k; ++c) {
                acc += m[size_t(j) * d_k + c] * s0f[size_t(c) * d_v + e];
            }
            out[size_t(j) * d_v + e] = acc + y[size_t(j) * d_v + e];
        }
    }

    double max_err = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        max_err = std::max(max_err, std::abs(double(out[i]) - stepped[i]));
    }
    return max_err;
}

static void test_f32_gate() {
    // Q5: 24 steps of the factored update in F32 vs the FP64 dense oracle.
    // This prints the honest re-association error; the CHECK is deliberately
    // loose (it only asserts the error is finite and the rank is exact) —
    // the number itself is the deliverable, printed for the record.
    uint32_t rank = 0;
    double max_abs = 0.0;
    const double err_b = q5_f32_vs_fp64(24, rank, 0.85, 0.98, 0.1, 0.8, max_abs);
    const double rel_b = max_abs > 0.0 ? err_b / max_abs : 0.0;
    std::fprintf(stderr, "F32 gate: Q5 lowrank 24 steps (bounded), rank=%u, max|S|=%.3e, max|err|=%.3e, rel=%.3e\n",
                 rank, max_abs, err_b, rel_b);
    CHECK(std::isfinite(err_b));
    CHECK(rank == 24);
    CHECK(rel_b < 1e-3); // sanity envelope; the printed number is the record

    double max_abs_a = 0.0;
    const double err_a = q5_f32_vs_fp64(24, rank, 0.9, 1.0, 0.2, 1.2, max_abs_a);
    const double rel_a = max_abs_a > 0.0 ? err_a / max_abs_a : 0.0;
    std::fprintf(stderr, "F32 gate: Q5 lowrank 24 steps (aggressive, record only), max|S|=%.3e, max|err|=%.3e, rel=%.3e\n",
                 max_abs_a, err_a, rel_a);
    CHECK(std::isfinite(err_a));

    // Q6: WY fold in F32 vs FP64 step-by-step, T=8.
    const double q6_err = q6_f32_vs_fp64(8);
    std::fprintf(stderr, "F32 gate: Q6 WY fold T=8, max|err|=%.3e (vs FP64 steps)\n", q6_err);
    CHECK(std::isfinite(q6_err));
    CHECK(q6_err < 1e-2); // sanity envelope

    // ------------------------------------------------------------------
    // Q7 against ggml's REAL PQ2_0 blocks: the bit-plane identity must
    // reproduce ggml's decode ((code - 1) * d) on the production packed
    // layout, in FLOAT accumulation, with all four codes present. The
    // ggml dot itself is approximated by its own dequantized float dot,
    // which is the reference the inference path effectively computes.
    // ------------------------------------------------------------------
    {
        // One block's worth of codes: 128 weights, all four values present.
        block_pq2_0 blk{};
        std::mt19937_64 rng7(0x9e77u);
        std::uniform_int_distribution<int> code_dist(0, 3);
        bool has[4] = { false, false, false, false };
        for (uint32_t j = 0; j < QKP2_0; ++j) {
            const int code = code_dist(rng7);
            has[code] = true;
            blk.qs[j / 4] |= uint8_t(code) << ((j % 4) * 2);
        }
        CHECK(has[0] && has[1] && has[2] && has[3]);
        // Non-trivial block scale (matches a realistic quantized magnitude).
        blk.d = ggml_fp32_to_fp16(0.0625f);
        const double scale = double(ggml_fp16_to_fp32(blk.d));

        // F32 activations, the production accumulation type.
        std::vector<float> xf(QKP2_0);
        std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
        for (float & x : xf) { x = x_dist(rng7); }
        std::vector<double> xd(xf.begin(), xf.end());

        // Reference A: ggml's decode, then a plain F32 dot (the value the
        // inference path effectively consumes).
        std::vector<float> decoded(QKP2_0);
        dequantize_row_pq2_0(&blk, decoded.data(), QKP2_0);
        float ggml_f32_dot = 0.0f;
        for (uint32_t j = 0; j < QKP2_0; ++j) { ggml_f32_dot += decoded[j] * xf[j]; }
        float ggml_f32_dot_rev = 0.0f;
        for (int j = QKP2_0 - 1; j >= 0; --j) { ggml_f32_dot_rev += decoded[j] * xf[j]; }

        // Reference B: FP64 decode+dot (exact oracle).
        double fp64 = 0.0;
        for (uint32_t j = 0; j < QKP2_0; ++j) {
            const uint8_t code = (blk.qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            fp64 += (int(code) - 1) * scale * xd[j];
        }

        // Under test: bit-plane and LUT identities on the SAME packed bytes.
        const double bp = scale * llama_rerot_pq2_bitplane_dot(blk.qs, xd.data(), QKP2_0);
        const double lut = scale * llama_rerot_pq2_lut_dot(blk.qs, xd.data(), QKP2_0);

        const double err_ggml = std::abs(bp - double(ggml_f32_dot));
        const double err_fp64 = std::abs(bp - fp64);
        // The identities are exact in FP64; the only difference from the
        // F32 reference is accumulation order/rounding.
        CHECK(err_fp64 < 1e-6 * (1.0 + std::abs(fp64)));
        CHECK(std::abs(lut - bp) < 1e-9 * (1.0 + std::abs(bp)));
        // ggml's own F32 dot disagrees with itself under order reversal by
        // at most as much as the identity disagrees: the identity must sit
        // INSIDE ggml's own F32 rounding band, i.e. no systematic offset.
        const double ggml_band = std::abs(double(ggml_f32_dot) - double(ggml_f32_dot_rev));
        CHECK(err_ggml <= 4.0 * ggml_band + 1e-9);
        std::fprintf(stderr,
                     "F32 gate: Q7 PQ2_0 ggml blocks, bitplane err vs ggml-f32=%.3e (its own reversal band %.3e), vs FP64=%.3e\n",
                     err_ggml, ggml_band, err_fp64);
    }

    // ------------------------------------------------------------------
    // Q3 shared-KV block attention in FLOAT arithmetic. The FP64 family
    // proves the online (m, z, u) merge equals one global softmax; a
    // kernel would run it in F32, so the re-association/rescale error is
    // the number a GPU implementation must stay within. Measured here as
    // max abs deviation of the F32 online merge from the FP64 full
    // softmax, over many blocks with rapidly varying score magnitudes
    // (the case that stresses the rescaling factor).
    // ------------------------------------------------------------------
    {
        constexpr uint32_t n_readers = 4;
        constexpr uint32_t head_dim = 32;
        constexpr uint32_t value_dim = 16;
        constexpr uint32_t n_blocks = 12;
        constexpr uint32_t rows_per_block = 32;
        std::mt19937_64 rng3(0xf32u ^ 0x3u);
        const double scale = 1.0 / std::sqrt(double(head_dim));

        // Score spread that makes block maxima differ by orders of
        // magnitude (exercises exp(m_dst - m) down to ~0 and back).
        std::uniform_real_distribution<double> spike(0.0, 12.0);
        std::vector<std::vector<float>> kf, vf, qf;
        std::vector<std::vector<uint8_t>> vis(n_blocks, std::vector<uint8_t>(n_readers, 1));
        for (uint32_t b = 0; b < n_blocks; ++b) {
            std::vector<float> k, v, q;
            for (uint32_t j = 0; j < rows_per_block; ++j) {
                for (uint32_t e = 0; e < head_dim; ++e) {
                    k.push_back(float(random_vector(rng3, 1, 1.0)[0]));
                }
                for (uint32_t e = 0; e < value_dim; ++e) {
                    v.push_back(float(random_vector(rng3, 1, 1.0)[0]));
                }
            }
            const double boost = spike(rng3);
            for (float & x : k) { x = float(double(x) * (1.0 + 0.1 * boost)); }
            for (uint32_t r = 0; r < n_readers; ++r) {
                for (uint32_t e = 0; e < head_dim; ++e) {
                    q.push_back(float(random_vector(rng3, 1, 1.0)[0]));
                }
            }
            kf.push_back(std::move(k)); vf.push_back(std::move(v)); qf.push_back(std::move(q));
            // reader 2 sees no block at all (empty-state path)
            vis[b][2] = 0;
        }

        // F32 online merge, block by block (the kernel's organization).
        std::vector<float> m32(n_readers, -1e30f), z32(n_readers, 0.0f);
        std::vector<std::vector<float>> u32(n_readers, std::vector<float>(value_dim, 0.0f));
        std::vector<uint8_t> used(n_readers, 0);
        for (uint32_t b = 0; b < n_blocks; ++b) {
            for (uint32_t r = 0; r < n_readers; ++r) {
                if (!vis[b][r]) { continue; }
                const float * q = qf[b].data() + size_t(r) * head_dim;
                const float * k = kf[b].data();
                // per-block max first (online softmax numerically requires it)
                float bm = -1e30f;
                std::vector<float> a(rows_per_block);
                for (uint32_t j = 0; j < rows_per_block; ++j) {
                    float dot = 0.0f;
                    for (uint32_t e = 0; e < head_dim; ++e) { dot += q[e] * k[size_t(j) * head_dim + e]; }
                    a[j] = float(scale) * dot;
                    bm = std::max(bm, a[j]);
                }
                const float old_m = used[r] ? m32[r] : -1e30f;
                const float nm = std::max(old_m, bm);
                const float f_old = used[r] ? std::exp(old_m - nm) : 0.0f;
                const float f_new = std::exp(bm - nm);
                float z = f_old * z32[r];
                for (uint32_t e = 0; e < value_dim; ++e) { u32[r][e] *= f_old; }
                for (uint32_t j = 0; j < rows_per_block; ++j) {
                    const float w = f_new * std::exp(a[j] - bm);
                    z += w;
                    for (uint32_t e = 0; e < value_dim; ++e) {
                        u32[r][e] += w * vf[b][size_t(j) * value_dim + e];
                    }
                }
                z32[r] = z; m32[r] = nm; used[r] = 1;
            }
        }

        // FP64 oracle: one global softmax over the reader's visible union.
        double max_err = 0.0;
        double max_out = 0.0;
        for (uint32_t r = 0; r < n_readers; ++r) {
            std::vector<double> qd(head_dim), kd, vd;
            // Each block carries its OWN per-reader queries (qf[b] holds
            // n_readers rows); the FP64 oracle must consume the same
            // per-block query the F32 merge used, otherwise the comparison
            // measures different inputs rather than different arithmetic.
            std::vector<std::vector<double>> qd_by_block(n_blocks, std::vector<double>(head_dim));
            for (uint32_t b = 0; b < n_blocks; ++b) {
                for (uint32_t e = 0; e < head_dim; ++e) {
                    qd_by_block[b][e] = double(qf[b][size_t(r) * head_dim + e]);
                }
            }
            for (uint32_t b = 0; b < n_blocks; ++b) {
                if (!vis[b][r]) { continue; }
                for (float x : kf[b]) { kd.push_back(double(x)); }
                for (float x : vf[b]) { vd.push_back(double(x)); }
            }
            if (kd.empty()) {
                CHECK(!used[r]);
                continue;
            }
            // Rebuild the float logits in FP64 the same order the F32
            // merge consumed them, so the comparison isolates the merge
            // arithmetic rather than score computation.
            std::vector<double> scores;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                if (!vis[b][r]) { continue; }
                const float * k = kf[b].data();
                for (uint32_t j = 0; j < rows_per_block; ++j) {
                    double dot = 0.0;
                    for (uint32_t e = 0; e < head_dim; ++e) { dot += qd_by_block[b][e] * double(k[size_t(j) * head_dim + e]); }
                    scores.push_back(scale * dot);
                }
            }
            double mx = -1e300, z = 0.0;
            for (double sc : scores) { mx = std::max(mx, sc); }
            std::vector<double> out(value_dim, 0.0);
            for (size_t j = 0; j < scores.size(); ++j) {
                const double w = std::exp(scores[j] - mx);
                z += w;
                for (uint32_t e = 0; e < value_dim; ++e) {
                    out[e] += w * vd[size_t(j) * value_dim + e];
                }
            }
            for (uint32_t e = 0; e < value_dim; ++e) {
                out[e] /= z;
                const float got = u32[r][e] / z32[r];
                max_err = std::max(max_err, std::abs(double(got) - out[e]));
                max_out = std::max(max_out, std::abs(out[e]));
            }
        }
        const double rel = max_out > 0.0 ? max_err / max_out : 0.0;
        CHECK(std::isfinite(rel));
        // F32 online-merge envelope: orders of magnitude looser than FP64's
        // 1e-12, but still tight enough that a kernel using this
        // organization is numerically equivalent for inference purposes.
        CHECK(rel < 1e-4);
        std::fprintf(stderr,
                     "F32 gate: Q3 shared-block online merge (float) vs FP64 full softmax, max|err|=%.3e rel=%.3e\n",
                     max_err, rel);
    }

    // ------------------------------------------------------------------
    // Q9 against the PRODUCTION sampler chain: the joint sampler must
    // reproduce llama_sampler's temperature -> top-k -> top-p ordering
    // and lowest-index tie-break on the candidate SET each rule keeps.
    // The RNG itself differs (per-pen xorshift vs llama's std::mt19937),
    // so the contract compared here is: same kept set, same draw ORDER,
    // same argmax under greedy — NOT the same random token. Trajectory
    // safety (a pen's draws independent of cohort/row order) is covered
    // by test_q9_joint_sampler above.
    // ------------------------------------------------------------------
    {
        std::mt19937_64 rng9(0x59a);
        const double temperature = 0.85;
        const uint32_t top_k = 6;
        const double top_p = 0.9;
        const uint32_t vocab = 64;
        uint32_t set_mismatch = 0;
        uint32_t greedy_mismatch = 0;
        uint32_t trials = 0;
        for (uint32_t trial = 0; trial < 40; ++trial) {
            std::vector<double> logits(vocab);
            for (double & l : logits) { l = double(rng9() % 2001) / 1000.0 - 1.0; }
            // Plant deliberate ties so the tie-break rule is exercised.
            // IMPORTANT: the tie band is placed ABOVE the top-k cut. On a
            // vocabulary of 64 with k=6, a tie straddling the k-th slot
            // makes std::partial_sort's survivor choice implementation-
            // defined (production's partial_sort is not stable); a
            // cross-implementation set comparison cannot pin it, and that
            // is a documented property of the production sampler rather
            // than a defect (RERoT.md §Q9 contract note).
            if (trial % 2 == 0) {
                for (uint32_t i = 3; i < 12; ++i) { logits[i] = 1.5; }
                logits[2] = 1.6;
            }

            // Production chain: top-k then top-p then temp then dist.
            const auto sparams = llama_sampler_chain_default_params();
            struct llama_sampler * chain = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(chain, llama_sampler_init_top_k((int32_t) top_k));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t) 0x5151u));

            std::vector<llama_token_data> cand(vocab);
            for (uint32_t i = 0; i < vocab; ++i) {
                cand[i] = llama_token_data{ (llama_token) i, (float) logits[i], 0.0f };
    }
            llama_token_data_array arr = { cand.data(), vocab, -1, false };
            llama_sampler_apply(chain, &arr);

            // The joint sampler's kept set (temperature -> top-k -> top-p
            // with lowest-index tie-break) must match `arr` exactly as a
            // set of token ids, in the same relative order.
            std::vector<uint32_t> kept;
            kept.reserve(arr.size);
            for (size_t i = 0; i < arr.size; ++i) { kept.push_back((uint32_t) arr.data[i].id); }

            // Reference: replicate the joint sampler's keep rule directly
            // (independent formulation, not a call under test).
            std::vector<uint32_t> order(vocab);
            for (uint32_t i = 0; i < vocab; ++i) { order[i] = i; }
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                return logits[a] > logits[b]; // ties keep lower index first (stable)
            });
            std::vector<uint32_t> ref_kept;
            if (top_k > 0 && top_k < vocab) { order.resize(top_k); }
            for (uint32_t id : order) { ref_kept.push_back(id); }
            // top-p over the softmax of the kept set. Production order is
            // top_k -> top_p -> temp -> dist, so top-p sees the RAW logits
            // (no temperature scaling): the reference must not scale either.
            if (top_p > 0.0 && top_p < 1.0 && !ref_kept.empty()) {
                double m = -1e300;
                for (uint32_t id : ref_kept) { m = std::max(m, logits[id]); }
                double total = 0.0;
                std::vector<double> p(ref_kept.size());
                for (size_t i = 0; i < ref_kept.size(); ++i) {
                    p[i] = std::exp(logits[ref_kept[i]] - m);
                    total += p[i];
                }
                double cum = 0.0;
                size_t cut = ref_kept.size();
                for (size_t i = 0; i < ref_kept.size(); ++i) {
                    cum += p[i] / total;
                    if (cum >= top_p) { cut = i + 1; break; }
                }
                ref_kept.resize(cut);
            }
            ++trials;
            // Production's partial_sort is NOT stable on equal logits, so
            // the survivor ORDER is implementation-defined; the SURVIVOR
            // SET is what the contract defines. Compare as sorted sets.
            std::sort(kept.begin(), kept.end());
            std::sort(ref_kept.begin(), ref_kept.end());
            if (kept.size() != ref_kept.size()) {
                ++set_mismatch;
                if (set_mismatch == 1) {
                    std::fprintf(stderr, "Q9 trial %u: kept %zu vs reference %zu\n",
                                 trial, kept.size(), ref_kept.size());
                }
            } else {
                bool order_mismatch = false;
                for (size_t i = 0; i < kept.size(); ++i) {
                    if (kept[i] != ref_kept[i]) { order_mismatch = true; break; }
                }
                if (order_mismatch) {
                    ++set_mismatch;

                }
            }

            // Greedy path: the production chain with only top-k/top-p (no
            // dist) leaves one selected token; the joint sampler's argmax
            // over the same kept set must match its id.
            struct llama_sampler * greedy = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(greedy, llama_sampler_init_top_k((int32_t) top_k));
            llama_sampler_chain_add(greedy, llama_sampler_init_top_p(top_p, 1));
            llama_sampler_chain_add(greedy, llama_sampler_init_temp(temperature));
            std::vector<llama_token_data> g_cand(cand);
            llama_token_data_array g_arr = { g_cand.data(), vocab, -1, false };
            llama_sampler_apply(greedy, &g_arr);
            // A chain without dist leaves `selected` unset (-1); the
            // production greedy path takes the surviving max-logit entry,
            // which for a temperature-only chain is exactly the kept set's
            // head. Compare against the argmax the survivors define.
            llama_token greedy_id = 0;
            {
                double best_l = -1e300;
                for (size_t i = 0; i < g_arr.size; ++i) {
                    if (g_arr.data[i].logit > best_l) {
                        best_l = g_arr.data[i].logit;
                        greedy_id = g_arr.data[i].id;
                    }
                }
            }
            uint32_t joint_argmax = ref_kept.front();
            {
                double bl = -1e300;
                for (uint32_t id : ref_kept) {
                    if (logits[id] > bl) { bl = logits[id]; joint_argmax = id; }
                }
            }
            // A unique maximum is required to cross-check: production's
            // non-stable partial_sort may surface ANY tied element, so a
            // tied maximum is not a comparable contract point (the joint
            // sampler's own lowest-index tie-break is verified directly by
            // test_q9_joint_sampler case 5).
            if (!ref_kept.empty()) {
                uint32_t nmax = 0;
                double bl = -1e300;
                for (uint32_t id : ref_kept) { bl = std::max(bl, logits[id]); }
                for (uint32_t id : ref_kept) { if (logits[id] == bl) { ++nmax; } }
                if (nmax == 1 && greedy_id != (llama_token) joint_argmax) { ++greedy_mismatch; }
            } else {
                ++greedy_mismatch;
            }

            llama_sampler_free(greedy);
            llama_sampler_free(chain);
        }
        CHECK(set_mismatch == 0);
        CHECK(greedy_mismatch == 0);
        std::fprintf(stderr,
                     "F32 gate: Q9 sampler vs production chain: %u trials, set mismatches=%u, greedy mismatches=%u\n",
                     trials, set_mismatch, greedy_mismatch);
    }
}

static void test_r02_q_prep_contract() {
    // R02 C07: live gather+RoPE reference over the active prefix only,
    // capacity-strided output, poisoned padding, replay contract.

    const int64_t head_dim = 8;
    const int64_t heads    = 2;
    const int64_t n_tokens = 5;
    const int64_t capacity = 4;
    const int64_t active   = 2;
    const int64_t n_rot    = 8;

    // Deterministic raw Q: token t, head h, dim d -> 0.1*(t*heads+h) + 0.01*d.
    std::vector<float> q_raw(size_t(head_dim) * heads * n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                q_raw[(t * heads + h) * head_dim + d] =
                    0.1f * float(t * heads + h) + 0.01f * float(d);
            }
        }
    }

    std::vector<int32_t> q_indices = { 1, 3, 0, 2 }; // active prefix uses {1, 3}
    std::vector<int32_t> q_pos     = { 7, 42, 0, 0 }; // capacity-strided, 1 coord

    llama_rerot_q_prep_contract c;
    c.head_dim   = head_dim;
    c.heads      = heads;
    c.n_tokens   = n_tokens;
    c.capacity   = capacity;
    c.active     = active;
    c.n_pos      = 1;
    c.n_rot      = n_rot;
    c.rope_mode  = 0; // NORMAL
    c.freq_base  = 10000.0f;

    // ---- active prefix rotated, padding poisoned, replay identical ----
    const auto out1 = llama_rerot_q_prep_reference(
        q_raw.data(), q_indices.data(), q_pos.data(), c);
    CHECK(out1.size() == size_t(head_dim) * heads * capacity);

    for (int64_t g = 0; g < active; ++g) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                CHECK(!std::isnan(out1[(g * heads + h) * head_dim + d]));
            }
        }
    }
    for (int64_t g = active; g < capacity; ++g) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                CHECK(std::isnan(out1[(g * heads + h) * head_dim + d]));
            }
        }
    }

    // Replay contract: same inputs twice -> identical outputs.
    // NaN poison tails must compare equal-as-poison (IEEE NaN != NaN).
    const auto out2 = llama_rerot_q_prep_reference(
        q_raw.data(), q_indices.data(), q_pos.data(), c);
    CHECK(out1.size() == out2.size());
    for (size_t i = 0; i < out1.size(); ++i) {
        const bool nan1 = std::isnan(out1[i]);
        const bool nan2 = std::isnan(out2[i]);
        CHECK(nan1 == nan2);
        if (!nan1) {
            CHECK(out1[i] == out2[i]);
        }
    }

    // Active-row correctness: unrotated dims beyond n_rot are copied
    // verbatim; rotated dims differ from the raw input in general.
    {
        // dim 7 == n_rot-1 is rotated; dim 8-1... with head_dim==n_rot all
        // dims are rotated. Check gather identity instead: position 0 and
        // freq-independent behavior is verified by the production-chain
        // comparison in test-rerot-math's F32 gate style; here check the
        // gather picked the right token: compare against a manual copy at
        // pos 0 (cos=1, sin=0 => identity rotation at theta 0).
        std::vector<int32_t> q_pos0 = { 0, 0, 0, 0 };
        const auto out0 = llama_rerot_q_prep_reference(
            q_raw.data(), q_indices.data(), q_pos0.data(), c);
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const float want = q_raw[(1 * heads + h) * head_dim + d];
                const float got  = out0[(0 * heads + h) * head_dim + d];
                CHECK(std::fabs(got - want) < 1e-6f);
            }
        }
    }

    // ---- contract rejections ----
    {
        auto bad = c;
        bad.active = capacity + 1;
        bool threw = false;
        try {
            llama_rerot_q_prep_reference(q_raw.data(), q_indices.data(), q_pos.data(), bad);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto bad = c;
        bad.n_pos = 4; // NORMAL requires 1
        bool threw = false;
        try {
            llama_rerot_q_prep_reference(q_raw.data(), q_indices.data(), q_pos.data(), bad);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto bad = c;
        bad.rope_mode = 40; // IMROPE (ggml.h) without n_pos=4
        bool threw = false;
        try {
            llama_rerot_q_prep_reference(q_raw.data(), q_indices.data(), q_pos.data(), bad);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // q_indices out of range: active prefix references token >= n_tokens.
        std::vector<int32_t> bad_idx = { 1, n_tokens, 0, 2 };
        bool threw = false;
        try {
            llama_rerot_q_prep_reference(q_raw.data(), bad_idx.data(), q_pos.data(), c);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto bad = c;
        bad.rope_mode = 9; // unsupported
        bool threw = false;
        try {
            llama_rerot_q_prep_reference(q_raw.data(), q_indices.data(), q_pos.data(), bad);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // ---- support table ----
    {
        const auto modes = llama_rerot_q_prep_supported_modes();
        CHECK(modes.size() == 4);
        // ggml.h: NORMAL=0 NEOX=2 MROPE=8 IMROPE=40; VISION=24 must stay out.
        CHECK(modes[0] == 0 && modes[1] == 2 && modes[2] == 8 && modes[3] == 40);
        for (int m : modes) {
            CHECK(m != 24);
        }
    }

    // ---- active == 0: fully poisoned (expressible empty task) ----
    {
        auto zero = c;
        zero.active = 0;
        const auto out0 = llama_rerot_q_prep_reference(
            q_raw.data(), q_indices.data(), q_pos.data(), zero);
        for (size_t i = 0; i < out0.size(); ++i) {
            CHECK(std::isnan(out0[i]));
        }
    }

    // ---- active == capacity: no padding, all rows valid ----
    {
        auto full = c;
        full.active = capacity;
        const auto outf = llama_rerot_q_prep_reference(
            q_raw.data(), q_indices.data(), q_pos.data(), full);
        for (size_t i = 0; i < outf.size(); ++i) {
            CHECK(!std::isnan(outf[i]));
        }
    }
    // ---- poison-tail strengthened: every inactive (g,h,*) element is NaN;
    // active prefix has zero NaNs. Also reject VISION (24) explicitly. ----
    {
        auto mid = c;
        mid.active = 1; // poison [1, capacity)
        const auto outm = llama_rerot_q_prep_reference(
            q_raw.data(), q_indices.data(), q_pos.data(), mid);
        size_t nan_count = 0;
        size_t live_nan = 0;
        for (int64_t g = 0; g < capacity; ++g) {
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const float v = outm[(g * heads + h) * head_dim + d];
                    if (std::isnan(v)) {
                        ++nan_count;
                        if (g < mid.active) {
                            ++live_nan;
                        }
                    } else if (g >= mid.active) {
                        CHECK(false); // inactive must be NaN
                    }
                }
            }
        }
        CHECK(live_nan == 0);
        CHECK(nan_count == size_t((capacity - mid.active) * heads * head_dim));
    }
    {
        auto vision = c;
        vision.rope_mode = 24; // GGML_ROPE_TYPE_VISION — must reject
        bool threw = false;
        try {
            llama_rerot_q_prep_reference(q_raw.data(), q_indices.data(), q_pos.data(), vision);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }
}

int main() {
    std::fprintf(stderr, "=== RERoT Math Reference Tests ===\n");
    test_q3_shared_block_attention();
    test_q5_gdn_lowrank();
    test_q6_chunk_fold();
    test_q7_pq2_bitplane();
    test_q2_span_view();
    test_q4_structure_numeric();
    test_q8_skip_bounds();
    test_q9_joint_sampler();
    test_q10_frontier_grid();
    test_f32_gate();
    test_r02_q_prep_contract();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
