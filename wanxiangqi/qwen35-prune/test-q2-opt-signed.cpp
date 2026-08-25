#include "q2-opt-signed.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace qwen35_prune;

static bool close_enough(double a, double b) {
    const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= 1e-10 * scale;
}

static bool check_decoder() {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q2_0);
    if (!tt || !tt->to_float || tt->type_size != sizeof(q2_opt_block) || tt->blck_size != 64) {
        std::fprintf(stderr, "stock Q2_0 decoder traits unavailable or layout mismatch\n");
        return false;
    }

    // Exercise both signs and extreme/subnormal finite FP16 values. The custom
    // encoder is only valid if its 18 emitted bytes are decoded identically by
    // the stock GGML_TYPE_Q2_0 decoder.
    const ggml_fp16_t scales[] = {
        ggml_fp32_to_fp16(+1.0f),
        ggml_fp32_to_fp16(-1.0f),
        (ggml_fp16_t) 0x0001u,
        (ggml_fp16_t) 0x8001u,
        (ggml_fp16_t) 0x7bffu,
        (ggml_fp16_t) 0xfbffu,
    };
    for (ggml_fp16_t scale : scales) {
        q2_opt_block b {};
        b.d = scale;
        for (int i = 0; i < 64; ++i) {
            const uint8_t c = i % 4;
            b.qs[i / 4] |= c << (2 * (i % 4));
        }
        float got[64];
        float stock[64];
        decode_q2_opt(b, got);
        tt->to_float(&b, stock, 64);
        for (int i = 0; i < 64; ++i) {
            if (got[i] != stock[i]) {
                std::fprintf(stderr, "stock decoder mismatch scale=0x%04x at %d: %.9g != %.9g\n",
                        (unsigned) scale, i, got[i], stock[i]);
                return false;
            }
        }
    }
    return true;
}

static bool check_case(const float * w, const float * h, const char * name, bool run_oracle) {
    q2_opt_block stock {}, pos {}, signed_b {}, oracle {}, oracle_pos {};
    const auto es = encode_q2_stock(w, h, stock);
    const auto ep = encode_q2_opt(w, h, pos, q2_opt_mode::positive);
    const auto en = encode_q2_opt(w, h, signed_b, q2_opt_mode::signed_scale);
    const auto sp = score_q2_opt(w, h, q2_opt_mode::positive);
    const auto sn = score_q2_opt(w, h, q2_opt_mode::signed_scale);
    q2_opt_block compare_b {};
    q2_opt_metrics compare_pos {};
    const auto compare_signed = encode_q2_opt_signed_compare(w, h, compare_b, compare_pos);

    if (!q2_opt_block_equal(compare_b, signed_b) ||
        !close_enough(compare_signed.weighted_sse, en.weighted_sse) ||
        !close_enough(compare_signed.unweighted_sse, en.unweighted_sse) ||
        !close_enough(compare_pos.weighted_sse, ep.weighted_sse) ||
        !close_enough(compare_pos.unweighted_sse, ep.unweighted_sse)) {
        std::fprintf(stderr, "%s: combined signed/positive sweep mismatch\n", name);
        return false;
    }

    if (!close_enough(sn.weighted_sse, en.weighted_sse) ||
        !close_enough(sn.unweighted_sse, en.unweighted_sse) ||
        !close_enough(sn.weighted_power, en.weighted_power) ||
        !close_enough(sp.weighted_sse, ep.weighted_sse) ||
        !close_enough(sp.unweighted_sse, ep.unweighted_sse) ||
        !close_enough(sp.weighted_power, ep.weighted_power)) {
        std::fprintf(stderr, "%s: metrics-only scorer mismatch\n", name);
        return false;
    }

    const double eps = 1e-9 * std::max(1.0, es.weighted_sse);
    if (en.weighted_sse > ep.weighted_sse + eps || ep.weighted_sse > es.weighted_sse + eps) {
        std::fprintf(stderr, "%s: inclusion failed signed=%g positive=%g stock=%g\n",
                name, en.weighted_sse, ep.weighted_sse, es.weighted_sse);
        return false;
    }

    q2_opt_block again {};
    encode_q2_opt(w, h, again, q2_opt_mode::signed_scale);
    if (!q2_opt_block_equal(signed_b, again)) {
        std::fprintf(stderr, "%s: non-deterministic output\n", name);
        return false;
    }

    if (run_oracle) {
        const auto eo = encode_q2_oracle(w, h, oracle, true);
        const auto eop = encode_q2_oracle(w, h, oracle_pos, false);
        if (!close_enough(en.weighted_sse, eo.weighted_sse) ||
            !close_enough(en.unweighted_sse, eo.unweighted_sse) ||
            !q2_opt_block_equal(signed_b, oracle)) {
            std::fprintf(stderr, "%s: oracle mismatch fast=(%g,%g,d=%g) oracle=(%g,%g,d=%g)\n",
                    name,
                    en.weighted_sse, en.unweighted_sse, ggml_fp16_to_fp32(signed_b.d),
                    eo.weighted_sse, eo.unweighted_sse, ggml_fp16_to_fp32(oracle.d));
            const auto * fb = reinterpret_cast<const unsigned char *>(&signed_b);
            const auto * ob = reinterpret_cast<const unsigned char *>(&oracle);
            std::fprintf(stderr, "  fast bytes:");
            for (size_t i = 0; i < sizeof(signed_b); ++i) std::fprintf(stderr, " %02x", fb[i]);
            std::fprintf(stderr, "\n  oracle bytes:");
            for (size_t i = 0; i < sizeof(oracle); ++i) std::fprintf(stderr, " %02x", ob[i]);
            std::fprintf(stderr, "\n");
            return false;
        }
        if (!close_enough(ep.weighted_sse, eop.weighted_sse) ||
            !close_enough(ep.unweighted_sse, eop.unweighted_sse) ||
            !q2_opt_block_equal(pos, oracle_pos)) {
            std::fprintf(stderr, "%s: positive oracle mismatch fast=(%g,%g,d=%g) oracle=(%g,%g,d=%g)\n",
                    name,
                    ep.weighted_sse, ep.unweighted_sse, ggml_fp16_to_fp32(pos.d),
                    eop.weighted_sse, eop.unweighted_sse, ggml_fp16_to_fp32(oracle_pos.d));
            return false;
        }
    }
    return true;
}

static bool check_thread_determinism(const float * w, const float * h) {
    q2_opt_block reference {};
    encode_q2_opt(w, h, reference, q2_opt_mode::signed_scale);
    std::atomic<bool> ok { true };
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; ++i) {
                q2_opt_block got {};
                encode_q2_opt(w, h, got, q2_opt_mode::signed_scale);
                if (!q2_opt_block_equal(got, reference)) ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (auto & thread : threads) thread.join();
    if (!ok.load(std::memory_order_relaxed)) {
        std::fprintf(stderr, "thread determinism failed\n");
        return false;
    }
    return true;
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static bool check_synchronized_permutation() {
    constexpr int N_IN = 16;
    constexpr int N_FF = 64;
    constexpr int N_OUT = 8;
    std::mt19937 rng(0x51574947u);
    std::normal_distribution<float> dist(0.0f, 0.25f);
    std::array<float, N_IN> x {};
    std::array<std::array<float, N_IN>, N_FF> gate {}, up {};
    std::array<std::array<float, N_FF>, N_OUT> down {};
    for (float & v : x) v = dist(rng);
    for (auto & row : gate) for (float & v : row) v = dist(rng);
    for (auto & row : up) for (float & v : row) v = dist(rng);
    for (auto & row : down) for (float & v : row) v = dist(rng);

    auto eval = [&](const std::array<int, N_FF> & gu_map, const std::array<int, N_FF> & d_map) {
        std::array<float, N_FF> a {};
        for (int j = 0; j < N_FF; ++j) {
            float g = 0.0f, u = 0.0f;
            for (int i = 0; i < N_IN; ++i) {
                g += gate[(size_t) gu_map[(size_t) j]][(size_t) i] * x[(size_t) i];
                u += up[(size_t) gu_map[(size_t) j]][(size_t) i] * x[(size_t) i];
            }
            a[(size_t) j] = silu(g) * u;
        }
        std::array<float, N_OUT> y {};
        for (int o = 0; o < N_OUT; ++o) {
            for (int j = 0; j < N_FF; ++j) y[(size_t) o] += down[(size_t) o][(size_t) d_map[(size_t) j]] * a[(size_t) j];
        }
        return y;
    };

    std::array<int, N_FF> identity {}, perm {};
    for (int i = 0; i < N_FF; ++i) identity[(size_t) i] = perm[(size_t) i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    const auto ref = eval(identity, identity);
    const auto synchronized = eval(perm, perm);
    for (int i = 0; i < N_OUT; ++i) {
        if (std::fabs(ref[(size_t) i] - synchronized[(size_t) i]) > 2e-5f) {
            std::fprintf(stderr, "synchronized FFN permutation changed output at %d\n", i);
            return false;
        }
    }

    auto wrong_down = perm;
    std::rotate(wrong_down.begin(), wrong_down.begin() + 1, wrong_down.end());
    const auto broken = eval(perm, wrong_down);
    float max_diff = 0.0f;
    for (int i = 0; i < N_OUT; ++i) max_diff = std::max(max_diff, std::fabs(ref[(size_t) i] - broken[(size_t) i]));
    if (max_diff < 1e-3f) {
        std::fprintf(stderr, "negative permutation test did not detect wrong down mapping\n");
        return false;
    }
    return true;
}

int main() {
    if (!check_decoder()) return 1;

    float w[64] = {};
    float h[64];
    for (float & x : h) x = 1.0f;
    if (!check_case(w, h, "zero", true)) return 1;

    for (float & x : w) x = 0.75f;
    if (!check_case(w, h, "constant", true)) return 1;

    for (int i = 0; i < 64; ++i) w[i] = i == 7 ? -9.0f : 0.2f * std::sin((float) i);
    if (!check_case(w, h, "negative-outlier", true)) return 1;

    for (int i = 0; i < 64; ++i) w[i] = i == 11 ? 9.0f : 0.2f * std::cos((float) i);
    if (!check_case(w, h, "positive-outlier", true)) return 1;

    for (int i = 0; i < 64; ++i) w[i] = 2.0f + 0.03f * std::sin((float) i);
    if (!check_case(w, h, "positive-bias", true)) return 1;

    for (int i = 0; i < 64; ++i) w[i] = -2.0f + 0.03f * std::cos((float) i);
    if (!check_case(w, h, "negative-bias", true)) return 1;

    for (int i = 0; i < 64; ++i) w[i] = (i % 4 - 1.5f) * 0.5f;
    if (!check_case(w, h, "midpoint-ties", true)) return 1;

    for (int i = 0; i < 64; ++i) h[i] = (i & 1) ? 0.0f : 1.0f + 0.03f * i;
    if (!check_case(w, h, "half-zero-weights", true)) return 1;

    for (float & x : h) x = 0.0f;
    if (!check_case(w, h, "all-zero-weights", true)) return 1;

    std::mt19937 rng(0xBA0u);
    std::normal_distribution<float> dist(0.0f, 1.3f);
    std::uniform_real_distribution<float> hdist(0.0f, 2.0f);
    for (int t = 0; t < 32; ++t) {
        for (int i = 0; i < 64; ++i) {
            w[i] = dist(rng);
            h[i] = hdist(rng);
        }
        if (!check_case(w, h, "random", true)) return 1;
    }
    if (!check_thread_determinism(w, h)) return 1;
    if (!check_synchronized_permutation()) return 1;

    std::puts("q2-opt-signed: all tests passed");
    return 0;
}
