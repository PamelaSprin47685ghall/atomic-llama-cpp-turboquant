#include "q2-opt-signed.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qwen35_prune {
namespace {

struct candidate_score {
    q2_opt_block block {};
    q2_opt_metrics metrics {};
    bool valid = false;
};

static inline uint16_t fp16_bits(float v) {
    return GGML_FP32_TO_FP16(v);
}

static inline float fp16_value(uint16_t bits) {
    return GGML_FP16_TO_FP32(bits);
}

static inline bool finite_half(uint16_t bits) {
    return (bits & 0x7c00u) != 0x7c00u;
}

// Nearest code for the stock Q2_0 reconstruction levels (code-1)*d.
// Expressing the choice through the three exact midpoints is equivalent to
// trying all four codes, but avoids four squared-distance evaluations in the
// hottest inner loop.  <= preserves the existing lower-code midpoint tie rule.
static inline uint8_t midpoint_code(float w, float d) {
    if (d == 0.0f) return 1u;
    const double a = std::fabs((double) d);
    const double y = std::signbit(d) ? -(double) w : (double) w;
    if (y <= -0.5 * a) return 0u;
    if (y <=  0.5 * a) return 1u;
    if (y <=  1.5 * a) return 2u;
    return 3u;
}

static candidate_score score_scale(const float * w, const float * h, uint16_t d_bits) {
    candidate_score result;
    if (!finite_half(d_bits)) {
        return result;
    }

    float d = fp16_value(d_bits);
    if (d == 0.0f) {
        d = 0.0f;
        d_bits = fp16_bits(0.0f);
    }

    result.block.d = d_bits;
    std::memset(result.block.qs, 0, sizeof(result.block.qs));

    for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
        const uint8_t code = midpoint_code(w[i], d);
        result.block.qs[i / 4] |= code << (2 * (i % 4));
        const int c = (int) code - 1;
        const double diff = (double) w[i] - (double) c * d;
        result.metrics.weighted_sse += (double) h[i] * diff * diff;
        result.metrics.unweighted_sse += diff * diff;
    }
    result.valid = std::isfinite(result.metrics.weighted_sse) && std::isfinite(result.metrics.unweighted_sse);
    return result;
}

// Deliberately independent slow scorer used only by the exhaustive FP16
// oracle. Keep the literal four-code search here instead of sharing the
// production midpoint selector, so the oracle can catch a regression in that
// optimization rather than proving the implementation with itself.
static candidate_score score_scale_oracle(const float * w, const float * h, uint16_t d_bits) {
    candidate_score result;
    if (!finite_half(d_bits)) return result;

    float d = fp16_value(d_bits);
    if (d == 0.0f) {
        d = 0.0f;
        d_bits = fp16_bits(0.0f);
        result.block.d = d_bits;
        std::memset(result.block.qs, 0x55, sizeof(result.block.qs));
        for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
            const double wi = w[i];
            result.metrics.weighted_sse += (double) h[i] * wi * wi;
            result.metrics.unweighted_sse += wi * wi;
        }
        result.valid = std::isfinite(result.metrics.weighted_sse) &&
                       std::isfinite(result.metrics.unweighted_sse);
        return result;
    }
    result.block.d = d_bits;
    std::memset(result.block.qs, 0, sizeof(result.block.qs));

    for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
        uint8_t best_code = 0;
        double best_error = std::numeric_limits<double>::infinity();
        for (uint8_t code = 0; code < 4; ++code) {
            const double recon = (double) ((int) code - 1) * d;
            const double diff = (double) w[i] - recon;
            const double error = diff * diff;
            if (error < best_error || (error == best_error && code < best_code)) {
                best_error = error;
                best_code = code;
            }
        }
        result.block.qs[i / 4] |= best_code << (2 * (i % 4));
        const double recon = (double) ((int) best_code - 1) * d;
        const double diff = (double) w[i] - recon;
        result.metrics.weighted_sse += (double) h[i] * diff * diff;
        result.metrics.unweighted_sse += diff * diff;
    }
    result.valid = std::isfinite(result.metrics.weighted_sse) &&
                   std::isfinite(result.metrics.unweighted_sse);
    return result;
}

static int compare_block_bytes(const q2_opt_block & a, const q2_opt_block & b) {
    return std::memcmp(&a, &b, sizeof(a));
}

static bool better(const candidate_score & a, const candidate_score & b) {
    if (!a.valid) return false;
    if (!b.valid) return true;
    if (a.metrics.weighted_sse != b.metrics.weighted_sse) return a.metrics.weighted_sse < b.metrics.weighted_sse;
    if (a.metrics.unweighted_sse != b.metrics.unweighted_sse) return a.metrics.unweighted_sse < b.metrics.unweighted_sse;
    const float ad = std::fabs(fp16_value(a.block.d));
    const float bd = std::fabs(fp16_value(b.block.d));
    if (ad != bd) return ad < bd;
    return compare_block_bytes(a.block, b.block) < 0;
}

struct formula_best {
    uint16_t d_bits = 0;
    double weighted = std::numeric_limits<double>::infinity();
    double unweighted = std::numeric_limits<double>::infinity();
    bool valid = false;
};

struct fast_search_state {
    const float * w = nullptr;
    const float * h = nullptr;
    double weighted_power = 0.0;
    formula_best best_formula {};
    candidate_score exact_tie_best {};
};

static bool formula_better(
        const fast_search_state & state,
        double we,
        double ue,
        uint16_t d_bits) {
    if (!state.best_formula.valid) return true;
    if (we != state.best_formula.weighted) return we < state.best_formula.weighted;
    if (ue != state.best_formula.unweighted) return ue < state.best_formula.unweighted;
    const float ad = std::fabs(fp16_value(d_bits));
    const float bd = std::fabs(fp16_value(state.best_formula.d_bits));
    if (ad != bd) return ad < bd;

    // A mathematically exact tie is rare but important for all-zero and
    // symmetric blocks. Resolve the final packed-byte key with the same
    // direct scorer as the slow oracle.
    const candidate_score ca = score_scale(state.w, state.h, d_bits);
    const candidate_score cb = score_scale(state.w, state.h, state.best_formula.d_bits);
    return better(ca, cb);
}

static void consider_formula(
        fast_search_state & state,
        double we,
        double ue,
        uint16_t d_bits) {
    if (!finite_half(d_bits)) return;
    if (we < 0.0 && we > -1e-12) we = 0.0;
    if (ue < 0.0 && ue > -1e-12) ue = 0.0;
    if (formula_better(state, we, ue, d_bits)) {
        state.best_formula = {d_bits, we, ue, true};
    }
}

static candidate_score materialize_current_best(const fast_search_state & state) {
    candidate_score result;
    if (!state.best_formula.valid) return result;
    result = score_scale(state.w, state.h, state.best_formula.d_bits);
    if (state.exact_tie_best.valid && better(state.exact_tie_best, result)) {
        result = state.exact_tie_best;
    }
    result.metrics.weighted_power = state.weighted_power;
    return result;
}

static candidate_score metrics_current_best(const fast_search_state & state) {
    candidate_score result;
    if (!state.best_formula.valid) return result;
    result.block.d = state.best_formula.d_bits;
    result.metrics.weighted_sse = state.best_formula.weighted;
    result.metrics.unweighted_sse = state.best_formula.unweighted;
    result.valid = std::isfinite(result.metrics.weighted_sse) && std::isfinite(result.metrics.unweighted_sse);
    return result;
}

static void direct_consider(
        fast_search_state & state,
        int sign,
        uint16_t positive_bits) {
    if (positive_bits > 0x7bffu) return;
    const float a = fp16_value(positive_bits);
    const uint16_t db = fp16_bits(sign * a);
    const candidate_score cur = score_scale(state.w, state.h, db);
    if (better(cur, state.exact_tie_best)) state.exact_tie_best = cur;
    consider_formula(state, cur.metrics.weighted_sse, cur.metrics.unweighted_sse, cur.block.d);
}

static void consider_interval_optimum(
        fast_search_state & state,
        int sign,
        double target,
        double lo,
        double hi,
        double Cw,
        double Aw,
        double Bw,
        double Cu,
        double Au,
        double Bu) {
    if (!(target >= 0.0) || !std::isfinite(target)) return;
    target = std::max(target, lo);
    if (std::isfinite(hi)) target = std::min(target, hi);
    target = std::min(target, (double) fp16_value(0x7bff));

    // For a fixed code assignment the objective is a convex quadratic in a.
    // Therefore the exact optimum over the discrete positive-FP16 lattice in
    // this interval can only be one of the two FP16 values enclosing clamp(a*).
    // This also subsumes the explicit below/at/above-breakpoint candidates from
    // BAOMU: when a* clamps to a cut, the adjacent intervals independently test
    // the representable value on their side. At an exactly representable cut,
    // the tied code assignments have identical squared error; final score_scale
    // reassigns codes and applies the canonical packed-byte tie break.
    const uint16_t center = std::min<uint16_t>(fp16_bits((float) target), 0x7bffu);
    const double center_value = fp16_value(center);
    uint16_t candidates[2] = { center, center };
    int n_candidates = 1;
    if (center_value < target && center < 0x7bffu) {
        candidates[1] = center + 1;
        n_candidates = 2;
    } else if (center_value > target && center > 0u) {
        candidates[0] = center - 1;
        candidates[1] = center;
        n_candidates = 2;
    }

    for (int k = 0; k < n_candidates; ++k) {
        const uint16_t hb = candidates[k];
        const double a = fp16_value(hb);
        if (a == 0.0 || a < lo || a > hi) continue;
        const uint16_t db = fp16_bits(sign * (float) a);
        const double we = Cw - 2.0*a*Aw + a*a*Bw;
        const double ue = Cu - 2.0*a*Au + a*a*Bu;
        consider_formula(state, we, ue, db);
    }
}

static candidate_score encode_fast(
        const float * w,
        const float * h,
        bool allow_negative,
        candidate_score * positive_best = nullptr,
        bool materialize = true) {
    double h_sum = 0.0;
    double Cw_base = 0.0;
    double Cu_base = 0.0;
    for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
        if (!std::isfinite(w[i]) || !std::isfinite(h[i]) || h[i] < 0.0f) {
            if (positive_best) *positive_best = {};
            return {};
        }
        const double wi = w[i];
        h_sum += h[i];
        Cw_base += (double) h[i] * wi * wi;
        Cu_base += wi * wi;
    }
    const bool use_unweighted_for_search = h_sum == 0.0;

    struct event {
        double a;
        uint8_t index;
        int8_t new_code;
    };
    struct ordered_weight {
        double magnitude;
        uint8_t index;
    };

    const auto weight_less = [](const ordered_weight & a, const ordered_weight & b) {
        if (a.magnitude != b.magnitude) return a.magnitude < b.magnitude;
        return a.index < b.index;
    };
    std::array<ordered_weight, Q2_OPT_BLOCK_SIZE> ordered_abs {};
    int n_ordered_abs = 0;
    for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
        if (w[i] != 0.0f) {
            ordered_abs[(size_t) n_ordered_abs++] = {std::fabs((double) w[i]), (uint8_t) i};
        }
    }
    std::sort(ordered_abs.begin(), ordered_abs.begin() + n_ordered_abs, weight_less);

    fast_search_state state { w, h, Cw_base };
    const candidate_score zero = score_scale(w, h, fp16_bits(0.0f));
    consider_formula(state, zero.metrics.weighted_sse, zero.metrics.unweighted_sse, zero.block.d);

    const int signs[2] = { +1, -1 };
    const int n_signs = allow_negative ? 2 : 1;
    for (int si = 0; si < n_signs; ++si) {
        const int sign = signs[si];
        std::array<event, 2 * Q2_OPT_BLOCK_SIZE> events {};
        int n_events = 0;
        std::array<ordered_weight, Q2_OPT_BLOCK_SIZE> negative {};
        std::array<ordered_weight, Q2_OPT_BLOCK_SIZE> positive {};
        int n_negative = 0;
        int n_positive = 0;
        std::array<int8_t, Q2_OPT_BLOCK_SIZE> codes {};

        double Cw = Cw_base, Aw = 0.0, Bw = 0.0;
        double Cu = Cu_base, Au = 0.0, Bu = 0.0;
        for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
            const double y = (double) sign * w[i];
            int c = 0;
            if (y < 0.0) {
                c = -1;
                negative[(size_t) n_negative++] = {-y, (uint8_t) i};
            } else if (y > 0.0) {
                c = 2;
                positive[(size_t) n_positive++] = {y, (uint8_t) i};
            }
            codes[(size_t) i] = (int8_t) c;
            const double hw = h[i];
            Aw += hw * c * y;
            Bw += hw * c * c;
            Au += c * y;
            Bu += c * c;
        }

        // Sign only partitions the already sorted |w| sequence. Filtering a
        // sorted sequence preserves the exact (magnitude,index) order, so both
        // orientations share one sort per block instead of sorting their
        // positive/negative subsets independently.
        n_negative = 0;
        n_positive = 0;
        for (int oi = 0; oi < n_ordered_abs; ++oi) {
            const auto & ow = ordered_abs[(size_t) oi];
            const double y = (double) sign * w[ow.index];
            if (y < 0.0) negative[(size_t) n_negative++] = ow;
            else if (y > 0.0) positive[(size_t) n_positive++] = ow;
        }

        const auto event_less = [](const event & a, const event & b) {
            if (a.a != b.a) return a.a < b.a;
            if (a.index != b.index) return a.index < b.index;
            return a.new_code < b.new_code;
        };

        // The breakpoint set is the merge of three monotone streams. Sorting
        // at most 64 positive and 64 negative magnitudes is cheaper than a
        // general sort of up to 128 event records, while producing the exact
        // same (a,index,new_code) order used by the sweep below.
        int ineg = 0, ipos_small = 0, ipos_large = 0;
        while (ineg < n_negative || ipos_small < n_positive || ipos_large < n_positive) {
            event best {std::numeric_limits<double>::infinity(), 0xffu, 0x7f};
            int stream = -1;
            if (ineg < n_negative) {
                const auto & x = negative[(size_t) ineg];
                const event cur {2.0 * x.magnitude, x.index, 0};
                if (stream < 0 || event_less(cur, best)) { best = cur; stream = 0; }
            }
            if (ipos_small < n_positive) {
                const auto & x = positive[(size_t) ipos_small];
                const event cur {2.0 * x.magnitude / 3.0, x.index, 1};
                if (stream < 0 || event_less(cur, best)) { best = cur; stream = 1; }
            }
            if (ipos_large < n_positive) {
                const auto & x = positive[(size_t) ipos_large];
                const event cur {2.0 * x.magnitude, x.index, 0};
                if (stream < 0 || event_less(cur, best)) { best = cur; stream = 2; }
            }
            events[(size_t) n_events++] = best;
            if (stream == 0) ++ineg;
            else if (stream == 1) ++ipos_small;
            else ++ipos_large;
        }

        double lo = 0.0;
        int ei = 0;
        while (true) {
            const double hi = ei < n_events ? events[(size_t) ei].a : std::numeric_limits<double>::infinity();
            const double A = use_unweighted_for_search ? Au : Aw;
            const double B = use_unweighted_for_search ? Bu : Bw;
            if (B > 0.0) {
                double a_star = A / B;
                if (a_star < lo) a_star = lo;
                if (std::isfinite(hi) && a_star > hi) a_star = hi;
                consider_interval_optimum(state, sign, a_star, lo, hi, Cw, Aw, Bw, Cu, Au, Bu);
            }
            if (ei >= n_events) break;

            const double cut = events[(size_t) ei].a;
            while (ei < n_events && events[(size_t) ei].a == cut) {
                const int i = events[(size_t) ei].index;
                const int old_c = codes[(size_t) i];
                const int new_c = events[(size_t) ei].new_code;
                const double y = (double) sign * w[i];
                const double hw = h[i];
                Aw += hw * (new_c - old_c) * y;
                Bw += hw * (new_c*new_c - old_c*old_c);
                Au += (new_c - old_c) * y;
                Bu += new_c*new_c - old_c*old_c;
                codes[(size_t) i] = (int8_t) new_c;
                ++ei;
            }
            lo = cut;
        }

        direct_consider(state, sign, 0x0001u);
        direct_consider(state, sign, 0x7bffu);
        if (si == 0 && positive_best) {
            *positive_best = materialize ? materialize_current_best(state) : metrics_current_best(state);
            positive_best->metrics.weighted_power = Cw_base;
        }
    }

    candidate_score result = materialize ? materialize_current_best(state) : metrics_current_best(state);
    result.metrics.weighted_power = Cw_base;
    return result;
}

} // namespace

q2_opt_metrics encode_q2_opt(const float * w, const float * h, q2_opt_block & out, q2_opt_mode mode) {
    const candidate_score best = encode_fast(w, h, mode == q2_opt_mode::signed_scale);
    if (!best.valid) {
        std::memset(&out, 0, sizeof(out));
        out.d = fp16_bits(0.0f);
        std::memset(out.qs, 0x55, sizeof(out.qs));
        return { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity() };
    }
    out = best.block;
    return best.metrics;
}

q2_opt_metrics score_q2_opt(
        const float * w,
        const float * h,
        q2_opt_mode mode) {
    const candidate_score best = encode_fast(
        w, h, mode == q2_opt_mode::signed_scale,
        /* positive_best = */ nullptr,
        /* materialize = */ false);
    if (!best.valid) {
        return { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
    }
    return best.metrics;
}

q2_opt_metrics encode_q2_opt_signed_compare(
        const float * w,
        const float * h,
        q2_opt_block & out,
        q2_opt_metrics & positive_metrics) {
    candidate_score positive;
    const candidate_score best = encode_fast(w, h, true, &positive);
    if (!best.valid || !positive.valid) {
        std::memset(&out, 0, sizeof(out));
        out.d = fp16_bits(0.0f);
        std::memset(out.qs, 0x55, sizeof(out.qs));
        positive_metrics = { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity() };
        return positive_metrics;
    }
    out = best.block;
    positive_metrics = positive.metrics;
    return best.metrics;
}

q2_opt_metrics encode_q2_stock(const float * w, const float * h, q2_opt_block & out) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q2_0);
    if (!tt || !tt->from_float_ref || !tt->to_float || tt->type_size != sizeof(q2_opt_block) ||
        tt->blck_size != Q2_OPT_BLOCK_SIZE) {
        throw std::runtime_error("stock GGML_TYPE_Q2_0 traits unavailable or layout mismatch");
    }

    tt->from_float_ref(w, &out, Q2_OPT_BLOCK_SIZE);

    float decoded[Q2_OPT_BLOCK_SIZE];
    tt->to_float(&out, decoded, Q2_OPT_BLOCK_SIZE);
    q2_opt_metrics metrics {};
    for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
        const double diff = (double) w[i] - decoded[i];
        metrics.weighted_sse += (double) h[i] * diff * diff;
        metrics.unweighted_sse += diff * diff;
        metrics.weighted_power += (double) h[i] * w[i] * w[i];
    }
    return metrics;
}

void decode_q2_opt(const q2_opt_block & block, float * out64) {
    const float d = fp16_value(block.d);
    for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
        const uint8_t code = (block.qs[i / 4] >> (2 * (i % 4))) & 3u;
        out64[i] = ((int) code - 1) * d;
    }
}

q2_opt_metrics encode_q2_oracle(const float * w, const float * h, q2_opt_block & out, bool allow_negative) {
    candidate_score best;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t hb = (uint16_t) bits;
        if (!finite_half(hb)) continue;
        const float d = fp16_value(hb);
        if (!allow_negative && std::signbit(d) && d != 0.0f) continue;
        const candidate_score cur = score_scale_oracle(w, h, hb);
        if (better(cur, best)) best = cur;
    }
    out = best.block;
    for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
        best.metrics.weighted_power += (double) h[i] * w[i] * w[i];
    }
    return best.metrics;
}

bool q2_opt_block_equal(const q2_opt_block & a, const q2_opt_block & b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

} // namespace qwen35_prune
