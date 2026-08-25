#include "expert-replacement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace qwen35_prune {
namespace {

struct token_eval {
    double loss = 0.0;
    int coverage = 0;
};

static void validate_problem(const expert_replacement_problem & p) {
    if (p.n_expert <= 0 || p.n_keep <= 0 || p.n_keep >= p.n_expert ||
        p.top_k <= 0 || p.candidate_depth < p.top_k ||
        p.n_tokens <= 0 || p.output_dim <= 0) {
        throw std::runtime_error("invalid expert-replacement problem dimensions");
    }
    if (p.candidate_depth < p.top_k + (p.n_expert - p.n_keep)) {
        throw std::runtime_error("expert-replacement candidate depth is insufficient for exact fixed-count Top-K reconstruction");
    }
    const size_t nc = (size_t) p.n_tokens * p.candidate_depth;
    if (p.candidate_ids.size() != nc || p.candidate_logits.size() != nc ||
        p.outputs.size() != nc * p.output_dim) {
        throw std::runtime_error("malformed expert-replacement problem payload");
    }
    for (int t = 0; t < p.n_tokens; ++t) {
        std::vector<uint8_t> seen((size_t) p.n_expert, 0);
        float prev = std::numeric_limits<float>::infinity();
        for (int r = 0; r < p.candidate_depth; ++r) {
            const size_t q = (size_t) t * p.candidate_depth + r;
            const int id = p.candidate_ids[q];
            const float logit = p.candidate_logits[q];
            if (id < 0 || id >= p.n_expert || seen[(size_t) id]) {
                throw std::runtime_error("invalid/duplicate expert candidate id");
            }
            if (!std::isfinite(logit) || logit > prev) {
                throw std::runtime_error("expert candidates are not sorted by finite router logits");
            }
            seen[(size_t) id] = 1;
            prev = logit;
        }
    }
}

static void select_ranks(
        const expert_replacement_problem & p,
        int token,
        const std::vector<uint8_t> & survivor,
        int * ranks,
        int & coverage) {
    int nsel = 0;
    coverage = 0;
    const size_t base = (size_t) token * p.candidate_depth;
    for (int r = 0; r < p.candidate_depth; ++r) {
        const int id = p.candidate_ids[base + r];
        if (!survivor[(size_t) id]) continue;
        ++coverage;
        if (nsel < p.top_k) ranks[nsel++] = r;
    }
    if (nsel != p.top_k) {
        throw std::runtime_error("expert-replacement candidate coverage fell below top-k");
    }
}

static void mixture(
        const expert_replacement_problem & p,
        int token,
        const int * ranks,
        float * out) {
    std::fill(out, out + p.output_dim, 0.0f);
    const size_t base = (size_t) token * p.candidate_depth;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < p.top_k; ++i) {
        max_logit = std::max(max_logit, p.candidate_logits[base + ranks[i]]);
    }
    double z = 0.0;
    double w[64];
    if (p.top_k > 64) throw std::runtime_error("expert-replacement top-k too large");
    for (int i = 0; i < p.top_k; ++i) {
        w[i] = std::exp((double) p.candidate_logits[base + ranks[i]] - max_logit);
        z += w[i];
    }
    for (int i = 0; i < p.top_k; ++i) {
        const float alpha = (float) (w[i] / z);
        const float * src = p.outputs.data() + (base + ranks[i]) * p.output_dim;
        for (int d = 0; d < p.output_dim; ++d) out[d] += alpha * src[d];
    }
}

static std::vector<float> baseline_mixtures(const expert_replacement_problem & p) {
    std::vector<float> baseline((size_t) p.n_tokens * p.output_dim);
    int ranks[64];
    for (int i = 0; i < p.top_k; ++i) ranks[i] = i;
    for (int t = 0; t < p.n_tokens; ++t) {
        float * dst = baseline.data() + (size_t) t * p.output_dim;
        mixture(p, t, ranks, dst);
    }
    return baseline;
}

static token_eval eval_token(
        const expert_replacement_problem & p,
        int token,
        const std::vector<uint8_t> & survivor,
        const std::vector<float> & baseline,
        std::vector<float> & scratch) {
    int ranks[64];
    if (p.top_k > 64) throw std::runtime_error("expert-replacement top-k too large");
    int coverage = 0;
    select_ranks(p, token, survivor, ranks, coverage);
    mixture(p, token, ranks, scratch.data());
    const float * ref = baseline.data() + (size_t) token * p.output_dim;
    double err2 = 0.0;
    for (int d = 0; d < p.output_dim; ++d) {
        const double e = (double) scratch[(size_t) d] - ref[d];
        err2 += e * e;
    }
    return {err2, coverage};
}

static double objective_with_cache(
        const expert_replacement_problem & p,
        const std::vector<uint8_t> & survivor,
        const std::vector<float> & baseline,
        int * min_coverage) {
    double sum = 0.0;
    int min_cov = p.candidate_depth;
    std::vector<float> scratch((size_t) p.output_dim);
    for (int t = 0; t < p.n_tokens; ++t) {
        const auto e = eval_token(p, t, survivor, baseline, scratch);
        sum += e.loss;
        min_cov = std::min(min_cov, e.coverage);
    }
    if (min_coverage) *min_coverage = min_cov;
    return sum;
}

static bool feasible_after_delete(
        const expert_replacement_problem & p,
        const std::vector<uint8_t> & survivor,
        int expert,
        const std::vector<int> & coverage) {
    for (int t = 0; t < p.n_tokens; ++t) {
        if (coverage[(size_t) t] > p.top_k) continue;
        const size_t base = (size_t) t * p.candidate_depth;
        for (int r = 0; r < p.candidate_depth; ++r) {
            if (p.candidate_ids[base + r] == expert && survivor[(size_t) expert]) return false;
        }
    }
    return true;
}

} // namespace

double evaluate_expert_replacement_objective(
        const expert_replacement_problem & p,
        const std::vector<uint8_t> & survivor_mask,
        int * minimum_candidate_coverage) {
    validate_problem(p);
    if (survivor_mask.size() != (size_t) p.n_expert) throw std::runtime_error("bad survivor mask size");
    const auto baseline = baseline_mixtures(p);
    return objective_with_cache(p, survivor_mask, baseline, minimum_candidate_coverage);
}

expert_replacement_result optimize_expert_replacement(
        const expert_replacement_problem & p,
        int max_swap_steps,
        double rel_tol) {
    validate_problem(p);
    if (max_swap_steps < 0 || rel_tol < 0.0) throw std::runtime_error("invalid expert-replacement optimizer options");

    const auto baseline = baseline_mixtures(p);
    std::vector<uint8_t> survivor((size_t) p.n_expert, 1);
    std::vector<int> coverage((size_t) p.n_tokens, p.candidate_depth);
    std::vector<double> token_loss((size_t) p.n_tokens, 0.0);
    std::vector<int64_t> usage((size_t) p.n_expert, 0);
    for (int t = 0; t < p.n_tokens; ++t) {
        const size_t base = (size_t) t * p.candidate_depth;
        for (int r = 0; r < p.top_k; ++r) ++usage[(size_t) p.candidate_ids[base + r]];
    }

    // Static backup value only breaks exact marginal ties.  It favors keeping
    // candidates close to the Top-8 boundary, preventing the reverse-greedy
    // path from arbitrarily exhausting useful backups before they become active.
    std::vector<double> reserve((size_t) p.n_expert, 0.0);
    for (int t = 0; t < p.n_tokens; ++t) {
        const size_t base = (size_t) t * p.candidate_depth;
        const double boundary = p.candidate_logits[base + p.top_k - 1];
        for (int r = p.top_k; r < p.candidate_depth; ++r) {
            const int id = p.candidate_ids[base + r];
            reserve[(size_t) id] += std::exp((double) p.candidate_logits[base + r] - boundary) /
                                   (double) (1 + r - p.top_k);
        }
    }

    double objective = 0.0;
    std::vector<float> scratch((size_t) p.output_dim);
    std::vector<int> selected_tokens;
    const int remove_count = p.n_expert - p.n_keep;
    for (int step = 0; step < remove_count; ++step) {
        int best = -1;
        double best_delta = std::numeric_limits<double>::infinity();
        double best_reserve = std::numeric_limits<double>::infinity();

        for (int e = 0; e < p.n_expert; ++e) {
            if (!survivor[(size_t) e] || !feasible_after_delete(p, survivor, e, coverage)) continue;
            double delta = 0.0;
            survivor[(size_t) e] = 0;
            for (int t = 0; t < p.n_tokens; ++t) {
                bool selected = false;
                const size_t base = (size_t) t * p.candidate_depth;
                int alive_seen = 0;
                for (int r = 0; r < p.candidate_depth && alive_seen < p.top_k; ++r) {
                    const int id = p.candidate_ids[base + r];
                    if (id == e) continue;
                    if (survivor[(size_t) id]) ++alive_seen;
                }
                // If e was not in the old selected set, this token is unchanged.
                int old_alive_seen = 0;
                for (int r = 0; r < p.candidate_depth && old_alive_seen < p.top_k; ++r) {
                    const int id = p.candidate_ids[base + r];
                    if (id == e) { selected = true; break; }
                    if (survivor[(size_t) id]) ++old_alive_seen;
                }
                if (!selected) continue;
                const auto ev = eval_token(p, t, survivor, baseline, scratch);
                delta += ev.loss - token_loss[(size_t) t];
            }
            survivor[(size_t) e] = 1;

            const double scale = std::max(1.0, std::fabs(objective));
            const double tie_eps = 1e-13 * scale;
            if (delta < best_delta - tie_eps ||
                (std::fabs(delta - best_delta) <= tie_eps &&
                 (reserve[(size_t) e] < best_reserve ||
                  (reserve[(size_t) e] == best_reserve && e < best)))) {
                best = e;
                best_delta = delta;
                best_reserve = reserve[(size_t) e];
            }
        }
        if (best < 0) throw std::runtime_error("expert-replacement coverage constraint made target expert count infeasible");

        survivor[(size_t) best] = 0;
        objective = 0.0;
        for (int t = 0; t < p.n_tokens; ++t) {
            const size_t base = (size_t) t * p.candidate_depth;
            for (int r = 0; r < p.candidate_depth; ++r) {
                if (p.candidate_ids[base + r] == best) {
                    --coverage[(size_t) t];
                    break;
                }
            }
            const auto ev = eval_token(p, t, survivor, baseline, scratch);
            token_loss[(size_t) t] = ev.loss;
            objective += ev.loss;
        }
    }

    int swap_steps = 0;
    for (; swap_steps < max_swap_steps; ++swap_steps) {
        int best_remove = -1;
        int best_add = -1;
        double best_delta = 0.0;
        const double improve_tol = rel_tol * std::max(1.0, std::fabs(objective));

        for (int r = 0; r < p.n_expert; ++r) {
            if (!survivor[(size_t) r]) continue;
            for (int a = 0; a < p.n_expert; ++a) {
                if (survivor[(size_t) a]) continue;
                bool feasible = true;
                for (int t = 0; t < p.n_tokens && feasible; ++t) {
                    const size_t base = (size_t) t * p.candidate_depth;
                    bool has_r = false, has_a = false;
                    for (int q = 0; q < p.candidate_depth; ++q) {
                        const int id = p.candidate_ids[base + q];
                        has_r |= id == r;
                        has_a |= id == a;
                    }
                    if (coverage[(size_t) t] - (has_r ? 1 : 0) + (has_a ? 1 : 0) < p.top_k) feasible = false;
                }
                if (!feasible) continue;

                survivor[(size_t) r] = 0;
                survivor[(size_t) a] = 1;
                double delta = 0.0;
                for (int t = 0; t < p.n_tokens; ++t) {
                    const size_t base = (size_t) t * p.candidate_depth;
                    bool touches = false;
                    for (int q = 0; q < p.candidate_depth; ++q) {
                        const int id = p.candidate_ids[base + q];
                        if (id == r || id == a) { touches = true; break; }
                    }
                    if (!touches) continue;
                    const auto ev = eval_token(p, t, survivor, baseline, scratch);
                    delta += ev.loss - token_loss[(size_t) t];
                }
                survivor[(size_t) a] = 0;
                survivor[(size_t) r] = 1;
                if (delta < best_delta - improve_tol ||
                    (std::fabs(delta - best_delta) <= improve_tol && delta < 0.0 &&
                     (best_remove < 0 || std::pair<int,int>{r,a} < std::pair<int,int>{best_remove,best_add}))) {
                    best_delta = delta;
                    best_remove = r;
                    best_add = a;
                }
            }
        }
        if (best_remove < 0) break;

        survivor[(size_t) best_remove] = 0;
        survivor[(size_t) best_add] = 1;
        objective = 0.0;
        for (int t = 0; t < p.n_tokens; ++t) {
            const size_t base = (size_t) t * p.candidate_depth;
            bool has_r = false, has_a = false;
            for (int q = 0; q < p.candidate_depth; ++q) {
                const int id = p.candidate_ids[base + q];
                has_r |= id == best_remove;
                has_a |= id == best_add;
            }
            coverage[(size_t) t] += (has_a ? 1 : 0) - (has_r ? 1 : 0);
            const auto ev = eval_token(p, t, survivor, baseline, scratch);
            token_loss[(size_t) t] = ev.loss;
            objective += ev.loss;
        }
    }

    expert_replacement_result out;
    out.baseline_topk_count = std::move(usage);
    out.objective = objective;
    out.objective_mean = objective / p.n_tokens;
    out.greedy_steps = remove_count;
    out.swap_steps = swap_steps;
    out.minimum_candidate_coverage = *std::min_element(coverage.begin(), coverage.end());
    for (int e = 0; e < p.n_expert; ++e) {
        (survivor[(size_t) e] ? out.survivors : out.removed).push_back(e);
    }
    if ((int) out.survivors.size() != p.n_keep || out.minimum_candidate_coverage < p.top_k) {
        throw std::runtime_error("expert-replacement optimizer violated fixed-size/coverage contract");
    }
    return out;
}

} // namespace qwen35_prune
