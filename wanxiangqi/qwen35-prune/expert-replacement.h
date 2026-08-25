#pragma once

#include <cstdint>
#include <vector>

namespace qwen35_prune {

struct expert_replacement_problem {
    int n_expert = 0;
    int n_keep = 0;
    int top_k = 0;
    int candidate_depth = 0;
    int n_tokens = 0;
    int output_dim = 0;

    // Per token, candidates are sorted by the original router logit in
    // descending order. outputs stores the downstream-mapped, unweighted
    // expert output for the same token/candidate pair.
    std::vector<int32_t> candidate_ids;   // [n_tokens, candidate_depth]
    std::vector<float> candidate_logits;  // [n_tokens, candidate_depth]
    std::vector<float> outputs;           // [n_tokens, candidate_depth, output_dim]
};

struct expert_replacement_result {
    std::vector<int> survivors;
    std::vector<int> removed;
    std::vector<int64_t> baseline_topk_count;
    double objective = 0.0;
    double objective_mean = 0.0;
    int minimum_candidate_coverage = 0;
    int greedy_steps = 0;
    int swap_steps = 0;
};

// Minimize the exact finite-sample replacement objective represented by p.
// When candidate_depth >= top_k + (n_expert-n_keep), the recorded candidates
// contain the exact post-prune Top-K for every possible fixed-size survivor
// set: at most n_expert-n_keep candidates can be deleted globally. Reverse
// greedy removes experts with the smallest current marginal loss and a full
// 1-for-1 swap search then refines the fixed-size set to a local optimum (up to
// rel_tol) under that exact finite-sample router objective.
expert_replacement_result optimize_expert_replacement(
        const expert_replacement_problem & p,
        int max_swap_steps = 16,
        double rel_tol = 1e-10);

// Exposed for diagnostics/tests. survivor_mask must have n_expert entries.
double evaluate_expert_replacement_objective(
        const expert_replacement_problem & p,
        const std::vector<uint8_t> & survivor_mask,
        int * minimum_candidate_coverage = nullptr);

} // namespace qwen35_prune
