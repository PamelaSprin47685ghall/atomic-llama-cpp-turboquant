#include "expert-replacement.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace qwen35_prune;

static expert_replacement_problem make_problem() {
    expert_replacement_problem p;
    p.n_expert = 6;
    p.n_keep = 4;
    p.top_k = 2;
    p.candidate_depth = 4;
    p.n_tokens = 4;
    p.output_dim = 2;
    // Experts 0 and 1 are very popular but functionally redundant. Expert 2
    // is less popular and unique. A frequency-only pruning rule would tend to
    // protect both 0/1; replacement-aware pruning should be willing to remove
    // one of them, but not both because the coverage/output objective updates.
    const int ids[4][4] = {
        {0,1,2,3}, {0,1,2,4}, {1,0,2,5}, {2,3,0,1},
    };
    const float logits[4][4] = {
        {4,3.9f,2,1}, {4,3.9f,2,1}, {4,3.9f,2,1}, {4,3,2,1},
    };
    for (int t = 0; t < 4; ++t) {
        for (int r = 0; r < 4; ++r) {
            p.candidate_ids.push_back(ids[t][r]);
            p.candidate_logits.push_back(logits[t][r]);
            const int e = ids[t][r];
            // 0 and 1 intentionally almost identical; 2 points in an
            // orthogonal direction and must be retained for token 3.
            if (e == 0) { p.outputs.push_back(1.0f); p.outputs.push_back(0.0f); }
            else if (e == 1) { p.outputs.push_back(1.001f); p.outputs.push_back(0.0f); }
            else if (e == 2) { p.outputs.push_back(0.0f); p.outputs.push_back(1.0f); }
            else { p.outputs.push_back(0.2f * e); p.outputs.push_back(0.1f * e); }
        }
    }
    return p;
}

int main() {
    auto p = make_problem();
    const auto r = optimize_expert_replacement(p, 8, 1e-12);
    assert(r.survivors.size() == 4);
    assert(r.removed.size() == 2);
    assert(r.minimum_candidate_coverage >= 2);
    const bool kept0 = std::find(r.survivors.begin(), r.survivors.end(), 0) != r.survivors.end();
    const bool kept1 = std::find(r.survivors.begin(), r.survivors.end(), 1) != r.survivors.end();
    assert(kept0 || kept1);
    assert(!(kept0 && kept1)); // one redundant popular expert is removable
    assert(std::find(r.survivors.begin(), r.survivors.end(), 2) != r.survivors.end());

    std::vector<uint8_t> mask(6, 0);
    for (int e : r.survivors) mask[(size_t) e] = 1;
    int min_cov = 0;
    const double obj = evaluate_expert_replacement_objective(p, mask, &min_cov);
    assert(std::fabs(obj - r.objective) < 1e-8);
    assert(min_cov == r.minimum_candidate_coverage);
    std::cout << "expert replacement optimizer OK\n";
    return 0;
}
