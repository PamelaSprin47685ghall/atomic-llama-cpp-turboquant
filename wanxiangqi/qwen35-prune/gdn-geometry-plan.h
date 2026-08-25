#pragma once

#include <cstdint>
#include <vector>

namespace qwen35_prune {

struct gdn_geometry_shrink_result {
    std::vector<double> matrix;
    double shrinkage = 0.0;
    double data_trace = 0.0;
    double prior_trace = 0.0;
};

// Build a PSD second-moment estimate from an empirical sum and a diagonal
// weight-space prior.  The prior is trace-matched to the empirical matrix and
// the shrinkage coefficient is n_dim / (n_samples + n_dim), so calibration
// can inform geometry without making an unregularized small-sample estimate.
gdn_geometry_shrink_result gdn_geometry_shrink_second_moment(
        const std::vector<double> & empirical_sum,
        uint64_t samples,
        const std::vector<double> & prior_diag,
        int dim);

// Deterministic pivoted-Cholesky subset selection.  At each step this chooses
// the coordinate with the largest residual conditional variance.  Ties are
// resolved by the smallest coordinate index.
std::vector<int> gdn_geometry_pivoted_cholesky_select(
        const std::vector<double> & psd,
        int dim,
        int rank);

// Conservative coordinate truncation baseline: keep the coordinates with the
// largest diagonal importance in a PSD/shrunk second-moment matrix.  Unlike
// pivoted Cholesky this does not assume omitted coordinates will be linearly
// reconstructed from the retained set.
std::vector<int> gdn_geometry_topk_diagonal_select(
        const std::vector<double> & psd,
        int dim,
        int rank);

double gdn_geometry_jaccard(const std::vector<int> & a, const std::vector<int> & b);

} // namespace qwen35_prune
