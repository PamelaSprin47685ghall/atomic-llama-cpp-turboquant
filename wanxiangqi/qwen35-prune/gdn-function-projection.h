#pragma once

#include <cstddef>
#include <vector>

namespace qwen35_prune {

struct gdn_function_projection_fit {
    int n_samples = 0;
    int n_features = 0;
    int n_outputs = 0;
    double ridge_relative = 0.0;
    double ridge_absolute = 0.0;
    double target_energy = 0.0;
    double residual_energy = 0.0;
    std::vector<float> weight;
};

struct gdn_function_projection_score {
    double target_energy = 0.0;
    double residual_energy = 0.0;
};

struct gdn_function_projection_path_point {
    double ridge_relative = 0.0;
    double ridge_absolute = 0.0;
    double train_residual_ratio = 0.0;
    double validation_residual_ratio = 0.0;
};

struct gdn_function_projection_path_fit {
    gdn_function_projection_fit best;
    double validation_target_energy = 0.0;
    double validation_residual_energy = 0.0;
    std::vector<gdn_function_projection_path_point> path;
    // One row-major [n_outputs, n_train] dual coefficient matrix per path
    // point. Retaining these is cheap when n_train << n_features and lets a
    // caller materialize candidate stock readouts for an outer/global
    // function-risk selector without re-forming the design kernels.
    std::vector<std::vector<double>> dual_path;
};

// Empirical L2 variable projection for a fixed nonlinear student feature map.
//
// features is row-major [n_samples, n_features], targets is row-major
// [n_samples, n_outputs]. The returned weight is row-major
// [n_outputs, n_features] and minimizes
//
//   ||Y - X W^T||_F^2 + lambda ||W||_F^2,
//
// with lambda = ridge_relative * trace(X^T X) / n_features. The solve is
// performed in sample space through X X^T, so this is an exact ridge solve
// rather than a headwise/block-coordinate approximation.
gdn_function_projection_fit gdn_fit_function_projection_dual(
        const std::vector<float> & features,
        const std::vector<float> & targets,
        int n_samples,
        int n_features,
        int n_outputs,
        double ridge_relative,
        int n_threads);

gdn_function_projection_score gdn_score_function_projection(
        const std::vector<float> & features,
        const std::vector<float> & targets,
        const std::vector<float> & weight,
        int n_samples,
        int n_features,
        int n_outputs,
        int n_threads);

// Validation-selected exact ridge path. The expensive feature kernels are
// formed once. Candidate lambdas are scored entirely in sample space using
// K_train = X_train X_train^T and K_valid,train = X_valid X_train^T. The
// primal readout W is materialized only for the validation-best candidate.
// This matters when n_samples << n_features, where tiny-ridge ERM can
// interpolate the calibration set while generalizing poorly.
gdn_function_projection_path_fit gdn_fit_function_projection_dual_path(
        const std::vector<float> & train_features,
        const std::vector<float> & train_targets,
        int n_train,
        const std::vector<float> & validation_features,
        const std::vector<float> & validation_targets,
        int n_validation,
        int n_features,
        int n_outputs,
        const std::vector<double> & ridge_relative_path,
        int n_threads);

std::vector<float> gdn_materialize_function_projection_weight(
        const std::vector<float> & train_features,
        const std::vector<double> & dual_coefficients,
        int n_train,
        int n_features,
        int n_outputs,
        int n_threads);

} // namespace qwen35_prune
