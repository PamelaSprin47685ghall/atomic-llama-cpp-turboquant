#include "gdn-function-projection.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qwen35_prune;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    // A deterministic overdetermined linear function. The function-space
    // projection should recover it to numerical precision at tiny ridge.
    constexpr int n = 12;
    constexpr int d = 3;
    constexpr int o = 2;
    std::vector<float> x((size_t) n * d);
    std::vector<float> y((size_t) n * o);
    const float true_w[o][d] = {{1.5f, -0.25f, 0.75f}, {-0.5f, 2.0f, 0.125f}};
    for (int i = 0; i < n; ++i) {
        x[(size_t) i * d + 0] = 0.2f * i - 0.7f;
        x[(size_t) i * d + 1] = (float) ((i * 7) % 5) - 2.0f;
        x[(size_t) i * d + 2] = (float) ((i * i + 3) % 11) * 0.1f;
        for (int j = 0; j < o; ++j) {
            float v = 0.0f;
            for (int k = 0; k < d; ++k) v += x[(size_t) i * d + k] * true_w[j][k];
            y[(size_t) i * o + j] = v;
        }
    }

    const auto fit = gdn_fit_function_projection_dual(x, y, n, d, o, 1e-10, 2);
    require(fit.residual_energy / fit.target_energy < 1e-12, "exact function projection residual too large");
    for (int j = 0; j < o; ++j) {
        for (int k = 0; k < d; ++k) {
            require(std::abs(fit.weight[(size_t) j * d + k] - true_w[j][k]) < 2e-4f,
                    "function projection failed to recover linear map");
        }
    }

    const auto score = gdn_score_function_projection(x, y, fit.weight, n, d, o, 2);
    require(score.residual_energy / score.target_energy < 1e-10, "function projection score mismatch");

    const std::vector<double> ridge_path = {1e-2, 1e-6, 1e-4};
    const auto path = gdn_fit_function_projection_dual_path(
            x, y, n, x, y, n, d, o, ridge_path, 2);
    require(path.path.size() == ridge_path.size(), "function projection ridge path size mismatch");
    require(path.dual_path.size() == ridge_path.size(), "function projection dual ridge path size mismatch");
    require(path.best.ridge_relative == 1e-6, "function projection ridge path did not select validation-best ridge");
    require(path.validation_residual_energy / path.validation_target_energy < 1e-8,
            "function projection ridge path residual too large");
    const auto path_w = gdn_materialize_function_projection_weight(
            x, path.dual_path[1], n, d, o, 2);
    const auto path_score = gdn_score_function_projection(x, y, path_w, n, d, o, 2);
    require(path_score.residual_energy / path_score.target_energy < 1e-8,
            "function projection dual materialization mismatch");

    // Orthogonal feature reparameterization must preserve the represented
    // function (and isotropic ridge penalty).
    std::vector<float> xr = x;
    const float c = 0.6f;
    const float s = 0.8f;
    for (int i = 0; i < n; ++i) {
        const float a = x[(size_t) i * d + 0];
        const float b = x[(size_t) i * d + 1];
        xr[(size_t) i * d + 0] = c * a - s * b;
        xr[(size_t) i * d + 1] = s * a + c * b;
    }
    const auto rotated = gdn_fit_function_projection_dual(xr, y, n, d, o, 1e-6, 2);
    require(std::abs(rotated.residual_energy / rotated.target_energy -
                     gdn_fit_function_projection_dual(x, y, n, d, o, 1e-6, 2).residual_energy /
                             fit.target_energy) < 1e-7,
            "orthogonal feature gauge changed function projection risk");

    std::cout << "gdn-function-projection: OK\n";
    return 0;
}
