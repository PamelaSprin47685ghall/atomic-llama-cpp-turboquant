#include "gdn-function-projection.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace qwen35_prune {
namespace {

static bool cholesky_inplace(std::vector<double> & a, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = a[(size_t) i * n + j];
            for (int k = 0; k < j; ++k) {
                sum -= a[(size_t) i * n + k] * a[(size_t) j * n + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !std::isfinite(sum)) return false;
                a[(size_t) i * n + j] = std::sqrt(sum);
            } else {
                a[(size_t) i * n + j] = sum / a[(size_t) j * n + j];
            }
        }
        for (int j = i + 1; j < n; ++j) a[(size_t) i * n + j] = 0.0;
    }
    return true;
}

static void cholesky_solve(
        const std::vector<double> & l,
        const double * rhs,
        double * solution,
        int n) {
    for (int i = 0; i < n; ++i) {
        double v = rhs[i];
        for (int j = 0; j < i; ++j) v -= l[(size_t) i * n + j] * solution[j];
        solution[i] = v / l[(size_t) i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double v = solution[i];
        for (int j = i + 1; j < n; ++j) v -= l[(size_t) j * n + i] * solution[j];
        solution[i] = v / l[(size_t) i * n + i];
    }
}

static void validate_shape(
        const std::vector<float> & features,
        const std::vector<float> & targets,
        int n_samples,
        int n_features,
        int n_outputs) {
    if (n_samples <= 0 || n_features <= 0 || n_outputs <= 0) {
        throw std::runtime_error("function projection dimensions must be positive");
    }
    if (features.size() != (size_t) n_samples * n_features ||
        targets.size() != (size_t) n_samples * n_outputs) {
        throw std::runtime_error("function projection input shape mismatch");
    }
}

static std::vector<double> make_self_kernel(
        const std::vector<float> & features,
        int n_samples,
        int n_features,
        int n_threads) {
    std::vector<double> kernel((size_t) n_samples * n_samples, 0.0);
    const int workers = std::max(1, std::min(n_threads, n_samples));
    std::atomic<int> next_i {0};
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&]() {
            for (;;) {
                const int i = next_i.fetch_add(1);
                if (i >= n_samples) break;
                const float * xi = features.data() + (size_t) i * n_features;
                for (int j = 0; j <= i; ++j) {
                    const float * xj = features.data() + (size_t) j * n_features;
                    long double dot = 0.0;
                    for (int d = 0; d < n_features; ++d) dot += (long double) xi[d] * xj[d];
                    const double value = (double) dot;
                    kernel[(size_t) i * n_samples + j] = value;
                    kernel[(size_t) j * n_samples + i] = value;
                }
            }
        });
    }
    for (auto & th : threads) th.join();
    return kernel;
}

static std::vector<double> make_cross_kernel(
        const std::vector<float> & lhs,
        int n_lhs,
        const std::vector<float> & rhs,
        int n_rhs,
        int n_features,
        int n_threads) {
    std::vector<double> kernel((size_t) n_lhs * n_rhs, 0.0);
    const int workers = std::max(1, std::min(n_threads, n_lhs));
    std::atomic<int> next_i {0};
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&]() {
            for (;;) {
                const int i = next_i.fetch_add(1);
                if (i >= n_lhs) break;
                const float * xi = lhs.data() + (size_t) i * n_features;
                double * row = kernel.data() + (size_t) i * n_rhs;
                for (int j = 0; j < n_rhs; ++j) {
                    const float * xj = rhs.data() + (size_t) j * n_features;
                    long double dot = 0.0;
                    for (int d = 0; d < n_features; ++d) dot += (long double) xi[d] * xj[d];
                    row[j] = (double) dot;
                }
            }
        });
    }
    for (auto & th : threads) th.join();
    return kernel;
}

static double vector_energy(const std::vector<float> & values) {
    long double energy = 0.0;
    for (float v : values) energy += (long double) v * v;
    return (double) energy;
}

} // namespace

gdn_function_projection_fit gdn_fit_function_projection_dual(
        const std::vector<float> & features,
        const std::vector<float> & targets,
        int n_samples,
        int n_features,
        int n_outputs,
        double ridge_relative,
        int n_threads) {
    validate_shape(features, targets, n_samples, n_features, n_outputs);
    if (!(ridge_relative > 0.0) || !std::isfinite(ridge_relative)) {
        throw std::runtime_error("function projection ridge must be finite and positive");
    }

    std::vector<double> kernel = make_self_kernel(features, n_samples, n_features, n_threads);
    long double feature_energy = 0.0;
    for (float x : features) feature_energy += (long double) x * x;
    const double ridge_absolute = std::max(
            1e-18,
            ridge_relative * (double) feature_energy / n_features);

    std::vector<double> chol = kernel;
    for (int i = 0; i < n_samples; ++i) chol[(size_t) i * n_samples + i] += ridge_absolute;
    if (!cholesky_inplace(chol, n_samples)) {
        throw std::runtime_error("function projection Cholesky failed");
    }

    gdn_function_projection_fit result;
    result.n_samples = n_samples;
    result.n_features = n_features;
    result.n_outputs = n_outputs;
    result.ridge_relative = ridge_relative;
    result.ridge_absolute = ridge_absolute;
    result.weight.resize((size_t) n_outputs * n_features);

    long double target_energy = 0.0;
    for (float y : targets) target_energy += (long double) y * y;
    std::atomic<int> next_output {0};
    std::mutex energy_mutex;
    long double residual_energy = 0.0;
    const int workers = std::max(1, std::min(n_threads, n_outputs));
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&]() {
            std::vector<double> rhs((size_t) n_samples);
            std::vector<double> alpha((size_t) n_samples);
            long double local_residual = 0.0;
            for (;;) {
                const int o = next_output.fetch_add(1);
                if (o >= n_outputs) break;
                for (int n = 0; n < n_samples; ++n) {
                    rhs[(size_t) n] = targets[(size_t) n * n_outputs + o];
                    alpha[(size_t) n] = 0.0;
                }
                cholesky_solve(chol, rhs.data(), alpha.data(), n_samples);

                float * wo = result.weight.data() + (size_t) o * n_features;
                for (int d = 0; d < n_features; ++d) {
                    long double value = 0.0;
                    for (int n = 0; n < n_samples; ++n) {
                        value += (long double) features[(size_t) n * n_features + d] * alpha[(size_t) n];
                    }
                    wo[d] = (float) value;
                }

                for (int n = 0; n < n_samples; ++n) {
                    long double pred = 0.0;
                    const double * kn = kernel.data() + (size_t) n * n_samples;
                    for (int j = 0; j < n_samples; ++j) pred += (long double) kn[j] * alpha[(size_t) j];
                    const long double e = (long double) targets[(size_t) n * n_outputs + o] - pred;
                    local_residual += e * e;
                }
            }
            std::lock_guard<std::mutex> lock(energy_mutex);
            residual_energy += local_residual;
        });
    }
    for (auto & th : threads) th.join();

    result.target_energy = (double) target_energy;
    result.residual_energy = (double) residual_energy;
    return result;
}

gdn_function_projection_score gdn_score_function_projection(
        const std::vector<float> & features,
        const std::vector<float> & targets,
        const std::vector<float> & weight,
        int n_samples,
        int n_features,
        int n_outputs,
        int n_threads) {
    validate_shape(features, targets, n_samples, n_features, n_outputs);
    if (weight.size() != (size_t) n_outputs * n_features) {
        throw std::runtime_error("function projection weight shape mismatch");
    }

    long double target_energy = 0.0;
    for (float y : targets) target_energy += (long double) y * y;

    std::atomic<int> next_output {0};
    std::mutex energy_mutex;
    long double residual_energy = 0.0;
    const int workers = std::max(1, std::min(n_threads, n_outputs));
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&]() {
            long double local_residual = 0.0;
            for (;;) {
                const int o = next_output.fetch_add(1);
                if (o >= n_outputs) break;
                const float * wo = weight.data() + (size_t) o * n_features;
                for (int n = 0; n < n_samples; ++n) {
                    const float * x = features.data() + (size_t) n * n_features;
                    long double pred = 0.0;
                    for (int d = 0; d < n_features; ++d) pred += (long double) x[d] * wo[d];
                    const long double e = (long double) targets[(size_t) n * n_outputs + o] - pred;
                    local_residual += e * e;
                }
            }
            std::lock_guard<std::mutex> lock(energy_mutex);
            residual_energy += local_residual;
        });
    }
    for (auto & th : threads) th.join();

    return {(double) target_energy, (double) residual_energy};
}

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
        int n_threads) {
    validate_shape(train_features, train_targets, n_train, n_features, n_outputs);
    validate_shape(validation_features, validation_targets, n_validation, n_features, n_outputs);
    if (ridge_relative_path.empty()) throw std::runtime_error("function projection ridge path is empty");
    for (double ridge : ridge_relative_path) {
        if (!(ridge > 0.0) || !std::isfinite(ridge)) {
            throw std::runtime_error("function projection ridge path contains an invalid value");
        }
    }

    const std::vector<double> train_kernel = make_self_kernel(train_features, n_train, n_features, n_threads);
    const std::vector<double> validation_kernel = make_cross_kernel(
            validation_features, n_validation, train_features, n_train, n_features, n_threads);
    const double feature_energy = vector_energy(train_features);
    const double train_target_energy = vector_energy(train_targets);
    const double validation_target_energy = vector_energy(validation_targets);

    gdn_function_projection_path_fit result;
    result.validation_target_energy = validation_target_energy;
    result.dual_path.reserve(ridge_relative_path.size());
    double best_validation = std::numeric_limits<double>::infinity();
    double best_ridge_relative = 0.0;
    double best_ridge_absolute = 0.0;

    // Keep one alpha matrix [n_outputs, n_train] for the winning lambda. This
    // is far smaller than materializing a full [n_outputs, n_features] W for
    // every candidate when calibration is sample-limited.
    std::vector<double> best_alpha;
    for (double ridge_relative : ridge_relative_path) {
        const double ridge_absolute = std::max(1e-18, ridge_relative * feature_energy / n_features);
        std::vector<double> chol = train_kernel;
        for (int i = 0; i < n_train; ++i) chol[(size_t) i * n_train + i] += ridge_absolute;
        if (!cholesky_inplace(chol, n_train)) {
            throw std::runtime_error("function projection path Cholesky failed");
        }

        std::vector<double> alpha((size_t) n_outputs * n_train);
        std::atomic<int> next_output {0};
        std::mutex energy_mutex;
        long double train_residual = 0.0;
        long double validation_residual = 0.0;
        const int workers = std::max(1, std::min(n_threads, n_outputs));
        std::vector<std::thread> threads;
        threads.reserve((size_t) workers);
        for (int w = 0; w < workers; ++w) {
            threads.emplace_back([&]() {
                std::vector<double> rhs((size_t) n_train);
                std::vector<double> solution((size_t) n_train);
                long double local_train = 0.0;
                long double local_validation = 0.0;
                for (;;) {
                    const int o = next_output.fetch_add(1);
                    if (o >= n_outputs) break;
                    for (int n = 0; n < n_train; ++n) {
                        rhs[(size_t) n] = train_targets[(size_t) n * n_outputs + o];
                        solution[(size_t) n] = 0.0;
                    }
                    cholesky_solve(chol, rhs.data(), solution.data(), n_train);
                    std::copy(solution.begin(), solution.end(), alpha.begin() + (size_t) o * n_train);

                    for (int n = 0; n < n_train; ++n) {
                        const double * row = train_kernel.data() + (size_t) n * n_train;
                        long double pred = 0.0;
                        for (int j = 0; j < n_train; ++j) pred += (long double) row[j] * solution[(size_t) j];
                        const long double e = (long double) train_targets[(size_t) n * n_outputs + o] - pred;
                        local_train += e * e;
                    }
                    for (int n = 0; n < n_validation; ++n) {
                        const double * row = validation_kernel.data() + (size_t) n * n_train;
                        long double pred = 0.0;
                        for (int j = 0; j < n_train; ++j) pred += (long double) row[j] * solution[(size_t) j];
                        const long double e = (long double) validation_targets[(size_t) n * n_outputs + o] - pred;
                        local_validation += e * e;
                    }
                }
                std::lock_guard<std::mutex> lock(energy_mutex);
                train_residual += local_train;
                validation_residual += local_validation;
            });
        }
        for (auto & th : threads) th.join();

        gdn_function_projection_path_point point;
        point.ridge_relative = ridge_relative;
        point.ridge_absolute = ridge_absolute;
        point.train_residual_ratio = train_target_energy > 0.0 ? (double) train_residual / train_target_energy : 0.0;
        point.validation_residual_ratio = validation_target_energy > 0.0 ?
                (double) validation_residual / validation_target_energy : 0.0;
        result.path.push_back(point);
        result.dual_path.push_back(alpha);

        if ((double) validation_residual < best_validation) {
            best_validation = (double) validation_residual;
            best_ridge_relative = ridge_relative;
            best_ridge_absolute = ridge_absolute;
            best_alpha = alpha;
            result.validation_residual_energy = best_validation;
            result.best.residual_energy = (double) train_residual;
        }
    }

    if (best_alpha.empty()) throw std::runtime_error("function projection path failed to select a ridge");
    result.best.n_samples = n_train;
    result.best.n_features = n_features;
    result.best.n_outputs = n_outputs;
    result.best.ridge_relative = best_ridge_relative;
    result.best.ridge_absolute = best_ridge_absolute;
    result.best.target_energy = train_target_energy;
    result.best.weight = gdn_materialize_function_projection_weight(
            train_features, best_alpha, n_train, n_features, n_outputs, n_threads);
    return result;
}

std::vector<float> gdn_materialize_function_projection_weight(
        const std::vector<float> & train_features,
        const std::vector<double> & dual_coefficients,
        int n_train,
        int n_features,
        int n_outputs,
        int n_threads) {
    if (n_train <= 0 || n_features <= 0 || n_outputs <= 0 ||
        train_features.size() != (size_t) n_train * n_features ||
        dual_coefficients.size() != (size_t) n_outputs * n_train) {
        throw std::runtime_error("function projection materialization shape mismatch");
    }
    std::vector<float> weight((size_t) n_outputs * n_features);
    std::atomic<int> next_output {0};
    const int workers = std::max(1, std::min(n_threads, n_outputs));
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&]() {
            for (;;) {
                const int o = next_output.fetch_add(1);
                if (o >= n_outputs) break;
                const double * alpha = dual_coefficients.data() + (size_t) o * n_train;
                float * wo = weight.data() + (size_t) o * n_features;
                for (int d = 0; d < n_features; ++d) {
                    long double value = 0.0;
                    for (int n = 0; n < n_train; ++n) {
                        value += (long double) train_features[(size_t) n * n_features + d] * alpha[n];
                    }
                    wo[d] = (float) value;
                }
            }
        });
    }
    for (auto & th : threads) th.join();
    return weight;
}

} // namespace qwen35_prune
