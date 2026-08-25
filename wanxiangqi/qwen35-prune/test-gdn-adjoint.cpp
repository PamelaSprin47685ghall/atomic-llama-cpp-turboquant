#include "gdn-adjoint.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace qwen35_prune;

static double objective(
        const std::vector<double> & q,
        const std::vector<double> & k,
        const std::vector<double> & v,
        const std::vector<double> & g,
        const std::vector<double> & beta,
        const std::vector<double> & state,
        const std::vector<double> & dy,
        const std::vector<double> & ds,
        int n_tokens,
        int dim) {
    const auto f = gdn_scalar_gate_forward_ref(q, k, v, g, beta, state, n_tokens, dim);
    double out = 0.0;
    for (size_t i = 0; i < f.output.size(); ++i) out += f.output[i] * dy[i];
    for (size_t i = 0; i < f.final_state.size(); ++i) out += f.final_state[i] * ds[i];
    return out;
}

static void check_vector(
        const char * name,
        std::vector<double> & x,
        const std::vector<double> & analytic,
        const std::function<double()> & eval) {
    const double eps = 1e-6;
    double worst_rel = 0.0;
    double worst_abs = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double old = x[i];
        x[i] = old + eps;
        const double plus = eval();
        x[i] = old - eps;
        const double minus = eval();
        x[i] = old;
        const double numeric = (plus - minus) / (2.0 * eps);
        const double abs_err = std::fabs(numeric - analytic[i]);
        const double rel = abs_err / std::max({1.0, std::fabs(numeric), std::fabs(analytic[i])});
        if (rel > worst_rel) {
            worst_rel = rel;
            worst_abs = abs_err;
            worst_i = i;
        }
    }
    if (worst_rel > 2e-7 && worst_abs > 2e-8) {
        throw std::runtime_error(std::string(name) + " finite-difference mismatch at " +
                                 std::to_string(worst_i) + ", rel=" + std::to_string(worst_rel) +
                                 ", abs=" + std::to_string(worst_abs));
    }
    std::cout << name << " worst_rel=" << worst_rel << " worst_abs=" << worst_abs << "\n";
}

int main() {
    constexpr int T = 4;
    constexpr int D = 3;
    std::mt19937 rng(123456);
    std::uniform_real_distribution<double> dist(-0.35, 0.35);
    auto random_vec = [&](size_t n) {
        std::vector<double> x(n);
        for (double & v : x) v = dist(rng);
        return x;
    };
    auto q = random_vec(T * D);
    auto k = random_vec(T * D);
    auto v = random_vec(T * D);
    auto g = random_vec(T);
    auto beta = random_vec(T);
    for (double & b : beta) b = 0.5 + b;
    auto state = random_vec(D * D);
    const auto dy = random_vec(T * D);
    const auto ds = random_vec(D * D);

    const auto grad = gdn_scalar_gate_backward_ref(q, k, v, g, beta, state, dy, ds, T, D);
    auto eval = [&]() { return objective(q, k, v, g, beta, state, dy, ds, T, D); };
    check_vector("q", q, grad.q, eval);
    check_vector("k", k, grad.k, eval);
    check_vector("v", v, grad.v, eval);
    check_vector("g", g, grad.g, eval);
    check_vector("beta", beta, grad.beta, eval);
    check_vector("state", state, grad.initial_state, eval);
    std::cout << "gdn-adjoint: OK\n";
    return 0;
}
