#include "gdn-adjoint.h"

#include <cmath>
#include <stdexcept>

namespace qwen35_prune {
namespace {

static void require_shape(
        const std::vector<double> & q,
        const std::vector<double> & k,
        const std::vector<double> & v,
        const std::vector<double> & g,
        const std::vector<double> & beta,
        const std::vector<double> & state,
        int n_tokens,
        int dim) {
    if (n_tokens <= 0 || dim <= 0 ||
        q.size() != (size_t) n_tokens * dim ||
        k.size() != q.size() || v.size() != q.size() ||
        g.size() != (size_t) n_tokens || beta.size() != (size_t) n_tokens ||
        state.size() != (size_t) dim * dim) {
        throw std::runtime_error("invalid scalar-gate GDN reference shape");
    }
}

} // namespace

gdn_scalar_gate_forward gdn_scalar_gate_forward_ref(
        const std::vector<double> & q,
        const std::vector<double> & k,
        const std::vector<double> & v,
        const std::vector<double> & g,
        const std::vector<double> & beta,
        const std::vector<double> & initial_state,
        int n_tokens,
        int dim) {
    require_shape(q, k, v, g, beta, initial_state, n_tokens, dim);
    gdn_scalar_gate_forward out;
    out.n_tokens = n_tokens;
    out.dim = dim;
    out.output.assign((size_t) n_tokens * dim, 0.0);
    out.final_state.resize((size_t) dim * dim);
    out.states.resize((size_t) (n_tokens + 1) * dim * dim);
    std::copy(initial_state.begin(), initial_state.end(), out.states.begin());
    const double scale = 1.0 / std::sqrt((double) dim);

    std::vector<double> s_bar((size_t) dim * dim);
    std::vector<double> u((size_t) dim);
    std::vector<double> delta((size_t) dim);
    for (int t = 0; t < n_tokens; ++t) {
        const double * s_prev = out.states.data() + (size_t) t * dim * dim;
        double * s = out.states.data() + (size_t) (t + 1) * dim * dim;
        const double * qt = q.data() + (size_t) t * dim;
        const double * kt = k.data() + (size_t) t * dim;
        const double * vt = v.data() + (size_t) t * dim;
        const double a = std::exp(g[(size_t) t]);
        for (int i = 0; i < dim * dim; ++i) s_bar[(size_t) i] = a * s_prev[i];
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (int i = 0; i < dim; ++i) sum += s_bar[(size_t) i * dim + j] * kt[i];
            u[(size_t) j] = sum;
            delta[(size_t) j] = beta[(size_t) t] * (vt[j] - sum);
        }
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                s[(size_t) i * dim + j] = s_bar[(size_t) i * dim + j] + kt[i] * delta[(size_t) j];
            }
        }
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (int i = 0; i < dim; ++i) sum += s[(size_t) i * dim + j] * qt[i];
            out.output[(size_t) t * dim + j] = scale * sum;
        }
    }
    const double * final = out.states.data() + (size_t) n_tokens * dim * dim;
    std::copy(final, final + (size_t) dim * dim, out.final_state.begin());
    return out;
}

gdn_scalar_gate_grad gdn_scalar_gate_backward_ref(
        const std::vector<double> & q,
        const std::vector<double> & k,
        const std::vector<double> & v,
        const std::vector<double> & g,
        const std::vector<double> & beta,
        const std::vector<double> & initial_state,
        const std::vector<double> & d_output,
        const std::vector<double> & d_final_state,
        int n_tokens,
        int dim) {
    require_shape(q, k, v, g, beta, initial_state, n_tokens, dim);
    if (d_output.size() != (size_t) n_tokens * dim ||
        d_final_state.size() != (size_t) dim * dim) {
        throw std::runtime_error("invalid scalar-gate GDN adjoint shape");
    }
    const auto fwd = gdn_scalar_gate_forward_ref(q, k, v, g, beta, initial_state, n_tokens, dim);
    const double scale = 1.0 / std::sqrt((double) dim);

    gdn_scalar_gate_grad grad;
    grad.q.assign(q.size(), 0.0);
    grad.k.assign(k.size(), 0.0);
    grad.v.assign(v.size(), 0.0);
    grad.g.assign(g.size(), 0.0);
    grad.beta.assign(beta.size(), 0.0);
    grad.initial_state.assign(initial_state.size(), 0.0);

    std::vector<double> bar_s = d_final_state;
    std::vector<double> bar_s_bar((size_t) dim * dim);
    std::vector<double> s_bar((size_t) dim * dim);
    std::vector<double> u((size_t) dim);
    std::vector<double> delta((size_t) dim);
    std::vector<double> bar_delta((size_t) dim);
    std::vector<double> bar_u((size_t) dim);

    for (int t = n_tokens - 1; t >= 0; --t) {
        const double * s_prev = fwd.states.data() + (size_t) t * dim * dim;
        const double * s = fwd.states.data() + (size_t) (t + 1) * dim * dim;
        const double * qt = q.data() + (size_t) t * dim;
        const double * kt = k.data() + (size_t) t * dim;
        const double * vt = v.data() + (size_t) t * dim;
        const double * bar_y = d_output.data() + (size_t) t * dim;
        const double a = std::exp(g[(size_t) t]);

        // y_t = scale * S_t^T q_t.
        for (int i = 0; i < dim; ++i) {
            double sum = 0.0;
            for (int j = 0; j < dim; ++j) {
                bar_s[(size_t) i * dim + j] += scale * qt[i] * bar_y[j];
                sum += s[(size_t) i * dim + j] * bar_y[j];
            }
            grad.q[(size_t) t * dim + i] += scale * sum;
        }

        for (int i = 0; i < dim * dim; ++i) s_bar[(size_t) i] = a * s_prev[i];
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (int i = 0; i < dim; ++i) sum += s_bar[(size_t) i * dim + j] * kt[i];
            u[(size_t) j] = sum;
            delta[(size_t) j] = beta[(size_t) t] * (vt[j] - sum);
        }

        // S_t = Sbar_t + k_t delta_t^T.
        bar_s_bar = bar_s;
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (int i = 0; i < dim; ++i) sum += bar_s[(size_t) i * dim + j] * kt[i];
            bar_delta[(size_t) j] = sum;
        }
        for (int i = 0; i < dim; ++i) {
            double sum = 0.0;
            for (int j = 0; j < dim; ++j) sum += bar_s[(size_t) i * dim + j] * delta[(size_t) j];
            grad.k[(size_t) t * dim + i] += sum;
        }

        // delta_t = beta_t * (v_t - Sbar_t^T k_t).
        double bar_beta = 0.0;
        for (int j = 0; j < dim; ++j) {
            grad.v[(size_t) t * dim + j] += beta[(size_t) t] * bar_delta[(size_t) j];
            bar_beta += bar_delta[(size_t) j] * (vt[j] - u[(size_t) j]);
            bar_u[(size_t) j] = -beta[(size_t) t] * bar_delta[(size_t) j];
        }
        grad.beta[(size_t) t] += bar_beta;

        // u_t = Sbar_t^T k_t.
        for (int i = 0; i < dim; ++i) {
            double sum = 0.0;
            for (int j = 0; j < dim; ++j) {
                bar_s_bar[(size_t) i * dim + j] += kt[i] * bar_u[(size_t) j];
                sum += s_bar[(size_t) i * dim + j] * bar_u[(size_t) j];
            }
            grad.k[(size_t) t * dim + i] += sum;
        }

        // Sbar_t = exp(g_t) * S_{t-1}.
        double bar_a = 0.0;
        for (int i = 0; i < dim * dim; ++i) bar_a += bar_s_bar[(size_t) i] * s_prev[i];
        grad.g[(size_t) t] += a * bar_a;
        for (int i = 0; i < dim * dim; ++i) bar_s[(size_t) i] = a * bar_s_bar[(size_t) i];
    }
    grad.initial_state = std::move(bar_s);
    return grad;
}

} // namespace qwen35_prune
