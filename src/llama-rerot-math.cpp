// llama-rerot-math.cpp — pure mathematical reference implementations for the
// RERoT compute-organization research line (2026-09-21 round). See
// llama-rerot-math.h for scope and contracts. FP64 throughout: these are
// oracles and kernel contracts, not production kernels.

#include "llama-rerot-math.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

[[noreturn]] void invalid_arg(const char * message) {
    throw std::invalid_argument(message);
    }

} // namespace

// ---------------------------------------------------------------------------
// [Q3] Shared-KV multi-reader block attention
// ---------------------------------------------------------------------------

void llama_rerot_attn_state_merge(llama_rerot_attn_state & dst, const llama_rerot_attn_state & src) {
    if (dst.empty() != src.empty()) {
        invalid_arg("RERoT attn merge: empty/non-empty state mismatch");
    }
    if (dst.empty()) {
        return;
    }
    if (dst.u.size() != src.u.size()) {
        invalid_arg("RERoT attn merge: value dimension mismatch");
    }

    const double m = std::max(dst.m, src.m);
    const double w_dst = std::exp(dst.m - m);
    const double w_src = std::exp(src.m - m);

    dst.z = w_dst * dst.z + w_src * src.z;
    for (size_t i = 0; i < dst.u.size(); ++i) {
        dst.u[i] = w_dst * dst.u[i] + w_src * src.u[i];
    }
    dst.m = m;
}

std::vector<llama_rerot_attn_state> llama_rerot_shared_block_attention(
    const std::vector<double> & queries,
    const std::vector<double> & keys,
    const std::vector<double> & values,
    uint32_t n_readers,
    uint32_t head_dim,
    uint32_t value_dim,
    const std::vector<uint8_t> & reader_visible,
    double scale) {
    if (n_readers == 0) {
        invalid_arg("RERoT shared block attention: no readers");
    }
    if (head_dim == 0 || value_dim == 0) {
        invalid_arg("RERoT shared block attention: zero dimension");
    }
    if (queries.size() != size_t(n_readers) * head_dim) {
        invalid_arg("RERoT shared block attention: queries size mismatch");
    }
    const size_t n_block = keys.size() / head_dim;
    if (keys.size() != n_block * head_dim || n_block == 0) {
        invalid_arg("RERoT shared block attention: keys size mismatch");
    }
    if (values.size() != n_block * value_dim) {
        invalid_arg("RERoT shared block attention: values size mismatch");
    }
    if (reader_visible.size() != n_readers) {
        invalid_arg("RERoT shared block attention: visibility size mismatch");
    }

    std::vector<llama_rerot_attn_state> states(n_readers);
    std::vector<uint32_t> visible_readers;
    visible_readers.reserve(n_readers);
    for (uint32_t r = 0; r < n_readers; ++r) {
        if (reader_visible[r] != 0) {
            visible_readers.push_back(r);
        }
    }
    if (visible_readers.empty()) {
        return states;
    }

    // One pass over the shared block: each K/V row is read once and every
    // visible reader consumes it while it is resident. Readers share the data
    // supply, never the softmax statistics (each keeps its own m/z/u).
    for (size_t j = 0; j < n_block; ++j) {
        const double * key = keys.data() + j * head_dim;
        const double * val = values.data() + j * value_dim;

        for (const uint32_t r : visible_readers) {
            const double * query = queries.data() + size_t(r) * head_dim;

            double dot = 0.0;
            for (uint32_t e = 0; e < head_dim; ++e) {
                dot += query[e] * key[e];
            }
            const double score = scale * dot;

            llama_rerot_attn_state & state = states[r];
            if (state.empty()) {
                state.m = score;
                state.z = 1.0;
                state.u.assign(val, val + value_dim);
            } else {
                const double m = std::max(state.m, score);
                const double w_old = std::exp(state.m - m);
                const double w_new = std::exp(score - m);
                state.z = w_old * state.z + w_new;
                for (uint32_t e = 0; e < value_dim; ++e) {
                    state.u[e] = w_old * state.u[e] + w_new * val[e];
                }
                state.m = m;
            }
        }
    }
    return states;
}

std::vector<double> llama_rerot_attn_state_output(const llama_rerot_attn_state & state) {
    if (state.empty()) {
        invalid_arg("RERoT attn output: state is empty");
    }
    if (!(state.z > 0.0) || !std::isfinite(state.z)) {
        invalid_arg("RERoT attn output: non-positive or non-finite normalizer");
    }
    std::vector<double> out(state.u.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = state.u[i] / state.z;
    }
    return out;
}

// ---------------------------------------------------------------------------
// [Q5] GDN common base + per-lane low-rank increments
// ---------------------------------------------------------------------------
//
// Layout conventions (shared by every helper below):
//   base : [d_k * d_v] row-major, B[j, e] = base[j * d_v + e]
//   u    : [d_k * r] row-major,  U[j, c] = u[j * r + c]
//   v    : [d_v * r] row-major,  V[e, c] = v[e * r + c]
//   S    : [d_k * d_v] row-major, S = a * B + U V^T

std::vector<double> llama_rerot_gdn_base_project(
    const std::vector<double> & base,
    const std::vector<double> & x_vectors,
    uint32_t d_k,
    uint32_t d_v) {
    if (d_k == 0 || d_v == 0) {
        invalid_arg("RERoT GDN base project: zero dimension");
    }
    if (base.size() != size_t(d_k) * d_v) {
        invalid_arg("RERoT GDN base project: base size mismatch");
    }
    if (x_vectors.empty() || x_vectors.size() % d_k != 0) {
        invalid_arg("RERoT GDN base project: x vectors size mismatch");
    }
    const size_t n_vectors = x_vectors.size() / d_k;
    std::vector<double> out(n_vectors * d_v);
    // B^T x = sum_j x_j * B[j, :] — one pass over the shared base serves all
    // vectors (all lanes' k_i / q_i in one call).
    for (size_t n = 0; n < n_vectors; ++n) {
        const double * x = x_vectors.data() + n * d_k;
        double * o = out.data() + n * d_v;
        for (uint32_t j = 0; j < d_k; ++j) {
            const double xj = x[j];
            if (xj == 0.0) {
                continue;
            }
            const double * brow = base.data() + size_t(j) * d_v;
            for (uint32_t e = 0; e < d_v; ++e) {
                o[e] += xj * brow[e];
            }
        }
    }
    return out;
}

llama_rerot_gdn_lowrank_state llama_rerot_gdn_lowrank_step(
    const llama_rerot_gdn_lowrank_state & state,
    const std::vector<double> & k,
    const std::vector<double> & v,
    double alpha,
    double beta,
    const std::vector<double> & base_proj_k) {
    if (k.empty() || v.empty()) {
        invalid_arg("RERoT GDN lowrank step: empty k/v");
    }
    const uint32_t d_k = uint32_t(k.size());
    const uint32_t d_v = uint32_t(v.size());
    const uint32_t r = state.r;
    if (state.u.size() != size_t(d_k) * r || state.v.size() != size_t(d_v) * r) {
        invalid_arg("RERoT GDN lowrank step: state size mismatch");
    }
    if (base_proj_k.size() != d_v) {
        invalid_arg("RERoT GDN lowrank step: base projection size mismatch");
    }

    // Sbar^T k = alpha * (a * (B^T k) + V (U^T k)); B^T k arrives precomputed
    // (shared-base projection, reused by every consumer of the same base).
    std::vector<double> u_t_k(r, 0.0);
    for (uint32_t c = 0; c < r; ++c) {
        double dot = 0.0;
        for (uint32_t j = 0; j < d_k; ++j) {
            dot += state.u[size_t(j) * r + c] * k[j];
        }
        u_t_k[c] = dot;
    }
    std::vector<double> sbar_t_k(d_v, 0.0);
    for (uint32_t e = 0; e < d_v; ++e) {
        double acc = state.a * base_proj_k[e];
        for (uint32_t c = 0; c < r; ++c) {
            acc += state.v[size_t(e) * r + c] * u_t_k[c];
        }
        sbar_t_k[e] = alpha * acc;
    }

    // delta = v - Sbar^T k. Factored update (exact, no approximation):
    //   S' = alpha*a * B + [alpha*U | beta*k] [V | delta]^T
    // i.e. every old U column scales by alpha, V columns stay, and one new
    // rank-1 term (beta*k) delta^T appends. Rank grows by one per step.
    llama_rerot_gdn_lowrank_state next;
    next.a = alpha * state.a;
    next.r = r + 1;
    next.u.assign(size_t(d_k) * next.r, 0.0);
    next.v.assign(size_t(d_v) * next.r, 0.0);
    for (uint32_t j = 0; j < d_k; ++j) {
        double * urow = next.u.data() + size_t(j) * next.r;
        for (uint32_t c = 0; c < r; ++c) {
            urow[c] = alpha * state.u[size_t(j) * r + c];
        }
        urow[r] = beta * k[j];
    }
    for (uint32_t e = 0; e < d_v; ++e) {
        double * vrow = next.v.data() + size_t(e) * next.r;
        for (uint32_t c = 0; c < r; ++c) {
            vrow[c] = state.v[size_t(e) * r + c];
        }
        vrow[r] = v[e] - sbar_t_k[e];
    }
    return next;
}

std::vector<double> llama_rerot_gdn_lowrank_output(
    const llama_rerot_gdn_lowrank_state & state,
    const std::vector<double> & q,
    const std::vector<double> & base_proj_q) {
    if (q.empty()) {
        invalid_arg("RERoT GDN lowrank output: empty q");
    }
    const uint32_t d_k = uint32_t(q.size());
    const uint32_t r = state.r;
    if (r == 0 || state.u.size() != size_t(d_k) * r || state.v.size() % r != 0) {
        invalid_arg("RERoT GDN lowrank output: state size mismatch");
    }
    const uint32_t d_v = uint32_t(state.v.size() / r);
    if (base_proj_q.size() != d_v) {
        invalid_arg("RERoT GDN lowrank output: base projection size mismatch");
    }

    // o = S^T q = a * (B^T q) + V (U^T q); B^T q arrives precomputed.
    std::vector<double> u_t_q(r, 0.0);
    for (uint32_t c = 0; c < r; ++c) {
        double dot = 0.0;
        for (uint32_t j = 0; j < d_k; ++j) {
            dot += state.u[size_t(j) * r + c] * q[j];
        }
        u_t_q[c] = dot;
    }
    std::vector<double> out(d_v);
    for (uint32_t e = 0; e < d_v; ++e) {
        double acc = state.a * base_proj_q[e];
        for (uint32_t c = 0; c < r; ++c) {
            acc += state.v[size_t(e) * r + c] * u_t_q[c];
        }
        out[e] = acc;
    }
    return out;
}

std::vector<double> llama_rerot_gdn_lowrank_dense(
    const llama_rerot_gdn_lowrank_state & state,
    const std::vector<double> & base,
    uint32_t d_k,
    uint32_t d_v) {
    if (base.size() != size_t(d_k) * d_v) {
        invalid_arg("RERoT GDN lowrank dense: base size mismatch");
    }
    if (state.u.size() != size_t(d_k) * state.r || state.v.size() != size_t(d_v) * state.r) {
        invalid_arg("RERoT GDN lowrank dense: state size mismatch");
    }

    std::vector<double> s(size_t(d_k) * d_v, 0.0);
    for (uint32_t j = 0; j < d_k; ++j) {
        for (uint32_t e = 0; e < d_v; ++e) {
            double acc = state.a * base[size_t(j) * d_v + e];
            for (uint32_t c = 0; c < state.r; ++c) {
                acc += state.u[size_t(j) * state.r + c] * state.v[size_t(e) * state.r + c];
            }
            s[size_t(j) * d_v + e] = acc;
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// [Q6] Known-token chunk recurrence (WY-style compact transition)
// ---------------------------------------------------------------------------
//
// Recurrence (RERoT.md §2.3): S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1}
// + beta_t k_t v_t^T, i.e. an affine map S_T = Phi S_0 + Y with
// Phi = A_T ... A_1, A_t = alpha_t (I - beta_t k_t k_t^T),
// Y = the same composition applied to zero (exact by linearity).
//
// Phi = G * M with G = prod alphas and M the product of the rank-1
// corrections. WY factorization of M (verified by induction on T):
//   M = I - K W K^T,   W = (I + L)^{-1} diag(beta),
//   L[j, c] = beta_j (k_j . k_c)   for j > c (strictly lower, ROW-indexed
//   beta — the column-indexed variant is NOT the same map when betas differ).
// A production kernel never materializes M: it applies the rank-T operator
// x -> G * (x - K (W (K^T x))), and the payoff is folding ONE chunk once and
// applying it to many lanes' states as shared matrix work.

namespace {

// Solve (I + L) x = b, L strictly lower row-major [t * t]. Forward substitution.
std::vector<double> solve_unit_lower(const std::vector<double> & l, std::vector<double> b) {
    const size_t t = b.size();
    for (size_t i = 0; i < t; ++i) {
        double acc = b[i];
        const double * row = l.data() + i * t;
        for (size_t j = 0; j < i; ++j) {
            acc -= row[j] * b[j];
            }
        if (!std::isfinite(acc)) {
            invalid_arg("RERoT GDN chunk fold: non-finite unit-lower solve");
            }
        b[i] = acc;
        }
    return b;
    }

} // namespace

llama_rerot_gdn_chunk llama_rerot_gdn_chunk_fold(
    uint32_t d_k,
    uint32_t d_v,
    const std::vector<double> & ks,
    const std::vector<double> & vs,
    const std::vector<double> & alphas,
    const std::vector<double> & betas) {
    if (d_k == 0 || d_v == 0) {
        invalid_arg("RERoT GDN chunk fold: zero dimension");
    }
    if (ks.empty() || ks.size() % d_k != 0) {
        invalid_arg("RERoT GDN chunk fold: ks size mismatch");
    }
    const size_t t = ks.size() / d_k;
    if (t == 0) {
        invalid_arg("RERoT GDN chunk fold: empty chunk");
    }
    if (vs.size() != t * d_v || alphas.size() != t || betas.size() != t) {
        invalid_arg("RERoT GDN chunk fold: chunk size mismatch");
    }

    // Y = zero-state arm of the recurrence, step by step (exact).
    std::vector<double> y(size_t(d_k) * d_v, 0.0);
    std::vector<double> sbar_t_k(d_v);
    for (size_t step = 0; step < t; ++step) {
        const double * k = ks.data() + step * d_k;
        const double * v = vs.data() + step * d_v;
        const double a = alphas[step];
        const double b = betas[step];

        for (uint32_t e = 0; e < d_v; ++e) {
            double dot = 0.0;
            for (uint32_t j = 0; j < d_k; ++j) {
                dot += y[size_t(j) * d_v + e] * k[j];
            }
            sbar_t_k[e] = a * dot;
        }
        for (size_t idx = 0; idx < y.size(); ++idx) {
            y[idx] *= a;
        }
        for (uint32_t j = 0; j < d_k; ++j) {
            const double coeff = b * k[j];
            for (uint32_t e = 0; e < d_v; ++e) {
                y[size_t(j) * d_v + e] += coeff * (v[e] - sbar_t_k[e]);
            }
        }
    }

    // W columns: solve (I + L) w = beta_c e_c for each column c.
    std::vector<double> gram(t * t, 0.0);
    for (size_t i = 0; i < t; ++i) {
        const double * ki = ks.data() + i * d_k;
        for (size_t j = 0; j <= i; ++j) {
            const double * kj = ks.data() + j * d_k;
            double dot = 0.0;
            for (uint32_t e = 0; e < d_k; ++e) {
                dot += ki[e] * kj[e];
            }
            gram[i * t + j] = dot;
        }
    }
    std::vector<double> l(t * t, 0.0);
    for (size_t j = 0; j < t; ++j) {
        for (size_t c = 0; c < j; ++c) {
            l[j * t + c] = betas[j] * gram[j * t + c];
        }
    }
    std::vector<std::vector<double>> w_cols(t);
    for (size_t c = 0; c < t; ++c) {
        std::vector<double> e(t, 0.0);
        e[c] = betas[c];
        w_cols[c] = solve_unit_lower(l, std::move(e));
    }

    double g_total = 1.0;
    for (size_t i = 0; i < t; ++i) {
        g_total *= alphas[i];
    }

    // M[:, j] = G * (e_j - K (W (K^T e_j))) — materialized for the reference
    // layer; production keeps the rank-T application.
    llama_rerot_gdn_chunk chunk;
    chunk.y = std::move(y);
    chunk.m.assign(size_t(d_k) * d_k, 0.0);
    std::vector<double> kt_ej(t);
    std::vector<double> w(t);
    for (uint32_t j = 0; j < d_k; ++j) {
        for (size_t step = 0; step < t; ++step) {
            kt_ej[step] = ks[step * d_k + j];
        }
        std::fill(w.begin(), w.end(), 0.0);
        for (size_t c = 0; c < t; ++c) {
            const double coeff = kt_ej[c];
            if (coeff == 0.0) {
                continue;
            }
            const std::vector<double> & col = w_cols[c];
            for (size_t r = 0; r < t; ++r) {
                w[r] += coeff * col[r];
            }
        }
        for (uint32_t i = 0; i < d_k; ++i) {
            double acc = (i == j) ? 1.0 : 0.0;
            for (size_t r = 0; r < t; ++r) {
                acc -= w[r] * ks[r * d_k + i];
            }
            chunk.m[size_t(i) * d_k + j] = g_total * acc;
        }
    }
    return chunk;
}

std::vector<double> llama_rerot_gdn_chunk_apply(
    const llama_rerot_gdn_chunk & chunk,
    const std::vector<double> & s_in,
    uint32_t d_k,
    uint32_t d_v) {
    if (chunk.m.size() != size_t(d_k) * d_k || chunk.y.size() != size_t(d_k) * d_v) {
        invalid_arg("RERoT GDN chunk apply: chunk size mismatch");
    }
    if (s_in.size() != size_t(d_k) * d_v) {
        invalid_arg("RERoT GDN chunk apply: state size mismatch");
    }

    // S_out = M S_in + Y. A production kernel applies M by its rank-T
    // factorization; the reference materializes the product.
    std::vector<double> out(size_t(d_k) * d_v, 0.0);
    for (uint32_t j = 0; j < d_k; ++j) {
        const double * mrow = chunk.m.data() + size_t(j) * d_k;
        double * orow = out.data() + size_t(j) * d_v;
        for (uint32_t c = 0; c < d_k; ++c) {
            const double coeff = mrow[c];
            if (coeff == 0.0) {
                continue;
            }
            const double * srow = s_in.data() + size_t(c) * d_v;
            for (uint32_t e = 0; e < d_v; ++e) {
                orow[e] += coeff * srow[e];
            }
        }
    }
    for (size_t idx = 0; idx < out.size(); ++idx) {
        out[idx] += chunk.y[idx];
    }
    return out;
}

std::vector<double> llama_rerot_gdn_steps_reference(
    const std::vector<double> & s0,
    uint32_t d_k,
    uint32_t d_v,
    const std::vector<double> & ks,
    const std::vector<double> & vs,
    const std::vector<double> & alphas,
    const std::vector<double> & betas) {
    if (d_k == 0 || d_v == 0 || s0.size() != size_t(d_k) * d_v) {
        invalid_arg("RERoT GDN steps reference: state size mismatch");
    }
    if (ks.empty() || ks.size() % d_k != 0) {
        invalid_arg("RERoT GDN steps reference: ks size mismatch");
    }
    const size_t t = ks.size() / d_k;
    if (vs.size() != t * d_v || alphas.size() != t || betas.size() != t) {
        invalid_arg("RERoT GDN steps reference: chunk size mismatch");
    }

    std::vector<double> s = s0;
    std::vector<double> sbar_t_k(d_v);
    for (size_t step = 0; step < t; ++step) {
        const double * k = ks.data() + step * d_k;
        const double * v = vs.data() + step * d_v;
        const double a = alphas[step];
        const double b = betas[step];

        for (uint32_t e = 0; e < d_v; ++e) {
            double dot = 0.0;
            for (uint32_t j = 0; j < d_k; ++j) {
                dot += s[size_t(j) * d_v + e] * k[j];
            }
            sbar_t_k[e] = a * dot;
        }
        for (size_t idx = 0; idx < s.size(); ++idx) {
            s[idx] *= a;
        }
        for (uint32_t j = 0; j < d_k; ++j) {
            const double coeff = b * k[j];
            for (uint32_t e = 0; e < d_v; ++e) {
                s[size_t(j) * d_v + e] += coeff * (v[e] - sbar_t_k[e]);
            }
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// [Q7] PQ2_0 bit-plane subset-sum inner product
// ---------------------------------------------------------------------------

double llama_rerot_pq2_bitplane_dot(const uint8_t * qs, const double * x, uint32_t n) {
    if (!qs || !x) {
        invalid_arg("RERoT PQ2 bitplane dot: null input");
    }
    if (n == 0 || n % 4 != 0) {
        invalid_arg("RERoT PQ2 bitplane dot: n must be a positive multiple of 4");
    }

    // code layout matches ggml dequantize_row_pq2_0: byte j/4, shift (j%4)*2,
    // code {0,1,2,3} -> {-1,0,+1,+2} via (code - 1) = b0 + 2*b1 - 1.
    double sum_all = 0.0;
    double sum_b0 = 0.0;
    double sum_b1 = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
        const uint8_t byte = qs[j / 4];
        const uint32_t shift = (j % 4) * 2;
        const uint8_t code = (byte >> shift) & 0x3;
        const double xj = x[j];
        sum_all += xj;
        if (code & 0x1) {
            sum_b0 += xj;
        }
        if (code & 0x2) {
            sum_b1 += xj;
        }
    }
    // sum_j (b0_j + 2 b1_j - 1) x_j
    return sum_b0 + 2.0 * sum_b1 - sum_all;
}

double llama_rerot_pq2_lut_dot(const uint8_t * qs, const double * x, uint32_t n) {
    if (!qs || !x) {
        invalid_arg("RERoT PQ2 LUT dot: null input");
    }
    if (n == 0 || n % 4 != 0) {
        invalid_arg("RERoT PQ2 LUT dot: n must be a positive multiple of 4");
    }

    // Per group of 4 weights: T[m] = sum of x_j over the subset selected by
    // the 4-bit mask m. The group's contribution is T[m0] + 2*T[m1] - T[15]
    // where m0/m1 are the group's low/high bit-plane masks — one table build
    // and two lookups per 4 weights (T-MAC-style), not one lookup per weight.
    double total = 0.0;
    double t[16];
    for (uint32_t group = 0; group < n / 4; ++group) {
        t[0] = 0.0;
        for (uint32_t m = 1; m < 16; ++m) {
            const uint32_t idx = __builtin_ctz(m);
            t[m] = t[m & (m - 1)] + x[group * 4 + idx];
        }

        const uint32_t byte = qs[group];
        uint32_t m0 = 0;
        uint32_t m1 = 0;
        for (uint32_t sub = 0; sub < 4; ++sub) {
            const uint8_t code = (byte >> (sub * 2)) & 0x3;
            m0 |= (code & 0x1u) << sub;
            m1 |= ((code >> 1) & 0x1u) << sub;
        }
        total += t[m0] + 2.0 * t[m1] - t[15];
    }
    return total;
}

// ---------------------------------------------------------------------------
// [Q2] Span-view reference
// ---------------------------------------------------------------------------

int64_t llama_rerot_span_effective_pos(int64_t query_virtual_pos, const llama_rerot_span_view & span) {
    // §2.4: within the span, s_j - v_j = phase is constant, so
    //   q_v + s_j - v_j = q_v + phase   for every key j in the span.
    // One effective query serves the whole span; j cancels.
    return query_virtual_pos + span.phase;
}

uint32_t llama_rerot_span_causal_len(const llama_rerot_span_view & span, int64_t query_virtual_pos) {
    if (span.len == 0) {
        invalid_arg("RERoT span causal len: empty span");
    }
    // Visible prefix = keys with virtual position <= query_virtual_pos.
    // Span covers virtuals [begin, begin+len); one cut replaces the per-key
    // causal mask.
    const int64_t end = int64_t(span.begin) + int64_t(span.len);
    const int64_t limit = std::min<int64_t>(query_virtual_pos + 1, end);
    const int64_t visible = limit - int64_t(span.begin);
    return visible <= 0 ? 0 : uint32_t(visible);
}

double llama_rerot_span_long_fraction(const std::vector<llama_rerot_span_view> & spans, uint32_t min_long) {
    uint64_t total = 0;
    uint64_t long_total = 0;
    for (const auto & span : spans) {
        if (span.len == 0) {
            invalid_arg("RERoT span long fraction: empty span");
        }
        total += span.len;
        if (span.len >= min_long) {
            long_total += span.len;
        }
    }
    if (total == 0) {
        return 0.0;
    }
    return double(long_total) / double(total);
}

// ---------------------------------------------------------------------------
// [Q4] Structure/numeric separation over a frozen run order
// ---------------------------------------------------------------------------

std::vector<int64_t> llama_rerot_virtual_starts(
    const llama_rerot_run_order & order,
    const llama_rerot_run_lengths & lengths) {
    if (lengths.len.size() != order.run_ids.size()) {
        invalid_arg("RERoT virtual starts: size mismatch");
    }
    std::vector<int64_t> starts(order.run_ids.size(), 0);
    int64_t acc = 0;
    for (size_t i = 0; i < order.run_ids.size(); ++i) {
        starts[i] = acc;
        acc += int64_t(lengths.len[i]);
        if (acc < starts[i]) {
            invalid_arg("RERoT virtual starts: overflow");
        }
    }
    return starts;
}

std::vector<int64_t> llama_rerot_virtual_starts_after_growth(
    const llama_rerot_run_order & order,
    const llama_rerot_run_lengths & lengths,
    size_t grown_run,
    uint32_t delta) {
    if (grown_run >= lengths.len.size()) {
        invalid_arg("RERoT virtual starts after growth: bad run index");
    }
    llama_rerot_run_lengths grown = lengths;
    const uint64_t new_len = uint64_t(grown.len[grown_run]) + uint64_t(delta);
    if (new_len > uint64_t(std::numeric_limits<uint32_t>::max())) {
        invalid_arg("RERoT virtual starts after growth: length overflow");
    }
    grown.len[grown_run] = uint32_t(new_len);
    return llama_rerot_virtual_starts(order, grown);
}

uint64_t llama_rerot_run_order_signature(const llama_rerot_run_order & order) {
    // FNV-1a over run ids: stable across processes, changes iff the order
    // (the structure) changes. Lengths/watermarks never enter the signature.
    uint64_t hash = 1469598103934665603ull;
    for (const uint32_t id : order.run_ids) {
        hash ^= uint64_t(id);
        hash *= 1099511628211ull;
    }
    return hash;
}

// ---------------------------------------------------------------------------
// [Q8] Skip-block bounds
// ---------------------------------------------------------------------------

double llama_rerot_skip_mass_bound(
    const std::vector<double> & query,
    const std::vector<double> & center,
    double radius,
    uint32_t n_block,
    double scale) {
    if (query.size() != center.size() || query.empty()) {
        invalid_arg("RERoT skip mass bound: size mismatch");
    }
    if (!(radius >= 0.0) || !std::isfinite(radius)) {
        invalid_arg("RERoT skip mass bound: negative radius");
    }
    if (n_block == 0) {
        invalid_arg("RERoT skip mass bound: empty block");
    }

    // q . k_j = q . c + q . (k_j - c) <= q . c + |q| * |k_j - c| <= q . c + |q| * r
    double dot = 0.0;
    double q_norm = 0.0;
    for (size_t i = 0; i < query.size(); ++i) {
        dot += query[i] * center[i];
        q_norm += query[i] * query[i];
    }
    q_norm = std::sqrt(q_norm);
    const double bound = scale * (dot + q_norm * radius);
    // n_block keys each contribute at most exp(bound).
    return double(n_block) * std::exp(bound);
}

std::pair<double, double> llama_rerot_skip_output_bound(double z_keep, double z_skip_bound, double v_max) {
    if (!(z_keep > 0.0) || !std::isfinite(z_keep) || !std::isfinite(z_skip_bound) || z_skip_bound < 0.0) {
        invalid_arg("RERoT skip output bound: non-finite or non-positive masses");
    }
    if (!(v_max >= 0.0) || !std::isfinite(v_max)) {
        invalid_arg("RERoT skip output bound: bad v_max");
    }
    const double delta = z_skip_bound / (z_keep + z_skip_bound);
    return { delta, 2.0 * v_max * delta };
}

// ---------------------------------------------------------------------------
// [Q9] Joint sampler contract
// ---------------------------------------------------------------------------

namespace {

// xorshift64* — one draw per step; the stream is the pen's own, so draw
// counts never depend on cohort composition (trajectory safety).
double next_unit(uint64_t & state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return double((state * 2685821657736338717ull) >> 11) / 9007199254740992.0;
}

} // namespace

uint64_t llama_rerot_joint_sample_seed(uint64_t base_seed, uint32_t pen) {
    // SplitMix64 over (base, pen): independent streams per pen, stable across
    // cohort-size and row-order changes.
    uint64_t z = base_seed + (uint64_t(pen) + 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint32_t llama_rerot_joint_sample_row(
    const std::vector<double> & logits,
    double temperature,
    uint32_t top_k,
    double top_p,
    uint64_t & rng_state) {
    if (logits.empty()) {
        invalid_arg("RERoT joint sample row: empty logits");
    }
    if (!(temperature > 0.0) || !std::isfinite(temperature)) {
        invalid_arg("RERoT joint sample row: bad temperature");
    }
    if (!(top_p >= 0.0 && top_p <= 1.0)) {
        invalid_arg("RERoT joint sample row: bad top_p");
    }

    // temperature
    std::vector<double> t(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        t[i] = logits[i] / temperature;
    }

    // top-k: keep the k largest; ties keep the LOWEST token index (stable
    // ordering with index as tie-break).
    if (top_k > 0 && top_k < t.size()) {
        std::vector<uint32_t> idx(t.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = uint32_t(i);
        }
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
            return t[a] > t[b];
        });
        std::vector<uint8_t> kept(t.size(), 0);
        for (uint32_t i = 0; i < top_k; ++i) {
            kept[idx[i]] = 1;
        }
        for (size_t i = 0; i < t.size(); ++i) {
            if (!kept[i]) {
                t[i] = -INFINITY;
            }
        }
    }

    // softmax over the surviving set
    double m = -INFINITY;
    for (double v : t) {
        m = std::max(m, v);
    }
    std::vector<double> probs(t.size(), 0.0);
    double z = 0.0;
    for (size_t i = 0; i < t.size(); ++i) {
        if (std::isfinite(t[i])) {
            probs[i] = std::exp(t[i] - m);
            z += probs[i];
        }
    }

    // top-p (nucleus): keep the smallest prefix of the probability-descending
    // order whose mass >= top_p; ties break to the lowest token index.
    if (top_p < 1.0) {
        std::vector<uint32_t> idx(t.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = uint32_t(i);
        }
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
            return probs[a] > probs[b];
        });
        double cum = 0.0;
        std::vector<uint8_t> in_nucleus(t.size(), 0);
        for (uint32_t i : idx) {
            in_nucleus[i] = 1;
            cum += probs[i] / z;
            if (cum >= top_p) {
                break;
            }
        }
        for (size_t i = 0; i < t.size(); ++i) {
            if (!in_nucleus[i]) {
                probs[i] = 0.0;
            }
        }
        z = 0.0;
        for (double p : probs) {
            z += p;
        }
    }

    // draw from the pen's OWN stream
    const double u = next_unit(rng_state);
    double target = u * z;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] <= 0.0) {
            continue;
        }
        target -= probs[i];
        if (target < 0.0) {
            return uint32_t(i);
        }
    }
    // Floating-point fallback: the last surviving index.
    for (size_t i = probs.size(); i-- > 0;) {
        if (probs[i] > 0.0) {
            return uint32_t(i);
        }
    }
    return uint32_t(logits.size() - 1);
}

std::vector<uint32_t> llama_rerot_joint_argmax_rows(
    const std::vector<std::vector<double>> & block_maxima) {
    if (block_maxima.empty() || block_maxima[0].empty()) {
        invalid_arg("RERoT joint argmax rows: empty input");
    }
    const size_t vocab = block_maxima[0].size();
    std::vector<double> best(vocab, -INFINITY);
    for (const auto & block : block_maxima) {
        if (block.size() != vocab) {
            invalid_arg("RERoT joint argmax rows: block size mismatch");
        }
        for (size_t i = 0; i < vocab; ++i) {
            best[i] = std::max(best[i], block[i]);
        }
    }
    // Lowest-index tie-break: strict > keeps the earlier maximum.
    uint32_t arg = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (best[i] > best[arg]) {
            arg = uint32_t(i);
        }
    }
    return std::vector<uint32_t>{ arg };
}
// ---------------------------------------------------------------------------
// [Q10] Frontier-grid joint verification
// ---------------------------------------------------------------------------

llama_rerot_grid_verdict llama_rerot_verify_grid(
    const llama_rerot_draft_grid & grid,
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> & reads) {
    if (grid.n_pens == 0 || grid.horizon == 0) {
        invalid_arg("RERoT verify grid: empty grid");
    }
    if (grid.draft.size() != size_t(grid.n_pens) * grid.horizon ||
        grid.truth.size() != size_t(grid.n_pens) * grid.horizon) {
        invalid_arg("RERoT verify grid: grid size mismatch");
    }
    if (reads.size() != grid.horizon) {
        invalid_arg("RERoT verify grid: reads size mismatch");
    }
    for (const auto & edges : reads) {
        for (const auto & e : edges) {
            if (e.first >= grid.n_pens || e.second >= grid.n_pens) {
                invalid_arg("RERoT verify grid: read edge out of range");
            }
        }
    }

    llama_rerot_grid_verdict verdict;
    verdict.accepted.assign(grid.n_pens, 0);
    verdict.replacement.assign(grid.n_pens, 0);

    // Column-by-column, STRONG barrier-after: cell (pen, h) reads sources'
    // column h-1 COMMITTED tokens. A cell survives iff the pen's own prefix
    // through h-1 survived AND every read source survived its column h-1 cell
    // (self-reads are trivially satisfied). Rejection at h publishes truth as
    // the replacement; dependents at h+1 then consume the replacement.
    std::vector<uint32_t> alive(grid.n_pens, 1); // prefix alive through h-1
    for (uint32_t h = 0; h < grid.horizon; ++h) {
        std::vector<uint32_t> next_alive(grid.n_pens, 0);
        for (uint32_t p = 0; p < grid.n_pens; ++p) {
            if (!alive[p]) {
                continue; // prefix already broken; nothing at h survives
            }
            const bool draft_ok = grid.draft[size_t(p) * grid.horizon + h] ==
                                  grid.truth[size_t(p) * grid.horizon + h];
            bool deps_ok = true;
            for (const auto & e : reads[h]) {
                if (e.first != p) {
                    continue;
                }
                if (h > 0 && e.second != p && !alive[e.second]) {
                    deps_ok = false; // source's column h-1 was rejected
                }
            }
            if (draft_ok && deps_ok) {
                next_alive[p] = 1;
                verdict.accepted[p] = h + 1;
            } else {
                verdict.replacement[p] = grid.truth[size_t(p) * grid.horizon + h];
            }
        }
        alive = std::move(next_alive);
    }
    return verdict;
}

llama_rerot_grid_verdict llama_rerot_verify_grid_naive(const llama_rerot_draft_grid & grid) {
    if (grid.n_pens == 0 || grid.horizon == 0) {
        invalid_arg("RERoT verify grid naive: empty grid");
    }
    if (grid.draft.size() != size_t(grid.n_pens) * grid.horizon ||
        grid.truth.size() != size_t(grid.n_pens) * grid.horizon) {
        invalid_arg("RERoT verify grid naive: grid size mismatch");
    }

    // Per-row independent verification — the counterexample engine: ignores
    // that a cell may have consumed another pen's DRAFT at column h-1, which
    // was then rejected and replaced.
    llama_rerot_grid_verdict verdict;
    verdict.accepted.assign(grid.n_pens, 0);
    verdict.replacement.assign(grid.n_pens, 0);
    for (uint32_t p = 0; p < grid.n_pens; ++p) {
        uint32_t h = 0;
        while (h < grid.horizon &&
               grid.draft[size_t(p) * grid.horizon + h] == grid.truth[size_t(p) * grid.horizon + h]) {
            ++h;
        }
        verdict.accepted[p] = h;
        if (h < grid.horizon) {
            verdict.replacement[p] = grid.truth[size_t(p) * grid.horizon + h];
        }
    }
    return verdict;
}

// ---------------------------------------------------------------------------
// [R02] Live Q-prep reference (C07): gather + effective-position RoPE over
// the active group prefix only, padding poisoned. Self-contained rotation
// math mirroring ggml rope NORMAL/NEOX (freq_base/freq_scale/freq_factors/
// YaRN correction); MROPE/IMROPE section modes are supported through the
// same per-coordinate machinery. No ggml dependency — this stays a pure
// reference module.
// ---------------------------------------------------------------------------

namespace {

// YaRN ramp/correction, mirroring ggml_cpu ops.cpp rope_yarn/rope_yarn_ramp.
float qprep_yarn_ramp(float low, float high, int64_t i0) {
    const float y = (float(i0) / 2.0f - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

void qprep_rope_yarn(
        float theta_extrap, float freq_scale, const float corr_dims[2],
        int64_t i0, float ext_factor, float mscale,
        float & cos_theta, float & sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = qprep_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * std::logf(1.0f / freq_scale);
    }
    cos_theta = std::cosf(theta) * mscale;
    sin_theta = std::sinf(theta) * mscale;
}

// YaRN correction-dims selection mirroring ggml rope_yarn_corr_dims.
void qprep_yarn_corr_dims(
        int64_t n_dims, int64_t n_ctx_orig,
        float freq_base, float beta_fast, float beta_slow,
        float corr_dims[2]) {
    const float start = std::floorf(float(n_ctx_orig) *
        std::logf(freq_base) / (beta_fast * std::logf(2.0f)));
    const float stop  = std::floorf(float(n_ctx_orig) *
        std::logf(freq_base) / (beta_slow * std::logf(2.0f)));
    corr_dims[0] = std::max(0.0f, std::min(float(n_dims - 1), start));
    corr_dims[1] = std::max(0.0f, std::min(float(n_dims - 1), stop));
}

void qprep_rope_cache(
        float theta_base, float freq_scale, const float * freq_factors,
        const float corr_dims[2], int64_t n_dims, float ext_factor, float mscale,
        float theta_scale, float sin_sign, std::vector<float> & cache) {
    cache.resize(size_t(n_dims) * 2);
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        qprep_rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale,
            cache[size_t(i0) + 0], cache[size_t(i0) + 1]);
        cache[size_t(i0) + 1] *= sin_sign;
        theta *= theta_scale;
    }
}

void qprep_rotate_normal(
        const std::vector<float> & cache, int64_t n_dims, int64_t n_rot,
        int64_t stride, float * vec) {
    // NORMAL (LLaMA): pairs (i, i + n_rot/2) for i in [0, n_rot/2).
    for (int64_t i = 0; i < n_rot / 2; ++i) {
        const float x = vec[i * stride];
        const float y = vec[(i + n_rot / 2) * stride];
        const float c = cache[2 * i + 0];
        const float s = cache[2 * i + 1];
        vec[i * stride]             = x * c - y * s;
        vec[(i + n_rot / 2) * stride] = x * s + y * c;
    }
    (void) n_dims;
}

void qprep_rotate_neox(
        const std::vector<float> & cache, int64_t n_dims, int64_t n_rot,
        int64_t stride, float * vec) {
    // NEOX (GPT-NeoX): pairs (i, i + n_rot/2) for i in [0, n_rot/2), same
    // pair order but the half layout differs for n_dims > n_rot. The ggml
    // neox path applies the rotation to the first n_rot dims in-place.
    for (int64_t i = 0; i < n_rot / 2; ++i) {
        const float x = vec[i * stride];
        const float y = vec[(i + n_rot / 2) * stride];
        const float c = cache[2 * i + 0];
        const float s = cache[2 * i + 1];
        vec[i * stride]             = x * c - y * s;
        vec[(i + n_rot / 2) * stride] = x * s + y * c;
    }
    (void) n_dims;
}

} // namespace

std::vector<int> llama_rerot_q_prep_supported_modes() {
    // R02 candidate support table (single source for the future Vulkan
    // dispatch gate): VISION is explicitly NOT supported (text-only v1,
    // matching build_rerot_q_groups' require_supported gate).
    return { 0 /* NORMAL */, 2 /* NEOX */, 8 /* MROPE */, 24 /* IMROPE */ };
}

std::vector<float> llama_rerot_q_prep_reference(
        const float * q_raw,
        const int32_t * q_indices,
        const int32_t * q_pos,
        const llama_rerot_q_prep_contract & c) {
    if (c.head_dim <= 0 || c.heads <= 0 || c.n_tokens <= 0) {
        invalid_arg("RERoT q-prep reference: invalid head/token dims");
    }
    if (c.capacity <= 0 || c.active < 0 || c.active > c.capacity) {
        invalid_arg("RERoT q-prep reference: active/capacity contract violated");
    }
    if (c.n_rot <= 0 || c.n_rot > c.head_dim || (c.n_rot % 2) != 0) {
        invalid_arg("RERoT q-prep reference: invalid rotary dims");
    }
    if (c.n_pos != 1 && c.n_pos != 4) {
        invalid_arg("RERoT q-prep reference: n_pos must be 1 or 4");
    }
    const bool multi_coord = c.rope_mode == 8 /* MROPE */ || c.rope_mode == 24 /* IMROPE */;
    if (multi_coord && c.n_pos != 4) {
        invalid_arg("RERoT q-prep reference: MROPE/IMROPE require n_pos == 4");
    }
    if (!multi_coord && c.n_pos != 1) {
        invalid_arg("RERoT q-prep reference: NORMAL/NEOX require n_pos == 1");
    }
    if (c.rope_mode != 0 && c.rope_mode != 2 && c.rope_mode != 8 && c.rope_mode != 24) {
        invalid_arg("RERoT q-prep reference: unsupported rope mode");
    }
    if (c.rope_mode == 8 || c.rope_mode == 24) {
        if (c.mrope_sections == nullptr) {
            invalid_arg("RERoT q-prep reference: MROPE/IMROPE require sections");
        }
    }
    if (q_raw == nullptr || q_indices == nullptr || q_pos == nullptr) {
        invalid_arg("RERoT q-prep reference: null input pointer");
    }

    float corr_dims[2] = { 0.0f, 0.0f };
    const float theta_scale = std::powf(c.freq_base, -2.0f / float(c.n_rot));
    if (c.ext_factor != 0.0f) {
        qprep_yarn_corr_dims(c.n_rot, c.n_ctx_orig, c.freq_base,
            c.beta_fast, c.beta_slow, corr_dims);
    }

    // Output: [head_dim, heads, capacity] row-major, padding poisoned.
    const size_t out_size = size_t(c.head_dim) * size_t(c.heads) * size_t(c.capacity);
    std::vector<float> out(out_size);
    const float nanf_val = std::nanf("");

    const size_t head_bytes = size_t(c.head_dim);
    std::vector<float> cache;

    for (int64_t g = 0; g < c.active; ++g) {
        const int32_t token = q_indices[g];
        if (token < 0 || token >= c.n_tokens) {
            invalid_arg("RERoT q-prep reference: q_indices out of range");
        }
        const float * src = q_raw + size_t(token) * size_t(c.heads) * head_bytes;
        float * dst = out.data() + size_t(g) * size_t(c.heads) * head_bytes;

        for (int64_t h = 0; h < c.heads; ++h) {
            float * vec = dst + size_t(h) * head_bytes;
            const float * svec = src + size_t(h) * head_bytes;
            std::copy(svec, svec + head_bytes, vec);
        }

        if (multi_coord) {
            // Sectioned multi-coordinate rotation: per-section dim range
            // rotated with its own coordinate (MROPE); IMROPE uses the same
            // sectioning with the text-delta convention applied by the
            // caller's q_pos (fourth coord pinned 0 by fill_spans).
            const int sections[4] = {
                c.mrope_sections[0], c.mrope_sections[1],
                c.mrope_sections[2], c.mrope_sections[3],
            };
            for (int sec = 0; sec < 4; ++sec) {
                const int64_t sec_begin = sec == 0 ? 0 : sections[sec - 1];
                const int64_t sec_end   = sections[sec];
                if (sec_end <= sec_begin || sec_end > c.n_rot) {
                    invalid_arg("RERoT q-prep reference: invalid mrope sections");
                }
                const int32_t p = q_pos[size_t(sec) * size_t(c.capacity) + g];
                const float theta_base = float(p) * std::powf(c.freq_base, 0.0f);
                qprep_rope_cache(theta_base, c.freq_scale, c.freq_factors, corr_dims,
                    sec_end - sec_begin, c.ext_factor, c.attn_factor, theta_scale, 1.0f, cache);
                for (int64_t h = 0; h < c.heads; ++h) {
                    float * vec = dst + size_t(h) * head_bytes + sec_begin;
                    qprep_rotate_normal(cache, sec_end - sec_begin, sec_end - sec_begin, 1, vec);
                }
            }
        } else {
            const int32_t p = q_pos[g];
            const float theta_base = float(p) * std::powf(c.freq_base, 0.0f);
            qprep_rope_cache(theta_base, c.freq_scale, c.freq_factors, corr_dims,
                c.n_rot, c.ext_factor, c.attn_factor, theta_scale, 1.0f, cache);
            for (int64_t h = 0; h < c.heads; ++h) {
                float * vec = dst + size_t(h) * head_bytes;
                if (c.rope_mode == 2) {
                    qprep_rotate_neox(cache, c.n_rot, c.n_rot, 1, vec);
                } else {
                    qprep_rotate_normal(cache, c.n_rot, c.n_rot, 1, vec);
                }
            }
        }
    }

    // Poison the padding: [active, capacity) groups all NaN.
    for (int64_t g = c.active; g < c.capacity; ++g) {
        float * dst = out.data() + size_t(g) * size_t(c.heads) * head_bytes;
        std::fill(dst, dst + size_t(c.heads) * head_bytes, nanf_val);
    }

    return out;
}