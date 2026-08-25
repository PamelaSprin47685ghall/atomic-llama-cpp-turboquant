#include "gdn-geometry-plan.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace qwen35_prune {

static void require_finite_nonnegative(double x, const char * what) {
    if (!std::isfinite(x) || x < 0.0) {
        throw std::runtime_error(std::string("invalid ") + what);
    }
}

gdn_geometry_shrink_result gdn_geometry_shrink_second_moment(
        const std::vector<double> & empirical_sum,
        uint64_t samples,
        const std::vector<double> & prior_diag,
        int dim) {
    if (dim <= 0 || empirical_sum.size() != (size_t) dim * dim || prior_diag.size() != (size_t) dim) {
        throw std::runtime_error("GDN geometry shrinkage shape mismatch");
    }
    if (samples == 0) throw std::runtime_error("GDN geometry shrinkage requires samples");

    gdn_geometry_shrink_result out;
    out.matrix.resize((size_t) dim * dim, 0.0);
    const double inv_n = 1.0 / (double) samples;

    for (int i = 0; i < dim; ++i) {
        const double d = empirical_sum[(size_t) i * dim + i] * inv_n;
        if (!std::isfinite(d) || d < -1e-12) {
            throw std::runtime_error("invalid empirical second-moment diagonal");
        }
        out.data_trace += std::max(0.0, d);
        require_finite_nonnegative(prior_diag[(size_t) i], "GDN geometry prior diagonal");
        out.prior_trace += prior_diag[(size_t) i];
    }

    out.shrinkage = (double) dim / ((double) samples + dim);
    const double prior_scale = out.prior_trace > 0.0 ? out.data_trace / out.prior_trace : 0.0;

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            const double a = 0.5 * (empirical_sum[(size_t) i * dim + j] +
                                    empirical_sum[(size_t) j * dim + i]) * inv_n;
            if (!std::isfinite(a)) throw std::runtime_error("non-finite GDN empirical second moment");
            out.matrix[(size_t) i * dim + j] = (1.0 - out.shrinkage) * a;
        }
        out.matrix[(size_t) i * dim + i] +=
                out.shrinkage * prior_scale * prior_diag[(size_t) i];
    }
    return out;
}

std::vector<int> gdn_geometry_pivoted_cholesky_select(
        const std::vector<double> & psd,
        int dim,
        int rank) {
    if (dim <= 0 || rank <= 0 || rank > dim || psd.size() != (size_t) dim * dim) {
        throw std::runtime_error("GDN pivoted-Cholesky shape mismatch");
    }

    std::vector<double> residual((size_t) dim, 0.0);
    std::vector<double> factor((size_t) dim * rank, 0.0);
    std::vector<uint8_t> used((size_t) dim, 0);
    double max_initial = 0.0;
    for (int i = 0; i < dim; ++i) {
        const double d = psd[(size_t) i * dim + i];
        if (!std::isfinite(d)) throw std::runtime_error("non-finite GDN geometry diagonal");
        residual[(size_t) i] = std::max(0.0, d);
        max_initial = std::max(max_initial, residual[(size_t) i]);
    }
    const double floor = std::max(1e-30, max_initial * 1e-14);

    std::vector<int> selected;
    selected.reserve((size_t) rank);
    for (int k = 0; k < rank; ++k) {
        int pivot = -1;
        double best = -1.0;
        for (int i = 0; i < dim; ++i) {
            if (used[(size_t) i]) continue;
            const double r = residual[(size_t) i];
            if (r > best) {
                best = r;
                pivot = i;
            }
        }
        if (pivot < 0) throw std::runtime_error("GDN pivoted-Cholesky exhausted coordinates");
        used[(size_t) pivot] = 1;
        selected.push_back(pivot);

        if (!(best > floor)) continue;
        const double denom = std::sqrt(std::max(0.0, best));
        factor[(size_t) pivot * rank + k] = denom;
        for (int i = 0; i < dim; ++i) {
            if (used[(size_t) i]) continue;
            double a = 0.5 * (psd[(size_t) i * dim + pivot] + psd[(size_t) pivot * dim + i]);
            for (int j = 0; j < k; ++j) {
                a -= factor[(size_t) i * rank + j] * factor[(size_t) pivot * rank + j];
            }
            const double l = a / denom;
            if (!std::isfinite(l)) throw std::runtime_error("non-finite GDN pivoted-Cholesky factor");
            factor[(size_t) i * rank + k] = l;
            residual[(size_t) i] = std::max(0.0, residual[(size_t) i] - l * l);
        }
    }
    return selected;
}

std::vector<int> gdn_geometry_topk_diagonal_select(
        const std::vector<double> & psd,
        int dim,
        int rank) {
    if (dim <= 0 || rank <= 0 || rank > dim || psd.size() != (size_t) dim * dim) {
        throw std::runtime_error("GDN top-k diagonal shape mismatch");
    }
    std::vector<int> order((size_t) dim);
    for (int i = 0; i < dim; ++i) {
        const double d = psd[(size_t) i * dim + i];
        if (!std::isfinite(d)) throw std::runtime_error("non-finite GDN top-k diagonal");
        order[(size_t) i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const double da = psd[(size_t) a * dim + a];
        const double db = psd[(size_t) b * dim + b];
        if (da != db) return da > db;
        return a < b;
    });
    order.resize((size_t) rank);
    return order;
}

double gdn_geometry_jaccard(const std::vector<int> & a, const std::vector<int> & b) {
    std::unordered_set<int> sa(a.begin(), a.end());
    std::unordered_set<int> sb(b.begin(), b.end());
    if (sa.size() != a.size() || sb.size() != b.size()) {
        throw std::runtime_error("Jaccard input contains duplicate coordinates");
    }
    size_t inter = 0;
    for (int x : sa) inter += sb.count(x) != 0;
    const size_t uni = sa.size() + sb.size() - inter;
    return uni ? (double) inter / (double) uni : 1.0;
}

} // namespace qwen35_prune
