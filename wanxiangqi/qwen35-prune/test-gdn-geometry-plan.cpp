#include "gdn-geometry-plan.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace qwen35_prune;

int main() {
    {
        const int n = 4;
        std::vector<double> sum(n*n, 0.0);
        sum[0*n+0] = 4.0;
        sum[1*n+1] = 3.0;
        sum[2*n+2] = 2.0;
        sum[3*n+3] = 1.0;
        std::vector<double> prior = {4.0, 3.0, 2.0, 1.0};
        auto shrunk = gdn_geometry_shrink_second_moment(sum, 1, prior, n);
        assert(std::abs(shrunk.shrinkage - 0.8) < 1e-12);
        auto sel = gdn_geometry_pivoted_cholesky_select(shrunk.matrix, n, 2);
        assert((sel == std::vector<int>{0, 1}));
    }
    {
        // Coordinates 0 and 1 are almost redundant; coordinate 2 has less
        // marginal energy but adds a genuinely new direction.
        const int n = 3;
        std::vector<double> a = {
            4.0, 3.9, 0.0,
            3.9, 4.0, 0.0,
            0.0, 0.0, 3.0,
        };
        auto sel = gdn_geometry_pivoted_cholesky_select(a, n, 2);
        assert(sel[0] == 0);
        assert(sel[1] == 2);
        auto topk = gdn_geometry_topk_diagonal_select(a, n, 2);
        assert((topk == std::vector<int>{0, 1}));
    }
    {
        std::vector<double> a = {1.0, 0.0, 0.0, 1.0};
        auto sel = gdn_geometry_pivoted_cholesky_select(a, 2, 2);
        assert((sel == std::vector<int>{0, 1}));
        assert(std::abs(gdn_geometry_jaccard({0, 1}, {1, 2}) - 1.0/3.0) < 1e-12);
    }
    std::cout << "test-gdn-geometry-plan: OK\n";
    return 0;
}
